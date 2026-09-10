#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, fcntl, hashlib, json, os, shutil, threading, time
from pathlib import Path
from .cache_store import atomic_write_json, cache_identity
from .direct_responses_client import DirectResponsesClient, EmptyMessageResponse, SchemaFailure, ToolCallResponse, TransportFailure
from .input_contract import prompt_view

ROOT=Path(__file__).resolve().parents[1]; RUN_ID=ROOT.name
E24=Path('/mnt/05d0ac9f-c990-4b3c-8829-fd9b0049b406/shangdongfang/code/ClarifyMER/experiment_outputs/metric_aligned_agent_phase_e24/run_20260819T000000_CST_direct_responses_gpt56sol_xhigh_v1')
FIELDS=['timestamp','interface','canonical_id','attempt','status','http_status','error_class','latency_sec','prompt_sha256','input_sha256','schema_sha256','response_sha256','input_tokens','output_tokens','reasoning_tokens','cache_identity']
BACKOFF=[15,30,60,120,300]; TRANSIENT={429,500,502,503,504}; SCHEMA_MAX_ATTEMPTS=3; EMPTY_MAX_ATTEMPTS=2

class StageBlocked(RuntimeError): pass

def sha(value):
    if not isinstance(value,str): value=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(value.encode()).hexdigest()

def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        if not exists: writer.writeheader()
        writer.writerow({key:row.get(key,'') for key in FIELDS}); handle.flush(); os.fsync(handle.fileno())

def state():
    path=ROOT/'metadata/pipeline_state.json'
    return json.loads(path.read_text()) if path.exists() else {'probe':{'passed':False},'canary':{'passed':False},'I1':{},'I2':{},'I3':{}}
def save_state(value): atomic_write_json(ROOT/'metadata/pipeline_state.json',value)

def heartbeat(stage,valid,failed,last_id,**extra):
    payload={'status':extra.pop('status','running'),'request_in_flight':extra.pop('request_in_flight',False),'stage':stage,'canonical_id':last_id,'attempt':extra.pop('attempt',None),'valid':valid,'failed':failed,'updated_at':time.time(),'pid':os.getpid()}
    payload.update(extra); atomic_write_json(ROOT/'metadata/heartbeat.json',payload)

TRANSPORT_DEFAULT={'total_invocations':0,'successful_calls':0,'cache_hits':0,'transport_retries':0,'schema_retries':0,'empty_response_retries':0,'429_count':0,'500_count':0,'502_count':0,'503_count':0,'504_count':0,'timeouts':0,'tool_calls':0,'unresolved_count':0,'request_in_flight':False,'last_request_time':None,'last_http_status':None,'last_latency_sec':None,'cumulative_usage':{'input_tokens':0,'output_tokens':0,'reasoning_tokens':0}}
def transport_state():
    path=ROOT/'metadata/transport_state.json'
    if path.exists():
        try:
            old=json.loads(path.read_text()); return {**TRANSPORT_DEFAULT,**old, 'cumulative_usage':{**TRANSPORT_DEFAULT['cumulative_usage'],**old.get('cumulative_usage',{})}}
        except Exception: pass
    return dict(TRANSPORT_DEFAULT)
def update_transport(**changes):
    value=transport_state()
    for key,change in changes.items():
        if key=='cumulative_usage':
            for sub,val in change.items(): value['cumulative_usage'][sub]=value['cumulative_usage'].get(sub,0)+(val if isinstance(val,(int,float)) else 0)
        elif key.endswith('_count') or key in {'total_invocations','successful_calls','cache_hits','transport_retries','schema_retries','empty_response_retries','timeouts','tool_calls','unresolved_count'}:
            value[key]=value.get(key,0)+(change if isinstance(change,(int,float)) else 0)
        else: value[key]=change
    value['updated_at']=time.time(); atomic_write_json(ROOT/'metadata/transport_state.json',value); return value

class RequestRateLimiter:
    def __init__(self,stage,valid,failed):
        self.stage=stage; self.valid=valid; self.failed=failed; self.last_started=None
        manifest=ROOT/'results/reasoner_call_manifest.csv'
        if manifest.exists():
            try:
                with manifest.open() as h: rows=list(csv.DictReader(h))
                if rows: self.last_started=float(rows[-1].get('timestamp') or 0) or None
            except Exception: pass
    def wait(self,canonical_id,attempt):
        while self.last_started is not None:
            remaining=max(0.0,20.0-(time.time()-self.last_started))
            if not remaining: break
            heartbeat(self.stage,self.valid,self.failed,canonical_id,attempt=attempt,status='rate_wait',request_in_flight=False)
            time.sleep(min(remaining,5))
        request_started=time.time()
        self.last_started=request_started
        update_transport(total_invocations=1,request_in_flight=True,last_request_time=request_started)
        return request_started

def seconds_until_next_start(last_started, now=None):
    return max(0.0,20.0-((time.time() if now is None else now)-last_started))

def pace(last_started, stage='I1', valid=0, failed=0, last_id=None):
    remaining=seconds_until_next_start(last_started)
    while remaining>0:
        heartbeat(stage,valid,failed,last_id,status='rate_wait',request_in_flight=False)
        time.sleep(min(remaining,5)); remaining=seconds_until_next_start(last_started)

def _inflight_heartbeat(stop,stage,valid,failed,cid,attempt):
    while not stop.wait(30):
        heartbeat(stage,valid,failed,cid,attempt=attempt,status='running',request_in_flight=True)

def prompt(interface,context,evidence=None,modality=None):
    body=prompt_view(interface,context,evidence,modality)
    rule=('Return exactly one JSON object matching the supplied schema and no prose. Use only supplied pre-Reason information. The cheap prediction is a routing reference, not a known-correct answer. Do not use tools, files, APIs, labels, outcomes, or metrics. decision must be KEEP_CHEAP or ACCEPT_REASON.')
    return rule+'\n'+json.dumps(body,ensure_ascii=False,separators=(',',':'))
def load_schema():
    text=(ROOT/'protocol/E26_OUTPUT_SCHEMA.json').read_text(); return json.loads(text),sha(text)
def rows_jsonl(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def stage_rows(stage):
    if stage in {'PROBE','CANARY'}:
        data=rows_jsonl(ROOT/'inputs/canary_I1_train.jsonl'); data=data[:3] if stage=='PROBE' else data
        return [(row,None,None) for row in data]
    ctx=rows_jsonl(ROOT/'inputs/I1_long_context_test.jsonl'); ev={str(x['canonical_id']):x for x in rows_jsonl(ROOT/'inputs/I2_structured_evidence_test.jsonl')}; mo={str(x['canonical_id']):x for x in rows_jsonl(ROOT/'inputs/I3_multimodal_summary_test.jsonl')}
    triples=[]
    for row in ctx:
        cid=str(row['canonical_id']); triples.append((row,ev.get(cid),mo.get(cid)))
    if len(triples)!=1623 or len({x[0]['canonical_id'] for x in triples})!=1623: raise StageBlocked('Session-5 canonical alignment failed')
    if stage in {'I2','I3'} and any(e is None for _,e,_ in triples): raise StageBlocked('missing structured evidence')
    if stage=='I3' and any(m is None for _,_,m in triples): raise StageBlocked('missing modality summary')
    return triples
def ensure_session5_inputs():
    st=state()
    if not (st.get('probe',{}).get('passed') and st.get('canary',{}).get('passed')): raise StageBlocked('Session-5 locked until probe and canary pass')
    for name in ('I1_long_context_test.jsonl','I2_structured_evidence_test.jsonl','I3_multimodal_summary_test.jsonl'):
        source=E24/'inputs'/name; target=ROOT/'inputs'/name
        if not target.exists(): shutil.copyfile(source,target)
    atomic_write_json(ROOT/'metadata/session5_access_status.json',{'session5_input_read':True,'session5_label_read':False,'session5_predictions_read':False,'session5_metrics':False,'phase':'interfaces'})
def retryable(exc):
    return isinstance(exc,TransportFailure) and (exc.status in TRANSIENT or exc.status is None or any(x in exc.detail.lower() for x in ('upstream_error','connection','timeout','reset','aborted')))
def usage_values(usage):
    usage=usage or {}; details=usage.get('output_tokens_details') or {}; return usage.get('input_tokens',''),usage.get('output_tokens',''),details.get('reasoning_tokens','')
def add_usage(total,usage):
    i,o,r=usage_values(usage)
    for key,val in [('input_tokens',i),('output_tokens',o),('reasoning_tokens',r)]:
        if isinstance(val,(int,float)): total[key]=total.get(key,0)+val
    return total
def save_envelope(stage,cid,attempt,envelope,error,raw_text=''):
    payload={'stage':stage,'canonical_id':cid,'attempt':attempt,'error':str(error),'raw_text':getattr(error,'raw_text',raw_text),'envelope':envelope,'saved_at':time.time()}
    atomic_write_json(ROOT/'results/response_envelopes'/stage/(cid+f'_attempt{attempt}.json'),payload)
def quarantine(stage,cid,reason,attempts):
    update_transport(unresolved_count=1,request_in_flight=False)
    atomic_write_json(ROOT/'results/unresolved'/stage/(cid+'.json'),{'stage':stage,'canonical_id':cid,'status':'quarantined','reason':reason,'attempts':attempts,'updated_at':time.time()})
    return None,False

def call_one(client,stage,row,evidence,modality,schema,schema_hash,rate_limiter,valid=0,failed=0):
    cid=str(row['canonical_id']); text=prompt('I1' if stage in {'PROBE','CANARY','I1'} else stage,row,evidence,modality); prompt_hash=sha(text); input_hash=sha({'context':row,'evidence':evidence,'modality':modality}); identity=cache_identity(RUN_ID,stage,cid,prompt_hash,schema_hash,input_hash); path=ROOT/'caches'/stage/(cid+'.json')
    if path.exists():
        try:
            cached=json.loads(path.read_text())
            if cached.get('status')=='success' and cached.get('cache_identity')==identity and cached.get('run_id')==RUN_ID:
                update_transport(cache_hits=1); return cached,True
        except Exception: pass
    schema_failures=empty_failures=transport_failures=0; final=None
    for attempt in range(1,7):
        rate_limiter.valid=valid; rate_limiter.failed=failed; request_started = rate_limiter.wait(cid,attempt)
        if not isinstance(request_started,(int,float)):
            request_started=time.time(); rate_limiter_start_missing=True
        else: rate_limiter_start_missing=False
        assert isinstance(request_started,(int,float))
        stop=threading.Event(); watcher=threading.Thread(target=_inflight_heartbeat,args=(stop,stage,valid,failed,cid,attempt),daemon=True); watcher.start()
        try:
            output,envelope,latency,payload=client.create(text,schema); stop.set(); watcher.join(timeout=1)
            latency_sec = max(0.0, time.time() - request_started)
            i,o,r=usage_values(envelope.get('usage')); update_transport(successful_calls=1,request_in_flight=False,last_http_status=envelope.get('http_status'),last_latency_sec=latency_sec,cumulative_usage={'input_tokens':i,'output_tokens':o,'reasoning_tokens':r})
            rec={'status':'success','run_id':RUN_ID,'model':'gpt-5.6-sol','reasoning_effort':'xhigh','temperature':None,'store':None,'tools':None,'interface':stage,'canonical_id':cid,'prompt_sha256':prompt_hash,'input_sha256':input_hash,'schema_sha256':schema_hash,'cache_identity':identity,'output':output,'envelope':envelope,'latency_sec':latency_sec,'request_started_at':request_started,'usage':envelope.get('usage')}
            atomic_write_json(path,rec); append(ROOT/'results/reasoner_call_manifest.csv',{'timestamp':time.time(),'interface':stage,'canonical_id':cid,'attempt':attempt,'status':'success','http_status':envelope.get('http_status'),'latency_sec':latency_sec,'prompt_sha256':prompt_hash,'input_sha256':input_hash,'schema_sha256':schema_hash,'response_sha256':envelope.get('response_sha256'),'input_tokens':i,'output_tokens':o,'reasoning_tokens':r,'cache_identity':identity}); return rec,False
        except ToolCallResponse as exc:
            stop.set(); watcher.join(timeout=1); latency_sec=max(0.0,time.time()-request_started); update_transport(tool_calls=1,request_in_flight=False,last_latency_sec=latency_sec); append(ROOT/'results/transport_events.csv',{'timestamp':time.time(),'interface':stage,'canonical_id':cid,'attempt':attempt,'status':'stop','error_class':'ToolCallResponse','latency_sec':latency_sec,'rate_limiter_start_missing':rate_limiter_start_missing,'prompt_sha256':prompt_hash,'input_sha256':input_hash,'schema_sha256':schema_hash,'cache_identity':identity}); raise StageBlocked(str(exc)) from exc
        except SchemaFailure as exc:
            stop.set(); watcher.join(timeout=1); schema_failures+=1; update_transport(schema_retries=1,request_in_flight=False,last_http_status=exc.http_status); save_envelope(stage,cid,attempt,exc.envelope,exc)
            latency_sec = max(0.0, time.time() - request_started)
            append(ROOT/'results/transport_events.csv',{'timestamp':time.time(),'interface':stage,'canonical_id':cid,'attempt':attempt,'status':'failure','http_status':exc.http_status,'error_class':'schema_retry','latency_sec':latency_sec,'rate_limiter_start_missing':rate_limiter_start_missing,'prompt_sha256':prompt_hash,'input_sha256':input_hash,'schema_sha256':schema_hash,'response_sha256':exc.envelope.get('response_sha256'),'cache_identity':identity})
            if schema_failures>=SCHEMA_MAX_ATTEMPTS: return quarantine(stage,cid,'schema_failure_exhausted',schema_failures)
            continue
        except EmptyMessageResponse as exc:
            stop.set(); watcher.join(timeout=1); empty_failures+=1; latency_sec = max(0.0, time.time() - request_started); update_transport(empty_response_retries=1,request_in_flight=False,last_http_status=exc.envelope.get('http_status'),last_latency_sec=latency_sec); save_envelope(stage,cid,attempt,exc.envelope,exc,exc.raw_text)
            append(ROOT/'results/transport_events.csv',{'timestamp':time.time(),'interface':stage,'canonical_id':cid,'attempt':attempt,'status':'failure','http_status':exc.envelope.get('http_status'),'error_class':'empty_response_retry','latency_sec':latency_sec,'rate_limiter_start_missing':rate_limiter_start_missing,'prompt_sha256':prompt_hash,'input_sha256':input_hash,'schema_sha256':schema_hash,'response_sha256':exc.envelope.get('response_sha256'),'cache_identity':identity})
            if empty_failures>=EMPTY_MAX_ATTEMPTS: return quarantine(stage,cid,'empty_response_circuit_breaker',empty_failures)
            continue
        except TransportFailure as exc:
            stop.set(); watcher.join(timeout=1); final=exc; transport_failures+=1; kind='transport_retryable' if retryable(exc) else 'transport_nonretryable'; latency_sec = max(0.0, time.time() - request_started); update_transport(transport_retries=1 if retryable(exc) else 0,request_in_flight=False,last_http_status=exc.status,last_latency_sec=latency_sec,**({f'{exc.status}_count':1} if exc.status in TRANSIENT else {})); append(ROOT/'results/transport_events.csv',{'timestamp':time.time(),'interface':stage,'canonical_id':cid,'attempt':attempt,'status':'failure','http_status':exc.status,'error_class':kind,'latency_sec':latency_sec,'rate_limiter_start_missing':rate_limiter_start_missing,'prompt_sha256':prompt_hash,'input_sha256':input_hash,'schema_sha256':schema_hash,'cache_identity':identity})
            if not retryable(exc) or attempt==6: return quarantine(stage,cid,'transport_failure_exhausted',transport_failures)
            continue
    return quarantine(stage,cid,str(final or 'unknown_failure'),max(schema_failures,empty_failures,transport_failures))

def run(stage):
    lock=(ROOT/'metadata/supervisor.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise StageBlocked('active E26 supervisor lock')
    try:
        st=state();
        if stage in {'I1','I2','I3'}: ensure_session5_inputs()
        if stage=='I2' and not st.get('I1',{}).get('passed'): raise StageBlocked('I1 not complete')
        if stage=='I3' and not st.get('I2',{}).get('passed'): raise StageBlocked('I2 not complete')
        schema,schema_hash=load_schema(); data=stage_rows(stage); client=DirectResponsesClient(); valid=failed=hits=0; unresolved=[]; usage={'input_tokens':0,'output_tokens':0,'reasoning_tokens':0}; started=time.time(); limiter=RequestRateLimiter(stage,valid,failed)
        for index,(row,evidence,modality) in enumerate(data,1):
            cid=str(row['canonical_id'])
            try:
                rec,hit=call_one(client,stage,row,evidence,modality,schema,schema_hash,limiter,valid,failed)
                if rec is None: failed+=1; unresolved.append(cid)
                else:
                    valid+=1; hits+=int(hit); add_usage(usage,rec.get('usage') or {})
            except StageBlocked as exc:
                failed+=1; heartbeat(stage,valid,failed,cid,status='stopped',request_in_flight=False,error=str(exc)); st[stage]={'passed':False,'valid':valid,'failed':failed,'expected':len(data),'error':str(exc),'updated_at':time.time()}; save_state(st); raise
            heartbeat(stage,valid,failed,cid,request_in_flight=False)
            if index%25==0 or index==len(data): atomic_write_json(ROOT/'metadata/progress.json',{'current_stage':stage,'valid':valid,'failed':failed,'expected':len(data),'remaining':len(data)-index,'cache_hits':hits,'total_calls':index-hits,'usage':usage,'mean_latency_sec':(time.time()-started)/max(1,valid-hits),'last_canonical_id':cid,'updated_at':time.time(),'supervisor_pid':os.getpid()})
            if index%100==0: atomic_write_json(ROOT/'metadata'/f'usage_{stage}_{index}.json',usage)
        # Bounded recovery pass: each unresolved item gets the same finite schema/transport policy once more.
        for cid in list(unresolved):
            item=next((x for x in data if str(x[0]['canonical_id'])==cid),None)
            if item is None: continue
            rec,hit=call_one(client,stage,item[0],item[1],item[2],schema,schema_hash,limiter,valid,failed)
            if rec is not None:
                valid+=1; failed-=1; unresolved.remove(cid); hits+=int(hit); add_usage(usage,rec.get('usage') or {})
        passed=(valid==len(data) and not unresolved and failed==0)
        st[stage]={'passed':passed,'valid':valid,'expected':len(data),'failed':len(unresolved),'unresolved':unresolved,'usage':usage,'updated_at':time.time()}; save_state(st); heartbeat(stage,valid,len(unresolved),unresolved[-1] if unresolved else (str(data[-1][0]['canonical_id']) if data else None),status='complete' if passed else 'partial_unresolved',request_in_flight=False)
        atomic_write_json(ROOT/'metadata/progress.json',{'current_stage':stage,'valid':valid,'failed':len(unresolved),'expected':len(data),'remaining':len(data)-valid,'cache_hits':hits,'total_calls':transport_state().get('total_invocations',0),'usage':usage,'last_canonical_id':unresolved[-1] if unresolved else str(data[-1][0]['canonical_id']),'updated_at':time.time(),'supervisor_pid':os.getpid()})
        print(json.dumps({'stage':stage,'passed':passed,'valid':valid,'failed':len(unresolved),'cache_hits':hits,'unresolved':unresolved,'usage':usage},indent=2)); return 0 if passed else 2
    finally: fcntl.flock(lock,fcntl.LOCK_UN); lock.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['PROBE','CANARY','I1','I2','I3']); args=ap.parse_args(); raise SystemExit(run(args.stage))
if __name__=='__main__': main()
