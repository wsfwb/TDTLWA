from __future__ import annotations
import csv,hashlib,json,os,time
from pathlib import Path
from .cache_store import atomic_write_json,cache_identity
from .direct_responses_client import DirectResponsesClient,extract_valid_output
from .retry_policy import is_retryable,backoff_seconds,classify_error,retry_after_from_exception
RUN=Path(__file__).resolve().parents[1];RUN_ID=RUN.name
ALLOW={
'I1':{'canonical_id','dialogue_id','speaker','turn_index','dialogue_length','current_utterance','preceding_turns','serialized_context','truncated_fixed_head_tail'},
'I2':{'canonical_id','full_prediction','residual_prediction','full_probability','residual_probability','full_confidence','residual_confidence','full_entropy','residual_entropy','full_margin','residual_margin','agreement','confidence_delta','entropy_delta','margin_delta','label_names'},
'I3':{'canonical_id','modality_summary_vector','modality_summary_dimensions','missing_modality_indicator','raw_audio_available','raw_video_available'}}
FORBIDDEN={'gold','session5_label','correctness','benefit','harm','neutral','test_wf1','test_metric','future_outcome','r1_prediction','r1_confidence','oracle'}
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def sanitize_row(interface,row):
    low={str(k).lower() for k in row}
    bad=low&FORBIDDEN
    if bad:raise ValueError('forbidden input fields: '+','.join(sorted(bad)))
    extra=set(row)-ALLOW[interface]
    if extra:raise ValueError('unapproved input fields: '+','.join(sorted(extra)))
    return {k:row[k] for k in row if k in ALLOW[interface]}
def build_prompt(interface,row,evidence=None,modality=None):
    rule=('Return exactly one JSON object matching the supplied schema and no prose. Use only the supplied pre-Reason information. '
          'The cheap prediction is a routing reference, not a known correct answer. Do not use external tools, files, APIs, labels, outcomes, or metrics. '
          'decision must be KEEP_CHEAP or ACCEPT_REASON.')
    body={'interface':interface,'context':sanitize_row('I1',row)}
    if interface in ('I2','I3'):body['structured_evidence']=sanitize_row('I2',evidence)
    if interface=='I3':body['modality_summary']=sanitize_row('I3',modality)
    return rule+'\n'+json.dumps(body,ensure_ascii=False,separators=(',',':'))
def _append(path,fields,row):
    path.parent.mkdir(parents=True,exist_ok=True);exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);(None if exists else w.writeheader());w.writerow({k:row.get(k,'') for k in fields})
FIELDS=['timestamp','interface','canonical_id','attempt','status','error_class','latency_sec','prompt_sha256','input_sha256','response_sha256','cache_identity']
def call_one(client,interface,cid,prompt,input_hash,schema,schema_hash,max_attempts=6):
    identity=cache_identity(RUN_ID,interface,cid,sha(prompt),input_hash,schema_hash);path=RUN/'cache'/interface/(cid+'.json')
    if path.exists():
        try:
            z=json.loads(path.read_text());
            if z.get('status')=='success' and z.get('cache_identity')==identity and z.get('run_id')==RUN_ID:return z,True
        except Exception:pass
    last=None
    for attempt in range(1,max_attempts+1):
        started=time.time()
        try:
            response=client.create(prompt,schema);out,raw=extract_valid_output(response);lat=time.time()-started
            rec={'run_id':RUN_ID,'status':'success','model':'gpt-5.6-sol','reasoning_effort':'xhigh','temperature':0.0,'interface':interface,'canonical_id':cid,'prompt_sha256':sha(prompt),'input_sha256':input_hash,'schema_sha256':schema_hash,'cache_identity':identity,'output':out,'response_id':raw.get('id'),'usage':raw.get('usage'),'tool_calls':0,'latency_sec':lat}
            atomic_write_json(path,rec);_append(RUN/'results/reasoner_call_manifest.csv',FIELDS,{'timestamp':time.time(),'interface':interface,'canonical_id':cid,'attempt':attempt,'status':'success','latency_sec':lat,'prompt_sha256':sha(prompt),'input_sha256':input_hash,'response_sha256':sha(json.dumps(raw,sort_keys=True,default=str)),'cache_identity':identity});return rec,False
        except Exception as e:
            last=e;lat=time.time()-started;kind=classify_error(e);_append(RUN/'results/transport_events.csv',FIELDS,{'timestamp':time.time(),'interface':interface,'canonical_id':cid,'attempt':attempt,'status':'failure','error_class':kind,'latency_sec':lat,'prompt_sha256':sha(prompt),'input_sha256':input_hash,'cache_identity':identity})
            if attempt>=max_attempts or not is_retryable(e):raise
            time.sleep(backoff_seconds(attempt,retry_after_from_exception(e),1.0))
    raise last
def load_rows(name):return [json.loads(x) for x in (RUN/'inputs'/name).read_text().splitlines() if x.strip()]
def stage_inputs(interface,canary=False):
    if canary:return [('I1',x,None,None) for x in load_rows('canary_I1_train.jsonl')]
    ctx={str(x['canonical_id']):x for x in load_rows('I1_long_context_test.jsonl')};ev={str(x['canonical_id']):x for x in load_rows('I2_structured_evidence_test.jsonl')};mo={str(x['canonical_id']):x for x in load_rows('I3_multimodal_summary_test.jsonl')}
    return [(interface,r,ev.get(cid),mo.get(cid)) for cid,r in ctx.items()]
def run_stage(stage,canary=False):
    schema_text=(RUN/'protocol/E24_OUTPUT_SCHEMA.json').read_text();schema=json.loads(schema_text);schema_hash=sha(schema_text)
    client=DirectResponsesClient();items=stage_inputs(stage,canary);valid=0;cache_hits=0;failures=0;consecutive_final_failures=0;latencies=[]
    for n,(prompt_interface,row,evidence,modality) in enumerate(items,1):
        cid=str(row['canonical_id']);prompt=build_prompt(prompt_interface,row,evidence,modality);input_hash=sha(json.dumps({'row':row,'evidence':evidence,'modality':modality},sort_keys=True,ensure_ascii=False))
        try:
            rec,hit=call_one(client,'CANARY' if canary else stage,cid,prompt,input_hash,schema,schema_hash);valid+=1;cache_hits+=int(hit);consecutive_final_failures=0;latencies.append(float(rec.get('latency_sec',0)))
        except Exception as e:
            failures+=1;consecutive_final_failures+=1
            if consecutive_final_failures>=3 or classify_error(e)=='nonretryable_error':
                atomic_write_json(RUN/'metadata/transport_state.json',{'status':'transport_blocked','stage':stage,'canonical_id':cid,'valid':valid,'failed':failures,'last_error_class':classify_error(e),'updated_at':time.time()});raise
        if n%25==0 or n==len(items):
            elapsed=sum(latencies);mean=(elapsed/len(latencies)) if latencies else None
            atomic_write_json(RUN/'metadata/progress.json',{'current_stage':'CANARY' if canary else stage,'valid':valid,'failed':failures,'missing':len(items)-valid,'remaining':len(items)-n,'cache_hits':cache_hits,'mean_latency':mean,'last_canonical_id':cid,'updated_at':time.time(),'supervisor_pid':os.getpid()})
        atomic_write_json(RUN/'metadata/heartbeat.json',{'pid':os.getpid(),'stage':'CANARY' if canary else stage,'valid':valid,'updated_at':time.time()})
    return {'stage':'CANARY' if canary else stage,'expected':len(items),'valid':valid,'failures':failures,'cache_hits':cache_hits,'passed':valid==len(items) and failures==0,'key_summary':client.key_summary,'base_url':client.base_url}
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['canary','I1','I2','I3']);a=ap.parse_args()
    result=run_stage('I1' if a.stage=='canary' else a.stage,a.stage=='canary');print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['passed'] else 2)
if __name__=='__main__':main()
