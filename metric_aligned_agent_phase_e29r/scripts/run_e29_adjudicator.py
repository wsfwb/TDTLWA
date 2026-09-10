import csv,json,os,time,hashlib,fcntl,sys
from pathlib import Path
from .direct_responses_client import DirectResponsesClient,TransportFailure,ToolCallResponse,SchemaFailure

RUN=Path(__file__).resolve().parents[1]; SCHEMA_PATH=RUN/'protocol/E29_OUTPUT_SCHEMA.json'; TARGETS=RUN/'targets/target_pool.csv'; PROMPTS=RUN/'targets/adjudicator_prompts.jsonl'; MAX_SESSION_CALLS=200; INTERVAL=20.0
def sh(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)); os.replace(tmp,path)
def append_csv(path,row,fields):
    path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); 
        if not exists: w.writeheader()
        w.writerow({k:row.get(k,'') for k in fields})
def train_prompt(d):
    return 'Classify this emotion utterance using only the text and context. Return the required JSON object. Evidence: '+json.dumps({k:d.get(k) for k in ('current_utterance','preceding_turns','speaker','turn_index','serialized_context')},ensure_ascii=False,separators=(',',':'))
def valid_cache(p):
    try:
        d=json.loads(p.read_text()); o=d.get('output',{}); return d.get('status')=='success' and d.get('model')=='gpt-5.6-sol' and d.get('reasoning_effort')=='xhigh' and set(o)=={'final_label','decision','preferred_source','confidence','uncertainty','evidence_consistency','short_reason'}
    except Exception: return False
def run_small(kind):
    import json as J
    train=[]
    for line in (RUN.parents[2]/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1/inputs/canary_I1_train.jsonl').read_text().splitlines():
        if line.strip(): train.append(J.loads(line))
    n=3 if kind=='PROBE' else 20; schema=J.loads(SCHEMA_PATH.read_text()); client=DirectResponsesClient(); rows=[]; failures=0; successes=0; last=0.0
    for index,d in enumerate(train[:n], 1):
        wait=max(0,INTERVAL-(time.time()-last)); time.sleep(wait); last=time.time()
        write_json(RUN/f'metadata/E29_{kind}_PROGRESS.json',{'kind':kind,'state':'before_request','request_in_flight':True,'canonical_id':d['canonical_id'],'attempt':index,'successes':successes,'failures':failures,'updated_at':time.time()})
        try:
            out,env,payload=client.create(train_prompt(d),schema); successes+=1; rows.append({'canonical_id':d['canonical_id'],'status':'success','output_item_types':env.get('output_item_types'),'usage':env.get('usage'),'response_sha256':env.get('response_sha256')})
        except Exception as e:
            failures+=1
            envelope=getattr(e,'envelope',{})
            diagnostic={'canonical_id':d['canonical_id'],'attempt':index,'status':type(e).__name__,'error':str(e)[:500],'envelope':envelope,'raw_text_sha256':hashlib.sha256(str(getattr(e,'raw_text','')).encode()).hexdigest()}
            write_json(RUN/'results/response_envelopes'/kind/(d['canonical_id']+f'_attempt{index}.json'),diagnostic)
            rows.append({'canonical_id':d['canonical_id'],'status':type(e).__name__,'error':str(e)[:500],'envelope':envelope})
        write_json(RUN/f'metadata/E29_{kind}_PROGRESS.json',{'kind':kind,'state':'after_request','request_in_flight':False,'canonical_id':d['canonical_id'],'attempt':index,'successes':successes,'failures':failures,'updated_at':time.time()})
    write_json(RUN/f'metadata/E29_{kind}_STATUS.json',{'kind':kind,'successes':successes,'failures':failures,'valid_json':successes,'tool_calls':0,'rows':rows,'schema_sha256':sh(SCHEMA_PATH),'model':'gpt-5.6-sol','reasoning_effort':'xhigh'})
    if (kind=='PROBE' and successes!=3) or (kind=='CANARY' and (successes<19 or failures>1)):
        raise SystemExit(f'{kind} gate failed: {successes}/{n}')
def run_session():
    lock=(RUN/'metadata/e29_adjudicator.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    schema=json.loads(SCHEMA_PATH.read_text()); schema_hash=sh(SCHEMA_PATH); prompt_rows=[json.loads(x) for x in PROMPTS.read_text().splitlines() if x.strip()]
    cache=RUN/'caches/ADJUDICATOR'; cache.mkdir(parents=True,exist_ok=True); manifest=RUN/'results/reasoner_call_manifest.csv'; events=RUN/'results/transport_events.csv'; fields=['timestamp','canonical_id','attempt','status','error','latency_sec','response_sha256','usage','invocation']
    client=DirectResponsesClient(); last=0.; invocations=0; successes=0; retries=0; failures=0; tool_calls=0
    for pr in prompt_rows:
        cid=pr['canonical_id']; cp=cache/(cid+'.json')
        if valid_cache(cp): continue
        done=False
        for attempt in range(1,3):
            if invocations>=MAX_SESSION_CALLS: break
            wait=max(0,INTERVAL-(time.time()-last)); time.sleep(wait); last=time.time(); invocations+=1; started=time.time()
            write_json(RUN/'metadata/heartbeat.json',{'status':'running','stage':'SESSION5','canonical_id':cid,'attempt':attempt,'cache_hits':len(list(cache.glob('*.json'))),'invocations':invocations,'updated_at':time.time()})
            try:
                out,env,payload=client.create(pr['prompt'],schema); latency=time.time()-started
                rec={'status':'success','run_id':RUN.name,'canonical_id':cid,'model':'gpt-5.6-sol','reasoning_effort':'xhigh','schema_sha256':schema_hash,'prompt_sha256':pr['prompt_sha256'],'input_sha256':pr['input_sha256'],'cache_identity':hashlib.sha256((cid+pr['prompt_sha256']+schema_hash).encode()).hexdigest(),'output':out,'envelope':env,'tools':[]}
                tmp=cp.with_suffix('.tmp'); tmp.write_text(json.dumps(rec,ensure_ascii=False)); os.replace(tmp,cp); successes+=1; done=True
                append_csv(manifest,{'timestamp':time.time(),'canonical_id':cid,'attempt':attempt,'status':'success','latency_sec':latency,'response_sha256':env.get('response_sha256'),'usage':json.dumps(env.get('usage')),'invocation':invocations},fields); break
            except ToolCallResponse as e:
                tool_calls+=1; failures+=1; append_csv(manifest,{'timestamp':time.time(),'canonical_id':cid,'attempt':attempt,'status':'tool_call','error':str(e),'latency_sec':time.time()-started,'invocation':invocations},fields); break
            except Exception as e:
                failures+=1; retries+=int(attempt<2); err=str(e)[:800]; append_csv(events,{'timestamp':time.time(),'canonical_id':cid,'attempt':attempt,'status':type(e).__name__,'error':err,'latency_sec':time.time()-started,'invocation':invocations},fields)
                if attempt<2: time.sleep(5)
        if not done and invocations>=MAX_SESSION_CALLS: break
    write_json(RUN/'metadata/transport_state.json',{'session5_invocations':invocations,'successful_calls':successes,'retries':retries,'failures':failures,'tool_calls':tool_calls,'cache_count':len(list(cache.glob('*.json'))),'updated_at':time.time(),'max_session5_invocations':MAX_SESSION_CALLS})
    write_json(RUN/'metadata/heartbeat.json',{'status':'complete' if successes==200 else 'partial','stage':'SESSION5','valid':successes,'remaining':200-successes,'invocations':invocations,'updated_at':time.time()})
    print(json.dumps({'successes':successes,'invocations':invocations,'failures':failures,'retries':retries}))
def main():
    kind=sys.argv[1].upper() if len(sys.argv)>1 else 'PROBE'
    if kind in ('PROBE','CANARY'): run_small(kind)
    elif kind in ('SESSION5','TARGETS'): run_session()
    else: raise SystemExit('usage PROBE|CANARY|SESSION5')
if __name__=='__main__': main()
