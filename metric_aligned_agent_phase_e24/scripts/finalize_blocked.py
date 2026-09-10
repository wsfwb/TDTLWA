#!/usr/bin/env python3
import csv,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];TDTL=Path('/mnt/05d0ac9f-c990-4b3c-8829-fd9b0049b406/shangdongfang/code/TDTL')
def write_json(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
api=json.loads((RUN/'metadata/reasoner_api_status.json').read_text())
events=api.get('events',[]); calls=1+len(events)
with (RUN/'results/reasoner_call_manifest.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['phase','attempt','model','reasoning_effort','status','error_class','latency_sec','tool_calls'])
 w.writerow(['initial_probe',1,'gpt-5.6-sol','xhigh','failure','generic_upstream_error',2.643831253051758,0])
 for e in events:w.writerow(['bounded_probe',e.get('attempt'),'gpt-5.6-sol','xhigh',e.get('status'),e.get('error_class',''),e.get('latency_sec'),0])
with (RUN/'results/transport_events.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['phase','attempt','status','error_class','latency_sec','error'])
 w.writerow(['initial_probe',1,'failure','generic_upstream_error',2.643831253051758,"HTTP 400 upstream_error"])
 for e in events:w.writerow(['bounded_probe',e.get('attempt'),e.get('status'),e.get('error_class',''),e.get('latency_sec'),e.get('error','')])
with (RUN/'results/input_manifest.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['interface','rows','unique_canonical','sha256'])
 for interface,name in [('I1','I1_long_context_test.jsonl'),('I2','I2_structured_evidence_test.jsonl'),('I3','I3_multimodal_summary_test.jsonl')]:
  p=RUN/'inputs'/name;ids=[json.loads(x)['canonical_id'] for x in p.read_text().splitlines() if x.strip()];w.writerow([interface,len(ids),len(set(ids)),hashlib.sha256(p.read_bytes()).hexdigest()])
write_json(RUN/'metadata/pipeline_state.json',{'status':'transport_blocked','current_stage':'direct_api_probe','canary_started':False,'session5_accessed':False,'I1_valid':0,'I2_valid':0,'I3_valid':0,'updated_at':time.time()})
write_json(RUN/'metadata/transport_state.json',{'status':'transport_blocked','actual_reasoner_calls':calls,'initial_probe_calls':1,'bounded_retry_probe_calls':len(events),'transport_failures':calls,'generic_upstream_failures':calls,'retries':max(0,len(events)-1),'parse_failures':0,'timeouts':0,'tool_calls':0,'last_error':api.get('error')})
write_json(RUN/'metadata/progress.json',{'current_stage':'blocked_before_canary','valid':0,'failed':calls,'remaining':0,'I1':0,'I2':0,'I3':0,'updated_at':time.time()})
write_json(RUN/'metadata/session5_access_status.json',{'session5_accessed':False,'session5_evaluation_count':0,'session5_candidate_evaluation_count':0,'session5_test_tuned':False,'clean_deployment_claim':False,'same_split_test_selected_posthoc_exploratory':True})
write_json(RUN/'metadata/INTEGRITY.json',{'status':'transport_blocked','reasoner_model':'gpt-5.6-sol','reasoning_effort':'xhigh','temperature':0.0,'wire_api':'responses','tool_calls':0,'codex_exec_used_for_e24_reasoner':False,'e23_cache_reused':False,'canary_started':False,'session5_accessed':False,'tdtl_modified':False,'upstream_artifacts_modified':False,'historical_artifacts_modified':False})
env={'python':sys.version,'platform':platform.platform(),'openai_sdk':__import__('openai').__version__,'endpoint_type':'direct Responses API','model':'gpt-5.6-sol','reasoning_effort':'xhigh','temperature':0.0,'PYTHONDONTWRITEBYTECODE':'1'}
(RUN/'metadata/ENVIRONMENT.txt').write_text(json.dumps(env,indent=2)+'\n')
headers=['interface','valid','expected','status'];rows=[['I1',0,1623,'not_started_transport_blocked'],['I2',0,1623,'not_started_transport_blocked'],['I3',0,1623,'not_started_transport_blocked']]
for name in ['I1_canonical_predictions.csv','I2_canonical_predictions.csv','I3_canonical_predictions.csv']:
 with (RUN/'results'/name).open('w',newline='') as f:csv.writer(f).writerow(['canonical_id','status','final_label','decision'])
for name,fields in [('candidate_leaderboard.csv',['candidate','weighted_f1','status']),('mechanism_summary.csv',['candidate','metric','value']),('bootstrap_summary.csv',['candidate','comparison','observed_delta','ci_lower','ci_upper']),('absolute_winner.csv',['candidate','weighted_f1','status']),('nontrivial_winner.csv',['candidate','weighted_f1','status']),('parse_failure_summary.csv',['parse_failures','timeouts','transport_failures'])]:
 with (RUN/'results'/name).open('w',newline='') as f:csv.writer(f).writerow(fields)
with (RUN/'results/parse_failure_summary.csv').open('a',newline='') as f:csv.writer(f).writerow([0,0,calls])
subprocess.run(['git','-C',str(TDTL),'rev-parse','HEAD'],stdout=(RUN/'metadata/tdtl_git_after.txt').open('w'),stderr=subprocess.DEVNULL)
subprocess.run(['git','-C',str(TDTL),'status','--short'],stdout=(RUN/'metadata/tdtl_status_after.txt').open('w'),stderr=subprocess.DEVNULL)
before=(RUN/'metadata/tdtl_git_before.txt').read_text() if (RUN/'metadata/tdtl_git_before.txt').exists() else ''
after=(RUN/'metadata/tdtl_git_after.txt').read_text() if (RUN/'metadata/tdtl_git_after.txt').exists() else ''
status_before=(RUN/'metadata/tdtl_status_before.txt').read_text() if (RUN/'metadata/tdtl_status_before.txt').exists() else ''
status_after=(RUN/'metadata/tdtl_status_after.txt').read_text() if (RUN/'metadata/tdtl_status_after.txt').exists() else ''
with (RUN/'metadata/tdtl_before_after_snapshot.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['check','before','after','match']);w.writerow(['git_head',before.strip(),after.strip(),before==after]);w.writerow(['git_status_sha256',hashlib.sha256(status_before.encode()).hexdigest(),hashlib.sha256(status_after.encode()).hexdigest(),status_before==status_after])
files=[p for p in RUN.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt']
with (RUN/'metadata/SHA256SUMS.txt').open('w') as f:
 for p in sorted(files):f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(RUN))+'\n')
print(json.dumps({'status':'transport_blocked','actual_reasoner_calls':calls,'I1':0,'I2':0,'I3':0,'session5_accessed':False,'tdtl_modified':not(before==after and status_before==status_after)}))
if __name__=='__main__':pass
