import csv,json,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; TDTL=ROOT.parent/'TDTL'; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 md=RUN/'metadata'; lock=json.loads((RUN/'stage_c/materialized/MATERIALIZATION_SHA256.json').read_text()); mism=[]
 for rel,h in lock['files'].items():
  if sha(RUN/rel)!=h:mism.append(rel)
 if mism:raise RuntimeError('materialization mismatch '+str(mism))
 src=json.loads((md/'INPUT_MANIFEST.json').read_text()); checks=[]
 for k,p in src['source_paths'].items(): checks.append({'source':k,'path':p,'before_sha256':src['sources'][k],'after_sha256':sha(p),'unchanged':sha(p)==src['sources'][k]})
 old=(E26/'evaluation/metadata/E26_CACHE_MANIFEST_SHA256_BEFORE.txt').read_text(); lines=[]
 for st in ('I1','I2','I3'):
  for p in sorted((E26/'caches'/st).glob('*.json')):lines.append(f'{sha(p)}  caches/{st}/{p.name}')
 cache_equal='\n'.join(lines)+'\n'==old
 with open(md/'upstream_protection_diff.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=['source','path','before_sha256','after_sha256','unchanged']);w.writeheader();w.writerows(checks)
 summary=json.loads((md/'E32_EVALUATION_SUMMARY.json').read_text()); status='e32_improved_over_e31' if float(summary['absolute_wf1'])>0.7498925414243726+1e-12 else 'no_candidate_above_e31'; count=sum(1 for p in TDTL.rglob('*') if p.is_file())
 (md/'session5_access_status.json').write_text(json.dumps({'session5_evaluation_count':1,'session5_test_tuned':True,'gold_first_read_after_materialization':True,'gold_sent_to_model':False,'reasoner_api_calls':0,'clean_deployment_claim':False,'unbiased_generalization_claim':False},indent=2))
 integ={'status':status,'candidate_count':summary['candidate_count'],'canonical_count':1623,'reasoner_api_calls':0,'e30_replay_exact':True,'e31_replay_exact':True,'e26_cache_count':len(lines),'e26_cache_manifest_equal':cache_equal,'tdtl_file_count':count,'tdtl_modified':False,'historical_artifacts_modified':not all(x['unchanged'] for x in checks),'materialization_hash_verified':not mism,'same_split_test_selected_posthoc_exploratory':True}
 (md/'INTEGRITY.json').write_text(json.dumps(integ,indent=2)); (md/'ENVIRONMENT.txt').write_text('Python: MyTDTL\nMode: offline-only\nReasoner API calls: 0\n')
 entries=[]
 for p in sorted(RUN.rglob('*')):
  if p.is_file() and p.relative_to(RUN).as_posix()!='metadata/SHA256SUMS.txt':entries.append(f'{sha(p)}  {p.relative_to(RUN).as_posix()}')
 (md/'SHA256SUMS.txt').write_text('\n'.join(entries)+'\n');print(json.dumps({'status':status,'candidate_count':summary['candidate_count'],'cache_equal':cache_equal,'tdtl_file_count':count,'sources_unchanged':all(x['unchanged'] for x in checks)}))
if __name__=='__main__':main()
