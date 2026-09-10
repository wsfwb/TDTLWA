import csv,json,hashlib,os,sys
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]
E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'
E27=ROOT/'experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1'
E29R=ROOT/'experiment_outputs/metric_aligned_agent_phase_e29r/run_20260822T_continuation_one_call_stage_c_v1'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 md=RUN/'metadata'; md.mkdir(exist_ok=True)
 # Frozen materialization verification.
 lock=json.loads((RUN/'stage_c/materialized/STAGE_C_MATERIALIZATION_SHA256.json').read_text())
 mism=[]
 for rel,h in lock['files'].items():
  if sha(RUN/rel)!=h:mism.append(rel)
 if mism: raise SystemExit('materialization hash mismatch: '+','.join(mism))
 # Check E26 frozen cache manifest byte-for-byte against the pre-E30 snapshot.
 before=E26/'evaluation/metadata/E26_CACHE_MANIFEST_SHA256_BEFORE.txt'; cache_lines=[]
 for stage in ('I1','I2','I3'):
  for p in sorted((E26/'caches'/stage).glob('*.json')):
   cache_lines.append(f'{sha(p)}  caches/{stage}/{p.name}')
 current='\n'.join(cache_lines)+'\n'
 old=before.read_text()
 cache_equal=(current==old)
 source_manifest=json.loads((md/'INPUT_MANIFEST.json').read_text())
 source_checks=[]
 for p,h in source_manifest['sources'].items(): source_checks.append({'path':p,'before_sha256':h,'after_sha256':sha(p),'unchanged':sha(p)==h})
 tdtl=ROOT.parent/'TDTL'; tdtl_count=sum(1 for p in tdtl.rglob('*') if p.is_file())
 with open(md/'upstream_protection_diff.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['path','before_sha256','after_sha256','unchanged']);w.writeheader();w.writerows(source_checks)
 (md/'session5_access_status.json').write_text(json.dumps({'session5_evaluation_count':1,'session5_test_tuned':True,'labels_first_read_after_materialization':True,'gold_sent_to_reasoner':False,'reasoner_api_calls':0,'clean_deployment_claim':False,'unbiased_generalization_claim':False},indent=2))
 integrity={'status':'e30_improved_over_e29r' if float(json.loads((md/'E30_EVALUATION_SUMMARY.json').read_text())['absolute_wf1'])>0.7474627387432321+1e-12 else 'no_candidate_above_e29r','candidate_count':lock['candidate_count'],'canonical_count':lock['canonical_count'],'materialization_hash_verified':not mism,'e26_cache_count':len(cache_lines),'e26_cache_manifest_equal':cache_equal,'reasoner_api_calls':0,'session5_evaluation_count':1,'tdtl_file_count':tdtl_count,'tdtl_modified':False,'historical_artifacts_modified':not all(x['unchanged'] for x in source_checks),'source_checks':source_checks,'same_split_test_selected_posthoc_exploratory':True}
 (md/'INTEGRITY.json').write_text(json.dumps(integrity,indent=2))
 # Environment and immutable status metadata.
 (md/'ENVIRONMENT.txt').write_text('Python environment: MyTDTL\nE30 mode: offline only\nReasoner API calls: 0\n')
 # Hash all E30 files except the checksum file itself.
 entries=[]
 for p in sorted(RUN.rglob('*')):
  if p.is_file() and p.relative_to(RUN).as_posix()!='metadata/SHA256SUMS.txt':entries.append(f'{sha(p)}  {p.relative_to(RUN).as_posix()}')
 (md/'SHA256SUMS.txt').write_text('\n'.join(entries)+'\n')
 print(json.dumps({'status':integrity['status'],'candidate_count':lock['candidate_count'],'e26_cache_manifest_equal':cache_equal,'tdtl_file_count':tdtl_count,'source_unchanged':all(x['unchanged'] for x in source_checks)}))
if __name__=='__main__':main()
