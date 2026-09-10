import csv,json,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; E27=ROOT/'experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1'; E29R=ROOT/'experiment_outputs/metric_aligned_agent_phase_e29r/run_20260822T_continuation_one_call_stage_c_v1'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 src=list(csv.DictReader(open(E26/'evaluation/results/e26_canonical_predictions.csv'))); target=list(csv.DictReader(open(E29R/'targets/target_pool.csv'))); cache=list((E29R/'caches/ADJUDICATOR').glob('*.json'))
 if len(src)!=1623 or len({r['canonical_id'] for r in src})!=1623 or len(target)!=200 or len(cache)!=200: raise RuntimeError('alignment/count failure')
 files=[E26/'evaluation/results/e26_canonical_predictions.csv',E27/'materialized/candidate_predictions.npz',E29R/'stage_c/materialized/candidate_predictions.npz']
 rec={'canonical_count':1623,'target_pool_count':200,'adjudicator_cache_count':200,'reasoner_api_calls':0,'sources':{str(p):sha(p) for p in files}}
 (RUN/'metadata/INPUT_MANIFEST.json').write_text(json.dumps(rec,indent=2)); print(json.dumps(rec))
if __name__=='__main__':main()
