import csv,json,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]
E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; E27=ROOT/'experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1'; E29R=ROOT/'experiment_outputs/metric_aligned_agent_phase_e29r/run_20260822T_continuation_one_call_stage_c_v1'; E30=ROOT/'experiment_outputs/metric_aligned_agent_phase_e30/run_20260822T194845_CST_offline_conditional_policy_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def main():
 e26=rows(E26/'evaluation/results/e26_canonical_predictions.csv'); e27=rows(E27/'results/e27_winner_canonical_predictions.csv'); e29=rows(E29R/'results/stage_c_winner_canonical_predictions.csv'); e30=rows(E30/'results/e30_absolute_winner_canonical_predictions.csv'); target=rows(E29R/'targets/target_pool.csv'); caches=list((E29R/'caches/ADJUDICATOR').glob('*.json'))
 sets=[{r['canonical_id'] for r in x} for x in (e26,e27,e29,e30)]
 if any(len(x)!=1623 for x in sets) or any(x!=sets[0] for x in sets[1:]):raise RuntimeError('canonical alignment failure')
 if len(target)!=200 or len({r['canonical_id'] for r in target})!=200 or len(caches)!=200:raise RuntimeError('target/cache failure')
 meta={'canonical_count':1623,'target_pool_count':200,'adjudicator_cache_count':200,'reasoner_api_calls':0,'e30_prediction_count':len(e30),'e30_replay_deferred_until_materialization':True,'sources':{str(p):sha(p) for p in [E26/'evaluation/results/e26_canonical_predictions.csv',E27/'results/e27_winner_canonical_predictions.csv',E29R/'results/stage_c_winner_canonical_predictions.csv',E30/'results/e30_absolute_winner_canonical_predictions.csv']}}
 (RUN/'metadata/INPUT_MANIFEST.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
if __name__=='__main__':main()
