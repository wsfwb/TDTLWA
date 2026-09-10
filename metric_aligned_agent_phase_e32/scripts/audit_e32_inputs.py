import csv,json,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]
S={'e26':ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1/evaluation/results/e26_canonical_predictions.csv','e27':ROOT/'experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1/results/e27_winner_canonical_predictions.csv','e29r':ROOT/'experiment_outputs/metric_aligned_agent_phase_e29r/run_20260822T_continuation_one_call_stage_c_v1/results/stage_c_winner_canonical_predictions.csv','e30':ROOT/'experiment_outputs/metric_aligned_agent_phase_e30/run_20260822T194845_CST_offline_conditional_policy_v1/results/e30_absolute_winner_canonical_predictions.csv','e31':ROOT/'experiment_outputs/metric_aligned_agent_phase_e31/run_20260822T213437_CST_local_conditional_refinement_v1/results/e31_absolute_winner_canonical_predictions.csv'}
E29R=ROOT/'experiment_outputs/metric_aligned_agent_phase_e29r/run_20260822T_continuation_one_call_stage_c_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def main():
 data={k:read(v) for k,v in S.items()}
 sets=[{r['canonical_id'] for r in x} for x in data.values()]
 if any(len(x)!=1623 for x in sets) or any(x!=sets[0] for x in sets[1:]):raise RuntimeError('canonical alignment failed')
 target=read(E29R/'targets/target_pool.csv'); caches=list((E29R/'caches/ADJUDICATOR').glob('*.json'))
 if len(target)!=200 or len({r['canonical_id'] for r in target})!=200 or len(caches)!=200:raise RuntimeError('target/cache count failed')
 out={'canonical_count':1623,'target_pool_count':200,'adjudicator_cache_count':200,'reasoner_api_calls':0,'sources':{k:sha(v) for k,v in S.items()},'source_paths':{k:str(v) for k,v in S.items()}}
 (RUN/'metadata/INPUT_MANIFEST.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
