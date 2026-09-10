import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; RUN=Path(__file__).resolve().parents[1]
E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; E27=ROOT/'experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1'; E29=ROOT/'experiment_outputs/metric_aligned_agent_phase_e29r/run_20260822T_continuation_one_call_stage_c_v1'; E30=ROOT/'experiment_outputs/metric_aligned_agent_phase_e30/run_20260822T194845_CST_offline_conditional_policy_v1'; E31=ROOT/'experiment_outputs/metric_aligned_agent_phase_e31/run_20260822T213437_CST_local_conditional_refinement_v1'; E32=ROOT/'experiment_outputs/metric_aligned_agent_phase_e32/run_20260823T002931_CST_label_pair_local_refinement_v1'
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    base=read(E26/'evaluation/results/e26_canonical_predictions.csv'); ids={r['canonical_id'] for r in base}; assert len(base)==1623 and len(ids)==1623
    paths=[E27/'results/e27_winner_canonical_predictions.csv',E29/'results/stage_c_winner_canonical_predictions.csv',E30/'results/e30_absolute_winner_canonical_predictions.csv',E31/'results/e31_absolute_winner_canonical_predictions.csv',E32/'results/e32_absolute_winner_canonical_predictions.csv',E32/'stage_c/features.json']
    for p in paths:
        if p.suffix=='.json': rr=json.loads(p.read_text()); assert len(rr)==1623 and {x['canonical_id'] for x in rr}==ids
        else: rr=read(p); assert len(rr)==1623 and {x['canonical_id'] for x in rr}==ids
    feat=json.loads((E32/'stage_c/features.json').read_text()); txt=json.dumps(feat).lower(); assert not any(k in txt for k in ('gold_label','benefit','harm','outcome','weighted_f1','wf1','metric','oracle'))
    assert len(list((E29/'caches/ADJUDICATOR').glob('*.json')))==200
    all_sources=[E26/'evaluation/results/e26_canonical_predictions.csv']+paths
    manifest={'canonical_count':1623,'canonical_unique':True,'e29r_target_count':200,'e29r_cache_count':200,'reasoner_api_calls':0,'e31_path':str(E31.relative_to(ROOT)),'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in all_sources if p.suffix!='.json'}}
    (RUN/'metadata/INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
