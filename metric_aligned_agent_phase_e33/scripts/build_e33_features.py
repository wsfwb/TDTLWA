import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; RUN=Path(__file__).resolve().parents[1]
E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; E32=ROOT/'experiment_outputs/metric_aligned_agent_phase_e32/run_20260823T002931_CST_label_pair_local_refinement_v1'
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def main():
    src=read(E26/'evaluation/results/e26_canonical_predictions.csv'); old=json.loads((E32/'stage_c/features.json').read_text()); e32={r['canonical_id']:int(r['prediction']) for r in read(E32/'results/e32_absolute_winner_canonical_predictions.csv')}; out=[]
    for r,o in zip(src,old):
        assert r['canonical_id']==o['canonical_id']; q=dict(o); q['e32_prediction']=e32[q['canonical_id']]; q.pop('gold_label_evaluation_only',None); out.append(q)
    txt=json.dumps(out,separators=(',',':')).lower(); assert not any(k in txt for k in ('gold_label','benefit','harm','outcome','weighted_f1','wf1','metric','oracle'))
    (RUN/'stage_c/features.json').write_text(json.dumps(out,separators=(',',':'))); (RUN/'metadata/FEATURES_MANIFEST.json').write_text(json.dumps({'canonical_count':len(out),'gold_in_features':False,'reasoner_api_calls':0},indent=2)); print(json.dumps({'canonical_count':len(out),'gold_in_features':False,'reasoner_api_calls':0}))
if __name__=='__main__':main()
