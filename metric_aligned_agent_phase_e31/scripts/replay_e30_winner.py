import csv,json
from pathlib import Path
from sklearn.metrics import f1_score
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'
def main():
 gold=list(csv.DictReader(open(E26/'evaluation/results/e26_canonical_predictions.csv'))); pred=list(csv.DictReader(open(RUN/'stage_c/materialized/e30_winner_predictions.csv'))); gm={r['canonical_id']:r for r in gold}; pm={r['canonical_id']:r for r in pred}
 if set(gm)!=set(pm) or len(pm)!=1623:raise RuntimeError('replay alignment failure')
 keys=sorted(gm); y=[int(gm[k]['gold_label_evaluation_only']) for k in keys]; p=[int(pm[k]['prediction']) for k in keys]; wf=float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0)); out={'canonical_count':len(p),'unique_canonical_count':len(set(pm)),'weighted_f1':wf,'expected_weighted_f1':0.7487425903546215,'exact_replay':abs(wf-0.7487425903546215)<=1e-12,'gold_read_stage':'post_materialization'}; (RUN/'results/e31_replay_summary.csv').write_text('metric,value\n'+'\n'.join(f'{k},{v}' for k,v in out.items())+'\n'); (RUN/'metadata/E31_GOLD_FIRST_READ.json').write_text(json.dumps({'gold_read':True,'stage':'e30_replay_after_materialization_hash','canonical_count':1623},indent=2)); print(json.dumps(out))
if __name__=='__main__':main()
