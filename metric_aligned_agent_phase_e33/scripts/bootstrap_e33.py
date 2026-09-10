import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
RUN=Path(__file__).resolve().parents[1]; ROOT=Path(__file__).resolve().parents[4]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; OUT=RUN/'results'
def load_pred(p):
    rows=list(csv.DictReader(open(p))); return np.asarray([int(r['prediction']) for r in rows]),np.asarray([int(r['gold_label_evaluation_only']) for r in rows])
def main():
    abs_p,y=load_pred(OUT/'e33_absolute_winner_canonical_predictions.csv'); leg_p,_=load_pred(OUT/'e33_legacy_nontrivial_winner_canonical_predictions.csv')
    e26=list(csv.DictReader(open(E26/'evaluation/results/e26_canonical_predictions.csv'))); gold=np.asarray([int(r['gold_label_evaluation_only']) for r in e26])
    refs={'Reconstructed Full':None,'E32':None,'E31':None}
    refs['Reconstructed Full']=np.asarray([int(r['full_prediction']) for r in e26])
    for name,path in [('E32',ROOT/'experiment_outputs/metric_aligned_agent_phase_e32/run_20260823T002931_CST_label_pair_local_refinement_v1/results/e32_absolute_winner_canonical_predictions.csv'),('E31',ROOT/'experiment_outputs/metric_aligned_agent_phase_e31/run_20260822T213437_CST_local_conditional_refinement_v1/results/e31_absolute_winner_canonical_predictions.csv')]:
        refs[name]=np.asarray([int(r['prediction']) for r in csv.DictReader(open(path))])
    rng=np.random.default_rng(20260822); n=len(gold); records=[]
    for winner_name,pred in [('absolute',abs_p),('legacy_nontrivial',leg_p)]:
        for ref_name,ref in refs.items():
            vals=np.empty(2000); base_delta=float(f1_score(gold,pred,labels=list(range(6)),average='weighted',zero_division=0)-f1_score(gold,ref,labels=list(range(6)),average='weighted',zero_division=0))
            for j in range(2000):
                idx=rng.integers(0,n,n); vals[j]=f1_score(gold[idx],pred[idx],labels=list(range(6)),average='weighted',zero_division=0)-f1_score(gold[idx],ref[idx],labels=list(range(6)),average='weighted',zero_division=0)
            records.append({'winner':winner_name,'reference':ref_name,'observed_delta':base_delta,'bootstrap_mean':float(vals.mean()),'p2_5':float(np.percentile(vals,2.5)),'p97_5':float(np.percentile(vals,97.5)),'replicate_count':2000,'seed':20260822,'posthoc_test_selected_uncertainty':True})
    with open(OUT/'e33_bootstrap_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0].keys()));w.writeheader();w.writerows(records)
    print(json.dumps({'replicate_count':2000,'comparisons':len(records)}))
if __name__=='__main__':main()
