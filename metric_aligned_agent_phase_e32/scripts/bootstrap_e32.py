import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1/evaluation/results/e26_canonical_predictions.csv'; OUT=RUN/'results'
def wf(y,p):
 return float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0))
def main():
 src=list(csv.DictReader(open(E26))); y=np.asarray([int(r['gold_label_evaluation_only']) for r in src]); bases={'Historical Full':[int(r['historical_full_prediction']) for r in src],'Reconstructed Full':[int(r['full_prediction']) for r in src],'D3':[int(r['residual_prediction']) for r in src]}; rng=np.random.default_rng(20260822); rows=[]
 winner_files=[('absolute','e32_absolute_winner_canonical_predictions.csv'),('legacy_nontrivial','e32_legacy_nontrivial_winner_canonical_predictions.csv'),('incremental_nontrivial','e32_incremental_nontrivial_winner_canonical_predictions.csv'),('harm_protected','e32_harm_protected_winner_canonical_predictions.csv')]
 for typ,file in winner_files:
  pth=OUT/file
  if not pth.exists():continue
  p=np.asarray([int(r['prediction']) for r in csv.DictReader(open(pth))])
  for b,q in bases.items():
   q=np.asarray(q); vals=[]
   for _ in range(2000):
    ix=rng.integers(0,len(y),len(y)); vals.append(wf(y[ix],p[ix])-wf(y[ix],q[ix]))
   rows.append({'winner_type':typ,'baseline':b,'observed_delta':wf(y,p)-wf(y,q),'bootstrap_mean':float(np.mean(vals)),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),'replicates':2000,'posthoc_test_selected_bootstrap':True})
 e30=np.asarray([int(r['e30_prediction']) for r in csv.DictReader(open(OUT/'e32_canonical_predictions.csv'))]); e31=np.asarray([int(r['e31_prediction']) for r in csv.DictReader(open(OUT/'e32_canonical_predictions.csv'))]); vals=[]
 for _ in range(2000):
  ix=rng.integers(0,len(y),len(y)); vals.append(wf(y[ix],e31[ix])-wf(y[ix],e30[ix]))
 delta={'observed_delta':wf(y,e31)-wf(y,e30),'bootstrap_mean':float(np.mean(vals)),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),'replicates':2000,'posthoc_test_selected_bootstrap':True}
 with open(OUT/'e32_bootstrap_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with open(OUT/'e32_e31_vs_e30_bootstrap.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(delta));w.writeheader();w.writerow(delta)
 print({'comparisons':len(rows),'e31_vs_e30':delta})
if __name__=='__main__':main()
