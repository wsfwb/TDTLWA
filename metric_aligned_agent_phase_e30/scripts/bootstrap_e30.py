import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; OUT=RUN/'results'
def wf(y,p):return float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0))
def main():
 src=list(csv.DictReader(open(E26/'evaluation/results/e26_canonical_predictions.csv'))); y=np.asarray([int(r['gold_label_evaluation_only']) for r in src]); names={'Historical_Full':'historical_full_prediction','Reconstructed_Full':'full_prediction','D3':'residual_prediction'}; rng=np.random.default_rng(20260822); rows=[]
 for typ,name in [('absolute','e30_absolute_winner'),('nontrivial','e30_nontrivial_winner')]:
  pfile=OUT/(name+'_canonical_predictions.csv')
  if not pfile.exists():continue
  p=np.asarray([int(r['prediction']) for r in csv.DictReader(open(pfile))])
  for b,col in names.items():
   q=np.asarray([int(r[col]) for r in src]); vals=[]
   for _ in range(2000):
    idx=rng.integers(0,len(y),len(y)); vals.append(wf(y[idx],p[idx])-wf(y[idx],q[idx]))
   rows.append({'winner_type':typ,'baseline':b,'observed_delta':wf(y,p)-wf(y,q),'bootstrap_mean':float(np.mean(vals)),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),'replicates':2000,'posthoc_test_selected_bootstrap':True})
 if not rows:raise RuntimeError('no winner predictions')
 with open(OUT/'e30_bootstrap_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'comparisons':len(rows),'replicates':2000}))
if __name__=='__main__':main()
