import csv,json,hashlib
from pathlib import Path
from sklearn.metrics import f1_score
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'
def main():
 lock=json.loads((RUN/'stage_c/materialized/MATERIALIZATION_SHA256.json').read_text())
 for rel,h in lock['files'].items():
  if hashlib.sha256((RUN/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('materialization hash mismatch')
 feat=json.loads((RUN/'stage_c/features.json').read_text()); gold=list(csv.DictReader(open(E26/'evaluation/results/e26_canonical_predictions.csv'))); gm={r['canonical_id']:r for r in gold}; rows=[]
 for name,key in (('E30','e30_prediction'),('E31','e31_prediction')):
  y=[int(gm[r['canonical_id']]['gold_label_evaluation_only']) for r in feat]; p=[int(r[key]) for r in feat]; expected=0.7487425903546215 if name=='E30' else 0.7498925414243726; rows.append({'name':name,'weighted_f1':float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0)),'expected':expected,'exact':abs(float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0))-expected)<=1e-12})
 with open(RUN/'results/e32_replay_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (RUN/'metadata/E32_GOLD_FIRST_READ.json').write_text(json.dumps({'gold_read':True,'stage':'replay_after_materialization_hash','canonical_count':1623},indent=2));print(rows)
if __name__=='__main__':main()
