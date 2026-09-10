import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from .e30_policy_core import action_provenance
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; MAT=RUN/'stage_c/materialized'; OUT=RUN/'results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_csv(p):
 with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def verify():
 rec=json.loads((MAT/'STAGE_C_MATERIALIZATION_SHA256.json').read_text())
 for rel,h in rec['files'].items():
  if sha(RUN/rel)!=h:raise RuntimeError('hash mismatch '+rel)
 if rec.get('gold_in_materialization'):raise RuntimeError('gold materialized')
def main():
 verify(); src=read_csv(E26/'evaluation/results/e26_canonical_predictions.csv'); y=np.asarray([int(r['gold_label_evaluation_only']) for r in src]); full=np.asarray([int(r['full_prediction']) for r in src]); d3=np.asarray([int(r['residual_prediction']) for r in src]); OUT.mkdir(parents=True,exist_ok=True); (RUN/'metadata/E30_GOLD_FIRST_READ.json').write_text(json.dumps({'gold_read':True,'stage':'offline_evaluation','canonical_count':len(src)},indent=2)); man=read_csv(MAT/'candidate_manifest.csv'); z=np.load(MAT/'candidate_predictions.npz',mmap_mode='r')['predictions']; rows=[]
 for i,m in enumerate(man):
  p=np.asarray(z[i],dtype=np.int8); cfg=json.loads(m['config_json']); bn=cfg.get('base','FULL'); base={'FULL':full,'D3':d3}.get(bn,full); a=action_provenance(p,base,d3); reason=a=='REASON'; benefit=(base!=y)&(p==y); harm=(base==y)&(p!=y); cap=benefit&reason; bad=harm&reason; calls=int(reason.sum()); ht=int(harm.sum()); per=f1_score(y,p,labels=list(range(6)),average=None,zero_division=0); r={'candidate_id':m['candidate_id'],'config_json':m['config_json'],'weighted_f1':float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0)),'accuracy':float(accuracy_score(y,p)),'reason_rate':calls/len(y),'reason_replacement_rate':calls/len(y),'keep_rate':float(np.mean(a=='KEEP')),'residual_rate':float(np.mean(a=='RESIDUAL')),'benefit_total':int(benefit.sum()),'benefit_captured':int(cap.sum()),'benefit_capture_rate':float(cap.sum()/max(1,benefit.sum())),'harm_total':ht,'harmful_replacements':int(bad.sum()),'danger_protection':float((ht-bad.sum())/max(1,ht)),'reason_benefit_precision':float(cap.sum()/max(1,calls)),'net_intervention':int(cap.sum()-bad.sum()),'deep_hard_repaired':int(cap.sum()),'base':bn}; r.update({f'f1_{j}':float(v) for j,v in enumerate(per)}); rows.append(r)
 rows.sort(key=lambda r:(-r['weighted_f1'],-r['accuracy'],r['reason_rate'],r['candidate_id'])); fields=list(rows[0])
 with open(OUT/'e30_candidate_leaderboard.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 absw=rows[0]; non=[r for r in rows if r['reason_rate']>=.02 and r['benefit_capture_rate']>=.25]; hp=[r for r in rows if r['danger_protection']>=.8]
 # Persist the joined canonical table once (gold is evaluation-only and is never in materialized inputs).
 with open(OUT/'e30_canonical_predictions.csv','w',newline='') as f:
  cw=csv.DictWriter(f,fieldnames=['canonical_id','gold_label_evaluation_only','full_prediction','residual_prediction','e26_prediction','e27_prediction','e29r_prediction'])
  cw.writeheader()
  for r in src: cw.writerow({k:r.get(k,'') for k in cw.fieldnames})
 for nm,sel in [('e30_absolute_winner',absw),('e30_nontrivial_winner',non[0] if non else None),('e30_harm_protected_winner',hp[0] if hp else None)]:
  with open(OUT/(nm+'.csv'),'w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();
   if sel:w.writerow(sel)
  if sel:
   j=next(j for j,m in enumerate(man) if m['candidate_id']==sel['candidate_id']); p=np.asarray(z[j],dtype=np.int8)
   with open(OUT/(nm+'_canonical_predictions.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['canonical_id','prediction','full_prediction','d3_prediction']);w.writeheader();[w.writerow({'canonical_id':r['canonical_id'],'prediction':int(v),'full_prediction':r['full_prediction'],'d3_prediction':r['residual_prediction']}) for r,v in zip(src,p)]
 (RUN/'metadata/E30_EVALUATION_SUMMARY.json').write_text(json.dumps({'candidate_count':len(rows),'absolute_winner':absw['candidate_id'],'absolute_wf1':absw['weighted_f1'],'nontrivial_count':len(non),'harm_protected_count':len(hp),'reasoner_api_calls':0},indent=2)); print(json.dumps({'candidate_count':len(rows),'absolute_winner':absw['candidate_id'],'weighted_f1':absw['weighted_f1'],'nontrivial':len(non),'harm_protected':len(hp)}))
if __name__=='__main__':main()
