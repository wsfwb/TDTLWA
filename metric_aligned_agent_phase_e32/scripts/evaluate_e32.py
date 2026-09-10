import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; OUT=RUN/'results'; MAT=RUN/'stage_c/materialized'
BASE={'Historical Full':0.734584681572,'Reconstructed Full':0.736587511906,'D3':0.733102620511,'E26':0.7417088710802171,'E27':0.7444884347126642,'E29R':0.7474627387432321,'E30':0.7487425903546215,'E31':0.7498925414243726}
def action(p,full,d3):return np.where(p==full,'KEEP',np.where((p==d3)&(p!=full),'RESIDUAL','REASON'))
def better(a,b):
 if b is None:return True
 return (-float(a['weighted_f1']),-float(a['accuracy']),float(a['final_reason_rate']),a['candidate_id'])<(-float(b['weighted_f1']),-float(b['accuracy']),float(b['final_reason_rate']),b['candidate_id'])
def main():
 src=list(csv.DictReader(open(E26/'evaluation/results/e26_canonical_predictions.csv'))); y=np.asarray([int(r['gold_label_evaluation_only']) for r in src]); full=np.asarray([int(r['full_prediction']) for r in src]); d3=np.asarray([int(r['residual_prediction']) for r in src]); rows=json.loads((RUN/'stage_c/features.json').read_text()); man=list(csv.DictReader(open(MAT/'candidate_manifest.csv'))); z=np.load(MAT/'candidate_predictions.npz',mmap_mode='r')['predictions'];
 keys={'FULL':'full_prediction','D3':'d3_prediction','E26':'e26_prediction','E27':'e27_prediction','E29R':'e29r_prediction','E30':'e30_prediction','E31':'e31_prediction'}
 if z.shape!=(len(man),len(src)):raise RuntimeError('materialized shape mismatch')
 fields=['candidate_id','config_json','weighted_f1','accuracy','reason_rate','final_reason_rate','base_reason_rate','incremental_replacement_rate','keep_rate','residual_rate','benefit_total','benefit_captured','benefit_capture_rate','harm_total','harmful_replacements','danger_protection','reason_benefit_precision','net_intervention','deep_hard_repaired','base']+[f'f1_{j}' for j in range(6)]+['delta_vs_'+n.lower().replace(' ','_') for n in BASE]
 best_abs=best_legacy=best_incr=best_hp=None; pred_abs=pred_legacy=pred_incr=pred_hp=None; OUT.mkdir(exist_ok=True)
 with open(OUT/'e32_candidate_leaderboard.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for i,m in enumerate(man):
   p=np.asarray(z[i],dtype=np.int8); cfg=json.loads(m['config_json']); bname=cfg.get('base','E30'); cheap=np.asarray([int(r[keys.get(bname,'e30_prediction')]) for r in rows]); final_a=action(p,full,d3); base_a=action(cheap,full,d3); new=p!=cheap; benefit=(cheap!=y)&(p==y); harm=(cheap==y)&(p!=y); captured=benefit&new; harmful=harm&new; calls=final_a=='REASON'; per=f1_score(y,p,labels=list(range(6)),average=None,zero_division=0); wf=float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0)); rec={'candidate_id':m['candidate_id'],'config_json':m['config_json'],'weighted_f1':wf,'accuracy':float(accuracy_score(y,p)),'reason_rate':float(np.mean(calls)),'final_reason_rate':float(np.mean(calls)),'base_reason_rate':float(np.mean(base_a=='REASON')),'incremental_replacement_rate':float(np.mean(new)),'keep_rate':float(np.mean(final_a=='KEEP')),'residual_rate':float(np.mean(final_a=='RESIDUAL')),'benefit_total':int(benefit.sum()),'benefit_captured':int(captured.sum()),'benefit_capture_rate':float(captured.sum()/max(1,benefit.sum())),'harm_total':int(harm.sum()),'harmful_replacements':int(harmful.sum()),'danger_protection':float(1-harmful.sum()/max(1,harm.sum())),'reason_benefit_precision':float(captured.sum()/max(1,calls.sum())),'net_intervention':int(captured.sum()-harmful.sum()),'deep_hard_repaired':int(captured.sum()),'base':bname}; rec.update({f'f1_{j}':float(v) for j,v in enumerate(per)}); rec.update({'delta_vs_'+n.lower().replace(' ','_'):wf-v for n,v in BASE.items()}); w.writerow(rec)
   if better(rec,best_abs):best_abs,pred_abs=rec,p.copy()
   if rec['final_reason_rate']>=.02 and rec['benefit_capture_rate']>=.25 and better(rec,best_legacy):best_legacy,pred_legacy=rec,p.copy()
   if rec['incremental_replacement_rate']>=.02 and rec['benefit_capture_rate']>=.25 and better(rec,best_incr):best_incr,pred_incr=rec,p.copy()
   if rec['danger_protection']>=.8 and better(rec,best_hp):best_hp,pred_hp=rec,p.copy()
 for nm,sel,pred in [('e32_absolute_winner',best_abs,pred_abs),('e32_legacy_nontrivial_winner',best_legacy,pred_legacy),('e32_incremental_nontrivial_winner',best_incr,pred_incr),('e32_harm_protected_winner',best_hp,pred_hp)]:
  with open(OUT/(nm+'.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(sel) if sel else None
  if sel:
   with open(OUT/(nm+'_canonical_predictions.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=['canonical_id','prediction','gold_label_evaluation_only']);w.writeheader();[w.writerow({'canonical_id':r['canonical_id'],'prediction':int(v),'gold_label_evaluation_only':r['gold_label_evaluation_only']}) for r,v in zip(src,pred)]
 with open(OUT/'e32_canonical_predictions.csv','w',newline='') as f:
  cols=['canonical_id','gold_label_evaluation_only','full_prediction','residual_prediction','e26_prediction','e27_prediction','e29r_prediction','e30_prediction','e31_prediction']; w=csv.DictWriter(f,fieldnames=cols);w.writeheader();[w.writerow({k:r.get(k,'') for k in cols}) for r in rows]
 (RUN/'metadata/E32_EVALUATION_SUMMARY.json').write_text(json.dumps({'candidate_count':len(man),'absolute_winner':best_abs['candidate_id'],'absolute_wf1':best_abs['weighted_f1'],'legacy_nontrivial_count':int(best_legacy is not None),'incremental_nontrivial_count':int(best_incr is not None),'harm_protected_count':int(best_hp is not None),'reasoner_api_calls':0},indent=2));print(json.dumps({'candidate_count':len(man),'absolute_winner':best_abs['candidate_id'],'weighted_f1':best_abs['weighted_f1'],'legacy':best_legacy is not None,'incremental':best_incr is not None,'harm':best_hp is not None}))
if __name__=='__main__':main()
