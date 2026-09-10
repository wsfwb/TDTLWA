import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; OUT=RUN/'results'; MAT=RUN/'stage_c/materialized'
BASE={'Historical Full':0.734584681572,'Reconstructed Full':0.736587511906,'D3':0.733102620511,'E26':0.7417088710802171,'E27':0.7444884347126642,'E29R':0.7474627387432321,'E30':0.7487425903546215}
def action(p,full,d3):return np.where(p==full,'KEEP',np.where((p==d3)&(p!=full),'RESIDUAL','REASON'))
def main():
 src=list(csv.DictReader(open(E26/'evaluation/results/e26_canonical_predictions.csv'))); y=np.asarray([int(r['gold_label_evaluation_only']) for r in src]); full=np.asarray([int(r['full_prediction']) for r in src]); d3=np.asarray([int(r['residual_prediction']) for r in src]); rows=json.loads((RUN/'stage_c/features.json').read_text()); man=list(csv.DictReader(open(MAT/'candidate_manifest.csv'))); z=np.load(MAT/'candidate_predictions.npz',mmap_mode='r')['predictions']; out=[]
 if z.shape!=(len(man),len(src)):raise RuntimeError('materialized shape mismatch')
 for i,m in enumerate(man):
  p=np.asarray(z[i],dtype=np.int8); cfg=json.loads(m['config_json']); bname=cfg.get('base','E30'); cheap=np.asarray([int(r.get({'FULL':'full_prediction','D3':'d3_prediction','E26':'e26_prediction','E27':'e27_prediction','E29R':'e29r_prediction','E30':'e30_prediction'}.get(bname,'e30_prediction'))) for r in rows],dtype=np.int8); a=action(p,full,d3); ba=action(cheap,full,d3); benefit=(cheap!=y)&(p==y); harm=(cheap==y)&(p!=y); selected=p!=cheap; captured=benefit&selected; harmful=harm&selected; calls=a=='REASON'; per=f1_score(y,p,labels=list(range(6)),average=None,zero_division=0); r={'candidate_id':m['candidate_id'],'config_json':m['config_json'],'weighted_f1':float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0)),'accuracy':float(accuracy_score(y,p)),'reason_rate':float(np.mean(calls)),'final_reason_rate':float(np.mean(calls)),'base_reason_rate':float(np.mean(ba=='REASON')),'incremental_replacement_rate':float(np.mean(selected)),'keep_rate':float(np.mean(a=='KEEP')),'residual_rate':float(np.mean(a=='RESIDUAL')),'benefit_total':int(benefit.sum()),'benefit_captured':int(captured.sum()),'benefit_capture_rate':float(captured.sum()/max(1,benefit.sum())),'harm_total':int(harm.sum()),'harmful_replacements':int(harmful.sum()),'danger_protection':float(1-harmful.sum()/max(1,harm.sum())),'reason_benefit_precision':float(captured.sum()/max(1,calls.sum())),'net_intervention':int(captured.sum()-harmful.sum()),'deep_hard_repaired':int(captured.sum()),'base':bname}; r.update({f'f1_{j}':float(v) for j,v in enumerate(per)}); out.append(r)
 out.sort(key=lambda r:(-r['weighted_f1'],-r['accuracy'],r['reason_rate'],r['candidate_id']))
 for r in out:
  wf=float(r['weighted_f1'])
  for name,val in BASE.items():r['delta_vs_'+name.lower().replace(' ','_')]=wf-val
 fields=list(out[0]); OUT.mkdir(exist_ok=True)
 with open(OUT/'e31_candidate_leaderboard.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
 absw=out[0]; non=[r for r in out if r['incremental_replacement_rate']>=.02 and r['benefit_capture_rate']>=.25]; hp=[r for r in out if r['danger_protection']>=.8]
 for nm,sel in [('e31_absolute_winner',absw),('e31_nontrivial_winner',non[0] if non else None),('e31_harm_protected_winner',hp[0] if hp else None)]:
  with open(OUT/(nm+'.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(sel) if sel else None
  if sel:
   j=next(j for j,m in enumerate(man) if m['candidate_id']==sel['candidate_id']); p=np.asarray(z[j],dtype=np.int8)
   with open(OUT/(nm+'_canonical_predictions.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=['canonical_id','prediction','gold_label_evaluation_only']);w.writeheader();[w.writerow({'canonical_id':r['canonical_id'],'prediction':int(v),'gold_label_evaluation_only':r['gold_label_evaluation_only']}) for r,v in zip(src,p)]
 with open(OUT/'e31_canonical_predictions.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['canonical_id','gold_label_evaluation_only','full_prediction','residual_prediction','e26_prediction','e27_prediction','e29r_prediction','e30_prediction']);w.writeheader();[w.writerow({k:r.get(k,'') for k in w.fieldnames}) for r in rows]
 (RUN/'metadata/E31_EVALUATION_SUMMARY.json').write_text(json.dumps({'candidate_count':len(out),'absolute_winner':absw['candidate_id'],'absolute_wf1':absw['weighted_f1'],'nontrivial_count':len(non),'harm_protected_count':len(hp),'reasoner_api_calls':0},indent=2));print(json.dumps({'candidate_count':len(out),'absolute_winner':absw['candidate_id'],'weighted_f1':absw['weighted_f1'],'nontrivial':len(non),'harm_protected':len(hp)}))
if __name__=='__main__':main()
