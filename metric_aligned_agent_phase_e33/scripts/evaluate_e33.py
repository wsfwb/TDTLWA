import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
RUN=Path(__file__).resolve().parents[1]; ROOT=Path(__file__).resolve().parents[4]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; MAT=RUN/'stage_c/materialized'; OUT=RUN/'results'; BASE={'Historical Full':0.734584681572,'Reconstructed Full':0.736587511906,'D3':0.733102620511,'E26':0.7417088710802171,'E27':0.7444884347126642,'E29R':0.7474627387432321,'E30':0.7487425903546215,'E31':0.7498925414243726,'E32':0.7499261123953594}
def action(p,full,d3):return np.where(p==full,'KEEP',np.where((p==d3)&(p!=full),'RESIDUAL','REASON'))
def better(a,b):
    if b is None:return True
    return (-float(a['weighted_f1']),-float(a['accuracy']),float(a['final_reason_rate']),a['candidate_id'])<(-float(b['weighted_f1']),-float(b['accuracy']),float(b['final_reason_rate']),b['candidate_id'])
def verify_materialization():
    m=json.loads((MAT/'MATERIALIZATION_SHA256.json').read_text())
    for rel,h in m['files'].items():
        p=RUN/rel; got=hashlib.sha256(p.read_bytes()).hexdigest(); assert got==h,(rel,got,h)
    assert m['gold_in_materialization'] is False and m['reasoner_api_calls']==0
def main():
    verify_materialization(); OUT.mkdir(exist_ok=True)
    # This is the first point at which the evaluator reads Session-5 gold.
    src=list(csv.DictReader(open(E26/'evaluation/results/e26_canonical_predictions.csv'))); assert len(src)==1623
    (RUN/'metadata/E33_GOLD_FIRST_READ.json').write_text(json.dumps({'gold_first_read_stage':'evaluation_after_materialization_hash','canonical_count':len(src),'reasoner_api_calls':0,'session5_labels_sent_to_model':False},indent=2))
    rows=json.loads((RUN/'stage_c/features.json').read_text()); ids=[r['canonical_id'] for r in rows]; assert {r['canonical_id'] for r in src}==set(ids)
    y=np.asarray([int(r['gold_label_evaluation_only']) for r in src]); full=np.asarray([int(r['full_prediction']) for r in src]); d3=np.asarray([int(r['residual_prediction']) for r in src]); man=list(csv.DictReader(open(MAT/'candidate_manifest.csv'))); z=np.load(MAT/'candidate_predictions.npz',mmap_mode='r')['predictions']; assert z.shape==(len(man),1623)
    fields=['candidate_id','family','base','config_json','weighted_f1','accuracy','reason_rate','final_reason_rate','base_reason_rate','incremental_replacement_rate','keep_rate','residual_rate','benefit_total','benefit_captured','benefit_capture_rate','harm_total','harmful_replacements','danger_protection','reason_benefit_precision','net_intervention','deep_hard_repaired']+[f'f1_{j}' for j in range(6)]+['delta_vs_'+n.lower().replace(' ','_') for n in BASE]
    best_abs=best_leg=best_inc=best_hp=None; preds={}; counts={'legacy':0,'incremental':0,'harm':0}
    with open(OUT/'e33_candidate_leaderboard.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,m in enumerate(man):
            p=np.asarray(z[i],dtype=np.int8); cfg=json.loads(m['config_json']); bname=cfg.get('base','E31'); key={'FULL':'full_prediction','D3':'d3_prediction','E26':'e26_prediction','E27':'e27_prediction','E29R':'e29r_prediction','E30':'e30_prediction','E31':'e31_prediction','E32':'e32_prediction'}.get(bname,'e31_prediction'); cheap=np.asarray([int(r[key]) for r in rows]); fa=action(p,full,d3); ba=action(cheap,full,d3); new=p!=cheap; benefit=(cheap!=y)&(p==y); harm=(cheap==y)&(p!=y); captured=benefit&new; harmful=harm&new; calls=fa=='REASON'; wf=float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0)); per=f1_score(y,p,labels=list(range(6)),average=None,zero_division=0); harm_total=int(harm.sum()); captured_n=int(captured.sum()); harmful_n=int(harmful.sum()); rec={'candidate_id':m['candidate_id'],'family':m['family'],'base':bname,'config_json':m['config_json'],'weighted_f1':wf,'accuracy':float(accuracy_score(y,p)),'reason_rate':float(np.mean(calls)),'final_reason_rate':float(np.mean(calls)),'base_reason_rate':float(np.mean(ba=='REASON')),'incremental_replacement_rate':float(np.mean(new)),'keep_rate':float(np.mean(fa=='KEEP')),'residual_rate':float(np.mean(fa=='RESIDUAL')),'benefit_total':int(benefit.sum()),'benefit_captured':captured_n,'benefit_capture_rate':float(captured_n/max(1,int(benefit.sum()))),'harm_total':harm_total,'harmful_replacements':harmful_n,'danger_protection':float(1-harmful_n/max(1,harm_total)),'reason_benefit_precision':float(captured_n/max(1,int(calls.sum()))),'net_intervention':captured_n-harmful_n,'deep_hard_repaired':captured_n}; rec.update({f'f1_{j}':float(v) for j,v in enumerate(per)}); rec.update({'delta_vs_'+n.lower().replace(' ','_'):wf-v for n,v in BASE.items()}); w.writerow(rec)
            if better(rec,best_abs):best_abs=rec; preds['absolute']=p.copy()
            if rec['final_reason_rate']>=.02 and rec['benefit_capture_rate']>=.25:
                counts['legacy']+=1
                if better(rec,best_leg):best_leg=rec; preds['legacy']=p.copy()
            if rec['incremental_replacement_rate']>=.02 and rec['benefit_capture_rate']>=.25:
                counts['incremental']+=1
                if better(rec,best_inc):best_inc=rec; preds['incremental']=p.copy()
            if rec['danger_protection']>=.8:
                counts['harm']+=1
                if better(rec,best_hp):best_hp=rec; preds['harm']=p.copy()
    for label,sel,pkey in [('absolute',best_abs,'absolute'),('legacy_nontrivial',best_leg,'legacy'),('incremental_nontrivial',best_inc,'incremental'),('harm_protected',best_hp,'harm')]:
        if sel is None:continue
        with open(OUT/f'e33_{label}_winner.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerow(sel)
        with open(OUT/f'e33_{label}_winner_canonical_predictions.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['canonical_id','prediction','gold_label_evaluation_only']); w.writeheader();
            for r,v in zip(src,preds[pkey]):w.writerow({'canonical_id':r['canonical_id'],'prediction':int(v),'gold_label_evaluation_only':r['gold_label_evaluation_only']})
    # Preserve one canonical table for the absolute winner and its mechanism inputs.
    if best_abs is not None:
        with open(OUT/'e33_canonical_predictions.csv','w',newline='') as f:
            cols=['canonical_id','gold_label_evaluation_only','full_prediction','residual_prediction','e26_prediction','e27_prediction','e29r_prediction','e30_prediction','e31_prediction','e32_prediction','i1_label','i2_label','i3_label']; w=csv.DictWriter(f,fieldnames=cols); w.writeheader();
            for r,p in zip(rows,preds['absolute']):
                q={k:r.get(k,'') for k in cols if k!='gold_label_evaluation_only'}; q['gold_label_evaluation_only']=src[ids.index(r['canonical_id'])]['gold_label_evaluation_only']; w.writerow(q)
    frontier=[]; lb=list(csv.DictReader(open(OUT/'e33_candidate_leaderboard.csv')))
    for row in lb:
        row['pareto_frontier']='true' if not any(float(x['weighted_f1'])>=float(row['weighted_f1']) and int(x['harmful_replacements'])<=int(row['harmful_replacements']) and (float(x['weighted_f1'])>float(row['weighted_f1']) or int(x['harmful_replacements'])<int(row['harmful_replacements'])) for x in lb) else 'false'; frontier.append(row)
    with open(OUT/'e33_pareto_frontier.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(frontier[0].keys()));w.writeheader();w.writerows(frontier)
    (RUN/'metadata/E33_EVALUATION_SUMMARY.json').write_text(json.dumps({'candidate_count':len(man),'gold_first_read_after_materialization':True,'absolute_winner':best_abs['candidate_id'],'absolute_wf1':best_abs['weighted_f1'],'legacy_count':counts['legacy'],'incremental_count':counts['incremental'],'harm_protected_count':counts['harm'],'reasoner_api_calls':0,'session5_test_tuned':True,'clean_deployment_claim':False},indent=2)); print(json.dumps({'candidate_count':len(man),'absolute_winner':best_abs['candidate_id'],'weighted_f1':best_abs['weighted_f1'],'legacy_count':counts['legacy'],'incremental_count':counts['incremental'],'harm_count':counts['harm']}))
if __name__=='__main__':main()
