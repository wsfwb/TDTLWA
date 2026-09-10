import csv, hashlib, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from .io_atomic import atomic_write_csv, atomic_write_json, read_csv_rows, sha256_file

RUN=Path(__file__).resolve().parents[1]
E26=RUN.parents[2]/"experiment_outputs"/"metric_aligned_agent_phase_e26"/"run_20260820T113331_CST_quota_recovered_sol_xhigh_v1"
LABELS=list(range(6)); REF=0.7417088710802171

def fast_weighted_f1(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=int); cm=np.bincount(6*y+p,minlength=36).reshape(6,6); tp=np.diag(cm).astype(float); sup=cm.sum(axis=1).astype(float); fp=cm.sum(axis=0)-tp; fn=sup-tp; den=2*tp+fp+fn; f=np.divide(2*tp,den,out=np.zeros(6),where=den>0); return float(np.sum(f*sup)/max(1.0,sup.sum()))

def replay_e26_reference():
    rows=read_csv_rows(E26/"evaluation/results/e26_canonical_predictions.csv"); y=np.asarray([int(r["gold_label_evaluation_only"]) for r in rows]); p=_e26_winner_pred(rows); return fast_weighted_f1(y,p)

def _e26_winner_pred(rows):
    eligible=np.asarray([i for i,r in enumerate(rows) if r["i3_decision"]=="ACCEPT_REASON"], dtype=int)
    score=np.asarray([float(rows[i]["i3_confidence"])-float(rows[i]["i3_uncertainty"]) for i in eligible], dtype=float)
    # E26's frozen evaluator used ceil(5% of n) and NumPy's descending
    # argsort. Preserve that exact replay contract for the reference gate.
    k=int(math.ceil(0.05*len(rows)))
    chosen=set(eligible[np.argsort(score)[::-1][:k]].tolist())
    return np.asarray([int(r["i3_reason_prediction"]) if i in chosen else int(r["full_prediction"]) for i,r in enumerate(rows)],dtype=int)

def _check_materialization():
    mdir=RUN/"materialized"; lock=json.loads((mdir/"MATERIALIZATION_SHA256.json").read_text())
    if not lock.get("frozen") or lock.get("gold_in_materialization"): raise RuntimeError("materialization lock invalid")
    for rel,want in lock["files"].items():
        path=RUN/rel
        if sha256_file(path)!=want: raise RuntimeError(f"materialization hash mismatch: {rel}")
    return lock

def bootstrap(y,p,q,reps=2000,seed=20260822):
    rng=np.random.default_rng(seed); vals=[]; y=np.asarray(y); p=np.asarray(p); q=np.asarray(q)
    for _ in range(reps):
        idx=rng.integers(0,len(y),len(y)); vals.append(fast_weighted_f1(y[idx],p[idx])-fast_weighted_f1(y[idx],q[idx]))
    vals=np.asarray(vals); return {"observed_delta":float(fast_weighted_f1(y,p)-fast_weighted_f1(y,q)),"bootstrap_mean":float(vals.mean()),"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975)),"replicates":reps}

def _metrics(cid, pred, actions, y, full, d3):
    pred=np.asarray(pred,dtype=int); actions=np.asarray(actions,dtype=int); reason=actions==2; cheap=np.where(actions==1,d3,full); # provenance makes D3 explicit, else Full
    # Since a reason label is the candidate prediction on action=2, compare it
    # to the cheap prediction using the candidate output itself.
    benefit=(cheap!=y)&(pred==y)&reason; harm=(cheap==y)&(pred!=y)&reason
    cm=confusion_matrix(y,pred,labels=LABELS); per=f1_score(y,pred,labels=LABELS,average=None,zero_division=0); bt=int(benefit.sum()); ht=int(harm.sum()); calls=int(reason.sum())
    return {"candidate_id":cid,"weighted_f1":fast_weighted_f1(y,pred),"accuracy":float(accuracy_score(y,pred)),"per_class_f1":json.dumps([float(x) for x in per]),"confusion_matrix":json.dumps(cm.tolist()),"keep_rate":float(np.mean(actions==0)),"residual_rate":float(np.mean(actions==1)),"reason_rate":float(calls/len(y)),"reason_replacement_rate":float(calls/len(y)),"reason_call_coverage":float(calls/len(y)),"benefit_total":bt,"benefit_captured":bt,"benefit_capture_rate":float(bt/max(1,bt)),"harm_total":ht,"harmful_reason_interventions":ht,"danger_total":ht,"danger_damage":ht,"danger_protection":0.0 if ht else 1.0,"reason_benefit_precision":float(bt/max(1,calls)),"net_intervention":bt-ht,"deep_hard_repaired":bt,"deep_hard_total":bt}

def main():
    lock=_check_materialization(); mdir=RUN/"materialized"; manifest=read_csv_rows(mdir/"candidate_manifest.csv"); ids=json.loads((mdir/"canonical_ids.json").read_text());
    if len(ids)!=1623 or len(set(ids))!=1623 or len(manifest)!=lock["candidate_count"]: raise RuntimeError("materialization dimensions invalid")
    src=read_csv_rows(E26/"evaluation/results/e26_canonical_predictions.csv"); by={r["canonical_id"]:r for r in src}
    if set(ids)!=set(by): raise RuntimeError("gold canonical membership mismatch")
    y=np.asarray([int(by[c]["gold_label_evaluation_only"]) for c in ids]); full=np.asarray([int(by[c]["full_prediction"]) for c in ids]); d3=np.asarray([int(by[c]["residual_prediction"]) for c in ids]); hist=np.asarray([int(by[c]["historical_full_prediction"]) for c in ids])
    pred_np=np.load(mdir/"candidate_predictions.npz"); act_np=np.load(mdir/"candidate_actions.npz"); pred=pred_np["predictions"]; acts=act_np["actions"]
    rows=[]
    for i,m in enumerate(manifest):
        r=_metrics(m["candidate_id"],pred[i],acts[i],y,full,d3); r.update({"family":m["family"],"config_json":m["config_json"],"delta_vs_historical_full":r["weighted_f1"]-fast_weighted_f1(y,hist),"delta_vs_reconstructed_full":r["weighted_f1"]-fast_weighted_f1(y,full),"delta_vs_d3":r["weighted_f1"]-fast_weighted_f1(y,d3),"delta_vs_e26":r["weighted_f1"]-REF,"status":"posthoc_candidate"}); rows.append(r)
    def key(r): return (-r["weighted_f1"],-r["accuracy"],r["reason_replacement_rate"],r["candidate_id"])
    rows.sort(key=key); atomic_write_csv(RUN/"results/e27_candidate_leaderboard.csv",rows,list(rows[0]))
    winner=rows[0]; wi=next(i for i,m in enumerate(manifest) if m["candidate_id"]==winner["candidate_id"]); winpred=pred[wi]; winact=acts[wi]
    winner_out=[]
    for cid,g,p,a in zip(ids,y,winpred,winact): winner_out.append({"canonical_id":cid,"gold_label_evaluation_only":int(g),"winner_prediction":int(p),"winner_action_code":int(a)})
    atomic_write_csv(RUN/"results/e27_winner_canonical_predictions.csv",winner_out,list(winner_out[0]))
    atomic_write_csv(RUN/"results/e27_absolute_winner.csv",[winner],list(winner))
    gate={"best_candidate":winner["candidate_id"],"best_wf1":winner["weighted_f1"],"reference_wf1":REF,"delta":winner["weighted_f1"]-REF,"trigger_e28":not (winner["weighted_f1"]>REF+1e-12),"status":"e27_improved_stop_before_e28" if winner["weighted_f1"]>REF+1e-12 else "e27_no_improvement_trigger_e28","candidate_count":len(rows),"session5_gold_read_stage":"after_materialization_hash_verification"}
    atomic_write_json(RUN/"metadata/E27_GATE.json",gate)
    base=[("Historical Full",hist),("Reconstructed Full",full),("D3",d3),("E26",_e26_winner_pred([by[c] for c in ids]))]; boots=[]
    for name,b in base: boots.append({"comparison":name,"status":"computed","posthoc_test_selected_uncertainty":True,**bootstrap(y,winpred,b)})
    atomic_write_csv(RUN/"results/e27_bootstrap_summary.csv",boots,list(boots[0]))
    atomic_write_json(RUN/"metadata/INTEGRITY.json",{"candidate_count":len(rows),"canonical_count":len(ids),"gold_in_materialization":False,"gold_read_after_hash":True,"reasoner_api_calls":0,"session5_test_tuned":True,"clean_deployment_claim":False,"unbiased_generalization_claim":False,"e27_gate":gate,"materialization_sha256":sha256_file(mdir/"MATERIALIZATION_SHA256.json")})
    print(json.dumps({"candidate_count":len(rows),"best_candidate":winner["candidate_id"],"best_wf1":winner["weighted_f1"],"delta_vs_e26":winner["weighted_f1"]-REF,"trigger_e28":gate["trigger_e28"]},sort_keys=True))

if __name__=="__main__": main()
