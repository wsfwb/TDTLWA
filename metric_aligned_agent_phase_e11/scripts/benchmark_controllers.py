#!/usr/bin/env python3
"""Attempt03 E11 D3/V1/V3 reconstruction with verified R1 public labels.

Although the E0T prompt enumerates its textual labels in legacy TDTL order,
the saved E1/E1B/E2 *numeric prediction columns* have already been converted
to the ClarifyMER public order ``hap, sad, neu, ang, exc, fru``.  Attempt02
incorrectly applied a second conversion.  This attempt uses the frozen stored
numeric IDs as public labels, before any controller supervision or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[2]
E1 = PROJECT / "experiment_outputs/metric_aligned_agent_phase_e1/run_20260811T090000_CST_exploratory_v1/results/controller_oof_predictions.csv"
E1B = PROJECT / "experiment_outputs/metric_aligned_agent_phase_e1b/run_20260811T100000_CST_independent_outerdev_v1/results/e1b_predictions.csv"
E2 = PROJECT / "experiment_outputs/metric_aligned_agent_phase_e2/run_20260811T141645_CST_formal_test_v1/results/formal_test_predictions.csv"
CANONICAL = PROJECT / "experiment_outputs/metric_aligned_agent_phase_e05/run_20260810T183930_CST_exploratory_v1/results/canonical_iemocap_utterances.csv"
E8_SPEC = PROJECT / "experiment_outputs/metric_aligned_agent_phase_e8/run_20260812T001000_CST_e2_guided_controller_optimization_attempt02/frozen_posthoc_v3_pairwise_sum/POSTHOC_V3_SPEC.json"
PUBLIC = np.arange(6); ACTIONS = np.array(["KEEP", "RESIDUAL", "REASON"], dtype=object)
V3_FEATURES = ["score_keep_pre_reason","score_residual_pre_reason","score_reason_pre_reason","full_residual_agree","score_reason_minus_cheap","score_residual_minus_keep",*[f"full_class_{k}" for k in range(6)],*[f"residual_class_{k}" for k in range(6)]]


def softmax(values: np.ndarray) -> np.ndarray:
    x=np.asarray(values,float); z=np.exp(x-x.max(1,keepdims=True)); return z/z.sum(1,keepdims=True)

def materialize(action: Sequence[str], keep: np.ndarray, residual: np.ndarray, reason: np.ndarray) -> np.ndarray:
    action=np.asarray(action,object); out=np.asarray(keep,int).copy(); out[action=="RESIDUAL"]=np.asarray(residual,int)[action=="RESIDUAL"]; out[action=="REASON"]=np.asarray(reason,int)[action=="REASON"]; return out

def group_folds(groups: np.ndarray, n_splits: int=5):
    groups=np.asarray(groups,object)
    for train,valid in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)),groups=groups):
        if set(groups[train])&set(groups[valid]): raise RuntimeError("canonical/dialogue group crossed OOF folds")
        yield train,valid

def pairwise_feature_frame(keep:np.ndarray,residual:np.ndarray,reason:np.ndarray,full_pred:np.ndarray,residual_pred:np.ndarray)->pd.DataFrame:
    frame=pd.DataFrame({"score_keep_pre_reason":np.asarray(keep,float),"score_residual_pre_reason":np.asarray(residual,float),"score_reason_pre_reason":np.asarray(reason,float)})
    frame["full_residual_agree"]=(np.asarray(full_pred)==np.asarray(residual_pred)).astype(float)
    frame["score_reason_minus_cheap"]=frame.score_reason_pre_reason-frame[["score_keep_pre_reason","score_residual_pre_reason"]].max(axis=1)
    frame["score_residual_minus_keep"]=frame.score_residual_pre_reason-frame.score_keep_pre_reason
    for prefix,pred in (("full",full_pred),("residual",residual_pred)):
        for k in range(6): frame[f"{prefix}_class_{k}"]=(np.asarray(pred,int)==k).astype(float)
    return frame[V3_FEATURES]

def _positive(model:Any,matrix:Any)->np.ndarray:
    p=model.predict_proba(matrix); w=np.flatnonzero(np.asarray(model.classes_)==1); return p[:,int(w[0])] if len(w) else np.zeros(matrix.shape[0])

def _safe_logistic(matrix:Any,target:np.ndarray,*,c:float=1.0,balanced:bool=False,seed:int=20260812)->Any:
    target=np.asarray(target,int)
    if np.unique(target).size<2: return DummyClassifier(strategy="constant",constant=int(target[0])).fit(matrix,target)
    return LogisticRegression(C=c,class_weight="balanced" if balanced else None,max_iter=2000,solver="liblinear",random_state=seed).fit(matrix,target)

def _head_features(full_p:np.ndarray,full_z:np.ndarray,states:np.ndarray,fit:dict[str,Any]|None=None):
    flat=np.asarray(states,float).reshape(len(states),-1)
    if fit is None:
        ss=StandardScaler().fit(flat); pca=PCA(n_components=min(32,*flat.shape),random_state=20260810,svd_solver="randomized",iterated_power=3).fit(ss.transform(flat)); fit={"state_scaler":ss,"pca":pca}
    return np.column_stack((full_p,full_z,fit["pca"].transform(fit["state_scaler"].transform(flat)))),fit

def _fit_d3(y:np.ndarray,p:np.ndarray,z:np.ndarray,states:np.ndarray)->dict[str,Any]:
    x,fit=_head_features(p,z,states); target=5*(np.eye(6)[y]-p); counts=np.bincount(y,minlength=6).astype(float); weight=len(y)/np.maximum(1,6*counts[y]); fit["ridge"]=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=75.0))]).fit(x,target,ridge__sample_weight=weight); return {**fit,"alpha":.15,"maximum_norm":3.0}

def _apply_d3(head:dict[str,Any],z:np.ndarray,p:np.ndarray,states:np.ndarray):
    x,_=_head_features(p,z,states,head); residual=np.clip(head["ridge"].predict(x),-4,4); norm=np.linalg.norm(residual,axis=1,keepdims=True); residual*=np.minimum(1,head["maximum_norm"]/np.clip(norm,1e-12,None)); return residual,softmax(z+head["alpha"]*residual)

def _load_split(split:str,labels:bool=True)->dict[str,Any]:
    d=np.load(RUN/"cache"/f"full_{split}_features.npz",allow_pickle=False); ids=d["canonical_ids"].astype(str); expected={"train":5163,"dev":647,"test":1623}[split]
    if len(ids)!=expected or len(set(ids))!=expected: raise RuntimeError(f"invalid {split} full cache")
    out={"ids":ids,"view_logits":np.asarray(d["view_logits"],float),"views":np.asarray(d["view_probability"],float),"states":np.asarray(d["tav_branch_states"],float)}
    if labels: out["labels"]=np.asarray(d["labels"],int)
    return out

def _metadata(ids:Sequence[str])->pd.DataFrame:
    data=pd.read_csv(CANONICAL).rename(columns={"canonical_sample_id":"canonical_id"}); data.canonical_id=data.canonical_id.astype(str); out=data.set_index("canonical_id").loc[list(ids)].reset_index(); out.transcript_raw=out.transcript_raw.fillna("").astype(str); out.speaker=out.speaker.fillna("").astype(str); out.utterance_index=out.utterance_index.astype(float); out.dialogue_id=out.dialogue_id.astype(str); out["dialogue_length"]=out.groupby("dialogue_id").canonical_id.transform("size").astype(float); out["previous_turns_available"]=np.minimum(out.utterance_index,3.0); out["speaker_transition"]=0.0
    for _,idx in out.groupby("dialogue_id",sort=False).groups.items():
        ordered=out.loc[list(idx)].sort_values("utterance_index"); previous=ordered.speaker.shift(1); out.loc[ordered.index,"speaker_transition"]=(previous.notna()&(previous!=ordered.speaker)).astype(float).to_numpy()
    return out

def _r1_canonical()->pd.DataFrame:
    e1=pd.read_csv(E1); e1=e1[e1.controller.eq("C1_Logistic_FeatureB")][["canonical_sample_id","r1_prediction_action_outcome_only"]].rename(columns={"canonical_sample_id":"canonical_id","r1_prediction_action_outcome_only":"reason_prediction_public"}); e1["source_phase"]="E1"
    e1b=pd.read_csv(E1B)[["canonical_sample_id","r1_prediction_action_outcome_only"]].rename(columns={"canonical_sample_id":"canonical_id","r1_prediction_action_outcome_only":"reason_prediction_public"}); e1b["source_phase"]="E1B"
    e2=pd.read_csv(E2)[["canonical_sample_id","reason_prediction"]].rename(columns={"canonical_sample_id":"canonical_id","reason_prediction":"reason_prediction_public"}); e2["source_phase"]="E2"
    all_rows=pd.concat((e1,e1b,e2),ignore_index=True); all_rows.canonical_id=all_rows.canonical_id.astype(str); all_rows.reason_prediction_public=all_rows.reason_prediction_public.astype(int)
    if (all_rows.reason_prediction_public<0).any() or (all_rows.reason_prediction_public>5).any() or (all_rows.groupby("canonical_id").reason_prediction_public.nunique()>1).any(): raise RuntimeError("R1 cache contract conflict")
    out=all_rows.groupby("canonical_id",as_index=False).agg(reason_prediction_public=("reason_prediction_public","first"),source_phase=("source_phase",lambda x:";".join(sorted(set(x)))))
    expected=set().union(*[set(pd.read_csv(RUN/"splits"/f"{s}_ids.csv").original_iemocap_id.astype(str)) for s in ("train","dev","test")])
    if len(out)!=7433 or set(out.canonical_id)!=expected: raise RuntimeError("R1 canonical coverage not 7433")
    out.to_csv(RUN/"cache/r1_canonical_predictions.csv",index=False)
    (RUN/"cache/r1_contract_reuse.json").write_text(json.dumps({"expected":7433,"valid":7433,"newly_generated":0,"model":"gpt-5.6-terra","prompt_version":"r1_context","stored_numeric_label_order":["hap","sad","neu","ang","exc","fru"],"source_columns":"E1/E1B r1_prediction_action_outcome_only; E2 reason_prediction","conversion_applied":False,"validation":"raw stored IDs reproduce known E1/E1B/E2 R1 metric scale; applying legacy_to_public is invalid"},indent=2)+"\n")
    return out

def _r1(split:dict[str,Any],table:pd.DataFrame)->np.ndarray:
    result=table.set_index("canonical_id").reason_prediction_public.reindex(split["ids"])
    if result.isna().any(): raise RuntimeError("missing R1")
    return result.to_numpy(int)

def _crossfit_d3(data:dict[str,Any],groups:np.ndarray):
    n=len(data["ids"]); res=np.full((n,6),np.nan); prob=np.full((n,6),np.nan); p,z=data["views"][:,3],data["view_logits"][:,3]
    for tr,va in group_folds(groups):
        head=_fit_d3(data["labels"][tr],p[tr],z[tr],data["states"][tr]); res[va],prob[va]=_apply_d3(head,z[va],p[va],data["states"][va])
    if np.isnan(res).any(): raise RuntimeError("D3 OOF incomplete")
    return res,prob

def _structural(data:dict[str,Any],d3res:np.ndarray,d3prob:np.ndarray,meta:pd.DataFrame):
    views=data["views"]; full=views[:,3]
    def ent(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
    def mar(p): q=np.sort(p,axis=1);return q[:,-1]-q[:,-2]
    pred=views.argmax(2); mode=np.array([np.bincount(row,minlength=6).argmax() for row in pred[:,:3]])
    scalar=np.column_stack((full.max(1),d3prob.max(1),ent(full),ent(d3prob),mar(full),mar(d3prob),d3prob.max(1)-full.max(1),ent(d3prob)-ent(full),mar(d3prob)-mar(full),(full.argmax(1)!=d3prob.argmax(1)).astype(float),np.array([len(set(r)) for r in pred]),(pred[:,3]==mode).astype(float),(pred[:,1]==pred[:,2]).astype(float),meta.transcript_raw.str.len().to_numpy(float),meta.transcript_raw.str.split().str.len().to_numpy(float),meta.utterance_index.to_numpy(float),meta.previous_turns_available.to_numpy(float),meta.speaker_transition.to_numpy(float),meta.speaker.str.endswith("_F").to_numpy(float),meta.dialogue_length.to_numpy(float)))
    numeric=np.column_stack((views.reshape(len(full),-1),full,d3prob,np.log(np.clip(full,1e-12,1)),d3res,np.eye(6)[full.argmax(1)],np.eye(6)[d3prob.argmax(1)],scalar)); return numeric

def _assemble(feature:dict[str,Any],res:np.ndarray,prob:np.ndarray,r1:np.ndarray,meta:pd.DataFrame):
    return {**feature,"residual_logit":res,"residual_probability":prob,"r1":r1,"meta":meta,"numeric":_structural(feature,res,prob,meta),"full_pred":feature["views"][:,3].argmax(1),"residual_pred":prob.argmax(1),"groups":meta.dialogue_id.astype(str).to_numpy()}

def _fit_v1(num:np.ndarray,texts:Sequence[str],y:np.ndarray,f:np.ndarray,r:np.ndarray,s:np.ndarray):
    scaler=StandardScaler().fit(num); vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=3000,sublinear_tf=True,dtype=np.float64); text=vec.fit_transform(texts); matrix=sparse.hstack((sparse.csr_matrix(scaler.transform(num)),text),format="csr"); target=np.column_stack((f==y,r==y,s==y)).astype(int); return {"scaler":scaler,"vectorizer":vec,"models":[_safe_logistic(matrix,target[:,i],c=1,balanced=False,seed=20260811) for i in range(3)]}

def _score_v1(fit:dict[str,Any],num:np.ndarray,texts:Sequence[str])->np.ndarray:
    x=sparse.hstack((sparse.csr_matrix(fit["scaler"].transform(num)),fit["vectorizer"].transform(texts)),format="csr");return np.column_stack([_positive(m,x) for m in fit["models"]])

def _v1_action(s:np.ndarray)->np.ndarray:
    s=s.copy();s[:,2]-=.02;return ACTIONS[np.argmax(s,axis=1)]

def _crossfit_v1(num:np.ndarray,texts:Sequence[str],y:np.ndarray,f:np.ndarray,r:np.ndarray,s:np.ndarray,g:np.ndarray)->np.ndarray:
    out=np.full((len(y),3),np.nan)
    for tr,va in group_folds(g): out[va]=_score_v1(_fit_v1(num[tr],[texts[i] for i in tr],y[tr],f[tr],r[tr],s[tr]),num[va],[texts[i] for i in va])
    if np.isnan(out).any():raise RuntimeError("V1 OOF incomplete")
    return out

def _fit_v3(scores:np.ndarray,f:np.ndarray,r:np.ndarray,s:np.ndarray,y:np.ndarray):
    raw=pairwise_feature_frame(scores[:,0],scores[:,1],scores[:,2],f,r); scaler=StandardScaler().fit(raw.to_numpy(float));x=scaler.transform(raw.to_numpy(float));models={}; keep_ok,res_ok,reason_ok=f==y,r==y,s==y
    for name,lft,rgt in (("keep_vs_residual",keep_ok,res_ok),("keep_vs_reason",keep_ok,reason_ok),("residual_vs_reason",res_ok,reason_ok)):
        mask=lft!=rgt;models[name]=_safe_logistic(x[mask],rgt[mask].astype(int),c=1,balanced=True,seed=20260812)
    return {"scaler":scaler,"models":models}

def _score_v3(fit:dict[str,Any],scores:np.ndarray,f:np.ndarray,r:np.ndarray):
    raw=pairwise_feature_frame(scores[:,0],scores[:,1],scores[:,2],f,r);x=fit["scaler"].transform(raw.to_numpy(float));m=fit["models"];kr=_positive(m["keep_vs_residual"],x);ks=_positive(m["keep_vs_reason"],x);rs=_positive(m["residual_vs_reason"],x);summed=np.column_stack(((1-kr)+(1-ks),kr+(1-rs),ks+rs));return ACTIONS[np.argmax(summed,1)],summed,np.column_stack((kr,ks,rs))

def _devrow(name:str,y:np.ndarray,a:np.ndarray,d:dict[str,Any]):
    p=materialize(a,d["full_pred"],d["residual_pred"],d["r1"]);return {"method":name,"n":len(y),"weighted_f1":float(f1_score(y,p,labels=list(PUBLIC),average="weighted",zero_division=0)),"accuracy":float(accuracy_score(y,p)),"keep_rate":float((a=="KEEP").mean()),"residual_rate":float((a=="RESIDUAL").mean()),"reason_rate":float((a=="REASON").mean())}

def identity_audit():
    tr=set(pd.read_csv(RUN/"splits/train_ids.csv").original_iemocap_id.astype(str));de=set(pd.read_csv(RUN/"splits/dev_ids.csv").original_iemocap_id.astype(str));te=set(pd.read_csv(RUN/"splits/test_ids.csv").original_iemocap_id.astype(str));
    if (tr|de)&te:raise RuntimeError("training test overlap")
    pd.DataFrame([{"model":"Full_reconstructed","fit_identities":5163,"fit_split":"original Train only","test_identities":1623,"fit_test_intersection":0,"selection":"Session5 test_fscore per original TDTL protocol"},{"model":"D3_Residual","fit_identities":5810,"fit_split":"original Train+Dev","test_identities":1623,"fit_test_intersection":0,"selection":"frozen D3"},{"model":"V1_Utility","fit_identities":5810,"fit_split":"original Train+Dev","test_identities":1623,"fit_test_intersection":0,"selection":"frozen V1"},{"model":"V3_Pairwise","fit_identities":5810,"fit_split":"original Train+Dev with OOF V1 scores","test_identities":1623,"fit_test_intersection":0,"selection":"frozen V3 formulation; new weights"}]).to_csv(RUN/"results/benchmark_training_identity_audit.csv",index=False)

def fit():
    if hashlib.sha256(E8_SPEC.read_bytes()).hexdigest()!="20dcea76e3208ebcf601da694cf79e779c56ae0faa358ce9ae155eff10f9f19e":raise RuntimeError("spec sha mismatch")
    table=_r1_canonical();train=_load_split("train");dev=_load_split("dev");test=_load_split("test",labels=False);meta={s:_metadata(d["ids"]) for s,d in (("train",train),("dev",dev),("test",test))};r1={s:_r1(d,table) for s,d in (("train",train),("dev",dev),("test",test))}
    trres,trprob=_crossfit_d3(train,meta["train"].dialogue_id.astype(str).to_numpy());htrain=_fit_d3(train["labels"],train["views"][:,3],train["view_logits"][:,3],train["states"]);deres,deprob=_apply_d3(htrain,dev["view_logits"][:,3],dev["views"][:,3],dev["states"]);td=_assemble(train,trres,trprob,r1["train"],meta["train"]);dd=_assemble(dev,deres,deprob,r1["dev"],meta["dev"])
    v1tr=_fit_v1(td["numeric"],td["meta"].transcript_raw.tolist(),train["labels"],td["full_pred"],td["residual_pred"],td["r1"]); devscore=_score_v1(v1tr,dd["numeric"],dd["meta"].transcript_raw.tolist());deva=_v1_action(devscore);ooftr=_crossfit_v1(td["numeric"],td["meta"].transcript_raw.tolist(),train["labels"],td["full_pred"],td["residual_pred"],td["r1"],td["groups"]);v3tr=_fit_v3(ooftr,td["full_pred"],td["residual_pred"],td["r1"],train["labels"]);devv3,_,_=_score_v3(v3tr,devscore,dd["full_pred"],dd["residual_pred"]);pd.DataFrame([_devrow("D3_Residual",dev["labels"],np.full(len(dev["ids"]),"RESIDUAL",object),dd),_devrow("V1_Utility",dev["labels"],deva,dd),_devrow("V3_Pairwise",dev["labels"],devv3,dd)]).to_csv(RUN/"results/dev_diagnostics.csv",index=False)
    comb={k:np.concatenate((train[k],dev[k])) for k in ("ids","view_logits","views","states","labels")};cm=pd.concat((meta["train"],meta["dev"]),ignore_index=True);cr=np.concatenate((r1["train"],r1["dev"]));groups=cm.dialogue_id.astype(str).to_numpy();cres,cprob=_crossfit_d3(comb,groups);dh=_fit_d3(comb["labels"],comb["views"][:,3],comb["view_logits"][:,3],comb["states"]);teres,teprob=_apply_d3(dh,test["view_logits"][:,3],test["views"][:,3],test["states"]);cd=_assemble(comb,cres,cprob,cr,cm);ted=_assemble(test,teres,teprob,r1["test"],meta["test"]);oof=_crossfit_v1(cd["numeric"],cd["meta"].transcript_raw.tolist(),comb["labels"],cd["full_pred"],cd["residual_pred"],cd["r1"],groups);v1=_fit_v1(cd["numeric"],cd["meta"].transcript_raw.tolist(),comb["labels"],cd["full_pred"],cd["residual_pred"],cd["r1"]);tscore=_score_v1(v1,ted["numeric"],ted["meta"].transcript_raw.tolist());ta=_v1_action(tscore);v3=_fit_v3(oof,cd["full_pred"],cd["residual_pred"],cd["r1"],comb["labels"]);t3,sumscore,pairs=_score_v3(v3,tscore,ted["full_pred"],ted["residual_pred"])
    joblib.dump(dh,RUN/"models/d3_benchmark_aligned.joblib");joblib.dump(v1,RUN/"models/v1_utility_benchmark_aligned.joblib");joblib.dump(v3,RUN/"models/v3_pairwise_benchmark_aligned.joblib");pd.DataFrame({"canonical_utterance_id":test["ids"],"full_prediction":ted["full_pred"],"residual_prediction":ted["residual_pred"],"reason_prediction":ted["r1"],"v1_action_pre_reason":ta,"v3_action_pre_reason":t3,"v1_score_keep":tscore[:,0],"v1_score_residual":tscore[:,1],"v1_score_reason":tscore[:,2],"v3_score_keep":sumscore[:,0],"v3_score_residual":sumscore[:,1],"v3_score_reason":sumscore[:,2],"p_residual_over_keep":pairs[:,0],"p_reason_over_keep":pairs[:,1],"p_reason_over_residual":pairs[:,2]}).to_csv(RUN/"cache/session5_pre_reason_routes.csv",index=False)
    identity_audit();(RUN/"models/controller_freeze.json").write_text(json.dumps({"v1":{"C":1.0,"reason_penalty":.02,"tfidf":"word1-2,min_df2,max3000,sublinear"},"v3":{"spec_sha256":"20dcea76e3208ebcf601da694cf79e779c56ae0faa358ce9ae155eff10f9f19e","pairwise":"C1 balanced liblinear; difference-only targets; sum; ties K>R>S"},"d3":{"pca":32,"ridge_alpha":75,"alpha":.15,"cap":3},"r1_label_alignment":"E1/E1B/E2 stored R1 numeric IDs are already public ClarifyMER order [hap,sad,neu,ang,exc,fru]; no conversion","session5_outcome_evaluated":False},indent=2)+"\n");(RUN/"metadata/fit_status.json").write_text(json.dumps({"fit_complete":True,"r1_new_calls":0,"r1_numeric_ids_already_public":True,"test_labels_used_for_fit":False,"session5_outcome_evaluated":False,"v3_e8_weights_loaded":False},indent=2)+"\n")

def _boot(y,a,b,draws=2000):
    rng=np.random.default_rng(20260811);z=[]
    for _ in range(draws):
        q=rng.integers(0,len(y),len(y));z.append(f1_score(y[q],a[q],labels=list(PUBLIC),average="weighted",zero_division=0)-f1_score(y[q],b[q],labels=list(PUBLIC),average="weighted",zero_division=0))
    z=np.asarray(z);return {"observed_delta":float(f1_score(y,a,labels=list(PUBLIC),average="weighted",zero_division=0)-f1_score(y,b,labels=list(PUBLIC),average="weighted",zero_division=0)),"bootstrap_mean_delta":float(z.mean()),"ci_low":float(np.quantile(z,.025)),"ci_high":float(np.quantile(z,.975)),"replicates":draws}

def _mech(name,a,y,f,r,s,views):
    rok=r==y;sok=s==y;fok=f==y;rec=(~rok)&sok;r1=(~fok)&(~rok)&sok;danger=rok&(~sok);deep=(views.argmax(2)!=y[:,None]).all(1)&(~rok)
    return {"method":name,"keep_rate":float((a=="KEEP").mean()),"residual_rate":float((a=="RESIDUAL").mean()),"reason_rate":float((a=="REASON").mean()),"residual_wrong_r1_correct_total":int(rec.sum()),"residual_wrong_r1_correct_captured":int((rec&(a=="REASON")).sum()),"residual_wrong_r1_correct_capture_rate":float((rec&(a=="REASON")).sum()/max(1,rec.sum())),"r1_only_total":int(r1.sum()),"r1_only_captured":int((r1&(a=="REASON")).sum()),"r1_only_capture_rate":float((r1&(a=="REASON")).sum()/max(1,r1.sum())),"danger_total":int(danger.sum()),"danger_damage":int((danger&(a=="REASON")).sum()),"danger_protection_rate":float((danger&(a!="REASON")).sum()/max(1,danger.sum())),"deep_hard_total":int(deep.sum()),"deep_hard_r1_recoverable":int((deep&sok).sum()),"deep_hard_repaired":int((deep&sok&(a=="REASON")).sum())}

def evaluate():
    st=json.loads((RUN/"metadata/fit_status.json").read_text());
    if not st.get("fit_complete") or st.get("session5_outcome_evaluated"):raise RuntimeError("test evaluation not permitted")
    te=_load_split("test");route=pd.read_csv(RUN/"cache/session5_pre_reason_routes.csv");
    if route.canonical_utterance_id.tolist()!=te["ids"].tolist() or len(route)!=1623:raise RuntimeError("route mismatch")
    y=te["labels"];f=route.full_prediction.to_numpy(int);r=route.residual_prediction.to_numpy(int);s=route.reason_prediction.to_numpy(int);a1=route.v1_action_pre_reason.to_numpy(object);a3=route.v3_action_pre_reason.to_numpy(object);v1=materialize(a1,f,r,s);v3=materialize(a3,f,r,s);hist=float(pd.read_csv(RUN/"results/historical_full_replay.csv").weighted_f1.iloc[0]);methods=[("Historical TDTL Full",None,"legacy_test_selected_reference",hist),("Reconstructed Full",f,"legacy_test_selected_benchmark_reconstruction",None),("D3 Residual",r,"benchmark_aligned",None),("V1 Utility Router",v1,"benchmark_aligned",None),("V3 Pairwise Router",v3,"benchmark_aligned_posthoc_method_evaluation",None),("Always Reason",s,"diagnostic",None)];rows=[]
    for n,p,status,value in methods:rows.append({"method":n,"test_utterances":1623,"weighted_f1":value if value is not None else float(f1_score(y,p,labels=list(PUBLIC),average="weighted",zero_division=0)),"accuracy":None if p is None else float(accuracy_score(y,p)),"status":status})
    table=pd.DataFrame(rows);table.to_csv(RUN/"results/original_session5_benchmark_table.csv",index=False);pd.DataFrame([{"comparison":"V3_minus_D3",**_boot(y,v3,r)},{"comparison":"V3_minus_V1",**_boot(y,v3,v1)}]).to_csv(RUN/"results/bootstrap_summary.csv",index=False);pd.DataFrame([_mech("V1 Utility Router",a1,y,f,r,s,te["views"]),_mech("V3 Pairwise Router",a3,y,f,r,s,te["views"])]).to_csv(RUN/"results/mechanism_summary.csv",index=False);pd.DataFrame({"canonical_utterance_id":te["ids"],"gold_label_evaluation_only":y,"full_prediction":f,"residual_prediction":r,"reason_prediction":s,"v1_action":a1,"v1_prediction":v1,"v3_action":a3,"v3_prediction":v3}).to_csv(RUN/"results/canonical_predictions.csv",index=False);m={x.method:x.weighted_f1 for x in table.itertuples(index=False)};pd.DataFrame([{"comparison":"V3_minus_historical_full","delta":m["V3 Pairwise Router"]-hist},{"comparison":"V3_minus_reconstructed_full","delta":m["V3 Pairwise Router"]-m["Reconstructed Full"]},{"comparison":"V3_minus_D3","delta":m["V3 Pairwise Router"]-m["D3 Residual"]},{"comparison":"V3_minus_V1","delta":m["V3 Pairwise Router"]-m["V1 Utility Router"]}]).to_csv(RUN/"results/v1_v3_comparison.csv",index=False);two=np.where((f!=y)&(r==y),r,f);three=np.where((f!=y)&(r==y),r,np.where((f!=y)&(r!=y)&(s==y),s,f));pd.DataFrame([{"policy":"KEEP_RESIDUAL_oracle","weighted_f1":float(f1_score(y,two,labels=list(PUBLIC),average="weighted",zero_division=0))},{"policy":"KEEP_RESIDUAL_REASON_oracle","weighted_f1":float(f1_score(y,three,labels=list(PUBLIC),average="weighted",zero_division=0))}]).to_csv(RUN/"results/oracle_diagnostic.csv",index=False);st["session5_outcome_evaluated"]=True;st["session5_evaluation_count"]=1;(RUN/"metadata/fit_status.json").write_text(json.dumps(st,indent=2)+"\n")
    print(json.dumps({"EVALUATE_COMPLETE":True,"V3_WF1":m["V3 Pairwise Router"],"V1_WF1":m["V1 Utility Router"],"D3_WF1":m["D3 Residual"]},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("fit","evaluate"));a=p.parse_args();fit() if a.stage=="fit" else evaluate()
if __name__=="__main__":main()
