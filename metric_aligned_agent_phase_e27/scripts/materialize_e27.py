"""Unlabelled E27 candidate materialization. Gold is rejected from candidate code."""
import csv, json, math, re
from pathlib import Path
import numpy as np
from .candidate_core import action_for, candidate_id, proposal_for, proposal_score, stable_prefix_mask, weighted_vote
from .io_atomic import atomic_write_csv, atomic_write_json, read_csv_rows, sha256_file

RUN = Path(__file__).resolve().parents[1]
E26 = RUN.parents[2] / "experiment_outputs" / "metric_aligned_agent_phase_e26" / "run_20260820T113331_CST_quota_recovered_sol_xhigh_v1"
N = 1623
FORBIDDEN = {"gold_label_evaluation_only", "gold", "benefit", "harm", "weighted_f1", "test_metric", "oracle"}

def validate_unlabeled_rows(rows):
    if not rows: raise ValueError("empty source rows")
    for row in rows:
        for key in row:
            if key.lower() in FORBIDDEN or any(token in key.lower() for token in ("gold", "outcome", "correctness")):
                raise ValueError(f"forbidden evaluation column: {key}")
        cid = row.get("canonical_id")
        if not cid: raise ValueError("missing canonical_id")
    if len({row["canonical_id"] for row in rows}) != len(rows): raise ValueError("duplicate canonical_id")
    return True

def sanitize_source_rows(rows):
    clean = []
    for row in rows:
        clean.append({k: v for k, v in row.items() if k not in FORBIDDEN and not any(token in k.lower() for token in ("gold", "outcome", "correctness"))})
    validate_unlabeled_rows(clean)
    return clean

def _f(row, name): return float(row[f"{name.lower()}_confidence"])
def _u(row, name): return float(row[f"{name.lower()}_uncertainty"])
def _l(row, name): return int(row[f"{name.lower()}_reason_prediction"])
def _accept(row, name): return row[f"{name.lower()}_decision"] == "ACCEPT_REASON"

def _base_label(row, base): return int(row["full_prediction"] if base == "FULL" else row["residual_prediction"])
def _action_arrays(rows, pred):
    full = np.asarray([int(r["full_prediction"]) for r in rows]); d3 = np.asarray([int(r["residual_prediction"]) for r in rows]); pred = np.asarray(pred, dtype=np.int8)
    actions = np.asarray([action_for(int(p), int(f), int(d)) for p, f, d in zip(pred, full, d3)], dtype=np.int8)
    return pred, actions

def _spec(parts, family, pred):
    config = dict(parts); config["family"] = family
    return {"candidate_id": candidate_id(config), "family": family, "config": config, "pred": np.asarray(pred, dtype=np.int8)}

def _prefix(rows, base, label, score, family, config_base):
    base_labels = np.asarray([_base_label(r, base) for r in rows], dtype=np.int8)
    eligible = np.asarray([int(label[i]) != int(base_labels[i]) for i in range(len(rows))])
    indices = np.flatnonzero(eligible)
    order = sorted(indices.tolist(), key=lambda i: (-float(score[i]), rows[i]["canonical_id"]))
    cap = min(math.ceil(0.2 * len(rows)), len(order))
    for k in range(cap + 1):
        pred = base_labels.copy(); pred[np.asarray(order[:k], dtype=int)] = np.asarray([label[i] for i in order[:k]], dtype=np.int8)
        yield _spec({**config_base, "base": base, "k": k, "eligible_count": len(order), "ranking_score": config_base.get("ranking_score", "")}, family, pred)

def _e26_specs(rows):
    specs=[]; by = {r["canonical_id"]: r for r in rows}
    e26_rows = read_csv_rows(E26 / "evaluation/results/e26_candidate_leaderboard.csv")
    ids = [r["canonical_id"] for r in rows]
    for er in e26_rows:
        name = er["candidate"]
        if name == "historical_full":
            pred = np.asarray([int(r["historical_full_prediction"]) for r in rows], dtype=np.int8)
        elif name == "reconstructed_full": pred = np.asarray([int(r["full_prediction"]) for r in rows], dtype=np.int8)
        elif name == "d3_residual": pred = np.asarray([int(r["residual_prediction"]) for r in rows], dtype=np.int8)
        elif name.startswith("reason_only_"):
            stage=name.split("_")[-1].upper(); pred=np.asarray([_l(r,stage) for r in rows], dtype=np.int8)
        elif name.endswith("_direct"):
            stage=name.split("_")[0].upper(); pred=np.asarray([_l(r,stage) if _accept(r,stage) else int(r["full_prediction"]) for r in rows], dtype=np.int8)
        elif "_confidence_" in name:
            stage, t = name.split("_confidence_"); stage=stage.upper(); threshold=float(t); pred=np.asarray([_l(r,stage) if _accept(r,stage) and _f(r,stage)>=threshold else int(r["full_prediction"]) for r in rows], dtype=np.int8)
        elif "_uncertainty_" in name:
            stage, t = name.split("_uncertainty_"); stage=stage.upper(); threshold=float(t); pred=np.asarray([_l(r,stage) if _accept(r,stage) and _u(r,stage)<=threshold else int(r["full_prediction"]) for r in rows], dtype=np.int8)
        elif "_budget_" in name:
            stage, b = name.split("_budget_"); stage=stage.upper(); b=float(b); eligible=[i for i,r in enumerate(rows) if _accept(r,stage)]
            score=np.asarray([_f(rows[i],stage)-_u(rows[i],stage) for i in eligible], dtype=float)
            k=int(math.ceil(b*len(rows)))
            chosen=set(np.asarray(eligible, dtype=int)[np.argsort(score)[::-1][:k]].tolist())
            pred=np.asarray([_l(r,stage) if i in chosen else int(r["full_prediction"]) for i,r in enumerate(rows)], dtype=np.int8)
        elif name.endswith("_fallback"):
            first, second = name.split("_")[:2]; first=first.upper(); second=second.upper(); pred=np.asarray([_l(r,first) if _accept(r,first) else (_l(r,second) if _accept(r,second) else int(r["full_prediction"])) for r in rows], dtype=np.int8)
        else: continue
        specs.append(_spec({"source": "E26", "candidate": name}, "F0_E26_REPLAY", pred))
    return specs

def enumerate_specs(rows):
    for spec in _e26_specs(rows):
        yield spec
    n=len(rows); cap=math.ceil(.2*n)
    # F1: single interface prefix sweeps.
    for base in ("FULL", "D3"):
        for stage in ("I1","I2","I3"):
            for score_name in ("confidence_minus_uncertainty","confidence","negative_uncertainty"):
                score=[_f(r,stage)-_u(r,stage) if score_name=="confidence_minus_uncertainty" else _f(r,stage) if score_name=="confidence" else -_u(r,stage) for r in rows]
                label=[_l(r,stage) if _accept(r,stage) else _base_label(r,base) for r in rows]
                yield from _prefix(rows,base,label,score,"F1_SINGLE_PREFIX",{"interface":stage,"ranking_score":score_name})
    # F2: proposal prefix sweeps.
    proposal_sources=["majority_2of3","unanimous_3of3","i3_i1","i3_i2","i3_any_confirm","best_supported_label"]
    score_modes=["mean_confidence","min_confidence","mean_margin","max_margin","support_bonus_0.10","support_bonus_0.25","support_bonus_0.50"]
    for base in ("FULL","D3"):
        for source in proposal_sources:
            props=[proposal_for(source,r) for r in rows]
            for score_mode in score_modes:
                label=[p[0] if p[0] is not None else _base_label(r,base) for r,p in zip(rows,props)]
                scores=[proposal_score(score_mode,p[1],r) if p[0] is not None else float("-inf") for r,p in zip(rows,props)]
                yield from _prefix(rows,base,label,scores,"F2_CONSENSUS_PREFIX",{"proposal_source":source,"ranking_score":score_mode})
    # F3: deterministic weighted votes.
    tie=["FULL","D3","I3","I2","I1"]
    for fw in (1,2,3,4):
        for dw in (0,1,2):
            for mode in ("unit","confidence","positive_margin"):
                for eligibility in ("accept_only","all_outputs"):
                    pred=[]
                    for r in rows:
                        labels={"FULL":int(r["full_prediction"]),"D3":int(r["residual_prediction"]),"I1":_l(r,"I1"),"I2":_l(r,"I2"),"I3":_l(r,"I3")}; weights={"FULL":float(fw),"D3":float(dw)}
                        for stage in ("I1","I2","I3"):
                            weights[stage]=0.0 if eligibility=="accept_only" and not _accept(r,stage) else 1.0 if mode=="unit" else _f(r,stage) if mode=="confidence" else max(_f(r,stage)-_u(r,stage),0.0)
                        pred.append(weighted_vote(labels,weights,tie))
                    yield _spec({"full_weight":fw,"d3_weight":dw,"reason_weight_mode":mode,"reason_eligibility":eligibility},"F3_WEIGHTED_VOTE",pred)
    # F4 deterministic cascades and their fixed prefix scores.
    def cascade(row, family, base):
        base_label=_base_label(row,base); l3,l1,l2=_l(row,"I3"),_l(row,"I1"),_l(row,"I2")
        if family=="i3_any_confirm": return l3 if _accept(row,"I3") and (l3==l1 or l3==l2) else base_label, (_f(row,"I3")-_u(row,"I3") if _accept(row,"I3") else float("-inf"))
        if family=="i3_all_confirm": return l3 if _accept(row,"I3") and l3==l1==l2 else base_label, (_f(row,"I3")-_u(row,"I3") if _accept(row,"I3") else float("-inf"))
        if family=="i3_veto_cheap": return l3 if _accept(row,"I3") and l3!=base_label and l3 not in (l1,l2) else base_label, (_f(row,"I3")-_u(row,"I3") if _accept(row,"I3") else float("-inf"))
        if family in ("i3_i2_i1","i3_i1_i2"):
            order=("I3","I2","I1") if family=="i3_i2_i1" else ("I3","I1","I2")
            for stage in order:
                if _accept(row,stage) and _l(row,stage)!=base_label: return _l(row,stage), _f(row,stage)-_u(row,stage)
            return base_label,float("-inf")
        prop,support=proposal_for("majority_2of3",row)
        if prop is not None: return prop, proposal_score("mean_margin",support,row)
        return base_label,float("-inf")
    for family in ("i3_any_confirm","i3_all_confirm","i3_veto_cheap","i3_i2_i1","i3_i1_i2","majority_fallback_full","majority_fallback_d3"):
        for base in ("FULL","D3"):
            if family.startswith("majority_fallback_") and ((family.endswith("full") and base!="FULL") or (family.endswith("d3") and base!="D3")): continue
            vals=[cascade(r,family,base) for r in rows]; labels=[x[0] for x in vals]; scores=[x[1] for x in vals]
            yield _spec({"base":base,"cascade":family,"threshold_mode":"none"},"F4_CASCADE",labels)
            yield from _prefix(rows,base,labels,scores,"F4_CASCADE_PREFIX",{"cascade":family,"threshold_mode":"prefix","ranking_score":"fixed_proposal_score"})

def main():
    materialized=RUN/"materialized"
    if any((materialized/name).exists() for name in ("candidate_manifest.csv","candidate_predictions.npz","candidate_actions.npz","MATERIALIZATION_SHA256.json")):
        raise RuntimeError("materialization already exists; refusing overwrite")
    source=read_csv_rows(E26/"evaluation/results/e26_canonical_predictions.csv")
    rows=sanitize_source_rows(source)
    allowed=json.loads((RUN/"protocol/E27_INPUT_SCHEMA.json").read_text())["allowed_fields"]
    if set(rows[0]) != set(allowed): raise ValueError("source schema differs from locked E27_INPUT_SCHEMA")
    specs=list(enumerate_specs(rows)); specs=list({x["candidate_id"]:x for x in specs}.values())
    if not specs or len(specs)>40000: raise RuntimeError(f"candidate count outside bound: {len(specs)}")
    ids=[r["canonical_id"] for r in rows]; pred=np.vstack([s["pred"] for s in specs]).astype(np.int8)
    actions=[]
    for s in specs:
        full=np.asarray([int(r["full_prediction"]) for r in rows]); d3=np.asarray([int(r["residual_prediction"]) for r in rows])
        action_codes={"KEEP":0,"RESIDUAL":1,"REASON":2}
        actions.append(np.asarray([action_codes[action_for(int(p),int(f),int(d))] for p,f,d in zip(s["pred"],full,d3)],dtype=np.int8))
    actions=np.vstack(actions).astype(np.int8); materialized.mkdir(parents=True,exist_ok=True)
    atomic_write_json(materialized/"canonical_ids.json",ids)
    np.savez_compressed(materialized/"candidate_predictions.npz",predictions=pred,candidate_ids=np.asarray([s["candidate_id"] for s in specs],dtype="U128"))
    np.savez_compressed(materialized/"candidate_actions.npz",actions=actions,candidate_ids=np.asarray([s["candidate_id"] for s in specs],dtype="U128"))
    fields=["candidate_id","family","config_json"]
    atomic_write_csv(materialized/"candidate_manifest.csv",[{"candidate_id":s["candidate_id"],"family":s["family"],"config_json":json.dumps(s["config"],sort_keys=True,separators=(",",":"))} for s in specs],fields)
    hashed={str(p.relative_to(RUN)):sha256_file(p) for p in (materialized/"canonical_ids.json",materialized/"candidate_predictions.npz",materialized/"candidate_actions.npz",materialized/"candidate_manifest.csv",RUN/"protocol/E27_INPUT_SCHEMA.json",RUN/"protocol/E27_SEARCH_SPACE.json")}
    atomic_write_json(materialized/"MATERIALIZATION_SHA256.json",{"files":hashed,"candidate_count":len(specs),"canonical_count":len(rows),"gold_in_materialization":False,"frozen":True})
    print(json.dumps({"candidate_count":len(specs),"canonical_count":len(rows),"gold_in_materialization":False},sort_keys=True))

if __name__=="__main__": main()
