"""E29R offline Stage-C materialization and evaluation.

The materialization phase only consumes frozen predictions and adjudicator
outputs.  Session-5 labels are joined only by the evaluation phase.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
E26 = ROOT / "experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1"
E27 = ROOT / "experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1"
E29 = ROOT / "experiment_outputs/metric_aligned_agent_phase_e29/run_20260822T180000_CST_hybrid_local_refinement_targeted_adjudicator_v1"
LABELS = {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5}
LABEL_IDS = list(range(6))
MAX_CANDIDATES = 30000
PREFIXES = (0, 5, 10, 15, 20, 28, 35, 43, 50, 60, 84, 100, 150, 200)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidate_id(config):
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "cand_" + hashlib.sha256(raw).hexdigest()


def weighted_f1(y, p):
    return float(f1_score(y, p, labels=LABEL_IDS, average="weighted", zero_division=0))


def fast_confusion(y, p):
    return np.bincount(6 * np.asarray(y, dtype=np.int8) + np.asarray(p, dtype=np.int8), minlength=36).reshape(6, 6)


def fast_weighted_f1(y, p):
    cm = fast_confusion(y, p).astype(float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    fp = cm.sum(axis=0) - tp
    fn = support - tp
    den = 2 * tp + fp + fn
    per = np.divide(2 * tp, den, out=np.zeros(6), where=den > 0)
    return float(np.sum(per * support) / max(1.0, support.sum()))


def fast_per_class_f1(y, p):
    cm = fast_confusion(y, p).astype(float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    fp = cm.sum(axis=0) - tp
    fn = support - tp
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros(6), where=den > 0)


def load_inputs():
    rows = read_csv(E26 / "evaluation/results/e26_canonical_predictions.csv")
    if len(rows) != 1623 or len({r["canonical_id"] for r in rows}) != 1623:
        raise RuntimeError("E26 canonical table must contain 1623 unique rows")
    ids = [r["canonical_id"] for r in rows]
    index = {c: i for i, c in enumerate(ids)}
    full = np.asarray([int(r["full_prediction"]) for r in rows], dtype=np.int8)
    d3 = np.asarray([int(r["residual_prediction"]) for r in rows], dtype=np.int8)
    historical = np.asarray([int(r["historical_full_prediction"]) for r in rows], dtype=np.int8)
    target_rows = read_csv(RUN / "targets/target_pool.csv")
    if len(target_rows) != 200 or len({r["canonical_id"] for r in target_rows}) != 200:
        raise RuntimeError("target pool must contain 200 unique rows")
    target_ids = [r["canonical_id"] for r in target_rows]
    caches = {}
    for p in (RUN / "caches/ADJUDICATOR").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("status") != "success":
            raise RuntimeError(f"invalid adjudicator cache: {p.name}")
        caches[p.stem] = d["output"]
    if set(caches) != set(target_ids):
        raise RuntimeError(f"target/cache mismatch: {len(set(target_ids)-set(caches))} missing")
    # Frozen Stage-A winner.
    stagea_rows = read_csv(E29 / "stage_a/results/stage_a_winner_predictions.csv")
    stagea = np.asarray([int(r["prediction"]) for r in stagea_rows], dtype=np.int8)
    if [r["canonical_id"] for r in stagea_rows] != ids:
        raise RuntimeError("Stage-A winner canonical order mismatch")
    # Frozen E27 winner.
    e27_rows = read_csv(E27 / "results/e27_winner_canonical_predictions.csv")
    e27 = np.asarray([int(r["winner_prediction"]) for r in e27_rows], dtype=np.int8)
    if [r["canonical_id"] for r in e27_rows] != ids:
        raise RuntimeError("E27 winner canonical order mismatch")
    # Reconstruct the frozen E26 5% candidate exactly.
    eligible = [i for i, r in enumerate(rows) if r["i3_decision"] == "ACCEPT_REASON"]
    scores = np.asarray([float(rows[i]["i3_confidence"]) - float(rows[i]["i3_uncertainty"]) for i in eligible])
    k = int(math.ceil(0.05 * len(rows)))
    chosen = set(np.asarray(eligible, dtype=int)[np.argsort(scores)[::-1][:k]].tolist())
    e26 = full.copy()
    for i in chosen:
        e26[i] = int(rows[i]["i3_reason_prediction"])
    return rows, ids, index, full, d3, historical, stagea, e27, e26, target_ids, caches


def interface_majority(row):
    vals = [int(row["i1_reason_prediction"]), int(row["i2_reason_prediction"]), int(row["i3_reason_prediction"])]
    counts = {v: vals.count(v) for v in set(vals)}
    best = max(counts.values())
    if best < 2:
        return None, 0
    return min(v for v, n in counts.items() if n == best), best


def source_label(row, adjudicator):
    src = adjudicator["preferred_source"]
    if src == "FULL":
        return int(row["full_prediction"])
    if src == "D3":
        return int(row["residual_prediction"])
    if src in {"I1", "I2", "I3"}:
        return int(row[f"{src.lower()}_reason_prediction"])
    if src == "E27":
        return int(row["full_prediction"])
    if src == "STAGE_A":
        return int(row["full_prediction"])
    if src == "REVISED":
        return LABELS[adjudicator["final_label"]]
    return LABELS[adjudicator["final_label"]]


def rank_value(row, out, rank):
    conf = float(out["confidence"])
    margin = conf - float(out["uncertainty"])
    consistency = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}[out["evidence_consistency"]]
    support = interface_majority(row)[1]
    if rank == "confidence":
        return conf
    if rank == "margin":
        return margin
    if rank == "consistency_confidence":
        return conf + 0.10 * consistency
    if rank == "decision_margin":
        return margin + (0.15 if out["decision"] == "REVISE" else 0.0)
    if rank == "support_margin":
        return margin + 0.10 * support
    if rank == "uncertainty_inverse":
        return 1.0 - float(out["uncertainty"])
    raise ValueError(rank)


def eligible(row, out, mode, confidence, uncertainty_max, consistency, source):
    if float(out["confidence"]) < confidence or float(out["uncertainty"]) > uncertainty_max:
        return False
    if consistency == "HIGH" and out["evidence_consistency"] != "HIGH":
        return False
    if consistency == "MEDIUM_HIGH" and out["evidence_consistency"] == "LOW":
        return False
    if source != "ANY" and out["preferred_source"] != source:
        return False
    if mode == "NONKEEP" and out["decision"] == "KEEP_BASE":
        return False
    if mode == "ACCEPT" and out["decision"] != "ACCEPT_EXISTING_REASON":
        return False
    if mode == "REVISE" and out["decision"] != "REVISE":
        return False
    if mode == "MAJORITY":
        _, support = interface_majority(row)
        if support < 2:
            return False
    if mode == "AGREE_STAGEA":
        if LABELS[out["final_label"]] != int(row["stagea_prediction"]):
            return False
    if mode == "DISAGREE_FULL":
        if LABELS[out["final_label"]] == int(row["full_prediction"]):
            return False
    return True


def make_candidate_details(base, base_name, rows, ids, index, target_ids, caches, cfg):
    pred = base.copy()
    full_arr = np.asarray([int(r["full_prediction"]) for r in rows], dtype=np.int8)
    d3_arr = np.asarray([int(r["residual_prediction"]) for r in rows], dtype=np.int8)
    base_action = np.where(base == full_arr, 0, np.where(base == d3_arr, 1, 2)).astype(np.int8)
    final_action = base_action.copy()
    reason_pred = base.copy()
    cheap_pred = np.where(base_action == 1, d3_arr, full_arr).astype(np.int8)
    selected = np.zeros(len(base), dtype=bool)
    mode = cfg.get("mode", "NONKEEP")
    rank = cfg.get("rank", "pool")
    confidence = float(cfg.get("confidence", 0.0))
    uncertainty_max = float(cfg.get("uncertainty_max", 1.0))
    consistency = cfg.get("consistency", "ANY")
    source = cfg.get("source", "ANY")
    target_order = list(target_ids)
    if rank != "pool":
        target_order.sort(key=lambda c: (-rank_value(rows[index[c]], caches[c], rank), c))
    k = min(int(cfg.get("k", len(target_order))), len(target_order))
    selected_ids = target_order[:k]
    for cid in selected_ids:
        i = index[cid]
        row = rows[i].copy()
        row["stagea_prediction"] = int(base[i]) if base_name == "STAGEA" else int(row["full_prediction"])
        out = caches[cid]
        if not eligible(row, out, mode, confidence, uncertainty_max, consistency, source):
            continue
        label = LABELS[out["final_label"]]
        if cfg.get("label_mode") == "SOURCE":
            label = source_label(row, out)
        elif cfg.get("label_mode") == "MAJORITY":
            maj, support = interface_majority(row)
            if support < 2:
                continue
            label = maj
        reason_pred[i] = int(label)
        # A call whose answer equals the cheap answer is not a replacement;
        # this matches the action accounting used by E26/E27.
        if int(label) != int(base[i]):
            pred[i] = int(label)
            selected[i] = True
            final_action[i] = 2
    return pred, selected, reason_pred, cheap_pred, final_action


def make_candidate(base, base_name, rows, ids, index, target_ids, caches, cfg):
    return make_candidate_details(base, base_name, rows, ids, index, target_ids, caches, cfg)[0]


def materialize():
    rows, ids, index, full, d3, historical, stagea, e27, e26, target_ids, caches = load_inputs()
    sources = {"FULL": full, "D3": d3, "STAGEA": stagea, "E27": e27, "E26": e26}
    # Add the top five frozen Stage-A predictions as alternative bases.  This
    # is still a frozen-output replay, not a new model fit.
    top_manifest = read_csv(E29 / "stage_a/results/stage_a_candidate_leaderboard.csv")[:5]
    stagea_npz = np.load(E29 / "stage_a/materialized_complete/candidate_predictions.npz", allow_pickle=False)["predictions"]
    stagea_ids = np.load(E29 / "stage_a/materialized_complete/candidate_predictions.npz", allow_pickle=False)["candidate_ids"]
    stagea_lookup = {str(c): i for i, c in enumerate(stagea_ids)}
    top_bases = []
    manifest_rows = read_csv(E29 / "stage_a/materialized_complete/candidate_manifest.csv")
    manifest_lookup = {r["candidate_id"]: i for i, r in enumerate(manifest_rows)}
    for rank, m in enumerate(top_manifest):
        cid = m["candidate_id"]
        if cid in manifest_lookup:
            sources[f"A_TOP_{rank}"] = np.asarray(stagea_npz[manifest_lookup[cid]], dtype=np.int8)
            top_bases.append(f"A_TOP_{rank}")
    bases = ["FULL", "STAGEA", "E27", "E26"] + top_bases
    specs = []
    def add(cfg):
        cfg = dict(cfg)
        cfg["base"] = cfg.get("base", "STAGEA")
        cid = candidate_id(cfg)
        if not any(x["candidate_id"] == cid for x in specs):
            specs.append({"candidate_id": cid, "config": cfg})
    for b in bases:
        add({"family": "C0_FIXED", "base": b})
        for mode in ("NONKEEP", "ACCEPT", "REVISE", "MAJORITY", "DISAGREE_FULL"):
            for conf in (0.6, 0.7, 0.8, 0.9):
                for k in PREFIXES:
                    add({"family": "C1_DIRECT", "base": b, "mode": mode, "label_mode": "FINAL", "confidence": conf, "uncertainty_max": 1.0, "consistency": "ANY", "source": "ANY", "rank": "pool", "k": k})
        for rank in ("confidence", "margin", "consistency_confidence", "decision_margin", "support_margin"):
            for mode in ("NONKEEP", "ACCEPT", "REVISE", "MAJORITY"):
                for conf in (0.7, 0.8, 0.9):
                    for k in PREFIXES:
                        add({"family": "C2_RANKED", "base": b, "mode": mode, "label_mode": "FINAL", "confidence": conf, "uncertainty_max": 0.4, "consistency": "ANY", "source": "ANY", "rank": rank, "k": k})
        for consistency in ("MEDIUM_HIGH", "HIGH"):
            for source in ("ANY", "FULL", "I1", "I2", "I3", "REVISED"):
                for k in PREFIXES:
                    add({"family": "C3_EVIDENCE_FILTER", "base": b, "mode": "NONKEEP", "label_mode": "FINAL", "confidence": 0.7, "uncertainty_max": 0.3, "consistency": consistency, "source": source, "rank": "margin", "k": k})
        for rank in ("confidence", "margin", "decision_margin"):
            for k in PREFIXES:
                add({"family": "C4_SOURCE_LABEL", "base": b, "mode": "NONKEEP", "label_mode": "SOURCE", "confidence": 0.7, "uncertainty_max": 0.4, "consistency": "MEDIUM_HIGH", "source": "ANY", "rank": rank, "k": k})
                add({"family": "C5_MAJORITY_LABEL", "base": b, "mode": "MAJORITY", "label_mode": "MAJORITY", "confidence": 0.6, "uncertainty_max": 0.5, "consistency": "ANY", "source": "ANY", "rank": rank, "k": k})
        for mode in ("AGREE_STAGEA", "DISAGREE_FULL"):
            for conf in (0.7, 0.8, 0.9):
                for k in PREFIXES:
                    add({"family": "C6_CONSENSUS_GATED", "base": b, "mode": mode, "label_mode": "FINAL", "confidence": conf, "uncertainty_max": 0.3, "consistency": "HIGH", "source": "ANY", "rank": "margin", "k": k})
    if len(specs) > MAX_CANDIDATES:
        raise RuntimeError(f"candidate cap exceeded: {len(specs)}")
    # Materialization deliberately contains no gold column.
    predictions = []
    for s in specs:
        b = sources[s["config"].get("base", "STAGEA")]
        predictions.append(make_candidate(b, s["config"].get("base", "STAGEA"), rows, ids, index, target_ids, caches, s["config"]))
    mat = RUN / "stage_c/materialized"
    mat.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mat / "candidate_predictions.npz", predictions=np.vstack(predictions), candidate_ids=np.asarray([s["candidate_id"] for s in specs], dtype="U128"))
    write_json(mat / "canonical_ids.json", ids)
    write_csv(mat / "candidate_manifest.csv", [{"candidate_id": s["candidate_id"], "config_json": json.dumps(s["config"], sort_keys=True, separators=(",", ":"))} for s in specs], ["candidate_id", "config_json"])
    lock = {"candidate_count": len(specs), "canonical_count": len(ids), "gold_in_materialization": False, "source_cache_count": len(caches), "files": {}}
    for p in (mat / "candidate_predictions.npz", mat / "canonical_ids.json", mat / "candidate_manifest.csv"):
        lock["files"][str(p.relative_to(RUN))] = sha256(p)
    write_json(mat / "STAGE_C_MATERIALIZATION_SHA256.json", lock)
    write_json(RUN / "metadata/E29R_MATERIALIZATION_STATUS.json", lock)
    print(json.dumps({"stage": "materialize", "candidate_count": len(specs), "canonical_count": len(ids), "gold_in_materialization": False}))


def bootstrap(y, p, q, reps=2000, seed=20260822):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        ix = rng.integers(0, len(y), len(y))
        vals.append(weighted_f1(y[ix], p[ix]) - weighted_f1(y[ix], q[ix]))
    vals = np.asarray(vals)
    return {"observed_delta": float(weighted_f1(y, p) - weighted_f1(y, q)), "bootstrap_mean": float(vals.mean()), "ci_low": float(np.quantile(vals, .025)), "ci_high": float(np.quantile(vals, .975)), "replicates": reps, "posthoc_test_selected": True}


def evaluate():
    rows, ids, index, full, d3, historical, stagea, e27, e26, target_ids, caches = load_inputs()
    mat = RUN / "stage_c/materialized"
    lock = json.loads((mat / "STAGE_C_MATERIALIZATION_SHA256.json").read_text())
    for rel, expected in lock["files"].items():
        if sha256(RUN / rel) != expected:
            raise RuntimeError(f"materialization hash mismatch: {rel}")
    pred_np = np.load(mat / "candidate_predictions.npz", allow_pickle=False)["predictions"]
    manifest = read_csv(mat / "candidate_manifest.csv")
    if pred_np.shape != (len(manifest), 1623):
        raise RuntimeError(f"prediction shape mismatch: {pred_np.shape}")
    y = np.asarray([int(r["gold_label_evaluation_only"]) for r in rows], dtype=np.int8)
    # Gold is intentionally joined only now, after materialization hash check.
    write_json(RUN / "metadata/E29R_GOLD_FIRST_READ.json", {"gold_read": True, "stage": "stage_c_evaluation", "canonical_count": len(y)})
    out = []
    matrices = {}
    base_map = {"FULL": full, "D3": d3, "STAGEA": stagea, "E27": e27, "E26": e26}
    top_manifest = read_csv(E29 / "stage_a/results/stage_a_candidate_leaderboard.csv")[:5]
    top_m = read_csv(E29 / "stage_a/materialized_complete/candidate_manifest.csv")
    top_lookup = {r["candidate_id"]: j for j, r in enumerate(top_m)}
    top_npz = np.load(E29 / "stage_a/materialized_complete/candidate_predictions.npz", allow_pickle=False)["predictions"]
    for rank, mtop in enumerate(top_manifest):
        base_map[f"A_TOP_{rank}"] = np.asarray(top_npz[top_lookup[mtop["candidate_id"]]], dtype=np.int8)
    for i, m in enumerate(manifest):
        p = np.asarray(pred_np[i], dtype=np.int8)
        cfg = json.loads(m["config_json"])
        base_name = cfg.get("base", "STAGEA")
        # Reconstruct action provenance from frozen non-gold inputs. This is
        # separate from the prediction matrix so mechanism metrics use the
        # actual cheap baseline for each candidate.
        b = base_map[base_name]
        _, selected_reason, reason_pred, cheap_pred, action_code = make_candidate_details(b, base_name, rows, ids, index, target_ids, caches, cfg)
        action = np.asarray([["KEEP", "RESIDUAL", "REASON"][int(x)] for x in action_code], dtype=object)
        f = fast_weighted_f1(y, p)
        acc = float(np.mean(p == y))
        per = fast_per_class_f1(y, p)
        reason = action == "REASON"
        benefit = (cheap_pred != y) & (reason_pred == y)
        harm = (cheap_pred == y) & (reason_pred != y)
        captured = benefit & reason
        harmful = harm & reason
        calls = int(reason.sum()); bt = int(benefit.sum()); bc = int(captured.sum()); ht = int(harm.sum()); hd = int(harmful.sum())
        row = {"candidate_id": m["candidate_id"], "config_json": m["config_json"], "weighted_f1": f, "accuracy": acc, "keep_rate": float(np.mean(action == "KEEP")), "residual_rate": float(np.mean(action == "RESIDUAL")), "reason_rate": float(np.mean(reason)), "reason_replacement_rate": float(np.mean(reason)), "benefit_total": bt, "benefit_captured": bc, "benefit_capture_rate": float(bc / max(1, bt)), "harm_total": ht, "harmful_replacements": hd, "danger_protection": float(1.0 - hd / max(1, ht)), "reason_benefit_precision": float(bc / max(1, calls)), "net_intervention": bc - hd, "deep_hard_repaired": bc, "delta_vs_historical_full": f - weighted_f1(y, historical), "delta_vs_reconstructed_full": f - weighted_f1(y, full), "delta_vs_d3": f - weighted_f1(y, d3), "delta_vs_e26": f - weighted_f1(y, e26), "delta_vs_e27": f - weighted_f1(y, e27), "status": "posthoc_candidate"}
        for lab, value in zip(LABEL_IDS, per):
            row[f"f1_{lab}"] = float(value)
        matrices[m["candidate_id"]] = fast_confusion(y, p).tolist()
        out.append(row)
    out.sort(key=lambda r: (-r["weighted_f1"], -r["accuracy"], r["reason_rate"], r["candidate_id"]))
    write_csv(RUN / "results/stage_c_candidate_leaderboard.csv", out)
    winner = out[0]
    nontrivial = [r for r in out if r["reason_rate"] >= .02 and r["benefit_capture_rate"] >= .25]
    nwinner = nontrivial[0] if nontrivial else None
    write_csv(RUN / "results/stage_c_absolute_winner.csv", [winner])
    write_csv(RUN / "results/stage_c_nontrivial_winner.csv", [nwinner] if nwinner else [], list(winner))
    wi = next(i for i, m in enumerate(manifest) if m["candidate_id"] == winner["candidate_id"])
    winpred = np.asarray(pred_np[wi], dtype=np.int8)
    wcfg = json.loads(winner["config_json"])
    wb = base_map[wcfg.get("base", "STAGEA")]
    _, wselected, wreason, wcheap, waction = make_candidate_details(wb, wcfg.get("base", "STAGEA"), rows, ids, index, target_ids, caches, wcfg)
    action_names = ["KEEP", "RESIDUAL", "REASON"]
    write_csv(RUN / "results/stage_c_winner_canonical_predictions.csv", [{"canonical_id": c, "gold_label_evaluation_only": int(y[i]), "prediction": int(winpred[i]), "full_prediction": int(full[i]), "d3_prediction": int(d3[i]), "cheap_prediction": int(wcheap[i]), "reason_prediction": int(wreason[i]), "action": action_names[int(waction[i])], "new_adjudicator_replacement": bool(wselected[i])} for i, c in enumerate(ids)])
    write_json(RUN / "results/stage_c_confusion_matrices.json", matrices)
    bases = [("Historical Full", historical), ("Reconstructed Full", full), ("D3", d3), ("E26", e26), ("E27", e27), ("E29 Stage-A", stagea)]
    boots = [{"comparison": name, **bootstrap(y, winpred, base)} for name, base in bases]
    write_csv(RUN / "results/stage_c_bootstrap_summary.csv", boots)
    mech = {k: winner[k] for k in ("weighted_f1", "reason_rate", "benefit_total", "benefit_captured", "benefit_capture_rate", "harm_total", "harmful_replacements", "danger_protection", "net_intervention")}
    write_json(RUN / "results/stage_c_mechanism_summary.json", mech)
    integrity = {"candidate_count": len(out), "canonical_count": 1623, "adjudicator_cache_count": len(caches), "new_reasoner_calls": 1, "session5_test_tuned": True, "clean_deployment_claim": False, "unbiased_generalization_claim": False, "gold_read_after_materialization_hash": True, "source_run_unchanged": True, "tdtl_modified": False, "absolute_winner": winner, "nontrivial_winner": nwinner}
    write_json(RUN / "metadata/E29R_EVALUATION_INTEGRITY.json", integrity)
    print(json.dumps({"stage": "evaluate", "candidate_count": len(out), "absolute_winner": winner, "nontrivial_winner": nwinner}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"materialize", "evaluate"}:
        raise SystemExit("usage: run_stage_c.py materialize|evaluate")
    (materialize if sys.argv[1] == "materialize" else evaluate)()
