import csv, json, hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

RUN = Path(__file__).resolve().parents[1]
E26 = RUN.parents[2] / "experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1"
MAT = RUN / "stage_a/materialized_complete"
OUT = RUN / "stage_a/results"
LABELS = list(range(6))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def verify_materialization():
    rec = json.loads((MAT / "STAGE_A_MATERIALIZATION_SHA256.json").read_text())
    for rel, expected in rec["files"].items():
        actual = sha256(RUN / rel)
        if actual != expected:
            raise RuntimeError(f"materialization hash mismatch: {rel}")
    if rec.get("gold_in_materialization"):
        raise RuntimeError("gold_in_materialization=true")
    return rec

def mechanism(pred, rows):
    full = np.asarray([int(r["full_prediction"]) for r in rows])
    d3 = np.asarray([int(r["residual_prediction"]) for r in rows])
    # Provenance is deterministic and label-free.
    action = np.where(pred == full, "KEEP", np.where(pred == d3, "RESIDUAL", "REASON"))
    reason = action == "REASON"
    benefit = (full != np.asarray([int(r["gold_label_evaluation_only"]) for r in rows])) & (pred == np.asarray([int(r["gold_label_evaluation_only"]) for r in rows]))
    harm = (full == np.asarray([int(r["gold_label_evaluation_only"]) for r in rows])) & (pred != np.asarray([int(r["gold_label_evaluation_only"]) for r in rows]))
    captured = benefit & reason
    harmful = harm & reason
    return {
        "keep_rate": float(np.mean(action == "KEEP")),
        "residual_rate": float(np.mean(action == "RESIDUAL")),
        "reason_rate": float(np.mean(reason)),
        "reason_replacement_rate": float(np.mean(reason)),
        "benefit_total": int(np.sum(benefit)),
        "benefit_captured": int(np.sum(captured)),
        "benefit_capture_rate": float(np.sum(captured) / max(1, np.sum(benefit))),
        "harm_total": int(np.sum(harm)),
        "harmful_replacements": int(np.sum(harmful)),
        "danger_protection": float(1.0 - np.sum(harmful) / max(1, np.sum(harm))),
        "reason_benefit_precision": float(np.sum(captured) / max(1, np.sum(reason))),
        "net_intervention": int(np.sum(captured) - np.sum(harmful)),
        "deep_hard_repaired": int(np.sum(captured & (d3 != np.asarray([int(r["gold_label_evaluation_only"]) for r in rows])))),
    }

def main():
    rec = verify_materialization()
    rows = read_csv(E26 / "evaluation/results/e26_canonical_predictions.csv")
    if len(rows) != 1623 or len({r["canonical_id"] for r in rows}) != 1623:
        raise RuntimeError("canonical table is not exactly 1623 unique rows")
    # This is the first point where gold is intentionally joined for evaluation.
    (RUN / "metadata").mkdir(exist_ok=True)
    (RUN / "metadata/stage_a_gold_first_read.json").write_text(json.dumps({"gold_read": True, "stage": "stage_a_evaluation", "canonical_count": len(rows)}, indent=2))
    gold = np.asarray([int(r["gold_label_evaluation_only"]) for r in rows], dtype=np.int8)
    cids = json.loads((MAT / "canonical_ids.json").read_text())
    if cids != [r["canonical_id"] for r in rows]:
        raise RuntimeError("canonical order mismatch")
    manifest = read_csv(MAT / "candidate_manifest.csv")
    pred = np.load(MAT / "candidate_predictions.npz", mmap_mode="r")["predictions"]
    if pred.shape != (len(manifest), 1623):
        raise RuntimeError(f"prediction shape {pred.shape} != manifest {len(manifest)}")
    OUT.mkdir(parents=True, exist_ok=True)
    leaderboard = []
    matrices = {}
    for i, m in enumerate(manifest):
        p = np.asarray(pred[i], dtype=np.int8)
        if np.any((p < 0) | (p > 5)):
            raise RuntimeError(f"invalid label in {m['candidate_id']}")
        f1 = float(f1_score(gold, p, labels=LABELS, average="weighted", zero_division=0))
        acc = float(accuracy_score(gold, p))
        per = precision_recall_fscore_support(gold, p, labels=LABELS, zero_division=0)[2]
        mech = mechanism(p, rows)
        row = {"candidate_id": m["candidate_id"], "config_json": m["config_json"], "weighted_f1": f1, "accuracy": acc, **mech}
        for label, value in zip(LABELS, per):
            row[f"f1_{label}"] = float(value)
        leaderboard.append(row)
        matrices[m["candidate_id"]] = confusion_matrix(gold, p, labels=LABELS).tolist()
    leaderboard.sort(key=lambda r: (-r["weighted_f1"], -r["accuracy"], r["reason_replacement_rate"], r["candidate_id"]))
    fields = list(leaderboard[0].keys())
    with open(OUT / "stage_a_candidate_leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(leaderboard)
    winner = leaderboard[0]
    with open(OUT / "stage_a_winner.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerow(winner)
    wi = next(i for i,m in enumerate(manifest) if m["candidate_id"] == winner["candidate_id"])
    wp = np.asarray(pred[wi], dtype=np.int8)
    with open(OUT / "stage_a_winner_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["canonical_id", "prediction", "full_prediction", "residual_prediction"]); w.writeheader()
        for r, value in zip(rows, wp): w.writerow({"canonical_id": r["canonical_id"], "prediction": int(value), "full_prediction": r["full_prediction"], "residual_prediction": r["residual_prediction"]})
    (OUT / "stage_a_confusion_matrices.json").write_text(json.dumps(matrices, separators=(",", ":")))
    (RUN / "metadata/E29_STAGE_A_RESULT.json").write_text(json.dumps({"candidate_count": len(leaderboard), "winner": winner["candidate_id"], "weighted_f1": winner["weighted_f1"], "materialization_hash": sha256(MAT / "STAGE_A_MATERIALIZATION_SHA256.json")}, indent=2))
    print(json.dumps({"candidate_count": len(leaderboard), "winner": winner["candidate_id"], "weighted_f1": winner["weighted_f1"]}))

if __name__ == "__main__":
    main()
