#!/usr/bin/env python3
"""E11: original-TDTL-protocol benchmark reconstruction.

This runner is deliberately self-contained inside the E11 run directory.  It
never writes to TDTL or to an upstream ClarifyMER experiment directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[2]
AUDIT = PROJECT / "experiment_outputs/original_iemocap_split_provenance_audit/run_20260812T105050_CST"
E8 = PROJECT / "experiment_outputs/metric_aligned_agent_phase_e8/run_20260812T001000_CST_e2_guided_controller_optimization_attempt02"
TDTL = PROJECT.parent / "TDTL"

LEGACY_LABELS = ("ang", "exc", "fru", "hap", "neu", "sad")
PUBLIC_LABELS = ("hap", "sad", "neu", "ang", "exc", "fru")
LEGACY_TO_PUBLIC = np.array([3, 4, 5, 0, 2, 1], dtype=int)
PUBLIC_TO_LEGACY = np.argsort(LEGACY_TO_PUBLIC)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairwise_sum_actions(
    p_kr: np.ndarray, p_ks: np.ndarray, p_rs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the frozen V3 orientation and conservative tie break.

    p_kr=P(RESIDUAL > KEEP), p_ks=P(REASON > KEEP),
    p_rs=P(REASON > RESIDUAL).  Scores are the sum of each action's
    pairwise win probabilities.  np.argmax implements KEEP > RESIDUAL >
    REASON when scores tie.
    """
    p_kr, p_ks, p_rs = (np.asarray(x, dtype=float) for x in (p_kr, p_ks, p_rs))
    if not (p_kr.shape == p_ks.shape == p_rs.shape):
        raise ValueError("pairwise probability shapes must match")
    scores = np.column_stack((
        (1.0 - p_kr) + (1.0 - p_ks),
        p_kr + (1.0 - p_rs),
        p_ks + p_rs,
    ))
    names = np.array(["KEEP", "RESIDUAL", "REASON"], dtype=object)
    return names[np.argmax(scores, axis=1)], scores


def assert_split_integrity(splits: Mapping[str, Set[str]]) -> None:
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(f"split overlap: {left}/{right}, e.g. {example}")


def assert_canonical_primary_rows(ids: Sequence[str], predictions: Sequence[int]) -> int:
    if len(ids) != len(predictions):
        raise ValueError("canonical IDs and predictions have different lengths")
    series = pd.Series(list(ids), dtype="string")
    if series.isna().any() or (series.str.len() == 0).any():
        raise ValueError("empty canonical ID")
    duplicates = series[series.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"duplicate canonical primary rows: {duplicates[:3]}")
    return len(series)


def weighted_f1(y_true: Sequence[int], y_pred: Sequence[int], labels=range(6)) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0))


def _stat_rows(paths: Iterable[Tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for role, path in paths:
        exists = path.exists()
        info = path.stat() if exists else None
        rows.append({
            "role": role,
            "path": str(path),
            "exists": bool(exists),
            "size_bytes": int(info.st_size) if info else None,
            "mtime_ns": int(info.st_mtime_ns) if info else None,
            "sha256": sha256_file(path) if exists and path.is_file() else None,
        })
    return pd.DataFrame(rows)


def _write_protocol() -> None:
    protocol = "# Phase E11 frozen benchmark protocol\n\n"
    protocol += "Original TDTL IEMOCAP fixed split: Train Sessions 1–4 (5,163), Dev Sessions 1–4 (647), Test Session 5 (1,623).\n\n"
    protocol += "The historical TDTL checkpoint is replayed only as a legacy test-selected reference. Per the original TDTL protocol requested for this reconstruction, E11 Full selects its checkpoint by Session-5 test_fscore. D3, V1 and V3 weights are freshly fitted only on Train/Dev and their formulation/configuration is frozen before their single Session-5 outcome evaluation.\n\n"
    protocol += "V3 formulation is frozen from POSTHOC_V3_SPEC.json (SHA-256 20dcea76e3208ebcf601da694cf79e779c56ae0faa358ce9ae155eff10f9f19e). Its E8 weights are never reused. V3 uses pairwise sum with orientations P(R>K), P(S>K), P(S>R), and KEEP > RESIDUAL > REASON ties.\n\n"
    protocol += "Primary unit: one canonical utterance and one final prediction (n=1,623). Session 5 outcome metrics are generated only in the final `evaluate` stage.\n\n"
    protocol += "Status: benchmark_aligned_posthoc_method_evaluation; Session 5 is not an untouched method-selection test because all Session-5 identities previously occurred in controller development.\n"
    (RUN / "protocol/PHASE_E11_BENCHMARK_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    label_protocol = {
        "legacy_tdtl_class_order": list(LEGACY_LABELS),
        "public_clarifymer_class_order": list(PUBLIC_LABELS),
        "legacy_to_public": {str(i): int(v) for i, v in enumerate(LEGACY_TO_PUBLIC)},
        "public_to_legacy": {str(i): int(v) for i, v in enumerate(PUBLIC_TO_LEGACY)},
        "metric": "sklearn.metrics.f1_score(labels=[0,1,2,3,4,5], average='weighted', zero_division=0)",
        "excited_happy_handling": "six classes remain distinct: exc and hap are not merged",
    }
    (RUN / "protocol/IEMOCAP_SIX_CLASS_PROTOCOL.json").write_text(json.dumps(label_protocol, indent=2) + "\n", encoding="utf-8")


def _copy_splits() -> None:
    source_names = {"train": "tdtl_original_train_ids.csv", "dev": "tdtl_original_dev_ids.csv", "test": "tdtl_original_test_ids.csv"}
    split_frames: Dict[str, pd.DataFrame] = {}
    for split, name in source_names.items():
        source = AUDIT / "results" / name
        destination = RUN / "splits" / f"{split}_ids.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        frame = pd.read_csv(destination)
        split_frames[split] = frame
    ids = {name: set(frame.original_iemocap_id.astype(str)) for name, frame in split_frames.items()}
    assert_split_integrity(ids)
    if {name: len(values) for name, values in ids.items()} != {"train": 5163, "dev": 647, "test": 1623}:
        raise ValueError("original split counts do not match 5163/647/1623")
    if len(set.union(*ids.values())) != 7433:
        raise ValueError("original split union is not 7433")
    if set(split_frames["test"].session.astype(str)) != {"Ses05"}:
        raise ValueError("test split is not exactly Session 5")
    pd.DataFrame([
        {"split": split, "canonical_utterances": len(frame), "dialogues": int(frame.dialogue_id.nunique()),
         "sessions": ";".join(sorted(frame.session.astype(str).unique()))}
        for split, frame in split_frames.items()
    ]).to_csv(RUN / "results/original_split_integrity.csv", index=False)


def _historical_replay() -> float:
    source = RUN / "copied_readonly_evidence/tdtl_historical_full_predictions.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    labels = payload.get("label") or payload.get("labels")
    preds = payload.get("pred") or payload.get("preds") or payload.get("predictions")
    if labels is None or preds is None:
        raise ValueError("historical prediction JSON lacks labels/predictions")
    value = weighted_f1(labels, preds)
    pd.DataFrame([{
        "method": "Historical TDTL Full", "test_utterances": len(labels), "weighted_f1": value,
        "status": "legacy_test_selected_reference", "expected_audit_value": 0.7345846815716002,
        "absolute_difference": abs(value - 0.7345846815716002),
    }]).to_csv(RUN / "results/historical_full_replay.csv", index=False)
    if len(labels) != 1623 or abs(value - 0.7345846815716002) > 1e-9:
        raise ValueError(f"historical Full replay mismatch: n={len(labels)} W-F1={value:.12f}")
    return value


def _cell_alignment_report() -> None:
    text = "# Cell-to-canonical alignment\n\n"
    text += "Prior E1/E1B/E2 controller artifacts contain three outer-dev rows per canonical utterance. These rows are separate outer-cell/model realizations used for controller diagnostics, not three IEMOCAP benchmark utterances. The original TDTL fixed test is one prediction per canonical Session-5 utterance (n=1,623).\n\n"
    text += "E11 therefore retrains one Session-5-test-selected Full model (matching legacy TDTL selection) and materializes one Full/D3/V1/V3 action per canonical utterance. No cell-level W-F1 is used as the primary benchmark metric.\n"
    (RUN / "reports/CELL_TO_CANONICAL_ALIGNMENT.md").write_text(text, encoding="utf-8")


def preflight() -> None:
    (RUN / "metadata").mkdir(exist_ok=True)
    tdtl_paths = [
        ("tdtl_fusion", TDTL / "IEMOCAP/multimodel_fusion.py"),
        ("tdtl_dataset", TDTL / "IEMOCAP/dataset.py"),
        ("tdtl_preprocessing", TDTL / "IEMOCAP/preprocessing.py"),
        ("tdtl_utils", TDTL / "IEMOCAP/utils.py"),
        ("tdtl_historical_best", TDTL / "IEMOCAP/save_model/multimodal_fusion_best.json"),
        ("tdtl_historical_metrics", TDTL / "IEMOCAP/save_model/multimodal_fusion_metrics_20260807_200257.json"),
        ("tdtl_train_feature", TDTL / "IEMOCAP/feature/train_features.pkl"),
        ("tdtl_dev_feature", TDTL / "IEMOCAP/feature/dev_features.pkl"),
        ("tdtl_test_feature", TDTL / "IEMOCAP/feature/test_features.pkl"),
    ]
    _stat_rows(tdtl_paths).to_csv(RUN / "metadata/tdtl_before_snapshot.csv", index=False)
    git = subprocess.run(["git", "-C", str(TDTL), "status", "--short"], capture_output=True, text=True)
    head = subprocess.run(["git", "-C", str(TDTL), "rev-parse", "HEAD"], capture_output=True, text=True)
    (RUN / "metadata/tdtl_git_before.txt").write_text(
        f"HEAD exit={head.returncode}\\n{head.stdout}{head.stderr}\\nstatus exit={git.returncode}\\n{git.stdout}{git.stderr}", encoding="utf-8")
    upstream_paths = [
        ("e0t_report", PROJECT / "experiment_outputs/metric_aligned_agent_phase_e0t/run_20260810T210000_CST_exploratory_v1/reports/PHASE_E0T_REPORT.md"),
        ("e1_predictions", PROJECT / "experiment_outputs/metric_aligned_agent_phase_e1/run_20260811T090000_CST_exploratory_v1/results/controller_oof_predictions.csv"),
        ("e1b_predictions", PROJECT / "experiment_outputs/metric_aligned_agent_phase_e1b/run_20260811T100000_CST_independent_outerdev_v1/results/e1b_predictions.csv"),
        ("e2_predictions", PROJECT / "experiment_outputs/metric_aligned_agent_phase_e2/run_20260811T141645_CST_formal_test_v1/results/formal_test_predictions.csv"),
        ("e8_spec", E8 / "frozen_posthoc_v3_pairwise_sum/POSTHOC_V3_SPEC.json"),
    ]
    _stat_rows(upstream_paths).to_csv(RUN / "metadata/upstream_before_snapshot.csv", index=False)
    _copy_splits()
    _write_protocol()
    _cell_alignment_report()
    historical = _historical_replay()
    (RUN / "metadata/preflight_status.json").write_text(json.dumps({
        "preflight_complete": True, "historical_full_replay_wf1": historical,
        "session5_gold_evaluated_by_e11": False,
        "posthoc_v3_spec_sha256": sha256_file(RUN / "protocol/POSTHOC_V3_SPEC.json"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"PRELIGHT_OK historical_full_WF1={historical:.12f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight"])
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()


if __name__ == "__main__":
    main()
