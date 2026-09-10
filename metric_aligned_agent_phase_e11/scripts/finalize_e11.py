#!/usr/bin/env python3
"""Write E11 attempt02 provenance/protection report without model evaluation."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[2]
TDTL = PROJECT.parent / "TDTL"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(before: Path, after: Path) -> tuple[bool, pd.DataFrame]:
    source = pd.read_csv(before)
    rows = []
    for row in source.itertuples(index=False):
        path = Path(str(row.path))
        exists = path.exists()
        stat = path.stat() if exists else None
        rows.append({
            "role": row.role,
            "path": str(path),
            "exists": exists,
            "size_bytes": stat.st_size if stat else None,
            "mtime_ns": stat.st_mtime_ns if stat else None,
            "sha256": digest(path) if exists and path.is_file() else None,
        })
    current = pd.DataFrame(rows)
    current.to_csv(after, index=False)
    check = source.merge(current, on=["role", "path"], how="outer", suffixes=("_before", "_after"), indicator=True)
    # The initial snapshot was stored in CSV, whose floating-point parsing can
    # lose a few hundred ns for an integer mtime_ns around 1e18.  Size+SHA are
    # exact; record mtime as matching when it differs only at that serialization
    # granularity, rather than falsely reporting a source modification.
    fields = ["exists", "size_bytes", "sha256"]
    changed = check[check._merge.ne("both")].copy()
    both = check[check._merge.eq("both")]
    if not both.empty:
        mismatch = np.zeros(len(both), dtype=bool)
        for field in fields:
            a = both[f"{field}_before"].fillna("<NA>").astype(str)
            b = both[f"{field}_after"].fillna("<NA>").astype(str)
            mismatch |= a.ne(b).to_numpy()
        mtime_delta = (both["mtime_ns_before"].astype(float) - both["mtime_ns_after"].astype(float)).abs()
        mismatch |= mtime_delta.gt(4096).to_numpy()
        changed = pd.concat((changed, both.loc[mismatch]), ignore_index=True)
    return changed.empty, changed


def git_text() -> str:
    pieces = []
    for command in (["git", "-C", str(TDTL), "rev-parse", "HEAD"], ["git", "-C", str(TDTL), "status", "--short"]):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        pieces.append("$ " + " ".join(command) + "\n" + result.stdout)
    return "\n".join(pieces)


def write_alignment_note() -> None:
    (RUN / "reports/CELL_TO_CANONICAL_ALIGNMENT.md").write_text(
        "# Cell-to-canonical alignment\n\n"
        "ClarifyMER controller artifacts often contain three outer-dev cell rows per canonical utterance. "
        "The original TDTL benchmark is different: primary evaluation is exactly one final prediction "
        "per canonical Session-5 utterance. E11 reconstructs one Full prediction, one D3 prediction, "
        "one cached R1 prediction, and one V1/V3 action for each of 1,623 canonical IDs. "
        "No 3-cell row average is used in the benchmark table.\n\n"
        "The reconstructed Full model is selected by Session-5 `test_fscore`, per the explicit "
        "user-approved legacy-TDTL policy. D3/V1/V3 weights have zero direct Train/Dev identity "
        "intersection with Session 5; their routes are generated before Session-5 outcomes are read.\n"
    )


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    """Small dependency-free Markdown renderer for the frozen report."""
    labels = [str(col) for col in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        out = []
        for value in values:
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                out.append(f"{float(value):.{digits}f}")
            elif pd.isna(value):
                out.append("")
            else:
                out.append(str(value).replace("|", "\\|"))
        rows.append(out)
    return "\n".join([
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_report(tdtl_ok: bool, upstream_ok: bool, changes: dict[str, int]) -> None:
    table = pd.read_csv(RUN / "results/original_session5_benchmark_table.csv")
    cmp = pd.read_csv(RUN / "results/v1_v3_comparison.csv")
    boot = pd.read_csv(RUN / "results/bootstrap_summary.csv")
    mech = pd.read_csv(RUN / "results/mechanism_summary.csv")
    oracle = pd.read_csv(RUN / "results/oracle_diagnostic.csv")
    full_epochs = pd.read_csv(RUN / "models/full_benchmark_aligned/legacy_test_selected_epoch_history.csv")
    best_epoch = full_epochs.loc[full_epochs.test_weighted_f1.idxmax()]
    v3 = table.set_index("method").loc["V3 Pairwise Router"]
    lines = [
        "# Phase E11 — Original TDTL protocol benchmark report",
        "",
        "## Status",
        "",
        "`partial_benchmark_reconstruction`: the original Session-5 split and historical Full replay were reconstructed, but the fresh benchmark-aligned V1/V3 controllers did not outperform D3. This is a valid negative benchmark result, not a reason to modify the frozen formulation.",
        "",
        "## Protocol",
        "",
        "- Train: Sessions 1–4, 5,163 utterances.",
        "- Dev: Sessions 1–4, 647 utterances.",
        "- Test: Session 5, 1,623 canonical utterances / one prediction each.",
        f"- Historical Full replay: `{table.loc[table.method.eq('Historical TDTL Full'), 'weighted_f1'].iloc[0]:.12f}`.",
        f"- Reconstructed Full checkpoint: epoch `{int(best_epoch.epoch)}`, selected with Session-5 `test_fscore={best_epoch.test_weighted_f1:.12f}` per the requested legacy protocol.",
        "- D3/V1/V3 weights were refit in E11; E8 V3 trained weights were not loaded.",
        "",
        "## Label-coordinate resolution",
        "",
        "E1/E1B/E2 controller gold/prediction tables use TDTL legacy numeric labels. E11 Full/D3/test labels use public ClarifyMER IDs. Therefore R1 must be converted exactly once via `legacy_to_public=[3,4,5,0,2,1]`. The valid attempt02 uses that conversion. Separate attempts with zero/double conversion are retained as invalid schema diagnostics and are excluded from every benchmark table.",
        "",
        "## Benchmark-aligned results",
        "",
        markdown_table(table, digits=12),
        "",
        "## V3 comparisons",
        "",
        markdown_table(cmp, digits=12),
        "",
        "## Paired canonical bootstrap (2,000 replicates)",
        "",
        markdown_table(boot, digits=12),
        "",
        "## Mechanism",
        "",
        markdown_table(mech, digits=6),
        "",
        "V3 calls REASON more often than V1 and captures more R1-recoverable cases, but its danger protection falls substantially; the extra harmful calls dominate the recovered errors. This is consistent with V3 failing to transfer from the controller-specific E2 post-hoc selection distribution to the original Session-5 protocol.",
        "",
        "## Oracle diagnostics (post-hoc only)",
        "",
        markdown_table(oracle, digits=12),
        "",
        "## Integrity and interpretation",
        "",
        "- `new_model_train_test_identity_overlap = 0` for Full (Train weights), D3, V1, and V3; see `results/benchmark_training_identity_audit.csv`.",
        "- Full is intentionally legacy test-selected, as requested; it is not dev-selected.",
        "- `historical_clean_formal_V1 = 0.748701` and `E8_posthoc_V3 = 0.774940` remain historical results on the controller-specific protocol.",
        f"- `Session5_benchmark_aligned_V3 = {float(v3.weighted_f1):.12f}`.",
        "- `Session5_benchmark_aligned_V3_is_not_untouched_method_selection_test = true`: all Session-5 identities appeared earlier in the controller development/model-selection lifecycle, although E11 model weights did not directly train on them.",
        f"- `tdtl_modified = {str(not tdtl_ok).lower()}` (expected false).",
        f"- `upstream_clarifymer_artifacts_modified = {str(not upstream_ok).lower()}` (expected false).",
        f"- Protection snapshot differences: TDTL={changes['tdtl']}, upstream={changes['upstream']}.",
        "",
        "## Result status",
        "",
        "The benchmark-aligned V3 result does not support a claim that the E8 post-hoc pairwise router transfers to the original TDTL Session-5 protocol. No post-evaluation tuning or rerun was performed.",
    ]
    (RUN / "reports/PHASE_E11_BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    (RUN / "metadata/tdtl_git_after.txt").write_text(git_text())
    tdtl_ok, tdtl_diff = snapshot(RUN / "metadata/tdtl_before_snapshot.csv", RUN / "metadata/tdtl_after_snapshot.csv")
    upstream_ok, upstream_diff = snapshot(RUN / "metadata/upstream_before_snapshot.csv", RUN / "metadata/upstream_after_snapshot.csv")
    tdtl_diff.to_csv(RUN / "metadata/tdtl_protection_differences.csv", index=False)
    upstream_diff.to_csv(RUN / "metadata/upstream_protection_differences.csv", index=False)
    state = {
        "valid_benchmark_attempt": "attempt02",
        "valid_controller_session5_evaluation_count": 1,
        "attempt01_and_attempt03_schema_invalid": True,
        "tdtl_modified": not tdtl_ok,
        "upstream_clarifymer_artifacts_modified": not upstream_ok,
        "tdtl_difference_count": int(len(tdtl_diff)),
        "upstream_difference_count": int(len(upstream_diff)),
        "new_model_train_test_identity_overlap": 0,
        "r1_legacy_to_public_conversion": [3, 4, 5, 0, 2, 1],
        "e8_v3_weights_loaded": False,
        "r1_new_calls": 0,
        "post_evaluation_tuning": False,
    }
    (RUN / "metadata/final_integrity_status.json").write_text(json.dumps(state, indent=2) + "\n")
    write_alignment_note()
    write_report(tdtl_ok, upstream_ok, {"tdtl": len(tdtl_diff), "upstream": len(upstream_diff)})
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
