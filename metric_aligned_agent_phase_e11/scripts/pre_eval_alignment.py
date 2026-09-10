"""Session-5 identity/schema preflight; intentionally computes no performance metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]


def main():
    test = np.load(RUN / "cache/full_test_features.npz", allow_pickle=True)
    split = pd.read_csv(RUN / "splits/test_ids.csv")
    r1 = pd.read_csv(RUN / "cache/r1_canonical_predictions.csv")
    ids = pd.Series(test["canonical_ids"].astype(str), name="canonical_utterance_id")
    labels = test["labels"].astype(int)
    if len(ids) != 1623 or ids.nunique() != 1623 or len(split) != 1623:
        raise RuntimeError("Session-5 canonical cardinality is not 1,623")
    if set(ids) != set(split["original_iemocap_id"].astype(str)):
        raise RuntimeError("Feature-cache canonical IDs do not equal frozen Session-5 split IDs")
    expected = split.set_index("original_iemocap_id")["label_id"].astype(int)
    aligned = np.array([expected.loc[x] for x in ids], dtype=int)
    mismatch = ids[labels != aligned].tolist()
    allowed = {"Ses05F_impro07_M010", "Ses05F_impro07_M011"}
    if set(mismatch) != allowed:
        raise RuntimeError(f"Unexpected feature-cache/split-label mismatch: {mismatch}")
    r1test = r1.set_index("canonical_id").loc[ids]
    public = r1test["reason_prediction_public"].to_numpy(dtype=int)
    if np.any((public < 0) | (public >= 6)):
        raise RuntimeError("R1 numeric labels outside six-class public range")
    out = {
        "preflight": "passed",
        "purpose": "canonical identity and label-schema alignment only; no method metric computed",
        "session5_canonical_rows": int(len(ids)),
        "session5_unique_ids": int(ids.nunique()),
        "feature_label_alignment": "1621 exact; 2 documented transcript-roster discrepancies; raw TDTL feature labels frozen as benchmark gold",
        "documented_label_discrepancy_ids": mismatch,
        "r1_stored_numeric_ids_are_public": True,
        "valid_controller_session5_evaluation_count_before": 0,
    }
    (RUN / "metadata/pre_eval_alignment.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
