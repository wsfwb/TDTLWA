import csv, json
from pathlib import Path
import numpy as np
from .run_stage_c import RUN, E29, load_inputs, make_candidate_details, read_csv, write_csv

rows, ids, index, full, d3, historical, stagea, e27, e26, target_ids, caches = load_inputs()
winner = read_csv(RUN / "results/stage_c_absolute_winner.csv")[0]
cfg = json.loads(winner["config_json"])
base_map = {"FULL": full, "D3": d3, "STAGEA": stagea, "E27": e27, "E26": e26}
if cfg["base"].startswith("A_TOP_"):
    top = read_csv(E29 / "stage_a/results/stage_a_candidate_leaderboard.csv")[:5]
    manifest = read_csv(E29 / "stage_a/materialized_complete/candidate_manifest.csv")
    lookup = {r["candidate_id"]: i for i, r in enumerate(manifest)}
    npz = np.load(E29 / "stage_a/materialized_complete/candidate_predictions.npz", allow_pickle=False)["predictions"]
    rank = int(cfg["base"].split("_")[-1])
    base = np.asarray(npz[lookup[top[rank]["candidate_id"]]], dtype=np.int8)
else:
    base = base_map[cfg["base"]]
_, selected, reason, cheap, action = make_candidate_details(base, cfg["base"], rows, ids, index, target_ids, caches, cfg)
names = ["KEEP", "RESIDUAL", "REASON"]
write_csv(RUN / "results/stage_c_winner_mechanism.csv", [{"canonical_id": c, "prediction": int(reason[i] if selected[i] else base[i]), "cheap_prediction": int(cheap[i]), "reason_prediction": int(reason[i]), "action": names[int(action[i])], "new_adjudicator_replacement": bool(selected[i])} for i, c in enumerate(ids)])
print(json.dumps({"winner": winner["candidate_id"], "rows": len(ids)}))
