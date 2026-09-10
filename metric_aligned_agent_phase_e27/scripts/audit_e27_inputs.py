import json, hashlib, os
from pathlib import Path
from .io_atomic import sha256_file, atomic_write_json, read_csv_rows

RUN = Path(__file__).resolve().parents[1]
E26 = RUN.parents[2] / "experiment_outputs" / "metric_aligned_agent_phase_e26" / "run_20260820T113331_CST_quota_recovered_sol_xhigh_v1"
SCHEMA = E26 / "protocol/E26_OUTPUT_SCHEMA.json"

def audit_inputs(write=True):
    table = read_csv_rows(E26 / "evaluation/results/e26_canonical_predictions.csv")
    ids = [r["canonical_id"] for r in table]
    cache_sets = {}; cache_counts = {}
    schema_hash = sha256_file(SCHEMA)
    for stage in ("I1", "I2", "I3"):
        paths = sorted((E26 / "caches" / stage).glob("*.json")); cache_counts[stage] = len(paths); stage_ids = set()
        for path in paths:
            d = json.loads(path.read_text()); cid = d.get("canonical_id")
            if d.get("status") != "success" or d.get("model") != "gpt-5.6-sol" or d.get("reasoning_effort") != "xhigh" or d.get("schema_sha256") != schema_hash:
                raise RuntimeError(f"invalid {stage} cache {path}")
            if d.get("tools") not in (None, []): raise RuntimeError(f"tool field in {path}")
            stage_ids.add(cid)
        cache_sets[stage] = stage_ids
    report = {"canonical_count": len(table), "unique_canonical_count": len(set(ids)), "cache_counts": cache_counts, "canonical_sets_equal": all(cache_sets[stage] == set(ids) for stage in cache_sets), "cache_set_sizes": {k: len(v) for k,v in cache_sets.items()}, "e26_schema_sha256": schema_hash, "e26_winner_reference": 0.7417088710802171, "source_sha256": sha256_file(E26 / "evaluation/results/e26_canonical_predictions.csv"), "source_integrity": json.loads((E26 / "evaluation/metadata/E26_EVALUATION_INTEGRITY.json").read_text())}
    if report["canonical_count"] != 1623 or report["unique_canonical_count"] != 1623 or not report["canonical_sets_equal"]: raise RuntimeError("E27 input alignment failed")
    if write: atomic_write_json(RUN / "metadata/E27_INPUT_AUDIT.json", report)
    return report

if __name__ == "__main__": print(json.dumps(audit_inputs(write=True), indent=2, sort_keys=True))
