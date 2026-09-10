import csv, hashlib, json, platform, sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
SRC = RUN.parents[1] / "metric_aligned_agent_phase_e29" / "run_20260822T180000_CST_hybrid_local_refinement_targeted_adjudicator_v1"

def read_csv(p):
    with p.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

winner=read_csv(RUN/"results/stage_c_absolute_winner.csv")[0]
nontrivial=read_csv(RUN/"results/stage_c_nontrivial_winner.csv")
boots=read_csv(RUN/"results/stage_c_bootstrap_summary.csv")
transport=json.loads((RUN/"metadata/transport_state.json").read_text())
source_cache=SRC/"caches/ADJUDICATOR"
new_cache=RUN/"caches/ADJUDICATOR"
common=[]; mismatches=[]
for p in source_cache.glob("*.json"):
    q=new_cache/p.name
    if q.exists():
        common.append(p.name)
        if sha(p)!=sha(q): mismatches.append(p.name)
integrity={
    "status":"e29r_complete_offline_stage_c",
    "source_run":str(SRC), "new_run":RUN.name,
    "source_cache_common_count":len(common), "source_cache_mismatches":mismatches,
    "source_cache_unchanged":not mismatches,
    "canonical_count":1623, "target_pool_count":200, "adjudicator_cache_count":len(list(new_cache.glob("*.json"))),
    "original_e29_remote_invocations":200, "continuation_new_reasoner_calls":1,
    "continuation_failures":int(transport.get("failures",0)), "continuation_retries":int(transport.get("retries",0)),
    "tool_calls":int(transport.get("tool_calls",0)), "model":"gpt-5.6-sol", "reasoning_effort":"xhigh",
    "candidate_count":13113, "session5_evaluation_count":1, "session5_test_tuned":True,
    "clean_deployment_claim":False, "unbiased_generalization_claim":False,
    "session5_labels_sent_to_reasoner":False, "tdtl_modified":False,
    "historical_artifacts_modified":False, "absolute_winner":winner,
    "nontrivial_winner":nontrivial[0] if nontrivial else None,
    "interpretation":"same-split test-selected posthoc exploratory"
}
(RUN/"metadata").mkdir(exist_ok=True)
(RUN/"metadata/E29R_EVALUATION_INTEGRITY.json").write_text(json.dumps(integrity,ensure_ascii=False,indent=2))
(RUN/"metadata/INTEGRITY.json").write_text(json.dumps(integrity,ensure_ascii=False,indent=2))
(RUN/"metadata/session5_access_status.json").write_text(json.dumps({"session5_labels_read":True,"session5_labels_sent_to_reasoner":False,"session5_used_for":["candidate_evaluation","candidate_selection","paired_bootstrap","mechanism_analysis"],"session5_evaluation_count":1,"session5_test_tuned":True,"clean_deployment_claim":False},ensure_ascii=False,indent=2))
(RUN/"metadata/ENVIRONMENT.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nmodel=gpt-5.6-sol\nreasoning_effort=xhigh\nnew_reasoner_calls=1\n")
(RUN/"metadata/upstream_protection_diff.csv").write_text("artifact,status,notes\nTDTL,unchanged,inherited E29 read-only audit; no E29R command targeted TDTL\nE29_source_cache,unchanged,common cache SHA-256 comparison\nE26_E27,read_only,used as frozen inputs\n")
def line(name):
    return next((b for b in boots if b["comparison"]==name),None)
def f(x): return float(x)
lines=[
"# E29R continuation report", "", "same-split test-selected posthoc exploratory", "",
"## Status", "", "E29R completed the one authorized continuation call and the offline Stage-C search. The original E29 run remains unchanged. This is not a clean-deployment or unbiased-generalization result.", "",
"## Absolute winner", "", f"- Candidate: `{winner['candidate_id']}`", f"- Configuration: `{winner['config_json']}`", f"- W-F1: **{f(winner['weighted_f1']):.12f}**", f"- Accuracy: {f(winner['accuracy']):.12f}", f"- KEEP / RESIDUAL / REASON: {f(winner['keep_rate']):.6%} / {f(winner['residual_rate']):.6%} / {f(winner['reason_rate']):.6%}", f"- BENEFIT captured / total: {winner['benefit_captured']} / {winner['benefit_total']} ({f(winner['benefit_capture_rate']):.2%})", f"- HARM damage / total: {winner['harmful_replacements']} / {winner['harm_total']}", f"- Danger protection: {f(winner['danger_protection']):.2%}", f"- Reason benefit precision: {f(winner['reason_benefit_precision']):.2%}", f"- Net intervention: {winner['net_intervention']}", "",
"## Baseline deltas", "", f"- Historical Full: {f(winner['delta_vs_historical_full']):+.12f}", f"- Reconstructed Full: {f(winner['delta_vs_reconstructed_full']):+.12f}", f"- D3: {f(winner['delta_vs_d3']):+.12f}", f"- E26: {f(winner['delta_vs_e26']):+.12f}", f"- E27: {f(winner['delta_vs_e27']):+.12f}", "",
"## Nontrivial result", "", "The absolute winner also meets the exploratory nontrivial filter (Reason replacement rate ≥2% and BENEFIT capture rate ≥25%). However, all 8 measured harmful replacements were incurred, so it is not a safe deployment policy.", "",
"## Bootstrap", "", "2,000 canonical paired resamples; intervals are posthoc test-selected uncertainty intervals, not generalization confidence intervals.", ""]
for b in boots: lines.append(f"- {b['comparison']}: delta={f(b['observed_delta']):+.12f}, 95% CI [{f(b['ci_low']):+.12f}, {f(b['ci_high']):+.12f}]")
lines += ["", "## Accounting and integrity", "", "- Original E29 Stage B: 200 invocations, 199 successes.", "- E29R continuation: exactly 1 new invocation, success; 200/200 caches available.", "- New Reasoner calls after completion: 0.", "- Stage-C candidates: 13,113.", "- Session-5 labels were used only for offline evaluation, selection, mechanism analysis, and bootstrap; never sent to Reasoner.", "- TDTL and historical artifacts: read-only / unchanged.", "- `clean_deployment_claim=false`; `unbiased_generalization_claim=false`.", ""]
(RUN/"reports").mkdir(exist_ok=True)
(RUN/"reports/PHASE_E29R_REPORT.md").write_text("\n".join(lines)+"\n")
(RUN/"reports/STAGE_C_REPORT.md").write_text("\n".join(lines)+"\n")
(RUN/"reports/POSTHOC_INTERPRETATION.md").write_text("This result is a same-split, test-selected, posthoc exploratory result. The numerical improvement is real for this fixed Session-5 split and candidate space, but the winner has zero danger protection under the reported mechanism accounting; it must not be presented as a deployable or unbiased generalization result.\n")
# Hash the principal artifacts, excluding the checksum file itself.
paths=[]
for sub in ("protocol","scripts","results","metadata","reports"):
    for p in (RUN/sub).rglob("*"):
        if p.is_file() and p.name not in {"SHA256SUMS.txt","SHA256SUMS.sha256"}: paths.append(p)
with (RUN/"metadata/SHA256SUMS.txt").open("w") as f:
    for p in sorted(paths): f.write(f"{sha(p)}  {p.relative_to(RUN)}\n")
(RUN/"metadata/SHA256SUMS.sha256").write_text(sha(RUN/"metadata/SHA256SUMS.txt")+"  metadata/SHA256SUMS.txt\n")
print(json.dumps({"status":integrity["status"],"winner_wf1":winner["weighted_f1"],"new_calls":1,"candidates":13113}))
