from pathlib import Path
import csv, json

RUN=Path(__file__).resolve().parents[1]
def main():
    lead=list(csv.DictReader((RUN/"results/e27_candidate_leaderboard.csv").open())); gate=json.loads((RUN/"metadata/E27_GATE.json").read_text()); w=lead[0]
    boot=list(csv.DictReader((RUN/"results/e27_bootstrap_summary.csv").open()))
    lines=["# PHASE E27 REPORT","", "same-split test-selected posthoc exploratory", "", "## Selection", "", f"- Candidates: {len(lead)} (fixed F0-F4 manifest; no post-hoc expansion).", f"- Winner: `{w['candidate_id']}`", f"- Family/config: `{w['family']}` / `{w['config_json']}`", f"- W-F1: {float(w['weighted_f1']):.12f}", f"- Accuracy: {float(w['accuracy']):.12f}", f"- Delta vs E26: {float(w['delta_vs_e26']):+.12f}", f"- Delta vs reconstructed Full: {float(w['delta_vs_reconstructed_full']):+.12f}", f"- Delta vs D3: {float(w['delta_vs_d3']):+.12f}", f"- E28 triggered: {gate['trigger_e28']}", "", "## Mechanism", "", f"- KEEP / RESIDUAL / REASON: {float(w['keep_rate']):.6f} / {float(w['residual_rate']):.6f} / {float(w['reason_rate']):.6f}", f"- BENEFIT captured / total: {w['benefit_captured']} / {w['benefit_total']} ({float(w['benefit_capture_rate']):.6f})", f"- HARM damage / total: {w['harmful_reason_interventions']} / {w['harm_total']}", f"- Danger protection: {float(w['danger_protection']):.6f}", f"- Net intervention: {w['net_intervention']}", "", "## Bootstrap", "", "- 2,000 canonical-level paired replicates; intervals are post-hoc test-selected uncertainty, not generalization CIs."]
    for b in boot:
        lines.append(f"- {b['comparison']}: delta={float(b['observed_delta']):+.12f}, CI=[{float(b['ci_low']):+.12f}, {float(b['ci_high']):+.12f}]")
    lines += ["", "## Integrity", "", "- Session-5 gold was loaded only after materialization hash verification.", "- Materialized predictions contain no gold/outcome fields.", "- No Reasoner calls were made in E27; E28 was not created.", "- E26 caches and upstream artifacts remained read-only.", "- clean_deployment_claim=false", "- unbiased_generalization_claim=false"]
    (RUN/"reports/PHASE_E27_REPORT.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
