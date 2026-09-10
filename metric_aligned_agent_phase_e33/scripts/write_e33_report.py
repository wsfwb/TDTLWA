import csv,json
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; OUT=RUN/'results'
def row(name):
    p=OUT/f'e33_{name}_winner.csv'; return next(csv.DictReader(open(p))) if p.exists() else None
def main():
    absr=row('absolute'); legr=row('legacy_nontrivial'); incr=row('incremental_nontrivial'); hpr=row('harm_protected')
    lines=['# PHASE E33 REPORT','', 'Result type: `same_split_test_selected_posthoc_exploratory`.', '', 'E33 is offline-only (`reasoner_api_calls=0`); no new Reasoner request or cache was created. Session-5 gold was read only after materialization hash verification.']
    lines += ['', '## Candidate summary', '', f"Candidate count: {sum(1 for _ in open(OUT/'e33_candidate_leaderboard.csv'))-1}.", '']
    for title,r in [('Absolute winner',absr),('Legacy nontrivial winner',legr),('Incremental nontrivial winner',incr),('Harm-protected winner',hpr)]:
        if not r: lines += [f'### {title}', 'No candidate satisfied this definition.', '']; continue
        lines += [f"### {title}", f"- candidate_id: `{r['candidate_id']}`", f"- family/base: `{r['family']}` / `{r['base']}`", f"- config: `{r['config_json']}`", f"- weighted F1: **{float(r['weighted_f1']):.12f}**", f"- Δ vs E32: {float(r['delta_vs_e32']):+.12f}; Δ vs E31: {float(r['delta_vs_e31']):+.12f}; Δ vs Reconstructed Full: {float(r['delta_vs_reconstructed_full']):+.12f}", f"- base/final/incremental Reason rate: {float(r['base_reason_rate']):.4%} / {float(r['final_reason_rate']):.4%} / {float(r['incremental_replacement_rate']):.4%}", f"- BENEFIT captured: {r['benefit_captured']}/{r['benefit_total']} ({float(r['benefit_capture_rate']):.2%}); HARM: {r['harmful_replacements']}/{r['harm_total']}; danger protection: {float(r['danger_protection']):.2%}; net intervention: {r['net_intervention']}", '']
    lines += ['## Interpretation', '', 'The absolute winner is a raw I1/I2/I3 tuple-gated posthoc candidate. It is not a clean deployment estimate, an untouched-test result, or an unbiased generalization estimate. The two diagnostic seeds are retained with `seeded_after_e32=true` and were not used as hidden exceptions.', '', 'All low-scoring candidates remain in `e33_candidate_leaderboard.csv`; the Pareto file marks the complete W-F1–harm frontier.']
    (RUN/'reports/PHASE_E33_REPORT.md').write_text('\n'.join(lines)+'\n')
    (RUN/'reports/RAW_INTERFACE_FUSION_REPORT.md').write_text('# Raw interface fusion\n\nE33 searched observed raw interface tuples, deterministic vote rules, decision-pattern gates, adjudicator gates and bounded two-rule compositions. No Reasoner calls were made.\n')
    (RUN/'reports/POSTHOC_INTERPRETATION.md').write_text('# Posthoc interpretation\n\nAll Session-5 comparisons are same-split, test-selected posthoc exploratory uncertainty analyses. They must not be interpreted as clean deployment or unbiased generalization.\n')
    (RUN/'reports/FAILURE_DIAGNOSIS.md').write_text('# Failure diagnosis\n\nE33 did not use a new model. Any residual damage is attributable to the fixed raw tuple/vote/decision search space and is reported in the full leaderboard and Pareto frontier.\n')
    if absr:
        with open(OUT/'e33_winner_mechanism.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(absr.keys()));w.writeheader();w.writerow(absr)
if __name__=='__main__':main()
