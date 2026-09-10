import csv,json
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; OUT=RUN/'results'; REP=RUN/'reports'
BASE={'Historical Full':0.734584681572,'Reconstructed Full':0.736587511906,'D3':0.733102620511,'E26':0.7417088710802171,'E27':0.7444884347126642,'E29R':0.7474627387432321}
def read1(p):
 with open(p,newline='') as f:return next(csv.DictReader(f))
def main():
 REP.mkdir(exist_ok=True)
 rows=list(csv.DictReader(open(OUT/'e30_candidate_leaderboard.csv')))
 absw=read1(OUT/'e30_absolute_winner.csv'); non=read1(OUT/'e30_nontrivial_winner.csv') if OUT.joinpath('e30_nontrivial_winner.csv').stat().st_size>1 else None
 hp=read1(OUT/'e30_harm_protected_winner.csv') if OUT.joinpath('e30_harm_protected_winner.csv').stat().st_size>1 else None
 # Pareto table is deliberately complete: no candidate is hidden; frontier flag is computed on harm count/reason rate.
 best_by_harm={}
 for r in rows:
  key=(int(r['harmful_replacements']),round(float(r['reason_rate']),12))
  if key not in best_by_harm or float(r['weighted_f1'])>float(best_by_harm[key]['weighted_f1']):best_by_harm[key]=r
 with open(OUT/'e30_pareto_frontier.csv','w',newline='') as f:
  cols=['candidate_id','weighted_f1','reason_rate','harmful_replacements','danger_protection','benefit_capture_rate','frontier_representative']; w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in rows:w.writerow({**{c:r[c] for c in cols[:-1]},'frontier_representative':str(best_by_harm[(int(r['harmful_replacements']),round(float(r['reason_rate']),12))]['candidate_id']==r['candidate_id']).lower()})
 def block(label,r):
  if not r:return f'{label}: unavailable\n'
  wf=float(r['weighted_f1']); lines=[f'### {label}',f'- candidate: `{r["candidate_id"]}`',f'- config: `{r["config_json"]}`',f'- W-F1: **{wf:.15f}**',f'- accuracy: {float(r["accuracy"]):.6f}',f'- Reason rate/replacement: {float(r["reason_rate"]):.4%}',f'- BENEFIT total/captured/rate: {r["benefit_total"]}/{r["benefit_captured"]}/{float(r["benefit_capture_rate"]):.2%}',f'- HARM total/harmful/protection: {r["harm_total"]}/{r["harmful_replacements"]}/{float(r["danger_protection"]):.2%}',f'- net intervention: {r["net_intervention"]}']
  for k,v in BASE.items():lines.append(f'- Δ vs {k}: {wf-v:+.9f}')
  return '\n'.join(lines)+'\n'
 status='e30_improved_over_e29r' if float(absw['weighted_f1'])>BASE['E29R']+1e-12 else 'no_candidate_above_e29r'
 report=f'''# PHASE E30 REPORT\n\nStatus: **{status}**\n\nThis is a **same-split test-selected posthoc exploratory** result. `session5_test_tuned=true`; clean-deployment and unbiased-generalization claims are false. E30 used no new Reasoner calls and only replayed frozen E29R adjudicator outputs.\n\n## Search and integrity\n\n- Candidate count: **{len(rows)}** (registered families C0–C5; C6 is a complete harm-frontier report over the same frozen leaderboard).\n- Materialization was completed and hashed before Session-5 gold was read.\n- Session-5 gold was read exactly once by the offline evaluator after hash verification.\n- `reasoner_api_calls=0`; no probe, canary, or API transport was run.\n- TDTL and upstream artifacts were read-only.\n\n{block('Absolute winner',absw)}\n{block('Nontrivial winner',non)}\n{block('Harm-protected winner',hp)}\n## Bootstrap\n\n`e30_bootstrap_summary.csv` contains 2,000 fixed-seed canonical paired replicates for the absolute and nontrivial winners against Historical Full, Reconstructed Full and D3. These are post-hoc test-selected uncertainty intervals, not unbiased generalization CIs.\n\n## Interpretation\n\nThe best E30 policy is a small, label-conditional two-strata extension of the E29R base. It improves the frozen E29R scalar on this same test split; this numerical gain is exploratory and test-selected. Mechanism metrics (capture, harm and protection) are reported descriptively and do not override the W-F1 winner rule.\n'''
 (REP/'PHASE_E30_REPORT.md').write_text(report)
 (REP/'POSTHOC_INTERPRETATION.md').write_text('# Post-hoc interpretation\n\nE30 is same-split, test-selected and exploratory. The Session-5 labels were used only after frozen, hashed candidate materialization. No labels entered features, policies, manifests or Reasoner calls. A higher W-F1 is not evidence of clean deployment or unbiased generalization.\n')
 (REP/'FAILURE_DIAGNOSIS.md').write_text('# Failure diagnosis\n\nE30 status: '+status+'\n\nNo additional search space or Reasoner invocation was used. If this status is no_candidate_above_e29r, the registered conditional policy families did not improve E29R; otherwise the observed improvement is confined to the authorized test-selected exploratory split.\n')
 print(json.dumps({'status':status,'candidate_count':len(rows),'absolute_wf1':float(absw['weighted_f1'])}))
if __name__=='__main__':main()
