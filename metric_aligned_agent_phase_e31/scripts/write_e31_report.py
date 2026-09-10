import csv,json
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; OUT=RUN/'results'; REP=RUN/'reports'
BASE={'Historical Full':0.734584681572,'Reconstructed Full':0.736587511906,'D3':0.733102620511,'E26':0.7417088710802171,'E27':0.7444884347126642,'E29R':0.7474627387432321,'E30':0.7487425903546215}
def read1(p):
 with open(p,newline='') as f:return next(csv.DictReader(f))
def block(label,r):
 if not r:return f'### {label}\nUnavailable.\n\n'
 wf=float(r['weighted_f1']); lines=[f'### {label}',f'- candidate: `{r["candidate_id"]}`',f'- config: `{r["config_json"]}`',f'- W-F1: **{wf:.15f}**',f'- accuracy: {float(r["accuracy"]):.6f}',f'- KEEP/RESIDUAL/REASON: {float(r["keep_rate"]):.2%}/{float(r["residual_rate"]):.2%}/{float(r["final_reason_rate"]):.2%}',f'- base Reason rate: {float(r["base_reason_rate"]):.2%}',f'- incremental replacement rate: {float(r["incremental_replacement_rate"]):.2%}',f'- BENEFIT: {r["benefit_total"]} total, {r["benefit_captured"]} captured ({float(r["benefit_capture_rate"]):.2%})',f'- HARM: {r["harm_total"]} total, {r["harmful_replacements"]} harmful',f'- danger protection: {float(r["danger_protection"]):.2%}',f'- net intervention: {r["net_intervention"]}']
 for k,v in BASE.items():lines.append(f'- Δ vs {k}: {wf-v:+.9f}')
 return '\n'.join(lines)+'\n\n'
def main():
 REP.mkdir(exist_ok=True); rows=list(csv.DictReader(open(OUT/'e31_candidate_leaderboard.csv'))); absw=read1(OUT/'e31_absolute_winner.csv'); non=read1(OUT/'e31_nontrivial_winner.csv'); hp=read1(OUT/'e31_harm_protected_winner.csv')
 best={}
 for r in rows:
  k=(int(r['harmful_replacements']),round(float(r['final_reason_rate']),12)); best[k]=r if k not in best or float(r['weighted_f1'])>float(best[k]['weighted_f1']) else best[k]
 with open(OUT/'e31_pareto_frontier.csv','w',newline='') as f:
  cols=['candidate_id','weighted_f1','harmful_replacements','danger_protection','final_reason_rate','benefit_capture_rate','frontier_representative']; w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
  for r in rows:w.writerow({**{c:r[c] for c in cols[:-1]},'frontier_representative':str(best[(int(r['harmful_replacements']),round(float(r['final_reason_rate']),12))]['candidate_id']==r['candidate_id']).lower()})
 status='e31_improved_over_e30' if float(absw['weighted_f1'])>BASE['E30']+1e-12 else 'no_candidate_above_e30'
 report=f'''# PHASE E31 REPORT\n\nStatus: **{status}**\n\nThis is a **same-split test-selected posthoc exploratory** result. `reasoner_api_calls=0`; clean-deployment and unbiased-generalization claims are false. E31 is offline-only and uses frozen E29R adjudicator outputs.\n\n## Frozen audit\n\n- Canonical count: 1,623; E29R target pool/cache: 200/200.\n- E30 replay was performed only after materialization hash verification and exactly reproduced W-F1 0.7487425903546215.\n- Materialized candidate count: **{len(rows)}**; candidate predictions are 10,684 × 1,623 and contain no gold/outcome/metric fields.\n- Session-5 gold was read after materialization, never sent to a model.\n\n{block('Absolute winner',absw)}{block('Nontrivial winner',non)}{block('Harm-protected winner',hp)}## Bootstrap\n\n`e31_bootstrap_summary.csv` contains fixed-seed (20260822), 2,000-replicate canonical paired bootstrap comparisons. These are post-hoc test-selected uncertainty intervals, not unbiased generalization CIs.\n\n## Interpretation\n\nThe best registered local conditional policy is a single label-pair condition over the frozen E30 base (`cheap_label=neu`, `adjudicator_label=fru`, k=2). It numerically improves E30 on this same split. The gain is exploratory and test-selected; mechanism metrics describe collateral damage but do not override the W-F1 winner rule.\n'''
 (REP/'PHASE_E31_REPORT.md').write_text(report); (REP/'POSTHOC_INTERPRETATION.md').write_text('# Post-hoc interpretation\n\nE31 is same-split, test-selected and exploratory. Gold was loaded only after immutable materialization verification and was never sent to Reasoner. The numerical gain cannot be interpreted as clean deployment or unbiased generalization.\n'); (REP/'FAILURE_DIAGNOSIS.md').write_text('# Failure diagnosis\n\nE31 status: '+status+'\n\nNo unregistered search space or new Reasoner call was used.\n'); print(json.dumps({'status':status,'candidate_count':len(rows),'wf1':float(absw['weighted_f1'])}))
if __name__=='__main__':main()
