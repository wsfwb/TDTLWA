import csv,json,hashlib
from pathlib import Path
from .e30_policy_core import LABELS
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]
E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; E29R=ROOT/'experiment_outputs/metric_aligned_agent_phase_e29r/run_20260822T_continuation_one_call_stage_c_v1'
FORBIDDEN=('gold','outcome','correctness','benefit','harm','wf1','metric','oracle')
def read_csv(p):
 with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
 src=read_csv(E26/'evaluation/results/e26_canonical_predictions.csv'); cache={}
 for p in (E29R/'caches/ADJUDICATOR').glob('*.json'):
  d=json.loads(p.read_text());
  if d.get('status')!='success' or d.get('model')!='gpt-5.6-sol' or d.get('reasoning_effort')!='xhigh': raise RuntimeError('invalid adjudicator cache')
  cache[d['canonical_id']]=d['output']
 wp={r['canonical_id']:int(r['prediction']) for r in read_csv(E29R/'results/stage_c_winner_canonical_predictions.csv')}
 if len(src)!=1623 or len({r['canonical_id'] for r in src})!=1623 or len(cache)!=200: raise RuntimeError('input count')
 rows=[]
 for r in src:
  c=r['canonical_id']; o=cache.get(c); al=LABELS[o['final_label']] if o else wp.get(c,int(r['full_prediction'])); dec=o['decision'] if o else 'KEEP_BASE'; conf=float(o['confidence']) if o else 0.; unc=float(o['uncertainty']) if o else 1.; ps=o.get('preferred_source','FULL') if o else 'FULL'; cons=o.get('evidence_consistency','LOW') if o else 'LOW'; sl={'FULL':int(r['full_prediction']),'D3':int(r['residual_prediction']),'I1':int(r['i1_reason_prediction']),'I2':int(r['i2_reason_prediction']),'I3':int(r['i3_reason_prediction']),'REVISED':al}.get(ps,al)
  rows.append({'canonical_id':c,'full_prediction':int(r['full_prediction']),'residual_prediction':int(r['residual_prediction']),'e26_prediction':int(r['full_prediction']),'e27_prediction':int(r['full_prediction']),'e29r_prediction':int(wp.get(c,int(r['full_prediction']))),'i1_label':int(r['i1_reason_prediction']),'i2_label':int(r['i2_reason_prediction']),'i3_label':int(r['i3_reason_prediction']),'adj_label':al,'adj_decision':dec,'adj_confidence':conf,'adj_uncertainty':unc,'preferred_source':ps,'source_label':sl,'evidence_consistency':cons,'full_d3_agreement':int(r['full_prediction'])==int(r['residual_prediction']),'target_pool_order':0 if o else 10**6,'support':sum(al==int(r[k]) for k in ('i1_reason_prediction','i2_reason_prediction','i3_reason_prediction'))})
 if any(any(z in k.lower() for z in FORBIDDEN) for r in rows for k in r): raise RuntimeError('forbidden feature')
 (RUN/'stage_c/features.json').write_text(json.dumps(rows,separators=(',',':'))); (RUN/'metadata/FEATURES_MANIFEST.json').write_text(json.dumps({'canonical_count':len(rows),'target_cache_count':len(cache),'gold_in_features':False},indent=2)); print(json.dumps({'canonical_count':len(rows),'target_count':len(cache)}))
if __name__=='__main__':main()
