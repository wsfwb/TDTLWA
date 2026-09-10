import hashlib,json,math,os
from pathlib import Path
import numpy as np
from .io_atomic import atomic_write_csv,atomic_write_json,read_csv_rows,sha256_file
from .stage_a_views import build_views
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; E27=ROOT/'experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1'
FORBIDDEN=('gold','outcome','correctness','benefit','harm','wf1','metric','oracle'); ORDER=('V_UNANIMOUS','V_I3I1','V_I3I2','V_ANY','V_MAJORITY','V_BEST_SUPPORTED')
def validate_materializer_rows(rows):
 if not rows: raise ValueError('empty')
 if any(any(x in k.lower() for x in FORBIDDEN) for r in rows for k in r): raise ValueError('forbidden')
 if len({r.get('canonical_id') for r in rows})!=len(rows): raise ValueError('duplicate')
 return True
def mask_union(a,b): return set(a)|set(b)
def mask_intersection(a,b): return set(a)&set(b)
def mask_difference(a,b): return set(a)-set(b)
def mask_symmetric_difference(a,b): return set(a)^set(b)
def _cid(c): return 'cand_'+hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _merge(rows,items,base='FULL'):
 by={x['canonical_id']:x for x in items}; return np.asarray([int(by[c]['proposal_label']) if c in by else int(r['full_prediction'] if base=='FULL' else r['residual_prediction']) for r in rows for c in [r['canonical_id']]],dtype=np.int8)
def _spec(cfg,pred): return {'candidate_id':_cid(cfg),'config':cfg,'pred':np.asarray(pred,dtype=np.int8)}
def enumerate_specs(rows,views):
 out=[]; w=set(x['canonical_id'] for x in views['V_ANY'][:28]); names=list(views)
 # A0 frozen references. These are replayed from predictions only; no metric
 # fields enter candidate construction.
 def add_a0(name,pred): out.append(_spec({'family':'A0','name':name},pred))
 add_a0('historical_full',[int(r['historical_full_prediction']) for r in rows]); add_a0('reconstructed_full',[int(r['full_prediction']) for r in rows]); add_a0('d3_residual',[int(r['residual_prediction']) for r in rows])
 e26_order=[i for i,r in enumerate(rows) if r['i3_decision']=='ACCEPT_REASON']; score=np.asarray([float(rows[i]['i3_confidence'])-float(rows[i]['i3_uncertainty']) for i in e26_order]); chosen=set(np.asarray(e26_order,dtype=int)[np.argsort(score)[::-1][:int(math.ceil(.05*len(rows)))].tolist()]); add_a0('e26_i3_budget_0.05',[int(r['i3_reason_prediction']) if i in chosen else int(r['full_prediction']) for i,r in enumerate(rows)])
 e27_manifest=read_csv_rows(E27/'materialized/candidate_manifest.csv'); e27_np=np.load(E27/'materialized/candidate_predictions.npz'); e27_map={str(c):e27_np['predictions'][i] for i,c in enumerate(e27_np['candidate_ids'])}; top_ids=[r['candidate_id'] for r in read_csv_rows(E27/'results/e27_candidate_leaderboard.csv')[:100]]
 for rank,candidate in enumerate(top_ids):
  if candidate in e27_map: add_a0(f'e27_top100_{rank:03d}',e27_map[candidate])
 # Keep the explicit E27 winner alias even if its rank-table position changes.
 winner=json.loads((E27/'metadata/E27_GATE.json').read_text())['best_candidate'];
 if winner in e27_map: add_a0('e27_winner',e27_map[winner])
 for name,v in views.items():
  ids=[x['canonical_id'] for x in v]; by={x['canonical_id']:x for x in v}
  for k in range(min(200,len(v))+1):
   s=set(ids[:k])
   for op,chosen in [('union',w|s),('intersection',w&s),('w28_minus',w-s),('view_minus',s-w),('symmetric',w^s)]: out.append(_spec({'family':'A1','view':name,'k':k,'op':op},_merge(rows,[by[c] for c in chosen if c in by])))
 for ai in range(len(names)):
  for bi in range(ai+1,len(names)):
   a,b=views[names[ai]],views[names[bi]]; ma={x['canonical_id']:x for x in a}; mb={x['canonical_id']:x for x in b}; lim=min(200,max(len(a),len(b)))
   for k in range(lim+1):
    sa={x['canonical_id'] for x in a[:k]}; sb={x['canonical_id'] for x in b[:k]}
    for op,chosen in [('intersection',sa&sb),('union',sa|sb),('a_minus_b',sa-sb),('b_minus_a',sb-sa)]: out.append(_spec({'family':'A2','a':names[ai],'b':names[bi],'k':k,'op':op},_merge(rows,[ma.get(c,mb.get(c)) for c in chosen])))
 core=(0,5,9,13,17,19,23,28,37,50); ext=(0,5,9,13,17,19,23,28,30,32,34,37,39,50,60,77,84,100,117)
 for src in ('V_ANY','V_MAJORITY','V_BEST_SUPPORTED'):
  for gate in ('G0','G1','G2','G3','G4','G5','G6','G7'):
   for k in ext:
    ex=views[src][:min(k,len(views[src]))]
    if gate=='G1': ex=[x for x in ex if x['support']==3]
    if gate=='G2': ex=[x for x in ex if x['support']>=2]
    if gate=='G3': ex=[x for x in ex if x['proposal_label']==x['d3_label']]
    if gate=='G4': ex=[x for x in ex if x['full_label']!=x['d3_label']]
    if gate=='G5': ex=[x for x in ex if x['min_margin']>=0]
    if gate=='G6': ex=[x for x in ex if x['mean_margin']>=.25]
    if gate=='G7': ex=[x for x in ex if x['mean_margin']>=.5]
    chosen={x['canonical_id']:x for x in views['V_UNANIMOUS'][:min(k,len(views['V_UNANIMOUS']))]}; chosen.update({x['canonical_id']:x for x in views['V_I3I1'][:min(k,len(views['V_I3I1']))]}); chosen.update({x['canonical_id']:x for x in ex}); out.append(_spec({'family':'A3','source':src,'gate':gate,'k':k},_merge(rows,list(chosen.values()))))
 for ai in range(len(names)):
  for bi in range(ai+1,len(names)):
   a,b=views[names[ai]],views[names[bi]]; ra={x['canonical_id']:1-i/max(1,len(a)-1) for i,x in enumerate(a)}; rb={x['canonical_id']:1-i/max(1,len(b)-1) for i,x in enumerate(b)}; ib={x['canonical_id']:x for x in a+b}
   for alpha in (.25,.5,1.,2.,4.):
    order=sorted(set(ra)|set(rb),key=lambda c:(-(ra.get(c,0)+alpha*rb.get(c,0)),c))
    for k in range(min(200,len(order))+1): out.append(_spec({'family':'A4','a':names[ai],'b':names[bi],'alpha':alpha,'k':k},_merge(rows,[ib[c] for c in order[:k]])))
 for src in names:
  for k in (9,13,17,19,23,28,30,32,34,37,39,50):
   parent=views[src][:min(k,len(views[src]))]
   for gate in ('none','full_eq_d3','full_neq_d3','support2','support3','support_ge2'):
    chosen=[x for x in parent if gate=='none' or (gate=='full_eq_d3' and x['full_label']==x['d3_label']) or (gate=='full_neq_d3' and x['full_label']!=x['d3_label']) or (gate=='support2' and x['support']==2) or (gate=='support3' and x['support']==3) or (gate=='support_ge2' and x['support']>=2)]
    out.append(_spec({'family':'A5','parent':src,'k':k,'gate':gate},_merge(rows,chosen)))
 return list({x['candidate_id']:x for x in out}.values())
def main():
 src=read_csv_rows(E26/'evaluation/results/e26_canonical_predictions.csv'); rows=[{k:v for k,v in r.items() if not any(x in k.lower() for x in FORBIDDEN)} for r in src]; validate_materializer_rows(rows); views=build_views(rows); atomic_write_json(RUN/'stage_a/views.json',views); specs=enumerate_specs(rows,views)
 if len(specs)>60000: raise RuntimeError(f'candidate cap {len(specs)}')
 mdir=RUN/os.environ.get('E29_STAGE_A_MATERIALIZED_DIR','stage_a/materialized_complete'); mdir.mkdir(parents=True,exist_ok=True); np.savez_compressed(mdir/'candidate_predictions.npz',predictions=np.vstack([s['pred'] for s in specs]),candidate_ids=np.asarray([s['candidate_id'] for s in specs],dtype='U128')); atomic_write_json(mdir/'canonical_ids.json',[r['canonical_id'] for r in rows]); atomic_write_csv(mdir/'candidate_manifest.csv',[{'candidate_id':s['candidate_id'],'config_json':json.dumps(s['config'],sort_keys=True,separators=(',',':'))} for s in specs],['candidate_id','config_json']); files={str(p.relative_to(RUN)):sha256_file(p) for p in (mdir/'candidate_predictions.npz',mdir/'canonical_ids.json',mdir/'candidate_manifest.csv',RUN/'protocol/E29_STAGE_A_SEARCH_SPACE.json')}; atomic_write_json(mdir/'STAGE_A_MATERIALIZATION_SHA256.json',{'files':files,'candidate_count':len(specs),'canonical_count':1623,'gold_in_materialization':False,'frozen':True}); print(json.dumps({'candidate_count':len(specs),'canonical_count':1623,'materialized_dir':str(mdir.relative_to(RUN))}))
if __name__=='__main__': main()
