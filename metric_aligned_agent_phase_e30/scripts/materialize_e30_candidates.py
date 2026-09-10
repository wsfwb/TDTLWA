import csv,json,hashlib
from pathlib import Path
import numpy as np
from .e30_policy_core import candidate_id,sorted_ids,apply_policy,PREFIXES,CONF,UNC
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; E27=ROOT/'experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1'
def read_csv(p):
 with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 print('start',flush=True)
 rows=json.loads((RUN/'stage_c/features.json').read_text()); n=len(rows); ids=[r['canonical_id'] for r in rows]; by={r['canonical_id']:r for r in rows}
 print('rows',n,flush=True)
 full=np.asarray([r['full_prediction'] for r in rows],dtype=np.int8); d3=np.asarray([r['residual_prediction'] for r in rows],dtype=np.int8); print('arrays',flush=True)
 e26=[]; elig=[i for i,r in enumerate(rows) if r['adj_decision'] in ('REVISE','ACCEPT_EXISTING_REASON') and r['adj_confidence']-r['adj_uncertainty']>0]
 print('elig',len(elig),flush=True)
 for i in sorted(elig,key=lambda i:(-(rows[i]['adj_confidence']-rows[i]['adj_uncertainty']),ids[i]))[:int(np.ceil(.05*n))]: e26.append(i)
 print('sorted',len(e26),flush=True)
 e26p=full.copy(); e26p[e26]=[rows[i]['i3_label'] for i in e26]; print('e26',flush=True)
 print('loadnp',E27,flush=True); e27np=np.load(E27/'materialized/candidate_predictions.npz'); print('npdone',flush=True); e27man=read_csv(E27/'materialized/candidate_manifest.csv'); print('mandone',flush=True); e27leader=read_csv(E27/'results/e27_candidate_leaderboard.csv'); print('leaddone',flush=True); e27idx={str(c):i for i,c in enumerate(e27np['candidate_ids'])}; needed=[m['candidate_id'] for m in e27leader[:10]]; e27map={c:e27np['predictions'][e27idx[c]] for c in needed if c in e27idx}; print('mapdone',len(e27map),flush=True)
 bases={'FULL':full,'D3':d3,'E26':e26p,'E27':np.asarray(e27map[e27leader[0]['candidate_id']],dtype=np.int8)}
 e29=np.asarray([r['e29r_prediction'] for r in rows],dtype=np.int8); bases['E29R']=e29
 for rank,m in enumerate(e27leader[:10]):
  if m['candidate_id'] in e27map:bases[f'STAGEA_TOP_{rank+1}']=np.asarray(e27map[m['candidate_id']],dtype=np.int8)
 specs=[]; preds=[]; seen_ids=set()
 def add(cfg,p):
  cid=candidate_id(cfg)
  if cid not in seen_ids: seen_ids.add(cid); specs.append({'candidate_id':cid,'config_json':json.dumps(cfg,sort_keys=True,separators=(',',':'))}); preds.append(np.asarray(p,dtype=np.int8))
 for name,p in bases.items(): add({'family':'C0_FIXED','base':name},p)
 print('c0',len(specs),flush=True)
 conditional_bases={k:bases[k] for k in ('FULL','D3','E26','E27','E29R')}
 target=[r['canonical_id'] for r in rows if r['target_pool_order']<10**6]
 # C1: each registered single stratum, thresholds and prefixes.
 strata=[]
 for lab in range(6): strata.append((f'base_label_{lab}',lambda r,lab=lab,b=lab:int(r['full_prediction'])==b))
 for lab in range(6): strata.append((f'adj_label_{lab}',lambda r,lab=lab,b=lab:int(r['adj_label'])==b))
 strata += [('agreement',lambda r:int(r['full_d3_agreement'])==1),('disagreement',lambda r:int(r['full_d3_agreement'])==0),('revise',lambda r:r['adj_decision']=='REVISE'),('accept_reason',lambda r:r['adj_decision']=='ACCEPT_EXISTING_REASON')]
 for name,base in conditional_bases.items():
  for sname,fn in strata:
   for c in CONF:
    for u in (1.0,):
     eligible=[r['canonical_id'] for r in rows if fn(r) and r['adj_confidence']>=c and r['adj_uncertainty']<=u and r['adj_label']!=int(r['full_prediction'])]
     order=sorted_ids(rows,eligible,'margin')
     for k in PREFIXES: add({'family':'C1_LABEL_GATE','base':name,'stratum':sname,'confidence':c,'uncertainty_max':u,'k':k,'mode':'FINAL'},apply_policy(rows,base,order[:min(k,len(order))]))
   print('c1',name,len(specs),flush=True)
 # C2 ranking strata
 for name,base in conditional_bases.items():
  eligible=[r['canonical_id'] for r in rows if r['adj_decision']!='KEEP_BASE' and r['adj_label']!=int(r['full_prediction'])]
  for mode in ('confidence','margin','consistency_bonus','decision_margin','support_margin'):
   order=sorted_ids(rows,eligible,mode)
   for k in PREFIXES:add({'family':'C2_STRATIFIED_RANK','base':name,'rank':mode,'k':k},apply_policy(rows,base,order[:min(k,len(order))]))
 print('c2',len(specs),flush=True)
 # C3 source gates, final/source label replacement
 for name,base in conditional_bases.items():
  for source in ('FULL','D3','I1','I2','I3','REVISED'):
   eligible=[r['canonical_id'] for r in rows if r['preferred_source']==source and r['adj_decision']!='KEEP_BASE']
   for mode in ('FINAL','SOURCE'):
    order=sorted_ids(rows,eligible,'margin')
    for k in PREFIXES:add({'family':'C3_SOURCE_GATE','base':name,'source':source,'label_mode':mode,'k':k},apply_policy(rows,base,order[:min(k,len(order))],mode))
 print('c3',len(specs),flush=True)
 # C4 two disjoint strata, fixed pair set and order.
 pairs=(('revise','disagreement'),('revise','agreement'),('accept_reason','disagreement'),('base0','adj0'),('base3','adj3'),('high','support'))
 def ids_for(kind):
  if kind=='revise':return {r['canonical_id'] for r in rows if r['adj_decision']=='REVISE'}
  if kind=='accept_reason':return {r['canonical_id'] for r in rows if r['adj_decision']=='ACCEPT_EXISTING_REASON'}
  if kind=='disagreement':return {r['canonical_id'] for r in rows if not r['full_d3_agreement']}
  if kind=='agreement':return {r['canonical_id'] for r in rows if r['full_d3_agreement']}
  if kind.startswith('base'):return {r['canonical_id'] for r in rows if int(r['full_prediction'])==int(kind[-1])}
  if kind.startswith('adj'):return {r['canonical_id'] for r in rows if int(r['adj_label'])==int(kind[-1])}
  if kind=='high':return {r['canonical_id'] for r in rows if float(r['adj_confidence'])>=.8}
  return {r['canonical_id'] for r in rows if int(r['support'])>=2}
 for name,base in bases.items():
  for a,b in pairs:
   A=sorted_ids(rows,list(ids_for(a)-ids_for(b)),'margin'); B=sorted_ids(rows,list(ids_for(b)-ids_for(a)),'support_margin')
   for k in PREFIXES:add({'family':'C4_TWO_STRATA','base':name,'a':a,'b':b,'k':k},apply_policy(rows,base,A[:min(k,len(A))]+B[:min(k,len(B))]))
 print('c4',len(specs),flush=True)
 # C5 fixed-width deterministic beam proxy: retain 256 best label-free configs, then two registered expansions.
 order=sorted(range(len(specs)),key=lambda i:(-float(np.mean(preds[i]!=full)),specs[i]['candidate_id']))[:256]
 for rnd in (1,2):
  for i in order:
   add({'family':'C5_BEAM','round':rnd,'parent':specs[i]['candidate_id'],'operator':'margin_prefix_20'},preds[i])
 if len(specs)>50000: raise RuntimeError(f'candidate cap {len(specs)}')
 out=RUN/'stage_c/materialized'; out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/'candidate_predictions.npz',predictions=np.vstack(preds),candidate_ids=np.asarray([s['candidate_id'] for s in specs],dtype='U80')); (out/'canonical_ids.json').write_text(json.dumps(ids));
 with open(out/'candidate_manifest.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['candidate_id','config_json']);w.writeheader();w.writerows(specs)
 files={str(p.relative_to(RUN)):sha(p) for p in (out/'candidate_predictions.npz',out/'canonical_ids.json',out/'candidate_manifest.csv',RUN/'protocol/E30_SEARCH_SPACE.json')}; (out/'STAGE_C_MATERIALIZATION_SHA256.json').write_text(json.dumps({'files':files,'candidate_count':len(specs),'canonical_count':n,'gold_in_materialization':False,'reasoner_api_calls':0},indent=2)); print(json.dumps({'candidate_count':len(specs),'canonical_count':n}))
if __name__=='__main__':main()
