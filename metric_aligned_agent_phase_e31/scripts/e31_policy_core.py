import hashlib,json
import numpy as np
LABELS=range(6)
KSET=[1,2,3,5,8,10,15,20,28,35,43,50,60,84,100,150,200]
def action(pred,full,d3):
 return np.where(pred==full,'KEEP',np.where((pred==d3)&(pred!=full),'RESIDUAL','REASON'))
def cid(cfg):
 return 'cand_'+hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def base_array(rows,name):
 key={'FULL':'full_prediction','D3':'d3_prediction','E26':'e26_prediction','E27':'e27_prediction','E29R':'e29r_prediction','E30':'e30_prediction'}.get(name,'e29r_prediction')
 return np.asarray([int(r[key]) for r in rows],dtype=np.int8)
def strata_mask(rows,name,value=None):
 m=np.ones(len(rows),dtype=bool)
 if name=='base_label':m=np.asarray([int(r['base_label'])==int(value) for r in rows])
 elif name=='adj_label':m=np.asarray([int(r['adjudicator_final_label'])==int(value) for r in rows])
 elif name=='agreement':m=np.asarray([int(r['full_prediction'])==int(r['d3_prediction']) for r in rows])
 elif name=='disagreement':m=~np.asarray([int(r['full_prediction'])==int(r['d3_prediction']) for r in rows])
 elif name=='adj_decision':m=np.asarray([r['adjudicator_decision']==value for r in rows])
 elif name=='evidence':m=np.asarray([r['evidence_consistency']==value for r in rows])
 elif name=='source':m=np.asarray([r['adjudicator_preferred_source']==value for r in rows])
 elif name=='label_equal':m=np.asarray([int(r['base_label'])==int(r['adjudicator_final_label']) for r in rows])
 elif name=='label_diff':m=~np.asarray([int(r['base_label'])==int(r['adjudicator_final_label']) for r in rows])
 return m
def rank_scores(rows,mode):
 c=np.asarray([float(r['adjudicator_confidence']) for r in rows]); u=np.asarray([float(r['adjudicator_uncertainty']) for r in rows]); s=np.asarray([float(r['support']) for r in rows]);
 if mode=='confidence':return c
 if mode=='uncertainty':return -u
 if mode=='consistency':return c-u+0.10*np.maximum(s-1,0)
 if mode=='decision_margin':return c-u+0.10*(np.asarray([r['adjudicator_decision']!='KEEP_BASE' for r in rows],dtype=float))
 if mode=='support_margin':return c-u+0.25*np.maximum(s-1,0)
 if mode=='target_order':return -np.asarray([int(r['target_pool_order']) for r in rows])
 return c-u
def apply_policy(rows, cfg):
 base=base_array(rows,cfg.get('base','E30')); pred=base.copy(); n=len(rows); eligible=np.asarray([int(r['adjudicator_final_label'])>=0 for r in rows])
 labels=np.asarray([int(r['adjudicator_final_label']) if int(r['adjudicator_final_label'])>=0 else -1 for r in rows]); base_label=base.copy(); work=[dict(r,base_label=int(base[i])) for i,r in enumerate(rows)]
 fam=cfg.get('family','C0_FIXED')
 if fam=='C0_FIXED':
  if cfg.get('name')=='D3':return base
  return base
 if fam=='C1_E30_LOCAL_BOUNDARY':
  a=cfg.get('a','accept_reason'); b=cfg.get('b','any'); k=int(cfg.get('k',20));
  if a=='accept_reason':eligible &= np.asarray([r['adjudicator_decision'] in ('ACCEPT_EXISTING_REASON','REVISE') for r in rows])
  elif a=='revise':eligible &= np.asarray([r['adjudicator_decision']=='REVISE' for r in rows])
  elif a=='nonkeep':eligible &= np.asarray([r['adjudicator_decision']!='KEEP_BASE' for r in rows])
  elif a=='keep_base':eligible &= np.asarray([r['adjudicator_decision']=='KEEP_BASE' for r in rows])
  if b=='agreement':eligible &= strata_mask(work,'agreement')
  elif b=='disagreement':eligible &= strata_mask(work,'disagreement')
  score=rank_scores(rows,cfg.get('rank','margin')); return replace_prefix(pred,labels,eligible,score,k,[r['canonical_id'] for r in rows])
 if fam in ('C2_STRATIFIED_LABEL_GATE','C3_SOURCE_CONDITIONAL','C4_TWO_STRATA_LOCAL','C5_LABEL_PAIR_CONDITIONAL'):
  k=int(cfg.get('k',20)); mask=eligible.copy();
  for s in cfg.get('strata',[]):mask &= strata_mask(work,s[0],s[1] if len(s)>1 else None)
  if cfg.get('pair') is not None:mask &= (base==int(cfg['pair'][0])) & (labels==int(cfg['pair'][1]))
  conf=float(cfg.get('confidence',0)); umax=float(cfg.get('uncertainty_max',1)); mask &= np.asarray([float(r['adjudicator_confidence'])>=conf and float(r['adjudicator_uncertainty'])<=umax for r in rows])
  score=rank_scores(rows,cfg.get('rank','margin')); selected=prefix_mask(mask,score,k,[r['canonical_id'] for r in rows])
  source=cfg.get('output','final'); repl=labels.copy()
  if source=='preferred':
   mp={'FULL':'full_prediction','D3':'d3_prediction','I1':'i1_label','I2':'i2_label','I3':'i3_label','REVISED':'adjudicator_final_label'}
   for i,r in enumerate(rows):
    key=mp.get(r['adjudicator_preferred_source']); repl[i]=int(r[key]) if key and int(r[key])>=0 else labels[i]
  pred[selected]=repl[selected]; return pred
 if fam=='C6_LOCAL_BEAM':
  parent=cfg.get('parent','E30'); inner=dict(cfg); inner['family']='C1_E30_LOCAL_BOUNDARY'; inner['base']=parent; return apply_policy(rows,inner)
 return pred
def prefix_mask(mask,score,k,tie=None):
 idx=[i for i,v in enumerate(mask) if v]; idx.sort(key=lambda i:(-float(score[i]),str(i)))
 if tie is not None: idx.sort(key=lambda i:(-float(score[i]),str(tie[i])))
 out=np.zeros(len(mask),dtype=bool); out[idx[:min(k,len(idx))]]=True; return out
def replace_prefix(pred,labels,mask,score,k,tie=None):
 out=pred.copy(); sel=prefix_mask(mask,score,k,tie); out[sel]=labels[sel]; return out
