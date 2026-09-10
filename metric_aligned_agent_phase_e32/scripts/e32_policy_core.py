import hashlib,json
import numpy as np
KSET=[0,1,2,3,4,5,8,10,15,20,28,35,43,50,60,84,100,150,200]
def cid(cfg):return 'cand_'+hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rank_score(rows,mode):
 c=np.asarray([float(r['adjudicator_confidence']) for r in rows]); u=np.asarray([float(r['adjudicator_uncertainty']) for r in rows]); s=np.asarray([float(r['support']) for r in rows]); d=np.asarray([r['adjudicator_decision'] not in ('KEEP_BASE','') for r in rows],dtype=float)
 return {'pool_order':-np.asarray([int(r['target_pool_order']) for r in rows]),'confidence':c,'confidence_minus_uncertainty':c-u,'consistency_bonus':c-u+0.1*np.maximum(s-1,0),'decision_weighted_margin':c-u+0.2*d,'interface_support_margin':c-u+0.25*np.maximum(s-1,0)}.get(mode,c-u)
def action(p,full,d3):return np.where(p==full,'KEEP',np.where((p==d3)&(p!=full),'RESIDUAL','REASON'))
def mask_stratum(rows,s,val=None):
 if s=='full_eq_d3':return np.asarray([int(r['full_prediction'])==int(r['d3_prediction']) for r in rows])
 if s=='full_neq_d3':return ~np.asarray([int(r['full_prediction'])==int(r['d3_prediction']) for r in rows])
 if s=='adj_accept':return np.asarray([r['adjudicator_decision']=='ACCEPT_EXISTING_REASON' for r in rows])
 if s=='adj_revise':return np.asarray([r['adjudicator_decision']=='REVISE' for r in rows])
 if s=='evidence_high':return np.asarray([r['evidence_consistency']=='HIGH' for r in rows])
 if s=='evidence_medhigh':return np.asarray([r['evidence_consistency'] in ('MEDIUM','HIGH') for r in rows])
 if s=='cheap_label':return np.asarray([int(r['base_label'])==int(val) for r in rows])
 if s=='adj_label':return np.asarray([int(r['adjudicator_final_label'])==int(val) for r in rows])
 if s=='source':return np.asarray([r['adjudicator_preferred_source']==val for r in rows])
 if s=='pair':return np.asarray([int(r['base_label'])==int(val[0]) and int(r['adjudicator_final_label'])==int(val[1]) for r in rows])
 return np.ones(len(rows),dtype=bool)
def apply(rows,cfg):
 key={'FULL':'full_prediction','D3':'d3_prediction','E26':'e26_prediction','E27':'e27_prediction','E29R':'e29r_prediction','E30':'e30_prediction','E31':'e31_prediction'}.get(cfg.get('base','E30'),'e30_prediction'); base=np.asarray([int(r[key]) for r in rows]); full=np.asarray([int(r['full_prediction']) for r in rows]); d3=np.asarray([int(r['d3_prediction']) for r in rows]); labels=np.asarray([int(r['adjudicator_final_label']) for r in rows]); work=[dict(r,base_label=int(base[i])) for i,r in enumerate(rows)]; eligible=labels>=0
 fam=cfg.get('family')
 if fam=='R0_FIXED_REPLAY':return base
 if fam in ('R1_SINGLE_LABEL_PAIR','R2_PAIR_LOCAL_BOUNDARY','R4_CONDITIONAL_PAIR_GATE','R5_BASE_COMPARISON','R6_LOCAL_BEAM'):
  pair=cfg.get('pair'); m=eligible.copy()
  if pair is not None:m &= mask_stratum(work,'pair',pair)
  for s,v in cfg.get('gates',[]):m &= mask_stratum(work,s,v)
  conf=float(cfg.get('confidence',0.0)); umax=float(cfg.get('uncertainty_max',1.0)); m &= np.asarray([float(r['adjudicator_confidence'])>=conf and float(r['adjudicator_uncertainty'])<=umax for r in rows])
  score=rank_score(rows,cfg.get('rank','confidence_minus_uncertainty')); idx=[i for i,v in enumerate(m) if v]; idx.sort(key=lambda i:(-float(score[i]),str(rows[i]['canonical_id']))); selected=idx[:min(int(cfg.get('k',0)),len(idx))]; out=base.copy(); replacement=np.asarray(labels if cfg.get('output','final')=='final' else [int(r['full_prediction']) if r['adjudicator_preferred_source']=='FULL' else int(r['d3_prediction']) if r['adjudicator_preferred_source']=='D3' else int(r['adjudicator_final_label']) for r in rows]); out[selected]=replacement[selected]; return out
 if fam=='R3_TWO_LABEL_PAIRS':
  out=base.copy(); score=rank_score(rows,cfg.get('rank','confidence_minus_uncertainty')); used=set()
  for pair,k in zip(cfg.get('pairs',[]),cfg.get('ks',[0,0])):
   m=eligible & mask_stratum(work,'pair',pair); m &= ~np.asarray([i in used for i in range(len(rows))]); idx=[i for i,v in enumerate(m) if v]; idx.sort(key=lambda i:(-float(score[i]),str(rows[i]['canonical_id']))); take=idx[:min(int(k),len(idx))]; out[take]=labels[take]; used.update(take)
  return out
 return base
