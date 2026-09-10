import hashlib, json
import numpy as np

KSET=(0,1,2,3,4,5,8,10,15,20,28,32,50,64,100)
LABELS=range(6)
BASE_KEYS={'FULL':'full_prediction','D3':'d3_prediction','E26':'e26_prediction','E27':'e27_prediction','E29R':'e29r_prediction','E30':'e30_prediction','E31':'e31_prediction','E32':'e32_prediction'}
SOURCE_KEYS={'I1':'i1_label','I2':'i2_label','I3':'i3_label','E29R':'e29r_prediction','E30':'e30_prediction','E31':'e31_prediction','E32':'e32_prediction','base':'base_label'}

def candidate_id(cfg):
    return 'cand_'+hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def majority_label(vals):
    vals=[int(v) for v in vals if int(v)>=0]
    if not vals: return -1
    counts={v:vals.count(v) for v in set(vals)}; mx=max(counts.values()); tied={v for v,c in counts.items() if c==mx}
    for v in (vals[2],vals[1],vals[0]):
        if v in tied:return int(v)
    return min(tied)

def weighted_majority_label(vals,weights):
    scores={}
    for v,w in zip(vals,weights):
        if int(v)>=0:scores[int(v)]=scores.get(int(v),0)+float(w)
    if not scores:return -1
    mx=max(scores.values()); tied={v for v,s in scores.items() if s==mx}
    for v in (int(vals[2]),int(vals[1]),int(vals[0])):
        if v in tied:return v
    return min(tied)

def tuple_mask(rows,t):
    t=tuple(map(int,t)); return np.asarray([(int(r.get('base_label',r.get('full_prediction',-1))),int(r.get('i1_label',-1)),int(r.get('i2_label',-1)),int(r.get('i3_label',-1)))==t for r in rows],dtype=bool)

def decision_mask(rows,t):
    return np.asarray([(r.get('i1_decision',''),r.get('i2_decision',''),r.get('i3_decision',''))==tuple(t) for r in rows],dtype=bool)

def score_array(rows,mode):
    c=np.asarray([float(r.get('adjudicator_confidence',0)) for r in rows]); u=np.asarray([float(r.get('adjudicator_uncertainty',1)) for r in rows]); sup=np.asarray([int(r.get('support',0)) for r in rows]); dec=np.asarray([r.get('adjudicator_decision','')=='ACCEPT_EXISTING_REASON' for r in rows],dtype=float)
    margin=np.asarray([max(float(r.get('i1_confidence',0))-float(r.get('i1_uncertainty',1)),float(r.get('i2_confidence',0))-float(r.get('i2_uncertainty',1)),float(r.get('i3_confidence',0))-float(r.get('i3_uncertainty',1))) for r in rows])
    return {'pool_order':-np.asarray([int(r.get('target_pool_order',9999)) for r in rows]),'confidence':c,'confidence_minus_uncertainty':c-u,'consistency_bonus':c-u+0.1*np.maximum(sup-1,0),'decision_weighted_margin':c-u+0.2*dec,'interface_support_margin':margin+0.25*np.maximum(sup-1,0)}.get(mode,c-u)

def source_label(rows,i,source):
    r=rows[i]
    if source in SOURCE_KEYS:return int(r.get(SOURCE_KEYS[source],-1))
    if source=='majority':return majority_label([r.get('i1_label',-1),r.get('i2_label',-1),r.get('i3_label',-1)])
    if source=='weighted_majority':return weighted_majority_label([r.get('i1_label',-1),r.get('i2_label',-1),r.get('i3_label',-1)],tuple(r.get('_weights',(1,1,1))))
    if source=='adjudicator_final_label':return int(r.get('adjudicator_final_label',-1))
    if source=='preferred_source':
        key={'FULL':'full_prediction','D3':'d3_prediction','I1':'i1_label','I2':'i2_label','I3':'i3_label','REVISED':'adjudicator_final_label'}.get(r.get('adjudicator_preferred_source',''),'adjudicator_final_label'); return int(r.get(key,-1))
    return int(r.get('base_label',r.get('full_prediction',-1)))

def _pick(rows,mask,k,mode):
    idx=[i for i,v in enumerate(mask) if v]; s=score_array(rows,mode); idx.sort(key=lambda i:(-float(s[i]),str(rows[i]['canonical_id']))); return idx[:min(int(k),len(idx))]

def apply(rows,cfg):
    base_name=cfg.get('base','E30'); base=np.asarray([int(r[BASE_KEYS.get(base_name,'e30_prediction')]) for r in rows],dtype=np.int8); out=base.copy(); fam=cfg.get('family')
    if fam=='R0_FIXED_REPLAY': return out
    def rule_mask(rule):
        m=np.ones(len(rows),dtype=bool)
        if rule.get('tuple') is not None:m &= tuple_mask(rows,rule['tuple'])
        if rule.get('decision_tuple') is not None:m &= decision_mask(rows,rule['decision_tuple'])
        if rule.get('pair') is not None:m &= np.asarray([(int(r.get('base_label',-1)),int(r.get('adjudicator_final_label',-1)))==tuple(rule['pair']) for r in rows])
        if rule.get('gate')=='adj_accept':m &= np.asarray([r.get('adjudicator_decision','')=='ACCEPT_EXISTING_REASON' for r in rows])
        if rule.get('gate')=='evidence_high':m &= np.asarray([r.get('evidence_consistency','')=='HIGH' for r in rows])
        if rule.get('gate')=='evidence_medhigh':m &= np.asarray([r.get('evidence_consistency','') in ('MEDIUM','HIGH') for r in rows])
        if rule.get('full_eq_d3') is True:m &= np.asarray([int(r['full_prediction'])==int(r['d3_prediction']) for r in rows])
        if rule.get('full_eq_d3') is False:m &= np.asarray([int(r['full_prediction'])!=int(r['d3_prediction']) for r in rows])
        if 'confidence' in rule:m &= np.asarray([float(r.get('adjudicator_confidence',0))>=float(rule['confidence']) for r in rows])
        if 'uncertainty_max' in rule:m &= np.asarray([float(r.get('adjudicator_uncertainty',1))<=float(rule['uncertainty_max']) for r in rows])
        if rule.get('target_pool_order') is not None:m &= np.asarray([int(r.get('target_pool_order',9999))<=int(rule['target_pool_order']) for r in rows])
        return m
    def apply_rule(rule,available=None):
        m=rule_mask(rule)
        if available is not None:m &= available
        sel=_pick(rows,m,rule.get('k',0),rule.get('rank','confidence_minus_uncertainty')); src=rule.get('source','base')
        for i in sel:
            v=source_label(rows,i,src)
            if v in LABELS:out[i]=v
        return np.asarray(sel,dtype=int)
    if fam=='R2_VOTE':
        m=np.ones(len(rows),dtype=bool); cond=cfg.get('vote_condition')
        if cond=='i2_eq_i3':m &= np.asarray([r['i2_label']==r['i3_label'] for r in rows])
        elif cond=='i1_eq_i2':m &= np.asarray([r['i1_label']==r['i2_label'] for r in rows])
        elif cond=='i1_eq_i3':m &= np.asarray([r['i1_label']==r['i3_label'] for r in rows])
        elif cond=='unanimous':m &= np.asarray([r['i1_label']==r['i2_label']==r['i3_label'] for r in rows])
        if cfg.get('support') is not None:m &= np.asarray([max([r['i1_label'],r['i2_label'],r['i3_label']].count(x) for x in set([r['i1_label'],r['i2_label'],r['i3_label']]))>=int(cfg['support']) for r in rows])
        for i in _pick(rows,m,cfg.get('k',0),cfg.get('rank','confidence_minus_uncertainty')):
            vals=[rows[i]['i1_label'],rows[i]['i2_label'],rows[i]['i3_label']]; out[i]=weighted_majority_label(vals,cfg.get('weights',(1,1,1))) if cfg.get('weighted') else majority_label(vals)
    elif fam=='R5_TWO_RULE':
        taken=np.ones(len(rows),dtype=bool); first=apply_rule(cfg['rules'][0]); taken[first]=False; apply_rule(cfg['rules'][1],taken)
    elif fam in ('R1_RAW_TUPLE','R3_DECISION_PATTERN','R4_ADJUDICATOR_GATE','R6_SEEDED'):
        apply_rule(cfg)
    return out

def compose_two_rules(rows,cfg): return apply(rows,{'family':'R5_TWO_RULE','base':'FULL','rules':cfg['rules']})
