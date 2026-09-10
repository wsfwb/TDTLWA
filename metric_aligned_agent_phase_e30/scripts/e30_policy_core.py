import hashlib,json
import numpy as np
LABELS={'hap':0,'sad':1,'neu':2,'ang':3,'exc':4,'fru':5}
PREFIXES=(5,10,15,20,28,35,43,50,60,84,100,150,200)
CONF=(0.6,0.7,0.8,0.9); UNC=(0.2,0.3,0.4,1.0)
def candidate_id(config): return 'cand_'+hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def action_provenance(pred,full,d3): return np.where(pred==full,'KEEP',np.where(pred==d3,'RESIDUAL','REASON'))
def score(row,mode):
 c=float(row.get('adj_confidence',0)); u=float(row.get('adj_uncertainty',1)); s=int(row.get('support',0)); b={'HIGH':.2,'MEDIUM':.1,'LOW':0}.get(row.get('evidence_consistency'),0)
 return {'confidence':c,'margin':c-u,'consistency_bonus':c-u+b,'decision_margin':c-u+(.1 if row.get('adj_decision')=='REVISE' else 0),'support_margin':c-u+.1*s}.get(mode,c-u)
def sorted_ids(rows,ids,mode):
 by={r['canonical_id']:r for r in rows}; return sorted(ids,key=lambda c:(-score(by[c],mode),c))
def apply_policy(rows,base,ids,label_mode='FINAL'):
 out=np.asarray(base,dtype=np.int8).copy(); idx={r['canonical_id']:i for i,r in enumerate(rows)}
 for c in ids:
  if c in idx: out[idx[c]]=int(rows[idx[c]]['source_label'] if label_mode=='SOURCE' else rows[idx[c]]['adj_label'])
 return out
