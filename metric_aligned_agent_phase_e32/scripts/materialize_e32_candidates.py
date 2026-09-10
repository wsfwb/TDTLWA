import csv,json,hashlib
from pathlib import Path
import numpy as np
from .e32_policy_core import apply,cid,KSET
RUN=Path(__file__).resolve().parents[1]
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def add(rows,arr,man,seen,cfg):
 c=cid(cfg)
 if c in seen:return
 seen.add(c); arr.append(apply(rows,cfg).astype(np.int8)); man.append({'candidate_id':c,'config_json':json.dumps(cfg,sort_keys=True,separators=(',',':')),'family':cfg['family'],'base_id':cfg.get('base','E30')})
def main():
 rows=json.loads((RUN/'stage_c/features.json').read_text()); arr=[]; man=[]; seen=set(); out=RUN/'stage_c/materialized'; out.mkdir(parents=True,exist_ok=True)
 for b in ('FULL','D3','E26','E27','E29R','E30','E31'):add(rows,arr,man,seen,{'family':'R0_FIXED_REPLAY','name':b,'base':b})
 for j in range(10):add(rows,arr,man,seen,{'family':'R0_FIXED_REPLAY','name':'E29_STAGEA_TOP10','rank':j,'base':'E29R'})
 all_pairs=sorted({(int(r['full_prediction']),int(r['adjudicator_final_label'])) for r in rows if int(r['adjudicator_final_label'])>=0})
 # The registered grid is bounded by max_candidates; deterministic frequency-priority
 # selection keeps all E31-relevant high-support pairs and reserves space for R2-R6.
 freq_all={p:sum(1 for r in rows if (int(r['full_prediction']),int(r['adjudicator_final_label']))==p) for p in all_pairs}
 pairs=sorted(all_pairs,key=lambda p:(-freq_all[p],p))[:8]
 if not pairs:pairs=[(3,5)]
 for pair in pairs:
  for conf in (0.5,0.6,0.7,0.8,0.9):
   for u in (1.0,):
    for rank in ('pool_order','confidence','confidence_minus_uncertainty','consistency_bonus','decision_weighted_margin','interface_support_margin'):
     for outmode in ('final','preferred_source'):
      for k in KSET:add(rows,arr,man,seen,{'family':'R1_SINGLE_LABEL_PAIR','base':'E30','pair':list(pair),'confidence':conf,'uncertainty_max':u,'rank':rank,'k':k,'output':outmode})
 # E31 local boundary around its registered pair.
 for k in (0,1,2,3,4,5,8,10,15,20):
  for conf in (0.5,0.6,0.7,0.8):
   for u in (0.6,1.0):
    for rank in ('confidence','confidence_minus_uncertainty','decision_weighted_margin','interface_support_margin'):
     for outmode in ('final','preferred_source'):add(rows,arr,man,seen,{'family':'R2_PAIR_LOCAL_BOUNDARY','base':'E30','pair':[3,5],'confidence':conf,'uncertainty_max':u,'rank':rank,'k':k,'output':outmode})
 # Deterministic two-pair compositions from the most frequent prediction-only pairs.
 freq={p:sum(1 for r in rows if (int(r['full_prediction']),int(r['adjudicator_final_label']))==p) for p in pairs}; top=sorted(pairs,key=lambda p:(-freq[p],p))[:10]
 for ai,a in enumerate(top):
  for b in top[ai+1:]:
   for k1,k2 in ((1,1),(2,1),(1,2),(2,2),(3,2),(2,3)):add(rows,arr,man,seen,{'family':'R3_TWO_LABEL_PAIRS','base':'E30','pairs':[list(a),list(b)],'ks':[k1,k2],'rank':'confidence_minus_uncertainty'})
 # One registered conditional gate.
 gates=[('full_eq_d3',None),('full_neq_d3',None),('adj_accept',None),('adj_revise',None),('evidence_high',None),('evidence_medhigh',None),('source','FULL'),('source','D3'),('source','I3'),('cheap_label',3),('adj_label',5)]
 for gate in gates:
  for base in ('E31','E30','E29R','E27','E26'):
   for k in (0,1,2,3,5,8,10,15,20):add(rows,arr,man,seen,{'family':'R4_CONDITIONAL_PAIR_GATE','base':base,'pair':[3,5],'gates':[list(gate)],'confidence':0.6,'uncertainty_max':1.0,'rank':'confidence_minus_uncertainty','k':k,'output':'final'})
 for base in ('E31','E30','E29R','E27','E26'):
  for pair in top:
   for k in (0,1,2,3,5,10,20,28,50):add(rows,arr,man,seen,{'family':'R5_BASE_COMPARISON','base':base,'pair':list(pair),'confidence':0.6,'uncertainty_max':1.0,'rank':'confidence_minus_uncertainty','k':k,'output':'final'})
  for rd in (1,2):
   for rank in ('confidence','confidence_minus_uncertainty','decision_weighted_margin','interface_support_margin'):
    for k in (0,1,2,3,5,10,20,28):add(rows,arr,man,seen,{'family':'R6_LOCAL_BEAM','base':base,'pair':[3,5],'confidence':0.6,'uncertainty_max':1.0,'rank':rank,'k':k,'output':'final','beam_round':rd})
 if len(man)>50000:raise RuntimeError('candidate cap exceeded')
 z=np.stack(arr); np.savez_compressed(out/'candidate_predictions.npz',predictions=z); (out/'canonical_ids.json').write_text(json.dumps([r['canonical_id'] for r in rows]))
 with open(out/'candidate_manifest.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=['candidate_id','config_json','family','base_id']);w.writeheader();w.writerows(man)
 (out/'MATERIALIZATION_SHA256.json').write_text(json.dumps({'files':{'stage_c/materialized/candidate_predictions.npz':sha(out/'candidate_predictions.npz'),'stage_c/materialized/canonical_ids.json':sha(out/'canonical_ids.json'),'stage_c/materialized/candidate_manifest.csv':sha(out/'candidate_manifest.csv'),'protocol/E32_SEARCH_SPACE.json':sha(RUN/'protocol/E32_SEARCH_SPACE.json')},'candidate_count':len(man),'canonical_count':len(rows),'gold_in_materialization':False,'reasoner_api_calls':0},indent=2));print(json.dumps({'candidate_count':len(man),'canonical_count':len(rows),'shape':list(z.shape),'pairs':len(pairs)}))
if __name__=='__main__':main()
