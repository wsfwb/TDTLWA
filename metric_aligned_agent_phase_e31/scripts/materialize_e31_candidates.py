import csv,json,hashlib
from pathlib import Path
import numpy as np
from .e31_policy_core import apply_policy,cid,KSET
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E29R=ROOT/'experiment_outputs/metric_aligned_agent_phase_e29r/run_20260822T_continuation_one_call_stage_c_v1'
def rr(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def add(cfg,rows,arrs,man,seen):
 c=cid(cfg)
 if c in seen:return
 seen.add(c); arrs.append(apply_policy(rows,cfg).astype(np.int8)); man.append({'candidate_id':c,'config_json':json.dumps(cfg,sort_keys=True,separators=(',',':')),'family':cfg.get('family'),'base_id':cfg.get('base','E30')})
def main():
 rows=json.loads((RUN/'stage_c/features.json').read_text()); arrs=[]; man=[]; seen=set(); out=RUN/'stage_c/materialized'; out.mkdir(parents=True,exist_ok=True)
 # Fixed references, including the frozen historical Stage-A top ten identities.
 for n in ('FULL','D3','E26','E27','E29R','E30'):add({'family':'C0_FIXED','name':n,'base':n},rows,arrs,man,seen)
 top=rr(E29R/'stage_a/results/stage_a_candidate_leaderboard.csv')[:10]
 for j,r in enumerate(top):add({'family':'C0_FIXED','name':'E29_STAGEA_TOP10','rank':j,'source_candidate_id':r['candidate_id'],'base':'E29R'},rows,arrs,man,seen)
 # One-factor local boundary around E30.
 for a in ('accept_reason','revise','nonkeep','keep_base'):
  for b in ('agreement','disagreement','any'):
   for k in (15,18,19,20,21,22,25,28,30,35):add({'family':'C1_E30_LOCAL_BOUNDARY','base':'E30','a':a,'b':b,'k':k,'rank':'margin'},rows,arrs,man,seen)
 confs=(0.6,0.7,0.8,0.9); uncs=(0.2,0.3,0.4,1.0)
 strata=[('agreement',None),('disagreement',None),('label_equal',None),('label_diff',None)]+[('base_label',i) for i in range(6)]+[('adj_label',i) for i in range(6)]+[('adj_decision',x) for x in ('KEEP_BASE','REVISE','ACCEPT_EXISTING_REASON')]+[('evidence',x) for x in ('LOW','MEDIUM','HIGH')]+[('source',x) for x in ('FULL','D3','I1','I2','I3','REVISED')]
 for s in strata:
  for conf in confs:
   for u in uncs:
    for k in KSET:add({'family':'C2_STRATIFIED_LABEL_GATE','base':'E30','strata':[list(s)],'confidence':conf,'uncertainty_max':u,'k':k,'rank':'margin','output':'final'},rows,arrs,man,seen)
 # Source-conditional replacement.
 for src in ('FULL','D3','I1','I2','I3','REVISED','ANY'):
  for output in ('final','preferred'):
   for conf in (0.6,0.7,0.8,0.9):
    for k in KSET:add({'family':'C3_SOURCE_CONDITIONAL','base':'E30','source':src,'output':output,'confidence':conf,'uncertainty_max':1.0,'k':k,'rank':'support_margin'},rows,arrs,man,seen)
 # Two-strata local composition, bounded to registered broad strata.
 pairs=[(('disagreement',None),('adj_decision','REVISE')),(('disagreement',None),('evidence','HIGH')),(('label_diff',None),('source','REVISED')),(('agreement',None),('adj_decision','KEEP_BASE')),(('base_label',0),('adj_label',1)),(('base_label',1),('adj_label',0)),(('label_equal',None),('evidence','HIGH')), (('label_diff',None),('evidence','MEDIUM'))]
 for s1,s2 in pairs:
  for rank in ('margin','confidence','consistency'):
   for k in KSET:add({'family':'C4_TWO_STRATA_LOCAL','base':'E30','strata':[list(s1),list(s2)],'k':k,'rank':rank,'confidence':0.6,'uncertainty_max':1.0},rows,arrs,man,seen)
 # Single registered label pairs only.
 for bl in range(6):
  for al in range(6):
   for k in KSET:add({'family':'C5_LABEL_PAIR_CONDITIONAL','base':'E30','pair':[bl,al],'k':k,'rank':'margin','confidence':0.6,'uncertainty_max':1.0,'output':'final'},rows,arrs,man,seen)
 # Deterministic local beam: registered C1/C2 operators applied to five frozen bases.
 for base in ('E30','E29R','E27','E26'):
  for round_no in (1,2):
   for a in ('accept_reason','revise','nonkeep','keep_base'):
    for b in ('agreement','disagreement','any'):
     for k in KSET[:10]:add({'family':'C6_LOCAL_BEAM','base':base,'parent':base,'round':round_no,'a':a,'b':b,'k':k,'rank':'margin'},rows,arrs,man,seen)
 if len(man)>50000:raise RuntimeError('candidate limit exceeded')
 z=np.stack(arrs,axis=0); np.savez_compressed(out/'candidate_predictions.npz',predictions=z); (out/'canonical_ids.json').write_text(json.dumps([r['canonical_id'] for r in rows]));
 with open(out/'candidate_manifest.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=['candidate_id','config_json','family','base_id']);w.writeheader();w.writerows(man)
 (out/'MATERIALIZATION_SHA256.json').write_text(json.dumps({'files':{'stage_c/materialized/candidate_predictions.npz':sha(out/'candidate_predictions.npz'),'stage_c/materialized/canonical_ids.json':sha(out/'canonical_ids.json'),'stage_c/materialized/candidate_manifest.csv':sha(out/'candidate_manifest.csv'),'protocol/E31_SEARCH_SPACE.json':sha(RUN/'protocol/E31_SEARCH_SPACE.json')},'candidate_count':len(man),'canonical_count':len(rows),'gold_in_materialization':False,'reasoner_api_calls':0},indent=2));print(json.dumps({'candidate_count':len(man),'canonical_count':len(rows),'shape':list(z.shape)}))
if __name__=='__main__':main()
