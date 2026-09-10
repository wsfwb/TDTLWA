import csv,json,hashlib
from pathlib import Path
import numpy as np
from .e33_policy_core import apply,candidate_id,KSET
RUN=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def add(rows,arr,man,seen,cfg):
    cfg=dict(cfg); cid=candidate_id(cfg)
    if cid in seen:return
    seen.add(cid); arr.append(apply(rows,cfg).astype(np.int8)); man.append({'candidate_id':cid,'config_json':json.dumps(cfg,sort_keys=True,separators=(',',':')),'family':cfg.get('family',''),'base_id':cfg.get('base','')})
def main():
    rows=json.loads((RUN/'stage_c/features.json').read_text()); out=RUN/'stage_c/materialized'; out.mkdir(parents=True,exist_ok=True); arr=[]; man=[]; seen=set()
    for b in ('FULL','D3','E26','E27','E29R','E30','E31','E32'):add(rows,arr,man,seen,{'family':'R0_FIXED_REPLAY','base':b,'name':b})
    tuples=sorted({(int(r['full_prediction']),int(r['i1_label']),int(r['i2_label']),int(r['i3_label'])) for r in rows})
    sources=('I1','I2','I3','majority','weighted_majority','adjudicator_final_label','E29R','E30','E31','base')
    # Raw tuple enumeration is complete over observed tuples, sources and k;
    # ranking variants are registered in the adjudicator-gated family below.
    ranks=('confidence_minus_uncertainty',)
    for t in tuples:
        for src in sources:
            for rank in ranks:
                for k in KSET:
                    cfg={'family':'R1_RAW_TUPLE','base':'E31','tuple':list(t),'source':src,'rank':rank,'k':k}
                    if src=='weighted_majority':cfg['weights']=[1,1,1]
                    add(rows,arr,man,seen,cfg)
    patterns=sorted({(r.get('i1_decision',''),r.get('i2_decision',''),r.get('i3_decision','')) for r in rows})
    for pat in patterns:
        for src in ('I1','I2','I3','majority','weighted_majority'):
            for k in KSET:
                add(rows,arr,man,seen,{'family':'R3_DECISION_PATTERN','base':'E31','decision_tuple':list(pat),'source':src,'rank':'confidence_minus_uncertainty','k':k,'weights':[1,1,1] if src=='weighted_majority' else None})
    for cond in ('i2_eq_i3','i1_eq_i2','i1_eq_i3','unanimous'):
        for weighted,weights in ((False,[1,1,1]),(True,[1,1,1]),(True,[2,1,1]),(True,[1,2,1]),(True,[1,1,2])):
            for support in (None,1,2,3):
                for k in KSET:
                    cfg={'family':'R2_VOTE','base':'E31','vote_condition':cond,'weighted':weighted,'weights':weights,'support':support,'rank':'confidence_minus_uncertainty','k':k}; add(rows,arr,man,seen,cfg)
    # Adjudicator-only gates use fixed registered thresholds and pool order.
    for gate in ('full_eq_d3','full_neq_d3','adj_accept','evidence_high','evidence_medhigh'):
        for conf in (.5,.6,.7,.8,.9):
            for umax in (.2,.4,.6,1.0):
                for order in (1,2,3,5,10,20,50,100,150,200):
                    for k in (0,1,2,3,5,10,20):
                        add(rows,arr,man,seen,{'family':'R4_ADJUDICATOR_GATE','base':'E32','source':'adjudicator_final_label','gate':gate,'full_eq_d3':True if gate=='full_eq_d3' else False if gate=='full_neq_d3' else None,'confidence':conf,'uncertainty_max':umax,'target_pool_order':order,'rank':'confidence_minus_uncertainty','k':k})
    # Two mutually exclusive tuple rules, limited to the 30 most frequent views.
    freq={t:sum(1 for r in rows if (int(r['full_prediction']),int(r['i1_label']),int(r['i2_label']),int(r['i3_label']))==t) for t in tuples}; top=sorted(tuples,key=lambda t:(-freq[t],t))[:30]
    for ai,a in enumerate(top):
        for b in top[ai+1:]:
            for s1,s2 in (('I1','I2'),('I1','I3'),('majority','I3')):
                for k1,k2 in ((1,1),(2,1),(1,2),(2,2),(3,2),(2,3)):
                    add(rows,arr,man,seen,{'family':'R5_TWO_RULE','base':'E31','rules':[{'tuple':list(a),'source':s1,'rank':'confidence_minus_uncertainty','k':k1},{'tuple':list(b),'source':s2,'rank':'confidence_minus_uncertainty','k':k2}]})
    # Explicitly marked post-E32 diagnostics, retained but never hidden.
    for base,vals in [('E31',(1,3,5,5)),('E31',(5,1,5,5))]:add(rows,arr,man,seen,{'family':'R6_SEEDED','base':base,'tuple':list(vals),'source':'I1','rank':'pool_order','k':1,'seeded_after_e32':True})
    if len(man)>100000:raise RuntimeError(f'candidate cap exceeded: {len(man)}')
    z=np.stack(arr); np.savez_compressed(out/'candidate_predictions.npz',predictions=z); (out/'canonical_ids.json').write_text(json.dumps([r['canonical_id'] for r in rows]))
    with open(out/'candidate_manifest.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['candidate_id','config_json','family','base_id']); w.writeheader(); w.writerows(man)
    material={'files':{'stage_c/materialized/candidate_predictions.npz':sha(out/'candidate_predictions.npz'),'stage_c/materialized/canonical_ids.json':sha(out/'canonical_ids.json'),'stage_c/materialized/candidate_manifest.csv':sha(out/'candidate_manifest.csv'),'protocol/E33_SEARCH_SPACE.json':sha(RUN/'protocol/E33_SEARCH_SPACE.json')},'candidate_count':len(man),'canonical_count':len(rows),'gold_in_materialization':False,'reasoner_api_calls':0}
    (out/'MATERIALIZATION_SHA256.json').write_text(json.dumps(material,indent=2)); print(json.dumps({'candidate_count':len(man),'canonical_count':len(rows),'shape':list(z.shape),'gold_in_materialization':False,'reasoner_api_calls':0}))
if __name__=='__main__':main()
