import csv, json, hashlib
from pathlib import Path
from .stage_a_views import build_views

RUN = Path(__file__).resolve().parents[1]
E26 = RUN.parents[2] / "experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1"
E27 = RUN.parents[2] / "experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1"
FORBIDDEN = ("gold", "outcome", "correctness", "benefit", "harm", "wf1", "metric", "oracle")

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
def sha256(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def clean_rows():
    raw=read_csv(E26/"evaluation/results/e26_canonical_predictions.csv")
    return [{k:v for k,v in r.items() if not any(x in k.lower() for x in FORBIDDEN)} for r in raw]

def main():
    rows=clean_rows()
    if len(rows)!=1623 or len({r['canonical_id'] for r in rows})!=1623: raise RuntimeError('canonical alignment')
    views=build_views(rows)
    by={r['canonical_id']:r for r in rows}
    full={r['canonical_id']:int(r['full_prediction']) for r in rows}
    # E27 W28 is the first 28 V_ANY interventions, with the locked ranking.
    e27_w28=[x['canonical_id'] for x in views['V_ANY'][:28]]
    winner=read_csv(RUN/"stage_a/results/stage_a_winner_predictions.csv")
    stage_a_pred={r['canonical_id']:int(r['prediction']) for r in winner}
    stage_a_interventions=[c for c in stage_a_pred if stage_a_pred[c]!=full[c]]
    seen=set(); selected=[]
    def add(cid,bucket,rank,meta):
        if cid in seen or cid not in by: return
        seen.add(cid); selected.append({'canonical_id':cid,'bucket':bucket,'bucket_rank':rank,**meta})
    # Bucket 1.
    for i,c in enumerate(e27_w28): add(c,'e27_w28',i,{'support':3,'max_margin':0.0})
    # Bucket 2: remaining V_ANY, fixed order.
    for i,x in enumerate(views['V_ANY'][28:84],28): add(x['canonical_id'],'v_any_remaining',i,{'support':x['support'],'max_margin':x['max_margin']})
    # Bucket 3: Stage-A interventions not already selected.
    for i,c in enumerate(stage_a_interventions): add(c,'stage_a_intervention',i,{'support':0,'max_margin':0.0})
    # Bucket 4: union by support count, then margin.
    union={}
    for name,v in views.items():
        for x in v:
            z=union.setdefault(x['canonical_id'],{'support':0,'max_margin':-1.0})
            z['support']+=1; z['max_margin']=max(z['max_margin'],x['max_margin'])
    for i,(c,z) in enumerate(sorted(union.items(),key=lambda kv:(-kv[1]['support'],-kv[1]['max_margin'],kv[0]))): add(c,'view_union',i,z)
    # Bucket 5: disagreement and no majority, uncertainty ascending/descending as locked diagnostic.
    no_major=[]
    for r in rows:
        labs=[int(r['i1_reason_prediction']),int(r['i2_reason_prediction']),int(r['i3_reason_prediction'])]
        if int(r['full_prediction'])!=int(r['residual_prediction']) and len(set(labs))>1:
            u=(float(r['i1_uncertainty'])+float(r['i2_uncertainty'])+float(r['i3_uncertainty']))/3
            no_major.append((u,r['canonical_id']))
    for i,(u,c) in enumerate(sorted(no_major,key=lambda z:(z[0],z[1]))): add(c,'no_majority_disagreement',i,{'support':0,'max_margin':-u})
    # Bucket 6 deterministic fallback.
    fallback=[]
    for r in rows:
        labels={int(r['i1_reason_prediction']),int(r['i2_reason_prediction']),int(r['i3_reason_prediction'])}
        fallback.append((-int(int(r['full_prediction'])!=int(r['residual_prediction'])),-len(labels),float(r['i1_uncertainty'])+float(r['i2_uncertainty'])+float(r['i3_uncertainty']),r['canonical_id']))
    for i,(_,_,u,c) in enumerate(sorted(fallback)): add(c,'deterministic_fallback',i,{'support':0,'max_margin':-u})
    if len(selected)<200: raise RuntimeError(f'only {len(selected)} target rows')
    selected=selected[:200]
    out=RUN/'targets'; out.mkdir(parents=True,exist_ok=True)
    fields=['canonical_id','bucket','bucket_rank','support','max_margin']
    with open(out/'target_pool.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(selected)
    manifest={'canonical_count':1623,'target_count':len(selected),'session5_labels_read':False,'gold_in_target_pool':False,'target_pool_sha256':sha256(out/'target_pool.csv'),'buckets':{}}
    for x in selected: manifest['buckets'][x['bucket']]=manifest['buckets'].get(x['bucket'],0)+1
    (out/'target_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__': main()
