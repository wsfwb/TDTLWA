import csv, json, hashlib
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
E26=RUN.parents[2]/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'
TARGETS=RUN/'targets/target_pool.csv'
LABELS=['hap','sad','neu','ang','exc','fru']

def read_jsonl(path):
    out={}
    for line in path.read_text().splitlines():
        if line.strip():
            d=json.loads(line); out[d['canonical_id']]=d
    return out
def read_cache(stage):
    out={}
    for p in (E26/'caches'/stage).glob('*.json'):
        d=json.loads(p.read_text()); out[d['canonical_id']]=d.get('output',{})
    return out
def prompt_for(cid, i1, i2, i3, o1, o2, o3, target_meta):
    def label(x): return LABELS[int(x)]
    evidence={
        'current_utterance':i1.get('current_utterance'),
        'preceding_turns':i1.get('preceding_turns',[]),
        'speaker':i1.get('speaker'),'turn_index':i1.get('turn_index'),'dialogue_position':i1.get('turn_index'),
        'full_prediction':label(i2['full_prediction']),'d3_prediction':label(i2['residual_prediction']),
        'full_probability':i2.get('full_probability'),'d3_probability':i2.get('residual_probability'),
        'full_confidence':i2.get('full_confidence'),'d3_confidence':i2.get('residual_confidence'),
        'full_entropy':i2.get('full_entropy'),'d3_entropy':i2.get('residual_entropy'),
        'full_margin':i2.get('full_margin'),'d3_margin':i2.get('residual_margin'),
        'agreement':i2.get('agreement'),'confidence_delta':i2.get('confidence_delta'),
        'margin_delta':i2.get('margin_delta'),'entropy_delta':i2.get('entropy_delta'),
        'modality_summary':i3.get('modality_summary_vector'),
        'missing_modality_indicator':i3.get('missing_modality_indicator'),
        'interfaces':{
            'I1':{'label':o1.get('final_label'),'decision':o1.get('decision'),'confidence':o1.get('confidence'),'uncertainty':o1.get('uncertainty'),'rationale':o1.get('rationale')},
            'I2':{'label':o2.get('final_label'),'decision':o2.get('decision'),'confidence':o2.get('confidence'),'uncertainty':o2.get('uncertainty'),'rationale':o2.get('rationale')},
            'I3':{'label':o3.get('final_label'),'decision':o3.get('decision'),'confidence':o3.get('confidence'),'uncertainty':o3.get('uncertainty'),'rationale':o3.get('rationale')},
        },
        'view_support':target_meta,
    }
    return ('You are an emotion adjudicator. Use only the evidence below. Do not infer or request gold labels, metrics, or outcomes. '
            'Return exactly the requested JSON object. Labels are hap,sad,neu,ang,exc,fru.\nEVIDENCE\n'+json.dumps(evidence,sort_keys=True,separators=(',',':'))+
            '\nSCHEMA FIELDS: final_label, decision (KEEP_BASE|ACCEPT_EXISTING_REASON|REVISE), preferred_source (FULL|D3|E27|STAGE_A|I1|I2|I3|REVISED), confidence [0,1], uncertainty [0,1], evidence_consistency (LOW|MEDIUM|HIGH), short_reason (1-800 chars).')

def main():
    targets=list(csv.DictReader(TARGETS.open()))
    i1=read_jsonl(E26/'inputs/I1_long_context_test.jsonl'); i2=read_jsonl(E26/'inputs/I2_structured_evidence_test.jsonl'); i3=read_jsonl(E26/'inputs/I3_multimodal_summary_test.jsonl')
    caches={s:read_cache(s) for s in ('I1','I2','I3')}
    out=RUN/'targets/adjudicator_prompts.jsonl'; out.parent.mkdir(exist_ok=True)
    rows=[]
    with out.open('w') as f:
        for t in targets:
            c=t['canonical_id']; p=prompt_for(c,i1[c],i2[c],i3[c],caches['I1'][c],caches['I2'][c],caches['I3'][c],{k:v for k,v in t.items() if k!='canonical_id'})
            rec={'canonical_id':c,'prompt':p,'prompt_sha256':hashlib.sha256(p.encode()).hexdigest(),'input_sha256':hashlib.sha256(json.dumps({'i1':i1[c],'i2':i2[c],'i3':i3[c]},sort_keys=True,separators=(',',':')).encode()).hexdigest()}
            f.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n'); rows.append(rec)
    (RUN/'metadata/E29_TARGET_PROMPTS.json').write_text(json.dumps({'count':len(rows),'gold_in_prompts':False,'prompt_hashes':[r['prompt_sha256'] for r in rows]},indent=2))
    print(json.dumps({'count':len(rows),'path':str(out)}))
if __name__=='__main__': main()
