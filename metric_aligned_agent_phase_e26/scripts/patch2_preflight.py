from __future__ import annotations
import json, os
from pathlib import Path
from scripts.run_e26 import RUN_ID, load_schema, prompt, sha
from scripts.cache_store import cache_identity
ROOT=Path(__file__).resolve().parents[1]
LABELS={'ang','sad','hap','neu','exc','fru'}; DECISIONS={'KEEP_CHEAP','ACCEPT_REASON'}
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    schema,schema_hash=load_schema(); ctx=rows(ROOT/'inputs/I1_long_context_test.jsonl'); ev={str(x['canonical_id']):x for x in rows(ROOT/'inputs/I2_structured_evidence_test.jsonl')}; mo={str(x['canonical_id']):x for x in rows(ROOT/'inputs/I3_multimodal_summary_test.jsonl')}
    seen=[]; errors=[]
    for p in sorted((ROOT/'caches/I1').glob('*.json')):
        rec=json.loads(p.read_text()); cid=str(rec.get('canonical_id')); seen.append(cid); row=next((r for r in ctx if str(r['canonical_id'])==cid),None)
        expected=cache_identity(RUN_ID,'I1',cid,sha(prompt('I1',row)),schema_hash,sha({'context':row,'evidence':ev.get(cid),'modality':mo.get(cid)})) if row else None
        out=rec.get('output') or {}
        if rec.get('status')!='success' or rec.get('run_id')!=RUN_ID or rec.get('model')!='gpt-5.6-sol' or rec.get('reasoning_effort')!='xhigh' or rec.get('cache_identity')!=expected: errors.append(f'{cid}:identity')
        if set(out)!={'final_label','decision','confidence','uncertainty','top2','rationale'} or out.get('final_label') not in LABELS or out.get('decision') not in DECISIONS: errors.append(f'{cid}:schema')
    ordered=[str(r['canonical_id']) for r in ctx]; missing=[c for c in ordered if c not in set(seen)]
    state=json.loads((ROOT/'metadata/pipeline_state.json').read_text()); result={'patch_id':'E26_RESUME_PATCH_02','i1_cache_count':len(seen),'unique_i1_count':len(set(seen)),'errors':errors,'next_missing_canonical_id':missing[0] if missing else None,'probe_count':len(list((ROOT/'caches/PROBE').glob('*.json'))),'canary_count':len(list((ROOT/'caches/CANARY').glob('*.json'))),'i2_count':len(list((ROOT/'caches/I2').glob('*.json'))),'i3_count':len(list((ROOT/'caches/I3').glob('*.json'))),'probe_passed':state.get('probe',{}).get('passed') is True,'canary_passed':state.get('canary',{}).get('passed') is True,'session5_label_read':False,'passed':len(seen)==489 and len(set(seen))==489 and not errors and missing and state.get('probe',{}).get('passed') and state.get('canary',{}).get('passed') and len(list((ROOT/'caches/I2').glob('*.json')))==0 and len(list((ROOT/'caches/I3').glob('*.json')))==0}
    (ROOT/'metadata/E26_RESUME_PATCH_02_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return 0 if result['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
