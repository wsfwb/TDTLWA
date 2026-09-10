from __future__ import annotations
import json, os
from pathlib import Path
from scripts.run_e26 import RUN_ID, load_schema, prompt, rows_jsonl, sha
from scripts.cache_store import cache_identity

ROOT=Path(__file__).resolve().parents[1]
LABELS={'ang','sad','hap','neu','exc','fru'}; DECISIONS={'KEEP_CHEAP','ACCEPT_REASON'}

def main():
    schema,schema_hash=load_schema()
    rows=rows_jsonl(ROOT/'inputs/I1_long_context_test.jsonl')
    evidence={str(x['canonical_id']):x for x in rows_jsonl(ROOT/'inputs/I2_structured_evidence_test.jsonl')}
    modality={str(x['canonical_id']):x for x in rows_jsonl(ROOT/'inputs/I3_multimodal_summary_test.jsonl')}
    cache_files=sorted((ROOT/'caches/I1').glob('*.json'))
    errors=[]; seen=[]
    for path in cache_files:
        try: rec=json.loads(path.read_text())
        except Exception as exc: errors.append(f'{path.name}:json:{exc}'); continue
        cid=str(rec.get('canonical_id')); seen.append(cid)
        row=next((r for r in rows if str(r['canonical_id'])==cid),None)
        expected_prompt=sha(prompt('I1',row)) if row else None
        expected=cache_identity(RUN_ID,'I1',cid,expected_prompt,schema_hash,sha({'context':row,'evidence':evidence.get(cid),'modality':modality.get(cid)})) if row else None
        out=rec.get('output') or {}
        if rec.get('status')!='success' or rec.get('run_id')!=RUN_ID or rec.get('model')!='gpt-5.6-sol' or rec.get('reasoning_effort')!='xhigh': errors.append(f'{path.name}:identity')
        if expected and rec.get('cache_identity')!=expected: errors.append(f'{path.name}:cache_identity')
        if set(out)!={'final_label','decision','confidence','uncertainty','top2','rationale'} or out.get('final_label') not in LABELS or out.get('decision') not in DECISIONS: errors.append(f'{path.name}:schema')
    ordered=[str(r['canonical_id']) for r in rows]; missing=[cid for cid in ordered if cid not in set(seen)]
    next_id=missing[0] if missing else None
    result={'run_id':RUN_ID,'i1_cache_count':len(cache_files),'unique_canonical_count':len(set(seen)),'expected_valid_cache_count':112,'errors':errors,'next_missing_canonical_id':next_id,'i2_cache_count':len(list((ROOT/'caches/I2').glob('*.json'))),'i3_cache_count':len(list((ROOT/'caches/I3').glob('*.json'))),'probe_cache_count':len(list((ROOT/'caches/PROBE').glob('*.json'))),'canary_cache_count':len(list((ROOT/'caches/CANARY').glob('*.json'))),'session5_label_read':False,'passed':len(cache_files)==112 and len(set(seen))==112 and not errors and next_id=='Ses05F_impro03_M011'}
    (ROOT/'metadata/resume_preflight.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2)); return 0 if result['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
