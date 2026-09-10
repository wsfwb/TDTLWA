from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    before=ROOT/'metadata/E26_RESUME_PATCH_01_BEFORE_SHA256.txt'; code_mismatches=[]; cache_mismatches=[]; checked=0
    for line in before.read_text().splitlines():
        if not line or line.startswith('#'): continue
        old,rel=line.split('  ',1); path=ROOT/rel; checked+=1
        if not path.exists() or digest(path)!=old:
            (code_mismatches if rel.startswith('scripts/') else cache_mismatches).append(rel)
    result={'checked_before_files':checked,'expected_patch_code_changes':code_mismatches,'cache_mismatches':cache_mismatches,'old_cache_integrity_preserved':not cache_mismatches,'current_i1_cache_count':len(list((ROOT/'caches/I1').glob('*.json'))),'current_probe_cache_count':len(list((ROOT/'caches/PROBE').glob('*.json'))),'current_canary_cache_count':len(list((ROOT/'caches/CANARY').glob('*.json')))}
    (ROOT/'metadata/E26_RESUME_PATCH_01_CACHE_INTEGRITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); return 0 if result['old_cache_integrity_preserved'] else 2
if __name__=='__main__': raise SystemExit(main())
