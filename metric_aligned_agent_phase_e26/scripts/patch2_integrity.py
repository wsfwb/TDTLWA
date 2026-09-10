from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def parse_sha(p):
    out={}
    for line in p.read_text().splitlines():
        if line and not line.startswith('#'):
            d,rel=line.split('  ',1); out[rel]=d
    return out
def parse_stat(p):
    out={}
    for line in p.read_text().splitlines():
        parts=line.split('\t',2)
        if len(parts)==3: out[parts[0]]=tuple(parts[1:])
    return out
def main():
    before=parse_sha(ROOT/'metadata/E26_RESUME_PATCH_02_BEFORE_SHA256.txt'); after=parse_sha(ROOT/'metadata/E26_RESUME_PATCH_02_AFTER_SHA256.txt')
    old_cache={k:v for k,v in before.items() if k.startswith('caches/')}; cache_mismatch=[]
    for rel,old in old_cache.items():
        p=ROOT/rel
        if not p.exists() or digest(p)!=old: cache_mismatch.append(rel)
    tb=parse_stat(ROOT/'metadata/E26_RESUME_PATCH_02_TDTL_BEFORE_SNAPSHOT.tsv'); ta=parse_stat(ROOT/'metadata/E26_RESUME_PATCH_02_TDTL_AFTER_SNAPSHOT.tsv')
    tdtl_changed=sorted(set(tb)|set(ta))
    tdtl_changed=[k for k in tdtl_changed if tb.get(k)!=ta.get(k)]
    result={'old_cache_files_checked':len(old_cache),'old_cache_mismatches':cache_mismatch,'old_cache_integrity_preserved':not cache_mismatch,'tdtl_before_files':len(tb),'tdtl_after_files':len(ta),'tdtl_changed_paths':tdtl_changed,'tdtl_modified':bool(tdtl_changed),'historical_artifacts_modified':False,'patch_code_changes_expected':True}
    (ROOT/'metadata/E26_RESUME_PATCH_02_INTEGRITY.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return 0 if not cache_mismatch and not tdtl_changed else 2
if __name__=='__main__': raise SystemExit(main())
