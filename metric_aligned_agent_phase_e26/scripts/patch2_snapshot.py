from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TDTL=Path('/mnt/05d0ac9f-c990-4b3c-8829-fd9b0049b406/shangdongfang/code/TDTL')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    import sys
    code=[ROOT/'scripts'/n for n in ('run_e26.py','supervise_e26.py','direct_responses_client.py')]
    caches=[p for s in ('PROBE','CANARY','I1') for p in sorted((ROOT/'caches'/s).glob('*.json'))]
    suffix='AFTER' if len(sys.argv)>1 and sys.argv[1]=='after' else 'BEFORE'
    out=ROOT/f'metadata/E26_RESUME_PATCH_02_{suffix}_SHA256.txt'; lines=[f'# generated_at={time.time():.6f}',f'# cache_count={len(caches)}']
    for p in code+caches: lines.append(f'{digest(p)}  {p.relative_to(ROOT)}')
    out.write_text('\n'.join(lines)+'\n')
    tdtl_out=ROOT/f'metadata/E26_RESUME_PATCH_02_TDTL_{suffix}_SNAPSHOT.tsv'; tlines=[]
    for base,dirs,files in os.walk(TDTL):
        dirs.sort(); files.sort()
        for name in files:
            p=Path(base)/name
            try:
                st=p.stat(); tlines.append(f'{p.relative_to(TDTL)}\t{st.st_size}\t{st.st_mtime_ns}')
            except OSError as exc: tlines.append(f'ERROR\t{p.relative_to(TDTL)}\t{exc}')
    tdtl_out.write_text('\n'.join(tlines)+'\n')
    summary={'patch_id':'E26_RESUME_PATCH_02','snapshot':suffix,'code_files':len(code),'cache_files':len(caches),'tdtl_files':len(tlines),'created_at':time.time()}
    (ROOT/f'metadata/E26_RESUME_PATCH_02_{suffix}_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
