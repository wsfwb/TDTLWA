from __future__ import annotations
import hashlib, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    files = [ROOT / 'scripts' / name for name in ('direct_responses_client.py', 'run_e26.py', 'supervise_e26.py')]
    files += sorted((ROOT / 'caches' / 'I1').glob('*.json'))
    files += sorted((ROOT / 'caches' / 'PROBE').glob('*.json'))
    files += sorted((ROOT / 'caches' / 'CANARY').glob('*.json'))
    import sys
    out_name = 'E26_RESUME_PATCH_01_AFTER_SHA256.txt' if len(sys.argv) > 1 and sys.argv[1] == 'after' else 'E26_RESUME_PATCH_01_BEFORE_SHA256.txt'
    out = ROOT / 'metadata' / out_name
    lines = [f'# generated_at={time.time():.6f}', f'# file_count={len(files)}']
    for path in files:
        lines.append(f'{digest(path)}  {path.relative_to(ROOT)}')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = {
        'generated_at': time.time(),
        'code_files': [str(p.relative_to(ROOT)) for p in files[:3]],
        'cache_counts': {stage: len(list((ROOT / 'caches' / stage).glob('*.json'))) for stage in ('PROBE', 'CANARY', 'I1')},
        'file_count': len(files),
    }
    (ROOT / 'metadata' / 'E26_RESUME_PATCH_01_BEFORE_SUMMARY.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
