from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path

def cache_identity(run_id: str, interface: str, canonical_id: str, prompt_hash: str, schema_hash: str, input_hash: str = '') -> str:
    payload = {'run_id': run_id, 'interface': interface, 'canonical_id': canonical_id,
               'model': 'gpt-5.6-sol', 'effort': 'xhigh', 'temperature': None,
               'prompt_sha256': prompt_hash, 'schema_sha256': schema_hash, 'input_sha256': input_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name): os.unlink(temp_name)
