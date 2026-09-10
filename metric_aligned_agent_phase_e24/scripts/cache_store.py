from __future__ import annotations
import fcntl,hashlib,json,os,tempfile
from pathlib import Path
def sha256_text(x:str)->str:return hashlib.sha256(x.encode()).hexdigest()
def cache_identity(run_id,interface,canonical_id,prompt_hash,input_hash,schema_hash):
    raw='|'.join([run_id,'gpt-5.6-sol','xhigh','0.0',interface,canonical_id,prompt_hash,input_hash,schema_hash])
    return sha256_text(raw)
def atomic_write_json(path:Path,obj)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=str(path.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(obj,f,ensure_ascii=False,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
class SingletonLock:
    def __init__(self,path):self.path=Path(path);self.handle=None
    def try_acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.handle=self.path.open('a+')
        try:fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);return True
        except BlockingIOError:self.handle.close();self.handle=None;return False
    def acquire(self):
        if not self.try_acquire():raise RuntimeError('supervisor already active')
    def release(self):
        if self.handle:fcntl.flock(self.handle.fileno(),fcntl.LOCK_UN);self.handle.close();self.handle=None
