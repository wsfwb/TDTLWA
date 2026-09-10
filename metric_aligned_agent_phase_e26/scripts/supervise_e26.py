#!/usr/bin/env python3
from __future__ import annotations
import json, os, time, traceback
import fcntl
from pathlib import Path
from . import run_e26
from .cache_store import atomic_write_json
from .supervisor import may_start_session5

ROOT=Path(__file__).resolve().parents[1]
def planned_stages(state):
    if not may_start_session5(state): return []
    return [stage for stage in ('I1','I2','I3') if not state.get(stage,{}).get('passed')]
def log(message):
    path=ROOT/'logs'/'supervisor.log'; path.parent.mkdir(exist_ok=True)
    with path.open('a',encoding='utf-8') as handle: handle.write(f'{time.time():.3f} {message}\n');handle.flush();os.fsync(handle.fileno())
def main():
    lock_path=ROOT/'metadata/pipeline_supervisor.lock'; lock=lock_path.open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('another E26 pipeline supervisor is active')
    try:
        pid=os.getpid(); lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':pid,'run_id':ROOT.name,'acquired_at':time.time()}));lock.flush();os.fsync(lock.fileno())
        atomic_write_json(ROOT/'metadata/e26_supervisor.pid',{'pid':pid,'run_id':ROOT.name,'written_at':time.time()})
        state=run_e26.state(); stages=planned_stages(state)
        if not stages: raise SystemExit('gates are not open or all interfaces are complete')
        atomic_write_json(ROOT/'metadata/supervisor_status.json',{'pid':pid,'stages':stages,'current_stage':stages[0],'started_at':time.time(),'status':'running'})
        final_status='stopped'; current_stage=stages[0]; final_error=None
        for stage in stages:
            current_stage=stage
            log(f'start {stage}')
            try:
                code=run_e26.run(stage)
            except Exception as exc:
                log(f'stop {stage}: {type(exc).__name__}: {exc}')
                atomic_write_json(ROOT/'metadata/supervisor_status.json',{'pid':os.getpid(),'stages':stages,'current_stage':stage,'status':'stopped','error':str(exc),'updated_at':time.time()})
                final_error=str(exc)
                raise
            if code != 0:
                log(f'stop {stage}: exit={code}')
                raise SystemExit(code)
            log(f'complete {stage}')
        final_status='all_interfaces_complete'
        atomic_write_json(ROOT/'metadata/supervisor_status.json',{'pid':pid,'stages':stages,'current_stage':None,'status':final_status,'completed_at':time.time()})
    finally:
        if 'final_status' in locals() and final_status != 'all_interfaces_complete':
            atomic_write_json(ROOT/'metadata/supervisor_status.json',{'pid':os.getpid(),'stages':locals().get('stages',[]),'current_stage':locals().get('current_stage'),'status':'stopped','error':locals().get('final_error'),'updated_at':time.time()})
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__': main()
