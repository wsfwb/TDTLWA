#!/usr/bin/env python3
from __future__ import annotations
import json,os,sys,time,traceback
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN))
from scripts.cache_store import SingletonLock,atomic_write_json
from scripts.run_direct_reasoner import run_stage
def main():
    lock=SingletonLock(RUN/'metadata/supervisor.lock')
    if not lock.try_acquire():print('second supervisor rejected',file=sys.stderr);return 73
    try:
        atomic_write_json(RUN/'metadata/supervisor_identity.json',{'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgrp(),'command':' '.join(sys.argv),'started_at':time.time(),'run':RUN.name})
        state={'status':'running','pid':os.getpid(),'current_stage':'CANARY','started_at':time.time(),'stages':{}}
        atomic_write_json(RUN/'metadata/pipeline_state.json',state)
        canary=run_stage('I1',canary=True);state['stages']['CANARY']=canary
        if not canary['passed']:
            state['status']='canary_blocked';atomic_write_json(RUN/'metadata/pipeline_state.json',state);return 2
        atomic_write_json(RUN/'metadata/canary_result.json',canary)
        for stage in ('I1','I2','I3'):
            state['current_stage']=stage;atomic_write_json(RUN/'metadata/pipeline_state.json',state)
            result=run_stage(stage);state['stages'][stage]=result;atomic_write_json(RUN/'metadata/pipeline_state.json',state)
            if not result['passed']:
                state['status']='transport_blocked';atomic_write_json(RUN/'metadata/pipeline_state.json',state);return 3
        state['status']='interfaces_complete';state['current_stage']=None;state['completed_at']=time.time();atomic_write_json(RUN/'metadata/pipeline_state.json',state);return 0
    except Exception as e:
        state={'status':'transport_blocked','pid':os.getpid(),'error_class':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()[-4000:],'updated_at':time.time()};atomic_write_json(RUN/'metadata/pipeline_state.json',state);return 4
    finally:lock.release()
if __name__=='__main__':raise SystemExit(main())
