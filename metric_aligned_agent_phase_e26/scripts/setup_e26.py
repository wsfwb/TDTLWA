#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E24=Path('/mnt/05d0ac9f-c990-4b3c-8829-fd9b0049b406/shangdongfang/code/ClarifyMER/experiment_outputs/metric_aligned_agent_phase_e24/run_20260819T000000_CST_direct_responses_gpt56sol_xhigh_v1')
TDTL=Path('/mnt/05d0ac9f-c990-4b3c-8829-fd9b0049b406/shangdongfang/code/TDTL')

def sha_file(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def snapshot(name: str, roots):
    rows=[]
    for root in roots:
        files=sorted(path for path in Path(root).rglob('*') if path.is_file())
        for path in files:
            stat=path.stat(); rows.append({'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':sha_file(path)})
    out=ROOT/'metadata'/f'{name}_snapshot.csv'
    with out.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['path','size','mtime_ns','sha256']); writer.writeheader();writer.writerows(rows)
    return len(rows)
def read_gpu():
    try:return subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader'],text=True,stderr=subprocess.STDOUT,timeout=30)
    except Exception as exc:return 'error: '+str(exc)
def git_metadata(path: str) -> dict:
    try:
        return {'status':'available',
                'git_head':subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip(),
                'git_status':subprocess.check_output(['git','-C',path,'status','--short'],text=True,stderr=subprocess.DEVNULL)}
    except subprocess.CalledProcessError:
        return {'status':'not_a_git_repository','git_head':None,'git_status':None}
def main():
    ROOT.joinpath('inputs').mkdir(exist_ok=True)
    source=E24/'inputs/canary_I1_train.jsonl'; target=ROOT/'inputs/canary_I1_train.jsonl'
    shutil.copyfile(source,target)
    input_rows=[json.loads(x) for x in target.read_text().splitlines() if x.strip()]
    if len(input_rows)!=20: raise RuntimeError(f'expected 20 Train canary rows, got {len(input_rows)}')
    metadata={'run_id':ROOT.name,'created_at':time.time(),'source_canary':str(source),'source_canary_sha256':sha_file(source),'copied_canary_sha256':sha_file(target),'source_session5_inputs':{k:str(E24/'inputs'/k) for k in ('I1_long_context_test.jsonl','I2_structured_evidence_test.jsonl','I3_multimodal_summary_test.jsonl')},'session5_input_read':False,'session5_label_read':False,'source_artifacts_read_only':True}
    (ROOT/'metadata/input_sources.json').write_text(json.dumps(metadata,indent=2)+'\n')
    process_text=subprocess.check_output(['ps','-eo','pid,ppid,pgid,etime,stat,args'],text=True)
    matching=[line for line in process_text.splitlines() if any(token in line for token in ('run_e26.py','run_direct_reasoner.py','run_sol_xhigh_reasoner.py','supervise_interfaces.py'))]
    (ROOT/'metadata/process_audit.json').write_text(json.dumps({'status':'no_active_reasoner_process' if not matching else 'active_related_process_detected','matching_processes':matching,'stopped_pids':[],'checked_at':time.time()},indent=2)+'\n')
    (ROOT/'metadata/GPU_STATUS.txt').write_text(read_gpu())
    try:
        import torch
        torch_info={'version':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':bool(torch.cuda.is_available()),'device_count':torch.cuda.device_count(),'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    except Exception as exc: torch_info={'error':str(exc)}
    env={'python':sys.version,'torch':torch_info,'reasoner_model':'gpt-5.6-sol','reasoning_effort':'xhigh','temperature_omitted':True,'store_omitted':True,'tools_omitted':True,'wire_api':'direct_responses','endpoint':'https://ep.aicorp.work'}
    (ROOT/'metadata/ENVIRONMENT.json').write_text(json.dumps(env,indent=2)+'\n')
    # Pre-canary integrity evidence intentionally excludes any Session-5 source input/content.
    tdtl_meta=git_metadata(str(TDTL))
    (ROOT/'metadata/tdtl_git_before.json').write_text(json.dumps(tdtl_meta,indent=2)+'\n')
    protected=[E24/'scripts',E24/'protocol',E24/'metadata']
    count=snapshot('before',protected)
    (ROOT/'metadata/session5_access_status.json').write_text(json.dumps({'session5_input_read':False,'session5_label_read':False,'session5_predictions_read':False,'session5_metrics':False,'phase':'pre_probe'},indent=2)+'\n')
    print(json.dumps({'canary_rows':len(input_rows),'snapshot_files':count,'gpu_visible':torch_info.get('cuda_available')},indent=2))
if __name__=='__main__': main()
