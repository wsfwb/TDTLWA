import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from .io_atomic import atomic_write_json,read_csv_rows,sha256_file
RUN=Path(__file__).resolve().parents[1]; ROOT=RUN.parents[2]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'; E27=ROOT/'experiment_outputs/metric_aligned_agent_phase_e27/run_20260822T000000_CST_frozen_multi_interface_fusion_v1'
def replay_inputs(write=True):
    src=read_csv_rows(E26/'evaluation/results/e26_canonical_predictions.csv'); ids={r['canonical_id'] for r in src}
    if len(src)!=1623 or len(ids)!=1623: raise RuntimeError('canonical alignment')
    counts={}; sets={}
    for stage in ('I1','I2','I3'):
        fs=sorted((E26/'caches'/stage).glob('*.json'))
        if len(fs)!=1623: raise RuntimeError(f'{stage} count')
        ds=[json.loads(p.read_text()) for p in fs]
        if any(d.get('status')!='success' or d.get('model')!='gpt-5.6-sol' or d.get('reasoning_effort')!='xhigh' or d.get('tool_call') or d.get('function_call') for d in ds): raise RuntimeError(f'{stage} cache identity')
        sets[stage]={d.get('canonical_id',p.stem) for d,p in zip(ds,fs)}; counts[stage]=len(sets[stage])
    if len({frozenset(v) for v in sets.values()})!=1: raise RuntimeError('cache sets')
    w=read_csv_rows(E27/'results/e27_winner_canonical_predictions.csv'); y=np.array([int(r['gold_label_evaluation_only']) for r in w]); p=np.array([int(r['winner_prediction']) for r in w]); wf=float(f1_score(y,p,labels=list(range(6)),average='weighted',zero_division=0))
    if abs(wf-.7444884347126642)>1e-12: raise RuntimeError(f'E27 replay {wf}')
    cfg=json.loads(read_csv_rows(E27/'results/e27_absolute_winner.csv')[0]['config_json']); out={'canonical_count':1623,'cache_counts':counts,'e26_wf1':.7417088710802171,'e27_wf1':wf,'e27_winner_k':int(cfg.get('k',28)),'e27_winner_config':cfg,'e26_schema_sha256':sha256_file(E26/'protocol/E26_OUTPUT_SCHEMA.json'),'e27_materialization_sha256':sha256_file(E27/'materialized/MATERIALIZATION_SHA256.json')}
    if write: atomic_write_json(RUN/'metadata/E29_INPUT_AUDIT.json',out)
    return out
