#!/usr/bin/env python3
import json,re
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];E23=RUN.parents[1]/'metric_aligned_agent_phase_e23/run_20260818T000000_CST_gpt56sol_xhigh_full_interface_v1'
patterns=re.compile(r'\b(exec|find|grep|rg|sed|python|workspace|tool)\b',re.I);hits=0;files=0
for p in (E23/'results/reasoner_outputs').glob('*/*.json'):
 files+=1
 try:
  x=json.loads(p.read_text());hits+=bool(patterns.search(x.get('stderr_tail','')))
 except Exception:pass
out={'e23_records_scanned':files,'tool_trace_records':hits,'invalid_for_primary_evaluation':'tool_access_contamination_detected' if hits else None,'e23_cache_eligible_for_e24':False}
(RUN/'metadata/e23_failure_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
