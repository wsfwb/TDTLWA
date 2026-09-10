#!/usr/bin/env python3
"""Read-only recheck; stopping is intentionally performed only after exact PID validation."""
import json,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
text=subprocess.check_output(['ps','-eo','pid,ppid,pgid,etime,stat,args'],text=True)
matches=[]
for line in text.splitlines():
 if 'ClarifyMER' in line and any(x in line for x in ('metric_aligned_agent_phase_e23','metric_aligned_agent_phase_e22r','run_sol_xhigh_reasoner','run_e22r')):matches.append(line.strip())
out={'timestamp':time.time(),'matches':matches,'status':'active_clarifymer_reasoner_process_found' if matches else 'no_active_clarifymer_reasoner_process','signals_sent':[]}
(RUN/'metadata/process_stop_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
