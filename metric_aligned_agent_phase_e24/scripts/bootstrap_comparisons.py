#!/usr/bin/env python3
import json
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
state=json.loads((RUN/'metadata/pipeline_state.json').read_text())
if state.get('status')!='interfaces_complete':raise SystemExit('bootstrap refused: no complete candidates')
