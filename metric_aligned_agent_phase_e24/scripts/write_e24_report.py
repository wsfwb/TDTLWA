#!/usr/bin/env python3
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
print((RUN/'reports/PHASE_E24_REPORT.md').read_text())
