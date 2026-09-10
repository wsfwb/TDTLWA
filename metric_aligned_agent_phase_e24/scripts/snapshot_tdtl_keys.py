#!/usr/bin/env python3
import csv,hashlib,os
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
SRC=Path('experiment_outputs/original_iemocap_split_provenance_audit/run_20260812T105050_CST/metadata/tdtl_before_snapshot.csv')
rows=list(csv.DictReader(SRC.open()));fields=['path','exists','size_bytes','mtime_ns','sha256']
with (RUN/'metadata/tdtl_keyfile_after.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rows:
  p=Path(r['path']);w.writerow({'path':str(p),'exists':p.exists(),'size_bytes':p.stat().st_size if p.exists() else '','mtime_ns':p.stat().st_mtime_ns if p.exists() else '','sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ''})
with (RUN/'metadata/tdtl_keyfile_before.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
after={r['path']:r for r in csv.DictReader((RUN/'metadata/tdtl_keyfile_after.csv').open())}
matches=all(r['size_bytes']==after[r['path']]['size_bytes'] and r['mtime_ns']==after[r['path']]['mtime_ns'] and r['sha256']==after[r['path']]['sha256'] for r in rows)
print('tdtl_keyfiles_match_reference='+str(matches).lower())
