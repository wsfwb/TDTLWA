import json, hashlib
from pathlib import Path
from .io_atomic import atomic_write_json, sha256_file

RUN=Path(__file__).resolve().parents[1]
def main():
    paths=[RUN/"materialized/MATERIALIZATION_SHA256.json",RUN/"results/e27_candidate_leaderboard.csv",RUN/"metadata/E27_GATE.json"]
    data={str(p.relative_to(RUN)):sha256_file(p) for p in paths}; atomic_write_json(RUN/"metadata/SHA256SUMS.txt.json",data); (RUN/"metadata/SHA256SUMS.txt").write_text("\n".join(f"{v}  {k}" for k,v in data.items())+"\n")
if __name__=="__main__": main()
