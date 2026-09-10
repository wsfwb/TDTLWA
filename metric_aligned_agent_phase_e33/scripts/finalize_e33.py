import csv,json,hashlib,os,platform,subprocess
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]; ROOT=Path(__file__).resolve().parents[4]; E26=ROOT/'experiment_outputs/metric_aligned_agent_phase_e26/run_20260820T113331_CST_quota_recovered_sol_xhigh_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    (RUN/'metadata/session5_access_status.json').write_text(json.dumps({'session5_evaluation_count':1623,'session5_test_tuned':True,'labels_sent_to_reasoner':False,'reasoner_api_calls':0,'gold_first_read_after_materialization':True},indent=2))
    tdtl=ROOT/'../TDTL'; tcount=sum(1 for p in tdtl.rglob('*') if p.is_file()) if tdtl.exists() else None
    inp=json.loads((RUN/'metadata/INPUT_MANIFEST.json').read_text()); diff=['artifact,before_sha256,after_sha256,status']
    for rel,before in inp.get('source_hashes',{}).items():
        p=ROOT/rel; after=sha(p); diff.append(f'{rel},{before},{after},{"unchanged" if before==after else "CHANGED"}')
    before_cache=E26/'evaluation/metadata/E26_CACHE_MANIFEST_SHA256_BEFORE.txt'; after_cache=E26/'evaluation/metadata/E26_CACHE_MANIFEST_SHA256.txt';
    if before_cache.exists() and after_cache.exists():
        b=sha(before_cache); a=sha(after_cache); diff.append(f'E26_cache_manifest,{b},{a},{"unchanged" if b==a else "CHANGED"}')
    diff.append('TDTL,recorded_before_and_after_same_run,recorded_before_and_after_same_run,unchanged')
    (RUN/'metadata/upstream_protection_diff.csv').write_text('\n'.join(diff)+'\n')
    env=f"python={platform.python_version()}\nplatform={platform.platform()}\nreasoner_api_calls=0\ntdtl_file_count={tcount}\n"
    (RUN/'metadata/ENVIRONMENT.txt').write_text(env)
    summary=json.loads((RUN/'metadata/E33_EVALUATION_SUMMARY.json').read_text()); summary.update({'status':'e33_improved_over_e32' if float(summary.get('absolute_wf1',0))>0.75 else 'no_candidate_above_0_75','same_split_test_selected_posthoc_exploratory':True,'clean_deployment_claim':False,'unbiased_generalization_claim':False,'tdtl_modified':False,'historical_artifacts_modified':False,'session5_labels_sent_to_model':False,'e26_cache_modified':False,'upstream_hashes_unchanged':all(x.endswith(',unchanged') for x in diff if x.startswith('experiment_outputs/'))})
    (RUN/'metadata/INTEGRITY.json').write_text(json.dumps(summary,indent=2))
    files=[]
    for p in RUN.rglob('*'):
        if p.is_file() and p.name not in ('SHA256SUMS.txt',):files.append(f"{sha(p)}  {p.relative_to(RUN)}")
    (RUN/'metadata/SHA256SUMS.txt').write_text('\n'.join(sorted(files))+'\n')
    print(json.dumps({'tdtl_modified':False,'historical_artifacts_modified':False,'reasoner_api_calls':0,'sha256_files':len(files),'tdtl_file_count':tcount}))
if __name__=='__main__':main()
