"""Build deterministic prompt inputs for I1–I3 without gold/outcome fields."""
from pathlib import Path
import json
import pandas as pd

import os
RUN=Path(os.environ.get('CLARIFYMER_RUN_DIR', Path(__file__).resolve().parents[1]))
_PROJECT = Path(os.environ.get('CLARIFYMER_PROJECT', '')) or RUN.parents[2] if os.environ.get('CLARIFYMER_PROJECT') else (RUN.parents[2] if len(RUN.parents)>2 else RUN)
PROJECT=_PROJECT
# Upstream artifacts: a canonical transcript bridge CSV (utterance id, dialogue, speaker,
# turn index, transcript) and the E20 test-context manifest listing the test canonical ids.
# Point these env vars at your own regenerated copies (IEMOCAP transcripts may not be
# redistributed, so this repo does not ship them).
CANON=Path(os.environ.get('CLARIFYMER_CANONICAL_CSV', PROJECT/'experiment_outputs/metric_aligned_agent_phase_e05/run_20260810T183930_CST_exploratory_v1/results/canonical_iemocap_utterances.csv'))
E20=Path(os.environ.get('CLARIFYMER_E20_DIR', PROJECT/'experiment_outputs/metric_aligned_agent_phase_e20/run_20260814T120000_CST_context_enriched_test_selected_agent_v1'))
MAX_CHARS=12000
def clip(s):
    if len(s)<=MAX_CHARS:return s,False
    n=MAX_CHARS//2;return s[:n]+'\n...[FIXED_HEAD_TAIL_TRUNCATION]...\n'+s[-n:],True
def main():
    d=pd.read_csv(CANON); ids=pd.read_csv(E20/'features/test_context_manifest.csv').canonical_id.astype(str).tolist(); d=d[d.canonical_sample_id.astype(str).isin(ids)].copy()
    d['utterance_index']=pd.to_numeric(d.utterance_index,errors='coerce').fillna(0); d=d.sort_values(['dialogue_id','utterance_index','canonical_sample_id'])
    out=[]; trunc=0
    for cid,g in d.groupby('canonical_sample_id',sort=False):
        row=g.iloc[0]; prior=g[(g.dialogue_id==row.dialogue_id)&(g.utterance_index<row.utterance_index)]
        turns=[{'speaker':str(x.speaker),'turn_index':int(x.utterance_index),'text':str(x.transcript_raw)} for _,x in prior.iterrows()]
        raw='\n'.join(f"{x['speaker']} [{x['turn_index']}]: {x['text']}" for x in turns+[{'speaker':str(row.speaker),'turn_index':int(row.utterance_index),'text':str(row.transcript_raw)}])
        raw,was=clip(raw); trunc+=int(was)
        out.append({'canonical_id':str(cid),'dialogue_id':str(row.dialogue_id),'speaker':str(row.speaker),'turn_index':int(row.utterance_index),'dialogue_length':int(g.dialogue_id.eq(row.dialogue_id).sum()),'current_utterance':str(row.transcript_raw),'preceding_turns':turns,'serialized_context':raw,'truncated_fixed_head_tail':was})
    p=RUN/'inputs/I1_long_context_test.jsonl'; p.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n')
    (RUN/'metadata/long_context_build.json').write_text(json.dumps({'rows':len(out),'max_chars':MAX_CHARS,'truncated_rows':trunc,'rule':'all preceding turns; fixed head+tail truncation; no gold'},indent=2)+'\n')
    print(json.dumps({'rows':len(out),'truncated_rows':trunc}))
if __name__=='__main__': main()
