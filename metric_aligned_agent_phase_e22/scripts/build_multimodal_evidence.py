"""Serialize frozen modality-summary vectors; raw audio/video is unavailable."""
import os
from pathlib import Path
import json
import numpy as np
RUN=Path(os.environ.get('CLARIFYMER_RUN_DIR', Path(__file__).resolve().parents[1]))
_DEFAULT_E20 = RUN.parents[2]/'experiment_outputs/metric_aligned_agent_phase_e20/run_20260814T120000_CST_context_enriched_test_selected_agent_v1' if len(RUN.parents)>2 else RUN/'e20'
E20=Path(os.environ.get('CLARIFYMER_E20_DIR', _DEFAULT_E20))
def main():
 d=dict(np.load(E20/'features/test_enriched.npz',allow_pickle=False)); out=[]
 for i,cid in enumerate(d['ids'].astype(str)):
  v=np.asarray(d['F2'][i],float); out.append({'canonical_id':cid,'modality_summary_vector':v.tolist(),'modality_summary_dimensions':int(len(v)),'raw_audio_available':False,'raw_video_available':False,'missing_modality_indicator':0})
 (RUN/'inputs/I3_multimodal_summary_test.jsonl').write_text('\n'.join(json.dumps(x) for x in out)+'\n'); (RUN/'metadata/multimodal_status.json').write_text(json.dumps({'summary_vectors_available':True,'raw_audio_available':False,'raw_video_available':False,'gold_in_prompt':False},indent=2)+'\n'); print(json.dumps({'rows':len(out),'raw_audio_video_available':False}))
if __name__=='__main__': main()
