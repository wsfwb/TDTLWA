"""Serialize Full/D3 pre-Reason evidence for I2/I3; never include gold/R1 outcome."""
import os
from pathlib import Path
import json
import numpy as np

RUN=Path(os.environ.get('CLARIFYMER_RUN_DIR', Path(__file__).resolve().parents[1]))
_DEFAULT_E20 = RUN.parents[2]/'experiment_outputs/metric_aligned_agent_phase_e20/run_20260814T120000_CST_context_enriched_test_selected_agent_v1' if len(RUN.parents)>2 else RUN/'e20'
E20=Path(os.environ.get('CLARIFYMER_E20_DIR', _DEFAULT_E20))
def main():
 d=dict(np.load(E20/'features/test_enriched.npz',allow_pickle=False))
 labels=['hap','sad','neu','ang','exc','fru']; out=[]
 for i,cid in enumerate(d['ids'].astype(str)):
  f=np.asarray(d['F1'][i,:6],float); r=np.asarray(d['F1'][i,6:12],float)
  out.append({'canonical_id':cid,'full_prediction':int(d['full_pred'][i]),'residual_prediction':int(d['residual_pred'][i]),'full_probability':f.tolist(),'residual_probability':r.tolist(),'full_confidence':float(d['F0'][i,36]),'residual_confidence':float(d['F0'][i,37]),'full_entropy':float(d['F0'][i,38]),'residual_entropy':float(d['F0'][i,39]),'full_margin':float(d['F0'][i,40]),'residual_margin':float(d['F0'][i,41]),'agreement':bool(d['full_pred'][i]==d['residual_pred'][i]),'confidence_delta':float(d['F0'][i,36]-d['F0'][i,37]),'margin_delta':float(d['F0'][i,40]-d['F0'][i,41]),'entropy_delta':float(d['F0'][i,38]-d['F0'][i,39]),'label_names':labels})
 (RUN/'inputs/I2_structured_evidence_test.jsonl').write_text('\n'.join(json.dumps(x) for x in out)+'\n'); print(json.dumps({'rows':len(out),'gold_in_prompt':False}))
if __name__=='__main__': main()
