from pathlib import Path
import json
import os
RUN=Path(os.environ.get('CLARIFYMER_RUN_DIR', Path(__file__).resolve().parents[1]))
def main():
 for fn in ('I1_long_context_test.jsonl','I2_structured_evidence_test.jsonl','I3_multimodal_summary_test.jsonl'):
  for line in (RUN/'inputs'/fn).read_text().splitlines():
   x=json.loads(line); assert not ({'gold','label','correctness','benefit','harm','test_wf1'} & set(k.lower() for k in x)); assert 'canonical_id' in x
 print('test_no_future_or_gold_in_prompt: PASS')
if __name__=='__main__':main()
