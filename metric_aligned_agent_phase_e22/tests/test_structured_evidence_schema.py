from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def main():
 x=json.loads((RUN/'inputs/I2_structured_evidence_test.jsonl').read_text().splitlines()[0]); req={'canonical_id','full_prediction','residual_prediction','full_probability','residual_probability','full_confidence','residual_confidence','agreement'}; assert req<=set(x); assert len(x['full_probability'])==6 and len(x['residual_probability'])==6; print('test_structured_evidence_schema: PASS')
if __name__=='__main__':main()
