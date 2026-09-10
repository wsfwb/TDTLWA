from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def main():
 for line in (RUN/'inputs/I1_long_context_test.jsonl').read_text().splitlines():
  x=json.loads(line); idx=[t['turn_index'] for t in x['preceding_turns']]; assert idx==sorted(idx); assert all(t['turn_index']<x['turn_index'] for t in x['preceding_turns'])
 print('test_long_context_order: PASS')
if __name__=='__main__':main()
