from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def main():
 s=json.loads((RUN/'protocol/E22_PROMPT_SCHEMA.json').read_text()); assert 'parse_fallback' in s and 'cheap action' in s['parse_fallback']; print('test_json_parse_fallback: PASS')
if __name__=='__main__':main()
