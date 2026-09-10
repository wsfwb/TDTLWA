from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def main():
 x=json.loads((RUN/'metadata/consensus_status.json').read_text()); assert x['calls_made']==0 and set(x['consensus_policies'])=={'majority','2_of_3','3_of_3'}; print('test_consensus_policy: PASS')
if __name__=='__main__':main()
