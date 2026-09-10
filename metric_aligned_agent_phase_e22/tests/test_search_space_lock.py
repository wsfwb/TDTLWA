from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def main():
 s=json.loads((RUN/'protocol/E22_SEARCH_SPACE.json').read_text()); assert s['interfaces']==['I0','I1','I2','I3','I4','I5']; assert s['confidence_thresholds']==[.4,.5,.6,.7,.8]; assert s['reason_budgets']==[.05,.1,.15,.2,.3,.5,1.0]; print('test_search_space_lock: PASS')
if __name__=='__main__':main()
