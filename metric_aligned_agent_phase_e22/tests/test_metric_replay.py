from pathlib import Path
import pandas as pd
RUN=Path(__file__).resolve().parents[1]
def main():
 d=pd.read_csv(RUN/'results/candidate_leaderboard.csv'); a=d[d.candidate.eq('P2_full_low_conf_top10__V2_benefit_harm_logistic__KEEP__th0.4__lh4__b0.05__cache_r1__c1')].iloc[0]; assert abs(float(a.weighted_f1)-.737106221305)<1e-6; assert abs(float(d[d.candidate.eq('Always Reason')].weighted_f1.iloc[0])-.533051798642)<1e-6; print('test_metric_replay: PASS')
if __name__=='__main__':main()
