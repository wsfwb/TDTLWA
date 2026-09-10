from pathlib import Path
import pandas as pd
RUN=Path(__file__).resolve().parents[1]
def main():
 p=pd.read_csv(RUN/'results/session5_canonical_predictions.csv'); assert len(p)==1623 and p.canonical_id.is_unique; print('test_canonical_alignment: PASS')
if __name__=='__main__':main()
