#!/usr/bin/env python3
"""Fill the E26 prediction table from mock caches, then replay E27-style gating.

Demonstrates the offline fusion logic end-to-end on synthetic data:
read caches -> per-utterance reasoner decisions -> simple single-source override
gates (E27 family) -> weighted F1 against the fixture's hidden gold.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

LABELS = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']  # legacy order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', default='/tmp/clarifymer_fixture')
    a = ap.parse_args()
    fx = Path(a.fixture)
    preds = {r['canonical_id']: r for r in csv.DictReader((fx / 'evaluation/results/e26_canonical_predictions.csv').open())}
    gold = {r['canonical_id']: int(r['gold']) for r in csv.DictReader((fx / 'splits/secret_gold.csv').open())}
    for iface, col in (('I1', 'i1'), ('I2', 'i2'), ('I3', 'i3')):
        for cid in preds:
            out = json.loads((fx / f'caches/{iface}/{cid}.json').read_text())['output']
            preds[cid][f'{col}_reason_prediction'] = out['final_label']
            preds[cid][f'{col}_decision'] = out['decision']
            preds[cid][f'{col}_confidence'] = out['confidence']
    ids = list(preds)
    gold_arr = np.array([gold[c] for c in ids])
    full = np.array([int(preds[c]['full_prediction']) for c in ids])
    def wf1(x): return f1_score(gold_arr, np.asarray(x), average='weighted')
    print(f"Full W-F1:       {wf1(full):.4f}")
    for thr in (0.7, 0.8, 0.9):
        fused, overrides = [], 0
        for c in ids:
            r = preds[c]
            lab = r['i1_reason_prediction']
            if r['i1_decision'] == 'ACCEPT_REASON' and float(r['i1_confidence']) >= thr and lab in LABELS:
                fused.append(LABELS.index(lab)); overrides += 1
            else:
                fused.append(int(r['full_prediction']))
        print(f"E27-gate thr={thr}: {wf1(fused):.4f}  (overrides={overrides})")

if __name__ == '__main__':
    main()
