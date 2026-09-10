#!/usr/bin/env python3
"""Generate a tiny synthetic fixture that mimics the pipeline's data contracts.

Produces (under --out):
  canonical.csv                        transcript bridge (E05 format, fake text)
  e20/features/test_enriched.npz       Full/D3 states (E20 format)
  e26-style prediction table           canonical predictions with I1/I2/I3 slots
  caches/I1|I2|I3/*.json               mock reasoner responses (schema-valid)
  splits/test_ids.csv                  canonical id list

24 fake dialogues x 3 turns. No IEMOCAP content is used.
"""
import argparse, csv, hashlib, json, os
from pathlib import Path
import numpy as np

LABELS = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']          # legacy order
FAKE_TEXTS = [
    "Wow, that is amazing news!", "I cannot believe this happened.", "Fine, whatever you say.",
    "How dare you say that to me!", "Yes! Finally it worked out!", "This is so frustrating...",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/clarifymer_fixture')
    ap.add_argument('--n', type=int, default=24, help='number of utterances (multiple of 3)')
    a = ap.parse_args()
    out = Path(a.out); (out / 'e20/features').mkdir(parents=True, exist_ok=True)
    for d in ('caches/I1', 'caches/I2', 'caches/I3', 'evaluation/results', 'splits'):
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n = a.n - a.n % 3

    # canonical bridge + ids
    rows, ids = [], []
    for i in range(n):
        dia = f'SYN{1 + i // 6:02d}_impro{i // 6:02d}'
        spk = 'SYN_A' if i % 2 == 0 else 'SYN_B'
        cid = f'{dia}_{spk[-1]}{i % 3:03d}'
        ids.append(cid)
        rows.append({'canonical_sample_id': cid, 'session': 'SYN01', 'dialogue_id': dia,
                     'speaker': spk, 'utterance_index': i % 3,
                     'transcript_raw': FAKE_TEXTS[i % len(FAKE_TEXTS)],
                     'transcript_normalized': FAKE_TEXTS[i % len(FAKE_TEXTS)].lower(),
                     'audio_source': '', 'video_source': '', 'start_time': '', 'end_time': ''})
    with (out / 'canonical.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out / 'e20/features/test_context_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['canonical_id']); [w.writerow([c]) for c in ids]
    with (out / 'splits/test_ids.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['original_iemocap_id', 'split']); [w.writerow([c, 'test']) for c in ids]

    # E20-style enriched states (identity: index = row order)
    p_full = rng.dirichlet(np.ones(6) * 2.0, size=n)
    p_res = rng.dirichlet(np.ones(6) * 2.0, size=n)
    F1 = np.concatenate([p_full, p_res], axis=1)
    F0 = np.zeros((n, 48))
    F0[:, 36] = p_full.max(1); F0[:, 37] = p_res.max(1)
    F0[:, 38] = -(p_full * np.log(p_full + 1e-9)).sum(1)
    F0[:, 39] = -(p_res * np.log(p_res + 1e-9)).sum(1)
    F0[:, 40] = np.sort(p_full, 1)[:, -1] - np.sort(p_full, 1)[:, -2]
    F0[:, 41] = np.sort(p_res, 1)[:, -1] - np.sort(p_res, 1)[:, -2]
    F2 = rng.normal(size=(n, 16)).astype(np.float32)   # modality summary vector (E20 F2)
    np.savez(out / 'e20/features/test_enriched.npz', ids=np.array(ids),
             F1=F1.astype(np.float32), F0=F0.astype(np.float32), F2=F2,
             full_pred=p_full.argmax(1), residual_pred=p_res.argmax(1))

    # E26-style prediction table (empty reasoner columns to fill)
    fields = ['canonical_id', 'gold_label_evaluation_only', 'full_prediction', 'residual_prediction',
              'i1_reason_prediction', 'i1_decision', 'i1_confidence', 'i1_uncertainty',
              'i2_reason_prediction', 'i2_decision', 'i2_confidence', 'i2_uncertainty',
              'i3_reason_prediction', 'i3_decision', 'i3_confidence', 'i3_uncertainty']
    with (out / 'evaluation/results/e26_canonical_predictions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, cid in enumerate(ids):
            w.writerow({'canonical_id': cid, 'gold_label_evaluation_only': '',
                        'full_prediction': int(p_full[i].argmax()), 'residual_prediction': int(p_res[i].argmax())})

    # hidden gold, correlated with full_pred (backbone is right ~60% of the time)
    gold = np.where(rng.random(n) < 0.6, p_full.argmax(1), rng.integers(0, 6, size=n))
    with (out / 'splits/secret_gold.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['canonical_id', 'gold'])
        [w.writerow([c, int(gold[i])]) for i, c in enumerate(ids)]
    # mock schema-valid reasoner caches; mock LLM is right ~50% of the time and
    # confidently overrides in those cases (so fusion gates can find real signal)
    schema = json.loads(Path(__file__).resolve().parents[1].joinpath('protocol/E26_OUTPUT_SCHEMA.json').read_text())
    correct = rng.random((n, 3)) < 0.5
    for j, iface in enumerate(('I1', 'I2', 'I3')):
        for i, cid in enumerate(ids):
            lab = LABELS[gold[i]] if correct[i, j] else LABELS[int(rng.integers(0, 6))]
            sure = bool(correct[i, j] and rng.random() < 0.8)
            rec = {'run_id': 'SYNTHETIC', 'status': 'success', 'model': 'mock', 'reasoning_effort': 'none',
                   'temperature': 0.0, 'interface': iface, 'canonical_id': cid,
                   'prompt_sha256': hashlib.sha256(f'{iface}{cid}'.encode()).hexdigest(),
                   'input_sha256': hashlib.sha256(cid.encode()).hexdigest(),
                   'schema_sha256': hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
                   'cache_identity': hashlib.sha256(f'{iface}{cid}syn'.encode()).hexdigest(),
                   'output': {'final_label': lab,
                              'decision': 'ACCEPT_REASON' if sure else 'KEEP_CHEAP',
                              'confidence': round(float(rng.uniform(0.82, 0.99)) if sure else rng.uniform(0.3, 0.7), 2),
                              'uncertainty': round(float(rng.uniform(0.0, 0.15)) if sure else rng.uniform(0.2, 0.5), 2),
                              'top2': [], 'rationale': 'synthetic rationale'},
                   'latency_sec': 0.01}
            (out / f'caches/{iface}/{cid}.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({'out': str(out), 'utterances': n, 'caches': 3 * n}))

if __name__ == '__main__':
    main()
