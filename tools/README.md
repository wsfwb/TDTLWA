# Tools

Utilities for validating the pipeline without IEMOCAP data or API access.

## `make_synthetic_fixture.py`

Generates a tiny synthetic dataset that mimics the pipeline's data contracts:
24 fake utterances in 4 fake dialogues, an E20-style `test_enriched.npz`
(`F1` Full/D3 probabilities, `F0` scalar diagnostics, `F2` modality summary),
hidden gold labels, and 72 mock reasoner responses that pass the real
`E26_OUTPUT_SCHEMA.json`.

```bash
python3 tools/make_synthetic_fixture.py --out /tmp/clarifymer_fixture
```

The mock reasoner is deliberately imperfect (~50% accurate, confidently
overrides only when correct) so the fusion demo has real signal to find.

## `run_fusion_on_fixture.py`

Fills the E26-style prediction table from the mock caches and replays an
E27-family policy (accept I1's `ACCEPT_REASON` override when self-confidence
≥ threshold), scoring weighted-F1 against the hidden gold:

```bash
python3 tools/run_fusion_on_fixture.py --fixture /tmp/clarifymer_fixture
```

Expected output shape:

```
Full W-F1:       0.55
E27-gate thr=0.7: 0.66  (overrides=7)
E27-gate thr=0.8: 0.66  (overrides=7)
E27-gate thr=0.9: 0.63  (overrides=4)
```

This is the whole method in miniature: a frozen "backbone" prediction, three
cached reasoner views, and a confidence-gated override policy searched offline.
