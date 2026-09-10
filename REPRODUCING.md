# Reproducing the experiments

Three levels of reproduction, in increasing order of external requirements.
Levels 1–2 need **no IEMOCAP data and no API calls**. Level 3 is the full chain.

> **Why data is not shipped:** IEMOCAP is distributed by USC SAIL under a
> license that forbids redistribution. Obtain it at
> <https://sail.usc.edu/iemocap/> (research license, free for academic use).
> The pipeline never embeds IEMOCAP text or labels in this repository.

## Level 1 — unit-level sanity checks (no data needed)

Every phase ships leak-guard and behavior tests:

```bash
python3 metric_aligned_agent_phase_e22/tests/test_long_context_order.py
python3 metric_aligned_agent_phase_e22/tests/test_structured_evidence_schema.py
```

(`test_no_future_or_gold_in_prompt` additionally checks a run directory's
`inputs/` files; point `CLARIFYMER_RUN_DIR` at one after Level 2.)

## Level 2 — end-to-end on synthetic data

Verified end-to-end demo (no IEMOCAP, no API):

```bash
python3 tools/make_synthetic_fixture.py --out /tmp/clarifymer_fixture
python3 tools/run_fusion_on_fixture.py --fixture /tmp/clarifymer_fixture
```

This exercises the real data contracts (E26 prediction table, I1/I2/I3 cache
records, `E26_OUTPUT_SCHEMA.json`) and replays an E27-family confidence-gated
override policy — the method in miniature. The E22 input builders also run on
the fixture:

```bash
mkdir -p /tmp/e22run/{scripts,inputs,metadata}
cp metric_aligned_agent_phase_e22/scripts/build_*.py /tmp/e22run/scripts/
export CLARIFYMER_RUN_DIR=/tmp/e22run \
       CLARIFYMER_E20_DIR=/tmp/clarifymer_fixture/e20 \
       CLARIFYMER_CANONICAL_CSV=/tmp/clarifymer_fixture/canonical.csv
for b in long_context_inputs structured_evidence multimodal_evidence; do
  python3 /tmp/e22run/scripts/build_$b.py
done
export CLARIFYMER_RUN_DIR=$PWD/metric_aligned_agent_phase_e22  # + inputs/ present
python3 metric_aligned_agent_phase_e22/tests/test_no_future_or_gold_in_prompt.py
```

## Level 3 — the real IEMOCAP chain

### Step 0 — prerequisites

| Requirement | Source |
|---|---|
| IEMOCAP (Session 1–5, full release) | USC SAIL license |
| TDTL code | public fork: `github.com/wsfwb/tdtl-anonymous` (`IEMOCAP/{dataset.py, model.py, multimodel_fusion.py}`) |
| TDTL pretrained encoders + checkpoint | train via the fork's protocol, or use your own |
| LLM API (Responses-API compatible) | set `~/.codex/auth.json` (`OPENAI_API_KEY`) and `~/.codex/config.toml` (`base_url`), or patch `load_auth()` in `metric_aligned_agent_phase_e24/scripts/direct_responses_client.py` |
| Python deps | `numpy pandas scikit-learn openai httpx` (+ `torch` only for Step 1) |

### Step 1 — backbone features & Full logits (external, frozen)

Using the `tdtl-anonymous` code and the original TDTL preprocessing, produce:

- `train_features.pkl / dev_features.pkl / test_features.pkl` — the native
  Session-5 split (5,163 / 647 / 1,623 utterances)
- Full-model logits for the test split (this is the 0.734585 baseline)

The pipeline reads these **read-only**. E11's `run_phase_e11.py` +
`finalize_e11.py` document the snapshot/verify protocol used to prove the
backbone was untouched (hash snapshots before and after).

### Step 2 — D3 residual (this repo, phase e11)

`metric_aligned_agent_phase_e11/scripts/benchmark_controllers.py` implements
D3: PCA(32) → StandardScaler → class-weighted Ridge(alpha=75) on
`5*(onehot − p)`, prediction capped at ±0.15 per class and norm-capped at 3.0.
Fit on dev, apply to test.

### Step 3 — build the three reasoner inputs (phase e22)

The builders produce the I1/I2/I3 JSONL files (1,623 rows each):

```bash
export CLARIFYMER_CANONICAL_CSV=/path/to/canonical_iemocap_utterances.csv  # transcript bridge
export CLARIFYMER_E20_DIR=/path/to/e20_dir                                # needs features/test_enriched.npz
python3 metric_aligned_agent_phase_e22/scripts/build_long_context_inputs.py    # -> I1
python3 metric_aligned_agent_phase_e22/scripts/build_structured_evidence.py    # -> I2
python3 metric_aligned_agent_phase_e22/scripts/build_multimodal_evidence.py    # -> I3
```

Schema of what each interface exposes is in
`metric_aligned_agent_phase_e24/scripts/run_direct_reasoner.py` (`ALLOW` /
`FORBIDDEN`). The leak-guard tests from Level 1 must pass on your generated
files before proceeding.

### Step 4 — call the reasoner (phases e24 → e26, e29r later)

Cost warning: the original run was **3 interfaces × 1,623 calls** with
`reasoning_effort=xhigh`; budget accordingly. Execution order and gates are in
`protocol/PHASE_E26_PROTOCOL.md` (train-only probes → 20 canaries → then, and
only then, the test split):

```bash
python3 metric_aligned_agent_phase_e24/scripts/run_direct_reasoner.py   # canary stage
python3 metric_aligned_agent_phase_e26/scripts/run_e26.py               # I1, I2, I3
```

Calls are cached content-addressed (`cache_identity` = SHA-256 of run,
interface, canonical id, prompt, inputs, schema); re-runs skip completed calls.

### Step 5 — frozen evaluation + fusion search (phases e27 → e33, no API)

```bash
python3 metric_aligned_agent_phase_e26/scripts/evaluate_e26_offline.py  # E26 W-F1 (0.741709)
# then, per phase: audit → bootstrap → materialize → evaluate → finalize
```

E29R adds 200 targeted rescue calls (`run_e29_adjudicator.py`); everything
after that (E27, E30, E31, E32, E33) is pure offline search over the caches —
no API cost. The E33 search enumerates ~29,690 declarative candidates across
families R0–R6 and reports the winner
(`base=E31, family=R1_RAW_TUPLE, tuple=[0,4,0,0], source=I1, k=5`,
W-F1 0.752767).

### Expected milestones

| After step | W-F1 |
|---|---|
| 1 (Full logits) | 0.734585 |
| 2 (+ D3) | 0.733103 |
| 4 (+ E26 reasoner) | 0.741709 |
| 5 (… → E33) | 0.752767 |

Exact numbers depend on your retrained backbone (the original used a specific
checkpoint; same architecture retrains land within ~±0.005) and on LLM
nondeterminism at the margins.

## Honest-result reminder

Every W-F1 above the E26 line is `same_split_test_selected_posthoc`: fusion
policies were selected on the same test set they report. Reproduce them as
such — they measure what the policy class can extract from the reasoner
cache, not clean-deployment generalization.
