# ClarifyMER Reasoner Pipeline (IEMOCAP chain)

Pure-code snapshot of the LLM-reasoner cache fusion pipeline that produced
**weighted-F1 0.752767** on the IEMOCAP Session-5 test set (1,623 utterances),
starting from a TDTL baseline of **0.734585**.

This repository contains **only pipeline code** — no datasets, no model weights,
no LLM response caches, and no TDTL backbone code.

- **[DESIGN.md](DESIGN.md)** — the method narrative: why each phase exists, label
  conventions, interface rationale, fusion-search design, leak-prevention, glossary.
- **[protocol/](protocol/)** — the frozen LLM I/O contracts (output JSON schemas,
  gated call sequence, retry/failure policy).
- [TDTL interface](#tdtl-interface-contract) — the contract the backbone must satisfy.

## Pipeline at a glance

```
TDTL backbone (frozen, external)
        │  full logits (6-class, IEMOCAP)
        ▼
[E11] D3 residual corrector ── PCA-32 + class-weighted Ridge residual on logits
        │
        ▼
[E24/E26] gpt-5.6-sol reasoner, 3 interfaces × 1,623 test utterances (real API calls)
        │    I1: long context   I2: structured evidence   I3: modality summary
        ▼
[E29R] +200 targeted reasoner calls (adjudicator / rescue round)
        │
        ▼
[E27 → E32] offline layered fusion search over the reasoner cache
        │    E27 first-layer fusion → E30 → E31 → E32 (progressively richer policies)
        ▼
[E33] post-hoc tuple-rule search (29,690 candidates)
             winner: match samples with (base, I1, I2, I3) = (hap, exc, hap, hap),
             take the 5 most self-confident reasoner outputs, override to `excited`
```

No step re-trains or modifies the TDTL backbone. E11 only re-runs training under the
identical architecture to produce a clean aligned run (the "Reconstructed Full",
0.736588); all later phases read its outputs read-only.

## Results ladder (IEMOCAP, native Session-5 test, weighted F1)

| Stage  | What it is                                            | W-F1     |
|--------|-------------------------------------------------------|----------|
| Full   | Historical TDTL checkpoint (`multimodal_fusion_best`) | 0.734585 |
| Full'  | Same-architecture retrain ("Reconstructed Full")      | 0.736588 |
| D3     | + residual corrector                                  | 0.733103 |
| E26    | + per-utterance reasoner decisions (3 interfaces)     | 0.741709 |
| E27    | + first-layer cache fusion                            | 0.744488 |
| E29R   | + 200 targeted rescue calls                           | 0.747463 |
| E30    | + layered fusion, round 2                             | 0.748743 |
| E31    | + layered fusion, round 3                             | 0.749893 |
| E32    | + layered fusion, round 4                             | 0.749926 |
| **E33**| + post-hoc tuple rule (this repo's headline)          | **0.752767** |

## Repository layout

Directory names are the original phase names, kept for provenance.

```
metric_aligned_agent_phase_e11/   Backbone alignment + D3/V1/V3 residual controllers
  scripts/run_phase_e11.py          self-contained runner: snapshots TDTL, retrains
  scripts/benchmark_controllers.py  D3 implementation (PCA-32 + Ridge alpha=75, cap 0.15)
  scripts/finalize_e11.py           provenance report: hashes + git status of TDTL
  scripts/pre_eval_alignment.py     Session-5 split identity preflight
metric_aligned_agent_phase_e24/   Reasoner infrastructure (dry-run/transport layer)
  scripts/run_direct_reasoner.py    prompt builder + I1/I2/I3 input whitelists
  scripts/direct_responses_client.py direct POST /responses client
  scripts/cache_store.py            content-addressed cache identities
metric_aligned_agent_phase_e26/   The 3×1,623 real reasoner call round
  scripts/run_e26.py                main loop (cache-aware, resumable)
  scripts/input_contract.py         per-interface input construction
  scripts/evaluate_e26_offline.py   frozen cache-only evaluation (W-F1 + bootstrap)
  scripts/supervise_e26.py          supervised execution with rate control
metric_aligned_agent_phase_e27/   First-layer fusion search over the cache
metric_aligned_agent_phase_e29r/  Targeted rescue round (+200 calls, adjudicator prompts)
metric_aligned_agent_phase_e30/   Layered fusion round 2 (policy core: e30_policy_core.py)
metric_aligned_agent_phase_e31/   Layered fusion round 3
metric_aligned_agent_phase_e32/   Layered fusion round 4
metric_aligned_agent_phase_e33/   Final post-hoc tuple-rule search
  scripts/e33_policy_core.py        candidate families R0–R6, tuple masks, scoring
  scripts/evaluate_e33.py           W-F1 evaluation of every candidate
  scripts/finalize_e33.py           winner materialization + report
```

Each phase directory follows the same pattern:
`audit inputs → bootstrap/build features → materialize candidates → evaluate → finalize/report`.

## Reasoner interfaces

Three independent views of each utterance are sent to `gpt-5.6-sol`
(`reasoning_effort=xhigh`, `temperature=0.0`, JSON-schema-strict output,
direct `POST /responses`). The prompt enforces a `KEEP_CHEAP` / `ACCEPT_REASON`
decision and forbids external tools, labels, or outcome information.

Input fields are whitelisted per interface (`run_direct_reasoner.py`):

| Interface | Allowed fields |
|-----------|----------------|
| **I1** long context | `canonical_id`, `dialogue_id`, `speaker`, `turn_index`, `dialogue_length`, `current_utterance`, `preceding_turns`, `serialized_context`, `truncated_fixed_head_tail` |
| **I2** structured evidence | `full/residual prediction/probability/confidence/entropy/margin`, `agreement`, `confidence_delta`, `entropy_delta`, `margin_delta`, `label_names` |
| **I3** modality summary | `modality_summary_vector`, `modality_summary_dimensions`, `missing_modality_indicator`, `raw_audio_available`, `raw_video_available` |

A global `FORBIDDEN` set rejects any field that could leak labels or test metrics:
`gold`, `session5_label`, `correctness`, `benefit`, `harm`, `neutral`, `test_wf1`,
`test_metric`, `future_outcome`, `r1_prediction`, `r1_confidence`, `oracle`.

## TDTL interface contract

The backbone is **not included** in this repository. To run the pipeline you need
a frozen TDTL-style model that provides:

1. **Feature pickles** per split (`train_features.pkl`, `dev_features.pkl`,
   `test_features.pkl`) — precomputed text/audio/visual features per utterance.
2. **Full logits** — 6-class logits for every test utterance, in the label order
   `[happy, sad, neutral, angry, excited, frustrated]` (the "legacy" order).
3. **Read-only usage** — the pipeline never writes into the backbone directory;
   E11 snapshots file hashes and git status before touching anything and
   verifies them after.

All later phases consume per-utterance records keyed by `canonical_id`
(one row per IEMOCAP test utterance, n=1,623), joined with the reasoner cache
by `cache_identity` (a SHA-256 of run id + interface + id + prompt + inputs + schema).

## E33 winning rule

Searched over 29,690 offline candidates (families: fixed replay, raw tuple,
vote, decision pattern, adjudicator gate, two-rule, seeded). The winner:

```json
{"base": "E31", "family": "R1_RAW_TUPLE", "k": 5,
 "rank": "confidence_minus_uncertainty", "source": "I1", "tuple": [0, 4, 0, 0]}
```

Meaning: for test samples whose `(base, I1, I2, I3)` label tuple equals
`(happy, excited, happy, happy)` — 28 such samples — override the top-5 ranked by
reasoner confidence-minus-uncertainty to `excited`. 4 of the 5 overrides are
correct, 1 is wrong; BENEFIT captured 5/5, HARM 0.

`reasoner_api_calls=0` at this stage — E33 is a pure replay of the existing cache.

## Important caveats

- **This is not a clean-deployment result.** The full ladder including E33 was
  selected post hoc on the same test set it is evaluated on
  (`same_split_test_selected_posthoc`). It should be read as an upper-bound /
  diagnostic analysis of what the reasoner cache contains, not an unbiased
  generalization estimate. The reports shipped in the original project state this
  explicitly; readers should quote the number with that qualifier.
- The TDTL backbone directory is treated as immutable throughout; every phase
  snapshots hashes before/after.
- No new reasoner API calls are required to replay the fusion phases (E27–E33):
  they are offline searches over a persisted cache. The cache itself is **not**
  included in this repository.

## Running

The scripts were written to run inside their original run-directory layout
(relative `cache/`, `inputs/`, `results/`, `reports/` paths). They are provided
as-is to document the method; reproducing end-to-end requires:

1. An IEMOCAP feature setup matching the TDTL interface above.
2. API credentials for a Responses-API-compatible endpoint — loaded at runtime
   from local config files (`~/.codex/auth.json` / `~/.codex/config.toml` in the
   original environment); no credentials are in the code.
3. Executing phases in order E11 → E24 → E26 → E29R → E27 → E30 → E31 → E32 → E33.

## Dependencies

Python 3.10+, `numpy`, `pandas`, `scikit-learn`, `openai`, `httpx`
(torch only needed if you re-run the E11 backbone alignment yourself).
