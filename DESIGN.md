# Method design notes — how and why this pipeline works

This document explains the *reasoning behind* the pipeline. `README.md` says
what each phase is; this says why the phases exist and how to think about them.

## 1. The problem

TDTL is a strong multimodal emotion-recognition backbone (text + audio +
video → 6-class logits), but it makes the same kind of mistakes everywhere:
confusing `happy` vs `excited`, `angry` vs `frustrated`, and missing the
conversational context that a text-only reading of one utterance would catch.
Fine-tuning the backbone further requires GPU training and risks breaking the
reproduced benchmark.

**Core idea:** leave the backbone frozen and add a *cheap-prediction +
LLM-reasoner* layer. For each test utterance, build three different textual
views of the evidence, ask a strong LLM (`gpt-5.6-sol`, reasoning effort
`xhigh`) to either confirm the backbone's prediction (`KEEP_CHEAP`) or
override it (`ACCEPT_REASON`), then fuse all of this offline.

The LLM never sees audio or video directly — only *structured descriptions*
of what the model computed. This is deliberate: it keeps the intervention
cheap (text-only API calls), auditable (every prompt/response cached), and
leak-safe (a whitelisted field set with a forbidden list enforced in code).

## 2. Label conventions (read this before reading code)

Two label orders coexist; confusing them is the #1 way to misread results.

| Space | Order | Where used |
|---|---|---|
| **legacy** | `[hap, sad, neu, ang, exc, fru]` | TDTL pickles, D3 residual, most cached arrays |
| **public/E26** | `[ang, sad, hap, neu, exc, fru]` | E26 evaluation, reports, paper |

Conversion: `legacy → public` is the permutation `[3,4,5,0,2,1]`
(i.e. public index `p` takes legacy index `LEGACY_TO_PUBLIC[p]`; see
`evaluate_e26_offline.py:LEGACY_TO_PUBLIC`).

So the E33 winner tuple `(0,4,0,0)` in the E26 space is
`(base=hap, I1=exc, I2=hap, I3=hap)` — base and two interfaces say happy,
I1 says excited, and the rule flips the most confident 5 of those 28 samples
to excited.

## 3. Why three interfaces?

Each interface is a different *hypothesis about what makes the LLM useful*:

- **I1 — long context.** Give the dialogue transcript around the utterance.
  Tests whether conversational context fixes backbone errors.
- **I2 — structured evidence.** Give the backbone's own self-diagnostics:
  probabilities, confidence, entropy, margin, full-vs-residual agreement.
  Tests whether the model's *uncertainty* signals identify correctable errors.
- **I3 — modality summary.** Give a compact numeric summary of the audio/visual
  branch states (missing-modality flags included). Tests whether the LLM can
  exploit what the multimodal encoders "felt" without seeing raw media.

They are called independently (one call per utterance per interface; 1,623 × 3
calls) so their votes can be fused as separate evidence sources later. Each
output is constrained by a JSON schema (`protocol/E26_OUTPUT_SCHEMA.json`):

```json
{"final_label": "exc", "decision": "ACCEPT_REASON",
 "confidence": 0.83, "uncertainty": 0.12, "top2": [...], "rationale": "..."}
```

`KEEP_CHEAP` / `ACCEPT_REASON` forces the model to take a stance relative to
the cheap prediction; `confidence` and `uncertainty` are self-reported and are
exactly what the later fusion layers rank by.

## 4. The action provenance vocabulary

Every fused prediction is annotated with *why it is what it is*:

- **KEEP** — kept the Full (backbone) prediction
- **RESIDUAL** — took the D3 residual corrector's prediction
- **REASON** — took a reasoner-derived prediction

`Reason rate` (2.71% base → 2.96% final in the E33 winner) = fraction of test
utterances where the final answer came from the reasoner chain. The whole
method's premise is that a *small, precisely targeted* override set beats
wholesale re-decoding.

## 5. The layered fusion search (E27 → E32)

The reasoner cache gives, per utterance: 3 LLM decisions + confidences +
uncertainties, the adjudicator's view (E29R), and the backbone/D3 predictions.
The fusion phases search for *selection policies* over this table:

- **E27** — first layer: simple gates (single-source overrides gated by
  agreement/consistency features).
- **E30** — policies that combine sources with confidence/uncertainty
  thresholds; ranks candidates by score modes like
  `confidence`, `margin`, `confidence_minus_uncertainty`.
- **E31 / E32** — same search with the previous layer's output added as a new
  candidate source, so each layer can only *keep or improve* on the last.

Every candidate is a declarative config (e.g. base source, gate condition,
k, rank mode); candidate ids are SHA-256 hashes of the config, which makes the
search fully reproducible and cache-addressable. Hundreds to tens of thousands
of configs are evaluated per layer — all offline, zero API calls after E29R.

## 6. Leak-prevention architecture

Three independent mechanisms, all enforced in code:

1. **Input whitelists** (`run_direct_reasoner.py:ALLOW`) — prompts are built
   only from approved fields; anything else raises.
2. **Forbidden-field rejection** (`FORBIDDEN` set) — any field whose name
   suggests gold labels, test metrics, or oracle outcomes aborts the run.
3. **Gated sequence** (`protocol/PHASE_E26_PROTOCOL.md`) — train-only probes
   and canaries must pass before any Session-5 input is sent, and Session-5
   gold is read only after all caches are frozen.

## 7. What the numbers mean (and don't mean)

The honest framing, from the reports: **the entire ladder is a same-split,
test-selected, post-hoc analysis.** Each fusion layer chose its best policy
*by looking at test W-F1*, so the gains are an upper bound on what the policy
class can extract from the cache, not a generalization estimate. A clean
deployment claim would require a held-out session or a pre-registered policy;
that is future work. The value demonstrated is:

1. a leak-controlled protocol for LLM-assisted correction of a frozen
   multimodal classifier, and
2. evidence that a tiny, targeted override set (~3% of utterances) selected
   from multi-view LLM evidence can lift W-F1 by ~1.8 points over the backbone.

## Glossary

| Term | Meaning |
|---|---|
| **Full** | TDTL backbone prediction (frozen model) |
| **Reconstructed Full / Full'** | Same-architecture retrain used to align artifacts (0.736588) |
| **D3** | PCA-32 + class-weighted Ridge residual corrector on Full logits |
| **I1/I2/I3** | The three reasoner interfaces (context / evidence / modality) |
| **E29R adjudicator** | Extra 200-call rescue round with adjudicator prompts |
| **KEEP / RESIDUAL / REASON** | Action provenance of a fused prediction |
| **cache_identity** | SHA-256 of (run, interface, id, prompt, inputs, schema) — content-addressed cache key |
| **Reason rate** | Fraction of final predictions that came from the reasoner chain |
| **BENEFIT / HARM** | Correct/incorrect interventions vs the base prediction |
| **post-hoc** | Policy selected using the same test set it reports |
