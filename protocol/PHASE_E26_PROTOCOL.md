# E26 — Quota-recovered Sol-xhigh direct Responses API replay

E26 is a new, isolated attempt after the user reported that direct API quota was restored. E25 is read-only historical diagnosis and is not reinterpreted.

## Fixed direct API contract

- Model: `gpt-5.6-sol`
- Reasoning effort: `xhigh`
- Wire API: direct HTTP Responses API (`POST /responses`)
- Output: `E26_OUTPUT_SCHEMA.json`
- Tools, tool choice, Codex exec, workspace, previous response, conversation state: absent
- V3 payload only: `model`, `input`, `reasoning`, `text.format=json_schema`
- `max_output_tokens`, `temperature`, and `store` are omitted.
- Concurrency and in-flight requests: one.

## Gated sequence

1. Three Train-only no-gold probes must produce 3/3 HTTP-200 schema-valid JSON, no tool/function output.
2. Twenty Train-only canaries must produce at least 19 valid JSON outputs, no schema/tool failures, and finish with five consecutive successes.
3. Only after both gates may I1, then I2, then I3 consume Session-5 inputs. I4 and I5 are not run.
4. Session-5 labels remain unread until all three interfaces are complete and frozen.

## Failure policy

HTTP 200 without a valid message JSON is a stop condition: E26 saves a redacted response envelope including HTTP status, content type, incomplete details, item types, and usage. Transient 429/5xx/network failures retry at most six times with 15/30/60/120/300-second backoffs, respecting Retry-After. No stage begins after an incomplete prior stage.

Any final test result, if obtained later, is `same-split test-selected posthoc exploratory result` and not a clean deployment estimate.
