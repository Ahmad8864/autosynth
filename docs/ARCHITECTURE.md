# Architecture and internals

This document covers the runtime, item states, acceptance policies, run database,
meta-optimization, and batch dispatch. Configuration defaults remain in code so this guide
does not become a second source of truth.

## Runtime

autosynth uses an event-sourced pipeline backed by SQLite. The pure `step()` function advances
an item from its current state and returns any new LLM requests. The dispatcher fulfills those
requests, and the store records each transition and response.

```
pipeline.step()        pure state machine: (state, responses) -> (state, requests)
dispatcher             reads ready items, calls step(), fulfills requests
  ├─ fulfill_local     threadpool over HTTP
  └─ fulfill_batch     provider batch APIs (see "Batch mode" below)
store                  SQLite + WAL, one run.db per run
llm                    provider routing, rate-limit, retry, cost accounting
```

Purity keeps state transitions deterministic, while persisted requests and responses make
resume possible. After a restart, local in-flight requests return to pending. Batch requests
keep their batch IDs and resume polling.

## Item state machine

```
PENDING → NEED_CANDIDATE → NEED_QUALITY → NEED_SCORES
              ↑                              │
              │                    [NEED_DECISION]
              │                              │
              └─ NEED_REFLECTION ← improve ──┤
                                             │
                                      [NEED_AUDIT] → ACCEPTED
                                             │
                                      audit failure

REJECTED is terminal when the round limit or a fatal error is reached.
```

`NEED_DECISION` is used only by judge-driven acceptance, and `NEED_AUDIT` only when final
auditing is enabled. Judge feedback can start the next challenger round directly without a
reflection request.

Without short-circuiting, `NEED_SCORES` emits all weak and strong solver requests together.
With `loop.short_circuit_strong`, it emits weak requests first and releases strong requests
only after the weak gate passes. Rubric judges are requested as solver responses arrive.
`cfg.dispatcher.concurrency` bounds concurrent work. State definitions and transitions live
in [`pipeline.py`](../src/autosynth/pipeline.py).

## Acceptance

`acceptance.mode` selects one of three policies. If unset, the domain's
`default_acceptance_mode` is used.

- **rubric** (default) — a judge scores each attempt. Fixed thresholds check weak performance,
  strong performance, and the gap between them.
- **verifiable** — `DomainAdapter.verify()` checks attempts in process. No rubric judge is
  called; correct-answer counts determine acceptance.
- **judge** — attempts are rubric-scored, then a loop judge decides whether to accept or
  improve the candidate. Its feedback goes directly to the next challenger round.

The current thresholds and limits are fields on
[`AcceptanceConfig`](../src/autosynth/config.py). Policy logic lives in
[`acceptance.py`](../src/autosynth/acceptance.py).

### Conditional strong evaluation

`loop.short_circuit_strong: true` runs weak attempts first. If they show that the example is
too easy, the round skips strong attempts and proceeds to improvement. This saves strong-model
and judge calls but adds another sequential stage to the round. It is enabled by default.

Judge mode has no separate weak threshold, so short-circuiting does not skip work there;
disable it if the added serialization is not useful.

### Final audit

With `audit.enabled: true`, a passing round enters `NEED_AUDIT` before acceptance. The auditor
can inspect the grounding source and scored attempts, which are not available to the earlier
quality check. Audit failures become challenger feedback for another round, subject to
`loop.max_rounds`.

The auditor uses the judge model by default. Configure `auditor` to use another model, override
`DomainAdapter.audit_prompt()` for a domain-specific prompt, or add harness `audit_rules`.

## Run database

Everything for a run lives under `outputs/<run_id>/`:

- `run.db` — SQLite + WAL and the source of truth for the run.
- `config.snapshot.yaml` — the exact config used. Resume reads this if you don't pass `--config`.
- `accepted.jsonl` / `hf_export/` — produced on `autosynth export`, not written automatically.

The schema is defined in [`store/schema.py`](../src/autosynth/store/schema.py) and can be
inspected with `sqlite3 outputs/<run_id>/run.db .schema`. Current tables are `runs`, `items`,
`rounds`, `requests`, `responses`, `solver_scores`, and `accepted`.

Accepted records include the domain payload, reference output, rubric, source metadata,
weak/strong summaries, per-attempt scores, and acceptance rationale. When enabled, the final
audit verdict is included as well. Domains can change the exported shape through
`DomainAdapter.format_accepted()`.

## Training exports

`autosynth export --format sft|dpo|grpo` converts accepted records into training datasets.
Use `--to jsonl|hf` to choose the output format. Solver prompts are rebuilt from the stored
candidate using the configured domain and the run's stored harness. Domain code is not
snapshotted, so changing `solver_prompt()` after a run can also change its exported prompts.

- **sft** — conversational message records. The completion is `reference_output`, or the
  highest-scoring eligible strong rollout with `--completion best-strong`.
- **dpo** — conversational preference records with an explicit prompt. Rollouts are ranked
  by score without regard to solver role, then the best and worst are paired. Score and
  margin thresholds apply in rubric mode; verifiable domains pair correct and incorrect
  attempts. Empty and truncated attempts are excluded.
- **grpo** — prompt records with columns for reward functions. Verifiable domains include a
  `solution`; rubric domains include `rubric` and `reference_output`. The `stats` column holds
  the weak and strong averages, standard deviations, and gap.

Every record includes `meta` provenance unless `--no-meta`. It contains model and sampling
settings, the harness fingerprint, acceptance statistics, and the audit verdict. Exports are
deduplicated by rendered prompt and sorted by item ID. Transforms live in
[`export.py`](../src/autosynth/export.py).

## Meta-optimization

`autosynth metaopt` evolves a `HarnessSpec`: role-specific instruction lists plus a small set
of structural settings. Harness rules are appended to each agent's system prompt. The model
and implementation live in [`harness.py`](../src/autosynth/harness.py) and
[`metaopt/`](../src/autosynth/metaopt/).

The loop is:

1. Evaluate the seed harness on training and validation items.
2. Select an accepted parent with Boltzmann sampling over mean validation scores.
3. Summarize failures from the parent's latest training run and ask the mutator for a structured
   edit.
4. Evaluate the child on training and validation items.
5. Accept the child only when its validation score exceeds the parent's running validation mean.

Re-evaluated parent scores are averaged to reduce the effect of a single noisy run. Training
results provide failure examples but do not decide acceptance. Mutations change the harness,
not Python source. Population, lineage, and per-iteration decisions are written under
`outputs/metaopt/<run_id>/iterations/`.

To run for real, add `metaopt: { enabled: true, max_iterations: 50, ... }` to your config and
point `metaopt.mutator` at a strong reasoning model. Meta-opt reuses your existing `domain`,
`acceptance`, `loop`, and agent settings.

## Batch mode

The dispatcher can submit requests through provider batch APIs instead of sending them
immediately. Batch pricing is often lower, but discounts and completion windows depend on the
provider. Enable it with `dispatcher.mode: batch`; use `batch_provider: mock` for an offline
run.

In batch mode, pending requests are grouped, submitted, and polled until the provider marks
them complete. The pipeline state machine does not change while the provider is processing a
batch. On resume, requests with a batch ID continue polling.

Three implementations of `BatchProvider` live in
[`dispatcher/batch.py`](../src/autosynth/dispatcher/batch.py):

- **`LiteLLMBatchProvider`** (`batch_provider: openai`, default) — an OpenAI-style
  upload, create, poll, and download flow through LiteLLM.
- **`AnthropicBatchProvider`** (`batch_provider: anthropic`) — Anthropic's native Message
  Batches REST API.
- **`MockBatchProvider`** (`batch_provider: mock`) — in-process, for tests and offline demos.

If a completed batch omits a tagged request, that request is failed and can retry or reach its
failure cap. It is not left in flight indefinitely.

### Batch limits

Batch requests use plain JSON mode rather than strict response schemas. OpenAI-style providers
receive `{"type": "json_object"}`; Anthropic relies on the prompt because its Messages API has
no `response_format` parameter. Per-request cost uses LiteLLM list prices and does not subtract
batch discounts, so budget checks may stop early. A run assumes one batch provider.
