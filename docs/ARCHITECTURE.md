# Architecture & internals

Deeper reference than the [README](../README.md) — the runtime model, the item state
machine, how acceptance is decided, the run database, the meta-optimization loop, and batch
mode. Exact values (thresholds, defaults) are not transcribed here; they live in code, and
this doc points at the source so it can't drift.

## Runtime

An event-sourced pipeline over a SQLite database. A pure `step()` function advances item
state; the dispatcher fulfills LLM requests and writes the responses back; the store is the
durable record.

```
pipeline.step()        pure state machine: (state, responses) -> (state, requests)
dispatcher             reads ready items, calls step(), fulfills requests
  ├─ fulfill_local     threadpool over HTTP
  └─ fulfill_batch     provider batch APIs (see "Batch mode" below)
store                  SQLite + WAL, one run.db per run
llm                    provider routing, rate-limit, retry, cost accounting
```

The fact that `step()` is pure is the only reason resume works. Kill the process at any
point — including mid-batch — and `autosynth resume` picks up exactly where it left off.
In-flight local requests revert to pending; in-flight batch requests stay tagged and get
polled.

## Item state machine

```
PENDING → NEED_CANDIDATE → NEED_QUALITY → NEED_SCORES → ACCEPTED
                                              │
                                              └─ NEED_REFLECTION → (next round) ─┐
                                                                                 │
                              ACCEPTED / REJECTED are terminals  ◄───────────────┘
```

`NEED_SCORES` fans out `N × weak + N × strong` solver requests in parallel; each judge fires
the moment its solver lands. Concurrency is bounded by `cfg.dispatcher.concurrency`. The
state enum and transitions live in `src/autosynth/pipeline.py`; `test_pure_pipeline.py` is
exhaustive coverage, including the partial-completion no-op invariant.

## Acceptance

For each source item, autosynth runs the five-step loop (challenger → quality → weak/strong
solvers → judge → evaluator, then reflector on reject) until the candidate is accepted or
`loop.max_rounds` is exhausted. Three regimes decide acceptance, selected by
`acceptance.mode` (or the domain's `default_acceptance_mode`):

- **rubric** (default) — the judge scores each rollout against the rubric; acceptance is a
  threshold-and-gap test (weak must stay low, strong must clear a floor without being "too
  easy", and the strong−weak gap must exceed a minimum).
- **verifiable** — the domain's `verify()` checks each rollout programmatically; the judge is
  skipped and acceptance is a count gate: *weak must fail, strong must succeed.*
- **judge** — after the rubric judge scores every rollout, a loop-judge LLM reads the
  per-rollout weak/strong patterns and decides accept/improve each round, supplying the next
  round's suggestion directly (the reflector is skipped).

**Exact default thresholds and knobs** (`weak_avg_max`, `min_gap`, `verifiable_*`, the
rubric weight cap, …) are fields of `AcceptanceConfig` in `src/autosynth/config.py` — that
model's defaults and docstring are the source of truth; calibration notes map each to the
paper. The decision logic itself is `src/autosynth/acceptance.py`.

**Conditional strong eval (cost saver).** `loop.short_circuit_strong: true` runs the weak
solver first and only scores the strong solver when the weak gate passes (the example is hard
enough), skipping strong + judge calls on the easy majority at the cost of serializing the
round. Off by default; leave it off under `judge` mode, where it only adds serialization
without skipping anything.

## Run database

Everything for a run lives under `outputs/<run_id>/`:

- `run.db` — SQLite + WAL. Queryable with the `sqlite3` CLI and safe to share.
- `config.snapshot.yaml` — the exact config used. Resume reads this if you don't pass `--config`.
- `accepted.jsonl` / `hf_export/` — produced on `autosynth export`, not written automatically.

The authoritative schema is the database itself — `sqlite3 outputs/<run_id>/run.db .schema`,
generated from `src/autosynth/store/schema.py`. At time of writing the tables are `runs`,
`items`, `rounds`, `requests`, `responses`, `solver_scores`, and `accepted`. Each accepted
record carries `input`, `reference_output`, `rubric`, `domain`, `source_id`, `metadata`, the
weak/strong/gap scores, per-attempt solver scores, and the acceptance rationale; the exact
serialization is `DomainAdapter.format_accepted` (`src/autosynth/domain.py`).

`test_store.py` covers `claim_pending` atomicity under threads and resume normalization;
`test_dispatcher.py` covers end-to-end accept, concurrent fulfill, budget abort, and
kill/resume.

## Meta-optimization

`autosynth metaopt` runs the paper's secondary loop: evolve the orchestrator's *prompts* over
generations. The unit of evolution is a `HarnessSpec` — a structured bag of rule strings
injected into each agent's system prompt, plus a couple of numeric knobs
(`src/autosynth/harness.py`, `src/autosynth/metaopt/`).

The loop, roughly:

1. Score the seed harness on training and validation source items.
2. Each iteration: Boltzmann-sample a parent from the population (T=0.1 over training scores),
   summarize that parent's most recent rejection reasons, ask the mutator LLM for a structured
   diff, apply it, dedupe, and re-evaluate.
3. Accept the mutation only if `child.val > parent.val` — the paper's gate. Gating is on
   validation score alone, and repeated evaluations of a candidate are averaged so acceptance
   isn't decided by a single noisy run.

Mutations operate on the harness, not on Python source — that preserves the main lever the
paper exercises (prompt-text edits) without the sandboxing headache of a code-editing agent.
Swap in your own mutator for richer edits. Population, lineage, and per-iteration decisions
are written under `outputs/metaopt/<run_id>/iterations/`.

To run for real, add `metaopt: { enabled: true, max_iterations: 50, ... }` to your config and
point `metaopt.mutator` at a strong reasoning model. Meta-opt reuses your existing `domain`,
`acceptance`, `loop`, and agent settings.

## Batch mode

The dispatcher can submit requests through provider batch APIs for the ~50% cost discount
instead of streaming them over HTTP. Enable it with `dispatcher.mode: batch` (and
`batch_provider: mock` for an offline run).

`mode: batch` swaps the dispatcher's fulfill strategy: pending requests are grouped by provider,
submitted as a batch, then polled each loop tick until the provider marks them done. The state
machine is unchanged across the SLA wait, so kill/resume works mid-batch — resume re-polls the
batches still tagged in-flight.

Three providers implement the `BatchProvider` protocol in `src/autosynth/dispatcher/batch.py`;
see the class docstrings for the wire detail:

- **`LiteLLMBatchProvider`** (`batch_provider: openai`, default) — the OpenAI-style
  upload → create → poll → download flow over LiteLLM; covers whatever LiteLLM's batch API
  supports.
- **`AnthropicBatchProvider`** (`batch_provider: anthropic`) — Anthropic's native Message
  Batches API, which LiteLLM's unified `create_batch` doesn't model, so it calls the REST
  endpoints directly.
- **`MockBatchProvider`** (`batch_provider: mock`) — in-process, for tests and offline demos.

Any request a terminal batch returns no result for is failed, so a partial or failed batch can't
strand the run.

**Limits:** JSON-mode requests go through plain provider JSON mode (OpenAI `{"type":
"json_object"}`; on Anthropic the prompt alone, as it has no `response_format` knob), not a strict
schema. Per-request cost is best-effort — litellm's list price, so the batch discount isn't
subtracted and budget enforcement over-estimates (stopping early, not overshooting). A run assumes
a single batch provider.
