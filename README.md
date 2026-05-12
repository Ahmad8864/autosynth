# autodata

A reusable, domain-agnostic implementation of **Autodata / Agentic
Self-Instruct** (Meta FAIR, [research blog][paper]). An LLM-driven loop
acts as a data scientist: it generates a candidate datapoint, audits it,
runs a weak and a strong solver, judges both against an auto-generated
rubric, and accepts only when the strong solver convincingly beats the
weak one on a quality-passing example. Failed rounds feed targeted
feedback back into the next attempt.

This repo is **not** a re-implementation of the paper's CS-paper-specific
setup. The reusable architecture has been extracted and every
domain-specific component is a pluggable `DomainAdapter`.

[paper]: https://facebookresearch.github.io/RAM/blogs/autodata/

---

## How this maps to Autodata / Agentic Self-Instruct

| Paper concept                            | This repo                                       |
| ---                                      | ---                                             |
| Generate→verify→evaluate→reflect→refine  | `pipeline.step()` (pure state machine)          |
| Challenger / quality / weak / strong / judge / reflector | `agents/*.py` builders + parsers |
| Acceptance criteria (weak/strong/gap)    | `config.AcceptanceConfig` + `evaluator.evaluate`|
| Durable per-source trajectories          | `store.Store` (SQLite, WAL mode)                |
| Positive-only rubric, weights ≤ 7        | enforced in `challenger.parse_response` / harness |
| "ALL rounds attempted" persisted         | rounds row materialized on first parse, updated in-place |
| Meta-opt population + Boltzmann (T=0.1)  | `metaopt.boltzmann_select`                      |
| Failure trajectory analysis              | `metaopt.aggregate_failures_from_db` (SQL)      |
| Code-editing agent ⇒ harness mutation    | `metaopt.Mutator` + `metaopt.apply_mutation`    |
| Train+val gate, accept only if val ↑     | `metaopt.MetaOptimizer.run` decision block      |

The defaults in `AcceptanceConfig` mirror the paper §3: weak avg ≤ 0.65,
weak max ≤ 0.75, strong avg ∈ [0.60, 0.95), gap ≥ 0.20. All are
overridable per run.

The paper's meta-optimization loop (evolving the orchestrator's prompts
themselves) is implemented in `autodata/metaopt.py` — see
_Meta-optimization_ below.

---

## Architecture

The runtime is an **event-sourced pipeline** over a SQLite store. Pure
`step()` advances item state; the dispatcher fulfills LLM requests and
commits responses; the store is the durable record of the run.

```
┌──────────────────────────────────────────────────────────────┐
│  Pipeline  — pure (state, responses) → (state, requests)     │
│  No I/O. No threads. No network.                             │
├──────────────────────────────────────────────────────────────┤
│  Dispatcher — reads ready items, calls step(), fulfills      │
│  requests, writes results. Two strategies:                   │
│    • fulfill_local   (threadpool over HTTP)                  │
│    • fulfill_batch   (provider batch APIs)                   │
├──────────────────────────────────────────────────────────────┤
│  Store — SQLite (WAL). One run.db per run. Resumable.        │
│  Tables: runs, items, rounds, requests, responses,           │
│          solver_scores, accepted.                            │
├──────────────────────────────────────────────────────────────┤
│  LLMClient — provider routing, RPM rate limit, retry, cost.  │
└──────────────────────────────────────────────────────────────┘
```

**States (5 + 2 terminal):**
`PENDING → NEED_CANDIDATE → NEED_QUALITY → NEED_SCORES`
plus `NEED_REFLECTION` on the rejection branch and `ACCEPTED` / `REJECTED`
as terminals. `NEED_SCORES` fans out `N × weak + N × strong` solver
requests concurrently; each judge fires as soon as its solver lands. The
dispatcher is bounded by `cfg.dispatcher.concurrency`.

**Legacy flow (deprecated, kept here for completeness):**

```
GroundingItem  ─▶  Challenger ─▶  Candidate
                      ▲              │
                      │              ▼
                Reflector          Quality
                      ▲              │ (pass)
                      │              ▼
                      │       WeakSolver × N   StrongSolver × N
                      │              │              │
                      │              ▼              ▼
                      │           Judge per attempt
                      │                   │
                      │                   ▼
                      └───── evaluator.evaluate ──▶ accept? ──▶ DatasetWriter
                                          │ no
                                          ▼
                                  (next round, with feedback)
```

Per source item the loop runs until accepted or `loop.max_rounds`
is exhausted. Every round — accepted or rejected — is written to
`trajectories/<source_id>.json`.

---

## Install

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv venv                          # create .venv
uv pip install -e .              # install the package
uv pip install -e ".[dev]"       # + tests / linters
uv pip install -e ".[hf]"        # + optional Hugging Face export
```

Activate the venv with `source .venv/bin/activate`, or prefix commands
with `uv run` (e.g. `uv run pytest`, `uv run autodata run …`).

Python ≥ 3.10.

---

## Mock demo (no API keys)

```bash
uv run autodata run --config configs/mock_demo.yaml
uv run autodata status outputs/mock-demo
uv run autodata inspect-run outputs/mock-demo            # all items
uv run autodata inspect-run outputs/mock-demo --stuck    # only non-terminal items
uv run autodata export --run outputs/mock-demo --format jsonl
```

The mock demo uses the in-process mock provider. The full loop runs in
~1 second and writes a single `run.db` (plus `config.snapshot.yaml`) under
`outputs/mock-demo/`. `autodata export` produces the legacy JSONL on
demand; `inspect-run` queries the database for a per-item state table.

**Resume any run:**

```bash
uv run autodata resume outputs/mock-demo
# or:
uv run autodata run --config configs/mock_demo.yaml --resume mock-demo
```

Kill the process mid-run (Ctrl-C); the state machine + SQLite ensure the
restart picks up exactly where it left off. In-flight local requests are
reverted to pending; in-flight batch requests are kept tagged and polled.

---

## API keys & real providers

`autodata` dispatches real LLM calls through [LiteLLM][litellm], so any
provider it supports works. Set the relevant env var(s):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export TOGETHER_API_KEY=...
export FIREWORKS_API_KEY=...
export OPENROUTER_API_KEY=...
# Ollama / local vLLM: no key needed
```

Then reference a model in your YAML config:

```yaml
challenger:    { provider_model: anthropic/claude-haiku-4-5, temperature: 0.8 }
weak_solver:   { provider_model: openai/gpt-4o-mini }
strong_solver: { provider_model: openai/gpt-4o }
judge:         { provider_model: anthropic/claude-haiku-4-5, temperature: 0.0 }
```

You can mix providers freely across roles. Use cheap models for the
weak solver and a frontier model for the strong solver to maximize the
expected weak-vs-strong gap.

[litellm]: https://docs.litellm.ai/

`${VAR}` and `${VAR:default}` interpolation works inside any string config
field — e.g. `api_base: ${OLLAMA_HOST:http://localhost:11434}`.

---

## Creating a new domain

A domain plugin is one Python class subclassing `DomainAdapter`. Six
abstract methods, no core changes required.

```bash
uv run autodata init-domain customer_support_tickets -o my_domain.py
```

Then fill in the methods (`load_grounding`, `generation_prompt`,
`validate_candidate`, `solver_prompt`, `quality_prompt`, `judge_prompt`)
and reference it in your config:

```yaml
domain:
  path: ./my_domain.py:CustomerSupportTickets
  params:
    source_csv: ./tickets.csv
```

Bundled examples: `domains/qa_from_documents.py`,
`domains/math_word_problems.py`, `examples/new_domain_template.py`.

---

## CLI

```
uv run autodata run --config CONFIG.yaml [--run-id ID] [--resume RUN_ID] [--verbose]
uv run autodata resume outputs/RUN_DIR
uv run autodata status outputs/RUN_DIR
uv run autodata inspect-run outputs/RUN_DIR [--stuck]
uv run autodata export --run outputs/RUN_DIR --format jsonl|hf [--out PATH]
uv run autodata metaopt --config CONFIG.yaml
uv run autodata init-domain NAME --out my_domain.py
```

---

## Outputs

Each run writes to `outputs/<run_id>/`:

| File                                  | Contents                                  |
| ---                                   | ---                                       |
| `run.db`                              | Authoritative SQLite database for the run. |
| `config.snapshot.yaml`                | Frozen config used for this run.          |
| `accepted.jsonl` *(on export)*        | Accepted dataset — produced by `autodata export`. |
| `hf_export/` *(on export)*            | Optional Hugging Face datasets directory. |

The `run.db` is the durable record. Tables: `runs`, `items`, `rounds`,
`requests`, `responses`, `solver_scores`, `accepted`. The DB is queryable
directly with `sqlite3` (e.g. "which items are stuck in NEED_SCORES?")
and self-contained for sharing.

Each accepted record carries: `input`, `reference_output`, `rubric`,
`domain`, `source_id`, `metadata`, `weak_avg`, `strong_avg`, `gap`,
per-attempt solver scores, and `acceptance_rationale`.

**Resume is durable at response granularity.** Killing a run between
LLM calls — even mid-batch — never loses progress. See
`store.normalize_for_resume` for the exact reconciliation table.

---

## Safety & quality guardrails

- **No silent acceptance.** Every accepted datapoint has an
  `acceptance_rationale` and a recorded `EvalReport` (the paper's
  trajectory contract).
- **No hardcoded domain assumptions.** Every domain-specific piece is in
  the `DomainAdapter`.
- **Deterministic IDs.** `stable_id(...)` produces reproducible
  `candidate_id`/`source_id`/`trajectory_id` — runs are resumable.
- **Budget controls.** `request_budget_usd` is a soft advisory hook;
  hard rate-limit/retry/timeout behavior is centralized in `LLMClient`.
- **Optional safety filter.** `safety.enabled: true` runs a conservative
  PII heuristic by default; plug in `module:attr` for your own DLP.
- **Anti-gaming.** Solvers are not adversarially told to be "weak" or
  "strong" — the role differential comes from model / temperature
  choice. The paper specifically warns this is a gaming vector.

---

## Meta-optimization

`autodata metaopt --config CONFIG.yaml` runs the paper's secondary loop:
evolve the orchestrator's instructions themselves. The unit of evolution
is a `HarnessSpec` — a structured set of rule strings injected into each
agent's system prompt (`challenger_rules`, `quality_rules`, …) plus a
few numeric knobs (`rubric_max_weight`, `require_self_test`).

The loop:

1. Evaluate the seed harness on both training and validation source items.
2. For each iteration:
   - Pick a parent from the population via **Boltzmann sampling** at
     `temperature = 0.1` over training scores.
   - Aggregate the rejection-reason histogram from the parent's last
     training run.
   - Send the parent + failure summary to the **mutator** LLM, which
     returns a structured `{rules_add, rules_remove_indices,
     rubric_max_weight, require_self_test}` diff.
   - Apply the mutation → child `HarnessSpec`. Duplicates (same
     fingerprint) are detected and skipped.
   - Evaluate child on training items. If it doesn't improve, reject.
   - Otherwise evaluate on validation items. Accept the mutation only
     if `child.val > parent.val` — same gate as the paper.

Mutations operate on `HarnessSpec`, not on Python source. This preserves
the expressive scope for prompt-text edits (the paper's main lever) while
avoiding the safety / sandboxing surface of a real code-editing agent.
This is called out explicitly so users can swap in their own mutator if
they want richer edits.

**Demo** (no API keys):

```bash
autodata metaopt --config configs/metaopt_mock.yaml
```

The mock-friendly scenario seeds at 0% accept rate; the mutator proposes
a source-specificity rule on iteration 1; the rule lifts train and val
to 100%; the mutation is accepted; subsequent iterations re-propose the
same rule and are rejected (no-op detection). Population, lineage, and
per-iteration decisions are written under
`outputs/metaopt/<run_id>/iterations/iter_NNN/`.

**Real run.** Add `metaopt: { enabled: true, max_iterations: 50, ... }`
to your existing YAML config and point `metaopt.mutator` at a strong
reasoning model. The loop reuses your `domain`, `acceptance`, `loop`,
and agent model settings — meta-opt is layered on top of the normal
run, not a separate codepath.

---

## Limitations

- LLM-as-judge inherits the usual biases of LLM-as-judge. The rubric
  cap at weight ≤ 7 and the positive-only rule (paper §meta-opt) reduce
  but do not eliminate this.
- The PII filter is a starting heuristic. Production use needs a real
  DLP integration via the `safety.filter` hook.
- Diversity / near-duplicate checks across accepted examples are not
  yet included — extend `store.insert_accepted` to add MinHash /
  embedding-based dedupe.
- Real provider batch implementations (OpenAI `/v1/batches`, Anthropic
  message batches) are not bundled. The `dispatcher_batch.BatchProvider`
  protocol is in place; the `MockBatchProvider` exercises submit / poll /
  fetch. Plug in a real provider when you need the 50% cost discount.

---

## Tests

```bash
uv run pytest
```

131 tests, all running against the in-process mock provider. No API keys
required. Coverage:

- schema validation and round-trip
- LLMClient: token-bucket math, glob-pattern rate limits, retry,
  cost accounting, mock dispatch
- Store: SQLite schema, `claim_pending` atomicity under threads, round
  materialization lifecycle, resume normalization (§4.2 table),
  JSONL export
- Pipeline: exhaustive state-transition tests on the pure `step()`
  function — including the §2.3 partial-completion noop invariant
- Agent builders/parsers
- Dispatcher: end-to-end accept, concurrent fulfill (100 requests, no
  duplicate responses), budget abort, resume after kill, unrecoverable
  failure cap
- Batch dispatcher: submit / poll / fetch / batch_id tagging
- Acceptance criteria branches
- Meta-optimization: Boltzmann selection, mutation application, full
  loop with accepted improving mutation
