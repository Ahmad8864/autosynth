# autodata

Generate synthetic datasets with an LLM loop that proposes, audits, solves, and judges its own work. Inspired by Meta FAIR's [Autodata / Agentic Self-Instruct][paper] paper, but rewritten to be domain-agnostic: every domain-specific piece lives in a small Python plugin, and the runtime is the same regardless of whether you're generating math word problems, support-ticket triage data, or QA pairs from your own docs.

The headline trick: for each candidate datapoint, run a *weak* solver and a *strong* solver, score both against an LLM-generated rubric, and only keep the example if the strong solver clearly beats the weak one on a quality-passing example. Failed rounds are reflected on and fed back into the next attempt.

[paper]: https://facebookresearch.github.io/RAM/blogs/autodata/

> **Status:** alpha (0.1.0). The API is still moving. Pin a commit if you're depending on it.

## Install

```bash
uv venv
uv pip install -e .             # core
uv pip install -e ".[dev]"      # + pytest, ruff
uv pip install -e ".[hf]"       # + Hugging Face export
```

Python 3.10+. Either activate the venv (`source .venv/bin/activate`) or prefix commands with `uv run`.

## Quick start (no API keys)

```bash
uv run autodata run --config configs/mock_demo.yaml
uv run autodata status outputs/mock-demo
uv run autodata export --run outputs/mock-demo --format jsonl
```

The mock demo uses an in-process scripted "provider" and finishes in about a second. It writes `outputs/mock-demo/run.db` plus a frozen config snapshot. The `export` step is opt-in — the SQLite database is the source of truth.

## Real providers

LLM calls go through [LiteLLM][litellm], so any provider it supports should work. Set the relevant key and reference the model in YAML:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

```yaml
challenger:    { provider_model: anthropic/claude-haiku-4-5, temperature: 0.8 }
weak_solver:   { provider_model: openai/gpt-4o-mini }
strong_solver: { provider_model: openai/gpt-4o }
judge:         { provider_model: anthropic/claude-haiku-4-5, temperature: 0.0 }
```

You can mix providers across roles. The cheaper-vs-frontier split between the two solvers is the whole point — that's what produces the weak/strong gap that drives acceptance.

`${VAR}` and `${VAR:default}` substitution works in any string field, so `api_base: ${OLLAMA_HOST:http://localhost:11434}` does what you'd expect.

See `configs/example_qa.yaml` and `configs/example_math.yaml` for full real-provider configs.

[litellm]: https://docs.litellm.ai/

## How it works

For each source item, autodata runs the same five-step loop until the candidate is accepted or `loop.max_rounds` is exhausted:

1. **Challenger** proposes a candidate `(input, reference_output, rubric)`.
2. **Quality** audits the candidate for obvious problems.
3. **Weak** and **strong** solvers each take N attempts at the input.
4. **Judge** scores every attempt against the rubric.
5. **Evaluator** decides accept / reject. If reject, **reflector** writes feedback for the next round.

The acceptance defaults come from §3 of the paper:

- weak average ≤ 0.65, weak max ≤ 0.75
- strong average in [0.60, 0.95)
- strong − weak gap ≥ 0.20
- quality must have passed

All of these are overridable in `acceptance:` in your config.

## Architecture

The runtime is an event-sourced pipeline over a SQLite database. A pure `step()` function advances item state; the dispatcher fulfills LLM requests and writes responses back; the store is the durable record.

```
pipeline.step()        pure state machine: (state, responses) -> (state, requests)
dispatcher             reads ready items, calls step(), fulfills requests
  ├─ fulfill_local     threadpool over HTTP
  └─ fulfill_batch     provider batch APIs (see "Batch" below)
store                  SQLite + WAL, one run.db per run
llm                    provider routing, rate-limit, retry, cost accounting
```

Item states: `PENDING → NEED_CANDIDATE → NEED_QUALITY → NEED_SCORES` with `NEED_REFLECTION` on the reject branch and `ACCEPTED` / `REJECTED` as terminals. `NEED_SCORES` fans out `N × weak + N × strong` solver requests in parallel; each judge fires the moment its solver lands. Concurrency is bounded by `cfg.dispatcher.concurrency`.

The fact that `step()` is pure is the only reason resume works. Kill the process at any point — including mid-batch — and `autodata resume` picks up exactly where it left off. In-flight local requests revert to pending; in-flight batch requests stay tagged and get polled.

## CLI

```
autodata run --config CONFIG.yaml [--run-id ID] [--resume RUN_ID] [-v]
autodata resume RUN_DIR
autodata status RUN_DIR
autodata inspect-run RUN_DIR [--stuck]
autodata export --run RUN_DIR --format jsonl|hf [--out PATH]
autodata metaopt --config CONFIG.yaml
autodata init-domain NAME --out my_domain.py
```

`status` is the one-liner; `inspect-run` is the detailed per-item table. `--stuck` filters to items that haven't reached a terminal state, which is what you want when something looks wrong.

## Run outputs

Everything for a run lives under `outputs/<run_id>/`:

- `run.db` — SQLite. Tables: `runs`, `items`, `rounds`, `requests`, `responses`, `solver_scores`, `accepted`. Queryable with the `sqlite3` CLI and safe to share.
- `config.snapshot.yaml` — the exact config used. Resume reads this if you don't pass `--config`.
- `accepted.jsonl` / `hf_export/` — produced on `autodata export`, not written automatically.

Each accepted record contains `input`, `reference_output`, `rubric`, `domain`, `source_id`, `metadata`, the weak/strong/gap scores, per-attempt solver scores, and the acceptance rationale.

## Writing a domain

A domain plugin is one class subclassing `DomainAdapter` with six methods. Scaffold one with:

```bash
uv run autodata init-domain customer_support -o my_domain.py
```

Fill in `load_grounding`, `generation_prompt`, `validate_candidate`, `solver_prompt`, `quality_prompt`, and `judge_prompt`, then point your config at it:

```yaml
domain:
  path: ./my_domain.py:CustomerSupport
  params:
    source_csv: ./tickets.csv
```

The two bundled domains (`src/autodata/domains/qa_from_documents.py`, `src/autodata/domains/math_word_problems.py`) are short and worth reading before you write your own.

## Meta-optimization

`autodata metaopt --config CONFIG.yaml` runs the paper's secondary loop: evolve the orchestrator's *prompts* over generations. The unit of evolution is a `HarnessSpec` — a structured bag of rule strings that get injected into each agent's system prompt, plus a couple of numeric knobs.

The loop, roughly:

1. Score the seed harness on training and validation source items.
2. Each iteration: Boltzmann-sample a parent from the population (T=0.1 over training scores), summarize that parent's most recent rejection reasons, ask the mutator LLM for a structured diff, apply it, dedupe, and re-evaluate.
3. Accept the mutation only if `child.val > parent.val` — the paper's gate.

Mutations operate on the harness, not on Python source. That preserves the main lever the paper exercises (prompt-text edits) without the sandboxing headache of a code-editing agent. Swap in your own mutator if you want richer edits.

Try it without keys:

```bash
uv run autodata metaopt --config configs/metaopt_mock.yaml
```

The mock scenario seeds at 0% accept, the mutator proposes a source-specificity rule on iteration 1 that lifts both train and val to 100%, that mutation is accepted, and subsequent iterations get deduplicated. Population, lineage, and per-iteration decisions are written under `outputs/metaopt/<run_id>/iterations/`.

To run for real, add `metaopt: { enabled: true, max_iterations: 50, ... }` to your existing config and point `metaopt.mutator` at a strong reasoning model. Meta-opt reuses your existing `domain`, `acceptance`, `loop`, and agent settings.

## Batch mode

The dispatcher can submit requests through provider batch APIs (OpenAI `/v1/batches`, Anthropic message batches) for the 50% cost discount. The `BatchProvider` protocol and a `MockBatchProvider` are in the box. Real provider implementations are not — wiring those up is the next piece of work. If you only have a few thousand requests, `fulfill_local` is fine.

## Safety and quality notes

- Every accepted datapoint carries an `acceptance_rationale` and a serialized `EvalReport`. There is no silent acceptance path.
- The built-in PII filter (`safety.enabled: true`) is a conservative heuristic, not a real DLP. For anything regulated, plug your own module in via `safety.filter`.
- Solvers are never *told* they're the weak or strong solver — the differential comes from the model/temperature choice. The paper flags adversarial prompting here as a gaming vector, so don't.
- There is no diversity / near-duplicate check on accepted examples yet. If you need that, extend `store.insert_accepted` with MinHash or embedding-based dedupe.
- LLM-as-judge bias is what it is. The rubric weight cap (≤ 7) and the positive-only rule from the paper help, but don't pretend they eliminate it.

## Tests

```bash
uv run pytest
```

The full suite (~130 tests) runs against the in-process mock provider — no keys, no network. The interesting bits to look at if you're touching the core:

- `test_pure_pipeline.py` — exhaustive state-transition coverage of `step()`, including the partial-completion no-op invariant.
- `test_store.py` — `claim_pending` atomicity under threads, resume normalization.
- `test_dispatcher.py` — end-to-end accept, 100-request concurrent fulfill, budget abort, kill/resume.

## License

MIT. See `LICENSE`.
