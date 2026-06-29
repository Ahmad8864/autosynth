# autosynth

[![PyPI](https://img.shields.io/pypi/v/autosynth.svg)](https://pypi.org/project/autosynth/)
[![CI](https://github.com/Ahmad8864/autosynth/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmad8864/autosynth/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/autosynth.svg)](https://pypi.org/project/autosynth/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generate synthetic datasets with an LLM loop that proposes, audits, solves, and judges its own work. Inspired by Meta FAIR's [Autodata / Agentic Self-Instruct][paper] paper ([blog post][blog]), but rewritten to be domain-agnostic: every domain-specific piece lives in a small Python plugin, and the runtime is the same regardless of whether you're generating math word problems, support-ticket triage data, or QA pairs from your own docs.

For each candidate datapoint, autosynth runs a *weak* solver and a *strong* solver, scores both against an LLM-generated rubric, and keeps the example only if the strong solver clearly beats the weak one on a quality-passing example. Failed rounds are reflected on and fed back into the next attempt.

[paper]: https://doi.org/10.48550/arXiv.2606.25996
[blog]: https://facebookresearch.github.io/RAM/blogs/autodata/

> **Status:** alpha. The API is still moving — pin a commit if you're depending on it.

## Install

```bash
uv pip install autosynth             # core
uv pip install "autosynth[hf]"       # + Hugging Face export
```

Python 3.10+. Plain `pip install autosynth` works too. For a from-source / editable install for development, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start (no API keys)

```bash
uv run autosynth run --config configs/mock_demo.yaml
uv run autosynth status outputs/mock-demo
uv run autosynth export --run outputs/mock-demo --format jsonl
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

For each source item, autosynth runs the same five-step loop until the candidate is accepted or `loop.max_rounds` is exhausted:

1. **Challenger** proposes a candidate `(input, reference_output, rubric)`.
2. **Quality** audits the candidate for obvious problems.
3. **Weak** and **strong** solvers each take N attempts at the input.
4. **Judge** scores every attempt against the rubric.
5. **Evaluator** decides accept / reject. If reject, **reflector** writes feedback for the next round.

### Acceptance modes

Three regimes decide whether a candidate is kept; pick per task with `acceptance.mode` (or omit it to use the domain's default):

- **rubric** (default) — the judge scores each rollout against the rubric; acceptance is a threshold-and-gap test. Best when quality is a matter of degree.
- **verifiable** — the domain checks answers programmatically (`verify()`), the judge is skipped, and acceptance is a count gate: *weak must fail, strong must succeed.* Use for checkable answers (math, code, exact extraction). The bundled `math_word_problems` domain ships this way.
- **judge** — a loop-judge LLM reads the per-rollout weak/strong patterns and decides accept/improve each round. Use for open-ended tasks where no fixed threshold fits.

```yaml
acceptance:
  mode: verifiable     # or: rubric | judge
```

The exact default thresholds live in `AcceptanceConfig` (`src/autosynth/config.py`); the mechanism, plus the `loop.short_circuit_strong` cost-saver, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#acceptance).

## Writing a domain

A domain plugin is one class subclassing `DomainAdapter` with six required methods. Scaffold one with:

```bash
uv run autosynth init-domain customer_support --out my_domain.py
```

Fill in `load_grounding`, `generation_prompt`, `validate_candidate`, `solver_prompt`, `quality_prompt`, and `judge_prompt`, then point your config at it. For a checkable-answer domain, also override `verify()` and set `default_acceptance_mode = "verifiable"` — the judge prompt is then unused.

```yaml
domain:
  path: ./my_domain.py:CustomerSupport
  params:
    source_csv: ./tickets.csv
```

The two bundled domains (`src/autosynth/domains/qa_from_documents.py`, `math_word_problems.py`) are short and worth reading before you write your own.

## Meta-optimization

`autosynth metaopt --config CONFIG.yaml` runs the paper's secondary loop: evolve the orchestrator's *prompts* over generations, keeping a mutation only when it beats its parent on validation. Try it without keys:

```bash
uv run autosynth metaopt --config configs/metaopt_mock.yaml
```

The algorithm, the `HarnessSpec` unit of evolution, and how to enable it for real are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#meta-optimization).

## CLI

```
autosynth run         --config CONFIG.yaml [--resume RUN_ID]   # generate a dataset
autosynth resume      RUN_DIR                                  # continue an interrupted run
autosynth status      RUN_DIR                                  # one-line progress
autosynth inspect-run RUN_DIR [--stuck]                        # detailed per-item table
autosynth export      --run RUN_DIR --format jsonl|hf          # write accepted records
autosynth metaopt     --config CONFIG.yaml                     # evolve the prompt harness
autosynth init-domain NAME --out my_domain.py                  # scaffold a domain plugin
```

Run `autosynth <command> --help` for the full flag set. `--stuck` filters to items that haven't reached a terminal state — what you want when something looks wrong.

## Run outputs

Everything for a run lives under `outputs/<run_id>/`: the `run.db` SQLite database (the source of truth, safe to share), a `config.snapshot.yaml` that resume reads back, and — only after `autosynth export` — `accepted.jsonl` / `hf_export/`. Inspect the database directly with `sqlite3 outputs/<run_id>/run.db .schema`; the table layout and accepted-record fields are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#run-database).

## Safety and limitations

- Every accepted datapoint carries an `acceptance_rationale` and a serialized `EvalReport`. There is no silent acceptance path.
- The built-in PII filter (`safety.enabled: true`) is a conservative heuristic, not a real DLP. For anything regulated, plug your own module in via `safety.filter`.
- Solvers are never *told* they're the weak or strong solver — the differential comes from the model/temperature choice. The paper flags adversarial prompting here as a gaming vector, so don't.
- There is no diversity / near-duplicate check on accepted examples yet. If you need that, extend `store.insert_accepted` with MinHash or embedding-based dedupe.
- LLM-as-judge bias is what it is. The rubric weight cap and the positive-only rule from the paper help, but don't pretend they eliminate it.

## Architecture

The runtime is an event-sourced pipeline over SQLite: a pure `step()` state machine, a dispatcher that fulfills LLM requests, and a durable store. Because `step()` is pure, you can kill a run at any point and `autosynth resume` picks up exactly where it left off. The full design — runtime, item state machine, and batch mode — is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Develop

```bash
uv run pytest          # runs offline against the in-process mock — no keys, no network
```

Setup, linting, commit conventions, and how to add a domain are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See `LICENSE`.
