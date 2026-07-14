# autosynth

[![PyPI](https://img.shields.io/pypi/v/autosynth.svg)](https://pypi.org/project/autosynth/)
[![CI](https://github.com/Ahmad8864/autosynth/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmad8864/autosynth/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/autosynth.svg)](https://pypi.org/project/autosynth/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

autosynth builds synthetic datasets with an LLM loop that proposes, checks, solves, and scores its own examples. It is inspired by Meta FAIR's [Autodata / Agentic Self-Instruct][paper] paper and accompanying [blog post][blog].

Task-specific behavior lives in a small Python plugin. The same runtime can generate math problems, support-ticket data, document-grounded QA, or other structured examples. It keeps candidates that pass quality checks and show a useful gap between a weak and a strong solver; failed attempts become feedback for the next round.

[paper]: https://doi.org/10.48550/arXiv.2606.25996
[blog]: https://facebookresearch.github.io/RAM/blogs/autodata/

> **Status:** alpha. APIs may change; pin a version or commit for production use.

## Install

```bash
uv pip install autosynth             # core
uv pip install "autosynth[hf]"       # + Hugging Face export
```

Python 3.10+. Plain `pip install autosynth` works too. For an editable source install, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start (no API keys)

```bash
uv run autosynth run --config configs/mock_demo.yaml
uv run autosynth status outputs/mock-demo
uv run autosynth export --run outputs/mock-demo --format jsonl
```

The mock demo uses a local scripted provider and normally finishes in about a second. It writes `outputs/mock-demo/run.db` and a config snapshot. Export is optional; SQLite remains the source of truth.

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

Roles can use different providers. Choose the solver models and temperatures so the strong solver has a meaningful advantage over the weak one.

String fields support `${VAR}` and `${VAR:default}` interpolation. For example, `api_base: ${OLLAMA_HOST:http://localhost:11434}` uses `OLLAMA_HOST` when set.

See `configs/example_qa.yaml` and `configs/example_math.yaml` for full real-provider configs.

[litellm]: https://docs.litellm.ai/

## How it works

For each source item, autosynth repeats this loop until a candidate is accepted or `loop.max_rounds` is reached:

1. **Challenger** proposes a candidate `(input, reference_output, rubric)`.
2. **Quality** audits the candidate for obvious problems.
3. **Weak** and **strong** solvers each take N attempts at the input.
4. **Judge** scores every attempt against the rubric.
5. **Evaluator** decides accept / reject. If reject, **reflector** writes feedback for the next round.

### Acceptance modes

Set `acceptance.mode` per task, or omit it to use the domain default:

- **rubric** (default) — the judge scores each attempt against the rubric, then fixed thresholds check the weak score, strong score, and gap.
- **verifiable** — the domain checks answers with `verify()`. This skips the judge and accepts by correct-answer counts. It suits math, code, exact extraction, and other checkable tasks.
- **judge** — after rubric scoring, another LLM reads the weak/strong results and decides whether to accept or improve the round.

```yaml
acceptance:
  mode: verifiable     # or: rubric | judge
```

Default thresholds live in [`AcceptanceConfig`](src/autosynth/config.py). See [Architecture: Acceptance](docs/ARCHITECTURE.md#acceptance) for the decision flow and `loop.short_circuit_strong`.

### Final audit

Set `audit.enabled: true` to run one more check before acceptance. The auditor sees the source and scored attempts, and checks leakage, source support, and rubric quality. Failures become feedback for the next round.

The audit uses the judge model unless `auditor` is configured separately. See [Architecture: Acceptance](docs/ARCHITECTURE.md#acceptance) for details.

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

The bundled [`qa_from_documents`](src/autosynth/domains/qa_from_documents.py) and [`math_word_problems`](src/autosynth/domains/math_word_problems.py) domains provide small working examples.

## Meta-optimization

`autosynth metaopt --config CONFIG.yaml` evolves the agent instructions, keeping a mutation only when it outperforms its parent on validation. The mock configuration runs without API keys:

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

Run `autosynth <command> --help` for all options. `--stuck` shows only non-terminal items.

## Run outputs

Each run writes to `outputs/<run_id>/`:

- `run.db` — the SQLite source of truth.
- `config.snapshot.yaml` — the configuration used for the run and its default resume configuration.
- `accepted.jsonl` or `hf_export/` — created only by `autosynth export`.

Inspect the schema with `sqlite3 outputs/<run_id>/run.db .schema`. The tables and accepted-record fields are described in [Architecture: Run database](docs/ARCHITECTURE.md#run-database).

## Safety and limitations

- Every accepted datapoint records its `acceptance_rationale` and per-attempt evaluation scores. There is no silent acceptance path.
- The built-in PII filter (`safety.enabled: true`) is a conservative heuristic, not a DLP system. Regulated workloads should provide a filter through `safety.filter`.
- Weak and strong solvers differ by model and sampling settings, not by instructions to behave weakly or strongly.
- Accepted examples are not checked for diversity or near-duplicates. Add a deduplication step if your dataset requires one.
- LLM judges remain biased and fallible. Rubric constraints reduce some failure modes but do not remove the need for evaluation.

## Architecture

The runtime is an event-sourced pipeline over SQLite: a pure `step()` state machine, a request dispatcher, and a durable store. Persisted requests, responses, and deterministic IDs let interrupted runs resume without starting over. See [Architecture & internals](docs/ARCHITECTURE.md) for the full design.

## Develop

```bash
uv run pytest
```

The test suite runs offline against in-process providers. Setup, linting, commit conventions, and domain development are covered in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See `LICENSE`.
