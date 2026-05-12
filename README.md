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
| Main / orchestrator agent                | `orchestrator.Orchestrator`                     |
| Challenger LLM                           | `agents.challenger.ChallengerAgent`             |
| Weak solver / strong solver              | `agents.solver.SolverAgent("weak"/"strong")`    |
| Quality verifier                         | `agents.verifier.VerifierJudge.quality_check`   |
| Judge with weighted rubric               | `agents.verifier.VerifierJudge.score`           |
| Reflection / targeted feedback           | `agents.reflector.Reflector`                    |
| Acceptance criteria (weak/strong/gap)    | `config.AcceptanceConfig` + `evaluator.evaluate`|
| Per-source trajectories with all rounds  | `writer.RunWriter` + `schemas.Trajectory`       |
| Positive-only rubric, weights ≤ 7        | enforced in `ChallengerAgent._parse`            |
| "ALL rounds attempted" persisted         | trajectory rewritten after every round          |

The defaults in `AcceptanceConfig` mirror the paper §3: weak avg ≤ 0.65,
weak max ≤ 0.75, strong avg ∈ [0.60, 0.95), gap ≥ 0.20. All are
overridable per run.

The paper's meta-optimization loop (evolving the orchestrator's prompts
themselves) is **not** implemented here — it is acknowledged as an
extension point; see _Limitations_.

---

## Architecture

```
GroundingItem  ─▶  ChallengerAgent ─▶  Candidate
                      ▲                   │
                      │                   ▼
                Reflector            VerifierJudge.quality_check
                      ▲                   │ (pass)
                      │                   ▼
                      │            WeakSolver × N   StrongSolver × N
                      │                   │              │
                      │                   ▼              ▼
                      │            VerifierJudge.score per attempt
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
uv run autodata inspect-run outputs/mock-demo
```

The mock demo uses the in-process `MockLLMProvider`. The full loop runs
in ~1 second and writes a real `accepted.jsonl`, `summary.json`, and
trajectory files so you can inspect the pipeline end-to-end.

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
uv run autodata run --config CONFIG.yaml [--run-id ID] [--verbose]
uv run autodata init-domain NAME --out my_domain.py
uv run autodata inspect-run outputs/RUN_ID
uv run autodata export --run outputs/RUN_ID --format jsonl|hf
```

---

## Outputs

Each run writes to `outputs/<run_id>/`:

| File                                  | Contents                                  |
| ---                                   | ---                                       |
| `config.snapshot.yaml`                | The frozen config used for this run.      |
| `accepted.jsonl`                      | Accepted datapoints — final dataset.      |
| `rejected.jsonl`                      | Source items exhausted without accept.    |
| `trajectories/<source_id>.json`       | All rounds (accepted **and** rejected) per source. |
| `summary.json`                        | Live counters; updated each acceptance/rejection. |
| `hf_export/`                          | Optional Hugging Face `datasets` directory. |

Each accepted record carries: `input`, `reference_output`, `rubric`,
`domain`, `source_id`, `metadata`, `weak_avg`, `strong_avg`, `gap`,
per-attempt solver scores, `acceptance_rationale`, and `trajectory_id`.

Trajectories are written incrementally after **every** round, so an
interrupted run can be resumed (`resume: true`, default).

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

## Limitations

- The meta-optimization loop (evolving the orchestrator's instructions
  themselves via a code-editing agent) is not implemented. It is the
  paper's secondary loop and would significantly expand scope.
- LLM-as-judge inherits the usual biases of LLM-as-judge. The rubric
  cap at weight ≤ 7 and the positive-only rule (paper §meta-opt) reduce
  but do not eliminate this.
- The PII filter is a starting heuristic. Production use needs a real
  DLP integration via the `safety.filter` hook.
- Diversity / near-duplicate checks across accepted examples are not
  yet included — extend `RunWriter.write_accepted` to add MinHash /
  embedding-based dedupe.
- Concurrency is currently sequential per source item. The orchestrator
  reads `max_concurrency` but does not yet parallelize; the
  per-source-item structure makes this an easy follow-on with
  `ThreadPoolExecutor`.

---

## Tests

```bash
uv run pytest
```

Tests run entirely against the `MockLLMProvider`, so no API keys are
needed. Covered:

- schema validation and round-trip
- mock LLM provider
- every acceptance-criterion branch
- registered + path-based domain loading
- a full mocked accept loop
- a full mocked reject-and-exhaust loop
- trajectory + config-snapshot writing
- resume behavior
