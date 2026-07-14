# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Independent final audit (`audit.enabled: true`): each round that passes acceptance takes one further LLM pass — the paper's post-loop quality verifier — re-checking leakage, source support, and rubric quality with the grounding source and scored rollouts in view. Failures feed back into the next round; `auditor:` overrides the model (defaults to judge).
- SFT, DPO, and GRPO exports with JSONL or Hugging Face output, provenance metadata, and generated dataset cards.

### Changed

### Fixed

## [0.3.0] - 2026-07-01

### Added

- Startup banner: each `run` now prints a summary of the experiment (domain, per-role models, and key loop/acceptance settings) before generation begins, so you can confirm at a glance what a run is configured to do.
- Strict structured outputs: the challenger and other fixed-shape roles now request schema-constrained responses from providers that support strict JSON mode, cutting malformed structured output.

### Changed

- Retuned defaults: `max_tokens` 2048 → 8192, `loop.max_rounds` 5 → 20, `loop.weak_samples` and `loop.strong_samples` 3 → 4, and `loop.short_circuit_strong` now defaults to `true` (the strong solver is scored only when the weak gate passes, to save cost). Set these explicitly in your config to keep the previous values.

### Fixed

## [0.2.1] - 2026-06-29

### Added

- Batch mode: set `dispatcher.mode: batch` to fulfill requests through provider batch APIs for the ~50% cost discount, choosing the backend with `batch_provider` (`openai` — any OpenAI-style provider via LiteLLM, `anthropic` — native Message Batches, or `mock` — in-process, no keys). Kill/resume works mid-batch.
- Tested and advertised support for Python 3.13 and 3.14.

### Changed

### Fixed

## [0.2.0] - 2026-06-29

### Added

- New `verifiable` acceptance mode for tasks with checkable answers (math, code, exact extraction): correctness is checked programmatically instead of by the LLM judge. Enable with `acceptance.mode: verifiable`.
- The bundled `math_word_problems` domain now uses verifiable mode by default, checking answers for exact numeric equality.
- New `judge` acceptance mode: an LLM decides accept/improve each round instead of fixed thresholds, for open-ended tasks where no threshold fits. Enable with `acceptance.mode: judge`.
- Optional conditional strong-solver evaluation (`loop.short_circuit_strong: true`): skip the strong solver on too-easy examples to save cost. Off by default.

### Changed

- Runs created by older versions keep resuming: their databases are upgraded automatically on open.
- Meta-optimization gates on validation score alone and averages repeated evaluations of a candidate, so acceptance isn't decided by a single noisy run.
- Unknown keys in a config you pass are now rejected (instead of silently ignored), so typos like `weak_avg_maxx` fail loudly; resuming from a run's snapshot still tolerates keys removed since it was written. Removed config keys that never had any effect: `seed`, `max_concurrency`, `request_budget_usd`, `hf_export`, and `loop.stop_on_first_accept` — delete these from older configs you pass with `--config`.
- The safety/PII filter now scans nested payload values and the reference answer, not only top-level payload fields.

### Fixed

- Fixed a rare stuck-item bug: an item could stall mid-run if two of its responses were committed in the same microsecond. The advance watermark is now a strictly-monotonic sequence rather than a wall-clock timestamp.
- The `Runner` now closes its database connection when a run finishes, avoiding a connection / WAL-file leak when many runs are created (e.g. during meta-optimization).
- `configs/mock_demo.yaml` had a YAML typo (`strong_solver:{` missing a space) that silently fell back to defaults; fixed, and now caught by strict config validation.

## [0.1.1] - 2026-05-12

Release-infrastructure and tooling pass. No user-facing functional changes.

### Added

- README badges: PyPI version, CI status, Python versions, license.
- Trusted-publisher release workflow on `v*` tags (PyPI publish via OIDC, no long-lived tokens).
- Pre-commit hooks (ruff, gitleaks, file hygiene) enforced locally and in CI.

### Changed

- License metadata switched to PEP 639 SPDX form (`license = "MIT"` + `license-files`).
- Version is now derived from git tags via `setuptools-scm` (no more hand-bumping `__version__`).
- Author field set in PyPI metadata.

## [0.1.0] - 2026-05-12

Initial release.

### Added

- Agentic generation loop: challenger → quality audit → weak/strong solvers → judge → evaluator → reflector.
- SQLite-backed event-sourced runtime with kill/resume.
- Domain plugin system (`DomainAdapter`) with bundled QA-from-documents and math-word-problems examples.
- Meta-optimization loop for evolving prompt harnesses across generations.
- In-process mock LLM provider for zero-API-key testing.
- LiteLLM-based real provider routing with per-role configuration.
- CLI: `run`, `resume`, `status`, `inspect-run`, `export`, `metaopt`, `init-domain`.

[Unreleased]: https://github.com/Ahmad8864/autosynth/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Ahmad8864/autosynth/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Ahmad8864/autosynth/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ahmad8864/autosynth/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Ahmad8864/autosynth/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Ahmad8864/autosynth/releases/tag/v0.1.0
