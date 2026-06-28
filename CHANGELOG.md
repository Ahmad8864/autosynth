# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- New `verifiable` acceptance mode for tasks with checkable answers (math, code, exact extraction): correctness is checked programmatically instead of by the LLM judge. Enable with `acceptance.mode: verifiable`.
- The bundled `math_word_problems` domain now uses verifiable mode by default, checking answers for exact numeric equality.

### Changed

- Runs created by older versions keep resuming: their databases are upgraded automatically on open.

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

[Unreleased]: https://github.com/Ahmad8864/autosynth/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Ahmad8864/autosynth/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Ahmad8864/autosynth/releases/tag/v0.1.0
