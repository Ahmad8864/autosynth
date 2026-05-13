# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org).

## [Unreleased]

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

[Unreleased]: https://github.com/Ahmad8864/autosynth/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ahmad8864/autosynth/releases/tag/v0.1.0
