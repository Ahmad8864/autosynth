# Contributing to autodata

Thanks for your interest in helping build autodata. This document covers how
to set up a dev environment, the standards we expect from changes, and the
two highest-leverage ways to contribute: adding a domain plugin or sharpening
the orchestrator's behavior.

## Dev setup

```bash
uv venv
uv pip install -e ".[dev]"
```

Activate with `source .venv/bin/activate`, or prefix commands with `uv run`.
Python ≥ 3.10 is required.

## Running the test suite

```bash
uv run pytest
```

All tests run against the in-process `MockLLMProvider`, so you do **not** need
any API keys to verify your change. If your change adds real-provider
behavior, gate it behind a test that uses `register_mock(...)` to script the
expected provider response.

## Lint / format

```bash
uv run ruff check .
uv run ruff format .
```

The CI workflow runs both on every PR. Configuration lives in `pyproject.toml`
under `[tool.ruff]`.

## What we look for in a PR

- **Tests for behavior changes.** A bug fix should add the test that would
  have caught it; a feature should ship with at least one happy-path test.
- **No new third-party dependencies in the core path.** The mock demo must
  keep working with zero network access. Heavy deps belong under an optional
  extra (see `pyproject.toml` `[project.optional-dependencies]`).
- **Domain-agnostic core.** Anything domain-specific belongs in a
  `DomainAdapter`, never in `orchestrator.py` / `evaluator.py` / `writer.py`.
- **Public API hygiene.** New symbols intended for users should be added to
  `src/autodata/__init__.py` and have a one-line docstring.

## Adding a domain

The fastest path: scaffold and edit.

```bash
uv run autodata init-domain my_domain --out my_domain.py
```

Then implement the six abstract methods and reference the file from your YAML
config (see `README.md` "Creating a new domain"). The bundled examples
(`src/autodata/domains/qa_from_documents.py`,
`src/autodata/domains/math_word_problems.py`) show the patterns.

## Reporting bugs / asking questions

Open a GitHub issue. Please include:

- Your config (with secrets redacted)
- The `summary.json` from the failing run
- One trajectory file from `outputs/<run_id>/trajectories/` if relevant
- Python version and `uv pip freeze | grep -E 'autodata|litellm|pydantic'`

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0 (see `LICENSE`).
