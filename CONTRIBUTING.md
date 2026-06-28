# Contributing

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

Activate with `source .venv/bin/activate`, or just prefix things with `uv run`. Needs Python 3.10+.

## Tests

```bash
uv run pytest
```

Tests use an in-process mock LLM, so no API keys needed. If you're adding real-provider behavior, script the response with `register_mock(...)` in your test.

## Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

Both run in CI. Config is in `pyproject.toml`.

## Pre-commit hooks

After cloning, install the hooks once:

```bash
uv run pre-commit install
```

That wires up ruff (lint + format), file hygiene checks (trailing whitespace, large files, YAML/TOML syntax), gitleaks secret scanning, and a commit-message check (see [Commit messages](#commit-messages)). They fire on every `git commit`. Run against the whole tree on demand with `uv run pre-commit run --all-files`.

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org):

```
<type>[(scope)][!]: <subject>
```

- **type**: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `style`, or `chore`.
- **subject**: imperative, lowercase, no trailing period ("add", not "added").
- Add `!` plus a `BREAKING CHANGE:` footer for anything that breaks a public import or behavior.
- An optional body (wrap at 72) covers *what* and *why*, not *how*.

```
feat(acceptance): add verifiable acceptance mode
fix(dispatcher): close consumption race on resume
docs(readme): document verifiable mode
refactor(acceptance)!: move evaluate() to autosynth.acceptance
```

The `commitizen` commit-msg hook checks this on every commit; `uv run cz commit` walks you through a valid message interactively.

## Before you open a PR

- Add a test for the thing you changed. Bug fixes get the test that would've caught the bug.
- Don't add new core dependencies. The mock demo needs to keep working offline. Heavy stuff goes under an optional extra.
- Keep the core domain-agnostic. Domain-specific code goes in a `DomainAdapter`, not in `orchestrator.py` / `evaluator.py` / `writer.py`.
- New public symbols go in `src/autosynth/__init__.py` with a one-line docstring.

## Adding a domain

Scaffold one and fill it in:

```bash
uv run autosynth init-domain my_domain --out my_domain.py
```

Implement the six abstract methods, then point your YAML config at the file (see the README). `src/autosynth/domains/qa_from_documents.py` and `math_word_problems.py` are good references.

## Bugs and questions

Open a GitHub issue. Helpful things to include:

- Your config (redact secrets)
- `summary.json` from the failing run
- A trajectory file from `outputs/<run_id>/trajectories/` if it's relevant
- Python version and `uv pip freeze | grep -E 'autosynth|litellm|pydantic'`

## License

Contributions are licensed under MIT (see `LICENSE`).
