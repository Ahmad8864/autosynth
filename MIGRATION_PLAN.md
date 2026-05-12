# Migration plan: event-sourced pipeline over SQLite

**Status:** v2 — addresses review of v1. Ready to implement.

This document specifies the refactor from the current
`Orchestrator → RunWriter (JSONL)` design to an event-sourced pipeline over
a durable SQLite store, with a pluggable dispatcher for local-thread or
provider-batch execution.

It is structured as a spec (§1–5), a file map (§6–7), an ordered commit
sequence (§8), and the compatibility surface adopters see (§9).

---

## What changed from v1

- Pipeline states reduced from 11 → 5 (sync branches are no longer states).
- Middleware DSL dropped; `LLMClient` is a single ~200 LOC class.
- Response cache removed from v1 (silent-collapse risk; deferred until use case is concrete).
- One `Dispatcher` class with a `fulfill` strategy, not two classes.
- `paused` status and `FAILED` terminal state removed; Ctrl-C + `--resume` and `REJECTED("unrecoverable: …")` cover them.
- Resume semantics specified for in-flight requests, including batch.
- Rounds-table write timing specified.
- `solver_scores` references both solver and judge responses.
- Partial-completion invariant on `step()` made explicit and testable.
- Operator UX (CLI + logs) specified — `autodata status`, `autodata resume`, `autodata export`.

---

## 1. Goals and non-goals

**Goals**

- Pure pipeline: `step(state, responses) -> (state, requests)`. No I/O.
- Durable: SQLite is the primary store. Any kill is safe; `--resume` works.
- Pluggable dispatch: local threads today, provider batch APIs without
  pipeline changes.
- CLI surface preserved: `autodata run --config CONFIG.yaml` unchanged.
- `DomainAdapter`, `HarnessSpec`, `RunConfig`, example domains, the
  `mock/` provider, and the paper's algorithm (acceptance rules, rubric
  semantics, reflection, Boltzmann selection, train/val gate) — all
  preserved.

**Non-goals**

- Distributed execution. Architecture *enables* it; not implementing.
- Postgres / DuckDB as alternative stores.
- New providers beyond LiteLLM coverage.
- New domains.
- Response caching across runs.
- Any change to the paper's algorithm.

---

## 2. Pipeline spec

### 2.1 Item states (5)

```
PENDING            created, no work started
NEED_CANDIDATE     waiting on challenger response
NEED_QUALITY       waiting on quality verifier response
NEED_SCORES        waiting on weak+strong attempts and their judge scores
NEED_REFLECTION    waiting on reflector response (round 2+ only)
ACCEPTED           terminal
REJECTED           terminal (rounds exhausted OR unrecoverable error)
```

`NEED_SCORES` is one state, not two. Weak and strong fan out concurrently
inside it; the state advances only when *all* N×2 judge responses for the
current round are in. This is simpler than `NEED_WEAK` → `NEED_STRONG` and
maps directly to how the dispatcher batches.

Sync work (structural validation, safety filter, acceptance evaluation)
happens *inside* the response handler for the preceding async state. These
aren't durable states because there's nothing to wait on.

### 2.2 The step function

```python
@dataclass(frozen=True)
class ItemState:
    item_id: str
    run_id: str
    source_id: str
    domain: str
    state: State
    current_round: int          # 1-indexed
    rounds_history: list[Round] # prior completed rounds (for reflector)
    candidate: Candidate | None # current round's candidate
    quality: QualityCheck | None
    weak_scores: list[SolverScore]
    strong_scores: list[SolverScore]
    pending_request_ids: set[str]
    source_metadata: dict[str, Any]

@dataclass(frozen=True)
class LLMRequest:
    request_id: str   # deterministic: stable_id(item_id, round, role, attempt)
    item_id: str
    round_n: int
    role: Literal["challenger","quality","weak","strong","judge","reflector"]
    model_key: str
    messages: list[Message]
    json_mode: bool
    attempt: int = 0
    parent_response_id: str | None = None  # judge req → solver resp it scores

@dataclass(frozen=True)
class StepResult:
    state: ItemState
    new_requests: list[LLMRequest]   # may be empty
    completed_round: Round | None    # set only when a round terminates

def step(item: ItemState,
         new_responses: list[Response],
         *,
         cfg: RunConfig,
         harness: HarnessSpec,
         domain: DomainAdapter) -> StepResult:
    """Pure. Deterministic in (item, new_responses, cfg, harness, domain).
    No I/O, no time.time(), no randomness."""
```

### 2.3 Partial-completion invariant (testable)

`step()` MUST be a no-op when called with responses that don't complete
the current state:

```
step(item in NEED_SCORES with 3/6 judge responses) → StepResult(
    state=item (unchanged),
    new_requests=[],
    completed_round=None,
)
```

Test `test_pure_pipeline.py::test_partial_responses_noop` enforces this.
Violating it causes double-emission of requests on the next dispatcher
loop iteration.

### 2.4 Transition diagram

```
PENDING ─sync→ emit [challenger_req] → NEED_CANDIDATE

NEED_CANDIDATE
  └─response (parse ok, struct valid, safety ok)→ emit [quality_req] → NEED_QUALITY
  └─response (any sync check fails)→
        if round_n < max_rounds:  emit [reflector_req] → NEED_REFLECTION
        else:                     REJECTED

NEED_QUALITY
  └─response (passed)→ emit [weak_req × N, strong_req × N] → NEED_SCORES
  └─response (failed)→
        if round_n < max_rounds:  emit [reflector_req] → NEED_REFLECTION
        else:                     REJECTED

NEED_SCORES
  └─response (solver attempt k done)→ emit [judge_req] for that response (no state change)
  └─response (judge score done, last of 2N)→ run acceptance evaluator (sync):
        accepted     → ACCEPTED  (completed_round set)
        rejected and round_n < max_rounds → emit [reflector_req] → NEED_REFLECTION
        rejected and round_n == max_rounds → REJECTED

NEED_REFLECTION
  └─response→ round_n += 1; rounds_history.append(prior); emit [challenger_req] → NEED_CANDIDATE
```

A `Round` row is materialized in storage at the *first* response of that
round (when the challenger response parses). It's updated in-place as
quality / scores / evaluation land. `accepted=1` is set only on the
`ACCEPTED` transition.

### 2.5 Parallelism

`NEED_SCORES` is the throughput unlock: `N` weak + `N` strong solver
requests fire at once, and each judge fires the moment its solver
response arrives. Across `M` items concurrently in `NEED_SCORES`, the
dispatcher has up to `M × 4N` requests in flight (bounded by `cfg.dispatcher.concurrency`).

### 2.6 Determinism

The dispatcher orders responses by `request_id` before invoking `step()`,
so the order in which the threadpool happens to complete requests does
not perturb state evolution. Critical for reproducible runs and
deterministic resume.

---

## 3. Storage (SQLite)

One `run.db` per run. WAL mode, `synchronous=NORMAL`. JSON blobs are
inline TEXT columns; the database is the run.

### 3.1 Schema

```sql
PRAGMA user_version = 1;  -- bumped on schema migration

CREATE TABLE runs (
    run_id            TEXT PRIMARY KEY,
    config_blob       TEXT NOT NULL,
    harness_blob      TEXT,
    started_at        TEXT NOT NULL,
    last_active_at    TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL,   -- 'running' | 'completed' | 'aborted'
    cost_usd_cap      REAL,            -- null = no cap
    cost_usd_actual   REAL NOT NULL DEFAULT 0
);

CREATE TABLE items (
    item_id           TEXT PRIMARY KEY,   -- stable_id(run_id, source_id)
    run_id            TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    domain            TEXT NOT NULL,
    state             TEXT NOT NULL,
    current_round     INTEGER NOT NULL DEFAULT 1,
    source_metadata   TEXT,               -- JSON
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,      -- ALSO: max(received_at) of consumed responses
    final_round       INTEGER,
    rejection_reasons TEXT,               -- JSON list
    UNIQUE(run_id, source_id)
);
CREATE INDEX items_run_state ON items(run_id, state, updated_at);

CREATE TABLE rounds (
    round_id          TEXT PRIMARY KEY,
    item_id           TEXT NOT NULL REFERENCES items(item_id),
    round_n           INTEGER NOT NULL,
    candidate_blob    TEXT,
    quality_blob      TEXT,
    eval_blob         TEXT,
    reflection        TEXT,
    started_at        TEXT NOT NULL,
    ended_at          TEXT,
    accepted          INTEGER NOT NULL DEFAULT 0,
    UNIQUE(item_id, round_n)
);
CREATE INDEX rounds_item ON rounds(item_id);

CREATE TABLE requests (
    request_id         TEXT PRIMARY KEY,
    item_id            TEXT NOT NULL REFERENCES items(item_id),
    round_n            INTEGER NOT NULL,
    role               TEXT NOT NULL,
    model_key          TEXT NOT NULL,
    attempt            INTEGER NOT NULL DEFAULT 0,
    messages_blob      TEXT NOT NULL,
    json_mode          INTEGER NOT NULL DEFAULT 0,
    parent_response_id TEXT,
    status             TEXT NOT NULL,    -- 'pending' | 'in_flight' | 'done' | 'failed'
    submitted_at       TEXT,
    completed_at       TEXT,
    batch_id           TEXT,             -- NULL for local, set for batch dispatcher
    failure_count      INTEGER NOT NULL DEFAULT 0,
    last_error         TEXT
);
CREATE INDEX requests_status ON requests(status);
CREATE INDEX requests_item ON requests(item_id, round_n);
CREATE INDEX requests_batch ON requests(batch_id) WHERE batch_id IS NOT NULL;

CREATE TABLE responses (
    response_id        TEXT PRIMARY KEY,  -- == request_id (1:1)
    request_id         TEXT NOT NULL REFERENCES requests(request_id),
    model              TEXT NOT NULL,
    text               TEXT NOT NULL,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    cost_usd           REAL,
    duration_ms        INTEGER,
    received_at        TEXT NOT NULL
);
CREATE INDEX responses_received ON responses(received_at);

CREATE TABLE solver_scores (
    score_id           TEXT PRIMARY KEY,
    round_id           TEXT NOT NULL REFERENCES rounds(round_id),
    solver             TEXT NOT NULL,    -- 'weak' | 'strong'
    attempt            INTEGER NOT NULL,
    total              REAL NOT NULL,
    per_criterion      TEXT,
    failure_modes      TEXT,
    solver_response_id TEXT NOT NULL REFERENCES responses(response_id),
    judge_response_id  TEXT NOT NULL REFERENCES responses(response_id),
    UNIQUE(round_id, solver, attempt)
);
CREATE INDEX scores_round ON solver_scores(round_id, solver);

CREATE TABLE accepted (
    accepted_id        TEXT PRIMARY KEY,
    item_id            TEXT NOT NULL REFERENCES items(item_id),
    round_id           TEXT NOT NULL REFERENCES rounds(round_id),
    payload_blob       TEXT NOT NULL,
    accepted_at        TEXT NOT NULL
);
```

`solver_scores` references both the solver's raw response AND the judge's
response — necessary for debugging bad scores and for the `inspect` CLI.

### 3.2 Hot-path queries

```sql
-- Items whose state can advance: any with unconsumed responses, or that just
-- became unblocked (no pending requests AND state is non-terminal AND has new
-- responses since last update).
SELECT i.item_id FROM items i
WHERE i.run_id = ?
  AND i.state NOT IN ('ACCEPTED','REJECTED')
  AND EXISTS (
    SELECT 1 FROM responses r
    JOIN requests q ON q.request_id = r.request_id
    WHERE q.item_id = i.item_id AND r.received_at > i.updated_at
  )
ORDER BY i.updated_at LIMIT ?;

-- Items that need a first step (PENDING, no requests yet)
SELECT item_id FROM items
WHERE run_id = ? AND state = 'PENDING' LIMIT ?;

-- Claim pending requests for dispatch (atomic)
UPDATE requests SET status='in_flight', submitted_at=?
WHERE request_id IN (
  SELECT request_id FROM requests WHERE status='pending' LIMIT ?
) RETURNING *;

-- Cost so far (cheap; indexed implicitly by responses_received via aggregate)
SELECT COALESCE(SUM(cost_usd), 0) FROM responses
WHERE request_id IN (SELECT request_id FROM requests WHERE item_id IN
  (SELECT item_id FROM items WHERE run_id = ?));

-- For metaopt: rejection reasons aggregated for a run
SELECT json_extract(eval_blob, '$.rejection_reasons')
FROM rounds r JOIN items i ON i.item_id = r.item_id
WHERE i.run_id = ? AND r.accepted = 0;
```

### 3.3 Round materialization timing

A `rounds` row is INSERTED when the challenger response is successfully
parsed (candidate stored, `accepted=0`). It is UPDATED in place as
`quality_blob`, `eval_blob`, `reflection` land. It is finalized
(`accepted=1`, `ended_at` set) only on the `ACCEPTED` transition. A kill
mid-round leaves a partial row; resume picks up from the last
in-flight/pending requests for that round_n.

### 3.4 Export

`autodata export <run_id> --format jsonl` reads `accepted` rows and emits
`accepted.jsonl` matching the legacy v1 schema. Lazy by default; never
written during the run.

`autodata export <run_id> --format hf` writes the HF dataset directory
(same shape as v1's `RunWriter.export_hf`).

---

## 4. Dispatcher

One class. The execution strategy (local-threads vs. provider-batch) is
the `fulfill` callable.

```python
class Dispatcher:
    def __init__(self, store: Store, llm: LLMClient,
                 pipeline: Pipeline, domain: DomainAdapter,
                 harness: HarnessSpec, cfg: RunConfig,
                 fulfill: Fulfill):
        ...

    def run(self) -> RunSummary:
        while True:
            advanced = self._advance_ready_items()
            in_flight = self._poll_in_flight()       # batch: fetch completed; local: no-op
            dispatched = self._dispatch_pending()    # calls self.fulfill
            if not advanced and not in_flight and not dispatched:
                if self._all_terminal():
                    return self._summarize()
                self._idle_sleep()                   # only matters for batch (poll cadence)
            self._check_budget()                     # exits cleanly if exceeded
            self._check_signals()                    # SIGINT/SIGTERM → graceful exit
```

### 4.1 Two `Fulfill` strategies

```python
Fulfill = Callable[[list[LLMRequest]], None]
# Implementation responsibility: pull requests, mark in_flight, perform IO,
# insert responses + mark done OR mark failed. Atomicity via store.tx().

def fulfill_local(requests, *, llm, store, pool):
    # pool.submit per request; as_completed → store.insert_response + mark_done
    ...

def fulfill_batch(requests, *, provider, store):
    # group by (provider, model), call provider.submit_batch, store.tag_batch
    ...
```

Both honor `cfg.dispatcher.concurrency` and `cfg.dispatcher.poll_interval_s`.
Local uses `ThreadPoolExecutor(max_workers=concurrency)`; batch uses
`concurrency` as the per-submit chunk size.

### 4.2 Resume semantics

On `Dispatcher.__init__` with `cfg.resume=True`, normalize request state:

| Pre-restart state | `batch_id` | Action |
|---|---|---|
| `pending` | — | leave; will dispatch normally |
| `in_flight` | `NULL` | revert to `pending` (local fulfill lost its work) |
| `in_flight` | not null | leave; poll loop will fetch from provider |
| `done` w/o matching `responses` row | — | revert to `pending` (rare; crash between insert_response and mark_done) |
| `failed` w/ `failure_count < cfg.max_request_failures` | — | revert to `pending` |
| `failed` w/ `failure_count >= cfg.max_request_failures` | — | leave; owning item marked REJECTED with `unrecoverable: <error>` |

All deterministic by `request_id`. Restart is idempotent. No double-submit
on batch resume because we never re-submit a request that already has a
`batch_id`.

### 4.3 Budget

`cfg.budget_usd` (default null = unlimited). At each loop iteration:

- `cost_so_far = SELECT SUM(cost_usd) ...` (cached, recomputed every K seconds).
- If `cost_so_far > 0.8 * budget` and not already warned → `logger.warning`.
- If `cost_so_far >= budget` → mark `runs.status='aborted'`, exit cleanly with status 2.

Resume re-checks; an `aborted` run becomes `running` again only if budget
was raised in the new config.

### 4.4 Signals

`SIGINT` / `SIGTERM` → set a stop flag, finish whatever requests are
currently in `as_completed` for ≤ `cfg.dispatcher.shutdown_grace_s`
seconds, then exit. `runs.status` stays `running` (a later resume picks
up). Pressing Ctrl-C twice forces immediate exit; in-flight responses
may be lost but will be reissued on resume.

---

## 5. LLMClient

One class. ~200 LOC. No middleware framework.

```python
class LLMClient:
    """Single entry point for completions. Handles provider routing
    (LiteLLM vs. mock), retries (tenacity), rate limiting (token bucket
    per (provider, model)), and cost accounting."""

    def __init__(self, cfg: LLMConfig, mock_registry: MockRegistry | None = None):
        self.rate_limiters = {}  # keyed by (provider, model)
        ...

    def complete(self, req: LLMRequest) -> Response:
        limiter = self._limiter_for(req.model_key)
        with limiter:
            return self._call_with_retry(req)

    def _call_with_retry(self, req): ...
    def _limiter_for(self, model_key): ...    # lazy-init token buckets
```

Cost accounting is **not** a wrapper — it's part of the `Response`
construction (LiteLLM returns `usage`, we map to `cost_usd` via a
provider-specific price table in `llm_pricing.py`, store on the response
row at insert time).

`MockProvider` is the same `LLMClient` with a `mock://` model string;
calls dispatch to the existing `register_mock(scenario, handler)`
registry. **Mock call-site is unchanged** for users — `mock/happy`,
`mock/reject` still work via `register_mock`.

### 5.1 Rate limiting

Per-(provider, model) token bucket. Config:

```yaml
llm:
  rate_limits:
    "anthropic/*":      { rpm: 4000, tpm: 400000 }
    "openai/*":         { rpm: 3500, tpm: 350000 }
    "mock/*":           { rpm: null }              # no limit
```

Glob matching; first match wins; missing entry = no limit. Buckets are
process-local — fine for `LocalDispatcher`; for distributed dispatch
later, swap in a Redis-backed limiter behind the same interface.

### 5.2 Retry

`tenacity` with exponential backoff, capped at `cfg.max_retries`. On
final failure, the request row's `failure_count += 1`, `last_error` is
set, and `status='failed'`. The dispatcher inspects `failure_count` to
decide reissue vs. terminal-reject (see §4.2).

---

## 6. File map

### Keep, unchanged (8)

`src/autodata/schemas.py`, `domain.py`, `domains/*`, `harness.py`,
`safety.py`, `utils.py`, `evaluator.py` (truly pure already).

### Modify (2)

- `config.py` — add `LLMConfig` (`rate_limits`, `max_retries`,
  `request_timeout_s`), `DispatcherConfig` (`concurrency`, `poll_interval_s`,
  `shutdown_grace_s`, `max_request_failures`), `budget_usd`. All optional
  with sensible defaults so existing configs run unchanged.
- `metaopt.py` — replace `Orchestrator` calls with `Runner`;
  `_collect_failures` switches from trajectory-glob to SQL aggregate
  (§3.2 last query). Inline-or-delete `aggregate_failures` (sole caller).
  ~80 LOC delta.

### Refactor (4) — agents split into request-builder + response-parser

Each agent file gets two module-level functions; the class wrapper stays
for the metaopt path but delegates to them.

- `agents/challenger.py`:
  `build_request(item, round_n, feedback, prior_payloads, domain, harness, rubric_max_weight) -> LLMRequest`
  `parse_response(text, item, round_n, domain, rubric_max_weight) -> Candidate`
- `agents/solver.py`:
  `build_request(candidate, role, attempt, domain, harness) -> LLMRequest`
  (no parse — response text *is* the attempt)
- `agents/verifier.py`:
  `build_quality_request(candidate, domain, harness) -> LLMRequest`
  `parse_quality(text) -> QualityCheck`
  `build_judge_request(candidate, solver_response, solver_role, attempt, domain, harness, parent_response_id) -> LLMRequest`
  `parse_judge(text, candidate, solver_role, attempt, solver_response, solver_response_id, judge_response_id) -> SolverScore`
- `agents/reflector.py`:
  `build_request(rounds_history, domain_name, leakage_rules, harness, acceptance) -> LLMRequest`
  `parse_response(text) -> ReflectionResult`

### Rewrite (3)

- `models.py` → `llm.py` — single `LLMClient` (§5).
- `orchestrator.py` → split into:
  - `pipeline.py` — `step()`, `ItemState`, transition logic.
  - `runner.py` — thin driver: opens store, builds dispatcher,
    calls `dispatcher.run()`, returns summary. ~100 LOC.
- `writer.py` → `store.py` — SQLite DAO + JSONL/HF export helpers.

### New (3)

- `pipeline.py`, `store.py`, `dispatcher.py`, `llm.py` (replacing
  `models.py`). Effective new files: 3.

### Delete (2)

- `orchestrator.py`, `writer.py`.

---

## 7. Test plan

Current: 8 files / ~39 test functions. After: 11 files / ~58 functions.

### Keep (3)

`tests/test_schemas.py`, `test_acceptance.py`, `test_harness.py`.

### Modify (2)

- `conftest.py` — `register_mock` API unchanged; one new fixture
  `run_with_mock(scenario, **overrides) -> Path` that builds a tiny
  RunConfig and returns the `run.db` path.
- `test_domain_loading.py` — likely passes as-is.

### Rewrite (3)

- `test_mock_provider.py` → `test_llm.py` — rate-limit bucket math, retry
  on transient errors, cost accounting, mock dispatch.
- `test_full_loop.py` → `test_integration.py` — same end-to-end
  assertions, but driven via `Runner` + `LocalDispatcher` + mock provider.
- `test_writer.py` → `test_store.py` — schema, `claim_pending`
  atomicity under concurrent threads, round materialization timing,
  resume normalization (§4.2 table), JSONL/HF export.

### New (3)

- `test_pure_pipeline.py` — exhaustive transitions on `step()`. **Highest
  priority new file.** Must include `test_partial_responses_noop` for
  §2.3.
- `test_dispatcher.py` — LocalDispatcher: 100 concurrent requests
  produce 100 valid responses; idempotent reissue after simulated crash;
  budget exit. BatchDispatcher: submit + poll + recover.
- `test_resume.py` — crash mid-`NEED_SCORES`, restart, verify (a) no
  double-submit, (b) no duplicate `accepted` rows, (c) `request_id`
  determinism.

### Modify (1)

- `test_metaopt.py` — `_collect_failures` SQL path; algorithm
  assertions unchanged. ~30% rewrite.

---

## 8. Commit sequence

Each commit leaves `pytest -x` green. Pause-points marked.

1. **`store.py` + SQLite schema** + `test_store.py`. Standalone module;
   nothing imports it yet. ~350 LOC + 120 LOC tests.

2. **`llm.py`** + `test_llm.py`. Standalone; `mock://` routing works;
   rate limit + retry + cost tested. Old `models.py` untouched. ~250 LOC
   + 100 LOC tests. *— PAUSE-POINT: foundations done.*

3. **Agent refactor (additive).** Add module-level `build_*` / `parse_*`
   functions to each agent file. Class wrappers stay, untouched.
   Existing `orchestrator.py` still works. ~150 LOC delta, 0 LOC
   deleted.

4. **`pipeline.py` step function** + `test_pure_pipeline.py`. Pure
   module, no I/O imports. ~400 LOC + 250 LOC tests. The partial-
   completion test is non-negotiable. *— PAUSE-POINT: pipeline logic
   isolated and tested.*

5. **`dispatcher.py` `LocalDispatcher`** + `test_dispatcher.py`. Uses
   store + llm + pipeline. ~250 LOC + 150 LOC tests.

6. **`runner.py` + cutover.** New `runner.py`. Rewrite
   `test_integration.py` and add `test_resume.py`. Update `cli.py` to
   route `autodata run` through `Runner`. **Delete** `orchestrator.py`,
   `writer.py`, `models.py`. ~250 LOC. *— PAUSE-POINT: full migration
   complete; legacy code gone.*

7. **Metaopt port.** Switch `_eval` to `Runner`, `_collect_failures` to
   SQL. Inline `aggregate_failures`. ~80 LOC delta.

8. **CLI polish.** `autodata status <run_id>` (one-shot table querying
   item states + counts + cost), `autodata resume <run_id>`,
   `autodata export <run_id> --format jsonl|hf`. ~80 LOC.

9. **`BatchDispatcher`.** OpenAI + Anthropic batch submission +
   polling. Behind `--dispatcher batch`. ~300 LOC + 80 LOC tests.

10. **README + this plan.** Update with new architecture, new CLI, and
    dispatcher flag examples. Remove the JSONL-as-primary references.

After step 6, the framework is fully migrated and usable. 7–10 are
incremental improvements that can be split into separate PRs if the
single-PR diff gets too big.

---

## 9. Operator UX

### CLI

```
autodata run --config CONFIG.yaml [--dispatcher local|batch] [--resume RUN_ID]
autodata resume RUN_ID                # alias for `run --resume`
autodata status RUN_ID                # one-shot table: states, counts, cost, ETA
autodata export RUN_ID --format jsonl|hf [--out PATH]
autodata inspect-run RUN_ID --stuck   # items in non-terminal states with last error
autodata metaopt --config CONFIG.yaml
autodata init-domain NAME
```

### Logs (loguru, structured)

- `DEBUG`: every state transition (`item=… round=… state=NEED_QUALITY→NEED_SCORES`).
- `INFO`: per-item terminal lines (`item=… ACCEPTED round=3 cost=$0.012`),
  cost milestones (`cost=$X (10% of budget)`), run start/stop.
- `WARNING`: rate-limit backoff, parse failures (recoverable), 80% budget.
- `ERROR`: unrecoverable per-item failures (the item is now `REJECTED`).

Default sink: stdout. Adopters can add file/JSON sinks via loguru
config — no new framework. No more Rich progress bar; `autodata status`
is the operator's UI, and `watch -n 2 autodata status RUN_ID` covers the
live-monitor case without adding a TUI dependency to the hot path.

### Failure surface

An item reaches `REJECTED` for three distinct reasons, all in
`rejection_reasons`:

- `exhausted_rounds: <last_round_reasons>` — paper-faithful path.
- `unrecoverable: <error>` — `failure_count` exceeded
  `cfg.max_request_failures` for any request the item owns.
- `safety: <reasons>` — safety filter rejected at round = max_rounds.

---

## 10. Meta-optimization

Algorithm unchanged. Code-level changes:

| Today | After |
|---|---|
| `MetaOptimizer._eval` calls `Orchestrator(...).run()` | calls `Runner(...).run()` |
| `_collect_failures` globs `<run_dir>/trajectories/*.json` | `SELECT json_extract(eval_blob, '$.rejection_reasons') FROM rounds WHERE ...` |
| Per-iter dirs `outputs/metaopt/<run>/iterations/iter_NNN/{train,val}/` | Same on-disk layout; each inner run is a `run.db` in that dir |
| `population.json`, `iterations_log.json` | Unchanged (small meta-state, JSON is fine) |

Meta-opt does NOT get its own SQLite store. It owns small structured
state; JSON is appropriate. The inner runs use SQLite.

---

## 11. Out of scope (explicit)

- Multi-machine dispatch (architecture enables; no impl).
- Postgres / DuckDB.
- New providers, new domains.
- Response cache across runs.
- TUI / web UI.
- Any change to the paper's algorithm.

---

## 12. Effort estimate

- ~1500 LOC churn (~1000 net new, ~500 deleted).
- 3 new test files, 4 modified, 3 unchanged.
- **Two focused sessions** to land commits 1–6 (full migration) with
  the suite green.
- One session for commits 7–10 (metaopt port, CLI polish, batch
  dispatcher).

The original v1 estimate of "one session" was optimistic; calibrated
honestly here.

---

## 13. Compatibility surface (what an existing user sees)

| Surface | Before | After |
|---|---|---|
| `autodata run --config X.yaml` | ✓ | ✓ unchanged |
| `autodata metaopt --config X.yaml` | ✓ | ✓ unchanged |
| `autodata init-domain` | ✓ | ✓ unchanged |
| `DomainAdapter` interface | ✓ | ✓ unchanged |
| `HarnessSpec`, `register_mock`, mock scenarios | ✓ | ✓ unchanged |
| YAML config | structure | structure preserved; new optional `dispatcher:` and `llm:` blocks with defaults that match v1 behavior |
| Run output | `outputs/<run_id>/{accepted,rejected}.jsonl + trajectories/*.json` | `outputs/<run_id>/{run.db, config.snapshot.yaml}`; legacy files produced by `autodata export` on demand |
| New CLI | — | `autodata status`, `autodata resume`, `autodata export` |

A v1 YAML config runs unmodified on v2 and produces identical accepted
items (mod LLM nondeterminism). The visible-on-disk change is one file
(`run.db`) replacing a tree of files; export reproduces the legacy
tree.

---

## Sign-off

This plan is ready to implement. Start at commit 1 (§8). The pause-points
after commits 2, 4, and 6 are natural review gates.
