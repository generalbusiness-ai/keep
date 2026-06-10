# Plan 005: Verify and lock in the lock-free SQLite read path with a concurrency regression test

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat e3bb33d..HEAD -- keep/document_store.py tests/test_concurrency.py`
> If either changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it as a
> STOP condition.

## Status

- **Priority**: P3
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: perf
- **Planned at**: commit `e3bb33d`, 2026-06-10

## Why this matters

A prior audit (TECH_DEBT_AUDIT.md, finding F015) recommended dropping the global
store lock for SELECTs so WAL-mode concurrent reads aren't serialized. **That fix
has since landed** — `document_store.py`'s read helpers already branch read-only
SQL around `self._lock` onto thread-local connections. But there is **no test that
proves it**, so a future refactor could silently re-introduce the lock on the read
path and no one would notice until production latency regressed. This plan does two
things: (1) confirm the optimization is actually present and correct as currently
written, and (2) add a concurrency regression test that fails if reads ever start
serializing again. The point is to convert an undocumented, untested win into a
guarded one. **This is primarily a verification + test plan; production code should
change little or not at all.**

## Current state

- `keep/document_store.py` — the SQLite store. Relevant facts:
  - `__init__` sets `self._lock = threading.RLock()` (around line 217) and a
    separate `self._connections_lock = threading.RLock()` (line 214). Each thread
    gets its **own** sqlite3 connection; WAL is enabled
    (`conn.execute("PRAGMA journal_mode=WAL")`, ~line 242).
  - There is a helper `_is_readonly_sql(sql)` (~line 70) returning `True` only for
    `SELECT`/`EXPLAIN`.
  - The read helpers already bypass the lock for read-only SQL. Example —
    `_fetchall` (~line 463):

```python
            if _is_readonly_sql(sql):
                return self._run_sql(
                    sql=sql, params=params,
                    materialize=lambda conn: conn.execute(sql, params).fetchall(),
                )
            with self._lock:
                return self._run_sql(...)
```

    The same readonly branch exists in `_execute` (~line 425) and `_fetchone`
    (~line 448). Writes go through `_execute_write`/`_executemany`, which always
    take `self._lock`.
  - `get()` (~line 2519) reads via `_fetchone` — so it is already on the
    lock-free path. Multi-statement transactions (e.g. the archive-replace in
    `put`, ~line 1675, which does `BEGIN IMMEDIATE` under `self._lock`) correctly
    stay locked.

- `tests/conftest.py` — owns store isolation and daemon cleanup. The real
  `DocumentStore` (not the `MockDocumentStore`) is what this test must exercise,
  against a temp store path.

- `tests/test_concurrency.py` — exists (the repo has concurrency tests already).
  **Read it** to learn the established pattern for spinning up the real store and
  hitting it from multiple threads; add the new test there or in a sibling file
  following the same pattern.

## Commands you will need

| Purpose             | Command                                                | Expected on success      |
|---------------------|--------------------------------------------------------|--------------------------|
| Inspect read path   | `grep -n "_is_readonly_sql\|with self._lock" keep/document_store.py` | shows readonly branches |
| Targeted tests      | `python -m pytest tests/test_concurrency.py -q`        | all pass                 |
| Full suite          | `python -m pytest tests/ -x -q`                        | all pass, no daemon left |
| Lint                | `ruff check tests/test_concurrency.py`                 | exit 0                   |

## Scope

**In scope**:
- `tests/test_concurrency.py` (modify) **or** a new `tests/test_store_concurrency.py`
  (create) — the regression test.
- `keep/document_store.py` — **only if** Step 1 finds the read path is NOT actually
  lock-free as described (i.e. the audit's claim that it's fixed is wrong). In that
  case the scope expands to making reads lock-free; see Step 3.

**Out of scope** (do NOT touch):
- The write path (`_execute_write`, `_executemany`, `BEGIN IMMEDIATE` blocks) —
  writes must keep serializing through `self._lock`. Do not "optimize" them.
- ChromaDB store (`keep/store.py`) and the embedding cache — different stores;
  not part of this finding.

## Git workflow

- Branch: `advisor/005-lockless-read-test`
- One commit: `test: guard lock-free SQLite read path against regression`.
- Do NOT push or open a PR unless the operator instructs it.

## Steps

### Step 1: Confirm the read path is genuinely lock-free (decision point)

Read `keep/document_store.py` and confirm ALL of:

1. `_is_readonly_sql` returns `True` for `SELECT`/`EXPLAIN` only.
2. `_execute`, `_fetchone`, `_fetchall` each have a readonly branch that runs
   **without** `with self._lock:`.
3. Each thread uses its own connection (so two threads' reads don't share a
   cursor), and WAL is on.

If all three hold → the optimization is present; proceed to Step 2 (test only;
no production change). **If any does NOT hold** → the audit's "already fixed"
premise is wrong; STOP and report what you found, including whether you should
proceed to implement the lock-free reads (Step 3) instead.

**Verify**: write down, in the PR description, the three confirmations with line
numbers.

### Step 2: Add a concurrency regression test that proves reads don't serialize

The test must fail if reads are forced back under the global write lock. A robust
shape (adapt to the existing `tests/test_concurrency.py` patterns):

- Create a real `DocumentStore` on a temp path; seed it with enough documents that
  a read does measurable work.
- Hold a **write** lock busy: from one thread, begin a slow exclusive write
  operation (or directly acquire `store._lock` in the test to simulate a writer
  holding it), then from other threads issue `get()`/`find`-style **reads** and
  assert they complete *without blocking* on that held lock.
  - Concretely: acquire `store._lock` in the main thread, then in worker threads
    call a read method (e.g. `store.get(collection, id)`); assert the workers
    return within a short timeout while the lock is held. If reads were under the
    lock, they would block until the main thread releases it, and the assertion
    (completed within timeout) would fail.
- Keep the timing assertion generous enough to avoid CI flakiness (e.g. reads
  must finish within a few seconds while the lock is held), but tight enough that a
  truly blocked read (which would wait indefinitely / until release) is caught. Do
  **not** assert on microbenchmark latencies — assert on the binary
  "did the read complete while a writer held the lock" property, which is
  deterministic, not timing-sensitive.

Avoid real timers/sleeps as correctness gates where possible — prefer
`threading.Event` to coordinate "lock is now held" → "now run reads" →
"reads completed" so the test is deterministic, not race-prone. (AGENTS.md flags
real-timer/real-network flakiness as a test smell.)

**Verify**: `python -m pytest tests/test_concurrency.py -q` (or the new file) →
the new test passes.

### Step 3 (ONLY if Step 1 failed): make reads lock-free

Do this **only** if Step 1 showed reads are still under `self._lock`. Mirror the
existing readonly-branch pattern: in `_execute`/`_fetchone`/`_fetchall`, run
`_is_readonly_sql(sql)` statements on the thread-local connection without taking
`self._lock`; keep the locked branch for everything else. Then the Step 2 test
should pass. If Step 1 succeeded, skip this step entirely.

**Verify**: `python -m pytest tests/ -x -q` → full suite green (no write-path
regressions).

### Step 4: Negative check — prove the test would catch a regression

Temporarily make a read method take `self._lock` (e.g. wrap `get`'s `_fetchone`
call in `with self._lock:`), run the new test, and confirm it **fails**. Then
revert the temporary change. This proves the test actually guards the property.

**Verify**: with the temporary lock added, the new test fails; after revert,
`python -m pytest tests/test_concurrency.py -q` passes and `git diff keep/` is
empty (if Step 3 was skipped).

## Test plan

- New test asserting concurrent reads proceed while a writer holds `store._lock`,
  in `tests/test_concurrency.py` (or `tests/test_store_concurrency.py`), modeled
  on the existing concurrency tests in that file.
- A documented negative check (Step 4) proving the test fails if reads are
  re-locked.
- Verification: `python -m pytest tests/test_concurrency.py -q` passes;
  `python -m pytest tests/ -x -q` full suite green.

## Done criteria

ALL must hold:

- [ ] Step 1's three confirmations recorded (read path is lock-free) OR Step 3
      completed to make it so.
- [ ] A new concurrency test exists and passes, asserting reads complete while a
      writer holds `store._lock`.
- [ ] Step 4 demonstrated (and recorded in the PR) that the test fails when a read
      is artificially re-locked.
- [ ] `python -m pytest tests/ -x -q` passes, no daemon/thread left running.
- [ ] If Step 1 succeeded, `git diff keep/document_store.py` is empty (test-only
      change).
- [ ] `plans/README.md` status row for 005 updated.

## STOP conditions

Stop and report back (do not improvise) if:

- Step 1 shows the read path is NOT lock-free (the "already fixed" premise is
  wrong) — report before implementing Step 3, so the maintainer knows the audit
  was stale.
- The new test is flaky across 5 consecutive runs (`pytest ... --count=5` if
  pytest-repeat is available, else run it 5× manually) — a flaky concurrency test
  is worse than none; report the nondeterminism instead of merging it.
- Making reads lock-free (Step 3) surfaces a correctness issue (e.g. a read that
  actually depends on the write lock for consistency) — STOP; that is a design
  question for the maintainer.

## Maintenance notes

- For the reviewer: the value here is the regression guard. Confirm the test
  asserts the *binary* "read completed while writer held the lock" property, not a
  latency threshold (latency assertions are flaky and will be disabled, defeating
  the purpose).
- If the store ever moves to a connection pool or a different locking model, this
  test encodes the invariant that must be preserved: SELECTs do not block on
  writes.
- Deferred out of scope: tuning WAL checkpoint cadence and the
  `pending_summaries` recovery-under-lock items (TECH_DEBT_AUDIT F036/F037) — those
  are separate findings.
