# Plan 007: Consolidate the duplicated remote client-log lifecycle and add request_id correlation

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan in
> `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat e3bb33d..HEAD -- keep/remote.py keep/mcp.py keep/logging_config.py`
> If any changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it as a
> STOP condition.

## Status

- **Priority**: P3
- **Effort**: M (spike + refactor)
- **Risk**: MED
- **Depends on**: none
- **Category**: direction
- **Planned at**: commit `e3bb33d`, 2026-06-10

## Why this matters

The hosted/remote backend is reachable through two clients — `RemoteKeeper`
(CLI/Python API transport) and the MCP `_RemoteBackend` (`keep mcp` in remote
mode). Both attach a rotating client-log handler via the same
`configure_client_log()` helper writing to `{config_dir}/keep-client.log`, and
both implement a near-identical `_log_call()` that emits one INFO line per HTTP
call. That logging lifecycle (attach in `__init__`, `_log_call`, detach in
`close`) is **copy-pasted** across the two classes, so they drift independently —
and critically, neither log line carries the daemon's `request_id`, which is the
one token that correlates a client-side call with the server-side error. An
operator debugging a hosted-mode failure today gets a client log that can't be
joined to the server log. This plan deduplicates the lifecycle into one place and
adds `request_id` to the emitted line so the two sides become correlatable.

**This is a spike + refactor**: confirm the duplication and the request_id source
first, then extract the shared piece and thread the id through.

## Current state

- `keep/logging_config.py` — `configure_client_log(log_dir)` (line ~164) creates a
  `RotatingFileHandler` on `{log_dir}/keep-client.log` and attaches it to the
  `"keep"` logger. Both clients call this.

- `keep/remote.py` — `RemoteKeeper`:
  - `__init__` attaches the handler:
    ```python
        self._client_log_handler = None
        log_dir = (config.config_dir if config and config.config_dir
                   else (config.path if config and config.path else None))
        if log_dir is not None:
            try:
                self._client_log_handler = configure_client_log(log_dir)
            except OSError as e:
                logger.debug("Could not attach client log at %s: %s", log_dir, e)
    ```
  - `_log_call` (line ~213) emits:
    ```python
        logger.info("remote: %s %s status=%d wall=%dms host=%s",
                    method, path, status, wall_ms, self.api_url)
    ```
  - `close()` (line ~563) detaches and closes the handler.
  - `_get`/`_post`/`_patch`/`_delete` call `_log_call(...)` then `_raise_for_status`.
  - The daemon's `request_id` is available on error: `_raise_for_status` already
    extracts it (see `test_remote_http_error_includes_daemon_request_id`). **Read
    `_raise_for_status`** to learn how/where the `request_id` is read from the
    response body or headers — that is the value to add to the success/al log line.

- `keep/mcp.py` — `_RemoteBackend` (line ~122):
  - `__init__` attaches the same handler via `configure_client_log` (line ~149),
    storing it as `self._log_handler`.
  - `_log_call` (line ~153) is a copy of the same INFO-line logic.
  - `close()` (line ~192) detaches/closes `self._log_handler` — a copy of
    `RemoteKeeper.close()`'s handler cleanup.

  So: same file, same helper, **duplicated** attach/`_log_call`/detach code, and
  the attribute is named differently (`_client_log_handler` vs `_log_handler`).

- The CLI-in-daemon-mode path logs through the daemon's own logs (server side) and
  is **out of scope** — it is not a remote client and does not use
  `configure_client_log`.

## Commands you will need

| Purpose                 | Command                                                       | Expected on success      |
|-------------------------|---------------------------------------------------------------|--------------------------|
| Find the duplication    | `grep -n "_log_call\|configure_client_log\|_log_handler\|_client_log_handler" keep/remote.py keep/mcp.py` | the sites above |
| Targeted tests          | `python -m pytest tests/test_remote.py tests/ -k "remote or client_log or mcp" -q` | all pass |
| Full suite              | `python -m pytest tests/ -x -q`                              | all pass, no daemon left |
| Lint                    | `ruff check keep/remote.py keep/mcp.py keep/logging_config.py` | exit 0                 |

## Scope

**In scope**:
- `keep/logging_config.py` — add a small shared helper (e.g. a `ClientLog` /
  `RemoteCallLogger` object, or a function pair) that owns: attach handler,
  emit a standardized call line (including optional `request_id`), and detach.
- `keep/remote.py` — replace the inline attach/`_log_call`/detach with the shared
  helper; thread `request_id` into the emitted line.
- `keep/mcp.py` — same replacement for `_RemoteBackend`.

**Out of scope** (do NOT touch):
- The log **file name** and location (`{config_dir}/keep-client.log`) — both
  clients already share it; do not rename it (existing tests assert the path, e.g.
  `test_remote_attaches_client_log_to_config_dir`).
- The CLI/daemon server-side logging.
- `_raise_for_status`'s error-message format — you may *read* the request_id from
  there but the existing error-path behavior and its tests must stay green.
- The retry/transport logic in either client.

## Git workflow

- Branch: `advisor/007-remote-log-consolidation`
- Commit style: short imperative, e.g.
  `remote: share client-log lifecycle and add request_id correlation`.
- Do NOT push or open a PR unless the operator instructs it.

## Steps

### Step 1: Spike — confirm the duplication and locate the request_id source

- Run the `grep` above and confirm both classes carry their own copy of
  attach/`_log_call`/detach.
- Read `RemoteKeeper._raise_for_status` and identify exactly where the daemon's
  `request_id` lives on a response (body field `request_id`? a response header?).
  Decide whether it is available on **success** responses too or only errors —
  this determines whether the standardized log line includes `request_id` always,
  on-error-only, or when present. Record the finding.

**Verify**: a short written note (PR description) naming the shared-helper shape
and the request_id source, with line numbers.

### Step 2: Add the shared helper in logging_config.py

Introduce one place that owns the client-call logging lifecycle. Keep it minimal
and dependency-light (it must not import `remote`/`mcp` to avoid cycles). It should
provide: attach (returns the handler, or None on failure), a method/function to
emit the standardized line — `remote: <METHOD> <path> status=<n> wall=<ms>ms host=<url>`
plus ` request_id=<id>` when an id is supplied — and detach.

Match the existing line format exactly for the fields that tests already assert
(`remote: GET /v1/notes/abc`, `host=...`), only **appending** the optional
`request_id=` field so existing assertions still pass.

**Verify**: `python -c "import keep.logging_config"` → exit 0; `ruff check
keep/logging_config.py` → clean.

### Step 3: Switch RemoteKeeper to the shared helper

Replace the inline attach in `__init__`, the `_log_call` body, and the detach in
`close()` with calls to the Step 2 helper. Thread the `request_id` (from Step 1)
into the emitted line — e.g. capture it where `_get`/`_post` handle the response.
Keep the public behavior identical: handler still attaches to `{config_dir}/keep-client.log`,
`close()` still removes it and sets the handle to `None`.

**Verify**: `python -m pytest tests/test_remote.py -q` → the existing client-log
tests (`test_remote_attaches_client_log_to_config_dir`,
`test_remote_close_removes_client_log_handler`) still pass unchanged.

### Step 4: Switch _RemoteBackend (mcp.py) to the shared helper

Apply the same replacement. After this, neither class should contain a hand-rolled
`_log_call` or duplicated attach/detach — both go through the Step 2 helper.

**Verify**: `grep -n "def _log_call" keep/remote.py keep/mcp.py` → ideally zero
matches (both now delegate), or the two remaining `_log_call` are one-line
delegators to the shared helper. `python -m pytest tests/ -k "remote or mcp or client_log" -q` → pass.

### Step 5: Add a request_id correlation test

Add a test (in `tests/test_remote.py`, following its existing style) that drives a
remote call whose response carries a `request_id` and asserts the
`keep-client.log` line for that call includes `request_id=<that id>`. If Step 1
found request_id is error-only, assert it on the error path instead. (If plan 003's
fake-server fixture has landed, prefer driving this through it for a real
round-trip; otherwise the existing mock style is acceptable.)

**Verify**: `python -m pytest tests/test_remote.py -q` → new test passes.

## Test plan

- Existing client-log tests must pass unchanged (proves the refactor preserved
  behavior and the log path/format).
- One new test asserting `request_id` appears in the client-log line for a call
  that has one.
- Verification: `python -m pytest tests/ -k "remote or mcp or client_log" -q`
  passes; `python -m pytest tests/ -x -q` full suite green.

## Done criteria

ALL must hold:

- [ ] A single shared helper in `keep/logging_config.py` owns attach / emit /
      detach for remote client-call logging.
- [ ] `keep/remote.py` and `keep/mcp.py` both use it; no duplicated `_log_call`
      body or duplicated attach/detach remains (or only thin delegators).
- [ ] The emitted log line includes `request_id=<id>` when one is available, and
      the pre-existing line fields/format are unchanged (existing tests pass).
- [ ] A new test asserts the `request_id` appears in the client log.
- [ ] `python -m pytest tests/ -x -q` passes, no daemon left running.
- [ ] `ruff check` clean on the three touched files.
- [ ] `git status` shows only `keep/logging_config.py`, `keep/remote.py`,
      `keep/mcp.py`, and `tests/test_remote.py` modified.
- [ ] `plans/README.md` status row for 007 updated.

## STOP conditions

Stop and report back (do not improvise) if:

- The shared helper would need to import `remote`/`mcp` (circular import) — rethink
  the seam; report rather than introducing a cycle.
- Step 1 finds `request_id` is **not** available on the client at all (not in body
  or headers on the relevant responses) — then the correlation half can't be done
  as scoped; complete the dedup (Steps 2–4) and report the request_id gap as a
  separate finding (it may require a daemon-side change to echo the id).
- An existing client-log test changes meaning (not just passes) — that signals the
  refactor altered behavior; STOP and reconcile.
- "Current state" excerpts don't match the live code (drift).

## Maintenance notes

- For the reviewer: the win is one lifecycle, not two. Confirm both clients route
  through the shared helper and that the log path/format assertions in
  `tests/test_remote.py` still hold (the file name `keep-client.log` and the
  `host=`/`status=` fields must not change).
- The CLI-via-daemon path intentionally logs server-side; this plan does not unify
  that, and shouldn't — note it so a future reader doesn't "finish the job"
  incorrectly.
- Deferred: if `request_id` turns out to be error-only, a follow-up could have the
  daemon echo a correlation id on success responses so every client line is
  joinable — out of scope here.
