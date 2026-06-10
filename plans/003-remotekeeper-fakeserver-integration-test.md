# Plan 003: Add an in-process fake-server integration test for the remote/hosted path

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat e3bb33d..HEAD -- keep/remote.py keep/flow_client.py tests/test_remote.py`
> If any of these changed since this plan was written, compare the "Current
> state" excerpts against the live code before proceeding; on a mismatch, treat
> it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `e3bb33d`, 2026-06-10

## Why this matters

The hosted/remote backend (`RemoteKeeper`, used when `[remote]` is configured or
`KEEPNOTES_API_KEY` is set) became a first-class feature over v0.154–v0.157 and
is the single most-churned area of the codebase right now. Yet the only default-
suite coverage (`tests/test_remote.py`) **mocks `keeper._client`** — it never
exercises a real HTTP round-trip. The only end-to-end coverage
(`tests/test_smoke_remote.py`) is opt-in and requires a live API key against
keepnotes.ai, so it does not run in CI. The result: a change to how `RemoteKeeper`
builds requests, parses responses, or maps CRUD onto `/v1/flow` can break every
hosted-mode user and pass the entire default suite green. AGENTS.md is explicit
that "features require tests that exercise the feature in full, and that will
fail if the user-visible behavior breaks." This plan adds an in-process fake HTTP
server that speaks the daemon's contract, so the remote path is tested for real
without any network or credentials.

## Current state

- `keep/remote.py` — `RemoteKeeper` is an httpx-backed flow-host client. Key facts
  the test depends on:
  - Constructor: `RemoteKeeper(api_url, api_key, config, *, project=None)`. It
    builds `self._client = httpx.Client(base_url=self.api_url, headers=..., timeout=...)`.
    `api_url` is validated by `validate_remote_api_url()` — **`http://127.0.0.1:<port>`
    and `http://localhost:<port>` are accepted** (loopback is allowed; that is how
    the existing tests at `tests/test_remote.py:13` already construct it against
    `http://localhost:9999`).
  - CRUD methods route through the flow boundary. `get`, `put`, `find`, `delete`,
    `tag` all call `flow_*` helpers in `keep/flow_client.py`, which call
    `host.run_flow(...)`. For `RemoteKeeper`, `run_flow` does
    `self._post("/v1/flow", json={...})` and wraps the response in a `FlowResult`:

```python
    def run_flow(self, state, *, params=None, budget=None, cursor_token=None,
                 state_doc_yaml=None, writable=True):
        from .state_doc_runtime import FlowResult
        resp = self._post("/v1/flow", json={
            "state": state, "params": params, "budget": budget,
            "cursor_token": cursor_token, "state_doc_yaml": state_doc_yaml,
            "writable": writable,
        })
        return FlowResult(
            status=resp.get("status", "error"),
            bindings=resp.get("bindings", {}),
            data=resp.get("data"),
            ticks=resp.get("ticks", 0),
            history=resp.get("history", []),
            cursor=resp.get("cursor"),
        )
```

  - `server_info()` does `self._client.get("/v1/ready")` and returns the JSON dict;
    `capabilities()`/`supports_capability()` read `info["capabilities"]`.
  - `_get`/`_post` log one INFO line per call and then call `_raise_for_status(resp)`
    before returning `resp.json()`.
  - `export_iter()` streams `GET /v1/export` (ndjson or a single JSON dict);
    `export_bundle()` does `GET /v1/export/bundles/{id}` (404 → None);
    `export_changes()` does `GET /v1/export/changes`.

- `keep/flow_client.py` — defines `get_item`, `put_item`, `find_items`,
  `delete_item`, `tag_item`. **You must read this file** to learn exactly what
  `state` string and `params` each helper sends to `run_flow`, and how it reads
  the returned `FlowResult` (e.g. which `bindings`/`data` keys it expects to build
  an `Item`). The fake server's `/v1/flow` handler must return a `FlowResult`-shaped
  body whose `bindings`/`data` satisfy those readers. Do not guess the shape —
  derive it from `flow_client.py` and from how the real daemon responds (see the
  daemon's `/v1/flow` handler in `keep/daemon_server.py` for the authoritative
  response shape).

- `tests/test_remote.py` — current tests (read the whole file). They construct
  `RemoteKeeper("http://localhost:9999", ...)` and either test `_raise_for_status`
  directly or replace `keeper._client` with a `MagicMock`. Example of the mock
  style (the gap this plan closes):

```python
    keeper._client = MagicMock()
    keeper._client.get.return_value = mock_response
    keeper._client.post.return_value = mock_response
```

- `tests/conftest.py` — owns per-test store isolation and **daemon cleanup**
  (`_isolate_test_store_and_cleanup_daemons`, autouse). It also has helpers like
  `_write_test_store_config(store)`. Any server you start in a test must be shut
  down in the same test (a `finally` or fixture teardown) — a leaked thread/socket
  is a test bug per AGENTS.md.

## Commands you will need

| Purpose            | Command                                          | Expected on success     |
|--------------------|--------------------------------------------------|-------------------------|
| Targeted tests     | `python -m pytest tests/test_remote.py -q`       | all pass                |
| New file only      | `python -m pytest tests/test_remote_fakeserver.py -q` | all pass (N new tests) |
| Full suite         | `python -m pytest tests/ -x -q`                  | all pass, no daemon left|
| Lint               | `ruff check tests/test_remote_fakeserver.py`     | exit 0                  |
| No leaked threads  | (see Step 4 — assert server thread joined)       | —                       |

## Suggested executor toolkit

- Read `keep/flow_client.py` and the `/v1/flow` handler in
  `keep/daemon_server.py` in full before writing the fake server — they are the
  authoritative source for the request/response contract you must emulate.

## Scope

**In scope** (the only files you may create/modify):
- `tests/test_remote_fakeserver.py` (create) — the new integration test module.
- Optionally `tests/conftest.py` — ONLY to add a reusable `fake_remote_server`
  fixture if you decide the server boilerplate belongs there. If you keep the
  server local to the new test file, do not touch conftest.

**Out of scope** (do NOT touch):
- `keep/remote.py`, `keep/flow_client.py`, `keep/daemon_server.py` — this is a
  test-only plan. If a real round-trip reveals a *bug* in these, STOP and report
  it; do not fix production code under this plan.
- The existing tests in `tests/test_remote.py` — leave them; the new file is
  additive. (You may later migrate the mock-based tests, but that is out of scope
  here.)
- `tests/test_smoke_remote.py` — the live-service tests stay opt-in.

## Git workflow

- Branch: `advisor/003-remote-fakeserver-test`
- One commit: `test: integration-test RemoteKeeper against an in-process fake server`.
- Do NOT push or open a PR unless the operator instructs it.

## Steps

### Step 1: Build a minimal threaded fake daemon server

In `tests/test_remote_fakeserver.py`, implement a fake server using the stdlib
`http.server.ThreadingHTTPServer` (or `socketserver.TCPServer`) bound to an
**ephemeral port** (`("127.0.0.1", 0)`, then read the actual port from
`server.server_address[1]`). Run it in a daemon thread. The handler must:

- Record every received request (method, path, parsed JSON body, query params) in
  a list the test can assert against.
- Route by `(method, path)`:
  - `GET /v1/ready` → 200 with a JSON dict including a `capabilities` object
    (mirror the real daemon's `/v1/ready` shape — read it from
    `keep/daemon_server.py`).
  - `POST /v1/flow` → 200 with a `FlowResult`-shaped body (`status`, `bindings`,
    `data`, `ticks`, `history`, `cursor`) whose contents are chosen so the
    `flow_client.py` helper for the operation under test can build the expected
    `Item`. The handler may branch on the request body's `state` field
    (e.g. `"get"`, `"put"`, `"find"`, `"delete"`, `"tag"`) to return an
    operation-appropriate payload.
  - `GET /v1/export` → 200 ndjson (set `Content-Type: application/x-ndjson`) with
    a couple of rows, to cover `export_iter`.
  - Anything else → 404.

Provide start/stop helpers (or a fixture) that yield the base URL
`http://127.0.0.1:<port>` and guarantee `server.shutdown()` + thread `join()` in
teardown.

**Verify**: `python -m pytest tests/test_remote_fakeserver.py -q` collecting at
least one trivially-passing smoke test that starts and stops the server →
exit 0, and the process exits (no hang).

### Step 2: Test the CRUD round-trips through the real httpx client

Add tests that construct a real `RemoteKeeper(base_url, "kn_test", StoreConfig(path=tmp_path, config_dir=tmp_path))`
pointed at the fake server (do **not** mock `_client`), and exercise:

- `put(content=..., id=..., tags=...)` → assert the server received
  `POST /v1/flow` with `state` matching what `flow_client.put_item` sends, and
  that the returned object is an `Item` with the id/summary/tags from the canned
  response.
- `get(id)` → returns an `Item` (or `None` for a not-found canned response).
- `find(query=..., limit=...)` → returns a `list[Item]`; assert the query/limit
  reached the server in the flow params.
- `delete(id)` → returns the expected bool.
- `tag(id, tags=...)` → returns the updated `Item`.
- `server_info()` / `capabilities()` → reads `/v1/ready` and exposes capabilities.

Derive each expected request/response shape from `keep/flow_client.py` and the
daemon handler — **if your canned response can't satisfy the helper's reader, that
tells you the real contract; encode it, don't work around it.**

**Verify**: `python -m pytest tests/test_remote_fakeserver.py -q` → all CRUD tests
pass.

### Step 3: Test error propagation and the client-log audit trail

- Make the fake server return a `500` with a JSON body containing `request_id`
  for one operation, and assert `RemoteKeeper` raises `httpx.HTTPStatusError`
  whose message contains the `request_id` (this mirrors the existing
  `test_remote_http_error_includes_daemon_request_id` but now through the real
  client + `_post`/`_get` path).
- Assert that after the round-trips, `{config_dir}/keep-client.log` exists and
  contains `remote: POST /v1/flow ...` and `host=http://127.0.0.1:<port>` lines
  (the `_log_call` audit trail — see the existing
  `test_remote_attaches_client_log_to_config_dir`).

**Verify**: `python -m pytest tests/test_remote_fakeserver.py -q` → these pass.

### Step 4: Prove no resource leak

Add an explicit teardown assertion (or rely on the fixture) that the server
thread is joined and the socket closed. Confirm the full suite leaves no daemon
or thread behind.

**Verify**: `python -m pytest tests/ -x -q` → all pass; the run terminates
promptly (no hang waiting on a live thread), and conftest's daemon-cleanup
reports nothing leaked.

## Test plan

- New file `tests/test_remote_fakeserver.py` with, at minimum:
  - one server smoke test (start/stop),
  - five CRUD round-trip tests (put, get, find, delete, tag),
  - one `server_info`/`capabilities` test,
  - one 500-with-request_id error-propagation test,
  - one client-log audit-trail test.
- Structural pattern to follow: the request-recording + canned-response style of
  the daemon handler, and the assertion style already in `tests/test_remote.py`.
- Verification: `python -m pytest tests/test_remote_fakeserver.py -q` → all new
  tests pass; `python -m pytest tests/ -x -q` → full suite green.

## Done criteria

ALL must hold:

- [ ] `tests/test_remote_fakeserver.py` exists and constructs a **real**
      `RemoteKeeper` against an in-process server (no `MagicMock` on `_client`).
- [ ] `python -m pytest tests/test_remote_fakeserver.py -q` passes with ≥9 tests.
- [ ] `python -m pytest tests/ -x -q` passes, terminates promptly, leaves no
      daemon/thread running.
- [ ] `ruff check tests/test_remote_fakeserver.py` exits 0.
- [ ] `git status` shows only `tests/test_remote_fakeserver.py` (and optionally
      `tests/conftest.py`) added/modified — no production code changed.
- [ ] `plans/README.md` status row for 003 updated.

## STOP conditions

Stop and report back (do not improvise) if:

- The excerpts in "Current state" don't match the live code (drift).
- Building a faithful canned `/v1/flow` response reveals that `RemoteKeeper` or
  a `flow_client` helper mishandles a real response shape (i.e. you find a
  production bug) — report it; do not patch production code here.
- A test hangs (server thread not shutting down) and you cannot make teardown
  deterministic after one fix attempt.
- `validate_remote_api_url` rejects your loopback base URL — re-check you used
  `http://127.0.0.1:<port>`/`http://localhost:<port>` exactly; if it still
  rejects, STOP (the existing tests prove loopback is allowed, so a rejection
  means drift).

## Maintenance notes

- For the reviewer: confirm the fake server's `/v1/flow` response shape is derived
  from the real daemon handler, not invented — otherwise the test passes against a
  fiction and the coverage is illusory. Cross-check the canned `bindings`/`data`
  keys against `keep/daemon_server.py`'s `/v1/flow` handler.
- When the daemon's flow response contract changes, this test must change with it
  — that is the point: it will fail loudly, which is what the smoke-only coverage
  could not do in CI.
- A natural follow-up (separate plan) is to migrate the mock-based tests in
  `tests/test_remote.py` onto this fixture and retire the `MagicMock(_client)`
  pattern.
