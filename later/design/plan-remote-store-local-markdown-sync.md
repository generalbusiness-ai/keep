# Plan: Remote Store With Local Markdown Sync

Date: 2026-04-09
Status: In Progress

Implemented so far:

- Phase 1 foundation: remote backend config (formerly split into `remote_store`
  and `remote_task`) is now a single `[remote]` section in `keep.toml`; the
  CLI/MCP RemoteKeeper path and hosted task delegation share the same credentials
- Phase 3 export transport: remote full export works through `RemoteKeeper`
- Phase 3 markdown baseline slice: one-shot markdown export now works against a
  remote authoritative store through note-bundle export transport
- Phase 4 foundation: markdown mirror registration is now local runtime state
  in the config directory rather than the `.markdown-mirrors` note
- Phase 4 runtime slice: the local daemon now uses a remote authoritative
  source for markdown export and sync while keeping filesystem writes local
- Phase 5 coarse change-feed slice: remote continuous sync now prefers a
  cursor-based remote export change feed and falls back to interval-based full
  re-export when the remote source does not support it
- Phase 6 incremental slice: remote change-feed events now carry affected-note
  hints so the local mirror runtime can incrementally rewrite remote note
  bundles instead of always rebuilding the whole mirror
- Phase 2 transport slice: the daemon now supports explicit remote mode with a
  configurable bind host, advertised URL, and mode-aware `Host` allowlisting
- Phase 5 compatibility slice: daemon `ready`/`health` now publish explicit
  capability descriptors used by remote clients instead of inference by method
  presence
- Phase 2 security hardening: non-loopback binds now require explicit trusted
  proxy acknowledgment, wildcard binds require an advertised URL, and auth
  token comparison is constant-time

Still pending:

- versioned compatibility policy for staged rollout beyond the current
  capability flags

## Goal

Support this configuration cleanly:

- the authoritative keep store is remote
- this machine owns a local filesystem export root
- keep continuously mirrors remote notes to local markdown
- CLI and MCP can still talk to the remote store through a small local setup

No setup wizard is needed for this mode. It should be configured explicitly.

## Problem

The current codebase has most of the pieces for remote note access, but not for
remote-store plus local-markdown-sync as a coherent architecture.

There are two different concerns that are currently conflated:

- remote execution against the authoritative store
- local side effects on this machine's filesystem

The current implementation is close on the first concern and still local-only on
the second.

## Current State

### What already exists

- `RemoteKeeper` is a workable client for remote CRUD, context, health, and
  `run_flow`.
- the daemon already exposes a useful HTTP surface for local clients
- markdown export and daemon-owned mirror polling already exist
- the codebase already expects local-only integrations to sit outside the core
  semantic boundary

### What does not yet fit

#### 1. The daemon is explicitly loopback-only

The current daemon HTTP server is intentionally bound to `127.0.0.1` and
rejects non-loopback `Host` headers. That is correct for the current local
daemon, but it is not yet a remotely reachable service.

This is a transport gap, not the main architectural gap.

#### 2. Markdown export still assumes local store access

`keep data export --sync` still seeds the export by constructing a local
`Keeper` against the local store path. That means the current seed path mirrors
the local store, not a remote store.

#### 3. Markdown export is not implemented against the hosted boundary

The markdown export code reaches directly into local `Keeper` internals:

- `export_iter()`
- `_document_store`
- version and part tables
- dependency service for inverse edges

That is enough for a local daemon, but not for a local process consuming a
remote store through `RemoteKeeper`.

#### 4. Mirror state currently lives with the store

Mirror registrations are stored in `.markdown-mirrors` inside the keep store,
and mirror polling drains a store-local `sync_outbox`.

That is the wrong ownership model for distributed sync. The export root belongs
to this machine, not to the remote store. Mirror registration and local export
state should therefore live locally.

#### 5. Exposing the current markdown-export admin handler remotely would be wrong

The existing `/v1/admin/markdown-export` handler performs filesystem writes in
the daemon process. If the daemon runs on the remote host, it writes on the
remote host's filesystem. That does not satisfy the desired configuration.

## Design Principles

### 1. Separate store ownership from mirror ownership

The remote service owns:

- note storage
- version and part history
- flow execution
- change publication

The local machine owns:

- markdown export roots
- mirror registration
- local watch-path exclusivity
- local mirror cursors and operational state
- local filesystem writes

### 2. Keep the local daemon a local integration runtime

For this configuration, the local daemon should not pretend to be the
authoritative store. Its role is:

- local transport for CLI and MCP if desired
- local markdown mirror runtime
- local filesystem policy enforcement
- local polling / scheduling / backoff

### 3. Do not force markdown sync through `run_flow`

`run_flow` remains the stable semantic boundary for note operations.

Markdown sync is a local integration concern. It may consume:

- note APIs
- export APIs
- change-feed APIs

It does not need to be modeled as a public state-doc flow if a more explicit
transport contract is clearer.

### 4. Prefer an intentionally simple first distributed version

The first working distributed version should optimize for correctness and clear
ownership, not minimal bytes transferred.

That means:

- baseline full export first
- simple remote change publication
- full or coarse re-export before fine-grained incremental rewrites

Incremental remote bundle sync can come second.

## Target Architecture

### Components

#### Remote store daemon

Runs near the authoritative store and exposes HTTPS APIs for:

- note CRUD and context
- flow execution
- full export streaming
- note-bundle export
- remote change feed

It does not manage local markdown mirrors.

#### Local mirror daemon

Runs on this machine and exposes local CLI/MCP-facing functionality.

It owns:

- mirror registration and listing
- mirror state persistence
- polling the remote change feed
- fetching remote export data
- writing markdown files locally

It may also provide the existing local loopback HTTP layer for the CLI and MCP.

#### CLI / MCP

Remain thin wrappers.

They should talk to one local runtime when present, but the local runtime may
internally use a remote store client for store-backed operations.

## Required API Surface

### 1. Remote note API

This mostly already exists:

- `POST /v1/notes`
- `GET /v1/notes/{id}`
- `PATCH /v1/notes/{id}/tags`
- `DELETE /v1/notes/{id}`
- `POST /v1/search`
- `POST /v1/flow`
- `GET /v1/notes/{id}/context`
- `GET /v1/health`

### 2. Remote export snapshot API

Add an explicit export endpoint for full-store export streaming.

Example shape:

- `GET /v1/export?include_system=false`

Response shape:

- header first
- then one self-contained note dict per record
- versions and parts inline, matching `export_iter()`

Implementation note:

- the transport may be JSONL/NDJSON or chunked JSON
- the contract matters more than the encoding

### 3. Remote note-bundle export API

Add a way to fetch one note bundle with everything required for markdown
rendering.

This should include:

- current note
- parts if requested
- versions if requested
- edge-tag metadata needed for link rewriting
- inverse-edge data needed for frontmatter

Example shape:

- `GET /v1/export/bundles/{id}`

or

- `POST /v1/export/bundles`

This API is what makes remote incremental mirror updates possible without local
direct database access.

### 4. Remote change-feed API

Replace the current single-consumer local `sync_outbox` assumption with a
cursor-based remote feed.

Required properties:

- monotonic cursor or sequence
- non-destructive reads
- resumable polling
- suitable for more than one local consumer

Example shape:

- `GET /v1/changes?after=cursor&limit=1000`

Each event should identify enough to decide whether the mirror must:

- re-export one note bundle
- re-export dependent note bundles
- do a whole-mirror rebuild

### 5. Remote capabilities / version API

Add a small capability descriptor so the local mirror runtime can detect whether
the remote endpoint supports:

- export snapshot
- bundle export
- change feed
- incremental dependencies

This allows staged rollout without hidden fallback behavior.

## Local State Ownership

### Mirror registrations move local

The current `.markdown-mirrors` note should not remain the canonical registry
for distributed mirrors.

Instead, the local mirror daemon should persist local mirror registrations in a
local runtime store, for example under the local keep config/runtime
directory.

Mirror registration should record:

- remote endpoint identity
- project / namespace if relevant
- local root path
- include-system / include-parts / include-versions
- polling interval
- last successful cursor
- local operational errors

### Local mirror state stays local

The `.keep-sync/` directory inside the export root still makes sense for
on-disk mapping and operational debugging:

- `.keep-sync/map.tsv`
- `.keep-sync/state.json`

But the authoritative registration state should be local runtime state, not a
remote note in the authoritative store.

## Configuration Model

The earlier draft of this section proposed two distinct config blocks —
`[remote_store]` for authoritative routing and `[remote_task]` for delegated
background processing. That split has since been collapsed: in practice the
same hosted service handles both, the same credentials apply to both, and
keeping them separate produced footgun-y key duplication.

The current shipping shape is a **single** `[remote]` section. CLI/MCP route
through `RemoteKeeper` over HTTPS, and any locally-running Keeper (e.g. in
the daemon) reuses the same credentials to spin up the `TaskClient` for
hosted background tasks.

Example:

```toml
[remote]
api_url = "https://keep.example.com"
api_key = "..."
project = "my-project"
```

Equivalent environment overrides:

- `KEEPNOTES_API_URL`
- `KEEPNOTES_API_KEY`
- `KEEPNOTES_PROJECT`

Env vars override TOML field-by-field at load time; `save_config()` preserves
the on-disk `[remote]` section even when env vars are present so the file
keeps working after the env is unset.

The legacy `[remote_store]` and `[remote_task]` sections are no longer parsed:
`load_config()` fails with a clear error pointing users at `[remote]`.

### Local mirror runtime configuration

May remain implicit or lightly configurable:

- local runtime dir
- polling defaults
- local daemon bind settings

## Phased Plan

## Phase 1: Clarify roles and configuration

### Scope

- stop overloading `config.remote`
- make remote-store mode explicit
- classify markdown mirror management as local integration state

### Changes

- introduce separate config concepts for remote store and remote task
  delegation
- make CLI and daemon startup paths choose explicitly between:
  - local authoritative store
  - remote authoritative store with local runtime

### Acceptance criteria

- there is one explicit code path for “remote authoritative store”
- mirror registration no longer depends on a note stored in the authoritative
  store

## Phase 2: Make the remote daemon intentionally reachable

### Scope

- add explicit remote daemon mode instead of weakening the local default

### Changes

- configurable bind address
- configurable advertised base URL
- retain loopback-only default for the local daemon
- require TLS directly or assume a trusted reverse proxy in remote mode
- replace the current loopback-only host check with a mode-aware policy

### Acceptance criteria

- local daemon still defaults to loopback-only
- remote store daemon can be exposed intentionally and safely

## Phase 3: Add remote export snapshot transport

### Scope

- make full export work through `RemoteKeeper` or an adjacent remote export
  client

### Changes

- add remote export endpoint
- implement `RemoteKeeper.export_iter()` or a sibling export client
- switch markdown baseline export to consume the remote export contract when in
  remote-store mode

### Acceptance criteria

- one-shot markdown export works correctly against a remote authoritative store
- the CLI no longer seeds a remote mirror by constructing a local `Keeper`

## Phase 4: Move mirror registry and polling local

### Scope

- create a local mirror runtime that owns local roots and local scheduling

### Changes

- move mirror registration out of `.markdown-mirrors`
- localize mirror list / add / remove / status
- keep `.keep-sync/` export-root metadata for debugging and path mapping
- enforce local overlap checks against local watches and local mirror roots

### Acceptance criteria

- `keep data export --sync` registers a local mirror for a remote store
- `keep data export --list` reports local mirror state
- no remote filesystem writes are required

Status on 2026-04-09:

- done for local mirror ownership
- done for local-daemon runtime using `RemoteKeeper` as the export source

## Phase 5: Add coarse remote change feed

### Scope

- get distributed continuous sync working before optimizing it

### Initial behavior

On any relevant remote change:

- mark the mirror dirty
- run a full re-export after debounce

This is intentionally coarse but correct.

### Changes

- add cursor-based remote change feed
- local mirror runtime stores the last applied cursor
- local mirror runtime schedules whole-mirror refreshes

### Acceptance criteria

- remote note changes propagate to the local markdown mirror
- polling is resumable and non-destructive
- more than one local consumer can observe the same remote change stream

Status on 2026-04-09:

- done for coarse whole-mirror rebuild scheduling through `GET /v1/export/changes`
- local mirror registrations now persist the last observed remote cursor
- local runtime still falls back to interval-based full re-export when the
  remote source lacks the change-feed API
- capability negotiation now happens through `GET /v1/ready` / `GET /v1/health`

## Phase 6: Add incremental note-bundle sync

### Scope

- reduce work for ordinary updates after the full-rebuild version is stable

### Changes

- add note-bundle export endpoint
- add dependency-aware remote change planning
- re-export:
  - changed note bundles
  - notes whose inverse-edge frontmatter depends on the changed notes
- preserve whole-mirror rebuild for structural namespace changes

### Acceptance criteria

- ordinary updates rewrite only affected bundles
- inserts/deletes/path collisions still trigger full rebuild when necessary

Status on 2026-04-09:

- done for remote incremental bundle rewrites driven by `affected_note_ids`
  carried on change-feed events
- source-note updates now rewrite dependent inverse-edge target notes without
  local graph access

## Phase 7: Capability and remote-mode cleanup

### Status on 2026-04-09

- done for explicit remote daemon bind/adverised-url settings
- done for mode-aware host allowlisting in local vs remote daemon mode
- done for remote client capability detection through daemon `ready`

## Phase 7: Thin local daemon / CLI cleanup

### Scope

- align the local runtime with the rest of the architecture

### Changes

- keep CLI and MCP as thin wrappers over local runtime services
- remove remote-store semantic shortcuts from CLI code
- keep local-only operations explicit

### Acceptance criteria

- hosted-capable operations do not silently bypass the configured remote store
- local runtime code is clearly separated from remote store semantics

## File Plan

### [keep/config.py](/Users/hugh/play/keep/keep/config.py)

- split remote-store configuration from remote-task-delegation configuration

### [keep/remote.py](/Users/hugh/play/keep/keep/remote.py)

- remain the primary remote note/flow client
- add or coordinate export transport support

### New export transport module

Possible file:

- `keep/remote_export.py`

Responsibilities:

- remote full export streaming
- remote bundle fetch
- remote change-feed polling helpers

### [keep/daemon_server.py](/Users/hugh/play/keep/keep/daemon_server.py)

- add explicit remote-mode transport policy
- add remote export and change-feed endpoints
- keep local markdown-export admin behavior local-only

### [keep/daemon_client.py](/Users/hugh/play/keep/keep/daemon_client.py)

- keep local daemon discovery separate from remote-store API access
- do not overload local discovery files as remote configuration

### [keep/markdown_export.py](/Users/hugh/play/keep/keep/markdown_export.py)

- introduce a source interface that can render from remote export-shaped data
  instead of direct local `Keeper` internals only

### [keep/markdown_mirrors.py](/Users/hugh/play/keep/keep/markdown_mirrors.py)

- split into:
  - local mirror runtime state and scheduling
  - export planning / rewrite helpers
- remove dependence on store-local `.markdown-mirrors` as the registry for the
  distributed mode

### [keep/cli_app.py](/Users/hugh/play/keep/keep/cli_app.py)

- route `keep data export --sync` through the local mirror runtime
- stop seeding remote mirrors via local `Keeper(store_path=...)`

### [keep/console_support.py](/Users/hugh/play/keep/keep/console_support.py)

- make remote authoritative-store mode explicit
- stop conflating remote hosted note access with local-only commands

## Testing Plan

### Contract tests

Add one shared behavior matrix for:

- local in-process store
- local daemon over loopback
- remote store daemon

### Export tests

Add tests for:

- remote full export snapshot
- remote note-bundle export
- local markdown rendering from remote export-shaped data
- mirror rebuild after remote changes

### Ownership tests

Add tests proving:

- local mirror registration does not create remote `.markdown-mirrors` notes
- local mirror roots are enforced locally
- remote daemon never writes local export roots

### Failure and recovery tests

Add tests for:

- remote daemon unavailable during polling
- cursor resume after restart
- stale local mirror state
- full rebuild fallback after incremental planning failure

## Documentation Plan

Update after implementation:

- [docs/ARCHITECTURE.md](/Users/hugh/play/keep/docs/ARCHITECTURE.md)
- [docs/KEEP-DATA.md](/Users/hugh/play/keep/docs/KEEP-DATA.md)
- [docs/KEEP-PENDING.md](/Users/hugh/play/keep/docs/KEEP-PENDING.md)

The docs should explain the two supported ownership models clearly:

- local authoritative store with local markdown mirrors
- remote authoritative store with local markdown mirrors

## Non-Goals

- no setup wizard for this mode
- no bidirectional markdown editing in the first distributed version
- no query-scoped or filtered mirrors
- no attempt to make the remote daemon own local filesystem paths

## Main Risks

### 1. Over-optimizing too early

If the first distributed version tries to preserve today's incremental sync
mechanics exactly, implementation complexity will rise before ownership is
correct. Start with remote change detection plus full local rebuild.

### 2. Leaving mirror state remote

If mirror registrations remain stored in the authoritative store, the design
will continue to confuse “which notes exist” with “which machine mirrors them”.
That confusion will keep reappearing.

### 3. Smuggling local semantics into the remote daemon

If the remote daemon starts accumulating local-mirror policy, the architecture
will regress. The remote service should publish store state; the local runtime
should own local effects.

## Recommended First Cut

Implement in this order:

1. explicit remote-store configuration
2. remote export snapshot endpoint
3. local mirror registry and baseline export from remote data
4. coarse remote change feed with whole-mirror rebuild
5. incremental bundle sync only after the above is stable

That path is the smallest change that respects the architecture.
