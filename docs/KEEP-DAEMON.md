# keep daemon

Manage the background daemon and its work queues.

For directly running the daemon in the foreground, use `keepd --store PATH`.

## Usage

```bash
keep daemon                   # Start the daemon and process work
keep daemon --list            # Show queue status + active mirrors
keep daemon --reindex         # Enqueue all notes for re-embedding, then show progress
keep daemon --retry           # Reset failed items back to pending, then show progress
keep daemon --purge           # Delete all pending work items
keep daemon --stop            # Stop the daemon

keepd --store ~/.keep         # Direct foreground daemon runner
```

## What it does

`keep daemon` starts the background daemon and processes queued work:
embedding, summarization, analysis, OCR, link extraction, and edge
processing.

`keepd` is the smallest direct entrypoint when you want to run the daemon
explicitly in the foreground. `keep daemon` is the full CLI management
command.

The daemon also services:

- **File and directory watches** registered via `keep put --watch`
- **Markdown sync mirrors** registered via `keep data export --sync`
- **Timer-driven background work** such as replenishment and retry cycles

When there's no work and no active watches or mirrors, the daemon exits after an idle timeout.

## Flags

### `--list` / `-l`

Show the current state of the work queue without starting the daemon:

```bash
keep daemon --list
```

Output includes:
- Pending, processing, and failed item counts
- Active markdown mirrors with their status
- Whether the daemon is running

### `--reindex`

Enqueue every note in the store for re-embedding. Use this after changing embedding providers or models, or to rebuild the search index from scratch:

```bash
keep daemon --reindex
```

This queues work and then shows progress. You can also let an existing daemon
pick it up.

### `--retry`

Reset failed items back to pending so they'll be retried:

```bash
keep daemon --retry
```

Useful after fixing a transient issue (e.g., a provider was down, Ollama wasn't running) that caused items to fail.

### `--purge`

Delete all pending work items from the queue:

```bash
keep daemon --purge
```

Use with care — this discards queued work permanently. Items that were already processed are unaffected.

### `--stop`

Stop the background daemon:

```bash
keep daemon --stop
```

Sends SIGTERM and waits up to 10 seconds for graceful shutdown. If the daemon is stuck, it falls back to SIGKILL. Also cleans up stale discovery files (port, token, PID).

## Markdown mirrors

When markdown sync mirrors are registered (`keep data export --sync`), the
daemon services them automatically. `keep daemon` and `keep daemon --list`
report the number of active mirrors:

```
Markdown mirrors active: 1
```

Mirror state (last run, pending changes, errors) is visible via:

```bash
keep data export --list
```

See [keep data](KEEP-DATA.md) for full sync documentation.

## Common workflows

### After import

```bash
keep data import backup.json
keep daemon                     # Process embeddings for imported notes
```

### After changing embedding provider

```bash
keep config --setup             # Change provider
keep daemon --reindex           # Queue re-embedding for all notes
keep daemon                     # Process
```

### Troubleshooting failed items

```bash
keep daemon --list              # Check for failed items
keep daemon --retry             # Reset them to pending
keep daemon                     # Reprocess
```

### Troubleshooting daemon request errors

Unexpected daemon request failures are returned with a `request_id`. The same
ID is written to the daemon log line for the exception, so operators can grep
the ops log for the matching failure without exposing note content in the
client-facing error.

Storage tracing records operation names and low-cardinality metadata only; it
does not attach raw SQL or note content to trace attributes.

If the daemon is alive but requests are hanging, send `SIGUSR1` to capture all
Python thread stacks in the daemon log:

```bash
kill -USR1 "$(cat ~/.keep/processor.pid)"
```

For custom store paths, use that store's `processor.pid` file. The signal does
not stop the daemon. SQLite statements that run longer than the diagnostic
threshold are also logged with a sanitized query shape, fingerprint, parameter
count, and callsite. Adjust thresholds with `KEEP_SQLITE_SLOW_MS` and
`KEEP_SQLITE_PROGRESS_MS`.

### Restarting the daemon

```bash
keep daemon --stop
keep daemon                     # Fresh start
```
