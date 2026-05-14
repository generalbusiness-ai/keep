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

As long as at least one watch, mirror, or scheduled timer (e.g. supernode
replenish) is active, the daemon stays running and polls them on schedule.
With no pending work, no active watches or mirrors, and no due timers, the
daemon waits out the **idle-exit deadline** (10 minutes by default) and
then exits to release its resources. Subsequent `keep` commands auto-start
a fresh daemon on demand.

The deadline is configurable via the `KEEP_DAEMON_IDLE_SECONDS` environment
variable. Set it to `0` to disable the deadline entirely — the daemon will
then run until signalled, intended for users running the daemon under
launchd or systemd supervision (see [Running as a service](#running-as-a-service)).

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

## Running as a service

Once started, the daemon stays alive as long as there's a registered watch,
mirror, or pending work, or it's still within the idle-exit deadline — see
[What it does](#what-it-does) above for the exact conditions. What it
doesn't survive on its own is a crash, an OS-level `kill`, or a reboot:
there's no built-in supervisor. For interactive use that's fine — the next
`keep` command auto-starts a fresh daemon. For unattended workflows that
rely on watches firing or queues draining without anyone running `keep` in
the foreground, install the daemon as a user service so it comes back
automatically. Without supervision, a daemon that dies leaves watches
dormant and queued work silent; the only surface for the backlog is
`keep daemon --list`.

When running under a supervisor, set `KEEP_DAEMON_IDLE_SECONDS=0` so the
daemon doesn't intentionally idle-exit and force the supervisor to respawn
it. Both example units below set this.

### macOS (launchd)

Save the following to
`~/Library/LaunchAgents/ai.keepnotes.keep-daemon.plist`. Replace `USERNAME`
with your account, and replace `/Users/USERNAME/.local/bin/keep` with the
actual path from `which keep`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>ai.keepnotes.keep-daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/USERNAME/.local/bin/keep</string>
    <string>daemon</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/Users/USERNAME/.local/bin</string>
    <key>HOME</key><string>/Users/USERNAME</string>
    <key>KEEP_STORE_PATH</key><string>/Users/USERNAME/.keep</string>
    <key>KEEP_DAEMON_IDLE_SECONDS</key><string>0</string>
  </dict>
  <key>StandardOutPath</key><string>/Users/USERNAME/.keep/keep-daemon.log</string>
  <key>StandardErrorPath</key><string>/Users/USERNAME/.keep/keep-daemon.log</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
</dict>
</plist>
```

Load it:

```bash
launchctl load ~/Library/LaunchAgents/ai.keepnotes.keep-daemon.plist
```

Key fields:

- `RunAtLoad` starts the daemon at login.
- `KeepAlive` makes launchd respawn the daemon after a crash or `kill -9`.
- `ThrottleInterval` caps respawn frequency at once per 10 seconds, so a
  fundamentally broken daemon won't burn CPU in a tight crash loop.

Logs land in `~/.keep/keep-daemon.log` (`tail -F` to watch live).

To stop or remove:

```bash
launchctl unload ~/Library/LaunchAgents/ai.keepnotes.keep-daemon.plist
rm ~/Library/LaunchAgents/ai.keepnotes.keep-daemon.plist
```

### Linux (systemd user unit)

Save the following to `~/.config/systemd/user/keep-daemon.service`. Replace
`/home/USERNAME/.local/bin/keep` with the actual path from `which keep`:

```ini
[Unit]
Description=Keep daemon
After=default.target

[Service]
Type=simple
ExecStart=/home/USERNAME/.local/bin/keep daemon
Environment=KEEP_STORE_PATH=%h/.keep
Environment=KEEP_DAEMON_IDLE_SECONDS=0
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now keep-daemon.service
```

On headless boxes where you aren't logged in interactively, allow the user
manager to run without an active session:

```bash
loginctl enable-linger "$USER"
```

Logs are available via `journalctl --user -u keep-daemon`.

To stop or remove:

```bash
systemctl --user disable --now keep-daemon.service
rm ~/.config/systemd/user/keep-daemon.service
```

### Verifying supervision

After installing either service, confirm respawn works:

```bash
keep daemon --stop
sleep 15
keep daemon --list      # should not print "Starting daemon..."
```

If the daemon is still down after the wait, check the service log
(`~/.keep/keep-daemon.log` on macOS, `journalctl --user -u keep-daemon` on
Linux). Backlog that accumulated while the daemon was down can be reset
with `keep daemon --retry`.

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
