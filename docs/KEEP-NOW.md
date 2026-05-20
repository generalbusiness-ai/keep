# keep now

Get or set the current working intentions.

The nowdoc is a singleton document that tracks what you're working on right now. It has full version history, so previous intentions are always accessible.

## Usage

```bash
keep now                              # Show current intentions
keep now "Working on auth flow"       # Update intentions
keep now "Auth work" -t project=myapp # Update with tag
echo "piped content" | keep now       # Set from stdin
```

## Options

| Option | Description |
|--------|-------------|
| `-t`, `--tag KEY=VALUE` | Set tag on the update (repeatable) |
| `--truncate` | Truncate content to `max_inline_length` instead of failing |
| `--scope NAME` | Write to a per-workstream version chain (doc id `now:NAME`) |
| `--auto-scope` | Derive `--scope` from the cwd (git `project/branch`, else cwd basename) |
| `-q`, `--quiet` | Suppress the trailing context dump on stdout |

## Per-workstream scope

When several agent sessions run in parallel — different worktrees, different
branches, different projects — they all clobber a single global `now`.
`--auto-scope` routes each session's writes to its own chain:

```bash
keep now "..." --auto-scope     # in ~/play/myproj on branch feature-x
                                # → writes to doc id `now:myproj/feature-x`
keep now --auto-scope           # shows that scoped chain
```

Slug derivation:

- In a git repo: `{project}/{branch}` where `project` is the basename of the
  main repo (worktrees inherit the main project's name).
- Detached HEAD: `{project}/{worktree-basename}`.
- Outside any git repo: cwd basename.

Pass `--scope NAME` to override the derived slug with an explicit name.

The plugin hooks shipped with keep use `--auto-scope` so each Claude Code or
Codex session gets its own chain automatically. Stop-event hooks add
`--quiet` so the formatted context dump doesn't leak onto stdout (which the
host parses as JSON).

## Version history

The nowdoc retains full history. Each update creates a new version. See [VERSIONING.md](VERSIONING.md) for details.

## Similar items and meta sections

When updating (`keep now "..."`), the output surfaces similar items and meta sections as occasions for reflection:

```yaml
---
id: now
tags:
  project: "myapp"
  topic: "auth"
similar:
  - %a1b2c3d4 OAuth2 token refresh pattern...
meta/todo:
  - %e5f6a7b8 validate redirect URIs
meta/learnings:
  - %c9d0e1f2 Token refresh needs clock sync
---
Working on auth flow
```

## keep move

When a string of work is complete, move the now history into a named note.

```bash
keep move "auth-string"                       # Move all of now's versions
keep move "auth-string" --source other-note   # Move from a different source
```

Moving to an existing name appends incrementally. The CLI itself only takes
`name` and `--source`; for tag-filtered moves or "only the current version",
drop down to the underlying flow:

```bash
keep flow move -p name=auth-string -p tags='{"project":"myapp"}'
keep flow move -p name=auth-string -p only_current=true
```

See [KEEP-MOVE.md](KEEP-MOVE.md) for details.

## keep prompt reflect

Deep structured reflection practice. Guides you through gathering context, examining actions, and updating intentions.

```bash
keep prompt reflect                       # Reflect on current work
keep prompt reflect "auth flow"           # Reflect with search context
keep prompt reflect --since P7D           # With time filter
```

See [KEEP-PROMPT.md](KEEP-PROMPT.md) for full prompt options including `--id`, `--tag`, `--since`, `--until`.

## See Also

- [KEEP-PROMPT.md](KEEP-PROMPT.md) — Agent prompts with context injection
- [KEEP-MOVE.md](KEEP-MOVE.md) — Move now history into named items
- [TAGGING.md](TAGGING.md) — Tag system and speech-act tracking
- [VERSIONING.md](VERSIONING.md) — Version history and navigation
- [META-TAGS.md](META-TAGS.md) — Contextual queries (`.meta/*`)
- [REFERENCE.md](REFERENCE.md) — Quick reference index
