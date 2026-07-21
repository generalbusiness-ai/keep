# Reflective Memory — Agent Reference Card

**Purpose:** Persistent memory for documents with semantic search.

**Default store:** `~/.keep/` in user home (auto-created)

## Commands

| Command | Description | Docs |
|---------|-------------|------|
| `keep now` | Get or set current working intentions | [KEEP-NOW.md](KEEP-NOW.md) |
| `keep put` | Add or update a document | [KEEP-PUT.md](KEEP-PUT.md) |
| `keep get` | Retrieve note(s) by ID | [KEEP-GET.md](KEEP-GET.md) |
| `keep find` | Search by meaning or text | [KEEP-FIND.md](KEEP-FIND.md) |
| `keep list` | List recent notes, filter by tags | [KEEP-LIST.md](KEEP-LIST.md) |
| `keep config` | Show configuration and paths | [KEEP-CONFIG.md](KEEP-CONFIG.md) |
| `keep move` | Move versions into a named note | [KEEP-MOVE.md](KEEP-MOVE.md) |
| `keep analyze` | Decompose a note into structural parts | [KEEP-ANALYZE.md](KEEP-ANALYZE.md) |
| `keep prompt` | Render agent prompt with context | [KEEP-PROMPT.md](KEEP-PROMPT.md) |
| `keep mcp` | Start MCP stdio server for AI agents | [KEEP-MCP.md](KEEP-MCP.md) |
| `keep edit` | Edit content in $EDITOR | — |
| `keep del` | Remove note or revert to previous version | — |
| `keep tag` | Add, update, or remove tags | [TAGGING.md](TAGGING.md) |
| `keep flow` | Run a state-doc flow synchronously | [KEEP-FLOW.md](KEEP-FLOW.md) |
| `keep data export` | Export store to JSON (default) or markdown directory (`--format md`) | [KEEP-DATA.md](KEEP-DATA.md) |
| `keep data import` | Import documents from JSON export file | [KEEP-DATA.md](KEEP-DATA.md) |
| `keep daemon` | Run or manage the background daemon and its work queues | [KEEP-DAEMON.md](KEEP-DAEMON.md) |

## Global Flags

```bash
keep --json <cmd>        # Output as JSON (supported by most commands)
keep --ids <cmd>         # Output only IDs, one per line (for piping)
keep --full <cmd>        # Output full notes with context (overrides --ids)
keep -v <cmd>            # Enable debug logging to stderr
```

`--json`, `--ids`, and `--full` control output format globally. `--ids` is useful for shell composition (`keep find --ids foo | xargs keep get`). `--full` expands each result with frontmatter context. These are mutually exclusive; `--full` takes precedence over `--ids`.

## Output Formats

Two output formats, consistent across all commands:

### Default: Summary Lines
One line per note: `id date summary` (search results include score: `id (score) date summary`)
```
%a1b2c3d4         2026-01-14 URI detection should use proper scheme validation...
%e5f6a7b8         (0.89) 2026-01-14 OAuth2 token refresh pattern...
file:///path/doc  2026-01-15 Document about authentication patterns...
```

### With `--json`: JSON Output
```json
{"id": "...", "summary": "...", "tags": {...}, "score": 0.823}
```

Version numbers are **selectors**: @V{0} = current, @V{1} = previous, @V{2} = two versions ago, @V{-1} = oldest archived, @V{-2} = second-oldest.
Part numbers are **1-indexed**: @P{1} = first part, @P{2} = second part, etc.

**Output width:** Summaries are truncated to fit the terminal. When stdout is not a TTY (e.g., piped through hooks), output uses 200 columns for wider summaries.

### Pipe Composition

```bash
keep find "auth" --json | ...                        # Process search results
keep list -l 5 --json | ...                          # Process recent notes

# Version history composition
diff <(keep get doc:1) <(keep get "doc:1@V{1}")      # Diff current vs previous
```

## Quick CLI

```bash
# Current intentions
keep now                              # Show current intentions
keep now "What's important now"       # Update intentions
keep prompt reflect                   # Structured reflection practice
keep prompt reflect "auth flow"       # Reflect with search context
keep prompt query "what do I know about auth?"  # Answer from memory context
keep prompt conversation              # Conversation analysis
keep move "name"                      # Move versions from now into named note
keep move "name" --source "other"     # Move from a different source note

# Add or update
keep put "inline text" -t topic=auth  # Text mode
keep put file:///path/to/doc.pdf      # URI mode
keep put /path/to/folder/             # Directory mode
keep put /path/to/repo/ -r            # Recursive + git changelog
keep put /path/ -r --watch            # Watch for changes
keep put /path/ -r -x "*.log"        # Recursive, excluding pattern
keep put "note" -i my-id             # Specify note ID
keep put "update" --summary "short"  # Provide explicit summary
keep put /path/ -f                   # Force re-index
keep get .ignore                      # View global ignore patterns
keep edit .ignore                     # Edit ignore patterns in $EDITOR

# Retrieve and edit
keep get ID                           # Current version
keep get ID -V 1                      # Previous version
keep get "ID@P{1}"                    # Part 1 (from analyze)
keep get ID --history                 # List all versions
keep get ID --parts                   # List structural parts
keep get ID --similar                 # Show similar notes
keep get ID --meta                    # Show metadata
keep get ID -t project=foo            # Filter by tag
keep edit ID                          # Edit content in $EDITOR
keep edit .ignore                     # Edit system docs

# Search
keep find "query"                     # Semantic search
keep find "query" --deep              # Follow tags/edges to discover related notes
keep find "query" --since P7D         # Last 7 days
keep find "query" --until 2026-01-01  # Before a date
keep find "query" --scope 'file:///path/*'  # Constrain to ID glob
keep find "query" -t project=foo      # Filter by tag
keep find "query" -a                  # Include all (no limit)
keep find --id ID                     # Find by ID

# List and filter
keep list                            # Recent notes
keep list -t project=myapp           # Filter by tag
keep list --sort created              # Sort order
keep list --since P7D --until now    # Date range
keep list -a                         # Include system notes
keep list "file:///"                 # Filter by ID prefix

# Remove
keep del ID                          # Remove note or revert to previous version

# Analyze (skips if parts are already current)
keep analyze ID                      # Decompose into parts (background)
keep analyze ID -t topic -t type     # With guidance tags
keep analyze ID --fg                 # Wait for completion
keep analyze ID --force              # Re-analyze even if current

# Flow
keep flow "state" -t target          # Run a state doc for a target note
keep flow get -p item_id=now         # Pass named parameters
keep flow get -p item_id=now --token-budget 2000  # Limit rendered output
keep flow -f file.yaml               # Load flow from file
keep flow -b 10 -c cursor            # Budget and cursor control

# Data management
keep data export backup.json         # Export store to JSON
keep data export - | gzip > bk.gz    # Export to stdout, compress
keep data export ~/vault --format md # Export as markdown directory (one file per note)
keep data export ~/vault --sync      # Markdown export + continuous daemon mirror
keep data export ~/vault --sync --stop  # Stop mirroring
keep data export --list              # List active sync directories
keep data import backup.json         # Import (merge, skip existing)
keep data import backup.json -m replace  # Import (replace all)

# Maintenance
keep daemon                          # Start daemon, process tasks
keep daemon --list                   # Show queue status + active mirrors
keep daemon --reindex                # Enqueue all notes for re-embedding
keep daemon --retry                  # Reset failed items back to pending
keep daemon --purge                  # Delete all pending work items
keep daemon --stop                   # Stop daemon
keepd --store ~/.keep                # Direct foreground daemon runner

# Config
keep config                          # Show configuration
keep config "path"                   # Show specific config path
keep config --setup                  # Run interactive setup
keep config --reset-system-docs      # Reset system docs
keep config --state-diagram          # Show state diagram
```

## Python API

See [PYTHON-API.md](PYTHON-API.md) for complete Python API reference.

```python
from keep import Keeper
kp = Keeper()
kp.put("note", tags={"project": "myapp"})
results = kp.find("authentication", limit=5)
```

### LangChain / LangGraph

```python
from keep.langchain import KeepStore, KeepNotesToolkit, KeepNotesRetriever
```

See [LANGCHAIN-INTEGRATION.md](LANGCHAIN-INTEGRATION.md) for full details.

## When to Use
- `put` / `put(uri=...)` — when referencing any file/URL worth remembering
- `put` / `put("text")` — capture conversation insights, decisions, notes
- `find` — before searching filesystem; may already be indexed
- `find --since` — filter to recent notes when recency matters

## Topics

- [OUTPUT.md](OUTPUT.md) — How to read the frontmatter output
- [TAGGING.md](TAGGING.md) — Tags, speech acts, project/topic organization
- [VERSIONING.md](VERSIONING.md) — Document versioning and history
- [META-TAGS.md](META-TAGS.md) — Contextual queries (`.meta/*`)
- [PROMPTS.md](PROMPTS.md) — Prompts for summarization, analysis, and agent workflows
- [SYSTEM-TAGS.md](SYSTEM-TAGS.md) — Auto-managed system tags

## More

- [AGENT-GUIDE.md](AGENT-GUIDE.md) — Working session patterns
- [QUICKSTART.md](QUICKSTART.md) — Installation and setup
- [PYTHON-API.md](PYTHON-API.md) — Python API reference
- [LANGCHAIN-INTEGRATION.md](LANGCHAIN-INTEGRATION.md) — LangChain/LangGraph integration
- [ARCHITECTURE.md](ARCHITECTURE.md) — How it works under the hood
