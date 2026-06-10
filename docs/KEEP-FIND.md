# keep find

Find notes by unified semantic + full-text search.

## Usage

```bash
keep find "authentication"            # Unified semantic + full-text search
keep find --id ID                     # Find notes similar to an existing note
```

## Options

| Option | Description |
|--------|-------------|
| `--id ID` | Find notes similar to this ID (instead of text query) |
| `-t`, `--tag KEY=VALUE` | Filter by tag (repeatable, AND logic) |
| `-l`, `--limit N` | Maximum results (default 10) |
| `--since DURATION` | Only notes updated since (see time filtering below) |
| `--until DURATION` | Only notes updated before (see time filtering below) |
| `-D`, `--deep` | Follow tags/edges from results to discover related notes |
| `--deep-depth N` | How many primary results contribute tags/edges to `--deep` follow (default 10, max 100) |
| `-a`, `--all` | Include hidden system notes (IDs starting with `.`) |
| `-S`, `--scope GLOB` | Constrain results to IDs matching glob pattern |

## Unified search

Search is unified semantic + full-text: a query like "authentication" finds notes by meaning using embeddings *and* by matching words in summaries. Results from both strategies are merged and ranked.

## Similar-to-note search

Find notes similar to an existing document:

```bash
keep find --id file:///path/to/doc.md           # Similar to this document
keep find --id %a1b2c3d4                        # Similar to this note
keep find --id %a1b2c3d4 --since P30D           # Similar notes from last 30 days
```

## Tag filtering

Combine semantic search with tag filters (AND logic):

```bash
keep find "auth" -t project=myapp               # Search within a project
keep find "auth" -t project -t topic=security    # Multiple tags (AND)
```

Tag filters are applied as **pre-filters on the vector search** — results are the best matches *within* the filtered set, not global results filtered afterward. This makes tags suitable for data isolation (per-user, per-project, etc.). See [TAGGING.md](TAGGING.md#tag-based-isolation).

## Time filtering

The `--since` and `--until` options accept ISO 8601 durations or dates:

```bash
keep find "auth" --since P7D           # Last 7 days
keep find "auth" --since P1W           # Last week
keep find "auth" --since PT1H          # Last hour
keep find "auth" --since P1DT12H       # 1 day 12 hours
keep find "auth" --since 2026-01-15    # Since specific date
keep find "auth" --until 2026-02-01    # Before specific date
keep find "auth" --since P30D --until P7D  # Between 30 and 7 days ago
```

## Output format

Results are displayed as summary lines with similarity score and date:

```
%a1b2c3d4         (0.89) 2026-01-14 OAuth2 token refresh pattern...
%e5f6a7b8         (0.82) 2026-01-13 Token handling and session management...
```

Dates reflect when the note was created. Use `--full` for complete frontmatter with tags, similar notes, and version navigation.

## Scoped search

Constrain results to notes whose IDs match a glob pattern:

```bash
keep find "auth" --scope 'file:///Users/me/notes/*'       # Only files in notes/
keep find "auth" --scope 'file:///path/to/memory*'         # memory/ dir + MEMORY.md
keep find "auth" --scope '*myproject*'                      # IDs containing myproject
```

The search runs globally (traversing all notes for semantic matching), but only notes whose base ID matches the glob are returned. Deep search can follow edges through out-of-scope notes, but results are still scoped.

## Deep search

Deep search (`--deep`) follows tags and edges from the primary results to discover related notes that wouldn't appear in a normal search. Results are grouped under the primary note that led to them.

```bash
keep find "authentication" --deep            # Primary results + related notes
keep find "auth" --deep -l 5                  # Top 5 with deep groups
keep find "auth" --deep --deep-depth 25       # Widen the follow (default 10, max 100)
```

When edge tags are defined (see [EDGE-TAGS.md](EDGE-TAGS.md)), deep search follows edges to find related notes via graph traversal. Without edges, it falls back to tag-based discovery — finding notes that share tags with the primary results.

`--deep-depth` controls how many of the top primary results contribute their tags/edges to the follow step: a larger value surfaces more bridge notes at the cost of a wider search, a smaller value keeps results tight. It defaults to 10 and is capped at 100.

## See Also

- [KEEP-LIST.md](KEEP-LIST.md) — List and filter by tags
- [KEEP-GET.md](KEEP-GET.md) — Retrieve full note details
- [TAGGING.md](TAGGING.md) — Tag filtering patterns
- [REFERENCE.md](REFERENCE.md) — Quick reference index
