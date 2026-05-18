# keep analyze

Decompose a note or string into meaningful parts.

## Usage

```bash
keep analyze ID                       # Analyze using configured provider
keep analyze ID -t topic -t type      # With guidance tags
```

## What it does

`analyze` uses an LLM to decompose content into meaningful sections, each
with its own summary, tags, and embedding. This enables targeted search:
`find` matches specific sections, not just whole documents.

Two modes, auto-detected:
- **Documents** (URI sources): structural decomposition — chapters, topics,
  headings, thematic units
- **Strings** (inline notes with version history): episodic decomposition —
  the version history is assembled chronologically and decomposed into
  distinct phases, topic shifts, or narrative arcs

Parts are the structural counterpart to versions:
- **Versions** (`@V{N}`) are temporal — each `put` adds one
- **Parts** (`@P{N}`) are structural — `analyze` replaces all parts on URI
  sources and forced re-runs; on inline notes with version history it
  appends new parts incrementally (see [Re-analysis](#re-analysis))

## Options

| Option | Description |
|--------|-------------|
| `-t`, `--tag KEY` | Guidance tag keys (repeatable). Fetches `.tag/KEY` descriptions to guide decomposition |
| `--foreground`, `--fg` | Run in foreground and wait for results (default: background) |
| `--force` | Re-analyze even if parts are already current |

The global `-s/--store` option (available on every `keep` subcommand) overrides
the store directory.

## Background processing

By default, `analyze` runs in the background, serialized with other ML work
(summarization, embedding). Use `--fg` to wait for results:

```bash
keep analyze doc:1                    # Returns immediately, runs in background
keep analyze doc:1 --fg               # Waits for completion
```

Background tasks are processed by the same queue as `keep daemon` summaries.

## Part addressing

Append `@P{N}` to any ID to access a specific part:

```bash
keep get "doc:1@P{1}"           # Part 1
keep get "doc:1@P{3}"           # Part 3
```

Parts include prev/next navigation:
```yaml
---
id: doc:1@P{2}
tags:
  topic: "analysis"
prev:
  - @P{1}
next:
  - @P{3}
---
Detailed analysis of the main argument...
```

## Parts in get output

When a document has parts, `keep get` shows a parts manifest:

```yaml
---
id: doc:1
similar:
  - doc:2 (0.85) 2026-01-14 Related document...
parts:
  - @P{1} Introduction and overview of the topic
  - @P{2} Detailed analysis of the main argument
  - @P{3} Conclusions and future directions
prev:
  - @V{1} 2026-01-13 Previous summary...
---
Document summary here...
```

## Parts in search results

Parts have their own embeddings and appear naturally in `find` results:

```bash
keep find "main argument"
# doc:1@P{2}  2026-01-14 Detailed analysis of the main argument...
```

## Smart skip

Analysis is expensive (LLM call per document). To avoid redundant work,
`analyze` tracks a content hash at the time of analysis. If the document
hasn't changed since the last analysis, the call is skipped:

```bash
keep analyze doc:1                    # Analyzes, stores _analyzed_hash
keep analyze doc:1                    # Skipped — parts are current
keep put doc:1 "updated content"      # Content changes
keep analyze doc:1                    # Re-analyzes (content changed)
```

Analysis is queued automatically by the `after-write` flow whenever a note
is stored or updated, so a daily cron that re-runs `keep put /path/to/docs/`
will only re-analyze files whose content actually changed.

Use `--force` to override the skip:

```bash
keep analyze doc:1 --force            # Re-analyze regardless
```

## Part tags

Parts do **not** inherit tags from their parent document. Each part carries only:

- `_base_id` — the parent document's ID (for navigation and join queries)
- `_part_num` — the 1-based part number
- `_start_line` / `_end_line` — line ranges when the analyzer tracks source spans
- Any tags the analyzer assigned to that specific part (e.g., classifier output)

This keeps parts clean sub-notes rather than clones of the parent's tag graph, and prevents drift when the parent is re-tagged without re-analysis. Document-level relationships like `references`, `cites`, and `informs` stay on the parent where they semantically belong.

The tradeoff is recovered automatically by `find`: a tag-filtered query like `find("X", tags={"project": "alpha"})` still returns parts of matching parents via a `_base_id` join, so nothing is lost from the caller's perspective.

If the analyzer itself decides a particular part warrants a topic or type tag (via classifier output or guidance tags), that tag lives on the part. Everything else comes from the parent at read time, not write time.

## Part immutability

Parts are machine-generated analysis results, not human observations.
They are treated as derived data — immutable except for tag corrections.

**Allowed:**
- Read, search, list — parts appear in `get`, `find`, `list` normally
- Tag editing — correct or override analyzer tagging decisions:
  ```bash
  keep tag "doc:1@P{2}" -t topic=oauth2    # Fix a tag
  keep tag "doc:1@P{2}" -r topic            # Remove a tag
  ```
- Re-analyze — `analyze` replaces all parts atomically (or appends new
  parts in incremental mode for inline notes — see below)
- Delete parent — removing the parent document removes its parts

**Blocked:**
- `put` with a part ID — parts cannot be created or overwritten directly
- `del` on individual parts — use re-analyze or delete the parent
- `move` to a part ID — parts belong to their parent

If a part's summary is wrong, re-analyze (with `--force` or better
guidance tags). The right fix is a better prompt, not manual editing.

## Re-analysis

Behavior depends on the source type and on whether prior analysis is recorded:

- **URI sources, and first-time analyze on inline notes**: full decomposition.
  All existing parts are replaced atomically.
- **Inline notes with `_analyzed_version` already recorded**: incremental
  decomposition. A small overlap of already-analyzed versions is included
  as context, the new versions since the cursor are marked for analysis,
  and only the *new* parts are appended — prior parts stay intact so the
  analyzed trajectory accumulates over time rather than being rebuilt on
  every write.
- **`--force`**: skips the incremental path and rebuilds parts from scratch.

```bash
keep analyze doc:1                    # Creates parts (or appends if incremental)
keep analyze doc:1 -t topic --force   # Full re-analyze — replaces all parts
```

### Sliding-window semantic for vstrings

For inline notes that accumulate many versions (e.g. `now`), analyze is a
*sliding window* over the most-recent versions, not an exhaustive log
processor. When the analyze cursor falls more than ~100 versions behind
the latest (a "backlog", typically caused by a daemon outage), the
intermediate versions cannot be reconstructed from the analyze window;
analyze logs a WARNING, rebases to the newest window via a full pass,
and the cursor advances to the latest version. The gap is intentionally
dropped — older versions are historical noise for trajectory analysis.

## Guidance tags

Tag keys passed with `-t` fetch the corresponding `.tag/KEY` system documents
(e.g., `.tag/topic`, `.tag/type`). These descriptions tell the LLM what each
tag means and what values are appropriate, producing better decomposition and
more consistent tagging — even with smaller models.

```bash
keep analyze doc:1 -t topic -t type   # Guided by tag descriptions
```

## Python API

```python
kp = Keeper()

# Analyze (skips if parts are current)
parts = kp.analyze("doc:1")
parts = kp.analyze("doc:1", tags=["topic", "type"])
parts = kp.analyze("doc:1", force=True)  # Override skip

# Enqueue for background processing (returns False if skipped)
enqueued = kp.enqueue_analyze("doc:1")
enqueued = kp.enqueue_analyze("doc:1", force=True)

# Access parts
part = kp.get_part("doc:1", 1)        # Returns Item
parts = kp.list_parts("doc:1")        # Returns list[PartInfo]

# Edit tags on a part (the only allowed mutation)
kp.tag_part("doc:1", 1, tags={"topic": "oauth2"})  # Update tag
kp.tag_part("doc:1", 1, tags={"topic": ""})         # Remove tag
```

## Custom analysis prompts

The LLM prompt used for analysis is configurable. Create a `.prompt/analyze/*` document whose match rules target specific tags, and its `## Prompt` section replaces the default system prompt for matching documents. See [PROMPTS.md](PROMPTS.md) for details.

## See Also

- [VERSIONING.md](VERSIONING.md) — Versions (temporal) vs parts (structural)
- [KEEP-GET.md](KEEP-GET.md) — Retrieving items and parts
- [KEEP-FIND.md](KEEP-FIND.md) — Search results include parts
- [META-TAGS.md](META-TAGS.md) — Contextual queries (`.meta/*`)
- [PROMPTS.md](PROMPTS.md) — Prompts for summarization, analysis, and agent workflows
- [REFERENCE.md](REFERENCE.md) — Quick reference index
