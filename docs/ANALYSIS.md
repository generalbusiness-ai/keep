# Document Analysis

Analysis decomposes documents into meaningful parts — themes, episodes, commitments — each with its own summary and embedding. This makes your store searchable at a finer grain than whole documents.

## The problem with whole-document search

Semantic search matches your query against document summaries. This works well for focused notes, but struggles with long or multi-topic content:

- A meeting transcript covers auth, pricing, and deployment. Searching for "authentication" matches weakly because the summary mentions all three topics equally.
- A working session (`now`) accumulates days of context across several projects. Searching for one project returns the whole session with a mediocre similarity score.
- A PDF has 20 pages. The summary captures the gist but loses the specific argument on page 12.

The summary is a lossy compression. Analysis recovers what was lost.

## What analysis produces

`keep analyze` breaks content into **parts** — each a coherent unit of meaning with its own summary, tags, and embedding vector:

```
Document: "Meeting notes 2026-02-18"
  @P{1}  Authentication: team agreed on OAuth2 + PKCE for the mobile app
  @P{2}  Pricing: decided to keep free tier at 1000 requests/day
  @P{3}  Deployment: migrating to us-east-1 by end of month
```

Now searching for "authentication" matches `@P{1}` directly — high similarity, precise result. The other parts match their own topics independently.

## Two decomposition modes

Analysis auto-detects the content type:

**Structural decomposition** (documents, URIs): Splits by headings, topic shifts, and natural section boundaries. A PDF becomes chapters. An article becomes arguments. A spec becomes requirements.

**Episodic decomposition** (strings with version history): Assembles the full version history chronologically and splits by time, topic shifts, or narrative arcs. A working session becomes project episodes. A learning journal becomes distinct insights.

Both modes also extract:
- **Commitments**: promises, requests, declarations, and their status (open, fulfilled, withdrawn)
- **User facts**: concrete details stated by the user (dates, names, decisions)

## How parts improve search

Parts participate in search alongside regular documents. When you `keep find`, results may include both whole documents and individual parts:

```bash
keep find "OAuth2 mobile authentication"
# %a1b2c3d4@P{1}   2026-02-18  Authentication: team agreed on OAuth2 + PKCE...
# %e5f6g7h8         2026-02-10  Auth library comparison notes...
```

The part `@P{1}` scores higher than the whole meeting note would, because its embedding is focused on authentication specifically.

This matters most for:
- **Long documents** where topics are mixed
- **Working sessions** (`now`) that span multiple projects
- **Conversations** where the user made specific commitments or decisions
- **Reference material** (PDFs, articles) where you need to find a specific section

## When to analyze

Analysis is an LLM call per document — not free. Use it selectively:

- **Rich content**: meeting notes, working sessions, long articles, multi-topic documents
- **Reference material**: PDFs, specs, guides you'll search repeatedly
- **Conversations**: transcripts where commitments and decisions are buried in dialogue

Skip it for:
- **Short notes**: a one-line learning or a quick tag update
- **Already focused content**: a note about a single topic doesn't benefit from decomposition

Analysis runs in the background by default, queued alongside summarization. Use `--fg` to wait for results.

## Smart skip

Analysis tracks a content hash. If the document hasn't changed since the last analysis, `analyze` is a no-op. This makes it safe to run repeatedly — only new or changed content triggers an LLM call.

```bash
keep analyze doc:1                    # Analyzes, records hash
keep analyze doc:1                    # Skipped — content unchanged
keep put "updated content" --id doc:1 # Content changes
keep analyze doc:1                    # Re-analyzes
```

## Guidance tags

Pass tag keys with `-t` to guide the decomposition. This fetches your `.tag/KEY` descriptions and includes them in the LLM prompt, producing better part boundaries and more consistent tagging:

```bash
keep analyze doc:1 -t topic -t project
```

If you've defined `.tag/topic` with values like "auth", "pricing", "deployment", the analyzer will use those categories to structure its decomposition.

## Parts vs versions

These are complementary dimensions of the same document:

| &nbsp; | Versions (`@V{N}`) | Parts (`@P{N}`) |
|---|---|---|
| **Dimension** | Temporal | Structural |
| **Created by** | `put` (each update adds one) | `analyze` (replaces on full pass, appends on incremental) |
| **Accumulation** | Append-only chain | Full replacement, or append-only when incremental on a vstring |
| **Purpose** | How knowledge evolved | What knowledge contains |

A document can have both. A working session might have 30 versions (temporal) and 5 parts (thematic episodes extracted from the full history).

## See Also

- [KEEP-ANALYZE.md](KEEP-ANALYZE.md) — CLI reference for `keep analyze`
- [VERSIONING.md](VERSIONING.md) — Document versioning (the temporal dimension)
- [KEEP-FIND.md](KEEP-FIND.md) — Search results include parts
- [TAGGING.md](TAGGING.md) — Tags and guidance tag descriptions
