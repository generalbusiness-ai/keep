# Quick Start

This guide gets you from zero to semantic search in five minutes.

## Install

```bash
uv tool install keep-skill --upgrade
```

Then run `keep` — it walks you through first-time setup: choosing embedding and summarization providers, and installing hooks for your coding tools.

```bash
keep
```

If [Ollama](https://ollama.com/) is running, it's offered as the default — no API keys needed. Otherwise choose from any detected provider (OpenAI, OpenRouter, Voyage, Gemini, etc.) or use the [hosted service](https://keep.generalbusiness.ai):

```bash
export KEEPNOTES_API_KEY=kn_...
```

Re-run setup anytime with `keep config --setup`. See [Configuration](KEEP-CONFIG.md) for all provider options and advanced setup.

Provider choice in one line:

- Use `openai` for the direct OpenAI API.
- Use `openrouter` for OpenRouter's hosted routing layer.
- Use `openai` with `base_url` for local OpenAI-compatible servers such as `llama-server`, vLLM, or LM Studio.

## Store a note

Let's remember something about Kate.

```bash
keep put "Kate prefers aisle seats"
```

```
%2a792222e4b9 2026-02-22 Kate prefers aisle seats
```

That's it. The note is stored, embedded, and searchable. No schema, no configuration. The `%`-prefixed ID is content-addressed — same text always gets the same ID.

Now add a few more things you know about Kate:

```bash
keep put "Kate is allergic to shellfish — carries an EpiPen"
keep put "Kate mentioned she is training for the Boston Marathon in April"
keep put "Q3 budget review moved to Thursday 2pm"
keep put "Kate loves the window table at Osteria Francescana but hates waiting for reservations"
```

Five notes. Five facts. No relationships defined, no graph edges, no manual categorization. Just things you know.

## Search by meaning

Here's where it gets interesting. You stored "Kate prefers aisle seats." Now ask:

```bash
keep find "booking flights for Kate"
```

```
%2a792222e4b9 2026-02-22 Kate prefers aisle seats
%5b7756afdf72 2026-02-22 Kate mentioned she is training for the Boston Marathon in April
```

You never said "aisle seats" in your query. You said "booking flights." The system understood that seat preferences are relevant to flight bookings.

And it pulled in the marathon too — because if Kate's flying somewhere in April, you might want to know about Boston.

Try another:

```bash
keep find "planning a dinner for the team, Kate will be there"
```

```
%da0362b8bc19 2026-02-22 Kate loves the window table at Osteria Francescana but hates waiting for reservations
%118ffd9a16c9 2026-02-22 Kate is allergic to shellfish -- carries an EpiPen
```

The restaurant preference. The shellfish allergy. Neither contains the word "dinner" or "team" — but both are exactly what you need to not bore Kate or kill her.

The budget meeting? Not returned. Because it's not relevant. Semantic search doesn't match keywords — it matches *meaning*.

## Index a document

Notes are great for quick facts. But you also have documents — policies, specs, PDFs, web pages.

```bash
keep put https://docs.generalbusiness.ai/samples/travel-policy.pdf
```

```
https://docs.generalbusiness.ai/samples/travel-policy.pdf 2026-02-22 Corporate travel policy covering flight booking procedures, hotel guidelines, meal per-diems, and expense reporting requirements.
```

The URL becomes the note ID. The PDF is fetched, text-extracted, summarized, and embedded. Re-indexing the same URL creates a new version automatically.

Now search again:

```bash
keep find "booking flights for Kate"
```

```
%2a792222e4b9 2026-02-22 Kate prefers aisle seats
%5b7756afdf72 2026-02-22 Kate mentioned she is training for the Boston Marathon in April
https://...travel-policy.pdf@P{1} 2026-02-22 Flight booking: economy for domestic under 4hrs, business for international. Must book 14+ days in advance.
```

One search. Three results from completely different sources: Kate's personal preference, the relevant *section* of a 30-page PDF, and context about her April plans. Your agent now has everything it needs to book a flight.

## Index a directory

Index an entire folder — recursively, with excludes, and set up a watch so changes are re-indexed automatically:

```bash
keep put ./docs/ -r                        # Index all files recursively
keep put ./docs/ -r -x "*.log" -x "*.tmp"  # With excludes
keep put ./docs/ -r --watch                # Index + watch for changes
```

The daemon polls watched directories in the background. No per-turn cost — files are re-indexed only when they change.

## Scoped search

When you have a large store, scope searches to a subset of items:

```bash
keep find "auth" --scope 'file:///Users/me/docs/*'
```

The search runs semantically across everything, but only returns items whose ID matches the glob. Useful for searching within a project, a folder of notes, or a specific collection.

## Tags

Tags add structure without breaking search. Add them at creation time:

```bash
keep put "Kate approved the vendor contract — signed copy in DocuSign" \
  -t project=acme-deal -t person=kate
```

Or update them later:

```bash
keep tag %a7c3e1f4b902 -t status=done
```

Query by tag:

```bash
keep list --tag project=acme-deal
keep list --tag person=kate
keep list --tag project                # Any note with a 'project' tag
```

Tags are exact-match filters. Search is fuzzy. Use both: tag for structure, search for discovery.

## Current intentions

Track what you're working on right now:

```bash
keep now                               # Show current intentions
keep now "Working on the auth bug"     # Update intentions
```

`now` is a living document — it versions automatically every time you update it. Your AI tools can read it at session start to pick up where you left off.

```bash
keep now -V 1                          # What were you doing before?
keep now --history                     # All versions
```

## Version history

Every note retains history on update:

```bash
keep get ID                            # Current version (shows prev nav)
keep get ID -V 1                       # Previous version
keep get ID --history                  # List all versions
```

Text updates use content-addressed IDs — same text, same ID:

```bash
keep put "my note"                     # Creates ID from content hash
keep put "my note" -t done             # Same ID, new version (tag change)
keep put "different note"              # Different ID (new document)
```

## Reading the output

**Search results** (`keep find`) show one line per result — `id date summary`:

```
%2a792222e4b9 2026-02-22 Kate prefers aisle seats
%5b7756afdf72 2026-02-22 Kate mentioned she is training for the Boston Marathon in April
```

**Full output** (`keep get`, `keep now`) uses YAML frontmatter with the document body below:

```
---
id: %2a792222e4b9
tags:
  _created: "2026-02-22T16:00:00"
  _updated: "2026-02-22T16:00:00"
similar:
  - %5b7756afdf72 (0.51) 2026-02-22 Kate mentioned she is training for the Boston Marathon in April
  - %da0362b8bc19 (0.48) 2026-02-22 Kate loves the window table at Osteria Francescana...
meta/todo:
  - %f819a3c0e472 Send Kate the revised SOW by Wednesday
prev:
  - @V{1} 2026-02-22 Previous version summary...
---
Kate prefers aisle seats
```

Key fields:
- **`similar:`** — related items with similarity scores (0-1). Each ID can be passed to `keep get`
- **`meta/*:`** — contextual items surfaced by tag rules (open commitments, learnings, decisions)
- **`prev:`** / **`next:`** — version navigation. `@V{1}` means "one version back", usable with `-V 1`
- **`tags:`** — user tags and system tags (`_created`, `_updated`, `_source`, etc.)

Other output formats: `--json` for machine-readable JSON, `--ids` for bare IDs only.

## What just happened?

You stored five sentences about Kate and indexed a 30-page PDF. No schema, no relationships, no entity extraction.

Then you searched with a question you'd never planned for — "booking flights for Kate" — and got back Kate's seat preference, the flight booking section of the travel policy, and a heads-up about the Boston Marathon. Three different sources, one query, zero configuration.

That's semantic memory. Your agent accumulates knowledge — notes, documents, conversations — and retrieves it by *intent* rather than by lookup key. The more you store, the more connections it finds.

## Tool integrations

On first use, `keep` detects coding tools and installs hooks into their configuration:

| Tool | What happens |
|------|-------------|
| **Claude Code** | Plugin: `/plugin marketplace add https://github.com/generalbusiness-ai/keep.git` then `/plugin install keep@generalbusiness-ai` |
| **VS Code Copilot** | MCP: `export KEEP_STORE_PATH=...` then `code --add-mcp "{\"name\":\"keep\",\"command\":\"keep\",\"args\":[\"--store\",\"$KEEP_STORE_PATH\",\"mcp\"]}"` |
| **Kiro** | Practice prompt + agent hooks. MCP: `export KEEP_STORE_PATH=...` then `kiro-cli mcp add --name keep --scope global -- keep --store "$KEEP_STORE_PATH" mcp` |
| **OpenAI Codex** | Practice prompt in `AGENTS.md`. MCP: `export KEEP_STORE_PATH=...` then `codex mcp add keep -- keep --store "$KEEP_STORE_PATH" mcp` |
| **OpenClaw** | Practice prompt + [plugin](OPENCLAW-INTEGRATION.md) |

Hooks inject `keep now` context at key moments (session start, prompt submit) so the agent always has current intentions and relevant context.

Manual MCP setup should use `KEEP_STORE_PATH` as the source of truth and pass
that resolved value via `--store`; see [KEEP-MCP.md](KEEP-MCP.md) for the full
launcher patterns.

Run `keep config` to see integration status.

## Next steps

- [Configuration](KEEP-CONFIG.md) — Providers, models, environment variables, and advanced setup
- [REFERENCE.md](REFERENCE.md) — Complete CLI reference
- [AGENT-GUIDE.md](AGENT-GUIDE.md) — Working session patterns for AI agents
- [TAGGING.md](TAGGING.md) — Structured tags, speech acts, and constrained vocabularies
- [VERSIONING.md](VERSIONING.md) — Version history and content-addressed IDs
- [PYTHON-API.md](PYTHON-API.md) — Python API for embedding keep in applications
- [ARCHITECTURE.md](ARCHITECTURE.md) — System internals
- [SKILL.md](../SKILL.md) — The reflective practice

Everything here also works via [MCP](KEEP-MCP.md) and the [Python API](PYTHON-API.md) — same engine, same semantic search, accessible from any language or tool.
