# OpenClaw Integration

## Why

OpenClaw writes memory files (markdown) which are a solid foundation for
long-term memory. But flat files aren't enough to reliably recall what's
happening, what needs follow-up, or to build a continuous improvement loop.

[Keep](https://github.com/generalbusiness-ai/keep) integrates with OpenClaw as a
**context engine** — it owns how context is captured, assembled, and enriched
across the agent's lifecycle.

What this gives you:

1. **Semantic context assembly** — every agent turn, keep surfaces relevant
   memories: similar items, open commitments, learnings, edge relationships.
   The agent starts each turn knowing what matters, not just what's recent.
2. **Continuous capture** — user and assistant messages are ingested as
   versioned items after each turn. Background processing (summarize, analyze,
   tag, link extraction) runs automatically.
3. **Inflection-aware reflection** — after each turn, keep detects topic shifts,
   commitments, and significant moments. When it finds one, it triggers the
   reflective practice in the background.
4. **Memory indexing** — daemon-driven watches keep workspace files, memory
   markdown, and git history continuously indexed. No manual steps needed.
5. **Session persistence** — each session is a versioned item keyed by its
   routing identity. Turns accumulate as versions. No archival step needed.
6. **Memory tools** — `memory_search` and `memory_get` provide semantic
   recall over memory files, replacing OpenClaw's built-in flat-file search.

---

## Quick Start

### 1. Install the plugin

```bash
# From ClawHub (recommended):
openclaw plugins install clawhub:keep

# Or from a local path (for development):
openclaw plugins install -l $(keep config openclaw-plugin)
```

The plugin auto-configures `plugins.slots.contextEngine: "keep"` on install.

### 2. Agent-guided setup

If the Python runtime (`keep` CLI) is not yet installed, the plugin registers
a bootstrap context engine that instructs the agent to guide you through setup.
On your first conversation after installing the plugin, the agent will walk
you through:

1. **Installing keep** — `uv pip install keep-skill[local]`
2. **Configuring providers** — `openclaw keep setup` (interactive wizard)
3. **Restarting** — `openclaw gateway restart`

You can also do this manually:

```bash
uv tool install keep-skill                    # API providers included
# Or: uv tool install 'keep-skill[local]'    # Local models, no API keys needed
keep config --setup                           # Interactive provider wizard
openclaw gateway restart
```

### 3. Verify

```bash
openclaw plugins list                         # Should show: keep (loaded)
openclaw keep doctor                          # Check keep store health
```

---

## How It Works

### Context Engine (automatic)

When the plugin is installed, OpenClaw activates keep as the context engine
(`plugins.slots.contextEngine: keep`). This means keep participates in every
stage of the agent lifecycle:

| Lifecycle method | What keep does |
|-----------------|----------------|
| **bootstrap** | Mark session for first-assemble enrichment |
| **assemble** | Render the `openclaw-assemble` prompt template — retrieves the requested note plus similar items, meta, edges, and session history via the `.state/get` state doc (with openclaw fragment) |
| **afterTurn** | Ingest each new user/assistant message as a version of the session item; detect inflection points (topic shifts, commitments, substantial work); trigger background reflection; set up workspace watches |
| **compact** | Advisory only — logs diagnostics (OpenClaw manages its own compaction) |
| **prepareSubagentSpawn** | Link child session to parent via tags, write spawn marker as first version of child item |
| **onSubagentEnded** | Clean up tracking state (child session item persists) |

The agent doesn't need to do anything — context flows in automatically.

### Practice Layer (agent-initiated)

The agent also has voluntary access to keep via MCP tools:

```
keep_flow(state="get", params={item_id: "now"})          # Current intentions
keep_flow(state="query-resolve", params={query: "topic"})        # Semantic search
keep_flow(state="put", params={content: "insight", tags: {type: "learning"}})  # Capture
keep_prompt(name="reflect")                                       # Full reflection
keep_help(topic="flow-actions")                                   # Documentation
```

These are injected into the agent's system prompt as cacheable static
instructions (`appendSystemContext`), so they don't add per-turn token cost.

### Memory Indexing (automatic)

The plugin sets up daemon-driven watches on the workspace directory (including
`memory/` and `MEMORY.md`) on the first agent turn. The daemon polls for
changes automatically — no manual indexing needed. Git repositories in the
workspace are also discovered and their commit history is ingested incrementally.

### Memory Tools (`memory_search` / `memory_get`)

The plugin registers `memory_search` and `memory_get` as agent tools,
providing the same interface as OpenClaw's built-in `memory-core` plugin.
OpenClaw's system prompt instructions ("run memory_search before answering
questions about prior work...") work automatically.

**`memory_search(query, maxResults?, minScore?)`** — Searches `MEMORY.md`
and `memory/*.md` using keep's semantic search with scope-constrained find.
Returns results in memory-core's format: `path`, `startLine`, `endLine`,
`score`, `snippet`, `source`. Line positions come from keep's part analysis
and keyword fallback.

**`memory_get(path, from?, lines?)`** — Safe file read constrained to
`MEMORY.md` and `memory/*.md` paths. Supports `from` (1-indexed start line)
and `lines` (count) for slicing. Path validation prevents traversal outside
memory paths.

---

## Slot Configuration

The plugin declares `kind: "context-engine"` in its manifest. On install,
OpenClaw automatically sets `plugins.slots.contextEngine: "keep"`.

For `memory_search`/`memory_get`, the plugin registers these tools directly
at runtime. To avoid conflicts with the built-in `memory-core` plugin, disable
it:

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "keep",
      "memory": "none"
    }
  }
}
```

If you prefer to keep memory-core active (e.g., for its QMD backend), the
plugin's `memory_search` registration will take precedence when keep is loaded.

---

## Customizing Context Assembly

The context assembly is driven by a keep prompt template
(`prompt-agent-openclaw-assemble`) backed by a state doc (`state-get/openclaw`).
Edit them to change what context the agent sees:

```bash
keep get .prompt/agent/openclaw-assemble       # View prompt template
keep get .state/get/openclaw                   # View state doc (retrieval queries)
keep config --reset-system-docs                # Restore defaults
```

The default runs parallel queries for:

1. **intentions** — current `now` content (cross-session intentions)
2. **similar** — semantically similar items to the current user prompt
3. **meta** — resolved meta-docs (learnings, todos, open commitments)
4. **edges** — edge relationships from `now` (references, speakers, threads)
5. **session** — the current session item (versioned turn history)

All results are token-budgeted so they fit within the context window.

---

## Plugin Configuration

Configure in OpenClaw's config under `plugins.entries.keep.config`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `contextBudgetRatio` | number | 0.3 | Fraction of token budget for keep context (0.0–1.0) |
| `captureHeartbeats` | boolean | false | Whether to capture heartbeat messages |

Example:

```json
{
  "plugins": {
    "entries": {
      "keep": {
        "enabled": true,
        "config": {
          "contextBudgetRatio": 0.25
        }
      }
    }
  }
}
```

---

## CLI Commands

The plugin registers CLI commands under `openclaw keep`:

| Command | Description |
|---------|-------------|
| `openclaw keep setup` | Run keep's interactive setup wizard (providers, models) |
| `openclaw keep doctor` | Check keep store health and diagnose issues |

---

## Provider Configuration

keep auto-detects AI providers from environment variables:

```bash
export OPENAI_API_KEY=...      # Handles both embeddings + summarization
export OPENROUTER_API_KEY=...  # Also handles both, via OpenRouter
# Or: GEMINI_API_KEY=...       # Also does both
# Or: VOYAGE_API_KEY=... and ANTHROPIC_API_KEY=...  # Separate services
```

If Ollama is running locally, it's auto-detected with no configuration needed.

Use `openai` for direct OpenAI, `openrouter` for OpenRouter, and `openai` with
`base_url` for local OpenAI-compatible servers such as `llama-server`, vLLM, or
LM Studio.

For explicit setup: `openclaw keep setup` or `keep config --setup`

For local-only operation (no API keys): `uv tool install 'keep-skill[local]'`

See [QUICKSTART.md](QUICKSTART.md) for full provider options and troubleshooting.

---

## Recommended: Daily Reflection Cron

For automatic deep reflection:

```bash
openclaw cron add \
  --name keep-reflect \
  --cron "0 21 * * *" --tz "America/New_York" \
  --session isolated \
  --agent-turn "Daily reflection. Run \`keep prompt reflect\` and follow the practice.

This is the deeper review — not just what happened, but what it means."
```

This runs nightly in an isolated session. Memory files (workspace `memory/`
directory, `MEMORY.md`, etc.) are indexed automatically by the plugin's
workspace watch — no manual `keep put` needed.

---

## Integration Layers

| Layer | Trigger | What it does | Latency |
|-------|---------|-------------|---------|
| **Context engine** | Every agent turn | Assemble context (assemble), capture messages (afterTurn), detect inflections | ~10-50ms |
| **Memory tools** | Agent-initiated | `memory_search` / `memory_get` over workspace memory files | ~50-200ms |
| **Workspace watches** | Daemon-driven | Index files, memory, git history automatically | Background |
| **Daily reflection** | Cron (optional) | Deep practice reflection | Isolated session |
| **Agent practice** | Agent-initiated | Voluntary reflection, search, capture | On demand |

---

## Architecture

```
OpenClaw Gateway (Node.js)
  └── keep plugin (TypeScript, in-process)
        ├── Context Engine
        │     ├── bootstrap (session init)
        │     ├── assemble (prompt template + state doc queries)
        │     ├── afterTurn (message ingest + inflection detection)
        │     ├── compact (advisory)
        │     └── prepareSubagentSpawn, onSubagentEnded
        ├── Bootstrap CE (when keep CLI is missing or unconfigured)
        │     └── assemble → injects setup instructions for agent
        └── MCP Client (stdio transport, global singleton)
              └── keep mcp (Python, persistent process)
                    ├── keep store (SQLite + ChromaDB)
                    └── daemon (watches, git ingest, background work)
```

The MCP transport is a `Symbol.for` global singleton that survives gateway
soft restarts (SIGUSR1). It spawns `keep mcp` as a persistent stdio process
at gateway start, passing the active keep store explicitly so OpenClaw's safe
stdio environment filtering cannot silently fall back to `~/.keep`. If the MCP
process exits between turns, the plugin auto-reconnects on the next operation.
All operations have per-type timeouts (8s for assemble/prompts, 15s for
writes, 30s for queries).

---

## Upgrading

After upgrading keep:

```bash
uv tool upgrade keep-skill
openclaw gateway restart
```

If using a local path install:

```bash
openclaw plugins install -l $(keep config openclaw-plugin)
openclaw gateway restart
```

---

## Troubleshooting

**Plugin loads but MCP not connected:**
```bash
keep config                    # Verify keep is configured
keep config providers          # Verify providers are set
openclaw keep doctor           # Run health check
```

The plugin auto-reconnects MCP on each operation. If the MCP process keeps
dying, check `keep mcp` stderr output or run it manually:
```bash
keep --store "$KEEP_STORE_PATH" mcp   # Should start and wait on stdin
```

**Context engine not activating:**
Check that `plugins.slots.contextEngine` is set to `"keep"`:
```bash
grep contextEngine ~/.openclaw/openclaw.json
```

**Slow context assembly:**
The assemble flow runs parallel queries. If one is slow (e.g., embedding
provider timeout), it blocks assembly. Check:
```bash
keep doctor --log              # Watch flow execution in real time
```

**Agent sees setup instructions instead of context:**
The bootstrap context engine is active — keep CLI is either not installed or
providers aren't configured. Follow the agent's instructions, then restart.

**Reset to defaults:**
```bash
# Reset context engine slot to legacy
# Edit ~/.openclaw/openclaw.json: set plugins.slots.contextEngine to "legacy"
openclaw gateway restart

# Reset keep's state docs
keep config --reset-system-docs
```
