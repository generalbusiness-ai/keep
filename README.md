# keep

Agent memory that pays attention.  The foundation of skillful action.

It includes [skill instructions](SKILL.md) for reflective practice, and a powerful semantic memory system with [command-line](docs/QUICKSTART.md) and [MCP](docs/KEEP-MCP.md) interfaces. Fully local, or use API keys for model providers, or [cloud-hosted](https://keep.generalbusiness.ai) for multi-agent use.

```bash
uv tool install keep-skill       # or: pip install keep-skill
keep                              # interactive first-run setup (providers + agent hooks)

# Index content
keep put https://inguz.substack.com/p/keep -t topic=practice
keep put "Rate limit is 100 req/min" -t topic=api

# Index a codebase — recursive, with daemon-driven watch for changes
keep put ./my-project/ -r --watch

# Search by meaning
keep find "what's the rate limit?"

# Track what you're working on
keep now "Debugging auth flow"

# Instructions for reflection
keep prompt reflect
```

---

## What It Does

Store anything — notes, files, URLs — and `keep` summarizes, embeds, and tags each note. You search by meaning, by keyword, and by graph traversal.

Content goes in as conversation, text, PDF, HTML, Office documents, audio, or images; what comes back is a summary with tags and semantic neighbors. Audio files auto-extract metadata (artist, album, year); image files auto-extract EXIF (camera, date, dimensions). Optional vision/transcription enrichment runs when a media provider is configured, and scanned PDFs and images get background OCR when an OCR provider is available.

Full content is only stored if it's short (~3kB, configurable) - everything else gets summarized. Chunked analysis and summarization produces a focused searchable index that references original files or URLs without filling your disk with duplicated data.

This is more than a vector store: selected tags become **edges**. A tag becomes an edge when its tagdoc declares an `_inverse` — for example, `.tag/speaker` declares `_inverse: said`, so `speaker: Deborah` on a note makes `keep get Deborah` show every note where Deborah spoke. Ordinary tags stay as labels; you opt into navigability per tag key. Bundled edges include `speaker`, `author`, `references`, `cites`, `from`/`to`/`cc`, and `git_commit`; you can define your own. When you retrieve any note, keep follows its edges and fires standing queries (`.meta/*`) — surfacing open commitments, past learnings, related files, commit history. The right things appear at the right time. See [docs/EDGE-TAGS.md](docs/EDGE-TAGS.md).

- **Summarize, embed, tag** — URLs, files, and text are summarized and indexed on ingest
- **Contextual feedback** — Open commitments and past learnings surface automatically
- **Search** — Semantic similarity, BM25 full-text, and ranked graph traversal; scope to a folder or project
- **Tag organization** — Speech acts, status, project, topic, type — structured and queryable
- **Deep search** — Follow edges and tags from results to discover related notes across the graph
- **Edge tags** — Turn tags into navigable relationships with automatic inverse links
- **Git changelog** — Commits indexed as searchable notes with edges to touched files
- **Parts** — `analyze` decomposes documents into searchable sections, each with its own embedding and tags
- **Strings** — Every note is a string of versions; reorganize history by meaning with `keep move`
- **Watches** — Daemon-driven directory and file monitoring; re-indexes on change
- **Works offline** — Local models (MLX, Ollama, etc.), or API providers (Voyage, OpenAI, Gemini, Anthropic, Mistral)

> **[keep.generalbusiness.ai](https://keep.generalbusiness.ai)** — Hosted service. No local setup, no API keys to manage. Same SDK, managed infrastructure.

### The Practice

keep is designed as a skill for AI agents — a practice, not just a tool. The [skill instructions](SKILL.md) teach agents to reflect before, during, and after action: check intentions, recognize commitments, capture learnings, notice breakdowns. `keep prompt reflect` guides a structured reflection ([details](docs/KEEP-PROMPT.md)); `keep now` tracks current intentions and surfaces what's relevant.

This works because the tool and the skill reinforce each other. The tool stores and retrieves; the skill says *when* and *why*. An agent that uses both develops *skillful action* across sessions — not just recall, but looking before acting, and a deep review of outcomes afterwards.

> Why build memory for AI agents? What does "reflective practice" mean here? **[Read our blog for the back-story →](https://generalbusiness.ai/blog/)**

### Integration

Run `keep` (or `keep config --setup`) and the interactive wizard offers to install hooks for the coding-tool integrations it detects on your system. The other rows below have their own setup commands.

| Tool | How to install |
|------|-------------|
| **[Hermes Agent](docs/HERMES-INTEGRATION.md)** | `curl -sSL https://keep.generalbusiness.ai/scripts/install-hermes.sh \| bash` — installs plugin and runs setup. Fully integrated, with [additional skills](https://github.com/generalbusiness-ai/hermes-skills) available. |
| **[OpenClaw](docs/OPENCLAW-INTEGRATION.md)** | `openclaw plugins install clawhub:keep` (or `-l $(keep config openclaw-plugin)` from a local checkout). After that, `keep config --setup` keeps the plugin upgraded. Context engine plugin — full memory assembly, session archival, reflection triggers. |
| **Claude Desktop** | `keep config --setup`, then `keep config mcpb` to generate and open the .mcpb bundle ([details](docs/CLAUDE-DESKTOP.md)) |
| **Claude Code** | `/plugin marketplace add https://github.com/generalbusiness-ai/keep.git` then `/plugin install keep@generalbusiness-ai` |
| **VS Code Copilot** | `export KEEP_STORE_PATH=...` then `code --add-mcp "{\"name\":\"keep\",\"command\":\"keep\",\"args\":[\"--store\",\"$KEEP_STORE_PATH\",\"mcp\"]}"` |
| **GitHub Copilot CLI** | Auto-installed by `keep config --setup` when `~/.copilot/` is present. |
| **Kiro** | `export KEEP_STORE_PATH=...` then `kiro-cli mcp add --name keep --scope global -- keep --store "$KEEP_STORE_PATH" mcp` |
| **OpenAI Codex** | `export KEEP_STORE_PATH=...` then `codex mcp add keep -- keep --store "$KEEP_STORE_PATH" mcp` |
| **LangChain** | [LangGraph BaseStore](docs/LANGCHAIN-INTEGRATION.md), retriever, tools, and middleware |
| **Any MCP client** | [Stdio server](docs/KEEP-MCP.md) with 3 tools (`keep_flow`, `keep_prompt`, `keep_help`) |

After install, just tell your agent: *Please read all the keep_help documentation, and then use keep_prompt(name="reflect") to save some notes about what you learn.*

Manual MCP setup should treat `KEEP_STORE_PATH` as the source of truth and bake
that value into the launched `keep` command with `--store`; many MCP hosts do
not inherit arbitrary shell environment variables into stdio server processes.

---

## Installation

**Python 3.11–3.13.** Use [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv tool install keep-skill        # or: pip install keep-skill
keep                               # interactive first-run setup
```

The first run launches an interactive wizard: it picks up any API keys you have in the environment, detects [Ollama](https://ollama.com/) if it's running locally, lets you pick embedding and summarization providers, and offers to install hooks for any agent tools it finds. Re-run anytime with `keep config --setup`.

**Skip the wizard** by setting one of:

```bash
export KEEPNOTES_API_KEY=kn_...    # Hosted at https://keep.generalbusiness.ai — no local setup needed
export OPENAI_API_KEY=...          # Or GEMINI_API_KEY, OPENROUTER_API_KEY — each does both embeddings + summarization
export VOYAGE_API_KEY=...          # Or MISTRAL_API_KEY — embeddings only; pair with ANTHROPIC_API_KEY etc. for summarization
export GOOGLE_CLOUD_PROJECT=...    # Vertex AI via Workload Identity / ADC
```

Local option without API keys: install Ollama (auto-detected), or on macOS Apple Silicon `uv tool install 'keep-skill[local]'` for MLX models.

Or use a local OpenAI-API server (llama-server, vLLM, LM Studio, LocalAI) by setting `[embedding] name = "openai"` with `base_url = "http://localhost:8801/v1"` in `keep.toml` — see [docs/KEEP-CONFIG.md](docs/KEEP-CONFIG.md).

**LangChain/LangGraph** integration: `pip install keep-skill[langchain]` or `pip install langchain-keep`.

See [docs/QUICKSTART.md](docs/QUICKSTART.md) and [docs/KEEP-CONFIG.md](docs/KEEP-CONFIG.md) for the full provider matrix and configuration options.

---

## Quick Start

```bash
# Index URLs, files, and notes (store auto-initializes on first use)
keep put https://example.com/api-docs -t topic=api
keep put "Token refresh needs clock sync" -t topic=auth

# Index a codebase — recursive, with auto-watch for changes
keep put ./my-project/ -r --watch
# git: 2 repo(s) queued for changelog ingest

# Search
keep find "authentication flow" --limit 5
keep find "auth" --deep                # Follow edges to discover related notes
keep find "auth" --scope 'file:///Users/me/project/*'  # Scoped to a folder

# Retrieve
keep get file:///path/to/doc.md
keep get ID --history                  # All versions
keep get ID --parts                    # Analyzed sections

# Tags
keep list --tag project=myapp          # Find by tag
keep list 'git://github.com/org/repo@*'  # All git tags/releases

# Current intentions
keep now                               # Show what you're working on
keep now "Fixing login bug"            # Update intentions
```

### Python API

```python
from keep import Keeper

kp = Keeper()

# Index
kp.put(uri="file:///path/to/doc.md", tags={"project": "myapp"})
kp.put("Rate limit is 100 req/min", tags={"topic": "api"})

# Search
results = kp.find("rate limit", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.summary}")

# Version history
prev = kp.get_version("doc:1", offset=1)
versions = kp.list_versions("doc:1")
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for configuration and more examples.

---

## Documentation

Full docs at **[docs.generalbusiness.ai](https://docs.generalbusiness.ai)** — or browse locally:

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — Setup, configuration, first steps
- **[docs/REFERENCE.md](docs/REFERENCE.md)** — Quick reference index
- **[docs/KEEP-PUT.md](docs/KEEP-PUT.md)** — Indexing: files, directories, URLs, git changelog, watches
- **[docs/KEEP-FIND.md](docs/KEEP-FIND.md)** — Semantic search, deep search, scoped search
- **[docs/TAGGING.md](docs/TAGGING.md)** — Tags, speech acts, project/topic organization
- **[docs/PROMPTS.md](docs/PROMPTS.md)** — Prompts for summarization, analysis, and agent workflows
- **[docs/OPENCLAW-INTEGRATION.md](docs/OPENCLAW-INTEGRATION.md)** — OpenClaw context engine plugin
- **[docs/KEEP-MCP.md](docs/KEEP-MCP.md)** — MCP server for AI agent integration
- **[docs/AGENT-GUIDE.md](docs/AGENT-GUIDE.md)** — Working session patterns
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — How it works under the hood
- **[SKILL.md](SKILL.md)** — The reflective practice (for AI agents)

---

## License

MIT

---

## Contributing

Published on [PyPI as `keep-skill`](https://pypi.org/project/keep-skill/).

Issues and PRs welcome:
- Provider implementations
- Performance improvements
- Documentation clarity

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
