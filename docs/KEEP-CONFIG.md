# keep config

Show configuration and resolve paths.

## Setup wizard

```bash
keep config --setup                   # Interactive setup wizard
```

The setup wizard walks you through choosing a store location, detecting providers, and configuring integrations. It runs automatically on first use, or you can re-run it at any time.

## Usage

```bash
keep config                           # Show all config
keep config file                      # Config file location
keep config tool                      # Package directory (SKILL.md location)
keep config docs                      # Documentation directory
keep config store                     # Store path
keep config mcpb                      # Generate .mcpb for Claude Desktop
keep config openclaw-plugin           # OpenClaw plugin directory
keep config providers                 # All provider config
keep config providers.embedding       # Embedding provider name
```

## Options

| Option | Description |
|--------|-------------|
| `--setup` | Run the interactive setup wizard |
| `--reset-system-docs` | Force reload system documents from bundled content |
| `--state-diagram` | Print a Mermaid state-transition diagram of the `.state/*` notes currently in the store (reflects any edits or additions) |
| `-s`, `--store PATH` | Override store directory (global option, available on every subcommand) |

## Config file location

The config file is `keep.toml` inside the config directory. The config directory is resolved in this order:

1. **`KEEP_CONFIG` environment variable** — explicit path to config directory
2. **Tree-walk** — search from current directory up to `~` for `.keep/keep.toml`
3. **Default** — `~/.keep/`

The tree-walk enables project-local stores: place a `.keep/keep.toml` in your project root and `keep` will use it when you're in that directory tree.

## Store path resolution

The store (where data lives) is resolved separately from config:

1. **`--store` CLI option** — per-command override
2. **`KEEP_STORE_PATH` environment variable**
3. **`store.path` in config file** — `[store]` section of `keep.toml`
4. **Config directory itself** — backwards compatibility default

## Config file format

```toml
[store]
version = 2
max_summary_length = 1000

[embedding]
name = "ollama"                        # or "voyage", "openai", "openrouter", "gemini", "mistral", "mlx", "sentence-transformers"
model = "nomic-embed-text"

[summarization]
name = "ollama"                        # or "anthropic", "openai", "openrouter", "gemini", "mistral", "mlx"
model = "gemma3:1b"

[media]
name = "ollama"                        # or "mlx" (auto-detected)
# vision_model = "llava"              # Ollama vision model for image description

[document]
name = "composite"

[tags]
project = "my-project"                 # Default tags applied to all new items
owner = "alice"
required = ["user"]                    # Tags that must be present on every put()
namespace_keys = ["category", "user"]  # LangGraph namespace-to-tag mapping

[remote_store]
api_url = "https://api.keepnotes.ai"   # Optional: authoritative remote store
api_key = "kn_..."
project = "my-project"

[remote_task]
api_url = "https://api.keepnotes.ai"   # Optional: hosted background task delegation
api_key = "kn_..."
project = "my-project"
```

### Tags section details

- **Default tags** (key = "value") — Applied to all new items. Overridden by user tags.
- **`required`** — List of tag keys that must be present on every `put()` call. Raises `ValueError` if missing. System docs (dot-prefix IDs like `.meta/`) are exempt.
- **`namespace_keys`** — Positional mapping for [LangChain integration](LANGCHAIN-INTEGRATION.md). Maps LangGraph namespace tuple components to Keep tag names.

## Providers

Keep needs an embedding provider (for search) and a summarization provider (for summaries). Most providers do both.

### Hosted Service

Sign up at [keepnotes.ai](https://keepnotes.ai) to get an API key — no local models, no database setup:

```bash
export KEEPNOTES_API_KEY=kn_...
keep put "test"                    # That's it — storage, search, and summarization handled
```

Works across all your tools (Claude Code, Kiro, Codex) with the same API key. Project isolation, media pipelines, and backups are managed for you.

Environment variables `KEEPNOTES_API_URL`, `KEEPNOTES_API_KEY`, and
`KEEPNOTES_PROJECT` target the remote authoritative store. In `keep.toml`, the
authoritative store is configured under `[remote_store]`. Hosted background task
delegation, when configured separately, uses `[remote_task]`.

### Ollama (Recommended Local Option)

[Ollama](https://ollama.com/) is the easiest way to run keep locally with no API keys. Install Ollama and go — keep handles the rest:

```bash
# 1. Install Ollama from https://ollama.com/
# 2. That's it:
keep put "test"                     # Auto-detected, models pulled automatically
```

Keep auto-detects Ollama and pulls the models it needs on first use. It picks the best available model for each task: dedicated embedding models for embeddings, generative models for summarization. Respects `OLLAMA_HOST` if set.

Ollama runs models in a separate server process, so keep itself stays lightweight (~36 MB RSS) regardless of model size.

### API Providers

Set environment variables for your preferred providers:

| Provider | Env Variable | Get API Key | Embeddings | Summarization |
|----------|--------------|-------------|------------|---------------|
| **Voyage AI** | `VOYAGE_API_KEY` | [dash.voyageai.com](https://dash.voyageai.com/) | yes | - |
| **Anthropic** | `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`* | [console.anthropic.com](https://console.anthropic.com/) | - | yes |
| **OpenAI** | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) | yes | yes |
| **OpenRouter** | `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai/) | yes | yes |
| **Google Gemini** | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/) | yes | yes |
| **Mistral** | `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai/) | yes | yes |
| **Vertex AI** | `GOOGLE_CLOUD_PROJECT` | GCP Workload Identity / ADC | yes | yes |

\* **Anthropic Authentication Methods:**
- **API Key** (`ANTHROPIC_API_KEY`): Recommended. Get from [console.anthropic.com](https://console.anthropic.com/). Format: `sk-ant-api03-...`
- **OAuth Token** (`CLAUDE_CODE_OAUTH_TOKEN`): For Claude Pro/Team subscribers. Generate via `claude setup-token`. Format: `sk-ant-oat01-...`
  - OAuth tokens from `claude setup-token` are primarily designed for Claude Code CLI authentication
  - For production use with `keep`, prefer using a standard API key from the Anthropic console

**Simplest setup** (single API key):
```bash
export OPENAI_API_KEY=...      # Does both embeddings + summarization
# Or: OPENROUTER_API_KEY=...   # Also does both, via OpenRouter
# Or: GEMINI_API_KEY=...       # Also does both
keep put "test"
```

OpenRouter accepts OpenRouter-prefixed model names such as
`openai/text-embedding-3-small` and `openai/gpt-4.1-mini`. For common models,
bare names are accepted and normalized to the prefixed form in config and
diagnostics.

**Best quality** (two API keys for optimal embeddings):
```bash
export VOYAGE_API_KEY=...      # Embeddings (Anthropic's partner)
export ANTHROPIC_API_KEY=...   # Summarization (cost-effective: claude-3-haiku)
keep put "test"
```

### Choosing Between OpenAI, OpenRouter, and Local OpenAI-Compatible Servers

These three paths overlap at the protocol level, but they are different
provider choices in keep:

| Use case | keep provider | Typical config |
|----------|---------------|----------------|
| Direct OpenAI API | `openai` | `OPENAI_API_KEY` |
| OpenRouter hosted routing layer | `openrouter` | `OPENROUTER_API_KEY` |
| Local or self-hosted OpenAI-compatible server (`llama-server`, vLLM, LM Studio, LocalAI) | `openai` | `name = "openai"` plus `base_url = "http://..."` |

- Choose **`openai`** when you want OpenAI's API directly.
- Choose **`openrouter`** when you want OpenRouter's routing, prefixed model names such as `openai/gpt-4.1-mini`, or OpenRouter-specific headers.
- Choose **`openai` + `base_url`** when you are talking to a local or self-hosted server that implements the OpenAI API. This is the path for **llama-server** (llama.cpp), **vLLM**, **LM Studio**, and similar servers.
- Do not use `openai` with `base_url = "https://openrouter.ai/api/v1"` in keep. OpenRouter is a first-class provider because its config, model naming, and headers differ from direct OpenAI and local compatible servers.

Auto-detection priority follows the same separation:

1. Direct provider keys (`VOYAGE_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`)
2. `OPENROUTER_API_KEY`
3. Local providers such as Ollama or MLX

### Local MLX Providers (Apple Silicon)

For offline operation without Ollama on macOS Apple Silicon. Models run in-process using Metal acceleration — faster cold-start but higher memory usage (~1 GB+). Ollama is generally recommended instead for better stability and performance, especially for background processing.

```bash
uv tool install 'keep-skill[local]'
keep put "test"             # No API key needed
```

### Claude Desktop Setup

Install the keep connector in Claude Desktop:

```bash
keep config mcpb
```

This generates a `.mcpb` bundle and opens it with Claude Desktop. You will be prompted to install the `keep` connector, which gives Claude Desktop full access to the memory system and help pages.

Any provider that works with keep (Ollama, MLX, API-based) works with Claude Desktop — MCP tools run locally via stdio.

### Available Models

Bundled default model choices live in
`keep/data/default-provider-models.yaml`. They are used only when the
corresponding provider is selected and `keep.toml` does not set an explicit
`model`.

| Provider | Type | Models |
|----------|------|--------|
| **Voyage** | Embeddings | `voyage-3.5-lite` (default), `voyage-3-large`, `voyage-code-3` |
| **Anthropic** | Summarization | `claude-haiku-4-5-20251001` (default), `claude-3-5-haiku-20241022` |
| **OpenAI** | Embeddings | `text-embedding-3-small` (default), `text-embedding-3-large` |
| **OpenAI** | Summarization | `gpt-4.1-mini` (default), `gpt-4.1` |
| **OpenRouter** | Embeddings | `openai/text-embedding-3-small` (default), `openai/text-embedding-3-large` |
| **OpenRouter** | Summarization | `openai/gpt-4.1-mini` (default), other OpenRouter model slugs |
| **Gemini** | Embeddings | `text-embedding-004` (default) |
| **Gemini** | Summarization | `gemini-2.5-flash` (default), `gemini-2.5-pro` |
| **Mistral** | Embeddings | `mistral-embed` (default, 1024 dims) |
| **Mistral** | Summarization | `mistral-small-latest` (default), `mistral-large-latest` |
| **Mistral** | OCR | `mistral-ocr-latest` — cloud OCR for images and PDFs |
| **Ollama** | Embeddings | `nomic-embed-text` (recommended), `mxbai-embed-large` |
| **Ollama** | Summarization | `gemma3:1b` (fast), `llama3.2:3b`, `mistral`, `phi3` |
| **Ollama** | Media | Vision models: `llava`, `moondream`, `bakllava` (images only) |
| **Ollama** | OCR | `glm-ocr` (auto-pulled on first use) — scanned PDFs and images |
| **MLX** | Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, Apple Silicon only) |
| **MLX** | Summarization | MLX models, e.g. `Llama-3.2-3B-Instruct-4bit` (Apple Silicon only) |
| **MLX** | Media | `mlx-vlm` for images, `mlx-whisper` for audio (Apple Silicon only) |
| **MLX** | OCR | `mlx-vlm` vision models (Apple Silicon only) |

### OpenAI-Compatible Local Servers

The `openai` provider supports a `base_url` parameter that redirects it to any
server implementing the OpenAI API (``/v1/embeddings``, ``/v1/chat/completions``).
This works with **llama-server** (llama.cpp), **vLLM**, **LM Studio**, **LocalAI**,
and similar tools.  No API key is required for local servers.

This section is for **local/self-hosted OpenAI-compatible servers**, not OpenRouter.
For OpenRouter, use `name = "openrouter"` with `OPENROUTER_API_KEY`.

Each model needs its own server process (llama-server binds one model per port):

```bash
# Terminal 1: embedding model
llama-server --model nomic-embed-text-v1.5.Q8_0.gguf --port 8801 --embedding

# Terminal 2: chat model
llama-server --model llama-3.2-3b-instruct.Q4_K_M.gguf --port 8802
```

Then configure `keep.toml`:

```toml
[embedding]
name = "openai"
model = "nomic-embed-text"
base_url = "http://localhost:8801/v1"

[summarization]
name = "openai"
model = "llama-3.2-3b"
base_url = "http://localhost:8802/v1"
```

No auto-detection or auto-pull — you manage the server processes yourself.
Embedding dimensions are detected automatically on first use.

### Media Description (optional)

When configured, images and audio files get model-generated descriptions alongside their extracted metadata, making them semantically searchable. Without this, media files are indexed with metadata only (EXIF, ID3 tags).

```toml
[media]
name = "mlx"
vision_model = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
whisper_model = "mlx-community/whisper-large-v3-turbo"
```

Install media dependencies (Apple Silicon): `pip install keep-skill[media]`

Auto-detected if `mlx-vlm` or `mlx-whisper` is installed, or if Ollama has a vision model (e.g. `llava`).

## Environment variables

```bash
KEEP_STORE_PATH=/path/to/store       # Override store location
KEEP_CONFIG=/path/to/.keep           # Override config directory
KEEP_TAG_PROJECT=myapp               # Auto-apply tags (any KEEP_TAG_* variable)
KEEP_VERBOSE=1                       # Debug logging to stderr
OLLAMA_HOST=http://localhost:11434   # Ollama server URL (auto-detected)
OPENROUTER_API_KEY=...               # For OpenRouter (embeddings + summarization)
OPENAI_API_KEY=sk-...                # For OpenAI (embeddings + summarization)
KEEP_OPENAI_API_KEY=sk-...           # Explicit OpenAI override used by keep
GEMINI_API_KEY=...                   # For Gemini (embeddings + summarization)
GOOGLE_CLOUD_PROJECT=my-project      # Vertex AI via Workload Identity / ADC
GOOGLE_CLOUD_LOCATION=us-east1       # Vertex AI region (default: us-east1)
VOYAGE_API_KEY=pa-...                # For Voyage embeddings only
ANTHROPIC_API_KEY=sk-ant-...         # For Anthropic summarization only
MISTRAL_API_KEY=...                  # For Mistral (embeddings + summarization + OCR)
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...  # OAuth token alternative
KEEPNOTES_API_KEY=kn_...             # For hosted keepnotes.ai service
```

## Data security

### Encryption at rest

Keep stores data in SQLite databases and ChromaDB files on disk. These are **not encrypted** by default.

If you store sensitive content (plans, credentials, reasoning traces), enable disk encryption:

| OS | Solution | How |
|----|----------|-----|
| **macOS** | FileVault | System Settings > Privacy & Security > FileVault |
| **Linux** | LUKS | Encrypt home directory or the partition containing `~/.keep/` |
| **Windows** | BitLocker | Settings > Privacy & security > Device encryption |

This is the recommended approach because it transparently covers both SQLite and ChromaDB's internal storage without application-level changes.

## Troubleshooting

Run `keep doctor` to check your installation:

```bash
keep doctor                           # Check providers, store, integrations
```

**No embedding provider configured:** Set an API key (e.g., `VOYAGE_API_KEY`), install and start Ollama, or install `keep-skill[local]`.

**Model download hangs:** First use of local models downloads weights (~minutes). Cached in `~/.cache/`.

**ChromaDB errors:** Delete `~/.keep/chroma/` to reset.

**Slow local summarization:** Large content is summarized in the background automatically. Use `keep daemon` to monitor progress.

**Claude Code hooks need `jq`:** The prompt-submit hook uses `jq` to extract context. Install with your package manager (e.g., `brew install jq`). Hooks are fail-safe without it, but prompt context won't be captured.

## Config subpaths

| Path | Returns |
|------|---------|
| `file` | Config file path (`~/.keep/keep.toml`) |
| `tool` | Package directory (where SKILL.md lives) |
| `docs` | Documentation directory |
| `store` | Store data path |
| `mcpb` | Generate `.mcpb` bundle for Claude Desktop |
| `openclaw-plugin` | OpenClaw plugin directory |
| `providers` | All provider configuration |
| `providers.embedding` | Embedding provider name |
| `providers.summarization` | Summarization provider name |
| `providers.media` | Media description provider name |

Subpath output is raw (unquoted) for shell scripting:

```bash
cat "$(keep config tool)/SKILL.md"    # Read the practice guide
ls "$(keep config store)"             # List store contents
```

## Resetting system documents

System documents (`.conversations`, `.domains`, `.tag/*`, etc.) are bundled with keep and loaded on first use. If they've been modified or corrupted:

```bash
keep config --reset-system-docs       # Reload all from bundled content
```

## See Also

- [QUICKSTART.md](QUICKSTART.md) — Get started with keep in 5 minutes
- [REFERENCE.md](REFERENCE.md) — Quick reference index
- [LANGCHAIN-INTEGRATION.md](LANGCHAIN-INTEGRATION.md) — LangChain/LangGraph integration
- [ARCHITECTURE.md](ARCHITECTURE.md) — System internals
