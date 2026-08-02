# Architecture Overview

## What is keep?

**keep** is a reflective memory system. It gives agents a comprehensive tool for persistent indexing, tagging, entity relationship management, summarization, semantic and timeline analysis, and powerful contextual recall. It's designed as an agent skill for Claude Code, OpenClaw, LangChain/LangGraph, and other agentic environments, enabling agents to remember information across sessions over time.

Published by Hugh Pyle, "inguz ᛜ outcomes", under the MIT license.
Contributions are welcome; code is conversation, "right speech" is encouraged.

---

## Core Concept

Every stored note has:
- **ID**: URI or custom identifier
- **Summary**: Human-readable text (stored, searchable)
- **Embedding**: Vector representation (for semantic search)
- **Tags**: Key-value metadata (for filtering)
- **Timestamps**: Created/updated/accessed (auto-managed)
- **Version History**: Previous versions archived automatically on update
- **Parts**: Optional structural decomposition (from `analyze`)

The original document content is **not stored** — only the summary and embedding.

---

## Architecture

keep is layered. Surface clients (CLI, MCP, LangChain, Claude Desktop bundle)
are thin wrappers that talk to a long-running daemon over HTTP. The daemon
hosts a `Keeper`, which composes provider, store, action, and flow modules.
Background work runs out-of-band on the daemon's queues.

```
┌────────────────────────────────────────────────────────────────────────┐
│  Surface clients                                                       │
│  ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ cli_app  │  │  mcp.py │  │ langchain/   │  │ mcpb.py (Claude    │   │
│  │ (typer)  │  │ (stdio) │  │ adapters     │  │ Desktop bundle)    │   │
│  └────┬─────┘  └────┬────┘  └──────┬───────┘  └──────┬─────────────┘   │
└───────┼─────────────┼──────────────┼─────────────────┼─────────────────┘
        │             │              │                 │
        │  HTTP (loopback, token-auth, host-header guarded)
        ▼             ▼              ▼                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Daemon (daemon.py / daemon_server.py / daemon_client.py)              │
│  Routes:  /v1/notes, /v1/notes/{id}, /v1/notes/{id}/tags,              │
│           /v1/notes/{id}/context, /v1/search, /v1/flow,                │
│           /v1/analyze, /v1/ready, /v1/health, /v1/admin/*              │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Keeper (api.py)                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  api.py:Keeper = ProviderLifecycleMixin                          │  │
│  │                + BackgroundProcessingMixin                       │  │
│  │                + SearchAugmentationMixin                         │  │
│  │                + ContextResolutionMixin                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  Implements high-level put/find/get/tag/move/delete/revert/analyze.    │
│  Many user-visible operations are dispatched through `actions/`.       │
│  Stable execution boundary: `run_flow()` over named state docs.        │
└────────────┬───────────────────┬────────────────────┬──────────────────┘
             │                   │                    │
             ▼                   ▼                    ▼
   ┌──────────────────┐ ┌─────────────────┐ ┌───────────────────────────┐
   │  Providers       │ │ Storage         │ │ Background work           │
   │  (providers/)    │ │ backends        │ │                           │
   │  embedding /     │ │ DocumentStore   │ │ pending_summaries.py      │
   │  summarization / │ │ (SQLite)        │ │ work_queue.py /           │
   │  document /      │ │ ChromaStore     │ │ work_processor.py         │
   │  media / OCR /   │ │ (vectors)       │ │ task_client.py            │
   │  analyzer        │ │ PendingQueue    │ │ (hosted delegation)       │
   │                  │ │ → backend.py    │ │ planner_stats.py          │
   └──────────────────┘ └─────────────────┘ └───────────────────────────┘
```

---

## Layers

### 1. Surface clients

**[cli_app.py](keep/cli_app.py)** — Typer command app
- Most commands are HTTP calls to the daemon via `daemon_client.http_request`
- A small set of commands stay local: setup, daemon control, MCP server, data
  import/export
- Auto-spawns the daemon on first use

The CLI is intentionally thin for ordinary note operations (`put`, `get`,
`find`, `tag`, flow execution): it resolves shell concerns, sends daemon HTTP
requests, and renders responses. Commands that need direct process control or
local filesystem traversal remain CLI-owned for now: setup/config discovery,
daemon lifecycle, MCP stdio startup, bulk directory ingestion, and data
import/export. Those commands may construct a local `Keeper` or use local graph
helpers, but that is an explicit exception to the daemon-backed command path.

**[mcp_surface.py](keep/mcp_surface.py) / [mcp.py](keep/mcp.py)** — shared MCP contract and stdio adapter
- SDK v2 `MCPServer` with MCP 2026-07-28 and legacy 2025-11-25 negotiation
- Three tools: `keep_flow`, `keep_prompt`, `keep_help`
- Dynamic prompt exposure: prompt docs tagged with `mcp_prompt` become native
  MCP prompts (protocol-level `list_prompts` / `get_prompt`)
- MCP resources: `keep://now` (current note) and `keep://{id}` (any note by ID)
- `mcp_surface.py` owns schemas, annotations, structured results, prompts,
  resources, and protocol errors so local and hosted deployments cannot drift
- `mcp.py` is a thin async adapter: blocking REST calls run in worker threads
  and delegate to the daemon or configured hosted backend. No local Keeper,
  models, or database.
- Structured output: `keep_flow` and `keep_prompt` return `CallToolResult` with
  `structuredContent`; flow failures use MCP tool-error semantics

**[langchain/](keep/langchain/)** — Framework adapters
- `KeepStore` (LangGraph `BaseStore` adapter)
- `KeepNotesToolkit` (LangChain tools)
- `KeepNotesRetriever` (`BaseRetriever` with now-context)
- `KeepNotesMiddleware` (LCEL runnable for auto-injection)
- See the [LangChain section](#langchain--langgraph-integration) below.

**[mcpb.py](keep/mcpb.py)** — Claude Desktop bundle
- Builds the .mcpb archive consumed by Claude Desktop's MCP loader

### 2. Daemon layer

**[daemon.py](keep/daemon.py)** — Daemon entry point
- Minimal `keepd --store PATH` or `python -m keep.daemon --store PATH` runner
- Constructs a `Keeper` with `defer_startup_maintenance=True` and runs the
  pending-work daemon loop

**[daemon_server.py](keep/daemon_server.py)** — HTTP query server
- `DaemonServer` exposes the daemon HTTP API with the routes shown above
- Auth: bearer token in `Authorization` header (random per daemon, persisted
  in `~/.keep/.processor.token`)
- Local mode defaults to loopback bind + strict loopback `Host` allowlist
- Remote mode is explicit (`--bind` / `KEEP_DAEMON_BIND_HOST`, optional
  `--advertised-url` / `KEEP_DAEMON_ADVERTISED_URL`) and uses a mode-aware
  `Host` allowlist derived from the bind host and advertised URL
- Non-loopback binds require explicit trusted-proxy acknowledgment
  (`--trusted-proxy` / `KEEP_DAEMON_TRUSTED_PROXY=1`); keep does not provide
  in-process TLS for the daemon HTTP server
- Wildcard remote binds (`0.0.0.0` / `::`) require `advertised_url` so the
  `Host` check remains active
- `GET /v1/ready` and `GET /v1/health` publish capability and network
  descriptors so remote clients can negotiate support explicitly
- Request handlers refuse new work while shutdown is in progress
- OpenTelemetry trace context is propagated from CLI/MCP into daemon spans
- Daemon error payloads include a `request_id`; CLI and remote clients include
  it in surfaced errors so an operator can correlate a failure with daemon logs
- Set `KEEP_TRACE=1` on the daemon to emit timing-tree traces. Storage spans
  cover SQLite document-store operations and ChromaDB embedding/metadata
  queries with low-cardinality attributes only; note content and raw SQL are
  intentionally omitted from trace attributes.

**[daemon_client.py](keep/daemon_client.py)** — Daemon discovery and HTTP
- `get_port()`: locate or auto-spawn the daemon for a store
- `http_request()`: shared HTTP plumbing used by both CLI and MCP

The CLI and MCP layers each have their own retry-on-disconnect logic so they
can gracefully follow a daemon that has restarted on a new port.

### 3. Keeper (core API)

**[api.py](keep/api.py)** — Main facade
- `Keeper` is composed from four mixins for organizational reasons:
  - `ProviderLifecycleMixin` ([_provider_lifecycle.py](keep/_provider_lifecycle.py))
    — lazy init with double-checked locking, GPU release helpers
  - `BackgroundProcessingMixin` ([_background_processing.py](keep/_background_processing.py))
    — task dispatch, processing pipeline, process spawning
  - `SearchAugmentationMixin` ([_search_augmentation.py](keep/_search_augmentation.py))
    — deep-follow, recency decay, RRF fusion
  - `ContextResolutionMixin` ([_context_resolution.py](keep/_context_resolution.py))
    — display-context assembly, prompt rendering, meta-doc resolution,
    similar-for-display
- Coordinates providers and stores
- Implements query operations with recency decay
- Content-based embedding dedup (skips re-embedding when content matches an
  existing document)

**[actions/](keep/actions/)** — Action implementations
A package of focused modules implementing user-visible operations behind
`Keeper`/flows:

```
analyze     find_supernodes     ocr             resolve_meta
auto_tag    generate            put             resolve_stubs
delete      get                 resolve_duplicates  stats
describe    list_parts          resolve_edges   summarize
extract_links list_versions     traverse        tag
find        move                                 ...
```

Most are dispatched from state-doc flows; some are still called directly
from `Keeper` methods during the migration to flows.

**[protocol.py](keep/protocol.py)** — Abstract interfaces
- `FlowHostProtocol`: minimal backend-neutral boundary (`run_flow` + `close`)
  — this is the stable semantic boundary shared by local and hosted stores
- `KeeperProtocol` (extends `FlowHostProtocol`) — richer object API used
  during migration to the flow boundary
- `VectorStoreProtocol`, `DocumentStoreProtocol`, `PendingQueueProtocol` —
  storage backend contracts
- Enables pluggable backends (local SQLite/ChromaDB or remote
  PostgreSQL/pgvector)

**[flow_client.py](keep/flow_client.py)** — Shared flow-backed wrappers
- Convenience helpers (`get`, `put`, `find`, `tag`, `delete`, `move`, `now`)
  over `FlowHostProtocol.run_flow`
- Used by both `Keeper` and `RemoteKeeper` — one semantic path for local and
  hosted
- Parameter normalization and response coercion only; no semantic behavior

**[remote.py](keep/remote.py)** — Remote client
- HTTP client implementing `FlowHostProtocol`
- Public methods delegate through `flow_client` wrappers, same as local
  `Keeper`
- Connects to the hosted REST API (keepmem)

### 4. Flow runtime (state docs)

The Keeper exposes `run_flow(state, params, ...)` as its stable execution
boundary. A "state" is a named YAML state-doc that declares rules,
predicates, and actions. The runtime evaluates them and dispatches actions
from `actions/` against the Keeper.

**[state_doc.py](keep/state_doc.py)** — Loader, compiler, evaluator
- Loads `.state/*` documents from the keep store
- Compiles CEL predicates
- Defines `AsyncActionEncountered` so a foreground flow can hand off to the
  background queue mid-evaluation

**[state_doc_runtime.py](keep/state_doc_runtime.py)** — Synchronous runtime
- Evaluates state docs with inline action execution
- Used for the read/query path: query resolution, context assembly, deep find
- Enforces a per-call tick budget

**[system_docs.py](keep/system_docs.py) / [builtin_state_docs.py](keep/builtin_state_docs.py)**
— System doc inventory
- Bundled `.state/*`, `.tag/*`, `.prompt/*`, `.now`, etc., installed into the
  store on first use
- IDs are stable, e.g. `_system:now → .now`, `_tag:act → .tag/act`

**[flow_env.py](keep/flow_env.py)** — Local flow execution environment
- `LocalFlowEnvironment` glues the runtime, action runner, and Keeper
  together for local execution

Flows that must complete before returning to the caller (find, get-context,
deep-find) run synchronously in this runtime. Write-side flows can suspend
and continue on the background work queue.

### 5. Background work

**[pending_summaries.py](keep/pending_summaries.py)** — Pending task queue
- SQLite-backed (`pending_summaries.db`)
- Deferred processing: `summarize`, `embed`, `reindex`, `ocr`, `analyze`
- Atomic dequeue with PID claims; stale-claim recovery for crashed processors
- Exponential backoff on failure (30s → 1h); dead-letter for exhausted retries

**[work_queue.py](keep/work_queue.py) / [work_processor.py](keep/work_processor.py)**
— Direct work queue
- Backed by the `continuation.db` SQLite file
- Enqueue/claim/complete/fail semantics for write-side flow continuations
  that can't complete synchronously
- Reuses the legacy `continue_work` table schema (hence the file name)

**[processors.py](keep/processors.py)** — Content processing helpers
- Content hashing, text normalization, processing-pipeline glue used by both
  the synchronous and pending paths

**[task_client.py](keep/task_client.py) / [task_workflows.py](keep/task_workflows.py)**
— Hosted task delegation
- When `config.remote` is set, expensive processing can be delegated to the
  hosted backend rather than run locally
- Initialized from `Keeper.__init__` when remote task delegation is configured

**[planner_stats.py](keep/planner_stats.py)** — Flow discriminator priors
- Precomputed statistics for flow planning
- Bootstrap rebuild is enqueued from `Keeper.__init__` when stats are missing

**[recovery.py](keep/recovery.py)** — DB recovery
- Detects and handles malformed SQLite databases (used by `document_store.py`)

### 6. Storage backends

**[document_store.py](keep/document_store.py)** — Document persistence (local)
- `DocumentStore` wraps SQLite (`documents.db`)
- Stores summaries, tags, timestamps, content hashes
- Version history: archives previous versions on update
- Parts table: structural decomposition from `analyze`
- Schema versioning + migrations (current `SCHEMA_VERSION = 14`)
- FTS index for keyword fallback search

**[store.py](keep/store.py)** — Vector persistence (local)
- `ChromaStore` wraps ChromaDB
- Handles vector storage, similarity search, metadata queries
- Versioned embeddings: `{id}@v{N}` for history
- Part embeddings: `{id}@p{N}` for structural decomposition

**[backend.py](keep/backend.py)** — Pluggable storage factory
- Creates store backends based on configuration
- External backends register via the `keep.backends` entry point
- Returns `StoreBundle` (doc store, vector store, pending queue, work queue,
  is_local flag)

**[paths.py](keep/paths.py) / [config.py](keep/config.py)** — Paths and config
- `config.py` detects available providers (platform, API keys, Ollama),
  persists choices in `keep.toml`, and auto-creates on first use
- `paths.py` resolves the store/config directories, honoring `KEEP_CONFIG`
  and `KEEP_STORE_PATH`

### 7. Providers

All providers register through `providers/base.py:ProviderRegistry`. The
registry is populated lazily on first use so optional dependencies don't
break startup.

#### Embedding Providers
Generate vector representations for semantic search.

- **gemini**: API-based, Google (`GEMINI_API_KEY` or `GOOGLE_CLOUD_PROJECT`
  for Vertex AI)
- **voyage**: API-based, Anthropic's recommended partner (`VOYAGE_API_KEY`)
- **openai**: API-based, high quality (`OPENAI_API_KEY`)
- **openrouter**: API-based routing layer over multiple model providers
  (`OPENROUTER_API_KEY`)
- **mistral**: API-based (`MISTRAL_API_KEY`)
- **ollama**: Local server, auto-detected, any model (`OLLAMA_HOST`)
- **sentence-transformers**: Local, CPU/GPU, no API key
- **mlx**: Apple Silicon optimized, local, no API key

Dimension is determined by the model and must be consistent across indexing
and queries. Embeddings are cached through `providers/embedding_cache.py`
(`embedding_cache.db`).

`openai` also supports `base_url` for local or self-hosted OpenAI-compatible
servers such as llama.cpp `llama-server`, vLLM, LM Studio, or LocalAI. That is
distinct from the `openrouter` provider, which has its own model naming and
headers even though both use the OpenAI SDK underneath.

#### Summarization Providers
Generate human-readable summaries from content.

- **anthropic**: LLM-based (`ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`)
- **openai**: LLM-based, high quality (`OPENAI_API_KEY`)
- **openrouter**: LLM-based routing layer (`OPENROUTER_API_KEY`)
- **gemini**: LLM-based, Google (`GEMINI_API_KEY` or `GOOGLE_CLOUD_PROJECT`)
- **mistral**: LLM-based (`MISTRAL_API_KEY`)
- **ollama**: LLM-based, local server, auto-detected (`OLLAMA_HOST`)
- **mlx**: LLM-based, local, Apple Silicon
- **truncate**: Simple text truncation (fallback)
- **first_paragraph**: First-paragraph extraction (non-LLM)
- **passthrough**: Store content as-is (with length limit)

**Contextual Summarization.** When documents have user tags (domain, topic,
project, etc.), the summarizer receives context from related items. This
produces summaries that highlight relevance to the tagged context rather
than generic descriptions.

How it works:
1. When processing pending summaries, the system checks for user tags
2. Finds similar items that share any of those tags (OR-union)
3. Boosts scores for items sharing multiple tags (+20% per additional match)
4. Top 5 related summaries are passed as context to the LLM
5. The summary reflects what's relevant to that context

Example: indexing a medieval text with `domain=practice` produces a summary
highlighting its relevance to contemplative practice, not just "a 13th-century
guide for anchoresses."

**Tag changes trigger re-summarization.** When user tags are added, removed,
or changed on an existing document, it's re-queued for contextual
summarization even if content is unchanged. The existing summary is
preserved until the new one is ready.

Non-LLM providers (`truncate`, `first_paragraph`, `passthrough`) ignore
context.

#### Document Providers
Fetch content from URIs with content regularization.

- **composite**: Handles `file://`, `https://` (default)
- Extensible for `s3://`, `gs://`, etc.

**Content regularization:**
- **PDF**: text extracted via `pypdf`; scanned pages (no extractable text)
  flagged for background OCR
- **HTML**: text extracted via BeautifulSoup (scripts/styles removed)
- **DOCX/PPTX**: text + tables/slides extracted via `python-docx` /
  `python-pptx`; auto-tags: author, title
- **Audio** (MP3, FLAC, OGG, WAV, AIFF, M4A, WMA): metadata via `tinytag`;
  auto-tags: artist, album, genre, year
- **Images** (JPEG, PNG, TIFF, WEBP): EXIF metadata via Pillow; auto-tags:
  dimensions, camera, date; flagged for background OCR
- **Other formats**: treated as plain text

Provider-extracted tags merge with user tags (user wins on collision). This
ensures both embedding and summarization receive clean text.

#### Content Extractor / OCR Providers
Extract text from scanned PDFs and images via optical character recognition.

- **mistral**: Cloud OCR via `mistral-ocr-latest` — high quality, images and
  PDFs (`MISTRAL_API_KEY`)
- **ollama**: Uses `glm-ocr` model (auto-pulled on first use)
- **mlx**: Apple Silicon — uses `mlx-vlm` vision models

OCR runs in the background via the pending queue (`keep daemon`), not
during `put()`. The flow:

1. During `put()`, content regularization detects scanned PDF pages (no
   extractable text) or image files
2. A placeholder is stored immediately so the item is indexed right away
3. The pages/image are enqueued for background OCR processing
4. `keep daemon` picks up the OCR task, renders pages to images, runs OCR,
   cleans and scores the text
5. The full OCR text replaces the placeholder and the item is re-embedded

Design points:
- Auto-detected: Ollama (with `glm-ocr`) > MLX > None. No configuration needed.
- Security: Pillow decompression bomb guard (250MP limit), PDF page cap
  (1000), temp directory cleanup
- OCR text is cleaned (whitespace normalized) and confidence-scored
- Graceful degradation: no OCR provider = metadata-only indexing

#### Media Description Providers (optional)
Generate text descriptions from media files, enriching metadata-only content.

- **mlx**: Apple Silicon — vision (`mlx-vlm`) + audio transcription
  (`mlx-whisper`)
- **ollama**: Local server — vision models only (`llava`, `moondream`,
  `bakllava`)

Media description runs in `Keeper.put()` between fetch and upsert.
Descriptions are appended to the metadata content before embedding/
summarization, making media files semantically searchable by their visual or
audio content.

Design points:
- Only triggered for non-text content types (`image/*`, `audio/*`)
- Lazy sub-provider loading: MLX composite only loads VLM for the first
  image, whisper for the first audio
- GPU-locked via `LockedMediaDescriber` (same file-lock pattern as
  summarization, see `model_lock.py`)
- Graceful degradation: errors never block indexing
- Optional dependency: `pip install keep-skill[media]` for MLX models

#### Analyzer Providers
Decompose content into structural parts with their own summaries, tags, and
embeddings (`analyzers.py` + `providers/base.py:AnalyzerProvider`).

- **SlidingWindowAnalyzer** (default): token-budgeted sliding windows with
  XML-style target marking, suited to small local models
- **SinglePassAnalyzer**: single-pass JSON decomposition for large-context
  models

Parts are produced by `analyze()` and stored as their own rows in
`document_parts`, with vectors at `{id}@p{N}` in the vector store.

#### Other provider modules
- **[hermes/](keep/hermes/)** — Hermes provider package (alternative
  inference backend)
- **[providers/embedding_cache.py](keep/providers/embedding_cache.py)** —
  `CachingEmbeddingProvider` wrapper used by `ProviderLifecycleMixin`
- **[providers/url_validation.py](keep/providers/url_validation.py)** —
  shared URL validation for HTTP-based providers

---

## Storage Layout

```
store_path/                   # default: ~/.keep
├── keep.toml                 # Provider and store configuration
├── documents.db              # SQLite: summaries, tags, versions, parts, FTS
├── chroma/                   # ChromaDB persistence (vectors + metadata)
├── pending_summaries.db      # Pending queue (summarize/embed/ocr/reindex/analyze)
├── continuation.db           # Direct work queue + flow continuations
├── embedding_cache.db        # SQLite cache for embeddings
├── planner_stats.db          # Flow planner priors
├── .processor.pid            # Daemon PID file
├── .processor.token          # Daemon HTTP auth token
├── .processor.port           # Daemon HTTP port
├── .processor.version        # Code version the daemon was started under
└── keep-ops.log[.N]          # Persistent operations log (rotating)
```

`documents.db` contains the `documents`, `document_versions`, and
`document_parts` tables (plus FTS shadow tables). The Chroma directory uses
ChromaDB's own on-disk format (sqlite + parquet segment files); keep does
not impose its own structure on it.

---

## Data Flow

### Indexing: `put(uri=…)` or `put(content=…)`

```
URI or content
    │
    ▼
┌─────────────────┐
│ Fetch / use     │ ← DocumentProvider (for URIs only)
│ input           │
└────────┬────────┘
         │ raw bytes
         ▼
┌─────────────────┐
│ Content         │ ← Extract text from HTML/PDF/DOCX/PPTX
│ regularization  │   (scripts/styles removed; scanned pages flagged)
└────────┬────────┘
         │ clean text (+ OCR page list if scanned)
         ▼
┌─────────────────┐
│ Media           │ ← Optional: vision description (images)
│ enrichment      │   or transcription (audio) appended
└────────┬────────┘
         │ enriched text
         ▼
┌──────────────────────────────────────────────┐
│ DocumentStore.upsert + placeholder summary   │
│ - tags, timestamps, content hash             │
│ - previous version archived if updated       │
└─────────────┬────────────────────────────────┘
              │
              ├─► PendingQueue.enqueue("summarize")
              ├─► PendingQueue.enqueue("embed")
              └─► PendingQueue.enqueue("ocr")  (if scanned PDF or image)
                                  │
                                  ▼
                        ┌──────────────────────┐
                        │ Background processor │
                        │ (pending_summaries / │
                        │  work_processor)     │
                        └──────────┬───────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
           summarize()         embed()             OCR
                │                  │                  │
                ▼                  ▼                  ▼
        DocumentStore.        VectorStore.       DocumentStore
        update_summary        upsert /           re-summarize +
                              upsert_version     re-embed
```

**Versioning on update**
- DocumentStore archives the current version before updating
- VectorStore adds a versioned embedding (`{id}@v{N}`) if content changed
- Same content (hash match) skips duplicate embedding

**Embedding dedup**
- Before computing an embedding, the Keeper checks if another document has
  the same content hash
- If a donor exists with a compatible embedding, it copies that vector
  instead of re-embedding
- Safety: dimension check prevents cross-model contamination

### Retrieval: `find(query)`

```
query text
    │
    ▼
  embed()  ← EmbeddingProvider
    │
    │ query vector
    ▼
┌───────────────────┐
│ VectorStore       │
│ query_embedding() │ ← cosine similarity search
└─────────┬─────────┘
          │
          ▼ results with distance scores
    ┌──────────────┐
    │ Apply decay  │ ← Recency weighting (ACT-R style)
    │ score × 0.5^(days/half_life)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Date filter  │ ← Optional --since / --until
    └──────┬───────┘
           │
           ▼
    ┌────────────────────────────┐
    │ Augmentation               │ ← deep follow, RRF fusion,
    │ (SearchAugmentationMixin)  │   tag boosts (when applicable)
    └──────┬─────────────────────┘
           │
           ▼
    list[Item] (sorted by effective score)
```

`find` is also reachable via the flow runtime (`find` / `find-deep` state
docs), which is the path used by MCP and the LangChain retriever.

### Analyze: `analyze(id)`

```
content
    │
    ▼
┌──────────────────────┐
│ AnalyzerProvider     │ ← SlidingWindowAnalyzer (default) or
│ analyze(chunks, …)   │   SinglePassAnalyzer
└──────────┬───────────┘
           │ list[{summary, tags}]
           ▼
┌──────────────────────┐
│ Keeper.analyze       │ ← Wraps into PartInfo, persists, embeds
└──────────┬───────────┘
           │
           ├─► DocumentStore.upsert_part  (rows in document_parts)
           └─► VectorStore.upsert_part    ({id}@p{N})
```

### Delete / Revert

`delete(id)` is a flat removal:

```
delete(id)
    │
    ▼
DocumentStore.delete + VectorStore.delete
(versions removed by default; pass delete_versions=False to keep history)
```

`revert(id)` is a separate operation that restores the previous version, or
falls back to `delete(id)` when there is no history:

```
revert(id)
    │
    ▼
  max_version(id)
    │
    ├── 0 versions → delete(id)
    │
    └── N versions → restore previous
            │
            ├─ get archived embedding from VectorStore (id@vN)
            ├─ DocumentStore.restore_latest_version()
            │    (promote latest version row to current, delete version row)
            ├─ VectorStore.upsert restored embedding as current
            ├─ VectorStore.delete versioned entry (id@vN)
            └─ delete stale parts (parts of the discarded version)
```

`delete_version(id, offset)` removes a specific archived version by public
selector (1=previous, -1=oldest archived, etc.).

---

## Key Design Decisions

**1. Schema as Data**
- System configuration is stored as documents in the store (`.now`,
  `.tag/*`, `.prompt/*`, `.state/*`, `.meta/*`)
- Bundled system docs are installed on first use by `system_docs.py`; they
  are then editable like any other note
- Flow definitions (`.state/*`) are loaded from this same store at runtime,
  so behavior is data-driven

**2. Daemon-mediated state**
- All non-trivial state lives in the daemon process. Surface clients are
  stateless and short-lived.
- This keeps model loading, GPU locks, embedding caches, and pending-work
  state in one place
- **Exception:** the Hermes integration (`keep/hermes/`) constructs an
  in-process `Keeper` directly inside the Hermes runtime. Reads (search,
  get, prompt rendering) and the synchronous part of writes go through
  this in-process Keeper for latency reasons; background work (embeddings,
  summaries, analysis) is still handled by an auto-started daemon. See
  [HERMES-INTEGRATION.md](HERMES-INTEGRATION.md).

**3. Lazy Provider Loading**
- Providers are registered at first use, not import time
- Avoids crashes when optional dependencies are missing
- `ProviderLifecycleMixin` handles double-checked locking for thread safety
  and supports GPU-memory release

**4. Separation of Concerns**
- Storage backends are provider-agnostic (only know about vectors / metadata)
- Providers are storage-agnostic (only know about text → vectors)
- Protocols define the boundary; implementations are pluggable

**5. No Original Content Storage**
- Reduces storage size
- Forces meaningful summarization
- URIs can be re-fetched if needed

**6. Immutable Items**
- `Item` is a frozen dataclass
- Updates via `put()` return a new `Item`
- Prevents accidental mutation bugs

**7. System Tag Protection**
- Tags prefixed with `_` are system-managed
- Source tags are filtered before storage
- A separate `INTERNAL_TAGS` set (in `types.py`) hides tags that exist for
  efficient queries but aren't user-facing

**8. Document Versioning**
- All documents retain history automatically on update
- Previous versions archived in the SQLite `document_versions` table
- Content-addressed IDs for text updates enable versioning via tag changes
- Embeddings stored for all versions (enables temporal search)
- No auto-pruning: history is preserved indefinitely

**9. Version-Based Addressing**
- Versions addressed by offset from current: 0=current, 1=previous,
  2=two-ago
- CLI uses `@V{N}` syntax for shell composition: `keep get "doc:1@V{1}"`
- Display format (v0, v1, v2) matches retrieval offset (`-V 0`, `-V 1`,
  `-V 2`)
- Offset computation assumes `list_versions()` returns newest-first ordering
- Security: literal ID lookup runs before `@V{N}` parsing to prevent
  confusion attacks

**10. Flow as the stable boundary**
- The hosted/local boundary is `run_flow(state, params)`, not a fixed
  object API
- Public helpers like `put`/`find`/`get` invoke named state docs, so the
  same flow definitions run locally and in the hosted backend
- Async actions inside a synchronous flow throw `AsyncActionEncountered`,
  which the runtime catches to hand off to the background work queue

---

## LangChain / LangGraph Integration

The `keep.langchain` module provides framework adapters on top of the API
layer:

```
┌─────────────────────────────────────────────────────────────┐
│  LangChain Layer (keep/langchain/)                          │
│  - KeepStore           LangGraph BaseStore adapter          │
│  - KeepNotesToolkit    LangChain tools                      │
│  - KeepNotesRetriever  BaseRetriever with now-context       │
│  - KeepNotesMiddleware LCEL runnable for auto-injection     │
└──────────────────┬──────────────────────────────────────────┘
                   │ uses Keeper API
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Keeper (api.py) → daemon → store                           │
└─────────────────────────────────────────────────────────────┘
```

`KeepStore` maps LangGraph's namespace/key model to keep's tag system via
configurable `namespace_keys`. Namespace components become regular keep
tags, visible to CLI and all query methods. Tag filtering is a **pre-filter
on the vector search**, making tags suitable for data isolation (per-user,
per-project). See [LANGCHAIN-INTEGRATION.md](LANGCHAIN-INTEGRATION.md).

---

## Extension Points

**New Embedding or Summarization Provider**
1. Implement the provider protocol (`EmbeddingProvider` or
   `SummarizationProvider`) from `providers/base.py`
2. Register it in the provider registry (typically by importing your module
   so its `register_*` calls run)
3. Reference the provider by name in `keep.toml`

**New Analyzer**
- Implement `AnalyzerProvider.analyze()` in `providers/base.py`
- Register through the provider registry
- Selected by name in the store config

**New Store Backend**
- Implement the protocols in [protocol.py](keep/protocol.py):
  `VectorStoreProtocol`, `DocumentStoreProtocol`, `PendingQueueProtocol`
- Local: ChromaDB + SQLite (built-in)
- Remote: PostgreSQL + pgvector (the keepmem package, registered via
  `keep.backends` entry point)
- Register new backends via the `keep.backends` entry point in
  `pyproject.toml`

**New Flow / State Doc**
- Author a `.state/*` document with rules and actions
- Add any new actions to `actions/` and wire them into the action runner
- Invoke via `run_flow("your-state", params=…)` from clients or other flows

**Framework Integration**
- Implement adapters on top of the daemon HTTP API or the local Keeper
- Current: LangChain/LangGraph ([keep/langchain/](keep/langchain/))
- Pattern: map framework concepts to keep tags + search
