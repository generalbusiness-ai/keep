# keep mcp

MCP (Model Context Protocol) server for AI agent integration.

Provides MCP access to keep's reflective memory, using a local interface (stdio).

The server uses MCP 2026-07-28 when the client supports the final protocol and
keeps the 2025-11-25 initialization path for existing hosts. The SDK negotiates
the era; configuration is identical for both.

## Quick Start

```bash
export KEEP_STORE_PATH=~/.keep
keep --store "$KEEP_STORE_PATH" mcp
```

Many MCP hosts launch stdio servers with a scrubbed environment. If you use a
non-default store, prefer an explicit `--store` command or a host-specific
`KEEP_STORE_PATH` setting instead of assuming your shell environment will be
inherited.

### Claude Desktop

```bash
keep config mcpb              # Generate and open .mcpb bundle
```

Generates a `.mcpb` bundle and opens it with Claude Desktop. You will be prompted to install the `keep` connector, which gives Claude Desktop full access to the memory system and help pages.

### Claude Code

```
/plugin marketplace add https://github.com/generalbusiness-ai/keep.git
/plugin install keep@generalbusiness-ai
```

The first command registers the marketplace, the second installs the plugin (MCP tools, skill instructions, and session hooks).

Alternatively, add just the MCP server manually:

```bash
claude mcp add --scope user keep -- keep --store "$KEEP_STORE_PATH" mcp
```

### Kiro

```bash
kiro-cli mcp add --name keep --scope global -- keep --store "$KEEP_STORE_PATH" mcp
```

### Codex

`keep config --setup` auto-installs `[mcp_servers.keep]` into
`~/.codex/config.toml` with the resolved store path baked into `--store`.
Codex.app launched from Finder does not inherit your shell environment, so
the baked-in path is more reliable than `KEEP_STORE_PATH`.

To add the entry manually instead:

```bash
codex mcp add keep -- keep --store "$KEEP_STORE_PATH" mcp
```

Prefer the explicit `--store` form above. Use `--env KEEP_STORE_PATH=...`
only if you specifically need Codex to launch `keep mcp` without embedded
command arguments.

### VS Code

```bash
code --add-mcp "{\"name\":\"keep\",\"command\":\"keep\",\"args\":[\"--store\",\"$KEEP_STORE_PATH\",\"mcp\"]}"
```

The server respects the `KEEP_STORE_PATH` environment variable for store
location, but explicit `--store` is more reliable when the MCP host sanitizes
child-process environments.

## Maintainer Checks

For host integrations, `keep mcp` itself should stay boring: one stdio server,
same tools everywhere. What varies is the host-native attachment point.

Current native MCP attachment targets:

- `Codex` — `~/.codex/config.toml` with `[mcp_servers.keep]`
- `Claude Code` — `claude mcp add --scope user keep -- keep --store "$KEEP_STORE_PATH" mcp`
- `Kiro` — `kiro-cli mcp add --name keep --scope global -- keep --store "$KEEP_STORE_PATH" mcp`
- `GitHub Copilot CLI` — `~/.copilot/mcp-config.json`
- `VS Code` — workspace or user `mcp.json`
- `OpenClaw` — installed `keep` plugin bundle, including the bundled `.mcp.json`

Regular tests should cover only:

- `keep mcp` server behavior
- installer output shape

Occasional real-host checks should stay explicit and opt-in because they depend
on external CLIs, real auth, and host-side behavior. Use:

```bash
uv run python scripts/check_host_mcp.py --host codex --host claude --run
uv run python scripts/check_host_mcp.py --host kiro --run
uv run python scripts/check_host_mcp.py --host github_copilot
```

The script does not join the normal `pytest` run. It validates host-native
config locations and, when `--run` is supplied, runs the best available
host-side smoke command for each selected tool.

## Tools

Three tools:

| Tool | Description | Annotations |
|------|-------------|-------------|
| `keep_flow` | Run any operation as a state-doc flow | may mutate, may be destructive, not generally idempotent |
| `keep_prompt` | Render an agent prompt with context injected | read-only |
| `keep_help` | Browse keep documentation | read-only |

All operations (search, put, get, tag, delete, move, stats) go through `keep_flow` with named state docs. See [FLOW-ACTIONS.md](use keep_help with topic="flow-actions") for the full action reference.

Tool calls return both text content and typed `structuredContent`. A failed
flow is an MCP tool error (`isError: true`), so clients do not have to parse an
`Error:` string to distinguish failure from a successful result. List results
use private, zero-duration cache hints because prompt and note catalogs can
change between calls.

## Resources

The MCP server also exposes read-only note resources.

Concrete resource:

- `keep://now` — the current note as JSON

Resource template:

- `keep://{id}` — any keep note ID, percent-encoded when needed

Examples:

- `keep://now`
- `keep://meeting-notes`
- `keep://file%3A%2F%2F%2FUsers%2Fhugh%2Fnotes.md`
- `keep://https%3A%2F%2Fexample.com%2Fdoc`

Resource contents use the same JSON shape returned by `GET /v1/notes/{id}`.

### keep_flow

```
state:          "query-resolve"                    # state doc name
params:         {query: "auth", bias: {now: 0}}    # flow parameters
budget:         3                                   # max ticks
token_budget:   2000                                # token-budgeted rendering
cursor:         "abc123"                            # resume a stopped flow
state_doc_yaml: "..."                               # inline YAML (custom flows)
```

Common state docs:

| State | Purpose | Key params |
|-------|---------|------------|
| `query-resolve` | Search with multi-step refinement | `query`, `tags`, `bias`, `since`, `until` |
| `get` | Retrieve note-first output with tags plus similar/meta/versions/edges context | `item_id` |
| `find-deep` | Search with edge traversal | `query` |
| `put` | Store content or index a URI | `content` or `uri`, `tags`, `id` |
| `tag` | Apply tags to one or more notes | `id` or `items`, `tags` |
| `delete` | Remove a note | `id` |
| `move` | Move versions between notes | `name`, `source`, `tags` |
| `stats` | Store profiling for query planning | `top_k` |

### keep_prompt

```
name:   "reflect"                  # prompt name (omit to list available)
text:   "auth flow"                # optional search context
id:     "now"                      # note for context injection
tags:   {"project": "myapp"}       # filter search results
since:  "P7D"
scope:  "file:///path/to/dir*"     # constrain results to ID glob
```

Returns the rendered prompt with placeholders expanded. Supports `{get}`, `{find}`, `{text}`, and `{binding_name}` placeholders (when the prompt doc has a `state` tag referencing a state doc flow). See [KEEP-PROMPT.md](use keep_help with topic="keep-prompt") for prompt details.

## MCP Prompts

The MCP server also exposes selected `.prompt/agent/*` notes as MCP Prompts. Exposure is store-driven: a prompt doc appears in MCP prompt listings only when it has an `mcp_prompt` tag.

Example prompt-doc tag:

```yaml
tags:
  context: prompt
  state: get
  mcp_prompt: text,id,since,token_budget
```

Supported MCP Prompt arguments are intentionally narrow and all optional:

- `text`
- `id`
- `since`
- `token_budget`

These MCP Prompts are thin wrappers over the same backend render path used by the `keep_prompt` tool. The tool remains the broader surface when you need `until`, `tags`, `deep`, or `scope`.

### keep_help

```
topic:  "index"                    # documentation topic (default: index)
```

## Agent Workflow

A typical session: orient, search, capture, reflect, update working
context. Each step is one tool call:

1. `keep_prompt(name="session-start")`
2. `keep_flow(state="query-resolve", params={"query": "topic"}, token_budget=2000)`
3. `keep_flow(state="put", params={"content": "insight", "tags": {"kind": "learning"}})`
4. `keep_prompt(name="reflect")`
5. `keep_flow(state="put", params={"id": "now", "content": "next steps"})`

## Concurrency

The MCP stdio server is a thin HTTP client to the daemon. The daemon uses a
`ThreadingHTTPServer` (one thread per request) and the underlying SQLite +
ChromaDB stores handle their own locking. Cross-process safety (multiple
agents sharing a store) is handled at the store layer.

## See Also

- [FLOW-ACTIONS.md](use keep_help with topic="flow-actions") — Action reference for all operations
- [KEEP-FLOW.md](use keep_help with topic="keep-flow") — Running, resuming, and steering flows
- [KEEP-PROMPT.md](use keep_help with topic="keep-prompt") — Agent prompts with context injection
- [AGENT-GUIDE.md](use keep_help with topic="agent-guide") — Working session patterns
