# Hermes Agent Integration

Keep provides a memory provider plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/). Once configured, Hermes gets persistent reflective memory across sessions — semantic search, conversation versioning, and agent prompts.

If you have Ollama running locally, it will be auto-detected. For other providers, put the appropriate API keys in your Hermes environment first. Supports multiple independent profiles.

## Install

```bash
curl -sSL https://keep.generalbusiness.ai/scripts/install-hermes.sh | bash
```

This finds your Hermes installation, installs the plugin, and runs the setup wizard. Choose your embedding and summarization providers when prompted.

Or manually:

```bash
# 1. Copy the plugin into Hermes
cp -r hermes-plugin /path/to/hermes-agent/plugins/memory/keep

# 2. Run the setup wizard — installs keep-skill, configures providers
hermes memory setup

# 3. Start a new Hermes session
hermes
```

When Hermes starts, ask it:

> "Follow the keep instructions in your system prompt."

## What the agent sees

The agent's system prompt is framed by a Hermes-specific wrapper (`.prompt/agent/system-hermes`) that renders alongside the built-in `memory` tool and the USER PROFILE / MEMORY blocks. It positions keep as the cross-session working memory (`now`) and long-term store, sitting above Hermes' pinned-essentials layer. The wrapper then includes the generic reflective-memory practice (`.prompt/agent/system`), so the same core instructions are available in any host.

On first session, the context includes step-by-step instructions to read the practice guide (`keep_help`), read the library (`keep_flow get`), and reflect (`keep_prompt reflect`). After the agent completes this and updates `now`, subsequent sessions show its own working context instead.

Three tools are available: `keep_flow` (all operations), `keep_help` (documentation), `keep_prompt` (context-injected prompts).

For `keep_flow`, the normal form is always `state + params`:

```text
keep_flow(state="get", params={"id": "now"})
keep_flow(state="list", params={"prefix": ".library", "include_hidden": true})
keep_flow(state="put", params={"id": "now", "content": "updated intentions"})
```

`keep_flow(state="get", ...)` returns the requested note in note-first form, including its tags and body, with contextual sections such as similar notes, related/meta sections, parts, and version navigation when available. It is not a bare exact-fetch API.

`state_doc_yaml` is available, but it is an advanced escape hatch for genuinely custom inline flows. Hermes agents, especially lighter models, are more reliable when examples stick to `state + params`.

## Agent behavior

Once the agent completes the initial practice (reading, reflecting, updating now), it typically generates its own workflow or checklist for using reflective memory effectively. This self-generated practice tends to be more durable than the initial instructions.

## Keep datastore

By default, the setup wizard creates the store at `$HERMES_HOME/keep` (e.g. `~/.hermes/keep`). This is written to `$HERMES_HOME/.env` as `KEEP_STORE_PATH`.

If `KEEP_STORE_PATH` is already set in the environment when Hermes starts, the provider uses that path instead. This allows sharing a store across profiles or pointing to an external store, but it overrides the per-profile default.

## Using keep CLI with the Hermes store

To use the `keep` standalone CLI, set `KEEP_STORE_PATH` to point at the store:

```bash
export KEEP_STORE_PATH=~/.hermes/keep
keep find "recent work"
keep get now
keep flow stats
```

When configuring external MCP hosts such as Codex or VS Code against the Hermes
store, first set `KEEP_STORE_PATH` to the Hermes store and then bake that value
into the launched command, for example `keep --store "$KEEP_STORE_PATH" mcp`.
Many MCP hosts do not inherit arbitrary shell environment variables into stdio
child processes.

## Architecture

- **In-process Keeper** handles reads (search, get, prompt rendering) and writes directly — no RPC overhead
- **Background daemon** (auto-started) handles embeddings and summaries asynchronously
- **Per-profile store** — defaults to `$HERMES_HOME/keep`; overridable via `KEEP_STORE_PATH`
- **Conversation versioning** — each turn is stored as a version of a per-channel item
