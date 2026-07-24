# Prompts

keep uses three families of prompt docs, all stored under `.prompt/`:

| Family | IDs | Used by |
|--------|-----|---------|
| `.prompt/summarize/` | `default`, `conversation`, ... | `keep put` (summarization) |
| `.prompt/analyze/` | `default`, `conversation`, ... | `keep analyze` (decomposition) |
| `.prompt/agent/` | `reflect`, `query`, `conversation`, `session-start`, ... | `keep prompt` CLI and `keep_prompt` MCP tool |

Summarize and analyze prompts replace the LLM system prompt for their respective operations. Agent prompts are rendered with context injection and returned as text — see [KEEP-PROMPT.md](KEEP-PROMPT.md) for the CLI and [KEEP-MCP.md](KEEP-MCP.md) for the MCP tool.

## Summarize and analyze prompts

Each summarize/analyze prompt doc has two parts:

1. **Match rules** — tag queries that determine when this prompt applies (same DSL as `.meta/*` docs)
2. **`## Prompt` section** — the actual system prompt text sent to the LLM

When a document is summarized or analyzed, keep scans the matching `.prompt/{type}/` docs, finds those whose match rules match the document's tags, and selects the most specific match (most rules matched). The `## Prompt` section from the winner is the system prompt sent to the model.

The prompt docs in the store are authoritative at runtime. Bundled prompt files are only the seed content used during system-doc migration. If the matching store prompt is missing or lacks a `## Prompt` section, the operation fails instead of silently falling back to Python defaults.

### Bundled

| ID | Match rule | Purpose |
|----|-----------|---------|
| `.prompt/summarize/default` | *(none — fallback)* | Default summarization prompt |
| `.prompt/summarize/conversation` | `type=conversation` | Preserves dates, names, facts from conversations |
| `.prompt/analyze/default` | *(none — fallback)* | Section-by-section extraction for arbitrary documents |
| `.prompt/analyze/conversation` | `type=conversation` | Fact extraction from conversations |
| `.prompt/analyze/paper` | `type=paper` | Structural decomposition of research papers along actual sections |

### Creating custom prompts

Create a new prompt doc with match rules targeting specific tags:

```bash
# Custom summarization for code documentation
keep put "$(cat <<'EOF'
topic=code

## Prompt

Summarize this code documentation in under 200 words.
Focus on: what the API does, key parameters, return values, and common pitfalls.
Begin with the function or class name.
EOF
)" --id .prompt/summarize/code
```

Match rules can combine multiple tags for higher specificity:

```bash
# Prompt for meeting notes in a specific project
keep put "$(cat <<'EOF'
kind=meeting project=myapp

## Prompt

Summarize this meeting in under 300 words.
Focus on decisions made, action items assigned, and deadlines mentioned.
List each action item with its owner.
EOF
)" --id .prompt/summarize/myapp-meetings
```

The most specific match wins — a prompt matching `kind=meeting project=myapp` (2 rules) beats one matching just `kind=meeting` (1 rule), which beats the default (0 rules).

## Agent prompts

Agent prompts live at `.prompt/agent/` and contain template placeholders (`{get}`, `{find}`, `{text}`) that are expanded with store context at render time. They are returned as text for the calling agent to follow — not sent to an LLM as a system prompt.

```bash
keep prompt reflect                      # Render the reflect prompt
keep prompt reflect "auth flow"          # With search context injected
keep prompt --list                       # List available agent prompts
```

### State-doc backed prompts

Dynamic agent prompts must reference a state doc via a `state` tag. The state doc flow runs and its bindings become `{binding_name}` placeholders in the template. This separates retrieval logic (state doc) from presentation (prompt template). A prompt that uses `{get}` or `{find}` without a `state` tag is invalid and fails at render time.

```bash
keep put "## Prompt
{similar}
Question: {text}
" --id .prompt/agent/my-query -t state=my-retrieval-flow -t context=prompt
```

See [KEEP-PROMPT.md](KEEP-PROMPT.md) for full usage, template placeholders, and state-doc binding details.

## Viewing prompt docs

```bash
keep get .prompt/summarize/default       # See a summarize/analyze prompt
keep get .prompt/agent/reflect           # See an agent prompt
keep list .prompt                        # All prompt docs
```

Prompt docs are editable and versioned like any other store document.

In hosted mode, `.prompt/*` notes are protected system notes. Editing them
requires an API key with the `admin` permission; a read/write key can render and
view prompts but cannot save changes to them.

## See Also

- [KEEP-PROMPT.md](KEEP-PROMPT.md) — Agent prompt CLI and template placeholders
- [TAGGING.md](TAGGING.md) — Tag descriptions and how `## Prompt` sections work in tag docs
- [META-TAGS.md](META-TAGS.md) — Contextual queries (same match-rule DSL)
- [KEEP-ANALYZE.md](KEEP-ANALYZE.md) — CLI reference for `keep analyze`
- [KEEP-PUT.md](KEEP-PUT.md) — Indexing documents (summarization)
- [REFERENCE.md](REFERENCE.md) — Quick reference index
