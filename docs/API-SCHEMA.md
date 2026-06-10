# Keep API Schema Reference

Concise reference for the keep memory API. Covers the data model, tools, parameter types, and return formats.

Interface: MCP (`keep_flow`, `keep_prompt`, `keep_help`), CLI (`keep <cmd>`), or Python (`Keeper`).

The MCP boundary is `keep_flow(state, params)` — every operation is a state-doc invocation. Examples below show the JSON payload that goes into the `params` field.

---

## Data Model

### Item

Every piece of stored content is an **item**.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (see ID Formats below) |
| `summary` | string | Generated or user-provided summary of the content |
| `tags` | `{key: value}` | Key-value metadata. Values are strings or lists of strings |
| `score` | float or null | Similarity score (0-1), present only in search results |

Items also carry system-managed timestamps in tags: `_created`, `_updated`, `_accessed`.

### Versions

New archived versions are created when `put` updates an existing item's content.
Tag-only updates (`tag`) and unchanged writes update in place.

| Selector | Meaning |
|----------|---------|
| `@V{0}` | Current version (default) |
| `@V{1}` | Previous version |
| `@V{N}` | N versions back |
| `@V{-1}` | Oldest archived version |

Append the selector to any ID: `%a1b2c3@V{1}`

### Parts

`analyze` decomposes a document into structural **parts** — sections with their own summaries, tags, and embeddings.

| Selector | Meaning |
|----------|---------|
| `@P{1}` | First part (1-indexed) |
| `@P{N}` | Nth part |

Parts appear independently in search results. Retrieve with:

```json
keep_flow(state="get", params={"item_id": "DOC_ID@P{1}"})
```

---

## ID Formats

| Format | Example | Created by |
|--------|---------|------------|
| `%hexhash` | `%a1b2c3d4e5f6` | Inline text when `id` is omitted (CLI, MCP, Python API) |
| URL | `https://example.com/doc` | URI put |
| `file://` URI | `file:///path/to/doc.pdf` | Local file put |
| Custom string | `my-notes` | User-specified `id` parameter |
| `now` | `now` | Working context (singleton) |
| `.tag/KEY` | `.tag/act` | Tag description (system doc) |
| `.tag/KEY/VALUE` | `.tag/act/commitment` | Tag value description (system doc) |

---

## Tags

Key-value pairs on every item. Keys are alphanumeric (plus `_`, `-`). Values are strings or lists of strings.

### Setting and removing

```json
keep_flow(state="tag", params={"id": "ID", "tags": {"topic": "auth"}})
keep_flow(state="tag", params={"id": "ID", "tags": {"old-tag": ""}})
keep_flow(state="put", params={"content": "text", "tags": {"project": "myapp"}})
```

Empty string value removes the tag.

### Filtering

Tags on `find` and `list` are **pre-filters** — the search only considers matching items.

```json
keep_flow(state="query-resolve", params={"query": "auth", "tags": {"project": "myapp"}})
keep_flow(state="list", params={"tags": {"status": "open"}})
```

Multiple tags use AND logic: all must match.

### Built-in tags

| Key | Constrained | Singular | Values |
|-----|:-----------:|:--------:|--------|
| `act` | yes | yes | `commitment`, `request`, `offer`, `assertion`, `assessment`, `declaration` |
| `status` | yes | yes | `open`, `blocked`, `fulfilled`, `declined`, `withdrawn`, `renegotiated` |
| `type` | no | no | `learning`, `breakdown`, `gotcha`, `reference`, `teaching`, `meeting`, `pattern`, `possibility`, `decision` |
| `project` | no | no | user-defined |
| `topic` | no | no | user-defined |

**Constrained:** only listed values accepted. **Singular:** new values replace old (not accumulate).

### System tags (auto-managed, read-only)

`_created`, `_updated`, `_updated_date`, `_accessed`, `_accessed_date`, `_source`, `_content_type`

These are hidden from default display but accessible via `--json` or Python API. See [SYSTEM-TAGS.md](SYSTEM-TAGS.md) for the full list.

---

## Time Filters

Both `since` and `until` accept two formats:

| Format | Example | Meaning |
|--------|---------|---------|
| ISO 8601 duration | `P3D` | 3 days ago |
| | `P1W` | 1 week ago |
| | `P1M` | ~30 days ago |
| | `PT1H` | 1 hour ago |
| | `P1Y` | ~365 days ago |
| Date | `2026-01-15` | Specific date |

`since` = items updated on or after. `until` = items updated before.

---

## Tools

### put (state doc)

Store text, a URL, or a document.
For inline text without an explicit `id`, keep uses a content-addressed ID.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `content` | string | one of | — | Text to store |
| `uri` | string | one of | — | URI to fetch and index (`http://`, `https://`, `file://`) |
| `id` | string | no | auto | Custom ID. If omitted: URI inputs use the URI as ID; inline text uses a content hash |
| `summary` | string | no | auto | User-provided summary (skips auto-summarization) |
| `tags` | `{str: str}` | no | none | Tags to set. Example: `{"topic": "auth"}` |

Exactly one of `content` or `uri` is required. After storing, the `after-write` flow runs in the background to summarize, tag, analyze, and link the item — this is governed by `.state/after-write` and is not a parameter on `put`.

**Returns:** `"Stored: %a1b2c3"` or `"Unchanged: %a1b2c3"` (idempotent on same content)

**Examples:**

```json
keep_flow(state="put", params={"content": "OAuth2 uses PKCE for public clients", "tags": {"topic": "auth"}})
keep_flow(state="put", params={"uri": "https://docs.example.com/api", "tags": {"kind": "reference"}})
keep_flow(state="put", params={"content": "My design notes", "id": "design-notes", "summary": "Architecture decisions"})
```

---

### query-resolve (state doc)

Search memory by meaning. Returns items ranked by semantic similarity with recency weighting. This is the iterative entry point — it routes between `query-branch` and `query-explore` when results are ambiguous. See [FLOWS.md](FLOWS.md) for the state machine.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `query` | string | yes | — | Natural language search query |
| `tags` | `{str: str}` | no | none | Pre-filter: only search items matching all tags |
| `since` | string | no | none | Time filter (see Time Filters) |
| `until` | string | no | none | Time filter (see Time Filters) |
| `deep` | bool | no | false | Follow tags and edges to discover related items beyond direct matches |
| `deep_follow_depth` | int | no | 10 | With `deep`: how many top primary results contribute tags/edges to the follow (clamped to 1–100) |
| `limit` | int | no | 10 | Maximum results |

**Returns:** Formatted list of results, one per line:
```
- ID  (score) date  summary text...
```

**Examples:**

```json
keep_flow(state="query-resolve", params={"query": "authentication patterns"})
keep_flow(state="query-resolve", params={"query": "open tasks", "tags": {"project": "myapp"}, "since": "P7D"})
keep_flow(state="query-resolve", params={"query": "architecture decisions", "deep": true})
```

---

### get (state doc)

Retrieve one note in note-first form, with attached context. The result starts with the requested note's tags and body, then may include contextual sections such as similar notes, meta sections, structural parts, linked items, and version navigation.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `item_id` | string | yes | — | Note ID. Use `"now"` for current working context |

**Returns:** YAML frontmatter for the requested note plus any available context sections:

```yaml
---
id: %a1b2c3
tags:
  project: "myapp"
  topic: "auth"
similar:
  - %e5f6a7 (0.89) 2026-01-14 Related item summary...
meta/todo:
  - %d3e4f5 Open task related to this item...
parts:
  - @P{1} Section one summary...
edges/informs:
  - other-note  Related note surfaced via edge tags...
prev:
  - @V{1} 2026-01-13 Previous version summary...
---
Item summary or content here
```

The exact attached sections depend on what exists for the requested note. Tags are always part of the returned note shape; `similar`, `meta/*`, `parts`, `edges/*`, `prev`, and `next` appear when relevant.

**Examples:**

```json
keep_flow(state="get", params={"item_id": "now"})
keep_flow(state="get", params={"item_id": "%a1b2c3"})
keep_flow(state="get", params={"item_id": "%a1b2c3@V{1}"})
keep_flow(state="get", params={"item_id": "%a1b2c3@P{1}"})
keep_flow(state="get", params={"item_id": ".tag/act"})
```

---

### Updating now

The current working context is just an item with `id="now"`. To update it, use `put`:

```json
keep_flow(state="put", params={"id": "now", "content": "Investigating flaky auth test. Suspect timing issue."})
keep_flow(state="put", params={"id": "now", "content": "Fixed the bug. Next: add regression test.", "tags": {"project": "myapp"}})
```

Each call creates a new version of the `now` item. To **read** current context, use `keep_flow(state="get", params={"item_id": "now"})`.

---

### tag (state doc)

Add, update, or remove tags on an existing item. Does not re-process content.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `id` | string | yes | — | Item ID |
| `tags` | `{str: str}` | yes | — | Tags to set. Empty string `""` deletes the tag |

**Returns:** `"Tagged %abc: set topic=auth; removed old-tag"`

**Examples:**

```json
keep_flow(state="tag", params={"id": "%a1b2c3", "tags": {"status": "fulfilled"}})
keep_flow(state="tag", params={"id": "%a1b2c3", "tags": {"topic": "auth", "obsolete": ""}})
```

---

### delete (state doc)

Permanently delete an item and all its versions.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `id` | string | yes | — | Item ID to delete |

**Returns:** `"Deleted: %a1b2c3"` or `"Not found: %a1b2c3"`

```json
keep_flow(state="delete", params={"id": "%a1b2c3"})
```

---

### list (state doc)

List recent items. Supports filtering by ID prefix, tags, and time range. This is plain enumeration, distinct from `query-resolve` which does semantic search.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `prefix` | string | no | none | ID prefix or glob pattern (e.g. `".tag/*"`) |
| `tags` | `{str: str}` | no | none | Filter by tag key=value pairs |
| `tag_keys` | `[str]` | no | none | Filter by presence of tag keys (any value) |
| `since` | string | no | none | Time filter |
| `until` | string | no | none | Time filter |
| `order_by` | string | no | `updated` | Sort key: `updated`, `accessed`, `created`, `id` |
| `include_hidden` | bool | no | false | Include system notes (dot-prefix IDs) |
| `limit` | int | no | 20 | Maximum results |

**Returns:** List of items, one per line:
```
- ID  date  summary text...
```

**Examples:**

```json
keep_flow(state="list", params={})
keep_flow(state="list", params={"tags": {"act": "commitment", "status": "open"}})
keep_flow(state="list", params={"prefix": ".tag/", "include_hidden": true})
keep_flow(state="list", params={"since": "P7D", "limit": 20})
```

---

### move (state doc)

Move versions from a source item into a named target. Used to archive working context or reorganize notes.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `name` | string | yes | — | Target item ID (created if new, appended if exists) |
| `source` | string | no | `"now"` | Source item to extract from |
| `tags` | `{str: str}` | no | none | Only move versions whose tags match (all must match) |
| `only_current` | bool | no | false | Move only the tip version, not full history |

**Note:** The MCP/flow parameter is `source` (matching the wrapper at
`.state/move`). The Python API method `kp.move()` uses `source_id=` for the
same field — historical name, same meaning.

**Returns:** `"Moved to: my-notes"`

**Examples:**

```json
keep_flow(state="move", params={"name": "auth-work", "tags": {"project": "myapp"}})
keep_flow(state="move", params={"name": "design-log", "only_current": true})
keep_flow(state="move", params={"name": "topic-notes", "source": "old-doc", "tags": {"topic": "auth"}})
```

---

### keep_prompt

Render an agent prompt template with live context injected from memory. Templates use `{get}` and `{find}` placeholders that expand to current item context and search results, plus any bindings from a state-doc-backed prompt's flow.

Signature comes from `Keeper.render_prompt()` in `keep/_context_resolution.py`.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `name` | string | no | none | Prompt name. Omit to list available prompts |
| `text` | string | no | none | Search query for `{find}` placeholder / state doc `text` param |
| `id` | string | no | `"now"` | Item ID for `{get}` placeholder / state doc `item_id` param |
| `since` | string | no | none | Lower-bound time filter for search context |
| `until` | string | no | none | Upper-bound time filter for search context |
| `tags` | `{str: str}` | no | none | Tag filter for search context |
| `limit` | int | no | 10 | Max search results |
| `deep` | bool | no | false | Follow tags/edges to discover related items |
| `deep_follow_depth` | int | no | 10 | With `deep`: how many top primary results contribute tags/edges to the follow (clamped to 1–100) |
| `scope` | string | no | none | ID glob to constrain search results |
| `token_budget` | int | no | template default | Token budget for rendered context |

**Returns:** Rendered prompt text with placeholders expanded, or list of available prompts.

**Bundled prompts** (see `keep list .prompt/agent --all`):

| Name | Purpose |
|------|---------|
| `reflect` | Structured reflection on actions and outcomes |
| `session-start` | Context and open commitments at session start |
| `session-end` | End-of-session synthesis |
| `subagent-start` | Subagent initialization context |
| `query` | Answer a question using memory context |
| `conversation` | Conversation analysis |
| `system` | Generic reflective-memory practice (host-agnostic system prompt) |
| `system-hermes` | Hermes-specific system-prompt wrapper around `system` |
| `hermes-assemble` | Hermes per-turn context assembly |
| `openclaw-assemble` | OpenClaw per-turn context assembly |

The `system*` and `*-assemble` prompts are intended for host integrations (Hermes, OpenClaw) rather than direct agent invocation, but they're discoverable here for completeness. Custom prompts can be added under `.prompt/agent/`.

**Examples:**

```json
keep_prompt()
keep_prompt(name="reflect")
keep_prompt(name="session-start")
keep_prompt(name="query", text="what do I know about auth?")
keep_prompt(name="reflect", text="deployment", since="P3D")
keep_prompt(name="query", text="auth", tags={"project": "myapp"}, deep=true)
keep_prompt(name="query", text="api docs", scope="https://docs.example.com/*")
```

---

## Common Patterns

### Session lifecycle

```json
keep_prompt(name="session-start")
keep_flow(state="get", params={"item_id": "now"})
keep_flow(state="put", params={"id": "now", "content": "Completed X. Next: Y."})
keep_prompt(name="reflect")
```

### Store and retrieve

```json
keep_flow(state="put", params={"content": "insight text", "tags": {"kind": "learning", "topic": "auth"}})
keep_flow(state="query-resolve", params={"query": "authentication insights"})
keep_flow(state="get", params={"item_id": "%returned_id"})
```

### Track commitments

```json
keep_flow(state="put", params={"content": "Will fix bug by Friday", "tags": {"act": "commitment", "status": "open"}})
keep_flow(state="list", params={"tags": {"act": "commitment", "status": "open"}})
keep_flow(state="tag", params={"id": "ID", "tags": {"status": "fulfilled"}})
```

### Index a document

```json
keep_flow(state="put", params={"uri": "https://docs.example.com/api", "tags": {"kind": "reference", "topic": "api"}})
keep_flow(state="query-resolve", params={"query": "API documentation"})
```

### Archive and pivot

```json
keep_flow(state="move", params={"name": "auth-work", "tags": {"project": "myapp"}})
keep_flow(state="put", params={"id": "now", "content": "Starting on database migration"})
```

---

## See Also

- [AGENT-GUIDE.md](AGENT-GUIDE.md) — Working session patterns and reflective practice
- [TAGGING.md](TAGGING.md) — Full tag system reference (speech acts, constraints, edge tags)
- [OUTPUT.md](OUTPUT.md) — How to read the YAML frontmatter output format
- [KEEP-MCP.md](KEEP-MCP.md) — MCP server setup and integration
- [REFERENCE.md](REFERENCE.md) — CLI command reference
