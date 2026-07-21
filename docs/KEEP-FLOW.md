# keep flow

Most flows in keep run behind the scenes — `put` triggers background
processing, `get` assembles context, `find` resolves queries. You never
see them. But sometimes you want to run a flow directly: re-process a
note, run a custom workflow, or pick up where an ambiguous query left off.

`keep flow` is the command for that. It runs any state doc — built-in or
custom — and gives you the result. If the flow stops partway (budget
exhausted, ambiguous results), you get a cursor to resume it. You can
adjust parameters between calls to steer the flow in a different direction.

For how flows work internally, see [FLOWS.md](FLOWS.md).
For built-in state doc reference, see [FLOW_STATE_DOCS.md](FLOW_STATE_DOCS.md).

## Running built-in state docs

Every flow is defined by a state doc stored as `.state/<name>`. Run one
by name:

```bash
# Re-process a note (summarize, tag, analyze)
keep flow after-write --target %abc123

# Assemble display context
keep flow get --target myproject

# Deep search with tag-edge traversal
keep flow find-deep -p query="OAuth2 design" -p limit=10

# Multi-step query resolution with custom thresholds
keep flow query-resolve -p query="auth patterns" -p margin_high=0.15
```

The `--target` flag sets the target note (`params.id`), making it
available to all actions in the flow. The `-p` flag sets arbitrary
parameters — thresholds, queries, limits — that the state doc's rules
reference.

## Running custom state docs

You can write your own state docs and run them from files. This is how
you build custom workflows: bulk operations, periodic reviews, data
pipelines — anything that combines keep's actions (find, get, tag,
summarize, analyze, traverse) into a sequence with conditions.

```yaml
# review.yaml — find all draft notes
match: sequence
rules:
  - id: drafts
    do: find
    with:
      query: ""
      tags: { status: draft }
      limit: 20
  - when: "drafts.count == 0"
    return:
      status: done
      with: { message: "No drafts found" }
  - return:
      status: done
      with: { drafts: "$drafts.results" }
```

```bash
keep flow --file review.yaml
```

### Piping from stdin

Agents can generate state docs on the fly and pipe them in:

```bash
cat <<'YAML' | keep flow --file - --target myproject
match: all
rules:
  - id: similar
    do: find
    with:
      similar_to: "{params.id}"
      limit: 5
  - id: tags
    do: get
    with:
      id: "{params.id}"
YAML
```

The `--file -` flag reads YAML from stdin. When combined with
`--target`, the piped state doc can reference `params.id` to operate
on the specified note.

## Resuming and steering

Flows don't always finish in one call. A multi-step query might exhaust
its budget. A branching search might stop with ambiguous results that
need human or agent judgment. When this happens, the response includes a
**cursor** — a token encoding where the flow stopped and what it found
so far.

The key insight: the cursor carries the flow's state, but **parameters
are always fresh**. You provide them on each call. This is how you
steer — review partial results, then adjust the query, thresholds, or
strategy before resuming.

```bash
# Start a query resolution flow
RESULT=$(keep flow query-resolve -p query="auth" --budget 3 --json)

# Flow stopped — check what it found
echo "$RESULT" | jq .status
# → "stopped"

# Results are too scattered — suppress noisy items with bias and resume
CURSOR=$(echo "$RESULT" | jq -r .cursor)
keep flow --cursor "$CURSOR" -p query="auth" -p 'bias={"now": 0}' --budget 5

# Or refine the query with tag filter and time range
keep flow query-resolve -p query="auth" -p 'tags={"project": "myapp"}' -p since=P7D

# Or refine the query entirely
keep flow --cursor "$CURSOR" -p query="OAuth2 token refresh" --budget 5
```

**Bias** controls per-item score weighting: `0`=exclude, `1`=neutral, `>1`=boost.
Common pattern: `bias={"now": 0}` suppresses the working context from results.

Each resume gets a fresh budget. The cursor's tick count is historical
(for diagnostics), not a remaining balance — `--budget 5` always means
5 new ticks.

### What's in a cursor

Cursors are opaque tokens — pass them to `--cursor` to resume, but
don't parse or construct them. The format may change between versions.

They are self-contained (no database, no server state), so they work
across CLI invocations, MCP calls, and piped workflows. You can store
them, pass them between processes, or discard them.

Three things are deliberately *not* in the cursor — they come from
the caller each time:

- **Params** — the caller's intent (query, thresholds, target note)
- **Budget** — resource allocation for this invocation
- **State doc source** — which flow definition to use

This separation means the flow remembers what it found, but the caller
always controls what to search for and how far to go.

> **Implementation detail:** cursors are currently base64url-encoded JSON
> containing the state name, tick count, and accumulated bindings.

## Composing flows

Flows return structured JSON, so they compose with other commands:

```bash
# Re-process all project notes
for id in $(keep list project/ --json | jq -r '.notes[].id'); do
  keep flow after-write --target "$id"
done

# Use flow output to drive further actions
RESULT=$(keep flow query-resolve -p query="deployment checklist" --json)
echo "$RESULT" | jq -r '.bindings.search.results[].id' | while read id; do
  keep get "$id"
done
```

## Error handling

When a flow errors, the response has `"status": "error"` and the
`data` field contains details. Errored flows have no cursor — they
cannot be resumed.

```bash
RESULT=$(keep flow nonexistent-state --json 2>/dev/null)
echo "$RESULT" | jq .status
# → "error"
```

---

## Command reference

### CLI: `keep flow`

```
keep flow <state> [options]           # run a stored state doc
keep flow --file <path> [options]     # run from a YAML file
keep flow --file - [options]          # run from stdin
keep flow --cursor <token> [options]  # resume a stopped flow
```

| Flag | Short | Description |
|------|-------|-------------|
| `--target ID` | `-t` | Target note ID (sets `params.id`) |
| `--budget N` | `-b` | Max ticks this invocation (default: from config) |
| `--token-budget N` | `--tokens` | Render agent-readable output within N tokens |
| `--cursor TOKEN` | `-c` | Resume a stopped flow from its cursor |
| `--file PATH` | `-f` | Load state doc from file, or `-` for stdin |
| `--param key=value` | `-p` | Set a flow parameter (repeatable) |

By default, the CLI emits the raw JSON response described below. Pass
`--token-budget N` for the same
token-budgeted text rendering available through MCP. `--budget` is separate:
it limits flow execution ticks, not response length. `--json` preserves the
raw response envelope when used with `--token-budget`.

| Field | Present | Description |
|-------|---------|-------------|
| `status` | Always | `done`, `error`, or `stopped` |
| `ticks` | Always | Number of ticks consumed |
| `data` | When flow returns data | Payload from `return.with` (or all bindings if `with:` omitted) |
| `cursor` | When `stopped` | Short ID to resume the flow (stored server-side) |
| `tried_queries` | When searches were run | Queries attempted across all ticks |

### MCP: `keep_flow`

```python
keep_flow(
    state="query-resolve",
    params={"query": "auth patterns", "bias": {"now": 0}, "since": "P7D"},
    budget=5,
    token_budget=2000,     # token-budgeted text rendering (omit for raw JSON)
    cursor=None,           # or a cursor ID from a previous call
    state_doc_yaml=None,   # or inline YAML string
)
```

When `token_budget` is set, the response is rendered as token-budgeted
text (using the same algorithm as search result rendering). When omitted,
the response is raw JSON. The `state_doc_yaml` parameter is the MCP
equivalent of `--file` — pass inline YAML instead of a state name.

## See also

- [FLOWS.md](use keep_help with topic="flows") — How flows work, state transitions, background processing
- [FLOW-ACTIONS.md](use keep_help with topic="flow-actions") — Available actions reference
- [FLOW_STATE_DOCS.md](use keep_help with topic="flow_state_docs") — Built-in state doc reference
