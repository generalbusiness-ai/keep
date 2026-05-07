# Trailmark Code Graph Ingest For File-Backed Notes

Date: 2026-04-23
Status: Proposed

## Problem

keep already indexes source files as notes, but the stored surface is still
mostly note-shaped: one note for the file, optional linear parts, plus ordinary
tag-driven edges discovered from links or explicit tags.

For source code, that is not enough. We want keep to retain the internal
structure of a file or repository:

- modules
- classes / structs / interfaces / contracts
- functions / methods
- relationships such as contains, calls, inherits, implements, imports

The existing Trailmark project already parses source code into this shape. The
goal is to reuse Trailmark's parsing and graph construction, but to represent
the result inside keep's own note and edge model rather than adopting
Trailmark's storage/query layer.

This note assumes a deliberately narrow scope:

- only file-backed note IDs are supported
- inline notes that merely contain pasted code are out of scope
- the initial implementation targets code files first; whole-repository ingest
  is a later extension

## Scope

### In scope

- file-backed notes whose IDs are `file://...`
- parsing the file content at write time
- creating keep notes for code components
- creating keep edges for code relationships
- re-ingesting on file updates
- enough metadata on component notes to support later search, display, and
  traversal work

### Out of scope

- inline notes with code snippets
- remote URLs that happen to return source code
- adopting Trailmark's `GraphStore` or `QueryEngine` as keep runtime services
- full semantic summaries for every component in the first pass
- cross-file name resolution beyond what Trailmark already emits
- repository-wide UX in CLI/MCP for browsing code graphs

## Why Trailmark Is The Right Boundary

Trailmark already has the parse-time model we need:

- `CodeGraph`
  - `nodes`
  - `edges`
  - `annotations`
  - `entrypoints`
  - `subgraphs`
  - `dependencies`
- `CodeUnit`
  - kind
  - source location
  - parameters
  - return type
  - exception types
  - complexity
  - branches
  - docstring
- `CodeEdge`
  - source
  - target
  - edge kind
  - confidence
  - optional location

This is the useful part. Trailmark's `GraphStore` and `QueryEngine` are
downstream conveniences for traversal and JSON serialization. keep does not
need them for persistence.

The desired boundary is therefore:

```text
Trailmark parser -> CodeGraph -> keep materialization
```

not:

```text
Trailmark parser -> GraphStore -> QueryEngine -> keep
```

## Why This Should Not Use keep Parts

keep parts are a linear decomposition of a parent note. They are appropriate
for sections of a document, but they are the wrong abstraction for source
code graphs.

Code structure needs:

- arbitrary node kinds
- stable node identities
- arbitrary many-to-many edges
- cross-file references
- non-linear traversal

That maps naturally onto keep's ordinary note model plus edge tables, not onto
`@P{N}` part sidecars.

The design therefore treats code components as first-class notes, with the
original file note acting as the root.

## Design Summary

When keep writes or refreshes a file-backed source note, the after-write flow
should be able to invoke a new `analyze_code` action.

`analyze_code` will:

1. decide whether the target file is supported source code
2. parse the current file on disk through Trailmark
3. convert the resulting `CodeGraph` into keep mutations
4. replace the file's existing code-graph sidecar notes and edges
5. record a code-graph hash on the parent file note so unchanged files can
   skip re-ingest

The resulting keep representation is:

- one parent file note
- many component notes rooted under that file
- typed edges between those component notes

## Triggering

The natural trigger is the existing `after-write` flow.

This works for the scoped case because file-backed writes already provide the
needed information:

- note ID
- file URI
- content type
- raw content in write context

The action should be enabled by a new built-in fragment under
`.state/after-write/*`, guarded roughly as:

```yaml
- id: codegraph
  when: |
    item.uri.startsWith("file://") &&
    !item.id.startsWith(".") &&
    (
      item.content_type.startsWith("text/") ||
      item.content_type in [
        "application/javascript",
        "application/x-typescript"
      ]
    )
  do: analyze_code
  with:
    item_id: "{params.item_id}"
```

The action itself remains the final authority on whether the path is a
supported language. The state-doc guard should stay broad and cheap.

This also means the generic `analyze` fragment should not continue to run for
code files once codegraph ingest exists. Otherwise source files will get both:

- LLM-derived `@p{N}` parts
- Trailmark-derived code component notes

That dual decomposition is confusing and wastes background work. The built-in
`analyze` fragment should therefore be narrowed to skip recognized code content
types once `.state/after-write/codegraph` is present.

## Parsing Boundary

keep should not import Trailmark private helpers directly.

Instead, Trailmark should expose one small public parsing surface suitable for
external callers, for example:

```python
parse_path(path: str, language: str = "auto") -> CodeGraph
parse_file(path: str, language: str | None = None) -> CodeGraph
```

That public API should:

- select the parser
- support explicit language override
- support auto-detection
- return raw `CodeGraph`
- not require `GraphStore`

If Trailmark does not yet provide such a public helper, keep can begin with a
thin adapter module that imports concrete parser classes directly, but that is
an interim compatibility choice, not the desired long-term contract.

Because keep's `after-write` processing is asynchronous and keep does not store
full note bodies, the first implementation should parse the current file on
disk from the `file://` URI rather than attempting exact write-snapshot
parsing. That introduces a small race if the file changes again before the
background task runs, but it keeps the integration simple and aligned with
Trailmark's current path-based API.

## keep Data Model

### Parent file note

The existing file note remains the canonical root. New system tags track code
graph state, for example:

```text
_codegraph_provider: trailmark
_codegraph_language: python
_codegraph_hash: abc123...
_codegraph_node_count: 42
_codegraph_edge_count: 67
_codegraph_status: ready
```

The hash should be derived from the file content hash, not from mutable output
details, so that unchanged source content skips unnecessary rebuilds.

### Component note IDs

Each Trailmark node becomes a keep note whose ID is namespaced under the parent
file note, but it should **not** itself look like a `file://` URI. keep has
many generic branches that treat IDs beginning with `file://` as real file
backed items. A derived component note that still starts with `file://` would
be at risk of being mistaken for an on-disk file item.

The ID must therefore be:

- stable across re-ingest of unchanged code
- unambiguous under one parent file
- distinct from part/version suffix conventions
- clearly not a file-backed source URI

Proposed shape:

```text
codegraph:{file_note_id}:{trailmark_node_id}
```

Examples:

```text
codegraph:file:///repo/app/auth.py:auth
codegraph:file:///repo/app/auth.py:auth:User
codegraph:file:///repo/app/auth.py:auth:User.login
```

This preserves the original Trailmark node identity while keeping the parent
file note available in `_base_id` / `_codegraph_parent` tags. The exact string
form can change, but the important property is that these notes do not begin
with `file://`.

### Component note summaries

The first implementation should avoid LLM dependency here.

Each component note summary should be generated deterministically from:

- node kind
- name
- signature-like fields
- docstring first line if present
- source location

Examples:

- `Module auth`
- `Class User - Authentication user model`
- `Method User.login(username, password) -> Session`

This gives useful search/display output immediately. Richer prose can be added
later as an optional follow-up pass.

### Component note tags

Each component note should carry enough structured metadata to support search
and later rendering:

```text
_source: codegraph
_base_id: {file note id}
_codegraph_parent: {file note id}
code_kind: function | method | class | module | ...
code_language: python
code_file: file:///repo/app/auth.py
code_node_id: auth:User.login
code_name: login
code_parent: auth:User
code_module_id: auth
code_start_line: 10
code_end_line: 42
code_start_col: 4
code_end_col: 18
code_complexity: 7
code_return_type: Session
code_docstring: Authenticates a user
```

Parameters and exception types can be stored as normalized scalar-or-list tag
values where convenient. We do not need a perfect schema in v1 as long as the
mapping is stable and documented.

## Edge Model

Trailmark edges should become ordinary keep edges between component notes.

New built-in edge-tag docs should define the predicates and inverses:

```text
.tag/code_contains   -> _inverse: code_contained_by
.tag/code_calls      -> _inverse: code_called_by
.tag/code_inherits   -> _inverse: code_inherited_by
.tag/code_implements -> _inverse: code_implemented_by
.tag/code_imports    -> _inverse: code_imported_by
```

Each component note will store outgoing edge tags that point at component note
IDs. keep's existing edge machinery will materialize and index the inverse
edges.

In v1, only edges whose targets are also materialized in the same ingest batch
should be written. keep auto-vivifies missing edge targets, which is useful for
general notes but would create noisy stubs for unresolved code references.
External or unresolved targets should therefore be dropped or deferred until a
later phase.

Edge confidence from Trailmark should be preserved as metadata on the source
note when possible, for example:

```text
code_calls_confidence: certain
```

or, if needed later, in a dedicated sidecar record. For the first pass, it is
acceptable to preserve confidence only in component tags or derived summary
text rather than extending the core edge table schema.

## Materialization Flow

`analyze_code` should follow this sequence:

1. Resolve `item_id` and confirm it is file-backed.
2. Resolve the current file path from the `file://` URI.
3. Optionally consume write context for metadata, but parse the current file on
   disk in v1.
4. Determine language.
   - Prefer explicit file extension mapping.
   - Allow a future override tag such as `code_language`.
5. Parse through the Trailmark adapter.
6. Normalize the resulting `CodeGraph` into keep mutations.
7. Delete the existing codegraph child-note prefix for this parent.
8. Recreate component notes.
9. Apply edge tags on component notes so keep rebuilds the edge graph.
10. Update parent bookkeeping tags.

The action should be replace-in-full for one parent file, not incremental at
the per-node level in v1. The working assumption is that code files are small
enough that full replacement is simpler and safer than fine-grained diffing.

## Mutation Shape

The action should emit normal flow mutations rather than writing directly to
SQLite tables.

Expected mutation types:

- `delete_items_by_prefix` for the codegraph child-note namespace
- `put_item` for each component note
- `set_tags` for parent bookkeeping

This preserves the existing flow/runtime boundary and avoids a second write
path inside keep.

The current mutation runtime only has `delete_prefix` for `@p` parts. A generic
item-prefix deletion mutation therefore needs to be added before codegraph
replacement can be implemented cleanly.

## Re-ingest And Cleanup

Re-ingest is driven by the parent file note's content hash.

- if `_codegraph_hash == content_hash`, skip
- if content changed, rebuild the codegraph sidecar notes from scratch
- if parsing fails, keep the existing parent note and record failure tags

Suggested failure tags on the parent:

```text
_codegraph_status: failed
_codegraph_error: unsupported-language | parse-error | adapter-error
```

The action should avoid partial replacement. If parsing fails before mutations
are emitted, the previous codegraph notes remain intact.

## Search And Retrieval Behavior

This design intentionally reuses keep's existing note and edge behavior.

That means we get these properties without a new graph query runtime:

- component notes are searchable by summary and tags
- `get` on a component note can show inverse edges
- deep search can follow code edges once the predicates are defined
- dependency views can later be layered on top of `NoteDependencyService`

The parent file note remains the user's main entry point. A later CLI/MCP
surface can add explicit commands for:

- list code components for a file
- show callers/callees
- jump from file note to module/class/function notes

Those are follow-up UX tasks, not blockers for storage design.

## Language Detection

Because this design is file-backed-only, the initial language decision should be
path-based.

That keeps the first implementation simple:

- map file extension to Trailmark language name
- optionally confirm the file's `_content_type` is one of the known code MIME
  types already recognized by keep
- ignore unsupported extensions
- optionally allow `code_language` override tag for ambiguous cases

This avoids expensive content heuristics and aligns with how Trailmark already
detects languages.

## Repository-Level Extension Later

The first implementation should ingest one file note at a time.

Later, repository-level ingest can layer on top:

- a directory note for the repository root
- per-file codegraph ingest
- optional repo-level aggregate notes or edges
- optional whole-repo Trailmark parsing for better cross-file calls/imports

That later extension should reuse the same component note shape so file-level
and repo-level ingestion do not diverge.

## Risks

### Trailmark node ID collisions

Trailmark currently derives module IDs from file stems in at least some
parsers. That is acceptable for a single-file ingest path, but it may collide
for multi-file repo parsing if two files share the same stem.

This is not a blocker for the scoped design because the parent file note ID is
part of every keep component note ID. It does matter if we later ingest a
whole repository into one combined Trailmark graph.

### Edge cardinality

A large codebase may create many component notes and edges. The initial scope
should therefore stay at file-backed note granularity, and should not
automatically expand one repo directory into a unified codegraph until scale
has been measured.

### Missing-target stub explosion

keep's edge-tag machinery auto-vivifies missing targets. That behavior is
useful for ordinary note linking, but it would create low-value stubs for code
references that are unresolved or out of scope. v1 should therefore only write
edges whose targets are being materialized in the same batch.

### after-write load

Code parsing adds more work to the background queue. The action should remain
cheap to skip and should only fire for supported file-backed source notes.

## Implementation Sequence

1. Add a Trailmark adapter module in keep.
   - public function: parse file path -> normalized keep-side graph DTO
2. Add a generic mutation/runtime operation for deleting non-part child notes
   by prefix.
3. Add built-in tag docs for code edge predicates.
4. Add `actions/analyze_code.py`.
5. Add built-in `.state/after-write/codegraph` fragment.
6. Narrow the built-in `.state/after-write/analyze` fragment so code content
   types skip normal part decomposition.
7. Add parent bookkeeping tags and skip logic.
8. Add tests:
   - supported file creates component notes
   - updates replace old sidecar notes
   - unsupported file no-ops
   - parse failure records failure status without deleting existing graph
   - edges materialize in keep and are visible through `get_context`
   - unresolved external targets do not auto-vivify junk stubs
9. Add docs for the new behavior and note shape.

## Open Questions

### Where should the adapter live?

Preferred: `keep/integrations/trailmark.py` or similar, clearly outside core
document-store code.

### Should component notes be embedded?

Probably yes, at least initially, because ordinary non-part `put_item`
mutations go through the normal note storage path and receive embeddings from
their summaries. That is acceptable for v1. If volume or noise becomes a
problem later, codegraph notes can get a dedicated storage path or an explicit
"do not embed" flag.

### Should we store full signatures in tags?

Likely yes, as plain strings. We do not need a normalized query language for
signatures yet.

### Should inferred edges be treated differently from certain edges?

Not at the storage layer in v1. Preserve confidence in tags or summaries first;
revisit only if search/traversal needs confidence-aware ranking.

## Decision

Proceed with a file-backed-only design in which Trailmark is used purely as a
parser and graph builder. keep remains the system of record and persists the
result as:

- one parent file note
- many code component notes
- ordinary keep edges between those component notes

with these v1 constraints:

- component notes must not have `file://` IDs
- generic item-prefix cleanup support must be added to the mutation runtime
- only edges to notes materialized in the same ingest batch are emitted
- code files should skip the normal `analyze` parts flow

This fits keep's existing flow architecture, avoids Trailmark's storage model,
and gives a concrete path to implementation without committing to repo-wide
graph services yet.
