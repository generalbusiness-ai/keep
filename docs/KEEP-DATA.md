# keep data — Export, Import, and Sync

Backup, restore, and continuously mirror keep stores.

## Export

```bash
keep data export backup.json                           # JSON to file (default)
keep data export backup.json --include-system          # Include system docs (.tag/*, .meta/*, .now, etc.)
keep data export -                                     # JSON to stdout (for piping)
keep data export ~/vault --format md                   # Markdown: one .md per note
keep data export ~/vault --format md --include-parts   # ...plus analysis parts as sidecars
keep data export ~/vault --format md --include-versions # ...plus archived versions as sidecars
keep data export ~/vault --sync                        # Markdown + register continuous mirror
keep data export ~/vault --sync --stop                 # Stop mirroring (keeps files)
keep data export --list                                # List active sync directories
```

Exports all user documents, versions, and parts as JSON. **System documents (dot-prefix ids like `.tag/*`, `.meta/*`, `.now`) are excluded by default** — pass `--include-system` to include them. Embeddings are excluded (they are model-dependent and regenerated on import).

### Markdown mode (`--format md`)

Markdown mode writes a **directory** with one `.md` file per note. The directory is created if it doesn't exist; for a one-shot export it must be empty, but `--sync` allows writing into an existing directory.

One-shot markdown export uses the configured authoritative store. If a
`[remote]` backend is configured, `keep data export ~/vault --format md`
exports from that remote backend through the remote export and note-bundle
APIs. Continuous `--sync` also works with a remote backend, but the daemon
still owns the local mirror root and writes files only on this machine.

Each file has flat YAML frontmatter followed by the note summary as the body. The frontmatter is one flat map — no nested `tags:` block — with three kinds of keys:

- **Reserved export metadata** (underscore-prefixed, read-only): `_id`, `_content_hash`, `_content_hash_full`, `_created`, `_updated`, `_accessed`, `_part_num`, `_version`, `_version_offset`, `_prev_part`, `_next_part`, `_prev_version`, `_next_version`.
- **User and system tags** promoted to top-level keys: `topic`, `project`, `_source`, `_analyzed_hash`, etc.
- **Inverse-edge predicates** as multi-value YAML lists: `said`, `recipient_of`, `cited_by`, etc. Each value is the canonical labeled-ref form `[[source_id|display name]]` when a display name is available, or just the source id otherwise.

Filenames mirror the id's path structure for easy browsing, using the `wget -m` convention:

| Note id                                | Path in export dir                                            |
|----------------------------------------|---------------------------------------------------------------|
| `auth-notes`                           | `auth-notes.md`                                               |
| `notes/2024/jan-meeting`               | `notes/2024/jan-meeting.md`                                   |
| `.tag/act/commitment`                  | `.tag/act/commitment.md`                                      |
| `file:///Users/x/README.md`            | `file/Users/x/README.md.md`                                   |
| `https://example.com/docs/guide`       | `https/example.com/docs/guide.md`                             |
| `thread:abc-123@host.com#frag`         | `thread/abc-123@host.com%23frag.md`                           |
| `mailto:foo@bar.com`                   | `mailto/foo@bar.com.md`                                       |

Any RFC 3986 URI scheme becomes a top-level directory. Inside each component, filesystem-unsafe characters (`:`, `#`, `?`, `\`, `*`, `<`, `>`, `|`, non-ASCII) are percent-encoded; `@`, `+`, `=`, `,`, `(`, `)`, space stay literal. `.md` is always appended to the last component, even for ids that already end in `.md`, so the suffix is unambiguous. Components that exceed the filesystem's per-component limit are truncated with a short SHA256 suffix; the full id is always in `_id`.

Two notes whose paths would collide case-insensitively (e.g. `state-actions.md` and `STATE-ACTIONS.md` on macOS APFS) are auto-disambiguated: the second one gets an 8-hex-char hash suffix on its stem. The frontmatter `_id` is unchanged — disambiguation is purely an on-disk detail.

#### Example output

```markdown
---
_id: auth-notes
_content_hash: abc123
_content_hash_full: def456
_prev_version: "[[auth-notes/@V{1}]]"
_next_part: "[[auth-notes/@P{1}]]"
topic: auth
project: security
_source: inline
_created: '2026-01-15T10:30:00'
_updated: '2026-02-01T14:22:00'
_accessed: '2026-02-19T09:00:00'
_analyzed_hash: abc123
said:
- "[[conv1|First conversation about auth]]"
- "[[conv2|Follow-up on OAuth design]]"
---

Authentication patterns for OAuth2...
```

#### Chain navigation

Parent notes link to their sidecars via `_prev_version` and `_next_part` frontmatter keys. Sidecars link back to their parent (or to sibling sidecars) via the same system:

- **Versions** form a linear chain: parent → `@V{1}` → `@V{2}` → ... → oldest. Each version has `_next_version` (toward parent) and `_prev_version` (toward older).
- **Parts** form a linear chain by `part_num`: parent → `@P{1}` → `@P{2}` → `@P{5}`. Each part has `_prev_part` (toward parent or lower part_num) and `_next_part` (toward higher part_num).

All chain-navigation values are `[[vault-local-ref]]` wikilinks that resolve in tools like Obsidian.

#### Inverse edges

When a note is the target of edge-tag relationships (e.g. `speaker: Deborah` on a conversation note creates an edge to `Deborah`), the inverse predicates appear in the target's frontmatter as multi-value lists:

```yaml
said:
- "[[conv1|First conversation]]"
- "[[conv2|Second conversation]]"
recipient_of:
- "[[thread:abc@mail.com|Re: Meeting notes]]"
```

Values use canonical `[[target|label]]` labeled-ref syntax when the source note has a resolvable display name; otherwise they're plain ids. Forward edge tags (like `speaker: Deborah` on the source note) pass through as ordinary tag values — no special treatment.

#### Edge-tag value rewriting

Edge-tag values in the frontmatter are rewritten from canonical keep ids to the exported vault-local path namespace, so `[[wikilinks]]` in the frontmatter resolve correctly when the vault is opened in Obsidian or similar tools.

#### Parts and versions sidecars

When `--include-parts` or `--include-versions` is passed, notes that have analysis parts or archived versions get a sidecar **directory** alongside the parent file:

```
rust-tutorial.md             ← the current note (parent file)
rust-tutorial/               ← sidecar dir (only created if parts/versions exist)
  @P{1}.md                   ← analysis part 1
  @P{2}.md                   ← analysis part 2
  @V{1}.md                   ← previous version (1 step back from current)
  @V{2}.md                   ← 2 steps back
  @V{3}.md                   ← 3 steps back
```

- **`@P{N}.md`** — analysis part with `_part_num = N`. Body is the part text; frontmatter has `_id` (parent), `_part_num`, `_created`, and promoted part tags.
- **`@V{N}.md`** — archived version at offset `N` from current (`@V{1}` is the most recent prior). The current version stays in the parent file — there is no `@V{0}.md`. Frontmatter has `_id` (parent), `_version_offset`, `_version` (absolute database version number), `_created`, `_content_hash`, and promoted version tags.

Notes with no parts or versions get no sidecar dir — `plain-note.md` stays a single flat file even when both flags are on.

#### Using with Obsidian

The exported directory can be opened directly as an Obsidian vault. All `[[wikilink]]` values in the frontmatter (chain navigation, inverse edges, forward edge tags) resolve to the exported files. The Obsidian graph view renders the full relationship structure.

### Continuous sync (`--sync`)

```bash
keep data export ~/vault --sync                        # Export + register mirror
keep data export ~/vault --sync --include-parts        # ...with parts sidecars
keep data export ~/vault --sync --stop                 # Stop mirroring
keep data export --list                                # List active mirrors
```

`--sync` performs an immediate one-shot markdown export with progress, then registers the directory as a **daemon-owned continuous mirror**.

- With a local authoritative store, the daemon watches keep mutations (note creates, updates, deletes, tag changes, edge changes, part/version changes) via a trigger-based sync outbox and automatically re-exports affected note bundles on a debounced interval.
- With a remote authoritative store, the local daemon owns the mirror registration and local files. It polls the remote export change feed when available and rewrites only the affected note bundles for ordinary updates. Structural changes or feed gaps still trigger a debounced whole-mirror rebuild. If the remote endpoint does not support the change feed yet, the local daemon falls back to coarse interval-based full re-exports.

- **Incremental updates**: ordinary content, tag, and edge changes rewrite only the affected note bundles (the changed note plus any notes whose inverse-edge frontmatter depends on it). A note insert or delete triggers a full re-export pass.
- **Mirror state**: the exported directory contains a `.keep-sync/` subdirectory with `map.tsv` (vault-path → keep-id mapping, plaintext, diffable) and `state.json` (operational bookkeeping).
- **Mirror registration**: the daemon stores registered mirrors as local runtime state in the keep config directory (for example `~/.keep/markdown-mirrors.yaml`), not as notes in the authoritative store.
- **Remote cursors**: for remote authoritative stores, the local mirror runtime also stores the last observed remote change cursor locally.
- **Path exclusivity**: a sync directory cannot overlap with a `keep put --watch` directory, and vice versa. `keep put` of files inside a sync root is rejected.
- **Stop**: `--sync --stop` removes the mirror registration but does not delete the exported files.
- **List**: `--list` shows all active sync directories with their status (last run, pending, errors).

Check sync status:

```bash
keep daemon                    # Shows "Markdown mirrors active: N" when mirrors are registered
```

### Markdown mode vs JSON mode

| Feature | JSON (`--format json`) | Markdown (`--format md`) |
|---|---|---|
| Output | Single file | Directory of `.md` files |
| Round-trip import | Yes (`keep data import`) | Yes (`keep data import PATH --format md`) |
| Human browsable | No | Yes (grep, Obsidian, etc.) |
| Continuous sync | No | Yes (`--sync`) |
| Parts/versions | Always included | Opt-in (`--include-parts`, `--include-versions`) |
| Embeddings | Excluded | Excluded |

## Import

```bash
keep data import backup.json                 # Merge: skip existing IDs
keep data import backup.json --mode replace  # Replace: clear store first (prompts for confirmation)
keep data import -                           # Read from stdin
keep data import ~/vault --format md         # Recursive markdown import
keep data import ~/vault                     # Auto-detect dir/.md as markdown
```

Markdown import walks `.md` files recursively, honors `_id` and `_source_uri`,
imports top-level scalar tags and scalar lists, and skips exporter-owned
metadata such as `_content_hash`, `_version_offset`, and chain navigation
frontmatter. Keep-export sidecars (`@P{N}.md`, `@V{N}.md`) are restored as
parts and archived versions when present.

### Import Modes

- **merge** (default) — imports new documents, skips any with IDs that already exist in the target store
- **replace** — deletes all existing documents first, then imports (requires confirmation)

### After Import

Imported documents, versions, and parts are queued for re-embedding. Run:

```bash
keep daemon     # Process embeddings in background
```

Until embeddings are processed, imported documents are retrievable by ID (`keep get`) and visible in `keep list`, but won't appear in semantic search (`keep find`).

## Export Format

```json
{
  "format": "keep-export",
  "version": 3,
  "exported_at": "2026-02-19T12:00:00",
  "store_info": {
    "document_count": 42,
    "version_count": 120,
    "part_count": 15,
    "collection": "default"
  },
  "documents": [
    {
      "id": "auth-notes",
      "summary": "Authentication patterns for OAuth2...",
      "tags": {"topic": "auth", "_source": "inline"},
      "content_hash": "abc123",
      "created_at": "2026-01-15T10:30:00",
      "updated_at": "2026-02-01T14:22:00",
      "accessed_at": "2026-02-19T09:00:00",
      "versions": [
        {"version": 1, "summary": "...", "tags": {}, "content_hash": "...", "created_at": "..."}
      ],
      "parts": [
        {"part_num": 1, "summary": "...", "tags": {}, "created_at": "..."}
      ]
    }
  ]
}
```

**What's included:** document summaries, tags (including system tags like `_source`), timestamps, version history, structural parts.

**What's excluded:** embeddings (model-dependent, regenerated on import), store configuration (target uses its own).

## Use Cases

- **Backup:** `keep data export backup-$(date +%Y%m%d).json`
- **Browse in Obsidian:** `keep data export ~/vault --format md --include-parts --include-versions`, then open `~/vault` as a vault
- **Continuous Obsidian mirror:** `keep data export ~/vault --sync --include-parts` — the daemon keeps the vault up to date
- **Migrate local to cloud:** Export from local store, import into hosted store (when supported)
- **Transfer between machines:** Export, copy file, import
- **Merge stores:** Export from one, `keep data import --store /other/path backup.json`

## Python API

```python
from keep import Keeper

kp = Keeper()

# Streaming export — yields header, then one dict per document.
# Each document dict is self-contained: versions and parts are
# included inline (not yielded separately).
for i, chunk in enumerate(kp.export_iter()):
    if i == 0:
        header = chunk   # {"format", "version", "exported_at", "store_info"}
    else:
        doc = chunk      # {"id", "summary", "tags", ..., "versions": [...], "parts": [...]}

# Convenience: collect everything into a single dict (loads all into memory)
data = kp.export_data()                          # All documents
data = kp.export_data(include_system=False)       # Skip system docs

# Import
stats = kp.import_data(data, mode="merge")        # Skip existing
stats = kp.import_data(data, mode="replace")      # Clear first
stats = kp.import_markdown("~/vault", mode="merge")  # Recursive markdown import
# stats = {"imported": 10, "skipped": 2, "versions": 5, "parts": 3, "queued": 10}
```
