# keep edit

Edit a note's content in your terminal editor.

## Usage

```bash
keep edit <id>
```

Opens the current content of the note in `$EDITOR` (or `$VISUAL`, falling back to `vi`). When you save and close the editor, the note is updated if the content changed.

## Examples

```bash
keep edit .ignore                    # Edit global ignore patterns
keep edit .prompt/agent/reflect      # Edit a prompt template
keep edit now                        # Edit current intentions
keep edit %a1b2c3d4                  # Edit an inline note
EDITOR=code keep edit .ignore        # Use VS Code
```

## How it works

1. Reads the current stored text of the note
2. Writes it to a temporary file (`.md` by default)
3. Opens the file in your editor
4. On save, compares with the original — if changed, calls `put` to update
5. The temp file is cleaned up automatically

System docs (`.ignore`, `.prompt/*`, `.state/*`) store their full content as the summary, so `keep edit` gives you the complete document.

When using the hosted service, editing any dot-prefixed system note requires an
API key with the `admin` permission. A read/write key can view these notes but
receives `403 Forbidden` when saving them. Select `admin` when creating the key,
configure keep to use that key, and then rerun `keep edit`.

## See Also

- [KEEP-PUT.md](KEEP-PUT.md) — Creating and updating notes
- [KEEP-GET.md](KEEP-GET.md) — Viewing notes
- [REFERENCE.md](REFERENCE.md) — Quick reference index
