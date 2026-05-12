# keep put

Add or update a note in the store.

## Usage

Four input modes, auto-detected:

```bash
keep put "my note"                    # Text mode (inline content)
keep put file:///path/to/doc.pdf      # URI mode (fetch and index)
keep put https://example.com/page     # URI mode (web content)
keep put /path/to/folder/             # Directory mode (index all files)
keep put -                            # Stdin mode (explicit)
echo "piped content" | keep put       # Stdin mode (detected)
```

Directory mode indexes all regular files in the folder (non-recursive by default).

## Options

| Option | Description |
|--------|-------------|
| `-t`, `--tag KEY=VALUE` | Tag as key=value (repeatable) |
| `-i`, `--id ID` | Custom document ID (auto-generated for text/stdin) |
| `--summary TEXT` | User-provided summary (skips auto-summarization) |
| `-r`, `--recurse` | Recurse into subdirectories (directory mode) |
| `-x`, `--exclude PATTERN` | Glob pattern to exclude (repeatable, directory mode) |
| `--watch` | Set up a daemon watch — re-index automatically on file changes |
| `--unwatch` | Remove an existing watch |
| `--interval DURATION` | Polling interval for watches (ISO 8601 duration, e.g. `PT5M`) |
| `-f`, `--force` | Force re-index even if content is unchanged |

## Directory mode

Index a folder of files. By default non-recursive; use `-r` to include subdirectories:

```bash
keep put ./docs/                           # All files in docs/ (flat)
keep put ./docs/ -r                        # All files in docs/ (recursive)
keep put ./src/ -r -x "*.pyc" -x "__pycache__"  # Recursive with excludes
```

Excludes use glob patterns matched against the relative path from the directory root. Hidden files, symlinks, and nested Git repositories or worktrees are always skipped. If the directory root is itself a linked worktree, its files are indexed normally.

## Watching for changes

The `--watch` flag sets up a daemon-driven watch that re-indexes files automatically when they change:

```bash
keep put ./notes/ -r --watch               # Index + watch for changes
keep put ./notes/ -r --watch -x "*.log"    # With exclude patterns
keep put https://example.com/doc --watch   # Watch a URL for changes
```

Watches persist across sessions — the daemon polls for changes in the background. Use `keep daemon` to see active watches. Excludes are captured at watch-creation time.

For git repositories, directory watches also track the resolved `HEAD` commit. That means commit-only events such as empty commits, checkouts, branch switches, resets, and rebases can trigger reprocessing even when no watched file changed. Tag-only git changes are still not watched.

## Global ignore patterns

The `.ignore` system doc contains glob patterns that are automatically excluded from all directory walks and watches — in addition to `.gitignore` and per-watch `--exclude` patterns.

```bash
keep get .ignore                           # View current patterns
keep edit .ignore                          # Edit in $EDITOR
```

Updating `.ignore` retroactively purges matching `file://` items from the store and cancels their pending work. Ships with sensible defaults for build artifacts, lock files, bytecode, and binaries.

## Text mode and content-addressed IDs

Text mode uses content-addressed IDs for automatic versioning:

```bash
keep put "my note"              # Creates %a1b2c3d4e5f6
keep put "my note" -t done      # Same ID, new version (tag change)
keep put "different note"       # Different ID (new document)
```

Same content = same ID = enables versioning through tag changes.

## Smart summary behavior

- **Short content** (under `max_summary_length`, default 1000 chars): stored verbatim as its own summary
- **Long content**: truncated placeholder stored immediately, real summary generated in background by `keep daemon`
- **`--summary` provided**: used as-is, skips auto-summarization

The LLM prompt used for summarization is configurable. Create a `.prompt/summarize/*` document whose match rules target specific tags, and its `## Prompt` section replaces the default summarization prompt for matching documents. See [PROMPTS.md](PROMPTS.md) for details.

## Update behavior

When updating an existing note (same ID):
- **Summary**: replaced with new summary
- **Tags**: merged — existing tags preserved, new tags override on key collision
- **Version**: previous version archived automatically

If the new content matches an archived version of the same note, keep treats it as a restore of known content. The current head is still archived, but keep restores the prior summary, auto-tags, and head embedding for that content instead of recomputing them.

Analysis parts are not restored per version. If the current note already has parts from a later state, those parts stay attached until you run `keep analyze` again.

## Contextual summarization

When you provide tags during indexing, the summarizer uses context from related items to produce more relevant summaries.

1. System finds similar items sharing your tags
2. Items with more matching tags rank higher (+20% score boost per tag)
3. Top related summaries are passed as context to the LLM
4. Summary highlights relevance to that context

Tag changes trigger re-summarization:
```bash
keep put doc.pdf                       # Generic summary
keep put doc.pdf -t topic=auth         # Re-queued for contextual summary
```

## Git changelog

When a directory is a git repository, `put -r` queues the commit history for background indexing:

```bash
keep put ./myproject/ -r
# 42 indexed, 0 errors from myproject/
# git: changelog ingest queued
```

Each commit becomes a searchable item (ID: git://repo#sha) with the commit message as its summary. Files get a git\_commit edge tag linking to their last commit. Git tags and releases are indexed as separate items (ID: git://repo@tag).

**Incremental:** On re-scan (or via a watch), only new commits since the last ingest are processed. A `git_watermark` tag on the directory tracks the last ingested SHA. For watched repositories, keep also notices `HEAD` movement even if the working tree is unchanged, so commit messages are picked up promptly after commits and checkouts.

**Querying git history:**

```bash
keep find "why was the auth flow changed"     # Finds commit messages by meaning
keep find "auth" --deep                       # File results + linked commit context
keep list 'git://myproject#*'                 # All indexed commits
keep list 'git://myproject@*'                 # All indexed tags/releases
keep get 'git://myproject@v1.0'               # A specific release
```

## Supported formats

| Format | Extensions | Content extracted | Auto-tags |
|--------|-----------|-------------------|-----------|
| **Text** | .md, .txt, .py, .js, .json, .yaml, ... | Full text | — |
| **PDF** | .pdf | Text from all pages; scanned pages OCR'd in background† | — |
| **HTML** | .html, .htm | Text (scripts/styles removed) | — |
| **DOCX** | .docx | Paragraphs + tables | `author`, `title` |
| **PPTX** | .pptx | Slides + notes | `author`, `title` |
| **Audio** | .mp3, .flac, .ogg, .wav, .aiff, .m4a, .wma | Structured metadata (+ transcription\*) | `artist`, `album`, `genre`, `year`, `title` |
| **Images** | .jpg, .png, .tiff, .webp | EXIF metadata + OCR text† (+ description\*) | `dimensions`, `camera`, `date` |

\* When a media description provider is configured (`[media]` in `keep.toml`), images get vision-model descriptions and audio files get speech-to-text transcription, appended to the extracted metadata. See [QUICKSTART.md](QUICKSTART.md#media-description-optional) for setup.

† **OCR (optical character recognition):** Scanned PDF pages (pages with no extractable text) and all image files are automatically queued for background OCR when an OCR provider is available. Keep auto-detects Ollama (using `glm-ocr`, pulled automatically on first use) or MLX (`mlx-vlm` on Apple Silicon). A placeholder is stored immediately so the item is indexed right away; the full OCR text replaces it once background processing completes via `keep daemon`. No configuration needed — if Ollama is running, OCR just works.

Auto-extracted tags merge with user-provided tags. User tags win on collision:

```bash
keep put file:///path/to/song.mp3                    # Auto-tags: artist, album, genre, year
keep put file:///path/to/song.mp3 -t genre="Nu Jazz" # Overrides auto-extracted genre
keep put file:///path/to/photo.jpg -t topic=vacation  # Adds topic alongside auto camera/date
```

## Indexing documents

Index important documents encountered during work:

```bash
keep put "https://docs.example.com/auth" -t topic=auth -t project=myapp
keep put "file:///path/to/design.pdf" -t kind=reference -t topic=architecture
```

## See Also

- [TAGGING.md](TAGGING.md) — Tag system, merge order, speech acts
- [VERSIONING.md](VERSIONING.md) — How versioning works
- [KEEP-GET.md](KEEP-GET.md) — Retrieve indexed documents
- [META-TAGS.md](META-TAGS.md) — Contextual queries (`.meta/*`)
- [PROMPTS.md](PROMPTS.md) — Prompts for summarization, analysis, and agent workflows
- [REFERENCE.md](REFERENCE.md) — Quick reference index
