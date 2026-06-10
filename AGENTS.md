# Agent Instructions

Design documents (including historical drafts) are in `later/`.  User documents are in `docs/`.

User-facing terminology:
* "notes" (not "items" or "documents" or whatever).
* "stubs" (not "auto-vivify", although that's fine for code and comments)

## Quality

- Be observant.  Check everything that you see in prompts and context.  Unexpected or unexplained behavior might be a bug.  Bugs must be investigated.  Investigations must identify the actual cause.
- Investigate thoroughly until the actual behavior is confirmed.  We have OTel instrumentation and debug logging.  If those are not sufficient for debugging, then they must be improved.  Don't write new code until you have exhaused all other options.  When replacing old code, clean up thoroughly.
- Take ownership now.  When there is a problem, it must be skillfully resolved.  There are no "pre-existing issues".  The cavalry will not arrive.  Track and fix problems when they are observed.  "Bigger changes" must never be deferred.  Do hard things carefully.
- Take care now.  Code is not ready until it has been reviewed for security, performance, consistency and correctness.  Code is not ready until it has been thoroughly tested.  Features require tests that exercise the feature in full, and that will fail if the user-visible behavior breaks.  Code is not ready until the user docs are accurate.
- Comments are required.  Explain to the next reader, to make their work easier.

## Modular Concerns

- The daemon manages ongoing work, and prioritizes it appropriately to ensure responsiveness.
- When using hosted services, the daemon only does local work such as file-watching, and the hosted services perform background work.
- CLI and MCP are thin wrappers over the daemon's services; they are concerned with shell and agent invocation, nothing more.
- Hooks are responsible for harness-specific functionality and formats; those are not the concern of the core.
- The API-based model providers are pluggable.  Provider-specific functionality is not the concern of the core.
- The datastore is pluggable.  Datastore-specific functionality is not the concern of the core.

## Development environment

- **Editable install**: In development mode, `keep-skill` is installed via `uv tool install --editable`.
  The uv Python path does NOT mean it's running old code — it runs whatever is on disk.
- **Daemon version**: check `~/.keep/.processor.version` to see what version
  the running daemon was started under. If it's old, the daemon needs a restart
  to pick up new code (`keep pending --stop` then let it auto-start).
- **Never run `uv tool upgrade` or `pip install`** — the editable install already tracks HEAD.

## Pre-commit hooks

- The repo uses [`pre-commit`](https://pre-commit.com); config lives in `.pre-commit-config.yaml` (all versions pinned).
- Current hooks:
  - **gitleaks** — blocks accidental secret disclosure
  - **ruff-check** — same lint gate enforced at release time, now run per-commit
  - **check-merge-conflict** — catches unresolved conflict markers
  - **check-added-large-files** — rejects files over 1 MB to keep the repo lean
- Install once per clone: `uv tool install pre-commit && pre-commit install`. This wires `.git/hooks/pre-commit` so every commit runs the hooks against staged changes.
- Ad-hoc full-tree scan: `pre-commit run --all-files`. Full git-history secret scan: `gitleaks detect --no-banner` (requires system gitleaks, e.g. `brew install gitleaks`).
- **Never bypass the hooks with `--no-verify`** unless the user explicitly asks. If gitleaks flags something, investigate: if it's a real secret, rotate it and remove it from history; if it's a false positive, add a narrow entry to `.gitleaksignore` (by fingerprint) or a rule allowlist in `.gitleaks.toml`. If ruff flags something, fix the lint.
- CI backstop: `.github/workflows/secret-scan.yml` runs gitleaks against the full git history on every push to `main` and every PR, in case a contributor skipped or bypassed the local hook. The CI gitleaks version is kept in sync with the pre-commit pin.

## Release process

- Feature work should be on its own worktree/branch.
- Do not commit to main, tag, push, or release without explicit user instruction.
- Major version is zero.  Minor version is updated with feature releases.  Patch version for maintenance.
- Use `scripts/release.sh` to release, gated by the checklist below.

**Before any release**

* If recent work was implementing a written design doc (`later/design/`), ensure that doc reflects the current status.
* Review all `docs/` for completeness and accuracy.
* Run `ruff check` (on all files) and fix all issues.
* Run `cve-lite --verbose` and fix all issues within range.
* Check CI for existing alerts.  If found: address them, and re-check by running the CI audit locally.
* Review all previously-ignored CVEs and other suppressed alerts.

## Testing

- Test fixtures in `tests/conftest.py` own per-test store isolation and daemon cleanup.
- If a test leaves a daemon running, that is a test bug and must be fixed.
- The full suite is `python -m pytest tests/ -x -q` (~1700 tests, ~90s).
- If tests don't clean up, or leave daemons running, that is a bug in the tests and must be addressed.
