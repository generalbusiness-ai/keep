# Release Process

Releases are produced by `scripts/release.sh`. **Do not run it without explicit
user instruction.** Releases bump the version, create a tag, push to GitHub,
upload to PyPI, and publish a GitHub release — all in one shot.

## Pre-release Checklist

Before asking for a release, walk through these checks. Fix anything that's
wrong rather than deferring it.

1. **Working tree is clean**
   ```bash
   git status
   ```
   `release.sh` refuses to run with uncommitted changes.

2. **Design docs reflect implemented work**
   If recent work implemented a written design in `later/design/`, update the
   doc to mark it implemented (or to capture how the implementation differs
   from the plan). Stale design docs erode trust in the design directory.

3. **User-facing docs are accurate**
   Skim `docs/` for anything affected by the release. New CLI flags, new
   actions, new state docs, new tags — all should be reflected in the
   relevant guides before shipping. The pre-release expectation is that
   `docs/` is correct.

4. **Lint is clean**
   ```bash
   ruff check
   ```
   Fix every reported issue. The release script doesn't gate on this, but
   pre-commit will block subsequent commits.

5. **Tests pass**
   ```bash
   python -m pytest tests/ -x -q
   ```
   ~2600 tests, ~3 minutes. The full suite must pass — no skipped failures,
   no leaked daemons. If a test leaves a daemon running, that is a test bug
   and must be fixed before release.

## Release Numbering

- **Major** is `0` and stays `0`.
- **Minor** bumps mean feature work — new commands, new actions, new tags,
  new state docs, behavior changes that users will notice.
- **Patch** bumps mean maintenance — bug fixes, doc fixes, perf, refactors
  with no user-visible behavior change.

## Running the Release

```bash
scripts/release.sh patch          # 0.130.1 → 0.130.2
scripts/release.sh minor          # 0.130.1 → 0.131.0
scripts/release.sh 0.131.0        # explicit version
```

`release.sh` runs six steps in order, exiting on the first failure:

1. **Bump version** via `scripts/bump_version.py`. Updates `pyproject.toml`,
   `uv.lock`, `SKILL.md`, the OpenClaw plugin manifests and source files
   (`openclaw.plugin.json`, `package.json`, `package-lock.json`,
   `src/index.ts`, `src/mcp-transport.ts`), and the Claude Code plugin
   manifest.
2. **Commit and tag.** Stages an explicit list of files (not `git add -A`)
   so unrelated working-tree changes can't sneak into the release commit.
   Builds the commit subject from the previous tag's commit message.
3. **Build** sdist and wheel via `uv build` into `dist/`. Verifies both
   files exist before continuing.
4. **Push** `main` and the new tag to `origin`.
5. **Upload to PyPI** via `uvx twine upload dist/keep_skill-VERSION*`.
   Credentials come from `~/.pypirc` or `TWINE_USERNAME`/`TWINE_PASSWORD`.
6. **Create the GitHub release** via `gh release create`. Minor releases
   get a "What's new" body assembled from `git log` since the last minor
   tag; patch releases use the release commit's body.

## Requirements

- `python3`, `uv`, `gh`, `git`
- PyPI credentials configured (`~/.pypirc` or env vars)
- `gh auth status` shows authenticated
- A clean working tree on `main`

## Hooks and CI Backstop

Pre-commit hooks (`.pre-commit-config.yaml`) run on every commit, including
the release commit:

- `gitleaks` — secret scan
- `ruff-check` — lint
- `check-merge-conflict` — conflict-marker check
- `check-added-large-files` — 1 MB cap

CI also runs gitleaks against the full git history on every push to `main`
and every PR (`.github/workflows/secret-scan.yml`).

**Never bypass hooks with `--no-verify`** unless the user explicitly asks.
If gitleaks flags something, investigate it as a real finding first.

## After the Release

- Verify the release page on GitHub shows the correct notes
- Verify `pip install keep-skill==NEW_VERSION` resolves the new version
- If it's a minor release, post a release note wherever you usually do

## Hotfix Path

A hotfix is just a patch release on `main`:

1. Land the fix on `main` (review, tests, lint)
2. `scripts/release.sh patch`

There is no separate hotfix branch in the workflow. The patch increment is
the hotfix.
