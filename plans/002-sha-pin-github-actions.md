# Plan 002: Pin all GitHub Actions to commit SHAs

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat e3bb33d..HEAD -- .github/workflows/`
> If any workflow file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: security
- **Planned at**: commit `e3bb33d`, 2026-06-10

## Why this matters

Every workflow references third-party actions by floating tag
(`actions/checkout@v6`, `astral-sh/setup-uv@v7`). Tags are mutable: whoever
controls those repos (or an attacker who compromises them) can re-point `v6` at
malicious code, and the next CI run would execute it. The repo otherwise pins
meticulously — Python deps have security floors, pre-commit hooks are version-
pinned, the gitleaks binary version is pinned — so floating action tags are the
one unpinned supply-chain surface. Pinning to immutable commit SHAs closes it.

Note on scope of risk: there is **no** GitHub Actions release/publish workflow in
this repo (releases run locally via `scripts/release.sh`), so these workflows do
not currently hold PyPI credentials. This is hygiene and reproducibility, not an
active credential-exposure fix — but it is cheap and removes the foot-gun before
a future workflow does carry a secret.

## Current state

Three workflow files, all under `.github/workflows/`:

- `test.yml` — uses `actions/checkout@v6` (one occurrence) and
  `astral-sh/setup-uv@v7` (one occurrence).
- `security.yml` — uses `actions/checkout@v6` (two jobs) and
  `astral-sh/setup-uv@v7` (two jobs).
- `secret-scan.yml` — uses `actions/checkout@v6` (one occurrence, with
  `fetch-depth: 0`).

Exact current lines (verify against live files):

```yaml
# test.yml
      - uses: actions/checkout@v6
      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          version: "latest"
```

```yaml
# security.yml (appears twice — license-check job and pip-audit job)
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v7
        with:
          version: "latest"
```

```yaml
# secret-scan.yml
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
```

There are no other `uses:` lines (confirm with the grep in Step 1). The
`secret-scan.yml` `Install gitleaks` step downloads a pinned binary via `curl`
and is **out of scope** — its version is already pinned (`GITLEAKS_VERSION: 8.30.1`).

## Commands you will need

| Purpose                  | Command                                              | Expected on success            |
|--------------------------|------------------------------------------------------|--------------------------------|
| List all action refs     | `grep -rn "uses:" .github/workflows/`                | the lines listed above         |
| Resolve a tag to a SHA   | `gh api repos/actions/checkout/git/ref/tags/v6 --jq .object.sha` | a 40-char hex SHA   |
| Validate YAML            | `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` | exit 0 |

`gh` (GitHub CLI) must be authenticated. If `gh` is unavailable or unauthenticated,
see the fallback in Step 2.

## Scope

**In scope** (the only files you may modify):
- `.github/workflows/test.yml`
- `.github/workflows/security.yml`
- `.github/workflows/secret-scan.yml`

**Out of scope** (do NOT touch):
- The `Install gitleaks` curl step in `secret-scan.yml` — already version-pinned.
- The `npm ci` / `node build.mjs` steps — those use the pinned npm lockfile, not
  a GitHub Action.
- Workflow logic, job names, triggers, matrix — only the `@tag` refs change.

## Git workflow

- Branch: `advisor/002-pin-actions`
- One commit: `ci: pin GitHub Actions to commit SHAs`.
- Do NOT push or open a PR unless the operator instructs it.

## Steps

### Step 1: Enumerate every action ref

Run `grep -rn "uses:" .github/workflows/` and record each `owner/repo@tag` and
the file/line. Expected: `actions/checkout@v6` (×4 across the three files) and
`astral-sh/setup-uv@v7` (×3 across test.yml + security.yml). If the set differs,
STOP and report (drift).

### Step 2: Resolve each tag to its current commit SHA

For each distinct `owner/repo@tag`, resolve the immutable SHA the tag currently
points at:

```bash
gh api repos/actions/checkout/git/ref/tags/v6 --jq .object.sha
gh api repos/astral-sh/setup-uv/git/ref/tags/v7 --jq .object.sha
```

If a tag is annotated (the first call returns a tag object, not a commit), you
may need to dereference: `gh api repos/<owner>/<repo>/git/tags/<sha> --jq .object.sha`.
The goal is a **commit** SHA (40 hex chars).

**Fallback if `gh` is unavailable**: STOP and report that `gh` is needed. Do NOT
guess SHAs and do NOT pin to a SHA you cannot verify points at the intended tag —
a wrong SHA breaks CI.

Record the mapping, e.g.:
```
actions/checkout@v6   -> <sha-A>
astral-sh/setup-uv@v7 -> <sha-B>
```

### Step 3: Replace each tag ref with `@<sha> # <tag>`

In each workflow file, replace `@v6` / `@v7` with the resolved SHA, keeping the
human-readable tag as a trailing comment so future maintainers know what version
the SHA corresponds to. GitHub's recommended format:

```yaml
      - uses: actions/checkout@<sha-A> # v6
      - uses: astral-sh/setup-uv@<sha-B> # v7
```

Apply to all occurrences found in Step 1. Do not change `with:` blocks,
`version: "latest"` (that is setup-uv's own input, not an action ref), or
anything else.

**Verify**: `grep -rn "uses:" .github/workflows/` → every line now has a
40-char SHA and a `# vN` comment; no bare `@v6`/`@v7` remains.
**Verify**: `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` → exit 0 (YAML still valid).

### Step 4: Sanity-check the SHAs resolve

For each pinned SHA, confirm it is a real commit in the action's repo:

```bash
gh api repos/actions/checkout/commits/<sha-A> --jq .sha
gh api repos/astral-sh/setup-uv/commits/<sha-B> --jq .sha
```

Each must echo back the same SHA (exit 0). If any 404s, the SHA is wrong — STOP.

## Test plan

- No application tests are affected (CI config only). The real verification is
  that the workflows still parse and the SHAs are valid commits (Steps 3–4).
- If the operator later pushes the branch, the proof is a green CI run; that is
  out of scope for local execution.

## Done criteria

ALL must hold:

- [ ] `grep -rn "uses:" .github/workflows/` shows zero bare `@vN` tag refs — every
      action is pinned to a 40-char SHA with a `# vN` comment.
- [ ] All three workflow files still parse as valid YAML (Step 3 verify).
- [ ] Every pinned SHA resolves to a real commit (Step 4).
- [ ] `git status` shows only the three workflow files modified.
- [ ] `plans/README.md` status row for 002 updated.

## STOP conditions

Stop and report back (do not improvise) if:

- The set of `uses:` refs differs from "Current state" (drift).
- `gh` is unavailable/unauthenticated and you cannot verifiably resolve a tag to
  a SHA. Do not guess.
- A resolved SHA fails the Step 4 commit-exists check.

## Maintenance notes

- Pinned SHAs do not auto-update. Add `astral-sh/setup-uv` and `actions/checkout`
  to whatever dependency-update mechanism the repo adopts (e.g. Dependabot for
  GitHub Actions), or plan to bump them manually when the `# vN` comment falls
  behind. Mention this in the PR description.
- Reviewer should confirm the `# vN` comments match the SHAs (the comment is the
  only human-readable signal of which version is pinned).
