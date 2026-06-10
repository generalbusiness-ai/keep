# Plan 004: Run the e2e-marked tests in CI

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat e3bb33d..HEAD -- .github/workflows/test.yml pyproject.toml tests/test_cli.py`
> If any changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it as a
> STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `e3bb33d`, 2026-06-10

## Why this matters

The repo has `@pytest.mark.e2e` tests that spin up the real daemon and drive the
`keep` CLI as a subprocess (`tests/test_cli.py` has several). These exercise the
most fragile integration points — daemon auto-start, port/token discovery,
subprocess wiring — exactly the seam that the recent remote-backend work churned.
But **no CI job runs them**: the only test workflow runs
`pytest ... -m "not slow and not e2e"`, explicitly excluding e2e, and there is no
other workflow that runs them. So a regression in daemon startup or CLI wiring
ships green and is only caught by a human running e2e locally. This plan adds a CI
job that runs the e2e suite, so that class of break is caught automatically.

## Current state

- `.github/workflows/test.yml` — the only test workflow. Its `test` job runs a
  matrix over Python 3.11/3.12/3.13 and ends with:

```yaml
      - name: Run tests
        run: uv run --python ${{ matrix.python-version }} pytest --tb=short -q -m "not slow and not e2e"
        env:
          OPENAI_API_KEY: sk-test-ci-dummy-key
```

  Steps before it (the same setup the e2e job will reuse): `actions/checkout`,
  `astral-sh/setup-uv`, `uv python install`, `uv lock --check`, build the OpenClaw
  plugin bundle (`npm ci && node build.mjs` in `keep/data/openclaw-plugin`), then
  `uv sync --python ... --extra dev --extra langchain`.

- `pyproject.toml` — marker definitions and default deselect:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not slow and not smoke'"
markers = [
    "slow: marks tests as slow (require ML models)",
    "e2e: marks tests as end-to-end (require real providers)",
    "smoke: end-to-end smoke tests against the live keep.generalbusiness.ai service (require KEEPNOTES_API_KEY; opt-in via -m smoke)",
]
```

  Note: the default `addopts` deselects `slow` and `smoke` but **not** `e2e`; the
  CI command adds `and not e2e` on top. So `pytest -m e2e` locally will run e2e
  tests (they are not globally disabled), which is what the new job relies on.

- `tests/test_cli.py` — contains the e2e tests. The `cli` fixture
  (`tests/test_cli.py:69`) detects the `e2e` marker and, when present, runs the
  real `keep` binary as a subprocess against a shared module-scoped store
  (`_shared_e2e_cli_env`, `tests/test_cli.py:31`) with env
  `KEEP_STORE_PATH` / `KEEP_CONFIG` set and `KEEP_LOCAL_ONLY=1`. It warms the
  daemon with retries and **cleans up daemons in teardown**
  (`_cleanup_daemons_under(root)`). So the e2e tests are self-contained: no live
  provider network calls (they use `KEEP_LOCAL_ONLY=1` and read-only `get`/`list`
  commands), but they do require a working daemon launch.

- These e2e tests use only local, read-only operations (e.g.
  `cli("get", ".meta/todo", ...)`), so they do **not** need real provider
  credentials — the dummy `OPENAI_API_KEY` already used by the unit job is
  sufficient. Confirm this by reading the e2e test bodies before relying on it.

## Commands you will need

| Purpose                | Command                                                     | Expected on success      |
|------------------------|-------------------------------------------------------------|--------------------------|
| Count e2e tests        | `python -m pytest tests/ -m e2e --collect-only -q \| tail -3` | a non-zero count       |
| Run e2e locally        | `python -m pytest tests/ -m e2e -q`                         | all pass, no daemon left |
| Validate workflow YAML | `python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"` | exit 0 |
| Full default suite     | `python -m pytest tests/ -x -q`                             | all pass (unchanged)     |

## Scope

**In scope** (the only file you may modify):
- `.github/workflows/test.yml` — add a new job (or a new step) that runs the e2e
  tests.

**Out of scope** (do NOT touch):
- `pyproject.toml` markers/addopts — the marker config is already correct; do not
  change the default deselect.
- The e2e tests themselves and their fixtures — if an e2e test is flaky or broken,
  STOP and report; do not "fix" it under this plan.
- The existing unit `test` job's command — it should keep excluding e2e so the
  fast matrix stays fast; e2e runs in its own job.

## Git workflow

- Branch: `advisor/004-e2e-ci`
- One commit: `ci: run e2e tests in a dedicated job`.
- Do NOT push or open a PR unless the operator instructs it.

## Steps

### Step 1: Confirm the e2e tests pass locally and count them

Before touching CI, establish the baseline:

```bash
python -m pytest tests/ -m e2e --collect-only -q | tail -3   # record the count
python -m pytest tests/ -m e2e -q                            # must pass locally
```

If the e2e tests do not pass locally (or leave a daemon running), STOP — CI should
not be wired to a red or leaky suite. Report which tests fail.

**Verify**: e2e suite passes locally; `~/.keep` / the temp store has no leftover
daemon (the fixtures clean up — confirm the run exits promptly).

### Step 2: Add a dedicated `e2e` job to test.yml

Add a second job alongside `test`. It must replicate the setup steps (checkout,
setup-uv, python install, build the OpenClaw bundle, `uv sync`) and then run only
the e2e tests. Use a single Python version (3.12 is the project's primary — the
`security.yml` jobs use 3.12) to keep it cheap; the unit matrix already covers
3.11/3.13. Pass the same dummy `OPENAI_API_KEY`. Sketch:

```yaml
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          version: "latest"
      - name: Set up Python
        run: uv python install 3.12
      - name: Build OpenClaw plugin bundle
        working-directory: keep/data/openclaw-plugin
        run: |
          npm ci
          node build.mjs
      - name: Install dependencies
        run: uv sync --python 3.12 --extra dev --extra langchain
      - name: Run e2e tests
        run: uv run --python 3.12 pytest --tb=short -q -m e2e
        env:
          OPENAI_API_KEY: sk-test-ci-dummy-key
```

If plan 002 (SHA-pin actions) has already landed, match its pinned-SHA format for
the `uses:` lines instead of the bare `@v6`/`@v7` shown here. (Check with
`grep -n "uses:" .github/workflows/test.yml` first.)

**Verify**: `python -c "import yaml; print(list(yaml.safe_load(open('.github/workflows/test.yml'))['jobs'].keys()))"`
→ lists both `test` and `e2e`.

### Step 3: Confirm the unit job still excludes e2e

The fast matrix job should be unchanged and still run `-m "not slow and not e2e"`,
so e2e work isn't duplicated across the 3-way matrix.

**Verify**: `grep -n "not e2e" .github/workflows/test.yml` → still present in the
`test` job's run command.

## Test plan

- No new application tests; this wires existing e2e tests into CI.
- The verification is: (a) e2e passes locally (Step 1), (b) the workflow YAML is
  valid and has the new job (Step 2), (c) the unit job is unchanged (Step 3).
- The ultimate proof is a green `e2e` job on a pushed branch — out of scope for
  local execution, but note it in the PR description so the operator watches the
  first run.

## Done criteria

ALL must hold:

- [ ] `python -m pytest tests/ -m e2e -q` passes locally with no daemon left running.
- [ ] `.github/workflows/test.yml` has a new `e2e` job that runs `pytest -m e2e`.
- [ ] The original `test` job still runs `-m "not slow and not e2e"`.
- [ ] The workflow file is valid YAML (Step 2 verify).
- [ ] `git status` shows only `.github/workflows/test.yml` modified.
- [ ] `plans/README.md` status row for 004 updated.

## STOP conditions

Stop and report back (do not improvise) if:

- The e2e suite fails or leaves a daemon running locally (Step 1) — that is a real
  bug to surface, not something to paper over by skipping tests.
- The e2e tests turn out to require real provider credentials or network access
  you can't satisfy in CI (read the test bodies — the current ones use
  `KEEP_LOCAL_ONLY=1` and read-only commands, so they should not). If they do,
  report it; the job may need a different approach (e.g. gated on a secret).
- "Current state" excerpts don't match the live files (drift).

## Maintenance notes

- For the reviewer: confirm the e2e job uses a single Python version (not the full
  matrix) so CI cost stays bounded, and that it isn't marked `continue-on-error`
  (a non-blocking job that's allowed to fail provides no protection).
- If e2e tests prove slow or occasionally flaky in CI, the next decision is
  whether to run them only on `push` to `main` (not every PR) or add a timeout —
  flag that trade-off in the PR rather than silently disabling the job.
- When new e2e tests are added, they are automatically picked up by `-m e2e`; no
  workflow change needed.
