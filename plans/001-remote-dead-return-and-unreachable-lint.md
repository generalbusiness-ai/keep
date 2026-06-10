# Plan 001: Remove the unreachable `return url` in remote.py and add a lint that catches unreachable code

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat e3bb33d..HEAD -- keep/remote.py pyproject.toml`
> If either file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `e3bb33d`, 2026-06-10

## Why this matters

`keep/remote.py` has a stray `return url` statement immediately after the
function's real `return`. It is unreachable, and `url` is not a name bound
anywhere in that function — so if a future refactor ever made it reachable, it
would raise `NameError` at runtime instead of failing loudly now. More
importantly, the fact that this slipped in undetected shows nothing in the lint
gate catches unreachable code. This plan deletes the dead line and turns on the
`ruff` rules that would have caught it, so the class of bug can't recur silently.

## Current state

- `keep/remote.py` — the hosted-API client module. The function
  `resolve_remote_config()` ends like this (around lines 119–123):

```python
    if not api_key:
        return None
    return RemoteConfig(api_url=api_url, api_key=api_key, project=project or None)
    return url
```

  The last line (`return url`) is unreachable dead code. `url` is undefined in
  this scope (the variable built earlier is named `api_url`). The real return is
  the `RemoteConfig(...)` line above it.

- `pyproject.toml` — the ruff config currently selects **only** docstring rules:

```toml
[tool.ruff.lint]
select = ["D"]
ignore = [
    "D100",  # Missing docstring in public module
    ...
]
```

  Ruff's `F` (pyflakes) rule group includes `F811`/unreachable-style checks, and
  the `PLW0101` (`useless-return`/unreachable) lives in the Pylint group. The
  rule that specifically flags code after a `return` is **`unreachable code`**,
  surfaced by ruff's pyflakes-derived `F` group is *not* it — the precise rule is
  ruff's `PLW0101`? No: the reliable, stable rule for this is the pyflakes
  **`F` group does not flag unreachable returns**. The dedicated check is ruff
  rule **`RET503`/`RET505`** (flake8-return) and, for plain dead code after a
  terminal statement, **`PLW0101` is not stable**. Use the approach in Step 2,
  which selects the **`F` group plus flake8-return (`RET`)** and verifies
  empirically that the offending line is caught *before* you delete it — do not
  assume a specific rule code; let the tool tell you.

## Commands you will need

| Purpose      | Command                                  | Expected on success        |
|--------------|------------------------------------------|----------------------------|
| Lint (all)   | `ruff check .`                           | exit 0, "All checks passed"|
| Lint a file  | `ruff check keep/remote.py`              | exit 0 or specific findings|
| Targeted test| `python -m pytest tests/test_remote.py -q`| all pass                  |
| Full suite   | `python -m pytest tests/ -x -q`          | all pass (~1700 tests)     |

(Ruff is installed via the `dev` extra; if `ruff` is not on PATH, use
`python -m ruff` or `uv run ruff`.)

## Scope

**In scope** (the only files you may modify):
- `keep/remote.py` (delete one line)
- `pyproject.toml` (extend the ruff `select` list)

**Out of scope** (do NOT touch):
- Any other `return`/control-flow in `remote.py` — only the single unreachable
  `return url` line is in scope.
- The existing `ignore = [...]` docstring entries — leave them exactly as they
  are. You are *adding* a rule group, not changing the docstring policy.
- Any code that a newly-enabled rule flags **elsewhere** in the repo. If turning
  on the rule group surfaces other violations, that is a STOP condition (see
  below) — do not fix unrelated files in this plan.

## Git workflow

- Branch: `advisor/001-remote-unreachable`
- Commit style matches the repo (`git log --oneline -5` shows short imperative
  subjects, e.g. `security: floor aiohttp>=3.14.0`). Use one commit, e.g.
  `remote: delete unreachable return; lint for dead code`.
- Do NOT push or open a PR unless the operator instructs it.

## Steps

### Step 1: Confirm the dead line and delete it

Open `keep/remote.py`, find `def resolve_remote_config`, and confirm the last
two lines of the function body are exactly:

```python
    return RemoteConfig(api_url=api_url, api_key=api_key, project=project or None)
    return url
```

Delete only the final `    return url` line. The function must still end with the
`return RemoteConfig(...)` line.

**Verify**: `grep -n "return url" keep/remote.py` → no output (exit 1).
**Verify**: `python -c "import keep.remote"` → exit 0, no error.

### Step 2: Turn on a lint rule group that catches unreachable code, empirically

Before editing `pyproject.toml`, prove the rule you pick actually flags the
pattern. Temporarily re-add the dead line you deleted in Step 1 to a scratch
file and test rule groups against it:

```bash
printf 'def f():\n    return 1\n    return x\n' > /tmp/dead.py
ruff check --select F /tmp/dead.py            # note whether "unreachable"/F-code appears
ruff check --select RET /tmp/dead.py          # flake8-return group
ruff check --select PLW0101 /tmp/dead.py      # pylint unreachable, if available
```

Pick the **smallest** rule (or rule group) whose output names the
`return x` line as unreachable. Record which one worked. (In current ruff this is
typically reported under the pyflakes/`F`-family as an unreachable-code warning;
if none of the above flags it, STOP — see STOP conditions.)

Then add that rule to `pyproject.toml`'s `[tool.ruff.lint]` `select` list,
preserving the existing `"D"`:

```toml
[tool.ruff.lint]
select = ["D", "<RULE-OR-GROUP-YOU-VERIFIED>"]
```

**Verify**: `rm /tmp/dead.py`. Then `ruff check .` → exit 0,
"All checks passed!" (the repo itself must be clean under the new rule, because
you already removed the only offender in Step 1).

### Step 3: Run the targeted and full test suites

The change is non-behavioral, but confirm nothing imports the deleted line's
surrounding code in a way that breaks.

**Verify**: `python -m pytest tests/test_remote.py -q` → all pass.
**Verify**: `python -m pytest tests/ -x -q` → all pass, no daemon left running.

## Test plan

- No new test is strictly required (this is dead-code removal + lint config), but
  add one cheap guard to `tests/test_remote.py` that pins the fixed behavior:
  a test asserting `resolve_remote_config` returns a `RemoteConfig` with the
  expected `api_url` when only `KEEPNOTES_API_KEY` is set (model it after the
  existing `TestResolveRemoteConfig` cases already in that file — see
  `tests/test_remote.py`, the class beginning near the `test_env_only_returns_remote_with_default_api_url`
  test). This locks in that the function's real return path is exercised.
- Verification: `python -m pytest tests/test_remote.py -q` → all pass, including
  the new case.

## Done criteria

ALL must hold:

- [ ] `grep -n "return url" keep/remote.py` returns nothing.
- [ ] `ruff check .` exits 0 with "All checks passed!".
- [ ] The new ruff rule, run against a scratch file containing code after a
      `return`, reports it (you verified this in Step 2).
- [ ] `python -m pytest tests/test_remote.py -q` passes.
- [ ] `python -m pytest tests/ -x -q` passes with no daemon left running.
- [ ] `git status` shows only `keep/remote.py`, `pyproject.toml`, and
      (optionally) `tests/test_remote.py` modified.
- [ ] `plans/README.md` status row for 001 updated.

## STOP conditions

Stop and report back (do not improvise) if:

- The lines in "Current state" don't match the live code (drift).
- Turning on the chosen rule group flags violations in files **other than**
  the one dead line — report the list; do NOT mass-fix them here (that is a
  separate cleanup decision for the maintainer).
- None of the candidate rules in Step 2 flag code-after-`return`. In that case,
  still complete Step 1 (delete the dead line) and report that no stable ruff
  rule covers this pattern, so the lint half of the plan is deferred.

## Maintenance notes

- For the reviewer: confirm the ruff `select` change did not silently start
  ignoring docstring rules (the `"D"` must remain first in the list).
- If a future change adds a new rule group to `select`, expect a one-time sweep
  of newly-surfaced findings — that is normal and unrelated to this plan.
