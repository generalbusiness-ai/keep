# Plan 006: Expose the deep-follow depth as a configurable parameter

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan in
> `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat e3bb33d..HEAD -- keep/_search_augmentation.py keep/flow_env.py`
> If either changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it as a
> STOP condition.

## Status

- **Priority**: P3
- **Effort**: S (spike/feature)
- **Risk**: LOW
- **Depends on**: none
- **Category**: direction
- **Planned at**: commit `e3bb33d`, 2026-06-10

## Why this matters

"Deep" search follows tags and edges from the primary results to surface bridge
documents — a headline capability of the tool. But the breadth of that follow is
**hard-coded**: `_deep_tag_follow(... top_k=10 ...)` always collects tags from the
first 10 primary items, with no way for a caller or user to widen or narrow it.
Power users running broad investigative searches can't ask for a deeper follow,
and there's no way to trade latency for recall. The architecture already threads a
`limit` through the call site, so exposing the depth is a small, additive change
with a safe default — the kind of capability the existing design makes cheap.

**This is a direction/spike plan**: the first deliverable is to *confirm the design*
(where the parameter should enter, what it should be named, and whether it belongs
on the `find`/`traverse` flow or only the internal API) and then implement the
smallest version that adds real value without changing default behavior.

## Current state

- `keep/_search_augmentation.py` — defines `_deep_tag_follow`:

```python
    def _deep_tag_follow(self, primary_items, chroma_coll, doc_coll, *,
                         embedding=None, top_k=10,
                         per_tag_fetch=1000, max_per_group=5):
        ...
        for item in primary_items[:top_k]:
            for k, v in iter_tag_pairs(item.tags, include_system=False):
                ...
```

  `top_k` controls how many primary items contribute their tags to the follow.
  `max_per_group` caps how many deep items are returned per primary.

- `keep/flow_env.py` (~line 478) — the only caller. It passes `max_per_group=limit`
  but **does not** pass `top_k`, so the follow depth is always the default 10:

```python
        if tagfollow_items:
            chroma_coll = self._keeper._resolve_chroma_collection()
            try:
                tag_groups = self._keeper._deep_tag_follow(
                    tagfollow_items,
                    chroma_coll,
                    doc_coll,
                    embedding=self._query_embedding,
                    max_per_group=limit,
                )
```

- The user-facing entry to deep search is the `find`/`traverse` flow with
  `deep=True`. **You must trace** how `deep` and `limit` reach this call site
  (start from `RemoteKeeper.find`/the daemon `find` flow and the `find` action in
  `keep/actions/find.py`), to decide where a new depth parameter should be
  introduced and how it would be surfaced (flow param → CLI flag → MCP arg). Do
  not guess the plumbing — read it.

## Commands you will need

| Purpose               | Command                                                  | Expected on success      |
|-----------------------|----------------------------------------------------------|--------------------------|
| Find the call site    | `grep -rn "_deep_tag_follow" keep/`                      | definition + 1 caller    |
| Trace deep/limit flow | `grep -rn "deep\b" keep/actions/find.py keep/flow_env.py keep/_search_augmentation.py` | the wiring |
| Targeted tests        | `python -m pytest tests/ -k "deep or tag_follow or traverse" -q` | all pass        |
| Full suite            | `python -m pytest tests/ -x -q`                         | all pass, no daemon left |
| Lint                  | `ruff check keep/_search_augmentation.py keep/flow_env.py` | exit 0                |

## Scope

**In scope** (expected; confirm during the spike):
- `keep/flow_env.py` — pass a depth value through to `_deep_tag_follow`.
- `keep/_search_augmentation.py` — only if the signature needs a clearer name or a
  validated bound (it already accepts `top_k`).
- The `find`/`traverse` flow param definition and the CLI flag — wherever the
  spike (Step 1) determines `deep`/`limit` are declared (likely `keep/actions/find.py`
  and/or a state-doc; identify it before editing).
- A test file under `tests/` for the new behavior.

**Out of scope** (do NOT touch):
- The deep-follow *scoring* (IDF weighting, RRF, `max_per_group` collapse logic) —
  this plan changes how many primaries are followed, not how results are ranked.
- The hosted backend's server-side flow implementation (sibling repo) — only the
  client-visible parameter and the local flow.
- Changing the **default** (must remain 10) — default behavior is unchanged.

## Git workflow

- Branch: `advisor/006-deep-follow-depth`
- Commit style: short imperative subject, e.g.
  `search: expose deep-follow depth (default unchanged)`.
- Do NOT push or open a PR unless the operator instructs it.

## Steps

### Step 1: Spike — decide the parameter's name, location, and surface

Trace the path from a user `find --deep --limit N` (and the MCP/remote equivalents)
down to `_deep_tag_follow`. Produce a short written design note (put it in the PR
description, not a new doc) answering:

- What should the parameter be called at each layer? (Recommend `deep_follow` or
  `deep_depth`; the internal arg is already `top_k` — keep internal/external names
  consistent with nearby code.)
- Where does it enter — is it a new flow param on `find`, a CLI option, an MCP arg,
  or just an internal API knob for now? Choose the **smallest** surface that
  delivers value; it is acceptable for the first cut to thread it from the `find`
  action down to `_deep_tag_follow` with a sensible default, without a new CLI flag,
  if adding the flag is disproportionate. State the choice and why.
- Confirm the default stays 10 and out-of-range values are clamped/validated.

**Verify**: the design note exists and names exact files/symbols for Step 2.

### Step 2: Thread the parameter through (default-preserving)

Implement the design from Step 1. At minimum, `keep/flow_env.py` should pass an
explicit depth to `_deep_tag_follow(... top_k=<resolved depth> ...)`, resolved from
the new parameter with a default of 10. Validate the bound (e.g. clamp to a
reasonable max so a user can't request following 10⁶ primaries).

**Verify**: `python -c "import keep.flow_env, keep._search_augmentation"` → exit 0.

### Step 3: Wire the user-facing surface (if Step 1 chose to)

If the spike decided to expose a CLI flag / flow param / MCP arg, add it, defaulting
to the same value. Keep naming and help-text style consistent with adjacent options
(read neighboring options in `keep/actions/find.py` / the CLI for tone).

**Verify**: if a CLI flag was added, `python -m keep find --help` shows it (run via
the e2e-style subprocess or `keep find --help`); if a flow param, a flow call with
the param set reaches `_deep_tag_follow` with the right `top_k` (assert in the test).

### Step 4: Tests

Add a test that:
- with the default, `_deep_tag_follow` is called with `top_k=10` (regression guard
  for unchanged default), and
- with the new parameter set to a different value, that value reaches
  `_deep_tag_follow` as `top_k` (the feature works), and
- an out-of-range value is clamped/rejected per Step 2.

Use the existing deep-search tests as the structural pattern (find them with
`grep -rln "deep\|_deep_tag_follow\|traverse" tests/`).

**Verify**: `python -m pytest tests/ -k "deep or tag_follow or traverse" -q` → all
pass, including the new cases.

## Test plan

- New tests covering: default depth unchanged (=10), custom depth honored, and
  bound validation. In the test file matching the surface chosen (e.g.
  `tests/test_traverse*.py` or the find-flow tests).
- Structural pattern: existing deep-follow/traverse tests.
- Verification: targeted `-k` run passes; full `python -m pytest tests/ -x -q`
  green.

## Done criteria

ALL must hold:

- [ ] A design note (in the PR description) records the chosen name/surface/default.
- [ ] `keep/flow_env.py` passes an explicit, parameterized depth to
      `_deep_tag_follow` (no longer relying on the implicit default).
- [ ] Default behavior is unchanged: with no parameter, depth is 10 (test proves it).
- [ ] A custom depth value reaches `_deep_tag_follow` as `top_k` (test proves it).
- [ ] Out-of-range values are clamped or rejected (test proves it).
- [ ] `python -m pytest tests/ -x -q` passes, no daemon left running.
- [ ] `ruff check` clean on touched files.
- [ ] `plans/README.md` status row for 006 updated.

## STOP conditions

Stop and report back (do not improvise) if:

- The spike (Step 1) reveals that `deep`/`limit` plumbing is more entangled than the
  call site suggests (e.g. depth is meaningful to the hosted server-side flow too)
  — report the design question before implementing a half-surface.
- Changing where the parameter enters would require touching the hosted-backend
  contract or a state-doc schema in a way that affects existing users — STOP; that
  is a maintainer decision.
- "Current state" excerpts don't match the live code (drift).

## Maintenance notes

- For the reviewer: the critical property is that the **default is unchanged** —
  this must be a purely additive capability. Scrutinize the default path.
- If/when this is exposed to the hosted backend, the server-side flow must accept
  and honor the same parameter; note that as a follow-up so the two sides don't
  drift.
- A natural extension (not in this plan): expose `max_per_group` and
  `per_tag_fetch` similarly, and document the deep-search tuning knobs in
  `docs/KEEP-FIND.md`.
