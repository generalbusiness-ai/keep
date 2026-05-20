"""Derive a workstream slug from the working directory.

The plugin hooks (``SessionStart``/``UserPromptSubmit``/``Stop``/``SubagentStop``)
update a ``now`` note to track in-flight work. When several Claude Code
sessions run in parallel — different worktrees, different branches, different
projects — a single shared ``now`` interleaves their updates.

Scoping the note as ``now:{workstream-slug}`` puts each line of work onto its
own version-chain. The slug is derived from the current working directory:

  - In a git repo: ``{project}/{branch}``. ``project`` is the basename of the
    common git dir's parent, so worktrees inherit the project of the main
    checkout. ``branch`` comes from ``git branch --show-current``.
  - Detached HEAD: ``{project}/{worktree-basename}`` — gives stable identity
    during rebase/bisect without polluting the bare ``now``.
  - Outside any git repo: cwd basename — sibling directories still get
    separate chains rather than dogpiling on ``now``.

The slug must be safe to embed in a doc-id (``now:{slug}``) and a tag value,
so each component is sanitized to ``[A-Za-z0-9._-]`` (slashes are preserved
between components only).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _run_git(cwd: Path, args: list[str]) -> str | None:
    """Run a git subcommand in *cwd*; return stripped stdout, or None on error."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _sanitize(part: str) -> str:
    """Collapse runs of unsafe characters in a single slug component to ``-``."""
    cleaned = _SAFE_RE.sub("-", part).strip("-")
    return cleaned or "unknown"


def derive_workstream_slug(cwd: Path | None = None) -> str:
    """Compute the workstream slug for *cwd* (or :func:`Path.cwd`)."""
    cwd = (cwd or Path.cwd()).resolve()

    common = _run_git(cwd, ["rev-parse", "--git-common-dir"])
    if common:
        # --git-common-dir is the *main* repo's .git, even from a worktree.
        # Its parent is the main project root; the basename gives us a
        # project identifier that's shared across all worktrees of the repo.
        common_path = Path(common)
        if not common_path.is_absolute():
            common_path = cwd / common_path
        project = _sanitize(common_path.resolve().parent.name)

        branch = _run_git(cwd, ["branch", "--show-current"])
        if branch:
            return f"{project}/{_sanitize(branch)}"

        # Detached HEAD: branch is empty. Use the worktree directory name —
        # for a worktree at .../worktrees/foo this still disambiguates from
        # the main checkout.
        toplevel = _run_git(cwd, ["rev-parse", "--show-toplevel"])
        ident = Path(toplevel).name if toplevel else cwd.name
        return f"{project}/{_sanitize(ident)}"

    # Not a git repo: cwd basename. Two sibling scratch dirs still get
    # separate chains; nothing falls through to the bare ``now``.
    return _sanitize(cwd.name)
