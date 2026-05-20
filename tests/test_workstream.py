"""Tests for keep.workstream.derive_workstream_slug.

The function is called from plugin hooks (UserPromptSubmit/Stop/etc.) to pick
a per-workstream ``now:{slug}`` doc id so parallel sessions don't interleave
on a single ``now``. The cases that matter:

  - in a checked-out git branch  → ``{project}/{branch}``
  - in a worktree of that repo   → still ``{project}/{branch}``, where
    ``project`` is the *main* repo's basename (worktrees share project)
  - detached HEAD                → ``{project}/{worktree-basename}``
  - outside any git repo         → cwd basename (no project prefix)
  - unsafe characters in branch  → sanitized in place

Each test builds a real ephemeral git repo in ``tmp_path`` rather than
mocking subprocess — the helper itself shells out to ``git``, so a real
repo gives us actual coverage.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from keep.workstream import derive_workstream_slug


def _git(cwd: Path, *args: str) -> None:
    """Run a git command in *cwd*. Raises on failure (test setup must succeed)."""
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        env={
            **os.environ,
            # Deterministic identity — required for commits in CI.
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


def _init_repo(path: Path, *, initial_branch: str = "main") -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", f"--initial-branch={initial_branch}")
    (path / "README.md").write_text("hi\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")


def test_non_git_uses_cwd_basename(tmp_path):
    """Outside any repo, slug is just the cwd basename."""
    work = tmp_path / "scratch-dir"
    work.mkdir()
    assert derive_workstream_slug(work) == "scratch-dir"


def test_git_branch_combines_project_and_branch(tmp_path):
    """Inside a checkout, slug is ``{project}/{branch}``."""
    repo = tmp_path / "myproj"
    _init_repo(repo, initial_branch="main")
    assert derive_workstream_slug(repo) == "myproj/main"


def test_git_branch_with_feature_branch(tmp_path):
    """Feature branch names flow through unchanged when already slug-safe."""
    repo = tmp_path / "myproj"
    _init_repo(repo)
    _git(repo, "checkout", "-b", "feature-x")
    assert derive_workstream_slug(repo) == "myproj/feature-x"


def test_worktree_inherits_main_project_name(tmp_path):
    """A worktree of ``myproj`` still gets project=myproj, not the worktree dir name.

    This is the load-bearing case for parallel-work isolation: two worktrees
    of the same repo on different branches must get different slugs but the
    same project prefix.
    """
    main = tmp_path / "myproj"
    _init_repo(main)
    wt = tmp_path / "myproj-wt-other"
    _git(main, "worktree", "add", "-b", "other-branch", str(wt))
    slug = derive_workstream_slug(wt)
    assert slug == "myproj/other-branch"


def test_detached_head_falls_back_to_worktree_basename(tmp_path):
    """Detached HEAD has no branch, so use the worktree directory basename."""
    repo = tmp_path / "myproj"
    _init_repo(repo)
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo), capture_output=True, text=True, check=True,
    ).stdout.strip()
    _git(repo, "checkout", head_sha)
    slug = derive_workstream_slug(repo)
    assert slug == "myproj/myproj"


def test_branch_slashes_are_collapsed(tmp_path):
    """Git allows ``feature/foo`` branches; the slug must stay flat.

    The doc-id is ``now:{slug}`` and the slug uses a single ``/`` as the
    project/branch separator. Embedded slashes inside the branch name would
    break that contract, so the sanitizer collapses them to ``-``.
    """
    repo = tmp_path / "myproj"
    _init_repo(repo)
    _git(repo, "checkout", "-b", "feature/sub-area")
    slug = derive_workstream_slug(repo)
    assert slug == "myproj/feature-sub-area"
    assert slug.count("/") == 1


def test_cwd_default_uses_process_cwd(tmp_path, monkeypatch):
    """When called with no argument, derive from os.getcwd."""
    work = tmp_path / "implicit-cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    assert derive_workstream_slug() == "implicit-cwd"
