"""Git changelog ingest — index commits as keep items.

Each commit becomes an item with edges to the files it touched.
Files get a ``git_commit`` tag pointing back to their last commit.
Incremental: tracks a watermark SHA so re-scans only process new commits.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

# Valid git SHA: 40-char (SHA-1) or 64-char (SHA-256) hex string.
# Used to prevent argument injection via stored watermark values.
_VALID_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")

logger = logging.getLogger(__name__)

# git log format: sha\x00author_name\x00author_email\x00date\x00subject
# Note: %b (body) is intentionally excluded — it contains newlines that
# corrupt --name-only file list parsing.
_LOG_FORMAT = "%H%x00%an%x00%ae%x00%aI%x00%s"


def _repo_name(repo_dir: Path) -> str:
    """Derive a stable repo identifier from git remote or absolute path.

    Uses the origin remote URL (stripped of .git suffix and protocol prefix)
    when available, otherwise falls back to the absolute directory path.

    Examples:
        github.com/generalbusiness-ai/keep   (from https://github.com/generalbusiness-ai/keep.git)
        /Users/hugh/play/local-project  (no remote)
    """
    # Try origin remote URL first
    url = _run_git(repo_dir, ["remote", "get-url", "origin"])
    if url:
        url = url.strip()
        # Strip protocol prefix and .git suffix
        for prefix in ("https://", "http://", "ssh://", "git@", "git://"):
            if url.startswith(prefix):
                url = url[len(prefix):]
                break
        # Strip credentials/userinfo (e.g. x-access-token:PAT@github.com/...)
        if "@" in url:
            url = url.split("@", 1)[1]
        # git@github.com:user/repo.git → github.com/user/repo
        url = url.replace(":", "/", 1) if ":" in url and "/" not in url.split(":")[0][1:] else url
        url = url.removesuffix(".git").removesuffix("/")
        return url

    # Fallback to absolute path
    return str(repo_dir.resolve())


def _run_git(repo_dir: Path, args: list[str], timeout: int = 60) -> str | None:
    """Run a git command in a repo directory. Returns stdout or None on error."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def is_git_repo(directory: Path) -> bool:
    """Check if a directory is inside a git repository."""
    return _run_git(directory, ["rev-parse", "--git-dir"]) is not None


def discover_git_roots(files: list[Path]) -> set[str]:
    """Find all git repo roots by walking up from each file's parent.

    Pure filesystem check (.git/ directory), no subprocess calls.
    Returns a set of absolute paths to repo roots.
    """
    roots: set[str] = set()
    checked: set[str] = set()
    for fpath in files:
        d = fpath.parent
        while d != d.parent and str(d) not in checked:
            checked.add(str(d))
            if (d / ".git").is_dir():
                roots.add(str(d))
                break
            d = d.parent
    return roots


def get_repo_root(directory: Path) -> Path | None:
    """Get the root of the git repo containing directory."""
    out = _run_git(directory, ["rev-parse", "--show-toplevel"])
    if out:
        return Path(out.strip())
    return None


def _parse_commits(raw: str, repo_dir: Path) -> list[dict[str, Any]]:
    """Parse git log output into commit dicts.

    Format: header line (NUL-delimited fields), blank line, file list,
    blank line, next commit...
    """
    commits = []
    repo = _repo_name(repo_dir)

    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if "\x00" not in line:
            i += 1
            continue

        parts = line.split("\x00")
        if len(parts) < 5:
            i += 1
            continue

        sha = parts[0]
        author_name = parts[1]
        author_email = parts[2].strip().lower()
        date = parts[3]
        subject = parts[4]
        i += 1

        # Skip blank line between header and file list
        while i < len(lines) and not lines[i].strip():
            i += 1

        # Read file list until blank line or next commit header
        files = []
        while i < len(lines) and lines[i].strip() and "\x00" not in lines[i]:
            files.append(lines[i].strip())
            i += 1

        sha_short = sha[:10]

        commits.append({
            "sha": sha,
            "sha_short": sha_short,
            "author_name": author_name,
            "author_email": author_email,
            "date": date,
            "message": subject,
            "subject": subject,
            "files": files,
            "id": f"git://{repo}#{sha_short}",
        })

    return commits


def get_commits_since(
    repo_dir: Path,
    watermark: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Get commits since watermark (or all if no watermark)."""
    # Security: sanitize limit to prevent negative-number argument injection
    limit = max(1, int(limit))
    args = ["log", f"--format={_LOG_FORMAT}", "--name-only", f"-{limit}"]
    if watermark:
        # Security: validate watermark is a hex SHA to prevent git argument
        # injection via stored watermark values (e.g. "--exec=..." or shell chars)
        if not _VALID_SHA_RE.match(watermark):
            # Invalid/corrupted watermark — fall back to full scan rather
            # than permanently wedging the repo with empty results.
            logger.warning(
                "Invalid watermark SHA %r, ignoring — will do full scan",
                watermark,
            )
        else:
            args.append(f"{watermark}..HEAD")

    raw = _run_git(repo_dir, args, timeout=30)
    if not raw:
        return []

    return _parse_commits(raw, repo_dir)


def get_tags(repo_dir: Path) -> list[dict[str, Any]]:
    """Get annotated and lightweight tags."""
    repo = _repo_name(repo_dir)
    raw = _run_git(
        repo_dir,
        ["tag", "-l", "--format=%(refname:short)%00%(objectname:short)%00%(subject)%00%(creatordate:iso-strict)"],
    )
    if not raw:
        return []

    tags = []
    for line in raw.strip().splitlines():
        parts = line.split("\x00")
        if len(parts) < 2:
            continue
        tag_name = parts[0]
        sha_short = parts[1]
        subject = parts[2] if len(parts) > 2 else f"Release {tag_name}"
        date = parts[3] if len(parts) > 3 else ""

        tags.append({
            "name": tag_name,
            "sha_short": sha_short,
            "subject": subject or f"Release {tag_name}",
            "date": date,
            "id": f"git://{repo}@{tag_name}",
        })

    return tags


def ingest_git_history(
    keeper: Any,
    repo_dir: Path,
    *,
    limit: int = 500,
) -> dict[str, int]:
    """Ingest git commits and tags into the keep store.

    Returns: {"commits": N, "tags": N, "files_tagged": N}
    """
    repo_dir = repo_dir.resolve()
    if not is_git_repo(repo_dir):
        return {"commits": 0, "tags": 0, "files_tagged": 0}

    # Find actual repo root (directory might be a subdirectory)
    root = get_repo_root(repo_dir) or repo_dir
    repo = _repo_name(root)

    # Read watermark from the directory item. The canonical shape is a
    # scalar (the tag is `_singular: "true"` — see .tag/git_watermark),
    # but tolerate pre-existing list-valued tags from stores that
    # predate the singular declaration: pick the most recently-added
    # value and let the SHA validator downstream catch anything weird.
    dir_uri = f"file://{root}"
    dir_item = keeper.get(dir_uri)
    watermark: str | None = None
    if dir_item:
        raw_watermark = (dir_item.tags or {}).get("git_watermark")
        if isinstance(raw_watermark, list):
            watermark = raw_watermark[-1] if raw_watermark else None
        elif isinstance(raw_watermark, str):
            watermark = raw_watermark

    # Get new commits
    commits = get_commits_since(root, watermark=watermark, limit=limit)
    if not commits:
        logger.info("Git ingest %s: no new commits (watermark=%s)", repo, watermark or "none")
        return {"commits": 0, "tags": 0, "files_tagged": 0}

    logger.info("Git ingest %s: %d commits since %s", repo, len(commits), watermark or "initial")

    # Index each commit — skip already-indexed ones to avoid
    # re-enqueuing after-write tasks for unchanged items.
    new_commits = 0
    for commit in commits:
        if keeper.exists(commit["id"]):
            continue

        # Build references to touched files
        refs = []
        for fname in commit["files"]:
            fpath = root / fname
            if fpath.exists():
                refs.append(f"file://{fpath}")

        tags: dict[str, Any] = {
            "git_sha": commit["sha"],
            "git_author": commit["author_name"],
            "author": commit["author_email"],
        }
        if refs:
            tags["references"] = refs

        try:
            keeper.put(
                commit["message"],
                id=commit["id"],
                tags=tags,
                created_at=commit["date"],
            )
            new_commits += 1
        except Exception as e:
            logger.warning("Failed to index commit %s: %s", commit["sha_short"], e)

    # Tag files with their last commit
    files_tagged = 0
    # Build file -> last commit mapping.  Commits are newest-first from
    # git log, so the first occurrence of each file is its latest commit.
    file_last_commit: dict[str, str] = {}
    for commit in commits:
        for fname in commit["files"]:
            fpath = root / fname
            file_uri = f"file://{fpath}"
            if file_uri not in file_last_commit:
                file_last_commit[file_uri] = commit["id"]

    for file_uri, commit_id in file_last_commit.items():
        if keeper.exists(file_uri):
            try:
                keeper.tag(file_uri, tags={"git_commit": commit_id})
                files_tagged += 1
            except Exception as e:
                logger.debug("Failed to tag %s with git_commit: %s", file_uri, e)

    # Index tags/releases — skip already-indexed ones
    git_tags = get_tags(root)
    tags_indexed = 0
    for gt in git_tags:
        if keeper.exists(gt["id"]):
            continue
        try:
            keeper.put(
                gt["subject"],
                id=gt["id"],
                tags={
                    "git_tag": gt["name"],
                },
                created_at=gt["date"] or None,
            )
            tags_indexed += 1
        except Exception as e:
            logger.debug("Failed to index tag %s: %s", gt["name"], e)

    # Update watermark to newest commit
    if commits:
        newest_sha = commits[0]["sha"]
        try:
            if not keeper.exists(dir_uri):
                keeper.put(f"Git repository: {repo}", id=dir_uri)
            keeper.tag(dir_uri, tags={"git_watermark": newest_sha})
        except Exception:
            pass

    skipped = len(commits) - new_commits
    if skipped:
        logger.info(
            "Git ingest %s: %d new commits (%d skipped), %d tags, tagged %d files",
            repo, new_commits, skipped, tags_indexed, files_tagged,
        )
    else:
        logger.info(
            "Git ingest %s: indexed %d commits, %d tags, tagged %d files",
            repo, new_commits, tags_indexed, files_tagged,
        )

    return {
        "commits": new_commits,
        "tags": tags_indexed,
        "files_tagged": files_tagged,
    }
