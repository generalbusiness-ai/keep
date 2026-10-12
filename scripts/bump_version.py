#!/usr/bin/env python3
"""Bump version across all files that carry it.

Usage:
    python scripts/bump_version.py 0.45.0
    python scripts/bump_version.py          # show current version
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every file that carries a version string, with a regex whose group(1) is
# the prefix to preserve and whose match is the full "prefix + old version".
# The optional third element is count (default 1); use 0 for "replace all".
TARGETS = [
    ("pyproject.toml",                                 r'(version\s*=\s*)"[^"]+"'),
    ("SKILL.md",                                       r'(version:\s*)\S+'),
    ("keep/data/openclaw-plugin/openclaw.plugin.json", r'("version":\s*)"[^"]+"'),
    ("keep/data/openclaw-plugin/package.json",         r'("version":\s*)"[^"]+"'),
    ("keep/data/openclaw-plugin/package-lock.json",    r'("version":\s*)"[^"]+"', 2),
    ("keep/data/openclaw-plugin/src/index.ts",             r'(version:\s*)"[^"]+"', 0),
    ("keep/data/openclaw-plugin/src/mcp-transport.ts",   r'(version:\s*)"[^"]+"'),
    ("claude-code-plugin/.claude-plugin/plugin.json",    r'("version":\s*)"[^"]+"'),
]


def current_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'version\s*=\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError("Cannot find version in pyproject.toml")
    return m.group(1)


def bump_file(path: Path, pattern: str, new_version: str, count: int = 1) -> int:
    text = path.read_text()
    if '"' in pattern:
        repl = rf'\g<1>"{new_version}"'
    else:
        repl = rf"\g<1>{new_version}"
    new_text, n = re.subn(pattern, repl, text, count=count)
    if n == 0:
        raise RuntimeError(f"Pattern not found in {path}")
    path.write_text(new_text)
    return n


def main() -> None:
    old = current_version()

    if len(sys.argv) < 2:
        print(f"Current version: {old}")
        print(f"\nUsage: python scripts/bump_version.py <new-version>")
        return

    new = sys.argv[1]
    if new == old:
        print(f"Already at {old}")
        return

    for entry in TARGETS:
        relpath, pattern = entry[0], entry[1]
        count = entry[2] if len(entry) > 2 else 1
        path = ROOT / relpath
        n = bump_file(path, pattern, new, count)
        suffix = f" ({n} occurrences)" if n > 1 else ""
        print(f"  {relpath}: {old} -> {new}{suffix}")

    print(f"\nBumped {len(TARGETS)} files to {new}")
    print(f"(keep/__init__.py reads from package metadata — no edit needed)")

    # Re-resolve the lock file so dependency versions stay current.
    print("\nRunning uv lock ...")
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)

    # Rebuild the OpenClaw plugin so dist/index.js matches the new version.
    # dist/index.js bundles @modelcontextprotocol/sdk and its dependencies, so
    # install exactly what package-lock.json pins first. Without `npm ci` the
    # bundle is built from whatever node_modules happens to be on disk, and a
    # lockfile security bump silently fails to reach the published wheel
    # (this happened in v0.163.1).
    plugin_dir = ROOT / "keep" / "data" / "openclaw-plugin"
    print("\nInstalling locked OpenClaw plugin dependencies ...")
    subprocess.run(["npm", "ci"], cwd=plugin_dir, check=True)
    print("\nRebuilding OpenClaw plugin ...")
    subprocess.run(["node", "build.mjs"], cwd=plugin_dir, check=True)


if __name__ == "__main__":
    main()
