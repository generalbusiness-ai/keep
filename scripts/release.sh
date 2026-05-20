#!/usr/bin/env bash
# release.sh — Build, tag, push, and publish a keep release.
#
# Usage:
#   scripts/release.sh patch    # 0.111.1 → 0.111.2
#   scripts/release.sh minor    # 0.111.1 → 0.112.0
#   scripts/release.sh 0.112.0  # explicit version
#
# Steps:
#   1. Bump version (scripts/bump_version.py)
#   2. Commit + tag
#   3. Build sdist + wheel
#   4. Push to origin with tags
#   5. Upload to PyPI via twine
#   6. Create GitHub release with formatted notes
#
# Requires: python3, uv, gh, git
# PyPI credentials: ~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD env vars

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Resolve version
# ---------------------------------------------------------------------------

current_version() {
  python3 -c "
import re
text = open('pyproject.toml').read()
m = re.search(r'version\s*=\s*\"([^\"]+)\"', text)
print(m.group(1))
"
}

OLD_VERSION=$(current_version)
IFS='.' read -r MAJOR MINOR PATCH <<< "$OLD_VERSION"

case "${1:-}" in
  patch)
    NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
    RELEASE_TYPE="patch"
    ;;
  minor)
    NEW_VERSION="$MAJOR.$((MINOR + 1)).0"
    RELEASE_TYPE="minor"
    ;;
  ""|--help|-h)
    echo "Usage: scripts/release.sh {patch|minor|X.Y.Z}"
    echo "Current version: $OLD_VERSION"
    exit 0
    ;;
  *)
    NEW_VERSION="$1"
    # Determine type from version comparison
    NEW_PATCH="${NEW_VERSION##*.}"
    if [ "$NEW_PATCH" = "0" ]; then
      RELEASE_TYPE="minor"
    else
      RELEASE_TYPE="patch"
    fi
    ;;
esac

TAG="v$NEW_VERSION"

echo "=== Release $OLD_VERSION → $NEW_VERSION ($RELEASE_TYPE) ==="
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: Working tree has uncommitted changes. Commit or stash first."
  exit 1
fi

if git tag -l "$TAG" | grep -q .; then
  echo "ERROR: Tag $TAG already exists."
  exit 1
fi

if ! command -v gh &>/dev/null; then
  echo "ERROR: gh CLI not found. Install: https://cli.github.com/"
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Bump version
# ---------------------------------------------------------------------------

echo "--- Bumping version ---"
python3 scripts/bump_version.py "$NEW_VERSION"
echo ""

# ---------------------------------------------------------------------------
# 2. Commit + tag
# ---------------------------------------------------------------------------

echo "--- Committing ---"

# Build the commit message from git log since last tag
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
COMMIT_SUBJECT="$TAG — $(git log --oneline -1 --format='%s' | sed "s/^v[0-9.]* — //")"

# Stage only the files that bump_version.py touches (plus uv.lock, which
# bump_version.py regenerates via `uv lock`). Using `git add -A` here would
# silently sweep any unrelated working-tree changes into the release commit —
# keep this list in sync with TARGETS in scripts/bump_version.py.
#
# Note: keep/data/openclaw-plugin/dist/index.js is a build artifact, gitignored
# via the nested .gitignore, and force-included in the wheel by pyproject.toml
# — deliberately not staged here.
git add \
    pyproject.toml \
    uv.lock \
    SKILL.md \
    keep/data/openclaw-plugin/openclaw.plugin.json \
    keep/data/openclaw-plugin/package.json \
    keep/data/openclaw-plugin/package-lock.json \
    keep/data/openclaw-plugin/src/index.ts \
    keep/data/openclaw-plugin/src/mcp-transport.ts \
    claude-code-plugin/.claude-plugin/plugin.json

git commit -m "$COMMIT_SUBJECT"
git tag "$TAG"

echo "Tagged $TAG"
echo ""

# ---------------------------------------------------------------------------
# 3. Build
# ---------------------------------------------------------------------------

echo "--- Building ---"
rm -rf dist/
uv build
echo ""

# Verify expected files exist
SDIST="dist/keep_skill-${NEW_VERSION}.tar.gz"
WHEEL=$(ls dist/keep_skill-"${NEW_VERSION}"-*.whl 2>/dev/null | head -1)

if [ ! -f "$SDIST" ]; then
  echo "ERROR: Expected $SDIST not found"
  exit 1
fi
if [ -z "$WHEEL" ]; then
  echo "ERROR: No wheel found for $NEW_VERSION"
  exit 1
fi

echo "Built: $SDIST"
echo "Built: $WHEEL"
echo ""

# ---------------------------------------------------------------------------
# 4. Push
# ---------------------------------------------------------------------------

echo "--- Pushing ---"
git push origin main "$TAG"
echo ""

# ---------------------------------------------------------------------------
# 5. Upload to PyPI
# ---------------------------------------------------------------------------

echo "--- Uploading to PyPI ---"
uvx twine upload "dist/keep_skill-${NEW_VERSION}"*
echo ""

# ---------------------------------------------------------------------------
# 6. GitHub release
# ---------------------------------------------------------------------------

echo "--- Creating GitHub release ---"

# Generate release notes based on type
if [ "$RELEASE_TYPE" = "minor" ]; then
  # Minor: structured format with ## What's new header.
  # Use LAST_TAG (most recent tag of any kind) so patch-release commits
  # that already have their own release notes are not repeated here.
  RELEASE_BODY=$(cat <<NOTES
## What's new

$(git log "$LAST_TAG".."$TAG" --pretty=format:'- %s' | grep -v "^- v[0-9]" || true)
NOTES
)
else
  # Patch: concise — summary line then bullet changes from commit body
  COMMIT_BODY=$(git log -1 --format='%b' "$TAG")
  if [ -n "$COMMIT_BODY" ]; then
    RELEASE_BODY="$COMMIT_BODY"
  else
    # Fall back to commit subject
    RELEASE_BODY=$(git log -1 --format='%s' "$TAG")
  fi
fi

gh release create "$TAG" \
  --title "$TAG" \
  --notes "$RELEASE_BODY" \
  "$SDIST" "$WHEEL"

echo ""
echo "=== Released $TAG ==="
echo "  PyPI: https://pypi.org/project/keep-skill/$NEW_VERSION/"
echo "  GitHub: https://github.com/generalbusiness-ai/keep/releases/tag/$TAG"
