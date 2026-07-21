"""Guards against retired tag vocabulary drifting back into active guidance."""

from pathlib import Path
import re


ROOT = Path(__file__).parent.parent

# These values describe the note's reflective content, not its entity type.
# Historical designs and migration fixtures intentionally retain old spellings;
# this guard covers only guidance that users, agents, or integrations consume.
CONTENT_KINDS = (
    "learning",
    "breakdown",
    "gotcha",
    "reference",
    "teaching",
    "meeting",
    "pattern",
    "possibility",
    "decision",
)
ACTIVE_GUIDANCE_ROOTS = (
    ROOT / "docs",
    ROOT / "keep" / "data" / "system",
    ROOT / "claude-code-plugin",
    ROOT / "hermes-plugin",
)
ACTIVE_GUIDANCE_FILES = (
    ROOT / "README.md",
    ROOT / "SKILL.md",
    ROOT / "keep" / "integrations.py",  # Canonical embedded protocol block.
)
_CONTENT_KIND_VALUES = "|".join(CONTENT_KINDS)
LEGACY_CONTENT_KIND_PATTERNS = (
    re.compile(rf"\btype\s*=\s*(?:{_CONTENT_KIND_VALUES})\b"),
    re.compile(
        rf"(?:[\"']type[\"']|\btype)\s*:\s*"
        rf"(?:[\"'](?:{_CONTENT_KIND_VALUES})[\"']|(?:{_CONTENT_KIND_VALUES})\b)"
    ),
)


def _active_guidance_files() -> list[Path]:
    """Return deterministic active guidance inputs for readable failures."""
    files = list(ACTIVE_GUIDANCE_FILES)
    for root in ACTIVE_GUIDANCE_ROOTS:
        files.extend(root.rglob("*.md"))
    return sorted(set(files))


def test_content_kinds_are_not_written_as_type_tags_in_active_guidance():
    """Content-kind examples must use ``kind`` so meta retrieval can find them."""
    failures: list[str] = []

    for path in _active_guidance_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(pattern.search(line) for pattern in LEGACY_CONTENT_KIND_PATTERNS):
                failures.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")

    assert failures == [], "Retired type=<content-kind> guidance found:\n" + "\n".join(failures)


def test_content_kind_drift_guard_recognizes_legacy_example_forms():
    """Keep the guard effective across CLI, Python-like, JSON, and YAML syntax."""
    legacy_examples = (
        "-t type=learning",
        'tags: {type: "breakdown"}',
        '"type": "teaching"',
        "type: reference",
    )

    for example in legacy_examples:
        assert any(pattern.search(example) for pattern in LEGACY_CONTENT_KIND_PATTERNS)

    allowed_examples = ("type=conversation", 'tags: {kind: "learning"}')
    for example in allowed_examples:
        assert not any(pattern.search(example) for pattern in LEGACY_CONTENT_KIND_PATTERNS)
