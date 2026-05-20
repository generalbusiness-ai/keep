"""Generate MCPB (MCP Bundle) for Claude Desktop.

Creates a .mcpb file that tells Claude Desktop how to connect to
a locally-installed keep MCP server.
"""

import importlib.metadata
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional


def _get_keep_path() -> str:
    """Resolve the full path to the keep executable."""
    path = shutil.which("keep")
    if not path:
        raise FileNotFoundError(
            "Cannot find 'keep' on PATH. Install keep-skill first: pip install keep-skill"
        )
    return path


def _get_version() -> str:
    """Get keep-skill version from package metadata or pyproject.toml fallback."""
    try:
        return importlib.metadata.version("keep-skill")
    except importlib.metadata.PackageNotFoundError:
        pass
    # Fallback: read from pyproject.toml next to this package
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip('"')
    return "0.0.0"


# Tool descriptions for the manifest (informational — the MCP server
# reports its own tools at runtime, but listing them here lets Claude
# Desktop show what's available before connecting).
TOOLS = [
    {"name": "keep_flow", "description": "Run a state-doc flow: search, put, get, tag, delete, move, stats — all operations."},
    {"name": "keep_prompt", "description": "Render an agent prompt with context injected from memory."},
    {"name": "keep_help", "description": "Browse keep documentation."},
]


def generate_manifest(keep_path: Optional[str] = None) -> dict:
    """Generate the MCPB manifest.json with the local keep path baked in."""
    if keep_path is None:
        keep_path = _get_keep_path()

    return {
        "manifest_version": "0.3",
        "name": "keep",
        "display_name": "keep",
        "version": _get_version(),
        "description": "Reflective Memory for AI",
        "long_description": (
            "Reflective memory with semantic search. "
            "Store facts, preferences, decisions, and documents. "
            "Search by meaning. Persist context across sessions."
        ),
        "author": {"name": "Hugh Pyle, inguz \u16dc outcomes llc"},
        "repository": {
            "type": "git",
            "url": "https://github.com/generalbusiness-ai/keep.git",
        },
        "homepage": "https://github.com/generalbusiness-ai/keep",
        "documentation": "https://docs.generalbusiness.ai/guides/",
        "license": "MIT",
        "keywords": ["memory", "reflection", "semantic-search", "context"],
        "icon": "icon.png",
        "server": {
            "type": "binary",
            "entry_point": "server/keep-mcp",
            "mcp_config": {
                "command": "${__dirname}/server/keep-mcp",
                "args": [],
            },
        },
        "tools": TOOLS,
    }


def generate_mcpb(
    output_path: Optional[Path] = None,
    store_path: Optional[Path] = None,
) -> Path:
    """Generate a .mcpb bundle file.

    Args:
        output_path: Where to write the .mcpb file.
                     Defaults to a temp directory.
        store_path: Resolved store path to bake into the wrapper script.
                    Claude Desktop runs outside the user's shell, so the
                    wrapper must explicitly pass ``--store``.

    Returns:
        Path to the generated .mcpb file.
    """
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / "keep.mcpb"

    keep_path = _get_keep_path()
    manifest = generate_manifest(keep_path)

    # Build the zip archive
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        # Create wrapper script that delegates to the installed keep.
        # Always bake in --store so Claude Desktop (which runs outside the
        # user's shell) connects to the correct store.
        server_dir = Path(tmpdir) / "server"
        server_dir.mkdir()
        wrapper = server_dir / "keep-mcp"
        store_arg = f' --store "{store_path}"' if store_path else ""
        wrapper.write_text(
            f"#!/bin/sh\nexec {keep_path} mcp{store_arg} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)

        # Include icon if available
        icon_src = Path(__file__).parent / "data" / "mcpb" / "icon.png"

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, "manifest.json")
            zf.write(wrapper, "server/keep-mcp")
            if icon_src.exists():
                zf.write(icon_src, "icon.png")

    return output_path
