"""Install hooks and protocol blocks for coding tools.

Detects installed tools (Claude Code, Codex, Kiro) and installs
the keep protocol block and hooks into their global configuration.

Tracked in keep.toml [integrations] so each tool is handled once.
These installers are for explicit setup flows only; normal commands must not
modify unrelated tool configuration without consent.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import StoreConfig

logger = logging.getLogger(__name__)

# Marker text to detect if protocol block is already installed
PROTOCOL_BLOCK_MARKER = "## keep: Reflective Memory"

# The protocol block content (matches SKILL.md Layer 0)
PROTOCOL_BLOCK = """\
## keep: Reflective Memory

*Your actions have consequences. Context is fleeting. Pay attention to what is being asked.*

**The practice is `keep_prompt(name="reflect")`.** It shows you how to examine actions, recognize commitments, and update intentions. Run it. Follow the steps. Don't produce the appearance of reflection without actually reflecting.

**Three tools:** `keep_flow` (all operations), `keep_help` (documentation), `keep_prompt` (practice prompts).

```
keep_prompt(name="reflect")                                                               # The practice
keep_flow(state="get", params={item_id: "now"}, token_budget=2000)                        # Current intentions
keep_flow(state="query-resolve", params={query: "topic"}, token_budget=2000)              # What do I know?
keep_flow(state="put", params={content: "what's true now", id: "now"})                    # Update intentions
keep_flow(state="put", params={content: "learning", tags: {kind: "learning"}})            # Capture insight
keep_flow(state="put", params={uri: "https://example.com/doc", tags: {topic: "X"}})       # Index a document
keep_flow(state="tag", params={id: "item", tags: {reviewed: "true"}})                     # Tag an item
keep_help(topic="flow-actions")                                                            # Action reference
```

**No MCP?** Every flow has a CLI equivalent: `keep flow get -p item_id=now`, `keep flow query-resolve -p query=X`, `keep flow put -p content=X`. Use Bash to run them.

**Protocol:** Reflect using `keep` tools — before, during, and after action. Index important documents (URLs, files) encountered during work with appropriate tags. When writing a plan, incorporate this practice into the plan itself. If the practice is unclear, use `keep_help(topic="index")`.
"""

# Bump this when hook definitions change — triggers re-install for existing users
HOOKS_VERSION = 13

# Claude Code plugin marketplace URL
CLAUDE_CODE_MARKETPLACE_URL = "https://github.com/generalbusiness-ai/keep.git"
CLAUDE_CODE_MARKETPLACE_NAME = "generalbusiness-ai"
CLAUDE_CODE_PLUGIN_NAME = "keep"

# Hook definitions for Claude Code
CLAUDE_CODE_HOOKS = {
    "SessionStart": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "keep prompt session-start </dev/null 2>/dev/null || true",
                    "statusMessage": "Reflecting...",
                }
            ],
        },
    ],
    "UserPromptSubmit": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "keep now 'User: ${.prompt|text}' --truncate -t type=conversation 2>/dev/null || true",
                    "statusMessage": "Reflecting...",
                }
            ],
        }
    ],
    "SubagentStart": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "keep prompt subagent-start </dev/null 2>/dev/null || true",
                    "statusMessage": "Loading context...",
                }
            ],
        }
    ],
    "SessionEnd": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "jq -r '.session_id // empty' 2>/dev/null | xargs -I{} keep move session-{} -t session={} 2>/dev/null || true",
                }
            ],
        }
    ],
}


# Tool definitions: key → (config dir relative to home, installer function name)
TOOL_CONFIGS = {
    "claude_code": ".claude",
    "codex": ".codex",
    "kiro": ".kiro",
    "openclaw": ".openclaw",
    "github_copilot": ".copilot",
}


def detect_new_tools(already_known: dict[str, Any]) -> dict[str, Path]:
    """Detect installed coding tools needing install or upgrade.

    A tool needs work if:
    - Not in config yet (new install)
    - Version in config is less than HOOKS_VERSION (upgrade)

    Returns dict mapping tool key to config directory path.
    """
    home = Path.home()
    tools: dict[str, Path] = {}

    for key, dirname in TOOL_CONFIGS.items():
        known_version = already_known.get(key)
        if isinstance(known_version, int) and known_version < 0:
            continue  # Explicitly opted out via wizard — skip permanently
        if isinstance(known_version, int) and known_version >= HOOKS_VERSION:
            continue  # Up to date — skip the stat
        # True (legacy boolean) or missing or old version → check for tool
        tool_dir = home / dirname
        if tool_dir.is_dir():
            tools[key] = tool_dir

    return tools


def _strip_protocol_block(content: str) -> str:
    """Remove the existing keep protocol block from markdown content.

    Strips from the PROTOCOL_BLOCK_MARKER line to the next `## ` heading
    or end of file. Removes trailing blank lines left by the removal.
    """
    # Match from the marker line to the next ## heading or EOF
    pattern = re.compile(
        r"(?m)^" + re.escape(PROTOCOL_BLOCK_MARKER) + r".*?"
        r"(?=^## |\Z)",
        re.DOTALL,
    )
    content = pattern.sub("", content)
    # Clean up trailing whitespace left by removal
    content = content.rstrip("\n")
    if content:
        content += "\n"
    return content


def _install_protocol_block(target_file: Path) -> bool:
    """Install or upgrade the protocol block in a markdown file.

    If the marker is present, strips the old block and appends the new one
    (upgrade). If absent, appends the block (new install).

    Returns True if the file was written, False if already up to date.
    """
    content = ""
    if target_file.exists():
        content = target_file.read_text(encoding="utf-8")
        if PROTOCOL_BLOCK in content:
            return False  # Already has the current block
        if PROTOCOL_BLOCK_MARKER in content:
            # Upgrade: strip old block, will append new one below
            content = _strip_protocol_block(content)

    # Ensure the file ends with a newline before appending
    if content and not content.endswith("\n"):
        content += "\n"
    if content:
        content += "\n"
    content += PROTOCOL_BLOCK

    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(content, encoding="utf-8")
    return True


def _is_keep_hook_group(hook_group: dict) -> bool:
    """Check if a hook group belongs to keep (contains 'keep now' or 'keep prompt')."""
    for hook in hook_group.get("hooks", []):
        cmd = hook.get("command", "") if isinstance(hook, dict) else ""
        if "keep now" in cmd or "keep get" in cmd or "keep reflect" in cmd or "keep prompt" in cmd:
            return True
    return False


def _strip_keep_hooks(existing_hooks: dict) -> dict:
    """Remove all keep hook groups from existing hooks, preserving user hooks."""
    cleaned: dict[str, list] = {}
    for event, hook_groups in existing_hooks.items():
        if not isinstance(hook_groups, list):
            cleaned[event] = hook_groups
            continue
        kept = [g for g in hook_groups if not (isinstance(g, dict) and _is_keep_hook_group(g))]
        if kept:
            cleaned[event] = kept
        # Drop empty event lists (clean up events that only had keep hooks)
    return cleaned


def _install_claude_code_hooks(settings_file: Path) -> bool:
    """Install keep hooks into Claude Code settings.json.

    Strips any existing keep hooks first (upgrade-safe), then merges
    current hook definitions. Returns True if file was written.
    """
    settings: dict[str, Any] = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cannot parse %s, skipping hook install: %s", settings_file, e)
            return False

    existing_hooks = settings.get("hooks", {})

    # Strip old keep hooks before installing new ones
    existing_hooks = _strip_keep_hooks(existing_hooks)

    # Merge new hook definitions
    for event, hook_list in CLAUDE_CODE_HOOKS.items():
        if event not in existing_hooks:
            existing_hooks[event] = []
        existing_hooks[event].extend(hook_list)

    settings["hooks"] = existing_hooks
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    return True


def _cleanup_claude_code_legacy(config_dir: Path) -> list[str]:
    """Remove legacy keep integration from Claude Code config files.

    Strips protocol block from CLAUDE.md and keep hooks from settings.json.
    Returns list of cleanup actions taken.
    """
    cleaned = []

    # Strip protocol block from CLAUDE.md
    claude_md = config_dir / "CLAUDE.md"
    if claude_md.exists():
        content = claude_md.read_text(encoding="utf-8")
        if PROTOCOL_BLOCK_MARKER in content:
            content = _strip_protocol_block(content)
            claude_md.write_text(content, encoding="utf-8")
            cleaned.append("removed legacy protocol block")

    # Strip keep hooks from settings.json
    settings_json = config_dir / "settings.json"
    if settings_json.exists():
        try:
            settings = json.loads(settings_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cleaned
        existing_hooks = settings.get("hooks", {})
        stripped = _strip_keep_hooks(existing_hooks)
        if stripped != existing_hooks:
            if stripped:
                settings["hooks"] = stripped
            else:
                settings.pop("hooks", None)
            settings_json.write_text(
                json.dumps(settings, indent=2) + "\n", encoding="utf-8"
            )
            cleaned.append("removed legacy hooks")

    return cleaned


def _try_install_claude_code_plugin() -> bool:
    """Try to install the keep plugin via claude CLI.

    Runs `claude plugin marketplace add` and `claude plugin install`.
    Uses a short timeout to avoid blocking. Returns True on success.
    """
    claude = shutil.which("claude")
    if not claude:
        return False

    try:
        # Add marketplace (idempotent)
        subprocess.run(
            [claude, "plugin", "marketplace", "add", CLAUDE_CODE_MARKETPLACE_URL],
            timeout=30,
            capture_output=True,
        )
        # Install plugin (idempotent)
        result = subprocess.run(
            [claude, "plugin", "install",
             f"{CLAUDE_CODE_PLUGIN_NAME}@{CLAUDE_CODE_MARKETPLACE_NAME}"],
            timeout=30,
            capture_output=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("claude plugin install failed: %s", e)
        return False


def install_claude_code(config_dir: Path) -> list[str]:
    """Install keep integration for Claude Code.

    Tries plugin-based install first (via claude CLI).
    Falls back to legacy protocol block + hooks if CLI unavailable.
    Cleans up legacy integration when migrating to plugin.

    Returns list of actions taken.
    """
    actions = []

    # Try plugin install via claude CLI
    if _try_install_claude_code_plugin():
        actions.append("plugin")
        # Clean up legacy integration if present
        cleanup = _cleanup_claude_code_legacy(config_dir)
        actions.extend(cleanup)
    else:
        # Fall back to legacy: protocol block + hooks
        claude_md = config_dir / "CLAUDE.md"
        if _install_protocol_block(claude_md):
            actions.append("protocol block")
        settings_json = config_dir / "settings.json"
        if _install_claude_code_hooks(settings_json):
            actions.append("hooks")

    return actions


def _install_codex_mcp_server(config_toml: Path) -> bool:
    """Add or update `[mcp_servers.keep]` in Codex's config.toml.

    Codex.app launched from Finder does NOT inherit the user's shell env,
    so `KEEP_STORE_PATH` is unavailable when codex spawns `keep mcp`.
    Bake the resolved store into `--store` so the right store is used
    regardless of how codex is launched. Mirrors install_github_copilot.

    Returns True if the file was written (new or updated).
    """
    import tomli_w

    from .daemon_client import resolve_store_path

    data: dict[str, Any] = {}
    if config_toml.exists():
        try:
            import tomllib

            with open(config_toml, "rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError) as e:
            logger.warning("Cannot parse %s, skipping MCP install: %s", config_toml, e)
            return False

    servers = data.setdefault("mcp_servers", {})
    keep_entry = {
        "command": "keep",
        # Bake the resolved store in — codex strips shell env on launch
        # from Finder, so an env-only override is not reliable.
        "args": ["--store", str(resolve_store_path()), "mcp"],
    }
    if servers.get("keep") == keep_entry:
        return False  # Already up to date

    servers["keep"] = keep_entry
    config_toml.parent.mkdir(parents=True, exist_ok=True)
    with open(config_toml, "wb") as f:
        tomli_w.dump(data, f)
    return True


def install_codex(config_dir: Path) -> list[str]:
    """Install protocol block and MCP server config for OpenAI Codex.

    Returns list of actions taken.
    """
    actions = []

    agents_md = config_dir / "AGENTS.md"
    if _install_protocol_block(agents_md):
        actions.append("protocol block")

    config_toml = config_dir / "config.toml"
    if _install_codex_mcp_server(config_toml):
        actions.append("MCP server")

    return actions


def _install_kiro_hooks(config_dir: Path) -> bool:
    """Install keep hooks into Kiro hooks directory.

    Copies .kiro.hook files from package data to ~/.kiro/hooks/.
    Returns True if any file was written.
    """
    hooks_dir = config_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Source hook files from package data
    source_dir = Path(__file__).parent / "data" / "kiro-hooks"
    wrote = False
    for src in source_dir.glob("*.kiro.hook"):
        dst = hooks_dir / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        wrote = True

    return wrote


def install_kiro(config_dir: Path) -> list[str]:
    """Install protocol block and hooks for Kiro.

    Steering file goes in ~/.kiro/steering/keep.md.
    Hooks go in ~/.kiro/hooks/*.kiro.hook (one per event).

    Returns list of actions taken.
    """
    actions = []

    steering_md = config_dir / "steering" / "keep.md"
    if _install_protocol_block(steering_md):
        actions.append("steering")

    if _install_kiro_hooks(config_dir):
        actions.append("hooks")

    return actions


def _upgrade_openclaw_plugin(plugin_dir: Path) -> bool:
    """Copy plugin files from package data to an existing plugin directory.

    Only overwrites files that ship with keep (index.ts, package.json,
    openclaw.plugin.json). Never creates directories or touches other
    plugins' files.

    Returns True if any file was written.
    """
    source_dir = Path(__file__).parent / "data" / "openclaw-plugin"
    wrote = False
    for src in source_dir.iterdir():
        if src.is_file():
            dst = plugin_dir / src.name
            new_content = src.read_text(encoding="utf-8")
            # Skip write if content is identical
            if dst.exists() and dst.read_text(encoding="utf-8") == new_content:
                continue
            dst.write_text(new_content, encoding="utf-8")
            wrote = True
    return wrote


def install_openclaw(config_dir: Path) -> list[str]:
    """Auto-upgrade keep plugin for OpenClaw.

    Only upgrades if the plugin is already installed (plugin_dir exists).
    Initial install still requires: openclaw plugins install -l $(keep config openclaw-plugin)

    Returns list of actions taken.
    """
    actions = []

    # Auto-upgrade plugin files only if already installed
    plugin_dir = config_dir / "extensions" / "keep"
    if plugin_dir.is_dir():
        if _upgrade_openclaw_plugin(plugin_dir):
            actions.append("plugin")

    return actions


def install_github_copilot(config_dir: Path) -> list[str]:
    """Install MCP server config for GitHub Copilot CLI.

    Adds/updates the keep entry in ~/.copilot/mcp-config.json.
    Returns list of actions taken.
    """
    from .daemon_client import resolve_store_path

    actions = []

    # Current Copilot CLI uses ~/.copilot/mcp-config.json for MCP servers.
    mcp_json = config_dir / "mcp-config.json"
    data: dict[str, Any] = {}
    if mcp_json.exists():
        try:
            data = json.loads(mcp_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cannot parse %s, skipping: %s", mcp_json, e)
            return actions

    servers = data.setdefault("mcpServers", {})
    keep_entry = {
        "type": "local",
        "command": "keep",
        # Bake in the resolved store because MCP hosts often launch stdio
        # servers with a scrubbed environment.
        "args": ["--store", str(resolve_store_path()), "mcp"],
        "tools": ["*"],
    }
    if servers.get("keep") == keep_entry:
        return actions  # Already up to date

    servers["keep"] = keep_entry
    mcp_json.parent.mkdir(parents=True, exist_ok=True)
    mcp_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    actions.append("MCP server")
    return actions


def check_and_install(config: "StoreConfig") -> None:
    """Explicitly check for coding tools and install integrations if needed.

    Fast path: one stat per unknown tool (tools already in config are skipped).
    When all tools in TOOL_CONFIGS are accounted for, this does zero I/O.
    """
    from .config import save_config

    # Bypass via environment variable
    if os.environ.get("KEEP_NO_SETUP"):
        return

    # Detect only tools not yet in config (one stat each)
    new_tools = detect_new_tools(config.integrations)
    if not new_tools:
        return  # All known tools handled, or none installed

    # Install integrations for newly detected tools
    installers = {
        "claude_code": install_claude_code,
        "codex": install_codex,
        "kiro": install_kiro,
        "openclaw": install_openclaw,
        "github_copilot": install_github_copilot,
    }

    for key, tool_dir in new_tools.items():
        installer = installers.get(key)
        if installer:
            actions = installer(tool_dir)
            if actions:
                upgrading = key in config.integrations
                verb = "upgraded" if upgrading else "installed"
                print(
                    f"keep: {verb} {' and '.join(actions)} for {key} ({tool_dir}/)",
                    file=sys.stderr,
                )
                if key == "openclaw" and "plugin" in actions:
                    print(
                        "keep: run 'openclaw gateway restart' to load updated plugin",
                        file=sys.stderr,
                    )
            config.integrations[key] = HOOKS_VERSION
        else:
            # Detected but no installer
            config.integrations[key] = 0
            logger.info(f"{key} detected but no installer defined")

    save_config(config)
