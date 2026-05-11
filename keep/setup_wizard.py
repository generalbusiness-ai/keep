"""Interactive first-run setup wizard.

Runs when no store exists yet. Detects available providers and coding
tools, presents choices, and creates the initial config.

Non-interactive fallback: if stdin is not a TTY, uses current silent
auto-detect behavior (no prompts).
"""

import os
import platform
import sys
from pathlib import Path
from typing import Any, Optional

from .config import (
    ProviderConfig,
    StoreConfig,
    _detect_ollama,
    _ollama_pick_models,
    detect_default_providers,
    get_default_provider_model,
    save_config,
)
from .daemon_client import stop_daemon
from .integrations import HOOKS_VERSION, TOOL_CONFIGS, detect_new_tools, install_claude_code, install_codex, install_github_copilot, install_kiro, install_openclaw


# --- Provider definitions (single source of truth for display names / models) ---

EMBEDDING_PROVIDERS: list[dict[str, Any]] = [
    # Ollama is handled dynamically (model depends on what's installed)
    {
        "key": "voyage",
        "name": "Voyage",
        "provider": "voyage",
        "env_keys": ["VOYAGE_API_KEY"],
        "env_hint": "VOYAGE_API_KEY",
    },
    {
        "key": "openai",
        "name": "OpenAI",
        "provider": "openai",
        "env_keys": ["KEEP_OPENAI_API_KEY", "OPENAI_API_KEY"],
        "env_hint": "OPENAI_API_KEY",
    },
    {
        "key": "gemini",
        "name": "Gemini",
        "provider": "gemini",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"],
        "env_hint": "GEMINI_API_KEY",
    },
    {
        "key": "mistral",
        "name": "Mistral",
        "provider": "mistral",
        "env_keys": ["MISTRAL_API_KEY"],
        "env_hint": "MISTRAL_API_KEY",
    },
    {
        "key": "openrouter",
        "name": "OpenRouter",
        "provider": "openrouter",
        "env_keys": ["OPENROUTER_API_KEY"],
        "env_hint": "OPENROUTER_API_KEY",
        "show_when_available_only": True,
    },
]

SUMMARIZATION_PROVIDERS: list[dict[str, Any]] = [
    # Ollama is handled dynamically
    {
        "key": "anthropic",
        "name": "Anthropic",
        "provider": "anthropic",
        "env_keys": ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"],
        "env_hint": "ANTHROPIC_API_KEY",
    },
    {
        "key": "openai",
        "name": "OpenAI",
        "provider": "openai",
        "env_keys": ["KEEP_OPENAI_API_KEY", "OPENAI_API_KEY"],
        "env_hint": "OPENAI_API_KEY",
    },
    {
        "key": "gemini",
        "name": "Gemini",
        "provider": "gemini",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"],
        "env_hint": "GEMINI_API_KEY",
    },
    {
        "key": "mistral",
        "name": "Mistral",
        "provider": "mistral",
        "env_keys": ["MISTRAL_API_KEY"],
        "env_hint": "MISTRAL_API_KEY",
    },
    {
        "key": "openrouter",
        "name": "OpenRouter",
        "provider": "openrouter",
        "env_keys": ["OPENROUTER_API_KEY"],
        "env_hint": "OPENROUTER_API_KEY",
        "show_when_available_only": True,
    },
]


def _has_env(*keys: str) -> bool:
    """Check if any of the given environment variables are set."""
    return any(os.environ.get(k) for k in keys)


def _is_interactive() -> bool:
    """Check if we're running in an interactive terminal."""
    return sys.stdin.isatty() and sys.stderr.isatty()


def _patch_question(question):
    """Patch a questionary Question: add Esc binding + remove inverted styles."""
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.styles import merge_styles
    from prompt_toolkit.styles.defaults import default_pygments_style

    app = question.application

    # Add Escape key to abort
    @app.key_bindings.add(Keys.Escape, eager=True)
    def _(event):
        event.app.exit(exception=KeyboardInterrupt)

    # Replace the merged style to remove prompt_toolkit's default_ui_style()
    # which sets 'selected: reverse' (causes white-background inversion).
    # Keep pygments defaults + questionary's own style.
    new_style = merge_styles([
        default_pygments_style(),
        app.style,
    ])
    app._merged_style = new_style
    app.renderer.style = new_style

    return question


def detect_embedding_choices(current: Optional[str] = None) -> list[dict[str, Any]]:
    """Build the list of embedding provider choices.

    Each choice is a dict with:
      - name: display name
      - value: (provider_name, params) tuple or None
      - available: bool
      - hint: status hint text
      - default: whether this should be pre-selected

    Args:
        current: Provider name from existing config (used as default when re-running setup).
    """
    choices = []

    # Check environment
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    local_only = bool(os.environ.get("KEEP_LOCAL_ONLY"))

    # 1. Ollama
    ollama = _detect_ollama()
    if ollama:
        embed_model, _ = _ollama_pick_models(ollama["models"])
        params: dict[str, Any] = {"model": embed_model}
        if ollama["base_url"] != "http://localhost:11434":
            params["base_url"] = ollama["base_url"]
        choices.append({
            "name": f"Ollama ({embed_model})",
            "value": ("ollama", params),
            "available": True,
            "hint": "local, no API key",
            "default": current == "ollama" if current else True,
        })

    # 2. API providers (unless KEEP_LOCAL_ONLY)
    if not local_only:
        has_default = any(c["default"] for c in choices)
        for p in EMBEDDING_PROVIDERS:
            model = get_default_provider_model("embedding", p["provider"])
            display = f"{p['name']} ({model})" if model else p["name"]
            available = _has_env(*p["env_keys"])
            if p.get("show_when_available_only") and not available:
                continue
            params = {"model": model} if model else {}
            if current:
                is_default = current == p["provider"]
            else:
                is_default = available and not has_default
            if is_default:
                has_default = True
            choices.append({
                "name": display,
                "value": (p["provider"], params) if available else None,
                "available": available,
                "hint": f"{p['env_hint']} found" if available else f"requires {p['env_hint']}",
                "default": is_default,
            })

    # 3. Local models
    if is_apple_silicon:
        try:
            import mlx.core  # noqa
            import sentence_transformers  # noqa
            mlx_model = get_default_provider_model("embedding", "mlx") or "all-MiniLM-L6-v2"
            choices.append({
                "name": f"MLX on Apple Silicon ({mlx_model})",
                "value": ("mlx", {"model": mlx_model}),
                "available": True,
                "hint": "Apple Silicon, local",
                "default": current == "mlx" if current else False,
            })
        except ImportError:
            mlx_model = get_default_provider_model("embedding", "mlx") or "all-MiniLM-L6-v2"
            choices.append({
                "name": f"MLX on Apple Silicon ({mlx_model})",
                "value": None,
                "available": False,
                "hint": "requires: uv tool install 'keep-skill[local]'",
                "default": False,
            })

    try:
        import sentence_transformers  # noqa
        st_model = get_default_provider_model("embedding", "sentence-transformers") or "all-MiniLM-L6-v2"
        choices.append({
            "name": f"sentence-transformers ({st_model})",
            "value": ("sentence-transformers", {"model": st_model}),
            "available": True,
            "hint": "local, no API key",
            "default": current == "sentence-transformers" if current else False,
        })
    except ImportError:
        st_model = get_default_provider_model("embedding", "sentence-transformers") or "all-MiniLM-L6-v2"
        choices.append({
            "name": f"sentence-transformers ({st_model})",
            "value": None,
            "available": False,
            "hint": "requires: uv tool install 'keep-skill[local]'",
            "default": False,
        })

    return choices


def detect_summarization_choices(current: Optional[str] = None) -> list[dict[str, Any]]:
    """Build the list of summarization provider choices.

    Args:
        current: Provider name from existing config (used as default when re-running setup).
    """
    choices = []

    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    local_only = bool(os.environ.get("KEEP_LOCAL_ONLY"))

    # 1. Ollama
    ollama = _detect_ollama()
    if ollama:
        # Filter out non-generative models (OCR, embedding, vision-only).
        # If the server is fresh and has no models yet, offer the configured
        # default; the Ollama provider will pull it on first use.
        _non_generative = ("embed", "ocr", "moondream", "bakllava")
        gen_models = [
            m for m in ollama["models"]
            if not any(k in m.split(":")[0] for k in _non_generative)
        ]
        chat_model = (
            gen_models[0]
            if gen_models
            else get_default_provider_model("summarization", "ollama") if not ollama["models"] else None
        )
        if chat_model:
            params: dict[str, Any] = {"model": chat_model}
            if ollama["base_url"] != "http://localhost:11434":
                params["base_url"] = ollama["base_url"]
            choices.append({
                "name": f"Ollama ({chat_model})",
                "value": ("ollama", params),
                "available": True,
                "hint": "local, no API key",
                "default": current == "ollama" if current else True,
            })

    # 2. API providers
    if not local_only:
        for p in SUMMARIZATION_PROVIDERS:
            model = get_default_provider_model("summarization", p["provider"])
            display = f"{p['name']} ({model})" if model else p["name"]
            available = _has_env(*p["env_keys"])
            if p.get("show_when_available_only") and not available:
                continue
            params = {"model": model} if model else {}
            if current:
                is_default = current == p["provider"]
            else:
                is_default = available and not any(c.get("default") for c in choices)
            choices.append({
                "name": display,
                "value": (p["provider"], params) if available else None,
                "available": available,
                "hint": f"{p['env_hint']} found" if available else f"requires {p['env_hint']}",
                "default": is_default,
            })

    # 3. Local MLX
    if is_apple_silicon:
        try:
            import mlx_lm  # noqa
            mlx_model = get_default_provider_model("summarization", "mlx") or "mlx-community/Llama-3.2-3B-Instruct-4bit"
            choices.append({
                "name": f"MLX on Apple Silicon ({mlx_model})",
                "value": ("mlx", {"model": mlx_model}),
                "available": True,
                "hint": "Apple Silicon, local",
                "default": current == "mlx" if current else False,
            })
        except ImportError:
            mlx_model = get_default_provider_model("summarization", "mlx") or "mlx-community/Llama-3.2-3B-Instruct-4bit"
            choices.append({
                "name": f"MLX on Apple Silicon ({mlx_model})",
                "value": None,
                "available": False,
                "hint": "requires: uv tool install 'keep-skill[local]'",
                "default": False,
            })

    # 4. Truncate fallback (always available)
    choices.append({
        "name": "None (truncate only)",
        "value": ("truncate", {}),
        "available": True,
        "hint": "no summarization, basic mode",
        "default": current == "truncate" if current else not any(c.get("default") for c in choices),
    })

    return choices


def detect_tool_choices() -> list[dict[str, Any]]:
    """Build list of coding tool integration choices.

    Returns dicts with:
      - key: tool key (e.g. "claude_code")
      - name: display name
      - config_dir: Path to config dir (or None if not found)
      - found: bool
    """
    home = Path.home()
    display_names = {
        "claude_code": "Claude Code",
        "codex": "Codex",
        "kiro": "Kiro",
        "openclaw": "OpenClaw",
        "github_copilot": "GitHub Copilot CLI",
    }
    choices = []
    for key, dirname in TOOL_CONFIGS.items():
        tool_dir = home / dirname
        found = tool_dir.is_dir()
        choices.append({
            "key": key,
            "name": display_names.get(key, key),
            "config_dir": tool_dir if found else None,
            "found": found,
        })
    return choices


def _run_tool_selection(tool_choices: list[dict]) -> list[dict]:
    """Interactive tool selection with checkboxes.

    Returns list of selected tool dicts (only found tools are selectable).
    """
    import questionary

    selectable = [t for t in tool_choices if t["found"]]
    if not selectable:
        return []

    # Found tools as checkboxes, not-found as separators
    not_found = [t for t in tool_choices if not t["found"]]
    qchoices = []
    for t in selectable:
        qchoices.append(questionary.Choice(
            title=f"{t['name']} (~/{TOOL_CONFIGS[t['key']]})",
            value=t["key"],
            checked=True,
        ))
    for t in not_found:
        qchoices.append(questionary.Separator(
            f"  {t['name']}  (~/{TOOL_CONFIGS[t['key']]} not found)"
        ))

    q = questionary.checkbox(
        "Install hooks/MCP:",
        choices=qchoices,
        pointer=">",
        instruction="(arrow keys, <space> to select/deselect)",
    )
    selected_keys = _patch_question(q).unsafe_ask()

    # questionary prints "done" when 0 selected; overwrite with "none"
    if not selected_keys:
        sys.stderr.write("\x1b[A\r\x1b[K? Install hooks/MCP: none\n")

    return [t for t in tool_choices if t["key"] in selected_keys]


def _run_provider_selection(
    label: str,
    choices: list[dict[str, Any]],
) -> Optional[tuple[str, dict]]:
    """Interactive single-select for a provider.

    Returns (provider_name, params) or None if nothing available.
    """
    import questionary

    available = [c for c in choices if c["available"]]
    if not available:
        return None

    # Build choices: available first, then unavailable
    default_value = None
    qchoices = []
    for c in choices:
        if not c["available"]:
            continue
        title = f"{c['name']} -- {c['hint']}"
        qc = questionary.Choice(title=title, value=c["value"])
        qchoices.append(qc)
        if c.get("default"):
            default_value = c["value"]

    q = questionary.select(
        label,
        choices=qchoices,
        default=default_value,
        pointer=">",
    )
    return _patch_question(q).unsafe_ask()


def run_wizard(
    config_dir: Path,
    store_path: Optional[Path] = None,
    restart_command: str = "keep",
) -> StoreConfig:
    """Run the interactive first-run setup wizard.

    Args:
        config_dir: Where keep.toml will be created
        store_path: Explicit store location (if provided via --store or env)
        restart_command: Command shown in cancel message (e.g. "keep" or "keep config --setup")

    Returns:
        StoreConfig ready to use
    """
    actual_store = store_path if store_path else config_dir

    print(file=sys.stderr)

    # Show store location
    display_path = actual_store
    home = Path.home()
    try:
        display_path = Path("~") / actual_store.relative_to(home)
    except ValueError:
        pass
    print(f"  keep config store -> {display_path}", file=sys.stderr)
    print(file=sys.stderr)

    # Non-interactive: fall back to silent config creation only.
    # Tool integrations are never installed implicitly.
    if not _is_interactive():
        print("  (non-interactive mode, using auto-detected defaults)", file=sys.stderr)
        from .config import create_default_config
        config = create_default_config(config_dir, store_path)
        save_config(config)
        return config

    # Load existing config if re-running setup
    existing = None
    from .config import CONFIG_FILENAME
    if (config_dir / CONFIG_FILENAME).exists():
        from .config import load_config
        try:
            existing = load_config(config_dir)
        except (FileNotFoundError, ValueError):
            pass

    try:
        return _run_interactive_setup(config_dir, store_path, actual_store, existing)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        print("  Setup cancelled.", file=sys.stderr)
        print(f"  Run `{restart_command}` again to restart setup.", file=sys.stderr)
        print(file=sys.stderr)
        raise SystemExit(130)


def _run_interactive_setup(
    config_dir: Path,
    store_path: Optional[Path],
    actual_store: Path,
    existing: Optional[StoreConfig] = None,
) -> StoreConfig:
    """Run the interactive prompts. Raises KeyboardInterrupt on Ctrl+C/Esc."""
    # --- Tool integrations ---
    tool_choices = detect_tool_choices()
    any_tools_found = any(t["found"] for t in tool_choices)

    selected_tools: list[dict] = []
    if any_tools_found:
        selected_tools = _run_tool_selection(tool_choices)
        print(file=sys.stderr)

    # --- Embedding provider ---
    current_embed = existing.embedding.name if existing and existing.embedding else None
    embed_choices = detect_embedding_choices(current=current_embed)
    any_embed_available = any(c["available"] for c in embed_choices)

    embedding_config: Optional[ProviderConfig] = None
    if any_embed_available:
        result = _run_provider_selection("Embeddings:", embed_choices)
        if result:
            embedding_config = ProviderConfig(name=result[0], params=result[1])
        print(file=sys.stderr)
    else:
        print("  No embedding provider available. You need one of:", file=sys.stderr)
        print("    - Install Ollama: https://ollama.com  (easiest)", file=sys.stderr)
        print("    - Set OPENAI_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY", file=sys.stderr)
        print("    - uv tool install 'keep-skill[local]'", file=sys.stderr)
        print(file=sys.stderr)
        print("  Set one up and run `keep` again.", file=sys.stderr)
        print(file=sys.stderr)
        raise SystemExit(1)

    # --- Summarization provider ---
    current_summ = existing.summarization.name if existing and existing.summarization else None
    summ_choices = detect_summarization_choices(current=current_summ)
    result = _run_provider_selection("Summarization:", summ_choices)
    summarization_config = ProviderConfig(name="truncate")
    if result:
        summarization_config = ProviderConfig(name=result[0], params=result[1])
    print(file=sys.stderr)

    # --- Build and save config ---
    store_path_str = None
    if store_path and store_path.resolve() != config_dir.resolve():
        store_path_str = str(store_path)

    # Detect additional providers that don't need user choice
    detected = detect_default_providers()

    config = StoreConfig(
        path=actual_store,
        config_dir=config_dir,
        store_path=store_path_str,
        embedding=embedding_config,
        summarization=summarization_config,
        document=ProviderConfig("composite"),
        media=detected.get("media"),
        content_extractor=detected.get("content_extractor"),
    )
    save_config(config)

    # Stop any running daemon so it picks up the new config on next start
    stop_daemon(config.path, force=True)

    # --- Install tool integrations ---
    installers = {
        "claude_code": install_claude_code,
        "codex": install_codex,
        "kiro": install_kiro,
        "openclaw": install_openclaw,
        "github_copilot": install_github_copilot,
    }

    installed_tools = []
    for tool in selected_tools:
        installer = installers.get(tool["key"])
        if installer and tool["config_dir"]:
            actions = installer(tool["config_dir"])
            if actions:
                installed_tools.append(tool["name"])
                config.integrations[tool["key"]] = HOOKS_VERSION

    # Mark unselected-but-found tools as skipped (-1 = explicit opt-out, won't re-prompt)
    for tool in tool_choices:
        if tool["found"] and tool["key"] not in config.integrations:
            config.integrations[tool["key"]] = -1

    if config.integrations:
        save_config(config)

    # --- Summary ---
    _print_summary(config, installed_tools)

    return config


def _print_summary(config: StoreConfig, installed_tools: list[str]) -> None:
    """Print the setup summary."""
    print("  ---", file=sys.stderr)
    print(file=sys.stderr)

    store_loc = config.config_dir if config.config_dir else config.path
    print(f"  Store initialized at {store_loc}", file=sys.stderr)

    if installed_tools:
        print(f"  Hooks installed for {', '.join(installed_tools)}", file=sys.stderr)
    else:
        print("  Hooks: none", file=sys.stderr)

    if config.embedding:
        embed_desc = config.embedding.name
        model = config.embedding.params.get("model", "")
        if model:
            embed_desc += f" ({model})"
        print(f"  Embeddings: {embed_desc}", file=sys.stderr)
    else:
        print("  Embeddings: none", file=sys.stderr)

    if config.summarization and config.summarization.name != "truncate":
        summ_desc = config.summarization.name
        model = config.summarization.params.get("model", "")
        if model:
            summ_desc += f" ({model})"
        print(f"  Summarization: {summ_desc}", file=sys.stderr)

    if config.media:
        media_desc = config.media.name
        model = config.media.params.get("model", "")
        if model:
            media_desc += f" ({model})"
        print(f"  Media: {media_desc}", file=sys.stderr)

    if config.content_extractor:
        ce_desc = config.content_extractor.name
        model = config.content_extractor.params.get("model", "")
        if model:
            ce_desc += f" ({model})"
        print(f"  Content extractor: {ce_desc}", file=sys.stderr)

    print(file=sys.stderr)
    print("  Run `keep now` to get started.", file=sys.stderr)
    print("  Run `keep config mcpb` to install in Claude Desktop.", file=sys.stderr)
    print(file=sys.stderr)


def needs_wizard(config_dir: Path) -> bool:
    """Check if the setup wizard should run (no existing config)."""
    from .config import CONFIG_FILENAME
    return not (config_dir / CONFIG_FILENAME).exists()
