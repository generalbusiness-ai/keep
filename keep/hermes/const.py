"""User-visible strings and tool schemas for the Hermes integration.

Centralised here so they can be reviewed, translated, or
overridden without touching provider logic.
"""

# -- Setup wizard ------------------------------------------------------------

SETUP_COMMAND = "hermes memory setup"

EMBEDDING_LABEL = "Embeddings"
EMBEDDING_EMPTY_MESSAGE = "No embedding provider available. You need one of:"
EMBEDDING_EMPTY_HINTS = [
    "Install Ollama (https://ollama.com) — free, local, easiest option",
    "Set OPENAI_API_KEY for OpenAI embeddings",
    "Set OPENROUTER_API_KEY for OpenRouter embeddings",
    "Set GEMINI_API_KEY for Google Gemini embeddings",
    f"Then run `{SETUP_COMMAND}` again.",
]

EMBEDDING_MISSING_ERROR = (
    "No Keep embedding provider is configured. Choose an available "
    f"embedding provider, then rerun `{SETUP_COMMAND}`."
)

KEEP_SKILL_MISSING_ERROR = (
    "keep-skill is not installed in Hermes's Python environment. "
    f"Run `uv pip install keep-skill` and rerun `{SETUP_COMMAND}`."
)

SUMMARIZATION_LABEL = "Summarization"

# -- System prompt blocks ----------------------------------------------------

SYSTEM_PROMPT_SETUP_REQUIRED = (
    "# Keep Memory\n"
    "Configured as Hermes's active external memory provider, but the "
    "Keep store for this Hermes profile is not set up yet.\n\n"
    "Tell the user to run: `{setup_command}`\n\n"
    "Until setup is completed, Keep memory context and Keep tools "
    "are unavailable."
)

PREFETCH_HEADER = ""

# -- Tool errors -------------------------------------------------------------

TOOL_ERROR_SETUP_REQUIRED = "Keep store is not configured for this Hermes profile"
TOOL_ERROR_SETUP_HINT = "Run `{setup_command}` and start a new session."
TOOL_ERROR_INACTIVE = "Keep is not active"

# -- Role labels for stored conversation content ----------------------------
# Shared across all keep integrations for consistency.

ROLE_USER = "User:"
ROLE_ASSISTANT = "Assistant:"

# -- Prompt names used by lifecycle hooks ------------------------------------

PROMPT_SYSTEM = "system-hermes"
PROMPT_HERMES_ASSEMBLE = "hermes-assemble"

# -- Tool schemas (hermes-specific format) -----------------------------------

FLOW_SCHEMA = {
    "name": "keep_flow",
    "description": (
        "Execute a keep operation via state-doc flow. "
        "Use state + params for normal calls. "
        "Only use state_doc_yaml for advanced custom flows.\n"
        "Examples:\n"
        '  Search: state="query-resolve", params={"query": "auth patterns"}\n'
        '  Get context: state="get", params={"item_id": "now"}\n'
        '  Store text: state="put", params={"content": "decision: use JWT", '
        '"tags": {"project": "auth"}}\n'
        '  Store URL: state="put", params={"uri": "https://example.com/article"}\n'
        '  List items: state="list", params={"prefix": ".tag/", '
        '"include_hidden": true}\n'
        '  Tag: state="tag", params={"id": "my-note", '
        '"tags": {"status": "done"}}\n'
        "When status is 'stopped', pass the returned cursor to continue."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": (
                    "State doc name: 'query-resolve', 'get', 'put', "
                    "'tag', 'delete', 'move', 'list', 'stats'."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Flow parameters as a JSON object. "
                    "Use this for normal calls; pass all operation arguments here. "
                    "Common keys: query, id, content, uri, tags, "
                    "summary, limit, since, scope, prefix, "
                    "include_hidden, deep, bias."
                ),
            },
            "token_budget": {
                "type": "integer",
                "description": "Token budget for rendered text output.",
            },
            "cursor": {
                "type": "string",
                "description": "Cursor from a previous 'stopped' result to resume.",
            },
            "state_doc_yaml": {
                "type": "string",
                "description": (
                    "Advanced only: inline YAML state doc override for custom "
                    "flows. Most operations should use state + params instead."
                ),
            },
        },
        "required": ["state"],
    },
}

HELP_SCHEMA = {
    "name": "keep_help",
    "description": (
        "Get help on keep operations. "
        "Call with no topic to get the full index of available guides. "
        "Topics include: 'agent-guide', 'flow-actions', "
        "'edge-tags', 'flows', 'keep-search', 'tagging'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Help topic name. Omit to get the index.",
            },
        },
        "required": [],
    },
}

PROMPT_SCHEMA = {
    "name": "keep_prompt",
    "description": (
        "Render an agent prompt with context injected from memory. "
        "Call with no name to list available prompts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Prompt name to render.",
            },
            "text": {
                "type": "string",
                "description": "Optional text or query for context.",
            },
            "id": {
                "type": "string",
                "description": 'Note ID for context (default: "now").',
            },
            "tags": {
                "type": "object",
                "description": "Optional tag filters for additional context.",
            },
            "since": {
                "type": "string",
                "description": "Only include items updated since this value.",
            },
            "until": {
                "type": "string",
                "description": "Only include items updated before this value.",
            },
            "deep": {
                "type": "boolean",
                "description": "Follow tags from results to discover related items.",
            },
            "scope": {
                "type": "string",
                "description": "ID glob to constrain search results.",
            },
            "token_budget": {
                "type": "integer",
                "description": "Token budget for rendered output.",
            },
        },
        "required": [],
    },
}
