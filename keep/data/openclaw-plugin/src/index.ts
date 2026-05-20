/**
 * keep — OpenClaw context engine plugin
 *
 * Requires plugins.slots.contextEngine: 'keep'.
 *
 * Design: each session is a keep item keyed by sessionKey. Turns become
 * versions of that item. The agent-managed `now` item is reserved for
 * explicit cross-session intentions — the plugin never writes to it.
 */

import { execFileSync } from "child_process";
import path from "node:path";
import fs from "node:fs";
import { delegateCompactionToRuntime } from "openclaw/plugin-sdk/core";
import { KeepMcpTransport } from "./mcp-transport.js";

// ---------------------------------------------------------------------------
// Startup check
// ---------------------------------------------------------------------------

function keepAvailable(): boolean {
  try {
    execFileSync("keep", ["config"], { timeout: 3000, stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Message content helpers
// ---------------------------------------------------------------------------

/** Extract text from an AgentMessage content field (string or content blocks). */
function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((b: any) => b.type === "text" && typeof b.text === "string")
      .map((b: any) => b.text)
      .join("\n");
  }
  if (content && typeof content === "object") {
    return JSON.stringify(content);
  }
  return "";
}

/** Truncate text to a char limit, appending ellipsis if truncated. */
function truncate(text: string, limit: number): string {
  if (text.length <= limit) return text;
  return text.slice(0, limit) + "…";
}

/** Rough token estimate: ~4 chars per token. */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// Roles worth ingesting into the session trace.
const INGEST_ROLES = new Set(["user", "assistant"]);

// ---------------------------------------------------------------------------
// Session identity helpers
// ---------------------------------------------------------------------------

/**
 * Stable session identifier for keep items. Prefers sessionKey (structured,
 * persistent across restarts) over sessionId (opaque UUID, per-record).
 */
function sessionItemId(params: { sessionKey?: string; sessionId: string }): string {
  return params.sessionKey || params.sessionId;
}

/**
 * Build tags for a session item. Always includes both identifiers when
 * available so items are queryable by either.
 */
function sessionTags(params: {
  sessionKey?: string;
  sessionId: string;
  extra?: Record<string, string>;
}): Record<string, string> {
  const tags: Record<string, string> = {};
  if (params.sessionKey) tags.session_key = params.sessionKey;
  tags.session_id = params.sessionId;
  if (params.extra) Object.assign(tags, params.extra);
  return tags;
}

/**
 * Format a batch of messages into a single turn content block.
 * User and assistant messages are labeled; others are skipped.
 */
function formatTurn(messages: any[], maxInlineLength: number): string {
  const parts: string[] = [];
  for (const msg of messages) {
    const role: string = msg.role || "unknown";
    if (!INGEST_ROLES.has(role)) continue;

    const text = extractText(msg.content);
    if (!text.trim()) continue;

    parts.push(`[${role}] ${truncate(text, maxInlineLength)}`);
  }
  return parts.join("\n\n");
}

// ---------------------------------------------------------------------------
// Global singleton: MCP transport survives SIGUSR1 module re-evaluation.
// Symbol.for() returns the same symbol across reloads; globalThis persists.
// ---------------------------------------------------------------------------

const KEEP_MCP_STATE = Symbol.for("keep.mcpTransportState");

interface KeepMcpState {
  mcp: KeepMcpTransport;
  eventsRegistered: boolean;
}

function getMcpState(api: any): KeepMcpState {
  const g = globalThis as any;
  if (g[KEEP_MCP_STATE]) return g[KEEP_MCP_STATE];
  const state: KeepMcpState = {
    mcp: new KeepMcpTransport({
      logger: {
        debug: (m: string) => api.logger?.debug?.(m),
        info: (m: string) => api.logger?.info(m),
        warn: (m: string) => api.logger?.warn(m),
        error: (m: string) => api.logger?.error(m),
      },
    }),
    eventsRegistered: false,
  };
  g[KEEP_MCP_STATE] = state;
  return state;
}

// ---------------------------------------------------------------------------
// Bootstrap context engine: injects setup instructions when keep is not
// fully available. The agent reads these and guides the user through setup.
// ---------------------------------------------------------------------------

const BOOTSTRAP_MESSAGES: Record<string, string> = {
  install: [
    "⚠️ keep plugin is installed but the Python runtime is not available.",
    "",
    "Guide the user through setup:",
    "1. Install Python 3.13 and uv (if not present): https://docs.astral.sh/uv/",
    "2. Install keep: uv pip install keep-skill[local]",
    "3. Restart: openclaw gateway restart",
    "",
    "After restart, if providers need configuration the agent will be prompted again.",
  ].join("\n"),
  configure: [
    "⚠️ keep is installed but no embedding/summarization providers are configured.",
    "",
    "Guide the user through provider setup:",
    "1. Run: openclaw keep setup",
    "   (interactive wizard — choose embedding and summarization providers)",
    "2. Restart: openclaw gateway restart",
  ].join("\n"),
};

function registerBootstrapContextEngine(api: any, mode: "install" | "configure") {
  const message = BOOTSTRAP_MESSAGES[mode];
  api.logger?.warn(`[keep] Registering bootstrap context engine (mode=${mode})`);

  api.registerContextEngine("keep", () => {
    return {
      info: {
        id: "keep",
        name: "keep (setup required)",
        version: "0.153.0",
        ownsCompaction: false,
      },
      async assemble(params: { messages: any[]; tokenBudget?: number }) {
        return {
          messages: params.messages,
          systemPromptAddition: message,
          estimatedTokens: 0,
        };
      },
    };
  });
}

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

export default function register(api: any) {
  if (!keepAvailable()) {
    api.logger?.warn("[keep] keep CLI not found — registering bootstrap context engine");
    registerBootstrapContextEngine(api, "install");
    return;
  }

  // Read keep's max_inline_length once at startup (the store's definition
  // of "inline-sized content"). Used for assistant message truncation.
  let maxInlineLength = 2000;
  try {
    const val = execFileSync("keep", ["config", "max_inline_length"], {
      encoding: "utf-8",
      timeout: 3000,
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
    const parsed = parseInt(val, 10);
    if (parsed > 0) maxInlineLength = parsed;
  } catch {
    // keep default is 2000; use that
  }

  // -----------------------------------------------------------------------
  // First-run health check
  // -----------------------------------------------------------------------

  let keepHealthy = true;
  try {
    const providers = execFileSync("keep", ["config", "providers"], {
      encoding: "utf-8",
      timeout: 5000,
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
    if (!providers || providers === "{}") {
      keepHealthy = false;
      api.logger?.warn(
        "[keep] No providers configured. Run `openclaw keep setup` " +
          "or `keep config --setup` to choose embedding and summarization providers.",
      );
    }
  } catch {
    keepHealthy = false;
    api.logger?.warn(
      "[keep] Provider check failed. Run `keep config --setup` to configure.",
    );
  }

  // -----------------------------------------------------------------------
  // CLI: `openclaw keep setup` / `openclaw keep doctor`
  // -----------------------------------------------------------------------

  api.registerCli((ctx: any) => {
    const keepCmd = ctx.program
      .command("keep")
      .description("Keep reflective memory management");

    keepCmd
      .command("setup")
      .description("Run keep's interactive setup wizard (providers, models)")
      .action(async () => {
        const { execFileSync: execSync } = await import("child_process");
        try {
          execSync("keep", ["config", "--setup"], {
            stdio: "inherit",
            timeout: 120_000,
          });
        } catch (err: any) {
          if (err.status) {
            process.exit(err.status);
          }
          throw err;
        }
      });

    keepCmd
      .command("doctor")
      .description("Check keep store health and diagnose issues")
      .action(async () => {
        const { execFileSync: execSync } = await import("child_process");
        try {
          execSync("keep", ["doctor"], {
            stdio: "inherit",
            timeout: 30_000,
          });
        } catch (err: any) {
          if (err.status) {
            process.exit(err.status);
          }
          throw err;
        }
      });
  }, { commands: ["keep"] });

  // If providers aren't configured, register bootstrap CE and stop.
  // The CLI is already registered above so `openclaw keep setup` works.
  if (!keepHealthy) {
    api.logger?.warn("[keep] Providers not configured — registering bootstrap context engine");
    registerBootstrapContextEngine(api, "configure");
    return;
  }

  // Shared MCP transport — global singleton survives SIGUSR1 reloads
  const mcpState = getMcpState(api);
  const mcp = mcpState.mcp;

  // Track first assemble per session (keyed by sessionKey for persistence).
  const sessionFirstAssemble = new Set<string>();

  // Track whether workspace watches have been set up.
  let watchesInitialized = false;

  // ---------------------------------------------------------------------------
  // Config helpers
  // ---------------------------------------------------------------------------

  function getConfig(): {
    contextBudgetRatio: number;
    captureHeartbeats: boolean;
    indexPaths: string[];
    indexExclude: string[];
  } {
    const cfg = api.pluginConfig ?? {};
    return {
      contextBudgetRatio: typeof cfg.contextBudgetRatio === "number"
        ? Math.max(0, Math.min(1, cfg.contextBudgetRatio))
        : 0.3,
      captureHeartbeats: cfg.captureHeartbeats === true,
      indexPaths: Array.isArray(cfg.indexPaths) ? cfg.indexPaths : ["./"],
      indexExclude: Array.isArray(cfg.indexExclude) ? cfg.indexExclude : [],
    };
  }

  // ---------------------------------------------------------------------------
  // Workspace watches
  // ---------------------------------------------------------------------------

  function ensureWatches(workspaceDir: string): void {
    if (watchesInitialized) return;
    watchesInitialized = true;

    const cfg = getConfig();

    for (const relPath of cfg.indexPaths) {
      const absPath = path.resolve(workspaceDir, relPath);
      if (!fs.existsSync(absPath)) continue;

      const stat = fs.statSync(absPath);

      const args: string[] = ["put", absPath];
      if (stat.isDirectory()) args.push("-r");
      args.push("--watch");
      for (const pattern of cfg.indexExclude) {
        args.push("--exclude", pattern);
      }

      try {
        execFileSync("keep", args, {
          encoding: "utf-8",
          timeout: 30_000,
          stdio: ["pipe", "pipe", "pipe"],
        });
        api.logger?.info(`[keep] Watch set up for ${relPath}`);
      } catch (err: any) {
        if (err.stderr?.includes("Already watching") || err.message?.includes("Already watching")) {
          api.logger?.debug(`[keep] Watch already active for ${relPath}`);
        } else {
          api.logger?.warn(`[keep] Failed to set up watch for ${relPath}: ${err.message}`);
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Gateway lifecycle: start/stop MCP transport
  // -----------------------------------------------------------------------

  if (!mcpState.eventsRegistered) {
    mcpState.eventsRegistered = true;
    api.on("gateway_start", async () => {
      try {
        await mcp.connect();
        api.logger?.info("[keep] MCP transport connected");
      } catch (err: any) {
        api.logger?.warn(
          `[keep] MCP connect failed, falling back to CLI: ${err.message}`,
        );
      }
    });

    api.on("gateway_stop", async () => {
      await mcp.disconnect();
    });
  }

  // -----------------------------------------------------------------------
  // memory_search / memory_get tools
  // -----------------------------------------------------------------------

  // Register memory tools that use keep's scoped find + line-range
  // enrichment. These provide the same interface as memory-core's tools
  // so OpenClaw's system prompt instructions ("run memory_search...") work.
  api.registerTool(
    (ctx: any) => {
      if (!ctx.workspaceDir) return null;

      const workspaceDir: string = ctx.workspaceDir;

      // Build the scope prefix from workspace memory paths.
      // SQLite LIKE is case-insensitive for ASCII, so "memory*" matches
      // both memory/ directory contents and MEMORY.md at workspace root.
      const memoryScope = `file://${path.resolve(workspaceDir, "memory")}*`;

      return [
        {
          name: "memory_search",
          label: "Memory Search",
          description:
            "Mandatory recall step: semantically search MEMORY.md + memory/*.md " +
            "before answering questions about prior work, decisions, dates, people, " +
            "preferences, or todos; returns top snippets with path + lines. " +
            "If response has disabled=true, memory retrieval is unavailable.",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string" },
              maxResults: { type: "number" },
              minScore: { type: "number" },
            },
            required: ["query"],
          },
          async execute(_toolCallId: string, params: any) {
            const query = typeof params?.query === "string" ? params.query.trim() : "";
            if (!query) {
              return {
                content: [{ type: "text", text: JSON.stringify({ results: [], error: "query required" }) }],
              };
            }

            const maxResults = typeof params?.maxResults === "number" ? params.maxResults : 10;
            const minScore = typeof params?.minScore === "number" ? params.minScore : 0;

            if (!mcp.connected) {
              try { await mcp.connect(); } catch {
                return {
                  content: [{
                    type: "text",
                    text: JSON.stringify({
                      results: [],
                      disabled: true,
                      unavailable: true,
                      error: "keep MCP not connected",
                      warning: "Memory search is unavailable because keep is not running.",
                      action: "Verify keep is installed (uv pip install keep-skill[local]) and restart: openclaw gateway restart",
                    }),
                  }],
                };
              }
            }

            try {
              // Use a simple find via an inline state doc — single query
              // with scope, no multi-step branching. query-resolve is too
              // slow and returns out-of-scope results for memory recall.
              // No token_budget — we need raw JSON, not rendered text.
              // (token_budget triggers text rendering in keep_flow MCP tool.)
              const result = await mcp.flow({
                state: "memory-search",
                params: {
                  query,
                  scope: memoryScope,
                  limit: maxResults,
                },
              });

              // State doc binding "results" contains { results: [...] }
              const searchResults = result.data?.results?.results
                || result.data?.results
                || [];

              // Map keep results to memory_search shape
              const mapped = (searchResults as any[])
                .filter((r: any) => {
                  const score = r.score ?? 0;
                  return score >= minScore;
                })
                .slice(0, maxResults)
                .map((r: any) => {
                  const id: string = r.id || "";
                  const tags = r.tags || {};

                  // Extract workspace-relative path
                  const filePrefix = `file://${workspaceDir}/`;
                  const relPath = id.startsWith(filePrefix)
                    ? id.slice(filePrefix.length)
                    : id.startsWith("file://")
                      ? id.slice(7)
                      : id;

                  // Use focus_summary (part match or keyword fallback) or summary
                  const snippet = tags._focus_summary || r.summary || "";

                  // Line range from part tags or keyword fallback
                  const startLine = parseInt(tags._focus_start_line, 10) || 1;
                  const endLine = parseInt(tags._focus_end_line, 10) || startLine;

                  const source = "memory";

                  return {
                    path: relPath,
                    startLine,
                    endLine,
                    score: r.score ?? 0,
                    snippet,
                    source,
                  };
                });

              return {
                content: [{
                  type: "text",
                  text: JSON.stringify({
                    results: mapped,
                    provider: "keep",
                    citations: "auto",
                  }, null, 2),
                }],
                details: { count: mapped.length },
              };
            } catch (err: any) {
              return {
                content: [{
                  type: "text",
                  text: JSON.stringify({
                    results: [],
                    disabled: true,
                    unavailable: true,
                    error: err.message,
                    warning: "Memory search failed.",
                    action: "Check keep store health with `keep doctor`.",
                  }),
                }],
              };
            }
          },
        },
        {
          name: "memory_get",
          label: "Memory Get",
          description:
            "Safe snippet read from MEMORY.md or memory/*.md with optional " +
            "from/lines; use after memory_search to pull only the needed " +
            "lines and keep context small.",
          parameters: {
            type: "object",
            properties: {
              path: { type: "string" },
              from: { type: "number" },
              lines: { type: "number" },
            },
            required: ["path"],
          },
          async execute(_toolCallId: string, params: any) {
            const relPath = typeof params?.path === "string" ? params.path.trim() : "";
            if (!relPath) {
              return {
                content: [{ type: "text", text: JSON.stringify({ path: "", text: "", disabled: true, error: "path required" }) }],
              };
            }

            // Resolve and validate path (constrain to memory/ and MEMORY.md)
            const absPath = path.isAbsolute(relPath)
              ? path.resolve(relPath)
              : path.resolve(workspaceDir, relPath);
            const resolved = path.relative(workspaceDir, absPath);

            const isMemoryPath =
              resolved === "MEMORY.md" ||
              resolved === "memory.md" ||
              resolved.startsWith("memory/") ||
              resolved.startsWith("memory" + path.sep);

            if (!isMemoryPath || resolved.startsWith("..") || path.isAbsolute(resolved)) {
              return {
                content: [{
                  type: "text",
                  text: JSON.stringify({ path: relPath, text: "", disabled: true, error: "path required" }),
                }],
              };
            }

            try {
              const content = fs.readFileSync(absPath, "utf-8");
              const allLines = content.split("\n");

              let text: string;
              if (typeof params?.from === "number" && params.from > 0) {
                const startIdx = params.from - 1; // 1-indexed to 0-indexed
                const count = typeof params?.lines === "number" && params.lines > 0
                  ? params.lines
                  : 50; // default 50 lines
                text = allLines.slice(startIdx, startIdx + count).join("\n");
              } else if (typeof params?.lines === "number" && params.lines > 0) {
                text = allLines.slice(0, params.lines).join("\n");
              } else {
                text = content;
              }

              return {
                content: [{ type: "text", text: JSON.stringify({ path: resolved, text }) }],
              };
            } catch (err: any) {
              return {
                content: [{
                  type: "text",
                  text: JSON.stringify({ path: relPath, text: "", disabled: true, error: err.message }),
                }],
              };
            }
          },
        },
      ];
    },
    { names: ["memory_search", "memory_get"] },
  );

  // -----------------------------------------------------------------------
  // Context Engine registration
  // -----------------------------------------------------------------------

  api.registerContextEngine("keep", () => {
    api.logger?.info("[keep] Context engine activated");

    return {
      info: {
        id: "keep",
        name: "keep reflective memory",
        version: "0.153.0",
        ownsCompaction: false,
      },

      // -------------------------------------------------------------------
      // bootstrap: initialize keep for this session
      // -------------------------------------------------------------------
      async bootstrap(params: {
        sessionId: string;
        sessionKey?: string;
        sessionFile: string;
      }) {
        if (!mcp.connected) {
          try { await mcp.connect(); } catch {
            return { bootstrapped: false, reason: "MCP not connected" };
          }
        }

        const itemId = sessionItemId(params);

        try {
          // Mark session for first-assemble enrichment
          sessionFirstAssemble.add(itemId);

          return { bootstrapped: true };
        } catch (err: any) {
          api.logger?.error(`[keep] bootstrap error: ${err.message}`);
          return { bootstrapped: false, reason: err.message };
        }
      },

      // -------------------------------------------------------------------
      // ingest: capture a single message as a version of the session item
      // -------------------------------------------------------------------
      async ingest(params: {
        sessionId: string;
        sessionKey?: string;
        message: any;
        isHeartbeat?: boolean;
      }) {
        const cfg = getConfig();

        if (params.isHeartbeat && !cfg.captureHeartbeats) {
          return { ingested: false };
        }

        if (!mcp.connected) {
          try { await mcp.connect(); } catch {
            return { ingested: false };
          }
        }

        const role: string = params.message.role || "unknown";

        if (!INGEST_ROLES.has(role)) {
          return { ingested: false };
        }

        try {
          const text = extractText(params.message.content);
          if (!text.trim()) {
            return { ingested: false };
          }

          const content = `[${role}] ${truncate(text, maxInlineLength)}`;
          const itemId = sessionItemId(params);

          await mcp.flow({
            state: "put",
            params: {
              content,
              id: itemId,
              tags: sessionTags({ ...params, extra: { role, type: "conversation", source: "openclaw" } }),
            },
          });

          return { ingested: true };
        } catch (err: any) {
          api.logger?.warn(`[keep] ingest error: ${err.message}`);
          return { ingested: false };
        }
      },

      // -------------------------------------------------------------------
      // ingestBatch: capture a completed turn as one version
      // -------------------------------------------------------------------
      async ingestBatch(params: {
        sessionId: string;
        sessionKey?: string;
        messages: any[];
        isHeartbeat?: boolean;
      }) {
        const cfg = getConfig();
        if (params.isHeartbeat && !cfg.captureHeartbeats) {
          return { ingestedCount: 0 };
        }
        if (!mcp.connected) {
          try { await mcp.connect(); } catch {
            return { ingestedCount: 0 };
          }
        }

        // Format all messages in the batch into a single turn block
        const turnContent = formatTurn(params.messages, maxInlineLength);
        if (!turnContent.trim()) {
          return { ingestedCount: 0 };
        }

        const itemId = sessionItemId(params);

        try {
          await mcp.flow({
            state: "put",
            params: {
              content: turnContent,
              id: itemId,
              tags: sessionTags({ ...params, extra: { type: "conversation", source: "openclaw" } }),
            },
          });

          // Count how many messages contributed to the turn
          const count = params.messages.filter(
            (m: any) => INGEST_ROLES.has(m.role || "") && extractText(m.content).trim(),
          ).length;

          return { ingestedCount: count };
        } catch (err: any) {
          api.logger?.warn(`[keep] ingestBatch error: ${err.message}`);
          return { ingestedCount: 0 };
        }
      },

      // -------------------------------------------------------------------
      // assemble: build model context with keep's memory
      //
      // Uses the openclaw-assemble prompt template, which references the
      // openclaw-assemble state doc for retrieval. The prompt pipeline
      // handles both fetching (5 parallel queries) and rendering (template
      // expansion with flow bindings). We just inject the result.
      // -------------------------------------------------------------------
      async assemble(params: {
        sessionId: string;
        sessionKey?: string;
        messages: any[];
        tokenBudget?: number;
        model?: string;
        prompt?: string;
      }) {
        const conversationTokens = params.messages.reduce((sum: number, m: any) => {
          return sum + estimateTokens(extractText(m.content));
        }, 0);

        if (!mcp.connected) {
          try { await mcp.connect(); } catch {
            return {
              messages: params.messages,
              estimatedTokens: conversationTokens,
            };
          }
        }

        try {
          const cfg = getConfig();

          const prompt = params.prompt
            ?? (() => {
              const lastUser = [...params.messages].reverse().find((m: any) => m.role === "user");
              return lastUser ? truncate(extractText(lastUser.content), maxInlineLength) : "";
            })();

          const totalBudget = params.tokenBudget || 8000;
          const keepBudget = Math.floor(totalBudget * cfg.contextBudgetRatio);

          const itemId = sessionItemId(params);
          const isFirstAssemble = sessionFirstAssemble.delete(itemId);

          // Call the prompt template — it runs the state doc flow internally,
          // expands bindings into the template, and returns rendered text.
          const contextText = await mcp.prompt({
            name: "openclaw-assemble",
            text: prompt || "session context",
            id: itemId,
            token_budget: keepBudget,
          });

          const keepTokens = estimateTokens(contextText);

          const parts: string[] = [];
          if (contextText.trim()) parts.push(`\`keep context\`:\n${contextText}`);

          if (isFirstAssemble) {
            parts.push(
              `\`keep tools\`: keep_flow, keep_help, and keep_prompt are available as tools. ` +
              `If unfamiliar with keep, start with keep_help(topic="flow-actions") ` +
              `and keep_help(topic="index") to learn the full capability set.`
            );
          }

          return {
            messages: params.messages,
            estimatedTokens: conversationTokens + keepTokens,
            systemPromptAddition: parts.length > 0 ? parts.join("\n\n") : undefined,
          };
        } catch (err: any) {
          api.logger?.warn(`[keep] assemble error: ${err.message}`);
          return {
            messages: params.messages,
            estimatedTokens: conversationTokens,
          };
        }
      },

      // -------------------------------------------------------------------
      // afterTurn: post-turn reflection and background work
      // -------------------------------------------------------------------
      async afterTurn(params: {
        sessionId: string;
        sessionKey?: string;
        sessionFile: string;
        messages: any[];
        prePromptMessageCount: number;
        autoCompactionSummary?: string;
        isHeartbeat?: boolean;
        tokenBudget?: number;
        runtimeContext?: Record<string, unknown>;
      }) {
        api.logger?.debug(`[keep] afterTurn (mcp=${mcp.connected}, sessionKey=${params.sessionKey}, newMsgs=${(params.messages?.length ?? 0) - (params.prePromptMessageCount ?? 0)})`);
        if (params.isHeartbeat) return;
        if (!mcp.connected) {
          try {
            api.logger?.info("[keep] afterTurn: MCP not connected, reconnecting...");
            await mcp.connect();
          } catch (err: any) {
            api.logger?.warn(`[keep] afterTurn: MCP reconnect failed: ${err.message}`);
            return;
          }
        }

        // Ensure workspace watches are set up (once per gateway lifetime).
        const workspaceDir = params.runtimeContext?.workspaceDir as string;
        if (workspaceDir && !watchesInitialized) {
          try { ensureWatches(workspaceDir); } catch {}
        }

        const newMessages = params.messages.slice(params.prePromptMessageCount);
        if (newMessages.length === 0) return;

        // Ingest each new message as a version of the session item
        const itemId = sessionItemId(params);
        for (const msg of newMessages) {
          const role: string = msg.role || "unknown";
          if (!INGEST_ROLES.has(role)) continue;
          const text = extractText(msg.content);
          if (!text.trim()) continue;
          const content = `[${role}] ${truncate(text, maxInlineLength)}`;
          try {
            await mcp.flow({
              state: "put",
              params: {
                content,
                id: itemId,
                tags: sessionTags({ ...params, extra: { role, type: "conversation", source: "openclaw" } }),
              },
            });
          } catch (err: any) {
            api.logger?.warn(`[keep] afterTurn ingest error: ${err.message}`);
          }
        }

        // Detect inflection signals in the new messages
        const signals = detectInflection(newMessages);

        if (signals.shouldReflect) {
          api.logger?.debug(
            `[keep] Inflection detected (${signals.reason}), triggering reflection`,
          );
          mcp.prompt({ name: "reflect" }).catch((err: any) => {
            api.logger?.warn(`[keep] afterTurn reflect error: ${err.message}`);
          });
        }

        // If OpenClaw auto-compacted, record as a version of the session item
        if (params.autoCompactionSummary) {
          const itemId = sessionItemId(params);
          mcp
            .flow({
              state: "put",
              params: {
                content: `[compaction] ${truncate(params.autoCompactionSummary, 2000)}`,
                id: itemId,
                tags: sessionTags({
                  ...params,
                  extra: { type: "compaction-summary" },
                }),
              },
            })
            .catch((err: any) => {
              api.logger?.warn(
                `[keep] afterTurn compaction index error: ${err.message}`,
              );
            });
        }
      },

      // -------------------------------------------------------------------
      // compact: delegate to runtime (ownsCompaction=false)
      // -------------------------------------------------------------------
      async compact(params: {
        sessionId: string;
        sessionKey?: string;
        sessionFile: string;
        tokenBudget?: number;
        force?: boolean;
        currentTokenCount?: number;
        compactionTarget?: "budget" | "threshold";
        customInstructions?: string;
        runtimeContext?: Record<string, unknown>;
      }) {
        api.logger?.debug(
          `[keep] compact: delegating to runtime (tokens=${params.currentTokenCount}, ` +
            `target=${params.compactionTarget || "budget"}, ` +
            `budget=${params.tokenBudget})`,
        );

        return delegateCompactionToRuntime(params);
      },

      // -------------------------------------------------------------------
      // prepareSubagentSpawn: link child to parent session
      // -------------------------------------------------------------------
      async prepareSubagentSpawn(params: {
        parentSessionKey: string;
        childSessionKey: string;
        ttlMs?: number;
      }) {
        if (!mcp.connected) {
          try { await mcp.connect(); } catch { return undefined; }
        }

        try {
          // Write spawn marker as first version of the child session item
          await mcp.flow({
            state: "put",
            params: {
              content: `Subagent spawned from ${params.parentSessionKey}`,
              id: params.childSessionKey,
              tags: {
                session_key: params.childSessionKey,
                parent_session: params.parentSessionKey,
                type: "subagent-spawn",
              },
            },
          });

          sessionFirstAssemble.add(params.childSessionKey);

          api.logger?.debug(
            `[keep] Prepared subagent ${params.childSessionKey} (parent: ${params.parentSessionKey})`,
          );

          return {
            rollback: async () => {
              try {
                await mcp.flow({
                  state: "delete",
                  params: { id: params.childSessionKey },
                });
              } catch {
                // best-effort cleanup
              }
              sessionFirstAssemble.delete(params.childSessionKey);
              api.logger?.debug(
                `[keep] Rolled back subagent prep for ${params.childSessionKey}`,
              );
            },
          };
        } catch (err: any) {
          api.logger?.warn(`[keep] prepareSubagentSpawn error: ${err.message}`);
          return undefined;
        }
      },

      // -------------------------------------------------------------------
      // onSubagentEnded: clean up tracking state
      // The child's session item persists — no move/archive needed.
      // -------------------------------------------------------------------
      async onSubagentEnded(params: {
        childSessionKey: string;
        reason: "deleted" | "completed" | "swept" | "released";
      }) {
        sessionFirstAssemble.delete(params.childSessionKey);
        api.logger?.debug(
          `[keep] Subagent ${params.childSessionKey} ended (${params.reason})`,
        );
      },

      async dispose() {
        sessionFirstAssemble.clear();
        watchesInitialized = false;
      },
    };
  });

  api.logger?.info("[keep] Registered: context engine");
}

// ---------------------------------------------------------------------------
// Inflection detection
// ---------------------------------------------------------------------------

type InflectionSignal = {
  shouldReflect: boolean;
  reason: string;
};

function detectInflection(messages: any[]): InflectionSignal {
  for (const msg of messages) {
    if (msg.role !== "user") continue;
    const text = extractText(msg.content).toLowerCase();

    if (
      /\b(let'?s move on|that'?s done|moving on|next topic|switching to|wrapping up|let'?s wrap)\b/.test(
        text,
      )
    ) {
      return { shouldReflect: true, reason: "explicit-transition" };
    }

    if (
      /\b(i'?ll do|let'?s do|we should|go ahead|ship it|merge it|deploy|let'?s build)\b/.test(
        text,
      )
    ) {
      return { shouldReflect: true, reason: "commitment" };
    }
  }

  for (const msg of messages) {
    if (msg.role !== "assistant") continue;
    const text = extractText(msg.content);
    if (text.length > 2000) {
      return { shouldReflect: true, reason: "substantial-response" };
    }
  }

  return { shouldReflect: false, reason: "" };
}


