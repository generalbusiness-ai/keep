/**
 * MCP transport layer for keep.
 *
 * Manages a persistent `keep mcp` stdio process. All keep operations
 * go through MCP tool calls (keep_flow, keep_prompt, keep_help).
 *
 * Lifecycle:
 *   connect()    → spawn process, initialize client
 *   callTool()   → invoke an MCP tool (with timeout)
 *   disconnect() → close transport, terminate process
 *
 * Auto-reconnects on transport error (process crash → respawn).
 * All calls have timeouts to prevent hung agent turns.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// ---------------------------------------------------------------------------
// Timeouts (milliseconds)
// ---------------------------------------------------------------------------

/** Max time to spawn and connect to keep mcp. */
const CONNECT_TIMEOUT_MS = 15_000;

/** Default per-call timeout for MCP tool invocations. */
const DEFAULT_CALL_TIMEOUT_MS = 10_000;

/** Timeout for calls that may trigger background work (put, move). */
const WRITE_CALL_TIMEOUT_MS = 15_000;

/** Timeout for assemble — on the hot path, must be fast. */
const ASSEMBLE_TIMEOUT_MS = 8_000;

/** Timeout for longer operations (reflect prompts, compact). */
const LONG_CALL_TIMEOUT_MS = 30_000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type KeepLogger = {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

export type KeepFlowParams = {
  state?: string;
  params?: Record<string, unknown>;
  budget?: number;
  token_budget?: number;
  cursor?: string;
  state_doc_yaml?: string;
};

export type KeepFlowResult = {
  status: "done" | "error" | "stopped";
  ticks: number;
  data?: Record<string, unknown>;
  cursor?: string;
  text?: string;
};

export type KeepPromptParams = {
  name?: string;
  text?: string;
  id?: string;
  tags?: Record<string, string>;
  since?: string;
  scope?: string;
  token_budget?: number;
};

export type KeepMcpLaunch = {
  command: string;
  args: string[];
  env?: Record<string, string>;
};

const FORWARDED_ENV_KEYS = new Set([
  "ANTHROPIC_API_KEY",
  "CLAUDE_CODE_OAUTH_TOKEN",
  "GEMINI_API_KEY",
  "GOOGLE_API_KEY",
  "GOOGLE_CLOUD_LOCATION",
  "GOOGLE_CLOUD_PROJECT",
  "HERMES_HOME",
  "KEEP_CONFIG",
  "KEEP_EMBED_PROBE",
  "KEEP_LOCAL_ONLY",
  "KEEP_STORE_PATH",
  "KEEP_TRACE",
  "KEEP_VERBOSE",
  "KEEPNOTES_API_KEY",
  "KEEPNOTES_API_URL",
  "KEEPNOTES_PROJECT",
  "MISTRAL_API_KEY",
  "OLLAMA_HOST",
  "OPENAI_API_KEY",
  "OPENROUTER_API_KEY",
  "VOYAGE_API_KEY",
]);

// ---------------------------------------------------------------------------
// Timeout helper
// ---------------------------------------------------------------------------

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`[keep-mcp] ${label} timed out after ${ms}ms`));
    }, ms);

    promise.then(
      (val) => { clearTimeout(timer); resolve(val); },
      (err) => { clearTimeout(timer); reject(err); },
    );
  });
}

export function buildKeepMcpLaunch(
  keepCommand: string,
  baseEnv: NodeJS.ProcessEnv = process.env,
): KeepMcpLaunch {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(baseEnv)) {
    if (typeof value !== "string" || value.trim() === "") continue;
    if (key.startsWith("KEEP_") || FORWARDED_ENV_KEYS.has(key)) {
      env[key] = value;
    }
  }

  const args = ["mcp"];
  const storePath = (baseEnv.KEEP_STORE_PATH || "").trim();
  if (storePath) {
    // Pass the store explicitly because many MCP hosts only inherit a
    // minimal safe environment into stdio children.
    args.unshift(storePath);
    args.unshift("--store");
  }

  return { command: keepCommand, args, env };
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

export class KeepMcpTransport {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private connecting: Promise<void> | null = null;
  private childPid: number | null = null;
  private logger: KeepLogger;
  private keepCommand: string;

  constructor(opts?: { logger?: KeepLogger; keepCommand?: string }) {
    this.logger = opts?.logger ?? {
      info: (m) => console.log(`[keep-mcp] ${m}`),
      warn: (m) => console.warn(`[keep-mcp] ${m}`),
      error: (m) => console.error(`[keep-mcp] ${m}`),
    };
    this.keepCommand = opts?.keepCommand ?? "keep";
  }

  /** Spawn `keep mcp` and connect the MCP client. */
  async connect(): Promise<void> {
    if (this.client) return;
    if (this.connecting) return this.connecting;

    this.connecting = withTimeout(
      this._connect(),
      CONNECT_TIMEOUT_MS,
      "connect",
    );
    try {
      await this.connecting;
    } finally {
      this.connecting = null;
    }
  }

  private async _connect(): Promise<void> {
    // Clean up orphaned process from a previous dead connection.
    // Only disconnect if the client is already gone (connection dead).
    // If client is live, connect() short-circuits before reaching here.
    if (!this.client && (this.transport || this.childPid)) {
      await this.disconnect();
    }

    const launch = buildKeepMcpLaunch(this.keepCommand);
    this.logger.info(
      launch.args[0] === "--store"
        ? `Spawning keep mcp process (store=${launch.args[1]})`
        : "Spawning keep mcp process",
    );

    this.transport = new StdioClientTransport(launch);

    this.client = new Client(
      { name: "keep-openclaw-plugin", version: "0.145.0" },
      { capabilities: {} },
    );

    // Handle transport close — mark as disconnected for reconnect
    const thisTransport = this.transport;
    this.transport.onclose = () => {
      this.logger.warn("keep mcp process exited");
      // Only clear references if this is still the active transport.
      // A new _connect() may have already replaced it.
      if (this.transport === thisTransport) {
        this.client = null;
        this.transport = null;
      }
    };

    this.transport.onerror = (err) => {
      this.logger.error(`keep mcp transport error: ${err}`);
    };

    await this.client.connect(this.transport);
    this.childPid = this.transport.pid ?? null;
    this.logger.info(`Connected to keep mcp (pid=${this.childPid})`);
  }

  /** Ensure connected, reconnecting if needed. */
  private async ensureConnected(): Promise<Client> {
    if (!this.client) {
      await this.connect();
    }
    if (!this.client) {
      throw new Error("Failed to connect to keep mcp");
    }
    return this.client;
  }

  /**
   * Call a keep MCP tool with timeout.
   * Auto-reconnects once on transport failure.
   */
  async callTool(
    name: string,
    args: Record<string, unknown>,
    timeoutMs: number = DEFAULT_CALL_TIMEOUT_MS,
  ): Promise<unknown> {
    const doCall = async (client: Client): Promise<unknown> => {
      const result = await client.callTool(
        { name, arguments: args },
        undefined,
        { timeout: timeoutMs },
      );
      return this.extractResult(result);
    };

    let client: Client;
    try {
      client = await this.ensureConnected();
    } catch (err) {
      this.logger.error(`Failed to connect: ${err}`);
      throw err;
    }

    try {
      return await doCall(client);
    } catch (err: any) {
      // If transport died, try one reconnect
      if (!this.client) {
        this.logger.warn("Reconnecting after transport failure");
        client = await this.ensureConnected();
        return await doCall(client);
      }
      throw err;
    }
  }

  /** Extract text/JSON from MCP tool result content blocks. */
  private extractResult(result: any): unknown {
    if (result.content && Array.isArray(result.content)) {
      const textParts = result.content
        .filter((c: any) => c.type === "text")
        .map((c: any) => c.text);
      if (textParts.length === 1) {
        try {
          return JSON.parse(textParts[0]);
        } catch {
          return textParts[0];
        }
      }
      if (textParts.length > 1) {
        return textParts.join("\n");
      }
    }
    return result;
  }

  /** Run a keep flow via MCP. Uses appropriate timeout per state doc. */
  async flow(params: KeepFlowParams): Promise<KeepFlowResult> {
    const args: Record<string, unknown> = {};
    if (params.state) args.state = params.state;
    if (params.params) args.params = params.params;
    if (params.budget) args.budget = params.budget;
    if (params.token_budget) args.token_budget = params.token_budget;
    if (params.cursor) args.cursor = params.cursor;
    if (params.state_doc_yaml) args.state_doc_yaml = params.state_doc_yaml;

    // Pick timeout based on operation type
    const timeout = this.timeoutForState(params.state);

    return (await this.callTool("keep_flow", args, timeout)) as KeepFlowResult;
  }

  /** Render a keep prompt via MCP. */
  async prompt(params: KeepPromptParams): Promise<string> {
    const args: Record<string, unknown> = {};
    if (params.name) args.name = params.name;
    if (params.text) args.text = params.text;
    if (params.id) args.id = params.id;
    if (params.tags) args.tags = params.tags;
    if (params.since) args.since = params.since;
    if (params.scope) args.scope = params.scope;
    if (params.token_budget != null) args.token_budget = params.token_budget;

    // Assemble prompts are on the hot path — use assemble timeout
    const timeout = params.name === "openclaw-assemble"
      ? ASSEMBLE_TIMEOUT_MS
      : LONG_CALL_TIMEOUT_MS;

    const result = await this.callTool("keep_prompt", args, timeout);
    return typeof result === "string" ? result : JSON.stringify(result);
  }

  /** Gracefully disconnect. */
  async disconnect(): Promise<void> {
    const pid = this.childPid;
    if (this.transport) {
      this.logger.info("Disconnecting keep mcp");
      try {
        await this.transport.close();
      } catch {
        // ignore close errors
      }
    }
    // Kill orphaned child process if transport.close() didn't reach it
    if (pid) {
      try { process.kill(pid, 0); process.kill(pid, "SIGTERM"); } catch {
        // process already exited
      }
    }
    this.client = null;
    this.transport = null;
    this.childPid = null;
  }

  /** Check if currently connected. */
  get connected(): boolean {
    return this.client !== null;
  }

  /** Select timeout based on the state doc being invoked. */
  private timeoutForState(state?: string): number {
    if (!state) return DEFAULT_CALL_TIMEOUT_MS;

    switch (state) {
      // Hot path — agent is waiting
      case "get":
      case "find-deep":
      case "stats":
        return ASSEMBLE_TIMEOUT_MS;

      // Memory search — single find but may need embedding cold start
      case "memory-search":
        return WRITE_CALL_TIMEOUT_MS;

      // Write operations — may trigger background work
      case "put":
      case "tag":
      case "move":
      case "delete":
        return WRITE_CALL_TIMEOUT_MS;

      // Multi-step query resolution
      case "query-resolve":
      case "query-branch":
      case "query-explore":
        return LONG_CALL_TIMEOUT_MS;

      // Compact / reflect
      case "openclaw-compact":
        return LONG_CALL_TIMEOUT_MS;

      default:
        return DEFAULT_CALL_TIMEOUT_MS;
    }
  }
}
