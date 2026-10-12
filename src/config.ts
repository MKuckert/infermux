import { existsSync, readFileSync } from "node:fs";
import { resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";
import * as yaml from "js-yaml";
import { Ajv2020 } from "ajv/dist/2020.js";
import type { ErrorObject, ValidateFunction } from "ajv/dist/2020.js";
import type { VramBackendName } from "./vram.js";

/**
 * Configuration model for infermux.
 *
 * Supported file formats: .json, .jsonc (naively stripped of // and /* *\/ comments),
 * .yaml / .yml.
 *
 * Structural validation is delegated to config.schema.json (draft 2020-12),
 * compiled with ajv; this module adds only what a schema cannot express:
 * built-in defaults and `${ENV_VAR}` expansion across the whole config
 * (applied before validation, so casts can supply schema-expected types).
 */

export interface ServerConfig {
  /** Host/interface to bind. Default: 0.0.0.0 */
  host: string;
  /** Port to listen on. Default: 8080 */
  port: number;
  /**
   * Optional Bearer token. When set (non-empty), every /v1/* request must
   * carry `Authorization: Bearer <token>`, otherwise 401. An empty string
   * (or absence) disables auth.
   */
  apiToken?: string;
}

export interface DefaultsConfig {
  /** Per-request deadline, in ms (covers queue wait + engine switch + upstream call). Default: 120000 */
  requestTimeoutMs: number;
  /** Interval between health poll attempts, in ms. Default: 500 */
  healthCheckIntervalMs: number;
  /** Total time to keep polling health after starting an engine, in ms. Default: 30000 */
  healthCheckTimeoutMs: number;
  /** Default GPU cool-down buffer after stopping an engine, in ms. Default: 3000 */
  coolDownMs: number;
  /**
   * VRAM must drop below this fraction of total (0..1) before a new engine is
   * started. Default: 0.15. Set to 0 to disable the VRAM gate entirely.
   */
  vramThresholdPct: number;
  /** How long to keep polling VRAM usage before giving up, in ms. Default: 30000 */
  vramCheckTimeoutMs: number;
  /** Poll interval for VRAM usage, in ms. Default: 500 */
  vramCheckIntervalMs: number;
  /** Grace period between SIGTERM and SIGKILL for child processes, in ms. Default: 5000 */
  childKillGraceMs: number;
  /** Max time to wait for a shell stop command (`commands.stop`) to finish, in ms. Default: 10000 */
  stopCommandTimeoutMs: number;
  /**
   * VRAM source: `"auto"` (default) probes nvidia-smi/rocm-smi/amd-smi, and on
   * macOS uses unified system memory; `"none"` is the no-op checker (VRAM
   * gating is skipped); otherwise forces a specific backend.
   */
  vramBackend: VramBackendName;
}

/**
 * Optional metric-scraping mappings for non-standard provider endpoints.
 * Values are JSON-pointer-ish paths (dot separated) into the JSON response of a
 * provider endpoint, e.g. `{ "vramUsed": "gpu.memory_used_mib" }`.
 */
export interface MetricsMapping {
  /** Name of the endpoint path (relative to targetBaseUrl) to GET for stats. e.g. "/metrics" */
  endpoint?: string;
  /** Dot-separated path into the JSON response for VRAM usage (MiB). */
  vramUsedMiB?: string;
  /** Dot-separated path into the JSON response for VRAM total (MiB). */
  vramTotalMiB?: string;
}

/**
 * Lifecycle mode B: separate start/stop shell commands. The proxy fires
 * `start` without holding a process handle (it may launch another server
 * process) and only tracks whether the engine was started; `stop` is run to
 * stop it.
 */
export interface EngineCommands {
  /** Shell command that starts the engine (run via /bin/sh, fire-and-forget). */
  start: string;
  /** Shell command that stops the engine (run via /bin/sh). */
  stop: string;
}

export interface ModelConfig {
  /** Provider name, e.g. "ollama" | "vllm" | "lmstudio". Requests to the same provider share one engine instance. */
  provider: string;
  /** Base URL of the provider's OpenAI-compatible API. */
  targetBaseUrl: string;
  /**
   * Model name/id as expected by the provider. If omitted, the request's `model`
   * field is passed through unchanged.
   */
  targetModel?: string;
  /**
   * Lifecycle mode A (mutually exclusive with `commands`): a single shell
   * command that starts the engine. The proxy spawns it, holds a process
   * handle on it, and stops it by killing the whole process group (SIGTERM,
   * then SIGKILL after `childKillGraceMs`). If the child dies on its own, the
   * next request restarts it.
   */
  command?: string;
  /** Lifecycle mode B (mutually exclusive with `command`): separate start/stop shell commands. */
  commands?: EngineCommands;
  /** Cooldown buffer after stopping, before starting the new engine. Overrides defaults. */
  coolDownMs?: number;
  /** Health/readiness endpoint path (relative to targetBaseUrl) that must return 200. */
  healthEndpoint?: string;
  /** Override VRAM threshold for this model (0..1). */
  vramThresholdPct?: number;
  /**
   * `false` = the engine runs on *another* host (or is fully externally managed):
   * the proxy never runs its start/stop commands locally, spawns no child,
   * skips the local VRAM gate, and only polls the health endpoint before
   * forwarding. Default: `true` (engine lives on the proxy host).
   */
  managed?: boolean;
  /**
   * Optional provider API key forwarded to the target endpoint as
   * `Authorization: Bearer <key>`. Any string in the config supports
   * `${ENV_VAR}` expansion, so secrets need not live in the config file
   * (unset vars are a load-time error).
   */
  targetApiKey?: string;
  /**
   * Optional static headers added to every upstream request, for providers
   * with non-OpenAI auth (e.g. `{ "x-api-key": "${LITELLM_KEY}" }`).
   * Values support `${ENV_VAR}` expansion; a `targetApiKey` (if set) takes
   * precedence over an `authorization` entry here.
   */
  targetHeaders?: Record<string, string>;
  /** Optional non-standard metrics mapping for the stats endpoint. */
  metrics?: MetricsMapping;
}

export interface InfermuxConfig {
  server: ServerConfig;
  defaults: Required<DefaultsConfig>;
  /** Model alias -> configuration. The `model` field of a request is matched against these keys. */
  models: Record<string, ModelConfig>;
}

export const DEFAULTS: Required<DefaultsConfig> = {
  requestTimeoutMs: 120_000,
  healthCheckIntervalMs: 500,
  healthCheckTimeoutMs: 30_000,
  coolDownMs: 3_000,
  vramThresholdPct: 0.15,
  vramCheckTimeoutMs: 30_000,
  vramCheckIntervalMs: 500,
  childKillGraceMs: 5_000,
  stopCommandTimeoutMs: 10_000,
  vramBackend: "auto",
};

export const SERVER_DEFAULTS: Pick<ServerConfig, "host" | "port"> = {
  host: "0.0.0.0",
  port: 8080,
};

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}

// The schema lives at the repo root; both src/config.ts and dist/config.js
// resolve ../config.schema.json to the same file.
const SCHEMA_PATH = fileURLToPath(new URL("../config.schema.json", import.meta.url));

function loadValidator(): ValidateFunction {
  let text: string;
  try {
    text = readFileSync(SCHEMA_PATH, "utf8");
  } catch (err) {
    throw new ConfigError(
      `config schema not readable at ${SCHEMA_PATH}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  let schema: object;
  try {
    schema = JSON.parse(text) as object;
  } catch (err) {
    throw new ConfigError(
      `config schema is not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  return new Ajv2020({ allErrors: true, strict: false }).compile(schema);
}

const validateSchema = loadValidator();

/** Matches `${VAR}` and `${VAR:cast}` references. */
const ENV_REF = /\$\{([A-Za-z_][A-Za-z0-9_]*)(?::[A-Za-z]+)?\}/g;

/**
 * Look up an env var, failing loudly at load time: a literal `${MISSING}`
 * silently sent to the provider would be an opaque auth failure at request
 * time.
 */
function lookupEnv(name: string, whole: string, ctx: string): string {
  const v = process.env[name];
  if (v === undefined) {
    throw new ConfigError(`${ctx}: environment variable ${name} is not set (referenced as ${whole})`);
  }
  return v;
}

/**
 * Plain search-replace of every `${VAR}` reference in a string. A `:cast`
 * specifier is only meaningful when the whole value is one reference (see
 * expandEnvValue); anywhere else it is a user error.
 */
function expandEnvText(value: string, ctx: string): string {
  // A `${` with no closing `}` cannot be a reference — fail loudly instead of
  // shipping the literal to the provider/shell. (Closed-but-malformed forms
  // like `${var:-default}` are left alone: they are legitimate shell syntax.
  // Likewise, ${...} sequences inside env var *values* are not re-expanded.)
  const unclosed = value.match(/\$\{[^}]*$/);
  if (unclosed) {
    throw new ConfigError(
      `${ctx}: unclosed environment variable reference ${unclosed[0]} (missing closing "}")`,
    );
  }
  return value.replace(ENV_REF, (whole, name: string) => {
    if (whole.includes(":")) {
      throw new ConfigError(
        `${ctx}: type cast in ${whole} is only allowed when the whole value is a single variable reference`,
      );
    }
    return lookupEnv(name, whole, ctx);
  });
}

/**
 * Expand a string value. If the *entire* value is a single typed reference
 * (e.g. `${OMLX_PORT:int}`), the result is cast to a number so it satisfies
 * numeric schema fields; otherwise every reference is substituted as text,
 * which allows variables nested inside larger strings
 * (e.g. `"http://localhost:${PORT}"`).
 */
function expandEnvValue(value: string, ctx: string): unknown {
  const whole = value.match(/^\$\{([A-Za-z_][A-Za-z0-9_]*):([A-Za-z]+)\}$/);
  if (whole) {
    const [, name, cast] = whole;
    const raw = lookupEnv(name!, whole[0], ctx);
    if (raw === "") {
      throw new ConfigError(`${ctx}: environment variable ${name} is empty; cannot cast to ${cast}`);
    }
    const n = Number(raw);
    if (cast === "int") {
      if (!Number.isInteger(n)) {
        throw new ConfigError(`${ctx}: environment variable ${name} is "${raw}", not an integer (referenced as ${whole[0]})`);
      }
      return n;
    }
    if (cast === "float") {
      if (!Number.isFinite(n)) {
        throw new ConfigError(`${ctx}: environment variable ${name} is "${raw}", not a number (referenced as ${whole[0]})`);
      }
      return n;
    }
    throw new ConfigError(`${ctx}: unknown type cast "${cast}" in ${whole[0]} (expected int or float)`);
  }
  return expandEnvText(value, ctx);
}

/**
 * Recursively expand `${ENV_VAR}` references in every string of the parsed
 * config — object values and keys alike. Runs before schema validation so
 * cast references can produce the types the schema expects.
 */
function expandConfigEnv(value: unknown, ctx: string): unknown {
  if (typeof value === "string") return expandEnvValue(value, ctx);
  if (Array.isArray(value)) {
    return value.map((v, i) => expandConfigEnv(v, `${ctx}[${i}]`));
  }
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    // Tracks which original key produced each expanded key, so an expansion
    // that collides with a sibling key is an error instead of a silent
    // overwrite (e.g. "x" and "${A}" with A=x).
    const sources = new Map<string, string>();
    for (const [k, v] of Object.entries(value)) {
      // Keys stay strings: plain expansion only (a cast in a key is an error).
      const key = expandEnvText(k, `${ctx} key "${k}"`);
      const prev = sources.get(key);
      if (prev !== undefined) {
        throw new ConfigError(
          `${ctx}: key "${k}" expands to "${key}", which collides with key "${prev}"`,
        );
      }
      sources.set(key, k);
      out[key] = expandConfigEnv(v, `${ctx}.${k}`);
    }
    return out;
  }
  return value;
}

/** Strip // and /* *\/ comments from a JSON document (string/comment-aware). */
export function stripJsonComments(text: string): string {
  let out = "";
  let inString = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    if (inString) {
      out += c;
      if (c === "\\") {
        out += text[++i] ?? "";
      } else if (c === '"') {
        inString = false;
      }
      continue;
    }
    if (c === '"') {
      inString = true;
      out += c;
      continue;
    }
    if (c === "/" && text[i + 1] === "/") {
      while (i < text.length && text[i] !== "\n") i++;
      out += text[i] ?? "";
      continue;
    }
    if (c === "/" && text[i + 1] === "*") {
      i += 2;
      while (i < text.length && !(text[i] === "*" && text[i + 1] === "/")) i++;
      i++; // skip closing /
      continue;
    }
    out += c;
  }
  return out;
}

function parseConfigText(raw: string, file: string): unknown {
  const ext = extname(file).toLowerCase();
  try {
    if (ext === ".yaml" || ext === ".yml") {
      return yaml.load(raw);
    }
    return JSON.parse(stripJsonComments(raw));
  } catch (err) {
    throw new ConfigError(
      `Failed to parse config file ${file}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/**
 * Validate a parsed config against config.schema.json. On failure, throws a
 * ConfigError listing every violation with its JSON path.
 */
function assertSchema(raw: unknown, fileName: string): void {
  if (validateSchema(raw)) return;
  const problems = (validateSchema.errors ?? []).map((e: ErrorObject) => {
    const path = e.instancePath || "(root)";
    // ajv's default messages are generic for these keywords; spell them out.
    const extra =
      e.keyword === "enum"
        ? ` (expected one of: ${(e.params as { allowedValues?: unknown[] }).allowedValues?.join(", ")})`
        : e.keyword === "not" && path.startsWith("/models/")
          ? " (command and commands are mutually exclusive — use one or the other)"
          : "";
    return `${path}: ${e.message ?? "invalid value"}${extra}`;
  });
  throw new ConfigError(`${fileName}: invalid config — ${problems.join("; ")}`);
}

/**
 * The lifecycle mode a model declares (mirrors EngineManager's engineKind):
 * how the shared engine of its provider is driven locally. `managed: false`
 * always means "no local lifecycle" (any declared commands are ignored).
 */
function modelLifecycleKind(m: ModelConfig): "command" | "commands" | "none" {
  if (m.managed === false) return "none";
  if (typeof m.command === "string") return "command";
  if (m.commands) return "commands";
  return "none";
}

/**
 * Schema-validated, env-expanded raw input -> InfermuxConfig: fills built-in
 * defaults (the only remaining transformation a schema cannot do).
 */
function finalize(raw: unknown): InfermuxConfig {
  const r = raw as {
    server?: { host?: string; port?: number; apiToken?: string };
    defaults?: Record<string, unknown>;
    models: Record<string, Record<string, unknown>>;
  };

  const server: ServerConfig = {
    host: r.server?.host ?? SERVER_DEFAULTS.host,
    port: r.server?.port ?? SERVER_DEFAULTS.port,
  };
  if (typeof r.server?.apiToken === "string") server.apiToken = r.server.apiToken;

  const defaults = {
    ...DEFAULTS,
    // The schema has already checked types/ranges for every one of these keys.
    ...(r.defaults ?? {}),
  } as Required<DefaultsConfig>;

  const models: Record<string, ModelConfig> = {};
  for (const [alias, entry] of Object.entries(r.models ?? {})) {
    // The schema guarantees the exact shape; the spread widens it to unknown.
    models[alias] = { ...entry } as unknown as ModelConfig;
  }

  // Models sharing a provider share one engine instance, so they must agree on
  // how it is driven: a mixed setup (e.g. one alias with `command`, another
  // with `commands` or none) would leave the engine un-restartable through
  // some aliases once it dies. Enforce one lifecycle mode per provider.
  const kindByProvider = new Map<string, { kind: string; alias: string }>();
  for (const [alias, m] of Object.entries(models)) {
    const kind = modelLifecycleKind(m);
    const prev = kindByProvider.get(m.provider);
    if (prev && prev.kind !== kind) {
      throw new ConfigError(
        `models "${prev.alias}" and "${alias}" both use provider "${m.provider}" but declare different lifecycles (${prev.kind} vs ${kind}); models sharing a provider share one engine and must use the same lifecycle mode (command, commands, or neither)`,
      );
    }
    if (!prev) kindByProvider.set(m.provider, { kind, alias });
  }

  return { server, defaults, models };
}

/**
 * Validate a parsed config object (see config.schema.json) and return the
 * fully-defaulted, typed InfermuxConfig.
 *
 * `${ENV_VAR}` references are expanded (with optional `:int`/`:float` casts)
 * across the whole object *before* schema validation, so a cast can supply
 * the numeric type a field expects.
 */
export function validateConfig(
  raw: unknown,
  fileName = "<memory>",
): InfermuxConfig {
  const expanded = expandConfigEnv(raw, fileName);
  assertSchema(expanded, fileName);
  return finalize(expanded);
}

/**
 * Load and validate a config file. Path resolution order is left to the caller;
 * pass an absolute (or cwd-relative) path.
 */
export function loadConfigFile(file: string): InfermuxConfig {
  const abs = resolve(file);
  if (!existsSync(abs)) {
    throw new ConfigError(`Config file not found: ${abs}`);
  }
  return validateConfig(parseConfigText(readFileSync(abs, "utf8"), abs), abs);
}
