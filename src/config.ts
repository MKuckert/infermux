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
 * built-in defaults and `${ENV_VAR}` expansion for secrets.
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
  /** Max time to wait for a shell `stopCommand` to finish, in ms. Default: 10000 */
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
  /** Shell command that starts the engine (run via /bin/sh, spawned as a child of the proxy). */
  startCommand?: string;
  /** Shell command that stops the engine (run via /bin/sh). */
  stopCommand?: string;
  /** Cooldown buffer after stopping, before starting the new engine. Overrides defaults. */
  coolDownMs?: number;
  /** Health/readiness endpoint path (relative to targetBaseUrl) that must return 200. */
  healthEndpoint?: string;
  /** Override VRAM threshold for this model (0..1). */
  vramThresholdPct?: number;
  /**
   * `false` = the engine runs on *another* host (or is fully externally managed):
   * the proxy never runs `startCommand`/`stopCommand` locally, spawns no child,
   * skips the local VRAM gate, and only polls the health endpoint before
   * forwarding. Default: `true` (engine lives on the proxy host).
   */
  managed?: boolean;
  /**
   * Optional provider API key forwarded to the target endpoint as
   * `Authorization: Bearer <key>`. Supports `${ENV_VAR}` expansion so secrets
   * need not live in the config file (unset vars are a load-time error).
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

/**
 * Expand `${ENV_VAR}` references from the environment. An unset variable is a
 * load-time config error — a literal `${MISSING}` silently sent to the
 * provider would be an opaque auth failure at request time.
 */
function expandEnv(value: string, ctx: string): string {
  return value.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (whole, name: string) => {
    const v = process.env[name];
    if (v === undefined) {
      throw new ConfigError(`${ctx}: environment variable ${name} is not set (referenced as ${whole})`);
    }
    return v;
  });
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
    // ajv's default enum message is generic; spell out the allowed values.
    const extra =
      e.keyword === "enum"
        ? ` (expected one of: ${(e.params as { allowedValues?: unknown[] }).allowedValues?.join(", ")})`
        : "";
    return `${path}: ${e.message ?? "invalid value"}${extra}`;
  });
  throw new ConfigError(`${fileName}: invalid config — ${problems.join("; ")}`);
}

/**
 * Schema-validated raw input -> InfermuxConfig: fills built-in defaults and
 * expands `${ENV_VAR}` references (the one transformation a schema cannot do).
 */
function finalize(raw: unknown, fileName: string): InfermuxConfig {
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
    const m = { ...entry } as unknown as ModelConfig;
    if (typeof m.targetApiKey === "string") {
      m.targetApiKey = expandEnv(m.targetApiKey, `${fileName}: models["${alias}"].targetApiKey`);
    }
    if (m.targetHeaders) {
      for (const [hk, v] of Object.entries(m.targetHeaders)) {
        m.targetHeaders[hk] = expandEnv(v, `${fileName}: models["${alias}"].targetHeaders["${hk}"]`);
      }
    }
    models[alias] = m;
  }

  return { server, defaults, models };
}

/**
 * Validate a parsed config object (see config.schema.json) and return the
 * fully-defaulted, typed InfermuxConfig.
 */
export function validateConfig(
  raw: unknown,
  fileName = "<memory>",
): InfermuxConfig {
  assertSchema(raw, fileName);
  return finalize(raw, fileName);
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
