import { existsSync, readFileSync } from "node:fs";
import { resolve, extname } from "node:path";
import * as yaml from "js-yaml";
import { VRAM_BACKENDS, type VramBackendName } from "./vram.js";

/**
 * Configuration model for infermux.
 *
 * Supported file formats: .json, .jsonc (naively stripped of // and /* *\/ comments),
 * .yaml / .yml.
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
   * precedence over a `authorization` entry here.
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

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}

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

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/**
 * Pick a duration field. Must be a finite, non-negative number, and > 0 when
 * `nonNegative` is false. Returns undefined when the key is absent.
 */
function pickDurMs(
  o: Record<string, unknown>,
  key: string,
  fileName: string,
  nonNegative: boolean,
): number | undefined {
  const v = o[key];
  if (v === undefined) return undefined;
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new ConfigError(`${fileName}: ${key} must be a finite number (milliseconds)`);
  }
  if (nonNegative ? v < 0 : v <= 0) {
    throw new ConfigError(`${fileName}: ${key} must be ${nonNegative ? ">= 0" : "> 0"} ms`);
  }
  return v;
}

function pickStr(o: Record<string, unknown>, key: string): string | undefined {
  const v = o[key];
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

function pickNum(o: Record<string, unknown>, key: string): number | undefined {
  const v = o[key];
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

export function validateConfig(
  raw: unknown,
  fileName = "<memory>",
): InfermuxConfig {
  if (!isObj(raw)) {
    throw new ConfigError(`${fileName}: config root must be an object`);
  }
  const serverRaw = isObj(raw.server) ? raw.server : {};
  const defaultsRaw = isObj(raw.defaults) ? raw.defaults : {};
  const modelsRaw = isObj(raw.models) ? raw.models : {};

  const server: ServerConfig = {
    host: pickStr(serverRaw, "host") ?? "0.0.0.0",
    port: pickNum(serverRaw, "port") ?? 8080,
  };
  if (pickStr(serverRaw, "apiToken") !== undefined) {
    server.apiToken = pickStr(serverRaw, "apiToken");
  }
  if (server.port < 0 || server.port > 65535) {
    throw new ConfigError(`${fileName}: server.port must be 0-65535`);
  }

  const d = defaultsRaw;
  const dur = (key: keyof DefaultsConfig, nonNegative: boolean) =>
    pickDurMs(d, key, `${fileName}: defaults.${key}`, nonNegative);
  const defaults: Required<DefaultsConfig> = {
    requestTimeoutMs: dur("requestTimeoutMs", false) ?? DEFAULTS.requestTimeoutMs,
    healthCheckIntervalMs: dur("healthCheckIntervalMs", false) ?? DEFAULTS.healthCheckIntervalMs,
    healthCheckTimeoutMs: dur("healthCheckTimeoutMs", false) ?? DEFAULTS.healthCheckTimeoutMs,
    coolDownMs: dur("coolDownMs", true) ?? DEFAULTS.coolDownMs,
    vramThresholdPct: pickNum(d, "vramThresholdPct") ?? DEFAULTS.vramThresholdPct,
    vramCheckTimeoutMs: dur("vramCheckTimeoutMs", false) ?? DEFAULTS.vramCheckTimeoutMs,
    vramCheckIntervalMs: dur("vramCheckIntervalMs", false) ?? DEFAULTS.vramCheckIntervalMs,
    childKillGraceMs: dur("childKillGraceMs", false) ?? DEFAULTS.childKillGraceMs,
    stopCommandTimeoutMs: dur("stopCommandTimeoutMs", false) ?? DEFAULTS.stopCommandTimeoutMs,
    vramBackend: (VRAM_BACKENDS as readonly string[]).includes(pickStr(d, "vramBackend") ?? "auto")
      ? ((pickStr(d, "vramBackend") ?? "auto") as VramBackendName)
      : (() => {
          throw new ConfigError(
            `${fileName}: defaults.vramBackend must be one of ${VRAM_BACKENDS.join(", ")}`,
          );
        })(),
  };
  if (defaults.vramThresholdPct < 0 || defaults.vramThresholdPct > 1) {
    throw new ConfigError(`${fileName}: defaults.vramThresholdPct must be between 0 and 1`);
  }

  const models: Record<string, ModelConfig> = {};
  for (const [alias, entry] of Object.entries(modelsRaw)) {
    if (!isObj(entry)) {
      throw new ConfigError(`${fileName}: models["${alias}"] must be an object`);
    }
    const provider = pickStr(entry, "provider");
    const targetBaseUrl = pickStr(entry, "targetBaseUrl");
    if (!provider) throw new ConfigError(`${fileName}: models["${alias}"].provider is required`);
    if (!targetBaseUrl)
      throw new ConfigError(`${fileName}: models["${alias}"].targetBaseUrl is required`);
    if (!/^https?:\/\//.test(targetBaseUrl))
      throw new ConfigError(
        `${fileName}: models["${alias}"].targetBaseUrl must start with http:// or https://`,
      );
    const m: ModelConfig = { provider, targetBaseUrl };
    const tm = pickStr(entry, "targetModel");
    if (tm !== undefined) m.targetModel = tm;
    const sc = pickStr(entry, "startCommand");
    if (sc !== undefined) m.startCommand = sc;
    const stc = pickStr(entry, "stopCommand");
    if (stc !== undefined) m.stopCommand = stc;
    const cool = pickDurMs(entry, "coolDownMs", `${fileName}: models["${alias}"].coolDownMs`, true);
    if (cool !== undefined) m.coolDownMs = cool;
    const he = pickStr(entry, "healthEndpoint");
    if (he !== undefined) {
      if (!he.startsWith("/"))
        throw new ConfigError(`${fileName}: models["${alias}"].healthEndpoint must start with "/"`);
      m.healthEndpoint = he;
    }
    const vt = pickNum(entry, "vramThresholdPct");
    if (vt !== undefined) {
      if (vt < 0 || vt > 1)
        throw new ConfigError(`${fileName}: models["${alias}"].vramThresholdPct must be 0..1`);
      m.vramThresholdPct = vt;
    }
    const key = pickStr(entry, "targetApiKey");
    if (key !== undefined) {
      m.targetApiKey = expandEnv(key, `${fileName}: models["${alias}"].targetApiKey`);
    }
    if (isObj(entry.targetHeaders)) {
      const headers: Record<string, string> = {};
      for (const [hk, hv] of Object.entries(entry.targetHeaders)) {
        if (typeof hv !== "string" || !hk.trim()) {
          throw new ConfigError(`${fileName}: models["${alias}"].targetHeaders must map strings to strings`);
        }
        headers[hk] = expandEnv(hv, `${fileName}: models["${alias}"].targetHeaders["${hk}"]`);
      }
      m.targetHeaders = headers;
    }
    if (typeof entry.managed === "boolean") m.managed = entry.managed;
    if (isObj(entry.metrics)) {
      const metrics: MetricsMapping = {};
      const me = pickStr(entry.metrics, "endpoint");
      if (me !== undefined) {
        if (!me.startsWith("/"))
          throw new ConfigError(`${fileName}: models["${alias}"].metrics.endpoint must start with "/"`);
        metrics.endpoint = me;
      }
      const vu = pickStr(entry.metrics, "vramUsedMiB");
      if (vu !== undefined) metrics.vramUsedMiB = vu;
      const vtt = pickStr(entry.metrics, "vramTotalMiB");
      if (vtt !== undefined) metrics.vramTotalMiB = vtt;
      m.metrics = metrics;
    }
    models[alias] = m;
  }
  if (Object.keys(models).length === 0) {
    throw new ConfigError(`${fileName}: at least one entry under "models" is required`);
  }

  return { server, defaults, models };
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
