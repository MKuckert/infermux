#!/usr/bin/env node
import { resolve } from "node:path";
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { loadConfigFile, ConfigError } from "./config.js";
import { startProxy } from "./server.js";
import type { RuntimeConfig } from "./types.js";

function printUsage(): void {
  console.log(`infermux — OpenAI-compatible proxy with single-concurrency queueing

Usage: infermux [--config <path>] [--log-level <debug|info|warn|error>]

Options:
  --config <path>      Config file (JSON / JSONC / YAML).
                       Default: $INFERMUX_CONFIG, else ./config.json, ./config.yaml, ./config.yml
  --log-level <level>  Log verbosity. Default: info (env INFERMUX_LOG_LEVEL)

Environment:
  INFERMUX_CONFIG          Config file path
  INFERMUX_LOG_LEVEL       Log level
  INFERMUX_HOST / PORT     Override server host / port
  INFERMUX_API_TOKEN       Override server api token
  INFERMUX_REQUEST_TIMEOUT_MS   Override defaults.requestTimeoutMs`
);
}

export function parseArgs(argv: string[]): { configPath?: string; logLevel?: string } {
  const out: { configPath?: string; logLevel?: string } = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") {
      printUsage();
      process.exit(0);
    } else if (a === "--config" || a === "-c") {
      out.configPath = argv[++i];
    } else if (a === "--log-level") {
      out.logLevel = argv[++i];
    } else {
      console.error(`infermux: unknown argument: ${a}`);
      printUsage();
      process.exit(2);
    }
  }
  return out;
}

function findDefaultConfig(): string {
  const candidates = ["config.json", "config.yaml", "config.yml", "config.jsonc"];
  for (const c of candidates) {
    const abs = resolve(process.cwd(), c);
    if (existsSync(abs)) return abs;
  }
  return resolve(process.cwd(), "config.json");
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const configPath =
    args.configPath ?? process.env.INFERMUX_CONFIG ?? findDefaultConfig();

  let config: RuntimeConfig;
  try {
    config = loadConfigFile(configPath) as RuntimeConfig;
  } catch (err) {
    if (err instanceof ConfigError) {
      console.error(`infermux: ${err.message}`);
    } else {
      console.error(`infermux: failed to load config: ${err}`);
    }
    process.exit(2);
  }

  // Environment overrides.
  if (process.env.INFERMUX_HOST) config.server.host = process.env.INFERMUX_HOST;
  if (process.env.INFERMUX_PORT) {
    const p = Number(process.env.INFERMUX_PORT);
    if (!Number.isInteger(p) || p < 0 || p > 65535) {
      console.error("infermux: INFERMUX_PORT must be an integer 0-65535");
      process.exit(2);
    }
    config.server.port = p;
  }
  if (process.env.INFERMUX_API_TOKEN !== undefined) {
    config.server.apiToken = process.env.INFERMUX_API_TOKEN || undefined;
  }
  if (process.env.INFERMUX_REQUEST_TIMEOUT_MS) {
    const t = Number(process.env.INFERMUX_REQUEST_TIMEOUT_MS);
    if (Number.isFinite(t) && t > 0) config.defaults.requestTimeoutMs = t;
  }

  const levelRaw = args.logLevel ?? process.env.INFERMUX_LOG_LEVEL ?? "info";
  const levels = ["debug", "info", "warn", "error"] as const;
  if (!levels.includes(levelRaw as (typeof levels)[number])) {
    console.error(`infermux: invalid --log-level "${levelRaw}" (expected ${levels.join(" | ")})`);
    process.exit(2);
  }
  const level = levelRaw as "debug" | "info" | "warn" | "error";

  const proxy = startProxy(config, level);
  const { log } = proxy.deps;

  let shuttingDown = false;
  const shutdown = async (signal: string): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    log(`received ${signal}; shutting down (draining queue, stopping engine)`);
    try {
      await proxy.close();
    } catch (err) {
      log(`shutdown error: ${err}`);
    }
    process.exit(0);
  };

  process.on("SIGINT", () => void shutdown("SIGINT"));
  process.on("SIGTERM", () => void shutdown("SIGTERM"));
  process.on("unhandledRejection", (err) => {
    log(`unhandled rejection: ${err}`);
  });
}

// Only run when invoked directly (`node dist/cli.js`), not when imported
// (e.g. by unit tests — importing must not start the server or exit).
const isMainEntry =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href;

if (isMainEntry) {
  main().catch((err) => {
    console.error(`infermux: fatal: ${err}`);
    process.exit(1);
  });
}
