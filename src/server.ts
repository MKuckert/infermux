import { serve as nodeServe } from "@hono/node-server";
import type { AddressInfo } from "node:net";
import type { RuntimeConfig } from "./types.js";
import { EngineManager } from "./engine.js";
import { SerialQueue } from "./queue.js";
import { Stats } from "./stats.js";
import { VramMonitor } from "./vram.js";
import { buildApp, type ProxyDeps } from "./proxy.js";

export type Logger = (msg: string) => void;

/**
 * Structural superset of node's http.Server and Bun's Server — all infermux
 * needs is `close()`. Declared locally so we don't depend on @types/bun.
 */
export interface ServerHandle {
  close(cb?: () => void): unknown;
}

interface BunServeHandle {
  close(): void;
  port?: number;
}

interface BunGlobal {
  serve(opts: {
    port: number;
    hostname: string;
    fetch: (req: Request) => Response | Promise<Response>;
  }): BunServeHandle;
}

export function makeLogger(prefix: string, level: "debug" | "info" | "warn" | "error" = "info"): Logger {
  const out = { debug: console.debug, info: console.info, warn: console.warn, error: console.error }[level];
  // The level is tagged so consumers can grep it; callers should not embed
  // their own level prefixes in the message.
  return (msg: string) => {
    out(`[${new Date().toISOString()}] [${level}] ${prefix}: ${msg}`);
  };
}

export interface ProxyInstance {
  deps: ProxyDeps;
  server: ServerHandle;
  /** Resolves with the actual bound port once the server is listening. */
  ready: Promise<number>;
  close(): Promise<void>;
}

export function startProxy(config: RuntimeConfig, level: "debug" | "info" | "warn" | "error" = "info"): ProxyInstance {
  const log = makeLogger("infermux", level);
  log(`starting (config: models=${Object.keys(config.models).length})`);
  const queue = new SerialQueue();
  const vram = new VramMonitor(log, config.defaults.vramBackend);
  const engine = new EngineManager(config.defaults, vram, log);
  const stats = new Stats(engine, queue);
  const deps: ProxyDeps = { config, engine, queue, stats, vram, log };
  const app = buildApp(deps);

  const { host, port } = config.server;

  // Hono is runtime-agnostic: serve on Bun natively, on Node via the adapter.
  const bun = (globalThis as { Bun?: BunGlobal }).Bun;
  const isBun = bun !== undefined;
  let server: ServerHandle;
  let ready: Promise<number>;
  const listeningLog = (): void => {
    log(
      `listening on http://${host}:${port} (models: ${Object.keys(config.models).join(", ")}, auth: ${config.server.apiToken ? "required" : "off"}, runtime: ${isBun ? "bun" : "node"})`,
    );
  };

  if (isBun) {
    const bs = bun!.serve({ port, hostname: host, fetch: app.fetch });
    server = {
      close: () => {
        bs.close();
      },
    };
    ready = Promise.resolve(bs.port ?? port);
    listeningLog();
  } else {
    const nodeServer = nodeServe({ fetch: app.fetch, hostname: host, port }, (info) => {
      const addr = info.address ?? `${host}:${port}`;
      log(
        `listening on http://${addr} (models: ${Object.keys(config.models).join(", ")}, auth: ${config.server.apiToken ? "required" : "off"}, runtime: node)`,
      );
    });
    server = nodeServer as unknown as ServerHandle; // http.Server is structurally compatible
    ready = new Promise<number>((resolvePort, rejectPort) => {
      nodeServer.once("listening", () => {
        const a = nodeServer.address();
        resolvePort(a ? (a as AddressInfo).port : port);
      });
      nodeServer.once("error", rejectPort);
    });
  }

  return {
    deps,
    server,
    ready,
    async close() {
      // 1. Stop accepting new connections (in-flight + queued ones keep going).
      // 2. Let queued requests finish *while the engine is still running*.
      // 3. Only then tear down the engine, so a request running through the
      //    queue doesn't restart an engine that was just shut down.
      const closed = new Promise<void>((resolve) => {
        // Bun's close() is synchronous and takes no callback; Node's does.
        if (isBun) {
          server.close();
          resolve();
        } else {
          server.close(() => resolve());
          // Node 18 keeps open keep-alive sockets after close() and waits for
          // them in the callback — idle ones must be destroyed explicitly, or
          // a single keep-alive client (the OpenAI SDKs do this) blocks shutdown.
          (server as unknown as { closeIdleConnections?: () => void }).closeIdleConnections?.();
        }
      });
      await queue.drain().catch(() => {});
      await engine.shutdown().catch(() => {});
      await closed;
    },
  };
}
