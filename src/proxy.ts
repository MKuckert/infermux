import { Hono, type Context } from "hono";
import { timingSafeEqual } from "node:crypto";
import type { RuntimeConfig, ModelConfig } from "./types.js";
import type { EngineManager } from "./engine.js";
import { EngineError, errMsg } from "./engine.js";
import type { SerialQueue } from "./queue.js";
import type { Stats } from "./stats.js";
import type { VramMonitor } from "./vram.js";

export interface ProxyDeps {
  config: RuntimeConfig;
  engine: EngineManager;
  queue: SerialQueue;
  stats: Stats;
  vram: VramMonitor;
  log: (msg: string) => void;
}

function safeTokenEqual(a: string, b: string): boolean {
  const ha = Buffer.from(a);
  const hb = Buffer.from(b);
  if (ha.length !== hb.length) return false;
  return timingSafeEqual(ha, hb);
}

/** OpenAI-style JSON error body. */
function errorBody(message: string, type = "invalid_request_error", code?: string | null) {
  return { error: { message, type, param: null, code: code ?? null } };
}

function bearerToken(c: { req: { header(name: string): string | undefined } }): string | null {
  const auth = c.req.header("authorization");
  if (!auth) return null;
  const m = auth.match(/^Bearer\s+(.+)$/i);
  return m?.[1] ?? null;
}

export function buildApp(deps: ProxyDeps): Hono {
  const { config, engine, queue, stats, vram, log } = deps;
  const app = new Hono();
  const token = config.server.apiToken;



  // Proxy self-health (always available, no auth).
  app.get("/health", (c) => c.json({ status: "ok", name: "infermux" }));

  // ------------------------------------------------------------------
  // Auth gate for inference endpoints
  // ------------------------------------------------------------------
  const auth = async (c: Context, next: () => Promise<void>) => {
    if (!token) return next();
    const got = bearerToken(c);
    if (got === null || !safeTokenEqual(got, token)) {
      return c.json(errorBody("Invalid API key", "authentication_error", "invalid_api_key"), 401);
    }
    return next();
  };

  // ------------------------------------------------------------------
  // /v1/models — list configured aliases
  // ------------------------------------------------------------------
  app.get("/v1/models", auth, (c) => {
    const data = Object.keys(config.models).map((id) => ({
      id,
      object: "model",
      created: 0,
      owned_by: config.models[id]?.provider ?? "unknown",
    }));
    return c.json({ object: "list", data });
  });

  // ------------------------------------------------------------------
  // /v1/stats — engine + queue + vram + throughput
  // ------------------------------------------------------------------
  app.get("/v1/stats", (c) => {
    // Refresh VRAM on demand (cheap: one CLI call).
    void vram.query().then((v) => stats.setVram(v));
    // Optional provider-side metrics scrape (fire and forget, cached).
    scrapeProviderMetrics();
    return c.json(stats.snapshot());
  });

  const activeMetricsCfg = (): ModelConfig | null => {
    const alias = engine.activeModelAlias;
    return alias ? (config.models[alias] ?? null) : null;
  };

  async function scrapeProviderMetrics(): Promise<void> {
    const cfg = activeMetricsCfg();
    if (!cfg?.metrics?.endpoint) return;
    const url = `${cfg.targetBaseUrl.replace(/\/$/, "")}${cfg.metrics.endpoint}`;
    // Same auth as inference requests — some providers guard metrics routes too.
    const headers: Record<string, string> = {};
    for (const [hk, hv] of Object.entries(cfg.targetHeaders ?? {})) headers[hk] = hv;
    if (cfg.targetApiKey) headers.authorization = `Bearer ${cfg.targetApiKey}`;
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(2_000) });
      void res.body?.cancel();
      if (!res.ok) return;
      const doc = (await res.json()) as unknown;
      const out: Record<string, number | string> = {};
      const pick = (path?: string): void => {
        if (!path) return;
        let cur: unknown = doc;
        for (const part of path.split(".")) {
          if (cur === null || typeof cur !== "object") return;
          cur = (cur as Record<string, unknown>)[part];
        }
        if (typeof cur === "number" || typeof cur === "string") {
          out[path] = cur;
        }
      };
      pick(cfg.metrics.vramUsedMiB);
      pick(cfg.metrics.vramTotalMiB);
      if (Object.keys(out).length > 0) {
        stats.setProviderMetrics(cfg.provider, out);
        if (cfg.metrics.vramUsedMiB && cfg.metrics.vramTotalMiB && out[cfg.metrics.vramUsedMiB] !== undefined) {
          // Provider-reported VRAM takes precedence over host CLI readings.
          const used = Number(out[cfg.metrics.vramUsedMiB!]);
          const total = Number(out[cfg.metrics.vramTotalMiB!]);
          if (Number.isFinite(used) && Number.isFinite(total) && total > 0) {
            stats.setVram({ usedMiB: used, totalMiB: total, source: `${cfg.provider}-metrics`, pctUsed: used / total });
          }
        }
      }
    } catch {
      /* metrics are best-effort */
    }
  }

  // ------------------------------------------------------------------
  // Inference endpoints
  // ------------------------------------------------------------------
  app.post("/v1/chat/completions", auth, (c) => handleInference(c, "/v1/chat/completions"));
  app.post("/v1/completions", auth, (c) => handleInference(c, "/v1/completions"));

  app.notFound((c) => c.json(errorBody(`Not found: ${c.req.path}`, "invalid_request_error"), 404));
  app.onError((err, c) => {
    log(`proxy: unhandled error: ${errMsg(err)}`);
    return c.json(errorBody("Internal proxy error", "proxy_error", "internal"), 500);
  });

  async function handleInference(c: Context, path: string) {
    const d = config.defaults;
    let body: Record<string, unknown>;
    try {
      body = (await c.req.json()) as Record<string, unknown>;
    } catch {
      return c.json(errorBody("Request body must be valid JSON"), 400);
    }
    const alias = typeof body.model === "string" ? body.model : "";
    const cfg = config.models[alias];
    if (!cfg) {
      return c.json(
        {
          error: {
            message: `The model '${alias || "(none)"}' does not exist or is not configured in infermux. Configured: ${Object.keys(config.models).join(", ")}`,
            type: "invalid_request_error",
            param: "model",
            code: "model_not_found",
          },
        },
        404,
      );
    }
    const isStream = body.stream === true;

    // Combined abort: client disconnect OR request deadline.
    const ctl = new AbortController();
    const onClientAbort = (): void => ctl.abort(new Error("client disconnected"));
    if (c.req.raw.signal.aborted) onClientAbort();
    else c.req.raw.signal.addEventListener("abort", onClientAbort, { once: true });
    const deadline = setTimeout(() => ctl.abort(new Error("request timeout")), d.requestTimeoutMs);

    const started = Date.now();
    stats.recordRequestStart();
    log(
      `proxy: queued ${c.req.method} ${path} model="${alias}" stream=${isStream} (queue=${queue.length})`,
    );

    try {
      return await queue.enqueue(async () => {
        // The client may have given up (deadline elapsed, socket closed) while
        // waiting in the queue: fail fast instead of burning an engine switch
        // for a request nobody is waiting for.
        if (ctl.signal.aborted) {
          const reason = ctl.signal.reason;
          if (reason instanceof Error && reason.message === "request timeout") {
            throw new RequestTimeoutError(`Request timed out after ${d.requestTimeoutMs}ms while queued`);
          }
          return c.json(errorBody("Client disconnected", "proxy_error", "client_disconnected"), 400);
        }
        // Engine transition happens under the same single-concurrency gate,
        // so the running request always matches the active engine.
        await engine.ensureReady(alias, cfg);

        const upstreamUrl = `${cfg.targetBaseUrl.replace(/\/$/, "")}${path}`;
        const upstreamBody: Record<string, unknown> = { ...body };
        if (cfg.targetModel !== undefined) upstreamBody.model = cfg.targetModel;
        const headers: Record<string, string> = {
          "content-type": "application/json",
        };
        // Static provider headers (x-api-key etc.); targetApiKey, if set,
        // wins over a user-provided authorization entry.
        for (const [hk, hv] of Object.entries(cfg.targetHeaders ?? {})) {
          headers[hk] = hv;
        }
        if (cfg.targetApiKey) headers.authorization = `Bearer ${cfg.targetApiKey}`;

        let upstream: Response;
        try {
          upstream = await fetch(upstreamUrl, {
            method: "POST",
            headers,
            body: JSON.stringify(upstreamBody),
            signal: ctl.signal,
          });
        } catch (err) {
          // (stats.recordError() happens exactly once, in the outer catch —
          // this error path always lands there, so recording here too would
          // double-count a single failed request.)
          if (ctl.signal.aborted) {
            const reason = ctl.signal.reason;
            if (reason instanceof Error && reason.message === "request timeout") {
              throw new RequestTimeoutError(`Upstream ${upstreamUrl} timed out after ${d.requestTimeoutMs}ms`);
            }
            return c.json(errorBody("Client disconnected", "proxy_error", "client_disconnected"), 400);
          }
          throw new UpstreamError(`Failed to reach ${upstreamUrl}: ${errMsg(err)}`);
        }

        if (isStream) {
          const headers: Record<string, string> = {
            "content-type": upstream.headers.get("content-type") ?? "text/event-stream",
            "cache-control": upstream.headers.get("cache-control") ?? "no-cache",
            connection: "keep-alive",
          };
          const body = new ReadableStream<Uint8Array>({
            async start(controller) {
              try {
                const tokens = await pipeStream(upstream, controller, ctl.signal, started);
                const elapsed = Date.now() - started;
                stats.recordCompletedRequest(tokens, elapsed);
                log(
                  `proxy: finished ${alias} stream: ~${tokens} tokens in ${(elapsed / 1000).toFixed(1)}s (queue=${queue.length})`,
                );
              } catch (err) {
                stats.recordError();
                log(`proxy: stream aborted: ${errMsg(err)}`);
              } finally {
                if (ctl.signal.aborted) {
                  const reason = ctl.signal.reason instanceof Error ? ctl.signal.reason.message : "aborted";
                  log(`proxy: stream for ${alias} ended early (aborted: ${reason}); client received a truncated stream`);
                }
                try {
                  controller.close();
                } catch {
                  /* already closed */
                }
              }
            },
            cancel() {
              // Client went away mid-stream.
              ctl.abort(new Error("client disconnected"));
            },
          });
          return c.body(body, upstream.status as 200, headers);
        }

        const text = await upstream.text();
        const elapsed = Date.now() - started;
        if (upstream.ok) {
          let tokens = 0;
          try {
            const json = JSON.parse(text) as { usage?: { completion_tokens?: number } };
            tokens = json.usage?.completion_tokens ?? 0;
          } catch {
            /* non-JSON body: keep raw passthrough */
          }
          // Only successful responses pollute the tokens/sec sample; a
          // 5xx error body is not a completion.
          stats.recordCompletedRequest(tokens, elapsed);
        } else {
          stats.recordError();
        }
        // Non-2xx upstream responses pass through verbatim (status + body)
        // so clients see the provider's real error.
        return c.body(text, upstream.status as 200, { "content-type": upstream.headers.get("content-type") ?? "application/json" });
      });
    } catch (err) {
      stats.recordError();
      if (err instanceof RequestTimeoutError) {
        return c.json(errorBody(err.message, "proxy_error", "timeout"), 504);
      }
      if (err instanceof UpstreamError || err instanceof EngineError) {
        return c.json(errorBody(err.message, "proxy_error", "engine_error"), 503);
      }
      if (ctl.signal.aborted) {
        return c.json(errorBody("Client disconnected", "proxy_error", "client_disconnected"), 400);
      }
      throw err;
    } finally {
      clearTimeout(deadline);
      c.req.raw.signal.removeEventListener("abort", onClientAbort);
    }
  }

  return app;
}

/**
 * Pipe an upstream SSE response to the client while estimating tokens/sec.
 * Returns the estimated token count (one per non-DONE data chunk carrying
 * non-empty content/text).
 */
type StreamController = ReadableStreamDefaultController<Uint8Array>;

async function pipeStream(
  upstream: Response,
  out: StreamController,
  signal: AbortSignal,
  startedAt: number,
): Promise<number> {
  const dec = new TextDecoder();
  const reader = upstream.body?.getReader();
  if (!reader) throw new Error("upstream has no response body");
  let tokens = 0;
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!signal.aborted) {
        try {
          out.enqueue(value);
        } catch {
          // Client already disconnected; stop relaying.
          break;
        }
      }
      if (!signal.aborted) {
        buffer += dec.decode(value, { stream: true });
        let idx: number;
        while ((idx = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, idx).replace(/\r$/, "");
          buffer = buffer.slice(idx + 1);
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "" || payload === "[DONE]") continue;
          try {
            const json = JSON.parse(payload) as Record<string, unknown>;
            const choices = Array.isArray(json.choices) ? (json.choices as Record<string, unknown>[]) : [];
            for (const choice of choices) {
              const delta = choice.delta as Record<string, unknown> | undefined;
              const text = choice.text as string | undefined;
              const piece =
                (typeof delta?.content === "string" && delta.content) ||
                (typeof text === "string" && text) ||
                "";
              if (piece.length > 0) tokens++;
            }
          } catch {
            /* keep-alive / malformed chunk: not a token */
          }
        }
      }
      if (signal.aborted) break;
    }
  } finally {
    try {
      reader.releaseLock();
      if (signal.aborted) await upstream.body?.cancel().catch(() => {});
    } catch {
      /* ignore */
    }
  }
  // The `startedAt` clock (consumed by the caller) includes queue wait and
  // engine switch time, so the derived tokens/sec is an honest end-to-end
  // figure from the client's perspective.
  return tokens;
}

/** Thrown when the per-request deadline expires before the upstream responds. */
export class RequestTimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RequestTimeoutError";
  }
}

export class UpstreamError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UpstreamError";
  }
}
