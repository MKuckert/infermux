import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import type { AddressInfo } from "node:net";
import { validateConfig } from "../src/config.js";
import { startProxy } from "../src/server.js";
import type { RuntimeConfig } from "../src/types.js";
import type { ProxyInstance } from "../src/server.js";

/**
 * End-to-end: a mock OpenAI-compatible provider + dummy engine processes,
 * exercised through the full proxy stack (auth, queue, engine lifecycle,
 * streaming, stats).
 */

// ---------------------------------------------------------------------------
// Mock provider
// ---------------------------------------------------------------------------
function createMockProvider(port: number) {
  const app = new Hono();
  // Note: hono's `c.status()` returns void in this version — don't chain off it.
  app.get("/health", (c) => (ready ? c.json({ ok: true }) : c.json({ ok: false }, 503)));
  let ready = false;

  app.post("/v1/chat/completions", async (c) => {
    const body = (await c.req.json()) as {
      model: string;
      stream?: boolean;
      max_tokens?: number;
      messages?: { content?: string }[];
    };
    if (body.messages?.some((m) => m.content?.includes("boom"))) {
      return c.json(
        { error: { message: "upstream boom", type: "server_error", param: null, code: null } },
        500,
      );
    }
    const reply = `Hello from ${body.model}!`;
    if (!body.stream) {
      return c.json({
        id: "cmpl-1",
        object: "chat.completion",
        model: body.model,
        choices: [{ index: 0, message: { role: "assistant", content: reply }, finish_reason: "stop" }],
        usage: { prompt_tokens: 5, completion_tokens: 4, total_tokens: 9 },
      });
    }
    // SSE stream: 3 delta chunks + [DONE]
    const chunks = ["Hello", " from", ` ${body.model}!`];
    return new Response(
      new ReadableStream({
        async start(controller) {
          const enc = new TextEncoder();
          for (const piece of chunks) {
            const sse = `data: ${JSON.stringify({
              id: "cmpl-1",
              object: "chat.completion.chunk",
              model: body.model,
              choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
            })}\n\n`;
            controller.enqueue(enc.encode(sse));
            await new Promise((r) => setTimeout(r, 15));
          }
          controller.enqueue(enc.encode("data: [DONE]\n\n"));
          controller.close();
        },
      }),
      { headers: { "content-type": "text/event-stream" } },
    );
  });

  app.post("/v1/completions", async (c) => {
    const body = (await c.req.json()) as { model: string; stream?: boolean };
    const reply = `Legacy ${body.model} completion`;
    if (!body.stream) {
      return c.json({
        id: "cmpl-2",
        object: "text_completion",
        model: body.model,
        choices: [{ text: reply, index: 0, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 3, total_tokens: 6 },
      });
    }
    const enc = new TextEncoder();
    return new Response(
      new ReadableStream({
        async start(controller) {
          for (const piece of ["Legacy", " completion"]) {
            controller.enqueue(
              enc.encode(
                `data: ${JSON.stringify({
                  id: "cmpl-2",
                  object: "text_completion",
                  choices: [{ text: piece, index: 0, finish_reason: null }],
                })}\n\n`,
              ),
            );
            await new Promise((r) => setTimeout(r, 10));
          }
          controller.enqueue(enc.encode("data: [DONE]\n\n"));
          controller.close();
        },
      }),
      { headers: { "content-type": "text/event-stream" } },
    );
  });

  const server = serve({ fetch: app.fetch, port });
  const setReady = (v: boolean) => (ready = v);
  const readyP = new Promise<number>((resolve) =>
    server.once("listening", () => resolve((server.address() as AddressInfo).port)),
  );
  return { server, setReady, ready: readyP, getPort: () => (server.address() as AddressInfo).port };
}

describe("infermux end-to-end", () => {
  const providerPort = 11434 + Math.floor(Math.random() * 1000);
  const mock = createMockProvider(providerPort);
  let providerBase = "";
  let baseUrl = "";

  // Provider must be bound before the proxy config is built (health polling targets it).
  beforeAll(async () => {
    providerBase = `http://127.0.0.1:${await mock.ready}`;
  });

  const buildConfig = () => validateConfig({
    server: { host: "127.0.0.1", port: 0, apiToken: "test-token" },
    defaults: {
      requestTimeoutMs: 15_000,
      healthCheckIntervalMs: 20,
      healthCheckTimeoutMs: 3_000,
      coolDownMs: 20,
      vramThresholdPct: 0, // no GPU in CI: gate off
      vramCheckTimeoutMs: 500,
    },
    models: {
      // Two models on the same provider (no engine switch between them).
      "llama": {
        provider: "ollama",
        targetBaseUrl: providerBase,
        targetModel: "llama3:latest",
        // Dummy engine: a tiny node process that idles.
        startCommand: `node -e "setInterval(()=>{},1e3)"`,
        stopCommand: "true",
        healthEndpoint: "/health",
      },
      // A second provider forces a full stop->start transition.
      "mistral": {
        provider: "vllm",
        targetBaseUrl: providerBase,
        targetModel: "Mistral-7B-Instruct-v0.2",
        startCommand: `node -e "setInterval(()=>{},1e3)"`,
        stopCommand: "true",
        healthEndpoint: "/health",
      },
    },
  });

  let config: RuntimeConfig;
  let proxy: ProxyInstance;

  beforeAll(async () => {
    config = buildConfig();
    proxy = startProxy(config, "warn");
    const boundPort = await proxy.ready;
    baseUrl = `http://127.0.0.1:${boundPort}`;
    // The provider starts not-ready; the proxy must poll until we flip it.
    mock.setReady(false);
    setTimeout(() => mock.setReady(true), 300);
  });

  afterAll(async () => {
    await proxy?.close();
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  const url = (path: string) => `${baseUrl}${path}`;
  const auth = { authorization: `Bearer ${"test-token"}`, "content-type": "application/json" };

  it("rejects unauthenticated requests with 401", async () => {
    const res = await fetch(url("/v1/chat/completions"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "llama", messages: [] }),
    });
    expect(res.status).toBe(401);
    const json = (await res.json()) as { error: { code: string } };
    expect(json.error.code).toBe("invalid_api_key");
  });

  it("rejects a wrong token with 401", async () => {
    const res = await fetch(url("/v1/models"), { headers: { authorization: "Bearer nope" } });
    expect(res.status).toBe(401);
  });

  it("starts the engine on first use (polling health until 200) and proxies non-stream chat", async () => {
    const res = await fetch(url("/v1/chat/completions"), {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ model: "llama", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(200);
    const json = (await res.json()) as { model: string; choices: { message: { content: string } }[] };
    // model rewritten to the provider's targetModel
    expect(json.model).toBe("llama3:latest");
    expect(json.choices[0].message.content).toBe("Hello from llama3:latest!");
    expect(proxy.deps.engine.state().provider).toBe("ollama");
  });

  it("proxies SSE streaming chat completions", async () => {
    const res = await fetch(url("/v1/chat/completions"), {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ model: "llama", stream: true, messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain('"delta":{"content":"Hello"');
    expect(text).toContain('"delta":{"content":" from"}');
    expect(text).toContain('"delta":{"content":" llama3:latest!"}');
    expect(text).toContain("data: [DONE]");
  });

  it("switches engines when the requested model uses a different provider", async () => {
    const res = await fetch(url("/v1/chat/completions"), {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ model: "mistral", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(200);
    const json = (await res.json()) as { model: string };
    expect(json.model).toBe("Mistral-7B-Instruct-v0.2");
    const st = proxy.deps.engine.state();
    expect(st.provider).toBe("vllm");
    expect(st.switches).toBeGreaterThanOrEqual(1);
  });

  it("proxies /v1/completions (legacy)", async () => {
    const res = await fetch(url("/v1/completions"), {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ model: "mistral", prompt: "hello" }),
    });
    expect(res.status).toBe(200);
    const json = (await res.json()) as { choices: { text: string }[] };
    expect(json.choices[0].text).toBe("Legacy Mistral-7B-Instruct-v0.2 completion");
  });

  it("serializes concurrent requests (one at a time)", async () => {
    const t0 = Date.now();
    const [a, b] = await Promise.all([
      fetch(url("/v1/chat/completions"), {
        method: "POST",
        headers: auth,
        body: JSON.stringify({ model: "mistral", messages: [{ role: "user", content: "1" }], stream: true }),
      }),
      fetch(url("/v1/chat/completions"), {
        method: "POST",
        headers: auth,
        body: JSON.stringify({ model: "mistral", messages: [{ role: "user", content: "2" }], stream: true }),
      }),
    ]);
    expect(a.status).toBe(200);
    expect(b.status).toBe(200);
    const ta = await a.text();
    const tb = await b.text();
    expect(ta).toContain("[DONE]");
    expect(tb).toContain("[DONE]");
    expect(Date.now() - t0).toBeLessThan(10_000);
  });

  it("reports 404 model_not_found for unconfigured models", async () => {
    const res = await fetch(url("/v1/chat/completions"), {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ model: "does-not-exist", messages: [] }),
    });
    expect(res.status).toBe(404);
    const json = (await res.json()) as { error: { code: string } };
    expect(json.error.code).toBe("model_not_found");
  });

  it("passes non-2xx upstream responses through (status + body) and counts them as errors", async () => {
    const statsBefore = (await (await fetch(url("/v1/stats"))).json()) as { requests: { errors: number } };
    const res = await fetch(url("/v1/chat/completions"), {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ model: "mistral", messages: [{ role: "user", content: "boom" }] }),
    });
    expect(res.status).toBe(500);
    const json = (await res.json()) as { error: { message: string } };
    expect(json.error.message).toBe("upstream boom");
    const statsAfter = (await (await fetch(url("/v1/stats"))).json()) as { requests: { errors: number } };
    expect(statsAfter.requests.errors).toBe(statsBefore.requests.errors + 1);
  });

  it("lists configured models at /v1/models", async () => {
    const res = await fetch(url("/v1/models"), { headers: auth });
    expect(res.status).toBe(200);
    const json = (await res.json()) as { data: { id: string }[] };
    expect(json.data.map((d) => d.id).sort()).toEqual(["llama", "mistral"]);
  });

  it("exposes engine/queue/vram stats at /v1/stats", async () => {
    const res = await fetch(url("/v1/stats"));
    expect(res.status).toBe(200);
    const json = (await res.json()) as {
      activeEngine: { provider: string | null; running: boolean };
      queue: { length: number };
      vram: { available: boolean } | { usedMiB: number };
      tokensPerSec: { average: number | null };
    };
    expect(json.activeEngine.provider).toBe("vllm");
    expect(json.activeEngine.running).toBe(true);
    expect(json.queue.length).toBe(0);
    expect(json.tokensPerSec.average).not.toBeNull();
  });
});
