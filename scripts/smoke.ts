/**
 * Smoke test: starts a mock OpenAI provider + the built infermux CLI,
 * then exercises it with real HTTP requests (auth, non-stream, stream,
 * stats, engine switch, shutdown).
 */
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import type { AddressInfo } from "node:net";

const ROOT = new URL("..", import.meta.url).pathname;

// --- mock provider ---------------------------------------------------------
const app = new Hono();
let ready = false;
app.get("/health", (c) => (ready ? c.json({ ok: true }) : c.status(503).json({ ok: false })));
app.post("/v1/chat/completions", async (c) => {
  const b = (await c.req.json()) as { model: string; stream?: boolean };
  if (!b.stream)
    return c.json({
      id: "cmpl-s", object: "chat.completion", model: b.model,
      choices: [{ index: 0, message: { role: "assistant", content: `ok:${b.model}` }, finish_reason: "stop" }],
      usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
    });
  const enc = new TextEncoder();
  return new Response(
    new ReadableStream({
      async start(ctrl) {
        for (const p of ["tok1 ", "tok2 ", "tok3!"]) {
          ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ id: "c", object: "chat.completion.chunk", model: b.model, choices: [{ index: 0, delta: { content: p }, finish_reason: null }] })}\n\n`));
          await new Promise((r) => setTimeout(r, 30));
        }
        ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
        ctrl.close();
      },
    }),
    { headers: { "content-type": "text/event-stream" } },
  );
});
const pServer = serve({ fetch: app.fetch, port: 18901 });
await new Promise<void>((r) => pServer.on("listening", () => r()));
const pPort = (pServer.address() as AddressInfo).port;
setTimeout(() => (ready = true), 250);
console.log(`[smoke] mock provider on :${pPort}`);

// --- infermux config -------------------------------------------------------
const config = {
  server: { host: "127.0.0.1", port: 18902, apiToken: "smoke-token" },
  defaults: { healthCheckIntervalMs: 50, healthCheckTimeoutMs: 5000, coolDownMs: 50, vramThresholdPct: 0 },
  models: {
    llama: {
      provider: "ollama", targetBaseUrl: `http://127.0.0.1:${pPort}`, targetModel: "llama3:latest",
      startCommand: `node -e "setInterval(()=>{},1e3)"`, stopCommand: "true", healthEndpoint: "/health",
    },
    mistral: {
      provider: "vllm", targetBaseUrl: `http://127.0.0.1:${pPort}`, targetModel: "Mistral-7B",
      startCommand: `node -e "setInterval(()=>{},1e3)"`, stopCommand: "true", healthEndpoint: "/health",
    },
  },
};
writeFileSync(`${ROOT}smoke-config.json`, JSON.stringify(config, null, 2));

// --- start the built CLI ---------------------------------------------------
const cli = spawn("node", ["dist/cli.js", "--config", "smoke-config.json", "--log-level", "info"], {
  cwd: ROOT, stdio: ["ignore", "inherit", "inherit"],
});
// wait for the proxy
for (let i = 0; i < 50; i++) {
  try {
    const r = await fetch("http://127.0.0.1:18902/health");
    if (r.ok) break;
  } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 200));
}
const B = "http://127.0.0.1:18902";
const H = { authorization: "Bearer smoke-token", "content-type": "application/json" };

const check = (name: string, ok: boolean, extra = ""): void => {
  console.log(`[smoke] ${ok ? "PASS" : "FAIL"} ${name}${extra ? ` — ${extra}` : ""}`);
  if (!ok) process.exitCode = 1;
};

// 1: unauthenticated
let r = await fetch(`${B}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, body: '{"model":"llama","messages":[]}' });
check("401 without token", r.status === 401, `got ${r.status}`);

// 2: non-stream (triggers engine start + health poll)
r = await fetch(`${B}/v1/chat/completions`, { method: "POST", headers: H, body: JSON.stringify({ model: "llama", messages: [{ role: "user", content: "hi" }] }) });
const j1 = (await r.json()) as any;
check("non-stream chat", r.status === 200 && j1.choices?.[0]?.message?.content === "ok:llama3:latest", JSON.stringify(j1).slice(0, 120));

// 3: stream
r = await fetch(`${B}/v1/chat/completions`, { method: "POST", headers: H, body: JSON.stringify({ model: "llama", stream: true, messages: [{ role: "user", content: "hi" }] }) });
const sse = await r.text();
check("stream chat", r.status === 200 && (r.headers.get("content-type") ?? "").includes("text/event-stream") && sse.includes("tok1") && sse.includes("[DONE]"), `(${sse.length} bytes)`);

// 4: engine switch to the other provider
r = await fetch(`${B}/v1/chat/completions`, { method: "POST", headers: H, body: JSON.stringify({ model: "mistral", messages: [{ role: "user", content: "hi" }] }) });
const j2 = (await r.json()) as any;
check("engine switch", r.status === 200 && j2.model === "Mistral-7B", JSON.stringify(j2).slice(0, 120));

// 5: models list
r = await fetch(`${B}/v1/models`, { headers: H });
const j3 = (await r.json()) as any;
check("models list", r.status === 200 && j3.data.length === 2);

// 6: stats
r = await fetch(`${B}/v1/stats`);
const j4 = (await r.json()) as any;
check("stats", r.status === 200 && j4.activeEngine?.provider === "vllm" && j4.queue?.length === 0 && typeof j4.tokensPerSec?.average === "number", JSON.stringify(j4.tokensPerSec));

// 7: graceful shutdown
cli.kill("SIGTERM");
await new Promise((r) => cli.on("exit", r));
console.log("[smoke] cli exited after SIGTERM");

pServer.close();
setTimeout(() => process.exit(process.exitCode ?? 0), 300);
