# infermux

A production-ready **OpenAI-compatible API proxy** for local inference engines
(vLLM, Ollama, LM Studio, llama.cpp server, …).

Local LLM servers are expensive to run — usually you can only afford **one at a
time**. infermux sits in front of any number of engines and solves this with:

1. **OpenAI API compatibility** — full `/v1/chat/completions` and
   `/v1/completions`, including SSE streaming (`"stream": true`).
2. **Strict single-concurrency queue** — exactly one request is processed at a
   time, FIFO, with a configurable per-request deadline so queued clients wait
   their turn instead of timing out early.
3. **Dynamic engine lifecycles** — when the requested model is served by a
   *different* engine than the active one, infermux automatically:
   1. runs the active engine's `stopCommand` (and/or `SIGTERM`→`SIGKILL` on the child process it spawned),
   2. waits a configurable `coolDownMs` buffer for GPU driver cleanup,
   3. verifies VRAM has dropped below a threshold (optional, OS-agnostic),
   4. runs the new engine's `startCommand`, then
   5. polls the provider's health endpoint until it returns HTTP 200 —
      *only then* forwards the request.
4. **Auth & config** — configurable host/port, optional `Bearer` token
   (401 on missing/invalid), JSON / JSONC / YAML config file.

## Quick start

```bash
npm install
npm run build

# configure
cp config.example.json config.json
# edit config.json: ports, apiToken, your models & their start/stop commands

npm start            # or: node dist/cli.js --config config.json
```

Try it (uses your normal OpenAI client, just pointed at infermux):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer replace-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"my-custom-llama","messages":[{"role":"user","content":"hi"}],"stream":true}'

# OpenAI SDK:
# client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="replace-me")
```

## Configuration

`config.json` (also accepts `.yaml`/`.yml`; JSON files may contain `//` comments):

```jsonc
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "apiToken": "replace-me"        // optional; omit to disable auth
  },
  "defaults": {
    "requestTimeoutMs": 120000,    // per-request deadline (queue + switch + upstream)
    "healthCheckIntervalMs": 500,  // poll interval
    "healthCheckTimeoutMs": 30000, // give up starting an engine after this
    "coolDownMs": 3000,            // default GPU cool-down after stopping
    "vramThresholdPct": 0.15,      // VRAM must drop below 15% (0 = disable gate)
    "vramBackend": "auto",         // auto | none | nvidia-smi | rocm-smi | amd-smi | macos
    "vramCheckTimeoutMs": 30000,
    "vramCheckIntervalMs": 500,
    "childKillGraceMs": 5000,      // SIGTERM → SIGKILL grace
    "stopCommandTimeoutMs": 10000
  },
  "models": {
    "my-custom-llama": {
      "provider": "ollama",                     // models sharing a provider share one engine
      "targetBaseUrl": "http://127.0.0.1:11434",
      "targetModel": "llama3:latest",           // request `model` is rewritten to this
      "startCommand": "ollama serve",
      "stopCommand": "pkill -f 'ollama serve'",
      "coolDownMs": 3000,                       // optional per-model override
      "healthEndpoint": "/api/version"          // must return HTTP 200
    },
    "my-vllm-model": {
      "provider": "vllm",
      "targetBaseUrl": "http://127.0.0.1:8000",
      "targetModel": "Mistral-7B-Instruct-v0.2",
      "startCommand": "vllm serve mistralai/Mistral-7B-Instruct-v0.2 --port 8000",
      "stopCommand": "pkill -f 'vllm serve'",
      "coolDownMs": 5000,
      "healthEndpoint": "/health"
    },
    "my-remote-mlx": {             // engine on ANOTHER host: proxy only health-checks + forwards
      "provider": "mlx",
      "targetBaseUrl": "http://192.168.1.50:8080",
      "managed": false,
      "healthEndpoint": "/health"
    }
  }
}
```

### Running the proxy on a different host than the engines

If the proxy must run somewhere else (a GPU box runs the engines, the proxy
runs on your laptop/server, in Docker, …):

- set `"managed": false` on every model that lives on the other host — the
  proxy will **not** run `startCommand`/`stopCommand` locally, spawn no child,
  and skip the local VRAM gate; it only polls `healthEndpoint` before forwarding.
- point `targetBaseUrl` at the engine host (from Docker, use
  `host.docker.internal` + `extra_hosts: ["host.docker.internal:host-gateway"]`).
- the local VRAM figures in `/v1/stats` no longer describe the engine host —
  set `vramBackend: "none"` to make the checker a no-op, and/or use the
  per-model `metrics` scrape (below) to surface the *remote* machine's usage.

Mixing is fine: managed (local) and unmanaged (remote) models can coexist; the
single-concurrency queue and engine-switching rules apply to all of them.

### Key semantics

- **Model aliases** — the request's `model` field is matched against the keys
  of `models`. `targetModel` (optional) rewrites it to whatever the provider
  itself expects. Unknown models get an OpenAI-style `404 model_not_found`.
- **Provider = engine** — two models with the same `provider` value are served
  by the *same* engine process; switching between them does **not** restart
  anything. A provider change triggers the full stop → cooldown → VRAM gate →
  start → health-poll transition.
- **Engine children** — `startCommand` is spawned by the proxy (shell) and
  tracked: on `SIGINT`/`SIGTERM` the proxy runs `stopCommand` and terminates
  the child's whole *process group* (SIGTERM, then SIGKILL after
  `childKillGraceMs`), so grandchildren of the shell (e.g. `ollama serve`
  behind `sh -c`) are killed too. If a managed child dies on its own, the next
  request restarts it. If a model has no `startCommand`, the engine is assumed
  to be externally managed.
  If a transition to a new engine **fails** (e.g. the health endpoint never
  comes up), the half-started child is killed and the previously active engine
  is restored best-effort, so the old models keep working.
- **Remote / unmanaged models** — `"managed": false` opts a model out of local
  lifecycle management (see above); for these, only the health poll runs.
- **VRAM source** — `vramBackend` picks the checker: `"auto"` probes
  `nvidia-smi`, `rocm-smi`, `amd-smi`, and on macOS uses *unified system
  memory* (Apple Silicon GPUs draw from system RAM, capped by `iogpu.wlm`, so
  system RAM is the right VRAM surrogate); `"none"` is the no-op checker;
  any other value forces a specific backend.
- **Stats exposure** — `/v1/stats` is intentionally unauthenticated (it reports
  queue depth, provider names and GPU VRAM, but no request payloads or
  secrets). Put the whole proxy behind auth if that is not acceptable for your
  deployment.
- **Queue** — requests (including their engine transition) run through a
  strictly serial queue: one in flight, the rest FIFO. The `requestTimeoutMs`
  deadline covers the *entire* request life; a request that exceeds it gets a
  `504` and the next client moves forward.

### Environment overrides

| Variable | Effect |
|---|---|
| `INFERMUX_CONFIG` | config file path |
| `INFERMUX_HOST` / `INFERMUX_PORT` | server binding |
| `INFERMUX_API_TOKEN` | auth token (empty string disables) |
| `INFERMUX_REQUEST_TIMEOUT_MS` | request deadline |
| `INFERMUX_LOG_LEVEL` | `debug` \| `info` \| `warn` \| `error` |

CLI flags: `--config <path>`, `--log-level <level>`.

## Proxy endpoints

| Route | Auth | Description |
|---|---|---|
| `POST /v1/chat/completions` | yes* | OpenAI chat completions (stream + non-stream) |
| `POST /v1/completions` | yes* | Legacy completions (stream + non-stream) |
| `GET /v1/models` | yes* | Lists configured aliases |
| `GET /v1/stats` | no | Engine / queue / VRAM / throughput stats |
| `GET /health` | no | Proxy self-health |

\* only if `server.apiToken` is set.

### `/v1/stats`

```jsonc
{
  "activeEngine": { "provider": "vllm", "modelAlias": "my-vllm-model", "pid": 1234, "running": true, "totalSwitches": 2 },
  "queue": { "length": 1, "waiting": 1, "running": true },
  "vram": { "usedMiB": 14200, "totalMiB": 24564, "freeMiB": 10364, "pctUsed": 0.578, "source": "nvidia-smi" },
  "tokensPerSec": { "average": 24.31, "samples": [23.1, 25.6] },
  "requests": { "total": 42, "errors": 1, "lastAt": 1712345678901 }
}
```

- **VRAM** is read from an OS-agnostic backend: `nvidia-smi` (NVIDIA),
  `rocm-smi`/`amd-smi` (AMD), and *unified system memory* via `vm_stat` on
  macOS. `vramBackend: "none"` makes the checker a no-op (gating skipped);
  on a host without any backend it is a no-op automatically.
- **tokens/sec** is a moving average over recent requests: streaming responses
  are estimated from non-empty SSE deltas; non-streaming responses use the
  exact `usage.completion_tokens` when present.
- **Non-standard provider metrics**: per-model `metrics` mapping lets
  `/v1/stats` scrape and surface provider endpoints:

  ```jsonc
  "metrics": {
    "endpoint": "/metrics/vram",
    "vramUsedMiB": "used_mib",     // dot-path into the JSON response
    "vramTotalMiB": "total_mib"
  }
  ```

## Runtimes: Node.js and Bun

Runs on both: Node ≥ 18.17 and Bun ≥ 1.1.

```bash
# Node
npm run build && npm start

# Bun (skips the build step — bun runs TypeScript directly)
npm run dev:bun
```

The Hono app is served via `Bun.serve` under Bun and `@hono/node-server`
under Node; everything else (queue, engine lifecycle, VRAM) is runtime-
agnostic.

## Docker

```bash
docker compose up --build        # serves on http://localhost:8080
```

Drop your own `config.json` next to `compose.yml` to override the example
(it is bind-mounted read-only into the container). Note that engine
start/stop commands and VRAM checks run *inside* the container; for engines
on the host, use `managed: false` (see above) plus
`extra_hosts: ["host.docker.internal:host-gateway"]`.

## Development

```bash
npm run dev        # run from source via tsx (Node)
npm run dev:bun    # run from source via bun
npm test           # vitest: config, queue, engine, vram, full e2e (mock provider)
npm run smoke      # end-to-end CLI smoke test (mock provider)
npm run build      # tsc -> dist/
```

### Project layout

```
src/
  cli.ts       entrypoint: arg/env parsing, config load, signal handling
  config.ts    typed config model + JSON/JSONC/YAML loading & validation
  server.ts    Hono app wiring, logger, graceful shutdown
  proxy.ts     OpenAI-compatible routes, auth, request forwarding, SSE
  queue.ts     strict single-concurrency FIFO queue
  engine.ts    engine lifecycle: stop → cooldown → VRAM gate → start → health
  vram.ts      OS-agnostic VRAM monitoring & threshold gating
  stats.ts     stats aggregation for /v1/stats
  types.ts     shared types (config + OpenAI request shapes)
```
