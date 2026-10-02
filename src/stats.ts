import type { EngineManager } from "./engine.js";
import type { SerialQueue } from "./queue.js";
import type { VramInfo } from "./vram.js";

/**
 * Lightweight stats aggregation for the /v1/stats endpoint.
 *
 * Tokens/sec is estimated from streaming responses: each SSE chunk carrying a
 * non-empty content delta is counted as ~1 token (a cheap, standard proxy
 * estimate), measured against elapsed time. Completed streaming requests are
 * folded into a moving average over the last N requests; non-streaming
 * responses with a `usage` field contribute exact counts.
 */
export class Stats {
  private samples: number[] = [];
  private lastVram: VramInfo | null = null;
  private requestsTotal = 0;
  private errorsTotal = 0;
  private lastRequestAt: number | null = null;
  private providerMetrics: Record<string, Record<string, number | string>> = {};

  constructor(
    private engine: EngineManager,
    private queue: SerialQueue,
    private maxSamples = 10,
  ) {}

  recordRequestStart(): void {
    this.requestsTotal++;
    this.lastRequestAt = Date.now();
  }

  recordError(): void {
    this.errorsTotal++;
  }

  /** Fold one completed request into the tokens/sec moving average. */
  recordCompletedRequest(tokens: number, elapsedMs: number): void {
    if (elapsedMs > 0 && tokens > 0) {
      const tps = tokens / (elapsedMs / 1000);
      this.samples.push(tps);
      if (this.samples.length > this.maxSamples) this.samples.shift();
    }
  }

  setVram(vram: VramInfo | null): void {
    this.lastVram = vram;
  }

  /** Store non-standard provider metrics scraped from a configured mapping. */
  setProviderMetrics(provider: string, metrics: Record<string, number | string>): void {
    this.providerMetrics[provider] = metrics;
  }

  get avgTokensPerSec(): number | null {
    if (this.samples.length === 0) return null;
    const sum = this.samples.reduce((a, b) => a + b, 0);
    return Math.round((sum / this.samples.length) * 100) / 100;
  }

  snapshot(): Record<string, unknown> {
    const eng = this.engine.state();
    const vram = this.lastVram;
    return {
      activeEngine: {
        provider: eng.provider,
        modelAlias: eng.modelAlias,
        pid: eng.pid,
        running: eng.childAlive,
        totalSwitches: eng.switches,
      },
      queue: {
        length: this.queue.length,
        waiting: this.queue.waiting,
        running: this.queue.running,
      },
      vram: vram
        ? {
            usedMiB: Math.round(vram.usedMiB),
            totalMiB: Math.round(vram.totalMiB),
            freeMiB: Math.round(vram.totalMiB - vram.usedMiB),
            pctUsed: Math.round(vram.pctUsed * 1000) / 1000,
            source: vram.source,
          }
        : { available: false },
      tokensPerSec: {
        average: this.avgTokensPerSec,
        samples: this.samples,
      },
      requests: {
        total: this.requestsTotal,
        errors: this.errorsTotal,
        lastAt: this.lastRequestAt,
      },
      providerMetrics: this.providerMetrics,
    };
  }
}
