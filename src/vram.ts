import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { platform, totalmem } from "node:os";

/**
 * OS-agnostic VRAM monitoring.
 *
 * Backends (auto-detected in this order): nvidia-smi, rocm-smi, amd-smi, and on
 * macOS (Apple Silicon/Intel with a GPU) the *unified system memory* via
 * `vm_stat` — on Apple Silicon the GPU draws from system RAM (capped by
 * `iogpu.wlm`, ~75% of total by default), so system memory is the right proxy.
 *
 * `VramMonitor` is configurable (`VramBackendName`); `"none"` selects the
 * no-op checker, and when auto-detection finds no backend all queries resolve
 * to `null` and callers treat VRAM gating as a no-op (logged once).
 */

/** Selectable VRAM sources. `auto` = first detected; `none` = no-op. */
export type VramBackendName = "auto" | "none" | "nvidia-smi" | "rocm-smi" | "amd-smi" | "macos";

export const VRAM_BACKENDS: readonly VramBackendName[] = [
  "auto",
  "none",
  "nvidia-smi",
  "rocm-smi",
  "amd-smi",
  "macos",
];

export interface VramInfo {
  usedMiB: number;
  totalMiB: number;
  source: string;
  /** Fraction of total (0..1). */
  pctUsed: number;
}

export interface VramSource {
  readonly name: string;
  /** Returns null when no reading is available. */
  query(): Promise<VramInfo | null>;
}

function run(cmd: string, args: string[]): Promise<{ ok: boolean; stdout: string }> {
  return new Promise((res) => {
    execFile(cmd, args, { timeout: 5_000, maxBuffer: 1 << 20 }, (err, stdout) => {
      res({ ok: !err, stdout: stdout ?? "" });
    });
  });
}

class NvidiaSmi implements VramSource {
  readonly name = "nvidia-smi";
  async query(): Promise<VramInfo | null> {
    const { ok, stdout } = await run("nvidia-smi", [
      "--query-gpu=memory.used,memory.total",
      "--format=csv,noheader,nounits",
    ]);
    if (!ok) return null;
    const first = stdout.split("\n")[0]?.trim();
    if (!first) return null;
    const [used, total] = first.split(",").map((s) => Number(s?.trim()));
    if (used === undefined || total === undefined || !Number.isFinite(used) || !Number.isFinite(total) || total <= 0)
      return null;
    return { usedMiB: used, totalMiB: total, source: this.name, pctUsed: used / total };
  }
}

/**
 * Parse `rocm-smi --csv` VRAM output. Handles two formats seen across
 * rocm-smi versions:
 *  - `GPU[0],16.0,13.2`          (total GB, free GB) — used = total - free
 *  - `GPU[0] 28.40,16.00,13.16`  (used %, total GB, free GB) — older builds
 * Labels are matched on the `GPU[n]` token so header lines are ignored.
 */
export function parseRocrsmiCsv(text: string): { usedMiB: number; totalMiB: number } | null {
  const num = (s: string): number => Number(s.trim());
  for (const line of text.split("\n")) {
    const cols = line.split(/[\s,;]+/).filter((s) => s.length > 0);
    const label = cols[0] ?? "";
    if (!/^GPU\[\d+\]$/.test(label) && !/^GPU\[\d+\]\s/.test(label)) continue;
    // Old format: the label cell may carry the used % itself (tab separated).
    const labelTail = label.match(/\s+(-?\d+(?:\.\d+)?)\s*$/);
    const vals: number[] = labelTail
      ? [Number(labelTail[1]), ...cols.slice(1).map(num)]
      : cols.slice(1).map(num);
    if (vals.length >= 3) {
      // [usedPct, totalGB, freeGB]
      const usedPct = vals[0] ?? NaN;
      const totalGb = vals[1] ?? NaN;
      const freeGb = vals[2] ?? NaN;
      if (
        Number.isFinite(usedPct) && usedPct >= 0 && usedPct <= 100 &&
        Number.isFinite(totalGb) && totalGb > 0 &&
        Number.isFinite(freeGb) && freeGb >= 0 && freeGb <= totalGb
      ) {
        const totalMiB = totalGb * 1024;
        const usedMiB = (usedPct / 100) * totalMiB;
        return { usedMiB, totalMiB };
      }
    } else if (vals.length >= 2) {
      // New format: [totalGB, freeGB]
      const totalGb = vals[0] ?? NaN;
      const freeGb = vals[1] ?? NaN;
      if (
        Number.isFinite(totalGb) && totalGb > 0 &&
        Number.isFinite(freeGb) && freeGb >= 0 && freeGb <= totalGb
      ) {
        const totalMiB = totalGb * 1024;
        const usedMiB = Math.max(0, totalGb - freeGb) * 1024;
        return { usedMiB, totalMiB };
      }
    }
  }
  return null;
}

class Rocrsmi implements VramSource {
  readonly name = "rocm-smi";
  async query(): Promise<VramInfo | null> {
    const { ok, stdout } = await run("rocm-smi", ["--showmeminfo", "vram", "--csv"]);
    if (!ok) return null;
    const parsed = parseRocrsmiCsv(stdout);
    if (!parsed) return null;
    return { ...parsed, source: this.name, pctUsed: parsed.totalMiB > 0 ? parsed.usedMiB / parsed.totalMiB : 0 };
  }
}

class AmdSmi implements VramSource {
  readonly name = "amd-smi";
  async query(): Promise<VramInfo | null> {
    const { ok, stdout } = await run("amd-smi", [
      "metrics",
      "--json",
      "--metrics",
      "gpu_vram_used_bytes,gpu_vram_total_bytes",
    ]);
    if (!ok) return null;
    return parseAmdSmiJson(stdout, this.name);
  }
}

type AmdMetric = number | string;

/**
 * Parse `amd-smi metrics --json` output. Tolerates the known shapes:
 * `"gpus": [ { metrics: {...} } ]`, a top-level `"metrics": {...}`, and metric
 * values that are numeric strings.
 */
export function parseAmdSmiJson(text: string, source: string): VramInfo | null {
  try {
    const doc = JSON.parse(text) as {
      gpus?: { metrics?: { gpu_vram_used_bytes?: AmdMetric; gpu_vram_total_bytes?: AmdMetric } }[];
      metrics?: { gpu_vram_used_bytes?: AmdMetric; gpu_vram_total_bytes?: AmdMetric };
    };
    const m = doc.gpus?.[0]?.metrics ?? doc.metrics;
    const toMiB = (v: AmdMetric | undefined): number | null => {
      const n = typeof v === "number" ? v : Number(v);
      return Number.isFinite(n) && n >= 0 ? n / (1024 * 1024) : null;
    };
    const used = toMiB(m?.gpu_vram_used_bytes);
    const total = toMiB(m?.gpu_vram_total_bytes);
    if (used === null || total === null || total <= 0) return null;
    return { usedMiB: used, totalMiB: total, source, pctUsed: used / total };
  } catch {
    return null;
  }
}

/**
 * Parse `vm_stat` output into a used/total reading. "Available" memory is
 * approximated as (free + inactive + speculative) pages, the conventional
 * Xcode Activity-Monitor-style definition; used = total - available.
 */
export function parseVmStat(
  text: string,
  totalBytes: number,
): { usedMiB: number; totalMiB: number } | null {
  if (totalBytes <= 0) return null;
  const pageSize = Number(text.match(/page size of (\d+) bytes/)?.[1] ?? NaN);
  if (!Number.isFinite(pageSize) || pageSize <= 0) return null;
  const pages = (name: string): number | null => {
    const m = text.match(new RegExp(`^${name}:\\s+([0-9]+)`, "m"));
    const n = Number(m?.[1]);
    return Number.isFinite(n) && n >= 0 ? n : null;
  };
  const free = pages("Pages free");
  const inactive = pages("Pages inactive");
  const speculative = pages("Pages speculative");
  if (free === null || inactive === null || speculative === null) return null;
  const available = (free + inactive + speculative) * pageSize;
  const used = Math.max(0, totalBytes - available);
  const totalMiB = totalBytes / (1024 * 1024);
  return { usedMiB: used / (1024 * 1024), totalMiB };
}

/**
 * macOS: unified memory proxy. Apple Silicon (and modern Intel Macs) expose no
 * separate VRAM figure — the GPU shares system RAM, capped by the `iogpu.wlm`
 * limit. We report system RAM usage as the VRAM surrogate so the threshold
 * gate still works. (Set `vramBackend: "none"` to disable entirely.)
 */
class MacosUnified implements VramSource {
  readonly name = "macos-unified";
  async query(): Promise<VramInfo | null> {
    const { ok, stdout } = await run("vm_stat", []);
    if (!ok) return null;
    const parsed = parseVmStat(stdout, totalmem());
    if (!parsed) return null;
    return { ...parsed, source: this.name, pctUsed: parsed.usedMiB / parsed.totalMiB };
  }
}

function detectBackend(): VramSource | null {
  if (
    existsSync("/usr/bin/nvidia-smi") ||
    existsSync("/usr/local/bin/nvidia-smi") ||
    existsSync("/usr/lib/nvidia/bin/nvidia-smi")
  )
    return new NvidiaSmi();
  if (existsSync("/usr/bin/rocm-smi") || existsSync("/opt/rocm/bin/rocm-smi"))
    return new Rocrsmi();
  if (existsSync("/usr/bin/amd-smi") || existsSync("/usr/local/bin/amd-smi")) return new AmdSmi();
  // Apple Silicon: GPU uses unified memory — track system RAM as the proxy.
  if (platform() === "darwin") return new MacosUnified();
  return null;
}

function makeBackend(name: VramBackendName): VramSource | null {
  if (name === "none") return null;
  if (name === "auto") return detectBackend();
  switch (name) {
    case "nvidia-smi":
      return new NvidiaSmi();
    case "rocm-smi":
      return new Rocrsmi();
    case "amd-smi":
      return new AmdSmi();
    case "macos":
      return platform() === "darwin" ? new MacosUnified() : null;
  }
}

export class VramMonitor {
  private backend: VramSource | null | undefined;
  private log: (msg: string) => void;
  private warned = false;
  private readonly backendName: VramBackendName;

  /**
   * @param log destination for one-time warnings
   * @param backend `"auto"` (default) probes all known sources; `"none"` is the
   *   no-op checker (VRAM gating is skipped); anything else forces a specific one.
   */
  constructor(log: (msg: string) => void = () => {}, backend: VramBackendName = "auto") {
    this.log = log;
    this.backendName = backend;
  }

  private resolveBackend(): VramSource | null {
    if (this.backend === undefined) this.backend = makeBackend(this.backendName);
    return this.backend;
  }

  /** Latest VRAM reading, or null if no backend/reading available. */
  async query(): Promise<VramInfo | null> {
    const backend = this.resolveBackend();
    if (!backend) {
      if (!this.warned) {
        this.warned = true;
        this.log(
          this.backendName === "none"
            ? "vram: disabled via vramBackend=none; VRAM gating is a no-op"
            : "vram: no GPU metrics backend detected (nvidia-smi/rocm-smi/amd-smi/macos); VRAM gating is a no-op",
        );
      }
      return null;
    }
    return backend.query();
  }

  /**
   * Poll until VRAM usage drops below `thresholdPct` (fraction of total), or the
   * timeout elapses. Resolves true if the gate passed (or VRAM is unavailable,
   * in which case we optimistically proceed).
   */
  async waitBelowThreshold(
    thresholdPct: number,
    timeoutMs: number,
    intervalMs: number,
  ): Promise<boolean> {
    if (thresholdPct <= 0) return true;
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const info = await this.query();
      if (info === null) return true; // no backend -> proceed
      if (info.pctUsed < thresholdPct) return true;
      if (Date.now() >= deadline) return false;
      await sleep(Math.min(intervalMs, Math.max(50, deadline - Date.now())));
    }
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
