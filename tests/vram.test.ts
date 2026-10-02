import { describe, expect, it } from "vitest";
import { parseRocrsmiCsv, parseAmdSmiJson, parseVmStat, VramMonitor } from "../src/vram.js";

describe("parseVmStat", () => {
  const sample = [
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
    "Pages free:                          100000.",
    "Pages active:                         200000.",
    "Pages inactive:                       150000.",
    "Pages speculative:                     10000.",
    "Pages wired down:                      50000.",
  ].join("\n");

  it("computes used = total - (free + inactive + speculative) * pageSize", () => {
    const total = 32 * 1024 * 1024 * 1024; // 32 GiB
    const out = parseVmStat(sample, total);
    expect(out).not.toBeNull();
    const available = (100_000 + 150_000 + 10_000) * 16_384;
    expect(out!.totalMiB).toBeCloseTo(total / (1024 * 1024), 0);
    expect(out!.usedMiB).toBeCloseTo((total - available) / (1024 * 1024), 0);
    expect(out!.usedMiB).toBeLessThan(out!.totalMiB);
  });

  it("clamps at 0 when available exceeds total and rejects bad input", () => {
    const out = parseVmStat(
      ["Mach Virtual Memory Statistics: (page size of 16384 bytes)",
        "Pages free: 99999999.", "Pages inactive: 999999.", "Pages speculative: 1."].join("\n"),
      4 * 1024 * 1024 * 1024,
    );
    expect(out?.usedMiB).toBe(0);
    expect(parseVmStat("no page size here", 4096)).toBeNull();
    expect(parseVmStat(sample, 0)).toBeNull();
    expect(parseVmStat("page size of 16384 bytes\nPages free: 1.", 4096)).toBeNull(); // missing inactive/speculative
  });
});

describe("VramMonitor backend selection", () => {
  it("'none' is the no-op checker: null readings, gate passes", async () => {
    const warnings: string[] = [];
    const m = new VramMonitor((msg) => warnings.push(msg), "none");
    expect(await m.query()).toBeNull();
    expect(await m.query()).toBeNull(); // warned only once
    expect(warnings.filter((w) => w.includes("vramBackend=none"))).toHaveLength(1);
    const t0 = Date.now();
    expect(await m.waitBelowThreshold(1, 60_000, 5_000)).toBe(true);
    expect(Date.now() - t0).toBeLessThan(1_000);
  });

  it("forced 'macos' backend is a no-op off-mac (query resolves null)", async () => {
    const m = new VramMonitor(() => {}, "macos");
    if (process.platform !== "darwin") {
      expect(await m.query()).toBeNull();
    }
  });
});

describe("parseRocrsmiCsv", () => {
  it("parses the modern `GPU[n],totalGB,freeGB` shape", () => {
    const out = parseRocrsmiCsv(
      [
        "GPU,Total Memory (GB),Free Memory (GB)",
        "GPU[0],16.0,13.2",
      ].join("\n"),
    );
    expect(out).not.toBeNull();
    expect(out!.totalMiB).toBeCloseTo(16 * 1024, 0);
    expect(out!.usedMiB).toBeCloseTo((16.0 - 13.2) * 1024, 0);
  });

  it("parses the older `GPU[n] usedPct, totalGB, freeGB` shape (tab separated)", () => {
    const out = parseRocrsmiCsv("GPU[0]\t28.40, 16.00, 13.16");
    expect(out).not.toBeNull();
    expect(out!.totalMiB).toBeCloseTo(16 * 1024, 0);
    expect(out!.usedMiB).toBeCloseTo(0.284 * 16 * 1024, 0);
  });

  it("parses the space-separated label variant and ignores header lines", () => {
    const out = parseRocrsmiCsv(
      ["GPU use (%), GPU total (GB), GPU free (GB)", "GPU[0] 28.40 16.00 13.16"].join("\n"),
    );
    expect(out).not.toBeNull();
    expect(out!.usedMiB).toBeCloseTo(0.284 * 16 * 1024, 0);
  });

  it("returns null for unparseable output and rejects free > total", () => {
    expect(parseRocrsmiCsv("")).toBeNull();
    expect(parseRocrsmiCsv("GPU[0],13.0,20.0")).toBeNull(); // free > total: nonsense
    expect(parseRocrsmiCsv("GPU[0]\t150.0, 16.0, 13.16")).toBeNull(); // pct > 100
  });
});

describe("parseAmdSmiJson", () => {
  it("parses the gpus[0].metrics shape", () => {
    const doc = JSON.stringify({
      gpus: [{ index: 0, metrics: { gpu_vram_used_bytes: 1024 * 1024, gpu_vram_total_bytes: 8 * 1024 * 1024 } }],
    });
    const info = parseAmdSmiJson(doc, "amd-smi");
    expect(info?.usedMiB).toBe(1);
    expect(info?.totalMiB).toBe(8);
    expect(info?.pctUsed).toBeCloseTo(0.125, 5);
  });

  it("tolerates top-level metrics and numeric strings", () => {
    const doc = JSON.stringify({ metrics: { gpu_vram_used_bytes: "1048576", gpu_vram_total_bytes: "8388608" } });
    const info = parseAmdSmiJson(doc, "amd-smi");
    expect(info?.usedMiB).toBe(1);
  });

  it("returns null on garbage", () => {
    expect(parseAmdSmiJson("not json", "amd-smi")).toBeNull();
    expect(parseAmdSmiJson(JSON.stringify({ gpus: [] }), "amd-smi")).toBeNull();
  });
});

describe("VramMonitor.waitBelowThreshold", () => {
  it("returns true immediately when the threshold is 0 (gate disabled)", async () => {
    const t0 = Date.now();
    const ok = await new VramMonitor(() => {}).waitBelowThreshold(0, 60_000, 500);
    expect(ok).toBe(true);
    expect(Date.now() - t0).toBeLessThan(5_000);
  });

  it("stops polling at the deadline even when the interval is larger than the remaining time", async () => {
    const t0 = Date.now();
    // threshold 1.0 (never satisfied if a backend exists), interval 60s, deadline 400ms.
    // Must not sleep for the full 60s interval: a single bounded overshoot is fine.
    // (ok is true when no GPU backend exists — the gate is a no-op; the point
    // of this test is that it never sleeps the full 60s interval.)
    await new VramMonitor(() => {}).waitBelowThreshold(1, 400, 60_000);
    expect(Date.now() - t0).toBeLessThan(5_000);
  });
});
