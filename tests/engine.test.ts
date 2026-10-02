import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createServer, type Server } from "node:http";
import { mkdtempSync, readFileSync, existsSync, unlinkSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EngineManager, EngineError } from "../src/engine.js";
import { VramMonitor } from "../src/vram.js";
import type { ModelConfig, RequiredDefaults } from "../src/types.js";

const tmp = mkdtempSync(join(tmpdir(), "infermux-engine-"));
const files = {
  bStart: join(tmp, "b-start"),
  bStop: join(tmp, "b-stop"),
  aStart: join(tmp, "a-start"),
  aStop: join(tmp, "a-stop"),
  grandPid: join(tmp, "grandchild.pid"),
};
for (const f of Object.values(files)) {
  try {
    unlinkSync(f);
  } catch {
    /* first run */
  }
}
const count = (f: string): number => (existsSync(f) ? readFileSync(f, "utf8").split("\n").filter(Boolean).length : 0);

afterAll(() => rmSync(tmp, { recursive: true, force: true }));

function makeManager(): EngineManager {
  const defaults: RequiredDefaults = {
    requestTimeoutMs: 10_000,
    healthCheckIntervalMs: 20,
    healthCheckTimeoutMs: 1_000,
    coolDownMs: 1,
    vramThresholdPct: 0, // gate off: no GPU in the sandbox
    vramCheckTimeoutMs: 200,
    vramCheckIntervalMs: 50,
    childKillGraceMs: 500,
    stopCommandTimeoutMs: 1_000,
    vramBackend: "none", // no-op VRAM checker in tests
  };
  return new EngineManager(defaults, new VramMonitor(() => {}, "none"), () => {});
}

describe("EngineManager", () => {
  let healthServer: Server;
  let healthBase: string;

  beforeAll(async () => {
    healthServer = createServer((req, res) => {
      const ok = req.url === "/ok";
      res.writeHead(ok ? 200 : 500);
      res.end(ok ? "ok" : "bad");
    });
    await new Promise<void>((r) => healthServer.listen(0, "127.0.0.1", r));
    healthBase = `http://127.0.0.1:${(healthServer.address() as { port: number }).port}`;
  });

  afterAll(async () => {
    await new Promise<void>((r) => healthServer.closeAllConnections?.() ?? healthServer.close(() => r()));
  });

  const idleCmd = `node -e "setInterval(()=>{},1e3)"`;

  it("runs a single transition when two concurrent ensureReady calls target the same new provider", async () => {
    const mgr = makeManager();
    // `a` is externally managed (no startCommand, no health probe).
    const a: ModelConfig = { provider: "a", targetBaseUrl: "http://127.0.0.1:1" };
    const b: ModelConfig = {
      provider: "b",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      startCommand: `sh -c 'echo s >> ${files.bStart}; exec ${idleCmd}'`,
      stopCommand: `echo t >> ${files.bStop}`,
    };
    await mgr.ensureReady("a-alias", a);
    expect(mgr.state().provider).toBe("a");
    const switchesBefore = mgr.state().switches;

    // Same new provider, two requests at once.
    await Promise.all([mgr.ensureReady("b1", b), mgr.ensureReady("b2", b)]);

    // `sh -c 'echo ...'` writes are async w.r.t. the health poll; give it a
    // grace period, then assert the *exact* count (this is the regression check
    // against a double start).
    for (let i = 0; i < 50 && count(files.bStart) < 1; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(count(files.bStart)).toBe(1); // started exactly once
    // `a` has no stopCommand, so nothing ran during the a->b transition.
    expect(count(files.bStop)).toBe(0);
    expect(mgr.state().switches).toBe(switchesBefore + 1); // no double transition
    expect(mgr.state().provider).toBe("b");
    await mgr.shutdown();
    // Shutdown ran b's stopCommand exactly once (no stale pairing, no re-run).
    expect(count(files.bStop)).toBe(1);
  });

  it("rolls back provider/alias and keeps the previous stopCommand paired after a failed transition", async () => {
    const mgr = makeManager();
    const a: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      stopCommand: `echo t >> ${files.aStop}`, // externally managed engine, but with a stopCommand
    };
    const b: ModelConfig = {
      provider: "b",
      targetBaseUrl: healthBase,
      healthEndpoint: "/bad", // never healthy -> transition must fail
      startCommand: idleCmd,
      // deliberately no stopCommand: a stale pairing from `a` must not leak.
    };
    const c: ModelConfig = { provider: "c", targetBaseUrl: healthBase, healthEndpoint: "/ok" };

    await mgr.ensureReady("a-alias", a);
    await expect(mgr.ensureReady("b", b)).rejects.toBeInstanceOf(EngineError);
    expect(mgr.state().provider).toBe("a");
    expect(mgr.state().modelAlias).toBe("a-alias");

    // a -> c: the active engine is `a` (externally managed) whose stopCommand
    // must be the *a* one again — it ran once for a->b and once for a->c.
    // (Pre-fix, the failed b left a null pairing and a's stopCommand was lost.)
    await mgr.ensureReady("c-alias", c);
    expect(count(files.aStop)).toBe(2);

    // Cleanup: `c` is externally managed, nothing to kill.
  });

  it(
    "kills the whole process group of a detached shell child (no orphaned grandchildren)",
    async () => {
    const mgr = makeManager();
    const a: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      startCommand: `sh -c 'sleep 30 & echo $! > ${files.grandPid}; wait'`,
    };
    await mgr.ensureReady("a-alias", a);
    // Wait for the grandchild pid to be recorded.
    for (let i = 0; i < 50 && !existsSync(files.grandPid); i++) {
      await new Promise((r) => setTimeout(r, 50));
    }
    const grandPid = Number(readFileSync(files.grandPid, "utf8").trim());
    expect(grandPid).toBeGreaterThan(0);
    expect(processIsAlive(grandPid)).toBe(true);

    await mgr.shutdown();
    // The shell exits; the group kill must have taken `sleep` down with it.
    for (let i = 0; i < 50 && processIsAlive(grandPid); i++) {
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(processIsAlive(grandPid)).toBe(false);
  }, 15_000);

  it("restores the previous engine when the new one fails its health check", async () => {
    const mgr = makeManager();
    const a: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      startCommand: `sh -c 'echo s >> ${files.aStart}; exec ${idleCmd}'`,
    };
    const b: ModelConfig = {
      provider: "b",
      targetBaseUrl: healthBase,
      healthEndpoint: "/bad",
      startCommand: idleCmd,
    };
    await mgr.ensureReady("a-alias", a);
    // The `sh -c 'echo ...'` startCommand writes its marker asynchronously;
    // allow a grace period before asserting exact counts.
    for (let i = 0; i < 100 && count(files.aStart) < 1; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(count(files.aStart)).toBe(1);

    await expect(mgr.ensureReady("b", b)).rejects.toBeInstanceOf(EngineError);

    // The failed `b` child must be reaped, and `a` must be back up.
    for (let i = 0; i < 50 && count(files.aStart) < 2; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    const st = mgr.state();
    expect(st.provider).toBe("a");
    expect(st.modelAlias).toBe("a-alias");
    expect(st.childAlive).toBe(true);
    expect(count(files.aStart)).toBe(2); // restored via its startCommand
    await mgr.shutdown();
  });

  it("treats managed:false as remote: no local start/stop, health poll only", async () => {
    // Dedicated marker files: `aStart`/`aStop` accumulate across tests.
    const a2Start = join(tmp, "a2-start");
    const a2Stop = join(tmp, "a2-stop");
    const remoteStart = join(tmp, "remote-start");
    const remoteStop = join(tmp, "remote-stop");
    for (const f of [a2Start, a2Stop, remoteStart, remoteStop]) {
      try {
        unlinkSync(f);
      } catch {
        /* first run */
      }
    }
    const mgr = makeManager();
    const a: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      startCommand: `sh -c 'echo s >> ${a2Start}; exec node -e "setInterval(()=>{},1e3)"'`,
      stopCommand: `echo t >> ${a2Stop}`,
    };
    const remote: ModelConfig = {
      provider: "remote",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      managed: false,
      // These must NEVER execute on the proxy host.
      startCommand: `echo start >> ${remoteStart}`,
      stopCommand: `echo t >> ${remoteStop}`,
    };

    await mgr.ensureReady("a-alias", a);
    for (let i = 0; i < 100 && count(a2Start) < 1; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(count(a2Start)).toBe(1);

    // a -> remote: the *managed* engine `a` must be stopped properly…
    await mgr.ensureReady("remote-alias", remote);
    const st = mgr.state();
    expect(st.provider).toBe("remote");
    expect(st.modelAlias).toBe("remote-alias");
    expect(st.childAlive).toBe(false);
    expect(st.pid).toBeNull();
    // …its stopCommand ran once, and the remote model's commands never did.
    expect(count(a2Stop)).toBe(1);
    expect(count(remoteStart)).toBe(0);
    expect(count(remoteStop)).toBe(0);

    // shutdown must not run the remote stopCommand either.
    await mgr.shutdown();
    expect(count(a2Stop)).toBe(1);
    expect(count(remoteStop)).toBe(0);
  });

  it("rejects new work after shutdown", async () => {
    const mgr = makeManager();
    const a: ModelConfig = { provider: "a", targetBaseUrl: healthBase, healthEndpoint: "/ok" };
    await mgr.shutdown();
    await expect(mgr.ensureReady("a-alias", a)).rejects.toThrow(/shutting down/);
  });
});

/**
 * liveness check that treats zombies as dead: `process.kill(pid, 0)` succeeds
 * against zombies, which would make a properly killed (but un-reaped in this
 * container) process look alive.
 */
function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
  } catch {
    return false;
  }
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
    return !/\) [Zz] /.test(stat);
  } catch {
    return false; // no /proc (non-Linux): fall back to signal check
  }
}
