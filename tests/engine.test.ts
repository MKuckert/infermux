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
    // `a` is externally managed (no command/commands, no health probe).
    const a: ModelConfig = { provider: "a", targetBaseUrl: "http://127.0.0.1:1" };
    const b: ModelConfig = {
      provider: "b",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      // commands mode: fire-and-forget launcher (exits immediately) + stop command.
      commands: {
        start: `sh -c 'echo s >> ${files.bStart}'`,
        stop: `echo t >> ${files.bStop}`,
      },
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
    // `a` has no stop command, so nothing ran during the a->b transition.
    expect(count(files.bStop)).toBe(0);
    expect(mgr.state().switches).toBe(switchesBefore + 1); // no double transition
    expect(mgr.state().provider).toBe("b");
    await mgr.shutdown();
    // Shutdown ran b's stop command exactly once (no stale pairing, no re-run).
    expect(count(files.bStop)).toBe(1);
  });

  it("rolls back provider/alias and keeps the previous stop command paired after a failed transition", async () => {
    const mgr = makeManager();
    // Dedicated marker files: the shared ones accumulate across tests.
    const aStart = join(tmp, "rb-a-start");
    const aStop = join(tmp, "rb-a-stop");
    const bStart = join(tmp, "rb-b-start");
    const bStop = join(tmp, "rb-b-stop");
    for (const f of [aStart, aStop, bStart, bStop]) {
      try {
        unlinkSync(f);
      } catch {
        /* first run */
      }
    }
    const a: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      commands: {
        start: `sh -c 'echo s >> ${aStart}'`,
        stop: `echo t >> ${aStop}`,
      },
    };
    const b: ModelConfig = {
      provider: "b",
      targetBaseUrl: healthBase,
      healthEndpoint: "/bad", // never healthy -> transition must fail
      commands: {
        start: `sh -c 'echo s >> ${bStart}'`,
        stop: `echo t >> ${bStop}`,
      },
    };
    const c: ModelConfig = { provider: "c", targetBaseUrl: healthBase, healthEndpoint: "/ok" };

    await mgr.ensureReady("a-alias", a);
    // a -> b: a's stop runs, b's start fires, health fails; the failed b is
    // torn down via *its* stop command and a is restored (re-started).
    await expect(mgr.ensureReady("b", b)).rejects.toBeInstanceOf(EngineError);
    expect(mgr.state().provider).toBe("a");
    expect(mgr.state().modelAlias).toBe("a-alias");

    // a -> c: the active engine is `a` again — its stop command must be the
    // *a* one (it ran once for a->b and once for a->c), and b's stop command
    // must not have leaked into the a->c stop (it ran exactly once: during
    // b's own failed transition).
    await mgr.ensureReady("c-alias", c);
    expect(count(aStop)).toBe(2);
    expect(count(bStop)).toBe(1);

    // Cleanup: `c` is externally managed, nothing to stop.
  });

  it(
    "kills the whole process group of a detached shell child (no orphaned grandchildren)",
    async () => {
    const mgr = makeManager();
    const a: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      command: `sh -c 'sleep 30 & echo $! > ${files.grandPid}; wait'`,
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
    // Dedicated marker file: `files.aStart` is written by earlier tests.
    const aStart = join(tmp, "rs-a-start");
    try {
      unlinkSync(aStart);
    } catch {
      /* first run */
    }
    const a: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      command: `sh -c 'echo s >> ${aStart}; exec ${idleCmd}'`,
    };
    const b: ModelConfig = {
      provider: "b",
      targetBaseUrl: healthBase,
      healthEndpoint: "/bad",
      command: idleCmd,
    };
    await mgr.ensureReady("a-alias", a);
    // The `sh -c 'echo ...'` start writes its marker asynchronously; allow a
    // grace period before asserting exact counts.
    for (let i = 0; i < 100 && count(aStart) < 1; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(count(aStart)).toBe(1);

    await expect(mgr.ensureReady("b", b)).rejects.toBeInstanceOf(EngineError);

    // The failed `b` child must be reaped, and `a` must be back up.
    for (let i = 0; i < 50 && count(aStart) < 2; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    const st = mgr.state();
    expect(st.provider).toBe("a");
    expect(st.modelAlias).toBe("a-alias");
    expect(st.running).toBe(true);
    expect(count(aStart)).toBe(2); // restored via its command
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
      commands: {
        start: `sh -c 'echo s >> ${a2Start}'`,
        stop: `echo t >> ${a2Stop}`,
      },
    };
    const remote: ModelConfig = {
      provider: "remote",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      managed: false,
      // These must NEVER execute on the proxy host.
      commands: { start: `echo start >> ${remoteStart}`, stop: `echo t >> ${remoteStop}` },
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
    expect(st.running).toBe(false);
    expect(st.pid).toBeNull();
    // …its stop command ran once, and the remote model's commands never did.
    expect(count(a2Stop)).toBe(1);
    expect(count(remoteStart)).toBe(0);
    expect(count(remoteStop)).toBe(0);

    // shutdown must not run the remote stop command either.
    await mgr.shutdown();
    expect(count(a2Stop)).toBe(1);
    expect(count(remoteStop)).toBe(0);
  });

  it("commands mode: no process handle, stop command actually stops the launched server", async () => {
    const srvPid = join(tmp, "cmd-server.pid");
    const stopMarker = join(tmp, "cmd-stop");
    for (const f of [srvPid, stopMarker]) {
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
      // The launcher starts a *separate* long-running server process and exits;
      // only the stop command can stop it.
      commands: {
        start: `sh -c '${idleCmd} & echo $! > ${srvPid}'`,
        stop: `sh -c 'kill $(cat ${srvPid}) 2>/dev/null; echo t >> ${stopMarker}'`,
      },
    };
    const b: ModelConfig = { provider: "b", targetBaseUrl: healthBase, healthEndpoint: "/ok" };

    await mgr.ensureReady("a-alias", a);
    // The engine holds no process handle: running=true but pid=null.
    const st = mgr.state();
    expect(st.provider).toBe("a");
    expect(st.running).toBe(true);
    expect(st.pid).toBeNull();

    // Wait for the launched server pid to be recorded and alive.
    for (let i = 0; i < 100 && !existsSync(srvPid); i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    const pid = Number(readFileSync(srvPid, "utf8").trim());
    expect(pid).toBeGreaterThan(0);
    expect(processIsAlive(pid)).toBe(true);

    // Switching away runs the stop command, which kills the launched server.
    await mgr.ensureReady("b-alias", b);
    expect(count(stopMarker)).toBe(1);
    for (let i = 0; i < 50 && processIsAlive(pid); i++) {
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(processIsAlive(pid)).toBe(false);
    await mgr.shutdown();
  }, 15_000);

  it("command mode: restarts the engine when its child dies on its own", async () => {
    const marker = join(tmp, "restart-marker");
    try {
      unlinkSync(marker);
    } catch {
      /* first run */
    }
    const mgr = makeManager();
    const a: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      // Exits by itself after a short while.
      command: `sh -c 'echo s >> ${marker}; exec node -e "setTimeout(()=>process.exit(0),300)"'`,
    };

    await mgr.ensureReady("a-alias", a);
    for (let i = 0; i < 100 && count(marker) < 1; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(count(marker)).toBe(1);

    // The child dies on its own; the engine reports not running.
    for (let i = 0; i < 100 && mgr.state().running; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(mgr.state().running).toBe(false);

    // The next request for the same provider restarts it.
    await mgr.ensureReady("a-alias", a);
    for (let i = 0; i < 100 && count(marker) < 2; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(count(marker)).toBe(2);
    await mgr.shutdown();
  }, 15_000);

  it("command mode: a failed restart is torn down and retried on the next request", async () => {
    const marker = join(tmp, "restart-fail-marker");
    try {
      unlinkSync(marker);
    } catch {
      /* first run */
    }
    const mgr = makeManager();
    // Same provider, two aliases: one healthy endpoint, one that never is.
    const ok: ModelConfig = {
      provider: "a",
      targetBaseUrl: healthBase,
      healthEndpoint: "/ok",
      command: `sh -c 'echo s >> ${marker}; exec node -e "setTimeout(()=>process.exit(0),300)"'`,
    };
    const bad: ModelConfig = { ...ok, healthEndpoint: "/bad" };

    await mgr.ensureReady("a-ok", ok);
    // Wait for the child to die on its own.
    for (let i = 0; i < 100 && mgr.state().running; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(mgr.state().running).toBe(false);

    // Restart with an unhealthy endpoint: must fail, tear the half-started
    // child down, and leave the provider active-but-stopped so a retry is
    // possible (not stuck behind a stale running flag).
    await expect(mgr.ensureReady("a-bad", bad)).rejects.toBeInstanceOf(EngineError);
    expect(mgr.state().provider).toBe("a");
    expect(mgr.state().running).toBe(false);

    // The next request retries and succeeds.
    await mgr.ensureReady("a-ok", ok);
    expect(mgr.state().running).toBe(true);
    await mgr.shutdown();
  }, 15_000);

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
