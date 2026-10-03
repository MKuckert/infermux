import { spawn, exec, type ChildProcess } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { promisify } from "node:util";
import type { ModelConfig, RequiredDefaults } from "./types.js";
import { sleep, type VramMonitor } from "./vram.js";

const execAsync = promisify(exec);

export type EngineLog = (msg: string) => void;

/** Error representing a failed engine transition (HTTP 503 upstream of the request). */
export class EngineError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EngineError";
  }
}

export interface EngineState {
  provider: string | null;
  modelAlias: string | null;
  pid: number | null;
  childAlive: boolean;
  /** Monotonic counter incremented on every stop/start transition. */
  switches: number;
}

/**
 * Manages at most one local inference engine at a time.
 *
 * `ensureReady(alias, cfg)` is the single entry point: if the requested
 * model's provider differs from the active one, it runs the full transition:
 *   stop old engine -> cooldown -> VRAM gate -> start new engine -> health poll.
 * The whole operation is serialized with an internal lock so concurrent
 * requests can never interleave two transitions.
 */
export class EngineManager {
  activeProvider: string | null = null;
  activeModelAlias: string | null = null;
  private child: ChildProcess | null = null;
  private childAlive = false;
  /** True once we have spawned a child for the active engine (false when the engine is externally managed). */
  private childSpawned = false;
  private switches = 0;
  private lock: Promise<unknown> = Promise.resolve();
  /** Config of the last *successful* start; used to restore the previous engine after a failed transition. */
  private activeCfg: ModelConfig | null = null;
  /** Stop command paired with the active engine (null when externally managed). */
  private stopCommandForActive: string | null = null;
  /** Latches on shutdown; subsequent ensureReady/transition calls fail fast. */
  private shuttingDown = false;

  constructor(
    private defaults: RequiredDefaults,
    private vram: VramMonitor,
    private log: EngineLog = () => {},
  ) {}

  state(): EngineState {
    return {
      provider: this.activeProvider,
      modelAlias: this.activeModelAlias,
      pid: this.child?.pid ?? null,
      childAlive: this.childAlive,
      switches: this.switches,
    };
  }

  /** Run a critical section, serialized with all other transitions. */
  private withLock<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.lock.then(fn, fn);
    this.lock = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  async ensureReady(alias: string, cfg: ModelConfig): Promise<void> {
    if (this.shuttingDown) throw new EngineError("infermux is shutting down");
    if (this.activeProvider === cfg.provider) {
      // If the engine we spawned died on its own, bring it back (only for
      // engines we actually manage on this host).
      if (cfg.managed !== false && this.childSpawned && !this.childAlive && cfg.startCommand) {
        this.log(`engine: ${cfg.provider} child not running; restarting`);
        await this.withLock(async () => {
          // Re-check under the lock: a concurrent transition may have changed
          // the active provider or already restarted the engine.
          if (
            this.activeProvider === cfg.provider &&
            cfg.managed !== false &&
            !this.childAlive &&
            cfg.startCommand
          ) {
            await this.startActive(alias, cfg);
          }
        });
      }
      return;
    }
    await this.withLock(async () => {
      if (this.shuttingDown) throw new EngineError("infermux is shutting down");
      // A concurrent transition for the same provider may have completed while we
      // waited for the lock (the fast path saw the *old* provider). Don't run a
      // redundant stop->start for an engine that is already active.
      if (this.activeProvider === cfg.provider) {
        this.log(`engine: ${cfg.provider} already active; skipping redundant transition`);
        return;
      }
      await this.transition(alias, cfg);
    });
  }

  private async transition(alias: string, cfg: ModelConfig): Promise<void> {
    const d = this.defaults;
    const from = this.activeProvider;
    const fromAlias = this.activeModelAlias;
    const fromCfg = this.activeCfg;
    const fromStopCmd = this.stopCommandForActive;
    this.log(`engine: preparing switch ${from ?? "(none)"} -> ${cfg.provider} (model "${alias}")`);
    const remote = cfg.managed === false;
    try {
      await this.stopActive();
      if (remote) {
        // The engine runs on another host: no local cool-down, no local VRAM
        // gate, no startCommand — just verify it is reachable/healthy.
        this.log(`engine: ${cfg.provider} is remote/externally managed; skipping local lifecycle steps`);
      } else {
        const coolDown = cfg.coolDownMs ?? d.coolDownMs;
        if (coolDown > 0) {
          this.log(`engine: cooling down ${coolDown}ms (GPU driver cleanup)`);
          await sleep(coolDown);
        }
        const vramOk = await this.vram.waitBelowThreshold(
          cfg.vramThresholdPct ?? d.vramThresholdPct,
          d.vramCheckTimeoutMs,
          d.vramCheckIntervalMs,
        );
        if (!vramOk) {
          this.log("engine: warning — VRAM did not drop below threshold in time; starting anyway");
        }
      }
      await this.startActive(alias, cfg);
      this.activeCfg = cfg;
      this.switches++;
    } catch (err) {
      // The new engine half-started: reap its child so it doesn't keep
      // holding VRAM while we roll back to the previous engine.
      await this.killChild();
      // Best effort: restore the previously active engine so it stays usable.
      let restored = false;
      if (from !== null && fromCfg?.startCommand) {
        this.log(`engine: new engine failed; restoring previous "${from}" engine`);
        try {
          await this.startActive(fromAlias ?? "", fromCfg);
          this.activeCfg = fromCfg;
          restored = true;
        } catch (restoreErr) {
          this.log(`engine: restoring previous "${from}" engine also failed: ${errMsg(restoreErr)}`);
        }
      }
      if (!restored) {
        // Roll back the active pointer so we don't claim a half-started engine.
        // (The alias goes back to the *previous* alias, not the failed one.)
        this.activeProvider = from;
        this.activeModelAlias = fromAlias;
        this.activeCfg = fromCfg;
        this.stopCommandForActive = fromStopCmd;
      }
      throw err;
    }
  }

  /** Stop the currently active engine (stopCommand + child signal). */
  private async stopActive(): Promise<void> {
    if (!this.activeProvider) return;
    if (this.stopCommandForActive) {
      this.log(`engine: running stopCommand: ${this.stopCommandForActive}`);
      try {
        await withTimeout(
          execAsync(this.stopCommandForActive),
          this.defaults.stopCommandTimeoutMs,
          "stopCommand",
        );
      } catch (err) {
        this.log(`engine: stopCommand failed (continuing): ${errMsg(err)}`);
      }
    }
    await this.killChild();
    // The active engine is gone; clear its stop command so it can't run again
    // against a *different* provider that later becomes active.
    this.stopCommandForActive = null;
  }

  /**
   * Kill the child's whole process group (it was spawned detached, so it is its
   * own group leader). `child.kill()` alone only kills the `sh -c` wrapper and
   * orphans grandchildren (e.g. `ollama serve`).
   */
  private killChildGroup(pid: number, signal: NodeJS.Signals): boolean {
    try {
      process.kill(-pid, signal);
      return true;
    } catch {
      // ESRCH (group already gone) is fine; EPERM/unsupported -> caller falls back.
      return false;
    }
  }

  private killDescendants(pids: number[], signal: NodeJS.Signals): void {
    for (const pid of pids) {
      try {
        process.kill(pid, signal);
      } catch {
        /* already gone */
      }
    }
  }

  private async killChild(): Promise<void> {
    const child = this.child;
    this.child = null;
    this.childSpawned = false;
    if (!child || child.pid === undefined) {
      this.childAlive = false;
      return;
    }
    // Collect the shell's descendants *before* killing it: the shell is a
    // session leader, and shells move background jobs into their *own* process
    // groups, so the group kill alone does not reach them; once the shell dies
    // they reparent to init and can no longer be found by ancestry.
    const descendants = collectDescendants(child.pid);
    this.log(`engine: terminating child pid=${child.pid} (group + ${descendants.length} descendant(s))`);
    try {
      if (!this.killChildGroup(child.pid, "SIGTERM")) child.kill("SIGTERM");
    } catch {
      /* already gone */
    }
    this.killDescendants(descendants, "SIGTERM");
    const exited = await waitForExit(child, this.defaults.childKillGraceMs);
    if (!exited) {
      this.log(`engine: child pid=${child.pid} ignored SIGTERM; sending SIGKILL to group + descendants`);
      try {
        if (!this.killChildGroup(child.pid, "SIGKILL")) child.kill("SIGKILL");
      } catch {
        /* ignore */
      }
      this.killDescendants(descendants, "SIGKILL");
      await waitForExit(child, 2_000);
    }
    this.childAlive = false;
  }

  private async startActive(alias: string, cfg: ModelConfig): Promise<void> {
    if (cfg.managed === false) {
      // Never run start/stop commands against a remote engine from the proxy
      // host; it must also not poison the stop pairing.
      this.stopCommandForActive = null;
      this.activeProvider = cfg.provider;
      this.activeModelAlias = alias;
      if (cfg.healthEndpoint) {
        this.log(`engine: polling health at ${cfg.targetBaseUrl.replace(/\/$/, "")}${cfg.healthEndpoint} (remote)`);
        await this.waitForHealth(cfg);
      }
      this.log(`engine: ${cfg.provider} is ready for model "${alias}" (remote)`);
      return;
    }
    if (cfg.startCommand) {
      this.log(`engine: starting ${cfg.provider} via: ${cfg.startCommand}`);
      const child = spawn(cfg.startCommand, {
        shell: true,
        stdio: ["ignore", "pipe", "pipe"],
        detached: true,
      });
      this.child = child;
      this.childAlive = true;
      this.childSpawned = true;
      // Scope the handlers to *this* child: a stale exit event from a
      // previous child must not clobber the state of a newly spawned one.
      child.on("exit", (code, signal) => {
        if (this.child !== child) return;
        this.childAlive = false;
        this.child = null;
        this.log(`engine: child exited (code=${code} signal=${signal ?? "none"})`);
      });
      child.on("error", (err) => {
        if (this.child !== child) return;
        this.childAlive = false;
        this.log(`engine: child error: ${err.message}`);
      });
      pipeToLog(child, cfg.provider, this.log);
    }
    // Pair the stop command with the engine we are about to activate. This is
    // done unconditionally (even for externally-managed engines, which have no
    // startCommand) so a leftover stop command from a previous provider can't
    // leak into a later stop of a different provider.
    this.stopCommandForActive = cfg.stopCommand ?? null;
    this.activeProvider = cfg.provider;
    this.activeModelAlias = alias;
    const healthUrl = cfg.healthEndpoint
      ? `${cfg.targetBaseUrl.replace(/\/$/, "")}${cfg.healthEndpoint}`
      : null;
    if (healthUrl) {
      this.log(`engine: polling health at ${healthUrl}`);
      await this.waitForHealth(cfg);
    } else if (cfg.startCommand) {
      // No health endpoint: give the engine a moment to bind its port, then probe
      // the base URL itself.
      await this.probeBaseReady(cfg);
    }
    this.log(`engine: ${cfg.provider} is ready for model "${alias}"`);
  }

  private async waitForHealth(cfg: ModelConfig): Promise<void> {
    const url = `${cfg.targetBaseUrl.replace(/\/$/, "")}${cfg.healthEndpoint!}`;
    const d = this.defaults;
    const deadline = Date.now() + d.healthCheckTimeoutMs;
    let lastErr: string | null = null;
    for (;;) {
      // Cap each probe at the *remaining* budget: the configured timeout is a
      // per-attempt value, but it must never push a single stalled probe past
      // the overall deadline.
      const remaining = Math.max(1, deadline - Date.now());
      const ok = await this.healthCheck(cfg, url, Math.min(d.healthCheckTimeoutMs, remaining));
      if (ok) return;
      lastErr = "non-200";
      if (Date.now() >= deadline) break;
      await sleep(Math.min(d.healthCheckIntervalMs, Math.max(50, deadline - Date.now())));
    }
    throw new EngineError(
      `Engine for "${cfg.provider}" did not become healthy at ${url} within ${d.healthCheckTimeoutMs}ms (${lastErr})`,
    );
  }

  private async probeBaseReady(cfg: ModelConfig): Promise<void> {
    // Best-effort: some providers have no dedicated health route.
    const url = cfg.targetBaseUrl.replace(/\/$/, "");
    const d = this.defaults;
    const deadline = Date.now() + Math.min(d.healthCheckTimeoutMs, 15_000);
    for (;;) {
      try {
        const res = await fetch(url, {
          signal: AbortSignal.timeout(1_000),
          // Any HTTP response (even 404) means the server is up.
          redirect: "manual",
        });
        void res.body?.cancel();
        return;
      } catch {
        if (Date.now() >= deadline) break;
        await sleep(d.healthCheckIntervalMs);
      }
    }
    this.log(`engine: warning — ${url} not reachable after startup; assuming ready`);
  }

  /** One health probe: HTTP 200 -> true. */
  private async healthCheck(cfg: ModelConfig, url: string, timeoutMs: number): Promise<
    boolean
  > {
    try {
      // Remote engines often protect every route (including /v1/models)
      // behind the provider key — send it when configured.
      const headers: Record<string, string> = {};
      if (cfg.targetApiKey) headers.authorization = `Bearer ${cfg.targetApiKey}`;
      const res = await fetch(url, {
        headers,
        signal: AbortSignal.timeout(timeoutMs),
        redirect: "manual",
      });
      const ok = res.status === 200;
      void res.body?.cancel();
      return ok;
    } catch {
      return false;
    }
  }

  /** Graceful shutdown: stop the active engine and kill the child. */
  async shutdown(): Promise<void> {
    await this.withLock(async () => {
      // Latch first so transitions queued behind us fail instead of
      // re-spawning an engine right after we tore it down.
      this.shuttingDown = true;
      if (!this.activeProvider) {
        await this.killChild();
        this.stopCommandForActive = null;
        this.childSpawned = false;
        this.activeCfg = null;
        this.log("engine: shutdown complete");
        return;
      }
      if (this.stopCommandForActive) {
        this.log(`engine: shutdown — running stopCommand: ${this.stopCommandForActive}`);
        try {
          await withTimeout(
            execAsync(this.stopCommandForActive),
            this.defaults.stopCommandTimeoutMs,
            "stopCommand",
          );
        } catch (err) {
          this.log(`engine: shutdown stopCommand failed (continuing): ${errMsg(err)}`);
        }
      }
      await this.killChild();
      this.activeProvider = null;
      this.activeModelAlias = null;
      this.stopCommandForActive = null;
      this.activeCfg = null;
      this.log("engine: shutdown complete");
    });
  }
}

/**
 * BFS over the Linux /proc tree to collect all descendant pids of `pid`.
 * Returns [] when /proc is unavailable (non-Linux) — callers fall back to the
 * process group, which covers the common `sh -c 'exec ...'` case there.
 */
export function collectDescendants(pid: number): number[] {
  let entries: string[];
  try {
    entries = readdirSync("/proc");
  } catch {
    return [];
  }
  const out: number[] = [];
  const seen = new Set<number>([pid]);
  let frontier = [pid];
  for (;;) {
    const next: number[] = [];
    for (const entry of entries) {
      if (!/^[0-9]+$/.test(entry)) continue;
      const childPid = Number(entry);
      if (seen.has(childPid)) continue;
      let ppid: number;
      try {
        const stat = readFileSync(`/proc/${entry}/stat`, "utf8");
        // comm (field 2) may contain spaces/parens — parse after the last ")".
        const fields = stat.slice(stat.lastIndexOf(")") + 2).trim().split(/\s+/);
        ppid = Number(fields[1]);
      } catch {
        continue; // vanished mid-walk
      }
      if (frontier.includes(ppid)) {
        seen.add(childPid);
        out.push(childPid);
        next.push(childPid);
      }
    }
    if (next.length === 0) break;
    frontier = next;
  }
  return out;
}

function pipeToLog(child: ChildProcess, provider: string, log: EngineLog): void {
  const fmt = (stream: NodeJS.ReadableStream | null, label: string): void => {
    stream?.on("data", (chunk: Buffer | string) => {
      const text = (Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)))
        .toString("utf8")
        .split("\n")
        .filter((l) => l.length > 0);
      for (const line of text) log(`engine[${provider}]: ${line}`);
    });
  };
  fmt(child.stdout, "stdout");
  fmt(child.stderr, "stderr");
}

function waitForExit(child: ChildProcess, ms: number): Promise<boolean> {
  return new Promise((resolve) => {
    if (child.exitCode !== null || child.signalCode !== null) {
      resolve(true);
      return;
    }
    const t = setTimeout(() => {
      child.removeListener("exit", onExit);
      resolve(false);
    }, ms);
    const onExit = (): void => {
      clearTimeout(t);
      resolve(true);
    };
    child.once("exit", onExit);
  });
}

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new EngineError(`${label} timed out after ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

export function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
