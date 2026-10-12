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

/**
 * How the active engine is driven locally:
 * - `"command"`: a single spawned child the proxy holds a process handle on
 *   (stopped by killing its whole process group).
 * - `"commands"`: separate start/stop shell commands; the proxy holds no
 *   process handle and only tracks whether the engine was started (stopped by
 *   running the stop command).
 */
export type EngineKind = "command" | "commands";

/** Local lifecycle kind for a model config (null = no local process management). */
export function engineKind(cfg: ModelConfig): EngineKind | null {
  if (cfg.managed === false) return null;
  if (typeof cfg.command === "string") return "command";
  if (cfg.commands) return "commands";
  // Neither `command` nor `commands`: externally managed — the proxy only
  // health-checks and forwards.
  return null;
}

export interface EngineState {
  provider: string | null;
  modelAlias: string | null;
  /** pid of the spawned child process (`command` mode only; null otherwise). */
  pid: number | null;
  /** true while the active engine process is running. */
  running: boolean;
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
  /** How the active engine is driven (null = none / remote / externally managed). */
  private kind: EngineKind | null = null;
  /** Handle of the spawned child (`command` mode only). */
  private child: ChildProcess | null = null;
  /** `command` mode: the spawned child is alive. */
  private childAlive = false;
  /** `commands` mode: the start command was fired and stop has not run yet. */
  private started = false;
  private switches = 0;
  private lock: Promise<unknown> = Promise.resolve();
  /** Config of the last *successful* start; used to restore the previous engine after a failed transition. */
  private activeCfg: ModelConfig | null = null;
  /** Stop command paired with the active engine (`commands` mode only). */
  private stopCommandForActive: string | null = null;
  /** Latches on shutdown; subsequent ensureReady/transition calls fail fast. */
  private shuttingDown = false;

  constructor(
    private defaults: RequiredDefaults,
    private vram: VramMonitor,
    private log: EngineLog = () => {},
  ) {}

  /** true while the active engine process is running. */
  get running(): boolean {
    if (this.kind === "command") return this.childAlive;
    if (this.kind === "commands") return this.started;
    return false;
  }

  state(): EngineState {
    return {
      provider: this.activeProvider,
      modelAlias: this.activeModelAlias,
      pid: this.child?.pid ?? null,
      running: this.running,
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
      // If the active engine is not running anymore, bring it back (only for
      // locally managed engines whose kind matches the requested config).
      const kind = engineKind(cfg);
      if (kind !== null && this.kind === kind && !this.running) {
        this.log(`engine: ${cfg.provider} not running; restarting`);
        await this.withLock(async () => {
          // Re-check under the lock: a concurrent transition may have changed
          // the active provider or already restarted the engine.
          if (this.activeProvider === cfg.provider && this.kind === kind && !this.running) {
            try {
              await this.startActive(alias, cfg);
            } catch (err) {
              // A failed restart must not leave a half-started engine behind
              // (it would keep holding VRAM, and in `commands` mode the stale
              // started flag would block any retry). Tear it down and keep the
              // kind paired so the next request retries cleanly.
              await this.stopActive();
              this.kind = kind;
              this.started = false;
              throw err;
            }
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
    const fromKind = this.kind;
    this.log(`engine: preparing switch ${from ?? "(none)"} -> ${cfg.provider} (model "${alias}")`);
    const remote = engineKind(cfg) === null;
    try {
      await this.stopActive();
      if (remote) {
        // The engine runs on another host: no local cool-down, no local VRAM
        // gate, no start command — just verify it is reachable/healthy.
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
      // The new engine half-started: tear it down so it doesn't keep holding
      // VRAM while we roll back to the previous engine.
      await this.stopActive();
      // Best effort: restore the previously active engine so it stays usable.
      let restored = false;
      if (from !== null && fromCfg && engineKind(fromCfg) !== null) {
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
        // The previous engine was already stopped and its restore failed, so
        // mark it not running: the next request for it re-runs its start.
        this.activeProvider = from;
        this.activeModelAlias = fromAlias;
        this.activeCfg = fromCfg;
        this.kind = fromKind;
        this.stopCommandForActive = fromStopCmd;
        this.started = false;
      }
      throw err;
    }
  }

  /**
   * Stop the currently active engine per its lifecycle kind:
   * - `commands`: run the paired stop command (only if we actually started
   *   something — tracked in `started`; there is no process handle to kill).
   * - `command`: kill the child's whole process group.
   */
  private async stopActive(): Promise<void> {
    if (!this.activeProvider) return;
    if (this.kind === "commands") {
      if (this.started && this.stopCommandForActive) {
        this.log(`engine: running stop command: ${this.stopCommandForActive}`);
        try {
          await withTimeout(
            execAsync(this.stopCommandForActive),
            this.defaults.stopCommandTimeoutMs,
            "stop command",
          );
        } catch (err) {
          this.log(`engine: stop command failed (continuing): ${errMsg(err)}`);
        }
      } else if (!this.started) {
        this.log("engine: no stop command run — engine was not started");
      }
      this.started = false;
    } else if (this.kind === "command") {
      await this.killChild();
    }
    // The active engine is gone; clear its kind/stop command so they can't run
    // again against a *different* provider that later becomes active.
    this.kind = null;
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
    const kind = engineKind(cfg);
    if (kind === null) {
      // Never run start/stop commands against a remote engine from the proxy
      // host; it must also not poison the stop pairing.
      this.kind = null;
      this.stopCommandForActive = null;
      this.started = false;
      this.activeProvider = cfg.provider;
      this.activeModelAlias = alias;
      if (cfg.healthEndpoint) {
        this.log(`engine: polling health at ${cfg.targetBaseUrl.replace(/\/$/, "")}${cfg.healthEndpoint} (remote)`);
        await this.waitForHealth(cfg);
      }
      this.log(`engine: ${cfg.provider} is ready for model "${alias}" (remote)`);
      return;
    }
    if (kind === "command") {
      this.log(`engine: starting ${cfg.provider} via command: ${cfg.command}`);
      const child = spawn(cfg.command!, {
        shell: true,
        stdio: ["ignore", "pipe", "pipe"],
        detached: true,
      });
      this.child = child;
      this.childAlive = true;
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
    } else {
      // `commands` mode: fire the start command without holding a process
      // handle — it may launch another server process that only `stop` can
      // stop. The launcher is unref'd so it cannot keep the proxy alive; its
      // output is piped to the log while it runs. Readiness is established by
      // the health poll below, not by the launcher's exit.
      this.log(`engine: starting ${cfg.provider} via commands.start: ${cfg.commands!.start}`);
      const launcher = spawn(cfg.commands!.start, {
        shell: true,
        stdio: ["ignore", "pipe", "pipe"],
        detached: true,
      });
      launcher.on("error", (err) => this.log(`engine: start command error: ${err.message}`));
      pipeToLog(launcher, cfg.provider, this.log);
      launcher.unref();
      this.started = true;
    }
    // Pair the stop command with the engine we are about to activate. This is
    // done unconditionally (even for externally-managed engines, which have no
    // start command) so a leftover stop command from a previous provider can't
    // leak into a later stop of a different provider.
    this.kind = kind;
    this.stopCommandForActive = kind === "commands" ? cfg.commands!.stop : null;
    this.activeProvider = cfg.provider;
    this.activeModelAlias = alias;
    const healthUrl = cfg.healthEndpoint
      ? `${cfg.targetBaseUrl.replace(/\/$/, "")}${cfg.healthEndpoint}`
      : null;
    if (healthUrl) {
      this.log(`engine: polling health at ${healthUrl}`);
      await this.waitForHealth(cfg);
    } else {
      // No health endpoint: give the engine a moment to bind its port, then
      // probe the base URL itself.
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

  /** Graceful shutdown: stop the active engine (stop command / process-group kill). */
  async shutdown(): Promise<void> {
    await this.withLock(async () => {
      // Latch first so transitions queued behind us fail instead of
      // re-spawning an engine right after we tore it down.
      this.shuttingDown = true;
      if (this.activeProvider) {
        await this.stopActive();
      } else {
        // No active engine, but reap a stray child if one exists.
        await this.killChild();
      }
      this.activeProvider = null;
      this.activeModelAlias = null;
      this.kind = null;
      this.stopCommandForActive = null;
      this.started = false;
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
