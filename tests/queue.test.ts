import { describe, it, expect } from "vitest";
import { SerialQueue } from "../src/queue.js";
import { sleep } from "../src/vram.js";

describe("SerialQueue", () => {
  it("runs exactly one task at a time, in FIFO order", async () => {
    const q = new SerialQueue();
    const events: string[] = [];
    let maxConcurrent = 0;
    let concurrent = 0;

    const mk = (id: string, ms: number) =>
      q.enqueue(async () => {
        concurrent++;
        maxConcurrent = Math.max(maxConcurrent, concurrent);
        events.push(`start-${id}`);
        await sleep(ms);
        events.push(`end-${id}`);
        concurrent--;
        return id;
      });

    const p1 = mk("a", 40);
    const p2 = mk("b", 5);
    const p3 = mk("c", 5);
    const results = await Promise.all([p1, p2, p3]);

    expect(maxConcurrent).toBe(1);
    expect(results).toEqual(["a", "b", "c"]);
    // b and c must not have started until a finished
    expect(events.indexOf("start-b")).toBeGreaterThan(events.indexOf("end-a"));
    expect(events.indexOf("start-c")).toBeGreaterThan(events.indexOf("end-b"));
    expect(q.length).toBe(0);
  });

  it("isolates failures: a throwing task does not stall the queue", async () => {
    const q = new SerialQueue();
    const bad = q.enqueue(async () => {
      await sleep(5);
      throw new Error("boom");
    });
    const good = q.enqueue(async () => "ok");
    await expect(bad).rejects.toThrow("boom");
    await expect(good).resolves.toBe("ok");
    expect(q.length).toBe(0);
  });

  it("tracks running/waiting counts", async () => {
    const q = new SerialQueue();
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const t1 = q.enqueue(async () => {
      await gate;
    });
    const t2 = q.enqueue(async () => {});
    await sleep(10);
    expect(q.running).toBe(true);
    expect(q.length).toBe(2);
    expect(q.waiting).toBe(1);
    release();
    await Promise.all([t1, t2]);
    expect(q.length).toBe(0);
  });

  it("drain() resolves once all tasks settle", async () => {
    const q = new SerialQueue();
    q.enqueue(() => sleep(10));
    q.enqueue(() => sleep(10));
    const t0 = Date.now();
    await q.drain();
    expect(Date.now() - t0).toBeGreaterThanOrEqual(15);
  });
});
