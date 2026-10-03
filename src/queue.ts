/**
 * A strict serial (single-concurrency) queue.
 *
 * Tasks are executed one at a time, in FIFO order. `enqueue` returns a promise
 * that settles with the task's result; a task that throws does not break the
 * queue. `length` is the number of tasks currently running + waiting.
 */
export class SerialQueue {
  private tail: Promise<unknown> = Promise.resolve();
  private depth = 0;

  /** Number of tasks currently running or waiting. */
  get length(): number {
    return this.depth;
  }

  /**
   * Number of tasks waiting (excluding the one currently running).
   * `depth` counts the running task too, so this is max(depth-1, 0) —
   * 0 both when idle and when exactly one task is in flight.
   */
  get waiting(): number {
    return Math.max(this.depth - 1, 0);
  }

  get running(): boolean {
    return this.depth > 0;
  }

  enqueue<T>(task: () => Promise<T>): Promise<T> {
    this.depth++;
    const run = this.tail.then(task);
    // The chain must never reject, or the queue would stall — so both paths
    // settle to undefined. Decrementing here (in the tail handler itself) means
    // `length` reflects the queue while a task runs, and is already decremented
    // before the next task starts: no microtask of overcount.
    this.tail = run.then(
      () => {
        this.depth--;
      },
      () => {
        this.depth--;
      },
    );
    return run;
  }

  /** Resolves once every enqueued task has settled. */
  drain(): Promise<void> {
    const current = this.tail;
    this.tail = Promise.resolve();
    return current.then(
      () => undefined,
      () => undefined,
    );
  }
}
