import { RpcError, type CallOptions } from
  '../../runtime/generated/typescript/inference/synurang_runtime.js';

const MAX_TIMER_MS = 2 ** 31 - 1;

/** One call's cancellation and elapsed timeout, including lazy preparation. */
export class CallScope {
  readonly #controller = new AbortController();
  readonly #startedAt = performance.now();
  readonly #timeoutMs: number | undefined;
  readonly #unlink: Array<() => void> = [];
  #timer: ReturnType<typeof setTimeout> | undefined;
  #closed = false;

  constructor(options: CallOptions | undefined, hostSignal: AbortSignal) {
    const timeoutMs = options?.timeoutMs;
    if (timeoutMs !== undefined &&
        (!Number.isSafeInteger(timeoutMs) || timeoutMs < 0)) {
      throw new RangeError('timeoutMs must be a nonnegative safe integer');
    }
    this.#timeoutMs = timeoutMs;
    for (const signal of [options?.signal, hostSignal]) {
      if (signal === undefined) continue;
      if (signal.aborted) {
        this.cancel();
        return;
      }
      const abort = () => this.cancel();
      signal.addEventListener('abort', abort, { once: true });
      this.#unlink.push(() => signal.removeEventListener('abort', abort));
    }
    if (timeoutMs !== undefined) this.#scheduleDeadline();
  }

  get signal(): AbortSignal { return this.#controller.signal; }

  /** Pass only the unspent budget when the native call can finally open. */
  get remainingOptions(): CallOptions {
    this.#check();
    return this.#timeoutMs === undefined
      ? { signal: this.signal }
      : { signal: this.signal, timeoutMs: Math.ceil(Math.max(0, this.#remaining())) };
  }

  #remaining(): number {
    return this.#timeoutMs! - (performance.now() - this.#startedAt);
  }

  #check(): void {
    if (!this.signal.aborted && this.#timeoutMs !== undefined && this.#remaining() <= 0) {
      this.cancel(4);
    }
    if (this.signal.aborted) throw this.signal.reason;
  }

  #scheduleDeadline(): void {
    if (this.#closed || this.signal.aborted) return;
    const remaining = this.#remaining();
    if (remaining <= 0) this.cancel(4);
    else this.#timer = setTimeout(() => this.#scheduleDeadline(), Math.min(remaining, MAX_TIMER_MS));
  }

  /** Stop waiting without cancelling an immutable module load shared by hosts. */
  wait<T>(operation: PromiseLike<T>): Promise<T> {
    const pending = Promise.resolve(operation);
    try { this.#check(); } catch (error) {
      // A cancelled waiter still observes a later rejection of its operation.
      void pending.catch(() => {});
      return Promise.reject(error);
    }
    return new Promise<T>((resolve, reject) => {
      const abort = () => {
        this.signal.removeEventListener('abort', abort);
        reject(this.signal.reason);
      };
      this.signal.addEventListener('abort', abort, { once: true });
      pending.then(value => {
        this.signal.removeEventListener('abort', abort);
        try { this.#check(); resolve(value); } catch (error) { reject(error); }
      }, error => {
        this.signal.removeEventListener('abort', abort);
        try { this.#check(); reject(error); } catch (aborted) { reject(aborted); }
      });
    });
  }

  cancel(code = 1): void {
    this.abort(new RpcError(code, code === 4 ? 'Deadline exceeded' : 'Call cancelled'));
  }

  abort(error: unknown): void {
    if (this.#closed || this.signal.aborted) return;
    const reason = error instanceof RpcError ? error :
      new RpcError(13, error instanceof Error ? error.message : String(error));
    this.#controller.abort(reason);
    this.#dispose();
  }

  #dispose(): void {
    if (this.#timer !== undefined) clearTimeout(this.#timer);
    this.#timer = undefined;
    for (const unlink of this.#unlink.splice(0)) unlink();
  }

  /** Dispose a completed call's timer and external abort listeners. */
  close(): void {
    this.#closed = true;
    this.#dispose();
  }
}
