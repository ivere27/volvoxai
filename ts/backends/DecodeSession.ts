import type {
  BackendEngineLike,
  BackendExecutionOptions,
} from './BackendEngine.js';
import { decodeRowModes } from '../generated/volvoxaiEnums.js';
import type { DecodeRowModeValue } from '../generated/volvoxaiEnums.js';

/**
 * Backend-neutral stateful decode orchestration.
 *
 * A session owns one backend's retained intermediate cache between `seed()`
 * and `step()` calls. The execution result is intentionally passed through
 * unchanged: host backends return output typed arrays while WebGPU returns
 * device-resident buffers with an explicit `readBuffer()` operation.
 */

const ROW_MODES = new Set<DecodeRowModeValue>(decodeRowModes);
const ENGINE_DECODE_TAIL: unique symbol = Symbol('volvoxai.decodeOperationTail');

type DecodeOperationOwner = BackendEngineLike & {
  [ENGINE_DECODE_TAIL]?: Promise<unknown>;
};

export type DecodeRowMode = DecodeRowModeValue;
export type DecodeExecutionMode =
  | 'incremental-row'
  | 'incremental-dependency'
  | 'incremental-seed'
  | 'ordinary-forward';

export interface DecodeSessionOptions {
  changedInputs?: string[] | null;
  rowMode?: DecodeRowMode;
  requireIncremental?: boolean;
}

export interface DecodeStepOptions extends BackendExecutionOptions {
  changedInputs?: string[];
  position?: number;
}

type NamedInputs = Record<string, ArrayBufferView>;

function inputNames(inputs: NamedInputs, label: string) {
  if (inputs == null || typeof inputs !== 'object' || ArrayBuffer.isView(inputs)) {
    throw new Error(`DecodeSession.${label} requires a named input object.`);
  }
  return Object.keys(inputs);
}

function changedInputNames(value: string[] | null | undefined, fallback: string[], label: string) {
  const names = value == null ? fallback : value;
  if (!Array.isArray(names) || names.some((name) => typeof name !== 'string' || name.length === 0)) {
    throw new Error(`DecodeSession ${label} changedInputs must be an array of non-empty tensor names.`);
  }
  return [...new Set(names)];
}

function ordinaryExecutionOptions(options: DecodeStepOptions): BackendExecutionOptions {
  const result = { ...options };
  delete result.changedInputs;
  delete result.position;
  delete result.incremental;
  delete result.incrementalReset;
  delete result.incrementalRowPosition;
  return result;
}

export class DecodeSession {
  readonly engine: BackendEngineLike;
  readonly backendName: string;
  readonly capabilities: {
    readonly incremental: boolean;
    readonly incrementalRows: boolean;
  };
  readonly changedInputs: readonly string[] | null;
  readonly rowMode: DecodeRowMode;
  readonly mode: DecodeExecutionMode;
  lastExecutionMode: DecodeExecutionMode | null;
  private _seeded: boolean;
  private _closed: boolean;
  private _cacheGeneration: number | null;
  private _tail: Promise<unknown>;

  constructor(engine: BackendEngineLike, {
    changedInputs = null,
    rowMode = 'auto',
    requireIncremental = false,
  }: DecodeSessionOptions = {}) {
    if (!engine || typeof engine.execute !== 'function') {
      throw new Error('DecodeSession requires a compiled backend engine with execute(inputs, options).');
    }
    if (!ROW_MODES.has(rowMode)) {
      throw new Error("DecodeSession rowMode must be 'auto', 'required', or 'disabled'.");
    }
    if (changedInputs != null) {
      changedInputs = changedInputNames(changedInputs, [], 'default');
    }

    const incremental = engine.capabilities.incrementalExecution === true;
    const incrementalRows = engine.capabilities.incrementalRows === true;
    if (requireIncremental && !incremental) {
      throw new Error(`Backend '${engine.backendName || engine.constructor?.name || 'unknown'}' does not support incremental execution.`);
    }
    if (rowMode === 'required' && !incrementalRows) {
      throw new Error(`Backend '${engine.backendName || engine.constructor?.name || 'unknown'}' does not support incremental row execution.`);
    }

    this.engine = engine;
    this.backendName = engine.backendName || engine.constructor?.name || 'unknown';
    this.capabilities = Object.freeze({ incremental, incrementalRows });
    this.changedInputs = changedInputs == null ? null : Object.freeze(changedInputs);
    this.rowMode = rowMode;
    this.mode = incrementalRows && rowMode !== 'disabled'
      ? 'incremental-row'
      : incremental
        ? 'incremental-dependency'
        : 'ordinary-forward';
    this.lastExecutionMode = null;
    this._seeded = false;
    this._closed = false;
    this._cacheGeneration = null;
    this._tail = Promise.resolve();
  }

  get seeded() {
    return this._seeded;
  }

  get closed() {
    return this._closed;
  }

  _enqueue<TResult>(operation: () => Promise<TResult> | TResult): Promise<TResult> {
    /* One engine owns one retained-intermediate cache. Serialize every
     * session's state transition on that engine so async device submissions
     * cannot overlap and both claim the same cache generation. */
    const owner = this.engine as DecodeOperationOwner;
    const engineTail = owner[ENGINE_DECODE_TAIL] ?? Promise.resolve();
    const pending = engineTail.then(operation, operation);
    const settled = pending.catch(() => undefined);
    owner[ENGINE_DECODE_TAIL] = settled;
    this._tail = settled;
    return pending;
  }

  _assertOpen() {
    if (this._closed) throw new Error('DecodeSession is closed.');
  }

  _ownsCurrentCache() {
    if (!this._seeded) return false;
    return this._cacheGeneration == null || this.engine.decodeCacheGeneration == null ||
      this._cacheGeneration === this.engine.decodeCacheGeneration;
  }

  _invalidate({ forceReset = false }: { forceReset?: boolean } = {}) {
    const resetEngine = forceReset || this._ownsCurrentCache();
    this._seeded = false;
    this._cacheGeneration = null;
    this.lastExecutionMode = null;
    if (resetEngine) this.engine.resetDecodeCache?.();
  }

  /**
   * Run a complete first pass and seed every retained intermediate.
   */
  seed(inputs: NamedInputs, options: DecodeStepOptions = {}) {
    return this._enqueue(async () => {
      this._assertOpen();
      const names = inputNames(inputs, 'seed');
      this.engine.resetDecodeCache?.();
      const seedGeneration = this.engine.decodeCacheGeneration ?? null;
      this._seeded = false;
      this._cacheGeneration = null;
      const executionOptions = ordinaryExecutionOptions(options);
      if (this.capabilities.incremental) {
        executionOptions.incremental = true;
        executionOptions.incrementalReset = true;
        executionOptions.changedInputs = changedInputNames(options.changedInputs, names, 'seed');
        this.engine._claimDecodeSessionExecution?.(executionOptions, seedGeneration);
      }
      let result;
      try {
        result = await this.engine.execute(inputs, executionOptions);
      } catch (error) {
        this._invalidate({ forceReset: true });
        throw error;
      }
      if (seedGeneration != null && this.engine.decodeCacheGeneration != null &&
          seedGeneration !== this.engine.decodeCacheGeneration) {
        this._invalidate();
        throw new Error('DecodeSession cache was reset or replaced during seed(); call seed() again.');
      }
      this._seeded = true;
      this._cacheGeneration = seedGeneration;
      this.lastExecutionMode = this.capabilities.incremental ? 'incremental-seed' : 'ordinary-forward';
      return result;
    });
  }

  /**
   * Run one decode update. `position` opts row-capable engines into their
   * fixed-shape W8A8 row/KV-cache path; other incremental engines rerun only
   * the dependency closure of `changedInputs`.
   */
  step(inputs: NamedInputs, options: DecodeStepOptions = {}) {
    return this._enqueue(async () => {
      this._assertOpen();
      const names = inputNames(inputs, 'step');
      if (!this._seeded) {
        throw new Error('DecodeSession.step requires a successful seed() pass.');
      }
      if (this._cacheGeneration != null && this.engine.decodeCacheGeneration != null &&
          this._cacheGeneration !== this.engine.decodeCacheGeneration) {
        this._seeded = false;
        this.lastExecutionMode = null;
        throw new Error('DecodeSession cache was reset or replaced; call seed() again.');
      }

      const executionOptions = ordinaryExecutionOptions(options);
      if (this.capabilities.incremental) {
        executionOptions.incremental = true;
        executionOptions.changedInputs = changedInputNames(
          options.changedInputs,
          this.changedInputs == null ? names : [...this.changedInputs],
          'step',
        );
        const position = options.position;
        const useRow = this.rowMode !== 'disabled' &&
          this.capabilities.incrementalRows && position != null;
        if (this.rowMode === 'required' && position == null) {
          throw new Error("DecodeSession rowMode 'required' needs an integer position on every step().");
        }
        if (useRow) {
          if (typeof position !== 'number' || !Number.isInteger(position) || position < 1) {
            throw new Error('DecodeSession row position must be an integer >= 1.');
          }
          executionOptions.incrementalRowPosition = position;
          this.lastExecutionMode = 'incremental-row';
        } else {
          this.lastExecutionMode = 'incremental-dependency';
        }
        this.engine._claimDecodeSessionExecution?.(executionOptions, this._cacheGeneration);
      } else {
        this.lastExecutionMode = 'ordinary-forward';
      }

      let result;
      try {
        result = await this.engine.execute(inputs, executionOptions);
      } catch (error) {
        this._invalidate({ forceReset: true });
        throw error;
      }
      if (this._cacheGeneration != null && this.engine.decodeCacheGeneration != null &&
          this._cacheGeneration !== this.engine.decodeCacheGeneration) {
        this._invalidate();
        throw new Error('DecodeSession cache was reset or replaced during step(); call seed() again.');
      }
      return result;
    });
  }

  /** Invalidate retained intermediates. The session may be seeded again. */
  reset() {
    return this._enqueue(async () => {
      this._assertOpen();
      this._invalidate();
      this.lastExecutionMode = null;
    });
  }

  /** Invalidate retained state and permanently close this session. */
  close() {
    return this._enqueue(async () => {
      if (this._closed) return;
      this._invalidate();
      this.lastExecutionMode = null;
      this._closed = true;
    });
  }

  dispose() {
    return this.close();
  }
}
