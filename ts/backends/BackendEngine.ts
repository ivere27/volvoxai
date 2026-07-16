import { DecodeSession } from './DecodeSession.js';
import { validatePortableQuantizedGraph } from '../ops/quantizedGraphValidation.js';
import type { PortableQuantizedGraph } from '../ops/quantizedGraphValidation.js';

export type BackendOutputLocation = 'host' | 'device';

export interface BackendCapabilityOptions {
  incrementalExecution?: boolean;
  incrementalRows?: boolean;
  outputLocation?: BackendOutputLocation;
}

export interface BackendCapabilities {
  readonly incrementalExecution: boolean;
  readonly incrementalRows: boolean;
  readonly outputLocation: BackendOutputLocation;
}

export interface BackendExecutionOptions {
  incremental?: boolean;
  incrementalReset?: boolean;
  incrementalRowPosition?: number;
  changedInputs?: string[];
  position?: number;
  [name: string | symbol]: any;
}

export interface BackendEngineLike {
  backendApiVersion: number;
  backendName: string;
  capabilities: BackendCapabilities;
  supportsIncrementalExecution?: boolean;
  supportsIncrementalRows?: boolean;
  decodeCacheGeneration?: number;
  allocateGraph(...args: any[]): any;
  execute(...args: any[]): Promise<any> | any;
  createDecodeSession(options?: any): DecodeSession;
  resetDecodeCache?(): void;
  _claimDecodeSessionExecution?(options: BackendExecutionOptions, generation: number | null): BackendExecutionOptions;
}

/** Version of the public JavaScript backend lifecycle contract. */
export const VOLVOXAI_BACKEND_API_VERSION = 1;

const OUTPUT_LOCATIONS = new Set(['host', 'device']);
const DECODE_SESSION_CLAIM = Symbol('volvoxai.decodeSessionClaim');

export function createBackendCapabilities({
  incrementalExecution = false,
  incrementalRows = false,
  outputLocation = 'host',
}: BackendCapabilityOptions = {}): Readonly<BackendCapabilities> {
  if (incrementalRows && !incrementalExecution) {
    throw new Error('incrementalRows requires incrementalExecution.');
  }
  if (!OUTPUT_LOCATIONS.has(outputLocation)) {
    throw new Error("Backend outputLocation must be 'host' or 'device'.");
  }
  return Object.freeze({
    incrementalExecution: incrementalExecution === true,
    incrementalRows: incrementalRows === true,
    outputLocation,
  });
}

/**
 * Public browser-backend contract.
 *
 * Implementations bind one Graph with `allocateGraph(graph)`, then execute it
 * through `execute(inputs, options)`. Extensible backends registered on
 * `VolvoxAI` may subclass this class or provide the same public properties and
 * methods. Device-output backends also expose their own explicit readback API.
 */
export abstract class BackendEngine {
  backendApiVersion!: number;
  backendName!: string;
  capabilities!: Readonly<BackendCapabilities>;
  supportsIncrementalExecution!: boolean;
  supportsIncrementalRows!: boolean;
  protected _decodeCacheGeneration: number;
  protected _decodeCacheGenerationListener: ((generation: number) => void) | null;
  protected _incrementalCacheValid = false;

  abstract allocateGraph(...args: any[]): any;
  abstract execute(...args: any[]): Promise<any> | any;

  constructor(name: string, capabilities: BackendCapabilityOptions = {}) {
    this._configureBackend(name, capabilities);
    this._decodeCacheGeneration = 0;
    this._decodeCacheGenerationListener = null;
  }

  _configureBackend(name: string, capabilities: BackendCapabilityOptions = {}) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('BackendEngine requires a non-empty backend name.');
    }
    this.backendApiVersion = VOLVOXAI_BACKEND_API_VERSION;
    this.backendName = name;
    this.capabilities = createBackendCapabilities(capabilities);
    // Compatibility aliases used by existing model sessions.
    this.supportsIncrementalExecution = this.capabilities.incrementalExecution;
    this.supportsIncrementalRows = this.capabilities.incrementalRows;
  }

  /** @internal Reject byte-domain graphs that bypassed package loading. */
  protected _assertPortableQuantizedGraph(graph: PortableQuantizedGraph): void {
    validatePortableQuantizedGraph(graph);
  }

  get decodeCacheGeneration() {
    return this._decodeCacheGeneration;
  }

  _advanceDecodeCacheGeneration() {
    if (!Number.isSafeInteger(this._decodeCacheGeneration) ||
        this._decodeCacheGeneration >= Number.MAX_SAFE_INTEGER) {
      throw new Error('Backend decode-cache generation exhausted its safe integer range.');
    }
    this._decodeCacheGeneration++;
    this._decodeCacheGenerationListener?.(this._decodeCacheGeneration);
  }

  /** @internal Notify a composing backend when this engine replaces state. */
  _setDecodeCacheGenerationListener(listener: ((generation: number) => void) | null = null) {
    if (listener != null && typeof listener !== 'function') {
      throw new Error('Decode-cache generation listener must be a function or null.');
    }
    this._decodeCacheGenerationListener = listener;
  }

  resetDecodeCache() {
    this._incrementalCacheValid = false;
    this._advanceDecodeCacheGeneration();
  }

  /** @internal Bind one execution to the cache generation owned by a session. */
  _claimDecodeSessionExecution(options: BackendExecutionOptions, generation: number | null) {
    Object.defineProperty(options, DECODE_SESSION_CLAIM, {
      configurable: true,
      value: Object.freeze({ engine: this, generation }),
    });
    return options;
  }

  /**
   * @internal Begin a backend execution before it reads or writes retained
   * intermediates. Direct legacy execute() calls replace the current session's
   * ownership while preserving their established incremental-cache behavior.
   */
  _beginDecodeExecution(options: BackendExecutionOptions = {}) {
    if (this.capabilities.incrementalExecution !== true) return false;
    const claim = options?.[DECODE_SESSION_CLAIM];
    if (claim?.engine === this) {
      if (claim.generation !== this._decodeCacheGeneration) {
        throw new Error('Decode cache was reset or replaced before execution.');
      }
      return true;
    }
    this._advanceDecodeCacheGeneration();
    return false;
  }

  createDecodeSession(options = {}) {
    return new DecodeSession(this, options);
  }
}

/** Validate the structural contract accepted from a registered backend factory. */
export function assertBackendEngine(engine: any, label = 'Backend'): BackendEngineLike {
  const capabilities = engine?.capabilities;
  const validCapabilities = capabilities && Object.isFrozen(capabilities) &&
    typeof capabilities.incrementalExecution === 'boolean' &&
    typeof capabilities.incrementalRows === 'boolean' &&
    (!capabilities.incrementalRows || capabilities.incrementalExecution) &&
    OUTPUT_LOCATIONS.has(capabilities.outputLocation);
  const validIncrementalLifecycle = capabilities?.incrementalExecution !== true ||
    (typeof engine.resetDecodeCache === 'function' &&
      Number.isSafeInteger(engine.decodeCacheGeneration) &&
      engine.decodeCacheGeneration >= 0);
  if (!engine || engine.backendApiVersion !== VOLVOXAI_BACKEND_API_VERSION ||
      typeof engine.backendName !== 'string' || !validCapabilities || !validIncrementalLifecycle ||
      typeof engine.allocateGraph !== 'function' || typeof engine.execute !== 'function' ||
      typeof engine.createDecodeSession !== 'function') {
    throw new Error(`${label} must implement BackendEngine API v${VOLVOXAI_BACKEND_API_VERSION}.`);
  }
  return engine;
}
