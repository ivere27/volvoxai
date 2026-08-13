import { DecodeSession } from './DecodeSession.js';
import { validatePortableQuantizedGraph } from '../ops/quantizedGraphValidation.js';
import { memoryLocations } from '../generated/volvoxaiEnums.js';
import type { PortableQuantizedGraph } from '../ops/quantizedGraphValidation.js';
import type { MemoryLocationValue } from '../generated/volvoxaiEnums.js';
import type { KVPagePlan } from './kvPageAddressing.js';

export type BackendOutputLocation = MemoryLocationValue;

export interface BackendCapabilityOptions {
  incrementalExecution?: boolean;
  incrementalRows?: boolean;
  /** See ProviderDecodeCapabilities.sequenceMajorRows. */
  sequenceMajorRows?: boolean;
  outputLocation?: BackendOutputLocation;
}

export interface BackendCapabilities {
  readonly incrementalExecution: boolean;
  readonly incrementalRows: boolean;
  readonly sequenceMajorRows: boolean;
  readonly outputLocation: BackendOutputLocation;
}

export interface BackendExecutionOptions {
  incremental?: boolean;
  incrementalReset?: boolean;
  incrementalRowPosition?: number;
  /**
   * Every lane's row this step, for a context that owns more than one slot.
   *
   * The batched spelling of `incrementalRowPosition`; both build the same
   * `DecodeRowSet` inside the engine, so a one-lane step cannot drift from a
   * batched one. Declaring both is refused rather than resolved by precedence.
   * A backend that cannot execute this must refuse it: silently falling back to
   * a full recompute would report a row step it did not run.
   */
  incrementalRowLanes?: readonly {
    readonly position?: number;
    readonly kvPages?: KVPagePlan | null;
    /** This lane holds no request; it occupies a row and its output is dropped. */
    readonly parked?: boolean;
  }[];
  changedInputs?: string[];
  position?: number;
  /**
   * Page table and active K/V length for the lane this step advances. Null or
   * absent keeps the established contiguous slice — which is also what the
   * identity mapping produces, so the two are one code path, not two.
   *
   * The one-lane spelling. A batched step carries one plan per lane inside
   * `incrementalRowLanes`, because lanes hold different pages.
   */
  kvPages?: KVPagePlan | null;
  [name: string | symbol]: any;
}

export interface BackendEngineLike {
  backendName: string;
  capabilities: BackendCapabilities;
  decodeCacheGeneration?: number;
  adapterInfo?: Readonly<Record<string, string>> | null;
  allocateGraph(...args: any[]): any;
  execute(...args: any[]): Promise<any> | any;
  createDecodeSession(options?: any): DecodeSession;
  resetDecodeCache?(): void;
  _claimDecodeSessionExecution?(options: BackendExecutionOptions, generation: number | null): BackendExecutionOptions;
}

const OUTPUT_LOCATIONS = new Set<unknown>(memoryLocations);
const DECODE_SESSION_CLAIM = Symbol('volvoxai.decodeSessionClaim');
const FORBIDDEN_INFERENCE_OPTIONS = Object.freeze([
  'training', 'trainingMode', 'dropout', 'dropoutSeed', 'dropoutCounter',
  'rng', 'seed', 'counter',
]);

/** @internal Reject training control data at every built-in inference engine. */
export function assertInferenceExecutionOptions(options: unknown, label: string): void {
  if (options == null) return;
  if (typeof options !== 'object' || Array.isArray(options)) {
    throw new Error(`${label} options must be an object.`);
  }
  for (const name of FORBIDDEN_INFERENCE_OPTIONS) {
    if (Object.prototype.hasOwnProperty.call(options, name)) {
      throw new Error(`${label} does not accept training or Dropout RNG options.`);
    }
  }
}

export function createBackendCapabilities({
  incrementalExecution = false,
  incrementalRows = false,
  sequenceMajorRows = false,
  outputLocation = 'host',
}: BackendCapabilityOptions = {}): Readonly<BackendCapabilities> {
  if (incrementalRows && !incrementalExecution) {
    throw new Error('incrementalRows requires incrementalExecution.');
  }
  if (!OUTPUT_LOCATIONS.has(outputLocation)) {
    throw new Error("Backend outputLocation must be 'host' or 'device'.");
  }
  if (sequenceMajorRows && !incrementalRows) {
    throw new Error('sequenceMajorRows requires incrementalRows.');
  }
  return Object.freeze({
    incrementalExecution: incrementalExecution === true,
    incrementalRows: incrementalRows === true,
    sequenceMajorRows: sequenceMajorRows === true,
    outputLocation,
  });
}

/** @internal Shared implementation base for VolvoxAI's built-in engines. */
export abstract class BackendEngine {
  backendName!: string;
  capabilities!: Readonly<BackendCapabilities>;
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
    this.backendName = name;
    this.capabilities = createBackendCapabilities(capabilities);
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
   * intermediates. A direct engine execution replaces the current session's
   * cache ownership.
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

/** @internal Validate the structural contract shared by built-in engines. */
export function assertBuiltInEngine(engine: any, label = 'Built-in backend'): BackendEngineLike {
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
  if (!engine || typeof engine.backendName !== 'string' ||
      !validCapabilities || !validIncrementalLifecycle ||
      typeof engine.allocateGraph !== 'function' || typeof engine.execute !== 'function' ||
      typeof engine.createDecodeSession !== 'function') {
    throw new Error(`${label} does not implement the built-in engine lifecycle.`);
  }
  return engine;
}
