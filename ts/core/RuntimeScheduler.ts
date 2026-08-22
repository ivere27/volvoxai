import { runtimeIdentity } from './Identity.js';
import { VolvoxAIError } from './RuntimeErrors.js';
import {
  registerExecutionResultCloseFinalizer,
  type ExecutionResult,
} from './ExecutionResult.js';
import type { ExecutionInputs, ExecutionOptions, RuntimeTypedArray } from '../types.js';
import {
  ExecutionMode,
  executionModes,
  type ExecutionModeValue,
} from '../generated/volvoxaiEnums.js';

const SCHEDULED_MODE = executionModes[ExecutionMode.Scheduled];

export type { ExecutionModeValue } from '../generated/volvoxaiEnums.js';
export type RequestFreshness = 'all' | 'latest' | 'drop-if-late';
export type ScheduledCompatibilityToken = string | object;
export type RuntimeRequestState =
  | 'queued'
  | 'submitted'
  | 'completed'
  | 'cancelled'
  | 'failed'
  | 'superseded';

export interface RuntimeSchedulerOptions {
  /** Requests which have been admitted but not completed. */
  readonly maxRequests?: number;
  /**
   * Logical retained-snapshot plus active-staging budget. A queued LATEST
   * replacement may transiently overlap its predecessor while the new copy is
   * made; this is not a process allocator high-water limit.
   */
  readonly maxInputBytes?: number;
  /** Runtime-wide cap applied in addition to each compiled batch contract. */
  readonly maxBatchSize?: number;
  /** Runtime-wide upper bound for scheduled micro-batch coalescing. */
  readonly maxBatchDelayMs?: number;
  /** Maximum eligible dispatches that may bypass a lower-priority route. */
  readonly maxPriorityBurst?: number;
}

export interface RuntimeResultBudgetOptions {
  /** Reserved future results plus published results that have not been closed. */
  readonly maxRetainedResults?: number;
  /** Exact logical output bytes reserved before provider execution. */
  readonly maxRetainedOutputBytes?: number;
}

export interface RuntimeExecutionConfiguration {
  readonly mode?: ExecutionModeValue;
  readonly scheduler?: RuntimeSchedulerOptions;
  /** Runtime-wide result ownership budget, including DIRECT and explicit contexts. */
  readonly results?: RuntimeResultBudgetOptions;
}

export interface RuntimeRunOptions extends ExecutionOptions {
  readonly mode?: ExecutionModeValue;
  readonly priority?: number;
  readonly deadlineMonotonicMs?: number;
  readonly freshness?: RequestFreshness;
  readonly streamKey?: string | null;
  readonly signal?: AbortSignal | null;
}

export interface RuntimeSubmitOptions extends ExecutionOptions {
  readonly priority?: number;
  readonly deadlineMonotonicMs?: number;
  readonly freshness?: RequestFreshness;
  readonly streamKey?: string | null;
  readonly signal?: AbortSignal | null;
}

export interface PreparedScheduledExecution {
  /** Provider-owned structural/opaque grouping identity. */
  readonly compatibilityToken: ScheduledCompatibilityToken;
  /** Provider-owned device/worker arbitration identity. */
  readonly resourceDomain: object;
  /** Provider-owned generation fence for device loss/recovery. */
  readonly deviceEpoch: object;
  /**
   * Frozen target-owned capability for the exact input snapshot prepared above.
   * The scheduler preserves identity and never inspects its resolved contents.
   */
  readonly preparedPlan: object;
  readonly maxBatchSize: number;
  readonly inputBytes: number;
  readonly outputBytes: number;
}

export interface ScheduledExecutionPreflight {
  /** Frozen target capability holding metadata-only semantic resolution. */
  readonly preparedPlan: object;
  readonly inputBytes: number;
  readonly outputBytes: number;
}

/** @internal Exact request handed back to the target after scheduler admission. */
export interface ScheduledExecutionRequest {
  readonly inputs: ExecutionInputs;
  readonly options: Readonly<ExecutionOptions>;
  readonly preparedPlan: object;
  readonly compatibilityToken: ScheduledCompatibilityToken;
  readonly resourceDomain: object;
  readonly deviceEpoch: object;
  readonly outputBytes: number;
}

/** @internal RuntimeScheduler talks only to this typed execution boundary. */
export interface ScheduledExecutionTarget {
  readonly backend: string;
  /**
   * Reserve the target's mutable scheduled route at admission time. This
   * synchronous claim prevents a later DIRECT call from barging ahead of work
   * that the scheduler has already accepted but has not dispatched yet.
   */
  reserveScheduledExecutionRoute?(): () => void;
  preflightScheduledExecution?(
    inputs: ExecutionInputs,
    options: Readonly<ExecutionOptions>,
  ): ScheduledExecutionPreflight;
  prepareScheduledExecution(
    inputs: ExecutionInputs,
    options: Readonly<ExecutionOptions>,
    preflightPlan?: object,
  ): PreparedScheduledExecution;
  executeScheduledBatch(
    requests: readonly Readonly<ScheduledExecutionRequest>[],
  ): Promise<readonly ExecutionResult[]>;
  closeScheduledExecutionRoute(): Promise<void>;
}

interface NormalizedSchedulerOptions {
  readonly maxRequests: number;
  readonly maxInputBytes: number;
  readonly maxBatchSize: number;
  readonly maxBatchDelayMs: number;
  readonly maxPriorityBurst: number;
}

interface PendingRequest {
  readonly handle: RuntimeRequestHandle;
  readonly target: ScheduledExecutionTarget;
  readonly routeToken: ScheduledCompatibilityToken;
  readonly resourceDomain: object;
  readonly deviceEpoch: object;
  readonly inputs: ExecutionInputs;
  readonly executionOptions: Readonly<ExecutionOptions>;
  readonly preparedPlan: object;
  readonly inputBytes: number;
  readonly outputBytes: number;
  readonly resultBudgetLease: RuntimeResultBudgetLease | null;
  readonly maxBatchSize: number;
  readonly priority: number;
  readonly deadline: number | null;
  readonly freshness: RequestFreshness;
  readonly streamKey: string | null;
  readonly arrival: number;
  readonly readyAt: number;
  readonly sequence: bigint;
  readonly resolve: (result: ExecutionResult) => void;
  readonly reject: (error: Error) => void;
  readonly abortSignal: AbortSignal | null;
  abortListener: (() => void) | null;
  releaseRouteClaim: (() => void) | null;
  settled: boolean;
  submitted: boolean;
  cancelCode: 'CANCELLED' | 'SUPERSEDED' | null;
  resultRetentionTransferred: boolean;
}

interface RouteState {
  readonly target: ScheduledExecutionTarget;
  readonly routeToken: ScheduledCompatibilityToken;
  readonly resourceDomain: object;
  readonly deviceEpoch: object;
  readonly queue: PendingRequest[];
  lastDispatch: bigint;
  bypassCount: number;
  inFlight: Promise<void> | null;
}

interface DomainState {
  active: boolean;
}

interface CapturedHostInput {
  readonly name: string;
  readonly data: RuntimeTypedArray;
  readonly shape: readonly number[];
  readonly rank: number;
}

interface CapturedHostInputs {
  readonly entries: readonly CapturedHostInput[];
  readonly inputBytes: number;
}

const DEFAULT_OPTIONS: NormalizedSchedulerOptions = Object.freeze({
  maxRequests: 256,
  maxInputBytes: 64 * 1024 * 1024,
  maxBatchSize: 32,
  maxBatchDelayMs: 0,
  maxPriorityBurst: 8,
});

const DEFAULT_RESULT_BUDGET_OPTIONS: Required<RuntimeResultBudgetOptions> = Object.freeze({
  maxRetainedResults: 64,
  maxRetainedOutputBytes: 64 * 1024 * 1024,
});

const MAX_TIMER_DELAY_MS = 2_147_483_647;
const REQUEST_PRIORITY_AGING_MS = 10;
const MAX_RUNTIME_REQUEST_PRIORITY = 1_000;
const MAX_STREAM_KEY_LENGTH = 1024;
const MAX_SCHEDULED_INPUTS = 1024;
const MAX_INPUT_NAME_LENGTH = 1024;
const MAX_INPUT_RANK = 1024;
const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint8Array.prototype) as object;
const TYPED_ARRAY_TAG = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  Symbol.toStringTag,
)?.get;
const TYPED_ARRAY_BUFFER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  'buffer',
)?.get;
const TYPED_ARRAY_BYTE_OFFSET = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  'byteOffset',
)?.get;
const TYPED_ARRAY_BYTE_LENGTH = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  'byteLength',
)?.get;
const TYPED_ARRAY_LENGTH = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  'length',
)?.get;
const ABORT_SIGNAL_ABORTED = typeof AbortSignal === 'undefined'
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, 'aborted')?.get;
const EVENT_TARGET_ADD_EVENT_LISTENER = typeof EventTarget === 'undefined'
  ? undefined
  : EventTarget.prototype.addEventListener;
const EVENT_TARGET_REMOVE_EVENT_LISTENER = typeof EventTarget === 'undefined'
  ? undefined
  : EventTarget.prototype.removeEventListener;

type RuntimeHostStorageKind =
  | 'float32'
  | 'int32'
  | 'int8'
  | 'uint8'
  | 'uint8-clamped';

interface RuntimeHostStorageInspection {
  readonly kind: RuntimeHostStorageKind;
  readonly buffer: ArrayBufferLike;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly length: number;
}

function runtimeHostStorageKind(value: unknown): RuntimeHostStorageKind | null {
  let tag: unknown;
  try {
    // The intrinsic %TypedArray%.prototype getter reads the internal typed-array
    // brand across realms and ignores a spoofed Symbol.toStringTag property.
    tag = TYPED_ARRAY_TAG?.call(value);
  } catch {
    return null;
  }
  if (tag === 'Float32Array') return 'float32';
  if (tag === 'Int32Array') return 'int32';
  if (tag === 'Int8Array') return 'int8';
  if (tag === 'Uint8Array') return 'uint8';
  if (tag === 'Uint8ClampedArray') return 'uint8-clamped';
  return null;
}

function inspectRuntimeHostStorage(value: unknown): RuntimeHostStorageInspection | null {
  const kind = runtimeHostStorageKind(value);
  if (kind === null) return null;
  try {
    const buffer = TYPED_ARRAY_BUFFER?.call(value) as ArrayBufferLike | undefined;
    const byteOffset = TYPED_ARRAY_BYTE_OFFSET?.call(value) as number | undefined;
    const byteLength = TYPED_ARRAY_BYTE_LENGTH?.call(value) as number | undefined;
    const length = TYPED_ARRAY_LENGTH?.call(value) as number | undefined;
    const bytesPerElement = kind === 'float32' || kind === 'int32' ? 4 : 1;
    if (!buffer || !Number.isSafeInteger(byteOffset) || (byteOffset as number) < 0 ||
        !Number.isSafeInteger(byteLength) || (byteLength as number) < 0 ||
        !Number.isSafeInteger(length) || (length as number) < 0 ||
        (length as number) > Math.floor(Number.MAX_SAFE_INTEGER / bytesPerElement) ||
        (length as number) * bytesPerElement !== byteLength) {
      return null;
    }
    return Object.freeze({
      kind,
      buffer,
      byteOffset: byteOffset as number,
      byteLength: byteLength as number,
      length: length as number,
    });
  } catch {
    return null;
  }
}

function canonicalRuntimeHostView(
  storage: RuntimeHostStorageInspection,
): RuntimeTypedArray {
  if (storage.kind === 'float32') {
    return new Float32Array(storage.buffer, storage.byteOffset, storage.length);
  }
  if (storage.kind === 'int32') {
    return new Int32Array(storage.buffer, storage.byteOffset, storage.length);
  }
  if (storage.kind === 'int8') {
    return new Int8Array(storage.buffer, storage.byteOffset, storage.length);
  }
  if (storage.kind === 'uint8-clamped') {
    return new Uint8ClampedArray(storage.buffer, storage.byteOffset, storage.length);
  }
  return new Uint8Array(storage.buffer, storage.byteOffset, storage.length);
}

function isScheduledCompatibilityToken(value: unknown): value is ScheduledCompatibilityToken {
  return (typeof value === 'string' && value.length > 0) ||
    (value !== null && typeof value === 'object' && Object.isFrozen(value));
}

function monotonicMilliseconds(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

/** @internal Read/register AbortSignal state without invoking shadowable instance properties. */
export function abortSignalIsAborted(signal: AbortSignal): boolean {
  if (!ABORT_SIGNAL_ABORTED) {
    throw new VolvoxAIError('ABI_UNSUPPORTED', 'AbortSignal intrinsic is unavailable.', {
      phase: 'execution',
    });
  }
  return Boolean(ABORT_SIGNAL_ABORTED.call(signal));
}

/** @internal Register through the captured EventTarget intrinsic. */
export function addAbortSignalListener(signal: AbortSignal, listener: () => void): void {
  if (!EVENT_TARGET_ADD_EVENT_LISTENER) {
    throw new VolvoxAIError('ABI_UNSUPPORTED', 'EventTarget intrinsic is unavailable.', {
      phase: 'execution',
    });
  }
  EVENT_TARGET_ADD_EVENT_LISTENER.call(signal, 'abort', listener, { once: true });
}

/** @internal Remove through the captured EventTarget intrinsic. */
export function removeAbortSignalListener(signal: AbortSignal, listener: () => void): void {
  if (!EVENT_TARGET_REMOVE_EVENT_LISTENER) {
    throw new VolvoxAIError('ABI_UNSUPPORTED', 'EventTarget intrinsic is unavailable.', {
      phase: 'execution',
    });
  }
  EVENT_TARGET_REMOVE_EVENT_LISTENER.call(signal, 'abort', listener);
}

function boundedInteger(value: number | undefined, fallback: number, label: string): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0) {
    throw new VolvoxAIError('INVALID_ARGUMENT', `${label} must be a positive safe integer.`, {
      phase: 'initialization',
    });
  }
  return resolved;
}

function normalizeResultBudgetOptions(
  options: RuntimeResultBudgetOptions | undefined,
): Readonly<Required<RuntimeResultBudgetOptions>> {
  if (options !== undefined && (!options || typeof options !== 'object' || Array.isArray(options))) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime result budget options must be an object.', {
      phase: 'initialization',
    });
  }
  let maxRetainedResults: number | undefined;
  let maxRetainedOutputBytes: number | undefined;
  try {
    maxRetainedResults = options?.maxRetainedResults;
    maxRetainedOutputBytes = options?.maxRetainedOutputBytes;
  } catch (error) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Could not snapshot Runtime result budget.', {
      phase: 'initialization', cause: error,
    });
  }
  return Object.freeze({
    maxRetainedResults: boundedInteger(
      maxRetainedResults,
      DEFAULT_RESULT_BUDGET_OPTIONS.maxRetainedResults,
      'Runtime result budget maxRetainedResults',
    ),
    maxRetainedOutputBytes: boundedInteger(
      maxRetainedOutputBytes,
      DEFAULT_RESULT_BUDGET_OPTIONS.maxRetainedOutputBytes,
      'Runtime result budget maxRetainedOutputBytes',
    ),
  });
}

/** @internal Split lease so a B>1 shared backing keeps bytes until its final lane closes. */
export interface RuntimeResultBudgetLease {
  readonly outputBytes: number;
  readonly hasResultSlot: boolean;
  acquireResultSlot(backend: string): void;
  transferResultSlotFrom(previous: RuntimeResultBudgetLease): void;
  releaseResultSlot(): void;
  releaseOutputBytes(): void;
  releaseAll(): void;
}

const RESULT_BUDGET_LEASE_STATES = new WeakMap<RuntimeResultBudgetLease, Readonly<{
  ledger: RuntimeResultBudget;
  hasResultSlot(): boolean;
  relinquishResultSlot(): void;
}>>();

/** @internal Runtime-owned output reservation ledger; it is not scheduler state. */
export class RuntimeResultBudget {
  readonly #options: Readonly<Required<RuntimeResultBudgetOptions>>;
  #retainedResults = 0;
  #retainedOutputBytes = 0;

  constructor(options?: RuntimeResultBudgetOptions) {
    this.#options = normalizeResultBudgetOptions(options);
  }

  get retainedResults(): number { return this.#retainedResults; }
  get retainedOutputBytes(): number { return this.#retainedOutputBytes; }
  get maxRetainedResults(): number { return this.#options.maxRetainedResults; }
  get maxRetainedOutputBytes(): number { return this.#options.maxRetainedOutputBytes; }

  reserve(
    outputBytes: number,
    backend: string,
    deferResultSlot = false,
  ): RuntimeResultBudgetLease {
    if (!Number.isSafeInteger(outputBytes) || outputBytes < 0) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `Backend '${backend}' reported an invalid exact output byte count.`, {
          phase: 'execution', backend,
        });
    }
    if ((!deferResultSlot && this.#retainedResults >= this.#options.maxRetainedResults) ||
        outputBytes > this.#options.maxRetainedOutputBytes - this.#retainedOutputBytes) {
      throw new VolvoxAIError('OVERLOADED', 'Runtime retained-result budget is full.', {
        phase: 'execution', backend,
        report: Object.freeze({
          maxRetainedResults: this.#options.maxRetainedResults,
          maxRetainedOutputBytes: this.#options.maxRetainedOutputBytes,
          retainedResults: this.#retainedResults,
          retainedOutputBytes: this.#retainedOutputBytes,
          requestedOutputBytes: outputBytes,
        }),
      });
    }
    if (!deferResultSlot) this.#retainedResults++;
    this.#retainedOutputBytes += outputBytes;
    let hasResultSlot = !deferResultSlot;
    let bytesReleased = false;
    const releaseResultSlot = () => {
      if (!hasResultSlot) return;
      hasResultSlot = false;
      this.#retainedResults--;
    };
    const releaseOutputBytes = () => {
      if (bytesReleased) return;
      bytesReleased = true;
      this.#retainedOutputBytes -= outputBytes;
    };
    const lease: RuntimeResultBudgetLease = {
      outputBytes,
      get hasResultSlot() { return hasResultSlot; },
      acquireResultSlot: (requestedBackend: string) => {
        if (hasResultSlot) return;
        if (this.#retainedResults >= this.#options.maxRetainedResults) {
          throw new VolvoxAIError('OVERLOADED', 'Runtime retained-result count is full.', {
            phase: 'execution', backend: requestedBackend,
          });
        }
        hasResultSlot = true;
        this.#retainedResults++;
      },
      transferResultSlotFrom: (previous: RuntimeResultBudgetLease) => {
        const state = RESULT_BUDGET_LEASE_STATES.get(previous);
        if (hasResultSlot || !state || state.ledger !== this || !state.hasResultSlot()) {
          throw new VolvoxAIError('ABI_UNSUPPORTED',
            'Runtime result-slot transfer did not carry an exact live reservation.', {
              phase: 'execution', backend,
            });
        }
        state.relinquishResultSlot();
        hasResultSlot = true;
      },
      releaseResultSlot,
      releaseOutputBytes,
      releaseAll: () => {
        releaseResultSlot();
        releaseOutputBytes();
      },
    };
    RESULT_BUDGET_LEASE_STATES.set(lease, {
      ledger: this,
      hasResultSlot: () => hasResultSlot,
      relinquishResultSlot: () => { hasResultSlot = false; },
    });
    return Object.freeze(lease);
  }
}

function normalizeSchedulerOptions(
  options: RuntimeSchedulerOptions | undefined,
): NormalizedSchedulerOptions {
  if (options !== undefined && (!options || typeof options !== 'object' || Array.isArray(options))) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime scheduler options must be an object.', {
      phase: 'initialization',
    });
  }
  const maxBatchDelayMs = options?.maxBatchDelayMs ?? DEFAULT_OPTIONS.maxBatchDelayMs;
  if (!Number.isFinite(maxBatchDelayMs) || maxBatchDelayMs < 0 || maxBatchDelayMs > 60_000) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Runtime scheduler maxBatchDelayMs must be finite and between 0 and 60000.', {
        phase: 'initialization',
      });
  }
  return Object.freeze({
    maxRequests: boundedInteger(
      options?.maxRequests, DEFAULT_OPTIONS.maxRequests, 'Runtime scheduler maxRequests',
    ),
    maxInputBytes: boundedInteger(
      options?.maxInputBytes, DEFAULT_OPTIONS.maxInputBytes, 'Runtime scheduler maxInputBytes',
    ),
    maxBatchSize: boundedInteger(
      options?.maxBatchSize, DEFAULT_OPTIONS.maxBatchSize, 'Runtime scheduler maxBatchSize',
    ),
    maxBatchDelayMs,
    maxPriorityBurst: boundedInteger(
      options?.maxPriorityBurst,
      DEFAULT_OPTIONS.maxPriorityBurst,
      'Runtime scheduler maxPriorityBurst',
    ),
  });
}

function executionOptions(options: RuntimeSubmitOptions): Readonly<ExecutionOptions> {
  const result: {
    adapter?: ExecutionOptions['adapter'];
    adapters?: ExecutionOptions['adapters'];
  } = {};
  if (Object.prototype.hasOwnProperty.call(options, 'adapter')) result.adapter = options.adapter;
  if (Object.prototype.hasOwnProperty.call(options, 'adapters')) result.adapters = options.adapters;
  return Object.freeze(result);
}

/** @internal Capture every externally observable submit option exactly once. */
export function captureRuntimeSubmitOptions(
  value: RuntimeSubmitOptions,
): Readonly<RuntimeSubmitOptions> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime submit options must be an object.', {
      phase: 'execution',
    });
  }
  try {
    const result: {
      priority?: number;
      deadlineMonotonicMs?: number;
      freshness?: RequestFreshness;
      streamKey?: string | null;
      signal?: AbortSignal | null;
      adapter?: ExecutionOptions['adapter'];
      adapters?: ExecutionOptions['adapters'];
    } = {
      priority: value.priority,
      deadlineMonotonicMs: value.deadlineMonotonicMs,
      freshness: value.freshness,
      streamKey: value.streamKey,
      signal: value.signal,
    };
    if (Object.prototype.hasOwnProperty.call(value, 'adapter')) result.adapter = value.adapter;
    if (Object.prototype.hasOwnProperty.call(value, 'adapters')) result.adapters = value.adapters;
    return Object.freeze(result);
  } catch (error) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Could not snapshot Runtime submit options.', {
      phase: 'execution', cause: error,
    });
  }
}

function validateRequestOptions(options: RuntimeSubmitOptions): void {
  if (options.priority !== undefined &&
      (!Number.isSafeInteger(options.priority) || options.priority < -MAX_RUNTIME_REQUEST_PRIORITY ||
        options.priority > MAX_RUNTIME_REQUEST_PRIORITY)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Runtime request priority must be a safe integer between -1000 and 1000.', {
        phase: 'execution',
      });
  }
  if (options.deadlineMonotonicMs !== undefined &&
      (!Number.isFinite(options.deadlineMonotonicMs) || options.deadlineMonotonicMs < 0)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Runtime request deadlineMonotonicMs must be a non-negative finite number.', {
        phase: 'execution',
      });
  }
  if (options.freshness !== undefined && options.freshness !== 'all' &&
      options.freshness !== 'latest' && options.freshness !== 'drop-if-late') {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime request freshness is invalid.', {
      phase: 'execution',
    });
  }
  if (options.streamKey !== undefined && options.streamKey !== null &&
      (typeof options.streamKey !== 'string' || options.streamKey.length === 0 ||
        options.streamKey.length > MAX_STREAM_KEY_LENGTH)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      `Runtime request streamKey must contain 1-${MAX_STREAM_KEY_LENGTH} UTF-16 units or be null.`, {
        phase: 'execution',
      });
  }
  if (options.freshness === 'latest' && !options.streamKey) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Runtime request freshness latest requires streamKey.', { phase: 'execution' });
  }
  if (options.freshness === 'drop-if-late' && options.deadlineMonotonicMs === undefined) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Runtime request freshness drop-if-late requires deadlineMonotonicMs.', {
        phase: 'execution',
      });
  }
  if (options.signal !== undefined && options.signal !== null &&
      !(typeof AbortSignal !== 'undefined' && options.signal instanceof AbortSignal)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Runtime request signal must be an AbortSignal or null.', { phase: 'execution' });
  }
}

function captureHostInputs(inputs: ExecutionInputs): CapturedHostInputs {
  const entries: CapturedHostInput[] = [];
  let inputBytes = 0;
  let enumeratedInputs = 0;
  try {
    if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Scheduled inputs must be a named tensor-view record.', { phase: 'execution' });
    }
    for (const name in inputs) {
      enumeratedInputs++;
      if (enumeratedInputs > MAX_SCHEDULED_INPUTS) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Scheduled inputs may enumerate at most ${MAX_SCHEDULED_INPUTS} names.`, {
            phase: 'execution',
          });
      }
      if (!Object.prototype.hasOwnProperty.call(inputs, name)) continue;
      if (name.length > MAX_INPUT_NAME_LENGTH) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Scheduled input names may contain at most ${MAX_INPUT_NAME_LENGTH} UTF-16 units.`, {
            phase: 'execution',
          });
      }
      const view = inputs[name];
      if (!view || typeof view !== 'object') {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Scheduled input '${name}' must be a tensor view.`, { phase: 'execution' });
      }
      const data = view.data;
      const shape = view.shape;
      if (!ArrayBuffer.isView(data)) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          'Scheduled device inputs require a provider-owned transferable input lease.', {
            phase: 'execution',
          });
      }
      const storage = inspectRuntimeHostStorage(data);
      if (storage === null) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Scheduled input '${name}' uses an unsupported host typed array.`, {
            phase: 'execution',
          });
      }
      if (!Array.isArray(shape)) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Scheduled input '${name}' shape must be an array.`, { phase: 'execution' });
      }
      const rank = shape.length;
      if (!Number.isSafeInteger(rank) || rank < 0 || rank > MAX_INPUT_RANK) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Scheduled input '${name}' rank must be at most ${MAX_INPUT_RANK}.`, {
            phase: 'execution',
          });
      }
      if (storage.byteLength > Number.MAX_SAFE_INTEGER - inputBytes) {
        throw new VolvoxAIError('OUT_OF_MEMORY',
          'Scheduled input byte count overflowed.', { phase: 'execution' });
      }
      inputBytes += storage.byteLength;
      // Canonicalize only the view/prototype, not its payload. This lets the
      // metadata-only shape/provider preflight consume cross-realm views while
      // preserving the bounded-copy admission order below.
      entries.push(Object.freeze({
        name,
        data: canonicalRuntimeHostView(storage),
        shape,
        rank,
      }));
    }
  } catch (error) {
    if (error instanceof VolvoxAIError) throw error;
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Could not inspect scheduled input storage.', {
      phase: 'execution', cause: error,
    });
  }
  return Object.freeze({ entries: Object.freeze(entries), inputBytes });
}

function captureHostInputShapes(captured: CapturedHostInputs): CapturedHostInputs {
  try {
    const entries = captured.entries.map(({ name, data, shape, rank }) => {
      const ownedShape = new Array<number>(rank);
      for (let axis = 0; axis < rank; axis++) ownedShape[axis] = shape[axis];
      return Object.freeze({ name, data, shape: Object.freeze(ownedShape), rank });
    });
    return Object.freeze({ entries: Object.freeze(entries), inputBytes: captured.inputBytes });
  } catch (error) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Could not snapshot scheduled input shape metadata.', {
        phase: 'execution', cause: error,
      });
  }
}

function capturedInputViews(captured: CapturedHostInputs): ExecutionInputs {
  return Object.freeze(Object.fromEntries(captured.entries.map(({ name, data, shape }) => [
    name,
    Object.freeze({ data, shape }),
  ])));
}

function copyHostData(data: RuntimeTypedArray): RuntimeTypedArray {
  const storage = inspectRuntimeHostStorage(data);
  if (storage === null) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Scheduled input storage changed to an unsupported typed array.', {
        phase: 'execution',
      });
  }
  const source = canonicalRuntimeHostView(storage);
  if (storage.kind === 'float32') return new Float32Array(source as Float32Array);
  if (storage.kind === 'int32') return new Int32Array(source as Int32Array);
  if (storage.kind === 'int8') return new Int8Array(source as Int8Array);
  if (storage.kind === 'uint8-clamped') {
    return new Uint8ClampedArray(source as Uint8ClampedArray);
  }
  return new Uint8Array(source as Uint8Array);
}

function cloneInputs(captured: CapturedHostInputs): ExecutionInputs {
  const cloned = Object.create(null) as Record<string, ExecutionInputs[string]>;
  try {
    const ownedData: RuntimeTypedArray[] = [];
    let ownedInputBytes = 0;
    for (const { data } of captured.entries) {
      const copy = copyHostData(data);
      const byteLength = TYPED_ARRAY_BYTE_LENGTH?.call(copy) as number | undefined;
      if (typeof byteLength !== 'number' || !Number.isSafeInteger(byteLength) ||
          byteLength < 0 || byteLength > Number.MAX_SAFE_INTEGER - ownedInputBytes) {
        throw new VolvoxAIError('OUT_OF_MEMORY',
          'Scheduled input snapshot byte count overflowed.', { phase: 'execution' });
      }
      ownedInputBytes += byteLength;
      ownedData.push(copy);
    }
    if (ownedInputBytes !== captured.inputBytes) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Scheduled input storage changed size while it was being admitted.', {
          phase: 'execution',
        });
    }
    for (let index = 0; index < captured.entries.length; index++) {
      const { name, shape } = captured.entries[index];
      cloned[name] = Object.freeze({
        data: ownedData[index],
        shape,
      });
    }
  } catch (error) {
    if (error instanceof VolvoxAIError) throw error;
    throw new VolvoxAIError('OUT_OF_MEMORY', 'Could not snapshot scheduled input storage.', {
      phase: 'execution', cause: error,
    });
  }
  return Object.freeze(cloned);
}

function requestError(
  code: 'CANCELLED' | 'SUPERSEDED' | 'DEADLINE_EXCEEDED' | 'HANDLE_DISPOSED',
  message: string,
  backend: string,
): VolvoxAIError {
  return new VolvoxAIError(code, message, { phase: 'execution', backend });
}

export class RuntimeRequestHandle {
  readonly id: string;
  readonly result: Promise<ExecutionResult>;
  #state: RuntimeRequestState = 'queued';
  #deadlineMissed: boolean | null = null;
  #cancel: (() => boolean) | null = null;

  /** @internal */
  constructor(result: Promise<ExecutionResult>) {
    this.id = runtimeIdentity('request');
    this.result = result;
  }

  get state(): RuntimeRequestState { return this.#state; }
  /** Null while nonterminal; terminal requests report their completion deadline outcome. */
  get deadlineMissed(): boolean | null { return this.#deadlineMissed; }
  wait(): Promise<ExecutionResult> { return this.result; }
  cancel(): boolean { return this.#cancel?.() ?? false; }

  /** @internal */
  _bindCancel(cancel: () => boolean): void { this.#cancel = cancel; }
  /** @internal Release the request/input graph after terminal settlement. */
  _clearCancel(): void { this.#cancel = null; }
  /** @internal */
  _setState(state: RuntimeRequestState): void { this.#state = state; }
  /** @internal */
  _setDeadlineMissed(value: boolean): void { this.#deadlineMissed = value; }
}

/**
 * One Runtime-owned actor for admission and route arbitration.
 *
 * It intentionally contains no public callback execution seam. A compiled
 * target supplies the typed batch operation and opaque route identities.
 */
export class RuntimeScheduler {
  readonly #options: NormalizedSchedulerOptions;
  readonly #resultBudget: RuntimeResultBudget | null;
  readonly #routesByTarget = new Map<
    ScheduledExecutionTarget,
    Map<ScheduledCompatibilityToken, RouteState>
  >();
  readonly #domains = new Map<object, DomainState>();
  readonly #dispatches = new Set<Promise<void>>();
  readonly #submittedRequests = new Set<PendingRequest>();
  readonly #retirements = new WeakMap<ScheduledExecutionTarget, Promise<void>>();
  readonly #activeRetirements = new Set<Promise<void>>();
  #state: 'open' | 'closing' | 'closed' = 'open';
  #requestCount = 0;
  #inputBytes = 0;
  #admittingInputBytes = 0;
  #stagingInputBytes = 0;
  #inputSnapshotCopies = 0;
  #sequence = 0n;
  #dispatchSequence = 0n;
  #drainQueued = false;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #timerWakeAt: number | null = null;
  #closePromise: Promise<void> | null = null;

  constructor(options?: RuntimeSchedulerOptions, resultBudget?: RuntimeResultBudget) {
    this.#options = normalizeSchedulerOptions(options);
    this.#resultBudget = resultBudget ?? null;
  }

  get requestCount(): number { return this.#requestCount; }
  get inputBytes(): number { return this.#inputBytes; }
  get admittingInputBytes(): number { return this.#admittingInputBytes; }
  get stagingInputBytes(): number { return this.#stagingInputBytes; }
  /** Current logical accounting, not a process allocator high-water reading. */
  get totalInputBytes(): number {
    return this.#inputBytes + this.#admittingInputBytes + this.#stagingInputBytes;
  }
  get inputSnapshotCopies(): number { return this.#inputSnapshotCopies; }

  submit(
    target: ScheduledExecutionTarget,
    inputs: ExecutionInputs,
    suppliedOptions: RuntimeSubmitOptions = {},
  ): RuntimeRequestHandle {
    if (this.#state !== 'open' || this.#retirements.has(target)) {
      throw new VolvoxAIError('HANDLE_DISPOSED',
        'Runtime scheduler or compiled target is closing.', {
          phase: 'lifecycle', backend: target.backend,
        });
    }
    const options = captureRuntimeSubmitOptions(suppliedOptions);
    this.#assertAcceptingTarget(target);
    validateRequestOptions(options);
    const now = monotonicMilliseconds();
    if (options.deadlineMonotonicMs !== undefined && options.deadlineMonotonicMs <= now) {
      throw requestError('DEADLINE_EXCEEDED',
        'Runtime request deadline has already elapsed.', target.backend);
    }
    if (options.signal && abortSignalIsAborted(options.signal)) {
      throw requestError('CANCELLED', 'Runtime request was already aborted.', target.backend);
    }

    const effectiveOptions = executionOptions(options);
    const freshness = options.freshness ?? 'all';
    const streamKey = options.streamKey ?? null;
    // Capture each external getter once and account exact host bytes before any
    // potentially large typed-array allocation.
    const capturedStorage = captureHostInputs(inputs);
    this.#assertAcceptingTarget(target);
    if (options.signal && abortSignalIsAborted(options.signal)) {
      throw requestError('CANCELLED', 'Runtime request was aborted while inputs were inspected.',
        target.backend);
    }
    if (options.deadlineMonotonicMs !== undefined &&
        options.deadlineMonotonicMs <= monotonicMilliseconds()) {
      throw requestError('DEADLINE_EXCEEDED',
        'Runtime request deadline elapsed while inputs were inspected.', target.backend);
    }
    let replacements = freshness === 'latest'
      ? this.#queuedStreamRequests(target, streamKey)
      : [];
    this.#assertAdmissionCapacity(target, capturedStorage.inputBytes, replacements);

    // Snapshot only bounded metadata, then let the target reject names,
    // dtypes, and shapes against zero-copy captured views before payload copy.
    const capturedInputs = captureHostInputShapes(capturedStorage);
    let preflightPlan: object | undefined;
    let preflightOutputBytes: number | undefined;
    if (target.preflightScheduledExecution) {
      const preflight = target.preflightScheduledExecution(
        capturedInputViews(capturedInputs),
        effectiveOptions,
      );
      const preflightInputBytes = preflight?.inputBytes;
      const candidateOutputBytes = preflight?.outputBytes;
      const candidatePreflightPlan = preflight?.preparedPlan;
      if (!preflight || typeof preflight !== 'object' || !Object.isFrozen(preflight) ||
          preflightInputBytes !== capturedInputs.inputBytes ||
          (this.#resultBudget !== null &&
            (!Number.isSafeInteger(candidateOutputBytes) || candidateOutputBytes < 0)) ||
          !candidatePreflightPlan || typeof candidatePreflightPlan !== 'object' ||
          !Object.isFrozen(candidatePreflightPlan)) {
        throw new VolvoxAIError('ABI_UNSUPPORTED',
          `Backend '${target.backend}' returned an invalid scheduled metadata preflight.`, {
            phase: 'execution', backend: target.backend,
          });
      }
      preflightPlan = candidatePreflightPlan;
      preflightOutputBytes = candidateOutputBytes;
    } else if (this.#resultBudget !== null) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `Backend '${target.backend}' does not expose exact scheduled output preflight.`, {
          phase: 'execution', backend: target.backend,
        });
    }
    this.#assertAcceptingTarget(target);
    if (options.signal && abortSignalIsAborted(options.signal)) {
      throw requestError('CANCELLED',
        'Runtime request was aborted while its metadata was being prepared.', target.backend);
    }
    if (options.deadlineMonotonicMs !== undefined &&
        options.deadlineMonotonicMs <= monotonicMilliseconds()) {
      throw requestError('DEADLINE_EXCEEDED',
        'Runtime request deadline elapsed while its metadata was being prepared.', target.backend);
    }
    replacements = freshness === 'latest'
      ? this.#queuedStreamRequests(target, streamKey)
      : [];
    this.#assertAdmissionCapacity(target, capturedInputs.inputBytes, replacements);

    const resultBudgetLease = this.#resultBudget?.reserve(
      preflightOutputBytes!, target.backend, replacements.length > 0,
    ) ?? null;
    const inputBytes = capturedInputs.inputBytes;
    this.#admittingInputBytes += inputBytes;
    let admittingInputReserved = true;
    let resultLeaseCommitted = false;
    try {
    const ownedInputs = cloneInputs(capturedInputs);
    this.#inputSnapshotCopies++;
    this.#assertAcceptingTarget(target);
    if (options.signal && abortSignalIsAborted(options.signal)) {
      throw requestError('CANCELLED', 'Runtime request was aborted while being admitted.',
        target.backend);
    }
    if (options.deadlineMonotonicMs !== undefined &&
        options.deadlineMonotonicMs <= monotonicMilliseconds()) {
      throw requestError('DEADLINE_EXCEEDED',
        'Runtime request deadline elapsed while inputs were being admitted.', target.backend);
    }

    const prepared = target.prepareScheduledExecution(
      ownedInputs,
      effectiveOptions,
      preflightPlan,
    );
    const reportedInputBytes = prepared.inputBytes;
    const reportedOutputBytes = prepared.outputBytes;
    const preparedMaxBatchSize = prepared.maxBatchSize;
    const routeToken = prepared.compatibilityToken;
    const preparedPlan = prepared.preparedPlan;
    const resourceDomain = prepared.resourceDomain;
    const deviceEpoch = prepared.deviceEpoch;
    if (reportedInputBytes !== inputBytes ||
        (this.#resultBudget !== null && reportedOutputBytes !== preflightOutputBytes) ||
        !Number.isSafeInteger(preparedMaxBatchSize) || preparedMaxBatchSize <= 0 ||
        !isScheduledCompatibilityToken(routeToken) ||
        !preparedPlan || typeof preparedPlan !== 'object' || !Object.isFrozen(preparedPlan) ||
        !resourceDomain || typeof resourceDomain !== 'object' || !Object.isFrozen(resourceDomain) ||
        !deviceEpoch || typeof deviceEpoch !== 'object' || !Object.isFrozen(deviceEpoch)) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `Backend '${target.backend}' returned an invalid scheduled route contract.`, {
          phase: 'execution', backend: target.backend,
        });
    }
    this.#assertAcceptingTarget(target);
    if (options.signal && abortSignalIsAborted(options.signal)) {
      throw requestError('CANCELLED', 'Runtime request was aborted while its route was prepared.',
        target.backend);
    }
    if (options.deadlineMonotonicMs !== undefined &&
        options.deadlineMonotonicMs <= monotonicMilliseconds()) {
      throw requestError('DEADLINE_EXCEEDED',
        'Runtime request deadline elapsed while its route was prepared.', target.backend);
    }
    replacements = freshness === 'latest'
      ? this.#queuedStreamRequests(target, streamKey)
      : [];
    this.#assertAdmissionCapacity(target, 0, replacements);

    let resolve!: (result: ExecutionResult) => void;
    let reject!: (error: Error) => void;
    const result = new Promise<ExecutionResult>((accept, refuse) => {
      resolve = accept;
      reject = refuse;
    });
    // Cancellation, deadlines, and latest-frame replacement are scheduler
    // initiated. Keep an intentionally unobserved handle from becoming a host
    // process unhandled rejection; awaiting `result` still observes rejection.
    void result.catch(() => undefined);
    const handle = new RuntimeRequestHandle(result);
    const deadline = options.deadlineMonotonicMs ?? null;
    const batchingReadyAt = now + this.#options.maxBatchDelayMs;
    // Until route T(1) is measured, consuming the entire remaining deadline
    // as batching slack is unsafe. Dispatch immediately when it constrains the
    // normal batching window (future: deadline - estimated execution cost).
    const readyAt = deadline !== null && deadline <= batchingReadyAt
      ? now
      : batchingReadyAt;
    this.#sequence += 1n;
    const request: PendingRequest = {
      handle,
      target,
      routeToken,
      resourceDomain,
      deviceEpoch,
      inputs: ownedInputs,
      executionOptions: effectiveOptions,
      preparedPlan,
      inputBytes,
      outputBytes: preflightOutputBytes ?? 0,
      resultBudgetLease,
      maxBatchSize: Math.min(preparedMaxBatchSize, this.#options.maxBatchSize),
      priority: options.priority ?? 0,
      deadline,
      freshness,
      streamKey,
      arrival: now,
      readyAt,
      sequence: this.#sequence,
      resolve,
      reject,
      abortSignal: options.signal ?? null,
      abortListener: null,
      releaseRouteClaim: null,
      settled: false,
      submitted: false,
      cancelCode: null,
      resultRetentionTransferred: false,
    };
    handle._bindCancel(() => this.#cancelRequest(request, 'CANCELLED'));

    const rawReleaseRouteClaim = target.reserveScheduledExecutionRoute?.() ?? null;
    if (rawReleaseRouteClaim !== null && typeof rawReleaseRouteClaim !== 'function') {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `Backend '${target.backend}' returned an invalid scheduled route claim.`, {
          phase: 'execution', backend: target.backend,
        });
    }
    if (rawReleaseRouteClaim !== null) {
      let claimReleased = false;
      request.releaseRouteClaim = () => {
        if (claimReleased) return;
        claimReleased = true;
        try {
          rawReleaseRouteClaim();
        } catch {
          // A target cleanup defect must not strand an already terminal handle.
        }
      };
    }
    let route: RouteState | undefined;
    try {
      // The synchronous target claim is an internal reentrancy boundary. It may
      // have closed/retired this actor or admitted competing work, so repeat
      // every volatile admission check before publishing the request.
      this.#assertAcceptingTarget(target);
      if (options.signal && abortSignalIsAborted(options.signal)) {
        throw requestError('CANCELLED',
          'Runtime request was aborted while its route was being claimed.', target.backend);
      }
      if (deadline !== null && deadline <= monotonicMilliseconds()) {
        throw requestError('DEADLINE_EXCEEDED',
          'Runtime request deadline elapsed while its route was being claimed.', target.backend);
      }
      replacements = freshness === 'latest'
        ? this.#queuedStreamRequests(target, streamKey)
        : [];
      this.#assertAdmissionCapacity(target, 0, replacements);

      const routes = this.#routesByTarget.get(target) ??
        new Map<ScheduledCompatibilityToken, RouteState>();
      if (!this.#routesByTarget.has(target)) this.#routesByTarget.set(target, routes);
      route = routes.get(routeToken);
      if (!route) {
        route = {
          target,
          routeToken,
          resourceDomain,
          deviceEpoch,
          queue: [],
          lastDispatch: 0n,
          bypassCount: 0,
          inFlight: null,
        };
        routes.set(routeToken, route);
      } else if (route.resourceDomain !== resourceDomain || route.deviceEpoch !== deviceEpoch) {
        throw new VolvoxAIError('ABI_UNSUPPORTED',
          `Backend '${target.backend}' changed a scheduled route's resource domain or device epoch.`, {
            phase: 'execution', backend: target.backend,
          });
      }
      if (!this.#domains.has(resourceDomain)) {
        this.#domains.set(resourceDomain, { active: false });
      }

      route.queue.push(request);
      this.#requestCount++;
      this.#admittingInputBytes -= request.inputBytes;
      admittingInputReserved = false;
      this.#inputBytes += request.inputBytes;
      if (request.abortSignal) {
        request.abortListener = () => { this.#cancelRequest(request, 'CANCELLED'); };
        addAbortSignalListener(request.abortSignal, request.abortListener);
        // Registration is an external platform boundary even though instance
        // overrides are bypassed. Never acquire/steal a LATEST result slot after
        // a synchronous cancellation has already released this admission.
        if (abortSignalIsAborted(request.abortSignal)) {
          this.#cancelRequest(request, 'CANCELLED');
        }
        if (request.settled) {
          resultLeaseCommitted = true;
          return handle;
        }
      }
      if (resultBudgetLease !== null && !resultBudgetLease.hasResultSlot) {
        const donor = replacements.find((replacement) =>
          !replacement.settled && !replacement.submitted &&
          replacement.resultBudgetLease?.hasResultSlot);
        if (donor?.resultBudgetLease) {
          resultBudgetLease.transferResultSlotFrom(donor.resultBudgetLease);
        } else {
          resultBudgetLease.acquireResultSlot(target.backend);
        }
      }
    } catch (error) {
      if (request.abortSignal && request.abortListener) {
        try {
          removeAbortSignalListener(request.abortSignal, request.abortListener);
        } catch {
          // Preserve the admission failure while detaching any partial listener.
        }
        request.abortListener = null;
      }
      if (route) {
        const index = route.queue.indexOf(request);
        if (index >= 0) {
          route.queue.splice(index, 1);
          this.#requestCount--;
          this.#inputBytes -= request.inputBytes;
        }
        this.#deleteRouteIfIdle(route);
      }
      request.releaseRouteClaim?.();
      request.releaseRouteClaim = null;
      throw error;
    }
    for (const replacement of replacements) {
      this.#cancelRequest(replacement, 'SUPERSEDED');
    }
    if (request.freshness === 'latest') this.#supersedeSubmittedStream(request);
    this.#scheduleDrain();
    resultLeaseCommitted = true;
    return handle;
    } catch (error) {
      if (admittingInputReserved) this.#admittingInputBytes -= inputBytes;
      if (!resultLeaseCommitted) resultBudgetLease?.releaseAll();
      throw error;
    }
  }

  retireTarget(target: ScheduledExecutionTarget): Promise<void> {
    const existing = this.#retirements.get(target);
    if (existing) return existing;
    // Defer the target boundary so the exact retirement promise is published
    // before a target can synchronously reenter retireTarget/close.
    const retirement = Promise.resolve().then(async () => {
      const routes = this.#routesByTarget.get(target);
      if (routes) {
        for (const route of routes.values()) {
          for (const request of [...route.queue]) this.#cancelRequest(request, 'CANCELLED');
        }
        const inFlight = Array.from(routes.values(), (route) => route.inFlight)
          .filter((pending): pending is Promise<void> => pending !== null);
        await Promise.allSettled(inFlight);
        this.#routesByTarget.delete(target);
      }
      await target.closeScheduledExecutionRoute();
    });
    this.#retirements.set(target, retirement);
    this.#activeRetirements.add(retirement);
    retirement.finally(() => this.#activeRetirements.delete(retirement)).catch(() => undefined);
    return retirement;
  }

  close(): Promise<void> {
    if (this.#closePromise) return this.#closePromise;
    this.#state = 'closing';
    if (this.#timer !== null) {
      clearTimeout(this.#timer);
      this.#timer = null;
      this.#timerWakeAt = null;
    }
    for (const routes of this.#routesByTarget.values()) {
      for (const route of routes.values()) {
        for (const request of [...route.queue]) this.#cancelRequest(request, 'CANCELLED');
      }
    }
    this.#closePromise = (async () => {
      await Promise.allSettled([...this.#dispatches]);
      const retirements = await Promise.allSettled([...this.#activeRetirements]);
      const targets = [...this.#routesByTarget.keys()];
      const closures = await Promise.allSettled(
        targets.map((target) => this.retireTarget(target)),
      );
      this.#routesByTarget.clear();
      this.#domains.clear();
      this.#state = 'closed';
      const failure = [...retirements, ...closures].find(
        (result): result is PromiseRejectedResult => result.status === 'rejected',
      );
      if (failure) throw failure.reason;
    })();
    return this.#closePromise;
  }

  #assertAcceptingTarget(target: ScheduledExecutionTarget): void {
    if (this.#state !== 'open' || this.#retirements.has(target)) {
      throw new VolvoxAIError('HANDLE_DISPOSED',
        'Runtime scheduler or compiled target is closing.', {
          phase: 'lifecycle', backend: target.backend,
        });
    }
  }

  #queuedStreamRequests(
    target: ScheduledExecutionTarget,
    streamKey: string | null,
  ): PendingRequest[] {
    if (streamKey === null) return [];
    const matches: PendingRequest[] = [];
    const routes = this.#routesByTarget.get(target);
    if (!routes) return matches;
    for (const route of routes.values()) {
      for (const request of route.queue) {
        if (!request.settled && !request.submitted && request.freshness === 'latest' &&
            request.streamKey === streamKey) {
          matches.push(request);
        }
      }
    }
    return matches;
  }

  #assertAdmissionCapacity(
    target: ScheduledExecutionTarget,
    inputBytes: number,
    replacements: readonly PendingRequest[] = [],
  ): void {
    // A queued LATEST predecessor remains physically owned until the new
    // snapshot has succeeded. Never discount it before that atomic commit.
    let retainedRequests = this.#requestCount;
    for (const request of replacements) {
      if (!request.settled && !request.submitted) retainedRequests--;
    }
    const retainedInputBytes = this.#inputBytes + this.#admittingInputBytes +
      this.#stagingInputBytes;
    if (retainedRequests >= this.#options.maxRequests ||
        inputBytes > this.#options.maxInputBytes - retainedInputBytes) {
      throw new VolvoxAIError('OVERLOADED',
        'Runtime scheduler input/request budget is full.', {
          phase: 'execution', backend: target.backend,
          report: Object.freeze({
            maxRequests: this.#options.maxRequests,
            maxInputBytes: this.#options.maxInputBytes,
            admittedRequests: this.#requestCount,
            admittedInputBytes: this.#inputBytes,
            admittingInputBytes: this.#admittingInputBytes,
            stagingInputBytes: this.#stagingInputBytes,
            totalInputBytes: retainedInputBytes,
          }),
        });
    }
  }

  #supersedeSubmittedStream(incoming: PendingRequest): void {
    if (incoming.streamKey === null) return;
    for (const request of [...this.#submittedRequests]) {
      if (request.target === incoming.target && request.freshness === 'latest' &&
          request.streamKey === incoming.streamKey) {
        this.#cancelRequest(request, 'SUPERSEDED');
      }
    }
  }

  #cancelRequest(
    request: PendingRequest,
    code: 'CANCELLED' | 'SUPERSEDED',
  ): boolean {
    if (request.settled) return false;
    if (request.submitted) {
      if (request.cancelCode !== null) return false;
      request.cancelCode = code;
      return true;
    }
    const routes = this.#routesByTarget.get(request.target);
    const route = routes?.get(request.routeToken);
    if (route) {
      const index = route.queue.indexOf(request);
      if (index >= 0) route.queue.splice(index, 1);
    }
    this.#settleRejected(request, requestError(
      code,
      code === 'SUPERSEDED'
        ? 'Runtime request was superseded by a newer queued frame.'
        : 'Runtime request was cancelled.',
      request.target.backend,
    ), code === 'SUPERSEDED' ? 'superseded' : 'cancelled');
    if (route) this.#deleteRouteIfIdle(route);
    this.#scheduleDrain();
    return true;
  }

  #scheduleDrain(): void {
    if (this.#state !== 'open' || this.#drainQueued) return;
    this.#drainQueued = true;
    queueMicrotask(() => {
      this.#drainQueued = false;
      this.#drain();
    });
  }

  #scheduleTimerAt(readyAt: number | null): void {
    if (this.#state !== 'open' || readyAt === null || !Number.isFinite(readyAt)) {
      if (this.#timer !== null) clearTimeout(this.#timer);
      this.#timer = null;
      this.#timerWakeAt = null;
      return;
    }
    const now = monotonicMilliseconds();
    const delay = Math.max(0, Math.min(MAX_TIMER_DELAY_MS, Math.ceil(readyAt - now)));
    const timerWakeAt = now + delay;
    if (this.#timer !== null && this.#timerWakeAt !== null &&
        this.#timerWakeAt <= timerWakeAt) return;
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timerWakeAt = timerWakeAt;
    this.#timer = setTimeout(() => {
      this.#timer = null;
      this.#timerWakeAt = null;
      this.#scheduleDrain();
    }, delay);
  }

  #drain(): void {
    if (this.#state !== 'open') return;
    const now = monotonicMilliseconds();
    // Deadlines remain live while their resource domain is executing unrelated
    // work. Expire every queue before considering domain availability.
    for (const routes of this.#routesByTarget.values()) {
      for (const route of routes.values()) this.#expireDeadlines(route, now);
    }
    for (const [domainIdentity, domain] of this.#domains) {
      if (domain.active) continue;
      const route = this.#selectRoute(domainIdentity, now);
      if (!route) continue;
      domain.active = true;
      this.#dispatch(route, domain, now);
    }

    let nextWake = Number.POSITIVE_INFINITY;
    for (const routes of this.#routesByTarget.values()) {
      for (const route of routes.values()) {
        for (const request of route.queue) {
          if (request.freshness === 'drop-if-late' && request.deadline !== null) {
            nextWake = Math.min(nextWake, request.deadline);
          }
        }
        const domain = this.#domains.get(route.resourceDomain);
        if (route.queue.length === 0 || route.inFlight !== null || domain?.active) continue;
        if (this.#routeReady(route, now)) continue;
        for (const request of route.queue) nextWake = Math.min(nextWake, request.readyAt);
      }
    }
    this.#scheduleTimerAt(Number.isFinite(nextWake) ? nextWake : null);
  }

  #selectRoute(resourceDomain: object, now: number): RouteState | null {
    const candidates: RouteState[] = [];
    for (const routes of this.#routesByTarget.values()) {
      for (const route of routes.values()) {
        if (route.resourceDomain !== resourceDomain || route.inFlight !== null ||
            route.queue.length === 0) continue;
        if (!this.#routeReady(route, now)) continue;
        candidates.push(route);
      }
    }
    candidates.sort((left, right) => {
      const leftStarved = left.bypassCount >= this.#options.maxPriorityBurst;
      const rightStarved = right.bypassCount >= this.#options.maxPriorityBurst;
      if (leftStarved !== rightStarved) return leftStarved ? -1 : 1;
      if (leftStarved && left.bypassCount !== right.bypassCount) {
        return right.bypassCount - left.bypassCount;
      }
      const leftHead = this.#bestRequest(left.queue, now);
      const rightHead = this.#bestRequest(right.queue, now);
      const leftPriority = this.#effectivePriority(leftHead, now);
      const rightPriority = this.#effectivePriority(rightHead, now);
      if (leftPriority !== rightPriority) return leftPriority > rightPriority ? -1 : 1;
      const leftDeadline = leftHead.deadline ?? Number.POSITIVE_INFINITY;
      const rightDeadline = rightHead.deadline ?? Number.POSITIVE_INFINITY;
      if (leftDeadline !== rightDeadline) return leftDeadline - rightDeadline;
      if (left.lastDispatch !== right.lastDispatch) {
        return left.lastDispatch < right.lastDispatch ? -1 : 1;
      }
      return leftHead.sequence < rightHead.sequence ? -1 : 1;
    });
    const selected = candidates[0] ?? null;
    if (selected) {
      selected.bypassCount = 0;
      for (let index = 1; index < candidates.length; index++) {
        candidates[index].bypassCount = Math.min(
          this.#options.maxPriorityBurst,
          candidates[index].bypassCount + 1,
        );
      }
    }
    return selected;
  }

  #routeReady(route: RouteState, now: number): boolean {
    const head = this.#bestRequest(route.queue, now);
    if (route.queue.length >= head.maxBatchSize) return true;
    return route.queue.some((request) => request.readyAt <= now);
  }

  #effectivePriority(request: PendingRequest, now: number): number {
    const elapsed = Math.max(0, now - request.arrival);
    const agePoints = Math.floor(elapsed / REQUEST_PRIORITY_AGING_MS);
    if (!Number.isSafeInteger(agePoints) ||
        agePoints > Number.MAX_SAFE_INTEGER - MAX_RUNTIME_REQUEST_PRIORITY) {
      return Number.MAX_SAFE_INTEGER;
    }
    return request.priority + agePoints;
  }

  #requestPrecedes(
    candidate: PendingRequest,
    current: PendingRequest,
    now: number,
  ): boolean {
    const candidatePriority = this.#effectivePriority(candidate, now);
    const currentPriority = this.#effectivePriority(current, now);
    if (candidatePriority !== currentPriority) return candidatePriority > currentPriority;
    const candidateDeadline = candidate.deadline ?? Number.POSITIVE_INFINITY;
    const currentDeadline = current.deadline ?? Number.POSITIVE_INFINITY;
    if (candidateDeadline !== currentDeadline) return candidateDeadline < currentDeadline;
    return candidate.sequence < current.sequence;
  }

  #bestRequest(queue: readonly PendingRequest[], now: number): PendingRequest {
    return queue.reduce((best, request) => {
      return this.#requestPrecedes(request, best, now) ? request : best;
    });
  }

  #expireDeadlines(route: RouteState, now: number): void {
    for (const request of [...route.queue]) {
      if (request.freshness === 'drop-if-late' &&
          request.deadline !== null && request.deadline <= now) {
        this.#removeQueued(route, request);
        this.#settleRejected(request, requestError(
          'DEADLINE_EXCEEDED', 'Runtime request deadline elapsed while queued.',
          request.target.backend,
        ), 'failed', now);
      }
    }
    this.#deleteRouteIfIdle(route);
  }

  #dispatch(route: RouteState, domain: DomainState, now: number): void {
    route.queue.sort((left, right) => {
      if (this.#requestPrecedes(left, right, now)) return -1;
      if (this.#requestPrecedes(right, left, now)) return 1;
      return 0;
    });
    // Avoid argument spreading: maxRequests/maxBatchSize are safe integers, not
    // JavaScript call-stack bounds. Only selected requests constrain this batch.
    let maxBatch = Math.min(route.queue.length, route.queue[0].maxBatchSize);
    for (let index = 1; index < maxBatch; index++) {
      maxBatch = Math.min(maxBatch, route.queue[index].maxBatchSize);
    }
    // B>1 duplicates every selected host snapshot into one stacked input.
    // Reserve those exact bytes inside the same global admission budget. If
    // insufficient, reduce deterministically; B=1 needs no staging copy.
    const stagingAvailable = this.#options.maxInputBytes -
      this.#inputBytes - this.#stagingInputBytes;
    let selectedCount = 1;
    let selectedStagingBytes = route.queue[0].inputBytes;
    for (let index = 1; index < maxBatch; index++) {
      const nextBytes = route.queue[index].inputBytes;
      if (nextBytes > stagingAvailable - selectedStagingBytes) break;
      selectedStagingBytes += nextBytes;
      selectedCount = index + 1;
    }
    const stagingInputBytes = selectedCount > 1 ? selectedStagingBytes : 0;
    this.#stagingInputBytes += stagingInputBytes;
    const selected = route.queue.splice(0, selectedCount);
    for (const request of selected) {
      request.submitted = true;
      request.handle._setState('submitted');
      this.#submittedRequests.add(request);
    }
    this.#dispatchSequence += 1n;
    route.lastDispatch = this.#dispatchSequence;
    // Publish route.inFlight/#dispatches before crossing into the target. The
    // target may synchronously call close/retire before returning its Promise.
    const completion = Promise.resolve().then(async () => {
      let completedAt: number | null = null;
      try {
        const results = await route.target.executeScheduledBatch(selected.map((request) =>
          Object.freeze({
            inputs: request.inputs,
            options: request.executionOptions,
            preparedPlan: request.preparedPlan,
            compatibilityToken: request.routeToken,
            resourceDomain: request.resourceDomain,
            deviceEpoch: request.deviceEpoch,
            outputBytes: request.outputBytes,
          })));
        // One physical batch has one completion instant. Per-lane cleanup must
        // not move a later lane across its freshness deadline.
        completedAt = monotonicMilliseconds();
        const distinctResults = new Set<object>();
        const validResults = Array.isArray(results) && results.length === selected.length &&
          results.every((result) => result !== null && typeof result === 'object' &&
            typeof (result as ExecutionResult).close === 'function' &&
            !distinctResults.has(result) && Boolean(distinctResults.add(result)));
        if (!validResults) {
          const closed = new Set<object>();
          for (const result of Array.isArray(results) ? results : []) {
            if (result !== null && typeof result === 'object' && 'close' in result &&
                typeof (result as ExecutionResult).close === 'function' && !closed.has(result)) {
              closed.add(result);
              await Promise.resolve().then(
                () => (result as ExecutionResult).close(),
              ).catch(() => undefined);
            }
          }
          throw new VolvoxAIError('ABI_UNSUPPORTED',
            `Backend '${route.target.backend}' returned the wrong batch result count.`, {
              phase: 'execution', backend: route.target.backend,
          });
        }
        try {
          this.#transferResultRetention(selected, results);
        } catch (error) {
          await Promise.allSettled(results.map((result) => result.close()));
          throw error;
        }
        const discardedResults: ExecutionResult[] = [];
        for (let index = 0; index < selected.length; index++) {
          const request = selected[index];
          const result = results[index];
          if (request.cancelCode !== null) {
            this.#settleRejected(request, requestError(
              request.cancelCode,
              request.cancelCode === 'SUPERSEDED'
                ? 'Runtime request was superseded after submission.'
                : 'Runtime request was cancelled after submission.',
              request.target.backend,
            ), request.cancelCode === 'SUPERSEDED' ? 'superseded' : 'cancelled', completedAt);
            discardedResults.push(result);
          } else if (request.deadline !== null && request.deadline <= completedAt &&
              request.freshness === 'drop-if-late') {
            this.#settleRejected(request, requestError(
              'DEADLINE_EXCEEDED', 'Runtime request completed after its freshness deadline.',
              request.target.backend,
            ), 'failed', completedAt);
            discardedResults.push(result);
          } else {
            this.#settleResolved(request, result, completedAt);
          }
        }
        await Promise.allSettled(discardedResults.map((result) => result.close()));
      } catch (error) {
        const failedAt = completedAt ?? monotonicMilliseconds();
        const failure = error instanceof Error
          ? error
          : new VolvoxAIError('EXECUTION_FAILED', 'Scheduled batch execution failed.', {
            phase: 'execution', backend: route.target.backend, cause: error,
          });
        for (const request of selected) {
          if (request.settled) continue;
          if (request.cancelCode !== null) {
            this.#settleRejected(request, requestError(
              request.cancelCode,
              request.cancelCode === 'SUPERSEDED'
                ? 'Runtime request was superseded after submission.'
                : 'Runtime request was cancelled after submission.',
              request.target.backend,
            ), request.cancelCode === 'SUPERSEDED' ? 'superseded' : 'cancelled', failedAt);
          } else if (request.deadline !== null && request.deadline <= failedAt &&
              request.freshness === 'drop-if-late') {
            // Match successful completion and the native v1 runtime: a hard
            // freshness deadline wins over a simultaneous provider/ABI error.
            // Retain the provider failure as the diagnostic cause.
            this.#settleRejected(request, new VolvoxAIError(
              'DEADLINE_EXCEEDED',
              'Runtime request failed after its freshness deadline.',
              {
                phase: 'execution',
                backend: request.target.backend,
                cause: failure,
              },
            ), 'failed', failedAt);
          } else {
            this.#settleRejected(request, failure, 'failed', failedAt);
          }
        }
      } finally {
        this.#stagingInputBytes -= stagingInputBytes;
        route.inFlight = null;
        domain.active = false;
        this.#deleteRouteIfIdle(route);
        this.#scheduleDrain();
      }
    });
    route.inFlight = completion;
    this.#dispatches.add(completion);
    completion.finally(() => this.#dispatches.delete(completion)).catch(() => undefined);
  }

  #removeQueued(route: RouteState, request: PendingRequest): void {
    const index = route.queue.indexOf(request);
    if (index >= 0) route.queue.splice(index, 1);
  }

  #transferResultRetention(
    requests: readonly PendingRequest[],
    results: readonly ExecutionResult[],
  ): void {
    if (this.#resultBudget === null) return;
    const leases = requests.map((request) => request.resultBudgetLease);
    if (leases.some((lease) => lease === null || !lease.hasResultSlot)) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        'Scheduled result does not carry its exact retained-output reservation.', {
          phase: 'execution', backend: requests[0].target.backend,
        });
    }
    let remaining = results.length;
    const unregister: (() => void)[] = [];
    try {
      for (let index = 0; index < results.length; index++) {
        const lease = leases[index]!;
        unregister.push(registerExecutionResultCloseFinalizer(results[index], () => {
          lease.releaseResultSlot();
          remaining--;
          if (remaining === 0) {
            for (const retained of leases) retained!.releaseOutputBytes();
          }
        }));
      }
    } catch (error) {
      for (const rollback of unregister) rollback();
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        'Scheduled result could not accept its retained-output lease.', {
          phase: 'execution', backend: requests[0].target.backend, cause: error,
        });
    }
    for (const request of requests) request.resultRetentionTransferred = true;
  }

  #deleteRouteIfIdle(route: RouteState): void {
    if (route.inFlight !== null || route.queue.length !== 0) return;
    const routes = this.#routesByTarget.get(route.target);
    if (routes?.get(route.routeToken) !== route) return;
    routes.delete(route.routeToken);
    for (const targetRoutes of this.#routesByTarget.values()) {
      for (const candidate of targetRoutes.values()) {
        if (candidate.resourceDomain === route.resourceDomain) return;
      }
    }
    const domain = this.#domains.get(route.resourceDomain);
    if (!domain?.active) this.#domains.delete(route.resourceDomain);
  }

  #settleResolved(
    request: PendingRequest,
    result: ExecutionResult,
    terminalAt: number,
  ): void {
    if (request.settled) return;
    request.settled = true;
    request.handle._setDeadlineMissed(
      request.deadline !== null && request.deadline <= terminalAt,
    );
    request.handle._setState('completed');
    this.#releaseAdmission(request);
    request.resolve(result);
  }

  #settleRejected(
    request: PendingRequest,
    error: Error,
    state: Extract<RuntimeRequestState, 'cancelled' | 'failed' | 'superseded'>,
    terminalAt: number = monotonicMilliseconds(),
  ): void {
    if (request.settled) return;
    request.settled = true;
    request.handle._setDeadlineMissed(
      request.deadline !== null && request.deadline <= terminalAt,
    );
    request.handle._setState(state);
    this.#releaseAdmission(request);
    request.reject(error);
  }

  #releaseAdmission(request: PendingRequest): void {
    this.#requestCount--;
    this.#inputBytes -= request.inputBytes;
    this.#submittedRequests.delete(request);
    request.handle._clearCancel();
    if (request.abortSignal && request.abortListener) {
      try {
        removeAbortSignalListener(request.abortSignal, request.abortListener);
      } catch {
        // Hostile signal cleanup cannot strand result/input ownership.
      }
      request.abortListener = null;
    }
    if (!request.resultRetentionTransferred) request.resultBudgetLease?.releaseAll();
    request.releaseRouteClaim?.();
    request.releaseRouteClaim = null;
  }
}

/** @internal Validate a Runtime default without allocating the scheduler. */
export function normalizeRuntimeExecutionConfiguration(
  configuration: RuntimeExecutionConfiguration | undefined,
): Readonly<Required<Pick<RuntimeExecutionConfiguration, 'mode'>> & {
  readonly scheduler: RuntimeSchedulerOptions;
  readonly results: RuntimeResultBudgetOptions;
}> {
  if (configuration !== undefined &&
      (!configuration || typeof configuration !== 'object' || Array.isArray(configuration))) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime execution configuration must be an object.', {
      phase: 'initialization',
    });
  }
  let suppliedMode: RuntimeExecutionConfiguration['mode'];
  let suppliedScheduler: RuntimeExecutionConfiguration['scheduler'];
  let suppliedResults: RuntimeExecutionConfiguration['results'];
  try {
    suppliedMode = configuration?.mode;
    suppliedScheduler = configuration?.scheduler;
    suppliedResults = configuration?.results;
  } catch (error) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Could not snapshot Runtime execution configuration.', {
        phase: 'initialization', cause: error,
      });
  }
  const mode = suppliedMode ?? SCHEDULED_MODE;
  if (!executionModes.includes(mode)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime execution mode is invalid.', {
      phase: 'initialization',
    });
  }
  // Retain detached scalar policy, but deliberately do not allocate a
  // RuntimeScheduler. DIRECT's one-shot path stays scheduler-state free.
  const scheduler = normalizeSchedulerOptions(suppliedScheduler);
  const results = normalizeResultBudgetOptions(suppliedResults);
  return Object.freeze({ mode, scheduler, results });
}
