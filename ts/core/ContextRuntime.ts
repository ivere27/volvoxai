import { Model } from './Model.js';
import {
  hasCanonicalResolvedShapePlanProvenance,
  resolveMinimumGraphShapes,
  type ResolvedShapePlan,
  type ResolvedTensorDescriptor,
} from './ResolvedShapePlan.js';
import { runtimeIdentity } from './Identity.js';
import {
  ExecutionResult,
  createExecutionResult,
  executionIdentity,
  normalizeBackendReport,
  registerExecutionResultCloseFinalizer,
  releaseBackendExecutionSnapshot,
  takeExecutionResultHostOutputs,
  type BackendExecutionSnapshot,
  type BackendHostTensorSnapshot,
  type ExecutionDecodeState,
  type ExecutionReport,
  type ExecutionRouteEvidence,
} from './ExecutionResult.js';
import { VolvoxAIError, runtimeError, type VolvoxAIErrorCode } from './RuntimeErrors.js';
import {
  BACKEND_MEMORY_SNAPSHOT_PROTOCOL,
  normalizeMemoryCaptureOptions,
  normalizeBackendMemorySnapshot,
  RuntimeMemoryCaptureSession,
  type BackendMemorySnapshot,
  type MemoryCaptureOptions,
  type RuntimeMemoryCaptureInput,
  type RuntimeMemoryEvidence,
} from './MemoryCapture.js';
import {
  assertBackendProvider,
  assertProviderCompiledModel,
  assertProviderExecutionContext,
  assertProviderInvariantResourceLease,
  assertProviderPreparedBatchRoute,
  capturedProviderInvariantResources,
  createBackendCompileInput,
  createBackendDeviceIdentity,
  validatedProviderInvariantResources,
  type BackendProvider,
  type BackendProviderCapabilities,
  type BackendProviderCompiledModel,
  type BackendProviderExecutionContext,
  type BackendProviderInvariantResourceLease,
  type ValidatedProviderInvariantResources,
  type BackendResolvedAdapterSelection,
  type BackendResolvedExecutionRequest,
} from '../backends/BackendProvider.js';
import {
  hasConservativeIndependentPublicBatchSemantics,
  type IndependentBatchSemanticsEvidence,
} from '../ops/independentBatchSemantics.js';
import type {
  AdapterSelector,
  DecodeExecutionOptions,
  ExecutionInputs,
  ExecutionOptions,
  RuntimeTypedArray,
} from '../types.js';
import type {
  BackendPolicyModeValue,
  DecodeRowModeValue,
  ExecutionModeValue,
  OperatorFallbackValue,
} from '../generated/volvoxaiEnums.js';
import {
  ExecutionMode,
  MemoryOwnerKind,
  MemorySnapshotPoint,
  OperationStage,
  decodeRowModes,
  executionModes,
} from '../generated/volvoxaiEnums.js';
import {
  acquireDeviceTensorInputLease,
  deviceTensorInputLeaseMatches,
  inspectDeviceTensorReference,
  releaseDeviceTensorInputLease,
  type DeviceTensorInputLease,
} from '../ops/deviceTensorReference.js';
import {
  RuntimeRequestHandle,
  RuntimeResultBudget,
  RuntimeScheduler,
  abortSignalIsAborted,
  addAbortSignalListener,
  captureRuntimeSubmitOptions,
  normalizeRuntimeExecutionConfiguration,
  removeAbortSignalListener,
  type PreparedScheduledExecution,
  type RuntimeExecutionConfiguration,
  type RuntimeResultBudgetLease,
  type RuntimeRunOptions,
  type RuntimeSubmitOptions,
  type ScheduledCompatibilityToken,
  type ScheduledExecutionPreflight,
  type ScheduledExecutionRequest,
  type ScheduledExecutionTarget,
} from './RuntimeScheduler.js';

export type BackendPolicy =
  | {
      readonly mode: Extract<BackendPolicyModeValue, 'prefer'>;
      readonly order: readonly string[];
      readonly operatorFallback: OperatorFallbackValue;
    }
  | {
      readonly mode: Extract<BackendPolicyModeValue, 'require'>;
      readonly backend: string;
      readonly operatorFallback: OperatorFallbackValue;
    };

export interface RuntimeOptions {
  readonly onDiagnostic?: ((event: RuntimeDiagnostic) => void) | null;
  /** Absence preserves the zero-sampling legacy report path. */
  readonly memoryCapture?: MemoryCaptureOptions;
  /** DIRECT is allocation-minimal; SCHEDULED allocates scheduling lazily. */
  readonly execution?: RuntimeExecutionConfiguration;
}

export interface ModelCompileOptions {
  readonly backend?: BackendPolicy;
}

export interface ExecutionContextOptions {
  readonly adapter?: Readonly<AdapterSelector> | null;
  /**
   * Explicit logical dimension used to validate batch-indexed adapter arrays.
   * Logical snapshots do not infer this application meaning from axis names.
   */
  readonly adapterBatchDimension?: string | null;
  readonly decode?: Readonly<{
    changedInputs?: readonly string[] | null;
    rowMode?: DecodeRowModeValue;
    requireIncremental?: boolean;
    /**
     * Dense decode slot capacity. Defaults to one.
     *
     * Declared rather than inferred: a scheduler slot exists whether or not a
     * binding happens to carry that leading extent, so a capacity read out of
     * the first sample would make a two-slot context decode one lane and report
     * success. Every step then carries one position per lane through
     * `DecodeExecutionOptions.positions`.
     */
    lanes?: number;
  }>;
  /**
   * Global slot ids to keep resident per declared weight bank, ascending. A
   * bank left out stays fully resident. Residency is context-private: sibling
   * contexts of the same CompiledModel may hold different slots, and it
   * participates in this context's plan-cache key.
   */
  readonly bankResidency?: Readonly<Record<string, readonly number[]>>;
}

export interface CompilationCandidateReport {
  readonly backend: string;
  readonly outcome: 'selected' | 'unavailable' | 'unsupported' | 'failed' | 'not-attempted';
  readonly code: VolvoxAIErrorCode | null;
  readonly message: string;
  readonly operatorFallback: BackendProviderCapabilities['operatorFallback'] | null;
  readonly device: Readonly<Record<string, string>> | null;
  readonly elapsedMs: number;
  readonly allocationBytes: number | null;
  readonly routeEvidence: Readonly<{
    tierFallback: boolean;
    operator: Readonly<{
      attestation: BackendProviderCapabilities['operatorFallback'] | null;
      used: boolean | null;
      offendingNode: string | number | null;
    }>;
  }>;
}

export interface CompilationReport {
  readonly compilationId: string;
  readonly requestedPolicy: BackendPolicy;
  readonly selectedBackend: string | null;
  readonly selectedDevice: Readonly<Record<string, string>> | null;
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly adapterRevisionId: string | null;
  readonly adapterRevisionIds: readonly string[];
  readonly compileTimeMs: number;
  readonly allocationBytes: number | null;
  readonly routeEvidence: CompilationCandidateReport['routeEvidence'] | null;
  /** Core-generated, exact-fingerprint evidence for scheduler batch safety. */
  readonly batchSemantics: Readonly<IndependentBatchSemanticsEvidence>;
  readonly candidates: readonly CompilationCandidateReport[];
  readonly memoryEvidence?: RuntimeMemoryEvidence;
}

export interface ExecutionFailureReport {
  readonly executionId: string;
  readonly contextId: string;
  readonly backend: string;
  readonly device: Readonly<Record<string, string>> | null;
  readonly outcome: 'failed';
  readonly code: VolvoxAIErrorCode;
  readonly message: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly shapeSignature: string | null;
  readonly adapterRevisionId: string | null;
  readonly adapterRevisionIds: readonly (string | null)[];
  readonly shapeBindTimeMs: number | null;
  readonly providerTimeMs: number | null;
  readonly executionTimeMs: number;
  readonly routeEvidence: ExecutionRouteEvidence;
  readonly decodeState: ExecutionDecodeState;
  readonly memoryEvidence?: RuntimeMemoryEvidence;
}

export type RuntimeDiagnostic =
  | { readonly kind: 'compilation'; readonly report: CompilationReport }
  | { readonly kind: 'execution'; readonly report: ExecutionReport }
  | { readonly kind: 'execution-error'; readonly report: ExecutionFailureReport };

interface ProviderEntry {
  readonly name: string;
  readonly provider: BackendProvider;
}

const EMPTY_EXECUTION_OPTIONS = Object.freeze({}) as Readonly<DecodeExecutionOptions>;
const EMPTY_ADAPTER_REVISION_IDS = Object.freeze([]) as readonly (string | null)[];
const EMPTY_ADAPTER_SELECTORS = Object.freeze([]) as readonly (
  Readonly<AdapterSelector> | null
)[];
const EMPTY_ADAPTER_SELECTION = Object.freeze({
  batchSize: null,
  selectors: EMPTY_ADAPTER_SELECTORS,
}) as Readonly<BackendResolvedAdapterSelection>;
const DIRECT_MODE = executionModes[ExecutionMode.Direct];
const SCHEDULED_MODE = executionModes[ExecutionMode.Scheduled];

function runtimeExecutionOptions(options: RuntimeRunOptions): Readonly<ExecutionOptions> {
  const result: {
    adapter?: ExecutionOptions['adapter'];
    adapters?: ExecutionOptions['adapters'];
  } = {};
  if (Object.prototype.hasOwnProperty.call(options, 'adapter')) result.adapter = options.adapter;
  if (Object.prototype.hasOwnProperty.call(options, 'adapters')) result.adapters = options.adapters;
  return Object.freeze(result);
}

function captureRuntimeRunOptions(value: RuntimeRunOptions): Readonly<RuntimeRunOptions> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime run options must be an object.', {
      phase: 'execution',
    });
  }
  try {
    const result: {
      mode?: RuntimeRunOptions['mode'];
      priority?: number;
      deadlineMonotonicMs?: number;
      freshness?: RuntimeRunOptions['freshness'];
      streamKey?: string | null;
      signal?: AbortSignal | null;
      adapter?: ExecutionOptions['adapter'];
      adapters?: ExecutionOptions['adapters'];
    } = {
      mode: value.mode,
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
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Could not snapshot Runtime run options.', {
      phase: 'execution', cause: error,
    });
  }
}

const RUNTIME_TYPED_ARRAY_BYTE_LENGTH = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Uint8Array.prototype) as object,
  'byteLength',
)?.get;

function runtimeTypedArrayByteLength(data: RuntimeTypedArray): number {
  const byteLength = RUNTIME_TYPED_ARRAY_BYTE_LENGTH?.call(data) as number | undefined;
  if (!Number.isSafeInteger(byteLength) || (byteLength as number) < 0) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Scheduled input storage has an invalid byte length.', { phase: 'execution' });
  }
  return byteLength as number;
}

type RuntimeHostStorageKind =
  | 'float32'
  | 'int32'
  | 'int8'
  | 'uint8'
  | 'uint8-clamped';

function runtimeHostStorageKind(data: RuntimeTypedArray): RuntimeHostStorageKind | null {
  if (data instanceof Float32Array) return 'float32';
  if (data instanceof Int32Array) return 'int32';
  if (data instanceof Int8Array) return 'int8';
  if (data instanceof Uint8ClampedArray) return 'uint8-clamped';
  if (data instanceof Uint8Array) return 'uint8';
  return null;
}

function assertRuntimeRunControlOptions(options: RuntimeRunOptions): void {
  if (options.priority !== undefined &&
      (!Number.isSafeInteger(options.priority) || options.priority < -1_000 ||
        options.priority > 1_000)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime request priority is invalid.', {
      phase: 'execution',
    });
  }
  if (options.deadlineMonotonicMs !== undefined &&
      (!Number.isFinite(options.deadlineMonotonicMs) || options.deadlineMonotonicMs < 0)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime request deadline is invalid.', {
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
        options.streamKey.length > 1024)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime request streamKey is invalid.', {
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
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime request signal is invalid.', {
      phase: 'execution',
    });
  }
}

function identity(prefix: 'compilation' | 'context'): string {
  return runtimeIdentity(prefix);
}

function backendName(value: unknown, label: string): string {
  if (typeof value !== 'string' || !/^[a-z][a-z0-9._-]*$/.test(value)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      `${label} must begin with a lowercase letter and contain only lowercase letters, digits, dot, underscore, or dash.`, {
        phase: 'selection',
      });
  }
  return value;
}

function uniqueNames(values: readonly string[], label: string): string[] {
  if (!Array.isArray(values) || values.length === 0) {
    throw new VolvoxAIError('INVALID_ARGUMENT', `${label} must contain at least one backend.`, {
      phase: 'selection',
    });
  }
  const result: string[] = [];
  for (const value of values) {
    const name = backendName(value, label);
    if (!result.includes(name)) result.push(name);
  }
  return result;
}

function freezePolicy(policy: BackendPolicy): BackendPolicy {
  return policy.mode === 'require'
    ? Object.freeze({ ...policy })
    : Object.freeze({ ...policy, order: Object.freeze([...policy.order]) });
}

function normalizePolicy(
  input: BackendPolicy | undefined,
  defaultOrder: readonly string[],
): BackendPolicy {
  if (input == null) {
    return freezePolicy({
      mode: 'prefer',
      order: uniqueNames(defaultOrder, 'Runtime backend order'),
      operatorFallback: 'allow',
    });
  }
  if (!input || typeof input !== 'object' ||
      Array.isArray(input) ||
      (input.operatorFallback !== 'allow' && input.operatorFallback !== 'forbid')) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Backend policy is invalid.', {
      phase: 'selection',
    });
  }
  if (input.mode === 'require') {
    return freezePolicy({
      mode: 'require',
      backend: backendName(input.backend, 'Required backend'),
      operatorFallback: input.operatorFallback,
    });
  }
  if (input.mode === 'prefer') {
    return freezePolicy({
      mode: 'prefer',
      order: uniqueNames(input.order, 'Preferred backend order'),
      operatorFallback: input.operatorFallback,
    });
  }
  throw new VolvoxAIError('INVALID_ARGUMENT', 'Backend policy mode must be prefer or require.', {
    phase: 'selection',
  });
}

function freezeCandidate(report: CompilationCandidateReport): CompilationCandidateReport {
  return Object.freeze({
    ...report,
    routeEvidence: Object.freeze({
      ...report.routeEvidence,
      operator: Object.freeze({ ...report.routeEvidence.operator }),
    }),
  });
}

function monotonicMilliseconds(): number {
  return typeof globalThis.performance?.now === 'function'
    ? globalThis.performance.now()
    : Date.now();
}

function elapsedMilliseconds(started: number): number {
  return Math.max(0, monotonicMilliseconds() - started);
}

const EXECUTE_PREPARED_SHAPE_PLAN = Symbol('executePreparedShapePlan');

function routeEvidence(
  tierFallback: boolean,
  attestation: BackendProviderCapabilities['operatorFallback'] | null,
  used: boolean | null,
  offendingNode: string | number | null = null,
): CompilationCandidateReport['routeEvidence'] {
  return Object.freeze({
    tierFallback,
    operator: Object.freeze({ attestation, used, offendingNode }),
  });
}

function freezeCompilationReport(
  compilationId: string,
  policy: BackendPolicy,
  snapshot: Model,
  selectedBackend: string | null,
  candidates: readonly CompilationCandidateReport[],
  compileTimeMs: number,
  batchSemantics: Readonly<IndependentBatchSemanticsEvidence>,
  memoryEvidence?: RuntimeMemoryEvidence,
): CompilationReport {
  const selected = candidates.find((candidate) => candidate.outcome === 'selected') || null;
  return Object.freeze({
    compilationId,
    requestedPolicy: policy,
    selectedBackend,
    selectedDevice: selected?.device || null,
    definitionId: snapshot.definitionId,
    topologyRevision: snapshot.topologyRevision,
    weightRevision: snapshot.weightRevision,
    weightRevisionId: snapshot.weightRevisionId,
    adapterRevisionId: null,
    adapterRevisionIds: Object.freeze([]),
    compileTimeMs,
    allocationBytes: selected?.allocationBytes ?? null,
    routeEvidence: selected?.routeEvidence || null,
    batchSemantics,
    candidates: Object.freeze(candidates.map(freezeCandidate)),
    ...(memoryEvidence === undefined ? {} : { memoryEvidence }),
  });
}

function captureMemoryEvidence(
  session: RuntimeMemoryCaptureSession | null,
  input: RuntimeMemoryCaptureInput,
): RuntimeMemoryEvidence | undefined {
  if (session === null) return undefined;
  try {
    return session.capture(input);
  } catch {
    // A best-effort collector must never replace a valid lifecycle outcome.
    return undefined;
  }
}

function captureBackendMemorySnapshot(
  session: RuntimeMemoryCaptureSession | null,
  context: BackendProviderExecutionContext,
  point: MemorySnapshotPoint,
  ownerId: string,
  resultId?: string,
): BackendMemorySnapshot | null {
  if (session === null || !session.policy.includeResourceInventory ||
      typeof context.captureMemorySnapshot !== 'function') {
    return null;
  }
  try {
    return normalizeBackendMemorySnapshot(context.captureMemorySnapshot(Object.freeze({
      protocol: BACKEND_MEMORY_SNAPSHOT_PROTOCOL,
      point,
      subject: Object.freeze({
        kind: MemoryOwnerKind.ExecutionContext,
        ownerId,
      }),
      ...(resultId === undefined ? {} : {
        result: Object.freeze({
          kind: MemoryOwnerKind.Result,
          ownerId: resultId,
        }),
      }),
    })));
  } catch {
    return null;
  }
}

function cloneSelector<T>(value: T): T {
  if (Array.isArray(value)) {
    return Object.freeze(value.map((entry) => cloneSelector(entry))) as T;
  }
  if (value && typeof value === 'object') {
    return Object.freeze(Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, entry]) => [key, cloneSelector(entry)]),
    )) as T;
  }
  return value;
}

function plainRecord(value: unknown): Readonly<Record<string, unknown>> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : null;
}

function assertExactOptionObject(
  value: unknown,
  allowed: ReadonlySet<string>,
  label: string,
): asserts value is Readonly<Record<string, unknown>> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', `${label} must be an object.`, {
      phase: 'execution',
    });
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new VolvoxAIError('INVALID_ARGUMENT', `${label} must be a plain object.`, {
      phase: 'execution',
    });
  }
  const unknown = Object.keys(value).find((name) => !allowed.has(name));
  if (unknown) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      `${label} does not accept option '${unknown}'.`, { phase: 'execution' });
  }
}

const CONTEXT_OPTION_NAMES = new Set([
  'adapter', 'adapterBatchDimension', 'decode', 'bankResidency',
]);
const DECODE_CONTEXT_OPTION_NAMES = new Set([
  'changedInputs', 'rowMode', 'requireIncremental', 'lanes',
]);
const EXECUTION_OPTION_NAMES = new Set(['adapter', 'adapters']);
const DECODE_EXECUTION_OPTION_NAMES = new Set([
  'adapter', 'adapters', 'changedInputs', 'position', 'positions',
]);

function cloneRuntimeData(data: ExecutionInputs[string]['data']): RuntimeTypedArray {
  if (data instanceof Float32Array) return new Float32Array(data);
  if (data instanceof Int32Array) return new Int32Array(data);
  if (data instanceof Int8Array) return new Int8Array(data);
  if (data instanceof Uint8ClampedArray) return new Uint8ClampedArray(data);
  if (data instanceof Uint8Array) return new Uint8Array(data);
  throw new VolvoxAIError('BACKEND_UNSUPPORTED',
    'Decode retained inputs require host typed-array storage; device inputs are ordinary-execute only.', {
      phase: 'execution',
    });
}

const EMPTY_DEVICE_INPUTS: Readonly<Record<string, DeviceTensorInputLease>> = Object.freeze({});

function assertExecutionContextOptions(options: unknown): void {
  assertExactOptionObject(options, CONTEXT_OPTION_NAMES, 'Execution context options');
  const decode = (options as ExecutionContextOptions).decode;
  if (decode !== undefined) {
    assertExactOptionObject(decode, DECODE_CONTEXT_OPTION_NAMES,
      'Execution context decode options');
    /* The context's dense slot capacity, declared here because it is a property
     * of the context and not of whichever binding happens to arrive first. The
     * decode lifecycle verifies each binding against it instead of inferring
     * one, which is what stops a two-slot context from quietly decoding one
     * lane of a two-lane batch. */
    const lanes = (decode as { lanes?: unknown }).lanes;
    if (lanes !== undefined && (!Number.isSafeInteger(lanes) || (lanes as number) < 1)) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Execution context decode lanes must be a positive integer.', {
          phase: 'compilation',
        });
    }
  }
  const residency = (options as ExecutionContextOptions).bankResidency;
  if (residency !== undefined && (residency === null || typeof residency !== 'object' ||
      Array.isArray(residency))) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Execution context bankResidency must be an object of slot-id arrays.', {
        phase: 'compilation',
      });
  }
  const batchDimension = (options as ExecutionContextOptions).adapterBatchDimension;
  if (batchDimension !== undefined && batchDimension !== null &&
      (typeof batchDimension !== 'string' || batchDimension.length === 0)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Execution context adapterBatchDimension must be a non-empty string or null.', {
        phase: 'compilation',
      });
  }
}

function assertExecutionOptions(
  options: unknown,
  operation: ExecutionDecodeState['operation'],
): void {
  assertExactOptionObject(
    options,
    operation === 'execute' ? EXECUTION_OPTION_NAMES : DECODE_EXECUTION_OPTION_NAMES,
    operation === 'execute' ? 'Execution options' : `Decode ${operation} options`,
  );
}

function executionRouteEvidence(
  capabilities: BackendProviderCapabilities,
  compilation: CompilationReport,
  backendReport: Readonly<Record<string, unknown>> | null,
  failureNode: string | number | null = null,
): ExecutionRouteEvidence {
  const backendRoute = plainRecord(backendReport?.route);
  const reportedUsed = typeof backendRoute?.operatorFallbackUsed === 'boolean'
    ? backendRoute.operatorFallbackUsed
    : null;
  if (capabilities.operatorFallback === 'none' && reportedUsed === true) {
    throw new VolvoxAIError('EXECUTION_FAILED',
      `Backend '${compilation.selectedBackend || 'unknown'}' violated its no-fallback attestation.`, {
        phase: 'execution',
        backend: compilation.selectedBackend,
        node: typeof backendRoute?.offendingNode === 'string' ||
          typeof backendRoute?.offendingNode === 'number'
          ? backendRoute.offendingNode
          : failureNode,
      });
  }
  if (capabilities.operatorFallback === 'reported' && reportedUsed == null) {
    throw new VolvoxAIError('EXECUTION_FAILED',
      `Backend '${compilation.selectedBackend || 'unknown'}' omitted required operator-route evidence.`, {
        phase: 'execution', backend: compilation.selectedBackend,
      });
  }
  const offendingNode = typeof backendRoute?.offendingNode === 'string' ||
      typeof backendRoute?.offendingNode === 'number'
    ? backendRoute.offendingNode
    : failureNode;
  return Object.freeze({
    tierFallback: compilation.routeEvidence?.tierFallback === true,
    operator: Object.freeze({
      attestation: capabilities.operatorFallback,
      used: capabilities.operatorFallback === 'none' ? false : reportedUsed,
      offendingNode,
    }),
  });
}

function executionFailureRouteEvidence(
  capabilities: BackendProviderCapabilities,
  compilation: CompilationReport,
  backendReport: Readonly<Record<string, unknown>> | null,
  failureNode: string | number | null,
): ExecutionRouteEvidence {
  const backendRoute = plainRecord(backendReport?.route);
  const reportedUsed = typeof backendRoute?.operatorFallbackUsed === 'boolean'
    ? backendRoute.operatorFallbackUsed
    : null;
  const offendingNode = typeof backendRoute?.offendingNode === 'string' ||
      typeof backendRoute?.offendingNode === 'number'
    ? backendRoute.offendingNode
    : failureNode;
  return Object.freeze({
    tierFallback: compilation.routeEvidence?.tierFallback === true,
    operator: Object.freeze({
      attestation: capabilities.operatorFallback,
      used: capabilities.operatorFallback === 'none' && reportedUsed == null
        ? false
        : reportedUsed,
      offendingNode,
    }),
  });
}

function executionDecodeState(
  operation: ExecutionDecodeState['operation'],
  options: DecodeExecutionOptions,
  backendReport: Readonly<Record<string, unknown>> | null,
): ExecutionDecodeState {
  const backendDecode = plainRecord(backendReport?.decode);
  const cacheGeneration = Number.isSafeInteger(backendDecode?.cacheGeneration) &&
      (backendDecode!.cacheGeneration as number) >= 0
    ? backendDecode!.cacheGeneration as number
    : null;
  const optionPosition = Number.isSafeInteger(options.position) ? options.position as number : null;
  const reportPosition = Number.isSafeInteger(backendDecode?.position)
    ? backendDecode!.position as number
    : null;
  const positiveDecodeInteger = (value: unknown): number | null =>
    Number.isSafeInteger(value) && (value as number) > 0 ? value as number : null;
  const activeSequenceLength = positiveDecodeInteger(backendDecode?.activeSequenceLength);
  const reportedLengths = backendDecode?.activeSequenceLengths;
  const activeSequenceLengths = Array.isArray(reportedLengths) &&
      reportedLengths.length > 0 &&
      reportedLengths.every((length) => Number.isSafeInteger(length) && length > 0)
    ? Object.freeze([...reportedLengths as readonly number[]])
    : null;
  const kvCapacity = positiveDecodeInteger(backendDecode?.kvCapacity);
  const kvCapacityClass = positiveDecodeInteger(backendDecode?.kvCapacityClass);
  const automaticResetCount = Number.isSafeInteger(backendDecode?.automaticResetCount) &&
      (backendDecode!.automaticResetCount as number) >= 0
    ? backendDecode!.automaticResetCount as number
    : 0;
  return Object.freeze({
    operation,
    mode: typeof backendDecode?.mode === 'string' && backendDecode.mode
      ? backendDecode.mode
      : null,
    cacheState: operation === 'seed'
      ? 'seeded'
      : operation === 'step' ? 'advanced' : 'not-applicable',
    cacheGeneration,
    position: operation === 'step' ? reportPosition ?? optionPosition : null,
    activeSequenceLength,
    activeSequenceLengths,
    kvCapacity,
    kvCapacityClass,
    semanticSeedSignature:
      typeof backendDecode?.semanticSeedSignature === 'string' &&
        backendDecode.semanticSeedSignature.length > 0
        ? backendDecode.semanticSeedSignature
        : null,
    automaticReset: backendDecode?.automaticReset === true,
    automaticResetReason:
      typeof backendDecode?.automaticResetReason === 'string' &&
        backendDecode.automaticResetReason.length > 0
        ? backendDecode.automaticResetReason
        : null,
    automaticResetCount,
  });
}

function explicitPublicBatchContract(
  snapshot: Model,
  compiled: BackendProviderCompiledModel,
): Readonly<{
  symbol: string;
  maximumBatchSize: number;
}> | null {
  if (compiled.batchContract.densePublicBatch !== 'single-invocation' ||
      compiled.batchContract.independentBatch !== 'compiler-proved/v1') {
    return null;
  }
  const symbol = snapshot.inputDescriptors[0]?.shape[0];
  if (typeof symbol !== 'string' ||
      snapshot.inputDescriptors.some((descriptor) => descriptor.shape[0] !== symbol) ||
      snapshot.outputDescriptors.some((descriptor) => descriptor.shape[0] !== symbol)) {
    return null;
  }
  const dimension = snapshot.graph.dimensions[symbol];
  if (!dimension || dimension.min !== 1 || dimension.multiple_of !== 1 ||
      !Number.isSafeInteger(dimension.max) || dimension.max < 1) {
    return null;
  }
  if (!hasConservativeIndependentPublicBatchSemantics(snapshot)) return null;
  return Object.freeze({ symbol, maximumBatchSize: dimension.max });
}

function exactPlanOutputBytes(plan: ResolvedShapePlan): number {
  let bytes = 0;
  for (const output of plan.outputs) {
    if (!Number.isSafeInteger(output.sizeBytes) || output.sizeBytes < 0 ||
        output.sizeBytes > Number.MAX_SAFE_INTEGER - bytes) {
      throw new VolvoxAIError('OUT_OF_MEMORY',
        'Resolved output byte count exceeds the safe integer range.', {
          phase: 'execution',
        });
    }
    bytes += output.sizeBytes;
  }
  return bytes;
}

function captureDirectExecutionInputs(
  inputs: ExecutionInputs,
  backend: string,
): ExecutionInputs {
  if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs) ||
      ArrayBuffer.isView(inputs)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'DIRECT execution inputs must be a named tensor-view record.', {
        phase: 'execution', backend,
      });
  }
  try {
    if (Object.getOwnPropertySymbols(inputs).length !== 0) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'DIRECT execution inputs cannot contain symbol keys.', {
          phase: 'execution', backend,
        });
    }
    const captured = Object.create(null) as Record<string, ExecutionInputs[string]>;
    for (const name of Object.getOwnPropertyNames(inputs)) {
      const view = inputs[name];
      if (!view || typeof view !== 'object') {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `DIRECT execution input '${name}' must contain data and an array shape.`, {
            phase: 'execution', backend,
          });
      }
      const data = view.data;
      const shape = view.shape;
      if (!Array.isArray(shape)) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `DIRECT execution input '${name}' must contain data and an array shape.`, {
            phase: 'execution', backend,
          });
      }
      // DIRECT intentionally does not copy payload bytes. Capture the data
      // identity and a detached immutable shape once so a lazy context-create
      // await cannot change the plan/provider metadata through object mutation.
      captured[name] = Object.freeze({
        data,
        shape: Object.freeze([...shape]),
      });
    }
    return Object.freeze(captured);
  } catch (error) {
    throw runtimeError(error, 'INVALID_ARGUMENT',
      'Could not capture DIRECT execution input metadata.', {
        phase: 'execution', backend,
      });
  }
}

function allocateRuntimeArrayLike(
  source: RuntimeTypedArray,
  length: number,
): RuntimeTypedArray {
  if (source instanceof Float32Array) return new Float32Array(length);
  if (source instanceof Int32Array) return new Int32Array(length);
  if (source instanceof Int8Array) return new Int8Array(length);
  if (source instanceof Uint8ClampedArray) return new Uint8ClampedArray(length);
  return new Uint8Array(length);
}

/** Context-aware runtime returned by `VolvoxAI.createRuntime()`. */
export class Runtime {
  readonly #providers = new Map<string, ProviderEntry>();
  readonly #initializationFailures = new Map<string, string>();
  readonly #order: string[] = [];
  readonly #onDiagnostic: ((event: RuntimeDiagnostic) => void) | null;
  readonly #memoryCapture: RuntimeMemoryCaptureSession | null;
  readonly #defaultExecutionMode: ExecutionModeValue;
  readonly #schedulerOptions: RuntimeExecutionConfiguration['scheduler'];
  readonly #resultBudget: RuntimeResultBudget;
  #scheduler: RuntimeScheduler | null = null;
  #state: 'open' | 'closing' | 'closed' = 'open';
  #retainedChildren = 0;
  #resolveChildDrain: (() => void) | null = null;
  #closePromise: Promise<void> | null = null;

  constructor({ onDiagnostic = null, memoryCapture, execution }: RuntimeOptions = {}) {
    if (onDiagnostic != null && typeof onDiagnostic !== 'function') {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime onDiagnostic must be a function or null.', {
        phase: 'initialization',
      });
    }
    this.#onDiagnostic = onDiagnostic;
    const policy = normalizeMemoryCaptureOptions(memoryCapture);
    this.#memoryCapture = policy === null ? null : new RuntimeMemoryCaptureSession(policy);
    const executionConfiguration = normalizeRuntimeExecutionConfiguration(execution);
    this.#defaultExecutionMode = executionConfiguration.mode;
    this.#schedulerOptions = executionConfiguration.scheduler;
    this.#resultBudget = new RuntimeResultBudget(executionConfiguration.results);
  }

  /** @internal Register one already-initialized context-aware provider. */
  _addProvider(name: string, value: unknown): void {
    this._assertAcceptingWork();
    name = backendName(name, 'Backend provider name');
    const provider = assertBackendProvider(value, `Backend provider '${name}'`);
    if (provider.backendName !== name) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `Backend provider '${name}' returned backendName '${provider.backendName}'.`, {
          phase: 'initialization', backend: name,
        });
    }
    if (this.#providers.has(name)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', `Backend provider '${name}' is already initialized.`, {
        phase: 'initialization', backend: name,
      });
    }
    this.#providers.set(name, {
      name,
      provider,
    });
    if (!this.#order.includes(name)) this.#order.push(name);
  }

  /** @internal Preserve initialization evidence for policy reports. */
  _addInitializationFailure(name: string, message: string): void {
    this._assertAcceptingWork();
    name = backendName(name, 'Backend name');
    this.#initializationFailures.set(name, message);
    if (!this.#order.includes(name)) this.#order.push(name);
  }

  listBackends(): readonly string[] {
    return Object.freeze([...this.#order]);
  }

  /** Lightweight proof that DIRECT has not allocated scheduling state. */
  inspectExecution(): Readonly<{
    defaultMode: ExecutionModeValue;
    schedulerAllocated: boolean;
    admittedRequests: number;
    admittedInputBytes: number;
    admittingInputBytes: number;
    stagingInputBytes: number;
    totalInputBytes: number;
    inputSnapshotCopies: number;
    retainedResults: number;
    retainedOutputBytes: number;
    maxRetainedResults: number;
    maxRetainedOutputBytes: number;
  }> {
    return Object.freeze({
      defaultMode: this.#defaultExecutionMode,
      schedulerAllocated: this.#scheduler !== null,
      admittedRequests: this.#scheduler?.requestCount ?? 0,
      admittedInputBytes: this.#scheduler?.inputBytes ?? 0,
      admittingInputBytes: this.#scheduler?.admittingInputBytes ?? 0,
      stagingInputBytes: this.#scheduler?.stagingInputBytes ?? 0,
      totalInputBytes: this.#scheduler?.totalInputBytes ?? 0,
      inputSnapshotCopies: this.#scheduler?.inputSnapshotCopies ?? 0,
      retainedResults: this.#resultBudget.retainedResults,
      retainedOutputBytes: this.#resultBudget.retainedOutputBytes,
      maxRetainedResults: this.#resultBudget.maxRetainedResults,
      maxRetainedOutputBytes: this.#resultBudget.maxRetainedOutputBytes,
    });
  }

  /**
   * Execute through the selected Runtime mode.
   *
   * DIRECT holds no queue-owned copy and creates no RuntimeScheduler. SCHEDULED
   * snapshots inputs at admission because execution outlives this call.
   */
  async run(
    compiled: CompiledModel,
    inputs: ExecutionInputs,
    suppliedOptions: RuntimeRunOptions = {},
  ): Promise<ExecutionResult> {
    this._assertAcceptingWork();
    this.#assertCompiledTarget(compiled);
    const options = captureRuntimeRunOptions(suppliedOptions);
    this._assertAcceptingWork();
    const mode = options.mode ?? this.#defaultExecutionMode;
    if (!executionModes.includes(mode)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime execution mode is invalid.', {
        phase: 'execution', backend: compiled.backend,
      });
    }
    if (this.#defaultExecutionMode === DIRECT_MODE && mode === SCHEDULED_MODE) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        `Runtime mode '${this.#defaultExecutionMode}' cannot be widened to '${mode}'.`, {
          phase: 'execution', backend: compiled.backend,
        });
    }
    assertRuntimeRunControlOptions(options);
    const providerOptions = runtimeExecutionOptions(options);
    const signal = options.signal ?? null;
    const deadline = options.deadlineMonotonicMs;
    const freshness = options.freshness;
    if (mode === DIRECT_MODE) {
      if (signal && abortSignalIsAborted(signal)) {
        throw new VolvoxAIError('CANCELLED', 'Runtime DIRECT execution was already aborted.', {
          phase: 'execution', backend: compiled.backend,
        });
      }
      if (deadline !== undefined && deadline <= monotonicMilliseconds()) {
        throw new VolvoxAIError('DEADLINE_EXCEEDED',
          'Runtime DIRECT execution deadline has already elapsed.', {
            phase: 'execution', backend: compiled.backend,
          });
      }
      let aborted = false;
      const onAbort = () => { aborted = true; };
      if (signal) addAbortSignalListener(signal, onAbort);
      try {
        const result = await compiled.runDirect(inputs, providerOptions);
        if (aborted || (freshness === 'drop-if-late' &&
            deadline !== undefined && deadline <= monotonicMilliseconds())) {
          await result.close().catch(() => undefined);
          throw new VolvoxAIError(
            aborted ? 'CANCELLED' : 'DEADLINE_EXCEEDED',
            aborted
              ? 'Runtime DIRECT execution was cancelled after dispatch.'
              : 'Runtime DIRECT execution completed after its freshness deadline.', {
              phase: 'execution', backend: compiled.backend,
            });
        }
        return result;
      } finally {
        if (signal) {
          try {
            removeAbortSignalListener(signal, onAbort);
          } catch {
            // Listener cleanup must not replace a successful owned result or
            // strand its Runtime retained-output reservation.
          }
        }
      }
    }
    return this.submit(compiled, inputs, {
      ...providerOptions,
      priority: options.priority,
      deadlineMonotonicMs: deadline,
      freshness,
      streamKey: options.streamKey,
      signal,
    }).result;
  }

  /** Admit stateless work into this Runtime's one lazily-created coordinator. */
  submit(
    compiled: CompiledModel,
    inputs: ExecutionInputs,
    suppliedOptions: RuntimeSubmitOptions = {},
  ): RuntimeRequestHandle {
    this._assertAcceptingWork();
    this.#assertCompiledTarget(compiled);
    const options = captureRuntimeSubmitOptions(suppliedOptions);
    this._assertAcceptingWork();
    if (this.#defaultExecutionMode !== SCHEDULED_MODE) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'DIRECT Runtime execution uses run(); submit() requires SCHEDULED mode.', {
          phase: 'execution', backend: compiled.backend,
        });
    }
    return this.#ensureScheduler().submit(compiled, inputs, options);
  }

  /** Compile one exact immutable logical model revision through the provider SPI. */
  async compile(
    snapshot: Model,
    options: ModelCompileOptions = {},
  ): Promise<CompiledModel> {
    this._assertAcceptingWork();
    if (!(snapshot instanceof Model)) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Runtime.compile requires a Model.', {
          phase: 'compilation',
        });
    }
    const policy = normalizePolicy(options.backend, this.#order);
    const compileInput = createBackendCompileInput(snapshot);
    const batchSemantics = compileInput.batchSemantics;
    const releaseRuntime = this.#retainChild();
    let transferred = false;
    try {
      const compilationId = identity('compilation');
      const compilationStarted = monotonicMilliseconds();
      const names = policy.mode === 'require' ? [policy.backend] : [...policy.order];
      const candidates: CompilationCandidateReport[] = [];

      for (const name of names) {
        const candidateStarted = monotonicMilliseconds();
        const reachedByTierFallback = candidates.length > 0;
        const entry = this.#providers.get(name);
        if (!entry) {
          const message = this.#initializationFailures.get(name) ||
            `Backend provider '${name}' is unavailable in this runtime.`;
          candidates.push({
            backend: name,
            outcome: 'unavailable',
            code: policy.mode === 'require' ? 'BACKEND_REQUIRED' : 'BACKEND_UNAVAILABLE',
            message,
            operatorFallback: null,
            device: null,
            elapsedMs: elapsedMilliseconds(candidateStarted),
            allocationBytes: null,
            routeEvidence: routeEvidence(reachedByTierFallback, null, null),
          });
          if (policy.mode === 'require') break;
          continue;
        }

        if (policy.operatorFallback === 'forbid' &&
            entry.provider.capabilities.operatorFallback !== 'none') {
          const device = createBackendDeviceIdentity(entry.provider.deviceIdentity);
          candidates.push({
            backend: name,
            outcome: 'unsupported',
            code: 'OPERATOR_FALLBACK_FORBIDDEN',
            message: `Backend '${name}' cannot attest that every operator remains on the selected tier.`,
            operatorFallback: entry.provider.capabilities.operatorFallback,
            device,
            elapsedMs: elapsedMilliseconds(candidateStarted),
            allocationBytes: null,
            routeEvidence: routeEvidence(
              reachedByTierFallback,
              entry.provider.capabilities.operatorFallback,
              null,
            ),
          });
          if (policy.mode === 'require') break;
          continue;
        }

        if (entry.provider.capabilities.dynamicShapeDomain.support !== 'full') {
          candidates.push({
            backend: name,
            outcome: 'unsupported',
            code: 'BACKEND_UNSUPPORTED',
            message: `Backend '${name}' does not attest the complete bounded shape domain.`,
            operatorFallback: entry.provider.capabilities.operatorFallback,
            device: createBackendDeviceIdentity(entry.provider.deviceIdentity),
            elapsedMs: elapsedMilliseconds(candidateStarted),
            allocationBytes: null,
            routeEvidence: routeEvidence(
              reachedByTierFallback,
              entry.provider.capabilities.operatorFallback,
              null,
            ),
          });
          if (policy.mode === 'require') break;
          continue;
        }

        try {
          let candidate: unknown = null;
          let compiled: BackendProviderCompiledModel | null = null;
          let invariantResources: Readonly<ValidatedProviderInvariantResources> | null = null;
          let candidateTransferred = false;
          let selectedCandidateRecorded = false;
          try {
            candidate = await entry.provider.compile(compileInput, Object.freeze({
              operatorFallback: policy.operatorFallback,
            }));
            compiled = assertProviderCompiledModel(candidate, name, compileInput);
            invariantResources = validatedProviderInvariantResources(compiled, name);
            if (entry.provider.capabilities.operatorFallback === 'none' &&
                compiled.compilationEvidence?.operatorFallbackUsed === true) {
              throw new VolvoxAIError(
                policy.operatorFallback === 'forbid'
                  ? 'OPERATOR_FALLBACK_FORBIDDEN'
                  : 'ABI_UNSUPPORTED',
                `Backend '${name}' contradicted its no-fallback attestation.`, {
                  phase: 'compilation',
                  backend: name,
                  node: compiled.compilationEvidence.offendingNode,
                },
              );
            }
            const device = createBackendDeviceIdentity(
              compiled.compilationEvidence?.device ?? entry.provider.deviceIdentity,
            );
            const allocationBytes = compiled.compilationEvidence?.allocationBytes ?? null;
            candidates.push({
              backend: name,
              outcome: 'selected',
              code: null,
              message: `Backend '${name}' compiled the model.`,
              operatorFallback: entry.provider.capabilities.operatorFallback,
              device,
              elapsedMs: elapsedMilliseconds(candidateStarted),
              allocationBytes,
              routeEvidence: routeEvidence(
                reachedByTierFallback,
                entry.provider.capabilities.operatorFallback,
                entry.provider.capabilities.operatorFallback === 'none'
                  ? false
                  : compiled.compilationEvidence?.operatorFallbackUsed ?? null,
                compiled.compilationEvidence?.offendingNode ?? null,
              ),
            });
            selectedCandidateRecorded = true;
            const memoryEvidence = captureMemoryEvidence(this.#memoryCapture, {
              stage: OperationStage.Compile,
              point: MemorySnapshotPoint.After,
              subject: Object.freeze({
                kind: MemoryOwnerKind.CompiledModel,
                ownerId: compilationId,
              }),
              backend: name,
              device,
              domainAttestation: compiled.compilationEvidence.shapeDomain,
            });
            const report = freezeCompilationReport(
              compilationId,
              policy,
              snapshot,
              name,
              candidates,
              elapsedMilliseconds(compilationStarted),
              batchSemantics,
              memoryEvidence,
            );
            const result = new CompiledModel(
              this,
              snapshot,
              compiled,
              entry.provider.capabilities,
              report,
              this.#onDiagnostic,
              this.#memoryCapture,
              this.#resultBudget,
              releaseRuntime,
            );
            candidateTransferred = true;
            transferred = true;
            this.#emit({ kind: 'compilation', report });
            return result;
          } catch (error) {
            if (!candidateTransferred) {
              if (selectedCandidateRecorded) candidates.pop();
              invariantResources ??= capturedProviderInvariantResources(candidate);
              try {
                const close = (candidate as Partial<BackendProviderCompiledModel> | null)?.close;
                if (typeof close === 'function') await close.call(candidate);
              } catch {
                // Preserve the construction failure, including when a hostile
                // compiled close accessor throws on this cleanup read.
              } finally {
                if (invariantResources !== null) {
                  try {
                    await invariantResources.close.call(invariantResources.owner);
                  } catch {
                    // Preserve the failure that prevented ownership transfer.
                  }
                }
              }
            }
            throw error;
          }
        } catch (error) {
          const failure = runtimeError(error, 'BACKEND_UNSUPPORTED',
            `Backend '${name}' compilation failed.`, {
              phase: 'compilation', backend: name,
            });
          candidates.push({
            backend: name,
            outcome: failure.code === 'BACKEND_UNAVAILABLE'
              ? 'unavailable'
              : failure.code === 'BACKEND_UNSUPPORTED'
                ? 'unsupported'
                : 'failed',
            code: failure.code,
            message: failure.message,
            operatorFallback: entry.provider.capabilities.operatorFallback,
            device: createBackendDeviceIdentity(entry.provider.deviceIdentity),
            elapsedMs: elapsedMilliseconds(candidateStarted),
            allocationBytes: null,
            routeEvidence: routeEvidence(
              reachedByTierFallback,
              entry.provider.capabilities.operatorFallback,
              null,
              failure.node,
            ),
          });
          if (policy.mode === 'require') break;
        }
      }

      const report = freezeCompilationReport(
        compilationId,
        policy,
        snapshot,
        null,
        candidates,
        elapsedMilliseconds(compilationStarted),
        batchSemantics,
        captureMemoryEvidence(this.#memoryCapture, {
          stage: OperationStage.Compile,
          point: MemorySnapshotPoint.Failure,
          subject: Object.freeze({
            kind: MemoryOwnerKind.Model,
            ownerId: snapshot.definitionId,
          }),
          backend: candidates.at(-1)?.backend ?? '',
          device: candidates.at(-1)?.device ?? null,
        }),
      );
      this.#emit({ kind: 'compilation', report });
      const last = candidates.at(-1);
      const code: VolvoxAIErrorCode = policy.mode === 'require'
        ? last?.code === 'OPERATOR_FALLBACK_FORBIDDEN'
          ? 'OPERATOR_FALLBACK_FORBIDDEN'
          : 'BACKEND_REQUIRED'
        : 'BACKEND_UNSUPPORTED';
      throw new VolvoxAIError(code,
        policy.mode === 'require'
          ? `Required backend '${policy.backend}' could not compile the model.`
          : 'No preferred backend provider could compile the model.', {
          phase: 'selection',
          backend: policy.mode === 'require' ? policy.backend : null,
          report,
        });
    } finally {
      if (!transferred) releaseRuntime();
    }
  }

  close(): Promise<void> {
    if (this.#closePromise) return this.#closePromise;
    this.#state = 'closing';
    this.#closePromise = (async () => {
      let firstError: unknown = null;
      // The coordinator owns route contexts. It must stop admission, settle
      // queued work and await submitted batches before child/provider teardown.
      try {
        await this.#scheduler?.close();
      } catch (error) {
        firstError = error;
      }
      if (this.#retainedChildren > 0) {
        await new Promise<void>((resolve) => {
          this.#resolveChildDrain = resolve;
        });
      }
      const closures = await Promise.allSettled(
        Array.from(this.#providers.values(), ({ provider }) =>
          Promise.resolve().then(() => provider.close())),
      );
      this.#state = 'closed';
      const failed = closures.find(
        (closure): closure is PromiseRejectedResult => closure.status === 'rejected',
      );
      if (firstError === null && failed) firstError = failed.reason;
      if (firstError !== null) {
        throw runtimeError(firstError, 'EXECUTION_FAILED', 'Runtime close failed.', {
          phase: 'lifecycle',
        });
      }
    })();
    return this.#closePromise;
  }

  dispose(): Promise<void> {
    return this.close();
  }

  /** @internal Reject new root/model work as soon as close starts. */
  _assertAcceptingWork(): void {
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Runtime is closing or closed.', {
        phase: 'lifecycle',
      });
    }
  }

  /** @internal A CompiledModel closes its persistent route before provider state. */
  _retireScheduledTarget(target: CompiledModel): Promise<void> {
    return this.#scheduler?.retireTarget(target) ?? target.closeScheduledExecutionRoute();
  }

  #ensureScheduler(): RuntimeScheduler {
    if (!this.#scheduler) {
      this.#scheduler = new RuntimeScheduler(this.#schedulerOptions, this.#resultBudget);
    }
    return this.#scheduler;
  }

  #assertCompiledTarget(compiled: CompiledModel): void {
    if (!(compiled instanceof CompiledModel) || !compiled._belongsToRuntime(this)) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Runtime execution requires a CompiledModel owned by this Runtime.', {
          phase: 'execution',
        });
    }
  }

  #retainChild(): () => void {
    this.#retainedChildren++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.#retainedChildren--;
      if (this.#retainedChildren === 0) {
        const resolve = this.#resolveChildDrain;
        this.#resolveChildDrain = null;
        resolve?.();
      }
    };
  }

  #emit(event: RuntimeDiagnostic): void {
    try {
      this.#onDiagnostic?.(event);
    } catch {
      // Application diagnostics must never change runtime control flow.
    }
  }
}

export class CompiledModel implements ScheduledExecutionTarget {
  readonly backend: string;
  readonly report: CompilationReport;
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly adapterRevisionId: null = null;
  readonly adapterRevisionIds: readonly string[] = Object.freeze([]);
  readonly #runtime: Runtime;
  readonly #snapshot: Model;
  readonly #compiled: BackendProviderCompiledModel;
  readonly #invariantResources: Readonly<ValidatedProviderInvariantResources>;
  readonly #capabilities: BackendProviderCapabilities;
  readonly #onDiagnostic: ((event: RuntimeDiagnostic) => void) | null;
  readonly #memoryCapture: RuntimeMemoryCaptureSession | null;
  readonly #resultBudget: RuntimeResultBudget;
  readonly #releaseRuntime: () => void;
  readonly #batchAxisSymbol: string | null;
  readonly #maximumBatchSize: number;
  readonly #scheduledPreflightPlans = new WeakMap<object, Readonly<{
    options: Readonly<ExecutionOptions>;
    plan: ResolvedShapePlan;
    inputs: readonly Readonly<{
      name: string;
      shape: readonly number[];
      storageKind: RuntimeHostStorageKind;
      byteLength: number;
    }>[];
    inputBytes: number;
    outputBytes: number;
    compatibilityToken: ScheduledCompatibilityToken;
    resourceDomain: object;
    deviceEpoch: object;
  }>>();
  readonly #scheduledPreparedPlans = new WeakMap<object, Readonly<{
    inputs: ExecutionInputs;
    options: Readonly<ExecutionOptions>;
    plan: ResolvedShapePlan;
    compatibilityToken: ScheduledCompatibilityToken;
    resourceDomain: object;
    deviceEpoch: object;
    outputBytes: number;
  }>>();
  /*
   * Intentional minimal context pool, default size one. The route lock leases
   * this single mutable context across shapes; measured domain parallelism may
   * justify a separately bounded pool later, but implicit per-shape contexts do not.
   */
  #scheduledContext: ExecutionContext | null = null;
  #scheduledContextPromise: Promise<ExecutionContext> | null = null;
  #scheduledRouteClosePromise: Promise<void> | null = null;
  #routeBusy = false;
  #routeWaiters: (() => void)[] | null = null;
  #scheduledRouteClaims = 0;
  #state: 'open' | 'closing' | 'closed' = 'open';
  #retainedContexts = 0;
  #resolveContextDrain: (() => void) | null = null;
  #closePromise: Promise<void> | null = null;

  constructor(
    runtime: Runtime,
    snapshot: Model,
    compiled: BackendProviderCompiledModel,
    capabilities: BackendProviderCapabilities,
    report: CompilationReport,
    onDiagnostic: ((event: RuntimeDiagnostic) => void) | null,
    memoryCapture: RuntimeMemoryCaptureSession | null,
    resultBudget: RuntimeResultBudget,
    releaseRuntime: () => void,
  ) {
    this.backend = compiled.backendName;
    this.report = report;
    this.definitionId = snapshot.definitionId;
    this.topologyRevision = snapshot.topologyRevision;
    this.weightRevision = snapshot.weightRevision;
    this.weightRevisionId = snapshot.weightRevisionId;
    this.#runtime = runtime;
    this.#snapshot = snapshot;
    this.#compiled = compiled;
    this.#invariantResources = validatedProviderInvariantResources(
      compiled,
      compiled.backendName,
    );
    this.#capabilities = capabilities;
    this.#onDiagnostic = onDiagnostic;
    this.#memoryCapture = memoryCapture;
    this.#resultBudget = resultBudget;
    const batchContract = explicitPublicBatchContract(snapshot, compiled);
    this.#batchAxisSymbol = batchContract?.symbol ?? null;
    this.#maximumBatchSize = batchContract?.maximumBatchSize ?? 1;
    this.#releaseRuntime = releaseRuntime;
  }

  /** Execute through this model's owning Runtime without exposing a route context. */
  run(
    inputs: ExecutionInputs,
    options: RuntimeRunOptions = {},
  ): Promise<ExecutionResult> {
    return this.#runtime.run(this, inputs, options);
  }

  /** Admit work to this model's owning Runtime coordinator. */
  submit(
    inputs: ExecutionInputs,
    options: RuntimeSubmitOptions = {},
  ): RuntimeRequestHandle {
    return this.#runtime.submit(this, inputs, options);
  }

  /** @internal Exact Runtime identity check; display/model strings are irrelevant. */
  _belongsToRuntime(runtime: Runtime): boolean { return this.#runtime === runtime; }

  preflightScheduledExecution(
    inputs: ExecutionInputs,
    options: Readonly<ExecutionOptions>,
  ): ScheduledExecutionPreflight {
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'CompiledModel is closing or closed.', {
        phase: 'lifecycle', backend: this.backend,
      });
    }
    if (Object.prototype.hasOwnProperty.call(options, 'adapters') || options.adapter != null) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        'Scheduled adapter arrays require an exact compiled adapter route contract.', {
          phase: 'execution', backend: this.backend,
        });
    }
    let plan: ResolvedShapePlan;
    try {
      plan = this.#snapshot.bindShapes(inputs);
    } catch (error) {
      throw runtimeError(error, 'INVALID_ARGUMENT',
        'Scheduled inputs do not satisfy the logical shape contract.', {
          phase: 'execution', backend: this.backend,
        });
    }
    let inputBytes = 0;
    const outputBytes = exactPlanOutputBytes(plan);
    const inputMetadata: {
      name: string;
      shape: readonly number[];
      storageKind: RuntimeHostStorageKind;
      byteLength: number;
    }[] = [];
    for (const name of this.#snapshot.inputNames) {
      const view = inputs[name];
      if (!view || !ArrayBuffer.isView(view.data) || view.data instanceof DataView) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          'Scheduled execution currently requires host inputs that can be snapshotted.', {
            phase: 'execution', backend: this.backend,
          });
      }
      const byteLength = runtimeTypedArrayByteLength(view.data as RuntimeTypedArray);
      const storageKind = runtimeHostStorageKind(view.data as RuntimeTypedArray);
      if (storageKind === null) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `Scheduled input '${name}' uses unsupported host storage.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      if (byteLength > Number.MAX_SAFE_INTEGER - inputBytes) {
        throw new VolvoxAIError('OUT_OF_MEMORY', 'Scheduled input byte count overflowed.', {
          phase: 'execution', backend: this.backend,
        });
      }
      inputBytes += byteLength;
      inputMetadata.push(Object.freeze({
        name,
        shape: view.shape,
        storageKind,
        byteLength,
      }));
    }
    let providerRoute;
    try {
      providerRoute = assertProviderPreparedBatchRoute(
        this.#compiled.prepareBatchRoute(plan),
        this.backend,
      );
    } catch (error) {
      throw runtimeError(error, 'ABI_UNSUPPORTED',
        `Backend '${this.backend}' could not prepare a scheduled batch route.`, {
          phase: 'execution', backend: this.backend,
        });
    }
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'CompiledModel is closing or closed.', {
        phase: 'lifecycle', backend: this.backend,
      });
    }
    if (providerRoute.deviceEpoch !== this.#invariantResources.deviceEpoch ||
        this.#invariantResources.owner.deviceEpoch !== this.#invariantResources.deviceEpoch) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `Backend '${this.backend}' prepared a route outside its invariant resource epoch.`, {
          phase: 'execution', backend: this.backend,
        });
    }
    const preparedPlan = Object.freeze({});
    this.#scheduledPreflightPlans.set(preparedPlan, Object.freeze({
      options,
      plan,
      inputs: Object.freeze(inputMetadata),
      inputBytes,
      outputBytes,
      compatibilityToken: providerRoute.compatibilityToken,
      resourceDomain: providerRoute.resourceDomain,
      deviceEpoch: providerRoute.deviceEpoch,
    }));
    return Object.freeze({
      preparedPlan,
      inputBytes,
      outputBytes,
    });
  }

  prepareScheduledExecution(
    inputs: ExecutionInputs,
    options: Readonly<ExecutionOptions>,
    preflightPlan?: object,
  ): PreparedScheduledExecution {
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'CompiledModel is closing or closed.', {
        phase: 'lifecycle', backend: this.backend,
      });
    }
    const token = preflightPlan ??
      this.preflightScheduledExecution(inputs, options).preparedPlan;
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'CompiledModel is closing or closed.', {
        phase: 'lifecycle', backend: this.backend,
      });
    }
    const prepared = token && typeof token === 'object'
      ? this.#scheduledPreflightPlans.get(token)
      : undefined;
    if (prepared) this.#scheduledPreflightPlans.delete(token);
    const exactInputSet = prepared !== undefined && prepared.options === options &&
      Object.keys(inputs).length === prepared.inputs.length &&
      prepared.inputs.every((expected) => {
        const view = inputs[expected.name];
        return view !== undefined && view.shape === expected.shape &&
          runtimeHostStorageKind(view.data as RuntimeTypedArray) === expected.storageKind &&
          runtimeTypedArrayByteLength(view.data as RuntimeTypedArray) === expected.byteLength;
      });
    if (!prepared || !exactInputSet) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        'Scheduled execution did not carry its exact metadata preflight capability.', {
          phase: 'execution', backend: this.backend,
        });
    }
    const requestBatch = this.#batchAxisSymbol === null
      ? null
      : prepared.plan.symbols[this.#batchAxisSymbol];
    const maximumBatchSize = requestBatch === 1 ? this.#maximumBatchSize : 1;
    // Move the same one-shot token from metadata preflight to the exact owned
    // snapshot; no second shape resolution or provider route preparation.
    this.#scheduledPreparedPlans.set(token, Object.freeze({
      inputs,
      options,
      plan: prepared.plan,
      compatibilityToken: prepared.compatibilityToken,
      resourceDomain: prepared.resourceDomain,
      deviceEpoch: prepared.deviceEpoch,
      outputBytes: prepared.outputBytes,
    }));
    return Object.freeze({
      compatibilityToken: prepared.compatibilityToken,
      resourceDomain: prepared.resourceDomain,
      deviceEpoch: prepared.deviceEpoch,
      preparedPlan: token,
      maxBatchSize: maximumBatchSize,
      inputBytes: prepared.inputBytes,
      outputBytes: prepared.outputBytes,
    });
  }

  /** @internal Synchronously fence accepted scheduled work against DIRECT barging. */
  reserveScheduledExecutionRoute(): () => void {
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'CompiledModel is closing or closed.', {
        phase: 'lifecycle', backend: this.backend,
      });
    }
    this.#scheduledRouteClaims++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.#scheduledRouteClaims--;
    };
  }

  async runDirect(
    inputs: ExecutionInputs,
    options: Readonly<ExecutionOptions>,
  ): Promise<ExecutionResult> {
    const release = await this.#acquireRoute(false, false);
    let outputBudgetLease: RuntimeResultBudgetLease | null = null;
    let outputBudgetTransferred = false;
    try {
      assertExecutionOptions(options, 'execute');
      const capturedInputs = captureDirectExecutionInputs(inputs, this.backend);
      let plan: ResolvedShapePlan;
      try {
        plan = this.#snapshot.bindShapes(capturedInputs);
      } catch (error) {
        throw runtimeError(error, 'INVALID_ARGUMENT',
          'DIRECT inputs do not satisfy the complete logical shape contract.', {
            phase: 'execution', backend: this.backend,
          });
      }
      if (this.#state !== 'open') {
        throw new VolvoxAIError('HANDLE_DISPOSED', 'CompiledModel is closing or closed.', {
          phase: 'lifecycle', backend: this.backend,
        });
      }
      outputBudgetLease = this.#resultBudget.reserve(
        exactPlanOutputBytes(plan), this.backend,
      );
      const context = await this.#ensureScheduledContext();
      const result = await context[EXECUTE_PREPARED_SHAPE_PLAN](
        capturedInputs,
        options,
        plan,
        outputBudgetLease,
      );
      outputBudgetTransferred = true;
      return result;
    } finally {
      if (!outputBudgetTransferred) outputBudgetLease?.releaseAll();
      release();
    }
  }

  async executeScheduledBatch(
    requests: readonly Readonly<ScheduledExecutionRequest>[],
  ): Promise<readonly ExecutionResult[]> {
    if (!Array.isArray(requests) || requests.length === 0 ||
        requests.length > this.#maximumBatchSize) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Scheduled batch size is outside the compiled route contract.', {
          phase: 'execution', backend: this.backend,
        });
    }
    const plans = this.#takeScheduledPlans(requests);
    const release = await this.#acquireRoute(true, true);
    try {
      for (let index = 0; index < requests.length; index++) {
        let currentRoute;
        try {
          currentRoute = assertProviderPreparedBatchRoute(
            this.#compiled.prepareBatchRoute(plans[index]),
            this.backend,
          );
        } catch (error) {
          throw runtimeError(error, 'ABI_UNSUPPORTED',
            `Backend '${this.backend}' could not re-attest the scheduled route.`, {
              phase: 'execution', backend: this.backend,
            });
        }
        if (currentRoute.deviceEpoch !== this.#invariantResources.deviceEpoch ||
            this.#invariantResources.owner.deviceEpoch !== this.#invariantResources.deviceEpoch) {
          throw new VolvoxAIError('ABI_UNSUPPORTED',
            `Backend '${this.backend}' invalidated its invariant resource epoch before dispatch.`, {
              phase: 'execution', backend: this.backend,
            });
        }
        if (currentRoute.compatibilityToken !== requests[index].compatibilityToken ||
            currentRoute.resourceDomain !== requests[index].resourceDomain ||
            currentRoute.deviceEpoch !== requests[index].deviceEpoch) {
          throw new VolvoxAIError('ABI_UNSUPPORTED',
            `Backend '${this.backend}' changed a prepared route identity before dispatch.`, {
              phase: 'execution', backend: this.backend,
            });
        }
      }
      const context = await this.#ensureScheduledContext();
      if (requests.length === 1) {
        return Object.freeze([
          await context[EXECUTE_PREPARED_SHAPE_PLAN](
            requests[0].inputs,
            requests[0].options,
            plans[0],
          ),
        ]);
      }
      if (this.#batchAxisSymbol === null) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          'Compiled model does not declare a provable public batch axis.', {
            phase: 'execution', backend: this.backend,
          });
      }
      const batchedInputs = this.#stackScheduledInputs(requests);
      let batchPlan: ResolvedShapePlan;
      try {
        batchPlan = this.#snapshot.bindShapes(batchedInputs);
      } catch (error) {
        throw runtimeError(error, 'INVALID_ARGUMENT',
          'Stacked scheduled inputs do not satisfy the logical batch shape contract.', {
            phase: 'execution', backend: this.backend,
          });
      }
      let batchRoute;
      try {
        batchRoute = assertProviderPreparedBatchRoute(
          this.#compiled.prepareBatchRoute(batchPlan),
          this.backend,
        );
      } catch (error) {
        throw runtimeError(error, 'ABI_UNSUPPORTED',
          `Backend '${this.backend}' could not attest the stacked batch route.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      if (batchRoute.deviceEpoch !== this.#invariantResources.deviceEpoch ||
          this.#invariantResources.owner.deviceEpoch !== this.#invariantResources.deviceEpoch) {
        throw new VolvoxAIError('ABI_UNSUPPORTED',
          `Backend '${this.backend}' invalidated its invariant resource epoch before dispatch.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      if (batchRoute.resourceDomain !== requests[0].resourceDomain ||
          batchRoute.deviceEpoch !== requests[0].deviceEpoch) {
        throw new VolvoxAIError('ABI_UNSUPPORTED',
          `Backend '${this.backend}' changed resource domain or device epoch before dispatch.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      const batchResult = await context[EXECUTE_PREPARED_SHAPE_PLAN](
        batchedInputs,
        requests[0].options,
        batchPlan,
      );
      let laneResults: readonly ExecutionResult[];
      try {
        laneResults = await this.#splitScheduledResult(batchResult, requests, plans);
      } catch (error) {
        await batchResult.close().catch(() => undefined);
        throw error;
      }
      try {
        await batchResult.close();
      } catch (error) {
        // The lanes have not crossed the scheduler publication boundary yet.
        // If retiring their shared batch owner fails, close every lane so no
        // detached host backing survives an execution that is reported failed.
        await Promise.allSettled(laneResults.map((result) => result.close()));
        throw runtimeError(error, 'EXECUTION_FAILED',
          `Backend '${this.backend}' batch-result retirement failed.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      return laneResults;
    } finally {
      release();
    }
  }

  closeScheduledExecutionRoute(): Promise<void> {
    if (this.#scheduledRouteClosePromise) return this.#scheduledRouteClosePromise;
    const pendingContext = this.#scheduledContextPromise;
    if (!this.#scheduledContext && !pendingContext) return Promise.resolve();
    this.#scheduledRouteClosePromise = (async () => {
      const release = await this.#acquireRoute(true, true);
      try {
        try {
          const context = this.#scheduledContext ?? await pendingContext!;
          await context.close();
        } finally {
          this.#scheduledContext = null;
          this.#scheduledContextPromise = null;
        }
      } finally {
        release();
      }
    })();
    return this.#scheduledRouteClosePromise;
  }

  #ensureScheduledContext(): Promise<ExecutionContext> {
    if (this.#scheduledContext) return Promise.resolve(this.#scheduledContext);
    if (!this.#scheduledContextPromise) {
      this.#scheduledContextPromise = this.createContext(
        this.#batchAxisSymbol === null
          ? {}
          : { adapterBatchDimension: this.#batchAxisSymbol },
      ).then((context) => {
        this.#scheduledContext = context;
        return context;
      }, (error) => {
        this.#scheduledContextPromise = null;
        throw error;
      });
    }
    return this.#scheduledContextPromise;
  }

  #takeScheduledPlans(
    requests: readonly Readonly<ScheduledExecutionRequest>[],
  ): readonly ResolvedShapePlan[] {
    const tokens = new Set<object>();
    const plans: ResolvedShapePlan[] = [];
    for (const request of requests) {
      const token = request.preparedPlan;
      const prepared = token && typeof token === 'object'
        ? this.#scheduledPreparedPlans.get(token)
        : undefined;
      if (!prepared || tokens.has(token) ||
          prepared.inputs !== request.inputs || prepared.options !== request.options ||
          prepared.compatibilityToken !== request.compatibilityToken ||
          prepared.resourceDomain !== request.resourceDomain ||
          prepared.deviceEpoch !== request.deviceEpoch ||
          prepared.outputBytes !== request.outputBytes) {
        throw new VolvoxAIError('ABI_UNSUPPORTED',
          'Scheduled request does not carry its exact prepared shape-plan capability.', {
            phase: 'execution', backend: this.backend,
          });
      }
      tokens.add(token);
      plans.push(prepared.plan);
    }
    for (const token of tokens) this.#scheduledPreparedPlans.delete(token);
    return Object.freeze(plans);
  }

  async #acquireRoute(wait: boolean, scheduled: boolean): Promise<() => void> {
    if (!scheduled && this.#scheduledRouteClaims > 0) {
      throw new VolvoxAIError('BUSY',
        'DIRECT execution cannot barge ahead of an accepted scheduled request.', {
          phase: 'execution', backend: this.backend,
        });
    }
    if (this.#routeBusy) {
      if (!wait) {
        throw new VolvoxAIError('BUSY',
          'DIRECT execution route is busy.', {
            phase: 'execution', backend: this.backend,
          });
      }
      await new Promise<void>((resolve) => (this.#routeWaiters ??= []).push(resolve));
      // Ownership is handed directly from the releasing holder. Keeping the
      // busy bit set prevents a DIRECT caller from barging between wake-up and
      // this continuation.
    } else {
      this.#routeBusy = true;
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const next = this.#routeWaiters?.shift();
      if (this.#routeWaiters?.length === 0) this.#routeWaiters = null;
      if (next) next();
      else this.#routeBusy = false;
    };
  }

  #stackScheduledInputs(
    requests: readonly Readonly<ScheduledExecutionRequest>[],
  ): ExecutionInputs {
    const stacked = Object.create(null) as Record<string, ExecutionInputs[string]>;
    for (const name of this.#snapshot.inputNames) {
      const first = requests[0].inputs[name];
      if (!first || !ArrayBuffer.isView(first.data) || first.data instanceof DataView ||
          first.shape[0] !== 1) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Scheduled input '${name}' is not a host B=1 tensor.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      const source = first.data as RuntimeTypedArray;
      const stackedLength = source.length * requests.length;
      if (!Number.isSafeInteger(stackedLength)) {
        throw new VolvoxAIError('OUT_OF_MEMORY',
          `Scheduled input '${name}' exceeds the host array index range.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      let data: RuntimeTypedArray;
      try {
        data = allocateRuntimeArrayLike(source, stackedLength);
      } catch (error) {
        throw new VolvoxAIError('OUT_OF_MEMORY',
          `Scheduled input '${name}' batch storage could not be allocated.`, {
            phase: 'execution', backend: this.backend, cause: error,
          });
      }
      for (let index = 0; index < requests.length; index++) {
        const view = requests[index].inputs[name];
        const sameShape = view?.shape.length === first.shape.length &&
          first.shape.every((dimension, axis) => view.shape[axis] === dimension);
        if (!view || !ArrayBuffer.isView(view.data) || view.data instanceof DataView ||
            view.data.constructor !== source.constructor || !sameShape ||
            view.data.length !== source.length) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            `Scheduled input '${name}' is incompatible with its batch route.`, {
              phase: 'execution', backend: this.backend,
            });
        }
        data.set(view.data as RuntimeTypedArray, index * source.length);
      }
      stacked[name] = Object.freeze({
        data,
        shape: Object.freeze([requests.length, ...first.shape.slice(1)]),
      });
    }
    return Object.freeze(stacked);
  }

  async #splitScheduledResult(
    batchResult: ExecutionResult,
    requests: readonly Readonly<ScheduledExecutionRequest>[],
    plans: readonly ResolvedShapePlan[],
  ): Promise<readonly ExecutionResult[]> {
    const batchSize = requests.length;
    // The B=N result is unpublished. Transfer its host storage directly, or
    // perform one device readback, then give lanes views over that one backing.
    const batchOutputs = await takeExecutionResultHostOutputs(
      batchResult,
      this.#snapshot.outputNames,
    );
    const created: ExecutionResult[] = [];
    try {
      for (let lane = 0; lane < batchSize; lane++) {
        const plan = plans[lane];
        const outputs: BackendHostTensorSnapshot[] = [];
        for (const expected of plan.outputs) {
          const batchTensor = batchResult.output(expected.name);
          const data = batchOutputs.get(expected.name)!;
          if (batchTensor.shape[0] !== batchSize || data.length % batchSize !== 0) {
            throw new VolvoxAIError('EXECUTION_FAILED',
              `Batched output '${expected.name}' does not expose the declared batch axis.`, {
                phase: 'execution', backend: this.backend,
              });
          }
          const laneElements = data.length / batchSize;
          const begin = lane * laneElements;
          const laneData = data.subarray(begin, begin + laneElements) as RuntimeTypedArray;
          outputs.push(Object.freeze({
            name: expected.name,
            shape: expected.shape,
            dtype: expected.dtype,
            location: 'host',
            ownership: 'transfer',
            data: laneData,
          }));
        }
        const scheduling = Object.freeze({
          plan: 'runtime-batch',
          batchSize,
          lane,
          // Every split result points at the one physical provider execution.
          // Only the owner lane contributes to additive invocation totals.
          dispatchId: batchResult.report.executionId,
          dispatchOwnerLane: 0,
          invocationAccounting: 'dispatch-owner/v1',
          trueBackendInvocations: lane === 0 ? 1 : 0,
        });
        const backendReport = Object.freeze({
          ...(batchResult.report.backendReport ?? {}),
          scheduling,
        });
        const report: ExecutionReport = Object.freeze({
          ...batchResult.report,
          executionId: executionIdentity(),
          shapeSignature: plan.signature,
          adapterRevisionId: null,
          adapterRevisionIds: EMPTY_ADAPTER_REVISION_IDS,
          backendReport,
        });
        created.push(createExecutionResult(
          this.backend,
          Object.freeze({ outputs: Object.freeze(outputs), backendReport }),
          report,
          plan.outputs,
        ));
      }
      return Object.freeze(created);
    } catch (error) {
      await Promise.allSettled(created.map((result) => result.close()));
      throw error;
    }
  }

  async createContext(options: ExecutionContextOptions = {}): Promise<ExecutionContext> {
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'CompiledModel is closing or closed.', {
        phase: 'lifecycle', backend: this.backend,
      });
    }
    try {
      assertExecutionContextOptions(options);
      if (options.adapter != null) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          'Logical model snapshots do not contain adapter revisions.', {
            phase: 'compilation', backend: this.backend,
          });
      }
      if (options.adapterBatchDimension != null &&
          !Object.prototype.hasOwnProperty.call(
            this.#snapshot.graph.dimensions,
            options.adapterBatchDimension,
          )) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Adapter batch dimension '${options.adapterBatchDimension}' is not declared by the logical model.`, {
            phase: 'compilation', backend: this.backend,
          });
      }
      const decode = options.decode;
      if (decode?.rowMode !== undefined && !decodeRowModes.includes(decode.rowMode)) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `Execution context decode rowMode '${String(decode.rowMode)}' is invalid.`, {
            phase: 'compilation', backend: this.backend,
          });
      }
      if (decode?.requireIncremental !== undefined &&
          typeof decode.requireIncremental !== 'boolean') {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          'Execution context decode requireIncremental must be boolean.', {
            phase: 'compilation', backend: this.backend,
          });
      }
      if (decode?.changedInputs !== undefined && decode.changedInputs !== null) {
        if (!Array.isArray(decode.changedInputs)) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            'Execution context decode changedInputs must be an array or null.', {
              phase: 'compilation', backend: this.backend,
            });
        }
        const seen = new Set<string>();
        for (const name of decode.changedInputs) {
          if (typeof name !== 'string' || !this.#snapshot.inputNames.includes(name) ||
              seen.has(name)) {
            throw new VolvoxAIError('INVALID_ARGUMENT',
              'Execution context decode changedInputs must contain unique declared public input names.', {
                phase: 'compilation', backend: this.backend,
              });
          }
          seen.add(name);
        }
      }
    } catch (error) {
      throw runtimeError(error, 'INVALID_ARGUMENT', 'Execution context options are invalid.', {
        phase: 'compilation', backend: this.backend,
      });
    }
    const releaseContext = this.#retainContext();
    let candidate: unknown = null;
    let backendContext: BackendProviderExecutionContext | null = null;
    let invariantResources: BackendProviderInvariantResourceLease | null = null;
    let invariantResourceRelease: (() => void) | null = null;
    let ownershipReleased = false;
    const releaseOwnership = () => {
      if (ownershipReleased) return;
      ownershipReleased = true;
      try {
        invariantResourceRelease?.();
      } finally {
        releaseContext();
      }
    };
    try {
      const decode = options.decode;
      const initialPlan = options.bankResidency === undefined && this.#snapshot.staticShapePlan
        ? this.#snapshot.staticShapePlan
        : resolveMinimumGraphShapes(
          this.#snapshot.graph,
          this.#snapshot.quantizationByTensor,
          options.bankResidency,
        );
      if (this.#compiled.invariantResources !== this.#invariantResources.owner) {
        throw new VolvoxAIError('ABI_UNSUPPORTED',
          `Backend '${this.backend}' changed its compiled invariant resource owner.`, {
            phase: 'compilation', backend: this.backend,
          });
      }
      const invariantOwner = this.#invariantResources.owner;
      if (invariantOwner.deviceEpoch !== this.#invariantResources.deviceEpoch) {
        throw new VolvoxAIError('DEVICE_LOST',
          `Backend '${this.backend}' invalidated its compiled resource generation.`, {
          phase: 'compilation', backend: this.backend,
        });
      }
      const rawInvariantResources = this.#invariantResources.open.call(invariantOwner);
      const rawInvariantRelease = (rawInvariantResources as
        Partial<BackendProviderInvariantResourceLease> | null)?.release;
      if (typeof rawInvariantRelease === 'function') {
        invariantResourceRelease = () => rawInvariantRelease.call(rawInvariantResources);
      }
      invariantResources = assertProviderInvariantResourceLease(
        rawInvariantResources,
        invariantOwner,
        this.backend,
        this.#invariantResources,
      );
      const providerOptions = Object.freeze({
        invariantResources,
        initialPlan,
        ...(decode === undefined
          ? {}
          : {
            decode: Object.freeze({
              ...(decode.changedInputs === undefined
                ? {}
                : {
                  changedInputs: decode.changedInputs === null
                    ? null
                    : Object.freeze([...decode.changedInputs]),
                }),
              ...(decode.rowMode === undefined ? {} : { rowMode: decode.rowMode }),
              ...(decode.requireIncremental === undefined
                ? {}
                : { requireIncremental: decode.requireIncremental }),
              ...(decode.lanes === undefined ? {} : { lanes: decode.lanes }),
            }),
          }),
      });
      candidate = await this.#compiled.createContext(providerOptions);
      backendContext = assertProviderExecutionContext(
        candidate,
        this.backend,
      );
      return new ExecutionContext(
        this.#snapshot,
        backendContext,
        this.#capabilities,
        this.report,
        options,
        this.#onDiagnostic,
        this.#memoryCapture,
        this.#resultBudget,
        releaseOwnership,
      );
    } catch (error) {
      try {
        const close = backendContext?.close ||
          (candidate as Partial<BackendProviderExecutionContext> | null)?.close;
        if (typeof close === 'function') await close.call(backendContext || candidate);
      } catch {
        // Preserve the provider contract/construction failure.
      }
      try {
        releaseOwnership();
      } catch {
        // Preserve the provider contract/construction failure even when a
        // malformed raw invariant lease has a hostile release callback.
      }
      throw runtimeError(error, 'BACKEND_UNSUPPORTED',
        `Backend '${this.backend}' could not create an execution context.`, {
          phase: 'compilation', backend: this.backend,
        });
    }
  }

  close(): Promise<void> {
    if (this.#closePromise) return this.#closePromise;
    this.#state = 'closing';
    this.#closePromise = (async () => {
      let firstError: unknown = null;
      try {
        try {
          await this.#runtime._retireScheduledTarget(this);
        } catch (error) {
          firstError = error;
        }
        if (this.#retainedContexts > 0) {
          await new Promise<void>((resolve) => {
            this.#resolveContextDrain = resolve;
          });
        }
        try {
          await this.#compiled.close();
        } catch (error) {
          if (firstError === null) firstError = error;
        } finally {
          try {
            await this.#invariantResources.close.call(this.#invariantResources.owner);
          } catch (error) {
            if (firstError === null) firstError = error;
          }
        }
        if (firstError !== null) throw runtimeError(firstError, 'EXECUTION_FAILED',
          `Backend '${this.backend}' compiled-model close failed.`, {
            phase: 'lifecycle', backend: this.backend,
          });
      } finally {
        this.#state = 'closed';
        this.#releaseRuntime();
      }
    })();
    return this.#closePromise;
  }

  dispose(): Promise<void> { return this.close(); }

  #retainContext(): () => void {
    this.#retainedContexts++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.#retainedContexts--;
      if (this.#retainedContexts === 0) {
        const resolve = this.#resolveContextDrain;
        this.#resolveContextDrain = null;
        resolve?.();
      }
    };
  }
}

export interface ExecutionContextDecode {
  seed(inputs: ExecutionInputs, options?: DecodeExecutionOptions): Promise<ExecutionResult>;
  /** A step may supply only its effective changed-input set after a successful seed. */
  step(inputs: ExecutionInputs, options?: DecodeExecutionOptions): Promise<ExecutionResult>;
  reset(): Promise<void>;
}

class ContextDecode implements ExecutionContextDecode {
  readonly #context: ExecutionContext;

  constructor(context: ExecutionContext) {
    this.#context = context;
  }

  seed(inputs: ExecutionInputs, options: DecodeExecutionOptions = {}): Promise<ExecutionResult> {
    return this.#context._decodeExecute('seed', inputs, options);
  }

  step(inputs: ExecutionInputs, options: DecodeExecutionOptions = {}): Promise<ExecutionResult> {
    return this.#context._decodeExecute('step', inputs, options);
  }

  reset(): Promise<void> {
    return this.#context._decodeReset();
  }
}

export class ExecutionContext {
  readonly id: string;
  readonly backend: string;
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly decode: ExecutionContextDecode;
  readonly #snapshot: Model;
  readonly #backendContext: BackendProviderExecutionContext;
  readonly #capabilities: BackendProviderCapabilities;
  readonly #compilationReport: CompilationReport;
  readonly #onDiagnostic: ((event: RuntimeDiagnostic) => void) | null;
  readonly #memoryCapture: RuntimeMemoryCaptureSession | null;
  readonly #resultBudget: RuntimeResultBudget;
  readonly #releaseCompiled: () => void;
  #adapter: ExecutionContextOptions['adapter'];
  readonly #adapterBatchDimension: string | null;
  readonly #decodeChangedInputs: readonly string[] | null;
  #retainedDecodeInputs: ExecutionInputs | null = null;
  #tail: Promise<void> = Promise.resolve();
  readonly #bankResidency: Readonly<Record<string, readonly number[]>> | undefined;
  #state: 'open' | 'closing' | 'closed' = 'open';
  #closePromise: Promise<void> | null = null;

  constructor(
    snapshot: Model,
    backendContext: BackendProviderExecutionContext,
    capabilities: BackendProviderCapabilities,
    compilationReport: CompilationReport,
    options: ExecutionContextOptions,
    onDiagnostic: ((event: RuntimeDiagnostic) => void) | null,
    memoryCapture: RuntimeMemoryCaptureSession | null,
    resultBudget: RuntimeResultBudget,
    releaseCompiled: () => void,
  ) {
    this.id = identity('context');
    this.backend = backendContext.backendName;
    this.definitionId = snapshot.definitionId;
    this.topologyRevision = snapshot.topologyRevision;
    this.weightRevision = snapshot.weightRevision;
    this.weightRevisionId = snapshot.weightRevisionId;
    this.#snapshot = snapshot;
    this.#backendContext = backendContext;
    this.#capabilities = capabilities;
    this.#compilationReport = compilationReport;
    this.#onDiagnostic = onDiagnostic;
    this.#memoryCapture = memoryCapture;
    this.#resultBudget = resultBudget;
    this.#releaseCompiled = releaseCompiled;
    this.#adapter = cloneSelector(options.adapter);
    this.#adapterBatchDimension = options.adapterBatchDimension ?? null;
    this.#bankResidency = options.bankResidency === undefined
      ? undefined
      : Object.freeze(Object.fromEntries(Object.entries(options.bankResidency).map(
        ([name, slots]) => [name, Object.freeze([...slots])]))); 
    this.#decodeChangedInputs = options.decode?.changedInputs == null
      ? null
      : Object.freeze([...options.decode.changedInputs]);
    this.decode = new ContextDecode(this);
  }

  get closed(): boolean { return this.#state === 'closed'; }
  get adapterRevisionId(): null { return null; }

  execute(
    inputs: ExecutionInputs,
    options: ExecutionOptions = EMPTY_EXECUTION_OPTIONS,
  ): Promise<ExecutionResult> {
    return this.#execute(inputs, options, null);
  }

  /** Module-private capability: only CompiledModel can name this symbol. */
  [EXECUTE_PREPARED_SHAPE_PLAN](
    inputs: ExecutionInputs,
    options: Readonly<ExecutionOptions>,
    plan: ResolvedShapePlan,
    outputBudgetLease: RuntimeResultBudgetLease | null = null,
  ): Promise<ExecutionResult> {
    if (!hasCanonicalResolvedShapePlanProvenance(
      plan,
      this.#snapshot.graph,
      this.#snapshot.quantizationByTensor,
    )) {
      outputBudgetLease?.releaseAll();
      return Promise.reject(new VolvoxAIError('ABI_UNSUPPORTED',
        'Prepared scheduled execution supplied a non-canonical shape plan.', {
          phase: 'execution', backend: this.backend,
        }));
    }
    const execution = this.#execute(inputs, options, plan, outputBudgetLease);
    return outputBudgetLease === null
      ? execution
      : execution.catch((error) => {
          outputBudgetLease.releaseAll();
          throw error;
        });
  }

  #execute(
    inputs: ExecutionInputs,
    options: Readonly<ExecutionOptions>,
    preparedPlan: ResolvedShapePlan | null,
    preReservedOutputBudgetLease: RuntimeResultBudgetLease | null = null,
  ): Promise<ExecutionResult> {
    let deviceInputs: Readonly<Record<string, DeviceTensorInputLease>>;
    try {
      deviceInputs = this.#acquireDeviceInputs(inputs, 'execute');
    } catch (error) {
      return Promise.reject(error);
    }
    return this.#enqueue(() => this.#run('execute', inputs,
      (request) => this.#backendContext.execute(request),
      options,
      deviceInputs,
      preparedPlan,
      preReservedOutputBudgetLease,
    )).finally(() => {
      for (const lease of Object.values(deviceInputs)) releaseDeviceTensorInputLease(lease);
    });
  }

  selectAdapter(adapter: ExecutionContextOptions['adapter']): Promise<void> {
    return this.#enqueue(() => {
      if (adapter != null) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          'Logical model snapshots do not contain adapter revisions.', {
          phase: 'execution', backend: this.backend,
        });
      }
      this.#adapter = adapter;
    });
  }

  _decodeExecute(
    operation: 'seed' | 'step',
    inputs: ExecutionInputs,
    options: DecodeExecutionOptions,
  ): Promise<ExecutionResult> {
    try {
      this.#acquireDeviceInputs(inputs, operation);
    } catch (error) {
      return Promise.reject(error);
    }
    return this.#enqueue(() => this.#run(operation, inputs, async (request) => {
      const execute = operation === 'seed'
        ? this.#backendContext.decodeSeed
        : this.#backendContext.decodeStep;
      if (typeof execute !== 'function') {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `Backend '${this.backend}' does not expose decode ${operation}.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      return execute.call(this.#backendContext, request);
    }, options, EMPTY_DEVICE_INPUTS));
  }

  #acquireDeviceInputs(
    inputs: ExecutionInputs,
    operation: ExecutionDecodeState['operation'],
  ): Readonly<Record<string, DeviceTensorInputLease>> {
    if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs) ||
        ArrayBuffer.isView(inputs)) return EMPTY_DEVICE_INPUTS;
    const found: Record<string, DeviceTensorInputLease> = {};
    try {
      for (const name of Object.getOwnPropertyNames(inputs)) {
        const view = inputs[name];
        if (!view || typeof view !== 'object') continue;
        const reference = inspectDeviceTensorReference(view.data);
        if (!reference) continue;
        if (operation !== 'execute') {
          throw new VolvoxAIError('BACKEND_UNSUPPORTED',
            'Device TensorResult inputs are supported only by ordinary execute, not decode seed/step.', {
              phase: 'execution', backend: this.backend,
            });
        }
        if (this.backend !== 'webgpu') {
          throw new VolvoxAIError('BACKEND_UNSUPPORTED',
            `Backend '${this.backend}' does not accept device TensorResult inputs; read the tensor to host storage first.`, {
              phase: 'execution', backend: this.backend,
            });
        }
        found[name] = acquireDeviceTensorInputLease(view.data);
      }
      return Object.keys(found).length === 0
        ? EMPTY_DEVICE_INPUTS
        : Object.freeze(found);
    } catch (error) {
      for (const lease of Object.values(found)) releaseDeviceTensorInputLease(lease);
      throw error;
    }
  }

  _decodeReset(): Promise<void> {
    return this.#enqueue(async () => {
      if (typeof this.#backendContext.decodeReset !== 'function') {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `Backend '${this.backend}' does not expose decode reset.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      await this.#backendContext.decodeReset();
      this.#retainedDecodeInputs = null;
    });
  }

  close(): Promise<void> {
    if (this.#closePromise) return this.#closePromise;
    this.#state = 'closing';
    const close = this.#tail.then(
      () => this.#backendContext.close(),
      () => this.#backendContext.close(),
    ).then(() => {
      this.#state = 'closed';
    }, (error) => {
      this.#state = 'closed';
      throw runtimeError(error, 'EXECUTION_FAILED',
        `Backend '${this.backend}' context close failed.`, {
          phase: 'lifecycle', backend: this.backend,
        });
    }).finally(() => {
      this.#retainedDecodeInputs = null;
      this.#releaseCompiled();
    });
    this.#tail = close.then(() => undefined, () => undefined);
    this.#closePromise = close;
    return close;
  }

  dispose(): Promise<void> { return this.close(); }

  #enqueue<TResult>(operation: () => Promise<TResult> | TResult): Promise<TResult> {
    if (this.#state !== 'open') {
      return Promise.reject(new VolvoxAIError('HANDLE_DISPOSED', 'ExecutionContext is closing or closed.', {
        phase: 'lifecycle', backend: this.backend,
      }));
    }
    const pending = this.#tail.then(operation, operation);
    this.#tail = pending.then(() => undefined, () => undefined);
    return pending;
  }

  async #run(
    executionOperation: ExecutionDecodeState['operation'],
    inputs: ExecutionInputs,
    operation: (request: BackendResolvedExecutionRequest) => Promise<BackendExecutionSnapshot>,
    options: DecodeExecutionOptions,
    deviceInputs: Readonly<Record<string, DeviceTensorInputLease>>,
    preparedPlan: ResolvedShapePlan | null = null,
    preReservedOutputBudgetLease: RuntimeResultBudgetLease | null = null,
  ): Promise<ExecutionResult> {
    const executionId = executionIdentity();
    const executionStarted = monotonicMilliseconds();
    let snapshot: BackendExecutionSnapshot | null = null;
    let snapshotTransferred = false;
    let reportedBackendReport: Readonly<Record<string, unknown>> | null = null;
    let resolvedOptions: Readonly<DecodeExecutionOptions> = EMPTY_EXECUTION_OPTIONS;
    let adapterRevisionIds: readonly (string | null)[] = EMPTY_ADAPTER_REVISION_IDS;
    let plan: ResolvedShapePlan | null = null;
    let outputBudgetLease = preReservedOutputBudgetLease;
    let outputBudgetTransferred = false;
    let shapeBindTimeMs: number | null = null;
    let providerTimeMs: number | null = null;
    try {
      assertExecutionOptions(options, executionOperation);
      if (options.position !== undefined &&
          (!Number.isSafeInteger(options.position) || options.position < 0)) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          'Decode position must be a non-negative safe integer.', {
            phase: 'execution', backend: this.backend,
          });
      }
      if (options.positions !== undefined) {
        /* Refused rather than resolved by precedence: a caller that sent both
         * has two beliefs about how many slots this context owns, and picking
         * one of them silently decodes a batch it did not ask for. */
        if (options.position !== undefined) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            'Decode accepts a scalar position or per-lane positions, not both.', {
              phase: 'execution', backend: this.backend,
            });
        }
        /* Three spellings, and the decode lifecycle separates them against the
         * declared lane count: a position advances the lane, `null` idles a
         * lane that still holds its request, and `-1` parks a lane that holds
         * none. A step where no lane advances is not a step. */
        if (!Array.isArray(options.positions) || options.positions.length === 0 ||
            options.positions.some((position) => position !== null &&
              (!Number.isSafeInteger(position) || position < -1))) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            'Decode positions must be a non-empty array of non-negative safe integers, ' +
            'null to idle a lane, or -1 to park one.', {
              phase: 'execution', backend: this.backend,
            });
        }
      }
      const optionChangedInputs = this.#normalizeChangedInputs(options.changedInputs);
      let bindingInputs = inputs;
      let effectiveStepChangedInputs: readonly string[] | null = null;
      if (executionOperation === 'step') {
        const prepared = this.#prepareDecodeStepInputs(inputs, optionChangedInputs);
        bindingInputs = prepared.inputs;
        effectiveStepChangedInputs = prepared.changedInputs;
      }
      for (const [name, lease] of Object.entries(deviceInputs)) {
        if (!deviceTensorInputLeaseMatches(lease, bindingInputs[name]?.data)) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            `Device tensor input '${name}' changed after execute accepted it.`, {
              phase: 'execution', backend: this.backend,
            });
        }
      }
      for (const name of this.#snapshot.inputNames) {
        if (inspectDeviceTensorReference(bindingInputs[name]?.data) && !deviceInputs[name]) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            `Device tensor input '${name}' was supplied after execute acceptance and has no ownership lease.`, {
              phase: 'execution', backend: this.backend,
            });
        }
      }
      if (preparedPlan !== null) {
        plan = preparedPlan;
        shapeBindTimeMs = 0;
      } else {
        const shapeBindStarted = monotonicMilliseconds();
        try {
          plan = this.#snapshot.bindShapes(bindingInputs, this.#bankResidency);
        } catch (error) {
          shapeBindTimeMs = elapsedMilliseconds(shapeBindStarted);
          throw runtimeError(error, 'INVALID_ARGUMENT',
            'Execution inputs do not satisfy the complete logical shape contract.', {
              phase: 'execution', backend: this.backend,
            });
        }
        shapeBindTimeMs = elapsedMilliseconds(shapeBindStarted);
      }
      // Public DIRECT/context/decode paths reserve from the Runtime-wide
      // ledger after exact shape resolution and before provider mutation.
      // Scheduled prepared plans already carry an admission-time reservation.
      if (outputBudgetLease !== null &&
          outputBudgetLease.outputBytes !== exactPlanOutputBytes(plan)) {
        throw new VolvoxAIError('ABI_UNSUPPORTED',
          'Prepared execution result reservation does not match its exact shape plan.', {
            phase: 'execution', backend: this.backend,
          });
      }
      if (preparedPlan === null && outputBudgetLease === null) {
        outputBudgetLease = this.#resultBudget.reserve(
          exactPlanOutputBytes(plan), this.backend,
        );
      }

      const mutableOptions: {
        -readonly [Key in keyof DecodeExecutionOptions]: DecodeExecutionOptions[Key]
      } = options === EMPTY_EXECUTION_OPTIONS && this.#adapter === undefined &&
          effectiveStepChangedInputs === null
        ? EMPTY_EXECUTION_OPTIONS
        : { ...options };
      if (!Object.prototype.hasOwnProperty.call(mutableOptions, 'adapter') &&
          !Object.prototype.hasOwnProperty.call(mutableOptions, 'adapters') &&
          this.#adapter !== undefined) {
        mutableOptions.adapter = this.#adapter;
      }

      let adapterSelection: BackendResolvedAdapterSelection;
      if (Object.prototype.hasOwnProperty.call(mutableOptions, 'adapters')) {
        if (!Array.isArray(mutableOptions.adapters)) {
          throw new VolvoxAIError('INVALID_ARGUMENT', 'Execution adapters must be an array.', {
            phase: 'execution', backend: this.backend,
          });
        }
        if (this.#adapterBatchDimension === null) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            'Batch-indexed adapters require an explicit context adapterBatchDimension.', {
              phase: 'execution', backend: this.backend,
            });
        }
        const batchSize = plan.symbols[this.#adapterBatchDimension];
        if (!Number.isSafeInteger(batchSize) || batchSize <= 0) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            `Resolved plan does not bind adapter batch dimension '${this.#adapterBatchDimension}'.`, {
              phase: 'execution', backend: this.backend,
            });
        }
        if (mutableOptions.adapters.length !== batchSize) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            `Execution adapters length ${mutableOptions.adapters.length} does not match resolved batch ${batchSize}.`, {
              phase: 'execution', backend: this.backend,
            });
        }
        const selectors = Object.freeze(mutableOptions.adapters.map((selector) => {
          if (selector != null) {
            throw new VolvoxAIError('INVALID_ARGUMENT',
              'Logical model snapshots do not contain adapter revisions.', {
                phase: 'execution', backend: this.backend,
              });
          }
          return null;
        }));
        mutableOptions.adapters = selectors;
        adapterRevisionIds = Object.freeze(selectors.map(() => null));
        adapterSelection = Object.freeze({ batchSize, selectors });
      } else {
        const selector = Object.prototype.hasOwnProperty.call(mutableOptions, 'adapter')
          ? mutableOptions.adapter
          : undefined;
        if (selector != null) {
          throw new VolvoxAIError('INVALID_ARGUMENT',
            'Logical model snapshots do not contain adapter revisions.', {
              phase: 'execution', backend: this.backend,
            });
        }
        if (Object.prototype.hasOwnProperty.call(mutableOptions, 'adapter')) {
          mutableOptions.adapter = selector;
        }
        adapterSelection = Object.prototype.hasOwnProperty.call(mutableOptions, 'adapter')
          ? Object.freeze({ batchSize: null, selectors: EMPTY_ADAPTER_SELECTORS })
          : EMPTY_ADAPTER_SELECTION;
      }

      if (executionOperation === 'step') {
        mutableOptions.changedInputs = effectiveStepChangedInputs!;
      } else if (optionChangedInputs !== undefined) {
        mutableOptions.changedInputs = optionChangedInputs;
      }
      resolvedOptions = mutableOptions === EMPTY_EXECUTION_OPTIONS
        ? EMPTY_EXECUTION_OPTIONS
        : Object.freeze(mutableOptions);

      const normalizedInputRecord = Object.create(null) as Record<
        string,
        ExecutionInputs[string]
      >;
      const mutableInputDescriptors: ResolvedTensorDescriptor[] = [];
      for (const name of this.#snapshot.inputNames) {
        const descriptor = plan.tensors[name];
        normalizedInputRecord[name] = Object.freeze({
          data: bindingInputs[name].data,
          shape: descriptor.shape,
        });
        mutableInputDescriptors.push(descriptor);
      }
      const normalizedInputs = Object.freeze(normalizedInputRecord) as ExecutionInputs;
      const inputDescriptors = Object.freeze(mutableInputDescriptors);
      // Prepare fallible host clones before a provider can commit device-side
      // decode mutation. Publication remains conditional on provider success.
      const stagedRetainedDecodeInputs = executionOperation === 'seed'
        ? this.#cloneDecodeInputs(normalizedInputs)
        : executionOperation === 'step'
          ? this.#cloneDecodeInputs(
            normalizedInputs,
            effectiveStepChangedInputs!,
            this.#retainedDecodeInputs,
          )
          : null;
      let executionCommitted = false;
      const commitCoreExecution = () => {
        if (executionCommitted) return;
        executionCommitted = true;
        this.#retainedDecodeInputs = null;
      };
      let providerCommitActive = true;
      const commitExecution = () => {
        if (!providerCommitActive) return;
        commitCoreExecution();
      };
      const request: BackendResolvedExecutionRequest = Object.freeze({
        operation: executionOperation,
        commitExecution,
        signature: plan.signature,
        inputs: normalizedInputs,
        deviceInputs,
        inputDescriptors,
        tensors: plan.tensors,
        outputDescriptors: plan.outputs,
        plan,
        adapters: adapterSelection,
        options: resolvedOptions,
      });

      // This is the first provider call in the execution. Every complete input,
      // adapter, and graph-wide shape check above is therefore pre-mutation.
      const providerStarted = monotonicMilliseconds();
      try {
        snapshot = await operation(request);
      } finally {
        // Provider requests are retainable objects. Revoke the public hook as
        // soon as this call settles so a late callback cannot clear decode
        // state published by a later FIFO operation.
        providerCommitActive = false;
        providerTimeMs = elapsedMilliseconds(providerStarted);
      }
      // External providers that do not expose an earlier mutation boundary
      // still commit no later than successful return. Decode candidates are
      // published only after that complete provider success.
      commitCoreExecution();
      if (executionOperation !== 'execute') {
        this.#retainedDecodeInputs = stagedRetainedDecodeInputs;
      }
      const backendReport = normalizeBackendReport(snapshot.backendReport);
      reportedBackendReport = backendReport;
      const route = executionRouteEvidence(
        this.#capabilities,
        this.#compilationReport,
        backendReport,
      );
      const decodeState = executionDecodeState(executionOperation, resolvedOptions, backendReport);
      const reportBase: ExecutionReport = Object.freeze({
        executionId,
        contextId: this.id,
        backend: this.backend,
        device: this.#compilationReport.selectedDevice,
        outcome: 'success',
        topologyRevision: this.topologyRevision,
        weightRevision: this.weightRevision,
        weightRevisionId: this.weightRevisionId,
        shapeSignature: plan.signature,
        adapterRevisionId: adapterRevisionIds.length === 1 ? adapterRevisionIds[0] : null,
        adapterRevisionIds,
        shapeBindTimeMs,
        providerTimeMs,
        executionTimeMs: elapsedMilliseconds(executionStarted),
        routeEvidence: route,
        decodeState,
        operatorFallback: this.#capabilities.operatorFallback,
        backendReport,
      });
      const result = this.#memoryCapture === null
        ? createExecutionResult(this.backend, snapshot, reportBase, plan.outputs)
        : createExecutionResult(
          this.backend,
          snapshot,
          reportBase,
          plan.outputs,
          (ownedResult) => {
            const backendMemory = captureBackendMemorySnapshot(
              this.#memoryCapture,
              this.#backendContext,
              MemorySnapshotPoint.After,
              this.id,
              ownedResult.id,
            );
            const memoryEvidence = captureMemoryEvidence(this.#memoryCapture, {
              stage: executionOperation === 'execute'
                ? OperationStage.Execute
                : OperationStage.Decode,
              point: MemorySnapshotPoint.After,
              subject: Object.freeze({
                kind: MemoryOwnerKind.ExecutionContext,
                ownerId: this.id,
              }),
              backend: this.backend,
              device: this.#compilationReport.selectedDevice,
              ...(backendMemory === null ? {} : {
                resources: backendMemory.resources,
                resourceInventory: backendMemory.resourceInventory,
              }),
            });
            // reportBase already carries the elapsed time measured before this
            // opt-in collector ran. Re-measuring here would charge sampling
            // cost to the execution the same report is timing, so an opt-in
            // diagnostic would silently change a published performance number.
            return Object.freeze({
              ...reportBase,
              ...(memoryEvidence === undefined ? {} : { memoryEvidence }),
            });
          },
        );
      snapshotTransferred = true;
      if (outputBudgetLease !== null) {
        try {
          registerExecutionResultCloseFinalizer(
            result,
            () => outputBudgetLease!.releaseAll(),
          );
          outputBudgetTransferred = true;
        } catch (error) {
          await result.close().catch(() => undefined);
          throw error;
        }
      }
      const report = result.report;
      this.#emit({ kind: 'execution', report });
      return result;
    } catch (error) {
      if (!outputBudgetTransferred) outputBudgetLease?.releaseAll();
      const backendMemory = captureBackendMemorySnapshot(
        this.#memoryCapture,
        this.#backendContext,
        MemorySnapshotPoint.Failure,
        this.id,
      );
      const memoryEvidence = captureMemoryEvidence(this.#memoryCapture, {
        stage: executionOperation === 'execute' ? OperationStage.Execute : OperationStage.Decode,
        point: MemorySnapshotPoint.Failure,
        subject: Object.freeze({
          kind: MemoryOwnerKind.ExecutionContext,
          ownerId: this.id,
        }),
        backend: this.backend,
        device: this.#compilationReport.selectedDevice,
        ...(backendMemory === null ? {} : {
          resources: backendMemory.resources,
          resourceInventory: backendMemory.resourceInventory,
        }),
      });
      if (snapshot && !snapshotTransferred) releaseBackendExecutionSnapshot(snapshot);
      const failure = runtimeError(error, 'EXECUTION_FAILED',
        `Backend '${this.backend}' execution failed.`, {
          phase: 'execution', backend: this.backend,
        });
      const failureReport: ExecutionFailureReport = Object.freeze({
        contextId: this.id,
        executionId,
        backend: this.backend,
        device: this.#compilationReport.selectedDevice,
        outcome: 'failed',
        code: failure.code,
        message: failure.message,
        topologyRevision: this.topologyRevision,
        weightRevision: this.weightRevision,
        weightRevisionId: this.weightRevisionId,
        shapeSignature: plan?.signature ?? null,
        adapterRevisionId: adapterRevisionIds.length === 1 ? adapterRevisionIds[0] : null,
        adapterRevisionIds,
        shapeBindTimeMs,
        providerTimeMs,
        executionTimeMs: elapsedMilliseconds(executionStarted),
        routeEvidence: executionFailureRouteEvidence(
          this.#capabilities,
          this.#compilationReport,
          reportedBackendReport,
          failure.node,
        ),
        decodeState: executionDecodeState(executionOperation, resolvedOptions, null),
        ...(memoryEvidence === undefined ? {} : { memoryEvidence }),
      });
      this.#emit({ kind: 'execution-error', report: failureReport });
      throw new VolvoxAIError(failure.code, failure.message, {
        phase: failure.phase,
        backend: failure.backend,
        node: failure.node,
        report: failureReport,
        cause: failure,
      });
    }
  }

  #emit(event: RuntimeDiagnostic): void {
    try {
      this.#onDiagnostic?.(event);
    } catch {
      // Application diagnostics must never change runtime control flow.
    }
  }

  #normalizeChangedInputs(value: DecodeExecutionOptions['changedInputs']):
    readonly string[] | undefined {
    if (value === undefined) return undefined;
    if (!Array.isArray(value)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'Decode changedInputs must be an array.', {
        phase: 'execution', backend: this.backend,
      });
    }
    const seen = new Set<string>();
    const normalized = value.map((name) => {
      if (typeof name !== 'string' || !this.#snapshot.inputNames.includes(name) ||
          seen.has(name)) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          'Decode changedInputs must contain unique declared public input names.', {
            phase: 'execution', backend: this.backend,
          });
      }
      seen.add(name);
      return name;
    });
    return Object.freeze(normalized);
  }

  #prepareDecodeStepInputs(
    supplied: ExecutionInputs,
    optionChangedInputs: readonly string[] | undefined,
  ): Readonly<{ inputs: ExecutionInputs; changedInputs: readonly string[] }> {
    const retained = this.#retainedDecodeInputs;
    if (retained === null) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Decode step requires a successful seed on this execution context.', {
          phase: 'execution', backend: this.backend,
        });
    }
    if (!supplied || typeof supplied !== 'object' || Array.isArray(supplied) ||
        ArrayBuffer.isView(supplied) || Object.getOwnPropertySymbols(supplied).length !== 0) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Decode step inputs must be a named tensor-view record.', {
          phase: 'execution', backend: this.backend,
        });
    }
    const suppliedNames = Object.getOwnPropertyNames(supplied);
    const declared = new Set(this.#snapshot.inputNames);
    if (suppliedNames.some((name) => !declared.has(name))) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Decode step inputs contain an undeclared public input.', {
          phase: 'execution', backend: this.backend,
        });
    }
    const complete = suppliedNames.length === this.#snapshot.inputNames.length &&
      this.#snapshot.inputNames.every((name) => Object.prototype.hasOwnProperty.call(supplied, name));
    const expected = optionChangedInputs ?? this.#decodeChangedInputs;
    const changedInputs = expected === null
      ? Object.freeze(this.#snapshot.inputNames.filter((name) =>
          Object.prototype.hasOwnProperty.call(supplied, name)))
      : expected;
    const changed = new Set(changedInputs);
    const missingChanged = changedInputs.filter((name) =>
      !Object.prototype.hasOwnProperty.call(supplied, name));
    const extraPartial = complete
      ? []
      : suppliedNames.filter((name) => !changed.has(name));
    if (missingChanged.length !== 0 || extraPartial.length !== 0) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Partial decode step inputs must exactly match the effective changedInputs set.', {
          phase: 'execution', backend: this.backend,
        });
    }
    const merged: Record<string, ExecutionInputs[string]> = {};
    for (const name of this.#snapshot.inputNames) {
      merged[name] = changed.has(name) ? supplied[name] : retained[name];
    }
    return Object.freeze({
      inputs: Object.freeze(merged),
      changedInputs: Object.freeze([...changedInputs]),
    });
  }

  #cloneDecodeInputs(
    inputs: ExecutionInputs,
    changedInputs: readonly string[] = this.#snapshot.inputNames,
    previous: ExecutionInputs | null = null,
  ): ExecutionInputs {
    const changed = new Set(changedInputs);
    const retained: Record<string, ExecutionInputs[string]> = {};
    for (const name of this.#snapshot.inputNames) {
      if (!changed.has(name) && previous !== null) {
        retained[name] = previous[name];
        continue;
      }
      const input = inputs[name];
      retained[name] = Object.freeze({
        data: cloneRuntimeData(input.data),
        shape: Object.freeze([...input.shape]),
      });
    }
    return Object.freeze(retained);
  }
}
