import { Model } from './Model.js';
import {
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
  releaseBackendExecutionSnapshot,
  type BackendExecutionSnapshot,
  type ExecutionDecodeState,
  type ExecutionReport,
  type ExecutionRouteEvidence,
} from './ExecutionResult.js';
import { VolvoxAIError, runtimeError, type VolvoxAIErrorCode } from './RuntimeErrors.js';
import {
  assertBackendProvider,
  assertProviderCompiledModel,
  assertProviderExecutionContext,
  createBackendCompileInput,
  createBackendDeviceIdentity,
  type BackendProvider,
  type BackendProviderCapabilities,
  type BackendProviderCompiledModel,
  type BackendProviderExecutionContext,
  type BackendResolvedAdapterSelection,
  type BackendResolvedExecutionRequest,
} from '../backends/BackendProvider.js';
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
  OperatorFallbackValue,
} from '../generated/volvoxaiEnums.js';
import { decodeRowModes } from '../generated/volvoxaiEnums.js';
import {
  acquireDeviceTensorInputLease,
  deviceTensorInputLeaseMatches,
  inspectDeviceTensorReference,
  releaseDeviceTensorInputLease,
  type DeviceTensorInputLease,
} from '../ops/deviceTensorReference.js';

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
  readonly candidates: readonly CompilationCandidateReport[];
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
    candidates: Object.freeze(candidates.map(freezeCandidate)),
  });
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
const DECODE_CONTEXT_OPTION_NAMES = new Set(['changedInputs', 'rowMode', 'requireIncremental']);
const EXECUTION_OPTION_NAMES = new Set(['adapter', 'adapters']);
const DECODE_EXECUTION_OPTION_NAMES = new Set([
  'adapter', 'adapters', 'changedInputs', 'position',
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

/** Context-aware runtime returned by `VolvoxAI.createRuntime()`. */
export class Runtime {
  readonly #providers = new Map<string, ProviderEntry>();
  readonly #initializationFailures = new Map<string, string>();
  readonly #order: string[] = [];
  readonly #onDiagnostic: ((event: RuntimeDiagnostic) => void) | null;
  #state: 'open' | 'closing' | 'closed' = 'open';
  #retainedChildren = 0;
  #resolveChildDrain: (() => void) | null = null;
  #closePromise: Promise<void> | null = null;

  constructor({ onDiagnostic = null }: RuntimeOptions = {}) {
    if (onDiagnostic != null && typeof onDiagnostic !== 'function') {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'Runtime onDiagnostic must be a function or null.', {
        phase: 'initialization',
      });
    }
    this.#onDiagnostic = onDiagnostic;
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
    this.#providers.set(name, { name, provider });
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
          let compiled: BackendProviderCompiledModel;
          try {
            candidate = await entry.provider.compile(compileInput, Object.freeze({
              operatorFallback: policy.operatorFallback,
            }));
            compiled = assertProviderCompiledModel(candidate, name, compileInput);
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
          } catch (error) {
            const close = (candidate as Partial<BackendProviderCompiledModel> | null)?.close;
            if (typeof close === 'function') {
              try { await close.call(candidate); } catch { /* Preserve the contract failure. */ }
            }
            throw error;
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
          const report = freezeCompilationReport(
            compilationId,
            policy,
            snapshot,
            name,
            candidates,
            elapsedMilliseconds(compilationStarted),
          );
          this.#emit({ kind: 'compilation', report });
          const result = new CompiledModel(
            this,
            snapshot,
            compiled,
            entry.provider.capabilities,
            report,
            this.#onDiagnostic,
            releaseRuntime,
          );
          transferred = true;
          return result;
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
      if (failed) {
        throw runtimeError(failed.reason, 'EXECUTION_FAILED', 'Runtime provider close failed.', {
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

export class CompiledModel {
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
  readonly #capabilities: BackendProviderCapabilities;
  readonly #onDiagnostic: ((event: RuntimeDiagnostic) => void) | null;
  readonly #releaseRuntime: () => void;
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
    this.#capabilities = capabilities;
    this.#onDiagnostic = onDiagnostic;
    this.#releaseRuntime = releaseRuntime;
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
    try {
      const decode = options.decode;
      const initialPlan = options.bankResidency === undefined && this.#snapshot.staticShapePlan
        ? this.#snapshot.staticShapePlan
        : resolveMinimumGraphShapes(
          this.#snapshot.graph,
          this.#snapshot.quantizationByTensor,
          options.bankResidency,
        );
      const providerOptions = Object.freeze({
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
        releaseContext,
      );
    } catch (error) {
      try {
        const close = backendContext?.close ||
          (candidate as Partial<BackendProviderExecutionContext> | null)?.close;
        if (typeof close === 'function') await close.call(backendContext || candidate);
      } catch {
        // Preserve the provider contract/construction failure.
      } finally {
        releaseContext();
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
      try {
        if (this.#retainedContexts > 0) {
          await new Promise<void>((resolve) => {
            this.#resolveContextDrain = resolve;
          });
        }
        await this.#compiled.close();
      } catch (error) {
        throw runtimeError(error, 'EXECUTION_FAILED',
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
  ): Promise<ExecutionResult> {
    const executionId = executionIdentity();
    const executionStarted = monotonicMilliseconds();
    let snapshot: BackendExecutionSnapshot | null = null;
    let snapshotTransferred = false;
    let reportedBackendReport: Readonly<Record<string, unknown>> | null = null;
    let resolvedOptions: Readonly<DecodeExecutionOptions> = EMPTY_EXECUTION_OPTIONS;
    let adapterRevisionIds: readonly (string | null)[] = EMPTY_ADAPTER_REVISION_IDS;
    let plan: ResolvedShapePlan | null = null;
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

      const normalizedInputRecord: Record<string, ExecutionInputs[string]> = {};
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
      const report: ExecutionReport = Object.freeze({
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
      const result = createExecutionResult(this.backend, snapshot, report, plan.outputs);
      snapshotTransferred = true;
      this.#emit({ kind: 'execution', report });
      return result;
    } catch (error) {
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
