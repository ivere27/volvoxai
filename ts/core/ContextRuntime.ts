import { Graph } from './Graph.js';
import { GraphLoader } from './GraphLoader.js';
import { ModelSnapshot, type ModelOutputDescriptor } from './ModelSnapshot.js';
import { runtimeIdentity } from './Identity.js';
import {
  ExecutionResult,
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
  createBackendDeviceIdentity,
  type BackendProvider,
  type BackendProviderCapabilities,
  type BackendProviderCompiledModel,
  type BackendProviderExecutionContext,
} from '../backends/BackendProvider.js';
import type { GraphLoaderOptions } from './GraphLoader.js';
import type {
  AdapterSelector,
  DecodeExecutionOptions,
  ExecutionInputs,
  ExecutionOptions,
} from '../types.js';
import type {
  BackendPolicyModeValue,
  DecodeRowModeValue,
  OperatorFallbackValue,
} from '../generated/volvoxaiEnums.js';

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
  readonly decode?: Readonly<{
    changedInputs?: readonly string[] | null;
    rowMode?: DecodeRowModeValue;
    requireIncremental?: boolean;
  }>;
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
  readonly adapterRevisionId: string | null;
  readonly adapterRevisionIds: readonly (string | null)[];
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
  snapshot: ModelSnapshot,
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
    adapterRevisionId: snapshot.adapterRevisionId,
    adapterRevisionIds: snapshot.adapterRevisionIds,
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

const CONTEXT_OPTION_NAMES = new Set(['adapter', 'decode']);
const DECODE_CONTEXT_OPTION_NAMES = new Set(['changedInputs', 'rowMode', 'requireIncremental']);
const EXECUTION_OPTION_NAMES = new Set(['adapter', 'adapters']);
const DECODE_EXECUTION_OPTION_NAMES = new Set([
  'adapter', 'adapters', 'changedInputs', 'position',
]);

function assertExecutionContextOptions(options: unknown): void {
  assertExactOptionObject(options, CONTEXT_OPTION_NAMES, 'Execution context options');
  const decode = (options as ExecutionContextOptions).decode;
  if (decode !== undefined) {
    assertExactOptionObject(decode, DECODE_CONTEXT_OPTION_NAMES,
      'Execution context decode options');
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

  createModel(graph: Graph): Model {
    this.#assertOpen();
    try {
      const snapshot = ModelSnapshot.capture(graph);
      return new Model(this, snapshot, this.#retainChild());
    } catch (error) {
      throw runtimeError(error, 'INVALID_ARGUMENT', 'Model graph is invalid.', {
        phase: 'initialization',
      });
    }
  }

  async loadModel(
    sources: string | readonly string[],
    options: GraphLoaderOptions = {},
  ): Promise<Model> {
    this.#assertOpen();
    const releaseRuntime = this.#retainChild();
    let transferred = false;
    try {
      const graph = new Graph();
      await GraphLoader.load(graph, sources, options);
      const snapshot = ModelSnapshot.capture(graph);
      const model = new Model(this, snapshot, releaseRuntime);
      transferred = true;
      return model;
    } catch (error) {
      throw runtimeError(error, 'INVALID_ARGUMENT', 'Model package could not be loaded.', {
        phase: 'initialization',
      });
    } finally {
      if (!transferred) releaseRuntime();
    }
  }

  async _compile(snapshot: ModelSnapshot, options: ModelCompileOptions = {}): Promise<CompiledModel> {
    this._assertAcceptingWork();
    const policy = normalizePolicy(options.backend, this.#order);
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

        try {
          let candidate: unknown = null;
          let compiled: BackendProviderCompiledModel;
          try {
            candidate = await entry.provider.compile(snapshot, {
              operatorFallback: policy.operatorFallback,
            });
            compiled = assertProviderCompiledModel(candidate, name);
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
            outcome: failure.code === 'BACKEND_UNAVAILABLE' ? 'unavailable' : 'failed',
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

  #assertOpen(): void {
    this._assertAcceptingWork();
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

export class Model {
  readonly #runtime: Runtime;
  readonly #releaseRuntime: () => void;
  #snapshot: ModelSnapshot;
  #state: 'open' | 'closing' | 'closed' = 'open';
  #retainedChildren = 0;
  #resolveChildDrain: (() => void) | null = null;
  #publicationTail: Promise<void> = Promise.resolve();
  #publicationHeld = false;
  #closePromise: Promise<void> | null = null;

  constructor(runtime: Runtime, snapshot: ModelSnapshot, releaseRuntime: () => void) {
    this.#runtime = runtime;
    this.#snapshot = snapshot;
    this.#releaseRuntime = releaseRuntime;
  }

  get definitionId(): string { return this.#snapshot.definitionId; }
  get topologyRevision(): number { return this.#snapshot.topologyRevision; }
  get weightRevision(): number { return this.#snapshot.weightRevision; }
  get weightRevisionId(): string { return this.#snapshot.weightRevisionId; }
  get adapterRevisionId(): string | null { return this.#snapshot.adapterRevisionId; }
  get adapterRevisionIds(): readonly string[] { return this.#snapshot.adapterRevisionIds; }

  async compile(options: ModelCompileOptions = {}): Promise<CompiledModel> {
    const release = this._retainChild();
    try {
      return await this.#runtime._compile(this.#snapshot, options);
    } finally {
      release();
    }
  }

  /** Publish a new private revision for future compilations; existing children stay pinned. */
  publishRevision(graph: Graph): ModelSnapshot {
    this.#assertOpen();
    if (this.#publicationHeld) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        'Model revision publication is already in progress.', {
          phase: 'execution',
        });
    }
    this.#snapshot = ModelSnapshot.derive(graph, this.#snapshot);
    return this.#snapshot;
  }

  /** @internal Serialize successor preparation and reject stale Trainers before mutation. */
  async _acquireRetainedRevision(expectedWeightRevisionId: string): Promise<() => void> {
    if (this.#retainedChildren === 0 || this.#state === 'closed') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Model has no retained child publisher.', {
        phase: 'lifecycle',
      });
    }
    let unlock!: () => void;
    const held = new Promise<void>((resolve) => { unlock = resolve; });
    const previous = this.#publicationTail;
    this.#publicationTail = previous.then(() => held, () => held);
    await previous.catch(() => undefined);
    if (this.#retainedChildren === 0) {
      unlock();
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Model has no retained child publisher.', {
        phase: 'lifecycle',
      });
    }
    if (this.#snapshot.weightRevisionId !== expectedWeightRevisionId) {
      unlock();
      throw new VolvoxAIError('EXECUTION_FAILED',
        'Model revision changed while a Trainer was preparing its successor.', {
          phase: 'execution',
        });
    }
    this.#publicationHeld = true;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.#publicationHeld = false;
      unlock();
    };
  }

  /** @internal Retained children may atomically publish while Model.close() drains them. */
  _publishRetainedRevision(
    graph: Graph,
    expectedWeightRevisionId: string = this.#snapshot.weightRevisionId,
  ): ModelSnapshot {
    if (this.#retainedChildren === 0 || this.#state === 'closed') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Model has no retained child publisher.', {
        phase: 'lifecycle',
      });
    }
    if (!this.#publicationHeld) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        'Model successor publication requires an acquired revision slot.', {
          phase: 'execution',
        });
    }
    if (this.#snapshot.weightRevisionId !== expectedWeightRevisionId) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        'Model revision changed while a Trainer was preparing its successor.', {
          phase: 'execution',
        });
    }
    this.#snapshot = ModelSnapshot.derive(graph, this.#snapshot);
    return this.#snapshot;
  }

  /** @internal Create private mutable storage for a retained Trainer. */
  _createRetainedWorkingGraph(): Graph {
    if (this.#retainedChildren === 0 || this.#state === 'closed') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Model has no retained child owner.', {
        phase: 'lifecycle',
      });
    }
    return this.#snapshot.createExecutionGraph();
  }

  /** @internal Reject a checkpoint whose graph cannot be a successor of this Model. */
  _assertRetainedDefinition(graph: Graph): void {
    if (this.#retainedChildren === 0 || this.#state === 'closed') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Model has no retained child owner.', {
        phase: 'lifecycle',
      });
    }
    if (!this.#snapshot.matchesDefinition(graph)) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Trainer checkpoint definition does not match the retained Model.', {
          phase: 'execution',
        });
    }
  }

  /** @internal Retain the model for a compiled operation or full-profile Trainer. */
  _retainChild(): () => void {
    this.#assertOpen();
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

  close(): Promise<void> {
    if (this.#closePromise) return this.#closePromise;
    this.#state = 'closing';
    this.#closePromise = (async () => {
      if (this.#retainedChildren > 0) {
        await new Promise<void>((resolve) => {
          this.#resolveChildDrain = resolve;
        });
      }
      this.#state = 'closed';
      this.#releaseRuntime();
    })();
    return this.#closePromise;
  }

  dispose(): Promise<void> { return this.close(); }

  #assertOpen(): void {
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Model is closing or closed.', {
        phase: 'lifecycle',
      });
    }
    this.#runtime._assertAcceptingWork();
  }
}

export class CompiledModel {
  readonly backend: string;
  readonly report: CompilationReport;
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly adapterRevisionId: string | null;
  readonly adapterRevisionIds: readonly string[];
  readonly #runtime: Runtime;
  readonly #snapshot: ModelSnapshot;
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
    snapshot: ModelSnapshot,
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
    this.adapterRevisionId = snapshot.adapterRevisionId;
    this.adapterRevisionIds = snapshot.adapterRevisionIds;
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
      this.#snapshot.resolveAdapterRevisionId(options.adapter);
    } catch (error) {
      throw runtimeError(error, 'INVALID_ARGUMENT', 'Execution context adapter selector is invalid.', {
        phase: 'compilation', backend: this.backend,
      });
    }
    const releaseContext = this.#retainContext();
    let candidate: unknown = null;
    let backendContext: BackendProviderExecutionContext | null = null;
    try {
      candidate = await this.#compiled.createContext(options as ExecutionOptions);
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
  readonly #snapshot: ModelSnapshot;
  readonly #backendContext: BackendProviderExecutionContext;
  readonly #capabilities: BackendProviderCapabilities;
  readonly #compilationReport: CompilationReport;
  readonly #outputDescriptors: readonly ModelOutputDescriptor[];
  readonly #onDiagnostic: ((event: RuntimeDiagnostic) => void) | null;
  readonly #releaseCompiled: () => void;
  #adapter: ExecutionContextOptions['adapter'];
  #adapterRevisionId: string | null;
  #tail: Promise<void> = Promise.resolve();
  #state: 'open' | 'closing' | 'closed' = 'open';
  #closePromise: Promise<void> | null = null;

  constructor(
    snapshot: ModelSnapshot,
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
    this.#outputDescriptors = Object.freeze(snapshot.outputDescriptors.map((output) => Object.freeze({
      ...output,
      shape: Object.freeze([...output.shape]),
    })));
    this.#onDiagnostic = onDiagnostic;
    this.#releaseCompiled = releaseCompiled;
    this.#adapter = cloneSelector(options.adapter);
    this.#adapterRevisionId = snapshot.resolveAdapterRevisionId(options.adapter);
    this.decode = new ContextDecode(this);
  }

  get closed(): boolean { return this.#state === 'closed'; }
  get adapterRevisionId(): string | null { return this.#adapterRevisionId; }

  execute(inputs: ExecutionInputs, options: ExecutionOptions = {}): Promise<ExecutionResult> {
    return this.#enqueue(() => this.#run('execute',
      (resolved) => this.#backendContext.execute(inputs, resolved),
      options,
    ));
  }

  selectAdapter(adapter: ExecutionContextOptions['adapter']): Promise<void> {
    return this.#enqueue(() => {
      let revisionId: string | null;
      try {
        revisionId = this.#snapshot.resolveAdapterRevisionId(adapter);
      } catch (error) {
        throw runtimeError(error, 'INVALID_ARGUMENT', 'Adapter selector is invalid.', {
          phase: 'execution', backend: this.backend,
        });
      }
      this.#adapter = cloneSelector(adapter);
      this.#adapterRevisionId = revisionId;
    });
  }

  _decodeExecute(
    operation: 'seed' | 'step',
    inputs: ExecutionInputs,
    options: DecodeExecutionOptions,
  ): Promise<ExecutionResult> {
    return this.#enqueue(() => this.#run(operation, async (resolved) => {
      const execute = operation === 'seed'
        ? this.#backendContext.decodeSeed
        : this.#backendContext.decodeStep;
      if (typeof execute !== 'function') {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `Backend '${this.backend}' does not expose decode ${operation}.`, {
            phase: 'execution', backend: this.backend,
          });
      }
      return execute.call(this.#backendContext, inputs, resolved);
    }, options));
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
    }).finally(this.#releaseCompiled);
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
    operation: (options: DecodeExecutionOptions) => Promise<BackendExecutionSnapshot>,
    options: DecodeExecutionOptions,
  ): Promise<ExecutionResult> {
    const executionId = executionIdentity();
    const executionStarted = monotonicMilliseconds();
    let snapshot: BackendExecutionSnapshot | null = null;
    let snapshotTransferred = false;
    let reportedBackendReport: Readonly<Record<string, unknown>> | null = null;
    assertExecutionOptions(options, executionOperation);
    const resolved: {
      -readonly [Key in keyof DecodeExecutionOptions]: DecodeExecutionOptions[Key]
    } = { ...options };
    if (!Object.prototype.hasOwnProperty.call(resolved, 'adapter') &&
        !Object.prototype.hasOwnProperty.call(resolved, 'adapters') &&
        this.#adapter !== undefined) {
      resolved.adapter = this.#adapter;
    }
    let adapterRevisionIds: readonly (string | null)[] = Object.freeze([
      this.#adapterRevisionId,
    ]);
    try {
      if (Object.prototype.hasOwnProperty.call(resolved, 'adapters')) {
        if (!Array.isArray(resolved.adapters)) {
          throw new VolvoxAIError('INVALID_ARGUMENT', 'Execution adapters must be an array.', {
            phase: 'execution', backend: this.backend,
          });
        }
        const selectors = Object.freeze(resolved.adapters.map((selector) => cloneSelector(selector)));
        resolved.adapters = selectors;
        adapterRevisionIds = Object.freeze(selectors.map((selector) =>
          this.#snapshot.resolveAdapterRevisionId(selector)));
      } else {
        const selector = Object.prototype.hasOwnProperty.call(resolved, 'adapter')
          ? cloneSelector(resolved.adapter)
          : undefined;
        if (Object.prototype.hasOwnProperty.call(resolved, 'adapter')) resolved.adapter = selector;
        adapterRevisionIds = Object.freeze([
          this.#snapshot.resolveAdapterRevisionId(selector),
        ]);
      }
      snapshot = await operation(resolved);
      const backendReport = normalizeBackendReport(snapshot.backendReport);
      reportedBackendReport = backendReport;
      const route = executionRouteEvidence(
        this.#capabilities,
        this.#compilationReport,
        backendReport,
      );
      const decodeState = executionDecodeState(executionOperation, resolved, backendReport);
      const report: ExecutionReport = Object.freeze({
        executionId,
        contextId: this.id,
        backend: this.backend,
        device: this.#compilationReport.selectedDevice,
        outcome: 'success',
        topologyRevision: this.topologyRevision,
        weightRevision: this.weightRevision,
        weightRevisionId: this.weightRevisionId,
        adapterRevisionId: adapterRevisionIds.length === 1 ? adapterRevisionIds[0] : null,
        adapterRevisionIds,
        executionTimeMs: elapsedMilliseconds(executionStarted),
        routeEvidence: route,
        decodeState,
        operatorFallback: this.#capabilities.operatorFallback,
        backendReport,
      });
      const result = new ExecutionResult(this.backend, snapshot, report, this.#outputDescriptors);
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
        adapterRevisionId: adapterRevisionIds.length === 1 ? adapterRevisionIds[0] : null,
        adapterRevisionIds,
        executionTimeMs: elapsedMilliseconds(executionStarted),
        routeEvidence: executionFailureRouteEvidence(
          this.#capabilities,
          this.#compilationReport,
          reportedBackendReport,
          failure.node,
        ),
        decodeState: executionDecodeState(executionOperation, resolved, null),
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
}
