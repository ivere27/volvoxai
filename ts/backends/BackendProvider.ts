import { assertBuiltInEngine } from './BackendEngine.js';
import { Model } from '../core/Model.js';
import type { BackendExecutionSnapshot } from '../core/ExecutionResult.js';
import type {
  BackendMemoryCaptureRequest,
  BackendMemorySnapshot,
} from '../core/MemoryCapture.js';
import type {
  Graph,
  InputDescriptor,
  TensorDescriptor,
} from '../core/Graph.js';
import type {
  AcceptedGraphShapeDomainProof,
  ResolvedShapePlan,
  ResolvedTensorDescriptor,
} from '../core/ResolvedShapePlan.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';
import type {
  AdapterSelector,
  BackendCapabilityContract,
  DecodeExecutionOptions,
  ExecutionInputs,
} from '../types.js';
import { memoryLocations } from '../generated/volvoxaiEnums.js';
import type {
  DecodeRowModeValue,
  MemoryLocationValue,
  OperatorFallbackValue,
} from '../generated/volvoxaiEnums.js';
import type { DeviceTensorInputLease } from '../ops/deviceTensorReference.js';
import {
  inspectIndependentPublicBatchSemantics,
} from '../ops/independentBatchSemantics.js';
import type { IndependentBatchSemanticsEvidence } from '../ops/independentBatchSemantics.js';
import type { ShapedRuntimeTensorView } from '../ops/shapeSystem.js';

/**
 * Opaque discriminator for the current exact JavaScript provider contract.
 * The numeric value is not a promise that historical layouts with the same
 * value remain compatible; providers must be built against these sources.
 */
export const VOLVOXAI_BACKEND_PROVIDER_VERSION = 1;
export const BACKEND_PROVIDER_INVARIANT_RESOURCE_OWNER_PROTOCOL =
  'compiled-invariant-resource-owner/v1' as const;
export const BACKEND_PROVIDER_INVARIANT_RESOURCE_LEASE_PROTOCOL =
  'compiled-invariant-resource-lease/v1' as const;

export type OperatorFallbackAttestation = 'none' | 'reported' | 'unknown';
export type DynamicShapeDomainSupport = 'full' | 'unsupported';

const MEMORY_LOCATIONS = new Set<unknown>(memoryLocations);
const DYNAMIC_SHAPE_CAPABILITY_MEMBERS = [
  'proofProtocol',
  'resourceProtocol',
  'support',
] as const;
const PROVIDER_CAPABILITY_MEMBERS = [
  'contextIsolation',
  'operatorFallback',
  'outputLocation',
  'dynamicShapeDomain',
] as const;

function hasExactOwnMembers(value: object, expected: readonly string[]): boolean {
  const members = Reflect.ownKeys(value);
  return members.length === expected.length && members.every((member) =>
    typeof member === 'string' && expected.includes(member));
}

export type BackendDeviceIdentity = Readonly<Record<string, string>>;

/**
 * Provider-wide declaration. `full` means compile() will either attest the
 * entire bounded domain represented by the canonical proof or reject it. It
 * never means "try one warm shape" or "discover support during dispatch".
 */
export interface BackendDynamicShapeDomainCapability {
  readonly proofProtocol: 'canonical-symbolic-domain-proof/v1';
  readonly resourceProtocol: 'bounded-resource-maxima/v1';
  readonly support: DynamicShapeDomainSupport;
}

export interface BackendShapeDomainCompilationAttestation {
  readonly proofProtocol: 'canonical-symbolic-domain-proof/v1';
  readonly resourceProtocol: 'bounded-resource-maxima/v1';
  readonly support: 'full';
  readonly graphFingerprint: string;
  readonly maximumTensorBytes: number;
  readonly maximumResidentBytes: number;
  readonly resourceLimitBytes: number | null;
  /** Exact immutable proof supplied in the compile input. */
  readonly proof: AcceptedGraphShapeDomainProof;
}

export interface BackendProviderCompilationEvidence {
  readonly device: BackendDeviceIdentity | null;
  readonly allocationBytes: number | null;
  readonly shapeDomain: BackendShapeDomainCompilationAttestation;
  /**
   * Core-owned proof bound to the exact canonical graph fingerprint. Required
   * whenever `batchContract.independentBatch` is `compiler-proved/v1`.
   */
  readonly batchSemantics?: Readonly<IndependentBatchSemanticsEvidence>;
  readonly operatorFallbackUsed?: boolean | null;
  readonly offendingNode?: string | number | null;
}

/**
 * Provider attestation for an explicitly declared public batch axis.
 * `single-invocation` forbids a provider from hiding B independent executes in
 * a loop and calling that dynamic batching.
 */
export interface BackendProviderBatchContract {
  readonly protocol: 'dense-public-batch/v1';
  readonly densePublicBatch: 'single-invocation' | 'unsupported';
  /** Provider compiler attestation; core still requires an independent operator proof. */
  readonly independentBatch: 'compiler-proved/v1' | 'unsupported';
  readonly deviceResident: boolean;
  readonly hostFallback: 'forbidden' | 'possible' | 'not-applicable';
}


export function createBackendProviderBatchContract(
  densePublicBatch: BackendProviderBatchContract['densePublicBatch'],
  options: Readonly<{
    independentBatch?: BackendProviderBatchContract['independentBatch'];
    deviceResident?: boolean;
    hostFallback?: BackendProviderBatchContract['hostFallback'];
  }> = {},
): Readonly<BackendProviderBatchContract> {
  if (densePublicBatch !== 'single-invocation' && densePublicBatch !== 'unsupported') {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Provider dense batch mode is invalid.', {
      phase: 'initialization',
    });
  }
  const independentBatch = options.independentBatch ?? 'unsupported';
  const deviceResident = options.deviceResident ?? false;
  const hostFallback = options.hostFallback ?? 'not-applicable';
  if ((independentBatch !== 'compiler-proved/v1' && independentBatch !== 'unsupported') ||
      typeof deviceResident !== 'boolean' ||
      !['forbidden', 'possible', 'not-applicable'].includes(hostFallback)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Provider dense batch attestation is invalid.', {
      phase: 'initialization',
    });
  }
  if ((deviceResident && hostFallback === 'not-applicable') ||
      (!deviceResident && hostFallback !== 'not-applicable')) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Batch host-fallback evidence must match device residency.', {
        phase: 'initialization',
      });
  }
  if (densePublicBatch === 'unsupported' && independentBatch !== 'unsupported') {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'An unsupported dense batch route cannot attest independent batch execution.', {
        phase: 'initialization',
      });
  }
  return Object.freeze({
    protocol: 'dense-public-batch/v1',
    densePublicBatch,
    independentBatch,
    deviceResident,
    hostFallback,
  });
}

/**
 * Provider-owned scheduling identities for one exact prepared batch route.
 * A compatibility token may be a structural string so providers do not need
 * an unbounded object interner; opaque objects retain identity semantics.
 * Resource domains and device generations always use object identity.
 */
export type BackendProviderCompatibilityToken = string | object;

export interface BackendProviderPreparedBatchRoute {
  readonly resourceDomain: object;
  readonly compatibilityToken: BackendProviderCompatibilityToken;
  readonly deviceEpoch: object;
}

function isFrozenOpaqueIdentity(value: unknown): value is object {
  return value !== null && typeof value === 'object' && Object.isFrozen(value);
}

function isCompatibilityToken(value: unknown): value is BackendProviderCompatibilityToken {
  return (typeof value === 'string' && value.length > 0) || isFrozenOpaqueIdentity(value);
}

/** Construct a frozen provider route without exposing how its identities are derived. */
export function createBackendProviderPreparedBatchRoute(
  resourceDomain: object,
  compatibilityToken: BackendProviderCompatibilityToken,
  deviceEpoch: object,
): Readonly<BackendProviderPreparedBatchRoute> {
  if (!isFrozenOpaqueIdentity(resourceDomain) ||
      !isCompatibilityToken(compatibilityToken) ||
      !isFrozenOpaqueIdentity(deviceEpoch)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Provider batch-route identities must be a non-empty structural token and frozen objects.', {
        phase: 'initialization',
      });
  }
  return Object.freeze({ resourceDomain, compatibilityToken, deviceEpoch });
}

/** @internal Validate the exact route returned by an untrusted provider. */
export function assertProviderPreparedBatchRoute(
  value: unknown,
  backendName: string,
): Readonly<BackendProviderPreparedBatchRoute> {
  const route = value as Partial<BackendProviderPreparedBatchRoute> | null;
  if (!route || typeof route !== 'object' || !Object.isFrozen(route) ||
      !hasExactOwnMembers(route, [
        'resourceDomain', 'compatibilityToken', 'deviceEpoch',
      ])) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid prepared batch route.`, {
        phase: 'execution', backend: backendName,
      });
  }
  // Capture provider accessors once, then publish only core-owned data fields.
  const resourceDomain = route.resourceDomain;
  const compatibilityToken = route.compatibilityToken;
  const deviceEpoch = route.deviceEpoch;
  if (!isFrozenOpaqueIdentity(resourceDomain) ||
      !isCompatibilityToken(compatibilityToken) ||
      !isFrozenOpaqueIdentity(deviceEpoch)) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid prepared batch route.`, {
        phase: 'execution', backend: backendName,
      });
  }
  return Object.freeze({ resourceDomain, compatibilityToken, deviceEpoch });
}

/** Copy provider-owned device metadata into a stable serializable identity. */
export function createBackendDeviceIdentity(value: unknown): BackendDeviceIdentity | null {
  if (value == null) return null;
  if (Array.isArray(value) || typeof value !== 'object') {
    throw new VolvoxAIError('ABI_UNSUPPORTED', 'Backend device identity must be a plain object or null.', {
      phase: 'initialization',
    });
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new VolvoxAIError('ABI_UNSUPPORTED', 'Backend device identity must be a plain object or null.', {
      phase: 'initialization',
    });
  }
  const identity: Record<string, string> = {};
  for (const [key, candidate] of Object.entries(value as Record<string, unknown>)) {
    if (typeof candidate === 'string' && candidate.trim()) identity[key] = candidate.trim();
  }
  return Object.keys(identity).length ? Object.freeze(identity) : null;
}

export interface BackendProviderCapabilityOptions {
  operatorFallback?: OperatorFallbackAttestation;
  outputLocation?: MemoryLocationValue;
  dynamicShapeDomain?: DynamicShapeDomainSupport;
}

export interface BackendProviderCapabilities {
  readonly contextIsolation: true;
  readonly operatorFallback: OperatorFallbackAttestation;
  readonly outputLocation: MemoryLocationValue;
  readonly dynamicShapeDomain: Readonly<BackendDynamicShapeDomainCapability>;
}

export function createBackendProviderCapabilities({
  operatorFallback = 'unknown',
  outputLocation = 'host',
  dynamicShapeDomain = 'unsupported',
}: BackendProviderCapabilityOptions = {}): Readonly<BackendProviderCapabilities> {
  if (!['none', 'reported', 'unknown'].includes(operatorFallback)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      `Invalid provider operatorFallback attestation '${operatorFallback}'.`, {
        phase: 'initialization',
      });
  }
  if (!MEMORY_LOCATIONS.has(outputLocation)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      `Invalid provider outputLocation '${outputLocation}'.`, {
        phase: 'initialization',
      });
  }
  if (dynamicShapeDomain !== 'full' && dynamicShapeDomain !== 'unsupported') {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      `Invalid provider dynamicShapeDomain attestation '${String(dynamicShapeDomain)}'.`, {
        phase: 'initialization',
      });
  }
  return Object.freeze({
    contextIsolation: true,
    operatorFallback,
    outputLocation,
    dynamicShapeDomain: Object.freeze({
      proofProtocol: 'canonical-symbolic-domain-proof/v1' as const,
      resourceProtocol: 'bounded-resource-maxima/v1' as const,
      support: dynamicShapeDomain,
    }),
  });
}

export interface BackendProviderCompileOptions {
  readonly operatorFallback: OperatorFallbackValue;
}

/** Immutable logical model view supplied to one provider compilation. */
export interface BackendLogicalCompileInput {
  readonly snapshot: Model;
  readonly graph: Graph;
  readonly graphFingerprint: string;
  /** Exact core-owned semantic proof providers may echo in compile evidence. */
  readonly batchSemantics: Readonly<IndependentBatchSemanticsEvidence>;
  readonly shapeDomainProof: AcceptedGraphShapeDomainProof;
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly inputNames: readonly string[];
  readonly inputDescriptors: readonly InputDescriptor[];
  readonly outputNames: readonly string[];
  readonly outputDescriptors: readonly TensorDescriptor[];
  readonly tensorCount: number;
  readonly nodeCount: number;
}

/** @internal Construct the exact immutable provider compile view. */
export function createBackendCompileInput(
  snapshot: Model,
): Readonly<BackendLogicalCompileInput> {
  if (!(snapshot instanceof Model)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      'Provider compilation requires a Model.', {
        phase: 'compilation',
      });
  }
  const batchSemantics = inspectIndependentPublicBatchSemantics(snapshot);
  return Object.freeze({
    snapshot,
    graph: snapshot.graph,
    graphFingerprint: snapshot.definitionFingerprint,
    batchSemantics,
    shapeDomainProof: snapshot.shapeDomainProof,
    definitionId: snapshot.definitionId,
    topologyRevision: snapshot.topologyRevision,
    weightRevision: snapshot.weightRevision,
    weightRevisionId: snapshot.weightRevisionId,
    inputNames: snapshot.inputNames,
    inputDescriptors: snapshot.inputDescriptors,
    outputNames: snapshot.outputNames,
    outputDescriptors: snapshot.outputDescriptors,
    tensorCount: snapshot.tensorCount,
    nodeCount: snapshot.nodeCount,
  });
}

export interface BackendResolvedAdapterSelection {
  readonly batchSize: number | null;
  readonly selectors: readonly (Readonly<AdapterSelector> | null)[];
}

/**
 * Complete concrete request delivered only after public binding and option
 * validation succeed. The plan is the single source of tensor geometry.
 */
export interface BackendResolvedExecutionRequest {
  readonly operation: 'execute' | 'seed' | 'step';
  /**
   * No-throw provider commit hook. A provider calls it exactly when pure
   * preflight/staging ends and retained decode state can no longer be
   * preserved. Core decode input publication still waits for success.
   * @internal
   */
  readonly commitExecution: () => void;
  readonly signature: string;
  readonly inputs: ExecutionInputs;
  /** Opaque result-ownership leases keyed by device-resident public input. */
  readonly deviceInputs: Readonly<Record<string, DeviceTensorInputLease>>;
  readonly inputDescriptors: readonly ResolvedTensorDescriptor[];
  readonly tensors: Readonly<Record<string, ResolvedTensorDescriptor>>;
  readonly outputDescriptors: readonly ResolvedTensorDescriptor[];
  readonly plan: ResolvedShapePlan;
  readonly adapters: Readonly<BackendResolvedAdapterSelection>;
  readonly options: Readonly<DecodeExecutionOptions>;
}

/** Narrow a resolved request for host-only providers and fail explicitly. */
export function requireHostExecutionInputs(
  request: BackendResolvedExecutionRequest,
  backend: string,
): Readonly<Record<string, ShapedRuntimeTensorView>> {
  if (Object.keys(request.deviceInputs).length !== 0) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `Backend '${backend}' does not accept device TensorResult inputs.`, {
        phase: 'execution', backend,
      });
  }
  for (const [name, view] of Object.entries(request.inputs)) {
    if (!ArrayBuffer.isView(view.data) || view.data instanceof DataView) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `Backend '${backend}' input '${name}' requires host typed-array storage.`, {
          phase: 'execution', backend,
        });
    }
  }
  return request.inputs as Readonly<Record<string, ShapedRuntimeTensorView>>;
}

export interface BackendProviderExecutionContext {
  readonly backendName: string;
  execute(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot>;
  decodeSeed?(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot>;
  decodeStep?(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot>;
  decodeReset?(): Promise<void>;
  /** Optional synchronous, versioned snapshot of provider-owned resources. */
  captureMemorySnapshot?(request: BackendMemoryCaptureRequest): BackendMemorySnapshot;
  close(): Promise<void> | void;
}

/** Counted compiled-artifact owner exposed by every provider compiled model. */
export interface BackendProviderInvariantResourceOwner {
  readonly protocol: typeof BACKEND_PROVIDER_INVARIANT_RESOURCE_OWNER_PROTOCOL;
  readonly ownerIdentity: object;
  /** Exact device/resource generation accepted by leases opened right now. */
  readonly deviceEpoch: object;
  /** Number of physically materialized invariant resources. */
  readonly resourceCount: number;
  /** Physical invariant bytes owned once by the compiled model. */
  readonly ownedBytes: number;
  /** Number of live context leases, including leases that borrowed no entries. */
  readonly borrowerCount: number;
  open(): BackendProviderInvariantResourceLease;
  close(): Promise<void> | void;
}

/**
 * Borrow-only context capability. Provider-specific leases may add typed
 * lookup methods, but creation/mutation authority must remain on the owner.
 */
export interface BackendProviderInvariantResourceLease {
  readonly protocol: typeof BACKEND_PROVIDER_INVARIANT_RESOURCE_LEASE_PROTOCOL;
  readonly ownerIdentity: object;
  readonly deviceEpoch: object;
  readonly borrowedResourceCount: number;
  readonly borrowedBytes: number;
  release(): void;
}

function validInvariantCount(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

export interface ValidatedProviderInvariantResources {
  readonly owner: BackendProviderInvariantResourceOwner;
  readonly ownerIdentity: object;
  readonly deviceEpoch: object;
  readonly open: BackendProviderInvariantResourceOwner['open'];
  readonly close: BackendProviderInvariantResourceOwner['close'];
}

const CAPTURED_COMPILED_INVARIANT_RESOURCES =
  new WeakMap<object, Readonly<ValidatedProviderInvariantResources>>();

function captureProviderInvariantResourceOwner(
  value: unknown,
  backendName: string,
): Readonly<ValidatedProviderInvariantResources> {
  const owner = value as Partial<BackendProviderInvariantResourceOwner> | null;
  const protocol = owner?.protocol;
  const ownerIdentity = owner?.ownerIdentity;
  const deviceEpoch = owner?.deviceEpoch;
  const resourceCount = owner?.resourceCount;
  const ownedBytes = owner?.ownedBytes;
  const borrowerCount = owner?.borrowerCount;
  const open = owner?.open;
  const close = owner?.close;
  if (!owner || typeof owner !== 'object' ||
      protocol !== BACKEND_PROVIDER_INVARIANT_RESOURCE_OWNER_PROTOCOL ||
      !isFrozenOpaqueIdentity(ownerIdentity) ||
      !isFrozenOpaqueIdentity(deviceEpoch) ||
      !validInvariantCount(resourceCount) ||
      !validInvariantCount(ownedBytes) ||
      !validInvariantCount(borrowerCount) ||
      typeof open !== 'function' || typeof close !== 'function') {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid invariant resource owner.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  return Object.freeze({
    owner: owner as BackendProviderInvariantResourceOwner,
    ownerIdentity,
    deviceEpoch,
    open,
    close,
  });
}

/** @internal Validate the mandatory compiled-model invariant owner seam. */
export function assertProviderInvariantResourceOwner(
  value: unknown,
  backendName: string,
): BackendProviderInvariantResourceOwner {
  return captureProviderInvariantResourceOwner(value, backendName).owner;
}

/** @internal Validate and bind one context lease to its exact compiled owner. */
export function assertProviderInvariantResourceLease(
  value: unknown,
  owner: BackendProviderInvariantResourceOwner,
  backendName: string,
  expected: Readonly<Pick<ValidatedProviderInvariantResources,
    'ownerIdentity' | 'deviceEpoch'>> = owner,
): BackendProviderInvariantResourceLease {
  const lease = value as Partial<BackendProviderInvariantResourceLease> | null;
  if (!lease || typeof lease !== 'object' || !Object.isFrozen(lease) ||
      lease.protocol !== BACKEND_PROVIDER_INVARIANT_RESOURCE_LEASE_PROTOCOL ||
      lease.ownerIdentity !== expected.ownerIdentity ||
      lease.deviceEpoch !== expected.deviceEpoch ||
      !validInvariantCount(lease.borrowedResourceCount) ||
      !validInvariantCount(lease.borrowedBytes) ||
      typeof lease.release !== 'function') {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid invariant resource lease.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  return lease as BackendProviderInvariantResourceLease;
}

/** @internal Retrieve the owner captured by compiled-model validation without rereading getters. */
export function validatedProviderInvariantResources(
  compiled: BackendProviderCompiledModel,
  backendName: string,
): Readonly<ValidatedProviderInvariantResources> {
  const resources = CAPTURED_COMPILED_INVARIANT_RESOURCES.get(compiled);
  if (resources === undefined) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' compiled model was not validated by core.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  return resources;
}

/** @internal Retrieve an owner captured before the rest of compiled validation. */
export function capturedProviderInvariantResources(
  value: unknown,
): Readonly<ValidatedProviderInvariantResources> | null {
  if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
    return null;
  }
  return CAPTURED_COMPILED_INVARIANT_RESOURCES.get(value) ?? null;
}

export interface BackendProviderContextOptions {
  /** Exact borrow-only lease opened by core from this compiled model. */
  readonly invariantResources: BackendProviderInvariantResourceLease;
  /**
   * Canonical metadata-only plan for eager physical preparation. Providers
   * may ignore it, but must never treat it as synthetic execution input or as
   * permission to narrow the compiled model's accepted shape domain.
   */
  readonly initialPlan?: ResolvedShapePlan;
  readonly decode?: Readonly<{
    changedInputs?: readonly string[] | null;
    rowMode?: DecodeRowModeValue;
    requireIncremental?: boolean;
    /** Dense decode slot capacity, declared by the context. Defaults to one. */
    lanes?: number;
  }>;
}

export interface BackendProviderCompiledModel {
  readonly backendName: string;
  readonly compilationEvidence: Readonly<BackendProviderCompilationEvidence>;
  readonly batchContract: Readonly<BackendProviderBatchContract>;
  readonly invariantResources: BackendProviderInvariantResourceOwner;
  /**
   * Pure, synchronous metadata attestation for this exact plan.
   *
   * This boundary runs before Runtime reserves/copies request payload storage.
   * It must not execute, mutate a context/device, retain request data, or grow
   * an unbounded per-request cache. Returned identities remain stable for every
   * admitted request that carries them.
   */
  prepareBatchRoute(plan: ResolvedShapePlan): Readonly<BackendProviderPreparedBatchRoute>;
  createContext(options: BackendProviderContextOptions):
    Promise<BackendProviderExecutionContext> | BackendProviderExecutionContext;
  close(): Promise<void> | void;
}

export interface BackendProvider {
  readonly providerVersion: typeof VOLVOXAI_BACKEND_PROVIDER_VERSION;
  readonly backendName: string;
  readonly deviceIdentity?: BackendDeviceIdentity | null;
  readonly capabilities: Readonly<BackendProviderCapabilities>;
  compile(
    input: BackendLogicalCompileInput,
    options: BackendProviderCompileOptions,
  ): Promise<BackendProviderCompiledModel> | BackendProviderCompiledModel;
  close(): Promise<void> | void;
}

export interface BackendProviderFactoryContext {
  readonly name: string;
  readonly runtime: unknown;
  readonly wasmUrl: string | URL;
}

export type BackendProviderFactory = (
  context: BackendProviderFactoryContext,
) => BackendProvider | null | Promise<BackendProvider | null>;

function validDomainCapability(value: unknown): value is BackendDynamicShapeDomainCapability {
  const capability = value as Partial<BackendDynamicShapeDomainCapability> | null;
  return !!capability && typeof capability === 'object' && Object.isFrozen(capability) &&
    hasExactOwnMembers(capability, DYNAMIC_SHAPE_CAPABILITY_MEMBERS) &&
    capability.proofProtocol === 'canonical-symbolic-domain-proof/v1' &&
    capability.resourceProtocol === 'bounded-resource-maxima/v1' &&
    (capability.support === 'full' || capability.support === 'unsupported');
}

export function assertBackendProvider(value: unknown, label = 'Backend provider'): BackendProvider {
  const provider = value as Partial<BackendProvider> | null;
  const capabilities = provider?.capabilities;
  const validCapabilities = capabilities && Object.isFrozen(capabilities) &&
    hasExactOwnMembers(capabilities, PROVIDER_CAPABILITY_MEMBERS) &&
    capabilities.contextIsolation === true &&
    ['none', 'reported', 'unknown'].includes(capabilities.operatorFallback) &&
    MEMORY_LOCATIONS.has(capabilities.outputLocation) &&
    validDomainCapability(capabilities.dynamicShapeDomain);
  if (!provider || provider.providerVersion !== VOLVOXAI_BACKEND_PROVIDER_VERSION ||
      typeof provider.backendName !== 'string' || provider.backendName.length === 0 ||
      !validCapabilities || typeof provider.compile !== 'function' ||
      typeof provider.close !== 'function') {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `${label} must match the current exact provider contract ` +
        `(VOLVOXAI_BACKEND_PROVIDER_VERSION ` +
        `${VOLVOXAI_BACKEND_PROVIDER_VERSION}); historical layouts are unsupported.`, {
        phase: 'initialization',
      });
  }
  if (provider.deviceIdentity !== undefined) createBackendDeviceIdentity(provider.deviceIdentity);
  return provider as BackendProvider;
}

function assertShapeDomainAttestation(
  value: unknown,
  input: BackendLogicalCompileInput,
  backendName: string,
): asserts value is BackendShapeDomainCompilationAttestation {
  const attestation = value as Partial<BackendShapeDomainCompilationAttestation> | null;
  if (!attestation || typeof attestation !== 'object' || !Object.isFrozen(attestation) ||
      attestation.proofProtocol !== 'canonical-symbolic-domain-proof/v1' ||
      attestation.resourceProtocol !== 'bounded-resource-maxima/v1' ||
      attestation.support !== 'full' ||
      attestation.graphFingerprint !== input.graphFingerprint ||
      attestation.proof !== input.shapeDomainProof ||
      !Number.isSafeInteger(attestation.maximumTensorBytes) ||
      (attestation.maximumTensorBytes as number) < 0 ||
      !Number.isSafeInteger(attestation.maximumResidentBytes) ||
      (attestation.maximumResidentBytes as number) < 0 ||
      (attestation.resourceLimitBytes !== null &&
        (!Number.isSafeInteger(attestation.resourceLimitBytes) ||
          (attestation.resourceLimitBytes as number) < 0)) ||
      (attestation.maximumTensorBytes as number) >
        (attestation.maximumResidentBytes as number) ||
      (attestation.resourceLimitBytes !== null &&
        (attestation.maximumResidentBytes as number) >
          (attestation.resourceLimitBytes as number))) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' did not attest the exact bounded shape domain.`, {
        phase: 'compilation', backend: backendName,
      });
  }
}

export function assertProviderCompiledModel(
  value: unknown,
  backendName: string,
  input: BackendLogicalCompileInput,
): BackendProviderCompiledModel {
  const compiled = value as Partial<BackendProviderCompiledModel> | null;
  if (!compiled || compiled.backendName !== backendName ||
      typeof compiled.prepareBatchRoute !== 'function' ||
      typeof compiled.createContext !== 'function' || typeof compiled.close !== 'function') {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid compiled model.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  const invariantResources = captureProviderInvariantResourceOwner(
    compiled.invariantResources,
    backendName,
  );
  /* Publish the captured callbacks before validating the remaining compiled
   * contract. Runtime.compile must be able to close this exact owner when a
   * later batch/evidence check rejects the candidate, without rereading a
   * hostile invariantResources getter. */
  CAPTURED_COMPILED_INVARIANT_RESOURCES.set(compiled, invariantResources);
  const batch = compiled.batchContract as Partial<BackendProviderBatchContract> | undefined;
  if (!batch || !Object.isFrozen(batch) || !hasExactOwnMembers(batch, [
    'protocol', 'densePublicBatch', 'independentBatch', 'deviceResident', 'hostFallback',
  ]) || batch.protocol !== 'dense-public-batch/v1' ||
      (batch.densePublicBatch !== 'single-invocation' && batch.densePublicBatch !== 'unsupported') ||
      (batch.independentBatch !== 'compiler-proved/v1' &&
        batch.independentBatch !== 'unsupported') ||
      (batch.densePublicBatch === 'unsupported' &&
        batch.independentBatch !== 'unsupported') ||
      typeof batch.deviceResident !== 'boolean' ||
      !['forbidden', 'possible', 'not-applicable'].includes(batch.hostFallback as string) ||
      ((batch.deviceResident && batch.hostFallback === 'not-applicable') ||
        (!batch.deviceResident && batch.hostFallback !== 'not-applicable'))) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid batch contract.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  const evidence = compiled.compilationEvidence;
  const batchEvidence = evidence?.batchSemantics;
  const expectedBatchEvidence = input.batchSemantics;
  if (!evidence || typeof evidence !== 'object' || !Object.isFrozen(evidence) ||
      (evidence.allocationBytes !== null &&
        (!Number.isSafeInteger(evidence.allocationBytes) || evidence.allocationBytes < 0)) ||
      (evidence.operatorFallbackUsed !== undefined &&
        evidence.operatorFallbackUsed !== null &&
        typeof evidence.operatorFallbackUsed !== 'boolean') ||
      (evidence.offendingNode !== undefined && evidence.offendingNode !== null &&
        typeof evidence.offendingNode !== 'string' &&
        typeof evidence.offendingNode !== 'number') ||
      (batch.independentBatch === 'compiler-proved/v1' && batchEvidence === undefined) ||
      (batchEvidence !== undefined && batchEvidence !== expectedBatchEvidence) ||
      (batch.independentBatch === 'compiler-proved/v1' &&
        expectedBatchEvidence.supported !== true)) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned invalid compilation evidence.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  createBackendDeviceIdentity(evidence.device);
  assertShapeDomainAttestation(evidence.shapeDomain, input, backendName);
  return compiled as BackendProviderCompiledModel;
}

export function assertProviderExecutionContext(
  value: unknown,
  backendName: string,
): BackendProviderExecutionContext {
  const context = value as Partial<BackendProviderExecutionContext> | null;
  if (!context || context.backendName !== backendName ||
      typeof context.execute !== 'function' || typeof context.close !== 'function' ||
      (context.captureMemorySnapshot !== undefined &&
        typeof context.captureMemorySnapshot !== 'function')) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid execution context.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  return context as BackendProviderExecutionContext;
}

type BuiltInEngine = {
  readonly backendName: string;
  readonly capabilities: BackendCapabilityContract;
  readonly adapterInfo?: Readonly<Record<string, string>> | null;
  readonly decodeCacheGeneration?: number;
  allocateGraph(...args: unknown[]): Promise<unknown> | unknown;
  execute(...args: unknown[]): Promise<unknown> | unknown;
  createDecodeSession(options?: unknown): unknown;
  fork?(): Promise<BuiltInEngine> | BuiltInEngine;
  dispose?(): void;
};

/** Provider wrapper for a built-in engine implementation. */
export class BuiltInBackendProvider implements BackendProvider {
  readonly providerVersion = VOLVOXAI_BACKEND_PROVIDER_VERSION;
  readonly backendName: string;
  readonly capabilities: Readonly<BackendProviderCapabilities>;
  readonly deviceIdentity: BackendDeviceIdentity | null;
  readonly #source: BuiltInEngine;
  #closed = false;

  constructor(source: BuiltInEngine) {
    this.#source = assertBuiltInEngine(source, 'Built-in backend provider source') as BuiltInEngine;
    this.backendName = this.#source.backendName;
    this.capabilities = createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: this.#source.capabilities.outputLocation,
      // Device providers state explicit non-support until their DS5+ bounded-
      // domain implementations. CPU lives in CPUBackendProvider so strict
      // WASM composition never imports the CPU engine or context.
      dynamicShapeDomain: 'unsupported',
    });
    this.deviceIdentity = createBackendDeviceIdentity(this.#source.adapterInfo) ||
      Object.freeze({
        backend: this.backendName,
        device: this.backendName === 'cpu-js' ? 'host' : this.backendName,
      });
  }

  async compile(
    input: BackendLogicalCompileInput,
    options: BackendProviderCompileOptions,
  ): Promise<BackendProviderCompiledModel> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', `Backend provider '${this.backendName}' is closed.`, {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (!(input?.snapshot instanceof Model) ||
        input.shapeDomainProof !== input.snapshot.shapeDomainProof ||
        input.graphFingerprint !== input.snapshot.definitionFingerprint) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'Backend provider compile requires the exact immutable logical compile view.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    void options;
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `Built-in backend '${this.backendName}' does not attest the full bounded shape domain.`, {
        phase: 'compilation', backend: this.backendName,
      });
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    this.#source.dispose?.();
  }
}
