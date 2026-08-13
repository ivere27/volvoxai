import { runtimeIdentity } from './Identity.js';
import { VolvoxAIError } from './RuntimeErrors.js';
import {
  MemoryBackingRelation,
  MemoryBoundKind,
  MemoryEnvelopeKind,
  MemoryEvidenceSource,
  MemoryInventoryKind,
  MemoryMetric,
  MemoryOwnerKind,
  MemoryResourceRole,
  MemorySnapshotPoint,
  MemorySpace,
  MemoryTemporalCoverage,
  MemoryValueRelation,
  OperationStage,
} from '../generated/volvoxaiEnums.js';

export const MEMORY_CAPTURE_PROTOCOL = 'volvoxai-memory-capture/v1' as const;
export const MEMORY_EVIDENCE_FORMAT = 'volvoxai-memory-evidence/v1' as const;
export const MEMORY_EVIDENCE_PROOF_PROTOCOL =
  'canonical-symbolic-domain-proof/v1' as const;
export const MEMORY_EVIDENCE_RESOURCE_PROTOCOL = 'bounded-resource-maxima/v1' as const;
export const BACKEND_MEMORY_SNAPSHOT_PROTOCOL =
  'volvoxai-backend-memory-snapshot/v1' as const;

const MAX_PERIODIC_SNAPSHOTS = 4096;
const MEMORY_CAPTURE_OPTION_NAMES = new Set([
  'protocol',
  'includeResourceInventory',
  'includeDomainAttestation',
  'requestedEnvelopes',
  'samplingIntervalNanoseconds',
  'maxPeriodicSnapshots',
]);
const KNOWN_ENVELOPES = new Set<number>([
  MemoryEnvelopeKind.ProcessRss,
  MemoryEnvelopeKind.ProcessPeakRss,
  MemoryEnvelopeKind.ProcessPss,
  MemoryEnvelopeKind.ProcessPrivateBytes,
  MemoryEnvelopeKind.ProcessManagedHeapUsed,
  MemoryEnvelopeKind.ProcessExternalBytes,
  MemoryEnvelopeKind.ProcessArrayBufferBytes,
  MemoryEnvelopeKind.DeviceProcessUsed,
  MemoryEnvelopeKind.DeviceTotalUsed,
  MemoryEnvelopeKind.DeviceTotalCapacity,
]);
const KNOWN_OWNER_KINDS = new Set<number>([
  MemoryOwnerKind.Process,
  MemoryOwnerKind.Runtime,
  MemoryOwnerKind.Model,
  MemoryOwnerKind.CompiledModel,
  MemoryOwnerKind.ExecutionContext,
  MemoryOwnerKind.Result,
  MemoryOwnerKind.BackendShared,
]);
const KNOWN_RESOURCE_ROLES = new Set<number>([
  MemoryResourceRole.Weights,
  MemoryResourceRole.PackedWeights,
  MemoryResourceRole.Input,
  MemoryResourceRole.Activation,
  MemoryResourceRole.KvCache,
  MemoryResourceRole.Scratch,
  MemoryResourceRole.ResultSnapshot,
  MemoryResourceRole.Arena,
  MemoryResourceRole.PlanCache,
  MemoryResourceRole.KernelCache,
  MemoryResourceRole.Metadata,
  MemoryResourceRole.RuntimeOverhead,
  MemoryResourceRole.AllocatorOverhead,
  MemoryResourceRole.DriverOverhead,
  MemoryResourceRole.Other,
]);
const KNOWN_MEMORY_SPACES = new Set<number>([
  MemorySpace.Host,
  MemorySpace.JsHeap,
  MemorySpace.JsExternal,
  MemorySpace.JsArrayBuffer,
  MemorySpace.WasmLinear,
  MemorySpace.NativeHeap,
  MemorySpace.MappedFile,
  MemorySpace.Device,
  MemorySpace.DeviceLocal,
  MemorySpace.HostVisibleDevice,
  MemorySpace.Unified,
]);
const KNOWN_MEMORY_METRICS = new Set<number>([
  MemoryMetric.Logical,
  MemoryMetric.Live,
  MemoryMetric.Reserved,
  MemoryMetric.Capacity,
  MemoryMetric.LogicalHighWater,
  MemoryMetric.LiveHighWater,
  MemoryMetric.ReservedHighWater,
  MemoryMetric.CapacityHighWater,
  MemoryMetric.CumulativeAllocated,
]);
const HIGH_WATER_METRICS = new Set<number>([
  MemoryMetric.LogicalHighWater,
  MemoryMetric.LiveHighWater,
  MemoryMetric.ReservedHighWater,
  MemoryMetric.CapacityHighWater,
]);
const KNOWN_VALUE_RELATIONS = new Set<number>([
  MemoryValueRelation.Exact,
  MemoryValueRelation.UpperBound,
  MemoryValueRelation.LowerBound,
  MemoryValueRelation.Requested,
  MemoryValueRelation.Estimated,
  MemoryValueRelation.Unavailable,
]);
const KNOWN_TEMPORAL_COVERAGE = new Set<number>([
  MemoryTemporalCoverage.Instant,
  MemoryTemporalCoverage.OperationWindow,
  MemoryTemporalCoverage.ResourceLifetime,
  MemoryTemporalCoverage.CaptureWindow,
  MemoryTemporalCoverage.ProcessLifetime,
  MemoryTemporalCoverage.SampledWindow,
]);

/**
 * Public capture policy corresponding to `MemoryCaptureOptions` in
 * `proto/volvoxai.proto`. Presence opts the runtime and all descendants in.
 */
export interface MemoryCaptureOptions {
  readonly protocol: typeof MEMORY_CAPTURE_PROTOCOL;
  readonly includeResourceInventory: boolean;
  readonly includeDomainAttestation: boolean;
  readonly requestedEnvelopes: readonly MemoryEnvelopeKind[];
  readonly samplingIntervalNanoseconds?: number;
  readonly maxPeriodicSnapshots?: number;
}

/**
 * JSON-safe protobuf projection. uint64 values stay safe-integer numbers in
 * public reports and are promoted to bigint only at an explicit wire boundary.
 */
export interface RuntimeMemoryByteSize {
  readonly bytes: number;
}

export interface RuntimeMemoryMonotonicTime {
  readonly nanoseconds: number;
}

export interface RuntimeMemoryOwnerRef {
  readonly kind: MemoryOwnerKind;
  readonly ownerId: string;
}

export interface RuntimeMemoryBoundTerm {
  readonly termId: string;
  readonly ownerKind: MemoryOwnerKind;
  readonly role: MemoryResourceRole;
  readonly space: MemorySpace;
  readonly upperBoundBytes: RuntimeMemoryByteSize;
}

export interface RuntimeMemoryPeakCase {
  readonly caseId: string;
  readonly terms: readonly RuntimeMemoryBoundTerm[];
  readonly totalBytes: RuntimeMemoryByteSize;
}

export interface RuntimeMemoryBoundProof {
  readonly budgetDomainId: string;
  readonly kind: MemoryBoundKind;
  readonly cases: readonly RuntimeMemoryPeakCase[];
  readonly maximumBytes: RuntimeMemoryByteSize;
  readonly limitBytes?: RuntimeMemoryByteSize;
}

export interface RuntimeMemoryDomainAttestation {
  readonly proofProtocol: typeof MEMORY_EVIDENCE_PROOF_PROTOCOL;
  readonly resourceProtocol: typeof MEMORY_EVIDENCE_RESOURCE_PROTOCOL;
  readonly graphFingerprint: string;
  readonly shapeDomainProofIdentity: string;
  readonly bounds: readonly RuntimeMemoryBoundProof[];
}

export interface RuntimeMemoryMeasurement {
  readonly metric: MemoryMetric;
  readonly bytes?: RuntimeMemoryByteSize;
  readonly source: MemoryEvidenceSource;
  readonly valueRelation: MemoryValueRelation;
  readonly temporalCoverage: MemoryTemporalCoverage;
}

export interface RuntimeMemoryResourceEvidence {
  readonly resourceId: string;
  readonly backingResourceId: string;
  readonly backingRelation: MemoryBackingRelation;
  readonly backingOffsetBytes?: RuntimeMemoryByteSize;
  readonly backingLengthBytes?: RuntimeMemoryByteSize;
  readonly owner: RuntimeMemoryOwnerRef;
  readonly role: MemoryResourceRole;
  readonly space: MemorySpace;
  readonly allocator: string;
  readonly consumers: readonly RuntimeMemoryOwnerRef[];
  readonly measurements: readonly RuntimeMemoryMeasurement[];
  readonly addressableBytes: RuntimeMemoryByteSize;
}

export interface RuntimeMemoryEnvelopeEvidence {
  readonly kind: MemoryEnvelopeKind;
  readonly bytes?: RuntimeMemoryByteSize;
  readonly source: MemoryEvidenceSource;
  readonly valueRelation: MemoryValueRelation;
  readonly temporalCoverage: MemoryTemporalCoverage;
  readonly sampler: string;
}

export interface RuntimeMemorySnapshot {
  readonly sequence: number;
  readonly monotonicTime?: RuntimeMemoryMonotonicTime;
  readonly stage: OperationStage;
  readonly point: MemorySnapshotPoint;
  readonly phaseOccurrenceId: string;
  readonly subject: RuntimeMemoryOwnerRef;
  readonly backend: string;
  readonly device: string;
  readonly domainAttestation?: RuntimeMemoryDomainAttestation;
  readonly resources: readonly RuntimeMemoryResourceEvidence[];
  readonly envelopes: readonly RuntimeMemoryEnvelopeEvidence[];
  readonly resourceInventory: MemoryInventoryKind;
}

export interface RuntimeMemoryEvidence {
  readonly format: typeof MEMORY_EVIDENCE_FORMAT;
  readonly captureId: string;
  readonly requiredFeatures: readonly string[];
  readonly snapshots: readonly RuntimeMemorySnapshot[];
}

/** Shape-domain facts already checked by the provider SPI. */
export interface RuntimeMemoryDomainAttestationInput {
  readonly proofProtocol: typeof MEMORY_EVIDENCE_PROOF_PROTOCOL;
  readonly resourceProtocol: typeof MEMORY_EVIDENCE_RESOURCE_PROTOCOL;
  readonly graphFingerprint: string;
  readonly maximumTensorBytes: number;
  readonly maximumResidentBytes: number;
  readonly resourceLimitBytes: number | null;
}

export interface RuntimeMemoryCaptureInput {
  readonly stage: OperationStage;
  readonly point: MemorySnapshotPoint;
  readonly subject: RuntimeMemoryOwnerRef;
  readonly backend?: string | null;
  readonly device?: Readonly<Record<string, string>> | string | null;
  readonly domainAttestation?: RuntimeMemoryDomainAttestationInput | null;
  readonly resources?: readonly RuntimeMemoryResourceEvidence[];
  readonly resourceInventory?: MemoryInventoryKind;
}

/** Optional, versioned provider-to-core resource collector boundary. */
export interface BackendMemoryCaptureRequest {
  readonly protocol: typeof BACKEND_MEMORY_SNAPSHOT_PROTOCOL;
  readonly point: MemorySnapshotPoint;
  readonly subject: RuntimeMemoryOwnerRef;
  /** Present after a successful result wrapper has taken output ownership. */
  readonly result?: RuntimeMemoryOwnerRef;
}

export interface BackendMemorySnapshot {
  readonly protocol: typeof BACKEND_MEMORY_SNAPSHOT_PROTOCOL;
  readonly resources: readonly RuntimeMemoryResourceEvidence[];
  readonly resourceInventory: MemoryInventoryKind;
}

export type RuntimeMemorySampler = (
  requested: readonly MemoryEnvelopeKind[],
) => readonly RuntimeMemoryEnvelopeEvidence[];

function invalidOptions(message: string): never {
  throw new VolvoxAIError('INVALID_ARGUMENT', message, { phase: 'initialization' });
}

function isPlainObject(value: unknown): value is Readonly<Record<string, unknown>> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function safeNonnegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function bytes(value: number): RuntimeMemoryByteSize {
  return Object.freeze({ bytes: value });
}

/** @internal Validate, detach, and freeze one runtime-scoped capture policy. */
export function normalizeMemoryCaptureOptions(
  value: MemoryCaptureOptions | undefined,
): Readonly<MemoryCaptureOptions> | null {
  if (value === undefined) return null;
  if (!isPlainObject(value)) {
    return invalidOptions('Runtime memoryCapture must be a plain object.');
  }
  const unknown = Reflect.ownKeys(value).find((key) =>
    typeof key !== 'string' || !MEMORY_CAPTURE_OPTION_NAMES.has(key));
  if (unknown !== undefined) {
    return invalidOptions(
      `Runtime memoryCapture does not accept option '${String(unknown)}'.`,
    );
  }
  if (value.protocol !== MEMORY_CAPTURE_PROTOCOL) {
    return invalidOptions(
      `Runtime memoryCapture.protocol must be exactly '${MEMORY_CAPTURE_PROTOCOL}'.`,
    );
  }
  if (typeof value.includeResourceInventory !== 'boolean' ||
      typeof value.includeDomainAttestation !== 'boolean') {
    return invalidOptions(
      'Runtime memoryCapture inventory and domain-attestation selectors must be boolean.',
    );
  }
  if (!Array.isArray(value.requestedEnvelopes)) {
    return invalidOptions('Runtime memoryCapture.requestedEnvelopes must be an array.');
  }
  const requestedEnvelopes: MemoryEnvelopeKind[] = [];
  const seen = new Set<number>();
  for (const [index, kind] of value.requestedEnvelopes.entries()) {
    if (!Number.isInteger(kind) || !KNOWN_ENVELOPES.has(kind)) {
      return invalidOptions(
        `Runtime memoryCapture.requestedEnvelopes[${index}] must be a known nonzero MemoryEnvelopeKind.`,
      );
    }
    if (seen.has(kind)) {
      return invalidOptions(
        `Runtime memoryCapture.requestedEnvelopes[${index}] is duplicated.`,
      );
    }
    seen.add(kind);
    requestedEnvelopes.push(kind);
  }
  if (!value.includeResourceInventory && !value.includeDomainAttestation &&
      requestedEnvelopes.length === 0) {
    return invalidOptions(
      'Runtime memoryCapture must select a resource inventory, domain attestation, or envelope.',
    );
  }

  const intervalPresent = Object.prototype.hasOwnProperty.call(
    value,
    'samplingIntervalNanoseconds',
  );
  const limitPresent = Object.prototype.hasOwnProperty.call(value, 'maxPeriodicSnapshots');
  if (intervalPresent !== limitPresent) {
    return invalidOptions(
      'Runtime memoryCapture periodic interval and snapshot limit must be present together.',
    );
  }
  if (intervalPresent) {
    if (!Number.isSafeInteger(value.samplingIntervalNanoseconds) ||
        (value.samplingIntervalNanoseconds as number) <= 0) {
      return invalidOptions(
        'Runtime memoryCapture.samplingIntervalNanoseconds must be a positive safe integer.',
      );
    }
    if (!Number.isInteger(value.maxPeriodicSnapshots) ||
        (value.maxPeriodicSnapshots as number) < 1 ||
        (value.maxPeriodicSnapshots as number) > MAX_PERIODIC_SNAPSHOTS) {
      return invalidOptions(
        `Runtime memoryCapture.maxPeriodicSnapshots must be in [1, ${MAX_PERIODIC_SNAPSHOTS}].`,
      );
    }
    return invalidOptions('Periodic runtime memory capture is not supported yet.');
  }

  return Object.freeze({
    protocol: MEMORY_CAPTURE_PROTOCOL,
    includeResourceInventory: value.includeResourceInventory,
    includeDomainAttestation: value.includeDomainAttestation,
    requestedEnvelopes: Object.freeze(requestedEnvelopes),
  });
}

interface RuntimeProcessMemoryUsage {
  readonly rss?: unknown;
  readonly heapUsed?: unknown;
  readonly external?: unknown;
  readonly arrayBuffers?: unknown;
}

interface RuntimeProcessLike {
  readonly memoryUsage?: () => RuntimeProcessMemoryUsage;
}

interface RuntimePerformanceMemory {
  readonly usedJSHeapSize?: unknown;
}

function processMemoryUsage(): RuntimeProcessMemoryUsage | null {
  const candidate = (globalThis as typeof globalThis & {
    process?: RuntimeProcessLike;
  }).process;
  if (typeof candidate?.memoryUsage !== 'function') return null;
  try {
    const sample = candidate.memoryUsage();
    return sample && typeof sample === 'object' ? sample : null;
  } catch {
    return null;
  }
}

function browserHeapUsed(): number | null {
  const memory = (globalThis.performance as Performance & {
    memory?: RuntimePerformanceMemory;
  } | undefined)?.memory;
  return safeNonnegativeInteger(memory?.usedJSHeapSize)
    ? memory.usedJSHeapSize
    : null;
}

function envelopeDescriptor(kind: MemoryEnvelopeKind): Readonly<{
  source: MemoryEvidenceSource;
  coverage: MemoryTemporalCoverage;
  sampler: string;
}> {
  switch (kind) {
    case MemoryEnvelopeKind.ProcessRss:
    case MemoryEnvelopeKind.ProcessPeakRss:
    case MemoryEnvelopeKind.ProcessPss:
    case MemoryEnvelopeKind.ProcessPrivateBytes:
      return Object.freeze({
        source: MemoryEvidenceSource.OsSampler,
        coverage: kind === MemoryEnvelopeKind.ProcessPeakRss
          ? MemoryTemporalCoverage.ProcessLifetime
          : MemoryTemporalCoverage.Instant,
        sampler: 'typescript-process-memory/v1',
      });
    case MemoryEnvelopeKind.ProcessManagedHeapUsed:
    case MemoryEnvelopeKind.ProcessExternalBytes:
    case MemoryEnvelopeKind.ProcessArrayBufferBytes:
      return Object.freeze({
        source: MemoryEvidenceSource.RuntimeCounter,
        coverage: MemoryTemporalCoverage.Instant,
        sampler: 'typescript-runtime-memory/v1',
      });
    case MemoryEnvelopeKind.DeviceProcessUsed:
    case MemoryEnvelopeKind.DeviceTotalUsed:
    case MemoryEnvelopeKind.DeviceTotalCapacity:
      return Object.freeze({
        source: MemoryEvidenceSource.DriverSampler,
        coverage: MemoryTemporalCoverage.Instant,
        sampler: 'typescript-driver-memory/v1',
      });
    default:
      throw new Error(`Unknown MemoryEnvelopeKind ${kind}.`);
  }
}

function unavailableEnvelope(kind: MemoryEnvelopeKind): RuntimeMemoryEnvelopeEvidence {
  const descriptor = envelopeDescriptor(kind);
  return Object.freeze({
    kind,
    source: descriptor.source,
    valueRelation: MemoryValueRelation.Unavailable,
    temporalCoverage: descriptor.coverage,
    sampler: descriptor.sampler,
  });
}

/**
 * Distinct collector identity for the browser heap fallback. It is not the
 * same method as the Node counter it stands in for, and `sampler` is the field
 * a consumer uses to tell two collectors apart.
 */
const BROWSER_HEAP_SAMPLER = 'browser-performance-memory/v1' as const;

/** @internal Best-effort process/runtime envelopes for TypeScript hosts. */
export const sampleRuntimeMemoryEnvelopes: RuntimeMemorySampler = (requested) => {
  const needsRuntimeSample = requested.some((kind) =>
    kind === MemoryEnvelopeKind.ProcessRss ||
    kind === MemoryEnvelopeKind.ProcessManagedHeapUsed ||
    kind === MemoryEnvelopeKind.ProcessExternalBytes ||
    kind === MemoryEnvelopeKind.ProcessArrayBufferBytes);
  const sample = needsRuntimeSample ? processMemoryUsage() : null;
  const browserHeap = sample === null &&
      requested.includes(MemoryEnvelopeKind.ProcessManagedHeapUsed)
    ? browserHeapUsed()
    : null;

  return Object.freeze(requested.map((kind) => {
    let measured: number | null = null;
    let relation = MemoryValueRelation.Exact;
    let sampler: string | null = null;
    if (kind === MemoryEnvelopeKind.ProcessRss && safeNonnegativeInteger(sample?.rss)) {
      measured = sample.rss;
    } else if (kind === MemoryEnvelopeKind.ProcessManagedHeapUsed) {
      if (safeNonnegativeInteger(sample?.heapUsed)) {
        measured = sample.heapUsed;
      } else if (browserHeap !== null) {
        // Browsers quantize performance.memory rather than reporting the live
        // heap, so this stands in for the exact counter without being one.
        measured = browserHeap;
        relation = MemoryValueRelation.Estimated;
        sampler = BROWSER_HEAP_SAMPLER;
      }
    } else if (kind === MemoryEnvelopeKind.ProcessExternalBytes &&
        safeNonnegativeInteger(sample?.external)) {
      measured = sample.external;
    } else if (kind === MemoryEnvelopeKind.ProcessArrayBufferBytes &&
        safeNonnegativeInteger(sample?.arrayBuffers)) {
      measured = sample.arrayBuffers;
    }
    if (measured === null) return unavailableEnvelope(kind);
    const descriptor = envelopeDescriptor(kind);
    return Object.freeze({
      kind,
      bytes: bytes(measured),
      source: descriptor.source,
      valueRelation: relation,
      temporalCoverage: descriptor.coverage,
      sampler: sampler ?? descriptor.sampler,
    });
  }));
};

function monotonicNanoseconds(): number | null {
  try {
    const now = typeof globalThis.performance?.now === 'function'
      ? globalThis.performance.now()
      : Date.now();
    const value = Math.floor(now * 1_000_000);
    return safeNonnegativeInteger(value) ? value : null;
  } catch {
    return null;
  }
}

function deviceIdentity(value: RuntimeMemoryCaptureInput['device']): string {
  if (typeof value === 'string') return value;
  if (!value) return '';
  return Object.entries(value)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, entry]) => `${key}=${entry}`)
    .join(',');
}

function coarseBound(
  budgetDomainId: string,
  kind: MemoryBoundKind,
  maximum: number,
  limit: number | null,
): RuntimeMemoryBoundProof {
  return Object.freeze({
    budgetDomainId,
    kind,
    cases: Object.freeze([]),
    maximumBytes: bytes(maximum),
    ...(limit === null ? {} : { limitBytes: bytes(limit) }),
  });
}

function projectDomainAttestation(
  value: RuntimeMemoryDomainAttestationInput,
): RuntimeMemoryDomainAttestation {
  return Object.freeze({
    proofProtocol: MEMORY_EVIDENCE_PROOF_PROTOCOL,
    resourceProtocol: MEMORY_EVIDENCE_RESOURCE_PROTOCOL,
    graphFingerprint: value.graphFingerprint,
    shapeDomainProofIdentity: `canonical-domain:${value.graphFingerprint}`,
    bounds: Object.freeze([
      coarseBound(
        'maximum-tensor',
        MemoryBoundKind.MaximumTensor,
        value.maximumTensorBytes,
        null,
      ),
      coarseBound(
        'provider-resident',
        MemoryBoundKind.OrdinaryResident,
        value.maximumResidentBytes,
        value.resourceLimitBytes,
      ),
    ]),
  });
}

function nonblank(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code !== 0x09 && code !== 0x0a && code !== 0x0b && code !== 0x0c &&
        code !== 0x0d && code !== 0x20) {
      return true;
    }
  }
  return false;
}

function exactOwnFields(
  value: Readonly<Record<string, unknown>>,
  allowed: ReadonlySet<string>,
  required: readonly string[],
): void {
  const keys = Reflect.ownKeys(value);
  if (keys.some((key) => typeof key !== 'string' || !allowed.has(key)) ||
      required.some((key) => !Object.prototype.hasOwnProperty.call(value, key))) {
    throw new Error('Invalid backend memory snapshot fields.');
  }
}

function cloneByteSize(value: unknown): RuntimeMemoryByteSize {
  if (!isPlainObject(value)) throw new Error('Invalid backend memory byte size.');
  exactOwnFields(value, new Set(['bytes']), ['bytes']);
  if (!safeNonnegativeInteger(value.bytes)) {
    throw new Error('Backend memory byte size must be a non-negative safe integer.');
  }
  return bytes(value.bytes);
}

function optionalByteSize(value: unknown): RuntimeMemoryByteSize | undefined {
  return value === undefined ? undefined : cloneByteSize(value);
}

function cloneOwner(value: unknown): RuntimeMemoryOwnerRef {
  if (!isPlainObject(value)) throw new Error('Invalid backend memory owner.');
  exactOwnFields(value, new Set(['kind', 'ownerId']), ['kind', 'ownerId']);
  if (!Number.isInteger(value.kind) || !KNOWN_OWNER_KINDS.has(value.kind as number) ||
      !nonblank(value.ownerId)) {
    throw new Error('Invalid backend memory owner identity.');
  }
  return Object.freeze({
    kind: value.kind as MemoryOwnerKind,
    ownerId: value.ownerId,
  });
}

function cloneMeasurement(value: unknown): RuntimeMemoryMeasurement {
  if (!isPlainObject(value)) throw new Error('Invalid backend memory measurement.');
  exactOwnFields(
    value,
    new Set(['metric', 'bytes', 'source', 'valueRelation', 'temporalCoverage']),
    ['metric', 'source', 'valueRelation', 'temporalCoverage'],
  );
  if (!Number.isInteger(value.metric) || !KNOWN_MEMORY_METRICS.has(value.metric as number) ||
      (value.source !== MemoryEvidenceSource.AllocatorCounter &&
        value.source !== MemoryEvidenceSource.RuntimeCounter &&
        value.source !== MemoryEvidenceSource.ApiRequest) ||
      !Number.isInteger(value.valueRelation) ||
      !KNOWN_VALUE_RELATIONS.has(value.valueRelation as number) ||
      !Number.isInteger(value.temporalCoverage) ||
      !KNOWN_TEMPORAL_COVERAGE.has(value.temporalCoverage as number)) {
    throw new Error('Invalid backend memory measurement axis.');
  }
  const measuredBytes = optionalByteSize(value.bytes);
  const unavailable = value.valueRelation === MemoryValueRelation.Unavailable;
  if (unavailable !== (measuredBytes === undefined)) {
    throw new Error('Backend memory measurement byte presence is invalid.');
  }
  const apiRequest = value.source === MemoryEvidenceSource.ApiRequest;
  const requested = value.valueRelation === MemoryValueRelation.Requested;
  if (apiRequest !== requested) {
    throw new Error('Backend memory API_REQUEST and REQUESTED must appear together.');
  }
  if ((HIGH_WATER_METRICS.has(value.metric as number) ||
      value.metric === MemoryMetric.CumulativeAllocated) &&
      value.temporalCoverage === MemoryTemporalCoverage.Instant) {
    throw new Error('Backend memory high-water evidence must name an epoch.');
  }
  if (value.temporalCoverage === MemoryTemporalCoverage.SampledWindow &&
      (!HIGH_WATER_METRICS.has(value.metric as number) ||
        (value.valueRelation !== MemoryValueRelation.LowerBound && !unavailable))) {
    throw new Error('Backend sampled-window evidence must be a lower-bound high-water value.');
  }
  return Object.freeze({
    metric: value.metric as MemoryMetric,
    ...(measuredBytes === undefined ? {} : { bytes: measuredBytes }),
    source: value.source as MemoryEvidenceSource,
    valueRelation: value.valueRelation as MemoryValueRelation,
    temporalCoverage: value.temporalCoverage as MemoryTemporalCoverage,
  });
}

function cloneResource(value: unknown): RuntimeMemoryResourceEvidence {
  if (!isPlainObject(value)) throw new Error('Invalid backend memory resource.');
  exactOwnFields(
    value,
    new Set([
      'resourceId',
      'backingResourceId',
      'backingRelation',
      'backingOffsetBytes',
      'backingLengthBytes',
      'owner',
      'role',
      'space',
      'allocator',
      'consumers',
      'measurements',
      'addressableBytes',
    ]),
    [
      'resourceId',
      'backingResourceId',
      'backingRelation',
      'owner',
      'role',
      'space',
      'allocator',
      'consumers',
      'measurements',
      'addressableBytes',
    ],
  );
  if (!nonblank(value.resourceId) || typeof value.backingResourceId !== 'string' ||
      (value.backingRelation !== MemoryBackingRelation.Independent &&
        value.backingRelation !== MemoryBackingRelation.Alias &&
        value.backingRelation !== MemoryBackingRelation.Suballocation) ||
      !Number.isInteger(value.role) || !KNOWN_RESOURCE_ROLES.has(value.role as number) ||
      !Number.isInteger(value.space) || !KNOWN_MEMORY_SPACES.has(value.space as number) ||
      !nonblank(value.allocator) || !Array.isArray(value.consumers) ||
      !Array.isArray(value.measurements) || value.measurements.length === 0) {
    throw new Error('Invalid backend memory resource descriptor.');
  }
  const backingOffsetBytes = optionalByteSize(value.backingOffsetBytes);
  const backingLengthBytes = optionalByteSize(value.backingLengthBytes);
  const addressableBytes = cloneByteSize(value.addressableBytes);
  if (value.backingRelation === MemoryBackingRelation.Independent) {
    if (value.backingResourceId !== '' || backingOffsetBytes !== undefined ||
        backingLengthBytes !== undefined) {
      throw new Error('Independent backend memory resource names a backing range.');
    }
  } else if (!nonblank(value.backingResourceId) || value.backingResourceId === value.resourceId ||
      backingOffsetBytes === undefined || backingLengthBytes === undefined ||
      backingLengthBytes.bytes !== addressableBytes.bytes ||
      !Number.isSafeInteger(backingOffsetBytes.bytes + backingLengthBytes.bytes)) {
    throw new Error('Backed backend memory resource has invalid range geometry.');
  }

  const consumers = Object.freeze(value.consumers.map(cloneOwner));
  const consumerKeys = new Set(consumers.map((owner) => `${owner.kind}\0${owner.ownerId}`));
  if (consumerKeys.size !== consumers.length) {
    throw new Error('Backend memory resource repeats a consumer.');
  }
  const measurements = Object.freeze(value.measurements.map(cloneMeasurement));
  const measurementKeys = new Set(measurements.map((measurement) => [
    measurement.metric,
    measurement.source,
    measurement.valueRelation,
    measurement.temporalCoverage,
  ].join('\0')));
  if (measurementKeys.size !== measurements.length) {
    throw new Error('Backend memory resource repeats a measurement axis.');
  }
  return Object.freeze({
    resourceId: value.resourceId,
    backingResourceId: value.backingResourceId,
    backingRelation: value.backingRelation as MemoryBackingRelation,
    ...(backingOffsetBytes === undefined ? {} : { backingOffsetBytes }),
    ...(backingLengthBytes === undefined ? {} : { backingLengthBytes }),
    owner: cloneOwner(value.owner),
    role: value.role as MemoryResourceRole,
    space: value.space as MemorySpace,
    allocator: value.allocator,
    consumers,
    measurements,
    addressableBytes,
  });
}

/** @internal Copy and fail closed on an optional provider resource snapshot. */
export function normalizeBackendMemorySnapshot(value: unknown): BackendMemorySnapshot | null {
  try {
    if (!isPlainObject(value)) return null;
    exactOwnFields(
      value,
      new Set(['protocol', 'resources', 'resourceInventory']),
      ['protocol', 'resources', 'resourceInventory'],
    );
    if (value.protocol !== BACKEND_MEMORY_SNAPSHOT_PROTOCOL ||
        !Array.isArray(value.resources) ||
        (value.resourceInventory !== MemoryInventoryKind.Partial &&
          value.resourceInventory !== MemoryInventoryKind.Complete)) {
      return null;
    }
    const resources = Object.freeze(value.resources.map(cloneResource));
    const byId = new Map(resources.map((resource) => [resource.resourceId, resource]));
    if (byId.size !== resources.length) return null;
    for (const resource of resources) {
      if (resource.backingRelation === MemoryBackingRelation.Independent) continue;
      const backing = byId.get(resource.backingResourceId);
      if (!backing || resource.backingOffsetBytes!.bytes + resource.backingLengthBytes!.bytes >
          backing.addressableBytes.bytes) {
        return null;
      }
      const visited = new Set([resource.resourceId]);
      let cursor: RuntimeMemoryResourceEvidence | undefined = backing;
      while (cursor && cursor.backingRelation !== MemoryBackingRelation.Independent) {
        if (visited.has(cursor.resourceId)) return null;
        visited.add(cursor.resourceId);
        cursor = byId.get(cursor.backingResourceId);
      }
      if (!cursor) return null;
    }
    if (value.resourceInventory === MemoryInventoryKind.Complete) {
      const suballocations = new Map<string, RuntimeMemoryResourceEvidence[]>();
      for (const resource of resources) {
        if (resource.backingRelation !== MemoryBackingRelation.Suballocation) continue;
        const siblings = suballocations.get(resource.backingResourceId) ?? [];
        siblings.push(resource);
        suballocations.set(resource.backingResourceId, siblings);
      }
      for (const siblings of suballocations.values()) {
        siblings.sort((left, right) =>
          left.backingOffsetBytes!.bytes - right.backingOffsetBytes!.bytes);
        let previousEnd = 0;
        let hasNonemptyRange = false;
        for (const resource of siblings) {
          const length = resource.backingLengthBytes!.bytes;
          const start = resource.backingOffsetBytes!.bytes;
          const end = start + length;
          if (length > 0 && hasNonemptyRange && start < previousEnd) return null;
          if (length > 0) {
            hasNonemptyRange = true;
            if (end > previousEnd) previousEnd = end;
          }
        }
      }
    }
    return Object.freeze({
      protocol: BACKEND_MEMORY_SNAPSHOT_PROTOCOL,
      resources,
      resourceInventory: value.resourceInventory as MemoryInventoryKind,
    });
  } catch {
    return null;
  }
}

function safeResources(
  input: RuntimeMemoryCaptureInput,
  includeResourceInventory: boolean,
): Readonly<{
  resources: readonly RuntimeMemoryResourceEvidence[];
  resourceInventory: MemoryInventoryKind;
}> {
  if (!includeResourceInventory || input.resources === undefined) {
    return Object.freeze({
      resources: Object.freeze([]),
      resourceInventory: MemoryInventoryKind.Partial,
    });
  }
  // Provider records have already crossed the versioned normalizing boundary.
  return Object.freeze({
    resources: Object.freeze([...input.resources]),
    resourceInventory: input.resourceInventory === MemoryInventoryKind.Complete
      ? MemoryInventoryKind.Complete
      : MemoryInventoryKind.Partial,
  });
}

/** @internal Runtime-scoped capture identity and best-effort sampler. */
export class RuntimeMemoryCaptureSession {
  readonly policy: Readonly<MemoryCaptureOptions>;
  readonly #sessionId = runtimeIdentity('memory-capture');
  readonly #sampler: RuntimeMemorySampler;
  #nextCapture = 1;

  constructor(
    policy: Readonly<MemoryCaptureOptions>,
    sampler: RuntimeMemorySampler = sampleRuntimeMemoryEnvelopes,
  ) {
    this.policy = policy;
    this.#sampler = sampler;
  }

  capture(input: RuntimeMemoryCaptureInput): RuntimeMemoryEvidence {
    if (!Number.isSafeInteger(this.#nextCapture)) {
      throw new Error('Runtime memory capture identity space is exhausted.');
    }
    const captureSequence = this.#nextCapture++;
    const captureId = `${this.#sessionId}-${captureSequence}`;
    let envelopes: readonly RuntimeMemoryEnvelopeEvidence[];
    try {
      envelopes = this.#sampler(this.policy.requestedEnvelopes);
    } catch {
      envelopes = Object.freeze(this.policy.requestedEnvelopes.map(unavailableEnvelope));
    }
    const observed = safeResources(input, this.policy.includeResourceInventory);
    const timestamp = monotonicNanoseconds();
    const snapshot: RuntimeMemorySnapshot = Object.freeze({
      sequence: 1,
      ...(timestamp === null ? {} : {
        monotonicTime: Object.freeze({ nanoseconds: timestamp }),
      }),
      stage: input.stage,
      point: input.point,
      phaseOccurrenceId: `${captureId}-phase-1`,
      subject: Object.freeze({ ...input.subject }),
      backend: input.backend ?? '',
      device: deviceIdentity(input.device),
      ...(this.policy.includeDomainAttestation && input.domainAttestation
        ? { domainAttestation: projectDomainAttestation(input.domainAttestation) }
        : {}),
      resources: observed.resources,
      envelopes,
      resourceInventory: observed.resourceInventory,
    });
    return Object.freeze({
      format: MEMORY_EVIDENCE_FORMAT,
      captureId,
      requiredFeatures: Object.freeze([]),
      snapshots: Object.freeze([snapshot]),
    });
  }
}
