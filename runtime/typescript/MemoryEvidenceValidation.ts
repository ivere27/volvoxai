import {
  MemoryBackingRelation,
  MemoryBoundProof,
  MemoryBoundKind,
  MemoryBoundTerm,
  MemoryByteSize,
  MemoryDomainAttestation,
  MemoryEnvelopeEvidence,
  MemoryEnvelopeKind,
  MemoryEvidence,
  MemoryEvidenceSource,
  MemoryInventoryKind,
  MemoryMeasurement,
  MemoryMetric,
  MemoryMonotonicTime,
  MemoryOwnerRef,
  MemoryOwnerKind,
  MemoryPeakCase,
  MemoryResourceEvidence,
  MemoryResourceRole,
  MemorySnapshot,
  MemorySnapshotPoint,
  MemorySpace,
  MemoryTemporalCoverage,
  MemoryValueRelation,
  OperationReport,
  OperationStage,
} from '../generated/typescript/volvoxai_lite.js';

export const MEMORY_EVIDENCE_FORMAT = 'volvoxai-memory-evidence/v1' as const;
export const MEMORY_EVIDENCE_PROOF_PROTOCOL =
  'canonical-symbolic-domain-proof/v1' as const;
export const MEMORY_EVIDENCE_RESOURCE_PROTOCOL =
  'bounded-resource-maxima/v1' as const;

const UINT64_MAX = 0xffff_ffff_ffff_ffffn;

export type MemoryEvidenceValidationCode =
  | 'INVALID_TYPE'
  | 'UNKNOWN_FIELD'
  | 'INVALID_FORMAT'
  | 'INVALID_FEATURE'
  | 'INVALID_UINT64'
  | 'INVALID_ENUM'
  | 'INVALID_IDENTITY'
  | 'INVALID_TIMELINE'
  | 'INVALID_PROOF'
  | 'INVALID_RESOURCE'
  | 'INVALID_MEASUREMENT'
  | 'INVALID_ENVELOPE'
  | 'INVALID_RESPONSE'
  | 'RESPONSE_TOO_LARGE';

/** Stable, path-bearing failure for decoded memory evidence. */
export class MemoryEvidenceValidationError extends Error {
  readonly code: MemoryEvidenceValidationCode;
  readonly path: string;

  constructor(code: MemoryEvidenceValidationCode, path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = 'MemoryEvidenceValidationError';
    this.code = code;
    this.path = path;
  }
}

type UnknownRecord = Record<string, unknown>;
type MessageConstructor = (abstract new (...args: never[]) => object) & {
  readonly name: string;
  readonly prototype: object;
};

interface ResourceDescriptor {
  readonly resourceId: string;
  readonly backingResourceId: string;
  readonly backingRelation: MemoryBackingRelation;
  readonly backingOffsetBytes: bigint | null;
  readonly backingLengthBytes: bigint | null;
  readonly addressableBytes: bigint;
  readonly owner: string;
  readonly role: MemoryResourceRole;
  readonly space: MemorySpace;
  readonly allocator: string;
}

interface OccurrenceDescriptor {
  readonly stage: OperationStage;
  readonly subject: string;
  readonly backend: string;
  readonly device: string;
  readonly singletonPoints: Set<MemorySnapshotPoint>;
}

function fail(
  code: MemoryEvidenceValidationCode,
  path: string,
  message: string,
): never {
  throw new MemoryEvidenceValidationError(code, path, message);
}

function record(
  value: unknown,
  path: string,
  expected: MessageConstructor,
): UnknownRecord {
  if (!(value instanceof expected) || Object.getPrototypeOf(value) !== expected.prototype) {
    fail('INVALID_TYPE', path, `must be a decoded ${expected.name} message`);
  }
  return value as unknown as UnknownRecord;
}

function exactFields(value: UnknownRecord, path: string, allowed: readonly string[]): void {
  const allowedSet = new Set(allowed);
  for (const key of allowed) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !('value' in descriptor)) {
      fail('INVALID_TYPE', `${path}.${key}`, 'must be an own data field');
    }
  }
  for (const key of Reflect.ownKeys(value)) {
    // Synurang tracks proto3 wrapper presence on decoded/generated messages
    // with one non-enumerable symbol. It is codec state, not a schema field.
    if (typeof key === 'symbol' && key.description === 'synurang.present') continue;
    if (typeof key !== 'string' || !allowedSet.has(key)) {
      fail('UNKNOWN_FIELD', path, `contains unknown field '${String(key)}'`);
    }
  }
}

function array(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) fail('INVALID_TYPE', path, 'must be an array');
  return value;
}

function nonblank(value: string): boolean {
  return /[^\u0009-\u000d\u0020]/.test(value);
}

function text(value: unknown, path: string, code: MemoryEvidenceValidationCode): string {
  if (typeof value !== 'string' || !nonblank(value)) {
    fail(code, path, 'must be a non-empty string');
  }
  return value;
}

function uint64(value: unknown, path: string): bigint {
  if (typeof value !== 'bigint' || value < 0n || value > UINT64_MAX) {
    fail('INVALID_UINT64', path, 'must be a uint64 bigint');
  }
  return value;
}

function byteSize(value: unknown, path: string): bigint {
  const message = record(value, path, MemoryByteSize);
  exactFields(message, path, ['bytes']);
  return uint64(message.bytes, `${path}.bytes`);
}

function optionalByteSize(value: unknown, path: string): bigint | null {
  return value === undefined ? null : byteSize(value, path);
}

function enumMembers(value: object): ReadonlySet<number> {
  return new Set(Object.values(value).filter((entry): entry is number =>
    typeof entry === 'number' && entry !== 0));
}

const MEMORY_BACKING_RELATIONS = enumMembers(MemoryBackingRelation);
const MEMORY_BOUND_KINDS = enumMembers(MemoryBoundKind);
const MEMORY_ENVELOPE_KINDS = enumMembers(MemoryEnvelopeKind);
const MEMORY_EVIDENCE_SOURCES = enumMembers(MemoryEvidenceSource);
const MEMORY_INVENTORY_KINDS = enumMembers(MemoryInventoryKind);
const MEMORY_METRICS = enumMembers(MemoryMetric);
const MEMORY_OWNER_KINDS = enumMembers(MemoryOwnerKind);
const MEMORY_RESOURCE_ROLES = enumMembers(MemoryResourceRole);
const MEMORY_SNAPSHOT_POINTS = enumMembers(MemorySnapshotPoint);
const MEMORY_SPACES = enumMembers(MemorySpace);
const MEMORY_TEMPORAL_COVERAGES = enumMembers(MemoryTemporalCoverage);
const MEMORY_VALUE_RELATIONS = enumMembers(MemoryValueRelation);
const OPERATION_STAGES = enumMembers(OperationStage);

function knownEnum<T extends number>(
  value: unknown,
  members: ReadonlySet<number>,
  path: string,
): T {
  if (typeof value !== 'number' || !Number.isInteger(value) || !members.has(value)) {
    fail('INVALID_ENUM', path, 'must be a known non-UNSPECIFIED enum value');
  }
  return value as T;
}

function owner(value: unknown, path: string): string {
  const message = record(value, path, MemoryOwnerRef);
  exactFields(message, path, ['kind', 'ownerId']);
  const kind = knownEnum<MemoryOwnerKind>(message.kind, MEMORY_OWNER_KINDS, `${path}.kind`);
  const ownerId = text(message.ownerId, `${path}.ownerId`, 'INVALID_IDENTITY');
  return `${kind}\u0000${ownerId}`;
}

function checkedAdd(left: bigint, right: bigint, path: string): bigint {
  if (left > UINT64_MAX - right) {
    fail('INVALID_UINT64', path, 'overflows uint64');
  }
  return left + right;
}

function validateDomainAttestation(value: unknown, path: string): void {
  const attestation = record(value, path, MemoryDomainAttestation);
  exactFields(attestation, path, [
    'proofProtocol',
    'resourceProtocol',
    'graphFingerprint',
    'shapeDomainProofIdentity',
    'bounds',
  ]);
  if (attestation.proofProtocol !== MEMORY_EVIDENCE_PROOF_PROTOCOL) {
    fail('INVALID_PROOF', `${path}.proofProtocol`, 'uses an unsupported proof protocol');
  }
  if (attestation.resourceProtocol !== MEMORY_EVIDENCE_RESOURCE_PROTOCOL) {
    fail('INVALID_PROOF', `${path}.resourceProtocol`, 'uses an unsupported resource protocol');
  }
  text(attestation.graphFingerprint, `${path}.graphFingerprint`, 'INVALID_PROOF');
  text(
    attestation.shapeDomainProofIdentity,
    `${path}.shapeDomainProofIdentity`,
    'INVALID_PROOF',
  );
  const bounds = array(attestation.bounds, `${path}.bounds`);
  if (bounds.length === 0) fail('INVALID_PROOF', `${path}.bounds`, 'must not be empty');
  const boundIds = new Set<string>();
  for (const [boundIndex, candidate] of bounds.entries()) {
    const boundPath = `${path}.bounds[${boundIndex}]`;
    const bound = record(candidate, boundPath, MemoryBoundProof);
    exactFields(bound, boundPath, [
      'budgetDomainId', 'kind', 'cases', 'maximumBytes', 'limitBytes',
    ]);
    const budgetDomainId = text(
      bound.budgetDomainId,
      `${boundPath}.budgetDomainId`,
      'INVALID_PROOF',
    );
    const kind = knownEnum<MemoryBoundKind>(
      bound.kind,
      MEMORY_BOUND_KINDS,
      `${boundPath}.kind`,
    );
    const boundId = `${budgetDomainId}\u0000${kind}`;
    if (boundIds.has(boundId)) {
      fail('INVALID_PROOF', boundPath, 'duplicates a budget-domain/kind bound');
    }
    boundIds.add(boundId);

    const maximum = optionalByteSize(bound.maximumBytes, `${boundPath}.maximumBytes`);
    if (maximum === null) {
      fail('INVALID_PROOF', `${boundPath}.maximumBytes`, 'is required, including exact zero');
    }
    const limit = optionalByteSize(bound.limitBytes, `${boundPath}.limitBytes`);
    if (limit !== null && maximum > limit) {
      fail('INVALID_PROOF', `${boundPath}.maximumBytes`, 'exceeds limitBytes');
    }

    const cases = array(bound.cases, `${boundPath}.cases`);
    const caseIds = new Set<string>();
    let casesMaximum = 0n;
    for (const [caseIndex, caseCandidate] of cases.entries()) {
      const casePath = `${boundPath}.cases[${caseIndex}]`;
      const peakCase = record(caseCandidate, casePath, MemoryPeakCase);
      exactFields(peakCase, casePath, ['caseId', 'terms', 'totalBytes']);
      const caseId = text(peakCase.caseId, `${casePath}.caseId`, 'INVALID_PROOF');
      if (caseIds.has(caseId)) fail('INVALID_PROOF', `${casePath}.caseId`, 'is duplicated');
      caseIds.add(caseId);
      const terms = array(peakCase.terms, `${casePath}.terms`);
      if (terms.length === 0) {
        fail('INVALID_PROOF', `${casePath}.terms`, 'must contain a term; omit cases for a coarse bound');
      }
      const termIds = new Set<string>();
      let sum = 0n;
      for (const [termIndex, termCandidate] of terms.entries()) {
        const termPath = `${casePath}.terms[${termIndex}]`;
        const term = record(termCandidate, termPath, MemoryBoundTerm);
        exactFields(term, termPath, [
          'termId', 'ownerKind', 'role', 'space', 'upperBoundBytes',
        ]);
        const termId = text(term.termId, `${termPath}.termId`, 'INVALID_PROOF');
        if (termIds.has(termId)) fail('INVALID_PROOF', `${termPath}.termId`, 'is duplicated');
        termIds.add(termId);
        knownEnum<MemoryOwnerKind>(
          term.ownerKind,
          MEMORY_OWNER_KINDS,
          `${termPath}.ownerKind`,
        );
        knownEnum<MemoryResourceRole>(
          term.role,
          MEMORY_RESOURCE_ROLES,
          `${termPath}.role`,
        );
        knownEnum<MemorySpace>(term.space, MEMORY_SPACES, `${termPath}.space`);
        const bytes = optionalByteSize(term.upperBoundBytes, `${termPath}.upperBoundBytes`);
        if (bytes === null) {
          fail('INVALID_PROOF', `${termPath}.upperBoundBytes`, 'is required, including exact zero');
        }
        sum = checkedAdd(sum, bytes, `${casePath}.terms`);
      }
      const total = optionalByteSize(peakCase.totalBytes, `${casePath}.totalBytes`);
      if (total === null) {
        fail('INVALID_PROOF', `${casePath}.totalBytes`, 'is required, including exact zero');
      }
      if (total !== sum) {
        fail('INVALID_PROOF', `${casePath}.totalBytes`, `must equal checked term sum ${sum}`);
      }
      if (caseIndex === 0 || total > casesMaximum) casesMaximum = total;
    }
    if (cases.length > 0 && maximum !== casesMaximum) {
      fail('INVALID_PROOF', `${boundPath}.maximumBytes`, `must equal case maximum ${casesMaximum}`);
    }
  }
}

function relationBytes(
  bytesValue: unknown,
  relation: MemoryValueRelation,
  path: string,
  code: 'INVALID_MEASUREMENT' | 'INVALID_ENVELOPE',
): bigint | null {
  const bytes = optionalByteSize(bytesValue, `${path}.bytes`);
  const unavailable = relation === MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE;
  if (unavailable !== (bytes === null)) {
    fail(
      code,
      `${path}.bytes`,
      unavailable
        ? 'must be absent when valueRelation is UNAVAILABLE'
        : 'is required unless valueRelation is UNAVAILABLE',
    );
  }
  return bytes;
}

const HIGH_WATER_METRICS = new Set<MemoryMetric>([
  MemoryMetric.MEMORY_METRIC_LOGICAL_HIGH_WATER,
  MemoryMetric.MEMORY_METRIC_LIVE_HIGH_WATER,
  MemoryMetric.MEMORY_METRIC_RESERVED_HIGH_WATER,
  MemoryMetric.MEMORY_METRIC_CAPACITY_HIGH_WATER,
]);

function validateMeasurement(value: unknown, path: string): string {
  const measurement = record(value, path, MemoryMeasurement);
  exactFields(measurement, path, [
    'metric', 'bytes', 'source', 'valueRelation', 'temporalCoverage',
  ]);
  const metric = knownEnum<MemoryMetric>(measurement.metric, MEMORY_METRICS, `${path}.metric`);
  const source = knownEnum<MemoryEvidenceSource>(
    measurement.source,
    MEMORY_EVIDENCE_SOURCES,
    `${path}.source`,
  );
  if (source !== MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_ALLOCATOR_COUNTER &&
      source !== MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_RUNTIME_COUNTER &&
      source !== MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_API_REQUEST) {
    fail('INVALID_MEASUREMENT', `${path}.source`, 'OS and driver samples belong in envelopes');
  }
  const relation = knownEnum<MemoryValueRelation>(
    measurement.valueRelation,
    MEMORY_VALUE_RELATIONS,
    `${path}.valueRelation`,
  );
  const temporal = knownEnum<MemoryTemporalCoverage>(
    measurement.temporalCoverage,
    MEMORY_TEMPORAL_COVERAGES,
    `${path}.temporalCoverage`,
  );
  relationBytes(measurement.bytes, relation, path, 'INVALID_MEASUREMENT');

  const apiRequest = source === MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_API_REQUEST;
  const requested = relation === MemoryValueRelation.MEMORY_VALUE_RELATION_REQUESTED;
  if (apiRequest !== requested) {
    fail(
      'INVALID_MEASUREMENT',
      `${path}.valueRelation`,
      'API_REQUEST and REQUESTED must appear together',
    );
  }
  if ((HIGH_WATER_METRICS.has(metric) ||
      metric === MemoryMetric.MEMORY_METRIC_CUMULATIVE_ALLOCATED) &&
      temporal === MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_INSTANT) {
    fail('INVALID_MEASUREMENT', `${path}.temporalCoverage`, 'must name a non-INSTANT epoch');
  }
  if (temporal === MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_SAMPLED_WINDOW) {
    if (!HIGH_WATER_METRICS.has(metric)) {
      fail('INVALID_MEASUREMENT', `${path}.metric`, 'must be a high-water metric for SAMPLED_WINDOW');
    }
    if (relation !== MemoryValueRelation.MEMORY_VALUE_RELATION_LOWER_BOUND &&
        relation !== MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE) {
      fail('INVALID_MEASUREMENT', `${path}.valueRelation`, 'must be LOWER_BOUND for SAMPLED_WINDOW');
    }
  }
  return `${metric}\u0000${source}\u0000${relation}\u0000${temporal}`;
}

function validateResource(value: unknown, path: string): ResourceDescriptor {
  const resource = record(value, path, MemoryResourceEvidence);
  exactFields(resource, path, [
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
  ]);
  const resourceId = text(resource.resourceId, `${path}.resourceId`, 'INVALID_IDENTITY');
  const backingRelation = knownEnum<MemoryBackingRelation>(
    resource.backingRelation,
    MEMORY_BACKING_RELATIONS,
    `${path}.backingRelation`,
  );
  const backingResourceId = typeof resource.backingResourceId === 'string'
    ? resource.backingResourceId
    : fail('INVALID_TYPE', `${path}.backingResourceId`, 'must be a string');
  const backingOffsetBytes = optionalByteSize(
    resource.backingOffsetBytes,
    `${path}.backingOffsetBytes`,
  );
  const backingLengthBytes = optionalByteSize(
    resource.backingLengthBytes,
    `${path}.backingLengthBytes`,
  );
  const addressableBytes = optionalByteSize(resource.addressableBytes, `${path}.addressableBytes`);
  if (addressableBytes === null) {
    fail('INVALID_RESOURCE', `${path}.addressableBytes`, 'is required, including exact zero');
  }
  if (backingRelation === MemoryBackingRelation.MEMORY_BACKING_RELATION_INDEPENDENT) {
    if (backingResourceId !== '' || backingOffsetBytes !== null || backingLengthBytes !== null) {
      fail('INVALID_RESOURCE', path, 'an INDEPENDENT resource must not name a backing range');
    }
  } else {
    if (!nonblank(backingResourceId) ||
        backingOffsetBytes === null || backingLengthBytes === null) {
      fail('INVALID_RESOURCE', path, 'a backed resource requires its backing ID, offset, and length');
    }
    if (backingResourceId === resourceId) {
      fail('INVALID_RESOURCE', `${path}.backingResourceId`, 'must not refer to itself');
    }
    if (addressableBytes !== backingLengthBytes) {
      fail('INVALID_RESOURCE', `${path}.addressableBytes`, 'must equal backingLengthBytes');
    }
    checkedAdd(backingOffsetBytes, backingLengthBytes, `${path}.backingLengthBytes`);
  }
  const ownerKey = owner(resource.owner, `${path}.owner`);
  const role = knownEnum<MemoryResourceRole>(
    resource.role,
    MEMORY_RESOURCE_ROLES,
    `${path}.role`,
  );
  const space = knownEnum<MemorySpace>(resource.space, MEMORY_SPACES, `${path}.space`);
  const allocator = text(resource.allocator, `${path}.allocator`, 'INVALID_RESOURCE');

  const consumerKeys = new Set<string>();
  for (const [index, candidate] of array(resource.consumers, `${path}.consumers`).entries()) {
    const key = owner(candidate, `${path}.consumers[${index}]`);
    if (consumerKeys.has(key)) {
      fail('INVALID_RESOURCE', `${path}.consumers[${index}]`, 'duplicates a consumer');
    }
    consumerKeys.add(key);
  }
  const measurements = array(resource.measurements, `${path}.measurements`);
  if (measurements.length === 0) {
    fail('INVALID_RESOURCE', `${path}.measurements`, 'must not be empty');
  }
  const measurementKeys = new Set<string>();
  for (const [index, candidate] of measurements.entries()) {
    const key = validateMeasurement(candidate, `${path}.measurements[${index}]`);
    if (measurementKeys.has(key)) {
      fail('INVALID_MEASUREMENT', `${path}.measurements[${index}]`, 'duplicates a measurement axis');
    }
    measurementKeys.add(key);
  }
  return {
    resourceId,
    backingResourceId,
    backingRelation,
    backingOffsetBytes,
    backingLengthBytes,
    addressableBytes,
    owner: ownerKey,
    role,
    space,
    allocator,
  };
}

function sameResource(left: ResourceDescriptor, right: ResourceDescriptor): boolean {
  return left.backingResourceId === right.backingResourceId &&
    left.backingRelation === right.backingRelation &&
    left.backingOffsetBytes === right.backingOffsetBytes &&
    left.backingLengthBytes === right.backingLengthBytes &&
    left.addressableBytes === right.addressableBytes &&
    left.owner === right.owner && left.role === right.role &&
    left.space === right.space && left.allocator === right.allocator;
}

function expectedEnvelopeSource(kind: MemoryEnvelopeKind): MemoryEvidenceSource {
  switch (kind) {
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_RSS:
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_PEAK_RSS:
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_PSS:
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_PRIVATE_BYTES:
      return MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_OS_SAMPLER;
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_MANAGED_HEAP_USED:
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_EXTERNAL_BYTES:
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_ARRAY_BUFFER_BYTES:
      return MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_RUNTIME_COUNTER;
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_DEVICE_PROCESS_USED:
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_DEVICE_TOTAL_USED:
    case MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_DEVICE_TOTAL_CAPACITY:
      return MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_DRIVER_SAMPLER;
    default:
      fail('INVALID_ENVELOPE', 'MemoryEnvelopeEvidence.kind', 'has no supported source family');
  }
}

function validateEnvelope(value: unknown, path: string): string {
  const envelope = record(value, path, MemoryEnvelopeEvidence);
  exactFields(envelope, path, [
    'kind', 'bytes', 'source', 'valueRelation', 'temporalCoverage', 'sampler',
  ]);
  const kind = knownEnum<MemoryEnvelopeKind>(
    envelope.kind,
    MEMORY_ENVELOPE_KINDS,
    `${path}.kind`,
  );
  const source = knownEnum<MemoryEvidenceSource>(
    envelope.source,
    MEMORY_EVIDENCE_SOURCES,
    `${path}.source`,
  );
  if (source !== expectedEnvelopeSource(kind)) {
    fail('INVALID_ENVELOPE', `${path}.source`, 'does not match the envelope kind');
  }
  const relation = knownEnum<MemoryValueRelation>(
    envelope.valueRelation,
    MEMORY_VALUE_RELATIONS,
    `${path}.valueRelation`,
  );
  if (relation === MemoryValueRelation.MEMORY_VALUE_RELATION_REQUESTED) {
    fail('INVALID_ENVELOPE', `${path}.valueRelation`, 'envelopes cannot report API-requested bytes');
  }
  const temporal = knownEnum<MemoryTemporalCoverage>(
    envelope.temporalCoverage,
    MEMORY_TEMPORAL_COVERAGES,
    `${path}.temporalCoverage`,
  );
  relationBytes(envelope.bytes, relation, path, 'INVALID_ENVELOPE');
  const sampler = text(envelope.sampler, `${path}.sampler`, 'INVALID_ENVELOPE');
  if (temporal === MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_SAMPLED_WINDOW &&
      relation !== MemoryValueRelation.MEMORY_VALUE_RELATION_LOWER_BOUND &&
      relation !== MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE) {
    fail('INVALID_ENVELOPE', `${path}.valueRelation`, 'must be LOWER_BOUND for SAMPLED_WINDOW');
  }
  if (kind === MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_PEAK_RSS &&
      (temporal !== MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_PROCESS_LIFETIME ||
        (relation !== MemoryValueRelation.MEMORY_VALUE_RELATION_EXACT &&
          relation !== MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE))) {
    fail(
      'INVALID_ENVELOPE',
      path,
      'PROCESS_PEAK_RSS must be EXACT over PROCESS_LIFETIME, or UNAVAILABLE',
    );
  }
  return `${kind}\u0000${sampler}`;
}

function validateCompleteInventory(
  resources: readonly ResourceDescriptor[],
  path: string,
): void {
  const ids = new Set(resources.map((resource) => resource.resourceId));
  for (const resource of resources) {
    if (resource.backingRelation !== MemoryBackingRelation.MEMORY_BACKING_RELATION_INDEPENDENT &&
        !ids.has(resource.backingResourceId)) {
      fail(
        'INVALID_RESOURCE',
        path,
        `complete inventory omits backing resource '${resource.backingResourceId}'`,
      );
    }
  }
  const siblings = new Map<string, ResourceDescriptor[]>();
  for (const resource of resources) {
    if (resource.backingRelation !==
        MemoryBackingRelation.MEMORY_BACKING_RELATION_SUBALLOCATION) continue;
    const group = siblings.get(resource.backingResourceId) ?? [];
    group.push(resource);
    siblings.set(resource.backingResourceId, group);
  }
  for (const [backingId, group] of siblings) {
    group.sort((left, right) => left.backingOffsetBytes! === right.backingOffsetBytes!
      ? 0
      : left.backingOffsetBytes! < right.backingOffsetBytes! ? -1 : 1);
    let previousEnd = 0n;
    let hasNonEmptyRange = false;
    for (const resource of group) {
      const start = resource.backingOffsetBytes!;
      const end = checkedAdd(start, resource.backingLengthBytes!, path);
      if (resource.backingLengthBytes! > 0n && hasNonEmptyRange && start < previousEnd) {
        fail(
          'INVALID_RESOURCE',
          path,
          `complete inventory has overlapping suballocations of '${backingId}'`,
        );
      }
      if (resource.backingLengthBytes! > 0n) {
        hasNonEmptyRange = true;
        if (end > previousEnd) previousEnd = end;
      }
    }
  }
}

function validateBackingGraph(definitions: ReadonlyMap<string, ResourceDescriptor>): void {
  // Check immediate containment first so every later chain traversal can
  // assume that all edges resolve to a valid resource descriptor.
  for (const resource of definitions.values()) {
    if (resource.backingRelation === MemoryBackingRelation.MEMORY_BACKING_RELATION_INDEPENDENT) {
      continue;
    }
    const backing = definitions.get(resource.backingResourceId);
    if (!backing) {
      fail(
        'INVALID_RESOURCE',
        'MemoryEvidence.snapshots',
        `resource '${resource.resourceId}' has unresolved backing '${resource.backingResourceId}'`,
      );
    }
    const rangeEnd = checkedAdd(
      resource.backingOffsetBytes!,
      resource.backingLengthBytes!,
      `MemoryEvidence resource '${resource.resourceId}' range`,
    );
    if (rangeEnd > backing.addressableBytes) {
      fail(
        'INVALID_RESOURCE',
        'MemoryEvidence.snapshots',
        `resource '${resource.resourceId}' exceeds backing '${resource.backingResourceId}'`,
      );
    }
  }

  // Resolve chains iteratively. Evidence can be adversarially deep, so a
  // recursive DFS would turn a validation failure into a JavaScript stack
  // overflow before it reached the fail-closed result.
  const rootOffset = new Map<string, bigint>();
  for (const resourceId of definitions.keys()) {
    if (rootOffset.has(resourceId)) continue;
    const chain: ResourceDescriptor[] = [];
    const active = new Set<string>();
    let currentId = resourceId;
    let baseOffset = 0n;
    while (true) {
      const cached = rootOffset.get(currentId);
      if (cached !== undefined) {
        baseOffset = cached;
        break;
      }
      if (active.has(currentId)) {
        fail(
          'INVALID_RESOURCE',
          'MemoryEvidence.snapshots',
          `backing cycle includes '${currentId}'`,
        );
      }
      active.add(currentId);
      const current = definitions.get(currentId)!;
      chain.push(current);
      if (current.backingRelation === MemoryBackingRelation.MEMORY_BACKING_RELATION_INDEPENDENT) {
        break;
      }
      currentId = current.backingResourceId;
    }
    for (let index = chain.length - 1; index >= 0; index--) {
      const current = chain[index];
      if (current.backingRelation !== MemoryBackingRelation.MEMORY_BACKING_RELATION_INDEPENDENT) {
        baseOffset = checkedAdd(
          baseOffset,
          current.backingOffsetBytes!,
          `MemoryEvidence resource '${current.resourceId}' absolute offset`,
        );
      }
      rootOffset.set(current.resourceId, baseOffset);
    }
  }
}

/** Validate one decoded v1 capture without mutating or freezing it. */
export function validateMemoryEvidence(
  value: unknown,
): MemoryEvidence {
  const evidence = record(value, 'MemoryEvidence', MemoryEvidence);
  exactFields(evidence, 'MemoryEvidence', ['format', 'captureId', 'requiredFeatures', 'snapshots']);
  if (evidence.format !== MEMORY_EVIDENCE_FORMAT) {
    fail('INVALID_FORMAT', 'MemoryEvidence.format', `must be exactly '${MEMORY_EVIDENCE_FORMAT}'`);
  }
  text(evidence.captureId, 'MemoryEvidence.captureId', 'INVALID_IDENTITY');

  const requiredFeatures = array(evidence.requiredFeatures, 'MemoryEvidence.requiredFeatures');
  const featureNames = new Set<string>();
  for (const [index, candidate] of requiredFeatures.entries()) {
    const featurePath = `MemoryEvidence.requiredFeatures[${index}]`;
    const feature = text(candidate, featurePath, 'INVALID_FEATURE');
    if (!/^[a-z][a-z0-9._-]*$/.test(feature)) {
      fail('INVALID_FEATURE', featurePath, 'must be a canonical lowercase feature token');
    }
    if (featureNames.has(feature)) fail('INVALID_FEATURE', featurePath, 'is duplicated');
    featureNames.add(feature);
    fail('INVALID_FEATURE', featurePath, `unsupported required feature '${feature}'`);
  }

  const snapshots = array(evidence.snapshots, 'MemoryEvidence.snapshots');
  if (snapshots.length === 0) {
    fail('INVALID_TIMELINE', 'MemoryEvidence.snapshots', 'must not be empty');
  }
  const definitions = new Map<string, ResourceDescriptor>();
  const occurrences = new Map<string, OccurrenceDescriptor>();
  let previousSequence: bigint | null = null;
  let previousMonotonicTime: bigint | null = null;

  for (const [snapshotIndex, candidate] of snapshots.entries()) {
    const snapshotPath = `MemoryEvidence.snapshots[${snapshotIndex}]`;
    const snapshot = record(candidate, snapshotPath, MemorySnapshot);
    exactFields(snapshot, snapshotPath, [
      'sequence',
      'monotonicTime',
      'stage',
      'point',
      'phaseOccurrenceId',
      'subject',
      'backend',
      'device',
      'domainAttestation',
      'resources',
      'envelopes',
      'resourceInventory',
    ]);
    const sequence = uint64(snapshot.sequence, `${snapshotPath}.sequence`);
    if (previousSequence !== null && sequence <= previousSequence) {
      fail('INVALID_TIMELINE', `${snapshotPath}.sequence`, 'must be strictly increasing');
    }
    previousSequence = sequence;

    if (snapshot.monotonicTime !== undefined) {
      const monotonic = record(
        snapshot.monotonicTime,
        `${snapshotPath}.monotonicTime`,
        MemoryMonotonicTime,
      );
      exactFields(monotonic, `${snapshotPath}.monotonicTime`, ['nanoseconds']);
      const nanoseconds = uint64(
        monotonic.nanoseconds,
        `${snapshotPath}.monotonicTime.nanoseconds`,
      );
      if (previousMonotonicTime !== null && nanoseconds < previousMonotonicTime) {
        fail('INVALID_TIMELINE', `${snapshotPath}.monotonicTime`, 'must not decrease');
      }
      previousMonotonicTime = nanoseconds;
    }
    const stage = knownEnum<OperationStage>(snapshot.stage, OPERATION_STAGES, `${snapshotPath}.stage`);
    const point = knownEnum<MemorySnapshotPoint>(
      snapshot.point,
      MEMORY_SNAPSHOT_POINTS,
      `${snapshotPath}.point`,
    );
    const phaseOccurrenceId = text(
      snapshot.phaseOccurrenceId,
      `${snapshotPath}.phaseOccurrenceId`,
      'INVALID_IDENTITY',
    );
    const subjectKey = owner(snapshot.subject, `${snapshotPath}.subject`);
    const backend = typeof snapshot.backend === 'string'
      ? snapshot.backend
      : fail('INVALID_TYPE', `${snapshotPath}.backend`, 'must be a string');
    const device = typeof snapshot.device === 'string'
      ? snapshot.device
      : fail('INVALID_TYPE', `${snapshotPath}.device`, 'must be a string');
    const occurrence = occurrences.get(phaseOccurrenceId);
    if (occurrence) {
      if (occurrence.stage !== stage || occurrence.subject !== subjectKey ||
          occurrence.backend !== backend || occurrence.device !== device) {
        fail('INVALID_TIMELINE', snapshotPath, 'changes metadata for one phase occurrence');
      }
      if (point !== MemorySnapshotPoint.MEMORY_SNAPSHOT_POINT_PERIODIC) {
        if (occurrence.singletonPoints.has(point)) {
          fail('INVALID_TIMELINE', `${snapshotPath}.point`, 'is duplicated for one phase occurrence');
        }
        occurrence.singletonPoints.add(point);
      }
    } else {
      occurrences.set(phaseOccurrenceId, {
        stage,
        subject: subjectKey,
        backend,
        device,
        singletonPoints: new Set(
          point === MemorySnapshotPoint.MEMORY_SNAPSHOT_POINT_PERIODIC ? [] : [point],
        ),
      });
    }

    if (snapshot.domainAttestation !== undefined) {
      validateDomainAttestation(snapshot.domainAttestation, `${snapshotPath}.domainAttestation`);
    }
    const inventory = knownEnum<MemoryInventoryKind>(
      snapshot.resourceInventory,
      MEMORY_INVENTORY_KINDS,
      `${snapshotPath}.resourceInventory`,
    );
    const snapshotResources: ResourceDescriptor[] = [];
    const snapshotResourceIds = new Set<string>();
    for (const [resourceIndex, resourceCandidate] of
      array(snapshot.resources, `${snapshotPath}.resources`).entries()) {
      const resourcePath = `${snapshotPath}.resources[${resourceIndex}]`;
      const descriptor = validateResource(resourceCandidate, resourcePath);
      if (snapshotResourceIds.has(descriptor.resourceId)) {
        fail('INVALID_RESOURCE', `${resourcePath}.resourceId`, 'is duplicated in one snapshot');
      }
      snapshotResourceIds.add(descriptor.resourceId);
      snapshotResources.push(descriptor);
      const existing = definitions.get(descriptor.resourceId);
      if (existing && !sameResource(existing, descriptor)) {
        fail('INVALID_RESOURCE', resourcePath, 'changes immutable resource identity metadata');
      }
      if (!existing) definitions.set(descriptor.resourceId, descriptor);
    }
    if (inventory === MemoryInventoryKind.MEMORY_INVENTORY_KIND_COMPLETE) {
      validateCompleteInventory(snapshotResources, `${snapshotPath}.resources`);
    }

    const envelopeKeys = new Set<string>();
    for (const [envelopeIndex, envelopeCandidate] of
      array(snapshot.envelopes, `${snapshotPath}.envelopes`).entries()) {
      const envelopePath = `${snapshotPath}.envelopes[${envelopeIndex}]`;
      const key = validateEnvelope(envelopeCandidate, envelopePath);
      if (envelopeKeys.has(key)) {
        fail('INVALID_ENVELOPE', envelopePath, 'duplicates an envelope kind/sampler');
      }
      envelopeKeys.add(key);
    }
  }

  validateBackingGraph(definitions);
  return evidence as unknown as MemoryEvidence;
}

/** Validate only the optional memory field on a decoded operation report. */
export function validateOperationReportMemoryEvidence(
  value: unknown,
): OperationReport {
  const report = record(value, 'OperationReport', OperationReport);
  if (report.memoryEvidence !== undefined) {
    validateMemoryEvidence(report.memoryEvidence);
  }
  return report as unknown as OperationReport;
}
