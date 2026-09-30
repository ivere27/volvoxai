import {
  MemoryBoundProof,
  MemoryBoundKind,
  MemoryBoundTerm,
  MemoryByteSize,
  MemoryDomainAttestation,
  MemoryInventoryKind,
  MemoryOwnerRef,
  MemoryOwnerKind,
  MemoryPeakCase,
  MemoryResourceRole,
  MemorySnapshot,
  MemoryAllocation,
  ResourceSnapshot,
  ObservationStatus,
  ObservedBytes,
  MemorySpace,
} from '../generated/typescript/volvoxai_lite.js';

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

const MEMORY_BOUND_KINDS = enumMembers(MemoryBoundKind);
const MEMORY_INVENTORY_KINDS = enumMembers(MemoryInventoryKind);
const MEMORY_OWNER_KINDS = enumMembers(MemoryOwnerKind);
const MEMORY_RESOURCE_ROLES = enumMembers(MemoryResourceRole);
const MEMORY_SPACES = enumMembers(MemorySpace);

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

export function validateMemoryBounds(value: unknown): MemoryDomainAttestation {
  validateDomainAttestation(value, 'MemoryDomainAttestation');
  return value as MemoryDomainAttestation;
}

/** Check identities and storage accounting without turning a partial inventory into a total. */
export function validateMemorySnapshot(value: unknown): MemorySnapshot {
  const snapshot = record(value, 'MemorySnapshot', MemorySnapshot);
  owner(snapshot.subject, 'MemorySnapshot.subject');
  knownEnum(snapshot.inventory, MEMORY_INVENTORY_KINDS, 'MemorySnapshot.inventory');
  const ids = new Set<bigint>();
  for (const [i, item] of array(snapshot.allocations, 'MemorySnapshot.allocations').entries()) {
    const path = `MemorySnapshot.allocations[${i}]`;
    const allocation = record(item, path, MemoryAllocation);
    const id = uint64(allocation.allocationId, `${path}.allocationId`);
    if (!id || ids.has(id)) fail('INVALID_IDENTITY', path, 'allocation IDs must be nonzero and unique');
    ids.add(id);
    owner(allocation.owner, `${path}.owner`);
    knownEnum(allocation.space, MEMORY_SPACES, `${path}.space`);
    knownEnum(allocation.role, MEMORY_RESOURCE_ROLES, `${path}.role`);
    text(allocation.allocator, `${path}.allocator`, 'INVALID_RESOURCE');
    uint64(allocation.capacityBytes, `${path}.capacityBytes`);
  }
  const present = (value as MemorySnapshot).toJson();
  for (const key of ['unconsumedResultBytes', 'retainedResultCapacityBytes', 'idleResultCapacityBytes'])
    if (present[key] !== undefined) uint64(snapshot[key], `MemorySnapshot.${key}`);
  if (present.idleResultCapacityBytes !== undefined && present.retainedResultCapacityBytes !== undefined &&
      typeof snapshot.idleResultCapacityBytes === 'bigint' && typeof snapshot.retainedResultCapacityBytes === 'bigint' &&
      snapshot.idleResultCapacityBytes > snapshot.retainedResultCapacityBytes)
    fail('INVALID_MEASUREMENT', 'MemorySnapshot', 'idle capacity exceeds retained result capacity');
  return value as MemorySnapshot;
}
const OBSERVATION_STATUSES = enumMembers(ObservationStatus);
function observed(value: ObservedBytes | undefined, path: string): void {
  if (!value) fail('INVALID_MEASUREMENT', path, 'missing observation status');
  knownEnum(value.status, OBSERVATION_STATUSES, `${path}.status`);
  if ((value.status === ObservationStatus.OBSERVATION_STATUS_AVAILABLE) !== (value.toJson().bytes !== undefined))
    fail('INVALID_MEASUREMENT', path, 'AVAILABLE requires bytes; unavailable observations omit bytes');
  if (value.toJson().bytes !== undefined) uint64(value.bytes, `${path}.bytes`);
}
export function validateResourceSnapshot(value: unknown): ResourceSnapshot {
  record(value, 'ResourceSnapshot', ResourceSnapshot);
  const snapshot = value as ResourceSnapshot;
  const start = uint64(snapshot.observationStartNs, 'ResourceSnapshot.observationStartNs');
  const end = uint64(snapshot.observationEndNs, 'ResourceSnapshot.observationEndNs');
  if (end < start) fail('INVALID_TIMELINE', 'ResourceSnapshot', 'observation ends before it starts');
  if (snapshot.memory) validateMemorySnapshot(snapshot.memory);
  if (!snapshot.wasm || !snapshot.cpu || !snapshot.gpu || !snapshot.process)
    fail('INVALID_MEASUREMENT', 'ResourceSnapshot', 'each observation must expose its support status');
  for (const part of [snapshot.wasm, snapshot.cpu, snapshot.gpu])
    knownEnum(part.status, OBSERVATION_STATUSES, 'ResourceSnapshot.status');
  const available = ObservationStatus.OBSERVATION_STATUS_AVAILABLE;
  if (snapshot.wasm.status === available) {
    const wasm = snapshot.wasm;
    const fields = ['allocatedBytes', 'freeBytes', 'allocatorMetadataBytes', 'modulePrefixBytes', 'pageSlackBytes', 'untrackedBytes'] as const;
    let total = 0n;
    for (const field of fields) total = checkedAdd(total, uint64(wasm[field], `wasm.${field}`), 'wasm');
    if (total !== wasm.linearBytes || wasm.largestFreeBlockBytes > wasm.freeBytes)
      fail('INVALID_MEASUREMENT', 'ResourceSnapshot.wasm', 'invalid linear memory partition');
  }
  observed(snapshot.process.resident, 'process.resident');
  observed(snapshot.process.peakResident, 'process.peakResident');
  uint64(snapshot.cpu.processTimeNs, 'cpu.processTimeNs');
  for (const [i, gpu] of snapshot.gpu.devices.entries()) {
    const path = `gpu.devices[${i}]`;
    text(gpu.deviceId, `${path}.deviceId`, 'INVALID_IDENTITY');
    knownEnum(gpu.utilizationStatus, OBSERVATION_STATUSES, `${path}.utilizationStatus`);
    for (const key of ['gpuUtilization', 'memoryUtilization'] as const) {
      const ratio = gpu.toJson()[key] !== undefined ? gpu[key] : undefined;
      if ((gpu.utilizationStatus === available) !== (ratio !== undefined) ||
          (ratio !== undefined && (!Number.isFinite(ratio) || ratio < 0 || ratio > 1)))
        fail('INVALID_MEASUREMENT', `${path}.${key}`, 'available utilization must be a finite ratio in [0, 1]');
    }
    observed(gpu.memoryTotal, `${path}.memoryTotal`);
    observed(gpu.memoryUsed, `${path}.memoryUsed`);
    if (gpu.memoryUsed?.status === available && gpu.memoryTotal?.status === available &&
        gpu.memoryUsed.bytes > gpu.memoryTotal.bytes)
      fail('INVALID_MEASUREMENT', path, 'device memory used exceeds total');
  }
  return snapshot;
}
