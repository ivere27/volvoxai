import test from 'node:test';
import assert from 'node:assert/strict';

import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';
import {
  MemoryBackingRelation,
  MemoryBoundKind,
  MemoryBoundProof,
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
  MemoryOwnerKind,
  MemoryOwnerRef,
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
} from '../runtime/generated/typescript/volvoxai_lite.js';
import {
  RuntimeServiceFfi,
  RuntimeServiceMethods,
} from '../runtime/generated/typescript/volvoxai_ffi.js';
import {
  MEMORY_EVIDENCE_FORMAT,
  MemoryEvidenceValidationError,
  validateMemoryEvidence,
  validateOperationReportMemoryEvidence,
} from '../runtime/typescript/MemoryEvidenceValidation.js';
import {
  createMemoryEvidenceValidatingPluginHost,
  DEFAULT_MAX_REPORT_RESPONSE_BYTES,
  MEMORY_EVIDENCE_NON_REPORT_RESPONSE_METHODS,
  MEMORY_EVIDENCE_REPORT_RESPONSE_DECODERS,
  MEMORY_EVIDENCE_REPORT_RESPONSE_METHODS,
} from '../runtime/typescript/MemoryEvidenceValidatingPluginHost.js';

const size = (bytes) => new MemoryByteSize({ bytes });
const contextOwner = () => new MemoryOwnerRef({
  kind: MemoryOwnerKind.MEMORY_OWNER_KIND_EXECUTION_CONTEXT,
  ownerId: 'context-1',
});

function exactMeasurement(bytes, {
  metric = MemoryMetric.MEMORY_METRIC_LIVE,
  source = MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_ALLOCATOR_COUNTER,
  relation = MemoryValueRelation.MEMORY_VALUE_RELATION_EXACT,
  temporal = MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_RESOURCE_LIFETIME,
} = {}) {
  return new MemoryMeasurement({
    metric,
    bytes: relation === MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE
      ? undefined
      : size(bytes),
    source,
    valueRelation: relation,
    temporalCoverage: temporal,
  });
}

function independentResource(resourceId, bytes, role, allocator) {
  return new MemoryResourceEvidence({
    resourceId,
    backingRelation: MemoryBackingRelation.MEMORY_BACKING_RELATION_INDEPENDENT,
    owner: contextOwner(),
    role,
    space: MemorySpace.MEMORY_SPACE_WASM_LINEAR,
    allocator,
    measurements: [exactMeasurement(bytes)],
    addressableBytes: size(bytes),
  });
}

function backedResource(resourceId, relation, offset, length, role) {
  return new MemoryResourceEvidence({
    resourceId,
    backingResourceId: 'arena',
    backingRelation: relation,
    backingOffsetBytes: size(offset),
    backingLengthBytes: size(length),
    owner: contextOwner(),
    role,
    space: MemorySpace.MEMORY_SPACE_WASM_LINEAR,
    allocator: 'wasm-arena-view',
    measurements: [exactMeasurement(length, {
      metric: MemoryMetric.MEMORY_METRIC_LOGICAL,
      source: MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_RUNTIME_COUNTER,
      temporal: MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_INSTANT,
    })],
    addressableBytes: size(length),
  });
}

function validEvidence() {
  const ordinary = new MemoryPeakCase({
    caseId: 'ordinary',
    terms: [
      new MemoryBoundTerm({
        termId: 'weights',
        ownerKind: MemoryOwnerKind.MEMORY_OWNER_KIND_COMPILED_MODEL,
        role: MemoryResourceRole.MEMORY_RESOURCE_ROLE_WEIGHTS,
        space: MemorySpace.MEMORY_SPACE_WASM_LINEAR,
        upperBoundBytes: size(100n),
      }),
      new MemoryBoundTerm({
        termId: 'activation',
        ownerKind: MemoryOwnerKind.MEMORY_OWNER_KIND_EXECUTION_CONTEXT,
        role: MemoryResourceRole.MEMORY_RESOURCE_ROLE_ACTIVATION,
        space: MemorySpace.MEMORY_SPACE_WASM_LINEAR,
        upperBoundBytes: size(50n),
      }),
    ],
    totalBytes: size(150n),
  });
  const decode = new MemoryPeakCase({
    caseId: 'decode',
    terms: [
      new MemoryBoundTerm({
        termId: 'weights',
        ownerKind: MemoryOwnerKind.MEMORY_OWNER_KIND_COMPILED_MODEL,
        role: MemoryResourceRole.MEMORY_RESOURCE_ROLE_WEIGHTS,
        space: MemorySpace.MEMORY_SPACE_WASM_LINEAR,
        upperBoundBytes: size(100n),
      }),
      new MemoryBoundTerm({
        termId: 'kv',
        ownerKind: MemoryOwnerKind.MEMORY_OWNER_KIND_EXECUTION_CONTEXT,
        role: MemoryResourceRole.MEMORY_RESOURCE_ROLE_KV_CACHE,
        space: MemorySpace.MEMORY_SPACE_WASM_LINEAR,
        upperBoundBytes: size(80n),
      }),
      new MemoryBoundTerm({
        termId: 'scratch',
        ownerKind: MemoryOwnerKind.MEMORY_OWNER_KIND_EXECUTION_CONTEXT,
        role: MemoryResourceRole.MEMORY_RESOURCE_ROLE_SCRATCH,
        space: MemorySpace.MEMORY_SPACE_WASM_LINEAR,
        upperBoundBytes: size(20n),
      }),
    ],
    totalBytes: size(200n),
  });
  const root = independentResource(
    'arena',
    100n,
    MemoryResourceRole.MEMORY_RESOURCE_ROLE_ARENA,
    'wasm-arena',
  );
  root.consumers.push(new MemoryOwnerRef({
    kind: MemoryOwnerKind.MEMORY_OWNER_KIND_COMPILED_MODEL,
    ownerId: 'compiled-1',
  }));
  root.consumers.push(contextOwner());
  root.measurements.push(exactMeasurement(100n, {
    metric: MemoryMetric.MEMORY_METRIC_LIVE_HIGH_WATER,
    temporal: MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_OPERATION_WINDOW,
  }));
  const copy = independentResource(
    'arena-copy',
    100n,
    MemoryResourceRole.MEMORY_RESOURCE_ROLE_ARENA,
    'wasm-arena',
  );
  copy.measurements.push(exactMeasurement(100n, {
    metric: MemoryMetric.MEMORY_METRIC_RESERVED,
    source: MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_API_REQUEST,
    relation: MemoryValueRelation.MEMORY_VALUE_RELATION_REQUESTED,
  }));
  const zero = independentResource(
    'zero-allocation',
    0n,
    MemoryResourceRole.MEMORY_RESOURCE_ROLE_METADATA,
    'wasm-arena',
  );
  const firstSnapshot = new MemorySnapshot({
    sequence: 1n,
    monotonicTime: new MemoryMonotonicTime({ nanoseconds: 10n }),
    stage: OperationStage.OPERATION_STAGE_EXECUTE,
    point: MemorySnapshotPoint.MEMORY_SNAPSHOT_POINT_BEFORE,
    phaseOccurrenceId: 'execute-1',
    subject: contextOwner(),
    backend: 'wasm',
    device: 'host',
    domainAttestation: new MemoryDomainAttestation({
      proofProtocol: 'canonical-symbolic-domain-proof/v1',
      resourceProtocol: 'bounded-resource-maxima/v1',
      graphFingerprint: 'graph-1',
      shapeDomainProofIdentity: 'shape-proof-1',
      bounds: [new MemoryBoundProof({
        budgetDomainId: 'provider-resident',
        kind: MemoryBoundKind.MEMORY_BOUND_KIND_ORDINARY_RESIDENT,
        cases: [ordinary, decode],
        maximumBytes: size(200n),
        limitBytes: size(256n),
      })],
    }),
    resources: [
      root,
      backedResource(
        'activation-a',
        MemoryBackingRelation.MEMORY_BACKING_RELATION_SUBALLOCATION,
        0n,
        40n,
        MemoryResourceRole.MEMORY_RESOURCE_ROLE_ACTIVATION,
      ),
      backedResource(
        'activation-b',
        MemoryBackingRelation.MEMORY_BACKING_RELATION_SUBALLOCATION,
        40n,
        60n,
        MemoryResourceRole.MEMORY_RESOURCE_ROLE_ACTIVATION,
      ),
      backedResource(
        'input-alias',
        MemoryBackingRelation.MEMORY_BACKING_RELATION_ALIAS,
        20n,
        40n,
        MemoryResourceRole.MEMORY_RESOURCE_ROLE_INPUT,
      ),
      copy,
      zero,
    ],
    envelopes: [
      new MemoryEnvelopeEvidence({
        kind: MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_RSS,
        bytes: size(1_000n),
        source: MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_OS_SAMPLER,
        valueRelation: MemoryValueRelation.MEMORY_VALUE_RELATION_EXACT,
        temporalCoverage: MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_INSTANT,
        sampler: 'procfs-rss',
      }),
      new MemoryEnvelopeEvidence({
        kind: MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_RSS,
        bytes: size(1_100n),
        source: MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_OS_SAMPLER,
        valueRelation: MemoryValueRelation.MEMORY_VALUE_RELATION_LOWER_BOUND,
        temporalCoverage: MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_SAMPLED_WINDOW,
        sampler: 'procfs-rss-window',
      }),
      new MemoryEnvelopeEvidence({
        kind: MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_MANAGED_HEAP_USED,
        bytes: size(500n),
        source: MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_RUNTIME_COUNTER,
        valueRelation: MemoryValueRelation.MEMORY_VALUE_RELATION_EXACT,
        temporalCoverage: MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_INSTANT,
        sampler: 'runtime-heap',
      }),
      new MemoryEnvelopeEvidence({
        kind: MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_DEVICE_PROCESS_USED,
        source: MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_DRIVER_SAMPLER,
        valueRelation: MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE,
        temporalCoverage: MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_INSTANT,
        sampler: 'driver-unavailable',
      }),
      new MemoryEnvelopeEvidence({
        kind: MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_PEAK_RSS,
        bytes: size(1_200n),
        source: MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_OS_SAMPLER,
        valueRelation: MemoryValueRelation.MEMORY_VALUE_RELATION_EXACT,
        temporalCoverage: MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_PROCESS_LIFETIME,
        sampler: 'getrusage-maxrss',
      }),
    ],
    resourceInventory: MemoryInventoryKind.MEMORY_INVENTORY_KIND_COMPLETE,
  });
  const secondSnapshot = new MemorySnapshot({
    sequence: 2n,
    monotonicTime: new MemoryMonotonicTime({ nanoseconds: 20n }),
    stage: OperationStage.OPERATION_STAGE_EXECUTE,
    point: MemorySnapshotPoint.MEMORY_SNAPSHOT_POINT_AFTER,
    phaseOccurrenceId: 'execute-1',
    subject: contextOwner(),
    backend: 'wasm',
    device: 'host',
    resources: [MemoryResourceEvidence.fromBinary(root.toBinary())],
    resourceInventory: MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL,
  });
  return new MemoryEvidence({
    format: MEMORY_EVIDENCE_FORMAT,
    captureId: 'capture-1',
    snapshots: [firstSnapshot, secondSnapshot],
  });
}

function assertInvalid(mutate, code, path) {
  const evidence = validEvidence();
  mutate(evidence);
  assert.throws(
    () => validateMemoryEvidence(evidence),
    (error) => error instanceof MemoryEvidenceValidationError &&
      error.code === code && `${error.path} ${error.message}`.includes(path),
  );
}

function staticResponseHost(response) {
  return {
    invoke() {
      return response;
    },
    openStream() {
      throw new Error('this test host does not implement streams');
    },
  };
}

function assertInvalidType(value) {
  assert.throws(
    () => validateMemoryEvidence(value),
    (error) => error instanceof MemoryEvidenceValidationError &&
      error.code === 'INVALID_TYPE',
  );
}

test('memory evidence validator accepts canonical max-of-sums, sharing, aliases, and exact zero', () => {
  const evidence = validEvidence();
  const decoded = MemoryEvidence.fromBinary(evidence.toBinary());
  assert.equal(validateMemoryEvidence(decoded), decoded);
  assert.equal(decoded.snapshots[0].domainAttestation.bounds[0].maximumBytes.bytes, 200n);
  assert.equal(decoded.snapshots[0].resources[5].addressableBytes.bytes, 0n);
  assert.equal(decoded.snapshots[0].envelopes[3].bytes, undefined);

  const report = new OperationReport({ memoryEvidence: decoded });
  assert.equal(validateOperationReportMemoryEvidence(report), report);
  const emptyReport = new OperationReport();
  assert.equal(validateOperationReportMemoryEvidence(emptyReport), emptyReport);
});

test('memory evidence v1 rejects every non-empty required feature set', () => {
  assertInvalid(
    (value) => value.requiredFeatures.push('test.feature-v1'),
    'INVALID_FEATURE',
    'requiredFeatures[0]',
  );
  assertInvalid(
    (value) => value.requiredFeatures.push('Not/Canonical'),
    'INVALID_FEATURE',
    'requiredFeatures[0]',
  );
});

test('nonblank strings use the protocol ASCII whitespace definition', () => {
  assertInvalid(
    (value) => { value.captureId = ' \t\n\v\f\r'; },
    'INVALID_IDENTITY',
    'captureId',
  );
  for (const captureId of ['\u0085', '\ufeff']) {
    const evidence = validEvidence();
    evidence.captureId = captureId;
    assert.equal(validateMemoryEvidence(evidence), evidence);
  }
});

test('memory evidence validator accepts only generated message instances', () => {
  const root = validEvidence();
  assertInvalidType({ ...root });
  assertInvalidType(Object.create({ ...root }));

  const plainNested = validEvidence();
  plainNested.snapshots[0] = { ...plainNested.snapshots[0] };
  assertInvalidType(plainNested);

  const prototypeNested = validEvidence();
  const resource = prototypeNested.snapshots[0].resources[0];
  prototypeNested.snapshots[0].resources[0] = Object.create({ ...resource });
  assertInvalidType(prototypeNested);
});

test('partial inventory records overlap without claiming disjoint physical accounting', () => {
  const evidence = validEvidence();
  evidence.snapshots[0].resourceInventory =
    MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL;
  evidence.snapshots[0].resources[2].backingOffsetBytes = size(20n);
  assert.equal(validateMemoryEvidence(evidence), evidence);
});

test('memory evidence validator rejects malformed proofs and timelines', async (t) => {
  const cases = [
    ['format', (v) => { v.format = 'memory/v2'; }, 'INVALID_FORMAT', '.format'],
    ['unknown field', (v) => { v.experimental = true; }, 'UNKNOWN_FIELD', 'MemoryEvidence'],
    ['sequence regression', (v) => { v.snapshots[1].sequence = 1n; },
      'INVALID_TIMELINE', 'sequence'],
    ['empty timeline', (v) => { v.snapshots = []; },
      'INVALID_TIMELINE', 'snapshots'],
    ['monotonic regression', (v) => { v.snapshots[1].monotonicTime.nanoseconds = 9n; },
      'INVALID_TIMELINE', 'monotonicTime'],
    ['occurrence drift', (v) => { v.snapshots[1].backend = 'cpu'; },
      'INVALID_TIMELINE', 'snapshots[1]'],
    ['duplicate singleton point', (v) => {
      v.snapshots[1].point = MemorySnapshotPoint.MEMORY_SNAPSHOT_POINT_BEFORE;
    }, 'INVALID_TIMELINE', '.point'],
    ['unspecified inventory', (v) => { v.snapshots[0].resourceInventory = 0; },
      'INVALID_ENUM', 'resourceInventory'],
    ['missing maximum', (v) => {
      v.snapshots[0].domainAttestation.bounds[0].maximumBytes = undefined;
    }, 'INVALID_PROOF', 'maximumBytes'],
    ['wrong case sum', (v) => {
      v.snapshots[0].domainAttestation.bounds[0].cases[0].totalBytes.bytes = 149n;
    }, 'INVALID_PROOF', 'totalBytes'],
    ['wrong case maximum', (v) => {
      v.snapshots[0].domainAttestation.bounds[0].maximumBytes.bytes = 150n;
    }, 'INVALID_PROOF', 'maximumBytes'],
    ['maximum above limit', (v) => {
      v.snapshots[0].domainAttestation.bounds[0].limitBytes.bytes = 199n;
    }, 'INVALID_PROOF', 'maximumBytes'],
    ['duplicate bound', (v) => {
      const bounds = v.snapshots[0].domainAttestation.bounds;
      bounds.push(MemoryBoundProof.fromBinary(bounds[0].toBinary()));
    }, 'INVALID_PROOF', 'bounds[1]'],
    ['duplicate case', (v) => {
      const cases = v.snapshots[0].domainAttestation.bounds[0].cases;
      cases.push(MemoryPeakCase.fromBinary(cases[0].toBinary()));
    }, 'INVALID_PROOF', 'caseId'],
    ['duplicate term', (v) => {
      const terms = v.snapshots[0].domainAttestation.bounds[0].cases[0].terms;
      terms.push(MemoryBoundTerm.fromBinary(terms[0].toBinary()));
    }, 'INVALID_PROOF', 'termId'],
    ['empty peak case', (v) => {
      v.snapshots[0].domainAttestation.bounds[0].cases[0].terms = [];
      v.snapshots[0].domainAttestation.bounds[0].cases[0].totalBytes = size(0n);
    }, 'INVALID_PROOF', 'terms'],
    ['uint64 term overflow', (v) => {
      v.snapshots[0].domainAttestation.bounds[0].cases[0].terms[0]
        .upperBoundBytes.bytes = 0x1_0000_0000_0000_0000n;
    }, 'INVALID_UINT64', 'upperBoundBytes'],
    ['unknown term enum', (v) => {
      v.snapshots[0].domainAttestation.bounds[0].cases[0].terms[0].role = 999;
    }, 'INVALID_ENUM', '.role'],
  ];
  for (const [name, mutate, code, path] of cases) {
    await t.test(name, () => assertInvalid(mutate, code, path));
  }
});

test('memory evidence validator rejects invalid resource identity and backing graphs', async (t) => {
  const cases = [
    ['missing addressable extent', (v) => {
      v.snapshots[0].resources[0].addressableBytes = undefined;
    }, 'INVALID_RESOURCE', 'addressableBytes'],
    ['backed extent mismatch', (v) => {
      v.snapshots[0].resources[1].addressableBytes.bytes = 39n;
    }, 'INVALID_RESOURCE', 'addressableBytes'],
    ['range exceeds parent', (v) => {
      v.snapshots[0].resourceInventory = MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL;
      v.snapshots[0].resources[1].backingOffsetBytes.bytes = 80n;
    }, 'INVALID_RESOURCE', 'exceeds backing'],
    ['unresolved backing', (v) => {
      v.snapshots[0].resourceInventory = MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL;
      v.snapshots[0].resources[1].backingResourceId = 'missing';
    }, 'INVALID_RESOURCE', 'unresolved backing'],
    ['backing cycle', (v) => {
      v.snapshots[0].resourceInventory = MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL;
      v.snapshots[0].resources = v.snapshots[0].resources.slice(0, 2);
      v.snapshots[1].resources = [];
      const root = v.snapshots[0].resources[0];
      root.backingRelation = MemoryBackingRelation.MEMORY_BACKING_RELATION_ALIAS;
      root.backingResourceId = 'activation-a';
      root.backingOffsetBytes = size(0n);
      root.backingLengthBytes = size(40n);
      root.addressableBytes = size(40n);
    }, 'INVALID_RESOURCE', 'backing cycle'],
    ['overlapping complete suballocations', (v) => {
      v.snapshots[0].resources[2].backingOffsetBytes.bytes = 20n;
    }, 'INVALID_RESOURCE', 'overlapping suballocations'],
    ['duplicate resource ID', (v) => {
      v.snapshots[0].resources.push(v.snapshots[0].resources[0]);
    }, 'INVALID_RESOURCE', 'resourceId'],
    ['resource metadata drift', (v) => {
      v.snapshots[1].resources[0].allocator = 'different-allocator';
    }, 'INVALID_RESOURCE', 'identity metadata'],
    ['duplicate consumer', (v) => {
      v.snapshots[0].resources[0].consumers.push(contextOwner());
    }, 'INVALID_RESOURCE', 'consumers[2]'],
    ['empty allocator', (v) => {
      v.snapshots[0].resources[0].allocator = '';
    }, 'INVALID_RESOURCE', '.allocator'],
    ['empty measurement inventory', (v) => {
      v.snapshots[0].resources[0].measurements = [];
    }, 'INVALID_RESOURCE', '.measurements'],
    ['complete inventory omits backing root', (v) => {
      v.snapshots[0].resources = v.snapshots[0].resources.slice(1);
    }, 'INVALID_RESOURCE', 'omits backing resource'],
    ['duplicate measurement axis', (v) => {
      const measurements = v.snapshots[0].resources[0].measurements;
      measurements.push(MemoryMeasurement.fromBinary(measurements[0].toBinary()));
    }, 'INVALID_MEASUREMENT', 'duplicates a measurement'],
  ];
  for (const [name, mutate, code, path] of cases) {
    await t.test(name, () => assertInvalid(mutate, code, path));
  }
});

test('memory evidence validator rejects contradictory measurement axes', async (t) => {
  const cases = [
    ['available measurement without bytes', (v) => {
      v.snapshots[0].resources[0].measurements[0].bytes = undefined;
    }, 'INVALID_MEASUREMENT', '.bytes'],
    ['API request marked exact', (v) => {
      v.snapshots[0].resources[0].measurements[0].source =
        MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_API_REQUEST;
    }, 'INVALID_MEASUREMENT', 'valueRelation'],
    ['resource uses OS sampler', (v) => {
      v.snapshots[0].resources[0].measurements[0].source =
        MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_OS_SAMPLER;
    }, 'INVALID_MEASUREMENT', '.source'],
    ['high-water metric at instant', (v) => {
      const measurement = v.snapshots[0].resources[0].measurements[0];
      measurement.metric = MemoryMetric.MEMORY_METRIC_LIVE_HIGH_WATER;
      measurement.temporalCoverage = MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_INSTANT;
    }, 'INVALID_MEASUREMENT', 'temporalCoverage'],
    ['unavailable envelope with bytes', (v) => {
      v.snapshots[0].envelopes[3].bytes = size(0n);
    }, 'INVALID_ENVELOPE', '.bytes'],
    ['available envelope without bytes', (v) => {
      v.snapshots[0].envelopes[0].bytes = undefined;
    }, 'INVALID_ENVELOPE', '.bytes'],
    ['envelope source mismatch', (v) => {
      v.snapshots[0].envelopes[0].source =
        MemoryEvidenceSource.MEMORY_EVIDENCE_SOURCE_RUNTIME_COUNTER;
    }, 'INVALID_ENVELOPE', '.source'],
    ['sampled window marked exact', (v) => {
      v.snapshots[0].envelopes[1].valueRelation =
        MemoryValueRelation.MEMORY_VALUE_RELATION_EXACT;
    }, 'INVALID_ENVELOPE', 'valueRelation'],
    ['envelope marked requested', (v) => {
      v.snapshots[0].envelopes[0].valueRelation =
        MemoryValueRelation.MEMORY_VALUE_RELATION_REQUESTED;
    }, 'INVALID_ENVELOPE', 'valueRelation'],
    ['peak RSS at instant', (v) => {
      v.snapshots[0].envelopes[4].temporalCoverage =
        MemoryTemporalCoverage.MEMORY_TEMPORAL_COVERAGE_INSTANT;
    }, 'INVALID_ENVELOPE', 'PROCESS_PEAK_RSS'],
    ['duplicate envelope axis', (v) => {
      const envelopes = v.snapshots[0].envelopes;
      envelopes.push(MemoryEnvelopeEvidence.fromBinary(envelopes[0].toBinary()));
    }, 'INVALID_ENVELOPE', 'duplicates an envelope'],
  ];
  for (const [name, mutate, code, path] of cases) {
    await t.test(name, () => assertInvalid(mutate, code, path));
  }
});

test('RuntimeServiceFfi rejects an invalid direct operation report at the host boundary', () => {
  const evidence = validEvidence();
  evidence.format = 'memory/v2';
  const response = new OperationReport({ memoryEvidence: evidence }).toBinary();
  const ffi = new RuntimeServiceFfi(
    createMemoryEvidenceValidatingPluginHost(staticResponseHost(response)),
  );

  assert.throws(
    () => ffi.closeRuntime(new pb.RuntimeRef()),
    (error) => error instanceof MemoryEvidenceValidationError &&
      error.code === 'INVALID_FORMAT',
  );
});

test('RuntimeServiceFfi rejects invalid memory evidence nested through PTQ and revision reports', () => {
  const evidence = validEvidence();
  evidence.snapshots[0].domainAttestation.bounds[0].maximumBytes.bytes = 199n;
  const response = new pb.PtqPackageInfo({
    plan: new pb.PtqPlanInfo({
      revision: new pb.RevisionInfo({
        report: new OperationReport({ memoryEvidence: evidence }),
      }),
    }),
  }).toBinary();
  const ffi = new RuntimeServiceFfi(
    createMemoryEvidenceValidatingPluginHost(staticResponseHost(response)),
  );

  assert.throws(
    () => ffi.writePtqPackage(new pb.WritePtqPackageRequest()),
    (error) => error instanceof MemoryEvidenceValidationError &&
      error.code === 'INVALID_PROOF',
  );
});

test('RuntimeServiceFfi returns a valid report after host-boundary validation', () => {
  const response = new OperationReport({ memoryEvidence: validEvidence() }).toBinary();
  const ffi = new RuntimeServiceFfi(
    createMemoryEvidenceValidatingPluginHost(staticResponseHost(response)),
  );

  const report = ffi.closeRuntime(new pb.RuntimeRef());
  assert.ok(report instanceof OperationReport);
  assert.ok(report.memoryEvidence instanceof MemoryEvidence);
  assert.equal(report.memoryEvidence.captureId, 'capture-1');
});

test('report-bearing RuntimeService responses are capped before protobuf decoding', () => {
  const response = new OperationReport({ memoryEvidence: validEvidence() }).toBinary();
  assert.ok(Number.isSafeInteger(DEFAULT_MAX_REPORT_RESPONSE_BYTES));
  assert.ok(DEFAULT_MAX_REPORT_RESPONSE_BYTES > 0);
  const ffi = new RuntimeServiceFfi(createMemoryEvidenceValidatingPluginHost(
    staticResponseHost(response),
    { maxReportResponseBytes: response.byteLength - 1 },
  ));

  assert.throws(
    () => ffi.closeRuntime(new pb.RuntimeRef()),
    (error) => error instanceof MemoryEvidenceValidationError &&
      error.code === 'RESPONSE_TOO_LARGE',
  );
});

test('response cap uses intrinsic Uint8Array length instead of subclass accessors', () => {
  class MisreportingResponse extends Uint8Array {
    get byteLength() {
      return 0;
    }
  }
  const encoded = new OperationReport({ memoryEvidence: validEvidence() }).toBinary();
  const response = new MisreportingResponse(encoded);
  const ffi = new RuntimeServiceFfi(createMemoryEvidenceValidatingPluginHost(
    staticResponseHost(response),
    { maxReportResponseBytes: encoded.byteLength - 1 },
  ));

  assert.throws(
    () => ffi.closeRuntime(new pb.RuntimeRef()),
    (error) => error instanceof MemoryEvidenceValidationError &&
      error.code === 'RESPONSE_TOO_LARGE',
  );
});

test('malformed report-bearing host responses use a stable validation error', async (t) => {
  const cases = [
    ['non-binary response', 'not-a-byte-array'],
    // OperationReport.code declares five bytes but supplies only one.
    ['truncated protobuf', new Uint8Array([0x12, 0x05, 0x61])],
  ];
  for (const [name, response] of cases) {
    await t.test(name, () => {
      const ffi = new RuntimeServiceFfi(
        createMemoryEvidenceValidatingPluginHost(staticResponseHost(response)),
      );
      assert.throws(
        () => ffi.closeRuntime(new pb.RuntimeRef()),
        (error) => error instanceof MemoryEvidenceValidationError &&
          error.code === 'INVALID_RESPONSE',
      );
    });
  }
});

test('validating host snapshots original protobuf bytes including unknown wire fields', () => {
  const encoded = new OperationReport({ memoryEvidence: validEvidence() }).toBinary();
  // Field 99, varint wire type, value 1. The generated decoder intentionally
  // discards this field, so the wrapper must return the original bytes.
  const response = new Uint8Array(new SharedArrayBuffer(encoded.byteLength + 3));
  response.set(encoded);
  response.set([0x98, 0x06, 0x01], encoded.byteLength);
  const host = createMemoryEvidenceValidatingPluginHost(staticResponseHost(response));

  const returned = host.invoke(
    'RuntimeService',
    RuntimeServiceMethods.CloseRuntime,
    new pb.RuntimeRef().toBinary(),
  );
  assert.notEqual(returned, response);
  assert.deepEqual(returned, response);
  response.fill(0);
  assert.notDeepEqual(returned, response);
  assert.deepEqual(returned.subarray(-3), new Uint8Array([0x98, 0x06, 0x01]));
});

test('memory-evidence host method collections exhaustively partition RuntimeServiceMethods', () => {
  const reportMethods = new Set(MEMORY_EVIDENCE_REPORT_RESPONSE_METHODS);
  const nonReportMethods = new Set(MEMORY_EVIDENCE_NON_REPORT_RESPONSE_METHODS);
  const runtimeMethods = new Set(Object.values(RuntimeServiceMethods));

  assert.deepEqual(
    new Set([...reportMethods].filter((method) => nonReportMethods.has(method))),
    new Set(),
  );
  assert.deepEqual(new Set([...reportMethods, ...nonReportMethods]), runtimeMethods);

  const containsOperationReport = (constructor, visiting = new Set()) => {
    if (constructor === OperationReport) return true;
    if (visiting.has(constructor)) return false;
    visiting.add(constructor);
    for (const field of constructor.fields) {
      if (field.kind !== 'message') continue;
      const nested = typeof field.messageType === 'string'
        ? pb[field.messageType]
        : field.messageType;
      assert.equal(typeof nested, 'function', `missing message type ${field.messageType}`);
      if (containsOperationReport(nested, visiting)) return true;
    }
    return false;
  };

  const descriptorReportMethods = new Set();
  for (const methodPath of runtimeMethods) {
    const rpcName = methodPath.slice(methodPath.lastIndexOf('/') + 1);
    const clientMethodName = rpcName[0].toLowerCase() + rpcName.slice(1);
    const clientMethod = RuntimeServiceFfi.prototype[clientMethodName];
    assert.equal(typeof clientMethod, 'function', `missing client method ${clientMethodName}`);
    const match = clientMethod.toString().match(/return pb\.([A-Za-z0-9_]+)\.fromBinary/);
    assert.ok(match, `missing generated response decoder for ${methodPath}`);
    const responseConstructor = pb[match[1]];
    assert.equal(typeof responseConstructor, 'function', `missing response ${match[1]}`);
    if (!containsOperationReport(responseConstructor)) continue;
    descriptorReportMethods.add(methodPath);
    assert.equal(
      MEMORY_EVIDENCE_REPORT_RESPONSE_DECODERS[methodPath],
      responseConstructor,
      `wrong validating decoder for ${methodPath}`,
    );
  }
  assert.deepEqual(reportMethods, descriptorReportMethods);
});
