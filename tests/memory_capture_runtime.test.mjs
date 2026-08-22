import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MemoryBackingRelation,
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
  Model,
  OperationStage,
  Runtime,
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  createBackendProviderBatchContract,
  createBackendProviderCapabilities,
  parseGraphDocument,
} from '../ts/index.js';
import { InvariantResourceStore } from '../ts/backends/InvariantResources.js';
import {
  BACKEND_MEMORY_SNAPSHOT_PROTOCOL,
  MEMORY_CAPTURE_PROTOCOL,
  normalizeMemoryCaptureOptions,
  sampleRuntimeMemoryEnvelopes,
} from '../ts/core/MemoryCapture.js';
import * as wire from '../runtime/generated/typescript/volvoxai_lite.js';
import {
  validateMemoryEvidence,
} from '../runtime/typescript/MemoryEvidenceValidation.js';

function identitySnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B'] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B'] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function shapeDomain(input) {
  return Object.freeze({
    proofProtocol: 'canonical-symbolic-domain-proof/v1',
    resourceProtocol: 'bounded-resource-maxima/v1',
    support: 'full',
    graphFingerprint: input.graphFingerprint,
    proof: input.shapeDomainProof,
    maximumTensorBytes: 16,
    maximumResidentBytes: 32,
    resourceLimitBytes: 4096,
  });
}

function hostOutput(request) {
  return Object.freeze({
    name: request.outputDescriptors[0].name,
    shape: request.outputDescriptors[0].shape,
    dtype: request.outputDescriptors[0].dtype,
    location: 'host',
    ownership: 'borrowed',
    data: new Float32Array(request.inputs.x.data),
  });
}

function backendResourceSnapshot(request, resourceBytes = 64) {
  return {
    protocol: BACKEND_MEMORY_SNAPSHOT_PROTOCOL,
    resources: [{
      resourceId: 'fixture-buffer-1',
      backingResourceId: '',
      backingRelation: MemoryBackingRelation.Independent,
      owner: {
        kind: request.subject.kind,
        ownerId: request.subject.ownerId,
      },
      role: MemoryResourceRole.Activation,
      space: MemorySpace.Host,
      allocator: 'fixture-runtime-counter/v1',
      consumers: [],
      measurements: [{
        metric: MemoryMetric.Live,
        bytes: { bytes: resourceBytes },
        source: MemoryEvidenceSource.RuntimeCounter,
        valueRelation: MemoryValueRelation.Exact,
        temporalCoverage: MemoryTemporalCoverage.Instant,
      }],
      addressableBytes: { bytes: resourceBytes },
    }],
    resourceInventory: MemoryInventoryKind.Complete,
  };
}

function providerFixture({
  name = 'memory-fixture',
  allocationBytes = 123_456,
  backendReport = Object.freeze({ allocationBytes: 78_901, activationCapacityBytes: 45_678 }),
  failCompile = false,
  failExecute = false,
  captureMemorySnapshot = null,
} = {}) {
  const captureRequests = [];
  const returnedMemorySnapshots = [];
  const returnedOutputs = [];
  let executeCalls = 0;
  const resourceDomain = Object.freeze({});
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: name,
    capabilities: createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'host',
      dynamicShapeDomain: 'full',
    }),
    async compile(input) {
      if (failCompile) throw new Error('fixture compilation failed');
      const compatibilityTokens = new Map();
      const invariantResources = new InvariantResourceStore(() => 0);
      return {
        backendName: name,
        invariantResources,
        batchContract: createBackendProviderBatchContract('unsupported'),
        prepareBatchRoute(plan) {
          let compatibilityToken = compatibilityTokens.get(plan.signature);
          if (!compatibilityToken) {
            compatibilityToken = Object.freeze({});
            compatibilityTokens.set(plan.signature, compatibilityToken);
          }
          return Object.freeze({
            resourceDomain,
            compatibilityToken,
            deviceEpoch: invariantResources.deviceEpoch,
          });
        },
        compilationEvidence: Object.freeze({
          device: Object.freeze({ vendor: 'Fixture Vendor', device: 'Fixture Device' }),
          allocationBytes,
          operatorFallbackUsed: false,
          offendingNode: null,
          shapeDomain: shapeDomain(input),
        }),
        async createContext() {
          const context = {
            backendName: name,
            async execute(request) {
              executeCalls++;
              if (typeof failExecute === 'function'
                ? failExecute(executeCalls)
                : failExecute) {
                throw new Error('fixture execution failed');
              }
              const output = hostOutput(request);
              returnedOutputs.push(output);
              return Object.freeze({
                outputs: Object.freeze([output]),
                backendReport,
              });
            },
            async close() {},
          };
          if (captureMemorySnapshot !== null) {
            context.captureMemorySnapshot = (request) => {
              captureRequests.push(request);
              const snapshot = captureMemorySnapshot(
                request,
                captureRequests.length,
                returnedOutputs.at(-1),
              );
              returnedMemorySnapshots.push(snapshot);
              return snapshot;
            };
          }
          return context;
        },
        async close() {},
      };
    },
    async close() {},
  };
  return {
    provider,
    captureRequests,
    returnedMemorySnapshots,
    returnedOutputs,
    get executeCalls() { return executeCalls; },
  };
}

function captureOptions(overrides = {}) {
  return {
    protocol: MEMORY_CAPTURE_PROTOCOL,
    includeResourceInventory: false,
    includeDomainAttestation: false,
    requestedEnvelopes: [MemoryEnvelopeKind.ProcessRss],
    ...overrides,
  };
}

async function compileFixture(runtime, fixture) {
  runtime._addProvider(fixture.provider.backendName, fixture.provider);
  return runtime.compile(identitySnapshot(), {
    backend: {
      mode: 'require',
      backend: fixture.provider.backendName,
      operatorFallback: 'forbid',
    },
  });
}

function onlySnapshot(report) {
  assert.equal(Object.prototype.hasOwnProperty.call(report, 'memoryEvidence'), true);
  assert.equal(report.memoryEvidence.snapshots.length, 1);
  return report.memoryEvidence.snapshots[0];
}

function assertSnapshotIdentity(snapshot, { stage, point, ownerKind, ownerId }) {
  assert.equal(snapshot.stage, stage);
  assert.equal(snapshot.point, point);
  assert.equal(snapshot.subject.kind, ownerKind);
  assert.equal(snapshot.subject.ownerId, ownerId);
}

function assertDeepFrozen(value, seen = new Set()) {
  if (value === null || typeof value !== 'object' || seen.has(value)) return;
  seen.add(value);
  assert.equal(Object.isFrozen(value), true);
  for (const key of Reflect.ownKeys(value)) assertDeepFrozen(value[key], seen);
}

function wireBytes(value) {
  return value === undefined
    ? undefined
    : new wire.MemoryByteSize({ bytes: BigInt(value.bytes) });
}

function wireOwner(value) {
  return new wire.MemoryOwnerRef({
    kind: value.kind,
    ownerId: value.ownerId,
  });
}

function wireDomainAttestation(value) {
  if (value === undefined) return undefined;
  return new wire.MemoryDomainAttestation({
    proofProtocol: value.proofProtocol,
    resourceProtocol: value.resourceProtocol,
    graphFingerprint: value.graphFingerprint,
    shapeDomainProofIdentity: value.shapeDomainProofIdentity,
    bounds: value.bounds.map((bound) => new wire.MemoryBoundProof({
      budgetDomainId: bound.budgetDomainId,
      kind: bound.kind,
      cases: bound.cases.map((peakCase) => new wire.MemoryPeakCase({
        caseId: peakCase.caseId,
        terms: peakCase.terms.map((term) => new wire.MemoryBoundTerm({
          termId: term.termId,
          ownerKind: term.ownerKind,
          role: term.role,
          space: term.space,
          upperBoundBytes: wireBytes(term.upperBoundBytes),
        })),
        totalBytes: wireBytes(peakCase.totalBytes),
      })),
      maximumBytes: wireBytes(bound.maximumBytes),
      limitBytes: wireBytes(bound.limitBytes),
    })),
  });
}

function wireResource(value) {
  return new wire.MemoryResourceEvidence({
    resourceId: value.resourceId,
    backingResourceId: value.backingResourceId,
    backingRelation: value.backingRelation,
    backingOffsetBytes: wireBytes(value.backingOffsetBytes),
    backingLengthBytes: wireBytes(value.backingLengthBytes),
    owner: wireOwner(value.owner),
    role: value.role,
    space: value.space,
    allocator: value.allocator,
    consumers: value.consumers.map(wireOwner),
    measurements: value.measurements.map((measurement) => new wire.MemoryMeasurement({
      metric: measurement.metric,
      bytes: wireBytes(measurement.bytes),
      source: measurement.source,
      valueRelation: measurement.valueRelation,
      temporalCoverage: measurement.temporalCoverage,
    })),
    addressableBytes: wireBytes(value.addressableBytes),
  });
}

/** Explicit public number DTO -> protobuf bigint wire-boundary projection. */
function toWireMemoryEvidence(value) {
  return new wire.MemoryEvidence({
    format: value.format,
    captureId: value.captureId,
    requiredFeatures: [...value.requiredFeatures],
    snapshots: value.snapshots.map((snapshot) => new wire.MemorySnapshot({
      sequence: BigInt(snapshot.sequence),
      monotonicTime: snapshot.monotonicTime === undefined
        ? undefined
        : new wire.MemoryMonotonicTime({
          nanoseconds: BigInt(snapshot.monotonicTime.nanoseconds),
        }),
      stage: snapshot.stage,
      point: snapshot.point,
      phaseOccurrenceId: snapshot.phaseOccurrenceId,
      subject: wireOwner(snapshot.subject),
      backend: snapshot.backend,
      device: snapshot.device,
      domainAttestation: wireDomainAttestation(snapshot.domainAttestation),
      resources: snapshot.resources.map(wireResource),
      envelopes: snapshot.envelopes.map((envelope) => new wire.MemoryEnvelopeEvidence({
        kind: envelope.kind,
        bytes: wireBytes(envelope.bytes),
        source: envelope.source,
        valueRelation: envelope.valueRelation,
        temporalCoverage: envelope.temporalCoverage,
        sampler: envelope.sampler,
      })),
      resourceInventory: snapshot.resourceInventory,
    })),
  });
}

function assertInvalidCaptureOptions(value, message) {
  assert.throws(
    () => new Runtime({ memoryCapture: value }),
    (error) => {
      assert.equal(error.code, 'INVALID_ARGUMENT');
      assert.equal(error.phase, 'initialization');
      assert.match(error.message, message);
      return true;
    },
  );
}

test('disabled memory capture preserves report shape and never samples or calls the provider hook',
  async (t) => {
    const originalMemoryUsage = process.memoryUsage;
    let processSamplerCalls = 0;
    process.memoryUsage = (...args) => {
      processSamplerCalls++;
      return originalMemoryUsage.apply(process, args);
    };
    t.after(() => { process.memoryUsage = originalMemoryUsage; });

    const fixture = providerFixture({
      captureMemorySnapshot: (request) => backendResourceSnapshot(request),
    });
    const runtime = new Runtime();
    const compiled = await compileFixture(runtime, fixture);
    const context = await compiled.createContext();
    const result = await context.execute({
      x: { data: Float32Array.of(1, 2), shape: [2] },
    });

    assert.equal(Object.prototype.hasOwnProperty.call(compiled.report, 'memoryEvidence'), false);
    assert.equal(Object.prototype.hasOwnProperty.call(result.report, 'memoryEvidence'), false);
    assert.equal(processSamplerCalls, 0);
    assert.equal(fixture.captureRequests.length, 0);

    await result.close();
    await context.close();
    await compiled.close();
    await runtime.close();
  });

test('memory capture options fail closed on protocol, selection, enum, duplicate, and periodic errors',
  () => {
    assertInvalidCaptureOptions(
      captureOptions({ protocol: 'volvoxai-memory-capture/v2' }),
      /protocol must be exactly/,
    );
    assertInvalidCaptureOptions(
      captureOptions({ requestedEnvelopes: [] }),
      /must select a resource inventory, domain attestation, or envelope/,
    );
    assertInvalidCaptureOptions(
      captureOptions({ requestedEnvelopes: [MemoryEnvelopeKind.Unspecified] }),
      /known nonzero MemoryEnvelopeKind/,
    );
    assertInvalidCaptureOptions(
      captureOptions({
        requestedEnvelopes: [MemoryEnvelopeKind.ProcessRss, MemoryEnvelopeKind.ProcessRss],
      }),
      /is duplicated/,
    );
    assertInvalidCaptureOptions(
      captureOptions({ samplingIntervalNanoseconds: 1_000_000 }),
      /interval and snapshot limit must be present together/,
    );
    assertInvalidCaptureOptions(
      captureOptions({
        samplingIntervalNanoseconds: 1_000_000,
        maxPeriodicSnapshots: 4,
      }),
      /Periodic runtime memory capture is not supported yet/,
    );
    assertInvalidCaptureOptions(
      { ...captureOptions(), unexpected: true },
      /does not accept option 'unexpected'/,
    );
  });

test('capture options are detached and deeply frozen before runtime use', () => {
  const requestedEnvelopes = [MemoryEnvelopeKind.ProcessRss];
  const source = captureOptions({
    includeResourceInventory: true,
    includeDomainAttestation: true,
    requestedEnvelopes,
  });
  const normalized = normalizeMemoryCaptureOptions(source);

  assert.notEqual(normalized, source);
  assert.notEqual(normalized.requestedEnvelopes, requestedEnvelopes);
  assert.equal(Object.isFrozen(normalized), true);
  assert.equal(Object.isFrozen(normalized.requestedEnvelopes), true);

  source.protocol = 'mutated';
  source.includeResourceInventory = false;
  source.includeDomainAttestation = false;
  requestedEnvelopes[0] = MemoryEnvelopeKind.DeviceTotalUsed;
  requestedEnvelopes.push(MemoryEnvelopeKind.ProcessPss);

  assert.deepEqual(normalized, {
    protocol: MEMORY_CAPTURE_PROTOCOL,
    includeResourceInventory: true,
    includeDomainAttestation: true,
    requestedEnvelopes: [MemoryEnvelopeKind.ProcessRss],
  });
});

test('compile and execute publish detached, JSON-safe memory evidence with typed provider resources',
  async () => {
    const requestedEnvelopes = [
      MemoryEnvelopeKind.ProcessRss,
      MemoryEnvelopeKind.DeviceProcessUsed,
    ];
    const options = captureOptions({
      includeResourceInventory: true,
      includeDomainAttestation: true,
      requestedEnvelopes,
    });
    const fixture = providerFixture({
      captureMemorySnapshot: (request, _captureCount, output) => {
        output.data[0] = 999;
        return backendResourceSnapshot(request);
      },
    });
    const runtime = new Runtime({ memoryCapture: options });

    // Runtime construction is the ownership boundary for capture policy.
    options.protocol = 'mutated-after-runtime-construction';
    options.includeResourceInventory = false;
    options.includeDomainAttestation = false;
    requestedEnvelopes.splice(0, requestedEnvelopes.length, MemoryEnvelopeKind.ProcessPss);

    const compiled = await compileFixture(runtime, fixture);
    const compileSnapshot = onlySnapshot(compiled.report);
    assertSnapshotIdentity(compileSnapshot, {
      stage: OperationStage.Compile,
      point: MemorySnapshotPoint.After,
      ownerKind: MemoryOwnerKind.CompiledModel,
      ownerId: compiled.report.compilationId,
    });
    assert.deepEqual(
      compileSnapshot.envelopes.map((envelope) => envelope.kind),
      [MemoryEnvelopeKind.ProcessRss, MemoryEnvelopeKind.DeviceProcessUsed],
    );
    assert.equal(compileSnapshot.resources.length, 0);
    assert.equal(compileSnapshot.resourceInventory, MemoryInventoryKind.Partial);
    assert.equal(compileSnapshot.domainAttestation.graphFingerprint,
      identitySnapshot().definitionFingerprint);

    const context = await compiled.createContext();
    const result = await context.execute({
      x: { data: Float32Array.of(4, 5), shape: [2] },
    });
    const executeSnapshot = onlySnapshot(result.report);
    assertSnapshotIdentity(executeSnapshot, {
      stage: OperationStage.Execute,
      point: MemorySnapshotPoint.After,
      ownerKind: MemoryOwnerKind.ExecutionContext,
      ownerId: context.id,
    });
    assert.equal(fixture.captureRequests.length, 1);
    assert.equal(Object.isFrozen(fixture.captureRequests[0]), true);
    assert.equal(Object.isFrozen(fixture.captureRequests[0].subject), true);
    assert.equal(fixture.captureRequests[0].protocol, BACKEND_MEMORY_SNAPSHOT_PROTOCOL);
    assert.equal(fixture.captureRequests[0].point, MemorySnapshotPoint.After);
    assert.deepEqual(fixture.captureRequests[0].subject, {
      kind: MemoryOwnerKind.ExecutionContext,
      ownerId: context.id,
    });
    assert.deepEqual(fixture.captureRequests[0].result, {
      kind: MemoryOwnerKind.Result,
      ownerId: result.id,
    });
    assert.equal(Object.isFrozen(fixture.captureRequests[0].result), true);
    assert.equal(executeSnapshot.resourceInventory, MemoryInventoryKind.Complete);
    assert.equal(executeSnapshot.resources.length, 1);
    assert.equal(executeSnapshot.resources[0].measurements[0].bytes.bytes, 64);
    assert.deepEqual(await result.output('y').read(), Float32Array.of(4, 5),
      'AFTER capture runs only after a borrowed output has been cloned into Result ownership');

    const returned = fixture.returnedMemorySnapshots[0];
    returned.resources[0].measurements[0].bytes.bytes = 999;
    returned.resources[0].addressableBytes.bytes = 999;
    returned.resources.push(returned.resources[0]);
    assert.equal(executeSnapshot.resources.length, 1);
    assert.equal(executeSnapshot.resources[0].measurements[0].bytes.bytes, 64);
    assert.equal(executeSnapshot.resources[0].addressableBytes.bytes, 64);

    assertDeepFrozen(compiled.report.memoryEvidence);
    assertDeepFrozen(result.report.memoryEvidence);
    const decoded = JSON.parse(JSON.stringify(result.report.memoryEvidence));
    assert.equal(decoded.snapshots[0].resources[0].measurements[0].bytes.bytes, 64);
    assert.equal(typeof decoded.snapshots[0].monotonicTime.nanoseconds, 'number');

    const wireEvidence = toWireMemoryEvidence(result.report.memoryEvidence);
    assert.equal(typeof wireEvidence.snapshots[0].sequence, 'bigint');
    assert.equal(
      typeof wireEvidence.snapshots[0].resources[0].measurements[0].bytes.bytes,
      'bigint',
    );
    const roundTripped = wire.MemoryEvidence.fromBinary(wireEvidence.toBinary());
    assert.equal(validateMemoryEvidence(roundTripped), roundTripped);
    assert.equal(
      roundTripped.snapshots[0].resources[0].measurements[0].bytes.bytes,
      64n,
    );

    await result.close();
    await context.close();
    await compiled.close();
    await runtime.close();
  });

test('malformed and throwing provider memory hooks degrade to an empty partial inventory',
  async () => {
    const fixture = providerFixture({
      captureMemorySnapshot: (_request, call) => {
        if (call === 1) {
          return {
            protocol: 'malformed-provider-memory-snapshot/v1',
            resources: [],
            resourceInventory: MemoryInventoryKind.Complete,
          };
        }
        throw new Error('fixture sampler failed');
      },
    });
    const runtime = new Runtime({
      memoryCapture: captureOptions({ includeResourceInventory: true }),
    });
    const compiled = await compileFixture(runtime, fixture);
    const context = await compiled.createContext();
    const malformed = await context.execute({
      x: { data: Float32Array.of(10), shape: [1] },
    });
    const throwing = await context.execute({
      x: { data: Float32Array.of(11), shape: [1] },
    });

    for (const result of [malformed, throwing]) {
      assert.equal(result.report.outcome, 'success');
      const snapshot = onlySnapshot(result.report);
      assert.deepEqual(snapshot.resources, []);
      assert.equal(snapshot.resourceInventory, MemoryInventoryKind.Partial);
    }
    assert.deepEqual(await malformed.output('y').read(), Float32Array.of(10));
    assert.deepEqual(await throwing.output('y').read(), Float32Array.of(11));
    assert.equal(fixture.captureRequests.length, 2);

    await malformed.close();
    await throwing.close();
    await context.close();
    await compiled.close();
    await runtime.close();
  });

test('compilation and execution failures publish FAILURE evidence for the affected owner',
  async () => {
    const memoryCapture = captureOptions({ includeResourceInventory: true });

    const compileFixtureProvider = providerFixture({
      name: 'compile-failure-fixture',
      failCompile: true,
    });
    const compileRuntime = new Runtime({ memoryCapture });
    let compileError;
    try {
      await compileFixture(compileRuntime, compileFixtureProvider);
    } catch (error) {
      compileError = error;
    }
    assert.equal(compileError.code, 'BACKEND_REQUIRED');
    const failedCompileSnapshot = onlySnapshot(compileError.report);
    assertSnapshotIdentity(failedCompileSnapshot, {
      stage: OperationStage.Compile,
      point: MemorySnapshotPoint.Failure,
      ownerKind: MemoryOwnerKind.Model,
      ownerId: compileError.report.definitionId,
    });
    await compileRuntime.close();

    const executeFixtureProvider = providerFixture({
      name: 'execute-failure-fixture',
      failExecute: true,
      captureMemorySnapshot: (request) => backendResourceSnapshot(request, 96),
    });
    const executeRuntime = new Runtime({ memoryCapture });
    const compiled = await compileFixture(executeRuntime, executeFixtureProvider);
    const context = await compiled.createContext();
    let executeError;
    try {
      await context.execute({ x: { data: Float32Array.of(7), shape: [1] } });
    } catch (error) {
      executeError = error;
    }
    assert.equal(executeError.code, 'EXECUTION_FAILED');
    const failedExecuteSnapshot = onlySnapshot(executeError.report);
    assertSnapshotIdentity(failedExecuteSnapshot, {
      stage: OperationStage.Execute,
      point: MemorySnapshotPoint.Failure,
      ownerKind: MemoryOwnerKind.ExecutionContext,
      ownerId: context.id,
    });
    assert.equal(failedExecuteSnapshot.resources[0].measurements[0].bytes.bytes, 96);
    assert.equal(executeFixtureProvider.captureRequests.length, 1);
    assert.equal(executeFixtureProvider.captureRequests[0].point, MemorySnapshotPoint.Failure);

    await context.close();
    await compiled.close();
    await executeRuntime.close();
  });

test('legacy allocationBytes and backendReport counters are never promoted to resources',
  async () => {
    const fixture = providerFixture({
      allocationBytes: 123_456,
      backendReport: Object.freeze({
        allocationBytes: 78_901,
        activationCapacityBytes: 45_678,
        pendingRetiredBufferCount: 3,
      }),
    });
    const runtime = new Runtime({
      memoryCapture: captureOptions({
        includeResourceInventory: true,
        requestedEnvelopes: [MemoryEnvelopeKind.ProcessPeakRss],
      }),
    });
    const compiled = await compileFixture(runtime, fixture);
    assert.equal(compiled.report.allocationBytes, 123_456);
    const compileSnapshot = onlySnapshot(compiled.report);
    assert.deepEqual(compileSnapshot.resources, []);
    assert.equal(compileSnapshot.resourceInventory, MemoryInventoryKind.Partial);

    const context = await compiled.createContext();
    const result = await context.execute({
      x: { data: Float32Array.of(8), shape: [1] },
    });
    assert.deepEqual(result.report.backendReport, {
      allocationBytes: 78_901,
      activationCapacityBytes: 45_678,
      pendingRetiredBufferCount: 3,
    });
    const executeSnapshot = onlySnapshot(result.report);
    assert.deepEqual(executeSnapshot.resources, []);
    assert.equal(executeSnapshot.resourceInventory, MemoryInventoryKind.Partial);
    const serializedEvidence = JSON.stringify(result.report.memoryEvidence);
    assert.equal(serializedEvidence.includes('allocationBytes'), false);
    assert.equal(serializedEvidence.includes('activationCapacityBytes'), false);

    await result.close();
    await context.close();
    await compiled.close();
    await runtime.close();
  });

test('an opt-in memory collector never inflates the execution time it reports', async () => {
  const burnMilliseconds = 40;
  const fixture = providerFixture({
    captureMemorySnapshot: (request) => {
      const until = performance.now() + burnMilliseconds;
      while (performance.now() < until) { /* deliberate collector cost */ }
      return backendResourceSnapshot(request);
    },
  });
  const runtime = new Runtime({
    memoryCapture: captureOptions({ includeResourceInventory: true }),
  });
  const compiled = await compileFixture(runtime, fixture);
  const context = await compiled.createContext();

  const started = performance.now();
  const result = await context.execute({ x: { data: Float32Array.of(8), shape: [1] } });
  const wallClockMs = performance.now() - started;

  assert.ok(wallClockMs >= burnMilliseconds,
    `the collector must actually have run, but the call took ${wallClockMs} ms`);
  assert.equal(onlySnapshot(result.report).resources.length, 1,
    'the collector result must still reach the report');
  // executionTimeMs describes the execution, not the diagnostic that observes
  // it. Charging collector cost to it would make an opt-in report field change
  // a published performance number.
  assert.ok(result.report.executionTimeMs < burnMilliseconds,
    `executionTimeMs ${result.report.executionTimeMs} absorbed the ` +
    `${burnMilliseconds} ms collector`);

  await result.close();
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('the browser managed-heap fallback is a distinct estimated collector', () => {
  const requested = [MemoryEnvelopeKind.ProcessManagedHeapUsed];
  const nodeEnvelope = sampleRuntimeMemoryEnvelopes(requested)[0];
  const savedProcess = Object.getOwnPropertyDescriptor(globalThis, 'process');
  const savedPerformance = Object.getOwnPropertyDescriptor(globalThis, 'performance');
  let browserEnvelope;
  try {
    // Hide the Node counter so the browser fallback is the only collector left.
    delete globalThis.process;
    Object.defineProperty(globalThis, 'performance', {
      configurable: true,
      value: { now: () => 0, memory: { usedJSHeapSize: 4_194_304 } },
    });
    browserEnvelope = sampleRuntimeMemoryEnvelopes(requested)[0];
  } finally {
    Object.defineProperty(globalThis, 'process', savedProcess);
    Object.defineProperty(globalThis, 'performance', savedPerformance);
  }

  assert.equal(nodeEnvelope.valueRelation, MemoryValueRelation.Exact);
  assert.equal(nodeEnvelope.sampler, 'typescript-runtime-memory/v1');

  assert.equal(browserEnvelope.bytes.bytes, 4_194_304);
  assert.equal(browserEnvelope.source, MemoryEvidenceSource.RuntimeCounter);
  // Browsers quantize performance.memory rather than exposing the live heap,
  // so the fallback must not claim EXACT and must not borrow the identity of
  // the counter it stands in for.
  assert.equal(browserEnvelope.valueRelation, MemoryValueRelation.Estimated);
  assert.notEqual(browserEnvelope.sampler, nodeEnvelope.sampler);
});
