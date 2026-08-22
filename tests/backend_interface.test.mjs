import test from 'node:test';
import assert from 'node:assert/strict';

import {
  Model,
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  VolvoxAI,
  createBackendProviderBatchContract,
  createBackendProviderPreparedBatchRoute,
  createBackendProviderCapabilities,
  parseGraphDocument,
  requireHostExecutionInputs,
} from '../ts/index.js';
import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { BackendEngine, assertBuiltInEngine } from '../ts/backends/BackendEngine.js';
import {
  BuiltInBackendProvider,
  assertBackendProvider,
  assertProviderCompiledModel,
  assertProviderPreparedBatchRoute,
  createBackendCompileInput,
} from '../ts/backends/BackendProvider.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { GraphExecutor } from '../ts/backends/GraphExecutor.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';
import { WebGPUEngine } from '../ts/backends/WebGPUEngine.js';
import { InvariantResourceStore } from '../ts/backends/InvariantResources.js';

class RecordingBackend extends BackendEngine {
  constructor(name = 'recording', capabilities = {}) {
    super(name, capabilities);
    this.calls = [];
    this.graph = null;
  }

  allocateGraph(graph) {
    this.resetDecodeCache();
    this.graph = graph;
    return this;
  }

  async execute(inputs, options = {}) {
    this._beginDecodeExecution(options);
    this.calls.push({ inputs, options });
    return { out: Float32Array.of(this.calls.length) };
  }
}

function emptyGraph() {
  return { tensors: new Map(), nodes: [], outputNames: [] };
}

function internalReluGraph(adapters = null) {
  const input = {
    name: 'x', shape: [1], dtype: 'float32', sizeBytes: 4,
    isInput: true, buffer: new Float32Array(1),
  };
  const output = {
    name: 'y', shape: [1], dtype: 'float32', sizeBytes: 4,
    buffer: new Float32Array(1),
  };
  return {
    tensors: new Map([[input.name, input], [output.name, output]]),
    nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input }, outputs: { out: output } }],
    outputNames: [output.name],
    topologyRevision: 0,
    adapters,
  };
}

function identityGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('x', [1]);
  const output = graph.addOp('Identity', { input }, {
    out: { name: 'y', shape: [1] },
  }).out;
  graph.setOutputs(output);
  return graph;
}

function logicalIdentitySnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1] } },
    nodes: [{
      id: 'identity', opType: 'Identity', inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1] } }, params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function genericByteAddGraph({ quantized = true, dtype = 'int8' } = {}) {
  const graph = new RuntimeGraph();
  const quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };
  const options = quantized ? { quantization } : {};
  const a = graph.addInput('a', [4], dtype, options);
  const b = graph.addInput('b', [4], dtype, options);
  const { out } = graph.addOp('Add', { a, b }, {
    out: {
      name: 'out', shape: [4], dtype,
      ...(quantized ? { quantization } : {}),
    },
  });
  graph.setOutputs(out);
  return graph;
}

function hostOutput(value) {
  return Object.freeze({
    name: 'y',
    shape: Object.freeze([1]),
    dtype: 'float32',
    location: 'host',
    ownership: 'borrowed',
    data: Float32Array.of(value),
  });
}

test('built-in engine lifecycle stays private and structurally consistent', () => {
  const cpu = new CPUEngine();
  assert.equal(cpu.backendName, 'cpu-js');
  assert.deepEqual(cpu.capabilities, {
    incrementalExecution: true, incrementalRows: true,
    sequenceMajorRows: false, outputLocation: 'host',
  });
  assert.equal(assertBuiltInEngine(cpu), cpu);
  assert.equal(typeof cpu.allocateGraph, 'function');
  assert.equal(typeof cpu.execute, 'function');
  assert.equal(typeof cpu.createDecodeSession, 'function');
  assert.equal(typeof GraphExecutor.prototype.createDecodeSession, 'function');
  assert.throws(
    () => assertBuiltInEngine({ allocateGraph() {}, execute() {} }, 'Incomplete backend'),
    /built-in engine lifecycle/,
  );
  assert.throws(
    () => assertBuiltInEngine({
      backendName: 'unsafe-incremental',
      capabilities: Object.freeze({
        incrementalExecution: false,
        incrementalRows: true,
        outputLocation: 'host',
      }),
      allocateGraph() {},
      execute() {},
      createDecodeSession() {},
    }, 'Unsafe incremental backend'),
    /built-in engine lifecycle/,
  );
});

test('non-CPU built-ins explicitly reject provider bounded domains before preparation', async () => {
  const events = { prepare: 0, allocations: [], disposed: 0 };
  class PreparedBackend extends RecordingBackend {
    constructor() { super('prepared'); }
    prepareGraph(graph) {
      events.prepare++;
      return Object.freeze({ nodeCount: graph.nodes.length });
    }
    fork() { return new PreparedBackend(); }
    allocateGraph(graph, preparedGraph) {
      events.allocations.push({ graph, preparedGraph, engine: this });
      return super.allocateGraph(graph);
    }
    dispose() { events.disposed++; }
  }

  const provider = new BuiltInBackendProvider(new PreparedBackend());
  assert.equal(provider.capabilities.dynamicShapeDomain.support, 'unsupported');
  assert.deepEqual(
    Reflect.ownKeys(provider.capabilities.dynamicShapeDomain).sort(),
    ['proofProtocol', 'resourceProtocol', 'support'].sort(),
  );
  assert.equal(Object.isFrozen(provider.capabilities.dynamicShapeDomain), true);
  await assert.rejects(
    provider.compile(createBackendCompileInput(logicalIdentitySnapshot()), {
      operatorFallback: 'forbid',
    }),
    (error) => error.code === 'BACKEND_UNSUPPORTED',
  );
  assert.equal(events.prepare, 0);
  assert.equal(events.allocations.length, 0);
  await provider.close();
  assert.equal(events.disposed, 1);
});

test('provider compilation evidence cannot spoof independent-batch proof identity', () => {
  const input = createBackendCompileInput(logicalIdentitySnapshot());
  const invariantResources = new InvariantResourceStore(() => 0);
  const compiled = {
    backendName: 'hostile-proof',
    invariantResources,
    batchContract: createBackendProviderBatchContract('unsupported'),
    compilationEvidence: Object.freeze({
      device: null,
      allocationBytes: 0,
      shapeDomain: Object.freeze({
        proofProtocol: 'canonical-symbolic-domain-proof/v1',
        resourceProtocol: 'bounded-resource-maxima/v1',
        support: 'full',
        graphFingerprint: input.graphFingerprint,
        proof: input.shapeDomainProof,
        maximumTensorBytes: 4,
        maximumResidentBytes: 8,
        resourceLimitBytes: 16,
      }),
      batchSemantics: Object.freeze({
        protocol: 'typed-independent-batch-proof/v1',
        supported: false,
        batchSymbol: null,
        coveredNodes: 0,
        graphFingerprint: `${input.graphFingerprint}:spoofed`,
        reason: 'hostile evidence',
        failedNode: null,
        failedTensor: null,
      }),
    }),
    prepareBatchRoute() {},
    createContext() {},
    close() {},
  };

  assert.throws(
    () => assertProviderCompiledModel(compiled, 'hostile-proof', input),
    (error) => error.code === 'ABI_UNSUPPORTED' && /compilation evidence/.test(error.message),
  );

  const missing = {
    ...compiled,
    backendName: 'missing-proof',
    batchContract: createBackendProviderBatchContract('single-invocation', {
      independentBatch: 'compiler-proved/v1',
    }),
    compilationEvidence: Object.freeze({
      device: null,
      allocationBytes: 0,
      shapeDomain: compiled.compilationEvidence.shapeDomain,
    }),
  };
  assert.throws(
    () => assertProviderCompiledModel(missing, 'missing-proof', input),
    (error) => error.code === 'ABI_UNSUPPORTED' && /compilation evidence/.test(error.message),
  );

  const copied = {
    ...missing,
    backendName: 'copied-proof',
    compilationEvidence: Object.freeze({
      ...missing.compilationEvidence,
      batchSemantics: Object.freeze({ ...input.batchSemantics }),
    }),
  };
  assert.throws(
    () => assertProviderCompiledModel(copied, 'copied-proof', input),
    (error) => error.code === 'ABI_UNSUPPORTED' && /compilation evidence/.test(error.message),
  );
});

test('WASM prepared schedules are frozen, pointer-free, and reject unknown operators', () => {
  const engine = new WasmEngine({
    instance: { exports: { memory: new WebAssembly.Memory({ initial: 1 }) } },
  });
  const firstGraph = identityGraph();
  const invariant = engine.prepareInvariantSchedule({
    nodes: firstGraph.nodes,
    tensorCount: firstGraph.tensors.size,
    outputNames: firstGraph.outputNames,
  });
  const prepared = engine.prepareGraph(firstGraph, invariant);

  assert.equal(Object.isFrozen(prepared), true);
  assert.equal(Object.isFrozen(prepared.schedule), true);
  assert.equal(Object.isFrozen(prepared.schedule[0]), true);
  assert.deepEqual(prepared.schedule.map(({ nodeIndex, opType, kernelRoute }) => ({
    nodeIndex, opType, kernelRoute,
  })), [{ nodeIndex: 0, opType: 'Identity', kernelRoute: 'shape-copy' }]);
  assert.equal(prepared.schedule[0].node, undefined);
  assert.equal(prepared.schedule[0].tensor, undefined);
  assert.throws(() => prepared.schedule.push({}), TypeError);

  const secondGraph = new RuntimeGraph();
  const secondInput = secondGraph.addInput('x', [7]);
  const secondOutput = secondGraph.addOp('Identity', { input: secondInput }, {
    out: { name: 'y', shape: [7] },
  }).out;
  secondGraph.setOutputs(secondOutput);
  const rebound = engine.prepareGraph(secondGraph, invariant);
  assert.strictEqual(rebound.schedule, prepared.schedule,
    'shape variants must retain the compiled-model schedule object');
  assert.strictEqual(rebound.inputPreflights, prepared.inputPreflights);

  const unsupported = new RuntimeGraph();
  const input = unsupported.addInput('x', [1]);
  const output = unsupported.addOp('UnknownKernel', { input }, {
    out: { name: 'y', shape: [1] },
  }).out;
  unsupported.setOutputs(output);
  assert.throws(() => engine.prepareGraph(unsupported), /operator 'UnknownKernel' is unsupported/);
  assert.throws(
    () => engine.prepareGraph(unsupported, invariant),
    /invariant prepared schedule does not match/,
  );
});

test('provider capability validation uses typed initialization failures', () => {
  for (const options of [
    { operatorFallback: 'sometimes' },
    { outputLocation: 'remote' },
  ]) {
    assert.throws(() => createBackendProviderCapabilities(options), (error) => {
      assert.equal(error.code, 'INVALID_ARGUMENT');
      assert.equal(error.phase, 'initialization');
      return true;
    });
  }
});

test('provider dense-batch attestation matches device residency exactly', () => {
  assert.deepEqual(createBackendProviderBatchContract('single-invocation'), {
    protocol: 'dense-public-batch/v1',
    densePublicBatch: 'single-invocation',
    independentBatch: 'unsupported',
    deviceResident: false,
    hostFallback: 'not-applicable',
  });
  assert.deepEqual(createBackendProviderBatchContract('single-invocation', {
    independentBatch: 'compiler-proved/v1',
    deviceResident: true,
    hostFallback: 'forbidden',
  }), {
    protocol: 'dense-public-batch/v1',
    densePublicBatch: 'single-invocation',
    independentBatch: 'compiler-proved/v1',
    deviceResident: true,
    hostFallback: 'forbidden',
  });
  for (const options of [
    { deviceResident: true, hostFallback: 'not-applicable' },
    { deviceResident: false, hostFallback: 'possible' },
    { independentBatch: 'invalid' },
  ]) {
    assert.throws(
      () => createBackendProviderBatchContract('single-invocation', options),
      (error) => error.code === 'INVALID_ARGUMENT' && error.phase === 'initialization',
    );
  }
});

test('prepared batch routes retain only frozen provider-owned opaque identities', () => {
  const resourceDomain = Object.freeze({});
  const compatibilityToken = Object.freeze({});
  const deviceEpoch = Object.freeze({});
  const route = createBackendProviderPreparedBatchRoute(
    resourceDomain, compatibilityToken, deviceEpoch,
  );
  const asserted = assertProviderPreparedBatchRoute(route, 'fixture');
  assert.notEqual(asserted, route);
  assert.equal(asserted.resourceDomain, resourceDomain);
  assert.equal(route.resourceDomain, resourceDomain);
  assert.equal(route.compatibilityToken, compatibilityToken);
  assert.equal(route.deviceEpoch, deviceEpoch);
  assert.equal(Object.isFrozen(route), true);

  const structural = createBackendProviderPreparedBatchRoute(
    resourceDomain, 'provider-route:v1', deviceEpoch,
  );
  assert.equal(
    assertProviderPreparedBatchRoute(structural, 'fixture').compatibilityToken,
    'provider-route:v1',
  );

  assert.throws(
    () => createBackendProviderPreparedBatchRoute({}, compatibilityToken, deviceEpoch),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.throws(
    () => createBackendProviderPreparedBatchRoute(resourceDomain, '', deviceEpoch),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.throws(
    () => assertProviderPreparedBatchRoute(Object.freeze({
      resourceDomain,
      compatibilityToken: {},
      deviceEpoch,
    }), 'fixture'),
    (error) => error.code === 'ABI_UNSUPPORTED',
  );
});

test('provider composition accepts only the current exact contract before compile', () => {
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: 'versioned-fixture',
    capabilities: createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'host',
      dynamicShapeDomain: 'full',
    }),
    async compile() { throw new Error('compile must not be reached'); },
    async close() {},
  };
  assert.equal(assertBackendProvider(provider), provider);
  const staleCapabilities = Object.freeze({
    ...provider.capabilities,
    legacyBatching: true,
  });
  assert.throws(
    () => assertBackendProvider({
      ...provider,
      // The current numeric discriminator alone does not make an older
      // capability layout compatible.
      providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
      capabilities: staleCapabilities,
    }),
    (error) => error.code === 'ABI_UNSUPPORTED' &&
      error.phase === 'initialization' &&
      error.message.includes('current exact provider contract') &&
      error.message.includes('historical layouts are unsupported'),
  );
  const malformedCapability = Object.freeze({
    ...provider.capabilities.dynamicShapeDomain,
    unknownMember: true,
  });
  assert.throws(
    () => assertBackendProvider({
      ...provider,
      capabilities: Object.freeze({
        ...provider.capabilities,
        dynamicShapeDomain: malformedCapability,
      }),
    }),
    (error) => error.code === 'ABI_UNSUPPORTED',
  );
  for (const providerVersion of [0, 0xffff_ffff]) {
    assert.throws(
      () => assertBackendProvider({ ...provider, providerVersion }),
      (error) => error.code === 'ABI_UNSUPPORTED' &&
        error.phase === 'initialization' &&
        error.message.includes('current exact provider contract') &&
        error.message.includes(
          `VOLVOXAI_BACKEND_PROVIDER_VERSION ${VOLVOXAI_BACKEND_PROVIDER_VERSION}`,
        ),
    );
  }
});

test('built-in allocation rejects generic arithmetic on quantized byte tensors', async () => {
  const error = /unsupported generic operator/;
  assert.throws(() => new CPUEngine().allocateGraph(genericByteAddGraph()), error);
  await assert.rejects(
    new WebGPUEngine({}).allocateGraph(genericByteAddGraph()),
    error,
  );
  const executor = new GraphExecutor({}, genericByteAddGraph(), { shaderLibrary: {} });
  await assert.rejects(executor.compile(), error);
  executor.dispose();
});

test('generic arithmetic cannot reinterpret unannotated byte tensors as F32', async () => {
  const error = /unsupported generic operator/;
  const graph = genericByteAddGraph({ quantized: false, dtype: 'uint8' });
  assert.throws(() => new CPUEngine().allocateGraph(graph), error);
  await assert.rejects(new WebGPUEngine({}).allocateGraph(graph), error);

  const executor = new GraphExecutor({}, graph, { shaderLibrary: {} });
  await assert.rejects(executor.compile(), error);
  executor.dispose();
});

test('DecodeSession maps seed and row steps onto internal execution options', async () => {
  const engine = new RecordingBackend('row-device', {
    incrementalExecution: true,
    incrementalRows: true,
  });
  engine.allocateGraph(emptyGraph());
  const decode = engine.createDecodeSession({ changedInputs: ['tokens', 'mask'] });
  assert.equal(decode.mode, 'incremental-row');

  await decode.seed({ tokens: Int32Array.of(1), mask: Int32Array.of(1), image: Float32Array.of(2) });
  await decode.step({ tokens: Int32Array.of(2), mask: Int32Array.of(1), image: Float32Array.of(2) }, {
    position: 1,
  });
  assert.deepEqual(engine.calls[0].options, {
    incremental: true,
    incrementalReset: true,
    changedInputs: ['tokens', 'mask', 'image'],
  });
  assert.deepEqual(engine.calls[1].options, {
    incremental: true,
    changedInputs: ['tokens', 'mask'],
    incrementalRowPosition: 1,
  });
  assert.equal(decode.lastExecutionMode, 'incremental-row');

  await decode.reset();
  await assert.rejects(decode.step({ tokens: Int32Array.of(3) }), /requires a successful seed/);
  await decode.close();
  await assert.rejects(decode.seed({ tokens: Int32Array.of(1) }), /closed/);
});

test('DecodeSession falls back to forward execution and detects a replaced cache', async () => {
  const ordinary = new RecordingBackend('ordinary');
  ordinary.allocateGraph(emptyGraph());
  const fallback = ordinary.createDecodeSession();
  assert.equal(fallback.mode, 'ordinary-forward');
  await fallback.seed({ x: Float32Array.of(1) }, { incremental: true, changedInputs: ['ignored'] });
  await fallback.step({ x: Float32Array.of(2) }, { position: 4, changedInputs: ['ignored'] });
  assert.deepEqual(ordinary.calls.map(({ options }) => options), [{}, {}]);
  assert.throws(
    () => ordinary.createDecodeSession({ requireIncremental: true }),
    /does not support incremental execution/,
  );

  const shared = new RecordingBackend('shared', { incrementalExecution: true });
  shared.allocateGraph(emptyGraph());
  const first = shared.createDecodeSession();
  const second = shared.createDecodeSession();
  await first.seed({ x: Float32Array.of(1) });
  await second.seed({ x: Float32Array.of(2) });
  await assert.rejects(first.step({ x: Float32Array.of(3) }), /cache was reset or replaced/);
  await second.step({ x: Float32Array.of(4) });
});

test('closing stale, unseeded, and concurrent sessions preserves one cache owner', async () => {
  const shared = new RecordingBackend('shared', { incrementalExecution: true });
  shared.allocateGraph(emptyGraph());
  const stale = shared.createDecodeSession();
  const active = shared.createDecodeSession();
  await stale.seed({ x: Float32Array.of(1) });
  await active.seed({ x: Float32Array.of(2) });
  await stale.close();
  await active.step({ x: Float32Array.of(3) });

  const unseeded = shared.createDecodeSession();
  await unseeded.close();
  await active.step({ x: Float32Array.of(4) });

  let releaseFirst;
  let signalFirstStarted;
  const firstStarted = new Promise((resolve) => { signalFirstStarted = resolve; });
  class DeferredBackend extends RecordingBackend {
    constructor() {
      super('deferred', { incrementalExecution: true });
      this.activeExecutions = 0;
      this.maxActiveExecutions = 0;
    }

    async execute(inputs, options = {}) {
      this.calls.push({ inputs, options });
      this.activeExecutions++;
      this.maxActiveExecutions = Math.max(this.maxActiveExecutions, this.activeExecutions);
      if (this.calls.length === 1) {
        signalFirstStarted();
        await new Promise((resolve) => { releaseFirst = resolve; });
      }
      this.activeExecutions--;
      return { out: Float32Array.of(this.calls.length) };
    }
  }

  const engine = new DeferredBackend();
  engine.allocateGraph(emptyGraph());
  const first = engine.createDecodeSession();
  const second = engine.createDecodeSession();
  const firstSeed = first.seed({ x: Float32Array.of(1) });
  await firstStarted;
  const secondSeed = second.seed({ x: Float32Array.of(2) });
  await Promise.resolve();
  assert.equal(engine.calls.length, 1);
  releaseFirst();
  await Promise.all([firstSeed, secondSeed]);
  assert.equal(engine.maxActiveExecutions, 1);
  assert.equal(engine.calls.length, 2);
  await assert.rejects(first.step({ x: Float32Array.of(3) }), /cache was reset or replaced/);
  await second.step({ x: Float32Array.of(4) });
});

test('direct execution replaces active decode-cache ownership', async () => {
  const engine = new RecordingBackend('shared', { incrementalExecution: true });
  engine.allocateGraph(emptyGraph());
  const decode = engine.createDecodeSession();
  await decode.seed({ x: Float32Array.of(1) });

  await engine.execute({ x: Float32Array.of(2) }, {
    incremental: true,
    incrementalReset: true,
    changedInputs: ['x'],
  });
  await assert.rejects(
    decode.step({ x: Float32Array.of(3) }),
    /cache was reset or replaced/,
  );
});

test('DecodeSession rejects unknown inputs and dynamically unavailable caching', async () => {
  const graph = internalReluGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);

  const unknown = engine.createDecodeSession({ changedInputs: ['x'], rowMode: 'disabled' });
  await unknown.seed({ x: Float32Array.of(-1) });
  await assert.rejects(
    unknown.step({ x: Float32Array.of(1), typo: Float32Array.of(1) }, {
      changedInputs: ['typo'],
    }),
    /'typo' is not a graph input/,
  );

  const training = engine.createDecodeSession({ requireIncremental: true });
  await assert.rejects(
    training.seed({ x: Float32Array.of(1) }, { training: {} }),
    /does not accept training or Dropout RNG options/,
  );
});

test('WebGPU context forks share only device caches and release them at the final owner', async () => {
  const outputBuffer = { label: 'output' };
  const events = [];
  class StubExecutor extends BackendEngine {
    constructor(graph) {
      super('webgpu', { incrementalExecution: true, outputLocation: 'device' });
      this.graph = graph;
      this.gpuBuffers = new Map([['out', outputBuffer]]);
      this.pipelines = [];
    }
    async compile() { this.resetDecodeCache(); events.push('compile'); }
    async execute(inputs, options = {}) {
      this._beginDecodeExecution(options);
      events.push({ inputs, options });
      return outputBuffer;
    }
    async readBuffer(buffer, sizeBytes, dtype) {
      events.push({ buffer, sizeBytes, dtype });
      return Int32Array.of(7);
    }
    async readBufferRange(buffer, byteOffset, sizeBytes, dtype) {
      events.push({ buffer, byteOffset, sizeBytes, dtype });
      return Int32Array.of(8);
    }
    async executeDeviceFeedbackDecode(inputs, options) {
      events.push({ deviceFeedback: true, inputs, options });
      return outputBuffer;
    }
    dispose() { this.resetDecodeCache(); events.push('dispose'); }
  }
  class StubWebGPUEngine extends WebGPUEngine {
    _createExecutor(graph) {
      return new StubExecutor(graph);
    }
  }

  const engine = new StubWebGPUEngine({ label: 'device' }, {
    adapterInfo: { vendor: 'TestVendor', device: 'Test GPU' },
  });
  const graph = emptyGraph();
  assert.equal(await engine.allocateGraph(graph), engine);
  assert.equal(engine.graph, graph);
  assert.equal(engine.gpuBuffers.get('out'), outputBuffer);
  assert.equal(await engine.execute({ x: Float32Array.of(1) }), outputBuffer);
  assert.deepEqual(await engine.readBuffer(outputBuffer, 4, 'int32'), Int32Array.of(7));
  assert.deepEqual(await engine.readBufferRange(outputBuffer, 4, 4, 'int32'), Int32Array.of(8));
  const generationBeforeFeedback = engine.decodeCacheGeneration;
  assert.equal(await engine.executeDeviceFeedbackDecode({ ids: Int32Array.of(0) }, {
    tokenInput: 'ids', keepInput: 'keep', output: 'out', tokenCount: 1,
  }), outputBuffer);
  assert.ok(engine.decodeCacheGeneration > generationBeforeFeedback);
  assert.equal(engine.capabilities.outputLocation, 'device');
  assert.deepEqual(engine.adapterInfo, { vendor: 'TestVendor', device: 'Test GPU' });

  const peer = engine.fork();
  assert.notEqual(peer, engine);
  assert.equal(peer.device, engine.device);
  assert.equal(peer.deviceState, engine.deviceState);
  assert.equal(engine.deviceState.referenceCount, 2);
  assert.equal(peer.executor, null);
  engine.deviceState.rejectedSpecializedShaders.add('retained-test-shader');

  const decode = engine.createDecodeSession();
  await decode.seed({ x: Float32Array.of(1) });
  await engine.executor.execute({ x: Float32Array.of(2) });
  await assert.rejects(
    decode.step({ x: Float32Array.of(3) }),
    /cache was reset or replaced/,
  );
  await decode.seed({ x: Float32Array.of(1) });
  assert.equal(engine.prepareForTraining, undefined,
    'inference WebGPU engines expose no training preparation');
  await decode.step({ x: Float32Array.of(2) });

  engine.dispose();
  assert.equal(peer.deviceState.referenceCount, 1);
  assert.equal(peer.deviceState.rejectedSpecializedShaders.has('retained-test-shader'), true);
  peer.dispose();
  assert.equal(peer.deviceState.referenceCount, 0);
  assert.equal(peer.deviceState.rejectedSpecializedShaders.size, 0);
  assert.equal(events[0], 'compile');
});

test('runtime-local provider composition covers logical compile, contexts, decode, and results', async () => {
  const name = 'interface-fixture';
  const events = [];
  let nextContext = 0;
  const resourceDomain = Object.freeze({});
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: name,
    capabilities: createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'host',
      dynamicShapeDomain: 'full',
    }),
    async compile(input, options) {
      events.push(['compile', input.outputNames, options]);
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
          device: null,
          allocationBytes: 0,
          operatorFallbackUsed: false,
          offendingNode: null,
          shapeDomain: Object.freeze({
            proofProtocol: 'canonical-symbolic-domain-proof/v1',
            resourceProtocol: 'bounded-resource-maxima/v1',
            support: 'full',
            graphFingerprint: input.graphFingerprint,
            proof: input.shapeDomainProof,
            maximumTensorBytes: 4,
            maximumResidentBytes: 8,
            resourceLimitBytes: 1024,
          }),
        }),
        async createContext() {
          const contextId = ++nextContext;
          let closed = false;
          const execute = async (request) => {
            assert.equal(closed, false);
            events.push(['execute', contextId, request.options]);
            const inputs = requireHostExecutionInputs(request, name);
            return Object.freeze({
              outputs: Object.freeze([hostOutput(inputs.x.data[0] + contextId)]),
              backendReport: Object.freeze({ contextId }),
            });
          };
          return {
            backendName: name,
            execute,
            decodeSeed: execute,
            decodeStep: execute,
            async decodeReset() { events.push(['decode-reset', contextId]); },
            async close() { closed = true; events.push(['context-close', contextId]); },
          };
        },
        async close() { events.push(['compiled-close']); },
      };
    },
    async close() { events.push(['provider-close']); },
  };
  const runtime = await VolvoxAI.createRuntime({
    backends: [name],
    providers: {
      [name]: async ({ name: requested }) => {
        assert.equal(requested, name);
        return provider;
      },
    },
  });
  assert.deepEqual(runtime.listBackends(), [name]);
  const compiled = await runtime.compile(logicalIdentitySnapshot(), {
    backend: { mode: 'require', backend: name, operatorFallback: 'forbid' },
  });
  assert.equal(compiled.backend, name);
  assert.equal(compiled.report.selectedBackend, name);

  const firstContext = await compiled.createContext();
  const secondContext = await compiled.createContext();
  const first = await firstContext.execute({ x: { data: Float32Array.of(2), shape: [1] } });
  const second = await secondContext.decode.seed({
    x: { data: Float32Array.of(5), shape: [1] },
  });
  assert.deepEqual(await first.output('y').read(), Float32Array.of(3));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(7));
  assert.deepEqual(first.report.backendReport, { contextId: 1 });
  await secondContext.decode.reset();

  await firstContext.close();
  assert.deepEqual(await first.output('y').read(), Float32Array.of(3));
  await secondContext.close();
  await compiled.close();
  await runtime.close();
  assert.deepEqual(events.slice(-4), [
    ['context-close', 1],
    ['context-close', 2],
    ['compiled-close'],
    ['provider-close'],
  ]);
  await Promise.all([first.close(), second.close()]);
});

test('inference entry exposes handles and the provider contract, not engine facades', async () => {
  const api = await import('../ts/index.js');
  for (const hidden of [
    'BackendEngine',
    'DecodeSession',
    'CPUEngine',
    'WasmEngine',
    'WebGPUEngine',
    'GraphExecutor',
  ]) {
    assert.equal(hidden in api, false, `${hidden} must remain internal`);
  }
  assert.equal(typeof VolvoxAI, 'object');
  assert.equal(Object.isFrozen(VolvoxAI), true);
  assert.deepEqual(Object.keys(VolvoxAI).sort(), [
    'createRuntime',
  ]);
});
