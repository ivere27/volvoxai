import test from 'node:test';
import assert from 'node:assert/strict';

import {
  Graph,
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  VolvoxAI,
  createBackendProviderCapabilities,
} from '../ts/index.js';
import { BackendEngine, assertBuiltInEngine } from '../ts/backends/BackendEngine.js';
import { BuiltInBackendProvider } from '../ts/backends/BackendProvider.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { GraphExecutor } from '../ts/backends/GraphExecutor.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';
import { WebGPUEngine } from '../ts/backends/WebGPUEngine.js';
import { WebNNEngine } from '../ts/backends/WebNNEngine.js';
import { ModelSnapshot } from '../ts/core/ModelSnapshot.js';

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
  const graph = new Graph();
  const input = graph.addInput('x', [1]);
  const output = graph.addOp('Identity', { input }, {
    out: { name: 'y', shape: [1] },
  }).out;
  graph.setOutputs(output);
  return graph;
}

function genericByteAddGraph({ quantized = true, dtype = 'int8' } = {}) {
  const graph = new Graph();
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
    data: Float32Array.of(value),
  });
}

test('built-in engine lifecycle stays private and structurally consistent', () => {
  const cpu = new CPUEngine();
  const webnn = new WebNNEngine({});
  assert.equal(cpu.backendName, 'cpu');
  assert.deepEqual(cpu.capabilities, {
    incrementalExecution: true, incrementalRows: true, outputLocation: 'host',
  });
  assert.equal(webnn.backendName, 'webnn');
  assert.deepEqual(webnn.capabilities, {
    incrementalExecution: false, incrementalRows: false, outputLocation: 'host',
  });
  for (const engine of [cpu, webnn]) {
    assert.equal(assertBuiltInEngine(engine), engine);
    assert.equal(typeof engine.allocateGraph, 'function');
    assert.equal(typeof engine.execute, 'function');
    assert.equal(typeof engine.createDecodeSession, 'function');
  }
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

test('built-in CompiledModel owns one immutable prepared graph across contexts', async () => {
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
  const compiled = await provider.compile(ModelSnapshot.capture(identityGraph()), {
    operatorFallback: 'forbid',
  });
  const first = await compiled.createContext();
  const second = await compiled.createContext();

  assert.equal(events.prepare, 1);
  assert.equal(events.allocations.length, 2);
  assert.equal(events.allocations[0].preparedGraph, events.allocations[1].preparedGraph);
  assert.equal(Object.isFrozen(events.allocations[0].preparedGraph), true);
  assert.notEqual(events.allocations[0].graph, events.allocations[1].graph,
    'contexts retain private tensor graphs while sharing only the prepared blueprint');
  assert.notEqual(events.allocations[0].engine, events.allocations[1].engine);

  await first.close();
  await second.close();
  await compiled.close();
  await provider.close();
  assert.equal(events.disposed, 3);
});

test('WASM prepared schedules are frozen, pointer-free, and reject unknown operators', () => {
  const engine = new WasmEngine({
    instance: { exports: { memory: new WebAssembly.Memory({ initial: 1 }) } },
  });
  const prepared = engine.prepareGraph(identityGraph());

  assert.equal(Object.isFrozen(prepared), true);
  assert.equal(Object.isFrozen(prepared.schedule), true);
  assert.equal(Object.isFrozen(prepared.schedule[0]), true);
  assert.deepEqual(prepared.schedule.map(({ nodeIndex, opType, kernelRoute }) => ({
    nodeIndex, opType, kernelRoute,
  })), [{ nodeIndex: 0, opType: 'Identity', kernelRoute: 'shape-copy' }]);
  assert.equal(prepared.schedule[0].node, undefined);
  assert.equal(prepared.schedule[0].tensor, undefined);
  assert.throws(() => prepared.schedule.push({}), TypeError);

  const unsupported = new Graph();
  const input = unsupported.addInput('x', [1]);
  const output = unsupported.addOp('UnknownKernel', { input }, {
    out: { name: 'y', shape: [1] },
  }).out;
  unsupported.setOutputs(output);
  assert.throws(() => engine.prepareGraph(unsupported), /operator 'UnknownKernel' is unsupported/);
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

test('WebNN uses only the current dataType/shape descriptor contract', async () => {
  const previousBuilder = globalThis.MLGraphBuilder;
  const operandDescriptors = [];
  const tensorDescriptors = [];
  class CurrentWebNNBuilder {
    input(name, descriptor) {
      operandDescriptors.push(descriptor);
      return { name, dataType: descriptor.dataType, shape: descriptor.shape };
    }
    relu(input) { return { ...input, name: 'y' }; }
    async build(outputs) { return outputs; }
  }
  globalThis.MLGraphBuilder = CurrentWebNNBuilder;
  const context = {
    async createTensor(descriptor) {
      tensorDescriptors.push(descriptor);
      return { descriptor, destroy() {} };
    },
    writeTensor() {},
    dispatch() {},
    async readTensor() { return Float32Array.of(3).buffer; },
  };
  try {
    const graph = new Graph();
    const input = graph.addInput('x', [1]);
    const output = graph.addOp('ReLU', { input }, {
      out: { name: 'y', shape: [1] },
    }).out;
    graph.setOutputs(output);
    const engine = new WebNNEngine(context);
    await engine.allocateGraph(graph);
    assert.deepEqual(operandDescriptors, [
      { dataType: 'float32', shape: [1] },
    ]);
    const result = await engine.execute({ x: Float32Array.of(3) });
    assert.deepEqual(result.y, Float32Array.of(3));
    assert.deepEqual(tensorDescriptors, [
      { dataType: 'float32', shape: [1], writable: true },
      { dataType: 'float32', shape: [1], readable: true },
    ]);
    assert.equal(tensorDescriptors.some((descriptor) =>
      'type' in descriptor || 'dimensions' in descriptor), false);
  } finally {
    if (previousBuilder === undefined) delete globalThis.MLGraphBuilder;
    else globalThis.MLGraphBuilder = previousBuilder;
  }
});

test('WebNN releases every partially created request tensor after failure', async () => {
  const previousBuilder = globalThis.MLGraphBuilder;
  class CurrentWebNNBuilder {
    input(name, descriptor) { return { name, ...descriptor }; }
    relu(input) { return { ...input, name: 'y' }; }
    async build(outputs) { return outputs; }
  }
  globalThis.MLGraphBuilder = CurrentWebNNBuilder;
  const makeGraph = () => {
    const graph = new Graph();
    const input = graph.addInput('x', [1]);
    const output = graph.addOp('ReLU', { input }, {
      out: { name: 'y', shape: [1] },
    }).out;
    graph.setOutputs(output);
    return graph;
  };
  const makeTensor = () => ({
    destroyed: false,
    destroy() { this.destroyed = true; },
  });
  try {
    const partial = [];
    let createCount = 0;
    const createFailure = new WebNNEngine({
      async createTensor() {
        createCount++;
        if (createCount === 2) throw new Error('output allocation failed');
        const tensor = makeTensor();
        partial.push(tensor);
        return tensor;
      },
      writeTensor() {}, dispatch() {}, async readTensor() { return new ArrayBuffer(4); },
    });
    await createFailure.allocateGraph(makeGraph());
    await assert.rejects(
      createFailure.execute({ x: Float32Array.of(1) }),
      /output allocation failed/,
    );
    assert.deepEqual(partial.map((tensor) => tensor.destroyed), [true]);

    const dispatched = [];
    const dispatchFailure = new WebNNEngine({
      async createTensor() {
        const tensor = makeTensor();
        dispatched.push(tensor);
        return tensor;
      },
      writeTensor() {},
      dispatch() { throw new Error('dispatch failed'); },
      async readTensor() { return new ArrayBuffer(4); },
    });
    await dispatchFailure.allocateGraph(makeGraph());
    await assert.rejects(
      dispatchFailure.execute({ x: Float32Array.of(1) }),
      /dispatch failed/,
    );
    assert.deepEqual(dispatched.map((tensor) => tensor.destroyed), [true, true]);
  } finally {
    if (previousBuilder === undefined) delete globalThis.MLGraphBuilder;
    else globalThis.MLGraphBuilder = previousBuilder;
  }
});

test('built-in allocation rejects generic arithmetic on quantized byte tensors', async () => {
  const error = /unsupported generic operator/;
  assert.throws(() => new CPUEngine().allocateGraph(genericByteAddGraph()), error);
  await assert.rejects(
    new WebGPUEngine({}).allocateGraph(genericByteAddGraph()),
    error,
  );
  await assert.rejects(
    new WebNNEngine({}).allocateGraph(genericByteAddGraph()),
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

  graph.adapters = {
    hasActive() { return true; },
    _pinExecution() { return Object.freeze({ kind: 'single' }); },
  };
  const adapter = engine.createDecodeSession({ requireIncremental: true });
  await assert.rejects(
    adapter.seed({ x: Float32Array.of(1) }),
    /unavailable while an adapter is active/,
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

test('runtime-local provider composition covers Model, contexts, decode, and results', async () => {
  const name = 'interface-fixture';
  const events = [];
  let nextContext = 0;
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: name,
    capabilities: createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'host',
    }),
    async compile(snapshot, options) {
      events.push(['compile', snapshot.outputNames, options]);
      return {
        backendName: name,
        async createContext() {
          const contextId = ++nextContext;
          let closed = false;
          const execute = async (inputs, options = {}) => {
            assert.equal(closed, false);
            events.push(['execute', contextId, options]);
            return Object.freeze({
              outputs: Object.freeze([hostOutput(inputs.x[0] + contextId)]),
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
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: name, operatorFallback: 'forbid' },
  });
  assert.equal(compiled.backend, name);
  assert.equal(compiled.report.selectedBackend, name);

  const firstContext = await compiled.createContext();
  const secondContext = await compiled.createContext();
  const first = await firstContext.execute({ x: Float32Array.of(2) });
  const second = await secondContext.decode.seed({ x: Float32Array.of(5) });
  assert.deepEqual(await first.output('y').read(), Float32Array.of(3));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(7));
  assert.deepEqual(first.report.backendReport, { contextId: 1 });
  await secondContext.decode.reset();

  await firstContext.close();
  assert.deepEqual(await first.output('y').read(), Float32Array.of(3));
  await secondContext.close();
  await compiled.close();
  await model.close();
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
    'WebNNEngine',
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
