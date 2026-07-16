import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BackendEngine,
  CPUEngine,
  Graph,
  GraphExecutor,
  VOLVOXAI_BACKEND_API_VERSION,
  VolvoxAI,
  WebGPUEngine,
  WebNNEngine,
  assertBackendEngine,
} from '../ts/index.js';

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

function reluGraph(adapters = null) {
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
  graph.outputNames = [out.name];
  return graph;
}

test('built-in browser engines expose backend API v1 and one lifecycle shape', () => {
  const cpu = new CPUEngine();
  const webnn = new WebNNEngine({});
  assert.equal(VOLVOXAI_BACKEND_API_VERSION, 1);
  assert.equal(cpu.backendApiVersion, 1);
  assert.equal(cpu.backendName, 'cpu');
  assert.deepEqual(cpu.capabilities, {
    incrementalExecution: true, incrementalRows: true, outputLocation: 'host',
  });
  assert.equal(webnn.backendName, 'webnn');
  assert.deepEqual(webnn.capabilities, {
    incrementalExecution: false, incrementalRows: false, outputLocation: 'host',
  });
  for (const engine of [cpu, webnn]) {
    assert.equal(typeof engine.allocateGraph, 'function');
    assert.equal(typeof engine.execute, 'function');
    assert.equal(typeof engine.createDecodeSession, 'function');
  }
  assert.equal(typeof GraphExecutor.prototype.createDecodeSession, 'function');
  assert.throws(
    () => assertBackendEngine({ allocateGraph() {}, execute() {} }, 'Incomplete backend'),
    /must implement BackendEngine API v1/,
  );
  assert.throws(
    () => assertBackendEngine({
      backendApiVersion: 1,
      backendName: 'unsafe-incremental',
      capabilities: Object.freeze({
        incrementalExecution: true,
        incrementalRows: false,
        outputLocation: 'host',
      }),
      allocateGraph() {},
      execute() {},
      createDecodeSession() {},
    }, 'Unsafe incremental backend'),
    /must implement BackendEngine API v1/,
  );
});

test('built-in browser allocation rejects generic arithmetic on quantized byte tensors', async () => {
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

test('DecodeSession maps seed and row steps onto the compatibility option contract', async () => {
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

test('DecodeSession falls back to ordinary forward and detects a replaced cache', async () => {
  const ordinary = new RecordingBackend('ordinary');
  ordinary.supportsIncrementalExecution = true;
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

test('closing a stale or unseeded DecodeSession preserves the active cache owner', async () => {
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
});

test('concurrent sessions serialize async seeds and retain one cache owner', async () => {
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

test('a direct legacy execute replaces an active DecodeSession cache owner', async () => {
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

test('DecodeSession rejects unknown changed inputs and dynamically unavailable caching', async () => {
  const graph = reluGraph();
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
    /cannot be combined with training execution/,
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

test('WebGPUEngine owns a GraphExecutor peer while preserving device readback access', async () => {
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
    async prepareForTraining() { this.resetDecodeCache(); events.push('prepare-training'); }
    dispose() { this.resetDecodeCache(); events.push('dispose'); }
  }
  class StubWebGPUEngine extends WebGPUEngine {
    _createExecutor(graph) {
      return new StubExecutor(graph);
    }
  }

  const engine = new StubWebGPUEngine({ label: 'device' });
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
  assert.ok(engine.decodeCacheGeneration > generationBeforeFeedback,
    'a direct device-feedback call replaces wrapper decode-cache ownership');
  assert.equal(engine.backendName, 'webgpu');
  assert.equal(engine.capabilities.outputLocation, 'device');
  assert.equal(engine.supportsIncrementalRows, true);
  const peer = engine.fork();
  assert.ok(peer instanceof WebGPUEngine);
  assert.notEqual(peer, engine);
  assert.equal(peer.device, engine.device);
  assert.equal(peer.executor, null);
  const decode = engine.createDecodeSession();
  await decode.seed({ x: Float32Array.of(1) });
  await engine.executor.execute({ x: Float32Array.of(2) });
  await assert.rejects(
    decode.step({ x: Float32Array.of(3) }),
    /cache was reset or replaced/,
  );
  await decode.seed({ x: Float32Array.of(1) });
  await engine.prepareForTraining();
  await assert.rejects(
    decode.step({ x: Float32Array.of(2) }),
    /cache was reset or replaced/,
  );
  engine.dispose();
  assert.equal(events[0], 'compile');
});

test('VolvoxAI registers a named out-of-tree backend without editing built-in selection', async () => {
  const name = 'test-npu';
  const unregister = VolvoxAI.registerBackend(name, () => new RecordingBackend(name));
  try {
    assert.ok(VolvoxAI.listBackends().includes(name));
    const runtime = await VolvoxAI.init(name);
    assert.deepEqual(runtime.engines.map(({ type }) => type), [name]);
    const graph = emptyGraph();
    const engine = await runtime.compile(graph);
    assert.equal(engine.backendName, name);
    assert.equal(engine.graph, graph);
  } finally {
    unregister();
  }
  assert.equal(VolvoxAI.listBackends().includes(name), false);
});

test('VolvoxAI detached compilation preserves the runtime engine graph owner', async () => {
  const runtime = new VolvoxAI();
  const primary = new CPUEngine();
  const primaryGraph = reluGraph();
  primary.allocateGraph(primaryGraph);
  runtime.engines.push({ type: 'cpu', engine: primary });

  const detachedGraph = reluGraph();
  const detached = await runtime.compileDetached(detachedGraph);
  assert.notEqual(detached, primary);
  assert.equal(primary.graph, primaryGraph);
  assert.equal(detached.graph, detachedGraph);
});
