import test from 'node:test';
import assert from 'node:assert/strict';

import {
  Graph,
  Runtime,
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  VolvoxAI,
  createBackendProviderCapabilities,
} from '../ts/index.js';

function identityGraph({ twoOutputs = false, weight = false } = {}) {
  const graph = new Graph();
  const input = graph.addInput('x', [1]);
  const first = graph.addOp('Identity', { input }, { out: { name: 'y', shape: [1] } }).out;
  if (weight) graph.addWeight('weight', [1], 'float32', Float32Array.of(1));
  if (twoOutputs) {
    const second = graph.addOp('Identity', { input }, { out: { name: 'z', shape: [1] } }).out;
    graph.setOutputs(first, second);
  } else {
    graph.setOutputs(first);
  }
  return graph;
}

function adapterGraph() {
  const graph = new Graph();
  const input = graph.addInput('x', [1, 1]);
  const weight = graph.addWeight('w', [1, 1], 'float32', Float32Array.of(1));
  const output = graph.addOp('MatMul', { input, weight }, {
    out: { name: 'y', shape: [1, 1] },
  }).out;
  graph.setOutputs(output);
  graph.stageAdapter('tenant', {
    kind: 'lora',
    targets: [{
      weight: 'w', rank: 1, alpha: 1,
      A: Float32Array.of(1), B: Float32Array.of(1),
    }],
  }, { activate: true });
  graph.stageAdapter('tenant', {
    kind: 'lora',
    targets: [{
      weight: 'w', rank: 1, alpha: 1,
      A: Float32Array.of(1), B: Float32Array.of(2),
    }],
  });
  return graph;
}

function standaloneConstantGraph() {
  const graph = new Graph();
  const constant = graph.addTensor('constant', [2], 'float32', {
    buffer: Float32Array.of(3, 4),
  });
  graph.setOutputs(constant);
  return { graph, constant };
}

function hostOutput(name, value) {
  return {
    name,
    shape: Object.freeze([1]),
    dtype: 'float32',
    location: 'host',
    data: Float32Array.of(value),
  };
}

function emptySafetensorsBytes() {
  const header = new TextEncoder().encode('{}');
  const bytes = new ArrayBuffer(8 + header.byteLength);
  new DataView(bytes).setBigUint64(0, BigInt(header.byteLength), true);
  new Uint8Array(bytes, 8).set(header);
  return bytes;
}

function providerFixture({
  name = 'fixture',
  deviceIdentity = null,
  allocationBytes = null,
  compilationFallbackUsed = undefined,
  compilationOffendingNode = undefined,
  execute = async () => ({ outputs: Object.freeze([hostOutput('y', 3)]) }),
} = {}) {
  const events = [];
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: name,
    deviceIdentity,
    capabilities: createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'host',
    }),
    async compile() {
      events.push('compile');
      return {
        backendName: name,
        compilationEvidence: Object.freeze({
          device: deviceIdentity,
          allocationBytes,
          ...(compilationFallbackUsed === undefined
            ? {}
            : { operatorFallbackUsed: compilationFallbackUsed }),
          ...(compilationOffendingNode === undefined
            ? {}
            : { offendingNode: compilationOffendingNode }),
        }),
        async createContext() {
          events.push('create-context');
          return {
            backendName: name,
            async execute(inputs, options) {
              events.push('execute');
              return execute(inputs, options);
            },
            async close() { events.push('context-close'); },
          };
        },
        async close() { events.push('compiled-close'); },
      };
    },
    async close() { events.push('provider-close'); },
  };
  return { events, provider };
}

test('parent close is logical immediately and physical cleanup follows retained descendants', async () => {
  const { events, provider } = providerFixture();
  const runtime = new Runtime();
  runtime._addProvider('fixture', provider);
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();

  let compiledSettled = false;
  const compiledClose = compiled.close().finally(() => { compiledSettled = true; });
  await Promise.resolve();
  assert.equal(compiledSettled, false);
  assert.equal(events.includes('compiled-close'), false);
  await assert.rejects(compiled.createContext(), (error) => error.code === 'HANDLE_DISPOSED');

  let runtimeSettled = false;
  const runtimeClose = runtime.close().finally(() => { runtimeSettled = true; });
  await Promise.resolve();
  assert.equal(runtimeSettled, false);
  assert.equal(events.includes('provider-close'), false);
  assert.throws(() => runtime.createModel(identityGraph()), (error) => error.code === 'HANDLE_DISPOSED');
  await assert.rejects(model.compile(), (error) => error.code === 'HANDLE_DISPOSED');

  const result = await context.execute({ x: Float32Array.of(1) });
  assert.deepEqual(await result.output('y').read(), Float32Array.of(3));
  await context.close();
  await compiledClose;
  await Promise.resolve();
  assert.equal(runtimeSettled, false, 'Runtime.close drains an open Model handle');
  await model.close();
  await runtimeClose;

  assert.deepEqual(events, [
    'compile',
    'create-context',
    'execute',
    'context-close',
    'compiled-close',
    'provider-close',
  ]);
  assert.deepEqual(await result.output('y').read(), Float32Array.of(3));
  await result.close();
});

test('public inference contexts reject training and Dropout RNG options before providers run', async () => {
  const { events, provider } = providerFixture();
  const runtime = new Runtime();
  runtime._addProvider('fixture', provider);
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });

  await assert.rejects(
    compiled.createContext({ training: true }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  const context = await compiled.createContext();
  for (const options of [
    { training: { dropout: { seed: 17, counter: 23 } } },
    { dropout: { seed: 17, counter: 23 } },
  ]) {
    await assert.rejects(
      context.execute({ x: Float32Array.of(1) }, options),
      (error) => error.code === 'INVALID_ARGUMENT',
    );
    await assert.rejects(
      context.decode.seed({ x: Float32Array.of(1) }, options),
      (error) => error.code === 'INVALID_ARGUMENT',
    );
  }
  assert.equal(events.includes('execute'), false,
    'rejected training options must not cross the provider boundary');

  const result = await context.execute({ x: Float32Array.of(2) });
  assert.deepEqual(await result.output('y').read(), Float32Array.of(3));
  await result.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('Runtime.close drains an accepted model load and the Model retention transferred from it', async () => {
  let releaseGraph;
  let signalGraphRequested;
  const graphRequested = new Promise((resolve) => { signalGraphRequested = resolve; });
  const graphGate = new Promise((resolve) => { releaseGraph = resolve; });
  const graphDocument = JSON.stringify({
    format: 'volvox-graph/v1',
    inputs: { x: { shape: [1], dtype: 'float32' } },
    nodes: [{
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: 'y' },
      outputs_shape: { out: [1] },
      outputs_dtype: { out: 'float32' },
    }],
    outputs: ['y'],
  });
  const fetch = async (source) => {
    if (source.endsWith('graph.json')) {
      signalGraphRequested();
      await graphGate;
      return { ok: true, statusText: 'OK', text: async () => graphDocument };
    }
    return {
      ok: true,
      statusText: 'OK',
      arrayBuffer: async () => emptySafetensorsBytes(),
    };
  };
  const runtime = new Runtime();
  const loading = runtime.loadModel('memory/model.safetensors', { fetch });
  await graphRequested;

  let runtimeClosed = false;
  const closing = runtime.close().then(() => { runtimeClosed = true; });
  await Promise.resolve();
  assert.equal(runtimeClosed, false, 'Runtime.close must retain an accepted load');

  releaseGraph();
  const model = await loading;
  await Promise.resolve();
  assert.equal(runtimeClosed, false, 'the loaded Model must adopt the in-flight retention');
  await assert.rejects(model.compile(), (error) => error.code === 'HANDLE_DISPOSED');

  await model.close();
  await closing;
  assert.equal(runtimeClosed, true);
});

test('a failed accepted model load releases its Runtime retention', async () => {
  let rejectGraph;
  let signalGraphRequested;
  const graphRequested = new Promise((resolve) => { signalGraphRequested = resolve; });
  const graphGate = new Promise((_, reject) => { rejectGraph = reject; });
  const runtime = new Runtime();
  const loading = runtime.loadModel('memory/model.safetensors', {
    fetch: async () => {
      signalGraphRequested();
      await graphGate;
    },
  });
  await graphRequested;

  let runtimeClosed = false;
  const closing = runtime.close().then(() => { runtimeClosed = true; });
  await Promise.resolve();
  assert.equal(runtimeClosed, false, 'Runtime.close must drain the accepted load failure');

  rejectGraph(new Error('graph fetch failed'));
  await assert.rejects(loading, (error) =>
    error.code === 'INVALID_ARGUMENT' && error.phase === 'initialization' &&
    /graph fetch failed/.test(error.message));
  await closing;
  assert.equal(runtimeClosed, true);
});

test('runtime model construction and package loading expose typed initialization failures', async () => {
  const runtime = new Runtime();
  assert.throws(() => runtime.createModel({}), (error) =>
    error.code === 'INVALID_ARGUMENT' && error.phase === 'initialization' &&
    /Model graph is invalid/.test(error.message));

  await assert.rejects(runtime.loadModel('memory/model.safetensors', {
    fetch: async (source) => source.endsWith('graph.json')
      ? {
          ok: true,
          statusText: 'OK',
          text: async () => JSON.stringify({ format: 'not-volvox', outputs: ['x'] }),
        }
      : { ok: true, statusText: 'OK', arrayBuffer: async () => emptySafetensorsBytes() },
  }), (error) =>
    error.code === 'INVALID_ARGUMENT' && error.phase === 'initialization' &&
    /volvox-graph\/v1/.test(error.message));
  await runtime.close();
});

test('failed provider snapshots release device ownership once and do not poison the context queue', async () => {
  const releases = { duplicate: 0, invalid: 0, report: 0 };
  const diagnostics = [];
  let execution = 0;
  const deviceOutput = (name, key) => ({
    name,
    shape: Object.freeze([1]),
    dtype: 'float32',
    location: 'device',
    deviceBuffer: { size: 4 },
    async read() { return Float32Array.of(1); },
    release() { releases[key]++; },
  });
  const { provider } = providerFixture({
    execute: async () => {
      execution++;
      if (execution === 1) {
        const duplicate = deviceOutput('y', 'duplicate');
        return { outputs: Object.freeze([duplicate, duplicate]) };
      }
      if (execution === 2) {
        return {
          outputs: Object.freeze([
            deviceOutput('y', 'invalid'),
            { ...hostOutput('z', 0), data: new Float32Array(0) },
          ]),
        };
      }
      if (execution === 3) {
        const backendReport = {};
        backendReport.self = backendReport;
        return {
          outputs: Object.freeze([
            deviceOutput('y', 'report'),
            hostOutput('z', 0),
          ]),
          backendReport,
        };
      }
      return { outputs: Object.freeze([hostOutput('y', 7), hostOutput('z', 8)]) };
    },
  });
  const runtime = new Runtime({ onDiagnostic: (event) => diagnostics.push(event) });
  runtime._addProvider('fixture', provider);
  const model = runtime.createModel(identityGraph({ twoOutputs: true }));
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();

  await assert.rejects(context.execute({ x: Float32Array.of(1) }), (error) => {
    assert.equal(error.code, 'EXECUTION_FAILED');
    assert.equal(error.report.outcome, 'failed');
    assert.equal(error.report.contextId, context.id);
    assert.ok(error.report.executionTimeMs >= 0);
    return true;
  });
  assert.equal(releases.duplicate, 1);

  await assert.rejects(context.execute({ x: Float32Array.of(2) }),
    (error) => error.code === 'EXECUTION_FAILED');
  assert.equal(releases.invalid, 1);

  await assert.rejects(context.execute({ x: Float32Array.of(3) }),
    (error) => error.code === 'EXECUTION_FAILED');
  assert.equal(releases.report, 1);
  assert.equal(diagnostics.filter(({ kind }) => kind === 'execution-error').length, 3);
  assert.equal(diagnostics.find(({ kind }) => kind === 'execution-error').report.outcome, 'failed');

  const recovered = await context.execute({ x: Float32Array.of(4) });
  assert.deepEqual(await recovered.output('y').read(), Float32Array.of(7));
  assert.deepEqual(await recovered.output('z').read(), Float32Array.of(8));

  await recovered.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('reported-route provider failures retain a serializable failure report', async () => {
  const diagnostics = [];
  const reported = providerFixture({
    name: 'reported-failure',
    execute: async () => { throw new Error('provider exploded'); },
  });
  reported.provider.capabilities = createBackendProviderCapabilities({
    operatorFallback: 'reported',
    outputLocation: 'host',
  });
  const runtime = new Runtime({ onDiagnostic: (event) => diagnostics.push(event) });
  runtime._addProvider('reported-failure', reported.provider);
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: {
      mode: 'require', backend: 'reported-failure', operatorFallback: 'allow',
    },
  });
  const context = await compiled.createContext();

  await assert.rejects(context.execute({ x: Float32Array.of(1) }), (error) => {
    assert.equal(error.code, 'EXECUTION_FAILED');
    assert.equal(error.report.outcome, 'failed');
    assert.match(error.report.message, /provider exploded$/);
    assert.deepEqual(error.report.routeEvidence.operator, {
      attestation: 'reported', used: null, offendingNode: null,
    });
    assert.doesNotThrow(() => JSON.stringify(error.report));
    return true;
  });
  const failure = diagnostics.find(({ kind }) => kind === 'execution-error');
  assert.ok(failure);
  assert.equal(failure.report.code, 'EXECUTION_FAILED');
  assert.deepEqual(failure.report.routeEvidence.operator, {
    attestation: 'reported', used: null, offendingNode: null,
  });

  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('execution rejects provider outputs that disagree with the retained model descriptors', async () => {
  const releases = { dtype: 0, shape: 0 };
  let execution = 0;
  const { provider } = providerFixture({
    execute: async () => {
      execution++;
      if (execution === 1) {
        return {
          outputs: Object.freeze([{
            name: 'y',
            shape: Object.freeze([1]),
            dtype: 'int32',
            location: 'device',
            deviceBuffer: { size: 4 },
            async read() { return Int32Array.of(1); },
            release() { releases.dtype++; },
          }]),
        };
      }
      if (execution === 2) {
        return {
          outputs: Object.freeze([{
            name: 'y',
            shape: Object.freeze([2]),
            dtype: 'float32',
            location: 'device',
            deviceBuffer: { size: 8 },
            async read() { return Float32Array.of(1, 2); },
            release() { releases.shape++; },
          }]),
        };
      }
      return { outputs: Object.freeze([hostOutput('y', 9)]) };
    },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', provider);
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();

  await assert.rejects(context.execute({ x: Float32Array.of(1) }), (error) => {
    assert.equal(error.code, 'EXECUTION_FAILED');
    assert.match(error.message, /declared dtype and shape/);
    return true;
  });
  assert.equal(releases.dtype, 1);

  await assert.rejects(context.execute({ x: Float32Array.of(2) }), (error) => {
    assert.equal(error.code, 'EXECUTION_FAILED');
    assert.match(error.message, /declared dtype and shape/);
    return true;
  });
  assert.equal(releases.shape, 1);

  const recovered = await context.execute({ x: Float32Array.of(3) });
  assert.deepEqual(await recovered.output('y').read(), Float32Array.of(9));
  assert.deepEqual(releases, { dtype: 1, shape: 1 });

  await recovered.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('device results own their buffer through parent closure and drain accepted reads before release', async () => {
  let releaseRead;
  let releases = 0;
  const backendReport = { route: { tier: 'fixture' } };
  const readGate = new Promise((resolve) => { releaseRead = resolve; });
  const { provider } = providerFixture({
    execute: async () => ({
      outputs: Object.freeze([{
        name: 'y',
        shape: Object.freeze([1]),
        dtype: 'float32',
        location: 'device',
        deviceBuffer: { size: 4 },
        async read() {
          await readGate;
          return Float32Array.of(11);
        },
        release() { releases++; },
      }]),
      backendReport,
    }),
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', provider);
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const result = await context.execute({ x: Float32Array.of(1) });
  backendReport.route.tier = 'mutated';
  assert.deepEqual(result.report.backendReport, { route: { tier: 'fixture' } });
  assert.equal(Object.isFrozen(result.report.backendReport.route), true);
  const tensor = result.output('y');
  assert.equal(tensor.location, 'device');
  assert.equal(tensor.deviceBuffer.size, 4);

  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
  assert.equal(releases, 0);

  const read = tensor.read();
  const close = result.close();
  await Promise.resolve();
  assert.equal(releases, 0);
  releaseRead();
  assert.deepEqual(await read, Float32Array.of(11));
  await close;
  assert.equal(releases, 1);
  await assert.rejects(tensor.read(), (error) => error.code === 'RESULT_DISPOSED');
  await result.close();
  assert.equal(releases, 1);
});

test('host result closure drains accepted reads and releases its retained snapshot', async () => {
  const { provider } = providerFixture({
    execute: async () => ({ outputs: Object.freeze([hostOutput('y', 17)]) }),
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', provider);
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const result = await context.execute({ x: Float32Array.of(1) });
  const tensor = result.output('y');

  const acceptedRead = tensor.read();
  const close = result.close();
  assert.deepEqual(await acceptedRead, Float32Array.of(17));
  await close;
  assert.equal(result.closed, true);
  await assert.rejects(tensor.read(), (error) => error.code === 'RESULT_DISPOSED');

  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('one context queues execution and close in FIFO order while rejecting late work', async () => {
  let releaseFirst;
  let signalFirst;
  const firstStarted = new Promise((resolve) => { signalFirst = resolve; });
  let calls = 0;
  let active = 0;
  let maxActive = 0;
  const { provider, events } = providerFixture({
    execute: async () => {
      const call = ++calls;
      active++;
      maxActive = Math.max(maxActive, active);
      if (call === 1) {
        signalFirst();
        await new Promise((resolve) => { releaseFirst = resolve; });
      }
      active--;
      return { outputs: Object.freeze([hostOutput('y', call)]) };
    },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', provider);
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();

  const first = context.execute({ x: Float32Array.of(1) });
  await firstStarted;
  const second = context.execute({ x: Float32Array.of(2) });
  const close = context.close();
  await assert.rejects(context.execute({ x: Float32Array.of(3) }),
    (error) => error.code === 'HANDLE_DISPOSED');
  await Promise.resolve();
  assert.equal(calls, 1);
  assert.equal(events.includes('context-close'), false);

  releaseFirst();
  const [firstResult, secondResult] = await Promise.all([first, second]);
  await close;
  assert.equal(maxActive, 1);
  assert.deepEqual(await firstResult.output('y').read(), Float32Array.of(1));
  assert.deepEqual(await secondResult.output('y').read(), Float32Array.of(2));
  assert.equal(events.at(-1), 'context-close');

  await Promise.all([firstResult.close(), secondResult.close()]);
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('two contexts from one compiled model execute concurrently with independent identities and results', async () => {
  let releaseExecutions;
  let signalBothEntered;
  const executionGate = new Promise((resolve) => { releaseExecutions = resolve; });
  const bothEntered = new Promise((resolve) => { signalBothEntered = resolve; });
  const entered = [];
  let nextBackendContext = 0;
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: 'overlap',
    capabilities: createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'host',
    }),
    async compile() {
      return {
        backendName: 'overlap',
        async createContext() {
          const backendContext = ++nextBackendContext;
          return {
            backendName: 'overlap',
            async execute() {
              entered.push(backendContext);
              if (entered.length === 2) signalBothEntered();
              await executionGate;
              return { outputs: Object.freeze([hostOutput('y', backendContext)]) };
            },
            async close() {},
          };
        },
        async close() {},
      };
    },
    async close() {},
  };
  const runtime = new Runtime();
  runtime._addProvider('overlap', provider);
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'overlap', operatorFallback: 'forbid' },
  });
  const [firstContext, secondContext] = await Promise.all([
    compiled.createContext(),
    compiled.createContext(),
  ]);

  const firstPending = firstContext.execute({ x: Float32Array.of(1) });
  const secondPending = secondContext.execute({ x: Float32Array.of(2) });
  await bothEntered;
  assert.deepEqual(entered, [1, 2], 'both provider contexts must enter before either is released');
  assert.notEqual(firstContext.id, secondContext.id);

  releaseExecutions();
  const [first, second] = await Promise.all([firstPending, secondPending]);
  assert.equal(first.report.contextId, firstContext.id);
  assert.equal(second.report.contextId, secondContext.id);
  assert.notEqual(first.report.contextId, second.report.contextId);
  assert.deepEqual(await first.output('y').read(), Float32Array.of(1));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(2));

  await Promise.all([first.close(), second.close()]);
  await Promise.all([firstContext.close(), secondContext.close()]);
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('built-in CPU contexts isolate caller storage and results survive later work and closure', async () => {
  const graph = identityGraph();
  assert.equal(graph.getTensor('x').buffer, undefined);
  assert.equal(graph.getTensor('y').buffer, undefined);

  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const model = runtime.createModel(graph);
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const firstContext = await compiled.createContext();
  const secondContext = await compiled.createContext();

  const first = await firstContext.execute({ x: Float32Array.of(2) });
  const other = await secondContext.execute({ x: Float32Array.of(7) });
  const later = await firstContext.execute({ x: Float32Array.of(9) });

  assert.equal(Object.isFrozen(first.outputs), true);
  assert.equal(first.outputs.set, undefined, 'result output collections are runtime read-only');
  assert.deepEqual([...first.outputs.keys()], ['y']);
  assert.deepEqual(compiled.report.selectedDevice, { backend: 'cpu', device: 'host' });
  assert.equal(compiled.report.allocationBytes, 8);
  assert.equal(compiled.report.adapterRevisionId, null);
  assert.ok(compiled.report.compileTimeMs >= 0);
  assert.deepEqual(first.report.routeEvidence, {
    tierFallback: false,
    operator: { attestation: 'none', used: false, offendingNode: null },
  });
  assert.equal(first.report.decodeState.operation, 'execute');
  assert.equal(first.report.decodeState.mode, null);
  assert.equal(first.report.decodeState.cacheState, 'not-applicable');
  assert.ok(Number.isSafeInteger(first.report.decodeState.cacheGeneration));
  assert.equal(first.report.decodeState.position, null);
  assert.deepEqual(first.report.device, compiled.report.selectedDevice);
  assert.equal(first.report.adapterRevisionId, null);
  assert.ok(first.report.executionTimeMs >= 0);

  assert.deepEqual(await first.output('y').read(), Float32Array.of(2));
  assert.deepEqual(await other.output('y').read(), Float32Array.of(7));
  assert.deepEqual(await later.output('y').read(), Float32Array.of(9));
  assert.equal(graph.getTensor('x').buffer, undefined);
  assert.equal(graph.getTensor('y').buffer, undefined);

  await firstContext.close();
  await secondContext.close();
  await compiled.close();
  await model.close();
  await runtime.close();
  assert.deepEqual(await first.output('y').read(), Float32Array.of(2));

  await Promise.all([first.close(), other.close(), later.close()]);
});

test('model snapshots retain standalone constants without sharing caller storage', async () => {
  const { graph, constant } = standaloneConstantGraph();
  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const model = runtime.createModel(graph);
  constant.buffer[0] = 100;
  constant.buffer[1] = 200;
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const result = await context.execute({});

  assert.deepEqual(await result.output('constant').read(), Float32Array.of(3, 4));
  assert.deepEqual(constant.buffer, Float32Array.of(100, 200));

  await result.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('weight-only publication retains definition identity and pins existing compilations', async () => {
  const { provider } = providerFixture();
  const runtime = new Runtime();
  runtime._addProvider('fixture', provider);
  const graph = identityGraph({ weight: true });
  const model = runtime.createModel(graph);
  const originalDefinition = model.definitionId;
  const originalWeights = model.weightRevisionId;
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });

  graph.applyTensorUpdate('weight', Float32Array.of(4));
  const weightRevision = model.publishRevision(graph);
  assert.equal(weightRevision.definitionId, originalDefinition);
  assert.notEqual(weightRevision.weightRevisionId, originalWeights);
  assert.equal(compiled.definitionId, originalDefinition);
  assert.equal(compiled.weightRevisionId, originalWeights);

  const differentGraph = new Graph();
  const differentInput = differentGraph.addInput('x', [1]);
  const differentOutput = differentGraph.addOp('Identity', { input: differentInput }, {
    out: { name: 'different', shape: [1] },
  }).out;
  differentGraph.addWeight('weight', [1], 'float32', Float32Array.of(4));
  differentGraph.setOutputs(differentOutput);
  assert.equal(differentGraph.topologyRevision, graph.topologyRevision);
  const differentRevision = model.publishRevision(differentGraph);
  assert.notEqual(differentRevision.definitionId, originalDefinition,
    'definition identity is structural rather than a revision-counter coincidence');

  const y = graph.getTensor('y');
  const z = graph.addOp('Identity', { input: y }, { out: { name: 'z', shape: [1] } }).out;
  graph.setOutputs(z);
  const topologyRevision = model.publishRevision(graph);
  assert.notEqual(topologyRevision.definitionId, originalDefinition);

  await compiled.close();
  await model.close();
  await runtime.close();
});

test('contexts pin concrete adapter revisions and serialize route changes', async () => {
  const graph = adapterGraph();
  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const model = runtime.createModel(graph);
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  assert.deepEqual(compiled.report.adapterRevisionIds, ['tenant@1', 'tenant@2']);
  assert.equal(compiled.report.adapterRevisionId, 'tenant@1');

  for (const adapter of [
    'tenant',
    { adapterId: 'tenant' },
    { adapter_id: 'tenant' },
    { name: 'tenant', versionId: 1 },
    { name: 'tenant', version_id: 1 },
    { name: 'tenant', version: '1' },
  ]) {
    await assert.rejects(
      compiled.createContext({ adapter }),
      (error) => error.code === 'INVALID_ARGUMENT' &&
        /unsupported field|object or null|positive integer/.test(error.message),
    );
  }

  const context = await compiled.createContext({
    adapter: { name: 'tenant', version: 1 },
  });
  const first = await context.execute({ x: Float32Array.of(2) });
  assert.deepEqual(await first.output('y').read(), Float32Array.of(4));
  assert.equal(first.report.adapterRevisionId, 'tenant@1');
  assert.deepEqual(first.report.adapterRevisionIds, ['tenant@1']);

  graph.stageAdapter('tenant', {
    kind: 'lora',
    targets: [{
      weight: 'w', rank: 1, alpha: 1,
      A: Float32Array.of(1), B: Float32Array.of(9),
    }],
  });
  await context.selectAdapter({ name: 'tenant' });
  assert.equal(context.adapterRevisionId, 'tenant@2');
  const second = await context.execute({ x: Float32Array.of(2) });
  assert.deepEqual(await second.output('y').read(), Float32Array.of(6));
  assert.equal(second.report.adapterRevisionId, 'tenant@2');
  assert.deepEqual(await first.output('y').read(), Float32Array.of(4));

  await Promise.all([first.close(), second.close()]);
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('backend tier policy and operator fallback policy produce exact serializable evidence', async () => {
  const diagnostics = [];
  const reported = providerFixture({
    name: 'reported',
    execute: async () => ({
      outputs: Object.freeze([hostOutput('y', 5)]),
      backendReport: Object.freeze({
        route: Object.freeze({ operatorFallbackUsed: true, offendingNode: 'identity-0' }),
      }),
    }),
  });
  reported.provider.capabilities = createBackendProviderCapabilities({
    operatorFallback: 'reported',
    outputLocation: 'host',
  });
  const strict = providerFixture({
    name: 'strict',
    deviceIdentity: Object.freeze({ vendor: 'Fixture', device: 'Strict-0' }),
    allocationBytes: 128,
  });
  const contradictory = providerFixture({
    name: 'contradictory',
    compilationFallbackUsed: true,
    compilationOffendingNode: 'identity-0',
  });
  const runtime = new Runtime({ onDiagnostic: (event) => diagnostics.push(event) });
  runtime._addProvider('reported', reported.provider);
  runtime._addProvider('strict', strict.provider);
  runtime._addProvider('contradictory', contradictory.provider);
  const model = runtime.createModel(identityGraph());

  const compiled = await model.compile({
    backend: {
      mode: 'prefer',
      order: ['reported', 'strict'],
      operatorFallback: 'forbid',
    },
  });
  assert.equal(compiled.backend, 'strict');
  assert.equal(reported.events.includes('compile'), false);
  assert.deepEqual(compiled.report.candidates.map(({ backend, outcome, code }) => ({
    backend, outcome, code,
  })), [
    { backend: 'reported', outcome: 'unsupported', code: 'OPERATOR_FALLBACK_FORBIDDEN' },
    { backend: 'strict', outcome: 'selected', code: null },
  ]);
  assert.doesNotThrow(() => JSON.stringify(compiled.report));
  assert.deepEqual(compiled.report.selectedDevice, { vendor: 'Fixture', device: 'Strict-0' });
  assert.equal(compiled.report.allocationBytes, 128);
  assert.equal(compiled.report.routeEvidence.tierFallback, true);
  assert.deepEqual(compiled.report.routeEvidence.operator, {
    attestation: 'none', used: false, offendingNode: null,
  });
  assert.ok(compiled.report.compileTimeMs >= compiled.report.candidates[0].elapsedMs);

  await assert.rejects(model.compile({
    backend: {
      mode: 'require',
      backend: 'reported',
      operatorFallback: 'forbid',
    },
  }), (error) => {
    assert.equal(error.code, 'OPERATOR_FALLBACK_FORBIDDEN');
    assert.equal(error.report.selectedBackend, null);
    return true;
  });
  await assert.rejects(model.compile({
    backend: {
      mode: 'require', backend: 'contradictory', operatorFallback: 'forbid',
    },
  }), (error) => {
    assert.equal(error.code, 'OPERATOR_FALLBACK_FORBIDDEN');
    assert.equal(error.report.candidates[0].code, 'OPERATOR_FALLBACK_FORBIDDEN');
    assert.equal(error.report.candidates[0].routeEvidence.operator.offendingNode, 'identity-0');
    return true;
  });
  assert.equal(contradictory.events.filter((event) => event === 'compiled-close').length, 1);
  assert.equal(diagnostics.filter(({ kind }) => kind === 'compilation').length, 3);

  const context = await compiled.createContext();
  const result = await context.execute({ x: Float32Array.of(1) });
  assert.equal(result.report.operatorFallback, 'none');
  assert.deepEqual(result.report.device, compiled.report.selectedDevice);
  assert.equal(result.report.routeEvidence.tierFallback, true);
  assert.deepEqual(result.report.decodeState, {
    operation: 'execute', mode: null, cacheState: 'not-applicable',
    cacheGeneration: null, position: null,
  });
  assert.doesNotThrow(() => JSON.stringify(result.report));

  const routed = await model.compile({
    backend: {
      mode: 'require', backend: 'reported', operatorFallback: 'allow',
    },
  });
  const routedContext = await routed.createContext();
  const routedResult = await routedContext.execute({ x: Float32Array.of(1) });
  assert.deepEqual(routedResult.report.routeEvidence, {
    tierFallback: false,
    operator: { attestation: 'reported', used: true, offendingNode: 'identity-0' },
  });

  await result.close();
  await routedResult.close();
  await routedContext.close();
  await routed.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('Model.compile rejects string, array, and auto backend shorthands', async () => {
  const { provider } = providerFixture();
  const runtime = new Runtime();
  runtime._addProvider('fixture', provider);
  const model = runtime.createModel(identityGraph());

  for (const backend of ['fixture', ['fixture'], 'auto']) {
    await assert.rejects(
      model.compile({ backend }),
      (error) => error.code === 'INVALID_ARGUMENT' && /policy is invalid/i.test(error.message),
    );
  }

  await model.close();
  await runtime.close();
});

test('provider validation reclaims partially created compiled models and contexts', async () => {
  let invalidCompiledCloses = 0;
  const invalidCompiledProvider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: 'invalid-compiled',
    capabilities: createBackendProviderCapabilities({ operatorFallback: 'none' }),
    async compile() {
      return {
        backendName: 'invalid-compiled',
        async close() { invalidCompiledCloses++; },
      };
    },
    async close() {},
  };
  const firstRuntime = new Runtime();
  firstRuntime._addProvider('invalid-compiled', invalidCompiledProvider);
  const firstModel = firstRuntime.createModel(identityGraph());
  await assert.rejects(firstModel.compile({
    backend: { mode: 'require', backend: 'invalid-compiled', operatorFallback: 'forbid' },
  }),
    (error) => error.code === 'BACKEND_REQUIRED');
  assert.equal(invalidCompiledCloses, 1);
  await firstModel.close();
  await firstRuntime.close();

  let invalidContextCloses = 0;
  let compiledCloses = 0;
  const invalidContextProvider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: 'invalid-context',
    capabilities: createBackendProviderCapabilities({ operatorFallback: 'none' }),
    async compile() {
      return {
        backendName: 'invalid-context',
        async createContext() {
          return {
            backendName: 'invalid-context',
            async close() { invalidContextCloses++; },
          };
        },
        async close() { compiledCloses++; },
      };
    },
    async close() {},
  };
  const secondRuntime = new Runtime();
  secondRuntime._addProvider('invalid-context', invalidContextProvider);
  const secondModel = secondRuntime.createModel(identityGraph());
  const compiled = await secondModel.compile({
    backend: { mode: 'require', backend: 'invalid-context', operatorFallback: 'forbid' },
  });
  await assert.rejects(compiled.createContext(), (error) => error.code === 'ABI_UNSUPPORTED');
  assert.equal(invalidContextCloses, 1);
  await compiled.close();
  assert.equal(compiledCloses, 1);
  await secondModel.close();
  await secondRuntime.close();
});

test('external provider factories are scoped to the Runtime that owns them', async () => {
  const name = 'context-runtime-fixture';
  const { provider, events } = providerFixture({ name });
  const runtime = await VolvoxAI.createRuntime({
    backends: [name],
    providers: {
      [name]: async ({ name: requested }) => {
        assert.equal(requested, name);
        return provider;
      },
    },
  });
  const model = runtime.createModel(identityGraph());
  const compiled = await model.compile({
    backend: { mode: 'require', backend: name, operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const result = await context.execute({ x: Float32Array.of(1) });
  assert.deepEqual(await result.output('y').read(), Float32Array.of(3));
  await result.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
  assert.equal(events.at(-1), 'provider-close');

  await assert.rejects(
    VolvoxAI.createRuntime({ backends: [name] }),
    (error) => error.code === 'INVALID_ARGUMENT' && error.phase === 'initialization' &&
      /Unknown backend provider/.test(error.message),
  );
  await assert.rejects(
    VolvoxAI.createRuntime({ backends: name }),
    (error) => error.code === 'INVALID_ARGUMENT' && error.phase === 'initialization' &&
      /non-empty ordered array/.test(error.message),
  );
  await assert.rejects(
    VolvoxAI.createRuntime({ backends: ['cpu', 'cpu'] }),
    (error) => error.code === 'INVALID_ARGUMENT' && error.phase === 'initialization' &&
      /occurs more than once/.test(error.message),
  );
  await assert.rejects(
    VolvoxAI.createRuntime({ backends: [name], providers: { [name]: 7 } }),
    (error) => error.code === 'ABI_UNSUPPORTED' && error.phase === 'initialization',
  );

  const instance = providerFixture({ name });
  const instanceRuntime = await VolvoxAI.createRuntime({
    backends: [name],
    providers: { [name]: instance.provider },
  });
  assert.deepEqual(instanceRuntime.listBackends(), [name]);
  await instanceRuntime.close();
  assert.equal(instance.events.at(-1), 'provider-close');
});

test('WebGPU initialization retries a transient null adapter result', { concurrency: false }, async (t) => {
  const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let requestCount = 0;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          requestCount++;
          if (requestCount === 1) return null;
          return {
            info: { vendor: 'fixture', device: 'transient-adapter' },
            async requestDevice() { return {}; },
          };
        },
      },
    },
  });
  t.after(() => {
    if (navigatorDescriptor) {
      Object.defineProperty(globalThis, 'navigator', navigatorDescriptor);
    } else {
      delete globalThis.navigator;
    }
  });

  const runtime = await VolvoxAI.createRuntime({ backends: ['webgpu'] });
  assert.equal(requestCount, 2);
  assert.deepEqual(runtime.listBackends(), ['webgpu']);
  await runtime.close();
});

test('the inference entry exposes the lifecycle/provider API without built-in engine facades', async () => {
  const api = await import('../ts/index.js');
  for (const hidden of [
    'BackendEngine',
    'DecodeSession',
    'CPUEngine',
    'WasmEngine',
    'WebGPUEngine',
    'WebNNEngine',
    'GraphExecutor',
    'AdapterManager',
    'ModelSnapshot',
    'runtimeError',
    'parseStrictJSON',
  ]) {
    assert.equal(hidden in api, false, `${hidden} must remain internal`);
  }
  for (const exposed of [
    'Runtime',
    'Model',
    'CompiledModel',
    'ExecutionContext',
    'ExecutionResult',
    'TensorResult',
    'VolvoxAI',
    'VOLVOXAI_BACKEND_PROVIDER_VERSION',
  ]) {
    assert.equal(exposed in api, true, `${exposed} must be public`);
  }
  assert.equal(typeof VolvoxAI, 'object');
  assert.equal(Object.isFrozen(VolvoxAI), true);
  assert.deepEqual(Object.keys(VolvoxAI).sort(), [
    'createRuntime',
  ]);
  assert.equal('registerBackendProvider' in api, false);
  assert.equal('listBackendProviders' in api, false);
  assert.equal('unregisterBackendProvider' in api, false);
});
