import test from 'node:test';
import assert from 'node:assert/strict';

import { Runtime } from '../ts/core/ContextRuntime.js';
import { Model } from '../ts/core/Model.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { createBackendCompileInput } from '../ts/backends/BackendProvider.js';
import { WebNNBackendProvider } from '../ts/backends/WebNNBackendProvider.js';
import { WebNNEngine } from '../ts/backends/WebNNEngine.js';

function dynamicReluSnapshot(maximum = 12) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 1, max: maximum } },
    inputs: { x: { dtype: 'float32', shape: [1, 'S'] } },
    nodes: [{
      id: 'relu',
      opType: 'ReLU',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 'S'] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function dynamicMatrixReluSnapshot(rows, maximum) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 1, max: maximum } },
    inputs: { x: { dtype: 'float32', shape: [rows, 'S'] } },
    nodes: [{
      id: 'relu',
      opType: 'ReLU',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [rows, 'S'] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function shaped(length, offset = 0) {
  return {
    x: {
      data: Float32Array.from(
        { length },
        (_, index) => index % 2 === 0 ? -index - offset - 1 : index + offset,
      ),
      shape: [1, length],
    },
  };
}

function expectedRelu(input) {
  return [...input.x.data].map((value) => Math.max(0, value));
}

function makeMockWebNN() {
  let loseContext;
  const lost = new Promise((resolve) => { loseContext = resolve; });
  const tensorLimit = {
    dataTypes: ['float32', 'int32', 'int8', 'uint8'],
    rankRange: { min: 0, max: 8 },
  };
  const state = {
    buildCount: 0,
    graphs: [],
    tensors: [],
    failNextBuild: false,
    deferredBuild: null,
    contextDestroyCount: 0,
    supportLimits: {
      maxTensorByteLength: 1024 * 1024,
      input: tensorLimit,
      constant: tensorLimit,
      output: tensorLimit,
      relu: { input: tensorLimit, output: tensorLimit },
    },
  };

  const cloneStorage = (value) => {
    if (value instanceof Float32Array) return new Float32Array(value);
    if (value instanceof Int32Array) return new Int32Array(value);
    if (value instanceof Int8Array) return new Int8Array(value);
    return new Uint8Array(value);
  };

  const evaluate = (operand, inputs) => {
    if (operand.kind === 'input') return inputs[operand.name].data;
    if (operand.kind === 'relu') {
      return Float32Array.from(evaluate(operand.input, inputs), (value) => Math.max(0, value));
    }
    throw new Error(`mock cannot evaluate '${String(operand.kind)}'`);
  };

  class MockMLGraphBuilder {
    constructor(context) { this.context = context; }
    input(name, descriptor) {
      return { kind: 'input', name, dataType: descriptor.dataType, shape: descriptor.shape };
    }
    relu(input) {
      return { kind: 'relu', input, dataType: input.dataType, shape: input.shape };
    }
    async build(outputs) {
      state.buildCount++;
      if (state.failNextBuild) {
        state.failNextBuild = false;
        throw new Error('injected WebNN graph build failure');
      }
      if (state.deferredBuild) await state.deferredBuild.promise;
      const graph = {
        outputs,
        destroyed: false,
        destroyCount: 0,
        destroy() {
          this.destroyed = true;
          this.destroyCount++;
        },
      };
      state.graphs.push(graph);
      return graph;
    }
  }

  const context = {
    accelerated: true,
    lost,
    opSupportLimits() { return state.supportLimits; },
    async createTensor(descriptor) {
      const tensor = {
        descriptor,
        data: null,
        destroyed: false,
        destroy() { this.destroyed = true; },
      };
      state.tensors.push(tensor);
      return tensor;
    },
    writeTensor(tensor, data) { tensor.data = cloneStorage(data); },
    dispatch(graph, inputs, outputs) {
      for (const [name, tensor] of Object.entries(outputs)) {
        tensor.data = evaluate(graph.outputs[name], inputs);
      }
    },
    async readTensor(tensor) {
      const data = tensor.data;
      return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    },
    destroy() { state.contextDestroyCount++; },
  };

  state.deferOneBuild = () => {
    let resolve;
    const promise = new Promise((done) => { resolve = done; });
    const deferred = {
      promise,
      release() {
        state.deferredBuild = null;
        resolve();
      },
    };
    state.deferredBuild = deferred;
    return deferred;
  };
  state.loseContext = async (message = 'injected WebNN context loss') => {
    loseContext({ message });
    await Promise.resolve();
  };
  return { state, context, Builder: MockMLGraphBuilder };
}

async function withMockBuilder(callback) {
  const previous = globalThis.MLGraphBuilder;
  const mock = makeMockWebNN();
  globalThis.MLGraphBuilder = mock.Builder;
  try {
    return await callback(mock);
  } finally {
    if (previous === undefined) delete globalThis.MLGraphBuilder;
    else globalThis.MLGraphBuilder = previous;
  }
}

async function runtimeFixture(mock, snapshot = dynamicReluSnapshot()) {
  const runtime = new Runtime();
  runtime._addProvider('webnn', new WebNNBackendProvider(new WebNNEngine(mock.context)));
  const compiled = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'webnn', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  return { snapshot, runtime, compiled, context };
}

function resolvedRequest(snapshot, length, offset = 0) {
  const inputs = shaped(length, offset);
  const plan = snapshot.bindShapes(inputs);
  return Object.freeze({
    operation: 'execute',
    commitExecution() {},
    signature: plan.signature,
    inputs: Object.freeze(inputs),
    deviceInputs: Object.freeze({}),
    inputDescriptors: Object.freeze(snapshot.inputNames.map((name) => plan.tensors[name])),
    tensors: plan.tensors,
    outputDescriptors: plan.outputs,
    plan,
    adapters: Object.freeze({ batchSize: null, selectors: Object.freeze([]) }),
    options: Object.freeze({}),
  });
}

test('WebNN builds one exact graph per signature and deterministically bounds its LRU', async () => {
  await withMockBuilder(async (mock) => {
    const { runtime, compiled, context } = await runtimeFixture(mock);
    const firstInput = shaped(2);
    const first = await context.execute(firstInput);
    assert.deepEqual([...await first.output('y').read()], expectedRelu(firstInput));
    assert.equal(first.report.backendReport.specializationCacheHit, false);
    assert.equal(first.report.backendReport.graphBuildCount, 1);

    const repeatedInput = shaped(2, 10);
    const repeated = await context.execute(repeatedInput);
    assert.deepEqual([...await repeated.output('y').read()], expectedRelu(repeatedInput));
    assert.equal(repeated.report.backendReport.specializationCacheHit, true);
    assert.equal(repeated.report.backendReport.graphBuildCount, 1);
    assert.equal(repeated.report.backendReport.graphBuildTimeMs, 0);

    const largeInput = shaped(7);
    const large = await context.execute(largeInput);
    assert.deepEqual([...await large.output('y').read()], expectedRelu(largeInput));
    assert.equal(large.report.backendReport.specializationCacheHit, false);
    assert.equal(large.report.backendReport.graphBuildCount, 2);

    let final = large;
    for (let length = 1; length <= 10; length++) {
      const result = await context.execute(shaped(length, length));
      if (final !== large) await final.close();
      final = result;
    }
    assert.equal(final.report.backendReport.graphBuildCount, 10);
    assert.equal(final.report.backendReport.specializationCacheEntries, 8);
    assert.equal(final.report.backendReport.specializationCacheEvictions, 2);
    assert.ok(final.report.backendReport.specializationCacheMetadataBytes < 1024 * 1024);
    assert.equal(mock.state.graphs.filter((graph) => graph.destroyed).length, 2);

    const rebuilt = await context.execute(shaped(1, 30));
    assert.equal(rebuilt.report.backendReport.specializationCacheHit, false);
    assert.equal(rebuilt.report.backendReport.graphBuildCount, 11);
    assert.equal(rebuilt.report.backendReport.specializationCacheEvictions, 3);

    await context.close();
    assert.ok(mock.state.graphs.every((graph) => graph.destroyed));
    assert.ok(mock.state.graphs.every((graph) => graph.destroyCount === 1));
    await compiled.close();
    await runtime.close();
    assert.equal(mock.state.contextDestroyCount, 1);
    assert.deepEqual([...await first.output('y').read()], expectedRelu(firstInput),
      'result storage remains stable after graph eviction and parent closure');
    await Promise.all([first.close(), repeated.close(), large.close(), final.close(), rebuilt.close()]);
    assert.ok(mock.state.tensors.every((tensor) => tensor.destroyed));
  });
});

test('failed WebNN graph publication leaves the prior cached signature executable', async () => {
  await withMockBuilder(async (mock) => {
    const { runtime, compiled, context } = await runtimeFixture(mock);
    const first = await context.execute(shaped(2));
    mock.state.failNextBuild = true;
    await assert.rejects(context.execute(shaped(5)), /injected WebNN graph build failure/);

    const recoveredInput = shaped(2, 20);
    const recovered = await context.execute(recoveredInput);
    assert.deepEqual([...await recovered.output('y').read()], expectedRelu(recoveredInput));
    assert.equal(recovered.report.backendReport.specializationCacheHit, true);
    assert.equal(recovered.report.backendReport.graphBuildCount, 1,
      'a failed candidate is not published or counted as a built graph');
    assert.equal(recovered.report.backendReport.specializationCacheEntries, 1);

    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all([first.close(), recovered.close()]);
  });
});

test('WebNN coalesces duplicate in-flight builds and closes a pending candidate safely', async () => {
  await withMockBuilder(async (mock) => {
    const snapshot = dynamicReluSnapshot();
    const provider = new WebNNBackendProvider(new WebNNEngine(mock.context));
    const compiled = await provider.compile(createBackendCompileInput(snapshot), {
      operatorFallback: 'forbid',
    });
    const context = compiled.createContext();

    const deferred = mock.state.deferOneBuild();
    const firstExecution = context.execute(resolvedRequest(snapshot, 3));
    const secondExecution = context.execute(resolvedRequest(snapshot, 3, 8));
    await Promise.resolve();
    assert.equal(mock.state.buildCount, 1);
    deferred.release();
    const [first, second] = await Promise.all([firstExecution, secondExecution]);
    assert.equal(first.backendReport.specializationCacheHit, false);
    assert.equal(second.backendReport.specializationCacheHit, true);
    assert.equal(second.backendReport.graphBuildCoalesced, true);
    assert.equal(second.backendReport.coalescedGraphBuildCount, 1);
    assert.equal(first.backendReport.graphBuildCount, 1);

    const closingDeferred = mock.state.deferOneBuild();
    const closingExecution = context.execute(resolvedRequest(snapshot, 4));
    await Promise.resolve();
    const close = context.close();
    closingDeferred.release();
    await assert.rejects(closingExecution, (error) => error?.code === 'HANDLE_DISPOSED');
    await close;
    assert.ok(mock.state.graphs.every((graph) => graph.destroyed));
    await compiled.close();
    provider.close();
  });
});

test('WebNN rejects device rank limits for the complete domain before graph build', async () => {
  await withMockBuilder(async (mock) => {
    mock.state.supportLimits = {
      ...mock.state.supportLimits,
      relu: {
        input: { dataTypes: ['float32'], rankRange: { min: 0, max: 1 } },
        output: { dataTypes: ['float32'], rankRange: { min: 0, max: 1 } },
      },
    };
    const runtime = new Runtime();
    runtime._addProvider('webnn', new WebNNBackendProvider(new WebNNEngine(mock.context)));
    await assert.rejects(runtime.compile(dynamicReluSnapshot(), {
      backend: { mode: 'require', backend: 'webnn', operatorFallback: 'forbid' },
    }), (error) => {
      assert.equal(error.code, 'BACKEND_REQUIRED');
      assert.match(error.report.candidates[0].message, /outside the MLContext support limits/);
      return true;
    });
    assert.equal(mock.state.buildCount, 0);
    await runtime.close();
  });
});

test('WebNN rejects invalid descriptor maxima and device tensor-byte limits before graph build', async () => {
  await withMockBuilder(async (mock) => {
    const provider = new WebNNBackendProvider(new WebNNEngine(mock.context));
    await assert.rejects(
      provider.compile(createBackendCompileInput(dynamicReluSnapshot(0x80000000)), {
        operatorFallback: 'forbid',
      }),
      (error) => error?.code === 'BACKEND_UNSUPPORTED' && /cannot represent/.test(error.message),
    );
    await assert.rejects(
      provider.compile(createBackendCompileInput(dynamicMatrixReluSnapshot(50_000, 50_000)), {
        operatorFallback: 'forbid',
      }),
      (error) => error?.code === 'BACKEND_UNSUPPORTED' &&
        /element count is not a valid WebNN dimension/.test(error.message),
    );

    mock.state.supportLimits = { ...mock.state.supportLimits, maxTensorByteLength: 32 };
    await assert.rejects(
      provider.compile(createBackendCompileInput(dynamicReluSnapshot(12)), {
        operatorFallback: 'forbid',
      }),
      (error) => error?.code === 'BACKEND_UNSUPPORTED' &&
        /maximum tensor bytes 48 exceed/.test(error.message),
    );
    assert.equal(mock.state.buildCount, 0);
    provider.close();
  });
});

test('WebNN rejects retained decode at binding and call time without mutating ordinary state', async () => {
  await withMockBuilder(async (mock) => {
    const { runtime, compiled, context } = await runtimeFixture(mock);
    await assert.rejects(compiled.createContext({
      decode: { changedInputs: ['x'], rowMode: 'disabled', requireIncremental: true },
    }), (error) => error?.code === 'BACKEND_UNSUPPORTED' && /does not support retained/.test(error.message));
    assert.equal(mock.state.buildCount, 0,
      'decode rejection at binding must not construct a WebNN graph');

    await assert.rejects(context.decode.seed(shaped(3)),
      (error) => error?.code === 'BACKEND_UNSUPPORTED' && /does not support retained/.test(error.message));
    assert.equal(mock.state.buildCount, 0,
      'decode rejection at call time must happen before graph publication');

    const input = shaped(3, 7);
    const ordinary = await context.execute(input);
    assert.deepEqual([...await ordinary.output('y').read()], expectedRelu(input));
    assert.equal(mock.state.buildCount, 1,
      'an unsupported decode call must not poison the ordinary execution context');

    await context.close();
    await compiled.close();
    await runtime.close();
    await ordinary.close();
  });
});

test('WebNN context loss is terminal and releases cached graphs without invalidating results', async () => {
  await withMockBuilder(async (mock) => {
    const input = shaped(3);
    const { runtime, compiled, context } = await runtimeFixture(mock);
    const result = await context.execute(input);
    await mock.state.loseContext();
    await assert.rejects(context.execute(shaped(3, 5)), (error) => {
      assert.equal(error.code, 'DEVICE_LOST');
      assert.match(error.message, /injected WebNN context loss/);
      return true;
    });
    assert.ok(mock.state.graphs.every((graph) => graph.destroyed));
    await context.close();
    await compiled.close();
    await runtime.close();
    assert.deepEqual([...await result.output('y').read()], expectedRelu(input));
    await result.close();
  });
});
