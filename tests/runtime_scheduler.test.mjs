import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';

import {
  ExecutionMode,
  Model,
  Runtime,
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  createBackendProviderBatchContract,
  createBackendProviderCapabilities,
  executionModes,
  parseGraphDocument,
} from '../ts/index.js';
import { CPUBackendProvider } from '../ts/backends/CPUBackendProvider.js';
import { createBackendCompileInput } from '../ts/backends/BackendProvider.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { RuntimeScheduler } from '../ts/core/RuntimeScheduler.js';
import { InvariantResourceStore } from '../ts/backends/InvariantResources.js';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, refuse) => {
    resolve = accept;
    reject = refuse;
  });
  return { promise, resolve, reject };
}

const nextTurn = () => new Promise((resolve) => setImmediate(resolve));
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const fakeResult = () => Object.freeze({ close: async () => {} });
const preparedRoute = ({
  compatibilityToken = Object.freeze({}),
  resourceDomain = Object.freeze({}),
  deviceEpoch = Object.freeze({}),
  preparedPlan = Object.freeze({}),
  maxBatchSize = 1,
  inputBytes = 8,
} = {}) => Object.freeze({
  compatibilityToken,
  resourceDomain,
  deviceEpoch,
  preparedPlan,
  maxBatchSize,
  inputBytes,
});

function batchedIdentitySnapshot(maxBatch = 8) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: maxBatch } },
    inputs: { x: { dtype: 'float32', shape: ['B', 2] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 2] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function softmaxVectorSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 1, max: 2 } },
    inputs: { x: { dtype: 'float32', shape: ['S'] } },
    nodes: [{
      id: 'softmax',
      opType: 'Softmax',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['S'] } },
      params: { axis: -1 },
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function batchedMatMulSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 8 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 2] } },
    nodes: [{
      id: 'matmul',
      opType: 'MatMul',
      inputs: { input: 'x', weight: 'w' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 2] } },
      params: {},
    }],
    outputs: ['y'],
  }, [{ name: 'w', dtype: 'float32', shape: [2, 2] }]);
  return Model.capture({
    graph,
    weights: {
      w: Object.freeze({
        name: 'w', dtype: 'float32', shape: Object.freeze([2, 2]),
        data: Float32Array.of(1, 0, 0, 1),
      }),
    },
  });
}

function manyShapeIdentitySnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 8 },
      S: { min: 1, max: 1025 },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 'S'] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 'S'] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function reservedInputNameSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: Object.fromEntries([
      ['__proto__', { dtype: 'float32', shape: ['B', 2] }],
    ]),
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: '__proto__' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 2] } },
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
    maximumTensorBytes: 256,
    maximumResidentBytes: 512,
    resourceLimitBytes: 4096,
  });
}

function schedulerProvider({
  blockFirst = null,
  blockCalls = null,
  onExecute = null,
  onContextClose = null,
  onCompiledClose = null,
  onProviderClose = null,
  flipCompatibilityAfterRouteCalls = null,
  deviceOutputs = false,
  deviceReleaseThrows = false,
} = {}) {
  const executions = [];
  const routePlans = [];
  let deviceReadCalls = 0;
  let deviceReleaseCalls = 0;
  let contextCount = 0;
  const resourceDomain = Object.freeze({});
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: 'fixture',
    capabilities: createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: deviceOutputs ? 'device' : 'host',
      dynamicShapeDomain: 'full',
    }),
    async compile(input) {
      const compatibilityTokens = new Map();
      const invariantResources = new InvariantResourceStore(() => 0);
      return {
        backendName: 'fixture',
        invariantResources,
        batchContract: createBackendProviderBatchContract('single-invocation', {
          independentBatch: 'compiler-proved/v1',
        }),
        prepareBatchRoute(plan) {
          routePlans.push(plan);
          let compatibilityToken = compatibilityTokens.get(plan.signature);
          if (!compatibilityToken) {
            compatibilityToken = Object.freeze({});
            compatibilityTokens.set(plan.signature, compatibilityToken);
          }
          if (flipCompatibilityAfterRouteCalls !== null &&
              routePlans.length > flipCompatibilityAfterRouteCalls) {
            compatibilityToken = Object.freeze({});
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
          shapeDomain: shapeDomain(input),
          batchSemantics: input.batchSemantics,
        }),
        createContext() {
          const contextId = ++contextCount;
          return {
            backendName: 'fixture',
            async execute(request) {
              const call = executions.length;
              const inputName = Object.keys(request.inputs)[0];
              const inputView = request.inputs[inputName];
              executions.push(Object.freeze({
                contextId,
                shape: Object.freeze([...inputView.shape]),
                values: Object.freeze([...inputView.data]),
                plan: request.plan,
              }));
              onExecute?.(call);
              const blocker = blockCalls?.[call] ?? (call === 0 ? blockFirst : null);
              if (blocker) await blocker;
              const data = Float32Array.from(inputView.data, (value) => value * 2);
              const output = deviceOutputs
                ? Object.freeze({
                    name: 'y',
                    shape: request.outputDescriptors[0].shape,
                    dtype: 'float32',
                    location: 'device',
                    logicalSizeBytes: data.byteLength,
                    deviceBuffer: Object.freeze({ size: data.byteLength }),
                    async read() {
                      deviceReadCalls++;
                      return new Float32Array(data);
                    },
                    release() {
                      deviceReleaseCalls++;
                      if (deviceReleaseThrows) throw new Error('device result release failed');
                    },
                  })
                : Object.freeze({
                    name: 'y',
                    shape: request.outputDescriptors[0].shape,
                    dtype: 'float32',
                    location: 'host',
                    ownership: 'transfer',
                    data,
                  });
              return Object.freeze({ outputs: Object.freeze([output]) });
            },
            close() { return onContextClose?.(); },
          };
        },
        close() { return onCompiledClose?.(); },
      };
    },
    close() { return onProviderClose?.(); },
  };
  return {
    provider,
    executions,
    routePlans,
    get contextCount() { return contextCount; },
    get deviceReadCalls() { return deviceReadCalls; },
    get deviceReleaseCalls() { return deviceReleaseCalls; },
  };
}

async function compile(runtime, snapshot = batchedIdentitySnapshot()) {
  return runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });
}

const input = (left, right) => Object.freeze({
  x: Object.freeze({ data: Float32Array.of(left, right), shape: Object.freeze([1, 2]) }),
});

test('public execution modes come from the generated protobuf contract', async () => {
  assert.deepEqual(executionModes, ['direct', 'scheduled']);
  assert.equal(Object.isFrozen(executionModes), true);
  assert.throws(() => {
    executionModes[ExecutionMode.Scheduled] = 'forged';
  }, TypeError);
  assert.equal(executionModes[ExecutionMode.Direct], 'direct');
  assert.equal(executionModes[ExecutionMode.Scheduled], 'scheduled');

  const runtime = new Runtime();
  assert.equal(runtime.inspectExecution().defaultMode, executionModes[ExecutionMode.Scheduled]);
  await runtime.close();
  assert.throws(
    () => new Runtime({ execution: { mode: 'invalid-mode' } }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.throws(
    () => new Runtime({ execution: { scheduler: { maxBatchDelayMs: -1 } } }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
});

test('DIRECT executes B=1 without allocating RuntimeScheduler state', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({ execution: { mode: 'direct' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  assert.deepEqual(runtime.inspectExecution(), {
    defaultMode: 'direct', schedulerAllocated: false,
    admittedRequests: 0, admittedInputBytes: 0,
    admittingInputBytes: 0,
    stagingInputBytes: 0, totalInputBytes: 0,
    inputSnapshotCopies: 0,
    retainedResults: 0, retainedOutputBytes: 0,
    maxRetainedResults: 64, maxRetainedOutputBytes: 64 * 1024 * 1024,
  });
  const result = await runtime.run(model, input(2, 3));
  assert.deepEqual(await result.output('y').read(), Float32Array.of(4, 6));
  assert.deepEqual(fixture.executions.map(({ shape }) => shape), [[1, 2]]);
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);

  await result.close();
  await model.close();
  await runtime.close();
});

test('DIRECT output reservations backpressure until close and survive Runtime close', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: {
      mode: 'direct',
      results: { maxRetainedResults: 1, maxRetainedOutputBytes: 8 },
    },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const first = await runtime.run(model, input(1, 2));
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);
  assert.equal(runtime.inspectExecution().retainedResults, 1);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 8);
  await assert.rejects(
    runtime.run(model, input(3, 4)),
    (error) => error.code === 'OVERLOADED',
  );
  assert.equal(fixture.executions.length, 1,
    'exact result admission must fail before a second provider invocation');

  await first.close();
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);
  const surviving = await runtime.run(model, input(5, 6));
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);

  await model.close();
  await runtime.close();
  assert.deepEqual(await surviving.output('y').read(), Float32Array.of(10, 12));
  assert.equal(runtime.inspectExecution().retainedResults, 1);
  await surviving.close();
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);
});

test('cold DIRECT reserves exact output capacity before provider context creation', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: {
      mode: 'direct',
      results: { maxRetainedResults: 1, maxRetainedOutputBytes: 8 },
    },
  });
  runtime._addProvider('fixture', fixture.provider);
  const firstModel = await compile(runtime);
  const coldModel = await compile(runtime);

  const retained = await runtime.run(firstModel, input(1, 2));
  assert.equal(fixture.contextCount, 1);
  assert.equal(runtime.inspectExecution().retainedResults, 1);
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);

  const invalid = Object.freeze({
    x: Object.freeze({ data: Float32Array.of(9), shape: Object.freeze([1, 1]) }),
  });
  await assert.rejects(
    runtime.run(coldModel, invalid),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  await assert.rejects(
    runtime.run(coldModel, input(3, 4)),
    (error) => error.code === 'OVERLOADED',
  );
  assert.equal(fixture.contextCount, 1,
    'invalid/full-ledger cold calls never create or cache a provider context');
  assert.equal(runtime.inspectExecution().retainedResults, 1);
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);

  await retained.close();
  const admitted = await runtime.run(coldModel, input(3, 4));
  assert.equal(fixture.contextCount, 2);
  assert.deepEqual(await admitted.output('y').read(), Float32Array.of(6, 8));
  await admitted.close();
  const reused = await runtime.run(coldModel, input(5, 6));
  assert.equal(fixture.contextCount, 2, 'the admitted cold model caches exactly one context');
  await reused.close();

  await firstModel.close();
  await coldModel.close();
  await runtime.close();
});

test('public ExecutionContext executions share the Runtime result budget', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: { results: { maxRetainedResults: 1, maxRetainedOutputBytes: 8 } },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const context = await model.createContext();

  const first = await context.execute(input(1, 2));
  await assert.rejects(
    context.execute(input(3, 4)),
    (error) => error.code === 'OVERLOADED',
  );
  assert.equal(fixture.executions.length, 1);
  await first.close();
  const second = await context.execute(input(3, 4));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(6, 8));

  await second.close();
  await context.close();
  await model.close();
  await runtime.close();
});

test('DIRECT Runtime rejects submit without allocating scheduler state', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({ execution: { mode: 'direct' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  assert.throws(
    () => runtime.submit(model, input(1, 2)),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  await assert.rejects(
    runtime.run(model, input(1, 2), { mode: 'scheduled' }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);

  await model.close();
  await runtime.close();
});

test('Runtime snapshots execution scheduler policy once without eager allocation', async () => {
  const gate = deferred();
  const fixture = schedulerProvider({ blockFirst: gate.promise });
  const reads = new Map();
  const schedulerValues = {
    maxRequests: 1,
    maxInputBytes: 8,
    maxBatchSize: 2,
    maxBatchDelayMs: 0,
    maxPriorityBurst: 1,
  };
  const schedulerOptions = {};
  for (const name of Object.keys(schedulerValues)) {
    Object.defineProperty(schedulerOptions, name, {
      enumerable: true,
      get() {
        reads.set(`scheduler.${name}`, (reads.get(`scheduler.${name}`) ?? 0) + 1);
        return schedulerValues[name];
      },
    });
  }
  const resultValues = {
    maxRetainedResults: 1,
    maxRetainedOutputBytes: 8,
  };
  const resultOptions = {};
  for (const name of Object.keys(resultValues)) {
    Object.defineProperty(resultOptions, name, {
      enumerable: true,
      get() {
        reads.set(`results.${name}`, (reads.get(`results.${name}`) ?? 0) + 1);
        return resultValues[name];
      },
    });
  }
  const execution = {};
  Object.defineProperties(execution, {
    mode: {
      enumerable: true,
      get() {
        reads.set('execution.mode', (reads.get('execution.mode') ?? 0) + 1);
        return 'scheduled';
      },
    },
    scheduler: {
      enumerable: true,
      get() {
        reads.set('execution.scheduler', (reads.get('execution.scheduler') ?? 0) + 1);
        return schedulerOptions;
      },
    },
    results: {
      enumerable: true,
      get() {
        reads.set('execution.results', (reads.get('execution.results') ?? 0) + 1);
        return resultOptions;
      },
    },
  });
  const runtime = new Runtime({ execution });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);
  assert.deepEqual(Object.fromEntries(reads), {
    'execution.mode': 1,
    'execution.scheduler': 1,
    'execution.results': 1,
    'scheduler.maxBatchDelayMs': 1,
    'scheduler.maxRequests': 1,
    'scheduler.maxInputBytes': 1,
    'scheduler.maxBatchSize': 1,
    'scheduler.maxPriorityBurst': 1,
    'results.maxRetainedResults': 1,
    'results.maxRetainedOutputBytes': 1,
  });

  schedulerValues.maxRequests = 2;
  schedulerValues.maxInputBytes = 16;
  resultValues.maxRetainedResults = 2;
  resultValues.maxRetainedOutputBytes = 16;
  const accepted = runtime.submit(model, input(1, 2));
  assert.throws(
    () => runtime.submit(model, input(3, 4)),
    (error) => error.code === 'OVERLOADED',
  );
  assert.equal(runtime.inspectExecution().admittedRequests, 1);
  assert.equal(reads.get('execution.scheduler'), 1);
  assert.equal(reads.get('execution.results'), 1);
  assert.equal(runtime.inspectExecution().maxRetainedResults, 1);
  assert.equal(runtime.inspectExecution().maxRetainedOutputBytes, 8);

  gate.resolve();
  const result = await accepted.result;
  await result.close();
  await model.close();
  await runtime.close();
});

test('DIRECT uses AbortSignal intrinsics and hostile cleanup cannot strand its result', async () => {
  const gate = deferred();
  const fixture = schedulerProvider({ blockFirst: gate.promise });
  const runtime = new Runtime({
    execution: {
      mode: 'scheduled',
      results: { maxRetainedResults: 1, maxRetainedOutputBytes: 8 },
    },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const firstController = new AbortController();
  const laterController = new AbortController();
  const added = [];
  const removed = [];
  const originalAdd = firstController.signal.addEventListener;
  Object.defineProperties(firstController.signal, {
    addEventListener: {
      configurable: true,
      value(type, listener, options) {
        added.push([type, listener]);
        return originalAdd.call(this, type, listener, options);
      },
    },
    removeEventListener: {
      configurable: true,
      value(type, listener) {
        removed.push([type, listener]);
        throw new Error('hostile signal cleanup');
      },
    },
  });
  const options = {
    mode: 'direct',
    signal: firstController.signal,
    freshness: 'all',
    deadlineMonotonicMs: performance.now() + 10_000,
  };
  const running = runtime.run(model, input(1, 2), options);
  await nextTurn();
  options.mode = 'scheduled';
  options.signal = laterController.signal;
  options.freshness = 'drop-if-late';
  options.deadlineMonotonicMs = 0;

  const busyOptions = { mode: 'direct' };
  const busy = runtime.run(model, input(3, 4), busyOptions);
  busyOptions.mode = 'scheduled';
  await assert.rejects(busy, (error) => error.code === 'BUSY');
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);

  laterController.abort();
  gate.resolve();
  const result = await running;
  assert.deepEqual(await result.output('y').read(), Float32Array.of(2, 4));
  assert.equal(added.length, 0, 'the shadowable addEventListener is never invoked');
  assert.equal(removed.length, 0, 'the shadowable removeEventListener is never invoked');
  assert.equal(runtime.inspectExecution().retainedResults, 1);

  await result.close();
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  await model.close();
  await runtime.close();
});

test('scheduled signal registration cannot cancel reentrantly or steal a LATEST donor slot',
  async () => {
    const fixture = schedulerProvider();
    const runtime = new Runtime({
      execution: {
        mode: 'scheduled',
        scheduler: { maxBatchDelayMs: 20, maxRequests: 1, maxInputBytes: 16 },
        results: { maxRetainedResults: 1, maxRetainedOutputBytes: 16 },
      },
    });
    runtime._addProvider('fixture', fixture.provider);
    const model = await compile(runtime);
    const oldFrame = runtime.submit(model, input(1, 2), {
      freshness: 'latest', streamKey: 'camera/front',
    });
    const oldFailure = oldFrame.result.catch((error) => error);

    const controller = new AbortController();
    let abortedReads = 0;
    let addCalls = 0;
    let removeCalls = 0;
    Object.defineProperties(controller.signal, {
      aborted: {
        configurable: true,
        get() {
          abortedReads++;
          return false;
        },
      },
      addEventListener: {
        configurable: true,
        value(_type, listener) {
          addCalls++;
          listener.call(this);
        },
      },
      removeEventListener: {
        configurable: true,
        value() {
          removeCalls++;
          throw new Error('hostile signal cleanup');
        },
      },
    });

    const newFrame = runtime.submit(model, input(7, 8), {
      freshness: 'latest', streamKey: 'camera/front',
      signal: controller.signal,
    });
    assert.equal(abortedReads, 0, 'the shadowable aborted getter is never invoked');
    assert.equal(addCalls, 0, 'registration bypasses the shadowable instance method');
    assert.equal(oldFrame.state, 'superseded');
    assert.equal(newFrame.state, 'queued');
    assert.equal(runtime.inspectExecution().retainedResults, 1);
    assert.equal(runtime.inspectExecution().retainedOutputBytes, 8);

    const [oldError, result] = await Promise.all([oldFailure, newFrame.result]);
    assert.equal(oldError.code, 'SUPERSEDED');
    assert.deepEqual(await result.output('y').read(), Float32Array.of(14, 16));
    assert.equal(removeCalls, 0, 'settlement bypasses the shadowable cleanup method');
    assert.equal(runtime.inspectExecution().retainedResults, 1);

    await result.close();
    assert.equal(runtime.inspectExecution().retainedResults, 0);
    assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);
    await model.close();
    await runtime.close();
  });

test('scheduled submit options are captured once and use the v1 priority range', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const reads = new Map();
  const values = {
    priority: 1000, deadlineMonotonicMs: undefined,
    freshness: 'all', streamKey: null, signal: null,
  };
  const options = {};
  for (const [name, value] of Object.entries(values)) {
    Object.defineProperty(options, name, {
      enumerable: true,
      get() {
        reads.set(name, (reads.get(name) ?? 0) + 1);
        return value;
      },
    });
  }
  Object.defineProperty(options, 'mode', {
    enumerable: true,
    get() {
      assert.fail('submit must not read a request-level execution mode');
    },
  });
  const result = await runtime.submit(model, input(2, 3), options).result;
  assert.deepEqual(Object.fromEntries(reads), Object.fromEntries(
    Object.keys(values).map((name) => [name, 1]),
  ));
  assert.throws(
    () => runtime.submit(model, input(4, 5), { priority: 1001 }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  await assert.rejects(
    runtime.run(model, input(4, 5), { mode: 'direct', priority: -1001 }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );

  await result.close();
  await model.close();
  await runtime.close();
});

test('SCHEDULED Runtime may narrow run to DIRECT while submit is always scheduled', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({ execution: { mode: 'scheduled' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const directResult = await runtime.run(model, input(2, 3), { mode: 'direct' });
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);
  await directResult.close();

  const scheduledResult = await runtime.submit(model, input(4, 5)).result;
  assert.deepEqual(await scheduledResult.output('y').read(), Float32Array.of(8, 10));
  assert.equal(runtime.inspectExecution().schedulerAllocated, true);
  await scheduledResult.close();
  await model.close();
  await runtime.close();
});

test('SCHEDULED combines compatible calls into one true provider B>1 invocation', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({ execution: { mode: 'scheduled' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const first = runtime.submit(model, input(1, 2));
  const second = runtime.submit(model, input(7, 9));
  const [firstResult, secondResult] = await Promise.all([first.result, second.result]);

  assert.equal(fixture.executions.length, 1);
  assert.deepEqual(fixture.executions[0].shape, [2, 2]);
  assert.deepEqual(await firstResult.output('y').read(), Float32Array.of(2, 4));
  assert.deepEqual(await secondResult.output('y').read(), Float32Array.of(14, 18));
  const scheduling = [firstResult, secondResult].map(
    (result) => result.report.backendReport.scheduling,
  );
  assert.deepEqual(scheduling.map((report) => report.batchSize), [2, 2]);
  assert.deepEqual(scheduling.map((report) => report.lane), [0, 1]);
  assert.deepEqual(scheduling.map((report) => report.dispatchOwnerLane), [0, 0]);
  assert.deepEqual(scheduling.map((report) => report.invocationAccounting),
    ['dispatch-owner/v1', 'dispatch-owner/v1']);
  assert.equal(scheduling[0].dispatchId, scheduling[1].dispatchId);
  assert.notEqual(scheduling[0].dispatchId, firstResult.report.executionId);
  assert.notEqual(firstResult.report.executionId, secondResult.report.executionId);
  assert.deepEqual(scheduling.map((report) => report.trueBackendInvocations), [1, 0]);
  assert.equal(scheduling.reduce(
    (total, report) => total + report.trueBackendInvocations, 0,
  ), 1);
  assert.equal(runtime.inspectExecution().admittedRequests, 0);
  assert.equal(runtime.inspectExecution().admittedInputBytes, 0);

  await Promise.all([firstResult.close(), secondResult.close()]);
  await model.close();
  await runtime.close();
});

test('B>1 device output is read back once and shared bytes release with the last lane', async () => {
  const fixture = schedulerProvider({ deviceOutputs: true });
  const runtime = new Runtime({
    execution: {
      mode: 'scheduled',
      results: { maxRetainedResults: 2, maxRetainedOutputBytes: 16 },
    },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const [first, second] = await Promise.all([
    runtime.submit(model, input(1, 2)).result,
    runtime.submit(model, input(3, 4)).result,
  ]);

  assert.equal(fixture.executions.length, 1);
  assert.equal(fixture.deviceReadCalls, 1,
    'the unpublished physical B=2 result performs one device readback');
  assert.equal(fixture.deviceReleaseCalls, 1);
  assert.deepEqual(await first.output('y').read(), Float32Array.of(2, 4));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(6, 8));
  assert.equal(fixture.deviceReadCalls, 1,
    'public lane reads use host views over the single readback backing');
  assert.equal(runtime.inspectExecution().retainedResults, 2);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 16);

  await first.close();
  assert.equal(runtime.inspectExecution().retainedResults, 1);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 16,
    'one live lane keeps the shared B=2 backing charged in full');
  assert.throws(
    () => runtime.submit(model, input(5, 6)),
    (error) => error.code === 'OVERLOADED',
  );
  await second.close();
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);

  const afterRelease = await runtime.submit(model, input(5, 6)).result;
  await afterRelease.close();
  await model.close();
  await runtime.close();
});

test('a failing physical batch-result close retires every unpublished split lane', async () => {
  const fixture = schedulerProvider({ deviceOutputs: true, deviceReleaseThrows: true });
  const runtime = new Runtime({
    execution: { results: { maxRetainedResults: 2, maxRetainedOutputBytes: 16 } },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const first = runtime.submit(model, input(1, 2));
  const second = runtime.submit(model, input(3, 4));

  await Promise.all([
    assert.rejects(first.result, (error) => error.code === 'EXECUTION_FAILED'),
    assert.rejects(second.result, (error) => error.code === 'EXECUTION_FAILED'),
  ]);
  assert.equal(fixture.deviceReadCalls, 1);
  assert.equal(fixture.deviceReleaseCalls, 1);
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);

  await model.close().catch(() => undefined);
  await runtime.close().catch(() => undefined);
});

test('scheduled output capacity rejects before cloning and queued cancel releases it', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: {
      mode: 'scheduled',
      scheduler: { maxBatchDelayMs: 1_000 },
      results: { maxRetainedResults: 1, maxRetainedOutputBytes: 8 },
    },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const queued = runtime.submit(model, input(1, 2));
  assert.equal(runtime.inspectExecution().inputSnapshotCopies, 1);
  assert.equal(runtime.inspectExecution().retainedResults, 1);
  assert.throws(
    () => runtime.submit(model, input(3, 4)),
    (error) => error.code === 'OVERLOADED',
  );
  assert.equal(runtime.inspectExecution().inputSnapshotCopies, 1,
    'output admission rejects before a second payload snapshot');
  assert.equal(fixture.executions.length, 0);

  assert.equal(queued.cancel(), true);
  await assert.rejects(queued.result, (error) => error.code === 'CANCELLED');
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);
  await model.close();
  await runtime.close();
});

test('stacked input staging stays inside maxInputBytes and falls back to B=1', async () => {
  const batchGate = deferred();
  const batchFixture = schedulerProvider({ blockFirst: batchGate.promise });
  const batchRuntime = new Runtime({
    execution: { scheduler: { maxInputBytes: 32 } },
  });
  batchRuntime._addProvider('fixture', batchFixture.provider);
  const batchModel = await compile(batchRuntime);
  const batchHandles = [
    batchRuntime.submit(batchModel, input(1, 2)),
    batchRuntime.submit(batchModel, input(3, 4)),
  ];
  await nextTurn();
  assert.deepEqual(batchFixture.executions.map(({ shape }) => shape), [[2, 2]]);
  assert.deepEqual(batchRuntime.inspectExecution(), {
    defaultMode: 'scheduled', schedulerAllocated: true,
    admittedRequests: 2, admittedInputBytes: 16,
    admittingInputBytes: 0,
    stagingInputBytes: 16, totalInputBytes: 32,
    inputSnapshotCopies: 2,
    retainedResults: 2, retainedOutputBytes: 16,
    maxRetainedResults: 64, maxRetainedOutputBytes: 64 * 1024 * 1024,
  });
  batchGate.resolve();
  const batchResults = await Promise.all(batchHandles.map((handle) => handle.result));
  assert.equal(batchRuntime.inspectExecution().stagingInputBytes, 0);
  assert.equal(batchRuntime.inspectExecution().totalInputBytes, 0);
  await Promise.all(batchResults.map((result) => result.close()));
  await batchModel.close();
  await batchRuntime.close();

  const fallbackGate = deferred();
  const fallbackFixture = schedulerProvider({ blockFirst: fallbackGate.promise });
  const fallbackRuntime = new Runtime({
    execution: { scheduler: { maxInputBytes: 24 } },
  });
  fallbackRuntime._addProvider('fixture', fallbackFixture.provider);
  const fallbackModel = await compile(fallbackRuntime);
  const fallbackHandles = [
    fallbackRuntime.submit(fallbackModel, input(5, 6)),
    fallbackRuntime.submit(fallbackModel, input(7, 8)),
  ];
  await nextTurn();
  assert.deepEqual(fallbackFixture.executions.map(({ shape }) => shape), [[1, 2]]);
  assert.equal(fallbackRuntime.inspectExecution().admittedInputBytes, 16);
  assert.equal(fallbackRuntime.inspectExecution().stagingInputBytes, 0);
  assert.equal(fallbackRuntime.inspectExecution().totalInputBytes, 16);
  fallbackGate.resolve();
  const fallbackResults = await Promise.all(
    fallbackHandles.map((handle) => handle.result),
  );
  assert.deepEqual(fallbackFixture.executions.map(({ shape }) => shape), [[1, 2], [1, 2]]);
  await Promise.all(fallbackResults.map((result) => result.close()));
  await fallbackModel.close();
  await fallbackRuntime.close();
});

test('scheduled execution reuses prepared B=1 plans and binds a combined B=N plan once', async () => {
  const snapshot = batchedIdentitySnapshot();
  const originalBindShapes = Model.prototype.bindShapes;
  const boundPlans = [];
  Model.prototype.bindShapes = function bindShapesSpy(inputs, bankResidency) {
    const plan = originalBindShapes.call(this, inputs, bankResidency);
    if (this === snapshot) boundPlans.push(plan);
    return plan;
  };

  const fixture = schedulerProvider();
  const runtime = new Runtime({ execution: { mode: 'scheduled' } });
  runtime._addProvider('fixture', fixture.provider);
  let model = null;
  const results = [];
  try {
    model = await compile(runtime, snapshot);
    const first = runtime.submit(model, input(1, 2));
    const second = runtime.submit(model, input(3, 4));
    results.push(...await Promise.all([first.result, second.result]));

    assert.equal(boundPlans.length, 3,
      'two prepared B=1 plans plus one provider B=2 plan; split must not bind again');
    assert.equal(fixture.executions[0].plan, boundPlans[2],
      'the provider receives the one resolved combined plan');
    assert.deepEqual(fixture.routePlans.map((plan) => plan.symbols.B), [1, 1, 1, 1, 2],
      'the provider re-attests each admission plan before the actual combined plan');
    assert.deepEqual(
      results.map((result) => result.report.shapeSignature),
      boundPlans.slice(0, 2).map((plan) => plan.signature),
    );

    const third = await runtime.submit(model, input(5, 6)).result;
    results.push(third);
    assert.equal(boundPlans.length, 4,
      'a singleton execution must consume its prepared plan without rebinding');
    assert.equal(fixture.executions[1].plan, boundPlans[3]);
    assert.deepEqual(fixture.routePlans.map((plan) => plan.symbols.B), [1, 1, 1, 1, 2, 1, 1],
      'a singleton re-attests the provider epoch without resolving its shape again');
    assert.equal(third.report.shapeBindTimeMs, 0);
  } finally {
    Model.prototype.bindShapes = originalBindShapes;
    await Promise.all(results.map((result) => result.close()));
    await model?.close();
    await runtime.close();
  }
});

test('dispatch rejects a provider compatibility token changed after admission', async () => {
  const fixture = schedulerProvider({ flipCompatibilityAfterRouteCalls: 2 });
  const runtime = new Runtime({ execution: { mode: 'scheduled' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const first = runtime.submit(model, input(1, 2));
  const second = runtime.submit(model, input(3, 4));
  await assert.rejects(first.result, (error) => error.code === 'ABI_UNSUPPORTED');
  await assert.rejects(second.result, (error) => error.code === 'ABI_UNSUPPORTED');
  assert.equal(fixture.executions.length, 0,
    'route identity must be re-attested before stacking or provider execution');
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);

  await model.close();
  await runtime.close();
});

test('close preserves the first route error while still tearing down compiled and provider state', async () => {
  const closures = [];
  const fixture = schedulerProvider({
    onContextClose() {
      closures.push('context');
      throw new Error('first route close failure');
    },
    onCompiledClose() {
      closures.push('compiled');
      throw new Error('later compiled close failure');
    },
    onProviderClose() {
      closures.push('provider');
      throw new Error('later provider close failure');
    },
  });
  const runtime = new Runtime({ execution: { mode: 'scheduled' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const result = await runtime.submit(model, input(1, 2)).result;
  await result.close();

  const runtimeClose = runtime.close().catch((error) => error);
  const compiledClose = model.close().catch((error) => error);
  const [runtimeFailure, compiledFailure] = await Promise.all([runtimeClose, compiledClose]);

  assert.equal(runtimeFailure.code, 'EXECUTION_FAILED');
  assert.match(runtimeFailure.message, /first route close failure/);
  assert.equal(compiledFailure.code, 'EXECUTION_FAILED');
  assert.match(compiledFailure.message, /first route close failure/);
  assert.deepEqual(closures, ['context', 'compiled', 'provider']);
});

test('built-in CPU consumes the coalesced explicit batch in one provider invocation', async () => {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const snapshot = batchedIdentitySnapshot();
  const model = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  assert.equal(model.report.batchSemantics.supported, true);
  assert.equal(model.report.batchSemantics.coveredNodes, 1);
  assert.equal(model.report.batchSemantics.graphFingerprint,
    snapshot.definitionFingerprint);

  const [first, second] = await Promise.all([
    runtime.submit(model, input(3, 4)).result,
    runtime.submit(model, input(8, 9)).result,
  ]);
  assert.deepEqual(await first.output('y').read(), Float32Array.of(3, 4));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(8, 9));
  assert.equal(first.report.backendReport.scheduling.batchSize, 2);
  assert.equal(first.report.backendReport.scheduling.trueBackendInvocations, 1);

  await Promise.all([first.close(), second.close()]);
  await model.close();
  await runtime.close();
});

test('a source domain capped at B=1 remains two physical invocations', async () => {
  const runtime = new Runtime();
  const fixture = schedulerProvider();
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime, batchedIdentitySnapshot(1));

  const [first, second] = await Promise.all([
    runtime.submit(model, input(3, 4)).result,
    runtime.submit(model, input(8, 9)).result,
  ]);
  assert.equal(model.report.batchSemantics.supported, true,
    'semantic independence does not widen the authored B bound');
  assert.equal(first.report.backendReport?.scheduling?.batchSize ?? 1, 1);
  assert.equal(second.report.backendReport?.scheduling?.batchSize ?? 1, 1);
  assert.equal(first.report.backendReport?.scheduling?.trueBackendInvocations ?? 1, 1);
  assert.equal(second.report.backendReport?.scheduling?.trueBackendInvocations ?? 1, 1);
  assert.equal(fixture.executions.length, 2);
  assert.deepEqual(fixture.executions.map(({ shape }) => shape), [[1, 2], [1, 2]]);

  await Promise.all([first.close(), second.close()]);
  await model.close();
  await runtime.close();
});

test('built-in batch contract attests only graphs with proved independent rows', async () => {
  const provider = new CPUBackendProvider(new CPUEngine());
  const identityInput = createBackendCompileInput(batchedIdentitySnapshot());
  const softmaxInput = createBackendCompileInput(softmaxVectorSnapshot());
  const identity = await provider.compile(identityInput, {
    operatorFallback: 'forbid',
  });
  const softmax = await provider.compile(softmaxInput, {
    operatorFallback: 'forbid',
  });

  assert.equal(identity.batchContract.independentBatch, 'compiler-proved/v1');
  assert.equal(softmax.batchContract.independentBatch, 'unsupported');
  assert.equal(identity.compilationEvidence.batchSemantics.supported, true);
  assert.equal(identity.compilationEvidence.batchSemantics, identityInput.batchSemantics);
  assert.equal(identity.compilationEvidence.batchSemantics.coveredNodes, 1);
  assert.equal(identity.compilationEvidence.batchSemantics.graphFingerprint,
    identity.compilationEvidence.shapeDomain.graphFingerprint);
  assert.equal(softmax.compilationEvidence.batchSemantics.supported, false);
  assert.equal(softmax.compilationEvidence.batchSemantics, softmaxInput.batchSemantics);
  assert.equal(softmax.compilationEvidence.batchSemantics.failedNode, 'softmax');

  await Promise.all([identity.close(), softmax.close()]);
  await provider.close();
});

test('a shared leading symbol does not batch Softmax across independent requests', async () => {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const model = await runtime.compile(softmaxVectorSnapshot(), {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  assert.equal(model.report.batchSemantics.supported, false);
  assert.equal(model.report.batchSemantics.failedNode, 'softmax');

  // Independent-row rejection is only a scheduler coalescing gate. One
  // caller-authored B=2 request retains the graph's intentional cross-B
  // Softmax semantics and remains a legal explicit bulk invocation.
  const bulk = await runtime.run(model, {
    x: { data: Float32Array.of(0, 0), shape: [2] },
  }, { mode: executionModes[ExecutionMode.Direct] });
  assert.deepEqual(await bulk.output('y').read(), Float32Array.of(0.5, 0.5));
  assert.equal(runtime.inspectExecution().schedulerAllocated, false);
  await bulk.close();

  const scalar = (value) => Object.freeze({
    x: Object.freeze({ data: Float32Array.of(value), shape: Object.freeze([1]) }),
  });

  const [first, second] = await Promise.all([
    runtime.submit(model, scalar(0)).result,
    runtime.submit(model, scalar(10)).result,
  ]);
  assert.deepEqual(await first.output('y').read(), Float32Array.of(1));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(1));
  assert.equal(first.report.backendReport.scheduling?.batchSize ?? 1, 1);
  assert.equal(second.report.backendReport.scheduling?.batchSize ?? 1, 1);
  assert.notEqual(first.report.executionId, second.report.executionId);

  await Promise.all([first.close(), second.close()]);
  await model.close();
  await runtime.close();
});

test('fixed-weight dense-last-axis MatMul retains true B>1 execution', async () => {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const model = await runtime.compile(batchedMatMulSnapshot(), {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });

  const [first, second] = await Promise.all([
    runtime.submit(model, input(2, 3)).result,
    runtime.submit(model, input(5, 7)).result,
  ]);
  assert.deepEqual(await first.output('y').read(), Float32Array.of(2, 3));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(5, 7));
  assert.equal(first.report.backendReport.scheduling.batchSize, 2);
  assert.equal(first.report.backendReport.scheduling.trueBackendInvocations, 1);

  await Promise.all([first.close(), second.close()]);
  await model.close();
  await runtime.close();
});

test('more than 1024 live built-in shape routes retain stable compatibility tokens', async () => {
  const runtime = new Runtime({
    execution: {
      mode: 'scheduled',
      scheduler: { maxRequests: 2048, maxBatchDelayMs: 1 },
      results: { maxRetainedResults: 2048 },
    },
  });
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const model = await runtime.compile(manyShapeIdentitySnapshot(), {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });

  const handles = [];
  for (let width = 1; width <= 1025; width++) {
    handles.push(runtime.submit(model, Object.freeze({
      x: Object.freeze({
        data: new Float32Array(width).fill(width),
        shape: Object.freeze([1, width]),
      }),
    })));
  }
  const results = await Promise.all(handles.map((handle) => handle.result));
  assert.equal(results.length, 1025);
  assert.deepEqual(await results[0].output('y').read(), Float32Array.of(1));
  assert.equal((await results.at(-1).output('y').read()).at(-1), 1025);

  await Promise.all(results.map((result) => result.close()));
  await model.close();
  await runtime.close();
});

test('scheduled batching preserves a public input literally named __proto__', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime, reservedInputNameSnapshot());
  const namedInput = (left, right) => Object.freeze(Object.fromEntries([
    ['__proto__', Object.freeze({
      data: Float32Array.of(left, right), shape: Object.freeze([1, 2]),
    })],
  ]));

  const [first, second] = await Promise.all([
    runtime.submit(model, namedInput(1, 2)).result,
    runtime.submit(model, namedInput(3, 4)).result,
  ]);
  assert.deepEqual(await first.output('y').read(), Float32Array.of(2, 4));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(6, 8));
  assert.equal(first.report.backendReport.scheduling.batchSize, 2);

  await Promise.all([first.close(), second.close()]);
  await model.close();
  await runtime.close();
});

test('different compiled targets share arbitration but never share a physical batch', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const firstModel = await compile(runtime);
  const secondModel = await compile(runtime);

  const [first, second] = await Promise.all([
    runtime.submit(firstModel, input(1, 1)).result,
    runtime.submit(secondModel, input(2, 2)).result,
  ]);
  assert.equal(fixture.executions.length, 2);
  assert.deepEqual(fixture.executions.map(({ shape }) => shape), [[1, 2], [1, 2]]);

  await Promise.all([first.close(), second.close(), firstModel.close(), secondModel.close()]);
  await runtime.close();
});

test('SCHEDULED latest supersedes only an older queued frame from the same stream', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: { mode: 'scheduled', scheduler: { maxBatchDelayMs: 20 } },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const oldFrame = runtime.submit(model, input(1, 1), {
    freshness: 'latest', streamKey: 'camera/front',
  });
  const oldFailure = oldFrame.result.catch((error) => error);
  const newFrame = runtime.submit(model, input(5, 6), {
    freshness: 'latest', streamKey: 'camera/front',
  });
  const [error, result] = await Promise.all([oldFailure, newFrame.result]);

  assert.equal(error.code, 'SUPERSEDED');
  assert.equal(oldFrame.state, 'superseded');
  assert.deepEqual(await result.output('y').read(), Float32Array.of(10, 12));
  assert.equal(fixture.executions.length, 1);

  await result.close();
  await model.close();
  await runtime.close();
});

test('LATEST preserves an explicitly ALL request with the same stream key', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: { mode: 'scheduled', scheduler: { maxBatchDelayMs: 10 } },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const preserved = runtime.submit(model, input(1, 2), {
    freshness: 'all', streamKey: 'camera/front',
  });
  const latest = runtime.submit(model, input(5, 6), {
    freshness: 'latest', streamKey: 'camera/front',
  });
  const [preservedResult, latestResult] = await Promise.all([
    preserved.result, latest.result,
  ]);

  assert.equal(preserved.state, 'completed');
  assert.deepEqual(await preservedResult.output('y').read(), Float32Array.of(2, 4));
  assert.deepEqual(await latestResult.output('y').read(), Float32Array.of(10, 12));
  assert.equal(fixture.executions.length, 1);
  assert.deepEqual(fixture.executions[0].shape, [2, 2]);

  await Promise.all([preservedResult.close(), latestResult.close()]);
  await model.close();
  await runtime.close();
});

test('queued LATEST replacement transfers count slots but requires physical byte headroom', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: {
      mode: 'scheduled',
      scheduler: { maxBatchDelayMs: 10, maxRequests: 1, maxInputBytes: 16 },
      results: { maxRetainedResults: 1, maxRetainedOutputBytes: 16 },
    },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const oldFrame = runtime.submit(model, input(1, 2), {
    freshness: 'latest', streamKey: 'camera/front',
  });
  const oldFailure = oldFrame.result.catch((error) => error);
  const newFrame = runtime.submit(model, input(7, 8), {
    freshness: 'latest', streamKey: 'camera/front',
  });
  assert.equal(runtime.inspectExecution().admittedRequests, 1);
  assert.equal(runtime.inspectExecution().admittedInputBytes, 8);
  assert.equal(runtime.inspectExecution().retainedResults, 1);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 8);

  const [error, result] = await Promise.all([oldFailure, newFrame.result]);
  assert.equal(error.code, 'SUPERSEDED');
  assert.deepEqual(await result.output('y').read(), Float32Array.of(14, 16));
  assert.equal(fixture.executions.length, 1);

  await result.close();
  await model.close();
  await runtime.close();
});

test('queued LATEST preserves the old frame when replacement bytes have no headroom', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: {
      mode: 'scheduled',
      scheduler: { maxBatchDelayMs: 20, maxRequests: 1, maxInputBytes: 8 },
      results: { maxRetainedResults: 1, maxRetainedOutputBytes: 8 },
    },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const oldFrame = runtime.submit(model, input(1, 2), {
    freshness: 'latest', streamKey: 'camera/front',
  });

  assert.throws(
    () => runtime.submit(model, input(7, 8), {
      freshness: 'latest', streamKey: 'camera/front',
    }),
    (error) => error.code === 'OVERLOADED',
  );
  assert.equal(oldFrame.state, 'queued');
  assert.equal(runtime.inspectExecution().admittedRequests, 1);
  assert.equal(runtime.inspectExecution().admittedInputBytes, 8);
  assert.equal(runtime.inspectExecution().inputSnapshotCopies, 1,
    'the rejected replacement never overlaps the old owned snapshot');

  const oldResult = await oldFrame.result;
  assert.deepEqual(await oldResult.output('y').read(), Float32Array.of(2, 4));
  assert.equal(fixture.executions.length, 1);
  await oldResult.close();
  await model.close();
  await runtime.close();
});

test('a newer LATEST frame logically supersedes an already-submitted LATEST result', async () => {
  const gate = deferred();
  const fixture = schedulerProvider({ blockFirst: gate.promise });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const oldFrame = runtime.submit(model, input(1, 2), {
    freshness: 'latest', streamKey: 'camera/front',
  });
  const oldFailure = oldFrame.result.catch((error) => error);
  await nextTurn();
  assert.equal(oldFrame.state, 'submitted');

  const newFrame = runtime.submit(model, input(5, 6), {
    freshness: 'latest', streamKey: 'camera/front',
  });
  let oldSettled = false;
  oldFailure.then(() => { oldSettled = true; });
  await Promise.resolve();
  assert.equal(oldFrame.state, 'submitted');
  assert.equal(oldSettled, false);
  assert.equal(runtime.inspectExecution().admittedRequests, 2);
  gate.resolve();

  const [error, result] = await Promise.all([oldFailure, newFrame.result]);
  assert.equal(error.code, 'SUPERSEDED');
  assert.equal(oldFrame.state, 'superseded');
  assert.deepEqual(await result.output('y').read(), Float32Array.of(10, 12));
  assert.equal(fixture.executions.length, 2);
  assert.equal(runtime.inspectExecution().retainedResults, 1,
    'the discarded submitted result releases its reservation after close');
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 8);

  await result.close();
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);
  await model.close();
  await runtime.close();
});

test('DROP_IF_LATE expires behind an active domain while ALL remains queued', async () => {
  const gate = deferred();
  const fixture = schedulerProvider({ blockFirst: gate.promise });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const running = runtime.submit(model, input(1, 2));
  await nextTurn();
  assert.equal(running.state, 'submitted');
  const expiring = runtime.submit(model, input(3, 4), {
    freshness: 'drop-if-late', deadlineMonotonicMs: performance.now() + 30,
  });
  const preserved = runtime.submit(model, input(5, 6), {
    freshness: 'all', deadlineMonotonicMs: performance.now() + 30,
  });
  const expiration = expiring.result.catch((error) => error);
  const observed = await Promise.race([
    expiration,
    delay(150).then(() => null),
  ]);

  assert.equal(observed?.code, 'DEADLINE_EXCEEDED');
  assert.equal(expiring.state, 'failed');
  assert.equal(preserved.state, 'queued');
  assert.equal(fixture.executions.length, 1);
  assert.equal(runtime.inspectExecution().retainedResults, 2,
    'queued DROP expiry releases only its own future-result reservation');
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 16);
  gate.resolve();
  const [result, preservedResult] = await Promise.all([running.result, preserved.result]);
  assert.deepEqual(await preservedResult.output('y').read(), Float32Array.of(10, 12));

  await Promise.all([result.close(), preservedResult.close()]);
  assert.equal(runtime.inspectExecution().retainedResults, 0);
  assert.equal(runtime.inspectExecution().retainedOutputBytes, 0);
  await model.close();
  await runtime.close();
});

test('an earlier deadline preempts an already-armed scheduled batching window', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: { mode: 'scheduled', scheduler: { maxBatchDelayMs: 250 } },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const waiting = runtime.submit(model, input(1, 2));
  await nextTurn();
  const expiring = runtime.submit(model, input(3, 4), {
    deadlineMonotonicMs: performance.now() + 30,
  });
  const observed = await Promise.race([
    expiring.result,
    delay(150).then(() => null),
  ]);

  assert.notEqual(observed, null);
  const waitingResult = await waiting.result;
  assert.equal(fixture.executions.length, 1);
  assert.deepEqual(fixture.executions[0].shape, [2, 2]);

  await Promise.all([observed.close(), waitingResult.close()]);
  await model.close();
  await runtime.close();
});

test('request handles report soft and hard deadline misses at terminal settlement', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 1 });
  const routeToken = Object.freeze({});
  const resourceDomain = Object.freeze({});
  const deviceEpoch = Object.freeze({});
  const gates = Array.from({ length: 4 }, () => deferred());
  const started = Array.from({ length: 4 }, () => deferred());
  let calls = 0;
  const target = {
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({ compatibilityToken: routeToken, resourceDomain, deviceEpoch });
    },
    async executeScheduledBatch() {
      const call = calls++;
      started[call].resolve();
      await gates[call].promise;
      return Object.freeze([fakeResult()]);
    },
    async closeScheduledExecutionRoute() {},
  };

  const lateAllDeadline = performance.now() + 30;
  const lateAll = scheduler.submit(target, input(1, 1), {
    freshness: 'all', deadlineMonotonicMs: lateAllDeadline,
  });
  await started[0].promise;
  assert.equal(lateAll.deadlineMissed, null);
  await delay(Math.max(0, lateAllDeadline - performance.now() + 10));
  gates[0].resolve();
  const lateAllResult = await lateAll.result;
  assert.equal(lateAll.deadlineMissed, true);

  const lateLatestDeadline = performance.now() + 30;
  const lateLatest = scheduler.submit(target, input(2, 2), {
    freshness: 'latest', streamKey: 'camera/front',
    deadlineMonotonicMs: lateLatestDeadline,
  });
  await started[1].promise;
  assert.equal(lateLatest.deadlineMissed, null);
  await delay(Math.max(0, lateLatestDeadline - performance.now() + 10));
  gates[1].resolve();
  const lateLatestResult = await lateLatest.result;
  assert.equal(lateLatest.deadlineMissed, true);

  const hardDeadline = performance.now() + 30;
  const hard = scheduler.submit(target, input(3, 3), {
    freshness: 'drop-if-late', deadlineMonotonicMs: hardDeadline,
  });
  await started[2].promise;
  assert.equal(hard.deadlineMissed, null);
  await delay(Math.max(0, hardDeadline - performance.now() + 10));
  gates[2].resolve();
  await assert.rejects(hard.result, (error) => error.code === 'DEADLINE_EXCEEDED');
  assert.equal(hard.deadlineMissed, true);

  const onTime = scheduler.submit(target, input(4, 4), {
    freshness: 'all', deadlineMonotonicMs: performance.now() + 1_000,
  });
  await started[3].promise;
  assert.equal(onTime.deadlineMissed, null);
  gates[3].resolve();
  const onTimeResult = await onTime.result;
  assert.equal(onTime.deadlineMissed, false);

  await Promise.all([lateAllResult.close(), lateLatestResult.close(), onTimeResult.close()]);
  await scheduler.close();
});

test('a hard deadline takes precedence over a simultaneous backend failure', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 1 });
  const routeToken = Object.freeze({});
  const resourceDomain = Object.freeze({});
  const deviceEpoch = Object.freeze({});
  const started = deferred();
  const gate = deferred();
  const providerFailure = new Error('provider failed after its fence');
  const target = {
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({ compatibilityToken: routeToken, resourceDomain, deviceEpoch });
    },
    async executeScheduledBatch() {
      started.resolve();
      await gate.promise;
      throw providerFailure;
    },
    async closeScheduledExecutionRoute() {},
  };

  const deadline = performance.now() + 30;
  const handle = scheduler.submit(target, input(1, 2), {
    freshness: 'drop-if-late',
    deadlineMonotonicMs: deadline,
  });
  await started.promise;
  await delay(Math.max(0, deadline - performance.now() + 10));
  gate.resolve();
  await assert.rejects(handle.result, (error) => {
    assert.equal(error.code, 'DEADLINE_EXCEEDED');
    assert.equal(error.cause, providerFailure);
    return true;
  });
  assert.equal(handle.state, 'failed');
  assert.equal(handle.deadlineMissed, true);
  await scheduler.close();
});

test('priority arbitration bounds starvation across ready routes', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: { scheduler: { maxBatchSize: 1, maxPriorityBurst: 2 } },
  });
  runtime._addProvider('fixture', fixture.provider);
  const highModel = await compile(runtime);
  const lowModel = await compile(runtime);

  const handles = [
    runtime.submit(highModel, input(1, 1), { priority: 100 }),
    runtime.submit(highModel, input(2, 2), { priority: 100 }),
    runtime.submit(highModel, input(3, 3), { priority: 100 }),
    runtime.submit(highModel, input(4, 4), { priority: 100 }),
    runtime.submit(lowModel, input(9, 9), { priority: -100 }),
  ];
  const results = await Promise.all(handles.map((handle) => handle.result));

  assert.deepEqual(
    fixture.executions.slice(0, 3).map(({ values }) => values),
    [[1, 1], [2, 2], [9, 9]],
  );

  await Promise.all(results.map((result) => result.close()));
  await Promise.all([highModel.close(), lowModel.close()]);
  await runtime.close();
});

test('request aging prevents same-route high-priority arrivals from starving an older request', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 1 });
  const firstGate = deferred();
  const firstStarted = deferred();
  const routeToken = Object.freeze({});
  const resourceDomain = Object.freeze({});
  const deviceEpoch = Object.freeze({});
  const executed = [];
  let calls = 0;
  const target = {
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({ compatibilityToken: routeToken, resourceDomain, deviceEpoch });
    },
    async executeScheduledBatch(requests) {
      const call = calls++;
      executed.push(requests[0].inputs.x.data[0]);
      if (call === 0) {
        firstStarted.resolve();
        await firstGate.promise;
      }
      return Object.freeze([fakeResult()]);
    },
    async closeScheduledExecutionRoute() {},
  };

  const first = scheduler.submit(target, input(100, 100), { priority: 1 });
  await firstStarted.promise;
  const oldLow = scheduler.submit(target, input(1, 1), { priority: 0 });
  await delay(35);
  const newerHigh = [2, 3, 4, 5].map((value) =>
    scheduler.submit(target, input(value, value), { priority: 1 }));
  firstGate.resolve();

  const handles = [first, oldLow, ...newerHigh];
  const results = await Promise.all(handles.map((handle) => handle.result));
  assert.deepEqual(executed, [100, 1, 2, 3, 4, 5]);
  await Promise.all(results.map((result) => result.close()));
  await scheduler.close();
});

test('one physical batch completion instant applies before slow cancelled-lane cleanup', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 2 });
  const routeToken = Object.freeze({});
  const resourceDomain = Object.freeze({});
  const deviceEpoch = Object.freeze({});
  const execution = deferred();
  const started = deferred();
  const slowClose = deferred();
  const slowCloseStarted = deferred();
  const target = {
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({
        compatibilityToken: routeToken,
        resourceDomain,
        deviceEpoch,
        maxBatchSize: 2,
      });
    },
    async executeScheduledBatch() {
      started.resolve();
      await execution.promise;
      return Object.freeze([
        Object.freeze({
          async close() {
            slowCloseStarted.resolve();
            await slowClose.promise;
          },
        }),
        fakeResult(),
      ]);
    },
    async closeScheduledExecutionRoute() {},
  };
  const cancelled = scheduler.submit(target, input(1, 2), { priority: 1 });
  const deadline = performance.now() + 100;
  const onTime = scheduler.submit(target, input(3, 4), {
    freshness: 'drop-if-late', deadlineMonotonicMs: deadline,
  });
  await started.promise;
  assert.equal(cancelled.cancel(), true);
  execution.resolve();
  await slowCloseStarted.promise;

  let stateAfterDeadline;
  try {
    await delay(Math.max(0, deadline - performance.now() + 20));
    stateAfterDeadline = onTime.state;
  } finally {
    slowClose.resolve();
  }
  assert.equal(stateAfterDeadline, 'completed');
  await assert.rejects(cancelled.result, (error) => error.code === 'CANCELLED');
  const result = await onTime.result;
  await result.close();
  await scheduler.close();
});

test('admission is bounded by owned input bytes and request count', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const fixture = schedulerProvider({ blockFirst: gate });
  const runtime = new Runtime({
    execution: { scheduler: { maxRequests: 1, maxInputBytes: 8 } },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const accepted = runtime.submit(model, input(1, 2));
  await new Promise((resolve) => setImmediate(resolve));
  assert.throws(
    () => runtime.submit(model, input(3, 4)),
    (error) => error.code === 'OVERLOADED',
  );
  release();
  const result = await accepted.result;
  await result.close();
  await model.close();
  await runtime.close();
});

test('busy DIRECT fails without fallback while an explicit submit queues', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const fixture = schedulerProvider({ blockFirst: gate });
  const runtime = new Runtime({ execution: { mode: 'scheduled' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const first = runtime.run(model, input(1, 2), { mode: 'direct' });
  await Promise.resolve();
  await assert.rejects(
    runtime.run(model, input(3, 4), { mode: 'direct' }),
    (error) => error.code === 'BUSY',
  );
  const queued = runtime.submit(model, input(5, 6));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(runtime.inspectExecution().schedulerAllocated, true);
  release();
  const [firstResult, queuedResult] = await Promise.all([first, queued.result]);
  assert.deepEqual(await queuedResult.output('y').read(), Float32Array.of(10, 12));

  await Promise.all([firstResult.close(), queuedResult.close()]);
  await model.close();
  await runtime.close();
});

test('an accepted scheduled request synchronously fences a same-turn DIRECT caller', async () => {
  const gate = deferred();
  const started = deferred();
  const fixture = schedulerProvider({
    blockFirst: gate.promise,
    onExecute(call) {
      if (call === 0) started.resolve();
    },
  });
  const runtime = new Runtime({ execution: { mode: 'scheduled' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const accepted = runtime.submit(model, input(1, 2));
  // No microtask is yielded here: the scheduler drain has not yet registered
  // itself as a route waiter, but admission already owns a synchronous claim.
  await assert.rejects(
    runtime.run(model, input(99, 99), { mode: 'direct' }),
    (error) => error.code === 'BUSY',
  );
  await started.promise;
  assert.deepEqual(fixture.executions.map(({ values }) => values), [[1, 2]]);

  gate.resolve();
  const result = await accepted.result;
  await result.close();
  await model.close();
  await runtime.close();
});

test('queued route ownership handoff prevents DIRECT barging before waiter resume', async () => {
  const firstGate = deferred();
  const secondGate = deferred();
  const secondStarted = deferred();
  const fixture = schedulerProvider({
    blockCalls: [firstGate.promise, secondGate.promise],
    onExecute(call) {
      if (call === 1) secondStarted.resolve();
    },
  });
  const runtime = new Runtime({ execution: { mode: 'scheduled' } });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);

  const first = runtime.run(model, input(1, 2), { mode: 'direct' });
  await Promise.resolve();
  const queued = runtime.submit(model, input(3, 4));
  let probing = true;
  const unexpectedResults = [];
  const unexpectedFailures = [];
  const probe = () => {
    if (!probing) return;
    runtime.run(model, input(99, 99), { mode: 'direct' }).then(
      (result) => { unexpectedResults.push(result); },
      (error) => {
        if (error.code !== 'BUSY') unexpectedFailures.push(error);
        if (probing) queueMicrotask(probe);
      },
    );
  };
  probe();
  firstGate.resolve();
  await secondStarted.promise;
  probing = false;
  await nextTurn();

  assert.equal(unexpectedFailures.length, 0);
  assert.equal(fixture.executions.some(({ values }) => values[0] === 99), false);
  secondGate.resolve();
  const [firstResult, queuedResult] = await Promise.all([first, queued.result]);
  await nextTurn();
  assert.equal(unexpectedResults.length, 0);

  await Promise.all([firstResult.close(), queuedResult.close()]);
  await model.close();
  await runtime.close();
});

test('scheduled route preparation observes the exact owned input snapshot', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 1 });
  const routeToken = Object.freeze({});
  const resourceDomainIdentity = Object.freeze({});
  let inputReads = 0;
  let preparedValue = null;
  let executedValue = null;
  const opaquePreparedPlan = Object.freeze({});
  const mutableInputs = {};
  Object.defineProperty(mutableInputs, 'x', {
    enumerable: true,
    get() {
      inputReads++;
      return input(inputReads, inputReads).x;
    },
  });
  const target = {
    resourceDomainIdentity,
    backend: 'direct-fixture',
    prepareScheduledExecution(ownedInputs) {
      preparedValue = ownedInputs.x.data[0];
      return preparedRoute({
        compatibilityToken: routeToken,
        resourceDomain: resourceDomainIdentity,
        preparedPlan: opaquePreparedPlan,
      });
    },
    async executeScheduledBatch(requests) {
      executedValue = requests[0].inputs.x.data[0];
      assert.equal(requests[0].preparedPlan, opaquePreparedPlan);
      return Object.freeze([fakeResult()]);
    },
    async closeScheduledExecutionRoute() {},
  };

  const handle = scheduler.submit(target, mutableInputs);
  const result = await handle.result;
  assert.equal(inputReads, 1);
  assert.equal(preparedValue, 1);
  assert.equal(executedValue, 1);

  await result.close();
  await scheduler.close();
});

test('scheduled input snapshots canonicalize typed-array subclasses by storage kind', async () => {
  class Float32Subclass extends Float32Array {}
  const fixture = schedulerProvider();
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const subclassInput = Object.freeze({
    x: Object.freeze({
      data: new Float32Subclass([3, 5]),
      shape: Object.freeze([1, 2]),
    }),
  });

  const result = await runtime.submit(model, subclassInput).result;
  assert.deepEqual(await result.output('y').read(), Float32Array.of(6, 10));
  assert.deepEqual(fixture.executions.map(({ values }) => values), [[3, 5]]);

  await result.close();
  await model.close();
  await runtime.close();
});

test('scheduled input snapshots canonicalize cross-realm typed arrays by intrinsic brand', async () => {
  const foreignFloat32 = runInNewContext('new Float32Array([3, 5])');
  assert.equal(foreignFloat32 instanceof Float32Array, false);
  Object.defineProperty(foreignFloat32, Symbol.toStringTag, {
    configurable: true,
    value: 'Uint8ClampedArray',
  });

  const fixture = schedulerProvider();
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const result = await runtime.submit(model, Object.freeze({
    x: Object.freeze({ data: foreignFloat32, shape: Object.freeze([1, 2]) }),
  })).result;
  assert.deepEqual(await result.output('y').read(), Float32Array.of(6, 10));
  assert.deepEqual(fixture.executions.map(({ values }) => values), [[3, 5]]);

  const scheduler = new RuntimeScheduler();
  const foreignClamped = runInNewContext('new Uint8ClampedArray([1, 300])');
  let preparedConstructor = null;
  let executedConstructor = null;
  const target = {
    backend: 'direct-fixture',
    prepareScheduledExecution(ownedInputs) {
      preparedConstructor = ownedInputs.u.data.constructor;
      return preparedRoute({ inputBytes: ownedInputs.u.data.byteLength });
    },
    async executeScheduledBatch(requests) {
      executedConstructor = requests[0].inputs.u.data.constructor;
      assert.deepEqual(requests[0].inputs.u.data, Uint8ClampedArray.of(1, 255));
      return Object.freeze([fakeResult()]);
    },
    async closeScheduledExecutionRoute() {},
  };
  const clampedHandle = scheduler.submit(target, Object.freeze({
    u: Object.freeze({ data: foreignClamped, shape: Object.freeze([2]) }),
  }));
  const clampedResult = await clampedHandle.result;
  assert.equal(preparedConstructor, Uint8ClampedArray);
  assert.equal(executedConstructor, Uint8ClampedArray);

  await Promise.all([result.close(), clampedResult.close()]);
  await Promise.all([model.close(), scheduler.close()]);
  await runtime.close();
});

test('exact input-byte preflight rejects before cloning or target preparation', async () => {
  const scheduler = new RuntimeScheduler({ maxInputBytes: 4 });
  const shape = [1, 2];
  let shapeReads = 0;
  Object.defineProperty(shape, 0, {
    configurable: true,
    enumerable: true,
    get() {
      shapeReads++;
      return 1;
    },
  });
  let prepareCalls = 0;
  const target = {
    resourceDomainIdentity: Object.freeze({}),
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      prepareCalls++;
      return preparedRoute({ resourceDomain: target.resourceDomainIdentity });
    },
    async executeScheduledBatch() {
      assert.fail('an over-budget request was executed');
    },
    async closeScheduledExecutionRoute() {},
  };

  assert.throws(
    () => scheduler.submit(target, Object.freeze({
      x: Object.freeze({ data: Float32Array.of(1, 2), shape }),
    })),
    (error) => error.code === 'OVERLOADED',
  );
  assert.equal(shapeReads, 0);
  assert.equal(prepareCalls, 0);
  class UnderreportingFloat32Array extends Float32Array {
    get byteLength() { return 0; }
  }
  assert.throws(
    () => scheduler.submit(target, Object.freeze({
      x: Object.freeze({
        data: new UnderreportingFloat32Array([1, 2]), shape: Object.freeze([1, 2]),
      }),
    })),
    (error) => error.code === 'OVERLOADED',
  );
  assert.equal(prepareCalls, 0);
  assert.equal(scheduler.requestCount, 0);
  await scheduler.close();
});

test('compiled metadata preflight rejects a large invalid input before payload copy', async () => {
  const fixture = schedulerProvider();
  const runtime = new Runtime({
    execution: { scheduler: { maxInputBytes: 32 * 1024 * 1024 } },
  });
  runtime._addProvider('fixture', fixture.provider);
  const model = await compile(runtime);
  const large = new Float32Array(4 * 1024 * 1024);

  assert.throws(
    () => runtime.submit(model, Object.freeze({
      wrong: Object.freeze({ data: large, shape: Object.freeze([1, large.length]) }),
    })),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.equal(runtime.inspectExecution().inputSnapshotCopies, 0);
  assert.equal(runtime.inspectExecution().admittedRequests, 0);
  assert.equal(runtime.inspectExecution().totalInputBytes, 0);
  assert.equal(fixture.routePlans.length, 0,
    'shape/name rejection happens before provider route preparation');

  await model.close();
  await runtime.close();
});

test('admission bounds stream metadata and rejects unsupported host views', async () => {
  const scheduler = new RuntimeScheduler();
  let prepareCalls = 0;
  const target = {
    resourceDomainIdentity: Object.freeze({}),
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      prepareCalls++;
      return preparedRoute({ resourceDomain: target.resourceDomainIdentity });
    },
    async executeScheduledBatch() {
      assert.fail('an invalid request was executed');
    },
    async closeScheduledExecutionRoute() {},
  };

  assert.throws(
    () => scheduler.submit(target, input(1, 2), {
      freshness: 'latest', streamKey: 'x'.repeat(1025),
    }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.throws(
    () => scheduler.submit(target, input(1, 2), { freshness: 'drop-if-late' }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.throws(
    () => scheduler.submit(target, Object.freeze({
      x: Object.freeze({ data: new Uint16Array([1, 2]), shape: Object.freeze([1, 2]) }),
    })),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  const excessiveRank = [];
  excessiveRank.length = 1025;
  assert.throws(
    () => scheduler.submit(target, Object.freeze({
      x: Object.freeze({ data: new Uint8Array(0), shape: excessiveRank }),
    })),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  const manyInputs = {};
  for (let index = 0; index < 1025; index++) {
    manyInputs[`x${index}`] = Object.freeze({
      data: new Uint8Array(0), shape: Object.freeze([0]),
    });
  }
  assert.throws(
    () => scheduler.submit(target, manyInputs),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.throws(
    () => scheduler.submit(target, Object.freeze({
      ['x'.repeat(1025)]: Object.freeze({
        data: new Uint8Array(0), shape: Object.freeze([0]),
      }),
    })),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.equal(prepareCalls, 0);
  await scheduler.close();
});

test('target cannot under-report owned input bytes', async () => {
  const scheduler = new RuntimeScheduler({ maxInputBytes: 16 });
  const target = {
    resourceDomainIdentity: Object.freeze({}),
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({
        resourceDomain: target.resourceDomainIdentity,
        inputBytes: 4,
      });
    },
    async executeScheduledBatch() {
      assert.fail('an under-reported request was executed');
    },
    async closeScheduledExecutionRoute() {},
  };

  assert.throws(
    () => scheduler.submit(target, input(1, 2)),
    (error) => error.code === 'ABI_UNSUPPORTED',
  );
  assert.equal(scheduler.requestCount, 0);
  assert.equal(scheduler.inputBytes, 0);
  await scheduler.close();
});

test('prepare reentrancy cannot admit a request after scheduler close begins', async () => {
  const scheduler = new RuntimeScheduler();
  const routeToken = Object.freeze({});
  let closePromise = null;
  const target = {
    resourceDomainIdentity: Object.freeze({}),
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      closePromise = scheduler.close();
      return preparedRoute({
        compatibilityToken: routeToken,
        resourceDomain: target.resourceDomainIdentity,
      });
    },
    async executeScheduledBatch() {
      assert.fail('a request admitted after reentrant close');
    },
    async closeScheduledExecutionRoute() {},
  };

  assert.throws(
    () => scheduler.submit(target, input(1, 2)),
    (error) => error.code === 'HANDLE_DISPOSED',
  );
  await closePromise;
  assert.equal(scheduler.requestCount, 0);
  assert.equal(scheduler.inputBytes, 0);
});

test('route-claim reentrancy cannot admit after close or bypass recomputed capacity', async () => {
  const closingScheduler = new RuntimeScheduler();
  const closingToken = Object.freeze({});
  const closingDomain = Object.freeze({});
  const closingEpoch = Object.freeze({});
  let closePromise = null;
  let closingClaimReleases = 0;
  const closingTarget = {
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({
        compatibilityToken: closingToken,
        resourceDomain: closingDomain,
        deviceEpoch: closingEpoch,
      });
    },
    reserveScheduledExecutionRoute() {
      closePromise = closingScheduler.close();
      return () => { closingClaimReleases++; };
    },
    async executeScheduledBatch() {
      assert.fail('a request admitted after a reentrant route-claim close');
    },
    async closeScheduledExecutionRoute() {},
  };
  assert.throws(
    () => closingScheduler.submit(closingTarget, input(1, 2)),
    (error) => error.code === 'HANDLE_DISPOSED',
  );
  await closePromise;
  assert.equal(closingClaimReleases, 1);
  assert.equal(closingScheduler.requestCount, 0);

  const capacityScheduler = new RuntimeScheduler({ maxRequests: 1, maxBatchSize: 1 });
  const capacityToken = Object.freeze({});
  const capacityDomain = Object.freeze({});
  const capacityEpoch = Object.freeze({});
  let nesting = false;
  let nestedHandle = null;
  let capacityClaimReleases = 0;
  const capacityTarget = {
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({
        compatibilityToken: capacityToken,
        resourceDomain: capacityDomain,
        deviceEpoch: capacityEpoch,
      });
    },
    reserveScheduledExecutionRoute() {
      if (!nesting && nestedHandle === null) {
        nesting = true;
        try {
          nestedHandle = capacityScheduler.submit(capacityTarget, input(9, 9));
        } finally {
          nesting = false;
        }
      }
      return () => { capacityClaimReleases++; };
    },
    async executeScheduledBatch() { return Object.freeze([fakeResult()]); },
    async closeScheduledExecutionRoute() {},
  };
  assert.throws(
    () => capacityScheduler.submit(capacityTarget, input(3, 4)),
    (error) => error.code === 'OVERLOADED',
  );
  const nestedResult = await nestedHandle.result;
  assert.equal(capacityClaimReleases, 2);
  assert.equal(capacityScheduler.requestCount, 0);
  await nestedResult.close();
  await capacityScheduler.close();
});

test('a throwing route-claim release cannot strand terminal settlement', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 1 });
  const routeToken = Object.freeze({});
  const resourceDomain = Object.freeze({});
  const deviceEpoch = Object.freeze({});
  let releaseCalls = 0;
  const target = {
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({ compatibilityToken: routeToken, resourceDomain, deviceEpoch });
    },
    reserveScheduledExecutionRoute() {
      return () => {
        releaseCalls++;
        throw new Error('hostile release');
      };
    },
    async executeScheduledBatch() { return Object.freeze([fakeResult()]); },
    async closeScheduledExecutionRoute() {},
  };

  const handle = scheduler.submit(target, input(1, 2));
  const result = await handle.result;
  assert.equal(handle.state, 'completed');
  assert.equal(releaseCalls, 1);
  assert.equal(scheduler.requestCount, 0);
  await result.close();
  await scheduler.close();
});

test('execute reentrancy publishes in-flight work before close snapshots it', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 1 });
  const routeToken = Object.freeze({});
  const execution = deferred();
  const started = deferred();
  let closePromise = null;
  let closeCalls = 0;
  const target = {
    resourceDomainIdentity: Object.freeze({}),
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({
        compatibilityToken: routeToken,
        resourceDomain: target.resourceDomainIdentity,
      });
    },
    executeScheduledBatch(requests) {
      started.resolve();
      closePromise = scheduler.close();
      return execution.promise.then(() => Object.freeze(requests.map(() => fakeResult())));
    },
    async closeScheduledExecutionRoute() { closeCalls++; },
  };

  const handle = scheduler.submit(target, input(1, 2));
  await started.promise;
  let closed = false;
  closePromise.then(() => { closed = true; });
  await nextTurn();
  assert.equal(closed, false);
  assert.equal(closeCalls, 0);

  execution.resolve();
  const result = await handle.result;
  await closePromise;
  assert.equal(closed, true);
  assert.equal(closeCalls, 1);
  await result.close();
});

test('concurrent retirement callers share and await the same in-flight fence', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 1 });
  const routeToken = Object.freeze({});
  const execution = deferred();
  const started = deferred();
  let closeCalls = 0;
  const target = {
    resourceDomainIdentity: Object.freeze({}),
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({
        compatibilityToken: routeToken,
        resourceDomain: target.resourceDomainIdentity,
      });
    },
    async executeScheduledBatch(requests) {
      started.resolve();
      await execution.promise;
      return Object.freeze(requests.map(() => fakeResult()));
    },
    async closeScheduledExecutionRoute() { closeCalls++; },
  };

  const handle = scheduler.submit(target, input(1, 2));
  await started.promise;
  const firstRetirement = scheduler.retireTarget(target);
  const secondRetirement = scheduler.retireTarget(target);
  assert.equal(firstRetirement, secondRetirement);
  let retired = false;
  secondRetirement.then(() => { retired = true; });
  await nextTurn();
  assert.equal(retired, false);
  assert.equal(closeCalls, 0);

  execution.resolve();
  const result = await handle.result;
  await firstRetirement;
  assert.equal(retired, true);
  assert.equal(closeCalls, 1);
  await result.close();
  await scheduler.close();
  assert.equal(closeCalls, 1);
});

test('a target cannot publish one owned result to multiple batch requests', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 2 });
  const routeToken = Object.freeze({});
  const deviceEpoch = Object.freeze({});
  let resultCloseCalls = 0;
  const sharedResult = Object.freeze({
    async close() { resultCloseCalls++; },
  });
  const target = {
    resourceDomainIdentity: Object.freeze({}),
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({
        compatibilityToken: routeToken,
        resourceDomain: target.resourceDomainIdentity,
        deviceEpoch,
        maxBatchSize: 2,
      });
    },
    async executeScheduledBatch() {
      return Object.freeze([sharedResult, sharedResult]);
    },
    async closeScheduledExecutionRoute() {},
  };

  const first = scheduler.submit(target, input(1, 2));
  const second = scheduler.submit(target, input(3, 4));
  const failures = await Promise.all([
    first.result.catch((error) => error),
    second.result.catch((error) => error),
  ]);
  assert.deepEqual(failures.map((error) => error.code), ['ABI_UNSUPPORTED', 'ABI_UNSUPPORTED']);
  assert.equal(resultCloseCalls, 1);
  assert.equal(scheduler.requestCount, 0);

  await scheduler.close();
});

test('submitted cancellation wins a simultaneous backend failure', async () => {
  const scheduler = new RuntimeScheduler({ maxBatchSize: 1 });
  const routeToken = Object.freeze({});
  const execution = deferred();
  const started = deferred();
  const target = {
    resourceDomainIdentity: Object.freeze({}),
    backend: 'direct-fixture',
    prepareScheduledExecution() {
      return preparedRoute({
        compatibilityToken: routeToken,
        resourceDomain: target.resourceDomainIdentity,
      });
    },
    async executeScheduledBatch() {
      started.resolve();
      await execution.promise;
      throw new Error('device lost');
    },
    async closeScheduledExecutionRoute() {},
  };

  const handle = scheduler.submit(target, input(1, 2));
  await started.promise;
  let settled = false;
  handle.result.catch(() => { settled = true; });
  assert.equal(handle.cancel(), true);
  assert.equal(handle.cancel(), false);
  await Promise.resolve();
  assert.equal(handle.state, 'submitted');
  assert.equal(settled, false);
  assert.equal(scheduler.requestCount, 1);
  assert.equal(scheduler.inputBytes, 8);
  execution.resolve();
  await assert.rejects(handle.result, (error) => error.code === 'CANCELLED');
  assert.equal(handle.state, 'cancelled');
  assert.equal(scheduler.requestCount, 0);

  await scheduler.close();
});
