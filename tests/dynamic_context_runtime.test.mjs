import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ExecutionResult,
  TensorResult,
  Model,
  Runtime,
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  createBackendProviderBatchContract,
  createBackendProviderCapabilities,
  parseGraphDocument,
} from '../ts/index.js';
import { InvariantResourceStore } from '../ts/backends/InvariantResources.js';
import {
  createBackendCompileInput,
} from '../ts/backends/BackendProvider.js';
import { CPUBackendProvider } from '../ts/backends/CPUBackendProvider.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

function typedData(dtype, values) {
  if (dtype === 'float32') return Float32Array.from(values);
  if (dtype === 'int32') return Int32Array.from(values);
  if (dtype === 'int8') return Int8Array.from(values);
  return Uint8Array.from(values);
}

function makeSnapshot(document, definitions = []) {
  const graph = parseGraphDocument(
    document,
    definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
  );
  const weights = Object.fromEntries(definitions.map((definition) => [
    definition.name,
    {
      name: definition.name,
      dtype: definition.dtype,
      shape: definition.shape,
      data: typedData(definition.dtype, definition.values),
    },
  ]));
  return Model.capture({ graph, weights });
}

function identitySnapshot({ dtype = 'float32', maxB = 8, rank2 = true } = {}) {
  const shape = rank2 ? ['B', 'S'] : ['B'];
  const dimensions = rank2
    ? { B: { min: 1, max: maxB }, S: { min: 1, max: 8 } }
    : { B: { min: 1, max: maxB } };
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions,
    inputs: { x: { dtype, shape } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype, shape } },
      params: {},
    }],
    outputs: ['y'],
  });
}

function identityChainSnapshot({ tensorBytes, nodeCount, weightBytes = 0 }) {
  assert.equal(tensorBytes % 4, 0);
  assert.equal(weightBytes % 4, 0);
  const shape = ['B'];
  const nodes = [];
  let input = 'x';
  for (let index = 0; index < nodeCount; index++) {
    const output = `value.${index}`;
    nodes.push({
      id: `identity.${index}`,
      opType: 'Identity',
      inputs: { input },
      outputs: { out: { tensor: output, dtype: 'float32', shape } },
      params: {},
    });
    input = output;
  }
  const definitions = weightBytes === 0 ? [] : [{
    name: 'unused.weight',
    dtype: 'float32',
    shape: [weightBytes / 4],
    values: new Float32Array(weightBytes / 4),
  }];
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: tensorBytes / 4 } },
    inputs: { x: { dtype: 'float32', shape } },
    nodes,
    outputs: [input],
  }, definitions);
}

function domainAttestation(input, overrides = {}) {
  return Object.freeze({
    proofProtocol: 'canonical-symbolic-domain-proof/v1',
    resourceProtocol: 'bounded-resource-maxima/v1',
    support: 'full',
    graphFingerprint: input.graphFingerprint,
    proof: input.shapeDomainProof,
    maximumTensorBytes: 1024,
    maximumResidentBytes: 4096,
    resourceLimitBytes: 1024 * 1024,
    ...overrides,
  });
}

function externalProvider({
  name = 'fixture',
  support = 'full',
  outputLocation = 'host',
  execute = async (request) => ({
    outputs: Object.freeze([Object.freeze({
      name: request.outputDescriptors[0].name,
      shape: request.outputDescriptors[0].shape,
      dtype: request.outputDescriptors[0].dtype,
      location: 'host',
      ownership: 'borrowed',
      data: typedData(request.outputDescriptors[0].dtype, request.inputs.x.data),
    })]),
  }),
} = {}) {
  const events = [];
  let compileInput = null;
  let executionRequest = null;
  let providerContextOptions = null;
  let compileCalls = 0;
  let createContextCalls = 0;
  let executeCalls = 0;
  const resourceDomain = Object.freeze({});
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: name,
    capabilities: createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation,
      dynamicShapeDomain: support,
    }),
    async compile(input) {
      compileCalls++;
      events.push('compile');
      compileInput = input;
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
          shapeDomain: domainAttestation(input),
        }),
        async createContext(options) {
          createContextCalls++;
          events.push('create-context');
          providerContextOptions = options;
          return {
            backendName: name,
            async execute(request) {
              executeCalls++;
              events.push('execute');
              executionRequest = request;
              return execute(request, executeCalls);
            },
            async close() { events.push('context-close'); },
          };
        },
        async close() { events.push('compiled-close'); },
      };
    },
    async close() { events.push('provider-close'); },
  };
  return {
    provider,
    events,
    get compileInput() { return compileInput; },
    get executionRequest() { return executionRequest; },
    get providerContextOptions() { return providerContextOptions; },
    get compileCalls() { return compileCalls; },
    get createContextCalls() { return createContextCalls; },
    get executeCalls() { return executeCalls; },
  };
}

async function compileFixture(snapshot, fixture, contextOptions = {}) {
  const runtime = new Runtime();
  runtime._addProvider(fixture.provider.backendName, fixture.provider);
  const compiled = await runtime.compile(snapshot, {
    backend: {
      mode: 'require',
      backend: fixture.provider.backendName,
      operatorFallback: 'forbid',
    },
  });
  const context = await compiled.createContext(contextOptions);
  return { runtime, compiled, context };
}

test('provider observes the exact logical proof and committed resolved descriptors', async () => {
  const snapshot = identitySnapshot();
  let providerStorage = null;
  const fixture = externalProvider({
    execute: async (request) => {
      providerStorage = new Float32Array(request.inputs.x.data);
      return {
        outputs: Object.freeze([Object.freeze({
          name: 'y',
          shape: request.outputDescriptors[0].shape,
          dtype: 'float32',
          location: 'host',
          ownership: 'borrowed',
          data: providerStorage,
        })]),
      };
    },
  });
  const { runtime, compiled, context } = await compileFixture(snapshot, fixture);

  assert.equal(fixture.compileInput.snapshot, snapshot);
  assert.equal(fixture.compileInput.graph, snapshot.graph);
  assert.equal(fixture.compileInput.shapeDomainProof, snapshot.shapeDomainProof);
  assert.equal(Object.isFrozen(fixture.compileInput), true);
  assert.equal(Object.isFrozen(fixture.provider.capabilities.dynamicShapeDomain), true);
  assert.equal('shapeSystem' in fixture.provider.capabilities.dynamicShapeDomain, false);
  assert.equal(Object.isFrozen(fixture.providerContextOptions), true);
  assert.equal(Object.isFrozen(fixture.providerContextOptions.initialPlan), true);
  assert.equal(
    fixture.providerContextOptions.initialPlan.graphFingerprint,
    snapshot.definitionFingerprint,
  );
  assert.deepEqual(fixture.providerContextOptions.initialPlan.tensors.x.shape, [1, 1]);
  assert.equal('data' in fixture.providerContextOptions.initialPlan.tensors.x, false);

  const values = Float32Array.of(1, 2, 3, 4, 5, 6);
  const result = await context.execute({ x: { data: values, shape: [2, 3] } });
  const request = fixture.executionRequest;
  assert.equal(Object.isFrozen(request), true);
  assert.equal(request.signature, request.plan.signature);
  assert.equal(request.inputDescriptors[0], request.plan.tensors.x);
  assert.equal(request.outputDescriptors[0], request.plan.tensors.y);
  assert.equal(request.tensors, request.plan.tensors);
  assert.equal(request.inputs.x.shape, request.plan.tensors.x.shape);
  assert.deepEqual(request.inputDescriptors[0].shape, [2, 3]);
  assert.deepEqual(request.outputDescriptors[0].shape, [2, 3]);

  providerStorage.fill(99);
  assert.deepEqual(await result.output('y').read(), values,
    'borrowed provider storage is snapshotted exactly once by the result');
  await context.close();
  await compiled.close();
  await runtime.close();
  assert.deepEqual(await result.output('y').read(), values,
    'result ownership is independent from every closed parent');
  await result.close();
});

test('complete shaped inputs bind before provider mutation and a failed bind does not poison FIFO state', async () => {
  const snapshot = identitySnapshot();
  const fixture = externalProvider();
  const { runtime, compiled, context } = await compileFixture(snapshot, fixture);

  await assert.rejects(
    context.execute({}),
    (error) => error.code === 'INVALID_ARGUMENT' && /complete logical shape contract/.test(error.message),
  );
  await assert.rejects(
    context.execute({ x: Float32Array.of(1, 2) }),
    (error) => error.code === 'INVALID_ARGUMENT',
    'raw typed arrays are not a dynamic-v1 compatibility input',
  );
  await assert.rejects(
    context.execute({ x: { data: Float32Array.of(1), shape: [1, 2] } }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.equal(fixture.executeCalls, 0);

  const result = await context.execute({
    x: { data: Float32Array.of(4, 5), shape: [1, 2] },
  });
  assert.equal(fixture.executeCalls, 1);
  assert.deepEqual(await result.output('y').read(), Float32Array.of(4, 5));

  await result.close();
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('unsupported bounded-domain capability is rejected before provider compile or context creation', async () => {
  const snapshot = identitySnapshot();
  const fixture = externalProvider({ support: 'unsupported' });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  await assert.rejects(
    runtime.compile(snapshot, {
      backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
    }),
    (error) => {
      assert.equal(error.code, 'BACKEND_REQUIRED');
      assert.equal(error.report.candidates[0].outcome, 'unsupported');
      assert.equal(error.report.candidates[0].code, 'BACKEND_UNSUPPORTED');
      return true;
    },
  );
  assert.equal(fixture.compileCalls, 0);
  assert.equal(fixture.createContextCalls, 0);
  await runtime.close();
});

test('batch-indexed adapter selections use an explicit resolved dimension and validate before provider mutation', async () => {
  const snapshot = identitySnapshot({ rank2: false });
  const fixture = externalProvider();
  const { runtime, compiled, context } = await compileFixture(
    snapshot,
    fixture,
    { adapterBatchDimension: 'B' },
  );
  const inputs = { x: { data: Float32Array.of(1, 2), shape: [2] } };

  await assert.rejects(
    context.execute(inputs, { adapters: [null] }),
    (error) => error.code === 'INVALID_ARGUMENT' && /resolved batch 2/.test(error.message),
  );
  await assert.rejects(
    context.execute(inputs, { adapters: [null, { name: 'missing' }] }),
    (error) => error.code === 'INVALID_ARGUMENT' && /do not contain adapter revisions/.test(error.message),
  );
  assert.equal(fixture.executeCalls, 0);

  const result = await context.execute(inputs, { adapters: [null, null] });
  assert.equal(fixture.executeCalls, 1);
  assert.equal(fixture.executionRequest.adapters.batchSize, 2);
  assert.deepEqual(fixture.executionRequest.adapters.selectors, [null, null]);

  await result.close();
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('provider context and decode options are validated, cloned, and deeply frozen before mutation', async () => {
  const snapshot = identitySnapshot();
  const fixture = externalProvider();
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'fixture', operatorFallback: 'forbid' },
  });

  await assert.rejects(
    compiled.createContext({ decode: { changedInputs: ['missing'] } }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.equal(fixture.createContextCalls, 0);

  const changedInputs = ['x'];
  const context = await compiled.createContext({
    decode: { changedInputs, rowMode: 'auto', requireIncremental: true },
  });
  changedInputs[0] = 'mutated';
  assert.equal(Object.isFrozen(fixture.providerContextOptions), true);
  assert.equal(Object.isFrozen(fixture.providerContextOptions.initialPlan), true);
  assert.equal(Object.isFrozen(fixture.providerContextOptions.decode), true);
  assert.equal(Object.isFrozen(fixture.providerContextOptions.decode.changedInputs), true);
  assert.deepEqual(fixture.providerContextOptions.decode.changedInputs, ['x']);

  await assert.rejects(
    context.decode.step(
      { x: { data: Float32Array.of(1), shape: [1, 1] } },
      { position: -1 },
    ),
    (error) => error.code === 'INVALID_ARGUMENT' && /non-negative safe integer/.test(error.message),
  );
  assert.equal(fixture.executeCalls, 0);

  await context.close();
  await compiled.close();
  await runtime.close();
});

test('device results expose exact logical bytes plus only mandatory four-byte alignment padding', async () => {
  const snapshot = identitySnapshot({ dtype: 'uint8', rank2: false });
  let releaseCount = 0;
  const fixture = externalProvider({
    outputLocation: 'device',
    execute: async (request, call) => {
      const size = call === 1 ? 8 : 4;
      return {
        outputs: Object.freeze([Object.freeze({
          name: 'y',
          shape: request.outputDescriptors[0].shape,
          dtype: 'uint8',
          location: 'device',
          logicalSizeBytes: 3,
          deviceBuffer: { size },
          async read() { return Uint8Array.of(7, 8, 9); },
          release() { releaseCount++; },
        })]),
      };
    },
  });
  const { runtime, compiled, context } = await compileFixture(snapshot, fixture);
  const inputs = { x: { data: Uint8Array.of(1, 2, 3), shape: [3] } };

  await assert.rejects(
    context.execute(inputs),
    (error) => error.code === 'EXECUTION_FAILED' && /required alignment padding/.test(error.message),
  );
  assert.equal(releaseCount, 1, 'rejected provider-owned device output is released');

  const result = await context.execute(inputs);
  assert.equal(result.output('y').deviceBuffer.size, 4);
  assert.deepEqual(await result.output('y').read(), Uint8Array.of(7, 8, 9));
  await result.close();
  assert.equal(releaseCount, 2, 'accepted aligned output is released with its result');

  await context.close();
  await compiled.close();
  await runtime.close();
});

test('CPU and WASM providers explicitly reject an issued device TensorResult input', async () => {
  const snapshot = identitySnapshot({ rank2: false });
  const fakeDevice = { queue: {} };
  let releaseCount = 0;
  const deviceFixture = externalProvider({
    name: 'device-fixture',
    outputLocation: 'device',
    execute: async (request) => ({
      outputs: Object.freeze([Object.freeze({
        name: 'y',
        shape: request.outputDescriptors[0].shape,
        dtype: 'float32',
        location: 'device',
        logicalSizeBytes: 4,
        deviceType: 'webgpu',
        device: fakeDevice,
        deviceBuffer: { size: 4 },
        async read() { return Float32Array.of(3); },
        release() { releaseCount++; },
      })]),
    }),
  });
  const producer = await compileFixture(snapshot, deviceFixture);
  const sourceResult = await producer.context.execute({
    x: { data: Float32Array.of(3), shape: [1] },
  });

  const cpuRuntime = new Runtime();
  cpuRuntime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const cpuCompiled = await cpuRuntime.compile(snapshot, {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  const cpuContext = await cpuCompiled.createContext();
  await assert.rejects(cpuContext.execute({
    x: { data: sourceResult.output('y'), shape: [1] },
  }), (error) => error.code === 'BACKEND_UNSUPPORTED' &&
    /does not accept device TensorResult inputs/.test(error.message));

  const wasmFixture = externalProvider({ name: 'wasm' });
  const wasm = await compileFixture(snapshot, wasmFixture);
  await assert.rejects(wasm.context.execute({
    x: { data: sourceResult.output('y'), shape: [1] },
  }), (error) => error.code === 'BACKEND_UNSUPPORTED' &&
    /Backend 'wasm' does not accept device TensorResult inputs/.test(error.message));
  assert.equal(wasmFixture.executeCalls, 0,
    'WASM rejects device input before provider mutation');

  await sourceResult.close();
  assert.equal(releaseCount, 1);
  await Promise.all([producer.context.close(), cpuContext.close(), wasm.context.close()]);
  await Promise.all([producer.compiled.close(), cpuCompiled.close(), wasm.compiled.close()]);
  await Promise.all([producer.runtime.close(), cpuRuntime.close(), wasm.runtime.close()]);
});

test('device TensorResult input is nominal, live, and ordinary-execute only', async () => {
  const snapshot = identitySnapshot({ rank2: false });
  const fakeDevice = { queue: {} };
  const fixture = externalProvider({
    name: 'webgpu',
    outputLocation: 'device',
    execute: async (request) => ({
      outputs: Object.freeze([Object.freeze({
        name: 'y', shape: request.outputDescriptors[0].shape, dtype: 'float32',
        location: 'device', logicalSizeBytes: 4, deviceType: 'webgpu', device: fakeDevice,
        deviceBuffer: { size: 4 }, async read() { return Float32Array.of(1); }, release() {},
      })]),
    }),
  });
  const { runtime, compiled, context } = await compileFixture(snapshot, fixture, {
    decode: { changedInputs: ['x'] },
  });
  const sourceResult = await context.execute({
    x: { data: Float32Array.of(1), shape: [1] },
  });
  const source = sourceResult.output('y');

  await assert.rejects(context.decode.seed({ x: { data: source, shape: [1] } }),
    (error) => error.code === 'BACKEND_UNSUPPORTED' &&
      /ordinary execute|does not accept device TensorResult/.test(error.message));
  await sourceResult.close();
  await assert.rejects(context.execute({ x: { data: source, shape: [1] } }),
    (error) => error.code === 'RESULT_DISPOSED');
  const forged = {
    location: 'device', dtype: 'float32', shape: [1], logicalSizeBytes: 4,
    deviceBuffer: { size: 4 },
  };
  await assert.rejects(context.execute({ x: { data: forged, shape: [1] } }),
    (error) => error.code === 'INVALID_ARGUMENT' && /live device TensorResult/.test(error.message));

  const forgedResult = new TensorResult({
    name: 'forged', shape: [1], dtype: 'float32', location: 'device',
    logicalSizeBytes: 4, deviceType: 'webgpu', device: fakeDevice,
    deviceBuffer: { size: 4 },
    async read() { return Float32Array.of(1); },
    release() {},
  }, () => false);
  await assert.rejects(context.execute({ x: { data: forgedResult, shape: [1] } }),
    (error) => error.code === 'INVALID_ARGUMENT' && /live device TensorResult/.test(error.message));
  await forgedResult._close();

  const forgedExecution = new ExecutionResult('webgpu', {
    outputs: [{
      name: 'forged', shape: [1], dtype: 'float32', location: 'device',
      logicalSizeBytes: 4, deviceType: 'webgpu', device: fakeDevice,
      deviceBuffer: { size: 4 },
      async read() { return Float32Array.of(1); },
      release() {},
    }],
  }, {}, [{ name: 'forged', shape: [1], dtype: 'float32' }]);
  await assert.rejects(context.execute({
    x: { data: forgedExecution.output('forged'), shape: [1] },
  }), (error) => error.code === 'INVALID_ARGUMENT' && /live device TensorResult/.test(error.message));
  await forgedExecution.close();

  await context.close();
  await compiled.close();
  await runtime.close();
});

test('public built-in CPU resolves each dynamic execution exactly once and reuses the isolated shape context', async () => {
  const snapshot = identitySnapshot();
  const originalBindShapes = Model.prototype.bindShapes;
  let bindCalls = 0;
  Model.prototype.bindShapes = function bindShapesSpy(inputs) {
    if (this === snapshot) bindCalls++;
    return originalBindShapes.call(this, inputs);
  };

  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  let compiled;
  let context;
  const results = [];
  try {
    compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
    });
    context = await compiled.createContext();
    for (const [shape, values] of [
      [[1, 2], [1, 2]],
      [[2, 3], [3, 4, 5, 6, 7, 8]],
      [[1, 2], [9, 10]],
    ]) {
      results.push(await context.execute({
        x: { data: Float32Array.from(values), shape },
      }));
    }
    assert.equal(bindCalls, 3, 'CPU executeResolved must not invoke snapshot.bindShapes again');
    assert.deepEqual(await results[0].output('y').read(), Float32Array.of(1, 2));
    assert.deepEqual(await results[1].output('y').read(), Float32Array.of(3, 4, 5, 6, 7, 8));
    assert.deepEqual(await results[2].output('y').read(), Float32Array.of(9, 10));
    assert.equal(results[0].report.backendReport.specializationCacheHit, false);
    assert.equal(results[1].report.backendReport.specializationCacheHit, false);
    assert.equal(results[2].report.backendReport.specializationCacheHit, true);
    assert.equal(results[2].report.backendReport.specializationCacheEntries, 2);
    assert.equal(results[2].report.backendReport.specializationCacheHits, 1);
    assert.equal(results[2].report.backendReport.specializationCacheMisses, 2);
    assert.equal(results[0].report.backendReport.logicalActivationBytes, 16);
    assert.equal(results[1].report.backendReport.logicalActivationBytes, 48);
    assert.ok(results[1].report.backendReport.activationCapacityHighWaterBytes >= 48);
    assert.ok(results[1].report.backendReport.activationGrowCount >= 1);
    for (const result of results) {
      assert.ok(Number.isFinite(result.report.shapeBindTimeMs));
      assert.ok(result.report.shapeBindTimeMs >= 0);
      assert.ok(Number.isFinite(result.report.providerTimeMs));
      assert.ok(result.report.providerTimeMs >= 0);
      assert.ok(result.report.executionTimeMs >=
        result.report.shapeBindTimeMs + result.report.providerTimeMs);
    }
    await context.close();
    context = null;
    assert.deepEqual(await results[0].output('y').read(), Float32Array.of(1, 2));
  } finally {
    Model.prototype.bindShapes = originalBindShapes;
    await context?.close();
    await compiled?.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('built-in CPU rejects a shape-valid domain above committed activation capacity', async () => {
  const snapshot = identitySnapshot({ maxB: 70_000_000, rank2: false });
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  await assert.rejects(
    runtime.compile(snapshot, {
      backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
    }),
    (error) => {
      assert.equal(error.code, 'BACKEND_REQUIRED');
      assert.equal(error.report.candidates[0].code, 'BACKEND_UNSUPPORTED');
      assert.equal(error.report.candidates[0].outcome, 'unsupported');
      assert.match(error.report.candidates[0].message, /committed-capacity limit/);
      return true;
    },
  );
  await runtime.close();
});

test('built-in CPU resource attestation includes transactional dynamic-slot growth', async () => {
  const snapshot = identitySnapshot({ maxB: 4, rank2: false });
  const provider = new CPUBackendProvider(new CPUEngine());
  const compiled = await provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  });
  const resources = compiled.compilationEvidence.shapeDomain;
  assert.equal('shapeSystem' in resources, false);
  assert.equal(resources.maximumTensorBytes, 16);
  assert.equal(
    resources.maximumResidentBytes,
    64,
    'one 32-byte liveness arena accounts for committed plus growth-candidate revisions',
  );
  assert.equal(resources.maximumActivationArenaBytes, 32);
  assert.equal(resources.maximumPersistentActivationBytes, 32);
  assert.equal(resources.maximumInputBytes, 16);
  assert.equal(resources.maximumDecodeResidentBytes, 96,
    'decode growth includes the old and replacement logical public-input snapshots');
  assert.equal(resources.capacityLimitBytes, 512 * 1024 * 1024);
  assert.equal(resources.residentLimitBytes, 1024 * 1024 * 1024);
  assert.equal(resources.resourceLimitBytes, resources.residentLimitBytes);
  await compiled.close();
  await provider.close();
});

test('CPU retained decode rejects committed capacity before allocating persistent slots', async () => {
  const mib = 1024 * 1024;
  const snapshot = identityChainSnapshot({ tensorBytes: 48 * mib, nodeCount: 10 });
  const provider = new CPUBackendProvider(new CPUEngine());
  const compiled = await provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  });
  const resources = compiled.compilationEvidence.shapeDomain;
  assert.equal(resources.maximumActivationArenaBytes, 96 * mib);
  assert.equal(resources.maximumPersistentActivationBytes, 528 * mib);
  assert.equal(resources.decodeContextSupported, false);
  const invariantResources = compiled.invariantResources.open();
  assert.throws(
    () => compiled.createContext({
      invariantResources,
      decode: { changedInputs: ['x'] },
    }),
    (error) => error?.code === 'BACKEND_UNSUPPORTED' &&
      /committed-capacity limit/.test(error.message),
  );
  invariantResources.release();
  await compiled.close();
  await provider.close();
});

test('CPU retained decode separately rejects resident transaction pressure', async () => {
  const mib = 1024 * 1024;
  const snapshot = identityChainSnapshot({
    tensorBytes: 51 * mib,
    nodeCount: 9,
    weightBytes: 3 * mib,
  });
  const provider = new CPUBackendProvider(new CPUEngine());
  const compiled = await provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  });
  const resources = compiled.compilationEvidence.shapeDomain;
  assert.equal(resources.maximumActivationArenaBytes, 102 * mib);
  assert.equal(resources.maximumPersistentActivationBytes, 510 * mib);
  assert.equal(resources.maximumInputBytes, 51 * mib);
  assert.equal(resources.maximumDecodeResidentBytes, 1128 * mib);
  assert.ok(resources.maximumPersistentActivationBytes <= resources.capacityLimitBytes);
  assert.ok(resources.maximumDecodeResidentBytes > resources.residentLimitBytes);
  assert.equal(resources.decodeContextSupported, false);
  const invariantResources = compiled.invariantResources.open();
  assert.throws(
    () => compiled.createContext({
      invariantResources,
      decode: { changedInputs: ['x'] },
    }),
    (error) => error?.code === 'BACKEND_UNSUPPORTED' &&
      /resident\/transaction limit/.test(error.message),
  );
  invariantResources.release();
  await compiled.close();
  await provider.close();
});
