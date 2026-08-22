import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  Model,
  Runtime,
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  VolvoxAI,
  createBackendProviderBatchContract,
  createBackendProviderCapabilities,
  parseGraphDocument,
} from '../ts/index.js';
import { InvariantResourceStore } from '../ts/backends/InvariantResources.js';

function identitySnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B'] } },
    nodes: [{
      id: 'identity', opType: 'Identity', inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B'] } }, params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function hostOutput(request, value, ownership = 'borrowed') {
  return Object.freeze({
    name: 'y',
    shape: request.outputDescriptors[0].shape,
    dtype: 'float32',
    location: 'host',
    ownership,
    data: Float32Array.from(value),
  });
}

function shapeDomain(input, overrides = {}) {
  return Object.freeze({
    proofProtocol: 'canonical-symbolic-domain-proof/v1',
    resourceProtocol: 'bounded-resource-maxima/v1',
    support: 'full',
    graphFingerprint: input.graphFingerprint,
    proof: input.shapeDomainProof,
    maximumTensorBytes: 16,
    maximumResidentBytes: 32,
    resourceLimitBytes: 1024,
    ...overrides,
  });
}

function providerFixture({
  name = 'fixture',
  support = 'full',
  operatorFallback = 'none',
  execute = async (request, contextId) => ({
    outputs: Object.freeze([
      hostOutput(request, Array.from(request.inputs.x.data, (value) => value + contextId)),
    ]),
  }),
  compileEvidence = null,
  createContext = null,
  batchContract = createBackendProviderBatchContract('unsupported'),
  invariantResourcesFactory = () => new InvariantResourceStore(() => 0),
  invariantResourcesGetter = null,
  compiledClose = null,
  compiledCloseGetter = null,
} = {}) {
  const events = [];
  let compileCalls = 0;
  let contextCalls = 0;
  let executeCalls = 0;
  const resourceDomain = Object.freeze({});
  const provider = {
    providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
    backendName: name,
    capabilities: createBackendProviderCapabilities({
      operatorFallback,
      outputLocation: 'host',
      dynamicShapeDomain: support,
    }),
    async compile(input, options) {
      compileCalls++;
      events.push(['compile', options]);
      const compatibilityTokens = new Map();
      const invariantResources = invariantResourcesFactory();
      let invariantOwnerReads = 0;
      const compiled = {
        backendName: name,
        batchContract,
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
        compilationEvidence: compileEvidence?.(input) ?? Object.freeze({
          device: null,
          allocationBytes: 0,
          operatorFallbackUsed: false,
          offendingNode: null,
          shapeDomain: shapeDomain(input),
        }),
        async createContext(options = {}) {
          contextCalls++;
          const contextId = contextCalls;
          events.push(['create-context', contextId, options]);
          if (createContext) return createContext({ contextId, events, input, options });
          return {
            backendName: name,
            async execute(request) {
              executeCalls++;
              events.push(['execute', contextId, request.signature]);
              return execute(request, contextId, executeCalls);
            },
            async close() { events.push(['context-close', contextId]); },
          };
        },
        async close() {
          events.push(['compiled-close']);
          await compiledClose?.(invariantResources);
        },
      };
      Object.defineProperty(compiled, 'invariantResources', invariantResourcesGetter === null
        ? { enumerable: true, value: invariantResources }
        : {
          enumerable: true,
          get() {
            return invariantResourcesGetter(invariantResources, invariantOwnerReads++);
          },
        });
      if (compiledCloseGetter !== null) {
        let compiledCloseReads = 0;
        Object.defineProperty(compiled, 'close', {
          enumerable: true,
          configurable: true,
          get() { return compiledCloseGetter(compiledCloseReads++); },
        });
      }
      return compiled;
    },
    async close() { events.push(['provider-close']); },
  };
  return {
    provider,
    events,
    get compileCalls() { return compileCalls; },
    get contextCalls() { return contextCalls; },
    get executeCalls() { return executeCalls; },
  };
}

async function compile(runtime, snapshot, backend = 'fixture', mode = 'require') {
  return runtime.compile(snapshot, {
    backend: mode === 'require'
      ? { mode, backend, operatorFallback: 'forbid' }
      : { mode, order: backend, operatorFallback: 'forbid' },
  });
}

test('parent closure is logical immediately and physical cleanup drains retained descendants', async () => {
  const fixture = providerFixture();
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await compile(runtime, identitySnapshot());
  const context = await compiled.createContext();

  let compiledSettled = false;
  const compiledClose = compiled.close().finally(() => { compiledSettled = true; });
  await Promise.resolve();
  assert.equal(compiledSettled, false);
  await assert.rejects(compiled.createContext(), (error) => error.code === 'HANDLE_DISPOSED');

  let runtimeSettled = false;
  const runtimeClose = runtime.close().finally(() => { runtimeSettled = true; });
  await Promise.resolve();
  assert.equal(runtimeSettled, false);
  await assert.rejects(
    runtime.compile(identitySnapshot()),
    (error) => error.code === 'HANDLE_DISPOSED',
  );

  const result = await context.execute({ x: { data: Float32Array.of(2), shape: [1] } });
  assert.deepEqual(await result.output('y').read(), Float32Array.of(3));
  await context.close();
  await compiledClose;
  await runtimeClose;
  assert.equal(compiledSettled, true);
  assert.equal(runtimeSettled, true);
  assert.deepEqual(await result.output('y').read(), Float32Array.of(3));
  assert.deepEqual(fixture.events.slice(-3), [
    ['context-close', 1], ['compiled-close'], ['provider-close'],
  ]);
  await result.close();
});

test('compiled model rejects a provider that swaps its invariant owner after validation', async () => {
  const stable = new InvariantResourceStore(() => 0);
  const swapped = new InvariantResourceStore(() => 0);
  const fixture = providerFixture({
    invariantResourcesFactory: () => stable,
    invariantResourcesGetter: (owner, read) => read === 0 ? owner : swapped,
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await compile(runtime, identitySnapshot());

  await assert.rejects(compiled.createContext(),
    (error) => error?.code === 'ABI_UNSUPPORTED' &&
      /changed its compiled invariant resource owner/.test(error.message));
  assert.equal(stable.borrowerCount, 0);
  assert.equal(swapped.borrowerCount, 0);

  await compiled.close();
  assert.throws(() => stable.open(), /compiled model is closed/);
  swapped.close();
  await runtime.close();
});

test('malformed invariant lease is released without replacing the validation failure', async () => {
  let owner;
  const fixture = providerFixture({
    invariantResourcesFactory() {
      owner = new InvariantResourceStore(() => 0);
      const open = owner.open.bind(owner);
      owner.open = () => {
        const lease = open();
        return Object.freeze({
          protocol: lease.protocol,
          ownerIdentity: lease.ownerIdentity,
          deviceEpoch: Object.freeze({ wrong: 'epoch' }),
          get borrowedResourceCount() { return lease.borrowedResourceCount; },
          get borrowedBytes() { return lease.borrowedBytes; },
          release() {
            lease.release();
            throw new Error('hostile malformed lease release');
          },
        });
      };
      return owner;
    },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await compile(runtime, identitySnapshot());

  await assert.rejects(compiled.createContext(),
    (error) => error?.code === 'ABI_UNSUPPORTED' &&
      /invalid invariant resource lease/.test(error.message) &&
      !/hostile malformed lease release/.test(error.message));
  assert.equal(owner.borrowerCount, 0);

  await compiled.close();
  await runtime.close();
});

test('post-validation compile failure closes candidate and captured owner exactly once', async () => {
  let allocationReads = 0;
  let compiledCloseCalls = 0;
  let ownerCloseCalls = 0;
  let owner;
  const fixture = providerFixture({
    invariantResourcesFactory() {
      owner = new InvariantResourceStore(() => 0);
      const close = owner.close.bind(owner);
      owner.close = () => {
        ownerCloseCalls++;
        return close();
      };
      return owner;
    },
    compileEvidence(input) {
      const evidence = {
        device: null,
        operatorFallbackUsed: false,
        offendingNode: null,
        shapeDomain: shapeDomain(input),
      };
      Object.defineProperty(evidence, 'allocationBytes', {
        enumerable: true,
        get() {
          allocationReads++;
          if (allocationReads > 3) {
            throw new Error('injected post-validation evidence failure');
          }
          return 0;
        },
      });
      return Object.freeze(evidence);
    },
    compiledClose() { compiledCloseCalls++; },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);

  await assert.rejects(compile(runtime, identitySnapshot()),
    (error) => error?.code === 'BACKEND_REQUIRED' &&
      error.report?.candidates?.[0]?.message.includes('injected post-validation evidence failure'));
  assert.equal(compiledCloseCalls, 1);
  assert.equal(ownerCloseCalls, 1);
  assert.equal(owner.borrowerCount, 0);
  assert.throws(() => owner.open(), /compiled model is closed/);

  await runtime.close();
});

test('compiled-contract rejection closes an owner captured before later validation', async () => {
  let compiledCloseCalls = 0;
  let ownerCloseCalls = 0;
  let owner;
  const fixture = providerFixture({
    batchContract: Object.freeze({ protocol: 'malformed-after-owner-capture' }),
    invariantResourcesFactory() {
      owner = new InvariantResourceStore((value) => value.byteLength);
      owner.define('weight', new Uint8Array(16));
      const close = owner.close.bind(owner);
      owner.close = () => {
        ownerCloseCalls++;
        return close();
      };
      return owner;
    },
    compiledClose() { compiledCloseCalls++; },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);

  await assert.rejects(compile(runtime, identitySnapshot()),
    (error) => error?.code === 'BACKEND_REQUIRED' &&
      error.report?.candidates?.[0]?.code === 'ABI_UNSUPPORTED' &&
      /invalid batch contract/.test(error.report.candidates[0].message));
  assert.equal(compiledCloseCalls, 1);
  assert.equal(ownerCloseCalls, 1);
  assert.equal(owner.resourceCount, 0);
  assert.equal(owner.ownedBytes, 0);
  assert.throws(() => owner.open(), /compiled model is closed/);

  await runtime.close();
});

test('hostile compiled close accessor cannot strand a captured invariant owner', async () => {
  let ownerCloseCalls = 0;
  let owner;
  const fixture = providerFixture({
    batchContract: Object.freeze({ protocol: 'malformed-after-owner-capture' }),
    invariantResourcesFactory() {
      owner = new InvariantResourceStore((value) => value.byteLength);
      owner.define('weight', new Uint8Array(16));
      const close = owner.close.bind(owner);
      owner.close = () => {
        ownerCloseCalls++;
        return close();
      };
      return owner;
    },
    compiledCloseGetter(read) {
      if (read === 0) return async () => {};
      throw new Error('hostile compiled close accessor');
    },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);

  await assert.rejects(compile(runtime, identitySnapshot()),
    (error) => error?.code === 'BACKEND_REQUIRED' &&
      error.report?.candidates?.[0]?.code === 'ABI_UNSUPPORTED' &&
      /invalid batch contract/.test(error.report.candidates[0].message) &&
      !/hostile compiled close accessor/.test(error.report.candidates[0].message));
  assert.equal(ownerCloseCalls, 1);
  assert.equal(owner.resourceCount, 0);
  assert.throws(() => owner.open(), /compiled model is closed/);

  await runtime.close();
});

test('core closes the captured invariant owner when provider compiled close throws', async () => {
  let owner;
  const fixture = providerFixture({
    invariantResourcesFactory() {
      owner = new InvariantResourceStore((value) => value.byteLength);
      owner.define('w', new Uint8Array(32));
      return owner;
    },
    compiledClose() { throw new Error('injected provider compiled close failure'); },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await compile(runtime, identitySnapshot());

  await assert.rejects(compiled.close(), /injected provider compiled close failure/);
  assert.equal(owner.resourceCount, 0);
  assert.equal(owner.ownedBytes, 0);
  assert.throws(() => owner.open(), /compiled model is closed/);
  await runtime.close();
});

test('one public context queues accepted execution and close in FIFO order', async () => {
  let releaseFirst;
  const firstGate = new Promise((resolve) => { releaseFirst = resolve; });
  const fixture = providerFixture({
    execute: async (request, contextId, call) => {
      if (call === 1) await firstGate;
      return { outputs: Object.freeze([hostOutput(request, [request.inputs.x.data[0] + contextId])]) };
    },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await compile(runtime, identitySnapshot());
  const context = await compiled.createContext();
  const first = context.execute({ x: { data: Float32Array.of(1), shape: [1] } });
  const second = context.execute({ x: { data: Float32Array.of(5), shape: [1] } });
  const close = context.close();
  await assert.rejects(
    context.execute({ x: { data: Float32Array.of(9), shape: [1] } }),
    (error) => error.code === 'HANDLE_DISPOSED',
  );
  await Promise.resolve();
  assert.equal(fixture.executeCalls, 1);
  releaseFirst();
  const [firstResult, secondResult] = await Promise.all([first, second]);
  await close;
  assert.equal(fixture.executeCalls, 2);
  assert.deepEqual(await firstResult.output('y').read(), Float32Array.of(2));
  assert.deepEqual(await secondResult.output('y').read(), Float32Array.of(6));
  await Promise.all([firstResult.close(), secondResult.close()]);
  await compiled.close();
  await runtime.close();
});

test('two contexts retain independent identities and execute concurrently', async () => {
  const fixture = providerFixture();
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await compile(runtime, identitySnapshot());
  const firstContext = await compiled.createContext();
  const secondContext = await compiled.createContext();
  const [first, second] = await Promise.all([
    firstContext.execute({ x: { data: Float32Array.of(10), shape: [1] } }),
    secondContext.execute({ x: { data: Float32Array.of(10), shape: [1] } }),
  ]);
  assert.notEqual(firstContext.id, secondContext.id);
  assert.deepEqual(await first.output('y').read(), Float32Array.of(11));
  assert.deepEqual(await second.output('y').read(), Float32Array.of(12));
  await Promise.all([first.close(), second.close(), firstContext.close(), secondContext.close()]);
  await compiled.close();
  await runtime.close();
});

test('backend policy records unsupported tiers and selects one final proved provider', async () => {
  const unsupported = providerFixture({ name: 'unsupported', support: 'unsupported' });
  const selected = providerFixture({ name: 'selected' });
  const diagnostics = [];
  const runtime = new Runtime({ onDiagnostic: (event) => diagnostics.push(event) });
  runtime._addProvider('unsupported', unsupported.provider);
  runtime._addProvider('selected', selected.provider);
  const compiled = await compile(
    runtime,
    identitySnapshot(),
    ['unsupported', 'selected'],
    'prefer',
  );
  assert.equal(unsupported.compileCalls, 0);
  assert.equal(selected.compileCalls, 1);
  assert.equal(compiled.backend, 'selected');
  assert.deepEqual(compiled.report.candidates.map(({ backend, outcome }) => ({ backend, outcome })), [
    { backend: 'unsupported', outcome: 'unsupported' },
    { backend: 'selected', outcome: 'selected' },
  ]);
  assert.equal(compiled.report.routeEvidence.tierFallback, true);
  assert.doesNotThrow(() => JSON.stringify(compiled.report));
  assert.equal(diagnostics[0].kind, 'compilation');
  await compiled.close();
  await runtime.close();

  const reported = providerFixture({ name: 'reported', operatorFallback: 'reported' });
  const strictRuntime = new Runtime();
  strictRuntime._addProvider('reported', reported.provider);
  await assert.rejects(
    compile(strictRuntime, identitySnapshot(), 'reported'),
    (error) => error.code === 'OPERATOR_FALLBACK_FORBIDDEN' &&
      error.report.candidates[0].code === 'OPERATOR_FALLBACK_FORBIDDEN',
  );
  assert.equal(reported.compileCalls, 0);
  await strictRuntime.close();
});

test('invalid provider evidence and contexts are reclaimed without accepting execution', async () => {
  let invalidBatchClosed = 0;
  const invalidBatch = providerFixture({
    name: 'invalid-batch',
    batchContract: Object.freeze({
      protocol: 'dense-public-batch/v1',
      densePublicBatch: 'single-invocation',
      independentBatch: 'compiler-proved/v1',
      deviceResident: true,
      hostFallback: 'not-applicable',
    }),
  });
  const invalidBatchCompile = invalidBatch.provider.compile;
  invalidBatch.provider.compile = async (...args) => {
    const candidate = await invalidBatchCompile(...args);
    candidate.close = async () => { invalidBatchClosed++; };
    return candidate;
  };
  const batchRuntime = new Runtime();
  batchRuntime._addProvider('invalid-batch', invalidBatch.provider);
  await assert.rejects(
    compile(batchRuntime, identitySnapshot(), 'invalid-batch'),
    (error) => error.code === 'BACKEND_REQUIRED' &&
      error.report.candidates[0].code === 'ABI_UNSUPPORTED',
  );
  assert.equal(invalidBatchClosed, 1);
  assert.equal(invalidBatch.contextCalls, 0);
  await batchRuntime.close();

  let invalidCompiledClosed = 0;
  const invalidEvidence = providerFixture({
    name: 'invalid-evidence',
    compileEvidence: (input) => Object.freeze({
      device: null,
      allocationBytes: 0,
      shapeDomain: shapeDomain(input, {
        maximumTensorBytes: 64,
        maximumResidentBytes: 32,
      }),
    }),
  });
  const originalCompile = invalidEvidence.provider.compile;
  invalidEvidence.provider.compile = async (...args) => {
    const candidate = await originalCompile(...args);
    candidate.close = async () => { invalidCompiledClosed++; };
    return candidate;
  };
  const firstRuntime = new Runtime();
  firstRuntime._addProvider('invalid-evidence', invalidEvidence.provider);
  await assert.rejects(
    compile(firstRuntime, identitySnapshot(), 'invalid-evidence'),
    (error) => error.code === 'BACKEND_REQUIRED' &&
      error.report.candidates[0].code === 'ABI_UNSUPPORTED',
  );
  assert.equal(invalidCompiledClosed, 1);
  assert.equal(invalidEvidence.contextCalls, 0);
  await firstRuntime.close();

  let invalidContextClosed = 0;
  const invalidContext = providerFixture({
    name: 'invalid-context',
    createContext: () => ({
      backendName: 'wrong-name',
      async execute() { throw new Error('must not execute'); },
      async close() { invalidContextClosed++; },
    }),
  });
  const secondRuntime = new Runtime();
  secondRuntime._addProvider('invalid-context', invalidContext.provider);
  const compiled = await compile(secondRuntime, identitySnapshot(), 'invalid-context');
  await assert.rejects(
    compiled.createContext(),
    (error) => error.code === 'ABI_UNSUPPORTED',
  );
  assert.equal(invalidContextClosed, 1);
  await compiled.close();
  await secondRuntime.close();
});

test('execution failures carry frozen revision and resolved-shape diagnostics without poisoning the queue', async () => {
  const diagnostics = [];
  const fixture = providerFixture({
    execute: async (request, contextId, call) => {
      if (call === 1) throw new Error('fixture dispatch failed');
      return { outputs: Object.freeze([hostOutput(request, [contextId])]) };
    },
  });
  const runtime = new Runtime({ onDiagnostic: (event) => diagnostics.push(event) });
  runtime._addProvider('fixture', fixture.provider);
  const snapshot = identitySnapshot();
  const compiled = await compile(runtime, snapshot);
  const context = await compiled.createContext();
  await assert.rejects(
    context.execute({ x: { data: Float32Array.of(1, 2), shape: [2] } }),
    (error) => {
      assert.equal(error.code, 'EXECUTION_FAILED');
      assert.equal(error.report.weightRevisionId, snapshot.weightRevisionId);
      assert.equal(typeof error.report.shapeSignature, 'string');
      assert.doesNotThrow(() => JSON.stringify(error.report));
      return true;
    },
  );
  const result = await context.execute({ x: { data: Float32Array.of(3), shape: [1] } });
  assert.deepEqual(await result.output('y').read(), Float32Array.of(1));
  assert.deepEqual(diagnostics.map(({ kind }) => kind), [
    'compilation', 'execution-error', 'execution',
  ]);
  await result.close();
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('ordinary provider commit clears retained decode before every post-commit failure', async () => {
  let failurePhase = 'pre-commit';
  let decodeStepCalls = 0;
  let lateCommit = null;
  const fixture = providerFixture({
    createContext: () => {
      const snapshot = (request) => Object.freeze({
        outputs: Object.freeze([hostOutput(request, request.inputs.x.data)]),
      });
      return {
        backendName: 'fixture',
        async execute(request) {
          lateCommit = request.commitExecution;
          if (failurePhase === 'pre-commit') throw new Error('pre-commit failure');
          request.commitExecution();
          throw new Error('post-commit failure');
        },
        async decodeSeed(request) { return snapshot(request); },
        async decodeStep(request) {
          decodeStepCalls++;
          return snapshot(request);
        },
        async close() {},
      };
    },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await compile(runtime, identitySnapshot());
  const context = await compiled.createContext({
    decode: { changedInputs: ['x'], rowMode: 'disabled', requireIncremental: true },
  });
  const input = { x: { data: Float32Array.of(1), shape: [1] } };
  const results = [];
  try {
    results.push(await context.decode.seed(input));
    await assert.rejects(context.execute(input), /pre-commit failure/);
    results.push(await context.decode.step(input));
    assert.equal(decodeStepCalls, 1,
      'a pre-commit failure preserves core-retained inputs and reaches the provider step');
    lateCommit();
    results.push(await context.decode.step(input));
    assert.equal(decodeStepCalls, 2,
      'a settled provider request cannot commit late and clear newer decode state');

    results.push(await context.decode.seed(input));
    failurePhase = 'post-commit';
    await assert.rejects(context.execute(input), /post-commit failure/);
    await assert.rejects(context.decode.step(input),
      (error) => error?.code === 'INVALID_ARGUMENT' && /successful seed/.test(error.message));
    assert.equal(decodeStepCalls, 2,
      'a post-commit failure clears core state before another provider call');
  } finally {
    await Promise.all(results.map((result) => result.close()));
    await context.close();
    await compiled.close();
    await runtime.close();
  }
});

test('decode provider commit preserves state before mutation and clears it after seed/step mutation', async () => {
  let failure = null;
  let seedCalls = 0;
  let stepCalls = 0;
  const fixture = providerFixture({
    createContext: () => {
      const snapshot = (request) => Object.freeze({
        outputs: Object.freeze([hostOutput(request, request.inputs.x.data)]),
      });
      const run = async (request, operation) => {
        if (operation === 'seed') seedCalls++;
        else stepCalls++;
        if (failure === `${operation}-pre`) throw new Error(`${failure} failure`);
        if (failure === `${operation}-post`) {
          request.commitExecution();
          throw new Error(`${failure} failure`);
        }
        return snapshot(request);
      };
      return {
        backendName: 'fixture',
        async execute(request) { return snapshot(request); },
        async decodeSeed(request) { return run(request, 'seed'); },
        async decodeStep(request) { return run(request, 'step'); },
        async close() {},
      };
    },
  });
  const runtime = new Runtime();
  runtime._addProvider('fixture', fixture.provider);
  const compiled = await compile(runtime, identitySnapshot());
  const context = await compiled.createContext({
    decode: { changedInputs: ['x'], rowMode: 'disabled', requireIncremental: true },
  });
  const input = (value) => ({ x: { data: Float32Array.of(value), shape: [1] } });
  const results = [];
  try {
    results.push(await context.decode.seed(input(1)));

    failure = 'seed-pre';
    await assert.rejects(context.decode.seed(input(2)), /seed-pre failure/);
    failure = null;
    results.push(await context.decode.step(input(3)));
    assert.equal(stepCalls, 1, 'a pre-commit reseed failure preserves the prior seed');

    failure = 'step-pre';
    await assert.rejects(context.decode.step(input(4)), /step-pre failure/);
    failure = null;
    results.push(await context.decode.step(input(5)));
    assert.equal(stepCalls, 3, 'a pre-commit step failure preserves the prior seed');

    failure = 'seed-post';
    await assert.rejects(context.decode.seed(input(6)), /seed-post failure/);
    failure = null;
    const stepCallsAfterSeedFailure = stepCalls;
    await assert.rejects(context.decode.step(input(7)),
      (error) => error?.code === 'INVALID_ARGUMENT' && /successful seed/.test(error.message));
    assert.equal(stepCalls, stepCallsAfterSeedFailure,
      'a post-commit reseed failure clears core state before another provider call');

    results.push(await context.decode.seed(input(8)));
    failure = 'step-post';
    await assert.rejects(context.decode.step(input(9)), /step-post failure/);
    failure = null;
    const stepCallsAfterStepFailure = stepCalls;
    await assert.rejects(context.decode.step(input(10)),
      (error) => error?.code === 'INVALID_ARGUMENT' && /successful seed/.test(error.message));
    assert.equal(stepCalls, stepCallsAfterStepFailure,
      'a post-commit step failure clears core state before another provider call');
    assert.equal(seedCalls, 4);
  } finally {
    await Promise.all(results.map((result) => result.close()));
    await context.close();
    await compiled.close();
    await runtime.close();
  }
});

test('external provider factories are scoped to their owning Runtime and options reject invalid composition', async () => {
  const name = 'context-runtime-fixture';
  const fixture = providerFixture({ name });
  const runtime = await VolvoxAI.createRuntime({
    backends: [name],
    providers: {
      [name]: async ({ name: requested }) => {
        assert.equal(requested, name);
        return fixture.provider;
      },
    },
  });
  const compiled = await compile(runtime, identitySnapshot(), name);
  const context = await compiled.createContext();
  const result = await context.execute({ x: { data: Float32Array.of(1), shape: [1] } });
  assert.deepEqual(await result.output('y').read(), Float32Array.of(2));
  await result.close();
  await context.close();
  await compiled.close();
  await runtime.close();
  assert.deepEqual(fixture.events.at(-1), ['provider-close']);

  await assert.rejects(
    VolvoxAI.createRuntime({ backends: [name] }),
    (error) => error.code === 'INVALID_ARGUMENT' && /Unknown backend provider/.test(error.message),
  );
  await assert.rejects(
    VolvoxAI.createRuntime({ backends: name }),
    (error) => error.code === 'INVALID_ARGUMENT' && /non-empty ordered array/.test(error.message),
  );
  await assert.rejects(
    VolvoxAI.createRuntime({ backends: ['cpu-js', 'cpu-js'] }),
    (error) => error.code === 'INVALID_ARGUMENT' && /occurs more than once/.test(error.message),
  );
});

test('built-in JavaScript CPU uses cpu-js and rejects the removed cpu identifier', async () => {
  await assert.rejects(
    VolvoxAI.createRuntime({ backends: ['cpu'] }),
    (error) => error.code === 'INVALID_ARGUMENT' &&
      /Unknown backend provider 'cpu'/.test(error.message),
  );

  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu-js'] });
  let compiled;
  let context;
  let result;
  try {
    assert.deepEqual(runtime.listBackends(), ['cpu-js']);
    compiled = await compile(runtime, identitySnapshot(), 'cpu-js');
    assert.equal(compiled.backend, 'cpu-js');
    assert.equal(compiled.report.selectedBackend, 'cpu-js');
    context = await compiled.createContext();
    result = await context.execute({ x: { data: Float32Array.of(3), shape: [1] } });
    assert.equal(result.backend, 'cpu-js');
    assert.equal(result.report.backend, 'cpu-js');
    assert.deepEqual(await result.output('y').read(), Float32Array.of(3));
  } finally {
    await result?.close();
    await context?.close();
    await compiled?.close();
    await runtime.close();
  }
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
    if (navigatorDescriptor) Object.defineProperty(globalThis, 'navigator', navigatorDescriptor);
    else delete globalThis.navigator;
  });

  const runtime = await VolvoxAI.createRuntime({ backends: ['webgpu'] });
  assert.equal(requestCount, 2);
  assert.deepEqual(runtime.listBackends(), ['webgpu']);
  const provider = runtime;
  await assert.rejects(
    provider.compile(identitySnapshot(), {
      backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    }),
    (error) => error.code === 'BACKEND_REQUIRED' &&
      error.report.candidates[0].outcome === 'unsupported',
  );
  await runtime.close();
});

test('inference entry exposes the provider lifecycle without legacy Model or engine facades', async () => {
  const api = await import('../ts/index.js');
  for (const hidden of [
    'BackendEngine', 'DecodeSession', 'CPUEngine', 'WasmEngine', 'WebGPUEngine',
    'GraphExecutor', 'AdapterManager', 'ModelSnapshot',
    'RuntimeGraph', 'Tensor', 'RuntimeGraphBuilder', 'RuntimeGraphLoader',
    'PagedKVCache', 'PagedKVError', 'runtimeError', 'parseStrictJSON',
  ]) {
    assert.equal(hidden in api, false, `${hidden} must remain internal`);
  }
  for (const exposed of [
    'Runtime', 'CompiledModel', 'ExecutionContext', 'ExecutionResult', 'TensorResult',
    'ModelLoader', 'ModelBuilder', 'Model',
    'ShapeEnvironment', 'ResolvedShapePlanError', 'VolvoxAI',
    'VOLVOXAI_BACKEND_PROVIDER_VERSION',
  ]) {
    assert.equal(exposed in api, true, `${exposed} must be public`);
  }
  assert.equal(VOLVOXAI_BACKEND_PROVIDER_VERSION, 1);
  assert.equal(Object.isFrozen(VolvoxAI), true);
  assert.deepEqual(Object.keys(VolvoxAI).sort(), ['createRuntime']);

  const source = await readFile(new URL('../ts/index.ts', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /export\s+type\s+\*\s+from\s+['"]\.\/types\.js['"]/,
    'inference types must use an allowlist instead of leaking the legacy contract');
  for (const hiddenType of ['GraphDocument', 'GraphContract', 'TensorLike', 'GraphNode']) {
    assert.doesNotMatch(source, new RegExp(`\\b${hiddenType}\\b`),
      `${hiddenType} must not be inference-exported`);
  }
});

test('legacy Graph-backed Model API is absent from Runtime and the inference root', async () => {
  const api = await import('../ts/index.js');
  for (const method of ['_createLegacyModel', '_loadLegacyModel']) {
    assert.equal(method in Runtime.prototype, false,
      `Runtime.prototype must not expose ${method}`);
  }
  // Model is the immutable logical handle, not the retired Graph-backed class:
  // it is captured from a package and never mutated through graph edits.
  assert.equal(typeof api.Model.capture, 'function');
  assert.equal(typeof api.Model.derive, 'function');
  for (const mutator of ['addNode', 'addOp', 'addWeight', 'setOutputs', 'stageAdapter']) {
    assert.equal(mutator in api.Model.prototype, false,
      `Model.prototype must not expose the graph mutator ${mutator}`);
  }
});
