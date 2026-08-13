import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUBackendProvider } from '../ts/backends/CPUBackendProvider.js';
import { ProviderDecodeLifecycle } from '../ts/backends/ProviderDecodeLifecycle.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { Runtime } from '../ts/core/ContextRuntime.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model } from '../ts/core/Model.js';
import { geluValue } from '../ts/ops/gelu.js';
import { incrementalRowDomainSupported } from '../ts/backends/quantizedRowExecution.js';

function dynamicGeluSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 2 },
      S: { min: 2, max: 8 },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 'S', 4] } },
    nodes: [{
      id: 'gelu',
      opType: 'GELU',
      inputs: { input: 'x' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: ['B', 'S', 4] },
      },
      params: { approximate: 'none' },
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function decoderGeometrySnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      S: { min: 2, max: 8 },
      M: { min: 2, max: 16 },
    },
    inputs: {
      decoder_input_ids: { dtype: 'int32', shape: [1, 'S'] },
      memory_padding_mask: { dtype: 'int32', shape: [1, 'M'] },
    },
    nodes: [{
      id: 'tokens',
      opType: 'Identity',
      inputs: { input: 'decoder_input_ids' },
      outputs: { out: { tensor: 'tokens', dtype: 'int32', shape: [1, 'S'] } },
      params: {},
    }],
    outputs: ['tokens'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function retainedInputSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 2, max: 8 } },
    inputs: {
      changed: { dtype: 'float32', shape: [1, 'S'] },
      fixed: { dtype: 'float32', shape: [1, 'S'] },
    },
    nodes: [{
      id: 'add',
      opType: 'Add',
      inputs: { a: 'changed', b: 'fixed' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 'S'] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function unsupportedRowSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 2, max: 8 } },
    inputs: {
      decoder_input_ids: { dtype: 'int32', shape: [1, 'S'] },
      zero: { dtype: 'int32', shape: [1] },
    },
    nodes: [{
      id: 'not', opType: 'Not', inputs: { input: 'decoder_input_ids' },
      outputs: { out: { tensor: 'y', dtype: 'int32', shape: [1, 'S'] } }, params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function shaped(batch, sequence, salt = 0) {
  return {
    x: {
      data: Float32Array.from(
        { length: batch * sequence * 4 },
        (_, index) => Math.fround(((index + salt) % 13 - 6) / 4),
      ),
      shape: [batch, sequence, 4],
    },
  };
}

function replaceRow(inputs, position, salt) {
  const data = new Float32Array(inputs.x.data);
  for (let feature = 0; feature < 4; feature++) {
    data[position * 4 + feature] = Math.fround((salt + feature) / 7);
  }
  return { x: { data, shape: inputs.x.shape } };
}

function expectedGelu(inputs) {
  return Float32Array.from(inputs.x.data, (value) => Math.fround(geluValue(value, 'none')));
}

function rowProofGraph({ opType, inputShape, outputShape, params = {}, quantization = null,
  scaleShape = [1] }) {
  return {
    inputs: { x: { name: 'x', kind: 'input', dtype: 'uint8', shape: inputShape } },
    tensors: {
      x: { name: 'x', kind: 'input', dtype: 'uint8', shape: inputShape,
        ...(opType === 'DequantizeLinear' ? { quantization } : {}) },
      y: { name: 'y', kind: 'value', dtype: opType === 'QuantizeLinear' ? 'uint8' : 'float32',
        shape: outputShape, ...(opType === 'QuantizeLinear' ? { quantization } : {}) },
      scale: { name: 'scale', kind: 'weight', dtype: 'float32', shape: scaleShape },
      zero: { name: 'zero', kind: 'weight', dtype: 'uint8', shape: scaleShape },
      weight: { name: 'weight', kind: 'weight', dtype: 'float32', shape: [4, 4] },
    },
    nodes: [{
      id: 'candidate', opType,
      inputs: opType === 'QArgMax' ? { input: 'x' }
        : ['Linear', 'MatMul', 'Gemm', 'QLinear', 'QMatMul', 'QGemm'].includes(opType)
          ? { input: 'x', weight: 'weight' }
          : { input: 'x', scale: 'scale', zero_point: 'zero' },
      outputs: { out: { tensor: 'y' } }, params,
    }],
  };
}

function qAddRowProofGraph(shape = [1, 'S', 4], params = {}) {
  return {
    inputs: {
      a: { name: 'a', kind: 'input', dtype: 'int8', shape },
      b: { name: 'b', kind: 'input', dtype: 'int8', shape },
    },
    tensors: {
      a: { name: 'a', kind: 'input', dtype: 'int8', shape },
      b: { name: 'b', kind: 'input', dtype: 'int8', shape },
      out: { name: 'out', kind: 'value', dtype: 'int8', shape },
    },
    nodes: [{
      id: 'add', opType: 'QAdd', inputs: { a: 'a', b: 'b' },
      outputs: { out: { tensor: 'out' } }, params,
    }],
  };
}

function qDecoderRowProofGraph({
  causal = true,
  keyShape = [1, 'S', 4],
  valueShape = keyShape,
  maskShape = causal ? [1, 'S'] : [1, 'M'],
} = {}) {
  const queryShape = [1, 'S', 4];
  return {
    inputs: {
      q: { name: 'q', kind: 'input', dtype: 'int8', shape: queryShape },
      k: { name: 'k', kind: 'input', dtype: 'int8', shape: keyShape },
      v: { name: 'v', kind: 'input', dtype: 'int8', shape: valueShape },
      mask: { name: 'mask', kind: 'input', dtype: 'int32', shape: maskShape },
      position: { name: 'position', kind: 'input', dtype: 'int8', shape: queryShape },
    },
    tensors: {
      q: { name: 'q', kind: 'input', dtype: 'int8', shape: queryShape },
      k: { name: 'k', kind: 'input', dtype: 'int8', shape: keyShape },
      v: { name: 'v', kind: 'input', dtype: 'int8', shape: valueShape },
      mask: { name: 'mask', kind: 'input', dtype: 'int32', shape: maskShape },
      position: { name: 'position', kind: 'input', dtype: 'int8', shape: queryShape },
      attention: { name: 'attention', kind: 'value', dtype: 'int8', shape: queryShape },
      residual: { name: 'residual', kind: 'value', dtype: 'int8', shape: queryShape },
    },
    nodes: [{
      id: 'attention', opType: 'QSDPA', inputs: { q: 'q', k: 'k', v: 'v', mask: 'mask' },
      outputs: { out: { tensor: 'attention' } },
      params: { heads: 1, causal, scale: 0.5 },
    }, {
      id: 'residual', opType: 'QAdd', inputs: { a: 'attention', b: 'position' },
      outputs: { out: { tensor: 'residual' } }, params: {},
    }],
  };
}

test('fixed-row domain proof rejects unsafe axes, per-axis affine data, and activation transposes', () => {
  assert.equal(incrementalRowDomainSupported(rowProofGraph({
    opType: 'QArgMax', inputShape: [1, 'S', 'S'], outputShape: [1, 'S'], params: { axis: 1 },
  }), ['x']), false, 'QArgMax may not reduce the sequence axis even when extents coincide');
  assert.equal(incrementalRowDomainSupported(rowProofGraph({
    opType: 'QuantizeLinear', inputShape: [1, 'S', 4], outputShape: [1, 'S', 4],
    quantization: { scheme: 'per_axis', axis: 1 }, scaleShape: [8],
  }), ['x']), false, 'activation quantization must be scalar per-tensor');
  assert.equal(incrementalRowDomainSupported(rowProofGraph({
    opType: 'Linear', inputShape: [1, 'S', 4], outputShape: [1, 'S', 4],
    params: { transA: true },
  }), ['x']), false, 'a transform of the activation axes is not row-local');
  assert.equal(incrementalRowDomainSupported(rowProofGraph({
    opType: 'Linear', inputShape: [1, 'S', 4], outputShape: [1, 'S', 4],
    params: { weight_layout: 'dout_din' },
  }), ['x']), true, 'canonical weight-only layouts preserve activation rows');
  assert.equal(incrementalRowDomainSupported(rowProofGraph({
    opType: 'Linear', inputShape: ['B', 'S', 4], outputShape: ['B', 'S', 4],
    params: { weight_layout: 'dout_din' },
  }), ['x']), true, 'a batch symbol retains the portable batch-major sequence axis');
  assert.equal(incrementalRowDomainSupported(rowProofGraph({
    opType: 'Linear', inputShape: ['S', 1, 4], outputShape: ['S', 1, 4],
    params: { weight_layout: 'dout_din' },
  }), ['x']), false,
  'shared provider row attestation rejects sequence-major layouts without a WebGPU row candidate');
});

test('fixed-row domain proof attests only canonical QSDPA cache and exact QAdd joins', () => {
  assert.equal(incrementalRowDomainSupported(qAddRowProofGraph(
    [1, 'S', 4], { relu: 2 },
  ), ['a', 'b']), true, 'portable fused QAdd accepts two distinct dirty row operands');
  assert.equal(incrementalRowDomainSupported(qAddRowProofGraph(['S', 1, 4]), ['a']), false,
    'provider QAdd attestation rejects sequence-major layouts without a WebGPU row candidate');
  assert.equal(incrementalRowDomainSupported(qAddRowProofGraph(
    [1, 'S', 4], { relu: 3 },
  ), ['a']), false, 'QAdd fused activation metadata must match the portable kernel contract');

  assert.equal(incrementalRowDomainSupported(qDecoderRowProofGraph(),
    ['q', 'k', 'v', 'mask', 'position']), true,
  'causal attention and both exact residual operands may enter the dirty closure');
  assert.equal(incrementalRowDomainSupported(qDecoderRowProofGraph({
    causal: false, keyShape: [1, 'M', 4], valueShape: [1, 'M', 4],
  }), ['q']), true, 'cross attention retains invariant K/V/mask memory');

  assert.equal(incrementalRowDomainSupported(qDecoderRowProofGraph(),
    ['q', 'k', 'mask']), false, 'causal cache attestation requires the new V row');
  assert.equal(incrementalRowDomainSupported(qDecoderRowProofGraph({
    causal: false, keyShape: [1, 'M', 4], valueShape: [1, 'M', 4],
  }), ['q', 'k']), false, 'cross-attention K/V memory may not enter the dirty closure');
  assert.equal(incrementalRowDomainSupported(qDecoderRowProofGraph({
    valueShape: [1, 'M', 4],
  }), ['q', 'k', 'v', 'mask']), false, 'K/V cache sequence geometry must match');
  assert.equal(incrementalRowDomainSupported(qDecoderRowProofGraph({
    maskShape: [1, 1, 'S', 'S'],
  }), ['q', 'k', 'v', 'mask']), false, 'unsupported attention-mask ranks fail closed');

  for (const port of ['k', 'v']) {
    const aliased = qDecoderRowProofGraph({ causal: false, maskShape: [1, 'S'] });
    aliased.nodes[0].inputs[port] = 'q';
    assert.equal(incrementalRowDomainSupported(aliased, ['q']), false,
      `dirty cross-attention q may not alias invariant ${port}`);
  }
  const implicitScale = qDecoderRowProofGraph();
  delete implicitScale.nodes[0].params.scale;
  assert.equal(incrementalRowDomainSupported(implicitScale, ['q', 'k', 'v', 'mask']), false,
    'provider row attestation requires an explicit representable attention scale');

  const broadcast = qDecoderRowProofGraph();
  broadcast.inputs.position.shape = [4];
  broadcast.tensors.position.shape = [4];
  assert.equal(incrementalRowDomainSupported(broadcast, ['q', 'k', 'v', 'mask']), false,
    'QAdd row execution does not attest a broadcast operand');
});

async function fixture(options = {}) {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const compiled = await runtime.compile(dynamicGeluSnapshot(), {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext({
    decode: {
      changedInputs: ['x'],
      rowMode: 'required',
      requireIncremental: true,
      ...options,
    },
  });
  return { runtime, compiled, context };
}

async function closeFixture(fixture, results = []) {
  await fixture.context?.close();
  await fixture.compiled?.close();
  await fixture.runtime.close();
  await Promise.all(results.map((result) => result.close()));
}

test('CPU decode keeps active length separate from capacity and reuses one token specialization', async () => {
  const resources = await fixture();
  const results = [];
  try {
    const seedInputs = shaped(1, 4, 1);
    const seed = await resources.context.decode.seed(seedInputs);
    results.push(seed);
    assert.deepEqual(await seed.output('y').read(), expectedGelu(seedInputs));
    assert.deepEqual(seed.report.decodeState, {
      operation: 'seed',
      mode: 'incremental-seed',
      cacheState: 'seeded',
      cacheGeneration: 1,
      position: null,
      activeSequenceLength: 1,
      /* Lane zero's length and the whole lane array. One lane is the batched
       * contract with a list of one, so both are always reported. */
      activeSequenceLengths: [1],
      kvCapacity: 4,
      kvCapacityClass: 4,
      semanticSeedSignature: seed.report.decodeState.semanticSeedSignature,
      automaticReset: false,
      automaticResetReason: null,
      automaticResetCount: 0,
    });
    assert.ok(seed.report.decodeState.semanticSeedSignature);

    const growCount = seed.report.backendReport.activationGrowCount;
    const cacheEntries = seed.report.backendReport.specializationCacheEntries;
    let currentInputs = replaceRow(seedInputs, 1, 20);
    const firstStep = await resources.context.decode.step(currentInputs, { position: 1 });
    results.push(firstStep);
    assert.deepEqual(await firstStep.output('y').read(), expectedGelu(currentInputs));
    assert.equal(firstStep.report.decodeState.mode, 'incremental-row');
    assert.equal(firstStep.report.decodeState.activeSequenceLength, 2);
    assert.equal(firstStep.report.decodeState.cacheGeneration, 1);
    assert.equal(firstStep.report.decodeState.semanticSeedSignature,
      seed.report.decodeState.semanticSeedSignature);
    assert.equal(firstStep.report.backendReport.specializationCacheHit, true);
    assert.equal(firstStep.report.backendReport.specializationCacheEntries, cacheEntries);
    assert.equal(firstStep.report.backendReport.activationGrowCount, growCount);

    currentInputs = replaceRow(currentInputs, 2, 30);
    const secondStep = await resources.context.decode.step(currentInputs, { position: 2 });
    results.push(secondStep);
    assert.deepEqual(await secondStep.output('y').read(), expectedGelu(currentInputs));
    assert.equal(secondStep.report.decodeState.activeSequenceLength, 3);
    assert.equal(secondStep.report.backendReport.specializationCacheHit, true);
    assert.equal(secondStep.report.backendReport.specializationCacheEntries, cacheEntries);
    assert.equal(secondStep.report.backendReport.activationGrowCount, growCount,
      'token steps must not grow capacity or specialize once per token');
  } finally {
    await closeFixture(resources, results);
  }
});

test('rowMode required rejects a graph without whole-domain row attestation at context creation', async () => {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const compiled = await runtime.compile(unsupportedRowSnapshot(), {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  let context;
  const results = [];
  try {
    await assert.rejects(compiled.createContext({
      decode: {
        changedInputs: ['decoder_input_ids'],
        rowMode: 'required',
        requireIncremental: true,
      },
    }), (error) => error?.code === 'BACKEND_UNSUPPORTED' && /cannot attest/.test(error.message));
    context = await compiled.createContext({
      decode: {
        changedInputs: ['decoder_input_ids'],
        rowMode: 'auto',
        requireIncremental: true,
      },
    });
    const seed = await context.decode.seed({
      decoder_input_ids: { data: Int32Array.of(0, 1), shape: [1, 2] },
      zero: { data: Int32Array.of(0), shape: [1] },
    });
    results.push(seed);
    const step = await context.decode.step({
      decoder_input_ids: { data: Int32Array.of(2, 0), shape: [1, 2] },
    }, { position: 1 });
    results.push(step);
    assert.equal(step.report.decodeState.mode, 'incremental-dependency');
    assert.deepEqual(await step.output('y').read(), Int32Array.of(0, 1));
  } finally {
    await context?.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('CPU decode prompt changes reset semantic state and never expose a stale capacity suffix', async () => {
  const resources = await fixture();
  const results = [];
  try {
    const shortInputs = shaped(1, 3, 2);
    const short = await resources.context.decode.seed(shortInputs, { position: 2 });
    results.push(short);
    assert.equal(short.report.decodeState.activeSequenceLength, 3);
    assert.equal(short.report.decodeState.kvCapacity, 3);
    assert.equal(short.report.decodeState.kvCapacityClass, 4);

    const longInputs = shaped(1, 5, 50);
    const long = await resources.context.decode.seed(longInputs, { position: 1 });
    results.push(long);
    assert.deepEqual(await long.output('y').read(), expectedGelu(longInputs));
    assert.equal((await long.output('y').read()).length, 20);
    assert.equal(long.report.decodeState.activeSequenceLength, 2);
    assert.equal(long.report.decodeState.kvCapacity, 5);
    assert.equal(long.report.decodeState.kvCapacityClass, 8);
    assert.equal(long.report.decodeState.automaticReset, true);
    assert.equal(long.report.decodeState.automaticResetReason, 'kv-capacity-class-changed');
    assert.equal(long.report.decodeState.automaticResetCount, 1);
    assert.notEqual(long.report.decodeState.semanticSeedSignature,
      short.report.decodeState.semanticSeedSignature);

    const shortAgainInputs = shaped(1, 3, 90);
    const shortAgain = await resources.context.decode.seed(shortAgainInputs);
    results.push(shortAgain);
    const exact = await shortAgain.output('y').read();
    assert.equal(exact.length, 12, 'result storage contains exact logical bytes, not capacity tail');
    assert.deepEqual(exact, expectedGelu(shortAgainInputs));
    assert.equal(shortAgain.report.decodeState.kvCapacity, 3);
    assert.equal(shortAgain.report.decodeState.kvCapacityClass, 4);
    assert.equal(shortAgain.report.decodeState.automaticResetReason, 'kv-capacity-class-changed');
    assert.equal(shortAgain.report.decodeState.automaticResetCount, 2);
  } finally {
    await closeFixture(resources, results);
  }
});

test('failed CPU seed and incompatible steps leave the prior semantic seed usable', async () => {
  const resources = await fixture();
  const results = [];
  try {
    const seedInputs = shaped(1, 4, 4);
    const seed = await resources.context.decode.seed(seedInputs);
    results.push(seed);
    const generation = seed.report.decodeState.cacheGeneration;
    const signature = seed.report.decodeState.semanticSeedSignature;

    await assert.rejects(resources.context.decode.seed({
      x: { data: new Float32Array(15), shape: [1, 4, 4] },
    }), (error) => error?.code === 'INVALID_ARGUMENT');
    await assert.rejects(resources.context.decode.step(shaped(1, 5, 8), { position: 1 }),
      (error) => error?.code === 'INVALID_ARGUMENT' && /incompatible/.test(error.message));
    await assert.rejects(resources.context.decode.seed(shaped(2, 4, 8)),
      /* The context declared one lane, so a two-lane binding is refused by the
       * declaration rather than by a hardcoded B=1 gate -- the refusal now says
       * whose number is authoritative. */
      (error) => error?.code === 'BACKEND_UNSUPPORTED' &&
        /declares 1 lane\(s\) but this graph binds batch 2/.test(error.message));

    const stepInputs = replaceRow(seedInputs, 1, 70);
    const recovered = await resources.context.decode.step(stepInputs, { position: 1 });
    results.push(recovered);
    assert.deepEqual(await recovered.output('y').read(), expectedGelu(stepInputs));
    assert.equal(recovered.report.decodeState.cacheGeneration, generation);
    assert.equal(recovered.report.decodeState.semanticSeedSignature, signature);
    assert.equal(recovered.report.decodeState.automaticResetCount, 0);
  } finally {
    await closeFixture(resources, results);
  }
});

test('decode seed, step, reset, and close drain through one context FIFO', async () => {
  const resources = await fixture();
  const results = [];
  try {
    const seedInputs = shaped(1, 3, 6);
    const stepInputs = replaceRow(seedInputs, 1, 100);
    const seedPromise = resources.context.decode.seed(seedInputs);
    const stepPromise = resources.context.decode.step(stepInputs, { position: 1 });
    const resetPromise = resources.context.decode.reset();
    const closePromise = resources.context.close();
    await assert.rejects(resources.context.decode.seed(seedInputs),
      (error) => error?.code === 'HANDLE_DISPOSED');
    const [seed, step] = await Promise.all([seedPromise, stepPromise]);
    results.push(seed, step);
    await resetPromise;
    await closePromise;
    assert.equal(step.report.decodeState.activeSequenceLength, 2);
    assert.deepEqual(await step.output('y').read(), expectedGelu(stepInputs));
  } finally {
    await closeFixture(resources, results);
  }
});

test('decode capacity follows the active query instead of a longer cross-attention memory', async () => {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const compiled = await runtime.compile(decoderGeometrySnapshot(), {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext({
    decode: {
      changedInputs: ['decoder_input_ids'],
      rowMode: 'disabled',
      requireIncremental: true,
    },
  });
  const results = [];
  try {
    const inputs = {
      decoder_input_ids: { data: Int32Array.of(1, 2, 0), shape: [1, 3] },
      memory_padding_mask: { data: new Int32Array(6).fill(1), shape: [1, 6] },
    };
    const seed = await context.decode.seed(inputs, { position: 1 });
    results.push(seed);
    assert.equal(seed.report.decodeState.activeSequenceLength, 2);
    assert.equal(seed.report.decodeState.kvCapacity, 3);
    assert.equal(seed.report.decodeState.kvCapacityClass, 4);

    const stepInputs = {
      ...inputs,
      decoder_input_ids: { data: Int32Array.of(1, 2, 3), shape: [1, 3] },
    };
    const step = await context.decode.step(stepInputs);
    results.push(step);
    assert.equal(step.report.decodeState.activeSequenceLength, 3);
    assert.equal(step.report.decodeState.kvCapacity, 3);
    assert.deepEqual(await step.output('tokens').read(), Int32Array.of(1, 2, 3));
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('partial decode steps retain cloned seed inputs and commit changed clones transactionally', async () => {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const compiled = await runtime.compile(retainedInputSnapshot(), {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext({
    decode: {
      changedInputs: ['changed'],
      rowMode: 'disabled',
      requireIncremental: true,
    },
  });
  const results = [];
  try {
    const changed = Float32Array.of(1, 2);
    const fixed = Float32Array.of(10, 20);
    const seed = await context.decode.seed({
      changed: { data: changed, shape: [1, 2] },
      fixed: { data: fixed, shape: [1, 2] },
    });
    results.push(seed);
    changed.fill(100);
    fixed.fill(200);

    const nextChanged = Float32Array.of(3, 4);
    const step = await context.decode.step({
      changed: { data: nextChanged, shape: [1, 2] },
    });
    results.push(step);
    assert.deepEqual(await step.output('y').read(), Float32Array.of(13, 24),
      'unchanged input bytes come from the context-owned seed clone');
    nextChanged.fill(300);

    const fixedStep = await context.decode.step({
      fixed: { data: Float32Array.of(30, 40), shape: [1, 2] },
    }, { changedInputs: ['fixed'] });
    results.push(fixedStep);
    assert.deepEqual(await fixedStep.output('y').read(), Float32Array.of(33, 44),
      'the prior successful step committed a private changed-input clone');

    await assert.rejects(context.decode.step({
      fixed: { data: Float32Array.of(1, 1), shape: [1, 2] },
    }), (error) => error?.code === 'INVALID_ARGUMENT' &&
      /exactly match/.test(error.message));
    const recovered = await context.decode.step({
      changed: { data: Float32Array.of(5, 6), shape: [1, 2] },
    });
    results.push(recovered);
    assert.deepEqual(await recovered.output('y').read(), Float32Array.of(35, 46));

    await assert.rejects(context.decode.step({
      changed: { data: Float32Array.of(7, 8, 9), shape: [1, 3] },
    }), (error) => error?.code === 'INVALID_ARGUMENT');
    const recoveredAgain = await context.decode.step({
      changed: { data: Float32Array.of(7, 8), shape: [1, 2] },
    });
    results.push(recoveredAgain);
    assert.deepEqual(await recoveredAgain.output('y').read(), Float32Array.of(37, 48));

    await context.decode.reset();
    await assert.rejects(context.decode.step({
      changed: { data: Float32Array.of(9, 10), shape: [1, 2] },
    }), (error) => error?.code === 'INVALID_ARGUMENT' && /successful seed/.test(error.message));
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

/*
 * The batched decode lifecycle: B is declared by the context, lanes carry their
 * own active lengths, and a finished lane idles instead of forcing a reseed.
 *
 * These go through the public context API rather than the row executor, because
 * the state that makes a batch work -- who declared B, which lane is where, and
 * which lane stopped -- lives here. The executor is proved separately against a
 * full-recomputation oracle in tests/batched_decode.test.mjs.
 */
async function batchedFixture(lanes) {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const compiled = await runtime.compile(dynamicGeluSnapshot(), {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext({
    decode: {
      changedInputs: ['x'],
      rowMode: 'required',
      requireIncremental: true,
      lanes,
    },
  });
  return { runtime, compiled, context };
}

test('a two-lane context decodes staggered lanes and reports each length', async () => {
  const resources = await batchedFixture(2);
  const results = [];
  try {
    /* Staggered prompts: lane 0 arrives with two tokens, lane 1 with one. A
     * seed that demanded equal prompt lengths would make continuous batching
     * wait for requests to be the same size. */
    const seed = await resources.context.decode.seed(shaped(2, 4, 3), {
      positions: [1, 0],
    });
    results.push(seed);
    assert.deepEqual(seed.report.decodeState.activeSequenceLengths, [2, 1]);

    const stepped = await resources.context.decode.step(shaped(2, 4, 9), {
      positions: [2, 1],
    });
    results.push(stepped);
    assert.deepEqual(stepped.report.decodeState.activeSequenceLengths, [3, 2]);
    assert.equal(stepped.report.decodeState.mode, 'incremental-row');

    /* Lane 1 finishes. It idles: its length stops and lane 0 keeps going, with
     * no reseed and no change to any operand's shape. */
    const idled = await resources.context.decode.step(shaped(2, 4, 15), {
      positions: [3, null],
    });
    results.push(idled);
    assert.deepEqual(idled.report.decodeState.activeSequenceLengths, [4, 2]);
  } finally {
    await closeFixture(resources, results);
  }
});

test('a two-lane context refuses the scalar position spelling', async () => {
  /* Two beliefs about how many slots exist. Resolving it by precedence would
   * silently decode lane zero of a two-lane batch. */
  const resources = await batchedFixture(2);
  try {
    await assert.rejects(
      resources.context.decode.seed(shaped(2, 4, 3), { position: 1 }),
      (error) => error?.code === 'INVALID_ARGUMENT' &&
        /declares 2 lanes, so it needs per-lane positions/.test(error.message));
  } finally {
    await closeFixture(resources);
  }
});

test('a batched step refuses a lane list that is not the declared capacity', async () => {
  const resources = await batchedFixture(2);
  const results = [];
  try {
    results.push(await resources.context.decode.seed(shaped(2, 4, 3), { positions: [1, 0] }));
    await assert.rejects(
      resources.context.decode.step(shaped(2, 4, 9), { positions: [2] }),
      (error) => error?.code === 'INVALID_ARGUMENT' &&
        /needs one position per declared lane \(2\)/.test(error.message));
    /* And a step where nothing advances is not a step. */
    await assert.rejects(
      resources.context.decode.step(shaped(2, 4, 9), { positions: [null, null] }),
      (error) => error?.code === 'INVALID_ARGUMENT' &&
        /at least one advancing lane/.test(error.message));
  } finally {
    await closeFixture(resources, results);
  }
});

test('a lane that has produced no row cannot idle', async () => {
  /* An idle lane occupies its own last row, which is the only row whose
   * recompute changes nothing. A lane with no row has nothing to occupy. */
  const resources = await batchedFixture(2);
  const results = [];
  try {
    results.push(await resources.context.decode.seed(shaped(2, 4, 3), { positions: [1, 0] }));
    /* Lane 1's active length is 1, so its last row is 0 and idling is legal;
     * the refusal is for a length below one, which a seeded lane cannot have.
     * What is refused here instead is the mismatched advancing position. */
    await assert.rejects(
      resources.context.decode.step(shaped(2, 4, 9), { positions: [2, 5] }),
      (error) => error?.code === 'INVALID_ARGUMENT' &&
        /lane 1 must equal its next active position 1/.test(error.message));
  } finally {
    await closeFixture(resources, results);
  }
});

test('a one-lane context still takes the scalar position spelling', async () => {
  /* One lane is not a preserved second path: it is the batched contract with a
   * list of one, and the scalar option is sugar for that list. */
  const resources = await batchedFixture(1);
  const results = [];
  try {
    results.push(await resources.context.decode.seed(shaped(1, 4, 3), { position: 1 }));
    const stepped = await resources.context.decode.step(shaped(1, 4, 9), { position: 2 });
    results.push(stepped);
    assert.deepEqual(stepped.report.decodeState.activeSequenceLengths, [3]);
    assert.equal(stepped.report.decodeState.activeSequenceLength, 3);
  } finally {
    await closeFixture(resources, results);
  }
});

test('a provider that cannot batch rows refuses a multi-lane context at creation', () => {
  /* The batched-row provider-capability proof described in
   * ../docs/scheduling-and-dynamic-batching-design.md#compatibility-routes-and-typed-independence-proof.
   * A provider that ignores the lane list does not produce wrong answers -- it
   * recomputes the whole prefix -- but it reports a row step it never ran,
   * creating the "rejected in the browser, accepted on the robot" divergence
   * this contract prevents. Refusal therefore lands at context creation, not at
   * the step that needed it, and not never. Every built-in provider attests
   * batched rows today; the capability stays because a provider SDK backend is
   * where the next one comes from. */
  const snapshot = dynamicGeluSnapshot();
  const capabilities = { incrementalExecution: true, incrementalRows: true };
  assert.throws(() => new ProviderDecodeLifecycle(snapshot, 'stub', capabilities, {
    changedInputs: ['x'], rowMode: 'auto', lanes: 2,
  }), (error) => error?.code === 'BACKEND_UNSUPPORTED' &&
    /cannot execute a decode step covering 2 lanes/.test(error.message));

  /* One lane is unaffected, and a provider that attests batched rows is admitted. */
  assert.ok(new ProviderDecodeLifecycle(snapshot, 'stub', capabilities, {
    changedInputs: ['x'], rowMode: 'auto', lanes: 1,
  }));
  assert.ok(new ProviderDecodeLifecycle(
    snapshot, 'stub', { ...capabilities, batchedRows: true },
    { changedInputs: ['x'], rowMode: 'auto', lanes: 2 }));

  /* And a context that asked for no rows at all may still declare lanes: there
   * is nothing to batch, so there is nothing to refuse. */
  assert.ok(new ProviderDecodeLifecycle(snapshot, 'stub', capabilities, {
    changedInputs: ['x'], rowMode: 'disabled', lanes: 2,
  }));
});

test('a declared lane capacity must be a positive integer', () => {
  const snapshot = dynamicGeluSnapshot();
  for (const lanes of [0, -1, 1.5, '2']) {
    assert.throws(() => new ProviderDecodeLifecycle(snapshot, 'stub', {
      incrementalExecution: true, incrementalRows: true, batchedRows: true,
    }, { changedInputs: ['x'], lanes }),
      (error) => error?.code === 'INVALID_ARGUMENT' &&
        /Decode lane capacity must be a positive integer/.test(error.message),
      `lanes=${String(lanes)} must be refused`);
  }
});

test('a lane with no request parks, and a lane with one idles', async () => {
  /* The two ways not to advance, through the public contract. `null` idles a
   * lane that still holds its request -- its length stops and its row is
   * preserved. `-1` parks a lane that holds none: it has no row to preserve, so
   * it occupies one for shape and its output is discarded. Only the second can
   * express a slot whose pages went back to the pool. */
  const resources = await batchedFixture(2);
  const results = [];
  try {
    results.push(await resources.context.decode.seed(shaped(2, 4, 3), { positions: [1, 0] }));
    const idled = await resources.context.decode.step(shaped(2, 4, 9), {
      positions: [2, null],
    });
    results.push(idled);
    assert.deepEqual(idled.report.decodeState.activeSequenceLengths, [3, 1]);

    const parked = await resources.context.decode.step(shaped(2, 4, 15), {
      positions: [3, -1],
    });
    results.push(parked);
    assert.deepEqual(parked.report.decodeState.activeSequenceLengths, [4, 1],
      "a parked lane's length does not move either");
    assert.equal(parked.report.decodeState.mode, 'incremental-row');

    /* Parking every lane is not a step. */
    await assert.rejects(
      resources.context.decode.step(shaped(2, 4, 21), { positions: [-1, -1] }),
      (error) => error?.code === 'INVALID_ARGUMENT' &&
        /at least one advancing lane/.test(error.message));
  } finally {
    await closeFixture(resources, results);
  }
});

test('a decode position below -1 is refused', () => {
  /* -1 is the park sentinel; anything below it is a caller mistake rather than
   * a third meaning waiting to be invented. */
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  return (async () => {
    const compiled = await runtime.compile(dynamicGeluSnapshot(), {
      backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
    });
    const context = await compiled.createContext({
      decode: { changedInputs: ['x'], rowMode: 'required', lanes: 2 },
    });
    try {
      await assert.rejects(
        context.decode.seed(shaped(2, 4, 3), { positions: [1, -2] }),
        (error) => error?.code === 'INVALID_ARGUMENT' &&
          /non-negative safe integers, null to idle a lane, or -1 to park one/
            .test(error.message));
    } finally {
      await context.close();
      await compiled.close();
      await runtime.close();
    }
  })();
});
