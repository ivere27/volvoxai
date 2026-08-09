import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BoundExecutionGraphError,
  createBoundExecutionGraph,
} from '../ts/core/BoundExecutionGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model } from '../ts/core/Model.js';
import {
  canonicalBankResidencySuffix,
  ResolvedShapePlanError,
  resolveGraphShapes,
} from '../ts/core/ResolvedShapePlan.js';
import { VolvoxAI } from '../ts/VolvoxAI.js';

const EXPERTS = 4;
const D_IN = 2;
const D_OUT = 1;

/** Expert k maps both input features to the constant k + 1. */
function expertPayload() {
  const data = new Float32Array(EXPERTS * D_IN * D_OUT);
  for (let expert = 0; expert < EXPERTS; expert++) {
    for (let feature = 0; feature < D_IN; feature++) {
      data[expert * D_IN * D_OUT + feature * D_OUT] = expert + 1;
    }
  }
  return data;
}

function bankDocument() {
  return {
    format: 'volvox-graph/v1',
    dimensions: { F: { min: 1, max: 8 } },
    banks: { experts: 'F' },
    inputs: {
      x: { dtype: 'float32', shape: [1, D_IN] },
      route_indices: { dtype: 'float32', shape: [1, 1] },
      route_weights: { dtype: 'float32', shape: [1, 1] },
    },
    nodes: [{
      id: 'mix',
      opType: 'MoELinear',
      inputs: {
        input: 'x',
        expert_weight: 'experts',
        route_indices: 'route_indices',
        route_weights: 'route_weights',
      },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, D_OUT] } },
      params: {},
    }],
    outputs: ['y'],
  };
}

function bankModel() {
  const data = expertPayload();
  const descriptor = {
    name: 'experts',
    dtype: 'float32',
    shape: [EXPERTS, D_IN, D_OUT],
  };
  const graph = parseGraphDocument(bankDocument(), [descriptor]);
  return {
    graph,
    model: Model.capture({
      graph,
      weights: { experts: { ...descriptor, data } },
    }),
  };
}

function biasedBankModel() {
  const document = bankDocument();
  document.banks.expert_bias = 'F';
  document.nodes[0].inputs.expert_bias = 'expert_bias';
  const descriptors = [
    { name: 'experts', dtype: 'float32', shape: [EXPERTS, D_IN, D_OUT] },
    { name: 'expert_bias', dtype: 'float32', shape: [EXPERTS, D_OUT] },
  ];
  const graph = parseGraphDocument(document, descriptors);
  return {
    graph,
    model: Model.capture({
      graph,
      weights: {
        experts: { ...descriptors[0], data: expertPayload() },
        expert_bias: {
          ...descriptors[1],
          data: Float32Array.from([10, 20, 30, 40]),
        },
      },
    }),
  };
}

function dualGatherBankModel() {
  const descriptors = [
    { name: 'data_bank', dtype: 'float32', shape: [EXPERTS, 1] },
    { name: 'index_bank', dtype: 'int32', shape: [EXPERTS, 1] },
  ];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      F: { min: 1, max: EXPERTS },
      I: { min: 1, max: EXPERTS },
    },
    banks: { data_bank: 'F', index_bank: 'I' },
    inputs: {},
    nodes: [{
      id: 'gather_banks',
      opType: 'Gather',
      inputs: { input: 'data_bank', indices: 'index_bank' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['I', 1, 1] } },
      params: { axis: 0 },
    }],
    outputs: ['y'],
  }, descriptors);
  return {
    graph,
    model: Model.capture({
      graph,
      weights: {
        data_bank: {
          ...descriptors[0],
          data: Float32Array.from([10, 20, 30, 40]),
        },
        index_bank: {
          ...descriptors[1],
          data: Int32Array.from([0, 3, 2, 1]),
        },
      },
    }),
  };
}

function inputs(expert) {
  return {
    x: { data: Float32Array.of(1, 0), shape: [1, D_IN] },
    route_indices: { data: Float32Array.of(expert), shape: [1, 1] },
    route_weights: { data: Float32Array.of(1), shape: [1, 1] },
  };
}

async function runBound(model, plan, values) {
  const bound = createBoundExecutionGraph(model, plan);
  const engine = new CPUEngine();
  engine.allocateGraph(bound.graph);
  const request = Object.fromEntries(
    Object.entries(values).map(([name, view]) => [name, view.data]),
  );
  return { bound, result: await engine.execute(request) };
}

test('a fully resident bank routes by global expert id', async () => {
  const { graph, model } = bankModel();
  const plan = resolveGraphShapes(graph, inputs(2), model.quantizationByTensor);

  assert.deepEqual(plan.bankResidency, {});
  assert.deepEqual(plan.tensors.experts.shape, [EXPERTS, D_IN, D_OUT]);

  const { result } = await runBound(model, plan, inputs(2));
  assert.equal(result.y[0], 3, 'expert 2 contributes 2 + 1');
});

test('a partially resident bank stages only the resident slots', async () => {
  const { graph, model } = bankModel();
  const residency = { experts: [2] };
  const plan = resolveGraphShapes(
    graph, inputs(2), model.quantizationByTensor, residency,
  );

  assert.deepEqual(plan.bankResidency, residency);
  // One slot instead of four: this is the memory the residency buys.
  assert.deepEqual(plan.tensors.experts.shape, [1, D_IN, D_OUT]);
  assert.equal(plan.tensors.experts.sizeBytes, D_IN * D_OUT * 4);

  const { bound, result } = await runBound(model, plan, inputs(2));
  assert.deepEqual(bound.graph.getTensor('experts').shape, [1, D_IN, D_OUT]);
  assert.deepEqual(
    Array.from(bound.graph.getTensor('experts').buffer), [3, 3],
    'the staged slice holds expert 2, not expert 0',
  );
  assert.deepEqual(bound.graph.nodes[0].residentSlots, [2]);
  assert.equal(bound.graph.nodes[0].residentSlotDomain, EXPERTS,
    'the bound node retains the complete global bank extent');
  // Routes stay global even though only one row is staged.
  assert.equal(result.y[0], 3);
});

test('routed expert tensors require one exact effective residency', async () => {
  const { graph, model } = biasedBankModel();
  const assertMisaligned = (bankResidency) => {
    const plan = resolveGraphShapes(
      graph, inputs(2), model.quantizationByTensor, bankResidency,
    );
    assert.throws(
      () => createBoundExecutionGraph(model, plan),
      (error) => {
        assert.ok(error instanceof BoundExecutionGraphError);
        assert.equal(error.code, 'CONSTRUCTION_FAILED');
        assert.match(
          String(error.cause?.message),
          /must use the same effective slot residency/,
        );
        return true;
      },
    );
  };

  assertMisaligned({ experts: [2], expert_bias: [1] });

  const aligned = { experts: [2], expert_bias: [2] };
  const plan = resolveGraphShapes(
    graph, inputs(2), model.quantizationByTensor, aligned,
  );
  const { bound, result } = await runBound(model, plan, inputs(2));
  assert.deepEqual(bound.graph.nodes[0].residentSlots, [2]);
  assert.equal(bound.graph.nodes[0].residentSlotDomain, EXPERTS);
  assert.equal(result.y[0], 33,
    'expert 2 weight and bias are staged from the same global row');
});

test('Gather takes slot mapping only from its data bank', async () => {
  const { graph, model } = dualGatherBankModel();
  const plan = resolveGraphShapes(
    graph, {}, model.quantizationByTensor,
    { data_bank: [3], index_bank: [1] },
  );
  const { bound, result } = await runBound(model, plan, {});
  assert.deepEqual(bound.graph.nodes[0].residentSlots, [3],
    'the lexically earlier indices port cannot replace the data-bank mapping');
  assert.equal(bound.graph.nodes[0].residentSlotDomain, EXPERTS);
  assert.equal(result.y[0], 40);
});

test('routing to a non-resident expert fails instead of reading another slot', async () => {
  const { graph, model } = bankModel();
  const plan = resolveGraphShapes(
    graph, inputs(0), model.quantizationByTensor, { experts: [2] },
  );
  await assert.rejects(
    () => runBound(model, plan, inputs(0)),
    /expert 0, which is not resident in this context/,
  );
});

test('two residencies of the same model produce distinct plan signatures', () => {
  const { graph, model } = bankModel();
  const first = resolveGraphShapes(
    graph, inputs(1), model.quantizationByTensor, { experts: [1] },
  );
  const second = resolveGraphShapes(
    graph, inputs(1), model.quantizationByTensor, { experts: [1, 2] },
  );
  const full = resolveGraphShapes(graph, inputs(1), model.quantizationByTensor);

  assert.equal(first.graphFingerprint, second.graphFingerprint);
  for (const [left, right] of [[first, second], [first, full], [second, full]]) {
    assert.notEqual(
      left.signature, right.signature,
      'plan-cache entries must not be shared across residencies',
    );
  }
});

test('bank residency identity is unambiguous for arbitrary tensor names', () => {
  assert.notEqual(
    canonicalBankResidencySuffix({ a: [1], b: [2] }),
    canonicalBankResidencySuffix({ 'a=1,b': [2] }),
    'cache identity must not depend on delimiter-safe tensor names',
  );
  assert.notEqual(
    canonicalBankResidencySuffix({ 'a,b': [1], c: [2, 3] }),
    canonicalBankResidencySuffix({ a: [1], 'b,c': [2, 3] }),
  );
});

test('bound materialization rejects forged bank slots before staging payloads', () => {
  const { model } = bankModel();
  const valid = model.bindShapes(inputs(2), { experts: [2] });
  const residency = Object.freeze({ experts: Object.freeze([999]) });
  const shapeSignature = valid.signature.slice(0, valid.signature.indexOf('|banks:'));
  const forged = Object.freeze({
    ...valid,
    signature: `${shapeSignature}${canonicalBankResidencySuffix(residency)}`,
    bankResidency: residency,
  });

  assert.throws(
    () => createBoundExecutionGraph(model, forged),
    (error) => {
      assert.ok(error instanceof BoundExecutionGraphError);
      assert.equal(error.code, 'PLAN_DESCRIPTOR_MISMATCH');
      assert.match(error.message, /slot id in \[0, 3\]/);
      return true;
    },
  );
  const malformed = Object.freeze({ ...valid, bankResidency: null });
  assert.throws(
    () => createBoundExecutionGraph(model, malformed),
    (error) => error instanceof BoundExecutionGraphError &&
      error.code === 'PLAN_DESCRIPTOR_MISMATCH',
  );
});

test('bank residency is validated against the declared bank', () => {
  const { graph, model } = bankModel();
  const reject = (residency, pattern) => {
    assert.throws(
      () => resolveGraphShapes(
        graph, inputs(0), model.quantizationByTensor, residency,
      ),
      (error) => {
        assert.ok(error instanceof ResolvedShapePlanError);
        assert.equal(error.code, 'INVALID_BANK_RESIDENCY');
        assert.match(error.message, pattern);
        return true;
      },
    );
  };

  reject({ missing: [0] }, /names no declared bank/);
  reject({ experts: [] }, /non-empty array of slot ids/);
  reject({ experts: [EXPERTS] }, /slot id in \[0, 3\]/);
  reject({ experts: [-1] }, /slot id in \[0, 3\]/);
  reject({ experts: [1, 1] }, /strictly ascending and unique/);
  reject({ experts: [2, 1] }, /strictly ascending and unique/);
});

test('Model.bindShapes accepts a residency for a static model', () => {
  const { model } = bankModel();
  const plan = model.bindShapes(inputs(3), { experts: [3] });
  assert.deepEqual(plan.tensors.experts.shape, [1, D_IN, D_OUT]);
  assert.deepEqual(model.bindShapes(inputs(3)).tensors.experts.shape,
    [EXPERTS, D_IN, D_OUT]);
});

test('two contexts of one CompiledModel hold different resident slots', async () => {
  const { model } = bankModel();
  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const compiled = await runtime.compile(model, {
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'allow' },
  });
  const first = await compiled.createContext({ bankResidency: { experts: [1] } });
  const second = await compiled.createContext({ bankResidency: { experts: [3] } });
  try {
    const left = await first.execute(inputs(1));
    const right = await second.execute(inputs(3));
    assert.deepEqual(await left.output('y').read(), Float32Array.of(2));
    assert.deepEqual(await right.output('y').read(), Float32Array.of(4));

    // Each context only holds its own family. The kernel diagnostic is the
    // cause of the wrapped backend failure.
    await assert.rejects(second.execute(inputs(1)), (error) => {
      let cause = error;
      const messages = [];
      while (cause) {
        messages.push(cause.message);
        cause = cause.cause;
      }
      assert.ok(
        messages.some((message) => /expert 1, which is not resident/.test(message)),
        messages.join(' <- '),
      );
      return true;
    });
    await Promise.all([left.close(), right.close()]);
  } finally {
    await Promise.all([first.close(), second.close()]);
    await compiled.close();
    await runtime.close();
  }
});

test('an unknown bank name is rejected during context creation', async () => {
  const { model } = bankModel();
  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const compiled = await runtime.compile(model, {
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'allow' },
  });
  try {
    await assert.rejects(
      compiled.createContext({ bankResidency: { missing: [0] } }),
      (error) => {
        assert.match(error.message, /names no declared bank/);
        return true;
      },
    );
  } finally {
    await compiled.close();
    await runtime.close();
  }
});

function quantizedBankDocument(scheme) {
  const reference = scheme === 'per_axis'
    ? { scheme: 'per_axis', axis: 0,
        scale_tensor: 'experts.scale', zero_point_tensor: 'experts.zero' }
    : { scheme: 'per_tensor',
        scale_tensor: 'experts.scale', zero_point_tensor: 'experts.zero' };
  return {
    format: 'volvox-graph/v1',
    dimensions: { F: { min: 1, max: EXPERTS } },
    banks: { experts: 'F' },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      // Gather preserves the affine domain, so the output declares it too.
      tensors: {
        experts: reference,
        y: scheme === 'per_axis'
          ? { scheme: 'per_axis', axis: 0,
              scale_tensor: 'y.scale', zero_point_tensor: 'y.zero' }
          : { scheme: 'per_tensor',
              scale_tensor: 'y.scale', zero_point_tensor: 'y.zero' },
      },
    },
    inputs: { slot: { dtype: 'int32', shape: [1] } },
    nodes: [{
      id: 'pick',
      opType: 'Gather',
      inputs: { input: 'experts', indices: 'slot' },
      outputs: { out: { tensor: 'y', dtype: 'int8', shape: [1, D_IN, D_OUT] } },
      params: { axis: 0 },
    }],
    outputs: ['y'],
  };
}

function quantizedBankDescriptors(scheme) {
  const extent = scheme === 'per_axis' ? EXPERTS : 1;
  return [
    { name: 'experts', dtype: 'int8', shape: [EXPERTS, D_IN, D_OUT] },
    { name: 'experts.scale', dtype: 'float32', shape: [extent] },
    { name: 'experts.zero', dtype: 'int8', shape: [extent] },
    { name: 'y.scale', dtype: 'float32', shape: [1] },
    { name: 'y.zero', dtype: 'int8', shape: [1] },
  ];
}

test('per-axis quantization on a gathered slot axis is refused upstream', () => {
  // Gather rejects reordering per-axis affine metadata, and MoELinear is
  // float32-only, so a per-axis-quantized bank on the slot axis has no legal
  // consumer. The residency slicing path is therefore never reached with one.
  const graph = parseGraphDocument(quantizedBankDocument('per_axis'),
                                   quantizedBankDescriptors('per_axis'));
  const hydrated = Object.freeze({
    experts: Object.freeze({
      scheme: 'per_axis', axis: 0,
      scales: Object.freeze([0.5, 0.25, 0.125, 0.0625]),
      zero_points: Object.freeze([0, 1, 2, 3]),
    }),
    y: Object.freeze({
      scheme: 'per_axis', axis: 0,
      scales: Object.freeze([0.5]),
      zero_points: Object.freeze([0]),
    }),
  });
  assert.throws(
    () => resolveGraphShapes(graph,
      { slot: { data: Int32Array.of(1), shape: [1] } }, hydrated),
    /reorder the per-axis affine metadata/,
  );
});

test('a per-tensor quantized bank keeps its scale across slicing', () => {
  const graph = parseGraphDocument(quantizedBankDocument('per_tensor'),
                                   quantizedBankDescriptors('per_tensor'));
  const hydrated = Object.freeze({
    experts: Object.freeze({ scheme: 'per_tensor', scale: 0.25, zero_point: -3 }),
    y: Object.freeze({ scheme: 'per_tensor', scale: 0.25, zero_point: -3 }),
  });
  const values = { slot: { data: Int32Array.of(1), shape: [1] } };

  const full = resolveGraphShapes(graph, values, hydrated);
  const sliced = resolveGraphShapes(graph, values, hydrated, { experts: [1, 3] });

  assert.deepEqual(sliced.tensors.experts.shape, [2, D_IN, D_OUT]);
  // One scale covers the whole tensor, so slicing must leave it untouched.
  assert.deepEqual(sliced.tensors.experts.quantization, full.tensors.experts.quantization);
  assert.equal(sliced.tensors.experts.sizeBytes, 2 * D_IN * D_OUT);
});
