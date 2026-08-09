import test from 'node:test';
import assert from 'node:assert/strict';

import {
  OperatorShapeContractError,
  WAVE_A_OPERATOR_SHAPE_FUNCTIONS,
  WAVE_A_SHAPE_FUNCTION_IDS,
  getOperatorShapeContract,
  inferConcreteOperatorShapes,
  proveOperatorShapeDomain,
} from '../ts/ops/operatorShapeContracts.js';
import {
  operatorShapeContracts,
  operatorShapeFunctionIds,
} from '../ts/generated/kernelRegistry.js';
import { ShapeEnvironment } from '../ts/ops/shapeSystem.js';

function tensor(shape, dtype = 'float32', quantization = undefined) {
  return quantization === undefined ? { shape, dtype } : { shape, dtype, quantization };
}

function assertContractError(operation, code, pattern = undefined) {
  let error;
  try {
    operation();
  } catch (caught) {
    error = caught;
  }
  assert.ok(error instanceof OperatorShapeContractError);
  assert.equal(error.code, code);
  if (pattern !== undefined) assert.match(error.message, pattern);
  return error;
}

const perTensorI8 = Object.freeze({
  scheme: 'per_tensor',
  scale: 0.25,
  zero_point: 0,
});

test('Wave-A registry exposes stable IDs, intentional mappings, and fail-closed lookup', () => {
  assert.equal(WAVE_A_SHAPE_FUNCTION_IDS.dense, 'volvox.shape.dense-last-axis.v1');
  assert.equal(
    WAVE_A_OPERATOR_SHAPE_FUNCTIONS.Linear,
    WAVE_A_OPERATOR_SHAPE_FUNCTIONS.Gemm,
  );
  assert.equal(
    WAVE_A_OPERATOR_SHAPE_FUNCTIONS.Gemm,
    WAVE_A_OPERATOR_SHAPE_FUNCTIONS.MatMul,
  );
  assert.equal(WAVE_A_OPERATOR_SHAPE_FUNCTIONS.Add, WAVE_A_OPERATOR_SHAPE_FUNCTIONS.Mul);
  assert.equal(WAVE_A_OPERATOR_SHAPE_FUNCTIONS.ReLU, WAVE_A_OPERATOR_SHAPE_FUNCTIONS.GELU);
  assert.equal(WAVE_A_OPERATOR_SHAPE_FUNCTIONS.PReLU, WAVE_A_OPERATOR_SHAPE_FUNCTIONS.ReLU);
  assert.notEqual(WAVE_A_OPERATOR_SHAPE_FUNCTIONS.Cast, WAVE_A_OPERATOR_SHAPE_FUNCTIONS.Identity);

  const layerNorm = getOperatorShapeContract('LayerNorm');
  const rmsNorm = getOperatorShapeContract('RMSNorm');
  assert.equal(layerNorm.shapeFunctionId, rmsNorm.shapeFunctionId);
  assert.deepEqual(layerNorm.ports.optionalInputs, ['bias']);
  assert.deepEqual(rmsNorm.ports.optionalInputs, []);
  assert.equal(Object.isFrozen(layerNorm), true);
  assert.equal(Object.isFrozen(layerNorm.ports), true);
  assert.equal(Object.isFrozen(layerNorm.ports.requiredInputs), true);
  assert.equal(Object.isFrozen(WAVE_A_OPERATOR_SHAPE_FUNCTIONS), true);

  for (const [operator, shapeFunctionId] of Object.entries(WAVE_A_OPERATOR_SHAPE_FUNCTIONS)) {
    assert.equal(shapeFunctionId, operatorShapeFunctionIds[operator]);
    assert.equal(operatorShapeContracts[operator].classification, 'canonical');
  }

  assertContractError(() => getOperatorShapeContract('relu'), 'UNKNOWN_OPERATOR', /no registered/);
  assertContractError(() => getOperatorShapeContract('Unknown'), 'UNKNOWN_OPERATOR', /Unknown/);
});

test('identity, activations, and Cast infer immutable shape-preserving outputs', () => {
  const firstShape = [2, 3];
  const identity = inferConcreteOperatorShapes('Identity', {
    inputs: { input: tensor(firstShape, 'int32') },
  });
  const relu = inferConcreteOperatorShapes('ReLU', {
    inputs: { input: tensor([1, 4, 8]) },
  });
  const gelu = inferConcreteOperatorShapes('GELU', {
    inputs: { input: tensor([3, 5]) },
    params: { approximate: 'tanh' },
  });
  const cast = inferConcreteOperatorShapes('Cast', {
    inputs: { input: tensor([2, 7], 'int32') },
    params: { to: 'float32' },
  });

  assert.deepEqual(identity.out, { shape: [2, 3], dtype: 'int32' });
  assert.deepEqual(relu.out, { shape: [1, 4, 8], dtype: 'float32' });
  assert.deepEqual(gelu.out, { shape: [3, 5], dtype: 'float32' });
  assert.deepEqual(cast.out, { shape: [2, 7], dtype: 'float32' });
  assert.notStrictEqual(identity.out.shape, firstShape);
  assert.deepEqual(firstShape, [2, 3]);
  assert.equal(Object.isFrozen(identity), true);
  assert.equal(Object.isFrozen(identity.out), true);
  assert.equal(Object.isFrozen(identity.out.shape), true);

  assertContractError(
    () => inferConcreteOperatorShapes('ReLU', {
      inputs: { input: tensor([4], 'int32') },
    }),
    'INVALID_DTYPE',
    /float32/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('GELU', {
      inputs: { input: tensor([4]) },
      params: { approximate: 'fast' },
    }),
    'INVALID_PARAMS',
    /none.*tanh/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('Identity', {
      inputs: { input: tensor([4]), extra: tensor([4]) },
    }),
    'INVALID_INPUT_PORTS',
    /unexpected \[extra\]/,
  );
});

test('QuantizeLinear and DequantizeLinear validate per-tensor and fixed per-axis shapes', () => {
  const perTensor = inferConcreteOperatorShapes('QuantizeLinear', {
    inputs: {
      input: tensor([2, 5]),
      scale: tensor([1]),
      zero_point: tensor([1], 'int8'),
    },
    declaredOutputs: { out: tensor([2, 5], 'int8', perTensorI8) },
  });
  assert.deepEqual(perTensor.out, {
    shape: [2, 5],
    dtype: 'int8',
    quantization: perTensorI8,
  });
  assert.notStrictEqual(perTensor.out.quantization, perTensorI8);

  const perAxis = {
    scheme: 'per_axis',
    axis: 1,
    scales: [0.5, 0.25, 0.125],
    zero_points: [0, 0, 0],
  };
  const quantized = inferConcreteOperatorShapes('QuantizeLinear', {
    inputs: {
      input: tensor([4, 3]),
      scale: tensor([3]),
      zero_point: tensor([3], 'uint8'),
    },
    declaredOutputs: { out: tensor([4, 3], 'uint8', perAxis) },
  });
  const dequantized = inferConcreteOperatorShapes('DequantizeLinear', {
    inputs: {
      input: quantized.out,
      scale: tensor([3]),
      zero_point: tensor([3], 'uint8'),
    },
  });
  assert.deepEqual(quantized.out.shape, [4, 3]);
  assert.equal(quantized.out.dtype, 'uint8');
  assert.equal(Object.isFrozen(quantized.out.quantization.scales), true);
  assert.deepEqual(dequantized.out, { shape: [4, 3], dtype: 'float32' });

  assertContractError(
    () => inferConcreteOperatorShapes('QuantizeLinear', {
      inputs: { input: tensor([4, 3]), scale: tensor([2]) },
      declaredOutputs: { out: tensor([4, 3], 'uint8', perAxis) },
    }),
    'SHAPE_MISMATCH',
    /shape \[3\]/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('DequantizeLinear', {
      inputs: { input: tensor([4], 'int8'), scale: tensor([1]) },
    }),
    'INVALID_QUANTIZATION',
    /is required/,
  );
});

test('QuantizeLinear reads dtype and affine metadata only from its declared output', () => {
  const authoredScale = 0.1;
  const inferred = inferConcreteOperatorShapes('QuantizeLinear', {
    inputs: { input: tensor([2, 4]), scale: tensor([1]) },
    declaredOutputs: {
      out: tensor([2, 4], 'int8', {
        scheme: 'per_tensor',
        scale: authoredScale,
        zero_point: 0,
      }),
    },
  });
  assert.equal(inferred.out.dtype, 'int8');
  assert.equal(inferred.out.quantization.scale, Math.fround(authoredScale));

  assertContractError(
    () => inferConcreteOperatorShapes('QuantizeLinear', {
      inputs: { input: tensor([2, 4]), scale: tensor([1]) },
      declaredOutputs: { out: tensor([2, 4], 'int8', perTensorI8) },
      params: { dtype: 'int8', quantization: perTensorI8 },
    }),
    'INVALID_PARAMS',
    /unsupported field/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('QuantizeLinear', {
      inputs: { input: tensor([2, 4]), scale: tensor([1]) },
    }),
    'INVALID_OUTPUT_PORTS',
    /exactly the 'out'/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('QuantizeLinear', {
      inputs: { input: tensor([2, 4]), scale: tensor([1]) },
      declaredOutputs: {
        out: tensor([2, 4], 'int8', {
          scheme: 'per_axis',
          axis: 3,
          scales: [0.25],
          zero_points: [0],
        }),
      },
    }),
    'INVALID_QUANTIZATION',
    /axis.*\[0, 2\)/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('QuantizeLinear', {
      inputs: { input: tensor([2, 4]), scale: tensor([1]) },
      declaredOutputs: {
        out: tensor([2, 4], 'int8', {
          scheme: 'per_tensor',
          scale: Number.MIN_VALUE,
          zero_point: 0,
        }),
      },
    }),
    'INVALID_QUANTIZATION',
    /representable as float32/,
  );
});

test('PReLU preserves shape with exact input and fixed-slope ports', () => {
  const perFeature = inferConcreteOperatorShapes('PReLU', {
    inputs: { input: tensor([2, 5, 8]), slope: tensor([8]) },
  });
  const scalar = inferConcreteOperatorShapes('PReLU', {
    inputs: { input: tensor([4, 11]), slope: tensor([1]) },
  });
  assert.deepEqual(perFeature.out, { shape: [2, 5, 8], dtype: 'float32' });
  assert.deepEqual(scalar.out, { shape: [4, 11], dtype: 'float32' });
  assert.deepEqual(getOperatorShapeContract('PReLU').ports.requiredInputs, ['input', 'slope']);

  assertContractError(
    () => inferConcreteOperatorShapes('PReLU', {
      inputs: { input: tensor([2, 5, 8]), slope: tensor([7]) },
    }),
    'SHAPE_MISMATCH',
    /match the last input extent 8/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('PReLU', {
      inputs: { input: tensor([2, 5, 8]), slope: tensor([8]) },
      params: { alpha: 0.1 },
    }),
    'INVALID_PARAMS',
    /unsupported field 'alpha'/,
  );
});

test('PReLU domain proof requires a constant slope and a fixed per-feature extent', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 8 },
    { name: 'S', min: 1, max: 64 },
    { name: 'D', min: 4, max: 16, multiple_of: 4 },
  ]);
  const fixedFeature = proveOperatorShapeDomain('PReLU', {
    environment,
    inputs: { input: tensor(['B', 'S', 8]), slope: tensor([8]) },
  });
  const scalarSlope = proveOperatorShapeDomain('PReLU', {
    environment,
    inputs: { input: tensor(['B', 'S', 'D']), slope: tensor([1]) },
  });
  const dynamicFeature = proveOperatorShapeDomain('PReLU', {
    environment,
    inputs: { input: tensor(['B', 'S', 'D']), slope: tensor([8]) },
  });
  const dynamicSlope = proveOperatorShapeDomain('PReLU', {
    environment,
    inputs: { input: tensor(['B', 'S', 'D']), slope: tensor(['D']) },
  });
  assert.equal(fixedFeature.supported, true);
  assert.deepEqual(fixedFeature.outputs.out.shape, ['B', 'S', 8]);
  assert.equal(scalarSlope.supported, true);
  assert.equal(dynamicFeature.supported, false);
  assert.equal(dynamicFeature.code, 'UNPROVABLE_DYNAMIC_FEATURE');
  assert.equal(dynamicSlope.supported, false);
  assert.equal(dynamicSlope.code, 'INVALID_DOMAIN');
  assert.match(dynamicSlope.reason, /constant shapes/);
});

test('normalization contracts preserve two concrete dynamic-prefix shapes and reject bad features', () => {
  const layerNorm = inferConcreteOperatorShapes('LayerNorm', {
    inputs: {
      input: tensor([2, 5, 8]),
      weight: tensor([8]),
      bias: tensor([8]),
    },
    params: { d_model: 8, eps: 1e-5 },
  });
  const rmsNorm = inferConcreteOperatorShapes('RMSNorm', {
    inputs: { input: tensor([1, 17, 8]), weight: tensor([8]) },
  });
  const groupNorm = inferConcreteOperatorShapes('GroupNorm', {
    inputs: {
      input: tensor([3, 9, 7, 8]),
      weight: tensor([8]),
      bias: tensor([8]),
    },
    params: { num_groups: 4 },
  });
  assert.deepEqual(layerNorm.out.shape, [2, 5, 8]);
  assert.deepEqual(rmsNorm.out.shape, [1, 17, 8]);
  assert.deepEqual(groupNorm.out.shape, [3, 9, 7, 8]);

  assertContractError(
    () => inferConcreteOperatorShapes('LayerNorm', {
      inputs: { input: tensor([2, 5, 8]), weight: tensor([7]) },
    }),
    'SHAPE_MISMATCH',
    /shape \[8\]/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('GroupNorm', {
      inputs: {
        input: tensor([1, 2, 3, 8]),
        weight: tensor([8]),
        bias: tensor([8]),
      },
      params: { num_groups: 3 },
    }),
    'SHAPE_MISMATCH',
    /divide 8 channels/,
  );
});

test('Linear, Gemm, and MatMul share fixed-contraction inference with explicit layouts', () => {
  const matmul = inferConcreteOperatorShapes('MatMul', {
    inputs: { input: tensor([2, 3, 4]), weight: tensor([4, 7]) },
  });
  const linear = inferConcreteOperatorShapes('Linear', {
    inputs: {
      input: tensor([5, 4]),
      weight: tensor([7, 4]),
      bias: tensor([7]),
    },
  });
  const gemm = inferConcreteOperatorShapes('Gemm', {
    inputs: { input: tensor([2, 4]), weight: tensor([9, 4]) },
    params: { transB: true },
  });
  assert.deepEqual(matmul.out.shape, [2, 3, 7]);
  assert.deepEqual(linear.out.shape, [5, 7]);
  assert.deepEqual(gemm.out.shape, [2, 9]);

  assertContractError(
    () => inferConcreteOperatorShapes('MatMul', {
      inputs: { input: tensor([2, 5]), weight: tensor([4, 7]) },
    }),
    'SHAPE_MISMATCH',
    /contracted weight extent 4/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('Linear', {
      inputs: { input: tensor([2, 4]), weight: tensor([7, 4]) },
      params: { weight_layout: 'din_dout', transB: false },
    }),
    'INVALID_PARAMS',
    /must not specify both/,
  );
});

test('Embedding appends a fixed table width to different dynamic prefixes', () => {
  const sequence = inferConcreteOperatorShapes('Embedding', {
    inputs: { input: tensor([2, 11], 'int32'), weight: tensor([32000, 16]) },
  });
  const imageTokens = inferConcreteOperatorShapes('Embedding', {
    inputs: { input: tensor([1, 4, 5], 'int32'), weight: tensor([512, 24]) },
  });
  assert.deepEqual(sequence.out, { shape: [2, 11, 16], dtype: 'float32' });
  assert.deepEqual(imageTokens.out, { shape: [1, 4, 5, 24], dtype: 'float32' });

  assertContractError(
    () => inferConcreteOperatorShapes('Embedding', {
      inputs: { input: tensor([2, 3], 'float32'), weight: tensor([10, 4]) },
    }),
    'INVALID_DTYPE',
    /int32/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('Embedding', {
      inputs: { input: tensor([2, 3], 'int32'), weight: tensor([10, 4, 2]) },
    }),
    'INVALID_RANK',
    /rank 2/,
  );
});

test('Add and Mul are exact-shape forms and never infer implicit broadcasting', () => {
  const add = inferConcreteOperatorShapes('Add', {
    inputs: { a: tensor([2, 3, 4]), b: tensor([2, 3, 4]) },
    params: { relu: 2 },
  });
  const multiply = inferConcreteOperatorShapes('Mul', {
    inputs: { a: tensor([1, 9]), b: tensor([1, 9]) },
  });
  assert.deepEqual(add.out.shape, [2, 3, 4]);
  assert.deepEqual(multiply.out.shape, [1, 9]);

  assertContractError(
    () => inferConcreteOperatorShapes('Add', {
      inputs: { a: tensor([2, 3]), b: tensor([1, 3]) },
    }),
    'SHAPE_MISMATCH',
    /broadcasting is explicit/,
  );
  assertContractError(
    () => inferConcreteOperatorShapes('Mul', {
      inputs: { a: tensor([2, 3]), b: tensor([3, 2]) },
    }),
    'SHAPE_MISMATCH',
    /exactly equal/,
  );
});

test('bounded-domain proof preserves symbolic prefixes without mutating specifications', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 8 },
    { name: 'S', min: 1, max: 2048 },
  ]);
  const sourceShape = ['B', 'S', 16];
  const proof = proveOperatorShapeDomain('ReLU', {
    environment,
    inputs: { input: tensor(sourceShape) },
  });
  assert.equal(proof.supported, true);
  assert.deepEqual(proof.outputs.out.shape, ['B', 'S', 16]);
  assert.deepEqual(sourceShape, ['B', 'S', 16]);
  assert.notStrictEqual(proof.outputs.out.shape, sourceShape);
  assert.equal(Object.isFrozen(proof), true);
  assert.equal(Object.isFrozen(proof.outputs), true);
  assert.equal(Object.isFrozen(proof.outputs.out), true);
  assert.equal(Object.isFrozen(proof.outputs.out.shape), true);
  assert.equal(Object.isFrozen(proof.facts), true);

  const exact = proveOperatorShapeDomain('Add', {
    environment,
    inputs: {
      a: tensor(['B', 'S', 16]),
      b: tensor(['B', 'S', 16]),
    },
  });
  assert.equal(exact.supported, true);
  assert.deepEqual(exact.outputs.out.shape, ['B', 'S', 16]);
});

test('bounded-domain proof accepts fixed contractions and rejects dynamic contractions', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 4 },
    { name: 'S', min: 1, max: 128 },
    { name: 'K', min: 2, max: 8 },
    { name: 'KFixed', min: 4, max: 4 },
  ]);
  const fixed = proveOperatorShapeDomain('MatMul', {
    environment,
    inputs: {
      input: tensor(['B', 'S', 4]),
      weight: tensor([4, 12]),
    },
  });
  const singletonSymbol = proveOperatorShapeDomain('MatMul', {
    environment,
    inputs: {
      input: tensor(['B', 'KFixed']),
      weight: tensor([4, 12]),
    },
  });
  const dynamic = proveOperatorShapeDomain('MatMul', {
    environment,
    inputs: {
      input: tensor(['B', 'S', 'K']),
      weight: tensor([4, 12]),
    },
  });
  assert.equal(fixed.supported, true);
  assert.deepEqual(fixed.outputs.out.shape, ['B', 'S', 12]);
  assert.equal(singletonSymbol.supported, true);
  assert.deepEqual(singletonSymbol.outputs.out.shape, ['B', 12]);
  assert.equal(dynamic.supported, false);
  assert.equal(dynamic.code, 'UNPROVABLE_DYNAMIC_CONTRACTION');
  assert.match(dynamic.reason, /complete domain/);
});

test('bounded-domain proof requires fixed normalization feature and channel axes', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 4 },
    { name: 'S', min: 1, max: 128 },
    { name: 'D', min: 8, max: 32, multiple_of: 8 },
    { name: 'C', min: 4, max: 16, multiple_of: 4 },
  ]);
  const feature = proveOperatorShapeDomain('LayerNorm', {
    environment,
    inputs: { input: tensor(['B', 'S', 'D']), weight: tensor([8]) },
  });
  const channel = proveOperatorShapeDomain('GroupNorm', {
    environment,
    inputs: {
      input: tensor(['B', 4, 4, 'C']),
      weight: tensor([4]),
      bias: tensor([4]),
    },
    params: { num_groups: 2 },
  });
  assert.equal(feature.supported, false);
  assert.equal(feature.code, 'UNPROVABLE_DYNAMIC_FEATURE');
  assert.equal(channel.supported, false);
  assert.equal(channel.code, 'UNPROVABLE_DYNAMIC_CHANNEL');
});

test('bounded-domain proof permits dynamic per-tensor quantization and rejects dynamic per-axis extent', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 8 },
    { name: 'S', min: 1, max: 64 },
  ]);
  const perTensor = proveOperatorShapeDomain('QuantizeLinear', {
    environment,
    inputs: {
      input: tensor(['B', 'S']),
      scale: tensor([1]),
      zero_point: tensor([1], 'int8'),
    },
    declaredOutputs: { out: tensor(['B', 'S'], 'int8', perTensorI8) },
  });
  const perAxis = proveOperatorShapeDomain('QuantizeLinear', {
    environment,
    inputs: {
      input: tensor(['B', 'S']),
      scale: tensor([64]),
      zero_point: tensor([64], 'int8'),
    },
    declaredOutputs: {
      out: tensor(['B', 'S'], 'int8', {
        scheme: 'per_axis',
        axis: 1,
        scales: Array(64).fill(0.25),
        zero_points: Array(64).fill(0),
      }),
    },
  });
  assert.equal(perTensor.supported, true);
  assert.deepEqual(perTensor.outputs.out.shape, ['B', 'S']);
  assert.equal(perAxis.supported, false);
  assert.equal(perAxis.code, 'UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT');
});

test('domain proof does not infer equality from coincident bounds on different symbols', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 8 },
    { name: 'LeftS', min: 1, max: 64 },
    { name: 'RightS', min: 1, max: 64 },
  ]);
  const proof = proveOperatorShapeDomain('Mul', {
    environment,
    inputs: {
      a: tensor(['B', 'LeftS']),
      b: tensor(['B', 'RightS']),
    },
  });
  assert.equal(proof.supported, false);
  assert.equal(proof.code, 'SHAPE_MISMATCH');
  assert.match(proof.reason, /not provably exact-shape equal/);
});
