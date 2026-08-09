import test from 'node:test';
import assert from 'node:assert/strict';

import { _cpuBatchNorm2D } from '../ts/ops/batchNorm2D.js';
import { _cpuConcat2 } from '../ts/ops/concat2.js';
import { _cpuCrossAttention } from '../ts/ops/crossAttention.js';
import { _cpuDropout, dropoutContext } from '../ts/ops/dropout.js';
import { _cpuExpand } from '../ts/ops/expand.js';
import { _cpuInterp1D } from '../ts/ops/interp1D.js';
import { _cpuMeanHeight } from '../ts/ops/meanHeight.js';
import { _cpuMoELinear } from '../ts/ops/moeLinear.js';
import { _cpuMoERouter } from '../ts/ops/moeRouter.js';
import { _cpuNot } from '../ts/ops/comparison.js';
import { _cpuProfileX } from '../ts/ops/profileX.js';
import { _cpuProfileY } from '../ts/ops/profileY.js';
import { _cpuRequantizeLinear } from '../ts/ops/requantizeLinear.js';
import { _cpuSpatialSoftargmaxY } from '../ts/ops/spatialSoftargmaxY.js';
import { _cpuSSMScan } from '../ts/ops/ssmScan.js';
import { _cpuWhere } from '../ts/ops/where.js';

const constructors = Object.freeze({
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
});

function elementCount(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function tensor(shape, dtype = 'float32', values = undefined, quantization = undefined) {
  const Constructor = constructors[dtype];
  const count = elementCount(shape);
  const source = values ?? Array.from({ length: count }, (_, index) =>
    dtype === 'float32' ? (index + 1) / 16 : index % 7);
  return {
    shape: [...shape], dtype, buffer: Constructor.from(source),
    ...(quantization === undefined ? {} : { quantization }),
  };
}

function output(shape, dtype = 'float32', quantization = undefined) {
  const result = tensor(shape, dtype, undefined, quantization);
  result.buffer.fill(dtype === 'uint8' ? 233 : dtype === 'int8' ? -99 : 99);
  return result;
}

function scenario(run, outputs) {
  return Object.freeze({ run, outputs: Object.freeze(outputs) });
}

function assertLegal(factory, label) {
  const candidate = factory();
  assert.doesNotThrow(candidate.run, label);
  for (const result of candidate.outputs) {
    assert.ok([...result.buffer].every(Number.isFinite), `${label} produced a non-finite value`);
  }
}

function assertRejectedWithoutWrite(factory, label) {
  const candidate = factory();
  const before = candidate.outputs.map((result) => [...result.buffer]);
  assert.throws(candidate.run, undefined, `${label} must reject its illegal descriptor`);
  candidate.outputs.forEach((result, index) => {
    assert.deepEqual([...result.buffer], before[index], `${label} wrote output before rejecting`);
  });
}

function moeRouter(prefix, feature, experts, topK, invalidOutput = false) {
  const input = tensor([...prefix, feature]);
  const weight = tensor([feature, experts]);
  const routeShape = [...prefix, topK];
  const indices = output(routeShape);
  const weights = output(invalidOutput ? [...prefix, topK + 1] : routeShape);
  return scenario(() => _cpuMoERouter({
    id: 'router', opType: 'MoERouter', inputs: { input, weight },
    outputs: { indices, weights }, params: { num_experts: experts, top_k: topK },
  }), [indices, weights]);
}

function moeLinear(prefix, feature, experts, topK, width, invalidRoute = false) {
  const input = tensor([...prefix, feature]);
  const expert_weight = tensor([experts, feature, width]);
  const routeShape = [...prefix, topK];
  const routes = elementCount(routeShape);
  const route_indices = tensor(
    routeShape, 'float32',
    Array.from({ length: routes }, (_, index) =>
      invalidRoute && index === routes - 1 ? experts : index % experts),
  );
  const route_weights = tensor(routeShape, 'float32', new Array(routes).fill(1 / topK));
  const out = output([...prefix, width]);
  return scenario(() => _cpuMoELinear({
    id: 'experts', opType: 'MoELinear',
    inputs: { input, expert_weight, route_indices, route_weights },
    outputs: { out }, params: {},
  }), [out]);
}

function crossAttention(qShape, keyLength, heads, invalidOutput = false) {
  const rank = qShape.length;
  const feature = qShape.at(-1);
  const kvShape = rank === 3 ? [qShape[0], keyLength, feature] : [keyLength, feature];
  const q = tensor(qShape);
  const kv = tensor(kvShape);
  const weight = tensor([3 * feature, feature]);
  const out = output(invalidOutput ? [elementCount(qShape), 1] : qShape);
  return scenario(() => _cpuCrossAttention({
    id: 'cross', opType: 'CrossAttention', inputs: { q, kv, weight },
    outputs: { out }, params: { heads },
  }), [out]);
}

function batchNorm(shape, invalidVector = false) {
  const channels = shape[3];
  const input = tensor(shape);
  const weight = tensor([channels], 'float32', new Array(channels).fill(1));
  const bias = tensor([channels], 'float32', new Array(channels).fill(0));
  const running_mean = tensor([channels], 'float32', new Array(channels).fill(0));
  const running_var = tensor(
    [invalidVector ? channels + 1 : channels], 'float32',
    new Array(invalidVector ? channels + 1 : channels).fill(1),
  );
  const out = output(shape);
  return scenario(() => _cpuBatchNorm2D({
    id: 'batch-norm', opType: 'BatchNorm2D',
    inputs: { input, weight, bias, running_mean, running_var },
    outputs: { out }, params: { eps: 1e-5 },
  }), [out]);
}

function interpolate(shape, size, invalidOutput = false) {
  const input = tensor(shape);
  const expected = [shape[0], shape[1], size];
  const out = output(invalidOutput ? [shape[0], size, shape[1]] : expected);
  return scenario(() => _cpuInterp1D({
    id: 'interp', opType: 'Interpolate1D', inputs: { input }, outputs: { out },
    params: { size },
  }), [out]);
}

function logicalNot(shape, invalidOutput = false) {
  const input = tensor(shape, 'int32', Array.from({ length: elementCount(shape) }, (_, i) => i - 2));
  const outShape = invalidOutput ? [...shape].reverse() : shape;
  const out = output(outShape, 'int32');
  return scenario(() => _cpuNot({
    id: 'not', opType: 'Not', inputs: { input }, outputs: { out }, params: {},
  }), [out]);
}

function mask(shape, dataDtype, conditionDtype, invalidMask = false) {
  const maskShape = invalidMask ? [shape[0], 1] : shape;
  const maskValues = Array.from({ length: elementCount(maskShape) }, (_, index) => index % 2);
  const condition = tensor(maskShape, conditionDtype, maskValues);
  const a = tensor(shape, dataDtype);
  const b = tensor(shape, dataDtype, new Array(elementCount(shape)).fill(-1));
  const out = output(shape, dataDtype);
  return scenario(() => _cpuWhere({
    id: 'mask', opType: 'Mask', inputs: { mask: condition, a, b }, outputs: { out }, params: {},
  }), [out]);
}

function broadcast(inputShape, outputShape, dtype = 'float32', affine = undefined, invalid = false) {
  const input = tensor(inputShape, dtype, undefined, affine?.input);
  const out = output(outputShape, dtype, affine?.output);
  const target = invalid ? outputShape.map((value, index) => index === outputShape.length - 1 ? value + 1 : value) : outputShape;
  return scenario(() => _cpuExpand({
    id: 'broadcast', opType: 'Broadcast', inputs: { input }, outputs: { out },
    params: { shape: target },
  }), [out]);
}

function concat2(aShape, bShape, axis, dtype = 'float32', affine = undefined, invalid = false) {
  const a = tensor(aShape, dtype, undefined, affine?.a);
  const actualBShape = invalid ? bShape.map((value, index) => index === axis ? value : value + 1) : bShape;
  const b = tensor(actualBShape, dtype, undefined, affine?.b);
  const outShape = [...aShape];
  outShape[axis] += bShape[axis];
  const out = output(outShape, dtype, affine?.out);
  return scenario(() => _cpuConcat2({
    id: 'concat2', opType: 'Concat2', inputs: { a, b }, outputs: { out },
    params: { axis, sigmoid: false },
  }), [out]);
}

function requantize(shape, inputDtype, outputDtype, invalidShape = false) {
  const inputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: inputDtype === 'int8' ? -1 : 127 };
  const outputQuantization = { scheme: 'per_tensor', scale: 0.5, zero_point: outputDtype === 'int8' ? -2 : 128 };
  const input = tensor(shape, inputDtype, undefined, inputQuantization);
  const out = output(invalidShape ? [elementCount(shape)] : shape, outputDtype, outputQuantization);
  return scenario(() => _cpuRequantizeLinear({
    id: 'requant', opType: 'RequantizeLinear', inputs: { input }, outputs: { out }, params: {},
  }), [out]);
}

function scan(opType, inputShape, stateWidth, invalidDelta = false) {
  const rank = inputShape.length;
  const batch = rank === 3 ? inputShape[0] : 1;
  const sequence = inputShape.at(-2);
  const channels = inputShape.at(-1);
  const input = tensor(inputShape);
  const deltaShape = invalidDelta ? [...inputShape].reverse() : inputShape;
  const delta = tensor(deltaShape, 'float32', new Array(elementCount(deltaShape)).fill(0.1));
  const A = tensor([channels, stateWidth], 'float32', new Array(channels * stateWidth).fill(-0.5));
  const B = tensor([stateWidth]);
  const C = tensor([sequence, stateWidth]);
  const out = output(inputShape);
  const state = output([batch, channels, stateWidth]);
  return scenario(() => _cpuSSMScan({
    id: 'scan', opType, inputs: { input, delta, A, B, C }, outputs: { out, state },
    params: { delta_softplus: true },
  }), [out, state]);
}

const visionKernels = Object.freeze({
  SpatialSoftargmaxY: Object.freeze({ run: _cpuSpatialSoftargmaxY, kind: 'reduced' }),
  MeanHeight: Object.freeze({ run: _cpuMeanHeight, kind: 'reduced' }),
  ProfileX: Object.freeze({ run: _cpuProfileX, kind: 'x' }),
  ProfileY: Object.freeze({ run: _cpuProfileY, kind: 'y' }),
});

function vision(operator, shape, invalidOutput = false) {
  const entry = visionKernels[operator];
  const [n, h, w, c] = shape;
  const expected = entry.kind === 'x' ? [n, 2 * c, w] :
    entry.kind === 'y' ? [n, 2 * c, h] : [n, c, w];
  const outShape = invalidOutput && expected.length === 3
    ? [expected[0], expected[2], expected[1]]
    : expected;
  const input = tensor(shape);
  const out = output(outShape);
  return scenario(() => entry.run({
    id: 'vision', opType: operator, inputs: { input }, outputs: { out }, params: {},
  }), [out]);
}

function dropout(shape, invalidOutput = false, training = false) {
  const input = tensor(shape);
  const out = output(invalidOutput ? [...shape].reverse() : shape);
  return scenario(() => _cpuDropout({
    id: 'dropout', opType: 'Dropout', inputs: { input }, outputs: { out }, params: { ratio: 0.25, seed: 7 },
  }, training ? { training: { dropout: dropoutContext({ seed: 11, counter: 3 }) } } : {}), [out]);
}

const axisAffineInput = Object.freeze({
  scheme: 'per_axis', axis: 1, scales: Object.freeze([0.5]), zero_points: Object.freeze([0]),
});
const axisAffineBroadcast = Object.freeze({
  scheme: 'per_axis', axis: 2, scales: Object.freeze([0.5]), zero_points: Object.freeze([0]),
});
const concatAffine = Object.freeze({
  a: Object.freeze({ scheme: 'per_axis', axis: 1, scales: Object.freeze([0.5, 0.25]), zero_points: Object.freeze([0, 0]) }),
  b: Object.freeze({ scheme: 'per_axis', axis: 1, scales: Object.freeze([0.125]), zero_points: Object.freeze([0]) }),
  out: Object.freeze({ scheme: 'per_axis', axis: 1, scales: Object.freeze([0.5, 0.25, 0.125]), zero_points: Object.freeze([0, 0, 0]) }),
});

const cases = Object.freeze([
  ['MoERouter', () => moeRouter([2], 3, 4, 2), () => moeRouter([1, 2], 2, 3, 1), () => moeRouter([2], 3, 4, 2, true)],
  ['MoELinear', () => moeLinear([2], 3, 2, 1, 4), () => moeLinear([1, 2], 2, 3, 2, 1), () => moeLinear([2], 3, 2, 1, 4, true)],
  ['CrossAttention', () => crossAttention([2, 2], 3, 1), () => crossAttention([2, 1, 4], 2, 2), () => crossAttention([2, 2], 3, 1, true)],
  ['BatchNorm2D', () => batchNorm([1, 2, 2, 2]), () => batchNorm([2, 1, 3, 1]), () => batchNorm([2, 1, 3, 1], true)],
  ['Interpolate1D', () => interpolate([1, 1, 2], 4), () => interpolate([2, 3, 4], 2), () => interpolate([1, 1, 2], 4, true)],
  ['Not', () => logicalNot([4]), () => logicalNot([2, 3]), () => logicalNot([2, 3], true)],
  ['Mask', () => mask([4], 'float32', 'int32'), () => mask([2, 2], 'int32', 'float32'), () => mask([2, 2], 'float32', 'int32', true)],
  ['Broadcast', () => broadcast([1, 2], [3, 1, 2]), () => broadcast([2, 1], [4, 2, 1], 'int8', { input: axisAffineInput, output: axisAffineBroadcast }), () => broadcast([2, 2], [2, 2], 'float32', undefined, true)],
  ['Concat2', () => concat2([2, 1], [2, 2], 1), () => concat2([1, 2], [1, 1], 1, 'int8', concatAffine), () => concat2([1, 2], [1, 1], 1, 'float32', undefined, true)],
  ['RequantizeLinear', () => requantize([4], 'int8', 'uint8'), () => requantize([2, 2], 'uint8', 'int8'), () => requantize([2, 2], 'int8', 'uint8', true)],
  ['SSMScan', () => scan('SSMScan', [2, 1], 1), () => scan('SSMScan', [2, 2, 1], 2), () => scan('SSMScan', [2, 1], 1, true)],
  ['SelectiveScan', () => scan('SelectiveScan', [3, 2], 2), () => scan('SelectiveScan', [2, 2, 1], 1), () => scan('SelectiveScan', [2, 1], 1, true)],
  ...Object.keys(visionKernels).map((operator) => [
    operator,
    () => vision(operator, [1, 2, 1, 1]),
    () => vision(operator, [2, 1, 3, 2]),
    () => vision(operator, [2, 1, 3, 2], true),
  ]),
  ['Dropout', () => dropout([4]), () => dropout([2, 3], false, true), () => dropout([2, 3], true)],
]);

test('formerly deferred CPU kernels accept two legal shapes and reject one illegal descriptor atomically', () => {
  assert.equal(cases.length, 17);
  for (const [operator, first, second, illegal] of cases) {
    assertLegal(first, `${operator} legal shape A`);
    assertLegal(second, `${operator} legal shape B`);
    assertRejectedWithoutWrite(illegal, `${operator} illegal shape`);
  }
});
