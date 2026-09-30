import assert from 'node:assert/strict';
import test from 'node:test';
import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';
import {receiptSpatialPlan} from '../examples/receipt_digit_reader/ReceiptSpatialPlan.js';
import {readSpatialPlane, spatialCell, spatialCoordinates, spatialColor} from '../examples/common/SpatialTensor.js';

function fixture() {
  const tensors = [], nodes = [], steps = [];
  const tensor = (name, shape) => { tensors.push({tensorId: tensors.length, name, shape: shape.map(BigInt)}); return name; };
  const add = (opType, inputs, shape, params = {}) => {
    const name = `v${nodes.length}`, id = `n${nodes.length}`; tensor(name, shape);
    nodes.push({id, opType, inputs, outputs: {out: {tensor: name}}, params});
    steps.push({scheduleIndex: steps.length, sourceNodeIds: [id], outputs: [tensors.length - 1]});
    return name;
  };
  const input = tensor('pixels', [1, 1, 5, 8]), weight = tensor('kernel', [3, 3, 1, 4]);
  const manifest = {abi: {input: {name: input}}, preprocess: {height: 5, width: 8}, decode: {slots: 2}};
  const reshape = add('Reshape', {input}, [1, 5, 8, 1]);
  const conv = add('Conv2D', {input: reshape, weight}, [1, 3, 4, 4],
    {data_layout: 'NHWC', weight_layout: 'HWIO', stride: [2, 2], pads: [0, 1, 2, 0]});
  const norm = add('GroupNorm', {input: conv}, [1, 3, 4, 4]);
  const transpose = add('Transpose', {input: norm}, [1, 4, 3, 4], {perm: [0, 3, 1, 2]});
  const flat = add('Reshape', {input: transpose}, [1, 4, 12]);
  const query = tensor('query', [1, 2, 4]);
  const score = add('BatchMatMul', {a: query, b: flat}, [1, 2, 12]);
  const attention = add('Softmax', {input: score}, [1, 2, 12], {axis: -1});
  const features = add('Transpose', {input: flat}, [1, 12, 4], {perm: [0, 2, 1]});
  const pooled = add('BatchMatMul', {a: attention, b: features}, [1, 2, 4]);
  const ambiguous = add('Reshape', {input: transpose}, [1, 2, 6, 4]);
  return {graph: {nodes}, plan: {tensors, steps}, manifest, names: {conv, norm, transpose, flat, attention, features, pooled, ambiguous}, add};
}

test('graph mapping preserves grid centers through stride, asymmetric padding, layout changes, flattening and slot attention', () => {
  const f = fixture(), {maps, attention} = receiptSpatialPlan(f.graph, f.plan, f.manifest), n = f.names;
  assert.deepEqual(maps.get(n.conv).grid, {height: 3, width: 4, originX: 0, originY: 1, stepX: 2, stepY: 2});
  assert.equal(maps.get(n.conv).globalMixing, false);
  assert.equal(maps.get(n.norm).globalMixing, true, 'GroupNorm does not have a local dependency boundary');
  assert.deepEqual(maps.get(n.transpose).axes, ['b', 'c', 'y', 'x']);
  assert.deepEqual(maps.get(n.flat).axes, ['b', 'c', 'p']);
  assert.deepEqual(maps.get(n.features).axes, ['b', 'p', 'c']);
  assert.deepEqual(maps.get(n.attention).axes, ['b', 's', 'p']);
  assert.equal(attention[0].name, n.attention);
  assert.equal(attention[0].step, 6);
  assert.equal(maps.has(n.pooled), false, 'contracting spatial positions destroys the image map');
  assert.equal(maps.has(n.ambiguous), false, 'matching element count is not a verified reshape');
  assert.equal(maps.has('query'), false, 'constant weights are not spatial just because their shape fits');
});

test('unsupported transpose, convolution and non-spatial softmax cannot fabricate image mappings', () => {
  for (const mutate of [
    f => { f.graph.nodes[1].params.stride = [1, 1]; },
    f => { f.graph.nodes[3].params.perm = [0, 1, 2, 3]; },
    f => { f.graph.nodes[6].params.axis = 1; },
    f => { f.graph.nodes[5].params.transpose_b = true; },
  ]) {
    const f = fixture(); mutate(f);
    assert.equal(receiptSpatialPlan(f.graph, f.plan, f.manifest).attention.length, 0);
  }
  const f = fixture(); f.plan.tensors[0].shape[0] = 2n;
  assert.equal(receiptSpatialPlan(f.graph, f.plan, f.manifest).maps.size, 0);
});

const available = pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE;
function captured(shape, data, dtype = pb.DataType.DATA_TYPE_F32, quantization) {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength), calls = [];
  const snapshot = {snapshotId: 1n, shape: shape.map(BigInt), dtype, logicalBytes: BigInt(bytes.length), quantization};
  return {snapshot, calls, session: {async readTensor(id, offset, limit) {
    calls.push({offset: Number(offset), limit});
    // Exercise unaligned transport views.
    const chunk = new Uint8Array(limit + 3); chunk.set(bytes.subarray(Number(offset), Number(offset) + limit), 3);
    return {status: available, data: chunk.subarray(3), sizeBytes: 24n};
  }}};
}
function descriptor(axes, shape, width = 3, height = 2) {
  return {axes, shape, kind: axes.includes('s') ? 'attention' : 'feature', inputWidth: width, inputHeight: height,
    grid: {width, height, originX: 0, originY: 0, stepX: 1, stepY: 1}};
}

test('planar and interleaved channel slices match exact stored pixels and bounded reads', async () => {
  const expected = [-0, 1, 2, 3, NaN, Infinity], other = [-2, 4, 6, 8, 10, 12];
  for (const [axes, shape, data, bytesRead] of [
    [['b', 'c', 'y', 'x'], [1, 2, 2, 3], [...other, ...expected], 24],
    [['b', 'y', 'x', 'c'], [1, 2, 3, 2], other.flatMap((v, i) => [v, expected[i]]), 44],
    [['b', 'c', 'p'], [1, 2, 6], [...other, ...expected], 24],
    [['b', 'p', 'c'], [1, 6, 2], other.flatMap((v, i) => [v, expected[i]]), 44],
    [['b', 's', 'p'], [1, 2, 6], [...other, ...expected], 24],
  ]) {
    const s = captured(shape, Float32Array.from(data)), d = descriptor(axes, shape);
    const plane = await readSpatialPlane(s.session, s.snapshot, pb, d, {index: 1});
    assert.deepEqual([...plane.values], expected);
    assert.equal(plane.bytesRead, bytesRead); assert.equal(plane.values.byteLength, 48);
    assert.equal(plane.min, -0); assert.equal(plane.max, 3); assert.equal(plane.nonfinite, 2);
    assert.deepEqual(spatialCoordinates(plane, 2, 1), axes.map(a => ({b: 0, c: 1, s: 1, y: 1, x: 2, p: 5})[a]));
  }
});

test('dequantization uses each actual channel before reduction, and scans without full-tensor allocation', async () => {
  const shape = [1, 2, 3, 2], d = descriptor(['b', 'y', 'x', 'c'], shape);
  const q = {perAxis: {axis: 3, scales: [0.5, 2], zeroPoints: [10, 20]}};
  const s = captured(shape, Uint8Array.from([8, 21, 10, 22, 12, 23, 14, 24, 16, 25, 18, 26]), pb.DataType.DATA_TYPE_U8, q);
  const one = await readSpatialPlane(s.session, s.snapshot, pb, d, {index: 0, domain: 'dequantized'});
  assert.deepEqual([...one.values], [-1, 0, 1, 2, 3, 4]);
  const mean = await readSpatialPlane(s.session, s.snapshot, pb, d, {reduction: 'mean-abs', domain: 'dequantized'});
  assert.deepEqual([...mean.values], [1.5, 2, 3.5, 5, 6.5, 8]);
  const largeShape = [1, 100, 101, 8], large = captured(largeShape, Float32Array.from({length: 80800}, (_, i) => i));
  const plane = await readSpatialPlane(large.session, large.snapshot, pb, descriptor(['b', 'y', 'x', 'c'], largeShape, 101, 100), {index: 3});
  assert.equal(plane.values.length, 10100); assert.equal(plane.values[10099], 10099 * 8 + 3);
  assert.ok(large.calls.length > 1 && large.calls.every(call => call.limit <= 65536));
  assert.equal(plane.bytesRead, (80800 - 7) * 4);
});

test('shape mismatches, invalid selections, absent values and truncated chunks fail explicitly', async () => {
  const shape = [1, 1, 2, 3], s = captured(shape, new Float32Array(6)), d = descriptor(['b', 'c', 'y', 'x'], shape);
  await assert.rejects(readSpatialPlane(s.session, s.snapshot, pb, {...d, shape: [1, 1, 3, 2]}), /shape differs/);
  await assert.rejects(readSpatialPlane(s.session, s.snapshot, pb, d, {index: 1}), /Invalid channel/);
  await assert.rejects(readSpatialPlane(s.session, s.snapshot, pb, d, {domain: 'dequantized'}), /no affine/);
  await assert.rejects(readSpatialPlane(s.session, {...s.snapshot, quantization: {}}, pb, d, {domain: 'dequantized'}), /no affine/);
  await assert.rejects(readSpatialPlane(s.session, s.snapshot, pb, {...d, axes: ['b', 'unknown', 'y', 'x']}), /Spatial axes/);
  for (const result of [
    {status: available, data: new Uint8Array(20), sizeBytes: 24n},
    {status: available, data: new Uint8Array(28), sizeBytes: 28n},
    {status: pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_BUDGET_EXCEEDED, data: new Uint8Array(24), sizeBytes: 24n},
  ]) await assert.rejects(readSpatialPlane({readTensor: async () => result}, s.snapshot, pb, d), /incomplete or unavailable/);
});

test('memory limits reject oversized maps and read spans before allocating or accessing tensor data', async () => {
  const noRead = {readTensor() { assert.fail('Oversized maps must not read data'); }};
  for (const [shape, error] of [[[1, 1025, 1024, 1], /1,048,576/], [[1, 512, 512, 128], /64 MiB/]]) {
    const snapshot = {dtype: pb.DataType.DATA_TYPE_F32, shape: shape.map(BigInt), logicalBytes: BigInt(shape.reduce((a, b) => a * b, 4))};
    await assert.rejects(readSpatialPlane(noRead, snapshot, pb, descriptor(['b', 'y', 'x', 'c'], shape, shape[2], shape[1])), error);
  }
});

test('overlay pixels align with sampling centers, including resized images and clamped edges', () => {
  const d = {inputWidth: 8, inputHeight: 5, grid: {width: 4, height: 3, originX: 0, originY: 1, stepX: 2, stepY: 2}};
  assert.deepEqual(spatialCell(d, 4, 3, 8, 5), {x: 2, y: 1});
  assert.deepEqual(spatialCell(d, 8, 6, 16, 10), {x: 2, y: 1});
  assert.deepEqual(spatialCell(d, 0, 0, 8, 5), {x: 0, y: 0});
  assert.deepEqual(spatialCell(d, 7, 4, 8, 5), {x: 3, y: 2});
  assert.deepEqual(spatialColor(1, 1, 1), [28, 177, 191]);
  assert.deepEqual(spatialColor(NaN, 0, 1), [230, 60, 210]);
  assert.deepEqual(spatialColor(-5, 0, 1), [22, 34, 88]);
  assert.deepEqual(spatialColor(5, 0, 1), [255, 222, 65]);
});
