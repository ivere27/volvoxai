import assert from 'node:assert/strict';
import test from 'node:test';
import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';
import {tensorLayout, tensorCoordinates, tensorIndex, readTensorPage, formatTensorValue, summarizeTensorPage}
  from '../examples/common/TensorValues.js';

const dtype = pb.DataType, available = pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE;
function snapshot(type, shape, byteCount, quantization) {
  return new pb.DebugTensorSnapshot({snapshotId: 7n, dtype: type, shape: shape.map(BigInt),
    logicalBytes: BigInt(byteCount), status: available, quantization});
}
function source(data) {
  const calls = [];
  return {calls, async readTensor(snapshotId, offset, limit) {
    calls.push({snapshotId, offset, limit});
    // The transport may return bytes at an unaligned offset within a larger allocation.
    const copy = new Uint8Array(limit + 3);
    copy.set(data.subarray(Number(offset), Number(offset) + limit), 3);
    return {status: available, data: copy.subarray(3), sizeBytes: 8n};
  }};
}
const bytes = hex => Uint8Array.from(Buffer.from(hex.replaceAll(' ', ''), 'hex'));

test('page distributions count finite values once, include both extrema, and exclude nonfinite values', () => {
  const page = summarizeTensorPage([-1, -0.5, 0, 0.5, 1, NaN, Infinity, -Infinity].map(value => ({value})), 4);
  assert.deepEqual(page, {min: -1, max: 1, bins: [1, 1, 1, 2], finite: 5, nonfinite: 3});
  assert.deepEqual(summarizeTensorPage([{value: 7}, {value: 7}], 4).bins, [0, 0, 2, 0]);
  assert.deepEqual(summarizeTensorPage([{value: NaN}], 2), {min: undefined, max: undefined, bins: [0, 0], finite: 0, nonfinite: 1});
});

test('F32 pages preserve finite values, signed zero and nonfinite values in unaligned little-endian bytes', async () => {
  const session = source(bytes('0000803f 00000080 0000807f 000080ff 0000c07f 0000003f'));
  const tensor = snapshot(dtype.DATA_TYPE_F32, [2, 3], 24);
  const rows = await readTensorPage(session, tensor, pb, 1n, 5);
  assert.deepEqual(rows.map(row => row.value), [-0, Infinity, -Infinity, NaN, 0.5]);
  assert.deepEqual(rows.map(row => row.coordinates), [[0n, 1n], [0n, 2n], [1n, 0n], [1n, 1n], [1n, 2n]]);
  assert.deepEqual(rows.map(row => formatTensorValue(row.value)), ['-0', 'Infinity', '-Infinity', 'NaN', '0.5']);
  assert.deepEqual(session.calls, [{snapshotId: 7n, offset: 4n, limit: 20}]);
  assert.ok(rows.every(row => row.dequantized === undefined));
});

test('F16 viewer decodes normals, subnormals, signed zero, infinities and NaN', async () => {
  const session = source(bytes('003c 00c0 0100 ff03 0004 0080 007c 00fc 017e'));
  const rows = await readTensorPage(session, snapshot(dtype.DATA_TYPE_F16, [9], 18), pb, 0n, 64);
  assert.deepEqual(rows.map(row => row.value), [1, -2, 2 ** -24, 1023 * 2 ** -24, 2 ** -14, -0, Infinity, -Infinity, NaN]);
  assert.equal(session.calls[0].limit, 18, 'last page reads only the remaining elements');
});

test('integer viewers preserve signed and unsigned storage values', async () => {
  for (const [type, hex, expected] of [
    [dtype.DATA_TYPE_U8, '007f80ff', [0, 127, 128, 255]],
    [dtype.DATA_TYPE_I8, '007f80ff', [0, 127, -128, -1]],
    [dtype.DATA_TYPE_I32, '00000080 ffffff7f ffffffff 00000000', [-2147483648, 2147483647, -1, 0]],
  ]) {
    const data = bytes(hex);
    const rows = await readTensorPage(source(data), snapshot(type, [4], data.length), pb, 0n, 64);
    assert.deepEqual(rows.map(row => row.value), expected);
  }
});

test('dequantized display uses the captured affine parameters and global coordinates across page boundaries', async () => {
  const perAxis = new pb.AffineQuantizationParameters({perAxis: new pb.PerAxisAffineQuantization({
    axis: 1, scales: [0.5, 2, 0.25], zeroPoints: [1, 10, 4],
  })});
  const session = source(bytes('0103050a0c10040608'));
  const rows = await readTensorPage(session, snapshot(dtype.DATA_TYPE_U8, [1, 3, 3], 9, perAxis), pb, 2n, 5);
  assert.deepEqual(rows.map(row => row.value), [5, 10, 12, 16, 4]);
  assert.deepEqual(rows.map(row => row.dequantized), [2, 0, 4, 12, 0]);
  const perTensor = new pb.AffineQuantizationParameters({perTensor: new pb.PerTensorAffineQuantization({scale: 0.25, zeroPoint: -2})});
  const signed = await readTensorPage(source(bytes('fe0006')), snapshot(dtype.DATA_TYPE_I8, [3], 3, perTensor), pb, 0n, 64);
  assert.deepEqual(signed.map(row => row.dequantized), [0, 0.5, 2]);
  perTensor.perTensor.scale = 0;
  await assert.rejects(readTensorPage(source(bytes('fe')), snapshot(dtype.DATA_TYPE_I8, [1], 1, perTensor), pb, 0n, 1), /invalid affine/);
});

test('row-major coordinate lookup handles scalars, empty dimensions, exact large indices and input errors', async () => {
  const shape = [1n, 1n, 320n, 672n];
  assert.equal(tensorIndex(shape, '[0, 0, 123, 456]'), 83112n);
  assert.deepEqual(tensorCoordinates(shape, 83112n), [0n, 0n, 123n, 456n]);
  assert.equal(tensorIndex([], '[]'), 0n);
  assert.deepEqual(tensorCoordinates([], 0n), []);
  assert.equal(tensorLayout(snapshot(dtype.DATA_TYPE_F32, [], 4), pb).count, 1n);
  const emptySource = {readTensor() { assert.fail('An empty tensor should not read any bytes'); }};
  assert.deepEqual(await readTensorPage(emptySource, snapshot(dtype.DATA_TYPE_F32, [0, 5], 0), pb, 0n, 256), []);
  const largeShape = [9007199254740995n, 2n];
  const largeIndex = tensorIndex(largeShape, '9007199254740994, 1');
  assert.equal(largeIndex, 18014398509481989n);
  assert.deepEqual(tensorCoordinates(largeShape, largeIndex), [9007199254740994n, 1n]);
  for (const input of ['0, 0, 320, 0', '0, -1, 0, 0', '0, 0', '0, 0, 1.5, 0', '0, 0, 1e2, 0'])
    assert.throws(() => tensorIndex(shape, input));
  assert.throws(() => tensorCoordinates([0n], 0n), /outside/);
  assert.throws(() => tensorCoordinates(shape, 215040n), /outside/);
  assert.throws(() => tensorLayout(snapshot(dtype.DATA_TYPE_F32, [2], 4), pb), /byte count/);
});

test('value pages use exact BigInt byte ranges and reject unavailable or truncated data', async () => {
  const offset = 9007199254740993n;
  const tensor = snapshot(dtype.DATA_TYPE_F32, [offset + 2n], (offset + 2n) * 4n);
  const session = {async readTensor(id, byteOffset, limit) {
    assert.equal(id, 7n); assert.equal(byteOffset, offset * 4n); assert.equal(limit, 8);
    return {status: available, data: bytes('0000803f 00000040'), sizeBytes: 8n};
  }};
  const rows = await readTensorPage(session, tensor, pb, offset, 1024);
  assert.deepEqual(rows.map(row => row.value), [1, 2]);
  assert.deepEqual(rows.map(row => row.index), [offset, offset + 1n]);
  for (const limit of [0, 1025, 1.5])
    await assert.rejects(readTensorPage(session, tensor, pb, 0n, limit), /1–1024/);
  await assert.rejects(readTensorPage(session, tensor, pb, -1n, 1), /outside/);
  await assert.rejects(readTensorPage(session, tensor, pb, offset + 2n, 1), /outside/);
  const small = snapshot(dtype.DATA_TYPE_F32, [2], 8);
  for (const chunk of [
    {status: pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_STATISTICS_ONLY, data: new Uint8Array(), sizeBytes: 8n},
    {status: available, data: new Uint8Array(4), sizeBytes: 8n},
    {status: available, data: new Uint8Array(12), sizeBytes: 12n},
  ]) await assert.rejects(readTensorPage({readTensor: async () => chunk}, small, pb, 0n, 64));
});
