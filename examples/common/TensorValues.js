/** Decode immutable tensor pages for display. No numerical engine work lives here. */
export function tensorDecoder(dtype, pb) {
  switch (dtype) {
    case pb.DataType.DATA_TYPE_F32: return {width: 4, read: (view, offset) => view.getFloat32(offset, true)};
    case pb.DataType.DATA_TYPE_I32: return {width: 4, read: (view, offset) => view.getInt32(offset, true)};
    case pb.DataType.DATA_TYPE_I8: return {width: 1, read: (view, offset) => view.getInt8(offset)};
    case pb.DataType.DATA_TYPE_U8: return {width: 1, read: (view, offset) => view.getUint8(offset)};
    case pb.DataType.DATA_TYPE_F16: return {width: 2, read: (view, offset) => {
      const bits = view.getUint16(offset, true), exponent = (bits >> 10) & 31, fraction = bits & 1023;
      const magnitude = exponent === 31 ? (fraction ? NaN : Infinity) :
        exponent === 0 ? fraction * 2 ** -24 : (1 + fraction / 1024) * 2 ** (exponent - 15);
      return bits & 32768 ? -magnitude : magnitude;
    }};
    default: throw new Error('This dtype has no value viewer. The raw tensor download is still available.');
  }
}

export function tensorLayout(snapshot, pb) {
  const {width} = tensorDecoder(snapshot.dtype, pb);
  const shape = snapshot.shape.map(BigInt);
  if (shape.some(extent => extent < 0n)) throw new Error('A captured tensor must have concrete dimensions');
  const count = shape.reduce((total, extent) => total * extent, 1n);
  if (count * BigInt(width) !== BigInt(snapshot.logicalBytes))
    throw new Error('Tensor shape and dtype do not match its captured byte count');
  return {shape, count, width};
}

export function tensorCoordinates(shape, index) {
  const count = shape.reduce((total, extent) => total * extent, 1n);
  if (index < 0n || index >= count) throw new Error('Element index is outside the tensor');
  const coordinates = new Array(shape.length);
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    coordinates[axis] = index % shape[axis];
    index /= shape[axis];
  }
  return coordinates;
}

export function tensorIndex(shape, text) {
  const contents = text.trim().replace(/^\[(.*)\]$/, '$1').trim();
  const parts = contents ? contents.split(',').map(value => value.trim()) : [];
  if (parts.length !== shape.length || parts.some(value => !/^\d+$/.test(value)))
    throw new Error(`Enter ${shape.length} non-negative coordinates separated by commas`);
  let index = 0n;
  for (let axis = 0; axis < shape.length; axis++) {
    const coordinate = BigInt(parts[axis]);
    if (coordinate >= shape[axis]) throw new Error(`Axis ${axis} must be less than ${shape[axis]}`);
    index = index * shape[axis] + coordinate;
  }
  return index;
}

export const formatTensorValue = value => Object.is(value, -0) ? '-0' : String(value);

/** Page-local storage statistics for visualization, never whole-tensor estimates. */
export function summarizeTensorPage(rows, binCount = 24) {
  const finite = rows.map(row => row.value).filter(Number.isFinite);
  const min = finite.length ? Math.min(...finite) : undefined;
  const max = finite.length ? Math.max(...finite) : undefined;
  const bins = Array(binCount).fill(0);
  for (const value of finite) {
    const bin = min === max ? Math.floor(binCount / 2) : Math.min(binCount - 1, Math.floor((value - min) / (max - min) * binCount));
    bins[bin]++;
  }
  return {min, max, bins, finite: finite.length, nonfinite: rows.length - finite.length};
}

export function dequantizeTensorValue(value, coordinates, parameters) {
  if (!parameters?.perTensor && !parameters?.perAxis) return undefined;
  let scale, zero;
  if (parameters.perTensor) {
    scale = parameters.perTensor.scale; zero = parameters.perTensor.zeroPoint;
  } else {
    const affine = parameters.perAxis;
    const index = Number(coordinates[affine.axis]);
    scale = affine.scales[index]; zero = affine.zeroPoints[index];
  }
  if (!(scale > 0) || !Number.isFinite(scale) || !Number.isInteger(zero))
    throw new Error('The capture has invalid affine quantization parameters');
  return (value - zero) * scale;
}

export async function readTensorPage(session, snapshot, pb, start, limit) {
  const {shape, count, width} = tensorLayout(snapshot, pb);
  if (!Number.isInteger(limit) || limit < 1 || limit > 1024)
    throw new Error('A value page must contain 1–1024 elements');
  if (start < 0n || (count ? start >= count : start !== 0n))
    throw new Error('Start element is outside the tensor');
  if (!count) return [];
  const size = Number(count - start < BigInt(limit) ? count - start : BigInt(limit));
  const offset = start * BigInt(width);
  const chunk = await session.readTensor(snapshot.snapshotId, offset, size * width);
  if (chunk.status !== pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE)
    throw new Error('This snapshot has no retained values');
  if (chunk.data.byteLength !== size * width)
    throw new Error('The tensor value page is incomplete');
  const view = new DataView(chunk.data.buffer, chunk.data.byteOffset, chunk.data.byteLength);
  const {read} = tensorDecoder(snapshot.dtype, pb);
  return Array.from({length: size}, (_, item) => {
    const index = start + BigInt(item), coordinates = tensorCoordinates(shape, index);
    const value = read(view, item * width);
    return {index, coordinates, value, dequantized: dequantizeTensorValue(value, coordinates, snapshot.quantization)};
  });
}
