import {tensorLayout, tensorDecoder, dequantizeTensorValue} from './TensorValues.js';

/** Read one immutable display plane. Transport chunks <=64 KiB; no full tensor copy.
 * Interleaved channels require scanning a larger span; planar channels do not.
 * These are display reductions only, not engine operators or attribution methods.
 */
export async function readSpatialPlane(session, snapshot, pb, descriptor, {index = 0, reduction = 'channel', domain = 'stored'} = {}) {
  const {shape: bigShape, count, width} = tensorLayout(snapshot, pb), {axes, grid} = descriptor;
  const shape = bigShape.map(Number), pixels = grid.height * grid.width;
  if (shape.length !== descriptor.shape.length || shape.some((n, i) => n !== descriptor.shape[i]) || axes.length !== shape.length)
    throw new Error('Captured shape differs from the verified spatial mapping. Use the original graph.');
  if (shape[axes.indexOf('b')] !== 1 || !Number.isSafeInteger(pixels) || pixels < 1 || pixels > 1048576 || count > BigInt(Number.MAX_SAFE_INTEGER))
    throw new Error('Spatial view supports a single image and at most 1,048,576 map cells.');
  const selectorAxis = axes.indexOf(axes.includes('s') ? 's' : 'c'), channels = shape[selectorAxis];
  const expectedAxes = ['b', axes.includes('s') ? 's' : 'c', ...(axes.includes('p') ? ['p'] : ['y', 'x'])];
  if (axes.length !== expectedAxes.length || new Set(axes).size !== axes.length || expectedAxes.some(a => !axes.includes(a)))
    throw new Error('Spatial axes must identify a batch, channel or slot, and the image grid');
  if (!Number.isInteger(index) || index < 0 || index >= channels || !['channel', 'mean-abs'].includes(reduction) ||
      (reduction === 'mean-abs' && axes.includes('s'))) throw new Error('Invalid channel or slot selection');
  if (!['stored', 'dequantized'].includes(domain) || (domain === 'dequantized' && !snapshot.quantization?.perTensor && !snapshot.quantization?.perAxis))
    throw new Error('This snapshot has no affine quantization parameters');
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  const pAxis = axes.indexOf('p'), yAxis = axes.indexOf('y'), xAxis = axes.indexOf('x');
  if (pAxis >= 0 ? shape[pAxis] !== pixels : shape[yAxis] !== grid.height || shape[xAxis] !== grid.width)
    throw new Error('Spatial axes do not match the grid');
  const selected = reduction === 'channel';
  const begin = selected ? index * strides[selectorAxis] : 0;
  const end = selected ? Number(count) - (channels - 1 - index) * strides[selectorAxis] : Number(count);
  if ((end - begin) * width > 64 * 1048576) throw new Error('Spatial view limits each read to 64 MiB. Select a smaller tensor.');
  const values = new Float64Array(pixels), {read} = tensorDecoder(snapshot.dtype, pb);
  const coordinates = new Array(shape.length).fill(0);
  let bytesRead = 0;
  for (let offset = begin * width; offset < end * width;) {
    const limit = Math.min(65536, end * width - offset);
    const chunk = await session.readTensor(snapshot.snapshotId, BigInt(offset), limit);
    if (chunk.status !== pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE || chunk.data.byteLength !== limit)
      throw new Error('The spatial tensor read is incomplete or unavailable');
    const view = new DataView(chunk.data.buffer, chunk.data.byteOffset, chunk.data.byteLength);
    for (let byte = 0; byte < limit; byte += width) {
      const element = (offset + byte) / width;
      if (selected && Math.floor(element / strides[selectorAxis]) % channels !== index) continue;
      for (let a = 0; a < shape.length; a++) coordinates[a] = Math.floor(element / strides[a]) % shape[a];
      const pixel = pAxis >= 0 ? coordinates[pAxis] : coordinates[yAxis] * grid.width + coordinates[xAxis];
      const raw = read(view, byte), value = domain === 'dequantized' ? dequantizeTensorValue(raw, coordinates, snapshot.quantization) : raw;
      if (selected) values[pixel] = value; else values[pixel] += Math.abs(value) / channels;
    }
    offset += limit; bytesRead += limit;
  }
  let min = Infinity, max = -Infinity, nonfinite = 0;
  for (const value of values) {
    if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); } else nonfinite++;
  }
  return {values, min: min === Infinity ? null : min, max: max === -Infinity ? null : max,
    nonfinite, bytesRead, descriptor, index, reduction, domain};
}

/** Pixel centers use the same half-pixel resize convention as receipt preprocessing. */
export function spatialCell(descriptor, x, y, imageWidth, imageHeight) {
  const g = descriptor.grid;
  const inputX = (x + 0.5) * descriptor.inputWidth / imageWidth - 0.5;
  const inputY = (y + 0.5) * descriptor.inputHeight / imageHeight - 0.5;
  return {x: Math.max(0, Math.min(g.width - 1, Math.round((inputX - g.originX) / g.stepX))),
    y: Math.max(0, Math.min(g.height - 1, Math.round((inputY - g.originY) / g.stepY)))};
}

export function spatialCoordinates(plane, x, y) {
  return plane.descriptor.axes.map(axis => ({b: 0, c: plane.index, s: plane.index, y, x, p: y * plane.descriptor.grid.width + x})[axis]);
}

/** Dark blue → cyan → yellow. Nonfinite cells are magenta, never silently zero. */
export function spatialColor(value, min, max) {
  if (!Number.isFinite(value) || min === null || max === null) return [230, 60, 210];
  const t = max === min ? 0.5 : Math.max(0, Math.min(1, (value - min) / (max - min)));
  const stops = [[22, 34, 88], [28, 177, 191], [255, 222, 65]], segment = t < 0.5 ? 0 : 1, f = t * 2 - segment;
  return stops[segment].map((a, i) => Math.round(a + (stops[segment + 1][i] - a) * f));
}
