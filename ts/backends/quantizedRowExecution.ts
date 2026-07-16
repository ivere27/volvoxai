/* Opt-in fixed-shape W8A8 autoregressive row execution.
 *
 * A fixed-shape decoder keeps [1,S,D] activations allocated across execute()
 * calls. After one full seed pass, a decode step only needs to overwrite row
 * `position`; causal QSDPA reads the retained K/V prefix and cross QSDPA reads
 * the retained full encoder K/V. These helpers deliberately accept only that
 * B=1 contract and fail before any operator writes output.
 */

const SUPPORTED_OPS = new Set([
  'QEmbedding', 'QLinear', 'QMatMul', 'QGemm', 'QAdd', 'QLayerNorm',
  'QGELU', 'QSiLU', 'QSDPA', 'QArgMax',
]);

function fail(node, message) {
  throw new Error(`W8A8 incremental row node ${String(node?.id ?? '<unnamed>')} ${message}`);
}

function typedStorage(tensor, node, label) {
  if (!tensor || !ArrayBuffer.isView(tensor.buffer) || tensor.buffer instanceof DataView) {
    fail(node, `${label} requires typed storage.`);
  }
  return tensor.buffer;
}

function cloneTensorWithView(tensor, shape, buffer) {
  return Object.assign(Object.create(Object.getPrototypeOf(tensor)), tensor, {
    shape,
    sizeBytes: buffer.byteLength,
    buffer,
  });
}

function sequenceRow(tensor, position, sequenceLength, node, label) {
  const storage = typedStorage(tensor, node, label);
  if (!Array.isArray(tensor.shape) || tensor.shape.length < 2 || tensor.shape[0] !== 1 ||
      tensor.shape[1] !== sequenceLength) {
    fail(node, `${label} must have B=1 fixed-sequence shape [1,${sequenceLength},...].`);
  }
  const width = tensor.shape.slice(2).reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(width) || width <= 0 || storage.length !== sequenceLength * width) {
    fail(node, `${label} has incompatible contiguous row storage.`);
  }
  const start = position * width;
  const view = storage.subarray(start, start + width);
  return cloneTensorWithView(tensor, [1, 1, ...tensor.shape.slice(2)], view);
}

function sequencePrefix(tensor, prefixLength, sequenceLength, node, label) {
  const storage = typedStorage(tensor, node, label);
  if (!Array.isArray(tensor.shape) || tensor.shape.length < 2 || tensor.shape[0] !== 1 ||
      tensor.shape[1] !== sequenceLength) {
    fail(node, `${label} must have B=1 fixed-sequence shape [1,${sequenceLength},...].`);
  }
  const width = tensor.shape.slice(2).reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(width) || width <= 0 || storage.length !== sequenceLength * width) {
    fail(node, `${label} has incompatible contiguous prefix storage.`);
  }
  const view = storage.subarray(0, prefixLength * width);
  return cloneTensorWithView(tensor, [1, prefixLength, ...tensor.shape.slice(2)], view);
}

function attentionMask(mask, { position, queryLength, keyLength, prefix }, node) {
  if (!mask) return null;
  const storage = typedStorage(mask, node, 'mask');
  const visibleKeys = prefix ? position + 1 : keyLength;
  if (mask.dtype !== 'int32') fail(node, 'mask must use I32 storage.');
  if (mask.shape.length === 1 && mask.shape[0] === keyLength && storage.length === keyLength) {
    const view = prefix ? storage.subarray(0, visibleKeys) : storage;
    return cloneTensorWithView(mask, [visibleKeys], view);
  }
  // Preserve [B,K] precedence when B and Q are both one.
  if (mask.shape.length === 2 && mask.shape[0] === 1 && mask.shape[1] === keyLength &&
      storage.length === keyLength) {
    const view = prefix ? storage.subarray(0, visibleKeys) : storage;
    return cloneTensorWithView(mask, [1, visibleKeys], view);
  }
  // Query-specific layouts remain safe for B=1 by selecting the current mask
  // row before presenting the local Q=1 call to the operator.
  if (mask.shape.length === 2 && mask.shape[0] === queryLength && mask.shape[1] === keyLength &&
      storage.length === queryLength * keyLength) {
    const start = position * keyLength;
    const view = storage.subarray(start, start + visibleKeys);
    return cloneTensorWithView(mask, [1, visibleKeys], view);
  }
  if (mask.shape.length === 3 && mask.shape[0] === 1 && mask.shape[1] === queryLength &&
      mask.shape[2] === keyLength && storage.length === queryLength * keyLength) {
    const start = position * keyLength;
    const view = storage.subarray(start, start + visibleKeys);
    return cloneTensorWithView(mask, [1, 1, visibleKeys], view);
  }
  fail(node, 'mask must use [K], [1,K], [Q,K], or [1,Q,K] storage.');
}

function outputTensor(node) {
  return node.outputs?.out || Object.values(node.outputs || {})[0];
}

function cloneNode(node, inputs, outputs, params = node.params) {
  return { ...node, inputs, outputs, params };
}

export function incrementalRowPosition(options, selectedNodes, cacheWasValid) {
  if (!Object.prototype.hasOwnProperty.call(options || {}, 'incrementalRowPosition')) return null;
  const position = options.incrementalRowPosition;
  if (options?.incremental !== true || options.incrementalReset === true || cacheWasValid !== true ||
      !(selectedNodes instanceof Set)) {
    throw new Error('W8A8 incrementalRowPosition requires a valid incremental seed pass and a non-reset dependency selection.');
  }
  if (!Number.isInteger(position) || position < 1) {
    throw new Error('W8A8 incrementalRowPosition must be an integer >= 1 after the full seed pass.');
  }
  return position;
}

export function quantizedRowNode(node, position) {
  if (!SUPPORTED_OPS.has(node?.opType)) {
    fail(node, `uses unsupported op '${String(node?.opType)}'.`);
  }
  const output = outputTensor(node);
  const sequenceLength = output?.shape?.[1];
  if (!Number.isInteger(sequenceLength) || sequenceLength <= 1 || position >= sequenceLength ||
      output?.shape?.[0] !== 1) {
    fail(node, `position ${position} is outside a B=1 fixed sequence output.`);
  }
  const rowOutput = sequenceRow(output, position, sequenceLength, node, 'output');
  const outputs = Object.fromEntries(Object.entries(node.outputs || {}).map(([name, tensor]) =>
    [name, tensor === output ? rowOutput : tensor]));

  if (node.opType === 'QEmbedding') {
    const input = sequenceRow(node.inputs?.input, position, sequenceLength, node, 'input IDs');
    return cloneNode(node, { ...node.inputs, input }, outputs);
  }

  if (['QLinear', 'QMatMul', 'QGemm'].includes(node.opType)) {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'a';
    const input = sequenceRow(node.inputs?.[inputName], position, sequenceLength, node, 'input');
    return cloneNode(node, { ...node.inputs, [inputName]: input }, outputs);
  }

  if (node.opType === 'QAdd') {
    const aName = node.inputs?.a ? 'a' : node.inputs?.input ? 'input' : 'x';
    const bName = node.inputs?.b ? 'b' : 'y';
    const a = sequenceRow(node.inputs?.[aName], position, sequenceLength, node, 'input a');
    const b = sequenceRow(node.inputs?.[bName], position, sequenceLength, node, 'input b');
    return cloneNode(node, { ...node.inputs, [aName]: a, [bName]: b }, outputs);
  }

  if (node.opType === 'QLayerNorm') {
    const input = sequenceRow(node.inputs?.input, position, sequenceLength, node, 'input');
    return cloneNode(node, { ...node.inputs, input }, outputs);
  }

  if (node.opType === 'QGELU' || node.opType === 'QSiLU') {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'data';
    const input = sequenceRow(node.inputs?.[inputName], position, sequenceLength, node, 'input');
    return cloneNode(node, { ...node.inputs, [inputName]: input }, outputs);
  }

  if (node.opType === 'QArgMax') {
    const input = node.inputs?.input;
    let axis = node.params?.axis;
    if (axis < 0) axis += input?.shape?.length ?? 0;
    if (!Number.isInteger(axis) || axis <= 1) {
      fail(node, 'QArgMax must reduce a feature axis after the sequence axis.');
    }
    return cloneNode(node, {
      ...node.inputs,
      input: sequenceRow(input, position, sequenceLength, node, 'input'),
    }, outputs);
  }

  const q = node.inputs?.q;
  const k = node.inputs?.k;
  const v = node.inputs?.v;
  const rank = q?.shape?.length;
  const queryLength = q?.shape?.[1];
  const keyLength = k?.shape?.[1];
  if (rank !== 3 || q.shape[0] !== 1 || queryLength !== sequenceLength ||
      k?.shape?.length !== 3 || v?.shape?.length !== 3 || k.shape[0] !== 1 || v.shape[0] !== 1 ||
      v.shape[1] !== keyLength || q.shape[2] !== k.shape[2] || q.shape[2] !== v.shape[2]) {
    fail(node, 'QSDPA requires B=1 rank-3 compatible Q/K/V storage.');
  }
  const causal = node.params?.causal === true;
  if (causal && keyLength !== queryLength) {
    fail(node, 'causal QSDPA requires matching fixed Q/K/V sequence lengths.');
  }
  const rowQ = sequenceRow(q, position, queryLength, node, 'q');
  const rowK = causal ? sequencePrefix(k, position + 1, keyLength, node, 'k') : k;
  const rowV = causal ? sequencePrefix(v, position + 1, keyLength, node, 'v') : v;
  const mask = attentionMask(node.inputs?.mask, {
    position, queryLength, keyLength, prefix: causal,
  }, node);
  const inputs = { ...node.inputs, q: rowQ, k: rowK, v: rowV };
  if (mask) inputs.mask = mask;
  return cloneNode(node, inputs, outputs, { ...node.params, causal: false });
}

export function prepareQuantizedRows(graph, selectedNodes, position) {
  const rows = new Map();
  if (!(selectedNodes instanceof Set) || selectedNodes.size === 0) {
    throw new Error('W8A8 incremental row requires at least one dependency-selected node.');
  }
  for (const nodeIndex of selectedNodes) {
    const node = graph?.nodes?.[nodeIndex];
    if (!node) throw new Error(`W8A8 incremental row selected invalid node index ${nodeIndex}.`);
    rows.set(nodeIndex, quantizedRowNode(node, position));
  }
  return rows;
}
