/* Opt-in fixed-shape autoregressive row execution.
 *
 * A fixed-shape decoder keeps [1,S,D] activations allocated across execute()
 * calls. After one full seed pass, a decode step only needs to overwrite row
 * `position`; causal attention reads the retained K/V prefix and cross
 * attention reads the retained full encoder K/V. These helpers deliberately
 * accept only one logical sequence with contiguous rows and fail before any
 * operator writes output.
 */

const SUPPORTED_OPS = new Set([
  'Embedding', 'Linear', 'MatMul', 'Gemm', 'BatchMatMul',
  'Add', 'Mul', 'LayerNorm', 'Reshape',
  'GELU', 'SiLU', 'CrossSDPA',
  'QEmbedding', 'QLinear', 'QMatMul', 'QGemm', 'QBatchMatMul',
  'QAdd', 'QLayerNorm',
  'QGELU', 'QSiLU', 'QSDPA', 'QArgMax',
  // Quantize/Dequantize convert one element to one element with no dependence
  // on any other position, so a row of the output needs exactly the matching
  // row of the input — the same relationship QGELU and QSiLU have. Excluding
  // them disqualified whole graphs for their boundary edges alone: a decoder
  // whose every compute node was row-decodable still fell back to full
  // recompute because it dequantizes its logits on the way out.
  'QuantizeLinear', 'DequantizeLinear',
  'Expand',
]);

// Operators taking one activation under any of these port spellings, whose
// output row depends only on the matching input row.
const ELEMENTWISE_ROW_OPS = new Set([
  'GELU', 'SiLU', 'QGELU', 'QSiLU',
  'QuantizeLinear', 'DequantizeLinear',
]);

const LINEAR_ROW_OPS = new Set([
  'Linear', 'MatMul', 'Gemm', 'QLinear', 'QMatMul', 'QGemm',
]);

const ATTENTION_ROW_OPS = new Set(['CrossSDPA', 'QSDPA']);
const BATCH_MATMUL_ROW_OPS = new Set(['BatchMatMul', 'QBatchMatMul']);

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

function positiveShape(tensor, node, label) {
  const shape = tensor?.shape;
  if (!Array.isArray(shape) || shape.length < 1 ||
      shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)) {
    fail(node, `${label} must have a positive integer shape.`);
  }
  return shape;
}

/* Resolve the physical row order for the three layouts emitted by the tiny
 * receipt decoders. The established [1,S,...] contract remains available to
 * scalar/token outputs, while newly admitted activation transforms are
 * restricted separately to [1,S,D], [S,1,D], or [S,D]. */
function sequenceLayout(tensor, sequenceLength, node, label) {
  const storage = typedStorage(tensor, node, label);
  const shape = positiveShape(tensor, node, label);
  let sequenceAxis = -1;
  if (shape.length >= 2 && shape[0] === 1 && shape[1] === sequenceLength) {
    sequenceAxis = 1;
  } else if (shape.length === 3 && shape[0] === sequenceLength && shape[1] === 1) {
    sequenceAxis = 0;
  } else if (shape.length === 2 && shape[0] === sequenceLength) {
    sequenceAxis = 0;
  } else {
    fail(node, `${label} must use contiguous [1,S,...], [S,1,D], or [S,D] storage with S=${sequenceLength}.`);
  }
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  const width = elements / sequenceLength;
  if (!Number.isSafeInteger(width) || width <= 0 || storage.length !== elements ||
      (tensor.sizeBytes != null && tensor.sizeBytes !== storage.byteLength)) {
    fail(node, `${label} has incompatible contiguous row storage.`);
  }
  return { storage, shape, sequenceAxis, sequenceLength, width };
}

function activationSequenceLayout(tensor, sequenceLength, node, label) {
  const layout = sequenceLayout(tensor, sequenceLength, node, label);
  const shape = layout.shape;
  const exact = (shape.length === 3 && shape[0] === 1 && shape[1] === sequenceLength) ||
    (shape.length === 3 && shape[0] === sequenceLength && shape[1] === 1) ||
    (shape.length === 2 && shape[0] === sequenceLength);
  if (!exact) {
    fail(node, `${label} must use [1,${sequenceLength},D], [${sequenceLength},1,D], or [${sequenceLength},D] storage.`);
  }
  return layout;
}

function inferOutputSequenceLayout(tensor, position, node) {
  const shape = positiveShape(tensor, node, 'output');
  let sequenceLength = null;
  if (shape.length >= 2 && shape[0] === 1 && shape[1] > 1) {
    sequenceLength = shape[1];
  } else if (shape.length === 3 && shape[0] > 1 && shape[1] === 1) {
    sequenceLength = shape[0];
  } else if (shape.length === 2 && shape[0] > 1) {
    sequenceLength = shape[0];
  }
  if (sequenceLength == null || !Number.isInteger(sequenceLength) ||
      position >= sequenceLength) {
    fail(node, `position ${position} is outside a supported fixed-sequence output.`);
  }
  return sequenceLayout(tensor, sequenceLength, node, 'output');
}

function rowFromLayout(tensor, position, layout) {
  const start = position * layout.width;
  const rowShape = [...layout.shape];
  rowShape[layout.sequenceAxis] = 1;
  return cloneTensorWithView(
    tensor, rowShape, layout.storage.subarray(start, start + layout.width),
  );
}

function sequenceRow(tensor, position, sequenceLength, node, label) {
  const layout = sequenceLayout(tensor, sequenceLength, node, label);
  return rowFromLayout(tensor, position, layout);
}

function attentionSequenceLayout(tensor, node, label) {
  const storage = typedStorage(tensor, node, label);
  const shape = positiveShape(tensor, node, label);
  let sequenceLength;
  let width;
  if (shape.length === 3 && shape[0] === 1) {
    sequenceLength = shape[1];
    width = shape[2];
  } else if (shape.length === 2) {
    sequenceLength = shape[0];
    width = shape[1];
  } else {
    fail(node, `${label} must use contiguous [1,S,D] or [S,D] attention storage.`);
  }
  if (storage.length !== sequenceLength * width ||
      (tensor.sizeBytes != null && tensor.sizeBytes !== storage.byteLength)) {
    fail(node, `${label} has incompatible contiguous attention storage.`);
  }
  return { storage, sequenceLength, width };
}

function attentionSequenceView(tensor, start, length, layout) {
  const offset = start * layout.width;
  return cloneTensorWithView(
    tensor, [length, layout.width],
    layout.storage.subarray(offset, offset + length * layout.width),
  );
}

function attentionRow(tensor, position, layout) {
  return attentionSequenceView(tensor, position, 1, layout);
}

function attentionPrefix(tensor, prefixLength, layout) {
  return attentionSequenceView(tensor, 0, prefixLength, layout);
}

function attentionFull(tensor, layout) {
  return attentionSequenceView(tensor, 0, layout.sequenceLength, layout);
}

function strictActivationRow(tensor, position, outputLayout, node, label) {
  const layout = activationSequenceLayout(
    tensor, outputLayout.sequenceLength, node, label,
  );
  return { layout, tensor: rowFromLayout(tensor, position, layout) };
}

function invariantInput(node, tensor, label, context) {
  if (tensor?.isWeight === true) return;
  if (context?.allowUnprovenInvariantInputs === true) return;
  if (!(context?.dirtyTensorNames instanceof Set)) {
    fail(node, `${label} requires dependency context proving that it is invariant.`);
  }
  if (!tensor?.name || context.dirtyTensorNames.has(tensor.name)) {
    fail(node, `${label} must remain invariant during incremental row execution.`);
  }
}

/* Select the current sequence row when an Add operand varies along the output
 * sequence axis, while retaining operands that safely broadcast over that
 * axis. Right-aligned broadcasting matters here: [D] is a feature vector, not
 * a sequence of length D, even when D happens to equal S. */
function broadcastRowOperand(tensor, position, outputLayout, node, label, opType) {
  const storage = typedStorage(tensor, node, label);
  const shape = tensor?.shape;
  const outputShape = outputLayout.shape;
  if (!Array.isArray(shape) || shape.length < 1 || shape.length > outputShape.length ||
      shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)) {
    fail(node, `${label} must have a positive shape no wider than the ${opType} output rank.`);
  }
  const offset = outputShape.length - shape.length;
  for (let outputAxis = 0; outputAxis < outputShape.length; outputAxis++) {
    const inputAxis = outputAxis - offset;
    const inputDimension = inputAxis < 0 ? 1 : shape[inputAxis];
    if (inputDimension !== 1 && inputDimension !== outputShape[outputAxis]) {
      fail(node, `${label} does not broadcast to the fixed-sequence ${opType} output.`);
    }
  }
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0 || storage.length !== elements) {
    fail(node, `${label} has incompatible broadcast storage.`);
  }

  const inputSequenceAxis = outputLayout.sequenceAxis - offset;
  if (inputSequenceAxis < 0 || inputSequenceAxis >= shape.length ||
      shape[inputSequenceAxis] === 1) return tensor;
  const sequenceLength = outputLayout.sequenceLength;
  if (shape[inputSequenceAxis] !== sequenceLength) {
    fail(node, `${label} has an incompatible sequence dimension.`);
  }
  const leading = shape.slice(0, inputSequenceAxis)
    .reduce((product, dimension) => product * dimension, 1);
  const width = shape.slice(inputSequenceAxis + 1)
    .reduce((product, dimension) => product * dimension, 1);
  if (leading !== 1 || !Number.isSafeInteger(width) || width <= 0 ||
      storage.length !== sequenceLength * width) {
    fail(node, `${label} does not have contiguous sequence-row storage.`);
  }
  const start = position * width;
  const rowShape = [...shape];
  rowShape[inputSequenceAxis] = 1;
  return cloneTensorWithView(tensor, rowShape, storage.subarray(start, start + width));
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

export function quantizedRowNode(node, position, context = {}) {
  if (!SUPPORTED_OPS.has(node?.opType)) {
    fail(node, `uses unsupported op '${String(node?.opType)}'.`);
  }
  const output = outputTensor(node);
  const outputLayout = inferOutputSequenceLayout(output, position, node);
  const sequenceLength = outputLayout.sequenceLength;
  const replaceOutput = (rowOutput) => Object.fromEntries(
    Object.entries(node.outputs || {}).map(([name, tensor]) =>
      [name, tensor === output ? rowOutput : tensor]),
  );
  const rowOutput = rowFromLayout(output, position, outputLayout);
  const outputs = replaceOutput(rowOutput);

  if (node.opType === 'Embedding' || node.opType === 'QEmbedding') {
    const input = sequenceRow(node.inputs?.input, position, sequenceLength, node, 'input IDs');
    return cloneNode(node, { ...node.inputs, input }, outputs);
  }

  if (node.opType === 'Reshape') {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'data';
    const input = node.inputs?.[inputName];
    const inputRow = strictActivationRow(
      input, position, outputLayout, node, 'input',
    );
    const strictOutputLayout = activationSequenceLayout(
      output, sequenceLength, node, 'output',
    );
    if (input?.dtype !== output?.dtype ||
        inputRow.layout.width !== strictOutputLayout.width) {
      fail(node, 'Reshape must preserve the dtype and contiguous logical row width.');
    }
    return cloneNode(
      node,
      { ...node.inputs, [inputName]: inputRow.tensor },
      replaceOutput(rowFromLayout(output, position, strictOutputLayout)),
    );
  }

  if (BATCH_MATMUL_ROW_OPS.has(node.opType)) {
    const a = node.inputs?.a;
    const b = node.inputs?.b;
    invariantInput(node, b, 'input b', context);
    const aRow = strictActivationRow(a, position, outputLayout, node, 'input a');
    const strictOutputLayout = activationSequenceLayout(
      output, sequenceLength, node, 'output',
    );
    return cloneNode(
      node,
      { ...node.inputs, a: aRow.tensor },
      replaceOutput(rowFromLayout(output, position, strictOutputLayout)),
    );
  }

  if (LINEAR_ROW_OPS.has(node.opType)) {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'a';
    const input = sequenceRow(node.inputs?.[inputName], position, sequenceLength, node, 'input');
    return cloneNode(node, { ...node.inputs, [inputName]: input }, outputs);
  }

  if (node.opType === 'Add' || node.opType === 'Mul' || node.opType === 'QAdd') {
    const aName = node.inputs?.a ? 'a' : node.inputs?.input ? 'input' : 'x';
    const bName = node.inputs?.b ? 'b' : 'y';
    const binaryOutputLayout = node.opType === 'Mul'
      ? activationSequenceLayout(output, sequenceLength, node, 'output')
      : outputLayout;
    const select = node.opType === 'Add' || node.opType === 'Mul'
      ? (tensor, label) => broadcastRowOperand(
          tensor, position, binaryOutputLayout, node, label, node.opType,
        )
      : (tensor, label) => sequenceRow(tensor, position, sequenceLength, node, label);
    const originalA = node.inputs?.[aName];
    const originalB = node.inputs?.[bName];
    const a = select(originalA, 'input a');
    const b = select(originalB, 'input b');
    if (node.opType === 'Add' || node.opType === 'Mul') {
      if (a === originalA) invariantInput(node, originalA, 'broadcast input a', context);
      if (b === originalB) invariantInput(node, originalB, 'broadcast input b', context);
    }
    const binaryOutputs = node.opType === 'Mul'
      ? replaceOutput(rowFromLayout(output, position, binaryOutputLayout))
      : outputs;
    return cloneNode(node, { ...node.inputs, [aName]: a, [bName]: b }, binaryOutputs);
  }

  if (node.opType === 'Expand') {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'data';
    const originalInput = node.inputs?.[inputName];
    const input = broadcastRowOperand(
      originalInput, position, outputLayout, node, 'input', node.opType,
    );
    if (input === originalInput) {
      invariantInput(node, originalInput, 'broadcast input', context);
    }
    return cloneNode(node, { ...node.inputs, [inputName]: input }, outputs);
  }

  if (node.opType === 'LayerNorm' || node.opType === 'QLayerNorm') {
    const input = sequenceRow(node.inputs?.input, position, sequenceLength, node, 'input');
    return cloneNode(node, { ...node.inputs, input }, outputs);
  }

  if (ELEMENTWISE_ROW_OPS.has(node.opType)) {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'data';
    const input = sequenceRow(node.inputs?.[inputName], position, sequenceLength, node, 'input');
    // Scale and zero point are per-tensor scalars shared by every row, so they
    // pass through untouched alongside the sliced activation.
    return cloneNode(node, { ...node.inputs, [inputName]: input }, outputs);
  }

  if (node.opType === 'QArgMax') {
    const input = node.inputs?.input;
    const inputLayout = sequenceLayout(input, sequenceLength, node, 'input');
    let axis = node.params?.axis;
    if (axis < 0) axis += input?.shape?.length ?? 0;
    if (!Number.isInteger(axis) || axis <= inputLayout.sequenceAxis) {
      fail(node, 'QArgMax must reduce a feature axis strictly after the sequence axis.');
    }
    return cloneNode(node, {
      ...node.inputs,
      input: rowFromLayout(input, position, inputLayout),
    }, outputs);
  }

  if (!ATTENTION_ROW_OPS.has(node.opType)) {
    fail(node, `has no row implementation for '${String(node.opType)}'.`);
  }
  const q = node.inputs?.q;
  const k = node.inputs?.k;
  const v = node.inputs?.v;
  const qLayout = attentionSequenceLayout(q, node, 'q');
  const kLayout = attentionSequenceLayout(k, node, 'k');
  const vLayout = attentionSequenceLayout(v, node, 'v');
  const attentionOutputLayout = attentionSequenceLayout(output, node, 'output');
  const queryLength = qLayout.sequenceLength;
  const keyLength = kLayout.sequenceLength;
  if (queryLength !== sequenceLength ||
      attentionOutputLayout.sequenceLength !== sequenceLength ||
      vLayout.sequenceLength !== keyLength ||
      qLayout.width !== kLayout.width || qLayout.width !== vLayout.width ||
      qLayout.width !== attentionOutputLayout.width) {
    fail(node, `${node.opType} requires compatible [1,S,D] or [S,D] Q/K/V/output storage.`);
  }
  const causal = node.params?.causal === true;
  if (causal && keyLength !== queryLength) {
    fail(node, `causal ${node.opType} requires matching fixed Q/K/V sequence lengths.`);
  }
  if (!causal) {
    invariantInput(node, k, 'noncausal input k', context);
    invariantInput(node, v, 'noncausal input v', context);
    if (node.inputs?.mask) {
      invariantInput(node, node.inputs.mask, 'noncausal input mask', context);
    }
  }
  const rowQ = attentionRow(q, position, qLayout);
  const rowK = causal
    ? attentionPrefix(k, position + 1, kLayout)
    : attentionFull(k, kLayout);
  const rowV = causal
    ? attentionPrefix(v, position + 1, vLayout)
    : attentionFull(v, vLayout);
  const mask = attentionMask(node.inputs?.mask, {
    position, queryLength, keyLength, prefix: causal,
  }, node);
  const inputs = { ...node.inputs, q: rowQ, k: rowK, v: rowV };
  if (mask) inputs.mask = mask;
  return cloneNode(
    node,
    inputs,
    replaceOutput(attentionRow(output, position, attentionOutputLayout)),
    { ...node.params, causal: false },
  );
}

export function prepareQuantizedRows(
  graph, selectedNodes, position,
  { changedInputs = [] }: { changedInputs?: string[] } = {},
) {
  const rows = new Map<number, any>();
  if (!(selectedNodes instanceof Set) || selectedNodes.size === 0) {
    throw new Error('W8A8 incremental row requires at least one dependency-selected node.');
  }
  if (!Array.isArray(changedInputs) || changedInputs.some((name) =>
    typeof name !== 'string' || name.length === 0)) {
    throw new Error('W8A8 incremental row changedInputs must contain graph-input names.');
  }
  const dirtyTensorNames = new Set(changedInputs);
  const selected: Array<[number, any]> = [];
  for (const nodeIndex of selectedNodes) {
    const node = graph?.nodes?.[nodeIndex];
    if (!node) throw new Error(`W8A8 incremental row selected invalid node index ${nodeIndex}.`);
    selected.push([nodeIndex, node]);
    for (const tensor of Object.values(node.outputs || {}) as any[]) {
      if (tensor?.name) dirtyTensorNames.add(tensor.name);
    }
  }
  const context = { dirtyTensorNames };
  for (const [nodeIndex, node] of selected) {
    rows.set(nodeIndex, quantizedRowNode(node, position, context));
  }
  return rows;
}
