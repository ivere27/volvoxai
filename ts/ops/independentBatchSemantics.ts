import type {
  Graph,
  NodeDescriptor,
  TensorDescriptor,
} from '../core/Graph.js';
import type { Model } from '../core/Model.js';

type LogicalShape = readonly (number | string)[];

export interface IndependentBatchSemanticsEvidence {
  readonly protocol: 'typed-independent-batch-proof/v1';
  readonly supported: boolean;
  readonly batchSymbol: string | null;
  readonly coveredNodes: number;
  readonly graphFingerprint: string;
  readonly reason: string | null;
  readonly failedNode: string | null;
  readonly failedTensor: string | null;
}

function freezeEvidence(
  snapshot: Model,
  values: Omit<IndependentBatchSemanticsEvidence, 'protocol' | 'graphFingerprint'>,
): Readonly<IndependentBatchSemanticsEvidence> {
  return Object.freeze({
    protocol: 'typed-independent-batch-proof/v1' as const,
    graphFingerprint: snapshot.definitionFingerprint,
    ...values,
  });
}

function hasOneLeadingBatchSymbol(shape: LogicalShape, symbol: string): boolean {
  return shape[0] === symbol && shape.indexOf(symbol, 1) === -1;
}

function batchAxis(shape: LogicalShape, symbol: string): number | null {
  const axis = shape.indexOf(symbol);
  return axis < 0 ? null : axis;
}

function sameLogicalShape(left: LogicalShape, right: LogicalShape): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function hasExactKeys(value: Readonly<Record<string, unknown>>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((name, index) => name === wanted[index]);
}

function integerArray(value: unknown): readonly number[] | null {
  return Array.isArray(value) && value.every((item) => Number.isSafeInteger(item))
    ? value as readonly number[]
    : null;
}

function logicalShapeArray(value: unknown): LogicalShape | null {
  return Array.isArray(value) && value.every((item) =>
    (typeof item === 'string' && item.length > 0) ||
      (Number.isSafeInteger(item) && (item as number) > 0))
    ? value as LogicalShape
    : null;
}

function normalizedAxis(value: unknown, rank: number): number | null {
  if (!Number.isSafeInteger(value) || rank <= 0) return null;
  const raw = value as number;
  const axis = raw < 0 ? raw + rank : raw;
  return axis >= 0 && axis < rank ? axis : null;
}

function exactPorts(node: NodeDescriptor, names: readonly string[]): boolean {
  return hasExactKeys(node.inputs, names);
}

function singleOutput(node: NodeDescriptor): string | null {
  const outputs = Object.values(node.outputs);
  return outputs.length === 1 ? outputs[0].tensor : null;
}

function tensorFor(graph: Graph, name: string | undefined): TensorDescriptor | null {
  return name === undefined ? null : graph.tensors[name] ?? null;
}

function isFixedTensor(graph: Graph, name: string | undefined): boolean {
  return tensorFor(graph, name)?.kind === 'weight';
}

function safeBatchedQuantization(tensor: TensorDescriptor, symbol: string): boolean {
  return tensor.quantization?.scheme !== 'per_axis' ||
    tensor.quantization.axis !== batchAxis(tensor.shape, symbol);
}

function isPerTensorQuantized(tensor: TensorDescriptor | null): boolean {
  return tensor?.quantization?.scheme === 'per_tensor';
}

function oneOutput(
  graph: Graph,
  node: NodeDescriptor,
): TensorDescriptor | null {
  const name = singleOutput(node);
  return tensorFor(graph, name ?? undefined);
}

function preservesBatchAxisAndShape(
  graph: Graph,
  node: NodeDescriptor,
  symbol: string,
): boolean {
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  return input !== null && output !== null &&
    batchAxis(input.shape, symbol) !== null &&
    batchAxis(input.shape, symbol) === batchAxis(output.shape, symbol) &&
    sameLogicalShape(input.shape, output.shape);
}

function alignedAxis(axis: number, inputRank: number, outputRank: number): number | null {
  const aligned = axis + outputRank - inputRank;
  return aligned >= 0 && aligned < outputRank ? aligned : null;
}

function safeBroadcastNode(
  graph: Graph,
  node: NodeDescriptor,
  symbol: string,
  ports: readonly string[],
): boolean {
  if (!exactPorts(node, ports) || !hasExactKeys(node.params, [])) return false;
  const output = oneOutput(graph, node);
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!output || outputAxis === null) return false;
  let batched = 0;
  for (const port of ports) {
    const tensor = tensorFor(graph, node.inputs[port]);
    if (!tensor) return false;
    if (tensor.kind === 'weight') continue;
    const inputAxis = batchAxis(tensor.shape, symbol);
    if (inputAxis === null ||
        alignedAxis(inputAxis, tensor.shape.length, output.shape.length) !== outputAxis) {
      return false;
    }
    batched++;
  }
  return batched > 0;
}

type ShapeMonomial = Readonly<{
  constant: bigint;
  symbols: ReadonlyMap<string, number>;
}>;

function shapeMonomial(shape: LogicalShape): ShapeMonomial | null {
  let constant = 1n;
  const symbols = new Map<string, number>();
  for (const dimension of shape) {
    if (typeof dimension === 'number') {
      if (!Number.isSafeInteger(dimension) || dimension <= 0) return null;
      constant *= BigInt(dimension);
    } else {
      symbols.set(dimension, (symbols.get(dimension) ?? 0) + 1);
    }
  }
  return { constant, symbols };
}

function sameShapeMonomial(left: ShapeMonomial | null, right: ShapeMonomial | null): boolean {
  if (!left || !right || left.constant !== right.constant ||
      left.symbols.size !== right.symbols.size) {
    return false;
  }
  for (const [name, exponent] of left.symbols) {
    if (right.symbols.get(name) !== exponent) return false;
  }
  return true;
}

function safeReshape(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['input']) || !hasExactKeys(node.params, ['shape'])) return false;
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const target = logicalShapeArray(node.params.shape);
  if (!input || !output || !target || !sameLogicalShape(target, output.shape)) return false;
  const inputAxis = batchAxis(input.shape, symbol);
  const outputAxis = batchAxis(output.shape, symbol);
  return inputAxis !== null && outputAxis !== null &&
    sameShapeMonomial(
      shapeMonomial(input.shape.slice(0, inputAxis)),
      shapeMonomial(output.shape.slice(0, outputAxis)),
    ) &&
    sameShapeMonomial(
      shapeMonomial(input.shape.slice(inputAxis + 1)),
      shapeMonomial(output.shape.slice(outputAxis + 1)),
    );
}

function safeTranspose(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['input']) || !hasExactKeys(node.params, ['perm'])) return false;
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const permutation = integerArray(node.params.perm);
  const inputAxis = input ? batchAxis(input.shape, symbol) : null;
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!input || !output || !permutation ||
      inputAxis === null || outputAxis === null || permutation.length !== input.shape.length ||
      permutation[outputAxis] !== inputAxis || new Set(permutation).size !== permutation.length ||
      permutation.some((axis) => axis < 0 || axis >= input.shape.length)) {
    return false;
  }
  return output.shape.every((dimension, axis) => dimension === input.shape[permutation[axis]]);
}

function safeSqueeze(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['input']) || !hasExactKeys(node.params, ['axes'])) return false;
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const rawAxes = integerArray(node.params.axes);
  const inputAxis = input ? batchAxis(input.shape, symbol) : null;
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!input || !output || !rawAxes || inputAxis === null || outputAxis === null) return false;
  const axes = rawAxes.map((axis) => normalizedAxis(axis, input.shape.length));
  if (axes.some((axis) => axis === null || axis === inputAxis) ||
      new Set(axes).size !== axes.length ||
      axes.some((axis) => input.shape[axis as number] !== 1)) {
    return false;
  }
  const removed = new Set(axes as number[]);
  return sameLogicalShape(input.shape.filter((_, axis) => !removed.has(axis)), output.shape) &&
    outputAxis === inputAxis - [...removed].filter((axis) => axis < inputAxis).length;
}

function safeUnsqueeze(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['input']) || !hasExactKeys(node.params, ['axes'])) return false;
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const rawAxes = integerArray(node.params.axes);
  const inputAxis = input ? batchAxis(input.shape, symbol) : null;
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!input || !output || !rawAxes || inputAxis === null || outputAxis === null ||
      output.shape.length !== input.shape.length + rawAxes.length) {
    return false;
  }
  const axes = rawAxes.map((axis) => normalizedAxis(axis, output.shape.length));
  if (axes.some((axis) => axis === null || axis === outputAxis) ||
      new Set(axes).size !== axes.length) {
    return false;
  }
  const inserted = new Set(axes as number[]);
  if ([...inserted].some((axis) => output.shape[axis] !== 1)) return false;
  return sameLogicalShape(output.shape.filter((_, axis) => !inserted.has(axis)), input.shape) &&
    outputAxis === inputAxis + [...inserted].filter((axis) => axis < outputAxis).length;
}

function safeSlice(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['input']) ||
      !hasExactKeys(node.params, ['starts', 'ends', 'axes', 'steps'])) {
    return false;
  }
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const starts = integerArray(node.params.starts);
  const ends = integerArray(node.params.ends);
  const axes = integerArray(node.params.axes);
  const steps = integerArray(node.params.steps);
  const inputAxis = input ? batchAxis(input.shape, symbol) : null;
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!input || !output || !starts || !ends || !axes || !steps ||
      inputAxis === null || outputAxis !== inputAxis || starts.length !== ends.length ||
      starts.length !== axes.length || starts.length !== steps.length ||
      steps.some((step) => step !== 1)) {
    return false;
  }
  const normalized = axes.map((axis) => normalizedAxis(axis, input.shape.length));
  return normalized.every((axis) => axis !== null && axis !== inputAxis) &&
    new Set(normalized).size === normalized.length;
}

function safeConcat(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!hasExactKeys(node.params, ['axis'])) return false;
  const output = oneOutput(graph, node);
  const names = Object.values(node.inputs);
  if (!output || names.length < 2) return false;
  const outputAxis = batchAxis(output.shape, symbol);
  const axis = normalizedAxis(node.params.axis, output.shape.length);
  return outputAxis !== null && axis !== null && axis !== outputAxis && names.every((name) => {
    const input = tensorFor(graph, name);
    return input !== null && batchAxis(input.shape, symbol) === outputAxis &&
      input.shape.length === output.shape.length;
  });
}

function safeExpand(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['input']) || !hasExactKeys(node.params, ['shape'])) return false;
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const target = logicalShapeArray(node.params.shape);
  if (!input || !output || !target || !sameLogicalShape(target, output.shape)) return false;
  const outputAxis = batchAxis(output.shape, symbol);
  if (outputAxis === null) return false;
  if (input.kind === 'weight') return true;
  const inputAxis = batchAxis(input.shape, symbol);
  return inputAxis !== null &&
    alignedAxis(inputAxis, input.shape.length, output.shape.length) === outputAxis;
}

function safeGatherOrEmbedding(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  const embedding = node.opType === 'Embedding';
  const expectedPorts = embedding ? ['input', 'weight'] : ['indices', 'input'];
  const expectedParams = embedding ? [] : ['axis'];
  if (!exactPorts(node, expectedPorts) || !hasExactKeys(node.params, expectedParams)) return false;
  const tableName = embedding ? node.inputs.weight : node.inputs.input;
  const indicesName = embedding ? node.inputs.input : node.inputs.indices;
  const table = tensorFor(graph, tableName);
  const indices = tensorFor(graph, indicesName);
  const output = oneOutput(graph, node);
  const axis = embedding ? 0 : normalizedAxis(node.params.axis, table?.shape.length ?? 0);
  const indicesAxis = indices ? batchAxis(indices.shape, symbol) : null;
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!table || !indices || !output || table.kind !== 'weight' || table.shape.length < 2 ||
      axis !== 0 || indicesAxis === null || outputAxis !== indicesAxis) {
    return false;
  }
  return sameLogicalShape([...indices.shape, ...table.shape.slice(1)], output.shape);
}

function safeReduction(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  const argMax = node.opType === 'ArgMax';
  const expected = argMax
    ? ['axis', 'keepdims', 'select_last_index']
    : ['axis', 'keepdims'];
  if (!exactPorts(node, ['input']) || !hasExactKeys(node.params, expected) ||
      typeof node.params.keepdims !== 'boolean' ||
      (argMax && node.params.select_last_index !== 0)) {
    return false;
  }
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const axis = normalizedAxis(node.params.axis, input?.shape.length ?? 0);
  const inputAxis = input ? batchAxis(input.shape, symbol) : null;
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!input || !output || inputAxis === null || outputAxis === null ||
      axis === null || axis === inputAxis) {
    return false;
  }
  const expectedShape = node.params.keepdims
    ? input.shape.map((dimension, index) => index === axis ? 1 : dimension)
    : input.shape.filter((_, index) => index !== axis);
  const expectedBatchAxis = node.params.keepdims || axis > inputAxis
    ? inputAxis
    : inputAxis - 1;
  return outputAxis === expectedBatchAxis && sameLogicalShape(expectedShape, output.shape);
}

function safeSoftmax(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['input']) || !hasExactKeys(node.params, ['axis']) ||
      !preservesBatchAxisAndShape(graph, node, symbol)) {
    return false;
  }
  const input = tensorFor(graph, node.inputs.input)!;
  const axis = normalizedAxis(node.params.axis, input.shape.length);
  return axis !== null && axis !== batchAxis(input.shape, symbol);
}

function safeDense(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  const quantized = node.opType === 'QLinear' || node.opType === 'QGemm';
  const hasBias = node.inputs.bias !== undefined;
  const ports = hasBias ? ['bias', 'input', 'weight'] : ['input', 'weight'];
  const params = node.opType === 'Linear' ? ['weight_layout'] : [];
  if (!exactPorts(node, ports) || !hasExactKeys(node.params, params) ||
      (node.opType === 'Linear' && node.params.weight_layout !== 'din_dout')) {
    return false;
  }
  const input = tensorFor(graph, node.inputs.input);
  const weight = tensorFor(graph, node.inputs.weight);
  const bias = tensorFor(graph, node.inputs.bias);
  const output = oneOutput(graph, node);
  const inputAxis = input ? batchAxis(input.shape, symbol) : null;
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!input || !weight || !output || input.shape.length < 2 ||
      inputAxis === null || inputAxis >= input.shape.length - 1 || outputAxis !== inputAxis ||
      input.shape.length !== output.shape.length ||
      weight.kind !== 'weight' || weight.shape.length !== 2 ||
      (hasBias && (!bias || bias.kind !== 'weight'))) {
    return false;
  }
  if (!quantized) return true;
  return isPerTensorQuantized(input) && isPerTensorQuantized(output) &&
    (weight.quantization?.scheme === 'per_tensor' ||
      (weight.quantization?.scheme === 'per_axis' && weight.quantization.axis === 0));
}

function safeMatMul(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  const hasBias = node.inputs.bias !== undefined;
  const ports = hasBias ? ['bias', 'input', 'weight'] : ['input', 'weight'];
  if (!exactPorts(node, ports) || !hasExactKeys(node.params, [])) return false;
  const activation = tensorFor(graph, node.inputs.input);
  const weight = tensorFor(graph, node.inputs.weight);
  const bias = tensorFor(graph, node.inputs.bias);
  const output = oneOutput(graph, node);
  const activationAxis = activation ? batchAxis(activation.shape, symbol) : null;
  return activation !== null && weight !== null && output !== null &&
    activation.shape.length >= 2 && activationAxis !== null &&
    activationAxis < activation.shape.length - 1 &&
    batchAxis(output.shape, symbol) === activationAxis &&
    activation.shape.length === output.shape.length && weight.kind === 'weight' &&
    weight.shape.length === 2 && (!hasBias || bias?.kind === 'weight');
}

function safeBatchMatMul(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['a', 'b']) || !hasExactKeys(node.params, [])) return false;
  const output = oneOutput(graph, node);
  const outputAxis = output ? batchAxis(output.shape, symbol) : null;
  if (!output || output.shape.length < 3 || outputAxis === null ||
      outputAxis >= output.shape.length - 2) return false;
  let batched = 0;
  for (const port of ['a', 'b'] as const) {
    const tensor = tensorFor(graph, node.inputs[port]);
    if (!tensor || tensor.shape.length < 2) return false;
    if (tensor.kind === 'weight') {
      if (tensor.shape.length > output.shape.length) return false;
      const alignedWeightAxis = outputAxis - (output.shape.length - tensor.shape.length);
      if (alignedWeightAxis >= 0 && tensor.shape[alignedWeightAxis] !== 1) return false;
      continue;
    }
    const inputAxis = batchAxis(tensor.shape, symbol);
    if (inputAxis === null || inputAxis >= tensor.shape.length - 2 ||
        alignedAxis(inputAxis, tensor.shape.length, output.shape.length) !== outputAxis) {
      return false;
    }
    if (node.opType === 'QBatchMatMul' && !isPerTensorQuantized(tensor)) return false;
    batched++;
  }
  return batched > 0 &&
    (node.opType !== 'QBatchMatMul' || isPerTensorQuantized(output));
}

function positiveIntegerPair(value: unknown): boolean {
  const pair = integerArray(value);
  return pair !== null && pair.length === 2 && pair.every((item) => item > 0);
}

function safeConvolution(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  const quantized = node.opType === 'QConv2D';
  const hasBias = node.inputs.bias !== undefined;
  const ports = hasBias ? ['bias', 'input', 'weight'] : ['input', 'weight'];
  const params = ['data_layout', 'dilation', 'groups', 'padding', 'pads', 'stride', 'weight_layout'];
  const pads = integerArray(node.params.pads);
  if (!exactPorts(node, ports) || !hasExactKeys(node.params, params) ||
      node.params.data_layout !== 'NHWC' ||
      node.params.weight_layout !== (quantized ? 'OHWI' : 'HWIO') ||
      !positiveIntegerPair(node.params.dilation) || !positiveIntegerPair(node.params.stride) ||
      !positiveIntegerPair(node.params.padding) || !pads || pads.length !== 4 ||
      pads.some((item) => item < 0) || !Number.isSafeInteger(node.params.groups) ||
      (node.params.groups as number) <= 0) {
    return false;
  }
  const input = tensorFor(graph, node.inputs.input);
  const weight = tensorFor(graph, node.inputs.weight);
  const bias = tensorFor(graph, node.inputs.bias);
  const output = oneOutput(graph, node);
  if (!input || !weight || !output || input.shape.length !== 4 || output.shape.length !== 4 ||
      batchAxis(input.shape, symbol) !== 0 || batchAxis(output.shape, symbol) !== 0 ||
      weight.kind !== 'weight' ||
      weight.shape.length !== 4 || (hasBias && bias?.kind !== 'weight')) {
    return false;
  }
  if (!quantized) return true;
  return isPerTensorQuantized(input) && isPerTensorQuantized(output) &&
    (weight.quantization?.scheme === 'per_tensor' ||
      (weight.quantization?.scheme === 'per_axis' && weight.quantization.axis === 0));
}

function safeNormalization(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  const layer = node.opType === 'LayerNorm';
  const params = layer ? ['d_model', 'eps'] : ['num_groups', 'eps'];
  if (!exactPorts(node, ['bias', 'input', 'weight']) || !hasExactKeys(node.params, params) ||
      typeof node.params.eps !== 'number' || !Number.isFinite(node.params.eps) ||
      node.params.eps <= 0) {
    return false;
  }
  const size = layer ? node.params.d_model : node.params.num_groups;
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const inputAxis = input ? batchAxis(input.shape, symbol) : null;
  return Number.isSafeInteger(size) && (size as number) > 0 && input !== null && output !== null &&
    (layer ? input.shape.length >= 2 : input.shape.length === 4) &&
    inputAxis !== null && (layer ? inputAxis < input.shape.length - 1 : inputAxis === 0) &&
    batchAxis(output.shape, symbol) === inputAxis && sameLogicalShape(input.shape, output.shape) &&
    isFixedTensor(graph, node.inputs.weight) && isFixedTensor(graph, node.inputs.bias);
}

function safeQuantizeBoundary(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  if (!exactPorts(node, ['input', 'scale', 'zero_point']) || !hasExactKeys(node.params, []) ||
      !preservesBatchAxisAndShape(graph, node, symbol) ||
      !isFixedTensor(graph, node.inputs.scale) || !isFixedTensor(graph, node.inputs.zero_point)) {
    return false;
  }
  const input = tensorFor(graph, node.inputs.input);
  const output = oneOutput(graph, node);
  const quantized = node.opType === 'QuantizeLinear' ? output : input;
  return quantized !== null && isPerTensorQuantized(quantized);
}

function safeIndependentBatchNode(graph: Graph, node: NodeDescriptor, symbol: string): boolean {
  switch (node.opType) {
    case 'Identity':
    case 'Sigmoid':
    case 'SiLU':
    case 'Not':
      return exactPorts(node, ['input']) && hasExactKeys(node.params, []) &&
        preservesBatchAxisAndShape(graph, node, symbol);
    case 'GELU':
      return exactPorts(node, ['input']) && hasExactKeys(node.params, ['approximate']) &&
        node.params.approximate === 'none' && preservesBatchAxisAndShape(graph, node, symbol);
    case 'Clip':
      return exactPorts(node, ['input']) && hasExactKeys(node.params, ['min', 'max']) &&
        typeof node.params.min === 'number' && Number.isFinite(node.params.min) &&
        typeof node.params.max === 'number' && Number.isFinite(node.params.max) &&
        node.params.min <= node.params.max && preservesBatchAxisAndShape(graph, node, symbol);
    case 'Cast':
      return exactPorts(node, ['input']) && hasExactKeys(node.params, ['to']) &&
        (node.params.to === 'float32' || node.params.to === 'int32') &&
        preservesBatchAxisAndShape(graph, node, symbol);
    case 'Add':
    case 'Sub':
    case 'Mul':
    case 'Div':
    case 'Equal':
    case 'GreaterOrEqual':
      return safeBroadcastNode(graph, node, symbol, ['a', 'b']);
    case 'Where':
      return safeBroadcastNode(graph, node, symbol, ['a', 'b', 'condition']);
    case 'MatMul':
      return safeMatMul(graph, node, symbol);
    case 'Linear':
    case 'QLinear':
    case 'QGemm':
      return safeDense(graph, node, symbol);
    case 'BatchMatMul':
    case 'QBatchMatMul':
      return safeBatchMatMul(graph, node, symbol);
    case 'Conv2D':
    case 'QConv2D':
      return safeConvolution(graph, node, symbol);
    case 'LayerNorm':
    case 'GroupNorm':
      return safeNormalization(graph, node, symbol);
    case 'Softmax':
      return safeSoftmax(graph, node, symbol);
    case 'ReduceSum':
    case 'ArgMax':
      return safeReduction(graph, node, symbol);
    case 'Reshape':
      return safeReshape(graph, node, symbol);
    case 'Transpose':
      return safeTranspose(graph, node, symbol);
    case 'Squeeze':
      return safeSqueeze(graph, node, symbol);
    case 'Unsqueeze':
      return safeUnsqueeze(graph, node, symbol);
    case 'Slice':
      return safeSlice(graph, node, symbol);
    case 'Concat':
      return safeConcat(graph, node, symbol);
    case 'Expand':
      return safeExpand(graph, node, symbol);
    case 'Gather':
    case 'Embedding':
      return safeGatherOrEmbedding(graph, node, symbol);
    case 'QuantizeLinear':
    case 'DequantizeLinear':
      return safeQuantizeBoundary(graph, node, symbol);
    default:
      return false;
  }
}

function nodeRejectionReason(node: NodeDescriptor): string {
  switch (node.opType) {
    case 'Reshape':
      return 'Reshape does not preserve the row-major products before and after the request axis.';
    case 'Transpose':
      return 'Transpose does not carry the request axis through its declared permutation.';
    case 'Squeeze':
    case 'Unsqueeze':
      return `${node.opType} removes, duplicates, or inconsistently shifts the request axis.`;
    case 'Slice':
      return 'Slice selects or reverses the request axis, or uses an unsupported step.';
    case 'Concat':
      return 'Concat joins the request axis or receives inconsistent request-axis provenance.';
    case 'Expand':
      return 'Expand does not align its input with the output request axis.';
    case 'Gather':
    case 'Embedding':
      return `${node.opType} is not fixed-table axis-0 selection by per-request indices.`;
    case 'ReduceSum':
    case 'ArgMax':
    case 'Softmax':
      return `${node.opType} reduces the request axis or has an unsupported canonical configuration.`;
    case 'BatchMatMul':
    case 'QBatchMatMul':
      return `${node.opType} places the request axis in a matrix dimension or misaligned batch dimension.`;
    case 'Conv2D':
    case 'QConv2D':
    case 'GroupNorm':
      return `${node.opType} does not preserve the canonical axis-0 request layout.`;
    case 'LayerNorm':
    case 'Linear':
    case 'QLinear':
    case 'QGemm':
    case 'MatMul':
      return `${node.opType} consumes the request axis as a feature or uses unsupported fixed parameters.`;
    default:
      return `Operator '${node.opType}' configuration is outside the typed independent-batch proof.`;
  }
}

/**
 * Typed, fail-closed proof that the public leading symbol is one independent
 * request axis. The canonical graph fingerprint binds the evidence to exact
 * operator parameters and descriptors; `failedNode`/`failedTensor` make a
 * refusal auditable without making the proof part of the public graph format.
 */
export function inspectIndependentPublicBatchSemantics(
  snapshot: Model,
): Readonly<IndependentBatchSemanticsEvidence> {
  const symbol = snapshot.inputDescriptors[0]?.shape[0];
  if (typeof symbol !== 'string' ||
      snapshot.inputDescriptors.some((descriptor) => !hasOneLeadingBatchSymbol(
        descriptor.shape,
        symbol,
      )) ||
      snapshot.outputDescriptors.some((descriptor) => !hasOneLeadingBatchSymbol(
        descriptor.shape,
        symbol,
      ))) {
    return freezeEvidence(snapshot, {
      supported: false,
      batchSymbol: typeof symbol === 'string' ? symbol : null,
      coveredNodes: 0,
      reason: 'Every public input and output must contain the same leading batch symbol exactly once.',
      failedNode: null,
      failedTensor: null,
    });
  }

  const dimension = snapshot.graph.dimensions[symbol];
  if (!dimension || dimension.min !== 1 || dimension.multiple_of !== 1 ||
      !Number.isSafeInteger(dimension.max) || dimension.max < 1 ||
      snapshot.graph.nodes.length === 0) {
    return freezeEvidence(snapshot, {
      supported: false,
      batchSymbol: symbol,
      coveredNodes: 0,
      reason: 'The batch symbol must have min=1, multiple_of=1, a finite positive max, and a non-empty graph.',
      failedNode: null,
      failedTensor: null,
    });
  }

  const { graph } = snapshot;
  for (const [name, descriptor] of Object.entries(graph.tensors)) {
    const occurrences = descriptor.shape.filter((value) => value === symbol).length;
    if (descriptor.kind === 'weight') {
      if (occurrences !== 0) {
        return freezeEvidence(snapshot, {
          supported: false,
          batchSymbol: symbol,
          coveredNodes: 0,
          reason: 'Fixed weights cannot contain the public batch symbol.',
          failedNode: null,
          failedTensor: name,
        });
      }
      continue;
    }
    if (occurrences > 1) {
      return freezeEvidence(snapshot, {
        supported: false,
        batchSymbol: symbol,
        coveredNodes: 0,
        reason: 'An execution tensor cannot contain the public batch symbol more than once.',
        failedNode: descriptor.kind === 'value' ? descriptor.producerNodeId : null,
        failedTensor: name,
      });
    }
    if (occurrences === 1 && !safeBatchedQuantization(descriptor, symbol)) {
      return freezeEvidence(snapshot, {
        supported: false,
        batchSymbol: symbol,
        coveredNodes: 0,
        reason: 'Per-axis quantization cannot use the public batch axis.',
        failedNode: descriptor.kind === 'value' ? descriptor.producerNodeId : null,
        failedTensor: name,
      });
    }
  }

  for (let index = 0; index < graph.nodes.length; index++) {
    const node = graph.nodes[index];
    if (!safeIndependentBatchNode(graph, node, symbol)) {
      return freezeEvidence(snapshot, {
        supported: false,
        batchSymbol: symbol,
        coveredNodes: index,
        reason: nodeRejectionReason(node),
        failedNode: node.id,
        failedTensor: singleOutput(node),
      });
    }
  }

  return freezeEvidence(snapshot, {
    supported: true,
    batchSymbol: symbol,
    coveredNodes: graph.nodes.length,
    reason: null,
    failedNode: null,
    failedTensor: null,
  });
}

/** @internal Boolean compatibility seam used by provider/core batch gates. */
export function hasConservativeIndependentPublicBatchSemantics(snapshot: Model): boolean {
  return inspectIndependentPublicBatchSemantics(snapshot).supported;
}
