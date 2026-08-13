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
  /* The rest of the re-views, and the select operators. A decoder carries all
   * of them and rejecting any one sends the whole graph back to full
   * recompute, so the executor has to cover what the attestation proves. */
  'Squeeze', 'Unsqueeze', 'Flatten', 'Identity', 'Transpose',
  'Sub', 'Div', 'Where', 'Equal', 'GreaterOrEqual', 'Clip',
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

const ROW_VIEW_OPS = new Set([
  'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten', 'Identity', 'Transpose',
]);

/* Elementwise over every operand, with scalars passing through. Where's
 * condition and both branches take the same treatment as Clip's bounds. */
const NARY_ELEMENTWISE_ROW_OPS = new Set([
  'Sub', 'Div', 'Where', 'Equal', 'GreaterOrEqual', 'Clip',
]);

const ATTENTION_ROW_OPS = new Set(['CrossSDPA', 'QSDPA']);
const BATCH_MATMUL_ROW_OPS = new Set(['BatchMatMul', 'QBatchMatMul']);

/* The provider advertises fixed-row execution only for this deliberately
 * narrow, whole-domain-proved subset. The lower-level row planner above
 * retains additional expert-only routes, but a provider must never infer
 * graph-wide eligibility from a successful sample shape or from the presence
 * of one supported operator. */
const DOMAIN_ATTESTED_ROW_OPS = new Set([
  'Embedding', 'QEmbedding',
  'Linear', 'MatMul', 'Gemm', 'QLinear', 'QMatMul', 'QGemm',
  'QAdd', 'QSDPA',
  'LayerNorm', 'QLayerNorm',
  'GELU', 'SiLU', 'QGELU', 'QSiLU',
  'QuantizeLinear', 'DequantizeLinear',
  'QArgMax',
  /* Added with the proofs below. Without them an FP32 decoder failed
   * attestation on 100 of its 163 nodes and recomputed its whole prefix every
   * token, at 36.8 ms against the native row path's 1.78 — the same operators
   * the native predicate proves, absent from this one. */
  'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten', 'Identity', 'Transpose',
  'Add', 'Mul', 'Sub', 'Div', 'Where', 'Equal', 'GreaterOrEqual', 'Clip',
  'Expand', 'CrossSDPA', 'BatchMatMul', 'QBatchMatMul',
]);

/* Operators whose output row is the matching row of every full-size operand.
 * A scalar operand is uniform across tokens and stays valid for any row; a
 * partially-broadcasting one is refused rather than indexed. */
const DOMAIN_ELEMENTWISE_ROW_OPS = new Set([
  'Add', 'Mul', 'Sub', 'Div', 'Where', 'Equal', 'GreaterOrEqual', 'Clip',
]);

/* Pure re-views: the contiguous flattening is identical, so a row keeps its
 * offset and width. Transpose qualifies only under permutationPreservesRows. */
const DOMAIN_MOVEMENT_ROW_OPS = new Set([
  'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten', 'Identity', 'Transpose',
]);

const DOMAIN_PRIMARY_INPUT_PORTS = Object.freeze({
  Embedding: ['input'], QEmbedding: ['input'],
  Linear: ['input', 'x', 'a'], MatMul: ['input', 'x', 'a'],
  Gemm: ['input', 'x', 'a'], QLinear: ['input', 'x', 'a'],
  QMatMul: ['input', 'x', 'a'], QGemm: ['input', 'x', 'a'],
  LayerNorm: ['input'], QLayerNorm: ['input'],
  GELU: ['input', 'x', 'data'], SiLU: ['input', 'x', 'data'],
  QGELU: ['input', 'x', 'data'], QSiLU: ['input', 'x', 'data'],
  QuantizeLinear: ['input', 'x', 'data'], DequantizeLinear: ['input', 'x', 'data'],
  QArgMax: ['input'],
  Reshape: ['input', 'x', 'data'], Squeeze: ['input', 'x', 'data'],
  Unsqueeze: ['input', 'x', 'data'], Flatten: ['input', 'x', 'data'],
  Identity: ['input', 'x', 'data'], Transpose: ['input', 'x', 'data'],
  Expand: ['input', 'x', 'data'],
});

function sameSymbolicShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, axis) => dimension === right[axis]);
}

function batchLikeDimension(dimension) {
  return dimension === 1 ||
    (typeof dimension === 'string' && /^(?:b|batch)$/i.test(dimension));
}

/*
 * The token axis.
 *
 * Batch-major `[B,S,...]` is the portable spelling every provider can execute
 * a row of. `[S,D]` and `[S,1,D]` place row `r` at the same offset — `r` times
 * the product of everything after it — and a decoder emits them freely, but
 * this attestation is shared by every provider and WebGPU has no row candidate
 * for them, which js_dynamic_decode_provider pins. So a provider that can
 * execute them says so, and one that cannot is unaffected.
 */
function leadingSequenceAxis(shape, sequenceMajor = false) {
  if (!Array.isArray(shape) || shape.length < 1) return -1;
  let axis = 0;
  while (axis < shape.length - 1 && batchLikeDimension(shape[axis])) axis++;
  if (axis === 0 && !sequenceMajor && !batchLikeDimension(shape[0])) return -1;
  return axis;
}

function sequenceDimension(shape, sequenceMajor = false) {
  if (!Array.isArray(shape) || shape.length < 2) return null;
  const axis = leadingSequenceAxis(shape, sequenceMajor);
  return axis < 0 ? null : shape[axis] ?? null;
}

function rowLocalAxis(shape, sequence, sequenceMajor = false) {
  if (!Array.isArray(shape) || shape.length < 2) return -1;
  const axis = leadingSequenceAxis(shape, sequenceMajor);
  /* A trailing width is not required: a `[B,T]` mask is one element per token,
   * which is row-local with width one. */
  return axis >= 0 && shape[axis] === sequence ? axis : -1;
}

/* A symbol whose declared domain is a single value is that value. Without this
 * a batch axis declared `B: {min: 1, max: 1}` compares unequal to the literal
 * it always is, and `[B,T,D]` and `[T,B,D]` look like different row widths.
 * The native side spells this pinned_extent. */
function pinnedDimension(graph, dimension) {
  if (typeof dimension !== 'string') return dimension;
  const constraint = graph?.dimensions?.[dimension];
  return constraint && constraint.min === constraint.max ? constraint.min : dimension;
}

function pinnedShape(graph, shape) {
  return Array.isArray(shape) ? shape.map((d) => pinnedDimension(graph, d)) : shape;
}

/* Everything after the token axis, which is the row width a row-local operator
 * has to agree on.
 *
 * The width is the *product* of the trailing extents: row r begins at r times
 * that product in either spelling, so a re-view that refactors the feature axis
 * moves nothing. Concrete extents are therefore multiplied out rather than
 * compared factor by factor -- `[T,B,960]` and `[1,T,B,3,320]` are both a row
 * of 960 and used to disagree as "960" against "3x320", which refused every
 * decoder at its projection reshape.
 *
 * Symbols cannot be multiplied, so they are kept as themselves and sorted: a
 * product is commutative, so `M x 320` and `320 x M` are one width, while `M`
 * and `T` stay distinct and match only when they match for every binding. */
function trailingWidth(graph, shape, axis) {
  if (!Array.isArray(shape) || axis < 0 || axis >= shape.length) return null;
  let concrete = 1;
  const symbols: string[] = [];
  for (const dimension of pinnedShape(graph, shape).slice(axis + 1)) {
    if (Number.isSafeInteger(dimension)) {
      if (dimension <= 0) return null;
      concrete *= dimension;
    } else if (typeof dimension === 'string' && dimension.length > 0) {
      symbols.push(dimension);
    } else return null;
  }
  return [...symbols.sort(), String(concrete)].join('x');
}

function scalarShape(shape) {
  return Array.isArray(shape) && shape.every(
    (dimension) => Number.isSafeInteger(dimension) && dimension > 0) &&
    shape.reduce((product, dimension) => product * dimension, 1) === 1;
}

/* A permutation relabels axes only when the axes that are not batch-like keep
 * their relative order; the contiguous flattening is then identical and a row
 * is one contiguous run in both spellings. The same rule as
 * vx_incremental_permutation_preserves_layout. */
function permutationPreservesRows(shape, permutation) {
  if (!Array.isArray(shape) || !Array.isArray(permutation) ||
      permutation.length !== shape.length) return false;
  const seen = new Set();
  for (const axis of permutation) {
    if (!Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length ||
        seen.has(axis)) return false;
    seen.add(axis);
  }
  const significant = permutation.filter((axis) => shape[axis] !== 1);
  for (let index = 1; index < significant.length; index++) {
    if (significant[index] < significant[index - 1]) return false;
  }
  return true;
}

function scalarAffineInput(graph, node, port) {
  const name = node.inputs?.[port];
  if (name === undefined) return port === 'zero_point';
  const shape = graph.tensors[name]?.shape;
  return Array.isArray(shape) && shape.length >= 1 &&
    shape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0) &&
    shape.reduce((product, dimension) => product * dimension, 1) === 1;
}

function linearRowParamsSupported(node) {
  const params = node.params || {};
  const fields = Object.keys(params);
  if (fields.some((field) => field !== 'weight_layout' && field !== 'transB')) return false;
  if (params.weight_layout !== undefined && params.transB !== undefined) return false;
  if (params.weight_layout !== undefined &&
      params.weight_layout !== 'din_dout' && params.weight_layout !== 'dout_din') return false;
  return params.transB === undefined || typeof params.transB === 'boolean';
}

function exactInputNames(node, expected) {
  const actual = Object.keys(node.inputs || {}).sort();
  return actual.length === expected.length &&
    expected.every((name, index) => actual[index] === name);
}

function qAddRowDomainSupported(graph, node, dirtyInputs, sequence, outputName) {
  const params = node.params || {};
  const relu = params.relu ?? 0;
  if (!exactInputNames(node, ['a', 'b']) ||
      Object.keys(params).some((name) => name !== 'relu') ||
      !Number.isInteger(relu) || relu < 0 || relu > 2 ||
      dirtyInputs.length < 1 || dirtyInputs.length > 2) return false;
  const aName = node.inputs.a;
  const bName = node.inputs.b;
  if (dirtyInputs.some((name) => name !== aName && name !== bName)) return false;
  /* Through pinnedShape, like the attention and movement proofs below. A
   * dynamic-shape package spells the batch axis `B` with a declared domain of
   * exactly one, and comparing that symbol against the literal 1 refused every
   * quantized decoder at its first QAdd. */
  const aShape = pinnedShape(graph, graph.tensors[aName]?.shape);
  const bShape = pinnedShape(graph, graph.tensors[bName]?.shape);
  const outputShape = pinnedShape(graph, graph.tensors[outputName]?.shape);
  // QAdd's retained row kernel slices both operands. It deliberately does not
  // claim the broader floating Add broadcasting route. The provider exposes
  // one shared capability bit, so match WebGPU's concrete [1,S,D] candidate
  // instead of advertising CPU/WASM's lower-level rank-two/other-axis routes.
  return Array.isArray(aShape) && aShape.length === 3 &&
    aShape[0] === 1 && aShape[1] === sequence &&
    sameSymbolicShape(aShape, bShape) && sameSymbolicShape(aShape, outputShape);
}

/* The token axis and the feature axis after it.
 *
 * This used to accept only a literal `[1,S,D]`, which is WebGPU's physical row
 * candidate. A provider that can also address the sequence-major spellings now
 * says so through `sequenceMajorRows`, exactly as crossSdpaRowDomainSupported
 * already reads them — so the strict shape is no longer this function's job,
 * it is the caller's capability. WebGPU passes `false` and resolves the same
 * `[1,S,D]`-only set it always did; leadingSequenceAxis refuses a leading
 * non-batch axis without the opt-in.
 *
 * Through pinnedShape for the same reason qAddRowDomainSupported needs it: a
 * dynamic-shape package spells the batch axis as a symbol pinned to one. */
function attentionRowLayout(graph, shape, sequenceMajor = false) {
  const resolved = pinnedShape(graph, shape);
  if (!Array.isArray(resolved) || resolved.length < 2) return null;
  const axis = leadingSequenceAxis(resolved, sequenceMajor);
  /* Exactly one axis after the token axis: heads are folded into the feature
   * width here, and a split `[.., S, H, D]` is a different kernel. */
  if (axis < 0 || axis !== resolved.length - 2) return null;
  return { rank: resolved.length, sequence: resolved[axis], feature: resolved[axis + 1] };
}

function qAttentionParamsSupported(node, feature) {
  const params = node.params || {};
  if (Object.keys(params).some((name) =>
    name !== 'heads' && name !== 'causal' && name !== 'scale')) return false;
  const heads = params.heads;
  if (!Number.isSafeInteger(feature) || feature <= 0 ||
      !Number.isSafeInteger(heads) || heads <= 0 || typeof params.causal !== 'boolean') return false;
  const headDimension = feature / heads;
  if (!Number.isSafeInteger(headDimension) || headDimension <= 0 ||
      feature % 4 !== 0 || headDimension % 4 !== 0 || headDimension > 64) return false;
  if (typeof params.scale !== 'number' || !Number.isFinite(params.scale) ||
      params.scale <= 0) return false;
  const scale = Math.fround(params.scale);
  return Number.isFinite(scale) && scale > 0;
}

function qAttentionMaskSupported(graph, maskName, qLayout, kLayout) {
  if (maskName === undefined) return true;
  const mask = graph.tensors[maskName];
  /* Through pinnedShape, like every other proof here: a `[B,T]` keep mask on a
   * package whose `B` is declared exactly one compared unequal to the literal
   * below and refused the whole decoder. Third occurrence of that one mistake
   * in this file -- the helper is not the hard part, remembering to call it is. */
  const shape = pinnedShape(graph, mask?.shape);
  if (mask?.dtype !== 'int32' || !Array.isArray(shape)) return false;
  const queries = qLayout.sequence;
  const keys = kLayout.sequence;
  return (shape.length === 1 && shape[0] === keys) ||
    (shape.length === 2 && shape[1] === keys &&
      (shape[0] === 1 || shape[0] === queries)) ||
    (shape.length === 3 && shape[0] === 1 &&
      shape[1] === queries && shape[2] === keys);
}

/* Null when the row proof holds, otherwise the clause that failed.
 *
 * This returned one boolean for eleven separate conditions, and the caller
 * printed "has a dirty K/V or a non-row query" for all of them -- which named
 * the wrong one for a decoder whose refusal was actually its head count. */
function qSdpaRowRefusal(graph, node, dirtyInputs, sequence, outputName, sequenceMajor) {
  const hasMask = Object.prototype.hasOwnProperty.call(node.inputs || {}, 'mask');
  if (!exactInputNames(node, hasMask ? ['k', 'mask', 'q', 'v'] : ['k', 'q', 'v'])) {
    return 'does not take exactly q/k/v' + (hasMask ? '/mask' : '');
  }
  const qName = node.inputs.q;
  const kName = node.inputs.k;
  const vName = node.inputs.v;
  const maskName = hasMask ? node.inputs.mask : undefined;
  const output = graph.tensors[outputName];
  const qLayout = attentionRowLayout(graph, graph.tensors[qName]?.shape, sequenceMajor);
  const kLayout = attentionRowLayout(graph, graph.tensors[kName]?.shape, sequenceMajor);
  const vLayout = attentionRowLayout(graph, graph.tensors[vName]?.shape, sequenceMajor);
  const outputLayout = attentionRowLayout(graph, output?.shape, sequenceMajor);
  if (!qLayout) return 'has no row layout for q';
  if (!kLayout) return 'has no row layout for k';
  if (!vLayout) return 'has no row layout for v';
  if (!outputLayout) return 'has no row layout for its output';
  if (qLayout.rank !== kLayout.rank || qLayout.rank !== vLayout.rank ||
      qLayout.rank !== outputLayout.rank) return 'mixes operand ranks';
  if (qLayout.sequence !== sequence || qLayout.sequence !== outputLayout.sequence) {
    return `walks '${qLayout.sequence}' where the graph walks '${sequence}'`;
  }
  if (kLayout.sequence !== vLayout.sequence) return 'has k and v on different axes';
  if (qLayout.feature !== kLayout.feature || qLayout.feature !== vLayout.feature ||
      qLayout.feature !== outputLayout.feature) return 'mixes operand feature widths';
  if (!qAttentionParamsSupported(node, qLayout.feature)) {
    return `has params unsupported at feature width ${qLayout.feature}`;
  }
  if (!qAttentionMaskSupported(graph, maskName, qLayout, kLayout)) {
    return 'has an unsupported mask layout';
  }
  const dirty = new Set(dirtyInputs);
  if (node.params.causal === true) {
    // A causal self-attention row consumes the newly computed Q/K/V row and
    // the retained prefix. The keep mask may be either stepped or invariant.
    if (qLayout.sequence !== kLayout.sequence) return 'has q and k on different axes';
    if (!dirty.has(qName) || !dirty.has(kName) || !dirty.has(vName)) {
      return 'is causal but its q/k/v are not all in the dirty closure';
    }
    return [...dirty].every((name) =>
      name === qName || name === kName || name === vName || name === maskName)
      ? null : 'joins a dirty input that is not one of its own operands';
  }
  // Cross attention may update only Q. Its K/V/mask are read as full retained
  // memory by quantizedRowNode and must therefore be outside the dirty closure.
  if (!dirty.has(qName)) return 'is cross attention whose q is not dirty';
  if (dirty.has(kName) || dirty.has(vName)) {
    return 'is cross attention over a recomputed memory';
  }
  if (maskName !== undefined && dirty.has(maskName)) {
    return 'is cross attention with a recomputed mask';
  }
  return null;
}

/**
 * Conservative symbolic-domain attestation used before a provider context is
 * created. Every dependency-reachable node must have a row-local proof for
 * every admitted sequence extent. Unsupported layout transforms, attention
 * decompositions, broadcasting, and multi-dirty-input joins fail closed.
 */
/* Elementwise over every operand: one output row reads the matching row of each
 * full-size operand, and a scalar operand is uniform so any row is valid. */
function elementwiseRowDomainSupported(graph, node, sequence, outputName, sequenceMajor) {
  const outputShape = graph.tensors[outputName]?.shape;
  if (rowLocalAxis(outputShape, sequence, sequenceMajor) < 0) return false;
  for (const name of Object.values(node.inputs || {}) as any[]) {
    if (typeof name !== 'string') return false;
    const shape = graph.tensors[name]?.shape;
    if (!Array.isArray(shape)) return false;
    if (scalarShape(shape)) continue;
    if (!sameSymbolicShape(shape, outputShape)) return false;
  }
  return true;
}

/* One query row against the whole of K and V. Sound only while K and V are
 * outside the dirty closure, which is what a retained cross-attention memory
 * gives and what qSdpaRowDomainSupported requires for the quantized twin. */
function crossSdpaRowDomainSupported(graph, node, dirty, sequence, outputName, sequenceMajor) {
  if (!exactInputNames(node, ['k', 'mask', 'q', 'v']) &&
      !exactInputNames(node, ['k', 'q', 'v'])) return false;
  const queryName = node.inputs?.q;
  const keyName = node.inputs?.k;
  const valueName = node.inputs?.v;
  const maskName = node.inputs?.mask;
  /* A dirty mask is not disqualifying, and hybrid_qsdpa_dirty_kv_supported does
   * not test one either: a causal self-attention's key mask is produced from
   * the same token stream, so it is dirty by construction, and row `r` reads
   * only its own prefix of it. */
  void maskName;
  const queryShape = pinnedShape(graph, graph.tensors[queryName]?.shape);
  const keyShape = pinnedShape(graph, graph.tensors[keyName]?.shape);
  const outputShape = pinnedShape(graph, graph.tensors[outputName]?.shape);
  if (rowLocalAxis(queryShape, sequence, sequenceMajor) < 0 ||
      rowLocalAxis(outputShape, sequence, sequenceMajor) < 0) return false;
  /* Whether a dirty K/V is legal depends on which attention this is, the same
   * split hybrid_qsdpa_dirty_kv_supported makes natively. Causal
   * self-attention deliberately consumes the prefix the dirty row-linear nodes
   * just produced, so its K and V are expected to be dirty. Cross-attention
   * reads a retained memory, and a dirty K or V there would mean the memory is
   * being recomputed, which one row cannot do. */
  if (node.params?.causal === true) {
    const keyAxis = leadingSequenceAxis(keyShape, sequenceMajor);
    if (keyAxis < 0 || keyShape?.[keyAxis] !== sequence) return false;
  } else if (dirty.has(keyName) || dirty.has(valueName)) {
    return false;
  }
  return sameSymbolicShape(queryShape, outputShape);
}

/* Row r of the output reads row r of `a` against the whole of `b`, so this is
 * row-local exactly while `b` carries no token axis. */
function batchMatMulRowDomainSupported(graph, node, dirtyInputs, sequence, outputName, sequenceMajor) {
  if (!exactInputNames(node, ['a', 'b'])) return false;
  const leftName = node.inputs?.a;
  const rightName = node.inputs?.b;
  if (dirtyInputs.length !== 1 || dirtyInputs[0] !== leftName) return false;
  const leftShape = graph.tensors[leftName]?.shape;
  const rightShape = graph.tensors[rightName]?.shape;
  const outputShape = graph.tensors[outputName]?.shape;
  const leftAxis = rowLocalAxis(leftShape, sequence, sequenceMajor);
  const outputAxis = rowLocalAxis(outputShape, sequence, sequenceMajor);
  if (leftAxis < 0 || outputAxis < 0 || leftAxis !== outputAxis) return false;
  if (!Array.isArray(rightShape) ||
      rightShape.some((dimension) => dimension === sequence)) return false;
  return leftShape.slice(0, leftAxis + 1).every(
    (dimension, axis) => dimension === outputShape[axis]);
}

/* Names the first node that fails attestation, the way VOLVOXAI_ROW_DEBUG does
 * natively. Without it a graph-wide false says nothing about which operator to
 * fix, and a decoder that silently recomputes its prefix is invisible. */
function rowDomainRefused(node, reason) {
  const name = node?.id ?? node?.name ?? '?';
  return rowDomainRefusedReason(`${name} op=${node?.opType} ${reason}`);
}

/* The seed refusals below reject the whole graph before any node is examined,
 * so `rowDomainRefused` never runs and the caller sees a bare false. That is
 * the shape of the bug this file keeps producing: a decoder silently recomputes
 * its prefix and nothing says why. Every `return false` here names itself. */
function rowDomainRefusedReason(reason) {
  /* Read through globalThis: this module also runs in a browser, where the
   * published build has no node typings and no `process`. */
  const environment = (globalThis as any)?.process?.env;
  if (environment?.VOLVOXAI_ROW_DEBUG) console.error(`[row] ${reason}`);
  return false;
}

export function incrementalRowDomainSupported(graph, changedInputs, options: any = {}) {
  const sequenceMajor = options?.sequenceMajorRows === true;
  if (!graph || !graph.inputs || !graph.tensors || !Array.isArray(graph.nodes))
    return rowDomainRefusedReason('graph carries no inputs, tensors, or nodes');
  if (!Array.isArray(changedInputs) || changedInputs.length === 0)
    return rowDomainRefusedReason('caller named no changed inputs');
  let sequence = null;
  const dirty = new Set();
  for (const name of changedInputs) {
    const input = graph.inputs[name];
    if (!input)
      return rowDomainRefusedReason(`changed input '${name}' is not a graph input`);
    const candidate = sequenceDimension(input?.shape, sequenceMajor);
    if (candidate == null) {
      return rowDomainRefusedReason(
        `changed input '${name}' shape [${input.shape}] has no token axis`);
    }
    if (sequence !== null && sequence !== candidate) {
      return rowDomainRefusedReason(
        `changed input '${name}' walks '${candidate}' but '${sequence}' was already selected`);
    }
    sequence = candidate;
    dirty.add(name);
  }
  let selected = 0;
  for (const node of graph.nodes) {
    const dirtyInputs = Object.values(node.inputs || {}).filter((name) => dirty.has(name));
    if (dirtyInputs.length === 0) continue;
    selected++;
    if (!DOMAIN_ATTESTED_ROW_OPS.has(node.opType))
      return rowDomainRefused(node, 'is not in the attested domain');
    const outputs = Object.values(node.outputs || {}) as any[];
    if (outputs.length !== 1) return rowDomainRefused(node, 'has multiple outputs');
    const outputName = outputs[0]?.tensor;
    if (!outputName || !graph.tensors[outputName])
      return rowDomainRefused(node, 'has no declared output tensor');
    if (node.opType === 'QAdd') {
      if (!qAddRowDomainSupported(graph, node, dirtyInputs, sequence, outputName))
        return rowDomainRefused(node, 'adds operands that are not both row-local');
      dirty.add(outputName);
      continue;
    }
    if (node.opType === 'QSDPA') {
      const refusal = qSdpaRowRefusal(
        graph, node, dirtyInputs, sequence, outputName, sequenceMajor);
      if (refusal !== null) return rowDomainRefused(node, refusal);
      dirty.add(outputName);
      continue;
    }
    if (DOMAIN_ELEMENTWISE_ROW_OPS.has(node.opType)) {
      if (!elementwiseRowDomainSupported(graph, node, sequence, outputName, sequenceMajor))
        return rowDomainRefused(node, 'is not elementwise over its operands');
      dirty.add(outputName);
      continue;
    }
    if (node.opType === 'CrossSDPA') {
      if (!crossSdpaRowDomainSupported(graph, node, dirty, sequence, outputName, sequenceMajor))
        return rowDomainRefused(node, 'has a dirty K/V or a non-row query');
      dirty.add(outputName);
      continue;
    }
    /* Through the set, which has always held both spellings: the proof reads
     * only shapes, so the quantized twin needs the same one and dispatching on
     * a bare string left QBatchMatMul unreachable while SUPPORTED_OPS could
     * already execute it. */
    if (BATCH_MATMUL_ROW_OPS.has(node.opType)) {
      if (!batchMatMulRowDomainSupported(graph, node, dirtyInputs, sequence, outputName, sequenceMajor))
        return rowDomainRefused(node, 'has a token axis on its right operand');
      dirty.add(outputName);
      continue;
    }
    if (dirtyInputs.length !== 1)
      return rowDomainRefused(node, 'joins several dirty inputs');
    const primaryPorts = DOMAIN_PRIMARY_INPUT_PORTS[node.opType] || [];
    const primaryName = primaryPorts.map((port) => node.inputs?.[port]).find(Boolean);
    if (!primaryName || primaryName !== dirtyInputs[0])
      return rowDomainRefused(node, 'takes its dirty input on a non-primary port');
    const inputShape = pinnedShape(graph, graph.tensors[primaryName]?.shape);
    const outputShape = pinnedShape(graph, graph.tensors[outputName]?.shape);
    const inputSequenceAxis = rowLocalAxis(inputShape, sequence, sequenceMajor);
    const outputSequenceAxis = rowLocalAxis(outputShape, sequence, sequenceMajor);
    if (inputSequenceAxis < 0 || outputSequenceAxis < 0)
      return rowDomainRefused(node, 'has no token axis on its input or output');

    if (node.opType === 'Embedding' || node.opType === 'QEmbedding') {
      if (outputShape.length !== inputShape.length + 1 ||
          !inputShape.every((dimension, axis) => dimension === outputShape[axis]))
        return rowDomainRefused(node, 'does not append a feature axis');
    } else if (LINEAR_ROW_OPS.has(node.opType)) {
      if (!linearRowParamsSupported(node))
        return rowDomainRefused(node, 'has unsupported dense params');
      if (outputShape.length !== inputShape.length ||
          !inputShape.slice(0, -1).every((dimension, axis) => dimension === outputShape[axis])) {
        return rowDomainRefused(node, 'changes a leading axis');
      }
    } else if (node.opType === 'QArgMax') {
      let axis = node.params?.axis;
      if (!Number.isSafeInteger(axis))
        return rowDomainRefused(node, 'has no integer axis');
      if (axis < 0) axis += inputShape.length;
      if (axis <= inputSequenceAxis || axis >= inputShape.length ||
          outputShape.length !== inputShape.length - 1 ||
          !outputShape.every((dimension, outputAxis) =>
            dimension === inputShape[outputAxis < axis ? outputAxis : outputAxis + 1]))
        return rowDomainRefused(node, 'reduces across or below the token axis');
    } else if (node.opType === 'QuantizeLinear' || node.opType === 'DequantizeLinear') {
      const quantizedName = node.opType === 'QuantizeLinear' ? outputName : primaryName;
      if (graph.tensors[quantizedName]?.quantization?.scheme !== 'per_tensor' ||
          !scalarAffineInput(graph, node, 'scale') ||
          !scalarAffineInput(graph, node, 'zero_point') ||
          !sameSymbolicShape(inputShape, outputShape))
        return rowDomainRefused(node, 'is not a shape-preserving per-tensor affine');
    } else if (DOMAIN_MOVEMENT_ROW_OPS.has(node.opType)) {
      /* A re-view keeps row r at r * width, so the token axis has to survive
       * and the width after it has to be unchanged. */
      if (trailingWidth(graph, inputShape, inputSequenceAxis) === null ||
          trailingWidth(graph, inputShape, inputSequenceAxis) !==
            trailingWidth(graph, outputShape, outputSequenceAxis))
        return rowDomainRefused(node, 'changes the row width');
      if (node.opType === 'Transpose' &&
          !permutationPreservesRows(inputShape, node.params?.perm))
        return rowDomainRefused(node, 'permutes non-unit axes');
    } else if (node.opType === 'Expand') {
      /* Widening leading axes keeps the row where it is; broadcasting the
       * token axis itself would make one output row read a different input
       * row, and is refused by the same width comparison. */
      if (trailingWidth(graph, inputShape, inputSequenceAxis) === null ||
          trailingWidth(graph, inputShape, inputSequenceAxis) !==
            trailingWidth(graph, outputShape, outputSequenceAxis))
        return rowDomainRefused(node, 'broadcasts across the token axis');
    } else if (!sameSymbolicShape(inputShape, outputShape)) {
      return rowDomainRefused(node, 'changes shape');
    }
    dirty.add(outputName);
  }
  if (selected === 0)
    return rowDomainRefusedReason('no node consumes a changed input');
  const environment = (globalThis as any)?.process?.env;
  if (environment?.VOLVOXAI_ROW_DEBUG) {
    console.error(`[row] attested '${sequence}' over ${selected} node(s)`);
  }
  return true;
}

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

/* Resolve the physical row order for the three supported sequence layouts.
 * The established [1,S,...] contract remains available to
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

  /* Every re-view is the same operation on a row: the contiguous flattening is
   * identical, so row `r` is one run of the same width in both spellings. The
   * attestation proved the width; this asserts it again on the physical
   * tensors, because the executor must never rely on a proof it cannot see. */
  if (ROW_VIEW_OPS.has(node.opType)) {
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
      fail(node, `${node.opType} must preserve the dtype and contiguous logical row width.`);
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

  if (NARY_ELEMENTWISE_ROW_OPS.has(node.opType)) {
    const inputs = { ...node.inputs };
    for (const [port, tensor] of Object.entries(node.inputs || {})) {
      const original = tensor as any;
      const selected = broadcastRowOperand(
        original, position, outputLayout, node, `input ${port}`, node.opType,
      );
      if (selected === original) invariantInput(node, original, `input ${port}`, context);
      inputs[port] = selected;
    }
    return cloneNode(node, inputs, outputs);
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
