/* Opt-in fixed-shape autoregressive row execution.
 *
 * A fixed-shape decoder keeps [1,S,D] activations allocated across execute()
 * calls. After one full seed pass, a decode step only needs to overwrite row
 * `position`; causal attention reads the retained K/V prefix and cross
 * attention reads the retained full encoder K/V. These helpers deliberately
 * accept only one logical sequence with contiguous rows and fail before any
 * operator writes output.
 *
 * When the caller owns a `PagedKVCache`, the retained causal K/V is addressed
 * through its page table instead of by the linear `position * width` slice.
 * The contiguous slice is not a second path kept beside the paged one: it is
 * what the paged resolver returns for the identity mapping, which is what a
 * contiguous cache produces. See `kvPageAddressing.ts`.
 */

import { defaultScratchAllocator, pagedSequenceView } from './kvPageAddressing.js';
import type { KVScratchAllocator } from './kvPageAddressing.js';
import {
  batchPrefixView, decodeRowSet, rowSpanGeometry, rowSpanView, singleLaneRowSet,
  stageKeepMask, writeRowIndices,
} from './decodeRowSet.js';
import type { DecodeRowSet, RowSpanGeometry } from './decodeRowSet.js';

/* Used only when a backend supplies no allocator of its own, which is every
 * host backend whose kernels read ordinary typed arrays. */
const sharedScratchAllocator = defaultScratchAllocator();

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

/* An axis that sits before the token axis.
 *
 * `lanes` is the slot count the context declares, so a package that spells its
 * batch axis as the literal it is -- which a scheduler-owned dense batch does --
 * is recognised without the proof guessing. Inferring "any leading integer is a
 * batch" instead would let `[S,D]` read D as a batch and admit graphs whose rows
 * are not rows at all. */
function batchLikeDimension(dimension, lanes = 1) {
  return dimension === 1 || (lanes > 1 && dimension === lanes) ||
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
function leadingSequenceAxis(shape, sequenceMajor = false, lanes = 1) {
  if (!Array.isArray(shape) || shape.length < 1) return -1;
  let axis = 0;
  while (axis < shape.length - 1 && batchLikeDimension(shape[axis], lanes)) axis++;
  if (axis === 0 && !sequenceMajor && !batchLikeDimension(shape[0], lanes)) return -1;
  return axis;
}

function sequenceDimension(shape, sequenceMajor = false, lanes = 1) {
  if (!Array.isArray(shape) || shape.length < 2) return null;
  const axis = leadingSequenceAxis(shape, sequenceMajor, lanes);
  return axis < 0 ? null : shape[axis] ?? null;
}

function rowLocalAxis(shape, sequence, sequenceMajor = false, lanes = 1) {
  if (!Array.isArray(shape) || shape.length < 2) return -1;
  const axis = leadingSequenceAxis(shape, sequenceMajor, lanes);
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

function qAddRowDomainSupported(graph, node, dirtyInputs, sequence, outputName, lanes) {
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
    aShape[0] === lanes && aShape[1] === sequence &&
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
function attentionRowLayout(graph, shape, sequenceMajor = false, lanes = 1) {
  const resolved = pinnedShape(graph, shape);
  if (!Array.isArray(resolved) || resolved.length < 2) return null;
  const axis = leadingSequenceAxis(resolved, sequenceMajor, lanes);
  /* Exactly one axis after the token axis: heads are folded into the feature
   * width here, and a split `[.., S, H, D]` is a different kernel. */
  if (axis < 0 || axis !== resolved.length - 2) return null;
  /* The lane extent, so the proof can require exact B equality across query,
   * key, value, memory, mask and output rather than assuming one lane. */
  const batch = axis === 0 ? 1 : resolved
    .slice(0, axis).reduce((product, dimension) =>
      Number.isSafeInteger(dimension) ? product * dimension : dimension, 1);
  return {
    rank: resolved.length, batch,
    sequence: resolved[axis], feature: resolved[axis + 1],
  };
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
  const lanes = qLayout.batch;
  return (shape.length === 1 && shape[0] === keys) ||
    (shape.length === 2 && shape[1] === keys &&
      (shape[0] === 1 || shape[0] === lanes || shape[0] === queries)) ||
    (shape.length === 3 && (shape[0] === 1 || shape[0] === lanes) &&
      shape[1] === queries && shape[2] === keys);
}

/* Null when the row proof holds, otherwise the clause that failed.
 *
 * This returned one boolean for eleven separate conditions, and the caller
 * printed "has a dirty K/V or a non-row query" for all of them -- which named
 * the wrong one for a decoder whose refusal was actually its head count. */
function qSdpaRowRefusal(graph, node, dirtyInputs, sequence, outputName, sequenceMajor, lanes) {
  const hasMask = Object.prototype.hasOwnProperty.call(node.inputs || {}, 'mask');
  if (!exactInputNames(node, hasMask ? ['k', 'mask', 'q', 'v'] : ['k', 'q', 'v'])) {
    return 'does not take exactly q/k/v' + (hasMask ? '/mask' : '');
  }
  const qName = node.inputs.q;
  const kName = node.inputs.k;
  const vName = node.inputs.v;
  const maskName = hasMask ? node.inputs.mask : undefined;
  const output = graph.tensors[outputName];
  const qLayout = attentionRowLayout(graph, graph.tensors[qName]?.shape, sequenceMajor, lanes);
  const kLayout = attentionRowLayout(graph, graph.tensors[kName]?.shape, sequenceMajor, lanes);
  const vLayout = attentionRowLayout(graph, graph.tensors[vName]?.shape, sequenceMajor, lanes);
  const outputLayout = attentionRowLayout(graph, output?.shape, sequenceMajor, lanes);
  if (!qLayout) return 'has no row layout for q';
  if (!kLayout) return 'has no row layout for k';
  if (!vLayout) return 'has no row layout for v';
  if (!outputLayout) return 'has no row layout for its output';
  if (qLayout.rank !== kLayout.rank || qLayout.rank !== vLayout.rank ||
      qLayout.rank !== outputLayout.rank) return 'mixes operand ranks';
  /* The batched row-decode state contract requires exact B equality: an
   * operand carrying fewer lanes than the step declares would have one lane
   * reading another lane's attention state, and no downstream shape would say
   * so. */
  if (qLayout.batch !== lanes || kLayout.batch !== lanes ||
      vLayout.batch !== lanes || outputLayout.batch !== lanes) {
    return `mixes lane extents ${qLayout.batch}/${kLayout.batch}/${vLayout.batch}/` +
      `${outputLayout.batch} where the context declares ${lanes}`;
  }
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
function elementwiseRowDomainSupported(graph, node, sequence, outputName, sequenceMajor, lanes) {
  const outputShape = graph.tensors[outputName]?.shape;
  if (rowLocalAxis(outputShape, sequence, sequenceMajor, lanes) < 0) return false;
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
function crossSdpaRowDomainSupported(graph, node, dirty, sequence, outputName, sequenceMajor, lanes) {
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
  if (rowLocalAxis(queryShape, sequence, sequenceMajor, lanes) < 0 ||
      rowLocalAxis(outputShape, sequence, sequenceMajor, lanes) < 0) return false;
  /* Exact B equality across query, memory and output. A `[1,S_mem,D]` memory
   * paired with a two-lane query is not a broadcast the row path may perform:
   * both lanes would attend over lane zero's memory. */
  const queryLayout = attentionRowLayout(graph, queryShape, sequenceMajor, lanes);
  const memoryLayout = attentionRowLayout(graph, keyShape, sequenceMajor, lanes);
  if (!queryLayout || !memoryLayout ||
      queryLayout.batch !== lanes || memoryLayout.batch !== lanes) return false;
  /* Whether a dirty K/V is legal depends on which attention this is, the same
   * split hybrid_qsdpa_dirty_kv_supported makes natively. Causal
   * self-attention deliberately consumes the prefix the dirty row-linear nodes
   * just produced, so its K and V are expected to be dirty. Cross-attention
   * reads a retained memory, and a dirty K or V there would mean the memory is
   * being recomputed, which one row cannot do. */
  if (node.params?.causal === true) {
    const keyAxis = leadingSequenceAxis(keyShape, sequenceMajor, lanes);
    if (keyAxis < 0 || keyShape?.[keyAxis] !== sequence) return false;
  } else if (dirty.has(keyName) || dirty.has(valueName)) {
    return false;
  }
  return sameSymbolicShape(queryShape, outputShape);
}

/* Row r of the output reads row r of `a` against the whole of `b`, so this is
 * row-local exactly while `b` carries no token axis. */
function batchMatMulRowDomainSupported(graph, node, dirtyInputs, sequence, outputName, sequenceMajor, lanes) {
  if (!exactInputNames(node, ['a', 'b'])) return false;
  const leftName = node.inputs?.a;
  const rightName = node.inputs?.b;
  if (dirtyInputs.length !== 1 || dirtyInputs[0] !== leftName) return false;
  const leftShape = graph.tensors[leftName]?.shape;
  const rightShape = graph.tensors[rightName]?.shape;
  const outputShape = graph.tensors[outputName]?.shape;
  const leftAxis = rowLocalAxis(leftShape, sequence, sequenceMajor, lanes);
  const outputAxis = rowLocalAxis(outputShape, sequence, sequenceMajor, lanes);
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
  /* The context's declared slot count, not something read out of a shape. A
   * proof that inferred it would attest whatever the sample binding happened to
   * carry, while the row-decode contract says the slot owner declares the lane
   * count. */
  const lanes = options?.lanes ?? 1;
  if (!Number.isSafeInteger(lanes) || lanes < 1)
    return rowDomainRefusedReason(`declared lane count ${String(lanes)} is not a positive integer`);
  /* More than one lane needs the batch-major spelling: the sequence-major
   * layouts have no axis a second lane could live on. */
  if (lanes > 1 && sequenceMajor)
    return rowDomainRefusedReason('sequence-major rows cannot carry more than one lane');
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
    const candidate = sequenceDimension(input?.shape, sequenceMajor, lanes);
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
      if (!qAddRowDomainSupported(graph, node, dirtyInputs, sequence, outputName, lanes))
        return rowDomainRefused(node, 'adds operands that are not both row-local');
      dirty.add(outputName);
      continue;
    }
    if (node.opType === 'QSDPA') {
      const refusal = qSdpaRowRefusal(
        graph, node, dirtyInputs, sequence, outputName, sequenceMajor, lanes);
      if (refusal !== null) return rowDomainRefused(node, refusal);
      dirty.add(outputName);
      continue;
    }
    if (DOMAIN_ELEMENTWISE_ROW_OPS.has(node.opType)) {
      if (!elementwiseRowDomainSupported(graph, node, sequence, outputName, sequenceMajor, lanes))
        return rowDomainRefused(node, 'is not elementwise over its operands');
      dirty.add(outputName);
      continue;
    }
    if (node.opType === 'CrossSDPA') {
      if (!crossSdpaRowDomainSupported(graph, node, dirty, sequence, outputName, sequenceMajor, lanes))
        return rowDomainRefused(node, 'has a dirty K/V or a non-row query');
      dirty.add(outputName);
      continue;
    }
    /* Through the set, which has always held both spellings: the proof reads
     * only shapes, so the quantized twin needs the same one and dispatching on
     * a bare string left QBatchMatMul unreachable while SUPPORTED_OPS could
     * already execute it. */
    if (BATCH_MATMUL_ROW_OPS.has(node.opType)) {
      if (!batchMatMulRowDomainSupported(graph, node, dirtyInputs, sequence, outputName, sequenceMajor, lanes))
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
    const inputSequenceAxis = rowLocalAxis(inputShape, sequence, sequenceMajor, lanes);
    const outputSequenceAxis = rowLocalAxis(outputShape, sequence, sequenceMajor, lanes);
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

function fail(node, message): never {
  throw new Error(`W8A8 incremental row node ${String(node?.id ?? '<unnamed>')} ${message}`);
}

function typedStorage(tensor, node, label) {
  if (!tensor || !ArrayBuffer.isView(tensor.buffer) || tensor.buffer instanceof DataView) {
    fail(node, `${label} requires typed storage.`);
  }
  return tensor.buffer;
}

/* `geometry` records how this view was addressed, for a runtime that has to
 * re-issue the addressing itself every step instead of receiving a view. Inert
 * for a host backend: it reads the buffer and never looks. See
 * `RowSpanGeometry`. */
function cloneTensorWithView(
  tensor, shape, buffer, geometry: RowSpanGeometry | null = null,
) {
  const clone = Object.assign(Object.create(Object.getPrototypeOf(tensor)), tensor, {
    shape,
    sizeBytes: buffer.byteLength,
    buffer,
  });
  if (geometry) clone._rowSpan = geometry;
  return clone;
}

function positiveShape(tensor, node, label) {
  const shape = tensor?.shape;
  if (!Array.isArray(shape) || shape.length < 1 ||
      shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)) {
    fail(node, `${label} must have a positive integer shape.`);
  }
  return shape;
}

/* Resolve the physical row order for the supported sequence layouts.
 *
 * `lanes` is the step's declared slot count, never a shape the tensor is
 * inspected for. One lane admits the three established spellings -- [1,S,...],
 * [S,1,D], [S,D] -- because a single sequence needs no batch axis. More than one
 * lane requires the batch-major [B,S,...] spelling with B equal to `lanes`: the
 * sequence-major spellings have nowhere to put a second lane, and inferring a
 * lane count from a leading extent that happens to match is how a two-lane
 * batch silently decodes one lane. */
function sequenceLayout(tensor, lanes, sequenceLength, node, label) {
  const storage = typedStorage(tensor, node, label);
  const shape = positiveShape(tensor, node, label);
  let batchAxis = -1;
  let sequenceAxis = -1;
  if (lanes > 1) {
    if (shape.length >= 2 && shape[0] === lanes && shape[1] === sequenceLength) {
      batchAxis = 0;
      sequenceAxis = 1;
    } else {
      fail(node, `${label} must use contiguous [${lanes},${sequenceLength},...] storage ` +
        `for a ${lanes}-lane step.`);
    }
  } else if (shape.length >= 2 && shape[0] === 1 && shape[1] === sequenceLength) {
    batchAxis = 0;
    sequenceAxis = 1;
  } else if (shape.length === 3 && shape[0] === sequenceLength && shape[1] === 1) {
    sequenceAxis = 0;
  } else if (shape.length === 2 && shape[0] === sequenceLength) {
    sequenceAxis = 0;
  } else {
    fail(node, `${label} must use contiguous [1,S,...], [S,1,D], or [S,D] storage with S=${sequenceLength}.`);
  }
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  const batch = batchAxis < 0 ? 1 : shape[batchAxis];
  const width = elements / (batch * sequenceLength);
  if (!Number.isSafeInteger(width) || width <= 0 || storage.length !== elements ||
      (tensor.sizeBytes != null && tensor.sizeBytes !== storage.byteLength)) {
    fail(node, `${label} has incompatible contiguous row storage.`);
  }
  return {
    storage, shape, batchAxis, batch, sequenceAxis, sequenceLength, width,
    /* Rows -- not elements -- between one lane and the next. Zero for a tensor
     * that broadcasts across lanes, so every lane reads lane zero's sequence. */
    laneStride: sequenceLength,
  };
}

function activationSequenceLayout(tensor, lanes, sequenceLength, node, label) {
  const layout = sequenceLayout(tensor, lanes, sequenceLength, node, label);
  const shape = layout.shape;
  const exact = (shape.length === 3 && layout.sequenceAxis === 1) ||
    (shape.length === 3 && layout.sequenceAxis === 0 && shape[1] === 1) ||
    (shape.length === 2 && layout.sequenceAxis === 0);
  if (!exact) {
    fail(node, `${label} must use [B,${sequenceLength},D], [${sequenceLength},1,D], or [${sequenceLength},D] storage.`);
  }
  return layout;
}

function inferOutputSequenceLayout(tensor, rowSet: DecodeRowSet, node) {
  const shape = positiveShape(tensor, node, 'output');
  const lanes = rowSet.lanes;
  let sequenceLength = null;
  if (lanes > 1) {
    if (shape.length >= 2 && shape[0] === lanes && shape[1] > 1) sequenceLength = shape[1];
  } else if (shape.length >= 2 && shape[0] === 1 && shape[1] > 1) {
    sequenceLength = shape[1];
  } else if (shape.length === 3 && shape[0] > 1 && shape[1] === 1) {
    sequenceLength = shape[0];
  } else if (shape.length === 2 && shape[0] > 1) {
    sequenceLength = shape[0];
  }
  if (sequenceLength == null || !Number.isInteger(sequenceLength)) {
    fail(node, `has no fixed-sequence output for a ${lanes}-lane step.`);
  }
  for (let lane = 0; lane < lanes; lane++) {
    if (rowSet.positions[lane] >= sequenceLength) {
      fail(node, `lane ${lane} position ${rowSet.positions[lane]} is outside a supported ` +
        'fixed-sequence output.');
    }
  }
  return sequenceLayout(tensor, lanes, sequenceLength, node, 'output');
}

/* The shape of a `lanes`-row span of `layout`. The token axis collapses to one
 * and the batch axis, when the spelling has one, carries the lane count. */
function rowSpanShape(layout, lanes) {
  const shape = [...layout.shape];
  shape[layout.sequenceAxis] = 1;
  if (layout.batchAxis >= 0) shape[layout.batchAxis] = lanes;
  return shape;
}

/* Rows between lanes for this tensor: its own stride when it carries one lane
 * per slot, zero when it broadcasts a single sequence across every lane. */
function laneStrideFor(layout, lanes, node, label) {
  if (layout.batch === lanes) return layout.laneStride;
  if (layout.batch === 1) return 0;
  fail(node, `${label} has batch extent ${layout.batch} where the step declares ${lanes} lane(s).`);
}

/* Attention operands keep the rank-agnostic spelling they have always used: a
 * one-lane step presents rank-2 `[S,D]` views, exactly the tensors this file
 * produced before batched decode existed, and a multi-lane step presents
 * `[B,S,D]`. Both are the same operand to every kernel here; only one of them
 * can express more than one lane. */
function attentionSpanShape(lanes, length, width) {
  return lanes === 1 ? [length, width] : [lanes, length, width];
}

function attentionSequenceLayout(tensor, lanes, node, label) {
  const storage = typedStorage(tensor, node, label);
  const shape = positiveShape(tensor, node, label);
  let batch;
  let sequenceLength;
  let width;
  if (shape.length === 3) {
    batch = shape[0];
    sequenceLength = shape[1];
    width = shape[2];
  } else if (shape.length === 2 && lanes === 1) {
    batch = 1;
    sequenceLength = shape[0];
    width = shape[1];
  } else {
    fail(node, `${label} must use contiguous [B,S,D] or [S,D] attention storage.`);
  }
  /* The batched row-decode contract requires exact B equality across query,
   * key, value, memory, mask and output: a memory or K/V pool with fewer lanes
   * than the step declares would have one lane read another lane's attention
   * state. */
  if (batch !== lanes) {
    fail(node, `${label} has batch extent ${batch} where the step declares ${lanes} lane(s).`);
  }
  if (storage.length !== batch * sequenceLength * width ||
      (tensor.sizeBytes != null && tensor.sizeBytes !== storage.byteLength)) {
    fail(node, `${label} has incompatible contiguous attention storage.`);
  }
  return { storage, batch, sequenceLength, width, laneStride: sequenceLength };
}

function attentionSequenceView(tensor, start, length, layout) {
  const offset = start * layout.width;
  return cloneTensorWithView(
    tensor, [length, layout.width],
    layout.storage.subarray(offset, offset + length * layout.width),
    start === 0
      ? rowSpanGeometry('window', layout.width, layout.laneStride, false, 'read')
      : null,
  );
}

function attentionFull(tensor, lanes, layout) {
  if (lanes > 1) return tensor;
  return attentionSequenceView(tensor, 0, layout.sequenceLength, layout);
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

/* Describe an operand that broadcasts against the output, or null when it does
 * not vary along the token axis and is therefore valid for any row.
 *
 * Right-aligned broadcasting matters here: [D] is a feature vector, not a
 * sequence of length D, even when D happens to equal S. */
function broadcastRowLayout(tensor, rowSet: DecodeRowSet, outputLayout, node, label, opType) {
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
      shape[inputSequenceAxis] === 1) return null;
  const sequenceLength = outputLayout.sequenceLength;
  if (shape[inputSequenceAxis] !== sequenceLength) {
    fail(node, `${label} has an incompatible sequence dimension.`);
  }
  /* An operand that varies along the token axis must carry the batch axis when
   * the step has more than one lane. A partial-rank operand would need the
   * lanes' rows staged under a shape that cannot say which axis is the lane, and
   * every elementwise operator in the attested domain already matches the
   * output rank exactly, so refusing costs nothing that is reachable. */
  const inputBatchAxis = outputLayout.batchAxis < 0 ? -1 : outputLayout.batchAxis - offset;
  if (rowSet.lanes > 1 && inputBatchAxis < 0) {
    fail(node, `${label} varies along the token axis without a batch axis for a ${rowSet.lanes}-lane step.`);
  }
  const batch = inputBatchAxis < 0 ? 1 : shape[inputBatchAxis];
  const leading = shape.slice(0, Math.max(inputBatchAxis, 0))
    .reduce((product, dimension) => product * dimension, 1);
  const width = shape.slice(inputSequenceAxis + 1)
    .reduce((product, dimension) => product * dimension, 1);
  if (leading !== 1 || !Number.isSafeInteger(width) || width <= 0 ||
      storage.length !== batch * sequenceLength * width) {
    fail(node, `${label} does not have contiguous sequence-row storage.`);
  }
  return {
    storage, shape, batchAxis: inputBatchAxis, batch,
    sequenceAxis: inputSequenceAxis, sequenceLength, width,
    laneStride: sequenceLength,
  };
}

function attentionMask(mask, { position, queryLength, keyLength, prefix }, node) {
  if (!mask) return null;
  const storage = typedStorage(mask, node, 'mask');
  const visibleKeys = prefix ? position + 1 : keyLength;
  if (mask.dtype !== 'int32') fail(node, 'mask must use I32 storage.');
  /* A key-only mask is the same window every step: it starts at key zero and
   * the step publishes how much of it is visible. A query-specific one is a
   * row selected at the position, so its address moves. The distinction is the
   * whole reason a device runtime can bind the first whole and must copy the
   * second. */
  const window = rowSpanGeometry('window', 1, 0, false, 'read');
  const queryRow = rowSpanGeometry('row', keyLength, 0, false, 'read');
  if (mask.shape.length === 1 && mask.shape[0] === keyLength && storage.length === keyLength) {
    const view = prefix ? storage.subarray(0, visibleKeys) : storage;
    return cloneTensorWithView(mask, [visibleKeys], view, window);
  }
  // Preserve [B,K] precedence when B and Q are both one.
  if (mask.shape.length === 2 && mask.shape[0] === 1 && mask.shape[1] === keyLength &&
      storage.length === keyLength) {
    const view = prefix ? storage.subarray(0, visibleKeys) : storage;
    return cloneTensorWithView(mask, [1, visibleKeys], view, window);
  }
  // Query-specific layouts remain safe for B=1 by selecting the current mask
  // row before presenting the local Q=1 call to the operator.
  if (mask.shape.length === 2 && mask.shape[0] === queryLength && mask.shape[1] === keyLength &&
      storage.length === queryLength * keyLength) {
    const start = position * keyLength;
    const view = storage.subarray(start, start + visibleKeys);
    return cloneTensorWithView(mask, [1, visibleKeys], view, queryRow);
  }
  if (mask.shape.length === 3 && mask.shape[0] === 1 && mask.shape[1] === queryLength &&
      mask.shape[2] === keyLength && storage.length === queryLength * keyLength) {
    const start = position * keyLength;
    const view = storage.subarray(start, start + visibleKeys);
    return cloneTensorWithView(mask, [1, 1, visibleKeys], view, queryRow);
  }
  fail(node, 'mask must use [K], [1,K], [Q,K], or [1,Q,K] storage.');
}

/* The step's keep mask over a padded key extent, one row per lane.
 *
 * This is the current dense-row active-length contract. Every lane's attention
 * operand is padded to `rowSet.keyCapacity` so one dispatch can cover the
 * batch, and the lane's own visible length is published through the keep mask
 * the ABI already carries. Reading the graph mask through a per-layout accessor
 * rather than slicing it keeps the query row each lane selects independent,
 * which is the whole difference between a batch of sequences and a batch of one
 * sequence. */
export function keepMaskReader(mask, { queryLength, keyLength }, rowSet: DecodeRowSet, node) {
  if (!mask) return null;
  const storage = typedStorage(mask, node, 'mask');
  if (mask.dtype !== 'int32') fail(node, 'mask must use I32 storage.');
  const shape = mask.shape;
  const lanes = rowSet.lanes;
  const queryRow = (lane: number) => rowSet.positions[lane];
  if (shape.length === 1 && shape[0] === keyLength && storage.length === keyLength) {
    return (_lane: number, key: number) => storage[key] !== 0;
  }
  if (shape.length === 2 && shape[1] === keyLength) {
    if (shape[0] === 1 && storage.length === keyLength) {
      return (_lane: number, key: number) => storage[key] !== 0;
    }
    if (shape[0] === lanes && storage.length === lanes * keyLength) {
      return (lane: number, key: number) => storage[lane * keyLength + key] !== 0;
    }
    if (shape[0] === queryLength && storage.length === queryLength * keyLength) {
      return (lane: number, key: number) => storage[queryRow(lane) * keyLength + key] !== 0;
    }
  }
  if (shape.length === 3 && shape[1] === queryLength && shape[2] === keyLength) {
    if (shape[0] === 1 && storage.length === queryLength * keyLength) {
      return (lane: number, key: number) => storage[queryRow(lane) * keyLength + key] !== 0;
    }
    if (shape[0] === lanes && storage.length === lanes * queryLength * keyLength) {
      return (lane: number, key: number) =>
        storage[(lane * queryLength + queryRow(lane)) * keyLength + key] !== 0;
    }
  }
  fail(node, 'mask must use [K], [1,K], [B,K], [Q,K], [1,Q,K], or [B,Q,K] storage.');
}

/* An I32 shape sample for the scratch allocator. WASM allocates from its own
 * heap and needs the element width, not the values. */
const INT32_SAMPLE = new Int32Array(1);

function outputTensor(node) {
  return node.outputs?.out || Object.values(node.outputs || {})[0];
}

function cloneNode(node, inputs, outputs, params = node.params) {
  return { ...node, inputs, outputs, params };
}

/**
 * Validate the scalar row-position option and return it, or null when absent.
 *
 * Kept as the one-lane spelling of a row set rather than a second path: the
 * caller turns it into `singleLaneRowSet`, so there is one row-set
 * implementation and a one-lane step cannot drift from a batched one.
 */
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

/**
 * The row set a step's options declare, or null when the step is not a row step.
 *
 * `incrementalRowLanes` is the batched spelling and `incrementalRowPosition` the
 * one-lane one; declaring both is a caller bug rather than a precedence question,
 * because the two would disagree about how many slots the context owns.
 */
export function decodeRowSetFromOptions(
  options, selectedNodes, cacheWasValid,
): DecodeRowSet | null {
  const lanes = options?.incrementalRowLanes;
  const scalar = incrementalRowPosition(options, selectedNodes, cacheWasValid);
  if (lanes === undefined || lanes === null) {
    return scalar === null ? null : singleLaneRowSet(scalar, options?.kvPages ?? null);
  }
  if (scalar !== null) {
    throw new Error(
      'W8A8 decode accepts incrementalRowPosition or incrementalRowLanes, not both.');
  }
  if (options?.incremental !== true || options.incrementalReset === true ||
      cacheWasValid !== true || !(selectedNodes instanceof Set)) {
    throw new Error('W8A8 incrementalRowLanes requires a valid incremental seed pass and a non-reset dependency selection.');
  }
  if (!Array.isArray(lanes) || lanes.length === 0) {
    throw new Error('W8A8 incrementalRowLanes must be a non-empty array of lane steps.');
  }
  for (const lane of lanes) {
    /* A parked lane carries no position: it holds no request, occupies a dense
     * row so the operands keep their shape, and its result is discarded. */
    if (lane?.parked === true) continue;
    /* Zero is admitted here, unlike the scalar spelling.
     *
     * The scalar floor of one is a proxy for "the seed already wrote row zero,
     * so a step may not claim to advance into it". That proxy holds for a lane
     * that advances and is simply wrong for one that idles: a lane whose only
     * row *is* row zero repeats it, recomputing identical bytes from unchanged
     * inputs. The engine cannot tell those apart -- it knows nothing about
     * seeds or lengths -- so the real check is the per-lane one in
     * `ProviderDecodeLifecycle`, which compares each lane's position against
     * that lane's own next active position. What is left here is the bound. */
    if (!Number.isInteger(lane?.position) || lane.position < 0) {
      throw new Error('W8A8 incrementalRowLanes positions must be non-negative integers, or the lane must be parked.');
    }
  }
  return decodeRowSet(lanes);
}

export interface QuantizedRowContext {
  dirtyTensorNames?: Set<string>;
  allowUnprovenInvariantInputs?: boolean;
  allocateKVScratch?: KVScratchAllocator;
}

/**
 * One node's row form for a whole step, plus the copies that form needs.
 *
 * `gatherRows` fills every staged operand and `scatterRows` writes every staged
 * output back. Both are null for a one-lane unpaged step, where every span is
 * the identity tier and aliases the retained activation directly -- the exact
 * views this function produced before batched decode existed.
 */
export function quantizedRowNode(
  node, rowSet: DecodeRowSet, context: QuantizedRowContext = {},
) {
  if (!SUPPORTED_OPS.has(node?.opType)) {
    fail(node, `uses unsupported op '${String(node?.opType)}'.`);
  }
  const lanes = rowSet.lanes;
  const allocate = context?.allocateKVScratch ?? sharedScratchAllocator;
  const reads: Array<() => void> = [];
  const writes: Array<() => void> = [];
  const scratchKey = (port: string) => `${String(node?.id ?? '?')}:${port}`;
  const isPaged = (tensor) => !!tensor?.name && rowSet.pagedTensors.has(tensor.name);

  /* One staged span, registered on the phase that has to run it. Reads are
   * deferred to dispatch time for the reason the paged K/V gather is: a row plan
   * is built before any node in the step runs, and an operand's bytes are
   * produced during the step. */
  const spanTensor = (
    tensor, layout, direction: 'read' | 'write', port, geometry: RowSpanGeometry,
  ) => {
    const rows = writeRowIndices(rowSet, geometry.laneStride, geometry.paged);
    const view = rowSpanView(
      layout.storage, layout.width, rows, direction, scratchKey(port), allocate,
      rowSet.parked);
    if (view.apply) (direction === 'read' ? reads : writes).push(view.apply);
    return cloneTensorWithView(
      tensor, rowSpanShape(layout, lanes), view.storage, geometry);
  };
  const readGeometry = (layout, port) => rowSpanGeometry(
    'row', layout.width, laneStrideFor(layout, lanes, node, port), false, 'read');

  const output = outputTensor(node);
  const outputLayout = inferOutputSequenceLayout(output, rowSet, node);
  const sequenceLength = outputLayout.sequenceLength;
  /* The write side of paging: the producing projection's row lands in the page
   * that backs its logical position, or the next step gathers a slot nobody
   * wrote. Identity for every tensor outside the pool, which is every tensor
   * when no page table is attached. */
  const outputGeometry = rowSpanGeometry(
    'row', outputLayout.width,
    laneStrideFor(outputLayout, lanes, node, 'output'), isPaged(output), 'write');
  const replaceOutput = (rowOutput) => Object.fromEntries(
    Object.entries(node.outputs || {}).map(([name, tensor]) =>
      [name, tensor === output ? rowOutput : tensor]),
  );
  const outputSpan = (layout = outputLayout) =>
    spanTensor(output, layout, 'write', 'out', outputGeometry);
  /* A row of one operand, selected at each lane's own position. */
  const readSpan = (tensor, port, label = `input ${port}`) => {
    const layout = sequenceLayout(tensor, lanes, sequenceLength, node, label);
    return spanTensor(tensor, layout, 'read', port, readGeometry(layout, label));
  };
  const strictReadSpan = (tensor, port, label = `input ${port}`) => {
    const layout = activationSequenceLayout(tensor, lanes, sequenceLength, node, label);
    return {
      layout,
      tensor: spanTensor(tensor, layout, 'read', port, readGeometry(layout, label)),
    };
  };
  /* An operand that broadcasts against the output: staged per lane when it
   * varies along the token axis, passed through when it does not. */
  const broadcastSpan = (tensor, port, label, opType) => {
    const layout = broadcastRowLayout(tensor, rowSet, outputLayout, node, label, opType);
    if (layout === null) return tensor;
    return spanTensor(tensor, layout, 'read', port, readGeometry(layout, label));
  };

  const finish = (rowNode) => {
    if (reads.length > 0) rowNode.gatherRows = () => { for (const run of reads) run(); };
    if (writes.length > 0) rowNode.scatterRows = () => { for (const run of writes) run(); };
    return rowNode;
  };
  const outputs = replaceOutput(outputSpan());

  if (node.opType === 'Embedding' || node.opType === 'QEmbedding') {
    const input = readSpan(node.inputs?.input, 'input', 'input IDs');
    return finish(cloneNode(node, { ...node.inputs, input }, outputs));
  }

  /* Every re-view is the same operation on a row: the contiguous flattening is
   * identical, so row `r` is one run of the same width in both spellings. The
   * attestation proved the width; this asserts it again on the physical
   * tensors, because the executor must never rely on a proof it cannot see. */
  if (ROW_VIEW_OPS.has(node.opType)) {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'data';
    const input = node.inputs?.[inputName];
    const inputRow = strictReadSpan(input, inputName, 'input');
    const strictOutputLayout = activationSequenceLayout(
      output, lanes, sequenceLength, node, 'output',
    );
    if (input?.dtype !== output?.dtype ||
        inputRow.layout.width !== strictOutputLayout.width) {
      fail(node, `${node.opType} must preserve the dtype and contiguous logical row width.`);
    }
    return finish(cloneNode(
      node,
      { ...node.inputs, [inputName]: inputRow.tensor },
      replaceOutput(outputSpan(strictOutputLayout)),
    ));
  }

  if (BATCH_MATMUL_ROW_OPS.has(node.opType)) {
    const b = node.inputs?.b;
    invariantInput(node, b, 'input b', context);
    const aRow = strictReadSpan(node.inputs?.a, 'a', 'input a');
    const strictOutputLayout = activationSequenceLayout(
      output, lanes, sequenceLength, node, 'output',
    );
    return finish(cloneNode(
      node,
      { ...node.inputs, a: aRow.tensor },
      replaceOutput(outputSpan(strictOutputLayout)),
    ));
  }

  if (LINEAR_ROW_OPS.has(node.opType)) {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'a';
    const input = readSpan(node.inputs?.[inputName], inputName, 'input');
    return finish(cloneNode(node, { ...node.inputs, [inputName]: input }, outputs));
  }

  if (node.opType === 'Add' || node.opType === 'Mul' || node.opType === 'QAdd') {
    const aName = node.inputs?.a ? 'a' : node.inputs?.input ? 'input' : 'x';
    const bName = node.inputs?.b ? 'b' : 'y';
    const binaryOutputLayout = node.opType === 'Mul'
      ? activationSequenceLayout(output, lanes, sequenceLength, node, 'output')
      : outputLayout;
    const select = node.opType === 'Add' || node.opType === 'Mul'
      ? (tensor, port, label) => broadcastSpan(tensor, port, label, node.opType)
      : (tensor, port, label) => readSpan(tensor, port, label);
    const originalA = node.inputs?.[aName];
    const originalB = node.inputs?.[bName];
    const a = select(originalA, aName, 'input a');
    const b = select(originalB, bName, 'input b');
    if (node.opType === 'Add' || node.opType === 'Mul') {
      if (a === originalA) invariantInput(node, originalA, 'broadcast input a', context);
      if (b === originalB) invariantInput(node, originalB, 'broadcast input b', context);
    }
    const binaryOutputs = node.opType === 'Mul'
      ? replaceOutput(outputSpan(binaryOutputLayout))
      : outputs;
    return finish(cloneNode(node, { ...node.inputs, [aName]: a, [bName]: b }, binaryOutputs));
  }

  if (NARY_ELEMENTWISE_ROW_OPS.has(node.opType)) {
    const inputs = { ...node.inputs };
    for (const [port, tensor] of Object.entries(node.inputs || {})) {
      const original = tensor as any;
      const selected = broadcastSpan(original, port, `input ${port}`, node.opType);
      if (selected === original) invariantInput(node, original, `input ${port}`, context);
      inputs[port] = selected;
    }
    return finish(cloneNode(node, inputs, outputs));
  }

  if (node.opType === 'Expand') {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'data';
    const originalInput = node.inputs?.[inputName];
    const input = broadcastSpan(originalInput, inputName, 'input', node.opType);
    if (input === originalInput) {
      invariantInput(node, originalInput, 'broadcast input', context);
    }
    return finish(cloneNode(node, { ...node.inputs, [inputName]: input }, outputs));
  }

  if (node.opType === 'LayerNorm' || node.opType === 'QLayerNorm') {
    const input = readSpan(node.inputs?.input, 'input', 'input');
    return finish(cloneNode(node, { ...node.inputs, input }, outputs));
  }

  if (ELEMENTWISE_ROW_OPS.has(node.opType)) {
    const inputName = node.inputs?.input ? 'input' : node.inputs?.x ? 'x' : 'data';
    const input = readSpan(node.inputs?.[inputName], inputName, 'input');
    // Scale and zero point are per-tensor scalars shared by every row, so they
    // pass through untouched alongside the sliced activation.
    return finish(cloneNode(node, { ...node.inputs, [inputName]: input }, outputs));
  }

  if (node.opType === 'QArgMax') {
    const input = node.inputs?.input;
    const inputLayout = sequenceLayout(input, lanes, sequenceLength, node, 'input');
    let axis = node.params?.axis;
    if (axis < 0) axis += input?.shape?.length ?? 0;
    if (!Number.isInteger(axis) || axis <= inputLayout.sequenceAxis) {
      fail(node, 'QArgMax must reduce a feature axis strictly after the sequence axis.');
    }
    return finish(cloneNode(node, {
      ...node.inputs,
      input: spanTensor(
        input, inputLayout, 'read', 'input', readGeometry(inputLayout, 'input')),
    }, outputs));
  }

  if (!ATTENTION_ROW_OPS.has(node.opType)) {
    fail(node, `has no row implementation for '${String(node.opType)}'.`);
  }
  const q = node.inputs?.q;
  const k = node.inputs?.k;
  const v = node.inputs?.v;
  const qLayout = attentionSequenceLayout(q, lanes, node, 'q');
  const kLayout = attentionSequenceLayout(k, lanes, node, 'k');
  const vLayout = attentionSequenceLayout(v, lanes, node, 'v');
  const attentionOutputLayout = attentionSequenceLayout(output, lanes, node, 'output');
  const queryLength = qLayout.sequenceLength;
  const keyLength = kLayout.sequenceLength;
  if (queryLength !== sequenceLength ||
      attentionOutputLayout.sequenceLength !== sequenceLength ||
      vLayout.sequenceLength !== keyLength ||
      qLayout.width !== kLayout.width || qLayout.width !== vLayout.width ||
      qLayout.width !== attentionOutputLayout.width) {
    fail(node, `${node.opType} requires compatible [B,S,D] or [S,D] Q/K/V/output storage.`);
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
  const width = qLayout.width;
  const attentionSpan = (tensor, layout, direction: 'read' | 'write', port) => {
    const geometry = rowSpanGeometry(
      'row', layout.width, layout.laneStride, false, direction);
    const rows = writeRowIndices(rowSet, layout.laneStride, false);
    const view = rowSpanView(
      layout.storage, layout.width, rows, direction, scratchKey(port), allocate,
      rowSet.parked);
    if (view.apply) (direction === 'read' ? reads : writes).push(view.apply);
    return cloneTensorWithView(
      tensor, attentionSpanShape(lanes, 1, layout.width), view.storage, geometry);
  };
  const rowQ = attentionSpan(q, qLayout, 'read', 'q');

  /* Paging applies to the retained causal K/V only. A cross-attention memory
   * is read whole and never appended to, so it has no page table and no
   * per-lane length -- giving it one would add a gather for no benefit. */
  const visibleKeys = causal ? rowSet.keyCapacity : keyLength;
  const causalOperand = (tensor, layout, port) => {
    const paged = isPaged(tensor);
    /* One lane needs no staging at all: its prefix is one contiguous run of the
     * pool under the identity mapping, which is what a contiguous cache is, and
     * `pagedSequenceView` returns exactly the slice this file used before pages
     * existed. Staging is the general implementation and this is its zero-copy
     * tier, not a preserved second path. */
    const prefix = rowSpanGeometry(
      'prefix', layout.width, layout.laneStride, paged, 'read');
    if (lanes === 1) {
      const plan = paged ? rowSet.pages[0] : null;
      if (!plan) {
        /* The address never moves; only the visible length does. A backend that
         * binds whole buffers executes this by publishing the length, which is
         * why one unpaged lane needs no staging machinery at all. */
        return cloneTensorWithView(
          tensor, attentionSpanShape(1, visibleKeys, layout.width),
          layout.storage.subarray(0, visibleKeys * layout.width),
          rowSpanGeometry('window', layout.width, layout.laneStride, false, 'read'));
      }
      const view = pagedSequenceView(
        layout.storage, layout.width, plan, scratchKey(port), allocate);
      if (view.gather) reads.push(view.gather);
      return cloneTensorWithView(
        tensor, attentionSpanShape(1, view.length, layout.width), view.storage, prefix);
    }
    const staged = batchPrefixView(
      layout.storage, layout.width, rowSet, layout.laneStride, paged,
      scratchKey(port), allocate);
    reads.push(staged.apply!);
    return cloneTensorWithView(
      tensor, attentionSpanShape(lanes, staged.keyCapacity, layout.width),
      staged.storage, prefix);
  };
  const keyTensor = causal
    ? causalOperand(k, kLayout, 'k')
    : attentionFull(k, lanes, kLayout);
  const valueTensor = causal
    ? causalOperand(v, vLayout, 'v')
    : attentionFull(v, lanes, vLayout);
  /* One lane keeps the slice `attentionMask` has always produced; more than one
   * needs the lengths published, because the padded key extent is the longest
   * lane's and every shorter lane must stop at its own. */
  const mask = lanes === 1
    ? attentionMask(node.inputs?.mask, {
        position: rowSet.positions[0], queryLength, keyLength, prefix: causal,
      }, node)
    : maskForLanes(
        node, rowSet, visibleKeys, causal, queryLength, keyLength, allocate, reads);
  const inputs = { ...node.inputs, q: rowQ, k: keyTensor, v: valueTensor };
  if (mask) inputs.mask = mask;
  else delete inputs.mask;
  void width;
  return finish(cloneNode(
    node,
    inputs,
    replaceOutput(attentionSpan(output, attentionOutputLayout, 'write', 'out')),
    { ...node.params, causal: false },
  ));
}

/* The `[B,K]` keep mask a multi-lane attention row reads.
 *
 * Always materialised for a causal row, even when the graph carries no mask:
 * the per-lane lengths have to reach the kernel somehow, and until the ABI
 * carries direct `q_len`/`kv_len` inputs the keep mask is where they go. */
function maskForLanes(
  node, rowSet: DecodeRowSet, keys: number, causal: boolean,
  queryLength: number, keyLength: number, allocate: KVScratchAllocator,
  reads: Array<() => void>,
) {
  const source = node.inputs?.mask;
  const read = keepMaskReader(source, { queryLength, keyLength }, rowSet, node);
  if (!causal && read === null) return null;
  const staged = stageKeepMask(
    rowSet.lanes, keys, causal ? rowSet.kvLengths : null, read,
    `${String(node?.id ?? '?')}:mask`, allocate, INT32_SAMPLE);
  reads.push(staged.apply);
  const template = source ?? {
    name: `${String(node?.id ?? '?')}.keep`, dtype: 'int32', isWeight: false,
  };
  /* Not a region of any tensor: the host computes it from the step's lane
   * lengths and the graph's mask, so no source row can name it and a device
   * runtime has to upload it rather than copy it. */
  return cloneTensorWithView(
    template, [rowSet.lanes, keys], staged.storage,
    rowSpanGeometry('keepMask', keys, 0, false, 'read'));
}

export function prepareQuantizedRows(
  graph, selectedNodes, rowSet: DecodeRowSet,
  { changedInputs = [], allocateKVScratch = undefined }: {
    changedInputs?: string[];
    /* Backend-owned staging storage. WASM must supply one: its kernels take
     * pointers into the compiled linear memory. */
    allocateKVScratch?: KVScratchAllocator;
  } = {},
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
  if (rowSet.pagedTensors.size > 0) assertPagedTensorDomain(selected, rowSet.pagedTensors);
  const context = { dirtyTensorNames, allocateKVScratch };
  for (const [nodeIndex, node] of selected) {
    rows.set(nodeIndex, quantizedRowNode(node, rowSet, context));
  }
  return rows;
}

/* A paged tensor may be read only as an attention `k` or `v` operand, and
 * written only by one producing node.
 *
 * Everything else in this file addresses a row as `position * width`. That is
 * correct for every tensor whose logical order is its physical order, which is
 * every tensor except a paged one -- so a Reshape or a residual Add that
 * happens to touch the K/V pool would read the wrong slot and produce a
 * plausible, wrong decoder. The row path proves its domain rather than
 * assuming it, so this refuses instead of guessing.
 *
 * Narrowing the paged set is the fix when this fires, not widening the rule:
 * every additional paged tensor is another operator that must learn page-table
 * addressing. */
function assertPagedTensorDomain(selected, pagedTensors: ReadonlySet<string>) {
  const producers = new Map<string, any>();
  for (const [, node] of selected) {
    for (const [port, tensor] of Object.entries(node.inputs || {}) as Array<[string, any]>) {
      const name = tensor?.name;
      if (!name || !pagedTensors.has(name)) continue;
      if (!ATTENTION_ROW_OPS.has(node.opType) || (port !== 'k' && port !== 'v')) {
        fail(node, `reads paged K/V tensor '${name}' on port '${port}', which has no page-table addressing.`);
      }
    }
    for (const tensor of Object.values(node.outputs || {}) as any[]) {
      const name = tensor?.name;
      if (!name || !pagedTensors.has(name)) continue;
      const prior = producers.get(name);
      if (prior && prior !== node) {
        fail(node, `writes paged K/V tensor '${name}' already written by node ${String(prior.id)}.`);
      }
      producers.set(name, node);
    }
  }
}
