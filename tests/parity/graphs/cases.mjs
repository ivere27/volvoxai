// Level 2: mixed multi-op graphs. Single ops can each be correct while their
// *composition* is buggy — wrong wiring, tensor fan-out/aliasing, a layout or
// dtype boundary, or a fusion that fires incorrectly. These curated small graphs
// exercise those seams across every tier vs pure-JS, plus a PyTorch oracle.
//
// MatMul weights use non-square [K,N] so the kernel unambiguously picks the
// input-major (din) layout, i.e. out = x @ W (matches torch).
import fs from 'fs';
import path from 'path';
import { seededFloat32, writeTensor } from '../lib/tensorio.mjs';
import { writeGraph, writeSafetensors, ensureDir } from '../lib/authorpkg.mjs';

const B = 8; // sequence/batch rows

export const cases = [
  {
    id: 'linear_gelu',
    inputs: { x: { shape: [1, B, 16], seed: 11 } },
    weights: { W1: { shape: [16, 32], seed: 12 } },
    nodes: [
      { opType: 'MatMul', inputs: { input: 'x', weight: 'W1' }, outputs: { out: 'h' }, outputs_shape: { out: [1, B, 32] } },
      { opType: 'GELU', inputs: { input: 'h' }, outputs: { out: 'y' }, outputs_shape: { out: [1, B, 32] } },
    ],
    outputs: ['y'],
  },
  {
    id: 'mlp',
    inputs: { x: { shape: [1, B, 16], seed: 21 } },
    weights: { W1: { shape: [16, 32], seed: 22 }, W2: { shape: [32, 16], seed: 23 } },
    nodes: [
      { opType: 'MatMul', inputs: { input: 'x', weight: 'W1' }, outputs: { out: 'h1' }, outputs_shape: { out: [1, B, 32] } },
      { opType: 'GELU', inputs: { input: 'h1' }, outputs: { out: 'a' }, outputs_shape: { out: [1, B, 32] } },
      { opType: 'MatMul', inputs: { input: 'a', weight: 'W2' }, outputs: { out: 'y' }, outputs_shape: { out: [1, B, 16] } },
    ],
    outputs: ['y'],
  },
  {
    // fan-out / residual: x feeds two nodes, then they recombine. Exercises
    // tensor aliasing and the residual Add wiring.
    id: 'gelu_residual',
    inputs: { x: { shape: [1, B, 16], seed: 31 } },
    weights: {},
    nodes: [
      { opType: 'GELU', inputs: { input: 'x' }, outputs: { out: 'g' }, outputs_shape: { out: [1, B, 16] } },
      { opType: 'Add', inputs: { a: 'g', b: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [1, B, 16] } },
    ],
    outputs: ['y'],
  },
  {
    // norm → linear: a layout/normalization boundary feeding a matmul.
    id: 'layernorm_linear',
    inputs: { x: { shape: [1, B, 16], seed: 41 } },
    weights: { g: { shape: [16], seed: 42, lo: 0.5, hi: 1.5 }, bta: { shape: [16], seed: 43, lo: -0.2, hi: 0.2 }, W: { shape: [16, 24], seed: 44 } },
    nodes: [
      { opType: 'LayerNorm', inputs: { input: 'x', weight: 'g', bias: 'bta' }, outputs: { out: 'n' }, outputs_shape: { out: [1, B, 16] }, params: { eps: 1e-5, d_model: 16 } },
      { opType: 'MatMul', inputs: { input: 'n', weight: 'W' }, outputs: { out: 'y' }, outputs_shape: { out: [1, B, 24] } },
    ],
    outputs: ['y'],
  },

  // Single-op cases added via the L2 machinery (weights/params/shape changes) and
  // scored by the generic torch interpreter. Reductions/softmax are last-axis.
  { id: 'softmax', inputs: { x: { shape: [8, 16], seed: 51 } }, weights: {},
    nodes: [{ opType: 'Softmax', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [8, 16] } }], outputs: ['y'] },
  { id: 'logsoftmax', inputs: { x: { shape: [8, 16], seed: 52 } }, weights: {},
    nodes: [{ opType: 'LogSoftmax', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [8, 16] } }], outputs: ['y'] },
  { id: 'reducemean', inputs: { x: { shape: [8, 16], seed: 53 } }, weights: {},
    nodes: [{ opType: 'ReduceMean', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [8] } }], outputs: ['y'] },
  { id: 'reducesum', inputs: { x: { shape: [8, 16], seed: 54 } }, weights: {},
    nodes: [{ opType: 'ReduceSum', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [8] } }], outputs: ['y'] },
  { id: 'rmsnorm', inputs: { x: { shape: [8, 16], seed: 55 } }, weights: { weight: { shape: [16], seed: 56, lo: 0.5, hi: 1.5 } },
    nodes: [{ opType: 'RMSNorm', inputs: { input: 'x', weight: 'weight' }, outputs: { out: 'y' }, outputs_shape: { out: [8, 16] }, params: { eps: 1e-5, d_model: 16 } }], outputs: ['y'] },
  { id: 'transpose', inputs: { x: { shape: [4, 8], seed: 57 } }, weights: {},
    nodes: [{ opType: 'Transpose', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [8, 4] }, params: { perm: [1, 0] } }], outputs: ['y'] },
  { id: 'concat', inputs: { in0: { shape: [4, 8], seed: 58 }, in1: { shape: [4, 8], seed: 59 } }, weights: {},
    nodes: [{ opType: 'Concat', inputs: { input0: 'in0', input1: 'in1' }, outputs: { out: 'y' }, outputs_shape: { out: [8, 8] }, params: { axis: 0, count: 2 } }], outputs: ['y'] },
  { id: 'prelu', inputs: { x: { shape: [4, 8], seed: 61 } }, weights: { slope: { shape: [8], seed: 62, lo: -0.3, hi: 0.3 } },
    nodes: [{ opType: 'PReLU', inputs: { input: 'x', slope: 'slope' }, outputs: { out: 'y' }, outputs_shape: { out: [4, 8] } }], outputs: ['y'] },

  // Vision ops — NHWC [N,H,W,C]. GlobalAveragePool over H,W; BatchNorm/GroupNorm per channel.
  { id: 'globalavgpool', inputs: { x: { shape: [1, 4, 4, 8], seed: 71 } }, weights: {},
    nodes: [{ opType: 'GlobalAveragePool', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 1, 1, 8] } }], outputs: ['y'] },
  { id: 'batchnorm', inputs: { x: { shape: [1, 4, 4, 8], seed: 72 } },
    weights: { weight: { shape: [8], seed: 73, lo: 0.5, hi: 1.5 }, bias: { shape: [8], seed: 74, lo: -0.3, hi: 0.3 }, running_mean: { shape: [8], seed: 75, lo: -0.5, hi: 0.5 }, running_var: { shape: [8], seed: 76, lo: 0.5, hi: 1.5 } },
    nodes: [{ opType: 'BatchNorm2D', inputs: { input: 'x', weight: 'weight', bias: 'bias', running_mean: 'running_mean', running_var: 'running_var' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 4, 4, 8] }, params: { eps: 1e-5 } }], outputs: ['y'] },
  { id: 'groupnorm', inputs: { x: { shape: [1, 4, 4, 8], seed: 77 } },
    weights: { weight: { shape: [8], seed: 78, lo: 0.5, hi: 1.5 }, bias: { shape: [8], seed: 79, lo: -0.3, hi: 0.3 } },
    nodes: [{ opType: 'GroupNorm', inputs: { input: 'x', weight: 'weight', bias: 'bias' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 4, 4, 8] }, params: { num_groups: 2, eps: 1e-5 } }], outputs: ['y'] },
  { id: 'linear', inputs: { x: { shape: [1, 8, 16], seed: 81 } }, weights: { W: { shape: [16, 32], seed: 82 }, b: { shape: [32], seed: 83 } },
    nodes: [{ opType: 'Linear', inputs: { input: 'x', weight: 'W', bias: 'b' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 8, 32] } }], outputs: ['y'] },
  { id: 'avgpool2d', inputs: { x: { shape: [1, 4, 4, 8], seed: 84 } }, weights: {}, skip: ['native-cpu'],
    nodes: [{ opType: 'AveragePool2D', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 2, 2, 8] }, params: { kernel: [2, 2], stride: [2, 2], padding: [0, 0] } }], outputs: ['y'] },
  { id: 'upsample', inputs: { x: { shape: [1, 2, 2, 8], seed: 85 } }, weights: {},
    nodes: [{ opType: 'UpsampleNearest2D', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 4, 4, 8] } }], outputs: ['y'] },
  { id: 'pad', inputs: { x: { shape: [1, 4, 4, 8], seed: 86 } }, weights: {}, skip: ['native-cpu'],
    nodes: [{ opType: 'Pad', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 6, 6, 8] }, params: { pads: [1, 1, 1, 1], value: 0 } }], outputs: ['y'] },
  { id: 'meanheight', inputs: { x: { shape: [1, 4, 4, 8], seed: 91 } }, weights: {}, skip: ['native-cpu'],
    nodes: [{ opType: 'MeanHeight', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 8, 4] } }], outputs: ['y'] },
  { id: 'argmax', inputs: { x: { shape: [8, 16], seed: 92 } }, weights: {}, skip: ['native-cpu', 'native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'ArgMax', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [8] }, outputs_dtype: { out: 'int32' }, params: { axis: 1 } }], outputs: ['y'] },

  // Shape/index family. Flatten/Squeeze/Unsqueeze/Identity are shape-only aliases
  // (copy) supported on every tier; Slice is F32 Full on CPU/WASM but [Missing] on
  // native-cpu (docs/operation_list.md), so it skips the native tiers.
  { id: 'flatten', inputs: { x: { shape: [2, 3, 4], seed: 111 } }, weights: {},
    nodes: [{ opType: 'Flatten', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [2, 12] }, params: { axis: 1 } }], outputs: ['y'] },
  { id: 'squeeze', inputs: { x: { shape: [1, 4, 8], seed: 112 } }, weights: {},
    nodes: [{ opType: 'Squeeze', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [4, 8] }, params: { axes: [0] } }], outputs: ['y'] },
  { id: 'unsqueeze', inputs: { x: { shape: [4, 8], seed: 113 } }, weights: {},
    nodes: [{ opType: 'Unsqueeze', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 4, 8] }, params: { axes: [0] } }], outputs: ['y'] },
  { id: 'identity', inputs: { x: { shape: [4, 8], seed: 114 } }, weights: {},
    nodes: [{ opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [4, 8] } }], outputs: ['y'] },
  { id: 'slice', inputs: { x: { shape: [4, 8], seed: 115 } }, weights: {}, skip: ['native-cpu', 'native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'Slice', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [4, 4] }, params: { starts: [2], ends: [6], axes: [1], steps: [1] } }], outputs: ['y'] },
  { id: 'expand', inputs: { x: { shape: [4, 1], seed: 116 } }, weights: {}, skip: ['native-cpu', 'native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'Expand', inputs: { input: 'x' }, outputs: { out: 'y' }, outputs_shape: { out: [4, 8] } }], outputs: ['y'] },

  // ---- F32 fusion / composition mixed graphs (the real L2 seams) -----------
  // SwiGLU: gate & up share x (fan-out); SiLU-gated multiply.
  { id: 'swiglu', inputs: { x: { shape: [1, 8, 16], seed: 101 } }, weights: { Wg: { shape: [16, 32], seed: 102 }, Wu: { shape: [16, 32], seed: 103 } },
    nodes: [
      { opType: 'MatMul', inputs: { input: 'x', weight: 'Wg' }, outputs: { out: 'g' }, outputs_shape: { out: [1, 8, 32] } },
      { opType: 'SiLU', inputs: { input: 'g' }, outputs: { out: 'sg' }, outputs_shape: { out: [1, 8, 32] } },
      { opType: 'MatMul', inputs: { input: 'x', weight: 'Wu' }, outputs: { out: 'u' }, outputs_shape: { out: [1, 8, 32] } },
      { opType: 'Mul', inputs: { a: 'sg', b: 'u' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 8, 32] } },
    ], outputs: ['y'] },
  // Chained residual adds.
  { id: 'residual_chain', inputs: { a: { shape: [1, 8, 16], seed: 104 }, b: { shape: [1, 8, 16], seed: 105 }, c: { shape: [1, 8, 16], seed: 106 } }, weights: {},
    nodes: [
      { opType: 'Add', inputs: { a: 'a', b: 'b' }, outputs: { out: 'h' }, outputs_shape: { out: [1, 8, 16] } },
      { opType: 'Add', inputs: { a: 'h', b: 'c' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 8, 16] } },
    ], outputs: ['y'] },
  // Concat → Sigmoid (detector class-head tail fusion). Ends in Sigmoid, so WASM/native
  // inherit the documented fast-sigmoid approximation (~8e-3); torch matches exact pure-JS.
  { id: 'concat_sigmoid', inputs: { in0: { shape: [4, 8], seed: 107 }, in1: { shape: [4, 8], seed: 108 } }, weights: {}, approxTol: { atol: 1.2e-2 },
    nodes: [
      { opType: 'Concat', inputs: { input0: 'in0', input1: 'in1' }, outputs: { out: 'h' }, outputs_shape: { out: [8, 8] }, params: { axis: 0, count: 2 } },
      { opType: 'Sigmoid', inputs: { input: 'h' }, outputs: { out: 'y' }, outputs_shape: { out: [8, 8] } },
    ], outputs: ['y'] },
  // Conv2D with fused ReLU6 (the #1 vision fusion pattern). NHWC in, OHWI weight.
  { id: 'conv_relu', inputs: { x: { shape: [1, 6, 6, 4], seed: 109 } }, weights: { w: { shape: [8, 3, 3, 4], seed: 110 }, b: { shape: [8], seed: 111 } },
    nodes: [{ opType: 'Conv2D', inputs: { input: 'x', weight: 'w', bias: 'b' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 4, 4, 8] },
      params: { stride: [1, 1], dilation: [1, 1], groups: 1, pads: [0, 0, 0, 0], padding: [0, 0], data_layout: 'NHWC', weight_layout: 'OHWI', relu: 2 } }], outputs: ['y'] },

  // ---- W8A8 int8 boundary: QuantizeLinear → DequantizeLinear round-trip ------
  // Constant scale/zero-point tensors; the int8 intermediate flows between nodes.
  { id: 'quant_roundtrip', inputs: { x: { shape: [1, 8, 16], seed: 121 } },
    weights: { qs: { shape: [1], dtype: 'f32', values: [0.02] }, qz: { shape: [1], dtype: 'i8', values: [0] } },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        xq: { scheme: 'per_tensor', scale_tensor: 'qs', zero_point_tensor: 'qz' },
      },
    },
    nodes: [
      { opType: 'QuantizeLinear', inputs: { input: 'x', scale: 'qs', zero_point: 'qz' }, outputs: { out: 'xq' }, outputs_shape: { out: [1, 8, 16] }, outputs_dtype: { out: 'int8' } },
      { opType: 'DequantizeLinear', inputs: { input: 'xq', scale: 'qs', zero_point: 'qz' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 8, 16] } },
    ], outputs: ['y'] },

  // ---- attention: isolated SDPA + a full transformer block ------------------
  { id: 'sdpa', inputs: { qkv: { shape: [1, 4, 24], seed: 131 } }, weights: {},
    nodes: [{ opType: 'SDPA', inputs: { qkv: 'qkv' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 4, 8] }, params: { heads: 2, causal: true } }], outputs: ['y'] },
  { id: 'attn_block', inputs: { x: { shape: [1, 4, 8], seed: 132 } },
    weights: { g: { shape: [8], seed: 133, lo: 0.5, hi: 1.5 }, bta: { shape: [8], seed: 134, lo: -0.2, hi: 0.2 }, Wqkv: { shape: [8, 24], seed: 135 }, Wout: { shape: [8, 8], seed: 136 } },
    nodes: [
      { opType: 'LayerNorm', inputs: { input: 'x', weight: 'g', bias: 'bta' }, outputs: { out: 'n' }, outputs_shape: { out: [1, 4, 8] }, params: { eps: 1e-5, d_model: 8 } },
      { opType: 'MatMul', inputs: { input: 'n', weight: 'Wqkv' }, outputs: { out: 'qkv' }, outputs_shape: { out: [1, 4, 24] } },
      { opType: 'SDPA', inputs: { qkv: 'qkv' }, outputs: { out: 'attn' }, outputs_shape: { out: [1, 4, 8] }, params: { heads: 2, causal: true } },
      { opType: 'MatMul', inputs: { input: 'attn', weight: 'Wout' }, outputs: { out: 'o' }, outputs_shape: { out: [1, 4, 8] } },
      { opType: 'Add', inputs: { a: 'x', b: 'o' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 4, 8] } },
    ], outputs: ['y'] },

  // ---- full W8A8 island: QuantizeLinear → QConv2D → DequantizeLinear --------
  // int8 OHWI weight + per-axis weight scales + int32 bias + requant to int8.
  {
    id: 'qconv_island',
    inputs: { x: { shape: [1, 4, 4, 4], seed: 141 } },
    weights: {
      qs: { shape: [1], dtype: 'f32', values: [0.02] }, qz: { shape: [1], dtype: 'i8', values: [0] },
      w: { shape: [8, 3, 3, 4], dtype: 'i8', seed: 142, lo: -40, hi: 40 },
      ws: { shape: [8], dtype: 'f32', values: [0.011, 0.012, 0.008, 0.010, 0.009, 0.013, 0.010, 0.007] },
      wz: { shape: [8], dtype: 'i8', values: [0, 0, 0, 0, 0, 0, 0, 0] },
      b: { shape: [8], dtype: 'i32', seed: 143, lo: -4000, hi: 4000 },
      ds: { shape: [1], dtype: 'f32', values: [0.05] }, dz: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        xq: { scheme: 'per_tensor', scale_tensor: 'qs', zero_point_tensor: 'qz' },
        w: { scheme: 'per_axis', axis: 0, scale_tensor: 'ws', zero_point_tensor: 'wz' },
        v: { scheme: 'per_tensor', scale_tensor: 'ds', zero_point_tensor: 'dz' },
      },
    },
    nodes: [
      { opType: 'QuantizeLinear', inputs: { input: 'x', scale: 'qs', zero_point: 'qz' }, outputs: { out: 'xq' }, outputs_shape: { out: [1, 4, 4, 4] }, outputs_dtype: { out: 'int8' } },
      { opType: 'QConv2D', inputs: { input: 'xq', weight: 'w', bias: 'b' }, outputs: { out: 'v' }, outputs_shape: { out: [1, 2, 2, 8] }, outputs_dtype: { out: 'int8' },
        params: { stride: [1, 1], dilation: [1, 1], groups: 1, pads: [0, 0, 0, 0], padding: [0, 0], data_layout: 'NHWC', weight_layout: 'OHWI', relu: 2 } },
      { opType: 'DequantizeLinear', inputs: { input: 'v', scale: 'ds', zero_point: 'dz' }, outputs: { out: 'y' }, outputs_shape: { out: [1, 2, 2, 8] } },
    ],
    outputs: ['y'],
  },
];

export function authorGraph(dir, c) {
  ensureDir(dir);
  ensureDir(path.join(dir, 'inputs'));
  ensureDir(path.join(dir, 'weights'));
  const CTOR = { f32: Float32Array, i8: Int8Array, i32: Int32Array, u8: Uint8Array };
  // Seeded generation, dtype-aware: f32 uniform, or integer LCG for int8/int32 weights.
  const gen = (spec) => {
    const n = spec.shape.reduce((a, b) => a * b, 1);
    const dtype = spec.dtype || 'f32';
    if (spec.values) return CTOR[dtype].from(spec.values);
    if (dtype === 'f32') return seededFloat32(n, spec.seed, spec.lo ?? -1, spec.hi ?? 1);
    const lo = spec.lo ?? -100, hi = spec.hi ?? 100, span = hi - lo + 1;
    const out = new CTOR[dtype](n); let s = (spec.seed >>> 0) || 1;
    for (let k = 0; k < n; k++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; out[k] = lo + (s % span); }
    return out;
  };

  const inMap = {};
  const inShapes = {};
  for (const [name, spec] of Object.entries(c.inputs)) {
    const arr = gen(spec);
    inMap[name] = arr; inShapes[name] = spec.shape;
    writeTensor(path.join(dir, 'inputs', `${name}.f32`), 'f32', arr);
  }
  const st = {};
  const wShapes = {};
  const wDtypes = {};
  for (const [name, spec] of Object.entries(c.weights || {})) {
    const dtype = spec.dtype || 'f32';
    const arr = gen(spec);
    st[name] = { dtype, shape: spec.shape, data: arr };
    wShapes[name] = spec.shape;
    wDtypes[name] = dtype;
    writeTensor(path.join(dir, 'weights', `${name}.${dtype}`), dtype, arr);
  }
  const cfgInputs = {};
  for (const [name, spec] of Object.entries(c.inputs)) cfgInputs[name] = { shape: spec.shape, dtype: 'float32' };
  // Case declarations are a compact test-source DSL, not persisted graph
  // documents. The author always materializes the complete typed v1 surface.
  const nodes = c.nodes.map((node) => ({
    ...node,
    outputs_dtype: node.outputs_dtype ?? Object.fromEntries(
      Object.keys(node.outputs || {}).map((port) => [port, 'float32']),
    ),
  }));
  const cfg = {
    format: 'volvox-graph/v1',
    inputs: cfgInputs,
    nodes,
    ...(c.quantization ? { quantization: c.quantization } : {}),
    outputs: c.outputs,
  };
  writeGraph(path.join(dir, 'graph.json'), cfg);
  writeSafetensors(path.join(dir, 'model.safetensors'), st);
  const outputTensor = c.outputs[0];
  const outputNode = [...nodes].reverse().find((node) =>
    Object.values(node.outputs || {}).includes(outputTensor));
  const outputPort = outputNode && Object.entries(outputNode.outputs || {})
    .find(([, tensorName]) => tensorName === outputTensor)?.[0];
  const outputShape = outputPort ? outputNode.outputs_shape?.[outputPort] : null;
  const outputDtype = outputPort ? outputNode.outputs_dtype?.[outputPort] || 'float32' : null;
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({
    id: c.id, inputs: Object.keys(c.inputs), weights: Object.keys(c.weights || {}),
    inShapes, wShapes, wDtypes, output: 'y', outputShape, outputDtype,
    skip: c.skip || [],
  }));
  return { inputs: inMap };
}
