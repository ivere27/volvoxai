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

const perTensor = (scaleTensor, zeroPointTensor) => ({
  scheme: 'per_tensor',
  scale_tensor: scaleTensor,
  zero_point_tensor: zeroPointTensor,
});

const perAxis = (axis, scaleTensor, zeroPointTensor) => ({
  scheme: 'per_axis',
  axis,
  scale_tensor: scaleTensor,
  zero_point_tensor: zeroPointTensor,
});

const affineTable = (tensors) => ({
  format: 'volvox-affine-safetensors/v1',
  tensors,
});

// These are the exact numerical gaps found by intersecting the generated
// exporter-qualified inventories for cpu-js, wasm, webgpu, and native-cpu.
// Every entry is an executable graph case, not a registry/source-count claim.
// run.mjs binds this list back to the generated portable inventory and requires
// current CPU-JS/WASM/native artifacts; the physical WebGPU campaign imports the
// same authored packages and inputs.
export const portableClosureOperators = Object.freeze([
  'BatchMatMul',
  'Cast',
  'CrossSDPA',
  'Dropout',
  'Equal',
  'Gather',
  'Gemm',
  'GreaterOrEqual',
  'Not',
  'QArgMax',
  'QBatchMatMul',
  'QEmbedding',
  'QGELU',
  'QGemm',
  'QGroupNorm',
  'QLayerNorm',
  'QLinear',
  'QMaskedMean',
  'QMatMul',
  'QSDPA',
  'QSiLU',
  'Resize',
  'Split',
  'Where',
]);

export const portableClosureCases = [
  {
    id: 'portable_batch_matmul', portableOperator: 'BatchMatMul',
    inputs: {
      a: { shape: [2, 2, 3], seed: 201 },
      b: { shape: [1, 3, 4], seed: 202 },
    },
    weights: {},
    nodes: [{
      opType: 'BatchMatMul', inputs: { a: 'a', b: 'b' },
      outputs: { out: 'y' }, outputShapes: { out: [2, 2, 4] }, params: {},
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_gemm', portableOperator: 'Gemm',
    inputs: { x: { shape: [2, 3], seed: 203 } },
    weights: {
      w: { shape: [4, 3], seed: 204 },
      bias: { shape: [4], seed: 205, lo: -0.25, hi: 0.25 },
    },
    nodes: [{
      opType: 'Gemm', inputs: { input: 'x', weight: 'w', bias: 'bias' },
      outputs: { out: 'y' }, outputShapes: { out: [2, 4] },
      params: { weight_layout: 'dout_din' },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_equal', portableOperator: 'Equal', exact: true,
    inputs: {
      a: { shape: [2, 1], dtype: 'i32', values: [2, 3] },
      b: { shape: [3], dtype: 'i32', values: [2, 3, 4] },
    },
    weights: {},
    nodes: [{
      opType: 'Equal', inputs: { a: 'a', b: 'b' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 3] }, outputDtypes: { out: 'int32' }, params: {},
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_greater_or_equal', portableOperator: 'GreaterOrEqual', exact: true,
    inputs: {
      a: { shape: [2, 1], dtype: 'i32', values: [3, 1] },
      b: { shape: [3], dtype: 'i32', values: [1, 2, 3] },
    },
    weights: {},
    nodes: [{
      opType: 'GreaterOrEqual', inputs: { a: 'a', b: 'b' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 3] }, outputDtypes: { out: 'int32' }, params: {},
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_not', portableOperator: 'Not', exact: true,
    inputs: {
      x: { shape: [2, 4], dtype: 'i32', values: [0, 1, -1, 0, 7, 0, 0, -3] },
    },
    weights: {},
    nodes: [{
      opType: 'Not', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 4] }, outputDtypes: { out: 'int32' }, params: {},
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_cast_f32_i32', portableOperator: 'Cast', exact: true,
    inputs: {
      x: { shape: [2, 4], values: [-7, -2, 0, 1, 3, 8, 17, 127] },
    },
    weights: {},
    nodes: [{
      opType: 'Cast', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 4] }, outputDtypes: { out: 'int32' },
      params: { to: 'int32' },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_where', portableOperator: 'Where', exact: true,
    inputs: {
      condition: { shape: [2, 4], dtype: 'i32', values: [1, 0, -3, 0, 0, 2, 0, -1] },
      x: { shape: [2, 4], values: [1, 2, 3, 4, 5, 6, 7, 8] },
      z: { shape: [2, 4], values: [-1, -2, -3, -4, -5, -6, -7, -8] },
    },
    weights: {},
    nodes: [{
      opType: 'Where', inputs: { condition: 'condition', a: 'x', b: 'z' },
      outputs: { out: 'y' }, outputShapes: { out: [2, 4] }, params: {},
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_gather', portableOperator: 'Gather', exact: true,
    inputs: {
      x: { shape: [2, 3], values: [6, 8, 10, 12, 14, 16] },
      indices: { shape: [2], dtype: 'i32', values: [-1, 0] },
    },
    weights: {},
    nodes: [{
      opType: 'Gather', inputs: { input: 'x', indices: 'indices' },
      outputs: { out: 'y' }, outputShapes: { out: [2, 2] },
      params: { axis: 1 },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_cross_sdpa', portableOperator: 'CrossSDPA',
    inputs: {
      q: { shape: [1, 2, 2], values: [0.2, -0.1, -0.3, 0.6] },
      k: { shape: [1, 3, 2], values: [0.3, 0.4, 0.1, -0.5, -0.6, 0.3] },
      v: { shape: [1, 3, 2], values: [0.5, -0.2, 0.7, 0.2, -0.4, 0.9] },
    },
    weights: {},
    nodes: [{
      opType: 'CrossSDPA', inputs: { q: 'q', k: 'k', v: 'v' },
      outputs: { out: 'y' }, outputShapes: { out: [1, 2, 2] },
      params: { heads: 1, causal: false, scale: 0.37 },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_resize_linear', portableOperator: 'Resize',
    inputs: { x: { shape: [1, 2, 3, 2], seed: 206 } },
    weights: {},
    nodes: [{
      opType: 'Resize', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputShapes: { out: [1, 3, 5, 2] },
      params: {
        mode: 'linear', data_layout: 'NHWC',
        coordinate_transformation_mode: 'half_pixel',
        align_corners: false, antialias: false,
      },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_resize_nearest', portableOperator: 'Resize', exact: true,
    inputs: { x: { shape: [1, 2, 3, 2], seed: 209 } },
    weights: {},
    nodes: [{
      opType: 'Resize', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputShapes: { out: [1, 4, 5, 2] },
      params: {
        mode: 'nearest', data_layout: 'NHWC',
        coordinate_transformation_mode: 'asymmetric', nearest_mode: 'floor',
        align_corners: false, antialias: false,
      },
    }],
    outputs: ['y'],
  },
  {
    // The portable physical Split contract uses equal slices. This case
    // publishes the first output.
    id: 'portable_split_first', portableOperator: 'Split', exact: true,
    inputs: { x: { shape: [2, 4], seed: 207 } },
    weights: {},
    nodes: [{
      opType: 'Split', inputs: { input: 'x' },
      outputs: { out0: 'y', out1: 'unused' },
      outputShapes: { out0: [2, 2], out1: [2, 2] },
      params: { axis: 1, split: [2, 2] },
    }],
    outputs: ['y'],
  },
  {
    // The paired case publishes the other concrete output, so both declared
    // Split slices are compared without weakening the one-output matrix wire.
    id: 'portable_split_second', portableOperator: 'Split', exact: true,
    inputs: { x: { shape: [2, 4], seed: 207 } },
    weights: {},
    nodes: [{
      opType: 'Split', inputs: { input: 'x' },
      outputs: { out0: 'unused', out1: 'y' },
      outputShapes: { out0: [2, 2], out1: [2, 2] },
      params: { axis: 1, split: [2, 2] },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_dropout_inference', portableOperator: 'Dropout', exact: true,
    inputs: { x: { shape: [2, 8], seed: 208 } },
    weights: {},
    nodes: [{
      opType: 'Dropout', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 8] }, params: { ratio: 0.25, seed: 7 },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_qbatch_matmul', portableOperator: 'QBatchMatMul', exact: true,
    inputs: {
      a: {
        shape: [2, 2, 4], dtype: 'i8',
        values: [2, -1, 3, 0, -2, 4, 1, -3, 1, 2, -2, 3, 4, -1, 0, 2],
      },
      b: {
        shape: [1, 4, 3], dtype: 'i8',
        values: [1, -2, 3, 2, 1, -1, -3, 2, 1, 4, -1, 2],
      },
    },
    weights: {
      a_scale: { shape: [1], values: [0.25] },
      a_zero: { shape: [1], dtype: 'i8', values: [0] },
      b_scale: { shape: [1], values: [0.125] },
      b_zero: { shape: [1], dtype: 'i8', values: [0] },
      y_scale: { shape: [1], values: [0.25] },
      y_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({
      a: perTensor('a_scale', 'a_zero'),
      b: perTensor('b_scale', 'b_zero'),
      y: perTensor('y_scale', 'y_zero'),
    }),
    nodes: [{
      opType: 'QBatchMatMul', inputs: { a: 'a', b: 'b' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 2, 3] }, outputDtypes: { out: 'int8' }, params: {},
    }],
    outputs: ['y'],
  },
  ...['QGemm', 'QMatMul', 'QLinear'].map((opType, ordinal) => ({
    id: `portable_${opType.toLowerCase()}`,
    portableOperator: opType,
    exact: true,
    inputs: {
      x: {
        shape: [2, 4], dtype: 'i8',
        values: ordinal === 0
          ? [2, -1, 3, 0, -2, 4, 1, -3]
          : [1, 3, -2, 2, 4, -1, 0, -3],
      },
    },
    weights: {
      w: {
        shape: [3, 4], dtype: 'i8',
        values: [1, -2, 3, 0, -1, 2, 1, -3, 4, 0, -2, 1],
      },
      bias: { shape: [3], dtype: 'i32', values: [1, -2, 3] },
      x_scale: { shape: [1], values: [0.25] },
      x_zero: { shape: [1], dtype: 'i8', values: [0] },
      w_scale: { shape: [3], values: [0.125, 0.25, 0.0625] },
      w_zero: { shape: [3], dtype: 'i8', values: [0, 0, 0] },
      y_scale: { shape: [1], values: [0.25] },
      y_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({
      x: perTensor('x_scale', 'x_zero'),
      w: perAxis(0, 'w_scale', 'w_zero'),
      y: perTensor('y_scale', 'y_zero'),
    }),
    nodes: [{
      opType, inputs: { input: 'x', weight: 'w', bias: 'bias' },
      outputs: { out: 'y' }, outputShapes: { out: [2, 3] },
      outputDtypes: { out: 'int8' }, params: {},
    }],
    outputs: ['y'],
  })),
  ...['QGELU', 'QSiLU'].map((opType, ordinal) => ({
    id: `portable_${opType.toLowerCase()}`,
    portableOperator: opType,
    exact: true,
    inputs: {
      x: {
        shape: [2, 8], dtype: 'i8',
        values: [-16, -9, -4, -1, 0, 1, 4, 12, -12, -3, 2, 5, 8, 13, 20, 31],
      },
    },
    weights: {
      x_scale: { shape: [1], values: [ordinal === 0 ? 0.125 : 0.0625] },
      x_zero: { shape: [1], dtype: 'i8', values: [0] },
      y_scale: { shape: [1], values: [0.125] },
      y_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({
      x: perTensor('x_scale', 'x_zero'),
      y: perTensor('y_scale', 'y_zero'),
    }),
    nodes: [{
      opType, inputs: { input: 'x' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 8] }, outputDtypes: { out: 'int8' }, params: {},
    }],
    outputs: ['y'],
  })),
  {
    id: 'portable_qlayernorm', portableOperator: 'QLayerNorm', exact: true,
    inputs: {
      x: { shape: [2, 4], dtype: 'i8', values: [1, 5, -3, 7, -4, 2, 6, 0] },
    },
    weights: {
      gamma: { shape: [4], values: [1, 0.75, 1.25, 0.5] },
      beta: { shape: [4], values: [0.125, -0.125, 0.25, 0] },
      x_scale: { shape: [1], values: [0.125] },
      x_zero: { shape: [1], dtype: 'i8', values: [0] },
      y_scale: { shape: [1], values: [0.125] },
      y_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({
      x: perTensor('x_scale', 'x_zero'),
      y: perTensor('y_scale', 'y_zero'),
    }),
    nodes: [{
      opType: 'QLayerNorm', inputs: { input: 'x', weight: 'gamma', bias: 'beta' },
      outputs: { out: 'y' }, outputShapes: { out: [2, 4] },
      outputDtypes: { out: 'int8' }, params: { eps: 1e-5, d_model: 4 },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_qgroupnorm', portableOperator: 'QGroupNorm', exact: true,
    inputs: {
      x: {
        shape: [1, 2, 2, 4], dtype: 'i8',
        values: [1, 5, -3, 7, -4, 2, 6, 0, 3, -2, 4, 8, -5, 1, 7, -1],
      },
    },
    weights: {
      gamma: { shape: [4], values: [1, 0.75, 1.25, 0.5] },
      beta: { shape: [4], values: [0.125, -0.125, 0.25, 0] },
      x_scale: { shape: [1], values: [0.125] },
      x_zero: { shape: [1], dtype: 'i8', values: [0] },
      y_scale: { shape: [1], values: [0.125] },
      y_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({
      x: perTensor('x_scale', 'x_zero'),
      y: perTensor('y_scale', 'y_zero'),
    }),
    nodes: [{
      opType: 'QGroupNorm', inputs: { input: 'x', weight: 'gamma', bias: 'beta' },
      outputs: { out: 'y' }, outputShapes: { out: [1, 2, 2, 4] },
      outputDtypes: { out: 'int8' },
      params: { num_groups: 2, eps: 1e-5, data_layout: 'NHWC' },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_qmaskedmean', portableOperator: 'QMaskedMean', exact: true,
    inputs: {
      x: {
        shape: [2, 3, 4], dtype: 'i8',
        values: [
          1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4,
          4, 3, 2, 1, -4, -3, -2, -1, 8, 7, 6, 5,
        ],
      },
      mask: { shape: [2, 3], dtype: 'i32', values: [1, 0, 1, 0, 1, 1] },
    },
    weights: {
      x_scale: { shape: [1], values: [0.25] },
      x_zero: { shape: [1], dtype: 'i8', values: [0] },
      y_scale: { shape: [1], values: [0.25] },
      y_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({
      x: perTensor('x_scale', 'x_zero'),
      y: perTensor('y_scale', 'y_zero'),
    }),
    nodes: [{
      opType: 'QMaskedMean', inputs: { input: 'x', mask: 'mask' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 4] }, outputDtypes: { out: 'int8' }, params: {},
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_qargmax', portableOperator: 'QArgMax', exact: true,
    inputs: {
      x: { shape: [2, 4], dtype: 'i8', values: [3, 9, 9, -2, -4, -1, 7, 2] },
    },
    weights: {
      x_scale: { shape: [1], values: [0.125] },
      x_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({ x: perTensor('x_scale', 'x_zero') }),
    nodes: [{
      opType: 'QArgMax', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputShapes: { out: [2] }, outputDtypes: { out: 'int32' }, params: { axis: -1 },
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_qembedding', portableOperator: 'QEmbedding', exact: true,
    inputs: {
      ids: { shape: [2, 2], dtype: 'i32', values: [0, 3, 1, 4] },
    },
    weights: {
      table: {
        shape: [5, 4], dtype: 'i8',
        values: [
          1, 0, -1, 2, 2, 3, 0, -2, -3, 1, 4, 2,
          5, -2, 1, 0, -1, 4, 2, 3,
        ],
      },
      table_scale: { shape: [5], values: [0.25, 0.125, 0.5, 0.25, 0.125] },
      table_zero: { shape: [5], dtype: 'i8', values: [0, 0, 0, 0, 0] },
      y_scale: { shape: [1], values: [0.125] },
      y_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({
      table: perAxis(0, 'table_scale', 'table_zero'),
      y: perTensor('y_scale', 'y_zero'),
    }),
    nodes: [{
      opType: 'QEmbedding', inputs: { input: 'ids', weight: 'table' }, outputs: { out: 'y' },
      outputShapes: { out: [2, 2, 4] }, outputDtypes: { out: 'int8' }, params: {},
    }],
    outputs: ['y'],
  },
  {
    id: 'portable_qsdpa', portableOperator: 'QSDPA', exact: true,
    inputs: {
      q: {
        shape: [1, 3, 4], dtype: 'i8',
        values: [1, 2, -1, 0, 2, -1, 3, 1, -2, 1, 0, 3],
      },
      k: {
        shape: [1, 3, 4], dtype: 'i8',
        values: [2, 0, -1, 1, -1, 3, 1, 0, 1, -2, 2, 3],
      },
      v: {
        shape: [1, 3, 4], dtype: 'i8',
        values: [4, 1, -2, 3, -3, 2, 4, -1, 1, -4, 2, 5],
      },
      mask: { shape: [1, 3], dtype: 'i32', values: [1, 1, 0] },
    },
    weights: {
      q_scale: { shape: [1], values: [0.125] },
      q_zero: { shape: [1], dtype: 'i8', values: [0] },
      k_scale: { shape: [1], values: [0.125] },
      k_zero: { shape: [1], dtype: 'i8', values: [0] },
      v_scale: { shape: [1], values: [0.25] },
      v_zero: { shape: [1], dtype: 'i8', values: [0] },
      y_scale: { shape: [1], values: [0.25] },
      y_zero: { shape: [1], dtype: 'i8', values: [0] },
    },
    quantization: affineTable({
      q: perTensor('q_scale', 'q_zero'),
      k: perTensor('k_scale', 'k_zero'),
      v: perTensor('v_scale', 'v_zero'),
      y: perTensor('y_scale', 'y_zero'),
    }),
    nodes: [{
      opType: 'QSDPA', inputs: { q: 'q', k: 'k', v: 'v', mask: 'mask' },
      outputs: { out: 'y' }, outputShapes: { out: [1, 3, 4] },
      outputDtypes: { out: 'int8' }, params: { heads: 1, causal: false, scale: 0.5 },
    }],
    outputs: ['y'],
  },
];

// Several cases carry skip: ['native-vulkan', 'native-opengl']. Those graphs route
// Linear/MatMul/PReLU or grouped convolution, which the native GPU backends cannot
// execute without a CPU operator fallback (the strict matrix forbids one). Verified
// against a pristine binary: a pre-existing coverage gap, not a regression.
export const cases = [
  {
    id: 'linear_gelu',
    skip: ['native-vulkan', 'native-opengl'],
    inputs: { x: { shape: [1, B, 16], seed: 11 } },
    weights: { W1: { shape: [16, 32], seed: 12 } },
    nodes: [
      { opType: 'MatMul', inputs: { input: 'x', weight: 'W1' }, outputs: { out: 'h' }, outputShapes: { out: [1, B, 32] } },
      { opType: 'GELU', inputs: { input: 'h' }, outputs: { out: 'y' }, outputShapes: { out: [1, B, 32] } },
    ],
    outputs: ['y'],
  },
  {
    id: 'mlp',
    skip: ['native-vulkan', 'native-opengl'],
    inputs: { x: { shape: [1, B, 16], seed: 21 } },
    weights: { W1: { shape: [16, 32], seed: 22 }, W2: { shape: [32, 16], seed: 23 } },
    nodes: [
      { opType: 'MatMul', inputs: { input: 'x', weight: 'W1' }, outputs: { out: 'h1' }, outputShapes: { out: [1, B, 32] } },
      { opType: 'GELU', inputs: { input: 'h1' }, outputs: { out: 'a' }, outputShapes: { out: [1, B, 32] } },
      { opType: 'MatMul', inputs: { input: 'a', weight: 'W2' }, outputs: { out: 'y' }, outputShapes: { out: [1, B, 16] } },
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
      { opType: 'GELU', inputs: { input: 'x' }, outputs: { out: 'g' }, outputShapes: { out: [1, B, 16] } },
      { opType: 'Add', inputs: { a: 'g', b: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [1, B, 16] } },
    ],
    outputs: ['y'],
  },
  {
    // norm → linear: a layout/normalization boundary feeding a matmul.
    id: 'layernorm_linear',
    skip: ['native-vulkan', 'native-opengl'],
    inputs: { x: { shape: [1, B, 16], seed: 41 } },
    weights: { g: { shape: [16], seed: 42, lo: 0.5, hi: 1.5 }, bta: { shape: [16], seed: 43, lo: -0.2, hi: 0.2 }, W: { shape: [16, 24], seed: 44 } },
    nodes: [
      { opType: 'LayerNorm', inputs: { input: 'x', weight: 'g', bias: 'bta' }, outputs: { out: 'n' }, outputShapes: { out: [1, B, 16] }, params: { eps: 1e-5, d_model: 16 } },
      { opType: 'MatMul', inputs: { input: 'n', weight: 'W' }, outputs: { out: 'y' }, outputShapes: { out: [1, B, 24] } },
    ],
    outputs: ['y'],
  },

  // Single-op cases added via the L2 machinery (weights/params/shape changes) and
  // scored by the generic torch interpreter. Reductions/softmax are last-axis.
  { id: 'softmax', inputs: { x: { shape: [8, 16], seed: 51 } }, weights: {},
    nodes: [{ opType: 'Softmax', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [8, 16] } }], outputs: ['y'] },
  { id: 'logsoftmax', inputs: { x: { shape: [8, 16], seed: 52 } }, weights: {},
    nodes: [{ opType: 'LogSoftmax', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [8, 16] } }], outputs: ['y'] },
  { id: 'reducemean', inputs: { x: { shape: [8, 16], seed: 53 } }, weights: {},
    nodes: [{ opType: 'ReduceMean', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [8] }, params: { axis: -1, keepdims: false } }], outputs: ['y'] },
  { id: 'reducesum', inputs: { x: { shape: [8, 16], seed: 54 } }, weights: {},
    nodes: [{ opType: 'ReduceSum', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [8] }, params: { axis: -1, keepdims: false } }], outputs: ['y'] },
  { id: 'rmsnorm', inputs: { x: { shape: [8, 16], seed: 55 } }, weights: { weight: { shape: [16], seed: 56, lo: 0.5, hi: 1.5 } },
    nodes: [{ opType: 'RMSNorm', inputs: { input: 'x', weight: 'weight' }, outputs: { out: 'y' }, outputShapes: { out: [8, 16] }, params: { eps: 1e-5, d_model: 16 } }], outputs: ['y'] },
  { id: 'transpose', inputs: { x: { shape: [4, 8], seed: 57 } }, weights: {},
    nodes: [{ opType: 'Transpose', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [8, 4] }, params: { perm: [1, 0] } }], outputs: ['y'] },
  { id: 'concat', inputs: { in0: { shape: [4, 8], seed: 58 }, in1: { shape: [4, 8], seed: 59 } }, weights: {},
    nodes: [{ opType: 'Concat', inputs: { input0: 'in0', input1: 'in1' }, outputs: { out: 'y' }, outputShapes: { out: [8, 8] }, params: { axis: 0 } }], outputs: ['y'] },
  { id: 'prelu', inputs: { x: { shape: [4, 8], seed: 61 } }, weights: { slope: { shape: [8], seed: 62, lo: -0.3, hi: 0.3 } }, skip: ['native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'PReLU', inputs: { input: 'x', slope: 'slope' }, outputs: { out: 'y' }, outputShapes: { out: [4, 8] } }], outputs: ['y'] },

  // Vision ops — NHWC [N,H,W,C]. GlobalAveragePool over H,W; BatchNorm/GroupNorm per channel.
  { id: 'globalavgpool', inputs: { x: { shape: [1, 4, 4, 8], seed: 71 } }, weights: {},
    nodes: [{ opType: 'GlobalAveragePool', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [1, 1, 1, 8] } }], outputs: ['y'] },
  { id: 'batchnorm', inputs: { x: { shape: [1, 4, 4, 8], seed: 72 } },
    weights: { weight: { shape: [8], seed: 73, lo: 0.5, hi: 1.5 }, bias: { shape: [8], seed: 74, lo: -0.3, hi: 0.3 }, running_mean: { shape: [8], seed: 75, lo: -0.5, hi: 0.5 }, running_var: { shape: [8], seed: 76, lo: 0.5, hi: 1.5 } },
    nodes: [{ opType: 'BatchNorm2D', inputs: { input: 'x', weight: 'weight', bias: 'bias', running_mean: 'running_mean', running_var: 'running_var' }, outputs: { out: 'y' }, outputShapes: { out: [1, 4, 4, 8] }, params: { eps: 1e-5 } }], outputs: ['y'] },
  { id: 'groupnorm', inputs: { x: { shape: [1, 4, 4, 8], seed: 77 } },
    weights: { weight: { shape: [8], seed: 78, lo: 0.5, hi: 1.5 }, bias: { shape: [8], seed: 79, lo: -0.3, hi: 0.3 } },
    nodes: [{ opType: 'GroupNorm', inputs: { input: 'x', weight: 'weight', bias: 'bias' }, outputs: { out: 'y' }, outputShapes: { out: [1, 4, 4, 8] }, params: { num_groups: 2, eps: 1e-5 } }], outputs: ['y'] },
  { id: 'linear', inputs: { x: { shape: [1, 8, 16], seed: 81 } }, weights: { W: { shape: [16, 32], seed: 82 }, b: { shape: [32], seed: 83 } }, skip: ['native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'Linear', inputs: { input: 'x', weight: 'W', bias: 'b' }, outputs: { out: 'y' }, outputShapes: { out: [1, 8, 32] } }], outputs: ['y'] },
  { id: 'avgpool2d', inputs: { x: { shape: [1, 4, 4, 8], seed: 84 } }, weights: {}, skip: ['native-cpu'],
    nodes: [{ opType: 'AveragePool2D', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [1, 2, 2, 8] }, params: { kernel: [2, 2], stride: [2, 2], padding: [0, 0] } }], outputs: ['y'] },
  { id: 'upsample', inputs: { x: { shape: [1, 2, 2, 8], seed: 85 } }, weights: {},
    nodes: [{ opType: 'UpsampleNearest2D', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [1, 4, 4, 8] } }], outputs: ['y'] },
  { id: 'pad', inputs: { x: { shape: [1, 4, 4, 8], seed: 86 } }, weights: {}, skip: ['native-cpu'],
    nodes: [{ opType: 'Pad', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [1, 6, 6, 8] }, params: { pads: [0, 1, 1, 0, 0, 1, 1, 0], value: 0 } }], outputs: ['y'] },
  { id: 'meanheight', inputs: { x: { shape: [1, 4, 4, 8], seed: 91 } }, weights: {}, skip: ['native-cpu'],
    nodes: [{ opType: 'MeanHeight', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [1, 8, 4] } }], outputs: ['y'] },
  { id: 'argmax', inputs: { x: { shape: [8, 16], seed: 92 } }, weights: {}, skip: ['native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'ArgMax', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [8] }, outputDtypes: { out: 'int32' }, params: { axis: 1, keepdims: false, select_last_index: 0 } }], outputs: ['y'] },

  // Shape/index family. Flatten/Squeeze/Unsqueeze/Identity are shape-only aliases
  // (copy) supported on every portable tier. Native CPU now also executes
  // ArgMax, Slice, and Expand; only strict native GPU routes remain skipped.
  { id: 'flatten', inputs: { x: { shape: [2, 3, 4], seed: 111 } }, weights: {},
    nodes: [{ opType: 'Flatten', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [2, 12] }, params: { axis: 1 } }], outputs: ['y'] },
  { id: 'squeeze', inputs: { x: { shape: [1, 4, 8], seed: 112 } }, weights: {},
    nodes: [{ opType: 'Squeeze', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [4, 8] }, params: { axes: [0] } }], outputs: ['y'] },
  { id: 'unsqueeze', inputs: { x: { shape: [4, 8], seed: 113 } }, weights: {},
    nodes: [{ opType: 'Unsqueeze', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [1, 4, 8] }, params: { axes: [0] } }], outputs: ['y'] },
  { id: 'identity', inputs: { x: { shape: [4, 8], seed: 114 } }, weights: {},
    nodes: [{ opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [4, 8] } }], outputs: ['y'] },
  { id: 'slice', inputs: { x: { shape: [4, 8], seed: 115 } }, weights: {}, skip: ['native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'Slice', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [4, 4] }, params: { starts: [2], ends: [6], axes: [1], steps: [1] } }], outputs: ['y'] },
  { id: 'expand', inputs: { x: { shape: [4, 1], seed: 116 } }, weights: {}, skip: ['native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'Expand', inputs: { input: 'x' }, outputs: { out: 'y' }, outputShapes: { out: [4, 8] }, params: { shape: [4, 8] } }], outputs: ['y'] },

  // ---- F32 fusion / composition mixed graphs (the real L2 seams) -----------
  // SwiGLU: gate & up share x (fan-out); SiLU-gated multiply.
  { id: 'swiglu', inputs: { x: { shape: [1, 8, 16], seed: 101 } }, weights: { Wg: { shape: [16, 32], seed: 102 }, Wu: { shape: [16, 32], seed: 103 } }, skip: ['native-vulkan', 'native-opengl'],
    nodes: [
      { opType: 'MatMul', inputs: { input: 'x', weight: 'Wg' }, outputs: { out: 'g' }, outputShapes: { out: [1, 8, 32] } },
      { opType: 'SiLU', inputs: { input: 'g' }, outputs: { out: 'sg' }, outputShapes: { out: [1, 8, 32] } },
      { opType: 'MatMul', inputs: { input: 'x', weight: 'Wu' }, outputs: { out: 'u' }, outputShapes: { out: [1, 8, 32] } },
      { opType: 'Mul', inputs: { a: 'sg', b: 'u' }, outputs: { out: 'y' }, outputShapes: { out: [1, 8, 32] } },
    ], outputs: ['y'] },
  // Chained residual adds.
  { id: 'residual_chain', inputs: { a: { shape: [1, 8, 16], seed: 104 }, b: { shape: [1, 8, 16], seed: 105 }, c: { shape: [1, 8, 16], seed: 106 } }, weights: {},
    nodes: [
      { opType: 'Add', inputs: { a: 'a', b: 'b' }, outputs: { out: 'h' }, outputShapes: { out: [1, 8, 16] } },
      { opType: 'Add', inputs: { a: 'h', b: 'c' }, outputs: { out: 'y' }, outputShapes: { out: [1, 8, 16] } },
    ], outputs: ['y'] },
  // Concat → Sigmoid (detector class-head tail fusion). Ends in Sigmoid, so WASM/native
  // inherit the documented fast-sigmoid approximation (~8e-3); torch matches exact pure-JS.
  { id: 'concat_sigmoid', inputs: { in0: { shape: [4, 8], seed: 107 }, in1: { shape: [4, 8], seed: 108 } }, weights: {}, approxTol: { atol: 1.2e-2 },
    nodes: [
      { opType: 'Concat', inputs: { input0: 'in0', input1: 'in1' }, outputs: { out: 'h' }, outputShapes: { out: [8, 8] }, params: { axis: 0 } },
      { opType: 'Sigmoid', inputs: { input: 'h' }, outputs: { out: 'y' }, outputShapes: { out: [8, 8] } },
    ], outputs: ['y'] },
  // Conv2D with fused ReLU6 (the #1 vision fusion pattern). NHWC in, canonical HWIO weight.
  { id: 'conv_relu', inputs: { x: { shape: [1, 6, 6, 4], seed: 109 } }, weights: { w: { shape: [3, 3, 4, 8], seed: 110 }, b: { shape: [8], seed: 111 } },
    nodes: [{ opType: 'Conv2D', inputs: { input: 'x', weight: 'w', bias: 'b' }, outputs: { out: 'y' }, outputShapes: { out: [1, 4, 4, 8] },
      params: { stride: [1, 1], dilation: [1, 1], groups: 1, pads: [0, 0, 0, 0], data_layout: 'NHWC', weight_layout: 'HWIO', relu: 2 } }], outputs: ['y'] },

  // Conv2D in the layout the microkernels actually index: HWIO for regular and
  // grouped convolution. This is what the exporter emits, so it is the path
  // every tier takes at runtime.
  { id: 'conv_hwio', inputs: { x: { shape: [1, 6, 6, 4], seed: 112 } }, weights: { w: { shape: [3, 3, 4, 8], seed: 113 }, b: { shape: [8], seed: 114 } },
    nodes: [{ opType: 'Conv2D', inputs: { input: 'x', weight: 'w', bias: 'b' }, outputs: { out: 'y' }, outputShapes: { out: [1, 4, 4, 8] },
      params: { stride: [1, 1], dilation: [1, 1], groups: 1, pads: [0, 0, 0, 0], padding: [0, 0], data_layout: 'NHWC', weight_layout: 'HWIO' } }], outputs: ['y'] },
  // Grouped conv, which the OHWI reader rejected outright.
  // groups>1 has no native-GPU route (the device conv entry points take no
  // groups argument), so Vulkan/OpenGL fall back to CPU -- a coverage gap, not
  // a layout failure. WebGPU carries the grouped path.
  { id: 'conv_hwio_grouped', inputs: { x: { shape: [1, 5, 5, 8], seed: 115 } }, weights: { w: { shape: [3, 3, 4, 8], seed: 116 }, b: { shape: [8], seed: 117 } }, skip: ['native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'Conv2D', inputs: { input: 'x', weight: 'w', bias: 'b' }, outputs: { out: 'y' }, outputShapes: { out: [1, 5, 5, 8] },
      params: { stride: [1, 1], dilation: [1, 1], groups: 2, pads: [1, 1, 1, 1], padding: [1, 1], data_layout: 'NHWC', weight_layout: 'HWIO' } }], outputs: ['y'] },
  // Depthwise in HWCM [kh,kw,C,M] -- the shape that selects the depthwise kernels.
  { id: 'conv_hwcm_depthwise', inputs: { x: { shape: [1, 6, 6, 4], seed: 118 } }, weights: { w: { shape: [3, 3, 4, 2], seed: 119 }, b: { shape: [8], seed: 120 } },
    nodes: [{ opType: 'Conv2D', inputs: { input: 'x', weight: 'w', bias: 'b' }, outputs: { out: 'y' }, outputShapes: { out: [1, 6, 6, 8] },
      params: { stride: [1, 1], dilation: [1, 1], groups: 4, pads: [1, 1, 1, 1], padding: [1, 1], data_layout: 'NHWC', weight_layout: 'HWCM' } }], outputs: ['y'] },
  // Conv1D on NLC activations with WIO weights, and ConvTranspose2D on HWIO.
  { id: 'conv1d_nlc', inputs: { x: { shape: [2, 9, 3], seed: 131 } }, weights: { w: { shape: [3, 3, 6], seed: 132 }, b: { shape: [6], seed: 133 } }, skip: ['native-cpu'],
    nodes: [{ opType: 'Conv1D', inputs: { input: 'x', weight: 'w', bias: 'b' }, outputs: { out: 'y' }, outputShapes: { out: [2, 5, 6] },
      params: { stride: 2, padding: 1, groups: 1, relu: 1 } }], outputs: ['y'] },
  { id: 'conv1d_grouped', inputs: { x: { shape: [1, 8, 4], seed: 134 } }, weights: { w: { shape: [5, 2, 8], seed: 135 }, b: { shape: [8], seed: 136 } }, skip: ['native-cpu', 'native-vulkan', 'native-opengl'],
    nodes: [{ opType: 'Conv1D', inputs: { input: 'x', weight: 'w', bias: 'b' }, outputs: { out: 'y' }, outputShapes: { out: [1, 8, 8] },
      params: { stride: 1, padding: 2, groups: 2 } }], outputs: ['y'] },
  { id: 'conv_transpose_hwio', inputs: { x: { shape: [2, 3, 4, 3], seed: 137 } }, weights: { w: { shape: [3, 3, 3, 5], seed: 138 }, b: { shape: [5], seed: 139 } }, skip: ['native-cpu'],
    nodes: [{ opType: 'ConvTranspose2D', inputs: { input: 'x', weight: 'w', bias: 'b' }, outputs: { out: 'y' }, outputShapes: { out: [2, 5, 7, 5] },
      params: { kernel: [3, 3], stride: [2, 2], padding: [1, 1] } }], outputs: ['y'] },

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
      { opType: 'QuantizeLinear', inputs: { input: 'x', scale: 'qs', zero_point: 'qz' }, outputs: { out: 'xq' }, outputShapes: { out: [1, 8, 16] }, outputDtypes: { out: 'int8' } },
      { opType: 'DequantizeLinear', inputs: { input: 'xq', scale: 'qs', zero_point: 'qz' }, outputs: { out: 'y' }, outputShapes: { out: [1, 8, 16] } },
    ], outputs: ['y'] },

  // ---- attention: isolated SDPA + a full transformer block ------------------
  { id: 'sdpa', inputs: { qkv: { shape: [1, 4, 24], seed: 131 } }, weights: {},
    nodes: [{ opType: 'SDPA', inputs: { qkv: 'qkv' }, outputs: { out: 'y' }, outputShapes: { out: [1, 4, 8] }, params: { heads: 2, causal: true } }], outputs: ['y'] },
  { id: 'attn_block', inputs: { x: { shape: [1, 4, 8], seed: 132 } }, skip: ['native-vulkan', 'native-opengl'],
    weights: { g: { shape: [8], seed: 133, lo: 0.5, hi: 1.5 }, bta: { shape: [8], seed: 134, lo: -0.2, hi: 0.2 }, Wqkv: { shape: [8, 24], seed: 135 }, Wout: { shape: [8, 8], seed: 136 } },
    nodes: [
      { opType: 'LayerNorm', inputs: { input: 'x', weight: 'g', bias: 'bta' }, outputs: { out: 'n' }, outputShapes: { out: [1, 4, 8] }, params: { eps: 1e-5, d_model: 8 } },
      { opType: 'MatMul', inputs: { input: 'n', weight: 'Wqkv' }, outputs: { out: 'qkv' }, outputShapes: { out: [1, 4, 24] } },
      { opType: 'SDPA', inputs: { qkv: 'qkv' }, outputs: { out: 'attn' }, outputShapes: { out: [1, 4, 8] }, params: { heads: 2, causal: true } },
      { opType: 'MatMul', inputs: { input: 'attn', weight: 'Wout' }, outputs: { out: 'o' }, outputShapes: { out: [1, 4, 8] } },
      { opType: 'Add', inputs: { a: 'x', b: 'o' }, outputs: { out: 'y' }, outputShapes: { out: [1, 4, 8] } },
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
      { opType: 'QuantizeLinear', inputs: { input: 'x', scale: 'qs', zero_point: 'qz' }, outputs: { out: 'xq' }, outputShapes: { out: [1, 4, 4, 4] }, outputDtypes: { out: 'int8' } },
      { opType: 'QConv2D', inputs: { input: 'xq', weight: 'w', bias: 'b' }, outputs: { out: 'v' }, outputShapes: { out: [1, 2, 2, 8] }, outputDtypes: { out: 'int8' },
        params: { stride: [1, 1], dilation: [1, 1], groups: 1, pads: [0, 0, 0, 0], padding: [0, 0], data_layout: 'NHWC', weight_layout: 'OHWI', relu: 2 } },
      { opType: 'DequantizeLinear', inputs: { input: 'v', scale: 'ds', zero_point: 'dz' }, outputs: { out: 'y' }, outputShapes: { out: [1, 2, 2, 8] } },
    ],
    outputs: ['y'],
  },
  ...portableClosureCases,
];

export function authorGraph(dir, c) {
  ensureDir(dir);
  ensureDir(path.join(dir, 'inputs'));
  ensureDir(path.join(dir, 'weights'));
  const CTOR = { f32: Float32Array, i8: Int8Array, i32: Int32Array, u8: Uint8Array };
  const GRAPH_DTYPE = { f32: 'float32', i8: 'int8', i32: 'int32', u8: 'uint8' };
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
  const inDtypes = {};
  for (const [name, spec] of Object.entries(c.inputs)) {
    const dtype = spec.dtype || 'f32';
    const arr = gen(spec);
    inMap[name] = arr; inShapes[name] = spec.shape; inDtypes[name] = dtype;
    writeTensor(path.join(dir, 'inputs', `${name}.${dtype}`), dtype, arr);
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
  for (const [name, spec] of Object.entries(c.inputs)) {
    const dtype = spec.dtype || 'f32';
    cfgInputs[name] = { shape: spec.shape, dtype: GRAPH_DTYPE[dtype] };
  }
  // Case declarations are a compact test-source DSL, not persisted graph
  // documents. The author always materializes the complete typed v1 surface.
  const nodes = c.nodes.map((node, index) => ({
    id: `${c.id}.node.${index}`,
    opType: node.opType,
    inputs: { ...node.inputs },
    outputs: Object.fromEntries(Object.entries(node.outputs || {}).map(
      ([port, tensor]) => [port, {
        tensor,
        shape: [...node.outputShapes[port]],
        dtype: node.outputDtypes?.[port] ?? 'float32',
      }],
    )),
    params: {
      ...((node.opType === 'MatMul' || node.opType === 'Linear' || node.opType === 'Gemm') &&
          node.inputs?.weight && node.params?.weight_layout === undefined
        ? { weight_layout: 'din_dout' }
        : {}),
      ...(node.params || {}),
    },
  }));
  const cfg = {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: cfgInputs,
    nodes,
    ...(c.quantization ? { quantization: c.quantization } : {}),
    outputs: c.outputs,
  };
  writeGraph(path.join(dir, 'graph.json'), cfg);
  writeSafetensors(path.join(dir, 'model.safetensors'), st);
  const outputTensor = c.outputs[0];
  const outputNode = [...c.nodes].reverse().find((node) =>
    Object.values(node.outputs || {}).includes(outputTensor));
  const outputPort = outputNode && Object.entries(outputNode.outputs || {})
    .find(([, tensorName]) => tensorName === outputTensor)?.[0];
  const outputShape = outputPort ? outputNode.outputShapes?.[outputPort] : null;
  const outputDtype = outputPort ? outputNode.outputDtypes?.[outputPort] || 'float32' : null;
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({
    id: c.id, inputs: Object.keys(c.inputs), weights: Object.keys(c.weights || {}),
    inShapes, inDtypes, wShapes, wDtypes, output: outputTensor, outputShape, outputDtype,
    skip: c.skip || [],
  }));
  return { inputs: inMap };
}
