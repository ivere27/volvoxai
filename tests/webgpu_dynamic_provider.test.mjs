import test from 'node:test';
import assert from 'node:assert/strict';

import { Runtime } from '../ts/core/ContextRuntime.js';
import { createBoundExecutionGraph } from '../ts/core/BoundExecutionGraph.js';
import { Model } from '../ts/core/Model.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { resolveGraphShapes } from '../ts/core/ResolvedShapePlan.js';
import { createBackendCompileInput } from '../ts/backends/BackendProvider.js';
import { preflightWebGPUExecutionInputs } from '../ts/backends/WebGPUDispatch.js';
import { WebGPUBackendProvider } from '../ts/backends/WebGPUBackendProvider.js';
import {
  checkedWebGPUPhysicalDomain,
  WEBGPU_CANONICAL_PHYSICAL_OPERATORS,
} from '../ts/backends/WebGPUPhysicalDomain.js';
import { WebGPUEngine } from '../ts/backends/WebGPUEngine.js';
import { CPUBackendProvider } from '../ts/backends/CPUBackendProvider.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import {
  operatorShapeContract,
  runtimeOperatorsByBackend,
} from '../ts/generated/kernelRegistry.js';

globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  STORAGE: 8,
  UNIFORM: 16,
});
globalThis.GPUMapMode ??= Object.freeze({ READ: 1 });

const ShaderLibrary = {
  getReLUShader: () => 'dynamic-relu',
  getQLinearShader: () => 'qlinear-scalar',
  getQLinearTiledShader: () => 'qlinear-tiled',
  getQLinearDotShader: () => 'qlinear-dot',
  getQLinearDotTiledShader: () => 'qlinear-dot-tiled',
  getTypedCopyShader: () => 'typed-copy',
  getCopy32Shader: () => 'copy32',
  getConcatCopyShader: () => 'concat-copy',
  getBroadcastAddShader: () => 'dynamic-add',
  getBroadcastSubShader: () => 'dynamic-sub',
  getBroadcastBinaryShader: (operation) =>
    operation.includes('av - bv') ? 'dynamic-sub' : 'dynamic-add',
  getLinearF32Shader: () => 'linear-f32-output-major',
  getLinearF32TiledShader: () => 'linear-f32-output-major-tiled',
  getLinearF32RowMajorShader: () => 'linear-f32-row-major',
  getLinearF32RowMajorTiledShader: () => 'linear-f32-row-major-tiled',
  getEmbeddingShader: () => 'embedding',
  getQBatchMatMulShader: () => 'qbatch-matmul',
  getQBatchMatMulDotShader: () => 'qbatch-matmul-dot',
  getQMaskedMeanShader: () => 'qmasked-mean',
  getQSDPAShader: () => 'qsdpa',
  getRequantizeLinearShader: () => 'requantize-linear',
  getConv2DShader: () => 'conv2d-scalar',
  getConv2DRegularC3Out16Shader: () => 'conv2d-c3-out16',
  getConv2DRegularOut16Shader: () => 'conv2d-regular-out16',
  getConv2DDepthwise8Shader: () => 'conv2d-depthwise8',
  getConv2DPointwise16TileShader: () => 'conv2d-pointwise16-tile',
  getConv2DPointwise8Vec4Shader: () => 'conv2d-pointwise8-vec4',
  getConv2DPointwise8Vec2Shader: () => 'conv2d-pointwise8-vec2',
};

function dynamicReluSnapshot({
  batchMax = 2,
  sequenceMax = 130,
  sequenceMultipleOf = 1,
  outputNames = ['y'],
  opType = 'ReLU',
} = {}) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: batchMax },
      S: { min: 1, max: sequenceMax, multiple_of: sequenceMultipleOf },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 'S'] } },
    nodes: [{
      id: 'relu',
      opType,
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 'S'] } },
      params: {},
    }],
    outputs: outputNames,
  }, []);
  return Model.capture({ graph, weights: {} });
}

function typedData(dtype, values) {
  if (dtype === 'float32') return Float32Array.from(values);
  if (dtype === 'int32') return Int32Array.from(values);
  if (dtype === 'int8') return Int8Array.from(values);
  return Uint8Array.from(values);
}

function dynamicQLinearSnapshot({
  biasValue = 0,
  sequenceMax = null,
  unrelatedSequenceMax = null,
} = {}) {
  const rowShape = Number.isSafeInteger(sequenceMax) && sequenceMax > 1;
  const unrelatedShape = Number.isSafeInteger(unrelatedSequenceMax) &&
    unrelatedSequenceMax > 1;
  const dimensions = unrelatedShape
    ? { P: { min: 1, max: unrelatedSequenceMax } }
    : rowShape
      ? { S: { min: 1, max: sequenceMax } }
      : { B: { min: 1, max: 9 } };
  const inputShape = unrelatedShape ? [1, 16] : rowShape ? [1, 'S', 16] : ['B', 16];
  const outputShape = unrelatedShape ? [1, 32] : rowShape ? [1, 'S', 32] : ['B', 32];
  const inputs = { x: { dtype: 'int8', shape: inputShape } };
  if (unrelatedShape) inputs.past = { dtype: 'float32', shape: [1, 'P'] };
  const nodes = [{
    id: 'qlinear',
    opType: 'QLinear',
    inputs: { input: 'x', weight: 'w', bias: 'bias' },
    outputs: { out: { tensor: 'y', dtype: 'int8', shape: outputShape } },
    params: {},
  }];
  if (unrelatedShape) {
    nodes.push({
      id: 'cache_relu', opType: 'ReLU', inputs: { input: 'past' },
      outputs: { out: { tensor: 'present', dtype: 'float32', shape: [1, 'P'] } },
      params: {},
    });
  }
  const document = {
    format: 'volvox-graph/v1',
    dimensions,
    inputs,
    nodes,
    outputs: unrelatedShape ? ['y', 'present'] : ['y'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        x: { scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx' },
        w: { scheme: 'per_axis', axis: 0, scale_tensor: 'sw', zero_point_tensor: 'zw' },
        y: { scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy' },
      },
    },
  };
  const definitions = [
    { name: 'w', dtype: 'int8', shape: [32, 16], values: new Array(512).fill(1) },
    { name: 'bias', dtype: 'int32', shape: [32], values: new Array(32).fill(biasValue) },
    { name: 'sx', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zx', dtype: 'int8', shape: [1], values: [0] },
    { name: 'sw', dtype: 'float32', shape: [32], values: new Array(32).fill(1) },
    { name: 'zw', dtype: 'int8', shape: [32], values: new Array(32).fill(0) },
    { name: 'sy', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zy', dtype: 'int8', shape: [1], values: [0] },
  ];
  const graph = parseGraphDocument(
    document,
    definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
  );
  const weights = Object.fromEntries(definitions.map((definition) => [definition.name, {
    name: definition.name,
    dtype: definition.dtype,
    shape: definition.shape,
    data: typedData(definition.dtype, definition.values),
  }]));
  return Model.capture({
    graph,
    weights,
    quantizationByTensor: {
      x: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      w: {
        scheme: 'per_axis', axis: 0,
        scales: new Array(32).fill(1), zero_points: new Array(32).fill(0),
      },
      y: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    },
  });
}

function dynamicQConvScalarFallbackSnapshot() {
  const document = {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'uint8', shape: ['B', 3, 3, 16] } },
    nodes: [{
      id: 'qconv', opType: 'QConv2D',
      inputs: { input: 'x', weight: 'w', bias: 'bias' },
      outputs: { out: { tensor: 'y', dtype: 'int8', shape: ['B', 3, 3, 16] } },
      params: {
        data_layout: 'NHWC', weight_layout: 'OHWI', groups: 1,
        stride: [1, 1], dilation: [1, 1], pads: [0, 0, 0, 0],
      },
    }],
    outputs: ['y'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        x: { scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx' },
        w: { scheme: 'per_axis', axis: 0, scale_tensor: 'sw', zero_point_tensor: 'zw' },
        y: { scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy' },
      },
    },
  };
  const definitions = [
    { name: 'w', dtype: 'int8', shape: [16, 1, 1, 16], values: new Array(256).fill(1) },
    { name: 'bias', dtype: 'int32', shape: [16], values: new Array(16).fill(0) },
    { name: 'sx', dtype: 'float32', shape: [1], values: [0.25] },
    { name: 'zx', dtype: 'uint8', shape: [1], values: [128] },
    { name: 'sw', dtype: 'float32', shape: [16], values: new Array(16).fill(0.5) },
    { name: 'zw', dtype: 'int8', shape: [16], values: new Array(16).fill(0) },
    { name: 'sy', dtype: 'float32', shape: [1], values: [0.125] },
    { name: 'zy', dtype: 'int8', shape: [1], values: [0] },
  ];
  const graph = parseGraphDocument(
    document,
    definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
  );
  return Model.capture({
    graph,
    weights: Object.fromEntries(definitions.map((definition) => [definition.name, {
      name: definition.name,
      dtype: definition.dtype,
      shape: definition.shape,
      data: typedData(definition.dtype, definition.values),
    }])),
    quantizationByTensor: {
      x: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
      w: {
        scheme: 'per_axis', axis: 0,
        scales: new Array(16).fill(0.5), zero_points: new Array(16).fill(0),
      },
      y: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
    },
  });
}

function twoLayerDynamicQLinearSnapshot() {
  const document = {
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 1, max: 3 } },
    inputs: { x: { dtype: 'int8', shape: [1, 'S', 16] } },
    nodes: [
      {
        id: 'qlinear1', opType: 'QLinear',
        inputs: { input: 'x', weight: 'w1', bias: 'bias1' },
        outputs: { out: { tensor: 'hidden', dtype: 'int8', shape: [1, 'S', 32] } },
        params: {},
      },
      {
        id: 'qlinear2', opType: 'QLinear',
        inputs: { input: 'hidden', weight: 'w2', bias: 'bias2' },
        outputs: { out: { tensor: 'y', dtype: 'int8', shape: [1, 'S', 16] } },
        params: {},
      },
    ],
    outputs: ['y'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        x: { scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx' },
        w1: { scheme: 'per_axis', axis: 0, scale_tensor: 'sw1', zero_point_tensor: 'zw1' },
        hidden: { scheme: 'per_tensor', scale_tensor: 'sh', zero_point_tensor: 'zh' },
        w2: { scheme: 'per_axis', axis: 0, scale_tensor: 'sw2', zero_point_tensor: 'zw2' },
        y: { scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy' },
      },
    },
  };
  const definitions = [
    { name: 'w1', dtype: 'int8', shape: [32, 16], values: new Array(512).fill(1) },
    { name: 'bias1', dtype: 'int32', shape: [32], values: new Array(32).fill(0) },
    { name: 'w2', dtype: 'int8', shape: [16, 32], values: new Array(512).fill(1) },
    { name: 'bias2', dtype: 'int32', shape: [16], values: new Array(16).fill(0) },
    { name: 'sx', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zx', dtype: 'int8', shape: [1], values: [0] },
    { name: 'sw1', dtype: 'float32', shape: [32], values: new Array(32).fill(1) },
    { name: 'zw1', dtype: 'int8', shape: [32], values: new Array(32).fill(0) },
    { name: 'sh', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zh', dtype: 'int8', shape: [1], values: [0] },
    { name: 'sw2', dtype: 'float32', shape: [16], values: new Array(16).fill(1) },
    { name: 'zw2', dtype: 'int8', shape: [16], values: new Array(16).fill(0) },
    { name: 'sy', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zy', dtype: 'int8', shape: [1], values: [0] },
  ];
  const graph = parseGraphDocument(
    document,
    definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
  );
  const weights = Object.fromEntries(definitions.map((definition) => [definition.name, {
    name: definition.name,
    dtype: definition.dtype,
    shape: definition.shape,
    data: typedData(definition.dtype, definition.values),
  }]));
  return Model.capture({
    graph,
    weights,
    quantizationByTensor: {
      x: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      w1: {
        scheme: 'per_axis', axis: 0,
        scales: new Array(32).fill(1), zero_points: new Array(32).fill(0),
      },
      hidden: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      w2: {
        scheme: 'per_axis', axis: 0,
        scales: new Array(16).fill(1), zero_points: new Array(16).fill(0),
      },
      y: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    },
  });
}

function dynamicAddSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 1, max: 4 } },
    inputs: {
      a: { dtype: 'float32', shape: ['S'] },
      b: { dtype: 'float32', shape: ['S'] },
    },
    nodes: [{
      id: 'add', opType: 'Add', inputs: { a: 'a', b: 'b' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['S'] } }, params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function dynamicAdd2DSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 1, max: 130 } },
    inputs: {
      a: { dtype: 'float32', shape: [1, 'S'] },
      b: { dtype: 'float32', shape: [1, 'S'] },
    },
    nodes: [{
      id: 'add', opType: 'Add', inputs: { a: 'a', b: 'b' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 'S'] } }, params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function embeddingValueSnapshot(origin, values = [0, 2]) {
  const inputs = {};
  const definitions = [
    { name: 'table', dtype: 'float32', shape: [3, 2], values: [1, 2, 3, 4, 5, 6] },
  ];
  const nodes = [];
  let ids = 'ids';
  if (origin === 'public') {
    inputs.ids = { dtype: 'int32', shape: [2] };
  } else if (origin === 'weight') {
    definitions.push({ name: 'ids', dtype: 'int32', shape: [2], values });
  } else if (origin === 'clip') {
    inputs.raw = { dtype: 'int32', shape: [2] };
    nodes.push({
      id: 'clip_ids', opType: 'Clip', inputs: { input: 'raw' },
      outputs: { out: { tensor: 'ids', dtype: 'int32', shape: [2] } },
      params: { min: 0, max: 2 },
    });
  } else if (origin === 'argmax') {
    inputs.logits = { dtype: 'float32', shape: [2, 3] };
    nodes.push({
      id: 'argmax_ids', opType: 'ArgMax', inputs: { input: 'logits' },
      outputs: { out: { tensor: 'ids', dtype: 'int32', shape: [2] } },
      params: { axis: 1, keepdims: false },
    });
  } else {
    inputs.raw = { dtype: 'float32', shape: [2] };
    nodes.push({
      id: 'unsafe_cast_ids', opType: 'Cast', inputs: { input: 'raw' },
      outputs: { out: { tensor: 'ids', dtype: 'int32', shape: [2] } },
      params: { to: 'int32' },
    });
  }
  nodes.push({
    id: 'embedding', opType: 'Embedding', inputs: { input: ids, weight: 'table' },
    outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2, 2] } }, params: {},
  });
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: {}, inputs, nodes, outputs: ['y'],
  }, definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })));
  return Model.capture({
    graph,
    weights: Object.fromEntries(definitions.map((definition) => [definition.name, {
      name: definition.name, dtype: definition.dtype, shape: definition.shape,
      data: typedData(definition.dtype, definition.values),
    }])),
  });
}

function dynamicEmbeddingSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: { S: { min: 1, max: 3 } },
    inputs: { ids: { dtype: 'int32', shape: ['S'] } },
    nodes: [{
      id: 'embedding', opType: 'Embedding', inputs: { input: 'ids', weight: 'table' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['S', 1] } }, params: {},
    }],
    outputs: ['y'],
  }, [{ name: 'table', dtype: 'float32', shape: [3, 1] }]);
  return Model.capture({
    graph,
    weights: {
      table: {
        name: 'table', dtype: 'float32', shape: [3, 1],
        data: Float32Array.from([10, 20, 30]),
      },
    },
  });
}

function dynamicQuantizedPredicateSnapshot() {
  const document = {
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 2 },
      S: { min: 1, max: 3 },
      Q: { min: 1, max: 2 },
      K: { min: 1, max: 3 },
    },
    inputs: {
      batch_a: { dtype: 'int8', shape: ['B', 'Q', 4] },
      batch_b: { dtype: 'int8', shape: [1, 4, 3] },
      tokens: { dtype: 'int8', shape: ['B', 'S', 4] },
      keep: { dtype: 'int32', shape: ['B', 'S'] },
      q: { dtype: 'int8', shape: ['B', 'Q', 8] },
      k: { dtype: 'int8', shape: ['B', 'K', 8] },
      v: { dtype: 'int8', shape: ['B', 'K', 8] },
      attention_mask: { dtype: 'int32', shape: ['B', 'Q', 'K'] },
      requant_input: { dtype: 'int8', shape: ['B', 'S'] },
    },
    nodes: [
      {
        id: 'batch', opType: 'QBatchMatMul',
        inputs: { a: 'batch_a', b: 'batch_b' },
        outputs: {
          out: { tensor: 'batch_out', dtype: 'int8', shape: ['B', 'Q', 3] },
        },
        params: {},
      },
      {
        id: 'mean', opType: 'QMaskedMean',
        inputs: { input: 'tokens', mask: 'keep' },
        outputs: { out: { tensor: 'mean_out', dtype: 'int8', shape: ['B', 4] } },
        params: {},
      },
      {
        id: 'attention', opType: 'QSDPA',
        inputs: { q: 'q', k: 'k', v: 'v', mask: 'attention_mask' },
        outputs: {
          out: { tensor: 'attention_out', dtype: 'int8', shape: ['B', 'Q', 8] },
        },
        params: { heads: 2, causal: false, scale: 0.5 },
      },
      {
        id: 'requant', opType: 'RequantizeLinear',
        inputs: { input: 'requant_input' },
        outputs: {
          out: { tensor: 'requant_out', dtype: 'uint8', shape: ['B', 'S'] },
        },
        params: {},
      },
    ],
    outputs: ['batch_out', 'mean_out', 'attention_out', 'requant_out'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        batch_a: { scheme: 'per_tensor', scale_tensor: 'scale_in', zero_point_tensor: 'zero_i8' },
        batch_b: { scheme: 'per_tensor', scale_tensor: 'scale_in', zero_point_tensor: 'zero_i8' },
        batch_out: { scheme: 'per_tensor', scale_tensor: 'scale_out', zero_point_tensor: 'zero_i8' },
        tokens: { scheme: 'per_tensor', scale_tensor: 'scale_in', zero_point_tensor: 'zero_i8' },
        mean_out: { scheme: 'per_tensor', scale_tensor: 'scale_out', zero_point_tensor: 'zero_i8' },
        q: { scheme: 'per_tensor', scale_tensor: 'scale_in', zero_point_tensor: 'zero_i8' },
        k: { scheme: 'per_tensor', scale_tensor: 'scale_in', zero_point_tensor: 'zero_i8' },
        v: { scheme: 'per_tensor', scale_tensor: 'scale_in', zero_point_tensor: 'zero_i8' },
        attention_out: { scheme: 'per_tensor', scale_tensor: 'scale_out', zero_point_tensor: 'zero_i8' },
        requant_input: { scheme: 'per_tensor', scale_tensor: 'scale_in', zero_point_tensor: 'zero_i8' },
        requant_out: { scheme: 'per_tensor', scale_tensor: 'scale_out', zero_point_tensor: 'zero_u8' },
      },
    },
  };
  const definitions = [
    { name: 'scale_in', dtype: 'float32', shape: [1], values: [0.5] },
    { name: 'scale_out', dtype: 'float32', shape: [1], values: [0.25] },
    { name: 'zero_i8', dtype: 'int8', shape: [1], values: [0] },
    { name: 'zero_u8', dtype: 'uint8', shape: [1], values: [128] },
  ];
  const graph = parseGraphDocument(
    document,
    definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
  );
  return Model.capture({
    graph,
    weights: Object.fromEntries(definitions.map((definition) => [definition.name, {
      name: definition.name,
      dtype: definition.dtype,
      shape: definition.shape,
      data: typedData(definition.dtype, definition.values),
    }])),
    quantizationByTensor: Object.freeze({
      batch_a: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
      batch_b: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
      batch_out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      tokens: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
      mean_out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      q: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
      k: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
      v: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
      attention_out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      requant_input: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
      requant_out: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
    }),
  });
}

function dynamicRegularConvSnapshot() {
  const weightData = Float32Array.from({ length: 3 * 3 * 5 * 16 }, (_, index) =>
    ((index % 11) - 5) / 16);
  const biasData = Float32Array.from({ length: 16 }, (_, index) => (index - 8) / 32);
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 2 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 4, 4, 5] } },
    nodes: [{
      id: 'conv', opType: 'Conv2D',
      inputs: { input: 'x', weight: 'weight', bias: 'bias' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 4, 4, 16] } },
      params: {
        groups: 1, weight_layout: 'HWIO', stride: [1, 1],
        pads: [1, 1, 1, 1], dilation: [1, 1],
      },
    }],
    outputs: ['y'],
  }, [
    { name: 'weight', dtype: 'float32', shape: [3, 3, 5, 16] },
    { name: 'bias', dtype: 'float32', shape: [16] },
  ]);
  return Model.capture({
    graph,
    weights: {
      weight: {
        name: 'weight', dtype: 'float32', shape: [3, 3, 5, 16],
        data: weightData,
      },
      bias: {
        name: 'bias', dtype: 'float32', shape: [16],
        data: biasData,
      },
    },
  });
}

function resourceBankSnapshot(banked) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: banked ? { F: { min: 1, max: 4 } } : {},
    ...(banked ? { banks: { experts: 'F' } } : {}),
    inputs: { slot: { dtype: 'int32', shape: [1] } },
    nodes: [{
      id: 'pick', opType: 'Gather', inputs: { input: 'experts', indices: 'slot' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 1] } },
      params: { axis: 0 },
    }],
    outputs: ['y'],
  }, [{ name: 'experts', dtype: 'float32', shape: [4, 1] }]);
  return Model.capture({
    graph,
    weights: {
      experts: {
        name: 'experts', dtype: 'float32', shape: [4, 1],
        data: Float32Array.of(10, 20, 30, 40),
      },
    },
  });
}

function dynamicResourceBankSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: {
      F: { min: 1, max: 4 },
      S: { min: 1, max: 3 },
    },
    banks: { experts: 'F' },
    inputs: { slot: { dtype: 'int32', shape: ['S'] } },
    nodes: [{
      id: 'pick', opType: 'Gather', inputs: { input: 'experts', indices: 'slot' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['S', 1] } },
      params: { axis: 0 },
    }],
    outputs: ['y'],
  }, [{ name: 'experts', dtype: 'float32', shape: [4, 1] }]);
  return Model.capture({
    graph,
    weights: {
      experts: {
        name: 'experts', dtype: 'float32', shape: [4, 1],
        data: Float32Array.of(10, 20, 30, 40),
      },
    },
  });
}

function invariantBankGatherSnapshot(slot = 0) {
  const definitions = [
    { name: 'experts', dtype: 'float32', shape: [4, 1] },
    { name: 'slot', dtype: 'int32', shape: [1] },
  ];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: { F: { min: 1, max: 4 } }, banks: { experts: 'F' }, inputs: {},
    nodes: [{
      id: 'pick', opType: 'Gather', inputs: { input: 'experts', indices: 'slot' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 1] } },
      params: { axis: 0 },
    }],
    outputs: ['y'],
  }, definitions);
  return Model.capture({
    graph,
    weights: {
      experts: {
        ...definitions[0], data: Float32Array.of(10, 20, 30, 40),
      },
      slot: { ...definitions[1], data: Int32Array.of(slot) },
    },
  });
}

function deviceProducedBankGatherSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: { F: { min: 1, max: 4 } }, banks: { experts: 'F' },
    inputs: { raw_slot: { dtype: 'int32', shape: [1] } },
    nodes: [
      {
        id: 'bounded_slot', opType: 'Clip', inputs: { input: 'raw_slot' },
        outputs: { out: { tensor: 'slot', dtype: 'int32', shape: [1] } },
        params: { min: 0, max: 3 },
      },
      {
        id: 'pick', opType: 'Gather', inputs: { input: 'experts', indices: 'slot' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 1] } },
        params: { axis: 0 },
      },
    ],
    outputs: ['y'],
  }, [{ name: 'experts', dtype: 'float32', shape: [4, 1] }]);
  return Model.capture({
    graph,
    weights: {
      experts: {
        name: 'experts', dtype: 'float32', shape: [4, 1],
        data: Float32Array.of(10, 20, 30, 40),
      },
    },
  });
}

function invariantBankMoESnapshot(expert = 0) {
  const definitions = [
    { name: 'experts', dtype: 'float32', shape: [4, 2, 1] },
    { name: 'route_indices', dtype: 'float32', shape: [1, 1] },
    { name: 'route_weights', dtype: 'float32', shape: [1, 1] },
  ];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: { F: { min: 1, max: 4 } }, banks: { experts: 'F' },
    inputs: { x: { dtype: 'float32', shape: [1, 2] } },
    nodes: [{
      id: 'mix', opType: 'MoELinear',
      inputs: {
        input: 'x', expert_weight: 'experts',
        route_indices: 'route_indices', route_weights: 'route_weights',
      },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 1] } },
      params: {},
    }],
    outputs: ['y'],
  }, definitions);
  return Model.capture({
    graph,
    weights: {
      experts: {
        ...definitions[0], data: Float32Array.of(1, 1, 2, 2, 3, 3, 4, 4),
      },
      route_indices: { ...definitions[1], data: Float32Array.of(expert) },
      route_weights: { ...definitions[2], data: Float32Array.of(1) },
    },
  });
}

function gatherValueSnapshot(origin) {
  const inputs = {
    data: { dtype: 'float32', shape: [3] },
    raw: { dtype: origin === 'clip' ? 'int32' : 'float32', shape: [2] },
  };
  const producer = origin === 'clip'
    ? {
      id: 'bounded_indices', opType: 'Clip', inputs: { input: 'raw' },
      outputs: { out: { tensor: 'indices', dtype: 'int32', shape: [2] } },
      params: { min: -3, max: 2 },
    }
    : {
      id: 'unproved_indices', opType: 'Cast', inputs: { input: 'raw' },
      outputs: { out: { tensor: 'indices', dtype: 'int32', shape: [2] } },
      params: { to: 'int32' },
    };
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: {}, inputs,
    nodes: [producer, {
      id: 'gather', opType: 'Gather', inputs: { input: 'data', indices: 'indices' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2] } },
      params: { axis: 0 },
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function moeValueSnapshot(internalRoutes) {
  const inputs = { x: { dtype: 'float32', shape: [1, 2] } };
  const nodes = [];
  const definitions = [
    { name: 'experts', dtype: 'float32', shape: [2, 2, 1], values: [1, 2, 3, 4] },
  ];
  if (internalRoutes) {
    definitions.push({ name: 'router', dtype: 'float32', shape: [2, 2], values: [1, 0, 0, 1] });
    nodes.push({
      id: 'router', opType: 'MoERouter', inputs: { input: 'x', weight: 'router' },
      outputs: {
        indices: { tensor: 'route_indices', dtype: 'float32', shape: [1, 1] },
        weights: { tensor: 'route_weights', dtype: 'float32', shape: [1, 1] },
      },
      params: { num_experts: 2, top_k: 1 },
    });
  } else {
    inputs.route_indices = { dtype: 'float32', shape: [1, 1] };
    inputs.route_weights = { dtype: 'float32', shape: [1, 1] };
  }
  nodes.push({
    id: 'moe', opType: 'MoELinear',
    inputs: {
      input: 'x', expert_weight: 'experts',
      route_indices: 'route_indices', route_weights: 'route_weights',
    },
    outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 1] } }, params: {},
  });
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1', dimensions: {}, inputs, nodes, outputs: ['y'],
  }, definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })));
  return Model.capture({
    graph,
    weights: Object.fromEntries(definitions.map((definition) => [definition.name, {
      name: definition.name, dtype: definition.dtype, shape: definition.shape,
      data: typedData(definition.dtype, definition.values),
    }])),
  });
}

function alternatingBroadcastSubSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      A: { min: 1, max: 3 },
      B: { min: 1, max: 3 },
    },
    inputs: {
      a: { dtype: 'float32', shape: ['A', 'B'] },
      b: { dtype: 'float32', shape: [1, 'B'] },
    },
    nodes: [{
      id: 'sub', opType: 'Sub', inputs: { a: 'a', b: 'b' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['A', 'B'] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function unequalSplitSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [4] } },
    nodes: [{
      id: 'split', opType: 'Split', inputs: { input: 'x' },
      outputs: {
        out0: { tensor: 'left', dtype: 'float32', shape: [1] },
        out1: { tensor: 'right', dtype: 'float32', shape: [3] },
      },
      params: { axis: 0, split: [1, 3] },
    }],
    outputs: ['left', 'right'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function inputMajorMatMulSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 19] } },
    nodes: [{
      id: 'matmul', opType: 'MatMul', inputs: { input: 'x', weight: 'w' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 23] } }, params: {},
    }],
    outputs: ['y'],
  }, [{ name: 'w', dtype: 'float32', shape: [19, 23] }]);
  return Model.capture({
    graph,
    weights: {
      w: {
        name: 'w', dtype: 'float32', shape: [19, 23],
        data: new Float32Array(19 * 23).fill(1),
      },
    },
  });
}

function outputAliasLinearSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      // O is output-only. Its declared interval contains the producer's fixed
      // width but must not replace the canonical O=3 proof relation.
      O: { min: 1, max: 1_000 },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 4] } },
    nodes: [{
      id: 'linear', opType: 'Linear',
      inputs: { input: 'x', weight: 'w', bias: 'bias' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 'O'] } },
      params: {},
    }],
    outputs: ['y'],
  }, [
    { name: 'w', dtype: 'float32', shape: [3, 4] },
    { name: 'bias', dtype: 'float32', shape: [3] },
  ]);
  return Model.capture({
    graph,
    weights: {
      w: {
        name: 'w', dtype: 'float32', shape: [3, 4],
        data: Float32Array.from([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
        ]),
      },
      bias: {
        name: 'bias', dtype: 'float32', shape: [3],
        data: Float32Array.from([1, 2, 3]),
      },
    },
  });
}

function affineConcatSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      Q: { min: 1, max: 4 },
      // The canonical producer relation, not a request-local concrete value,
      // supplies the output extent used by the resource proof.
      M: { min: 3, max: 6 },
    },
    inputs: {
      prefix: { dtype: 'float32', shape: [1, 2, 1] },
      suffix: { dtype: 'float32', shape: [1, 'Q', 1] },
    },
    nodes: [{
      id: 'concat', opType: 'Concat',
      inputs: { input0: 'prefix', input1: 'suffix' },
      outputs: { out: { tensor: 'joined', dtype: 'float32', shape: [1, 'M', 1] } },
      params: { axis: 1 },
    }],
    outputs: ['joined'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function unsupportedWideMoERouterSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 2] } },
    nodes: [{
      id: 'router', opType: 'MoERouter',
      inputs: { input: 'x', weight: 'router_weight' },
      outputs: {
        indices: { tensor: 'indices', dtype: 'float32', shape: ['B', 9] },
        weights: { tensor: 'weights', dtype: 'float32', shape: ['B', 9] },
      },
      params: { num_experts: 16, top_k: 9 },
    }],
    outputs: ['indices', 'weights'],
  }, [{ name: 'router_weight', dtype: 'float32', shape: [2, 16] }]);
  return Model.capture({
    graph,
    weights: {
      router_weight: {
        name: 'router_weight', dtype: 'float32', shape: [2, 16],
        data: new Float32Array(32),
      },
    },
  });
}

function invalidStaticQLayerNormAffineSnapshot() {
  const document = {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 2 } },
    inputs: { x: { dtype: 'int8', shape: ['B', 4] } },
    nodes: [{
      id: 'norm', opType: 'QLayerNorm',
      inputs: { input: 'x', weight: 'gamma', bias: 'beta' },
      outputs: { out: { tensor: 'y', dtype: 'int8', shape: ['B', 4] } },
      params: { eps: 1e-5, d_model: 4 },
    }],
    outputs: ['y'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        x: { scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx' },
        y: { scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy' },
      },
    },
  };
  const definitions = [
    { name: 'gamma', dtype: 'float32', shape: [4], values: [1, Number.NaN, 1, 1] },
    { name: 'beta', dtype: 'float32', shape: [4], values: [0, 0, 0, 0] },
    { name: 'sx', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zx', dtype: 'int8', shape: [1], values: [0] },
    { name: 'sy', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zy', dtype: 'int8', shape: [1], values: [0] },
  ];
  const graph = parseGraphDocument(
    document,
    definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
  );
  return Model.capture({
    graph,
    weights: Object.fromEntries(definitions.map((definition) => [definition.name, {
      name: definition.name,
      dtype: definition.dtype,
      shape: definition.shape,
      data: typedData(definition.dtype, definition.values),
    }])),
    quantizationByTensor: {
      x: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      y: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    },
  });
}

function mockDevice(limitOverrides = {}) {
  let loseDevice;
  const state = {
    buffers: [],
    bindGroups: [],
    pipelineCreates: 0,
    dispatches: [],
    submissions: [],
    workResolvers: [],
    failLabel: null,
    failWriteLabel: null,
    failWriteAt: null,
    writeCount: 0,
    failNextSubmit: false,
    failNextUnlabeledBuffer: false,
    failPipelineCode: null,
    failNextBindGroup: false,
    autoCompleteWork: false,
  };
  const device = {
    state,
    limits: {
      maxBufferSize: 1024 * 1024,
      maxStorageBufferBindingSize: 1024 * 1024,
      maxUniformBufferBindingSize: 64 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxBindingsPerBindGroup: 8,
      maxBindGroups: 4,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBuffersPerShaderStage: 12,
      ...limitOverrides,
    },
    lost: new Promise((resolve) => { loseDevice = resolve; }),
    createBuffer(descriptor) {
      if (state.failNextUnlabeledBuffer && descriptor.label === undefined) {
        state.failNextUnlabeledBuffer = false;
        throw new Error('injected unlabeled allocation failure');
      }
      if (state.failLabel === descriptor.label) {
        state.failLabel = null;
        throw new Error(`injected allocation failure for ${descriptor.label}`);
      }
      const buffer = {
        descriptor,
        size: descriptor.size,
        bytes: new Uint8Array(descriptor.size),
        destroyed: false,
        async mapAsync() {},
        getMappedRange() { return this.bytes.buffer.slice(0); },
        unmap() {},
        destroy() { this.destroyed = true; },
      };
      state.buffers.push(buffer);
      return buffer;
    },
    createShaderModule({ code }) { return { code }; },
    async createComputePipelineAsync({ compute }) {
      if (state.failPipelineCode === compute.module.code) {
        throw new Error(`injected pipeline failure for ${compute.module.code}`);
      }
      state.pipelineCreates++;
      return {
        code: compute.module.code,
        getBindGroupLayout() { return {}; },
      };
    },
    createBindGroup(descriptor) {
      if (state.failNextBindGroup) {
        state.failNextBindGroup = false;
        throw new Error('injected bind-group specialization failure');
      }
      state.bindGroups.push(descriptor);
      return descriptor;
    },
    createCommandEncoder() {
      const operations = [];
      return {
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          operations.push({ type: 'copy', source, sourceOffset, destination, destinationOffset, size });
        },
        beginComputePass() {
          let pipeline;
          let bindGroup;
          return {
            setPipeline(value) { pipeline = value; },
            setBindGroup(_index, value) { bindGroup = value; },
            dispatchWorkgroups(x, y, z) {
              operations.push({ type: 'dispatch', pipeline, bindGroup, x, y, z });
            },
            end() {},
          };
        },
        finish() { return { operations }; },
      };
    },
    queue: {
      writeBuffer(destination, offset, source, sourceOffset = 0, size = undefined) {
        state.writeCount++;
        if (state.failWriteAt === state.writeCount) {
          state.failWriteAt = null;
          throw new Error(`injected write failure at ${state.writeCount}`);
        }
        if (state.failWriteLabel === destination.descriptor?.label) {
          state.failWriteLabel = null;
          throw new Error(`injected write failure for ${destination.descriptor.label}`);
        }
        const view = ArrayBuffer.isView(source);
        const sourceBuffer = view ? source.buffer : source;
        const unitBytes = view && !(source instanceof DataView)
          ? source.BYTES_PER_ELEMENT
          : 1;
        const relativeOffset = sourceOffset * unitBytes;
        const baseOffset = (view ? source.byteOffset : 0) + relativeOffset;
        const availableBytes = view ? source.byteLength : source.byteLength;
        const byteLength = size === undefined
          ? availableBytes - relativeOffset
          : size * unitBytes;
        if (relativeOffset < 0 || byteLength < 0 ||
            relativeOffset + byteLength > availableBytes) {
          throw new RangeError('mock GPUQueue.writeBuffer source range exceeds its view');
        }
        destination.bytes.set(new Uint8Array(sourceBuffer, baseOffset, byteLength), offset);
      },
      submit(commandBuffers) {
        if (state.failNextSubmit) {
          state.failNextSubmit = false;
          throw new Error('injected submit failure');
        }
        state.submissions.push(commandBuffers.flatMap((buffer) => buffer.operations));
        for (const commandBuffer of commandBuffers) {
          for (const operation of commandBuffer.operations) {
            if (operation.type === 'copy') {
              operation.destination.bytes.set(operation.source.bytes.subarray(
                operation.sourceOffset,
                operation.sourceOffset + operation.size,
              ), operation.destinationOffset);
              continue;
            }
            state.dispatches.push(operation);
            if (operation.pipeline.code !== 'dynamic-relu') continue;
            const entries = new Map(operation.bindGroup.entries.map((entry) => [
              entry.binding,
              entry.resource.buffer,
            ]));
            const input = new Float32Array(entries.get(0).bytes.buffer);
            const output = new Float32Array(entries.get(1).bytes.buffer);
            const elements = new Uint32Array(entries.get(2).bytes.buffer)[0];
            for (let index = 0; index < elements; index++) {
              output[index] = Math.max(0, input[index]);
            }
            continue;
          }
          for (const operation of commandBuffer.operations) {
            if (operation.type !== 'dispatch' ||
                !operation.pipeline.code.startsWith('qlinear-')) continue;
            const entries = new Map(operation.bindGroup.entries.map((entry) => [
              entry.binding,
              entry.resource.buffer,
            ]));
            const input = new Int8Array(entries.get(0).bytes.buffer);
            const weight = new Int8Array(entries.get(1).bytes.buffer);
            const multipliers = new Float32Array(entries.get(2).bytes.buffer);
            const weightZeroPoints = new Int32Array(entries.get(3).bytes.buffer);
            const bias = new Int32Array(entries.get(4).bytes.buffer);
            const output = new Int8Array(entries.get(5).bytes.buffer);
            const params = entries.get(6).bytes.buffer;
            const pu = new Uint32Array(params);
            const pi = new Int32Array(params);
            const [rows, dIn, dOut] = pu;
            for (let row = 0; row < rows; row++) {
              for (let channel = 0; channel < dOut; channel++) {
                let accumulator = bias[channel];
                for (let feature = 0; feature < dIn; feature++) {
                  accumulator += (input[row * dIn + feature] - pi[8]) *
                    (weight[channel * dIn + feature] - weightZeroPoints[channel]);
                }
                output[row * dOut + channel] = Math.max(-128, Math.min(127,
                  Math.round(accumulator * multipliers[channel]) + pi[9]));
              }
            }
          }
          for (const operation of commandBuffer.operations) {
            if (operation.type !== 'dispatch' ||
                operation.pipeline.code !== 'linear-f32-output-major') continue;
            const entries = new Map(operation.bindGroup.entries.map((entry) => [
              entry.binding,
              entry.resource.buffer,
            ]));
            const input = new Float32Array(entries.get(0).bytes.buffer);
            const weight = new Float32Array(entries.get(1).bytes.buffer);
            const bias = new Float32Array(entries.get(3).bytes.buffer);
            const output = new Float32Array(entries.get(4).bytes.buffer);
            const [rows, dIn, dOut] = new Uint32Array(entries.get(5).bytes.buffer);
            for (let row = 0; row < rows; row++) {
              for (let channel = 0; channel < dOut; channel++) {
                let value = bias[channel];
                for (let feature = 0; feature < dIn; feature++) {
                  value += input[row * dIn + feature] * weight[channel * dIn + feature];
                }
                output[row * dOut + channel] = value;
              }
            }
          }
          for (const operation of commandBuffer.operations) {
            if (operation.type !== 'dispatch' ||
                operation.pipeline.code !== 'concat-copy') continue;
            const entries = new Map(operation.bindGroup.entries.map((entry) => [
              entry.binding,
              entry.resource.buffer,
            ]));
            const input = new Float32Array(entries.get(0).bytes.buffer);
            const output = new Float32Array(entries.get(1).bytes.buffer);
            const [elements, axisOffset, inputAxis, outputAxis, inner] =
              new Uint32Array(entries.get(2).bytes.buffer);
            for (let index = 0; index < elements; index++) {
              const innerIndex = index % inner;
              const axisIndex = Math.floor(index / inner) % inputAxis;
              const outerIndex = Math.floor(index / (inner * inputAxis));
              const outputIndex = (outerIndex * outputAxis + axisOffset + axisIndex) * inner +
                innerIndex;
              output[outputIndex] = input[index];
            }
          }
          for (const operation of commandBuffer.operations) {
            if (operation.type !== 'dispatch' ||
                !['dynamic-add', 'dynamic-sub'].includes(operation.pipeline.code)) continue;
            const entries = new Map(operation.bindGroup.entries.map((entry) => [
              entry.binding,
              entry.resource.buffer,
            ]));
            const a = new Float32Array(entries.get(0).bytes.buffer);
            const b = new Float32Array(entries.get(1).bytes.buffer);
            const output = new Float32Array(entries.get(2).bytes.buffer);
            const metadata = new Uint32Array(entries.get(3).bytes.buffer);
            const total = metadata[0];
            const rank = metadata[1];
            for (let index = 0; index < total; index++) {
              let remaining = index;
              let aOffset = 0;
              let bOffset = 0;
              for (let axis = 0; axis < rank; axis++) {
                const outputStride = metadata[2 + axis];
                const coordinate = Math.floor(remaining / outputStride);
                remaining %= outputStride;
                aOffset += coordinate * metadata[2 + rank + axis];
                bOffset += coordinate * metadata[2 + 2 * rank + axis];
              }
              output[index] = operation.pipeline.code === 'dynamic-sub'
                ? a[aOffset] - b[bOffset]
                : a[aOffset] + b[bOffset];
            }
          }
          for (const operation of commandBuffer.operations) {
            if (operation.type !== 'dispatch' ||
                operation.pipeline.code !== 'conv2d-regular-out16') continue;
            const entries = new Map(operation.bindGroup.entries.map((entry) => [
              entry.binding,
              entry.resource.buffer,
            ]));
            const input = new Float32Array(entries.get(0).bytes.buffer);
            const weight = new Float32Array(entries.get(1).bytes.buffer);
            const bias = new Float32Array(entries.get(2).bytes.buffer);
            const output = new Float32Array(entries.get(3).bytes.buffer);
            const params = new Uint32Array(entries.get(4).bytes.buffer);
            const [batch, inH, inW, inC, outC, outH, outW, kh, kw, sy, sx,
              padTop, padLeft, , relu, dilationY, dilationX] = params;
            for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
              for (let outputY = 0; outputY < outH; outputY++) {
                for (let outputX = 0; outputX < outW; outputX++) {
                  for (let outputChannel = 0; outputChannel < outC; outputChannel++) {
                    let sum = bias[outputChannel];
                    for (let inputChannel = 0; inputChannel < inC; inputChannel++) {
                      for (let kernelY = 0; kernelY < kh; kernelY++) {
                        const inputY = outputY * sy + kernelY * dilationY - padTop;
                        if (inputY < 0 || inputY >= inH) continue;
                        for (let kernelX = 0; kernelX < kw; kernelX++) {
                          const inputX = outputX * sx + kernelX * dilationX - padLeft;
                          if (inputX < 0 || inputX >= inW) continue;
                          const inputIndex = ((batchIndex * inH + inputY) * inW + inputX) *
                            inC + inputChannel;
                          const weightIndex = (((kernelY * kw + kernelX) * inC +
                            inputChannel) * outC) + outputChannel;
                          sum += input[inputIndex] * weight[weightIndex];
                        }
                      }
                    }
                    if (relu === 1) sum = Math.max(sum, 0);
                    else if (relu >= 2) sum = Math.min(Math.max(sum, 0), 6);
                    output[((batchIndex * outH + outputY) * outW + outputX) * outC +
                      outputChannel] = sum;
                  }
                }
              }
            }
          }
        }
      },
      onSubmittedWorkDone() {
        if (state.autoCompleteWork) return Promise.resolve();
        return new Promise((resolve) => state.workResolvers.push(resolve));
      },
    },
  };
  state.finishSubmittedWork = async () => {
    const resolvers = state.workResolvers.splice(0);
    for (const resolve of resolvers) resolve();
    await Promise.resolve();
  };
  state.loseDevice = async (message = 'injected device loss') => {
    state.autoCompleteWork = true;
    loseDevice({ reason: 'unknown', message });
    await state.finishSubmittedWork();
    await Promise.resolve();
  };
  return device;
}

async function fixture({
  snapshot = dynamicReluSnapshot(),
  device = mockDevice(),
  contextOptions = {},
  shaderLibrary = ShaderLibrary,
} = {}) {
  const sourceEngine = new WebGPUEngine(device, { shaderLibrary });
  const contextEngines = [];
  const originalFork = sourceEngine.fork.bind(sourceEngine);
  sourceEngine.fork = (invariantWeightBorrow) => {
    const engine = originalFork(invariantWeightBorrow);
    contextEngines.push(engine);
    return engine;
  };
  const provider = new WebGPUBackendProvider(sourceEngine);
  const runtime = new Runtime();
  runtime._addProvider('webgpu', provider);
  const compiled = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext(contextOptions);
  return { device, engine: contextEngines[0], runtime, compiled, context };
}

function shaped(length, offset = 0) {
  return {
    x: {
      data: Float32Array.from({ length }, (_, index) => index % 2 ? index + offset : -index - 1),
      shape: [1, length],
    },
  };
}

function requiredCompileFailure(pattern) {
  return (error) => {
    assert.equal(error?.code, 'BACKEND_REQUIRED');
    const message = error?.report?.candidates?.at(-1)?.message || '';
    assert.match(message, pattern);
    return true;
  };
}

test('WebGPU dynamic provider reuses pipelines, grows capacity, and snapshots exact outputs', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const first = await context.execute(shaped(63));
  assert.deepEqual(
    [...await first.output('y').read()],
    [...shaped(63).x.data].map((value) => Math.max(0, value)),
  );
  assert.equal(first.report.backendReport.specializationCacheHit, false);
  assert.equal(first.report.backendReport.specializationRebindCount, 1);
  assert.equal(device.state.pipelineCreates, 1);
  assert.deepEqual(device.state.dispatches.at(-1), {
    ...device.state.dispatches.at(-1), x: 1, y: 1, z: 1,
  });

  const repeated = await context.execute(shaped(63, 10));
  assert.equal(repeated.report.backendReport.specializationCacheHit, true);
  assert.equal(repeated.report.backendReport.specializationRebindCount, 1);
  assert.equal(device.state.pipelineCreates, 1);

  const oldTensorBuffers = device.state.buffers.filter((buffer) =>
    buffer.descriptor.label === 'Tensor_x' || buffer.descriptor.label === 'Tensor_y');
  const large = await context.execute(shaped(129));
  assert.equal(large.report.backendReport.specializationRebindCount, 2);
  assert.equal(large.report.backendReport.activationGrowCount, 1);
  assert.equal(device.state.pipelineCreates, 1,
    'uniform-only shape changes retain the device-wide compute pipeline');
  assert.equal(device.state.dispatches.at(-1).x, 3);
  assert.equal(large.report.backendReport.specializationBufferCreateCount, 1,
    'the shape uniform keeps one context-owned buffer identity');
  assert.equal(large.report.backendReport.bindGroupCreateCount, 2,
    'tensor growth replaces the one bind group that references grown buffers');
  assert.ok(large.report.backendReport.pendingRetiredBufferCount >= 2,
    'replaced tensor buffers remain context-owned behind a queue fence');
  assert.ok(oldTensorBuffers.every((buffer) => !buffer.destroyed),
    'replaced submitted buffers remain alive until queue completion');
  await device.state.finishSubmittedWork();
  assert.ok(oldTensorBuffers.every((buffer) => buffer.destroyed));

  const capacityAfterGrow = large.report.backendReport.activationCapacityBytes;
  const smallAgain = await context.execute(shaped(32));
  assert.equal(smallAgain.report.backendReport.activationCapacityBytes, capacityAfterGrow);
  assert.ok(smallAgain.report.backendReport.logicalActivationBytes < capacityAfterGrow);
  assert.equal(device.state.pipelineCreates, 1);
  assert.equal(smallAgain.report.backendReport.specializationBufferCreateCount, 1);
  assert.equal(smallAgain.report.backendReport.liveSpecializationBufferCount, 1);
  assert.equal(smallAgain.report.backendReport.bindGroupCreateCount, 2,
    'a uniform-only rebinding does not create another bind group');
  assert.equal(smallAgain.report.backendReport.liveBindGroupCount, 1);
  assert.equal(smallAgain.report.backendReport.pendingRetiredBufferCount, 0);
  assert.ok(smallAgain.report.backendReport.bindGroupReuseCount >= 1);
  assert.deepEqual([...await first.output('y').read()],
    [...shaped(63).x.data].map((value) => Math.max(0, value)),
    'the first exact result remains stable after context buffer reuse');

  await device.state.finishSubmittedWork();
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
  assert.deepEqual([...await large.output('y').read()],
    [...shaped(129).x.data].map((value) => Math.max(0, value)),
    'result-owned GPU storage remains readable after every parent closes');
  await Promise.all([first.close(), repeated.close(), large.close(), smallAgain.close()]);
});

test('WebGPU hands a live dynamic TensorResult to another context on the same device without readback', async () => {
  const device = mockDevice();
  device.state.autoCompleteWork = true;
  const producer = await fixture({ device });
  const consumer = await fixture({ device, snapshot: dynamicAdd2DSnapshot() });
  const sourceValues = shaped(33).x.data;
  const sourceResult = await producer.context.execute({
    x: { data: sourceValues, shape: [1, 33] },
  });
  const source = sourceResult.output('y');
  let readCalls = 0;
  const originalRead = source.read.bind(source);
  source.read = async () => {
    readCalls++;
    return originalRead();
  };

  const submissionsBefore = device.state.submissions.length;
  const consumed = await consumer.context.execute({
    a: { data: source, shape: [1, 33] },
    b: { data: source, shape: [1, 33] },
  });
  assert.equal(readCalls, 0, 'device handoff never calls TensorResult.read()');
  const requestSubmissions = device.state.submissions.slice(submissionsBefore);
  const deviceInputSubmissions = requestSubmissions.filter((operations) =>
    operations.some((operation) => operation.type === 'copy' &&
      operation.source === source.deviceBuffer));
  assert.equal(deviceInputSubmissions.length, 1,
    'all device input copies use one queue submission');
  assert.equal(deviceInputSubmissions[0].filter((operation) =>
    operation.type === 'copy' && operation.source === source.deviceBuffer).length, 2,
  'both device inputs are encoded into the one D2D submission');
  const deviceCopySubmission = requestSubmissions.findIndex((operations) =>
    operations.some((operation) => operation.type === 'copy' &&
      operation.source === source.deviceBuffer));
  const computeSubmission = requestSubmissions.findIndex((operations) =>
    operations.some((operation) => operation.type === 'dispatch'));
  assert.ok(deviceCopySubmission >= 0 && computeSubmission > deviceCopySubmission,
    'D2D input copies are ordered before consumer compute on the same queue');
  assert.deepEqual([...await consumed.output('y').read()],
    [...sourceValues].map((value) => Math.max(0, value) * 2));
  for (const label of ['Tensor_a', 'Tensor_b']) {
    const destination = device.state.buffers.filter((buffer) =>
      buffer.descriptor.label === label).at(-1);
    assert.deepEqual([...new Float32Array(destination.bytes.buffer).slice(0, 33)],
      [...sourceValues].map((value) => Math.max(0, value)),
      `the exact logical source bytes reached ${label} by D2D copy`);
  }

  await Promise.all([sourceResult.close(), consumed.close()]);
  await Promise.all([producer.context.close(), consumer.context.close()]);
  await Promise.all([producer.compiled.close(), consumer.compiled.close()]);
  await Promise.all([producer.runtime.close(), consumer.runtime.close()]);
});

test('WebGPU result close retires snapshot buffers behind submitted copy completion', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const result = await context.execute(shaped(16));
  const snapshot = result.output('y').deviceBuffer;

  await result.close();
  assert.equal(snapshot.destroyed, false,
    'result close cannot destroy a buffer referenced by its submitted snapshot copy');
  await device.state.finishSubmittedWork();
  assert.equal(snapshot.destroyed, true,
    'the requested release destroys the snapshot after queue completion');

  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('WebGPU result close drains a delayed read before retiring its snapshot', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const result = await context.execute(shaped(16));
  const snapshot = result.output('y').deviceBuffer;
  const originalCreateBuffer = device.createBuffer;
  let releaseMap;
  device.createBuffer = function createBufferWithDelayedMap(descriptor) {
    const buffer = originalCreateBuffer.call(this, descriptor);
    if (descriptor.label === undefined) {
      buffer.mapAsync = () => new Promise((resolve) => { releaseMap = resolve; });
    }
    return buffer;
  };

  const reading = result.output('y').read();
  for (let turn = 0; turn < 4 && typeof releaseMap !== 'function'; turn++) {
    await Promise.resolve();
  }
  assert.equal(typeof releaseMap, 'function', 'the result read reached its map fence');
  const closing = result.close();
  let closeSettled = false;
  void closing.then(() => { closeSettled = true; });
  await Promise.resolve();
  assert.equal(closeSettled, false, 'close waits for the accepted result read');
  assert.equal(snapshot.destroyed, false,
    'the snapshot remains live while the delayed read references it');

  releaseMap();
  assert.deepEqual([...await reading],
    [...shaped(16).x.data].map((value) => Math.max(0, value)));
  await Promise.resolve();
  assert.equal(closeSettled, false,
    'the snapshot copy retirement fence still owns storage after read completion');
  await device.state.finishSubmittedWork();
  await closing;
  assert.equal(snapshot.destroyed, true);

  device.createBuffer = originalCreateBuffer;
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('WebGPU result construction failure retires submitted snapshots behind a fence', async () => {
  const { device, engine, runtime, compiled, context } = await fixture();
  const originalSnapshotOutputs = engine.snapshotOutputs.bind(engine);
  engine.snapshotOutputs = async () => {
    const snapshots = await originalSnapshotOutputs();
    snapshots.get('y').deviceBuffer.size += 4;
    return snapshots;
  };

  await assert.rejects(context.execute(shaped(16)),
    (error) => error.code === 'EXECUTION_FAILED' && /required alignment padding/.test(error.message));
  const rejectedSnapshot = device.state.buffers.filter((buffer) =>
    buffer.descriptor.label === 'Result_y').at(-1);
  assert.equal(rejectedSnapshot?.destroyed, false,
    'core result validation cannot immediately destroy a submitted snapshot');
  await device.state.finishSubmittedWork();
  assert.equal(rejectedSnapshot?.destroyed, true,
    'construction-failure cleanup retires the snapshot after queue completion');

  engine.snapshotOutputs = originalSnapshotOutputs;
  device.state.autoCompleteWork = true;
  const recovered = await context.execute(shaped(16, 3));
  assert.deepEqual([...await recovered.output('y').read()],
    [...shaped(16, 3).x.data].map((value) => Math.max(0, value)));
  await recovered.close();
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('WebGPU device input lease keeps the source alive through an accepted queued execute', async () => {
  const device = mockDevice();
  const producer = await fixture({ device });
  const consumer = await fixture({ device });
  const sourceResult = await producer.context.execute(shaped(16));
  const source = sourceResult.output('y');
  let consumerEntered = false;
  let blockerResult = null;
  const blocker = consumer.context.execute({
    x: { data: Float32Array.from({ length: 16 }, () => 1), shape: [1, 16] },
  }).then((result) => {
    consumerEntered = true;
    blockerResult = result;
  });
  const handoff = consumer.context.execute({
    x: { data: source, shape: [1, 16] },
  });
  const closing = sourceResult.close();
  let closeSettled = false;
  void closing.then(() => { closeSettled = true; });
  assert.equal(sourceResult.closed, false);
  assert.equal(source.deviceBuffer.destroyed, false,
    'close cannot destroy a source leased by an accepted execute');
  await blocker;
  assert.equal(consumerEntered, true);
  const consumed = await handoff;
  await Promise.resolve();
  assert.equal(closeSettled, false,
    'consumer submission completion does not retire its source lease');
  assert.equal(source.deviceBuffer.destroyed, false,
    'submitted D2D and compute work retain the source until queue completion');
  await device.state.finishSubmittedWork();
  await closing;
  assert.equal(sourceResult.closed, true);
  assert.equal(source.deviceBuffer.destroyed, true,
    'source releases only after the consumer queue fence completes');
  assert.deepEqual([...await consumed.output('y').read()],
    [...shaped(16).x.data].map((value) => Math.max(0, value)));

  device.state.autoCompleteWork = true;
  await Promise.all([blockerResult.close(), consumed.close()]);
  await Promise.all([producer.context.close(), consumer.context.close()]);
  await Promise.all([producer.compiled.close(), consumer.compiled.close()]);
  await Promise.all([producer.runtime.close(), consumer.runtime.close()]);
});

test('WebGPU device input retirement survives a later consumer submit failure', async () => {
  const device = mockDevice();
  const producer = await fixture({ device });
  const consumer = await fixture({ device });
  const sourceResult = await producer.context.execute(shaped(8));
  const source = sourceResult.output('y');
  const originalSubmit = device.queue.submit;
  let consumerSubmits = 0;
  device.queue.submit = function submitWithComputeFailure(commandBuffers) {
    consumerSubmits++;
    if (consumerSubmits === 2) throw new Error('injected consumer compute submit failure');
    return originalSubmit.call(this, commandBuffers);
  };

  const execution = consumer.context.execute({
    x: { data: source, shape: [1, 8] },
  });
  const closing = sourceResult.close();
  let closeSettled = false;
  void closing.then(() => { closeSettled = true; });
  await assert.rejects(execution, /injected consumer compute submit failure/);
  device.queue.submit = originalSubmit;
  await Promise.resolve();
  assert.equal(closeSettled, false);
  assert.equal(source.deviceBuffer.destroyed, false,
    'an accepted D2D copy keeps its source alive after a later submit fails');

  await device.state.finishSubmittedWork();
  await closing;
  assert.equal(source.deviceBuffer.destroyed, true);
  device.state.autoCompleteWork = true;
  await Promise.all([producer.context.close(), consumer.context.close()]);
  await Promise.all([producer.compiled.close(), consumer.compiled.close()]);
  await Promise.all([producer.runtime.close(), consumer.runtime.close()]);
});

test('WebGPU device handoff rejects a different physical device and incompatible dtype or shape', async () => {
  const first = await fixture({ device: mockDevice() });
  const second = await fixture({ device: mockDevice() });
  first.device.state.autoCompleteWork = true;
  second.device.state.autoCompleteWork = true;
  const result = await first.context.execute(shaped(4));
  const source = result.output('y');

  await assert.rejects(second.context.execute({
    x: { data: source, shape: [1, 4] },
  }), (error) => error.code === 'INVALID_ARGUMENT' && /same physical GPUDevice/.test(error.message));
  await assert.rejects(first.context.execute({
    x: { data: source, shape: [2, 2] },
  }), /device result shape/);
  const int8 = await fixture({ device: first.device, snapshot: dynamicQLinearSnapshot() });
  await assert.rejects(int8.context.execute({
    x: { data: source, shape: [1, 4] },
  }), /dtype 'float32'.*requires 'int8'|requires 'int8'.*float32/);

  await result.close();
  await Promise.all([first.context.close(), second.context.close(), int8.context.close()]);
  await Promise.all([first.compiled.close(), second.compiled.close(), int8.compiled.close()]);
  await Promise.all([first.runtime.close(), second.runtime.close(), int8.runtime.close()]);
});

test('WebGPU elides only byte-identical specialization writes across distinct signatures', async () => {
  const snapshot = dynamicQLinearSnapshot({ unrelatedSequenceMax: 4 });
  const { device, runtime, compiled, context } = await fixture({ snapshot });
  device.state.autoCompleteWork = true;
  const x = Int8Array.from({ length: 16 }, (_, index) => (index % 5) - 2);
  const execute = (past) => context.execute({
    x: { data: x, shape: [1, 16] },
    past: { data: Float32Array.from(past), shape: [1, past.length] },
  });
  const results = [];
  try {
    assert.equal(compiled.report.selectedBackend, 'webgpu');
    const first = await execute([-1]);
    results.push(first);
    assert.deepEqual([...await first.output('present').read()], [0]);
    assert.equal(first.report.backendReport.specializationWriteCount, 3);
    assert.equal(first.report.backendReport.specializationWriteSkipCount, 1,
      'a newly allocated zero-initialized buffer needs no all-zero upload');

    const second = await execute([-2, 3]);
    results.push(second);
    assert.deepEqual([...await second.output('present').read()], [0, 3]);
    assert.deepEqual([...await second.output('y').read()], [...await first.output('y').read()],
      'the fixed token branch remains exact while an independent cache extent changes');
    assert.equal(second.report.backendReport.specializationRebindCount, 2);
    assert.equal(second.report.backendReport.specializationWriteCount, 4,
      'only the changed cache-shape uniform reaches the queue');
    assert.equal(second.report.backendReport.specializationWriteSkipCount, 4,
      'QLinear multipliers, zero points, and fixed-row parameters are byte-identical');
    assert.equal(second.report.backendReport.specializationWriteSkipBytes, 448);
    assert.equal(second.report.backendReport.specializationContentBytes, 336);

    const third = await execute([4, -5, 6]);
    results.push(third);
    assert.deepEqual([...await third.output('present').read()], [4, 0, 6]);
    assert.equal(third.report.backendReport.specializationWriteCount, 5);
    assert.equal(third.report.backendReport.specializationWriteSkipCount, 7);
    assert.equal(third.report.backendReport.bindGroupReuseCount >= 2, true);
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('failed WebGPU capacity staging leaves the previous shape generation executable', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const first = await context.execute(shaped(32));
  device.state.failLabel = 'Tensor_x';
  await assert.rejects(context.execute(shaped(129)), (error) => {
    assert.equal(error?.code, 'OUT_OF_MEMORY');
    assert.match(error.message, /injected allocation failure/);
    return true;
  });

  const recovered = await context.execute(shaped(32, 20));
  assert.deepEqual([...await recovered.output('y').read()],
    [...shaped(32, 20).x.data].map((value) => Math.max(0, value)));
  assert.equal(recovered.report.backendReport.specializationRebindCount, 1,
    'failed candidates do not publish a new specialization generation');
  assert.equal(recovered.report.backendReport.specializationWriteCount,
    first.report.backendReport.specializationWriteCount,
    'a pre-commit allocation failure leaves the old specialization executable');
  assert.equal(recovered.report.backendReport.specializationContentBytes,
    first.report.backendReport.specializationContentBytes,
    'a pre-commit failure preserves the exact committed content mirror');

  await device.state.finishSubmittedWork();
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
  await Promise.all([first.close(), recovered.close()]);
});

test('WebGPU provider rejects an asynchronously scoped tensor OOM before result publication', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const scopes = [];
  device.pushErrorScope = (filter) => { scopes.push({ filter, error: null }); };
  device.popErrorScope = () => Promise.resolve(scopes.pop()?.error ?? null);
  const originalCreateBuffer = device.createBuffer;
  let failTensor = true;
  device.createBuffer = function createBufferWithScopedOOM(descriptor) {
    const buffer = originalCreateBuffer.call(this, descriptor);
    if (failTensor && descriptor.label === 'Tensor_x') {
      failTensor = false;
      const scope = [...scopes].reverse().find(({ filter }) => filter === 'out-of-memory');
      scope.error = { name: 'GPUOutOfMemoryError', message: 'not enough memory left' };
    }
    return buffer;
  };

  await assert.rejects(context.execute(shaped(32)),
    (error) => error?.code === 'OUT_OF_MEMORY' && /not enough memory left/.test(error.message));
  const rejected = device.state.buffers.filter((buffer) =>
    String(buffer.descriptor.label || '').startsWith('Tensor_'));
  assert.ok(rejected.length >= 2 && rejected.every((buffer) => buffer.destroyed),
    'the provider does not retain any invalid candidate tensor buffer');

  const recovered = await context.execute(shaped(32, 5));
  assert.deepEqual([...await recovered.output('y').read()],
    [...shaped(32, 5).x.data].map((value) => Math.max(0, value)));
  assert.equal(recovered.report.backendReport.specializationRebindCount, 1);
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
  await recovered.close();
});

test('steady WebGPU execution scopes only its mandatory result allocation', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const scopes = [];
  const pushes = [];
  device.pushErrorScope = (filter) => {
    pushes.push(filter);
    scopes.push({ filter, error: null });
  };
  device.popErrorScope = () => Promise.resolve(scopes.pop()?.error ?? null);
  device.state.autoCompleteWork = true;

  const first = await context.execute(shaped(16));
  pushes.length = 0;
  const second = await context.execute(shaped(16, 10));
  assert.deepEqual(pushes, ['validation', 'out-of-memory'],
    'the hot path adds one nested scope pair for result allocation, not upload or compute submit');
  assert.deepEqual([...await second.output('y').read()],
    [...shaped(16, 10).x.data].map((value) => Math.max(0, value)));

  await Promise.all([first.close(), second.close()]);
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('failed public value preflight publishes no shape generation or cache metadata', async () => {
  const { device, runtime, compiled, context } = await fixture({
    snapshot: dynamicEmbeddingSnapshot(),
  });
  device.state.autoCompleteWork = true;
  const execute = (values) => context.execute({
    ids: { data: Int32Array.from(values), shape: [values.length] },
  });
  const first = await execute([0]);
  const before = first.report.backendReport;
  const bufferCount = device.state.buffers.length;
  const pipelineCreates = device.state.pipelineCreates;
  const dispatchCount = device.state.dispatches.length;

  await assert.rejects(execute([0, 1, 3]), /token id 3 is outside vocabulary size 3/);
  assert.equal(device.state.buffers.length, bufferCount,
    'invalid values allocate no candidate GPU generation');
  assert.equal(device.state.pipelineCreates, pipelineCreates);
  assert.equal(device.state.dispatches.length, dispatchCount);

  const recovered = await execute([2]);
  const after = recovered.report.backendReport;
  assert.equal(after.shapeSignature, before.shapeSignature,
    'the prior concrete signature remains current');
  assert.equal(after.specializationRebindCount, before.specializationRebindCount);
  assert.equal(after.specializationCacheEntries, before.specializationCacheEntries);
  assert.equal(after.specializationCacheMisses, before.specializationCacheMisses);
  assert.equal(after.specializationCacheEvictions, before.specializationCacheEvictions);
  assert.equal(after.specializationCacheHits, before.specializationCacheHits + 1,
    'only the recovered valid request updates cache telemetry');
  assert.deepEqual(recovered.output('y').shape, [1, 1]);

  await Promise.all([first.close(), recovered.close()]);
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('canonical quantized predicates specialize WebGPU at dynamic min/max shapes', async () => {
  const { device, runtime, compiled, context } = await fixture({
    snapshot: dynamicQuantizedPredicateSnapshot(),
  });
  device.state.autoCompleteWork = true;
  const inputs = (batch, sequence, queries, keys) => ({
    batch_a: {
      data: Int8Array.from({ length: batch * queries * 4 }, (_, index) => index % 5 - 2),
      shape: [batch, queries, 4],
    },
    batch_b: {
      data: Int8Array.from({ length: 12 }, (_, index) => index % 3 - 1),
      shape: [1, 4, 3],
    },
    tokens: {
      data: Int8Array.from({ length: batch * sequence * 4 }, (_, index) => index % 7 - 3),
      shape: [batch, sequence, 4],
    },
    keep: { data: new Int32Array(batch * sequence).fill(1), shape: [batch, sequence] },
    q: {
      data: Int8Array.from({ length: batch * queries * 8 }, (_, index) => index % 5 - 2),
      shape: [batch, queries, 8],
    },
    k: {
      data: Int8Array.from({ length: batch * keys * 8 }, (_, index) => index % 5 - 2),
      shape: [batch, keys, 8],
    },
    v: {
      data: Int8Array.from({ length: batch * keys * 8 }, (_, index) => index % 7 - 3),
      shape: [batch, keys, 8],
    },
    attention_mask: {
      data: new Int32Array(batch * queries * keys).fill(1),
      shape: [batch, queries, keys],
    },
    requant_input: {
      data: Int8Array.from({ length: batch * sequence }, (_, index) => index % 5 - 2),
      shape: [batch, sequence],
    },
  });
  const results = [];
  try {
    assert.equal(device.state.pipelineCreates, 4,
      'the complete proved operator domain is precompiled before execution');
    const minimum = await context.execute(inputs(1, 1, 1, 1));
    results.push(minimum);
    assert.deepEqual(minimum.output('batch_out').shape, [1, 1, 3]);
    assert.deepEqual(minimum.output('mean_out').shape, [1, 4]);
    assert.deepEqual(minimum.output('attention_out').shape, [1, 1, 8]);
    assert.deepEqual(minimum.output('requant_out').shape, [1, 1]);
    assert.equal(minimum.report.backendReport.specializationRebindCount, 1);

    const maximum = await context.execute(inputs(2, 3, 2, 3));
    results.push(maximum);
    assert.deepEqual(maximum.output('batch_out').shape, [2, 2, 3]);
    assert.deepEqual(maximum.output('mean_out').shape, [2, 4]);
    assert.deepEqual(maximum.output('attention_out').shape, [2, 2, 8]);
    assert.deepEqual(maximum.output('requant_out').shape, [2, 3]);
    assert.equal(maximum.report.backendReport.specializationRebindCount, 2);
    assert.equal(device.state.pipelineCreates, 4,
      'min/max shape changes rewrite bindings and uniforms without compiling new pipelines');
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('failed ordinary WebGPU preflight preserves a decode seed until a dispatch succeeds', async () => {
  const { device, runtime, compiled, context } = await fixture({
    snapshot: dynamicEmbeddingSnapshot(),
    contextOptions: {
      decode: {
        changedInputs: ['ids'],
        rowMode: 'disabled',
        requireIncremental: true,
      },
    },
  });
  device.state.autoCompleteWork = true;
  const results = [];
  const input = (...ids) => ({ ids: { data: Int32Array.from(ids), shape: [ids.length] } });
  try {
    const seed = await context.decode.seed(input(0));
    results.push(seed);
    const semanticSeedSignature = seed.report.decodeState.semanticSeedSignature;
    assert.equal(seed.report.decodeState.cacheGeneration, 1);

    const dispatchCount = device.state.dispatches.length;
    await assert.rejects(context.execute(input(3)),
      /token id 3 is outside vocabulary size 3/);
    assert.equal(device.state.dispatches.length, dispatchCount,
      'the rejected ordinary request never crosses the dispatch boundary');

    device.state.failLabel = 'Tensor_ids';
    await assert.rejects(context.execute(input(1, 2)), /Tensor_ids/);
    assert.equal(device.state.dispatches.length, dispatchCount,
      'a rejected specialization candidate remains before the commit boundary');

    const step = await context.decode.step(input(1));
    results.push(step);
    assert.equal(step.report.decodeState.cacheGeneration, 1);
    assert.equal(step.report.decodeState.mode, 'incremental-dependency');
    assert.equal(step.report.decodeState.semanticSeedSignature, semanticSeedSignature,
      'provider and core decode state both retain the accepted seed');

    const ordinary = await context.execute(input(2));
    results.push(ordinary);
    await assert.rejects(context.decode.step(input(1)),
      (error) => error?.code === 'INVALID_ARGUMENT' && /successful seed/.test(error.message));
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('WebGPU commit failures clear core and provider decode state for execute, seed, and step', async () => {
  const { device, runtime, compiled, context } = await fixture({
    snapshot: dynamicEmbeddingSnapshot(),
    contextOptions: {
      decode: {
        changedInputs: ['ids'],
        rowMode: 'disabled',
        requireIncremental: true,
      },
    },
  });
  device.state.autoCompleteWork = true;
  const results = [];
  const input = (id) => ({ ids: { data: Int32Array.of(id), shape: [1] } });
  const assertUnseeded = () => assert.rejects(context.decode.step(input(1)),
    (error) => error?.code === 'INVALID_ARGUMENT' && /successful seed/.test(error.message));
  try {
    const firstSeed = await context.decode.seed(input(0));
    results.push(firstSeed);
    assert.equal(firstSeed.report.backendReport.specializationRebindCount, 1);
    device.state.failWriteLabel = 'Tensor_ids';
    await assert.rejects(context.execute(input(1)), /injected write failure for Tensor_ids/);
    await assertUnseeded();

    const afterWriteFailure = await context.decode.seed(input(0));
    results.push(afterWriteFailure);
    assert.equal(afterWriteFailure.report.backendReport.specializationRebindCount, 2,
      'the provider mirrors the executor physical-signature invalidation');
    device.state.failNextSubmit = true;
    await assert.rejects(context.decode.seed(input(1)), /injected submit failure/);
    await assertUnseeded();

    const afterSubmitFailure = await context.decode.seed(input(0));
    results.push(afterSubmitFailure);
    assert.equal(afterSubmitFailure.report.backendReport.specializationRebindCount, 3,
      'a failed submit cannot leave the provider same-signature fast path eligible');
    device.state.failLabel = 'Result_y';
    await assert.rejects(context.decode.step(input(1)), /Result_y/);
    await assertUnseeded();
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('failed WebGPU specialization allocation rolls back its complete candidate', async () => {
  const { device, runtime, compiled, context } = await fixture();
  device.state.failNextUnlabeledBuffer = true;
  await assert.rejects(context.execute(shaped(32)), (error) => {
    assert.equal(error?.code, 'OUT_OF_MEMORY');
    assert.match(error.message, /injected unlabeled allocation failure/);
    return true;
  });
  const rejectedTensorBuffers = device.state.buffers.filter((buffer) =>
    String(buffer.descriptor.label || '').startsWith('Tensor_'));
  assert.ok(rejectedTensorBuffers.length === 2 &&
    rejectedTensorBuffers.every((buffer) => buffer.destroyed));

  const recovered = await context.execute(shaped(32, 5));
  assert.equal(recovered.report.backendReport.specializationRebindCount, 1);
  assert.equal(recovered.report.backendReport.specializationCacheHit, false);
  assert.deepEqual([...await recovered.output('y').read()],
    [...shaped(32, 5).x.data].map((value) => Math.max(0, value)));
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
  await recovered.close();
});

test('failed WebGPU result allocation destroys earlier snapshots and preserves the context', async () => {
  const { device, runtime, compiled, context } = await fixture({
    snapshot: dynamicReluSnapshot({ outputNames: ['x', 'y'] }),
  });
  device.state.failLabel = 'Result_y';
  await assert.rejects(context.execute(shaped(16)), (error) => {
    assert.equal(error?.code, 'OUT_OF_MEMORY');
    assert.match(error.message, /Result_y/);
    return true;
  });
  const rejectedFirstSnapshot = device.state.buffers.find((buffer) =>
    buffer.descriptor.label === 'Result_x');
  assert.equal(rejectedFirstSnapshot?.destroyed, true,
    'a later result allocation failure releases every earlier snapshot');

  const recovered = await context.execute(shaped(16, 9));
  assert.deepEqual([...await recovered.output('x').read()], [...shaped(16, 9).x.data]);
  assert.deepEqual([...await recovered.output('y').read()],
    [...shaped(16, 9).x.data].map((value) => Math.max(0, value)));
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
  await recovered.close();
});

test('WebGPU shape rebind applies retirement backpressure before a third generation', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const first = await context.execute(shaped(16));
  const second = await context.execute(shaped(33));
  const liveTensorBuffers = () => device.state.buffers.filter((buffer) =>
    String(buffer.descriptor.label || '').startsWith('Tensor_') && !buffer.destroyed);
  assert.equal(liveTensorBuffers().length, 4,
    'one submitted generation and one current generation coexist');

  let thirdSettled = false;
  const thirdPromise = context.execute(shaped(65)).then((result) => {
    thirdSettled = true;
    return result;
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(thirdSettled, false,
    'another shape waits for the prior retirement fence');
  assert.equal(liveTensorBuffers().length, 4,
    'backpressure prevents a third live tensor generation');

  await device.state.finishSubmittedWork();
  const third = await thirdPromise;
  assert.equal(liveTensorBuffers().length, 4);
  await device.state.finishSubmittedWork();
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
  await Promise.all([first.close(), second.close(), third.close()]);
});

test('WebGPU bank shape rebind remains bounded while an older generation retires', async () => {
  const { device, runtime, compiled, context } = await fixture({
    snapshot: dynamicResourceBankSnapshot(),
    contextOptions: { bankResidency: { experts: [0] } },
    shaderLibrary: { ...ShaderLibrary, getGatherInt32Shader: () => 'gather-i32' },
  });
  const slots = (length) => ({
    slot: { data: new Int32Array(length), shape: [length] },
  });
  const first = await context.execute(slots(1));
  const second = await context.execute(slots(2));

  let thirdSettled = false;
  const thirdPromise = context.execute(slots(3)).then((result) => {
    thirdSettled = true;
    return result;
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(thirdSettled, false,
    'host bank binding may stage a candidate, but device staging waits for retirement');

  await device.state.finishSubmittedWork();
  const third = await thirdPromise;
  await device.state.finishSubmittedWork();
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
  await Promise.all([first.close(), second.close(), third.close()]);
});

test('WebGPU retains submitted buffers and fails closed when no retirement fence exists', async () => {
  const device = mockDevice();
  device.queue.onSubmittedWorkDone = undefined;
  const { runtime, compiled, context } = await fixture({ device });
  const first = await context.execute(shaped(16));
  const firstGeneration = device.state.buffers.filter((buffer) =>
    buffer.descriptor.label === 'Tensor_x' || buffer.descriptor.label === 'Tensor_y');
  const second = await context.execute(shaped(33));
  assert.ok(firstGeneration.every((buffer) => !buffer.destroyed),
    'a missing queue fence must not destroy submitted tensor storage');
  await assert.rejects(
    context.execute(shaped(65)),
    /cannot safely specialize another shape without a queue retirement fence/,
  );
  assert.ok(firstGeneration.every((buffer) => !buffer.destroyed),
    'a rejected third generation keeps the submitted generation context-owned');
  await context.close();
  assert.ok(firstGeneration.every((buffer) => buffer.destroyed));
  await compiled.close();
  await runtime.close();
  await Promise.all([first.close(), second.close()]);
});

test('WebGPU resource attestation includes host weights, result snapshots, and the plan LRU', async () => {
  const snapshot = inputMajorMatMulSnapshot();
  const device = mockDevice();
  const provider = new WebGPUBackendProvider(
    new WebGPUEngine(device, { shaderLibrary: ShaderLibrary }),
  );
  const compiled = await provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  });
  try {
    /* Maximum-domain bytes for this graph:
     *   host model + context weight clone:        2 * 1,748
     *   GPU weight + 3 generations of x/y:       1,748 + 3 * (304 + 368)
     *   old + replacement host input snapshots:  2 * 304
     *   forward/row/candidate dense auxiliaries plus exact host mirror:
     *                                           4 * (92 + 16)
     *   conservative row scratch/copy uniforms:   304 + 368 + 2 * 16
     *   exact device result snapshot:             368
     *   context plan metadata LRU:                1 MiB */
    const resources = compiled.compilationEvidence.shapeDomain;
    assert.equal(resources.maximumInputBytes, 304);
    assert.equal(
      resources.maximumResidentBytes,
      2 * 1_748 + 1_748 + 3 * (304 + 368) + 4 * (92 + 16) +
        304 + 368 + 2 * 16 + 368 + 2 * 304 + 1024 * 1024,
    );
  } finally {
    await compiled.close();
    await provider.close();
  }
});

test('WebGPU resource attestation charges host and device bank rebind coexistence', async () => {
  const maximumResidentBytes = async (snapshot) => {
    const provider = new WebGPUBackendProvider(
      new WebGPUEngine(mockDevice(), {
        shaderLibrary: { ...ShaderLibrary, getGatherInt32Shader: () => 'gather-i32' },
      }),
    );
    const compiled = await provider.compile(createBackendCompileInput(snapshot), {
      operatorFallback: 'forbid',
    });
    try {
      return compiled.compilationEvidence.shapeDomain.maximumResidentBytes;
    } finally {
      await compiled.close();
      await provider.close();
    }
  };
  const ordinarySnapshot = resourceBankSnapshot(false);
  const bankedSnapshot = resourceBankSnapshot(true);
  const ordinaryBytes = await maximumResidentBytes(ordinarySnapshot);
  const bankedBytes = await maximumResidentBytes(bankedSnapshot);
  const ordinaryAuxiliary = checkedWebGPUPhysicalDomain(
    createBackendCompileInput(ordinarySnapshot), mockDevice(),
  ).maximumAuxiliaryBytes;
  const bankedAuxiliary = checkedWebGPUPhysicalDomain(
    createBackendCompileInput(bankedSnapshot), mockDevice(),
  ).maximumAuxiliaryBytes;
  const fullBankBytes = 4 * Float32Array.BYTES_PER_ELEMENT;
  const bankSlotTableGenerationsAndMirror = 4 * (bankedAuxiliary - ordinaryAuxiliary);
  assert.equal(bankedBytes - ordinaryBytes - bankSlotTableGenerationsAndMirror, 4 * fullBankBytes,
    'banking adds committed+candidate+staging host payloads and one candidate GPU buffer');
});

test('WebGPU resolves output-only fixed aliases for bounds and concrete specialization', async () => {
  const snapshot = outputAliasLinearSnapshot();
  assert.deepEqual(snapshot.shapeDomainProof.symbolRelations, { B: 'B', O: 3 });
  const device = mockDevice({
    maxBufferSize: 128,
    maxStorageBufferBindingSize: 128,
  });
  const { runtime, compiled, context } = await fixture({ snapshot, device });
  device.state.autoCompleteWork = true;
  const input = Float32Array.from([
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16,
  ]);
  const result = await context.execute({ x: { data: input, shape: [4, 4] } });
  assert.deepEqual(result.output('y').shape, [4, 3]);
  assert.deepEqual([...await result.output('y').read()], [
    2, 4, 6,
    6, 8, 10,
    10, 12, 14,
    14, 16, 18,
  ]);
  const outputBuffer = device.state.buffers.find((buffer) =>
    buffer.descriptor.label === 'Tensor_y' && !buffer.destroyed);
  assert.equal(outputBuffer?.size, 4 * 3 * Float32Array.BYTES_PER_ELEMENT,
    'the capacity proof uses O=3 instead of the wider authored O interval');
  await result.close();
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('WebGPU resolves affine output bounds and re-specializes Concat at min/max shapes', async () => {
  const snapshot = affineConcatSnapshot();
  assert.deepEqual(snapshot.shapeDomainProof.affineSymbolRelations, {
    M: { source: 'Q', offset: 2 },
  });
  const device = mockDevice({
    maxBufferSize: 64,
    maxStorageBufferBindingSize: 64,
  });
  const { runtime, compiled, context } = await fixture({ snapshot, device });
  device.state.autoCompleteWork = true;
  const execute = async (suffix) => context.execute({
    prefix: { data: Float32Array.from([10, 11]), shape: [1, 2, 1] },
    suffix: { data: Float32Array.from(suffix), shape: [1, suffix.length, 1] },
  });
  const minimum = await execute([20]);
  assert.deepEqual(minimum.output('joined').shape, [1, 3, 1]);
  assert.deepEqual([...await minimum.output('joined').read()], [10, 11, 20]);
  const maximum = await execute([20, 21, 22, 23]);
  assert.deepEqual(maximum.output('joined').shape, [1, 6, 1]);
  assert.deepEqual([...await maximum.output('joined').read()], [10, 11, 20, 21, 22, 23]);
  const outputBuffer = device.state.buffers.findLast((buffer) =>
    buffer.descriptor.label === 'Tensor_joined' && !buffer.destroyed);
  assert.equal(outputBuffer?.size, 6 * Float32Array.BYTES_PER_ELEMENT);
  const minimumAgain = await execute([30]);
  assert.deepEqual(minimumAgain.output('joined').shape, [1, 3, 1]);
  assert.deepEqual([...await minimumAgain.output('joined').read()], [10, 11, 30],
    'the reused Concat uniforms carry the latest affine extent and offsets');
  assert.equal(minimumAgain.report.backendReport.specializationRebindCount, 3);
  assert.ok(minimumAgain.report.backendReport.specializationBufferReuseCount >= 2);
  assert.ok(minimumAgain.report.backendReport.bindGroupReuseCount >= 2);
  assert.deepEqual(device.state.dispatches.slice(-2).map(({ x, y, z }) => [x, y, z]), [
    [1, 1, 1],
    [1, 1, 1],
  ]);
  await Promise.all([minimum.close(), maximum.close(), minimumAgain.close()]);
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('WebGPU rewrites broadcast strides when equal-byte shapes alternate', async () => {
  const device = mockDevice();
  const { runtime, compiled, context } = await fixture({
    snapshot: alternatingBroadcastSubSnapshot(),
    device,
  });
  device.state.autoCompleteWork = true;
  const execute = (aShape, a, b) => context.execute({
    a: { data: Float32Array.from(a), shape: aShape },
    b: { data: Float32Array.from(b), shape: [1, aShape[1]] },
  });

  const first = await execute([2, 3], [1, 2, 3, 4, 5, 6], [10, 20, 30]);
  assert.deepEqual([...await first.output('y').read()], [-9, -18, -27, -6, -15, -24]);
  const initialBuffers = new Map(['a', 'b', 'y'].map((name) => [name,
    device.state.buffers.findLast((buffer) =>
      buffer.descriptor.label === `Tensor_${name}` && !buffer.destroyed),
  ]));

  const transposedExtent = await execute([3, 2], [1, 2, 3, 4, 5, 6], [100, 200]);
  assert.deepEqual([...await transposedExtent.output('y').read()],
    [-99, -198, -97, -196, -95, -194]);
  for (const [name, buffer] of initialBuffers) {
    assert.equal(device.state.buffers.findLast((candidate) =>
      candidate.descriptor.label === `Tensor_${name}` && !candidate.destroyed), buffer,
    `${name} retains its buffer identity while only concrete strides change`);
  }
  assert.equal(transposedExtent.report.backendReport.specializationBufferCreateCount, 1);
  assert.equal(transposedExtent.report.backendReport.bindGroupCreateCount, 1);
  assert.ok(transposedExtent.report.backendReport.bindGroupReuseCount >= 1);

  const firstExtentAgain = await execute([2, 3], [7, 8, 9, 10, 11, 12], [1, 2, 3]);
  assert.deepEqual([...await firstExtentAgain.output('y').read()],
    [6, 6, 6, 9, 9, 9],
    'returning to an earlier exact signature cannot retain the intervening stride metadata');
  assert.equal(firstExtentAgain.report.backendReport.specializationRebindCount, 3);
  assert.equal(firstExtentAgain.report.backendReport.specializationBufferCreateCount, 1);
  assert.equal(firstExtentAgain.report.backendReport.bindGroupCreateCount, 1);
  assert.ok(firstExtentAgain.report.backendReport.specializationBufferReuseCount >= 2);
  assert.ok(firstExtentAgain.report.backendReport.bindGroupReuseCount >= 2);

  await Promise.all([first.close(), transposedExtent.close(), firstExtentAgain.close()]);
  await context.close();
  await compiled.close();
  await runtime.close();
});

test('WebGPU provider decode reuses one specialization and rolls back a failed reseed', async () => {
  const { device, runtime, compiled, context } = await fixture({
    contextOptions: {
      decode: {
        changedInputs: ['x'],
        rowMode: 'disabled',
        requireIncremental: true,
      },
    },
  });
  const results = [];
  try {
    const seedInput = shaped(32);
    const seed = await context.decode.seed(seedInput);
    results.push(seed);
    assert.deepEqual([...await seed.output('y').read()],
      [...seedInput.x.data].map((value) => Math.max(0, value)));
    assert.equal(seed.report.decodeState.mode, 'incremental-seed');
    assert.equal(seed.report.decodeState.activeSequenceLength, 1);
    assert.equal(seed.report.decodeState.kvCapacity, 32);
    assert.equal(seed.report.decodeState.kvCapacityClass, 32);
    assert.equal(seed.report.decodeState.cacheGeneration, 1);
    assert.equal(seed.report.decodeState.automaticReset, false);
    const semanticSignature = seed.report.decodeState.semanticSeedSignature;
    const specializationCount = seed.report.backendReport.specializationRebindCount;

    const stepInput = shaped(32, 17);
    const step = await context.decode.step(stepInput);
    results.push(step);
    assert.deepEqual([...await step.output('y').read()],
      [...stepInput.x.data].map((value) => Math.max(0, value)));
    assert.equal(step.report.decodeState.mode, 'incremental-dependency');
    assert.equal(step.report.decodeState.activeSequenceLength, 2);
    assert.equal(step.report.decodeState.cacheGeneration, 1);
    assert.equal(step.report.decodeState.semanticSeedSignature, semanticSignature);
    assert.equal(step.report.backendReport.specializationCacheHit, true);
    assert.equal(step.report.backendReport.specializationRebindCount, specializationCount,
      'a token step must reuse the exact-shape specialization');
    assert.equal(device.state.pipelineCreates, 1);

    device.state.failLabel = 'Tensor_x';
    await assert.rejects(context.decode.seed(shaped(129)), /injected allocation failure/);

    const recoveredInput = shaped(32, 41);
    const recovered = await context.decode.step(recoveredInput);
    results.push(recovered);
    assert.deepEqual([...await recovered.output('y').read()],
      [...recoveredInput.x.data].map((value) => Math.max(0, value)));
    assert.equal(recovered.report.decodeState.cacheGeneration, 1,
      'failed candidate publication must not invalidate the committed seed');
    assert.equal(recovered.report.decodeState.semanticSeedSignature, semanticSignature);
    assert.equal(recovered.report.backendReport.specializationRebindCount, specializationCount);

    await context.decode.reset();
    await assert.rejects(context.decode.step(recoveredInput),
      (error) => error?.code === 'INVALID_ARGUMENT' && /successful seed/.test(error.message));
  } finally {
    await device.state.finishSubmittedWork();
    device.state.autoCompleteWork = true;
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('WebGPU plan metadata LRU is bounded and does not retain one graph arena per shape', async () => {
  const { device, runtime, compiled, context } = await fixture();
  device.state.autoCompleteWork = true;
  let final;
  for (let length = 1; length <= 10; length++) {
    const result = await context.execute(shaped(length));
    await final?.close();
    final = result;
  }
  assert.equal(final.report.backendReport.specializationCacheEntries, 8);
  assert.equal(final.report.backendReport.specializationCacheEvictions, 2);
  assert.equal(final.report.backendReport.liveSpecializationBufferCount, 1);
  assert.equal(final.report.backendReport.liveBindGroupCount, 1);
  assert.ok(final.report.backendReport.specializationCacheMetadataBytes < 1024 * 1024);
  await final.close();
  await context.close();
  await compiled.close();
  await runtime.close();
});

function qlinearInputs(batch) {
  return {
    x: {
      data: Int8Array.from({ length: batch * 16 }, (_, index) => (index % 5) - 2),
      shape: [batch, 16],
    },
  };
}

function qlinearRowInputs(offset = 0) {
  return {
    x: {
      data: Int8Array.from({ length: 3 * 16 }, (_, index) => ((index + offset) % 5) - 2),
      shape: [1, 3, 16],
    },
  };
}

test('failed WebGPU row-plan allocation is normalized and rolls back partial resources', async () => {
  const snapshot = dynamicQLinearSnapshot({ sequenceMax: 3 });
  const { device, runtime, compiled, context } = await fixture({
    snapshot,
    contextOptions: {
      decode: {
        changedInputs: ['x'],
        rowMode: 'required',
        requireIncremental: true,
      },
    },
  });
  device.state.autoCompleteWork = true;
  const results = [];
  try {
    results.push(await context.decode.seed(qlinearRowInputs()));
    const buffersBeforeRejectedPlan = device.state.buffers.length;
    device.state.failLabel = 'QLinear_zero_points_qlinear';
    await assert.rejects(
      context.decode.step(qlinearRowInputs(1), { position: 1 }),
      (error) => error?.code === 'OUT_OF_MEMORY' &&
        /QLinear_zero_points_qlinear/.test(error.message),
    );
    const rejected = device.state.buffers.slice(buffersBeforeRejectedPlan);
    assert.ok(rejected.length > 0 && rejected.every((buffer) => buffer.destroyed),
      'row scratch and params created before the rejected candidate are destroyed');

    results.push(await context.decode.seed(qlinearRowInputs(2)));
    const recovered = await context.decode.step(qlinearRowInputs(3), { position: 1 });
    results.push(recovered);
    assert.equal(recovered.report.decodeState.mode, 'incremental-row');
    assert.ok(device.state.buffers.some((buffer) =>
      String(buffer.descriptor.label || '').startsWith('IncrementalRow_0_') &&
      !buffer.destroyed));
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('WebGPU publishes a multi-node row specialization only after every node succeeds', async () => {
  const { device, runtime, compiled, context } = await fixture({
    snapshot: twoLayerDynamicQLinearSnapshot(),
    contextOptions: {
      decode: {
        changedInputs: ['x'],
        rowMode: 'required',
        requireIncremental: true,
      },
    },
  });
  device.state.autoCompleteWork = true;
  const results = [];
  try {
    results.push(await context.decode.seed(qlinearRowInputs()));
    const buffersBeforeRejectedGeneration = device.state.buffers.length;
    device.state.failLabel = 'QLinear_zero_points_qlinear2';
    await assert.rejects(
      context.decode.step(qlinearRowInputs(1), { position: 1 }),
      (error) => error?.code === 'OUT_OF_MEMORY' &&
        /QLinear_zero_points_qlinear2/.test(error.message),
    );
    const rejected = device.state.buffers.slice(buffersBeforeRejectedGeneration);
    assert.ok(rejected.some((buffer) =>
      String(buffer.descriptor.label || '').startsWith('IncrementalRow_0_')),
    'the injected failure happens after the first node was completely staged');
    assert.ok(rejected.length > 0 && rejected.every((buffer) => buffer.destroyed),
      'failure in a later selected node destroys the complete staged row generation');

    results.push(await context.decode.seed(qlinearRowInputs(2)));
    const recovered = await context.decode.step(qlinearRowInputs(3), { position: 1 });
    results.push(recovered);
    assert.equal(recovered.report.decodeState.mode, 'incremental-row');
    for (const nodeIndex of [0, 1]) {
      assert.ok(device.state.buffers.some((buffer) =>
        String(buffer.descriptor.label || '').startsWith(`IncrementalRow_${nodeIndex}_`) &&
        !buffer.destroyed),
      `recovery publishes row resources for node ${nodeIndex}`);
    }
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

async function cpuQLinear(snapshot, input) {
  const runtime = new Runtime();
  runtime._addProvider('cpu-js', new CPUBackendProvider(new CPUEngine()));
  const compiled = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const result = await context.execute(input);
  const values = [...await result.output('y').read()];
  await result.close();
  await context.close();
  await compiled.close();
  await runtime.close();
  return values;
}

function regularConvInputs(batch, offset = 0) {
  return {
    x: {
      data: Float32Array.from({ length: batch * 4 * 4 * 5 }, (_, index) =>
        ((index + offset) % 17 - 8) / 8),
      shape: [batch, 4, 4, 5],
    },
  };
}

test('WebGPU regular-out16 Conv2D reuses one pipeline across B=1->2->1', async () => {
  const snapshot = dynamicRegularConvSnapshot();
  const { device, runtime, compiled, context } = await fixture({ snapshot });
  device.state.autoCompleteWork = true;
  assert.equal(device.state.pipelineCreates, 7,
    'the scalar and six portable Conv routes compile once before publication');

  const results = [];
  try {
    for (const [batch, offset] of [[1, 0], [2, 3], [1, 7]]) {
      const result = await context.execute(regularConvInputs(batch, offset));
      results.push(result);
      assert.deepEqual(result.output('y').shape, [batch, 4, 4, 16]);
      assert.deepEqual(result.report.backendReport.selectedTactics,
        ['webgpu.conv2d.regular-out16']);
    }
    assert.equal(device.state.pipelineCreates, 7,
      'batch rebinds retain the precompiled regular-out16 pipeline');
    assert.deepEqual(results.map((result) =>
      result.report.backendReport.specializationRebindCount), [1, 2, 3]);
    const replay = await context.execute(regularConvInputs(1, 0));
    results.push(replay);
    assert.deepEqual([...await results[0].output('y').read()],
      [...await replay.output('y').read()],
      'returning to B=1 rewrites exact shape metadata without changing its output');
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('a mid-commit WebGPU write failure invalidates the old specialization before recovery', async () => {
  const snapshot = twoLayerDynamicQLinearSnapshot();
  const { device, runtime, compiled, context } = await fixture({ snapshot });
  device.state.autoCompleteWork = true;
  const input = (sequence, offset = 0) => ({
    x: {
      data: Int8Array.from({ length: sequence * 16 },
        (_, index) => ((index + offset) % 5) - 2),
      shape: [1, sequence, 16],
    },
  });
  const results = [];
  try {
    const first = await context.execute(input(1));
    results.push(first);
    const firstRebindCount = first.report.backendReport.specializationRebindCount;

    // Invariant multiplier and zero-point payloads are elided. Fail the second
    // changed params write, after node 1's reused uniform contains the rejected
    // S=2 geometry.
    device.state.failWriteAt = device.state.writeCount + 2;
    await assert.rejects(context.execute(input(2)), /injected write failure at/);

    const recoveryInput = input(1, 3);
    const recovered = await context.execute(recoveryInput);
    results.push(recovered);
    assert.equal(recovered.report.backendReport.specializationRebindCount,
      firstRebindCount + 1,
      'the prior signature is not executable until every reused uniform is rewritten');
    assert.equal(recovered.report.backendReport.specializationWriteCount,
      first.report.backendReport.specializationWriteCount + 6,
      'an uncertain partial commit clears the mirror and forces all six writes');
    assert.equal(recovered.report.backendReport.specializationWriteSkipCount,
      first.report.backendReport.specializationWriteSkipCount,
      'no write from the forced recovery may rely on pre-failure mirrors');
    assert.deepEqual([...await recovered.output('y').read()],
      await cpuQLinear(snapshot, recoveryInput));
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
    await Promise.all(results.map((result) => result.close()));
  }
});

test('WebGPU selects QLinear scalar/tiled tactics from resolved rows and matches CPU at maxima', async () => {
  const snapshot = dynamicQLinearSnapshot();
  const { device, runtime, compiled, context } = await fixture({ snapshot });
  device.state.autoCompleteWork = true;
  assert.equal(device.state.pipelineCreates, 2,
    'both proved portable tactics compile before the context is published');

  const scalarInput = qlinearInputs(1);
  const scalar = await context.execute(scalarInput);
  assert.deepEqual(scalar.report.backendReport.selectedTactics, ['webgpu.qlinear.scalar']);
  assert.deepEqual([...await scalar.output('y').read()], await cpuQLinear(snapshot, scalarInput));

  const tiledInput = qlinearInputs(2);
  const tiled = await context.execute(tiledInput);
  assert.deepEqual(tiled.report.backendReport.selectedTactics, ['webgpu.qlinear.tiled']);
  assert.deepEqual([...await tiled.output('y').read()], await cpuQLinear(snapshot, tiledInput));

  const maximumInput = qlinearInputs(9);
  const maximum = await context.execute(maximumInput);
  assert.deepEqual(maximum.report.backendReport.selectedTactics, ['webgpu.qlinear.tiled']);
  assert.deepEqual([...await maximum.output('y').read()], await cpuQLinear(snapshot, maximumInput));
  assert.equal(device.state.pipelineCreates, 2,
    'concrete scalar/tiled selection reuses the compile-time device pipelines');

  await context.close();
  await compiled.close();
  await runtime.close();
  await Promise.all([scalar.close(), tiled.close(), maximum.close()]);
});

test('WebGPU physical proof separates portable Conv specialists from packed-dot shaders', () => {
  const device = mockDevice();
  const conv = checkedWebGPUPhysicalDomain(
    createBackendCompileInput(dynamicRegularConvSnapshot()), device,
  );
  assert.ok(conv.requiredShaderMethods.includes('getConv2DShader'));
  assert.ok(conv.optionalShaderMethods.includes('getConv2DRegularOut16Shader'));
  assert.ok(conv.optionalShaderMethods.includes('getConv2DRegularC3Out16Shader'));
  assert.ok(!conv.packedDot4OptionalShaderMethods.includes('getConv2DRegularOut16Shader'));

  const quantized = checkedWebGPUPhysicalDomain(
    createBackendCompileInput(dynamicQuantizedPredicateSnapshot()), device,
  );
  assert.ok(quantized.requiredShaderMethods.includes('getQBatchMatMulShader'));
  assert.ok(quantized.packedDot4OptionalShaderMethods.includes('getQBatchMatMulDotShader'));
  assert.ok(!quantized.optionalShaderMethods.includes('getQBatchMatMulDotShader'));
});

test('WebGPU physical proof keeps QConv scalar fallback when its optional tile exceeds an axis', () => {
  const device = mockDevice();
  device.limits = {
    ...device.limits,
    maxComputeWorkgroupsPerDimension: 2,
  };
  const proof = checkedWebGPUPhysicalDomain(
    createBackendCompileInput(dynamicQConvScalarFallbackSnapshot()),
    device,
  );
  assert.ok(proof.requiredShaderMethods.includes('getQConv2DShader'));
  assert.ok(!proof.requiredShaderMethods.includes('getQConv2DTiledShader'));
  assert.ok(!proof.packedDot4OptionalShaderMethods.includes('getQConv2DDotTiledShader'));
});

test('WebGPU without packed dot precompiles feature-independent Conv specialists', async () => {
  const device = mockDevice();
  const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
  const provider = new WebGPUBackendProvider(engine);
  const runtime = new Runtime();
  runtime._addProvider('webgpu', provider);
  const snapshot = dynamicRegularConvSnapshot();
  const compiled = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
  });

  assert.equal(device.state.pipelineCreates, 7,
    'the scalar route and six feature-independent Conv specialists compile before publication');
  assert.deepEqual(
    device.state.buffers.map((buffer) => buffer.descriptor.label).sort(),
    snapshot.weightNames.map((name) => `Tensor_${name}`).sort(),
    'compile owns only fixed invariant weights; no context activation arena exists yet',
  );

  await compiled.close();
  await runtime.close();
});

test('WebGPU physical registry covers every canonical advertised route', () => {
  const advertised = runtimeOperatorsByBackend.webgpu.filter((operator) =>
    operatorShapeContract(operator)?.classification === 'canonical').sort();
  assert.deepEqual(WEBGPU_CANONICAL_PHYSICAL_OPERATORS, advertised);
  assert.equal(advertised.length, 86);
  assert.ok(!advertised.includes('NonMaxSuppression'));
});

test('WebGPU proves every canonical index/route value source or fails closed', () => {
  const device = mockDevice();
  for (const snapshot of [
    embeddingValueSnapshot('public'),
    embeddingValueSnapshot('weight'),
    embeddingValueSnapshot('clip'),
    embeddingValueSnapshot('argmax'),
    gatherValueSnapshot('clip'),
    moeValueSnapshot(false),
  ]) {
    assert.doesNotThrow(() => checkedWebGPUPhysicalDomain(
      createBackendCompileInput(snapshot), device,
    ));
  }

  assert.throws(
    () => checkedWebGPUPhysicalDomain(
      createBackendCompileInput(embeddingValueSnapshot('weight', [0, 3])), device,
    ),
    /Embedding IDs invariant I32 payload is outside \[0, 2\]/,
  );
  assert.throws(
    () => checkedWebGPUPhysicalDomain(
      createBackendCompileInput(embeddingValueSnapshot('unsafe')), device,
    ),
    /Embedding IDs has no complete host or static value-range preflight/,
  );
  assert.throws(
    () => checkedWebGPUPhysicalDomain(
      createBackendCompileInput(gatherValueSnapshot('unsafe')), device,
    ),
    /Gather indices has no complete host or static value-range preflight/,
  );
  assert.throws(
    () => checkedWebGPUPhysicalDomain(
      createBackendCompileInput(moeValueSnapshot(true)), device,
    ),
    /MoELinear route indices is produced on device and has no pre-dispatch value validation route/,
  );
  assert.equal(device.state.buffers.length, 0,
    'value-domain proof never allocates context or specialization resources');
});

test('WebGPU preflights invariant routes against each context bank residency', () => {
  const gather = invariantBankGatherSnapshot(0);
  const negativeGather = invariantBankGatherSnapshot(-1);
  const moe = invariantBankMoESnapshot(0);
  const x = { data: Float32Array.of(1, 1), shape: [1, 2] };

  for (const snapshot of [gather, negativeGather, moe]) {
    assert.doesNotThrow(() => checkedWebGPUPhysicalDomain(
      createBackendCompileInput(snapshot), mockDevice(),
    ));
  }

  const gatherResident = createBoundExecutionGraph(gather, resolveGraphShapes(
    gather.graph, {}, gather.quantizationByTensor, { experts: [0] },
  ));
  const gatherMissing = createBoundExecutionGraph(gather, resolveGraphShapes(
    gather.graph, {}, gather.quantizationByTensor, { experts: [1] },
  ));
  assert.doesNotThrow(() => preflightWebGPUExecutionInputs(gatherResident.graph, {}));
  assert.throws(
    () => preflightWebGPUExecutionInputs(gatherMissing.graph, {}),
    /Gather node pick slot 0 is not resident in this context/,
  );

  const negativeResident = createBoundExecutionGraph(negativeGather, resolveGraphShapes(
    negativeGather.graph, {}, negativeGather.quantizationByTensor, { experts: [3] },
  ));
  const negativeMissing = createBoundExecutionGraph(negativeGather, resolveGraphShapes(
    negativeGather.graph, {}, negativeGather.quantizationByTensor, { experts: [2] },
  ));
  assert.doesNotThrow(() => preflightWebGPUExecutionInputs(negativeResident.graph, {}));
  assert.throws(
    () => preflightWebGPUExecutionInputs(negativeMissing.graph, {}),
    /Gather node pick slot 3 is not resident in this context/,
  );

  const moeResident = createBoundExecutionGraph(moe, resolveGraphShapes(
    moe.graph, { x }, moe.quantizationByTensor, { experts: [0] },
  ));
  const moeMissing = createBoundExecutionGraph(moe, resolveGraphShapes(
    moe.graph, { x }, moe.quantizationByTensor, { experts: [1] },
  ));
  assert.doesNotThrow(() => preflightWebGPUExecutionInputs(
    moeResident.graph, { x: x.data },
  ));
  assert.throws(
    () => preflightWebGPUExecutionInputs(moeMissing.graph, { x: x.data }),
    /MoELinear node mix has invalid route index 0 at offset 0/,
  );
});

test('WebGPU admits device-produced bank Gather only for a fully resident context', async () => {
  const snapshot = deviceProducedBankGatherSnapshot();
  const shaderLibrary = {
    ...ShaderLibrary,
    getTypedClipShader: () => 'clip-i32',
    getGatherInt32Shader: () => 'gather-i32',
  };
  const input = {
    raw_slot: { data: Int32Array.of(3), shape: [1] },
  };

  const full = await fixture({ snapshot, shaderLibrary });
  full.device.state.autoCompleteWork = true;
  let fullResult;
  try {
    fullResult = await full.context.execute(input);
    assert.equal(full.device.state.dispatches.length, 2,
      'the default context binds the complete bank and reaches both dispatches');
  } finally {
    await fullResult?.close();
    await full.context.close();
    await full.compiled.close();
    await full.runtime.close();
  }

  const explicitFull = await fixture({
    snapshot,
    contextOptions: { bankResidency: { experts: [0, 1, 2, 3] } },
    shaderLibrary,
  });
  explicitFull.device.state.autoCompleteWork = true;
  let explicitFullResult;
  try {
    explicitFullResult = await explicitFull.context.execute(input);
    assert.equal(explicitFull.device.state.dispatches.length, 2,
      'an explicit identity slot list is also a fully resident bank');
  } finally {
    await explicitFullResult?.close();
    await explicitFull.context.close();
    await explicitFull.compiled.close();
    await explicitFull.runtime.close();
  }

  const partialDevice = mockDevice();
  const partial = await fixture({
    snapshot,
    device: partialDevice,
    contextOptions: { bankResidency: { experts: [0] } },
    shaderLibrary,
  });
  try {
    await assert.rejects(
      partial.context.execute(input),
      /cannot preflight device-produced indices against this context's partial bank residency/,
    );
    assert.equal(partialDevice.state.dispatches.length, 0,
      'an unsafe partial-residency plan fails before any dispatch');
  } finally {
    partialDevice.state.autoCompleteWork = true;
    await partial.context.close();
    await partial.compiled.close();
    await partial.runtime.close();
  }
});

test('WebGPU rejects MoERouter geometry beyond its shader domain before context publication', async () => {
  await assert.rejects(
    fixture({ snapshot: unsupportedWideMoERouterSnapshot() }),
    requiredCompileFailure(/MoERouter top_k 9 exceeds the shader's proved maximum 8/),
  );
});

test('WebGPU rejects non-finite invariant quantized norm affines during physical proof', async () => {
  await assert.rejects(
    fixture({ snapshot: invalidStaticQLayerNormAffineSnapshot() }),
    requiredCompileFailure(/QLayerNorm weight must be one finite invariant F32 tensor/),
  );
});

test('WebGPU proves and precompiles dynamic broadcast arithmetic during compile', async () => {
  const device = mockDevice();
  const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
  const provider = new WebGPUBackendProvider(engine);
  const runtime = new Runtime();
  runtime._addProvider('webgpu', provider);
  const compiled = await runtime.compile(dynamicAddSnapshot(), {
    backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
  });
  assert.equal(device.state.pipelineCreates, 1);
  assert.equal(device.state.buffers.length, 0,
    'bounded-domain compilation does not allocate context tensor arenas');
  await compiled.close();
  device.state.autoCompleteWork = true;
  assert.equal(device.state.buffers.length, 0);
  await runtime.close();
});

test('WebGPU physical proof uses legal multiple_of maxima and rejects unequal Split tactics', async () => {
  {
    const device = mockDevice({ maxComputeWorkgroupsPerDimension: 2 });
    const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
    const provider = new WebGPUBackendProvider(engine);
    const runtime = new Runtime();
    runtime._addProvider('webgpu', provider);
    const compiled = await runtime.compile(dynamicReluSnapshot({
      batchMax: 1,
      sequenceMax: 130,
      sequenceMultipleOf: 64,
    }), {
      backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    });
    assert.equal(device.state.pipelineCreates, 1,
      'raw max 130 normalizes to the legal multiple_of maximum 128');
    await compiled.close();
    await runtime.close();
  }

  {
    const device = mockDevice({ maxComputeWorkgroupsPerDimension: 4 });
    device.state.autoCompleteWork = true;
    const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
    const provider = new WebGPUBackendProvider(engine);
    const runtime = new Runtime();
    runtime._addProvider('webgpu', provider);
    const compiled = await runtime.compile(dynamicReluSnapshot(), {
      backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    });
    const context = await compiled.createContext();
    const input = Float32Array.from({ length: 260 }, (_, index) => index - 130);
    const result = await context.execute({ x: { data: input, shape: [2, 130] } });
    assert.deepEqual(
      device.state.dispatches.slice(-1).map(({ x, y, z }) => [x, y, z]),
      [[4, 2, 1]],
      'a linear workload wider than one dispatch dimension is tiled over y',
    );
    assert.deepEqual(
      [...await result.output('y').read()],
      [...input].map((value) => Math.max(0, value)),
    );
    await result.close();
    await context.close();
    await compiled.close();
    await runtime.close();
  }

  {
    const device = mockDevice({ maxComputeWorkgroupsPerDimension: 4 });
    const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
    const provider = new WebGPUBackendProvider(engine);
    const runtime = new Runtime();
    runtime._addProvider('webgpu', provider);
    const compiled = await runtime.compile(dynamicReluSnapshot({ opType: 'Identity' }), {
      backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    });
    assert.equal(compiled.backend, 'webgpu',
      'shape-only copies tile their complete bounded domain over two dimensions');
    await compiled.close();
    await runtime.close();
  }

  {
    const device = mockDevice();
    const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
    const provider = new WebGPUBackendProvider(engine);
    const runtime = new Runtime();
    runtime._addProvider('webgpu', provider);
    await assert.rejects(runtime.compile(unequalSplitSnapshot(), {
      backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    }), requiredCompileFailure(/supports only equal-sized output slices/));
    assert.equal(device.state.pipelineCreates, 0);
    assert.equal(device.state.buffers.length, 0);
    await runtime.close();
  }

  {
    const device = mockDevice();
    const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
    const provider = new WebGPUBackendProvider(engine);
    const runtime = new Runtime();
    runtime._addProvider('webgpu', provider);
    const snapshot = inputMajorMatMulSnapshot();
    const compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    });
    assert.equal(device.state.pipelineCreates, 2,
      'the B=1 scalar and B>1 tiled input-major routes are proved before publication');
    assert.deepEqual(
      device.state.buffers.map((buffer) => buffer.descriptor.label).sort(),
      snapshot.weightNames.map((name) => `Tensor_${name}`).sort(),
    );
    await compiled.close();
    await runtime.close();
  }
});

test('WebGPU rejects required shader compilation and device limits before context creation', async () => {
  for (const [device, message] of [
    [Object.assign(mockDevice(), {}), /required bounded-domain shader route/],
    [mockDevice({ maxBufferSize: 1000 }), /maxBufferSize or maxStorageBufferBindingSize/],
    [mockDevice({ maxStorageBufferBindingSize: 1000 }), /maxBufferSize or maxStorageBufferBindingSize/],
    [mockDevice({ maxComputeWorkgroupsPerDimension: 2 }), /uniform elementwise dispatch/],
    [mockDevice({ maxComputeWorkgroupStorageSize: 8575 }), /below the proved route requirement 8576/],
    [mockDevice({ maxBindingsPerBindGroup: 7 }), /below the proved route requirement 8/],
    [mockDevice({ maxStorageBuffersPerShaderStage: 6 }), /below the proved route requirement 7/],
    [mockDevice({ maxUniformBufferBindingSize: 15 }), /elementwise parameters uniform allocation 16/],
  ]) {
    if (message.source.includes('shader')) device.state.failPipelineCode = 'dynamic-relu';
    const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
    const provider = new WebGPUBackendProvider(engine);
    const runtime = new Runtime();
    runtime._addProvider('webgpu', provider);
    await assert.rejects(runtime.compile(dynamicReluSnapshot(), {
      backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    }), requiredCompileFailure(message));
    assert.equal(device.state.buffers.length, 0);
    await runtime.close();
  }

  for (const [snapshot, device, message] of [
    [dynamicReluSnapshot({ batchMax: 2, sequenceMax: 0x80000000 }),
      mockDevice({ maxComputeWorkgroupsPerDimension: 0xffffffff }),
      /u32 descriptor arithmetic/],
    [dynamicQLinearSnapshot(), mockDevice({ maxComputeWorkgroupsPerDimension: 1 }),
      /QLinear tiled dispatch/],
    [dynamicQLinearSnapshot({ biasValue: 0x7fffffff }), mockDevice(),
      /affine or I32 accumulator bound is not representable/],
  ]) {
    const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
    const provider = new WebGPUBackendProvider(engine);
    const runtime = new Runtime();
    runtime._addProvider('webgpu', provider);
    await assert.rejects(runtime.compile(snapshot, {
      backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    }), requiredCompileFailure(message));
    assert.equal(device.state.buffers.length, 0);
    await runtime.close();
  }
});

test('failed bind-group specialization is rolled back without touching the prior uniforms', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const first = await context.execute(shaped(32));
  device.state.failNextBindGroup = true;
  await assert.rejects(context.execute(shaped(129)), /injected bind-group specialization failure/);
  const recovered = await context.execute(shaped(32, 7));
  assert.deepEqual([...await recovered.output('y').read()],
    [...shaped(32, 7).x.data].map((value) => Math.max(0, value)));
  assert.equal(recovered.report.backendReport.specializationRebindCount, 1);
  assert.equal(recovered.report.backendReport.specializationBufferCreateCount, 1);
  device.state.autoCompleteWork = true;
  await context.close();
  await compiled.close();
  await runtime.close();
  await Promise.all([first.close(), recovered.close()]);
});

test('device loss makes the WebGPU context terminal and close still releases context buffers', async () => {
  const { device, runtime, compiled, context } = await fixture();
  const result = await context.execute(shaped(16));
  await device.state.loseDevice();
  await assert.rejects(context.execute(shaped(16)), (error) => error?.code === 'DEVICE_LOST');
  await context.close();
  const contextBuffers = device.state.buffers.filter((buffer) =>
    String(buffer.descriptor.label || '').startsWith('Tensor_'));
  assert.ok(contextBuffers.length > 0 && contextBuffers.every((buffer) => buffer.destroyed));
  await compiled.close();
  await runtime.close();
  assert.equal(result.closed, false,
    'device-owned results remain independently releasable after parent cleanup');
  await result.close();
});

test('device loss invalidates the compiled invariant epoch and destroys owner buffers once', async () => {
  const snapshot = dynamicQLinearSnapshot();
  const device = mockDevice();
  const provider = new WebGPUBackendProvider(
    new WebGPUEngine(device, { shaderLibrary: ShaderLibrary }),
  );
  const compiled = await provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  });
  const owner = compiled.invariantResources;
  const lease = owner.open();
  const epoch = owner.deviceEpoch;
  const ownerBuffers = device.state.buffers.filter((buffer) =>
    String(buffer.descriptor.label || '').startsWith('Tensor_'));
  assert.ok(owner.resourceCount >= 2, 'host and device invariant resources are compiled-owned');
  assert.ok(owner.ownedBytes > 0);
  assert.ok(ownerBuffers.length > 0 && ownerBuffers.every((buffer) => !buffer.destroyed));

  await device.state.loseDevice();
  assert.notEqual(owner.deviceEpoch, epoch);
  assert.equal(owner.resourceCount, 0);
  assert.equal(owner.ownedBytes, 0);
  assert.ok(ownerBuffers.every((buffer) => buffer.destroyed));
  assert.throws(() => owner.open(), /device epoch is invalid/);
  assert.throws(() => lease.borrow(`host-weight:${snapshot.weightNames[0]}`),
    /device epoch is invalid/);

  lease.release();
  await compiled.close();
  await provider.close();
});

test('scoped invariant-weight allocation failure publishes no compiled owner and retry succeeds', async () => {
  const snapshot = dynamicQLinearSnapshot();
  const device = mockDevice();
  const scopes = [];
  device.pushErrorScope = (filter) => { scopes.push({ filter, error: null }); };
  device.popErrorScope = () => Promise.resolve(scopes.pop()?.error ?? null);
  const originalCreateBuffer = device.createBuffer;
  let failInvariant = true;
  device.createBuffer = function createBufferWithScopedInvariantOOM(descriptor) {
    const buffer = originalCreateBuffer.call(this, descriptor);
    if (failInvariant && String(descriptor.label || '').startsWith('Tensor_')) {
      failInvariant = false;
      const scope = [...scopes].reverse().find(({ filter }) => filter === 'out-of-memory');
      scope.error = { name: 'GPUOutOfMemoryError', message: 'invariant allocation failed' };
    }
    return buffer;
  };
  const provider = new WebGPUBackendProvider(
    new WebGPUEngine(device, { shaderLibrary: ShaderLibrary }),
  );

  await assert.rejects(provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  }), (error) => error?.code === 'OUT_OF_MEMORY' &&
    /invariant allocation failed/.test(error.message));
  const rejectedBuffers = device.state.buffers.filter((buffer) =>
    String(buffer.descriptor.label || '').startsWith('Tensor_'));
  assert.ok(rejectedBuffers.length > 0 && rejectedBuffers.every((buffer) => buffer.destroyed));

  const compiled = await provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  });
  assert.ok(compiled.invariantResources.resourceCount >= 2);
  assert.equal(compiled.compilationEvidence.allocationBytes,
    compiled.invariantResources.ownedBytes);
  await compiled.close();
  await provider.close();
});

test('synchronous invariant-weight allocation failure is classified as compilation OOM', async () => {
  const snapshot = dynamicQLinearSnapshot();
  const device = mockDevice();
  const originalCreateBuffer = device.createBuffer;
  let failInvariant = true;
  device.createBuffer = function createBufferWithSynchronousInvariantOOM(descriptor) {
    if (failInvariant && String(descriptor.label || '').startsWith('Tensor_')) {
      failInvariant = false;
      throw new Error('synchronous invariant allocation failed');
    }
    return originalCreateBuffer.call(this, descriptor);
  };
  const provider = new WebGPUBackendProvider(
    new WebGPUEngine(device, { shaderLibrary: ShaderLibrary }),
  );

  await assert.rejects(provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  }), (error) => error?.code === 'OUT_OF_MEMORY' &&
    error?.phase === 'compilation' &&
    /synchronous invariant allocation failed/.test(error.message));
  assert.equal(device.state.buffers.length, 0);

  const compiled = await provider.compile(createBackendCompileInput(snapshot), {
    operatorFallback: 'forbid',
  });
  await compiled.close();
  await provider.close();
});
