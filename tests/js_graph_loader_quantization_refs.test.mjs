import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/core/Graph.js';
import { GraphLoader } from '../ts/core/GraphLoader.js';
import { SafetensorsFile } from '../ts/core/Safetensors.js';

const FORMAT = 'volvox-affine-safetensors/v1';

function tensorFile(overrides = {}, metadata = {}) {
  const definitions = {
    w: ['I8', [2, 2], Int8Array.of(1, 2, 3, 4)],
    bias: ['I32', [2], Int32Array.of(0, 0)],
    sx: ['F32', [1], Float32Array.of(0.125)],
    zx: ['I8', [1], Int8Array.of(-3)],
    sw: ['F32', [2], Float32Array.of(0.25, 1 / 3)],
    zw: ['I8', [2], Int8Array.of(0, 0)],
    sy: ['F32', [1], Float32Array.of(0.5)],
    zy: ['U8', [1], Uint8Array.of(127)],
    ...overrides,
  };
  const file = SafetensorsFile.empty({ metadata });
  for (const [name, [dtype, shape, values]] of Object.entries(definitions)) {
    if (dtype != null) file.addTensor(name, dtype, shape, values);
  }
  return file.toArrayBuffer();
}

function graphDocument() {
  return {
    format: 'volvox-graph/v1',
    inputs: { x: { shape: [1, 2], dtype: 'int8' } },
    quantization: {
      format: FORMAT,
      tensors: {
        x: {
          scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx',
        },
        w: {
          scheme: 'per_axis', axis: 0, scale_tensor: 'sw', zero_point_tensor: 'zw',
        },
        y: {
          scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy',
        },
      },
    },
    nodes: [{
      opType: 'QLinear',
      inputs: { input: 'x', weight: 'w', bias: 'bias' },
      outputs: { out: 'y' },
      outputs_shape: { out: [1, 2] },
      outputs_dtype: { out: 'uint8' },
      params: {},
    }],
    outputs: ['y'],
  };
}

function attentionDocument() {
  return {
    format: 'volvox-graph/v1',
    inputs: {
      q: { shape: [1, 2, 8], dtype: 'float32' },
      k: { shape: [1, 3, 8], dtype: 'float32' },
      v: { shape: [1, 3, 8], dtype: 'float32' },
    },
    nodes: [{
      id: 'attention',
      opType: 'CrossSDPA',
      inputs: { q: 'q', k: 'k', v: 'v' },
      outputs: { out: 'y' },
      outputs_shape: { out: [1, 2, 8] },
      outputs_dtype: { out: 'float32' },
      params: { heads: 2, causal: false, scale: 0.5 },
    }],
    outputs: ['y'],
  };
}

async function load(document, buffer = tensorFile()) {
  const fetch = async (url) => {
    if (url === 'graph.json') return { ok: true, json: async () => document };
    if (url === 'model.safetensors') {
      return { ok: true, arrayBuffer: async () => buffer };
    }
    return { ok: false, statusText: 'not found' };
  };
  return GraphLoader.load(new Graph(), ['model.safetensors'], {
    graphUrl: 'graph.json', fetch,
  });
}

test('GraphLoader hydrates v1 affine references from safetensors', async () => {
  const graph = await load(graphDocument());

  assert.deepEqual(graph.getTensor('x').quantization, {
    scheme: 'per_tensor', scale: 0.125, zero_point: -3,
  });
  assert.deepEqual(graph.getTensor('w').quantization, {
    scheme: 'per_axis', axis: 0,
    scales: [0.25, Math.fround(1 / 3)], zero_points: [0, 0],
  });
  assert.deepEqual(graph.getTensor('y').quantization, {
    scheme: 'per_tensor', scale: 0.5, zero_point: 127,
  });
  assert.equal(Object.isFrozen(graph.getTensor('w').quantization), true);
  assert.deepEqual([...graph.getTensor('sw').buffer], [0.25, Math.fround(1 / 3)]);
});

test('GraphLoader v1 rejects every legacy inline quantization location', async () => {
  const input = graphDocument();
  input.inputs.x.quantization = { scale: 1, zero_point: 0 };
  await assert.rejects(load(input), /forbidden inline quantization/);

  const output = graphDocument();
  output.nodes[0].outputs_quantization = {
    out: { scale: 1, zero_point: 0 },
  };
  await assert.rejects(load(output), /forbidden inline quantization/);

  const weight = graphDocument();
  weight.weights_quantization = {};
  await assert.rejects(load(weight), /forbids legacy field 'weights_quantization'/);

  const companion = graphDocument();
  companion.weights_quantization_storage = {
    format: 'volvoxai-f32-companion-scales-v1',
  };
  await assert.rejects(
    load(companion),
    /forbids legacy field 'weights_quantization_storage'/,
  );

  const standalone = graphDocument();
  standalone.standaloneTensors = {
    state: {
      shape: [1], dtype: 'int8', hasBuffer: false,
      quantization: { scale: 1, zero_point: 0 },
    },
  };
  await assert.rejects(load(standalone), /Standalone tensor 'state'.*inline quantization/);

  const centralInline = graphDocument();
  centralInline.quantization.tensors.x = {
    scheme: 'per_tensor', scale: 0.125, zero_point: -3,
  };
  await assert.rejects(
    load(centralInline),
    /Quantization descriptor for 'x' has unsupported field 'scale'/,
  );
});

test('GraphLoader recursively rejects retired affine params but keeps attention scale', async (t) => {
  const accepted = await load(attentionDocument());
  assert.equal(accepted.nodes[0].params.scale, 0.5);

  const forbidden = [
    'quantization', 'zero_point',
    'input_scale', 'input_zero_point',
    'output_scale', 'output_zero_point',
    'weight_scale', 'weight_zero_point',
    'scales', 'zero_points',
    'scale_tensor', 'zero_point_tensor',
  ];
  for (const field of forbidden) {
    await t.test(field, async () => {
      const document = attentionDocument();
      document.nodes[0].params.private = [{ affine: { [field]: 0.25 } }];
      await assert.rejects(
        load(document),
        (error) => {
          assert.match(error.message, /retired affine payload/);
          assert.ok(
            error.message.includes(`params.private[0].affine.${field}`),
            error.message,
          );
          return true;
        },
      );
    });
  }
});

test('GraphLoader accepts only volvox-graph/v1', async () => {
  const document = graphDocument();
  document.format = 'volvox-graph/v2';
  await assert.rejects(
    load(document),
    /graph\.json format must be 'volvox-graph\/v1'.*volvox-graph\/v2/,
  );
});

test('GraphLoader rejects obsolete companion metadata in safetensors', async () => {
  const metadata = {
    weights_quantization_storage: 'volvoxai-f32-companion-scales-v1',
    weights_quantization: JSON.stringify({ w: { scheme: 'per_axis', axis: 0 } }),
  };
  await assert.rejects(
    load(graphDocument(), tensorFile({}, metadata)),
    /forbidden legacy metadata 'weights_quantization_storage'/,
  );
});

test('GraphLoader validates referenced parameter dtype, shape, and values', async () => {
  await assert.rejects(
    load(graphDocument(), tensorFile({ sx: ['F16', [1], Uint16Array.of(0)] })),
    /Scale tensor 'sx'.*rank-1 F32/,
  );
  await assert.rejects(
    load(graphDocument(), tensorFile({ sw: ['F32', [1], Float32Array.of(0.25)] })),
    /parameters for 'w' must have shape \[2\]/,
  );
  await assert.rejects(
    load(graphDocument(), tensorFile({ sy: ['F32', [1], Float32Array.of(0)] })),
    /scales for 'y' must be finite and positive/,
  );
  await assert.rejects(
    load(graphDocument(), tensorFile({ zy: ['I8', [1], Int8Array.of(0)] })),
    /Zero-point tensor 'zy'.*rank-1 U8/,
  );
});

test('GraphLoader requires Q/DQ operands to use the descriptor references', async () => {
  const document = graphDocument();
  document.inputs = { f: { shape: [2], dtype: 'float32' } };
  document.nodes = [{
    opType: 'QuantizeLinear',
    inputs: { input: 'f', scale: 'wrong_scale', zero_point: 'zy' },
    outputs: { out: 'y' },
    outputs_shape: { out: [2] },
    outputs_dtype: { out: 'uint8' },
    params: {},
  }];
  document.quantization.tensors = { y: document.quantization.tensors.y };
  const buffer = tensorFile({
    wrong_scale: ['F32', [1], Float32Array.of(0.5)],
  });

  await assert.rejects(load(document, buffer), /parameter inputs must match.*quantization references/);
});

test('GraphLoader prevents quantization parameters from becoming public outputs', async () => {
  const document = graphDocument();
  document.outputs = ['sy'];
  await assert.rejects(load(document), /quantization parameter tensors cannot be public graph outputs/);
});

test('GraphLoader prevents quantization parameters from entering graph topology', async () => {
  const input = graphDocument();
  input.inputs.sy = { shape: [1], dtype: 'float32' };
  await assert.rejects(
    load(input),
    /Quantization parameter 'sy'.*cannot be exposed as a graph input/,
  );

  const output = graphDocument();
  output.quantization.tensors = {
    x: output.quantization.tensors.x,
    w: output.quantization.tensors.w,
  };
  output.nodes[0].outputs.out = 'sx';
  output.outputs = ['sx'];
  await assert.rejects(
    load(output),
    /Quantization parameter 'sx'.*cannot be produced by a graph node/,
  );
});
