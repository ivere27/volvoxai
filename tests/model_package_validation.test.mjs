import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { validateModelPackages } from '../tools/validate_model_packages.mjs';
import { SafetensorsFile } from '../ts/core/Safetensors.js';

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'volvox-model-package-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function writeJson(filename, value) {
  await mkdir(path.dirname(filename), { recursive: true });
  await writeFile(filename, `${JSON.stringify(value)}\n`);
}

async function writeEmptySafetensors(filename) {
  await mkdir(path.dirname(filename), { recursive: true });
  await writeFile(filename, new Uint8Array(SafetensorsFile.empty().toArrayBuffer()));
}

function rawSafetensors(header, data = new Uint8Array()) {
  const headerBytes = new TextEncoder().encode(
    typeof header === 'string' ? header : JSON.stringify(header),
  );
  const bytes = new Uint8Array(8 + headerBytes.byteLength + data.byteLength);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  bytes.set(headerBytes, 8);
  bytes.set(data, 8 + headerBytes.byteLength);
  return bytes;
}

function basicGraph(overrides = {}) {
  return {
    format: 'volvox-graph/v1',
    inputs: { x: { shape: [1], dtype: 'float32' } },
    nodes: [],
    outputs: ['x'],
    ...overrides,
  };
}

test('model package validator accepts canonical primary and subgraph documents', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), {
    format: 'volvox-graph/v1', inputs: { x: { shape: [1], dtype: 'float32' } },
    nodes: [], outputs: ['x'],
  });
  await writeJson(path.join(root, 'router.graph.json'), {
    format: 'volvox-graph/v1', inputs: { route: { shape: [1], dtype: 'int32' } },
    nodes: [], outputs: ['route'],
  });
  await writeJson(path.join(root, 'config.json'), { model_type: 'external-training-settings' });
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));

  const files = await validateModelPackages([root]);
  assert.deepEqual(files.map((filename) => path.basename(filename)), [
    'graph.json', 'router.graph.json',
  ]);
});

test('model package validator accepts a semantically valid known operator', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), {
    format: 'volvox-graph/v1',
    inputs: { x: { shape: [2, 4], dtype: 'float32' } },
    nodes: [{
      opType: 'Softmax',
      inputs: { input: 'x' },
      outputs: { out: 'y' },
      outputs_shape: { out: [2, 4] },
      outputs_dtype: { out: 'float32' },
      params: { axis: -1 },
    }],
    outputs: ['y'],
  });
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));

  await assert.doesNotReject(validateModelPackages([root]));
});

test('model package validator fails closed when semantic validation is unavailable', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), basicGraph());
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));
  const previousPython = process.env.PYTHON;
  process.env.PYTHON = path.join(root, 'missing-python');
  try {
    await assert.rejects(
      validateModelPackages([root]),
      /strict Python RuntimeIR validation failed/,
    );
  } finally {
    if (previousPython === undefined) delete process.env.PYTHON;
    else process.env.PYTHON = previousPython;
  }
});

test('model package validator rejects malformed contracts for known operators', async (t) => {
  const cases = [
    ['QConv2D storage contract', {
      inputs: {
        x: { shape: [1, 4, 4, 3], dtype: 'float32' },
        w: { shape: [2, 3, 3, 3], dtype: 'float32' },
      },
      nodes: [{
        opType: 'QConv2D', inputs: { input: 'x', weight: 'w' }, outputs: { out: 'y' },
        outputs_shape: { out: [1, 2, 2, 2] }, outputs_dtype: { out: 'float32' },
        params: {},
      }],
      outputs: ['y'],
    }, /QConv2D.*canonical physical I8\/U8/],
    ['Conv2D channel geometry', {
      inputs: {
        x: { shape: [1, 4, 4, 3], dtype: 'float32' },
        w: { shape: [2, 3, 3, 4], dtype: 'float32' },
      },
      nodes: [{
        opType: 'Conv2D', inputs: { input: 'x', weight: 'w' }, outputs: { out: 'y' },
        outputs_shape: { out: [1, 2, 2, 2] }, outputs_dtype: { out: 'float32' },
        params: { weight_layout: 'OHWI' },
      }],
      outputs: ['y'],
    }, /supported F32 NHWC rank-4 convolution geometry/],
    ['Softmax canonical axis', {
      inputs: { x: { shape: [2, 4], dtype: 'float32' } },
      nodes: [{
        opType: 'Softmax', inputs: { input: 'x' }, outputs: { out: 'y' },
        outputs_shape: { out: [2, 4] }, outputs_dtype: { out: 'float32' },
        params: { axis: 0 },
      }],
      outputs: ['y'],
    }, /must use the canonical last axis/],
  ];

  for (const [label, document, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      await writeJson(path.join(root, 'graph.json'), {
        format: 'volvox-graph/v1',
        ...document,
      });
      await writeEmptySafetensors(path.join(root, 'model.safetensors'));
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

test('model package validator fails wrong formats and graph documents under noncanonical names', async (t) => {
  const cases = [
    ['graph.json', { nodes: [] }, /format must be exactly/],
    ['graph.json', { format: 'Volvox-Graph\/v1', nodes: [] }, /format must be exactly/],
    ['graph.json', {
      format: 'volvox-graph/v1', inputs: { x: { shape: [1], dtype: 'F32' } },
      nodes: [], outputs: ['x'],
    }, /canonical lowercase dtype/],
    ['graph.json', { format: 'volvox-graph/v1', nodes: [{ op: 'Identity' }] }, /requires a non-empty opType/],
    ['graph.json', {
      format: 'volvox-graph/v1', inputs: { x: { shape: [1], dtype: 'float32' } },
      nodes: [{
        op: 'Identity', opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
        outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' },
      }], outputs: ['y'],
    }, /forbids alias field 'op'/],
    ['graph.json', {
      format: 'volvox-graph/v1', inputs: { x: { shape: [1], dtype: 'float32' } },
      nodes: [{
        opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
        outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' }, params: [],
      }], outputs: ['y'],
    }, /params must be an object/],
    ['graph.json', {
      format: 'volvox-graph/v1', inputs: { x: { shape: [1], dtype: 'float32' } },
      nodes: [{
        opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
        outputs_shape: { out: [1] }, outputs_dtype: { out: 'float16' },
      }], outputs: ['y'],
    }, /outputs_dtype must use canonical lowercase dtypes/],
    ['graph.json', {
      format: 'volvox-graph/v1', inputs: { x: { shape: [1], dtype: 'float32' } },
      nodes: [{
        opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
        outputs_shape: { out: { shape: [1], dtype: 'int8' } },
      }], outputs: ['y'],
    }, /outputs_shape must provide one positive-integer shape array per output/],
    ['graph.json', { format: 'volvox-graph/v1', nodes: [], outputs: { y: 'y' } }, /outputs must be a non-empty array/],
    ['graph.json', { format: 'volvox-graph/v1', nodes: [], outputs: [] }, /outputs must be a non-empty array/],
    ['graph.json', { format: 'volvox-graph/v1', nodes: [] }, /outputs must be a non-empty array/],
    ['model.json', { format: 'volvox-graph/v1', nodes: [] }, /must use graph\.json/],
  ];
  for (const [filename, document, expected] of cases) {
    await t.test(filename + JSON.stringify(document), async () => {
      const root = await fixture(t);
      await writeJson(path.join(root, filename), document);
      await writeEmptySafetensors(path.join(root, 'model.safetensors'));
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

test('model package validator requires a primary graph.json beside optional subgraphs', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'router.graph.json'), {
    format: 'volvox-graph/v1',
    inputs: { route: { shape: [1], dtype: 'int32' } },
    nodes: [],
    outputs: ['route'],
  });
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));
  await assert.rejects(validateModelPackages([root]), /no primary graph\.json file found/);
});

test('model package validator requires at least one safetensors weight file', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), {
    format: 'volvox-graph/v1',
    inputs: { x: { shape: [1], dtype: 'float32' } },
    nodes: [],
    outputs: ['x'],
  });
  await assert.rejects(validateModelPackages([root]), /no \*\.safetensors weight file found/);
});

test('model package validator enforces central safetensors affine references', async (t) => {
  const root = await fixture(t);
  const weights = SafetensorsFile.empty();
  weights.addTensor('q.scale', 'F32', [1], Float32Array.of(0.25));
  weights.addTensor('q.zero_point', 'I8', [1], Int8Array.of(0));
  await writeFile(path.join(root, 'model.safetensors'), new Uint8Array(weights.toArrayBuffer()));
  const graph = {
    format: 'volvox-graph/v1',
    inputs: { x: { shape: [2], dtype: 'float32' } },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        q: {
          scheme: 'per_tensor', scale_tensor: 'q.scale', zero_point_tensor: 'q.zero_point',
        },
      },
    },
    nodes: [{
      opType: 'QuantizeLinear',
      inputs: { input: 'x', scale: 'q.scale', zero_point: 'q.zero_point' },
      outputs: { out: 'q' }, outputs_shape: { out: [2] }, outputs_dtype: { out: 'int8' },
    }],
    outputs: ['q'],
  };
  await writeJson(path.join(root, 'graph.json'), graph);
  await assert.doesNotReject(validateModelPackages([root]));

  graph.nodes[0].outputs_quantization = {
    out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  };
  await writeJson(path.join(root, 'graph.json'), graph);
  await assert.rejects(validateModelPackages([root]), /forbids inline outputs_quantization/);
  delete graph.nodes[0].outputs_quantization;

  graph.standaloneTensors = {
    state: {
      shape: [1], dtype: 'int8', hasBuffer: false,
      quantization: { scale: 1, zero_point: 0 },
    },
  };
  await writeJson(path.join(root, 'graph.json'), graph);
  await assert.rejects(validateModelPackages([root]), /standalone tensor "state" forbids inline quantization/);
  delete graph.standaloneTensors;

  graph.nodes[0].inputs.scale = 'wrong.scale';
  await writeJson(path.join(root, 'graph.json'), graph);
  await assert.rejects(validateModelPackages([root]), /operands must match its central output references/);
});

test('model package validator scopes safetensors names to the nearest split graph directory', async (t) => {
  const root = await fixture(t);
  for (const role of ['encoder', 'decoder']) {
    const directory = path.join(root, role);
    const weights = SafetensorsFile.empty();
    weights.addTensor('shared.scale', 'F32', [1], Float32Array.of(0.25));
    weights.addTensor('shared.zero_point', 'I8', [1], Int8Array.of(0));
    await mkdir(directory, { recursive: true });
    await writeFile(
      path.join(directory, 'model.safetensors'),
      new Uint8Array(weights.toArrayBuffer()),
    );
    await writeJson(path.join(directory, 'graph.json'), {
      format: 'volvox-graph/v1',
      inputs: { x: { shape: [2], dtype: 'float32' } },
      quantization: {
        format: 'volvox-affine-safetensors/v1',
        tensors: {
          q: {
            scheme: 'per_tensor',
            scale_tensor: 'shared.scale',
            zero_point_tensor: 'shared.zero_point',
          },
        },
      },
      nodes: [{
        opType: 'QuantizeLinear',
        inputs: { input: 'x', scale: 'shared.scale', zero_point: 'shared.zero_point' },
        outputs: { out: 'q' }, outputs_shape: { out: [2] }, outputs_dtype: { out: 'int8' },
      }],
      outputs: ['q'],
    });
  }

  await assert.doesNotReject(validateModelPackages([root]));
});

test('model package validator rejects duplicate JSON keys before interpreting graph format', async (t) => {
  const root = await fixture(t);
  await writeFile(
    path.join(root, 'graph.json'),
    '{"format":"volvox-graph/v2","format":"volvox-graph/v1","inputs":{"x":{"shape":[1],"dtype":"float32"}},"nodes":[],"outputs":["x"]}',
  );
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));
  await assert.rejects(validateModelPackages([root]), /duplicate object key "format"/);
});

test('model package validator always parses safetensors for non-quantized graphs', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), basicGraph());
  await writeFile(path.join(root, 'model.safetensors'), new Uint8Array());
  await assert.rejects(validateModelPackages([root]), /safetensors header is truncated/);
});

test('model package validator enforces exact safetensors tensor spans and coverage', async (t) => {
  const tensor = (dtype, shape, data_offsets) => ({ dtype, shape, data_offsets });
  const cases = [
    ['invalid header length', rawSafetensors(''), /header length is invalid/],
    ['invalid header JSON', rawSafetensors('{'), /invalid safetensors header/],
    ['duplicate header key', rawSafetensors(
      '{"w":{"dtype":"F32","dtype":"I8","shape":[1],"data_offsets":[0,1]}}',
      Uint8Array.of(0),
    ), /duplicate object key "dtype"/],
    ['empty tensor name', rawSafetensors({ '': tensor('I8', [1], [0, 1]) }, Uint8Array.of(0)), /invalid header record/],
    ['extra tensor field', rawSafetensors({ w: { ...tensor('I8', [1], [0, 1]), extra: true } }, Uint8Array.of(0)), /invalid header record/],
    ['invalid metadata', rawSafetensors({ __metadata__: null }), /metadata must contain only string values/],
    ['unsupported dtype', rawSafetensors({ w: tensor('F64', [1], [0, 8]) }, new Uint8Array(8)), /unsupported dtype/],
    ['invalid shape', rawSafetensors({ w: tensor('I8', [-1], [0, 0]) }), /invalid shape/],
    ['non-integer offset', rawSafetensors({ w: tensor('I8', [1], [0, 1.5]) }, Uint8Array.of(0)), /invalid data offsets/],
    ['out-of-bounds span', rawSafetensors({ w: tensor('F32', [1], [0, 4]) }, new Uint8Array(3)), /data span is invalid/],
    ['wrong byte span', rawSafetensors({ w: tensor('F32', [1], [0, 3]) }, new Uint8Array(3)), /data span is invalid/],
    ['overlap', rawSafetensors({
      a: tensor('I8', [2], [0, 2]), b: tensor('I8', [2], [1, 3]),
    }, new Uint8Array(3)), /gap, overlap, or aliased span/],
    ['alias', rawSafetensors({
      a: tensor('I8', [1], [0, 1]), b: tensor('I8', [1], [0, 1]),
    }, new Uint8Array(1)), /gap, overlap, or aliased span/],
    ['gap', rawSafetensors({
      a: tensor('I8', [1], [0, 1]), b: tensor('I8', [1], [2, 3]),
    }, new Uint8Array(3)), /gap, overlap, or aliased span/],
    ['unaccounted payload', rawSafetensors({
      a: tensor('I8', [1], [0, 1]),
    }, new Uint8Array(2)), /do not account for the full data payload/],
  ];
  for (const [label, weights, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      await writeJson(path.join(root, 'graph.json'), basicGraph());
      await writeFile(path.join(root, 'model.safetensors'), weights);
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

test('model package validator accepts complete contiguous supported safetensors payloads', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), basicGraph({
    inputs: { x: { shape: [1, 1, 1, 1], dtype: 'float32' } },
    nodes: [{
      opType: 'Conv2D', inputs: { input: 'x', weight: 'half_weight' }, outputs: { out: 'y' },
      outputs_shape: { out: [1, 1, 1, 1] }, outputs_dtype: { out: 'float32' },
      params: { weight_layout: 'OHWI' },
    }],
    outputs: ['y'],
  }));
  await writeFile(path.join(root, 'model.safetensors'), rawSafetensors({
    half_weight: { dtype: 'F16', shape: [1, 1, 1, 1], data_offsets: [0, 2] },
    integer_weight: { dtype: 'I32', shape: [1], data_offsets: [2, 6] },
  }, new Uint8Array(6)));
  await assert.doesNotReject(validateModelPackages([root]));
});

test('model package validator resolves node inputs and public outputs topologically', async (t) => {
  const cases = [
    ['missing node input', {
      nodes: [{
        opType: 'Identity', inputs: { input: 'missing' }, outputs: { out: 'y' },
        outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' },
      }], outputs: ['y'],
    }, /input "missing" is unresolved or not topologically available/],
    ['forward node input', {
      nodes: [
        {
          opType: 'Identity', inputs: { input: 'later' }, outputs: { out: 'first' },
          outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' },
        },
        {
          opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'later' },
          outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' },
        },
      ], outputs: ['first'],
    }, /input "later" is unresolved or not topologically available/],
    ['missing public output', { outputs: ['missing'] }, /public output "missing" is unresolved/],
    ['duplicate produced tensor', {
      nodes: [{
        opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'x' },
        outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' },
      }], outputs: ['x'],
    }, /output "x" collides with an existing tensor/],
  ];
  for (const [label, overrides, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      await writeJson(path.join(root, 'graph.json'), basicGraph(overrides));
      await writeEmptySafetensors(path.join(root, 'model.safetensors'));
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

test('model package validator recursively rejects retired affine params but permits semantic scale', async (t) => {
  const retired = [
    'quantization', 'zero_point',
    'input_scale', 'input_zero_point', 'output_scale', 'output_zero_point',
    'weight_scale', 'weight_zero_point', 'scales', 'zero_points',
    'scale_tensor', 'zero_point_tensor',
  ];
  const root = await fixture(t);
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));
  const graph = basicGraph({
    nodes: [{
      opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' }, params: { scale: 0.5 },
    }],
    outputs: ['y'],
  });
  await writeJson(path.join(root, 'graph.json'), graph);
  await assert.doesNotReject(validateModelPackages([root]));

  for (const field of retired) {
    graph.nodes[0].params = { nested: [{ [field]: 1 }] };
    await writeJson(path.join(root, 'graph.json'), graph);
    await assert.rejects(validateModelPackages([root]), new RegExp(`params\\.nested\\[0\\]\\.${field}`));
  }
});

test('model package validator reads and validates affine parameter payload values', async (t) => {
  const root = await fixture(t);
  const graph = {
    format: 'volvox-graph/v1',
    inputs: { x: { shape: [2], dtype: 'float32' } },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        q: { scheme: 'per_tensor', scale_tensor: 'q.scale', zero_point_tensor: 'q.zero_point' },
      },
    },
    nodes: [{
      opType: 'QuantizeLinear',
      inputs: { input: 'x', scale: 'q.scale', zero_point: 'q.zero_point' },
      outputs: { out: 'q' }, outputs_shape: { out: [2] }, outputs_dtype: { out: 'int8' },
    }],
    outputs: ['q'],
  };
  await writeJson(path.join(root, 'graph.json'), graph);

  for (const invalidScale of [0, -0.25, Number.NaN, Number.POSITIVE_INFINITY]) {
    const weights = SafetensorsFile.empty();
    weights.addTensor('q.scale', 'F32', [1], Float32Array.of(invalidScale));
    weights.addTensor('q.zero_point', 'I8', [1], Int8Array.of(-128));
    await writeFile(path.join(root, 'model.safetensors'), new Uint8Array(weights.toArrayBuffer()));
    await assert.rejects(validateModelPackages([root]), /must be finite and positive F32 values/);
  }

  const wrongScaleDtype = SafetensorsFile.empty();
  wrongScaleDtype.addTensor('q.scale', 'F16', [1], Uint16Array.of(0x3c00));
  wrongScaleDtype.addTensor('q.zero_point', 'I8', [1], Int8Array.of(0));
  await writeFile(path.join(root, 'model.safetensors'), new Uint8Array(wrongScaleDtype.toArrayBuffer()));
  await assert.rejects(validateModelPackages([root]), /rank-1 safetensors F32 and I8 arrays/);

  const wrongZeroDtype = SafetensorsFile.empty();
  wrongZeroDtype.addTensor('q.scale', 'F32', [1], Float32Array.of(0.25));
  wrongZeroDtype.addTensor('q.zero_point', 'U8', [1], Uint8Array.of(0));
  await writeFile(path.join(root, 'model.safetensors'), new Uint8Array(wrongZeroDtype.toArrayBuffer()));
  await assert.rejects(validateModelPackages([root]), /rank-1 safetensors F32 and I8 arrays/);

  const wrongShape = SafetensorsFile.empty();
  wrongShape.addTensor('q.scale', 'F32', [2], Float32Array.of(0.25, 0.5));
  wrongShape.addTensor('q.zero_point', 'I8', [2], Int8Array.of(-128, 127));
  await writeFile(path.join(root, 'model.safetensors'), new Uint8Array(wrongShape.toArrayBuffer()));
  await assert.rejects(validateModelPackages([root]), /arrays of length 1/);
});
