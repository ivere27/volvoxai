import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { validateModelPackages } from '../tools/validate_model_packages.mjs';
import { SafetensorsFile } from '../ts/core/Safetensors.js';

const FORMAT = 'volvox-graph/v1';

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

function output(tensor, shape, dtype = 'float32') {
  return { tensor, dtype, shape };
}

function identityNode({ id = 'identity', input = 'x', tensor = 'y', shape = [1], dtype = 'float32' } = {}) {
  return {
    id,
    opType: 'Identity',
    inputs: { input },
    outputs: { out: output(tensor, shape, dtype) },
    params: {},
  };
}

function basicGraph(overrides = {}) {
  return {
    format: FORMAT,
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1] } },
    nodes: [],
    outputs: ['x'],
    ...overrides,
  };
}

async function writePackage(root, graph = basicGraph()) {
  await writeJson(path.join(root, 'graph.json'), graph);
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));
}

test('accepts constant and bounded-dynamic primary and named subgraph documents', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), basicGraph({
    dimensions: { B: { min: 1, max: 8, multiple_of: 1 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 4] } },
    nodes: [identityNode({ shape: ['B', 4] })],
    outputs: ['y'],
  }));
  await writeJson(path.join(root, 'router.graph.json'), basicGraph({
    inputs: { route: { dtype: 'int32', shape: [1] } },
    outputs: ['route'],
  }));
  await writeJson(path.join(root, 'config.json'), { model_type: 'application-owned' });
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));

  const files = await validateModelPackages([root]);
  assert.deepEqual(files.map((filename) => path.basename(filename)), [
    'graph.json', 'router.graph.json',
  ]);
});

test('accepts one semantically valid unified-output operator document', async (t) => {
  const root = await fixture(t);
  await writePackage(root, basicGraph({
    inputs: { x: { dtype: 'float32', shape: [2, 4] } },
    nodes: [{
      id: 'softmax',
      opType: 'Softmax',
      inputs: { input: 'x' },
      outputs: { out: output('y', [2, 4]) },
      params: { axis: -1 },
    }],
    outputs: ['y'],
  }));
  await assert.doesNotReject(validateModelPackages([root]));
});

test('rejects the legacy v1 split-output layout with one re-export diagnostic', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), {
    format: FORMAT,
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: 'y' },
      outputs_shape: { out: [1] },
      outputs_dtype: { out: 'float32' },
      params: {},
    }],
    outputs: ['y'],
  });
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));
  await assert.rejects(
    validateModelPackages([root]),
    /uses legacy split output descriptors; re-export with unified outputs/,
  );
});

test('enforces the exact closed root, input, node, and output descriptor fields', async (t) => {
  const cases = [
    ['root extension', (graph) => { graph.source = {}; }, /graph root has unsupported field "source"/],
    ['retired shape system', (graph) => {
      graph.shape_system = 'volvox-bounded-shape/v1';
    }, /graph root has unsupported field "shape_system"/],
    ['input extension', (graph) => { graph.inputs.x.image_normalization = 'zero-one'; }, /input "x" has unsupported field/],
    ['node extension', (graph) => { graph.nodes[0].source_name = 'onnx'; }, /node 0 has unsupported field/],
    ['missing params', (graph) => { delete graph.nodes[0].params; }, /node 0 requires field "params"/],
    ['output extension', (graph) => { graph.nodes[0].outputs.out.quantization = {}; }, /output "out" has unsupported field/],
    ['missing id', (graph) => { delete graph.nodes[0].id; }, /node 0 requires field "id"/],
  ];
  for (const [label, mutate, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      const graph = basicGraph({ nodes: [identityNode()], outputs: ['y'] });
      mutate(graph);
      await writePackage(root, graph);
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

test('validates bounded dimensions, symbol references, and conservative maximum allocation', async (t) => {
  const cases = [
    ['invalid symbol name', (graph) => { graph.dimensions = { 'bad-name': { min: 1, max: 4 } }; }, /name must match/],
    ['unknown symbol', (graph) => { graph.inputs.x.shape = ['Unknown']; }, /declared dimension symbol/],
    ['unsupported constraint field', (graph) => { graph.dimensions = { B: { min: 1, max: 4, sample: 2 } }; }, /unsupported field "sample"/],
    ['inverted bounds', (graph) => { graph.dimensions = { B: { min: 8, max: 1 } }; }, /min <= max/],
    ['empty multiple domain', (graph) => { graph.dimensions = { B: { min: 5, max: 7, multiple_of: 4 } }; }, /no legal multiple_of value/],
    ['maximum byte overflow', (graph) => {
      graph.dimensions = { B: { min: 1, max: Number.MAX_SAFE_INTEGER } };
      graph.inputs.x.shape = ['B'];
    }, /maximum byte length exceeds Number.MAX_SAFE_INTEGER/],
    ['maximum element overflow', (graph) => {
      graph.inputs.x.shape = [Number.MAX_SAFE_INTEGER, 2];
    }, /maximum element count exceeds Number.MAX_SAFE_INTEGER/],
  ];
  for (const [label, mutate, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      const graph = basicGraph();
      mutate(graph);
      await writePackage(root, graph);
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

test('fails closed when canonical Python semantic validation is unavailable', async (t) => {
  const root = await fixture(t);
  await writePackage(root);
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

test('rejects malformed canonical operator contracts through semantic validation', async (t) => {
  const cases = [
    ['Conv2D geometry', basicGraph({
      inputs: {
        x: { dtype: 'float32', shape: [1, 4, 4, 3] },
        w: { dtype: 'float32', shape: [3, 3, 3, 2] },
      },
      nodes: [{
        id: 'conv', opType: 'Conv2D', inputs: { input: 'x', weight: 'w' },
        outputs: { out: output('y', [1, 2, 2, 3]) },
        params: { weight_layout: 'HWIO' },
      }],
      outputs: ['y'],
    }), /rank-4 convolution geometry|canonical (?:concrete shape|whole-domain) inference/],
    ['Softmax axis', basicGraph({
      inputs: { x: { dtype: 'float32', shape: [2, 4] } },
      nodes: [{
        id: 'softmax', opType: 'Softmax', inputs: { input: 'x' },
        outputs: { out: output('y', [2, 4]) }, params: { axis: 0 },
      }],
      outputs: ['y'],
    }), /last axis/],
  ];
  for (const [label, graph, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      await writePackage(root, graph);
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

test('qualification rejects a graph missing from any portable backend member', async (t) => {
  const root = await fixture(t);
  await writePackage(root, basicGraph({
    dimensions: { B: { min: 1, max: 8, multiple_of: 1 } },
    inputs: {
      x: { dtype: 'float32', shape: ['B', 4] },
      indices: { dtype: 'int32', shape: ['B', 4] },
    },
    nodes: [{
      id: 'gather',
      opType: 'GatherElements',
      inputs: { input: 'x', indices: 'indices' },
      outputs: { out: output('y', ['B', 4]) },
      params: { axis: 1 },
    }],
    outputs: ['y'],
  }));
  await assert.rejects(
    validateModelPackages([root]),
    /operator 'GatherElements' is not admitted by 'native-cpu'/,
  );
});

test('requires canonical filenames, one primary graph, and safetensors storage', async (t) => {
  await t.test('noncanonical graph filename', async (subtest) => {
    const root = await fixture(subtest);
    await writeJson(path.join(root, 'model.json'), basicGraph());
    await writeEmptySafetensors(path.join(root, 'model.safetensors'));
    await assert.rejects(validateModelPackages([root]), /must use graph\.json/);
  });
  await t.test('named subgraph without primary', async (subtest) => {
    const root = await fixture(subtest);
    await writeJson(path.join(root, 'router.graph.json'), basicGraph());
    await writeEmptySafetensors(path.join(root, 'model.safetensors'));
    await assert.rejects(validateModelPackages([root]), /no primary graph\.json file found/);
  });
  await t.test('missing safetensors', async (subtest) => {
    const root = await fixture(subtest);
    await writeJson(path.join(root, 'graph.json'), basicGraph());
    await assert.rejects(validateModelPackages([root]), /no \*\.safetensors weight file found/);
  });
});

test('rejects duplicate JSON keys before interpreting the graph', async (t) => {
  const root = await fixture(t);
  await writeFile(
    path.join(root, 'graph.json'),
    '{"format":"volvox-graph/v2","format":"volvox-graph/v1"}',
  );
  await writeEmptySafetensors(path.join(root, 'model.safetensors'));
  await assert.rejects(validateModelPackages([root]), /duplicate object key "format"/);
});

test('validates every static safetensors shape, allocation, span, and payload byte', async (t) => {
  const tensor = (dtype, shape, data_offsets) => ({ dtype, shape, data_offsets });
  const cases = [
    ['truncated file', new Uint8Array(), /header is truncated/],
    ['invalid header length', rawSafetensors(''), /header length is invalid/],
    ['invalid header JSON', rawSafetensors('{'), /invalid safetensors header/],
    ['duplicate header field', rawSafetensors(
      '{"w":{"dtype":"F32","dtype":"I8","shape":[1],"data_offsets":[0,1]}}',
      Uint8Array.of(0),
    ), /duplicate object key "dtype"/],
    ['empty tensor name', rawSafetensors({
      '': tensor('I8', [1], [0, 1]),
    }, Uint8Array.of(0)), /invalid header record/],
    ['extra tensor field', rawSafetensors({
      w: { ...tensor('I8', [1], [0, 1]), extra: true },
    }, Uint8Array.of(0)), /invalid header record/],
    ['invalid metadata', rawSafetensors({ __metadata__: null }), /metadata must contain only string values/],
    ['zero extent', rawSafetensors({ w: tensor('I8', [0], [0, 0]) }), /invalid shape/],
    ['unsafe allocation', rawSafetensors({
      w: tensor('F32', [Number.MAX_SAFE_INTEGER], [0, 0]),
    }), /safe static allocation limits/],
    ['unsupported dtype', rawSafetensors({
      w: tensor('F64', [1], [0, 8]),
    }, new Uint8Array(8)), /unsupported dtype/],
    ['wrong byte span', rawSafetensors({
      w: tensor('F32', [1], [0, 3]),
    }, new Uint8Array(3)), /data span is invalid/],
    ['noninteger offset', rawSafetensors({
      w: tensor('I8', [1], [0, 1.5]),
    }, Uint8Array.of(0)), /invalid data offsets/],
    ['out-of-range offset', rawSafetensors({
      w: tensor('I8', [1], [0, 2]),
    }, Uint8Array.of(0)), /data span is invalid/],
    ['overlap', rawSafetensors({
      a: tensor('I8', [2], [0, 2]), b: tensor('I8', [2], [1, 3]),
    }, new Uint8Array(3)), /gap, overlap, or aliased span/],
    ['alias', rawSafetensors({
      a: tensor('I8', [1], [0, 1]), b: tensor('I8', [1], [0, 1]),
    }, Uint8Array.of(0)), /gap, overlap, or aliased span/],
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

test('accepts complete contiguous supported static safetensors payloads', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), basicGraph({
    inputs: { x: { dtype: 'float32', shape: [1, 1, 1, 1] } },
    nodes: [{
      id: 'conv', opType: 'Conv2D', inputs: { input: 'x', weight: 'half_weight' },
      outputs: { out: output('y', [1, 1, 1, 1]) },
      params: { weight_layout: 'HWIO' },
    }],
    outputs: ['y'],
  }));
  await writeFile(path.join(root, 'model.safetensors'), rawSafetensors({
    half_weight: { dtype: 'F16', shape: [1, 1, 1, 1], data_offsets: [0, 2] },
    integer_weight: { dtype: 'I32', shape: [1], data_offsets: [2, 6] },
  }, new Uint8Array(6)));
  await assert.doesNotReject(validateModelPackages([root]));
});

test('resolves unified output tensor names topologically', async (t) => {
  const cases = [
    ['missing input', basicGraph({
      nodes: [identityNode({ input: 'missing' })], outputs: ['y'],
    }), /input "missing" is unresolved or not topologically available/],
    ['forward input', basicGraph({
      nodes: [
        identityNode({ id: 'first', input: 'later', tensor: 'first' }),
        identityNode({ id: 'later', tensor: 'later' }),
      ],
      outputs: ['first'],
    }), /input "later" is unresolved or not topologically available/],
    ['duplicate tensor', basicGraph({
      nodes: [identityNode({ tensor: 'x' })], outputs: ['x'],
    }), /output "x" collides with an existing tensor/],
    ['missing public output', basicGraph({ outputs: ['missing'] }), /public output "missing" is unresolved/],
  ];
  for (const [label, graph, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      await writePackage(root, graph);
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

async function writeBankPackage(root, graph, shape = [2, 1], includeWeight = true) {
  await writeJson(path.join(root, 'graph.json'), graph);
  const weights = SafetensorsFile.empty();
  if (includeWeight) {
    weights.addTensor(
      'bank.weight',
      'F32',
      shape,
      new Float32Array(shape.reduce((count, axis) => count * axis, 1)),
    );
  }
  await writeFile(
    path.join(root, 'model.safetensors'),
    new Uint8Array(weights.toArrayBuffer()),
  );
}

test('accepts and validates the current optional weight-bank root', async (t) => {
  await t.test('valid bank', async (subtest) => {
    const root = await fixture(subtest);
    await writeBankPackage(root, basicGraph({
      dimensions: { F: { min: 1, max: 2 } },
      banks: { 'bank.weight': 'F' },
    }));
    await assert.doesNotReject(validateModelPackages([root]));
  });

  const cases = [
    ['non-object table', [], { F: { min: 1, max: 2 } }, [2, 1], true, /banks must be an object/],
    ['empty dimension', { 'bank.weight': '' }, { F: { min: 1, max: 2 } }, [2, 1], true,
      /must name a non-empty declared dimension/],
    ['unknown dimension', { 'bank.weight': 'Missing' }, { F: { min: 1, max: 2 } }, [2, 1], true,
      /references undeclared dimension "Missing"/],
    ['missing weight', { 'bank.weight': 'F' }, { F: { min: 1, max: 2 } }, [2, 1], false,
      /does not name a supplied fixed weight/],
    ['rank-one weight', { 'bank.weight': 'F' }, { F: { min: 1, max: 2 } }, [2], true,
      /needs a slot axis and at least one payload axis/],
    ['slots outside bounds', { 'bank.weight': 'F' }, { F: { min: 1, max: 2 } }, [3, 1], true,
      /supplies 3 slots outside dimension "F" bounds \[1, 2\]/],
    ['slots violate multiple', { 'bank.weight': 'F' },
      { F: { min: 1, max: 4, multiple_of: 2 } }, [3, 1], true,
      /supplies 3 slots, which is not a multiple of 2/],
  ];
  for (const [label, banks, dimensions, shape, includeWeight, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      await writeBankPackage(root, basicGraph({ banks, dimensions }), shape, includeWeight);
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

function quantizedGraph(scale = 'q.scale') {
  return basicGraph({
    inputs: { x: { dtype: 'float32', shape: [2] } },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        q: {
          scheme: 'per_tensor',
          scale_tensor: 'q.scale',
          zero_point_tensor: 'q.zero_point',
        },
      },
    },
    nodes: [{
      id: 'quantize',
      opType: 'QuantizeLinear',
      inputs: { input: 'x', scale, zero_point: 'q.zero_point' },
      outputs: { out: output('q', [2], 'int8') },
      params: {},
    }],
    outputs: ['q'],
  });
}

async function writeQuantizationWeights(root, {
  scaleDtype = 'F32',
  scaleShape = [1],
  scaleValues = Float32Array.of(0.25),
  zeroPointDtype = 'I8',
  zeroPointShape = [1],
  zeroPointValues = Int8Array.of(0),
} = {}) {
  const weights = SafetensorsFile.empty();
  weights.addTensor('q.scale', scaleDtype, scaleShape, scaleValues);
  weights.addTensor('q.zero_point', zeroPointDtype, zeroPointShape, zeroPointValues);
  await writeFile(path.join(root, 'model.safetensors'), new Uint8Array(weights.toArrayBuffer()));
}

test('enforces central affine references against unified output descriptors', async (t) => {
  const root = await fixture(t);
  await writeJson(path.join(root, 'graph.json'), quantizedGraph());
  await writeQuantizationWeights(root);
  await assert.doesNotReject(validateModelPackages([root]));

  await writeJson(path.join(root, 'graph.json'), quantizedGraph('wrong.scale'));
  await assert.rejects(
    validateModelPackages([root]),
    /operands must match its central output references/,
  );

  await writeJson(path.join(root, 'graph.json'), quantizedGraph());
  await writeQuantizationWeights(root, { scaleValues: Float32Array.of(0) });
  await assert.rejects(
    validateModelPackages([root]),
    /must be finite and positive F32 values/,
  );
});

test('rejects wrong affine parameter dtype, shape, and value range', async (t) => {
  const cases = [
    ['scale dtype', {
      scaleDtype: 'I32', scaleValues: Int32Array.of(1),
    }, /must be rank-1 safetensors F32 and I8 arrays of length 1/],
    ['zero-point dtype', {
      zeroPointDtype: 'U8', zeroPointValues: Uint8Array.of(0),
    }, /must be rank-1 safetensors F32 and I8 arrays of length 1/],
    ['scale shape', {
      scaleShape: [2], scaleValues: Float32Array.of(0.25, 0.5),
    }, /must be rank-1 safetensors F32 and I8 arrays of length 1/],
    ['negative scale', {
      scaleValues: Float32Array.of(-0.25),
    }, /must be finite and positive F32 values/],
    ['non-finite scale', {
      scaleValues: Float32Array.of(Number.NaN),
    }, /must be finite and positive F32 values/],
  ];
  for (const [label, options, expected] of cases) {
    await t.test(label, async (subtest) => {
      const root = await fixture(subtest);
      await writeJson(path.join(root, 'graph.json'), quantizedGraph());
      await writeQuantizationWeights(root, options);
      await assert.rejects(validateModelPackages([root]), expected);
    });
  }
});

test('reserves affine parameter tensors from node and public outputs', async (t) => {
  await t.test('public output', async (subtest) => {
    const root = await fixture(subtest);
    const graph = quantizedGraph();
    graph.outputs = ['q.scale'];
    await writeJson(path.join(root, 'graph.json'), graph);
    await writeQuantizationWeights(root);
    await assert.rejects(
      validateModelPackages([root]),
      /quantization parameter "q\.scale" cannot be a public output/,
    );
  });

  await t.test('node output', async (subtest) => {
    const root = await fixture(subtest);
    const graph = quantizedGraph();
    graph.nodes.push(identityNode({
      id: 'parameter_collision', input: 'q', tensor: 'q.scale', shape: [1], dtype: 'float32',
    }));
    await writeJson(path.join(root, 'graph.json'), graph);
    await writeQuantizationWeights(root);
    await assert.rejects(
      validateModelPackages([root]),
      /cannot produce reserved quantization parameter "q\.scale"/,
    );
  });
});

test('scopes duplicate safetensors names to each nearest split graph directory', async (t) => {
  const root = await fixture(t);
  for (const role of ['encoder', 'decoder']) {
    const directory = path.join(root, role);
    await mkdir(directory, { recursive: true });
    await writeJson(path.join(directory, 'graph.json'), quantizedGraph());
    await writeQuantizationWeights(directory);
  }
  await assert.doesNotReject(validateModelPackages([root]));
});

test('recursively rejects retired inline affine params', async (t) => {
  const root = await fixture(t);
  const graph = basicGraph({
    nodes: [identityNode()],
    outputs: ['y'],
  });
  await writePackage(root, graph);
  await assert.doesNotReject(validateModelPackages([root]));

  for (const field of ['quantization', 'zero_point', 'input_scale', 'weight_scale']) {
    graph.nodes[0].params = { nested: [{ [field]: 1 }] };
    await writeJson(path.join(root, 'graph.json'), graph);
    await assert.rejects(
      validateModelPackages([root]),
      new RegExp(`params\\.nested\\[0\\]\\.${field}`),
    );
  }
});
