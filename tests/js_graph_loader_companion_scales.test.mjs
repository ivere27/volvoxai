import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/core/Graph.js';
import { GraphLoader } from '../ts/core/GraphLoader.js';
import { SafetensorsFile } from '../ts/core/Safetensors.js';

const STORAGE_FORMAT = 'volvoxai-f32-companion-scales-v1';
const EXACT_SCALES = Float32Array.of(0.1, 1 / 3);

function defaultManifest() {
  return {
    weight: { scheme: 'per_axis', axis: 0 },
    scalar_weight: { scheme: 'per_tensor' },
  };
}

function companionMetadata(manifest = defaultManifest(), options = {}) {
  const metadata = {};
  if (options.includeStorage !== false) {
    metadata.weights_quantization_storage = Object.prototype.hasOwnProperty.call(options, 'storage')
      ? options.storage
      : STORAGE_FORMAT;
  }
  if (options.includeQuantization !== false) {
    metadata.weights_quantization = Object.prototype.hasOwnProperty.call(options, 'encoded')
      ? options.encoded
      : JSON.stringify(manifest);
  }
  return metadata;
}

function tensorFile(entries, metadata = {}) {
  const file = SafetensorsFile.empty({ metadata });
  for (const { name, dtype, shape, values } of entries) {
    file.addTensor(name, dtype, shape, values);
  }
  return file.toArrayBuffer();
}

function weightFile({
  metadata = companionMetadata(),
  includeWeight = true,
  weightDtype = 'I8',
  weightShape = [2, 2],
  weightValues = null,
  includeWeightCompanion = true,
  weightCompanionDtype = 'F32',
  weightCompanionShape = [2],
  weightCompanionValues = EXACT_SCALES,
  includeScalarWeight = true,
  includeScalarCompanion = true,
  scalarCompanionShape = [1],
  scalarCompanionValues = Float32Array.of(0.125),
  includeBias = true,
} = {}) {
  const entries = [];
  if (includeWeight) {
    const values = weightValues ?? (weightDtype === 'F32'
      ? Float32Array.of(1, 2, 3, 4)
      : Int8Array.of(1, 2, 3, 4));
    entries.push({ name: 'weight', dtype: weightDtype, shape: weightShape, values });
  }
  if (includeScalarWeight) {
    entries.push({ name: 'scalar_weight', dtype: 'I8', shape: [1], values: Int8Array.of(5) });
  }
  if (includeBias) {
    entries.push({ name: 'bias', dtype: 'I32', shape: [2], values: Int32Array.of(0, 0) });
  }
  if (includeWeightCompanion) {
    entries.push({
      name: 'weight_scale',
      dtype: weightCompanionDtype,
      shape: weightCompanionShape,
      values: weightCompanionValues,
    });
  }
  if (includeScalarCompanion) {
    entries.push({
      name: 'scalar_weight_scale',
      dtype: 'F32',
      shape: scalarCompanionShape,
      values: scalarCompanionValues,
    });
  }
  return tensorFile(entries, metadata);
}

function companionConfig() {
  return {
    inputs: {
      input: {
        shape: [1, 2],
        dtype: 'int8',
        quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
      },
    },
    weights_quantization_storage: {
      format: STORAGE_FORMAT,
    },
    nodes: [{
      op: 'QLinear',
      inputs: { input: 'input', weight: 'weight', bias: 'bias' },
      outputs: { out: 'output' },
      outputs_shape: { out: [1, 2] },
      outputs_dtype: { out: 'int8' },
      outputs_quantization: {
        out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      },
    }],
    outputs: ['output'],
  };
}

async function loadConfig(config, buffers = [weightFile()]) {
  const sources = buffers.map((_, index) => `weights-${index}.safetensors`);
  const fetch = async (url) => {
    if (url === 'config.json') return { ok: true, json: async () => config };
    const index = sources.indexOf(url);
    if (index >= 0) return { ok: true, arrayBuffer: async () => buffers[index] };
    return { ok: false, statusText: 'not found' };
  };
  return GraphLoader.load(new Graph(), sources, { configUrl: 'config.json', fetch });
}

async function loadRawConfig(configText, buffers = [weightFile()]) {
  const sources = buffers.map((_, index) => `weights-${index}.safetensors`);
  const fetch = async (url) => {
    if (url === 'config.json') return { ok: true, text: async () => configText };
    const index = sources.indexOf(url);
    if (index >= 0) return { ok: true, arrayBuffer: async () => buffers[index] };
    return { ok: false, statusText: 'not found' };
  };
  return GraphLoader.load(new Graph(), sources, { configUrl: 'config.json', fetch });
}

test('GraphLoader hydrates strict metadata-owned F32 companion scales', async () => {
  const graph = await loadConfig(companionConfig());
  assert.deepEqual(graph.getTensor('weight').quantization, {
    scheme: 'per_axis', axis: 0, scales: [...EXACT_SCALES], zero_points: [0, 0],
  });
  assert.deepEqual(
    new Uint32Array(Float32Array.from(graph.getTensor('weight').quantization.scales).buffer),
    new Uint32Array(EXACT_SCALES.buffer),
    'hydration preserves the exact F32 bit patterns',
  );
  assert.deepEqual(graph.getTensor('scalar_weight').quantization, {
    scheme: 'per_tensor', scale: 0.125, zero_point: 0,
  });
  assert.equal(Object.isFrozen(graph.getTensor('weight').quantization), true);
  assert.equal(Object.isFrozen(graph.getTensor('weight').quantization.scales), true);
  assert.equal(graph.getTensor('weight_scale'), undefined);
  assert.equal(graph.getTensor('scalar_weight_scale'), undefined);
  assert.ok(graph.weightFiles[0].getTensor('weight_scale'), 'the safetensors file retains the companion');
});

test('GraphLoader merges unique local companion manifests across files', async () => {
  const first = weightFile({
    metadata: companionMetadata({ weight: { scheme: 'per_axis', axis: 0 } }),
    includeScalarWeight: false,
    includeScalarCompanion: false,
  });
  const second = weightFile({
    metadata: companionMetadata({ scalar_weight: { scheme: 'per_tensor' } }),
    includeWeight: false,
    includeWeightCompanion: false,
    includeBias: false,
  });
  const graph = await loadConfig(companionConfig(), [first, second]);
  assert.deepEqual(graph.getTensor('weight').quantization.scales, [...EXACT_SCALES]);
  assert.equal(graph.getTensor('scalar_weight').quantization.scale, 0.125);
  assert.equal(graph.getTensor('weight_scale'), undefined);
  assert.equal(graph.getTensor('scalar_weight_scale'), undefined);
});

test('GraphLoader keeps marker-free inline quantization separate from companion metadata', async () => {
  const config = companionConfig();
  delete config.weights_quantization_storage;
  config.weights_quantization = {
    weight: {
      scheme: 'per_axis', axis: 0, scales: [0.25, 0.5], zero_points: [0, 0],
    },
    scalar_weight: {
      scheme: 'per_tensor', scale: 0.125, zero_point: 0,
    },
  };
  const graph = await loadConfig(config);
  assert.deepEqual(graph.getTensor('weight').quantization.scales, [0.25, 0.5]);
  assert.ok(graph.getTensor('weight_scale'), 'metadata companions are not reserved without the config marker');

  config.weights_quantization.weight.scales_offset = 0;
  config.weights_quantization.weight.scales_count = 2;
  await assert.rejects(loadConfig(config), /uses unsupported binary scale offset fields/);
});

test('GraphLoader requires companion-mode config to omit weights_quantization', async () => {
  for (const value of [{}, null]) {
    const config = companionConfig();
    config.weights_quantization = value;
    await assert.rejects(loadConfig(config), /config\.json to omit weights_quantization/);
  }
});

test('GraphLoader rejects duplicate keys in raw config JSON', async () => {
  const encoded = JSON.stringify(companionConfig());
  const duplicateMarker =
    `{"weights_quantization_storage":{"format":"${STORAGE_FORMAT}"},${encoded.slice(1)}`;
  await assert.rejects(
    loadRawConfig(duplicateMarker),
    /Config 'config\.json' contains duplicate object key "weights_quantization_storage"/,
  );

  const duplicateNestedShape = encoded.replace('"shape":[1,2]', '"shape":[1,2],"shape":[1,2]');
  await assert.rejects(
    loadRawConfig(duplicateNestedShape),
    /Config 'config\.json' contains duplicate object key "shape"/,
  );
});

test('GraphLoader validates paired exact companion metadata markers', async () => {
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({ metadata: {} })]),
    /metadata to declare at least one weight/,
  );
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({ metadata: companionMetadata({}) })]),
    /metadata to declare at least one weight/,
  );
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({
      metadata: companionMetadata(defaultManifest(), { includeQuantization: false }),
    })]),
    /must contain both weights_quantization_storage and weights_quantization/,
  );
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({
      metadata: companionMetadata(defaultManifest(), { includeStorage: false }),
    })]),
    /must contain both weights_quantization_storage and weights_quantization/,
  );
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({
      metadata: companionMetadata(defaultManifest(), { storage: 'unknown' }),
    })]),
    /weights_quantization_storage must equal/,
  );
});

test('GraphLoader validates compact JSON companion manifests', async () => {
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({
      metadata: companionMetadata(defaultManifest(), { encoded: '{' }),
    })]),
    /weights_quantization metadata is not valid JSON/,
  );
  for (const encoded of ['null', '[]', '1']) {
    await assert.rejects(
      loadConfig(companionConfig(), [weightFile({
        metadata: companionMetadata(defaultManifest(), { encoded }),
      })]),
      /weights_quantization metadata must decode to an object/,
    );
  }
  for (const encoded of [
    '{"weight":{"scheme":"per_axis","axis":1},"\\u0077eight":{"scheme":"per_axis","axis":0},"scalar_weight":{"scheme":"per_tensor"}}',
    '{"weight":{"scheme":"per_axis","sch\\u0065me":"per_axis","axis":0},"scalar_weight":{"scheme":"per_tensor"}}',
  ]) {
    await assert.rejects(
      loadConfig(companionConfig(), [weightFile({
        metadata: companionMetadata(defaultManifest(), { encoded }),
      })]),
      /weights_quantization metadata contains duplicate object key "(?:weight|scheme)"/,
    );
  }
});

test('GraphLoader rejects inline, referenced, offset, and zero fields in companion descriptors', async () => {
  for (const [field, value] of [
    ['scales', [0.25, 0.5]],
    ['scale', 0.25],
    ['scales_tensor', 'other'],
    ['scales_offset', 0],
    ['scales_count', 2],
    ['zero_point', 0],
    ['zero_points', [0, 0]],
  ]) {
    const manifest = defaultManifest();
    manifest.weight[field] = value;
    await assert.rejects(
      loadConfig(companionConfig(), [weightFile({ metadata: companionMetadata(manifest) })]),
      new RegExp(`unsupported field '${field}'`),
    );
  }

  const manifest = defaultManifest();
  manifest.scalar_weight.scale = 0.125;
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({ metadata: companionMetadata(manifest) })]),
    /unsupported field 'scale'/,
  );
});

test('GraphLoader validates companion base dtype, shape, axis, and finite positive values', async () => {
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({ weightDtype: 'F32' })]),
    /Declared weight 'weight' must have safetensors dtype I8/,
  );
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({ weightCompanionShape: [1, 2] })]),
    /weight_scale.*must have shape \[2\]/,
  );
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({
      metadata: companionMetadata({ weight: { scheme: 'per_axis', axis: 0 } }),
      weightShape: [0, 2],
      weightValues: new Int8Array(0),
      weightCompanionShape: [0],
      weightCompanionValues: new Float32Array(0),
      includeScalarWeight: false,
      includeScalarCompanion: false,
      includeBias: false,
    }), tensorFile([{
      name: 'bias', dtype: 'I32', shape: [2], values: Int32Array.of(0, 0),
    }])]),
    /axis-0 dimension must be positive/,
  );

  const wrongAxis = defaultManifest();
  wrongAxis.weight.axis = 1;
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({ metadata: companionMetadata(wrongAxis) })]),
    /axis must be 0/,
  );

  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({ weightCompanionValues: Float32Array.of(0, 0.5) })]),
    /weight_scale.*value 0 must be finite and positive/,
  );
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({ scalarCompanionShape: [] })]),
    /scalar_weight_scale.*must have shape \[1\]/,
  );
});

test('GraphLoader validates companion dtype and same-file ownership', async () => {
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile({
      weightCompanionDtype: 'I32',
      weightCompanionValues: Int32Array.of(1, 1),
    })]),
    /weight_scale.*must have safetensors dtype F32/,
  );

  const declaredWithoutLocalCompanion = weightFile({ includeWeightCompanion: false });
  const remoteCompanion = tensorFile([{
    name: 'weight_scale', dtype: 'F32', shape: [2], values: EXACT_SCALES,
  }]);
  await assert.rejects(
    loadConfig(companionConfig(), [declaredWithoutLocalCompanion, remoteCompanion]),
    /must store declared weight 'weight' and its companion 'weight_scale' in the same safetensors file/,
  );
});

test('GraphLoader reserves claimed companion names from every graph topology path', async () => {
  const inputCollision = companionConfig();
  inputCollision.inputs.weight_scale = { shape: [2], dtype: 'float32' };
  await assert.rejects(
    loadConfig(inputCollision),
    /Companion scale 'weight_scale' is reserved storage/,
  );

  const outputCollision = companionConfig();
  outputCollision.nodes[0].outputs.out = 'weight_scale';
  outputCollision.outputs = ['weight_scale'];
  await assert.rejects(
    loadConfig(outputCollision),
    /Companion scale 'weight_scale' is reserved storage/,
  );

  const modelType = 'companion-scale-reservation-test';
  const hadBuilder = Object.prototype.hasOwnProperty.call(GraphLoader.ModelBuilders, modelType);
  const previousBuilder = GraphLoader.ModelBuilders[modelType];
  GraphLoader.ModelBuilders[modelType] = (graph) => {
    graph.addTensor('weight_scale', [1], 'float32');
  };
  const builderCollision = companionConfig();
  delete builderCollision.inputs;
  delete builderCollision.nodes;
  delete builderCollision.outputs;
  builderCollision.model_type = modelType;
  try {
    await assert.rejects(
      loadConfig(builderCollision),
      /Companion scale 'weight_scale' is reserved storage/,
    );
  } finally {
    if (hadBuilder) GraphLoader.ModelBuilders[modelType] = previousBuilder;
    else delete GraphLoader.ModelBuilders[modelType];
  }
});

test('GraphLoader enforces global uniqueness for manifests, bases, and companions', async () => {
  const duplicateManifest = weightFile();
  await assert.rejects(
    loadConfig(companionConfig(), [duplicateManifest, duplicateManifest]),
    /declares weight 'weight' in both/,
  );

  const duplicateBase = tensorFile([{
    name: 'weight', dtype: 'I8', shape: [2, 2], values: Int8Array.of(1, 2, 3, 4),
  }]);
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile(), duplicateBase]),
    /Tensor 'weight' occurs in both/,
  );

  const duplicateCompanion = tensorFile([{
    name: 'weight_scale', dtype: 'F32', shape: [2], values: EXACT_SCALES,
  }]);
  await assert.rejects(
    loadConfig(companionConfig(), [weightFile(), duplicateCompanion]),
    /Tensor 'weight_scale' occurs in both/,
  );
});

test('GraphLoader rejects malformed companion storage config markers', async () => {
  const unsupported = companionConfig();
  unsupported.weights_quantization_storage.format = 'unknown';
  await assert.rejects(loadConfig(unsupported), /Unsupported weights_quantization_storage format 'unknown'/);

  const extra = companionConfig();
  extra.weights_quantization_storage.version = 1;
  await assert.rejects(loadConfig(extra), /unsupported field 'version'/);
});
