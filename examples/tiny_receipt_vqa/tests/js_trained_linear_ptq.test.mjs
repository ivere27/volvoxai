import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

import { SafetensorsFile } from '../../../ts/full.js';
import {
  LINEAR_ISLAND_PACKAGE_FORMAT,
  LINEAR_SAMPLE_FORMAT,
  TRAINED_SOURCE_FORMAT,
  inspectTrainedLinearMappings,
  loadTrainedTinyReceiptPackage,
  materializeTrainedLinearIsland,
} from '../tools/materialize_trained_linear_ptq.mjs';

async function fixture({ layout = 'IN_OUT', metadata = {}, intermediate = false } = {}) {
  const root = await mkdtemp(join(tmpdir(), 'volvox-tiny-receipt-trained-ptq-'));
  const source = join(root, 'full_model');
  const output = join(root, 'linear_island');
  await import('node:fs/promises').then(({ mkdir }) => mkdir(source));
  const config = {
    format: 'volvox.api.v1',
    inputs: {
      'vqa.fixture.input': { shape: [1, 2], dtype: 'float32' },
    },
    outputs: { logits: 'vqa.fixture.output' },
    nodes: [...(intermediate ? [{
      op: 'ReLU',
      inputs: { input: 'vqa.fixture.input' },
      outputs: { out: 'vqa.fixture.hidden' },
      outputs_shape: { out: [1, 2] },
      outputs_dtype: { out: 'float32' },
      params: {},
    }] : []), {
      opType: 'Linear',
      inputs: {
        input: intermediate ? 'vqa.fixture.hidden' : 'vqa.fixture.input',
        weight: 'vqa.fixture.weight',
        bias: 'vqa.fixture.bias',
      },
      outputs: { out: 'vqa.fixture.output' },
      outputs_shape: { out: [1, 3] },
      outputs_dtype: { out: 'float32' },
      params: { weight_layout: layout },
    }],
  };
  const weights = SafetensorsFile.empty({
    metadata: {
      format: 'tiny_receipt_vqa.volvox.v2',
      'volvox.model_format': 'volvox.api.v1',
      'volvox.model_origin': 'api',
      ...metadata,
    },
  });
  weights.addTensor(
    'vqa.fixture.weight',
    'F32',
    [2, 3],
    new Uint8Array(Float32Array.of(1, 2, 3, 4, 5, 6).buffer),
  );
  weights.addTensor(
    'vqa.fixture.bias',
    'F32',
    [3],
    new Uint8Array(Float32Array.of(0.25, -0.5, 0.75).buffer),
  );
  await Promise.all([
    writeFile(join(source, 'config.json'), `${JSON.stringify(config, null, 2)}\n`),
    writeFile(join(source, 'model.safetensors'), new Uint8Array(weights.toArrayBuffer())),
  ]);
  return { root, source, output, config };
}

const sampleDocument = {
  format: LINEAR_SAMPLE_FORMAT,
  samples: [
    { 'vqa.fixture.input': [1, 2] },
    { 'vqa.fixture.input': [-2, 3] },
  ],
};

test('trained TinyReceipt mapping identifies the exact IN_OUT to OUT_IN contract', async (t) => {
  const files = await fixture();
  t.after(() => rm(files.root, { recursive: true, force: true }));
  const source = await loadTrainedTinyReceiptPackage(files.source);
  assert.equal(source.format, TRAINED_SOURCE_FORMAT);
  const mappings = await inspectTrainedLinearMappings(source);
  assert.equal(mappings.length, 1);
  assert.deepEqual(mappings[0], {
    source: {
      input: 'vqa.fixture.input',
      output: 'vqa.fixture.output',
      weight: 'vqa.fixture.weight',
      bias: 'vqa.fixture.bias',
      weight_shape: [2, 3],
      weight_layout: 'IN_OUT',
    },
    target: {
      input: 'island.input',
      output: 'island.output',
      weight: 'w8a8.vqa.fixture.weight',
      weight_scale: 'w8a8.vqa.fixture.weight_scale',
      bias: 'w8a8.vqa.fixture.bias_i32',
      weight_shape: [3, 2],
      weight_layout: 'OUT_IN',
    },
  });
});

test('VolvoxAI PTQ calibrates and materializes a runnable trained Linear island', async (t) => {
  const files = await fixture();
  t.after(() => rm(files.root, { recursive: true, force: true }));
  const result = await materializeTrainedLinearIsland({
    sourceDirectory: files.source,
    weightName: 'vqa.fixture.weight',
    sampleDocument,
    outputDirectory: files.output,
  });

  assert.equal(result.manifest.format, LINEAR_ISLAND_PACKAGE_FORMAT);
  assert.equal(result.manifest.scope, 'standalone_extracted_linear_operator');
  assert.equal(result.manifest.runnable_as_full_tiny_receipt_vqa, false);
  assert.equal(result.calibration.representative, true);
  assert.equal(result.manifest.mapping.weight_layout.transform, 'transpose_2d');
  assert.equal(result.manifest.mapping.weight_layout.quantized_axis, 0);
  assert.equal(result.config.nodes[0].opType, 'QLinear');
  assert.equal(result.config.nodes[0].inputs.weight, 'w8a8.vqa.fixture.weight');
  assert.equal(result.config.nodes[0].inputs.bias, 'w8a8.vqa.fixture.bias_i32');
  assert.ok(Number.isFinite(result.manifest.validation.max_abs_error_against_f32_island));

  const weights = SafetensorsFile.fromArrayBuffer(result.weightsBuffer);
  assert.deepEqual(weights.listTensorNames(), [
    'w8a8.vqa.fixture.weight',
    'w8a8.vqa.fixture.weight_scale',
    'w8a8.vqa.fixture.bias_i32',
  ]);
  assert.deepEqual(weights.getTensor('w8a8.vqa.fixture.weight').shape, [3, 2]);
  assert.deepEqual(
    [...weights.toRuntimeTypedArray(weights.getTensor('w8a8.vqa.fixture.weight'))],
    [32, 127, 51, 127, 64, 127],
  );
  assert.equal(weights.getTensor('w8a8.vqa.fixture.bias_i32').dtype, 'I32');

  const diskManifest = JSON.parse(await readFile(
    join(files.output, 'island_manifest.json'), 'utf8',
  ));
  assert.equal(diskManifest.format, LINEAR_ISLAND_PACKAGE_FORMAT);
  assert.equal(diskManifest.source.format, TRAINED_SOURCE_FORMAT);
});

test('structural smoke is explicitly non-representative', async (t) => {
  const files = await fixture();
  t.after(() => rm(files.root, { recursive: true, force: true }));
  const result = await materializeTrainedLinearIsland({
    sourceDirectory: files.source,
    weightName: 'vqa.fixture.weight',
    structuralSmoke: true,
  });
  assert.equal(result.calibration.representative, false);
  assert.match(result.manifest.limitations.at(-1), /not representative/);
});

test('island validation snapshots a real intermediate Linear input', async (t) => {
  const files = await fixture({ intermediate: true });
  t.after(() => rm(files.root, { recursive: true, force: true }));
  const result = await materializeTrainedLinearIsland({
    sourceDirectory: files.source,
    weightName: 'vqa.fixture.weight',
    sampleDocument,
  });
  assert.equal(result.mapping.source.input, 'vqa.fixture.hidden');
  assert.ok(Number.isFinite(result.manifest.validation.max_abs_error_against_f32_island));
});

test('trained mapping rejects ambiguous formats and unsupported layout guesses', async (t) => {
  const badMetadata = await fixture({ metadata: { format: 'published-pytorch-int8-v1' } });
  t.after(() => rm(badMetadata.root, { recursive: true, force: true }));
  await assert.rejects(
    loadTrainedTinyReceiptPackage(badMetadata.source),
    /TinyReceipt v2 API-model metadata/,
  );

  const badLayout = await fixture({ layout: 'OUT_IN' });
  t.after(() => rm(badLayout.root, { recursive: true, force: true }));
  const source = await loadTrainedTinyReceiptPackage(badLayout.source);
  assert.deepEqual(await inspectTrainedLinearMappings(source), []);
  await assert.rejects(
    materializeTrainedLinearIsland({
      sourcePackage: source,
      weightName: 'vqa.fixture.weight',
      sampleDocument,
    }),
    /requires explicit IN_OUT storage/,
  );
});
