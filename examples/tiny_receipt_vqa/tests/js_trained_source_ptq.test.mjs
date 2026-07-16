import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { SafetensorsFile, packPTQWeight } from '../../../ts/full.js';
import {
  NAMED_SOURCE_LAYOUT,
  NORMALIZED_TRAINED_SOURCE_FORMAT,
  NORMALIZED_TRAINED_SOURCE_RUNTIME,
  exportTrainedPTQSource,
} from '../tools/export_trained_ptq_source.mjs';

function values(length, offset = 0) {
  return Float32Array.from({ length }, (_, index) => (offset + index + 1) / 16);
}

function elements(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

async function trainedFixture() {
  const root = await mkdtemp(join(tmpdir(), 'volvox-tiny-receipt-source-ptq-'));
  const source = join(root, 'full_model');
  const output = join(root, 'normalized');
  await mkdir(source);
  const d = 4;
  const ff = 8;
  const rank = 1;
  const hidden = 2;
  const vocab = { itos: ['<pad>', '<bos>', '<eos>', '<unk>', 'a', 'b', 'c'] };
  const file = SafetensorsFile.empty({
    metadata: {
      format: 'tiny_receipt_vqa.volvox.v2',
      'volvox.model_format': 'volvox.api.v1',
      'volvox.model_origin': 'api',
    },
  });
  const originals = new Map();
  let serial = 0;
  const add = (name, shape, data = null) => {
    const buffer = data ?? values(elements(shape), serial++);
    originals.set(name, buffer);
    file.addTensor(name, 'F32', shape, new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength));
  };

  add('vqa.img_pos', [1, 210, d]);
  add('vqa.q_pos', [2, d]);
  add('vqa.y_pos', [1, 2, d]);
  add('vqa.type_img', [1, 1, d]);
  add('vqa.type_q', [1, 1, d]);
  add('vqa.tok.weight', [vocab.itos.length, d]);
  add('vqa.out_bias', [vocab.itos.length]);
  add('vqa.lora.scale', [1], Float32Array.of(2));

  const channels = [2, 3, 4, d, d];
  let inputChannels = 1;
  for (const [slot, index] of [0, 1, 3, 5, 7].entries()) {
    const outputChannels = channels[slot];
    add(`vqa.stem.${index}.conv.weight`, [3, 3, inputChannels, outputChannels]);
    add(`vqa.stem.${index}.norm.weight`, [outputChannels]);
    add(`vqa.stem.${index}.norm.bias`, [outputChannels]);
    inputChannels = outputChannels;
    if (index + 1 < 9 && [2, 4, 6, 8].includes(index + 1)) {
      const residual = index + 1;
      for (const suffix of ['conv1', 'conv2']) {
        add(`vqa.stem.${residual}.${suffix}.weight`, [3, 3, outputChannels, outputChannels]);
      }
      for (const suffix of ['norm1', 'norm2']) {
        add(`vqa.stem.${residual}.${suffix}.weight`, [outputChannels]);
        add(`vqa.stem.${residual}.${suffix}.bias`, [outputChannels]);
      }
    }
  }

  const addAttention = (root) => {
    add(`${root}.qkv.weight`, [d, 3 * d]);
    add(`${root}.qkv.bias`, [3 * d]);
    add(`${root}.output.weight`, [d, d]);
    add(`${root}.output.bias`, [d]);
    add(`${root}.norm.weight`, [d]);
    add(`${root}.norm.bias`, [d]);
  };
  const addFfn = (root) => {
    add(`${root}.norm.weight`, [d]);
    add(`${root}.norm.bias`, [d]);
    for (const [linear, input, outputWidth] of [
      ['linear1', d, ff],
      ['linear2', ff, d],
    ]) {
      add(`${root}.${linear}.base.weight`, [input, outputWidth]);
      add(`${root}.${linear}.base.bias`, [outputWidth]);
      add(`${root}.${linear}.lora_a`, [input, rank]);
      add(`${root}.${linear}.lora_b`, [rank, outputWidth]);
    }
  };
  addAttention('vqa.encoder.0.self_attention');
  addFfn('vqa.encoder.0.feed_forward');
  addAttention('vqa.decoder.0.self_attention');
  for (const projection of ['query', 'key', 'value', 'output']) {
    add(`vqa.decoder.0.cross_attention.${projection}.weight`, [d, d]);
    add(`vqa.decoder.0.cross_attention.${projection}.bias`, [d]);
  }
  add('vqa.decoder.0.cross_attention.norm.weight', [d]);
  add('vqa.decoder.0.cross_attention.norm.bias', [d]);
  addFfn('vqa.decoder.0.feed_forward');
  add('vqa.router.linear1.weight', [d, d / 2]);
  add('vqa.router.linear1.bias', [d / 2]);
  add('vqa.router.linear2.weight', [d / 2, 8]);
  add('vqa.router.linear2.bias', [8]);
  for (const group of ['memory_adapters', 'decoder_adapters']) {
    add(`vqa.${group}.down.weight`, [8, d, hidden]);
    add(`vqa.${group}.down.bias`, [8, hidden]);
    add(`vqa.${group}.up.weight`, [8, hidden, d]);
    add(`vqa.${group}.up.bias`, [8, d]);
  }
  add('vqa.norm.weight', [d]);
  add('vqa.norm.bias', [d]);

  const config = {
    format: 'volvox.api.v1',
    inputs: {
      'calibration.q': { shape: [1, 1, d], dtype: 'float32' },
      'calibration.k': { shape: [1, 1, d], dtype: 'float32' },
      'calibration.v': { shape: [1, 1, d], dtype: 'float32' },
      'calibration.linear_input': { shape: [1, d], dtype: 'float32' },
    },
    outputs: ['calibration.attention', 'calibration.linear_output'],
    nodes: [{
      opType: 'CrossSDPA',
      inputs: { q: 'calibration.q', k: 'calibration.k', v: 'calibration.v' },
      outputs: { out: 'calibration.attention' },
      outputs_shape: { out: [1, 1, d] },
      outputs_dtype: { out: 'float32' },
      params: { heads: 1, causal: false },
    }, {
      opType: 'Linear',
      inputs: {
        input: 'calibration.linear_input',
        weight: 'vqa.router.linear1.weight',
        bias: 'vqa.router.linear1.bias',
      },
      outputs: { out: 'calibration.linear_output' },
      outputs_shape: { out: [1, d / 2] },
      outputs_dtype: { out: 'float32' },
      params: { weight_layout: 'IN_OUT' },
    }],
  };
  await Promise.all([
    writeFile(join(source, 'config.json'), `${JSON.stringify(config)}\n`),
    writeFile(join(source, 'model.safetensors'), new Uint8Array(file.toArrayBuffer())),
    writeFile(join(source, 'vocab.json'), `${JSON.stringify(vocab)}\n`),
  ]);
  return { root, source, output, originals, channels };
}

function transpose2d(values, rows, columns) {
  const output = new Float32Array(values.length);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      output[column * rows + row] = values[row * columns + column];
    }
  }
  return output;
}

function hwioToOihw(values, height, width, inputs, outputs) {
  const result = new Float32Array(values.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    for (let input = 0; input < inputs; input++) for (let out = 0; out < outputs; out++) {
      result[((out * inputs + input) * height + y) * width + x] =
        values[((y * width + x) * inputs + input) * outputs + out];
    }
  }
  return result;
}

test('trained full-model weights map through JS PTQ into the distinct named source contract', async (t) => {
  const fixture = await trainedFixture();
  t.after(() => rm(fixture.root, { recursive: true, force: true }));
  const result = await exportTrainedPTQSource({
    sourceDirectory: fixture.source,
    outputDirectory: fixture.output,
  });

  assert.equal(result.manifest.format, NORMALIZED_TRAINED_SOURCE_FORMAT);
  assert.equal(result.manifest.runtime, NORMALIZED_TRAINED_SOURCE_RUNTIME);
  assert.deepEqual(result.manifest.safetensors.layout, NAMED_SOURCE_LAYOUT);
  assert.deepEqual(result.dimensions.stem_channels, fixture.channels);
  assert.equal(result.dimensions.max_out_len, 2);
  assert.equal(result.dimensions.lora_alpha, 2);
  assert.equal(result.manifest.activation_profile, null);

  const weights = SafetensorsFile.fromArrayBuffer(result.weightsBuffer);
  const convSource = fixture.originals.get('vqa.stem.0.conv.weight');
  const convExpected = packPTQWeight(
    hwioToOihw(convSource, 3, 3, 1, 2), [2, 1, 3, 3], { axis: 0 },
  );
  assert.deepEqual(
    [...weights.toRuntimeTypedArray(weights.getTensor('quantized.stem.0.net.0.weight'))],
    [...convExpected.data],
  );
  assert.deepEqual(weights.getTensor('quantized.stem.0.net.0.weight').shape, [2, 1, 3, 3]);

  const qkvSource = fixture.originals.get('vqa.encoder.0.self_attention.qkv.weight');
  const qkvExpected = packPTQWeight(transpose2d(qkvSource, 4, 12), [12, 4], { axis: 0 });
  assert.deepEqual(
    [...weights.toRuntimeTypedArray(weights.getTensor(
      'quantized.encoder.layers.0.self_attn.in_proj_weight',
    ))],
    [...qkvExpected.data],
  );
  assert.deepEqual(
    weights.getTensor('quantized.decoder.layers.0.multihead_attn.in_proj_weight').shape,
    [12, 4],
  );
  assert.deepEqual(weights.getTensor('quantized.memory_adapters.7.down.weight').shape, [2, 4]);
  assert.deepEqual(weights.getTensor('float32.q_pos').shape, [1, 2, 4]);
  assert.deepEqual(
    [...weights.getTensor('quantized.head.weight').getBytes()],
    [...weights.getTensor('quantized.tok.weight').getBytes()],
  );
  assert.deepEqual(
    [...weights.getTensor('scale.head.weight').getBytes()],
    [...weights.getTensor('scale.tok.weight').getBytes()],
  );

  const diskManifest = JSON.parse(await readFile(join(fixture.output, 'manifest.json'), 'utf8'));
  assert.equal(diskManifest.source.layout_mapping.conv, 'HWIO_to_OIHW');
  assert.equal(diskManifest.source.layout_mapping.expert, 'packed_8_IN_OUT_to_per_family_OUT_IN');
  assert.equal(diskManifest.tensors['head.weight'].dtype, 'int8_per_out');

  const calibrated = await exportTrainedPTQSource({
    sourceDirectory: fixture.source,
    sampleDocument: {
      format: 'volvoxai-tiny-receipt-vqa-named-calibration-samples-v1',
      samples: [{
        'calibration.q': [1, 0, -1, 0.5],
        'calibration.k': [0.5, -0.5, 1, 0],
        'calibration.v': [2, 1, 0, -1],
        'calibration.linear_input': [1, -2, 3, -4],
      }],
    },
  });
  assert.equal(calibrated.activation.profile.format,
    'volvoxai-tiny-receipt-vqa-w8a8-activation-scales-v1');
  assert.equal(calibrated.activation.report.representative, true);
  assert.equal(calibrated.activation.report.qualification,
    'single_global_fallback_not_per_edge_calibration');
  assert.ok(calibrated.activation.profile.default_scale > 0);

  const packageDirectory = join(fixture.root, 'package');
  const materializer = fileURLToPath(new URL(
    '../tools/materialize_tiny_receipt_vqa_w8a8.py', import.meta.url,
  ));
  const assembled = spawnSync('python3', [
    materializer,
    '--manifest', join(fixture.output, 'manifest.json'),
    '--activation-scale', '0.125',
    '--out-dir', packageDirectory,
  ], { encoding: 'utf8' });
  if (assembled.error?.code === 'ENOENT' ||
      (assembled.status !== 0 && /No module named|dependencies are unavailable/.test(
        String(assembled.stderr),
      ))) {
    t.diagnostic('Python SafeTensors dependencies unavailable; skipped cross-language assembly check.');
  } else {
    assert.equal(assembled.status, 0, assembled.stderr);
    const packageManifest = JSON.parse(await readFile(
      join(packageDirectory, 'package_manifest.json'), 'utf8',
    ));
    assert.equal(packageManifest.source.format, NORMALIZED_TRAINED_SOURCE_FORMAT);
    assert.equal(packageManifest.source.runtime, NORMALIZED_TRAINED_SOURCE_RUNTIME);
    assert.equal(packageManifest.explicit_families.phone.config,
      'explicit_family_phone_config.json');
  }
});
