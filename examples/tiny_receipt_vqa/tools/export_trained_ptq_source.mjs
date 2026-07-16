#!/usr/bin/env node

/**
 * Normalize the native TinyReceipt trainer's F32 package into the named INT8
 * source contract consumed by materialize_tiny_receipt_vqa_w8a8.py.
 *
 * Quantization math is owned by ts/training/Quantization.ts. This file owns
 * only TinyReceipt-specific name and layout mapping.
 */

import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  CPUEngine,
  Graph,
  calibratePTQ,
  derivePTQParameters,
  materializePTQWeights,
} from '../../../ts/full.js';
import {
  LINEAR_SAMPLE_FORMAT,
  TRAINED_SOURCE_FORMAT,
  assembleTrainedSourceGraph,
  loadTrainedTinyReceiptPackage,
  makeTrainedStructuralSample,
  normalizeTrainedCalibrationSamples,
} from './materialize_trained_linear_ptq.mjs';

export const NORMALIZED_TRAINED_SOURCE_FORMAT =
  'tiny_receipt_vqa_volvox_trained_int8_safetensors_v1';
export const NORMALIZED_TRAINED_SOURCE_RUNTIME =
  'volvoxai-js-ptq-native-trained-f32-v1';
export const ACTIVATION_PROFILE_FORMAT =
  'volvoxai-tiny-receipt-vqa-w8a8-activation-scales-v1';

export const NAMED_SOURCE_LAYOUT = Object.freeze({
  int8_per_out: 'quantized.<state_dict_key>',
  scale_fp32: 'scale.<state_dict_key>',
  float32: 'float32.<state_dict_key>',
});
export const NAMED_SOURCE_QUANTIZATION = 'symmetric per-output-channel INT8';

const FAMILY_COUNT = 8;
const IMAGE_TOKENS = 210;

function count(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function sameShape(actual, expected) {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

function finiteF32(values, label) {
  if (!(values instanceof Float32Array)) throw new Error(`${label} must use F32 storage.`);
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) throw new Error(`${label} contains non-finite value ${index}.`);
  }
  return values;
}

function sourceF32(source, name, expectedShape = null) {
  const entry = source.weights.getTensor(name);
  if (!entry || entry.dtype !== 'F32') throw new Error(`trained source is missing F32 tensor '${name}'.`);
  if (expectedShape && !sameShape(entry.shape, expectedShape)) {
    throw new Error(
      `trained tensor '${name}' has shape [${entry.shape}], expected [${expectedShape}].`,
    );
  }
  return {
    shape: [...entry.shape],
    data: finiteF32(source.weights.toRuntimeTypedArray(entry), `trained tensor '${name}'`),
  };
}

function transpose2d(tensor, label) {
  const [rows, columns] = tensor.shape;
  if (tensor.shape.length !== 2) throw new Error(`${label} requires rank-2 storage.`);
  const output = new Float32Array(tensor.data.length);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      output[column * rows + row] = tensor.data[row * columns + column];
    }
  }
  return { shape: [columns, rows], data: output };
}

function hwioToOihw(tensor, label) {
  if (tensor.shape.length !== 4) throw new Error(`${label} requires rank-4 HWIO storage.`);
  const [height, width, inputs, outputs] = tensor.shape;
  const output = new Float32Array(tensor.data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let input = 0; input < inputs; input++) {
        for (let out = 0; out < outputs; out++) {
          output[((out * inputs + input) * height + y) * width + x] =
            tensor.data[((y * width + x) * inputs + input) * outputs + out];
        }
      }
    }
  }
  return { shape: [outputs, inputs, height, width], data: output };
}

function concatRows(tensors, label) {
  if (tensors.length === 0 || tensors.some((tensor) => tensor.shape.length !== 2)) {
    throw new Error(`${label} requires rank-2 tensors.`);
  }
  const columns = tensors[0].shape[1];
  if (tensors.some((tensor) => tensor.shape[1] !== columns)) {
    throw new Error(`${label} tensors must have matching row width.`);
  }
  const rows = tensors.reduce((total, tensor) => total + tensor.shape[0], 0);
  const output = new Float32Array(rows * columns);
  let offset = 0;
  for (const tensor of tensors) {
    output.set(tensor.data, offset);
    offset += tensor.data.length;
  }
  return { shape: [rows, columns], data: output };
}

function concatVectors(tensors, label) {
  if (tensors.length === 0 || tensors.some((tensor) => tensor.shape.length !== 1)) {
    throw new Error(`${label} requires rank-1 tensors.`);
  }
  const length = tensors.reduce((total, tensor) => total + tensor.shape[0], 0);
  const output = new Float32Array(length);
  let offset = 0;
  for (const tensor of tensors) {
    output.set(tensor.data, offset);
    offset += tensor.data.length;
  }
  return { shape: [length], data: output };
}

function expertOutIn(tensor, family, label) {
  if (tensor.shape.length !== 3 || tensor.shape[0] !== FAMILY_COUNT) {
    throw new Error(`${label} requires [${FAMILY_COUNT},in,out] expert storage.`);
  }
  const [, inputs, outputs] = tensor.shape;
  const sourceOffset = family * inputs * outputs;
  const output = new Float32Array(inputs * outputs);
  for (let input = 0; input < inputs; input++) {
    for (let out = 0; out < outputs; out++) {
      output[out * inputs + input] = tensor.data[sourceOffset + input * outputs + out];
    }
  }
  return { shape: [outputs, inputs], data: output };
}

function expertVector(tensor, family, label) {
  if (tensor.shape.length !== 2 || tensor.shape[0] !== FAMILY_COUNT) {
    throw new Error(`${label} requires [${FAMILY_COUNT},width] expert storage.`);
  }
  const width = tensor.shape[1];
  return {
    shape: [width],
    data: new Float32Array(tensor.data.subarray(family * width, (family + 1) * width)),
  };
}

function contiguousLayerCount(names, expression, label) {
  const indices = new Set();
  for (const name of names) {
    const match = expression.exec(name);
    if (match) indices.add(Number(match[1]));
  }
  if (indices.size === 0) throw new Error(`trained source has no ${label} layers.`);
  const ordered = [...indices].sort((left, right) => left - right);
  if (ordered.some((value, index) => value !== index)) {
    throw new Error(`trained ${label} layer indices must be contiguous from zero.`);
  }
  return ordered.length;
}

function graphOp(node) {
  return node?.opType ?? node?.op;
}

function deriveDimensions(source, vocab) {
  if (!Array.isArray(vocab?.itos) || vocab.itos.length < 4 ||
      vocab.itos.slice(0, 4).join('\0') !== ['<pad>', '<bos>', '<eos>', '<unk>'].join('\0')) {
    throw new Error('trained vocab.json must contain the canonical itos special-token prefix.');
  }
  const token = sourceF32(source, 'vqa.tok.weight');
  if (token.shape.length !== 2 || token.shape[0] !== vocab.itos.length) {
    throw new Error('trained token table must be [vocabulary,d_model].');
  }
  const [vocabSize, dModel] = token.shape;
  const names = source.weights.listTensorNames();
  const encLayers = contiguousLayerCount(
    names, /^vqa\.encoder\.(\d+)\.self_attention\.qkv\.weight$/, 'encoder',
  );
  const decLayers = contiguousLayerCount(
    names, /^vqa\.decoder\.(\d+)\.self_attention\.qkv\.weight$/, 'decoder',
  );
  const heads = new Set(
    source.config.nodes
      .filter((node) => ['SDPA', 'CrossSDPA'].includes(graphOp(node)))
      .map((node) => node.params?.heads),
  );
  if (heads.size !== 1 || !Number.isInteger([...heads][0]) || [...heads][0] <= 0) {
    throw new Error('trained attention nodes must agree on one positive heads value.');
  }
  const ff = sourceF32(source, 'vqa.encoder.0.feed_forward.linear1.base.weight');
  if (ff.shape.length !== 2 || ff.shape[0] !== dModel || ff.shape[1] % dModel !== 0) {
    throw new Error('trained encoder FFN width is incompatible with d_model.');
  }
  const qPosition = sourceF32(source, 'vqa.q_pos');
  const yPosition = sourceF32(source, 'vqa.y_pos');
  const imagePosition = sourceF32(source, 'vqa.img_pos', [1, IMAGE_TOKENS, dModel]);
  void imagePosition;
  if (qPosition.shape.length !== 2 || qPosition.shape[1] !== dModel ||
      yPosition.shape.length !== 3 || yPosition.shape[0] !== 1 || yPosition.shape[2] !== dModel) {
    throw new Error('trained position tables have incompatible shapes.');
  }
  const adapter = sourceF32(source, 'vqa.memory_adapters.down.weight');
  if (adapter.shape.length !== 3 || adapter.shape[0] !== FAMILY_COUNT || adapter.shape[1] !== dModel) {
    throw new Error('trained task adapters must use [8,d_model,bottleneck] storage.');
  }
  const loraA = sourceF32(source, 'vqa.encoder.0.feed_forward.linear1.lora_a');
  const loraScale = sourceF32(source, 'vqa.lora.scale', [1]);
  if (loraA.shape.length !== 2 || loraA.shape[0] !== dModel || loraA.shape[1] <= 0 ||
      !Number.isFinite(loraScale.data[0]) || loraScale.data[0] <= 0) {
    throw new Error('trained LoRA tensors have an incompatible rank/scale contract.');
  }
  const stemChannels = [0, 1, 3, 5, 7].map((index) => {
    const tensor = sourceF32(source, `vqa.stem.${index}.conv.weight`);
    if (tensor.shape.length !== 4 || tensor.shape[0] !== 3 || tensor.shape[1] !== 3) {
      throw new Error(`trained stem ${index} weight must use [3,3,in,out] HWIO storage.`);
    }
    return tensor.shape[3];
  });
  if (stemChannels[3] !== dModel || stemChannels[4] !== dModel) {
    throw new Error('trained final stem channels must equal d_model.');
  }
  return Object.freeze({
    vocab_size: vocabSize,
    d_model: dModel,
    heads: [...heads][0],
    enc_layers: encLayers,
    dec_layers: decLayers,
    ff_mult: ff.shape[1] / dModel,
    dropout: 0,
    max_q_len: qPosition.shape[0],
    max_out_len: yPosition.shape[1],
    img_tokens: IMAGE_TOKENS,
    use_adapters: true,
    use_router: true,
    adapter_bottleneck: adapter.shape[2],
    adapter_families: FAMILY_COUNT,
    lora_r: loraA.shape[1],
    lora_alpha: Math.fround(loraScale.data[0] * loraA.shape[1]),
    lora_dropout: 0,
    lora_targets: 'transformer',
    stem_channels: stemChannels,
  });
}

class MappingBuilder {
  constructor(source, dimensions) {
    this.source = source;
    this.dimensions = dimensions;
    this.quantized = [];
    this.float32 = [];
    this.sourceNames = new Set();
  }

  q(key, sourceName, transform = null) {
    const source = sourceF32(this.source, sourceName);
    const target = transform ? transform(source, sourceName) : {
      shape: [...source.shape], data: new Float32Array(source.data),
    };
    this.quantized.push({ key, target, sourceNames: [sourceName] });
    this.sourceNames.add(sourceName);
  }

  qCombined(key, sourceNames, combine) {
    const sources = sourceNames.map((name) => sourceF32(this.source, name));
    this.quantized.push({ key, target: combine(sources, key), sourceNames: [...sourceNames] });
    for (const name of sourceNames) this.sourceNames.add(name);
  }

  f(key, sourceName, transform = null) {
    const source = sourceF32(this.source, sourceName);
    const target = transform ? transform(source, sourceName) : {
      shape: [...source.shape], data: new Float32Array(source.data),
    };
    this.float32.push({ key, target, sourceNames: [sourceName] });
    this.sourceNames.add(sourceName);
  }

  build() {
    this.f('img_pos', 'vqa.img_pos');
    this.f('q_pos', 'vqa.q_pos', (tensor) => ({ shape: [1, ...tensor.shape], data: tensor.data }));
    this.f('y_pos', 'vqa.y_pos');
    this.f('type_img', 'vqa.type_img');
    this.f('type_q', 'vqa.type_q');
    this.f('out_bias', 'vqa.out_bias');

    const convBlocks = [0, 1, 3, 5, 7];
    const residualBlocks = [2, 4, 6, 8];
    for (const index of convBlocks) {
      this.q(`stem.${index}.net.0.weight`, `vqa.stem.${index}.conv.weight`, hwioToOihw);
      this.f(`stem.${index}.net.1.weight`, `vqa.stem.${index}.norm.weight`);
      this.f(`stem.${index}.net.1.bias`, `vqa.stem.${index}.norm.bias`);
    }
    for (const index of residualBlocks) {
      this.q(`stem.${index}.net.0.weight`, `vqa.stem.${index}.conv1.weight`, hwioToOihw);
      this.f(`stem.${index}.net.1.weight`, `vqa.stem.${index}.norm1.weight`);
      this.f(`stem.${index}.net.1.bias`, `vqa.stem.${index}.norm1.bias`);
      this.q(`stem.${index}.net.3.weight`, `vqa.stem.${index}.conv2.weight`, hwioToOihw);
      this.f(`stem.${index}.net.4.weight`, `vqa.stem.${index}.norm2.weight`);
      this.f(`stem.${index}.net.4.bias`, `vqa.stem.${index}.norm2.bias`);
    }
    this.q('tok.weight', 'vqa.tok.weight');

    const attention = (external, native) => {
      this.q(`${external}.in_proj_weight`, `${native}.qkv.weight`, transpose2d);
      this.f(`${external}.in_proj_bias`, `${native}.qkv.bias`);
      this.q(`${external}.out_proj.weight`, `${native}.output.weight`, transpose2d);
      this.f(`${external}.out_proj.bias`, `${native}.output.bias`);
    };
    const ffn = (external, native) => {
      for (const linear of ['linear1', 'linear2']) {
        this.q(`${external}.${linear}.base.weight`, `${native}.${linear}.base.weight`, transpose2d);
        this.f(`${external}.${linear}.base.bias`, `${native}.${linear}.base.bias`);
        this.q(`${external}.${linear}.lora_a.weight`, `${native}.${linear}.lora_a`, transpose2d);
        this.q(`${external}.${linear}.lora_b.weight`, `${native}.${linear}.lora_b`, transpose2d);
      }
    };
    for (let layer = 0; layer < this.dimensions.enc_layers; layer++) {
      const external = `encoder.layers.${layer}`;
      const native = `vqa.encoder.${layer}`;
      attention(`${external}.self_attn`, `${native}.self_attention`);
      ffn(external, `${native}.feed_forward`);
      this.f(`${external}.norm1.weight`, `${native}.self_attention.norm.weight`);
      this.f(`${external}.norm1.bias`, `${native}.self_attention.norm.bias`);
      this.f(`${external}.norm2.weight`, `${native}.feed_forward.norm.weight`);
      this.f(`${external}.norm2.bias`, `${native}.feed_forward.norm.bias`);
    }
    for (let layer = 0; layer < this.dimensions.dec_layers; layer++) {
      const external = `decoder.layers.${layer}`;
      const native = `vqa.decoder.${layer}`;
      attention(`${external}.self_attn`, `${native}.self_attention`);
      const crossNative = `${native}.cross_attention`;
      const crossExternal = `${external}.multihead_attn`;
      const projections = ['query', 'key', 'value'];
      this.qCombined(
        `${crossExternal}.in_proj_weight`,
        projections.map((name) => `${crossNative}.${name}.weight`),
        (items, label) => concatRows(items.map((item) => transpose2d(item, label)), label),
      );
      const crossBiasNames = projections.map((name) => `${crossNative}.${name}.bias`);
      const crossBiases = crossBiasNames.map((name) => sourceF32(this.source, name));
      this.float32.push({
        key: `${crossExternal}.in_proj_bias`,
        target: concatVectors(crossBiases, `${crossExternal}.in_proj_bias`),
        sourceNames: crossBiasNames,
      });
      for (const name of crossBiasNames) this.sourceNames.add(name);
      this.q(`${crossExternal}.out_proj.weight`, `${crossNative}.output.weight`, transpose2d);
      this.f(`${crossExternal}.out_proj.bias`, `${crossNative}.output.bias`);
      ffn(external, `${native}.feed_forward`);
      for (const [norm, sourceNorm] of [
        [1, `${native}.self_attention.norm`],
        [2, `${native}.cross_attention.norm`],
        [3, `${native}.feed_forward.norm`],
      ]) {
        this.f(`${external}.norm${norm}.weight`, `${sourceNorm}.weight`);
        this.f(`${external}.norm${norm}.bias`, `${sourceNorm}.bias`);
      }
    }
    this.q('router.net.0.weight', 'vqa.router.linear1.weight', transpose2d);
    this.f('router.net.0.bias', 'vqa.router.linear1.bias');
    this.q('router.net.2.weight', 'vqa.router.linear2.weight', transpose2d);
    this.f('router.net.2.bias', 'vqa.router.linear2.bias');

    for (const [external, native] of [
      ['memory_adapters', 'vqa.memory_adapters'],
      ['decoder_adapters', 'vqa.decoder_adapters'],
    ]) {
      const down = sourceF32(this.source, `${native}.down.weight`);
      const downBias = sourceF32(this.source, `${native}.down.bias`);
      const up = sourceF32(this.source, `${native}.up.weight`);
      const upBias = sourceF32(this.source, `${native}.up.bias`);
      for (let family = 0; family < FAMILY_COUNT; family++) {
        this.quantized.push({
          key: `${external}.${family}.down.weight`,
          target: expertOutIn(down, family, `${native}.down.weight`),
          sourceNames: [`${native}.down.weight`],
        });
        this.float32.push({
          key: `${external}.${family}.down.bias`,
          target: expertVector(downBias, family, `${native}.down.bias`),
          sourceNames: [`${native}.down.bias`],
        });
        this.quantized.push({
          key: `${external}.${family}.up.weight`,
          target: expertOutIn(up, family, `${native}.up.weight`),
          sourceNames: [`${native}.up.weight`],
        });
        this.float32.push({
          key: `${external}.${family}.up.bias`,
          target: expertVector(upBias, family, `${native}.up.bias`),
          sourceNames: [`${native}.up.bias`],
        });
      }
      for (const name of [
        `${native}.down.weight`, `${native}.down.bias`, `${native}.up.weight`, `${native}.up.bias`,
      ]) this.sourceNames.add(name);
    }
    this.f('norm.weight', 'vqa.norm.weight');
    this.f('norm.bias', 'vqa.norm.bias');

    for (const entry of [...this.quantized, ...this.float32]) {
      if (count(entry.target.shape) !== entry.target.data.length) {
        throw new Error(`mapped tensor '${entry.key}' has inconsistent shape/storage.`);
      }
    }
    if (this.quantized.some((entry) => entry.target.shape[0] <= 0) ||
        this.dimensions.d_model <= 0) {
      throw new Error('mapped quantized weights require a positive output-channel axis.');
    }
    return this;
  }
}

function byteView(values) {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function materializeNamedWeights(mapping) {
  const graph = new Graph();
  const requests = [];
  for (const [index, entry] of mapping.quantized.entries()) {
    const sourceName = `ptq.source.${index}`;
    graph.addWeight(sourceName, entry.target.shape, 'float32', { buffer: entry.target.data });
    requests.push({
      name: sourceName,
      outputName: `quantized.${entry.key}`,
      scaleName: `scale.${entry.key}`,
      axis: 0,
    });
  }
  const artifact = materializePTQWeights(graph, requests, {
    includeUnselected: false,
    metadata: {
      normalized_source_format: NORMALIZED_TRAINED_SOURCE_FORMAT,
      normalized_source_runtime: NORMALIZED_TRAINED_SOURCE_RUNTIME,
    },
  });
  const token = artifact.weights.getTensor('quantized.tok.weight');
  const tokenScale = artifact.weights.getTensor('scale.tok.weight');
  artifact.weights.addTensor('quantized.head.weight', 'I8', token.shape, token.getBytes());
  artifact.weights.addTensor('scale.head.weight', 'F32', tokenScale.shape, tokenScale.getBytes());
  for (const entry of mapping.float32) {
    artifact.weights.addTensor(
      `float32.${entry.key}`, 'F32', entry.target.shape, byteView(entry.target.data),
    );
  }
  return artifact;
}

function manifestTensors(mapping) {
  const output = {};
  for (const entry of mapping.quantized) {
    output[entry.key] = { dtype: 'int8_per_out', shape: [...entry.target.shape] };
  }
  output['head.weight'] = {
    dtype: 'int8_per_out',
    shape: [...mapping.quantized.find((entry) => entry.key === 'tok.weight').target.shape],
  };
  for (const entry of mapping.float32) {
    output[entry.key] = { dtype: 'float32', shape: [...entry.target.shape] };
  }
  return output;
}

async function globalActivationProfile(source, samples, representative) {
  const executor = new CPUEngine().allocateGraph(source.graph);
  const tensorNames = [...source.graph.tensors.values()]
    .filter((tensor) => !tensor.isWeight && tensor.dtype === 'float32')
    .map((tensor) => tensor.name);
  const calibrator = await calibratePTQ(executor, samples, { tensorNames });
  let minimum = Infinity;
  let maximum = -Infinity;
  let sampleCount = 0;
  for (const observer of calibrator.observers.values()) {
    const snapshot = observer.snapshot();
    minimum = Math.min(minimum, snapshot.minimum);
    maximum = Math.max(maximum, snapshot.maximum);
    sampleCount += snapshot.sampleCount;
  }
  const parameters = derivePTQParameters(
    { minimum, maximum, sampleCount }, { dtype: 'int8', scheme: 'symmetric' },
  );
  return {
    profile: {
      format: ACTIVATION_PROFILE_FORMAT,
      profile_id: representative
        ? 'volvox-trained-global-fallback-v1'
        : 'volvox-trained-zero-smoke-global-fallback-v1',
      default_scale: parameters.scale,
      scales: {},
    },
    report: {
      format: 'volvoxai-tiny-receipt-vqa-trained-global-calibration-v1',
      representative,
      source: representative ? LINEAR_SAMPLE_FORMAT : 'all_zero_structural_smoke',
      graph_tensor_count: tensorNames.length,
      sample_count: samples.length,
      observed_min: parameters.observed_min,
      observed_max: parameters.observed_max,
      observed_element_count: parameters.sample_count,
      default_scale: parameters.scale,
      qualification: 'single_global_fallback_not_per_edge_calibration',
    },
  };
}

function hash(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

async function outputDirectory(path, sourceRoot) {
  const target = resolve(path);
  if (target === resolve(sourceRoot)) throw new Error('output must not overwrite the trained package.');
  try {
    const info = await stat(target);
    if (!info.isDirectory() || (await readdir(target)).length !== 0) {
      throw new Error(`output directory '${target}' must be absent or empty.`);
    }
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    await mkdir(target, { recursive: true });
  }
  return target;
}

/** Export the full trained parameter mapping; optionally calibrate one global activation fallback. */
export async function exportTrainedPTQSource({
  sourceDirectory,
  sourcePackage = null,
  outputDirectory: requestedOutput = null,
  sampleDocument = null,
  structuralSmoke = false,
} = {}) {
  if (sampleDocument != null && structuralSmoke) {
    throw new Error('sampleDocument and structuralSmoke are mutually exclusive.');
  }
  let source = sourcePackage ?? await loadTrainedTinyReceiptPackage(sourceDirectory);
  const vocabPath = join(source.root, 'vocab.json');
  let vocab;
  try {
    vocab = JSON.parse(await readFile(vocabPath, 'utf8'));
  } catch (error) {
    throw new Error(`could not read trained vocab.json: ${error.message}`);
  }
  const dimensions = deriveDimensions(source, vocab);
  const mapping = new MappingBuilder(source, dimensions).build();
  const artifact = materializeNamedWeights(mapping);
  const weightsBuffer = artifact.weights.toArrayBuffer();

  let activation = null;
  if (sampleDocument != null || structuralSmoke) {
    source = source.graph ? source : await assembleTrainedSourceGraph(source);
    const samples = structuralSmoke
      ? [makeTrainedStructuralSample(source)]
      : normalizeTrainedCalibrationSamples(source, sampleDocument);
    activation = await globalActivationProfile(source, samples, !structuralSmoke);
  }
  const manifest = {
    format: NORMALIZED_TRAINED_SOURCE_FORMAT,
    runtime: NORMALIZED_TRAINED_SOURCE_RUNTIME,
    files: { model_safetensors: 'model_int8.safetensors' },
    safetensors: {
      layout: NAMED_SOURCE_LAYOUT,
      quantization: NAMED_SOURCE_QUANTIZATION,
    },
    config: dimensions,
    vocab: { itos: [...vocab.itos] },
    tensors: manifestTensors(mapping),
    source: {
      format: TRAINED_SOURCE_FORMAT,
      config_sha256: source.hashes.config_sha256,
      weights_sha256: source.hashes.weights_sha256,
      layout_mapping: {
        conv: 'HWIO_to_OIHW',
        linear_lora_router: 'IN_OUT_to_OUT_IN',
        self_attention_qkv: 'IN_3OUT_to_3OUT_IN',
        cross_attention_qkv: 'transpose_then_QKV_row_concat',
        expert: 'packed_8_IN_OUT_to_per_family_OUT_IN',
        q_pos: 'Q_D_to_1_Q_D',
        token_head: 'exact_tied_quantized_copy',
      },
      mapped_source_tensor_count: mapping.sourceNames.size,
    },
    activation_profile: activation == null ? null : {
      file: 'activation_profile.json',
      representative: activation.report.representative,
      qualification: activation.report.qualification,
    },
  };

  if (requestedOutput != null) {
    const target = await outputDirectory(requestedOutput, source.root);
    const writes = [
      writeFile(join(target, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`),
      writeFile(join(target, 'model_int8.safetensors'), new Uint8Array(weightsBuffer)),
    ];
    if (activation != null) {
      writes.push(
        writeFile(join(target, 'activation_profile.json'), `${JSON.stringify(activation.profile, null, 2)}\n`),
        writeFile(join(target, 'calibration_report.json'), `${JSON.stringify(activation.report, null, 2)}\n`),
      );
    }
    await Promise.all(writes);
  }
  return Object.freeze({ source, dimensions, mapping, artifact, weightsBuffer, activation, manifest });
}

function argumentsFrom(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    if (flag === '--structural-smoke') options.structuralSmoke = true;
    else if (['--source', '--out-dir', '--samples'].includes(flag)) {
      if (index + 1 >= argv.length) throw new Error(`${flag} requires a value.`);
      options[flag === '--source' ? 'sourceDirectory' : flag === '--out-dir' ? 'outputDirectory' : 'samplesPath'] =
        argv[++index];
    } else throw new Error(`unknown argument '${flag}'.`);
  }
  if (!options.sourceDirectory || !options.outputDirectory) {
    throw new Error('usage: --source FULL_MODEL --out-dir DIR [--samples JSON | --structural-smoke]');
  }
  if (options.samplesPath && options.structuralSmoke) {
    throw new Error('--samples and --structural-smoke are mutually exclusive.');
  }
  return options;
}

async function main(argv) {
  const options = argumentsFrom(argv);
  const sampleDocument = options.samplesPath
    ? JSON.parse(await readFile(options.samplesPath, 'utf8'))
    : null;
  const result = await exportTrainedPTQSource({ ...options, sampleDocument });
  process.stdout.write(`${JSON.stringify({
    event: 'tiny_receipt_trained_ptq_source_exported',
    format: result.manifest.format,
    tensors: Object.keys(result.manifest.tensors).length,
    activation_profile: result.activation != null,
    output: resolve(options.outputDirectory),
  })}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`error: ${error.message}\n`);
    process.exitCode = 1;
  });
}
