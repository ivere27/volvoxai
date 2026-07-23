#!/usr/bin/env node

/**
 * Calibrate and materialize one Linear island from the native TinyReceipt
 * F32 `full_model` checkpoint package.
 *
 * This intentionally is not a whole-model graph lowering pass. The trained
 * graph still contains routing, MoELinear, CrossSDPA, residuals, and LoRA
 * branches whose W8A8 rewrite needs model-specific fusion decisions. This
 * tool provides an executable bridge for one canonical operator boundary:
 * run the real F32 graph over named samples, observe a selected Linear input
 * and output through VolvoxAI PTQ, transpose its IN_OUT weight to QLinear's
 * OUT_IN layout, and emit a standalone W8A8 graph.
 */

import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  Graph,
  GraphLoader,
  PTQCalibrator,
  SafetensorsFile,
  VolvoxAI,
  materializePTQWeights,
  quantizePTQ,
} from '../../../ts/full.js';

export const TRAINED_SOURCE_FORMAT =
  'volvoxai-tiny-receipt-vqa-trained-f32-api-package-v1';
export const LINEAR_SAMPLE_FORMAT =
  'volvoxai-tiny-receipt-vqa-named-calibration-samples-v1';
export const LINEAR_ISLAND_PACKAGE_FORMAT =
  'volvoxai-tiny-receipt-vqa-trained-linear-w8a8-island-v1';

const SOURCE_GRAPH_FORMAT = 'volvox-graph/v1';
const SOURCE_MODEL_FORMAT = 'tiny_receipt_vqa.volvox.v2';
const SOURCE_WEIGHT_LAYOUT = 'IN_OUT';
const TARGET_WEIGHT_LAYOUT = 'OUT_IN';

function object(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function nonEmptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function elementCount(shape, label) {
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new Error(`${label} requires a non-empty shape.`);
  }
  let count = 1;
  for (const [index, dimension] of shape.entries()) {
    if (!Number.isSafeInteger(dimension) || dimension <= 0) {
      throw new Error(`${label} has invalid dimension ${index}.`);
    }
    count *= dimension;
    if (!Number.isSafeInteger(count)) throw new Error(`${label} is too large.`);
  }
  return count;
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function arrayBufferOf(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

function typedValues(dtype, values, expectedLength, label) {
  if (!Array.isArray(values) && !ArrayBuffer.isView(values)) {
    throw new Error(`${label} must be an array.`);
  }
  if (values.length !== expectedLength) {
    throw new Error(`${label} has ${values.length} values, expected ${expectedLength}.`);
  }
  let ArrayType;
  if (dtype === 'float32') ArrayType = Float32Array;
  else if (dtype === 'int32') ArrayType = Int32Array;
  else if (dtype === 'int8') ArrayType = Int8Array;
  else if (dtype === 'uint8') ArrayType = Uint8Array;
  else throw new Error(`${label} uses unsupported input dtype '${dtype}'.`);
  const output = new ArrayType(expectedLength);
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new Error(`${label} contains a non-finite number at index ${index}.`);
    }
    if (dtype !== 'float32' && !Number.isInteger(value)) {
      throw new Error(`${label} contains a non-integer value at index ${index}.`);
    }
    output[index] = value;
    if (dtype !== 'float32' && output[index] !== value) {
      throw new Error(`${label} contains a value outside ${dtype} at index ${index}.`);
    }
  }
  return output;
}

function sourceFetch(source) {
  return async (url) => {
    if (url === source.graphPath) {
      return { ok: true, json: async () => source.graphDocument };
    }
    if (url === source.weightsPath) {
      return { ok: true, arrayBuffer: async () => source.weightsBuffer.slice(0) };
    }
    return { ok: false, statusText: `unexpected local source ${url}` };
  };
}

function assertSourceContract(graph, weights) {
  object(graph, 'trained graph');
  if (graph.format !== SOURCE_GRAPH_FORMAT) {
    throw new Error(
      `trained graph.format must be '${SOURCE_GRAPH_FORMAT}', got '${String(graph.format)}'.`,
    );
  }
  if (!Array.isArray(graph.nodes) || !graph.inputs || !graph.outputs) {
    throw new Error('trained graph must contain inputs, outputs, and nodes.');
  }
  const metadata = object(weights.metadata, 'trained weights metadata');
  if (metadata.format !== SOURCE_MODEL_FORMAT ||
      metadata['volvox.model_format'] !== SOURCE_GRAPH_FORMAT ||
      metadata['volvox.model_origin'] !== 'api') {
    throw new Error(
      'trained weights must carry TinyReceipt v2 API-model metadata ' +
      `(format='${SOURCE_MODEL_FORMAT}', volvox.model_format='${SOURCE_GRAPH_FORMAT}', ` +
      "volvox.model_origin='api').",
    );
  }
}

/** Load and validate the trained `full_model` package pair. */
export async function loadTrainedTinyReceiptPackage(directory) {
  const root = resolve(directory);
  const graphPath = join(root, 'graph.json');
  const weightsPath = join(root, 'model.safetensors');
  let graphBytes;
  let weightsBytes;
  try {
    [graphBytes, weightsBytes] = await Promise.all([readFile(graphPath), readFile(weightsPath)]);
  } catch (error) {
    throw new Error(`could not read trained TinyReceipt package '${root}': ${error.message}`);
  }
  let graphDocument;
  try {
    graphDocument = JSON.parse(graphBytes.toString('utf8'));
  } catch (error) {
    throw new Error(`trained graph.json is invalid JSON: ${error.message}`);
  }
  let weights;
  try {
    weights = SafetensorsFile.fromArrayBuffer(arrayBufferOf(weightsBytes));
  } catch (error) {
    throw new Error(`trained model.safetensors is invalid: ${error.message}`);
  }
  assertSourceContract(graphDocument, weights);
  return Object.freeze({
    format: TRAINED_SOURCE_FORMAT,
    root,
    graphPath,
    weightsPath,
    graphDocument,
    weights,
    graphBytes,
    weightsBytes,
    weightsBuffer: arrayBufferOf(weightsBytes),
    hashes: Object.freeze({
      graph_sha256: sha256(graphBytes),
      weights_sha256: sha256(weightsBytes),
    }),
  });
}

function linearMapping(source, weightName) {
  nonEmptyString(weightName, 'source Linear weight name');
  if (!weightName.startsWith('vqa.') || !weightName.endsWith('.weight')) {
    throw new Error("source Linear weight must be a TinyReceipt 'vqa.*.weight' tensor.");
  }
  const matches = source.graphDocument.nodes.filter((node) =>
    node?.opType === 'Linear' && node?.inputs?.weight === weightName);
  if (matches.length !== 1) {
    throw new Error(
      `source Linear weight '${weightName}' must be consumed by exactly one Linear node; found ${matches.length}.`,
    );
  }
  const node = matches[0];
  const inputNames = Object.keys(object(node.inputs, `Linear '${weightName}' inputs`)).sort();
  if (!inputNames.includes('input') || !inputNames.includes('weight') ||
      inputNames.some((name) => !['input', 'weight', 'bias'].includes(name))) {
    throw new Error(`Linear '${weightName}' must use exact input/weight and optional bias operands.`);
  }
  if (node.params?.weight_layout !== SOURCE_WEIGHT_LAYOUT || node.params?.transB != null) {
    throw new Error(
      `Linear '${weightName}' requires explicit ${SOURCE_WEIGHT_LAYOUT} storage without transB.`,
    );
  }
  const outputs = Object.entries(object(node.outputs, `Linear '${weightName}' outputs`));
  if (outputs.length !== 1 || outputs[0][0] !== 'out') {
    throw new Error(`Linear '${weightName}' must have exactly one 'out' output.`);
  }
  const inputName = nonEmptyString(node.inputs.input, `Linear '${weightName}' input`);
  const outputName = nonEmptyString(outputs[0][1], `Linear '${weightName}' output`);
  const outputShape = node.outputs_shape?.out;
  elementCount(outputShape, `Linear '${weightName}' output`);

  const inputTensor = source.graph?.tensors.get(inputName);
  const outputTensor = source.graph?.tensors.get(outputName);
  if (!inputTensor || !outputTensor || inputTensor.dtype !== 'float32' || outputTensor.dtype !== 'float32') {
    throw new Error(`Linear '${weightName}' requires F32 input and output graph tensors.`);
  }
  const sourceWeight = source.graph.tensors.get(weightName);
  if (!sourceWeight?.isWeight || sourceWeight.dtype !== 'float32' ||
      !(sourceWeight.buffer instanceof Float32Array) || sourceWeight.shape.length !== 2) {
    throw new Error(`Linear '${weightName}' requires one rank-2 F32 trained weight.`);
  }
  const inputWidth = inputTensor.shape.at(-1);
  const outputWidth = outputTensor.shape.at(-1);
  if (sourceWeight.shape[0] !== inputWidth || sourceWeight.shape[1] !== outputWidth ||
      outputShape.at(-1) !== outputWidth) {
    throw new Error(
      `Linear '${weightName}' IN_OUT shape must be [${inputWidth},${outputWidth}].`,
    );
  }
  const biasName = node.inputs.bias;
  const sourceBias = biasName == null ? null : source.graph.tensors.get(biasName);
  if (!sourceBias?.isWeight || sourceBias.dtype !== 'float32' ||
      !(sourceBias.buffer instanceof Float32Array) ||
      sourceBias.shape.length !== 1 || sourceBias.shape[0] !== outputWidth) {
    throw new Error(`Linear '${weightName}' requires an F32 [${outputWidth}] bias for QLinear materialization.`);
  }

  const stem = weightName.slice(0, -'.weight'.length);
  return Object.freeze({
    source: Object.freeze({
      input: inputName,
      output: outputName,
      weight: weightName,
      bias: biasName,
      weight_shape: Object.freeze([...sourceWeight.shape]),
      weight_layout: SOURCE_WEIGHT_LAYOUT,
    }),
    target: Object.freeze({
      input: 'island.input',
      output: 'island.output',
      weight: `w8a8.${stem}.weight`,
      weight_scale: `w8a8.${stem}.weight_scale`,
      bias: `w8a8.${stem}.bias_i32`,
      weight_shape: Object.freeze([outputWidth, inputWidth]),
      weight_layout: TARGET_WEIGHT_LAYOUT,
    }),
    inputShape: Object.freeze([...inputTensor.shape]),
    outputShape: Object.freeze([...outputTensor.shape]),
    node,
    sourceWeight,
    sourceBias,
  });
}

export async function assembleTrainedSourceGraph(source) {
  const graph = new Graph();
  await GraphLoader.load(graph, source.weightsPath, {
    graphUrl: source.graphPath,
    fetch: sourceFetch(source),
  });
  return Object.freeze({ ...source, graph });
}

/** Enumerate only mappings that satisfy the exact trained Linear contract. */
export async function inspectTrainedLinearMappings(sourcePackage) {
  const source = sourcePackage.graph ? sourcePackage : await assembleTrainedSourceGraph(sourcePackage);
  const candidates = [];
  for (const node of source.graphDocument.nodes) {
    const weightName = node?.opType === 'Linear' ? node?.inputs?.weight : null;
    if (typeof weightName !== 'string' || !weightName.startsWith('vqa.')) continue;
    try {
      const mapping = linearMapping(source, weightName);
      candidates.push({ source: mapping.source, target: mapping.target });
    } catch {
      // Inspection is deliberately conservative: malformed/unsupported nodes
      // are omitted; selecting one still returns the precise validation error.
    }
  }
  return Object.freeze(candidates);
}

export function normalizeTrainedCalibrationSamples(source, document) {
  object(document, 'calibration sample document');
  if (document.format !== LINEAR_SAMPLE_FORMAT || !Array.isArray(document.samples) ||
      document.samples.length === 0) {
    throw new Error(
      `calibration samples require format '${LINEAR_SAMPLE_FORMAT}' and a non-empty samples array.`,
    );
  }
  const inputTensors = [...source.graph.tensors.values()].filter((tensor) => tensor.isInput);
  const expectedNames = inputTensors.map((tensor) => tensor.name).sort();
  return document.samples.map((raw, sampleIndex) => {
    object(raw, `calibration sample ${sampleIndex}`);
    const actualNames = Object.keys(raw).sort();
    if (actualNames.length !== expectedNames.length ||
        actualNames.some((name, index) => name !== expectedNames[index])) {
      throw new Error(
        `calibration sample ${sampleIndex} must provide exactly: ${expectedNames.join(', ')}.`,
      );
    }
    return Object.fromEntries(inputTensors.map((tensor) => [
      tensor.name,
      typedValues(
        tensor.dtype,
        raw[tensor.name],
        elementCount(tensor.shape, `input '${tensor.name}'`),
        `calibration sample ${sampleIndex} input '${tensor.name}'`,
      ),
    ]));
  });
}

export function makeTrainedStructuralSample(source) {
  return Object.fromEntries(
    [...source.graph.tensors.values()]
      .filter((tensor) => tensor.isInput)
      .map((tensor) => [
        tensor.name,
        typedValues(
          tensor.dtype,
          new Array(elementCount(tensor.shape, `input '${tensor.name}'`)).fill(0),
          elementCount(tensor.shape, `input '${tensor.name}'`),
          `structural input '${tensor.name}'`,
        ),
      ]),
  );
}

function transposeInOutToOutIn(values, inputWidth, outputWidth) {
  if (!(values instanceof Float32Array) || values.length !== inputWidth * outputWidth) {
    throw new Error('IN_OUT transpose requires compatible F32 storage.');
  }
  const output = new Float32Array(values.length);
  for (let input = 0; input < inputWidth; input++) {
    for (let out = 0; out < outputWidth; out++) {
      output[out * inputWidth + input] = values[input * outputWidth + out];
    }
  }
  return output;
}

function packageFetch(graphDocument, weightsBuffer) {
  return async (url) => url === 'graph.json'
    ? { ok: true, json: async () => graphDocument }
    : url === 'model.safetensors'
      ? { ok: true, arrayBuffer: async () => weightsBuffer.slice(0) }
      : { ok: false, statusText: `unexpected generated package source ${url}` };
}

/**
 * Observe selected F32 values through the public retained runtime lifecycle.
 * The Model captures a graph snapshot while the selected values are declared
 * outputs; the caller's output selection is restored before execution starts.
 */
export async function calibrateRuntimeOutputs(
  graph,
  samples,
  tensorNames,
  { captureSampleIndex = null } = {},
) {
  if (!(graph instanceof Graph)) throw new Error('runtime calibration requires a Graph.');
  if (!Array.isArray(tensorNames) || tensorNames.length === 0) {
    throw new Error('runtime calibration requires at least one tensor name.');
  }
  if (!samples || (typeof samples[Symbol.iterator] !== 'function' &&
      typeof samples[Symbol.asyncIterator] !== 'function')) {
    throw new Error('runtime calibration samples must be an iterable or async iterable.');
  }
  for (const name of tensorNames) {
    const tensor = graph.getTensor(name);
    if (!tensor || tensor.dtype !== 'float32') {
      throw new Error(`runtime calibration tensor '${name}' must be an F32 graph tensor.`);
    }
  }
  if (captureSampleIndex != null &&
      (!Number.isSafeInteger(captureSampleIndex) || captureSampleIndex < 0)) {
    throw new Error('captureSampleIndex must be a non-negative safe integer or null.');
  }

  const originalOutputs = [...graph.outputNames];
  let outputsRestored = false;
  let runtime;
  let model;
  let compiled;
  let context;
  try {
    graph.setOutputs(tensorNames);
    runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
    model = runtime.createModel(graph);
    graph.setOutputs(originalOutputs);
    outputsRestored = true;

    compiled = await model.compile({
      backend: {
        mode: 'require',
        backend: 'cpu',
        operatorFallback: 'forbid',
      },
    });
    context = await compiled.createContext();

    const calibrator = new PTQCalibrator();
    let captured = null;
    let executionIndex = 0;
    for await (const sample of samples) {
      if (!sample || typeof sample !== 'object' || Array.isArray(sample)) {
        throw new Error(`runtime calibration sample ${executionIndex} must be a named input object.`);
      }
      const result = await context.execute(sample);
      try {
        const observations = [];
        for (const name of tensorNames) {
          const values = await result.output(name).read();
          if (!(values instanceof Float32Array)) {
            throw new Error(`runtime calibration tensor '${name}' did not return F32 values.`);
          }
          for (let index = 0; index < values.length; index++) {
            if (!Number.isFinite(values[index])) {
              throw new Error(
                `runtime calibration tensor '${name}' contains a non-finite value at index ${index}.`,
              );
            }
          }
          observations.push([name, values]);
        }
        for (const [name, values] of observations) calibrator.observe(name, values);
        if (executionIndex === captureSampleIndex) {
          captured = Object.freeze(Object.fromEntries(
            observations.map(([name, values]) => [name, new Float32Array(values)]),
          ));
        }
        executionIndex++;
      } finally {
        await result.close();
      }
    }
    if (executionIndex === 0) {
      throw new Error('runtime calibration received no samples.');
    }
    if (captureSampleIndex != null && captured == null) {
      throw new Error(`runtime calibration did not receive sample ${captureSampleIndex}.`);
    }
    return Object.freeze({ calibrator, captured });
  } finally {
    if (!outputsRestored) {
      try { graph.setOutputs(originalOutputs); } catch { /* Preserve the primary failure. */ }
    }
    await context?.close().catch(() => undefined);
    await compiled?.close().catch(() => undefined);
    await model?.close().catch(() => undefined);
    await runtime?.close().catch(() => undefined);
  }
}

async function validateIsland(graphDocument, weightsBuffer, firstSourceInput, inputParameters,
                              outputParameters, referenceOutput) {
  const graph = new Graph();
  await GraphLoader.load(graph, 'model.safetensors', {
    graphUrl: 'graph.json',
    fetch: packageFetch(graphDocument, weightsBuffer),
  });
  const packedInput = quantizePTQ(firstSourceInput, inputParameters);
  let runtime;
  let model;
  let compiled;
  let context;
  let result;
  let output;
  try {
    runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
    model = runtime.createModel(graph);
    compiled = await model.compile({
      backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
    });
    context = await compiled.createContext();
    result = await context.execute({ 'island.input': packedInput.data });
    output = await result.output('island.output').read();
  } finally {
    await result?.close().catch(() => undefined);
    await context?.close().catch(() => undefined);
    await compiled?.close().catch(() => undefined);
    await model?.close().catch(() => undefined);
    await runtime?.close().catch(() => undefined);
  }
  if (!(output instanceof Int8Array) || output.length !== referenceOutput.length) {
    throw new Error('generated QLinear island did not return canonical I8 output storage.');
  }
  let maximumError = 0;
  for (let index = 0; index < output.length; index++) {
    const dequantized = (output[index] - outputParameters.zero_point) * outputParameters.scale;
    maximumError = Math.max(maximumError, Math.abs(referenceOutput[index] - dequantized));
  }
  return Object.freeze({
    input_saturation_count: packedInput.saturationCount,
    max_abs_error_against_f32_island: maximumError,
  });
}

async function prepareOutputDirectory(outputDirectory, sourceDirectory) {
  const output = resolve(outputDirectory);
  if (output === resolve(sourceDirectory)) {
    throw new Error('output directory must not overwrite the trained source package.');
  }
  try {
    const info = await stat(output);
    if (!info.isDirectory() || (await readdir(output)).length !== 0) {
      throw new Error(`output directory '${output}' must be absent or empty.`);
    }
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    await mkdir(output, { recursive: true });
  }
  return output;
}

/**
 * Run the trained graph, calibrate one selected Linear boundary, and emit an
 * executable standalone QLinear package. `sampleDocument` must contain every
 * graph input. `structuralSmoke` substitutes one all-zero sample and is never
 * labelled representative calibration.
 */
export async function materializeTrainedLinearIsland({
  sourcePackage,
  sourceDirectory,
  weightName,
  sampleDocument = null,
  structuralSmoke = false,
  outputDirectory = null,
} = {}) {
  if ((sampleDocument == null) === !structuralSmoke) {
    throw new Error('provide exactly one of sampleDocument or structuralSmoke=true.');
  }
  let source = sourcePackage ?? await loadTrainedTinyReceiptPackage(sourceDirectory);
  source = source.graph ? source : await assembleTrainedSourceGraph(source);
  const mapping = linearMapping(source, weightName);
  const samples = structuralSmoke
    ? [makeTrainedStructuralSample(source)]
    : normalizeTrainedCalibrationSamples(source, sampleDocument);

  const { calibrator, captured } = await calibrateRuntimeOutputs(
    source.graph,
    samples,
    [mapping.source.input, mapping.source.output],
    { captureSampleIndex: 0 },
  );
  const parameters = calibrator.parameters({ dtype: 'int8', scheme: 'symmetric' });
  const inputParameters = parameters[mapping.source.input];
  const outputParameters = parameters[mapping.source.output];

  const canonical = new Graph();
  const [outputWidth, inputWidth] = mapping.target.weight_shape;
  canonical.addWeight(mapping.target.weight, mapping.target.weight_shape, 'float32', {
    buffer: transposeInOutToOutIn(mapping.sourceWeight.buffer, inputWidth, outputWidth),
  });
  canonical.addWeight(mapping.target.bias, [outputWidth], 'float32', {
    buffer: new Float32Array(mapping.sourceBias.buffer),
  });
  const artifact = materializePTQWeights(canonical, [{
    name: mapping.target.weight,
    outputName: mapping.target.weight,
    scaleName: mapping.target.weight_scale,
    axis: 0,
    bias: mapping.target.bias,
    biasOutputName: mapping.target.bias,
    inputScale: inputParameters.scale,
  }], {
    includeUnselected: false,
    metadata: {
      source_format: TRAINED_SOURCE_FORMAT,
      package_format: LINEAR_ISLAND_PACKAGE_FORMAT,
      source_weight: mapping.source.weight,
      source_weight_layout: SOURCE_WEIGHT_LAYOUT,
      target_weight_layout: TARGET_WEIGHT_LAYOUT,
    },
  });
  const inputScaleName = `${mapping.target.input}.scale`;
  const inputZeroPointName = `${mapping.target.input}.zero_point`;
  const outputScaleName = `${mapping.target.output}.scale`;
  const outputZeroPointName = `${mapping.target.output}.zero_point`;
  artifact.weights.addTensor(inputScaleName, 'F32', [1], Float32Array.of(inputParameters.scale));
  artifact.weights.addTensor(inputZeroPointName, 'I8', [1], Int8Array.of(inputParameters.zero_point));
  artifact.weights.addTensor(outputScaleName, 'F32', [1], Float32Array.of(outputParameters.scale));
  artifact.weights.addTensor(outputZeroPointName, 'I8', [1], Int8Array.of(outputParameters.zero_point));
  const weightsBuffer = artifact.weights.toArrayBuffer();
  const graphDocument = {
    format: SOURCE_GRAPH_FORMAT,
    inputs: {
      [mapping.target.input]: {
        shape: [...mapping.inputShape],
        dtype: 'int8',
      },
    },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        ...artifact.quantization.tensors,
        [mapping.target.input]: {
          scheme: 'per_tensor',
          scale_tensor: inputScaleName,
          zero_point_tensor: inputZeroPointName,
        },
        [mapping.target.output]: {
          scheme: 'per_tensor',
          scale_tensor: outputScaleName,
          zero_point_tensor: outputZeroPointName,
        },
      },
    },
    nodes: [{
      id: `ptq.${mapping.source.weight}`,
      opType: 'QLinear',
      inputs: {
        input: mapping.target.input,
        weight: mapping.target.weight,
        bias: mapping.target.bias,
      },
      outputs: { out: mapping.target.output },
      outputs_shape: { out: [...mapping.outputShape] },
      outputs_dtype: { out: 'int8' },
      params: {},
    }],
    outputs: [mapping.target.output],
  };

  const calibratedIslandInput = captured?.[mapping.source.input];
  if (!(calibratedIslandInput instanceof Float32Array)) {
    throw new Error(`calibrated Linear input '${mapping.source.input}' has no F32 result.`);
  }
  const referenceOutput = captured?.[mapping.source.output];
  if (!(referenceOutput instanceof Float32Array)) {
    throw new Error(`calibrated Linear output '${mapping.source.output}' has no F32 result.`);
  }
  const validation = await validateIsland(
    graphDocument,
    weightsBuffer,
    new Float32Array(calibratedIslandInput),
    inputParameters,
    outputParameters,
    new Float32Array(referenceOutput),
  );
  const calibration = {
    format: 'volvoxai-tiny-receipt-vqa-trained-linear-calibration-v1',
    representative: !structuralSmoke,
    source: structuralSmoke ? 'all_zero_structural_smoke' : LINEAR_SAMPLE_FORMAT,
    sample_count: samples.length,
    observed_tensors: {
      [mapping.source.input]: inputParameters,
      [mapping.source.output]: outputParameters,
    },
  };
  const manifest = {
    format: LINEAR_ISLAND_PACKAGE_FORMAT,
    scope: 'standalone_extracted_linear_operator',
    runnable_as_full_tiny_receipt_vqa: false,
    source: {
      format: TRAINED_SOURCE_FORMAT,
      directory_basename: basename(source.root),
      graph_format: SOURCE_GRAPH_FORMAT,
      model_metadata_format: SOURCE_MODEL_FORMAT,
      graph_sha256: source.hashes.graph_sha256,
      weights_sha256: source.hashes.weights_sha256,
    },
    mapping: {
      names: {
        input: [mapping.source.input, mapping.target.input],
        output: [mapping.source.output, mapping.target.output],
        weight: [mapping.source.weight, mapping.target.weight],
        bias: [mapping.source.bias, mapping.target.bias],
      },
      weight_layout: {
        source: SOURCE_WEIGHT_LAYOUT,
        source_shape: [...mapping.source.weight_shape],
        transform: 'transpose_2d',
        target: TARGET_WEIGHT_LAYOUT,
        target_shape: [...mapping.target.weight_shape],
        quantized_axis: 0,
      },
    },
    calibration,
    validation,
    files: {
      graph: 'graph.json',
      weights: 'model.safetensors',
      calibration: 'calibration.json',
    },
    limitations: [
      'The package starts at a calibrated intermediate Linear input, not a receipt image.',
      'It does not lower routing, MoELinear, CrossSDPA, residual/LoRA fusion, or autoregressive host policy.',
      structuralSmoke
        ? 'All-zero structural smoke calibration is not representative and must not be used for release accuracy claims.'
        : 'Calibration covers only the selected Linear input/output boundary.',
    ],
  };

  if (outputDirectory != null) {
    const output = await prepareOutputDirectory(outputDirectory, source.root);
    await Promise.all([
      writeFile(join(output, 'graph.json'), `${JSON.stringify(graphDocument, null, 2)}\n`),
      writeFile(join(output, 'model.safetensors'), new Uint8Array(weightsBuffer)),
      writeFile(join(output, 'calibration.json'), `${JSON.stringify(calibration, null, 2)}\n`),
      writeFile(join(output, 'island_manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`),
    ]);
  }
  return Object.freeze({ source, mapping, calibration, graphDocument, artifact, weightsBuffer, manifest });
}

function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    if (flag === '--structural-smoke') options.structuralSmoke = true;
    else if (['--source', '--weight', '--samples', '--out-dir'].includes(flag)) {
      if (index + 1 >= argv.length) throw new Error(`${flag} requires a value.`);
      options[{
        '--source': 'sourceDirectory',
        '--weight': 'weightName',
        '--samples': 'samplesPath',
        '--out-dir': 'outputDirectory',
      }[flag]] = argv[++index];
    } else {
      throw new Error(`unknown argument '${flag}'.`);
    }
  }
  for (const name of ['sourceDirectory', 'weightName', 'outputDirectory']) {
    if (!options[name]) throw new Error(`missing --${name === 'sourceDirectory' ? 'source' : name === 'weightName' ? 'weight' : 'out-dir'}.`);
  }
  if ((options.samplesPath == null) === !options.structuralSmoke) {
    throw new Error('provide exactly one of --samples or --structural-smoke.');
  }
  return options;
}

async function main(argv) {
  const options = parseArguments(argv);
  const sampleDocument = options.samplesPath == null
    ? null
    : JSON.parse(await readFile(options.samplesPath, 'utf8'));
  const result = await materializeTrainedLinearIsland({ ...options, sampleDocument });
  process.stdout.write(`${JSON.stringify({
    event: 'tiny_receipt_linear_ptq_materialized',
    format: result.manifest.format,
    representative_calibration: result.calibration.representative,
    source_weight: result.mapping.source.weight,
    output: resolve(options.outputDirectory),
  })}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`error: ${error.message}\n`);
    process.exitCode = 1;
  });
}
