#!/usr/bin/env node

import path from 'node:path';
import process from 'node:process';

import { fixture, p, ok, tensors, safetensors } from './proto_fixture.mjs';

const SCHEMA = 'volvoxai.dynamic-shape-performance/v1';

function argument(name, fallback) {
  const prefix = `--${name}=`;
  const match = process.argv.slice(2).find((value) => value.startsWith(prefix));
  return match === undefined ? fallback : match.slice(prefix.length);
}

function integerArgument(name, fallback, minimum, maximum) {
  const value = Number(argument(name, String(fallback)));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function rejectUnknownArguments() {
  const known = new Set(['backend', 'wasm', 'samples', 'warmup']);
  for (const value of process.argv.slice(2)) {
    const match = /^--([^=]+)=/.exec(value);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${value}'`);
  }
}

function percentile(values, fraction) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.max(0, Math.ceil(ordered.length * fraction) - 1)];
}

function variance(values) {
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  return values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length;
}

function summarize(values) {
  return Object.freeze({
    samples: values.length,
    p50Ms: percentile(values, 0.5),
    p95Ms: percentile(values, 0.95),
    varianceMs2: variance(values),
    minimumMs: Math.min(...values),
    maximumMs: Math.max(...values),
  });
}

function deterministic(length, salt) {
  return Float32Array.from({ length }, (_, index) =>
    Math.fround((((index * 37 + salt * 17) % 257) - 128) / 257));
}

function snapshot(_authoring, document, weights = []) {
  return { document, weights: safetensors(weights) };
}

function linearSnapshot(
  authoring,
  { prefix, symbol = null, minimum = 1, maximum = 1, width, weight },
) {
  const logicalPrefix = prefix.map((dimension) => dimension === '$' ? symbol : dimension);
  const dimensions = symbol === null ? {} : { [symbol]: { min: minimum, max: maximum } };
  const shape = [...logicalPrefix, width];
  return snapshot(authoring, {
    format: 'volvox-graph/v1',
    dimensions,
    inputs: { input: { dtype: 'float32', shape } },
    nodes: [{
      id: 'linear', opType: 'Linear',
      inputs: { input: 'input', weight: 'weight' },
      outputs: { out: { tensor: 'out', dtype: 'float32', shape } },
      params: { weight_layout: 'din_dout' },
    }],
    outputs: ['out'],
  }, [{ name: 'weight', dtype: 'float32', shape: [width, width], data: weight }]);
}

function convSnapshot(
  authoring,
  { height, width, dynamic, minimum, maximum, channels, weight },
) {
  const dimensions = dynamic
    ? { H: { min: minimum, max: maximum }, W: { min: minimum, max: maximum } }
    : {};
  const shape = [1, dynamic ? 'H' : height, dynamic ? 'W' : width, channels];
  return snapshot(authoring, {
    format: 'volvox-graph/v1',
    dimensions,
    inputs: { input: { dtype: 'float32', shape } },
    nodes: [{
      id: 'conv', opType: 'Conv2D',
      inputs: { input: 'input', weight: 'weight' },
      outputs: { out: { tensor: 'out', dtype: 'float32', shape } },
      params: {
        stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1,
        data_layout: 'NHWC', weight_layout: 'HWIO',
      },
    }],
    outputs: ['out'],
  }, [{
    name: 'weight', dtype: 'float32', shape: [3, 3, channels, channels], data: weight,
  }]);
}

function addSnapshot({
  authoring,
  batch,
  sequence,
  dynamic,
  maximumBatch,
  maximumSequence,
  width,
}) {
  const dimensions = dynamic
    ? { B: { min: 1, max: maximumBatch }, S: { min: 1, max: maximumSequence } }
    : {};
  const shape = dynamic ? ['B', 'S', width] : [batch, sequence, width];
  return snapshot(authoring, {
    format: 'volvox-graph/v1',
    dimensions,
    inputs: {
      left: { dtype: 'float32', shape },
      right: { dtype: 'float32', shape },
    },
    nodes: [{
      id: 'add', opType: 'Add', inputs: { a: 'left', b: 'right' },
      outputs: { out: { tensor: 'out', dtype: 'float32', shape } }, params: {},
    }],
    outputs: ['out'],
  });
}

function padPrefix(value, length) {
  const result = new Float32Array(length);
  result.set(value);
  return result;
}

function padSpatial(value, activeHeight, activeWidth, maximumHeight, maximumWidth, channels) {
  const result = new Float32Array(maximumHeight * maximumWidth * channels);
  for (let row = 0; row < activeHeight; row++) {
    result.set(
      value.subarray(row * activeWidth * channels, (row + 1) * activeWidth * channels),
      row * maximumWidth * channels,
    );
  }
  return result;
}

function padBatchSequence(value, activeBatch, activeSequence, maximumSequence, width) {
  const result = new Float32Array(activeBatch * maximumSequence * width);
  for (let batch = 0; batch < activeBatch; batch++) {
    const source = batch * activeSequence * width;
    const target = batch * maximumSequence * width;
    result.set(value.subarray(source, source + activeSequence * width), target);
  }
  return result;
}

function extractSpatial(value, height, width, sourceWidth, channels) {
  const result = new Float32Array(height * width * channels);
  for (let row = 0; row < height; row++) {
    result.set(
      value.subarray(row * sourceWidth * channels, (row * sourceWidth + width) * channels),
      row * width * channels,
    );
  }
  return result;
}

function maximumAbsoluteError(left, right) {
  if (left.length !== right.length) return null;
  let maximum = 0;
  for (let index = 0; index < left.length; index++) {
    maximum = Math.max(maximum, Math.abs(left[index] - right[index]));
  }
  return maximum;
}

async function compile(f, snapshot, backend) {
  const model = await f.load(snapshot.document, snapshot.weights);
  const start = performance.now();
  const compiled = await f.compile(model, backend);
  ok(await f.inference.releaseModel(new p.ModelRef(model)));
  return {compiled, wallTimeMs: performance.now() - start};
}
async function execute(f, context, inputs, {read = false} = {}) {
  const start = performance.now();
  const result = ok(await f.inference.execute(new p.ExecuteRequest({contextId: context.contextId, inputs: tensors(inputs)})));
  const wallTimeMs = performance.now() - start;
  try {
    return {wallTimeMs, report: result.report, metrics: result.metrics, output: read ? await f.read(result, 'out') : null};
  } finally { ok(await f.inference.releaseResult(new p.ResultRef(result))); }
}
function telemetry(observation) {
  const {report, metrics} = observation;
  return {metrics, shapePlan: report.route?.shapePlan, route: report.route};
}

async function measure(runtime, backend, workload, samples, warmup) {
  const dynamicOwner = await compile(runtime, workload.dynamicSnapshot, backend);
  const activeOwner = await compile(runtime, workload.activeSnapshot, backend);
  const paddedOwner = await compile(runtime, workload.paddedSnapshot, backend);
  const dynamicContext =  ok(await runtime.inference.createExecutionContext(new p.CreateExecutionContextRequest(dynamicOwner.compiled)));
  const activeContext =  ok(await runtime.inference.createExecutionContext(new p.CreateExecutionContextRequest(activeOwner.compiled)));
  const paddedContext =  ok(await runtime.inference.createExecutionContext(new p.CreateExecutionContextRequest(paddedOwner.compiled)));
  try {
    const coldActive = await execute(runtime, dynamicContext, workload.activeInputs, { read: true });
    const coldPadded = await execute(runtime, dynamicContext, workload.paddedInputs, { read: true });
    const staticActive = await execute(runtime, activeContext, workload.activeInputs, { read: true });
    const staticPadded = await execute(runtime, paddedContext, workload.paddedInputs, { read: true });
    const extractedPadded = workload.extractPadded(coldPadded.output);
    const dynamicActiveError = maximumAbsoluteError(coldActive.output, staticActive.output);
    const dynamicPaddedError = maximumAbsoluteError(coldPadded.output, staticPadded.output);
    const paddedActiveError = maximumAbsoluteError(extractedPadded, staticActive.output);
    if (dynamicActiveError === null || dynamicActiveError > 1e-5 ||
        dynamicPaddedError === null || dynamicPaddedError > 1e-5 ||
        paddedActiveError === null || paddedActiveError > 1e-5) {
      throw new Error(`${workload.name} numerical parity failed`);
    }

    for (let index = 0; index < warmup; index++) {
      await execute(runtime, dynamicContext, workload.activeInputs);
      await execute(runtime, activeContext, workload.activeInputs);
      await execute(runtime, paddedContext, workload.paddedInputs);
    }
    const warmDynamic = [];
    const warmStatic = [];
    const warmPadded = [];
    for (let index = 0; index < samples; index++) {
      warmDynamic.push((await execute(runtime, dynamicContext, workload.activeInputs)).wallTimeMs);
      warmStatic.push((await execute(runtime, activeContext, workload.activeInputs)).wallTimeMs);
      warmPadded.push((await execute(runtime, paddedContext, workload.paddedInputs)).wallTimeMs);
    }
    const alternating = [];
    for (let index = 0; index < samples; index++) {
      alternating.push((await execute(runtime,
        dynamicContext,
        index % 2 === 0 ? workload.activeInputs : workload.paddedInputs,
      )).wallTimeMs);
    }
    const adversarial = [];
    let finalAdversarialReport = coldPadded;
    for (const inputs of workload.adversarialInputs) {
      const observation = await execute(runtime, dynamicContext, inputs);
      adversarial.push(observation.wallTimeMs);
      finalAdversarialReport = observation;
    }
    return Object.freeze({
      name: workload.name,
      dimensions: workload.dimensions,
      logicalInputBytes: Object.freeze({
        active: workload.inputBytes(workload.activeInputs),
        padded: workload.inputBytes(workload.paddedInputs),
      }),
      compile: Object.freeze({
        dynamicWallTimeMs: dynamicOwner.wallTimeMs,
        dynamicReportedTimeMs: Number(dynamicOwner.compiled.compileTimeNs) / 1e6,
        activeStaticWallTimeMs: activeOwner.wallTimeMs,
        paddedStaticWallTimeMs: paddedOwner.wallTimeMs,
      }),
      coldSpecialization: Object.freeze({
        active: telemetry(coldActive),
        padded: telemetry(coldPadded),
      }),
      warm: Object.freeze({
        dynamicActive: summarize(warmDynamic),
        staticActive: summarize(warmStatic),
        paddedMaximum: summarize(warmPadded),
        paddedToDynamicP50Ratio:
          percentile(warmPadded, 0.5) / percentile(warmDynamic, 0.5),
      }),
      alternating: summarize(alternating),
      adversarial: Object.freeze({
        ...summarize(adversarial),
        signatureCount: workload.adversarialInputs.length,
        finalTelemetry: telemetry(finalAdversarialReport),
      }),
      parity: Object.freeze({
        dynamicActiveMaximumAbsoluteError: dynamicActiveError,
        dynamicPaddedMaximumAbsoluteError: dynamicPaddedError,
        paddedActiveRegionMaximumAbsoluteError: paddedActiveError,
      }),
    });
  } finally {
    for (const context of [dynamicContext, activeContext, paddedContext])
      ok(await runtime.inference.releaseExecutionContext(new p.ExecutionContextRef(context)));
    for (const owner of [dynamicOwner, activeOwner, paddedOwner])
      ok(await runtime.inference.releaseCompiledModel(new p.CompiledModelRef(owner.compiled)));
  }
}

function workloads() {
  const authoring = null;
  const linearWidth = 64;
  const linearWeight = deterministic(linearWidth * linearWidth, 1);
  const batchActive = deterministic(2 * linearWidth, 2);
  const batchPadded = padPrefix(batchActive, 16 * linearWidth);
  const sequenceActive = deterministic(1 * 32 * linearWidth, 3);
  const sequencePadded = padPrefix(sequenceActive, 1 * 256 * linearWidth);
  const channels = 4;
  const convActive = deterministic(16 * 16 * channels, 4);
  const convPadded = padSpatial(convActive, 16, 16, 48, 48, channels);
  const convWeight = deterministic(3 * 3 * channels * channels, 5);
  const addWidth = 32;
  const addLeft = deterministic(2 * 8 * addWidth, 6);
  const addRight = deterministic(2 * 8 * addWidth, 7);
  const addPaddedLeft = new Float32Array(8 * 32 * addWidth);
  addPaddedLeft.set(padBatchSequence(addLeft, 2, 8, 32, addWidth));
  const addPaddedRight = new Float32Array(8 * 32 * addWidth);
  addPaddedRight.set(padBatchSequence(addRight, 2, 8, 32, addWidth));
  const shaped = (name, data, shape) => ({ [name]: { data, shape } });
  const twoInputs = (left, right, shape) => ({
    left: { data: left, shape }, right: { data: right, shape },
  });
  return [
    {
      name: 'batch-linear',
      dimensions: { B: { active: 2, maximum: 16 }, feature: linearWidth },
      dynamicSnapshot: linearSnapshot(authoring, {
        prefix: ['$'], symbol: 'B', minimum: 1, maximum: 16,
        width: linearWidth, weight: linearWeight,
      }),
      activeSnapshot: linearSnapshot(authoring, {
        prefix: [2], width: linearWidth, weight: linearWeight,
      }),
      paddedSnapshot: linearSnapshot(authoring, {
        prefix: [16], width: linearWidth, weight: linearWeight,
      }),
      activeInputs: shaped('input', batchActive, [2, linearWidth]),
      paddedInputs: shaped('input', batchPadded, [16, linearWidth]),
      adversarialInputs: Array.from({ length: 12 }, (_, index) => {
        const batch = index + 1;
        return shaped('input', deterministic(batch * linearWidth, 20 + index), [batch, linearWidth]);
      }),
      extractPadded: (value) => value.slice(0, batchActive.length),
      inputBytes: (inputs) => inputs.input.data.byteLength,
    },
    {
      name: 'sequence-linear',
      dimensions: { B: 1, S: { active: 32, maximum: 256 }, feature: linearWidth },
      dynamicSnapshot: linearSnapshot(authoring, {
        prefix: [1, '$'], symbol: 'S', minimum: 1, maximum: 256,
        width: linearWidth, weight: linearWeight,
      }),
      activeSnapshot: linearSnapshot(authoring, {
        prefix: [1, 32], width: linearWidth, weight: linearWeight,
      }),
      paddedSnapshot: linearSnapshot(authoring, {
        prefix: [1, 256], width: linearWidth, weight: linearWeight,
      }),
      activeInputs: shaped('input', sequenceActive, [1, 32, linearWidth]),
      paddedInputs: shaped('input', sequencePadded, [1, 256, linearWidth]),
      adversarialInputs: [1, 7, 15, 31, 63, 127, 255, 2, 9, 33, 129, 256].map((sequence, index) =>
        shaped('input', deterministic(sequence * linearWidth, 40 + index),
          [1, sequence, linearWidth])),
      extractPadded: (value) => value.slice(0, sequenceActive.length),
      inputBytes: (inputs) => inputs.input.data.byteLength,
    },
    {
      name: 'spatial-conv2d',
      dimensions: { B: 1, H: { active: 16, maximum: 48 }, W: { active: 16, maximum: 48 }, channels },
      dynamicSnapshot: convSnapshot(authoring, {
        dynamic: true, minimum: 8, maximum: 48, channels, weight: convWeight,
      }),
      activeSnapshot: convSnapshot(authoring, {
        dynamic: false, height: 16, width: 16, channels, weight: convWeight,
      }),
      paddedSnapshot: convSnapshot(authoring, {
        dynamic: false, height: 48, width: 48, channels, weight: convWeight,
      }),
      activeInputs: shaped('input', convActive, [1, 16, 16, channels]),
      paddedInputs: shaped('input', convPadded, [1, 48, 48, channels]),
      adversarialInputs: [[8, 8], [9, 17], [16, 16], [17, 31], [24, 48], [31, 9],
        [32, 32], [47, 47], [48, 48], [8, 48], [48, 8], [25, 33]].map(([height, width], index) =>
        shaped('input', deterministic(height * width * channels, 60 + index),
          [1, height, width, channels])),
      extractPadded: (value) => extractSpatial(value, 16, 16, 48, channels),
      inputBytes: (inputs) => inputs.input.data.byteLength,
    },
    {
      name: 'multi-input-exact-add',
      dimensions: { B: { active: 2, maximum: 8 }, S: { active: 8, maximum: 32 }, feature: addWidth },
      dynamicSnapshot: addSnapshot({
        authoring,
        dynamic: true, maximumBatch: 8, maximumSequence: 32, width: addWidth,
      }),
      activeSnapshot: addSnapshot({
        authoring, batch: 2, sequence: 8, dynamic: false, width: addWidth,
      }),
      paddedSnapshot: addSnapshot({
        authoring, batch: 8, sequence: 32, dynamic: false, width: addWidth,
      }),
      activeInputs: twoInputs(addLeft, addRight, [2, 8, addWidth]),
      paddedInputs: twoInputs(addPaddedLeft, addPaddedRight, [8, 32, addWidth]),
      adversarialInputs: [[1, 1], [1, 32], [2, 8], [3, 7], [4, 16], [5, 9],
        [6, 24], [7, 31], [8, 32], [8, 2], [8, 16], [8, 1]].map(([batch, sequence], index) => {
        const length = batch * sequence * addWidth;
        return twoInputs(deterministic(length, 80 + index), deterministic(length, 100 + index),
          [batch, sequence, addWidth]);
      }),
      extractPadded: (value) => {
        const result = new Float32Array(addLeft.length);
        for (let batch = 0; batch < 2; batch++) {
          for (let sequence = 0; sequence < 8; sequence++) {
            const source = (batch * 32 + sequence) * addWidth;
            const target = (batch * 8 + sequence) * addWidth;
            result.set(value.subarray(source, source + addWidth), target);
          }
        }
        return result;
      },
      inputBytes: (inputs) => inputs.left.data.byteLength + inputs.right.data.byteLength,
    },
  ];
}

async function main() {
  rejectUnknownArguments();
  const backend = argument('backend', 'wasm');
  if (backend !== 'wasm') {
    throw new Error("--backend must be 'wasm'; use the browser WebGPU harness for webgpu");
  }
  const samples = integerArgument('samples', 15, 3, 1001);
  const warmup = integerArgument('warmup', 3, 0, 1000);
  const wasmUrl = path.resolve(argument('wasm', 'dist/0.6.0/volvoxai.wasm'));
  const originalLog = console.log;
  console.log = (...values) => process.stderr.write(`${values.map(String).join(' ')}\n`);
  const runtime = await fixture({wasmUrl});
  try {
    const results = [];
    for (const workload of workloads()) {
      results.push(await measure(runtime, backend, workload, samples, warmup));
    }
    originalLog(JSON.stringify({
      schema: SCHEMA,
      recordedAt: new Date().toISOString(),
      environment: {
        runtime: 'node', version: process.version,
        platform: process.platform, architecture: process.arch, backend,
      },
      protocol: {
        samples, warmup,
        timing: 'generated proto Execute wall time; C timing and shape evidence reported separately',
        workloads: 'constant active, padded maximum, polymorphic cold/warm/alternating/adversarial',
      },
      results,
    }, (_key, value) => typeof value === 'bigint' ? value.toString() : value, 2));
  } finally {
    await runtime.close();
    console.log = originalLog;
  }
}

main().catch((error) => {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
});
