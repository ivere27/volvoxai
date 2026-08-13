#!/usr/bin/env node

import {
  Model,
  VolvoxAI,
  parseGraphDocument,
} from '../ts/index.js';

const SCHEMA = 'volvoxai.padded-static-baseline/v1';

function integerArgument(name, fallback, minimum, maximum) {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  const value = raw == null ? fallback : Number(raw.slice(prefix.length));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function rejectUnknownArguments() {
  const known = new Set(['samples', 'warmup']);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
}

function percentile(values, fraction) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(ordered.length * fraction) - 1];
}

function deterministicValues(length, salt) {
  return Float32Array.from({ length }, (_, index) =>
    Math.fround((((index * 37 + salt * 17) % 257) - 128) / 257));
}

function linearGraph(shape, width, weightValues) {
  const weightShape = [width, width];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { input: { dtype: 'float32', shape } },
    nodes: [{
      id: 'linear',
      opType: 'Linear',
      inputs: { input: 'input', weight: 'weight' },
      outputs: {
        out: { tensor: 'out', dtype: 'float32', shape: [...shape.slice(0, -1), width] },
      },
      params: { weight_layout: 'din_dout' },
    }],
    outputs: ['out'],
  }, [{ name: 'weight', dtype: 'float32', shape: weightShape }]);
  return Model.capture({
    graph,
    weights: {
      weight: {
        name: 'weight', dtype: 'float32', shape: weightShape,
        data: new Float32Array(weightValues),
      },
    },
  });
}

function convGraph(height, width, channels, weightValues) {
  const inputShape = [1, height, width, channels];
  const weightShape = [3, 3, channels, channels];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { input: { dtype: 'float32', shape: inputShape } },
    nodes: [{
      id: 'conv',
      opType: 'Conv2D',
      inputs: { input: 'input', weight: 'weight' },
      outputs: {
        out: { tensor: 'out', dtype: 'float32', shape: inputShape },
      },
      params: {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 1,
        weight_layout: 'HWIO',
      },
    }],
    outputs: ['out'],
  }, [{ name: 'weight', dtype: 'float32', shape: weightShape }]);
  return Model.capture({
    graph,
    weights: {
      weight: {
        name: 'weight', dtype: 'float32', shape: weightShape,
        data: new Float32Array(weightValues),
      },
    },
  });
}

function padPrefix(active, paddedLength) {
  const padded = new Float32Array(paddedLength);
  padded.set(active);
  return padded;
}

function padSpatial(active, activeHeight, activeWidth, maximumHeight, maximumWidth, channels) {
  const padded = new Float32Array(maximumHeight * maximumWidth * channels);
  for (let row = 0; row < activeHeight; row++) {
    const sourceOffset = row * activeWidth * channels;
    const targetOffset = row * maximumWidth * channels;
    padded.set(active.subarray(sourceOffset, sourceOffset + activeWidth * channels), targetOffset);
  }
  return padded;
}

function extractSpatial(value, height, width, paddedWidth, channels) {
  const active = new Float32Array(height * width * channels);
  for (let row = 0; row < height; row++) {
    const sourceOffset = row * paddedWidth * channels;
    const targetOffset = row * width * channels;
    active.set(value.subarray(sourceOffset, sourceOffset + width * channels), targetOffset);
  }
  return active;
}

function parity(left, right) {
  if (left.length !== right.length) return { equalLength: false, maximumAbsoluteError: null };
  let maximumAbsoluteError = 0;
  for (let index = 0; index < left.length; index++) {
    maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(left[index] - right[index]));
  }
  return {
    equalLength: true,
    maximumAbsoluteError,
    pass: maximumAbsoluteError <= 1e-5,
  };
}

async function compileGraph(runtime, snapshot) {
  const compiled = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  return { compiled, context };
}

async function executeAndRead(context, input, shape) {
  const result = await context.execute({ input: { data: input, shape } });
  try {
    return await result.output('out').read();
  } finally {
    await result.close();
  }
}

async function timedExecution(context, input, shape) {
  const started = performance.now();
  const result = await context.execute({ input: { data: input, shape } });
  await result.close();
  return performance.now() - started;
}

async function measurePair(runtime, specification, samples, warmup) {
  const activeOwner = await compileGraph(runtime, specification.activeGraph);
  const paddedOwner = await compileGraph(runtime, specification.paddedGraph);
  try {
    for (let index = 0; index < warmup; index++) {
      await timedExecution(activeOwner.context, specification.activeInput, specification.activeShape);
      await timedExecution(paddedOwner.context, specification.paddedInput, specification.paddedShape);
    }
    const activeSamplesMs = [];
    const paddedSamplesMs = [];
    for (let index = 0; index < samples; index++) {
      if (index % 2 === 0) {
        activeSamplesMs.push(await timedExecution(
          activeOwner.context, specification.activeInput, specification.activeShape,
        ));
        paddedSamplesMs.push(await timedExecution(
          paddedOwner.context, specification.paddedInput, specification.paddedShape,
        ));
      } else {
        paddedSamplesMs.push(await timedExecution(
          paddedOwner.context, specification.paddedInput, specification.paddedShape,
        ));
        activeSamplesMs.push(await timedExecution(
          activeOwner.context, specification.activeInput, specification.activeShape,
        ));
      }
    }
    const activeOutput = await executeAndRead(
      activeOwner.context, specification.activeInput, specification.activeShape,
    );
    const paddedOutput = specification.extractPaddedOutput(
      await executeAndRead(
        paddedOwner.context, specification.paddedInput, specification.paddedShape,
      ),
    );
    const activeP50 = percentile(activeSamplesMs, 0.5);
    const paddedP50 = percentile(paddedSamplesMs, 0.5);
    return {
      name: specification.name,
      dimensions: specification.dimensions,
      activeLogicalInputBytes: specification.activeInput.byteLength,
      paddedLogicalInputBytes: specification.paddedInput.byteLength,
      paddedToActiveElementRatio:
        specification.paddedInput.length / specification.activeInput.length,
      active: {
        compileTimeMs: activeOwner.compiled.report.compileTimeMs,
        executionP50Ms: activeP50,
        executionP95Ms: percentile(activeSamplesMs, 0.95),
      },
      padded: {
        compileTimeMs: paddedOwner.compiled.report.compileTimeMs,
        executionP50Ms: paddedP50,
        executionP95Ms: percentile(paddedSamplesMs, 0.95),
      },
      paddedToActiveP50Ratio: paddedP50 / activeP50,
      activeRegionParity: parity(activeOutput, paddedOutput),
    };
  } finally {
    await activeOwner.context.close();
    await paddedOwner.context.close();
    await activeOwner.compiled.close();
    await paddedOwner.compiled.close();
  }
}

async function main() {
  rejectUnknownArguments();
  const samples = integerArgument('samples', 31, 5, 1000);
  const warmup = integerArgument('warmup', 5, 0, 1000);
  const linearWidth = 128;
  const linearWeight = deterministicValues(linearWidth * linearWidth, 1);
  const batchActive = deterministicValues(2 * linearWidth, 2);
  const sequenceActive = deterministicValues(1 * 64 * linearWidth, 3);
  const spatialChannels = 8;
  const spatialActiveHeight = 24;
  const spatialActiveWidth = 24;
  const spatialMaximumHeight = 72;
  const spatialMaximumWidth = 72;
  const spatialActive = deterministicValues(
    spatialActiveHeight * spatialActiveWidth * spatialChannels,
    4,
  );
  const convWeight = deterministicValues(3 * 3 * spatialChannels * spatialChannels, 5);
  const workloads = [
    {
      name: 'batch-linear',
      dimensions: { activeBatch: 2, maximumBatch: 16, feature: linearWidth },
      activeGraph: linearGraph([2, linearWidth], linearWidth, linearWeight),
      paddedGraph: linearGraph([16, linearWidth], linearWidth, linearWeight),
      activeInput: batchActive,
      activeShape: [2, linearWidth],
      paddedInput: padPrefix(batchActive, 16 * linearWidth),
      paddedShape: [16, linearWidth],
      extractPaddedOutput: (value) => value.slice(0, batchActive.length),
    },
    {
      name: 'sequence-linear',
      dimensions: { batch: 1, activeSequence: 64, maximumSequence: 512, feature: linearWidth },
      activeGraph: linearGraph([1, 64, linearWidth], linearWidth, linearWeight),
      paddedGraph: linearGraph([1, 512, linearWidth], linearWidth, linearWeight),
      activeInput: sequenceActive,
      activeShape: [1, 64, linearWidth],
      paddedInput: padPrefix(sequenceActive, 512 * linearWidth),
      paddedShape: [1, 512, linearWidth],
      extractPaddedOutput: (value) => value.slice(0, sequenceActive.length),
    },
    {
      name: 'spatial-conv2d',
      dimensions: {
        batch: 1,
        activeHeight: spatialActiveHeight,
        activeWidth: spatialActiveWidth,
        maximumHeight: spatialMaximumHeight,
        maximumWidth: spatialMaximumWidth,
        channels: spatialChannels,
      },
      activeGraph: convGraph(
        spatialActiveHeight,
        spatialActiveWidth,
        spatialChannels,
        convWeight,
      ),
      paddedGraph: convGraph(
        spatialMaximumHeight,
        spatialMaximumWidth,
        spatialChannels,
        convWeight,
      ),
      activeInput: spatialActive,
      activeShape: [1, spatialActiveHeight, spatialActiveWidth, spatialChannels],
      paddedInput: padSpatial(
        spatialActive,
        spatialActiveHeight,
        spatialActiveWidth,
        spatialMaximumHeight,
        spatialMaximumWidth,
        spatialChannels,
      ),
      paddedShape: [1, spatialMaximumHeight, spatialMaximumWidth, spatialChannels],
      extractPaddedOutput: (value) => extractSpatial(
        value,
        spatialActiveHeight,
        spatialActiveWidth,
        spatialMaximumWidth,
        spatialChannels,
      ),
    },
  ];

  const originalLog = console.log;
  console.log = (...values) => process.stderr.write(`${values.join(' ')}\n`);
  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  try {
    const results = [];
    for (const workload of workloads) {
      results.push(await measurePair(runtime, workload, samples, warmup));
    }
    originalLog(JSON.stringify({
      schema: SCHEMA,
      recordedAt: new Date().toISOString(),
      environment: {
        node: process.version,
        platform: process.platform,
        architecture: process.arch,
      },
      protocol: { backend: 'cpu', samples, warmup, order: 'alternating' },
      results,
    }, null, 2));
  } finally {
    await runtime.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
});
