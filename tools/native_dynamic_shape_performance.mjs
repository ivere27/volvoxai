#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const SCHEMA = 'volvoxai.native-dynamic-shape-performance/v1';

function stringArgument(name, fallback) {
  const prefix = `--${name}=`;
  const argument = process.argv.slice(2).find((value) => value.startsWith(prefix));
  const value = argument == null ? fallback : argument.slice(prefix.length);
  if (!value) throw new Error(`--${name} must not be empty`);
  return value;
}

function integerArgument(name, fallback, minimum, maximum) {
  const value = Number(stringArgument(name, String(fallback)));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function rejectUnknownArguments() {
  const known = new Set(['native-build-dir', 'active', 'maximum', 'warmup', 'samples']);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/u.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
}

function percentile(values, fraction) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(ordered.length * fraction) - 1];
}

function variance(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
}

function checkedSpawn(command, args, options, label) {
  const child = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 8 * 1024 * 1024,
    ...options,
  });
  if (child.error) throw child.error;
  if (child.status !== 0) {
    throw new Error(
      `${label} failed: ${child.stderr?.trim() || child.stdout?.trim() || `exit ${child.status}`}`,
    );
  }
  return child;
}

async function buildWorker(buildDirectory, temporaryDirectory) {
  const nativeBuildDirectory = path.join(buildDirectory, 'native');
  const linkFile = path.join(
    nativeBuildDirectory,
    'CMakeFiles',
    'volvoxai.dir',
    'link.txt',
  );
  let linkCommand;
  try {
    linkCommand = (await readFile(linkFile, 'utf8')).trim();
  } catch (error) {
    throw new Error(`cannot read '${linkFile}'; run 'make build_native' first: ${error.message}`);
  }
  const tokens = (linkCommand.match(/[^\s"']+|"[^"]*"|'[^']*'/gu) || []).map(
    (token) => (/^(["']).*\1$/u.test(token) ? token.slice(1, -1) : token),
  );
  const compiler = tokens.shift();
  if (!compiler || tokens.some((token) => /["']/u.test(token))) {
    throw new Error(`native CPU link command '${linkFile}' is unsupported`);
  }
  const releaseMapTokens = tokens.filter((token) =>
    /^-Wl,(?:-Map,|-Map=|--Map,|--Map=)/u.test(token));
  if (releaseMapTokens.length !== 1) {
    throw new Error(
      `native CPU link command '${linkFile}' must contain exactly one ` +
      'removable release map output',
    );
  }
  for (const releaseMapToken of releaseMapTokens) {
    tokens.splice(tokens.indexOf(releaseMapToken), 1);
  }
  const mainObjectIndex = tokens.findIndex((token) => token.endsWith('/cli/main.c.o'));
  const outputIndex = tokens.indexOf('-o');
  if (mainObjectIndex < 0 || outputIndex < 0 || outputIndex + 1 >= tokens.length) {
    throw new Error(`native CPU link command '${linkFile}' has no replaceable CLI entry`);
  }
  const source = path.join(ROOT, 'tools', 'native_dynamic_shape_performance.c');
  const object = path.join(temporaryDirectory, 'native_dynamic_shape_performance.o');
  const executable = path.join(temporaryDirectory, 'native_dynamic_shape_performance');
  checkedSpawn(compiler, [
    '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
    '-I', path.join(ROOT, 'native', 'include'),
    '-I', path.join(ROOT, 'native', 'src'),
    '-I', path.join(ROOT, 'native', 'src', 'runtime'),
    '-c', source, '-o', object,
  ], { cwd: ROOT }, 'native dynamic benchmark helper compilation');
  tokens.splice(mainObjectIndex, 1);
  const adjustedOutputIndex = tokens.indexOf('-o');
  tokens.splice(adjustedOutputIndex, 0, object);
  tokens[adjustedOutputIndex + 2] = executable;
  checkedSpawn(compiler, tokens, { cwd: nativeBuildDirectory },
    'native dynamic benchmark helper link');
  return { executable, compiler, linkFile };
}

function graphDocument(maximum, dynamic) {
  const extent = dynamic ? 'N' : maximum;
  return {
    format: 'volvox-graph/v1',
    dimensions: dynamic ? { N: { min: 2, max: maximum, multiple_of: 2 } } : {},
    inputs: {
      a: { shape: [extent], dtype: 'float32' },
      b: { shape: [extent], dtype: 'float32' },
    },
    nodes: [{
      id: 'add',
      opType: 'Add',
      inputs: { a: 'a', b: 'b' },
      outputs: { out: { tensor: 'sum', dtype: 'float32', shape: [extent] } },
      params: {},
    }],
    outputs: ['sum'],
  };
}

function parseWorker(child, samples) {
  const lines = child.stdout.trim().split(/\r?\n/u).filter(Boolean);
  if (lines.length > 1) process.stderr.write(`${lines.slice(0, -1).join('\n')}\n`);
  let evidence;
  try {
    evidence = JSON.parse(lines.at(-1));
  } catch {
    throw new Error('native dynamic benchmark worker returned invalid JSON');
  }
  if (evidence?.schema !== 'volvoxai.native-dynamic-shape-worker/v1' ||
      evidence?.warm?.dynamicActiveMs?.length !== samples ||
      evidence?.warm?.paddedMaximumMs?.length !== samples ||
      evidence.warm.dynamicActiveMs.some((value) => !Number.isFinite(value) || value < 0) ||
      evidence.warm.paddedMaximumMs.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error('native dynamic benchmark worker returned malformed evidence');
  }
  return evidence;
}

async function main() {
  rejectUnknownArguments();
  const active = integerArgument('active', 8_192, 2, 1_048_576);
  const maximum = integerArgument('maximum', 65_536, 64, 1_048_576);
  const warmup = integerArgument('warmup', 5, 0, 100);
  const samples = integerArgument('samples', 31, 3, 101);
  if ((active & 1) !== 0 || (maximum & 1) !== 0 || active > maximum) {
    throw new Error('--active and --maximum must be even, with active <= maximum');
  }
  const buildDirectory = path.resolve(stringArgument('native-build-dir', 'build/cmake'));
  const temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'volvox-native-dynamic-'));
  try {
    const dynamicGraph = path.join(temporaryDirectory, 'dynamic.graph.json');
    const paddedGraph = path.join(temporaryDirectory, 'padded.graph.json');
    await Promise.all([
      writeFile(dynamicGraph, `${JSON.stringify(graphDocument(maximum, true))}\n`, 'utf8'),
      writeFile(paddedGraph, `${JSON.stringify(graphDocument(maximum, false))}\n`, 'utf8'),
    ]);
    const worker = await buildWorker(buildDirectory, temporaryDirectory);
    const child = checkedSpawn(worker.executable, [
      dynamicGraph,
      paddedGraph,
      String(active),
      String(maximum),
      String(warmup),
      String(samples),
    ], { cwd: ROOT }, 'native dynamic benchmark worker');
    const raw = parseWorker(child, samples);
    const dynamicSamples = raw.warm.dynamicActiveMs;
    const paddedSamples = raw.warm.paddedMaximumMs;
    const activeLogicalBytes = raw.cold.active.logicalBytes;
    const maximumLogicalBytes = raw.cold.maximum.logicalBytes;
    const adversarialColdCount = raw.adversarial.filter(
      ({ sample }) => sample.plan === 'cold',
    ).length;
    const evidence = {
      schema: SCHEMA,
      recordedAt: new Date().toISOString(),
      environment: {
        runtime: 'native',
        nativeApiVersion: raw.nativeApiVersion,
        platform: process.platform,
        architecture: process.arch,
        backend: 'native-cpu',
        cpuThreads: 1,
        buildDirectory: path.relative(ROOT, buildDirectory) || '.',
        compiler: worker.compiler,
        linkDescription: path.relative(ROOT, worker.linkFile),
      },
      workload: raw.workload,
      compilation: raw.compile,
      coldSpecialization: raw.cold,
      warm: {
        dynamicActive: {
          samplesMs: dynamicSamples,
          p50Ms: percentile(dynamicSamples, 0.5),
          p95Ms: percentile(dynamicSamples, 0.95),
          varianceMs2: variance(dynamicSamples),
        },
        independentlyCompiledPaddedMaximum: {
          samplesMs: paddedSamples,
          p50Ms: percentile(paddedSamples, 0.5),
          p95Ms: percentile(paddedSamples, 0.95),
          varianceMs2: variance(paddedSamples),
        },
      },
      workAndCapacity: {
        activeLogicalBytes,
        maximumLogicalBytes,
        paddedToActiveLogicalByteRatio: maximumLogicalBytes / activeLogicalBytes,
        activeInitialCapacityBytes: raw.cold.active.arenaCapacity,
        capacityAfterMaximumBytes: raw.cold.maximum.arenaCapacity,
        finalHighWaterBytes: raw.adversarial.at(-1).sample.arenaHighWater,
        growCount: raw.adversarial.at(-1).sample.arenaGrows,
      },
      cache: {
        smallToLargeToSmall: raw.cold.returnActive.plan === 'hit',
        adversarial: raw.adversarial,
        coldCount: adversarialColdCount,
        deterministicEvictionObserved: raw.adversarial.at(-1).sample.plan === 'cold',
      },
      parity: raw.parity,
      memory: raw.memory,
      comparison: {
        paddedToDynamicP50Ratio:
          percentile(paddedSamples, 0.5) / percentile(dynamicSamples, 0.5),
        lowerLogicalWork: activeLogicalBytes < maximumLogicalBytes,
        lowerInitialCapacity:
          raw.cold.active.arenaCapacity < raw.cold.maximum.arenaCapacity,
      },
    };
    const valid = raw.cold.active.plan === 'cold' &&
      raw.cold.maximum.plan === 'cold' && raw.cold.returnActive.plan === 'hit' &&
      Object.values(raw.parity).every(Boolean) &&
      evidence.comparison.lowerLogicalWork && evidence.comparison.lowerInitialCapacity &&
      evidence.cache.deterministicEvictionObserved &&
      raw.cold.active.logicalBytes === active * Float32Array.BYTES_PER_ELEMENT * 3 &&
      raw.cold.maximum.logicalBytes === maximum * Float32Array.BYTES_PER_ELEMENT * 3;
    process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
    if (!valid) throw new Error('native dynamic-shape correctness/performance evidence failed');
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
