#!/usr/bin/env node

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

const encoderCases = [
  { name: '160x336 C48 G16', height: 160, width: 336, channels: 48, groups: 16, count: 1 },
  { name: '80x168 C96 G32', height: 80, width: 168, channels: 96, groups: 32, count: 3 },
  { name: '40x84 C192 G32', height: 40, width: 84, channels: 192, groups: 32, count: 3 },
  { name: '20x42 C320 G32', height: 20, width: 42, channels: 320, groups: 32, count: 3 },
  { name: '10x21 C320 G32', height: 10, width: 21, channels: 320, groups: 32, count: 3 },
];

function positiveInteger(argument, fallback, label) {
  const value = argument == null ? fallback : Number(argument);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return value;
}

const warmup = positiveInteger(process.env.GROUPNORM_WARMUP, 8, 'GROUPNORM_WARMUP');
const iterations = positiveInteger(process.env.GROUPNORM_ITERATIONS, 20, 'GROUPNORM_ITERATIONS');
const samples = positiveInteger(process.env.GROUPNORM_SAMPLES, 7, 'GROUPNORM_SAMPLES');
const spatialTile = positiveInteger(process.env.GROUPNORM_TILE, 16, 'GROUPNORM_TILE');

async function build(directory, name, definitions) {
  const output = join(directory, `${name}.wasm`);
  await run(clang, [
    '--target=wasm32', '-std=c11', '-O3', '-msimd128', '-nostdlib',
    ...definitions,
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot, maxBuffer: 1024 * 1024 });
  return output;
}

import { wasmToolImports } from './wasm_host_imports.mjs';

async function instantiate(path) {
  const bytes = await readFile(path);
  const { instance } = await WebAssembly.instantiate(bytes, wasmToolImports());
  return instance.exports;
}

function allocate(api, bytes) {
  const pointer = Number(api.alloc_bytes(bytes));
  assert.ok(Number.isSafeInteger(pointer) && pointer > 0);
  const required = pointer + bytes;
  if (required > api.memory.buffer.byteLength) {
    api.memory.grow(Math.ceil((required - api.memory.buffer.byteLength) / 65_536));
  }
  return pointer;
}

function prepare(api, specification) {
  api.reset_heap();
  const elements = specification.height * specification.width * specification.channels;
  const input = allocate(api, elements * Float32Array.BYTES_PER_ELEMENT);
  const weight = allocate(api, specification.channels * Float32Array.BYTES_PER_ELEMENT);
  const bias = allocate(api, specification.channels * Float32Array.BYTES_PER_ELEMENT);
  const output = allocate(api, elements * Float32Array.BYTES_PER_ELEMENT);
  const inputValues = new Float32Array(api.memory.buffer, input, elements);
  for (let index = 0; index < elements; index++) {
    inputValues[index] = Math.fround(((index * 37 + 19) % 509 - 254) / 31);
  }
  const weightValues = new Float32Array(
    api.memory.buffer, weight, specification.channels,
  );
  const biasValues = new Float32Array(
    api.memory.buffer, bias, specification.channels,
  );
  for (let channel = 0; channel < specification.channels; channel++) {
    weightValues[channel] = Math.fround(((channel * 13 + 7) % 43 - 21) / 19);
    biasValues[channel] = Math.fround(((channel * 11 + 5) % 31 - 15) / 17);
  }
  const call = () => api.groupnorm_f32(
    input, weight, bias, output, 1,
    specification.height, specification.width,
    specification.channels, specification.groups, 1e-5,
  );
  return { call, output, elements };
}

function medianMilliseconds(call) {
  for (let index = 0; index < warmup; index++) assert.equal(call(), 1);
  const timings = [];
  for (let sample = 0; sample < samples; sample++) {
    const begin = performance.now();
    for (let index = 0; index < iterations; index++) assert.equal(call(), 1);
    timings.push((performance.now() - begin) / iterations);
  }
  timings.sort((left, right) => left - right);
  return timings[Math.floor(timings.length / 2)];
}

const directory = await mkdtemp(join(tmpdir(), 'volvoxai-groupnorm-benchmark-'));
try {
  const [scalarPath, simdPath] = await Promise.all([
    build(directory, 'scalar', ['-DVOLVOXAI_DISABLE_GROUPNORM_WASM_SIMD=1']),
    build(directory, 'simd', [
      `-DVOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE=${spatialTile}`,
    ]),
  ]);
  const [scalarApi, simdApi] = await Promise.all([
    instantiate(scalarPath), instantiate(simdPath),
  ]);
  const results = [];
  for (const specification of encoderCases) {
    const scalar = prepare(scalarApi, specification);
    const simd = prepare(simdApi, specification);
    assert.equal(scalar.call(), 1);
    assert.equal(simd.call(), 1);
    assert.deepEqual(
      new Uint8Array(
        simdApi.memory.buffer, simd.output,
        simd.elements * Float32Array.BYTES_PER_ELEMENT,
      ).slice(),
      new Uint8Array(
        scalarApi.memory.buffer, scalar.output,
        scalar.elements * Float32Array.BYTES_PER_ELEMENT,
      ).slice(),
      `${specification.name} must bit-match the scalar kernel`,
    );
    const scalarMs = medianMilliseconds(scalar.call);
    const simdMs = medianMilliseconds(simd.call);
    results.push({
      ...specification,
      scalar_ms: scalarMs,
      simd_ms: simdMs,
      speedup: scalarMs / simdMs,
      reduction_percent: (1 - simdMs / scalarMs) * 100,
    });
  }
  const scalarWeightedMs = results.reduce(
    (total, result) => total + result.count * result.scalar_ms, 0,
  );
  const simdWeightedMs = results.reduce(
    (total, result) => total + result.count * result.simd_ms, 0,
  );
  process.stdout.write(`${JSON.stringify({
    node: process.version,
    spatial_tile: spatialTile,
    warmup,
    iterations,
    samples,
    results,
    weighted_13_node_aggregate: {
      scalar_ms: scalarWeightedMs,
      simd_ms: simdWeightedMs,
      speedup: scalarWeightedMs / simdWeightedMs,
      reduction_percent: (1 - simdWeightedMs / scalarWeightedMs) * 100,
    },
  }, null, 2)}\n`);
} finally {
  await rm(directory, { recursive: true, force: true });
}
