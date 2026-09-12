#!/usr/bin/env node

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
const sizes = [69_760, 380_192, 1_290_240, 2_580_480];

async function build(directory, name, definitions = []) {
  const output = join(directory, `${name}.wasm`);
  await run(clang, [
    '--target=wasm32', '-std=c11', '-O3', '-msimd128', '-nostdlib',
    ...definitions,
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

import { wasmToolImports } from './wasm_host_imports.mjs';

async function instantiate(path) {
  const bytes = await readFile(path);
  const { instance } = await WebAssembly.instantiate(bytes, wasmToolImports());
  return instance.exports;
}

function prepare(api, elements) {
  api.reset_heap();
  const input = Number(api.alloc_bytes(elements * 4));
  const output = Number(api.alloc_bytes(elements * 4));
  const requiredBytes = output + elements * 4;
  if (api.memory.buffer.byteLength < requiredBytes) {
    api.memory.grow(Math.ceil(
      (requiredBytes - api.memory.buffer.byteLength) / 65_536,
    ));
  }
  const values = new Float32Array(api.memory.buffer, input, elements);
  for (let index = 0; index < elements; index++) {
    values[index] = Math.fround(((index * 37) % 257 - 128) / 13);
  }
  return () => api.silu_f32(input, output, elements);
}

function measure(call, warmup = 20, iterations = 100, samples = 7) {
  for (let iteration = 0; iteration < warmup; iteration++) call();
  const timings = [];
  for (let sample = 0; sample < samples; sample++) {
    const start = performance.now();
    for (let iteration = 0; iteration < iterations; iteration++) call();
    timings.push((performance.now() - start) / iterations);
  }
  timings.sort((left, right) => left - right);
  return timings[Math.floor(timings.length / 2)];
}

const directory = await mkdtemp(join(tmpdir(), 'volvoxai-silu-benchmark-'));
try {
  const [scalarPath, simdPath] = await Promise.all([
    build(directory, 'scalar', ['-DVOLVOXAI_DISABLE_SILU_WASM_SIMD=1']),
    build(directory, 'simd'),
  ]);
  const [scalarApi, simdApi] = await Promise.all([
    instantiate(scalarPath),
    instantiate(simdPath),
  ]);
  for (const elements of sizes) {
    const scalar = measure(prepare(scalarApi, elements));
    const simd = measure(prepare(simdApi, elements));
    process.stdout.write(`${JSON.stringify({
      elements,
      scalar_ms: scalar,
      simd_ms: simd,
      speedup: scalar / simd,
      reduction_percent: (1 - simd / scalar) * 100,
    })}\n`);
  }
} finally {
  await rm(directory, { recursive: true, force: true });
}
