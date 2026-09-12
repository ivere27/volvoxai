#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import process from 'node:process';

const VX_DTYPE_U8 = 5;

function parsePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseArguments(argv) {
  let wasmPath = 'dist/0.4.0/volvoxai.wasm';
  let warmup = 20;
  let iterations = 100;
  let samples = 7;
  const cases = [];
  let pathSeen = false;
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === '--warmup') warmup = parsePositiveInteger(argv[++index], '--warmup');
    else if (argument === '--iterations') iterations = parsePositiveInteger(argv[++index], '--iterations');
    else if (argument === '--samples') samples = parsePositiveInteger(argv[++index], '--samples');
    else if (argument === '--case') {
      const specification = argv[++index] || '';
      const separator = specification.lastIndexOf(':');
      if (separator <= 0) throw new Error('--case must use name:elements');
      cases.push({
        name: specification.slice(0, separator),
        elements: parsePositiveInteger(specification.slice(separator + 1), '--case elements'),
      });
    }
    else if (!argument.startsWith('-') && !pathSeen) {
      wasmPath = argument;
      pathSeen = true;
    } else {
      throw new Error(`unknown argument: ${argument}`);
    }
  }
  if (cases.length === 0) {
    cases.push(
      { name: '64k', elements: 65_536 },
      { name: '384k', elements: 393_216 },
      { name: '1.25m', elements: 1_310_720 },
      { name: '2.5m', elements: 2_621_440 },
    );
  }
  return { wasmPath, warmup, iterations, samples, cases };
}

import { wasmToolImports } from './wasm_host_imports.mjs';

async function instantiate(path) {
  const bytes = await readFile(path);
  const { instance } = await WebAssembly.instantiate(bytes, wasmToolImports());
  return instance.exports;
}

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1] + ordered[middle]) / 2
    : ordered[middle];
}

function measure(call, warmup, iterations, samples) {
  for (let index = 0; index < warmup; index++) call();
  const durations = [];
  for (let sample = 0; sample < samples; sample++) {
    const begin = performance.now();
    for (let index = 0; index < iterations; index++) call();
    durations.push((performance.now() - begin) / iterations);
  }
  return median(durations);
}

function allocateHarness(api, maximumElements) {
  api.reset_heap();
  const input = api.alloc_bytes(maximumElements + 16);
  const scale = api.alloc_bytes(4);
  const zero = api.alloc_bytes(1);
  const output = api.alloc_bytes(maximumElements * 4 + 16);
  const requiredBytes = output + maximumElements * 4;
  const currentBytes = api.memory.buffer.byteLength;
  if (currentBytes < requiredBytes) {
    api.memory.grow(Math.ceil((requiredBytes - currentBytes) / 65_536));
  }
  const bytes = new Uint8Array(api.memory.buffer, input, maximumElements);
  for (let index = 0; index < bytes.length; index++) bytes[index] = (index * 73 + 151) & 255;
  new Float32Array(api.memory.buffer, scale, 1)[0] = 0.01953125;
  new Uint8Array(api.memory.buffer, zero, 1)[0] = 128;
  return { input, scale, zero, output };
}

const options = parseArguments(process.argv.slice(2));
const api = await instantiate(options.wasmPath);
if (typeof api.dequantize_linear_typed !== 'function') {
  throw new Error(`${options.wasmPath} does not export dequantize_linear_typed`);
}

const cases = options.cases;
const harness = allocateHarness(api, Math.max(...cases.map((entry) => entry.elements)));
const results = [];

for (const entry of cases) {
  const canonicalCall = () => {
    const status = api.dequantize_linear_typed(
      harness.input, VX_DTYPE_U8, harness.scale, harness.zero, VX_DTYPE_U8,
      harness.output, entry.elements,
    );
    if (status !== 1) throw new Error(`canonical kernel rejected ${entry.name}`);
  };
  const canonicalMs = measure(canonicalCall, options.warmup, options.iterations, options.samples);
  results.push({
    name: entry.name,
    elements: entry.elements,
    canonical_ms: canonicalMs,
    canonical_gigaelements_per_second: entry.elements / canonicalMs / 1e6,
    canonical_effective_gigabytes_per_second: entry.elements * 5 / canonicalMs / 1e6,
  });
}

console.log(JSON.stringify({
  wasm: options.wasmPath,
  warmup: options.warmup,
  iterations: options.iterations,
  samples: options.samples,
  results,
}, null, 2));
