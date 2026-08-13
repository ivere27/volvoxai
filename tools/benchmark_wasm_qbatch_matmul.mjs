import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { performance } from 'node:perf_hooks';

const artifactPath = process.argv[2];
if (!artifactPath || process.argv.length !== 3) {
  throw new Error(
    'usage: node tools/benchmark_wasm_qbatch_matmul.mjs <volvoxai.wasm>',
  );
}

// Canonical protobuf DataType values used by the public native/WASM ABI.
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;

const module = await WebAssembly.compile(await readFile(artifactPath));
const env = {
  expf: Math.exp,
  logf: Math.log,
  powf: Math.pow,
  sqrtf: Math.sqrt,
  tanhf: Math.tanh,
  sinf: Math.sin,
  cosf: Math.cos,
};
const { exports: api } = await WebAssembly.instantiate(
  module, { env, math: env },
);
assert.ok(api.memory instanceof WebAssembly.Memory);
assert.equal(typeof api.qbatch_matmul_i8u8, 'function');
assert.equal(
  typeof api.qbatch_matmul_i8u8_simd128,
  'function',
  'artifact must expose the standard SIMD128 QBatchMatMul ABI',
);

function allocate(bytes) {
  const pointer = Number(api.alloc_bytes(bytes));
  assert.ok(Number.isInteger(pointer) && pointer > 0);
  const required = pointer + bytes;
  if (required > api.memory.buffer.byteLength) {
    api.memory.grow(
      Math.ceil((required - api.memory.buffer.byteLength) / 65536),
    );
  }
  return pointer;
}

function write(values) {
  const pointer = allocate(values.byteLength);
  new Uint8Array(api.memory.buffer, pointer, values.byteLength).set(
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
  );
  return pointer;
}

function inputValues(length, dtype, multiplier, offset) {
  if (dtype === VX_DTYPE_I8) {
    return Int8Array.from(
      { length },
      (_, index) => ((index * multiplier + offset) % 255) - 127,
    );
  }
  return Uint8Array.from(
    { length },
    (_, index) => (index * multiplier + offset) & 0xff,
  );
}

function averageMilliseconds(call, iterations) {
  for (let warmup = 0; warmup < 3; warmup++) assert.equal(call(), 1);
  const start = performance.now();
  for (let iteration = 0; iteration < iterations; iteration++) {
    assert.equal(call(), 1);
  }
  return (performance.now() - start) / iterations;
}

function benchmarkCase(spec) {
  api.reset_heap();
  const a = inputValues(spec.m * spec.k, spec.aDtype, 29, 17);
  const b = inputValues(spec.k * spec.n, spec.bDtype, 37, 11);
  const aPointer = write(a);
  const bPointer = write(b);
  const scalarOutputPointer = allocate(spec.m * spec.n);
  const simdOutputPointer = allocate(spec.m * spec.n);
  const common = [
    spec.m,
    spec.k,
    spec.n,
    0.03125,
    spec.aDtype === VX_DTYPE_I8 ? -7 : 139,
    0.0078125,
    spec.bDtype === VX_DTYPE_I8 ? -13 : 131,
    0.0625,
    spec.outputDtype === VX_DTYPE_I8 ? -3 : 127,
    spec.aDtype,
    spec.bDtype,
    spec.outputDtype,
  ];
  const scalarCall = () => api.qbatch_matmul_i8u8(
    aPointer, bPointer, scalarOutputPointer, ...common,
  );
  const simdCall = () => api.qbatch_matmul_i8u8_simd128(
    aPointer, bPointer, simdOutputPointer, ...common,
  );
  assert.equal(scalarCall(), 1);
  assert.equal(simdCall(), 1);
  const scalarOutput = new Uint8Array(
    api.memory.buffer, scalarOutputPointer, spec.m * spec.n,
  ).slice();
  const simdOutput = new Uint8Array(
    api.memory.buffer, simdOutputPointer, spec.m * spec.n,
  ).slice();
  assert.deepEqual(
    simdOutput,
    scalarOutput,
    `${spec.name}: SIMD output must byte-match scalar`,
  );
  const scalarMilliseconds = averageMilliseconds(
    scalarCall, spec.iterations,
  );
  const simdMilliseconds = averageMilliseconds(simdCall, spec.iterations);
  let checksum = 0;
  for (const value of simdOutput) checksum = (checksum + value) >>> 0;
  return {
    ...spec,
    scalarMilliseconds,
    simdMilliseconds,
    speedup: scalarMilliseconds / simdMilliseconds,
    checksum,
  };
}

const cases = [
  {
    name: 'encoder attention M=218 K=40 N=218 U8S8',
    m: 218,
    k: 40,
    n: 218,
    aDtype: VX_DTYPE_U8,
    bDtype: VX_DTYPE_I8,
    outputDtype: VX_DTYPE_I8,
    iterations: 100,
  },
  {
    name: 'decoder M=1 K=320 N=320 U8S8',
    m: 1,
    k: 320,
    n: 320,
    aDtype: VX_DTYPE_U8,
    bDtype: VX_DTYPE_I8,
    outputDtype: VX_DTYPE_I8,
    iterations: 100,
  },
  {
    name: 'encoder M=192 K=320 N=192 S8U8',
    m: 192,
    k: 320,
    n: 192,
    aDtype: VX_DTYPE_I8,
    bDtype: VX_DTYPE_U8,
    outputDtype: VX_DTYPE_U8,
    iterations: 8,
  },
  {
    name: 'odd tails M=17 K=65 N=63 U8S8',
    m: 17,
    k: 65,
    n: 63,
    aDtype: VX_DTYPE_U8,
    bDtype: VX_DTYPE_I8,
    outputDtype: VX_DTYPE_U8,
    iterations: 100,
  },
];

console.log(
  `WASM QBatchMatMul scalar/SIMD128 benchmark: ${basename(artifactPath)}`,
);
console.log(`Node ${process.version}; deterministic byte inputs; exact parity required.`);
for (const spec of cases) {
  const result = benchmarkCase(spec);
  console.log(
    `${result.name}: scalar=${result.scalarMilliseconds.toFixed(4)} ms, ` +
    `simd128=${result.simdMilliseconds.toFixed(4)} ms, ` +
    `speedup=${result.speedup.toFixed(2)}x, checksum=${result.checksum}`,
  );
}
