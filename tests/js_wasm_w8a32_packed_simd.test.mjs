import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';
// Canonical protobuf DataType values used by the public native/WASM ABI.
const VX_DTYPE_UNSPECIFIED = 0;
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;
const VX_DTYPE_I32 = 16;
const VX_DTYPE_F32 = 18;

async function buildInstrumentedWasm(directory) {
  const output = join(directory, 'volvoxai-w8a32-simd.wasm');
  await run(clang, [
    '--target=wasm32', '-std=c11', '-O3', '-msimd128', '-nostdlib',
    '-DVOLVOXAI_W8A32_SIMD_TESTING=1',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot, maxBuffer: 1024 * 1024 });
  return output;
}

function importsFor(module) {
  const imports = {};
  for (const descriptor of WebAssembly.Module.imports(module)) {
    assert.equal(descriptor.kind, 'function',
      `unexpected ${descriptor.kind} import ${descriptor.module}.${descriptor.name}`);
    imports[descriptor.module] ||= {};
    imports[descriptor.module][descriptor.name] = () => 0;
  }
  return imports;
}

function signedWeights(dIn, dOut) {
  return Int8Array.from({ length: dIn * dOut }, (_, index) => {
    const column = Math.floor(index / dIn);
    const k = index % dIn;
    return ((column * 37 + k * 29 + 11) % 255) - 127;
  });
}

function unsignedWeights(dIn, dOut) {
  return Uint8Array.from({ length: dIn * dOut }, (_, index) => {
    const column = Math.floor(index / dIn);
    const k = index % dIn;
    return (column * 53 + k * 31 + 7) & 0xff;
  });
}

function inputValues(rows, dIn) {
  return Float32Array.from({ length: rows * dIn }, (_, index) =>
    (((index * 13 + Math.floor(index / dIn) * 5) % 19) - 9) / 8);
}

function makeAllocator(api, memory) {
  return (bytes) => {
    const pointer = Number(api.alloc_bytes(bytes));
    assert.ok(Number.isInteger(pointer) && pointer > 0,
      `failed to allocate ${bytes} bytes`);
    const end = pointer + bytes;
    if (end > memory.buffer.byteLength) {
      memory.grow(Math.ceil((end - memory.buffer.byteLength) / 65536));
    }
    return pointer;
  };
}

function writeArray(memory, allocate, values) {
  if (values === null) return 0;
  const pointer = allocate(values.byteLength);
  new Uint8Array(memory.buffer, pointer, values.byteLength).set(
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
  );
  return pointer;
}

function assertFloatArraysEqual(actual, expected, label, tolerance = null) {
  assert.equal(actual.length, expected.length);
  let maxAbsoluteError = 0;
  let maxRelativeError = 0;
  for (let index = 0; index < actual.length; index++) {
    const absoluteError = Math.abs(actual[index] - expected[index]);
    const relativeError = absoluteError / Math.max(Math.abs(expected[index]), 1e-12);
    maxAbsoluteError = Math.max(maxAbsoluteError, absoluteError);
    maxRelativeError = Math.max(maxRelativeError, relativeError);
    if (tolerance) {
      const limit = tolerance.absolute + tolerance.relative * Math.abs(expected[index]);
      assert.ok(absoluteError <= limit,
        `${label} at output ${index}: actual=${actual[index]}, expected=${expected[index]}, ` +
        `error=${absoluteError}, limit=${limit}`);
    } else {
      assert.equal(actual[index], expected[index], `${label} at output ${index}`);
    }
  }
  return { maxAbsoluteError, maxRelativeError };
}

function executeCase(api, memory, spec) {
  const {
    name, rows, dIn, dOut, weightDtype, weights, scales, scaleElements,
    zeroPoints, zeroPointDtype, zeroPointElements, bias, expectedSimdCalls,
    inputs = inputValues(rows, dIn), tolerance = null,
  } = spec;
  api.reset_heap();
  api.reset_w8a32_wasm_simd_calls();
  const allocate = makeAllocator(api, memory);
  const inputPointer = writeArray(memory, allocate, inputs);
  const weightPointer = writeArray(memory, allocate, weights);
  const scalePointer = writeArray(memory, allocate, scales);
  const zeroPointPointer = writeArray(memory, allocate, zeroPoints);
  const biasPointer = writeArray(memory, allocate, bias);
  const packedBytes = Number(api.packed_q8_weight_canonical_size(dIn, dOut));
  const widenedBytes = Number(api.packed_q8_weight_size(dIn, dOut));
  assert.ok(packedBytes > 0, `${name}: packed size`);
  assert.ok(widenedBytes > packedBytes,
    `${name}: W8A32 canonical pack must omit the widened W8A8 payload`);
  const packedPointer = allocate(packedBytes);
  const packedOutputPointer = allocate(rows * dOut * 4);
  const portableOutputPointer = allocate(rows * dOut * 4);

  assert.equal(api.pack_q8_weight_canonical(
    packedPointer, packedBytes, weightPointer, dIn, dOut, weightDtype, 1,
  ), 1, `${name}: pack output-major weights`);
  {
    const header = new DataView(memory.buffer, packedPointer, 48);
    assert.equal(header.getUint32(32, true), 0,
      `${name}: canonical-only pair N blocks`);
    assert.equal(header.getUint32(36, true), 0,
      `${name}: canonical-only pair K blocks`);
    assert.equal(header.getUint32(40, true), packedBytes,
      `${name}: canonical-only payload ends at the pair-data offset`);
    assert.equal(header.getUint32(44, true), 0,
      `${name}: canonical-only pair flags`);
  }
  assert.equal(api.matmul_quantized_f32(
    inputPointer, weightPointer, scalePointer, zeroPointPointer, biasPointer,
    portableOutputPointer, rows, dIn, dOut, weightDtype, scaleElements,
    zeroPointDtype, zeroPointElements,
  ), 1, `${name}: portable reference`);
  assert.equal(api.matmul_quantized_f32_packed(
    inputPointer, packedPointer, scalePointer, zeroPointPointer, biasPointer,
    packedOutputPointer, rows, dIn, dOut, weightDtype, scaleElements,
    zeroPointDtype, zeroPointElements,
  ), 1, `${name}: packed kernel`);

  assert.equal(api.w8a32_wasm_simd_calls(), expectedSimdCalls,
    `${name}: SIMD dispatch count`);
  const packedOutput = new Float32Array(
    memory.buffer, packedOutputPointer, rows * dOut).slice();
  const portableOutput = new Float32Array(
    memory.buffer, portableOutputPointer, rows * dOut).slice();
  return assertFloatArraysEqual(packedOutput, portableOutput, name, tolerance);
}

test('actual WASM SIMD128 packed W8A32 M=1 is correct and preserves fallbacks', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-w8a32-simd-'));
  try {
    const module = await WebAssembly.compile(
      await readFile(await buildInstrumentedWasm(directory)),
    );
    const { exports: api } = await WebAssembly.instantiate(module, importsFor(module));
    const memory = api.memory;
    assert.ok(memory instanceof WebAssembly.Memory);
    assert.equal(typeof api.w8a32_wasm_simd_calls, 'function');
    assert.equal(typeof api.packed_q8_weight_canonical_size, 'function');
    assert.equal(typeof api.pack_q8_weight_canonical, 'function');
    assert.equal(
      api.packed_q8_weight_canonical_size(
        Math.floor(0x7fffffff / 255) + 1, 1,
      ),
      0,
      'canonical size must reject K beyond the exact I32 accumulation bound',
    );
    assert.equal(api.packed_q8_weight_canonical_size(1, 0xffffffff), 0,
      'canonical size must reject a byte-count overflow');

    executeCase(api, memory, {
      name: 'I8 asymmetric per-channel F32 zero point scalar fallback',
      rows: 1, dIn: 31, dOut: 11, weightDtype: VX_DTYPE_I8,
      weights: signedWeights(31, 11),
      scales: Float32Array.from({ length: 11 }, (_, column) => (column % 4 + 1) / 256),
      scaleElements: 11,
      zeroPoints: Float32Array.from({ length: 11 }, (_, column) =>
        ((column * 3) % 11) - 5.5),
      zeroPointDtype: VX_DTYPE_F32, zeroPointElements: 11,
      bias: Float32Array.from({ length: 11 }, (_, column) => (column * 3 - 12) / 4),
      expectedSimdCalls: 0,
    });

    executeCase(api, memory, {
      name: 'U8 asymmetric scalar zero point fallback without bias',
      rows: 1, dIn: 17, dOut: 9, weightDtype: VX_DTYPE_U8,
      weights: unsignedWeights(17, 9), scales: Float32Array.of(0.125),
      scaleElements: 1, zeroPoints: Uint8Array.of(127),
      zeroPointDtype: VX_DTYPE_U8, zeroPointElements: 1, bias: null,
      expectedSimdCalls: 0,
    });

    executeCase(api, memory, {
      name: 'I8 odd-K/odd-N SIMD with per-channel scales and symmetric metadata',
      rows: 1, dIn: 963, dOut: 11, weightDtype: VX_DTYPE_I8,
      weights: signedWeights(963, 11),
      scales: Float32Array.from({ length: 11 }, (_, column) => (column % 4 + 1) / 256),
      scaleElements: 11,
      zeroPoints: Float32Array.from({ length: 11 }, () => 0),
      zeroPointDtype: VX_DTYPE_F32, zeroPointElements: 11,
      bias: Float32Array.from({ length: 11 }, (_, column) => (column * 3 - 12) / 4),
      expectedSimdCalls: 1,
    });

    executeCase(api, memory, {
      name: 'U8 symmetric descriptor stays on the scalar fallback',
      rows: 1, dIn: 17, dOut: 9, weightDtype: VX_DTYPE_U8,
      weights: unsignedWeights(17, 9), scales: Float32Array.of(0.125),
      scaleElements: 1, zeroPoints: Uint8Array.of(0),
      zeroPointDtype: VX_DTYPE_U8, zeroPointElements: 1, bias: null,
      expectedSimdCalls: 0,
    });

    executeCase(api, memory, {
      name: 'I8 full SIMD panel with no zero point and no bias',
      rows: 1, dIn: 7, dOut: 8, weightDtype: VX_DTYPE_I8,
      weights: signedWeights(7, 8), scales: Float32Array.of(0.015625),
      scaleElements: 1, zeroPoints: null,
      zeroPointDtype: VX_DTYPE_UNSPECIFIED, zeroPointElements: 0, bias: null,
      expectedSimdCalls: 1,
    });

    executeCase(api, memory, {
      name: 'M=1 N<8 scalar fallback with I8 zero point and bias',
      rows: 1, dIn: 15, dOut: 7, weightDtype: VX_DTYPE_I8,
      weights: signedWeights(15, 7), scales: Float32Array.of(0.0625),
      scaleElements: 1, zeroPoints: Int8Array.of(-3),
      zeroPointDtype: VX_DTYPE_I8, zeroPointElements: 1,
      bias: Float32Array.from({ length: 7 }, (_, column) => column / 4),
      expectedSimdCalls: 0,
    });

    executeCase(api, memory, {
      name: 'M>1 scalar fallback with U8 per-channel I32 zero points',
      rows: 2, dIn: 19, dOut: 9, weightDtype: VX_DTYPE_U8,
      weights: unsignedWeights(19, 9),
      scales: Float32Array.from({ length: 9 }, (_, column) => (column % 3 + 1) / 32),
      scaleElements: 9,
      zeroPoints: Int32Array.from({ length: 9 }, (_, column) => 120 + column),
      zeroPointDtype: VX_DTYPE_I32, zeroPointElements: 9,
      bias: Float32Array.from({ length: 9 }, (_, column) => (column - 4) / 8),
      expectedSimdCalls: 0,
    });

    const nonDyadicError = executeCase(api, memory, {
      name: 'non-dyadic symmetric I8 SIMD tolerance',
      rows: 1, dIn: 257, dOut: 11, weightDtype: VX_DTYPE_I8,
      inputs: Float32Array.from({ length: 257 }, (_, index) =>
        Math.sin(index * 0.17) * 1.137 + 0.0031),
      weights: signedWeights(257, 11),
      scales: Float32Array.from({ length: 11 }, (_, column) =>
        0.0037 + column * 0.00091),
      scaleElements: 11, zeroPoints: null,
      zeroPointDtype: VX_DTYPE_UNSPECIFIED, zeroPointElements: 0,
      bias: Float32Array.from({ length: 11 }, (_, column) =>
        Math.cos(column * 0.11) * 0.37),
      expectedSimdCalls: 1,
      tolerance: { absolute: 2e-4, relative: 2e-5 },
    });
    t.diagnostic(
      `non-dyadic max abs error=${nonDyadicError.maxAbsoluteError}, ` +
      `max relative error=${nonDyadicError.maxRelativeError}`,
    );

    const cancellationInputs = new Float32Array(320);
    cancellationInputs.fill(Math.fround(9.133296013), 0, 160);
    cancellationInputs.fill(Math.fround(-9.133296013), 160);
    const cancellationError = executeCase(api, memory, {
      name: 'bounded cancellation remains inside the documented F32 tolerance',
      rows: 1, dIn: 320, dOut: 8, weightDtype: VX_DTYPE_I8,
      inputs: cancellationInputs, weights: new Int8Array(320 * 8).fill(127),
      scales: Float32Array.of(0.00798), scaleElements: 1,
      zeroPoints: null, zeroPointDtype: VX_DTYPE_UNSPECIFIED,
      zeroPointElements: 0,
      bias: null, expectedSimdCalls: 1,
      tolerance: { absolute: 2e-4, relative: 0 },
    });
    t.diagnostic(
      `bounded cancellation max abs error=${cancellationError.maxAbsoluteError}`,
    );

    const extremeCancellationInputs = new Float32Array(1280);
    extremeCancellationInputs.fill(Math.fround(34.235813), 0, 640);
    extremeCancellationInputs.fill(Math.fround(-34.235813), 640);
    const extremeCancellationError = executeCase(api, memory, {
      name: 'audited maximum-K cancellation uses compensated SIMD accumulation',
      rows: 1, dIn: 1280, dOut: 8, weightDtype: VX_DTYPE_I8,
      inputs: extremeCancellationInputs,
      weights: new Int8Array(1280 * 8).fill(127),
      scales: Float32Array.of(0.024), scaleElements: 1,
      zeroPoints: null, zeroPointDtype: VX_DTYPE_UNSPECIFIED,
      zeroPointElements: 0,
      bias: null, expectedSimdCalls: 1,
      tolerance: { absolute: 2e-4, relative: 0 },
    });
    t.diagnostic(
      `maximum-K cancellation max abs error=${extremeCancellationError.maxAbsoluteError}`,
    );

    executeCase(api, memory, {
      name: 'K above the audited Tiny envelope stays scalar',
      rows: 1, dIn: 1281, dOut: 8, weightDtype: VX_DTYPE_I8,
      weights: signedWeights(1281, 8), scales: Float32Array.of(0.015625),
      scaleElements: 1, zeroPoints: null,
      zeroPointDtype: VX_DTYPE_UNSPECIFIED, zeroPointElements: 0, bias: null,
      expectedSimdCalls: 0,
    });

    executeCase(api, memory, {
      name: 'activation outside the audited Tiny envelope stays scalar',
      rows: 1, dIn: 1, dOut: 8, weightDtype: VX_DTYPE_I8,
      inputs: Float32Array.of(35.25), weights: new Int8Array(8).fill(1),
      scales: Float32Array.of(0.015625), scaleElements: 1,
      zeroPoints: null, zeroPointDtype: VX_DTYPE_UNSPECIFIED,
      zeroPointElements: 0,
      bias: null, expectedSimdCalls: 0,
    });

    executeCase(api, memory, {
      name: 'weight scale above the audited Tiny envelope stays scalar',
      rows: 1, dIn: 320, dOut: 8, weightDtype: VX_DTYPE_I8,
      weights: signedWeights(320, 8), scales: Float32Array.of(0.0251),
      scaleElements: 1, zeroPoints: null,
      zeroPointDtype: VX_DTYPE_UNSPECIFIED, zeroPointElements: 0,
      bias: null, expectedSimdCalls: 0,
    });

    executeCase(api, memory, {
      name: 'non-finite activation stays scalar',
      rows: 1, dIn: 1, dOut: 8, weightDtype: VX_DTYPE_I8,
      inputs: Float32Array.of(Infinity), weights: new Int8Array(8).fill(1),
      scales: Float32Array.of(0.015625), scaleElements: 1,
      zeroPoints: null, zeroPointDtype: VX_DTYPE_UNSPECIFIED,
      zeroPointElements: 0,
      bias: null, expectedSimdCalls: 0,
    });

    api.reset_heap();
    api.reset_w8a32_wasm_simd_calls();
    const allocate = makeAllocator(api, memory);
    const malformedInput = writeArray(memory, allocate, Float32Array.of(1, -2, 3));
    const malformedWeightValues = signedWeights(3, 8);
    const malformedWeight = writeArray(memory, allocate, malformedWeightValues);
    const malformedScale = writeArray(memory, allocate, Float32Array.of(0.125));
    const malformedZeroPoint = writeArray(memory, allocate, Int8Array.of(0));
    const malformedPackedBytes = Number(api.packed_q8_weight_canonical_size(3, 8));
    const malformedPacked = allocate(malformedPackedBytes);
    const malformedOutput = allocate(8 * 4);
    new Uint8Array(memory.buffer, malformedPacked, malformedPackedBytes).fill(0xa5);
    assert.equal(api.pack_q8_weight_canonical(
      malformedPacked, malformedPackedBytes - 1, malformedWeight, 3, 8,
      VX_DTYPE_I8, 1,
    ), 0, 'an undersized canonical destination must be rejected');
    assert.deepEqual(
      [...new Uint8Array(memory.buffer, malformedPacked, malformedPackedBytes)],
      Array(malformedPackedBytes).fill(0xa5),
      'canonical packing must validate capacity before its first write',
    );
    new Uint8Array(memory.buffer, malformedOutput, 8 * 4).fill(0xa5);
    assert.equal(api.pack_q8_weight_canonical(
      malformedPacked, malformedPackedBytes, malformedWeight, 3, 8,
      VX_DTYPE_I8, 1,
    ), 1);
    assert.equal(api.matmul_quantized_f32_packed(
      malformedInput, malformedPacked, malformedScale, malformedZeroPoint, 0,
      malformedOutput, 1, 3, 8, VX_DTYPE_I8, 1, VX_DTYPE_I8, 0,
    ), 0, 'non-null zero-point data with zero elements must be rejected');
    assert.equal(api.w8a32_wasm_simd_calls(), 0);
    assert.deepEqual(
      [...new Uint8Array(memory.buffer, malformedOutput, 8 * 4)],
      Array(8 * 4).fill(0xa5),
      'a rejected malformed descriptor must not write output',
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
