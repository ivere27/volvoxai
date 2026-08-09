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
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;

async function buildInstrumentedWasm(directory) {
  const output = join(directory, 'volvoxai-w8a8-simd.wasm');
  await run(clang, [
    '--target=wasm32', '-std=c11', '-O3', '-msimd128', '-nostdlib',
    '-DVOLVOXAI_W8A8_SIMD_TESTING=1',
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
  const pointer = allocate(values.byteLength);
  new Uint8Array(memory.buffer, pointer, values.byteLength).set(
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
  );
  return pointer;
}

function byteValues(length, dtype, stride, offset) {
  if (dtype === VX_DTYPE_I8) {
    return Int8Array.from({ length }, (_, index) =>
      ((index * stride + offset) % 255) - 127);
  }
  return Uint8Array.from({ length }, (_, index) =>
    (index * stride + offset) & 0xff);
}

function executeCase(api, memory, spec) {
  const {
    name, rows, dIn, dOut, inputDtype, weightDtype, outputDtype,
    inputZeroPoint, outputZeroPoint, expectedSimdCalls,
    expectedSymmetricCalls = 0,
    expectedResult = 1,
    inputScale = 0.03125, outputScale = 0.0625,
    weightScale = (column) => 0.00390625 * (1 + column % 7),
    inputValues = null, weightValues = null, biasValues = null,
    weightZeroPointValues = null,
    checkMalformedPairHeader = false,
    packMode = 'wide',
  } = spec;
  api.reset_heap();
  api.reset_w8a8_wasm_simd_calls();
  const allocate = makeAllocator(api, memory);
  const input = inputValues ?? byteValues(rows * dIn, inputDtype, 29, 17);
  const weight = weightValues ?? byteValues(dOut * dIn, weightDtype, 37, 11);
  const bias = biasValues ?? Int32Array.from(
    { length: dOut }, (_, column) => column * 31 - 113);
  const scales = Float32Array.from({ length: dOut }, (_, column) =>
    weightScale(column));
  const zeroPoints = weightZeroPointValues ?? Int32Array.from(
    { length: dOut }, (_, column) =>
      weightDtype === VX_DTYPE_I8 ? column % 19 - 9 : 111 + column % 23);
  const inputPointer = writeArray(memory, allocate, input);
  const weightPointer = writeArray(memory, allocate, weight);
  const biasPointer = writeArray(memory, allocate, bias);
  const scalePointer = writeArray(memory, allocate, scales);
  const zeroPointPointer = writeArray(memory, allocate, zeroPoints);
  const portableOutputPointer = allocate(rows * dOut);
  const packedOutputPointer = allocate(rows * dOut);
  new Uint8Array(memory.buffer, portableOutputPointer, rows * dOut).fill(0xa5);
  new Uint8Array(memory.buffer, packedOutputPointer, rows * dOut).fill(0xa5);
  const canonicalOnly = packMode === 'canonical';
  assert.ok(canonicalOnly || packMode === 'wide', `${name}: known pack mode`);
  const packedBytes = Number((canonicalOnly
    ? api.packed_q8_weight_canonical_size
    : api.packed_q8_weight_size)(dIn, dOut));
  const packedPointer = allocate(packedBytes);

  const pack = canonicalOnly
    ? api.pack_q8_weight_canonical
    : api.pack_q8_weight;
  assert.equal(pack(
    packedPointer, packedBytes, weightPointer, dIn, dOut, weightDtype, 1,
  ), 1, `${name}: pack weights`);
  {
    const header = new DataView(memory.buffer, packedPointer, 48);
    if (canonicalOnly) {
      assert.ok(Number(api.packed_q8_weight_size(dIn, dOut)) > packedBytes,
        `${name}: canonical pack omits the widened payload`);
      assert.equal(header.getUint32(32, true), 0,
        `${name}: canonical pair-pack N blocks`);
      assert.equal(header.getUint32(36, true), 0,
        `${name}: canonical pair-pack K blocks`);
      assert.equal(header.getUint32(40, true), packedBytes,
        `${name}: canonical bytes end at pair-data offset`);
      assert.equal(header.getUint32(44, true), 0,
        `${name}: canonical pair-pack flags`);
    } else {
      const pairNBlocks = Math.ceil(dOut / 8);
      const pairKBlocks = Math.ceil(dIn / 2);
      const pairDataOffset = header.getUint32(40, true);
      assert.equal(header.getUint32(32, true), pairNBlocks,
        `${name}: widened pair-pack N blocks`);
      assert.equal(header.getUint32(36, true), pairKBlocks,
        `${name}: widened pair-pack K blocks`);
      assert.equal(packedBytes,
        pairDataOffset + pairNBlocks * pairKBlocks * 32,
        `${name}: widened K2/N8 payload bytes`);
      assert.equal((header.getUint32(44, true) & 1) !== 0,
        weightDtype === VX_DTYPE_I8,
        `${name}: pair-pack signed-weight flag`);
    }
  }
  assert.equal(api.qlinear_i8u8(
    inputPointer, weightPointer, biasPointer, scalePointer, zeroPointPointer,
    portableOutputPointer, rows, dIn, dOut,
    inputScale, inputZeroPoint, outputScale, outputZeroPoint,
    inputDtype, weightDtype, outputDtype,
  ), expectedResult, `${name}: portable reference`);
  assert.equal(api.qlinear_i8u8_packed(
    inputPointer, packedPointer, biasPointer, scalePointer, zeroPointPointer,
    packedOutputPointer, rows, dIn, dOut,
    inputScale, inputZeroPoint, outputScale, outputZeroPoint,
    inputDtype, weightDtype, outputDtype,
  ), expectedResult, `${name}: packed kernel`);
  assert.equal(api.w8a8_wasm_simd_calls(), expectedSimdCalls,
    `${name}: SIMD dispatch count`);
  assert.equal(api.w8a8_wasm_symmetric_i8_calls(), expectedSymmetricCalls,
    `${name}: symmetric-I8 dispatch count`);
  assert.deepEqual(
    new Uint8Array(memory.buffer, packedOutputPointer, rows * dOut).slice(),
    new Uint8Array(memory.buffer, portableOutputPointer, rows * dOut).slice(),
    `${name}: exact output bytes`,
  );
  if (expectedResult === 0) {
    assert.deepEqual(
      new Uint8Array(memory.buffer, packedOutputPointer, rows * dOut).slice(),
      new Uint8Array(rows * dOut).fill(0xa5),
      `${name}: rejection must precede the first output write`,
    );
  }
  if (checkMalformedPairHeader) {
    const header = new DataView(memory.buffer, packedPointer, 48);
    const originalPairKBlocks = header.getUint32(36, true);
    header.setUint32(36, originalPairKBlocks + 1, true);
    try {
      new Uint8Array(memory.buffer, packedOutputPointer, rows * dOut).fill(0xa5);
      assert.equal(api.qlinear_i8u8_packed(
        inputPointer, packedPointer, biasPointer, scalePointer, zeroPointPointer,
        packedOutputPointer, rows, dIn, dOut,
        inputScale, inputZeroPoint, outputScale, outputZeroPoint,
        inputDtype, weightDtype, outputDtype,
      ), 0, `${name}: malformed pair-pack metadata must be rejected`);
      assert.deepEqual(
        new Uint8Array(memory.buffer, packedOutputPointer, rows * dOut).slice(),
        new Uint8Array(rows * dOut).fill(0xa5),
        `${name}: malformed pair metadata must fail before output writes`,
      );
    } finally {
      header.setUint32(36, originalPairKBlocks, true);
    }
  }
}

test('baseline WASM SIMD128 packed W8A8 is exact across byte types and tails', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-w8a8-simd-'));
  try {
    const module = await WebAssembly.compile(
      await readFile(await buildInstrumentedWasm(directory)),
    );
    const { exports: api } = await WebAssembly.instantiate(
      module, importsFor(module));
    const memory = api.memory;
    assert.ok(memory instanceof WebAssembly.Memory);
    assert.equal(typeof api.w8a8_wasm_simd_calls, 'function');
    assert.equal(typeof api.w8a8_wasm_symmetric_i8_calls, 'function');
    assert.equal(typeof api.qlinear_i8u8_packed, 'function',
      'the canonical packed QLinear ABI must remain exported');
    assert.equal(api.vx_qlinear_i8u8_packed, undefined,
      'the packed QLinear ABI must not depend on its internal C symbol');
    assert.equal(api.vx_packed_q8_prefers_signed_activations, undefined,
      'the native-only packed-domain policy helper must not enter WASM');

    executeCase(api, memory, {
      name: 'I8 input/weight/output with odd MR, K, and N tails',
      rows: 5, dIn: 67, dOut: 13,
      inputDtype: VX_DTYPE_I8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_I8,
      inputZeroPoint: -7, outputZeroPoint: 3, expectedSimdCalls: 1,
    });
    executeCase(api, memory, {
      name: 'U8 input/weight/output with asymmetric zero points',
      rows: 3, dIn: 19, dOut: 11,
      inputDtype: VX_DTYPE_U8, weightDtype: VX_DTYPE_U8,
      outputDtype: VX_DTYPE_U8,
      inputZeroPoint: 131, outputZeroPoint: 127, expectedSimdCalls: 1,
    });
    executeCase(api, memory, {
      name: 'U8 input and I8 weight with I8 output',
      rows: 6, dIn: 18, dOut: 16,
      inputDtype: VX_DTYPE_U8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_I8,
      inputZeroPoint: 149, outputZeroPoint: -11, expectedSimdCalls: 1,
    });
    executeCase(api, memory, {
      name: 'I8 input and U8 weight with U8 output',
      rows: 2, dIn: 33, dOut: 9,
      inputDtype: VX_DTYPE_I8, weightDtype: VX_DTYPE_U8,
      outputDtype: VX_DTYPE_U8,
      inputZeroPoint: -19, outputZeroPoint: 173, expectedSimdCalls: 1,
    });
    executeCase(api, memory, {
      name: 'N<8 canonical-only W8A8 pack stays exact on the scalar path',
      rows: 3, dIn: 19, dOut: 7,
      inputDtype: VX_DTYPE_U8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_U8,
      inputZeroPoint: 137, outputZeroPoint: 121, expectedSimdCalls: 0,
      weightZeroPointValues: new Int32Array(7),
      packMode: 'canonical',
    });
    executeCase(api, memory, {
      name: 'bounded seed M=402 K=320 N=320',
      rows: 402, dIn: 320, dOut: 320,
      inputDtype: VX_DTYPE_I8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_I8,
      inputZeroPoint: 0, outputZeroPoint: 0, expectedSimdCalls: 1,
      expectedSymmetricCalls: 1,
      weightZeroPointValues: new Int32Array(320),
    });
    executeCase(api, memory, {
      name: 'symmetric I8 odd M and K tails',
      rows: 7, dIn: 67, dOut: 16,
      inputDtype: VX_DTYPE_I8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_I8,
      inputZeroPoint: 0, outputZeroPoint: 0, expectedSimdCalls: 1,
      expectedSymmetricCalls: 1,
      weightZeroPointValues: new Int32Array(16),
    });
    executeCase(api, memory, {
      name: 'affine U8 panel with symmetric I8 weights',
      rows: 7, dIn: 67, dOut: 16,
      inputDtype: VX_DTYPE_U8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_U8,
      inputZeroPoint: 137, outputZeroPoint: 121, expectedSimdCalls: 1,
      weightZeroPointValues: new Int32Array(16),
      checkMalformedPairHeader: true,
    });
    executeCase(api, memory, {
      name: 'affine I8 panel with symmetric I8 weights and nonzero zero points',
      rows: 4, dIn: 65, dOut: 16,
      inputDtype: VX_DTYPE_I8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_I8,
      inputZeroPoint: -13, outputZeroPoint: 9, expectedSimdCalls: 1,
      weightZeroPointValues: new Int32Array(16),
    });
    executeCase(api, memory, {
      name: 'affine U8 symmetric-weight path with odd K and N tails',
      rows: 3, dIn: 19, dOut: 13,
      inputDtype: VX_DTYPE_U8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_U8,
      inputZeroPoint: 137, outputZeroPoint: 121, expectedSimdCalls: 1,
      weightZeroPointValues: new Int32Array(13),
    });
    executeCase(api, memory, {
      name: 'vector requantization preserves round-to-nearest ties-to-even',
      rows: 1, dIn: 1, dOut: 8,
      inputDtype: VX_DTYPE_I8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_I8,
      inputZeroPoint: 0, outputZeroPoint: 0, expectedSimdCalls: 1,
      expectedSymmetricCalls: 1,
      inputScale: 1, outputScale: 2, weightScale: () => 1,
      inputValues: Int8Array.of(1),
      weightValues: Int8Array.of(1, 3, 5, 7, -1, -3, -5, -7),
      biasValues: new Int32Array(8),
      weightZeroPointValues: new Int32Array(8),
    });
    executeCase(api, memory, {
      name: 'non-finite multiplier is rejected by portable and packed kernels',
      rows: 2, dIn: 3, dOut: 8,
      inputDtype: VX_DTYPE_I8, weightDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_I8,
      inputZeroPoint: 0, outputZeroPoint: 17, expectedSimdCalls: 0,
      expectedResult: 0,
      inputScale: 3.4028234663852886e38,
      outputScale: 1.1754943508222875e-38,
      weightScale: () => 3.4028234663852886e38,
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
