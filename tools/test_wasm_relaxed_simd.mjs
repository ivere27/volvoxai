#!/usr/bin/env node
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';

const sectionName = 'volvoxai.relaxed_simd.v1';
const opcode = Buffer.from([0xfd, 0x93, 0x02]);
const baselineOnly = process.argv[2] === '--baseline-only';
const artifactPaths = process.argv.slice(baselineOnly ? 3 : 2);

if (artifactPaths.length === 0) {
  throw new Error('usage: test_wasm_relaxed_simd.mjs [--baseline-only] <parent.wasm> [...]');
}

const children = [];
for (const artifactPath of artifactPaths) {
  const bytes = await readFile(artifactPath);
  assert.equal(WebAssembly.validate(bytes), true,
    `${artifactPath} must remain a baseline-valid WebAssembly module`);
  const parentModule = new WebAssembly.Module(bytes);
  const sections = WebAssembly.Module.customSections(parentModule, sectionName);
  assert.equal(sections.length, 1, `${artifactPath} must contain exactly one ${sectionName} section`);
  const child = Buffer.from(sections[0]);
  assert.equal(child.subarray(0, 8).toString('hex'), '0061736d01000000',
    `${artifactPath} custom section must contain a WebAssembly child module`);
  assert.ok(child.indexOf(opcode) >= 0,
    `${artifactPath} child must contain i32x4.relaxed_dot_i8x16_i7x16_add_s`);
  children.push(child);
}
for (let index = 1; index < children.length; index++) {
  assert.deepEqual(children[index], children[0],
    'inference and full profiles must embed identical accelerator bytes');
}

if (baselineOnly) {
  console.log(`Verified baseline validation and ${sectionName} embedding.`);
  process.exit(0);
}

const childModule = new WebAssembly.Module(children[0]);
assert.deepEqual(WebAssembly.Module.imports(childModule), [
  { module: 'env', name: 'memory', kind: 'memory' },
], 'the accelerator must import only the parent memory');
assert.deepEqual(WebAssembly.Module.exports(childModule), [
  { name: 'qlinear_i8u8_relaxed', kind: 'function' },
], 'the accelerator must expose only its versioned QLinear ABI');

const parentBytes = await readFile(artifactPaths[0]);
const parentModule = new WebAssembly.Module(parentBytes);
const mathImports = {
  expf: Math.exp,
  tanhf: Math.tanh,
  logf: Math.log,
  sinf: Math.sin,
  cosf: Math.cos,
  powf: Math.pow,
};
const parent = await WebAssembly.instantiate(parentModule, { env: mathImports });
const memory = parent.exports.memory;
assert.ok(memory instanceof WebAssembly.Memory, 'parent module must export its memory');
const child = await WebAssembly.instantiate(childModule, { env: { memory } });
const qlinear = child.exports.qlinear_i8u8_relaxed;
assert.equal(typeof qlinear, 'function');

// Canonical protobuf DataType values used by the public native/WASM ABI.
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;
const VX_DTYPE_I32 = 16;
const VX_PACKED_Q8_MAGIC_V8Q1 = 0x31513856;
const VX_PACKED_Q8_MAGIC_V8Q2 = 0x32513856;
const VX_PACKED_Q8_HEADER_WORDS = 12;
const VX_PACKED_Q8_NR = 8;
const dIn = 17;
const dOut = 7;
const rawInput = Uint8Array.from([
  0, 127, 128, 255, 1, 254, 64, 192, 3,
  253, 126, 129, 17, 239, 85, 170, 128,
]);
const rawWeights = new Uint8Array(dIn * dOut);
const edgeBytes = [0, 127, 128, 255, 1, 254, 64, 192, 17, 239];
for (let column = 0; column < dOut; column++) {
  for (let k = 0; k < dIn; k++) {
    rawWeights[column * dIn + k] =
      edgeBytes[(column * 3 + k * 7) % edgeBytes.length];
  }
}
const biasValues = Int32Array.from([11, -17, 23, -29, 31, -37, 41]);
const weightScales = Float32Array.from([1, 0.5, 2, 0.25, 1, 0.5, 2]);
const signedWeightZeroPoints = Int32Array.from([-128, 127, 0, -7, 63, -64, 1]);
const unsignedWeightZeroPoints = Int32Array.from([0, 255, 128, 7, 200, 1, 254]);

function allocate(bytes) {
  const pointer = Number(parent.exports.alloc_bytes(bytes));
  assert.ok(Number.isInteger(pointer) && pointer > 0, `failed to allocate ${bytes} bytes`);
  const needed = pointer + bytes;
  if (needed > memory.buffer.byteLength) {
    memory.grow(Math.ceil((needed - memory.buffer.byteLength) / 65536));
  }
  return pointer;
}

const inputPointer = allocate(dIn);
const weightPointer = allocate(dIn * dOut);
const packedBytes = Number(parent.exports.packed_q8_weight_size(dIn, dOut));
assert.ok(packedBytes > 0);
const packedPointer = allocate(packedBytes);
const biasPointer = allocate(dOut * 4);
const scalesPointer = allocate(dOut * 4);
const zeroPointsPointer = allocate(dOut * 4);
const outputPointer = allocate(dOut);
const baselineOutputPointer = allocate(dOut);

function writeBytes(pointer, values) {
  new Uint8Array(memory.buffer, pointer, values.byteLength).set(
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
  );
}

function align16(value) {
  return Math.ceil(value / 16) * 16;
}

function expectedV8Q2HeaderWords(inputDimension, outputDimension,
    weightDtype, bytes) {
  const nBlocks = Math.ceil(outputDimension / VX_PACKED_Q8_NR);
  const sumsOffset = align16(VX_PACKED_Q8_HEADER_WORDS * 4);
  const dataOffset = align16(
    sumsOffset + nBlocks * VX_PACKED_Q8_NR * Int32Array.BYTES_PER_ELEMENT,
  );
  const pairDataOffset = align16(
    dataOffset + nBlocks * inputDimension * VX_PACKED_Q8_NR,
  );
  assert.equal(bytes, pairDataOffset,
    'the wasm V8Q2 pack must end at its aligned pair-data offset');
  return [
    VX_PACKED_Q8_MAGIC_V8Q2, bytes, inputDimension, outputDimension,
    nBlocks, weightDtype, sumsOffset, dataOffset,
    0, 0, pairDataOffset, 0,
  ];
}

function readPackedHeaderWords(pointer) {
  const view = new DataView(
    memory.buffer, pointer, VX_PACKED_Q8_HEADER_WORDS * 4,
  );
  return Array.from(
    { length: VX_PACKED_Q8_HEADER_WORDS },
    (_, index) => view.getUint32(index * 4, true),
  );
}

function semanticByte(raw, dtype) {
  return dtype === VX_DTYPE_I8 && raw >= 128 ? raw - 256 : raw;
}

function roundTiesEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function expectedOutput(inputDtype, weightDtype, outputDtype, inputZeroPoint,
    weightZeroPoints, outputZeroPoint) {
  const minimum = outputDtype === VX_DTYPE_I8 ? -128 : 0;
  const maximum = outputDtype === VX_DTYPE_I8 ? 127 : 255;
  return Array.from({ length: dOut }, (_, column) => {
    let accumulator = biasValues[column];
    for (let k = 0; k < dIn; k++) {
      const x = semanticByte(rawInput[k], inputDtype) - inputZeroPoint;
      const w = semanticByte(rawWeights[column * dIn + k], weightDtype) -
        weightZeroPoints[column];
      accumulator += x * w;
    }
    const transformed = accumulator * weightScales[column] / 256 + outputZeroPoint;
    const quantized = Math.max(minimum, Math.min(maximum, roundTiesEven(transformed)));
    return outputDtype === VX_DTYPE_I8 && quantized < 0 ?
      quantized + 256 : quantized;
  });
}

writeBytes(inputPointer, rawInput);
writeBytes(weightPointer, rawWeights);
writeBytes(biasPointer, biasValues);
writeBytes(scalesPointer, weightScales);

for (const inputDtype of [VX_DTYPE_I8, VX_DTYPE_U8]) {
  for (const weightDtype of [VX_DTYPE_I8, VX_DTYPE_U8]) {
    for (const outputDtype of [VX_DTYPE_I8, VX_DTYPE_U8]) {
      const inputZeroPoint = inputDtype === VX_DTYPE_I8 ? -7 : 131;
      const outputZeroPoint = outputDtype === VX_DTYPE_I8 ? -3 : 121;
      const weightZeroPoints = weightDtype === VX_DTYPE_I8
        ? signedWeightZeroPoints : unsignedWeightZeroPoints;
      writeBytes(zeroPointsPointer, weightZeroPoints);
      new Uint8Array(memory.buffer, outputPointer, dOut).fill(0xa5);
      new Uint8Array(memory.buffer, baselineOutputPointer, dOut).fill(0xa5);
      assert.equal(parent.exports.pack_q8_weight(
        packedPointer, packedBytes, weightPointer, dIn, dOut, weightDtype, 1,
      ), 1, 'parent must create the canonical V8Q2 panel');
      assert.deepEqual(
        readPackedHeaderWords(packedPointer),
        expectedV8Q2HeaderWords(dIn, dOut, weightDtype, packedBytes),
        'parent and child must share the exact wasm V8Q2 header contract',
      );
      assert.equal(parent.exports.qlinear_i8u8(
        inputPointer, weightPointer, biasPointer, scalesPointer,
        zeroPointsPointer, baselineOutputPointer,
        1, dIn, dOut, 1, inputZeroPoint, 256, outputZeroPoint,
        inputDtype, weightDtype, outputDtype,
      ), 1, 'parent portable QLinear must accept the descriptor');
      assert.equal(qlinear(
        inputPointer, weightPointer, packedPointer, biasPointer,
        scalesPointer, zeroPointsPointer, outputPointer,
        1, dIn, dOut, 1, inputZeroPoint, 256, outputZeroPoint,
        inputDtype, weightDtype, outputDtype,
      ), 1, 'eligible M=1 descriptor must execute');
      assert.deepEqual(
        [...new Uint8Array(memory.buffer, outputPointer, dOut)],
        expectedOutput(inputDtype, weightDtype, outputDtype, inputZeroPoint,
          weightZeroPoints, outputZeroPoint),
        `full-byte result mismatch for input=${inputDtype}, weight=${weightDtype}, output=${outputDtype}`,
      );
      assert.deepEqual(
        [...new Uint8Array(memory.buffer, outputPointer, dOut)],
        [...new Uint8Array(memory.buffer, baselineOutputPointer, dOut)],
        'Relaxed-SIMD output must byte-match the authoritative portable kernel',
      );
    }
  }
}

function assertCorruptPackedHeaderRejected(word, replacement, label) {
  const byteOffset = word * 4;
  const header = new DataView(
    memory.buffer, packedPointer, VX_PACKED_Q8_HEADER_WORDS * 4,
  );
  const original = header.getUint32(byteOffset, true);
  header.setUint32(byteOffset, replacement, true);
  try {
    new Uint8Array(memory.buffer, outputPointer, dOut).fill(0xa5);
    assert.equal(qlinear(
      inputPointer, weightPointer, packedPointer, biasPointer,
      scalesPointer, zeroPointsPointer, outputPointer,
      1, dIn, dOut, 1, 131, 256, 121,
      VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
    ), 0, label);
    assert.deepEqual(
      [...new Uint8Array(memory.buffer, outputPointer, dOut)],
      Array(dOut).fill(0xa5),
      `${label}: rejection must precede the first output write`,
    );
  } finally {
    header.setUint32(byteOffset, original, true);
  }
}

assertCorruptPackedHeaderRejected(
  0, VX_PACKED_Q8_MAGIC_V8Q1,
  'a legacy V8Q1 magic must not be interpreted as a V8Q2 pack',
);
assertCorruptPackedHeaderRejected(
  1, packedBytes - 16,
  'a truncated V8Q2 byte count must fail closed',
);
assertCorruptPackedHeaderRejected(
  6, 32,
  'a legacy V8Q1 sums offset must fail closed',
);
assertCorruptPackedHeaderRejected(
  10, packedBytes - 16,
  'a malformed V8Q2 pair-data offset must fail closed',
);
assertCorruptPackedHeaderRejected(
  11, 1,
  'unexpected wasm V8Q2 pair flags must fail closed',
);

new Uint8Array(memory.buffer, outputPointer, dOut).fill(0xa5);
assert.equal(qlinear(
  inputPointer, weightPointer, packedPointer, biasPointer,
  scalesPointer, zeroPointsPointer, outputPointer,
  2, dIn, dOut, 1, 131, 256, 121,
  VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
), 0, 'M>1 must fail closed to the baseline path');
assert.deepEqual([...new Uint8Array(memory.buffer, outputPointer, dOut)],
  Array(dOut).fill(0xa5), 'rejected descriptors must not write output bytes');

const crossingOutputPointer = memory.buffer.byteLength - 1;
new Uint8Array(memory.buffer, crossingOutputPointer, 1).fill(0xa5);
assert.equal(qlinear(
  inputPointer, weightPointer, packedPointer, biasPointer,
  scalesPointer, zeroPointsPointer, crossingOutputPointer,
  1, dIn, dOut, 1, 131, 256, 121,
  VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
), 0, 'a descriptor crossing the current imported-memory bound must fail closed');
assert.equal(new Uint8Array(memory.buffer, crossingOutputPointer, 1)[0], 0xa5,
  'an out-of-memory output span must be rejected before its first write');

// The portable ABI rejects if its canonical centered accumulator crosses I32
// at any K prefix, even when an algebraically rearranged final sum would fit.
// Keep the optional child fail-closed for the same extreme-but-valid shape.
const prefixK = 60000;
const prefixInput = new Uint8Array(prefixK).fill(33);
const prefixWeight = new Int8Array(prefixK).fill(123);
prefixInput.fill(12, 0, 766);
prefixWeight.fill(-98, 0, 766);
const prefixInputPointer = allocate(prefixInput.byteLength);
const prefixWeightPointer = allocate(prefixWeight.byteLength);
const prefixPackedBytes = Number(parent.exports.packed_q8_weight_size(prefixK, 1));
const prefixPackedPointer = allocate(prefixPackedBytes);
const prefixBiasPointer = allocate(4);
const prefixScalePointer = allocate(4);
const prefixZeroPointPointer = allocate(4);
const prefixChildOutput = allocate(1);
const prefixParentOutput = allocate(1);
writeBytes(prefixInputPointer, prefixInput);
writeBytes(prefixWeightPointer, prefixWeight);
writeBytes(prefixBiasPointer, Int32Array.of(2140852154));
writeBytes(prefixScalePointer, Float32Array.of(1));
writeBytes(prefixZeroPointPointer, Int32Array.of(-54));
assert.equal(parent.exports.pack_q8_weight(
  prefixPackedPointer, prefixPackedBytes, prefixWeightPointer,
  prefixK, 1, VX_DTYPE_I8, 1,
), 1);
new Uint8Array(memory.buffer, prefixChildOutput, 1).fill(0xa5);
new Uint8Array(memory.buffer, prefixParentOutput, 1).fill(0xa5);
assert.equal(parent.exports.qlinear_i8u8(
  prefixInputPointer, prefixWeightPointer, prefixBiasPointer,
  prefixScalePointer, prefixZeroPointPointer, prefixParentOutput,
  1, prefixK, 1, 1, 215, 1, 0,
  VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8,
), 0, 'the portable kernel must reject an overflowing canonical K prefix');
assert.equal(qlinear(
  prefixInputPointer, prefixWeightPointer, prefixPackedPointer,
  prefixBiasPointer, prefixScalePointer, prefixZeroPointPointer,
  prefixChildOutput, 1, prefixK, 1, 1, 215, 1, 0,
  VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8,
), 0, 'the child must preserve the portable prefix-overflow rejection');
assert.equal(new Uint8Array(memory.buffer, prefixParentOutput, 1)[0], 0xa5);
assert.equal(new Uint8Array(memory.buffer, prefixChildOutput, 1)[0], 0xa5);

const benchmarkK = 320;
const benchmarkN = 320;
const benchmarkInput = Uint8Array.from(
  { length: benchmarkK }, (_, index) => (index * 73 + 128) & 0xff,
);
const benchmarkWeight = Uint8Array.from(
  { length: benchmarkK * benchmarkN },
  (_, index) => (index * 29 + Math.floor(index / benchmarkK) * 17 + 128) & 0xff,
);
const benchmarkBias = new Int32Array(benchmarkN);
const benchmarkScales = new Float32Array(benchmarkN).fill(0.0078125);
const benchmarkZeroPoints = new Int32Array(benchmarkN);
const benchmarkInputF32 = Float32Array.from(
  benchmarkInput, (raw) => semanticByte(raw, VX_DTYPE_I8) * 0.25,
);
const benchmarkBiasF32 = new Float32Array(benchmarkN);
const benchmarkInputPointer = allocate(benchmarkK);
const benchmarkWeightPointer = allocate(benchmarkK * benchmarkN);
const benchmarkPackedBytes = Number(
  parent.exports.packed_q8_weight_size(benchmarkK, benchmarkN),
);
const benchmarkPackedPointer = allocate(benchmarkPackedBytes);
const benchmarkBiasPointer = allocate(benchmarkN * 4);
const benchmarkScalesPointer = allocate(benchmarkN * 4);
const benchmarkZeroPointsPointer = allocate(benchmarkN * 4);
const benchmarkRelaxedOutput = allocate(benchmarkN);
const benchmarkPackedOutput = allocate(benchmarkN);
const benchmarkPortableOutput = allocate(benchmarkN);
const benchmarkInputF32Pointer = allocate(benchmarkInputF32.byteLength);
const benchmarkBiasF32Pointer = allocate(benchmarkBiasF32.byteLength);
const benchmarkW8A32Output = allocate(benchmarkN * Float32Array.BYTES_PER_ELEMENT);
writeBytes(benchmarkInputPointer, benchmarkInput);
writeBytes(benchmarkWeightPointer, benchmarkWeight);
writeBytes(benchmarkBiasPointer, benchmarkBias);
writeBytes(benchmarkScalesPointer, benchmarkScales);
writeBytes(benchmarkZeroPointsPointer, benchmarkZeroPoints);
writeBytes(benchmarkInputF32Pointer, benchmarkInputF32);
writeBytes(benchmarkBiasF32Pointer, benchmarkBiasF32);
assert.equal(parent.exports.pack_q8_weight(
  benchmarkPackedPointer, benchmarkPackedBytes, benchmarkWeightPointer,
  benchmarkK, benchmarkN, VX_DTYPE_I8, 1,
), 1);

const relaxedCall = () => qlinear(
  benchmarkInputPointer, benchmarkWeightPointer, benchmarkPackedPointer,
  benchmarkBiasPointer, benchmarkScalesPointer, benchmarkZeroPointsPointer,
  benchmarkRelaxedOutput, 1, benchmarkK, benchmarkN,
  0.25, 0, 8, 0, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
);
const packedCall = () => parent.exports.qlinear_i8u8_packed(
  benchmarkInputPointer, benchmarkPackedPointer, benchmarkBiasPointer,
  benchmarkScalesPointer, benchmarkZeroPointsPointer, benchmarkPackedOutput,
  1, benchmarkK, benchmarkN, 0.25, 0, 8, 0,
  VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
);
const portableCall = () => parent.exports.qlinear_i8u8(
  benchmarkInputPointer, benchmarkWeightPointer, benchmarkBiasPointer,
  benchmarkScalesPointer, benchmarkZeroPointsPointer, benchmarkPortableOutput,
  1, benchmarkK, benchmarkN, 0.25, 0, 8, 0,
  VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
);
const w8a32Call = () => parent.exports.matmul_quantized_f32_packed(
  benchmarkInputF32Pointer, benchmarkPackedPointer, benchmarkScalesPointer,
  benchmarkZeroPointsPointer, benchmarkBiasF32Pointer, benchmarkW8A32Output,
  1, benchmarkK, benchmarkN, VX_DTYPE_I8, benchmarkN,
  VX_DTYPE_I32, benchmarkN,
);
assert.equal(relaxedCall(), 1);
assert.equal(packedCall(), 1);
assert.equal(portableCall(), 1);
assert.equal(w8a32Call(), 1);
assert.deepEqual(
  [...new Uint8Array(memory.buffer, benchmarkRelaxedOutput, benchmarkN)],
  [...new Uint8Array(memory.buffer, benchmarkPackedOutput, benchmarkN)],
  '320x320 Relaxed-SIMD benchmark output must match packed baseline',
);
assert.deepEqual(
  [...new Uint8Array(memory.buffer, benchmarkRelaxedOutput, benchmarkN)],
  [...new Uint8Array(memory.buffer, benchmarkPortableOutput, benchmarkN)],
  '320x320 Relaxed-SIMD benchmark output must match portable baseline',
);
const expectedW8A32 = Float32Array.from({ length: benchmarkN }, (_, column) => {
  let accumulator = 0;
  for (let k = 0; k < benchmarkK; k++) {
    accumulator += benchmarkInputF32[k] *
      semanticByte(
        benchmarkWeight[column * benchmarkK + k], VX_DTYPE_I8);
  }
  return accumulator * benchmarkScales[column];
});
assert.deepEqual(
  [...new Float32Array(memory.buffer, benchmarkW8A32Output, benchmarkN)],
  [...expectedW8A32],
  '320x320 packed W8A32 benchmark output must match the shared I8-weight oracle',
);

/* Prove the produced parent artifact selected compensated F32 SIMD rather
 * than silently benchmarking the scalar/double fallback. */
const probeK = 257;
const probeN = 8;
const probeInput = Float32Array.from({ length: probeK }, (_, index) =>
  Math.sin(index * 0.17) * 1.137 + 0.0031);
const probeWeight = Int8Array.from({ length: probeK * probeN }, (_, index) => {
  const column = Math.floor(index / probeK);
  const k = index % probeK;
  return ((column * 37 + k * 29 + 11) % 255) - 127;
});
const probeScale = Float32Array.from({ length: probeN }, (_, column) =>
  0.0037 + column * 0.00091);
const probeBias = Float32Array.from({ length: probeN }, (_, column) =>
  Math.cos(column * 0.11) * 0.37);
const probeInputPointer = allocate(probeInput.byteLength);
const probeWeightPointer = allocate(probeWeight.byteLength);
const probeScalePointer = allocate(probeScale.byteLength);
const probeBiasPointer = allocate(probeBias.byteLength);
const probePackedBytes = Number(parent.exports.packed_q8_weight_size(probeK, probeN));
const probePackedPointer = allocate(probePackedBytes);
const probeOutputPointer = allocate(probeN * Float32Array.BYTES_PER_ELEMENT);
writeBytes(probeInputPointer, probeInput);
writeBytes(probeWeightPointer, probeWeight);
writeBytes(probeScalePointer, probeScale);
writeBytes(probeBiasPointer, probeBias);
assert.equal(parent.exports.pack_q8_weight(
  probePackedPointer, probePackedBytes, probeWeightPointer,
  probeK, probeN, VX_DTYPE_I8, 1,
), 1);
assert.equal(parent.exports.matmul_quantized_f32_packed(
  probeInputPointer, probePackedPointer, probeScalePointer, 0,
  probeBiasPointer, probeOutputPointer, 1, probeK, probeN,
  VX_DTYPE_I8, probeN, 0, 0,
), 1);
const expectedCompensated = new Float32Array(probeN);
const scalarDouble = new Float32Array(probeN);
for (let column = 0; column < probeN; column++) {
  let accumulator = 0;
  let correction = 0;
  let doubleAccumulator = 0;
  for (let k = 0; k < probeK; k++) {
    const product = Math.fround(probeInput[k] * probeWeight[column * probeK + k]);
    const corrected = Math.fround(product - correction);
    const next = Math.fround(accumulator + corrected);
    correction = Math.fround(Math.fround(next - accumulator) - corrected);
    accumulator = next;
    doubleAccumulator += probeInput[k] * probeWeight[column * probeK + k];
  }
  expectedCompensated[column] = Math.fround(
    Math.fround(accumulator * probeScale[column]) + probeBias[column]);
  scalarDouble[column] = Math.fround(
    doubleAccumulator * probeScale[column] + probeBias[column]);
}
assert.notDeepEqual([...expectedCompensated], [...scalarDouble],
  'dispatch sentinel must distinguish compensated F32 from scalar/double');
assert.deepEqual(
  [...new Float32Array(memory.buffer, probeOutputPointer, probeN)],
  [...expectedCompensated],
  'the produced parent artifact must select compensated F32 SIMD',
);

function averageMilliseconds(call, warmup = 100, iterations = 1000) {
  for (let index = 0; index < warmup; index++) assert.equal(call(), 1);
  const start = performance.now();
  for (let index = 0; index < iterations; index++) assert.equal(call(), 1);
  return (performance.now() - start) / iterations;
}

const relaxedMs = averageMilliseconds(relaxedCall);
const packedMs = averageMilliseconds(packedCall);
const portableMs = averageMilliseconds(portableCall);
const w8a32Ms = averageMilliseconds(w8a32Call);
console.log(
  `Verified ${sectionName}: shared parent memory, final dot opcode, all 8 dtype combinations, fail-closed dispatch; 320x320 M=1 W8A8 relaxed=${relaxedMs.toFixed(4)} ms, W8A8 packed=${packedMs.toFixed(4)} ms, W8A8 portable=${portableMs.toFixed(4)} ms, W8A32 packed=${w8a32Ms.toFixed(4)} ms, W8A32/relaxed=${(w8a32Ms / relaxedMs).toFixed(2)}x.`,
);
