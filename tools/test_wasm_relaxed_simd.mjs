#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const SECTION_NAME = 'volvoxai.relaxed_simd.v1';
const RELAXED_DOT_OPCODE = Buffer.from([0xfd, 0x93, 0x02]);
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;
const OUTPUT_SENTINEL = 0xa5;

const baselineOnly = process.argv[2] === '--baseline-only';
const artifactPaths = process.argv.slice(baselineOnly ? 3 : 2);
if (artifactPaths.length === 0) {
  throw new Error(
    'usage: test_wasm_relaxed_simd.mjs [--baseline-only] <parent.wasm> [...]',
  );
}

const embeddedChildren = [];
for (const artifactPath of artifactPaths) {
  const bytes = await readFile(artifactPath);
  assert.equal(
    WebAssembly.validate(bytes),
    true,
    `${artifactPath} must remain valid without Relaxed-SIMD support`,
  );
  const parentModule = new WebAssembly.Module(bytes);
  const sections = WebAssembly.Module.customSections(parentModule, SECTION_NAME);
  assert.equal(sections.length, 1, `${artifactPath} must embed exactly one child`);
  const child = Buffer.from(sections[0]);
  assert.equal(child.subarray(0, 8).toString('hex'), '0061736d01000000');
  assert.notEqual(
    child.indexOf(RELAXED_DOT_OPCODE),
    -1,
    'the child must contain the final Relaxed-SIMD dot opcode',
  );
  embeddedChildren.push(child);
}
for (const child of embeddedChildren.slice(1)) {
  assert.deepEqual(
    child,
    embeddedChildren[0],
    'inference and full profiles must embed identical accelerator bytes',
  );
}

if (baselineOnly) {
  console.log(`Verified baseline parents and ${SECTION_NAME} embedding.`);
  process.exit(0);
}

const malformedChild = Buffer.from(embeddedChildren[0]);
malformedChild[0] ^= 0xff;
assert.equal(WebAssembly.validate(malformedChild), false);
await assert.rejects(
  WebAssembly.compile(malformedChild),
  WebAssembly.CompileError,
  'a malformed optional child must never become executable',
);

const childModule = new WebAssembly.Module(embeddedChildren[0]);
assert.deepEqual(WebAssembly.Module.imports(childModule), [
  { module: 'env', name: 'memory', kind: 'memory' },
]);
assert.deepEqual(WebAssembly.Module.exports(childModule), [
  { name: 'qlinear_i8u8_relaxed', kind: 'function' },
]);

import { wasmToolImports } from './wasm_host_imports.mjs';

const parentModule = new WebAssembly.Module(await readFile(artifactPaths[0]));
const parent = await WebAssembly.instantiate(parentModule, wasmToolImports());
const memory = parent.exports.memory;
assert.ok(memory instanceof WebAssembly.Memory);
const child = await WebAssembly.instantiate(childModule, { env: { memory } });
const qlinear = child.exports.qlinear_i8u8_relaxed;
assert.equal(typeof qlinear, 'function');
assert.equal(qlinear.length, 17, 'the child must expose the v1 descriptor ABI');
assert.equal(
  qlinear(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
  0,
  'a zero descriptor must fail closed',
);

function allocate(byteLength) {
  const pointer = Number(parent.exports.alloc_bytes(byteLength));
  assert.ok(Number.isSafeInteger(pointer) && pointer > 0);
  const required = pointer + byteLength;
  if (required > memory.buffer.byteLength) {
    memory.grow(Math.ceil((required - memory.buffer.byteLength) / 65536));
  }
  return pointer;
}

function write(pointer, values) {
  new Uint8Array(memory.buffer, pointer, values.byteLength).set(
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
  );
}

function outputBytes(pointer, count) {
  return [...new Uint8Array(memory.buffer, pointer, count)];
}

// Odd K and N exercise both the K tail and the final packed N8 panel. Raw
// bytes deliberately cross both signed/unsigned boundaries.
const dIn = 17;
const dOut = 9;
const input = Uint8Array.from(
  { length: dIn },
  (_, index) => [0, 255, 128, 127, 3, 249, 64][index % 7],
);
const weights = Uint8Array.from(
  { length: dIn * dOut },
  (_, index) => [255, 0, 128, 127, 17, 239, 64, 192][(index * 5 + 3) % 8],
);
const bias = Int32Array.from({ length: dOut }, (_, index) => 13 - index * 3);
const scales = Float32Array.from(
  { length: dOut },
  (_, index) => [0.25, 0.5, 1, 2][index % 4],
);
const signedZeroPoints = Int32Array.from(
  { length: dOut },
  (_, index) => [-91, -7, 0, 63, 117][index % 5],
);
const unsignedZeroPoints = Int32Array.from(
  { length: dOut },
  (_, index) => [3, 97, 128, 211, 251][index % 5],
);

const inputPointer = allocate(input.byteLength);
const weightPointer = allocate(weights.byteLength);
const packedBytes = Number(parent.exports.packed_q8_weight_size(dIn, dOut));
assert.ok(packedBytes > 0);
const packedPointer = allocate(packedBytes);
const biasPointer = allocate(bias.byteLength);
const scalePointer = allocate(scales.byteLength);
const zeroPointPointer = allocate(unsignedZeroPoints.byteLength);
const portableOutputPointer = allocate(dOut);
const childOutputPointer = allocate(dOut);
write(inputPointer, input);
write(weightPointer, weights);
write(biasPointer, bias);
write(scalePointer, scales);

function runDescriptor(inputDtype, weightDtype, outputDtype) {
  const weightZeroPoints = weightDtype === VX_DTYPE_I8
    ? signedZeroPoints
    : unsignedZeroPoints;
  const inputZeroPoint = inputDtype === VX_DTYPE_I8 ? -37 : 139;
  const outputZeroPoint = outputDtype === VX_DTYPE_I8 ? -19 : 173;
  write(zeroPointPointer, weightZeroPoints);
  assert.equal(
    parent.exports.pack_q8_weight(
      packedPointer,
      packedBytes,
      weightPointer,
      dIn,
      dOut,
      weightDtype,
      1,
    ),
    1,
  );
  new Uint8Array(memory.buffer, portableOutputPointer, dOut).fill(OUTPUT_SENTINEL);
  new Uint8Array(memory.buffer, childOutputPointer, dOut).fill(OUTPUT_SENTINEL);
  assert.equal(parent.exports.qlinear_i8u8(
    inputPointer,
    weightPointer,
    biasPointer,
    scalePointer,
    zeroPointPointer,
    portableOutputPointer,
    1,
    dIn,
    dOut,
    0.5,
    inputZeroPoint,
    0.25,
    outputZeroPoint,
    inputDtype,
    weightDtype,
    outputDtype,
  ), 1);
  assert.equal(qlinear(
    inputPointer,
    weightPointer,
    packedPointer,
    biasPointer,
    scalePointer,
    zeroPointPointer,
    childOutputPointer,
    1,
    dIn,
    dOut,
    0.5,
    inputZeroPoint,
    0.25,
    outputZeroPoint,
    inputDtype,
    weightDtype,
    outputDtype,
  ), 1);
  assert.deepEqual(
    outputBytes(childOutputPointer, dOut),
    outputBytes(portableOutputPointer, dOut),
    `Relaxed-SIMD must byte-match portable C for ${inputDtype}/${weightDtype}/${outputDtype}`,
  );
}

for (const inputDtype of [VX_DTYPE_I8, VX_DTYPE_U8]) {
  for (const weightDtype of [VX_DTYPE_I8, VX_DTYPE_U8]) {
    for (const outputDtype of [VX_DTYPE_I8, VX_DTYPE_U8]) {
      runDescriptor(inputDtype, weightDtype, outputDtype);
    }
  }
}

// Corrupt the exact pack used by the successful U8 route. Rejection must
// precede every output store, and restoring the bytes must make the same child
// instance usable again.
runDescriptor(VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8);
const header = new DataView(memory.buffer, packedPointer, 48);
const magic = header.getUint32(0, true);
header.setUint32(0, 0, true);
new Uint8Array(memory.buffer, childOutputPointer, dOut).fill(OUTPUT_SENTINEL);
assert.equal(qlinear(
  inputPointer,
  weightPointer,
  packedPointer,
  biasPointer,
  scalePointer,
  zeroPointPointer,
  childOutputPointer,
  1,
  dIn,
  dOut,
  0.5,
  139,
  0.25,
  173,
  VX_DTYPE_U8,
  VX_DTYPE_U8,
  VX_DTYPE_U8,
), 0);
assert.deepEqual(outputBytes(childOutputPointer, dOut), Array(dOut).fill(OUTPUT_SENTINEL));
header.setUint32(0, magic, true);
runDescriptor(VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8);

console.log(
  `Verified ${SECTION_NAME}: shared parent memory, 8 asymmetric I8/U8 descriptors, odd K/N tails, byte-exact portable parity, and fail-closed reuse.`,
);
