#!/usr/bin/env node
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';

const artifactPath = process.argv[2];
if (!artifactPath) {
  throw new Error('usage: benchmark_wasm_w8a8_seed.mjs <volvoxai.wasm>');
}

const bytes = await readFile(artifactPath);
assert.equal(WebAssembly.validate(bytes), true,
  `${artifactPath} must be a baseline-valid WebAssembly module`);
const module = new WebAssembly.Module(bytes);
assert.equal(
  WebAssembly.Module.customSections(module, 'volvoxai.relaxed_simd.v1').length,
  1,
  'the fixed parent must contain exactly one optional Relaxed-SIMD child',
);
const math = {
  expf: Math.exp,
  logf: Math.log,
  powf: Math.pow,
  sqrtf: Math.sqrt,
  tanhf: Math.tanh,
  sinf: Math.sin,
  cosf: Math.cos,
};
const instance = await WebAssembly.instantiate(module, { env: math, math });
const api = instance.exports;
const memory = api.memory;
assert.ok(memory instanceof WebAssembly.Memory);

const I8 = 2;
const inputScale = 1 / 32;
const inputZeroPoint = 0;
const outputScale = 1 / 16;
const outputZeroPoint = 0;

function resetHeap() {
  assert.equal(typeof api.reset_heap, 'function');
  api.reset_heap();
}

function allocate(byteLength) {
  const pointer = Number(api.alloc_bytes(byteLength));
  assert.ok(Number.isSafeInteger(pointer) && pointer > 0,
    `could not allocate ${byteLength} bytes`);
  const required = pointer + byteLength;
  if (required > memory.buffer.byteLength) {
    memory.grow(Math.ceil((required - memory.buffer.byteLength) / 65536));
  }
  return pointer;
}

function write(pointer, value) {
  new Uint8Array(memory.buffer, pointer, value.byteLength).set(
    new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
  );
}

function checksum(pointer, elements) {
  const values = new Uint8Array(memory.buffer, pointer, elements);
  let result = 0x811c9dc5;
  for (const value of values) {
    result ^= value;
    result = Math.imul(result, 0x01000193) >>> 0;
  }
  return `0x${result.toString(16).padStart(8, '0')}`;
}

function assertEqualBytes(leftPointer, rightPointer, elements, label) {
  const left = new Uint8Array(memory.buffer, leftPointer, elements);
  const right = new Uint8Array(memory.buffer, rightPointer, elements);
  for (let index = 0; index < elements; index++) {
    assert.equal(left[index], right[index], `${label} differs at byte ${index}`);
  }
}

function benchmark(call, { minimumMs = 150, maximumIterations = 20 } = {}) {
  assert.equal(call(), 1, 'benchmark warmup rejected its descriptor');
  let iterations = 0;
  const start = performance.now();
  do {
    assert.equal(call(), 1, 'benchmark call rejected its descriptor');
    iterations++;
  } while (iterations < maximumIterations && performance.now() - start < minimumMs);
  return (performance.now() - start) / iterations;
}

function roundTiesEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function requantizeI8(accumulator, weightScale) {
  const productScale = Math.fround(Math.fround(inputScale) * weightScale);
  const multiplier = Math.fround(productScale / Math.fround(outputScale));
  const scaled = Math.fround(Math.fround(accumulator) * multiplier);
  const transformed = Math.fround(scaled + Math.fround(outputZeroPoint));
  return Math.max(-128, Math.min(127, roundTiesEven(transformed)));
}

function qconvOracle({ input, weight, bias, scales, zeroPoints, inputHeight,
  inputWidth, inputChannels, outputHeight, outputWidth, outputChannels,
  kernelHeight, kernelWidth, strideY, strideX, paddingTop, paddingLeft }) {
  const output = new Int8Array(outputHeight * outputWidth * outputChannels);
  for (let outputY = 0; outputY < outputHeight; outputY++) {
    for (let outputX = 0; outputX < outputWidth; outputX++) {
      for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
        let accumulator = bias[outputChannel];
        const weightBase = outputChannel * kernelHeight * kernelWidth * inputChannels;
        for (let kernelY = 0; kernelY < kernelHeight; kernelY++) {
          const inputY = outputY * strideY + kernelY - paddingTop;
          if (inputY < 0 || inputY >= inputHeight) continue;
          for (let kernelX = 0; kernelX < kernelWidth; kernelX++) {
            const inputX = outputX * strideX + kernelX - paddingLeft;
            if (inputX < 0 || inputX >= inputWidth) continue;
            const inputBase = (inputY * inputWidth + inputX) * inputChannels;
            const kernelBase = weightBase +
              (kernelY * kernelWidth + kernelX) * inputChannels;
            for (let channel = 0; channel < inputChannels; channel++) {
              accumulator += (input[inputBase + channel] - inputZeroPoint) *
                (weight[kernelBase + channel] - zeroPoints[outputChannel]);
            }
          }
        }
        output[(outputY * outputWidth + outputX) * outputChannels + outputChannel] =
          requantizeI8(accumulator, scales[outputChannel]);
      }
    }
  }
  return output;
}

function runQConvCase({ label, inputChannels }) {
  const inputHeight = 320;
  const inputWidth = 672;
  const outputHeight = 160;
  const outputWidth = 336;
  const outputChannels = 48;
  const kernelHeight = 3;
  const kernelWidth = 3;
  const inputElements = inputHeight * inputWidth * inputChannels;
  const weightElements = outputChannels * kernelHeight * kernelWidth * inputChannels;
  const outputElements = outputHeight * outputWidth * outputChannels;
  const input = Int8Array.from({ length: inputElements },
    (_, index) => (index * 29 % 221) - 110);
  const weight = Int8Array.from({ length: weightElements },
    (_, index) => (index * 37 % 211) - 105);
  const bias = Int32Array.from({ length: outputChannels },
    (_, column) => column * 41 - 503);
  const scales = Float32Array.from({ length: outputChannels },
    (_, column) => 1 / (48 + column % 7 * 8));
  const zeroPoints = new Int32Array(outputChannels);
  const expected = qconvOracle({
    input, weight, bias, scales, zeroPoints, inputHeight, inputWidth,
    inputChannels, outputHeight, outputWidth, outputChannels, kernelHeight,
    kernelWidth, strideY: 2, strideX: 2, paddingTop: 1, paddingLeft: 1,
  });

  resetHeap();
  const inputPointer = allocate(input.byteLength);
  const weightPointer = allocate(weight.byteLength);
  const biasPointer = allocate(bias.byteLength);
  const scalesPointer = allocate(scales.byteLength);
  const zeroPointsPointer = allocate(zeroPoints.byteLength);
  const portableOutputPointer = allocate(outputElements);
  const packedOutputPointer = allocate(outputElements);
  const packedDIn = kernelHeight * kernelWidth * inputChannels;
  const columnsPointer = allocate(outputHeight * outputWidth * packedDIn);
  const packedBytes = Number(api.packed_q8_weight_size(packedDIn, outputChannels));
  const packedWeightPointer = allocate(packedBytes);
  write(inputPointer, input);
  write(weightPointer, weight);
  write(biasPointer, bias);
  write(scalesPointer, scales);
  write(zeroPointsPointer, zeroPoints);
  assert.equal(api.pack_q8_weight(
    packedWeightPointer, packedBytes, weightPointer,
    packedDIn, outputChannels, I8, 1,
  ), 1, `${label} weight packing failed`);

  const portableCall = () => api.qconv2d_i8u8(
    inputPointer, weightPointer, biasPointer, scalesPointer, zeroPointsPointer,
    portableOutputPointer, 1, inputHeight, inputWidth, inputChannels,
    outputHeight, outputWidth, outputChannels, kernelHeight, kernelWidth,
    inputChannels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
    inputScale, inputZeroPoint, outputScale, outputZeroPoint, I8, I8, I8,
  );
  const im2colCall = () => api.qconv2d_im2col_i8u8?.(
    inputPointer, columnsPointer, biasPointer, zeroPointsPointer,
    1, inputHeight, inputWidth, inputChannels,
    outputHeight, outputWidth, outputChannels, kernelHeight, kernelWidth,
    2, 2, 1, 1, 1, 1, 1, 1, inputZeroPoint, I8,
    I8,
  ) ?? 0;
  const packedGemmCall = () => api.qlinear_i8u8_packed(
    columnsPointer, packedWeightPointer, biasPointer, scalesPointer,
    zeroPointsPointer, packedOutputPointer, outputHeight * outputWidth,
    packedDIn, outputChannels, inputScale, inputZeroPoint,
    outputScale, outputZeroPoint, I8, I8, I8,
  );
  const im2colPackedCall = () => im2colCall() === 1 ? packedGemmCall() : 0;

  assert.equal(portableCall(), 1, `${label} portable QConv rejected its descriptor`);
  const actual = new Int8Array(memory.buffer, portableOutputPointer, outputElements);
  for (let index = 0; index < outputElements; index++) {
    assert.equal(actual[index], expected[index], `${label} differs from its JS oracle at ${index}`);
  }
  const portableMs = benchmark(portableCall, { minimumMs: 200, maximumIterations: 5 });
  if (typeof api.qconv2d_im2col_i8u8 !== 'function') {
    return {
      label, inputChannels, portableMs, packedAvailable: false,
      checksum: checksum(portableOutputPointer, outputElements),
    };
  }
  assert.equal(im2colPackedCall(), 1,
    `${label} im2col plus baseline-packed QLinear rejected its descriptor`);
  assertEqualBytes(portableOutputPointer, packedOutputPointer, outputElements,
    `${label} im2col plus packed QLinear`);
  const im2colMs = benchmark(im2colCall, { minimumMs: 100, maximumIterations: 10 });
  const packedGemmMs = benchmark(packedGemmCall, { minimumMs: 150, maximumIterations: 10 });
  const im2colPackedMs = benchmark(im2colPackedCall,
    { minimumMs: 200, maximumIterations: 10 });
  assertEqualBytes(portableOutputPointer, packedOutputPointer, outputElements,
    `${label} timed im2col plus packed QLinear`);
  return {
    label, inputChannels, portableMs, packedAvailable: true,
    im2colMs, packedGemmMs, im2colPackedMs,
    checksum: checksum(packedOutputPointer, outputElements),
  };
}

const linearCases = [
  { label: 'encoder/cross attention', rows: 402, dIn: 320, dOut: 320, calls: 32 },
  { label: 'encoder FFN up', rows: 402, dIn: 320, dOut: 1280, calls: 6 },
  { label: 'encoder FFN down', rows: 402, dIn: 1280, dOut: 320, calls: 6 },
  { label: 'decoder attention', rows: 192, dIn: 320, dOut: 320, calls: 24 },
  { label: 'decoder FFN up', rows: 192, dIn: 320, dOut: 1280, calls: 4 },
  { label: 'decoder FFN down', rows: 192, dIn: 1280, dOut: 320, calls: 4 },
];

function runLinearCase(test) {
  const input = Int8Array.from({ length: test.rows * test.dIn },
    (_, index) => (index * 29 % 221) - 110);
  const weight = Int8Array.from({ length: test.dOut * test.dIn },
    (_, index) => (index * 37 % 211) - 105);
  const bias = Int32Array.from({ length: test.dOut },
    (_, column) => column * 41 - 503);
  const scales = Float32Array.from({ length: test.dOut },
    (_, column) => 1 / (48 + column % 7 * 8));
  const zeroPoints = new Int32Array(test.dOut);
  const outputElements = test.rows * test.dOut;

  resetHeap();
  const inputPointer = allocate(input.byteLength);
  const weightPointer = allocate(weight.byteLength);
  const biasPointer = allocate(bias.byteLength);
  const scalesPointer = allocate(scales.byteLength);
  const zeroPointsPointer = allocate(zeroPoints.byteLength);
  const referencePointer = allocate(outputElements);
  const packedOutputPointer = allocate(outputElements);
  const packedBytes = Number(api.packed_q8_weight_size(test.dIn, test.dOut));
  const packedPointer = allocate(packedBytes);
  write(inputPointer, input);
  write(weightPointer, weight);
  write(biasPointer, bias);
  write(scalesPointer, scales);
  write(zeroPointsPointer, zeroPoints);
  assert.equal(api.pack_q8_weight(
    packedPointer, packedBytes, weightPointer, test.dIn, test.dOut, I8, 1,
  ), 1, `${test.label} weight packing failed`);

  const portableCall = () => api.qlinear_i8u8(
    inputPointer, weightPointer, biasPointer, scalesPointer, zeroPointsPointer,
    referencePointer, test.rows, test.dIn, test.dOut,
    inputScale, inputZeroPoint, outputScale, outputZeroPoint, I8, I8, I8,
  );
  const packedCall = () => api.qlinear_i8u8_packed(
    inputPointer, packedPointer, biasPointer, scalesPointer, zeroPointsPointer,
    packedOutputPointer, test.rows, test.dIn, test.dOut,
    inputScale, inputZeroPoint, outputScale, outputZeroPoint, I8, I8, I8,
  );
  assert.equal(portableCall(), 1, `${test.label} portable QLinear rejected its descriptor`);
  assert.equal(packedCall(), 1, `${test.label} packed QLinear rejected its descriptor`);
  assertEqualBytes(referencePointer, packedOutputPointer, outputElements, test.label);
  const packedMs = benchmark(packedCall);
  const portableMs = benchmark(portableCall, { minimumMs: 100, maximumIterations: 8 });
  assertEqualBytes(referencePointer, packedOutputPointer, outputElements, test.label);
  return {
    ...test,
    packedMs,
    portableMs,
    checksum: checksum(packedOutputPointer, outputElements),
  };
}

console.log(`WASM TinyReceipt exact seed benchmark: ${artifactPath}`);
console.log('QConv labels: canonical portable parent vs im2col + explicit baseline SIMD128 packed QLinear; parity oracle: independent JavaScript I8 implementation.');
for (const result of [
  runQConvCase({ label: 'materialized grayscale stem', inputChannels: 1 }),
  runQConvCase({ label: 'RGB stem deployment proxy', inputChannels: 3 }),
]) {
  let message =
    `  ${result.label}: B=1 H=320 W=672 C=${result.inputChannels} ` +
    `K=3x3 O=48 S=2 -> 160x336x48 portable=${result.portableMs.toFixed(3)} ms`;
  if (result.packedAvailable) {
    message += ` im2col+packed=${result.im2colPackedMs.toFixed(3)} ms ` +
      `(layout=${result.im2colMs.toFixed(3)}, packed-gemm=${result.packedGemmMs.toFixed(3)}) ` +
      `speedup=${(result.portableMs / result.im2colPackedMs).toFixed(2)}x`;
  } else {
    message += ' im2col+packed=unavailable';
  }
  console.log(`${message} parity=pass checksum=${result.checksum}`);
}

console.log('QLinear label: explicit baseline SIMD128 packed kernel vs authoritative raw portable kernel.');
let packedWeightedMs = 0;
let portableWeightedMs = 0;
for (const result of linearCases.map(runLinearCase)) {
  packedWeightedMs += result.packedMs * result.calls;
  portableWeightedMs += result.portableMs * result.calls;
  console.log(
    `  ${result.label}: calls=${result.calls} M=${result.rows} K=${result.dIn} N=${result.dOut} ` +
    `baseline-packed=${result.packedMs.toFixed(3)} ms portable=${result.portableMs.toFixed(3)} ms ` +
    `speedup=${(result.portableMs / result.packedMs).toFixed(2)}x parity=pass ` +
    `checksum=${result.checksum}`,
  );
}
console.log(
  `  exact weighted base-dense subset: baseline-packed=${packedWeightedMs.toFixed(3)} ms ` +
  `portable=${portableWeightedMs.toFixed(3)} ms ` +
  `speedup=${(portableWeightedMs / packedWeightedMs).toFixed(2)}x`,
);
console.log(
  'Relaxed-SIMD child: embedded but not used here; its v1 QLinear ABI accepts only M=1, while every seed case above has M=192 or M=402.',
);
