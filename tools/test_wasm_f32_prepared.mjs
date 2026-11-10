#!/usr/bin/env node
// Build with the pinned repository toolchain:
// docker run --rm -v "$PWD:/workspace" -w /workspace volvoxai-build:latest \
//   node tools/build_wasm_f32_fixture.mjs
// Then: node tools/test_wasm_f32_prepared.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const artifact = process.argv[2] ?? 'build/wasm/f32-fixture/f32.wasm';
const {instance} = await WebAssembly.instantiate(await readFile(artifact));
const a = instance.exports;
let allocations = [], checks = 0, linearCases = 0, lstmCases = 0;
function alloc(bytes) {
  const pointer = a.test_f32_alloc(bytes);
  assert.ok(pointer);
  allocations.push(pointer);
  return pointer;
}
function put(values) {
  const pointer = alloc(values.byteLength);
  new Uint8Array(a.memory.buffer, pointer, values.byteLength).set(
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  return pointer;
}
function clear() {
  a.test_f32_clear(0); a.test_f32_clear(1);
  for (const pointer of allocations) a.test_f32_free(pointer);
  allocations = [];
  assert.equal(a.test_f32_bytes(0), 0);
  assert.equal(a.test_f32_bytes(1), 0);
  assert.equal(a.test_f32_live_bytes(), 0, 'every allocation is released');
}
function equal(left, right, count, label) {
  const l = new Float32Array(a.memory.buffer, left, count);
  const r = new Float32Array(a.memory.buffer, right, count);
  const lb = new Uint32Array(a.memory.buffer, left, count);
  const rb = new Uint32Array(a.memory.buffer, right, count);
  for (let i = 0; i < count; i++) {
    if (Number.isNaN(l[i]) && Number.isNaN(r[i])) continue;
    assert.equal(lb[i], rb[i], `${label}[${i}] ${l[i]} != ${r[i]}`);
    checks++;
  }
}
let state = 711;
function values(count, scale = 1) {
  return Float32Array.from({length: count}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return ((state / 2 ** 32) * 2 - 1) * scale;
  });
}

for (const m of [1, 3, 4, 5, 9]) for (const k of [1, 7, 17, 257])
  for (const n of [1, 7, 8, 9, 16]) for (const outIn of [0, 1]) {
    const x = put(values(m * k)), w = put(values(k * n));
    const bias = put(values(n)), y = alloc(m * n * 4), reference = alloc(m * n * 4);
    const packed = a.test_f32_prepare(0, w, k, n, outIn);
    assert.ok(packed);
    const bytes = a.test_f32_bytes(0);
    assert.equal(a.test_f32_prepare(0, w, k, n, outIn), packed);
    assert.equal(a.test_f32_bytes(0), bytes);
    assert.equal(a.test_f32_find(0, w, k + 1, n, outIn), 0);
    for (let inputRevision = 0; inputRevision < 2; inputRevision++) {
      new Float32Array(a.memory.buffer, x, m * k).set(values(m * k));
      a[outIn ? 'matmul_f32_out_in' : 'matmul_f32'](x, w, bias, reference, m, k, n);
      assert.equal(a.vx_gemm_f32_run_packed(x, packed, bias, y, m, k, n), 1);
      equal(y, reference, m * n, `Linear ${m}/${k}/${n}/${outIn}`);
    }
    linearCases++;
    clear();
  }

// The edge prologue must retain a negative-zero bias; mixed nonfinite lanes
// cannot affect adjacent outputs. A bias at memory end catches tail overreads.
for (const m of [1, 3, 4, 5]) for (const n of [1, 7, 8, 9]) {
  const k = 3, x = put(new Float32Array(m * k).fill(-0));
  const w = put(new Float32Array(k * n).fill(1));
  const packed = a.test_f32_prepare(0, w, k, n, 1);
  const y = alloc(m * n * 4), ref = alloc(m * n * 4);
  const bias = a.memory.buffer.byteLength - n * 4;
  new Float32Array(a.memory.buffer, bias, n).fill(-0);
  a.matmul_f32_out_in(x, w, bias, ref, m, k, n);
  assert.equal(a.vx_gemm_f32_run_packed(x, packed, bias, y, m, k, n), 1);
  equal(y, ref, m * n, 'signed zero and bias bounds');
  new Float32Array(a.memory.buffer, x, m * k).set([Infinity, -Infinity, NaN]);
  a.matmul_f32_out_in(x, w, bias, ref, m, k, n);
  assert.equal(a.vx_gemm_f32_run_packed(x, packed, bias, y, m, k, n), 1);
  equal(y, ref, m * n, 'nonfinite');
  clear();
}

for (const direction of [0, 1, 2]) for (const hidden of [1, 2, 7, 8])
  for (const [time, batch, width] of [[1, 1, 1], [3, 3, 9], [7, 1, 5], [5, 5, 3]]) {
    const directions = direction === 2 ? 2 : 1;
    const x = put(values(time * batch * width, .5));
    const w = put(values(directions * 4 * hidden * width, .1));
    const r = put(values(directions * 4 * hidden * hidden, .1));
    const bias = put(values(directions * 8 * hidden, .1));
    const h = put(values(directions * batch * hidden, .1));
    const c = put(values(directions * batch * hidden, .1));
    const count = time * directions * batch * hidden;
    const output = alloc(count * 4), reference = alloc(count * 4);
    const packedW = [], packedR = [];
    for (let d = 0; d < directions; d++) {
      packedW.push(a.test_f32_prepare(0, w + d * 4 * hidden * width * 4, width, 4 * hidden, 1));
      packedR.push(a.test_f32_prepare(0, r + d * 4 * hidden * hidden * 4, hidden, 4 * hidden, 1));
      assert.ok(packedW[d] && packedR[d]);
    }
    const wp = put(new Uint32Array(packedW)), rp = put(new Uint32Array(packedR));
    for (let revision = 0; revision < 2; revision++) {
      new Float32Array(a.memory.buffer, x, time * batch * width).set(values(time * batch * width, .5));
      const initialH = revision ? h : 0, initialC = revision ? c : 0;
      const before = a.test_f32_live_bytes();
      assert.equal(a.lstm_f32(x, w, r, bias, initialH, initialC, reference,
        time, batch, width, hidden, direction), 1);
      assert.equal(a.lstm_f32_packed(x, w, r, bias, initialH, initialC, output,
        time, batch, width, hidden, direction, wp, rp), 1);
      equal(output, reference, count, `LSTM ${time}/${batch}/${width}/${hidden}/${direction}`);
      assert.equal(a.test_f32_live_bytes(), before, 'LSTM frees invocation workspace');
    }
    lstmCases++;
    clear();
  }

// Stores belong to independent engines. Clearing/replacing one cannot alter
// another; mutable graph values are handled by the runtime's uncached route.
{
  const w = put(values(8 * 9)), x = put(values(9));
  const y = alloc(8 * 4), ref = alloc(8 * 4);
  const first = a.test_f32_prepare(0, w, 9, 8, 1);
  const second = a.test_f32_prepare(1, w, 9, 8, 1);
  assert.ok(first && second && first !== second);
  a.test_f32_clear(0);
  assert.equal(a.test_f32_find(0, w, 9, 8, 1), 0);
  assert.equal(a.test_f32_find(1, w, 9, 8, 1), second);
  a.matmul_f32_out_in(x, w, 0, ref, 1, 9, 8);
  a.vx_gemm_f32_run_packed(x, second, 0, y, 1, 9, 8);
  equal(y, ref, 8, 'independent owner');
  a.test_f32_clear(1);
  new Float32Array(a.memory.buffer, w, 72).set(values(72));
  const changed = a.test_f32_prepare(0, w, 9, 8, 1);
  a.matmul_f32_out_in(x, w, 0, ref, 1, 9, 8);
  a.vx_gemm_f32_run_packed(x, changed, 0, y, 1, 9, 8);
  equal(y, ref, 8, 'weight revision after invalidation');
  assert.equal(a.test_f32_prepare(0, w, 1 << 24, 8, 1), 0, 'bounded fallback before source read');
  clear();
}
// Admission remains valid when validation's pack allocations fail but a later
// owner can allocate successfully. Test each allocation boundary and the cap;
// no failed store allocation, descriptor, or panel may survive the fallback.
{
  const w = put(values(17 * 9));
  const before = a.test_f32_live_bytes();
  const bound = a.test_f32_bound_add(0, 17, 9);
  assert.ok(bound > 17 * 16 * 4);
  for (let failure = 0; failure < 3; failure++) {
    a.test_f32_fail_after(failure);
    assert.equal(a.test_f32_prepare(0, w, 17, 9, 1), 0);
    assert.equal(a.test_f32_bytes(0), 0);
    assert.equal(a.test_f32_live_bytes(), before);
    assert.equal(a.test_f32_bound_add(0, 17, 9), bound);
    a.test_f32_fail_after(-1);
    assert.ok(a.test_f32_prepare(0, w, 17, 9, 1));
    assert.equal(a.test_f32_bytes(0), bound);
    a.test_f32_clear(0);
  }
  assert.equal(a.test_f32_bound_add(0, 1 << 24, 8), 0);
  let capped = 0;
  for (let i = 0; i < 100; i++) capped = a.test_f32_bound_add(capped, 1024, 1024);
  assert.equal(capped, 64 * 1024 * 1024);
  clear();
}
console.log(JSON.stringify({linearCases, lstmCases, bitExactChecks: checks,
  zeroSignAndTailBounds: 'PASS', dynamicInputs: 'PASS', ownerRelease: 'PASS',
  allocationFailureAndBound: 'PASS'}));
