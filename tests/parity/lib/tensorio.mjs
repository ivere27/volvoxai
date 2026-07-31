// Typed raw tensor I/O for the parity harness.
// Mirrors the native CLI's raw format: a flat little-endian buffer whose suffix
// names the dtype (.f32/.i32/.u8/.i8). No headers, no shape — shape lives
// in policy.json / graph.json, exactly like `native/volvoxai run --input`.
import fs from 'fs';

const CTOR = {
  f32: Float32Array,
  i32: Int32Array,
  u8: Uint8Array,
  i8: Int8Array,
};

export function ctorFor(dtype) {
  const C = CTOR[dtype];
  if (!C) throw new Error(`unsupported parity dtype: ${dtype}`);
  return C;
}

export function readTensor(file, dtype) {
  const C = ctorFor(dtype);
  const b = fs.readFileSync(file);
  return new C(b.buffer, b.byteOffset, b.byteLength / C.BYTES_PER_ELEMENT);
}

export function writeTensor(file, dtype, arr) {
  const C = ctorFor(dtype);
  const view = arr instanceof C ? arr : C.from(arr);
  fs.writeFileSync(file, Buffer.from(view.buffer, view.byteOffset, view.byteLength));
}

// Deterministic uint8 input (LCG) so every tier/job feeds identical bytes without
// committing a large fixture. Same seed → same bytes everywhere.
export function seededUint8(n, seed) {
  const out = new Uint8Array(n);
  let s = (seed >>> 0) || 1;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = (s >>> 24) & 0xff;
  }
  return out;
}

// Deterministic float32 input in [lo, hi) — same LCG, so a fixture written here
// is byte-identical to what native and the Python external oracle read back.
export function seededFloat32(n, seed, lo = 0, hi = 1) {
  const out = new Float32Array(n);
  let s = (seed >>> 0) || 1;
  const span = hi - lo;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = lo + (s >>> 8) / 0x1000000 * span; // 24-bit mantissa fraction
  }
  return out;
}
