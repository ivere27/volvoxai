// Per-op parity cases (Level 1). Each case is one op as a single-node graph, run
// on every backend and checked against pure-JS (consistency) + PyTorch (correctness).
//
// Family 1: weightless elementwise. Ranges are chosen to exercise the nonlinearity
// (negatives + positives) and keep Div's divisor away from zero. Extend op-family by
// family; MatMul/LayerNorm/Conv (weighted) come next.
import fs from 'fs';
import path from 'path';
import { seededFloat32, writeTensor } from '../lib/tensorio.mjs';
import { singleNodeGraph, writeGraph, writeEmptySafetensors, ensureDir } from '../lib/authorpkg.mjs';

export const cases = [
  { id: 'relu_1x64', op: 'ReLU', kind: 'unary', shape: [1, 64] },
  { id: 'gelu_1x64', op: 'GELU', kind: 'unary', shape: [1, 64] },
  { id: 'silu_1x64', op: 'SiLU', kind: 'unary', shape: [1, 64] },
  // Sigmoid: ts/ops/sigmoid.ts is exact 1/(1+e^-x); the WASM/native C kernels use a
  // fast approximation (~8e-3 vs exact). Documented, not hidden — the PyTorch oracle
  // matches the exact pure-JS reference, confirming which tier is approximate.
  { id: 'sigmoid_1x64', op: 'Sigmoid', kind: 'unary', shape: [1, 64], approxTol: { atol: 1.2e-2 } },
  // These six runtime registrations lack native-cpu exporter qualification
  // (`+Q`) in generated v1. Strict native package admission rejects them by
  // design; expected skips are not portable-coverage passes.
  { id: 'tanh_1x64', op: 'Tanh', kind: 'unary', shape: [1, 64], skip: ['native-cpu'] },
  { id: 'add_1x64', op: 'Add', kind: 'binary', shape: [1, 64] },
  { id: 'mul_1x64', op: 'Mul', kind: 'binary', shape: [1, 64] },
  { id: 'sub_1x64', op: 'Sub', kind: 'binary', shape: [1, 64] },
  { id: 'div_1x64', op: 'Div', kind: 'binary', shape: [1, 64] },
  // Activations with params / fixed hard variants — all exact torch equivalents.
  { id: 'leakyrelu_1x64', op: 'LeakyReLU', kind: 'unary', shape: [1, 64], params: { alpha: 0.1 }, skip: ['native-cpu'] },
  { id: 'clip_1x64', op: 'Clip', kind: 'unary', shape: [1, 64], params: { min: -1, max: 1 } },
  { id: 'hardsigmoid_1x64', op: 'HardSigmoid', kind: 'unary', shape: [1, 64], skip: ['native-cpu'] },
  { id: 'hardswish_1x64', op: 'HardSwish', kind: 'unary', shape: [1, 64], skip: ['native-cpu'] },
  // Sin/Cos also have no WebGPU registration or native GPU qualification.
  { id: 'sin_1x64', op: 'Sin', kind: 'unary', shape: [1, 64], skip: ['native-cpu', 'webgpu', 'native-vulkan', 'native-opengl'] },
  { id: 'cos_1x64', op: 'Cos', kind: 'unary', shape: [1, 64], skip: ['native-cpu', 'webgpu', 'native-vulkan', 'native-opengl'] },
];

// Deterministic per-case seeds so every backend + the torch oracle see identical bytes.
function seedFor(id, which) {
  let h = 2166136261 >>> 0;
  for (const ch of `${id}:${which}`) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
  return h || 1;
}

// Author one case's package + input files under `dir`; return { inputs: {name: Float32Array} }.
export function authorCase(dir, c) {
  ensureDir(dir);
  ensureDir(path.join(dir, 'inputs'));
  const n = c.shape.reduce((a, b) => a * b, 1);

  let inputsSpec, values;
  if (c.kind === 'unary') {
    inputsSpec = { input: 'x' };
    values = { x: seededFloat32(n, seedFor(c.id, 'x'), -4, 4) };
  } else {
    inputsSpec = { a: 'a', b: 'b' };
    const bLo = c.op === 'Div' ? 1 : -2, bHi = c.op === 'Div' ? 3 : 2;
    values = {
      a: seededFloat32(n, seedFor(c.id, 'a'), -2, 2),
      b: seededFloat32(n, seedFor(c.id, 'b'), bLo, bHi),
    };
  }

  writeGraph(path.join(dir, 'graph.json'),
    singleNodeGraph({ opType: c.op, inputs: inputsSpec, params: c.params, outShape: c.shape }));
  writeEmptySafetensors(path.join(dir, 'model.safetensors'));
  for (const [name, arr] of Object.entries(values)) {
    writeTensor(path.join(dir, 'inputs', `${name}.f32`), 'f32', arr);
  }
  // record which graph-input names map to which files, for native + torch
  fs.writeFileSync(path.join(dir, 'meta.json'),
    JSON.stringify({
      id: c.id,
      op: c.op,
      kind: c.kind,
      shape: c.shape,
      output: 'y',
      outputShape: c.shape,
      outputDtype: 'float32',
      inputs: Object.keys(values),
      params: c.params || {},
      skip: c.skip || [],
    }));
  return { inputs: values };
}
