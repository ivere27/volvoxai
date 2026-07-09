// Generates a small CNN blueprint (config.json + .safetensors + input.f32), runs it
// through the JS CPUEngine to capture the reference output, and writes everything to
// an output dir. The native C engine consumes the same files.
// Usage: node tools/gen_test_model.mjs <outdir>
import { Graph, CPUEngine } from '../dist/volvoxai.js';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const outdir = process.argv[2] || '/tmp/volvox_model';
mkdirSync(outdir, { recursive: true });

// deterministic weight filler
const fill = (n, seed) => Float32Array.from({ length: n }, (_, i) => Math.sin((i + 1) * 0.1 + seed) * 0.3);

const g = new Graph();
const weights = {}; // name -> Float32Array
const W = (name, shape, seed) => { const t = g.addWeight(name, shape); t.buffer = fill(shape.reduce((a, b) => a * b, 1), seed); weights[name] = t.buffer; return t; };

const x = g.addInput('x', [1, 8, 8, 3]);
// Conv(3->8,3x3,p1) -> BN -> ReLU -> MaxPool2 -> Conv(8->16,3x3,p1) -> ReLU -> Add(self) -> GAP -> Reshape -> MatMul(16->10)
let t = g.addOp('Conv2D', { input: x, weight: W('c1.w', [3, 3, 3, 8], 1), bias: W('c1.b', [8], 2) }, { out: [1, 8, 8, 8] }, { stride: [1, 1], padding: [1, 1], weight_layout: 'HWIO' }).out;
t = g.addOp('BatchNorm2D', { input: t, weight: W('bn.w', [8], 3), bias: W('bn.b', [8], 4), running_mean: W('bn.rm', [8], 5), running_var: W('bn.rv', [8], 0.5) }, { out: [1, 8, 8, 8] }, { eps: 1e-5 }).out;
// make running_var positive
for (let i = 0; i < weights['bn.rv'].length; i++) weights['bn.rv'][i] = Math.abs(weights['bn.rv'][i]) + 0.2;
t = g.addOp('ReLU', { input: t }, { out: [1, 8, 8, 8] }).out;
t = g.addOp('MaxPool2D', { input: t }, { out: [1, 4, 4, 8] }, { kernel: [2, 2], stride: [2, 2] }).out;
t = g.addOp('Conv2D', { input: t, weight: W('c2.w', [3, 3, 8, 16], 6), bias: W('c2.b', [16], 7) }, { out: [1, 4, 4, 16] }, { stride: [1, 1], padding: [1, 1], weight_layout: 'HWIO' }).out;
t = g.addOp('ReLU', { input: t }, { out: [1, 4, 4, 16] }).out;
t = g.addOp('Add', { a: t, b: t }, { out: [1, 4, 4, 16] }).out;      // same-shape add (tests add_f32)
t = g.addOp('GlobalAveragePool', { input: t }, { out: [1, 1, 1, 16] }).out;
t = g.addOp('Reshape', { input: t }, { out: [1, 16] }).out;
t = g.addOp('MatMul', { input: t, weight: W('fc.w', [16, 10], 8), bias: W('fc.b', [10], 9) }, { out: [1, 10] }).out;
g.outputNames = [t.name];

// ---- reference run on the JS CPU engine ----
const input = fill(1 * 8 * 8 * 3, 42);
const cpu = new CPUEngine();
cpu.allocateGraph(g);
const res = await cpu.execute(g, { x: input });
const expected = Array.from(res[t.name]);

// ---- serialize blueprint the native engine (and GraphLoader) can read ----
const config = { inputs: { x: { shape: x.shape, dtype: 'float32' } }, nodes: [] };
for (const node of g.nodes) {
  const inputs = {}; for (const [k, tt] of Object.entries(node.inputs)) inputs[k] = tt.name;
  const outputs = {}, outputs_shape = {};
  for (const [k, tt] of Object.entries(node.outputs)) { outputs[k] = tt.name; outputs_shape[k] = tt.shape; }
  config.nodes.push({ opType: node.opType, inputs, outputs, outputs_shape, params: node.params });
}
writeFileSync(join(outdir, 'config.json'), JSON.stringify(config, null, 1));

// ---- safetensors (F32 weights) ----
const header = {}; let offset = 0; const chunks = [];
for (const [name, buf] of Object.entries(weights)) {
  const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const shape = g.tensors.get(name).shape;
  header[name] = { dtype: 'F32', shape, data_offsets: [offset, offset + bytes.length] };
  offset += bytes.length; chunks.push(bytes);
}
let headerStr = JSON.stringify(header);
while ((8 + headerStr.length) % 8 !== 0) headerStr += ' ';           // 8-byte align data section
const headerBytes = new TextEncoder().encode(headerStr);
const lenBuf = new ArrayBuffer(8); new DataView(lenBuf).setBigUint64(0, BigInt(headerBytes.length), true);
const total = 8 + headerBytes.length + offset;
const out = new Uint8Array(total); let pos = 0;
out.set(new Uint8Array(lenBuf), pos); pos += 8;
out.set(headerBytes, pos); pos += headerBytes.length;
for (const c of chunks) { out.set(c, pos); pos += c.length; }
writeFileSync(join(outdir, 'model.safetensors'), out);

// ---- input + expected ----
writeFileSync(join(outdir, 'input.f32'), Buffer.from(input.buffer));
writeFileSync(join(outdir, 'expected.json'), JSON.stringify(expected));
console.log(`wrote ${outdir}/ {config.json, model.safetensors, input.f32, expected.json}`);
console.log('JS reference output:', expected.map((v) => v.toFixed(5)).join(' '));
