// Generates a small CNN graph document (graph.json + .safetensors + input.f32), runs it
// through the public CPU runtime to capture the reference output, and writes everything to
// an output dir. The native C engine consumes the same files.
// Usage: node tools/gen_test_model.mjs <outdir>
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const packageVersion = JSON.parse(
  readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
).version;
const { Model, VolvoxAI, parseGraphDocument } = await import(
  new URL(`../dist/${packageVersion}/volvoxai.js`, import.meta.url)
);

const outdir = process.argv[2] || '/tmp/volvox_model';
mkdirSync(outdir, { recursive: true });

// deterministic weight filler
const fill = (n, seed) => Float32Array.from({ length: n }, (_, i) => Math.sin((i + 1) * 0.1 + seed) * 0.3);

const weights = {}; // name -> Float32Array
const weightSpecs = [];
const W = (name, shape, seed) => {
  const data = fill(shape.reduce((a, b) => a * b, 1), seed);
  weights[name] = data;
  weightSpecs.push({ name, dtype: 'float32', shape });
};
W('c1.w', [3, 3, 3, 8], 1);
W('c1.b', [8], 2);
W('c2.w', [3, 3, 8, 16], 6);
W('c2.b', [16], 7);
W('fc.w', [16, 10], 8);
W('fc.b', [10], 9);

const output = (tensor, shape) => ({ out: { tensor, dtype: 'float32', shape } });
const graphDocument = {
  format: 'volvox-graph/v1',
  dimensions: {},
  inputs: { x: { shape: [1, 8, 8, 3], dtype: 'float32' } },
  // Conv -> ReLU -> MaxPool -> Conv -> ReLU -> Add -> GAP -> Reshape -> Linear.
  nodes: [
    {
      id: 'conv1', opType: 'Conv2D',
      inputs: { input: 'x', weight: 'c1.w', bias: 'c1.b' },
      outputs: output('conv1.out', [1, 8, 8, 8]),
      params: { stride: [1, 1], padding: [1, 1], weight_layout: 'HWIO' },
    },
    {
      id: 'relu1', opType: 'ReLU', inputs: { input: 'conv1.out' },
      outputs: output('relu1.out', [1, 8, 8, 8]), params: {},
    },
    {
      id: 'pool', opType: 'MaxPool2D', inputs: { input: 'relu1.out' },
      outputs: output('pool.out', [1, 4, 4, 8]),
      params: { kernel: [2, 2], stride: [2, 2] },
    },
    {
      id: 'conv2', opType: 'Conv2D',
      inputs: { input: 'pool.out', weight: 'c2.w', bias: 'c2.b' },
      outputs: output('conv2.out', [1, 4, 4, 16]),
      params: { stride: [1, 1], padding: [1, 1], weight_layout: 'HWIO' },
    },
    {
      id: 'relu2', opType: 'ReLU', inputs: { input: 'conv2.out' },
      outputs: output('relu2.out', [1, 4, 4, 16]), params: {},
    },
    {
      id: 'residual', opType: 'Add', inputs: { a: 'relu2.out', b: 'relu2.out' },
      outputs: output('residual.out', [1, 4, 4, 16]), params: {},
    },
    {
      id: 'gap', opType: 'GlobalAveragePool', inputs: { input: 'residual.out' },
      outputs: output('gap.out', [1, 1, 1, 16]), params: {},
    },
    {
      id: 'reshape', opType: 'Reshape', inputs: { input: 'gap.out' },
      outputs: output('flat', [1, 16]), params: { shape: [1, 16] },
    },
    {
      id: 'classifier', opType: 'Linear',
      inputs: { input: 'flat', weight: 'fc.w', bias: 'fc.b' },
      outputs: output('logits', [1, 10]),
      params: { weight_layout: 'din_dout' },
    },
  ],
  outputs: ['logits'],
};
const logicalGraph = parseGraphDocument(graphDocument, weightSpecs);
const snapshot = Model.capture({
  graph: logicalGraph,
  weights: Object.fromEntries(weightSpecs.map(({ name, dtype, shape }) => [name, {
    name, dtype, shape, data: weights[name],
  }])),
});

// ---- reference run on the JS CPU runtime ----
const input = fill(1 * 8 * 8 * 3, 42);
const runtime = await VolvoxAI.createRuntime({ backends: ['cpu-js'] });
let compiled;
let context;
let result;
let expected;
try {
  compiled = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
  });
  context = await compiled.createContext();
  result = await context.execute({ x: { data: input, shape: [1, 8, 8, 3] } });
  expected = Array.from(await result.output('logits').read());
} finally {
  await result?.close();
  await context?.close();
  await compiled?.close();
  await runtime.close();
}

writeFileSync(join(outdir, 'graph.json'), JSON.stringify(graphDocument, null, 1));

// ---- safetensors (F32 weights) ----
const header = {}; let offset = 0; const chunks = [];
for (const [name, buf] of Object.entries(weights)) {
  const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const shape = weightSpecs.find((candidate) => candidate.name === name).shape;
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
console.log(`wrote ${outdir}/ {graph.json, model.safetensors, input.f32, expected.json}`);
console.log('JS reference output:', expected.map((v) => v.toFixed(5)).join(' '));
