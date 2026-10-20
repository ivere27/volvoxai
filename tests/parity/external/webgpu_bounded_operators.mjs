/** Complete-domain admission and changing bindings, using real GPU commands. */
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { CASES, graphOf, runGraph, compare, extractedCase, requirePhysicalDevice } from './webgpu_device_bridge.mjs';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
const api = await import(pathToFileURL(path.resolve(args.get('--bundle'))).href);
const { device, description } = await requirePhysicalDevice({
  allowNamedSwiftShaderUnit: args.get('--adapter') === 'swiftshader-unit',
});
const options = { bundle: path.resolve(args.get('--bundle')), wasm: path.resolve(args.get('--wasm')), device };
console.log(`bounded operators adapter: ${description}`);
const records = [];
const selected = name => !args.has('--only') || name.toLowerCase().includes(args.get('--only').toLowerCase());
const repeat = (data, times) => {
  const output = new data.constructor(data.length * times);
  for (let i = 0; i < times; i++) output.set(data, i * data.length);
  return output;
};
const roundTrips = async (name, graph, batches, expected) => {
  const cpu = await runGraph(api, options, graph, batches, 'wasm');
  const gpu = await runGraph(api, options, graph, batches, 'webgpu');
  assert.equal(gpu.rounds.length, batches.length);
  assert.equal(gpu.route.fallbackNodes, 0);
  for (let round = 0; round < batches.length; round++) {
    for (const [output, values] of cpu.rounds[round]) {
      compare(`${name}/${round}/${output}`, gpu.rounds[round].get(output), values, 1e-4);
      if (expected) compare(`${name}/${round}/independent/${output}`, gpu.rounds[round].get(output), expected(round, output), 1e-4);
    }
  }
  console.log(`bounded operators ${name}: PASS (${batches.length} bindings, ${gpu.encodes} dispatches)`);
  records.push({ name, bindings: batches.length, dispatches: gpu.encodes });
};

try {
  const fixtures = [...CASES];
  if (args.has('--cases')) for (const name of ['QConv2D', 'QConv2DDepthwise']) {
    const fixture = extractedCase(args.get('--cases'), name); assert.ok(fixture); fixtures.push(fixture);
  }
  const roles = new Set(['input', 'data', 'a', 'b', 'q', 'k', 'v', 'qkv', 'mask', 'condition', 'indices']);
  for (const [name, original, tensors, , fixtureOracle] of fixtures) {
    if (!selected(name)) continue;
    // These fixtures reshape/index the leading axis itself. They have separate
    // bounded contracts below or in webgpu_graph_domains / webgpu_decode.
    if (original.nodes.some(n => ['Reshape', 'Expand', 'Transpose', 'Gather'].includes(n.opType))) continue;
    const base = tensors[0].shape[0];
    if (!base || tensors[0].shape.length < 2) continue;
    const graph = structuredClone(original), varying = new Set();
    graph.dimensions = { B: { min: base, max: base * 3, multiple_of: base } };
    for (const node of graph.nodes) for (const [role, name] of Object.entries(node.inputs)) {
      if (!roles.has(role) && !node.opType.startsWith('Concat')) continue;
      const descriptor = graph.inputs[name];
      if (descriptor?.shape.length >= 2 && descriptor.shape[0] === base) varying.add(name);
    }
    if (!varying.size) continue;
    const outputs = graph.nodes.flatMap(n => Object.values(n.outputs));
    if (outputs.some(output => output.shape[0] !== base)) continue;
    for (const name of varying) graph.inputs[name].shape[0] = 'B';
    for (const output of outputs) output.shape[0] = 'B';
    const factors = [1, 3, 2, 1];
    const batches = factors.map(factor => tensors.map(t => varying.has(t.name)
      ? { ...t, shape: [base * factor, ...t.shape.slice(1)], data: repeat(t.data, factor) } : t));
    await roundTrips(name, graph, batches, fixtureOracle ? (round, output) => repeat(fixtureOracle.get(output), factors[round]) : null);
  }

  // Reductions can vary their feature width; only norm/dense shader families
  // require an invariant last dimension. This used to be over-restricted in C.
  for (const op of ['Softmax', 'LogSoftmax', 'ReduceSum', 'ReduceMean', 'ArgMax']) {
    const name = `${op}/dynamic-feature`; if (!selected(name)) continue;
    const soft = op.includes('Softmax'), argmax = op === 'ArgMax';
    const graph = graphOf([{ id: 'n', opType: op, inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: argmax ? 'int32' : 'float32', shape: soft ? [2, 'D'] : [2] } },
      params: soft ? { axis: -1 } : { axis: -1, keepdims: false } }], { x: { dtype: 'float32', shape: [2, 'D'] } }, ['y']);
    graph.dimensions = { D: { min: 1, max: 9 } };
    const batches = [1, 9, 5, 2].map(d => [{ name: 'x', shape: [2, d], data: Float32Array.from({ length: 2 * d }, (_, i) => Math.sin(i * .7)) }]);
    const expected = round => {
      const x = batches[round][0].data, d = x.length / 2, values = [];
      for (let row = 0; row < 2; row++) {
        const data = [...x.slice(row * d, (row + 1) * d)], sum = data.reduce((a, b) => a + b, 0), max = Math.max(...data);
        if (argmax) values.push(data.indexOf(max));
        else if (!soft) values.push(op === 'ReduceSum' ? sum : sum / d);
        else { const total = data.reduce((s, value) => s + Math.exp(value - max), 0);
          values.push(...data.map(value => op === 'Softmax' ? Math.exp(value - max) / total : value - max - Math.log(total))); }
      }
      return (argmax ? Int32Array : Float32Array).from(values);
    };
    await roundTrips(name, graph, batches, expected);
  }

  // Spatial extents are activation state, while channels, kernel geometry and
  // per-axis affine tables stay fixed. Put QConv after another node so its
  // physical proof must select its own metadata, and cover omitted bias.
  for (const ByteArray of [Int8Array, Uint8Array]) for (const groups of [1, 2])
    for (const dilation of [1, 2]) for (const withBias of [false, true]) {
      const name = `QConv2D/dynamic-spatial/${ByteArray.name}/groups-${groups}/dilation-${dilation}/bias-${withBias}`;
      if (!selected(name)) continue;
      const unsigned = ByteArray === Uint8Array, dtype = unsigned ? 'uint8' : 'int8';
      const channels = 2, outChannels = 4, kh = 3, kw = 2, perGroup = channels / groups;
      const inputZero = unsigned ? 127 : -2, outputZero = unsigned ? 121 : -3;
      const weightZeros = ByteArray.from({ length: outChannels }, (_, i) => unsigned ? 120 + i : i - 2);
      const weights = ByteArray.from({ length: outChannels * kh * kw * perGroup }, (_, i) =>
        weightZeros[Math.floor(i / (kh * kw * perGroup))] + (i * 7 % 11) - 5);
      const bias = Int32Array.from({ length: outChannels }, (_, i) => i * 3 - 4);
      const tables = {
        w: { shape: [outChannels, kh, kw, perGroup], data: weights },
        'x.scale': { shape: [1], data: Float32Array.of(.125) },
        'x.zero': { shape: [1], data: ByteArray.of(inputZero) },
        'w.scale': { shape: [outChannels], data: Float32Array.from({ length: outChannels }, () => .25) },
        'w.zero': { shape: [outChannels], data: weightZeros },
        'y.scale': { shape: [1], data: Float32Array.of(.03125) },
        'y.zero': { shape: [1], data: ByteArray.of(outputZero) },
        ...(withBias ? { bias: { shape: [outChannels], data: bias } } : {}),
      };
      const activation = stem => ({ scheme: 'per_tensor', scale_tensor: `${stem}.scale`, zero_point_tensor: `${stem}.zero` });
      const shape = ['B', 'H', 'W', channels], outputShape = ['B', 'H', 'W', outChannels];
      const pads = [dilation, 0, dilation, dilation];
      const graph = graphOf([
        { id: 'view', opType: 'Identity', inputs: { input: 'x' },
          outputs: { out: { tensor: 'view', shape, dtype } }, params: {} },
        { id: 'conv', opType: 'QConv2D', inputs: { input: 'view', weight: 'w', ...(withBias ? { bias: 'bias' } : {}) },
          outputs: { out: { tensor: 'y', shape: outputShape, dtype } },
          params: { groups, stride: [1, 1], dilation: [dilation, dilation], pads } },
      ], { x: { shape, dtype } }, ['y'], tables, {
        x: activation('x'), view: activation('x'), y: activation('y'),
        w: { scheme: 'per_axis', axis: 0, scale_tensor: 'w.scale', zero_point_tensor: 'w.zero' },
      });
      graph.dimensions = { B: { min: 1, max: 2 }, H: { min: 3, max: 9 }, W: { min: 2, max: 8 } };
      const batches = [[1, 3, 2], [2, 9, 8], [1, 5, 7], [1, 3, 2]].map(([b, h, w]) => [{
        name: 'x', shape: [b, h, w, channels],
        data: ByteArray.from({ length: b * h * w * channels }, (_, i) => inputZero + (i * 13 % 17) - 8),
      }]);
      const expected = round => {
        const [{ shape: [batch, height, width], data }] = batches[round];
        const output = new ByteArray(batch * height * width * outChannels);
        for (let b = 0; b < batch; b++) for (let y = 0; y < height; y++)
          for (let x = 0; x < width; x++) for (let o = 0; o < outChannels; o++) {
            let sum = withBias ? bias[o] : 0;
            const group = Math.floor(o / (outChannels / groups));
            for (let ky = 0; ky < kh; ky++) for (let kx = 0; kx < kw; kx++) {
              const iy = y + ky * dilation - pads[0], ix = x + kx * dilation - pads[1];
              if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
              for (let c = 0; c < perGroup; c++) {
                const a = data[((b * height + iy) * width + ix) * channels + group * perGroup + c] - inputZero;
                const w = weights[((o * kh + ky) * kw + kx) * perGroup + c] - weightZeros[o];
                sum += a * w;
              }
            }
            output[((b * height + y) * width + x) * outChannels + o] =
              Math.max(unsigned ? 0 : -128, Math.min(unsigned ? 255 : 127, sum + outputZero));
          }
        return output;
      };
      await roundTrips(name, graph, batches, expected);
  }
  // Head norms use [B,S,H,D], followed by flattened head-aware RoPE. Cover
  // changing S and each public position form against an independent formula.
  for (const interleaved of [false, true]) for (const mode of ['offset', 'shared', 'batch']) {
    const name = `RoPE/head-norm-D128/${interleaved}/${mode}`;
    if (!selected(name)) continue;
    const B = 2, H = 3, D = 128, R = interleaved ? 126 : 128, theta = 1000000;
    const shape = [B, 'S', H * D], headShape = [B, 'S', H, D];
    const norm = Float32Array.from({ length: D }, (_, i) => .7 + .4 * i / D);
    const inputs = { x: { shape, dtype: 'float32' } };
    if (mode !== 'offset') inputs.positions = { shape: mode === 'shared' ? ['S'] : [B, 'S'], dtype: 'int32' };
    const graph = graphOf([
      { id: 'heads', opType: 'Reshape', inputs: { input: 'x' }, params: { shape: headShape },
        outputs: { out: { tensor: 'heads', shape: headShape, dtype: 'float32' } } },
      { id: 'norm', opType: 'RMSNorm', inputs: { input: 'heads', weight: 'norm' }, params: { eps: 1e-6 },
        outputs: { out: { tensor: 'normed', shape: headShape, dtype: 'float32' } } },
      { id: 'flat', opType: 'Reshape', inputs: { input: 'normed' }, params: { shape },
        outputs: { out: { tensor: 'flat', shape, dtype: 'float32' } } },
      { id: 'rope', opType: 'RoPE', inputs: { input: 'flat', ...(mode === 'offset' ? {} : { position_ids: 'positions' }) },
        params: { heads: H, rotary_dim: R, theta, position_offset: 19, interleaved },
        outputs: { out: { tensor: 'y', shape, dtype: 'float32' } } },
    ], inputs, ['y'], { norm: { shape: [D], data: norm } });
    graph.dimensions = { S: { min: 1, max: 5 } };
    const batches = [1, 5, 3, 1].map(S => [
      { name: 'x', shape: [B, S, H * D], data: Float32Array.from({ length: B * S * H * D }, (_, i) => Math.sin(i * .17)) },
      ...(mode === 'offset' ? [] : [{ name: 'positions', shape: mode === 'shared' ? [S] : [B, S],
        data: Int32Array.from({ length: mode === 'shared' ? S : B * S }, (_, i) => 7 + i * 3) }]),
    ]);
    const expected = round => {
      const x = batches[round][0].data, S = batches[round][0].shape[1], positions = batches[round][1]?.data;
      const out = new Float32Array(x.length);
      for (let b = 0; b < B; b++) for (let s = 0; s < S; s++) for (let h = 0; h < H; h++) {
        const base = ((b * S + s) * H + h) * D;
        let sum = 0; for (let d = 0; d < D; d++) sum += x[base + d] ** 2;
        const inverse = 1 / Math.sqrt(sum / D + 1e-6);
        for (let d = 0; d < D; d++) out[base + d] = x[base + d] * inverse * norm[d];
        const position = mode === 'offset' ? 19 + s : positions[mode === 'shared' ? s : b * S + s];
        for (let pair = 0; pair < R / 2; pair++) {
          const a = interleaved ? pair * 2 : pair, z = interleaved ? a + 1 : pair + R / 2;
          const frequency = Math.fround(theta ** Math.fround(2 * pair / R));
          const angle = Math.fround(Math.fround(position) / frequency), left = out[base + a], right = out[base + z];
          out[base + a] = left * Math.cos(angle) - right * Math.sin(angle);
          out[base + z] = right * Math.cos(angle) + left * Math.sin(angle);
        }
      }
      return out;
    };
    await roundTrips(name, graph, batches, expected);
  }
  for (const kvHeads of [1, 2]) {
    const name = `CrossSDPA/compact-GQA-D128/kv-${kvHeads}`;
    if (!selected(name)) continue;
    const B = 2, H = 4, D = 128, shape = [B, 'S', H * D], kvShape = [B, 'S', kvHeads * D];
    const graph = graphOf([{ id: 'attention', opType: 'CrossSDPA', inputs: { q: 'q', k: 'k', v: 'v' },
      params: { heads: H, kv_heads: kvHeads, causal: true }, outputs: { out: { tensor: 'y', shape, dtype: 'float32' } } }],
      { q: { shape, dtype: 'float32' }, k: { shape: kvShape, dtype: 'float32' }, v: { shape: kvShape, dtype: 'float32' } }, ['y']);
    graph.dimensions = { S: { min: 1, max: 5 } };
    const batches = [1, 5, 3, 1].map(S => ['q', 'k', 'v'].map((name, j) => ({ name,
      shape: [B, S, (j ? kvHeads : H) * D],
      data: Float32Array.from({ length: B * S * (j ? kvHeads : H) * D }, (_, i) => .7 * Math.sin(i * .17 + j * .3)),
    })));
    await roundTrips(name, graph, batches);
  }
  assert.ok(records.length, 'no cases selected');
  console.log(`bounded operators: ${records.length} cases passed`);
  if (args.has('--out')) await Deno.writeTextFile(args.get('--out'), JSON.stringify({ adapter: description, records }, null, 2));
} finally { device.destroy(); await device.lost; }
