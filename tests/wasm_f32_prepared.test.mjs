import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {runGraph} from './parity/external/webgpu_device_bridge.mjs';

const {version} = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const f = Math.fround;
const values = (n, shift = 0) => Float32Array.from({length: n}, (_, i) =>
  f((((i * 29 + shift * 13) % 127) - 63) / 128));
function exact(actual, reference) {
  assert.deepEqual(new Uint32Array(actual.buffer), new Uint32Array(reference.buffer));
}

for (const profile of ['inference', 'full']) {
  if (process.env.VOLVOXAI_F32_PROFILE && process.env.VOLVOXAI_F32_PROFILE !== profile) continue;
  const stem = profile === 'full' ? 'volvoxai' : 'volvoxai.lite';
  const api = await import(`../dist/${version}/${stem}.js`);
  const runner = {...api, InferenceWasmHost: api.EngineHost};
  const wasm = process.env[`VOLVOXAI_F32_${profile.toUpperCase()}_WASM`] ??
    fileURLToPath(new URL(`../dist/${version}/${stem}.wasm`, import.meta.url));

  for (const mutable of [false, true]) test(`${profile}: F32 Linear ${mutable ? 'mutable' : 'immutable'} weights and changing inputs`, async () => {
    const k = 17, n = 9, weight = values(k * n, 2), bias = values(n, 3);
    const graph = {format: 'volvox-graph/v1', dimensions: {M: {min: 1, max: 9}},
      inputs: {x: {dtype: 'float32', shape: ['M', k]},
        ...(mutable ? {w: {dtype: 'float32', shape: [n, k]}} : {})},
      outputs: ['y'], nodes: [{id: 'linear', opType: 'Linear',
        inputs: {input: 'x', weight: 'w', bias: 'bias'},
        outputs: {out: {tensor: 'y', dtype: 'float32', shape: ['M', n]}},
        params: {weight_layout: 'dout_din'}}],
      __weights: {bias: {shape: [n], data: bias},
        ...(!mutable ? {w: {shape: [n, k], data: weight}} : {})}};
    const batches = [1, 9, 3, 1].map((m, revision) => [
      {name: 'x', shape: [m, k], data: values(m * k, revision)},
      ...(mutable ? [{name: 'w', shape: [n, k], data: values(n * k, revision + 2)}] : []),
    ]);
    const result = await runGraph(runner, {wasm}, graph, batches, 'wasm');
    for (let round = 0; round < batches.length; round++) {
      const input = batches[round][0], w = mutable ? batches[round][1].data : weight;
      const reference = new Float32Array(input.shape[0] * n);
      for (let row = 0; row < input.shape[0]; row++) for (let col = 0; col < n; col++) {
        let sum = bias[col];
        for (let inner = 0; inner < k; inner++)
          sum = f(sum + f(input.data[row * k + inner] * w[col * k + inner]));
        reference[row * n + col] = sum;
      }
      exact(result.rounds[round].get('y'), reference);
    }
  });

  for (const direction of [0, 1, 2]) test(`${profile}: LSTM direction ${direction} immutable packs match changing graph weights`, async () => {
    const width = 5, hidden = 7, dirs = direction === 2 ? 2 : 1;
    const w = values(dirs * 4 * hidden * width, 2), r = values(dirs * 4 * hidden * hidden, 3);
    const b = values(dirs * 8 * hidden, 4);
    const graph = mutable => ({format: 'volvox-graph/v1', dimensions: {T: {min: 1, max: 5}},
      inputs: {x: {dtype: 'float32', shape: ['T', 1, width]},
        ...(mutable ? {w: {dtype: 'float32', shape: [dirs, 4 * hidden, width]},
          r: {dtype: 'float32', shape: [dirs, 4 * hidden, hidden]}} : {})},
      outputs: ['y'], nodes: [{id: 'lstm', opType: 'LSTM',
        inputs: {input: 'x', weight: 'w', recurrent_weight: 'r', bias: 'b'},
        outputs: {out: {tensor: 'y', dtype: 'float32', shape: ['T', dirs, 1, hidden]}},
        params: {direction}}],
      __weights: {b: {shape: [dirs, 8 * hidden], data: b}, ...(!mutable ? {
        w: {shape: [dirs, 4 * hidden, width], data: w},
        r: {shape: [dirs, 4 * hidden, hidden], data: r}} : {})}});
    const batches = [1, 5, 2, 1].map((time, revision) => [
      {name: 'x', shape: [time, 1, width], data: values(time * width, revision)},
    ]);
    const dynamic = batches.map(batch => [...batch,
      {name: 'w', shape: [dirs, 4 * hidden, width], data: w},
      {name: 'r', shape: [dirs, 4 * hidden, hidden], data: r}]);
    const reference = await runGraph(runner, {wasm}, graph(true), dynamic, 'wasm');
    const actual = await runGraph(runner, {wasm}, graph(false), batches, 'wasm');
    for (let round = 0; round < batches.length; round++)
      exact(actual.rounds[round].get('y'), reference.rounds[round].get('y'));
    // Rebind W/R in the same dynamic context; no immutable pack may shadow it.
    const changedW = values(w.length, 11), changedR = values(r.length, 12);
    const changing = [dynamic[0], [dynamic[0][0], {...dynamic[0][1], data: changedW},
      {...dynamic[0][2], data: changedR}], dynamic[0]];
    const changed = await runGraph(runner, {wasm}, graph(true), changing, 'wasm');
    exact(changed.rounds[0].get('y'), changed.rounds[2].get('y'));
    assert.notDeepEqual(changed.rounds[0].get('y'), changed.rounds[1].get('y'));
  });
}
