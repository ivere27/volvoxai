import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { Graph } from '../ts/core/Graph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

function closeArray(actual, expected, tolerance = 2e-5) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++) {
    if (Number.isNaN(expected[index])) {
      assert.ok(Number.isNaN(actual[index]), `index ${index}: ${actual[index]} should be NaN`);
    } else {
      const allowed = tolerance * Math.max(1, Math.abs(expected[index]));
      assert.ok(Math.abs(actual[index] - expected[index]) <= allowed,
        `index ${index}: ${actual[index]} != ${expected[index]}`);
    }
  }
}

function unaryGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [7]);
  const sin = graph.addOp('Sin', { input }, { out: { name: 'sin', shape: [7] } }).out;
  const cos = graph.addOp('Cos', { input }, { out: { name: 'cos', shape: [7] } }).out;
  graph.setOutputs([sin.name, cos.name]);
  return graph;
}

function ropeGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 2, 6]);
  const positions = graph.addWeight('positions', [2, 2], 'int32', {
    buffer: Int32Array.of(0, 2, 1, 3),
  });
  const neox = graph.addOp('RoPE', { input }, {
    out: { name: 'neox', shape: [2, 2, 6] },
  }, { rotary_dim: 4, theta: 10, position_offset: 1, interleaved: false }).out;
  const gptj = graph.addOp('RoPE', { input, position_ids: positions }, {
    out: { name: 'gptj', shape: [2, 2, 6] },
  }, { rotary_dim: 4, theta: 10, interleaved: true }).out;
  graph.setOutputs([neox.name, gptj.name]);
  return graph;
}

function expectedRoPE(input, positions, interleaved, offset = 0) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < 4; row++) {
    const position = positions ? positions[row] : offset + (row % 2);
    const base = row * 6;
    for (let pair = 0; pair < 2; pair++) {
      const left = interleaved ? pair * 2 : pair;
      const right = interleaved ? left + 1 : pair + 2;
      const angle = position / Math.pow(10, (2 * pair) / 4);
      const c = Math.cos(angle), s = Math.sin(angle);
      output[base + left] = input[base + left] * c - input[base + right] * s;
      output[base + right] = input[base + left] * s + input[base + right] * c;
    }
    output[base + 4] = input[base + 4];
    output[base + 5] = input[base + 5];
  }
  return output;
}

function ssmGraph(opType, variableB) {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 3, 2]);
  const delta = graph.addInput('delta', [2, 3, 2]);
  const A = graph.addWeight('A', [2, 2], 'float32', {
    buffer: Float32Array.of(-0.5, -1, -0.25, -0.75),
  });
  const B = graph.addWeight('B', variableB ? [2, 3, 2] : [2], 'float32', {
    buffer: variableB ? Float32Array.from({ length: 12 }, (_, i) => 0.1 + i * 0.025) :
      Float32Array.of(0.25, -0.5),
  });
  const C = graph.addWeight('C', variableB ? [2] : [3, 2], 'float32', {
    buffer: variableB ? Float32Array.of(0.75, -0.25) :
      Float32Array.of(1, 0.5, 0.75, -0.25, 0.25, 1.25),
  });
  const D = graph.addWeight('D', [2], 'float32', { buffer: Float32Array.of(0.1, -0.2) });
  const z = graph.addWeight('z', [2, 3, 2], 'float32', {
    buffer: Float32Array.from({ length: 12 }, (_, i) => -0.6 + i * 0.1),
  });
  const initial = graph.addWeight('initial', [2, 2, 2], 'float32', {
    buffer: Float32Array.of(0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.15, -0.05),
  });
  const outputs = graph.addOp(opType, {
    input, delta, A, B, C, D, z, initial_state: initial,
  }, {
    out: { name: 'out', shape: [2, 3, 2] },
    state: { name: 'state', shape: [2, 2, 2] },
  }, { delta_softplus: variableB });
  graph.setOutputs([outputs.out.name, outputs.state.name]);
  return graph;
}

function referenceScan(graph, input, delta) {
  const node = graph.nodes[0];
  const { A, B, C, D, z, initial_state: initial } = node.inputs;
  const variableB = B.shape.length === 3;
  const state = new Float32Array(initial.buffer);
  const output = new Float32Array(input.length);
  const softplus = (value) => value > 20 ? value : value < -20 ? Math.exp(value) : Math.log(1 + Math.exp(value));
  for (let batch = 0; batch < 2; batch++) {
    for (let sequence = 0; sequence < 3; sequence++) {
      for (let channel = 0; channel < 2; channel++) {
        const inputIndex = (batch * 3 + sequence) * 2 + channel;
        const stateBase = (batch * 2 + channel) * 2;
        const dt = node.params.delta_softplus ? softplus(delta[inputIndex]) : delta[inputIndex];
        let result = D.buffer[channel] * input[inputIndex];
        for (let n = 0; n < 2; n++) {
          const bIndex = variableB ? (batch * 3 + sequence) * 2 + n : n;
          const cIndex = variableB ? n : sequence * 2 + n;
          state[stateBase + n] = Math.exp(dt * A.buffer[channel * 2 + n]) * state[stateBase + n] +
            dt * B.buffer[bIndex] * input[inputIndex];
          result += state[stateBase + n] * C.buffer[cIndex];
        }
        const gate = z.buffer[inputIndex];
        output[inputIndex] = result * gate / (1 + Math.exp(-gate));
      }
    }
  }
  return { output, state };
}

async function cpu(graph, inputs) {
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute(inputs);
}

test('portable Sin/Cos, RoPE, and selective scan agree on CPU(JS) and C/WASM', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-sequence-ops-'));
  try {
    const wasmPath = join(directory, 'volvoxai.wasm');
    await run(clang, [
      '--target=wasm32', '-O3', '-msimd128', '-nostdlib', '-Wl,--no-entry',
      '-Wl,--export-all', '-Wl,--allow-undefined', '-o', wasmPath,
      'native/src/kernels/kernels.c',
    ], { cwd: repositoryRoot });
    const wasm = await WasmEngine.init(wasmPath);
    assert.ok(wasm);

    const unaryInput = Float32Array.of(0, -0, Math.PI / 2, -Math.PI, 1e-4, Infinity, NaN);
    const unaryCpu = await cpu(unaryGraph(), { input: unaryInput });
    closeArray(unaryCpu.sin, Float32Array.from(unaryInput, Math.sin));
    closeArray(unaryCpu.cos, Float32Array.from(unaryInput, Math.cos));
    const unaryWasmGraph = unaryGraph();
    wasm.compile(unaryWasmGraph);
    const unaryWasm = await wasm.execute({ input: unaryInput });
    closeArray(unaryWasm.sin, unaryCpu.sin);
    closeArray(unaryWasm.cos, unaryCpu.cos);

    const ropeInput = Float32Array.from({ length: 24 }, (_, index) => index * 0.25 - 2);
    const ropeCpu = await cpu(ropeGraph(), { input: ropeInput });
    closeArray(ropeCpu.neox, expectedRoPE(ropeInput, null, false, 1));
    closeArray(ropeCpu.gptj, expectedRoPE(ropeInput, Int32Array.of(0, 2, 1, 3), true));
    const ropeWasmGraph = ropeGraph();
    wasm.compile(ropeWasmGraph);
    const ropeWasm = await wasm.execute({ input: ropeInput });
    closeArray(ropeWasm.neox, ropeCpu.neox);
    closeArray(ropeWasm.gptj, ropeCpu.gptj);

    const scanInput = Float32Array.from({ length: 12 }, (_, i) => 0.2 + i * 0.1);
    const scanDelta = Float32Array.from({ length: 12 }, (_, i) => -0.4 + i * 0.08);
    for (const [opType, variableB] of [['SSMScan', false], ['SelectiveScan', true]]) {
      const expectedGraph = ssmGraph(opType, variableB);
      const expected = referenceScan(expectedGraph, scanInput, scanDelta);
      const scanCpu = await cpu(ssmGraph(opType, variableB), { input: scanInput, delta: scanDelta });
      closeArray(scanCpu.out, expected.output, 5e-5);
      closeArray(scanCpu.state, expected.state, 5e-5);
      const scanWasmGraph = ssmGraph(opType, variableB);
      wasm.compile(scanWasmGraph);
      const scanWasm = await wasm.execute({ input: scanInput, delta: scanDelta });
      closeArray(scanWasm.out, scanCpu.out, 5e-5);
      closeArray(scanWasm.state, scanCpu.state, 5e-5);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('sequence operator contracts reject ambiguous layouts', async () => {
  const invalidRoPE = new Graph();
  const input = invalidRoPE.addInput('input', [2, 3]);
  invalidRoPE.addOp('RoPE', { input }, { out: { name: 'out', shape: [2, 3] } }, { rotary_dim: 3 });
  const engine = new CPUEngine();
  engine.allocateGraph(invalidRoPE);
  await assert.rejects(engine.execute({ input: Float32Array.of(1, 2, 3, 4, 5, 6) }), /valid rotary parameters/);

  const invalidScan = ssmGraph('SSMScan', false);
  invalidScan.nodes[0].inputs.B.shape = [1, 2];
  const scanEngine = new CPUEngine();
  scanEngine.allocateGraph(invalidScan);
  await assert.rejects(scanEngine.execute({
    input: new Float32Array(12), delta: new Float32Array(12),
  }), /canonical F32 selective-scan tensors/);
});
