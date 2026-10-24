import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, ok, safetensors, statelessGraph, tensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {readPackageVersion} from '../tools/release_version.mjs';
import {VxTrainingServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';

const wasmUrl = new URL(`../dist/${readPackageVersion()}/volvoxai.wasm`, import.meta.url).pathname;

for (const transpose of [false, true]) {
  test(`WASM BatchMatMul VJP reduces both broadcast axes${transpose ? ' through Transpose' : ''}`, async t => {
    const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
    const training = new VxTrainingServiceClient(reportTransport(f.host));
    const a = Array.from({length: 12}, (_, i) => Math.fround(Math.sin(i + 1) * .3));
    const b = Array.from({length: 36}, (_, i) => Math.fround(Math.cos(i + 2) * .4));
    const x = Float32Array.from(a, (_, i) => .8 + (i % 3) * .2);
    const bShape = transpose ? [1, 3, 4, 3] : [1, 3, 3, 4];
    const nodes = [{id: 'weight_input', opType: 'Mul', params: {}, inputs: {a: 'x', b: 'a'},
      outputs: {out: {tensor: 'left', dtype: 'float32', shape: [2, 1, 2, 3]}}}];
    if (transpose) nodes.push({id: 'transpose', opType: 'Transpose', inputs: {input: 'b'},
      params: {perm: [0, 1, 3, 2]},
      outputs: {out: {tensor: 'right', dtype: 'float32', shape: [1, 3, 3, 4]}}});
    nodes.push({id: 'product', opType: 'BatchMatMul', params: {}, inputs: {a: 'left', b: transpose ? 'right' : 'b'},
      outputs: {out: {tensor: 'logits', dtype: 'float32', shape: [2, 3, 2, 4]}}});
    const graph = statelessGraph({format: 'volvox-graph/v1', dimensions: {},
      inputs: {x: {dtype: 'float32', shape: [2, 1, 2, 3]}}, nodes, outputs: ['logits']});
    const targets = Int32Array.from({length: 12}, (_, i) => i % 4);
    // Independent scalar forward plus central finite differences. No engine
    // values or gradient formulas are used to construct the expected update.
    function loss() {
      let total = 0, rowIndex = 0;
      for (let batchA = 0; batchA < 2; batchA++) for (let batchB = 0; batchB < 3; batchB++) {
        for (let row = 0; row < 2; row++, rowIndex++) {
          const logits = [0, 0, 0, 0];
          for (let col = 0; col < 4; col++) for (let k = 0; k < 3; k++) {
            const ai = batchA * 6 + row * 3 + k;
            const bi = batchB * 12 + (transpose ? col * 3 + k : k * 4 + col);
            logits[col] += a[ai] * x[ai] * b[bi];
          }
          const maximum = Math.max(...logits);
          total += Math.log(logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0)) +
            maximum - logits[targets[rowIndex]];
        }
      }
      return total / 12;
    }
    const initial = {a: [...a], b: [...b]}, expected = {};
    const initialLoss = loss();
    for (const [name, values] of Object.entries({a, b})) expected[name] = values.map((value, i) => {
      values[i] = value + 1e-4; const upper = loss();
      values[i] = value - 1e-4; const lower = loss();
      values[i] = value;
      return value - .1 * (upper - lower) / 2e-4;
    });
    const model = await f.load(graph, safetensors([
      {name: 'a', shape: [2, 1, 2, 3], data: Float32Array.from(initial.a)},
      {name: 'b', shape: bShape, data: Float32Array.from(initial.b)},
    ]));
    const trainer = ok(await training.createTrainer(new p.CreateTrainerRequest({modelId: model.modelId, backend: 'wasm'})));
    const step = ok(await training.trainStep(new p.TrainStepRequest({trainerId: trainer.trainerId,
      inputs: tensors({x: {shape: [2, 1, 2, 3], data: x}}), trainableNames: ['a', 'b'],
      losses: [new p.CrossEntropyLoss({name: 'ce', logitsName: 'logits',
        targets: tensors({target: {shape: [2, 3, 2], data: targets}})[0]})],
      optimizer: new p.TrainerOptimizerOptions({kind: p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,
        learningRate: .1})})));
    assert.ok(Math.abs(step.loss - initialLoss) < 2e-6);
    const bytes = ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest(trainer))).shards[0];
    const size = Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + size)));
    for (const name of ['a', 'b']) {
      const [start, end] = header[name].data_offsets;
      const actual = new Float32Array(bytes.slice(8 + size + start, 8 + size + end).buffer);
      for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i] - expected[name][i]) < 1e-7,
        `${name}[${i}]: ${actual[i]} != ${expected[name][i]}`);
    }
  });
}
