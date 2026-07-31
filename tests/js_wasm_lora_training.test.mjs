import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { ModelBuilder, VolvoxAI } from '../ts/wasm.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

async function buildFullWasm(output) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output,
    'native/src/kernels/kernels.c', 'native/src/kernels/training_kernels.c',
  ], { cwd: repositoryRoot });
}

function explicitLoRAGraph() {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1, 2]);
  const base = builder.weight('base', [3, 2], 'float32', Float32Array.from([
    1, 0.2,
    0.1, 0.5,
    0, 0,
  ]));
  const lora = builder.loraLinear(input, base, {
    rank: 1,
    layout: 'dout',
    opType: 'MatMul',
    aInitializer: 'ones',
    bInitializer: { type: 'normal', seed: 7, stddev: 0.1 },
    name: 'decoder',
  });
  builder.outputs(lora.out);
  return {
    graph: builder.graph,
    base,
    a: lora.A,
    b: lora.B,
    logits: lora.out,
    trainableTensors: lora.trainableTensors,
  };
}

function expectedLogits(input, base, a, b) {
  const lowRank = input[0] * a[0] + input[1] * a[1];
  return Array.from(b, (delta, output) =>
    input[0] * base[output * 2] + input[1] * base[output * 2 + 1] + lowRank * delta);
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index++) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}

function closeArray(actual, expected, label, tolerance = 2e-5) {
  assert.equal(actual.length, expected.length, `${label} length`);
  for (let index = 0; index < actual.length; index++) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
      `${label}[${index}]: ${actual[index]} != ${expected[index]}`);
  }
}

test('WASM-only correction trains explicit LoRA through a dout base and refreshes packed weights', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-lora-training-'));
  let runtime;
  let model;
  let compiledBefore;
  let contextBefore;
  let trainer;
  let compiledAfter;
  let contextAfter;
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);

    const { graph, base, a, b, logits, trainableTensors } = explicitLoRAGraph();
    runtime = await VolvoxAI.createRuntime({ wasmUrl: wasmPath });
    model = runtime.createModel(graph);
    compiledBefore = await model.compile({
      backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
    });
    contextBefore = await compiledBefore.createContext();
    trainer = await VolvoxAI.createTrainer(model, { wasmUrl: wasmPath });
    const x = Float32Array.from([0.7, -0.2]);
    const inputs = { x };
    const beforeResult = await contextBefore.execute(inputs);
    const beforeInference = Array.from(await beforeResult.output(logits.name).read());
    await beforeResult.close();
    assert.equal(argmax(beforeInference), 0,
      'the initial prediction should be wrong before the correction');

    const beforeBase = Float32Array.from(base.buffer);
    const beforeA = Float32Array.from(a.buffer);
    const beforeB = Float32Array.from(b.buffer);
    const originalRevision = model.weightRevisionId;
    const options = {
      inputs,
      targets: [2],
      trainableTensors,
      updateMode: 'sgd',
      optimizer: { learningRate: 1 },
    };

    const first = await trainer.trainStep(options);
    assert.equal(first.backend, 'wasm');
    assert.deepEqual([...first.updatedTensorNames].sort(),
      [...trainableTensors].sort());
    assert.deepEqual(base.buffer, beforeBase);
    assert.deepEqual(a.buffer, beforeA);
    assert.deepEqual(b.buffer, beforeB);

    const losses = [first.loss];
    for (let step = 1; step < 12; step++) {
      losses.push((await trainer.trainStep(options)).loss);
    }
    assert.ok(losses.at(-1) < losses[0],
      `correction loss did not decrease: ${losses[0]} -> ${losses.at(-1)}`);
    assert.deepEqual(base.buffer, beforeBase);
    assert.equal(graph.trainingStep, 0);
    assert.equal(trainer.trainingStep, 12);
    assert.equal(model.weightRevisionId, originalRevision,
      'WASM Trainer updates stay private until commit');

    const pinnedResult = await contextBefore.execute(inputs);
    assert.deepEqual(
      Array.from(await pinnedResult.output(logits.name).read()),
      beforeInference,
      'an existing context remains pinned to the pre-training weight revision',
    );
    await pinnedResult.close();

    await trainer.commit();
    assert.notEqual(model.weightRevisionId, originalRevision);

    compiledAfter = await model.compile({
      backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
    });
    contextAfter = await compiledAfter.createContext();
    const afterResult = await contextAfter.execute(inputs);
    const afterInference = Array.from(await afterResult.output(logits.name).read());
    await afterResult.close();
    assert.notDeepEqual(afterInference, beforeInference);
    assert.equal(argmax(afterInference), 2,
      'the corrected class should win after repeated online updates');

    const committedRevision = model.weightRevisionId;
    await trainer.trainStep(options);
    assert.equal(trainer.hasUncommittedUpdates, true);
    await trainer.rollback();
    assert.equal(trainer.trainingStep, 12);
    assert.equal(trainer.hasUncommittedUpdates, false);
    assert.equal(model.weightRevisionId, committedRevision);
  } finally {
    await contextAfter?.close();
    await compiledAfter?.close();
    await trainer?.close();
    await contextBefore?.close();
    await compiledBefore?.close();
    await model?.close();
    await runtime?.close();
    await rm(directory, { recursive: true, force: true });
  }
});
