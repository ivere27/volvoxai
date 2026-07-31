import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { Graph, ModelBuilder } from '../ts/wasm.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';
import { WasmQuantizedLoRATrainer } from '../ts/training/WasmQuantizedLoRATrainer.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

async function buildFullWasm(output, { weightSync = true } = {}) {
  const sources = [
    'native/src/kernels/kernels.c',
    'native/src/kernels/training_kernels.c',
  ];
  if (weightSync) sources.push('native/src/training/quantization.c');
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-Inative/include',
    '-o', output,
    ...sources,
  ], { cwd: repositoryRoot });
}

function trainingGraph() {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1, 2]);
  const base = builder.weight('base', [2, 2], 'float32', Float32Array.of(
    1, 0,
    0, 1,
  ));
  const lora = builder.loraLinear(input, base, {
    rank: 1,
    layout: 'din',
    aInitializer: 'ones',
    bInitializer: { type: 'normal', seed: 7, stddev: 0.2 },
    name: 'decoder',
  });
  builder.outputs(lora.out);
  return { graph: builder.graph, lora };
}

function qdescriptor(scale, zeroPoint = 0) {
  return { scheme: 'per_tensor', scale, zero_point: zeroPoint };
}

function qweightDescriptor(scales) {
  return {
    scheme: 'per_axis', axis: 0, scales, zero_points: scales.map(() => 0),
  };
}

function inferenceGraph({ nonzeroBias = false, targetVariant = null } = {}) {
  const graph = new Graph();
  const input = graph.addInput('qx', [1, 2], 'int8', {
    quantization: qdescriptor(0.25),
  });
  const aDtype = targetVariant === 'uint8' ? 'uint8' : 'int8';
  const aBuffer = targetVariant === 'uint8'
    ? Uint8Array.of(4, 8)
    : Int8Array.of(4, -8);
  const aQuantization = targetVariant === 'axis1'
    ? { scheme: 'per_axis', axis: 1, scales: [0.25, 0.5], zero_points: [0, 0] }
    : targetVariant === 'nonzero-zp'
      ? { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [1] }
      : qweightDescriptor([0.25]);
  const A = graph.addWeight('decoder.a.i8', [1, 2], aDtype, {
    buffer: aBuffer,
    quantization: aQuantization,
  });
  const aBias = graph.addWeight('decoder.a.bias', [1], 'int32', {
    buffer: Int32Array.of(nonzeroBias ? 1 : 0),
  });
  const { out: hidden } = graph.addOp('QLinear', {
    input, weight: A, bias: aBias,
  }, {
    out: {
      name: 'decoder.hidden.i8', shape: [1, 1], dtype: 'int8',
      quantization: qdescriptor(0.25),
    },
  });
  const B = graph.addWeight('decoder.b.i8', [2, 1], 'int8', {
    buffer: Int8Array.of(3, -4),
    quantization: qweightDescriptor([0.5, 0.25]),
  });
  const bBias = graph.addWeight('decoder.b.bias', [2], 'int32', {
    buffer: Int32Array.of(0, 0),
  });
  const { out } = graph.addOp('QLinear', {
    input: hidden, weight: B, bias: bBias,
  }, {
    out: {
      name: 'decoder.out.i8', shape: [1, 2], dtype: 'int8',
      quantization: qdescriptor(0.125),
    },
  });
  graph.setOutputs(out);
  return { graph, input, hidden, out, A, B };
}

function bindings() {
  return [
    { master: 'decoder.lora_a', target: 'decoder.a.i8', transpose: true },
    { master: 'decoder.lora_b', target: 'decoder.b.i8', transpose: true },
  ];
}

test('WASM quantized LoRA keeps F32 masters and refreshes one unchanged W8 graph', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-quantized-lora-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const engine = await WasmEngine.init(wasmPath);
    assert.ok(engine);

    await t.test('explicit dequantization initializes transposed F32 masters once', async () => {
      const training = trainingGraph();
      const inference = inferenceGraph();
      const session = await WasmQuantizedLoRATrainer.create(
        engine, training.graph, inference.graph, { bindings: bindings() },
      );
      try {
        const initialized = await session.initializeMastersFromQuantized();
        assert.deepEqual(initialized.map((tensor) => tensor.name), [
          'decoder.lora_a', 'decoder.lora_b',
        ]);
        assert.deepEqual([...training.lora.A.buffer], [1, -2]);
        assert.deepEqual([...training.lora.B.buffer], [1.5, -1]);
        await assert.rejects(
          session.initializeMastersFromQuantized(),
          /can only be initialized once/,
        );
      } finally {
        await session.dispose();
      }
    });

    await t.test('an applied train step stages both factors, advances once, and compiles once', async () => {
      const training = trainingGraph();
      const inference = inferenceGraph();
      const session = await WasmQuantizedLoRATrainer.create(
        engine, training.graph, inference.graph, { bindings: bindings() },
      );
      let originalAllocate = null;
      try {
        await session.sync();
        const topologyRevision = inference.graph.topologyRevision;
        const weightRevision = inference.graph.weightRevision;
        const aIdentity = inference.A;
        const bIdentity = inference.B;
        const activationDescriptors = [
          inference.input.quantization,
          inference.hidden.quantization,
          inference.out.quantization,
        ];
        const beforeA = new Int8Array(inference.A.buffer);
        const beforeB = new Int8Array(inference.B.buffer);
        const inferenceInputs = { qx: Int8Array.of(3, -1) };
        const beforeOutput = new Int8Array(
          (await engine.execute(inferenceInputs))[inference.out.name],
        );

        originalAllocate = engine.allocateGraph.bind(engine);
        let compileCount = 0;
        engine.allocateGraph = async (graph) => {
          compileCount++;
          return originalAllocate(graph);
        };
        const result = await session.trainStep({
          inputs: { x: Float32Array.of(0.75, -0.25) },
          targets: [1],
          trainableTensors: ['decoder.lora_a', 'decoder.lora_b'],
          updateMode: 'sgd',
          optimizer: { learningRate: 0.5 },
        });

        assert.equal(result.updatedTensors.length, 2);
        assert.deepEqual(result.quantizedWeights.map(({ target }) => target), [
          'decoder.a.i8', 'decoder.b.i8',
        ]);
        assert.equal(compileCount, 1);
        assert.equal(inference.graph.topologyRevision, topologyRevision);
        assert.equal(inference.graph.weightRevision, weightRevision + 1);
        assert.equal(inference.graph.getTensor(inference.A.name), aIdentity);
        assert.equal(inference.graph.getTensor(inference.B.name), bIdentity);
        assert.equal(inference.input.quantization, activationDescriptors[0]);
        assert.equal(inference.hidden.quantization, activationDescriptors[1]);
        assert.equal(inference.out.quantization, activationDescriptors[2]);
        assert.ok(
          !beforeA.every((value, index) => value === inference.A.buffer[index]) ||
          !beforeB.every((value, index) => value === inference.B.buffer[index]),
          'at least one W8 factor should change after the F32 update',
        );
        const output = await engine.execute(inferenceInputs);
        assert.ok(output[inference.out.name] instanceof Int8Array);
        assert.notDeepEqual(
          [...output[inference.out.name]],
          [...beforeOutput],
          'post-update inference must use refreshed Q8 packed weights and scales',
        );
      } finally {
        if (originalAllocate) engine.allocateGraph = originalAllocate;
        await session.dispose();
      }
    });

    await t.test('accumulation without an optimizer update performs no W8 compile', async () => {
      const training = trainingGraph();
      const inference = inferenceGraph();
      const session = await WasmQuantizedLoRATrainer.create(
        engine, training.graph, inference.graph, { bindings: bindings() },
      );
      let originalAllocate = null;
      try {
        const revision = inference.graph.weightRevision;
        originalAllocate = engine.allocateGraph.bind(engine);
        let compileCount = 0;
        engine.allocateGraph = async (graph) => {
          compileCount++;
          return originalAllocate(graph);
        };
        const result = await session.trainStep({
          inputs: { x: Float32Array.of(0.5, 0.25) },
          targets: [1],
          trainableTensors: ['decoder.lora_a', 'decoder.lora_b'],
          updateMode: 'sgd',
          optimizer: { learningRate: 0.1 },
          gradientAccumulationSteps: 2,
        });
        assert.equal(result.updatedTensors.length, 0);
        assert.equal(result.quantizedWeights.length, 0);
        assert.equal(compileCount, 0);
        assert.equal(inference.graph.weightRevision, revision);
      } finally {
        if (originalAllocate) engine.allocateGraph = originalAllocate;
        await session.dispose();
      }
    });

    await t.test('a later invalid master leaves every staged W8 target unchanged', async () => {
      const training = trainingGraph();
      const inference = inferenceGraph();
      const session = await WasmQuantizedLoRATrainer.create(
        engine, training.graph, inference.graph, { bindings: bindings() },
      );
      try {
        await session.sync();
        const revision = inference.graph.weightRevision;
        const aBytes = new Int8Array(inference.A.buffer);
        const bBytes = new Int8Array(inference.B.buffer);
        const aDescriptor = inference.A.quantization;
        const bDescriptor = inference.B.quantization;
        training.lora.B.buffer[0] = Number.NaN;

        await assert.rejects(session.sync(), /I8 weight quantization.*rejected/);
        assert.deepEqual(inference.A.buffer, aBytes);
        assert.deepEqual(inference.B.buffer, bBytes);
        assert.equal(inference.A.quantization, aDescriptor);
        assert.equal(inference.B.quantization, bDescriptor);
        assert.equal(inference.graph.weightRevision, revision);
      } finally {
        await session.dispose();
      }
    });

    await t.test('a failed inference compile restores the prior W8 snapshot and revision', async () => {
      const training = trainingGraph();
      const inference = inferenceGraph();
      const session = await WasmQuantizedLoRATrainer.create(
        engine, training.graph, inference.graph, { bindings: bindings() },
      );
      let originalAllocate = null;
      try {
        await session.sync();
        const revision = inference.graph.weightRevision;
        const aBytes = new Int8Array(inference.A.buffer);
        const bBytes = new Int8Array(inference.B.buffer);
        const aDescriptor = inference.A.quantization;
        const bDescriptor = inference.B.quantization;
        training.lora.A.buffer[0] += 0.5;

        originalAllocate = engine.allocateGraph.bind(engine);
        let compileCount = 0;
        engine.allocateGraph = async (graph) => {
          compileCount++;
          if (compileCount === 1) throw new Error('forced quantized compile failure');
          return originalAllocate(graph);
        };
        await assert.rejects(session.sync(), /forced quantized compile failure/);
        assert.equal(compileCount, 2, 'the second compile restores the previous allocation');
        assert.deepEqual(inference.A.buffer, aBytes);
        assert.deepEqual(inference.B.buffer, bBytes);
        assert.equal(inference.A.quantization, aDescriptor);
        assert.equal(inference.B.quantization, bDescriptor);
        assert.equal(inference.graph.weightRevision, revision);
        await engine.execute({ qx: Int8Array.of(3, -1) });
      } finally {
        if (originalAllocate) engine.allocateGraph = originalAllocate;
        await session.dispose();
      }
    });

    await t.test('checkpointed optimizer settings remain authoritative without call overrides', async () => {
      const training = trainingGraph();
      const inference = inferenceGraph();
      training.graph.optimizerDescriptor = {
        updateMode: 'sgd',
        optimizer: {
          learningRate: 0.125,
          weightDecay: 0.25,
          maxGradNorm: 0.75,
        },
      };
      const session = await WasmQuantizedLoRATrainer.create(
        engine, training.graph, inference.graph, { bindings: bindings() },
      );
      try {
        const result = await session.trainStep({
          inputs: { x: Float32Array.of(0.75, -0.25) },
          targets: [1],
          trainableTensors: ['decoder.lora_a', 'decoder.lora_b'],
        });
        assert.equal(result.updatedTensors.length, 2);
        assert.deepEqual(training.graph.optimizerDescriptor, {
          updateMode: 'sgd',
          optimizer: {
            learningRate: 0.125,
            weightDecay: 0.25,
            maxGradNorm: 0.75,
          },
        });
      } finally {
        await session.dispose();
      }
    });

    await t.test('nonzero accumulator bias is rejected before trainer mutation', async () => {
      const training = trainingGraph();
      const inference = inferenceGraph({ nonzeroBias: true });
      const trainingRevision = training.graph.weightRevision;
      await assert.rejects(
        WasmQuantizedLoRATrainer.create(
          engine, training.graph, inference.graph, { bindings: bindings() },
        ),
        /all-zero I32 bias/,
      );
      assert.equal(training.graph.weightRevision, trainingRevision);
    });

    await t.test('invalid target storage, descriptors, and layouts fail before compilation', async () => {
      for (const [targetVariant, pattern] of [
        ['uint8', /must be an initialized rank-2 I8 weight/],
        ['axis1', /symmetric axis-0 per-row quantization/],
        ['nonzero-zp', /symmetric axis-0 per-row quantization/],
      ]) {
        const training = trainingGraph();
        const inference = inferenceGraph({ targetVariant });
        await assert.rejects(
          WasmQuantizedLoRATrainer.create(
            engine, training.graph, inference.graph, { bindings: bindings() },
          ),
          pattern,
          targetVariant,
        );
      }
      const training = trainingGraph();
      const inference = inferenceGraph();
      const mismatched = bindings();
      mismatched[0] = { ...mismatched[0], transpose: false };
      await assert.rejects(
        WasmQuantizedLoRATrainer.create(
          engine, training.graph, inference.graph, { bindings: mismatched },
        ),
        /does not match.*with transpose|does not match target/,
      );
    });

    await t.test('a stale full sidecar is rejected before graph or optimizer mutation', async () => {
      const stalePath = join(directory, 'volvoxai.stale-full.wasm');
      await buildFullWasm(stalePath, { weightSync: false });
      const staleEngine = await WasmEngine.init(stalePath);
      assert.ok(staleEngine);
      const training = trainingGraph();
      const inference = inferenceGraph();
      const trainingRevision = training.graph.weightRevision;
      const inferenceRevision = inference.graph.weightRevision;

      await assert.rejects(
        WasmQuantizedLoRATrainer.create(
          staleEngine, training.graph, inference.graph, { bindings: bindings() },
        ),
        /volvoxai_training_quantize_weight_f32_to_i8.*unavailable/,
      );
      assert.equal(training.graph.trainingStep, 0);
      assert.equal(training.graph.optimizerState, null);
      assert.equal(training.graph.weightRevision, trainingRevision);
      assert.equal(inference.graph.weightRevision, inferenceRevision);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
