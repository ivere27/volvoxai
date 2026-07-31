import test from 'node:test';
import assert from 'node:assert/strict';

import {
  Graph,
  SafetensorsFile,
  VOLVOX_ADAPTER_FORMAT,
  VOLVOX_ADAPTER_MANIFEST_KEY,
} from '../ts/index.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { GraphExecutor } from '../ts/backends/GraphExecutor.js';

function close(actual, expected, epsilon = 1e-5) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= epsilon, `index ${i}: got ${actual[i]}, expected ${expected[i]}`);
  }
}

function rawSafetensors(header, data = new Uint8Array(0)) {
  const headerBytes = new TextEncoder().encode(header);
  const buffer = new ArrayBuffer(8 + headerBytes.byteLength + data.byteLength);
  new DataView(buffer).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  new Uint8Array(buffer, 8, headerBytes.byteLength).set(headerBytes);
  new Uint8Array(buffer, 8 + headerBytes.byteLength).set(data);
  return buffer;
}

test('safetensors byte access separates snapshots from writable views', () => {
  const file = SafetensorsFile.empty();
  const tensor = file.addTensor('values', 'U8', [2], Uint8Array.from([1, 2]));
  assert.equal(tensor.sizeBytes, 2);

  const snapshot = tensor.getBytes();
  snapshot[0] = 9;
  assert.deepEqual(tensor.getBytes(), Uint8Array.from([1, 2]));

  const writableView = tensor.getMutableBytes();
  writableView[0] = 7;
  assert.deepEqual(tensor.getBytes(), Uint8Array.from([7, 2]));

  const readOnlyFile = SafetensorsFile.fromArrayBuffer(file.toArrayBuffer());
  assert.throws(() => readOnlyFile.getMutableTensor('values'), /not opened writable/);
});

test('safetensors headers reject duplicate keys at every record level', () => {
  for (const { header, data, key } of [
    {
      header: '{"values":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"values":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
      data: Uint8Array.of(1),
      key: 'values',
    },
    {
      header: '{"__metadata__":{},"__metadata__":{}}',
      key: '__metadata__',
    },
    {
      header: '{"__metadata__":{"profile":"first","profile":"second"}}',
      key: 'profile',
    },
    {
      header: '{"values":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}',
      data: Uint8Array.of(1),
      key: 'dtype',
    },
  ]) {
    assert.throws(
      () => SafetensorsFile.fromArrayBuffer(rawSafetensors(header, data)),
      new RegExp(`Safetensors header contains duplicate object key "${key}"`),
    );
  }
});

test('safetensors metadata remains a string-to-string map on read and write', () => {
  assert.throws(
    () => SafetensorsFile.fromArrayBuffer(rawSafetensors('{"__metadata__":7}')),
    /__metadata__ must be an object mapping strings to strings/,
  );
  assert.throws(
    () => SafetensorsFile.fromArrayBuffer(rawSafetensors('{"__metadata__":{"profile":7}}')),
    /__metadata__ value for 'profile' must be a string/,
  );

  const file = SafetensorsFile.empty({
    metadata: { profile: 'custom-v1', unrelated_note: 'preserved' },
  });
  const roundTrip = SafetensorsFile.fromArrayBuffer(file.toArrayBuffer());
  assert.deepEqual(roundTrip.metadata, {
    profile: 'custom-v1',
    unrelated_note: 'preserved',
  });

  file.metadata.profile = 7;
  assert.throws(
    () => file.toArrayBuffer(),
    /__metadata__ value for 'profile' must be a string/,
  );
});

function linearGraph({ inputShape = [1, 2], weight = [1, 0, 0, 1], bias = [0.5, -0.5], dtype = 'float32', scale = null, zeroPoint = null, layout = 'din' } = {}) {
  const graph = new Graph();
  const input = graph.addInput('x', inputShape);
  const base = graph.addWeight('w', [2, 2], dtype);
  base.buffer = dtype === 'int8'
    ? Int8Array.from(weight)
    : dtype === 'uint8' ? Uint8Array.from(weight) : Float32Array.from(weight);
  const biasTensor = graph.addWeight('bias', [2]);
  biasTensor.buffer = Float32Array.from(bias);
  const inputs = { input, weight: base, bias: biasTensor };
  if (scale) {
    const scaleTensor = graph.addWeight('scale', [2]);
    scaleTensor.buffer = Float32Array.from(scale);
    inputs.scale = scaleTensor;
  }
  if (zeroPoint) {
    const zeroPointTensor = graph.addWeight('zero_point', [zeroPoint.length], 'uint8');
    zeroPointTensor.buffer = Uint8Array.from(zeroPoint);
    inputs.zero_point = zeroPointTensor;
  }
  const outShape = [...inputShape.slice(0, -1), 2];
  const { out } = graph.addOp('MatMul', inputs, { out: outShape });
  graph.nodes[0].wLayout = (dtype === 'int8' || dtype === 'uint8') ? 'dout' : layout;
  graph.setOutputs([out.name]);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return { graph, engine, out };
}

function loraTarget({ A = [1, 2], B = [3, 4], rank = 1, alpha = 1, ...extra } = {}) {
  return { weight: 'w', A: Float32Array.from(A), B: Float32Array.from(B), rank, alpha, ...extra };
}

test('adapter authoring rejects undeclared fields and duplicate control', () => {
  const { graph } = linearGraph();
  const rejectedTargetFields = [
    ['baseTensor', 'w'],
    ['target', 'w'],
    ['kind', 'lora'],
    ['a', Float32Array.of(1, 2)],
    ['b', Float32Array.of(3, 4)],
    ['adapterB', Float32Array.of(3, 4)],
  ];
  for (const [field, value] of rejectedTargetFields) {
    assert.throws(
      () => graph.stageAdapter(`unsupported-${field}`, {
        kind: 'lora',
        targets: [{ ...loraTarget(), [field]: value }],
      }),
      new RegExp(`unsupported field '${field}'`),
    );
  }
  for (const [field, value] of [['type', 'lora'], ['activate', true]]) {
    assert.throws(
      () => graph.stageAdapter(`unsupported-${field}`, {
        kind: 'lora', targets: [loraTarget()], [field]: value,
      }),
      new RegExp(`unsupported field '${field}'`),
    );
  }
  assert.throws(
    () => graph.stageAdapter('missing-kind', { targets: [loraTarget()] }),
    /requires kind 'lora'/,
  );
  assert.throws(
    () => graph.stageAdapter('record-targets', { kind: 'lora', targets: { w: loraTarget() } }),
    /targets must be an array/,
  );
  for (const field of ['values', 'buffer']) {
    assert.throws(
      () => graph.stageAdapter(`unsupported-storage-${field}`, {
        kind: 'lora',
        targets: [{
          ...loraTarget(),
          A: { data: Float32Array.of(1, 2), [field]: Float32Array.of(1, 2) },
        }],
      }),
      new RegExp(`unsupported field '${field}'`),
    );
  }

  graph.stageAdapter('canonical', { kind: 'lora', targets: [loraTarget()] });
  assert.throws(
    () => graph.updateAdapter('canonical', { w: { a: Float32Array.of(1, 2) } }),
    /Unsupported adapter tensor role 'a'/,
  );
});

test('immutable versions hot-swap atomically and execution pins before input copy', async () => {
  const { graph, engine, out } = linearGraph();
  const sourceA = new Float32Array([1, 2]);
  const metadata = { origin: 'first' };
  const first = graph.stageAdapter('hot', { kind: 'lora', targets: [loraTarget({ A: sourceA })], metadata }, { activate: true });
  sourceA.fill(100);
  metadata.origin = 'mutated';
  assert.equal(graph.listAdapters()[0].metadata.origin, 'first');
  const second = graph.stageAdapter('hot', {
    kind: 'lora',
    targets: [loraTarget({ A: [1, 0], B: [1, 1] })],
  });
  assert.equal(first.version, 1);
  assert.equal(second.version, 2);

  graph.activateAdapter('hot', 1);
  const inputBuffer = graph.getTensor('x').buffer;
  const copyInput = inputBuffer.set.bind(inputBuffer);
  inputBuffer.set = (data) => {
    graph.activateAdapter('hot', 2);
    copyInput(data);
  };
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], [14.5, 16.5]);
  delete inputBuffer.set;
  assert.equal(graph.activeAdapter().version, 2);
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], [4.5, 2.5]);
  close((await engine.execute({ x: new Float32Array([2, 1]) }, { adapter: { name: 'hot', version: 1, scale: 0.5 } }))[out.name], [8.5, 8.5]);
  close((await engine.execute({ x: new Float32Array([2, 1]) }, { adapter: null }))[out.name], [2.5, 0.5]);
  close((await engine.execute({ x: new Float32Array([2, 1]) }, { adapter: { name: 'hot', version: 1, scale: 0 } }))[out.name], [2.5, 0.5]);

  const third = graph.updateAdapter('hot', { w: { B: new Float32Array([1, 1]) } }, {
    version: 2,
    mode: 'add',
    activate: true,
  });
  assert.equal(third.version, 3);
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], [6.5, 4.5]);
  assert.throws(() => graph.removeAdapter('hot', 3), /Activate the base model/);
  assert.equal(graph.removeAdapter('hot', 1), true);
  await assert.rejects(
    () => engine.execute({ x: new Float32Array([2, 1]) }, { adapter: { name: 'hot', version: 1 } }),
    /not staged/,
  );
});

test('merge parity also holds for dout-first F32 weights', async () => {
  const { graph, engine, out } = linearGraph({ weight: [1, 2, 3, 4], layout: 'dout' });
  graph.stageAdapter('dout', { kind: 'lora', targets: [loraTarget()] }, { activate: true });
  const expected = new Float32Array((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name]);
  close(expected, [16.5, 25.5]);
  graph.mergeAdapter('dout');
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], expected);
  graph.unmergeAdapter();
});

test('LoRA merge matches unmerged output and unmerge restores exact base', async () => {
  const { graph, engine, out } = linearGraph();
  const original = new Float32Array(graph.getTensor('w').buffer);
  graph.stageAdapter('merge', { kind: 'lora', targets: [loraTarget()] }, { activate: true });
  const expected = new Float32Array((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name]);

  graph.mergeAdapter('merge');
  assert.equal(graph.activeAdapter(), null);
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], expected);
  assert.throws(() => graph.activateAdapter(null), /Unmerge/);
  assert.throws(() => graph.setTensorBuffer('w', new Float32Array(original)), /Unmerge/);
  assert.throws(() => graph.applyTensorUpdate('w', new Float32Array(4), { mode: 'add' }), /Unmerge/);
  assert.throws(() => graph.stageAdapter('late', { kind: 'lora', targets: [loraTarget()] }, { activate: true }), /Unmerge/);
  assert.equal(graph.listAdapters().some((adapter) => adapter.name === 'late'), false);
  await assert.rejects(() => engine.execute({ x: new Float32Array([2, 1]) }, { adapter: { name: 'merge' } }), /merged/);
  assert.throws(() => graph.mergeAdapter('merge'), /already merged/);

  const mergedFirst = graph.getTensor('w').buffer[0];
  graph.getTensor('w').buffer[0] += 1;
  assert.throws(() => graph.unmergeAdapter(), /Merged base tensor 'w' changed/);
  graph.getTensor('w').buffer[0] = mergedFirst;

  graph.unmergeAdapter();
  assert.deepEqual(graph.getTensor('w').buffer, original);
  assert.equal(graph.activeAdapter().name, 'merge');
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], expected);
  assert.throws(() => graph.patchNodeAt(0, { params: { transB: 1 } }), /Remove all adapter versions/);
});

test('LoRA accepts PEFT matrix layout and rejects other adapter kinds', async (t) => {
  await t.test('PEFT A=[rank,din], B=[dout,rank] is normalized explicitly', async () => {
    const { graph, engine, out } = linearGraph();
    graph.stageAdapter('peft', {
      kind: 'lora',
      targets: [{
        weight: 'w',
        rank: 1,
        alpha: 1,
        layout: 'peft',
        A: { data: new Float32Array([1, 2]), shape: [1, 2] },
        B: { data: new Float32Array([3, 4]), shape: [2, 1] },
      }],
    }, { activate: true });
    close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], [14.5, 16.5]);
  });

  assert.throws(() => {
    const { graph } = linearGraph();
    graph.stageAdapter('unsupported', { kind: 'other', targets: [loraTarget()] });
  }, /Only LoRA is supported/);

  assert.throws(() => {
    const { graph } = linearGraph({ dtype: 'int8', scale: [1, 1] });
    graph.stageAdapter('quantized-base', { kind: 'lora', targets: [loraTarget()] });
  }, /requires an F32 runtime base tensor/);

  assert.throws(() => {
    const { graph } = linearGraph();
    graph.stageAdapter('zero-alpha', { kind: 'lora', targets: [loraTarget({ alpha: 0 })] });
  }, /positive finite alpha/);

  assert.throws(() => {
    const { graph } = linearGraph();
    graph.stageAdapter('bad-layout', {
      kind: 'lora',
      targets: [loraTarget({ layout: 'definitely_invalid' })],
    });
  }, /unsupported matrix layout/);
});

test('batch routes apply distinct LoRA adapters per request', async () => {
  const { graph, engine, out } = linearGraph({ inputShape: [2, 1, 2], bias: [0, 0] });
  graph.stageAdapter('a', { kind: 'lora', targets: [loraTarget({ A: [1, 0], B: [1, 1] })] });
  graph.stageAdapter('b', { kind: 'lora', targets: [loraTarget({ A: [0, 1], B: [2, 4] })] });
  const result = await engine.execute({ x: new Float32Array([2, 1, 3, 4]) }, {
    adapters: [{ name: 'a' }, { name: 'b', scale: 0.5 }],
  });
  close(result[out.name], [4, 3, 7, 12]);
  await assert.rejects(
    () => engine.execute({ x: new Float32Array([2, 1, 3, 4]) }, { adapter: 'a' }),
    /selector must be an object or null/,
  );
  close((await engine.execute({ x: new Float32Array([2, 1, 3, 4]) }, { adapters: [{ name: 'a' }] }))[out.name], [4, 3, 6, 7]);
  close((await engine.execute({ x: new Float32Array([2, 1, 3, 4]) }, { adapters: [] }))[out.name], [2, 1, 3, 4]);
  close((await engine.execute({ x: new Float32Array([2, 1, 3, 4]) }, { adapters: [null, null] }))[out.name], [2, 1, 3, 4]);
  await assert.rejects(
    () => engine.execute({ x: new Float32Array([2, 1, 3, 4]) }, { adapters: [{ name: 'a' }, { name: 'b' }, { name: 'a' }] }),
    /requires one adapter route or 2 routes.*received 3/,
  );
  await assert.rejects(
    () => engine.execute({ x: new Float32Array([2, 1, 3, 4]) }, { adapters: [null, null, null] }),
    /requires one adapter route or 2 routes.*received 3/,
  );

  const folded = linearGraph({ inputShape: [2, 2, 2], bias: [0, 0] });
  folded.graph.stageAdapter('a', { kind: 'lora', targets: [loraTarget({ A: [1, 0], B: [1, 1] })] });
  folded.graph.stageAdapter('b', { kind: 'lora', targets: [loraTarget({ A: [0, 1], B: [2, 4] })] });
  const foldedResult = await folded.engine.execute({ x: new Float32Array([2, 1, 3, 4, 5, 6, 7, 8]) }, {
    adapters: [{ name: 'a' }, { name: 'b' }],
  });
  close(foldedResult[folded.out.name], [4, 3, 6, 7, 17, 30, 23, 40]);
  await assert.rejects(
    () => folded.engine.execute({ x: new Float32Array(8) }, { adapters: [{ name: 'a' }, { name: 'b' }, { name: 'a' }, { name: 'b' }] }),
    /requires one adapter route or 2 routes.*received 4/,
  );
});

test('adapter-only safetensors round-trip carries the v1 manifest', async () => {
  const { graph, engine, out } = linearGraph();
  graph.stageAdapter('portable', {
    kind: 'lora',
    targets: [loraTarget()],
    metadata: { dataset: 'fixture-v1' },
  });
  assert.ok(graph.exportAdapter('portable', { as: 'file' }) instanceof SafetensorsFile);
  const bytes = graph.exportAdapter('portable');
  const file = SafetensorsFile.fromArrayBuffer(bytes);
  const manifest = JSON.parse(file.metadata[VOLVOX_ADAPTER_MANIFEST_KEY]);
  assert.equal(manifest.format, VOLVOX_ADAPTER_FORMAT);
  assert.equal(manifest.adapter_id, 'portable');
  assert.deepEqual(manifest.metadata, { dataset: 'fixture-v1' });
  assert.equal(manifest.targets[0].layout, 'din_r_r_dout');
  assert.deepEqual(file.listTensorNames(), ['adapter.0.a', 'adapter.0.b']);

  assert.throws(() => graph.loadAdapter(bytes, { name: 'loaded' }), /must match/);
  const loaded = graph.loadAdapter(bytes, { activate: true });
  assert.equal(loaded.sourceVersion, '1');
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], [14.5, 16.5]);

  const missingLayout = JSON.parse(JSON.stringify(manifest));
  delete missingLayout.targets[0].layout;
  file.metadata[VOLVOX_ADAPTER_MANIFEST_KEY] = JSON.stringify(missingLayout);
  assert.throws(() => graph.loadAdapter(file), /explicit matrix layout/);
  const missingKind = JSON.parse(JSON.stringify(manifest));
  delete missingKind.kind;
  file.metadata[VOLVOX_ADAPTER_MANIFEST_KEY] = JSON.stringify(missingKind);
  assert.throws(() => graph.loadAdapter(file), /Only LoRA is supported/);
  const missingAlpha = JSON.parse(JSON.stringify(manifest));
  delete missingAlpha.targets[0].alpha;
  file.metadata[VOLVOX_ADAPTER_MANIFEST_KEY] = JSON.stringify(missingAlpha);
  assert.throws(() => graph.loadAdapter(file), /positive finite alpha/);
  const targetMetadata = JSON.parse(JSON.stringify(manifest));
  targetMetadata.targets[0].metadata = { unsupported: 'value' };
  file.metadata[VOLVOX_ADAPTER_MANIFEST_KEY] = JSON.stringify(targetMetadata);
  assert.throws(() => graph.loadAdapter(file), /unsupported field 'metadata'/);
  const mixedKind = JSON.parse(JSON.stringify(manifest));
  mixedKind.targets[0].kind = 'other';
  file.metadata[VOLVOX_ADAPTER_MANIFEST_KEY] = JSON.stringify(mixedKind);
  assert.throws(() => graph.loadAdapter(file), /Only LoRA is supported/);

  assert.throws(() => graph.stageAdapter('bad-metadata', {
    kind: 'lora',
    targets: [loraTarget()],
    metadata: { nested: { unsupported: true } },
  }), /must be a string/);
});

test('adapter safetensors import accepts F16 tensors', async () => {
  const { graph, engine, out } = linearGraph();
  const f16Bytes = (bits) => {
    const bytes = new Uint8Array(bits.length * 2);
    const view = new DataView(bytes.buffer);
    bits.forEach((value, index) => view.setUint16(index * 2, value, true));
    return bytes;
  };
  const manifest = {
    format: VOLVOX_ADAPTER_FORMAT,
    adapter_id: 'f16',
    version_id: 'source-1',
    kind: 'lora',
    targets: [{
      weight: 'w',
      a: 'a',
      b: 'b',
      layout: 'din_r_r_dout',
      rank: 1,
      alpha: 1,
      scale: 1,
      tensors: [
        { role: 'a', name: 'a', shape: [2, 1], dtype: 'F16' },
        { role: 'b', name: 'b', shape: [1, 2], dtype: 'F16' },
      ],
    }],
  };
  const file = SafetensorsFile.empty({
    metadata: { [VOLVOX_ADAPTER_MANIFEST_KEY]: JSON.stringify(manifest) },
  });
  file.addTensor('a', 'F16', [2, 1], f16Bytes([0x3c00, 0x4000]));
  file.addTensor('b', 'F16', [1, 2], f16Bytes([0x4200, 0x4400]));
  graph.loadAdapter(file, { activate: true });
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], [14.5, 16.5]);

  const mismatched = JSON.parse(JSON.stringify(manifest));
  mismatched.targets[0].tensors[0].shape = [1, 2];
  file.metadata[VOLVOX_ADAPTER_MANIFEST_KEY] = JSON.stringify(mismatched);
  assert.throws(() => graph.loadAdapter(file), /tensor spec does not match its payload/);
});

test('adapter staging converts numeric views by value and rejects non-finite payloads', async () => {
  const { graph, engine, out } = linearGraph();
  graph.stageAdapter('f64', {
    kind: 'lora',
    targets: [loraTarget({ A: new Float64Array([1, 2]), B: new Int16Array([3, 4]) })],
  }, { activate: true });
  close((await engine.execute({ x: new Float32Array([2, 1]) }))[out.name], [14.5, 16.5]);

  assert.throws(() => graph.stageAdapter('invalid', {
    kind: 'lora',
    targets: [loraTarget({ A: [1, Number.NaN] })],
  }), /non-finite/);
});

test('WebGPU prepares adapter routing and compiled graphs reject later merges', async () => {
  globalThis.GPUBufferUsage ||= { STORAGE: 1, COPY_DST: 2, UNIFORM: 4 };
  const { graph } = linearGraph();
  graph.stageAdapter('gpu', { kind: 'lora', targets: [loraTarget()] }, { activate: true });
  const pipeline = { getBindGroupLayout: () => ({}) };
  const device = {
    createShaderModule: () => ({}),
    createComputePipelineAsync: async () => pipeline,
    createBuffer: () => ({
      destroyed: false,
      destroy() { this.destroyed = true; },
    }),
    createBindGroup: () => ({}),
    queue: { writeBuffer() {} },
  };
  const executor = new GraphExecutor(device, graph);
  for (const tensor of graph.tensors.values()) executor.gpuBuffers.set(tensor.name, {});
  executor.adapterPipeline = pipeline;
  const plan = graph.adapters._pinExecution({});
  const dispatches = await executor._prepareAdapterDispatches(plan);
  assert.equal(dispatches.get(0).length, 1);
  const factors = [...executor.adapterTargetBuffers.values()][0];

  graph.activateAdapter(null);
  graph.adapters._markAcceleratedBackend('test-backend');
  assert.throws(() => graph.mergeAdapter('gpu'), /Cannot merge after accelerated compilation/);
  assert.equal(graph.removeAdapter('gpu', 1), true);
  assert.equal(executor.adapterTargetBuffers.size, 0);
  assert.equal(factors.a.destroyed, true);
  assert.equal(factors.b.destroyed, true);
  executor.dispose();
});
