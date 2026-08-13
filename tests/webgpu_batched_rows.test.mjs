import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { WebGPUEngine } from '../ts/backends/WebGPUEngine.js';
import { PagedKVCache } from '../ts/core/PagedKVCache.js';
import { kvPagePlanForLane } from '../ts/backends/kvPageAddressing.js';
import {
  batchPrefixSourceRows, decodeRowSet, rowSpanCopyRuns, stageKeepMask,
  writeRowIndices,
} from '../ts/backends/decodeRowSet.js';

/*
 * B>1 WebGPU decode, at the level a host without a GPU can settle: which bytes
 * the step moves, and how many dispatches it takes to move them.
 *
 * The arithmetic is not in scope here and does not need to be -- the attention
 * kernel already carried a batch axis and a `[B,K]` keep mask, and the RTX 3090
 * campaign checks the numbers. What changed for B>1 is the *plumbing*: a row is
 * no longer one contiguous span, so the step gathers a set of rows, stages a
 * padded per-lane K/V prefix, uploads a keep mask that exists nowhere until the
 * host builds it, and scatters back skipping parked lanes. Those fail
 * differently from a wrong kernel, which is why they are asserted separately --
 * the same split `test_paged_attention` and `test_paged_decode` make in C.
 *
 * Every expectation is computed with the *shared* addressing functions rather
 * than restated here. A test that recomputed the mapping its own way would pass
 * a WebGPU path that had quietly grown a second arithmetic, which is the exact
 * failure this work removed.
 */

globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1, COPY_SRC: 2, COPY_DST: 4, STORAGE: 8, UNIFORM: 16,
});
globalThis.GPUMapMode ??= Object.freeze({ READ: 1 });

const SEQUENCE = 8;
const WIDTH = 8;
const VOCABULARY = 6;
const HEADS = 2;
const LANES = 2;
const PAGED_TENSORS = ['self.k', 'self.v'];

/* The compiler only needs a distinct source string per kernel; nothing here
 * executes WGSL. Naming them by the accessor keeps the dispatch assertions
 * readable without listing every shader the library exposes. */
const ShaderLibrary = new Proxy({}, {
  get: (_target, property) => (...args) =>
    `${String(property).replace(/^get|Shader$/g, '').toLowerCase()}` +
    (args.length ? `:${args.map(String).join(',')}` : ''),
});

/*
 * Records what a step asks the device to do, and moves the bytes so a later
 * assertion can read the staged contents rather than trusting the offsets.
 *
 * It also enforces the usage flags a real driver enforces. That is not
 * pedantry: buffers are allocated once at compile time from the set of tensors
 * some row candidate stages, and a lane count that set did not anticipate asks
 * for a copy out of a buffer with no COPY_SRC. A real adapter drops the whole
 * command buffer, every row keeps its seed value, and it presents as an
 * attention bug. A permissive mock passes and proves nothing.
 */
function requireUsage(buffer, flag, role) {
  if (((buffer?.descriptor?.usage ?? 0) & flag) === 0) {
    throw new Error(
      `GPU buffer '${buffer?.descriptor?.label ?? '<unlabeled>'}' lacks the usage a ` +
      `${role} needs; a real adapter would drop the command buffer.`);
  }
}

function recordingDevice() {
  const state = { operations: [], writes: [], buffers: [] };
  const device = {
    limits: {
      maxBufferSize: 1 << 28,
      maxStorageBufferBindingSize: 1 << 28,
      maxComputeWorkgroupsPerDimension: 65535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64,
      maxBindingsPerBindGroup: 8,
      maxBindGroups: 4,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBuffersPerShaderStage: 12,
    },
    lost: new Promise(() => {}),
    createBuffer(descriptor) {
      const buffer = {
        descriptor,
        label: descriptor.label,
        size: descriptor.size,
        bytes: new Uint8Array(descriptor.size),
        destroyed: false,
        async mapAsync() {},
        getMappedRange() { return this.bytes.buffer.slice(0); },
        unmap() {},
        destroy() { this.destroyed = true; },
      };
      state.buffers.push(buffer);
      return buffer;
    },
    createShaderModule({ code }) { return { code }; },
    async createComputePipelineAsync({ compute }) {
      return { code: compute.module.code, getBindGroupLayout() { return {}; } };
    },
    createBindGroup(descriptor) { return descriptor; },
    createCommandEncoder() {
      const operations = [];
      return {
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          requireUsage(source, GPUBufferUsage.COPY_SRC, 'copy source');
          requireUsage(destination, GPUBufferUsage.COPY_DST, 'copy destination');
          operations.push({
            type: 'copy', source, sourceOffset, destination, destinationOffset, size,
          });
        },
        clearBuffer(destination, offset, size) {
          requireUsage(destination, GPUBufferUsage.COPY_DST, 'clear destination');
          operations.push({ type: 'clear', destination, offset, size });
        },
        beginComputePass() {
          let pipeline;
          let bindGroup;
          return {
            setPipeline(value) { pipeline = value; },
            setBindGroup(_index, value) { bindGroup = value; },
            dispatchWorkgroups(x, y, z) {
              operations.push({ type: 'dispatch', pipeline, bindGroup, x, y, z });
            },
            end() {},
          };
        },
        finish() { return { operations }; },
      };
    },
    queue: {
      writeBuffer(destination, offset, source, sourceOffset = 0, size = undefined) {
        requireUsage(destination, GPUBufferUsage.COPY_DST, 'write destination');
        const view = ArrayBuffer.isView(source);
        const unit = view && !(source instanceof DataView) ? source.BYTES_PER_ELEMENT : 1;
        const base = (view ? source.byteOffset : 0) + sourceOffset * unit;
        const length = size === undefined
          ? (view ? source.byteLength : source.byteLength) - sourceOffset * unit
          : size * unit;
        const bytes = new Uint8Array(view ? source.buffer : source, base, length);
        destination.bytes.set(bytes, offset);
        state.writes.push({
          destination, offset, size: length, bytes: Uint8Array.from(bytes),
        });
      },
      submit(commandBuffers) {
        for (const commandBuffer of commandBuffers) {
          for (const operation of commandBuffer.operations) {
            state.operations.push(operation);
            if (operation.type === 'copy') {
              operation.destination.bytes.set(operation.source.bytes.subarray(
                operation.sourceOffset, operation.sourceOffset + operation.size,
              ), operation.destinationOffset);
            } else if (operation.type === 'clear') {
              operation.destination.bytes.fill(
                0, operation.offset, operation.offset + operation.size);
            }
          }
        }
      },
      onSubmittedWorkDone: async () => {},
    },
  };
  return { device, state };
}

const perTensor = () => ({ scheme: 'per_tensor', scale: 0.125, zero_point: 0 });

function byteWeight(graph, name, rows, columns, seed) {
  const values = new Int8Array(rows * columns);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      values[row * columns + column] = ((row * 3 + column * 5 + seed) % 9) - 4;
    }
  }
  return graph.addWeight(name, [rows, columns], 'int8', {
    buffer: values,
    quantization: {
      scheme: 'per_axis', axis: 0,
      scales: new Array(rows).fill(0.125), zero_points: new Array(rows).fill(0),
    },
  });
}

function qlinear(graph, input, name, seed, lanes) {
  const weight = byteWeight(graph, `${name}.weight`, WIDTH, WIDTH, seed);
  const bias = graph.addWeight(`${name}.bias`, [WIDTH], 'int32', {
    buffer: new Int32Array(WIDTH),
  });
  return graph.addOp('QLinear', { input, weight, bias }, {
    out: {
      name, shape: [lanes, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }).out;
}

function decoderGraph(lanes) {
  const graph = new RuntimeGraph();
  const ids = graph.addInput('y_ids', [lanes, SEQUENCE], 'int32');
  const keep = graph.addInput('y_keep', [lanes, SEQUENCE], 'int32');
  const embedding = byteWeight(graph, 'embedding.weight', VOCABULARY, WIDTH, 1);
  const embedded = graph.addOp('QEmbedding', { input: ids, weight: embedding }, {
    out: {
      name: 'embed', shape: [lanes, SEQUENCE, WIDTH], dtype: 'int8',
      quantization: perTensor(),
    },
  }).out;
  const q = qlinear(graph, embedded, 'self.q', 2, lanes);
  const k = qlinear(graph, embedded, 'self.k', 3, lanes);
  const v = qlinear(graph, embedded, 'self.v', 4, lanes);
  const attended = graph.addOp('QSDPA', { q, k, v, mask: keep }, {
    out: {
      name: 'self.attention', shape: [lanes, SEQUENCE, WIDTH], dtype: 'int8',
      quantization: perTensor(),
    },
  }, { heads: HEADS, causal: true, scale: 0.5 }).out;
  const logits = qlinear(graph, attended, 'logits', 5, lanes);
  graph.setOutputs([logits.name]);
  return graph;
}

function denseInputs(lanes, lengths) {
  const ids = new Int32Array(lanes * SEQUENCE);
  const keep = new Int32Array(lanes * SEQUENCE);
  for (let lane = 0; lane < lanes; lane++) {
    for (let position = 0; position < lengths[lane]; position++) {
      ids[lane * SEQUENCE + position] = 1 + ((lane * 3 + position) % (VOCABULARY - 1));
      keep[lane * SEQUENCE + position] = 1;
    }
  }
  return { y_ids: ids, y_keep: keep };
}

async function seededEngine(lanes) {
  const { device, state } = recordingDevice();
  const engine = new WebGPUEngine(device, { shaderLibrary: ShaderLibrary });
  const graph = decoderGraph(lanes);
  await engine.allocateGraph(graph);
  await engine.execute(denseInputs(lanes, new Array(lanes).fill(1)), {
    incremental: true, incrementalReset: true, changedInputs: ['y_ids', 'y_keep'],
  });
  state.operations.length = 0;
  state.writes.length = 0;
  return { engine, graph, state, device };
}

/** The compiled row plan of the node producing `outputName`. */
function planFor(engine, outputName) {
  for (const plan of engine.executor.incrementalRowPlans.values()) {
    if (Object.values(plan.node.outputs || {})
      .some((tensor) => tensor?.name === outputName)) return plan;
  }
  return null;
}

/** The buffer a named tensor's scratch staging lives in, for one row plan. */
function scratchBuffer(engine, outputName, tensorName) {
  return planFor(engine, outputName)?.scratchByName.get(tensorName) ?? null;
}

test('a batched WebGPU row step gathers each lane at its own position', async () => {
  const { engine, state } = await seededEngine(LANES);
  /* Staggered on purpose: lane 0 is one token ahead for the whole run, so no
   * step has both lanes at the padded maximum. A padded-maximum implementation
   * is wrong only on the short lane. */
  const positions = [3, 2];
  const lengths = positions.map((position) => position + 1);
  await engine.execute(denseInputs(LANES, lengths), {
    incremental: true,
    changedInputs: ['y_ids', 'y_keep'],
    incrementalRowLanes: positions.map((position) => ({ position })),
  });

  const rowSet = decodeRowSet(positions.map((position) => ({ position })));
  const plan = planFor(engine, 'self.q');
  assert.ok(plan, 'the query projection has a compiled row plan');
  assert.equal(plan.lanes, LANES, 'the plan is compiled for the declared lanes');

  const embedScratch = scratchBuffer(engine, 'self.q', 'embed');
  const rowBytes = WIDTH; // int8
  const expected = writeRowIndices(rowSet, SEQUENCE, false);
  const gathers = state.operations.filter((operation) =>
    operation.type === 'copy' && operation.destination === embedScratch);
  assert.equal(gathers.length, LANES,
    'one gather per lane: the lanes are at different rows, so the rows do not merge');
  for (let lane = 0; lane < LANES; lane++) {
    assert.equal(gathers[lane].sourceOffset, expected[lane] * rowBytes,
      `lane ${lane} reads row ${expected[lane]}`);
    assert.equal(gathers[lane].destinationOffset, lane * rowBytes,
      `lane ${lane} stages into its own dense row`);
    assert.equal(gathers[lane].size, rowBytes);
  }
});

test('a batched step dispatches once per node, not once per lane', async () => {
  const { engine, state } = await seededEngine(LANES);
  const positions = [3, 2];
  await engine.execute(denseInputs(LANES, positions.map((p) => p + 1)), {
    incremental: true,
    changedInputs: ['y_ids', 'y_keep'],
    incrementalRowLanes: positions.map((position) => ({ position })),
  });
  const dispatches = state.operations.filter((operation) => operation.type === 'dispatch');
  const nodes = engine.executor.incrementalRowPlans.size;
  assert.ok(nodes >= 5, 'the decoder closure covers embedding, q/k/v, attention, logits');
  assert.equal(dispatches.length, nodes,
    'the whole batch is one dispatch per node; lanes never multiply the kernel count');

  /* The claim that makes it a batch: every attention dispatch covers `lanes`
   * along the workgroup batch axis rather than one. */
  const attention = dispatches.find((operation) => operation.pipeline.code === 'qsdpa');
  assert.ok(attention, 'the attention node dispatched');
  assert.equal(attention.z, LANES, 'the attention grid spans every lane');
});

test('a batched step stages the padded K/V prefix and zeroes what no lane wrote', async () => {
  const { engine, state } = await seededEngine(LANES);
  const positions = [3, 2];
  await engine.execute(denseInputs(LANES, positions.map((p) => p + 1)), {
    incremental: true,
    changedInputs: ['y_ids', 'y_keep'],
    incrementalRowLanes: positions.map((position) => ({ position })),
  });

  const rowSet = decodeRowSet(positions.map((position) => ({ position })));
  assert.equal(rowSet.keyCapacity, 4, 'the padded extent is the longest lane');
  const keyScratch = scratchBuffer(engine, 'self.attention', 'self.k');
  assert.ok(keyScratch, 'the batched attention stages its key operand');

  const sources = batchPrefixSourceRows(rowSet, SEQUENCE, false);
  const copies = state.operations.filter((operation) =>
    operation.type === 'copy' && operation.destination === keyScratch);
  const clears = state.operations.filter((operation) =>
    operation.type === 'clear' && operation.destination === keyScratch);

  /* Lane 0's prefix is four consecutive rows and lane 1's is three, so the run
   * merge yields one copy per lane rather than one per token -- and lane 1's
   * fourth staged row is padding no source fills. */
  assert.equal(copies.length, 2, 'each lane contributes one merged run');
  assert.equal(copies[0].sourceOffset, sources[0] * WIDTH);
  assert.equal(copies[0].size, 4 * WIDTH);
  assert.equal(copies[1].sourceOffset, sources[rowSet.keyCapacity] * WIDTH);
  assert.equal(copies[1].size, 3 * WIDTH);
  assert.equal(clears.length, 1, 'the one unfilled staged row is cleared');
  assert.equal(clears[0].offset, (rowSet.keyCapacity + 3) * WIDTH);
  assert.equal(clears[0].size, WIDTH);
  assert.deepEqual(
    Array.from(keyScratch.bytes.subarray(
      (rowSet.keyCapacity + 3) * WIDTH, (rowSet.keyCapacity + 4) * WIDTH)),
    new Array(WIDTH).fill(0),
    'padding carries zeroes, not the retained key of another request');
});

test('a batched step uploads the keep mask the shared builder produces', async () => {
  const { engine, state } = await seededEngine(LANES);
  const positions = [3, 2];
  const inputs = denseInputs(LANES, positions.map((p) => p + 1));
  await engine.execute(inputs, {
    incremental: true,
    changedInputs: ['y_ids', 'y_keep'],
    incrementalRowLanes: positions.map((position) => ({ position })),
  });

  const rowSet = decodeRowSet(positions.map((position) => ({ position })));
  const plan = planFor(engine, 'self.attention');
  assert.ok(plan?.keepMask, 'the batched attention carries a keep-mask spec');
  assert.equal(plan.keepMask.sourceName, 'y_keep',
    'the mask reads the graph input the caller supplied');

  const maskBuffer = plan.scratchByName.get(plan.keepMask.name);
  const upload = state.writes.find((write) => write.destination === maskBuffer);
  assert.ok(upload, 'the mask is uploaded, not copied: no tensor holds it');

  /* Built independently with the same function CPU and WASM use. Restating the
   * rule here instead would let a WebGPU-only mask drift from theirs. */
  const source = inputs.y_keep;
  const reference = stageKeepMask(
    LANES, rowSet.keyCapacity, rowSet.kvLengths,
    (lane, key) => source[lane * SEQUENCE + key] !== 0,
    'reference', (_key, elements) => new Int32Array(elements), new Int32Array(1));
  reference.apply();
  assert.deepEqual(
    Array.from(new Int32Array(
      upload.bytes.buffer, upload.bytes.byteOffset, LANES * rowSet.keyCapacity)),
    Array.from(reference.storage),
    'the uploaded mask is exactly what the shared builder produces');

  /* And the lengths really are in it: lane 1 is one token shorter, so the
   * padded key its operand shares is masked out for that lane alone. */
  const staged = new Int32Array(
    upload.bytes.buffer, upload.bytes.byteOffset, LANES * rowSet.keyCapacity);
  assert.equal(staged[rowSet.keyCapacity - 1], 1, 'lane 0 sees the padded extent');
  assert.equal(staged[LANES * rowSet.keyCapacity - 1], 0,
    'lane 1 stops at its own length');
});

test('a parked lane occupies a row and its result is discarded', async () => {
  const { engine, state } = await seededEngine(LANES);
  await engine.execute(denseInputs(LANES, [4, 3]), {
    incremental: true,
    changedInputs: ['y_ids', 'y_keep'],
    incrementalRowLanes: [{ position: 3 }, { parked: true }],
  });

  const plan = planFor(engine, 'self.q');
  const outputScratch = plan.scratchByName.get('self.q');
  const scatters = state.operations.filter((operation) =>
    operation.type === 'copy' && operation.source === outputScratch);
  assert.equal(scatters.length, 1,
    'only the live lane is written back; the parked one computed a row nobody keeps');
  assert.equal(scatters[0].destinationOffset, 3 * WIDTH,
    'the live lane writes its own position');

  const dispatches = state.operations.filter((operation) => operation.type === 'dispatch');
  const attention = dispatches.find((operation) => operation.pipeline.code === 'qsdpa');
  assert.equal(attention.z, LANES,
    'the parked lane still occupies a dense row, so the grid does not shrink');
});

test('batched lanes address their own pages', async () => {
  const { engine, state } = await seededEngine(LANES);
  /* One page per token and two lanes claiming alternately is what scatters a
   * mapping; it is what a scheduler running two requests concurrently makes. */
  const cache = new PagedKVCache({
    lanes: LANES, pageTokens: 1, laneTokenCapacity: SEQUENCE / 2,
    maxPages: SEQUENCE, policy: 'paged',
  });
  /* Staggered so the lanes' current slots are not adjacent either: two lanes at
   * equal lengths land on consecutive physical pages, which merges into one
   * copy and would prove nothing about addressing them separately. */
  for (let step = 0; step < 3; step++) {
    for (let lane = 0; lane < LANES; lane++) {
      if (lane === 1 && step === 2) continue;
      cache.append(lane, 1);
    }
  }
  const pages = [0, 1].map((lane) => kvPagePlanForLane(cache, lane, PAGED_TENSORS));
  const positions = pages.map((plan) => plan.kvLength - 1);

  await engine.execute(denseInputs(LANES, positions.map((p) => p + 1)), {
    incremental: true,
    changedInputs: ['y_ids', 'y_keep'],
    incrementalRowLanes: positions.map((position, lane) => ({
      position, kvPages: pages[lane],
    })),
  });

  const rowSet = decodeRowSet(positions.map((position, lane) => ({
    position, kvPages: pages[lane],
  })));
  const expected = writeRowIndices(rowSet, SEQUENCE, true);
  assert.notDeepEqual(Array.from(expected), Array.from(writeRowIndices(rowSet, SEQUENCE, false)),
    'the mapping is genuinely scattered; an identity run would prove nothing');
  const runs = rowSpanCopyRuns(expected);
  assert.equal(runs.length, LANES,
    'the lanes hold non-adjacent slots, so their writes do not merge');

  const keyPlan = planFor(engine, 'self.k');
  const keyScratch = keyPlan.scratchByName.get('self.k');
  const scatters = state.operations.filter((operation) =>
    operation.type === 'copy' && operation.source === keyScratch);
  assert.equal(scatters.length, runs.length);
  for (let index = 0; index < runs.length; index++) {
    assert.equal(scatters[index].destinationOffset, runs[index].source * WIDTH,
      `run ${index} writes the physical slot the page table names`);
    assert.equal(scatters[index].sourceOffset, runs[index].destination * WIDTH);
  }

  /* The read half. Each lane's visible prefix is gathered out of its own pages
   * into logical order, which is where a page table that was ignored would
   * quietly hand attention the other lane's tokens. */
  const attentionPlan = planFor(engine, 'self.attention');
  const prefixScratch = attentionPlan.scratchByName.get('self.k');
  const sources = batchPrefixSourceRows(rowSet, SEQUENCE, true);
  const gathers = state.operations.filter((operation) =>
    operation.type === 'copy' && operation.destination === prefixScratch);
  assert.deepEqual(
    gathers.map((operation) => [operation.sourceOffset / WIDTH, operation.size / WIDTH]),
    rowSpanCopyRuns(sources).map((run) => [run.source, run.rows]),
    'the staged prefix is gathered run by run from the pages each lane holds');
});

test('one lane stages exactly what it staged before batching existed', async () => {
  const { engine, state } = await seededEngine(1);
  await engine.execute(denseInputs(1, [4]), {
    incremental: true,
    changedInputs: ['y_ids', 'y_keep'],
    incrementalRowPosition: 3,
  });

  /* The identity tier. One lane's rows are trivially consecutive, its causal
   * K/V is a window based at slot zero, and its keep mask is read whole -- so
   * the K/V operands are bound directly and nothing is staged for them. */
  const plan = planFor(engine, 'self.attention');
  assert.ok(plan, 'the one-lane attention still has a row plan');
  assert.equal(plan.keepMask, null,
    'one lane publishes its length through seq_kv, not through a built mask');
  assert.ok(!plan.scratchInputs.has('self.k'), 'the one-lane key operand is not staged');
  assert.ok(!plan.scratchInputs.has('self.v'), 'the one-lane value operand is not staged');
  assert.equal(state.operations.filter((operation) => operation.type === 'clear').length, 0,
    'nothing is padded, so nothing is cleared');

  const outputScratch = plan.scratchByName.get('self.attention');
  const scatters = state.operations.filter((operation) =>
    operation.type === 'copy' && operation.source === outputScratch);
  assert.equal(scatters.length, 1);
  assert.equal(scatters[0].destinationOffset, 3 * WIDTH);
});

test('a batched step stages the token ids the caller supplied this step', async () => {
  /* The first node in the closure, and the one every later mismatch is
   * downstream of. If the ids never reach the staged operand the whole step
   * silently re-embeds whatever the seed left, which looks like an attention
   * bug and is not one. */
  const { engine, state, device } = await seededEngine(LANES);
  const positions = [3, 2];
  const inputs = denseInputs(LANES, positions.map((p) => p + 1));
  await engine.execute(inputs, {
    incremental: true,
    changedInputs: ['y_ids', 'y_keep'],
    incrementalRowLanes: positions.map((position) => ({ position })),
  });
  void device;
  const idsBuffer = engine.executor.gpuBuffers.get('y_ids');
  assert.deepEqual(
    Array.from(new Int32Array(idsBuffer.bytes.buffer, idsBuffer.bytes.byteOffset,
      LANES * SEQUENCE)),
    Array.from(inputs.y_ids),
    'the whole ids input reaches the device for a batched step');

  const plan = planFor(engine, 'embed');
  const staged = plan.scratchByName.get('y_ids');
  const rowSet = decodeRowSet(positions.map((position) => ({ position })));
  const rows = writeRowIndices(rowSet, SEQUENCE, false);
  assert.deepEqual(
    Array.from(new Int32Array(staged.bytes.buffer, staged.bytes.byteOffset, LANES)),
    Array.from(rows, (row) => inputs.y_ids[row]),
    'each lane stages the id at its own position');
  void state;
});
