import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

const EXPERTS = 4;
const D_IN = 2;
const D_OUT = 1;

async function buildForwardWasm(directory) {
  const output = join(directory, 'volvoxai.wasm');
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

/** Training kernels live in a separate translation unit and a separate build. */
async function buildFullWasm(directory) {
  const output = join(directory, 'volvoxai.full.wasm');
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output,
    'native/src/kernels/kernels.c', 'native/src/kernels/training_kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

/** Expert k maps every input feature to the constant k + 1. */
function bankPayload(slots) {
  const data = new Float32Array(slots.length * D_IN * D_OUT);
  for (let row = 0; row < slots.length; row++) {
    for (let feature = 0; feature < D_IN; feature++) {
      data[row * D_IN * D_OUT + feature * D_OUT] = slots[row] + 1;
    }
  }
  return data;
}

/**
 * Build a MoELinear graph whose staged expert weight holds only `slots`.
 * Route indices stay global, exactly as the bound graph produces them.
 */
function bankGraph(slots, routed, gates = [1], rows = 1) {
  const routedList = Array.isArray(routed) ? routed : [routed];
  assert.equal(routedList.length % rows, 0);
  assert.equal(gates.length, routedList.length);
  const topK = routedList.length / rows;
  const graph = new RuntimeGraph();
  const input = graph.addInput('x', [rows, D_IN]);
  input.buffer = Float32Array.from({ length: rows * D_IN }, (_, index) =>
    index % D_IN === 0 ? 1 : 0);
  const experts = graph.addWeight('experts', [slots.length, D_IN, D_OUT]);
  experts.buffer = bankPayload(slots);
  const indices = graph.addInput('route_indices', [rows, topK]);
  indices.buffer = Float32Array.from(routedList);
  const weights = graph.addInput('route_weights', [rows, topK]);
  weights.buffer = Float32Array.from(gates);
  const { out } = graph.addOp('MoELinear', {
    input,
    expert_weight: experts,
    route_indices: indices,
    route_weights: weights,
  }, { out: [rows, D_OUT] });
  graph.setOutputs([out.name]);
  // The bound graph attaches this; do the same by hand for the kernel test.
  if (slots.length !== EXPERTS) graph.nodes[0].residentSlots = Object.freeze([...slots]);
  return { graph, outputName: out.name };
}

function request(routedExpert) {
  return {
    x: Float32Array.of(1, 0),
    route_indices: Float32Array.of(routedExpert),
    route_weights: Float32Array.of(1),
  };
}

test('WASM MoELinear routes a partially resident expert bank by global slot id', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-moe-bank-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.equal(typeof wasm.api.moe_linear_f32, 'function');
    assert.equal(
      typeof wasm.api.vx_moe_linear_banked_f32, 'function',
      'forward WASM exports the banked MoELinear entry',
    );

    await t.test('a fully resident bank matches the portable CPU kernel', async () => {
      for (const expert of [0, 1, 2, 3]) {
        const slots = [0, 1, 2, 3];
        const wasmCase = bankGraph(slots, expert);
        const cpuCase = bankGraph(slots, expert);
        wasm.allocateGraph(wasmCase.graph);
        const actual = await wasm.execute(request(expert));
        const cpu = new CPUEngine();
        cpu.allocateGraph(cpuCase.graph);
        const expected = await cpu.execute(request(expert));
        assert.deepEqual(
          Array.from(actual[wasmCase.outputName]),
          Array.from(expected[cpuCase.outputName]),
        );
        assert.equal(actual[wasmCase.outputName][0], expert + 1);
      }
    });

    await t.test('only the resident slot is staged and routing stays global', async () => {
      const { graph, outputName } = bankGraph([2], 2);
      assert.deepEqual(graph.getTensor('experts').shape, [1, D_IN, D_OUT]);
      wasm.allocateGraph(graph);
      const result = await wasm.execute(request(2));
      assert.equal(result[outputName][0], 3);

      const cpuCase = bankGraph([2], 2);
      const cpu = new CPUEngine();
      cpu.allocateGraph(cpuCase.graph);
      const expected = await cpu.execute(request(2));
      assert.equal(result[outputName][0], expected[cpuCase.outputName][0]);
    });

    await t.test('two resident slots mix in staged order', async () => {
      const { graph, outputName } = bankGraph([1, 3], [1, 3], [0.25, 0.75]);
      wasm.allocateGraph(graph);
      const result = await wasm.execute({
        x: Float32Array.of(1, 0),
        route_indices: Float32Array.of(1, 3),
        route_weights: Float32Array.of(0.25, 0.75),
      });
      // 0.25 * 2 + 0.75 * 4
      assert.ok(Math.abs(result[outputName][0] - 3.5) < 1e-6);
    });

    await t.test('resident slot metadata survives repeated shape rebinds', async () => {
      const small = bankGraph([2], [2], [1], 1);
      const large = bankGraph([2], [2, 2, 2, 2], [1, 1, 1, 1], 4);
      const capacities = Object.freeze(Object.fromEntries(
        [...large.graph.tensors.entries()]
          .filter(([, tensor]) => tensor.isWeight !== true)
          .map(([name, tensor]) => [name, tensor.sizeBytes]),
      ));
      const smallInputs = request(2);
      const largeInputs = {
        x: Float32Array.of(1, 0, 1, 0, 1, 0, 1, 0),
        route_indices: Float32Array.of(2, 2, 2, 2),
        route_weights: Float32Array.of(1, 1, 1, 1),
      };

      wasm.compile(small.graph, wasm.prepareGraph(small.graph), {
        // Start with the small logical capacities but prove the larger shape
        // as the ceiling. The first rebind must therefore grow/replan the
        // activation suffix and move its resident-slot metadata table.
        tensorMaximumBytes: capacities,
        shapeSignature: 'rows=1',
      });
      assert.deepEqual(Array.from((await wasm.execute(smallInputs))[small.outputName]), [3]);
      const smallVariantBytes = wasm.inspectArena().variantBytes;
      const initialSlotTablePointer = wasm.nodeMetadata.get(
        small.graph.nodes[0],
      )?.slotTablePointer;
      assert.ok(Number.isSafeInteger(initialSlotTablePointer));

      wasm.rebindGraph(large.graph, wasm.prepareGraph(large.graph), {
        shapeSignature: 'rows=4',
      });
      assert.deepEqual(
        Array.from((await wasm.execute(largeInputs))[large.outputName]),
        [3, 3, 3, 3],
      );
      const grownSlotTablePointer = wasm.nodeMetadata.get(
        large.graph.nodes[0],
      )?.slotTablePointer;
      assert.ok(Number.isSafeInteger(grownSlotTablePointer));
      assert.notEqual(grownSlotTablePointer, initialSlotTablePointer,
        'activation growth must relocate and rebuild variant-owned metadata');
      assert.ok(wasm.inspectArena().activationGrowCount > 0);
      assert.ok(wasm.inspectArena().variantBytes >= smallVariantBytes,
        'resident slot table bytes belong to each committed variant arena');

      wasm.rebindGraph(small.graph, wasm.prepareGraph(small.graph), {
        shapeSignature: 'rows=1-again',
      });
      assert.deepEqual(Array.from((await wasm.execute(smallInputs))[small.outputName]), [3]);
    });

    await t.test('the training entries route a banked expert weight too', async () => {
      // The forward training kernel shares the resident-slot contract with
      // inference: `experts` counts staged rows, routes stay global.
      const full = await WasmEngine.init(await buildFullWasm(directory));
      assert.equal(typeof full.api.volvoxai_training_moe_linear_banked_f32, 'function');
      assert.equal(
        typeof full.api.volvoxai_training_moe_linear_backward_banked_f32, 'function');

      const slots = [1, 3];
      const domain = slots[slots.length - 1] + 1;
      const staged = bankPayload(slots);
      const rowsTable = new Uint32Array(domain).fill(0xffffffff);
      slots.forEach((slot, row) => { rowsTable[slot] = row; });

      const alloc = (bytes) => {
        const pointer = Number(full.api.alloc_bytes(bytes));
        assert.ok(Number.isSafeInteger(pointer) && pointer >= 0);
        return pointer;
      };
      full.api.reset_heap();
      const put = (typed) => {
        const pointer = alloc(typed.byteLength);
        new Uint8Array(full.mem.buffer, pointer, typed.byteLength).set(
          new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
        return pointer;
      };
      const pIn = put(Float32Array.of(1, 0));
      const pWeight = put(staged);
      const pIndices = put(Float32Array.of(3));
      const pGates = put(Float32Array.of(1));
      const pOut = alloc(4);
      const pTable = put(rowsTable);

      assert.equal(full.api.volvoxai_training_moe_linear_banked_f32(
        pIn, pWeight, 0, pIndices, pGates, pOut, 1, D_IN, D_OUT,
        slots.length, 1, pTable, domain), 1);
      // Global slot 3 is staged row 1, whose constant is 3 + 1.
      assert.equal(new Float32Array(full.mem.buffer, pOut, 1)[0], 4);

      // A non-resident global id must be refused, not remapped.
      const pAbsent = put(Float32Array.of(2));
      assert.equal(full.api.volvoxai_training_moe_linear_banked_f32(
        pIn, pWeight, 0, pAbsent, pGates, pOut, 1, D_IN, D_OUT,
        slots.length, 1, pTable, domain), 0);
    });

    await t.test('a non-resident route is refused instead of reading a neighbour', async () => {
      const { graph } = bankGraph([2], 0);
      wasm.allocateGraph(graph);
      await assert.rejects(
        () => wasm.execute(request(0)),
        /not resident in this context|rejected its canonical descriptor/,
      );
    });

    await t.test('an unreasonable sparse slot domain is rejected before allocation', () => {
      const firstUnrepresentableContiguousF32Slot = 2 ** 24;
      const { graph } = bankGraph(
        [firstUnrepresentableContiguousF32Slot],
        firstUnrepresentableContiguousF32Slot,
      );
      assert.throws(
        () => wasm.allocateGraph(graph),
        /unrepresentable resident slot domain/,
      );
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
