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
import { PagedKVCache } from '../ts/core/PagedKVCache.js';
import { kvPagePlanForLane, classifyPlan } from '../ts/backends/kvPageAddressing.js';

/*
 * Paged-KV equivalence gate: private paged KV is numerically identical to
 * contiguous KV.
 *
 * The comparison is only worth something if the paged run actually pages, so
 * the page tables here are forced onto the `gather` tier — logical pages
 * 0,1,2,3 of the decoding lane land on physical pages 0,2,4,6, which is what a
 * scheduler interleaving two requests produces. A run that silently fell back
 * to the identity mapping would pass a naive version of this test while
 * proving nothing, so the tier is asserted.
 */

const SEQUENCE = 8;
const WIDTH = 4;
const VOCABULARY = 6;

function floatWeight(graph, name, shape, seed, scale = 0.05) {
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  return graph.addWeight(name, shape, 'float32', {
    buffer: Float32Array.from({ length: elements }, (_, index) =>
      ((((index * 7) + seed * 5) % 17) - 8) * scale),
  });
}

function linear(graph, input, name, seed) {
  const weight = floatWeight(graph, `${name}.weight`, [WIDTH, WIDTH], seed);
  const bias = floatWeight(graph, `${name}.bias`, [WIDTH], seed + 11, 0.01);
  return graph.addOp('Linear', { input, weight, bias }, {
    out: { name, shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }, { weight_layout: 'OUT_IN' }).out;
}

/* Deliberately flat: every node between the projections and attention would be
 * another operator that has to learn page-table addressing, and the row path
 * refuses those rather than addressing them with linear arithmetic. */
function decoderGraph() {
  const graph = new RuntimeGraph();
  const ids = graph.addInput('ids', [1, SEQUENCE], 'int32');
  const keep = graph.addInput('keep', [1, SEQUENCE], 'int32');
  const embedding = floatWeight(graph, 'embedding.weight', [VOCABULARY, WIDTH], 1, 0.09);
  const embedded = graph.addOp('Embedding', { input: ids, weight: embedding }, {
    out: { name: 'embedded', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;
  const q = linear(graph, embedded, 'self.q', 2);
  const k = linear(graph, embedded, 'self.k', 3);
  const v = linear(graph, embedded, 'self.v', 4);
  const attended = graph.addOp('CrossSDPA', { q, k, v, mask: keep }, {
    out: { name: 'self.attention', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }, { heads: 1, causal: true, scale: 0.5 }).out;
  const logits = linear(graph, attended, 'logits', 5);
  graph.setOutputs([logits.name]);
  return graph;
}

function decoderInputs(ids, keep) {
  return { ids: Int32Array.from(ids), keep: Int32Array.from(keep) };
}

function tensorRow(graph, name, row) {
  const tensor = graph.tensors.get(name);
  return Array.from(tensor.buffer.subarray(row * WIDTH, (row + 1) * WIDTH));
}

/*
 * One page per token, two lanes, one physical page pool exactly the size of
 * the retained `[1,S,D]` activation. Appending alternately gives the decoding
 * lane pages 0,2,4,6: contiguous in logical order, scattered in physical.
 */
function interleavedCache() {
  return new PagedKVCache({
    lanes: 2,
    pageTokens: 1,
    laneTokenCapacity: SEQUENCE / 2,
    maxPages: SEQUENCE,
    policy: 'paged',
  });
}

function planFor(cache, lane) {
  return kvPagePlanForLane(cache, lane, ['self.k', 'self.v']);
}

async function runDecode(engine, graph, cache) {
  const ids = [1, 0, 0, 0, 0, 0, 0, 0];
  const keep = [1, 0, 0, 0, 0, 0, 0, 0];
  const names = ['ids', 'keep'];
  const rows = [];
  const tiers = [];

  const seed = await engine.execute(decoderInputs(ids, keep), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  rows.push(Array.from(seed.logits.subarray(0, WIDTH)));
  /* The seed is a full forward, so it writes the retained K/V linearly. The
   * cache has to agree: logical page zero of the decoding lane is physical
   * page zero, which is what claiming it first gives. */
  if (cache) cache.append(0, 1);

  for (const [position, id] of [[1, 3], [2, 5], [3, 2]]) {
    ids[position] = id;
    keep[position] = 1;
    if (cache) {
      /* A second lane claiming a page between the decoding lane's tokens is
       * what scatters the mapping. A single-lane cache has no interleaver and
       * stays on its identity map, which is the point of that variant. */
      if (cache.lanes > 1) cache.append(1, 1);
      cache.append(0, 1);
    }
    const options = {
      incremental: true, changedInputs: names, incrementalRowPosition: position,
    };
    if (cache) {
      const plan = planFor(cache, 0);
      tiers.push(classifyPlan(plan).tier);
      options.kvPages = plan;
    }
    const result = await engine.execute(decoderInputs(ids, keep), options);
    rows.push(Array.from(
      result.logits.subarray(position * WIDTH, (position + 1) * WIDTH)));
  }
  return { rows, tiers };
}

async function verifyPagedMatchesContiguous(makeEngine) {
  const contiguousGraph = decoderGraph();
  const contiguousEngine = await makeEngine(contiguousGraph);
  const contiguous = await runDecode(contiguousEngine, contiguousGraph, null);

  const pagedGraph = decoderGraph();
  const pagedEngine = await makeEngine(pagedGraph);
  const cache = interleavedCache();
  const paged = await runDecode(pagedEngine, pagedGraph, cache);

  assert.deepEqual(paged.tiers, ['gather', 'gather', 'gather'],
    'the paged run must actually page, not fall back to the identity mapping');
  assert.deepEqual(Array.from(cache.pageTable.subarray(0, 4)), [0, 2, 4, 6]);
  assert.deepEqual(Array.from(cache.kvLength), [4, 3]);

  assert.deepEqual(paged.rows, contiguous.rows,
    'paged KV must be bit-identical to contiguous KV at every step');

  /* Direct evidence for the write half: the projection wrote logical row 1 of
   * the decoding lane into physical slot 2. A gather that read the right bytes
   * from the wrong slot would still fail here. */
  for (const name of ['self.k', 'self.v']) {
    for (const [logical, physical] of [[0, 0], [1, 2], [2, 4], [3, 6]]) {
      assert.deepEqual(tensorRow(pagedGraph, name, physical),
        tensorRow(contiguousGraph, name, logical),
        `${name} logical row ${logical} must live in physical slot ${physical}`);
    }
  }
  return { contiguousGraph, pagedGraph, cache };
}

test('CPU paged KV row decode is identical to contiguous row decode', async () => {
  await verifyPagedMatchesContiguous(async (graph) => {
    const engine = new CPUEngine();
    engine.allocateGraph(graph);
    return engine;
  });
});

test('the identity mapping reproduces the contiguous slice with no gather', async () => {
  /* The claim that paged and contiguous are one code path rather than two.
   * With one lane whose pages are its own identity map, the resolver returns
   * the same view `attentionPrefix` built before paging existed. */
  const graph = decoderGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const cache = new PagedKVCache({
    lanes: 1, pageTokens: 2, laneTokenCapacity: SEQUENCE, policy: 'contiguous',
  });
  const { tiers, rows } = await runDecode(engine, graph, cache);
  assert.deepEqual(tiers, ['identity', 'identity', 'identity']);

  const reference = decoderGraph();
  const referenceEngine = new CPUEngine();
  referenceEngine.allocateGraph(reference);
  const unpaged = await runDecode(referenceEngine, reference, null);
  assert.deepEqual(rows, unpaged.rows);
});

test('a paged K/V tensor read outside attention is refused, not addressed linearly', async () => {
  /* Widening the paged set is how this fires in practice, and the failure has
   * to be a refusal: a Reshape reading a paged tensor with `position * width`
   * arithmetic produces a decoder that is wrong and looks right. */
  const graph = decoderGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const cache = interleavedCache();
  const names = ['ids', 'keep'];
  await engine.execute(decoderInputs([1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  cache.append(0, 1);
  cache.append(0, 1);
  await assert.rejects(
    engine.execute(decoderInputs([1, 3, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]), {
      incremental: true,
      changedInputs: names,
      incrementalRowPosition: 1,
      /* `embedded` feeds three projections and attention reads none of it on
       * k or v, so declaring it paged must be refused. */
      kvPages: kvPagePlanForLane(cache, 0, ['self.k', 'self.v', 'embedded']),
    }),
    /reads paged K\/V tensor 'embedded'/);
});

test('WASM paged KV row decode is identical to contiguous row decode', {
  timeout: 120_000,
}, async (t) => {
  const run = promisify(execFile);
  const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
  const clang = process.env.CLANG || 'clang';
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-paged-kv-'));
  try {
    const output = join(directory, 'volvoxai.wasm');
    await run(clang, [
      '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
      '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
      '-o', output, 'native/src/kernels/kernels.c',
    ], { cwd: repositoryRoot });
    await verifyPagedMatchesContiguous(async (graph) => {
      const engine = await WasmEngine.init(output);
      engine.compile(graph);
      return engine;
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
