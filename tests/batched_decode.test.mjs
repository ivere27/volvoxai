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
import { incrementalRowDomainSupported } from '../ts/backends/quantizedRowExecution.js';

/*
 * B>1 row-decode equivalence with independent per-lane active lengths.
 *
 * Two things are proved here and they fail differently.
 *
 * 1. Every step of a batched decode matches an *independent dense full
 *    recomputation* of the same tokens. This is the row-decode equivalence gate
 *    in ../docs/scheduling-and-dynamic-batching-design.md#stateful-sessions-prefill-decode-and-paged-kv,
 *    and it catches a lane reading the wrong row, prefix, or page.
 *
 * 2. Every step is bit-identical to decoding each lane on its own at B=1. This
 *    is the stronger claim and it is the one that says the padded batch is not
 *    an approximation. It holds because the keep mask carries each lane's active
 *    length and every kernel here *skips* a masked key rather than scoring it
 *    and multiplying by zero -- so a lane's visible key set and its accumulation
 *    order are exactly what the one-lane decode produces. If a future kernel
 *    starts scoring masked keys, this test is what notices.
 *
 * The lanes are deliberately staggered: lane 0 is two tokens ahead of lane 1
 * for the whole run, which is what a scheduler that admitted them one step apart
 * produces, and it is the case a padded-maximum implementation gets wrong only
 * on the short lane.
 */

const SEQUENCE = 8;
const WIDTH = 4;
const VOCABULARY = 6;
const LANES = 2;

function floatWeight(graph, name, shape, seed, scale = 0.05) {
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  return graph.addWeight(name, shape, 'float32', {
    buffer: Float32Array.from({ length: elements }, (_, index) =>
      ((((index * 7) + seed * 5) % 17) - 8) * scale),
  });
}

function linear(graph, input, name, seed, lanes) {
  const weight = floatWeight(graph, `${name}.weight`, [WIDTH, WIDTH], seed);
  const bias = floatWeight(graph, `${name}.bias`, [WIDTH], seed + 11, 0.01);
  return graph.addOp('Linear', { input, weight, bias }, {
    out: { name, shape: [lanes, SEQUENCE, WIDTH], dtype: 'float32' },
  }, { weight_layout: 'dout_din' }).out;
}

/* Deliberately flat, for the reason paged_kv_row_decode's graph is: every node
 * between the projections and attention is another operator that has to agree
 * about which row belongs to which lane. */
function decoderGraph(lanes) {
  const graph = new RuntimeGraph();
  const ids = graph.addInput('ids', [lanes, SEQUENCE], 'int32');
  const keep = graph.addInput('keep', [lanes, SEQUENCE], 'int32');
  const embedding = floatWeight(graph, 'embedding.weight', [VOCABULARY, WIDTH], 1, 0.09);
  const embedded = graph.addOp('Embedding', { input: ids, weight: embedding }, {
    out: { name: 'embedded', shape: [lanes, SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;
  const q = linear(graph, embedded, 'self.q', 2, lanes);
  const k = linear(graph, embedded, 'self.k', 3, lanes);
  const v = linear(graph, embedded, 'self.v', 4, lanes);
  const attended = graph.addOp('CrossSDPA', { q, k, v, mask: keep }, {
    out: { name: 'self.attention', shape: [lanes, SEQUENCE, WIDTH], dtype: 'float32' },
  }, { heads: 1, causal: true, scale: 0.5 }).out;
  const logits = linear(graph, attended, 'logits', 5, lanes);
  graph.setOutputs([logits.name]);
  return graph;
}

/*
 * The token stream. Lane 0 starts with two prompt tokens and lane 1 with one,
 * so their active lengths differ by one for the whole generation and neither
 * lane's length is ever the padded maximum for both.
 */
const PROMPTS = [[1, 3], [4]];
const GENERATED = [[5, 2, 4], [1, 3, 5]];

function laneTokens(lane, steps) {
  return [...PROMPTS[lane], ...GENERATED[lane].slice(0, steps)];
}

function denseInputs(lanes, steps) {
  const ids = new Int32Array(lanes * SEQUENCE);
  const keep = new Int32Array(lanes * SEQUENCE);
  for (let lane = 0; lane < lanes; lane++) {
    const tokens = laneTokens(lane, steps);
    for (let position = 0; position < tokens.length; position++) {
      ids[lane * SEQUENCE + position] = tokens[position];
      keep[lane * SEQUENCE + position] = 1;
    }
  }
  return { ids, keep };
}

function lanePosition(lane, step) {
  return PROMPTS[lane].length - 1 + step;
}

async function cpuEngine(graph) {
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine;
}

function logitRow(buffer, lane, position, lanes) {
  const base = (lane * SEQUENCE + position) * WIDTH;
  void lanes;
  return Array.from(buffer.subarray(base, base + WIDTH));
}

/**
 * The independent oracle: a dense full recomputation over the same tokens, on
 * a graph and engine that have never executed a row step.
 */
async function fullRecomputation(makeEngine, lanes, steps) {
  const graph = decoderGraph(lanes);
  const engine = await makeEngine(graph);
  const { ids, keep } = denseInputs(lanes, steps);
  const result = await engine.execute({ ids, keep });
  const rows = [];
  for (let lane = 0; lane < lanes; lane++) {
    rows.push(logitRow(result.logits, lane, lanePosition(lane, steps), lanes));
  }
  return rows;
}

/**
 * Run a batched decode, returning each step's per-lane logit rows.
 *
 * `dispatches` counts node executions so the test can assert the batch costs
 * one dispatch per node rather than one per lane, which is the throughput
 * condition a device row-decode path exists to satisfy.
 */
async function batchedDecode(makeEngine, lanes, cache, steps = GENERATED[0].length) {
  const graph = decoderGraph(lanes);
  const engine = await makeEngine(graph);
  const names = ['ids', 'keep'];
  const dispatches = [];
  const tiers = [];
  const rows = [];

  const runNode = engine._runNode?.bind(engine);
  let counter = 0;
  if (runNode) {
    engine._runNode = (node, execution) => { counter++; return runNode(node, execution); };
  }

  const seed = denseInputs(lanes, 0);
  await engine.execute(seed, {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  /* The seed is a full forward, so it wrote every lane's retained K/V linearly.
   * The cache has to agree: each lane's prompt occupies its own logical pages
   * from zero, which is what claiming them first gives. */
  if (cache) {
    for (let lane = 0; lane < lanes; lane++) cache.append(lane, PROMPTS[lane].length);
  }

  for (let step = 1; step <= steps; step++) {
    const inputs = denseInputs(lanes, step);
    if (cache) for (let lane = 0; lane < lanes; lane++) cache.append(lane, 1);
    const laneSteps = [];
    for (let lane = 0; lane < lanes; lane++) {
      const plan = cache
        ? kvPagePlanForLane(cache, lane, ['self.k', 'self.v'])
        : null;
      if (plan) tiers.push(classifyPlan(plan).tier);
      laneSteps.push({ position: lanePosition(lane, step), kvPages: plan });
    }
    counter = 0;
    const result = await engine.execute(inputs, {
      incremental: true, changedInputs: names, incrementalRowLanes: laneSteps,
    });
    dispatches.push(counter);
    const stepRows = [];
    for (let lane = 0; lane < lanes; lane++) {
      stepRows.push(logitRow(result.logits, lane, lanePosition(lane, step), lanes));
    }
    rows.push(stepRows);
  }
  return { rows, dispatches, tiers, graph };
}

/** Each lane decoded alone through the established B=1 row path. */
async function singleLaneDecodes(makeEngine, steps = GENERATED[0].length) {
  const perLane = [];
  for (let lane = 0; lane < LANES; lane++) {
    const graph = decoderGraph(1);
    const engine = await makeEngine(graph);
    const names = ['ids', 'keep'];
    const inputsFor = (count) => {
      const ids = new Int32Array(SEQUENCE);
      const keep = new Int32Array(SEQUENCE);
      const tokens = laneTokens(lane, count);
      for (let position = 0; position < tokens.length; position++) {
        ids[position] = tokens[position];
        keep[position] = 1;
      }
      return { ids, keep };
    };
    await engine.execute(inputsFor(0), {
      incremental: true, incrementalReset: true, changedInputs: names,
    });
    const rows = [];
    for (let step = 1; step <= steps; step++) {
      const position = lanePosition(lane, step);
      const result = await engine.execute(inputsFor(step), {
        incremental: true, changedInputs: names, incrementalRowPosition: position,
      });
      rows.push(logitRow(result.logits, 0, position, 1));
    }
    perLane.push(rows);
  }
  return perLane;
}

test('batched decode matches a dense full recomputation at every step', async () => {
  const batched = await batchedDecode(cpuEngine, LANES, null);
  for (let step = 1; step <= GENERATED[0].length; step++) {
    const oracle = await fullRecomputation(cpuEngine, LANES, step);
    assert.deepEqual(batched.rows[step - 1], oracle,
      `step ${step} must equal an independent dense recomputation`);
  }
});

test('batched decode is bit-identical to decoding each lane at B=1', async () => {
  /* The claim that the padded batch is not an approximation. Lane 1 is one
   * token shorter than lane 0 at every step, so its operand is padded and its
   * length reaches the kernel only through the keep mask. */
  const batched = await batchedDecode(cpuEngine, LANES, null);
  const single = await singleLaneDecodes(cpuEngine);
  for (let step = 0; step < batched.rows.length; step++) {
    for (let lane = 0; lane < LANES; lane++) {
      assert.deepEqual(batched.rows[step][lane], single[lane][step],
        `lane ${lane} step ${step + 1} must be bit-identical to its own B=1 decode`);
    }
  }
});

test('a batched step costs one dispatch per node, not one per lane', async () => {
  /* The physical-batching condition. The single-lane run establishes the
   * per-node count, and the two-lane run has to match it: sharing the context,
   * weights and page pool while still dispatching per lane would put the
   * per-request overhead back on top of the batched backend. */
  const batched = await batchedDecode(cpuEngine, LANES, null);
  const single = await batchedDecode(cpuEngine, 1, null);
  assert.deepEqual(batched.dispatches, single.dispatches,
    'a two-lane step must execute the same number of nodes as a one-lane step');
  assert.ok(batched.dispatches[0] > 0, 'the step must execute nodes at all');
});

/*
 * The paged batched case, over a shared prompt.
 *
 * Sharing is not decoration here, it is what makes the run self-consistent. A
 * seed pass writes every lane's K/V at `lane * S + position`, so a lane whose
 * page table sends logical token `j` anywhere else would read a slot the seed
 * never wrote -- and with a lowest-page-first free list, only lane zero's pages
 * can coincide with its seed rows. Publishing lane zero's prompt and letting
 * lane one acquire it puts both lanes' logical prefix on the pages the seed
 * actually filled, which is exactly the serving case paging exists for.
 *
 * Lane one's *own* seed rows deliberately hold different tokens, so a run that
 * read the identity rows instead of the shared pages produces a different
 * answer rather than accidentally the right one.
 */
const SHARED_PROMPT = [1, 3];
const DECOY_PROMPT = [2, 5];
const SHARED_GENERATED = [[5, 2, 4], [1, 3, 5]];

function sharedInputs(steps, { decoy }) {
  const ids = new Int32Array(LANES * SEQUENCE);
  const keep = new Int32Array(LANES * SEQUENCE);
  for (let lane = 0; lane < LANES; lane++) {
    const prompt = decoy && lane === 1 ? DECOY_PROMPT : SHARED_PROMPT;
    const tokens = [...prompt, ...SHARED_GENERATED[lane].slice(0, steps)];
    for (let position = 0; position < tokens.length; position++) {
      ids[lane * SEQUENCE + position] = tokens[position];
      keep[lane * SEQUENCE + position] = 1;
    }
  }
  return { ids, keep };
}

test('batched decode over scattered shared pages matches a dense recomputation', async () => {
  /* Batched row decode combined with immutable shared paged KV: the lanes are
   * batched *and* their pages are interleaved, which is what a scheduler serving
   * two requests out of one page pool produces. The tier is asserted because a
   * run that silently fell back to the identity mapping would pass while proving
   * nothing. */
  const cache = new PagedKVCache({
    lanes: LANES,
    pageTokens: 1,
    laneTokenCapacity: SEQUENCE,
    maxPages: LANES * SEQUENCE,
    policy: 'paged',
  });
  const graph = decoderGraph(LANES);
  const engine = await cpuEngine(graph);
  const names = ['ids', 'keep'];
  /* The seed carries the decoy for lane one, so its identity rows hold tokens
   * the shared prefix does not. */
  await engine.execute(sharedInputs(0, { decoy: true }), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  cache.append(0, SHARED_PROMPT.length);
  cache.publishPrefix('shared', 0, SHARED_PROMPT.length);
  assert.equal(cache.acquirePrefix('shared', 1), SHARED_PROMPT.length);
  assert.deepEqual(
    Array.from(cache.pageTable.subarray(cache.pagesPerLane, cache.pagesPerLane + 2)),
    [0, 1], "lane one's logical prefix must sit on the published pages");

  const steps = SHARED_GENERATED[0].length;
  const rows = [];
  const tiers = [];
  for (let step = 1; step <= steps; step++) {
    for (let lane = 0; lane < LANES; lane++) cache.append(lane, 1);
    const laneSteps = [];
    const stepTiers = [];
    for (let lane = 0; lane < LANES; lane++) {
      const plan = kvPagePlanForLane(cache, lane, ['self.k', 'self.v']);
      stepTiers.push(classifyPlan(plan).tier);
      laneSteps.push({ position: SHARED_PROMPT.length - 1 + step, kvPages: plan });
    }
    tiers.push(stepTiers);
    const result = await engine.execute(sharedInputs(step, { decoy: false }), {
      incremental: true, changedInputs: names, incrementalRowLanes: laneSteps,
    });
    const stepRows = [];
    for (let lane = 0; lane < LANES; lane++) {
      stepRows.push(logitRow(result.logits, lane, SHARED_PROMPT.length - 1 + step, LANES));
    }
    rows.push(stepRows);
  }
  /* Lane one holds pages [0,1,...] against an identity base of `pagesPerLane`,
   * so it is scattered from its first step. Lane zero starts on its own identity
   * run and leaves it as soon as lane one takes a page between its tokens --
   * which is precisely the interleaving this test exists to exercise. A run that
   * silently stayed on the identity mapping would pass a naive version of this
   * test while proving nothing, so both facts are asserted rather than the
   * weaker "some lane paged". */
  assert.deepEqual(tiers.map((step) => step[1]), tiers.map(() => 'gather'),
    `lane one must page at every step, saw ${JSON.stringify(tiers)}`);
  assert.deepEqual(tiers.slice(1).map((step) => step[0]), tiers.slice(1).map(() => 'gather'),
    `lane zero must page once the pool is interleaved, saw ${JSON.stringify(tiers)}`);
  assert.equal(tiers[0][0], 'identity',
    "lane zero's first step is still its own identity run");

  /* The oracle recomputes the *semantic* stream: lane one's prompt is the
   * shared one, which is what its page table says it is reading. */
  for (let step = 1; step <= steps; step++) {
    const oracleGraph = decoderGraph(LANES);
    const oracleEngine = await cpuEngine(oracleGraph);
    const dense = await oracleEngine.execute(sharedInputs(step, { decoy: false }));
    const expected = [];
    for (let lane = 0; lane < LANES; lane++) {
      expected.push(logitRow(dense.logits, lane, SHARED_PROMPT.length - 1 + step, LANES));
    }
    assert.deepEqual(rows[step - 1], expected,
      `paged batched step ${step} must equal an independent dense recomputation`);

    /* Direct evidence for the write half: each lane's generated K/V row went to
     * the physical slot its page table names, not to `lane * S + position`. */
    const position = SHARED_PROMPT.length - 1 + step;
    for (const name of ['self.k', 'self.v']) {
      const actual = graph.tensors.get(name);
      const reference = oracleGraph.tensors.get(name);
      for (let lane = 0; lane < LANES; lane++) {
        const slot = cache.pageTable[lane * cache.pagesPerLane + position];
        assert.deepEqual(
          Array.from(actual.buffer.subarray(slot * WIDTH, (slot + 1) * WIDTH)),
          Array.from(reference.buffer.subarray(
            (lane * SEQUENCE + position) * WIDTH, (lane * SEQUENCE + position + 1) * WIDTH)),
          `${name} lane ${lane} position ${position} must live in physical slot ${slot}`);
      }
    }
  }
});

test('a step whose activations carry one lane is refused, not decoded as one lane', async () => {
  /* The failure this guards is silent: a two-lane scheduler handing a one-lane
   * graph to the row path would decode lane zero and return lane one's slot
   * untouched, and every shape downstream would still agree. */
  const graph = decoderGraph(1);
  const engine = await cpuEngine(graph);
  const names = ['ids', 'keep'];
  const single = denseInputs(1, 0);
  await engine.execute(single, {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  await assert.rejects(
    engine.execute(denseInputs(1, 1), {
      incremental: true,
      changedInputs: names,
      incrementalRowLanes: [{ position: 1 }, { position: 1 }],
    }),
    /has no fixed-sequence output for a 2-lane step/);
});

test('a lane whose page plan publishes the wrong length is refused', async () => {
  /* The row path produced the prefix; a plan that claims a different length is
   * asking attention to read a token nobody wrote. Checked once in the row set
   * so every backend inherits it. */
  const cache = new PagedKVCache({
    lanes: LANES, pageTokens: 1, laneTokenCapacity: SEQUENCE,
    maxPages: LANES * SEQUENCE, policy: 'paged',
  });
  cache.append(0, 2);
  cache.append(1, 1);
  const graph = decoderGraph(LANES);
  const engine = await cpuEngine(graph);
  const names = ['ids', 'keep'];
  await engine.execute(denseInputs(LANES, 0), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  await assert.rejects(
    engine.execute(denseInputs(LANES, 1), {
      incremental: true,
      changedInputs: names,
      incrementalRowLanes: [
        { position: 2, kvPages: kvPagePlanForLane(cache, 0, ['self.k', 'self.v']) },
        { position: 1, kvPages: kvPagePlanForLane(cache, 1, ['self.k', 'self.v']) },
      ],
    }),
    /page plan publishes K\/V length 2 at position 2/);
});

test('the row attestation admits a declared lane count and refuses an undeclared one', async () => {
  /* Batch is what the context declares, never what a shape suggests. The same
   * graph is attested for two lanes and refused for one, and nothing about the
   * graph changed between the two calls. */
  const graph = decoderGraph(LANES);
  const logical = {
    inputs: Object.fromEntries(['ids', 'keep'].map((name) => [name, {
      shape: [LANES, SEQUENCE], dtype: 'int32',
    }])),
    tensors: {},
    nodes: [],
  };
  for (const [name, tensor] of graph.tensors.entries()) {
    logical.tensors[name] = { shape: tensor.shape, dtype: tensor.dtype };
  }
  for (const node of graph.nodes) {
    logical.nodes.push({
      id: node.id,
      opType: node.opType,
      params: node.params,
      inputs: Object.fromEntries(
        Object.entries(node.inputs).map(([port, tensor]) => [port, tensor.name])),
      outputs: Object.fromEntries(
        Object.entries(node.outputs).map(([port, tensor]) => [port, { tensor: tensor.name }])),
    });
  }
  assert.equal(
    incrementalRowDomainSupported(logical, ['ids', 'keep'], { lanes: LANES }), true,
    'a two-lane graph must attest when two lanes are declared');
  assert.equal(
    incrementalRowDomainSupported(logical, ['ids', 'keep'], { lanes: 1 }), false,
    'the same graph must not attest as one lane');
  assert.equal(
    incrementalRowDomainSupported(logical, ['ids', 'keep'],
      { lanes: LANES, sequenceMajorRows: true }), false,
    'sequence-major rows cannot carry a second lane');
});

/*
 * A finished lane cannot be dropped from a dense batch.
 *
 * The activations are `[B,S,D]`, so removing a lane changes every operand's
 * shape and forces a reseed -- which is exactly the head-of-line stall
 * continuous batching exists to remove. The lane idles instead: its active
 * length stops moving and it repeats its own last row, which is byte-for-byte
 * what is already there because the recompute reads the same unchanged inputs
 * and writes the same bytes to the same slot.
 *
 * That identity is the load-bearing claim, so it is asserted rather than
 * argued: after the step, every retained tensor row belonging to the idle lane
 * must be bit-identical to what it was before, and the advancing lane must
 * still match its own uninterrupted B=1 decode.
 */
async function laneSnapshot(graph, lane, position) {
  const rows = {};
  for (const name of ['embedded', 'self.q', 'self.k', 'self.v', 'self.attention', 'logits']) {
    const tensor = graph.tensors.get(name);
    const base = (lane * SEQUENCE + position) * WIDTH;
    rows[name] = Array.from(tensor.buffer.subarray(base, base + WIDTH));
  }
  return rows;
}

test('an idle lane keeps a dense batch running and changes nothing', async () => {
  const graph = decoderGraph(LANES);
  const engine = await cpuEngine(graph);
  const names = ['ids', 'keep'];
  await engine.execute(denseInputs(LANES, 0), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });

  /* Both lanes advance for one step, then lane 1 finishes and idles while
   * lane 0 keeps generating. */
  await engine.execute(denseInputs(LANES, 1), {
    incremental: true,
    changedInputs: names,
    incrementalRowLanes: [
      { position: lanePosition(0, 1) }, { position: lanePosition(1, 1) },
    ],
  });
  const idlePosition = lanePosition(1, 1);
  const before = await laneSnapshot(graph, 1, idlePosition);

  const inputs = denseInputs(LANES, 2);
  /* Lane 1's tokens beyond its last are not written: it produced none. */
  inputs.ids[SEQUENCE + lanePosition(1, 2)] = 0;
  inputs.keep[SEQUENCE + lanePosition(1, 2)] = 0;
  const result = await engine.execute(inputs, {
    incremental: true,
    changedInputs: names,
    /* The idle lane repeats its own last row. The engine takes concrete rows,
     * so the lifecycle is where `null` becomes "your last row". */
    incrementalRowLanes: [
      { position: lanePosition(0, 2) }, { position: idlePosition },
    ],
  });

  const after = await laneSnapshot(graph, 1, idlePosition);
  assert.deepEqual(after, before,
    "an idle lane's retained rows must be bit-identical after the step");

  /* And the advancing lane is unaffected by sharing the step with an idle one. */
  const single = await singleLaneDecodes(cpuEngine);
  assert.deepEqual(
    logitRow(result.logits, 0, lanePosition(0, 2), LANES), single[0][1],
    'the advancing lane must match its own uninterrupted B=1 decode');
});

test('minimum and maximum lane positions both decode', async () => {
  /* The ends of the admitted range, which a padded-maximum implementation gets
   * wrong at exactly one of them: position 1 is the first row a step may write,
   * and capacity - 1 is the last. */
  const graph = decoderGraph(LANES);
  const engine = await cpuEngine(graph);
  const names = ['ids', 'keep'];
  const ids = new Int32Array(LANES * SEQUENCE);
  const keep = new Int32Array(LANES * SEQUENCE);
  /* Lane 0 fills its whole capacity; lane 1 holds a single prompt token, so the
   * two lanes sit at opposite ends of the range in the same step. */
  for (let position = 0; position < SEQUENCE - 1; position++) {
    ids[position] = (position % (VOCABULARY - 1)) + 1;
    keep[position] = 1;
  }
  ids[SEQUENCE] = 2;
  keep[SEQUENCE] = 1;
  await engine.execute({ ids, keep }, {
    incremental: true, incrementalReset: true, changedInputs: names,
  });

  const stepped = { ids: Int32Array.from(ids), keep: Int32Array.from(keep) };
  stepped.ids[SEQUENCE - 1] = 4;
  stepped.keep[SEQUENCE - 1] = 1;
  stepped.ids[SEQUENCE + 1] = 3;
  stepped.keep[SEQUENCE + 1] = 1;
  const result = await engine.execute(stepped, {
    incremental: true,
    changedInputs: names,
    incrementalRowLanes: [{ position: SEQUENCE - 1 }, { position: 1 }],
  });

  const oracleGraph = decoderGraph(LANES);
  const oracleEngine = await cpuEngine(oracleGraph);
  const dense = await oracleEngine.execute(stepped);
  for (const [lane, position] of [[0, SEQUENCE - 1], [1, 1]]) {
    assert.deepEqual(
      logitRow(result.logits, lane, position, LANES),
      logitRow(dense.logits, lane, position, LANES),
      `lane ${lane} at position ${position} must equal a dense recomputation`);
  }
});

test('a step past capacity is refused for the offending lane alone', async () => {
  const graph = decoderGraph(LANES);
  const engine = await cpuEngine(graph);
  const names = ['ids', 'keep'];
  await engine.execute(denseInputs(LANES, 0), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  await assert.rejects(
    engine.execute(denseInputs(LANES, 1), {
      incremental: true,
      changedInputs: names,
      /* Lane 1's position is outside the fixed sequence. The refusal has to name
       * the lane: "position out of range" for a batch says nothing about which
       * request the caller mis-stepped. */
      incrementalRowLanes: [{ position: 2 }, { position: SEQUENCE }],
    }),
    /lane 1 position 8 is outside a supported fixed-sequence output/);
});

test('two batched contexts with different lane counts stay isolated', async () => {
  /* Nothing is shared between contexts, but the staging buffers are keyed by
   * node id, so a scratch allocator accidentally hoisted to module scope would
   * make two contexts alias each other's staged rows -- and only when their
   * lane counts differed would the sizes disagree loudly enough to notice. */
  const twoLaneGraph = decoderGraph(LANES);
  const twoLaneEngine = await cpuEngine(twoLaneGraph);
  const oneLaneGraph = decoderGraph(1);
  const oneLaneEngine = await cpuEngine(oneLaneGraph);
  const names = ['ids', 'keep'];

  await twoLaneEngine.execute(denseInputs(LANES, 0), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  await oneLaneEngine.execute(denseInputs(1, 0), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });

  const interleaved = [];
  for (let step = 1; step <= GENERATED[0].length; step++) {
    const two = await twoLaneEngine.execute(denseInputs(LANES, step), {
      incremental: true,
      changedInputs: names,
      incrementalRowLanes: [
        { position: lanePosition(0, step) }, { position: lanePosition(1, step) },
      ],
    });
    const one = await oneLaneEngine.execute(denseInputs(1, step), {
      incremental: true, changedInputs: names, incrementalRowPosition: lanePosition(0, step),
    });
    interleaved.push([
      logitRow(two.logits, 0, lanePosition(0, step), LANES),
      logitRow(one.logits, 0, lanePosition(0, step), 1),
    ]);
  }
  for (const [batched, single] of interleaved) {
    assert.deepEqual(batched, single,
      'interleaving a two-lane and a one-lane context must not disturb either');
  }
});

test('WASM batched decode matches the CPU reference', { timeout: 180_000 }, async (t) => {
  const run = promisify(execFile);
  const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
  const clang = process.env.CLANG || 'clang';
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-batched-decode-'));
  try {
    const output = join(directory, 'volvoxai.wasm');
    await run(clang, [
      '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
      '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
      '-o', output, 'native/src/kernels/kernels.c',
    ], { cwd: repositoryRoot });
    const makeEngine = async (graph) => {
      const engine = await WasmEngine.init(output);
      engine.compile(graph);
      return engine;
    };
    const wasm = await batchedDecode(makeEngine, LANES, null);
    const single = await singleLaneDecodes(makeEngine);
    for (let step = 0; step < wasm.rows.length; step++) {
      for (let lane = 0; lane < LANES; lane++) {
        assert.deepEqual(wasm.rows[step][lane], single[lane][step],
          `WASM lane ${lane} step ${step + 1} must be bit-identical to its own B=1 decode`);
      }
    }
    for (let step = 1; step <= GENERATED[0].length; step++) {
      const oracle = await fullRecomputation(makeEngine, LANES, step);
      assert.deepEqual(wasm.rows[step - 1], oracle,
        `WASM step ${step} must equal an independent dense recomputation`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

/*
 * The quantized routes, batched.
 *
 * QEmbedding, QLayerNorm and QLinear are the WASM decode routes that passed a
 * literal row count of one to their kernels. With more than one lane that
 * computed lane zero and left every other lane's staged output row
 * uninitialised, which the scatter then wrote into the retained image -- silently
 * wrong, and invisible to the FP32 graph above because those routes take their
 * extents from the row node's own shapes. This graph is deliberately just those
 * three operators.
 */
function quantized() {
  return { scheme: 'per_tensor', scale: 0.125, zero_point: 0 };
}

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

function quantizedDecoderGraph(lanes) {
  const graph = new RuntimeGraph();
  const ids = graph.addInput('ids', [lanes, SEQUENCE], 'int32');
  const embedding = byteWeight(graph, 'embedding.weight', VOCABULARY, WIDTH, 1);
  const embedded = graph.addOp('QEmbedding', { input: ids, weight: embedding }, {
    out: {
      name: 'embed', shape: [lanes, SEQUENCE, WIDTH], dtype: 'int8', quantization: quantized(),
    },
  }).out;
  const gamma = graph.addWeight('norm.weight', [WIDTH], 'float32', {
    buffer: Float32Array.of(1, 0.75, 1.25, 0.5),
  });
  const beta = graph.addWeight('norm.bias', [WIDTH], 'float32', {
    buffer: Float32Array.of(0.125, -0.125, 0.25, 0),
  });
  const normalized = graph.addOp(
    'QLayerNorm', { input: embedded, weight: gamma, bias: beta }, {
      out: {
        name: 'normalized', shape: [lanes, SEQUENCE, WIDTH], dtype: 'int8',
        quantization: quantized(),
      },
    }, { eps: 1e-5, d_model: WIDTH }).out;
  const weight = byteWeight(graph, 'logits.weight', WIDTH, WIDTH, 2);
  const bias = graph.addWeight('logits.bias', [WIDTH], 'int32', {
    buffer: new Int32Array(WIDTH),
  });
  const logits = graph.addOp('QLinear', { input: normalized, weight, bias }, {
    out: {
      name: 'logits', shape: [lanes, SEQUENCE, WIDTH], dtype: 'int8', quantization: quantized(),
    },
  }).out;
  graph.setOutputs([logits.name]);
  return graph;
}

function quantizedIds(lanes, steps) {
  const ids = new Int32Array(lanes * SEQUENCE);
  for (let lane = 0; lane < lanes; lane++) {
    const tokens = laneTokens(lane, steps);
    for (let position = 0; position < tokens.length; position++) {
      ids[lane * SEQUENCE + position] = tokens[position];
    }
  }
  return { ids };
}

async function quantizedBatchedDecode(makeEngine, lanes) {
  const graph = quantizedDecoderGraph(lanes);
  const engine = await makeEngine(graph);
  const names = ['ids'];
  await engine.execute(quantizedIds(lanes, 0), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  const rows = [];
  for (let step = 1; step <= GENERATED[0].length; step++) {
    const laneSteps = [];
    for (let lane = 0; lane < lanes; lane++) {
      laneSteps.push({ position: lanePosition(lane, step) });
    }
    const options = { incremental: true, changedInputs: names };
    if (lanes === 1) options.incrementalRowPosition = laneSteps[0].position;
    else options.incrementalRowLanes = laneSteps;
    const result = await engine.execute(quantizedIds(lanes, step), options);
    const stepRows = [];
    for (let lane = 0; lane < lanes; lane++) {
      const position = lanePosition(lane, step);
      const base = (lane * SEQUENCE + position) * WIDTH;
      stepRows.push(Array.from(result.logits.subarray(base, base + WIDTH)));
    }
    rows.push(stepRows);
  }
  return { rows, graph };
}

async function quantizedOracle(makeEngine, lanes, steps) {
  const graph = quantizedDecoderGraph(lanes);
  const engine = await makeEngine(graph);
  const result = await engine.execute(quantizedIds(lanes, steps));
  const rows = [];
  for (let lane = 0; lane < lanes; lane++) {
    const position = lanePosition(lane, steps);
    const base = (lane * SEQUENCE + position) * WIDTH;
    rows.push(Array.from(result.logits.subarray(base, base + WIDTH)));
  }
  return rows;
}

test('batched quantized decode matches a dense recomputation', async () => {
  const batched = await quantizedBatchedDecode(cpuEngine, LANES);
  for (let step = 1; step <= GENERATED[0].length; step++) {
    assert.deepEqual(batched.rows[step - 1], await quantizedOracle(cpuEngine, LANES, step),
      `quantized step ${step} must equal an independent dense recomputation`);
  }
});

test('WASM batched quantized decode honours the lane count', { timeout: 180_000 }, async (t) => {
  /* The regression test for the literal row count of one. Before the fix, lane
   * 1's logits row came back as whatever the staging buffer last held. */
  const run = promisify(execFile);
  const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
  const clang = process.env.CLANG || 'clang';
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-batched-quantized-'));
  try {
    const output = join(directory, 'volvoxai.wasm');
    await run(clang, [
      '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
      '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
      '-o', output, 'native/src/kernels/kernels.c',
    ], { cwd: repositoryRoot });
    const makeEngine = async (graph) => {
      const engine = await WasmEngine.init(output);
      engine.compile(graph);
      /* No JS fallback: the point is that the compiled kernels walk `lanes`
       * rows, so a route quietly handled in JavaScript would prove nothing. */
      for (const name of ['_cpuQEmbedding', '_cpuQLayerNorm', '_cpuQLinear']) {
        engine[name] = () => { throw new Error(`WASM batched decode called JS fallback ${name}`); };
      }
      return engine;
    };
    const wasm = await quantizedBatchedDecode(makeEngine, LANES);
    for (let step = 1; step <= GENERATED[0].length; step++) {
      assert.deepEqual(wasm.rows[step - 1], await quantizedOracle(cpuEngine, LANES, step),
        `WASM quantized step ${step} must equal the CPU dense recomputation`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('a parked lane leaves the retained image untouched on CPU and WASM', async (t) => {
  /* The write half of parking, on both host backends. A parked lane occupies a
   * row so the operands keep their shape and its output is discarded -- so
   * every byte outside the live lane must be exactly what it was, including the
   * row the parked lane nominally sat on. */
  const verify = async (makeEngine) => {
    const graph = decoderGraph(LANES);
    const engine = await makeEngine(graph);
    const names = ['ids', 'keep'];
    await engine.execute(denseInputs(LANES, 0), {
      incremental: true, incrementalReset: true, changedInputs: names,
    });
    /* One round with both lanes live, so lane 1 holds real bytes to protect. */
    await engine.execute(denseInputs(LANES, 1), {
      incremental: true,
      changedInputs: names,
      incrementalRowLanes: [
        { position: lanePosition(0, 1) }, { position: lanePosition(1, 1) },
      ],
    });
    const snapshot = new Map();
    for (const name of ['embedded', 'self.q', 'self.k', 'self.v', 'self.attention', 'logits']) {
      snapshot.set(name, Array.from(graph.tensors.get(name).buffer));
    }

    const result = await engine.execute(denseInputs(LANES, 2), {
      incremental: true,
      changedInputs: names,
      incrementalRowLanes: [{ position: lanePosition(0, 2) }, { parked: true }],
    });

    for (const [name, before] of snapshot) {
      const after = Array.from(graph.tensors.get(name).buffer);
      /* Lane 1's whole half of every tensor is untouched: the parked lane wrote
       * nothing, not even the row it occupied. */
      const laneStart = SEQUENCE * WIDTH;
      assert.deepEqual(after.slice(laneStart), before.slice(laneStart),
        `${name}: a parked lane must not write any byte of its own lane`);
    }
    return logitRow(result.logits, 0, lanePosition(0, 2), LANES);
  };

  /* Each backend is compared against *itself*: the live lane's row must equal
   * what that same backend produces decoding lane zero alone. Comparing WASM's
   * result to the CPU's would instead measure the two kernels' accumulation
   * orders against each other, which differ in the last bit for reasons that
   * have nothing to do with parking. */
  const cpu = await verify(cpuEngine);
  const cpuSingle = await singleLaneDecodes(cpuEngine);
  assert.deepEqual(cpu, cpuSingle[0][1],
    'the live lane must be unaffected by sharing the step with a parked one');

  const run = promisify(execFile);
  const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
  const clang = process.env.CLANG || 'clang';
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-parked-'));
  try {
    const output = join(directory, 'volvoxai.wasm');
    await run(clang, [
      '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
      '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
      '-o', output, 'native/src/kernels/kernels.c',
    ], { cwd: repositoryRoot });
    const makeEngine = async (graph) => {
      const engine = await WasmEngine.init(output);
      engine.compile(graph);
      return engine;
    };
    const wasm = await verify(makeEngine);
    const wasmSingle = await singleLaneDecodes(makeEngine);
    assert.deepEqual(wasm, wasmSingle[0][1],
      'WASM must park a lane without disturbing the live lane either');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
