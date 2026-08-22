// Paged KV decode parity on a physical WebGPU adapter.
//
//   deno run --unstable-webgpu --unstable-sloppy-imports --allow-all \
//     tests/parity/kvcache/paged_kv_webgpu.mjs
//
// A page table only means something if the pages are actually scattered, so
// this drives a deliberately interleaved mapping — logical pages 0,1,2,3 of the
// decoding lane land on physical pages 0,2,4,6, which is what a scheduler
// running two requests concurrently produces — and compares every step against
// a contiguous CPU run of the same graph.
//
// Two things are being proved, and they are different:
//
//   1. The retained K/V ends up in the *mapped* physical slots, not at
//      `position * width`. That is the write half, and a gather that read the
//      right bytes from the wrong slot would still fail it.
//   2. The attention output matches the contiguous CPU oracle exactly. Both
//      sides are int8, so "exactly" means equal, not close.
//
// Build the harness bundle first (`make parity_paged_kv_bundle`, on a host with
// the npm dev dependencies): ShaderLibrary imports .wgsl, which only esbuild's
// text loader resolves, so the campaign runs the same bundled code the release
// bundle contains.
import {
  classifyPlan,
  CPUEngine,
  kvPagePlanForLane,
  PagedKVCache,
  RuntimeGraph,
  WebGPUEngine,
} from './out/paged_kv_internals.js';

const SEQUENCE = 8;
const WIDTH = 8;
const VOCABULARY = 6;
const HEADS = 2;
const PAGED_TENSORS = ['self.k', 'self.v'];
const STEPS = [[1, 3], [2, 5], [3, 2]];

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

function qlinear(graph, input, name, seed, lanes = 1) {
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

/* Flat on purpose: every node between a projection and attention would be
 * another operator that has to learn page-table addressing, and the row path
 * refuses those rather than addressing them with linear arithmetic. */
function decoderGraph(lanes = 1) {
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
  graph.setOutputs([logits.name, 'self.attention', 'self.k', 'self.v', 'embed', 'self.q']);
  return graph;
}

const decoderInputs = (ids, keep) => ({
  y_ids: Int32Array.from(ids), y_keep: Int32Array.from(keep),
});

/* Two lanes claiming pages alternately is what scatters the mapping. One page
 * per token keeps the arithmetic obvious; a real deployment uses far larger
 * pages, and the addressing is identical either way. */
function interleavedCache() {
  return new PagedKVCache({
    lanes: 2,
    pageTokens: 1,
    laneTokenCapacity: SEQUENCE / 2,
    maxPages: SEQUENCE,
    policy: 'paged',
  });
}

async function readTensor(engine, name) {
  const entry = (await engine.snapshotOutputs()).get(name);
  if (!entry) throw new Error(`WebGPU output snapshot has no '${name}'.`);
  return Array.from(
    await engine.readBuffer(entry.deviceBuffer, entry.sizeBytes, entry.dtype));
}

async function runCpu() {
  const graph = decoderGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const ids = [1, 0, 0, 0, 0, 0, 0, 0];
  const keep = [1, 0, 0, 0, 0, 0, 0, 0];
  const rows = [];
  await engine.execute(decoderInputs(ids, keep), {
    incremental: true, incrementalReset: true, changedInputs: ['y_ids', 'y_keep'],
  });
  for (const [position, id] of STEPS) {
    ids[position] = id;
    keep[position] = 1;
    await engine.execute(decoderInputs(ids, keep), {
      incremental: true,
      changedInputs: ['y_ids', 'y_keep'],
      incrementalRowPosition: position,
    });
    rows.push({
      position,
      attention: Array.from(graph.tensors.get('self.attention').buffer
        .subarray(position * WIDTH, (position + 1) * WIDTH)),
      logits: Array.from(graph.tensors.get('logits').buffer
        .subarray(position * WIDTH, (position + 1) * WIDTH)),
      key: Array.from(graph.tensors.get('self.k').buffer
        .subarray(position * WIDTH, (position + 1) * WIDTH)),
      value: Array.from(graph.tensors.get('self.v').buffer
        .subarray(position * WIDTH, (position + 1) * WIDTH)),
    });
  }
  return rows;
}

async function runWebGPU(paged) {
  const engine = await WebGPUEngine.init();
  if (!engine) throw new Error('No WebGPU adapter.');
  const graph = decoderGraph();
  await engine.allocateGraph(graph);
  const cache = paged ? interleavedCache() : null;
  const ids = [1, 0, 0, 0, 0, 0, 0, 0];
  const keep = [1, 0, 0, 0, 0, 0, 0, 0];
  const tiers = [];

  await engine.execute(decoderInputs(ids, keep), {
    incremental: true, incrementalReset: true, changedInputs: ['y_ids', 'y_keep'],
  });
  /* The seed is a full forward, so it writes the retained K/V linearly. The
   * cache has to agree: logical page zero of the decoding lane is physical page
   * zero, which is what claiming it first gives. */
  if (cache) cache.append(0, 1);

  const slotOf = [];
  for (const [position, id] of STEPS) {
    ids[position] = id;
    keep[position] = 1;
    const options = {
      incremental: true,
      changedInputs: ['y_ids', 'y_keep'],
      incrementalRowPosition: position,
    };
    if (cache) {
      cache.append(1, 1);
      cache.append(0, 1);
      const plan = kvPagePlanForLane(cache, 0, PAGED_TENSORS);
      tiers.push(classifyPlan(plan).tier);
      slotOf.push(cache.physicalTokenIndex(0, position));
      options.kvPages = plan;
    } else {
      slotOf.push(position);
    }
    await engine.execute(decoderInputs(ids, keep), options);
  }
  const result = {
    tiers,
    slotOf,
    key: await readTensor(engine, 'self.k'),
    value: await readTensor(engine, 'self.v'),
    attention: await readTensor(engine, 'self.attention'),
    logits: await readTensor(engine, 'logits'),
    pageTable: cache ? Array.from(cache.pageTable.subarray(0, 4)) : null,
  };
  engine.dispose();
  return result;
}

/* Two requests decoding concurrently on one WebGPU context.
 *
 * This is the question paging exists to answer. Both lanes share the context,
 * the weights, the compiled pipelines and one physical page pool; only the
 * page table separates them. The prompt is seeded once and published as a
 * shared prefix, so both lanes' logical page zero is the *same* physical page —
 * copy-on-write territory — while every token they generate afterwards lands
 * in pages of their own.
 *
 * The oracle is one independent single-request CPU decode per lane. If the
 * lanes leaked into each other, a lane's logits would follow the other's
 * tokens and this would not match.
 */
async function runBatchedLanes() {
  const lanes = 2;
  const engine = await WebGPUEngine.init();
  if (!engine) throw new Error('No WebGPU adapter.');
  const graph = decoderGraph(lanes);
  await engine.allocateGraph(graph);
  /* Staggered prompts, so the lanes' active lengths differ for the whole run
   * and neither is ever the padded maximum for both. A padded-maximum
   * implementation is wrong only on the short lane, which is exactly the case
   * an equal-length batch would hide. */
  const prompts = [[1, 3], [4]];
  const generated = [[5, 2], [1, 3]];
  const ids = new Int32Array(lanes * SEQUENCE);
  const keep = new Int32Array(lanes * SEQUENCE);
  for (let lane = 0; lane < lanes; lane++) {
    for (const [index, token] of prompts[lane].entries()) {
      ids[lane * SEQUENCE + index] = token;
      keep[lane * SEQUENCE + index] = 1;
    }
  }
  const batchedInputs = () => ({ y_ids: Int32Array.from(ids), y_keep: Int32Array.from(keep) });
  await engine.execute(batchedInputs(), {
    incremental: true, incrementalReset: true, changedInputs: ['y_ids', 'y_keep'],
  });

  const produced = [[], []];
  const trace = [[], []];
  const whole = [];
  let dispatchNodes = 0;
  for (let step = 0; step < generated[0].length; step++) {
    const positions = [];
    for (let lane = 0; lane < lanes; lane++) {
      const position = prompts[lane].length + step;
      ids[lane * SEQUENCE + position] = generated[lane][step];
      keep[lane * SEQUENCE + position] = 1;
      positions.push(position);
    }
    await engine.execute(batchedInputs(), {
      incremental: true,
      changedInputs: ['y_ids', 'y_keep'],
      /* One step for both lanes. This is the whole claim: `lanes.length`
       * sequences advance in the dispatches of one. */
      incrementalRowLanes: positions.map((position) => ({ position })),
    });
    dispatchNodes = engine.executor.incrementalRowPlans.size;
    const logits = await readTensor(engine, 'logits');
    const key = await readTensor(engine, 'self.k');
    const value = await readTensor(engine, 'self.v');
    const attention = await readTensor(engine, 'self.attention');
    whole.push({
      logits, 'self.attention': attention, 'self.k': key, 'self.v': value,
      embed: await readTensor(engine, 'embed'),
      'self.q': await readTensor(engine, 'self.q'),
    });
    for (let lane = 0; lane < lanes; lane++) {
      const row = (lane * SEQUENCE + positions[lane]) * WIDTH;
      produced[lane].push(logits.slice(row, row + WIDTH));
      trace[lane].push({
        key: key.slice(row, row + WIDTH),
        value: value.slice(row, row + WIDTH),
        attention: attention.slice(row, row + WIDTH),
      });
    }
  }
  engine.dispose();
  return { produced, prompts, generated, dispatchNodes, trace, whole };
}

/* The same two-lane run on the CPU, whose batched decode is already proved
 * bit-identical to per-lane decoding. Comparing whole tensors against it says
 * *which row* diverges, which per-lane logits alone cannot. */
async function runCpuBatched(prompts, generated) {
  const lanes = prompts.length;
  const graph = decoderGraph(lanes);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const ids = new Int32Array(lanes * SEQUENCE);
  const keep = new Int32Array(lanes * SEQUENCE);
  for (let lane = 0; lane < lanes; lane++) {
    for (const [index, token] of prompts[lane].entries()) {
      ids[lane * SEQUENCE + index] = token;
      keep[lane * SEQUENCE + index] = 1;
    }
  }
  const inputs = () => ({ y_ids: Int32Array.from(ids), y_keep: Int32Array.from(keep) });
  await engine.execute(inputs(), {
    incremental: true, incrementalReset: true, changedInputs: ['y_ids', 'y_keep'],
  });
  const snapshots = [];
  for (let step = 0; step < generated[0].length; step++) {
    const positions = [];
    for (let lane = 0; lane < lanes; lane++) {
      const position = prompts[lane].length + step;
      ids[lane * SEQUENCE + position] = generated[lane][step];
      keep[lane * SEQUENCE + position] = 1;
      positions.push(position);
    }
    await engine.execute(inputs(), {
      incremental: true, changedInputs: ['y_ids', 'y_keep'],
      incrementalRowLanes: positions.map((position) => ({ position })),
    });
    snapshots.push(Object.fromEntries(
      ['logits', 'self.attention', 'self.k', 'self.v', 'embed', 'self.q'].map((name) =>
        [name, Array.from(graph.tensors.get(name).buffer)])));
  }
  return snapshots;
}

/** One lane decoded on its own at B=1, on the CPU: the oracle for a batch lane. */
async function runCpuLane(prompt, generated) {
  const graph = decoderGraph(1);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const ids = new Array(SEQUENCE).fill(0);
  const keep = new Array(SEQUENCE).fill(0);
  for (const [index, token] of prompt.entries()) { ids[index] = token; keep[index] = 1; }
  await engine.execute(decoderInputs(ids, keep), {
    incremental: true, incrementalReset: true, changedInputs: ['y_ids', 'y_keep'],
  });
  const produced = [];
  for (const [step, token] of generated.entries()) {
    const position = prompt.length + step;
    ids[position] = token;
    keep[position] = 1;
    await engine.execute(decoderInputs(ids, keep), {
      incremental: true,
      changedInputs: ['y_ids', 'y_keep'],
      incrementalRowPosition: position,
    });
    produced.push({
      logits: Array.from(graph.tensors.get('logits').buffer
        .subarray(position * WIDTH, (position + 1) * WIDTH)),
      key: Array.from(graph.tensors.get('self.k').buffer
        .subarray(position * WIDTH, (position + 1) * WIDTH)),
      value: Array.from(graph.tensors.get('self.v').buffer
        .subarray(position * WIDTH, (position + 1) * WIDTH)),
      attention: Array.from(graph.tensors.get('self.attention').buffer
        .subarray(position * WIDTH, (position + 1) * WIDTH)),
    });
  }
  return produced;
}

async function runConcurrentSlots() {
  const laneTokens = [[3, 4], [5, 2]];
  const engine = await WebGPUEngine.init();
  if (!engine) throw new Error('No WebGPU adapter.');
  const graph = decoderGraph();
  await engine.allocateGraph(graph);
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 1, laneTokenCapacity: 4, maxPages: 8, policy: 'paged',
  });
  const ids = [1, 0, 0, 0, 0, 0, 0, 0];
  const keep = [1, 0, 0, 0, 0, 0, 0, 0];

  await engine.execute(decoderInputs(ids, keep), {
    incremental: true, incrementalReset: true, changedInputs: ['y_ids', 'y_keep'],
  });
  /* The seed wrote token zero at physical slot zero. Publishing it and letting
   * both lanes acquire it is what makes one prefill serve two requests. */
  cache.append(0, 1);
  cache.publishPrefix('prompt', 0, 1);
  cache.releaseLane(0);
  cache.acquirePrefix('prompt', 0);
  cache.acquirePrefix('prompt', 1);

  const produced = [[], []];
  const tables = [[], []];
  const tiers = [[], []];
  for (let step = 0; step < laneTokens[0].length; step++) {
    for (const lane of [0, 1]) {
      const position = cache.kvLength[lane];
      cache.append(lane, 1);
      ids[position] = laneTokens[lane][step];
      keep[position] = 1;
      const plan = kvPagePlanForLane(cache, lane, PAGED_TENSORS);
      tiers[lane].push(classifyPlan(plan).tier);
      await engine.execute(decoderInputs(ids, keep), {
        incremental: true,
        changedInputs: ['y_ids', 'y_keep'],
        incrementalRowPosition: position,
        kvPages: plan,
      });
      const logits = await readTensor(engine, 'logits');
      produced[lane].push(logits.slice(position * WIDTH, (position + 1) * WIDTH));
    }
  }
  for (const lane of [0, 1]) {
    const base = lane * cache.pagesPerLane;
    tables[lane] = Array.from(cache.pageTable.subarray(base, base + 3));
  }
  engine.dispose();
  return { produced, tables, tiers, sharedPage: cache.pageIsShared(0, 0) };
}

/** One independent single-request CPU decode: the oracle for one lane. */
async function runCpuRequest(tokens) {
  const graph = decoderGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const ids = [1, 0, 0, 0, 0, 0, 0, 0];
  const keep = [1, 0, 0, 0, 0, 0, 0, 0];
  await engine.execute(decoderInputs(ids, keep), {
    incremental: true, incrementalReset: true, changedInputs: ['y_ids', 'y_keep'],
  });
  const produced = [];
  tokens.forEach((token, index) => {
    const position = index + 1;
    ids[position] = token;
    keep[position] = 1;
    engine.execute(decoderInputs(ids, keep), {
      incremental: true,
      changedInputs: ['y_ids', 'y_keep'],
      incrementalRowPosition: position,
    });
    produced.push(Array.from(graph.tensors.get('logits').buffer
      .subarray(position * WIDTH, (position + 1) * WIDTH)));
  });
  return produced;
}

function equal(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

const failures = [];
const check = (condition, message) => {
  if (condition) console.log(`ok ${message}`);
  else { console.log(`not ok ${message}`); failures.push(message); }
};

const cpu = await runCpu();
const contiguous = await runWebGPU(false);
const paged = await runWebGPU(true);

check(equal(paged.tiers, ['gather', 'gather', 'gather']),
  `the paged run actually pages (tiers: ${paged.tiers.join(',')})`);
check(equal(paged.pageTable, [0, 2, 4, 6]),
  `the decoding lane holds scattered pages (${paged.pageTable?.join(',')})`);

/* WebGPU's contiguous run against the CPU oracle first: a failure here is a
 * pre-existing decode bug, not a paging bug, and separating the two is the
 * difference between a diagnosis and a guess. */
for (const [index, [position]] of STEPS.entries()) {
  const row = (values, slot) => values.slice(slot * WIDTH, (slot + 1) * WIDTH);
  check(equal(row(contiguous.key, position), cpu[index].key),
    `contiguous WebGPU self.k row ${position} matches CPU`);
  check(equal(row(contiguous.value, position), cpu[index].value),
    `contiguous WebGPU self.v row ${position} matches CPU`);

  // The write half: logical row N lives in the slot the page table names.
  const slot = paged.slotOf[index];
  check(equal(row(paged.key, slot), cpu[index].key),
    `paged WebGPU self.k logical row ${position} lives in physical slot ${slot}`);
  check(equal(row(paged.value, slot), cpu[index].value),
    `paged WebGPU self.v logical row ${position} lives in physical slot ${slot}`);

  /* The read half, and the one that matters. Attention output is not paged, so
   * it stays at `position * width` -- but its *value* can only be right if the
   * staged prefix fed the shader the mapped slots in logical order. A staging
   * copy that read the wrong pages lands here and nowhere else. */
  check(equal(row(paged.attention, position), cpu[index].attention),
    `paged WebGPU self.attention row ${position} matches the contiguous CPU oracle`);
  check(equal(row(paged.logits, position), cpu[index].logits),
    `paged WebGPU logits row ${position} matches the contiguous CPU oracle`);
  check(equal(row(contiguous.attention, position), cpu[index].attention),
    `contiguous WebGPU self.attention row ${position} matches CPU`);
}

const concurrent = await runConcurrentSlots();
const laneTokens = [[3, 4], [5, 2]];
check(concurrent.sharedPage,
  'both lanes hold the same physical page for the shared prompt');
/* Lowest free page first, and the two lanes take turns: lane 0 gets 1 then 3,
 * lane 1 gets 2 then 4. Neither lane's mapping is contiguous, which is the
 * point -- a scheduler interleaving requests produces exactly this. */
check(equal(concurrent.tables[0], [0, 1, 3]) && equal(concurrent.tables[1], [0, 2, 4]),
  `lanes share page 0 and interleave afterwards (${concurrent.tables[0]} | ${concurrent.tables[1]})`);
check(equal(concurrent.tiers[0], ['identity', 'gather']) &&
      equal(concurrent.tiers[1], ['gather', 'gather']),
  `both lanes reach the gather tier (${concurrent.tiers[0]} | ${concurrent.tiers[1]})`);
for (const lane of [0, 1]) {
  const oracle = await runCpuRequest(laneTokens[lane]);
  for (const [index, expected] of oracle.entries()) {
    check(equal(concurrent.produced[lane][index], expected),
      `concurrent slot ${lane} token ${index + 1} matches its own single-request CPU decode`);
  }
}

/*
 * B>1: two sequences advancing in one set of dispatches.
 *
 * The oracle is each lane decoded on its own at B=1, and the comparison is
 * equality rather than closeness. That is not optimism about float error --
 * both sides are int8, and every kernel here *skips* a masked key rather than
 * scoring it and multiplying by zero, so a lane's visible key set and its
 * accumulation order in the padded batch are exactly what its one-lane decode
 * produces. If a future kernel starts scoring masked keys, this is what
 * notices. The lanes are staggered so the shorter one is never at the padded
 * maximum, which is the case a padded-maximum implementation gets wrong.
 */
const batched = await runBatchedLanes();
const cpuBatched = await runCpuBatched(batched.prompts, batched.generated);
/* Whole tensors against the CPU's own two-lane run first. Per-lane logits say
 * *that* a batch diverged; this says *which row*, which is the difference
 * between a diagnosis and a bisect -- the first failure of this work was a
 * missing buffer usage flag that presented as wrong attention, and this is
 * what named the embedding as the actual origin. */
for (const [step, expected] of cpuBatched.entries()) {
  for (const name of ['embed', 'self.q', 'self.k', 'self.v', 'self.attention', 'logits']) {
    const got = batched.whole[step][name];
    const bad = [];
    for (let row = 0; row * WIDTH < expected[name].length; row++) {
      const a = got.slice(row * WIDTH, (row + 1) * WIDTH);
      const b = expected[name].slice(row * WIDTH, (row + 1) * WIDTH);
      if (!equal(a, b)) bad.push(`${row}[${a}]!=[${b}]`);
    }
    check(bad.length === 0,
      `batched step ${step} ${name} matches the CPU batched run` +
      (bad.length ? ` (rows ${bad.slice(0, 3).join(' ')})` : ''));
  }
}
check(batched.dispatchNodes >= 5,
  `the batched step covers the whole decoder closure (${batched.dispatchNodes} nodes)`);
for (const lane of [0, 1]) {
  const oracle = await runCpuLane(batched.prompts[lane], batched.generated[lane]);
  for (const [index, expected] of oracle.entries()) {
    for (const port of ['key', 'value', 'attention']) {
      check(equal(batched.trace[lane][index][port], expected[port]),
        `batched lane ${lane} token ${index + 1} ${port} matches its own B=1 decode`);
    }
    check(equal(batched.produced[lane][index], expected.logits),
      `batched lane ${lane} token ${index + 1} is bit-identical to its own B=1 decode`);
  }
}

console.log(failures.length ? `FAIL ${failures.length} check(s)` : 'PASS paged KV WebGPU');
if (failures.length) Deno.exit(1);
