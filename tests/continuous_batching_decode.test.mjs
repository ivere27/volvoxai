import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { PagedKVCache } from '../ts/core/PagedKVCache.js';
import { ContinuousBatchScheduler } from '../ts/core/ContinuousBatchScheduler.js';
import { classifyPlan } from '../ts/backends/kvPageAddressing.js';

/*
 * The pieces, composed.
 *
 * Everything else tests one layer: the row executor against a dense oracle, the
 * paged cache against its corpus, the scheduler against its corpus with a
 * synthetic step. Nothing until here runs the scheduler over a *real* batched
 * decode, which is where the layers' assumptions about each other live -- who
 * owns the lane count, when the page plan is read, what a round does when one
 * request finishes before another.
 *
 * This is the stateful-session composition gate described in
 * ../docs/scheduling-and-dynamic-batching-design.md#stateful-sessions-prefill-decode-and-paged-kv:
 * compare scheduler outputs with independently executed single-request contexts
 * at every decode step.
 *
 * The two requests share a prompt, so admission binds a published prefix and
 * both go straight to decoding -- the time-to-first-token that sharing exists
 * to buy, and the path a synthetic step cannot exercise. They ask for different
 * token counts, so one retires while the other keeps going and its lane parks.
 */

const SEQUENCE = 8;
const WIDTH = 4;
const VOCABULARY = 6;
const LANES = 2;
const PAGED_TENSORS = ['self.k', 'self.v'];
const PROMPT = [1, 3];
/*
 * Different token counts on purpose: the shorter request retires mid-run, the
 * scheduler releases its lane, and its pages go back to the pool. The dense
 * batch still needs a row for that lane in the following rounds, and it has no
 * resident page to write -- so it *parks*, occupying a row for shape while its
 * output is discarded. That is the case a fixed-batch implementation gets
 * wrong, and the reason `parked` exists.
 */
const GENERATED = [[5, 2, 4], [4, 1]];

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

function tokensFor(request, steps) {
  return [...PROMPT, ...GENERATED[request].slice(0, steps)];
}

async function engineFor(lanes) {
  const graph = decoderGraph(lanes);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return { graph, engine };
}

function laneInputs(lanes, tokensByLane) {
  const ids = new Int32Array(lanes * SEQUENCE);
  const keep = new Int32Array(lanes * SEQUENCE);
  for (let lane = 0; lane < lanes; lane++) {
    const tokens = tokensByLane[lane] ?? [];
    for (let position = 0; position < tokens.length; position++) {
      ids[lane * SEQUENCE + position] = tokens[position];
      keep[lane * SEQUENCE + position] = 1;
    }
  }
  return { ids, keep };
}

function row(buffer, lane, position) {
  const base = (lane * SEQUENCE + position) * WIDTH;
  return Array.from(buffer.subarray(base, base + WIDTH));
}

/** One request decoded alone, through the established B=1 row path. */
async function singleRequestDecode(request) {
  const { engine } = await engineFor(1);
  const names = ['ids', 'keep'];
  await engine.execute(laneInputs(1, [tokensFor(request, 0)]), {
    incremental: true, incrementalReset: true, changedInputs: names,
  });
  const produced = [];
  for (let step = 1; step <= GENERATED[request].length; step++) {
    const position = PROMPT.length - 1 + step;
    const result = await engine.execute(laneInputs(1, [tokensFor(request, step)]), {
      incremental: true, changedInputs: names, incrementalRowPosition: position,
    });
    produced.push(row(result.logits, 0, position));
  }
  return produced;
}

test('the scheduler drives a real batched decode that matches single-request contexts',
  async () => {
    const { graph, engine } = await engineFor(LANES);
    const names = ['ids', 'keep'];
    const cache = new PagedKVCache({
      lanes: LANES,
      pageTokens: 1,
      laneTokenCapacity: SEQUENCE,
      maxPages: LANES * SEQUENCE,
      policy: 'paged',
    });

    /* The prompt is seeded once and published, exactly as a server holding a
     * shared system prompt does. Both lanes then bind the same physical pages,
     * so the prefill they skip is real work and not bookkeeping. */
    await engine.execute(laneInputs(LANES, [PROMPT, PROMPT]), {
      incremental: true, incrementalReset: true, changedInputs: names,
    });
    cache.append(0, PROMPT.length);
    cache.publishPrefix('shared', 0, PROMPT.length);
    cache.releaseLane(0);

    const scheduler = new ContinuousBatchScheduler({
      cache, maxQueueDepth: 4, pagedTensors: PAGED_TENSORS,
    });
    const ids = [];
    for (const request of [0, 1]) {
      ids.push(scheduler.submit({
        promptTokens: PROMPT.length,
        maxTokens: GENERATED[request].length,
        promptKey: 'shared',
        payload: request,
      }));
    }

    /* Every lane's token stream, grown as the scheduler produces tokens. A
     * lane with no live request keeps its last stream, because an idle lane
     * repeats its own last row and must therefore see the same inputs. */
    const streams = [tokensFor(0, 0), tokensFor(1, 0)];
    const produced = [[], []];
    const batchWidths = [];
    const tiers = [];

    await scheduler.runUntilIdle(async (batch) => {
      assert.ok(batch.every((work) => work.phase === 'decode'),
        'a shared prefix covering the whole prompt must skip prefill entirely');
      batchWidths.push(batch.length);

      /* Grow each active lane's stream by the token this step will write. */
      const active = new Map();
      for (const work of batch) {
        const request = work.payload;
        streams[work.slot] = tokensFor(request, work.generated + 1);
        active.set(work.slot, work);
      }

      /* Every declared lane appears, active or not. A dense batch cannot drop a
       * lane without changing every operand's shape, so an unoccupied or
       * finished slot repeats its own last row -- which writes the same bytes
       * it already holds. */
      /* Every declared lane appears, live or not. A dense batch cannot drop a
       * lane without changing every operand's shape, and a slot whose request
       * retired has had its pages returned to the pool -- so it parks rather
       * than idling: there is no row left for it to repeat. */
      const lanes = [];
      for (let slot = 0; slot < LANES; slot++) {
        const work = active.get(slot);
        if (!work) {
          lanes.push({ parked: true });
          continue;
        }
        /* The scheduler publishes an immutable snapshot that includes this
         * step's provisional reservation. A fresh cache view would expose only
         * committed length until the callback succeeds. */
        const plan = work.kvPages;
        assert.equal(plan.lane, slot);
        assert.equal(plan.kvLength, work.position + work.tokens);
        tiers.push(classifyPlan(plan).tier);
        lanes.push({ position: work.position, kvPages: plan });
      }

      const result = await engine.execute(laneInputs(LANES, streams), {
        incremental: true, changedInputs: names, incrementalRowLanes: lanes,
      });
      return batch.map((work) => {
        produced[work.payload].push(row(result.logits, work.slot, work.position));
        return { value: null };
      });
    });

    /* Both requests completed, and the scheduler entered the engine once per
     * round rather than once per request. */
    assert.equal(scheduler.results.length, 2);
    for (const result of scheduler.results) {
      assert.equal(result.state, 'completed', result.error?.message);
    }
    const telemetry = scheduler.telemetry();
    assert.equal(telemetry.prefixReuses, 2, 'both requests must bind the shared prefix');
    assert.equal(telemetry.prefillTokens, 0, 'a bound prefix prefills nothing');
    assert.equal(telemetry.steps, GENERATED[0].length + GENERATED[1].length);
    assert.equal(telemetry.dispatches, GENERATED[0].length,
      'one dispatch per round: the round, not the request, is what enters the engine');
    /* The shorter request retires first, so the last round carries one live
     * lane beside one parked one -- the case an implementation that assumed a
     * full batch gets wrong. */
    assert.deepEqual(batchWidths, [2, 2, 1]);
    assert.ok(tiers.includes('gather'),
      `the run must actually page, saw ${JSON.stringify(tiers)}`);

    /* And the numbers: every token of every request equals what that request
     * produces decoding alone. */
    for (const request of [0, 1]) {
      const alone = await singleRequestDecode(request);
      assert.deepEqual(produced[request], alone,
        `request ${request} must match its own single-request decode at every step`);
    }
    void graph;
    void ids;
  });

test('a request retiring mid-generation leaves the survivor bit-identical', async () => {
  /* The same run, stopped one round early, must agree token for token with the
   * full run up to that point. A lane whose neighbour retired must not shift:
   * the padded key extent shrinks when the longer lane is alone, and a kernel
   * that let the padding participate would change the survivor's answer exactly
   * at the round the other request left. */
  const runTo = async (rounds) => {
    const { engine } = await engineFor(LANES);
    const names = ['ids', 'keep'];
    await engine.execute(laneInputs(LANES, [PROMPT, PROMPT]), {
      incremental: true, incrementalReset: true, changedInputs: names,
    });
    const streams = [tokensFor(0, 0), tokensFor(1, 0)];
    const produced = [];
    for (let step = 1; step <= rounds; step++) {
      const lanes = [];
      for (let lane = 0; lane < LANES; lane++) {
        const live = step <= GENERATED[lane].length;
        if (live) streams[lane] = tokensFor(lane, step);
        const position = live
          ? PROMPT.length - 1 + step
          : PROMPT.length - 1 + GENERATED[lane].length;
        lanes.push({ position });
      }
      const result = await engine.execute(laneInputs(LANES, streams), {
        incremental: true, changedInputs: names, incrementalRowLanes: lanes,
      });
      produced.push(row(result.logits, 0, PROMPT.length - 1 + step));
    }
    return produced;
  };
  const short = await runTo(2);
  const full = await runTo(3);
  assert.deepEqual(full.slice(0, 2), short,
    "the surviving lane's earlier tokens must not change");

  /* And the survivor's whole stream equals its own single-request decode, which
   * is the statement that losing a neighbour changed nothing at all. */
  assert.deepEqual(full, await singleRequestDecode(0));
});
