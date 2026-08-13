import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { PagedKVCache } from '../ts/core/PagedKVCache.js';
import {
  ContinuousBatchScheduler,
  SchedulerError,
} from '../ts/core/ContinuousBatchScheduler.js';

/* native/tests/test_continuous_batch_scheduler.c runs this same corpus through the C
 * scheduler. The model is synthetic on both sides: one token per step, finish
 * at maxTokens. What is compared is the scheduling — who is admitted, into
 * which slot, holding which pages, retired in what order. */

const VECTORS = JSON.parse(
  readFileSync(new URL('./continuous_batching_vectors.json', import.meta.url), 'utf8'));

function attempt(run) {
  try {
    run();
    return 'ok';
  } catch (error) {
    if (error instanceof SchedulerError) return error.code;
    throw error;
  }
}

class Harness {
  constructor(options) {
    this.cache = new PagedKVCache({
      lanes: options.slots,
      pageTokens: options.pageTokens,
      laneTokenCapacity: options.laneTokenCapacity,
      maxPages: options.maxPages,
      policy: 'paged',
    });
    this.scheduler = new ContinuousBatchScheduler({
      cache: this.cache,
      maxQueueDepth: options.maxQueueDepth,
      /* The shared scheduling corpus asserts the complete retirement history.
       * Keep that independent of the production default's bounded archive. */
      maxRetainedResults: 1024,
    });
    /* Corpus ids are stable labels; the scheduler assigns its own. */
    this.ids = new Map();
    this.failing = new Set();
  }

  /* One call carries the round's lanes, so the outcome list is per lane and in
   * the same order. A synthetic lane error exercises the scheduler's atomic
   * batch contract: every reservation in the selected batch rolls back. */
  runStep = (batch) => batch.map((work) => {
    if (this.failing.delete(work.requestId)) {
      return { error: new Error(`synthetic failure for request ${work.requestId}`) };
    }
    /* The step must see a page for everything the published length covers.
     * Resolving each one is the cheapest possible stand-in for a decoder and
     * catches a scheduler that publishes a length it did not reserve. */
    for (let position = 0; position < work.kvPages.kvLength; position++) {
      this.cache.physicalTokenIndex(work.slot, position);
    }
    return { value: work.position };
  });

  label(requestId) {
    for (const [label, id] of this.ids) if (id === requestId) return label;
    return requestId;
  }
}

async function applyOperation(harness, operation, label) {
  const { scheduler, cache } = harness;
  const expected = operation.status ?? 'ok';
  switch (operation.op) {
    case 'submit': {
      const status = attempt(() => {
        harness.ids.set(operation.id, scheduler.submit({
          promptTokens: operation.promptTokens,
          maxTokens: operation.maxTokens,
          promptKey: operation.promptKey,
        }));
      });
      assert.equal(status, expected, `${label} submit`);
      if (status !== 'ok') harness.ids.delete(operation.id);
      return;
    }
    case 'cancel': {
      assert.equal(scheduler.cancel(harness.ids.get(operation.id)), operation.value,
        `${label} cancel`);
      return;
    }
    case 'fail_next_step': {
      harness.failing.add(harness.ids.get(operation.id));
      return;
    }
    case 'step': {
      await scheduler.step(harness.runStep);
      return;
    }
    case 'run_until_idle': {
      await scheduler.runUntilIdle(harness.runStep);
      return;
    }
    case 'close': {
      await scheduler.close(harness.runStep, { drain: operation.drain === true });
      return;
    }
    case 'expect_state': {
      assert.equal(scheduler.state(harness.ids.get(operation.id)), operation.state,
        `${label} state`);
      return;
    }
    case 'expect_slot': {
      assert.equal(scheduler.slotOf(harness.ids.get(operation.id)), operation.slot,
        `${label} slot`);
      return;
    }
    case 'expect_generated': {
      const result = scheduler.results.find(
        (entry) => entry.requestId === harness.ids.get(operation.id));
      assert.equal(result?.generated, operation.value, `${label} generated`);
      return;
    }
    case 'expect_queue_depth': {
      assert.equal(scheduler.queueDepth, operation.value, `${label} queue depth`);
      return;
    }
    case 'expect_lengths': {
      assert.deepEqual(Array.from(cache.kvLength), operation.kv, `${label} kv lengths`);
      return;
    }
    case 'expect_page_table': {
      const base = operation.slot * cache.pagesPerLane;
      assert.deepEqual(
        Array.from(cache.pageTable.subarray(base, base + cache.pagesPerLane)),
        operation.pages, `${label} page table`);
      return;
    }
    case 'expect_results': {
      assert.deepEqual(
        scheduler.results.map((entry) => harness.label(entry.requestId)),
        operation.order, `${label} result order`);
      if (operation.states) {
        assert.deepEqual(scheduler.results.map((entry) => entry.state),
          operation.states, `${label} result states`);
      }
      return;
    }
    case 'expect_telemetry': {
      const telemetry = scheduler.telemetry();
      for (const [field, value] of Object.entries(operation)) {
        /* `note` carries the derivation of the numbers beside them, so the
         * corpus can say *why* a count is what it is without a reader having to
         * re-derive it. It is documentation, not an expectation. */
        if (field === 'op' || field === 'note') continue;
        assert.equal(telemetry[field], value, `${label} telemetry.${field}`);
      }
      return;
    }
    default:
      throw new Error(`${label} uses unknown operation '${operation.op}'`);
  }
}

test('continuous batching corpus is self-describing', () => {
  assert.equal(VECTORS.version, 'volvox-continuous-batching/v1');
  assert.equal(VECTORS.cases.length > 0, true);
});

for (const testCase of VECTORS.cases) {
  test(`continuous batching vector: ${testCase.name}`, async () => {
    const harness = new Harness(testCase.options);
    let index = 0;
    for (const operation of testCase.operations) {
      await applyOperation(harness, operation, `${testCase.name}[${index++}]`);
    }
  });
}

test('an identical schedule produces an identical result list', async () => {
  const run = async () => {
    const cache = new PagedKVCache({
      lanes: 3, pageTokens: 2, laneTokenCapacity: 16, maxPages: 12, policy: 'paged',
    });
    const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 16 });
    const lengths = [1, 5, 2, 4, 3, 1, 6, 2];
    lengths.forEach((maxTokens, index) => {
      scheduler.submit({ promptTokens: 2 + (index % 3) * 2, maxTokens });
    });
    await scheduler.runUntilIdle((batch) => batch.map(() => ({ value: null })));
    return scheduler.results.map((entry) => `${entry.requestId}:${entry.state}:${entry.generated}`);
  };
  assert.deepEqual(await run(), await run());
});

test('a slow request does not block a short one behind it', async () => {
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 4, laneTokenCapacity: 32, maxPages: 16, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 8 });
  const slow = scheduler.submit({ promptTokens: 4, maxTokens: 12 });
  const quick = scheduler.submit({ promptTokens: 4, maxTokens: 1 });
  await scheduler.runUntilIdle((batch) => batch.map(() => ({ value: null })));
  const order = scheduler.results.map((entry) => entry.requestId);
  /* Both were admitted in the first round; the short one retires while the
   * long one is still generating. Admission order is not completion order,
   * which is the whole point. */
  assert.deepEqual(order, [quick, slow]);
});

test('page reuse after retirement cannot expose the previous occupant', async () => {
  const cache = new PagedKVCache({
    lanes: 1, pageTokens: 2, laneTokenCapacity: 8, maxPages: 4, policy: 'paged',
  });
  const erased = [];
  cache.setPageEraser((page) => erased.push(page));
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 4 });
  scheduler.submit({ promptTokens: 2, maxTokens: 1 });
  scheduler.submit({ promptTokens: 2, maxTokens: 1 });

  const generations = [];
  await scheduler.runUntilIdle((batch) => batch.map((work) => {
    generations.push([work.requestId, work.slotGeneration]);
    return { value: null };
  }));
  assert.equal(scheduler.results.length, 2);
  /* The second request occupies the same lane and the same physical pages.
   * Its lane generation differs, and every page it received was erased on
   * recycle -- the two independent defences against reading the first
   * request's KV. */
  const [first, second] = scheduler.results.map((entry) => entry.requestId);
  const generationOf = (id) => generations.find(([request]) => request === id)[1];
  assert.notEqual(generationOf(first), generationOf(second));
  assert.equal(erased.length > 2, true);
});

test('a prompt shorter than one page is not shareable, and says so by doing nothing', async () => {
  /* A prefix must end on a page boundary, so there is nothing to publish. The
   * request must still run normally -- publication is an optimisation, and a
   * cache that cannot take a prefix must never fail the request that produced
   * it. */
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 4, laneTokenCapacity: 16, maxPages: 8, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 4 });
  scheduler.submit({ promptTokens: 3, maxTokens: 1, promptKey: 'tiny' });
  scheduler.submit({ promptTokens: 3, maxTokens: 1, promptKey: 'tiny' });
  await scheduler.runUntilIdle((batch) => batch.map(() => ({ value: null })));
  assert.equal(cache.hasPrefix('tiny'), false);
  const telemetry = scheduler.telemetry();
  assert.equal(telemetry.prefixPublications, 0);
  assert.equal(telemetry.prefixReuses, 0);
  assert.equal(telemetry.completed, 2);
  /* Both prompts were prefilled in full: nothing was shared, and the counter
   * says so rather than quietly reporting a saving that did not happen. */
  assert.equal(telemetry.prefillTokens, 6);
  assert.equal(telemetry.sharedPromptTokens, 0);
});

test('a resident prefix longer than the prompt is refused, not bound', async () => {
  /* Two requests declaring the same identity for different prompt lengths is a
   * caller error, and the shorter one must not silently inherit tokens it never
   * sent. The scheduler prefills it honestly instead. */
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 2, laneTokenCapacity: 16, maxPages: 16, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 4 });
  scheduler.submit({ promptTokens: 8, maxTokens: 1, promptKey: 'shared' });
  await scheduler.step((batch) => batch.map(() => ({ value: null })));
  assert.equal(cache.hasPrefix('shared'), true);

  scheduler.submit({ promptTokens: 4, maxTokens: 1, promptKey: 'shared' });
  await scheduler.step((batch) => batch.map(() => ({ value: null })));
  const telemetry = scheduler.telemetry();
  assert.equal(telemetry.prefixReuses, 0, 'an over-long prefix must not be bound');
  assert.equal(telemetry.sharedPromptTokens, 0);
  assert.equal(telemetry.prefillTokens, 12, 'both prompts prefilled in full');
  await scheduler.runUntilIdle((batch) => batch.map(() => ({ value: null })));
  assert.equal(scheduler.telemetry().completed, 2);
});

test('sharing collapses repeated prompts to one prefill after the first round', async () => {
  /* The claim the feature exists to make, as a number -- and the exact shape of
   * its limit. Four requests share an 8-token prompt on two slots:
   *
   *   round 1: both admitted slots miss. Publication happens *after* a prefill
   *            runs, so nothing is resident when they are admitted.
   *   round 2+: the remaining two bind the published pages and skip prefill.
   *
   * So the saving is real but starts one round late. Holding a same-key request
   * back until its sibling publishes would close that gap; it would also break
   * strict head-first admission, so it is not done here. The counters report
   * what happened rather than what the feature aspires to. */
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 4, laneTokenCapacity: 16, maxPages: 16, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 8 });
  for (let request = 0; request < 4; request++) {
    scheduler.submit({ promptTokens: 8, maxTokens: 2, promptKey: 'system' });
  }
  await scheduler.runUntilIdle((batch) => batch.map(() => ({ value: null })));
  const telemetry = scheduler.telemetry();
  assert.equal(telemetry.completed, 4);
  assert.equal(telemetry.prefixPublications, 1, 'published once, not per request');
  assert.equal(telemetry.prefixReuses, 2);
  assert.equal(telemetry.prefillTokens, 16, 'two prompts prefilled, not four');
  assert.equal(telemetry.sharedPromptTokens, 16, 'two prompts served from pages');
});

test('backpressure refuses rather than growing the queue without bound', () => {
  const cache = new PagedKVCache({
    lanes: 1, pageTokens: 2, laneTokenCapacity: 8, maxPages: 4, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 2 });
  scheduler.submit({ promptTokens: 2, maxTokens: 1 });
  scheduler.submit({ promptTokens: 2, maxTokens: 1 });
  assert.throws(() => scheduler.submit({ promptTokens: 2, maxTokens: 1 }),
    (error) => error.code === 'QUEUE_FULL');
  assert.equal(scheduler.queueDepth, 2);
});

test('draining close finishes admitted work and cancels the queue', async () => {
  const cache = new PagedKVCache({
    lanes: 1, pageTokens: 2, laneTokenCapacity: 8, maxPages: 4, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 4 });
  const admitted = scheduler.submit({ promptTokens: 2, maxTokens: 2 });
  const queued = scheduler.submit({ promptTokens: 2, maxTokens: 2 });
  await scheduler.step((batch) => batch.map(() => ({ value: null })));
  await scheduler.close((batch) => batch.map(() => ({ value: null })), { drain: true });
  const states = new Map(scheduler.results.map((entry) => [entry.requestId, entry.state]));
  assert.equal(states.get(admitted), 'completed');
  assert.equal(states.get(queued), 'cancelled');
  /* Nothing is abandoned silently: both accepted requests carry an outcome. */
  assert.equal(scheduler.results.length, 2);
});

test('the scheduler hands every step a page plan for its own lane', async () => {
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 2, laneTokenCapacity: 8, maxPages: 8, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({
    cache, maxQueueDepth: 4, pagedTensors: ['self.k', 'self.v'],
  });
  scheduler.submit({ promptTokens: 2, maxTokens: 2 });
  scheduler.submit({ promptTokens: 2, maxTokens: 2 });
  const seen = [];
  await scheduler.runUntilIdle((batch) => batch.map((work) => {
    seen.push({
      slot: work.slot,
      lane: work.kvPages.lane,
      kvLength: work.kvPages.kvLength,
      position: work.position,
      paged: [...work.kvPages.pagedTensors].sort(),
    });
    return { value: null };
  }));
  for (const entry of seen) {
    assert.equal(entry.lane, entry.slot, 'a plan must address its own lane');
    assert.deepEqual(entry.paged, ['self.k', 'self.v']);
    /* The published length always covers the position being written. */
    assert.equal(entry.kvLength > entry.position, true);
  }
});

test('a round advances every decoding lane in one dispatch', async () => {
  /* The reason the callback takes a list. Engines execute a batched decode step
   * as one dispatch per node regardless of lane count, so a scheduler that still
   * called per slot would put the per-request dispatch straight back on top of
   * the backend that had just removed it.
   *
   * Both counters are checked because either alone can be made to look good:
   * `steps` counts the lane-work that was asked for and `dispatches` counts the
   * times the backend was entered, and it is the ratio that is the win. */
  const cache = new PagedKVCache({
    lanes: 3, pageTokens: 1, laneTokenCapacity: 6, maxPages: 18, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 8 });
  const widths = [];
  for (let request = 0; request < 3; request++) {
    scheduler.submit({ promptTokens: 1, maxTokens: 3 });
  }
  await scheduler.runUntilIdle((batch) => {
    widths.push({ phase: batch[0].phase, lanes: batch.length });
    return batch.map(() => ({ value: null }));
  });

  const decodes = widths.filter((entry) => entry.phase === 'decode');
  assert.ok(decodes.length > 0, 'the run must decode');
  assert.deepEqual(decodes.map((entry) => entry.lanes), decodes.map(() => 3),
    `every decode call must carry all three lanes, saw ${JSON.stringify(decodes)}`);
  /* Prefill stays one request per call: prompts of different lengths are not
   * one dense step, and this run's prompts are only equal by construction. */
  assert.deepEqual(widths.filter((entry) => entry.phase === 'prefill')
    .map((entry) => entry.lanes), [1, 1, 1]);

  const telemetry = scheduler.telemetry();
  assert.equal(telemetry.steps, 3 * 1 + 3 * 3, 'three prefills and nine lane-steps');
  assert.equal(telemetry.dispatches, 3 + 3, 'three prefill calls and three decode calls');
  assert.equal(telemetry.completed, 3);
});

test('a batched dispatch that throws retires every lane in it', async () => {
  /* A throw means the dispatch did not happen, so no lane in it advanced.
   * Retiring only one would leave the others believing they had produced a
   * token the backend never computed. */
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 1, laneTokenCapacity: 6, maxPages: 12, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 8 });
  scheduler.submit({ promptTokens: 1, maxTokens: 3 });
  scheduler.submit({ promptTokens: 1, maxTokens: 3 });
  let decodes = 0;
  await scheduler.runUntilIdle((batch) => {
    if (batch[0].phase === 'decode' && decodes++ === 0) {
      throw new Error('synthetic batched dispatch failure');
    }
    return batch.map(() => ({ value: null }));
  });
  assert.equal(scheduler.results.length, 2);
  for (const result of scheduler.results) {
    assert.equal(result.state, 'failed');
    assert.match(result.error.message, /synthetic batched dispatch failure/);
  }
  assert.equal(scheduler.telemetry().activeSlots, 0, 'both lanes must be released');
});

test('a step returning the wrong number of outcomes fails the whole batch', async () => {
  /* Losing track of which result belongs to which request is not recoverable by
   * guessing an alignment: the guess hands one request another's tokens. */
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 1, laneTokenCapacity: 6, maxPages: 12, policy: 'paged',
  });
  const scheduler = new ContinuousBatchScheduler({ cache, maxQueueDepth: 8 });
  scheduler.submit({ promptTokens: 1, maxTokens: 3 });
  scheduler.submit({ promptTokens: 1, maxTokens: 3 });
  await scheduler.runUntilIdle((batch) =>
    batch[0].phase === 'decode' ? [{ value: null }] : batch.map(() => ({ value: null })));
  assert.equal(scheduler.results.length, 2);
  for (const result of scheduler.results) {
    assert.equal(result.state, 'failed');
    assert.match(result.error.message, /2-lane step returned 1 outcome\(s\)/);
  }
});
