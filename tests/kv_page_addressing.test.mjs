import test from 'node:test';
import assert from 'node:assert/strict';

import { PagedKVCache } from '../ts/core/PagedKVCache.js';
import {
  classifyPlan,
  defaultScratchAllocator,
  kvPageCopyRanges,
  kvPagePlanForLane,
  pagedRunOrigin,
  pagedSequenceView,
  pagedSlot,
} from '../ts/backends/kvPageAddressing.js';

/* The addressing layer between a page table and an attention operand. These
 * are pure functions, so the properties WebGPU and the native GPU packs depend
 * on are provable here rather than only on a machine with an adapter. */

function cacheWith(schedule, options) {
  const cache = new PagedKVCache(options);
  for (const [lane, tokens] of schedule) cache.append(lane, tokens);
  return cache;
}

test('an identity mapping is one run and costs no copy', () => {
  const cache = cacheWith([[0, 6]], {
    lanes: 2, pageTokens: 2, laneTokenCapacity: 8, policy: 'contiguous',
  });
  const plan = kvPagePlanForLane(cache, 0, ['k']);
  assert.equal(classifyPlan(plan).tier, 'identity');
  assert.deepEqual(kvPageCopyRanges(plan), [{ source: 0, destination: 0, tokens: 6 }]);
  assert.equal(pagedRunOrigin(plan), 0);

  const storage = Float32Array.from({ length: 32 }, (_, index) => index);
  const view = pagedSequenceView(storage, 2, plan, 'k', defaultScratchAllocator());
  assert.equal(view.gather, null, 'a run must alias the pool rather than copy it');
  assert.deepEqual(Array.from(view.storage), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
});

test('a lane based away from the origin is still one run', () => {
  const cache = cacheWith([[1, 4]], {
    lanes: 2, pageTokens: 2, laneTokenCapacity: 8, policy: 'contiguous',
  });
  const plan = kvPagePlanForLane(cache, 1, ['k']);
  assert.equal(classifyPlan(plan).tier, 'identity');
  /* Identity for lane one means pages 4 and 5, so the run starts at physical
   * slot 8 -- which a backend binding its operands at buffer offset zero
   * cannot read, and pagedRunOrigin says so. */
  assert.deepEqual(kvPageCopyRanges(plan), [{ source: 8, destination: 0, tokens: 4 }]);
  assert.equal(pagedRunOrigin(plan), 8);
});

test('an interleaved mapping coalesces into one copy per contiguous page run', () => {
  const cache = cacheWith([[0, 2], [1, 2], [0, 2], [0, 2]], {
    lanes: 2, pageTokens: 2, laneTokenCapacity: 8, maxPages: 8, policy: 'paged',
  });
  const plan = kvPagePlanForLane(cache, 0, ['k']);
  assert.deepEqual(Array.from(plan.pageTable), [0, 2, 3, -1]);
  assert.equal(classifyPlan(plan).tier, 'gather');
  /* Pages 2 and 3 are adjacent, so the four tokens they hold are one copy.
   * Three logical pages become two copies, not three. */
  assert.deepEqual(kvPageCopyRanges(plan), [
    { source: 0, destination: 0, tokens: 2 },
    { source: 4, destination: 2, tokens: 4 },
  ]);
  assert.equal(pagedRunOrigin(plan), -1);
});

test('a gather reads the mapped slots in logical order', () => {
  const cache = cacheWith([[0, 1], [1, 1], [0, 1], [1, 1], [0, 1]], {
    lanes: 2, pageTokens: 1, laneTokenCapacity: 4, maxPages: 8, policy: 'paged',
  });
  const plan = kvPagePlanForLane(cache, 0, ['k']);
  assert.deepEqual(Array.from(plan.pageTable), [0, 2, 4, -1]);
  assert.deepEqual([0, 1, 2].map((position) => pagedSlot(plan, position)), [0, 2, 4]);

  const width = 2;
  const storage = Float32Array.from({ length: 16 }, (_, index) => index);
  const view = pagedSequenceView(storage, width, plan, 'k', defaultScratchAllocator());
  assert.equal(view.tier, 'gather');
  assert.notEqual(view.gather, null);
  /* The buffer is chosen now and filled later: the last token of the prefix is
   * produced by the same step that reads it. */
  assert.deepEqual(Array.from(view.storage), [0, 0, 0, 0, 0, 0]);
  view.gather();
  assert.deepEqual(Array.from(view.storage), [0, 1, 4, 5, 8, 9]);
});

test('gather storage grows logarithmically, not once per generated token', () => {
  const steps = 16;
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 1, laneTokenCapacity: steps, maxPages: steps * 2, policy: 'paged',
  });
  const storage = Float32Array.from({ length: steps * 4 }, (_, index) => index);
  const requested = [];
  const allocate = defaultScratchAllocator();
  const gatherBuffers = new Set();
  for (let step = 0; step < steps; step++) {
    cache.append(0, 1);
    cache.append(1, 1);
    const plan = kvPagePlanForLane(cache, 0, ['k']);
    const view = pagedSequenceView(storage, 2, plan, 'k', (key, elements, sample) => {
      requested.push(elements);
      return allocate(key, elements, sample);
    });
    if (!view.gather) continue;
    view.gather();
    gatherBuffers.add(view.storage.buffer);
  }
  /* The first step's single page is trivially its own identity map, so it
   * needs no gather and never reaches the allocator. Every later step does,
   * and each asks for two more elements than the last. */
  assert.equal(requested.length, steps - 1);
  assert.deepEqual(requested.slice(0, 3), [4, 6, 8]);
  /* Fifteen growing requests, capacities 4, 8, 16, 32: doubling is what keeps
   * a generation from allocating once per generated token. */
  assert.equal(gatherBuffers.size, 4);
});

test('an unmapped position is refused rather than addressed as slot zero', () => {
  const cache = new PagedKVCache({
    lanes: 1, pageTokens: 2, laneTokenCapacity: 8, policy: 'paged',
  });
  cache.append(0, 2);
  const plan = kvPagePlanForLane(cache, 0, ['k']);
  const stale = Object.freeze({ ...plan, kvLength: 6 });
  assert.equal(pagedSlot(stale, 4), -1);
  assert.throws(() => kvPageCopyRanges(stale), /not backed by a resident page/);
});
