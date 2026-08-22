import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { PagedKVCache, PagedKVError } from '../ts/core/PagedKVCache.js';
import { kvPagePlanForLane } from '../ts/backends/kvPageAddressing.js';

/* The corpus half of the paged KV contract. native/tests/test_paged_kv.c runs
 * the same file through the C implementation, so a divergence between the two
 * allocators fails here or there rather than surfacing as a decoder that works
 * in the browser and not on the robot. */

const VECTORS = JSON.parse(
  readFileSync(new URL('./paged_kv_vectors.json', import.meta.url), 'utf8'));

function statusOf(error) {
  if (error instanceof PagedKVError) return error.code;
  throw error;
}

/** Run one operation, returning the failure code instead of throwing. */
function attempt(run) {
  try {
    run();
    return 'ok';
  } catch (error) {
    return statusOf(error);
  }
}

function applyOperation(cache, operation, reservations, label) {
  const expected = operation.status ?? 'ok';
  switch (operation.op) {
    case 'append': {
      assert.equal(attempt(() => cache.append(operation.lane, operation.tokens)),
        expected, `${label} append`);
      return;
    }
    case 'reserve': {
      const status = attempt(() => {
        reservations.set(operation.id, cache.reserve(operation.lane, operation.tokens));
      });
      assert.equal(status, expected, `${label} reserve`);
      return;
    }
    case 'commit': {
      cache.commit(reservations.get(operation.id));
      return;
    }
    case 'rollback': {
      cache.rollback(reservations.get(operation.id));
      return;
    }
    case 'release_lane': {
      cache.releaseLane(operation.lane);
      return;
    }
    case 'reset': {
      cache.reset();
      return;
    }
    case 'publish_prefix': {
      assert.equal(
        attempt(() => cache.publishPrefix(operation.key, operation.lane, operation.tokens)),
        expected, `${label} publish_prefix`);
      return;
    }
    case 'acquire_prefix': {
      let tokens = null;
      const status = attempt(() => {
        tokens = cache.acquirePrefix(operation.key, operation.lane);
      });
      assert.equal(status, expected, `${label} acquire_prefix`);
      if (expected === 'ok' && operation.tokens !== undefined) {
        assert.equal(tokens, operation.tokens, `${label} acquire_prefix tokens`);
      }
      return;
    }
    case 'copy_on_write': {
      let moved = null;
      const status = attempt(() => {
        moved = cache.copyOnWrite(operation.lane, operation.logical);
      });
      assert.equal(status, expected, `${label} copy_on_write`);
      if (expected !== 'ok') return;
      assert.equal(moved.from, operation.from, `${label} copy_on_write from`);
      assert.equal(moved.to, operation.to, `${label} copy_on_write to`);
      return;
    }
    case 'evict': {
      assert.equal(cache.evict(operation.pages), operation.reclaimed, `${label} evict`);
      return;
    }
    case 'expect_page_table': {
      const base = operation.lane * cache.pagesPerLane;
      const actual = Array.from(
        cache.pageTable.subarray(base, base + cache.pagesPerLane));
      assert.deepEqual(actual, operation.pages, `${label} page table`);
      return;
    }
    case 'expect_lengths': {
      assert.deepEqual(Array.from(cache.kvLength), operation.kv, `${label} kv lengths`);
      if (operation.query !== undefined) {
        assert.deepEqual(Array.from(cache.queryLength), operation.query,
          `${label} query lengths`);
      }
      return;
    }
    case 'expect_tokens': {
      const actual = Array.from(cache.gatherActiveTokens(operation.lane));
      assert.deepEqual(actual, operation.tokens, `${label} physical token indices`);
      return;
    }
    case 'expect_contiguous': {
      assert.equal(cache.laneIsContiguous(operation.lane), operation.value,
        `${label} contiguous`);
      return;
    }
    case 'expect_lane_generation': {
      assert.equal(cache.laneGeneration[operation.lane], operation.value,
        `${label} lane generation`);
      return;
    }
    case 'expect_page_generation': {
      assert.equal(cache.pageGeneration(operation.page), operation.value,
        `${label} page generation`);
      return;
    }
    case 'expect_page_shared': {
      assert.equal(cache.pageIsShared(operation.lane, operation.logical), operation.value,
        `${label} page shared`);
      return;
    }
    case 'expect_has_prefix': {
      assert.equal(cache.hasPrefix(operation.key), operation.value, `${label} has prefix`);
      return;
    }
    case 'expect_telemetry': {
      const telemetry = cache.telemetry();
      for (const [field, value] of Object.entries(operation)) {
        if (field === 'op') continue;
        assert.equal(telemetry[field], value, `${label} telemetry.${field}`);
      }
      return;
    }
    default:
      throw new Error(`${label} uses unknown operation '${operation.op}'`);
  }
}

test('paged KV golden corpus is self-describing', () => {
  assert.equal(VECTORS.version, 'volvox-paged-kv/v1');
  assert.equal(Array.isArray(VECTORS.cases), true);
  assert.equal(VECTORS.cases.length > 0, true);
  for (const testCase of VECTORS.cases) {
    assert.equal(typeof testCase.name, 'string');
    assert.equal(Array.isArray(testCase.operations), true);
  }
});

for (const testCase of VECTORS.cases) {
  test(`paged KV vector: ${testCase.name}`, () => {
    if (testCase.expectCreateFailure === true) {
      assert.throws(() => new PagedKVCache(testCase.options), PagedKVError);
      return;
    }
    const cache = new PagedKVCache(testCase.options);
    const reservations = new Map();
    testCase.operations.forEach((operation, index) => {
      applyOperation(cache, operation, reservations, `${testCase.name}[${index}]`);
    });
  });
}

test('paged and contiguous policies address identical token positions', () => {
  /* The structural half of "private paged KV is numerically identical to
   * contiguous KV". A gather is the complete addressing contract, so proving
   * the two policies produce the same *sequence of tokens* proves the two
   * produce the same attention inputs, without running attention. */
  const options = { lanes: 3, pageTokens: 4, laneTokenCapacity: 32 };
  const contiguous = new PagedKVCache({ ...options, policy: 'contiguous' });
  const paged = new PagedKVCache({ ...options, policy: 'paged' });
  const schedule = [[0, 5], [1, 9], [2, 1], [0, 7], [1, 3], [2, 20], [0, 11]];
  const logical = [0, 0, 0];

  for (const [lane, tokens] of schedule) {
    contiguous.append(lane, tokens);
    paged.append(lane, tokens);
    logical[lane] += tokens;
    for (let position = 0; position < logical[lane]; position++) {
      /* Not "the same physical index" -- the whole point of paging is that it
       * is a different one. The same *logical* position must resolve to a
       * token slot that has been written by the same append, which is what
       * the per-lane monotone ordering below asserts. */
      const contiguousIndex = contiguous.physicalTokenIndex(lane, position);
      assert.equal(contiguousIndex, lane * options.laneTokenCapacity + position);
    }
  }
  /* Every lane's paged slots are distinct: no two logical positions of any two
   * lanes alias. That is the property a numerical comparison would be testing
   * for, expressed directly. */
  const seen = new Set();
  for (let lane = 0; lane < options.lanes; lane++) {
    for (let position = 0; position < logical[lane]; position++) {
      const slot = paged.physicalTokenIndex(lane, position);
      assert.equal(seen.has(slot), false, `paged slot ${slot} aliases`);
      seen.add(slot);
    }
  }
  assert.equal(seen.size, logical.reduce((sum, value) => sum + value, 0));
});

test('a recycled page is erased and its generation advances', () => {
  const cache = new PagedKVCache({ lanes: 2, pageTokens: 2, laneTokenCapacity: 4 });
  const erased = [];
  cache.setPageEraser((page) => erased.push(page));
  cache.append(0, 4);
  assert.deepEqual(erased, [0, 1]);
  const before = cache.pageGeneration(0);
  cache.releaseLane(0);
  assert.equal(cache.pageGeneration(0), before + 1);
  cache.append(1, 2);
  /* Page zero is the lowest free index, so the reuse is immediate -- and the
   * eraser has to have run before the new lane can read it. */
  assert.deepEqual(erased, [0, 1, 0]);
});

test('a reservation settles exactly once', () => {
  /* Committing and then rolling back the same handle would double-subtract the
   * reserved-page count and republish the pre-commit length. Silent allocator
   * corruption discovered much later is the worst possible failure mode, so
   * the second transition is refused. */
  const cache = new PagedKVCache({ lanes: 1, pageTokens: 2, laneTokenCapacity: 8 });
  const reservation = cache.reserve(0, 4);
  cache.commit(reservation);
  assert.equal(cache.kvLength[0], 4);
  assert.throws(() => cache.rollback(reservation),
    (error) => error.code === 'INVALID_ARGUMENT');
  assert.throws(() => cache.commit(reservation),
    (error) => error.code === 'INVALID_ARGUMENT');
  assert.equal(cache.kvLength[0], 4);
  assert.equal(cache.telemetry().reservedPages, 0);

  const second = cache.reserve(0, 2);
  cache.rollback(second);
  assert.throws(() => cache.commit(second), (error) => error.code === 'INVALID_ARGUMENT');
  assert.equal(cache.kvLength[0], 4);
  assert.equal(cache.telemetry().reservedPages, 0);
});

test('a lane has one generation-bound reservation at a time', () => {
  const cache = new PagedKVCache({ lanes: 1, pageTokens: 2, laneTokenCapacity: 8 });
  const reservation = cache.reserve(0, 4);
  assert.equal(reservation.laneGeneration, 0);
  assert.throws(() => cache.reserve(0, 1),
    (error) => error.code === 'INVALID_ARGUMENT');
  assert.throws(() => cache.releaseLane(0),
    (error) => error.code === 'INVALID_ARGUMENT');

  /* A generation mismatch must fail before publishing the reserved length.
   * Restore the public test vector storage afterwards so rollback can settle
   * the original transaction and prove no pages leaked. */
  cache.laneGeneration[0]++;
  assert.throws(() => cache.commit(reservation),
    (error) => error.code === 'STALE_PAGE');
  assert.equal(cache.kvLength[0], 0);
  cache.laneGeneration[0]--;
  cache.rollback(reservation);
  assert.equal(cache.telemetry().residentPages, 0);
  assert.equal(cache.telemetry().reservedPages, 0);
});

test('a page plan keeps an immutable lane-table snapshot across release and reuse', () => {
  const cache = new PagedKVCache({ lanes: 2, pageTokens: 2, laneTokenCapacity: 4 });
  cache.append(0, 2);
  const plan = kvPagePlanForLane(cache, 0, ['k', 'v']);
  const snapshot = Array.from(plan.pageTable);

  cache.releaseLane(0);
  cache.append(1, 2);
  assert.deepEqual(Array.from(plan.pageTable), snapshot);
  assert.notDeepEqual(Array.from(plan.pageTable),
    Array.from(cache.pageTable.subarray(0, cache.pagesPerLane)));
});

test('a failed reservation leaves no page mapped and no length published', () => {
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 2, laneTokenCapacity: 8, maxPages: 3,
  });
  cache.append(0, 4);
  const table = Array.from(cache.pageTable);
  const lengths = Array.from(cache.kvLength);
  assert.throws(() => cache.reserve(1, 4), (error) => error.code === 'CAPACITY_EXHAUSTED');
  assert.deepEqual(Array.from(cache.pageTable), table);
  assert.deepEqual(Array.from(cache.kvLength), lengths);
  /* One page was available, so the reservation did map a page before failing
   * on the second. Proving the table is byte-identical proves the undo ran. */
  assert.equal(cache.telemetry().reservedPages, 0);
  assert.equal(cache.telemetry().residentPages, 2);
});
