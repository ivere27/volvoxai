import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
  batchPrefixSourceRows, decodeRowSet, rowSpanTier, rowSpanView, stageKeepMask,
  writeRowIndices,
} from '../ts/backends/decodeRowSet.js';
import { defaultScratchAllocator } from '../ts/backends/kvPageAddressing.js';

/*
 * The TypeScript half of the shared row-set addressing corpus.
 *
 * `native/tests/test_decode_row_set.c` reads the same file. The design document
 * flagged this gap directly: shape inference has a shared corpus and the
 * decode/row proofs had none, so a page table or a per-lane length that meant
 * something different in the two runtimes produced a decoder that worked in the
 * browser and not on the robot. B>1 decode is exactly that surface growing, so
 * the corpus comes with it rather than after it.
 *
 * The expectations are derived from the addressing contract, not recorded from a
 * run -- see the `note` field in the vectors.
 */

const VECTORS = JSON.parse(readFileSync(
  fileURLToPath(new URL('./decode_row_set_vectors.json', import.meta.url)), 'utf8'));

function planFor(vector, lane) {
  if (!vector.paged || vector.positions[lane] === -1) return null;
  const table = Int32Array.from(vector.page_tables[lane]);
  return Object.freeze({
    lane,
    kvLength: vector.positions[lane] + 1,
    pageTokens: vector.page_tokens,
    pageTable: table,
    pagesPerLane: vector.pages_per_lane,
    pagedTensors: new Set(['k']),
  });
}

/* The graph keep mask, read in its own declared layout. The row executor builds
 * the same accessors; this restates them against the corpus so the two runtimes
 * agree on what "[Q,K] selects each lane's own query row" means. */
function maskReader(vector, rowSet) {
  const mask = vector.mask;
  if (!mask) return null;
  const values = Int32Array.from(mask.values);
  const keys = mask.keys;
  const queries = mask.queries;
  const queryRow = (lane) => rowSet.positions[lane];
  switch (mask.layout) {
    case 'K': return (_lane, key) => values[key] !== 0;
    case 'BK': return (lane, key) => values[lane * keys + key] !== 0;
    case 'QK': return (lane, key) => values[queryRow(lane) * keys + key] !== 0;
    case 'BQK': return (lane, key) =>
      values[(lane * queries + queryRow(lane)) * keys + key] !== 0;
    default: throw new Error(`unknown mask layout '${mask.layout}'`);
  }
}

test('the shared row-set addressing corpus holds in TypeScript', () => {
  assert.equal(VECTORS.version, 'volvox-decode-row-set/v1');
  assert.ok(VECTORS.cases.length > 0);
  for (const vector of VECTORS.cases) {
    const label = vector.name;
    const rowSet = decodeRowSet(vector.positions.map((position, lane) => position === -1
      ? { parked: true }
      : { position, kvPages: planFor(vector, lane) }));
    const expect = vector.expect;

    assert.equal(rowSet.lanes, vector.lanes, `${label}: lanes`);
    assert.deepEqual(Array.from(rowSet.kvLengths), expect.kv_lengths, `${label}: kv_lengths`);
    assert.equal(rowSet.keyCapacity, expect.key_capacity, `${label}: key_capacity`);
    assert.equal(rowSet.unpaged, !vector.paged, `${label}: unpaged`);
    if (expect.live !== undefined) {
      assert.equal(rowSet.live, expect.live, `${label}: live`);
      assert.deepEqual(rowSet.parked.map(Number), expect.parked, `${label}: parked`);
    }

    const writeRows = writeRowIndices(rowSet, vector.lane_stride, vector.paged === true);
    assert.deepEqual(Array.from(writeRows), expect.write_rows, `${label}: write_rows`);
    assert.equal(rowSpanTier(writeRows), expect.tier, `${label}: tier`);

    const prefixRows = batchPrefixSourceRows(
      rowSet, vector.lane_stride, vector.paged === true);
    assert.deepEqual(Array.from(prefixRows), expect.prefix_rows, `${label}: prefix_rows`);

    /* A cross-attention memory publishes no per-lane length, so its keep mask
     * spans the whole memory and clamping is off. */
    const clamp = vector.clamp_to_length !== false;
    const keys = clamp ? rowSet.keyCapacity : vector.memory_keys;
    const keep = stageKeepMask(
      rowSet.lanes, keys, clamp ? rowSet.kvLengths : null,
      maskReader(vector, rowSet), label, defaultScratchAllocator(), new Int32Array(1));
    keep.apply();
    assert.deepEqual(Array.from(keep.storage), expect.keep_mask, `${label}: keep_mask`);
  }
});

test('a keep mask is materialised even when the graph carries none', () => {
  /* Until the ABI carries `kv_len`, the keep mask is the only place a lane's
   * length can reach the kernel. A step that skipped the mask because the graph
   * had none would let every lane attend over the longest lane's prefix. */
  const rowSet = decodeRowSet([{ position: 3 }, { position: 1 }]);
  const keep = stageKeepMask(
    2, rowSet.keyCapacity, rowSet.kvLengths, null, 'none',
    defaultScratchAllocator(), new Int32Array(1));
  keep.apply();
  assert.deepEqual(Array.from(keep.storage), [1, 1, 1, 1, 1, 1, 0, 0]);
});

test('a mixed paged and unpaged step is refused', () => {
  /* A tensor is paged for the whole step or for none of it. One lane addressing
   * the same storage through a page table while another addresses it linearly
   * writes two lanes under two different meanings. */
  const plan = Object.freeze({
    lane: 0, kvLength: 2, pageTokens: 1,
    pageTable: Int32Array.from([0, 1]), pagesPerLane: 2,
    pagedTensors: new Set(['k']),
  });
  assert.throws(() => decodeRowSet([{ position: 1, kvPages: plan }, { position: 1 }]),
    /mixes 1 paged lane\(s\) with 1 unpaged/);
});

test('lanes that claim different paged tensor sets are refused', () => {
  const table = Int32Array.from([0, 1]);
  const make = (lane, names) => Object.freeze({
    lane, kvLength: 2, pageTokens: 1, pageTable: table, pagesPerLane: 2,
    pagedTensors: new Set(names),
  });
  assert.throws(() => decodeRowSet([
    { position: 1, kvPages: make(0, ['k', 'v']) },
    { position: 1, kvPages: make(1, ['k']) },
  ]), /lane 1 claims a different paged tensor set than lane 0/);
});

test('a parked lane occupies a row and writes nothing', () => {
  /* The distinction that makes a released slot expressible. An *idle* lane
   * still holds its request and repeats its own last row; a *parked* lane holds
   * none, has no row to repeat, and -- once its pages went back to the pool --
   * nowhere to write one. So it occupies a dense row for shape and its output
   * is discarded. */
  const rowSet = decodeRowSet([{ position: 3 }, { parked: true }]);
  assert.equal(rowSet.lanes, 2);
  assert.equal(rowSet.live, 1);
  assert.deepEqual([...rowSet.parked], [false, true]);
  assert.deepEqual(Array.from(rowSet.kvLengths), [4, 0],
    'a parked lane publishes no length');
  assert.equal(rowSet.keyCapacity, 4, 'a parked lane does not widen the operand');
});

test('a parked lane is skipped by the scatter and never aliases', () => {
  /* Both halves of what "parked" costs. The staged tier is forced even when the
   * rows happen to be consecutive, because an identity span aliases the source
   * and the kernel would write the parked row directly -- leaving no scatter to
   * skip. */
  const storage = Float32Array.from({ length: 12 }, (_, index) => index);
  const before = Array.from(storage);
  const rows = Int32Array.from([0, 1]);
  const span = rowSpanView(
    storage, 3, rows, 'write', 'parked', defaultScratchAllocator(), [false, true]);
  assert.equal(span.tier, 'gather', 'a parked lane must not take the aliasing tier');
  span.storage.set([90, 91, 92, 80, 81, 82]);
  span.apply();
  assert.deepEqual(Array.from(storage.subarray(0, 3)), [90, 91, 92],
    'the live lane is written back');
  assert.deepEqual(Array.from(storage.subarray(3)), before.slice(3),
    'the parked lane leaves every other byte untouched');
});

test('a step with no live lane is refused', () => {
  assert.throws(() => decodeRowSet([{ parked: true }, { parked: true }]),
    /needs at least one lane that is not parked/);
});

test('parked lanes do not disturb the paged uniformity rule', () => {
  /* A parked lane addresses nothing, so it cannot address it under a second
   * meaning -- the rule is about lanes that actually read and write the pool. */
  const plan = Object.freeze({
    lane: 0, kvLength: 2, pageTokens: 1,
    pageTable: Int32Array.from([0, 1]), pagesPerLane: 2,
    pagedTensors: new Set(['k']),
  });
  const rowSet = decodeRowSet([{ position: 1, kvPages: plan }, { parked: true }]);
  assert.equal(rowSet.unpaged, false);
  assert.deepEqual([...rowSet.pagedTensors], ['k']);
  /* But a genuinely mixed step is still refused. */
  const second = Object.freeze({ ...plan, lane: 1 });
  assert.throws(() => decodeRowSet([
    { position: 1, kvPages: second }, { position: 1 }, { parked: true },
  ]), /mixes 1 paged lane\(s\) with 1 unpaged/);
});
