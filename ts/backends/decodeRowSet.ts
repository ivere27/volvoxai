/* The set of rows one decode step writes.
 *
 * Row execution was built around a single scalar `position`: one logical
 * sequence, one row, and the row is one contiguous span at `position * width`.
 * B>1 decode breaks exactly that invariant. With `lanes` sequences retained in
 * one `[B,S,W]` activation, the rows a step writes are `b * S + position[b]`
 * for each lane, and lanes at different positions make that a *set* of spans,
 * not one span. A paged K/V tensor is worse: its rows are page-table slots and
 * bear no relation to each other at all.
 *
 * So the scalar becomes a `DecodeRowSet` and the span becomes a `RowSpan` with
 * two tiers, the same shape `kvPageAddressing.ts` already uses:
 *
 *   identity   The row indices are consecutive and ascending, so the span is
 *              one subarray of the source with no copy. One lane always lands
 *              here, which is why B=1 decode keeps the exact zero-copy views it
 *              had before this file existed -- B=1 is not a preserved second
 *              path, it is this path's cheapest tier.
 *
 *   gather     Anything else: the lanes' rows are staged into a dense
 *              `[lanes, 1, W]` buffer in lane order, the kernel runs once over
 *              all of them, and an output span is written back afterwards. The
 *              staging is `lanes * W` elements against a kernel that does
 *              `lanes * W * W_out` work, and it is what buys one dispatch for
 *              the whole batch instead of one per lane.
 *
 * Lane lengths never enter a shape. `kvLengths` is `I32[B]` read as a value,
 * and the padded key extent every lane's operand shares is `max(kvLengths)`,
 * with the per-lane length folded into the keep mask. That costs a lane whose
 * prefix is short the arithmetic of the longest lane; removing that cost needs
 * direct `q_len`/`kv_len` kernel inputs.
 */

import { pagedSlot } from './kvPageAddressing.js';
import type { KVPagePlan, KVScratchAllocator } from './kvPageAddressing.js';
import type { RuntimeTypedArray } from '../types.js';

/** One lane's contribution to a step. */
export interface DecodeLaneStep {
  /** Logical token position this lane writes. Absent for a parked lane. */
  readonly position?: number;
  /** The lane's page table, or null for an unpaged contiguous lane. */
  readonly kvPages?: KVPagePlan | null;
  /**
   * This lane holds no request; its output is discarded.
   *
   * A dense `[B,S,D]` batch cannot drop a lane without changing every operand's
   * shape, so a slot whose request retired still occupies a row. It needs no
   * page and no length: the kernel computes its row because the shape says
   * `lanes` rows exist, and the scatter simply does not write it back. Nothing
   * in the pool is touched, which is what makes a *released* lane -- whose
   * pages went back to the free list -- expressible at all.
   *
   * Distinct from an *idle* lane, which still holds its request and repeats its
   * own last row so that row stays bit-identical. A parked lane has no row to
   * repeat.
   */
  readonly parked?: boolean;
}

export interface DecodeRowSet {
  /**
   * Dense lane count, declared by the context that owns the slots.
   *
   * Never inferred from an operand shape: a scheduler slot exists whether or
   * not a tensor happens to be spelled with that leading extent, and inferring
   * it means a graph whose activations are `[1,S,D]` silently decodes one lane
   * of a two-lane batch.
   */
  readonly lanes: number;
  /** `I32[B]` logical write position per lane; zero for a parked lane. */
  readonly positions: Int32Array;
  /** Which lanes hold no request. Their staged output rows are discarded. */
  readonly parked: readonly boolean[];
  /** Lanes that actually advance. At least one, or the step is not a step. */
  readonly live: number;
  /**
   * `I32[B]` visible key/value length per lane. A value, never a shape.
   *
   * Causal decode appends one K/V row per token, so this is `position + 1` --
   * derived rather than accepted, because a length the row path did not produce
   * would make attention read a prefix nobody wrote. It is still materialised
   * as its own array: stage two hands exactly this to the kernel as `kv_len`.
   */
  readonly kvLengths: Int32Array;
  /** Per-lane page table; entry null when every lane is unpaged. */
  readonly pages: readonly (KVPagePlan | null)[];
  /** True when no lane is paged, which is the contiguous allocation policy. */
  readonly unpaged: boolean;
  /** The padded key extent every lane's attention operand shares. */
  readonly keyCapacity: number;
  /**
   * The activation tensors that live in the page pool, shared by every lane.
   *
   * One set for the whole step, not one per lane: a tensor is paged or it is
   * not, and a step where one lane addressed a tensor through a page table
   * while another addressed it linearly would write two lanes of the same
   * storage under two different meanings.
   */
  readonly pagedTensors: ReadonlySet<string>;
}

export type RowSpanTier = 'identity' | 'gather';

/**
 * A step's rows of one tensor, presented as `lanes` consecutive rows.
 *
 * `apply` is separated from the view for the reason `KVPageView.gather` is: a
 * row plan is built for the whole step before any node runs, but an input row's
 * bytes are produced *during* the step. A copy performed at plan time would
 * read the previous step's values.
 */
export interface RowSpan {
  readonly storage: RuntimeTypedArray;
  readonly tier: RowSpanTier;
  /** Copies source rows in ('read') or staged rows out ('write'); null when the view aliases. */
  readonly apply: (() => void) | null;
}

/**
 * What kind of span an operand presents, in the four shapes decode produces.
 *
 *   row        One row per lane, selected at that lane's own position. Every
 *              ordinary activation, and the query of an attention node.
 *   prefix     `lanes * keyCapacity` rows in lane-major logical order: the
 *              retained causal K/V of a multi-lane step.
 *   window     A growing prefix of the source that always begins at element
 *              zero. The one-lane causal K/V and a `[K]` keep mask are this:
 *              the address never moves and only the visible length changes,
 *              which is why a backend binding whole buffers can execute them
 *              without staging anything.
 *   keepMask   `[lanes, keyCapacity]` built by the host from the step's lane
 *              lengths and the graph's mask. It is a *value the step computes*,
 *              not a region of a tensor, so no source row can name it.
 */
export type RowSpanKind = 'row' | 'prefix' | 'window' | 'keepMask';

/**
 * How an operand of a row step is addressed, as data rather than as code.
 *
 * A host runtime never needs this: it gets typed-array views and the addressing
 * has already happened. A *device* runtime does, because it holds one compiled
 * binding across every step and has to re-issue the copies each time the
 * positions move. WebGPU used to answer that by back-deriving a stride from a
 * one-position sample and assuming `address = position * stride` forever, which
 * a page table and a second lane both falsify.
 *
 * Publishing the geometry instead means the three runtimes resolve rows through
 * the same `writeRowIndices` / `batchPrefixSourceRows` the shared corpus checks
 * against the native twin, rather than each carrying its own arithmetic.
 */
export interface RowSpanGeometry {
  readonly kind: RowSpanKind;
  /** Elements per row. Bytes are this times the operand's dtype width. */
  readonly width: number;
  /**
   * Rows between one lane and the next in the *source* tensor. Zero for an
   * operand that broadcasts one sequence across every lane, and meaningless for
   * a paged one, whose rows are page-table slots.
   */
  readonly laneStride: number;
  readonly paged: boolean;
  readonly direction: 'read' | 'write';
}

export function rowSpanGeometry(
  kind: RowSpanKind, width: number, laneStride: number,
  paged: boolean, direction: 'read' | 'write',
): RowSpanGeometry {
  return Object.freeze({ kind, width, laneStride, paged, direction });
}

/**
 * The source rows this geometry names for `rowSet`, or null when the operand
 * has no source rows to name.
 *
 * The single entry point a device runtime needs: it turns "which operand is
 * this?" into "which rows does it read or write *this* step?" without the
 * caller reconstructing either mapping. `-1` entries are padding beyond a
 * lane's own length and belong to nothing.
 */
export function rowSpanSourceRows(
  rowSet: DecodeRowSet, geometry: RowSpanGeometry,
): Int32Array | null {
  if (geometry.kind === 'window' || geometry.kind === 'keepMask') return null;
  return geometry.kind === 'prefix'
    ? batchPrefixSourceRows(rowSet, geometry.laneStride, geometry.paged)
    : writeRowIndices(rowSet, geometry.laneStride, geometry.paged);
}

/**
 * `rows` collapsed into the fewest contiguous source runs that reproduce it.
 *
 * A staged span is `rows.length` copies in the worst case and one in the best,
 * and which it is depends on the step, not on the plan: two lanes at adjacent
 * positions of a shared page are one run, and the same two lanes a step later
 * may not be. Merging here rather than at each call site is what keeps "how
 * scattered is this step, really?" a number a test can assert -- `kvPageCopyRanges`
 * answers the same question for a single lane's page table.
 *
 * `-1` rows are padding: they end the current run and produce no copy, which is
 * what leaves the destination available for a separate zeroing pass.
 */
export function rowSpanCopyRuns(
  rows: Int32Array,
): Array<{ source: number; destination: number; rows: number }> {
  const runs: Array<{ source: number; destination: number; rows: number }> = [];
  for (let index = 0; index < rows.length; index++) {
    const source = rows[index];
    if (source < 0) continue;
    const last = runs[runs.length - 1];
    if (last && last.source + last.rows === source &&
        last.destination + last.rows === index) {
      last.rows++;
      continue;
    }
    runs.push({ source, destination: index, rows: 1 });
  }
  return runs;
}

/**
 * The staged destination rows no source fills, as runs.
 *
 * The complement of `rowSpanCopyRuns` over the same array. A backend that
 * stages into a reused buffer has to clear these: the keep mask already stops a
 * kernel from reading a lane's padding, but padding still holding *another
 * request's* bytes turns a masking mistake from wrong arithmetic into a
 * cross-request leak. Returned as runs so the clear costs one call per gap.
 */
export function rowSpanPaddingRuns(
  rows: Int32Array,
): Array<{ destination: number; rows: number }> {
  const runs: Array<{ destination: number; rows: number }> = [];
  for (let index = 0; index < rows.length; index++) {
    if (rows[index] >= 0) continue;
    const last = runs[runs.length - 1];
    if (last && last.destination + last.rows === index) {
      last.rows++;
      continue;
    }
    runs.push({ destination: index, rows: 1 });
  }
  return runs;
}

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`Decode row set ${label} must be a positive integer.`);
  }
  return value as number;
}

/**
 * Build the row set for a step from its lanes.
 *
 * A one-element list is the scalar `position` spelling, so the scalar caller
 * and the batched caller share one implementation and cannot drift.
 */
export function decodeRowSet(lanes: readonly DecodeLaneStep[]): DecodeRowSet {
  if (!Array.isArray(lanes) || lanes.length === 0) {
    throw new Error('Decode row set requires at least one lane.');
  }
  const count = lanes.length;
  const positions = new Int32Array(count);
  const kvLengths = new Int32Array(count);
  const pages: (KVPagePlan | null)[] = [];
  const parked: boolean[] = [];
  let keyCapacity = 0;
  let unpaged = true;
  let live = 0;
  for (let lane = 0; lane < count; lane++) {
    const step = lanes[lane];
    if (step?.parked === true) {
      /* Row zero is a placeholder, not a destination: the scatter skips a
       * parked lane, so nothing is ever written there. It is chosen because it
       * is always in bounds and always initialised, which keeps the lane's
       * staged *input* row readable without a page. */
      positions[lane] = 0;
      kvLengths[lane] = 0;
      pages.push(null);
      parked.push(true);
      continue;
    }
    const position = step?.position;
    if (!Number.isSafeInteger(position) || (position as number) < 0) {
      throw new Error(`Decode row set lane ${lane} needs a non-negative integer position.`);
    }
    parked.push(false);
    live++;
    const plan = step?.kvPages ?? null;
    /* The lane's visible length is its position plus one, and a page plan that
     * publishes anything else is a caller bug, not a length to honour: the
     * gather would read a slot this step never wrote. Checked here so every
     * backend inherits the check instead of restating it. */
    const length = (position as number) + 1;
    if (plan && plan.kvLength !== length) {
      throw new Error(
        `Decode row set lane ${lane} page plan publishes K/V length ${plan.kvLength} ` +
        `at position ${position}.`);
    }
    positions[lane] = position as number;
    kvLengths[lane] = length;
    pages.push(plan);
    if (plan) unpaged = false;
    if (length > keyCapacity) keyCapacity = length;
  }
  if (live === 0) {
    throw new Error('Decode row set needs at least one lane that is not parked.');
  }
  const paged = pages.filter((plan): plan is KVPagePlan => plan !== null);
  /* Parked lanes are excluded from the uniformity check: they address nothing,
   * so they cannot address it under a second meaning. The rule the check exists
   * for -- a tensor is paged for the whole step or for none of it -- is about
   * the lanes that actually read and write the pool. */
  if (paged.length !== 0 && paged.length !== live) {
    throw new Error(
      `Decode row set mixes ${paged.length} paged lane(s) with ${live - paged.length} ` +
      'unpaged; a tensor is paged for the whole step or for none of it.');
  }
  const pagedTensors: ReadonlySet<string> = paged.length === 0
    ? new Set<string>()
    : paged[0].pagedTensors;
  for (const plan of paged) {
    if (plan.pagedTensors.size !== pagedTensors.size ||
        [...plan.pagedTensors].some((name) => !pagedTensors.has(name))) {
      throw new Error(
        `Decode row set lane ${plan.lane} claims a different paged tensor set than lane 0.`);
    }
  }
  return Object.freeze({
    lanes: count,
    positions,
    kvLengths,
    parked: Object.freeze(parked),
    live,
    pages: Object.freeze(pages),
    unpaged,
    keyCapacity,
    pagedTensors,
  });
}

/** The single-lane row set, which is what the scalar row position means. */
export function singleLaneRowSet(
  position: number, kvPages: KVPagePlan | null = null,
): DecodeRowSet {
  return decodeRowSet([{ position, kvPages }]);
}

/**
 * Physical row index of lane `laneIndex`'s token `token`.
 *
 * `laneStride` is the distance in rows between two lanes of the source tensor:
 * the sequence extent for a `[B,S,W]` activation, and zero for one that carries
 * no batch axis or broadcasts across it. A paged tensor ignores it entirely --
 * its rows are page-table slots in a shared pool, which is the whole point.
 */
export function laneRowIndex(
  rowSet: DecodeRowSet, laneIndex: number, token: number, laneStride: number, paged: boolean,
): number {
  const plan = paged ? rowSet.pages[laneIndex] : null;
  if (!plan) return laneIndex * laneStride + token;
  const slot = pagedSlot(plan, token);
  if (slot < 0) {
    throw new Error(
      `Paged K/V lane ${plan.lane} has no resident page for position ${token}.`);
  }
  return slot;
}

/**
 * The row each lane writes this step, in lane order.
 *
 * `paged` is per tensor, not per lane: a lane owns a page table but only the
 * tensors in `pagedTensors` are addressed through it, and applying it to an
 * ordinary activation would send that activation's row to a K/V slot.
 */
export function writeRowIndices(
  rowSet: DecodeRowSet, laneStride: number, paged: boolean,
): Int32Array {
  const rows = new Int32Array(rowSet.lanes);
  for (let lane = 0; lane < rowSet.lanes; lane++) {
    rows[lane] = laneRowIndex(rowSet, lane, rowSet.positions[lane], laneStride, paged);
  }
  return rows;
}

function consecutive(rows: Int32Array): boolean {
  for (let index = 1; index < rows.length; index++) {
    if (rows[index] !== rows[index - 1] + 1) return false;
  }
  return rows[0] >= 0;
}

/**
 * The tier a set of row indices falls into, without allocating anything.
 *
 * Exported so a caller can ask "did batching change how this operand is read?"
 * and so the shared golden corpus can assert the answer against the native
 * twin. A staged run that silently qualified as `identity` would pass a naive
 * batched test while only ever decoding lane zero's rows.
 */
export function rowSpanTier(rows: Int32Array): RowSpanTier {
  return consecutive(rows) ? 'identity' : 'gather';
}

/**
 * One source row per staged row of a `[lanes, keyCapacity, W]` prefix operand,
 * in lane-major logical order, with -1 for a lane's padding beyond its own
 * length.
 *
 * Separated from `batchPrefixView` so the mapping is a value the shared corpus
 * can compare across runtimes: the copy needs storage, the addressing does not.
 */
export function batchPrefixSourceRows(
  rowSet: DecodeRowSet, laneStride: number, paged: boolean,
): Int32Array {
  const { lanes, keyCapacity } = rowSet;
  const sources = new Int32Array(lanes * keyCapacity).fill(-1);
  for (let lane = 0; lane < lanes; lane++) {
    const length = rowSet.kvLengths[lane];
    for (let token = 0; token < length; token++) {
      sources[lane * keyCapacity + token] =
        laneRowIndex(rowSet, lane, token, laneStride, paged);
    }
  }
  return sources;
}

/**
 * `rows.length` rows of `storage` as one dense `rows.length * width` span.
 *
 * `direction` decides which copy the caller gets and is not a convenience: an
 * operand must be filled before the kernel runs and an output must be written
 * back after it, and a span that offered both would let a caller stage an
 * output's stale contents over the value the kernel just produced.
 */
export function rowSpanView(
  storage: RuntimeTypedArray,
  width: number,
  rows: Int32Array,
  direction: 'read' | 'write',
  key: string,
  allocate: KVScratchAllocator,
  /**
   * Lanes whose staged output must not be written back.
   *
   * A parked lane occupies a dense row so the operand keeps its shape, and its
   * result is discarded. Passing the mask also forces the staged tier: an
   * identity span aliases the source, so a parked lane whose row happened to
   * fall next to a live one would be written by the kernel directly, with no
   * scatter left to skip.
   */
  parked: readonly boolean[] | null = null,
): RowSpan {
  positiveInteger(width, 'row width');
  const count = rows.length;
  for (let lane = 0; lane < count; lane++) {
    const start = rows[lane] * width;
    if (rows[lane] < 0 || start + width > storage.length) {
      throw new Error(
        `Decode row ${rows[lane]} of width ${width} is outside ${storage.length} elements.`);
    }
  }
  const hasParked = parked !== null && parked.some((value) => value);
  if (!hasParked && consecutive(rows)) {
    const start = rows[0] * width;
    return {
      storage: storage.subarray(start, start + count * width) as RuntimeTypedArray,
      tier: 'identity',
      apply: null,
    };
  }
  const staged = allocate(key, count * width, storage);
  if (staged.length !== count * width) {
    throw new Error(
      `Decode row span '${key}' needs ${count * width} elements but received ${staged.length}.`);
  }
  /* Indices are resolved now and copied later. The mapping is fixed for the
   * whole step; the bytes are not. */
  const source = Int32Array.from(rows, (row) => row * width);
  const apply = direction === 'read'
    ? () => {
        for (let lane = 0; lane < count; lane++) {
          staged.set(
            storage.subarray(source[lane], source[lane] + width) as any, lane * width);
        }
      }
    : () => {
        for (let lane = 0; lane < count; lane++) {
          /* The whole of what "parked" costs: the kernel computed this row
           * because the shape said it existed, and nothing writes it back. */
          if (parked !== null && parked[lane]) continue;
          storage.set(
            staged.subarray(lane * width, (lane + 1) * width) as any, source[lane]);
        }
      };
  return { storage: staged, tier: 'gather', apply };
}

export interface BatchPrefixView {
  readonly storage: RuntimeTypedArray;
  /** Padded key extent: `max(kvLengths)`. The shape every lane shares. */
  readonly keyCapacity: number;
  readonly tier: RowSpanTier;
  readonly apply: (() => void) | null;
}

/**
 * Every lane's visible K/V prefix as one dense `[lanes, keyCapacity, width]`
 * operand.
 *
 * The padding beyond a lane's own length is zeroed rather than left as whatever
 * the pool holds. The keep mask already stops a kernel from reading it, so this
 * is the second of two defences and not the load-bearing one -- but a padded
 * tail carrying another request's bytes is the failure that a masking mistake
 * turns from wrong arithmetic into a cross-request leak.
 */
export function batchPrefixView(
  storage: RuntimeTypedArray,
  width: number,
  rowSet: DecodeRowSet,
  laneStride: number,
  paged: boolean,
  key: string,
  allocate: KVScratchAllocator,
): BatchPrefixView {
  const { lanes, keyCapacity } = rowSet;
  const staged = allocate(key, lanes * keyCapacity * width, storage);
  if (staged.length !== lanes * keyCapacity * width) {
    throw new Error(`Decode K/V staging '${key}' needs ${lanes * keyCapacity * width} ` +
      `elements but received ${staged.length}.`);
  }
  /* One resolved source row per staged row, built now for the same reason
   * rowSpanView resolves its indices now: the mapping is what the page table
   * says this step, and the bytes arrive later. */
  const sources = batchPrefixSourceRows(rowSet, laneStride, paged);
  const apply = () => {
    for (let row = 0; row < sources.length; row++) {
      const destination = row * width;
      const source = sources[row];
      if (source < 0) {
        staged.fill(0, destination, destination + width);
        continue;
      }
      staged.set(
        storage.subarray(source * width, source * width + width) as any, destination);
    }
  };
  return { storage: staged, keyCapacity, tier: 'gather', apply };
}

/**
 * `mask(lane, key)` for the graph's keep mask, in its own declared layout.
 *
 * Returns null when the node carries no mask, which reads as "keep everything".
 */
export type KeepMaskReader = (lane: number, key: number) => boolean;

/**
 * The `[lanes, keys]` keep mask, optionally with each lane's active length
 * baked in.
 *
 * This function implements the current dense-row active-length contract. The
 * kernel ABI already carries a `[B,K]` I32 keep mask on every backend, so
 * publishing per-lane lengths through it costs nothing in contract terms: a key
 * beyond lane `b`'s length is simply not kept. Every kernel here skips a masked
 * key outright rather than scoring it and multiplying by zero, so lane `b`'s
 * visible key set and its accumulation order are exactly what a one-lane decode
 * of the same lane produces -- the padded batch is bit-identical, not merely
 * close.
 *
 * `lengths` is null for a cross-attention memory, which every lane reads whole:
 * there is no per-lane length there to publish, only the query row each lane
 * selects.
 */
export function stageKeepMask(
  lanes: number,
  keys: number,
  lengths: Int32Array | null,
  read: KeepMaskReader | null,
  key: string,
  allocate: KVScratchAllocator,
  sample: Int32Array,
): { storage: Int32Array; apply: () => void } {
  const staged = allocate(key, lanes * keys, sample) as Int32Array;
  if (staged.length !== lanes * keys || !(staged instanceof Int32Array)) {
    throw new Error(`Decode keep mask '${key}' needs ${lanes * keys} I32 elements.`);
  }
  /* Deferred like every other staged operand, and for the same reason: the
   * graph's keep mask is a *changed input* of this step, so a step that filled
   * this at plan time would publish the previous step's visibility -- the new
   * token would be masked out of its own attention. The one-lane path reads the
   * mask through a view and inherits the deferral for free, which is why this
   * only ever went wrong on the batched path. */
  const apply = () => {
    for (let lane = 0; lane < lanes; lane++) {
      const length = lengths === null ? keys : lengths[lane];
      const base = lane * keys;
      for (let token = 0; token < keys; token++) {
        staged[base + token] = token < length && (read === null || read(lane, token)) ? 1 : 0;
      }
    }
  };
  return { storage: staged, apply };
}
