/* Reading a paged K/V prefix as the contiguous operand attention kernels take.
 *
 * Row execution used to spell "the visible K/V prefix" as
 * `storage.subarray(0, (position + 1) * width)`. That expression assumes the
 * address of token `p` is a linear function of `p`, which is exactly the
 * assumption a page table removes. This module replaces the expression with a
 * resolver that keeps the old behaviour as its fastest tier.
 *
 * Three tiers, cheapest first:
 *
 *   identity   The lane's pages are `lane * pagesPerLane + i`. The prefix is
 *              one contiguous run at `lane * laneTokenCapacity * width`, which
 *              is the same view the pre-paging code produced, from the same
 *              storage, with no copy. Contiguous KV costs nothing here.
 *
 *   run        The lane's pages are consecutive but not based at the identity
 *              origin — the ordinary result of a free list handing out a run.
 *              Still one contiguous view, still no copy.
 *
 *   gather     Anything else: copy the active tokens into a reusable scratch
 *              buffer in logical order. This is the real cost of paging on a
 *              backend whose attention kernel cannot address a page table, and
 *              it is paid per attention node per step. A kernel that *can*
 *              address the page table should read `plan.pageTable` directly
 *              and never call `pagedSequenceView` at all.
 *
 * The scratch buffers live on the plan and are keyed by tensor, so a decode
 * loop reuses them instead of allocating one per token.
 */

import type { PagedKVCache } from '../core/PagedKVCache.js';
import type { RuntimeTypedArray } from '../types.js';

export type KVPageTier = 'identity' | 'run' | 'gather';

export interface KVPagePlan {
  readonly lane: number;
  /** Active key/value length. A value, never a shape. */
  readonly kvLength: number;
  readonly pageTokens: number;
  /** Lane-local logical -> physical page mapping; -1 is unmapped. */
  readonly pageTable: Int32Array;
  readonly pagesPerLane: number;
  /**
   * Activation tensors that live in the page pool — the attention `k` and `v`
   * operands and nothing else.
   *
   * Deliberately narrow. A general "any tensor may be paged" rule would need
   * every row-local operator to address a page table, and the row path proves
   * its domain rather than assuming it: a node outside attention that touches
   * a paged tensor is refused, not silently addressed with linear arithmetic.
   */
  readonly pagedTensors: ReadonlySet<string>;
}

export interface KVPageView {
  readonly storage: RuntimeTypedArray;
  readonly length: number;
  readonly tier: KVPageTier;
  /**
   * Runs the copy, or null when the view already aliases the pool.
   *
   * Separating "where the operand is" from "when it is filled" is not an
   * optimisation — it is required. Row plans are built for the whole step
   * before any node executes, and the last token of the visible prefix is
   * produced *by* that step's projection. A gather performed while the plan
   * was being built would read the previous step's bytes for the current
   * position and yield a decoder that is wrong and looks plausible.
   */
  readonly gather: (() => void) | null;
}

/**
 * Storage for one gather, owned by the backend.
 *
 * WASM is why this is a callback rather than a plain array: its kernels take
 * pointers into the compiled linear memory, so a gather buffer allocated as an
 * ordinary typed array is rejected at dispatch. The backend allocates from its
 * own heap and hands back a view, and the addressing code stays backend-blind.
 */
export type KVScratchAllocator = (
  key: string,
  elements: number,
  sample: RuntimeTypedArray,
) => RuntimeTypedArray;

/**
 * Capacity to allocate for a request of `elements`.
 *
 * Geometric, because a decode loop's gather grows by one token per step: an
 * exact-fit allocator would allocate once per generated token for the whole
 * generation, which is the unbounded per-token growth the decode performance
 * contract forbids. Doubling makes it logarithmic in the lane capacity.
 */
export function kvScratchCapacity(elements: number, current: number): number {
  if (current >= elements) return current;
  let capacity = Math.max(current, 1);
  while (capacity < elements) capacity *= 2;
  return capacity;
}

function defaultScratchAllocator(): KVScratchAllocator {
  const buffers = new Map<string, RuntimeTypedArray>();
  return (key, elements, sample) => {
    let buffer = buffers.get(key);
    if (!buffer || buffer.length < elements || buffer.constructor !== sample.constructor) {
      const capacity = kvScratchCapacity(elements, buffer?.length ?? 0);
      buffer = new (sample.constructor as any)(capacity) as RuntimeTypedArray;
      buffers.set(key, buffer);
    }
    return (buffer.length === elements
      ? buffer
      : buffer.subarray(0, elements)) as RuntimeTypedArray;
  };
}

export { defaultScratchAllocator };

function activePageCount(plan: KVPagePlan): number {
  return Math.ceil(plan.kvLength / plan.pageTokens);
}

/**
 * Build a lane's plan from a cache. The page table is copied into a lane-local
 * view so a kernel cannot accidentally address another lane's mapping.
 */
export function kvPagePlanForLane(
  cache: PagedKVCache,
  lane: number,
  pagedTensors: Iterable<string> = [],
  /** Provisional visible length for an uncommitted scheduler reservation. */
  visibleLength: number = cache.kvLength[lane],
): KVPagePlan {
  if (!Number.isSafeInteger(visibleLength) || visibleLength < cache.kvLength[lane] ||
      visibleLength > cache.laneTokenCapacity) {
    throw new RangeError(`KV visible length ${visibleLength} is outside lane ${lane}'s capacity.`);
  }
  const base = lane * cache.pagesPerLane;
  /* A plan may outlive the scheduler turn that created it. Keep a snapshot,
   * not a live view that release/reuse can mutate underneath submitted work. */
  const pageTable = new Int32Array(
    cache.pageTable.subarray(base, base + cache.pagesPerLane),
  );
  return Object.freeze({
    lane,
    kvLength: visibleLength,
    pageTokens: cache.pageTokens,
    pageTable,
    pagesPerLane: cache.pagesPerLane,
    pagedTensors: new Set(pagedTensors),
  });
}

/**
 * Which tier this lane's mapping falls into, and where the run starts.
 *
 * `identity` is a strict subset of `run`; it is reported separately because it
 * is the property that makes paged and contiguous KV the same code path, and a
 * caller checking "did paging change anything?" is asking about that one.
 */
export function classifyPlan(plan: KVPagePlan): { tier: KVPageTier; firstPage: number } {
  const pages = activePageCount(plan);
  if (pages === 0) return { tier: 'identity', firstPage: plan.lane * plan.pagesPerLane };
  const first = plan.pageTable[0];
  if (first < 0) return { tier: 'gather', firstPage: -1 };
  let consecutive = true;
  for (let logical = 1; logical < pages; logical++) {
    if (plan.pageTable[logical] !== first + logical) { consecutive = false; break; }
  }
  if (!consecutive) return { tier: 'gather', firstPage: -1 };
  return {
    tier: first === plan.lane * plan.pagesPerLane ? 'identity' : 'run',
    firstPage: first,
  };
}

/**
 * The active K/V prefix of `plan` as `length * width` contiguous elements.
 *
 * `storage` is the whole page pool: token slot `s` occupies elements
 * `[s * width, (s + 1) * width)`. Under the identity map that is the same
 * `[B, S, D]` activation decode has always retained, so the returned view is
 * byte-identical to the one `attentionPrefix` used to build.
 */
export function pagedSequenceView(
  storage: RuntimeTypedArray,
  width: number,
  plan: KVPagePlan,
  key: string,
  allocate: KVScratchAllocator,
): KVPageView {
  const length = plan.kvLength;
  if (length <= 0) {
    return {
      storage: storage.subarray(0, 0) as RuntimeTypedArray,
      length: 0,
      tier: 'identity',
      gather: null,
    };
  }
  const { tier, firstPage } = classifyPlan(plan);
  if (tier !== 'gather') {
    const start = firstPage * plan.pageTokens * width;
    const view = storage.subarray(start, start + length * width) as RuntimeTypedArray;
    if (view.length !== length * width) {
      throw new Error(
        `Paged K/V run for lane ${plan.lane} needs ${length * width} elements at ${start} ` +
        `but the page pool holds ${storage.length}.`);
    }
    return { storage: view, length, tier, gather: null };
  }
  const buffer = allocate(key, length * width, storage);
  if (buffer.length !== length * width) {
    throw new Error(
      `Paged K/V gather '${key}' needs ${length * width} elements but received ${buffer.length}.`);
  }
  const slots = new Int32Array(length);
  for (let position = 0; position < length; position++) {
    const slot = pagedSlot(plan, position);
    if (slot < 0) {
      throw new Error(
        `Paged K/V lane ${plan.lane} position ${position} is not backed by a resident page.`);
    }
    slots[position] = slot;
  }
  /* Slots are resolved now and copied later: the mapping is fixed for the
   * whole step, but the bytes are not. */
  const gather = () => {
    for (let position = 0; position < length; position++) {
      const source = slots[position] * width;
      buffer.set(storage.subarray(source, source + width) as any, position * width);
    }
  };
  return { storage: buffer, length, tier: 'gather', gather };
}

/**
 * The physical token slot backing logical `position`, or -1 when unmapped.
 *
 * The write side is the half a gather cannot cover: a gathered read is a copy,
 * but a decode step must place its new K/V row in the page that backs it, or
 * the next step's gather reads a slot nobody wrote. Returning a *slot index*
 * rather than a byte offset lets the row executor keep using its existing
 * `slot * width` row arithmetic with one substituted index.
 */
export function pagedSlot(plan: KVPagePlan, position: number): number {
  const page = plan.pageTable[Math.floor(position / plan.pageTokens)];
  if (page < 0) return -1;
  return page * plan.pageTokens + (position % plan.pageTokens);
}

/**
 * The minimal set of contiguous token runs covering a lane's active prefix.
 *
 * `source` is a physical token slot, `destination` a logical position, and
 * `tokens` the run length. Consecutive pages coalesce, so an identity mapping
 * yields one run and a fully scattered one yields a run per page. A backend
 * that stages a paged prefix into a contiguous buffer issues exactly one copy
 * per entry, which is what makes "how paged is this lane, really?" a number
 * rather than a guess.
 */
export function kvPageCopyRanges(
  plan: KVPagePlan,
): Array<{ source: number; destination: number; tokens: number }> {
  const ranges: Array<{ source: number; destination: number; tokens: number }> = [];
  for (let position = 0; position < plan.kvLength; position++) {
    const slot = pagedSlot(plan, position);
    if (slot < 0) {
      throw new Error(
        `Paged K/V lane ${plan.lane} position ${position} is not backed by a resident page.`);
    }
    const last = ranges[ranges.length - 1];
    if (last && last.source + last.tokens === slot &&
        last.destination + last.tokens === position) {
      last.tokens++;
      continue;
    }
    ranges.push({ source: slot, destination: position, tokens: 1 });
  }
  return ranges;
}

/**
 * The physical slot a lane's prefix starts at when it is one contiguous run,
 * or -1 when the mapping needs a gather.
 *
 * A backend whose attention operands are bound at a fixed buffer offset can
 * only read a lane whose run starts where its binding does. Answering with -1
 * lets that backend refuse rather than read the wrong pages.
 */
export function pagedRunOrigin(plan: KVPagePlan): number {
  if (plan.kvLength <= 0) return 0;
  const ranges = kvPageCopyRanges(plan);
  return ranges.length === 1 && ranges[0].destination === 0 ? ranges[0].source : -1;
}

export function pagedWriteOffset(plan: KVPagePlan, position: number, width: number): number {
  const slot = pagedSlot(plan, position);
  if (slot < 0) {
    throw new Error(
      `Paged K/V lane ${plan.lane} has no resident page for write position ${position}.`);
  }
  return slot * width;
}
