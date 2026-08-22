/* Paged (block) KV cache: the named resource decode never had.
 *
 * Until this file existed, KV was a side effect of arena policy — a retained
 * `[1,S,D]` activation that decode sliced with `subarray(0, (pos+1)*width)`.
 * That representation has no owner, so there is nowhere to put a page table,
 * and it carries the active length *as a shape*, which becomes a ragged tensor
 * the moment two lanes disagree. Both problems are solved the same way: KV
 * becomes an object with a page table, and length becomes a value.
 *
 * Three properties this file is built around.
 *
 * 1. Length is `I32[B]`, never a shape. `kvLength` and `queryLength` are
 *    runtime values read by kernels. No output shape is ever derived from
 *    them, so nothing here can produce a public ragged tensor.
 *
 * 2. Contiguous KV is not a separate path — it is the allocation policy whose
 *    page table happens to be the identity map. `'contiguous'` pins logical
 *    page `i` of lane `b` to physical page `b * pagesPerLane + i`, which makes
 *    `physicalTokenIndex()` reduce to the old `b * capacity + pos` arithmetic
 *    exactly. Paged and contiguous results are therefore identical by
 *    construction rather than by test, which is what keeps the two from
 *    drifting the way this repository's three shape inferences did.
 *
 * 3. Every allocation is transactional. A step reserves all the pages it needs
 *    before it writes a token or publishes a new active length; a failure at
 *    any reservation point rolls the page table back to its exact prior state.
 *
 * The native twin is `native/src/runtime/paged_kv.c`. Both are driven by
 * `tests/paged_kv_vectors.json`, so an operation that behaves differently in
 * the two runtimes fails a test rather than producing a decoder that works in
 * the browser and not on the robot.
 */

export type PageAllocationPolicy = 'paged' | 'contiguous';

export type PagedKVFailureCode =
  | 'CAPACITY_EXHAUSTED'
  | 'INVALID_ARGUMENT'
  | 'PREFIX_NOT_FOUND'
  | 'PREFIX_CONFLICT'
  | 'PAGE_SHARED'
  | 'STALE_PAGE';

export interface PagedKVCacheOptions {
  /** Dense slot capacity. One lane per scheduler slot; membership may churn. */
  readonly lanes: number;
  /** Tokens per page. Backend-specific: large for CPU/WASM, small for device. */
  readonly pageTokens: number;
  /** Logical token capacity of one lane. Bounds the per-lane page table width. */
  readonly laneTokenCapacity: number;
  /**
   * Hard bound on resident physical pages. Defaults to the fully private
   * `lanes * pagesPerLane`, which is exactly the contiguous high-water mark;
   * a smaller bound is the whole point of paging and makes `reserve` fail
   * closed instead of growing without limit.
   */
  readonly maxPages?: number;
  /** Bytes one token occupies in one page. Telemetry only; not addressing. */
  readonly bytesPerToken?: number;
  readonly policy?: PageAllocationPolicy;
  /**
   * Zero a page's tokens when it is recycled. Generation tags already stop a
   * stale *handle* from reading a reused page, but a kernel addressing a
   * freshly reserved page before the step writes it would otherwise observe
   * another request's bytes. Both defences are kept: the tag catches the bug
   * in the runtime, the clear bounds the damage if the tag is ever bypassed.
   */
  readonly clearOnRecycle?: boolean;
}

export interface PagedKVTelemetry {
  /** Bytes the active lengths actually mean. */
  readonly logicalBytes: number;
  /** Bytes held by pages assigned to a lane or a published prefix. */
  readonly residentBytes: number;
  /** Bytes held by open reservations that have not committed. */
  readonly reservedBytes: number;
  readonly residentPages: number;
  readonly reservedPages: number;
  readonly freePages: number;
  readonly sharedPages: number;
  /** Resident bytes that no active length reaches: partial tail pages. */
  readonly fragmentationBytes: number;
  readonly highWaterPages: number;
  readonly prefixHits: number;
  readonly prefixMisses: number;
  readonly copyOnWrites: number;
  readonly evictions: number;
  readonly allocationFailures: number;
}

/** Open, uncommitted page reservation. Commit publishes, rollback undoes. */
export interface PagedKVReservation {
  readonly lane: number;
  /** Lane occupant generation captured before any page is reserved. */
  readonly laneGeneration: number;
  readonly tokens: number;
  readonly priorLength: number;
  readonly pages: readonly number[];
  readonly logicalPages: readonly number[];
}

export class PagedKVError extends Error {
  readonly code: PagedKVFailureCode;

  constructor(code: PagedKVFailureCode, message: string) {
    super(message);
    this.name = 'PagedKVError';
    this.code = code;
  }
}

const UNMAPPED = -1;

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new PagedKVError('INVALID_ARGUMENT', `${label} must be a positive integer.`);
  }
  return value as number;
}

interface SharedPrefix {
  readonly key: string;
  readonly pages: readonly number[];
  readonly tokens: number;
  /** Lanes currently holding this prefix. Zero means the entry is evictable. */
  laneReferences: number;
  /** Monotonic admission order. The eviction policy is oldest-unreferenced. */
  readonly sequence: number;
  lastUse: number;
}

/**
 * Context-local paged KV owner.
 *
 * Ownership is deliberately context-local, and stays so: no page
 * moves between two caches, so there is no device-level owner to reason about
 * and no cross-context aliasing to prove absent.
 */
export class PagedKVCache {
  readonly lanes: number;
  readonly pageTokens: number;
  readonly laneTokenCapacity: number;
  readonly pagesPerLane: number;
  readonly maxPages: number;
  readonly bytesPerToken: number;
  readonly policy: PageAllocationPolicy;
  readonly clearOnRecycle: boolean;

  /** `I32[B, pagesPerLane]`. Logical page -> physical page, -1 unmapped. */
  readonly pageTable: Int32Array;
  /** `I32[B]`. Active key/value length. Never an input to shape inference. */
  readonly kvLength: Int32Array;
  /** `I32[B]`. Active query length; prefill and decode differ only here. */
  readonly queryLength: Int32Array;
  /** `I32[B]`. Slot generation, so a stale step cannot address a reused lane. */
  readonly laneGeneration: Int32Array;

  readonly #refCount: Int32Array;
  readonly #pageGeneration: Int32Array;
  /* Which shared prefix each lane holds, tracked explicitly rather than
   * inferred by comparing the lane's page table against the prefix's pages.
   * A lane that copies-on-write no longer matches its own prefix, and the
   * comparison would then leak the reference forever — the prefix would stay
   * pinned and uneviatable for the rest of the cache's life. */
  readonly #lanePrefix: (SharedPrefix | null)[];
  /** Reservations that have neither committed nor rolled back. */
  readonly #openReservations = new WeakSet<PagedKVReservation>();
  /** Exactly one transaction may own a lane's unpublished page-table state. */
  readonly #openReservationByLane: (PagedKVReservation | null)[];
  /** Binary min-heap of free physical pages: lowest index first, portably. */
  readonly #freeHeap: Int32Array;
  #freeCount = 0;
  #reservedPages = 0;
  #highWaterPages = 0;
  #prefixSequence = 0;
  #clock = 0;
  readonly #prefixes = new Map<string, SharedPrefix>();
  readonly #counters = {
    prefixHits: 0,
    prefixMisses: 0,
    copyOnWrites: 0,
    evictions: 0,
    allocationFailures: 0,
  };
  /** Storage hook. Set by a backend that needs recycled pages cleared. */
  #clearPage: ((page: number) => void) | null = null;

  constructor(options: PagedKVCacheOptions) {
    this.lanes = positiveInteger(options?.lanes, 'lanes');
    this.pageTokens = positiveInteger(options?.pageTokens, 'pageTokens');
    this.laneTokenCapacity = positiveInteger(
      options?.laneTokenCapacity, 'laneTokenCapacity');
    this.pagesPerLane = Math.ceil(this.laneTokenCapacity / this.pageTokens);
    const privatePages = this.lanes * this.pagesPerLane;
    this.maxPages = options?.maxPages === undefined
      ? privatePages
      : positiveInteger(options.maxPages, 'maxPages');
    this.bytesPerToken = options?.bytesPerToken === undefined
      ? 1
      : positiveInteger(options.bytesPerToken, 'bytesPerToken');
    this.policy = options?.policy ?? 'paged';
    if (this.policy !== 'paged' && this.policy !== 'contiguous') {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Page allocation policy '${String(this.policy)}' is not 'paged' or 'contiguous'.`);
    }
    /* Contiguous is the identity map, and the identity map needs every private
     * page to exist. A smaller bound would silently stop reproducing the
     * contiguous baseline it is there to reproduce. */
    if (this.policy === 'contiguous' && this.maxPages < privatePages) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Contiguous policy needs ${privatePages} pages for the identity map; bound is ${this.maxPages}.`);
    }
    this.clearOnRecycle = options?.clearOnRecycle !== false;

    this.pageTable = new Int32Array(this.lanes * this.pagesPerLane).fill(UNMAPPED);
    this.kvLength = new Int32Array(this.lanes);
    this.queryLength = new Int32Array(this.lanes);
    this.laneGeneration = new Int32Array(this.lanes);
    this.#refCount = new Int32Array(this.maxPages);
    this.#pageGeneration = new Int32Array(this.maxPages);
    this.#lanePrefix = new Array(this.lanes).fill(null);
    this.#openReservationByLane = new Array(this.lanes).fill(null);
    this.#freeHeap = new Int32Array(this.maxPages);
    /* Seed ascending so the very first allocations are 0,1,2,... — the order
     * the contiguous identity map also produces for lane zero. */
    for (let page = 0; page < this.maxPages; page++) this.#freeHeap[page] = page;
    this.#freeCount = this.maxPages;
  }

  /** Install the backend's page eraser used when a page is recycled. */
  setPageEraser(erase: ((page: number) => void) | null): void {
    this.#clearPage = erase;
  }

  // --- physical addressing -------------------------------------------------

  /**
   * Physical page holding logical page `logical` of `lane` under the identity
   * map. This single expression is what makes contiguous a policy rather than
   * a second code path.
   */
  identityPhysicalPage(lane: number, logical: number): number {
    return lane * this.pagesPerLane + logical;
  }

  /** Whether this lane's mapping is the identity map, i.e. plain contiguous. */
  laneIsContiguous(lane: number): boolean {
    this.#assertLane(lane);
    const pages = Math.ceil(this.kvLength[lane] / this.pageTokens);
    const base = lane * this.pagesPerLane;
    for (let logical = 0; logical < pages; logical++) {
      if (this.pageTable[base + logical] !== this.identityPhysicalPage(lane, logical)) {
        return false;
      }
    }
    return true;
  }

  physicalPage(lane: number, logical: number): number {
    this.#assertLane(lane);
    if (!Number.isSafeInteger(logical) || logical < 0 || logical >= this.pagesPerLane) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Logical page ${logical} is outside lane capacity ${this.pagesPerLane}.`);
    }
    return this.pageTable[lane * this.pagesPerLane + logical];
  }

  /**
   * Flat physical token index of `position` in `lane`. Under the identity map
   * this is exactly `lane * laneTokenCapacity + position`, which is the
   * arithmetic every backend already open-codes.
   */
  physicalTokenIndex(lane: number, position: number): number {
    const logical = Math.floor(position / this.pageTokens);
    const page = this.physicalPage(lane, logical);
    if (page < 0) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Lane ${lane} position ${position} is not backed by a resident page.`);
    }
    return page * this.pageTokens + (position % this.pageTokens);
  }

  /**
   * Physical token index of every active key position of `lane`, in order.
   * This is the gather a kernel without page-table addressing consumes; a
   * kernel that has one reads `pageTable` and never materialises this.
   */
  gatherActiveTokens(lane: number, out?: Int32Array): Int32Array {
    this.#assertLane(lane);
    const length = this.kvLength[lane];
    const target = out && out.length >= length ? out : new Int32Array(length);
    for (let position = 0; position < length; position++) {
      target[position] = this.physicalTokenIndex(lane, position);
    }
    return target.length === length ? target : target.subarray(0, length);
  }

  // --- transactional reservation ------------------------------------------

  /**
   * Reserve every page `tokens` more tokens on `lane` will need. Nothing is
   * published: `kvLength` still reads its old value, so a caller that fails
   * between reserve and commit leaves a lane no step can observe as extended.
   */
  reserve(lane: number, tokens: number): PagedKVReservation {
    this.#assertLane(lane);
    if (this.#openReservationByLane[lane] !== null) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Lane ${lane} already has an open reservation; commit or roll it back first.`);
    }
    if (!Number.isSafeInteger(tokens) || tokens < 0) {
      throw new PagedKVError('INVALID_ARGUMENT', `Reserved token count ${tokens} must be >= 0.`);
    }
    const priorLength = this.kvLength[lane];
    const nextLength = priorLength + tokens;
    if (nextLength > this.laneTokenCapacity) {
      this.#counters.allocationFailures++;
      throw new PagedKVError('CAPACITY_EXHAUSTED',
        `Lane ${lane} would reach ${nextLength} tokens over capacity ${this.laneTokenCapacity}.`);
    }
    const base = lane * this.pagesPerLane;
    const firstLogical = Math.floor(priorLength / this.pageTokens);
    const lastLogical = nextLength === 0 ? -1 : Math.floor((nextLength - 1) / this.pageTokens);
    const pages: number[] = [];
    const logicalPages: number[] = [];
    try {
      for (let logical = firstLogical; logical <= lastLogical; logical++) {
        if (this.pageTable[base + logical] !== UNMAPPED) continue;
        const page = this.#allocatePage(lane, logical);
        this.pageTable[base + logical] = page;
        pages.push(page);
        logicalPages.push(logical);
      }
    } catch (error) {
      /* Partial reservations are the failure this repository cannot afford to
       * leave behind: a half-mapped lane reads another request's pages. Undo
       * every page this call mapped before rethrowing. */
      for (let index = pages.length - 1; index >= 0; index--) {
        this.pageTable[base + logicalPages[index]] = UNMAPPED;
        this.#releasePage(pages[index]);
      }
      throw error;
    }
    this.#reservedPages += pages.length;
    this.#noteHighWater();
    const reservation = Object.freeze({
      lane,
      laneGeneration: this.laneGeneration[lane],
      tokens,
      priorLength,
      pages: Object.freeze(pages),
      logicalPages: Object.freeze(logicalPages),
    });
    this.#openReservations.add(reservation);
    this.#openReservationByLane[lane] = reservation;
    return reservation;
  }

  /** Publish the reserved extension. After this the new tokens are readable. */
  commit(reservation: PagedKVReservation): void {
    this.#assertReservation(reservation);
    this.#settle(reservation);
    this.#reservedPages -= reservation.pages.length;
    this.kvLength[reservation.lane] = reservation.priorLength + reservation.tokens;
    this.queryLength[reservation.lane] = reservation.tokens;
  }

  /** Return the reservation's pages and restore the exact prior page table. */
  rollback(reservation: PagedKVReservation): void {
    this.#assertReservation(reservation);
    this.#settle(reservation);
    const base = reservation.lane * this.pagesPerLane;
    for (let index = reservation.pages.length - 1; index >= 0; index--) {
      this.pageTable[base + reservation.logicalPages[index]] = UNMAPPED;
      this.#releasePage(reservation.pages[index]);
    }
    this.#reservedPages -= reservation.pages.length;
    this.kvLength[reservation.lane] = reservation.priorLength;
  }

  /** Reserve and commit in one step. Throws without side effects on failure. */
  append(lane: number, tokens: number): PagedKVReservation {
    const reservation = this.reserve(lane, tokens);
    this.commit(reservation);
    return reservation;
  }

  // --- lane lifecycle ------------------------------------------------------

  /**
   * Drop every private page of `lane` and bump its generation. The bump is
   * what stops work submitted against the old occupant from landing in the
   * new one once a scheduler reuses the slot.
   */
  releaseLane(lane: number): void {
    this.#assertLane(lane);
    this.#assertNoOpenReservation(lane, 'released');
    const base = lane * this.pagesPerLane;
    for (let logical = 0; logical < this.pagesPerLane; logical++) {
      const page = this.pageTable[base + logical];
      if (page === UNMAPPED) continue;
      this.pageTable[base + logical] = UNMAPPED;
      this.#releasePage(page);
    }
    const held = this.#lanePrefix[lane];
    if (held) {
      held.laneReferences--;
      this.#lanePrefix[lane] = null;
    }
    this.kvLength[lane] = 0;
    this.queryLength[lane] = 0;
    this.laneGeneration[lane]++;
  }

  /** Reset every lane and prefix. Page contents are recycled, not retained. */
  reset(): void {
    /* Preflight every lane before changing the first one, so an open
     * transaction cannot turn reset into a partially applied lifecycle step. */
    for (let lane = 0; lane < this.lanes; lane++) {
      this.#assertNoOpenReservation(lane, 'reset');
    }
    for (let lane = 0; lane < this.lanes; lane++) this.releaseLane(lane);
    for (const prefix of this.#prefixes.values()) {
      for (const page of prefix.pages) this.#releasePage(page);
    }
    this.#prefixes.clear();
  }

  // --- prefix sharing ------------------------------------------------------

  /**
   * Publish `tokens` of `lane` as an immutable shared prefix under `key`.
   *
   * A prefix must end on a page boundary. A partial tail page is still being
   * appended to, and publishing it would hand another request a page whose
   * bytes are about to change, violating immutable-prefix ownership.
   */
  publishPrefix(key: string, lane: number, tokens: number): void {
    this.#assertLane(lane);
    this.#assertNoOpenReservation(lane, 'published as a prefix');
    if (typeof key !== 'string' || key.length === 0) {
      throw new PagedKVError('INVALID_ARGUMENT', 'Prefix identity must be a non-empty key.');
    }
    if (!Number.isSafeInteger(tokens) || tokens <= 0 || tokens > this.kvLength[lane]) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Prefix length ${tokens} is outside lane ${lane}'s active length ${this.kvLength[lane]}.`);
    }
    if (tokens % this.pageTokens !== 0) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Prefix length ${tokens} must end on a ${this.pageTokens}-token page boundary.`);
    }
    const existing = this.#prefixes.get(key);
    const pageCount = tokens / this.pageTokens;
    const base = lane * this.pagesPerLane;
    const pages: number[] = [];
    for (let logical = 0; logical < pageCount; logical++) {
      const page = this.pageTable[base + logical];
      if (page === UNMAPPED) {
        throw new PagedKVError('INVALID_ARGUMENT',
          `Lane ${lane} logical page ${logical} is unmapped and cannot be published.`);
      }
      pages.push(page);
    }
    if (existing) {
      /* One identity, one page range. Republishing the same key over different
       * pages would let two requests holding "the same" prefix read different
       * bytes, which is a correctness failure no amount of refcounting fixes. */
      const same = existing.tokens === tokens &&
        existing.pages.length === pages.length &&
        existing.pages.every((page, index) => page === pages[index]);
      if (!same) {
        throw new PagedKVError('PREFIX_CONFLICT',
          `Prefix '${key}' is already published over a different page range.`);
      }
      existing.lastUse = ++this.#clock;
      return;
    }
    /* The cache's own reference keeps published pages resident after the
     * publishing lane is released; eviction is what removes them. */
    for (const page of pages) this.#refCount[page]++;
    this.#prefixes.set(key, {
      key,
      pages: Object.freeze(pages),
      tokens,
      laneReferences: 0,
      sequence: ++this.#prefixSequence,
      lastUse: ++this.#clock,
    });
  }

  /** Whether a published prefix with this identity is resident. */
  hasPrefix(key: string): boolean {
    return this.#prefixes.has(key);
  }

  /**
   * Bind a published prefix into `lane`. The lane must be empty: a prefix is
   * a *prefix*, and grafting one under existing tokens would renumber them.
   * Returns the bound token count.
   */
  acquirePrefix(key: string, lane: number): number {
    this.#assertLane(lane);
    this.#assertNoOpenReservation(lane, 'bound to a prefix');
    const prefix = this.#prefixes.get(key);
    if (!prefix) {
      this.#counters.prefixMisses++;
      throw new PagedKVError('PREFIX_NOT_FOUND', `Prefix '${key}' is not resident.`);
    }
    if (this.kvLength[lane] !== 0) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Lane ${lane} must be empty before a shared prefix is bound.`);
    }
    const base = lane * this.pagesPerLane;
    for (let logical = 0; logical < prefix.pages.length; logical++) {
      const page = prefix.pages[logical];
      this.pageTable[base + logical] = page;
      this.#refCount[page]++;
    }
    prefix.laneReferences++;
    prefix.lastUse = ++this.#clock;
    this.#lanePrefix[lane] = prefix;
    this.kvLength[lane] = prefix.tokens;
    this.queryLength[lane] = 0;
    this.#counters.prefixHits++;
    this.#noteHighWater();
    return prefix.tokens;
  }

  /** Whether this lane's logical page is shared and therefore not writable. */
  pageIsShared(lane: number, logical: number): boolean {
    const page = this.physicalPage(lane, logical);
    return page !== UNMAPPED && this.#refCount[page] > 1;
  }

  /**
   * Give `lane` a private copy of a shared logical page before it is written.
   *
   * The caller copies the bytes; this moves the mapping and the reference
   * counts. Returning the pair lets a backend issue exactly one page-sized
   * copy with no knowledge of the allocator.
   */
  copyOnWrite(lane: number, logical: number): { readonly from: number; readonly to: number } {
    this.#assertLane(lane);
    this.#assertNoOpenReservation(lane, 'changed by copy-on-write');
    const source = this.physicalPage(lane, logical);
    if (source === UNMAPPED) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Lane ${lane} logical page ${logical} is unmapped.`);
    }
    if (this.#refCount[source] <= 1) {
      return Object.freeze({ from: source, to: source });
    }
    const target = this.#allocatePage(lane, logical);
    this.pageTable[lane * this.pagesPerLane + logical] = target;
    this.#refCount[source]--;
    this.#counters.copyOnWrites++;
    this.#noteHighWater();
    return Object.freeze({ from: source, to: target });
  }

  /**
   * Evict unreferenced published prefixes until `pages` are free, oldest use
   * first. A prefix bound to a live lane is never a candidate, so eviction
   * cannot remove KV an in-flight request still reads.
   */
  evict(pages: number): number {
    if (!Number.isSafeInteger(pages) || pages < 0) {
      throw new PagedKVError('INVALID_ARGUMENT', `Eviction target ${pages} must be >= 0.`);
    }
    let reclaimed = 0;
    while (this.#freeCount < pages) {
      let victim: SharedPrefix | null = null;
      for (const prefix of this.#prefixes.values()) {
        if (prefix.laneReferences > 0) continue;
        /* Deterministic: least-recently-used, ties broken by publication
         * order, so an identical schedule evicts an identical set. */
        if (!victim || prefix.lastUse < victim.lastUse ||
            (prefix.lastUse === victim.lastUse && prefix.sequence < victim.sequence)) {
          victim = prefix;
        }
      }
      if (!victim) break;
      this.#prefixes.delete(victim.key);
      for (const page of victim.pages) this.#releasePage(page);
      this.#counters.evictions++;
      reclaimed += victim.pages.length;
    }
    return reclaimed;
  }

  // --- telemetry -----------------------------------------------------------

  telemetry(): Readonly<PagedKVTelemetry> {
    let logicalTokens = 0;
    for (let lane = 0; lane < this.lanes; lane++) logicalTokens += this.kvLength[lane];
    let residentPages = 0;
    let sharedPages = 0;
    for (let page = 0; page < this.maxPages; page++) {
      if (this.#refCount[page] <= 0) continue;
      residentPages++;
      if (this.#refCount[page] > 1) sharedPages++;
    }
    const residentBytes = residentPages * this.pageTokens * this.bytesPerToken;
    /* Shared tokens are counted once per holding lane in logicalTokens, which
     * is the honest reading of "bytes the active lengths mean" but would make
     * fragmentation negative. Clamp rather than redefine either number. */
    const logicalBytes = logicalTokens * this.bytesPerToken;
    return Object.freeze({
      logicalBytes,
      residentBytes,
      reservedBytes: this.#reservedPages * this.pageTokens * this.bytesPerToken,
      residentPages,
      reservedPages: this.#reservedPages,
      freePages: this.#freeCount,
      sharedPages,
      fragmentationBytes: Math.max(0, residentBytes - logicalBytes),
      highWaterPages: this.#highWaterPages,
      prefixHits: this.#counters.prefixHits,
      prefixMisses: this.#counters.prefixMisses,
      copyOnWrites: this.#counters.copyOnWrites,
      evictions: this.#counters.evictions,
      allocationFailures: this.#counters.allocationFailures,
    });
  }

  /** Page recycle tag. A handle carrying a stale tag must not be addressed. */
  pageGeneration(page: number): number {
    if (!Number.isSafeInteger(page) || page < 0 || page >= this.maxPages) {
      throw new PagedKVError('INVALID_ARGUMENT', `Physical page ${page} is out of range.`);
    }
    return this.#pageGeneration[page];
  }

  // --- internals -----------------------------------------------------------

  #assertLane(lane: number): void {
    if (!Number.isSafeInteger(lane) || lane < 0 || lane >= this.lanes) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Lane ${lane} is outside slot capacity ${this.lanes}.`);
    }
  }

  #assertReservation(reservation: PagedKVReservation): void {
    if (!reservation || !Number.isSafeInteger(reservation.lane)) {
      throw new PagedKVError('INVALID_ARGUMENT', 'Reservation handle is not valid.');
    }
    this.#assertLane(reservation.lane);
    if (!this.#openReservations.has(reservation) ||
        this.#openReservationByLane[reservation.lane] !== reservation) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Lane ${reservation.lane}'s reservation was already committed or rolled back.`);
    }
    if (reservation.laneGeneration !== this.laneGeneration[reservation.lane]) {
      throw new PagedKVError('STALE_PAGE',
        `Lane ${reservation.lane}'s reservation belongs to generation ` +
        `${reservation.laneGeneration}, not ${this.laneGeneration[reservation.lane]}.`);
    }
    if (this.kvLength[reservation.lane] !== reservation.priorLength) {
      throw new PagedKVError('STALE_PAGE',
        `Lane ${reservation.lane}'s active length changed while its reservation was open.`);
    }
  }

  #assertNoOpenReservation(lane: number, action: string): void {
    if (this.#openReservationByLane[lane] !== null) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Lane ${lane} cannot be ${action} while a reservation is open.`);
    }
  }

  /* A reservation settles exactly once.
   *
   * Without this, committing and then rolling back the same handle
   * double-subtracts `reservedPages` and republishes the pre-commit length --
   * silently, and long after the caller's mistake. Refusing the second
   * transition turns a corrupted allocator into a thrown error. */
  #settle(reservation: PagedKVReservation): void {
    if (!this.#openReservations.delete(reservation)) {
      throw new PagedKVError('INVALID_ARGUMENT',
        `Lane ${reservation.lane}'s reservation was already committed or rolled back.`);
    }
    this.#openReservationByLane[reservation.lane] = null;
  }

  #allocatePage(lane: number, logical: number): number {
    if (this.policy === 'contiguous') {
      const page = this.identityPhysicalPage(lane, logical);
      if (this.#refCount[page] !== 0) {
        this.#counters.allocationFailures++;
        throw new PagedKVError('CAPACITY_EXHAUSTED',
          `Contiguous page ${page} for lane ${lane} logical ${logical} is already resident.`);
      }
      this.#takeFreePage(page);
      this.#refCount[page] = 1;
      this.#recycle(page);
      return page;
    }
    if (this.#freeCount === 0) {
      this.#counters.allocationFailures++;
      throw new PagedKVError('CAPACITY_EXHAUSTED',
        `Page budget of ${this.maxPages} pages is exhausted.`);
    }
    const page = this.#popFreePage();
    this.#refCount[page] = 1;
    this.#recycle(page);
    return page;
  }

  #recycle(page: number): void {
    if (this.clearOnRecycle && this.#clearPage) this.#clearPage(page);
  }

  #releasePage(page: number): void {
    if (this.#refCount[page] <= 0) {
      throw new PagedKVError('STALE_PAGE', `Physical page ${page} was released twice.`);
    }
    this.#refCount[page]--;
    if (this.#refCount[page] > 0) return;
    /* The tag advances only on the transition to unreferenced, so a handle
     * taken while the page was shared stays valid for as long as any holder
     * keeps it. */
    this.#pageGeneration[page]++;
    this.#pushFreePage(page);
  }

  #noteHighWater(): void {
    let resident = 0;
    for (let page = 0; page < this.maxPages; page++) {
      if (this.#refCount[page] > 0) resident++;
    }
    if (resident > this.#highWaterPages) this.#highWaterPages = resident;
  }

  /* Lowest-index-first free list. A binary heap rather than a stack because
   * allocation order has to be reproducible in C from the same operation
   * sequence, and "lowest free page" is a property of the set, not of the
   * order pages happened to be released in. */
  #pushFreePage(page: number): void {
    let index = this.#freeCount++;
    this.#freeHeap[index] = page;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.#freeHeap[parent] <= this.#freeHeap[index]) break;
      const swap = this.#freeHeap[parent];
      this.#freeHeap[parent] = this.#freeHeap[index];
      this.#freeHeap[index] = swap;
      index = parent;
    }
  }

  #popFreePage(): number {
    const page = this.#freeHeap[0];
    this.#freeHeap[0] = this.#freeHeap[--this.#freeCount];
    let index = 0;
    for (;;) {
      const left = index * 2 + 1;
      const right = left + 1;
      let smallest = index;
      if (left < this.#freeCount && this.#freeHeap[left] < this.#freeHeap[smallest]) {
        smallest = left;
      }
      if (right < this.#freeCount && this.#freeHeap[right] < this.#freeHeap[smallest]) {
        smallest = right;
      }
      if (smallest === index) break;
      const swap = this.#freeHeap[smallest];
      this.#freeHeap[smallest] = this.#freeHeap[index];
      this.#freeHeap[index] = swap;
      index = smallest;
    }
    return page;
  }

  /** Remove one specific page from the free heap; used by contiguous pinning. */
  #takeFreePage(page: number): void {
    let index = -1;
    for (let scan = 0; scan < this.#freeCount; scan++) {
      if (this.#freeHeap[scan] === page) { index = scan; break; }
    }
    if (index < 0) {
      this.#counters.allocationFailures++;
      throw new PagedKVError('CAPACITY_EXHAUSTED', `Physical page ${page} is not free.`);
    }
    this.#freeHeap[index] = this.#freeHeap[--this.#freeCount];
    /* Re-heapify from the hole in both directions: the replacement may be
     * smaller than the parent or larger than a child. */
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.#freeHeap[parent] <= this.#freeHeap[index]) break;
      const swap = this.#freeHeap[parent];
      this.#freeHeap[parent] = this.#freeHeap[index];
      this.#freeHeap[index] = swap;
      index = parent;
    }
    for (;;) {
      const left = index * 2 + 1;
      const right = left + 1;
      let smallest = index;
      if (left < this.#freeCount && this.#freeHeap[left] < this.#freeHeap[smallest]) {
        smallest = left;
      }
      if (right < this.#freeCount && this.#freeHeap[right] < this.#freeHeap[smallest]) {
        smallest = right;
      }
      if (smallest === index) break;
      const swap = this.#freeHeap[smallest];
      this.#freeHeap[smallest] = this.#freeHeap[index];
      this.#freeHeap[index] = swap;
      index = smallest;
    }
  }
}
