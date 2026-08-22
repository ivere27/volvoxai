/* Batch scheduler: contribution admission and token budgets for stateless and stateful workloads.
 *
 * Sits above ExecutionContext and owns the admission, grouping (by GroupKey),
 * token budget dispatching, slot/KV lifecycle, and failure containment.
 *
 * This is the internal batch-policy core, not the production Runtime owner. Its
 * contracts are documented in
 * docs/scheduling-and-dynamic-batching-design.md#internal-batch-scheduler-core:
 *   - Group compatibility includes rows_per_lane until queries are flat-packed.
 *   - Results copy only their own slice.
 *   - Dispatch is work-conserving by default.
 *   - Telemetry reports utilization and padding waste from bounded observations.
 *
 * Two clocks, deliberately separate.
 *
 *   The *policy* clock decides when a `fill_first` group has waited long
 *   enough. It is injectable so a test can drive admission and dispatch
 *   composition without depending on thread scheduling, as required by the
 *   core's determinism contract.
 *
 *   The *telemetry* clock measures `device_busy` and `wall`. It is always the
 *   real monotonic clock, because utilization is a claim about a machine.
 *   Feeding a virtual time into one side of that ratio and a real duration
 *   into the other produces a number that looks like a measurement and is not
 *   one — the earlier version clamped the result to 1.0, which hid it.
 */

import {
  PagedKVCache,
  PagedKVError,
  type PagedKVReservation,
} from './PagedKVCache.js';
import type { KVPagePlan } from '../backends/kvPageAddressing.js';
import { kvPagePlanForLane } from '../backends/kvPageAddressing.js';
import type { RuntimeTypedArray } from '../types.js';

export type RequestKind = 'stateless' | 'prefill' | 'decode';

export type RequestState =
  | 'queued'
  | 'reserved'
  | 'prefill'
  | 'decoding'
  | 'completed'
  | 'cancelled'
  | 'failed';

export type SchedulerFailureCode =
  | 'QUEUE_FULL'
  | 'INVALID_ARGUMENT'
  | 'CANCELLED'
  | 'SCHEDULER_CLOSED'
  | 'STEP_FAILED';

export class SchedulerError extends Error {
  readonly code: SchedulerFailureCode;

  constructor(code: SchedulerFailureCode, message: string) {
    super(message);
    this.name = 'SchedulerError';
    this.code = code;
  }
}

export interface GroupKey {
  readonly modelId: string;
  readonly adapterRevision: string | null;
  readonly shapeSignatureMinusBatch: string;
  readonly rowsPerLane: number;
}

export function formatGroupKey(key: GroupKey): string {
  const component = (value: string | null): string => {
    if (value === null) return '-1:';
    return `${value.length}:${value}`;
  };
  return `${component(key.modelId)}|${component(key.adapterRevision)}|` +
    `${component(key.shapeSignatureMinusBatch)}|${key.rowsPerLane}`;
}

function formatReadyGroupKey(kind: RequestKind, key: GroupKey): string {
  return `${kind.length}:${kind}|${formatGroupKey(key)}`;
}

export interface Contribution {
  readonly requestId: number;
  readonly groupKey: GroupKey;
  readonly groupKeyString: string;
  readonly rows: number;
  readonly kind: RequestKind;
  readonly stateRef: {
    readonly slot: number;
    readonly slotGeneration: number;
    readonly position: number;
    readonly kvPages?: KVPagePlan;
  } | null;
  readonly payload: unknown;
  readonly inputs?: Readonly<Record<string, RuntimeTypedArray>>;
}

export type DispatchPolicy =
  | { readonly mode: 'work_conserving' }
  | { readonly mode: 'fill_first'; readonly maxWaitMicros: number };

export interface BatchSchedulerOptions {
  readonly cache?: PagedKVCache | null;
  /** Maximum requests waiting in queue. Beyond it, `submit` refuses. */
  readonly maxQueueDepth?: number;
  /** Token budget per dispatch. Upper bound on lanes * rows_per_lane. */
  readonly tokenBudgetPerDispatch?: number;
  /** Maximum lanes (batch size) in one dispatch. */
  readonly maxLanes?: number;
  /** Activation tensors backed by the page pool; forwarded on every plan. */
  readonly pagedTensors?: readonly string[];
  /** Dispatch policy. Defaults to work-conserving. */
  readonly dispatchPolicy?: DispatchPolicy;
  /** Batch divisibility constraint (multiple_of). */
  readonly multipleOf?: number;
  /**
   * Policy clock in microseconds: stamps arrivals and decides `fill_first`
   * deadlines. Defaults to the real monotonic clock. Inject one to make
   * dispatch composition reproducible; it never feeds telemetry.
   */
  readonly clock?: () => number;
  /** How many recent queue-depth samples the percentiles are drawn from. */
  readonly queueDepthWindow?: number;
  /** Bounded recent result window retained by the legacy inspection API. */
  readonly maxRetainedResults?: number;
}

export interface SubmitStatelessOptions {
  readonly modelId?: string;
  readonly adapterRevision?: string | null;
  readonly shapeSignatureMinusBatch?: string;
  readonly rowsPerLane?: number;
  readonly inputs?: Readonly<Record<string, RuntimeTypedArray>>;
  readonly payload?: unknown;
}

export interface SubmitLLMOptions {
  readonly modelId?: string;
  readonly adapterRevision?: string | null;
  readonly shapeSignatureMinusBatch?: string;
  readonly promptTokens: number;
  readonly maxTokens: number;
  readonly promptKey?: string | null;
  readonly payload?: unknown;
}

export interface BatchStepWork {
  readonly requestId: number;
  readonly kind: RequestKind;
  /**
   * Rows this contribution occupies on the batch axis: one for a decode token,
   * the chunk length for a prefill, `rowsPerLane` for a stateless item. Equal
   * to the group key's `rowsPerLane` for every member of a dispatch. Keeping
   * it in the compatibility key prevents ragged mixed-query dispatch before a
   * flat-packed query layout exists.
   */
  readonly rows: number;
  readonly slot: number;
  readonly slotGeneration: number;
  readonly phase: 'prefill' | 'decode';
  /** Physical write position within the lane, in logical token order. */
  readonly position: number;
  readonly tokens: number;
  readonly kvPages?: KVPagePlan;
  readonly inputs?: Readonly<Record<string, RuntimeTypedArray>>;
  readonly payload: unknown;
  readonly generated: number;
  /**
   * A bulk-batch row that exists only to satisfy `multiple_of`.
   *
   * It duplicates the first contribution's inputs rather than carrying zeros,
   * because a zero row through a normalization produces NaN and poisons the
   * diagnosis of the rows that mattered. Its outcome is discarded and reaches
   * no request.
   */
  readonly padding: boolean;
}

export interface BatchStepOutcome {
  readonly value?: unknown;
  readonly outputs?: Readonly<Record<string, RuntimeTypedArray>>;
  readonly finished?: boolean;
  readonly error?: Error;
}

export interface DispatchMetadata {
  readonly kind: RequestKind;
  readonly groupKey: GroupKey;
  readonly usefulCount: number;
  readonly paddedCount: number;
  readonly totalRows: number;
  readonly usefulRows: number;
}

export type BatchRunStep = (
  batch: readonly BatchStepWork[],
  metadata?: DispatchMetadata,
) => readonly BatchStepOutcome[] | Promise<readonly BatchStepOutcome[]>;

export interface BatchSchedulerResult {
  readonly requestId: number;
  readonly kind: RequestKind;
  readonly state: 'completed' | 'cancelled' | 'failed';
  readonly values: readonly unknown[];
  readonly outputs?: Readonly<Record<string, RuntimeTypedArray>>;
  readonly generated: number;
  readonly error: Error | null;
}

export interface BatchSchedulerTelemetry {
  readonly dispatches: number;
  readonly rowsDispatched: number;
  readonly rowsUseful: number;
  readonly deviceBusyMicros: number;
  readonly wallMicros: number;
  readonly utilization: number;
  readonly paddingWaste: number;
  readonly queueDepth: number;
  readonly activeSlots: number;
  readonly freeSlots: number;
  readonly rounds: number;
  readonly steps: number;
  readonly admitted: number;
  readonly completed: number;
  readonly cancelled: number;
  readonly failed: number;
  readonly admissionStalls: number;
  readonly maxQueueDepthSeen: number;
  readonly queueDelayRounds: number;
  readonly prefixReuses: number;
  readonly prefixPublications: number;
  readonly prefillTokens: number;
  readonly sharedPromptTokens: number;
  readonly groupCount: number;
  readonly queueDepthP50: number;
  readonly queueDepthP99: number;
}

interface RequestRecord {
  readonly id: number;
  readonly kind: RequestKind;
  readonly modelId: string;
  readonly adapterRevision: string | null;
  readonly shapeSignatureMinusBatch: string;
  readonly rowsPerLane: number;
  readonly promptTokens: number;
  readonly maxTokens: number;
  readonly promptKey: string | null;
  readonly inputs: Readonly<Record<string, RuntimeTypedArray>> | null;
  readonly payload: unknown;
  readonly submittedRound: number;
  readonly submittedTimeMicros: number;
  sharedTokens: number;
  prefilledTokens: number;
  state: RequestState;
  slot: number;
  slotGeneration: number;
  generated: number;
  values: unknown[];
  outputs?: Record<string, RuntimeTypedArray>;
  error: Error | null;
  cancelRequested: boolean;
}

const NO_SLOT = -1;
const DEFAULT_QUEUE_DEPTH_WINDOW = 1024;

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new SchedulerError('INVALID_ARGUMENT', `${label} must be a positive integer.`);
  }
  return value as number;
}

/** Telemetry clock. Real, monotonic, microseconds. Never injectable. */
function monotonicMicros(): number {
  return typeof performance !== 'undefined'
    ? Math.round(performance.now() * 1000)
    : Date.now() * 1000;
}

export function cloneRuntimeArray(value: RuntimeTypedArray): RuntimeTypedArray {
  if (value instanceof Float32Array) return new Float32Array(value);
  if (value instanceof Int32Array) return new Int32Array(value);
  if (value instanceof Int8Array) return new Int8Array(value);
  if (value instanceof Uint8ClampedArray) return new Uint8ClampedArray(value);
  return new Uint8Array(value);
}

export function copySlice(
  source: RuntimeTypedArray,
  startOffset: number,
  length: number,
): RuntimeTypedArray {
  const subarray = source.subarray(startOffset, startOffset + length);
  return cloneRuntimeArray(subarray);
}

export class BatchScheduler {
  readonly cache: PagedKVCache | null;
  readonly slots: number;
  readonly maxQueueDepth: number;
  readonly tokenBudgetPerDispatch: number;
  readonly maxLanes: number;
  readonly pagedTensors: readonly string[];
  readonly dispatchPolicy: DispatchPolicy;
  readonly multipleOf: number;
  readonly queueDepthWindow: number;
  readonly maxRetainedResults: number;

  #nextRequestId = 1;
  #round = 0;
  #closed = false;
  #draining = false;
  #inRound = false;
  #stepTail: Promise<void> = Promise.resolve();

  readonly #requests = new Map<number, RequestRecord>();
  readonly #queue: RequestRecord[] = [];
  readonly #occupants: (RequestRecord | null)[];
  readonly #results: BatchSchedulerResult[] = [];
  readonly #clock: () => number;
  #policyNowOverride: number | null = null;

  /* Wall is the span the scheduler was actually driven over: from the start of
   * the first step to the end of the last. Every dispatch happens inside that
   * span, so device_busy <= wall holds by construction rather than by clamp. */
  #sessionStartMicros = -1;
  #lastActivityMicros = -1;

  /* Fixed-capacity ring. Percentiles over a recent window, not over a list
   * that grows once per round for the life of the process. */
  readonly #queueDepthSamples: number[] = [];
  #queueDepthCursor = 0;

  readonly #counters = {
    dispatches: 0,
    rowsDispatched: 0,
    rowsUseful: 0,
    deviceBusyMicros: 0,
    steps: 0,
    admitted: 0,
    completed: 0,
    cancelled: 0,
    failed: 0,
    admissionStalls: 0,
    maxQueueDepthSeen: 0,
    queueDelayRounds: 0,
    prefixReuses: 0,
    prefixPublications: 0,
    prefillTokens: 0,
    sharedPromptTokens: 0,
  };

  constructor(options: BatchSchedulerOptions = {}) {
    this.cache = options.cache ?? null;
    this.slots = this.cache?.lanes ?? options.maxLanes ?? 8;
    this.maxQueueDepth = options.maxQueueDepth === undefined
      ? this.slots * 4
      : positiveInteger(options.maxQueueDepth, 'maxQueueDepth');
    this.tokenBudgetPerDispatch = options.tokenBudgetPerDispatch === undefined
      ? 2048
      : positiveInteger(options.tokenBudgetPerDispatch, 'tokenBudgetPerDispatch');
    this.maxLanes = options.maxLanes === undefined
      ? this.slots
      : positiveInteger(options.maxLanes, 'maxLanes');
    this.pagedTensors = Object.freeze([...(options.pagedTensors ?? [])]);
    this.dispatchPolicy = options.dispatchPolicy ?? { mode: 'work_conserving' };
    this.multipleOf = options.multipleOf === undefined
      ? 1
      : positiveInteger(options.multipleOf, 'multipleOf');
    if (this.multipleOf > this.maxLanes) {
      throw new SchedulerError('INVALID_ARGUMENT',
        `multipleOf ${this.multipleOf} exceeds maxLanes ${this.maxLanes}.`);
    }
    this.queueDepthWindow = options.queueDepthWindow === undefined
      ? DEFAULT_QUEUE_DEPTH_WINDOW
      : positiveInteger(options.queueDepthWindow, 'queueDepthWindow');
    this.maxRetainedResults = options.maxRetainedResults === undefined
      ? this.maxQueueDepth
      : positiveInteger(options.maxRetainedResults, 'maxRetainedResults');
    this.#clock = options.clock ?? monotonicMicros;
    this.#occupants = new Array(this.slots).fill(null);
  }

  get closed(): boolean { return this.#closed; }
  get queueDepth(): number { return this.#queue.length; }
  get round(): number { return this.#round; }
  get results(): readonly BatchSchedulerResult[] { return this.#results; }

  state(requestId: number): RequestState | null {
    return this.#requests.get(requestId)?.state ??
      this.#results.find((entry) => entry.requestId === requestId)?.state ?? null;
  }

  slotOf(requestId: number): number {
    return this.#requests.get(requestId)?.slot ?? NO_SLOT;
  }

  result(requestId: number): BatchSchedulerResult | null {
    return this.#results.find((entry) => entry.requestId === requestId) ?? null;
  }

  /** Remove one retained result so long-lived callers can acknowledge it. */
  takeResult(requestId: number): BatchSchedulerResult | null {
    const index = this.#results.findIndex((entry) => entry.requestId === requestId);
    if (index < 0) return null;
    return this.#results.splice(index, 1)[0];
  }

  /**
   * Enqueue a stateless request.
   */
  submitStateless(options: SubmitStatelessOptions): number {
    if (this.#closed) {
      throw new SchedulerError('SCHEDULER_CLOSED', 'The scheduler is closed.');
    }
    if (this.#queue.length >= this.maxQueueDepth) {
      throw new SchedulerError('QUEUE_FULL',
        `Queue depth ${this.maxQueueDepth} is full; retry after a request retires.`);
    }
    const modelId = options.modelId ?? 'default';
    const adapterRevision = options.adapterRevision ?? null;
    const rowsPerLane = options.rowsPerLane === undefined
      ? 1
      : positiveInteger(options.rowsPerLane, 'rowsPerLane');
    const minimumRows = rowsPerLane * this.multipleOf;
    if (!Number.isSafeInteger(minimumRows) || minimumRows > this.tokenBudgetPerDispatch) {
      throw new SchedulerError('INVALID_ARGUMENT',
        `rowsPerLane ${rowsPerLane} cannot satisfy multipleOf ${this.multipleOf} ` +
        `inside token budget ${this.tokenBudgetPerDispatch}.`);
    }
    const shapeSignatureMinusBatch = options.shapeSignatureMinusBatch ?? 'stateless';

    const request: RequestRecord = {
      id: this.#nextRequestId++,
      kind: 'stateless',
      modelId,
      adapterRevision,
      shapeSignatureMinusBatch,
      rowsPerLane,
      promptTokens: 0,
      maxTokens: 0,
      promptKey: null,
      inputs: options.inputs ?? null,
      payload: options.payload ?? null,
      submittedRound: this.#round,
      submittedTimeMicros: this.#policyNow(),
      sharedTokens: 0,
      prefilledTokens: 0,
      state: 'queued',
      slot: NO_SLOT,
      slotGeneration: 0,
      generated: 0,
      values: [],
      error: null,
      cancelRequested: false,
    };
    this.#requests.set(request.id, request);
    this.#queue.push(request);
    this.#counters.maxQueueDepthSeen = Math.max(
      this.#counters.maxQueueDepthSeen, this.#queue.length);
    return request.id;
  }

  /**
   * Enqueue a stateful LLM request.
   */
  submitLLM(options: SubmitLLMOptions): number {
    if (this.#closed) {
      throw new SchedulerError('SCHEDULER_CLOSED', 'The scheduler is closed.');
    }
    if (!this.cache) {
      throw new SchedulerError('INVALID_ARGUMENT', 'LLM requests require a scheduler with a PagedKVCache.');
    }
    const promptTokens = positiveInteger(options?.promptTokens, 'promptTokens');
    const maxTokens = positiveInteger(options?.maxTokens, 'maxTokens');
    if (promptTokens + maxTokens > this.cache.laneTokenCapacity) {
      throw new SchedulerError('INVALID_ARGUMENT',
        `Request needs ${promptTokens + maxTokens} tokens over the lane capacity ` +
        `${this.cache.laneTokenCapacity}.`);
    }
    if (this.#queue.length >= this.maxQueueDepth) {
      throw new SchedulerError('QUEUE_FULL',
        `Queue depth ${this.maxQueueDepth} is full; retry after a request retires.`);
    }
    const modelId = options.modelId ?? 'default';
    const adapterRevision = options.adapterRevision ?? null;
    const shapeSignatureMinusBatch = options.shapeSignatureMinusBatch ?? 'llm';

    const request: RequestRecord = {
      id: this.#nextRequestId++,
      kind: 'decode',
      modelId,
      adapterRevision,
      shapeSignatureMinusBatch,
      rowsPerLane: 1,
      promptTokens,
      maxTokens,
      payload: options.payload ?? null,
      promptKey: typeof options.promptKey === 'string' && options.promptKey.length > 0
        ? options.promptKey
        : null,
      inputs: null,
      submittedRound: this.#round,
      submittedTimeMicros: this.#policyNow(),
      sharedTokens: 0,
      prefilledTokens: 0,
      state: 'queued',
      slot: NO_SLOT,
      slotGeneration: 0,
      generated: 0,
      values: [],
      error: null,
      cancelRequested: false,
    };
    this.#requests.set(request.id, request);
    this.#queue.push(request);
    this.#counters.maxQueueDepthSeen = Math.max(
      this.#counters.maxQueueDepthSeen, this.#queue.length);
    return request.id;
  }

  /**
   * Backward-compatible submit: behaves as submitLLM if promptTokens is present, else submitStateless.
   */
  submit(options: SubmitLLMOptions | SubmitStatelessOptions): number {
    if ('promptTokens' in options) {
      return this.submitLLM(options as SubmitLLMOptions);
    }
    return this.submitStateless(options as SubmitStatelessOptions);
  }

  /**
   * Cancel request at any stage.
   */
  cancel(requestId: number): boolean {
    const request = this.#requests.get(requestId);
    if (!request) return false;
    if (request.state === 'completed' || request.state === 'cancelled' ||
        request.state === 'failed') return false;
    if (request.state === 'queued') {
      const index = this.#queue.indexOf(request);
      if (index >= 0) this.#queue.splice(index, 1);
      this.#publish(request, 'cancelled', null);
      return true;
    }
    if (!this.#inRound) {
      this.#release(request);
      this.#publish(request, 'cancelled', null);
      return true;
    }
    request.cancelRequested = true;
    return true;
  }

  /**
   * Advance one scheduling round: admit stateful requests, then dispatch every
   * ready group.
   *
   * Work-conserving means the device does not idle while a ready group waits,
   * so a round dispatches *every* ready group rather than one and then returning.
   * Each group takes at most one dispatch per round, which bounds
   * the round: a group held back by the token budget drains over successive
   * rounds instead of spinning inside one.
   */
  step(
    runStep?: BatchRunStep | null,
    nowMicros?: number,
  ): Promise<boolean> {
    const operation = () => this.#stepRound(runStep, nowMicros);
    const pending = this.#stepTail.then(operation, operation);
    this.#stepTail = pending.then(() => undefined, () => undefined);
    return pending;
  }

  async #stepRound(
    runStep?: BatchRunStep | null,
    nowMicros?: number,
  ): Promise<boolean> {
    if (this.#closed && !this.#draining) {
      throw new SchedulerError('SCHEDULER_CLOSED', 'The scheduler is closed.');
    }
    const stepFn = runStep ?? this.#defaultRunStep;
    if (typeof stepFn !== 'function') {
      throw new SchedulerError('INVALID_ARGUMENT', 'A scheduling step needs a runStep function.');
    }

    const telemetryNow = monotonicMicros();
    if (this.#sessionStartMicros < 0) this.#sessionStartMicros = telemetryNow;

    this.#policyNowOverride = nowMicros ?? null;
    this.#round++;
    this.#inRound = true;
    this.#sampleQueueDepth();

    try {
      const policyNow = this.#policyNow();
      let worked = false;
      const admitted = (this.#draining || !this.cache) ? [] : this.#admitStateful();
      if (admitted.length > 0) worked = true;

      // 1. Prefill is chunked so one prompt can never bypass the dispatch
      // token budget. Stage 1 still dispatches one sequence per prefill call.
      const prefillRequests = this.#occupants.filter(
        (request): request is RequestRecord => request?.state === 'prefill',
      );
      for (const request of prefillRequests) {
        const remaining = request.promptTokens - request.prefilledTokens;
        if (remaining <= 0) {
          request.state = 'decoding';
          continue;
        }
        worked = true;
        const prefillRows = Math.min(remaining, this.tokenBudgetPerDispatch);
        const groupKey: GroupKey = {
          modelId: request.modelId,
          adapterRevision: request.adapterRevision,
          shapeSignatureMinusBatch: request.shapeSignatureMinusBatch,
          rowsPerLane: prefillRows,
        };
        await this.#dispatchBatch(
          [request], 'prefill', groupKey, () => prefillRows,
          stepFn);
      }

      // 2. Dispatch every ready group whose policy lets it go this round.
      const readyGroups = this.#collectReadyGroups(prefillRequests);
      for (const key of this.#groupOrder(readyGroups)) {
        const groupEntries = readyGroups.get(key)!;
        const sample = groupEntries[0];
        const kind = sample.kind;
        const rowsPerLane = kind === 'decode' ? 1 : sample.rowsPerLane;

        if (this.#holdForFill(kind, groupEntries, policyNow)) continue;

        const maxByBudget = Math.floor(this.tokenBudgetPerDispatch / rowsPerLane);
        const limit = Math.min(this.maxLanes, maxByBudget);
        let takeCount = Math.min(groupEntries.length, limit);

        /* Bulk-batch multiple-of contract: with n >= k send floor(n/k)*k and
         * keep the remainder queued. The n < k case is padded inside
         * #dispatchBatch, where a padding row can actually be built. */
        if (this.multipleOf > 1 && takeCount >= this.multipleOf) {
          takeCount = Math.floor(takeCount / this.multipleOf) * this.multipleOf;
        }
        if (takeCount <= 0) continue;

        worked = true;
        const batch = groupEntries.slice(0, takeCount);
        const groupKey: GroupKey = {
          modelId: sample.modelId,
          adapterRevision: sample.adapterRevision,
          shapeSignatureMinusBatch: sample.shapeSignatureMinusBatch,
          rowsPerLane,
        };
        const tokensFor = kind === 'decode'
          ? () => 1
          : (r: RequestRecord) => r.rowsPerLane;

        await this.#dispatchBatch(batch, kind, groupKey, tokensFor, stepFn);
      }

      this.#retireCancelled();
      return worked;
    } finally {
      this.#inRound = false;
      this.#policyNowOverride = null;
      this.#lastActivityMicros = monotonicMicros();
    }
  }

  /**
   * Run steps until all queues are drained and all active slots are retired.
   */
  async runUntilIdle(
    runStep?: BatchRunStep | null,
    maxRounds = 100000,
    nowMicros?: number,
  ): Promise<void> {
    const currentWall = nowMicros;
    for (let round = 0; round < maxRounds; round++) {
      const activeCount = this.#occupants.filter((slot) => slot !== null).length;
      if (this.#queue.length === 0 && activeCount === 0) return;
      const stepWall = currentWall !== undefined ? currentWall + round * 1000 : undefined;
      const worked = await this.step(runStep, stepWall);
      if (!worked && this.#queue.length === 0 && activeCount === 0) return;
    }
    throw new SchedulerError('STEP_FAILED',
      `Scheduler did not reach idle within ${maxRounds} rounds.`);
  }

  /**
   * Close scheduler.
   */
  async close(
    runStep: BatchRunStep | null = null,
    { drain = false }: { drain?: boolean } = {},
    nowMicros?: number,
  ): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;

    // A submitted callback owns its lane/page snapshots through completion.
    // Close is a barrier, never an early lane release racing that callback.
    await this.#stepTail;

    /* Queued work is cancelled either way: with `drain` the *admitted* work
     * finishes, and a request that never reached a slot has nothing to drain.
     * Publishing it is what keeps close from abandoning accepted work
     * silently. */
    const abandoned = [...this.#queue];
    this.#queue.length = 0;
    for (const request of abandoned) this.#publish(request, 'cancelled', null);

    if (drain && runStep) {
      this.#draining = true;
      try {
        while (this.#occupants.some((slot) => slot !== null)) {
          if (!await this.step(runStep, nowMicros)) break;
        }
      } finally {
        this.#draining = false;
      }
    }

    for (let slot = 0; slot < this.slots; slot++) {
      const request = this.#occupants[slot];
      if (!request) continue;
      this.#release(request);
      this.#publish(request, 'cancelled', null);
    }
  }

  telemetry(): Readonly<BatchSchedulerTelemetry> {
    const activeSlots = this.#occupants.reduce(
      (count, slot) => count + (slot ? 1 : 0), 0);
    const wall = this.#sessionStartMicros < 0
      ? 0
      : Math.max(0, this.#lastActivityMicros - this.#sessionStartMicros);
    const utilization = wall > 0 ? this.#counters.deviceBusyMicros / wall : 0;
    const paddingWaste = this.#counters.rowsDispatched > 0
      ? 1.0 - (this.#counters.rowsUseful / this.#counters.rowsDispatched)
      : 0;

    const activeGroups = new Set<string>();
    for (const req of this.#queue) {
      activeGroups.add(formatReadyGroupKey(req.kind, {
        modelId: req.modelId,
        adapterRevision: req.adapterRevision,
        shapeSignatureMinusBatch: req.shapeSignatureMinusBatch,
        rowsPerLane: req.rowsPerLane,
      }));
    }
    for (const req of this.#occupants) {
      if (req && req.state === 'decoding') {
        activeGroups.add(formatReadyGroupKey('decode', {
          modelId: req.modelId,
          adapterRevision: req.adapterRevision,
          shapeSignatureMinusBatch: req.shapeSignatureMinusBatch,
          rowsPerLane: 1,
        }));
      }
    }

    return Object.freeze({
      dispatches: this.#counters.dispatches,
      rowsDispatched: this.#counters.rowsDispatched,
      rowsUseful: this.#counters.rowsUseful,
      deviceBusyMicros: this.#counters.deviceBusyMicros,
      wallMicros: wall,
      utilization,
      paddingWaste,
      queueDepth: this.#queue.length,
      activeSlots,
      freeSlots: this.slots - activeSlots,
      rounds: this.#round,
      steps: this.#counters.steps,
      admitted: this.#counters.admitted,
      completed: this.#counters.completed,
      cancelled: this.#counters.cancelled,
      failed: this.#counters.failed,
      admissionStalls: this.#counters.admissionStalls,
      maxQueueDepthSeen: this.#counters.maxQueueDepthSeen,
      queueDelayRounds: this.#counters.queueDelayRounds,
      prefixReuses: this.#counters.prefixReuses,
      prefixPublications: this.#counters.prefixPublications,
      prefillTokens: this.#counters.prefillTokens,
      sharedPromptTokens: this.#counters.sharedPromptTokens,
      groupCount: activeGroups.size,
      queueDepthP50: this.#queueDepthPercentile(0.5),
      queueDepthP99: this.#queueDepthPercentile(0.99),
    });
  }

  // --- internals -------------------------------------------------------------

  #policyNow(): number {
    return this.#policyNowOverride ?? this.#clock();
  }

  #sampleQueueDepth(): void {
    if (this.#queueDepthSamples.length < this.queueDepthWindow) {
      this.#queueDepthSamples.push(this.#queue.length);
      return;
    }
    this.#queueDepthSamples[this.#queueDepthCursor] = this.#queue.length;
    this.#queueDepthCursor = (this.#queueDepthCursor + 1) % this.queueDepthWindow;
  }

  #queueDepthPercentile(fraction: number): number {
    if (this.#queueDepthSamples.length === 0) return this.#queue.length;
    const sorted = [...this.#queueDepthSamples].sort((a, b) => a - b);
    const index = Math.min(sorted.length - 1, Math.floor(sorted.length * fraction));
    return sorted[index];
  }

  #defaultRunStep: BatchRunStep = (batch) => {
    return batch.map((work) => ({
      value: work.position,
      outputs: work.inputs ? Object.freeze({ ...work.inputs }) : undefined,
    }));
  };

  #admitStateful(): RequestRecord[] {
    if (!this.cache) return [];
    const admitted: RequestRecord[] = [];
    for (let slot = 0; slot < this.slots; slot++) {
      if (this.#occupants[slot] !== null) continue;
      const statefulIndex = this.#queue.findIndex((r) => r.kind === 'decode' && r.state === 'queued');
      if (statefulIndex < 0) break;
      const request = this.#queue[statefulIndex];

      const shared = this.#acquireSharedPrefix(slot, request);
      request.sharedTokens = shared;
      request.prefilledTokens = shared;
      if (shared > 0) this.#counters.prefixReuses++;
      this.#counters.sharedPromptTokens += shared;
      this.#queue.splice(statefulIndex, 1);
      request.state = 'prefill';
      request.slot = slot;
      request.slotGeneration = this.cache.laneGeneration[slot];
      this.#occupants[slot] = request;
      this.#counters.admitted++;
      this.#counters.queueDelayRounds += this.#round - request.submittedRound;
      admitted.push(request);
    }
    return admitted;
  }

  #acquireSharedPrefix(slot: number, request: RequestRecord): number {
    if (!this.cache || !request.promptKey || !this.cache.hasPrefix(request.promptKey)) return 0;
    try {
      const tokens = this.cache.acquirePrefix(request.promptKey, slot);
      if (tokens <= request.promptTokens) return tokens;
      this.cache.releaseLane(slot);
      return 0;
    } catch (error) {
      if (!(error instanceof PagedKVError)) throw error;
      return 0;
    }
  }

  #publishPrompt(request: RequestRecord): void {
    if (!this.cache || !request.promptKey || this.cache.hasPrefix(request.promptKey)) return;
    const pageTokens = this.cache.pageTokens;
    const publishable = Math.floor(request.promptTokens / pageTokens) * pageTokens;
    if (publishable <= 0) return;
    try {
      this.cache.publishPrefix(request.promptKey, request.slot, publishable);
      this.#counters.prefixPublications++;
    } catch (error) {
      if (!(error instanceof PagedKVError)) throw error;
    }
  }

  #collectReadyGroups(justAdmitted: readonly RequestRecord[]): Map<string, RequestRecord[]> {
    const groups = new Map<string, RequestRecord[]>();

    // Decoding stateful requests
    for (let slot = 0; slot < this.slots; slot++) {
      const request = this.#occupants[slot];
      if (!request || request.state !== 'decoding') continue;
      if (justAdmitted.includes(request)) continue;
      const keyStr = formatReadyGroupKey('decode', {
        modelId: request.modelId,
        adapterRevision: request.adapterRevision,
        shapeSignatureMinusBatch: request.shapeSignatureMinusBatch,
        rowsPerLane: 1,
      });
      const list = groups.get(keyStr) ?? [];
      list.push(request);
      groups.set(keyStr, list);
    }

    // Queued stateless requests
    for (const request of this.#queue) {
      if (request.kind !== 'stateless' || request.state !== 'queued') continue;
      const keyStr = formatReadyGroupKey('stateless', {
        modelId: request.modelId,
        adapterRevision: request.adapterRevision,
        shapeSignatureMinusBatch: request.shapeSignatureMinusBatch,
        rowsPerLane: request.rowsPerLane,
      });
      const list = groups.get(keyStr) ?? [];
      list.push(request);
      groups.set(keyStr, list);
    }

    return groups;
  }

  /**
   * Deterministic order over the ready groups: oldest arrival first.
   *
   * Every ready group is dispatched in the round, so this decides sequence
   * rather than who gets served — which is why it needs no cross-round state.
   * The previous round-robin cursor kept an array of every group key ever
   * seen and scanned linearly per round; under adversarial shapes that grew
   * without bound and made the round cost climb with it, violating bounded
   * control-plane backpressure.
   */
  #groupOrder(groups: Map<string, RequestRecord[]>): string[] {
    return Array.from(groups.keys()).sort((left, right) => {
      const leftOldest = Math.min(...groups.get(left)!.map((r) => r.id));
      const rightOldest = Math.min(...groups.get(right)!.map((r) => r.id));
      if (leftOldest !== rightOldest) return leftOldest - rightOldest;
      return left < right ? -1 : left > right ? 1 : 0;
    });
  }

  /**
   * `fill_first`: hold this group back until it fills or its oldest member
   * ages out.
   *
   * The decision is per group. Holding every group because the first one
   * examined was short would let a full batch wait on an unrelated partial
   * one. Decode is never held at all: those lanes are already admitted and
   * holding resident KV, so waiting buys no batching and costs a token of
   * latency per round.
   */
  #holdForFill(
    kind: RequestKind,
    entries: readonly RequestRecord[],
    policyNowMicros: number,
  ): boolean {
    if (this.dispatchPolicy.mode !== 'fill_first') return false;
    if (kind === 'decode') return false;
    if (entries.length >= this.maxLanes) return false;
    const oldest = entries.reduce(
      (min, entry) => Math.min(min, entry.submittedTimeMicros), Infinity);
    return policyNowMicros - oldest < this.dispatchPolicy.maxWaitMicros;
  }

  async #dispatchBatch(
    requests: readonly RequestRecord[],
    kind: RequestKind,
    groupKey: GroupKey,
    tokensFor: (request: RequestRecord) => number,
    runStep: BatchRunStep,
  ): Promise<void> {
    const batch: BatchStepWork[] = [];
    const activeRequests: RequestRecord[] = [];
    const reservations = new Map<number, PagedKVReservation>();

    for (const request of requests) {
      const tokens = tokensFor(request);
      let reservation: PagedKVReservation | null = null;
      if (kind === 'decode' || kind === 'prefill') {
        if (!this.cache) continue;
        try {
          reservation = this.cache.reserve(request.slot, tokens);
          reservations.set(request.id, reservation);
        } catch (error) {
          for (const open of reservations.values()) this.cache.rollback(open);
          for (const member of requests) {
            this.#release(member);
            this.#publish(member, 'failed', error as Error);
          }
          return;
        }
      }

      activeRequests.push(request);

      /* The published length already counts this step's tokens, so the write
       * starts that many tokens back. Zero only for a prefill into an empty
       * lane — a prefill that bound a shared prefix starts *after* it, and
       * writing at zero would overwrite pages other lanes hold. */
      const position = reservation
        ? reservation.priorLength
        : 0;

      const kvPages = reservation && this.cache
        ? kvPagePlanForLane(
          this.cache,
          request.slot,
          this.pagedTensors,
          reservation.priorLength + reservation.tokens,
        )
        : undefined;

      batch.push(Object.freeze({
        requestId: request.id,
        kind,
        rows: groupKey.rowsPerLane,
        slot: request.slot,
        slotGeneration: request.slotGeneration,
        phase: kind === 'prefill' ? 'prefill' : 'decode',
        position,
        tokens,
        kvPages,
        inputs: request.inputs ? Object.freeze({ ...request.inputs }) : undefined,
        payload: request.payload,
        generated: request.generated,
        padding: false,
      }));
    }

    if (batch.length === 0) return;

    // Stateless requests in batch are now dequeued
    if (kind === 'stateless') {
      for (const req of activeRequests) {
        const qIdx = this.#queue.indexOf(req);
        if (qIdx >= 0) this.#queue.splice(qIdx, 1);
      }
    }

    const usefulCount = batch.length;

    /* Bulk-batch padding contract. Only the `[B,...]` path can carry a padding
     * row: it is a duplicate of the first contribution's inputs. A row-decode
     * lane has no inputs to duplicate — it *is* a slot with KV state — and
     * `multiple_of` constrains the bulk batch axis, not the row path, so decode
     * and prefill are dispatched at their true count and report no phantom rows. */
    let paddedCount = usefulCount;
    if (this.multipleOf > 1 && kind === 'stateless' && usefulCount % this.multipleOf !== 0) {
      paddedCount = Math.ceil(usefulCount / this.multipleOf) * this.multipleOf;
      if (paddedCount > this.maxLanes ||
          paddedCount * groupKey.rowsPerLane > this.tokenBudgetPerDispatch) {
        const error = new SchedulerError('INVALID_ARGUMENT',
          'No legal padded batch fits maxLanes and tokenBudgetPerDispatch.');
        for (const request of activeRequests) this.#publish(request, 'failed', error);
        return;
      }
      const template = batch[0];
      for (let extra = usefulCount; extra < paddedCount; extra++) {
        batch.push(Object.freeze({ ...template, requestId: -1, padding: true }));
      }
    }

    const usefulRows = usefulCount * groupKey.rowsPerLane;
    const totalRows = paddedCount * groupKey.rowsPerLane;

    const metadata: DispatchMetadata = Object.freeze({
      kind,
      groupKey,
      usefulCount,
      paddedCount,
      usefulRows,
      totalRows,
    });

    const startMicros = monotonicMicros();
    let outcomes: readonly BatchStepOutcome[];

    try {
      outcomes = await runStep(batch, metadata);
    } catch (error) {
      this.#counters.deviceBusyMicros += Math.max(0, monotonicMicros() - startMicros);
      this.#counters.dispatches++;
      this.#counters.rowsDispatched += totalRows;
      if (this.cache) {
        for (const reservation of reservations.values()) this.cache.rollback(reservation);
      }
      for (const request of activeRequests) {
        if (kind === 'decode' || kind === 'prefill') this.#release(request);
        this.#publish(request, 'failed', error as Error);
      }
      return;
    }
    this.#counters.deviceBusyMicros += Math.max(0, monotonicMicros() - startMicros);
    this.#counters.dispatches++;
    this.#counters.rowsDispatched += totalRows;

    /* One outcome per row, padding included. A callback that returns a
     * different number did not run the batch this scheduler described, and
     * guessing which rows it meant would hand some request another's result. */
    if (!Array.isArray(outcomes) || outcomes.length !== batch.length) {
      /* A row-decode batch is a list of lanes; a bulk batch is a list of rows
       * on the `[B,...]` axis. Naming the unit the caller supplied is what
       * makes the message point at their callback. */
      const unit = kind === 'stateless' ? 'row' : 'lane';
      const error = new SchedulerError('STEP_FAILED',
        `A ${batch.length}-${unit} step returned ` +
        `${Array.isArray(outcomes) ? outcomes.length : 'no'} outcome(s).`);
      if (this.cache) {
        for (const reservation of reservations.values()) this.cache.rollback(reservation);
      }
      for (const request of activeRequests) {
        if (kind === 'decode' || kind === 'prefill') this.#release(request);
        this.#publish(request, 'failed', error);
      }
      return;
    }

    const failedOutcome = outcomes.slice(0, activeRequests.length)
      .find((outcome) => outcome?.error)?.error;
    if (failedOutcome) {
      if (this.cache) {
        for (const reservation of reservations.values()) this.cache.rollback(reservation);
      }
      for (const request of activeRequests) {
        if (kind === 'decode' || kind === 'prefill') this.#release(request);
        this.#publish(request, 'failed', failedOutcome);
      }
      return;
    }

    if (this.cache) {
      try {
        for (const reservation of reservations.values()) this.cache.commit(reservation);
      } catch (error) {
        // Commit performs no fallible allocation. If an invariant nevertheless
        // fails, release every affected lane rather than expose a partial batch.
        for (const reservation of reservations.values()) {
          try { this.cache.rollback(reservation); } catch { /* already committed */ }
        }
        for (const request of activeRequests) {
          if (kind === 'decode' || kind === 'prefill') this.#release(request);
          this.#publish(request, 'failed', error as Error);
        }
        return;
      }
    }

    this.#counters.steps += usefulCount;
    this.#counters.rowsUseful += usefulRows;

    for (const [index, request] of activeRequests.entries()) {
      const outcome = outcomes[index];
      if (kind === 'decode' && this.cache) {
        if (this.cache.laneGeneration[request.slot] !== request.slotGeneration) {
          this.#release(request);
          this.#publish(request, 'failed',
            new SchedulerError('STEP_FAILED',
              `Slot ${request.slot} was reused while request ${request.id} was in flight.`));
          continue;
        }
      }

      if (outcome && Object.prototype.hasOwnProperty.call(outcome, 'value')) {
        request.values.push(outcome.value);
      }

      if (outcome?.outputs) {
        // Clone only own slice
        const clonedOutputs: Record<string, RuntimeTypedArray> = {};
        for (const [name, tensor] of Object.entries(outcome.outputs)) {
          if (ArrayBuffer.isView(tensor)) {
            clonedOutputs[name] = cloneRuntimeArray(tensor as RuntimeTypedArray);
          }
        }
        request.outputs = Object.freeze(clonedOutputs);
      }

      if (kind === 'stateless') {
        this.#publish(request, 'completed', null);
      } else if (kind === 'prefill') {
        request.prefilledTokens += tokensFor(request);
        this.#counters.prefillTokens += tokensFor(request);
        if (request.prefilledTokens >= request.promptTokens) {
          this.#publishPrompt(request);
          request.state = 'decoding';
        } else {
          request.state = 'prefill';
        }
      } else if (kind === 'decode') {
        request.generated++;
        request.state = 'decoding';
        if (outcome?.finished === true || request.generated >= request.maxTokens) {
          this.#release(request);
          this.#publish(request, 'completed', null);
        }
      }
    }
  }

  #retireCancelled(): void {
    for (let slot = 0; slot < this.slots; slot++) {
      const request = this.#occupants[slot];
      if (!request || !request.cancelRequested) continue;
      this.#release(request);
      this.#publish(request, 'cancelled', null);
    }
  }

  #release(request: RequestRecord): void {
    if (request.slot === NO_SLOT || !this.cache) return;
    this.cache.releaseLane(request.slot);
    this.#occupants[request.slot] = null;
    request.slot = NO_SLOT;
  }

  #publish(request: RequestRecord, state: BatchSchedulerResult['state'], error: Error | null): void {
    if (request.state === 'completed' || request.state === 'cancelled' ||
        request.state === 'failed') return;
    request.state = state;
    request.error = error;
    if (state === 'completed') this.#counters.completed++;
    else if (state === 'cancelled') this.#counters.cancelled++;
    else this.#counters.failed++;

    this.#results.push(Object.freeze({
      requestId: request.id,
      kind: request.kind,
      state,
      values: Object.freeze([...request.values]),
      outputs: request.outputs,
      generated: request.generated,
      error,
    }));
    // Completed records no longer retain caller inputs/payloads. The legacy
    // inspection window is bounded; production uses RuntimeRequestHandle.
    this.#requests.delete(request.id);
    while (this.#results.length > this.maxRetainedResults) this.#results.shift();
  }
}
