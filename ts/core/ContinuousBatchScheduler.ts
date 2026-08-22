/* Continuous batching scheduler — implemented via BatchScheduler.
 *
 * Exposes the narrow stateful/Paged-KV adapter used by the shared test vectors
 * while delegating admission, slot/KV lifecycle, and dispatching to BatchScheduler.
 */

import { PagedKVCache, PagedKVError } from './PagedKVCache.js';
import type { KVPagePlan } from '../backends/kvPageAddressing.js';
import {
  BatchScheduler,
  SchedulerError,
  type RequestState,
  type SchedulerFailureCode,
  type BatchSchedulerResult,
  type BatchSchedulerTelemetry,
  type BatchStepWork,
  type BatchStepOutcome,
  type BatchRunStep,
} from './BatchScheduler.js';

export { SchedulerError, PagedKVError };
export type { RequestState, SchedulerFailureCode };

export interface SchedulerOptions {
  readonly cache: PagedKVCache;
  /** Maximum requests waiting for a slot. Beyond it, `submit` refuses. */
  readonly maxQueueDepth?: number;
  /** Bounded recent result window exposed by the inspection API. */
  readonly maxRetainedResults?: number;
  /** Activation tensors backed by the page pool; forwarded on every plan. */
  readonly pagedTensors?: readonly string[];
}

export interface SubmitOptions {
  /** Prompt token count. Reserved atomically before the request is admitted. */
  readonly promptTokens: number;
  /** Upper bound on generated tokens. Bounds the request's KV working set. */
  readonly maxTokens: number;
  /** Opaque caller payload handed back on every step. */
  readonly payload?: unknown;
  /** Identity of this request's prompt, for prefix sharing. */
  readonly promptKey?: string;
}

export interface SchedulerStepWork {
  readonly requestId: number;
  readonly slot: number;
  readonly slotGeneration: number;
  readonly phase: 'prefill' | 'decode';
  readonly position: number;
  readonly tokens: number;
  readonly kvPages: KVPagePlan;
  readonly payload: unknown;
  readonly generated: number;
}

export interface SchedulerStepOutcome {
  readonly value?: unknown;
  readonly finished?: boolean;
  readonly error?: Error;
}

export type SchedulerRunStep = (
  batch: readonly SchedulerStepWork[],
) => readonly SchedulerStepOutcome[] | Promise<readonly SchedulerStepOutcome[]>;

export interface SchedulerResult {
  readonly requestId: number;
  readonly state: 'completed' | 'cancelled' | 'failed';
  readonly values: readonly unknown[];
  readonly generated: number;
  readonly error: Error | null;
}

export interface SchedulerTelemetry {
  readonly admitted: number;
  readonly completed: number;
  readonly cancelled: number;
  readonly failed: number;
  readonly queueDepth: number;
  readonly activeSlots: number;
  readonly freeSlots: number;
  readonly rounds: number;
  readonly steps: number;
  readonly dispatches: number;
  readonly admissionStalls: number;
  readonly maxQueueDepthSeen: number;
  readonly queueDelayRounds: number;
  readonly prefixReuses: number;
  readonly prefixPublications: number;
  readonly prefillTokens: number;
  readonly sharedPromptTokens: number;
}

export class ContinuousBatchScheduler extends BatchScheduler {
  declare readonly cache: PagedKVCache;

  constructor(options: SchedulerOptions) {
    if (!(options?.cache instanceof PagedKVCache)) {
      throw new SchedulerError('INVALID_ARGUMENT', 'A scheduler owns one PagedKVCache.');
    }
    super({
      cache: options.cache,
      maxQueueDepth: options.maxQueueDepth,
      maxRetainedResults: options.maxRetainedResults,
      pagedTensors: options.pagedTensors,
      maxLanes: options.cache.lanes,
      /* The budget is the whole cache: this scheduler's contract is "advance
       * every active lane", so the knob must not cut a round short. */
      tokenBudgetPerDispatch: options.cache.laneTokenCapacity * options.cache.lanes,
    });
  }

  submit(options: SubmitOptions): number {
    return this.submitLLM({
      promptTokens: options?.promptTokens,
      maxTokens: options?.maxTokens,
      promptKey: options?.promptKey,
      payload: options?.payload,
    });
  }

  /**
   * Narrow the contribution work item back to the lane-shaped one this API
   * publishes.
   *
   * Every lane here comes from a slot, so `kvPages` is always present; a
   * stateless submission reaching this adapter would be a caller using the
   * inherited `submitStateless` against a KV-shaped callback, and it is
   * refused rather than handed an undefined page plan.
   */
  #adapt(runStep: SchedulerRunStep): BatchRunStep {
    return (batch) => {
      const adaptedWorks: SchedulerStepWork[] = batch.map((work) => {
        if (!work.kvPages) {
          throw new SchedulerError('INVALID_ARGUMENT',
            'ContinuousBatchScheduler steps lanes backed by the KV cache; ' +
            'submit stateless work through BatchScheduler instead.');
        }
        return {
          requestId: work.requestId,
          slot: work.slot,
          slotGeneration: work.slotGeneration,
          phase: work.phase,
          position: work.position,
          tokens: work.tokens,
          kvPages: work.kvPages,
          payload: work.payload,
          generated: work.generated,
        };
      });
      return runStep(adaptedWorks);
    };
  }

  async step(
    runStep?: SchedulerRunStep | BatchRunStep | null,
    nowMicros?: number,
  ): Promise<boolean> {
    if (typeof runStep !== 'function') {
      throw new SchedulerError('INVALID_ARGUMENT', 'A scheduling round needs a runStep function.');
    }
    return super.step(this.#adapt(runStep as SchedulerRunStep), nowMicros);
  }

  async runUntilIdle(
    runStep?: SchedulerRunStep | BatchRunStep | null,
    maxRounds = 100000,
    nowMicros?: number,
  ): Promise<void> {
    if (typeof runStep !== 'function') {
      throw new SchedulerError('INVALID_ARGUMENT', 'A scheduling round needs a runStep function.');
    }
    return super.runUntilIdle(this.#adapt(runStep as SchedulerRunStep), maxRounds, nowMicros);
  }

  async close(
    runStep: SchedulerRunStep | BatchRunStep | null = null,
    { drain = false }: { drain?: boolean } = {},
    nowMicros?: number,
  ): Promise<void> {
    const adapter = runStep ? this.#adapt(runStep as SchedulerRunStep) : null;
    return super.close(adapter, { drain }, nowMicros);
  }
}
