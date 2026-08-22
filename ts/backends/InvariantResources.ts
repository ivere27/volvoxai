import {
  BACKEND_PROVIDER_INVARIANT_RESOURCE_LEASE_PROTOCOL,
  BACKEND_PROVIDER_INVARIANT_RESOURCE_OWNER_PROTOCOL,
  type BackendProviderInvariantResourceLease,
  type BackendProviderInvariantResourceOwner,
} from './BackendProvider.js';

/** A context can name and borrow a compiled resource, but cannot create one. */
export interface InvariantResourceLease<T> extends BackendProviderInvariantResourceLease {
  borrow(key: string): T;
}

export type InvariantResourceSize<T> = (value: T) => number;

interface Entry<T> {
  readonly make: () => T;
  state: 'unmaterialized' | 'materializing' | 'materialized' | 'failed';
  value: T | undefined;
  size: number;
  refs: number;
  failure: unknown;
}

/**
 * Exact compiled-artifact resource owner.
 *
 * Producer authority (`define`/`defineLazy`) stays on the compiled model. An
 * execution context receives only an `InvariantResourceLease`, whose sole
 * resource operation is `borrow(key)`. Materialized entries remain owned when
 * the borrower count reaches zero and are disposed only by compiled close or
 * explicit epoch invalidation.
 */
export class InvariantResourceStore<T> implements BackendProviderInvariantResourceOwner {
  readonly protocol = BACKEND_PROVIDER_INVARIANT_RESOURCE_OWNER_PROTOCOL;
  readonly ownerIdentity = Object.freeze({});
  readonly #entries = new Map<string, Entry<T>>();
  readonly #size: InvariantResourceSize<T>;
  readonly #dispose: ((value: T) => void) | null;
  #deviceEpoch: object = Object.freeze({});
  #borrowerCount = 0;
  #closed = false;
  #invalidated = false;

  constructor(size: InvariantResourceSize<T>, dispose?: (value: T) => void) {
    if (typeof size !== 'function' || (dispose !== undefined && typeof dispose !== 'function')) {
      throw new TypeError('[InvariantResourceStore] size and dispose must be functions.');
    }
    this.#size = size;
    this.#dispose = dispose ?? null;
  }

  get deviceEpoch(): object { return this.#deviceEpoch; }
  get borrowerCount(): number { return this.#borrowerCount; }

  /** Number of resources that have been physically materialized. */
  get resourceCount(): number {
    let count = 0;
    for (const entry of this.#entries.values()) {
      if (entry.state === 'materialized') count++;
    }
    return count;
  }

  /** Physical bytes owned exactly once by this compiled model. */
  get ownedBytes(): number {
    let total = 0;
    for (const entry of this.#entries.values()) total += entry.size;
    return total;
  }

  /** Install an already materialized resource before any context can borrow it. */
  define(key: string, value: T): void {
    this.#assertDefinition(key);
    const size = this.#checkedSize(value, key);
    this.#entries.set(key, {
      make: () => value,
      state: 'materialized',
      value,
      size,
      refs: 0,
      failure: undefined,
    });
  }

  /**
   * Install a compiled-owned lazy producer. The callback is retained by this
   * owner and is never exposed through a context lease.
   */
  defineLazy(key: string, make: () => T): void {
    this.#assertDefinition(key);
    if (typeof make !== 'function') {
      throw new TypeError('[InvariantResourceStore] key must be non-empty and make a function.');
    }
    this.#entries.set(key, {
      make,
      state: 'unmaterialized',
      value: undefined,
      size: 0,
      refs: 0,
      failure: undefined,
    });
  }

  open(): InvariantResourceLease<T> {
    this.#assertBorrowable();
    const held = new Set<string>();
    const epoch = this.#deviceEpoch;
    const store = this;
    let released = false;
    let borrowedBytes = 0;
    this.#borrowerCount++;
    return Object.freeze({
      protocol: BACKEND_PROVIDER_INVARIANT_RESOURCE_LEASE_PROTOCOL,
      ownerIdentity: this.ownerIdentity,
      deviceEpoch: epoch,
      get borrowedResourceCount() { return held.size; },
      get borrowedBytes() { return borrowedBytes; },
      borrow(key: string): T {
        if (released) {
          throw new Error('[InvariantResourceStore] the invariant resource lease is released.');
        }
        store.#assertBorrowable(epoch);
        const entry = store.#entries.get(key);
        if (entry === undefined) {
          throw new Error(`[InvariantResourceStore] resource '${key}' is not defined by the compiled model.`);
        }
        if (entry.state !== 'materialized') {
          if (entry.state === 'materializing') {
            throw new Error(`[InvariantResourceStore] resource '${key}' has a reentrant producer.`);
          }
          if (entry.state === 'failed') {
            throw new Error(`[InvariantResourceStore] resource '${key}' producer previously failed.`, {
              cause: entry.failure,
            });
          }
          entry.state = 'materializing';
          try {
            const value = entry.make();
            entry.size = store.#checkedSize(value, key);
            entry.value = value;
            entry.state = 'materialized';
          } catch (error) {
            entry.state = 'failed';
            entry.failure = error;
            throw error;
          }
        }
        if (!held.has(key)) {
          held.add(key);
          entry.refs++;
          borrowedBytes += entry.size;
        }
        return entry.value as T;
      },
      release(): void {
        if (released) return;
        released = true;
        for (const key of held) {
          const entry = store.#entries.get(key);
          if (entry !== undefined && entry.refs > 0) entry.refs--;
        }
        held.clear();
        borrowedBytes = 0;
        if (store.#borrowerCount > 0) store.#borrowerCount--;
      },
    });
  }

  /**
   * Permanently invalidate this compiled owner's current device generation.
   * Recovery must publish a new compiled model; this owner cannot reopen.
   */
  invalidate(): void {
    if (this.#closed || this.#invalidated) return;
    this.#invalidated = true;
    this.#deviceEpoch = Object.freeze({});
    this.#disposeMaterialized();
  }

  close(): void {
    if (this.#closed) return;
    if (this.#borrowerCount !== 0) {
      throw new Error(
        `[InvariantResourceStore] cannot close with ${this.#borrowerCount} live invariant resource lease(s).`,
      );
    }
    this.#closed = true;
    try {
      this.#disposeMaterialized();
    } finally {
      this.#entries.clear();
    }
  }

  #checkedSize(value: T, key: string): number {
    let size: number;
    try {
      size = this.#size(value);
    } catch (error) {
      try { this.#dispose?.(value); } catch { /* Preserve the size failure. */ }
      throw error;
    }
    if (!Number.isSafeInteger(size) || size < 0) {
      try { this.#dispose?.(value); } catch { /* Preserve the invalid-size error. */ }
      throw new Error(`[InvariantResourceStore] resource '${key}' has an invalid byte size.`);
    }
    return size;
  }

  #disposeMaterialized(): void {
    let firstError: unknown = undefined;
    let failed = false;
    for (const entry of this.#entries.values()) {
      const value = entry.value;
      const materialized = entry.state === 'materialized';
      entry.value = undefined;
      entry.size = 0;
      entry.refs = 0;
      entry.state = 'unmaterialized';
      if (materialized) {
        try {
          this.#dispose?.(value as T);
        } catch (error) {
          if (!failed) {
            failed = true;
            firstError = error;
          }
        }
      }
    }
    if (failed) throw firstError;
  }

  #assertProducerOpen(): void {
    if (this.#closed) throw new Error('[InvariantResourceStore] the compiled model is closed.');
    if (this.#invalidated) {
      throw new Error('[InvariantResourceStore] the compiled model device epoch is invalid.');
    }
  }

  #assertDefinition(key: string): void {
    this.#assertProducerOpen();
    if (typeof key !== 'string' || key.length === 0) {
      throw new TypeError('[InvariantResourceStore] key must be non-empty.');
    }
    if (this.#entries.has(key)) {
      throw new Error(`[InvariantResourceStore] resource '${key}' is already defined.`);
    }
  }

  #assertBorrowable(expectedEpoch: object = this.#deviceEpoch): void {
    this.#assertProducerOpen();
    if (expectedEpoch !== this.#deviceEpoch) {
      throw new Error('[InvariantResourceStore] the invariant resource lease epoch is invalid.');
    }
  }
}
