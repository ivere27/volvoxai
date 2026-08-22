import test from 'node:test';
import assert from 'node:assert/strict';

import { InvariantResourceStore } from '../ts/backends/InvariantResources.js';
import {
  assertProviderInvariantResourceLease,
  assertProviderInvariantResourceOwner,
} from '../ts/backends/BackendProvider.js';

function hostStore() {
  return new InvariantResourceStore((value) => value.byteLength);
}

function deviceStore(destroyed) {
  return new InvariantResourceStore(
    (value) => value.byteLength,
    (value) => { destroyed.push(value.label); },
  );
}

test('compiled lazy producer runs once and N contexts receive one exact resource', () => {
  const store = hostStore();
  let makes = 0;
  store.defineLazy('w', () => {
    makes++;
    return new Uint8Array(1024);
  });

  const leases = Array.from({ length: 8 }, () => store.open());
  const values = leases.map((lease) => lease.borrow('w'));
  assert.equal(makes, 1);
  assert.equal(new Set(values).size, 1);
  assert.equal(store.resourceCount, 1);
  assert.equal(store.ownedBytes, 1024);
  assert.equal(store.borrowerCount, 8);
  for (const lease of leases) {
    assert.equal(lease.borrowedResourceCount, 1);
    assert.equal(lease.borrowedBytes, 1024);
  }
});

test('lease has no producer or mutation authority', () => {
  const store = hostStore();
  store.define('w', new Uint8Array(4));
  const lease = store.open();

  assert.deepEqual(Reflect.ownKeys(lease).sort(), [
    'borrow', 'borrowedBytes', 'borrowedResourceCount', 'deviceEpoch',
    'ownerIdentity', 'protocol', 'release',
  ].sort());
  assert.equal('define' in lease, false);
  assert.equal('defineLazy' in lease, false);
  assert.throws(() => lease.borrow('context-private-copy'), /not defined by the compiled model/);
  assert.equal(store.resourceCount, 1);
});

test('repeated borrow by one context counts one reference and one byte total', () => {
  const store = hostStore();
  store.define('w', new Uint8Array(8));
  const lease = store.open();
  assert.equal(lease.borrow('w'), lease.borrow('w'));
  assert.equal(lease.borrowedResourceCount, 1);
  assert.equal(lease.borrowedBytes, 8);
});

test('zero borrowers retains physical ownership and reopen reuses the same value', () => {
  const destroyed = [];
  const store = deviceStore(destroyed);
  let makes = 0;
  store.defineLazy('w', () => ({
    label: `weight-${++makes}`,
    byteLength: 512,
  }));

  const first = store.open();
  const value = first.borrow('w');
  first.release();
  assert.equal(store.borrowerCount, 0);
  assert.equal(store.resourceCount, 1);
  assert.deepEqual(destroyed, []);

  const reopened = store.open();
  assert.equal(reopened.borrow('w'), value);
  assert.equal(makes, 1);
  reopened.release();
  assert.deepEqual(destroyed, []);

  store.close();
  assert.deepEqual(destroyed, ['weight-1']);
});

test('release is idempotent and compiled close rejects live context leases', () => {
  const store = hostStore();
  store.define('w', new Uint8Array(4));
  const lease = store.open();
  lease.borrow('w');
  assert.throws(() => store.close(), /1 live invariant resource lease/);
  lease.release();
  lease.release();
  assert.equal(store.borrowerCount, 0);
  store.close();
  assert.throws(() => store.open(), /compiled model is closed/);
});

test('device epoch invalidation disposes once and permanently rejects old/new leases', () => {
  const destroyed = [];
  const store = deviceStore(destroyed);
  store.define('w', { label: 'weight', byteLength: 16 });
  const lease = store.open();
  const epoch = lease.deviceEpoch;
  lease.borrow('w');

  store.invalidate();
  assert.notEqual(store.deviceEpoch, epoch);
  assert.equal(store.resourceCount, 0);
  assert.equal(store.ownedBytes, 0);
  assert.deepEqual(destroyed, ['weight']);
  assert.throws(() => lease.borrow('w'), /device epoch is invalid/);
  assert.throws(() => store.open(), /device epoch is invalid/);
  lease.release();
  store.close();
  assert.deepEqual(destroyed, ['weight']);
});

test('failed eager definition is transactional and does not reserve the key', () => {
  const disposed = [];
  const store = new InvariantResourceStore(
    () => -1,
    (value) => disposed.push(value.label),
  );
  assert.throws(() => store.define('w', { label: 'bad' }), /invalid byte size/);
  assert.equal(store.resourceCount, 0);
  assert.equal(store.ownedBytes, 0);
  assert.deepEqual(disposed, ['bad']);
  // The failed definition left no half-published entry.
  store.defineLazy('w', () => ({ label: 'also-bad' }));
});

test('throwing size callback disposes the candidate and leaves the key reusable', () => {
  const disposed = [];
  const store = new InvariantResourceStore(
    () => { throw new Error('injected size failure'); },
    (value) => disposed.push(value.label),
  );
  assert.throws(() => store.define('w', { label: 'candidate' }), /injected size failure/);
  assert.deepEqual(disposed, ['candidate']);
  assert.equal(store.resourceCount, 0);
  assert.equal(store.ownedBytes, 0);
  store.defineLazy('w', () => ({ label: 'retry' }));
});

test('throwing undefined from a lazy producer remains fail-closed', () => {
  const store = hostStore();
  let calls = 0;
  store.defineLazy('w', () => {
    calls++;
    throw undefined;
  });
  const lease = store.open();
  assert.throws(() => lease.borrow('w'));
  assert.throws(() => lease.borrow('w'), /producer previously failed/);
  assert.equal(calls, 1);
  lease.release();
});

test('reentrant lazy production fails closed without publishing a resource', () => {
  const store = hostStore();
  let lease;
  store.defineLazy('w', () => lease.borrow('w'));
  lease = store.open();
  assert.throws(() => lease.borrow('w'), /reentrant producer/);
  assert.equal(store.resourceCount, 0);
  assert.equal(store.ownedBytes, 0);
  assert.throws(() => lease.borrow('w'), /producer previously failed/);
});

test('provider validators bind leases to one exact owner identity and epoch', () => {
  const first = hostStore();
  const second = hostStore();
  const owner = assertProviderInvariantResourceOwner(first, 'fixture');
  const lease = assertProviderInvariantResourceLease(first.open(), owner, 'fixture');
  assert.equal(lease.ownerIdentity, owner.ownerIdentity);
  assert.throws(
    () => assertProviderInvariantResourceLease(second.open(), owner, 'fixture'),
    (error) => error.code === 'ABI_UNSUPPORTED',
  );
});

test('throwing disposer still clears every materialized resource and close stays terminal', () => {
  const attempts = [];
  const store = new InvariantResourceStore(
    (value) => value.byteLength,
    (value) => {
      attempts.push(value.label);
      if (value.label === 'a') throw new Error('injected dispose failure');
    },
  );
  store.define('a', { label: 'a', byteLength: 4 });
  store.define('b', { label: 'b', byteLength: 8 });
  assert.throws(() => store.close(), /injected dispose failure/);
  assert.deepEqual(attempts, ['a', 'b']);
  assert.equal(store.resourceCount, 0);
  assert.equal(store.ownedBytes, 0);
  assert.throws(() => store.open(), /compiled model is closed/);
  store.close();
  assert.deepEqual(attempts, ['a', 'b']);
});
