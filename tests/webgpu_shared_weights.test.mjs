import test from 'node:test';
import assert from 'node:assert/strict';

import { GraphExecutor } from '../ts/backends/GraphExecutor.js';
import { InvariantResourceStore } from '../ts/backends/InvariantResources.js';

globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1, COPY_SRC: 2, COPY_DST: 4, STORAGE: 8, UNIFORM: 16,
});

function mockDevice() {
  const state = { buffers: [], writes: [] };
  return {
    state,
    createBuffer(descriptor) {
      const buffer = {
        descriptor,
        size: descriptor.size,
        usage: descriptor.usage,
        bytes: new Uint8Array(descriptor.size),
        destroyed: false,
        destroy() { this.destroyed = true; },
      };
      state.buffers.push(buffer);
      return buffer;
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        state.writes.push({ buffer, offset, byteLength: data.byteLength });
      },
      onSubmittedWorkDone() { return Promise.resolve(); },
    },
  };
}

function graphOver(hostWeight, weightName = 'w') {
  const weight = {
    name: weightName, dtype: 'float32', shape: [4], sizeBytes: 16,
    isWeight: true, buffer: hostWeight,
  };
  const value = {
    name: 'y', dtype: 'float32', shape: [4], sizeBytes: 16,
    isWeight: false, buffer: null,
  };
  return {
    tensors: new Map([[weightName, weight], ['y', value]]),
    nodes: [],
    getTensor(name) { return this.tensors.get(name); },
  };
}

function compiledDeviceWeights(device, weights) {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const store = new InvariantResourceStore(
    (entry) => entry.capacityBytes,
    (entry) => entry.buffer.destroy(),
  );
  for (const [name, values] of Object.entries(weights)) {
    store.defineLazy(name, () => {
      const capacityBytes = Math.max(4, Math.ceil(values.byteLength / 4) * 4);
      const buffer = device.createBuffer({ label: `Tensor_${name}`, size: capacityBytes, usage });
      device.queue.writeBuffer(buffer, 0, values);
      return Object.freeze({ name, buffer, capacityBytes, usage });
    });
  }
  return store;
}

function openExecutor(device, graph, store) {
  const lease = store.open();
  const invariantWeightBorrow = Object.freeze({
    isContextPrivateWeight() { return false; },
    borrowDeviceWeight(name) { return lease.borrow(name); },
  });
  const executor = new GraphExecutor(device, graph, {
    invariantWeightBorrow,
    shaderLibrary: {},
  });
  return { executor, lease };
}

function stage(executor, graph, replaceWeightNames = new Set()) {
  const staged = executor.resources.stageTensorBuffers(
    graph, undefined, 2, false, [], new Set(), replaceWeightNames,
  );
  executor.gpuBuffers = staged.buffers;
  executor.tensorBufferUsages = staged.usages;
  executor.tensorCapacityBytes = staged.capacities;
  const retired = executor.resources.commitTensorBuffers(staged);
  return { ...staged, retired };
}

function closeContext(context) {
  context.executor.resources.resetCompilationResources();
  context.lease.release();
}

test('N contexts borrow one compiled device weight and keep activations private', () => {
  const device = mockDevice();
  const values = Float32Array.of(1, 2, 3, 4);
  const graph = graphOver(values);
  const owner = compiledDeviceWeights(device, { w: values });
  const contexts = Array.from({ length: 8 }, () => openExecutor(device, graph, owner));
  const staged = contexts.map(({ executor }) => stage(executor, graph));

  assert.equal(owner.borrowerCount, 8);
  assert.equal(owner.resourceCount, 1);
  assert.equal(owner.ownedBytes, 16);
  assert.equal(new Set(staged.map((entry) => entry.buffers.get('w'))).size, 1);
  assert.equal(new Set(staged.map((entry) => entry.buffers.get('y'))).size, 8);
  assert.equal(
    device.state.writes.filter((write) => write.buffer.descriptor.label === 'Tensor_w').length,
    1,
  );

  for (const context of contexts) closeContext(context);
  owner.close();
});

test('close-all then reopen reuses compiled weight until compiled close', () => {
  const device = mockDevice();
  const values = Float32Array.of(1, 2, 3, 4);
  const graph = graphOver(values);
  const owner = compiledDeviceWeights(device, { w: values });

  const first = openExecutor(device, graph, owner);
  const firstBuffer = stage(first.executor, graph).buffers.get('w');
  closeContext(first);
  assert.equal(owner.borrowerCount, 0);
  assert.equal(firstBuffer.destroyed, false);

  const reopened = openExecutor(device, graph, owner);
  assert.equal(stage(reopened.executor, graph).buffers.get('w'), firstBuffer);
  assert.equal(device.state.writes.length, 1);
  closeContext(reopened);
  assert.equal(firstBuffer.destroyed, false);

  owner.close();
  assert.equal(firstBuffer.destroyed, true);
});

test('independent compiled owners never alias even over identical host storage', () => {
  const device = mockDevice();
  const values = Float32Array.of(1, 2, 3, 4);
  const graph = graphOver(values);
  const firstOwner = compiledDeviceWeights(device, { w: values });
  const secondOwner = compiledDeviceWeights(device, { w: values });
  const first = openExecutor(device, graph, firstOwner);
  const second = openExecutor(device, graph, secondOwner);

  const firstBuffer = stage(first.executor, graph).buffers.get('w');
  const secondBuffer = stage(second.executor, graph).buffers.get('w');
  assert.notEqual(firstBuffer, secondBuffer);
  assert.equal(device.state.writes.length, 2);

  closeContext(first);
  closeContext(second);
  firstOwner.close();
  secondOwner.close();
});

test('context cannot fall back to allocating an undeclared invariant copy', () => {
  const device = mockDevice();
  const values = Float32Array.of(1, 2, 3, 4);
  const graph = graphOver(values);
  const owner = compiledDeviceWeights(device, {});
  const context = openExecutor(device, graph, owner);

  assert.throws(
    () => stage(context.executor, graph),
    /resource 'w' is not defined by the compiled model/,
  );
  assert.equal(device.state.buffers.length, 0);
  context.lease.release();
  owner.close();
});

test('context-private bank replacement does not retire compiled invariant storage', async () => {
  const device = mockDevice();
  const values = Float32Array.of(1, 2, 3, 4);
  const graph = graphOver(values);
  const owner = compiledDeviceWeights(device, { w: values });
  const first = openExecutor(device, graph, owner);
  const second = openExecutor(device, graph, owner);
  const shared = stage(first.executor, graph).buffers.get('w');
  assert.equal(stage(second.executor, graph).buffers.get('w'), shared);

  const replacement = stage(second.executor, graph, new Set(['w']));
  for (const lease of replacement.retired) lease.release();
  assert.notEqual(replacement.buffers.get('w'), shared);
  assert.equal(shared.destroyed, false);

  closeContext(first);
  closeContext(second);
  assert.equal(shared.destroyed, false);
  owner.close();
  assert.equal(shared.destroyed, true);
});

test('out-of-order context close never destroys a sibling compiled weight', () => {
  const device = mockDevice();
  const values = Float32Array.of(1, 2, 3, 4);
  const graph = graphOver(values);
  const owner = compiledDeviceWeights(device, { w: values });
  const contexts = Array.from({ length: 4 }, () => openExecutor(device, graph, owner));
  const shared = stage(contexts[0].executor, graph).buffers.get('w');
  for (const context of contexts.slice(1)) {
    assert.equal(stage(context.executor, graph).buffers.get('w'), shared);
  }

  for (const index of [2, 0, 3, 1]) {
    closeContext(contexts[index]);
    assert.equal(shared.destroyed, false);
  }
  assert.equal(owner.borrowerCount, 0);
  owner.close();
  assert.equal(shared.destroyed, true);
});

test('failed rebind rollback keeps sibling and compiled ownership intact', async () => {
  const device = mockDevice();
  const values = Float32Array.of(1, 2, 3, 4);
  const graph = graphOver(values);
  const maxima = new Map([['w', 16], ['y', 16]]);
  const owner = compiledDeviceWeights(device, { w: values });
  const first = openExecutor(device, graph, owner);
  const rejected = openExecutor(device, graph, owner);
  await first.executor.rebindGraph(graph, {
    shapeSignature: 'first', tensorMaximumBytes: maxima,
  });
  const shared = first.executor.gpuBuffers.get('w');

  await assert.rejects(rejected.executor.rebindGraph(graph, {
    shapeSignature: 'rejected',
    tensorMaximumBytes: maxima,
    beforeCommit() { throw new Error('injected rollback'); },
  }), /injected rollback/);
  assert.equal(shared.destroyed, false);
  assert.equal(first.executor.gpuBuffers.get('w'), shared);

  first.executor.dispose();
  first.lease.release();
  rejected.executor.dispose();
  rejected.lease.release();
  assert.equal(shared.destroyed, false);
  owner.close();
  assert.equal(shared.destroyed, true);
});

test('successful bank replacement retires only context-private generation after fence', async () => {
  const device = mockDevice();
  const values = Float32Array.of(1, 2, 3, 4);
  const graph = graphOver(values);
  const maxima = new Map([['w', 16], ['y', 16]]);
  const owner = compiledDeviceWeights(device, { w: values });
  const first = openExecutor(device, graph, owner);
  const second = openExecutor(device, graph, owner);
  await first.executor.rebindGraph(graph, {
    shapeSignature: 'first', tensorMaximumBytes: maxima,
  });
  await second.executor.rebindGraph(graph, {
    shapeSignature: 'second', tensorMaximumBytes: maxima,
  });
  const shared = first.executor.gpuBuffers.get('w');
  assert.equal(second.executor.gpuBuffers.get('w'), shared);

  await second.executor.rebindGraph(graph, {
    shapeSignature: 'second-bank',
    bankResidency: { w: [0] },
    tensorMaximumBytes: maxima,
  });
  await second.executor._awaitPendingRetirement();
  assert.notEqual(second.executor.gpuBuffers.get('w'), shared);
  assert.equal(shared.destroyed, false);

  first.executor.dispose();
  first.lease.release();
  second.executor.dispose();
  second.lease.release();
  assert.equal(shared.destroyed, false);
  owner.close();
  assert.equal(shared.destroyed, true);
});

test('distinct weight slices remain distinct compiled resource identities', () => {
  const device = mockDevice();
  const blob = new ArrayBuffer(32);
  const firstValues = new Float32Array(blob, 0, 4);
  const secondValues = new Float32Array(blob, 16, 4);
  firstValues.set([1, 2, 3, 4]);
  secondValues.set([9, 8, 7, 6]);
  const tensor = (name, buffer) => ({
    name, dtype: 'float32', shape: [4], sizeBytes: 16, isWeight: true, buffer,
  });
  const graph = {
    tensors: new Map([
      ['first', tensor('first', firstValues)],
      ['second', tensor('second', secondValues)],
    ]),
    nodes: [],
  };
  const owner = compiledDeviceWeights(device, { first: firstValues, second: secondValues });
  const context = openExecutor(device, graph, owner);
  const staged = stage(context.executor, graph);
  assert.notEqual(staged.buffers.get('first'), staged.buffers.get('second'));
  assert.equal(device.state.writes.length, 2);
  closeContext(context);
  owner.close();
});
