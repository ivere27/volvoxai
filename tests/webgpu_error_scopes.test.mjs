import test from 'node:test';
import assert from 'node:assert/strict';

import {
  captureWebGPUErrorScopesSync,
  runWebGPUErrorScopedSync,
} from '../ts/backends/WebGPUErrorScopes.js';
import { GraphExecutor } from '../ts/backends/GraphExecutor.js';
import { snapshotWebGPUOutputs } from '../ts/backends/WebGPUResults.js';
import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';

globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  STORAGE: 8,
  UNIFORM: 16,
});
globalThis.GPUMapMode ??= Object.freeze({ READ: 1 });

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, decline) => {
    resolve = accept;
    reject = decline;
  });
  return { promise, resolve, reject };
}

function scopedDevice({ deferredPops = false } = {}) {
  const stack = [];
  const pending = [];
  const log = [];
  const device = {
    log,
    pending,
    deferPops: deferredPops,
    pushErrorScope(filter) {
      log.push(`push:${filter}`);
      stack.push({ filter, error: null });
    },
    popErrorScope() {
      const scope = stack.pop();
      if (!scope) throw new Error('mock error-scope stack underflow');
      log.push(`pop:${scope.filter}`);
      if (!device.deferPops) return Promise.resolve(scope.error);
      const completion = deferred();
      pending.push({ ...completion, scope });
      return completion.promise;
    },
    report(filter, message) {
      for (let index = stack.length - 1; index >= 0; index--) {
        if (stack[index].filter !== filter) continue;
        stack[index].error = { name: filter === 'out-of-memory'
          ? 'GPUOutOfMemoryError' : 'GPUValidationError', message };
        return;
      }
      throw new Error(`mock uncaptured ${filter}: ${message}`);
    },
  };
  return device;
}

test('WebGPU error scopes use nested LIFO order without spanning an await', async () => {
  const device = scopedDevice({ deferredPops: true });
  const captured = captureWebGPUErrorScopesSync(device, { label: 'candidate' }, () => {
    device.log.push('operation');
    return 17;
  });

  assert.equal(captured.value, 17);
  assert.deepEqual(device.log, [
    'push:validation',
    'push:out-of-memory',
    'operation',
    'pop:out-of-memory',
    'pop:validation',
  ]);
  assert.equal(device.pending.length, 2);
  for (const completion of device.pending) completion.resolve(null);
  await captured.check;
});

test('WebGPU scoped OOM and validation errors reject with stable public codes', async () => {
  const oomDevice = scopedDevice();
  await assert.rejects(
    runWebGPUErrorScopedSync(oomDevice, { label: 'tensor generation' }, () => {
      oomDevice.report('out-of-memory', 'not enough memory left');
      return { invalid: true };
    }),
    (error) => error?.code === 'OUT_OF_MEMORY' &&
      error?.phase === 'execution' && /not enough memory left/.test(error.message),
  );

  const validationDevice = scopedDevice();
  await assert.rejects(
    runWebGPUErrorScopedSync(validationDevice, { label: 'bind group' }, () => {
      validationDevice.report('validation', 'Tensor_v1 is invalid');
      return { invalid: true };
    }),
    (error) => error?.code === 'EXECUTION_FAILED' &&
      error?.backend === 'webgpu' && /Tensor_v1 is invalid/.test(error.message),
  );
});

test('WebGPU error-scope cleanup preserves direct throws and rejects partial mocks', async () => {
  const device = scopedDevice();
  const direct = new Error('synchronous createBuffer failure');
  assert.throws(
    () => captureWebGPUErrorScopesSync(device, { label: 'allocation' }, () => {
      throw direct;
    }),
    (error) => error === direct,
  );
  assert.deepEqual(device.log.slice(-2), [
    'pop:out-of-memory',
    'pop:validation',
  ]);

  const unsupported = {};
  assert.equal(await runWebGPUErrorScopedSync(
    unsupported, { label: 'scope-free test device' }, () => 9), 9);
  assert.throws(
    () => captureWebGPUErrorScopesSync(
      { pushErrorScope() {} }, { label: 'partial test device' }, () => 0),
    (error) => error?.code === 'EXECUTION_FAILED' && /paired/.test(error.message),
  );
});

test('WebGPU error-scope pop rejection reports device loss', async () => {
  const device = scopedDevice({ deferredPops: true });
  const captured = captureWebGPUErrorScopesSync(device, { label: 'lost allocation' }, () => 1);
  device.pending[0].reject(new Error('GPU device was lost while resolving the scope'));
  device.pending[1].resolve(null);
  await assert.rejects(captured.check,
    (error) => error?.code === 'DEVICE_LOST' && error?.backend === 'webgpu');
});

test('concurrent callers pop complete scope stacks before either completion settles', async () => {
  const device = scopedDevice({ deferredPops: true });
  const first = captureWebGPUErrorScopesSync(device, { label: 'first' }, () => {
    device.log.push('operation:first');
    return 'first';
  });
  const second = captureWebGPUErrorScopesSync(device, { label: 'second' }, () => {
    device.log.push('operation:second');
    return 'second';
  });
  assert.deepEqual(device.log, [
    'push:validation', 'push:out-of-memory', 'operation:first',
    'pop:out-of-memory', 'pop:validation',
    'push:validation', 'push:out-of-memory', 'operation:second',
    'pop:out-of-memory', 'pop:validation',
  ]);
  // Resolve the later operation first to prove promise completion order does
  // not own or disturb the already-balanced device stack.
  device.pending[2].resolve(null);
  device.pending[3].resolve(null);
  await second.check;
  device.pending[0].resolve(null);
  device.pending[1].resolve(null);
  await first.check;
});

function graphWithInputElements(elements) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('x', [elements], 'float32');
  graph.setOutputs([input.name]);
  return graph;
}

function executorDevice() {
  const scopes = scopedDevice();
  const state = {
    buffers: [],
    failBufferLabel: null,
    failKind: 'out-of-memory',
    submissions: [],
  };
  return Object.assign(scopes, {
    state,
    limits: { maxComputeWorkgroupsPerDimension: 65535 },
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
      if (state.failBufferLabel === descriptor.label) {
        state.failBufferLabel = null;
        scopes.report(state.failKind,
          state.failKind === 'out-of-memory' ? 'not enough memory left' : `${descriptor.label} is invalid`);
      }
      return buffer;
    },
    createCommandEncoder() {
      const copies = [];
      return {
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          copies.push({ source, sourceOffset, destination, destinationOffset, size });
        },
        finish() { return { copies }; },
      };
    },
    queue: {
      writeBuffer() {},
      submit(commandBuffers) { state.submissions.push(commandBuffers); },
      onSubmittedWorkDone() { return Promise.resolve(); },
    },
  });
}

test('scoped tensor-generation OOM rolls back and never publishes the invalid buffer', async () => {
  const firstGraph = graphWithInputElements(4);
  const secondGraph = graphWithInputElements(8);
  const device = executorDevice();
  const executor = new GraphExecutor(device, firstGraph, { shaderLibrary: {} });
  const maxima = new Map([['x', 32]]);

  await executor.rebindGraph(firstGraph, {
    shapeSignature: 'x:[4]', tensorMaximumBytes: maxima,
  });
  const committed = executor.gpuBuffers.get('x');
  device.state.failBufferLabel = 'Tensor_x';
  await assert.rejects(executor.rebindGraph(secondGraph, {
    shapeSignature: 'x:[8]', tensorMaximumBytes: maxima,
  }), (error) => error?.code === 'OUT_OF_MEMORY' && /not enough memory left/.test(error.message));

  const rejected = device.state.buffers.at(-1);
  assert.equal(rejected.destroyed, true);
  assert.equal(executor.currentShapeSignature, 'x:[4]');
  assert.equal(executor.gpuBuffers.get('x'), committed);
  assert.equal(committed.destroyed, false);
  executor.dispose();
});

test('scoped tensor-generation validation failure also preserves the committed generation', async () => {
  const firstGraph = graphWithInputElements(4);
  const secondGraph = graphWithInputElements(8);
  const device = executorDevice();
  const executor = new GraphExecutor(device, firstGraph, { shaderLibrary: {} });
  const maxima = new Map([['x', 32]]);
  await executor.rebindGraph(firstGraph, {
    shapeSignature: 'x:[4]', tensorMaximumBytes: maxima,
  });
  const committed = executor.gpuBuffers.get('x');

  device.state.failKind = 'validation';
  device.state.failBufferLabel = 'Tensor_x';
  await assert.rejects(executor.rebindGraph(secondGraph, {
    shapeSignature: 'x:[8]', tensorMaximumBytes: maxima,
  }), (error) => error?.code === 'EXECUTION_FAILED' && /Tensor_x is invalid/.test(error.message));
  assert.equal(device.state.buffers.at(-1).destroyed, true);
  assert.equal(executor.currentShapeSignature, 'x:[4]');
  assert.equal(executor.gpuBuffers.get('x'), committed);
  executor.dispose();
});

test('a candidate remains private while its already-popped allocation scopes are unresolved', async () => {
  const firstGraph = graphWithInputElements(4);
  const secondGraph = graphWithInputElements(8);
  const device = executorDevice();
  const executor = new GraphExecutor(device, firstGraph, { shaderLibrary: {} });
  const maxima = new Map([['x', 32]]);
  await executor.rebindGraph(firstGraph, {
    shapeSignature: 'x:[4]', tensorMaximumBytes: maxima,
  });
  const committed = executor.gpuBuffers.get('x');

  device.deferPops = true;
  device.state.failBufferLabel = 'Tensor_x';
  const rejectedRebind = executor.rebindGraph(secondGraph, {
    shapeSignature: 'x:[8]', tensorMaximumBytes: maxima,
  });
  for (let turn = 0; turn < 4 && device.pending.length === 0; turn++) {
    await Promise.resolve();
  }
  assert.equal(device.pending.length, 2,
    'both scopes were synchronously popped before rebind yielded');
  assert.equal(executor.currentShapeSignature, 'x:[4]');
  assert.equal(executor.gpuBuffers.get('x'), committed,
    'the invalid candidate is not visible while pop promises remain unresolved');

  for (const completion of device.pending.splice(0)) {
    completion.resolve(completion.scope.error);
  }
  await assert.rejects(rejectedRebind, (error) => error?.code === 'OUT_OF_MEMORY');
  assert.equal(executor.currentShapeSignature, 'x:[4]');
  assert.equal(executor.gpuBuffers.get('x'), committed);
  executor.dispose();
});

test('scoped result-snapshot OOM destroys invalid snapshots before returning', async () => {
  const graph = graphWithInputElements(4);
  const device = executorDevice();
  const source = device.createBuffer({
    label: 'Tensor_x', size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  device.state.failBufferLabel = 'Result_x';
  await assert.rejects(snapshotWebGPUOutputs(device, graph, new Map([['x', source]])),
    (error) => error?.code === 'OUT_OF_MEMORY' && /result snapshot/i.test(error.message));
  const rejected = device.state.buffers.at(-1);
  assert.equal(rejected.descriptor.label, 'Result_x');
  assert.equal(rejected.destroyed, true);
  assert.equal(device.state.submissions.length, 0,
    'allocation scopes reject invalid result buffers before any copy submission');
});
