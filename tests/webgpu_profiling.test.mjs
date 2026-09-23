import assert from 'node:assert/strict';
import test from 'node:test';
import {createWebGPUHostBridge} from '../ts/backends/WebGPUHostBridge.js';

async function fixture(t, supported = true, values = [100n, 123456n]) {
  const previous = ['GPUBufferUsage', 'GPUMapMode', 'GPUShaderStage'].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]);
  Object.defineProperty(globalThis, 'GPUBufferUsage', {configurable: true, value: {QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8}});
  Object.defineProperty(globalThis, 'GPUMapMode', {configurable: true, value: {READ: 1}});
  Object.defineProperty(globalThis, 'GPUShaderStage', {configurable: true, value: {COMPUTE: 1}});
  t.after(() => { for (const [name, descriptor] of previous) descriptor ? Object.defineProperty(globalThis, name, descriptor) : delete globalThis[name]; });
  let map, lose;
  const mapped = new Promise(resolve => {map = resolve;});
  const counts = {queries: 0, buffers: 0, destroyed: 0, writes: 0, resolves: 0, submits: 0, encodes: 0};
  const device = {
    features: new Set(supported ? ['timestamp-query'] : []),
    lost: new Promise(resolve => {lose = resolve;}),
    pushErrorScope() {}, popErrorScope: async () => null,
    queue: {submit() {counts.submits++;}, writeBuffer() {}, onSubmittedWorkDone: async () => {}},
    createQuerySet(desc) {
      assert.equal(desc.type, 'timestamp'); assert.ok(desc.count >= 2 && desc.count <= 2048); counts.queries++;
      return {destroy() {counts.destroyed++;}};
    },
    createBuffer(descriptor) {
      counts.buffers++;
      return {size: descriptor.size, mapAsync: () => mapped, getMappedRange: () => new BigUint64Array(values).buffer,
        unmap() {}, destroy() {counts.destroyed++;}};
    },
    createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createComputePipeline: () => ({getBindGroupLayout: () => ({})}), createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass(desc) {
        if (desc?.timestampWrites) {
          assert.equal(desc.timestampWrites.endOfPassWriteIndex, desc.timestampWrites.beginningOfPassWriteIndex + 1); counts.writes++;
        }
        return {end() {}, setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {counts.encodes++;}};
      },
      resolveQuerySet() {counts.resolves++;}, copyBufferToBuffer() {}, finish: () => ({}),
    }),
  };
  const catalog = {SHADER_NAMES: ['test'], SHADER_CATALOG_HASH: '0'.repeat(64), SHADER_ENTRY_POINTS: ['main'],
    SHADER_LAYOUTS: [{slots: [], writesMask: 0, paramsBinding: 0xffffffff}], loadShaderPack: async () => ['test']};
  const bridge = await createWebGPUHostBridge({device, catalog});
  const events = [], activities = [];
  const memory = new WebAssembly.Memory({initial: 1});
  bridge.attach(memory, (observer, action, resource, low, high) =>
    events.push({observer, action, resource, bytes: BigInt(low) | BigInt(high) << 32n, destroyed: counts.destroyed}),
    (activity, source, destination, bytes, ticket, startUs, endUs) =>
      activities.push({activity, source, destination, bytes, ticket, startUs, endUs}));
  t.after(async () => {lose({reason: 'destroyed', message: 'test'}); bridge.close(); await bridge.waitForCompletion();});
  return {bridge, api: bridge.imports, memory, counts, map, lose, device, events, activities};
}

test('host-only activity observes uploads, submission and asynchronous completion without GPU timestamps', async t => {
  const {api, bridge, memory, counts, activities, device} = await fixture(t);
  for (let i = 0; i < 3; i++) { api.vx_gpu_begin(); api.vx_gpu_ensure(1024, 16, 0); api.vx_gpu_end(); }
  await bridge.waitForCompletion();
  assert.deepEqual(activities, []);
  let complete;
  device.queue.onSubmittedWorkDone = () => new Promise(resolve => { complete = resolve; });
  api.vx_gpu_begin_activity(); api.vx_gpu_ensure(1024, 16, 0); api.vx_gpu_end();
  assert.deepEqual(activities.map(e => e.activity), [1, 2, 4]);
  const [copy, submit, completion] = activities;
  assert.equal(copy.bytes, 16); assert.equal(copy.source, 1); assert.equal(copy.destination, 8);
  assert.ok(copy.endUs >= copy.startUs && submit.endUs >= submit.startUs);
  assert.ok(completion.ticket > 0);
  assert.equal(api.vx_gpu_await_read(completion.ticket, 0), -2);
  new Uint8Array(memory.buffer, 0, 40).fill(77);
  complete(); await bridge.waitForCompletion();
  assert.equal(new Uint8Array(memory.buffer)[0], 77, 'promise completion must not write WASM');
  memory.grow(1);
  assert.equal(api.vx_gpu_await_read(completion.ticket, 65536), 0);
  const result = new DataView(memory.buffer, 65536, 40);
  assert.equal(result.getBigUint64(32, true), BigInt(Math.floor(completion.startUs)) * 1000n);
  assert.ok(result.getBigUint64(0, true) >= 0n);
  assert.equal(result.getUint32(24, true), 0, 'a host completion is not a device interval');
  api.vx_gpu_await_release(completion.ticket);
  assert.equal(api.vx_gpu_await_read(completion.ticket, 0), -1);
  assert.equal(counts.queries, 0);
  assert.equal(counts.submits, 4, 'observation adds no queue submission');
});

test('pending completion tickets are bounded, releaseable and fail on device loss', async t => {
  const {api, bridge, activities, device, lose} = await fixture(t);
  let complete;
  const pending = new Promise(resolve => { complete = resolve; });
  device.queue.onSubmittedWorkDone = () => pending;
  for (let i = 0; i < 129; i++) { api.vx_gpu_begin_activity(); api.vx_gpu_end(); }
  const waits = activities.filter(e => e.activity === 4);
  assert.equal(waits.length, 129);
  assert.ok(waits.slice(0, 128).every(e => e.ticket > 0));
  assert.equal(waits[128].ticket, 0, 'exhaustion is reported synchronously for dropped-event accounting');
  api.vx_gpu_await_release(waits[0].ticket);
  api.vx_gpu_begin_activity(); api.vx_gpu_end();
  assert.ok(activities.at(-1).ticket > 0);
  lose({reason: 'destroyed'}); await bridge.waitForCompletion();
  assert.equal(api.vx_gpu_await_read(waits[1].ticket, 0), -1);
  for (const e of activities) if (e.ticket) api.vx_gpu_await_release(e.ticket);
  complete(); await Promise.resolve();
});

test('device timestamps expose causal start bounds without claiming clock calibration', async t => {
  let now = 10;
  t.mock.method(performance, 'now', () => now++);
  const {api, bridge, memory, map} = await fixture(t, true, [100n, 200n]);
  await Promise.all([bridge.prepareTracing(), bridge.prepareTracing()]);
  const ticket = api.vx_gpu_begin_trace(1, 0);
  api.vx_gpu_end(); map(); await bridge.waitForCompletion();
  assert.equal(api.vx_gpu_trace_read(ticket, 0), 0);
  const result = new DataView(memory.buffer, 0, 40);
  assert.equal(result.getBigUint64(0, true), 100n);
  assert.equal(result.getUint32(24, true), 2);
  assert.ok(result.getBigUint64(8, true) <= result.getBigUint64(16, true));
  assert.equal(result.getBigUint64(32, true), 0n);
  api.vx_gpu_trace_release(ticket);
});

test('disabled collection allocates no query resources; elapsed data writes only on C polling', async t => {
  const {api, bridge, memory, counts, map} = await fixture(t);
  for (let i = 0; i < 100; i++) { api.vx_gpu_begin(); assert.equal(api.vx_gpu_end(), 0); }
  await bridge.waitForCompletion();
  assert.equal(counts.queries, 0); assert.equal(counts.buffers, 0);
  await bridge.prepareTracing();
  const ticket = api.vx_gpu_begin_trace(1, 0); assert.ok(ticket > 0);
  assert.equal(api.vx_gpu_end(), 0);
  assert.equal(api.vx_gpu_trace_read(ticket, 0), -2);
  new BigUint64Array(memory.buffer)[0] = 77n;
  map(); await bridge.waitForCompletion();
  assert.equal(new BigUint64Array(memory.buffer)[0], 77n);
  assert.equal(api.vx_gpu_trace_read(ticket, 0), 0);
  assert.equal(new BigUint64Array(memory.buffer)[0], 123356n);
  assert.deepEqual([counts.queries, counts.buffers, counts.writes, counts.resolves, counts.submits], [4, 8, 1, 1, 101]);
  api.vx_gpu_trace_release(ticket); await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 0);
});

test('early ticket release waits for submission and mapping before reusing query storage', async t => {
  const {api, bridge, counts, map} = await fixture(t);
  await bridge.prepareTracing();
  const ticket = api.vx_gpu_begin_trace(1, 0);
  api.vx_gpu_trace_release(ticket);
  await Promise.resolve(); assert.equal(counts.destroyed, 0);
  api.vx_gpu_end(); await Promise.resolve(); assert.equal(counts.destroyed, 0);
  map(); await bridge.waitForCompletion(); assert.equal(counts.destroyed, 0);
});

test('released but pending queries still consume the device resource limit', async t => {
  const {api, bridge, counts, map} = await fixture(t);
  for (let i = 0; i < 4; i++) {
    await bridge.prepareTracing();
    const ticket = api.vx_gpu_begin_trace(1, 0); assert.ok(ticket > 0);
    api.vx_gpu_trace_release(ticket); assert.equal(api.vx_gpu_end(), 0);
  }
  assert.equal(api.vx_gpu_begin_trace(1, 0), -1);
  assert.equal(api.vx_gpu_end(), 0);
  assert.equal(counts.queries, 4); assert.equal(counts.destroyed, 0);
  assert.equal(counts.submits, 5);
  map(); await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 0);
  const next = api.vx_gpu_begin_trace(1, 0); assert.ok(next > 0);
  assert.equal(api.vx_gpu_end(), 0); api.vx_gpu_trace_release(next);
  await bridge.waitForCompletion(); assert.equal(counts.destroyed, 0);
});

test('unsupported timestamps preserve the ordinary pass', async t => {
  const {api, bridge, counts} = await fixture(t, false);
  await bridge.prepareTracing();
  assert.equal(api.vx_gpu_begin_trace(1, 0), 0);
  assert.equal(api.vx_gpu_end(), 0); await bridge.waitForCompletion();
  assert.equal(counts.queries, 0); assert.equal(counts.submits, 1);
});

test('device loss drains timestamp mapping without waiting for an unresolved map promise', async t => {
  const {api, bridge, counts, lose} = await fixture(t);
  await bridge.prepareTracing();
  const ticket = api.vx_gpu_begin_trace(1, 0); api.vx_gpu_end();
  lose({reason: 'unknown', message: 'injected loss'});
  await bridge.waitForCompletion();
  assert.equal(api.vx_gpu_trace_read(ticket, 0), -3);
  api.vx_gpu_trace_release(ticket); await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 3);
});

test('a timestamp allocation exception preserves the numerical submission', async t => {
  const {api, bridge, counts, device} = await fixture(t);
  device.createBuffer = () => {throw new Error('injected allocation failure');};
  await bridge.prepareTracing();
  assert.equal(api.vx_gpu_begin_trace(1, 0), -1);
  assert.equal(api.vx_gpu_end(), 0); await bridge.waitForCompletion();
  assert.equal(counts.submits, 1); assert.equal(counts.destroyed, 1);
});

for (const failingResource of ['query', 'resolve', 'staging']) {
  test(`asynchronous timestamp ${failingResource} allocation failure preserves output and submission`, async t => {
    const {api, bridge, counts, device, memory, map} = await fixture(t);
    assert.equal(api.vx_gpu_ensure(1024, 16, 0), 1);
    const scopes = [];
    device.pushErrorScope = filter => scopes.push({filter, error: null});
    device.popErrorScope = async () => scopes.pop().error;
    const fail = () => {
      scopes.findLast(scope => scope.filter === 'out-of-memory').error = {message: 'timestamp OOM'};
    };
    const createQuery = device.createQuerySet, createBuffer = device.createBuffer;
    device.createQuerySet = descriptor => {
      if (failingResource === 'query') fail();
      return createQuery(descriptor);
    };
    device.createBuffer = descriptor => {
      if (failingResource === 'resolve' && descriptor.usage & GPUBufferUsage.QUERY_RESOLVE ||
          failingResource === 'staging' && descriptor.usage & GPUBufferUsage.MAP_READ) fail();
      return createBuffer(descriptor); // WebGPU returns an object; it does not throw.
    };
    await bridge.prepareTracing();
    assert.equal(scopes.length, 0);
    device.createBuffer = createBuffer;
    assert.equal(api.vx_gpu_begin_trace(1, 0), -1);
    assert.equal(api.vx_gpu_end(), 0);
    const output = api.vx_gpu_snapshot(1024, 0, 16, 0);
    map(); await bridge.waitForCompletion();
    assert.equal(api.vx_gpu_readback(output, 256, 16), 0);
    assert.deepEqual([...new BigUint64Array(memory.buffer, 256, 2)], [100n, 123456n]);
    assert.equal(counts.writes, 0);
    assert.equal(counts.resolves, 0, 'invalid timing objects never enter the numerical command buffer');
    assert.equal(counts.submits, 2, 'one execution and one output snapshot, with no retry');
    api.vx_gpu_readback_release(output);
  });
}

test('closing during timestamp validation drains and destroys unpublished storage', async t => {
  const {bridge, device, counts} = await fixture(t);
  let finish;
  const validation = new Promise(resolve => { finish = resolve; });
  device.popErrorScope = () => validation;
  const preparing = bridge.prepareTracing();
  assert.equal(counts.queries, 1);
  bridge.close();
  finish(null);
  await preparing; await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 3);
  await bridge.prepareTracing();
  assert.equal(counts.queries, 1, 'closing never publishes or replenishes a pool');
});

test('reversed timestamps are unavailable rather than underflowing to a huge interval', async t => {
  const {api, bridge, map} = await fixture(t, true, [101n, 100n]);
  await bridge.prepareTracing();
  const ticket = api.vx_gpu_begin_trace(1, 0); api.vx_gpu_end(); map(); await bridge.waitForCompletion();
  assert.equal(api.vx_gpu_trace_read(ticket, 0), -1);
});

test('quantized equal timestamps are a measured zero duration', async t => {
  const {api, bridge, map, memory} = await fixture(t, true, [100n, 100n]);
  await bridge.prepareTracing();
  const ticket = api.vx_gpu_begin_trace(1, 0); api.vx_gpu_end(); map(); await bridge.waitForCompletion();
  new BigUint64Array(memory.buffer)[0] = 77n;
  assert.equal(api.vx_gpu_trace_read(ticket, 0), 0);
  assert.equal(new BigUint64Array(memory.buffer)[0], 0n);
});


test('node intervals share one query batch and submission, with independent reads and lifetime', async t => {
  const {api, bridge, memory, counts, map} = await fixture(t, true, [10n, 20n, 100n, 250n]);
  await bridge.prepareTracing();
  const batch = api.vx_gpu_begin_trace(2, 1); assert.ok(batch > 0);
  api.vx_gpu_trace_release(batch); // C releases its admission lease immediately.
  const first = api.vx_gpu_trace_node_begin(); assert.ok(first > 0); api.vx_gpu_trace_node_end();
  const second = api.vx_gpu_trace_node_begin(); assert.ok(second > first); api.vx_gpu_trace_node_end();
  assert.equal(api.vx_gpu_end(), 0);
  assert.equal(counts.queries, 4); assert.equal(counts.buffers, 8);
  assert.equal(counts.writes, 2); assert.equal(counts.submits, 1);
  api.vx_gpu_trace_release(first);
  map(); await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 0, 'the second interval still owns the batch');
  assert.equal(api.vx_gpu_trace_read(second, 0), 0);
  assert.equal(new BigUint64Array(memory.buffer)[0], 150n);
  api.vx_gpu_trace_release(second); await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 0);
});

test('node query exhaustion preserves untimed work and does not invalidate existing intervals', async t => {
  const {api, bridge, counts, map} = await fixture(t);
  await bridge.prepareTracing();
  const batch = api.vx_gpu_begin_trace(1, 1); api.vx_gpu_trace_release(batch);
  const first = api.vx_gpu_trace_node_begin(); api.vx_gpu_trace_node_end();
  assert.equal(api.vx_gpu_trace_node_begin(), -1); api.vx_gpu_trace_node_end();
  assert.equal(api.vx_gpu_end(), 0); map(); await bridge.waitForCompletion();
  assert.equal(api.vx_gpu_trace_read(first, 0), 0);
  api.vx_gpu_trace_release(first); await bridge.waitForCompletion();
  assert.equal(counts.queries, 4); assert.equal(counts.submits, 1); assert.equal(counts.destroyed, 0);
});

test('a node spans its program intervals without adding their durations', async t => {
  const {api, bridge, memory, counts, map} = await fixture(t, true, [10n, 11n, 20n, 40n, 50n, 90n]);
  await bridge.prepareTracing();
  const batch = api.vx_gpu_begin_trace(3, 1); api.vx_gpu_trace_release(batch);
  const node = api.vx_gpu_trace_node_begin();
  const first = api.vx_gpu_trace_program_begin(); api.vx_gpu_trace_program_end();
  const second = api.vx_gpu_trace_program_begin(); api.vx_gpu_trace_program_end();
  api.vx_gpu_trace_node_end(); api.vx_gpu_end(); map(); await bridge.waitForCompletion();
  for (const [ticket, duration] of [[node, 80n], [first, 20n], [second, 40n]]) {
    assert.equal(api.vx_gpu_trace_read(ticket, 0), 0);
    assert.equal(new BigUint64Array(memory.buffer)[0], duration);
    api.vx_gpu_trace_release(ticket);
  }
  assert.equal(counts.submits, 1); assert.equal(counts.writes, 3);
  await bridge.waitForCompletion(); assert.equal(counts.destroyed, 0);
});

test('program query overflow invalidates a partial owning node while preserving numerical commands', async t => {
  const {api, bridge, memory, counts, map} = await fixture(t, true, [10n, 11n, 20n, 40n]);
  new Uint32Array(memory.buffer, 64, 9).set([1, 0, 0, 0, 0xffffffff, 0, 1, 1, 1]);
  await bridge.prepareTracing();
  const batch = api.vx_gpu_begin_trace(2, 1); api.vx_gpu_trace_release(batch);
  const node = api.vx_gpu_trace_node_begin();
  const program = api.vx_gpu_trace_program_begin();
  assert.equal(api.vx_gpu_encode(64), 0); api.vx_gpu_trace_program_end();
  assert.equal(api.vx_gpu_trace_program_begin(), -1);
  assert.equal(api.vx_gpu_encode(64), 0); api.vx_gpu_trace_program_end();
  api.vx_gpu_trace_node_end(); api.vx_gpu_end(); map(); await bridge.waitForCompletion();
  assert.equal(api.vx_gpu_trace_read(node, 0), -1);
  assert.equal(api.vx_gpu_trace_read(program, 0), 0);
  assert.equal(new BigUint64Array(memory.buffer)[0], 20n);
  assert.equal(counts.encodes, 2); assert.equal(counts.submits, 1);
  api.vx_gpu_trace_release(node); api.vx_gpu_trace_release(program);
});

test('an invalid node timestamp does not erase a valid neighbouring interval', async t => {
  const {api, bridge, map} = await fixture(t, true, [20n, 10n, 100n, 110n]);
  await bridge.prepareTracing();
  const batch = api.vx_gpu_begin_trace(2, 1); api.vx_gpu_trace_release(batch);
  const first = api.vx_gpu_trace_node_begin(); api.vx_gpu_trace_node_end();
  const second = api.vx_gpu_trace_node_begin(); api.vx_gpu_trace_node_end();
  api.vx_gpu_end(); map(); await bridge.waitForCompletion();
  assert.equal(api.vx_gpu_trace_read(first, 0), -1);
  assert.equal(api.vx_gpu_trace_read(second, 0), 0);
});

test('device loss retires a shared node batch exactly once after its last ticket', async t => {
  const {api, bridge, counts, lose} = await fixture(t);
  await bridge.prepareTracing();
  const batch = api.vx_gpu_begin_trace(2, 1); api.vx_gpu_trace_release(batch);
  const first = api.vx_gpu_trace_node_begin(); api.vx_gpu_trace_node_end();
  const second = api.vx_gpu_trace_node_begin(); api.vx_gpu_trace_node_end();
  api.vx_gpu_end(); api.vx_gpu_trace_release(first);
  lose({reason: 'unknown', message: 'injected loss'}); await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 0);
  assert.equal(api.vx_gpu_trace_read(second, 0), -3);
  api.vx_gpu_trace_release(second); await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 3);
});


test('auxiliary commands before and after measured nodes retain an untimed compute pass', async t => {
  const {api, bridge, memory, counts, map} = await fixture(t);
  new Uint32Array(memory.buffer, 0, 9).set([1, 0, 0, 0, 0xffffffff, 0, 1, 1, 1]);
  await bridge.prepareTracing();
  const batch = api.vx_gpu_begin_trace(1, 1); api.vx_gpu_trace_release(batch);
  assert.equal(api.vx_gpu_encode(0), 0); // decode feedback before the first node
  const node = api.vx_gpu_trace_node_begin(); assert.ok(node > 0);
  assert.equal(api.vx_gpu_encode(0), 0); api.vx_gpu_trace_node_end();
  assert.equal(api.vx_gpu_encode(0), 0); // a row copy outside the node schedule
  assert.equal(api.vx_gpu_end(), 0); map(); await bridge.waitForCompletion();
  assert.equal(api.vx_gpu_trace_read(node, 64), 0);
  assert.equal(counts.encodes, 3); assert.equal(counts.writes, 1); assert.equal(counts.submits, 1);
  api.vx_gpu_trace_release(node); await bridge.waitForCompletion();
  assert.equal(counts.destroyed, 0);
});


test('memory inventory, buffer reuse and actual deferred destruction have stable identities', async t => {
  const {api, bridge, events} = await fixture(t);
  assert.equal(api.vx_gpu_ensure(256, 16, 0), 1);
  assert.deepEqual(events, []);
  assert.equal(api.vx_gpu_memory_start(7, 32), 1);
  assert.deepEqual(events.map(e => [e.observer, e.action, e.bytes]), [[7, 0, 16n]]);
  const existing = events[0].resource;
  assert.equal(api.vx_gpu_ensure(256, 16, 0), 1); // Reuse has no allocation event.
  assert.equal(events.length, 1);
  assert.equal(api.vx_gpu_ensure(256, 32, 0), 1); // Old allocation retires asynchronously.
  assert.equal(events.length, 2);
  assert.equal(events[1].action, 1);
  assert.notEqual(events[1].resource, existing);
  assert.equal(events[1].bytes, 32n);
  await bridge.waitForCompletion();
  assert.equal(events[2].action, 2);
  assert.equal(events[2].resource, existing);
  assert.equal(events[2].destroyed, 1);
  api.vx_gpu_release(256);
  assert.equal(events.length, 3);
  await bridge.waitForCompletion();
  assert.equal(events[3].resource, events[1].resource);
  assert.equal(events[3].action, 2);
  assert.equal(events[3].destroyed, 2);
  api.vx_gpu_memory_stop(7);
  assert.equal(api.vx_gpu_ensure(256, 16, 0), 1);
  api.vx_gpu_release(256); await bridge.waitForCompletion();
  assert.equal(events.length, 4);
});

test('memory inventory includes released but still owned buffers and reports bounded identity loss', async t => {
  const {api, bridge, events} = await fixture(t);
  api.vx_gpu_ensure(256, 16, 0);
  api.vx_gpu_release(256);
  assert.equal(api.vx_gpu_memory_start(1, 1), 1);
  assert.equal(events.length, 1);
  assert.equal(events[0].action, 0);
  assert.equal(events[0].destroyed, 0);
  api.vx_gpu_ensure(512, 32, 0);
  assert.equal(events[1].action, 3); // The new live identity cannot fit.
  await bridge.waitForCompletion();
  assert.equal(events[2].action, 2);
  assert.equal(events[2].resource, events[0].resource);
  api.vx_gpu_release(512); await bridge.waitForCompletion();
  assert.equal(events.length, 3); // No invented free for an unknown allocation.
  api.vx_gpu_ensure(768, 64, 0);
  assert.equal(events[3].action, 1);
  assert.notEqual(events[3].resource, events[0].resource);
  assert.equal(api.vx_gpu_memory_start(2, 4), 1);
  assert.equal(events[4].observer, 2);
  assert.equal(events[4].action, 0);
  api.vx_gpu_memory_stop(1);
  api.vx_gpu_release(768); await bridge.waitForCompletion();
  assert.deepEqual(events.slice(5).map(e => [e.observer, e.action]), [[2, 2]]);
});

test('memory-only capture requests no timestamps; stopped capture receives no late query-buffer frees', async t => {
  const {api, bridge, events, counts, map} = await fixture(t);
  api.vx_gpu_memory_start(1, 8);
  api.vx_gpu_begin(); api.vx_gpu_end(); await bridge.waitForCompletion();
  assert.equal(counts.queries, 0);
  assert.deepEqual(events, []);
  await bridge.prepareTracing();
  const ticket = api.vx_gpu_begin_trace(1, 0);
  assert.deepEqual(events.map(e => [e.action, e.bytes]), Array.from({length: 8}, () => [1, 16384n]));
  api.vx_gpu_trace_release(ticket);
  api.vx_gpu_end();
  api.vx_gpu_memory_stop(1);
  map(); await bridge.waitForCompletion();
  assert.equal(events.length, 8);
  assert.equal(counts.destroyed, 0);
});
