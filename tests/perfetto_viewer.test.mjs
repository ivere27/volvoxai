import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {openPerfetto} from '../examples/common/Perfetto.js';
import {operatorQueries, traceQueries} from '../examples/common/PerfettoQueries.js';

function queryRows(events, args, queries) {
  // PerfettoSQL uses SQLite. Verify the queries against its slice/args contract.
  const result = spawnSync('python3', ['-c', `
import json, sqlite3, sys
data = json.load(sys.stdin)
db = sqlite3.connect(':memory:')
db.row_factory = sqlite3.Row
db.execute('CREATE TABLE slice(name TEXT, category TEXT, dur INTEGER, arg_set_id INTEGER, id INTEGER, ts INTEGER)')
db.executemany('INSERT INTO slice VALUES (?, ?, ?, ?, ?, ?)', data['events'])
db.create_function('EXTRACT_ARG', 2, lambda ident, key: data['args'][str(ident)].get(key))
print(json.dumps([[dict(row) for row in db.execute(query)] for query in data['queries']]))
db.close()
`], {encoding: 'utf8', input: JSON.stringify({
    events: events.map((e, i) => e.length === 4 ? [...e, i, i * 1_000_000] : e),
    args: Object.fromEntries(args), queries: queries.map(({sql}) => sql),
  })});
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

function fixture() {
  const events = [], args = new Map();
  return {
    add(category, duration, extra = {}, name = 'MatMul', timestamp = events.length * 1_000_000) {
      const id = args.size + 1;
      args.set(id, Object.fromEntries(Object.entries({
        runtimeId: '1', modelId: '2', compiledModelId: '3', contextId: '4',
        backend: 'wasm', phase: 'forward', ...extra,
      }).map(([key, value]) => [`args.${key}`, value])));
      events.push([name, category, duration, id, id, timestamp]);
    },
    rows(title, metadata = {}) {
      return queryRows(events, args, traceQueries(metadata).filter(q => q.title === title))[0];
    },
  };
}

test('operator queries keep host, device, phase and nested scopes separate', () => {
  const events = [];
  const args = new Map();
  function event(category, duration, extra = {}, name = 'MatMul') {
    const id = args.size + 1;
    args.set(id, {
      'args.backend': 'cpu', 'args.phase': 'forward',
      'args.metadataTruncated': 0, ...extra,
    });
    events.push([name, category, duration, id]);
  }
  event('host.node', 2_000_000);
  event('host.node', 4_000_000);
  event('host.node', 1_000_000, {'args.phase': 'backward'});
  event('host.node', 500_000, {'args.backend': 'webgpu'});
  event('host.operation', 100_000_000);
  event('host.program', 100_000_000);
  event('host.wait', 100_000_000);
  event('host.node', 100_000_000, {'args.metadataTruncated': 1});
  event('host.node', -1);
  const device = {'args.backend': 'cuda', 'args.scheduleIndex': 0, 'args.deviceDurationNs': '3000000'};
  event('device.interval', 0, device); // Bounded CUDA observation is an instant.
  event('device.interval', 0, {...device, 'args.deviceDurationNs': '5000000'});
  event('device.interval', 123, {...device, 'args.backend': 'vulkan'}); // Always use device elapsed metadata.
  event('device.interval', 0, {...device, 'args.phase': 'backward'});
  event('device.interval', 0, {...device, 'args.deviceDurationNs': '0', 'args.backend': 'webgpu'});
  event('device.interval', 0, {...device, 'args.scheduleIndex': null}); // Whole pass.
  event('device.program', 0, device);
  event('device.copy', 0, device);
  event('device.interval', 0, {...device, 'args.metadataTruncated': 1});
  event('device.interval', 0, {...device, 'args.deviceDurationNs': null});
  const rows = queryRows(events, args, operatorQueries);
  assert.deepEqual(rows[0], [
    {backend: 'cpu', phase: 'forward', operator: 'MatMul', calls: 2, total_ms: 6, avg_ms: 3, max_ms: 4},
    {backend: 'cpu', phase: 'backward', operator: 'MatMul', calls: 1, total_ms: 1, avg_ms: 1, max_ms: 1},
    {backend: 'webgpu', phase: 'forward', operator: 'MatMul', calls: 1, total_ms: .5, avg_ms: .5, max_ms: .5},
  ]);
  const gpuRows = rows[1].sort((a, b) => `${a.backend}/${a.phase}`.localeCompare(`${b.backend}/${b.phase}`));
  assert.deepEqual(gpuRows, [
    {backend: 'cuda', phase: 'backward', operator: 'MatMul', calls: 1, total_ms: 3, avg_ms: 3, max_ms: 3},
    {backend: 'cuda', phase: 'forward', operator: 'MatMul', calls: 2, total_ms: 8, avg_ms: 4, max_ms: 5},
    {backend: 'vulkan', phase: 'forward', operator: 'MatMul', calls: 1, total_ms: 3, avg_ms: 3, max_ms: 3},
    {backend: 'webgpu', phase: 'forward', operator: 'MatMul', calls: 1, total_ms: 0, avg_ms: 0, max_ms: 0},
  ]);
});

test('slow nodes preserve model, compilation, context and phase; percentages use their own timing domain', () => {
  const f = fixture();
  const node = {scheduleIndex: 0, output: 'layer.0', fused: true};
  f.add('host.node', 2_000_000, node);
  f.add('host.node', 4_000_000, node);
  f.add('host.node', 3_000_000, {...node, scheduleIndex: 1, output: 'layer.1'});
  for (const extra of [{modelId: '20'}, {compiledModelId: '30'}, {contextId: '40'}, {phase: 'backward'}]) {
    f.add('host.node', 10_000_000, {...node, ...extra});
  }
  f.add('host.node', 0, {...node, backend: 'cpu'});
  f.add('host.node', 99_000_000, {...node, metadataTruncated: true});
  f.add('host.program', 99_000_000, node);
  f.add('device.interval', 0, {...node, deviceDurationNs: '8000000'});
  f.add('device.interval', 0, {...node, scheduleIndex: null, deviceDurationNs: '99000000'});
  const rows = f.rows('Slow nodes');
  assert.equal(rows.length, 8);
  const main = rows.find(r => r.domain === 'Host' && r.model === '2' && r.compiled_model === '3'
    && r.context === '4' && r.phase === 'forward' && r.backend === 'wasm' && r.node_index === 0);
  assert.deepEqual(main, {domain: 'Host', backend: 'wasm', model: '2', compiled_model: '3', context: '4',
    phase: 'forward', node_index: 0, output: 'layer.0', operator: 'MatMul', fused: 1,
    calls: 2, total_ms: 6, avg_ms: 3, max_ms: 4, node_time_pct: 66.67});
  assert.equal(rows.find(r => r.node_index === 1).node_time_pct, 33.33);
  assert.equal(rows.find(r => r.domain === 'GPU').node_time_pct, 100);
  assert.equal(rows.find(r => r.backend === 'cpu').node_time_pct, null);
});

test('run statistics and chronological rows distinguish first observed call, outliers and separate contexts', () => {
  const f = fixture();
  const durations = [30, ...Array.from({length: 20}, (_, i) => i + 1), 800];
  // Reverse insertion verifies that first-observed and per-context run numbers use time.
  for (let i = durations.length - 1; i >= 0; i--) {
    f.add('host.operation', durations[i] * 1e6, {executionId: String(9007199254740993n + BigInt(i))}, 'Execute', i * 1e9);
  }
  f.add('host.operation', 2e6, {contextId: '5'}, 'Execute');
  f.add('host.operation', 200e6, {}, 'CompileModel');
  f.add('host.operation', -1, {}, 'Execute');
  f.add('host.operation', 999e6, {metadataTruncated: true}, 'Execute');
  f.add('host.node', 999e6, {}, 'Execute');
  const summaries = f.rows('Run summary');
  const main = summaries.find(r => r.operation === 'Execute' && r.context === '4');
  assert.deepEqual(main, {backend: 'wasm', model: '2', compiled_model: '3', context: '4', operation: 'Execute',
    samples: 22, first_ms: 30, later_avg_ms: 48.095, avg_ms: 47.273, median_ms: 11.5, p95_ms: 30, max_ms: 800});
  const singleton = summaries.find(r => r.operation === 'CompileModel');
  assert.equal(singleton.later_avg_ms, null);
  assert.equal(singleton.median_ms, 200);
  assert.equal(singleton.p95_ms, 200);
  const runs = f.rows('Runs').filter(r => r.operation === 'Execute' && r.context === '4');
  assert.deepEqual(runs.map(r => r.host_ms), durations);
  assert.deepEqual(runs.map(r => r.run), durations.map((_, i) => i + 1));
  assert.equal(runs[0].execution, '9007199254740993');
  assert.equal(runs.at(-1).start_ms, 21000);
});

test('copies and waits retain independent host, device and asynchronous observations and queue identities', () => {
  const f = fixture();
  const copy = {backend: 'cuda', deviceId: '1', queueId: '2', copySource: 1, copyDestination: 8, copyBytes: '1048576'};
  f.add('host.copy', 2e6, copy, 'Upload');
  f.add('host.copy', 3e6, copy, 'Upload');
  f.add('device.copy', 0, {...copy, deviceDurationNs: '1000000'}, 'Upload');
  f.add('host.copy', 9e6, {...copy, queueId: '3'}, 'Upload');
  f.add('host.copy', 99e6, {...copy, metadataTruncated: true}, 'Upload');
  const queue = {backend: 'cuda', deviceId: '1', queueId: '2'};
  f.add('host.wait', 4e6, queue, 'Synchronize');
  f.add('host.await', 6e6, queue, 'Completion');
  f.add('host.await', 7e6, queue, 'Completion');
  f.add('host.await', -1, queue, 'Completion');
  f.add('host.submit', 500000, queue, 'Submit');
  const rows = f.rows('Copies and waits');
  assert.equal(rows.length, 6);
  const host = rows.find(r => r.activity === 'Host copy' && r.queue === '2');
  assert.equal(host.calls, 2);
  assert.equal(host.total_ms, 5);
  assert.equal(host.copy_bytes, 2097152);
  assert.equal(host.source, 'host');
  assert.equal(host.destination, 'device');
  assert.equal(rows.find(r => r.activity === 'GPU copy').total_ms, 1);
  assert.equal(rows.find(r => r.activity === 'Blocking wait').copy_bytes, null);
  assert.equal(rows.find(r => r.activity === 'Async completion').total_ms, 13);
  assert.equal(rows.find(r => r.activity === 'Host submission').total_ms, .5);
});

test('memory uses authoritative peaks, distinguishes missing history and safely quotes metadata', () => {
  const f = fixture();
  const mib = n => String(n * 1048576);
  const malicious = "cuda'); DROP TABLE slice; --";
  f.add('memory.allocation', 0, {liveBytes: mib(4)}, 'host.arena', 1e6);
  f.add('memory.allocation', 0, {liveBytes: mib(8)}, 'host.arena', 3e6);
  f.add('memory.allocation', 0, {liveBytes: mib(6)}, 'host.arena', 5e6);
  f.add('memory.allocation', 0, {liveBytes: mib(8)}, 'host.arena', 7e6);
  f.add('memory.allocation', 0, {liveBytes: mib(16)}, malicious, 2e6);
  const base = {scope: 'runtime', inventory: 'partial', observationStartNs: '1000000',
    existingBytes: mib(4), liveBytes: mib(6), peakBytes: mib(8), allocatedBytes: mib(4),
    freedBytes: mib(2), droppedEvents: '0', accountingComplete: true};
  const metadata = {memory: true, droppedEvents: '2', allocators: [
    {allocator: 'host.arena', ...base},
    {allocator: malicious, ...base, peakBytes: mib(64), droppedEvents: '2'},
    {allocator: 'webgpu.buffer', ...base, scope: 'backend_shared', accountingComplete: false},
  ]};
  const rows = f.rows('Memory', metadata);
  const host = rows.find(r => r.allocator === 'host.arena');
  assert.deepEqual(host, {allocator: 'host.arena', scope: 'runtime', inventory: 'partial',
    accounting: 'complete accounting', dropped_events: 0, observed_from_ms: 1,
    existing_mib: 4, live_mib: 6, growth_mib: 2, peak_mib: 8, peak_at_ms: 3, allocated_mib: 4, freed_mib: 2});
  assert.equal(rows[0].allocator, malicious);
  assert.equal(rows[0].peak_mib, 64, 'missing events must not understate the recorded peak');
  assert.equal(rows[0].peak_at_ms, null, 'cannot locate the first peak with missing history');
  assert.equal(rows.find(r => r.scope === 'backend_shared').accounting, 'tracked subset');
  assert.equal(rows.find(r => r.scope === 'backend_shared').peak_at_ms, null);
  assert.deepEqual(f.rows('Memory', {memory: false}), []);
  const capture = f.rows('Capture', {memory: false, droppedEvents: '2', devices: [{backend: 'cuda', support: 'unavailable'}]});
  assert.equal(capture.find(r => r.section === 'Capture' && r.item === 'Memory').value, 'disabled');
  assert.equal(capture.find(r => r.item === 'Dropped events').value, '2');
  assert.equal(capture.find(r => r.item === 'Timestamp support').value, 'unavailable');
});

function browser(t) {
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'window');
  const listeners = new Set();
  const messages = [];
  const popup = {
    closed: false,
    postMessage(data, origin, transfer) { messages.push({data, origin, transfer}); },
    close() { this.closed = true; },
  };
  const opened = [];
  const window = {
    location: {protocol: 'http:'},
    open(url, target) { opened.push({url, target}); return popup; },
    addEventListener(name, handler) { assert.equal(name, 'message'); listeners.add(handler); },
    removeEventListener(name, handler) { assert.equal(name, 'message'); listeners.delete(handler); },
  };
  Object.defineProperty(globalThis, 'window', {configurable: true, value: window});
  t.mock.timers.enable({apis: ['setInterval', 'setTimeout']});
  t.after(() => {
    if (saved) Object.defineProperty(globalThis, 'window', saved);
    else delete globalThis.window;
    assert.equal(listeners.size, 0, 'message listeners must be retired');
  });
  return {window, popup, messages, opened, listeners,
    pong(origin = 'https://ui.perfetto.dev', source = popup) {
      for (const listener of listeners) listener({data: 'PONG', origin, source});
    },
  };
}

test('opens all analysis tabs and sends original bytes only to the ready Perfetto window', async t => {
  const b = browser(t);
  const blob = new Blob(['{"traceEvents":[]}']);
  const opening = openPerfetto(blob, 'receipt.json');
  assert.equal(b.opened.length, 1, 'popup must open before yielding the click handler');
  assert.equal(b.opened[0].url, 'about:blank');
  await nextTurn();
  const commands = JSON.parse(new URLSearchParams(new URL(b.popup.location).hash.split('?')[1]).get('startupCommands'));
  assert.deepEqual(commands.map(command => [command.id, command.args[1]]), [
    ['dev.perfetto.RunQueryAndShowTab', 'Capture'],
    ['dev.perfetto.RunQueryAndShowTab', 'Memory'],
    ['dev.perfetto.RunQueryAndShowTab', 'Copies and waits'],
    ['dev.perfetto.RunQueryAndShowTab', 'GPU operators'],
    ['dev.perfetto.RunQueryAndShowTab', 'Host operators'],
    ['dev.perfetto.RunQueryAndShowTab', 'Runs'],
    ['dev.perfetto.RunQueryAndShowTab', 'Run summary'],
    ['dev.perfetto.RunQueryAndShowTab', 'Slow nodes'],
  ]);
  b.pong('https://unrelated.example');
  b.pong('https://ui.perfetto.dev', {});
  await Promise.resolve();
  assert.ok(b.messages.every(message => message.data === 'PING'));
  b.pong();
  await opening;
  const sent = b.messages.at(-1);
  assert.equal(sent.origin, 'https://ui.perfetto.dev');
  assert.equal(new TextDecoder().decode(sent.data.perfetto.buffer), await blob.text());
  assert.equal(sent.data.perfetto.fileName, 'receipt.json');
  assert.equal(sent.data.perfetto.shareable, false);
  assert.equal(sent.data.perfetto.downloadable, true);
  assert.deepEqual(sent.transfer, [sent.data.perfetto.buffer]);
  const count = b.messages.length;
  t.mock.timers.tick(60_000);
  assert.equal(b.messages.length, count);
  assert.equal(b.popup.closed, false);
});

test('reports blocked popups and unsupported local-file pages', async t => {
  const b = browser(t);
  b.window.open = () => null;
  await assert.rejects(openPerfetto(new Blob(['trace'])), /Allow pop-ups/);
  b.window.location.protocol = 'file:';
  await assert.rejects(openPerfetto(new Blob(['trace'])), /HTTP/);
  await assert.rejects(openPerfetto(new Blob([])), /Capture or choose/);
});

test('stops waiting if the viewer tab closes', async t => {
  const b = browser(t);
  const rejection = assert.rejects(openPerfetto(new Blob(['{"traceEvents":[]}'])), /Perfetto closed/);
  await nextTurn();
  b.popup.closed = true;
  t.mock.timers.tick(200);
  await rejection;
});

test('timeouts stop timers and retire the unresponsive viewer', async t => {
  const b = browser(t);
  const rejection = assert.rejects(openPerfetto(new Blob(['{"traceEvents":[]}'])), /did not respond/);
  await nextTurn();
  t.mock.timers.tick(30_000);
  await rejection;
  const count = b.messages.length;
  t.mock.timers.tick(30_000);
  assert.equal(b.messages.length, count);
  assert.equal(b.popup.closed, true);
});

test('failed file reads retire the blank tab before starting the handshake', async t => {
  const b = browser(t);
  await assert.rejects(openPerfetto({size: 1, arrayBuffer: async () => {throw new Error('read failed');}}), /read failed/);
  assert.equal(b.popup.closed, true);
  assert.equal(b.listeners.size, 0);
});

test('BASIC captures open on run summary and invalid files retire the blank tab', async t => {
  const b = browser(t);
  const opening = openPerfetto(new Blob(['{"otherData":{"detail":"basic"},"traceEvents":[]}']));
  await nextTurn();
  const commands = JSON.parse(new URLSearchParams(new URL(b.popup.location).hash.split('?')[1]).get('startupCommands'));
  assert.equal(commands.at(-1).args[1], 'Run summary');
  b.pong();
  await opening;
  await assert.rejects(openPerfetto(new Blob(['{"unrelated":[]}'])), /Chrome Trace JSON/);
  assert.equal(b.popup.closed, true);
});

test('captures with lost events open on measurement coverage first', async t => {
  const b = browser(t);
  const opening = openPerfetto(new Blob(['{"otherData":{"detail":"nodes","droppedEvents":"1"},"traceEvents":[]}']));
  await nextTurn();
  const commands = JSON.parse(new URLSearchParams(new URL(b.popup.location).hash.split('?')[1]).get('startupCommands'));
  assert.equal(commands.at(-1).args[1], 'Capture');
  b.pong();
  await opening;
});
