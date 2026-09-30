import assert from 'node:assert/strict';
import test from 'node:test';
import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';
import {summarizeProfile, profileEvidence} from '../examples/common/workbench/ProfileData.js';
import {startEngineTrace} from '../examples/common/EngineTrace.js';

const available = pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE;
const unsupported = pb.ObservationStatus.OBSERVATION_STATUS_UNSUPPORTED;
const trace = overrides => ({info: new pb.TraceInfo(), events: [], resources: [], plans: [], ...overrides});
const event = overrides => new pb.TraceEvent({planId: 17n, backend: 'webgpu', trackId: 2n, name: 'measured work', activity: pb.TraceActivity.TRACE_ACTIVITY_COMPUTE,
  node: {scheduleIndex: 3, outputName: 'features', fused: true}, ...overrides});

test('missing capture sections report a reload action instead of becoming empty measurements or a TypeError', () => {
  const mixedCapture = {info: new pb.TraceInfo({options: {utilization: true}, resourceSnapshots: {count: 3n}}), events: [], blob: new Blob()};
  assert.throws(() => summarizeProfile(mixedCapture, pb), /missing resources, plans\. Reload the page/);
  for (const section of ['events', 'resources', 'plans']) {
    const partial = trace(); delete partial[section];
    assert.throws(() => summarizeProfile(partial, pb), new RegExp(`missing ${section}\\.`));
  }
  assert.doesNotThrow(() => summarizeProfile(trace(), pb), 'empty sections are valid; omitted sections are not');
});

test('profile projections join plans and never double count nested host or GPU intervals', () => {
  const plan = new pb.ExecutionPlan({planId: 17n, steps: [{scheduleIndex: 3, operatorName: 'FusedConv', sourceNodeIds: ['conv', 'relu']}],
    tensors: [{name: 'features', logicalBytes: 64n, shape: [4n, 4n], offsetBytes: 0n}]});
  const summary = summarizeProfile(trace({plans: [plan], events: [
    event({host: {startNs: 1_000_000n, durationNs: 10_000_000n}}),
    event({program: {name: 'kernel'}, host: {durationNs: 8_000_000n}}),
    event({device: {elapsedNs: 9_000_000n}}),
    event({program: {name: 'kernel'}, device: {elapsedNs: 4_000_000n}}),
    event({program: {name: 'kernel'}, device: {elapsedNs: 3_000_000n}}),
    event({activity: pb.TraceActivity.TRACE_ACTIVITY_SYNCHRONIZE, host: {durationNs: 20_000_000n}}),
    event({metadataTruncated: true, host: {durationNs: 30_000_000n}}),
  ]}), pb);
  assert.equal(summary.nodes.length, 1);
  const row = summary.nodes[0];
  assert.equal(summary.hostMs, 10);
  assert.equal(row.deviceMs, 7);
  assert.equal(row.deviceIntervals, 2);
  assert.equal(row.deviceBasis, 'program intervals');
  assert.equal(row.calls, 1);
  assert.deepEqual(row.sources, ['conv', 'relu']);
  assert.equal(row.tensor, plan.tensors[0]);
  assert.equal(row.tensor.offsetBytes, 0n);
  assert.equal(row.tensor.firstStep, undefined, 'missing lifetime must not become step zero');
  assert.equal(row.share, 1);
  assert.deepEqual(summary.timeline.map(i => [i.track, i.start, i.duration]), [['2', 1, 10]]);
});

test('unknown CPU/GPU/RSS observations stay distinct from measured zero and multiple occupied cores', () => {
  const empty = summarizeProfile(trace({events: [event({host: {durationNs: 0n}})], resources: [new pb.ResourceSnapshot({
    cpu: {status: unsupported}, gpu: {status: unsupported}, process: {resident: {status: unsupported}},
  })]}), pb);
  assert.equal(empty.cpuCores, null); assert.equal(empty.cpuLabel, 'Unsupported');
  assert.equal(empty.rss, null); assert.equal(empty.nodes[0].hostMs, 0);
  assert.equal(empty.nodes[0].deviceMs, null);
  assert.equal(empty.nodes[0].share, null, 'zero total has no defined duration share');
  const snapshots = [0n, 100n].map(t => new pb.ResourceSnapshot({observationEndNs: t,
    cpu: {status: available, source: 'process-clock', processTimeNs: t * 2n},
    process: {resident: {status: available, bytes: 0n}}, wasm: {status: available, linearBytes: t},
  }));
  const measured = summarizeProfile(trace({resources: snapshots}), pb);
  assert.equal(measured.cpuCores, 2); assert.equal(measured.rss, 0n);
  assert.equal(measured.sampledWasmPeak, 100n);
  assert.equal(summarizeProfile(trace({resources: snapshots.slice(0, 1)}), pb).cpuLabel, 'Needs comparable samples');
  snapshots[1].cpu.source = 'different-clock';
  assert.equal(summarizeProfile(trace({resources: snapshots}), pb).cpuCores, null);
});

test('profile JSON retains exact 64-bit identities, loss and unknown placement', () => {
  const original = trace({info: new pb.TraceInfo({events: {dropped: 2n}, resourceSnapshots: {dropped: 3n}, plans: {dropped: 4n}}),
    plans: [new pb.ExecutionPlan({planId: 9007199254740993n, tensors: [{name: 'out', logicalBytes: 0n}]})]});
  const summary = summarizeProfile(original, pb);
  assert.deepEqual(summary.loss, {events: 2n, resources: 3n, plans: 4n});
  const json = JSON.parse(JSON.stringify(profileEvidence(original, {label: 'test'}, 0)));
  assert.equal(json.plans[0].planId, '9007199254740993');
  assert.equal(json.plans[0].tensors[0].offsetBytes, undefined);
  assert.equal(json.wallMs, 0);
});

function traceClient() {
  const calls = [], releases = [];
  const ready = () => new pb.TraceInfo({traceId: 5n, report: new pb.OperationReport(), state: pb.TraceState.TRACE_STATE_READY,
    plans: {count: 2n}});
  // Two AIP-158 pages per listing: the engine returns an opaque token for page two.
  const page = (Response, field, request) => {
    calls.push([field, request.pageToken]);
    return new Response({report: new pb.OperationReport(), [field]: [{}], nextPageToken: request.pageToken ? '' : 'next'});
  };
  class Client {
    async startTrace() { return ready(); }
    async stopTrace() { return ready(); }
    async releaseTrace(ref) { releases.push(ref.traceId); }
    async listTraceEvents(request) { return page(pb.ListTraceEventsResponse, 'events', request); }
    async listTraceResourceSnapshots(request) { return page(pb.ListTraceResourceSnapshotsResponse, 'resourceSnapshots', request); }
    async getTracePlan(request) {
      calls.push(['plan', request.planId]);
      return new pb.GetTracePlanResponse({report: new pb.OperationReport(), plan: {planId: request.planId}});
    }
    async exportChromeTrace(request) {
      return new pb.ExportChromeTraceResponse({report: new pb.OperationReport(),
        data: new TextEncoder().encode(request.pageToken ? ']}' : '{"traceEvents":['), nextPageToken: request.pageToken ? '' : 'next'});
    }
  }
  return {api: {pb, VxProfilingServiceClient: Client}, calls, releases};
}

test('trace adapter follows page tokens, fetches plans by ID and releases its collector', async () => {
  const f = traceClient(), capture = await startEngineTrace(f.api, {}, 1n), result = await capture.finish();
  assert.deepEqual([result.events.length, result.resources.length, result.plans.length], [2, 2, 2]);
  assert.deepEqual(f.calls, [['events', ''], ['events', 'next'], ['resourceSnapshots', ''], ['resourceSnapshots', 'next'],
    ['plan', 1n], ['plan', 2n]]);
  assert.deepEqual(JSON.parse(await result.blob.text()), {traceEvents: []});
  await capture.release(); assert.deepEqual(f.releases, [5n]);
});
