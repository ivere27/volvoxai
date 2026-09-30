import test from 'node:test';
import assert from 'node:assert/strict';
import {analyzeProfile, compareProfiles} from '../tools/profile_report.mjs';

function capture() {
  return {format: 'volvoxai-profile/v2', info: {collectorBytes: '4096', events: {dropped: '2'}},
    resources: [
      {observationStartNs: '1000000000', observationEndNs: '1000000000', cpu: {status: 1, source: 'os', processTimeNs: '5000000000'}},
      {observationStartNs: '2000000000', observationEndNs: '2000000000', cpu: {status: 1, source: 'os', processTimeNs: '7500000000'}},
    ], events: [
      {name: 'Execute', backend: 'cpu', lineage: {executionId: '99'}, planId: '1', host: {durationNs: '100'}},
      {name: 'Add', backend: 'cpu', lineage: {executionId: '99'}, node: {scheduleIndex: 0}, host: {durationNs: '40'}},
    ], plans: [{planId: '1', graphFingerprint: 'graph', shapeSignature: 'x:4', backend: 'cpu', placementComplete: true,
      allocations: [{allocationId: '1', capacityBytes: '128', role: 'MEMORY_RESOURCE_ROLE_ARENA'}],
      steps: [{scheduleIndex: 0}, {scheduleIndex: 1}], tensors: [
        {tensorId: 0, name: 'a', allocationId: '1', offsetBytes: '0', logicalBytes: '64', firstStep: 0, lastStep: 0},
        {tensorId: 1, name: 'view', allocationId: '1', offsetBytes: '32', logicalBytes: '32', firstStep: 0, lastStep: 0},
        {tensorId: 2, name: 'reuse', allocationId: '1', offsetBytes: '0', logicalBytes: '96', firstStep: 1, lastStep: 1},
      ]}],
  };
}
test('analysis unions aliases and reused ranges while CPU can occupy multiple cores', () => {
  const report = analyzeProfile(capture());
  assert.equal(report.cpu.averageOccupiedCores, 2.5);
  assert.equal(report.memory.plans[0].arena.mappedTensorPeakBytes, 96n);
  assert.equal(report.memory.plans[0].arena.capacityOutsideMappedPeakBytes, 32n);
  assert.equal(report.memory.plans[0].arena.peakStep, 1);
  assert.equal(report.coverage.droppedEvents, 2n);
  assert.equal(report.executions.length, 1);
  assert.equal(report.executions[0].durationNs, 100n);
  assert.equal(report.timing.length, 2);
});
test('unknown telemetry stays unknown, and invalid evidence cannot become a convincing report', () => {
  const profile = capture();
  profile.resources = [{cpu: {status: 2}}];
  assert.equal(analyzeProfile(profile).cpu.averageOccupiedCores, null);
  assert.equal(analyzeProfile(profile).memory.wasm, null);
  profile.plans[0].tensors[0].logicalBytes = '129';
  assert.throws(() => analyzeProfile(profile), /exceeds allocation/);
  profile.plans = [];
  profile.resources = [{wasm: {status: 1, linearBytes: '10', allocatedBytes: '11'}}];
  assert.throws(() => analyzeProfile(profile), /partition/);
});
test('comparison requires graph, shape and backend agreement and preserves exact integer deltas', () => {
  const before = capture(), after = capture();
  after.events[0].host.durationNs = '80';
  let report = compareProfiles(before, after);
  assert.equal(report.comparablePlans, true);
  assert.equal(report.executionMedianNs.delta, -20n);
  after.plans[0].shapeSignature = 'x:8';
  report = compareProfiles(before, after);
  assert.equal(report.comparablePlans, false);
  assert.equal(report.observedPeakLinearBytes.delta, null);
});

test('timing groups keep contexts, concrete plans and phases distinct', () => {
  const profile = capture();
  const original = profile.events[1];
  profile.events.push(
    {...original, lineage: {...original.lineage, executionId: '100'}},
    {...original, lineage: {...original.lineage, contextId: '2'}},
    {...original, phase: 'TRACE_PHASE_BACKWARD'},
    {...original, planId: '2'},
  );
  const rows = analyzeProfile(profile).timing.filter(row => row.layer === 'node');
  assert.equal(rows.length, 4);
  assert.deepEqual(rows.map(row => row.count).sort(), [1, 1, 1, 2]);
});
