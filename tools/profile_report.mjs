#!/usr/bin/env node
/** Offline analysis of canonical public profiling messages. No engine hooks. */
import {readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';

const available = value => value === 1 || value === 'OBSERVATION_STATUS_AVAILABLE';
const bytes = value => {
  if (value === undefined) return 0n;
  if (typeof value === 'number' && !Number.isSafeInteger(value)) throw new Error('unsafe integer in profile');
  const n = BigInt(value);
  if (n < 0n || n > 0xffffffffffffffffn) throw new Error('invalid uint64 in profile');
  return n;
};
const json = value => JSON.stringify(value, (_, item) => typeof item === 'bigint' ? String(item) : item, 2) + '\n';
const roleIs = (value, name, number) => value === number || value === `MEMORY_RESOURCE_ROLE_${name}`;
const ordered = values => [...values].sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
const quantile = (values, ratio) => values.length ? ordered(values)[Math.ceil(values.length * ratio) - 1] : undefined;

/** Read an immutable capture through the generated service, preserving all sections. */
export async function collectProfile(profiling, p, traceId) {
  const check = value => {
    if (value.report?.status) throw new Error(value.report.message || `profiling status ${value.report.status}`);
    return value;
  };
  const info = check(await profiling.getTrace(new p.TraceRef({traceId})));
  if (info.state !== p.TraceState.TRACE_STATE_READY) throw new Error('stop and drain the trace before collecting');
  const artifact = {format: 'volvoxai-profile/v2', info: info.toJson(), events: [], resources: [], plans: []};
  // AIP-158 listing: request pages until next_page_token is empty.
  for (const [name, list, Request, field] of [
    ['events', r => profiling.listTraceEvents(r), p.ListTraceEventsRequest, 'events'],
    ['resources', r => profiling.listTraceResourceSnapshots(r), p.ListTraceResourceSnapshotsRequest, 'resourceSnapshots']]) {
    let pageToken = '';
    do {
      const page = check(await list(new Request({traceId, pageToken, pageSize: 1024})));
      artifact[name].push(...page[field].map(item => item.toJson()));
      pageToken = page.nextPageToken;
    } while (pageToken);
  }
  for (let planId = 1n; planId <= info.plans.count; planId++)
    artifact.plans.push(check(await profiling.getTracePlan(new p.GetTracePlanRequest({traceId, planId}))).plan.toJson());
  return artifact;
}

function unionBytes(ranges) {
  ranges.sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
  let total = 0n, end = 0n;
  for (const [start, next] of ranges) {
    if (next > end) { total += next - (start > end ? start : end); end = next; }
  }
  return total;
}
function analyzePlan(plan, index) {
  const allocations = new Map((plan.allocations || []).map(a => [String(a.allocationId), a]));
  const arenas = new Map([...allocations].filter(([, a]) => roleIs(a.role, 'ARENA', 8)));
  const tensors = plan.tensors || [], steps = plan.steps || [];
  let capacity = 0n, peak = 0n, peakStep = null;
  for (const a of arenas.values()) capacity += bytes(a.capacityBytes);
  const missing = tensors.filter(t => bytes(t.logicalBytes) && (t.allocationId === undefined || t.firstStep === undefined || t.lastStep === undefined));
  // Union byte ranges per allocation: aliases never add a second buffer.
  for (let step = 0; step < steps.length; step++) {
    const active = new Map();
    for (const tensor of tensors) {
      const id = String(tensor.allocationId);
      if (!arenas.has(id) || tensor.firstStep === undefined || tensor.lastStep === undefined ||
          step < tensor.firstStep || step > tensor.lastStep) continue;
      const start = bytes(tensor.offsetBytes), end = start + bytes(tensor.logicalBytes);
      if (end > bytes(arenas.get(id).capacityBytes)) throw new Error(`plan ${plan.planId}: tensor exceeds allocation`);
      if (!active.has(id)) active.set(id, []);
      active.get(id).push([start, end]);
    }
    let live = 0n;
    for (const ranges of active.values()) live += unionBytes(ranges);
    if (live > peak) { peak = live; peakStep = step; }
  }
  const byRole = {};
  for (const allocation of allocations.values()) {
    const role = String(allocation.role ?? 'UNSPECIFIED');
    byRole[role] = (byRole[role] || 0n) + bytes(allocation.capacityBytes);
  }
  return {
    planId: plan.planId, contextId: plan.lineage?.contextId, backend: plan.backend,
    graphFingerprint: plan.graphFingerprint, shapeSignature: plan.shapeSignature,
    evidence: `/plans/${index}`, placementComplete: !!plan.placementComplete,
    sourceMappingComplete: !!plan.sourceMappingComplete, metadataTruncated: !!plan.metadataTruncated,
    allocationCapacityByRole: byRole,
    arena: {capacityBytes: capacity, mappedTensorPeakBytes: peak, peakStep,
      capacityOutsideMappedPeakBytes: capacity - peak,
      interpretation: 'Schedule-derived union of known tensor ranges. This is not measured resident memory or a promise that the remainder can be removed.'},
    tensorsWithoutPlacementOrLifetime: missing.length,
    largestTensors: [...tensors].sort((a, b) => bytes(a.logicalBytes) > bytes(b.logicalBytes) ? -1 : bytes(a.logicalBytes) < bytes(b.logicalBytes) ? 1 : 0)
      .slice(0, 10).map(t => ({tensorId: t.tensorId ?? 0, name: t.name, dtype: t.dtype, shape: t.shape,
        logicalBytes: bytes(t.logicalBytes), allocationId: t.allocationId, firstStep: t.firstStep, lastStep: t.lastStep})),
  };
}
export function analyzeProfile(profile) {
  if (profile.format !== 'volvoxai-profile/v2') throw new Error('expected volvoxai-profile/v2');
  const info = profile.info || {}, resources = profile.resources || [], events = profile.events || [];
  const wasm = resources.flatMap((r, i) => available(r.wasm?.status) ? [{...r.wasm, evidence: `/resources/${i}/wasm`}] : []);
  for (const sample of wasm) {
    const sum = ['allocatedBytes', 'freeBytes', 'allocatorMetadataBytes', 'modulePrefixBytes', 'pageSlackBytes', 'untrackedBytes']
      .reduce((n, field) => n + bytes(sample[field]), 0n);
    if (sum !== bytes(sample.linearBytes)) throw new Error('invalid WASM memory partition');
  }
  const cpu = resources.flatMap((r, i) => available(r.cpu?.status) ? [{time: (bytes(r.observationStartNs) + bytes(r.observationEndNs)) / 2n,
    cpu: bytes(r.cpu.processTimeNs), source: r.cpu.source, evidence: `/resources/${i}/cpu`}] : []);
  const first = cpu[0], last = cpu.at(-1);
  const elapsed = first && last ? last.time - first.time : 0n;
  const cpuTime = first && last ? last.cpu - first.cpu : 0n;
  const cpuUsage = elapsed > 0n && cpuTime >= 0n && first.source === last.source ? {
    averageOccupiedCores: Number(cpuTime) / Number(elapsed), processCpuTimeNs: cpuTime, elapsedNs: elapsed,
    scope: 'process, all threads', evidence: [first.evidence, last.evidence],
  } : {averageOccupiedCores: null, reason: 'two usable process CPU observations are required'};
  const groups = new Map(), executions = [];
  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    if (event.host && event.lineage?.executionId && !event.node && !event.program &&
        ['Execute', 'ExecuteTensors', 'DecodeStep', 'DecodePrefill', 'ExecutePrefix'].includes(event.name))
      executions.push({executionId: event.lineage.executionId, planId: event.planId,
        backend: event.backend, durationNs: bytes(event.host.durationNs), evidence: `/events/${i}`});
    if (!event.host && !event.device) continue;
    // Separate host/device and node/program strata: nested intervals overlap.
    const layer = event.program ? 'program' : event.node ? 'node' : 'operation';
    const domain = event.device ? 'device elapsed' : 'host wall';
    const lineage = Object.fromEntries(['runtimeId', 'modelId', 'compiledModelId', 'contextId', 'graphRevision']
      .map(field => [field, event.lineage?.[field] ?? '0']));
    const identity = {lineage, planId: event.planId ?? '0', backend: event.backend, domain, layer,
      phase: event.phase ?? 0, activity: event.activity ?? 0, name: event.name,
      scheduleIndex: event.node?.scheduleIndex ?? (event.node ? 0 : undefined),
      program: event.program};
    const key = JSON.stringify(identity);
    if (!groups.has(key)) groups.set(key, {...identity, samples: [], evidence: `/events/${i}`});
    groups.get(key).samples.push(bytes(event.device?.elapsedNs ?? event.host?.durationNs));
  }
  const timing = [...groups.values()].map(({samples, ...group}) => ({...group, count: samples.length,
    totalNs: samples.reduce((a, b) => a + b, 0n), medianNs: quantile(samples, .5), p95Ns: quantile(samples, .95)}))
    .sort((a, b) => a.totalNs > b.totalNs ? -1 : a.totalNs < b.totalNs ? 1 : 0);
  const observedPeak = field => wasm.reduce((peak, sample) => bytes(sample[field]) > peak ? bytes(sample[field]) : peak, 0n);
  const gpu = resources.flatMap((r, i) => r.gpu ? [{...r.gpu, evidence: `/resources/${i}/gpu`}] : []);
  return {
    format: 'volvoxai-profile-analysis/v1',
    coverage: {droppedEvents: bytes(info.events?.dropped), droppedResourceSnapshots: bytes(info.resourceSnapshots?.dropped),
      droppedPlans: bytes(info.plans?.dropped), collectorBytes: bytes(info.collectorBytes),
      resourceSnapshotCount: resources.length, executionPlanCount: (profile.plans || []).length,
      cpuStatuses: [...new Set(resources.map(r => r.cpu?.status ?? 'NOT_COLLECTED'))],
      gpuStatuses: [...new Set(resources.map(r => r.gpu?.status ?? 'NOT_COLLECTED'))]},
    memory: {wasm: wasm.length ? {first: wasm[0], last: wasm.at(-1),
      observedPeakLinearBytes: observedPeak('linearBytes'), observedPeakAllocatedBytes: observedPeak('allocatedBytes'),
      interpretation: 'Sampled heap observations. Peaks between samples may be missed; linear memory does not shrink.'} : null,
      process: resources.map((r, i) => ({...r.process, evidence: `/resources/${i}/process`})),
      plans: (profile.plans || []).map(analyzePlan)},
    cpu: cpuUsage,
    gpu: {latest: gpu.at(-1) || null, scope: 'hardware-wide driver window; includes other applications',
      interpretation: 'Compute and memory activity are driver utilization, not kernel duration, SM occupancy or VRAM allocation percentage.'},
    executions, timing,
    timingInterpretation: 'Totals are within each row. Host/device, operation/node/program and concurrent intervals overlap; do not sum rows into utilization.',
  };
}
export function compareProfiles(before, after) {
  const a = analyzeProfile(before), b = analyzeProfile(after);
  const median = analysis => quantile(analysis.executions.map(e => e.durationNs), .5);
  const beforeMedian = median(a), afterMedian = median(b);
  const signatures = analysis => analysis.memory.plans.map(p => `${p.graphFingerprint}/${p.shapeSignature}/${p.backend}`).sort();
  const comparablePlans = JSON.stringify(signatures(a)) === JSON.stringify(signatures(b)) && a.memory.plans.length > 0;
  const delta = (x, y) => x === undefined || y === undefined ? null : bytes(y) - bytes(x);
  return {format: 'volvoxai-profile-comparison/v1', comparablePlans,
    interpretation: 'Confirm equal inputs, outputs and environment separately. Matching graph/shape/backend metadata alone does not prove numerical equivalence.',
    executionMedianNs: {before: beforeMedian, after: afterMedian, delta: delta(beforeMedian, afterMedian)},
    observedPeakLinearBytes: {before: a.memory.wasm?.observedPeakLinearBytes, after: b.memory.wasm?.observedPeakLinearBytes,
      delta: delta(a.memory.wasm?.observedPeakLinearBytes, b.memory.wasm?.observedPeakLinearBytes)},
    coverage: {before: a.coverage, after: b.coverage}};
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [command, file, second, output] = process.argv.slice(2);
  if (!['analyze', 'compare'].includes(command) || !file || (command === 'compare' && !second))
    throw new Error('usage: node tools/profile_report.mjs analyze capture.json [report.json] | compare before.json after.json [report.json]');
  const profile = JSON.parse(await readFile(file, 'utf8'));
  const report = command === 'compare' ? compareProfiles(profile, JSON.parse(await readFile(second, 'utf8'))) : analyzeProfile(profile);
  const destination = command === 'compare' ? output : second;
  if (destination) await writeFile(destination, json(report));
  else process.stdout.write(json(report));
}
