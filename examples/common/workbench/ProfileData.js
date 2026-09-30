/** Display projections of typed trace observations. No inferred hardware utilization. */
const nsToMs = value => Number(value) / 1e6;
const available = (value, pb) => value?.status === pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE;

export function summarizeProfile(trace, pb) {
  const missing = ['events', 'resources', 'plans'].filter(section => !Array.isArray(trace?.[section]));
  if (missing.length) throw new Error(`Incomplete profile capture: missing ${missing.join(', ')}. Reload the page to load the current capture code, then record again.`);
  const plans = new Map(trace.plans.map(plan => [String(plan.planId), plan]));
  const rows = new Map(), timeline = [];
  for (const event of trace.events) {
    if (!event.node || event.metadataTruncated || event.activity !== pb.TraceActivity.TRACE_ACTIVITY_COMPUTE || (!event.host && !event.device)) continue;
    if (event.host && event.program) continue; // Programs are nested inside their host node.
    const key = `${event.planId}:${event.node.scheduleIndex}:${event.backend}`;
    const plan = plans.get(String(event.planId)), step = plan?.steps.find(step => step.scheduleIndex === event.node.scheduleIndex);
    const tensor = plan?.tensors.find(t => t.name === event.node.outputName);
    const row = rows.get(key) ?? {key, step: event.node.scheduleIndex, name: step?.operatorName ?? event.name,
      output: event.node.outputName, backend: event.backend, fused: event.node.fused, sources: step?.sourceNodeIds ?? [],
      tensor, plan, hostMs: null, deviceMs: null, calls: 0, devicePrograms: [], deviceNodes: []};
    if (event.host) {
      row.hostMs = (row.hostMs ?? 0) + nsToMs(event.host.durationNs); row.calls++;
      timeline.push({key, track: String(event.trackId), name: row.name, output: row.output, start: nsToMs(event.host.startNs), duration: nsToMs(event.host.durationNs)});
    } else (event.program ? row.devicePrograms : row.deviceNodes).push(nsToMs(event.device.elapsedNs));
    rows.set(key, row);
  }
  const nodes = [...rows.values()];
  for (const row of nodes) {
    // Prefer program intervals when available, never add enclosing device-node intervals.
    const durations = row.devicePrograms.length ? row.devicePrograms : row.deviceNodes;
    row.deviceMs = durations.length ? durations.reduce((a, b) => a + b, 0) : null;
    row.deviceBasis = row.devicePrograms.length ? 'program intervals' : row.deviceNodes.length ? 'node intervals' : 'not captured';
    row.deviceIntervals = durations.length;
    delete row.devicePrograms; delete row.deviceNodes;
  }
  const hostMs = nodes.reduce((sum, row) => sum + (row.hostMs ?? 0), 0);
  for (const row of nodes) row.share = hostMs && row.hostMs !== null ? row.hostMs / hostMs : null;
  const samples = trace.resources, latest = samples.at(-1);
  const heaps = samples.filter(s => available(s.wasm, pb));
  const cpu = samples.filter(s => available(s.cpu, pb));
  let cpuCores = null;
  if (cpu.length >= 2) {
    const first = cpu[0], last = cpu.at(-1), elapsed = last.observationEndNs - first.observationEndNs;
    const consumed = last.cpu.processTimeNs - first.cpu.processTimeNs;
    if (first.cpu.source === last.cpu.source && elapsed > 0n && consumed >= 0n) cpuCores = Number(consumed) / Number(elapsed);
  }
  const gpu = available(latest?.gpu, pb) ? latest.gpu.devices : [];
  const observed = value => available(value, pb) && value.bytes !== undefined ? value.bytes : null;
  return {nodes, timeline, hostMs, cpuCores, gpu, latest, heaps,
    cpuLabel: cpuCores === null ? available(latest?.cpu, pb) ? 'Needs comparable samples' : observationLabel(latest?.cpu, pb) : `${cpuCores.toFixed(2)} cores`,
    wasm: available(latest?.wasm, pb) ? latest.wasm : null,
    sampledWasmPeak: heaps.length ? heaps.reduce((peak, s) => s.wasm.linearBytes > peak ? s.wasm.linearBytes : peak, 0n) : null,
    rss: observed(latest?.process?.resident), allocators: trace.info.allocators ?? [],
    loss: {events: trace.info.events?.dropped ?? 0n, resources: trace.info.resourceSnapshots?.dropped ?? 0n, plans: trace.info.plans?.dropped ?? 0n}};
}

export function observationLabel(observation, pb) {
  if (!observation) return 'Not captured';
  const s = pb.ObservationStatus;
  return observation.status === s.OBSERVATION_STATUS_UNSUPPORTED ? 'Unsupported' :
    observation.status === s.OBSERVATION_STATUS_FAILED ? 'Failed' :
    observation.status === s.OBSERVATION_STATUS_AVAILABLE ? 'Available' : 'Not collected';
}

export function profileEvidence(trace, source, wallMs) {
  return {format: 'volvoxai-workbench-profile/v1', source, wallMs,
    info: trace.info.toJson(), events: trace.events.map(e => e.toJson()),
    resources: (trace.resources ?? []).map(s => s.toJson()), plans: (trace.plans ?? []).map(p => p.toJson())};
}
