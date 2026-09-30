/** Example UI adapter over the generated profiling service. */
function checked(value) {
  const report = value.report;
  if (report?.status !== 0) throw new Error(report?.message ?? 'Profiling request failed');
  return value;
}

/** Collects every page of an AIP-158 List* method (page_token/next_page_token). */
export async function listAll(list, request, field) {
  const records = [];
  let pageToken = '';
  do {
    const page = checked(await list({...request, pageToken}));
    records.push(...page[field]);
    pageToken = page.nextPageToken;
    await new Promise(resolve => setTimeout(resolve, 0));
  } while (pageToken);
  return records;
}

export async function startEngineTrace(api, host, runtimeId) {
  const { pb, VxProfilingServiceClient } = api;
  const client = new VxProfilingServiceClient(host);
  const started = checked(await client.startTrace(new pb.StartTraceRequest({
    runtimeId,
    options: new pb.TraceOptions({
      detail: pb.TraceDetail.TRACE_DETAIL_NODES, deviceTiming: true, memory: true,
      utilization: true, executionPlans: true, capacityBytes: 32n * 1024n * 1024n,
    }),
  })));
  const ref = new pb.TraceRef({ traceId: started.traceId });
  let released = false;
  async function release() {
    if (released) return;
    released = true;
    await client.releaseTrace(ref);
  }
  return {
    release,
    /** Records an application range, like torch.profiler.record_function. */
    async annotate(name, startNs, endNs) {
      checked({report: await client.annotateTrace(new pb.AnnotateTraceRequest({
        traceId: ref.traceId, name, startNs, endNs,
      }))});
    },
    async finish() {
      try {
        // StopTrace replies when the trace is READY; no polling is needed.
        const info = checked(await client.stopTrace(ref));
        const traceId = ref.traceId;
        const events = await listAll(r => client.listTraceEvents(new pb.ListTraceEventsRequest(r)),
          {traceId, pageSize: 1024}, 'events');
        const resources = await listAll(r => client.listTraceResourceSnapshots(new pb.ListTraceResourceSnapshotsRequest(r)),
          {traceId, pageSize: 1024}, 'resourceSnapshots');
        const plans = [];
        for (let planId = 1n; planId <= info.plans.count; planId++)
          plans.push(checked(await client.getTracePlan(new pb.GetTracePlanRequest({traceId, planId}))).plan);
        const chunks = [];
        let pageToken = '';
        do {
          const page = checked(await client.exportChromeTrace(new pb.ExportChromeTraceRequest({traceId, pageToken})));
          chunks.push(page.data);
          pageToken = page.nextPageToken;
          await new Promise(resolve => setTimeout(resolve, 0));
        } while (pageToken);
        return { info, events, resources, plans, blob: new Blob(chunks, { type: 'application/json' }) };
      } finally { await release(); }
    },
  };
}

export function downloadEngineTrace(trace, filename) {
  if (!trace) return;
  const url = URL.createObjectURL(trace.blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function nodeMeasurements(trace, contextId) {
  const records = trace?.events.filter(event => event.host !== undefined && event.node !== undefined && event.program === undefined && !event.metadataTruncated &&
    (contextId === undefined || event.lineage?.contextId === contextId)) ?? [];
  const totals = new Map();
  for (const event of records) {
    const key = event.node.outputName;
    totals.set(key, (totals.get(key) ?? 0) + Number(event.host.durationNs) / 1e6);
  }
  return totals;
}

export const profileMs = (value, digits = 2) => value === null ? '—' : value.toFixed(digits);

/**
 * Replays a READY trace into the page's Performance timeline (User Timing), so
 * Chrome DevTools shows engine calls and nodes on a custom track beside the
 * page's own work. The WASM host samples the same clock as performance.now(),
 * so TraceInfo.captureOriginNs places host spans exactly. This runs after
 * collection and adds nothing to the measured work. Device intervals have no
 * exact host position and are left to the Perfetto export.
 */
export function measureEngineTrace(trace, {performance = globalThis.performance, track = 'VolvoxAI'} = {}) {
  const origin = Number(trace.info.captureOriginNs) / 1e6;
  let count = 0;
  for (const event of trace.events) {
    if (!event.host || event.metadataTruncated) continue;
    const start = origin + Number(event.host.startNs) / 1e6;
    const name = event.node ? `${event.node.scheduleIndex} ${event.name}` : event.name;
    performance.measure(name, {start, end: start + Number(event.host.durationNs) / 1e6,
      detail: {devtools: {dataType: 'track-entry', track, trackGroup: 'Engine',
        tooltipText: event.node?.outputName || event.name,
        properties: [['backend', event.backend], ['phase', String(event.phase)]]}}});
    count++;
  }
  return count;
}
