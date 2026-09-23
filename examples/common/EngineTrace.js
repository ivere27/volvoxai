/** Example UI adapter over the generated profiling service. */
function checked(value) {
  const report = value.report;
  if (report?.status !== 0) throw new Error(report?.message ?? 'Profiling request failed');
  return value;
}

export async function startEngineTrace(api, host, runtimeId) {
  const { pb, VxProfilingServiceClient } = api;
  const client = new VxProfilingServiceClient(host);
  const started = checked(await client.startTrace(new pb.StartTraceRequest({
    runtimeId, deviceTiming: true, memory: true, detail: pb.TraceDetail.TRACE_DETAIL_NODES, capacityBytes: 32n * 1024n * 1024n,
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
    async finish() {
      try {
        let info = checked(await client.stopTrace(ref));
        while (info.state !== pb.TraceState.TRACE_STATE_READY) {
          info = checked(await client.getTrace(ref));
          if (info.state !== pb.TraceState.TRACE_STATE_READY)
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        const events = [];
        let offset = 0n;
        for (;;) {
          const page = checked(await client.readTrace(new pb.ReadTraceRequest({
            traceId: ref.traceId, offset, limit: 128,
          })));
          events.push(...page.events);
          if (page.eof) break;
          offset = page.nextOffset;
          await new Promise(resolve => setTimeout(resolve, 0));
        }
        const chunks = [];
        offset = 0n;
        for (;;) {
          const chunk = checked(await client.exportChromeTrace(new pb.ExportChromeTraceRequest({
            traceId: ref.traceId, offset, limit: 128,
          })));
          chunks.push(chunk.data);
          if (chunk.eof) break;
          offset = chunk.nextOffset;
          await new Promise(resolve => setTimeout(resolve, 0));
        }
        return { info, events, blob: new Blob(chunks, { type: 'application/json' }) };
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
