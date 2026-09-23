/** Analysis of the engine's recorded observations; all times are displayed in ms. */
const identity = `
  EXTRACT_ARG(arg_set_id, 'args.runtimeId') AS runtime,
  EXTRACT_ARG(arg_set_id, 'args.modelId') AS model,
  EXTRACT_ARG(arg_set_id, 'args.compiledModelId') AS compiled_model,
  EXTRACT_ARG(arg_set_id, 'args.contextId') AS context,
  EXTRACT_ARG(arg_set_id, 'args.backend') AS backend`;
const scope = 'runtime, model, compiled_model, context, backend';
const complete = "COALESCE(EXTRACT_ARG(arg_set_id, 'args.metadataTruncated'), 0) = 0";

// Device instants have dur=0 in Perfetto. The measured interval is in the args.
const nodes = `
SELECT ${identity},
  CASE category WHEN 'host.node' THEN 'Host' ELSE 'GPU' END AS domain,
  EXTRACT_ARG(arg_set_id, 'args.phase') AS phase,
  EXTRACT_ARG(arg_set_id, 'args.scheduleIndex') AS node_index,
  EXTRACT_ARG(arg_set_id, 'args.output') AS output,
  EXTRACT_ARG(arg_set_id, 'args.fused') AS fused,
  name AS operator,
  CASE category WHEN 'host.node' THEN dur
    ELSE CAST(EXTRACT_ARG(arg_set_id, 'args.deviceDurationNs') AS INTEGER)
  END AS elapsed_ns
FROM slice
WHERE (category = 'host.node' OR (category = 'device.interval'
  AND EXTRACT_ARG(arg_set_id, 'args.scheduleIndex') IS NOT NULL))
  AND ${complete}`;

export const operatorQueries = ['Host', 'GPU'].map(domain => ({
  title: `${domain} operators`,
  sql: `WITH nodes AS (${nodes})
SELECT backend, phase, operator, COUNT(*) AS calls,
  ROUND(SUM(elapsed_ns) / 1e6, 3) AS total_ms,
  ROUND(AVG(elapsed_ns) / 1e6, 3) AS avg_ms,
  ROUND(MAX(elapsed_ns) / 1e6, 3) AS max_ms
FROM nodes WHERE domain = '${domain}' AND elapsed_ns >= 0
GROUP BY backend, phase, operator
ORDER BY SUM(elapsed_ns) DESC`,
}));

const slowNodes = {
  title: 'Slow nodes',
  sql: `WITH nodes AS (${nodes}), totals AS (
  SELECT ${scope}, domain, phase, node_index, output, operator, fused,
    COUNT(*) AS calls, SUM(elapsed_ns) AS total_ns,
    AVG(elapsed_ns) AS avg_ns, MAX(elapsed_ns) AS max_ns
  FROM nodes WHERE elapsed_ns >= 0 AND node_index IS NOT NULL
  GROUP BY ${scope}, domain, phase, node_index, output, operator, fused
)
SELECT domain, backend, operator,
  ROUND(total_ns / 1e6, 3) AS total_ms,
  ROUND(100.0 * total_ns / NULLIF(SUM(total_ns) OVER (
    PARTITION BY ${scope}, domain, phase), 0), 2) AS node_time_pct,
  calls, ROUND(avg_ns / 1e6, 3) AS avg_ms, ROUND(max_ns / 1e6, 3) AS max_ms,
  output, node_index, fused, phase, model, compiled_model, context
FROM totals ORDER BY domain, total_ns DESC`,
};

// One row is one recorded engine host call, not an end-to-end user request.
const operations = `
SELECT id, ts, dur, ${identity}, name AS operation,
  EXTRACT_ARG(arg_set_id, 'args.executionId') AS execution
FROM slice WHERE category = 'host.operation' AND dur >= 0 AND ${complete}`;
const runScope = `${scope}, operation`;
const rankedRuns = `WITH operations AS (${operations}), ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY ${runScope} ORDER BY ts, id) AS run,
    ROW_NUMBER() OVER (PARTITION BY ${runScope} ORDER BY dur, id) AS duration_rank,
    COUNT(*) OVER (PARTITION BY ${runScope}) AS samples
  FROM operations
)`;

const runSummary = {
  title: 'Run summary',
  sql: `${rankedRuns}
SELECT operation, backend, COUNT(*) AS samples,
  ROUND(MAX(CASE WHEN run = 1 THEN dur END) / 1e6, 3) AS first_ms,
  ROUND(AVG(CASE WHEN run > 1 THEN dur END) / 1e6, 3) AS later_avg_ms,
  ROUND(AVG(dur) / 1e6, 3) AS avg_ms,
  ROUND(AVG(CASE WHEN duration_rank IN ((samples + 1) / 2, (samples + 2) / 2)
    THEN dur END) / 1e6, 3) AS median_ms,
  ROUND(MAX(CASE WHEN duration_rank = (95 * samples + 99) / 100
    THEN dur END) / 1e6, 3) AS p95_ms,
  ROUND(MAX(dur) / 1e6, 3) AS max_ms, model, compiled_model, context
FROM ranked GROUP BY ${runScope} ORDER BY SUM(dur) DESC`,
};

const runs = {
  title: 'Runs',
  sql: `${rankedRuns}
SELECT operation, backend, run,
  ROUND(ts / 1e6, 3) AS start_ms, ROUND(dur / 1e6, 3) AS host_ms,
  execution, model, compiled_model, context, id AS slice_id
FROM ranked ORDER BY ts, id`,
};

// Display labels for MemorySpace in proto/volvoxai.proto; no runtime SDK import.
function memorySpace(column) {
  return `CASE ${column}
    WHEN 0 THEN 'unspecified' WHEN 1 THEN 'host' WHEN 2 THEN 'JS heap'
    WHEN 3 THEN 'JS external' WHEN 4 THEN 'ArrayBuffer' WHEN 5 THEN 'WASM linear'
    WHEN 6 THEN 'native heap' WHEN 7 THEN 'mapped file' WHEN 8 THEN 'device'
    WHEN 9 THEN 'device local' WHEN 10 THEN 'host-visible device'
    WHEN 11 THEN 'unified' ELSE CAST(${column} AS TEXT) END`;
}

const copiesAndWaits = {
  title: 'Copies and waits',
  sql: `WITH activities AS (
  SELECT EXTRACT_ARG(arg_set_id, 'args.backend') AS backend,
    EXTRACT_ARG(arg_set_id, 'args.deviceId') AS device,
    EXTRACT_ARG(arg_set_id, 'args.queueId') AS queue,
    CASE category WHEN 'host.copy' THEN 'Host copy'
      WHEN 'device.copy' THEN 'GPU copy' WHEN 'host.submit' THEN 'Host submission'
      WHEN 'host.wait' THEN 'Blocking wait' ELSE 'Async completion' END AS activity,
    name, EXTRACT_ARG(arg_set_id, 'args.copySource') AS source,
    EXTRACT_ARG(arg_set_id, 'args.copyDestination') AS destination,
    CAST(EXTRACT_ARG(arg_set_id, 'args.copyBytes') AS INTEGER) AS bytes,
    CASE category WHEN 'device.copy'
      THEN CAST(EXTRACT_ARG(arg_set_id, 'args.deviceDurationNs') AS INTEGER)
      ELSE dur END AS elapsed_ns
  FROM slice WHERE category IN (
    'host.copy', 'device.copy', 'host.submit', 'host.wait', 'host.await')
    AND ${complete}
)
SELECT activity, backend, name,
  COUNT(*) AS calls, SUM(bytes) AS copy_bytes,
  ROUND(SUM(elapsed_ns) / 1e6, 3) AS total_ms,
  ROUND(AVG(elapsed_ns) / 1e6, 3) AS avg_ms,
  ROUND(MAX(elapsed_ns) / 1e6, 3) AS max_ms,
  ${memorySpace('source')} AS source, ${memorySpace('destination')} AS destination,
  device, queue
FROM activities WHERE elapsed_ns >= 0
GROUP BY backend, device, queue, activity, name, source, destination
ORDER BY activity, SUM(elapsed_ns) DESC`,
};

// otherData is not imported by Perfetto. Pass only its small summary as SQL
// values in the startup commands, keeping the original trace bytes unchanged.
function literal(value) {
  if (value == null) return 'NULL';
  if (typeof value === 'boolean') return value ? '1' : '0';
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  if (typeof value === 'string') return `'${value.replaceAll("'", "''").replaceAll('\0', '')}'`;
  return 'NULL';
}

function values(rows, columns) {
  return rows.length ? `VALUES ${rows.map(row => `(${row.map(literal).join(', ')})`).join(',\n')}`
    : `SELECT ${Array(columns).fill('NULL').join(', ')} WHERE 0`;
}

function memoryQuery(metadata) {
  const allocators = Array.isArray(metadata.allocators) ? metadata.allocators : [];
  const rows = allocators.map(m => [m.allocator, m.scope, m.inventory,
    m.observationStartNs, m.existingBytes, m.liveBytes, m.peakBytes,
    m.allocatedBytes, m.freedBytes, m.droppedEvents, m.accountingComplete]);
  return {
    title: 'Memory',
    sql: `WITH summary(allocator, scope, inventory, start_ns, existing, live,
  peak, allocated, freed, dropped, complete) AS (${values(rows, 11)}),
history AS (
  SELECT name AS allocator, ts,
    CAST(EXTRACT_ARG(arg_set_id, 'args.liveBytes') AS INTEGER) AS live
  FROM slice WHERE category = 'memory.allocation' AND ${complete}
)
SELECT allocator,
  ROUND(CAST(peak AS INTEGER) / 1048576.0, 3) AS peak_mib,
  CASE WHEN complete = 1 AND CAST(dropped AS INTEGER) = 0 THEN (
    SELECT ROUND(MIN(ts) / 1e6, 3) FROM history
    WHERE history.allocator = summary.allocator AND history.live = CAST(summary.peak AS INTEGER)
  ) END AS peak_at_ms,
  ROUND(CAST(live AS INTEGER) / 1048576.0, 3) AS live_mib,
  ROUND((CAST(live AS INTEGER) - CAST(existing AS INTEGER)) / 1048576.0, 3) AS growth_mib,
  ROUND(CAST(existing AS INTEGER) / 1048576.0, 3) AS existing_mib,
  ROUND(CAST(allocated AS INTEGER) / 1048576.0, 3) AS allocated_mib,
  ROUND(CAST(freed AS INTEGER) / 1048576.0, 3) AS freed_mib,
  CASE complete WHEN 1 THEN 'complete accounting' WHEN 0 THEN 'tracked subset'
    ELSE 'unknown accounting' END AS accounting,
  CAST(dropped AS INTEGER) AS dropped_events, scope, inventory,
  ROUND(CAST(start_ns AS INTEGER) / 1e6, 3) AS observed_from_ms
FROM summary ORDER BY CAST(peak AS INTEGER) DESC`,
  };
}

function captureQuery(metadata) {
  const flag = value => value === true ? 'enabled' : value === false ? 'disabled' : 'unknown';
  const rows = [
    ['Capture', 'Format', metadata.format ?? 'unknown'],
    ['Capture', 'Detail', metadata.detail ?? 'unknown'],
    ['Capture', 'Dropped events', metadata.droppedEvents ?? 'unknown'],
    ['Capture', 'GPU timing', flag(metadata.deviceTiming)],
    ['Capture', 'Memory', flag(metadata.memory)],
    ['Reading the tables', 'Time range', 'Entire capture; timeline selection does not filter these tables'],
    ['Reading the tables', 'Host and GPU', 'Separate elapsed observations; do not add them or interpret as utilization'],
    ['Reading the tables', 'Runs', 'Recorded host calls, not end-to-end requests; first means first observed call'],
    ['Reading the tables', 'p95', 'Nearest-rank percentile; check the sample count'],
    ['Reading the tables', 'Node time %', 'Share within the same model, compilation, context, backend, phase and timing domain'],
    ['Reading the tables', 'Async completion', 'May overlap other waits; does not measure CPU busy or blocking time'],
    ['Reading the tables', 'Memory', 'Requested allocator capacity; partial inventory, not physical RAM/VRAM or tensor payload'],
    ['Reading the tables', 'Empty results', 'No usable observations; not a zero cost measurement'],
  ];
  for (const d of Array.isArray(metadata.devices) ? metadata.devices : []) {
    rows.push([d.backend, 'Timestamp support', d.support ?? 'unknown'],
      [d.backend, 'Node intervals', d.nodeIntervals ?? 'unknown'],
      [d.backend, 'Copy intervals', d.copyIntervals ?? 'unknown'],
      [d.backend, 'Failed intervals', d.failedIntervals ?? 'unknown']);
  }
  return {
    title: 'Capture',
    sql: `WITH info(section, item, value) AS (${values(rows, 3)})
SELECT section, item, value FROM info
UNION ALL
SELECT 'Capture', 'Truncated observations', CAST(COUNT(*) AS TEXT)
FROM slice WHERE EXTRACT_ARG(arg_set_id, 'args.metadataTruncated') = 1`,
  };
}

/** Pure query definitions, also used by the saved-file viewer and regression tests. */
export function traceQueries(metadata = {}) {
  return [slowNodes, runSummary, runs, ...operatorQueries, copiesAndWaits,
    memoryQuery(metadata), captureQuery(metadata)];
}
