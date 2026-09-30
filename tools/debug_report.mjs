#!/usr/bin/env node
/**
 * Offline comparison of two debug sessions, like TFLite QuantizationDebugger,
 * onnxruntime.quantization.qdq_loss_debug or Polygraphy's layer-wise compare.
 * Observations are aligned by tensor, point (BEFORE/AFTER) and source node ID,
 * never by capture-local IDs. A PTQ activation renamed by authoring is matched
 * to the float tensor it represents. No engine hooks.
 *
 *   node tools/debug_report.mjs compare reference.json candidate.json [report.json]
 *        [--mode auto|exact|quantization] [--atol N] [--rtol N] [--min-sqnr DB]
 *
 * `exact` compares same-precision runs (backends, fused vs preserved schedules)
 * with atol/rtol. `quantization` ranks layers by SQNR; every layer differs, so
 * a first difference alone carries no information. The command exits with
 * status 2 when no observation could be compared.
 *
 * Collect artifacts with collectDebugSession() from a PAUSED or finished session.
 */
import {createHash} from 'node:crypto';
import {readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';

const json = value => JSON.stringify(value, (_, item) => typeof item === 'bigint' ? String(item) : item, 2) + '\n';

/** Reads the plan, every event and (optionally) raw bytes into a JSON artifact. */
export async function collectDebugSession(debug, p, debugSessionId, {values = true} = {}) {
  const check = value => {
    if (value.report?.status) throw new Error(value.report.message || `debug status ${value.report.status}`);
    return value;
  };
  const ref = new p.DebugSessionRef({debugSessionId});
  const info = check(await debug.getDebugSession(ref));
  const plan = check(await debug.getDebugPlan(ref)).plan;
  const events = [];
  let pageToken = '';
  do {
    const page = check(await debug.listDebugEvents(new p.ListDebugEventsRequest({debugSessionId, pageToken, pageSize: 4096})));
    events.push(...page.events);
    pageToken = page.nextPageToken;
  } while (pageToken);
  const artifact = {format: 'volvoxai-debug/v1', info: info.toJson(), plan: plan.toJson(), events: []};
  for (const event of events) {
    const record = {...event.toJson(), snapshots: []};
    for (const snapshot of event.snapshots) {
      const entry = snapshot.toJson();
      if (values && snapshot.status === p.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE) {
        const chunks = [];
        let readOffset = 0n, size = 1n;
        while (readOffset < size) {
          const chunk = check(await debug.readDebugTensor(new p.ReadDebugTensorRequest({
            debugSessionId, snapshotId: snapshot.snapshotId, readOffset, readLimit: 1 << 20})));
          chunks.push(Buffer.from(chunk.data));
          readOffset += BigInt(chunk.data.byteLength);
          size = chunk.sizeBytes;
          if (!chunk.data.byteLength && readOffset < size) throw new Error('debug tensor read did not advance');
        }
        entry.valuesBase64 = Buffer.concat(chunks).toString('base64');
      }
      record.snapshots.push(entry);
    }
    artifact.events.push(record);
  }
  return artifact;
}

const decoders = {
  DATA_TYPE_F32: [4, (v, i) => v.getFloat32(i, true)],
  DATA_TYPE_F64: [8, (v, i) => v.getFloat64(i, true)],
  DATA_TYPE_F16: [2, (v, i) => f16(v.getUint16(i, true))],
  DATA_TYPE_BF16: [2, (v, i) => bf16(v.getUint16(i, true))],
  DATA_TYPE_I64: [8, (v, i) => Number(v.getBigInt64(i, true))],
  DATA_TYPE_U64: [8, (v, i) => Number(v.getBigUint64(i, true))],
  DATA_TYPE_I32: [4, (v, i) => v.getInt32(i, true)],
  DATA_TYPE_U32: [4, (v, i) => v.getUint32(i, true)],
  DATA_TYPE_I16: [2, (v, i) => v.getInt16(i, true)],
  DATA_TYPE_U16: [2, (v, i) => v.getUint16(i, true)],
  DATA_TYPE_I8: [1, (v, i) => v.getInt8(i)],
  DATA_TYPE_U8: [1, (v, i) => v.getUint8(i)],
  DATA_TYPE_BOOL: [1, (v, i) => v.getUint8(i) ? 1 : 0],
};
function f16(bits) {
  const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >> 10) & 0x1f, fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
const bf16View = new DataView(new ArrayBuffer(4));
function bf16(bits) {
  bf16View.setUint32(0, bits << 16 >>> 0);
  return bf16View.getFloat32(0);
}
/** Per-element affine scale of a quantized snapshot, or null when unquantized. */
function scaleOf(snapshot) {
  const q = snapshot.quantization, shape = (snapshot.shape ?? []).map(Number);
  if (q?.perTensor) return () => q.perTensor.scale ?? 0;
  if (!q?.perAxis) return null;
  const axis = q.perAxis.axis ?? 0, inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1);
  return i => q.perAxis.scales[Math.floor(i / inner) % shape[axis]];
}
/** Real values: quantized storage is dequantized with its exact affine parameters. */
function realValues(snapshot) {
  if (!snapshot.valuesBase64) return null;
  const decoder = decoders[snapshot.dtype];
  if (!decoder) return null;
  const bytes = Buffer.from(snapshot.valuesBase64, 'base64');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const [width, read] = decoder, count = bytes.byteLength / width;
  const q = snapshot.quantization, shape = (snapshot.shape ?? []).map(Number);
  const axis = q?.perAxis ? q.perAxis.axis ?? 0 : null;
  const inner = axis === null ? 1 : shape.slice(axis + 1).reduce((a, b) => a * b, 1);
  const values = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const x = read(view, i * width);
    if (q?.perTensor) values[i] = (q.perTensor.scale ?? 0) * (x - (q.perTensor.zeroPoint ?? 0));
    else if (q?.perAxis) {
      const c = Math.floor(i / inner) % shape[axis];
      values[i] = q.perAxis.scales[c] * (x - (q.perAxis.zeroPoints?.[c] ?? 0));
    } else values[i] = x;
  }
  return values;
}

// PTQ authoring renames each quantized activation with a stable derivation
// shared by native/src/training/ptq_names.c and tools/exporter/typed_ptq.py:
// __ptq__.<sha256("volvox-typed-ptq/v1\0<source tensor>\0activation")[:20]>.activation
// A collision appends ".1", ".2", ... to that stem.
const PTQ_FORMAT = 'volvox-typed-ptq/v1';
export const ptqActivationName = source =>
  `__ptq__.${createHash('sha256').update(`${PTQ_FORMAT}\0${source}\0activation`).digest('hex').slice(0, 20)}.activation`;
function ptqAliases(...artifacts) {
  const aliases = new Map();
  for (const artifact of artifacts)
    for (const tensor of artifact.plan?.tensors ?? [])
      if (tensor.name && !tensor.name.startsWith('__ptq__.')) aliases.set(ptqActivationName(tensor.name), tensor.name);
  return aliases;
}
const CONSTANT_ROLES = new Set(['MEMORY_RESOURCE_ROLE_WEIGHTS', 'MEMORY_RESOURCE_ROLE_PACKED_WEIGHTS']);
const DEBUG_POINT_AFTER = 'DEBUG_POINT_AFTER';
const isAfter = point => point === DEBUG_POINT_AFTER || point === 2;

/** Every captured snapshot with the source tensor it represents and its kind. */
function index(artifact, aliases) {
  const plan = artifact.plan ?? {}, tensors = plan.tensors ?? [], steps = plan.steps ?? [];
  const constantAllocations = new Set((plan.allocations ?? [])
    .filter(allocation => CONSTANT_ROLES.has(allocation.role)).map(allocation => String(allocation.allocationId)));
  const produced = new Set(steps.flatMap(step => step.outputs ?? []).map(Number));
  let aliased = 0;
  const canonical = tensor => {
    if (!tensor?.name) return '';
    // Packages record provenance; older ones fall back to the name derivation.
    if (tensor.sourceTensorName) { aliased++; return tensor.sourceTensorName; }
    const stem = tensor.name.replace(/(\.activation)\.\d+$/, '$1');
    const source = aliases.get(stem);
    if (source) aliased++;
    return source ?? tensor.name;
  };
  const entries = [];
  for (const event of artifact.events ?? []) {
    const step = steps[event.step ?? 0];
    for (const snapshot of event.snapshots ?? []) {
      const tensorId = snapshot.tensorId ?? 0, tensor = tensors[tensorId];
      const kind = produced.has(tensorId) ? 'activation'
        : constantAllocations.has(String(tensor?.allocationId)) ? 'constant' : 'external';
      entries.push({nodes: step?.sourceNodeIds ?? [], tensor: tensor?.name, source: canonical(tensor),
        point: event.point, step: event.step ?? 0, kind, snapshot});
    }
  }
  const byNode = new Map(), byOutput = new Map();
  for (const entry of entries) {
    for (const node of entry.nodes) {
      const key = `${entry.source}\0${entry.point}\0${node}`;
      if (!byNode.has(key)) byNode.set(key, entry);
    }
    // An output has one producer per graph, but a quantized graph can carry the
    // same source value twice (quantized producer and dequantized boundary).
    if (isAfter(entry.point) && !byOutput.has(entry.source)) byOutput.set(entry.source, entry);
  }
  return {entries, byNode, byOutput, aliased};
}

function counterpart(entry, candidate) {
  for (const node of entry.nodes) {
    const match = candidate.byNode.get(`${entry.source}\0${entry.point}\0${node}`);
    if (match) return {match, node};
  }
  if (isAfter(entry.point)) {
    const match = candidate.byOutput.get(entry.source);
    if (match) return {match, node: entry.nodes[0]};
  }
  return null;
}

const nonfinite = statistics => ['nanCount', 'positiveInfinityCount', 'negativeInfinityCount']
  .reduce((n, field) => n + Number(statistics?.[field] ?? 0), 0);
const comparableStatus = status => status === 'DEBUG_TENSOR_STATUS_AVAILABLE' || status === 1;

/** Error metrics of candidate `y` against reference `x`, over finite pairs. */
function errorMetrics(x, y, scale, {atol, rtol}) {
  let count = 0, maxAbs = 0, maxRel = 0, sumAbs = 0, se = 0, ss = 0, yy = 0, dot = 0, scaled = 0, mismatches = 0;
  for (let i = 0; i < x.length; i++) {
    const diff = Math.abs(x[i] - y[i]);
    if (!Number.isFinite(diff)) continue;
    count++;
    maxAbs = Math.max(maxAbs, diff);
    maxRel = Math.max(maxRel, diff / Math.max(Math.abs(x[i]), Number.EPSILON));
    sumAbs += diff; se += diff * diff; ss += x[i] * x[i]; yy += y[i] * y[i]; dot += x[i] * y[i];
    if (scale) { const s = scale(i); scaled += s > 0 ? (diff / s) ** 2 : 0; }
    if (diff > atol + rtol * Math.abs(x[i])) mismatches++;
  }
  const metrics = {elements: x.length, finitePairs: count, maxAbsError: maxAbs, maxRelError: maxRel,
    meanAbsError: count ? sumAbs / count : null, rmse: count ? Math.sqrt(se / count) : null,
    identical: count > 0 && se === 0, mismatches};
  // Signal-to-quantization-noise ratio; absent for identical values or a zero signal.
  metrics.sqnrDb = se > 0 && ss > 0 ? 10 * Math.log10(ss / se) : null;
  metrics.cosine = ss > 0 && yy > 0 ? dot / Math.sqrt(ss * yy) : null;
  metrics.relativeL2 = ss > 0 ? Math.sqrt(se / ss) : null;
  // TFLite's rmse/scale: about 1/sqrt(12) = 0.289 for uniformly distributed rounding error.
  if (scale && count) metrics.rmseOverScale = Math.sqrt(scaled / count);
  return metrics;
}

/**
 * Compares a candidate session with a reference one. `mode` is 'exact',
 * 'quantization' or 'auto' (quantization when any aligned pair differs in dtype
 * or quantization). `comparable` is false when no values could be compared,
 * so an empty comparison can never read as "no difference".
 */
export function compareDebugSessions(reference, candidate,
  {atol = 1e-5, rtol = 1e-3, minSqnrDb = 20, mode = 'auto'} = {}) {
  for (const artifact of [reference, candidate])
    if (artifact.format !== 'volvoxai-debug/v1') throw new Error('expected volvoxai-debug/v1');
  if (!['auto', 'exact', 'quantization'].includes(mode)) throw new Error(`unknown comparison mode ${mode}`);
  const aliases = ptqAliases(reference, candidate);
  const left = index(reference, aliases), right = index(candidate, aliases);
  const rows = [], unmatchedObservations = [], used = new Set();
  let activations = 0, matchedActivations = 0;
  for (const entry of left.entries) {
    if (entry.kind === 'activation') activations++;
    const found = counterpart(entry, right);
    if (!found) {
      unmatchedObservations.push({nodes: entry.nodes, tensor: entry.tensor, point: entry.point,
        step: entry.step, kind: entry.kind});
      continue;
    }
    const {match, node} = found;
    used.add(match);
    if (entry.kind === 'activation') matchedActivations++;
    const a = entry.snapshot, b = match.snapshot;
    const row = {node, nodes: entry.nodes, tensor: entry.tensor, candidateTensor: match.tensor,
      point: entry.point, kind: entry.kind, referenceStep: entry.step, candidateStep: match.step,
      referenceStatus: a.status, candidateStatus: b.status, referenceDtype: a.dtype, candidateDtype: b.dtype,
      quantized: !!(a.quantization || b.quantization),
      referenceNonfinite: nonfinite(a.realStatistics ?? a.statistics),
      candidateNonfinite: nonfinite(b.realStatistics ?? b.statistics)};
    const x = comparableStatus(a.status) ? realValues(a) : null;
    const y = comparableStatus(b.status) ? realValues(b) : null;
    if (x && y && x.length === y.length)
      Object.assign(row, errorMetrics(x, y, scaleOf(b) ?? scaleOf(a), {atol, rtol}));
    else if (x && y) row.comparison = 'shape differs';
    else if (!comparableStatus(a.status) || !comparableStatus(b.status)) row.comparison = 'snapshot has no values';
    else if (!decoders[a.dtype] || !decoders[b.dtype]) row.comparison = 'unsupported dtype';
    else row.comparison = 'values not captured in both sessions';
    rows.push(row);
  }
  const effective = mode !== 'auto' ? mode
    : rows.some(row => row.quantized || row.referenceDtype !== row.candidateDtype) ? 'quantization' : 'exact';
  for (const row of rows) {
    const differentNonfinite = row.referenceNonfinite !== row.candidateNonfinite;
    row.mismatch = differentNonfinite || (row.elements === undefined ? false
      : effective === 'exact' ? row.mismatches > 0 : row.sqnrDb !== null && row.sqnrDb < minSqnrDb);
  }
  const order = point => point === 'DEBUG_POINT_BEFORE' || point === 1 ? 0 : 1;
  rows.sort((a, b) => a.referenceStep - b.referenceStep || order(a.point) - order(b.point));
  const compared = rows.filter(row => row.elements !== undefined).length;
  const worst = effective === 'quantization'
    ? rows.filter(row => isAfter(row.point) && row.kind === 'activation' && row.sqnrDb !== null && row.sqnrDb !== undefined)
      .sort((a, b) => a.sqnrDb - b.sqnrDb).slice(0, 10)
      .map(({node, tensor, point, sqnrDb, cosine, relativeL2, rmseOverScale}) =>
        ({node, tensor, point, sqnrDb, cosine, relativeL2, rmseOverScale}))
    : [];
  return {format: 'volvoxai-debug-comparison/v2', mode: effective, tolerance: {atol, rtol, minSqnrDb},
    comparable: compared > 0,
    reason: compared > 0 ? undefined : rows.length
      ? 'aligned observations carry no comparable values; capture with values: true'
      : 'no observation aligned by tensor, point and source node',
    matched: rows.length, compared, unmatched: unmatchedObservations.length,
    candidateOnly: right.entries.filter(entry => !used.has(entry)).length,
    activationCoverage: {matched: matchedActivations, total: activations,
      ratio: activations ? matchedActivations / activations : null},
    aliases: {ptqActivations: left.aliased + right.aliased},
    firstMismatch: compared > 0 ? rows.find(row => row.mismatch) ?? null : null,
    worst, rows, unmatchedObservations,
    interpretation: effective === 'quantization'
      ? 'Errors accumulate through the graph: a low-SQNR layer can inherit error from earlier layers. The first mismatch is where a difference became observable, not proof of its cause.'
      : 'The first mismatch is where a difference became observable, not proof of its cause.'};
}

function parseArguments(argv) {
  const positional = [], options = {};
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (!flag.startsWith('--')) { positional.push(flag); continue; }
    const value = argv[++i];
    if (value === undefined) throw new Error(`${flag} needs a value`);
    if (flag === '--mode') options.mode = value;
    else if (flag === '--atol') options.atol = Number(value);
    else if (flag === '--rtol') options.rtol = Number(value);
    else if (flag === '--min-sqnr') options.minSqnrDb = Number(value);
    else throw new Error(`unknown option ${flag}`);
  }
  return {positional, options};
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const {positional: [command, first, second, output], options} = parseArguments(process.argv.slice(2));
  if (command !== 'compare' || !first || !second)
    throw new Error('usage: node tools/debug_report.mjs compare reference.json candidate.json [report.json] ' +
      '[--mode auto|exact|quantization] [--atol N] [--rtol N] [--min-sqnr DB]');
  const report = compareDebugSessions(JSON.parse(await readFile(first, 'utf8')),
    JSON.parse(await readFile(second, 'utf8')), options);
  if (output) await writeFile(output, json(report)); else process.stdout.write(json(report));
  if (!report.comparable) {
    process.stderr.write(`debug_report: not comparable: ${report.reason}\n`);
    process.exitCode = 2;
  }
}
