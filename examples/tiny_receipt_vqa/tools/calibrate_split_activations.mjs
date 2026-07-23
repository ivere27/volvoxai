#!/usr/bin/env node
/**
 * Observe every float32 activation of a split package and emit a PTQ scale map.
 *
 * This is the activation-calibration half of PTQ: the weights quantize data-free,
 * but the activation affines can only come from running the graph on real inputs
 * and watching the ranges. The mechanism is `graph.setOutputs(names)`, which
 * promotes any intermediate tensor to a graph output so its values can be read —
 * the same trick the legacy materialize path uses, generalized here to sweep the
 * whole graph.
 *
 * Tensors are observed in batches because promoting all of them at once makes
 * every intermediate live for the whole execution.
 *
 *   node --import tsx examples/tiny_receipt_vqa/tools/calibrate_split_activations.mjs \
 *     --package build/tiny-receipt-hf-fp32 \
 *     --calibration-data /path/to/disjoint/calibration-records.json \
 *     --samples 16 \
 *     --prefixes-per-record 3 \
 *     --route-sweep balanced-public \
 *     --out build/tiny-receipt-fp32-calibration.json
 *
 * Calibration data is application-owned and explicit; no task or adapter is
 * inferred from filenames or record order:
 *
 *   { "format": "volvox-calibration-records/v1",
 *     "required_families": ["phone", "address"],
 *     "records": [{ "id": "...", "image": "images/....jpg",
 *                   "question": "...", "target": "<field>...</answer>",
 *                   "family": "address" }] }
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { Graph, GraphLoader, VolvoxAI } from '../../../ts/index.js';
import { TinyReceiptByteFallbackBPEVocab } from '../TinyReceiptSplitSession.js';
import { TinyReceiptCharVocab } from '../TinyReceiptW8A8Session.js';

const BATCH = Number(process.env.VOLVOX_CALIBRATION_BATCH ?? 24);
const BACKEND = process.env.VOLVOX_CALIBRATION_BACKEND ?? 'cpu';
export const CALIBRATION_FORMAT = 'volvox-calibration-records/v1';
const HOISTED_DECODER_KEEP = 'v4_keep';
const GRAPH_FORMAT = 'volvox-graph/v1';
const SPLIT_GRAPH_KINDS = Object.freeze(['encoder', 'decoder']);
const CALIBRATOR_SOURCE_PATH = fileURLToPath(import.meta.url);
const CALIBRATOR_LOGICAL_PATH =
  'examples/tiny_receipt_vqa/tools/calibrate_split_activations.mjs';
const FIXTURE_FILES = Object.freeze({
  encoder: Object.freeze({
    image: 'image.f32',
    question_ids: 'question_ids.i32',
    family_ids: 'family_ids.i32',
  }),
  decoder: Object.freeze({
    decoder_input_ids: 'decoder_input_ids.i32',
    memory: 'memory.f32',
    memory_padding_mask: 'memory_padding_mask.i32',
    family_ids: 'family_ids.i32',
  }),
});
const MASK_SHAPE_OPS = new Set([
  'Identity', 'Reshape', 'Expand', 'Squeeze', 'Unsqueeze', 'Transpose', 'Slice',
]);
const PYTHON_WHITESPACE_CODEPOINTS = new Set([
  0x0009, 0x000A, 0x000B, 0x000C, 0x000D,
  0x001C, 0x001D, 0x001E, 0x001F,
  0x0020, 0x0085, 0x00A0, 0x1680,
  ...[...Array(11).keys()].map((offset) => 0x2000 + offset),
  0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
]);

export function normalizeCalibrationRoutingMode(value = 'auto') {
  if (value !== 'auto' && value !== 'explicit') {
    throw new Error("--routing-mode must be 'auto' or 'explicit'");
  }
  return value;
}

export function normalizeCalibrationRouteSweep(value = 'none') {
  if (!['none', 'balanced-public', 'all-public'].includes(value)) {
    throw new Error(
      "--route-sweep must be 'none', 'balanced-public', or 'all-public'",
    );
  }
  return value;
}

/** Resolve the requested policy against the package's immutable graph ABI. */
export function calibrationRoutingModeForGraph(graphRoutingMode, requestedMode = 'auto') {
  const requested = normalizeCalibrationRoutingMode(requestedMode);
  if (graphRoutingMode === 'auto') return requested;
  if (graphRoutingMode === 'specialized') return 'specialized';
  throw new Error(`unsupported calibration graph routing mode '${graphRoutingMode}'`);
}

function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index++) {
    const name = values[index].replace(/^--/, '');
    const value = values[index + 1];
    if (value === undefined || value.startsWith('--')) result[name] = true;
    else {
      result[name] = value;
      index++;
    }
  }
  return result;
}

function packageFetch(graphDocument, weightsBuffer) {
  return async (url) => url === 'graph.json'
    ? { ok: true, json: async () => graphDocument }
    : url === 'model.safetensors'
      ? { ok: true, arrayBuffer: async () => weightsBuffer.slice(0) }
      : { ok: false, statusText: `unexpected source ${url}` };
}

async function loadGraph(packageDir, kind) {
  const graphDocument = JSON.parse(
    await readFile(join(packageDir, kind, 'graph.json'), 'utf8'),
  );
  calibrationGraphAbi(graphDocument, kind);
  const raw = await readFile(join(packageDir, kind, 'model.safetensors'));
  const graph = new Graph();
  await GraphLoader.load(graph, 'model.safetensors', {
    graphUrl: 'graph.json',
    fetch: packageFetch(graphDocument, raw.buffer.slice(
      raw.byteOffset, raw.byteOffset + raw.byteLength,
    )),
  });
  return { graph, document: graphDocument };
}

async function fixtureInputs(fixtures, document, kind) {
  const files = FIXTURE_FILES[kind];
  // Graph inputs are positional names (input0…); map by declared source_name.
  const inputs = {};
  const identities = {};
  const derived = [];
  for (const [name, descriptor] of Object.entries(document.inputs || {})) {
    const source = descriptor.source_name || name;
    const file = files[source];
    if (!file) {
      if (kind === 'decoder' && source === HOISTED_DECODER_KEEP) {
        derived.push(name);
        continue;
      }
      throw new Error(`no fixture for graph input '${name}' (${source})`);
    }
    const Kind = file.endsWith('.i32') ? Int32Array : Float32Array;
    const raw = await readFile(join(fixtures, file));
    inputs[name] = new Kind(
      raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
    );
    identities[file] = contentIdentity(raw, `fixtures/${file}`);
  }
  if (derived.length) {
    const decoderEntry = Object.entries(document.inputs || {}).find(([name, descriptor]) =>
      (descriptor.source_name || name) === 'decoder_input_ids');
    const decoderIds = decoderEntry && inputs[decoderEntry[0]];
    if (!(decoderIds instanceof Int32Array)) {
      throw new Error('hoisted decoder keep mask requires decoder_input_ids');
    }
    for (const name of derived) {
      inputs[name] = Int32Array.from(decoderIds, (value) =>
        value === 0 ? 0 : 1);
    }
  }
  return Object.freeze({
    inputs: Object.freeze(inputs),
    identities: Object.freeze(identities),
  });
}

/** Ranges for the float32 graph inputs, taken from the fixture itself.
 *
 * A graph input is an activation edge like any other: quantized operators that
 * read it need its affine. It is also the one edge no amount of graph execution
 * will reveal, because the sweep observes what nodes *produce*. The values are
 * already in hand, so measure them directly.
 *
 * This is not hypothetical. Once layout normalization removed the identity
 * transpose that used to stand between `memory` and the cross-attention
 * projections, eight `Linear` nodes began reading the graph input directly and
 * silently dropped out of quantization for want of a range.
 */
export function observeFiniteRange(observations, name, values, label = 'tensor') {
  if (!observations || typeof observations !== 'object') {
    throw new Error('finite range observation requires an observations object');
  }
  if (typeof name !== 'string' || name.length === 0
      || values == null || typeof values[Symbol.iterator] !== 'function') {
    throw new Error(`${label} '${name}' has no iterable values to observe`);
  }
  let minimum = Infinity;
  let maximum = -Infinity;
  let count = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error(`${label} '${name}' contains a non-finite value at index ${count}`);
    }
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
    count++;
  }
  if (count === 0) throw new Error(`${label} '${name}' has no values to observe`);

  const previous = observations[name];
  if (previous !== undefined
      && (!Number.isFinite(previous?.min) || !Number.isFinite(previous?.max)
        || previous.min > previous.max)) {
    throw new Error(`${label} '${name}' has an invalid existing calibration range`);
  }
  observations[name] = previous
    ? { min: Math.min(previous.min, minimum), max: Math.max(previous.max, maximum) }
    : { min: minimum, max: maximum };
  return observations[name];
}

export function observeInputs(document, inputs, observations) {
  for (const [name, descriptor] of Object.entries(document.inputs || {})) {
    if ((descriptor?.dtype ?? 'float32') !== 'float32') continue;
    const values = inputs[name];
    // Accumulate across samples like observeBatch does; overwriting would keep
    // only the last sample's range and silently undo the multi-sample sweep.
    observeFiniteRange(observations, name, values, 'graph input');
  }
}

/**
 * An immutable Embedding is a finite lookup table, so its complete output
 * domain is knowable without guessing which IDs representative decode happens
 * to visit. Merge the whole table range into the observed output range. This is
 * essential for autoregressive models: a BOS/PAD seed alone otherwise clips
 * legal token rows before the first decoder block.
 */
export function widenImmutableEmbeddingRanges(document, tensorLookup, observations) {
  if (typeof tensorLookup !== 'function' || observations == null
      || typeof observations !== 'object') {
    throw new Error('embedding range widening requires a tensor lookup and observations');
  }
  let widened = 0;
  for (const node of document.nodes || []) {
    if (node?.opType !== 'Embedding') continue;
    const weightName = node.inputs?.weight;
    const outputName = node.outputs?.out;
    if (typeof weightName !== 'string' || typeof outputName !== 'string') {
      throw new Error('Embedding calibration requires exact weight and out ports');
    }
    const weight = tensorLookup(weightName);
    if (!weight?.isWeight || weight.dtype !== 'float32'
        || !(weight.buffer instanceof Float32Array) || weight.buffer.length === 0) {
      throw new Error(
        `Embedding '${outputName}' requires a finite immutable F32 table '${weightName}'`,
      );
    }
    let minimum = Infinity;
    let maximum = -Infinity;
    for (const value of weight.buffer) {
      if (!Number.isFinite(value)) {
        throw new Error(`Embedding table '${weightName}' contains non-finite values`);
      }
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    const previous = observations[outputName];
    observations[outputName] = previous
      ? { min: Math.min(previous.min, minimum), max: Math.max(previous.max, maximum) }
      : { min: minimum, max: maximum };
    widened++;
  }
  return widened;
}

/** Every float32 tensor a node produces — the PTQ candidates.
 *
 * Graph outputs are included. They need no `setOutputs` promotion — they are
 * already readable — but they still need *observing*, and excluding them from
 * the sweep is not the same thing. Skipping them left the decoder's LM head
 * (`Linear -> logits`, its single largest matmul) with no calibrated range, so
 * `ptq.py` had to leave it in float32 and the byte domain ended one node early.
 */
function candidateTensorNames(document) {
  const names = [...(document.outputs || [])];
  for (const node of document.nodes || []) {
    names.push(...Object.values(node.outputs || {}));
  }
  return [...new Set(names)];
}

function exactImmutableF32Values(tensor, predicate) {
  if (!tensor?.isWeight || tensor.dtype !== 'float32'
      || !(tensor.buffer instanceof Float32Array) || tensor.buffer.length === 0) {
    return false;
  }
  for (const value of tensor.buffer) {
    if (!predicate(value)) return false;
  }
  return true;
}

function soleTensor(mapping, label) {
  if (!mapping || typeof mapping !== 'object' || Array.isArray(mapping)) {
    throw new Error(`${label} must be a tensor map`);
  }
  const values = Object.values(mapping);
  if (values.length !== 1 || typeof values[0] !== 'string') {
    throw new Error(`${label} must name exactly one tensor`);
  }
  return values[0];
}

/**
 * Prove the F32 tensors that are semantic {0,-Inf} attention masks rather than
 * affine-quantizable numbers.
 *
 * The proof begins only at immutable F32 tensors whose complete value set is a
 * subset of {0,-Inf} and includes -Inf. It may pass through a canonical Where
 * with an immutable all-zero alternate, shape-only operators, and addition of
 * two already-proven masks. The only legal exit is mask + ordinary F32 logits,
 * whose result must feed Softmax exclusively. Any graph output or ambiguous
 * fan-out rejects the pattern; names and source-framework labels play no role.
 */
export function proveAdditiveAttentionMaskExclusions(document, tensorLookup) {
  if (!document || !Array.isArray(document.nodes) || typeof tensorLookup !== 'function') {
    throw new Error('attention-mask proof requires graph nodes and a tensor lookup');
  }
  const graphOutputs = new Set(document.outputs || []);
  const consumers = new Map();
  for (const [index, node] of document.nodes.entries()) {
    if (!node?.inputs || typeof node.inputs !== 'object' || Array.isArray(node.inputs)) {
      throw new Error(`attention-mask proof node[${index}] inputs must be a tensor map`);
    }
    for (const tensor of Object.values(node.inputs)) {
      if (typeof tensor !== 'string') {
        throw new Error(`attention-mask proof node[${index}] has a non-string input`);
      }
      const uses = consumers.get(tensor) ?? [];
      uses.push({ index, node });
      consumers.set(tensor, uses);
    }
  }

  const maskDomain = new Set();
  const zeroConstants = new Set();
  for (const name of new Set(document.nodes.flatMap((node) =>
    Object.values(node.inputs || {})))) {
    const tensor = tensorLookup(name);
    if (exactImmutableF32Values(tensor, (value) => value === 0)) {
      zeroConstants.add(name);
    }
    if (exactImmutableF32Values(
      tensor, (value) => value === 0 || value === Number.NEGATIVE_INFINITY,
    ) && [...tensor.buffer].some((value) => value === Number.NEGATIVE_INFINITY)) {
      maskDomain.add(name);
    }
  }

  const reasons = new Map();
  let changed = true;
  while (changed) {
    changed = false;
    for (const [index, node] of document.nodes.entries()) {
      const inputs = Object.values(node.inputs);
      const relevant = node.opType === 'Where' || MASK_SHAPE_OPS.has(node.opType)
        || node.opType === 'Add';
      if (!relevant) continue;
      const output = soleTensor(node.outputs, `attention-mask proof node[${index}] outputs`);
      if (maskDomain.has(output)) continue;
      let reason = null;
      if (node.opType === 'Where' && inputs.length === 3) {
        const condition = node.inputs.condition;
        const x = node.inputs.x;
        const y = node.inputs.y;
        const conditionDtype = tensorLookup(condition)?.dtype;
        if (!['bool', 'int32'].includes(conditionDtype)) continue;
        if ((maskDomain.has(x) && zeroConstants.has(y))
            || (maskDomain.has(y) && zeroConstants.has(x))) {
          reason = 'zero-negative-infinity-where-mask';
        }
      } else if (MASK_SHAPE_OPS.has(node.opType) && inputs.length === 1
          && maskDomain.has(inputs[0])) {
        reason = 'shape-only-attention-mask';
      } else if (node.opType === 'Add' && inputs.length === 2
          && maskDomain.has(inputs[0]) && maskDomain.has(inputs[1])) {
        reason = 'combined-attention-mask';
      }
      if (reason !== null) {
        if (graphOutputs.has(output) || tensorLookup(output)?.dtype !== 'float32') {
          throw new Error(
            `attention-mask proof cannot classify public or non-F32 tensor '${output}'`,
          );
        }
        maskDomain.add(output);
        reasons.set(output, reason);
        changed = true;
      }
    }
  }

  const terminal = new Map();
  for (const [index, node] of document.nodes.entries()) {
    if (node.opType !== 'Add') continue;
    const inputs = Object.values(node.inputs);
    if (inputs.length !== 2) continue;
    const maskInputs = inputs.filter((name) => maskDomain.has(name));
    if (maskInputs.length !== 1) continue;
    const numeric = inputs.find((name) => !maskDomain.has(name));
    const output = soleTensor(node.outputs, `attention-mask proof node[${index}] outputs`);
    const uses = consumers.get(output) ?? [];
    if (tensorLookup(numeric)?.dtype !== 'float32'
        || tensorLookup(output)?.dtype !== 'float32'
        || graphOutputs.has(output)
        || uses.length === 0
        || uses.some(({ node: use }) => use.opType !== 'Softmax'
          || soleTensor(use.inputs, 'Softmax inputs') !== output)) {
      throw new Error(
        `attention mask '${maskInputs[0]}' has an ambiguous numeric Add at '${output}'`,
      );
    }
    terminal.set(output, 'additive-attention-logits-before-softmax');
  }

  // A proven mask may not escape through an operator the proof did not model.
  for (const name of maskDomain) {
    const tensor = tensorLookup(name);
    if (tensor?.isWeight && !(consumers.get(name)?.length)) continue;
    if (graphOutputs.has(name)) {
      throw new Error(`attention mask '${name}' may not be a public graph output`);
    }
    for (const { index, node } of consumers.get(name) ?? []) {
      const output = soleTensor(node.outputs, `attention-mask proof node[${index}] outputs`);
      const modeled = (MASK_SHAPE_OPS.has(node.opType) && maskDomain.has(output))
        || (node.opType === 'Where' && maskDomain.has(output))
        || (node.opType === 'Add'
          && (maskDomain.has(output) || terminal.has(output)));
      if (!modeled) {
        throw new Error(
          `attention mask '${name}' escapes through unsupported ${node.opType} node[${index}]`,
        );
      }
    }
  }

  const excluded = new Map([...reasons, ...terminal]);
  return Object.freeze({
    tensors: Object.freeze([...excluded].map(([name]) => name).sort()),
    reasons: Object.freeze(Object.fromEntries([...excluded].sort(([left], [right]) =>
      left.localeCompare(right)))),
  });
}

/** Resolve candidate dtypes from the loaded runtime graph, not JSON defaults. */
export function runtimeF32CandidateTensors(document, tensorLookup, excluded = []) {
  if (typeof tensorLookup !== 'function') {
    throw new Error('runtime F32 candidate discovery requires a tensor lookup');
  }
  const excludedNames = new Set(excluded);
  return Object.freeze(candidateTensorNames(document).filter((name) =>
    tensorLookup(name)?.dtype === 'float32' && !excludedNames.has(name)));
}

/** A completed map must cover every runnable F32 edge that PTQ may consume. */
export function assertCompleteCalibrationRanges(
  document, runtimeCandidates, observations, label = 'graph',
) {
  const required = new Set(runtimeCandidates || []);
  for (const [name, descriptor] of Object.entries(document.inputs || {})) {
    if ((descriptor?.dtype ?? 'float32') === 'float32') required.add(name);
  }
  const missing = [];
  const invalid = [];
  for (const name of required) {
    const range = observations?.[name];
    if (range === undefined) missing.push(name);
    else if (!Number.isFinite(range?.min) || !Number.isFinite(range?.max)
        || range.min > range.max) invalid.push(name);
  }
  if (missing.length || invalid.length) {
    const details = [];
    if (missing.length) details.push(`missing: ${missing.sort().join(', ')}`);
    if (invalid.length) details.push(`invalid: ${invalid.sort().join(', ')}`);
    throw new Error(`${label}: incomplete F32 calibration ranges (${details.join('; ')})`);
  }
  return required.size;
}

export function sha256Hex(value) {
  return createHash('sha256').update(value).digest('hex');
}

function logicalArtifactPath(value, label = 'artifact path') {
  if (typeof value !== 'string' || value.length === 0 || value.includes('\\')) {
    throw new Error(`${label} must be a non-empty POSIX relative path`);
  }
  const parts = value.split('/');
  if (value.startsWith('/') || parts.some((part) => !part || part === '.' || part === '..')) {
    throw new Error(`${label} must be a normalized logical relative path`);
  }
  return value;
}

function contentIdentity(bytes, logicalPath) {
  const path = logicalArtifactPath(logicalPath);
  return Object.freeze({
    path,
    sha256: sha256Hex(bytes),
    bytes: bytes.byteLength,
  });
}

export async function fileIdentity(physicalPath, logicalPath) {
  const bytes = await readFile(resolve(physicalPath));
  return contentIdentity(bytes, logicalPath);
}

export async function packageIdentity(packageDir, kinds = SPLIT_GRAPH_KINDS) {
  const resolvedPackage = resolve(packageDir);
  const selectedKinds = [...kinds];
  if (selectedKinds.length === 0 || new Set(selectedKinds).size !== selectedKinds.length
      || selectedKinds.some((kind) => !SPLIT_GRAPH_KINDS.includes(kind))) {
    throw new Error('package identity requires unique encoder/decoder graph kinds');
  }
  const entries = await Promise.all(selectedKinds.map(async (kind) => {
    const [graph, weights] = await Promise.all([
      fileIdentity(
        join(resolvedPackage, kind, 'graph.json'), `${kind}/graph.json`,
      ),
      fileIdentity(
        join(resolvedPackage, kind, 'model.safetensors'),
        `${kind}/model.safetensors`,
      ),
    ]);
    return [kind, Object.freeze({ graph, weights })];
  }));
  return Object.freeze({
    graphs: Object.freeze(Object.fromEntries(entries)),
  });
}

function normalizedContentIdentity(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be a content identity`);
  }
  const keys = Object.keys(value).sort();
  if (keys.join(',') !== 'bytes,path,sha256') {
    throw new Error(`${label} must contain exactly path, sha256, and bytes`);
  }
  const path = logicalArtifactPath(value.path, `${label}.path`);
  if (typeof value.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(value.sha256)) {
    throw new Error(`${label}.sha256 must be a lowercase SHA-256 digest`);
  }
  if (!Number.isSafeInteger(value.bytes) || value.bytes < 0) {
    throw new Error(`${label}.bytes must be a non-negative integer`);
  }
  return Object.freeze({ path, sha256: value.sha256, bytes: value.bytes });
}

function normalizedPackageIdentity(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).join(',') !== 'graphs') {
    throw new Error('calibration package must contain exactly graphs');
  }
  const graphs = value.graphs;
  if (!graphs || typeof graphs !== 'object' || Array.isArray(graphs)
      || Object.keys(graphs).length === 0) {
    throw new Error('calibration package graphs must not be empty');
  }
  const normalized = {};
  for (const [kind, entry] of Object.entries(graphs)) {
    if (!SPLIT_GRAPH_KINDS.includes(kind) || !entry || typeof entry !== 'object'
        || Array.isArray(entry) || Object.keys(entry).sort().join(',') !== 'graph,weights') {
      throw new Error('calibration package graph entries must be encoder/decoder graph and weights');
    }
    normalized[kind] = Object.freeze({
      graph: normalizedContentIdentity(entry.graph, `package.${kind}.graph`),
      weights: normalizedContentIdentity(entry.weights, `package.${kind}.weights`),
    });
  }
  return Object.freeze({ graphs: Object.freeze(normalized) });
}

function normalizedFixtureIdentity(value) {
  if (value == null) return null;
  if (typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).join(',') !== 'files'
      || !value.files || typeof value.files !== 'object' || Array.isArray(value.files)
      || Object.keys(value.files).length === 0) {
    throw new Error('calibration fixtures must contain non-empty files');
  }
  const files = {};
  for (const [name, identity] of Object.entries(value.files).sort()) {
    if (!Object.values(FIXTURE_FILES).some((mapping) => Object.values(mapping).includes(name))) {
      throw new Error(`calibration fixtures contain unknown logical file ${name}`);
    }
    const normalized = normalizedContentIdentity(identity, `fixtures.files.${name}`);
    if (normalized.path !== `fixtures/${name}`) {
      throw new Error(`fixtures.files.${name}.path must be fixtures/${name}`);
    }
    files[name] = normalized;
  }
  return Object.freeze({ files: Object.freeze(files) });
}

function familyCounts(contract, familyIds) {
  if (!contract) return null;
  const counts = Object.fromEntries(contract.familyNames.map((name) => [name, 0]));
  for (const familyId of familyIds) {
    if (!Number.isSafeInteger(familyId) || familyId < 0
        || familyId >= contract.familyNames.length) {
      throw new Error(`cannot record provenance for invalid family ID ${familyId}`);
    }
    counts[contract.familyNames[familyId]]++;
  }
  return Object.freeze(counts);
}

export function buildCalibrationProvenance({
  calibrationManifest,
  fixturesArtifact = null,
  samples = [],
  requiredFamilyIds = [],
  routeExecutions = null,
  routeSweep = 'none',
  contract = null,
  prefixesPerRecord,
  backend,
  batch,
  kinds,
  routingMode,
  packageArtifact,
  calibratorSource,
  affineExclusions = {},
}) {
  if (!['auto', 'explicit', 'specialized', 'fixture'].includes(routingMode)) {
    throw new Error(
      'calibration provenance routing mode must be auto, explicit, specialized, or fixture',
    );
  }
  const manifestIdentity = calibrationManifest == null
    ? null
    : normalizedContentIdentity(calibrationManifest, 'calibration_manifest');
  const fixtureIdentity = normalizedFixtureIdentity(fixturesArtifact);
  const sourcePackage = normalizedPackageIdentity(packageArtifact);
  const sourceIdentity = normalizedContentIdentity(
    calibratorSource, 'calibrator_source',
  );
  const sweep = normalizeCalibrationRouteSweep(routeSweep);
  const selected = [...samples];
  const executions = routeExecutions == null ? null : [...routeExecutions];
  if (executions?.length === 0) {
    throw new Error('calibration route executions must not be empty');
  }
  const recordFamilyIds = selected.map((sample) => sample.expectedFamilyId);
  const executionFamilyIds = executions?.map((sample) => sample.familyIds[0]) ?? null;
  const executionCounts = executionFamilyIds == null
    ? null
    : familyCounts(contract, executionFamilyIds);
  if (sweep === 'all-public') {
    if (!contract || selected.length === 0 || executions == null) {
      throw new Error('all-public route provenance requires records and executions');
    }
    const expected = selected.length * contract.familyNames.length;
    if (executions.length !== expected
        || contract.familyNames.some((name) => executionCounts[name] !== selected.length)) {
      throw new Error('all-public route provenance must execute every record on every family');
    }
  } else if (sweep === 'balanced-public') {
    if (!contract || selected.length < contract.familyNames.length || executions == null) {
      throw new Error(
        'balanced-public route provenance requires at least one record per public route',
      );
    }
    const counts = Object.values(executionCounts);
    if (executions.length !== selected.length
        || Math.max(...counts) - Math.min(...counts) > 1
        || counts.some((count) => count === 0)) {
      throw new Error(
        'balanced-public route provenance must distribute records across every family',
      );
    }
  }
  return Object.freeze({
    calibration_manifest: manifestIdentity,
    fixtures: fixtureIdentity,
    selected_records: Object.freeze({
      count: selected.length,
      ids: Object.freeze(selected.map((sample) => sample.id)),
    }),
    family_counts: Object.freeze({
      required: familyCounts(contract, requiredFamilyIds),
      record: familyCounts(contract, recordFamilyIds),
    }),
    route_executions: Object.freeze({
      mode: sweep === 'none' ? 'graph-routing' : sweep,
      count: executions?.length ?? 1,
      family_counts: executionCounts,
    }),
    settings: Object.freeze({
      samples: selected.length || 1,
      prefixes_per_record: prefixesPerRecord,
      backend,
      batch,
      graphs: Object.freeze([...kinds]),
      routing_mode: routingMode,
    }),
    package: sourcePackage,
    calibrator_source: sourceIdentity,
    affine_exclusions: Object.freeze({ ...affineExclusions }),
  });
}

export function buildCalibrationReport(provenance, ranges) {
  if (!provenance || typeof provenance !== 'object'
      || !ranges || typeof ranges !== 'object') {
    throw new Error('calibration report requires provenance and range maps');
  }
  return Object.freeze({ provenance, ...ranges });
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return value;
}

function nonemptyString(value, label) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

/** Match Python clean_text: LF replacement, whitespace collapse/strip, then NFC. */
export function cleanCalibrationText(value) {
  const result = [];
  let pendingSpace = false;
  for (const character of String(value ?? '').replace(/\n/g, ' ')) {
    if (PYTHON_WHITESPACE_CODEPOINTS.has(character.codePointAt(0))) {
      pendingSpace = result.length > 0;
      continue;
    }
    if (pendingSpace) result.push(' ');
    result.push(character);
    pendingSpace = false;
  }
  return result.join('').normalize('NFC');
}

/** Read and validate the family/token ABI from the package. */
export async function calibrationPackageContract(packageManifest, vocabulary, config = {}) {
  const orderedNames = packageManifest?.families?.ordered_names;
  const nameToId = packageManifest?.families?.name_to_id;
  const autoFamilyId = packageManifest?.families?.auto_id;
  if (!Array.isArray(orderedNames) || orderedNames.length === 0
      || new Set(orderedNames).size !== orderedNames.length
      || orderedNames.some((name, index) => typeof name !== 'string'
        || nameToId?.[name] !== index)
      || autoFamilyId !== -1) {
    throw new Error('package manifest has an invalid canonical family ABI');
  }
  const tokenizerManifest = packageManifest?.tokenizer;
  const tokenIds = tokenizerManifest?.token_ids;
  const itos = vocabulary?.itos;
  if (!Array.isArray(itos) || itos.length === 0 || !tokenIds) {
    throw new Error('calibration requires a packaged TinyReceipt tokenizer');
  }
  let tokenizer;
  if (tokenizerManifest?.type === 'char-vocab') {
    if (tokenizerManifest.version !== 1 || tokenizerManifest.itos_key !== 'itos') {
      throw new Error('calibration requires the char-vocab v1 package contract');
    }
    tokenizer = new TinyReceiptCharVocab(itos, tokenIds);
  } else if (tokenizerManifest?.type === 'byte_fallback_bpe') {
    if (tokenizerManifest.version !== 1 || tokenizerManifest.itos_key !== 'itos'
        || tokenizerManifest.merges_key !== 'merges') {
      throw new Error('calibration requires the byte_fallback_bpe v1 package contract');
    }
    tokenizer = await TinyReceiptByteFallbackBPEVocab.fromJSON(
      vocabulary,
      tokenIds,
      {
        vocabSize: tokenizerManifest.vocab_size,
        normalization: tokenizerManifest.normalization,
        tokenizerHash: tokenizerManifest.tokenizer_hash,
      },
    );
  } else {
    throw new Error('calibration requires a supported TinyReceipt package tokenizer');
  }
  for (const [name, literal] of Object.entries({
    pad: '<pad>', bos: '<bos>', eos: '<eos>', unk: '<unk>',
  })) {
    const id = tokenIds[name];
    if (!Number.isSafeInteger(id) || id < 0 || itos[id] !== literal) {
      throw new Error(`package tokenizer ${name} ID does not identify ${literal}`);
    }
  }
  const questionLength = positiveInteger(Number(config.max_q_len), 'config.max_q_len');
  const decoderLength = positiveInteger(
    Number(packageManifest?.generation?.decoder_input_length),
    'generation.decoder_input_length',
  );
  if (decoderLength < 2) {
    throw new Error('generation.decoder_input_length must hold at least BOS and EOS');
  }
  return Object.freeze({
    familyNames: Object.freeze([...orderedNames]),
    nameToFamilyId: Object.freeze({ ...nameToId }),
    autoFamilyId,
    tokenIds: Object.freeze({
      pad: tokenIds.pad, bos: tokenIds.bos, eos: tokenIds.eos, unk: tokenIds.unk,
    }),
    tokenizerKind: tokenizerManifest.type,
    tokenizer,
    itos: tokenizer.itos,
    stoi: tokenizer.stoi,
    questionLength,
    decoderLength,
  });
}

function calibrationFamilyId(value, contract, label) {
  const familyId = typeof value === 'string'
    ? contract.nameToFamilyId[value]
    : value;
  if (!Number.isSafeInteger(familyId) || familyId < 0
      || familyId >= contract.familyNames.length) {
    throw new Error(
      `${label} must be a canonical family name or ID from 0 through `
      + `${contract.familyNames.length - 1}`,
    );
  }
  return familyId;
}

/** Validate the application-owned representative-data contract without guessing. */
export function normalizeCalibrationManifest(document, contract) {
  if (!document || document.format !== CALIBRATION_FORMAT || !Array.isArray(document.records)) {
    throw new Error(`calibration manifest must use format '${CALIBRATION_FORMAT}' and records[]`);
  }
  if (document.records.length === 0) throw new Error('calibration records[] must not be empty');
  const requiredFamilyIds = [];
  if (document.required_families !== undefined) {
    if (!Array.isArray(document.required_families)) {
      throw new Error('calibration required_families must be an array');
    }
    for (const [index, family] of document.required_families.entries()) {
      requiredFamilyIds.push(calibrationFamilyId(
        family, contract, `required_families[${index}]`,
      ));
    }
    if (new Set(requiredFamilyIds).size !== requiredFamilyIds.length) {
      throw new Error('calibration required_families must not contain duplicates');
    }
  }
  const records = document.records.map((record, index) => {
    const label = `records[${index}]`;
    if (!record || typeof record !== 'object' || Array.isArray(record)) {
      throw new Error(`${label} must be an object`);
    }
    const question = cleanCalibrationText(nonemptyString(record.question, `${label}.question`));
    const target = cleanCalibrationText(nonemptyString(record.target, `${label}.target`));
    if (!question || !target) throw new Error(`${label} question and target must contain text`);
    return Object.freeze({
      id: nonemptyString(record.id, `${label}.id`),
      image: nonemptyString(record.image, `${label}.image`),
      question,
      target,
      expectedFamilyId: calibrationFamilyId(record.family, contract, `${label}.family`),
    });
  });
  const available = new Set(records.map((record) => record.expectedFamilyId));
  const missing = requiredFamilyIds.filter((family) => !available.has(family));
  if (missing.length) {
    throw new Error(
      `calibration records do not cover required families: `
      + missing.map((id) => contract.familyNames[id]).join(', '),
    );
  }
  return Object.freeze({
    records: Object.freeze(records),
    requiredFamilyIds: Object.freeze(requiredFamilyIds),
  });
}

/** Deterministically stratify by the records' real family without relabeling them. */
export function selectCalibrationRecords(records, count, familyCount, requiredFamilyIds = []) {
  positiveInteger(count, 'calibration sample count');
  positiveInteger(familyCount, 'calibration family count');
  if (!Array.isArray(records) || records.length < count) {
    throw new Error(`calibration requested ${count} samples but only ${records?.length ?? 0} exist`);
  }
  if (count < requiredFamilyIds.length) {
    throw new Error(
      `calibration needs at least ${requiredFamilyIds.length} samples for required family coverage`,
    );
  }
  const buckets = Array.from({ length: familyCount }, () => []);
  for (const record of records) {
    if (!Number.isSafeInteger(record?.expectedFamilyId)
        || record.expectedFamilyId < 0 || record.expectedFamilyId >= familyCount) {
      throw new Error('calibration record has an invalid normalized family ID');
    }
    buckets[record.expectedFamilyId].push(record);
  }
  const priority = [...requiredFamilyIds];
  for (let family = 0; family < familyCount; family++) {
    if (buckets[family].length && !priority.includes(family)) priority.push(family);
  }
  const selected = [];
  for (let offset = 0; selected.length < count; offset++) {
    let added = false;
    for (const family of priority) {
      if (selected.length >= count) break;
      if (offset < buckets[family].length) {
        selected.push(buckets[family][offset]);
        added = true;
      }
    }
    if (!added) break;
  }
  const covered = new Set(selected.map((record) => record.expectedFamilyId));
  const missing = requiredFamilyIds.filter((family) => !covered.has(family));
  if (selected.length !== count || missing.length) {
    throw new Error('selected calibration samples do not satisfy required family coverage');
  }
  return Object.freeze(selected);
}

/** Expand execution routes without changing any record's semantic family. */
export function calibrationRouteExecutions(samples, contract, routeSweep = 'none') {
  const mode = normalizeCalibrationRouteSweep(routeSweep);
  if (!Array.isArray(samples) || samples.length === 0) {
    throw new Error('calibration route execution expansion requires representative samples');
  }
  if (!contract || !Array.isArray(contract.familyNames)
      || contract.familyNames.length === 0) {
    throw new Error('calibration route execution expansion requires a family contract');
  }
  if (mode === 'none') {
    return Object.freeze(samples.map((sample) => Object.freeze({
      ...sample,
      routeFamilyId: null,
    })));
  }
  if (mode === 'balanced-public') {
    const familyCount = contract.familyNames.length;
    return Object.freeze(samples.map((sample, index) => Object.freeze({
      ...sample,
      routeFamilyId: index % familyCount,
    })));
  }
  const executions = [];
  for (const sample of samples) {
    for (let familyId = 0; familyId < contract.familyNames.length; familyId++) {
      executions.push(Object.freeze({ ...sample, routeFamilyId: familyId }));
    }
  }
  return Object.freeze(executions);
}

export function encodeCalibrationQuestion(text, contract) {
  const ids = new Int32Array(contract.questionLength).fill(contract.tokenIds.pad);
  const cleaned = cleanCalibrationText(text);
  if (contract.tokenizerKind === 'byte_fallback_bpe') {
    const encoded = contract.tokenizer.encode(cleaned, {
      addEos: true,
      maxLength: contract.questionLength,
    });
    ids.set(encoded);
    return ids;
  }
  const characters = [...cleaned];
  const contentLength = Math.min(characters.length, ids.length - 1);
  for (let index = 0; index < contentLength; index++) {
    ids[index] = contract.stoi.get(characters[index]) ?? contract.tokenIds.unk;
  }
  ids[contentLength] = contract.tokenIds.eos;
  return ids;
}

export function encodeCalibrationTarget(text, contract) {
  const cleaned = cleanCalibrationText(text);
  if (contract.tokenizerKind === 'byte_fallback_bpe') {
    return Int32Array.from(contract.tokenizer.encode(cleaned, {
      addBos: true,
      addEos: true,
      maxLength: contract.decoderLength,
    }));
  }
  const ids = [contract.tokenIds.bos];
  for (const character of cleaned) {
    ids.push(contract.stoi.get(character) ?? contract.tokenIds.unk);
  }
  ids.push(contract.tokenIds.eos);
  if (ids.length > contract.decoderLength) {
    ids.length = contract.decoderLength;
    ids[ids.length - 1] = contract.tokenIds.eos;
  }
  return Int32Array.from(ids);
}

/** Match numpy.linspace(..., dtype=int64) from the reference ONNX PTQ. */
export function teacherForcedPrefixLengths(targetIds, count = 3) {
  positiveInteger(count, 'prefixes per calibration record');
  if (!targetIds || !Number.isSafeInteger(targetIds.length) || targetIds.length < 2) {
    throw new Error('teacher-forced target must contain at least BOS and EOS');
  }
  const maximum = Math.max(1, targetIds.length - 1);
  if (count === 1) return Object.freeze([1]);
  const lengths = new Set();
  for (let index = 0; index < count; index++) {
    lengths.add(Math.trunc(1 + ((maximum - 1) * index) / (count - 1)));
  }
  return Object.freeze([...lengths].sort((left, right) => left - right));
}

export function buildTeacherForcedPrefix(targetIds, prefixLength, contract) {
  if (!Number.isSafeInteger(prefixLength) || prefixLength < 1
      || prefixLength >= targetIds.length || prefixLength > contract.decoderLength) {
    throw new Error('teacher-forced prefix length is outside the target sequence');
  }
  const decoderIds = new Int32Array(contract.decoderLength).fill(contract.tokenIds.pad);
  decoderIds.set(targetIds.subarray(0, prefixLength));
  const decoderKeep = new Int32Array(contract.decoderLength);
  decoderKeep.fill(1, 0, prefixLength);
  return Object.freeze({ decoderIds, decoderKeep, prefixLength });
}

/** Representative samples: image + exact question/target/family provenance. */
async function representativeSamples(manifestPath, packageDir, count) {
  const resolvedManifestPath = resolve(manifestPath);
  const [vocabulary, packageManifest, config, calibrationBytes] = await Promise.all([
    readFile(join(packageDir, 'vocab.json'), 'utf8').then(JSON.parse),
    readFile(join(packageDir, 'package_manifest.json'), 'utf8').then(JSON.parse),
    readFile(join(packageDir, 'config.json'), 'utf8').then(JSON.parse),
    readFile(resolvedManifestPath),
  ]);
  const calibrationDocument = JSON.parse(calibrationBytes.toString('utf8'));
  const contract = await calibrationPackageContract(packageManifest, vocabulary, config);
  const normalized = normalizeCalibrationManifest(calibrationDocument, contract);
  const picked = selectCalibrationRecords(
    normalized.records, count, contract.familyNames.length, normalized.requiredFamilyIds,
  );
  const script = [
    'from PIL import Image',
    'import sys',
    "im=Image.open(sys.argv[1]).convert('L').resize((672,320), Image.BILINEAR)",
    'sys.stdout.buffer.write(im.tobytes())',
  ].join('\n');
  const samples = [];
  for (const record of picked) {
    const imagePath = isAbsolute(record.image)
      ? record.image
      : resolve(dirname(resolvedManifestPath), record.image);
    const raw = execFileSync(
      'python3', ['-c', script, imagePath],
      { maxBuffer: 64 * 1024 * 1024 },
    );
    const image = new Float32Array(raw.length);
    for (let index = 0; index < raw.length; index++) {
      image[index] = (raw[index] / 255 - 0.5) / 0.5;
    }
    samples.push(Object.freeze({
      id: record.id,
      image,
      questionIds: encodeCalibrationQuestion(record.question, contract),
      targetIds: encodeCalibrationTarget(record.target, contract),
      expectedFamilyId: record.expectedFamilyId,
    }));
  }
  return Object.freeze({
    samples: Object.freeze(samples),
    contract,
    requiredFamilyIds: normalized.requiredFamilyIds,
    manifest: contentIdentity(calibrationBytes, 'calibration-records.json'),
  });
}

// One runtime for the whole sweep. Creating it per batch — as this did — meant
// re-instantiating the WASM module for every one of the ~17 batches per sample,
// which is what made a multi-sample calibration look infeasible. The *model*
// still has to be recompiled per batch, because promoting tensors to outputs
// changes the topology, but that is far cheaper than a fresh runtime.
let sharedRuntime = null;
async function calibrationRuntime() {
  if (!sharedRuntime) {
    sharedRuntime = await VolvoxAI.createRuntime(BACKEND === 'wasm'
      ? { backends: ['wasm'], wasmUrl: pathToFileURL(resolve('dist/0.3.0/volvoxai.wasm')) }
      : { backends: ['cpu'] });
  }
  return sharedRuntime;
}

function sameNames(actual, expected) {
  return actual.length === expected.length
    && [...actual].sort().every((name, index) => name === [...expected].sort()[index]);
}

/**
 * Classify and validate the two current split calibration ABIs.
 *
 * A specialized package has already frozen `family_ids`. A runtime-selectable
 * package keeps `family_ids` on both graphs. Both decoder ABIs hoist the causal
 * keep mask as the explicit `v4_keep` input. Rejecting mixtures prevents an
 * encoder from selecting one family while the decoder silently executes
 * another specialization.
 */
export function calibrationGraphAbi(document, kind) {
  if (!document || typeof document !== 'object' || Array.isArray(document)
      || document.format !== GRAPH_FORMAT) {
    throw new Error(`calibration ${kind} graph must use format '${GRAPH_FORMAT}'`);
  }
  if (!SPLIT_GRAPH_KINDS.includes(kind)) {
    throw new Error(`calibration graph kind must be ${SPLIT_GRAPH_KINDS.join(' or ')}`);
  }
  if (!document.inputs || typeof document.inputs !== 'object'
      || Array.isArray(document.inputs)) {
    throw new Error(`calibration ${kind} graph inputs must be an object`);
  }
  const inputsBySemantic = {};
  for (const [name, descriptor] of Object.entries(document.inputs)) {
    if (!descriptor || typeof descriptor !== 'object' || Array.isArray(descriptor)) {
      throw new Error(`calibration ${kind} input '${name}' descriptor must be an object`);
    }
    const semantic = descriptor.source_name ?? name;
    if (typeof semantic !== 'string' || semantic.length === 0) {
      throw new Error(`calibration ${kind} input '${name}' has an invalid semantic name`);
    }
    if (Object.hasOwn(inputsBySemantic, semantic)) {
      throw new Error(`calibration ${kind} graph has duplicate '${semantic}' inputs`);
    }
    inputsBySemantic[semantic] = Object.freeze({ name, descriptor });
  }

  const core = kind === 'encoder'
    ? ['image', 'question_ids']
    : ['decoder_input_ids', 'memory', 'memory_padding_mask'];
  const specialized = kind === 'encoder'
    ? core
    : [...core, HOISTED_DECODER_KEEP];
  const automatic = kind === 'encoder'
    ? [...core, 'family_ids']
    : [...core, 'family_ids', HOISTED_DECODER_KEEP];
  const semantics = Object.keys(inputsBySemantic);
  const routingMode = sameNames(semantics, specialized)
    ? 'specialized'
    : sameNames(semantics, automatic) ? 'auto' : null;
  if (routingMode === null) {
    throw new Error(
      `calibration ${kind} graph inputs [${semantics.sort().join(', ')}] do not match `
      + `the current specialized [${specialized.join(', ')}] or AUTO `
      + `[${automatic.join(', ')}] ABI`,
    );
  }

  const expectedDtypes = kind === 'encoder'
    ? { image: 'float32', question_ids: 'int32', family_ids: 'int32' }
    : {
      decoder_input_ids: 'int32', memory: 'float32', memory_padding_mask: 'int32',
      family_ids: 'int32', [HOISTED_DECODER_KEEP]: 'int32',
    };
  for (const semantic of semantics) {
    const actual = inputsBySemantic[semantic].descriptor.dtype;
    if (actual !== expectedDtypes[semantic]) {
      throw new Error(
        `calibration ${kind} input '${semantic}' must be ${expectedDtypes[semantic]}, got ${actual}`,
      );
    }
  }
  return Object.freeze({
    kind,
    routingMode,
    inputs: Object.freeze(inputsBySemantic),
  });
}

export function bindEncoderCalibrationInputs(
  document, sample, autoFamilyId, requestedRoutingMode = 'auto', routeFamilyId = null,
) {
  const abi = calibrationGraphAbi(document, 'encoder');
  const requested = normalizeCalibrationRoutingMode(requestedRoutingMode);
  const inputs = {
    [abi.inputs.image.name]: sample.image,
    [abi.inputs.question_ids.name]: sample.questionIds,
  };
  if (abi.routingMode === 'auto') {
    if (routeFamilyId !== null) {
      if (!Number.isSafeInteger(routeFamilyId) || routeFamilyId < 0) {
        throw new Error('encoder calibration route family ID must be non-negative');
      }
      inputs[abi.inputs.family_ids.name] = Int32Array.of(routeFamilyId);
    } else if (requested === 'auto') {
      if (autoFamilyId !== -1) {
        throw new Error('AUTO encoder calibration requires canonical family ID -1');
      }
      inputs[abi.inputs.family_ids.name] = Int32Array.of(autoFamilyId);
    } else {
      if (!Number.isSafeInteger(sample?.expectedFamilyId) || sample.expectedFamilyId < 0) {
        throw new Error('explicit encoder calibration requires a non-negative record family ID');
      }
      inputs[abi.inputs.family_ids.name] = Int32Array.of(sample.expectedFamilyId);
    }
  } else if (routeFamilyId !== null) {
    throw new Error('public route sweep requires a runtime-selectable family input');
  }
  return inputs;
}

export function bindDecoderCalibrationInputs(document, produced, prefix) {
  const abi = calibrationGraphAbi(document, 'decoder');
  const compatibleRouting = produced.routingMode === undefined
    || produced.routingMode === abi.routingMode
    || (abi.routingMode === 'auto' && produced.routingMode === 'explicit');
  if (!compatibleRouting) {
    throw new Error(
      `split calibration routing mode mismatch: encoder is ${produced.routingMode}, `
      + `decoder is ${abi.routingMode}`,
    );
  }
  const inputs = {};
  for (const [name, descriptor] of Object.entries(document.inputs || {})) {
    const source = descriptor.source_name || name;
    if (source === 'memory') inputs[name] = produced.memory;
    else if (source === 'memory_padding_mask') inputs[name] = produced.memoryMask;
    else if (source === 'decoder_input_ids') {
      if (descriptor.shape && (descriptor.shape.length !== 2 || descriptor.shape[0] !== 1
          || descriptor.shape[1] !== prefix.decoderIds.length)) {
        throw new Error(`decoder input '${name}' does not match the fixed prefix length`);
      }
      inputs[name] = prefix.decoderIds;
    } else if (source === 'family_ids') inputs[name] = produced.familyIds;
    else if (source === HOISTED_DECODER_KEEP) {
      if (descriptor.shape && (descriptor.shape.length !== 2 || descriptor.shape[0] !== 1
          || descriptor.shape[1] !== prefix.decoderKeep.length)) {
        throw new Error(`decoder keep input '${name}' does not match the fixed prefix length`);
      }
      inputs[name] = prefix.decoderKeep;
    } else {
      throw new Error(`unknown decoder calibration input '${name}' (${source})`);
    }
  }
  return inputs;
}

export function validateSelectedCalibrationFamily(
  sample, selectedFamilyIds, contract, routingMode, routeFamilyId = null,
) {
  if (!['auto', 'explicit', 'specialized'].includes(routingMode)) {
    throw new Error(`invalid selected-family calibration routing mode '${routingMode}'`);
  }
  if (!(selectedFamilyIds instanceof Int32Array) || selectedFamilyIds.length !== 1
      || !Number.isSafeInteger(selectedFamilyIds[0]) || selectedFamilyIds[0] < 0
      || selectedFamilyIds[0] >= contract.familyNames.length) {
    throw new Error(
      `encoder selected invalid family ${selectedFamilyIds?.[0]} `
      + `for calibration record ${sample.id}`,
    );
  }
  if (routeFamilyId !== null && (!Number.isSafeInteger(routeFamilyId)
      || routeFamilyId < 0 || routeFamilyId >= contract.familyNames.length)) {
    throw new Error(`route sweep family ID ${routeFamilyId} is outside the public catalog`);
  }
  if (routeFamilyId !== null && selectedFamilyIds[0] !== routeFamilyId) {
    throw new Error(
      `route sweep requested ${contract.familyNames[routeFamilyId]} but encoder selected `
      + `${contract.familyNames[selectedFamilyIds[0]]} for calibration record ${sample.id}`,
    );
  }
  if (routeFamilyId === null
      && routingMode !== 'auto' && selectedFamilyIds[0] !== sample.expectedFamilyId) {
    throw new Error(
      `${routingMode} encoder selects ${contract.familyNames[selectedFamilyIds[0]]} but `
      + `calibration record ${sample.id} is ${contract.familyNames[sample.expectedFamilyId]}`,
    );
  }
  return selectedFamilyIds[0];
}

export function assertCalibrationRoutingRecords(samples, contract, routingMode) {
  if (!['auto', 'explicit', 'specialized'].includes(routingMode)) {
    throw new Error(`invalid calibration routing mode '${routingMode}'`);
  }
  if (routingMode !== 'specialized') return null;
  const families = new Set(samples.map((sample) => sample.expectedFamilyId));
  if (families.size !== 1) {
    throw new Error(
      'specialized calibration records must all target the one frozen package family',
    );
  }
  const [familyId] = families;
  if (!Number.isSafeInteger(familyId) || familyId < 0
      || familyId >= contract.familyNames.length) {
    throw new Error('specialized calibration records contain an invalid family ID');
  }
  return familyId;
}

/** Run the encoder on each sample to obtain the decoder's real `memory` input. */
async function encodeSamples(
  packageDir, samples, contract, requestedRoutingMode, routeSweep,
) {
  const { graph, document } = await loadGraph(packageDir, 'encoder');
  const abi = calibrationGraphAbi(document, 'encoder');
  const sweep = normalizeCalibrationRouteSweep(routeSweep);
  if (sweep !== 'none' && abi.routingMode !== 'auto') {
    throw new Error('public route sweep requires a runtime-selectable family input');
  }
  const routingMode = sweep !== 'none'
    ? 'explicit'
    : calibrationRoutingModeForGraph(abi.routingMode, requestedRoutingMode);
  assertCalibrationRoutingRecords(samples, contract, routingMode);
  const runtime = await calibrationRuntime();
  const model = runtime.createModel(graph);
  const compiled = await model.compile({
    backend: { mode: 'require', backend: BACKEND, operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const produced = [];
  for (const sample of samples) {
    const inputs = bindEncoderCalibrationInputs(
      document, sample, contract.autoFamilyId, requestedRoutingMode,
      sample.routeFamilyId,
    );
    const result = await context.execute(inputs);
    const selectedFamilyIds = Int32Array.from(
      await result.output('selected_family_ids').read(),
    );
    validateSelectedCalibrationFamily(
      sample, selectedFamilyIds, contract, routingMode, sample.routeFamilyId,
    );
    produced.push({
      id: sample.id,
      memory: Float32Array.from(await result.output('memory').read()),
      memoryMask: Int32Array.from(await result.output('memory_padding_mask').read()),
      familyIds: selectedFamilyIds,
      expectedFamilyId: sample.expectedFamilyId,
      routeFamilyId: sample.routeFamilyId,
      targetIds: sample.targetIds,
      routingMode,
    });
    await result.close?.();
  }
  await context.close?.();
  await compiled.close?.();
  await model.close?.();
  return produced;
}

async function observeBatch(
  packageDir, kind, inputSets, names, observations, cached, onSample = null,
) {
  const { graph } = cached ?? await loadGraph(packageDir, kind);
  const original = [...graph.outputNames];
  // Keep only tensors the loaded graph actually knows and that are float32.
  const selected = names.filter((name) => {
    const tensor = graph.getTensor(name);
    return tensor && tensor.dtype === 'float32';
  });
  if (selected.length === 0) return 0;

  let model;
  let compiled;
  let context;
  try {
    graph.setOutputs([...new Set([...original, ...selected])]);
    const runtime = await calibrationRuntime();
    model = runtime.createModel(graph);
    graph.setOutputs(original);
    compiled = await model.compile({
      backend: { mode: 'require', backend: BACKEND, operatorFallback: 'forbid' },
    });
    context = await compiled.createContext();
    for (const [index, inputs] of inputSets.entries()) {
      let result;
      try {
        result = await context.execute(inputs);
        for (const name of selected) {
          const values = await result.output(name).read();
          observeFiniteRange(observations, name, values, `${kind} runtime output`);
        }
        onSample?.(index);
      } finally {
        await result?.close().catch(() => undefined);
      }
    }
    return selected.length;
  } finally {
    await context?.close().catch(() => undefined);
    await compiled?.close().catch(() => undefined);
    await model?.close().catch(() => undefined);
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const packageDir = options.package;
  const fixtures = options.fixtures;
  const out = options.out;
  const requestedRoutingMode = normalizeCalibrationRoutingMode(
    options['routing-mode'] ?? 'auto',
  );
  const routeSweep = normalizeCalibrationRouteSweep(
    options['route-sweep'] ?? 'none',
  );
  if (!packageDir || !out || (!options['calibration-data'] && !fixtures)) {
    throw new Error(
      'pass --package <dir> --out <file.json> and either '
      + '--calibration-data <records.json> or --fixtures <dir>',
    );
  }
  const kinds = options.graph ? [options.graph] : ['encoder', 'decoder'];
  const report = {};
  const affineExclusions = {};
  const fixtureFileIdentities = {};
  positiveInteger(BATCH, 'calibration batch');
  const [packageArtifact, calibratorSource] = await Promise.all([
    // Quantization transforms the split package as one unit. Bind both source
    // graphs and both weight stores even when --graph limits an observation
    // sweep for diagnostics.
    packageIdentity(packageDir, SPLIT_GRAPH_KINDS),
    fileIdentity(CALIBRATOR_SOURCE_PATH, CALIBRATOR_LOGICAL_PATH),
  ]);

  // Calibrating on one input is the mistake this option exists to prevent. A
  // single-sample sweep produces ranges that fit that sample and clip every
  // other one: measured here, a package calibrated on one fixture reproduced
  // that fixture exactly and then read the wrong number off unseen receipts.
  // The affines have to see the spread of real inputs.
  const calibrationManifestPath = options['calibration-data'];
  const representative = calibrationManifestPath
    ? await representativeSamples(
      calibrationManifestPath, packageDir, Number(options.samples ?? 16),
    )
    : null;
  const samples = representative?.samples ?? null;
  const contract = representative?.contract ?? null;
  const requiredFamilyIds = representative?.requiredFamilyIds ?? [];
  if (!samples && requestedRoutingMode === 'explicit') {
    throw new Error('--routing-mode explicit requires --calibration-data records');
  }
  if (!samples && routeSweep !== 'none') {
    throw new Error('--route-sweep requires --calibration-data records');
  }
  if (samples) {
    const familyCounts = new Array(contract.familyNames.length).fill(0);
    for (const sample of samples) familyCounts[sample.expectedFamilyId]++;
    console.log(
      `calibrating over ${samples.length} representative samples on ${BACKEND}; `
      + `record-family coverage [${familyCounts.join(', ')}]`,
    );
  }

  const prefixesPerRecord = kinds.includes('decoder')
    ? positiveInteger(
      Number(options['prefixes-per-record'] ?? 3), 'prefixes per calibration record',
    )
    : 0;
  const executionSamples = samples
    ? calibrationRouteExecutions(samples, contract, routeSweep)
    : null;
  const encodedSamples = executionSamples
    ? await encodeSamples(
      packageDir, executionSamples, contract, requestedRoutingMode, routeSweep,
    )
    : null;
  if (encodedSamples) {
    const routingMode = encodedSamples[0].routingMode;
    const routedCounts = new Array(contract.familyNames.length).fill(0);
    for (const sample of encodedSamples) {
      routedCounts[sample.familyIds[0]]++;
    }
    console.log(`${routingMode} route-execution coverage `
      + `[${routedCounts.join(', ')}] over ${encodedSamples.length} execution(s); `
      + `semantic record labels remain unchanged`);
  }

  for (const kind of kinds) {
    const cachedGraph = await loadGraph(packageDir, kind);
    const { document } = cachedGraph;
    const maskExclusions = proveAdditiveAttentionMaskExclusions(
      document, (name) => cachedGraph.graph.getTensor(name),
    );
    const candidates = runtimeF32CandidateTensors(
      document, (name) => cachedGraph.graph.getTensor(name), maskExclusions.tensors,
    );
    affineExclusions[kind] = Object.freeze({
      semantic_domain: 'additive-attention-mask-{0,-Infinity}/v1',
      count: maskExclusions.tensors.length,
      tensors: maskExclusions.tensors,
      reasons: maskExclusions.reasons,
    });
    const observations = {};
    const inputSets = [];
    if (samples && kind === 'encoder') {
      for (const sample of executionSamples) {
        inputSets.push(bindEncoderCalibrationInputs(
          document, sample, contract.autoFamilyId, requestedRoutingMode,
          sample.routeFamilyId,
        ));
      }
    } else if (samples) {
      // The decoder reads `memory`, an encoder *output*, so its inputs only
      // exist once the encoder has run. Calibrating it from the fixture instead
      // is not a smaller approximation than it sounds: the shipped fixture image
      // is entirely zeros, so every downstream range came from a blank receipt.
      for (const produced of encodedSamples) {
        for (const prefixLength of teacherForcedPrefixLengths(
          produced.targetIds, prefixesPerRecord,
        )) {
          const prefix = buildTeacherForcedPrefix(
            produced.targetIds, prefixLength, contract,
          );
          inputSets.push(bindDecoderCalibrationInputs(document, produced, prefix));
        }
      }
    } else {
      const fixture = await fixtureInputs(fixtures, document, kind);
      inputSets.push(fixture.inputs);
      for (const [name, identity] of Object.entries(fixture.identities)) {
        const existing = fixtureFileIdentities[name];
        if (existing && (existing.sha256 !== identity.sha256
            || existing.bytes !== identity.bytes || existing.path !== identity.path)) {
          throw new Error(`fixture identity changed while reading ${name}`);
        }
        fixtureFileIdentities[name] = identity;
      }
    }

    for (const inputs of inputSets) observeInputs(document, inputs, observations);
    for (let start = 0; start < candidates.length; start += BATCH) {
      const batch = candidates.slice(start, start + BATCH);
      await observeBatch(
        packageDir, kind, inputSets, batch, observations, cachedGraph,
        (index) => process.stderr.write(
          `\r${kind}: sample ${index + 1}/${inputSets.length}, `
          + `${Math.min(start + BATCH, candidates.length)}/${candidates.length} tensors`,
        ),
      );
    }
    const widenedEmbeddings = widenImmutableEmbeddingRanges(
      document, (name) => cachedGraph.graph.getTensor(name), observations,
    );
    assertCompleteCalibrationRanges(document, candidates, observations, kind);
    process.stderr.write('\n');
    report[kind] = observations;
    console.log(`${kind}: observed ${Object.keys(observations).length} of `
      + `${candidates.length} candidate float32 tensors over ${inputSets.length} sample(s); `
      + `covered ${widenedEmbeddings} immutable Embedding table(s); `
      + `excluded ${maskExclusions.tensors.length} proven additive-mask tensor(s)`);

    // A calibration is only as good as what it was shown, and the failure mode
    // is silent: ranges derived from a degenerate input clip every real one,
    // while any check that replays the same input keeps reporting a match. The
    // shipped e2e fixture is a plumbing artifact whose image is 215040 zeros,
    // and using it here cost a full round of wrong answers before anyone looked
    // at a real receipt. Refuse to write a map that says the model saw nothing.
    const degenerate = Object.entries(observations)
      .filter(([, range]) => range.min === range.max);
    const collapsed = degenerate.length / Math.max(1, Object.keys(observations).length);
    if (collapsed > 0.2) {
      throw new Error(
        `${kind}: ${degenerate.length} of ${Object.keys(observations).length} observed `
        + 'tensors have a zero-width range. The calibration inputs carry no signal '
        + '— pass --calibration-data <records.json> --samples N to sweep representative data.',
      );
    }
    for (const [name, descriptor] of Object.entries(document.inputs || {})) {
      const range = observations[name];
      if ((descriptor?.dtype ?? 'float32') === 'float32'
          && range && range.min === range.max) {
        console.warn(`  WARNING ${kind} input '${name}' is constant `
          + `(${range.min}); every range downstream of it is meaningless.`);
      }
    }
  }
  const fixturesArtifact = representative
    ? null
    : Object.freeze({
      files: Object.freeze(Object.fromEntries(
        Object.entries(fixtureFileIdentities).sort(([left], [right]) =>
          left.localeCompare(right)),
      )),
    });
  const provenance = buildCalibrationProvenance({
    calibrationManifest: representative?.manifest ?? null,
    fixturesArtifact,
    samples: samples ?? [],
    requiredFamilyIds,
    routeExecutions: encodedSamples,
    routeSweep,
    contract,
    prefixesPerRecord,
    backend: BACKEND,
    batch: BATCH,
    kinds,
    routingMode: encodedSamples?.[0]?.routingMode ?? 'fixture',
    packageArtifact,
    calibratorSource,
    affineExclusions,
  });
  await writeFile(out, `${JSON.stringify(buildCalibrationReport(provenance, report), null, 1)}\n`);
  console.log(`wrote ${out}`);
}

if (process.argv[1]
    && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
