// Integrity and lifecycle helpers for transient parity artifacts.
//
// The wire objects produced here are deliberately plain JSON. Deno producers
// can import this module through their Node compatibility layer; Python and
// native producers can instead emit the same documented schema and SHA-256
// fields before atomically renaming their temporary file into place.

import {
  closeSync,
  existsSync,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readlinkSync,
  readSync,
  readdirSync,
  realpathSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';

export const PARITY_ARTIFACT_SCHEMA = 'volvoxai.parity-artifact';
export const PARITY_FINGERPRINT_SCHEMA = 'volvoxai.parity-fingerprint';
export const PARITY_RUN_MANIFEST_SCHEMA = 'volvoxai.parity-run-manifest';
export const PARITY_RUNTIME_EVIDENCE_SCHEMA = 'volvoxai.runtime-evidence';
export const PARITY_RUNTIME_FAILURE_EVIDENCE_SCHEMA =
  'volvoxai.runtime-failure-evidence';
export const PARITY_NATIVE_CAPABILITY_EVIDENCE_SCHEMA =
  'volvoxai.parity.native-capability-evidence';
export const PARITY_SCHEMA_VERSION = 1;

export const PARITY_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const REPO_ROOT = path.resolve(PARITY_ROOT, '..', '..');
export const PARITY_OUT_ROOT = path.join(PARITY_ROOT, 'out');

const RESULT_STATUSES = new Set(['success', 'error', 'expected-skip']);
const JOB_EXPECTATIONS = new Set(['required', 'expected-skip']);
const SHA256_RE = /^[0-9a-f]{64}$/;
const GLOB_META_RE = /[*?\[\]{}]/;

function assertNonEmptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function assertExactObject(value, label, keys) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new Error(`${label} must contain exactly: ${expected.join(', ')}`);
  }
  return value;
}

function assertNonNegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be a non-negative integer`);
  return value;
}

function assertRevision(value, label) {
  if (Number.isSafeInteger(value) && value >= 0) return value;
  if (typeof value === 'string' && /^(?:0|[1-9][0-9]*)$/.test(value)) return value;
  throw new Error(`${label} must be a non-negative safe integer or exact unsigned decimal string`);
}

function assertNullableString(value, label) {
  if (value !== null) assertNonEmptyString(value, label);
  return value;
}

function validateDeviceIdentity(value, label) {
  if (value === null) return value;
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length === 0) {
    throw new Error(`${label} must be a non-empty string map or null`);
  }
  for (const [key, entry] of Object.entries(value)) {
    assertNonEmptyString(key, `${label} key`);
    assertNonEmptyString(entry, `${label}.${key}`);
  }
  return value;
}

function validatePolicy(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      !['allow', 'forbid'].includes(value.operatorFallback)) {
    throw new Error(`${label} is invalid`);
  }
  if (value.mode === 'require') {
    assertExactObject(value, label, ['mode', 'backend', 'operatorFallback']);
    assertNonEmptyString(value.backend, `${label}.backend`);
  } else if (value.mode === 'prefer') {
    assertExactObject(value, label, ['mode', 'order', 'operatorFallback']);
    if (!Array.isArray(value.order) || value.order.length === 0) {
      throw new Error(`${label}.order must be a non-empty array`);
    }
    value.order.forEach((backend, index) => assertNonEmptyString(backend, `${label}.order[${index}]`));
  } else {
    throw new Error(`${label}.mode must be require or prefer`);
  }
  return value;
}

function validateOperatorRoute(value, label) {
  assertExactObject(value, label, ['attestation', 'used', 'offendingNode']);
  if (!['none', 'reported', 'unknown'].includes(value.attestation)) {
    throw new Error(`${label}.attestation is invalid`);
  }
  if (value.used !== null && typeof value.used !== 'boolean') {
    throw new Error(`${label}.used must be boolean or null`);
  }
  if (value.offendingNode !== null && typeof value.offendingNode !== 'string' &&
      !Number.isSafeInteger(value.offendingNode)) {
    throw new Error(`${label}.offendingNode must be a string, integer, or null`);
  }
  if (value.attestation === 'none' && value.used !== false) {
    throw new Error(`${label} contradicts its no-fallback attestation`);
  }
  return value;
}

function validateRoute(value, label) {
  assertExactObject(value, label, ['tierFallback', 'operator']);
  if (typeof value.tierFallback !== 'boolean') throw new Error(`${label}.tierFallback must be boolean`);
  validateOperatorRoute(value.operator, `${label}.operator`);
  return value;
}

function validateCompilationRevisions(value, label) {
  assertExactObject(value, label, [
    'topologyRevision', 'weightRevision', 'weightRevisionId',
    'adapterRevisionId', 'adapterRevisionIds',
  ]);
  assertRevision(value.topologyRevision, `${label}.topologyRevision`);
  assertRevision(value.weightRevision, `${label}.weightRevision`);
  assertNonEmptyString(value.weightRevisionId, `${label}.weightRevisionId`);
  assertNullableString(value.adapterRevisionId, `${label}.adapterRevisionId`);
  if (!Array.isArray(value.adapterRevisionIds)) throw new Error(`${label}.adapterRevisionIds must be an array`);
  value.adapterRevisionIds.forEach((id, index) =>
    assertNonEmptyString(id, `${label}.adapterRevisionIds[${index}]`));
  if (value.adapterRevisionId !== null && !value.adapterRevisionIds.includes(value.adapterRevisionId)) {
    throw new Error(`${label}.adapterRevisionId is not present in adapterRevisionIds`);
  }
  return value;
}

function validateExecutionRevisions(value, label) {
  assertExactObject(value, label, [
    'topologyRevision', 'weightRevision', 'weightRevisionId',
    'adapterRevisionId', 'adapterRevisionIds',
  ]);
  assertRevision(value.topologyRevision, `${label}.topologyRevision`);
  assertRevision(value.weightRevision, `${label}.weightRevision`);
  assertNonEmptyString(value.weightRevisionId, `${label}.weightRevisionId`);
  assertNullableString(value.adapterRevisionId, `${label}.adapterRevisionId`);
  if (!Array.isArray(value.adapterRevisionIds)) {
    throw new Error(`${label}.adapterRevisionIds must be an array`);
  }
  value.adapterRevisionIds.forEach((id, index) =>
    assertNullableString(id, `${label}.adapterRevisionIds[${index}]`));
  const expectedSingle = value.adapterRevisionIds.length === 1
    ? value.adapterRevisionIds[0]
    : null;
  if (value.adapterRevisionId !== expectedSingle) {
    throw new Error(`${label}.adapterRevisionId does not match adapterRevisionIds`);
  }
  return value;
}

function validateDecodeState(value, label) {
  assertExactObject(value, label, ['operation', 'mode', 'cacheState', 'cacheGeneration', 'position']);
  if (!['execute', 'seed', 'step'].includes(value.operation) ||
      !['not-applicable', 'seeded', 'advanced'].includes(value.cacheState)) {
    throw new Error(`${label} is invalid`);
  }
  assertNullableString(value.mode, `${label}.mode`);
  if (value.cacheGeneration !== null) {
    assertNonNegativeInteger(value.cacheGeneration, `${label}.cacheGeneration`);
  }
  if (value.position !== null) assertNonNegativeInteger(value.position, `${label}.position`);
  return value;
}

/** Validate the native CLI's machine-readable failure-channel report. */
export function validateRuntimeFailureEvidence(evidence) {
  assertExactObject(evidence, 'runtime failure evidence', [
    'schema', 'version', 'outcome', 'status', 'stage', 'request', 'source',
    'report', 'lineage',
  ]);
  if (evidence.schema !== PARITY_RUNTIME_FAILURE_EVIDENCE_SCHEMA ||
      evidence.version !== PARITY_SCHEMA_VERSION || evidence.outcome !== 'failure') {
    throw new Error('unsupported runtime failure evidence schema, version, or outcome');
  }
  assertNonEmptyString(evidence.status, 'runtime failure evidence.status');
  assertNonEmptyString(evidence.stage, 'runtime failure evidence.stage');

  const request = assertExactObject(evidence.request, 'runtime failure evidence.request', [
    'backend', 'policy',
  ]);
  if (request.backend !== null) assertNonEmptyString(request.backend, 'runtime failure request.backend');
  const policy = assertExactObject(request.policy, 'runtime failure evidence.request.policy', [
    'mode', 'operatorFallback',
  ]);
  if (!((policy.mode === 'require' && policy.operatorFallback === 'forbid' &&
         request.backend !== null) ||
        (policy.mode === 'prefer' && policy.operatorFallback === 'allow' &&
         request.backend === null))) {
    throw new Error('runtime failure evidence has inconsistent request policy');
  }

  const source = assertExactObject(evidence.source, 'runtime failure evidence.source', [
    'graphPath', 'weightPaths',
  ]);
  assertNonEmptyString(source.graphPath, 'runtime failure evidence.source.graphPath');
  if (!Array.isArray(source.weightPaths)) {
    throw new Error('runtime failure evidence.source.weightPaths must be an array');
  }
  const sourceWeightPaths = new Set();
  source.weightPaths.forEach((weightPath, index) => {
    assertNonEmptyString(weightPath, `runtime failure evidence.source.weightPaths[${index}]`);
    if (sourceWeightPaths.has(weightPath)) {
      throw new Error(`duplicate runtime failure evidence weight path: ${weightPath}`);
    }
    sourceWeightPaths.add(weightPath);
  });

  const report = assertExactObject(evidence.report, 'runtime failure evidence.report', [
    'backend', 'device', 'reason', 'message', 'offendingNode',
    'candidateOutcomes', 'routeEvidence', 'fallbackEvidence',
    'tierFallback', 'operatorFallbackUsed', 'routeAttested',
  ]);
  for (const field of [
    'backend', 'reason', 'message', 'offendingNode',
    'candidateOutcomes', 'routeEvidence', 'fallbackEvidence',
  ]) {
    assertNullableString(report[field], `runtime failure evidence.report.${field}`);
  }
  validateDeviceIdentity(report.device, 'runtime failure evidence.report.device');
  for (const field of ['tierFallback', 'operatorFallbackUsed', 'routeAttested']) {
    if (typeof report[field] !== 'boolean') {
      throw new Error(`runtime failure evidence.report.${field} must be boolean`);
    }
  }

  const lineage = assertExactObject(evidence.lineage, 'runtime failure evidence.lineage', [
    'runtimeId', 'modelId', 'compilationId', 'definitionId', 'weightRevisionId',
    'topologyRevision', 'weightRevision',
  ]);
  const identities = {
    runtimeId: 'runtime',
    modelId: 'model',
    compilationId: 'compiled',
    definitionId: 'graph',
    weightRevisionId: 'weight',
  };
  for (const [field, kind] of Object.entries(identities)) {
    if (lineage[field] !== null &&
        (typeof lineage[field] !== 'string' ||
         !new RegExp(`^native-${kind}-[1-9][0-9]*$`).test(lineage[field]))) {
      throw new Error(`runtime failure evidence.lineage.${field} is invalid`);
    }
  }
  for (const field of ['topologyRevision', 'weightRevision']) {
    if (lineage[field] !== null &&
        (typeof lineage[field] !== 'string' || !/^[1-9][0-9]*$/.test(lineage[field]))) {
      throw new Error(`runtime failure evidence.lineage.${field} is not null or a positive decimal string`);
    }
  }
  return evidence;
}

/** Validate the fail-closed wire evidence for an expected native GPU compile rejection. */
export function validateNativeCapabilityEvidence(evidence) {
  assertExactObject(evidence, 'native capability evidence', [
    'schema', 'version', 'campaign', 'authority', 'model', 'request',
    'adapter', 'result', 'runtimeFailureEvidence',
    'runtimeFailureEvidenceSha256', 'nativeLogSha256',
  ]);
  if (evidence.schema !== PARITY_NATIVE_CAPABILITY_EVIDENCE_SCHEMA ||
      evidence.version !== PARITY_SCHEMA_VERSION) {
    throw new Error('unsupported native capability evidence schema or version');
  }

  const campaign = assertExactObject(evidence.campaign, 'native capability campaign', [
    'id', 'sourceFingerprint', 'registryFingerprint',
  ]);
  for (const field of ['id', 'sourceFingerprint', 'registryFingerprint']) {
    if (!SHA256_RE.test(campaign[field] ?? '')) {
      throw new Error(`native capability campaign.${field} must be a SHA-256 digest`);
    }
  }

  const authority = assertExactObject(evidence.authority, 'native capability authority', [
    'kind', 'target', 'exporterQualified',
  ]);
  if (authority.kind !== 'kernel-registry' ||
      !['backend:vulkan', 'backend:opengl'].includes(authority.target) ||
      authority.exporterQualified !== false) {
    throw new Error('native capability evidence has invalid registry authority');
  }

  const model = assertExactObject(evidence.model, 'native capability model', [
    'id', 'graphPath', 'graphSha256', 'weights',
  ]);
  assertNonEmptyString(model.id, 'native capability model.id');
  normalizeRelativeLabel(model.graphPath, 'native capability model.graphPath');
  if (!SHA256_RE.test(model.graphSha256 ?? '')) {
    throw new Error('native capability model.graphSha256 must be a SHA-256 digest');
  }
  if (!Array.isArray(model.weights) || model.weights.length === 0) {
    throw new Error('native capability model.weights must be a non-empty array');
  }
  const weightPaths = new Set();
  for (const [index, weight] of model.weights.entries()) {
    const label = `native capability model.weights[${index}]`;
    assertExactObject(weight, label, ['path', 'sha256']);
    normalizeRelativeLabel(weight.path, `${label}.path`);
    if (weightPaths.has(weight.path)) throw new Error(`duplicate native capability weight: ${weight.path}`);
    weightPaths.add(weight.path);
    if (!SHA256_RE.test(weight.sha256 ?? '')) {
      throw new Error(`${label}.sha256 must be a SHA-256 digest`);
    }
  }

  const request = assertExactObject(evidence.request, 'native capability request', [
    'tier', 'backend', 'policy',
  ]);
  if (!['native-vulkan', 'native-opengl'].includes(request.tier) ||
      !['vulkan', 'opengl'].includes(request.backend) ||
      request.tier !== `native-${request.backend}`) {
    throw new Error('native capability evidence has invalid tier/backend request');
  }
  const requestPolicy = assertExactObject(request.policy, 'native capability request.policy', [
    'mode', 'operatorFallback',
  ]);
  if (requestPolicy.mode !== 'require' || requestPolicy.operatorFallback !== 'forbid') {
    throw new Error('native capability evidence did not request a strict no-fallback backend');
  }
  validateDeviceIdentity(evidence.adapter, 'native capability adapter');

  const result = assertExactObject(evidence.result, 'native capability result', [
    'outcome', 'exitCode', 'status', 'stage', 'reason', 'offendingNode', 'message',
  ]);
  if (result.outcome !== 'expected-compile-rejection' ||
      !Number.isSafeInteger(result.exitCode) || result.exitCode === 0 ||
      result.status !== 'BACKEND_UNSUPPORTED' || result.stage !== 'compile') {
    throw new Error('native capability evidence is not an expected compile rejection');
  }
  assertNonEmptyString(result.reason, 'native capability result.reason');
  assertNonEmptyString(result.message, 'native capability result.message');
  if (result.offendingNode !== null && typeof result.offendingNode !== 'string' &&
      !Number.isSafeInteger(result.offendingNode)) {
    throw new Error('native capability result.offendingNode must be a string, integer, or null');
  }
  if (!SHA256_RE.test(evidence.nativeLogSha256 ?? '')) {
    throw new Error('native capability nativeLogSha256 must be a SHA-256 digest');
  }
  validateRuntimeFailureEvidence(evidence.runtimeFailureEvidence);
  if (!SHA256_RE.test(evidence.runtimeFailureEvidenceSha256 ?? '')) {
    throw new Error('native capability runtimeFailureEvidenceSha256 must be a SHA-256 digest');
  }
  return evidence;
}

/** Validate the exact machine-readable lifecycle evidence sealed into JS parity manifests. */
export function validateRuntimeEvidence(evidence) {
  assertExactObject(evidence, 'runtime evidence', [
    'schema', 'version', 'compilation', 'execution', 'stableResult',
  ]);
  if (evidence.schema !== PARITY_RUNTIME_EVIDENCE_SCHEMA || evidence.version !== PARITY_SCHEMA_VERSION) {
    throw new Error('unsupported runtime evidence schema or version');
  }

  const compilation = assertExactObject(evidence.compilation, 'runtime evidence compilation', [
    'compilationId', 'policy', 'selectedBackend', 'selectedDevice',
    'definitionId', 'revisions', 'route',
  ]);
  assertNonEmptyString(compilation.compilationId, 'runtime evidence compilation.compilationId');
  validatePolicy(compilation.policy, 'runtime evidence compilation.policy');
  assertNonEmptyString(compilation.selectedBackend, 'runtime evidence compilation.selectedBackend');
  validateDeviceIdentity(compilation.selectedDevice, 'runtime evidence compilation.selectedDevice');
  assertNonEmptyString(compilation.definitionId, 'runtime evidence compilation.definitionId');
  validateCompilationRevisions(compilation.revisions, 'runtime evidence compilation.revisions');
  validateRoute(compilation.route, 'runtime evidence compilation.route');
  if (compilation.policy.mode === 'require' &&
      (compilation.policy.backend !== compilation.selectedBackend || compilation.route.tierFallback)) {
    throw new Error('runtime evidence required-backend selection used tier fallback');
  }

  const execution = assertExactObject(evidence.execution, 'runtime evidence execution', [
    'executionId', 'contextId', 'backend', 'device', 'outcome',
    'revisions', 'route', 'decodeState',
  ]);
  assertNonEmptyString(execution.executionId, 'runtime evidence execution.executionId');
  assertNonEmptyString(execution.contextId, 'runtime evidence execution.contextId');
  assertNonEmptyString(execution.backend, 'runtime evidence execution.backend');
  validateDeviceIdentity(execution.device, 'runtime evidence execution.device');
  if (execution.outcome !== 'success') throw new Error('runtime evidence execution did not succeed');
  validateExecutionRevisions(execution.revisions, 'runtime evidence execution.revisions');
  validateRoute(execution.route, 'runtime evidence execution.route');
  validateDecodeState(execution.decodeState, 'runtime evidence execution.decodeState');
  const availableAdapterRevisions = new Set(compilation.revisions.adapterRevisionIds);
  const executionUsesUnknownAdapter = execution.revisions.adapterRevisionIds.some(
    (revision) => revision !== null && !availableAdapterRevisions.has(revision),
  );
  if (execution.backend !== compilation.selectedBackend ||
      canonicalJson(execution.device) !== canonicalJson(compilation.selectedDevice) ||
      execution.revisions.topologyRevision !== compilation.revisions.topologyRevision ||
      execution.revisions.weightRevision !== compilation.revisions.weightRevision ||
      execution.revisions.weightRevisionId !== compilation.revisions.weightRevisionId ||
      executionUsesUnknownAdapter ||
      execution.route.tierFallback !== compilation.route.tierFallback) {
    throw new Error('runtime evidence execution does not match its compilation');
  }

  const stable = assertExactObject(evidence.stableResult, 'runtime evidence stableResult', [
    'outputs', 'freshCallerOwnedReads', 'readableAfterContextClose',
    'contextClosedBeforeResult', 'resultClosedAfterVerification',
  ]);
  if (!Array.isArray(stable.outputs) || stable.outputs.length === 0) {
    throw new Error('runtime evidence stableResult.outputs must be a non-empty array');
  }
  const names = new Set();
  const bytesPerElement = { float32: 4, int32: 4, int8: 1, uint8: 1 };
  for (const [index, output] of stable.outputs.entries()) {
    const label = `runtime evidence stableResult.outputs[${index}]`;
    assertExactObject(output, label, ['name', 'shape', 'dtype', 'location', 'byteLength']);
    assertNonEmptyString(output.name, `${label}.name`);
    if (names.has(output.name)) throw new Error(`runtime evidence has duplicate output '${output.name}'`);
    names.add(output.name);
    if (!Array.isArray(output.shape) || output.shape.length === 0) throw new Error(`${label}.shape is invalid`);
    let elements = 1;
    for (const [dimensionIndex, dimension] of output.shape.entries()) {
      if (!Number.isSafeInteger(dimension) || dimension <= 0 ||
          !Number.isSafeInteger(elements * dimension)) {
        throw new Error(`${label}.shape[${dimensionIndex}] is invalid`);
      }
      elements *= dimension;
    }
    if (!(output.dtype in bytesPerElement) || !['host', 'device'].includes(output.location) ||
        output.byteLength !== elements * bytesPerElement[output.dtype]) {
      throw new Error(`${label} has inconsistent dtype, location, or byte length`);
    }
  }
  for (const field of [
    'freshCallerOwnedReads', 'readableAfterContextClose',
    'contextClosedBeforeResult', 'resultClosedAfterVerification',
  ]) {
    if (stable[field] !== true) throw new Error(`runtime evidence stableResult.${field} must be true`);
  }
  return evidence;
}

/** Validate the strict opaque-native CLI evidence consumed by native parity tiers. */
export function validateNativeRuntimeEvidence(evidence, expectedBackend) {
  validateRuntimeEvidence(evidence);
  assertNonEmptyString(expectedBackend, 'expected native backend');
  const { compilation, execution } = evidence;
  if (compilation.policy.mode !== 'require' ||
      compilation.policy.backend !== expectedBackend ||
      compilation.policy.operatorFallback !== 'forbid' ||
      compilation.selectedBackend !== expectedBackend ||
      execution.backend !== expectedBackend) {
    throw new Error(`native runtime evidence does not prove strict '${expectedBackend}' selection`);
  }
  for (const [label, report] of [
    ['compilation', compilation],
    ['execution', execution],
  ]) {
    if (report.route.tierFallback ||
        report.route.operator.attestation !== 'reported' ||
        report.route.operator.used !== false ||
        report.route.operator.offendingNode !== null) {
      throw new Error(`native ${label} evidence does not attest an exact no-fallback operator route`);
    }
    if (typeof report.revisions.topologyRevision !== 'string' ||
        !/^[0-9]+$/.test(report.revisions.topologyRevision) ||
        typeof report.revisions.weightRevision !== 'string' ||
        !/^[0-9]+$/.test(report.revisions.weightRevision) ||
        !/^native-weight-[0-9]+$/.test(report.revisions.weightRevisionId) ||
        !report.revisions.adapterRevisionIds.every(
          (revision) => revision === null || /^native-adapter-[0-9]+:[0-9]+$/.test(revision),
        )) {
      throw new Error(`native ${label} revision evidence is incomplete`);
    }
  }
  if (!/^native-compiled-[0-9]+$/.test(compilation.compilationId) ||
      !/^native-graph-[0-9]+$/.test(compilation.definitionId) ||
      !/^native-execution-[0-9]+$/.test(execution.executionId) ||
      !/^native-context-[0-9]+$/.test(execution.contextId)) {
    throw new Error('native runtime evidence has invalid lifecycle identities');
  }
  return evidence;
}

/** Reduce public compilation/result reports and an observed stability check to the parity wire schema. */
export function createRuntimeEvidence({ compilation, execution, stableResult } = {}) {
  // Runtime reports may add provider-local decode telemetry without changing
  // this versioned parity wire schema. Project the v1 lifecycle fields instead
  // of retaining the report object by reference and accidentally widening the
  // signed artifact whenever execution diagnostics evolve.
  const decodeState = execution?.decodeState == null
    ? execution?.decodeState
    : {
        operation: execution.decodeState.operation,
        mode: execution.decodeState.mode,
        cacheState: execution.decodeState.cacheState,
        cacheGeneration: execution.decodeState.cacheGeneration,
        position: execution.decodeState.position,
      };
  const evidence = {
    schema: PARITY_RUNTIME_EVIDENCE_SCHEMA,
    version: PARITY_SCHEMA_VERSION,
    compilation: {
      compilationId: compilation?.compilationId,
      policy: compilation?.requestedPolicy,
      selectedBackend: compilation?.selectedBackend,
      selectedDevice: compilation?.selectedDevice ?? null,
      definitionId: compilation?.definitionId,
      revisions: {
        topologyRevision: compilation?.topologyRevision,
        weightRevision: compilation?.weightRevision,
        weightRevisionId: compilation?.weightRevisionId,
        adapterRevisionId: compilation?.adapterRevisionId ?? null,
        adapterRevisionIds: compilation?.adapterRevisionIds,
      },
      route: compilation?.routeEvidence,
    },
    execution: {
      executionId: execution?.executionId,
      contextId: execution?.contextId,
      backend: execution?.backend,
      device: execution?.device ?? null,
      outcome: execution?.outcome,
      revisions: {
        topologyRevision: execution?.topologyRevision,
        weightRevision: execution?.weightRevision,
        weightRevisionId: execution?.weightRevisionId,
        adapterRevisionId: execution?.adapterRevisionId ?? null,
        adapterRevisionIds: execution?.adapterRevisionIds,
      },
      route: execution?.routeEvidence,
      decodeState,
    },
    stableResult,
  };
  validateRuntimeEvidence(evidence);
  return canonicalValue(evidence);
}

function canonicalValue(value, seen = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('canonical JSON cannot contain non-finite numbers');
    return Object.is(value, -0) ? 0 : value;
  }
  if (typeof value !== 'object') throw new Error(`canonical JSON cannot contain ${typeof value}`);
  if (seen.has(value)) throw new Error('canonical JSON cannot contain cycles');
  seen.add(value);
  if (Array.isArray(value)) {
    const result = value.map((item) => canonicalValue(item, seen));
    seen.delete(value);
    return result;
  }
  const result = {};
  for (const key of Object.keys(value).sort()) {
    if (value[key] === undefined) throw new Error(`canonical JSON cannot contain undefined at ${key}`);
    result[key] = canonicalValue(value[key], seen);
  }
  seen.delete(value);
  return result;
}

/** Deterministic compact JSON, used only as hash input. */
export function canonicalJson(value) {
  return JSON.stringify(canonicalValue(value));
}

export function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

/** Hash a file without loading a large safetensors or build artifact at once. */
export function sha256FileSync(file) {
  const fd = openSync(file, 'r');
  const hash = createHash('sha256');
  const chunk = Buffer.allocUnsafe(1024 * 1024);
  let size = 0;
  try {
    for (;;) {
      const n = readSync(fd, chunk, 0, chunk.length, null);
      if (n === 0) break;
      hash.update(chunk.subarray(0, n));
      size += n;
    }
    const after = fstatSync(fd);
    if (!after.isFile() || after.size !== size) {
      throw new Error(`file changed while hashing: ${file}`);
    }
  } finally {
    closeSync(fd);
  }
  return { sha256: hash.digest('hex'), size };
}

function relativeFileLabel(file, root) {
  const absolute = path.resolve(file);
  const relative = path.relative(path.resolve(root), absolute);
  if (relative === '' || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error(`fingerprinted file must be below repository root: ${file}`);
  }
  return relative.split(path.sep).join('/');
}

function normalizeRelativeLabel(value, label) {
  const normalized = assertNonEmptyString(value, label).split(path.sep).join('/').replace(/^\.\//, '');
  const canonical = path.posix.normalize(normalized);
  if (canonical === '.' || path.posix.isAbsolute(canonical) || canonical === '..' ||
      canonical.startsWith('../') || canonical !== normalized || GLOB_META_RE.test(canonical)) {
    throw new Error(`${label} must be an exact relative path`);
  }
  return canonical;
}

function normalizeFileSpecs(specs, root) {
  const byLabel = new Map();
  for (const spec of specs) {
    const file = typeof spec === 'string' ? spec : spec?.file;
    assertNonEmptyString(file, 'fingerprint file');
    const absolute = path.isAbsolute(file) ? path.resolve(file) : path.resolve(root, file);
    const label = typeof spec === 'object' && spec?.label != null
      ? normalizeRelativeLabel(spec.label, 'fingerprint file label')
      : relativeFileLabel(absolute, root);
    if (byLabel.has(label) && byLabel.get(label) !== absolute) {
      throw new Error(`duplicate fingerprint label points to different files: ${label}`);
    }
    byLabel.set(label, absolute);
  }
  return [...byLabel]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([label, file]) => ({ path: label, ...sha256FileSync(file) }));
}

const SOURCE_STATE_EXCLUDES = [
  ':(exclude)tests/parity/out/**',
  ':(exclude)tests/parity/goldens/**',
  ':(exclude)tests/parity/backward/out/**',
  ':(exclude)tests/parity/decode/out/**',
  ':(exclude)tests/parity/kvcache/out/**',
];

/**
 * Git revision plus a content hash of tracked changes and untracked source.
 * Generated parity outputs and committed golden artifacts are excluded so a
 * producer does not invalidate its own manifest while publishing results.
 */
export function sourceStateSync(repoRoot = REPO_ROOT) {
  try {
    const revision = execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    const trackedDiff = execFileSync(
      'git', ['diff', '--binary', 'HEAD', '--', '.', ...SOURCE_STATE_EXCLUDES], {
        cwd: repoRoot,
        encoding: 'buffer',
        maxBuffer: 256 * 1024 * 1024,
        stdio: ['ignore', 'pipe', 'ignore'],
      },
    );
    const untrackedOutput = execFileSync(
      'git', ['ls-files', '--others', '--exclude-standard', '-z', '--', '.', ...SOURCE_STATE_EXCLUDES], {
        cwd: repoRoot,
        encoding: 'buffer',
        maxBuffer: 64 * 1024 * 1024,
        stdio: ['ignore', 'pipe', 'ignore'],
      },
    );
    const untracked = untrackedOutput.toString('utf8').split('\0').filter(Boolean).sort()
      .map((label) => {
        const normalized = normalizeRelativeLabel(label, 'untracked source path');
        const file = path.resolve(repoRoot, normalized);
        const info = lstatSync(file);
        if (info.isSymbolicLink()) {
          const target = readlinkSync(file, 'utf8');
          return { path: normalized, type: 'symlink', sha256: sha256Bytes(target), size: Buffer.byteLength(target) };
        }
        if (!info.isFile()) throw new Error(`untracked source is not a file: ${normalized}`);
        return { path: normalized, type: 'file', ...sha256FileSync(file) };
      });
    const contentState = canonicalJson({
      trackedDiffSha256: sha256Bytes(trackedDiff),
      untracked,
    });
    const dirty = trackedDiff.length > 0 || untracked.length > 0;
    return {
      revision,
      state: dirty ? 'dirty' : 'clean',
      statusSha256: sha256Bytes(contentState),
    };
  } catch {
    return { revision: null, state: 'unavailable', statusSha256: null };
  }
}

/**
 * Collect graph.json, every top-level safetensors file, and declared file
 * inputs for selected whole-model policy entries.
 */
export function policyModelFilesSync(policy, {
  repoRoot = REPO_ROOT,
  selectedCases = Object.keys(policy?.models ?? {}),
} = {}) {
  if (!policy || typeof policy !== 'object' || !policy.models || typeof policy.models !== 'object') {
    throw new Error('parity policy must contain a models object');
  }
  const models = [];
  const inputs = [];
  for (const caseId of [...new Set(selectedCases)].sort()) {
    const model = policy.models[caseId];
    if (!model) throw new Error(`unknown parity policy case: ${caseId}`);
    const modelDir = path.resolve(repoRoot, assertNonEmptyString(model.dir, `${caseId}.dir`));
    models.push(path.join(modelDir, 'graph.json'));
    const weights = statSync(modelDir).isDirectory()
      ? readdirSync(modelDir).filter((name) => name.endsWith('.safetensors')).sort()
      : [];
    if (weights.length === 0) throw new Error(`no safetensors file found for ${caseId}: ${modelDir}`);
    for (const weight of weights) models.push(path.join(modelDir, weight));
    for (const input of model.inputs ?? []) {
      if (input.file) inputs.push(path.resolve(repoRoot, input.file));
    }
  }
  return { models, inputs };
}

/**
 * Build the stable fingerprint shared by all producer manifests in a parity
 * campaign. Generated input files must be passed through `fixtureFiles` after
 * they are materialized; tier-specific JS/WASM/native files go in buildFiles.
 */
export function createParityFingerprintSync({
  repoRoot = REPO_ROOT,
  policyFile = path.join(PARITY_ROOT, 'policy.json'),
  selectedCases,
  fixtureFiles = [],
  buildFiles = [],
  sourceFiles = [],
  extraModelFiles = [],
  source = sourceStateSync(repoRoot),
} = {}) {
  const policyPath = path.isAbsolute(policyFile) ? policyFile : path.resolve(repoRoot, policyFile);
  const policy = JSON.parse(readFileSync(policyPath, 'utf8'));
  const collected = policyModelFilesSync(policy, { repoRoot, selectedCases });
  const components = {
    policy: normalizeFileSpecs([policyPath], repoRoot),
    models: normalizeFileSpecs([...collected.models, ...extraModelFiles], repoRoot),
    inputs: normalizeFileSpecs([...collected.inputs, ...fixtureFiles], repoRoot),
    builds: normalizeFileSpecs(buildFiles, repoRoot),
    sources: normalizeFileSpecs(sourceFiles, repoRoot),
    source: canonicalValue(source),
  };
  const base = {
    schema: PARITY_FINGERPRINT_SCHEMA,
    version: PARITY_SCHEMA_VERSION,
    algorithm: 'sha256',
    components,
  };
  return { ...base, digest: sha256Bytes(canonicalJson(base)) };
}

export function validateParityFingerprint(fingerprint) {
  if (fingerprint?.schema !== PARITY_FINGERPRINT_SCHEMA || fingerprint?.version !== PARITY_SCHEMA_VERSION) {
    throw new Error('unsupported parity fingerprint schema or version');
  }
  if (fingerprint.algorithm !== 'sha256' || !SHA256_RE.test(fingerprint.digest ?? '')) {
    throw new Error('invalid parity fingerprint digest');
  }
  for (const group of ['policy', 'models', 'inputs', 'builds', 'sources']) {
    if (!Array.isArray(fingerprint.components?.[group])) {
      throw new Error(`parity fingerprint components.${group} must be an array`);
    }
    for (const entry of fingerprint.components[group]) {
      normalizeRelativeLabel(entry?.path, `fingerprint ${group} path`);
      if (!SHA256_RE.test(entry?.sha256 ?? '') || !Number.isSafeInteger(entry?.size) || entry.size < 0) {
        throw new Error(`invalid parity fingerprint ${group} file record`);
      }
    }
  }
  if (!fingerprint.components.source || typeof fingerprint.components.source !== 'object') {
    throw new Error('parity fingerprint components.source must be an object');
  }
  const { digest, ...base } = fingerprint;
  const expected = sha256Bytes(canonicalJson(base));
  if (digest !== expected) throw new Error('parity fingerprint digest does not match its components');
  return fingerprint;
}

function pathInsideRoot(candidate, root, { allowRoot = false, createParent = false } = {}) {
  const rootPath = path.resolve(root);
  const value = assertNonEmptyString(candidate, 'output path');
  if (GLOB_META_RE.test(value)) throw new Error(`output path must be exact, not a glob: ${value}`);
  const absolute = path.isAbsolute(value) ? path.resolve(value) : path.resolve(rootPath, value);
  const relative = path.relative(rootPath, absolute);
  const escapes = relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative);
  if (escapes || (!allowRoot && relative === '')) {
    throw new Error(`output path escapes or names parity out root: ${value}`);
  }
  if (createParent) {
    mkdirSync(rootPath, { recursive: true });
    let existing = path.dirname(absolute);
    while (!existsSync(existing)) existing = path.dirname(existing);
    assertRealPathInsideRoot(existing, rootPath, value);
    mkdirSync(path.dirname(absolute), { recursive: true });
  }
  if (existsSync(rootPath)) {
    const parent = path.dirname(absolute);
    if (existsSync(parent)) assertRealPathInsideRoot(parent, rootPath, value);
  }
  return absolute;
}

function assertRealPathInsideRoot(candidate, root, original) {
  const realRoot = realpathSync(root);
  const realCandidate = realpathSync(candidate);
  const relative = path.relative(realRoot, realCandidate);
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error(`output path follows a symlink outside parity out root: ${original}`);
  }
}

export function outputRelativePath(file, outputRoot = PARITY_OUT_ROOT) {
  const absolute = pathInsideRoot(file, outputRoot);
  return path.relative(path.resolve(outputRoot), absolute).split(path.sep).join('/');
}

/** Atomically replace one JSON file beneath the explicitly supplied root. */
export function atomicWriteJsonSync(file, value, { outputRoot = PARITY_OUT_ROOT } = {}) {
  canonicalJson(value); // Reject cycles, undefined values, and non-finite numbers before touching disk.
  const serialized = JSON.stringify(value, null, 2);
  if (serialized === undefined) throw new Error('JSON artifact cannot be undefined');
  const destination = pathInsideRoot(file, outputRoot, { createParent: true });
  const temporary = path.join(
    path.dirname(destination),
    `.${path.basename(destination)}.${process.pid}.${randomUUID()}.tmp`,
  );
  const content = `${serialized}\n`;
  let fd;
  try {
    fd = openSync(temporary, 'wx', 0o600);
    writeFileSync(fd, content, 'utf8');
    fsyncSync(fd);
    closeSync(fd);
    fd = undefined;
    renameSync(temporary, destination);
  } catch (error) {
    if (fd !== undefined) closeSync(fd);
    rmSync(temporary, { force: true });
    throw error;
  }
  return destination;
}

/**
 * Remove only the exact paths supplied beneath tests/parity/out. Directories
 * require an explicit opt-in; the out root itself and glob-like paths reject.
 */
export function removeParityOutputsSync(files, {
  outputRoot = PARITY_OUT_ROOT,
  allowDirectories = false,
} = {}) {
  if (!Array.isArray(files)) throw new Error('parity output removal requires an explicit path array');
  const targets = [];
  const seen = new Set();
  for (const file of files) {
    const target = pathInsideRoot(file, outputRoot);
    if (seen.has(target)) continue;
    seen.add(target);
    if (!existsSync(target)) continue;
    const info = lstatSync(target);
    if (info.isDirectory() && !info.isSymbolicLink() && !allowDirectories) {
      throw new Error(`refusing to remove parity output directory without allowDirectories: ${file}`);
    }
    targets.push({ target, info });
  }
  const removed = [];
  for (const { target, info } of targets) {
    rmSync(target, { force: true, recursive: info.isDirectory() && !info.isSymbolicLink() });
    removed.push(outputRelativePath(target, outputRoot));
  }
  return removed;
}

function normalizeJob(job) {
  const caseId = assertNonEmptyString(job?.case ?? job?.caseId, 'manifest job case');
  const tier = assertNonEmptyString(job?.tier, 'manifest job tier');
  const expectation = job?.expectation ?? 'required';
  if (!JOB_EXPECTATIONS.has(expectation)) throw new Error(`invalid manifest job expectation: ${expectation}`);
  return { case: caseId, tier, expectation };
}

function jobKey(job) {
  return `${job.case}\0${job.tier}`;
}

function assertResultSlot(manifest, caseId, tier) {
  const key = jobKey({ case: caseId, tier });
  if (!manifest.selection.jobs.some((candidate) => jobKey(candidate) === key)) {
    throw new Error(`result is outside manifest selection: ${caseId}/${tier}`);
  }
  if (manifest.results.some((candidate) => jobKey(candidate) === key)) {
    throw new Error(`duplicate manifest result: ${caseId}/${tier}`);
  }
}

function resultRequiresRuntimeEvidence(manifest, tier) {
  return manifest.producer?.runtimeEvidenceRequired === true ||
    manifest.producer?.runtimeEvidenceTiers?.includes(tier) === true;
}

function resultRequiresCapabilityEvidence(manifest, tier) {
  return manifest.producer?.capabilityEvidenceRequired === true ||
    manifest.producer?.capabilityEvidenceTiers?.includes(tier) === true;
}

function derivedManifestOutcome(jobs, results) {
  if (jobs.size !== results.size) return null;
  for (const [key, result] of results) {
    const expectation = jobs.get(key).expectation;
    if (result.status === 'error' || (result.status === 'expected-skip' && expectation !== 'expected-skip')) {
      return 'error';
    }
  }
  return 'success';
}

/** Create one producer-command manifest with an explicit case x tier job set. */
export function createRunManifest({
  command,
  fingerprint,
  jobs,
  producer = { kind: 'node' },
  runId = randomUUID(),
  startedAt = new Date().toISOString(),
} = {}) {
  assertNonEmptyString(command, 'manifest command');
  assertNonEmptyString(runId, 'manifest runId');
  validateParityFingerprint(fingerprint);
  if (!Array.isArray(jobs) || jobs.length === 0) throw new Error('manifest jobs must be a non-empty array');
  const normalized = jobs.map(normalizeJob).sort((a, b) => jobKey(a).localeCompare(jobKey(b)));
  const keys = new Set();
  for (const job of normalized) {
    const key = jobKey(job);
    if (keys.has(key)) throw new Error(`duplicate manifest job: ${job.case}/${job.tier}`);
    keys.add(key);
  }
  return {
    schema: PARITY_RUN_MANIFEST_SCHEMA,
    version: PARITY_SCHEMA_VERSION,
    runId,
    command,
    fingerprint,
    selection: {
      cases: [...new Set(normalized.map((job) => job.case))],
      tiers: [...new Set(normalized.map((job) => job.tier))],
      jobs: normalized,
    },
    producer: canonicalValue(producer),
    startedAt,
    completedAt: null,
    outcome: null,
    results: [],
  };
}

export function recordRunResult(manifest, result) {
  validateRunManifest(manifest);
  const normalized = normalizeJob(result);
  assertResultSlot(manifest, normalized.case, normalized.tier);
  const status = result?.status;
  if (!RESULT_STATUSES.has(status)) throw new Error(`invalid manifest result status: ${status}`);
  const entry = { case: normalized.case, tier: normalized.tier, status };
  if (status === 'success') {
    const artifact = result.artifact;
    if (!artifact || typeof artifact !== 'object') throw new Error('successful result must identify its artifact');
    const artifactPath = normalizeRelativeLabel(artifact.path, 'artifact path');
    if (!SHA256_RE.test(artifact.sha256 ?? '') || !Number.isSafeInteger(artifact.size) || artifact.size < 0) {
      throw new Error('successful result has invalid artifact SHA-256 or size');
    }
    entry.artifact = { path: artifactPath, sha256: artifact.sha256, size: artifact.size };
  } else if (status === 'error') {
    entry.error = assertNonEmptyString(result.error instanceof Error ? result.error.message : result.error, 'result error');
  } else {
    entry.reason = assertNonEmptyString(result.reason, 'expected skip reason');
  }
  if (result.durationMs != null) {
    if (!Number.isFinite(result.durationMs) || result.durationMs < 0) throw new Error('durationMs must be finite and non-negative');
    entry.durationMs = result.durationMs;
  }
  if (resultRequiresRuntimeEvidence(manifest, normalized.tier) &&
      result.metadata?.runtimeEvidence == null && status === 'success') {
    throw new Error('successful runtime parity result must include runtime evidence');
  }
  if (resultRequiresCapabilityEvidence(manifest, normalized.tier) &&
      (status !== 'expected-skip' || result.metadata?.capabilityEvidence == null)) {
    throw new Error('native capability job must record an expected skip with capability evidence');
  }
  if (result.metadata?.runtimeEvidence != null) {
    validateRuntimeEvidence(result.metadata.runtimeEvidence);
  }
  if (result.metadata?.capabilityEvidence != null) {
    if (status !== 'expected-skip') {
      throw new Error('native capability evidence is valid only for an expected skip');
    }
    validateNativeCapabilityEvidence(result.metadata.capabilityEvidence);
  }
  if (result.metadata != null) entry.metadata = canonicalValue(result.metadata);
  manifest.results.push(entry);
  return entry;
}

function artifactEnvelope(manifest, caseId, tier, payload, createdAt) {
  return {
    schema: PARITY_ARTIFACT_SCHEMA,
    version: PARITY_SCHEMA_VERSION,
    provenance: {
      runId: manifest.runId,
      command: manifest.command,
      fingerprint: manifest.fingerprint.digest,
      case: caseId,
      tier,
      createdAt,
    },
    payload,
  };
}

/** Atomically emit an artifact envelope and record its exact bytes in-memory. */
export function writeRunArtifactSync({
  manifest,
  file,
  case: caseName,
  caseId = caseName,
  tier,
  payload,
  outputRoot = PARITY_OUT_ROOT,
  createdAt = new Date().toISOString(),
  durationMs,
  metadata,
} = {}) {
  validateRunManifest(manifest);
  assertNonEmptyString(caseId, 'artifact case');
  assertNonEmptyString(tier, 'artifact tier');
  assertResultSlot(manifest, caseId, tier);
  if (resultRequiresRuntimeEvidence(manifest, tier) && metadata?.runtimeEvidence == null) {
    throw new Error('successful runtime parity result must include runtime evidence');
  }
  if (metadata?.runtimeEvidence != null) validateRuntimeEvidence(metadata.runtimeEvidence);
  const destination = atomicWriteJsonSync(
    file,
    artifactEnvelope(manifest, caseId, tier, payload, createdAt),
    { outputRoot },
  );
  const digest = sha256FileSync(destination);
  recordRunResult(manifest, {
    case: caseId,
    tier,
    status: 'success',
    artifact: { path: outputRelativePath(destination, outputRoot), ...digest },
    durationMs,
    metadata,
  });
  return destination;
}

export function validateRunManifest(manifest, {
  requireComplete = false,
  requireFinalized = false,
  expectedFingerprint,
  expectedRunId,
} = {}) {
  if (manifest?.schema !== PARITY_RUN_MANIFEST_SCHEMA || manifest?.version !== PARITY_SCHEMA_VERSION) {
    throw new Error('unsupported parity run manifest schema or version');
  }
  assertNonEmptyString(manifest.runId, 'manifest runId');
  assertNonEmptyString(manifest.command, 'manifest command');
  validateParityFingerprint(manifest.fingerprint);
  const expectedDigest = typeof expectedFingerprint === 'string'
    ? expectedFingerprint
    : expectedFingerprint?.digest;
  if (expectedDigest != null && manifest.fingerprint.digest !== expectedDigest) {
    throw new Error('manifest fingerprint does not match the current parity campaign');
  }
  if (expectedRunId != null && manifest.runId !== expectedRunId) {
    throw new Error('manifest runId does not match the current parity campaign');
  }
  if (!Array.isArray(manifest.selection?.jobs) || !Array.isArray(manifest.results)) {
    throw new Error('manifest selection.jobs and results must be arrays');
  }
  if (manifest.producer?.runtimeEvidenceRequired != null &&
      typeof manifest.producer.runtimeEvidenceRequired !== 'boolean') {
    throw new Error('manifest producer.runtimeEvidenceRequired must be boolean');
  }
  if (manifest.producer?.runtimeEvidenceTiers != null) {
    if (!Array.isArray(manifest.producer.runtimeEvidenceTiers) ||
        new Set(manifest.producer.runtimeEvidenceTiers).size !==
          manifest.producer.runtimeEvidenceTiers.length) {
      throw new Error('manifest producer.runtimeEvidenceTiers must be a unique array');
    }
    manifest.producer.runtimeEvidenceTiers.forEach((tier, index) =>
      assertNonEmptyString(tier, `manifest producer.runtimeEvidenceTiers[${index}]`));
  }
  if (manifest.producer?.capabilityEvidenceRequired != null &&
      typeof manifest.producer.capabilityEvidenceRequired !== 'boolean') {
    throw new Error('manifest producer.capabilityEvidenceRequired must be boolean');
  }
  if (manifest.producer?.capabilityEvidenceTiers != null) {
    if (!Array.isArray(manifest.producer.capabilityEvidenceTiers) ||
        new Set(manifest.producer.capabilityEvidenceTiers).size !==
          manifest.producer.capabilityEvidenceTiers.length) {
      throw new Error('manifest producer.capabilityEvidenceTiers must be a unique array');
    }
    manifest.producer.capabilityEvidenceTiers.forEach((tier, index) =>
      assertNonEmptyString(tier, `manifest producer.capabilityEvidenceTiers[${index}]`));
  }
  const jobs = new Map();
  for (const raw of manifest.selection.jobs) {
    const job = normalizeJob(raw);
    const key = jobKey(job);
    if (jobs.has(key)) throw new Error(`duplicate manifest job: ${job.case}/${job.tier}`);
    jobs.set(key, job);
  }
  const results = new Map();
  for (const result of manifest.results) {
    const key = jobKey(normalizeJob(result));
    if (!jobs.has(key)) throw new Error(`manifest result is outside selection: ${result.case}/${result.tier}`);
    if (results.has(key)) throw new Error(`duplicate manifest result: ${result.case}/${result.tier}`);
    if (!RESULT_STATUSES.has(result.status)) throw new Error(`invalid manifest result status: ${result.status}`);
    if (result.status === 'success') {
      if (!result.artifact || !SHA256_RE.test(result.artifact.sha256 ?? '') ||
          !Number.isSafeInteger(result.artifact.size) || result.artifact.size < 0) {
        throw new Error(`invalid successful artifact record: ${result.case}/${result.tier}`);
      }
      normalizeRelativeLabel(result.artifact.path, 'artifact path');
      if (resultRequiresRuntimeEvidence(manifest, result.tier) &&
          result.metadata?.runtimeEvidence == null) {
        throw new Error(`successful runtime result lacks evidence: ${result.case}/${result.tier}`);
      }
      if (result.metadata?.runtimeEvidence != null) {
        validateRuntimeEvidence(result.metadata.runtimeEvidence);
      }
    } else if (result.status === 'error') {
      assertNonEmptyString(result.error, 'result error');
    } else {
      assertNonEmptyString(result.reason, 'expected skip reason');
    }
    if (resultRequiresCapabilityEvidence(manifest, result.tier) &&
        (result.status !== 'expected-skip' || result.metadata?.capabilityEvidence == null)) {
      throw new Error(`native capability result lacks evidence: ${result.case}/${result.tier}`);
    }
    if (result.metadata?.capabilityEvidence != null) {
      if (result.status !== 'expected-skip') {
        throw new Error(`native capability evidence is attached to non-skip: ${result.case}/${result.tier}`);
      }
      validateNativeCapabilityEvidence(result.metadata.capabilityEvidence);
    }
    results.set(key, result);
  }
  if (requireComplete && jobs.size !== results.size) {
    const missing = [...jobs].filter(([key]) => !results.has(key)).map(([, job]) => `${job.case}/${job.tier}`);
    throw new Error(`manifest is incomplete; missing results: ${missing.join(', ')}`);
  }
  if ((manifest.completedAt == null) !== (manifest.outcome == null)) {
    throw new Error('manifest completedAt and outcome must be set together');
  }
  if (manifest.outcome != null) {
    const expectedOutcome = derivedManifestOutcome(jobs, results);
    if (expectedOutcome == null || manifest.outcome !== expectedOutcome) {
      throw new Error('manifest outcome does not match its selected job results');
    }
  }
  if (requireFinalized && (manifest.completedAt == null || !['success', 'error'].includes(manifest.outcome))) {
    throw new Error('manifest is not finalized');
  }
  return manifest;
}

/** Complete a manifest and derive an outcome from every selected job. */
export function finalizeRunManifest(manifest, { completedAt = new Date().toISOString() } = {}) {
  validateRunManifest(manifest, { requireComplete: true });
  const jobs = new Map(manifest.selection.jobs.map((job) => [jobKey(job), job]));
  const results = new Map(manifest.results.map((result) => [jobKey(result), result]));
  manifest.completedAt = completedAt;
  manifest.outcome = derivedManifestOutcome(jobs, results);
  return manifest;
}

export function writeRunManifestSync(file, manifest, { outputRoot = PARITY_OUT_ROOT } = {}) {
  validateRunManifest(manifest);
  return atomicWriteJsonSync(file, manifest, { outputRoot });
}

export function readRunManifestSync(file, {
  outputRoot = PARITY_OUT_ROOT,
  requireComplete = false,
  requireFinalized = false,
  expectedFingerprint,
  expectedRunId,
} = {}) {
  const source = pathInsideRoot(file, outputRoot);
  const manifest = JSON.parse(readFileSync(source, 'utf8'));
  return validateRunManifest(manifest, {
    requireComplete,
    requireFinalized,
    expectedFingerprint,
    expectedRunId,
  });
}

/**
 * Validate schema, current run/fingerprint provenance, manifest record, and
 * exact artifact bytes. Returns only the producer payload.
 */
export function readRunArtifactSync(file, manifest, {
  case: caseName,
  caseId = caseName,
  tier,
  outputRoot = PARITY_OUT_ROOT,
} = {}) {
  validateRunManifest(manifest);
  assertNonEmptyString(caseId, 'artifact case');
  assertNonEmptyString(tier, 'artifact tier');
  const source = pathInsideRoot(file, outputRoot);
  const relative = outputRelativePath(source, outputRoot);
  const record = manifest.results.find((result) =>
    result.case === caseId && result.tier === tier && result.status === 'success');
  if (!record) throw new Error(`manifest has no successful result for ${caseId}/${tier}`);
  if (record.artifact.path !== relative) throw new Error(`artifact path is not recorded for ${caseId}/${tier}`);
  const actual = sha256FileSync(source);
  if (actual.sha256 !== record.artifact.sha256 || actual.size !== record.artifact.size) {
    throw new Error(`artifact bytes do not match manifest for ${caseId}/${tier}`);
  }
  const artifact = JSON.parse(readFileSync(source, 'utf8'));
  if (artifact?.schema !== PARITY_ARTIFACT_SCHEMA || artifact?.version !== PARITY_SCHEMA_VERSION) {
    throw new Error(`unsupported parity artifact schema or version: ${relative}`);
  }
  const provenance = artifact.provenance;
  if (provenance?.runId !== manifest.runId ||
      provenance?.fingerprint !== manifest.fingerprint.digest ||
      provenance?.case !== caseId || provenance?.tier !== tier ||
      provenance?.command !== manifest.command) {
    throw new Error(`artifact provenance does not match current run fingerprint: ${caseId}/${tier}`);
  }
  return artifact.payload;
}
