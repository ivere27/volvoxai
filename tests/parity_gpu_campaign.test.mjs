import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mkdtempSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { sha256FileSync } from './parity/lib/artifact.mjs';
import {
  physicalAdapterFromCapabilityLog,
  requireExactConsensusModels,
  requireExactPhysicalAdapterSet,
  requireFiniteFloat32Buffers,
  requireNativeCapabilityEvidenceAgainstExpectation,
} from './parity/lib/gpu_campaign.mjs';

const fp32 = { model: 'efficientdet_lite0_fp32', tolerance: 1e-5 };
const int8 = { model: 'efficientdet_lite0_int8', tolerance: 0 };

test('GPU consensus requires each expected model exactly once', () => {
  assert.equal(requireExactConsensusModels([int8, fp32]).length, 2);
  assert.throws(() => requireExactConsensusModels([fp32, fp32]), /duplicate model/);
  assert.throws(() => requireExactConsensusModels([fp32]), /missing model.*int8/);
  assert.throws(
    () => requireExactConsensusModels([fp32, { model: int8.model, tolerance: 1 }]),
    /unexpected tolerance/,
  );
  assert.throws(
    () => requireExactConsensusModels([fp32, { model: 'unknown', tolerance: 0 }]),
    /unexpected model/,
  );
});

test('GPU consensus rejects identical non-finite float32 outputs before exact comparison', () => {
  const finite = new Uint8Array(Float32Array.of(1, -2, 3).buffer);
  assert.equal(
    requireFiniteFloat32Buffers(
      [finite, finite, finite],
      ['webgpu', 'native-opengl', 'native-vulkan'],
      'consensus',
    ).length,
    3,
  );

  const nonFinite = new Uint8Array(Float32Array.of(Number.NaN).buffer);
  assert.throws(
    () => requireFiniteFloat32Buffers(
      [nonFinite, nonFinite, nonFinite],
      ['webgpu', 'native-opengl', 'native-vulkan'],
      'consensus',
    ),
    /webgpu is non-finite at 0/,
  );
});

test('GPU consensus requires the exact physical adapter evidence set', () => {
  const adapters = {
    webgpu: { device: 'NVIDIA GeForce RTX 3090' },
    'native-vulkan': { backend: 'vulkan', device: 'NVIDIA GeForce RTX 3090' },
    'native-opengl': { backend: 'opengl', device: 'NVIDIA GeForce RTX 3090' },
  };
  assert.deepEqual(
    requireExactPhysicalAdapterSet(adapters, 'consensus', { requiredIdentity: 'RTX 3090' }),
    adapters,
  );
  assert.throws(
    () => requireExactPhysicalAdapterSet({ ...adapters, 'native-vulkan': undefined }, 'consensus'),
    /identity is unavailable/,
  );
  const { ['native-opengl']: _missing, ...incomplete } = adapters;
  assert.throws(() => requireExactPhysicalAdapterSet(incomplete, 'consensus'), /adapter evidence is/);
  assert.throws(
    () => requireExactPhysicalAdapterSet({ ...adapters, cuda: adapters.webgpu }, 'consensus'),
    /adapter evidence is/,
  );
  assert.throws(
    () => requireExactPhysicalAdapterSet({
      ...adapters,
      'native-vulkan': { backend: 'opengl', device: 'NVIDIA GeForce RTX 3090' },
    }, 'consensus'),
    /native adapter backend is 'opengl', expected 'vulkan'/,
  );
});

function capabilityExpectation({
  modelId = 'efficientdet_lite0_fp32',
  graphPath = `models/${modelId}/graph.json`,
  weightPaths = [`models/${modelId}/model.safetensors`],
} = {}) {
  return {
    campaign: {
      id: '1'.repeat(64),
      sourceFingerprint: '2'.repeat(64),
      registryFingerprint: '3'.repeat(64),
    },
    authority: {
      kind: 'kernel-registry', target: 'backend:vulkan', exporterQualified: false,
    },
    model: {
      id: modelId,
      graphPath,
      graphSha256: '4'.repeat(64),
      weights: weightPaths.map((weightPath, index) => ({
        path: weightPath,
        sha256: ((5 + index) % 16).toString(16).repeat(64),
      })),
    },
    request: {
      tier: 'native-vulkan',
      backend: 'vulkan',
      policy: { mode: 'require', operatorFallback: 'forbid' },
    },
    result: {
      outcome: 'expected-compile-rejection',
      status: 'BACKEND_UNSUPPORTED',
      stage: 'compile',
      reason: 'CANONICAL_SHAPE_CONTRACT_UNSUPPORTED',
    },
  };
}

function failureEvidence(message, expected, source = {
  graphPath: expected.model.graphPath,
  weightPaths: expected.model.weights.map((weight) => weight.path),
}) {
  return {
    schema: 'volvoxai.runtime-failure-evidence',
    version: 1,
    outcome: 'failure',
    status: 'BACKEND_UNSUPPORTED',
    stage: 'compile',
    request: {
      backend: expected.request.backend,
      policy: expected.request.policy,
    },
    source,
    report: {
      backend: expected.request.backend,
      device: { backend: expected.request.backend, device: `builtin:${expected.request.backend}` },
      reason: expected.result.reason,
      message,
      offendingNode: null,
      candidateOutcomes: 'vulkan:domain-unsupported',
      routeEvidence: 'bounded_domain=unsupported',
      fallbackEvidence: 'tier=forbidden;operator=forbidden',
      tierFallback: false,
      operatorFallbackUsed: false,
      routeAttested: true,
    },
    lineage: {
      runtimeId: 'native-runtime-1',
      modelId: 'native-model-2',
      compilationId: 'native-compiled-3',
      definitionId: 'native-graph-4',
      weightRevisionId: 'native-weight-5',
      topologyRevision: '6',
      weightRevision: '1',
    },
  };
}

function writeJson(file, value) {
  writeFileSync(file, `${JSON.stringify(value)}\n`);
  return sha256FileSync(file).sha256;
}

function capabilityFixture(t, {
  successRoute = false,
  expected = capabilityExpectation(),
  failureSource,
} = {}) {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'volvox-native-capability-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const message = 'node lacks a canonical native shape contract';
  const configuredAdapter = process.env.VOLVOXAI_PARITY_GPU_ADAPTER ||
    process.env.VOLVOXAI_PARITY_WEBGPU_ADAPTER || 'Parity Test';
  const log = [
    `[VolvoxAI GPU] Vulkan Compute initialized successfully! Device: ${configuredAdapter} Discrete GPU; packed INT8 dot: unavailable`,
    ...(successRoute ? ['Backend: vulkan'] : []),
    `Native init failed: ${expected.result.status} [${expected.result.reason}]: ${message}`,
    '',
  ].join('\n');
  const nativeLog = path.join(dir, 'native.log');
  writeFileSync(nativeLog, log);
  const rawFailure = failureEvidence(message, expected, failureSource);
  const runtimeFailureEvidenceFile = path.join(dir, 'runtime-failure-evidence.json');
  const runtimeFailureEvidenceSha256 = writeJson(runtimeFailureEvidenceFile, rawFailure);
  const adapter = physicalAdapterFromCapabilityLog(log, 'vulkan', 'test capability log');
  const evidence = {
    schema: 'volvoxai.parity.native-capability-evidence',
    version: 1,
    campaign: expected.campaign,
    authority: expected.authority,
    model: expected.model,
    request: expected.request,
    adapter,
    result: {
      ...expected.result,
      exitCode: 1,
      offendingNode: null,
      message,
    },
    runtimeFailureEvidence: rawFailure,
    runtimeFailureEvidenceSha256,
    nativeLogSha256: sha256FileSync(nativeLog).sha256,
  };
  return { evidence, expected, nativeLog, runtimeFailureEvidenceFile };
}

function validateCapabilityFixture(fixture) {
  return requireNativeCapabilityEvidenceAgainstExpectation(
    fixture.evidence,
    fixture.expected,
    {
      nativeLog: fixture.nativeLog,
      runtimeFailureEvidenceFile: fixture.runtimeFailureEvidenceFile,
    },
  );
}

test('native GPU capability evidence is bound to an exact expectation and raw reports', (t) => {
  const fixture = capabilityFixture(t);
  const accepted = validateCapabilityFixture(fixture);
  assert.equal(accepted.result.reason, 'CANONICAL_SHAPE_CONTRACT_UNSUPPORTED');
  assert.equal(accepted.authority.exporterQualified, false);
});

test('native GPU capability evidence rejects stale identity, wrong reason, fallback, and tampering', (t) => {
  const fixture = capabilityFixture(t);
  const options = {
    nativeLog: fixture.nativeLog,
    runtimeFailureEvidenceFile: fixture.runtimeFailureEvidenceFile,
  };

  const stale = structuredClone(fixture.evidence);
  stale.campaign.id = '0'.repeat(64);
  assert.throws(
    () => requireNativeCapabilityEvidenceAgainstExpectation(stale, fixture.expected, options),
    /stale or mismatched campaign/,
  );

  const wrongReason = structuredClone(fixture.evidence);
  wrongReason.result.reason = 'BACKEND_UNAVAILABLE';
  assert.throws(
    () => requireNativeCapabilityEvidenceAgainstExpectation(wrongReason, fixture.expected, options),
    /unexpected result.reason/,
  );

  const absentCompileLineage = structuredClone(fixture.evidence);
  absentCompileLineage.runtimeFailureEvidence.lineage.compilationId = null;
  assert.throws(
    () => requireNativeCapabilityEvidenceAgainstExpectation(
      absentCompileLineage,
      fixture.expected,
      options,
    ),
    /lacks real compile lineage/,
  );

  const fallback = structuredClone(fixture.evidence);
  fallback.runtimeFailureEvidence.report.operatorFallbackUsed = true;
  writeJson(fixture.runtimeFailureEvidenceFile, fallback.runtimeFailureEvidence);
  fallback.runtimeFailureEvidenceSha256 = sha256FileSync(fixture.runtimeFailureEvidenceFile).sha256;
  assert.throws(
    () => requireNativeCapabilityEvidenceAgainstExpectation(fallback, fixture.expected, options),
    /contradicts the capability rejection/,
  );

  writeFileSync(fixture.runtimeFailureEvidenceFile, '{}\n');
  assert.throws(
    () => requireNativeCapabilityEvidenceAgainstExpectation(
      fixture.evidence,
      fixture.expected,
      options,
    ),
    /digest does not match/,
  );
});

test('native GPU capability evidence rejects a raw failure from a different model source', (t) => {
  const expected = capabilityExpectation({ modelId: 'efficientdet_lite0_fp32' });
  const fixture = capabilityFixture(t, {
    expected,
    failureSource: {
      graphPath: 'models/efficientdet_lite0_int8/graph.json',
      weightPaths: ['models/efficientdet_lite0_int8/model.safetensors'],
    },
  });
  assert.throws(
    () => validateCapabilityFixture(fixture),
    /runtime failure evidence has the wrong model source/,
  );
});

test('native GPU capability evidence rejects a successful execution route line', (t) => {
  assert.throws(
    () => capabilityFixture(t, { successRoute: true }),
    /rejected execution\/success route line/,
  );
});
