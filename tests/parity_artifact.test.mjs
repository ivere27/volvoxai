import test from 'node:test';
import assert from 'node:assert/strict';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

import {
  PARITY_SCHEMA_VERSION,
  atomicWriteJsonSync,
  canonicalJson,
  createParityFingerprintSync,
  createRuntimeEvidence,
  createRunManifest,
  finalizeRunManifest,
  readRunArtifactSync,
  readRunManifestSync,
  recordRunResult,
  removeParityOutputsSync,
  sha256Bytes,
  sourceStateSync,
  validateNativeCapabilityEvidence,
  validateNativeRuntimeEvidence,
  validateParityFingerprint,
  validateRuntimeFailureEvidence,
  validateRuntimeEvidence,
  writeRunArtifactSync,
  writeRunManifestSync,
} from './parity/lib/artifact.mjs';

function runtimeEvidenceFixture() {
  return createRuntimeEvidence({
    compilation: {
      compilationId: 'compilation-1',
      requestedPolicy: {
        mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid',
      },
      selectedBackend: 'cpu-js',
      selectedDevice: { type: 'host' },
      definitionId: 'definition-1',
      topologyRevision: 0,
      weightRevision: 3,
      weightRevisionId: 'definition-1:weight:3',
      adapterRevisionId: null,
      adapterRevisionIds: ['tenant:v2'],
      routeEvidence: {
        tierFallback: false,
        operator: { attestation: 'none', used: false, offendingNode: null },
      },
    },
    execution: {
      executionId: 'execution-2',
      contextId: 'context-1',
      backend: 'cpu-js',
      device: { type: 'host' },
      outcome: 'success',
      topologyRevision: 0,
      weightRevision: 3,
      weightRevisionId: 'definition-1:weight:3',
      adapterRevisionId: null,
      adapterRevisionIds: [null],
      routeEvidence: {
        tierFallback: false,
        operator: { attestation: 'none', used: false, offendingNode: null },
      },
      decodeState: {
        operation: 'execute',
        mode: null,
        cacheState: 'not-applicable',
        cacheGeneration: null,
        position: null,
      },
    },
    stableResult: {
      outputs: [{
        name: 'y', shape: [1, 2], dtype: 'float32', location: 'host', byteLength: 8,
      }],
      freshCallerOwnedReads: true,
      readableAfterContextClose: true,
      contextClosedBeforeResult: true,
      resultClosedAfterVerification: true,
    },
  });
}

function runtimeFailureEvidenceFixture() {
  return {
    schema: 'volvoxai.runtime-failure-evidence',
    version: 1,
    outcome: 'failure',
    status: 'BACKEND_UNSUPPORTED',
    stage: 'compile',
    request: {
      backend: 'vulkan',
      policy: { mode: 'require', operatorFallback: 'forbid' },
    },
    source: {
      graphPath: 'models/toy/graph.json',
      weightPaths: ['models/toy/model.safetensors'],
    },
    report: {
      backend: 'vulkan',
      device: { backend: 'vulkan', device: 'builtin:vulkan' },
      reason: 'BOUNDED_DOMAIN_UNSUPPORTED',
      message: 'built-in backend cannot attest the complete declared shape domain',
      offendingNode: null,
      candidateOutcomes: 'vulkan:domain-unsupported',
      routeEvidence: 'domain=unsupported',
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

function nativeCapabilityEvidenceFixture() {
  return {
    schema: 'volvoxai.parity.native-capability-evidence',
    version: 1,
    campaign: {
      id: '1'.repeat(64),
      sourceFingerprint: '2'.repeat(64),
      registryFingerprint: '3'.repeat(64),
    },
    authority: {
      kind: 'kernel-registry', target: 'backend:vulkan', exporterQualified: false,
    },
    model: {
      id: 'toy',
      graphPath: 'models/toy/graph.json',
      graphSha256: '4'.repeat(64),
      weights: [{ path: 'models/toy/model.safetensors', sha256: '5'.repeat(64) }],
    },
    request: {
      tier: 'native-vulkan',
      backend: 'vulkan',
      policy: { mode: 'require', operatorFallback: 'forbid' },
    },
    adapter: { backend: 'vulkan', device: 'Example GPU' },
    result: {
      outcome: 'expected-compile-rejection',
      exitCode: 1,
      status: 'BACKEND_UNSUPPORTED',
      stage: 'compile',
      reason: 'BOUNDED_DOMAIN_UNSUPPORTED',
      offendingNode: null,
      message: 'built-in backend cannot attest the complete declared shape domain',
    },
    runtimeFailureEvidence: runtimeFailureEvidenceFixture(),
    runtimeFailureEvidenceSha256: '6'.repeat(64),
    nativeLogSha256: '7'.repeat(64),
  };
}

function fixture(t) {
  const repoRoot = mkdtempSync(path.join(os.tmpdir(), 'volvox-parity-artifact-'));
  t.after(() => rmSync(repoRoot, { recursive: true, force: true }));
  const modelDir = path.join(repoRoot, 'models', 'toy');
  const inputDir = path.join(repoRoot, 'inputs');
  const fixtureDir = path.join(repoRoot, 'tests', 'parity', 'out', 'fixtures');
  const buildDir = path.join(repoRoot, 'dist', '1.0.0');
  mkdirSync(modelDir, { recursive: true });
  mkdirSync(inputDir, { recursive: true });
  mkdirSync(fixtureDir, { recursive: true });
  mkdirSync(buildDir, { recursive: true });
  writeFileSync(path.join(modelDir, 'graph.json'),
    '{"format":"volvox-graph/v1","nodes":[]}\n');
  writeFileSync(path.join(modelDir, 'model.safetensors'), 'weights-v1');
  writeFileSync(path.join(inputDir, 'tokens.i32'), 'tokens-v1');
  writeFileSync(path.join(fixtureDir, 'generated.u8'), 'fixture-v1');
  writeFileSync(path.join(buildDir, 'volvoxai.js'), 'build-v1');
  const policyFile = path.join(repoRoot, 'tests', 'parity', 'policy.json');
  mkdirSync(path.dirname(policyFile), { recursive: true });
  const sourceFile = path.join(repoRoot, 'tests', 'parity', 'harness.mjs');
  writeFileSync(sourceFile, 'export const harness = 1;\n');
  writeFileSync(policyFile, JSON.stringify({
    models: {
      toy: {
        dir: 'models/toy',
        inputs: [
          { name: 'tokens', file: 'inputs/tokens.i32', dtype: 'i32' },
          { name: 'pixels', gen: 'seededU8', seed: 7 },
        ],
      },
    },
  }));
  const outputRoot = path.join(repoRoot, 'tests', 'parity', 'out');
  const options = {
    repoRoot,
    policyFile,
    selectedCases: ['toy'],
    fixtureFiles: [path.join(fixtureDir, 'generated.u8')],
    buildFiles: [path.join(buildDir, 'volvoxai.js')],
    sourceFiles: [sourceFile],
    source: { revision: '0123456789abcdef', state: 'clean', statusSha256: sha256Bytes('') },
  };
  return { repoRoot, outputRoot, options };
}

test('parity fingerprint hashes policy, model, inputs, build, and source deterministically', (t) => {
  const { options } = fixture(t);
  const first = createParityFingerprintSync(options);
  const second = createParityFingerprintSync({ ...options, fixtureFiles: [...options.fixtureFiles].reverse() });

  assert.equal(first.version, PARITY_SCHEMA_VERSION);
  assert.equal(first.digest, second.digest);
  assert.deepEqual(first.components.models.map((entry) => entry.path), [
    'models/toy/graph.json',
    'models/toy/model.safetensors',
  ]);
  assert.deepEqual(first.components.inputs.map((entry) => entry.path), [
    'inputs/tokens.i32',
    'tests/parity/out/fixtures/generated.u8',
  ]);
  assert.equal(first.components.builds[0].path, 'dist/1.0.0/volvoxai.js');
  assert.equal(first.components.sources[0].path, 'tests/parity/harness.mjs');
  assert.equal(validateParityFingerprint(first), first);

  writeFileSync(options.fixtureFiles[0], 'fixture-v2');
  const changed = createParityFingerprintSync(options);
  assert.notEqual(changed.digest, first.digest);
  writeFileSync(options.sourceFiles[0], 'export const harness = 2;\n');
  assert.notEqual(createParityFingerprintSync(options).digest, changed.digest);

  const forged = structuredClone(first);
  forged.components.source.state = 'dirty';
  assert.throws(() => validateParityFingerprint(forged), /does not match its components/);
});

test('dirty Git source fingerprints hash content, not only the changed path list', (t) => {
  const repoRoot = mkdtempSync(path.join(os.tmpdir(), 'volvox-parity-source-state-'));
  t.after(() => rmSync(repoRoot, { recursive: true, force: true }));
  const git = (...args) => execFileSync('git', args, { cwd: repoRoot, stdio: 'ignore' });
  git('init', '-q');
  writeFileSync(path.join(repoRoot, 'tracked.mjs'), 'export const value = 1;\n');
  git('add', 'tracked.mjs');
  git('-c', 'user.name=Parity Test', '-c', 'user.email=parity@example.invalid', 'commit', '-qm', 'fixture');

  const clean = sourceStateSync(repoRoot);
  assert.equal(clean.state, 'clean');
  writeFileSync(path.join(repoRoot, 'tracked.mjs'), 'export const value = 2;\n');
  const dirtyTwo = sourceStateSync(repoRoot);
  writeFileSync(path.join(repoRoot, 'tracked.mjs'), 'export const value = 3;\n');
  const dirtyThree = sourceStateSync(repoRoot);
  assert.equal(dirtyTwo.state, 'dirty');
  assert.notEqual(dirtyTwo.statusSha256, dirtyThree.statusSha256);

  writeFileSync(path.join(repoRoot, 'untracked.mjs'), 'export const extra = 1;\n');
  const untrackedOne = sourceStateSync(repoRoot);
  writeFileSync(path.join(repoRoot, 'untracked.mjs'), 'export const extra = 2;\n');
  const untrackedTwo = sourceStateSync(repoRoot);
  assert.notEqual(untrackedOne.statusSha256, untrackedTwo.statusSha256);
});

test('canonical JSON is key-order stable and rejects cyclic arrays', () => {
  assert.equal(canonicalJson({ z: 1, a: { d: 2, b: 3 } }), '{"a":{"b":3,"d":2},"z":1}');
  const cyclic = [];
  cyclic.push(cyclic);
  assert.throws(() => canonicalJson(cyclic), /cycles/);
});

test('runtime evidence binds selection, revisions, routing, context, and stable outputs', () => {
  const evidence = runtimeEvidenceFixture();
  assert.equal(validateRuntimeEvidence(evidence), evidence);
  assert.equal(evidence.compilation.selectedBackend, 'cpu-js');
  assert.equal(evidence.execution.contextId, 'context-1');
  assert.equal(evidence.execution.revisions.weightRevisionId, 'definition-1:weight:3');
  assert.equal(evidence.stableResult.readableAfterContextClose, true);

  const noAdapters = structuredClone(evidence);
  noAdapters.compilation.revisions.adapterRevisionIds = [];
  noAdapters.execution.revisions.adapterRevisionIds = [];
  assert.equal(validateRuntimeEvidence(noAdapters), noAdapters);

  const exactNativeRevisions = structuredClone(evidence);
  exactNativeRevisions.compilation.revisions.topologyRevision = '18446744073709551615';
  exactNativeRevisions.execution.revisions.topologyRevision = '18446744073709551615';
  exactNativeRevisions.compilation.revisions.weightRevision = '9007199254740992';
  exactNativeRevisions.execution.revisions.weightRevision = '9007199254740992';
  assert.equal(validateRuntimeEvidence(exactNativeRevisions), exactNativeRevisions);

  const invalidExactRevision = structuredClone(exactNativeRevisions);
  invalidExactRevision.execution.revisions.topologyRevision = '-1';
  assert.throws(() => validateRuntimeEvidence(invalidExactRevision), /exact unsigned decimal string/);

  const native = structuredClone(exactNativeRevisions);
  native.compilation.compilationId = 'native-compiled-11';
  native.compilation.definitionId = 'native-graph-7';
  native.compilation.policy.backend = 'cpu';
  native.compilation.selectedBackend = 'cpu';
  native.compilation.revisions.weightRevisionId = 'native-weight-9';
  native.compilation.revisions.adapterRevisionId = 'native-adapter-10:1';
  native.compilation.revisions.adapterRevisionIds = ['native-adapter-10:1'];
  native.compilation.route.operator.attestation = 'reported';
  native.execution.executionId = 'native-execution-13';
  native.execution.contextId = 'native-context-12';
  native.execution.backend = 'cpu';
  native.execution.revisions.weightRevisionId = 'native-weight-9';
  native.execution.revisions.adapterRevisionId = 'native-adapter-10:1';
  native.execution.revisions.adapterRevisionIds = ['native-adapter-10:1'];
  native.execution.route.operator.attestation = 'reported';
  assert.equal(validateNativeRuntimeEvidence(native, 'cpu'), native);

  const unattestedNative = structuredClone(native);
  unattestedNative.execution.route.operator.attestation = 'unknown';
  assert.throws(
    () => validateNativeRuntimeEvidence(unattestedNative, 'cpu'),
    /exact no-fallback operator route/,
  );

  const fallback = structuredClone(evidence);
  fallback.compilation.route.tierFallback = true;
  assert.throws(() => validateRuntimeEvidence(fallback), /required-backend selection used tier fallback/);

  const mismatchedRevision = structuredClone(evidence);
  mismatchedRevision.execution.revisions.weightRevision = 4;
  assert.throws(() => validateRuntimeEvidence(mismatchedRevision), /does not match its compilation/);

  const unknownAdapterRevision = structuredClone(evidence);
  unknownAdapterRevision.execution.revisions.adapterRevisionId = 'tenant:v3';
  unknownAdapterRevision.execution.revisions.adapterRevisionIds = ['tenant:v3'];
  assert.throws(
    () => validateRuntimeEvidence(unknownAdapterRevision),
    /does not match its compilation/,
  );

  const inconsistentAdapterRevision = structuredClone(evidence);
  inconsistentAdapterRevision.execution.revisions.adapterRevisionId = 'tenant:v2';
  assert.throws(
    () => validateRuntimeEvidence(inconsistentAdapterRevision),
    /does not match adapterRevisionIds/,
  );

  const unstable = structuredClone(evidence);
  unstable.stableResult.readableAfterContextClose = false;
  assert.throws(() => validateRuntimeEvidence(unstable), /must be true/);

  const extraField = structuredClone(evidence);
  extraField.execution.retry = 'cpu-js';
  assert.throws(() => validateRuntimeEvidence(extraField), /must contain exactly/);
});

test('native capability wire evidence is strict and embeds the typed CLI failure report', () => {
  const failure = runtimeFailureEvidenceFixture();
  assert.equal(validateRuntimeFailureEvidence(failure), failure);
  const evidence = nativeCapabilityEvidenceFixture();
  assert.equal(validateNativeCapabilityEvidence(evidence), evidence);

  const absentLineage = structuredClone(failure);
  for (const field of Object.keys(absentLineage.lineage)) absentLineage.lineage[field] = null;
  assert.equal(validateRuntimeFailureEvidence(absentLineage), absentLineage);

  const zeroIdentity = structuredClone(failure);
  zeroIdentity.lineage.runtimeId = 'native-runtime-0';
  assert.throws(() => validateRuntimeFailureEvidence(zeroIdentity), /lineage.runtimeId is invalid/);

  const zeroRevision = structuredClone(failure);
  zeroRevision.lineage.weightRevision = '0';
  assert.throws(() => validateRuntimeFailureEvidence(zeroRevision), /positive decimal string/);

  const duplicateWeightPath = structuredClone(failure);
  duplicateWeightPath.source.weightPaths.push(duplicateWeightPath.source.weightPaths[0]);
  assert.throws(
    () => validateRuntimeFailureEvidence(duplicateWeightPath),
    /duplicate runtime failure evidence weight path/,
  );

  const fallback = structuredClone(evidence);
  fallback.runtimeFailureEvidence.report.operatorFallbackUsed = true;
  assert.equal(
    validateNativeCapabilityEvidence(fallback),
    fallback,
    'wire validation preserves typed failures; contextual policy rejects fallback use',
  );

  const untypedLineage = structuredClone(evidence);
  untypedLineage.runtimeFailureEvidence.lineage.compilationId = '3';
  assert.throws(
    () => validateNativeCapabilityEvidence(untypedLineage),
    /lineage.compilationId is invalid/,
  );

  const extra = structuredClone(evidence);
  extra.runtimeFailureEvidence.report.retryBackend = 'cpu-js';
  assert.throws(() => validateNativeCapabilityEvidence(extra), /must contain exactly/);

  const successExit = structuredClone(evidence);
  successExit.result.exitCode = 0;
  assert.throws(() => validateNativeCapabilityEvidence(successExit), /compile rejection/);
});

test('atomic writes and exact removals stay beneath the parity output root', (t) => {
  const { repoRoot, outputRoot } = fixture(t);
  const destination = atomicWriteJsonSync('nested/result.json', { ok: true }, { outputRoot });
  assert.deepEqual(JSON.parse(readFileSync(destination, 'utf8')), { ok: true });
  assert.equal(readdirSync(path.dirname(destination)).some((name) => name.endsWith('.tmp')), false);
  assert.throws(() => atomicWriteJsonSync('../escaped.json', {}, { outputRoot }), /escapes or names/);
  assert.throws(() => atomicWriteJsonSync('*.json', {}, { outputRoot }), /exact, not a glob/);
  assert.throws(() => removeParityOutputsSync(['.'], { outputRoot }), /escapes or names/);

  const outside = path.join(repoRoot, 'outside');
  mkdirSync(outside);
  symlinkSync(outside, path.join(outputRoot, 'outside-link'));
  assert.throws(
    () => atomicWriteJsonSync('outside-link/result.json', {}, { outputRoot }),
    /symlink outside/,
  );
  assert.equal(existsSync(path.join(outside, 'result.json')), false);

  mkdirSync(path.join(outputRoot, 'directory'));
  assert.throws(
    () => removeParityOutputsSync(['nested/result.json', 'directory'], { outputRoot }),
    /without allowDirectories/,
  );
  assert.equal(existsSync(destination), true, 'removal preflights every target before mutating');
  assert.deepEqual(removeParityOutputsSync(['nested/result.json'], { outputRoot }), ['nested/result.json']);
  assert.equal(existsSync(destination), false);
  assert.deepEqual(
    removeParityOutputsSync(['directory'], { outputRoot, allowDirectories: true }),
    ['directory'],
  );
});

test('run manifests reject stale or modified artifacts', (t) => {
  const { options, outputRoot } = fixture(t);
  const fingerprint = createParityFingerprintSync(options);
  const manifest = createRunManifest({
    command: 'produce cpu-js',
    fingerprint,
    jobs: [
      { case: 'toy', tier: 'cpu-js', expectation: 'required' },
      { case: 'missing-op', tier: 'cpu-js', expectation: 'expected-skip' },
    ],
    producer: { kind: 'node', backend: 'cpu-js' },
    runId: 'run-current',
    startedAt: '2026-07-20T00:00:00.000Z',
  });
  const artifactFile = writeRunArtifactSync({
    manifest,
    file: 'results/toy.cpu-js.json',
    case: 'toy',
    tier: 'cpu-js',
    payload: { sigs: { y: { n: 1 } } },
    outputRoot,
    createdAt: '2026-07-20T00:00:01.000Z',
  });
  recordRunResult(manifest, {
    case: 'missing-op',
    tier: 'cpu-js',
    status: 'expected-skip',
    reason: 'declared unsupported operator',
  });
  finalizeRunManifest(manifest, { completedAt: '2026-07-20T00:00:02.000Z' });
  assert.equal(manifest.outcome, 'success');

  writeRunManifestSync('manifests/produce-cpu-js.json', manifest, { outputRoot });
  const loaded = readRunManifestSync('manifests/produce-cpu-js.json', {
    outputRoot,
    requireComplete: true,
    requireFinalized: true,
    expectedFingerprint: fingerprint,
    expectedRunId: 'run-current',
  });
  assert.deepEqual(
    readRunArtifactSync(artifactFile, loaded, { case: 'toy', tier: 'cpu-js', outputRoot }),
    { sigs: { y: { n: 1 } } },
  );

  writeFileSync(options.fixtureFiles[0], 'fixture-v2');
  const currentFingerprint = createParityFingerprintSync(options);
  assert.throws(
    () => readRunManifestSync('manifests/produce-cpu-js.json', {
      outputRoot,
      expectedFingerprint: currentFingerprint,
    }),
    /current parity campaign/,
  );
  const staleManifest = structuredClone(loaded);
  staleManifest.fingerprint = currentFingerprint;
  assert.throws(
    () => readRunArtifactSync(artifactFile, staleManifest, { case: 'toy', tier: 'cpu-js', outputRoot }),
    /current run fingerprint/,
  );

  writeFileSync(artifactFile, `${readFileSync(artifactFile, 'utf8')} `);
  assert.throws(
    () => readRunArtifactSync(artifactFile, loaded, { case: 'toy', tier: 'cpu-js', outputRoot }),
    /bytes do not match manifest/,
  );
});

test('runtime producer manifests require validated lifecycle evidence before writing', (t) => {
  const { options, outputRoot } = fixture(t);
  const manifest = createRunManifest({
    command: 'produce cpu-js',
    fingerprint: createParityFingerprintSync(options),
    jobs: [{ case: 'toy', tier: 'cpu-js' }],
    producer: {
      kind: 'node', backend: 'cpu-js', strictBackend: true, runtimeEvidenceRequired: true,
    },
  });
  const file = 'results/toy.runtime.json';
  assert.throws(() => writeRunArtifactSync({
    manifest,
    file,
    case: 'toy',
    tier: 'cpu-js',
    payload: { sigs: {} },
    outputRoot,
  }), /must include runtime evidence/);
  assert.equal(existsSync(path.join(outputRoot, file)), false);

  const runtimeEvidence = runtimeEvidenceFixture();
  writeRunArtifactSync({
    manifest,
    file,
    case: 'toy',
    tier: 'cpu-js',
    payload: { sigs: {}, runtimeEvidence },
    metadata: { runtimeEvidence },
    outputRoot,
  });
  finalizeRunManifest(manifest);
  assert.equal(manifest.results[0].metadata.runtimeEvidence.execution.contextId, 'context-1');
  assert.equal(manifest.outcome, 'success');
});

test('runtime evidence can be required for execution tiers but not fixture tiers', (t) => {
  const { options, outputRoot } = fixture(t);
  const manifest = createRunManifest({
    command: 'decode produce cpu-js',
    fingerprint: createParityFingerprintSync(options),
    jobs: [
      { case: 'toy', tier: 'prompt' },
      { case: 'toy', tier: 'cpu-js' },
    ],
    producer: { kind: 'node', runtimeEvidenceTiers: ['cpu-js'] },
  });
  writeRunArtifactSync({
    manifest,
    file: 'results/prompt.json',
    case: 'toy',
    tier: 'prompt',
    payload: { tokens: [1] },
    outputRoot,
  });
  assert.throws(() => writeRunArtifactSync({
    manifest,
    file: 'results/cpu-js.json',
    case: 'toy',
    tier: 'cpu-js',
    payload: { tokens: [2] },
    outputRoot,
  }), /must include runtime evidence/);
});

test('manifest completion fails required skips and producer errors', (t) => {
  const { options } = fixture(t);
  const fingerprint = createParityFingerprintSync(options);
  const incomplete = createRunManifest({
    command: 'produce wasm',
    fingerprint,
    jobs: [{ case: 'toy', tier: 'wasm' }],
  });
  assert.throws(() => finalizeRunManifest(incomplete), /incomplete/);
  recordRunResult(incomplete, {
    case: 'toy',
    tier: 'wasm',
    status: 'error',
    error: 'backend initialization failed',
  });
  assert.equal(finalizeRunManifest(incomplete).outcome, 'error');

  const requiredSkip = createRunManifest({
    command: 'produce native-cpu',
    fingerprint,
    jobs: [{ case: 'toy', tier: 'native-cpu' }],
  });
  recordRunResult(requiredSkip, {
    case: 'toy',
    tier: 'native-cpu',
    status: 'expected-skip',
    reason: 'binary absent',
  });
  assert.equal(finalizeRunManifest(requiredSkip).outcome, 'error');
});

test('native capability manifest jobs require structured evidence and cannot masquerade as success', (t) => {
  const { options } = fixture(t);
  const makeManifest = () => createRunManifest({
    command: 'native-sig native-vulkan',
    fingerprint: createParityFingerprintSync(options),
    jobs: [{ case: 'toy', tier: 'native-vulkan', expectation: 'expected-skip' }],
    producer: {
      kind: 'native-fold',
      capabilityEvidenceTiers: ['native-vulkan'],
    },
  });

  const missing = makeManifest();
  assert.throws(() => recordRunResult(missing, {
    case: 'toy', tier: 'native-vulkan', status: 'expected-skip', reason: 'unsupported',
  }), /must record an expected skip with capability evidence/);

  const unexpectedSuccess = makeManifest();
  assert.throws(() => recordRunResult(unexpectedSuccess, {
    case: 'toy',
    tier: 'native-vulkan',
    status: 'success',
    artifact: { path: 'toy.json', sha256: '8'.repeat(64), size: 1 },
  }), /must record an expected skip with capability evidence/);

  const accepted = makeManifest();
  recordRunResult(accepted, {
    case: 'toy',
    tier: 'native-vulkan',
    status: 'expected-skip',
    reason: 'BOUNDED_DOMAIN_UNSUPPORTED',
    metadata: { capabilityEvidence: nativeCapabilityEvidenceFixture() },
  });
  assert.equal(finalizeRunManifest(accepted).outcome, 'success');
});
