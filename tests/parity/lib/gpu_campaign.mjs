import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  canonicalJson,
  createParityFingerprintSync,
  sha256FileSync,
  validateNativeCapabilityEvidence,
} from './artifact.mjs';
import {
  requirePhysicalAdapterIdentity,
  requirePhysicalNativeAdapterIdentity,
} from './backend.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PARITY_ROOT = path.resolve(HERE, '..');
const ROOT = path.resolve(PARITY_ROOT, '..', '..');
const POLICY_FILE = path.join(PARITY_ROOT, 'policy.json');
const KERNEL_REGISTRY_FILE = path.join(ROOT, 'proto', 'kernel_registry.proto');
const GENERATED_KERNEL_REGISTRY_FILE = path.join(ROOT, 'ts', 'generated', 'kernelRegistry.ts');
const SHA256_RE = /^[0-9a-f]{64}$/;

export const PHYSICAL_GPU_TIERS = Object.freeze([
  'webgpu', 'native-vulkan', 'native-opengl',
]);
export const REQUIRED_EXECUTION_GPU_TIERS = Object.freeze(['webgpu']);
export const NATIVE_GPU_CAPABILITY_TIERS = Object.freeze([
  'native-vulkan', 'native-opengl',
]);
export const GPU_CONSENSUS_TIERS = PHYSICAL_GPU_TIERS;
export const GPU_CONSENSUS_MODELS = Object.freeze([
  Object.freeze({ model: 'efficientdet_lite0_fp32', tolerance: 1e-5 }),
  Object.freeze({ model: 'efficientdet_lite0_int8', tolerance: 0 }),
]);

function exactObject(value, label, keys) {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (canonicalJson(actual) !== canonicalJson(expected)) {
    throw new Error(`${label} must contain exactly: ${expected.join(', ')}`);
  }
  return value;
}

function exactArray(value, expected, label) {
  if (!Array.isArray(value) || canonicalJson(value) !== canonicalJson(expected)) {
    throw new Error(`${label} must be ${canonicalJson(expected)}`);
  }
}

function policyFromDisk() {
  return JSON.parse(fs.readFileSync(POLICY_FILE, 'utf8'));
}

function distFile(name) {
  const version = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
  return path.join(ROOT, 'dist', version, name);
}

function fixtureFile(modelName, input) {
  const ext = input.dtype === 'f32' ? 'f32' : input.dtype;
  return path.join(PARITY_ROOT, 'out', 'fixtures', modelName, `${input.name}.${ext}`);
}

function generatedFixtureFiles(policy, names) {
  return names.flatMap((name) => (policy.models[name]?.inputs || [])
    .filter((input) => input.gen)
    .map((input) => fixtureFile(name, input)));
}

function wholeBuildFiles(tier) {
  if (tier === 'cpu-js') return [distFile('volvoxai.js')];
  if (tier === 'wasm') return [distFile('volvoxai.js'), distFile('volvoxai.wasm')];
  if (tier === 'webgpu') return [distFile('volvoxai.js')];
  if (tier.startsWith('native-')) {
    const binary = path.join(ROOT, 'native', 'volvoxai');
    return fs.existsSync(binary) ? [binary] : [];
  }
  throw new Error(`unknown whole-model build tier: ${tier}`);
}

export function validateGpuCampaignPolicy(policy) {
  const campaign = exactObject(policy?.gpuCampaign, 'gpuCampaign policy', [
    'physicalTiers', 'requiredExecutionTiers', 'nativeCapabilityTiers', 'consensusTiers',
  ]);
  exactArray(campaign.physicalTiers, PHYSICAL_GPU_TIERS, 'gpuCampaign.physicalTiers');
  exactArray(
    campaign.requiredExecutionTiers,
    REQUIRED_EXECUTION_GPU_TIERS,
    'gpuCampaign.requiredExecutionTiers',
  );
  exactArray(
    campaign.nativeCapabilityTiers,
    NATIVE_GPU_CAPABILITY_TIERS,
    'gpuCampaign.nativeCapabilityTiers',
  );
  exactArray(campaign.consensusTiers, GPU_CONSENSUS_TIERS, 'gpuCampaign.consensusTiers');
  if (policy.models == null || typeof policy.models !== 'object' || Array.isArray(policy.models)) {
    throw new Error('GPU campaign policy must contain models');
  }
  for (const [modelName, model] of Object.entries(policy.models)) {
    const expectations = exactObject(model.gpuCampaign, `${modelName}.gpuCampaign`, PHYSICAL_GPU_TIERS);
    for (const tier of PHYSICAL_GPU_TIERS) {
      if (!model.tiers?.includes(tier)) throw new Error(`${modelName} does not declare GPU tier ${tier}`);
      const entry = expectations[tier];
      if (tier === 'webgpu') {
        exactObject(entry, `${modelName}.gpuCampaign.${tier}`, ['expectation']);
        if (entry.expectation !== 'required') {
          throw new Error(`${modelName}.gpuCampaign.webgpu must be required`);
        }
        continue;
      }
      exactObject(entry, `${modelName}.gpuCampaign.${tier}`, [
        'expectation', 'status', 'stage', 'reason', 'authority',
      ]);
      if (entry.expectation !== 'expected-skip' || entry.status !== 'BACKEND_UNSUPPORTED' ||
          entry.stage !== 'compile' || typeof entry.reason !== 'string' || !entry.reason) {
        throw new Error(`${modelName}.gpuCampaign.${tier} is not an exact compile capability skip`);
      }
      const backend = nativeBackendForTier(tier);
      exactObject(entry.authority, `${modelName}.gpuCampaign.${tier}.authority`, [
        'kind', 'target', 'exporterQualified',
      ]);
      if (entry.authority.kind !== 'kernel-registry' ||
          entry.authority.target !== `backend:${backend}` ||
          entry.authority.exporterQualified !== false) {
        throw new Error(`${modelName}.gpuCampaign.${tier} has invalid registry authority`);
      }
    }
  }
  return policy;
}

export function gpuJobExpectation(policy, modelName, tier) {
  validateGpuCampaignPolicy(policy);
  const model = policy.models[modelName];
  if (!model) throw new Error(`unknown parity model: ${modelName}`);
  const expectation = model.gpuCampaign[tier];
  if (!expectation) throw new Error(`${modelName} has no GPU campaign expectation for ${tier}`);
  return expectation;
}

export function nativeBackendForTier(tier) {
  const backend = { 'native-vulkan': 'vulkan', 'native-opengl': 'opengl' }[tier];
  if (!backend) throw new Error(`unsupported native GPU capability tier: ${tier}`);
  return backend;
}

export function physicalAdapterFromCapabilityLog(
  output,
  backend,
  label = `native ${backend} capability probe`,
) {
  const text = String(output);
  if (/^Backend: /m.test(text)) {
    throw new Error(`${label}: rejected execution/success route line in a compile-rejection log`);
  }
  let matches;
  let identity;
  if (backend === 'vulkan') {
    matches = [...text.matchAll(
      /^\[VolvoxAI GPU\] Vulkan Compute initialized successfully! Device: ([^;\r\n]+); packed INT8 dot: (enabled|unavailable)\r?$/gm,
    )];
    if (matches.length === 1) {
      identity = { backend, device: matches[0][1], packedInt8Dot: matches[0][2] };
    }
  } else {
    matches = [...text.matchAll(
      /^\[VolvoxAI GPU\] OpenGL Compute initialized: (.*?) \/ (.*?) \/ ([^\r\n]+)\r?$/gm,
    )];
    if (matches.length === 1) {
      identity = {
        backend,
        vendor: matches[0][1],
        device: matches[0][2],
        version: matches[0][3],
      };
    }
  }
  if (matches?.length !== 1) {
    throw new Error(`${label}: expected exactly one physical ${backend} initialization, found ${matches?.length ?? 0}`);
  }
  return requirePhysicalNativeAdapterIdentity(identity, backend, label);
}

/** Current whole-model producer fingerprint shared by run.mjs and the native probe. */
export function wholeModelCampaignFingerprint(tier, names) {
  const policy = validateGpuCampaignPolicy(policyFromDisk());
  const selected = names == null
    ? Object.entries(policy.models)
      .filter(([, model]) => model.tiers.includes(tier))
      .map(([name]) => name)
    : [...new Set(names)];
  if (selected.length === 0) throw new Error(`no policy models selected for tier ${tier}`);
  for (const name of selected) {
    if (!policy.models[name]?.tiers?.includes(tier)) {
      throw new Error(`parity model ${name} does not declare tier ${tier}`);
    }
  }
  return createParityFingerprintSync({
    repoRoot: ROOT,
    policyFile: POLICY_FILE,
    selectedCases: selected,
    fixtureFiles: generatedFixtureFiles(policy, selected),
    buildFiles: wholeBuildFiles(tier),
  });
}

function sourceFingerprint(fingerprint) {
  const value = fingerprint?.components?.source?.statusSha256;
  if (!SHA256_RE.test(value ?? '')) {
    throw new Error('native GPU capability evidence requires an exact Git source fingerprint');
  }
  return value;
}

function currentRegistryAuthority(target) {
  const registryFingerprint = sha256FileSync(KERNEL_REGISTRY_FILE).sha256;
  const generated = fs.readFileSync(GENERATED_KERNEL_REGISTRY_FILE, 'utf8');
  if (!generated.includes(`// proto/kernel_registry.proto SHA-256: ${registryFingerprint}`)) {
    throw new Error('generated kernel registry is stale for native GPU capability authority');
  }
  const declaration = generated.split(/\r?\n/).find((line) =>
    line.startsWith('export const exporterQualifiedOperators = '));
  if (!declaration) throw new Error('generated kernel registry has no exporter qualification projection');
  const escaped = target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(`"${escaped}":Object\\.freeze\\(\\[([^\\]]*)\\]\\)`).exec(declaration);
  if (!match) throw new Error(`generated kernel registry has no target ${target}`);
  if (match[1].trim() !== '') {
    throw new Error(`${target} now has exporter qualification; its expected capability skip is invalid`);
  }
  return {
    authority: { kind: 'kernel-registry', target, exporterQualified: false },
    registryFingerprint,
  };
}

function modelIdentityFromFingerprint(modelName, fingerprint, policy) {
  const model = policy.models[modelName];
  const graphPath = `${model.dir.replace(/\/$/, '')}/graph.json`;
  const graph = fingerprint.components.models.find((entry) => entry.path === graphPath);
  if (!graph) throw new Error(`whole-model fingerprint lacks ${graphPath}`);
  const prefix = `${model.dir.replace(/\/$/, '')}/`;
  const weights = fingerprint.components.models
    .filter((entry) => entry.path.startsWith(prefix) && entry.path.endsWith('.safetensors'))
    .map((entry) => ({ path: entry.path, sha256: entry.sha256 }));
  if (weights.length === 0) throw new Error(`whole-model fingerprint lacks weights for ${modelName}`);
  return { id: modelName, graphPath, graphSha256: graph.sha256, weights };
}

/** Exact policy and current-file identity a native capability probe must seal. */
export function nativeCapabilityEvidenceExpectation(modelName, tier, names) {
  const policy = validateGpuCampaignPolicy(policyFromDisk());
  const expectation = gpuJobExpectation(policy, modelName, tier);
  if (expectation.expectation !== 'expected-skip') {
    throw new Error(`${modelName}/${tier} is not a native capability-skip job`);
  }
  const backend = nativeBackendForTier(tier);
  const selected = names ?? Object.entries(policy.models)
    .filter(([, model]) => model.tiers.includes(tier))
    .map(([name]) => name);
  if (!selected.includes(modelName)) throw new Error(`${modelName} is outside the ${tier} campaign`);
  const fingerprint = wholeModelCampaignFingerprint(tier, selected);
  const registry = currentRegistryAuthority(expectation.authority.target);
  if (canonicalJson(registry.authority) !== canonicalJson(expectation.authority)) {
    throw new Error(`${modelName}/${tier} policy disagrees with current kernel registry authority`);
  }
  return {
    campaign: {
      id: fingerprint.digest,
      sourceFingerprint: sourceFingerprint(fingerprint),
      registryFingerprint: registry.registryFingerprint,
    },
    authority: registry.authority,
    model: modelIdentityFromFingerprint(modelName, fingerprint, policy),
    request: {
      tier,
      backend,
      policy: { mode: 'require', operatorFallback: 'forbid' },
    },
    result: {
      outcome: 'expected-compile-rejection',
      status: expectation.status,
      stage: expectation.stage,
      reason: expectation.reason,
    },
  };
}

/**
 * Validate a producer rejection against one already resolved expectation.
 * This is deliberately independent of repository/model discovery so wire and
 * contextual validation can be tested with hermetic synthetic fixtures.
 */
export function requireNativeCapabilityEvidenceAgainstExpectation(evidence, expected, {
  nativeLog,
  runtimeFailureEvidenceFile,
} = {}) {
  validateNativeCapabilityEvidence(evidence);
  if (expected == null || typeof expected !== 'object' ||
      typeof expected.model?.id !== 'string' || !expected.model.id ||
      typeof expected.request?.tier !== 'string' || !expected.request.tier ||
      !Array.isArray(expected.model?.weights)) {
    throw new Error('native capability expectation is invalid');
  }
  const modelName = expected.model.id;
  const tier = expected.request.tier;
  for (const field of ['campaign', 'authority', 'model', 'request']) {
    if (canonicalJson(evidence[field]) !== canonicalJson(expected[field])) {
      throw new Error(`${modelName}/${tier} capability evidence has stale or mismatched ${field}`);
    }
  }
  for (const field of ['outcome', 'status', 'stage', 'reason']) {
    if (evidence.result[field] !== expected.result[field]) {
      throw new Error(`${modelName}/${tier} capability evidence has unexpected result.${field}`);
    }
  }
  if (evidence.result.exitCode !== 1) {
    throw new Error(`${modelName}/${tier} capability evidence has unexpected process exit code`);
  }
  const failure = evidence.runtimeFailureEvidence;
  if (canonicalJson(failure.request) !== canonicalJson(expected.request.policy == null
    ? null
    : { backend: expected.request.backend, policy: expected.request.policy })) {
    throw new Error(`${modelName}/${tier} runtime failure evidence has the wrong strict request`);
  }
  const expectedSource = {
    graphPath: expected.model.graphPath,
    weightPaths: expected.model.weights.map((weight) => weight.path),
  };
  if (canonicalJson(failure.source) !== canonicalJson(expectedSource)) {
    throw new Error(`${modelName}/${tier} runtime failure evidence has the wrong model source`);
  }
  const requiredLineage = [
    'runtimeId', 'modelId', 'compilationId', 'definitionId', 'weightRevisionId',
    'topologyRevision', 'weightRevision',
  ];
  if (requiredLineage.some((field) => failure.lineage[field] == null)) {
    throw new Error(`${modelName}/${tier} runtime failure evidence lacks real compile lineage`);
  }
  if (failure.status !== expected.result.status || failure.stage !== expected.result.stage ||
      failure.report.backend !== expected.request.backend ||
      failure.report.reason !== expected.result.reason ||
      failure.report.message !== evidence.result.message ||
      failure.report.offendingNode !== evidence.result.offendingNode ||
      failure.report.tierFallback || failure.report.operatorFallbackUsed) {
    throw new Error(`${modelName}/${tier} runtime failure evidence contradicts the capability rejection`);
  }
  if (failure.report.device?.backend !== expected.request.backend ||
      typeof failure.report.device?.device !== 'string' || !failure.report.device.device) {
    throw new Error(`${modelName}/${tier} runtime failure evidence lacks the requested backend device`);
  }
  if (runtimeFailureEvidenceFile == null) {
    throw new Error(`${modelName}/${tier} raw runtime failure evidence is missing`);
  }
  const failureDigest = sha256FileSync(runtimeFailureEvidenceFile).sha256;
  if (failureDigest !== evidence.runtimeFailureEvidenceSha256) {
    throw new Error(`${modelName}/${tier} raw runtime failure evidence digest does not match`);
  }
  const rawFailure = JSON.parse(fs.readFileSync(runtimeFailureEvidenceFile, 'utf8'));
  if (canonicalJson(rawFailure) !== canonicalJson(failure)) {
    throw new Error(`${modelName}/${tier} nested and raw runtime failure evidence differ`);
  }
  const adapter = requirePhysicalNativeAdapterIdentity(
    evidence.adapter,
    expected.request.backend,
    `${modelName}/${tier} capability probe`,
  );
  if (nativeLog == null) throw new Error(`${modelName}/${tier} capability probe log is missing`);
  const logBytes = fs.readFileSync(nativeLog);
  if (sha256FileSync(nativeLog).sha256 !== evidence.nativeLogSha256) {
    throw new Error(`${modelName}/${tier} capability probe log digest does not match evidence`);
  }
  const logText = logBytes.toString('utf8');
  const logAdapter = physicalAdapterFromCapabilityLog(
    logText,
    expected.request.backend,
    `${modelName}/${tier} capability probe log`,
  );
  if (canonicalJson(adapter) !== canonicalJson(logAdapter)) {
    throw new Error(`${modelName}/${tier} capability adapter disagrees with its raw log`);
  }
  const failures = [...logText.matchAll(
    /^Native init failed: ([A-Z_]+) \[([A-Z0-9_]+)\]: ([^\r\n]+)\r?$/gm,
  )];
  if (failures.length !== 1 || failures[0][1] !== evidence.result.status ||
      failures[0][2] !== evidence.result.reason || failures[0][3] !== evidence.result.message) {
    throw new Error(`${modelName}/${tier} capability rejection disagrees with its raw log`);
  }
  return Object.freeze({ ...evidence, adapter });
}

/** Validate a producer rejection against current policy, registry, files, and raw reports. */
export function requireExpectedNativeCapabilityEvidence(evidence, {
  modelName,
  tier,
  names,
  nativeLog,
  runtimeFailureEvidenceFile,
} = {}) {
  const expected = nativeCapabilityEvidenceExpectation(modelName, tier, names);
  return requireNativeCapabilityEvidenceAgainstExpectation(evidence, expected, {
    nativeLog,
    runtimeFailureEvidenceFile,
  });
}

const explicitSource = Object.freeze({
  revision: null,
  state: 'explicit-files',
  statusSha256: null,
});

export function gpuConsensusFingerprint() {
  return createParityFingerprintSync({
    repoRoot: ROOT,
    policyFile: POLICY_FILE,
    selectedCases: GPU_CONSENSUS_MODELS.map((entry) => entry.model),
    buildFiles: [
      distFile('volvoxai.js'),
      distFile('volvoxai.wasm'),
      path.join(ROOT, 'native', 'volvoxai'),
    ],
    sourceFiles: [
      path.join(ROOT, 'package.json'),
      path.join(PARITY_ROOT, 'gpu_consensus_check.sh'),
      path.join(PARITY_ROOT, 'webgpu_efficientdet_deno.js'),
      path.join(HERE, 'artifact.mjs'),
      path.join(HERE, 'backend.mjs'),
      fileURLToPath(import.meta.url),
      path.join(HERE, 'tensorio.mjs'),
    ],
    source: explicitSource,
  });
}

export function kvCacheCampaignFingerprint() {
  return createParityFingerprintSync({
    repoRoot: ROOT,
    policyFile: POLICY_FILE,
    selectedCases: [],
    buildFiles: [distFile('volvoxai.js'), distFile('volvoxai.wasm')],
    sourceFiles: [
      path.join(ROOT, 'package.json'),
      path.join(PARITY_ROOT, 'kvcache', 'kvcache_parity.mjs'),
      path.join(HERE, 'artifact.mjs'),
      path.join(HERE, 'backend.mjs'),
      fileURLToPath(import.meta.url),
    ],
    source: explicitSource,
  });
}

export function requireExactConsensusModels(models) {
  if (!Array.isArray(models)) throw new Error('GPU consensus summary models must be an array');
  const expected = new Map(GPU_CONSENSUS_MODELS.map((entry) => [entry.model, entry.tolerance]));
  const remaining = new Map(expected);
  for (const entry of models) {
    if (entry == null || typeof entry !== 'object' || typeof entry.model !== 'string') {
      throw new Error('GPU consensus summary has an invalid model entry');
    }
    if (!expected.has(entry.model)) {
      throw new Error(`GPU consensus summary has unexpected model: ${entry.model}`);
    }
    if (!remaining.has(entry.model)) {
      throw new Error(`GPU consensus summary has duplicate model: ${entry.model}`);
    }
    if (entry.tolerance !== expected.get(entry.model)) {
      throw new Error(`GPU consensus summary has unexpected tolerance for ${entry.model}`);
    }
    remaining.delete(entry.model);
  }
  if (remaining.size !== 0) {
    throw new Error(`GPU consensus summary is missing model(s): ${[...remaining.keys()].join(', ')}`);
  }
  return models;
}

export function requireFiniteFloat32Buffers(buffers, labels, context) {
  if (!Array.isArray(buffers) || !Array.isArray(labels) || buffers.length !== labels.length) {
    throw new Error(`${context}: float32 buffer labels do not match`);
  }
  return buffers.map((bytes, index) => {
    if (!(bytes instanceof Uint8Array) || bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`${context}: ${labels[index]} is not a complete float32 buffer`);
    }
    const values = new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
    const bad = values.findIndex((value) => !Number.isFinite(value));
    if (bad !== -1) throw new Error(`${context}: ${labels[index]} is non-finite at ${bad}`);
    return values;
  });
}

export function requireExactPhysicalAdapterSet(adapters, label, options) {
  if (adapters == null || typeof adapters !== 'object' || Array.isArray(adapters)) {
    throw new Error(`${label}: adapter evidence must be an object`);
  }
  const actualKeys = Object.keys(adapters).sort();
  const expectedKeys = [...GPU_CONSENSUS_TIERS].sort();
  if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
    throw new Error(`${label}: adapter evidence is ${JSON.stringify(actualKeys)}, expected ${JSON.stringify(expectedKeys)}`);
  }
  return {
    webgpu: requirePhysicalAdapterIdentity(adapters.webgpu, `${label}/webgpu`, options),
    'native-vulkan': requirePhysicalNativeAdapterIdentity(
      adapters['native-vulkan'], 'vulkan', `${label}/native-vulkan`, options,
    ),
    'native-opengl': requirePhysicalNativeAdapterIdentity(
      adapters['native-opengl'], 'opengl', `${label}/native-opengl`, options,
    ),
  };
}
