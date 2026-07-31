import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createParityFingerprintSync } from './artifact.mjs';
import {
  requirePhysicalAdapterIdentity,
  requirePhysicalNativeAdapterIdentity,
} from './backend.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PARITY_ROOT = path.resolve(HERE, '..');
const ROOT = path.resolve(PARITY_ROOT, '..', '..');

export const REQUIRED_GPU_TIERS = Object.freeze(['webgpu', 'native-vulkan', 'native-opengl']);
export const GPU_CONSENSUS_MODELS = Object.freeze([
  Object.freeze({ model: 'efficientdet_lite0_fp32', tolerance: 1e-5 }),
  Object.freeze({ model: 'efficientdet_lite0_int8', tolerance: 0 }),
]);

function distFile(name) {
  const version = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
  return path.join(ROOT, 'dist', version, name);
}

const explicitSource = Object.freeze({
  revision: null,
  state: 'explicit-files',
  statusSha256: null,
});

export function gpuConsensusFingerprint() {
  return createParityFingerprintSync({
    repoRoot: ROOT,
    policyFile: path.join(PARITY_ROOT, 'policy.json'),
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
    policyFile: path.join(PARITY_ROOT, 'policy.json'),
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
  const expectedKeys = [...REQUIRED_GPU_TIERS].sort();
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
