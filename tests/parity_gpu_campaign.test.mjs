import test from 'node:test';
import assert from 'node:assert/strict';

import {
  requireExactConsensusModels,
  requireExactPhysicalAdapterSet,
  requireFiniteFloat32Buffers,
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
    requireFiniteFloat32Buffers([finite, finite, finite], ['webgpu', 'native-opengl', 'native-vulkan'], 'consensus').length,
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
