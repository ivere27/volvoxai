import test from 'node:test';
import assert from 'node:assert/strict';

import {
  formatAdapterIdentity,
  requirePhysicalAdapterIdentity,
  requirePhysicalNativeGpuLog,
  requirePhysicalWebGPU,
} from './parity/lib/backend.mjs';

test('physical WebGPU parity requires a retained non-software adapter identity', () => {
  const compiled = (backend, selectedDevice) => ({
    backend,
    report: { selectedDevice },
  });
  const identity = requirePhysicalWebGPU({
    backend: 'webgpu',
    report: {
      selectedDevice: { vendor: 'NVIDIA', device: 'GeForce RTX 3090', backend: 'Vulkan' },
    },
  });
  assert.deepEqual(identity, { vendor: 'NVIDIA', device: 'GeForce RTX 3090', backend: 'Vulkan' });
  assert.equal(formatAdapterIdentity(identity), 'vendor=NVIDIA, device=GeForce RTX 3090, backend=Vulkan');

  assert.throws(
    () => requirePhysicalWebGPU(compiled('cpu-js', { device: 'RTX' })),
    /compiled on 'cpu-js'/,
  );
  assert.throws(
    () => requirePhysicalWebGPU(compiled('webgpu', null)),
    /identity is unavailable/,
  );
  assert.throws(
    () => requirePhysicalWebGPU(compiled('webgpu', ['NVIDIA GeForce RTX 3090'])),
    /identity is unavailable/,
  );
  for (const device of ['Google SwiftShader', 'llvmpipe (LLVM 18)', 'Mesa lavapipe', 'CPU']) {
    assert.throws(
      () => requirePhysicalWebGPU(compiled('webgpu', { device })),
      /software adapter rejected/,
    );
  }
});

test('native physical-GPU parity binds route evidence to the initialized device', () => {
  const vulkan = [
    '[VolvoxAI GPU] Vulkan Compute initialized successfully! Device: NVIDIA GeForce RTX 3090; packed INT8 dot: enabled',
    'Backend: vulkan',
  ].join('\n');
  assert.deepEqual(
    requirePhysicalNativeGpuLog(vulkan, 'vulkan', 'native Vulkan', { requiredIdentity: 'rtx 3090' }),
    { backend: 'vulkan', device: 'NVIDIA GeForce RTX 3090', packedInt8Dot: 'enabled' },
  );

  const opengl = [
    '[VolvoxAI GPU] OpenGL Compute initialized: NVIDIA Corporation / NVIDIA GeForce RTX 3090/PCIe/SSE2 / 4.6.0 NVIDIA 570.86.16',
    'Backend: opengl',
  ].join('\n');
  assert.deepEqual(
    requirePhysicalNativeGpuLog(opengl, 'opengl', 'native OpenGL', { requiredIdentity: 'RTX 3090' }),
    {
      backend: 'opengl',
      vendor: 'NVIDIA Corporation',
      device: 'NVIDIA GeForce RTX 3090/PCIe/SSE2',
      version: '4.6.0 NVIDIA 570.86.16',
    },
  );

  assert.throws(
    () => requirePhysicalNativeGpuLog('Backend: vulkan\n', 'vulkan'),
    /exactly one Vulkan device initialization/,
  );
  assert.throws(
    () => requirePhysicalNativeGpuLog(vulkan, 'opengl'),
    /reported \["vulkan"\], expected opengl/,
  );
  assert.throws(
    () => requirePhysicalNativeGpuLog(`${vulkan}\n${vulkan}`, 'vulkan'),
    /reported \["vulkan","vulkan"\], expected vulkan/,
  );
  const [vulkanInit] = vulkan.split('\n');
  assert.throws(
    () => requirePhysicalNativeGpuLog(`${vulkanInit}\n${vulkanInit}\nBackend: vulkan`, 'vulkan'),
    /exactly one Vulkan device initialization, found 2/,
  );
  assert.throws(
    () => requirePhysicalNativeGpuLog(
      '[VolvoxAI GPU] Vulkan Compute initialized successfully! Device: llvmpipe (LLVM 18); packed INT8 dot: unavailable\nBackend: vulkan\n',
      'vulkan',
    ),
    /software adapter rejected/,
  );
  assert.throws(
    () => requirePhysicalNativeGpuLog(vulkan, 'vulkan', 'native Vulkan', { requiredIdentity: 'RTX 4090' }),
    /does not match required identity 'RTX 4090'/,
  );
});

test('physical WebGPU parity can require a named hardware adapter', () => {
  const info = { vendor: 'NVIDIA', device: 'NVIDIA GeForce RTX 3090', backend: 'Vulkan' };
  assert.deepEqual(
    requirePhysicalAdapterIdentity(info, 'required GPU', { requiredIdentity: 'rtx 3090' }),
    info,
  );
  assert.throws(
    () => requirePhysicalAdapterIdentity(info, 'required GPU', { requiredIdentity: 'RTX 4090' }),
    /does not match required identity 'RTX 4090'/,
  );
});
