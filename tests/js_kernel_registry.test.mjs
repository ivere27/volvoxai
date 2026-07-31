import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CPU_JS_SUPPORTED_OPERATORS,
  WASM_KERNEL_ROUTES,
  exporterQualifiedOperators,
  exporterTargetSupports,
  kernelBackends,
  kernelRoute,
  kernelRoutesByBackend,
  kernelVariants,
  runtimeOperatorsByBackend,
  runtimeSupportModeByBackend,
  runtimeSupportsOperator,
  targetProfiles,
} from '../ts/generated/kernelRegistry.js';
import { fullKernelVariants } from '../ts/generated/kernelRegistryFull.js';

function assertFrozenTree(value, label) {
  if (value === null || typeof value !== 'object') return;
  assert.equal(Object.isFrozen(value), true, label);
  for (const [key, child] of Object.entries(value)) {
    assertFrozenTree(child, `${label}.${key}`);
  }
}

test('generated kernel inventory exposes a route for every registered operator', () => {
  assert.deepEqual(kernelBackends, [
    'cpu-js', 'wasm', 'webgpu', 'native-cpu', 'webnn',
    'vulkan', 'opengl', 'metal', 'cuda', 'nnapi',
  ]);

  for (const backend of kernelBackends) {
    const operators = runtimeOperatorsByBackend[backend];
    const routes = kernelRoutesByBackend[backend];
    assert.deepEqual(new Set(Object.keys(routes)), new Set(operators), backend);
    for (const operator of operators) {
      assert.equal(runtimeSupportsOperator(backend, operator), true, `${backend}:${operator}`);
      assert.equal(kernelRoute(backend, operator), routes[operator], `${backend}:${operator}`);
      assert.notEqual(routes[operator], '', `${backend}:${operator}`);
    }
    assert.equal(runtimeSupportsOperator(backend, 'DefinitelyUnknown'), false, backend);
    assert.equal(kernelRoute(backend, 'DefinitelyUnknown'), null, backend);
  }

  assert.deepEqual(
    new Set(CPU_JS_SUPPORTED_OPERATORS),
    new Set(runtimeOperatorsByBackend['cpu-js']),
  );
  assert.strictEqual(WASM_KERNEL_ROUTES, kernelRoutesByBackend.wasm);
  assert.equal(kernelRoute('wasm', 'QLinear'), 'qlinear');
  assert.equal(runtimeSupportModeByBackend.webgpu, 'direct');
  assert.equal(runtimeSupportModeByBackend.cuda, 'dynamic');
});

test('generated exporter profiles remain distinct from runtime registration', () => {
  assert.deepEqual(targetProfiles, {
    portable: ['cpu-js', 'wasm', 'webgpu', 'native-cpu'],
    browser: ['cpu-js', 'wasm', 'webgpu'],
  });
  for (const members of Object.values(targetProfiles)) {
    for (const target of members) {
      assert.ok(exporterQualifiedOperators[target], target);
    }
  }

  assert.equal(exporterTargetSupports('backend:webnn', 'Conv2D'), true);
  assert.equal(exporterTargetSupports('backend:webnn', 'QLinear'), false);
  assert.equal(exporterTargetSupports('unknown-target', 'Conv2D'), false);

  for (const target of [
    'backend:vulkan', 'backend:opengl', 'backend:metal', 'backend:cuda',
  ]) {
    assert.deepEqual(exporterQualifiedOperators[target], [], target);
    assert.equal(exporterTargetSupports(target, 'QLinear'), false, target);
  }
});

test('generated registry views are deeply immutable', () => {
  assertFrozenTree(kernelBackends, 'kernelBackends');
  assertFrozenTree(runtimeOperatorsByBackend, 'runtimeOperatorsByBackend');
  assertFrozenTree(kernelRoutesByBackend, 'kernelRoutesByBackend');
  assertFrozenTree(runtimeSupportModeByBackend, 'runtimeSupportModeByBackend');
  assertFrozenTree(exporterQualifiedOperators, 'exporterQualifiedOperators');
  assertFrozenTree(targetProfiles, 'targetProfiles');
  assertFrozenTree(kernelVariants, 'kernelVariants');
  assertFrozenTree(fullKernelVariants, 'fullKernelVariants');
});

test('inference kernel variants are valid registry registrations and exclude full-only work', () => {
  assert.ok(kernelVariants.length > 0);
  assert.equal(new Set(kernelVariants.map((variant) => variant.id)).size, kernelVariants.length);

  for (const variant of kernelVariants) {
    assert.ok(kernelBackends.includes(variant.backend), variant.id);
    assert.equal(variant.phase, 'forward', variant.id);
    assert.ok(variant.entrypointId.length > 0, variant.id);
    assert.ok(variant.predicateId.length > 0, variant.id);
    assert.ok(variant.priority > 0, variant.id);
    assert.ok(variant.operators.length > 0, variant.id);
    for (const operator of variant.operators) {
      assert.equal(runtimeSupportsOperator(variant.backend, operator), true,
        `${variant.id}:${operator}`);
    }
    assert.doesNotMatch(
      `${variant.id} ${variant.entrypointId} ${variant.predicateId}`,
      /training|backward|optimizer|ptq/i,
      variant.id,
    );
  }

  const packedWasm = kernelVariants.find(
    (variant) => variant.id === 'wasm.qlinear.packed-simd128',
  );
  assert.deepEqual(packedWasm?.requiredFeatures, ['wasm.simd128']);
  assert.ok(packedWasm?.operators.includes('QConv2D'));

  const cpuReference = kernelVariants.find(
    (variant) => variant.id === 'cpu-js.qlinear.reference',
  );
  assert.equal(cpuReference?.entrypointId, 'ts/ops/qLinear.ts::_cpuQLinear');
  assert.deepEqual(
    new Set(cpuReference?.operators),
    new Set(['QLinear', 'QMatMul', 'QGemm']),
  );
  assert.deepEqual(fullKernelVariants, []);
});
