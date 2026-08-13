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
  operatorShapeContract,
  operatorShapeContracts,
  operatorShapeFunctionIds,
  runtimeOperatorsByBackend,
  runtimeSupportModeByBackend,
  runtimeSupportsOperator,
  shapeContractClassifications,
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
    'cpu-js', 'wasm', 'webgpu', 'native-cpu',
    'vulkan', 'opengl', 'metal', 'cuda',
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

  assert.equal(exporterTargetSupports('unknown-target', 'Conv2D'), false);

  const nativeGpuOperators = new Set([
    'Add', 'ArgMax', 'BatchMatMul', 'Cast', 'Clip', 'Concat', 'Conv2D',
    'DequantizeLinear', 'Div', 'Embedding', 'Equal', 'Expand', 'GELU', 'Gather',
    'GreaterOrEqual', 'GroupNorm', 'LayerNorm', 'Linear', 'Mul', 'Not',
    'QAdd', 'QArgMax', 'QBatchMatMul', 'QConv2D', 'QEmbedding', 'QGELU',
    'QGemm', 'QGroupNorm', 'QLayerNorm', 'QLinear', 'QMaskedMean', 'QMatMul',
    'QSDPA', 'QSiLU', 'QuantizeLinear', 'ReduceSum', 'RequantizeLinear',
    'Reshape', 'Sigmoid', 'SiLU', 'Slice', 'Softmax', 'Squeeze', 'Sub',
    'Transpose', 'Unsqueeze', 'Where',
  ]);
  // Metal has no F32 MaxPool2D or ResizeNearest2D kernel, so it alone stays
  // unqualified for both.
  const spatialBackends = new Set(nativeGpuOperators)
    .add('MaxPool2D').add('ResizeNearest2D');
  for (const target of [
    'backend:vulkan', 'backend:opengl', 'backend:metal', 'backend:cuda',
  ]) {
    const expected = target === 'backend:metal' ? nativeGpuOperators : spatialBackends;
    assert.deepEqual(new Set(exporterQualifiedOperators[target]), expected, target);
    assert.equal(exporterTargetSupports(target, 'QLinear'), true, target);
    assert.equal(exporterTargetSupports(target, 'QSDPA'), true, target);
  }
  for (const operator of ['MaxPool2D', 'ResizeNearest2D']) {
    assert.equal(exporterTargetSupports('backend:vulkan', operator), true, operator);
    assert.equal(exporterTargetSupports('backend:metal', operator), false, operator);
  }
});

test('generated shape-contract routes cover the runtime vocabulary exactly once', () => {
  const runtimeOperators = runtimeOperatorsByBackend['cpu-js'];
  assert.deepEqual(new Set(Object.keys(operatorShapeContracts)), new Set(runtimeOperators));
  assert.deepEqual(new Set(Object.keys(operatorShapeFunctionIds)), new Set(runtimeOperators));
  assert.deepEqual(shapeContractClassifications, [
    'canonical', 'deferred', 'bounded-value-dependent',
  ]);

  for (const operator of runtimeOperators) {
    const contract = operatorShapeContract(operator);
    assert.ok(contract, operator);
    assert.equal(contract.shapeFunctionId, operatorShapeFunctionIds[operator], operator);
    assert.match(contract.shapeFunctionId,
      /^volvox\.shape\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*\.v[1-9][0-9]*$/u);
    assert.ok(shapeContractClassifications.includes(contract.classification), operator);
  }
  assert.equal(operatorShapeContract('DefinitelyUnknown'), null);
  assert.equal(operatorShapeContracts.Identity.shapeFunctionId, 'volvox.shape.identity.v1');
  assert.equal(operatorShapeContracts.Linear.shapeFunctionId,
    'volvox.shape.dense-last-axis.v1');
  assert.equal(operatorShapeContracts.NonMaxSuppression.classification,
    'bounded-value-dependent');
});

test('generated registry views are deeply immutable', () => {
  assertFrozenTree(kernelBackends, 'kernelBackends');
  assertFrozenTree(shapeContractClassifications, 'shapeContractClassifications');
  assertFrozenTree(operatorShapeContracts, 'operatorShapeContracts');
  assertFrozenTree(operatorShapeFunctionIds, 'operatorShapeFunctionIds');
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
  const cudaQBatch = kernelVariants.find(
    (variant) => variant.id === 'cuda.qbatch-matmul.dp4a',
  );
  assert.equal(cudaQBatch?.entrypointId, 'vx_cuda_qbatch_matmul_i8u8');
  assert.deepEqual(cudaQBatch?.operators, ['QBatchMatMul']);
  assert.deepEqual(cudaQBatch?.requiredFeatures,
    ['cuda.device-residency', 'cuda.sm61-dp4a']);
  assert.match(cudaQBatch?.predicateId ?? '', /arbitrary-k-tail/u);
  for (const [id, operators, entrypoint, predicate] of [
    [
      'cuda.qlinear.warp-dp4a',
      ['QLinear', 'QGemm', 'QMatMul'],
      'vx_cuda_qlinear_warp_dp4a_i8u8',
      /k-ge32-warp-launch-capacity.*arbitrary-k-tail/u,
    ],
    [
      'cuda.qlinear.thread-dp4a',
      ['QLinear', 'QGemm', 'QMatMul'],
      'vx_cuda_qlinear_i8u8',
      /small-k-or-warp-launch-capacity.*arbitrary-k-tail/u,
    ],
    [
      'cuda.qconv2d.warp-dp4a',
      ['QConv2D'],
      'vx_cuda_qconv2d_warp_dp4a_i8u8',
      /input-per-group-ge4-terms-ge64.*per-spatial-tail/u,
    ],
    [
      'cuda.qconv2d.thread-dp4a',
      ['QConv2D'],
      'vx_cuda_qconv2d_i8u8',
      /small-reduction-or-warp-launch-capacity.*per-spatial-tail/u,
    ],
  ]) {
    const variant = kernelVariants.find((item) => item.id === id);
    assert.equal(variant?.entrypointId, entrypoint);
    assert.deepEqual(variant?.operators, operators);
    assert.deepEqual(variant?.requiredFeatures,
      ['cuda.device-residency', 'cuda.sm61-dp4a']);
    assert.match(variant?.predicateId ?? '', predicate);
  }
  for (const [backend, entrypoint, feature] of [
    ['webgpu', 'qBatchMatMulDot.wgsl', 'webgpu.packed-4x8-integer-dot-product'],
    ['vulkan', 'qBatchMatMulDot', 'vulkan.packed-4x8-integer-dot-product'],
  ]) {
    const dot = kernelVariants.find(
      (variant) => variant.id === `${backend}.qbatch-matmul.dot`,
    );
    const scalar = kernelVariants.find(
      (variant) => variant.id === `${backend}.qbatch-matmul.scalar`,
    );
    assert.equal(dot?.entrypointId, entrypoint);
    assert.deepEqual(dot?.operators, ['QBatchMatMul']);
    assert.deepEqual(dot?.requiredFeatures, [feature]);
    assert.match(dot?.predicateId ?? '', /i32-safe/u);
    assert.equal(scalar?.requiredFeatures.length, 0);
    assert.deepEqual(scalar?.operators, ['QBatchMatMul']);
  }
  const vulkanQConvDot = kernelVariants.find(
    (variant) => variant.id === 'vulkan.qconv2d.dot-tiled',
  );
  const vulkanQConvTiled = kernelVariants.find(
    (variant) => variant.id === 'vulkan.qconv2d.tiled',
  );
  const vulkanQConvScalar = kernelVariants.find(
    (variant) => variant.id === 'vulkan.qconv2d.scalar',
  );
  assert.equal(vulkanQConvDot?.priority, 300);
  assert.deepEqual(vulkanQConvDot?.requiredFeatures,
    ['vulkan.packed-4x8-integer-dot-product']);
  assert.equal(vulkanQConvTiled?.entrypointId, 'qConv2DInt8Tiled');
  assert.equal(vulkanQConvTiled?.priority, 200);
  assert.deepEqual(vulkanQConvTiled?.requiredFeatures, []);
  assert.match(vulkanQConvTiled?.predicateId ?? '', /workgroup-8x4/u);
  assert.equal(vulkanQConvScalar?.priority, 100);
  const vulkanConv = kernelVariants.find(
    (variant) => variant.id === 'vulkan.conv2d.regular-out16',
  );
  assert.equal(vulkanConv?.entrypointId, 'conv2DRegularOut16');
  assert.deepEqual(vulkanConv?.operators, ['Conv2D']);
  assert.deepEqual(vulkanConv?.requiredFeatures, []);
  assert.equal(vulkanConv?.priority, 250);
  assert.match(vulkanConv?.predicateId ?? '', /output-channels16/u);
  const vulkanConvC3 = kernelVariants.find(
    (variant) => variant.id === 'vulkan.conv2d.c3-out16',
  );
  assert.equal(vulkanConvC3?.entrypointId, 'conv2DRegularC3Out16');
  assert.equal(vulkanConvC3?.priority, 260);
  assert.match(vulkanConvC3?.predicateId ?? '', /input-channels3/u);
  const webgpuConv = kernelVariants.find(
    (variant) => variant.id === 'webgpu.conv2d.regular-out16',
  );
  assert.equal(webgpuConv?.entrypointId, 'conv2DRegularOut16.wgsl');
  assert.deepEqual(webgpuConv?.operators, ['Conv2D']);
  assert.deepEqual(webgpuConv?.requiredFeatures, []);
  assert.equal(webgpuConv?.priority, 250);
  assert.match(webgpuConv?.predicateId ?? '', /output-channels16/u);
  assert.equal(kernelVariants.find(
    (variant) => variant.id === 'webgpu.conv2d.scalar',
  )?.entrypointId, 'conv2D.wgsl');
  assert.deepEqual(fullKernelVariants, []);
});
