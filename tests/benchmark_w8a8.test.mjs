import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createBenchmarkPlan,
  formatBenchmarkReport,
  parseBenchmarkArguments,
  runBenchmark,
} from '../tools/benchmark_w8a8.mjs';

test('W8A8 benchmark dry run is deterministic and states its comparison limits', () => {
  const options = parseBenchmarkArguments([
    '--dry-run', '--json', '--seed', '17',
    '--linear', '1,4,3',
    '--conv', '1,3,3,2,3,3',
  ]);
  const first = createBenchmarkPlan(options);
  const second = createBenchmarkPlan(options);

  assert.deepEqual(first, second);
  assert.equal(first.dry_run, true);
  assert.equal(first.execution.backend, 'CPU(JS) portable reference kernels');
  assert.match(first.execution.scope, /excludes graph construction\/loading/);
  assert.equal(first.workloads.length, 2);

  const qlinear = first.workloads.find((workload) => workload.op === 'QLinear');
  assert.deepEqual(qlinear.variants.map((variant) => variant.id), [
    'canonical_w8a8', 'w8a32_reference', 'fp32_reference',
  ]);
  assert.equal(qlinear.variants[0].storage.live_tensor_bytes, 31);
  assert.equal(qlinear.variants[1].storage.live_tensor_bytes, 76);
  assert.equal(qlinear.variants[2].storage.live_tensor_bytes, 88);

  const qconv = first.workloads.find((workload) => workload.op === 'QConv2D');
  assert.deepEqual(qconv.variants.map((variant) => variant.id), ['canonical_w8a8', 'fp32_reference']);
  assert.match(qconv.unavailable_variants[0].reason, /no corresponding W8A32 Conv2D reference kernel/);
  assert.match(formatBenchmarkReport(first), /Dry run: no kernels executed/);
});

test('W8A8 benchmark executes a small QLinear comparison and reports timings', () => {
  const report = runBenchmark({
    op: 'qlinear',
    warmup: 0,
    iterations: 1,
    seed: 19,
    linear: { rows: 1, inputFeatures: 4, outputFeatures: 3 },
  });
  const variants = report.workloads[0].variants;

  assert.equal(report.dry_run, false);
  assert.deepEqual(variants.map((variant) => variant.id), [
    'canonical_w8a8', 'w8a32_reference', 'fp32_reference',
  ]);
  for (const variant of variants) {
    assert.equal(variant.timing_ms.warmup_iterations, 0);
    assert.equal(variant.timing_ms.iterations, 1);
    assert.ok(variant.timing_ms.total_ms >= 0);
    assert.match(variant.result_checksum, /^0x[0-9a-f]{8}$/);
  }
});
