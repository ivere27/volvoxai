import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  acquirePerformanceLock,
  assertFixedHotWorkSpan,
  assertMetricMatchesSpan,
  assertPerformanceGraphUnchanged,
  balancedBlockSchedule,
  bootstrapMedianInterval,
  buildBenchmarkArguments,
  calibrateGrayscaleSingleFixedWork,
  classifyPerformanceObject,
  compareReleaseTextLayout,
  grayscaleSingleFixedWorkPolicy,
  logicalObjectSource,
  median,
  parseBenchmarkAffinity,
  parseCpuList,
  parsePosixNmSymbols,
  parsePosixNmTextSymbols,
  summarizeMetricBlocks} from '../tools/native_w8a8_release_benchmark.mjs';

test('performance gate lock rejects a concurrent writer without stealing ownership', async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), 'volvoxai-perf-lock-'));
  const lockDirectory = path.join(temporary, 'gate.lock');
  const options = {
    buildDirectory: path.join(temporary, 'build'),
    output: path.join(temporary, 'result.json'),
  };
  try {
    const first = await acquirePerformanceLock(options, lockDirectory);
    await assert.rejects(
      acquirePerformanceLock(options, lockDirectory),
      /already locked/u,
    );
    await first.release();
    const second = await acquirePerformanceLock(options, lockDirectory);
    await second.release();
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

function metricBlocks(ratios, value = 10) {
  return ratios.map((ratio, index) => ({
    block: index + 1,
    runs: balancedBlockSchedule(index + 1).map((variant) => ({
      variant,
      hotMetricsMs: {
        synthetic: variant === 'baseline' ? value : value * ratio,
      },
    })),
  }));
}

test('CPU-list parsing expands ranges, preserves order, and removes duplicates', () => {
  assert.deepEqual(parseCpuList('4,0-2,2,7-8'), [4, 0, 1, 2, 7, 8]);
  assert.throws(() => parseCpuList('3-1'), /invalid CPU-list range/u);
  assert.throws(() => parseCpuList('0,,2'), /invalid CPU-list component/u);
});

test('benchmark timing affinity is explicit and requires four pool CPUs', () => {
  assert.deepEqual(
    parseBenchmarkAffinity('Timing affinity: single-cpu=10 pool-cpus=4\n'),
    { singleCpu: 10, poolCpuCount: 4 },
  );
  assert.throws(
    () => parseBenchmarkAffinity('Timing affinity: single-cpu=0 pool-cpus=3\n'),
    /invalid timing affinity/u,
  );
});

test('alternating ABBA blocks balance variant and pair order', () => {
  assert.deepEqual(
    balancedBlockSchedule(1),
    ['baseline', 'candidate', 'candidate', 'baseline'],
  );
  assert.deepEqual(
    balancedBlockSchedule(2),
    ['candidate', 'baseline', 'baseline', 'candidate'],
  );
  const schedule = [1, 2, 3, 4].flatMap(balancedBlockSchedule);
  assert.equal(schedule.filter((variant) => variant === 'baseline').length, 8);
  assert.equal(schedule.filter((variant) => variant === 'candidate').length, 8);
});

test('grayscale fixed work freezes one shared count from excluded warmups', () => {
  const variants = balancedBlockSchedule(1);
  const observations = [
    [56.292, 23],
    [56.553, 23],
    [57.327, 21],
    [54.780, 20],
  ];
  const warmups = variants.map((variant, index) => ({
    variant,
    ordinal: index + 1,
    warmup: true,
    block: 0,
    hotSpans: [{
      label: 'grayscale-qconv-single',
      milliseconds: observations[index][0],
      iterations: observations[index][1],
    }],
  }));
  const first = calibrateGrayscaleSingleFixedWork(warmups);
  const second = calibrateGrayscaleSingleFixedWork(warmups);
  assert.equal(first.formulaVersion, 'shared-fastest-warmup-v1');
  assert.equal(first.targetMilliseconds, 200);
  assert.equal(first.headroom, 1.1);
  assert.equal(first.fixedIterations, 90);
  assert.ok(first.designedMinimumSpanMilliseconds >= 220);
  assert.match(first.sha256, /^[0-9a-f]{64}$/u);
  assert.deepEqual(second, first);
  assert.throws(
    () => calibrateGrayscaleSingleFixedWork(warmups.slice(0, 3)),
    /exactly four warmup/u,
  );
  assert.throws(
    () => calibrateGrayscaleSingleFixedWork(
      warmups.map((run) => ({ ...run, variant: 'baseline' })),
    ),
    /two warmups per variant/u,
  );
});

test('grayscale fixed work honors a higher global minimum span', () => {
  assert.equal(grayscaleSingleFixedWorkPolicy(50).targetMilliseconds, 200);
  assert.equal(grayscaleSingleFixedWorkPolicy(500).targetMilliseconds, 500);
  assert.throws(
    () => grayscaleSingleFixedWorkPolicy(0),
    /minimum timed span must be positive/u,
  );
});

test('fixed hot work fails closed on a short span or a different count', () => {
  const fixedWork = {
    spanLabel: 'grayscale-qconv-single',
    targetMilliseconds: 200,
    fixedIterations: 90,
  };
  assert.doesNotThrow(() => assertFixedHotWorkSpan({
    label: 'grayscale-qconv-single',
    milliseconds: 220,
    iterations: 90,
  }, fixedWork));
  assert.throws(() => assertFixedHotWorkSpan({
    label: 'grayscale-qconv-single',
    milliseconds: 199.999,
    iterations: 90,
  }, fixedWork), /requires at least 200 ms/u);
  assert.throws(() => assertFixedHotWorkSpan({
    label: 'grayscale-qconv-single',
    milliseconds: 220,
    iterations: 89,
  }, fixedWork), /exactly 90 iterations/u);
});

test('measured benchmark arguments freeze work while warmups remain adaptive', () => {
  const affinity = { tasksetList: '0,2,4,6', selected: [0, 2, 4, 6] };
  const options = { minimumTimedMs: 50 };
  const warmup = buildBenchmarkArguments('/tmp/baseline', affinity, options);
  const measured = buildBenchmarkArguments('/tmp/candidate', affinity, options, {
    fixedIterations: 90,
  });
  assert.ok(!warmup.some((value) => value.startsWith('--grayscale-single-iterations=')));
  assert.ok(measured.includes('--grayscale-single-iterations=90'));
  assert.ok(measured.includes('--single-cpu=0'));
});

test('reported hot metric must equal the measured span average', () => {
  const span = {
    label: 'grayscale-qconv-single',
    milliseconds: 225,
    iterations: 90,
  };
  assert.doesNotThrow(() => assertMetricMatchesSpan(
    'grayscaleStemQconvSingle',
    2.5,
    span,
  ));
  assert.throws(() => assertMetricMatchesSpan(
    'grayscaleStemQconvSingle',
    2.6,
    span,
  ), /does not match/u);
});

test('private performance graph integrity rejects concurrent replacement', () => {
  const before = {
    variant: 'baseline',
    target: 'baseline-target',
    compiler: '/usr/bin/clang',
    nativeBuildDirectory: '/tmp/build/native',
    linkFile: '/tmp/build/native/link.txt',
    linkCommand: 'clang object.o -o baseline',
    linkInterfaceArguments: [],
    link: { bytes: 30, sha256: '1'.repeat(64), modifiedTimeNs: '100' },
    executable: '/tmp/build/native/baseline',
    executableEvidence: {
      bytes: 100,
      sha256: '2'.repeat(64),
      modifiedTimeNs: '101',
    },
    logicalSources: ['source.c.o'],
    objectGraphSha256: '3'.repeat(64),
    objects: [{
      token: 'object.o',
      category: 'baseline-o3',
      logicalSource: 'source.c.o',
      bytes: 50,
      sha256: '4'.repeat(64),
      modifiedTimeNs: '102',
    }],
    flags: {},
  };
  const unchanged = structuredClone(before);
  assert.doesNotThrow(() => assertPerformanceGraphUnchanged(
    before,
    unchanged,
    'during measurement',
  ));
  const replaced = structuredClone(before);
  replaced.executableEvidence.sha256 = '5'.repeat(64);
  assert.throws(() => assertPerformanceGraphUnchanged(
    before,
    replaced,
    'during measurement',
  ), /performance graph changed during measurement/u);
});

test('logical source projection ignores CMake target shard names', () => {
  assert.equal(
    logicalObjectSource(
      'CMakeFiles/volvoxai_inference_release_engine_shard_2_hot_objects.dir/' +
      'src/kernels/qlinear_w8a8.c.o',
    ),
    'src/kernels/qlinear_w8a8.c.o',
  );
  assert.equal(logicalObjectSource('-Wl,--gc-sections'), null);
});

test('performance graph classifier keeps profile-specific CLI anchors distinct', () => {
  const candidate =
    'CMakeFiles/volvoxai_native_w8a8_candidate_cli_anchor.dir/cli/main.c.o';
  const baseline =
    'CMakeFiles/volvoxai_native_w8a8_baseline_cli_anchor.dir/cli/main.c.o';
  assert.equal(classifyPerformanceObject(candidate, 'candidate'), 'candidate-cli-anchor');
  assert.equal(classifyPerformanceObject(baseline, 'baseline'), 'baseline-cli-anchor');
  assert.equal(classifyPerformanceObject(candidate, 'baseline'), 'unknown');
  assert.equal(classifyPerformanceObject(baseline, 'candidate'), 'unknown');
});

test('POSIX nm evidence requires one defined code address per hot symbol', () => {
  const symbols = parsePosixNmSymbols(
    'vx_qlinear_i8u8_packed t 00010c0e0 1eef\n' +
    'vx_qgemm_validate T 00010b4e0 16e\n' +
    'qconv2d_i8u8 t 0000ffb20 105c\n',
    ['vx_qlinear_i8u8_packed', 'vx_qgemm_validate', 'qconv2d_i8u8'],
  );
  assert.equal(symbols.vx_qlinear_i8u8_packed.address, '10c0e0');
  assert.equal(symbols.vx_qgemm_validate.address, '10b4e0');
  assert.equal(symbols.qconv2d_i8u8.address, 'ffb20');
  assert.throws(
    () => parsePosixNmSymbols('vx_qgemm_validate d 10b4e0 16e\n', ['vx_qgemm_validate']),
    /invalid code symbol/u,
  );
  assert.throws(
    () => parsePosixNmSymbols('', ['vx_qgemm_validate']),
    /exactly one defined/u,
  );
  assert.throws(
    () => parsePosixNmSymbols(
      'vx_qgemm_validate t 10b4e0 16e\nvx_qgemm_validate t 20b4e0 16e\n',
      ['vx_qgemm_validate'],
    ),
    /found 2/u,
  );
});

test('full text layout compares duplicate symbol multisets and maps CLI main', () => {
  const release = [
    'duplicate t 000020000000000001 10',
    'duplicate t 000020000000000101 20',
    'core T 000020000000000201 30',
    'main T 000020000000000301 40',
    'atexit t 000020000000000401 12',
    '_fini T 000020000000000415 0',
  ].join('\n');
  const candidate = [
    '_fini T 20000000000915 0',
    'helper w 20000000000801 8',
    'main T 20000000000401 100',
    'volvoxai_native_w8a8_release_cli_anchor T 20000000000301 40',
    'core T 20000000000201 30',
    'duplicate t 20000000000101 20',
    'duplicate t 20000000000001 10',
    'atexit t 20000000000901 12',
  ].join('\n');
  const result = compareReleaseTextLayout(release, candidate);
  assert.equal(result.releaseTextSymbolCount, 6);
  assert.equal(result.releaseTextSymbolNameCount, 5);
  assert.equal(result.candidateOnlyAtOrBeforeAnchorCount, 0);
  assert.equal(result.candidateOnlyAfterAnchorCount, 2);
  assert.equal(result.relocatableImplicitLinkerTailSymbols.length, 2);
  assert.match(result.productionTextProjectionSha256, /^[0-9a-f]{64}$/u);
  const reordered = compareReleaseTextLayout(
    `${release.split('\n').reverse().join('\n')}\n`,
    `${candidate.split('\n').reverse().join('\n')}\n`,
  );
  assert.equal(
    reordered.productionTextProjectionSha256,
    result.productionTextProjectionSha256,
  );
});

test('full text layout rejects missing duplicate and candidate-only pre-anchor code', () => {
  const release = [
    'duplicate t 100 10',
    'duplicate t 120 10',
    'main T 200 20',
  ].join('\n');
  const baseCandidate = [
    'duplicate t 100 10',
    'volvoxai_native_w8a8_release_cli_anchor T 200 20',
    'main T 300 80',
  ].join('\n');
  assert.throws(
    () => compareReleaseTextLayout(release, baseCandidate),
    /missing production text symbol/u,
  );
  assert.throws(
    () => compareReleaseTextLayout(
      'core t 100 10\nmain T 200 20\n',
      'core t 100 10\nwrapper t 180 8\n' +
      'volvoxai_native_w8a8_release_cli_anchor T 200 20\nmain T 300 80\n',
    ),
    /at or before the release CLI anchor/u,
  );
});

test('full text parser rejects malformed addresses and empty code sets', () => {
  assert.throws(
    () => parsePosixNmTextSymbols('bad T nope 10\n'),
    /invalid code address/u,
  );
  assert.throws(
    () => parsePosixNmTextSymbols('data D 100 10\n'),
    /no defined text symbols/u,
  );
});

test('median and seeded bootstrap are deterministic', () => {
  assert.equal(median([9, 1, 5]), 5);
  assert.equal(median([8, 2, 4, 6]), 5);
  const first = bootstrapMedianInterval([0.01, 0.02, 0.03, 0.04], 2_000, 7);
  const second = bootstrapMedianInterval([0.01, 0.02, 0.03, 0.04], 2_000, 7);
  assert.deepEqual(first, second);
});

test('the hard gate accepts an exact two-percent upper bound', () => {
  const result = summarizeMetricBlocks(metricBlocks([1.02, 1.02, 1.02, 1.02]), 'synthetic', {
    bootstrapResamples: 2_000,
    maxRegressionPercent: 2,
  });
  assert.equal(result.status, 'pass');
  assert.ok(Math.abs(result.upper95DeltaPercent - 2) < 1e-9);
});

test('the hard gate rejects a confidently greater-than-two-percent regression', () => {
  const result = summarizeMetricBlocks(
    metricBlocks([1.0201, 1.0201, 1.0201, 1.0201]),
    'synthetic',
    { bootstrapResamples: 2_000, maxRegressionPercent: 2 },
  );
  assert.equal(result.status, 'regression');
  assert.ok(result.lower95DeltaPercent > 2);
});

test('paired ABBA block logs cancel multiplicative monotonic drift', () => {
  let ordinal = 0;
  const blocks = [1, 2, 3, 4].map((block) => ({
    block,
    runs: balancedBlockSchedule(block).map((variant) => {
      const drift = 1.01 ** ordinal++;
      return {
        variant,
        hotMetricsMs: {
          synthetic: 10 * drift * (variant === 'candidate' ? 1.01 : 1),
        },
      };
    }),
  }));
  const result = summarizeMetricBlocks(blocks, 'synthetic', {
    bootstrapResamples: 2_000,
    maxRegressionPercent: 2,
  });
  assert.ok(Math.abs(result.estimateDeltaPercent - 1) < 1e-9);
  assert.equal(result.status, 'pass');
});

test('large between-block dispersion is evidence of instability', () => {
  const result = summarizeMetricBlocks(metricBlocks([0.90, 1.10, 0.90, 1.10]), 'synthetic', {
    bootstrapResamples: 2_000,
    maxRegressionPercent: 2,
  });
  assert.equal(result.status, 'unstable');
  assert.ok(result.robustSigmaPercent > 5);
});
