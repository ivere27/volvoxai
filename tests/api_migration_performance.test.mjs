import assert from 'node:assert/strict';
import test from 'node:test';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {extractMetrics, fixedSchedule, registerPlan, reportCounts, runRegistration, summarizePhase, validatePlan}
  from '../tools/api_migration_performance.mjs';

const qualification = {latency: {median_noninferiority_fraction: 0.02, one_sided_confidence: 0.95,
  inconclusive_is_pass: false, order: 'ABBA', aa_control: 'required before gate',
  bootstrap: 'deterministic block bootstrap, 10000 resamples', independent_rounds_minimum: 3,
  samples_per_case_per_round: 15, warmup_vqa: 3, warmup_native_handoff: 10}};
const command = {cwd: '.', argv: [process.execPath, 'harness.mjs', '--out', '{out}',
  '--warmup', '{warmup}', '--repeat', '{samples}'], reportPath: '{out}/result.json', artifacts: ['harness.mjs']};
function input() {
  return {schema: 'volvoxai.api-migration-performance-plan/v1', output: 'measurements',
    qualificationPlan: 'qualification.json', cases: [{id: 'tiny', adapter: 'tinystories',
      metrics: ['tinystories/short/request_ms'], baseline: command, candidate: command}]};
}
test('preregistration freezes all AA controls before fixed ABBA candidate blocks', () => {
  const plan = validatePlan(input(), qualification);
  const schedule = fixedSchedule(plan);
  assert.equal(schedule.length, 24);
  assert.ok(schedule.slice(0, 12).every(run => run.phase === 'aa' && run.source === 'baseline'));
  assert.deepEqual(schedule.slice(12, 16).map(run => run.source), ['baseline', 'candidate', 'candidate', 'baseline']);
  assert.throws(() => validatePlan({...input(), rounds: 2}, qualification), /at least 3/);
  assert.throws(() => validatePlan(input(), {latency: {...qualification.latency, median_noninferiority_fraction: .03}}), /immutable/);
  const native = input(); native.cases[0].nativeHandoff = true;
  assert.equal(validatePlan(native, qualification).cases[0].warmup, 10);
  native.cases[0].warmup = 3;
  assert.throws(() => validatePlan(native, qualification), /warmup/);
});
test('adapters preserve workload/case identity and convert nanoseconds to milliseconds', () => {
  assert.deepEqual(extractMetrics({variants: {fp32: {status: 'executed', samples: [
    {case_index: 0, request_ns: 2e6}, {case_index: 1, request_ns: 9e6}, {case_index: 0, request_ns: 3e6},
  ]}}}, 'migration-vqa'), {'migration-vqa/fp32/case:0/request_ms': [2, 3],
    'migration-vqa/fp32/case:1/request_ms': [9]});
  assert.deepEqual(extractMetrics({rows: [{variant: 'int8', mode: 'native', samples: [
    {case_index: 0, request_ms: 4, decoder_ms: 3},
  ]}]}, 'receipt'), {'receipt/int8/native/case:0/request_ms': [4],
    'receipt/int8/native/case:0/decoder_ms': [3]});
  assert.deepEqual(extractMetrics({workloads: [{name: 'short', samples: [{request_ms: 2, first_token_ms: 1}]}]},
    'tinystories'), {'tinystories/short/request_ms': [2], 'tinystories/short/first_token_ms': [1]});
  assert.deepEqual(extractMetrics({samples: [{case_index: 2, request_ms: 4, decode_ms_per_token: 1.5}]}, 'receipt'),
    {'receipt/case:2/request_ms': [4], 'receipt/case:2/decode_ms_per_token': [1.5]});
  assert.throws(() => extractMetrics({variants: {fp32: {status: 'failed', samples: []}}}, 'migration-vqa'), /did not execute/);
});
test('report count adapters accept receipt per-case, handoff top-level and migration timing fields', () => {
  assert.deepEqual(reportCounts({warmup_per_case: 3, repeat_per_case: 15}, 'receipt'), {warmup: 3, repeat: 15});
  assert.deepEqual(reportCounts({warmup: 10, repeat: 15}, 'receipt'), {warmup: 10, repeat: 15});
  assert.deepEqual(reportCounts({timing: {warmup: 3, repeat: 15}}, 'tinystories'), {warmup: 3, repeat: 15});
  assert.deepEqual(reportCounts({timing: {warmup: 3, repeat: 15}}, 'migration-vqa'), {warmup: 3, repeat: 15});
  assert.throws(() => reportCounts({warmup: '3', repeat: 15}, 'receipt'), /integer/);
});
test('Generate chunks and dtypes retain distinct metrics including context costs', () => {
  const plan = input();
  plan.cases[0].adapter = 'generate';
  plan.cases[0].metrics = ['generate/F32%2Fone6/request_ms', 'generate/I8%2Fsix1/continuation_ms'];
  assert.equal(validatePlan(plan, qualification).cases[0].adapter, 'generate');
  const report = {warmup: 3, repeat: 15, repetitions_per_sample: 10, workloads: [
    {name: 'F32/one6', samples: [{request_ms: 2, first_packet_ms: 2, context_create_ms: .1, request_with_context_ms: 2.2}]},
    {name: 'I8/six1', samples: [{request_ms: 6, first_packet_ms: 1, continuation_ms: 5}]},
  ]};
  assert.deepEqual(reportCounts(report, 'generate'), {warmup: 3, repeat: 15});
  assert.deepEqual(extractMetrics(report, 'generate'), {
    'generate/F32%2Fone6/request_ms': [2], 'generate/F32%2Fone6/first_packet_ms': [2],
    'generate/F32%2Fone6/context_create_ms': [.1], 'generate/F32%2Fone6/request_with_context_ms': [2.2],
    'generate/I8%2Fsix1/request_ms': [6], 'generate/I8%2Fsix1/first_packet_ms': [1],
    'generate/I8%2Fsix1/continuation_ms': [5],
  });
  assert.throws(() => extractMetrics({workloads: [{name: 'F32/one6', samples: []}]}, 'generate'), /no raw samples/);
});
function measured(plan, ratio) {
  return fixedSchedule(plan).map(run => {
    const value = run.variant === 'baseline' ? 10 : 10 * ratio;
    return {...run, samples: {'tinystories/short/request_ms': new Array(15).fill(value)},
      hotMetricsMs: {'tinystories/short/request_ms': value}};
  });
}
test('A/A checks both directions, candidate uses strict 2 percent bound, and p95 is reported', () => {
  const plan = validatePlan(input(), qualification);
  const aa = summarizePhase(measured(plan, 1), plan, 'aa');
  assert.equal(aa.qualified, true);
  assert.equal(aa.cases.tiny.metrics['tinystories/short/request_ms'].candidateP95Ms, 10);
  assert.equal(summarizePhase(measured(plan, .9), plan, 'aa').qualified, false);
  assert.equal(summarizePhase(measured(plan, 1.01), plan, 'ab').qualified, true);
  assert.equal(summarizePhase(measured(plan, 1.03), plan, 'ab').qualified, false);
  const unstable = measured(plan, 1);
  for (const run of unstable) if (run.variant === 'candidate' && run.round === 1)
    run.hotMetricsMs['tinystories/short/request_ms'] = 20;
  const uncertainty = summarizePhase(unstable, plan, 'ab');
  assert.equal(uncertainty.qualified, false);
  assert.equal(uncertainty.cases.tiny.metrics['tinystories/short/request_ms'].status, 'inconclusive');
});
test('registration hashes commands/artifacts and cannot overwrite existing evidence', async t => {
  const directory = await mkdtemp(path.join(tmpdir(), 'vx-api-performance-'));
  t.after(() => rm(directory, {recursive: true, force: true}));
  await writeFile(path.join(directory, 'harness.mjs'), 'throw new Error("this harness must not execute during registration");\n');
  await writeFile(path.join(directory, 'qualification.json'), JSON.stringify(qualification));
  await writeFile(path.join(directory, 'plan.json'), JSON.stringify(input()));
  const filename = await registerPlan(path.join(directory, 'plan.json'));
  const registered = JSON.parse(await readFile(filename));
  assert.equal(registered.schedule.length, 24);
  assert.equal(registered.quiet, null);
  assert.ok(registered.commands.tiny.baseline.files.some(file => file.path.endsWith('harness.mjs') && file.sha256.length === 64));
  await assert.rejects(registerPlan(path.join(directory, 'plan.json')), error => error.code === 'EEXIST');
});
test('failed idle proof stops before any measurement and preserves rejection evidence', async t => {
  const directory = await mkdtemp(path.join(tmpdir(), 'vx-api-performance-noise-'));
  t.after(() => rm(directory, {recursive: true, force: true}));
  await writeFile(path.join(directory, 'harness.mjs'),
    'import {writeFileSync} from "node:fs"; writeFileSync("timing-was-started", "yes");\n');
  await writeFile(path.join(directory, 'idle.mjs'),
    'import {writeFileSync} from "node:fs"; writeFileSync(process.argv.at(-1), JSON.stringify({hostQuiet:false,evidence:{externalCpuBusy:true}}));\n');
  const plan = input();
  plan.quiet = {command: {cwd: '.', argv: [process.execPath, 'idle.mjs', '{out}/report.json'],
    reportPath: '{out}/report.json', artifacts: ['idle.mjs']}};
  await writeFile(path.join(directory, 'qualification.json'), JSON.stringify(qualification));
  await writeFile(path.join(directory, 'plan.json'), JSON.stringify(plan));
  const registration = await registerPlan(path.join(directory, 'plan.json'));
  const result = await runRegistration(registration);
  assert.equal(result.performanceQualified, false);
  assert.equal(result.hostQuietVerified, false);
  assert.match(result.error, /host was not quiet before/);
  assert.deepEqual(result.runs, []);
  assert.equal(result.quietEvidence.length, 1);
  assert.equal(result.quietEvidence[0].report.hostQuiet, false);
  await assert.rejects(readFile(path.join(directory, 'timing-was-started')), {code: 'ENOENT'});
});
