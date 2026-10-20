#!/usr/bin/env node
/** Fixed-count API migration A/A and ABBA performance qualification.
 *
 * Register first: node tools/api_migration_performance.mjs register plan.json
 * Run later on a quiet host: node tools/api_migration_performance.mjs run OUT/preregistration.json
 *
 * Plan schema: volvoxai.api-migration-performance-plan/v1. Required fields:
 * qualificationPlan, output, cases[].{id,adapter,metrics,baseline,candidate}.
 * Commands are {cwd,argv,reportPath,artifacts}; argv is an executable plus its
 * arguments, never a shell string. {out}, {samples}, {warmup} are substituted.
 * Output paths are unique per run. Each artifact is a file path relative to cwd.
 * rounds defaults to 3, samples to 15, and cases[].warmup to 3 (10 when
 * nativeHandoff:true). Counts may increase only before registration.
 *
 * Metric keys: tinystories/WORKLOAD/FIELD_ms, generate/WORKLOAD/FIELD_ms,
 * migration-vqa/VARIANT/case:INDEX/request_ms,
 * receipt[/VARIANT/MODE][/case:INDEX]/FIELD_ms. FIELD_ms_per_token is also
 * supported and retains its ms/token unit. Components are URI-encoded.
 * Receipt reports can contain samples directly or rows[].samples. Every metric
 * must be explicitly selected before measurement; no post-hoc subset is gated.
 *
 * Optional quiet:{proofPath} or quiet:{command:{cwd,argv,reportPath,artifacts}}.
 * A proof is JSON {hostQuiet:true,validFrom:ISO,validUntil:ISO,evidence:...}; its
 * validity must cover the entire run. A command emits JSON {hostQuiet:true,
 * evidence:...} and runs before and after each measurement. Absence of proof
 * permits collection but never performance qualification.
 */
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdir, open, readFile, stat, writeFile} from 'node:fs/promises';
import {hostname, platform, arch, cpus} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {median, summarizeMetricBlocks} from './native_w8a8_release_benchmark.mjs';

const SCRIPT = fileURLToPath(import.meta.url);
const SUMMARY = fileURLToPath(new URL('./native_w8a8_release_benchmark.mjs', import.meta.url));
const PLAN = 'volvoxai.api-migration-performance-plan/v1';
const REGISTRATION = 'volvoxai.api-migration-performance-registration/v1';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const plain = value => JSON.stringify(value, null, 2) + '\n';
const component = value => encodeURIComponent(String(value));
function requireValue(condition, message) { if (!condition) throw new Error(message); }
const positiveInteger = value => Number.isSafeInteger(value) && value > 0;
async function json(filename) { return JSON.parse(await readFile(filename, 'utf8')); }
async function writeNew(filename, value) {
  await mkdir(path.dirname(filename), {recursive: true});
  await writeFile(filename, plain(value), {flag: 'wx'});
}
async function fingerprint(filename) {
  const bytes = await readFile(filename);
  return {path: path.resolve(filename), bytes: bytes.length, sha256: sha(bytes)};
}
function command(spec, base) {
  requireValue(spec && Array.isArray(spec.argv) && spec.argv.length &&
    spec.argv.every(arg => typeof arg === 'string' && arg.length), 'command argv must be a nonempty string array');
  requireValue(typeof spec.cwd === 'string' && typeof spec.reportPath === 'string' &&
    spec.reportPath.includes('{out}'), 'commands require cwd and a reportPath containing {out}');
  requireValue(Array.isArray(spec.artifacts) && spec.artifacts.length &&
    spec.artifacts.every(file => typeof file === 'string' && file.length), 'command artifacts must list harness and measured artifact files');
  const cwd = path.resolve(base, spec.cwd);
  return {...spec, cwd, artifacts: spec.artifacts.map(file => path.resolve(cwd, file))};
}
export function validatePlan(input, qualification, base = process.cwd()) {
  requireValue(input?.schema === PLAN, `plan schema must be ${PLAN}`);
  const policy = qualification?.latency;
  requireValue(policy?.median_noninferiority_fraction === 0.02 &&
    policy.one_sided_confidence === 0.95 && policy.inconclusive_is_pass === false &&
    policy.order === 'ABBA' && policy.aa_control === 'required before gate' &&
    policy.bootstrap === 'deterministic block bootstrap, 10000 resamples' &&
    policy.independent_rounds_minimum === 3 && policy.samples_per_case_per_round === 15 &&
    policy.warmup_vqa === 3 && policy.warmup_native_handoff === 10,
  'qualification policy differs from the immutable API migration gate');
  const rounds = input.rounds ?? 3, samples = input.samples ?? 15;
  requireValue(positiveInteger(rounds) && rounds >= 3 && positiveInteger(samples) && samples >= 15,
    'at least 3 independent rounds and 15 samples per run are required');
  requireValue(typeof input.output === 'string' && input.output && Array.isArray(input.cases) && input.cases.length,
    'output and nonempty cases are required');
  const ids = new Set();
  const cases = input.cases.map(item => {
    requireValue(typeof item.id === 'string' && /^[a-zA-Z0-9_-]+$/.test(item.id) && !ids.has(item.id), 'case IDs must be unique path-safe names');
    ids.add(item.id);
    requireValue(['receipt', 'tinystories', 'generate', 'migration-vqa'].includes(item.adapter), 'unknown report adapter');
    const warmup = item.warmup ?? (item.nativeHandoff ? 10 : 3);
    requireValue(positiveInteger(warmup) && warmup >= (item.nativeHandoff ? 10 : 3), 'insufficient warmup');
    requireValue(Array.isArray(item.metrics) && item.metrics.length &&
      item.metrics.every(metric => typeof metric === 'string' && metric.length) &&
      new Set(item.metrics).size === item.metrics.length, 'each case requires unique preregistered metrics');
    return {...item, warmup, baseline: command(item.baseline, base), candidate: command(item.candidate, base)};
  });
  let quiet = null;
  if (input.quiet) {
    requireValue(Boolean(input.quiet.proofPath) !== Boolean(input.quiet.command), 'choose quiet proofPath or command');
    quiet = input.quiet.proofPath ? {proofPath: path.resolve(base, input.quiet.proofPath)}
      : {command: command(input.quiet.command, base)};
  }
  return {...input, rounds, samples, cases, quiet, output: path.resolve(base, input.output),
    timeoutMs: input.timeoutMs ?? 3_600_000};
}
export function fixedSchedule(plan) {
  const schedule = [];
  for (const phase of ['aa', 'ab']) for (const item of plan.cases)
    for (let round = 0; round < plan.rounds; round++) for (const [position, variant] of
      ['baseline', 'candidate', 'candidate', 'baseline'].entries()) {
      schedule.push({id: `${phase}-${item.id}-${round}-${position}`, phase, caseId: item.id, round, position,
        variant, source: phase === 'aa' ? 'baseline' : variant});
    }
  return schedule;
}
async function executable(commandSpec) {
  const name = commandSpec.argv[0];
  const candidates = name.includes('/') ? [path.resolve(commandSpec.cwd, name)]
    : (process.env.PATH ?? '').split(path.delimiter).map(dir => path.resolve(dir, name));
  for (const filename of candidates) {
    try { if ((await stat(filename)).isFile()) return filename; } catch { /* Try next PATH entry. */ }
  }
  throw new Error(`executable unavailable: ${name}`);
}
async function commandEvidence(spec) {
  const files = new Set([await executable(spec), ...spec.artifacts]);
  for (const arg of spec.argv.slice(1)) if (!arg.includes('{')) {
    const filename = path.resolve(spec.cwd, arg);
    try { if ((await stat(filename)).isFile()) files.add(filename); } catch { /* Non-file argument. */ }
  }
  return {cwd: spec.cwd, argv: spec.argv, reportPath: spec.reportPath,
    files: await Promise.all([...files].map(fingerprint))};
}
export async function registerPlan(planPath) {
  const filename = path.resolve(planPath), base = path.dirname(filename);
  const input = await json(filename);
  requireValue(typeof input.qualificationPlan === 'string', 'qualificationPlan path is required');
  const qualificationPath = path.resolve(base, input.qualificationPlan);
  const qualification = await json(qualificationPath);
  const plan = validatePlan(input, qualification, base);
  requireValue(positiveInteger(plan.timeoutMs), 'timeoutMs must be positive');
  const commands = {};
  for (const item of plan.cases) commands[item.id] = {
    baseline: await commandEvidence(item.baseline), candidate: await commandEvidence(item.candidate)};
  const quiet = plan.quiet?.command ? {command: await commandEvidence(plan.quiet.command)}
    : plan.quiet?.proofPath ? {proof: await fingerprint(plan.quiet.proofPath)} : null;
  const registration = {schema: REGISTRATION, createdAt: new Date().toISOString(), plan, qualification,
    files: await Promise.all([filename, qualificationPath, SCRIPT, SUMMARY].map(fingerprint)), commands, quiet,
    schedule: fixedSchedule(plan), statistics: {bootstrapResamples: 10_000, maxRegressionPercent: 2,
      confidence: 0.95, aaTwoDirectionsRequired: true, p95ReportedOnly: true},
    host: {hostname: hostname(), platform: platform(), arch: arch(), cpu: cpus()[0]?.model, node: process.version}};
  const destination = path.join(plan.output, 'preregistration.json');
  await writeNew(destination, registration);
  return destination;
}

/** Return all native units converted to ms; grouping never mixes VQA cases. */
export function extractMetrics(report, adapter) {
  const metrics = {};
  function rows(prefix, samples) {
    requireValue(Array.isArray(samples) && samples.length, `${prefix} has no raw samples`);
    for (const sample of samples) {
      requireValue(sample && typeof sample === 'object' && !Array.isArray(sample), `${prefix} samples must contain named timing fields`);
      const group = sample.case_index === undefined ? prefix : `${prefix}/case:${component(sample.case_index)}`;
      for (const [field, value] of Object.entries(sample)) {
        const unit = /_(ms|ns)(?:_per_token)?$/.exec(field);
        if (!unit) continue;
        requireValue(typeof value === 'number' && Number.isFinite(value) && value >= 0, `${group}/${field} is invalid`);
        const key = `${group}/${field.replace(/_ns(?=_per_token$|$)/, '_ms')}`;
        (metrics[key] ??= []).push(unit[1] === 'ns' ? value / 1e6 : value);
      }
    }
  }
  if (adapter === 'receipt') {
    if (Array.isArray(report.rows)) for (const row of report.rows) {
      requireValue(row.supported !== false && !row.error, 'unsupported receipt row cannot qualify');
      rows(`receipt/${component(row.variant)}/${component(row.mode)}`, row.samples);
    } else rows('receipt', report.samples);
  } else if (adapter === 'tinystories' || adapter === 'generate') {
    requireValue(Array.isArray(report.workloads) && report.workloads.length, `${adapter} workloads missing`);
    for (const workload of report.workloads) rows(`${adapter}/${component(workload.name)}`, workload.samples);
  } else if (adapter === 'migration-vqa') {
    requireValue(report.variants && Object.keys(report.variants).length, 'VQA variants missing');
    for (const [name, variant] of Object.entries(report.variants)) {
      requireValue(variant.status === 'executed', `VQA ${name} did not execute successfully`);
      rows(`migration-vqa/${component(name)}`, variant.samples);
    }
  } else throw new Error(`unknown adapter: ${adapter}`);
  return metrics;
}
export function reportCounts(report, adapter) {
  const timing = report.timing ?? report;
  const warmup = adapter === 'receipt' ? report.warmup_per_case ?? timing.warmup : timing.warmup;
  const repeat = adapter === 'receipt' ? report.repeat_per_case ?? timing.repeat : timing.repeat;
  requireValue(positiveInteger(warmup) && positiveInteger(repeat),
    `${adapter} report requires explicit integer warmup and repeat counts`);
  return {warmup, repeat};
}
function percentile95(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.ceil(sorted.length * 0.95) - 1];
}
export function summarizePhase(runs, plan, phase) {
  const cases = {};
  for (const item of plan.cases) {
    const selected = runs.filter(run => run.phase === phase && run.caseId === item.id);
    requireValue(selected.length === plan.rounds * 4, `${phase}/${item.id} has incomplete blocks`);
    const blocks = Array.from({length: plan.rounds}, (_, round) => ({runs: selected.filter(run => run.round === round)
      .sort((a, b) => a.position - b.position)}));
    const metrics = {};
    for (const metric of item.metrics) {
      const stats = summarizeMetricBlocks(blocks, metric, {bootstrapResamples: 10_000, maxRegressionPercent: 2});
      const reverse = phase === 'aa' ? summarizeMetricBlocks(blocks.map(block => ({runs: block.runs.map(run => ({
        ...run, variant: run.variant === 'baseline' ? 'candidate' : 'baseline'}))})), metric,
      {bootstrapResamples: 10_000, maxRegressionPercent: 2}) : null;
      const samples = variant => selected.filter(run => run.variant === variant).flatMap(run => run.samples[metric]);
      metrics[metric] = {...stats, unit: metric.endsWith('_per_token') ? 'ms/token' : 'ms',
        ...(reverse ? {reverseControl: reverse} : {}),
        baselineP95Ms: percentile95(samples('baseline')), candidateP95Ms: percentile95(samples('candidate')),
        rawSampleCountPerVariant: {baseline: samples('baseline').length, candidate: samples('candidate').length},
        qualified: stats.status === 'pass' && (!reverse || reverse.status === 'pass')};
    }
    cases[item.id] = {metrics, qualified: Object.values(metrics).every(metric => metric.qualified)};
  }
  return {cases, qualified: Object.values(cases).every(item => item.qualified)};
}
function substitute(text, replacements) {
  return text.replace(/\{(out|samples|warmup)\}/g, (_, key) => String(replacements[key]));
}
async function spawnReport(spec, out, counts, timeoutMs) {
  await mkdir(out, {recursive: false});
  const replacements = {out, ...counts};
  const argv = spec.argv.map(arg => substitute(arg, replacements));
  const reportPath = path.resolve(spec.cwd, substitute(spec.reportPath, replacements));
  requireValue(reportPath.startsWith(out + path.sep), 'reportPath must be inside this run output directory');
  await writeNew(path.join(out, 'command.json'), {cwd: spec.cwd, argv, reportPath, startedAt: new Date().toISOString()});
  const stdout = await open(path.join(out, 'stdout.log'), 'wx');
  const stderr = await open(path.join(out, 'stderr.log'), 'wx');
  let result;
  try {
    result = await new Promise(resolve => {
      const child = spawn(argv[0], argv.slice(1), {cwd: spec.cwd, shell: false, stdio: ['ignore', stdout.fd, stderr.fd]});
      let error = null;
      const timeout = setTimeout(() => { error = `command exceeded ${timeoutMs} ms`; child.kill('SIGKILL'); }, timeoutMs);
      child.on('error', reason => { error = reason.message; });
      child.on('close', (code, signal) => { clearTimeout(timeout); resolve({code, signal, error}); });
    });
  } finally { await Promise.all([stdout.close(), stderr.close()]); }
  await writeNew(path.join(out, 'completion.json'), {code: result.code, signal: result.signal,
    error: result.error, finishedAt: new Date().toISOString()});
  requireValue(!result.error && result.code === 0, `command failed; inspect ${out}`);
  return {report: await json(reportPath), evidence: await fingerprint(reportPath), argv, cwd: spec.cwd};
}
async function verifyFingerprints(registration, current = null) {
  const commands = current ? [[registration.plan.cases.find(item => item.id === current.caseId), current.source]]
    : registration.plan.cases.flatMap(item => ['baseline', 'candidate'].map(variant => [item, variant]));
  for (const [item, variant] of commands) {
    requireValue(await executable(item[variant]) === registration.commands[item.id][variant].files[0].path,
      `preregistered executable resolution changed: ${item.id}/${variant}`);
  }
  if (registration.plan.quiet?.command)
    requireValue(await executable(registration.plan.quiet.command) === registration.quiet.command.files[0].path,
      'quiet probe executable resolution changed');
  const groups = [registration.files, ...commands.map(([item, variant]) => registration.commands[item.id][variant].files),
    registration.quiet?.command?.files ?? [],
  registration.quiet?.proof ? [registration.quiet.proof] : []];
  const unique = new Map();
  for (const expected of groups.flat()) {
    const previous = unique.get(expected.path);
    requireValue(!previous || (previous.sha256 === expected.sha256 && previous.bytes === expected.bytes),
      `conflicting fingerprints for ${expected.path}`);
    unique.set(expected.path, expected);
  }
  for (const expected of unique.values()) {
    const actual = await fingerprint(expected.path);
    requireValue(actual.sha256 === expected.sha256 && actual.bytes === expected.bytes,
      `preregistered file changed: ${expected.path}`);
  }
}
function quietProof(proof, started, finished) {
  return proof?.hostQuiet === true && Boolean(proof.evidence) &&
    Date.parse(proof.validFrom) <= started && Date.parse(proof.validUntil) >= finished;
}
export async function runRegistration(filename) {
  const registration = await json(filename);
  requireValue(registration.schema === REGISTRATION, 'unexpected preregistration schema');
  const {plan} = registration;
  const original = await json(registration.files[0].path);
  const qualification = await json(registration.files[1].path);
  const expectedPlan = validatePlan(original, qualification, path.dirname(registration.files[0].path));
  requireValue(JSON.stringify(plan) === JSON.stringify(expectedPlan) &&
    JSON.stringify(qualification) === JSON.stringify(registration.qualification),
  'registration differs from its original plan or qualification policy');
  requireValue(JSON.stringify(registration.schedule) === JSON.stringify(fixedSchedule(plan)), 'schedule differs from preregistration');
  await verifyFingerprints(registration);
  const started = Date.now();
  await writeNew(path.join(plan.output, 'execution-start.json'), {startedAt: new Date(started).toISOString(),
    preregistration: await fingerprint(filename), pid: process.pid});
  const result = {schema: 'volvoxai.api-migration-performance/v1', preregistration: await fingerprint(filename),
    startedAt: new Date(started).toISOString(), runs: [], quietEvidence: [],
    hostQuietVerified: false, performanceQualified: false};
  let quiet = Boolean(plan.quiet);
  try {
    const proof = plan.quiet?.proofPath ? await json(plan.quiet.proofPath) : null;
    for (const phase of ['aa', 'ab']) {
      for (const scheduled of registration.schedule.filter(run => run.phase === phase)) {
        const item = plan.cases.find(item => item.id === scheduled.caseId);
        const out = path.join(plan.output, scheduled.id);
        await verifyFingerprints(registration, scheduled);
        for (const moment of ['before', 'measurement', 'after']) {
          if (moment !== 'measurement') {
            if (plan.quiet?.command) {
              const check = await spawnReport(plan.quiet.command, `${out}-quiet-${moment}`, {}, plan.timeoutMs);
              result.quietEvidence.push({run: scheduled.id, moment, ...check});
              quiet &&= check.report.hostQuiet === true && Boolean(check.report.evidence);
              requireValue(quiet, `host was not quiet ${moment} ${scheduled.id}; this attempt cannot qualify`);
            }
            continue;
          }
          const captured = await spawnReport(item[scheduled.source], out, {samples: plan.samples, warmup: item.warmup}, plan.timeoutMs);
          const timing = reportCounts(captured.report, item.adapter);
          requireValue(timing.warmup === item.warmup && timing.repeat === plan.samples,
            `${scheduled.id} report does not prove preregistered warmup/sample counts`);
          if (captured.report.format === 'volvoxai-receipt-native-tensors/v1')
            requireValue(timing.warmup >= 10, 'native handoff reports require at least 10 preregistered warmups');
          const available = extractMetrics(captured.report, item.adapter), samples = {}, hotMetricsMs = {};
          for (const metric of item.metrics) {
            const values = available[metric];
            requireValue(values?.length === plan.samples && values.every(value => value > 0),
              `${scheduled.id}/${metric} requires exactly ${plan.samples} positive raw samples`);
            samples[metric] = values;
            hotMetricsMs[metric] = median(values);
          }
          const run = {...scheduled, report: captured.evidence, argv: captured.argv, cwd: captured.cwd, samples, hotMetricsMs};
          result.runs.push(run);
          await writeNew(path.join(out, 'measurement.json'), run);
        }
        await verifyFingerprints(registration, scheduled);
      }
      result[phase] = summarizePhase(result.runs, plan, phase);
      if (!result[phase].qualified) break;
    }
    await verifyFingerprints(registration);
    if (proof) {
      result.quietEvidence.push({proof, file: await fingerprint(plan.quiet.proofPath)});
      quiet &&= quietProof(proof, started, Date.now());
    }
    result.hostQuietVerified = quiet;
    result.performanceQualified = quiet && result.aa?.qualified === true && result.ab?.qualified === true;
  } catch (error) {
    result.error = error.message;
  } finally {
    result.finishedAt = new Date().toISOString();
    await writeNew(path.join(plan.output, 'report.json'), result);
  }
  return result;
}
async function main() {
  const [operation, filename, ...extra] = process.argv.slice(2);
  requireValue(filename && !extra.length && ['register', 'run'].includes(operation),
    'usage: api_migration_performance.mjs register PLAN.json | run PREREGISTRATION.json');
  if (operation === 'register') console.log(await registerPlan(filename));
  else {
    const result = await runRegistration(path.resolve(filename));
    console.log(plain({performanceQualified: result.performanceQualified, error: result.error ?? null}));
    if (!result.performanceQualified) process.exitCode = 1;
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === SCRIPT) {
  try { await main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
