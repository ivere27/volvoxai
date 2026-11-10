#!/usr/bin/env node
/** Measure completed public WASM / WebGPU calls for a prepared TTS package. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {TtsSession, validateManifest, benchmarkPrepared, closeBenchmarkSession} from './TtsSession.js';
import {loadRuntimeRelease, releaseUrls} from '../common/RuntimeRelease.js';

const options = {backend: 'wasm', profile: 'full', warmup: 2, repeats: 5};
const argv = globalThis.Deno?.args ?? process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--report-waveform-differences') { options.reportWaveformDifferences = true; continue; }
  if (!['--manifest', '--out', '--backend', '--profile', '--warmup', '--repeats'].includes(argv[i]) || !argv[i + 1])
    throw new Error('Usage: benchmark_web.mjs --manifest FILE --out FILE [--backend wasm|webgpu --profile inference|full --warmup 2 --repeats 5]');
  options[argv[i].slice(2)] = argv[++i];
}
for (const key of ['warmup', 'repeats']) {
  options[key] = Number(options[key]);
  if (!Number.isSafeInteger(options[key]) || options[key] < (key === 'repeats' ? 1 : 0)) throw new Error(`Invalid ${key}.`);
}
if (!options.manifest || !options.out || !['wasm', 'webgpu'].includes(options.backend) ||
    !['inference', 'full'].includes(options.profile) || (options.backend === 'webgpu' && options.profile !== 'full'))
  throw new Error('A manifest, report path, and compatible backend/profile are required.');
const manifestUrl = pathToFileURL(path.resolve(options.manifest));
const manifestBytes = await fs.readFile(manifestUrl), manifest = validateManifest(JSON.parse(manifestBytes));
const hash = value => createHash('sha256').update(value).digest('hex');
const report = {format: 'volvoxai-tts-web-benchmark/v1', status: 'failed', backend: options.backend, profile: options.profile,
  manifest_sha256: hash(manifestBytes), model: manifest.model, scope: manifest.scope ?? 'whole-model',
  timing_scope: 'All completed Volvox stages with input binding, synchronization, inline host outputs and duration-shape admission; excludes model loading and WAV encoding.',
  runtime: globalThis.Deno?.version ?? {node: process.version}, warmup: options.warmup, repeats: options.repeats,
  report_waveform_differences: !!options.reportWaveformDifferences, cases: [], diagnostics: []};
report.caller_sha256 = hash(await fs.readFile(fileURLToPath(import.meta.url)));
report.session_sha256 = hash(await fs.readFile(new URL('./TtsSession.js', import.meta.url)));
const originalFetch = globalThis.fetch;
globalThis.fetch = async (source, init) => {
  const url = typeof source === 'string' ? source : source instanceof URL ? source.href : source?.url;
  return url?.startsWith('file:') ? new Response(await fs.readFile(fileURLToPath(url))) : originalFetch(source, init);
};
let session;
try {
  if (manifest.volvox.status !== 'ready') {
    report.status = 'unsupported'; report.reason = manifest.volvox.reason;
  } else {
    report.model_artifacts = {};
    const assets = new Set();
    if (manifest.volvox.pipeline) {
      for (const entry of manifest.volvox.pipeline) {
        assets.add(entry.manifest);
        const stageUrl = new URL(entry.manifest, manifestUrl);
        const stage = JSON.parse(await fs.readFile(stageUrl, 'utf8'));
        for (const asset of [stage.volvox.graph, ...stage.volvox.weights]) {
          const absolute = fileURLToPath(new URL(asset, stageUrl));
          assets.add(path.relative(path.dirname(fileURLToPath(manifestUrl)), absolute));
        }
      }
    } else for (const asset of [manifest.volvox.graph, ...manifest.volvox.weights]) assets.add(asset);
    for (const fixture of manifest.cases) for (const descriptor of [...fixture.input_files, ...fixture.output_files]) assets.add(descriptor.path);
    for (const asset of assets) report.model_artifacts[asset] = hash(await fs.readFile(new URL(asset, manifestUrl)));
    if (options.backend === 'webgpu') {
      const adapter = await globalThis.navigator?.gpu?.requestAdapter({powerPreference: 'high-performance'});
      if (!adapter) throw new Error('A WebGPU adapter is required.');
      const info = adapter.info ?? await adapter.requestAdapterInfo?.() ?? {};
      report.device = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, info[key] ?? '']));
      if (/swiftshader|llvmpipe|lavapipe|software/i.test(Object.values(report.device).join(' ')))
        throw new Error('Physical GPU required for this benchmark.');
    }
    const {jsUrl, wasmUrl} = releaseUrls(options.profile);
    report.artifacts = {js_sha256: hash(await fs.readFile(jsUrl)), wasm_sha256: hash(await fs.readFile(wasmUrl))};
    const runtime = await loadRuntimeRelease(options.profile), opened = performance.now();
    session = await TtsSession.open({manifestUrl, ...runtime, backend: options.backend,
      onDiagnostic: message => { report.diagnostics.push(String(message)); }});
    report.load_compile_ms = performance.now() - opened;
    for (const fixture of manifest.cases) {
      const {summary} = await benchmarkPrepared(session, fixture, options);
      report.cases.push(summary);
    }
    report.status = report.cases.every(row => row.comparison.pass) ? 'passed' : 'numerical_difference';
  }
} catch (error) {
  report.error = String(error);
  report.failure_report = error.report ?? error.errors?.find(value => value.report)?.report;
}
finally {
  try { await closeBenchmarkSession(session, report); }
  finally {
    globalThis.fetch = originalFetch;
    await fs.mkdir(path.dirname(path.resolve(options.out)), {recursive: true});
    await fs.writeFile(options.out, JSON.stringify(report, (_key, value) => typeof value === 'bigint' ? value.toString() : value, 2) + '\n');
  }
}
console.log(JSON.stringify({status: report.status, backend: report.backend, reason: report.reason ?? report.error,
  cases: report.cases.map(value => ({id: value.id, median_ms: value.timing.median_ms, comparison: value.comparison.pass}))}, null, 2));
if (report.status !== 'passed') process.exitCode = report.status === 'unsupported' ? 2 : report.status === 'numerical_difference' ? 3 : 1;
