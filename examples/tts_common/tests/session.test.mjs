import assert from 'node:assert/strict';
import test from 'node:test';
import {tensorValues, waveformDifference, pcm16Wav, timingSummary, validateManifest, benchmarkPrepared,
  closeBenchmarkSession, TtsSession, TtsPipeline} from '../TtsSession.js';

test('I64 token conversion rejects overflow instead of wrapping token IDs', () => {
  const data = new BigInt64Array([0n, 2147483647n, -2147483648n]);
  const descriptor = {name: 'tokens', dtype: 'int64', shape: [3]};
  assert.deepEqual([...tensorValues(new Uint8Array(data.buffer), descriptor, true)], [0, 2147483647, -2147483648]);
  data[1] = 2147483648n;
  assert.throws(() => tensorValues(new Uint8Array(data.buffer), descriptor, true), /I32/);
  assert.throws(() => tensorValues(new Uint8Array(data.buffer), {...descriptor, shape: [4]}, true), /bytes/);
});
test('comparison refuses invalid audio and reports actual numerical error', () => {
  assert.equal(waveformDifference([.1, -.2], [.1, -.2]).pass, true);
  assert.equal(waveformDifference([.1, .2], [.1, -.2]).pass, false);
  assert.throws(() => waveformDifference([NaN], [0]), /non-finite/);
  assert.throws(() => waveformDifference([0], [NaN]), /non-finite/);
  assert.throws(() => waveformDifference([0], [Infinity]), /non-finite/);
  assert.throws(() => waveformDifference([], []), /empty/);
  assert.throws(() => waveformDifference([1], [1, 2]), /lengths/);
});
test('WAV encoding has a valid mono PCM header and clips without normalization', () => {
  const bytes = pcm16Wav([-2, -1, 0, 1, 2], 24000), view = new DataView(bytes.buffer);
  assert.equal(new TextDecoder().decode(bytes.slice(0, 4)), 'RIFF');
  assert.equal(view.getUint32(24, true), 24000);
  assert.equal(view.getUint32(40, true), 10);
  assert.deepEqual(Array.from({length: 5}, (_, i) => view.getInt16(44 + i * 2, true)), [-32768, -32768, 0, 32767, 32767]);
  assert.throws(() => pcm16Wav([Infinity], 24000), /non-finite/);
});
test('timing uses median and interpolated p90 with waveform duration', () => {
  const summary = timingSummary([30, 10, 40, 20], 2);
  assert.equal(summary.median_ms, 25); assert.equal(summary.p90_ms, 37);
  assert.equal(summary.real_time_factor, .0125);
});
test('unsupported full model is a valid manifest, missing export status is not', () => {
  const manifest = {format: 'volvoxai-tts-example/v1', sample_rate: 16000, cases: [{id: 'hello'}], volvox: {status: 'unsupported'}};
  assert.equal(validateManifest(manifest), manifest);
  assert.throws(() => validateManifest({...manifest, volvox: {}}), /status/);
  assert.throws(() => validateManifest({...manifest, cases: []}), /no prepared requests/);
  assert.throws(() => validateManifest({...manifest, cases: undefined}), /no prepared requests/);
});
test('every first, warmup and measured output must pass before timings are published', async () => {
  const descriptor = {name: 'audio', dtype: 'float32', shape: [1, 2], path: 'reference.bin'};
  const fixture = {id: 'sample', validation: {reference_engine: 'onnxruntime'}, output_files: [descriptor]};
  for (let wrongCall = 0; wrongCall < 5; wrongCall++) {
    let calls = 0;
    const session = {manifest: {sample_rate: 24000}, readBytes: async () => new Uint8Array(Float32Array.of(.1, -.2).buffer),
      prepare: async () => ({}), execute: async () => ({shape: [1, 2], elapsed_ms: 1, report: {route: {}},
        audio: Float32Array.of(.1, calls++ === wrongCall ? .2 : -.2)})};
    await assert.rejects(benchmarkPrepared(session, fixture, {warmup: 2, repeats: 2}), /comparison failed/);
    assert.equal(calls, wrongCall + 1);
  }
});


test('pipeline executes every stage and feeds the newly computed intermediate on every request', async () => {
  let encoderCalls = 0, decoderCalls = 0;
  const pipeline = Object.assign(new TtsPipeline(), {backend: 'wasm', manifest: {volvox: {
    shape_feedback: {stage: 'encoder', output: 'frames', index: 0, expected: 2}}}, stages: [
    {name: 'encoder', session: {manifest: {volvox: {inputs: {tokens: 'x'}}}, prepareValues: x => x,
      executeTensors: async ({tokens}) => {
        encoderCalls++;
        return {outputs: {features: {shape: [2], data: Float32Array.from(tokens.data, x => x*2)},
          frames: {shape: [1], data: Int32Array.of(2)}}, report: {route: {attested: true}}};
      }}},
    {name: 'decoder', session: {manifest: {volvox: {inputs: {features: 'x'}}}, prepareValues: x => x,
      executeTensors: async ({features}) => {
        decoderCalls++;
        return {outputs: {audio: {shape: [1,2], data: Float32Array.from(features.data, x => x+1)}}, report: {route: {attested: true}}};
      }}}
  ]});
  for (const values of [[1,2], [3,4]]) {
    const result = await pipeline.execute({tokens: {shape: [2], data: Float32Array.from(values)}});
    assert.deepEqual([...result.audio], values.map(x => 2*x+1));
    assert.deepEqual(Object.keys(result.report.route.stages), ['encoder','decoder']);
  }
  assert.equal(encoderCalls, 2); assert.equal(decoderCalls, 2);
  pipeline.manifest.volvox.shape_feedback.expected = 3;
  await assert.rejects(pipeline.execute({tokens: {shape: [2], data: Float32Array.of(5,6)}}), /admits 3 frames/);
  assert.equal(encoderCalls, 3); assert.equal(decoderCalls, 2);
});


test('diagnostic measurement retains numerical failure and never accepts nonfinite audio', async () => {
  const fixture = {id: 'sample', validation: {reference_engine: 'onnxruntime'},
    output_files: [{name: 'audio', dtype: 'float32', shape: [1], path: 'reference.bin'}]};
  const session = {manifest: {sample_rate: 24000}, readBytes: async () => new Uint8Array(Float32Array.of(.1).buffer),
    prepare: async () => ({}), execute: async () => ({shape: [1], elapsed_ms: 1, report: {}, audio: Float32Array.of(.2)})};
  const measured = await benchmarkPrepared(session, fixture, {warmup: 1, repeats: 2, reportWaveformDifferences: true});
  assert.equal(measured.summary.comparison.pass, false);
  assert.equal(measured.summary.comparison.strict_elementwise_pass, false);
  assert.equal(measured.summary.timing.samples_ms.length, 2);
  session.execute = async () => ({shape: [1], elapsed_ms: 1, report: {}, audio: Float32Array.of(NaN)});
  await assert.rejects(benchmarkPrepared(session, fixture, {reportWaveformDifferences: true}), /non-finite/);
});

test('declared waveform bounds retain the independent strict comparison', () => {
  const comparison = waveformDifference([.00002], [0], {waveform: {max_absolute_error: 1/32767, max_rmse: 1/32767}});
  assert.equal(comparison.pass, true);
  assert.equal(comparison.strict_elementwise_pass, false);
  assert.equal(comparison.mismatched_samples, 1);
});

test('failed pipeline open closes every earlier stage even if cleanup rejects', async () => {
  const originalOpen = TtsSession.open, closed = [];
  let opened = 0;
  TtsSession.open = async () => {
    const index = opened++;
    if (index === 2) throw new Error('third stage load failed');
    return {close: async () => { closed.push(index); if (index === 1) throw new Error('second stage close failed'); }};
  };
  const document = {format: 'volvoxai-tts-example/v1', sample_rate: 24000,
    cases: [{id: 'request'}], volvox: {status: 'ready'}};
  try {
    await assert.rejects(TtsPipeline.open({manifest: {volvox: {pipeline: [
      {name: 'first', manifest: 'first.json'}, {name: 'second', manifest: 'second.json'}, {name: 'third', manifest: 'third.json'}]}},
      base: new URL('file:///tmp/manifest.json'),
      api: {EngineHost: class { async close() { closed.push('host'); } }},
      readBytes: async () => new TextEncoder().encode(JSON.stringify(document))}),
    error => /third stage load failed/.test(error.message) && /second stage close failed/.test(error.message));
    assert.deepEqual(closed, [1, 0, 'host']);
  } finally { TtsSession.open = originalOpen; }
});

test('pipeline owns one host across stages and closes it once after all stage resources', async () => {
  const originalOpen = TtsSession.open, hosts = [], closed = [], diagnostic = [];
  const api = {EngineHost: class {
    constructor(options) { hosts.push(this); options.onDiagnostic('created'); }
    async close() { closed.push('host'); }
  }};
  TtsSession.open = async ({host, manifestUrl}) => {
    assert.equal(host, hosts[0]);
    return {close: async () => closed.push(manifestUrl.pathname)};
  };
  const document = {format: 'volvoxai-tts-example/v1', sample_rate: 24000,
    cases: [{id: 'request'}], volvox: {status: 'ready'}};
  try {
    const pipeline = await TtsPipeline.open({manifest: {volvox: {pipeline: [
      {name: 'encoder', manifest: 'encoder.json'}, {name: 'decoder', manifest: 'decoder.json'}]}},
      api, base: new URL('file:///tmp/manifest.json'), onDiagnostic: message => diagnostic.push(message),
      readBytes: async () => new TextEncoder().encode(JSON.stringify(document))});
    assert.equal(hosts.length, 1);
    await pipeline.close(); await pipeline.close();
    assert.deepEqual(closed, ['/tmp/decoder.json', '/tmp/encoder.json', 'host']);
    assert.deepEqual(diagnostic, ['created']);
  } finally { TtsSession.open = originalOpen; }
});

test('borrowed stage releases every handle without closing the pipeline host', async () => {
  const released = [], type = class { constructor(value) { Object.assign(this, value); } };
  const session = Object.assign(new TtsSession(), {ownsHost: false,
    host: {close: async () => released.push('host')}, api: {pb: {Ref: type}},
    resources: [['runtime', 'Ref', {id: 1}], ['model', 'Ref', {id: 2}], ['context', 'Ref', {id: 3}]],
    client: Object.fromEntries(['runtime', 'model', 'context'].map(method => [method, async ({id}) => {
      released.push(id); if (method === 'model') throw Error('release failed'); return {status: 0};
    }]))});
  await assert.rejects(session.close(), /release failed/);
  await session.close();
  assert.deepEqual(released, [3, 2, 1]);
});

test('benchmark cleanup failure invalidates a previously successful receipt', async () => {
  const report = {status: 'passed', error: 'earlier diagnostic'};
  await closeBenchmarkSession({close: async () => { throw new Error('host close refused'); }}, report);
  assert.equal(report.status, 'failed');
  assert.match(report.error, /earlier diagnostic.*host close refused/);
});

test('a zero-node execution cannot attest the selected backend', async () => {
  const session = Object.assign(new TtsSession(), {backend: 'wasm', client: {
    execute: async () => ({report: {status: 0, backend: 'wasm', route: {
      provider: 'wasm', builtin: true, attested: true, activeNodes: 0, selectedNodes: 0,
      fallbackNodes: 0, missingNodes: 0}}})}});
  await assert.rejects(session.executeTensors({}), /did not attest complete/);
});

test('diagnostic measurements preserve the worst first-run error when later runs match', async () => {
  let count = 0;
  const fixture = {id: 'sample', validation: {reference_engine: 'onnxruntime'},
    output_files: [{name: 'audio', dtype: 'float32', shape: [1], path: 'reference.bin'}]};
  const session = {manifest: {sample_rate: 24000}, readBytes: async () => new Uint8Array(Float32Array.of(0).buffer),
    prepare: async () => ({}), execute: async () => ({shape: [1], elapsed_ms: 1, report: {},
      audio: Float32Array.of(count++ ? 0 : 1)})};
  const {summary} = await benchmarkPrepared(session, fixture, {warmup: 0, repeats: 1, reportWaveformDifferences: true});
  assert.equal(summary.comparison.pass, false);
  assert.equal(summary.comparison.strict_elementwise_pass, false);
  assert.equal(summary.comparison.max_rmse_all_executions, 1);
  assert.equal(summary.comparison.max_mismatched_samples_all_executions, 1);
});

test('missing and Volvox reference provenance are rejected before an ONNX comparison executes', async () => {
  for (const fixture of [{}, {validation: {}}, {validation: {reference_engine: 'volvoxai'}}])
    await assert.rejects(benchmarkPrepared({}, fixture, {reportWaveformDifferences: true}), /ONNX Runtime reference/);
});

test('benchmark rejects invalid reference samples and retains its deterministic parity requirement', async () => {
  const fixture = {id: 'sample', validation: {reference_engine: 'onnxruntime'},
    output_files: [{name: 'audio', dtype: 'float32', shape: [1], path: 'reference.bin'}]};
  for (const value of [NaN, Infinity]) {
    const session = {manifest: {sample_rate: 24000}, readBytes: async () => new Uint8Array(Float32Array.of(value).buffer),
      prepare: async () => ({}), execute: async () => ({shape: [1], elapsed_ms: 1, report: {}, audio: Float32Array.of(0)})};
    await assert.rejects(benchmarkPrepared(session, fixture, {reportWaveformDifferences: true}), /non-finite/);
  }
  await assert.rejects(benchmarkPrepared({}, {...fixture,
    validation: {...fixture.validation, deterministic: false}}), /deterministic/);
});
