import assert from 'node:assert/strict';
import {readFile, stat} from 'node:fs/promises';
import {createServer} from 'node:http';
import {resolve, sep} from 'node:path';
import test from 'node:test';
import {readPackageVersion} from '../../../tools/release_version.mjs';
import {createPrng, sampleToken} from '../Sampling.js';
import {TinyStoriesSession} from '../TinyStoriesSession.js';
import {Tokenizer} from '../Tokenizer.js';

test('seeded sampling is repeatable, greedy skips randomness, probabilities are normalized', () => {
  const logits = Float32Array.of(3, 1, 5, 2), left = createPrng(42), right = createPrng(42);
  for (let i = 0; i < 30; i++) assert.deepEqual(sampleToken(logits, 0, 4, .7, left, 3), sampleToken(logits, 0, 4, .7, right, 3));
  const greedy = sampleToken(logits, 0, 4, 0, () => { throw new Error('greedy used PRNG'); }, 4);
  assert.equal(greedy.tokenId, 2);
  const sample = sampleToken(logits, 0, 4, .7, createPrng(0), 3);
  assert.ok(Math.abs(sample.candidates.reduce((n, row) => n + row.probability, 0) - 1) < 1e-7);
  assert.deepEqual(sample.candidates.map(row => row.tokenId), [2, 0, 3]);
  assert.throws(() => sampleToken(Float32Array.of(NaN), 0, 1, 1, createPrng(1)), /Non-finite/);
});

test('real TinyStories generation, isolated prefill/decode captures and token logits agree', async t => {
  const root = resolve(new URL('../../..', import.meta.url).pathname), version = readPackageVersion();
  const modelDirectory = process.env.VOLVOXAI_TINYSTORIES_PACKAGE ?? 'models/tinystories_1m';
  try { await stat(resolve(root, modelDirectory, 'model.safetensors')); await stat(resolve(root, `dist/${version}/volvoxai.wasm`)); }
  catch { t.skip('build the full runtime and make models_tinystories first'); return; }
  const server = createServer(async (request, response) => {
    const path = resolve(root, '.' + new URL(request.url, 'http://localhost').pathname);
    if (!path.startsWith(root + sep)) { response.writeHead(403).end(); return; }
    try { response.end(await readFile(path)); } catch { response.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}/${modelDirectory}/`;
  const api = await import(new URL(`../../../dist/${version}/volvoxai.js`, import.meta.url));
  const graph = JSON.parse(await readFile(resolve(root, modelDirectory, 'graph.json'), 'utf8'));
  const tokenizer = new Tokenizer(); await tokenizer.load(base + 'vocab.bin', base + 'merges.txt');
  const session = await TinyStoriesSession.open({api, graph, tokenizer, graphUrl: base + 'graph.json', weightsUrl: base + 'model.safetensors',
    wasmUrl: resolve(root, `dist/${version}/volvoxai.wasm`)}); t.after(() => session.close());
  const settings = {prompt: 'Once upon a time, Lily', requestedNewTokens: 3, seed: 42, temperature: 0, topK: 40};
  const ordinaryLogits = [];
  const normal = await session.generate(settings, {onToken: (run, record, logits) => ordinaryLogits.push(logits.slice())});
  const repeat = await session.generate(settings);
  assert.deepEqual(normal.generatedTokenIds, repeat.generatedTokenIds);
  assert.equal(normal.records.length, 3); assert.equal(normal.records[0].phase, 'prefill'); assert.equal(normal.records[1].phase, 'decode');
  assert.equal(normal.sequenceCapacity, normal.promptLength + 3);
  for (const prediction of [0, 1, 2]) {
    const debug = await session.createDebugSession(settings, {prediction, run: normal, capture: {
      tensorNames: [session.contract.outputName], inputs: false, outputs: true, values: true, constants: false}});
    try {
      assert.equal(debug.info.target, api.pb.DebugTarget.DEBUG_TARGET_EXECUTE);
      await debug.continue(); assert.equal(debug.info.state, api.pb.DebugState.DEBUG_STATE_COMPLETED);
      const state = await debug.decodeState(); assert.equal(state.slots[0].writeBegin, prediction === 0 ? 0 : normal.records[prediction].position);
      const snapshot = debug.tensors.find(value => debug.plan.tensors[value.tensorId].name === session.contract.outputName);
      assert.ok(snapshot);
      const values = new Float32Array(await (await debug.tensorBlob(snapshot)).arrayBuffer());
      const row = normal.records[prediction].position * session.contract.vocabularySize;
      for (let token = 0; token < session.contract.vocabularySize; token++)
        assert.ok(Math.abs(values[row + token] - ordinaryLogits[prediction][row + token]) <= 1e-4,
          `normal/debug logits differ at prediction ${prediction}, token ${token}`);
      for (const candidate of normal.records[prediction].candidates)
        assert.ok(Math.abs(values[row + candidate.tokenId] - candidate.logit) <= 1e-4,
          `prediction ${prediction}, token ${candidate.tokenId}: ${values[row + candidate.tokenId]} != ${candidate.logit}`);
    } finally { await debug.release(); await debug.release(); }
  }
  const measured = await session.generate(settings, {profiled: true});
  assert.deepEqual(measured.generatedTokenIds, normal.generatedTokenIds);
  assert.equal(measured.phaseRecords.length, measured.records.length);
  assert.ok(measured.phaseRecords.every(row => row.executionId !== '0' && row.modelId !== '0'));
  assert.ok(measured.trace.events.length > 0);
  const stopped = await session.generate(settings, {shouldStop: () => true});
  assert.equal(stopped.stopReason, 'stopped'); assert.equal(stopped.records.length, 0);
});
