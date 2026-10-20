import assert from 'node:assert/strict';
import test from 'node:test';
import * as pb from '../../../runtime/generated/typescript/volvoxai_lite.js';
import {TinyReceiptSession, chooseReceiptToken} from '../TinyReceiptSession.js';
import {TinyReceiptByteFallbackBPEVocab} from '../TinyReceiptByteFallbackBPEVocab.js';

const tensor = (data, shape, dtype) => ({data, shape, dtype});
const geo = {image: [1, 1, 1, 2], decoder_layers: 1, attention_heads: 1, attention_head_width: 1};
const encInputs = ['image', 'question_ids', 'question_position_ids', 'family_ids'];
const encOutputs = ['memory_padding_mask', 'selected_family_ids', 'cross_k_0', 'cross_v_0'];
const decInputs = ['decoder_input_ids', 'position_ids', 'family_ids', 'memory_padding_mask', 'past_padding_mask', 'cross_k_0', 'cross_v_0', 'past_k_0', 'past_v_0'];
const decOutputs = ['logits', 'present_padding_mask', 'present_k_0', 'present_v_0'];
const mapping = names => Object.fromEntries(names.map(name => [name, name]));
const manifest = {format: 'volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2',
  shape_contract: {fixed_geometry: geo, dimensions: {Q: {min: 1, max: 4}, P: {min: 1, max: 4}, R: {min: 2, max: 5}}},
  cache_contract: {format: 'masked-zero-sentinel-v1', initial_past_length: 1, sentinel_mask_value: 1},
  generation: {maximum_new_tokens: 4}, families: {ordered_names: ['phone']}, tokenizer: {token_ids: {bos: 1, eos: 2}},
  graphs: {encoder: {graph: {path: 'encoder.json'}, weights: {path: 'encoder.safetensors'}, inputs: mapping(encInputs), outputs: mapping(encOutputs)},
    decoder: {graph: {path: 'decoder.json'}, weights: {path: 'decoder.safetensors'}, inputs: mapping(decInputs), outputs: mapping(decOutputs)}}};
const vocab = {bos: 1, eos: 2, itos: ['<pad>', '<bos>', '<eos>', '<unk>', '<answer>', '7', '</answer>', 'Q'],
  encodeQuestion: () => [7, 2], decode: ids => ids.filter(id => id > 2).map(id => vocab.itos[id]).join('')};
const raw = {data: Uint8Array.of(0, 255), width: 2, height: 1, channels: 1};

function fixture() {
  const requests = [], contexts = new Map(), compiled = new Map(), retainedInputs = new Map();
  let next = 10n, clock = 0n;
  const report = (compilation = false) => ({status: 0, backend: 'wasm', route: {provider: 'wasm', attested: true,
    activeNodes: 1, selectedNodes: 1, fallbackNodes: 0, missingNodes: 0}, fallback: {operatorFallbackUsed: false},
    ...(compilation ? {compilation: {policyMode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID,
      candidates: [{outcome: pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED, backend: 'wasm'}]}} : {})});
  const reply = values => ({report: report(), ...values});
  const read = value => new (value.dtype === pb.DataType.DATA_TYPE_F32 ? Float32Array : Int32Array)(value.inline.slice().buffer);
  class Host { async close() { requests.push({method: 'host.close'}); } }
  class Inference {
    async createRuntime(request) { requests.push({method: 'createRuntime', request}); return reply({runtimeId: 1n}); }
    async loadModel(request) { const role = request.graphPath.includes('encoder') ? 'encoder' : 'decoder';
      return reply({modelId: role === 'encoder' ? 2n : 3n}); }
    async compileModel(request) { const id = next++; compiled.set(id, request.modelId === 2n ? 'encoder' : 'decoder');
      requests.push({method: 'compile', request}); return {compiledModelId: id, report: report(true)}; }
    async createExecutionContext(request) { const id = next++; contexts.set(id, compiled.get(request.compiledModelId));
      requests.push({method: 'context', id, request}); return reply({contextId: id}); }
    async execute(request) {
      const role = contexts.get(request.contextId), feed = Object.fromEntries(request.inputs.values.map(input => [input.name, {data: read(input), shape: input.shape.map(Number)}]));
      for (const name of request.inputs.reuse) feed[name] = retainedInputs.get(request.contextId)[name];
      retainedInputs.set(request.contextId, feed);
      requests.push({method: 'execute', role, contextId: request.contextId, feed});
      if (fixture.refuseRole === role) return {report: {status: 1, message: 'Refused execution'}};
      let output;
      if (role === 'encoder') output = {memory_padding_mask: tensor(new Int32Array(212), [1, 212], 'int32'), selected_family_ids: tensor(Int32Array.of(0), [1], 'int32'),
        cross_k_0: tensor(new Float32Array(212), [1, 1, 212, 1], 'float32'), cross_v_0: tensor(new Float32Array(212), [1, 1, 212, 1], 'float32')};
      else {
        const position = feed.position_ids.data[0], length = feed.past_padding_mask.shape[1], logits = new Float32Array(8).fill(-10); logits[[4, 5, 6, 2][position]] = 10;
        output = {logits: tensor(logits, [1, 1, 8], 'float32'), present_padding_mask: tensor(Int32Array.from([...feed.past_padding_mask.data, 0]), [1, length + 1], 'int32')};
        for (const kind of ['k', 'v']) output[`present_${kind}_0`] = tensor(Float32Array.from([...feed[`past_${kind}_0`].data, feed.decoder_input_ids.data[0]]), [1, 1, length + 1, 1], 'float32');
      }
      const id = next++; return reply({executionId: id + 100n, outputs: Object.entries(output).map(([name, value]) => ({
        name, shape: value.shape.map(BigInt), dtype: value.dtype === 'float32' ? pb.DataType.DATA_TYPE_F32 : pb.DataType.DATA_TYPE_I32,
        inline: new Uint8Array(value.data.buffer)}))});
    }
    async releaseCompiledModel(request) { requests.push({method: 'releaseCompiled', request}); return report(); }
    async releaseExecutionContext(request) { requests.push({method: 'releaseContext', request}); return report(); }
  }
  class Debug {
    async createDebugSession(request) { requests.push({method: 'debug', request}); return reply({debugSessionId: 99n}); }
    async getDebugPlan() { return reply({plan: {steps: []}}); }
    async releaseDebugSession(request) { requests.push({method: 'releaseDebug', request}); return report(); }
  }
  class Platform { async getMonotonicTime() { clock += 1000n; return {nanoseconds: clock}; } }
  class Profiling {
    async startTrace(request) { requests.push({method: 'trace', request}); return reply({traceId: 88n}); }
    async annotateTrace(request) { requests.push({method: 'annotate', request}); return report(); }
    async stopTrace() { return reply({plans: {count: 0n}}); }
    async listTraceEvents() { return reply({events: [], nextPageToken: ''}); }
    async listTraceResourceSnapshots() { return reply({resourceSnapshots: [], nextPageToken: ''}); }
    async exportChromeTrace() { return reply({data: new TextEncoder().encode('{}'), nextPageToken: ''}); }
    async releaseTrace() { requests.push({method: 'releaseTrace'}); return report(); }
  }
  const api = {pb, FullEngineHost: Host, VxInferenceServiceClient: Inference, VxDebugServiceClient: Debug, VxPlatformServiceClient: Platform, VxProfilingServiceClient: Profiling};
  const fetch = async url => ({ok: true, json: async () => {
    if (String(url).includes('package_manifest')) return manifest;
    const encoder = String(url).includes('encoder'), inputs = encoder ? encInputs : decInputs, outputs = encoder ? encOutputs : decOutputs;
    return {nodes: [], execution: {entrypoints: (encoder ? [['evaluate', 1]] : [['initialize', 2], ['transition', 3]])
      .map(([name, effect]) => ({name, effect, inputs, outputs})), invariant_inputs: encoder ? [] : ['family_ids', 'memory_padding_mask', 'cross_k_0', 'cross_v_0']}};
  }});
  return {api, fetch, requests};
}
async function open(t) {
  const f = fixture(), original = TinyReceiptByteFallbackBPEVocab.fromJSON;
  TinyReceiptByteFallbackBPEVocab.fromJSON = async () => vocab;
  t.after(() => { TinyReceiptByteFallbackBPEVocab.fromJSON = original; fixture.refuseRole = null; });
  const session = await TinyReceiptSession.open({...f, packageUrl: 'https://fixture/model/'}); t.after(() => session.close()); return {...f, session};
}

test('split generation carries real explicit cache prefixes, sentinel and chosen input tokens', async t => {
  const f = await open(t), output = await f.session.run(raw, 'question', {maxTokens: 4});
  assert.deepEqual(output.tokens, [4, 5, 6, 2]); assert.equal(output.parsed.answer, '7'); assert.equal(output.reachedEos, true);
  const calls = f.requests.filter(item => item.method === 'execute' && item.role === 'decoder');
  assert.deepEqual(calls.map(item => [...item.feed.decoder_input_ids.data]), [[1], [4], [5], [6]]);
  assert.deepEqual(calls.map(item => [...item.feed.past_k_0.data]), [[0], [0, 1], [0, 1, 4], [0, 1, 4, 5]]);
  assert.deepEqual(calls.map(item => [...item.feed.past_padding_mask.data]), [[1], [1, 0], [1, 0, 0], [1, 0, 0, 0]]);
  assert.equal(new Set(calls.map(item => item.contextId)).size, 1);
});
test('debug selected decode call replays an independent prefix and forwards capture/limits/boundaries', async t => {
  const f = await open(t); await f.session.run(raw, 'question', {maxTokens: 4}); const count = f.requests.length;
  const debug = await f.session.createDebugSession(raw, 'question', {target: 'decode', position: 2,
    preserveNodeBoundaries: true, capture: {values: true, tensorNames: ['past_k_0']}, limits: {maxBytes: 1048576n}});
  const request = f.requests.findLast(item => item.method === 'debug').request;
  const input = Object.fromEntries(request.execute.inputs.values.map(value => [value.name, value]));
  assert.deepEqual([...new Float32Array(input.past_k_0.inline.slice().buffer)], [0, 1, 4]);
  assert.deepEqual([...new Int32Array(input.decoder_input_ids.inline.slice().buffer)], [5]);
  assert.equal(request.capture.values, true); assert.equal(request.limits.maxBytes, 1048576n);
  assert.equal(f.requests.findLast(item => item.method === 'compile').request.preserveNodeBoundaries, true);
  assert.equal(debug.sourceDetails.position, 2); assert.equal(debug.sourceDetails.pastLength, 3);
  assert.equal(f.requests.slice(count).filter(item => item.method === 'releaseContext').length, 2);
  for (const call of f.requests.slice(count).filter(item => item.method === 'execute')) assert.notEqual(call.contextId, f.session.contexts[call.role]);
  await debug.release(); const again = await f.session.run(raw, 'question', {maxTokens: 4}); assert.deepEqual(again.tokens, [4, 5, 6, 2]);
});
test('debug prefill prepares BOS/sentinel; replay failure releases both independent contexts', async t => {
  const f = await open(t), debug = await f.session.createDebugSession(raw, 'question', {target: 'prefill', preserveNodeBoundaries: false});
  assert.equal(debug.sourceDetails.inputToken, 1); assert.equal(debug.sourceDetails.pastLength, 1); await debug.release();
  const before = f.requests.length; fixture.refuseRole = 'encoder';
  await assert.rejects(f.session.createDebugSession(raw, 'question', {target: 'prefill'}), /Refused execution/);
  assert.equal(f.requests.slice(before).filter(item => item.method === 'releaseContext').length, 2);
});
test('selected decode refuses an input different from the retained generated prefix', async t => {
  const f = await open(t); await f.session.run(raw, 'question', {maxTokens: 4});
  await assert.rejects(f.session.createDebugSession({...raw, data: Uint8Array.of(255, 0)}, 'question', {target: 'decode', position: 1}), /inputs changed/);
  f.session.clearGeneration(); await assert.rejects(f.session.createDebugSession(raw, 'question', {target: 'decode', position: 1}), /reached decode position/);
});
test('explicit trace spans encoder, prefill and each decode invocation with engine clock annotations', async t => {
  const f = await open(t), output = await f.session.run(raw, 'question', {maxTokens: 4, profile: true});
  assert.deepEqual(output.phases.map(item => item.phase), ['encoder', 'prefill', 'decode', 'decode', 'decode']);
  assert.equal(f.requests.filter(item => item.method === 'annotate').length, 5);
  assert.equal(f.requests.filter(item => item.method === 'releaseTrace').length, 1);
  assert.ok(output.trace.blob instanceof Blob); assert.ok(output.phases.every(item => item.startNs < item.endNs));
  assert.deepEqual(f.session.phaseForProfile({lineage: {executionId: output.phases[3].executionId, modelId: f.session.models.decoder}}), {target: 'decode', position: 2});
});
test('token policy preserves greedy ties, seeded draw semantics and refuses nonfinite logits', () => {
  assert.equal(chooseReceiptToken(Float32Array.of(2, 2, 1)), 0);
  assert.equal(chooseReceiptToken(Float32Array.of(2, 2, 1), {sampling: 'sample', temperature: 1, topK: 1}, () => .9), 0);
  assert.throws(() => chooseReceiptToken(Float32Array.of(NaN)), /finite/);
  assert.throws(() => chooseReceiptToken(Float32Array.of(1, 2), {sampling: 'sample', temperature: 0}), /Invalid sampling/);
});
