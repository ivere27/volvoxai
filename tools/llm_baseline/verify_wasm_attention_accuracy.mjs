/** Public WASM long-key F32 attention against an independent two-pass F64 oracle.
 * Both release profiles execute the actual SIMD128/portable kernel through
 * generated dispatch. This does not time kernels or substitute a JS executor.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';

const [profile, outputPath, ...rest] = process.argv.slice(2);
assert.ok(['inference', 'full'].includes(profile) && outputPath,
  'usage: inference|full NEW_JSON [--bundle PATH --wasm PATH]');
const options = new Map();
for (let i = 0; i < rest.length; i += 2) options.set(rest[i], rest[i + 1]);
const version = JSON.parse(await fs.readFile('package.json', 'utf8')).version;
const stem = profile === 'inference' ? 'volvoxai.lite' : 'volvoxai';
const bundle = path.resolve(options.get('--bundle') ?? `dist/${version}/${stem}.js`);
const wasm = path.resolve(options.get('--wasm') ?? `dist/${version}/${stem}.wasm`);
try { await fs.stat(outputPath); throw new Error('refusing to overwrite evidence'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const artifacts = [bundle, wasm, path.resolve(process.argv[1]), path.resolve('package.json')];
const identities = async () => Object.fromEntries(await Promise.all(artifacts.map(async file => {
  const bytes = await fs.readFile(file);
  return [file, { bytes: bytes.byteLength, sha256: crypto.createHash('sha256').update(bytes).digest('hex') }];
})));
const before = await identities();
const api = await import(pathToFileURL(bundle));
const p = api.pb;
const host = new (api.FullEngineHost ?? api.EngineHost)({ wasmUrl: wasm });
const engine = new api.VxInferenceServiceClient(host);
const platform = new api.VxPlatformServiceClient(host);
const checked = packet => {
  const report = packet.report ?? packet;
  assert.equal(report.status, p.NativeStatus.NATIVE_STATUS_OK, report.message);
  return packet;
};
const route = report => {
  assert.equal(report.backend, 'wasm');
  assert.ok(report.route?.attested && report.route.builtin && report.route.activeNodes > 0);
  assert.equal(report.route.activeNodes, report.route.selectedNodes);
  assert.equal(report.route.fallbackNodes, 0);
  assert.equal(report.route.missingNodes, 0);
  assert.equal(report.fallback?.operatorFallbackUsed ?? false, false);
  return report.toJson();
};
const result = { format: 'volvox-wasm-long-attention-accuracy/v1', status: 'failed', profile,
  backend: 'wasm', authority: 'Independent two-pass JavaScript F64 dot/Math.exp softmax on immutable F32 input bytes',
  tolerance: { absolute: 2e-6, relative: 1e-6 }, qualifiesPerformance: false,
  numericalTensorCapture: true, serviceActions: 0, identitiesBefore: before, cases: [] };
const matrix = [
  { name: 'dense-long-SIMD', queries: 7, keys: 1537, heads: 2, kvHeads: 2, width: 64, mask: 'none' },
  { name: 'dense-4097-SIMD', queries: 3, keys: 4097, heads: 2, kvHeads: 2, width: 64, mask: 'none' },
  { name: 'key-mask-channel-tail', queries: 7, keys: 513, heads: 2, kvHeads: 2, width: 37, mask: 'K' },
  { name: 'query-mask-all-masked-tail', queries: 7, keys: 513, heads: 2, kvHeads: 2, width: 37, mask: 'QK' },
  { name: 'grouped-long-mask', queries: 7, keys: 1537, heads: 4, kvHeads: 2, width: 80, mask: 'QK' },
  { name: 'short-48-boundary', queries: 3, keys: 48, heads: 2, kvHeads: 2, width: 37, mask: 'QK' },
  { name: 'first-partial-49-boundary', queries: 3, keys: 49, heads: 4, kvHeads: 2, width: 37, mask: 'QK' },
];

function inputs(spec) {
  let state = 710913;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return Math.fround(Math.fround((state >>> 8) / 16777216 - .5) * Math.fround(.3));
  };
  const q = Float32Array.from({ length: spec.queries * spec.heads * spec.width }, random);
  const k = Float32Array.from({ length: spec.keys * spec.kvHeads * spec.width }, random);
  const v = new Float32Array(k.length);
  for (let key = 0; key < spec.keys; key++) for (let head = 0; head < spec.kvHeads; head++)
    for (let channel = 0; channel < spec.width; channel++)
      v[(key * spec.kvHeads + head) * spec.width + channel] =
        Math.fround(100 + .25 * head + .01337 * (channel % 11) + .001 * (key % 7 - 3));
  const mask = spec.mask === 'none' ? null : Int32Array.from(
    { length: (spec.mask === 'K' ? 1 : spec.queries) * spec.keys }, (_, i) =>
      (spec.mask !== 'QK' || Math.floor(i / spec.keys) !== 1) && i % spec.keys % 7 !== 0 ? 1 : 0);
  return { q, k, v, mask };
}

function reference(spec, data) {
  const { queries: Q, keys: K, heads: H, kvHeads: KV, width: D } = spec;
  const output = new Float64Array(Q * H * D);
  const scores = new Float64Array(K);
  for (let row = 0; row < Q; row++) for (let head = 0; head < H; head++) {
    const kvHead = Math.floor(head / (H / KV));
    let maximum = -Infinity;
    for (let key = 0; key < K; key++) {
      let score = 0;
      for (let d = 0; d < D; d++) score += data.q[(row * H + head) * D + d] * data.k[(key * KV + kvHead) * D + d];
      if (data.mask && !data.mask[(spec.mask === 'K' ? 0 : row * K) + key]) score = -Infinity;
      scores[key] = score;
      maximum = Math.max(maximum, score);
    }
    if (maximum === -Infinity) continue; // Independently specified all-masked zero.
    let total = 0;
    for (let key = 0; key < K; key++) { scores[key] = Math.exp(scores[key] - maximum); total += scores[key]; }
    for (let d = 0; d < D; d++) {
      let sum = 0;
      for (let key = 0; key < K; key++) sum += scores[key] * data.v[(key * KV + kvHead) * D + d];
      output[(row * H + head) * D + d] = sum / total;
    }
  }
  return output;
}

function reversed(spec, data) {
  const reverseRows = (array, width) => {
    const output = new array.constructor(array.length);
    for (let key = 0; key < spec.keys; key++) output.set(array.subarray(key * width, (key + 1) * width),
      (spec.keys - 1 - key) * width);
    return output;
  };
  const mask = data.mask ? new Int32Array(data.mask.length) : null;
  if (mask) for (let row = 0; row < (spec.mask === 'K' ? 1 : spec.queries); row++)
    mask.set(reverseRows(data.mask.subarray(row * spec.keys, (row + 1) * spec.keys), 1), row * spec.keys);
  return { q: data.q, k: reverseRows(data.k, spec.kvHeads * spec.width),
    v: reverseRows(data.v, spec.kvHeads * spec.width), mask };
}

function compare(actual, expected) {
  assert.equal(actual.length, expected.length);
  let maxAbsolute = 0, squared = 0, failingElements = 0;
  for (let i = 0; i < actual.length; i++) {
    const delta = Math.abs(actual[i] - expected[i]);
    assert.ok(Number.isFinite(actual[i]), `nonfinite attention element${i}`);
    maxAbsolute = Math.max(maxAbsolute, delta);
    squared += delta * delta;
    if (delta > result.tolerance.absolute + result.tolerance.relative * Math.abs(expected[i])) failingElements++;
  }
  return { elements: actual.length, maxAbsolute, rms: Math.sqrt(squared / actual.length), failingElements,
    pass: failingElements === 0 };
}

let runtime;
let failure;
try {
  const info = await platform.getPlatformInfo(new p.Empty());
  assert.equal(info.buildProfile, profile === 'full' ? p.BuildProfile.BUILD_PROFILE_FULL : p.BuildProfile.BUILD_PROFILE_INFERENCE);
  result.platform = info.toJson();
  runtime = checked(await engine.createRuntime(new p.CreateRuntimeRequest({ cpuThreads: 1 })));
  for (const spec of matrix) {
    const inputsSpec = { q: { dtype: 'float32', shape: [1, spec.queries, spec.heads * spec.width] },
      k: { dtype: 'float32', shape: [1, spec.keys, spec.kvHeads * spec.width] },
      v: { dtype: 'float32', shape: [1, spec.keys, spec.kvHeads * spec.width] } };
    if (spec.mask !== 'none') inputsSpec.keep = { dtype: 'int32',
      shape: spec.mask === 'K' ? [spec.keys] : [spec.queries, spec.keys] };
    const graph = { format: 'volvox-graph/v1', dimensions: {}, inputs: inputsSpec,
      nodes: [{ id: 'attention', opType: 'CrossSDPA', inputs: { q: 'q', k: 'k', v: 'v',
        ...(spec.mask !== 'none' ? { mask: 'keep' } : {}) }, params: { heads: spec.heads,
        kv_heads: spec.kvHeads, causal: false, scale: 1 }, outputs: { out: { tensor: 'attention',
          dtype: 'float32', shape: [1, spec.queries, spec.heads * spec.width] } } }], outputs: ['attention'],
      execution: { stateless: {}, entrypoints: [{ name: 'evaluate', effect: 1, allowed_ranges: [1],
        inputs: Object.keys(inputsSpec), outputs: ['attention'] }] } };
    const model = checked(await engine.loadModel(new p.LoadModelRequest({ runtimeId: runtime.runtimeId,
      package: new p.ModelPackage({ graphDocument: new TextEncoder().encode(JSON.stringify(graph)) }) })));
    let compiled, context;
    const retained = [];
    try {
      compiled = checked(await engine.compileModel(new p.CompileModelRequest({ modelId: model.modelId,
        policy: new p.BackendPolicy({ mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
          backends: ['wasm'], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })));
      const compileRoute = route(compiled.report);
      context = checked(await engine.createExecutionContext(new p.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph: new p.GraphExecution() })));
      const original = inputs(spec);
      const expected = reference(spec, original);
      for (const [order, data] of [['original', original], ['reversed', reversed(spec, original)]]) {
        const oracle = order === 'original' ? expected : reference(spec, data);
        const bindings = ['q', 'k', 'v'].map(name => new p.Tensor({ name,
          dtype: p.DataType.DATA_TYPE_F32, shape: inputsSpec[name].shape.map(BigInt), inline: new Uint8Array(data[name].buffer) }));
        if (data.mask) bindings.push(new p.Tensor({ name: 'keep', dtype: p.DataType.DATA_TYPE_I32,
          shape: inputsSpec.keep.shape.map(BigInt), inline: new Uint8Array(data.mask.buffer) }));
        const packet = checked(await engine.execute(new p.ExecuteRequest({ contextId: context.contextId,
          inputs: new p.InputBindings({ values: bindings }), scope: new p.ExecutionScope({ entrypoint: 'evaluate', full: new p.Empty() }),
          outputs: new p.OutputOptions({ storage: p.OutputStorage.OUTPUT_STORAGE_INLINE }) })));
        const executeRoute = route(packet.report);
        assert.equal(packet.outputs.length, 1);
        const output = packet.outputs[0];
        assert.equal(output.name, 'attention');
        assert.equal(output.dtype, p.DataType.DATA_TYPE_F32);
        assert.equal(output.buffer, undefined);
        assert.deepEqual(output.shape.map(Number), [1, spec.queries, spec.heads * spec.width]);
        assert.equal(output.inline.byteLength, oracle.length * 4);
        const actual = new Float32Array(output.inline.slice().buffer);
        const comparison = compare(actual, oracle);
        const permutationOracle = compare(oracle, expected);
        const row = { ...spec, order, comparison, permutationOracle, compileRoute, executeRoute,
          inputSHA256: Object.fromEntries(['q', 'k', 'v', ...(data.mask ? ['mask'] : [])].map(name =>
            [name, crypto.createHash('sha256').update(new Uint8Array(data[name].buffer)).digest('hex')])) };
        result.cases.push(row);
        retained.push({ output, bytes: output.inline.slice() });
        assert.ok(comparison.pass, `${spec.name}/${order}: ${JSON.stringify(comparison)}`);
        assert.ok(permutationOracle.pass, 'independent attention must preserve key/value/mask permutation');
        if (spec.mask === 'QK') assert.ok(actual.subarray(spec.heads * spec.width, 2 * spec.heads * spec.width).every(value => value === 0),
          'all-masked query must be exactly zero');
        console.log(`PASS ${profile} ${spec.name}/${order} max_abs=${comparison.maxAbsolute}`);
      }
    } finally {
      if (context) checked(await engine.releaseExecutionContext(new p.ExecutionContextRef({ contextId: context.contextId })));
      if (compiled) checked(await engine.releaseCompiledModel(new p.CompiledModelRef({ compiledModelId: compiled.compiledModelId })));
      checked(await engine.releaseModel(new p.ModelRef({ modelId: model.modelId })));
      for (const { output, bytes } of retained) assert.deepEqual(output.inline, bytes, 'INLINE snapshot mutated after owner retirement');
    }
  }
  result.status = 'pass';
  result.retainedInlineAfterOwnerRetirement = true;
} catch (error) {
  failure = error;
  result.error = { message: error.message, stack: error.stack };
} finally {
  if (runtime) checked(await engine.releaseRuntime(new p.RuntimeRef({ runtimeId: runtime.runtimeId })));
  await host.close();
  result.identitiesAfter = await identities();
  assert.deepEqual(result.identitiesAfter, before, 'artifact changed during qualification');
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
}
if (failure) throw failure;
