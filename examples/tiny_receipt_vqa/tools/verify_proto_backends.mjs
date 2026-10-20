/** Verify split VQA explicit KV inference using the actual release bundles. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const options = { backend: 'wasm', profile: 'inference', limit: '0', variants: 'fp32,int8,ptq,ptq-wasm' };
for (let index = 2; index < process.argv.length; index += 2) {
  const name = process.argv[index].replace(/^--/, '');
  assert.ok(['work', 'out', 'backend', 'profile', 'limit', 'variants', 'baseline-capture'].includes(name));
  assert.ok(process.argv[index + 1]);
  options[name] = process.argv[index + 1];
}
assert.ok(options.work && options.out);
assert.ok(['wasm', 'webgpu'].includes(options.backend));
assert.ok(['inference', 'full'].includes(options.profile));
assert.ok(options.backend !== 'webgpu' || options.profile === 'full');
assert.ok(Number.isInteger(Number(options.limit)) && Number(options.limit) >= 0);
const work = path.resolve(options.work), out = path.resolve(options.out);
const version = JSON.parse(readFileSync(path.join(root, 'package.json'))).version;
const stem = options.profile === 'full' ? 'volvoxai' : 'volvoxai.lite';
const bundle = path.join(root, 'dist', version, `${stem}.js`);
const wasm = path.join(root, 'dist', version, `${stem}.wasm`);
const api = await import(pathToFileURL(bundle));
const { pb, VxInferenceServiceClient } = api;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = value => JSON.stringify(value, (_, item) => typeof item === 'bigint' ? item.toString() : item, 2);
const plain = value => JSON.parse(json(value));
const readJson = filename => JSON.parse(readFileSync(filename));
const corpusBytes = readFileSync(path.join(work, 'corpus/corpus.json'));
const corpus = JSON.parse(corpusBytes);
const images = readFileSync(path.join(work, 'corpus/images.u8'));
assert.equal(hash(images), corpus.images_sha256);
assert.equal(images.length, corpus.evaluation.length * 320 * 672);
const count = Math.min(Number(options.limit) || corpus.evaluation.length, corpus.evaluation.length);
const cases = readJson(path.join(work, 'fixtures/cases.json'));
const report = { schema: 'volvoxai.vqa-proto-web-verification/v1', backend: options.backend,
  profile: options.profile, samples: count, corpus_sha256: hash(corpusBytes),
  runtime: globalThis.Deno?.version ?? { node: process.version },
  artifacts: { [path.basename(bundle)]: hash(readFileSync(bundle)), [path.basename(wasm)]: hash(readFileSync(wasm)) }, variants: {} };
const baselineDirectory = options['baseline-capture'] ? path.resolve(options['baseline-capture']) : null;
const baseline = baselineDirectory ? readJson(path.join(baselineDirectory, 'report.json')) : null;
if (baseline) {
  assert.equal(baseline.backend, report.backend);
  assert.equal(baseline.profile, report.profile);
  assert.equal(baseline.corpus_sha256, report.corpus_sha256);
}
report.baseline_equivalence = {status: baseline ? 'pending' : 'not_checked',
  ...(baseline ? {report_sha256: hash(readFileSync(path.join(baselineDirectory, 'report.json'))),
    float_atol: 1e-4, float_rtol: 1e-4} : {})};
mkdirSync(path.dirname(out), { recursive: true });
let adapterInfo;
if (options.backend === 'webgpu') {
  const gpu = globalThis.navigator?.gpu;
  assert.ok(gpu, 'WebGPU is unavailable');
  const requestAdapter = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (...args) => {
    const adapter = await requestAdapter(...args);
    assert.ok(adapter, 'no physical WebGPU adapter');
    adapterInfo = Object.fromEntries(['vendor', 'architecture', 'device', 'description']
      .map(name => [name, String(adapter.info?.[name] ?? '')]));
    const identity = Object.values(adapterInfo).join(' ');
    assert.match(identity, /nvidia|3090/i);
    assert.doesNotMatch(identity, /swiftshader|lavapipe|llvmpipe|software/i);
    return adapter;
  };
}
function attest(value) {
  assert.equal(value.backend, options.backend);
  assert.equal(value.route?.provider, options.backend);
  assert.equal(value.route?.attested, true);
  assert.equal(value.route.activeNodes, value.route.selectedNodes);
  assert.equal(value.route.fallbackNodes, 0);
  assert.equal(value.route.missingNodes, 0);
  assert.ok(!value.fallback?.operatorFallbackUsed);
  assert.ok(!value.compilation?.tierFallbackUsed);
  if (options.backend === 'webgpu') assert.ok(adapterInfo);
}
const i32 = (values, shape) => ({ data: Int32Array.from(values), shape, dtype: 'int32' });
const f32 = (values, shape) => ({ data: Float32Array.from(values), shape, dtype: 'float32' });
function readTensors(filename) {
  const bytes = readFileSync(`${filename}.bin`);
  return Object.fromEntries(Object.entries(readJson(`${filename}.json`)).map(([name, item]) => {
    const data = Uint8Array.from(bytes.subarray(item.offset, item.offset + item.bytes));
    return [name, { shape: item.shape, dtype: item.dtype,
      data: item.dtype === 'float32' ? new Float32Array(data.buffer) : new Int32Array(data.buffer) }];
  }));
}
function compare(actual, expected) {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort());
  const result = {};
  for (const [name, value] of Object.entries(actual)) {
    const reference = expected[name];
    assert.deepEqual(value.shape, reference.shape, name);
    assert.equal(value.dtype, reference.dtype, name);
    let maxAbs = 0, sumSquare = 0, exact = true, close = true;
    for (let index = 0; index < value.data.length; index++) {
      const x = value.data[index], y = reference.data[index], delta = Math.abs(x - y);
      assert.ok(Number.isFinite(x) && Number.isFinite(y), name);
      maxAbs = Math.max(maxAbs, delta); sumSquare += delta * delta;
      exact &&= x === y; close &&= delta <= 1e-4 + 1e-4 * Math.abs(y);
    }
    result[name] = { max_abs: maxAbs, rmse: Math.sqrt(sumSquare / value.data.length), exact, close_1e_4: close };
    if (name === 'logits') result[name].argmax_equal = argmax(value.data) === argmax(reference.data);
  }
  return result;
}
function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index++) if (values[index] > values[best]) best = index;
  return best;
}
function decoderFeed(encoded) {
  const feed = Object.fromEntries(Object.entries(encoded).filter(([name]) => name.startsWith('cross_') || name === 'memory_padding_mask'));
  Object.assign(feed, { decoder_input_ids: i32([1], [1, 1]), position_ids: i32([0], [1]),
    family_ids: encoded.selected_family_ids, past_padding_mask: i32([1], [1, 1]) });
  for (let layer = 0; layer < 4; layer++) for (const kind of ['k', 'v']) feed[`past_${kind}_${layer}`] = f32(new Float32Array(320), [1, 8, 1, 40]);
  return feed;
}
function advance(feed, output, token, position, checkPrefix = true) {
  const past = feed.past_padding_mask.shape[1];
  assert.deepEqual(output.present_padding_mask.shape, [1, past + 1]);
  assert.deepEqual(output.present_padding_mask.data.subarray(0, past), feed.past_padding_mask.data);
  assert.equal(output.present_padding_mask.data[past], 0);
  for (let layer = 0; layer < 4; layer++) for (const kind of ['k', 'v']) {
    const before = `past_${kind}_${layer}`, after = `present_${kind}_${layer}`;
    assert.deepEqual(output[after].shape, [1, 8, past + 1, 40]);
    for (let head = 0; checkPrefix && head < 8; head++) {
      assert.deepEqual(output[after].data.subarray(head * (past + 1) * 40, (head * (past + 1) + past) * 40),
        feed[before].data.subarray(head * past * 40, (head + 1) * past * 40), `cache prefix changed: ${after}`);
    }
    feed[before] = output[after];
  }
  feed.past_padding_mask = output.present_padding_mask;
  feed.decoder_input_ids = i32([token], [1, 1]);
  feed.position_ids = i32([position + 1], [1]);
}

for (const variant of options.variants.split(',')) {
  const entry = report.variants[variant] = { status: 'failed', stage: 'load', packages: {}, routes: {}, executions: { encoder: 0, decoder: 0 } };
  let host;
  try {
    adapterInfo = undefined;
    const packagePath = path.join(work, variant);
    const manifest = readJson(path.join(packagePath, 'package_manifest.json'));
    const Constructor = options.profile === 'full' ? api.FullEngineHost : api.EngineHost;
    host = new Constructor({ wasmUrl: pathToFileURL(wasm), onDiagnostic: message => {
      (entry.diagnostics ??= []).push(message);
      console.log(message);
    } });
    const client = new VxInferenceServiceClient(host);
    const runtime = await client.createRuntime(new pb.CreateRuntimeRequest({ cpuThreads: 1 }));
    const contexts = {}, compiledIds = {}, executionContracts = {};
    let decoderInitialized = false;
    for (const role of ['encoder', 'decoder']) {
      const assets = manifest.graphs[role];
      const graph = readFileSync(path.join(packagePath, assets.graph.path));
      const weights = readFileSync(path.join(packagePath, assets.weights.path));
      executionContracts[role] = JSON.parse(graph).execution;
      entry.packages[role] = { 'graph.json': hash(graph), 'model.safetensors': hash(weights) };
      const model = await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
        package: new pb.ModelPackage({ graphDocument: graph, weightShards: [weights] }) }));
      entry.stage = `compile ${role}`;
      const compiled = await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
        policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
          backends: [options.backend], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) }));
      attest(compiled.report);
      entry.routes[role] = { compilation: plain(compiled.report) };
      compiledIds[role] = compiled.compiledModelId;
      contexts[role] = (await client.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution() }))).contextId;
    }
    if (adapterInfo) entry.adapter = adapterInfo;
    async function newDecoder() {
      await client.releaseExecutionContext(new pb.ExecutionContextRef({contextId: contexts.decoder}));
      contexts.decoder = (await client.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiledIds.decoder, graph: new pb.GraphExecution()}))).contextId;
      decoderInitialized = false;
    }
    async function execute(role, feed) {
      const contract = manifest.graphs[role], executionContract = executionContracts[role];
      assert.deepEqual(Object.keys(feed).sort(), Object.keys(contract.inputs).sort());
      const effect = role === 'encoder' ? 1 : decoderInitialized ? 3 : 2;
      const entrypoint = executionContract.entrypoints.find(entry => entry.effect === effect)?.name;
      assert.ok(entrypoint);
      const reuse = effect === 3 ? executionContract.invariant_inputs ?? [] : [];
      const values = Object.entries(feed).map(([name, value]) => new pb.Tensor({ name: contract.inputs[name],
        shape: value.shape.map(BigInt), dtype: value.dtype === 'float32' ? pb.DataType.DATA_TYPE_F32 : pb.DataType.DATA_TYPE_I32,
        inline: new Uint8Array(value.data.buffer, value.data.byteOffset, value.data.byteLength) })).filter(value => !reuse.includes(value.name));
      const execution = await client.execute(new pb.ExecuteRequest({contextId: contexts[role],
        inputs: new pb.InputBindings({values, reuse}), scope: new pb.ExecutionScope({entrypoint, full: new pb.Empty()})}));
      if (role === 'decoder') decoderInitialized = true;
      attest(execution.report);
      entry.routes[role].execution ??= plain(execution.report);
      const result = {};
      for (const [semantic, name] of Object.entries(contract.outputs)) {
        const tensor = execution.outputs.find(tensor => tensor.name === name);
        assert.ok(tensor && [pb.DataType.DATA_TYPE_F32, pb.DataType.DATA_TYPE_I32].includes(tensor.dtype));
        const isFloat = tensor.dtype === pb.DataType.DATA_TYPE_F32;
        const bytes = Uint8Array.from(tensor.inline);
        const data = isFloat ? new Float32Array(bytes.buffer) : new Int32Array(bytes.buffer);
        assert.ok(data.every(Number.isFinite), name);
        result[semantic] = {shape: tensor.shape.map(Number), dtype: isFloat ? 'float32' : 'int32', data};
      }
      if (baseline) {
        const index = entry.executions.encoder + entry.executions.decoder;
        const captured = baseline.variants[variant]?.captures?.[index];
        assert.ok(captured, `${variant}: missing baseline capture ${index}`);
        assert.equal(captured.role, role, `${variant}: capture ${index} role`);
        assert.equal(captured.stage, entry.stage, `${variant}: capture ${index} stage`);
        const comparison = compare(result, readTensors(path.join(baselineDirectory, captured.output_stem)));
        for (const [name, metric] of Object.entries(comparison)) {
          assert.ok(result[name].dtype === 'float32' ? metric.close_1e_4 : metric.exact,
            `${variant}: capture ${index} ${name} differs from baseline: ${json(metric)}`);
        }
      }
      entry.executions[role]++;
      return result;
    }
    entry.stage = 'component fixtures'; entry.fixtures = {};
    const referenceVariant = ['fp32', 'int8'].includes(variant) ? variant : 'fp32';
    for (const fixture of cases) for (const role of ['encoder', 'decoder']) {
      const folder = path.join(work, 'fixtures', referenceVariant, fixture.id);
      const feed = readTensors(path.join(folder, `${role}-inputs`));
      if (role === 'decoder') await newDecoder();
      const output = await execute(role, feed);
      entry.fixtures[`${role}-${fixture.id}`] = compare(output, readTensors(path.join(folder, `${role}-outputs`)));
      if (role === 'decoder') advance(feed, output, 1, fixture.P - 1, variant !== 'int8');
      console.log(`${options.backend}/${variant}: ${role} fixture ${fixture.id} complete`);
      writeFileSync(out, json(report) + '\n');
    }
    entry.stage = 'autoregressive'; entry.predictions = [];
    const started = performance.now();
    for (let index = 0; index < count; index++) {
      const pixels = new Float32Array(320 * 672), offset = index * pixels.length;
      for (let i = 0; i < pixels.length; i++) pixels[i] = Math.fround(Math.fround(images[offset + i] / 255) - .5) / .5;
      const ids = corpus.evaluation[index].question_ids;
      const encoded = await execute('encoder', { image: f32(pixels, [1, 1, 320, 672]),
        question_ids: i32(ids, [1, ids.length]), question_position_ids: i32(ids.map((_, i) => i), [1, ids.length]), family_ids: i32([-1], [1]) });
      await newDecoder();
      const feed = decoderFeed(encoded), tokens = [1];
      const expected = baseline?.variants[variant]?.predictions?.[index];
      if (baseline) {
        assert.ok(expected, `${variant}: missing baseline answer ${index}`);
        assert.equal(encoded.selected_family_ids.data[0], expected.family);
      }
      for (let position = 0; position < 191; position++) {
        if (expected) assert.equal(tokens[position], expected.tokens[position],
          `${variant}: answer ${index} input token ${position}`);
        const output = await execute('decoder', feed);
        assert.deepEqual(output.logits.shape, [1, 1, 1536]);
        const token = argmax(output.logits.data);
        if (expected) {
          assert.equal(token, expected.tokens[position + 1], `${variant}: answer ${index} token ${position + 1}`);
          assert.equal(token === 2, expected.tokens[position + 1] === 2, `${variant}: answer ${index} EOS ${position + 1}`);
        }
        tokens.push(token); advance(feed, output, token, position);
        if (token === 2) break;
      }
      entry.predictions.push({ tokens, family: encoded.selected_family_ids.data[0], eos: tokens.at(-1) === 2 });
      if (expected) assert.deepEqual(entry.predictions.at(-1), expected, `${variant}: answer ${index} complete token/EOS sequence`);
      if ((index + 1) % 16 === 0 || index + 1 === count) {
        entry.seconds = (performance.now() - started) / 1000;
        console.log(`${options.backend}/${options.profile}/${variant}: ${index + 1}/${count}, ${entry.seconds.toFixed(1)}s`);
        writeFileSync(out, json(report) + '\n');
      }
    }
    if (baseline) {
      assert.equal(entry.predictions.length, baseline.variants[variant].predictions.length);
      assert.equal(entry.executions.encoder + entry.executions.decoder, baseline.variants[variant].captures.length);
      entry.baseline_equivalence = 'pass';
    }
    Object.assign(entry, { status: 'executed', stage: 'complete', seconds: (performance.now() - started) / 1000 });
  } catch (error) {
    entry.error = String(error);
    if (error.report) entry.operation_report = plain(error.report);
    console.log(variant, entry.stage, entry.error);
  } finally {
    await host?.close();
    writeFileSync(out, json(report) + '\n');
  }
}
if (baseline) {
  report.baseline_equivalence.status = Object.values(report.variants).every(entry => entry.baseline_equivalence === 'pass') ? 'pass' : 'fail';
  writeFileSync(out, json(report) + '\n');
}
if (Object.values(report.variants).some(entry => entry.status !== 'executed')) process.exitCode = 1;
