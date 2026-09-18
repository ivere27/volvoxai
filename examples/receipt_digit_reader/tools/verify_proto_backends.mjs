/** Execute the shared receipt corpus on required WASM or physical WebGPU. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const options = { backend: 'webgpu', profile: 'full', limit: 0, variants: 'fp32,int8,ptq,ptq-wasm' };
for (let index = 2; index < process.argv.length; index += 2) {
  const name = process.argv[index].replace(/^--/, '');
  assert.ok(['backend', 'profile', 'limit', 'variants', 'corpus', 'out'].includes(name));
  assert.ok(process.argv[index + 1]);
  options[name] = process.argv[index + 1];
}
assert.ok(['wasm', 'webgpu'].includes(options.backend));
assert.ok(['inference', 'full'].includes(options.profile));
assert.ok(options.backend !== 'webgpu' || options.profile === 'full', 'WebGPU belongs to the full browser profile');
assert.ok(options.corpus && options.out);
assert.ok(Number.isInteger(Number(options.limit)) && Number(options.limit) >= 0);
const corpus = path.resolve(options.corpus), outputPath = path.resolve(options.out);
const version = JSON.parse(readFileSync(path.join(root, 'package.json'))).version;
const dist = path.join(root, 'dist', version);
const stem = options.profile === 'full' ? 'volvoxai' : 'volvoxai.lite';
const bundlePath = path.join(dist, `${stem}.js`), wasmPath = path.join(dist, `${stem}.wasm`);
const api = await import(pathToFileURL(bundlePath));
const { pb, VxInferenceServiceClient } = api;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = value => JSON.stringify(value, (_, item) => typeof item === 'bigint' ? item.toString() : item, 2);
const plain = value => JSON.parse(json(value));
const manifestBytes = readFileSync(path.join(corpus, 'corpus.json'));
const manifest = JSON.parse(manifestBytes);
const images = readFileSync(path.join(corpus, 'images.u8'));
assert.equal(hash(images), manifest.images_sha256);
assert.equal(images.length, manifest.samples * 320 * 672);
const count = Math.min(Number(options.limit) || manifest.samples, manifest.samples);
const report = {
  schema: 'volvoxai.receipt-web-backend-verification/v1', backend: options.backend,
  profile: options.profile, samples: count, corpus_sha256: hash(manifestBytes),
  runtime: globalThis.Deno?.version ?? { node: process.version },
  artifacts: { [path.basename(bundlePath)]: hash(readFileSync(bundlePath)), [path.basename(wasmPath)]: hash(readFileSync(wasmPath)) },
  variants: {},
};
mkdirSync(path.dirname(outputPath), { recursive: true });
let currentAdapterInfo;
if (options.backend === 'webgpu') {
  const gpu = globalThis.navigator?.gpu;
  assert.ok(gpu, 'WebGPU is unavailable');
  const requestAdapter = gpu.requestAdapter.bind(gpu);
  // Observe the exact adapter acquired by the unchanged product bridge. The
  // C report identifies its built-in route, while WebGPU owns hardware info.
  gpu.requestAdapter = async (...args) => {
    const adapter = await requestAdapter(...args);
    assert.ok(adapter, 'no physical WebGPU adapter');
    const info = adapter.info ?? {};
    currentAdapterInfo = Object.fromEntries(['vendor', 'architecture', 'device', 'description']
      .map(name => [name, String(info[name] ?? '')]));
    const identity = Object.values(currentAdapterInfo).join(' ');
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
  if (options.backend === 'webgpu') {
    assert.ok(currentAdapterInfo, 'the product bridge must acquire a verified physical adapter');
  }
}

function record(ids) {
  return [ids.slice(0, 12), ids.slice(12)].map(block => {
    const blank = block.indexOf(10);
    return block.slice(0, blank < 0 ? block.length : blank).join('');
  }).join('/');
}

function compare(actual, expected) {
  let maxAbs = 0, sumAbs = 0, sumSquare = 0, slotsEqual = 0, recordsEqual = 0, close = true;
  for (let sample = 0; sample < count; sample++) {
    const actualIds = [], expectedIds = [];
    for (let slot = 0; slot < 16; slot++) {
      let actualBest = 0, expectedBest = 0;
      const base = sample * 176 + slot * 11;
      for (let digit = 0; digit < 11; digit++) {
        const index = base + digit, delta = Math.abs(actual[index] - expected[index]);
        assert.ok(Number.isFinite(actual[index]));
        maxAbs = Math.max(maxAbs, delta); sumAbs += delta; sumSquare += delta * delta;
        close &&= delta <= 1e-4 + 1e-4 * Math.abs(expected[index]);
        if (actual[index] > actual[base + actualBest]) actualBest = digit;
        if (expected[index] > expected[base + expectedBest]) expectedBest = digit;
      }
      actualIds.push(actualBest); expectedIds.push(expectedBest);
      slotsEqual += Number(actualBest === expectedBest);
    }
    recordsEqual += Number(record(actualIds) === record(expectedIds));
  }
  return { max_abs_logit_diff: maxAbs, mean_abs_logit_diff: sumAbs / actual.length,
    rmse: Math.sqrt(sumSquare / actual.length), logits_close_atol_rtol_1e_4: close,
    slot_argmax_agreement: slotsEqual / (count * 16), record_agreement: recordsEqual / count };
}

for (const variant of options.variants.split(',')) {
  const entry = report.variants[variant] = { status: 'failed', stage: 'load' };
  let host;
  try {
    currentAdapterInfo = undefined;
    const packagePath = path.join(corpus, variant);
    for (const [name, checksum] of Object.entries(manifest.packages[variant])) {
      assert.equal(hash(readFileSync(path.join(packagePath, name))), checksum);
    }
    const Constructor = options.profile === 'full' ? api.FullEngineHost : api.EngineHost;
    host = new Constructor({ wasmUrl: pathToFileURL(wasmPath) });
    const client = new VxInferenceServiceClient(host);
    const runtime = await client.createRuntime(new pb.CreateRuntimeRequest({ cpuThreads: 1 }));
    const model = await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: readFileSync(path.join(packagePath, 'graph.json')),
        weightShards: [readFileSync(path.join(packagePath, 'model.safetensors'))] }),
    }));
    entry.stage = 'compile';
    const compiled = await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: [options.backend], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }),
    }));
    entry.compilation = plain(compiled.report);
    attest(compiled.report);
    if (currentAdapterInfo) entry.adapter = currentAdapterInfo;
    const context = await client.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId }));
    const produced = new Float32Array(count * 176);
    entry.stage = 'execute';
    const started = performance.now();
    for (let index = 0; index < count; index++) {
      const values = new Float32Array(320 * 672), offset = index * values.length;
      for (let element = 0; element < values.length; element++) {
        values[element] = Math.fround(Math.fround(images[offset + element] / 255) - .5) / .5;
      }
      const execution = await client.execute(new pb.ExecuteRequest({ contextId: context.contextId,
        inputs: [new pb.Tensor({ name: 'input0', shape: [1n, 1n, 320n, 672n],
          dtype: pb.DataType.DATA_TYPE_F32, inline: new Uint8Array(values.buffer) })],
      }));
      try {
        attest(execution.report);
        if (index === 0) entry.execution = plain(execution.report);
        const deadline = performance.now() + 60000;
        for (;;) {
          const info = await client.getResult(new pb.ResultRef({ resultId: execution.resultId }));
          if (info.state !== pb.ResultState.RESULT_STATE_PENDING) {
            assert.equal(info.state, pb.ResultState.RESULT_STATE_READY);
            break;
          }
          assert.ok(performance.now() < deadline, 'result timed out');
          await new Promise(resolve => setTimeout(resolve, 1));
        }
        const tensor = (await client.readOutput(new pb.ReadOutputRequest({ resultId: execution.resultId, name: 'slot_logits' }))).tensor;
        assert.deepEqual(tensor.shape, [1n, 16n, 11n]);
        assert.equal(tensor.dtype, pb.DataType.DATA_TYPE_F32);
        produced.set(new Float32Array(Uint8Array.from(tensor.inline).buffer), index * 176);
      } finally {
        await client.releaseResult(new pb.ResultRef({ resultId: execution.resultId }));
      }
      if ((index + 1) % 250 === 0) {
        entry.completed_samples = index + 1;
        entry.seconds = (performance.now() - started) / 1000;
        writeFileSync(outputPath, json(report) + '\n');
        console.log(`${options.backend}/${options.profile}/${variant}: ${index + 1}/${count}`);
      }
    }
    const reference = new Float32Array(Uint8Array.from(readFileSync(path.join(packagePath, 'reference.f32'))).buffer);
    Object.assign(entry, { status: 'executed', stage: 'complete', seconds: (performance.now() - started) / 1000,
      comparison: compare(produced, reference.subarray(0, count * 176)) });
    const name = `${path.basename(outputPath, '.json')}-${variant}-logits.f32`;
    writeFileSync(path.join(path.dirname(outputPath), name), new Uint8Array(produced.buffer));
    console.log(variant, json(entry.comparison));
  } catch (error) {
    entry.error = String(error);
    if (error.report) entry.operation_report = plain(error.report);
    console.log(variant, entry.error);
  } finally {
    await host?.close();
    writeFileSync(outputPath, json(report) + '\n');
  }
}
if (Object.values(report.variants).some(entry => entry.status !== 'executed')) process.exitCode = 1;
