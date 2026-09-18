/** Repeat native receipt PTQ calibration and inference using release WASM.
 * Run after verify_proto_ptq.py: node this-file.mjs <verification-directory>.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { closeSync, openSync, readFileSync, readSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const work = path.resolve(process.argv[2]);
const version = JSON.parse(readFileSync(path.join(root, 'package.json'))).version;
const dist = path.join(root, 'dist', version);
const full = await import(pathToFileURL(path.join(dist, 'volvoxai.js')));
const inferenceBundle = await import(pathToFileURL(path.join(dist, 'volvoxai.lite.js')));
const { pb, VxInferenceServiceClient, VxQuantizationServiceClient, FullEngineHost } = full;
const sampleBytes = 320 * 672 * 4;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const readPackage = directory => new pb.ModelPackage({
  graphDocument: readFileSync(path.join(directory, 'graph.json')),
  weightShards: [readFileSync(path.join(directory, 'model.safetensors'))],
});
const input = bytes => new pb.Tensor({
  name: 'input0', shape: [1n, 1n, 320n, 672n], dtype: pb.DataType.DATA_TYPE_F32, inline: bytes,
});
const report = { schema: 'volvoxai.receipt-proto-ptq-wasm-verification/v1', artifacts: {}, comparisons: {} };
for (const name of ['volvoxai.lite.js', 'volvoxai.js', 'volvoxai.lite.wasm', 'volvoxai.wasm']) {
  report.artifacts[name] = hash(readFileSync(path.join(dist, name)));
}

const host = new FullEngineHost({ wasmUrl: path.join(dist, 'volvoxai.wasm') });
try {
  const inference = new VxInferenceServiceClient(host);
  const quantization = new VxQuantizationServiceClient(host);
  const source = readPackage(path.join(work, 'prepared'));
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  const model = await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId, package: source }));
  const authored = await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({
    sourceGraph: source.graphDocument, weightShards: source.weightShards,
    config: new pb.PtqAuthoringConfig({
      activationDtype: pb.DataType.DATA_TYPE_I8,
      activationScheme: pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC,
      weightDtype: pb.DataType.DATA_TYPE_I8,
      floatOperators: ['BatchMatMul', 'Linear', 'GroupNorm', 'SiLU', 'LayerNorm', 'Add'],
    }),
  }));
  assert.deepEqual(JSON.parse(new TextDecoder().decode(authored.templateGraph)),
    JSON.parse(readFileSync(path.join(work, 'ptq/template.graph.json'))));
  report.native_template_equal = true;
  const plan = await quantization.createPtqPlan(new pb.CreatePtqPlanRequest({
    modelId: model.modelId, templateGraph: authored.templateGraph,
    observers: authored.observers, layers: authored.layers, profileNames: ['default'],
  }));
  const calibrationPath = path.join(work, 'calibration.f32');
  const samples = statSync(calibrationPath).size / sampleBytes;
  assert.ok(Number.isInteger(samples) && samples > 0);
  const file = openSync(calibrationPath, 'r');
  const raw = Buffer.alloc(sampleBytes);
  try {
    for (let index = 0; index < samples; index++) {
      assert.equal(readSync(file, raw, 0, sampleBytes, index * sampleBytes), sampleBytes);
      await quantization.calibratePtqPlan(new pb.CalibratePtqPlanRequest({
        ptqPlanId: plan.ptqPlanId, profileName: 'default', sampleName: `calibration-${index}`,
        sampleCount: 1n, inputs: [input(raw)],
      }));
      if ((index + 1) % 32 === 0) console.log(`WASM calibration ${index + 1}/${samples}`);
    }
  } finally {
    closeSync(file);
  }
  const observed = await quantization.inspectPtqPlan(new pb.PtqPlanRef({ ptqPlanId: plan.ptqPlanId }));
  assert.equal(observed.coverage.complete, true);
  assert.equal(observed.calibrationSamples, BigInt(samples));
  report.calibration_samples = samples;
  report.observers = observed.tensors.length;
  report.coverage_complete = observed.coverage.complete;
  const packed = await quantization.writePtqPackage(new pb.WritePtqPackageRequest({ ptqPlanId: plan.ptqPlanId }));
  mkdirSync(path.join(work, 'ptq-wasm'), { recursive: true });
  writeFileSync(path.join(work, 'ptq-wasm/graph.json'), packed.graph);
  writeFileSync(path.join(work, 'ptq-wasm/model.safetensors'), packed.weights);
  report.wasm_package = { graph_sha256: hash(packed.graph), weights_sha256: hash(packed.weights), weights_bytes: packed.weights.length };
} finally {
  await host.close();
}

function record(logits) {
  const ids = [];
  for (let slot = 0; slot < 16; slot++) {
    let best = 0;
    for (let digit = 1; digit < 11; digit++) if (logits[slot * 11 + digit] > logits[slot * 11 + best]) best = digit;
    ids.push(best);
  }
  return [ids.slice(0, 12), ids.slice(12)].map(block => block.slice(0, block.indexOf(10) < 0 ? block.length : block.indexOf(10)).join('')).join('/');
}

const batch = readFileSync(path.join(work, 'eval-smoke.f32'));
const sampleCount = batch.length / sampleBytes;
assert.ok(Number.isInteger(sampleCount) && sampleCount > 0);
for (const profile of ['inference', 'full']) {
  const owner = profile === 'full'
    ? new FullEngineHost({ wasmUrl: path.join(dist, 'volvoxai.wasm') })
    : new inferenceBundle.EngineHost({ wasmUrl: path.join(dist, 'volvoxai.lite.wasm') });
  try {
    const client = new VxInferenceServiceClient(owner);
    const runtime = await client.createRuntime(new pb.CreateRuntimeRequest());
    for (const variant of ['fp32', 'int8', 'ptq', 'ptq-wasm']) {
      const model = await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId, package: readPackage(path.join(work, variant)) }));
      const compiled = await client.compileModel(new pb.CompileModelRequest({
        modelId: model.modelId,
        policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'] }),
      }));
      const context = await client.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId }));
      const referenceBytes = readFileSync(path.join(work, `volvoxai_${variant === 'ptq-wasm' ? 'ptq' : variant}-smoke-logits.f32`));
      const reference = new Float32Array(Uint8Array.from(referenceBytes).buffer);
      let maxAbs = 0, recordsEqual = 0;
      for (let index = 0; index < sampleCount; index++) {
        const result = await client.execute(new pb.ExecuteRequest({ contextId: context.contextId, inputs: [input(batch.subarray(index * sampleBytes, (index + 1) * sampleBytes))] }));
        try {
          const output = (await client.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name: 'slot_logits' }))).tensor;
          assert.deepEqual(output.shape, [1n, 16n, 11n]);
          const actual = new Float32Array(output.inline.slice().buffer);
          const expected = reference.subarray(index * 176, (index + 1) * 176);
          for (let element = 0; element < actual.length; element++) {
            assert.ok(Number.isFinite(actual[element]));
            maxAbs = Math.max(maxAbs, Math.abs(actual[element] - expected[element]));
          }
          recordsEqual += Number(record(actual) === record(expected));
        } finally {
          await client.releaseResult(new pb.ResultRef({ resultId: result.resultId }));
        }
      }
      report.comparisons[`${profile}_${variant}`] = { samples: sampleCount, max_abs_logit_diff_vs_native: maxAbs, equal_records: recordsEqual };
      console.log(profile, variant, report.comparisons[`${profile}_${variant}`]);
      if (variant === 'fp32') assert.ok(maxAbs < 1e-3);
      await client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId: context.contextId }));
      await client.releaseCompiledModel(new pb.CompiledModelRef({ compiledModelId: compiled.compiledModelId }));
      await client.releaseModel(new pb.ModelRef({ modelId: model.modelId }));
    }
  } finally {
    await owner.close();
  }
}
writeFileSync(path.join(work, 'wasm-report.json'), JSON.stringify(report, null, 2) + '\n');
