/** Repeat native split VQA calibration through the full WASM C proto service. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const work = path.resolve(process.argv[2]);
const version = JSON.parse(readFileSync(path.join(root, 'package.json'))).version;
const dist = path.join(root, 'dist', version);
const bundle = path.join(dist, 'volvoxai.js'), wasm = path.join(dist, 'volvoxai.wasm');
const { FullEngineHost, VxInferenceServiceClient, VxQuantizationServiceClient, pb } = await import(pathToFileURL(bundle));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = value => JSON.stringify(value, (_, item) => typeof item === 'bigint' ? item.toString() : item, 2);
const native = JSON.parse(readFileSync(path.join(work, 'native-ptq.json')));
const manifest = JSON.parse(readFileSync(path.join(work, 'ptq/package_manifest.json')));
const report = { schema: 'volvoxai.vqa-proto-ptq-wasm/v1', runtime: { node: process.version },
  artifacts: { [path.basename(bundle)]: hash(readFileSync(bundle)), [path.basename(wasm)]: hash(readFileSync(wasm)) }, roles: {} };
for (const role of ['encoder', 'decoder']) {
  const started = performance.now();
  const host = new FullEngineHost({ wasmUrl: pathToFileURL(wasm) });
  try {
    const inference = new VxInferenceServiceClient(host), quantization = new VxQuantizationServiceClient(host);
    const graph = readFileSync(path.join(work, 'fp32', role, 'graph.json'));
    const weights = readFileSync(path.join(work, 'fp32', role, 'model.safetensors'));
    const config = pb.PtqAuthoringConfig.fromBinary(readFileSync(path.join(work, 'ptq', role, 'author-config.pb')));
    const authored = await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({ sourceGraph: graph, weightShards: [weights], config }));
    assert.deepEqual(JSON.parse(new TextDecoder().decode(authored.templateGraph)),
      JSON.parse(readFileSync(path.join(work, 'ptq', role, 'template.graph.json'))));
    const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest({ cpuThreads: 1 }));
    const model = await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: graph, weightShards: [weights] }) }));
    const plan = await quantization.createPtqPlan(new pb.CreatePtqPlanRequest({ modelId: model.modelId,
      templateGraph: authored.templateGraph, observers: authored.observers, layers: authored.layers, profileNames: ['default'] }));
    const folder = path.join(work, 'calibration', role);
    const files = readdirSync(folder).filter(name => name.endsWith('.pb')).sort();
    assert.equal(files.length, native.roles[role].calibration_samples);
    for (let index = 0; index < files.length; index++) {
      const request = pb.CalibratePtqPlanRequest.fromBinary(readFileSync(path.join(folder, files[index])));
      request.ptqPlanId = plan.ptqPlanId;
      await quantization.calibratePtqPlan(request);
      if ((index + 1) % 32 === 0 || index + 1 === files.length) console.log(`WASM C PTQ ${role}: ${index + 1}/${files.length}`);
    }
    const observed = await quantization.inspectPtqPlan(new pb.PtqPlanRef({ ptqPlanId: plan.ptqPlanId }));
    assert.equal(observed.coverage.complete, true);
    assert.equal(observed.calibrationSamples, BigInt(files.length));
    const packed = await quantization.writePtqPackage(new pb.WritePtqPackageRequest({ ptqPlanId: plan.ptqPlanId }));
    const target = path.join(work, 'ptq-wasm', role);
    mkdirSync(target, { recursive: true });
    writeFileSync(path.join(target, 'graph.json'), packed.graph);
    writeFileSync(path.join(target, 'model.safetensors'), packed.weights);
    for (const [kind, name, bytes] of [['graph', 'graph.json', packed.graph], ['weights', 'model.safetensors', packed.weights]]) {
      manifest.graphs[role][kind] = { path: `${role}/${name}`, bytes: bytes.length, sha256: hash(bytes) };
    }
    report.roles[role] = { native_template_equal: true, calibration_samples: files.length, coverage_complete: true,
      quantized_nodes: authored.quantizedNodes, retained_float_nodes: authored.retainedFloatNodes,
      graph_sha256: hash(packed.graph), weights_sha256: hash(packed.weights), weights_bytes: packed.weights.length,
      seconds: (performance.now() - started) / 1000, plan: observed };
    writeFileSync(path.join(work, 'wasm-ptq.json'), json(report) + '\n');
  } finally {
    await host.close();
  }
}
manifest.variant.requested = 'wasm-c-ptq';
writeFileSync(path.join(work, 'ptq-wasm/package_manifest.json'), json(manifest) + '\n');
for (const name of ['config.json', 'vocab.json']) copyFileSync(path.join(work, 'fp32', name), path.join(work, 'ptq-wasm', name));
