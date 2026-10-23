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
const { FullEngineHost, VxQuantizationServiceClient, pb } = await import(pathToFileURL(bundle));
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
    const quantization = new VxQuantizationServiceClient(host);
    const request = pb.CreateQuantizationPlanRequest.fromBinary(
      readFileSync(path.join(work, 'ptq', role, 'plan.pb')));
    request.cpuThreads = 1;
    const plan = await quantization.createQuantizationPlan(request);
    assert.equal(plan.info.requiresCalibration, true);
    assert.equal(plan.info.ready, false);
    const folder = path.join(work, 'calibration', role);
    const files = readdirSync(folder).filter(name => name.endsWith('.pb')).sort();
    assert.equal(files.length, native.roles[role].calibration_samples);
    for (let index = 0; index < files.length; index++) {
      const request = pb.CalibrateQuantizationPlanRequest.fromBinary(readFileSync(path.join(folder, files[index])));
      request.quantizationPlanId = plan.quantizationPlanId;
      await quantization.calibrateQuantizationPlan(request);
      if ((index + 1) % 32 === 0 || index + 1 === files.length) console.log(`WASM C PTQ ${role}: ${index + 1}/${files.length}`);
    }
    const observed = (await quantization.inspectQuantizationPlan(new pb.QuantizationPlanRef({ quantizationPlanId: plan.quantizationPlanId }))).info;
    assert.equal(observed.calibration.coverage.complete, true);
    assert.equal(observed.calibration.samples, BigInt(files.length));
    const packed = await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest({ quantizationPlanId: plan.quantizationPlanId }));
    assert.deepEqual(JSON.parse(new TextDecoder().decode(packed.bytes.graph)).nodes,
      JSON.parse(readFileSync(path.join(work, 'ptq', role, 'graph.json'))).nodes);
    await quantization.releaseQuantizationPlan(new pb.QuantizationPlanRef(plan));
    const target = path.join(work, 'ptq-wasm', role);
    mkdirSync(target, { recursive: true });
    writeFileSync(path.join(target, 'graph.json'), packed.bytes.graph);
    writeFileSync(path.join(target, 'model.safetensors'), packed.bytes.weightShards[0]);
    for (const [kind, name, bytes] of [['graph', 'graph.json', packed.bytes.graph], ['weights', 'model.safetensors', packed.bytes.weightShards[0]]]) {
      manifest.graphs[role][kind] = { path: `${role}/${name}`, bytes: bytes.length, sha256: hash(bytes) };
    }
    report.roles[role] = { native_graph_nodes_equal: true, calibration_samples: files.length, coverage_complete: true,
      quantized_nodes: observed.quantized.reduce((count, item) => count + item.nodes, 0n), retained_float_nodes: observed.floatNodes,
      graph_sha256: hash(packed.bytes.graph), weights_sha256: hash(packed.bytes.weightShards[0]), weights_bytes: packed.bytes.weightShards[0].length,
      seconds: (performance.now() - started) / 1000, plan: observed };
    writeFileSync(path.join(work, 'wasm-ptq.json'), json(report) + '\n');
  } finally {
    await host.close();
  }
}
manifest.variant.requested = 'wasm-c-ptq';
writeFileSync(path.join(work, 'ptq-wasm/package_manifest.json'), json(manifest) + '\n');
for (const name of ['config.json', 'vocab.json']) copyFileSync(path.join(work, 'fp32', name), path.join(work, 'ptq-wasm', name));
