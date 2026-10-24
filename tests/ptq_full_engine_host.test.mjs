import { observeActivations } from '../tools/calibration/activation_observer.mjs';
import { statelessGraph, fixture as protoFixture, safetensors as protoSafetensors, tensors as protoTensors } from '../tools/proto_fixture.mjs';
import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { FullEngineHost } from '../ts/full.js';
import { EngineHost } from '../ts/host/EngineHost.js';
import {
  VxPlatformServiceClient,
  VxInferenceServiceClient,
  VxQuantizationServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';
import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PACKAGE_VERSION = JSON.parse(
  readFileSync(path.join(ROOT, 'package.json'), 'utf8'),
).version;
const FULL_WASM = process.env.VOLVOXAI_FULL_WASM ??
  path.join(ROOT, 'dist', PACKAGE_VERSION, 'volvoxai.wasm');
const INFERENCE_WASM = process.env.VOLVOXAI_INFERENCE_WASM ?? path.join(
  ROOT, 'dist', PACKAGE_VERSION, 'volvoxai.lite.wasm',
);
function safetensors(entries) {
  const header = {};
  const payloads = [];
  let offset = 0;
  for (const [name, { shape, values, dtype = 'F32' }] of Object.entries(entries)) {
    const ArrayType = { F32: Float32Array, I32: Int32Array, I8: Int8Array, U8: Uint8Array }[dtype];
    const payload = new Uint8Array(ArrayType.from(values).buffer);
    header[name] = { dtype, shape, data_offsets: [offset, offset + payload.byteLength] };
    payloads.push(payload);
    offset += payload.byteLength;
  }
  let headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const padding = (8 - (headerBytes.byteLength % 8)) % 8;
  if (padding) {
    const padded = new Uint8Array(headerBytes.byteLength + padding);
    padded.set(headerBytes);
    padded.fill(0x20, headerBytes.byteLength);
    headerBytes = padded;
  }
  const result = new Uint8Array(8 + headerBytes.byteLength + offset);
  new DataView(result.buffer).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  result.set(headerBytes, 8);
  let cursor = 8 + headerBytes.byteLength;
  for (const payload of payloads) {
    result.set(payload, cursor);
    cursor += payload.byteLength;
  }
  return result;
}

function fixture() {
  const graph = new TextEncoder().encode(JSON.stringify(statelessGraph({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { input: { shape: [1, 2], dtype: 'float32' } },
    nodes: [{
      id: 'dense',
      opType: 'Linear',
      inputs: { input: 'input', weight: 'dense.weight', bias: 'dense.bias' },
      outputs: {
        out: { tensor: 'hidden', shape: [1, 2], dtype: 'float32' },
      },
      params: { weight_layout: 'dout_din' },
    }, {
      id: 'activation',
      opType: 'GELU',
      inputs: { input: 'hidden' },
      outputs: {
        out: { tensor: 'output', shape: [1, 2], dtype: 'float32' },
      },
      params: {},
    }],
    outputs: ['output'],
  }, null, 1)));
  const weights = safetensors({
    'dense.weight': { shape: [2, 2], values: [1, 2, 3, 4] },
    'dense.bias': { shape: [2], values: [0.25, -0.5] },
  });
  return { graph, weights };
}

function config({bits = 8, groupSize = 0, mode = 'STATIC', dtype = 'I8',
  scheme = pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC, reduceRange = false,
  floatOperators = [], floatNodes = [], selectedNodes = [], rules = []} = {}) {
  const weights = new pb.WeightQuantization({format: new pb.WeightFormat({bits, groupSize})});
  return new pb.QuantizationConfig({weights,
    staticActivations: mode === 'STATIC' ? new pb.StaticActivations({
      dtype: pb.DataType[`DATA_TYPE_${dtype}`], scheme, reduceWeightRange: reduceRange}) : undefined,
    rules: [...rules, ...floatOperators.map(operatorName => new pb.QuantizationRule({operatorName, keepFloat: new pb.Empty()})),
      ...floatNodes.map(nodeId => new pb.QuantizationRule({nodeId, keepFloat: new pb.Empty()})),
      ...selectedNodes.map(nodeId => new pb.QuantizationRule({nodeId, weights})),
      ...(selectedNodes.length ? [new pb.QuantizationRule({nodeGlob: '*', keepFloat: new pb.Empty()})] : [])],
  });
}
function checkedPlan(plan) {
  checkedReport(plan);
  assert.ok(plan.info);
  assert.ok(plan.quantizationPlanId > 0n);
  return plan;
}

function largePtqFixture(width, rows, totalBytes) {
  const document = new TextEncoder().encode(JSON.stringify(statelessGraph({
    format: 'volvox-graph/v1', dimensions: {},
    inputs: {x: {shape: [1, width], dtype: 'float32'}},
    nodes: [{id: 'linear', opType: 'Linear', inputs: {input: 'x', weight: 'w'},
      outputs: {out: {tensor: 'y', shape: [1, rows], dtype: 'float32'}},
      params: {weight_layout: 'dout_din'}}], outputs: ['y'],
  })));
  const data = new Float32Array(width * rows);
  for (let row = 0; row < rows; row++) data.fill(row % 2 ? -.125 : .125, row * width, (row + 1) * width);
  const weights = protoSafetensors([{name: 'w', shape: [rows, width], data}]);
  const graph = totalBytes === undefined ? document : new Uint8Array(totalBytes - weights.byteLength).fill(0x20);
  graph.set(document);
  if (totalBytes !== undefined) assert.equal(graph.byteLength + weights.byteLength, totalBytes);
  return {graph, weights};
}

async function largePtqLifecycle({width, rows, totalBytes, importance}) {
  const {graph, weights} = largePtqFixture(width, rows, totalBytes);
  const values = Float32Array.from({length: width}, (_, i) => i % 2 ? .25 : 1);
  const input = new pb.Tensor({name: 'x', dtype: pb.DataType.DATA_TYPE_F32,
    shape: [1n, BigInt(width)], inline: new Uint8Array(values.buffer)});
  const policy = importance ? new pb.QuantizationConfig({weights: new pb.WeightQuantization({
    format: new pb.WeightFormat({bits: 4, groupSize: 32}),
    algorithm: pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE,
  })}) : config({scheme: pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC});
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    const inference = new VxInferenceServiceClient(reportTransport(host));
    const runtime = checkedReport(await inference.createRuntime(new pb.CreateRuntimeRequest({cpuThreads: 1})));
    if (totalBytes === 64 * 1024 * 1024 + 1) {
      const rejected = await inference.loadModel(new pb.LoadModelRequest({runtimeId: runtime.runtimeId,
        package: new pb.ModelPackage({graphDocument: graph, weightShards: [weights]})}));
      assert.equal(rejected.report.code, pb.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE,
        'ordinary LoadModel retains its inline transport policy');
    }
    const request = new pb.CreateQuantizationPlanRequest({config: policy,
      bytes: new pb.ModelBytes({graph, weightShards: [weights]}),
      calibration: new pb.CalibrationOptions({cpuThreads: 1})});
    const plan = checkedPlan(await quantization.createQuantizationPlan(request));
    const ref = new pb.QuantizationPlanRef(plan);
    assert.equal(plan.info.requiresCalibration, true);
    assert.equal(plan.info.ready, false);
    // Accepted bytes belong to the plan, not the caller's typed-array storage.
    graph.fill(0); weights.fill(0);
    request.bytes.graph = new Uint8Array(); request.bytes.weightShards = [];
    checkedReport(await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
      quantizationPlanId: plan.quantizationPlanId, sampleName: 'snapshot', sampleCount: 1n, inputs: [input],
    })));
    const state = checkedReport(await quantization.inspectQuantizationPlan(ref)).info;
    assert.equal(state.ready, true);
    assert.equal(state.calibration.samples, 1n);
    assert.equal(state.calibration.coverage.complete, true);
    const exported = checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan)));
    const again = checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan)));
    assert.deepEqual(again.bytes.graph, exported.bytes.graph);
    assert.deepEqual(again.bytes.weightShards, exported.bytes.weightShards);
    checkedReport(await quantization.releaseQuantizationPlan(ref));
    const model = checkedReport(await inference.loadModel(new pb.LoadModelRequest({runtimeId: runtime.runtimeId,
      package: new pb.ModelPackage({graphDocument: exported.bytes.graph, weightShards: exported.bytes.weightShards})})));
    const compiled = checkedReport(await inference.compileModel(new pb.CompileModelRequest({modelId: model.modelId,
      packedArithmetic: importance ? pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8 : pb.PackedArithmetic.PACKED_ARITHMETIC_F32,
      policy: new pb.BackendPolicy({mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm']})})));
    const context = checkedReport(await inference.createExecutionContext(new pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()})));
    const result = checkedReport(await inference.execute(new pb.ExecuteRequest({contextId: context.contextId,
      inputs: new pb.InputBindings({values: [input]}),
      scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()})})));
    const output = result.outputs.find(tensor => tensor.name === 'y');
    assert.deepEqual(output.shape, [1n, BigInt(rows)]);
    const actual = new Float32Array(output.inline.slice().buffer);
    for (let row = 0; row < rows; row++) {
      const expected = width * .625 * (row % 2 ? -.125 : .125);
      assert.ok(Math.abs(actual[row] - expected) <= .02 + .006 * Math.abs(expected),
        `${importance ? 'importance' : 'static'} row ${row}: ${actual[row]} versus ${expected}`);
    }
    checkedReport(await inference.releaseExecutionContext(new pb.ExecutionContextRef(context)));
    checkedReport(await inference.releaseCompiledModel(new pb.CompiledModelRef(compiled)));
    checkedReport(await inference.releaseModel(new pb.ModelRef(model)));
    checkedReport(await inference.releaseRuntime(new pb.RuntimeRef(runtime)));
  } finally { await host.close(); }
}

test('calibrated PTQ owns WASM byte sources below, at and above 64 MiB', {timeout: 180_000}, async () => {
  for (const delta of [-1, 0, 1]) for (const importance of [false, true]) {
    await largePtqLifecycle({width: 32, rows: 2, totalBytes: 64 * 1024 * 1024 + delta, importance});
  }
});

test('calibrated PTQ consumes WASM weights above 72 MiB through export and inference', {timeout: 240_000}, async () => {
  const width = 4096, rows = 4609;
  assert.ok(width * rows * Float32Array.BYTES_PER_ELEMENT > 72 * 1024 * 1024);
  for (const importance of [false, true]) await largePtqLifecycle({width, rows, importance});
});

test('PTQ byte export preserves a retained shard larger than 72 MiB', {timeout: 180_000}, async () => {
  const {graph} = largePtqFixture(32, 2);
  const retained = new Float32Array(72 * 1024 * 1024 / 4 + 1);
  retained[0] = 1.25; retained[retained.length - 1] = -2.5;
  const weights = protoSafetensors([
    {name: 'w', shape: [2, 32], data: new Float32Array(64).fill(.125)},
    {name: 'retained', shape: [retained.length], data: retained},
  ]);
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({
      config: config(), bytes: new pb.ModelBytes({graph, weightShards: [weights]}),
    })));
    graph.fill(0); weights.fill(0); retained.fill(0);
    checkedReport(await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
      quantizationPlanId: plan.quantizationPlanId, sampleName: 'large-export', sampleCount: 1n,
      inputs: protoTensors({x: {data: new Float32Array(32).fill(1), shape: [1, 32]}}),
    })));
    const exported = checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan)));
    checkedReport(await quantization.releaseQuantizationPlan(new pb.QuantizationPlanRef(plan)));
    const shard = exported.bytes.weightShards[0];
    assert.ok(shard.byteLength > 72 * 1024 * 1024);
    const view = new DataView(shard.buffer, shard.byteOffset, shard.byteLength);
    const headerBytes = Number(view.getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder().decode(shard.subarray(8, 8 + headerBytes))).retained;
    assert.equal(metadata.dtype, 'F32');
    assert.deepEqual(metadata.shape, [72 * 1024 * 1024 / 4 + 1]);
    assert.equal(view.getFloat32(8 + headerBytes + metadata.data_offsets[0], true), 1.25);
    assert.equal(view.getFloat32(8 + headerBytes + metadata.data_offsets[1] - 4, true), -2.5);
  } finally { await host.close(); }
});

test('inference EngineHost keeps quantization plans unregistered', async () => {
  const host = new EngineHost({wasmUrl: INFERENCE_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    await assert.rejects(quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config(), bytes: new pb.ModelBytes({graph: new TextEncoder().encode('{}')})})), error => error.name === 'RpcError' && error.code === 12);
  } finally { await host.close(); }
});

test('quantization policy separates weight grouping and activation execution', () => {
  const fields = Message => Message.fields.map(field => field.name);
  assert.deepEqual(fields(pb.WeightFormat), ['bits', 'group_size']);
  assert.deepEqual(fields(pb.WeightQuantization), ['format', 'algorithm']);
  assert.deepEqual(fields(pb.StaticActivations), ['dtype', 'scheme', 'reduce_weight_range']);
  assert.equal(fields(pb.QuantizationConfig).includes('precision'), false);
  assert.ok(fields(pb.CalibrationOptions).includes('backend'));
  assert.ok(fields(pb.QuantizationPlanInfo).includes('calibration_backend'));
});

test('WASM PTQ preserves explicit CPU observations and refuses required CUDA observations', async () => {
  const {graph, weights} = fixture();
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    const importance = new pb.QuantizationConfig({weights: new pb.WeightQuantization({
      format: new pb.WeightFormat({bits: 2, groupSize: 32}),
      algorithm: pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE,
    })});
    for (const policy of [config(), importance]) {
      const exports = [];
      for (const backend of ['', 'cpu']) {
        const request = new pb.CreateQuantizationPlanRequest({config: policy,
          bytes: new pb.ModelBytes({graph, weightShards: [weights]}),
          calibration: new pb.CalibrationOptions({backend, cpuThreads: 1})});
        const decoded = pb.CreateQuantizationPlanRequest.fromBinary(request.toBinary());
        assert.equal(decoded.calibration.backend, backend);
        const plan = checkedPlan(await quantization.createQuantizationPlan(decoded));
        assert.equal(plan.info.requiresCalibration, true);
        assert.equal(plan.info.calibrationBackend, 'cpu');
        const ref = new pb.QuantizationPlanRef(plan);
        checkedReport(await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
          quantizationPlanId: plan.quantizationPlanId, sampleName: 'cpu-required', sampleCount: 1n,
          inputs: protoTensors({input: {data: Float32Array.of(.5, -.25), shape: [1, 2]}}),
        })));
        const inspected = checkedReport(await quantization.inspectQuantizationPlan(ref));
        assert.equal(inspected.info.calibrationBackend, 'cpu');
        assert.equal(inspected.info.calibration.samples, 1n);
        exports.push(checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan))).bytes);
        checkedReport(await quantization.releaseQuantizationPlan(ref));
      }
      assert.deepEqual(exports[0], exports[1], 'default and explicit CPU produce the same package');
      const denied = await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: policy,
        bytes: new pb.ModelBytes({graph, weightShards: [weights]}),
        calibration: new pb.CalibrationOptions({backend: 'cuda'})}));
      assert.equal(denied.quantizationPlanId, 0n);
      assert.equal(denied.report.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNAVAILABLE);
      assert.equal(denied.report.code, pb.OperationCode.OPERATION_CODE_BACKEND_UNAVAILABLE);
    }
    for (const backend of ['', 'cpu', 'cuda']) {
      const policy = config({bits: 3, groupSize: 32, mode: 'FLOAT'});
      policy.weights.algorithm = pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP;
      const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: policy,
        bytes: new pb.ModelBytes({graph, weightShards: [weights]}),
        calibration: new pb.CalibrationOptions({backend})})));
      assert.equal(plan.info.requiresCalibration, false);
      assert.equal(plan.info.calibrationBackend, '', 'data-free fitting creates no calibration executor');
      assert.equal(checkedReport(await quantization.inspectQuantizationPlan(new pb.QuantizationPlanRef(plan))).info.calibrationBackend, '');
      checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan)));
      checkedReport(await quantization.releaseQuantizationPlan(new pb.QuantizationPlanRef(plan)));
    }
    const invalid = await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({
      config: config({bits: 3, groupSize: 32, mode: 'FLOAT'}),
      bytes: new pb.ModelBytes({graph, weightShards: [weights]}),
      calibration: new pb.CalibrationOptions({backend: 'webgpu'})}));
    assert.equal(invalid.quantizationPlanId, 0n);
    assert.equal(invalid.report.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  } finally { await host.close(); }
});

test('static plan owns its source and reports calibration readiness', async () => {
  const {graph, weights} = fixture();
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const platform = new VxPlatformServiceClient(reportTransport(host));
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    assert.equal((await platform.getPlatformInfo(new pb.Empty())).buildProfile, pb.BuildProfile.BUILD_PROFILE_FULL);
    const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config({dtype: 'U8', scheme: pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC, selectedNodes: ['dense']}), bytes: new pb.ModelBytes({graph: graph, weightShards: [weights]}), calibration: new pb.CalibrationOptions({profileNames: ['default']})})));
    assert.equal(plan.info.requiresCalibration, true);
    assert.equal(plan.info.ready, false);
    assert.equal(plan.info.quantized.reduce((total, item) => total + item.nodes, 0n), 1n);
    assert.equal(plan.info.floatNodes, 1n);
    const ref = new pb.QuantizationPlanRef(plan);
    const state = (checkedReport(await quantization.inspectQuantizationPlan(ref))).info;
    assert.equal(state.ready, false);
    const incomplete = await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan));
    assert.notEqual(incomplete.report.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.equal(incomplete.bytes?.graph?.byteLength ?? 0, 0);
    checkedReport(await quantization.releaseQuantizationPlan(ref));
  } finally { await host.close(); }
});

test('unsupported weight and activation combinations fail before publishing a plan', async () => {
  const {graph, weights} = fixture();
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    for (const options of [
      {bits: 0}, {bits: 5}, {mode: 'UNSPECIFIED'},
      {mode: 'STATIC', dtype: 'F16'}, {bits: 4, groupSize: 0},
      {bits: 4, groupSize: 128}, {bits: 8, groupSize: 128},
      {bits: 4, groupSize: 48, mode: 'FLOAT', dtype: 'F16'},
    ]) {
      const result = await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config(options), bytes: new pb.ModelBytes({graph: graph, weightShards: [weights]})}));
      const report = result.report;
      assert.notEqual(report.status, pb.NativeStatus.NATIVE_STATUS_OK, JSON.stringify(options));
      assert.equal(result.quantizationPlanId, 0n);
    }
  } finally { await host.close(); }
});

test('explicit selector rules reach quantization planning', async () => {
  const {graph, weights} = fixture();
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    for (const selection of [{floatOperators: ['GELU']}, {floatNodes: ['activation']}, {selectedNodes: ['dense']}]) {
      const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config(selection), bytes: new pb.ModelBytes({graph: graph, weightShards: [weights]})})));
      assert.equal(plan.info.quantized.reduce((total, item) => total + item.nodes, 0n), 1n);
      assert.equal(plan.info.floatNodes, 1n);
      checkedReport(await quantization.releaseQuantizationPlan(new pb.QuantizationPlanRef(plan)));
    }
    const weightless = await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({
      config: config({floatNodes: ['dense']}),
      bytes: new pb.ModelBytes({graph, weightShards: [weights]}),
    }));
    assert.equal(weightless.report.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
    assert.equal(weightless.quantizationPlanId, 0n);
  } finally { await host.close(); }
});

test('invalid graph sources and private WASM paths return typed reports', async () => {
  const {weights} = fixture();
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    for (const [source, status] of [
      [{bytes: new pb.ModelBytes({graph: new TextEncoder().encode('{not json'), weightShards: [weights]})}, pb.NativeStatus.NATIVE_STATUS_INVALID_GRAPH],
      [{files: new pb.ModelFiles({graphPath: 'graph.json'})}, pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED],
    ]) {
      const result = await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({...source, config: config()}));
      assert.equal((result.report).status, status);
      assert.equal(result.quantizationPlanId, 0n);
    }
  } finally { await host.close(); }
});

test('structural name limits remain INVALID_GRAPH rather than OOM', async () => {
  const input = 'x'.repeat(128);
  const document = statelessGraph({
    format: 'volvox-graph/v1', dimensions: {},
    inputs: {[input]: {shape: [1], dtype: 'float32'}},
    nodes: [{id: 'gelu', opType: 'GELU', inputs: {input},
      outputs: {out: {tensor: 'output', shape: [1], dtype: 'float32'}}, params: {}}],
    outputs: ['output'],
  });
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    const result = await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config(), bytes: new pb.ModelBytes({graph: new TextEncoder().encode(JSON.stringify(document)), weightShards: [safetensors({})]})}));
    assert.equal(result.report.status, pb.NativeStatus.NATIVE_STATUS_INVALID_GRAPH);
    assert.equal(result.quantizationPlanId, 0n);
  } finally { await host.close(); }
});

test('missing and incompatible sidecars reject quantization transport', async () => {
  const request = new pb.CreateQuantizationPlanRequest({config: config(), bytes: new pb.ModelBytes({graph: new TextEncoder().encode('{}')})});
  for (const [wasmUrl, error] of [[path.join(ROOT, 'missing-ptq.wasm'), /ENOENT/],
    [INFERENCE_WASM, /imports do not match the full profile/]]) {
    const host = new FullEngineHost({wasmUrl});
    try {
      const quantization = new VxQuantizationServiceClient(reportTransport(host));
      await assert.rejects(quantization.createQuantizationPlan(request), error);
    } finally { await host.close(); }
  }
});

test('C PTQ transposes din_dout Linear weights and preserves dynamic rows', async () => {
  const graph = new TextEncoder().encode(JSON.stringify(statelessGraph({
    format: 'volvox-graph/v1', dimensions: { S: { min: 1, max: 3 }, source: { min: 3, max: 3 } }, banks: { weight: 'source' },
    inputs: { input: { shape: [1, 'S', 3], dtype: 'float32' } },
    nodes: [{ id: 'dense', opType: 'Linear', inputs: { input: 'input', weight: 'weight' },
      outputs: { out: { tensor: 'output', shape: [1, 'S', 2], dtype: 'float32' } },
      params: { weight_layout: 'din_dout' } }], outputs: ['output'],
  })));
  const weights = safetensors({ weight: { shape: [3, 2], values: [.25, 2, -.75, .5, 1.5, -1] } });
  const values = Float32Array.of(-1, .5, 1, .25, -.75, .5, 0, 1, -1);
  const input = rows => new pb.Tensor({ name: 'input', dtype: pb.DataType.DATA_TYPE_F32,
    shape: [1n, BigInt(rows), 3n], inline: new Uint8Array(values.buffer, 0, rows * 3 * 4) });
  const host = new FullEngineHost({ wasmUrl: FULL_WASM });
  const readerHost = new EngineHost({ wasmUrl: INFERENCE_WASM });
  try {
    const quantization = new VxQuantizationServiceClient(host);
    const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config({scheme: pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC}), bytes: new pb.ModelBytes({graph: graph, weightShards: [weights]}), calibration: new pb.CalibrationOptions({profileNames: ['dynamic']})})));
    for (const rows of [1, 3, 2]) await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
      quantizationPlanId: plan.quantizationPlanId, profileName: 'dynamic', sampleName: `rows-${rows}`, sampleCount: 1n, inputs: [input(rows)] }));
    const state = (await quantization.inspectQuantizationPlan(new pb.QuantizationPlanRef({ quantizationPlanId: plan.quantizationPlanId }))).info;
    assert.equal(state.calibration.coverage.complete, true);
    assert.equal(state.calibration.samples, 3n);
    const packed = await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest({ quantizationPlanId: plan.quantizationPlanId }));
    assert.deepEqual(JSON.parse(new TextDecoder().decode(packed.bytes.graph)).banks, { weight: 'source' });
    const reader = new VxInferenceServiceClient(readerHost);
    const readRuntime = await reader.createRuntime(new pb.CreateRuntimeRequest());
    const quantized = await reader.loadModel(new pb.LoadModelRequest({ runtimeId: readRuntime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: packed.bytes.graph, weightShards: [packed.bytes.weightShards[0]] }) }));
    const compiled = await reader.compileModel(new pb.CompileModelRequest({ modelId: quantized.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'] }) }));
    const context = await reader.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()}));
    for (const rows of [3, 1, 2]) {
      const result = await reader.execute(new pb.ExecuteRequest({ contextId: context.contextId, inputs: new pb.InputBindings({values: [input(rows)] }), scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()})}));
      const output = result.outputs.find(tensor => tensor.name === 'output');
      assert.deepEqual(output.shape, [1n, BigInt(rows), 2n]);
      const actual = new Float32Array(output.inline.slice().buffer);
      for (let row = 0; row < rows; row++) {
        const [a, b, c] = values.subarray(row * 3, row * 3 + 3);
        assert.ok(Math.abs(actual[row * 2] - (.25 * a - .75 * b + 1.5 * c)) < .04);
        assert.ok(Math.abs(actual[row * 2 + 1] - (2 * a + .5 * b - c)) < .04);
      }
    }
  } finally {
    await readerHost.close();
    await host.close();
  }
});

for (const [precision, weightBits, activationBits] of [['W4A8', 4, 8], ['W4A16', 4, 16], ['W8A8', 8, 8], ['W8A16', 8, 16]]) {
  test(`grouped ${precision} bytes and arithmetic survive full-to-lite WASM publication`, async () => {
    const k = 131, n = 3, rows = 3, limit = 2 ** (weightBits - 1) - 1;
    const perWord = 32 / weightBits, offset = limit + 1, words = Math.ceil(k / perWord);
    const nearestEven = value => {
      const lower = Math.floor(value), fraction = value - lower;
      return lower + (fraction > .5 || (fraction === .5 && lower % 2 !== 0) ? 1 : 0);
    };
    // Integer or half-integer source weights give exact power-of-two scales.
    // These activations and decoded weights are also exactly representable
    // in FP16, so the expected result needs no copy of the engine's converter.
    const weight = Float32Array.from({length: n * k}, (_, index) => {
      const channel = Math.floor(index / k), column = index % k;
      if (channel === 2) return 0;
      if (column === 0) return limit;
      if (column >= 128) return [limit * 2, -limit * 2, 3][column - 128];
      return [-2.5, -1.5, -.5, .5, 1.5, 2.5, 0][(column + channel) % 7];
    });
    const values = Float32Array.from({length: rows * k}, (_, index) => {
      const row = Math.floor(index / k), column = index % k;
      if (row === 0) return 0;
      return (row === 1 ? 1 : -1) * (column === 0 ? 127 : ((column % 13) - 6) / 2);
    });
    const bias = Float32Array.of(.25, -.5, 1);
    const expected = new Float32Array(rows * n), quantized = new Int32Array(n * k);
    for (let channel = 0; channel < n; channel++) {
      for (let column = 0; column < k; column++) {
        const scale = channel === 2 ? 1 : column < 128 ? 1 : 2;
        quantized[channel * k + column] = nearestEven(weight[channel * k + column] / scale);
      }
      for (let row = 0; row < rows; row++) {
        let sum = 0;
        for (let column = 0; column < k; column++) {
          const value = values[row * k + column];
          const activation = activationBits === 16 ? value : nearestEven(value); // Row scale is exactly one.
          const scale = channel === 2 ? 1 : column < 128 ? 1 : 2;
          sum += activation * quantized[channel * k + column] * scale;
        }
        expected[row * n + channel] = sum + bias[channel];
      }
    }
    const inputs = protoTensors({input: {data: values, shape: [rows, k]}});
    for (const layout of ['dout_din', 'din_dout']) {
      const document = statelessGraph({format: 'volvox-graph/v1', dimensions: {},
        inputs: {input: {shape: [rows, k], dtype: 'float32'}}, nodes: [{id: 'dense', opType: 'Linear',
          inputs: {input: 'input', weight: 'weight', bias: 'bias'}, params: {weight_layout: layout},
          outputs: {out: {tensor: 'output', shape: [rows, n], dtype: 'float32'}}}], outputs: ['output']});
      const graph = new TextEncoder().encode(JSON.stringify(document));
      const source = safetensors({weight: {shape: layout === 'dout_din' ? [n, k] : [k, n],
        values: layout === 'dout_din' ? weight : Float32Array.from({length: k * n},
          (_, index) => weight[(index % n) * k + Math.floor(index / n)])}, bias: {shape: [n], values: bias}});
      const sourceSnapshot = source.slice();
      // Creating a grouped plan owns the packed snapshot without a source Runtime or Model.
      const packingHost = new FullEngineHost({wasmUrl: FULL_WASM});
      const quantization = new VxQuantizationServiceClient(reportTransport(packingHost));
      let packed;
      try {
        const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config({bits: weightBits, groupSize: 128, mode: activationBits === 8 ? 'DYNAMIC' : 'FLOAT',
            dtype: activationBits === 8 ? 'I8' : 'F16', selectedNodes: ['dense']}), bytes: new pb.ModelBytes({graph: graph, weightShards: [source]})})));
        assert.equal(plan.info.quantized.reduce((total, item) => total + item.nodes, 0n), 1n);
        assert.equal(plan.info.floatNodes, 0n);
        assert.equal(plan.info.requiresCalibration, false);
        assert.equal(plan.info.ready, true);
        assert.equal(plan.info.calibration, undefined);
        const inspected = (checkedReport(await quantization.inspectQuantizationPlan(new pb.QuantizationPlanRef(plan)))).info;
        assert.equal(inspected.ready, true);
        assert.equal(inspected.requiresCalibration, false);
        const refused = await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
          quantizationPlanId: plan.quantizationPlanId, sampleName: 'unused', sampleCount: 1n, inputs,
        }));
        assert.notEqual(refused.report.status, pb.NativeStatus.NATIVE_STATUS_OK);
        packed = checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan)));
        const repeated = checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan)));
        assert.deepEqual(repeated.bytes.graph, packed.bytes.graph);
        assert.deepEqual(repeated.bytes.weightShards[0], packed.bytes.weightShards[0]);
        checkedReport(await quantization.releaseQuantizationPlan(new pb.QuantizationPlanRef(plan)));
        const converted = JSON.parse(new TextDecoder().decode(packed.bytes.graph)).nodes[0];
        assert.equal(converted.opType, 'PackedLinear');
        assert.deepEqual(converted.params, {group_size: 128, weight_bits: weightBits});
        const bytes = new DataView(packed.bytes.weightShards[0].buffer, packed.bytes.weightShards[0].byteOffset, packed.bytes.weightShards[0].byteLength);
        const headerLength = Number(bytes.getBigUint64(0, true));
        const header = JSON.parse(new TextDecoder().decode(packed.bytes.weightShards[0].subarray(8, 8 + headerLength)));
        assert.equal(header.__metadata__?.profile_coverage, undefined);
        assert.equal(header.__metadata__?.activation_sources, undefined);
        const wordEntry = header[converted.inputs.weight], scaleEntry = header[converted.inputs.scale];
        assert.equal(wordEntry.dtype, 'I32');
        assert.deepEqual(wordEntry.shape, [n, words]);
        assert.equal(scaleEntry.dtype, 'F32');
        assert.deepEqual(scaleEntry.shape, [n, 2]);
        for (let channel = 0; channel < n; channel++) {
          for (let group = 0; group < 2; group++)
            assert.equal(bytes.getFloat32(8 + headerLength + scaleEntry.data_offsets[0] + (channel * 2 + group) * 4, true),
              channel === 2 ? 1 : group + 1);
          for (let column = 0; column < words * perWord; column++) {
            const word = bytes.getUint32(8 + headerLength + wordEntry.data_offsets[0] +
              (channel * words + Math.floor(column / perWord)) * 4, true);
            assert.equal(((word >>> ((column % perWord) * weightBits)) & (2 * offset - 1)) - offset,
              column < k ? quantized[channel * k + column] : 0);
          }
        }
        assert.deepEqual(source, sourceSnapshot);
      } finally { await packingHost.close(); }
      // Package bytes remain usable after the authoring host has closed.
      for (const full of [true, false]) {
        const reader = await protoFixture({full, wasmUrl: full ? FULL_WASM : INFERENCE_WASM});
        try {
          const model = await reader.load(JSON.parse(new TextDecoder().decode(packed.bytes.graph)), packed.bytes.weightShards[0]);
          const compiled = await reader.compile(model, 'wasm', activationBits === 8
            ? pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8 : pb.PackedArithmetic.PACKED_ARITHMETIC_F16);
          const result = await reader.evaluate({compiledModelId: compiled.compiledModelId, inputs});
          try { assert.deepEqual(await reader.read(result, 'output'), expected, `${precision} ${layout} full=${full}`); }
          finally { await reader.releaseOutputs(result); }
        } finally { await reader.close(); }
      }
    }
  });
}


test('C PTQ exports a bias-free HWIO convolution between float regions', async () => {
  const shape = [1, 1, 3, 3];
  const graph = new TextEncoder().encode(JSON.stringify(statelessGraph({
    format: 'volvox-graph/v1', dimensions: {},
    inputs: { input: { shape: [1, 6], dtype: 'float32' } },
    nodes: [{
      id: 'view', opType: 'Reshape', inputs: { input: 'input' },
      outputs: { out: { tensor: 'image', shape: [1, 1, 3, 2], dtype: 'float32' } },
      params: { shape: [1, 1, 3, 2] },
    }, {
      id: 'conv', opType: 'Conv2D', inputs: { input: 'image', weight: 'weight' },
      outputs: { out: { tensor: 'features', shape, dtype: 'float32' } },
      params: { data_layout: 'NHWC', weight_layout: 'HWIO', stride: [1, 1] },
    }, {
      id: 'float', opType: 'SiLU', inputs: { input: 'features' },
      outputs: { out: { tensor: 'output', shape, dtype: 'float32' } }, params: {},
    }],
    outputs: ['output'],
  })));
  const weights = safetensors({
    weight: { shape: [1, 1, 2, 3], values: [0.5, -1, 2, 1, 0.25, -0.5] },
  });
  const values = Float32Array.of(-1, -0.5, 0, 0.5, 1, 1.5);
  const input = new pb.Tensor({
    name: 'input', dtype: pb.DataType.DATA_TYPE_F32,
    shape: [1n, 6n], inline: new Uint8Array(values.buffer),
  });
  const host = new FullEngineHost({ wasmUrl: FULL_WASM });
  const inferenceHost = new EngineHost({ wasmUrl: INFERENCE_WASM });
  try {
    const quantization = new VxQuantizationServiceClient(host);
    const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config({scheme: pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC, floatOperators: ['SiLU']}), bytes: new pb.ModelBytes({graph: graph, weightShards: [weights]}), calibration: new pb.CalibrationOptions({profileNames: ['default']})})));
    assert.equal(plan.info.quantized.reduce((total, item) => total + item.nodes, 0n), 1n);
    await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
      quantizationPlanId: plan.quantizationPlanId, profileName: 'default',
      sampleName: 'sample', sampleCount: 1n, inputs: [input],
    }));
    const packed = await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest({ quantizationPlanId: plan.quantizationPlanId }));
    const reader = new VxInferenceServiceClient(inferenceHost);
    const readRuntime = await reader.createRuntime(new pb.CreateRuntimeRequest());
    const quantized = await reader.loadModel(new pb.LoadModelRequest({
      runtimeId: readRuntime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: packed.bytes.graph, weightShards: [packed.bytes.weightShards[0]] }),
    }));
    const compiled = await reader.compileModel(new pb.CompileModelRequest({
      modelId: quantized.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'] }),
    }));
    const context = await reader.createExecutionContext(new pb.CreateExecutionContextRequest({compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()}));
    const result = await reader.execute(new pb.ExecuteRequest({ contextId: context.contextId, inputs: new pb.InputBindings({values: [input] }), scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()})}));
    const output = result.outputs.find(tensor => tensor.name === 'output');
    assert.deepEqual(output.shape, shape.map(BigInt));
    const actual = new Float32Array(output.inline.slice().buffer);
    const weight = [0.5, -1, 2, 1, 0.25, -0.5];
    for (let row = 0; row < 3; row++) {
      for (let channel = 0; channel < 3; channel++) {
        const linear = values[row * 2] * weight[channel] + values[row * 2 + 1] * weight[3 + channel];
        const expected = linear / (1 + Math.exp(-linear));
        assert.ok(Math.abs(actual[row * 3 + channel] - expected) < 0.04);
      }
    }
  } finally {
    await inferenceHost.close();
    await host.close();
  }
});

test('static Conv4 PTQ publishes packed odd-width weights to both WASM profiles', async () => {
  const shape = [1, 2, 3, 3], channels = 3, width = 3;
  const graph = new TextEncoder().encode(JSON.stringify(statelessGraph({
    format: 'volvox-graph/v1', dimensions: {},
    inputs: {input: {shape, dtype: 'float32'}},
    nodes: [{id: 'conv', opType: 'Conv2D',
      inputs: {input: 'input', weight: 'weight', bias: 'bias'},
      outputs: {out: {tensor: 'output', shape, dtype: 'float32'}},
      params: {data_layout: 'NHWC', weight_layout: 'HWIO', stride: [1, 1]}}],
    outputs: ['output'],
  })));
  const weights = safetensors({
    weight: {shape: [1, 1, width, channels], values: [7, -3, 1, -2, 7, -1, .5, 1.25, -7]},
    bias: {shape: [channels], values: [.25, -.5, .125]},
  });
  const values = Float32Array.from({length: 18}, (_, i) => (i % 9 - 4) / 4);
  const inputs = protoTensors({input: {shape, data: values}});
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  const quantization = new VxQuantizationServiceClient(reportTransport(host));
  let published;
  try {
    const settings = config({bits: 4, selectedNodes: ['conv']});
    settings.weights.algorithm = pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP;
    // Apply the algorithm to the explicit rule as well as the default policy.
    for (const rule of settings.rules) if (rule.nodeId === 'conv')
      rule.weights.algorithm = pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP;
    const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({
      config: settings, bytes: new pb.ModelBytes({graph, weightShards: [weights]}),
    })));
    assert.equal(plan.info.requiresCalibration, true);
    assert.equal(plan.info.ready, false);
    checkedReport(await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
      quantizationPlanId: plan.quantizationPlanId, sampleName: 'conv4', sampleCount: 1n, inputs,
    })));
    const inspected = checkedReport(await quantization.inspectQuantizationPlan(new pb.QuantizationPlanRef(plan)));
    assert.equal(inspected.info.ready, true);
    published = checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan)));
    const repeated = checkedReport(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan)));
    assert.deepEqual(repeated.bytes.graph, published.bytes.graph);
    assert.deepEqual(repeated.bytes.weightShards[0], published.bytes.weightShards[0]);
    checkedReport(await quantization.releaseQuantizationPlan(new pb.QuantizationPlanRef(plan)));
  } finally { await host.close(); }

  const document = JSON.parse(new TextDecoder().decode(published.bytes.graph));
  const conv = document.nodes.find(node => node.opType === 'QConv2D');
  assert.ok(conv);
  assert.equal(conv.params.weight_bits, 4);
  assert.deepEqual(conv.params.kernel_size, [1, 1]);
  const bytes = published.bytes.weightShards[0];
  const headerLength = Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLength)));
  const entries = Object.fromEntries(Object.entries(header).filter(([name]) => name !== '__metadata__')
    .map(([name, entry]) => {
      const Type = {F32: Float32Array, I32: Int32Array, I8: Int8Array, U8: Uint8Array}[entry.dtype];
      const data = bytes.slice(8 + headerLength + entry.data_offsets[0], 8 + headerLength + entry.data_offsets[1]);
      return [name, {...entry, values: new Type(data.buffer)}];
    }));
  const packed = entries[conv.inputs.weight];
  assert.equal(packed.dtype, 'U8');
  assert.deepEqual(packed.shape, [channels, 2]);
  assert.equal(packed.values.length, channels * Math.ceil(width / 2));
  const decoded = new Int8Array(channels * width);
  for (let oc = 0; oc < channels; oc++) {
    assert.equal(packed.values[oc * 2 + 1] >>> 4, 8, 'unused high nibble is encoded zero');
    for (let k = 0; k < width; k++) decoded[oc * width + k] =
      ((packed.values[oc * 2 + Math.floor(k / 2)] >>> ((k % 2) * 4)) & 15) - 8;
  }
  const weightAffine = document.quantization.tensors[conv.inputs.weight];
  assert.deepEqual(Array.from(entries[weightAffine.zero_point_tensor].values), [8, 8, 8]);
  // An independent existing QConv8 consumer receives the exact centered codes
  // and scales; packing must not change arithmetic or published float outputs.
  const unpacked = structuredClone(document);
  const referenceConv = unpacked.nodes.find(node => node.opType === 'QConv2D');
  delete referenceConv.params.weight_bits;
  delete referenceConv.params.kernel_size;
  const referenceEntries = {...entries,
    [conv.inputs.weight]: {shape: [channels, 1, 1, width], dtype: 'I8', values: decoded},
    [weightAffine.zero_point_tensor]: {shape: [channels], dtype: 'I8', values: new Int8Array(channels)},
  };
  for (const full of [false, true]) {
    const reader = await protoFixture({full, wasmUrl: full ? FULL_WASM : INFERENCE_WASM});
    try {
      const outputs = [];
      for (const [modelGraph, modelWeights] of [[document, bytes], [unpacked, safetensors(referenceEntries)]]) {
        const model = await reader.load(modelGraph, modelWeights);
        const compiled = await reader.compile(model);
        const result = await reader.evaluate({compiledModelId: compiled.compiledModelId, inputs});
        try { outputs.push(await reader.read(result, 'output')); }
        finally { await reader.releaseOutputs(result); }
      }
      assert.deepEqual(outputs[0], outputs[1], `Conv4 publication matches centered I8 reference, full=${full}`);
    } finally { await reader.close(); }
  }
});

for (const profile of ['inference', 'full']) {
  test(`${profile} WASM spatial convolutions match an independent NHWC reference`, async () => {
    const host = profile === 'full'
      ? new FullEngineHost({ wasmUrl: FULL_WASM })
      : new EngineHost({ wasmUrl: INFERENCE_WASM });
    try {
      const client = new VxInferenceServiceClient(host);
      const runtime = await client.createRuntime(new pb.CreateRuntimeRequest({ cpuThreads: 1 }));
      for (const config of [
        { name: 'gray-stem', shape: [1, 7, 9, 1], kernel: [3, 3], stride: [2, 2], dilation: [1, 1], padding: [1, 0, 1, 2], channels: 16, groups: 1, bias: true },
        { name: 'channel-tail', shape: [2, 5, 7, 3], kernel: [2, 3], stride: [1, 2], dilation: [2, 1], padding: [0, 2, 1, 1], channels: 19, groups: 1, bias: false },
        { name: 'grouped', shape: [1, 6, 5, 4], kernel: [2, 2], stride: [2, 1], dilation: [1, 2], padding: [1, 1, 0, 1], channels: 6, groups: 2, bias: true },
      ]) {
        const { shape, kernel, stride, dilation, padding, channels, groups } = config;
        const [batch, height, width, inputChannels] = shape;
        const [kh, kw] = kernel, [sy, sx] = stride, [dy, dx] = dilation;
        const [top, left, bottom, right] = padding;
        const oh = Math.floor((height + top + bottom - (kh - 1) * dy - 1) / sy) + 1;
        const ow = Math.floor((width + left + right - (kw - 1) * dx - 1) / sx) + 1;
        const outputShape = [batch, oh, ow, channels], perGroup = inputChannels / groups;
        const input = Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => ((i * 7) % 19 - 9) / 11);
        const weight = Float32Array.from({ length: kh * kw * perGroup * channels }, (_, i) => ((i * 5) % 17 - 8) / 13);
        const bias = Float32Array.from({ length: channels }, (_, i) => config.bias ? (i - 4) / 17 : 0);
        const tensors = { weight: { shape: [kh, kw, perGroup, channels], values: weight } };
        if (config.bias) tensors.bias = { shape: [channels], values: bias };
        const graph = new TextEncoder().encode(JSON.stringify(statelessGraph({
          format: 'volvox-graph/v1', dimensions: {},
          inputs: { input: { shape, dtype: 'float32' } },
          nodes: [{ id: 'conv', opType: 'Conv2D',
            inputs: { input: 'input', weight: 'weight', ...(config.bias ? { bias: 'bias' } : {}) },
            outputs: { out: { tensor: 'output', shape: outputShape, dtype: 'float32' } },
            params: { data_layout: 'NHWC', weight_layout: 'HWIO', stride, dilation,
              pads: padding, groups } }],
          outputs: ['output'],
        })));
        const model = await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
          package: new pb.ModelPackage({ graphDocument: graph, weightShards: [safetensors(tensors)] }) }));
        const compiled = await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
          policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
            backends: ['wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) }));
        const context = await client.createExecutionContext(new pb.CreateExecutionContextRequest({compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()}));
        const result = await client.execute(new pb.ExecuteRequest({ contextId: context.contextId,
          inputs: new pb.InputBindings({values: [new pb.Tensor({ name: 'input', shape: shape.map(BigInt), dtype: pb.DataType.DATA_TYPE_F32,
            inline: new Uint8Array(input.buffer) })] }), scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()})}));
        try {
          assert.equal(result.report.route.provider, 'wasm');
          assert.equal(result.report.route.fallbackNodes, 0);
          const output = result.outputs.find(tensor => tensor.name === 'output');
          assert.deepEqual(output.shape, outputShape.map(BigInt));
          const actual = new Float32Array(output.inline.slice().buffer);
          for (let n = 0; n < batch; n++) for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) for (let oc = 0; oc < channels; oc++) {
            let expected = bias[oc];
            const firstChannel = Math.floor(oc / (channels / groups)) * perGroup;
            for (let ky = 0; ky < kh; ky++) for (let kx = 0; kx < kw; kx++) {
              const iy = y * sy - top + ky * dy, ix = x * sx - left + kx * dx;
              if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
              for (let ci = 0; ci < perGroup; ci++) expected +=
                input[((n * height + iy) * width + ix) * inputChannels + firstChannel + ci] *
                weight[((ky * kw + kx) * perGroup + ci) * channels + oc];
            }
            const index = ((n * oh + y) * ow + x) * channels + oc;
            assert.ok(Math.abs(actual[index] - expected) < 2e-5,
              `${config.name}[${index}]: ${actual[index]} vs ${expected}`);
          }
        } finally {
          await client.releaseExecutionContext(new pb.ExecutionContextRef(context));
        }
      }
    } finally {
      await host.close();
    }
  });
}

for (const profile of ['inference', 'full']) {
  test(`${profile} WASM packed integer dispatch preserves affine quantization exactly`, async () => {
    const host = profile === 'full'
      ? new FullEngineHost({ wasmUrl: FULL_WASM })
      : new EngineHost({ wasmUrl: INFERENCE_WASM });
    try {
      const client = new VxInferenceServiceClient(host);
      const runtime = await client.createRuntime(new pb.CreateRuntimeRequest({ cpuThreads: 1 }));
      for (const config of [
        { name: 'conv-padding', conv: true, shape: [1, 5, 6, 3], outputShape: [1, 2, 3, 16], k: 27, n: 16, signed: false, asymmetricWeight: false },
        { name: 'conv4-padding', conv: true, bits: 4, shape: [1, 5, 6, 3], outputShape: [1, 2, 3, 16], k: 27, n: 16, signed: false, asymmetricWeight: false },
        { name: 'gemv-affine', conv: false, shape: [1, 17], outputShape: [1, 32], k: 17, n: 32, signed: true, asymmetricWeight: true },
        { name: 'gemm-tails', conv: false, shape: [1, 3, 37], outputShape: [1, 3, 19], k: 37, n: 19, signed: false, asymmetricWeight: false },
      ]) {
        const { shape, outputShape, k, n } = config;
        const InputArray = config.signed ? Int8Array : Uint8Array;
        const OutputArray = config.signed ? Uint8Array : Int8Array;
        const inputZero = config.signed ? -11 : 131, outputZero = config.signed ? 123 : -3;
        const input = InputArray.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => inputZero + ((i * 13) % 41) - 20);
        if (!config.signed) input[7] = 250;
        const weight = Int8Array.from({ length: k * n }, (_, i) => (i * 7) % (config.bits === 4 ? 16 : 17) - 8);
        const bias = Int32Array.from({ length: n }, (_, i) => (i * 23) % 129 - 64);
        const weightScale = Float32Array.from({ length: n }, (_, i) => 2 ** (-5 + i % 3));
        const weightZero = Int8Array.from({ length: n }, (_, i) => config.asymmetricWeight ? i % 3 - 1 : 0);
        const tensors = {
          weight: { shape: config.conv ? [n, 3, 3, 3] : [n, k], values: weight, dtype: 'I8' },
          bias: { shape: [n], values: bias, dtype: 'I32' },
          'input.scale': { shape: [1], values: [0.125] },
          'input.zero': { shape: [1], values: [inputZero], dtype: config.signed ? 'I8' : 'U8' },
          'weight.scale': { shape: [n], values: weightScale },
          'weight.zero': { shape: [n], values: weightZero, dtype: 'I8' },
          'output.scale': { shape: [1], values: [0.25] },
          'output.zero': { shape: [1], values: [outputZero], dtype: config.signed ? 'U8' : 'I8' },
        };
        if (config.bits === 4) {
          const rowBytes = Math.ceil(k / 2), packed = new Uint8Array(n * rowBytes).fill(0x88);
          for (let oc = 0; oc < n; oc++) for (let term = 0; term < k; term++) {
            const index = oc * rowBytes + Math.floor(term / 2), shift = (term % 2) * 4;
            packed[index] = (packed[index] & ~(15 << shift)) | ((weight[oc * k + term] + 8) << shift);
          }
          tensors.weight = {shape: [n, rowBytes], values: packed, dtype: 'U8'};
          tensors['weight.zero'] = {shape: [n], values: new Uint8Array(n).fill(8), dtype: 'U8'};
        }
        const quantization = Object.fromEntries(['input', 'weight', 'output'].map(name => [name, {
          scheme: name === 'weight' ? 'per_axis' : 'per_tensor',
          ...(name === 'weight' ? { axis: 0 } : {}),
          scale_tensor: `${name}.scale`, zero_point_tensor: `${name}.zero`,
        }]));
        const graph = new TextEncoder().encode(JSON.stringify(statelessGraph({
          format: 'volvox-graph/v1', dimensions: {},
          inputs: { input: { shape, dtype: config.signed ? 'int8' : 'uint8' } },
          nodes: [{ id: 'quantized', opType: config.conv ? 'QConv2D' : 'QLinear',
            inputs: { input: 'input', weight: 'weight', bias: 'bias' },
            outputs: { out: { tensor: 'output', shape: outputShape, dtype: config.signed ? 'uint8' : 'int8' } },
            params: config.conv ? { data_layout: 'NHWC', weight_layout: 'OHWI',
              stride: [2, 2], dilation: [1, 1], pads: [1, 0, 0, 1], groups: 1, relu: 0,
              ...(config.bits === 4 ? {weight_bits: 4, kernel_size: [3, 3]} : {}) }
              : {} }],
          outputs: ['output'], quantization: { format: 'volvox-affine-safetensors/v1', tensors: quantization },
        })));
        const model = await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
          package: new pb.ModelPackage({ graphDocument: graph, weightShards: [safetensors(tensors)] }) }));
        const compiled = await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
          policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
            backends: ['wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) }))
          .catch(error => { error.message = `${config.name}: ${error.message}`; throw error; });
        const context = await client.createExecutionContext(new pb.CreateExecutionContextRequest({compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()}));
        const result = await client.execute(new pb.ExecuteRequest({ contextId: context.contextId,
          inputs: new pb.InputBindings({values: [new pb.Tensor({ name: 'input', shape: shape.map(BigInt),
            dtype: config.signed ? pb.DataType.DATA_TYPE_I8 : pb.DataType.DATA_TYPE_U8,
            inline: new Uint8Array(input.buffer) })] }), scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()})}));
        try {
          assert.equal(result.report.route.provider, 'wasm');
          assert.equal(result.report.route.fallbackNodes, 0);
          const output = result.outputs.find(tensor => tensor.name === 'output');
          assert.deepEqual(output.shape, outputShape.map(BigInt));
          const actual = new OutputArray(output.inline.slice().buffer);
          const expected = new OutputArray(actual.length);
          for (let row = 0; row < actual.length / n; row++) for (let oc = 0; oc < n; oc++) {
            let accumulator = bias[oc];
            for (let term = 0; term < k; term++) {
              let value;
              if (config.conv) {
                const ky = Math.floor(term / 9), kx = Math.floor(term / 3) % 3, ci = term % 3;
                const y = Math.floor(row / 3) * 2 - 1 + ky, x = row % 3 * 2 + kx;
                value = y < 0 || y >= 5 || x >= 6 ? inputZero : input[(y * 6 + x) * 3 + ci];
              } else value = input[row * k + term];
              accumulator += (value - inputZero) * (weight[oc * k + term] - weightZero[oc]);
            }
            const transformed = accumulator * (0.125 * weightScale[oc] / 0.25) + outputZero;
            const lower = Math.floor(transformed), fraction = transformed - lower;
            const rounded = lower + (fraction > 0.5 || (fraction === 0.5 && lower % 2 !== 0) ? 1 : 0);
            expected[row * n + oc] = Math.min(config.signed ? 255 : 127, Math.max(config.signed ? 0 : -128, rounded));
          }
          assert.deepEqual(actual, expected, config.name);
        } finally {
          await client.releaseExecutionContext(new pb.ExecutionContextRef(context));
        }
      }
    } finally {
      await host.close();
    }
  });
}

test('static quantization owns calibration and returns a reusable package snapshot', async () => {
  const { graph, weights } = fixture();
  const files = new Map([['graph.json', graph], ['weights.safetensors', weights]]);
  const host = new FullEngineHost({
    wasmUrl: FULL_WASM,
    fetch: async (source) => ({ ok: files.has(source), arrayBuffer: async () => files.get(source).slice().buffer }),
  });
  try {
    const inference = new VxInferenceServiceClient(reportTransport(host));
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    const success = (value) => { assert.equal(value.report?.status, 0, value.report?.message); return value; };
    const runtime = success(await inference.createRuntime(new pb.CreateRuntimeRequest()));
    const model = success(await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId, graphPath: 'graph.json', weightPaths: ['weights.safetensors'] })));
    const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config(), bytes: new pb.ModelBytes({graph: graph, weightShards: [weights]}), calibration: new pb.CalibrationOptions({profileNames: ['default']})})));
    assert.equal(plan.info.requiresCalibration, true);
    assert.equal(plan.info.ready, false);
    // Calibration uses the plan's snapshot after the caller reuses its input buffers.
    graph.fill(0);
    weights.fill(0);
    const incomplete = await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest({ quantizationPlanId: plan.quantizationPlanId }));
    assert.equal(incomplete.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
    assert.equal(incomplete.bytes?.graph?.length ?? 0, 0);
    success(await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
      quantizationPlanId: plan.quantizationPlanId, profileName: 'default', sampleName: 'one', sampleCount: 1n,
      inputs: [new pb.Tensor({ name: 'input', dtype: pb.DataType.DATA_TYPE_F32, shape: [1n, 2n], inline: new Uint8Array(Float32Array.of(1, -0.5).buffer) })],
    })));
    const pathExport = await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest({quantizationPlanId: plan.quantizationPlanId, outputFiles: new pb.ModelFiles({graphPath: 'graph.json', weightPaths: ['model.safetensors']})}));
    assert.equal(pathExport.report.status, pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
    assert.equal(pathExport.bytes?.graph?.length ?? 0, 0);
    assert.equal(pathExport.bytes?.weightShards?.[0]?.length ?? 0, 0);
    const packed = success(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest({ quantizationPlanId: plan.quantizationPlanId })));
    assert.equal(packed.plan.ready, true);
    assert.equal(packed.plan.calibration.coverage.complete, true);
    assert.ok(packed.bytes.graph.length > 0 && packed.bytes.weightShards[0].length > 0);
    const snapshot = packed.bytes.graph.slice();
    const repeated = success(await quantization.exportQuantizedModel(new pb.ExportQuantizedModelRequest({ quantizationPlanId: plan.quantizationPlanId })));
    assert.deepEqual(repeated.bytes.graph, snapshot);
    assert.deepEqual(repeated.bytes.weightShards[0], packed.bytes.weightShards[0]);
    assert.equal((await quantization.releaseQuantizationPlan(new pb.QuantizationPlanRef({ quantizationPlanId: plan.quantizationPlanId }))).status, 0);
    assert.deepEqual(packed.bytes.graph, snapshot);
    files.set('quantized/graph.json', packed.bytes.graph);
    files.set('quantized/weights.safetensors', packed.bytes.weightShards[0]);
    const quantized = success(await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId, graphPath: 'quantized/graph.json', weightPaths: ['quantized/weights.safetensors'] })));
    const compiled = success(await inference.compileModel(new pb.CompileModelRequest({ modelId: quantized.modelId })));
    assert.ok(compiled.compiledModelId > 0n);
    const execute = async compiledModelId => {
      const context = await inference.createExecutionContext(new pb.CreateExecutionContextRequest({compiledModelId: compiledModelId, graph: new pb.GraphExecution()}));
      const result = success(await inference.execute(new pb.ExecuteRequest({contextId: context.contextId,
        inputs: new pb.InputBindings({values: [new pb.Tensor({name: 'input', dtype: pb.DataType.DATA_TYPE_F32,
          shape: [1n,2n], inline: new Uint8Array(Float32Array.of(1,-0.5).buffer)})]}), scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()})})));
      try {
        const tensor = result.outputs.find(tensor => tensor.name === 'output');
        assert.equal(tensor.dtype, pb.DataType.DATA_TYPE_F32);
        return new Float32Array(tensor.inline.slice().buffer);
      } finally { assert.equal((await inference.releaseExecutionContext(new pb.ExecutionContextRef(context))).status, 0); }
    };
    const reference = success(await inference.compileModel(new pb.CompileModelRequest({modelId: model.modelId})));
    const expected = await execute(reference.compiledModelId), actual = await execute(compiled.compiledModelId);
    for (let index = 0; index < expected.length; index++)
      assert.ok(Math.abs(actual[index] - expected[index]) < 0.03,
        `PTQ output ${index}: ${actual[index]} versus ${expected[index]}`);
  } finally { await host.close(); }
});

test('quantization inspection reports intermediate activation observations', async () => {
  const graph = statelessGraph({format: 'volvox-graph/v1', dimensions: {},
    inputs: {x: {dtype: 'float32', shape: [1,2]}}, nodes: [{id:'dense',opType:'Linear',
      inputs:{input:'x',weight:'w'},outputs:{out:{tensor:'y',dtype:'float32',shape:[1,2]}},
      params:{weight_layout:'din_dout'}}], outputs:['y']});
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    const plan = checkedPlan(await quantization.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({config: config(), bytes: new pb.ModelBytes({graph: new TextEncoder().encode(JSON.stringify(graph)), weightShards: [protoSafetensors([{name:'w',shape:[2,2],data:Float32Array.of(1,0,0,2)}])]}), calibration: new pb.CalibrationOptions({profileNames: ['observations']})})));
    for (const [index, data] of [Float32Array.of(-1,3), Float32Array.of(4,-2)].entries()) {
      checkedReport(await quantization.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
        quantizationPlanId: plan.quantizationPlanId, profileName: 'observations', sampleName: String(index),
        sampleCount: 1n, inputs: protoTensors({x: {data, shape: [1,2]}}),
      })));
    }
    const info = (checkedReport(await quantization.inspectQuantizationPlan(new pb.QuantizationPlanRef(plan)))).info;
    assert.equal(info.ready, true);
    assert.equal(info.calibration.samples, 2n);
    const observations = Object.fromEntries(info.calibration.affines.map(tensor => [tensor.tensorName,
      {min: tensor.observedMin, max: tensor.observedMax, elements: Number(tensor.observedValues)}]));
    assert.deepEqual(observations.x, {min:-2,max:4,elements:4});
    assert.deepEqual(observations.y, {min:-4,max:6,elements:4});
    checkedReport(await quantization.releaseQuantizationPlan(new pb.QuantizationPlanRef(plan)));
  } finally {await host.close();}
});

test('calibration tooling reads arbitrary intermediate ranges through C debug statistics', async () => {
  const graph = statelessGraph({format: 'volvox-graph/v1', dimensions: {},
    inputs: {x: {dtype: 'float32', shape: [1,2]}}, nodes: [{id:'dense',opType:'MatMul',
      inputs:{input:'x',weight:'w'},outputs:{out:{tensor:'y',dtype:'float32',shape:[1,2]}},params:{}}], outputs:['y']});
  const f = await protoFixture({full: true, wasmUrl: FULL_WASM});
  try {
    const model = await f.load(graph, protoSafetensors([{name:'w',shape:[2,2],data:Float32Array.of(1,0,0,2)}]));
    const observations = {};
    const observed = await observeActivations({host: f.host, modelId: model.modelId, document: graph,
      names: ['x', 'y'], observations, inputSets: [
        {x: {data: Float32Array.of(-1,3), shape: [1,2]}},
        {x: {data: Float32Array.of(4,-2), shape: [1,2]}},
      ]});
    assert.equal(observed, 2);
    assert.deepEqual(observations.x, {min:-2,max:4,samples:2,elements:4});
    assert.deepEqual(observations.y, {min:-4,max:6,samples:2,elements:4});
  } finally {await f.close();}
});

test('grouped weight fitting preserves finite F32 reconstruction at extreme magnitudes', async () => {
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    const words = new Uint32Array(128).fill(0x7f028f5b);
    words[0] = 0xff7fffff;
    const original = new Float32Array(words.buffer);
    for (const bits of [4, 8]) {
      const errors = [];
      for (const algorithm of [pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN,
        pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP]) {
        const packed = checkedReport(await quantization.quantizeWeight(new pb.QuantizeWeightRequest({
          weight: new pb.Tensor({name: 'extreme', shape: [1n, 128n],
            dtype: pb.DataType.DATA_TYPE_F32, inline: new Uint8Array(words.buffer)}),
          quantization: new pb.WeightQuantization({format: new pb.WeightFormat({bits, groupSize: 128}), algorithm}),
        })));
        const restored = checkedReport(await quantization.dequantizeWeight(new pb.DequantizeWeightRequest({weight: packed.weight})));
        const values = new Float32Array(restored.tensor.inline.slice().buffer);
        assert.ok(values.every(Number.isFinite), `W${bits}/${pb.WeightAlgorithm[algorithm]}`);
        errors.push(values.reduce((error, value, index) => error + (value - original[index]) ** 2, 0));
      }
      assert.ok(errors[1] <= errors[0], `W${bits} MSE fitting cannot worsen actual F32 reconstruction`);
    }
  } finally { await host.close(); }
});

test('grouped tensor conversion preserves every supported group and partial word', async () => {
  const host = new FullEngineHost({wasmUrl: FULL_WASM});
  try {
    const quantization = new VxQuantizationServiceClient(reportTransport(host));
    for (const groupSize of [32, 64, 128, 256]) for (const bits of [4, 8]) {
      for (const width of [1, 7, groupSize - 1, groupSize, groupSize + 1, 2 * groupSize + 3]) {
        const groups = Math.ceil(width / groupSize), rows = 2;
        const scales = Array.from({length: rows * groups}, (_, index) => (index + 1) / 8);
        const original = Float32Array.from({length: rows * width}, (_, index) => {
          const row = Math.floor(index / width), column = index % width;
          return ((column % 15) - 7) * scales[row * groups + Math.floor(column / groupSize)];
        });
        const converted = checkedReport(await quantization.quantizeWeight(new pb.QuantizeWeightRequest({
          weight: new pb.Tensor({name: 'groups', shape: [BigInt(rows), BigInt(width)],
            dtype: pb.DataType.DATA_TYPE_F32, inline: new Uint8Array(original.buffer)}),
          quantization: new pb.WeightQuantization({format: new pb.WeightFormat({bits, groupSize})}),
          preserveScales: scales,
        })));
        assert.equal(converted.weight.format.groupSize, groupSize);
        assert.deepEqual(converted.weight.scales.shape, [BigInt(rows), BigInt(groups)]);
        const restored = checkedReport(await quantization.dequantizeWeight(
          new pb.DequantizeWeightRequest({weight: converted.weight})));
        assert.deepEqual(new Float32Array(restored.tensor.inline.slice().buffer), original,
          `G${groupSize}/W${bits}/K${width}`);
        if (width > groupSize && groupSize !== 32) {
          converted.weight.format.groupSize = 32;
          const rejected = await quantization.dequantizeWeight(new pb.DequantizeWeightRequest({weight: converted.weight}));
          assert.notEqual(rejected.report.status, pb.NativeStatus.NATIVE_STATUS_OK);
        }
      }
    }
  } finally { await host.close(); }
});
