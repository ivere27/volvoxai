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
  path.join(ROOT, 'dist', PACKAGE_VERSION, 'volvoxai.full.wasm');
const INFERENCE_WASM = path.join(
  ROOT, 'dist', PACKAGE_VERSION, 'volvoxai.wasm',
);
function safetensors(entries) {
  const header = {};
  const payloads = [];
  let offset = 0;
  for (const [name, { shape, values }] of Object.entries(entries)) {
    const payload = new Uint8Array(Float32Array.from(values).buffer);
    header[name] = { dtype: 'F32', shape, data_offsets: [offset, offset + payload.byteLength] };
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
  const graph = new TextEncoder().encode(JSON.stringify({
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
  }, null, 1));
  const weights = safetensors({
    'dense.weight': { shape: [2, 2], values: [1, 2, 3, 4] },
    'dense.bias': { shape: [2], values: [0.25, -0.5] },
  });
  return { graph, weights };
}

test('inference EngineHost keeps PTQ authoring unregistered', async () => {
  const quantization = new VxQuantizationServiceClient(
    reportTransport(new EngineHost({ wasmUrl: INFERENCE_WASM })),
  );
  await assert.rejects(
    (quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({
      sourceGraph: new TextEncoder().encode('{}'),
    }))),
    error => error.name === 'RpcError' && error.code === 12,
  );
});

test('FullEngineHost validates every typed storage decision before WASM', async () => {
  const quantization = new VxQuantizationServiceClient(
    reportTransport(new FullEngineHost({ wasmUrl: FULL_WASM })),
  );
  const invalidActivation = await quantization.authorPtqTemplate(
    new pb.AuthorPtqTemplateRequest({
      config: new pb.PtqAuthoringConfig({ activationDtype: pb.DataType.DATA_TYPE_F32 }),
    }),
  );
  assert.equal(invalidActivation.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  assert.equal(invalidActivation.report?.stage, pb.OperationStage.OPERATION_STAGE_PTQ_AUTHOR);

  const invalidWeight = await quantization.authorPtqTemplate(
    new pb.AuthorPtqTemplateRequest({
      config: new pb.PtqAuthoringConfig({ weightDtype: pb.DataType.DATA_TYPE_U8 }),
    }),
  );
  assert.equal(invalidWeight.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  assert.equal(invalidWeight.report?.stage, pb.OperationStage.OPERATION_STAGE_PTQ_AUTHOR);

  const invalidScheme = await quantization.authorPtqTemplate(
    new pb.AuthorPtqTemplateRequest({
      config: new pb.PtqAuthoringConfig({ activationScheme: 99 }),
    }),
  );
  assert.equal(invalidScheme.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  assert.equal(invalidScheme.report?.stage, pb.OperationStage.OPERATION_STAGE_PTQ_AUTHOR);
});

test('generated quantization client receives the complete authored plan', async () => {
  const { graph, weights } = fixture();
  const host = new FullEngineHost({ wasmUrl: FULL_WASM });
  const platform = new VxPlatformServiceClient(reportTransport(host));
  const quantization = new VxQuantizationServiceClient(reportTransport(host));

  const platformInfo = await platform.getPlatformInfo(new pb.Empty());
  assert.equal(platformInfo.profile, pb.BuildProfile.BUILD_PROFILE_FULL);

  const info = await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({
    // Bytes win over the competing input path fields. The output path is an
    // independent request and is covered by the transport test below.
    sourceGraphPath: 'must-not-be-opened.graph.json',
    weightPaths: ['must-not-be-opened.safetensors'],
    sourceGraph: graph,
    weightShards: [weights],
    config: new pb.PtqAuthoringConfig({
      activationDtype: pb.DataType.DATA_TYPE_U8,
      activationScheme: pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC,
      weightDtype: pb.DataType.DATA_TYPE_I8,
      reduceRange: true,
      floatOperators: [],
      floatNodes: [],
      selectedNodes: ['dense'],
    }),
  }));

  assert.equal(info.report?.status, pb.NativeStatus.NATIVE_STATUS_OK,
    info.report?.message);
  assert.equal(info.report?.stage, pb.OperationStage.OPERATION_STAGE_PTQ_AUTHOR);
  assert.equal(info.quantizedNodes, 1n);
  assert.equal(info.retainedFloatNodes, 1n);
  assert.ok(info.templateGraph?.byteLength);
  assert.deepEqual(info.requiredObservations,
    info.observers.map((observer) => observer.tensorName));
  assert.ok(info.observers.length > 0);
  assert.ok(info.observers.every((observer) =>
    observer.dtype === pb.DataType.DATA_TYPE_U8 &&
    observer.scheme === pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC &&
    observer.quantizedTensorName.length > 0));

  assert.equal(info.layers.length, 1);
  const layer = info.layers[0];
  assert.equal(layer.mode, pb.PtqMode.PTQ_MODE_W8A8);
  assert.equal(layer.kind, pb.PtqLayerKind.PTQ_LAYER_KIND_QLINEAR);
  assert.equal(layer.nodeIndex, 0);
  assert.equal(layer.weightAxis, 0);
  assert.equal(layer.nodeId, 'dense');
  assert.equal(layer.inputTensorName, 'input');
  assert.equal(layer.outputTensorName, 'hidden');
  assert.equal(layer.sourceWeightName, 'dense.weight');
  assert.match(layer.packedWeightName, /^__ptq__\.[0-9a-f]{20}\.weight$/);
  assert.equal(layer.sourceBiasName, 'dense.bias');
  assert.match(layer.packedBiasName, /^__ptq__\.[0-9a-f]{20}\.bias$/);
});

test('browser authoring refuses every filesystem output request', async () => {
  const { graph, weights } = fixture();
  const quantization = new VxQuantizationServiceClient(
    reportTransport(new FullEngineHost({ wasmUrl: FULL_WASM })),
  );
  const info = await quantization.authorPtqTemplate(
    new pb.AuthorPtqTemplateRequest({
      sourceGraph: graph,
      weightShards: [weights],
      templateGraphPath: 'must-not-be-written.graph.json',
    }),
  );
  assert.equal(
    info.report?.status,
    pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED,
  );
  assert.equal(info.report?.code, pb.OperationCode.OPERATION_CODE_TRANSPORT_UNSUPPORTED);
  assert.equal(info.templateGraph?.byteLength ?? 0, 0);
});

test('all three PTQ selection lists reach the C authoring config', async () => {
  const { graph, weights } = fixture();
  const quantization = new VxQuantizationServiceClient(
    reportTransport(new FullEngineHost({ wasmUrl: FULL_WASM })),
  );
  for (const { config, quantizedNodes, retainedFloatNodes, layers } of [
    {
      config: { floatOperators: ['GELU'] },
      quantizedNodes: 1n,
      retainedFloatNodes: 1n,
      layers: 1,
    },
    {
      config: { floatNodes: ['dense'] },
      quantizedNodes: 1n,
      retainedFloatNodes: 1n,
      layers: 0,
    },
    {
      config: { selectedNodes: ['dense'] },
      quantizedNodes: 1n,
      retainedFloatNodes: 1n,
      layers: 1,
    },
  ]) {
    const info = await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({
      sourceGraph: graph,
      weightShards: [weights],
      config: new pb.PtqAuthoringConfig(config),
    }));
    assert.equal(info.report?.status, pb.NativeStatus.NATIVE_STATUS_OK,
      info.report?.message);
    assert.equal(info.quantizedNodes, quantizedNodes);
    assert.equal(info.retainedFloatNodes, retainedFloatNodes);
    assert.equal(info.layers.length, layers);
  }
});

test('authoring failures stay in PtqTemplateInfo.report', async () => {
  const { weights } = fixture();
  const quantization = new VxQuantizationServiceClient(
    reportTransport(new FullEngineHost({ wasmUrl: FULL_WASM })),
  );
  const info = await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({
    sourceGraph: new TextEncoder().encode('{not json'),
    weightShards: [weights],
  }));
  assert.equal(info.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_GRAPH);
  assert.equal(info.report?.stage, pb.OperationStage.OPERATION_STAGE_PTQ_AUTHOR);
  assert.match(info.report?.message ?? '', /JSON/i);
});

test('structural name limits are INVALID_GRAPH rather than OOM', async () => {
  const input = 'x'.repeat(128);
  const graph = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { [input]: { shape: [1], dtype: 'float32' } },
    nodes: [{
      id: 'gelu',
      opType: 'GELU',
      inputs: { input },
      outputs: {
        out: { tensor: 'output', shape: [1], dtype: 'float32' },
      },
      params: {},
    }],
    outputs: ['output'],
  }));
  const quantization = new VxQuantizationServiceClient(
    reportTransport(new FullEngineHost({ wasmUrl: FULL_WASM })),
  );
  const info = await quantization.authorPtqTemplate(
    new pb.AuthorPtqTemplateRequest({ sourceGraph: graph }),
  );
  assert.equal(info.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_GRAPH);
  assert.equal(info.report?.code, pb.OperationCode.OPERATION_CODE_INVALID_GRAPH);
  assert.match(info.report?.message ?? '', /tensor name.*limit/i);
});

test('WASM authoring uses byte transport and refuses private path access', async () => {
  const quantization = new VxQuantizationServiceClient(reportTransport(new FullEngineHost({ wasmUrl: FULL_WASM })));
  const result = await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({ sourceGraphPath: 'graph.json' }));
  assert.equal(result.report?.status, pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
});

test('missing and incompatible sidecars reject the transport call', async () => {
  const request = new pb.AuthorPtqTemplateRequest({ sourceGraph: new TextEncoder().encode('{}') });
  const missing = new VxQuantizationServiceClient(reportTransport(new FullEngineHost({ wasmUrl: path.join(ROOT, 'missing-ptq.wasm') })));
  await assert.rejects((missing.authorPtqTemplate(request)), /ENOENT/);
  const incompatible = new VxQuantizationServiceClient(reportTransport(new FullEngineHost({ wasmUrl: INFERENCE_WASM })));
  await assert.rejects((incompatible.authorPtqTemplate(request)), /imports do not match the full profile/);
});

test('C PTQ calibrates a retained model and returns a reusable package snapshot', async () => {
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
    const authored = success(await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({ sourceGraph: graph, weightShards: [weights] })));
    const plan = success(await quantization.createPtqPlan(new pb.CreatePtqPlanRequest({
      modelId: model.modelId, templateGraph: authored.templateGraph,
      profileNames: ['default'], observers: authored.observers, layers: authored.layers,
    })));
    assert.equal(plan.report.lineage.modelId, model.modelId);
    assert.equal(plan.report.lineage.runtimeId, runtime.runtimeId);
    const incomplete = await quantization.writePtqPackage(new pb.WritePtqPackageRequest({ ptqPlanId: plan.ptqPlanId }));
    assert.equal(incomplete.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
    assert.equal(incomplete.graph?.length ?? 0, 0);
    success(await quantization.calibratePtqPlan(new pb.CalibratePtqPlanRequest({
      ptqPlanId: plan.ptqPlanId, profileName: 'default', sampleName: 'one', sampleCount: 1n,
      inputs: [new pb.Tensor({ name: 'input', dtype: pb.DataType.DATA_TYPE_F32, shape: [1n, 2n], inline: new Uint8Array(Float32Array.of(1, -0.5).buffer) })],
    })));
    const pathExport = await quantization.writePtqPackage(new pb.WritePtqPackageRequest({
      ptqPlanId: plan.ptqPlanId, outputGraphPath: 'graph.json', outputWeightsPath: 'model.safetensors',
    }));
    assert.equal(pathExport.report.status, pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
    assert.equal(pathExport.graph?.length ?? 0, 0);
    assert.equal(pathExport.weights?.length ?? 0, 0);
    const packed = success(await quantization.writePtqPackage(new pb.WritePtqPackageRequest({ ptqPlanId: plan.ptqPlanId })));
    assert.equal(packed.report.lineage.modelId, model.modelId);
    assert.equal(packed.report.lineage.runtimeId, runtime.runtimeId);
    assert.ok(packed.graph.length > 0 && packed.weights.length > 0);
    const snapshot = packed.graph.slice();
    const repeated = success(await quantization.writePtqPackage(new pb.WritePtqPackageRequest({ ptqPlanId: plan.ptqPlanId })));
    assert.deepEqual(repeated.graph, snapshot);
    assert.deepEqual(repeated.weights, packed.weights);
    assert.equal((await quantization.releasePtqPlan(new pb.PtqPlanRef({ ptqPlanId: plan.ptqPlanId }))).status, 0);
    assert.deepEqual(packed.graph, snapshot);
    files.set('quantized/graph.json', packed.graph);
    files.set('quantized/weights.safetensors', packed.weights);
    const quantized = success(await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId, graphPath: 'quantized/graph.json', weightPaths: ['quantized/weights.safetensors'] })));
    const compiled = success(await inference.compileModel(new pb.CompileModelRequest({ modelId: quantized.modelId })));
    assert.ok(compiled.compiledModelId > 0n);
    const execute = async compiledModelId => {
      const result = success(await inference.run(new pb.RunRequest({compiledModelId,
        inputs: [new pb.Tensor({name: 'input', dtype: pb.DataType.DATA_TYPE_F32,
          shape: [1n,2n], inline: new Uint8Array(Float32Array.of(1,-0.5).buffer)})]})));
      try {
        const tensor = success(await inference.readOutput(new pb.ReadOutputRequest({
          resultId: result.resultId, name: 'output'}))).tensor;
        assert.equal(tensor.dtype, pb.DataType.DATA_TYPE_F32);
        return new Float32Array(tensor.inline.slice().buffer);
      } finally { assert.equal((await inference.releaseResult(new pb.ResultRef(result))).status, 0); }
    };
    const reference = success(await inference.compileModel(new pb.CompileModelRequest({modelId: model.modelId})));
    const expected = await execute(reference.compiledModelId), actual = await execute(compiled.compiledModelId);
    for (let index = 0; index < expected.length; index++)
      assert.ok(Math.abs(actual[index] - expected[index]) < 0.03,
        `PTQ output ${index}: ${actual[index]} versus ${expected[index]}`);
  } finally { await host.close(); }
});
