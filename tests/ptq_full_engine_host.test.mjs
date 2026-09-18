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

test('C PTQ transposes din_dout Linear weights and preserves dynamic rows', async () => {
  const graph = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1', dimensions: { S: { min: 1, max: 3 }, source: { min: 3, max: 3 } }, banks: { weight: 'source' },
    inputs: { input: { shape: [1, 'S', 3], dtype: 'float32' } },
    nodes: [{ id: 'dense', opType: 'Linear', inputs: { input: 'input', weight: 'weight' },
      outputs: { out: { tensor: 'output', shape: [1, 'S', 2], dtype: 'float32' } },
      params: { weight_layout: 'din_dout' } }], outputs: ['output'],
  }));
  const weights = safetensors({ weight: { shape: [3, 2], values: [.25, 2, -.75, .5, 1.5, -1] } });
  const values = Float32Array.of(-1, .5, 1, .25, -.75, .5, 0, 1, -1);
  const input = rows => new pb.Tensor({ name: 'input', dtype: pb.DataType.DATA_TYPE_F32,
    shape: [1n, BigInt(rows), 3n], inline: new Uint8Array(values.buffer, 0, rows * 3 * 4) });
  const host = new FullEngineHost({ wasmUrl: FULL_WASM });
  const readerHost = new EngineHost({ wasmUrl: INFERENCE_WASM });
  try {
    const inference = new VxInferenceServiceClient(host), quantization = new VxQuantizationServiceClient(host);
    const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
    const model = await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: graph, weightShards: [weights] }) }));
    const authored = await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({
      sourceGraph: graph, weightShards: [weights], config: new pb.PtqAuthoringConfig({
        activationDtype: pb.DataType.DATA_TYPE_I8, activationScheme: pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC }) }));
    const plan = await quantization.createPtqPlan(new pb.CreatePtqPlanRequest({ modelId: model.modelId,
      templateGraph: authored.templateGraph, observers: authored.observers, layers: authored.layers, profileNames: ['dynamic'] }));
    for (const rows of [1, 3, 2]) await quantization.calibratePtqPlan(new pb.CalibratePtqPlanRequest({
      ptqPlanId: plan.ptqPlanId, profileName: 'dynamic', sampleName: `rows-${rows}`, sampleCount: 1n, inputs: [input(rows)] }));
    const state = await quantization.inspectPtqPlan(new pb.PtqPlanRef({ ptqPlanId: plan.ptqPlanId }));
    assert.equal(state.coverage.complete, true);
    assert.equal(state.calibrationSamples, 3n);
    const packed = await quantization.writePtqPackage(new pb.WritePtqPackageRequest({ ptqPlanId: plan.ptqPlanId }));
    assert.deepEqual(JSON.parse(new TextDecoder().decode(packed.graph)).banks, { weight: 'source' });
    const reader = new VxInferenceServiceClient(readerHost);
    const readRuntime = await reader.createRuntime(new pb.CreateRuntimeRequest());
    const quantized = await reader.loadModel(new pb.LoadModelRequest({ runtimeId: readRuntime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: packed.graph, weightShards: [packed.weights] }) }));
    const compiled = await reader.compileModel(new pb.CompileModelRequest({ modelId: quantized.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'] }) }));
    const context = await reader.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId }));
    for (const rows of [3, 1, 2]) {
      const result = await reader.execute(new pb.ExecuteRequest({ contextId: context.contextId, inputs: [input(rows)] }));
      try {
        const output = (await reader.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name: 'output' }))).tensor;
        assert.deepEqual(output.shape, [1n, BigInt(rows), 2n]);
        const actual = new Float32Array(output.inline.slice().buffer);
        for (let row = 0; row < rows; row++) {
          const [a, b, c] = values.subarray(row * 3, row * 3 + 3);
          assert.ok(Math.abs(actual[row * 2] - (.25 * a - .75 * b + 1.5 * c)) < .04);
          assert.ok(Math.abs(actual[row * 2 + 1] - (2 * a + .5 * b - c)) < .04);
        }
      } finally {
        await reader.releaseResult(new pb.ResultRef({ resultId: result.resultId }));
      }
    }
  } finally {
    await readerHost.close();
    await host.close();
  }
});

test('C PTQ exports a bias-free HWIO convolution between float regions', async () => {
  const shape = [1, 1, 3, 3];
  const graph = new TextEncoder().encode(JSON.stringify({
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
  }));
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
    const inference = new VxInferenceServiceClient(host);
    const quantization = new VxQuantizationServiceClient(host);
    const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
    const model = await inference.loadModel(new pb.LoadModelRequest({
      runtimeId: runtime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: graph, weightShards: [weights] }),
    }));
    const authored = await quantization.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({
      sourceGraph: graph, weightShards: [weights],
      config: new pb.PtqAuthoringConfig({
        activationDtype: pb.DataType.DATA_TYPE_I8,
        activationScheme: pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC, floatOperators: ['SiLU'],
      }),
    }));
    assert.equal(authored.quantizedNodes, 1n);
    assert.equal(authored.layers[0].sourceBiasName, '');
    assert.ok(authored.layers[0].packedBiasName);
    const plan = await quantization.createPtqPlan(new pb.CreatePtqPlanRequest({
      modelId: model.modelId, templateGraph: authored.templateGraph,
      observers: authored.observers, layers: authored.layers, profileNames: ['default'],
    }));
    await quantization.calibratePtqPlan(new pb.CalibratePtqPlanRequest({
      ptqPlanId: plan.ptqPlanId, profileName: 'default',
      sampleName: 'sample', sampleCount: 1n, inputs: [input],
    }));
    const packed = await quantization.writePtqPackage(new pb.WritePtqPackageRequest({ ptqPlanId: plan.ptqPlanId }));
    const reader = new VxInferenceServiceClient(inferenceHost);
    const readRuntime = await reader.createRuntime(new pb.CreateRuntimeRequest());
    const quantized = await reader.loadModel(new pb.LoadModelRequest({
      runtimeId: readRuntime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: packed.graph, weightShards: [packed.weights] }),
    }));
    const compiled = await reader.compileModel(new pb.CompileModelRequest({
      modelId: quantized.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'] }),
    }));
    const result = await reader.run(new pb.RunRequest({ compiledModelId: compiled.compiledModelId, inputs: [input] }));
    const output = (await reader.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name: 'output' }))).tensor;
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
        const graph = new TextEncoder().encode(JSON.stringify({
          format: 'volvox-graph/v1', dimensions: {},
          inputs: { input: { shape, dtype: 'float32' } },
          nodes: [{ id: 'conv', opType: 'Conv2D',
            inputs: { input: 'input', weight: 'weight', ...(config.bias ? { bias: 'bias' } : {}) },
            outputs: { out: { tensor: 'output', shape: outputShape, dtype: 'float32' } },
            params: { data_layout: 'NHWC', weight_layout: 'HWIO', stride, dilation,
              pads: padding, groups } }],
          outputs: ['output'],
        }));
        const model = await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
          package: new pb.ModelPackage({ graphDocument: graph, weightShards: [safetensors(tensors)] }) }));
        const compiled = await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
          policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
            backends: ['wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) }));
        const result = await client.run(new pb.RunRequest({ compiledModelId: compiled.compiledModelId,
          inputs: [new pb.Tensor({ name: 'input', shape: shape.map(BigInt), dtype: pb.DataType.DATA_TYPE_F32,
            inline: new Uint8Array(input.buffer) })] }));
        try {
          assert.equal(result.report.route.provider, 'wasm');
          assert.equal(result.report.route.fallbackNodes, 0);
          const output = (await client.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name: 'output' }))).tensor;
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
          await client.releaseResult(new pb.ResultRef({ resultId: result.resultId }));
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
        { name: 'gemv-affine', conv: false, shape: [1, 17], outputShape: [1, 32], k: 17, n: 32, signed: true, asymmetricWeight: true },
        { name: 'gemm-tails', conv: false, shape: [1, 3, 37], outputShape: [1, 3, 19], k: 37, n: 19, signed: false, asymmetricWeight: false },
      ]) {
        const { shape, outputShape, k, n } = config;
        const InputArray = config.signed ? Int8Array : Uint8Array;
        const OutputArray = config.signed ? Uint8Array : Int8Array;
        const inputZero = config.signed ? -11 : 131, outputZero = config.signed ? 123 : -3;
        const input = InputArray.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => inputZero + ((i * 13) % 41) - 20);
        if (!config.signed) input[7] = 250;
        const weight = Int8Array.from({ length: k * n }, (_, i) => (i * 7) % 17 - 8);
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
        const quantization = Object.fromEntries(['input', 'weight', 'output'].map(name => [name, {
          scheme: name === 'weight' ? 'per_axis' : 'per_tensor',
          ...(name === 'weight' ? { axis: 0 } : {}),
          scale_tensor: `${name}.scale`, zero_point_tensor: `${name}.zero`,
        }]));
        const graph = new TextEncoder().encode(JSON.stringify({
          format: 'volvox-graph/v1', dimensions: {},
          inputs: { input: { shape, dtype: config.signed ? 'int8' : 'uint8' } },
          nodes: [{ id: 'quantized', opType: config.conv ? 'QConv2D' : 'QLinear',
            inputs: { input: 'input', weight: 'weight', bias: 'bias' },
            outputs: { out: { tensor: 'output', shape: outputShape, dtype: config.signed ? 'uint8' : 'int8' } },
            params: config.conv ? { data_layout: 'NHWC', weight_layout: 'OHWI',
              stride: [2, 2], dilation: [1, 1], pads: [1, 0, 0, 1], groups: 1, relu: 0 }
              : {} }],
          outputs: ['output'], quantization: { format: 'volvox-affine-safetensors/v1', tensors: quantization },
        }));
        const model = await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
          package: new pb.ModelPackage({ graphDocument: graph, weightShards: [safetensors(tensors)] }) }));
        const compiled = await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
          policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
            backends: ['wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) }))
          .catch(error => { error.message = `${config.name}: ${error.message}`; throw error; });
        const result = await client.run(new pb.RunRequest({ compiledModelId: compiled.compiledModelId,
          inputs: [new pb.Tensor({ name: 'input', shape: shape.map(BigInt),
            dtype: config.signed ? pb.DataType.DATA_TYPE_I8 : pb.DataType.DATA_TYPE_U8,
            inline: new Uint8Array(input.buffer) })] }));
        try {
          assert.equal(result.report.route.provider, 'wasm');
          assert.equal(result.report.route.fallbackNodes, 0);
          const output = (await client.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name: 'output' }))).tensor;
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
          await client.releaseResult(new pb.ResultRef({ resultId: result.resultId }));
        }
      }
    } finally {
      await host.close();
    }
  });
}

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
