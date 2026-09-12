import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
/** The generated proto clients driving the TypeScript engine. */
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { EngineHost } from '../ts/host/EngineHost.js';
import { loadModelControlWasmDispatchFactory } from '../ts/core/ModelControlWasm.js';
import {
  VxInferenceServiceClient,
  VxPlanningServiceClient,
  VxPlatformServiceClient,
  VxSchedulerServiceClient} from '../runtime/generated/typescript/inference/volvoxai_ffi.js';
import * as pb from '../runtime/generated/typescript/inference/volvoxai_lite.js';

const GRAPH_PATH = 'fixture/graph.json';
const WEIGHTS_PATH = 'fixture/model.safetensors';
const SOURCE_REVISION_A_PATH = 'fixture/source-a.graph.json';
const SOURCE_REVISION_B_PATH = 'fixture/source-b.graph.json';
const PROTOTYPE_BANK_GRAPH_PATH = 'fixture/prototype-bank.graph.json';
const PROTOTYPE_BANK_WEIGHTS_PATH = 'fixture/prototype-bank.safetensors';
const WASM = fileURLToPath(new URL('../dist/0.4.0/volvoxai.wasm', import.meta.url));
const GRAPH = new TextEncoder().encode(JSON.stringify({
  format: 'volvox-graph/v1',
  dimensions: { B: { min: 1, max: 1 } },
  inputs: { x: { dtype: 'float32', shape: ['B', 2] } },
  nodes: [{
    id: 'projection',
    opType: 'MatMul',
    inputs: { input: 'x', weight: 'parameter' },
    outputs: {
      out: { tensor: 'logits', dtype: 'float32', shape: ['B', 2] },
    },
    params: {},
  }],
  outputs: ['logits'],
}));

function safetensors() {
  const values = Float32Array.of(0.2, -0.4, 0.1, 0.3);
  let header = new TextEncoder().encode(JSON.stringify({
    parameter: { dtype: 'F32', shape: [2, 2], data_offsets: [0, values.byteLength] },
  }));
  const padding = (8 - (header.byteLength % 8)) % 8;
  if (padding) {
    const padded = new Uint8Array(header.byteLength + padding);
    padded.set(header);
    padded.fill(0x20, header.byteLength);
    header = padded;
  }
  const output = new Uint8Array(8 + header.byteLength + values.byteLength);
  new DataView(output.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  output.set(header, 8);
  output.set(new Uint8Array(values.buffer), 8 + header.byteLength);
  return output;
}

function orderedSafetensors(metadataPosition) {
  const tensorA = '"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}';
  const metadata = '"__metadata__":{"fixture":"ordered"}';
  const tensorB = '"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}';
  const members = metadataPosition === 'middle'
    ? [tensorA, metadata, tensorB]
    : [tensorA, tensorB, metadata];
  let header = new TextEncoder().encode(`{${members.join(',')}}`);
  const padding = (8 - (header.byteLength % 8)) % 8;
  if (padding !== 0) {
    const padded = new Uint8Array(header.byteLength + padding);
    padded.set(header);
    padded.fill(0x20, header.byteLength);
    header = padded;
  }
  const output = new Uint8Array(8 + header.byteLength + 8);
  new DataView(output.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  output.set(header, 8);
  output.set(new Uint8Array(Float32Array.of(1, 2).buffer), 8 + header.byteLength);
  return output;
}

function malformedSafetensorsByteLength() {
  let header = new TextEncoder().encode(
    '{"bad":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}',
  );
  const padding = (8 - (header.byteLength % 8)) % 8;
  if (padding !== 0) {
    const padded = new Uint8Array(header.byteLength + padding);
    padded.set(header);
    padded.fill(0x20, header.byteLength);
    header = padded;
  }
  const output = new Uint8Array(8 + header.byteLength + 4);
  new DataView(output.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  output.set(header, 8);
  return output;
}

function prototypeBankSafetensors() {
  const values = new Float32Array(8);
  let header = new TextEncoder().encode(
    '{"__proto__":{"dtype":"F32","shape":[4,1],"data_offsets":[0,16]},' +
    '"constructor":{"dtype":"F32","shape":[4,1],"data_offsets":[16,32]}}',
  );
  const padding = (8 - (header.byteLength % 8)) % 8;
  if (padding !== 0) {
    const padded = new Uint8Array(header.byteLength + padding);
    padded.set(header);
    padded.fill(0x20, header.byteLength);
    header = padded;
  }
  const output = new Uint8Array(8 + header.byteLength + values.byteLength);
  new DataView(output.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  output.set(header, 8);
  output.set(new Uint8Array(values.buffer), 8 + header.byteLength);
  return output;
}

const FILES = new Map([
  [GRAPH_PATH, GRAPH],
  [WEIGHTS_PATH, safetensors()],
  [SOURCE_REVISION_A_PATH, new TextEncoder().encode(
    '{"format":"volvox-graph/v1","dimensions":{},"inputs":{"x":{"dtype":"float32","shape":[1]}},"nodes":[],"outputs":["x"]}',
  )],
  [SOURCE_REVISION_B_PATH, new TextEncoder().encode(
    '{\n  "outputs": ["x"], "nodes": [], "inputs": {"x": {"shape": [1], "dtype": "float32"}},\n  "dimensions": {}, "format": "volvox-graph/v1"\n}',
  )],
  [PROTOTYPE_BANK_GRAPH_PATH, new TextEncoder().encode(
    '{"format":"volvox-graph/v1","dimensions":{"E":{"min":1,"max":8}},' +
    '"inputs":{},"nodes":[],"outputs":["__proto__","constructor"],' +
    '"banks":{"__proto__":"E","constructor":"E"}}',
  )],
  [PROTOTYPE_BANK_WEIGHTS_PATH, prototypeBankSafetensors()],
]);

async function fileFetch(source) {
  const bytes = FILES.get(String(source));
  if (!bytes) return { ok: false, status: 404 };
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return {
    ok: true,
    status: 200,
    headers: new Headers({ 'content-length': String(bytes.byteLength) }),
    arrayBuffer: async () => buffer,
    text: async () => new TextDecoder().decode(bytes),
    json: async () => JSON.parse(new TextDecoder().decode(bytes)),
  };
}

function makeHost() {
  return new EngineHost({
    wasmUrl: WASM,
    fetch: fileFetch,
    resolveModelSource: (graphPath, weightPaths) => ({
      graphUrl: graphPath,
      weightSources: weightPaths,
    }),
  });
}

const asBytes = (view) => new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
const isOk = (report) => report?.status === pb.NativeStatus.NATIVE_STATUS_OK;

function projectedParameterPlan(parameters = []) {
  const input = new pb.GraphTensorRef({ tensorIndex: 0, name: 'x' });
  const output = new pb.GraphTensorRef({ tensorIndex: 1, name: 'y' });
  const node = new pb.GraphNodeRef({ scheduleIndex: 0, id: 'node' });
  const plan = new pb.GraphPlan({
    graphFingerprint: 'volvox-graph-source/v1:fixture',
    planIdentity: 'volvox-plan/v1:fixture',
    tensors: [
      new pb.GraphTensor({
        tensorIndex: 0,
        name: 'x',
        kind: pb.GraphTensorKind.GRAPH_TENSOR_KIND_INPUT,
        dtype: pb.DataType.DATA_TYPE_F32,
        shape: [new pb.GraphAxis({ fixedExtent: 1n })],
        canonicalBirthStep: -1,
        canonicalLastUseStep: 0,
      }),
      new pb.GraphTensor({
        tensorIndex: 1,
        name: 'y',
        kind: pb.GraphTensorKind.GRAPH_TENSOR_KIND_VALUE,
        dtype: pb.DataType.DATA_TYPE_F32,
        shape: [new pb.GraphAxis({ fixedExtent: 1n })],
        producer: new pb.GraphTensorProducer({ node, port: 'out' }),
        canonicalBirthStep: 0,
        canonicalLastUseStep: 1,
      }),
    ],
    nodes: [new pb.GraphNode({
      definitionIndex: 0,
      scheduleIndex: 0,
      id: 'node',
      operatorName: 'MatMul',
      shapeFunctionId: 'volvox.shape.dense-last-axis.v1',
      inputs: [new pb.GraphPortBinding({ port: 'input', tensor: input })],
      outputs: [new pb.GraphPortBinding({ port: 'out', tensor: output })],
      parameters,
    })],
    outputs: [output],
    shapeDomain: new pb.ShapeDomainAnalysis({
      supported: new pb.ShapeDomainProof({
        proofIdentity: 'shape-proof/v1:fixture',
        kind: pb.ShapeDomainProofKind.SHAPE_DOMAIN_PROOF_KIND_SINGLETON_EXHAUSTIVE,
        nodes: [new pb.ShapeDomainNodeProof({
          node,
          facts: [pb.ShapeDomainFact.SHAPE_DOMAIN_FACT_SINGLETON_PUBLIC_DOMAIN],
        })],
      }),
    }),
    independentBatch: new pb.IndependentBatchAnalysis({
      supported: new pb.IndependentBatchContract({
        proofIdentity: 'batch-proof/v1:fixture',
        coveredNodes: 1,
      }),
    }),
  });
  plan.tensors[1].publicOutput = { outputIndex: 0 };
  return plan;
}

function assertReportOnlyResolvedGraphPlan(
  value,
  requestedGraphPlanId,
  expectedLineageGraphPlanId = requestedGraphPlanId,
) {
  assert.notEqual(value.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
  assert.equal(value.report?.lineage?.graphPlanId, expectedLineageGraphPlanId);
  assert.equal(value.graphPlanId, 0n);
  assert.equal(value.graphFingerprint, '');
  assert.equal(value.planIdentity, '');
  assert.equal(
    value.sourceKind,
    pb.GraphPlanSourceKind.GRAPH_PLAN_SOURCE_KIND_UNSPECIFIED,
  );
  assert.equal(value.signature, '');
  assert.equal(value.signatureDigest, 0n);
  assert.deepEqual(value.symbols, []);
  assert.deepEqual(value.tensors, []);
  assert.deepEqual(value.outputs, []);
  assert.deepEqual(value.weightBanks, []);
  assert.equal(value.executionSlice, undefined);
  assert.equal(value.weightBytes, 0n);
}

function inputs(values = [1, 2]) {
  const x = new Float32Array(values);
  return [new pb.Tensor({
    name: 'x',
    shape: [1n, 2n],
    dtype: pb.DataType.DATA_TYPE_F32,
    inline: asBytes(x),
  })];
}

async function loadAndCompile(inference, runtimeId) {
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId,
    graphPath: GRAPH_PATH,
    weightPaths: [WEIGHTS_PATH],
  }));
  assert.ok(isOk(model.report), model.report?.message);
  const compiled = await inference.compileModel(
    new pb.CompileModelRequest({ modelId: model.modelId }));
  assert.ok(isOk(compiled.report), compiled.report?.message);
  return { model, compiled };
}

test('the 50-RPC proto API drives generated C/WASM dispatch end to end', async () => {
  const host = makeHost();
  const platform = new VxPlatformServiceClient(reportTransport(host));
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const planning = new VxPlanningServiceClient(reportTransport(host));

  const info = await platform.getPlatformInfo(new pb.Empty());
  assert.deepEqual(info.compiledBackends, ['wasm']);
  assert.equal(info.transport, pb.TransportProfile.TRANSPORT_PROFILE_REMOTE);

  // execution_mode is optional and absence deliberately means DIRECT.
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest({
    memoryCapture: new pb.MemoryCaptureOptions({
      protocol: 'volvoxai-memory-capture/v1',
      includeResourceInventory: true,
      includeDomainAttestation: true,
    }),
  }));
  assert.ok(isOk(runtime.report), runtime.report?.message);
  const { model, compiled } = await loadAndCompile(inference, runtime.runtimeId);

  const graphPlan = await planning.createGraphPlan(new pb.CreateGraphPlanRequest({
    modelId: model.modelId,
  }));
  assert.ok(isOk(graphPlan.report), graphPlan.report?.message);
  assert.ok(graphPlan.graphPlanId > 0n);
  assert.equal(
    graphPlan.sourceKind,
    pb.GraphPlanSourceKind.GRAPH_PLAN_SOURCE_KIND_MODEL,
  );
  assert.ok(graphPlan.plan?.graphFingerprint);
  assert.ok(graphPlan.plan?.planIdentity);
  const shapeDomainProofIdentity = graphPlan.plan?.shapeDomain?.supported?.proofIdentity;
  assert.ok(shapeDomainProofIdentity);
  const compileSnapshot = compiled.report?.memoryEvidence?.snapshots
    .find((snapshot) => snapshot.domainAttestation);
  const compileDomainAttestation = compileSnapshot?.domainAttestation;
  assert.equal(
    compileSnapshot?.subject?.kind,
    pb.MemoryOwnerKind.MEMORY_OWNER_KIND_COMPILED_MODEL,
  );
  assert.ok(compileSnapshot?.subject?.ownerId);
  assert.equal(
    compileDomainAttestation?.shapeDomainProofIdentity,
    shapeDomainProofIdentity,
    'memory evidence must echo the exact Planning proof identity',
  );
  assert.equal(
    compileDomainAttestation?.graphFingerprint,
    graphPlan.plan?.graphFingerprint,
  );
  assert.equal(
    compileDomainAttestation?.proofProtocol,
    'canonical-symbolic-domain-proof/v1',
  );
  assert.equal(
    compileDomainAttestation?.resourceProtocol,
    'bounded-resource-maxima/v1',
  );
  assert.deepEqual(
    compileDomainAttestation?.bounds.map(({ budgetDomainId, kind }) => ({
      budgetDomainId,
      kind,
    })),
    [{
      budgetDomainId: 'maximum-tensor',
      kind: pb.MemoryBoundKind.MEMORY_BOUND_KIND_MAXIMUM_TENSOR,
    }, {
      budgetDomainId: 'provider-resident',
      kind: pb.MemoryBoundKind.MEMORY_BOUND_KIND_ORDINARY_RESIDENT,
    }],
  );
  const maximumTensor = compileDomainAttestation?.bounds.find(
    ({ budgetDomainId }) => budgetDomainId === 'maximum-tensor',
  );
  const providerResident = compileDomainAttestation?.bounds.find(
    ({ budgetDomainId }) => budgetDomainId === 'provider-resident',
  );
  assert.ok((maximumTensor?.maximumBytes?.bytes ?? 0n) > 0n);
  assert.ok(
    (providerResident?.maximumBytes?.bytes ?? 0n) >=
      (maximumTensor?.maximumBytes?.bytes ?? 0n),
  );
  assert.equal(providerResident?.limitBytes, undefined);
  assert.equal(graphPlan.report?.lineage?.runtimeId, runtime.runtimeId);
  assert.equal(graphPlan.report?.lineage?.modelId, model.modelId);
  assert.equal(graphPlan.report?.lineage?.graphPlanId, graphPlan.graphPlanId);

  const describedPlan = await planning.getGraphPlan(new pb.GraphPlanRef({
    graphPlanId: graphPlan.graphPlanId,
  }));
  assert.ok(isOk(describedPlan.report), describedPlan.report?.message);
  assert.equal(describedPlan.plan?.planIdentity, graphPlan.plan?.planIdentity);
  assert.equal(describedPlan.sourceKind, graphPlan.sourceKind);

  const rejectedModelResidency = await planning.resolveGraphPlan(
    new pb.ResolveGraphPlanRequest({
      graphPlanId: graphPlan.graphPlanId,
      minimum: new pb.Empty(),
      bankResidency: [new pb.BankResidency({ bank: 'parameter', slots: [0] })],
    }),
  );
  assert.equal(
    rejectedModelResidency.report?.status,
    pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
  );
  assertReportOnlyResolvedGraphPlan(rejectedModelResidency, graphPlan.graphPlanId);

  assert.ok(isOk(await inference.releaseModel(new pb.ModelRef({
    modelId: model.modelId,
  }))));

  const resolvedPlan = await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
    graphPlanId: graphPlan.graphPlanId,
    minimum: new pb.Empty(),
    call: new pb.CallIntent({
      kind: pb.CallIntentKind.CALL_INTENT_KIND_FORWARD,
    }),
  }));
  assert.ok(isOk(resolvedPlan.report), resolvedPlan.report?.message);
  assert.equal(resolvedPlan.graphPlanId, graphPlan.graphPlanId);
  assert.equal(resolvedPlan.planIdentity, graphPlan.plan?.planIdentity);
  assert.ok(resolvedPlan.signature);
  assert.equal(
    resolvedPlan.executionSlice?.selection,
    pb.NodeSelection.NODE_SELECTION_ALL,
  );
  assert.deepEqual(
    resolvedPlan.executionSlice?.selectedNodes.map(({ scheduleIndex }) => scheduleIndex),
    [0],
  );
  assert.equal(
    resolvedPlan.report?.lineage?.modelId,
    model.modelId,
    'a GraphPlan must retain its model after the public model handle is released',
  );

  const packageBytes = FILES.get(WEIGHTS_PATH);
  const headerJsonBytes = Number(new DataView(
    packageBytes.buffer,
    packageBytes.byteOffset,
    8,
  ).getBigUint64(0, true));
  const inspected = await planning.inspectSafetensors(new pb.InspectSafetensorsRequest({
    inlineHeader: new pb.SafetensorsInlineHeaderSource({
      headerPrefix: packageBytes.slice(0, 8 + headerJsonBytes),
      fileSize: BigInt(packageBytes.byteLength),
    }),
  }));
  assert.ok(isOk(inspected.report), inspected.report?.message);
  assert.equal(inspected.headerJsonBytes, BigInt(headerJsonBytes));
  assert.equal(inspected.dataRegionOffset, BigInt(8 + headerJsonBytes));
  assert.equal(inspected.fileBytes, BigInt(packageBytes.byteLength));
  assert.equal(inspected.diagnostic, undefined);
  assert.deepEqual(inspected.tensors.map(({ name }) => name), ['parameter']);

  assert.equal(compiled.report?.stage, pb.OperationStage.OPERATION_STAGE_COMPILE);
  assert.equal(compiled.report?.lineage?.runtimeId, runtime.runtimeId);
  assert.equal(compiled.report?.lineage?.modelId, model.modelId);
  assert.equal(compiled.report?.lineage?.compiledModelId, compiled.compiledModelId);
  assert.ok(compiled.report?.timings);
  assert.ok(compiled.report?.accounting);
  assert.equal(
    compiled.report?.compilation?.policyMode,
    pb.BackendPolicyMode.BACKEND_POLICY_MODE_PREFER,
  );
  assert.equal(
    compiled.report?.compilation?.operatorFallback,
    pb.OperatorFallback.OPERATOR_FALLBACK_ALLOW,
  );
  assert.ok(compiled.report?.compilation?.candidates.some((candidate) =>
    candidate.outcome === pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED &&
    candidate.status === pb.NativeStatus.NATIVE_STATUS_OK));
  assert.equal(compiled.report?.route?.provider, compiled.report?.backend);
  assert.equal(compiled.report?.route?.attested, true);
  assert.equal(compiled.report?.fallback?.operatorFallbackUsed, false);
  assert.equal(compiled.report?.memoryEvidence?.format, 'volvoxai-memory-evidence/v1');
  assert.ok(compiled.report?.memoryEvidence?.snapshots.length);
  const context = await inference.createExecutionContext(
    new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId }));
  assert.ok(isOk(context.report), context.report?.message);
  assert.deepEqual(context.inputs.map((input) => input.name), ['x']);

  const baseAdapter = await inference.selectAdapter(new pb.SelectAdapterRequest({
    contextId: context.contextId,
  }));
  assert.ok(isOk(baseAdapter), baseAdapter.message);
  const invalidAdapter = await inference.selectAdapter(new pb.SelectAdapterRequest({
    contextId: context.contextId,
    revision: new pb.AdapterRevisionRef(),
  }));
  assert.equal(
    invalidAdapter.status,
    pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
    'a present revision key must never be reinterpreted as the base model',
  );

  const executed = await inference.execute(new pb.ExecuteRequest({
    contextId: context.contextId,
    inputs: inputs(),
  }));
  assert.ok(isOk(executed.report), executed.report?.message);
  assert.equal(executed.report?.lineage?.runtimeId, runtime.runtimeId);
  assert.equal(executed.report?.lineage?.modelId, model.modelId);
  assert.equal(executed.report?.lineage?.compiledModelId, compiled.compiledModelId);
  assert.equal(executed.report?.lineage?.contextId, context.contextId);
  assert.ok(executed.executionId > 0n);
  assert.equal(executed.report?.lineage?.executionId, executed.executionId);
  assert.ok(executed.report?.timings);
  assert.equal(executed.report?.route?.provider, executed.report?.backend);
  assert.equal(executed.report?.route?.attested, true);
  assert.ok(executed.report?.route?.shapePlan?.signature);
  assert.equal(executed.report?.fallback?.operatorFallbackUsed, false);
  assert.equal(executed.report?.decode?.enabled, false);
  assert.ok((executed.report?.accounting?.resultBytes ?? 0n) > 0n);
  assert.equal(executed.report?.memoryEvidence?.format, 'volvoxai-memory-evidence/v1');
  assert.ok(executed.report?.memoryEvidence?.snapshots.length);

  const result = await inference.getResult(new pb.ResultRef({ resultId: executed.resultId }));
  assert.equal(result.executionId, executed.executionId);
  assert.equal(result.outputs[0]?.name, 'logits');
  assert.equal(result.outputs[0]?.dtype, pb.DataType.DATA_TYPE_F32);
  const read = await inference.readOutput(new pb.ReadOutputRequest({
    resultId: executed.resultId,
    name: 'logits',
  }));
  assert.ok(isOk(read.report), read.report?.message);
  assert.equal(BigInt(read.tensor?.inline?.byteLength ?? 0), read.requiredBytes);

  assert.ok(isOk(await planning.releaseGraphPlan(new pb.GraphPlanRef({
    graphPlanId: graphPlan.graphPlanId,
  }))));
  const releasedPlan = await planning.getGraphPlan(new pb.GraphPlanRef({
    graphPlanId: graphPlan.graphPlanId,
  }));
  assert.equal(
    releasedPlan.report?.status,
    pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED,
  );
  const releasedResolution = await planning.resolveGraphPlan(
    new pb.ResolveGraphPlanRequest({
      graphPlanId: graphPlan.graphPlanId,
      minimum: new pb.Empty(),
    }),
  );
  assert.equal(
    releasedResolution.report?.status,
    pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED,
  );
  assertReportOnlyResolvedGraphPlan(releasedResolution, graphPlan.graphPlanId);
  assert.ok(isOk(await planning.releaseGraphPlan(new pb.GraphPlanRef({
    graphPlanId: graphPlan.graphPlanId,
  }))));

  for (const release of [
    async () => (await inference.releaseResult(new pb.ResultRef({ resultId: executed.resultId }))),
    async () => (await inference.releaseExecutionContext(
      new pb.ExecutionContextRef({ contextId: context.contextId }))),
    async () => (await inference.releaseCompiledModel(
      new pb.CompiledModelRef({ compiledModelId: compiled.compiledModelId }))),
    async () => (await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }))),
    async () => (await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId: runtime.runtimeId }))),
  ]) {
    assert.ok(isOk(await release()));
  }
});

test('standalone GraphPlan authoring owns a typed source and requires an explicit binding', async () => {
  const planning = new VxPlanningServiceClient(reportTransport(makeHost()));
  const created = await planning.createGraphPlan(new pb.CreateGraphPlanRequest({
    graph: new pb.GraphPlanningSource({
      graphDocument: GRAPH.slice(),
      weights: [new pb.PlanningWeight({
        name: 'parameter',
        dtype: pb.DataType.DATA_TYPE_F32,
        shape: [2n, 2n],
      })],
    }),
  }));
  assert.ok(isOk(created.report), created.report?.message);
  assert.equal(
    created.sourceKind,
    pb.GraphPlanSourceKind.GRAPH_PLAN_SOURCE_KIND_STANDALONE_GRAPH,
  );

  const missingBinding = await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
    graphPlanId: created.graphPlanId,
  }));
  assert.equal(
    missingBinding.report?.status,
    pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
  );
  assertReportOnlyResolvedGraphPlan(missingBinding, created.graphPlanId);

  const resolved = await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
    graphPlanId: created.graphPlanId,
    exact: new pb.ExactShapeBinding({
      inputs: [new pb.ConcreteInputShape({
        name: 'x',
        dtype: pb.DataType.DATA_TYPE_F32,
        shape: [1n, 2n],
      })],
    }),
    call: new pb.CallIntent({
      kind: pb.CallIntentKind.CALL_INTENT_KIND_ALL_CROSS_CALL_LIVE,
    }),
  }));
  assert.ok(isOk(resolved.report), resolved.report?.message);
  assert.equal(resolved.planIdentity, created.plan?.planIdentity);
  assert.equal(
    resolved.executionSlice?.kind,
    pb.CallIntentKind.CALL_INTENT_KIND_ALL_CROSS_CALL_LIVE,
  );
  assert.equal(
    resolved.executionSlice?.crossCallLiveTensors.length,
    created.plan?.tensors.filter(({ kind }) =>
      kind !== pb.GraphTensorKind.GRAPH_TENSOR_KIND_WEIGHT).length,
  );
  assert.ok(isOk(await planning.releaseGraphPlan(new pb.GraphPlanRef({
    graphPlanId: created.graphPlanId,
  }))));
});

test('remote GraphPlan creation preserves the direct C failure report', async () => {
  const factory = await loadModelControlWasmDispatchFactory(WASM);
  const control = factory.create();
  try {
    const request = new pb.CreateGraphPlanRequest({
      graph: new pb.GraphPlanningSource({ graphDocument: new Uint8Array() }),
    });
    const direct = (await new VxPlanningServiceClient(reportTransport(control))
      .createGraphPlan(request));
    const remote = await new VxPlanningServiceClient(reportTransport(makeHost())).createGraphPlan(request);

    assert.notEqual(direct.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.equal(remote.report?.status, direct.report?.status);
    assert.equal(remote.report?.code, direct.report?.code);
    assert.equal(remote.report?.stage, direct.report?.stage);
    assert.deepEqual(remote.report?.toBinary(), direct.report?.toBinary());
    assert.equal(remote.graphPlanId, 0n);
    assert.equal(remote.plan, undefined);
  } finally {
    (await control.close());
  }
});

test('model GraphPlan fingerprints exact source bytes but shares semantic plan identity', async () => {
  const host = makeHost();
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const planning = new VxPlanningServiceClient(reportTransport(host));
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  assert.ok(isOk(runtime.report), runtime.report?.message);

  const models = [];
  const plans = [];
  for (const graphPath of [SOURCE_REVISION_A_PATH, SOURCE_REVISION_B_PATH]) {
    const model = await inference.loadModel(new pb.LoadModelRequest({
      runtimeId: runtime.runtimeId,
      graphPath,
    }));
    assert.ok(isOk(model.report), model.report?.message);
    models.push(model);
    const plan = await planning.createGraphPlan(new pb.CreateGraphPlanRequest({
      modelId: model.modelId,
    }));
    assert.ok(isOk(plan.report), plan.report?.message);
    plans.push(plan);
  }
  assert.notEqual(plans[0].plan?.graphFingerprint, plans[1].plan?.graphFingerprint);
  assert.equal(plans[0].plan?.planIdentity, plans[1].plan?.planIdentity);

  for (const plan of plans) {
    assert.ok(isOk(await planning.releaseGraphPlan(new pb.GraphPlanRef({
      graphPlanId: plan.graphPlanId,
    }))));
  }
  for (const model of models) {
    assert.ok(isOk(await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }))));
  }
  assert.ok(isOk(await inference.releaseRuntime(new pb.RuntimeRef({
    runtimeId: runtime.runtimeId,
  }))));
});

test('Planning accepts a public weight output and omits an unused dimension binding', async () => {
  const planning = new VxPlanningServiceClient(reportTransport(makeHost()));
  const graphDocument = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1',
    dimensions: { UNUSED: { min: 1, max: 4 } },
    inputs: {},
    nodes: [],
    outputs: ['parameter'],
  }));
  const created = await planning.createGraphPlan(new pb.CreateGraphPlanRequest({
    graph: new pb.GraphPlanningSource({
      graphDocument,
      weights: [new pb.PlanningWeight({
        name: 'parameter',
        dtype: pb.DataType.DATA_TYPE_F32,
        shape: [2n],
      })],
    }),
  }));
  assert.ok(isOk(created.report), created.report?.message);
  const weight = created.plan?.tensors.find(({ name }) => name === 'parameter');
  assert.equal(weight?.kind, pb.GraphTensorKind.GRAPH_TENSOR_KIND_WEIGHT);
  assert.equal(weight?.canonicalBirthStep, -1);
  assert.equal(weight?.canonicalLastUseStep, -1);
  assert.equal(weight?.publicOutput?.outputIndex, 0);

  const resolved = await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
    graphPlanId: created.graphPlanId,
    minimum: new pb.Empty(),
  }));
  assert.ok(isOk(resolved.report), resolved.report?.message);
  assert.deepEqual(resolved.symbols, []);
  assert.equal(resolved.outputs[0]?.name, 'parameter');
  assert.ok(isOk(await planning.releaseGraphPlan(new pb.GraphPlanRef({
    graphPlanId: created.graphPlanId,
  }))));
});

test('public Planning preserves uint64 tensor totals beyond JavaScript safe integers', async () => {
  const planning = new VxPlanningServiceClient(reportTransport(makeHost()));
  const graphDocument = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { huge: { dtype: 'float32', shape: [1_000_000_000, 1_000_000_000] } },
    nodes: [],
    outputs: ['huge'],
  }));
  const created = await planning.createGraphPlan(new pb.CreateGraphPlanRequest({
    graph: new pb.GraphPlanningSource({ graphDocument }),
  }));
  assert.ok(isOk(created.report), created.report?.message);

  const resolved = await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
    graphPlanId: created.graphPlanId,
    minimum: new pb.Empty(),
  }));
  assert.ok(isOk(resolved.report), resolved.report?.message);
  assert.equal(resolved.tensors[0]?.elementCount, 1_000_000_000_000_000_000n);
  assert.equal(resolved.tensors[0]?.byteSize, 4_000_000_000_000_000_000n);
  assert.equal(resolved.logicalActivationBytes, 4_000_000_000_000_000_000n);
  assert.ok(isOk(await planning.releaseGraphPlan(new pb.GraphPlanRef({
    graphPlanId: created.graphPlanId,
  }))));
});

test('Planning resolves a partial weight bank as compact resident rows', async () => {
  const planning = new VxPlanningServiceClient(reportTransport(makeHost()));
  const graphDocument = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1',
    dimensions: { E: { min: 1, max: 8 } },
    inputs: {},
    nodes: [],
    outputs: ['experts'],
    banks: { experts: 'E' },
  }));
  const created = await planning.createGraphPlan(new pb.CreateGraphPlanRequest({
    graph: new pb.GraphPlanningSource({
      graphDocument,
      weights: [new pb.PlanningWeight({
        name: 'experts',
        dtype: pb.DataType.DATA_TYPE_F32,
        shape: [4n, 2n],
      })],
    }),
  }));
  assert.ok(isOk(created.report), created.report?.message);

  const resolved = await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
    graphPlanId: created.graphPlanId,
    minimum: new pb.Empty(),
    bankResidency: [new pb.BankResidency({ bank: 'experts', slots: [1, 3] })],
  }));
  assert.ok(isOk(resolved.report), resolved.report?.message);
  assert.deepEqual(resolved.tensors.find(({ name }) => name === 'experts')?.shape, [2n, 2n]);
  assert.equal(resolved.weightBytes, 16n);
  assert.deepEqual(resolved.weightBanks[0]?.partial?.slots, [1, 3]);
  assert.ok(isOk(await planning.releaseGraphPlan(new pb.GraphPlanRef({
    graphPlanId: created.graphPlanId,
  }))));
});

test('model residency preserves prototype-looking bank names as own data', async () => {
  const host = makeHost();
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const planning = new VxPlanningServiceClient(reportTransport(host));
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  assert.ok(isOk(runtime.report), runtime.report?.message);
  const beyondDeclaredSlots = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    graphPath: PROTOTYPE_BANK_GRAPH_PATH,
    weightPaths: [PROTOTYPE_BANK_WEIGHTS_PATH],
    bankResidency: [new pb.BankResidency({ bank: '__proto__', slots: [4] })],
  }));
  assert.equal(
    beyondDeclaredSlots.report?.status,
    pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
  );
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    graphPath: PROTOTYPE_BANK_GRAPH_PATH,
    weightPaths: [PROTOTYPE_BANK_WEIGHTS_PATH],
    bankResidency: [
      new pb.BankResidency({ bank: '__proto__', slots: [1] }),
      new pb.BankResidency({ bank: 'constructor', slots: [2] }),
    ],
  }));
  assert.ok(isOk(model.report), model.report?.message);
  const created = await planning.createGraphPlan(new pb.CreateGraphPlanRequest({
    modelId: model.modelId,
  }));
  assert.ok(isOk(created.report), created.report?.message);
  const resolved = await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
    graphPlanId: created.graphPlanId,
    minimum: new pb.Empty(),
  }));
  assert.ok(isOk(resolved.report), resolved.report?.message);
  assert.deepEqual(
    resolved.weightBanks.map((bank) => [bank.tensor?.name, bank.partial?.slots]),
    [['__proto__', [1]], ['constructor', [2]]],
  );

  assert.ok(isOk(await planning.releaseGraphPlan(new pb.GraphPlanRef({
    graphPlanId: created.graphPlanId,
  }))));
  assert.ok(isOk(await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }))));
  assert.ok(isOk(await inference.releaseRuntime(new pb.RuntimeRef({
    runtimeId: runtime.runtimeId,
  }))));
});

test('SafeTensors Planning preserves metadata order and typed diagnostics', async () => {
  const planning = new VxPlanningServiceClient(reportTransport(makeHost()));
  const inspect = async (bytes) => {
    const length = Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(0, true));
    return (await planning.inspectSafetensors(new pb.InspectSafetensorsRequest({ inlineHeader: new pb.SafetensorsInlineHeaderSource({
      headerPrefix: bytes.slice(0, 8 + length), fileSize: BigInt(bytes.length),
    }) })));
  };
  for (const position of ['middle', 'end']) {
    const parsed = await inspect(orderedSafetensors(position));
    assert.ok(isOk(parsed.report), parsed.report?.message);
    assert.deepEqual(parsed.metadata.map(({ key, value }) => [key, value]), [['fixture', 'ordered']]);
    assert.deepEqual(parsed.tensors.map(({ name }) => name), ['a', 'b']);
  }
  const refusal = await inspect(malformedSafetensorsByteLength());
  assert.equal(refusal.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  assert.equal(refusal.diagnostic?.code, pb.SafetensorsDiagnosticCode.SAFETENSORS_DIAGNOSTIC_CODE_BYTE_LENGTH_MISMATCH);
  assert.equal(refusal.diagnostic?.entry?.index, 0);
});

test('generated dispatch enforces canonical BackendPolicy name bounds', async () => {
  const host = makeHost();
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  assert.ok(isOk(runtime.report), runtime.report?.message);
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    graphPath: GRAPH_PATH,
    weightPaths: [WEIGHTS_PATH],
  }));
  assert.ok(isOk(model.report), model.report?.message);

  const compile = async (backends, mode = pb.BackendPolicyMode.BACKEND_POLICY_MODE_PREFER) =>
    (await inference.compileModel(new pb.CompileModelRequest({
      modelId: model.modelId,
      policy: new pb.BackendPolicy({ backends, mode }),
    })));

  const accepted = await compile(
    ['fixture.provider_name'],
    pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
  );
  assert.equal(accepted.report?.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_REQUIRED);

  for (const backend of ['vulkan', 'opengl', 'metal', 'cuda', 'webgpu']) {
    const unavailable = await compile(
      [backend], pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
    );
    assert.equal(unavailable.report?.status,
      pb.NativeStatus.NATIVE_STATUS_BACKEND_UNAVAILABLE, backend);
    assert.equal(unavailable.compiledModelId, 0n, backend);
  }
  const cpu = await compile(['wasm'], pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE);
  assert.ok(isOk(cpu.report), cpu.report?.message);
  assert.ok(isOk(await inference.releaseCompiledModel(
    new pb.CompiledModelRef({ compiledModelId: cpu.compiledModelId }))));

  for (const backends of [
    ['Uppercase'],
    [`a${'b'.repeat(63)}`],
    Array.from({ length: 17 }, (_, index) => `provider-${index}`),
    ['wasm', 'wasm'],
  ]) {
    const rejected = await compile(backends);
    assert.equal(
      rejected.report?.status,
      pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
      `policy should reject ${JSON.stringify(backends)}`,
    );
  }

  assert.ok(isOk(await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }))));
  assert.ok(isOk(await inference.releaseRuntime(
    new pb.RuntimeRef({ runtimeId: runtime.runtimeId }))));
});

test('unknown handles fail closed and Release is idempotent', async () => {
  const inference = new VxInferenceServiceClient(reportTransport(makeHost()));
  const stale = await inference.getResult(new pb.ResultRef({ resultId: 999999n }));
  assert.ok(
    stale.report?.status === pb.NativeStatus.NATIVE_STATUS_RESULT_DISPOSED
      || stale.report?.status === pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED,
  );
  const prefix = await inference.executePrefix(
    new pb.ExecutePrefixRequest({ contextId: 1n, rowCount: 1 }));
  assert.equal(prefix.report?.status, pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
  assert.ok(isOk(await inference.releaseResult(new pb.ResultRef({ resultId: 424242n }))));
});

test('Planning normalizes signed invalid handle lineage to the uint64 domain', async () => {
  const planning = new VxPlanningServiceClient(reportTransport(makeHost()));
  for (const graphPlanId of [-7n, 0n]) {
    const info = await planning.getGraphPlan(new pb.GraphPlanRef({ graphPlanId }));
    assert.equal(info.report?.status, pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
    assert.equal(info.report?.lineage?.graphPlanId, 0n);
    assert.equal(info.graphPlanId, 0n);
    assert.equal(
      info.sourceKind,
      pb.GraphPlanSourceKind.GRAPH_PLAN_SOURCE_KIND_UNSPECIFIED,
    );
    assert.equal(info.plan, undefined);

    const resolved = await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
      graphPlanId,
      minimum: new pb.Empty(),
    }));
    assertReportOnlyResolvedGraphPlan(resolved, graphPlanId, 0n);

    const released = await planning.releaseGraphPlan(new pb.GraphPlanRef({ graphPlanId }));
    assert.ok(isOk(released), released.message);
    assert.equal(released.lineage?.graphPlanId, 0n);
  }

  for (const [modelId, expectedLineageModelId] of [[999_999n, 999_999n], [-7n, 0n]]) {
    const created = await planning.createGraphPlan(new pb.CreateGraphPlanRequest({ modelId }));
    assert.equal(created.report?.status, pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
    assert.equal(created.report?.lineage?.modelId, expectedLineageModelId);
    assert.equal(created.graphPlanId, 0n);
    assert.equal(
      created.sourceKind,
      pb.GraphPlanSourceKind.GRAPH_PLAN_SOURCE_KIND_UNSPECIFIED,
    );
    assert.equal(created.plan, undefined);
  }
});

test('invalid enum and transport values fail closed', async () => {
  const host = makeHost();
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const planning = new VxPlanningServiceClient(reportTransport(host));
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  assert.ok(isOk(runtime.report), runtime.report?.message);
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    graphPath: GRAPH_PATH,
    weightPaths: [WEIGHTS_PATH],
  }));
  assert.ok(isOk(model.report), model.report?.message);

  const invalidPolicy = await inference.compileModel(new pb.CompileModelRequest({
    modelId: model.modelId,
    policy: new pb.BackendPolicy({ mode: 99 }),
  }));
  assert.equal(invalidPolicy.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  const emptyRequired = await inference.compileModel(new pb.CompileModelRequest({
    modelId: model.modelId,
    policy: new pb.BackendPolicy({
      mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
    }),
  }));
  assert.equal(emptyRequired.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);

  const compiled = await inference.compileModel(
    new pb.CompileModelRequest({ modelId: model.modelId }));
  assert.ok(isOk(compiled.report), compiled.report?.message);
  const invalidContext = await inference.createExecutionContext(
    new pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId,
      decodeRowMode: 99,
    }));
  assert.equal(invalidContext.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);

  const foreignView = await inference.run(new pb.RunRequest({
    compiledModelId: compiled.compiledModelId,
    inputs: [new pb.Tensor({
      name: 'x',
      shape: [1n, 2n],
      dtype: pb.DataType.DATA_TYPE_F32,
      view: new pb.BufferView({ handle: 1n, length: 8n }),
    })],
  }));
  assert.equal(
    foreignView.report?.status,
    pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED,
  );

  for (const request of [
    new pb.InspectSafetensorsRequest({
      path: new pb.SafetensorsPathSource({ path: WEIGHTS_PATH }),
    }),
    new pb.InspectSafetensorsRequest({
      headerView: new pb.SafetensorsHeaderViewSource({
        headerPrefix: new pb.BufferView({ handle: 1n, length: 8n }),
        fileSize: 8n,
      }),
    }),
  ]) {
    const refused = await planning.inspectSafetensors(request);
    assert.equal(
      refused.report?.status,
      pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED,
    );
    assert.equal(
      refused.report?.stage,
      pb.OperationStage.OPERATION_STAGE_SAFETENSORS_INSPECT,
    );
    assert.equal(refused.diagnostic, undefined);
  }

  assert.ok(isOk(await inference.releaseCompiledModel(new pb.CompiledModelRef({
    compiledModelId: compiled.compiledModelId,
  }))));
  assert.ok(isOk(await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }))));
  assert.ok(isOk(await inference.releaseRuntime(
    new pb.RuntimeRef({ runtimeId: runtime.runtimeId }))));
});

test('compiled and context children survive parent-handle release', async () => {
  const host = makeHost();
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const scheduler = new VxSchedulerServiceClient(reportTransport(host));

  const direct = await inference.createRuntime(new pb.CreateRuntimeRequest({
    executionMode: pb.ExecutionMode.EXECUTION_MODE_DIRECT,
  }));
  const scheduled = await inference.createRuntime(new pb.CreateRuntimeRequest({
    executionMode: pb.ExecutionMode.EXECUTION_MODE_SCHEDULED,
  }));
  const directLineage = await loadAndCompile(inference, direct.runtimeId);
  const scheduledLineage = await loadAndCompile(inference, scheduled.runtimeId);

  assert.ok(isOk((await inference.getModelRevision(new pb.ModelRef({
    modelId: scheduledLineage.model.modelId,
  }))).report));
  const context = await inference.createExecutionContext(
    new pb.CreateExecutionContextRequest({
      compiledModelId: scheduledLineage.compiled.compiledModelId,
    }));

  // Releasing parent ids must neither block nor close retained children.
  for (const model of [directLineage.model, scheduledLineage.model]) {
    assert.ok(isOk(await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }))));
  }
  for (const runtime of [direct, scheduled]) {
    assert.ok(isOk(await inference.releaseRuntime(
      new pb.RuntimeRef({ runtimeId: runtime.runtimeId }))));
  }

  const refused = await scheduler.submit(new pb.SubmitRequest({
    compiledModelId: directLineage.compiled.compiledModelId,
    inputs: [],
  }));
  assert.equal(refused.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);

  const ran = await inference.run(new pb.RunRequest({
    compiledModelId: scheduledLineage.compiled.compiledModelId,
    inputs: inputs([0, 0]),
  }));
  assert.ok(isOk(ran.report), ran.report?.message);
  const submitted = await scheduler.submit(new pb.SubmitRequest({
    compiledModelId: scheduledLineage.compiled.compiledModelId,
    inputs: inputs([0, 0]),
  }));
  assert.ok(isOk(submitted.report), submitted.report?.message);
  const waited = await scheduler.waitRequest(
    new pb.RequestRef({ requestId: submitted.requestId }));
  assert.equal(waited.state, pb.RequestState.REQUEST_STATE_SUCCEEDED);
  const taken = await scheduler.takeRequestResult(
    new pb.RequestRef({ requestId: submitted.requestId }));
  assert.ok(taken.resultId > 0n);
  assert.equal((await scheduler.takeRequestResult(
    new pb.RequestRef({ requestId: submitted.requestId }))).report?.status,
  pb.NativeStatus.NATIVE_STATUS_RESULT_DISPOSED);

  // A context retains its compiled parent too.
  for (const compiled of [directLineage.compiled, scheduledLineage.compiled]) {
    assert.ok(isOk(await inference.releaseCompiledModel(new pb.CompiledModelRef({
      compiledModelId: compiled.compiledModelId,
    }))));
  }
  const contextResult = await inference.execute(new pb.ExecuteRequest({
    contextId: context.contextId,
    inputs: inputs(),
  }));
  assert.ok(isOk(contextResult.report), contextResult.report?.message);

  assert.ok(isOk(await scheduler.releaseRequest(
    new pb.RequestRef({ requestId: submitted.requestId }))));
  for (const resultId of [ran.resultId, taken.resultId, contextResult.resultId]) {
    assert.ok(isOk(await inference.releaseResult(new pb.ResultRef({ resultId }))));
  }
  assert.ok(isOk(await inference.releaseExecutionContext(
    new pb.ExecutionContextRef({ contextId: context.contextId }))));
});
