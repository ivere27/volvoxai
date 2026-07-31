import test from 'node:test';
import assert from 'node:assert/strict';

import {
  AdapterRevision,
  BackendPolicy,
  BackendPolicyMode,
  CompileModelRequest,
  CreatePtqPlanRequest,
  CreateTrainerRequest,
  CreateRuntimeRequest,
  CrossEntropyLoss,
  DataType,
  DecodeRowMode,
  LoadModelRequest,
  MemoryLocation,
  NativeStatus,
  OperationReport,
  OperationStage,
  OperatorFallback,
  PublishAdapterRequest,
  PtqLayerKind,
  PtqLayerSpec,
  PtqMode,
  PtqObserverSpec,
  PtqScheme,
  RevisionInfo,
  RuntimeHandle,
  SelectAdapterRequest,
  Tensor,
  TrainerOptimizerOptions,
  TrainStepRequest,
  TrainingOptimizerKind,
  WritePtqPackageRequest,
} from '../runtime/generated/typescript/volvoxai_lite.js';
import {
  RuntimeServiceFfi,
  RuntimeServiceMethods,
} from '../runtime/generated/typescript/volvoxai_ffi.js';
import {
  BackendPolicyMode as ContractBackendPolicyMode,
  DataType as ContractDataType,
  DecodeRowMode as ContractDecodeRowMode,
  MemoryLocation as ContractMemoryLocation,
  NativeStatus as ContractNativeStatus,
  OperationStage as ContractOperationStage,
  OperatorFallback as ContractOperatorFallback,
  backendPolicyModes,
  decodeRowModes,
  memoryLocations,
  nativeStatusCodes,
  operationStages,
  operatorFallbackValues,
  runtimeDTypes,
} from '../ts/generated/volvoxaiEnums.js';
import {
  FullOperationStage as ContractFullOperationStage,
  PtqLayerKind as ContractPtqLayerKind,
  PtqMode as ContractPtqMode,
  PtqScheme as ContractPtqScheme,
  TrainingOptimizerKind as ContractTrainingOptimizerKind,
  fullOperationStages,
  ptqLayerKinds,
  ptqModes,
  ptqSchemes,
  trainingOptimizerKinds,
} from '../ts/generated/volvoxaiFullEnums.js';
import { normalizeTrainingUpdateMode } from '../ts/training/TrainingOptimizer.js';

test('generated messages preserve graph paths, policy, and tensor bytes', () => {
  const load = new LoadModelRequest({
    runtimeId: 'runtime-1',
    graphPath: 'models/demo/graph.json',
    weightPaths: ['models/demo/model.safetensors'],
  });
  assert.deepEqual(
    LoadModelRequest.fromBinary(load.toBinary()),
    load,
  );

  const compile = new CompileModelRequest({
    modelId: 'model-2',
    policy: new BackendPolicy({
      mode: BackendPolicyMode.BACKEND_POLICY_MODE_PREFER,
      backends: ['webgpu', 'wasm'],
      operatorFallback: OperatorFallback.OPERATOR_FALLBACK_FORBID,
    }),
  });
  assert.deepEqual(
    CompileModelRequest.fromBinary(compile.toBinary()),
    compile,
  );
  assert.equal(DecodeRowMode.DECODE_ROW_MODE_REQUIRED, 2);

  const input = new Tensor({
    name: 'tokens',
    shape: [1n, 2n],
    dtype: DataType.DATA_TYPE_I32,
    data: new Uint8Array([7, 0, 0, 0, 9, 0, 0, 0]),
  });
  assert.deepEqual(Tensor.fromBinary(input.toBinary()), input);
  assert.deepEqual(
    [
      DataType.DATA_TYPE_UNSPECIFIED,
      DataType.DATA_TYPE_BOOL,
      DataType.DATA_TYPE_F4,
      DataType.DATA_TYPE_F6_E2M3,
      DataType.DATA_TYPE_F6_E3M2,
      DataType.DATA_TYPE_U8,
      DataType.DATA_TYPE_I8,
      DataType.DATA_TYPE_F8_E5M2,
      DataType.DATA_TYPE_F8_E4M3,
      DataType.DATA_TYPE_F8_E8M0,
      DataType.DATA_TYPE_F8_E4M3FNUZ,
      DataType.DATA_TYPE_F8_E5M2FNUZ,
      DataType.DATA_TYPE_I16,
      DataType.DATA_TYPE_U16,
      DataType.DATA_TYPE_F16,
      DataType.DATA_TYPE_BF16,
      DataType.DATA_TYPE_I32,
      DataType.DATA_TYPE_U32,
      DataType.DATA_TYPE_F32,
      DataType.DATA_TYPE_C64,
      DataType.DATA_TYPE_F64,
      DataType.DATA_TYPE_I64,
      DataType.DATA_TYPE_U64,
    ],
    Array.from({ length: 23 }, (_value, index) => index),
  );
});

test('generated reports preserve lifecycle identity and route evidence', () => {
  const report = new OperationReport({
    nativeStatus: 0,
    code: 'OK',
    stage: OperationStage.OPERATION_STAGE_EXECUTE,
    backend: 'cpu',
    device: 'host',
    executionId: 101n,
    runtimeId: 11n,
    modelId: 22n,
    compiledModelId: 33n,
    contextId: 44n,
    graphRevision: 55n,
    weightRevision: 66n,
    adapterRevision: 77n,
    policyMode: BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
    operatorFallback: OperatorFallback.OPERATOR_FALLBACK_FORBID,
    routeAttested: true,
    routeEvidence: 'all:cpu',
  });
  assert.deepEqual(OperationReport.fromBinary(report.toBinary()), report);

  const revision = new RevisionInfo({
    graphId: 11n,
    graphRevision: 12n,
    weightId: 21n,
    weightRevision: 22n,
    adapterId: 31n,
    adapterRevision: 32n,
    report,
  });
  assert.deepEqual(RevisionInfo.fromBinary(revision.toBinary()), revision);

  const publish = new PublishAdapterRequest({
    modelId: 'model-2',
    adapterName: 'receipt-vqa',
    versionName: 'revision-2',
  });
  assert.deepEqual(PublishAdapterRequest.fromBinary(publish.toBinary()), publish);

  const selected = new SelectAdapterRequest({
    contextId: 'context-4',
    adapterId: 31n,
    adapterRevision: 32n,
  });
  assert.deepEqual(SelectAdapterRequest.fromBinary(selected.toBinary()), selected);
  assert.deepEqual(
    AdapterRevision.fromBinary(new AdapterRevision({
      adapterId: 31n,
      adapterRevision: 32n,
      report,
    }).toBinary()).adapterRevision,
    32n,
  );
});

test('generated Trainer messages preserve explicit private-step options', () => {
  const create = new CreateTrainerRequest({
    modelId: 'model-2',
    backend: 'cpu',
    rngSeed: 1234n,
  });
  assert.deepEqual(CreateTrainerRequest.fromBinary(create.toBinary()), create);

  const step = new TrainStepRequest({
    trainerId: 'trainer-3',
    losses: [new CrossEntropyLoss({
      name: 'language-model-loss',
      logitsName: 'logits',
      targets: [7, 9],
      ignoreIndex: -1,
      rowIndex: -1,
      weight: 0.75,
      normalizer: 2,
    })],
    trainableNames: ['projection.weight'],
    optimizer: new TrainerOptimizerOptions({
      kind: TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW,
      learningRate: Math.fround(0.001),
      weightDecay: Math.fround(0.01),
    }),
    accumulationSteps: 4,
    flushAccumulation: true,
  });
  assert.deepEqual(TrainStepRequest.fromBinary(step.toBinary()), step);
});

test('generated PTQ messages preserve the complete authoring contract', () => {
  const create = new CreatePtqPlanRequest({
    modelId: 'model-2',
    templateGraphPath: 'authoring/graph.json',
    observers: [new PtqObserverSpec({
      tensorName: 'hidden',
      dtype: DataType.DATA_TYPE_I8,
      scheme: PtqScheme.PTQ_SCHEME_SYMMETRIC,
    })],
    layers: [new PtqLayerSpec({
      mode: PtqMode.PTQ_MODE_W8A8,
      kind: PtqLayerKind.PTQ_LAYER_KIND_QLINEAR,
      nodeIndex: 3,
      weightAxis: 0,
      inputTensorName: 'hidden',
      outputTensorName: 'projected',
      sourceWeightName: 'projection.weight',
      packedWeightName: 'projection.weight.i8',
      sourceBiasName: 'projection.bias',
      packedBiasName: 'projection.bias.i32',
    })],
  });
  assert.deepEqual(CreatePtqPlanRequest.fromBinary(create.toBinary()), create);

  const write = new WritePtqPackageRequest({
    ptqPlanId: 'ptq-plan-4',
    outputGraphPath: 'dist/quantized/graph.json',
    outputWeightsPath: 'dist/quantized/model.safetensors',
  });
  assert.deepEqual(WritePtqPackageRequest.fromBinary(write.toBinary()), write);
});

test('slim inference/full enum projections exactly match generated Synurang enums', () => {
  const screamingSnake = (name) => name.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toUpperCase();
  const assertProjection = (contract, generated, prefix) => {
    for (const [name, value] of Object.entries(contract)) {
      if (typeof value !== 'number') continue;
      assert.equal(generated[`${prefix}${screamingSnake(name)}`], value, name);
    }
  };

  assertProjection(ContractNativeStatus, NativeStatus, 'NATIVE_STATUS_');
  assertProjection(ContractOperationStage, OperationStage, 'OPERATION_STAGE_');
  assertProjection(ContractDataType, DataType, 'DATA_TYPE_');
  assertProjection(ContractMemoryLocation, MemoryLocation, 'MEMORY_LOCATION_');
  assertProjection(ContractBackendPolicyMode, BackendPolicyMode, 'BACKEND_POLICY_MODE_');
  assertProjection(ContractOperatorFallback, OperatorFallback, 'OPERATOR_FALLBACK_');
  assertProjection(ContractDecodeRowMode, DecodeRowMode, 'DECODE_ROW_MODE_');
  assertProjection(ContractFullOperationStage, OperationStage, 'OPERATION_STAGE_');
  assertProjection(
    ContractTrainingOptimizerKind,
    TrainingOptimizerKind,
    'TRAINING_OPTIMIZER_KIND_',
  );
  assertProjection(ContractPtqMode, PtqMode, 'PTQ_MODE_');
  assertProjection(ContractPtqScheme, PtqScheme, 'PTQ_SCHEME_');
  assertProjection(ContractPtqLayerKind, PtqLayerKind, 'PTQ_LAYER_KIND_');

  assert.deepEqual(runtimeDTypes, ['float32', 'int8', 'uint8', 'int32']);
  assert.deepEqual(memoryLocations, ['host', 'device']);
  assert.deepEqual(backendPolicyModes, ['prefer', 'require']);
  assert.deepEqual(operatorFallbackValues, ['allow', 'forbid']);
  assert.deepEqual(decodeRowModes, ['disabled', 'auto', 'required']);
  assert.deepEqual(trainingOptimizerKinds, ['adamw', 'sgd']);
  assert.deepEqual(ptqModes, ['w8a8']);
  assert.deepEqual(ptqSchemes, ['symmetric', 'asymmetric']);
  assert.deepEqual(ptqLayerKinds, ['qlinear', 'qconv2d']);
  assert.equal(nativeStatusCodes[0], 'OK');
  assert.equal(nativeStatusCodes.at(-1), 'REVISION_CONFLICT');
  assert.ok(operationStages.includes('adapter'));
  assert.ok(fullOperationStages.includes('trainer-export'));
  assert.ok(fullOperationStages.includes('ptq-write'));
});

test('training optimizer numbers use the protobuf contract and default to AdamW', () => {
  assert.equal(normalizeTrainingUpdateMode(), 'adamw');
  assert.equal(normalizeTrainingUpdateMode(ContractTrainingOptimizerKind.Adamw), 'adamw');
  assert.equal(normalizeTrainingUpdateMode(ContractTrainingOptimizerKind.Sgd), 'sgd');
  assert.throws(() => normalizeTrainingUpdateMode(2), /SGD or AdamW/);
});

test('generated FFI client exposes the complete application lifecycle route', () => {
  assert.deepEqual(Object.values(RuntimeServiceMethods), [
    '/volvoxai.runtime.RuntimeService/CreateRuntime',
    '/volvoxai.runtime.RuntimeService/CloseRuntime',
    '/volvoxai.runtime.RuntimeService/ReleaseRuntime',
    '/volvoxai.runtime.RuntimeService/LoadModel',
    '/volvoxai.runtime.RuntimeService/GetModelRevision',
    '/volvoxai.runtime.RuntimeService/PublishAdapter',
    '/volvoxai.runtime.RuntimeService/ReleaseModel',
    '/volvoxai.runtime.RuntimeService/CreateTrainer',
    '/volvoxai.runtime.RuntimeService/CloseTrainer',
    '/volvoxai.runtime.RuntimeService/ReleaseTrainer',
    '/volvoxai.runtime.RuntimeService/SetTrainerInput',
    '/volvoxai.runtime.RuntimeService/TrainStep',
    '/volvoxai.runtime.RuntimeService/CommitTrainer',
    '/volvoxai.runtime.RuntimeService/RollbackTrainer',
    '/volvoxai.runtime.RuntimeService/ExportTrainerWeights',
    '/volvoxai.runtime.RuntimeService/CreatePtqPlan',
    '/volvoxai.runtime.RuntimeService/ClosePtqPlan',
    '/volvoxai.runtime.RuntimeService/ReleasePtqPlan',
    '/volvoxai.runtime.RuntimeService/CalibratePtqPlan',
    '/volvoxai.runtime.RuntimeService/InspectPtqPlan',
    '/volvoxai.runtime.RuntimeService/WritePtqPackage',
    '/volvoxai.runtime.RuntimeService/CompileModel',
    '/volvoxai.runtime.RuntimeService/GetCompiledModelReport',
    '/volvoxai.runtime.RuntimeService/ReleaseCompiledModel',
    '/volvoxai.runtime.RuntimeService/CreateExecutionContext',
    '/volvoxai.runtime.RuntimeService/CloseExecutionContext',
    '/volvoxai.runtime.RuntimeService/ReleaseExecutionContext',
    '/volvoxai.runtime.RuntimeService/SetInput',
    '/volvoxai.runtime.RuntimeService/Execute',
    '/volvoxai.runtime.RuntimeService/DecodeSeed',
    '/volvoxai.runtime.RuntimeService/DecodeStep',
    '/volvoxai.runtime.RuntimeService/ResetDecode',
    '/volvoxai.runtime.RuntimeService/SelectAdapter',
    '/volvoxai.runtime.RuntimeService/RebindAdapter',
    '/volvoxai.runtime.RuntimeService/GetResult',
    '/volvoxai.runtime.RuntimeService/ReadOutput',
    '/volvoxai.runtime.RuntimeService/ReleaseResult',
  ]);
  assert.equal(
    RuntimeServiceMethods.RegisterProvider,
    undefined,
    'provider callback composition is an SPI, not an application FFI operation',
  );

  const calls = [];
  const client = new RuntimeServiceFfi({
    invoke(serviceName, methodName, data) {
      calls.push({ serviceName, methodName, data });
      return new RuntimeHandle({ runtimeId: 'runtime-1' }).toBinary();
    },
    openStream() {
      throw new Error('unexpected stream');
    },
  });
  const response = client.createRuntime(new CreateRuntimeRequest());
  assert.equal(response.runtimeId, 'runtime-1');
  assert.deepEqual(calls, [{
    serviceName: 'RuntimeService',
    methodName: '/volvoxai.runtime.RuntimeService/CreateRuntime',
    data: new Uint8Array(),
  }]);
});
