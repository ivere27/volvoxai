# Generated full API contract

Source: `proto/volvoxai.proto`. Regenerate with `make proto_codegen`.

Schema SHA-256: `5da3842c3c492794657b5cf40519d85ae253dd297c1f06fcf0b306219118eedf`.

Protobuf defaults and engine defaults are distinct. Required flags and
structured rules are explicit annotations; remaining semantic constraints
are described in the source comments and enforced by C. An absent rule
does not imply that every value or state is accepted.

`DescribeApi` returns the same contracts, filtered to the current build.
Use `GetPlatformInfo` and `ListBackends` for build/transport and runtime
backend availability; `CompileModel` decides model-specific admission.

## VxPlatformService.GetPlatformInfo

`volvoxai.v1.Empty` → `volvoxai.v1.PlatformInfo`

Effect: `API_EFFECT_READ_ONLY`.



## VxPlatformService.DescribeApi

`volvoxai.v1.DescribeApiRequest` → `volvoxai.v1.ApiDescription`

Effect: `API_EFFECT_READ_ONLY`.

Discover the methods available in this build and their schema contracts.

## VxPlatformService.GetMonotonicTime

`volvoxai.v1.Empty` → `volvoxai.v1.MonotonicTime`

Effect: `API_EFFECT_READ_ONLY`.



## VxPlatformService.DescribeStatus

`volvoxai.v1.DescribeStatusRequest` → `volvoxai.v1.StatusDescription`

Effect: `API_EFFECT_READ_ONLY`.



## VxProfilingService.StartTrace

`volvoxai.v1.StartTraceRequest` → `volvoxai.v1.TraceInfo`

Effect: `API_EFFECT_CREATE`.



- `3`: `runtime_id`

## VxProfilingService.StopTrace

`volvoxai.v1.TraceRef` → `volvoxai.v1.TraceInfo`

Effect: `API_EFFECT_MUTATE`.

Close admission; poll until READY when accepted host operations are still running.

- `3`: `trace_id`

## VxProfilingService.GetTrace

`volvoxai.v1.TraceRef` → `volvoxai.v1.TraceInfo`

Effect: `API_EFFECT_READ_ONLY`.

Inspect collection or poll draining observations without stopping admission.

- `3`: `trace_id`

## VxProfilingService.ReadTrace

`volvoxai.v1.ReadTraceRequest` → `volvoxai.v1.TracePage`

Effect: `API_EFFECT_READ_ONLY`.

READY traces are immutable. Pagination is repeatable and does not consume records.

- `3`: `trace_id`

## VxProfilingService.ExportChromeTrace

`volvoxai.v1.ExportChromeTraceRequest` → `volvoxai.v1.TraceChunk`

Effect: `API_EFFECT_READ_ONLY`.

Concatenate JSON fragments from offset zero until eof. Offsets count source
events, as in ReadTrace; each call serializes only its requested page.

- `3`: `trace_id`

## VxProfilingService.ReleaseTrace

`volvoxai.v1.TraceRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.

Stops collection and retires the handle. Accepted work retains its collector safely.

## VxProfilingService.GetMemorySnapshot

`volvoxai.v1.GetMemorySnapshotRequest` → `volvoxai.v1.MemorySnapshotResponse`

Effect: `API_EFFECT_READ_ONLY`.

On-demand observation. Does not require or start a trace.

## VxTextService.CreateTokenizer

`volvoxai.v1.CreateTokenizerRequest` → `volvoxai.v1.TokenizerHandle`

Effect: `API_EFFECT_CREATE`.



## VxTextService.EncodeText

`volvoxai.v1.EncodeTextRequest` → `volvoxai.v1.EncodedText`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `tokenizer_id`

## VxTextService.DecodeTokens

`volvoxai.v1.DecodeTokensRequest` → `volvoxai.v1.DecodedText`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `tokenizer_id`

## VxTextService.ReleaseTokenizer

`volvoxai.v1.TokenizerRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.CreateRuntime

`volvoxai.v1.CreateRuntimeRequest` → `volvoxai.v1.RuntimeHandle`

Effect: `API_EFFECT_CREATE`.



## VxInferenceService.ReleaseRuntime

`volvoxai.v1.RuntimeRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.ListBackends

`volvoxai.v1.RuntimeRef` → `volvoxai.v1.BackendList`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `runtime_id`

## VxInferenceService.LoadModel

`volvoxai.v1.LoadModelRequest` → `volvoxai.v1.ModelHandle`

Effect: `API_EFFECT_CREATE`.



- `3`: `runtime_id`

## VxInferenceService.GetModelInfo

`volvoxai.v1.ModelRef` → `volvoxai.v1.ModelInfo`

Effect: `API_EFFECT_READ_ONLY`.

Inspect the immutable logical input/output contract without compiling or
executing the model. Symbol names, bounds and divisibility are preserved.

- `3`: `model_id`

## VxInferenceService.GetModelRevision

`volvoxai.v1.ModelRef` → `volvoxai.v1.RevisionInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `model_id`

## VxInferenceService.PublishAdapter

`volvoxai.v1.PublishAdapterRequest` → `volvoxai.v1.AdapterRevision`

Effect: `API_EFFECT_MUTATE`.



- `3`: `model_id`

## VxInferenceService.ReleaseModel

`volvoxai.v1.ModelRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.CompileModel

`volvoxai.v1.CompileModelRequest` → `volvoxai.v1.CompiledModelHandle`

Effect: `API_EFFECT_CREATE`.



- `3`: `model_id`

## VxInferenceService.ReleaseCompiledModel

`volvoxai.v1.CompiledModelRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.Run

`volvoxai.v1.RunRequest` → `volvoxai.v1.ExecutionResultHandle`

Effect: `API_EFFECT_EXECUTE`.

Direct submission of one logical request against a compiled
route, including a caller-authored bulk B=N binding. Allocates no request
handle and no Runtime coordinator. A busy route returns BUSY and never
falls back to scheduling.

- `3`: `compiled_model_id`
- `6`: `inputs`

## VxInferenceService.CreateExecutionContext

`volvoxai.v1.CreateExecutionContextRequest` → `volvoxai.v1.ExecutionContextHandle`

Effect: `API_EFFECT_CREATE`.



- `3`: `compiled_model_id`

## VxInferenceService.ReleaseExecutionContext

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.GetInputAffineQuantization

`volvoxai.v1.GetInputAffineQuantizationRequest` → `volvoxai.v1.AffineQuantization`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `context_id`

## VxInferenceService.Execute

`volvoxai.v1.ExecuteRequest` → `volvoxai.v1.ExecutionResultHandle`

Effect: `API_EFFECT_EXECUTE`.



- `3`: `context_id`
- `6`: `inputs`

## VxInferenceService.GetTensorInteropInfo

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.TensorInteropInfo`

Effect: `API_EFFECT_READ_ONLY`.

Query backend representation and producer ordering. No address is returned.

- `3`: `context_id`

## VxInferenceService.ExecuteTensors

`volvoxai.v1.ExecuteTensorsRequest` → `volvoxai.v1.TensorBatch`

Effect: `API_EFFECT_EXECUTE`.

Completes execution before returning independently retained, mutable
output tensors. Non-host buffer inputs must match the context's backend and
device. Producers keep buffers alive and unchanged until completion.
CUDA producers order writes on the reported consumer_stream. Other native
producers finish writes before calling. GPU snapshots stay on the GPU.
Inputs, reuse_inputs and feedback together bind every model input exactly
once. References use this context's last successful ExecuteTensors call;
other executions or a failure after input commit invalidate that state.
Validation failures before commit preserve it. Selecting outputs limits
snapshots, not graph computation. Unselected outputs remain available for
the next call's feedback without publishing buffer handles. CPU WASM
accepts inline inputs and buffers owned by the same module. WebGPU
retained GPU tensor execution is not yet supported.

- `3`: `context_id`

## VxInferenceService.ExecutePrefix

`volvoxai.v1.ExecutePrefixRequest` → `volvoxai.v1.ExecutionResultHandle`

Effect: `API_EFFECT_EXECUTE`.

Recompute only the leading row_count rows of a fixed-shape sequence
graph. Ordinary execution without retained decode/KV state.

- `3`: `context_id`
- `6`: `inputs`

## VxInferenceService.DecodePrefill

`volvoxai.v1.DecodePrefillRequest` → `volvoxai.v1.ExecutionResultHandle`

Effect: `API_EFFECT_EXECUTE`.



- `3`: `context_id`
- `6`: `inputs`

## VxInferenceService.DecodeStep

`volvoxai.v1.DecodeStepRequest` → `volvoxai.v1.ExecutionResultHandle`

Effect: `API_EFFECT_EXECUTE`.



- `3`: `context_id`
- `5`: `context_id`

## VxInferenceService.DecodeGenerate

`volvoxai.v1.DecodeGenerateRequest` → `volvoxai.v1.ExecutionResultHandle`

Effect: `API_EFFECT_EXECUTE`.



- `3`: `context_id`
- `5`: `context_id`

## VxInferenceService.GetDecodeState

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.DecodeContextState`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `context_id`

## VxInferenceService.ResetDecode

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `3`: `context_id`

## VxInferenceService.ConfigureDecodeCache

`volvoxai.v1.ConfigureDecodeCacheRequest` → `volvoxai.v1.DecodeCacheState`

Effect: `API_EFFECT_MUTATE`.



- `3`: `context_id`
- `5`: `context_id`

## VxInferenceService.GetDecodeCache

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.DecodeCacheState`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `context_id`

## VxInferenceService.PublishDecodePrefix

`volvoxai.v1.PublishDecodePrefixRequest` → `volvoxai.v1.DecodeCacheState`

Effect: `API_EFFECT_MUTATE`.



- `3`: `context_id`

## VxInferenceService.ReuseDecodePrefix

`volvoxai.v1.ReuseDecodePrefixRequest` → `volvoxai.v1.DecodeCacheState`

Effect: `API_EFFECT_MUTATE`.



- `3`: `context_id`

## VxInferenceService.ReleaseDecodeLane

`volvoxai.v1.DecodeLaneRef` → `volvoxai.v1.DecodeCacheState`

Effect: `API_EFFECT_MUTATE`.



- `3`: `context_id`

## VxInferenceService.EvictDecodePrefixes

`volvoxai.v1.EvictDecodePrefixesRequest` → `volvoxai.v1.DecodeCacheState`

Effect: `API_EFFECT_MUTATE`.



- `3`: `context_id`

## VxInferenceService.SelectAdapter

`volvoxai.v1.SelectAdapterRequest` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `3`: `context_id`

## VxInferenceService.RebindAdapter

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.

Adopts the single model revision most recently published by
PublishAdapter. Existing contexts never change revisions implicitly.

- `3`: `context_id`

## VxInferenceService.GetResult

`volvoxai.v1.ResultRef` → `volvoxai.v1.ResultInfo`

Effect: `API_EFFECT_READ_ONLY`.

Nonblocking completion query. PENDING is successful acceptance, not BUSY.
READY exposes all outputs. FAILED carries the execution failure report.

- `3`: `result_id`

## VxInferenceService.ReadOutput

`volvoxai.v1.ReadOutputRequest` → `volvoxai.v1.ReadOutputResponse`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `result_id`

## VxInferenceService.ReleaseResult

`volvoxai.v1.ResultRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxSchedulerService.Submit

`volvoxai.v1.SubmitRequest` → `volvoxai.v1.RequestHandle`

Effect: `API_EFFECT_CREATE`.



- `3`: `compiled_model_id`
- `4`: `compiled_model_id`
- `6`: `inputs`

## VxSchedulerService.PollRequest

`volvoxai.v1.RequestRef` → `volvoxai.v1.RequestInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `request_id`

## VxSchedulerService.WaitRequest

`volvoxai.v1.RequestRef` → `volvoxai.v1.RequestInfo`

Effect: `API_EFFECT_READ_ONLY`.

Completes asynchronously when the request becomes terminal. A call deadline
or cancellation ends only this wait; CancelRequest cancels engine work.

- `3`: `request_id`

## VxSchedulerService.CancelRequest

`volvoxai.v1.RequestRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `3`: `request_id`

## VxSchedulerService.TakeRequestResult

`volvoxai.v1.RequestRef` → `volvoxai.v1.ExecutionResultHandle`

Effect: `API_EFFECT_MUTATE`.

Transfers the retained immutable result to the caller, who releases it
through VxInferenceService.ReleaseResult.

- `3`: `request_id`

## VxSchedulerService.ReleaseRequest

`volvoxai.v1.RequestRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxSchedulerService.CreateBatchQueue

`volvoxai.v1.CreateBatchQueueRequest` → `volvoxai.v1.BatchQueueHandle`

Effect: `API_EFFECT_CREATE`.



## VxSchedulerService.SubmitBatchWork

`volvoxai.v1.SubmitBatchWorkRequest` → `volvoxai.v1.BatchWorkHandle`

Effect: `API_EFFECT_CREATE`.



- `3`: `queue_id`

## VxSchedulerService.NextBatchDispatch

`volvoxai.v1.BatchQueueRef` → `volvoxai.v1.BatchDispatch`

Effect: `API_EFFECT_MUTATE`.



- `3`: `queue_id`

## VxSchedulerService.CompleteBatchDispatch

`volvoxai.v1.CompleteBatchDispatchRequest` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `3`: `queue_id`

## VxSchedulerService.GetBatchWork

`volvoxai.v1.BatchWorkRef` → `volvoxai.v1.BatchWorkInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `queue_id`

## VxSchedulerService.TakeBatchWork

`volvoxai.v1.BatchWorkRef` → `volvoxai.v1.BatchWorkInfo`

Effect: `API_EFFECT_MUTATE`.

Copies a terminal value/result packet and removes it from queue retention.
Returns BUSY for active work and NOT_FOUND after an earlier take/eviction.

- `3`: `queue_id`

## VxSchedulerService.GetBatchQueue

`volvoxai.v1.BatchQueueRef` → `volvoxai.v1.BatchQueueInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `queue_id`

## VxSchedulerService.CancelBatchWork

`volvoxai.v1.BatchWorkRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `3`: `queue_id`

## VxSchedulerService.CloseBatchQueue

`volvoxai.v1.CloseBatchQueueRequest` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `3`: `queue_id`

## VxSchedulerService.ReleaseBatchQueue

`volvoxai.v1.BatchQueueRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxPlanningService.CreateGraphPlan

`volvoxai.v1.CreateGraphPlanRequest` → `volvoxai.v1.GraphPlanHandle`

Effect: `API_EFFECT_CREATE`.



## VxPlanningService.GetGraphPlan

`volvoxai.v1.GraphPlanRef` → `volvoxai.v1.GraphPlanInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `graph_plan_id`

## VxPlanningService.EditGraphPlan

`volvoxai.v1.EditGraphPlanRequest` → `volvoxai.v1.GraphPlanHandle`

Effect: `API_EFFECT_CREATE`.



- `3`: `graph_plan_id`

## VxPlanningService.ExportGraphPlan

`volvoxai.v1.GraphPlanRef` → `volvoxai.v1.ExportedGraphPlan`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `graph_plan_id`

## VxPlanningService.SerializeGraph

`volvoxai.v1.GraphDefinition` → `volvoxai.v1.SerializedGraph`

Effect: `API_EFFECT_READ_ONLY`.

Serialize typed authoring state without requiring weight/scale payloads.
CreateGraphPlan or LoadModel performs binding and semantic validation.

## VxPlanningService.ResolveGraphPlan

`volvoxai.v1.ResolveGraphPlanRequest` → `volvoxai.v1.ResolvedGraphPlan`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `graph_plan_id`

## VxPlanningService.ReleaseGraphPlan

`volvoxai.v1.GraphPlanRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxPlanningService.InspectSafetensors

`volvoxai.v1.InspectSafetensorsRequest` → `volvoxai.v1.SafetensorsInfo`

Effect: `API_EFFECT_READ_ONLY`.



## VxPlanningService.ReadSafetensors

`volvoxai.v1.ReadSafetensorsRequest` → `volvoxai.v1.SafetensorsContent`

Effect: `API_EFFECT_READ_ONLY`.



## VxPlanningService.WriteSafetensors

`volvoxai.v1.WriteSafetensorsRequest` → `volvoxai.v1.SafetensorsArtifact`

Effect: `API_EFFECT_CREATE`.



## VxTrainingService.InitializeTensor

`volvoxai.v1.InitializeTensorRequest` → `volvoxai.v1.InitializedTensor`

Effect: `API_EFFECT_CREATE`.



## VxTrainingService.BuildLoraLinear

`volvoxai.v1.BuildLoraLinearRequest` → `volvoxai.v1.LoraLinearPlan`

Effect: `API_EFFECT_CREATE`.



- `3`: `graph_plan_id`

## VxTrainingService.BuildRoutedAdapter

`volvoxai.v1.BuildRoutedAdapterRequest` → `volvoxai.v1.RoutedAdapterPlan`

Effect: `API_EFFECT_CREATE`.



- `3`: `graph_plan_id`

## VxTrainingService.CreateTrainer

`volvoxai.v1.CreateTrainerRequest` → `volvoxai.v1.TrainerHandle`

Effect: `API_EFFECT_CREATE`.



- `3`: `model_id`

## VxTrainingService.ReleaseTrainer

`volvoxai.v1.TrainerRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxTrainingService.TrainStep

`volvoxai.v1.TrainStepRequest` → `volvoxai.v1.TrainStepResult`

Effect: `API_EFFECT_EXECUTE`.



- `3`: `trainer_id`

## VxTrainingService.GetTrainStep

`volvoxai.v1.TrainStepRef` → `volvoxai.v1.TrainStepResult`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `trainer_id`

## VxTrainingService.ReadTrainerParameters

`volvoxai.v1.ReadTrainerParametersRequest` → `volvoxai.v1.TensorBatch`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `trainer_id`

## VxTrainingService.CommitTrainer

`volvoxai.v1.TrainerRef` → `volvoxai.v1.RevisionInfo`

Effect: `API_EFFECT_MUTATE`.



- `3`: `trainer_id`

## VxTrainingService.RollbackTrainer

`volvoxai.v1.TrainerRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `3`: `trainer_id`

## VxTrainingService.ExportTrainerWeights

`volvoxai.v1.ExportTrainerWeightsRequest` → `volvoxai.v1.TrainerWeights`

Effect: `API_EFFECT_CREATE`.



- `3`: `trainer_id`

## VxTrainingService.GetTrainerState

`volvoxai.v1.TrainerRef` → `volvoxai.v1.TrainerState`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `trainer_id`

## VxTrainingService.ResetTrainerAccumulation

`volvoxai.v1.TrainerRef` → `volvoxai.v1.TrainerState`

Effect: `API_EFFECT_MUTATE`.



- `3`: `trainer_id`

## VxTrainingService.ExportTrainerCheckpoint

`volvoxai.v1.ExportTrainerCheckpointRequest` → `volvoxai.v1.TrainerCheckpointInfo`

Effect: `API_EFFECT_CREATE`.



- `3`: `trainer_id`

## VxQuantizationService.ExportQuantizedTrainerWeights

`volvoxai.v1.ExportQuantizedTrainerWeightsRequest` → `volvoxai.v1.QuantizedTrainerWeights`

Effect: `API_EFFECT_CREATE`.



- `3`: `trainer_id`

## VxQuantizationService.QuantizeWeight

`volvoxai.v1.QuantizeWeightRequest` → `volvoxai.v1.QuantizedWeight`

Effect: `API_EFFECT_CREATE`.



## VxQuantizationService.DequantizeWeight

`volvoxai.v1.DequantizeWeightRequest` → `volvoxai.v1.InitializedTensor`

Effect: `API_EFFECT_CREATE`.



## VxQuantizationService.AuthorPtqTemplate

`volvoxai.v1.AuthorPtqTemplateRequest` → `volvoxai.v1.PtqTemplateInfo`

Effect: `API_EFFECT_CREATE`.



## VxQuantizationService.CreatePtqPlan

`volvoxai.v1.CreatePtqPlanRequest` → `volvoxai.v1.PtqPlanHandle`

Effect: `API_EFFECT_CREATE`.



- `3`: `model_id`

## VxQuantizationService.ReleasePtqPlan

`volvoxai.v1.PtqPlanRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxQuantizationService.CalibratePtqPlan

`volvoxai.v1.CalibratePtqPlanRequest` → `volvoxai.v1.PtqCalibrationInfo`

Effect: `API_EFFECT_EXECUTE`.



- `3`: `ptq_plan_id`

## VxQuantizationService.InspectPtqPlan

`volvoxai.v1.PtqPlanRef` → `volvoxai.v1.PtqPlanInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `ptq_plan_id`

## VxQuantizationService.WritePtqPackage

`volvoxai.v1.WritePtqPackageRequest` → `volvoxai.v1.PtqPackageInfo`

Effect: `API_EFFECT_CREATE`.



- `3`: `ptq_plan_id`

## VxBufferService.GetBufferInfo

`volvoxai.v1.BufferHandle` → `volvoxai.v1.BufferInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `3`: `buffer_id`

## VxBufferService.AllocateBuffers

`volvoxai.v1.AllocateBuffersRequest` → `volvoxai.v1.BufferHandles`

Effect: `API_EFFECT_CREATE`.



## VxBufferService.RetainBuffers

`volvoxai.v1.BufferRefs` → `volvoxai.v1.BufferHandles`

Effect: `API_EFFECT_CREATE`.



## VxBufferService.ReleaseBuffers

`volvoxai.v1.BufferRefs` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxBufferService.CopyTensors

`volvoxai.v1.CopyTensorsRequest` → `volvoxai.v1.TensorBatch`

Effect: `API_EFFECT_EXECUTE`.



## VxBufferService.BeginBufferAccess

`volvoxai.v1.BufferAccessRequest` → `volvoxai.v1.BufferAccess`

Effect: `API_EFFECT_CREATE`.



## VxBufferService.EndBufferAccess

`volvoxai.v1.EndBufferAccessRequest` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxBufferService.ImportDLPack

`volvoxai.v1.ImportDLPackRequest` → `volvoxai.v1.TensorBatch`

Effect: `API_EFFECT_CREATE`.



## VxBufferService.ExportDLPack

`volvoxai.v1.ExportDLPackRequest` → `volvoxai.v1.DLPackExport`

Effect: `API_EFFECT_CREATE`.



## volvoxai.v1.VocabularyToken



### id (1)

`uint32`.

Protobuf default: `0`.

### value (2)

`bytes`.

Protobuf default: `""`.

## volvoxai.v1.TokenizerVocabulary



### tokens (1)

`volvoxai.v1.VocabularyToken` repeated.

Protobuf default: `[]`.

## volvoxai.v1.CreateTokenizerRequest



### tokens (1)

`volvoxai.v1.TokenizerVocabulary`; oneof `vocabulary`.

Protobuf default: `null`.

### vocabulary_json (2)

`bytes`; oneof `vocabulary`.

Protobuf default: `""`.

### vocabulary_binary (3)

`bytes`; oneof `vocabulary`.

Protobuf default: `""`.

### merges (4)

`bytes`.

Protobuf default: `""`.

UTF-8 merge lines, ordered by priority. Blank/comment lines, unresolved
pairs and repeated pairs are ignored. Leading U+0120 denotes a space.

## volvoxai.v1.TokenizerRef



### tokenizer_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Tokenizer`.

## volvoxai.v1.TokenizerHandle



### tokenizer_id (1)

`int64`.

Protobuf default: `"0"`.

### vocabulary_size (2)

`uint32`.

Protobuf default: `0`.

### merge_count (3)

`uint32`.

Protobuf default: `0`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.EncodeTextRequest



### tokenizer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Tokenizer`.

### text (2)

`string`.

Protobuf default: `""`.

### max_tokens (3)

`uint32`.

Protobuf default: `0`.

Engine default: 256

Default 256; explicit zero returns empty.

### mode (4)

`volvoxai.v1.TokenizationMode`.

Protobuf default: `"TOKENIZATION_MODE_AUTO"`.

## volvoxai.v1.EncodedText



### tokens (1)

`uint32` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.DecodeTokensRequest



### tokenizer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Tokenizer`.

### tokens (2)

`uint32` repeated.

Protobuf default: `[]`.

## volvoxai.v1.DecodedText



### text (1)

`string`.

Protobuf default: `""`.

Decode each token independently with UTF-8 replacement, then replace
U+0120 with space and concatenate. Unknown IDs contribute empty text.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchCacheOptions



### lanes (1)

`uint32`.

Protobuf default: `0`.

### page_tokens (2)

`uint32`.

Protobuf default: `0`.

### lane_token_capacity (3)

`uint32`.

Protobuf default: `0`.

### max_pages (4)

`uint32`.

Protobuf default: `0`.

default: fully private lane capacity

### policy (5)

`volvoxai.v1.DecodeCachePolicy`.

Protobuf default: `"DECODE_CACHE_POLICY_PAGED"`.

## volvoxai.v1.CreateBatchQueueRequest



### max_queue_depth (1)

`uint32`.

Protobuf default: `0`.

default: 4 * lane count

### token_budget (2)

`uint32`.

Protobuf default: `0`.

per dispatch, including padding; default 2048

### max_lanes (3)

`uint32`.

Protobuf default: `0`.

default: cache lanes, or 8 without a cache

### multiple_of (4)

`uint32`.

Protobuf default: `0`.

stateless batch divisibility; default 1

### policy (5)

`volvoxai.v1.BatchQueuePolicy`.

Protobuf default: `"BATCH_QUEUE_POLICY_WORK_CONSERVING"`.

### max_wait_ns (6)

`uint64`.

Protobuf default: `"0"`.

fill-first only; rounded up to the scheduler clock tick

### queue_depth_window (7)

`uint32`.

Protobuf default: `0`.

default 1024 recent rounds

### retained_results (8)

`uint32`.

Protobuf default: `0`.

default: 2 * queue depth + lane count

### cache (9)

`volvoxai.v1.BatchCacheOptions`.

Protobuf default: `null`.

required for decode work; metadata only

## volvoxai.v1.BatchQueueRef



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

## volvoxai.v1.BatchQueueHandle



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchGroup



### model (1)

`string`.

Protobuf default: `""`.

omitted/empty defaults to "default"

### adapter_revision (2)

`string`.

Protobuf default: `""`.

absence differs from an empty revision

### shape_signature (3)

`string`.

Protobuf default: `""`.

defaults to "stateless" or "llm"

## volvoxai.v1.StatelessBatchWork



### rows (1)

`uint32`.

Protobuf default: `0`.

### inputs (2)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

copied at submission, returned unchanged to worker

## volvoxai.v1.DecodeBatchWork



### prompt_tokens (1)

`uint32`.

Protobuf default: `0`.

### max_tokens (2)

`uint32`.

Protobuf default: `0`.

### prompt_key (3)

`string`.

Protobuf default: `""`.

Context-local identity of the complete prompt semantics, including model,
adapter and positional inputs. Only complete prompt pages are published.

## volvoxai.v1.SubmitBatchWorkRequest



### queue_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

### group (2)

`volvoxai.v1.BatchGroup`.

Protobuf default: `null`.

### payload (3)

`bytes`.

Protobuf default: `""`.

uninterpreted application data, copied at submission

### stateless (4)

`volvoxai.v1.StatelessBatchWork`; oneof `work`.

Protobuf default: `null`.

### decode (5)

`volvoxai.v1.DecodeBatchWork`; oneof `work`.

Protobuf default: `null`.

## volvoxai.v1.BatchWorkRef



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

### work_id (2)

`uint32`.

Protobuf default: `0`.

Handle kind: `BatchWork`.

queue-scoped, never reused

## volvoxai.v1.BatchWorkHandle



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### work_id (2)

`uint32`.

Protobuf default: `0`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchDispatchWork



### work_id (1)

`uint32`.

Protobuf default: `0`.

zero for a padding item

### rows (2)

`uint32`.

Protobuf default: `0`.

### lane (3)

`sint32`.

Protobuf default: `0`.

-1 for stateless work

### lane_generation (4)

`uint32`.

Protobuf default: `0`.

### position (5)

`uint32`.

Protobuf default: `0`.

### tokens (6)

`uint32`.

Protobuf default: `0`.

### kv_length (7)

`uint32`.

Protobuf default: `0`.

### pages (8)

`sint32` repeated.

Protobuf default: `[]`.

logical-to-physical pages; -1 is unmapped

### page_tokens (9)

`uint32`.

Protobuf default: `0`.

### generated (10)

`uint32`.

Protobuf default: `0`.

### padding (11)

`bool`.

Protobuf default: `false`.

duplicates the first item's inputs; outcome is discarded

### payload (12)

`bytes`.

Protobuf default: `""`.

### inputs (13)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

## volvoxai.v1.BatchDispatch



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### dispatch_id (2)

`uint64`.

Protobuf default: `"0"`.

Zero means no work is ready. The same nonzero dispatch is returned until
completed; polling never repeats admission or a reservation. Only one
dispatch is exposed at a time. A round visits each ready group once.

### kind (3)

`volvoxai.v1.BatchWorkKind`.

Protobuf default: `"BATCH_WORK_KIND_STATELESS"`.

### group (4)

`volvoxai.v1.BatchGroup`.

Protobuf default: `null`.

### rows_per_item (5)

`uint32`.

Protobuf default: `0`.

### useful_items (6)

`uint32`.

Protobuf default: `0`.

### useful_rows (7)

`uint32`.

Protobuf default: `0`.

### total_rows (8)

`uint32`.

Protobuf default: `0`.

### work (9)

`volvoxai.v1.BatchDispatchWork` repeated.

Protobuf default: `[]`.

### round (10)

`uint64`.

Protobuf default: `"0"`.

### report (11)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchWorkOutcome



### finished (1)

`bool`.

Protobuf default: `false`.

stop decode before max_tokens; ignored for prefill

### value (2)

`bytes`.

Protobuf default: `""`.

appended to this request's value history

### outputs (3)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

copied; replaces this request's last outputs

### error (4)

`string`.

Protobuf default: `""`.

any useful item's error fails the entire dispatch

## volvoxai.v1.CompleteBatchDispatchRequest



### queue_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

### dispatch_id (2)

`uint64`.

Protobuf default: `"0"`.

### outcomes (3)

`volvoxai.v1.BatchWorkOutcome` repeated.

Protobuf default: `[]`.

Exactly one outcome per dispatched item, including padding, unless error
is supplied. Invalid IDs/counts/payloads leave the pending dispatch intact.

### error (4)

`string`.

Protobuf default: `""`.

worker-wide failure; rolls back reservations

## volvoxai.v1.BatchValue



### data (1)

`bytes`.

Protobuf default: `""`.

## volvoxai.v1.BatchWorkInfo



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### work_id (2)

`uint32`.

Protobuf default: `0`.

### state (3)

`volvoxai.v1.BatchWorkState`.

Protobuf default: `"BATCH_WORK_STATE_UNKNOWN"`.

### generated (4)

`uint32`.

Protobuf default: `0`.

### lane (5)

`sint32`.

Protobuf default: `0`.

### values (6)

`volvoxai.v1.BatchValue` repeated.

Protobuf default: `[]`.

### outputs (7)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### error (8)

`string`.

Protobuf default: `""`.

### report (9)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchQueueInfo



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### closed (2)

`bool`.

Protobuf default: `false`.

### draining (3)

`bool`.

Protobuf default: `false`.

### pending_dispatch_id (4)

`uint64`.

Protobuf default: `"0"`.

### rounds (5)

`uint64`.

Protobuf default: `"0"`.

### dispatches (6)

`uint64`.

Protobuf default: `"0"`.

### rows_dispatched (7)

`uint64`.

Protobuf default: `"0"`.

### rows_useful (8)

`uint64`.

Protobuf default: `"0"`.

### worker_busy_ns (9)

`uint64`.

Protobuf default: `"0"`.

Time from first exposure to completion, summed over nonoverlapping worker
dispatches. This measures worker turnaround, not GPU hardware utilization.

### wall_ns (10)

`uint64`.

Protobuf default: `"0"`.

### worker_utilization (11)

`double`.

Protobuf default: `0`.

### padding_waste (12)

`double`.

Protobuf default: `0`.

### queue_depth (13)

`uint32`.

Protobuf default: `0`.

### max_queue_depth_seen (14)

`uint32`.

Protobuf default: `0`.

### queue_depth_p50 (15)

`uint32`.

Protobuf default: `0`.

### queue_depth_p99 (16)

`uint32`.

Protobuf default: `0`.

### group_count (17)

`uint32`.

Protobuf default: `0`.

### active_lanes (18)

`uint32`.

Protobuf default: `0`.

### free_lanes (19)

`uint32`.

Protobuf default: `0`.

### admitted (20)

`uint64`.

Protobuf default: `"0"`.

### completed (21)

`uint64`.

Protobuf default: `"0"`.

### cancelled (22)

`uint64`.

Protobuf default: `"0"`.

### failed (23)

`uint64`.

Protobuf default: `"0"`.

### steps (24)

`uint64`.

Protobuf default: `"0"`.

### admission_stalls (25)

`uint64`.

Protobuf default: `"0"`.

### queue_delay_rounds (26)

`uint64`.

Protobuf default: `"0"`.

### prefix_reuses (27)

`uint64`.

Protobuf default: `"0"`.

### prefix_publications (28)

`uint64`.

Protobuf default: `"0"`.

### prefill_tokens (29)

`uint64`.

Protobuf default: `"0"`.

### shared_prompt_tokens (30)

`uint64`.

Protobuf default: `"0"`.

### active_records (31)

`uint32`.

Protobuf default: `0`.

### retained_results (32)

`uint32`.

Protobuf default: `0`.

### max_retained_results (33)

`uint32`.

Protobuf default: `0`.

### resident_pages (34)

`uint32`.

Protobuf default: `0`.

### reserved_pages (35)

`uint32`.

Protobuf default: `0`.

### free_pages (36)

`uint32`.

Protobuf default: `0`.

### shared_pages (37)

`uint32`.

Protobuf default: `0`.

### evictions (38)

`uint64`.

Protobuf default: `"0"`.

### report (39)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.CloseBatchQueueRequest



### queue_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

### drain (2)

`bool`.

Protobuf default: `false`.

Stops admission and cancels queued work. With drain, already admitted work
remains available through Next/Complete until terminal. Without drain all
pending work is cancelled and dispatch IDs are revoked. Release always
cancels and frees the policy owner; worker device resources remain its own.

## volvoxai.v1.Empty



## volvoxai.v1.RuntimeRef



### runtime_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Runtime`.

## volvoxai.v1.ModelRef



### model_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Model`.

## volvoxai.v1.CompiledModelRef



### compiled_model_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `CompiledModel`.

## volvoxai.v1.ExecutionContextRef



### context_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

## volvoxai.v1.ResultRef



### result_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `ExecutionResult`.

## volvoxai.v1.RequestRef



### request_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Request`.

## volvoxai.v1.TrainerRef



### trainer_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Trainer`.

## volvoxai.v1.PtqPlanRef



### ptq_plan_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `PtqPlan`.

## volvoxai.v1.GraphPlanRef



### graph_plan_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `GraphPlan`.

## volvoxai.v1.BufferHandle

An owner-scoped capability, never a pointer. Copying an ID does not retain it.
RetainBuffers returns fresh IDs; ReleaseBuffers retires IDs idempotently.

### buffer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Buffer`.

## volvoxai.v1.BufferRefs



### buffer_ids (1)

`int64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.BufferHandles



### buffers (1)

`volvoxai.v1.BufferHandle` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BufferView

A range within the public logical extent, not allocation capacity.

### buffer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Buffer`.

### offset_bytes (2)

`uint64`.

Protobuf default: `"0"`.

### length_bytes (3)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.NativeResource

Local transport only. handle is a host/CUDA pointer, VkBuffer, GLuint, or
id<MTLBuffer>. device_context identifies the owning VkDevice/EGLContext/
MTLDevice; CUDA validates its primary context. These are not wire identities.

### kind (1)

`volvoxai.v1.NativeResourceKind`.

Protobuf default: `"NATIVE_RESOURCE_KIND_UNSPECIFIED"`.

### handle (2)

`uint64`.

Protobuf default: `"0"`.

### size_bytes (3)

`uint64`.

Protobuf default: `"0"`.

### device_id (4)

`int32`.

Protobuf default: `0`.

### device_context (5)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.BorrowedBuffer

Caller-owned local memory. All accesses finish before native dispatch
returns. The caller keeps memory valid and unchanged during the call.
Browser dispatch rejects this descriptor before dereferencing. A remote
native adapter must reject local descriptors before forwarding the call.
Offsets apply to resource contents, never to an opaque object address.

### resource (1)

`volvoxai.v1.NativeResource`.

Protobuf default: `null`.

### offset_bytes (2)

`uint64`.

Protobuf default: `"0"`.

### length_bytes (3)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.Tensor

A dense tensor. name is a binding label, not storage identity. It is ignored
by storage operations, and required by model input and output bindings.
Storage ownership belongs exclusively to buffer.buffer_id.

- `1`: `inline`, `buffer`, `borrowed`

### name (1)

`string`.

Protobuf default: `""`.

### shape (2)

`int64` repeated.

Protobuf default: `[]`.

### dtype (3)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### inline (5)

`bytes`; oneof `payload`.

Protobuf default: `""`.

### buffer (6)

`volvoxai.v1.BufferView`; oneof `payload`.

Protobuf default: `null`.

### borrowed (7)

`volvoxai.v1.BorrowedBuffer`; oneof `payload`.

Protobuf default: `null`.

## volvoxai.v1.TensorBatch

Outputs are independent snapshots unless explicitly documented as shared.

### outputs (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BufferInfo



### size_bytes (1)

`uint64`.

Protobuf default: `"0"`.

### kind (2)

`volvoxai.v1.NativeResourceKind`.

Protobuf default: `"NATIVE_RESOURCE_KIND_UNSPECIFIED"`.

### device_id (3)

`int32`.

Protobuf default: `0`.

### device_context (4)

`uint64`.

Protobuf default: `"0"`.

### cpu_accessible (5)

`bool`.

Protobuf default: `false`.

### read_only (6)

`bool`.

Protobuf default: `false`.

### report (7)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.AllocateBuffersRequest

Allocates zero-initialized CPU storage. Device snapshots are produced by
execution; allocation on arbitrary GPU devices is not yet supported.

### sizes_bytes (1)

`uint64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.CopyTensorsRequest

Produces independent CPU snapshots or explicit host readback. inline_result
returns portable bytes; into specifies one local host destination per source.
Both options cannot be set together. The default returns owned buffer IDs.
Batches validate every source and destination before any destination write.

### sources (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### into (2)

`volvoxai.v1.BorrowedBuffer` repeated.

Protobuf default: `[]`.

### inline_result (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.BufferAccessRequest

Access is an explicit lease. Conflicting writes return BUSY. host_mapping
requires direct CPU access and never introduces a hidden staging copy.

### view (1)

`volvoxai.v1.BufferView`.

Protobuf default: `null`.

### mode (2)

`volvoxai.v1.BufferAccessMode`.

Protobuf default: `"BUFFER_ACCESS_MODE_READ"`.

### host_mapping (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.BufferAccess



### access_id (1)

`int64`.

Protobuf default: `"0"`.

### memory (2)

`volvoxai.v1.BorrowedBuffer`.

Protobuf default: `null`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.CudaStreamCompletion

Local-native CUDA completion evidence. All accesses must already be queued
on these 1..64 streams, on the allocation's device and primary context.
Keep every stream alive through the call. Values are live CUstream handles,
or 1 for the legacy default stream. Zero and the per-thread stream (2) are
rejected: dispatch may run on another host thread. Capturing streams are
unsupported. Arbitrary integers are not safe substitutes for live handles.

### device_id (1)

`int32`.

Protobuf default: `0`.

### streams (2)

`uint64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.EndBufferAccessRequest

Retires the access permission; all views bound to it must stop being used.
With cuda, C records stream completion without a host wait and defers the
dependency until this allocation is accessed/reused. Final destruction or
shutdown may wait for recorded work. The caller must declare every stream
that accessed the range, and must enqueue no further accesses. At most 64
distinct pending consumer streams are retained per allocation; exceeding
that budget returns BUSY and leaves the access permission active.
Without cuda, external work must already be complete; native CUDA also drains
the context conservatively. Retired IDs are harmless. Validation failure
leaves live IDs active; a driver failure can add ordering dependencies but
does not retire IDs. CUDA evidence is rejected on WASM/remote transports.

### access_ids (1)

`int64` repeated.

Protobuf default: `[]`.

### cuda (2)

`volvoxai.v1.CudaStreamCompletion`.

Protobuf default: `null`.

## volvoxai.v1.ImportDLPackRequest

A local pointer to a standard DLManagedTensor or DLManagedTensorVersioned.
A successful import consumes its deleter exactly once, on final release.
Failure leaves ownership with the caller. Producer synchronization must be
complete before import. Dense CPU and CUDA tensors are supported.

### managed_tensor (1)

`uint64`.

Protobuf default: `"0"`.

### versioned (2)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ExportDLPackRequest



### tensor (1)

`volvoxai.v1.Tensor`.

Protobuf default: `null`.

### versioned (2)

`bool`.

Protobuf default: `false`.

### access_id (3)

`int64`.

Protobuf default: `"0"`.

Optional existing writable BeginBufferAccess lease covering this tensor.
EndBufferAccess ends its permission even if DLPack aliases remain alive;
those aliases only retain allocation lifetime and must no longer be used.
Zero creates the ordinary lease whose permission ends at the deleter.

## volvoxai.v1.DLPackExport

The standard DLPack deleter releases the allocation reference and, for an
ordinary export, its writable permission. Scoped permission ends explicitly
via EndBufferAccess. Public IDs may retire earlier. No autograd edge is created.
Keep the native library loaded until every exported deleter has run.

### managed_tensor (1)

`uint64`.

Protobuf default: `"0"`.

### versioned (2)

`bool`.

Protobuf default: `false`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.DimensionConstraint



### kind (1)

`volvoxai.v1.DimensionKind`.

Protobuf default: `"DIMENSION_KIND_UNSPECIFIED"`.

### symbol (2)

`string`.

Protobuf default: `""`.

### min (3)

`int64`.

Protobuf default: `"0"`.

### max (4)

`int64`.

Protobuf default: `"0"`.

### multiple_of (5)

`int64`.

Protobuf default: `"0"`.

## volvoxai.v1.TensorSpec

Logical model contract. Concrete request shapes are carried only by Tensor.
No byte size is reported because a symbolic tensor has many concrete byte
sizes within its bounded domain.

### name (1)

`string`.

Protobuf default: `""`.

### dtype (2)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### dimensions (3)

`volvoxai.v1.DimensionConstraint` repeated.

Protobuf default: `[]`.

### location (4)

`volvoxai.v1.MemoryLocation`.

Protobuf default: `"MEMORY_LOCATION_HOST"`.

## volvoxai.v1.TensorInfo



### name (1)

`string`.

Protobuf default: `""`.

### shape (2)

`int64` repeated.

Protobuf default: `[]`.

### dtype (3)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### byte_size (4)

`uint64`.

Protobuf default: `"0"`.

### location (5)

`volvoxai.v1.MemoryLocation`.

Protobuf default: `"MEMORY_LOCATION_HOST"`.

## volvoxai.v1.TensorInteropInfo



### backend (1)

`string`.

Protobuf default: `""`.

### device (2)

`volvoxai.v1.NativeResource`.

Protobuf default: `null`.

Kind and device identity; handle and size_bytes are zero.

### consumer_stream (3)

`int64`.

Protobuf default: `"0"`.

CUDA legacy default stream (DLPack stream value 1); zero otherwise.
Borrowed producers order writes on this stream before ExecuteTensors.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.AffineQuantization

Resolved per-tensor affine metadata. Numeric values are loaded from the
safetensors tensors referenced by graph.quantization.tensors; they are not
graph JSON parameters.

### defined (1)

`bool`.

Protobuf default: `false`.

### scale (2)

`float`.

Protobuf default: `0`.

### zero_point (3)

`int32`.

Protobuf default: `0`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.NodeRef



### index (1)

`int32`.

Protobuf default: `0`.

### op (2)

`string`.

Protobuf default: `""`.

### output (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.CompilationCandidate



### backend (1)

`string`.

Protobuf default: `""`.

### outcome (2)

`volvoxai.v1.CandidateOutcome`.

Protobuf default: `"CANDIDATE_OUTCOME_UNSPECIFIED"`.

### status (3)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

### reason (4)

`string`.

Protobuf default: `""`.

## volvoxai.v1.CompilationEvidence



### policy_mode (1)

`volvoxai.v1.BackendPolicyMode`.

Protobuf default: `"BACKEND_POLICY_MODE_PREFER"`.

### operator_fallback (2)

`volvoxai.v1.OperatorFallback`.

Protobuf default: `"OPERATOR_FALLBACK_ALLOW"`.

### candidates (3)

`volvoxai.v1.CompilationCandidate` repeated.

Protobuf default: `[]`.

### tier_fallback_used (4)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ShapePlanEvidence

Dynamic-shape binding evidence for one committed shape signature.

### signature_digest (1)

`fixed64`.

Protobuf default: `"0"`.

### signature (2)

`string`.

Protobuf default: `""`.

### origin (3)

`volvoxai.v1.ShapePlanOrigin`.

Protobuf default: `"SHAPE_PLAN_ORIGIN_UNSPECIFIED"`.

### bind_time_ns (4)

`uint64`.

Protobuf default: `"0"`.

### logical_bytes (5)

`uint64`.

Protobuf default: `"0"`.

### arena_required_bytes (6)

`uint64`.

Protobuf default: `"0"`.

### arena_capacity_bytes (7)

`uint64`.

Protobuf default: `"0"`.

### arena_high_water_bytes (8)

`uint64`.

Protobuf default: `"0"`.

### arena_grow_count (9)

`uint64`.

Protobuf default: `"0"`.

### resource_generation (10)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.RouteEvidence

Which backend actually executed. `attested` is false when any active node
lacked a proved route; a consumer must not read the selected backend as
execution proof without it.

### provider (1)

`string`.

Protobuf default: `""`.

### builtin (2)

`bool`.

Protobuf default: `false`.

### attested (3)

`bool`.

Protobuf default: `false`.

### active_nodes (4)

`uint32`.

Protobuf default: `0`.

### selected_nodes (5)

`uint32`.

Protobuf default: `0`.

### fallback_nodes (6)

`uint32`.

Protobuf default: `0`.

### missing_nodes (7)

`uint32`.

Protobuf default: `0`.

### digest (8)

`fixed64`.

Protobuf default: `"0"`.

### shape_plan (9)

`volvoxai.v1.ShapePlanEvidence`.

Protobuf default: `null`.

## volvoxai.v1.FallbackEvidence



### operator_fallback_used (1)

`bool`.

Protobuf default: `false`.

### count (2)

`uint32`.

Protobuf default: `0`.

### first (3)

`volvoxai.v1.NodeRef`.

Protobuf default: `null`.

## volvoxai.v1.DecodeState



### enabled (1)

`bool`.

Protobuf default: `false`.

### prefilled (2)

`bool`.

Protobuf default: `false`.

### mode (3)

`volvoxai.v1.DecodeMode`.

Protobuf default: `"DECODE_MODE_UNSPECIFIED"`.

### last_position (4)

`int32`.

Protobuf default: `0`.

Absent before the first successful step.

## volvoxai.v1.Lineage

Immutable identity of the graph, weight, and adapter generations an
operation ran against.

### execution_id (1)

`uint64`.

Protobuf default: `"0"`.

### runtime_id (2)

`uint64`.

Protobuf default: `"0"`.

### model_id (3)

`uint64`.

Protobuf default: `"0"`.

### compiled_model_id (4)

`uint64`.

Protobuf default: `"0"`.

### context_id (5)

`uint64`.

Protobuf default: `"0"`.

### graph_id (6)

`uint64`.

Protobuf default: `"0"`.

### graph_revision (7)

`uint64`.

Protobuf default: `"0"`.

### weight_id (8)

`uint64`.

Protobuf default: `"0"`.

### weight_revision (9)

`uint64`.

Protobuf default: `"0"`.

### adapter_id (10)

`uint64`.

Protobuf default: `"0"`.

### adapter_revision (11)

`uint64`.

Protobuf default: `"0"`.

### graph_plan_id (12)

`uint64`.

Protobuf default: `"0"`.

Nonzero when a planning operation creates or names a positive GraphPlan
id. A stale positive reference is echoed for diagnosis; a constructor
failure before handle allocation and SafeTensors inspection leave zero.
A model-derived plan also carries the retained Model revision above; a
standalone authoring plan leaves those model fields zero.

## volvoxai.v1.OperationReport



### status (1)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

### stage (2)

`volvoxai.v1.OperationStage`.

Protobuf default: `"OPERATION_STAGE_NONE"`.

### code (3)

`volvoxai.v1.OperationCode`.

Protobuf default: `"OPERATION_CODE_NONE"`.

Machine-readable detail, independent of message wording.

### message (4)

`string`.

Protobuf default: `""`.

### offending_node (5)

`volvoxai.v1.NodeRef`.

Protobuf default: `null`.

### backend (6)

`string`.

Protobuf default: `""`.

### device (7)

`string`.

Protobuf default: `""`.

### lineage (8)

`volvoxai.v1.Lineage`.

Protobuf default: `null`.

### compilation (11)

`volvoxai.v1.CompilationEvidence`.

Protobuf default: `null`.

### route (12)

`volvoxai.v1.RouteEvidence`.

Protobuf default: `null`.

### fallback (13)

`volvoxai.v1.FallbackEvidence`.

Protobuf default: `null`.

### decode (14)

`volvoxai.v1.DecodeState`.

Protobuf default: `null`.

### input_issue (16)

`volvoxai.v1.InputValidationIssue`.

Protobuf default: `null`.

First input violation, reported before the input batch changes state.
Absent for failures unrelated to input validation. Never infer retry
safety for other failures from the absence of this field.

## volvoxai.v1.InputValidationIssue



### code (1)

`volvoxai.v1.InputValidationCode`.

Protobuf default: `"INPUT_VALIDATION_CODE_UNSPECIFIED"`.

### input_name (2)

`string`.

Protobuf default: `""`.

### input_index (3)

`uint64`.

Protobuf default: `"0"`.

Zero-based index into the supplied batch; absent for a missing input.

### axis (4)

`uint32`.

Protobuf default: `0`.

### expected (5)

`volvoxai.v1.TensorSpec`.

Protobuf default: `null`.

Present when a declared input can be identified.

### actual (6)

`volvoxai.v1.TensorInfo`.

Protobuf default: `null`.

Metadata only; no tensor payload is echoed. Absent for missing inputs.

### expected_byte_size (7)

`uint64`.

Protobuf default: `"0"`.

### required_extent (8)

`int64`.

Protobuf default: `"0"`.

Previously bound extent for symbol equality or decode shape continuity.

### names_truncated (9)

`bool`.

Protobuf default: `false`.

Runtime evidence bounds names to 255 bytes and symbols to 63 bytes.
Query GetModelInfo for the complete names when this flag is true.

### actual_rank (10)

`uint64`.

Protobuf default: `"0"`.

Actual rank is preserved even for an invalid native binding exceeding
max_tensor_rank. actual.shape then contains only the first eight axes.

## volvoxai.v1.PlatformInfo



### api_version (1)

`uint32`.

Protobuf default: `0`.

### library_version (2)

`string`.

Protobuf default: `""`.

### profile (3)

`volvoxai.v1.BuildProfile`.

Protobuf default: `"BUILD_PROFILE_UNSPECIFIED"`.

### transport (4)

`volvoxai.v1.TransportProfile`.

Protobuf default: `"TRANSPORT_PROFILE_UNSPECIFIED"`.

### compiled_backends (5)

`string` repeated.

Protobuf default: `[]`.

Backends compiled into this library, in canonical lower-case spelling.
Availability at runtime is reported by VxInferenceService.ListBackends.

### max_tensor_rank (6)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.DescribeApiRequest

Empty filters return the method catalogue for this build. include_types
adds only the transitive request/response types of the selected methods.
A method filter requires a service (e.g. VxInferenceService / Run).
Unknown or unavailable names return NOT_FOUND, never an empty success.

- `2`: `method`, `service`

### service (1)

`string`.

Protobuf default: `""`.

### method (2)

`string`.

Protobuf default: `""`.

### include_types (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ApiRule

Fields are proto field paths, relative to the annotated request/message.
REQUIRES means the first non-default/present field requires the remaining
fields. EXACTLY_ONE counts non-default/present fields (including messages).
These discoverable rules supplement the C validator; graph-dependent
validity and backend admission are still decided by the operation.

### kind (1)

`volvoxai.v1.ApiRuleKind`.

Protobuf default: `"API_RULE_KIND_UNSPECIFIED"`.

### fields (2)

`string` repeated.

Protobuf default: `[]`.

### description (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ApiField



### name (1)

`string`.

Protobuf default: `""`.

### number (2)

`uint32`.

Protobuf default: `0`.

### type (3)

`string`.

Protobuf default: `""`.

Scalar protobuf spelling, or a fully-qualified message/enum name.

### repeated (4)

`bool`.

Protobuf default: `false`.

### has_presence (5)

`bool`.

Protobuf default: `false`.

### oneof (6)

`string`.

Protobuf default: `""`.

### description (7)

`string`.

Protobuf default: `""`.

### required (8)

`bool`.

Protobuf default: `false`.

True only for explicitly documented semantic requirements. False is
not proof that omission is valid for every operation using this type.

### proto_default (9)

`string`.

Protobuf default: `""`.

Protobuf default (JSON spelling), distinct from an engine default.

### engine_default (10)

`string`.

Protobuf default: `""`.

Documented engine behavior when omitted/zero, not a codec substitution.

### handle_kind (11)

`string`.

Protobuf default: `""`.

### rules (12)

`volvoxai.v1.ApiRule` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ApiMessage



### name (1)

`string`.

Protobuf default: `""`.

### description (2)

`string`.

Protobuf default: `""`.

### fields (3)

`volvoxai.v1.ApiField` repeated.

Protobuf default: `[]`.

### rules (4)

`volvoxai.v1.ApiRule` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ApiEnumValue



### name (1)

`string`.

Protobuf default: `""`.

### number (2)

`int32`.

Protobuf default: `0`.

### description (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ApiEnum



### name (1)

`string`.

Protobuf default: `""`.

### description (2)

`string`.

Protobuf default: `""`.

### values (3)

`volvoxai.v1.ApiEnumValue` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ApiMethod



### service (1)

`string`.

Protobuf default: `""`.

### name (2)

`string`.

Protobuf default: `""`.

### request_type (3)

`string`.

Protobuf default: `""`.

### response_type (4)

`string`.

Protobuf default: `""`.

### description (5)

`string`.

Protobuf default: `""`.

### effect (6)

`volvoxai.v1.ApiEffect`.

Protobuf default: `"API_EFFECT_UNSPECIFIED"`.

### rules (7)

`volvoxai.v1.ApiRule` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ApiDescription



### methods (1)

`volvoxai.v1.ApiMethod` repeated.

Protobuf default: `[]`.

### messages (2)

`volvoxai.v1.ApiMessage` repeated.

Protobuf default: `[]`.

### enums (3)

`volvoxai.v1.ApiEnum` repeated.

Protobuf default: `[]`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### schema_sha256 (5)

`string`.

Protobuf default: `""`.

## volvoxai.v1.BackendList



### backends (1)

`string` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.MonotonicTime

Current CLOCK_MONOTONIC reading in nanoseconds. Submit deadlines use this
clock domain. The representation does not imply nanosecond clock precision.

### nanoseconds (1)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.DescribeStatusRequest



### status (1)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

## volvoxai.v1.StatusDescription



### status (1)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

### name (2)

`string`.

Protobuf default: `""`.

### description (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.RuntimeBudget

Queued, in-flight, and caller-retained results share these Runtime-wide
bounds. Admission precharges each route's declared worst-case output sum; a
validated success shrinks that charge to its owned snapshot bytes. The slot
and remaining bytes return on failure or final result release.

Every field is optional: an unset field takes the engine default. Defaults
are 64 scheduled requests, 64 MiB scheduled input bytes, zero batch delay,
64 unconsumed results, and 64 MiB unconsumed result bytes.

### max_scheduled_requests (1)

`uint64`.

Protobuf default: `"0"`.

Engine default: 64

### max_scheduled_input_bytes (2)

`uint64`.

Protobuf default: `"0"`.

Engine default: 67108864 bytes

### max_batch_delay_ns (3)

`uint64`.

Protobuf default: `"0"`.

Engine default: 0

Maximum coalescing delay for scheduled requests. Zero dispatches
immediately; deadlines may shorten a nonzero delay.
Rounded up to the scheduler's millisecond coalescing tick; at most
4294967295000000 ns. Representation and scheduling precision are distinct.

### max_unconsumed_results (4)

`uint64`.

Protobuf default: `"0"`.

Engine default: 64

### max_unconsumed_result_bytes (5)

`uint64`.

Protobuf default: `"0"`.

Engine default: 67108864 bytes

## volvoxai.v1.CreateRuntimeRequest



### debug (1)

`bool`.

Protobuf default: `false`.

### cpu_threads (2)

`int32`.

Protobuf default: `0`.

Engine default: 0 selects the engine default thread count.

### execution_mode (3)

`volvoxai.v1.ExecutionMode`.

Protobuf default: `"EXECUTION_MODE_DIRECT"`.

Engine default: EXECUTION_MODE_DIRECT

### budget (4)

`volvoxai.v1.RuntimeBudget`.

Protobuf default: `null`.

Engine default: All RuntimeBudget engine defaults.

Absence takes every engine default.

## volvoxai.v1.RuntimeHandle



### runtime_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BankResidency

Which slots of a declared weight bank to materialize.

Residency is part of the immutable loaded Model source. Every compiled
context owns a private engine and descriptor table, borrows the compiled
immutable weight blobs, and materializes the same selected rows in a
context-owned copy-on-write overlay. Slot ids are ascending, unique, and
index the bank's current declared extent, not its dimension's append
capacity. Route indices stay in that global slot space at execution, so a
model loaded with a subset still routes by the ids the exporter emitted.
Each present entry names a nonempty strict subset; omit a bank to make every
currently declared slot resident. Duplicate bank entries are invalid.

### bank (1)

`string`.

Protobuf default: `""`.

Weight tensor named by the graph document's "banks" table.

### slots (2)

`uint32` repeated.

Protobuf default: `[]`.

## volvoxai.v1.LoadModelRequest

graph_path names graph.json or a named *.graph.json document. The loader
requires its root discriminator to be exactly "volvox-graph/v1" and never
searches alternate filenames. weight_paths are retained in order. A bank
left out of bank_residency is fully resident.
Use exactly one of graph_path or package. weight_paths requires graph_path.

- `1`: `graph_path`, `package`
- `2`: `weight_paths`, `graph_path`

### runtime_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Runtime`.

### graph_path (2)

`string`.

Protobuf default: `""`.

### weight_paths (3)

`string` repeated.

Protobuf default: `[]`.

### bank_residency (4)

`volvoxai.v1.BankResidency` repeated.

Protobuf default: `[]`.

Engine default: All declared slots are resident.

### package (5)

`volvoxai.v1.ModelPackage`.

Protobuf default: `null`.

## volvoxai.v1.ModelPackage

Portable source for authoring/export -> inference. The loader snapshots all
bytes before publishing the Model; callers may reuse their buffers after
the call. Descendants retain the accepted snapshot after ReleaseModel.
graph_document uses volvox-graph/v1; shards are ordered SafeTensors files.
Total graph + shard bytes must be <= 64 MiB on every transport. A web host
may impose a tighter maxPackageBytes. No path resolution or fetch occurs.

### graph_document (1)

`bytes`; required.

Protobuf default: `""`.

### weight_shards (2)

`bytes` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ModelInfo



### model_id (1)

`int64`.

Protobuf default: `"0"`.

### inputs (2)

`volvoxai.v1.TensorSpec` repeated.

Protobuf default: `[]`.

### outputs (3)

`volvoxai.v1.TensorSpec` repeated.

Protobuf default: `[]`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.ModelHandle



### model_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.RevisionInfo



### graph_id (1)

`uint64`.

Protobuf default: `"0"`.

### graph_revision (2)

`uint64`.

Protobuf default: `"0"`.

### weight_id (3)

`uint64`.

Protobuf default: `"0"`.

### weight_revision (4)

`uint64`.

Protobuf default: `"0"`.

### adapter_id (5)

`uint64`.

Protobuf default: `"0"`.

### adapter_revision (6)

`uint64`.

Protobuf default: `"0"`.

### report (7)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.PublishAdapterRequest

Publishing another package under the same adapter_name creates the next
immutable revision of that adapter.

### model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Model`.

### adapter_name (2)

`string`.

Protobuf default: `""`.

### package_path (3)

`string`.

Protobuf default: `""`.

Empty publishes a metadata-only route revision, which is useful to
providers that own adapter storage.

### version_name (4)

`string`.

Protobuf default: `""`.

Empty uses the package/provider version.

## volvoxai.v1.AdapterRevision



### adapter_id (1)

`uint64`.

Protobuf default: `"0"`.

### adapter_revision (2)

`uint64`.

Protobuf default: `"0"`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BackendPolicy



### mode (1)

`volvoxai.v1.BackendPolicyMode`.

Protobuf default: `"BACKEND_POLICY_MODE_PREFER"`.

### backends (2)

`string` repeated.

Protobuf default: `[]`.

Engine default: The host's default ordered backend policy.

PREFER tries this ordered list. REQUIRE accepts exactly one entry. An
empty list selects the host's default ordered backend policy. A populated
list has at most 16 unique names. Each name is 1..63 ASCII characters,
begins with a lower-case letter, and then contains only lower-case letters,
digits, dot, underscore, or dash.

### operator_fallback (3)

`volvoxai.v1.OperatorFallback`.

Protobuf default: `"OPERATOR_FALLBACK_ALLOW"`.

## volvoxai.v1.CompileModelRequest



### model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Model`.

### policy (2)

`volvoxai.v1.BackendPolicy`.

Protobuf default: `null`.

## volvoxai.v1.CompiledModelHandle



### compiled_model_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### compile_time_ns (3)

`uint64`.

Protobuf default: `"0"`.

Host compilation duration measured with the monotonic clock.

### memory_bounds (4)

`volvoxai.v1.MemoryDomainAttestation`.

Protobuf default: `null`.

## volvoxai.v1.RunRequest

Inputs form one atomic, complete named batch. Every tensor is validated
before the engine or provider may commit any input mutation.

### compiled_model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `CompiledModel`.

### inputs (2)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

## volvoxai.v1.CreateExecutionContextRequest



### compiled_model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `CompiledModel`.

### decode_row_mode (2)

`volvoxai.v1.DecodeRowMode`.

Protobuf default: `"DECODE_ROW_MODE_DISABLED"`.

### require_incremental (3)

`bool`.

Protobuf default: `false`.

### decode_lanes (4)

`uint32`.

Protobuf default: `0`.

Engine default: 1

Dense decode slots owned by this context, never inferred from an input.
Omitted selects one lane. More than one requires [lanes, sequence, ...].

### decode_inputs (5)

`string` repeated.

Protobuf default: `[]`.

Engine default: All model inputs.

Roots refreshed when DecodeStep reuses the existing input values (empty
inputs). Omitted selects all model inputs. Name only decoder inputs when
encoder/memory inputs must remain invariant across steps.

## volvoxai.v1.ExecutionContextHandle



### context_id (1)

`int64`.

Protobuf default: `"0"`.

### inputs (2)

`volvoxai.v1.TensorSpec` repeated.

Protobuf default: `[]`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.GetInputAffineQuantizationRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### name (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ExecuteRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### inputs (2)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

## volvoxai.v1.TensorOutputSelection



### names (1)

`string` repeated.

Protobuf default: `[]`.

Empty explicitly selects no outputs. Names must be unique graph outputs.

## volvoxai.v1.TensorFeedback



### input_name (1)

`string`.

Protobuf default: `""`.

### output_name (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ExecuteTensorsRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### inputs (2)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### reuse_inputs (3)

`string` repeated.

Protobuf default: `[]`.

Preserve the current value and shape of these named inputs. No external
buffer is retained: changes to the original producer are not observed.

### feedback (4)

`volvoxai.v1.TensorFeedback` repeated.

Protobuf default: `[]`.

Bind inputs from the previous execution's internal graph outputs. Shapes
come from those outputs and must satisfy the new complete input contract.
Device values used as host-validated indices/routes must instead be
provided explicitly in inputs as host tensors.

### outputs (5)

`volvoxai.v1.TensorOutputSelection`.

Protobuf default: `null`.

Absent selects all graph outputs. Present selects only these snapshots.

## volvoxai.v1.ExecutePrefixRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### row_count (2)

`int32`.

Protobuf default: `0`.

### inputs (3)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

## volvoxai.v1.DecodePrefillRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### inputs (2)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### position (3)

`int32`; oneof `cursor`.

Protobuf default: `0`.

### lane_positions (4)

`volvoxai.v1.DecodeLanePositions`; oneof `cursor`.

Protobuf default: `null`.

## volvoxai.v1.DecodeLanePositions



### positions (1)

`int32` repeated.

Protobuf default: `[]`.

One final prompt position per declared lane; every position is >= 0.

## volvoxai.v1.DecodeLaneAction



### position (1)

`int32`; oneof `action`.

Protobuf default: `0`.

Must equal this lane's next active position.

### idle (2)

`bool`; oneof `action`.

Protobuf default: `false`.

True recomputes this lane's last row without advancing its length.

### parked (3)

`bool`; oneof `action`.

Protobuf default: `false`.

True skips this lane's reads and writes. Its active length is retained.

## volvoxai.v1.DecodeLaneActions



### lanes (1)

`volvoxai.v1.DecodeLaneAction` repeated.

Protobuf default: `[]`.

## volvoxai.v1.DecodeStepRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### position (2)

`int32`; oneof `cursor`.

Protobuf default: `0`.

### lane_actions (4)

`volvoxai.v1.DecodeLaneActions`; oneof `cursor`.

Protobuf default: `null`.

### dependency_update (5)

`volvoxai.v1.Empty`; oneof `cursor`.

Protobuf default: `null`.

Recompute the entire dependency closure of supplied inputs, using the
retained values of every other input. Empty inputs execute no nodes.
This does not advance the cursor. Requires a prefilled single-lane
AUTO context without paged KV; REQUIRED row contexts reject it.

### inputs (3)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

After one successful complete prefill, a step may replace an atomic
subset of values while preserving the prefilled semantic shape signature
exactly. Empty reuses every prefilled input.

## volvoxai.v1.DecodeContextState



### lanes (1)

`uint32`.

Protobuf default: `0`.

### prefilled (2)

`bool`.

Protobuf default: `false`.

### mode (3)

`volvoxai.v1.DecodeMode`.

Protobuf default: `"DECODE_MODE_UNSPECIFIED"`.

### active_lengths (4)

`uint32` repeated.

Protobuf default: `[]`.

Zero before prefill/reset; a successful step changes advancing lanes only.

### parked (5)

`bool` repeated.

Protobuf default: `[]`.

### cache_generation (6)

`uint64`.

Protobuf default: `"0"`.

Changes on a successful prefill and when the cache is reset/invalidated.

### report (7)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.DecodeGenerateRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### token_input (2)

`string`.

Protobuf default: `""`.

A prefilled, single-lane required-row context. The terminal ArgMax or
QArgMax output at the previous position becomes the next embedding input.
All three tensors are distinct I32[1,S] tensors; keep_input masks causal
self-attention. C writes a visible keep value for each generated position.

### keep_input (3)

`string`.

Protobuf default: `""`.

### token_output (4)

`string`.

Protobuf default: `""`.

### token_count (5)

`uint32`.

Protobuf default: `0`.

Number of positions to append, starting at the current active length.
The entire range is admitted before mutation. One owned result snapshots
the final outputs; WebGPU never reads intermediate tokens back to the CPU.

### cache_generation (6)

`uint64`.

Protobuf default: `"0"`.

Optional resume guard from GetDecodeState. Reset/prefill invalidates it.

## volvoxai.v1.ConfigureDecodeCacheRequest

Convert an existing prefill into a context-owned page cache. K/V tensors
are internal activations, read only by causal self-attention. C retains the
current tensor storage, moves the live prefixes and owns all later page
reservation, copy-on-write and retirement. Tensor shapes never become ragged.
ResetDecode or another DecodePrefill detaches the cache and its prefixes.

### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### tensors (2)

`string` repeated.

Protobuf default: `[]`.

### page_tokens (3)

`uint32`.

Protobuf default: `0`.

### max_pages (4)

`uint32`.

Protobuf default: `0`.

Omitted uses all complete physical pages in the existing tensor pool.

### policy (5)

`volvoxai.v1.DecodeCachePolicy`.

Protobuf default: `"DECODE_CACHE_POLICY_PAGED"`.

### clear_on_recycle (6)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.DecodeLaneRef



### context_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### lane (2)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.PublishDecodePrefixRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### key (2)

`string`.

Protobuf default: `""`.

Context-local identity. Include every input value affecting the prefix.

### lane (3)

`uint32`.

Protobuf default: `0`.

### tokens (4)

`uint32`.

Protobuf default: `0`.

Must end at a complete page. Published pages remain immutable through COW.

## volvoxai.v1.ReuseDecodePrefixRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### key (2)

`string`.

Protobuf default: `""`.

### lane (3)

`uint32`.

Protobuf default: `0`.

Must be empty, for example after ReleaseDecodeLane. C restores the saved
decoder-input/output prefix as well as shared K/V pages. Other lanes and
independently retained ExecutionResults remain unchanged.

## volvoxai.v1.EvictDecodePrefixesRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### free_pages (2)

`uint32`.

Protobuf default: `0`.

Reclaim unused prefixes in LRU order until this many pages are free.

## volvoxai.v1.DecodeCacheLane



### active_length (1)

`uint32`.

Protobuf default: `0`.

### generation (2)

`uint32`.

Protobuf default: `0`.

### pages (3)

`int32` repeated.

Protobuf default: `[]`.

Logical page -> physical page; -1 means unmapped.

## volvoxai.v1.DecodeCacheState



### configured (1)

`bool`.

Protobuf default: `false`.

### policy (2)

`volvoxai.v1.DecodeCachePolicy`.

Protobuf default: `"DECODE_CACHE_POLICY_PAGED"`.

### page_tokens (3)

`uint32`.

Protobuf default: `0`.

### lane_token_capacity (4)

`uint32`.

Protobuf default: `0`.

### max_pages (5)

`uint32`.

Protobuf default: `0`.

### lanes (6)

`volvoxai.v1.DecodeCacheLane` repeated.

Protobuf default: `[]`.

### page_generations (7)

`uint32` repeated.

Protobuf default: `[]`.

### tensors (8)

`string` repeated.

Protobuf default: `[]`.

### logical_bytes (9)

`uint64`.

Protobuf default: `"0"`.

Logical/resident counts describe page use, not GPU allocations. Pool
storage stays allocated until the context closes; snapshots are separate.

### resident_bytes (10)

`uint64`.

Protobuf default: `"0"`.

### reserved_bytes (11)

`uint64`.

Protobuf default: `"0"`.

### pool_bytes (12)

`uint64`.

Protobuf default: `"0"`.

### prefix_snapshot_bytes (13)

`uint64`.

Protobuf default: `"0"`.

### resident_pages (14)

`uint32`.

Protobuf default: `0`.

### reserved_pages (15)

`uint32`.

Protobuf default: `0`.

### free_pages (16)

`uint32`.

Protobuf default: `0`.

### shared_pages (17)

`uint32`.

Protobuf default: `0`.

### fragmentation_bytes (18)

`uint64`.

Protobuf default: `"0"`.

### high_water_pages (19)

`uint32`.

Protobuf default: `0`.

### prefix_hits (20)

`uint64`.

Protobuf default: `"0"`.

### prefix_misses (21)

`uint64`.

Protobuf default: `"0"`.

### copy_on_writes (22)

`uint64`.

Protobuf default: `"0"`.

### evictions (23)

`uint64`.

Protobuf default: `"0"`.

### allocation_failures (24)

`uint64`.

Protobuf default: `"0"`.

### cache_generation (25)

`uint64`.

Protobuf default: `"0"`.

### report (26)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.AdapterRevisionRef



### adapter_id (1)

`uint64`.

Protobuf default: `"0"`.

### adapter_revision (2)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.SelectAdapterRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### revision (2)

`volvoxai.v1.AdapterRevisionRef`.

Protobuf default: `null`.

Engine default: Select the immutable base-model adapter.

Absence explicitly selects the immutable base-model adapter. Presence
pins one exact nonzero revision published by this Model.

## volvoxai.v1.ExecutionResultHandle



### result_id (1)

`int64`.

Protobuf default: `"0"`.

### execution_id (2)

`uint64`.

Protobuf default: `"0"`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### state (4)

`volvoxai.v1.ResultState`.

Protobuf default: `"RESULT_STATE_UNSPECIFIED"`.

Execute/Run/decode submit exactly once. Never repeat execution to await a
pending result: use GetResult with this handle. The execution ID is stable.

### metrics (5)

`volvoxai.v1.ExecutionMetrics`.

Protobuf default: `null`.

## volvoxai.v1.ResultInfo



### result_id (1)

`int64`.

Protobuf default: `"0"`.

### execution_id (2)

`uint64`.

Protobuf default: `"0"`.

### outputs (3)

`volvoxai.v1.TensorInfo` repeated.

Protobuf default: `[]`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### state (5)

`volvoxai.v1.ResultState`.

Protobuf default: `"RESULT_STATE_UNSPECIFIED"`.

### metrics (6)

`volvoxai.v1.ExecutionMetrics`.

Protobuf default: `null`.

## volvoxai.v1.ReadOutputRequest

Setting `into` asks the runtime to copy the named snapshot into
caller-owned memory and echo the written range; the response then carries a
view rather than inline bytes. Leaving it absent returns inline bytes.
A pending result returns BUSY without writing the destination. A failed
result returns its terminal execution failure. ReleaseResult may retire a
pending handle; device work drains without writing retired host memory.

### result_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionResult`.

### name (2)

`string`.

Protobuf default: `""`.

### into (3)

`volvoxai.v1.BorrowedBuffer`.

Protobuf default: `null`.

- `7`: `into`

## volvoxai.v1.ReadOutputResponse



### tensor (1)

`volvoxai.v1.Tensor`.

Protobuf default: `null`.

### required_bytes (2)

`uint64`.

Protobuf default: `"0"`.

Bytes the named snapshot requires. On BUFFER_TOO_SMALL this is the
capacity the caller must supply; the tensor payload is then unset.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.SubmitOptions

Priority is clamped to [-1000, 1000]. Higher values run first; queued age
raises effective priority, and equal effective priorities use
earliest-deadline-first, then request id.

### priority (1)

`int32`.

Protobuf default: `0`.

### deadline_monotonic_ns (2)

`uint64`.

Protobuf default: `"0"`.

Absolute monotonic nanoseconds in the VxPlatformService clock domain.
Rounded up to the scheduler clock tick.
Zero disables the deadline. It is always an admission/EDF target; only
DROP_IF_LATE makes it a hard queued/completion cutoff.

### freshness (3)

`volvoxai.v1.RequestFreshness`.

Protobuf default: `"REQUEST_FRESHNESS_ALL"`.

### stream_key (4)

`uint64`.

Protobuf default: `"0"`.

Required and nonzero for LATEST. Numeric identity copied by value.

## volvoxai.v1.SubmitRequest

SCHEDULED validates normalized metadata, reserves the Runtime request/input
and result budgets, then copies host payload bytes. Caller descriptors,
names, and payloads may be released once Submit returns.

### compiled_model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `CompiledModel`.

### inputs (2)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### options (3)

`volvoxai.v1.SubmitOptions`.

Protobuf default: `null`.

## volvoxai.v1.RequestHandle



### request_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.RequestInfo



### request_id (1)

`int64`.

Protobuf default: `"0"`.

### state (2)

`volvoxai.v1.RequestState`.

Protobuf default: `"REQUEST_STATE_UNSPECIFIED"`.

### status (3)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

BUSY while queued or running; otherwise the terminal execution status.

### owned_input_bytes (4)

`uint64`.

Protobuf default: `"0"`.

### deadline_missed (5)

`bool`.

Protobuf default: `false`.

Set when accepted work crossed its target. ALL and LATEST still publish a
successful result; DROP_IF_LATE instead reports DEADLINE_EXCEEDED.

### report (6)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.PerTensorAffineQuantization

Exact affine values hydrated from the tensors named by the logical graph.
Scales are finite and positive. Per-axis scales and zero points have equal,
nonzero lengths, and axis is in the target tensor's normalized rank.

### scale (1)

`float`.

Protobuf default: `0`.

### zero_point (2)

`int32`.

Protobuf default: `0`.

## volvoxai.v1.PerAxisAffineQuantization



### axis (1)

`uint32`.

Protobuf default: `0`.

### scales (2)

`float` repeated.

Protobuf default: `[]`.

### zero_points (3)

`int32` repeated.

Protobuf default: `[]`.

## volvoxai.v1.AffineQuantizationParameters



### per_tensor (1)

`volvoxai.v1.PerTensorAffineQuantization`; oneof `parameters`.

Protobuf default: `null`.

### per_axis (2)

`volvoxai.v1.PerAxisAffineQuantization`; oneof `parameters`.

Protobuf default: `null`.

## volvoxai.v1.TensorAffineQuantization

tensor_name is nonempty and unique in GraphPlanningSource, and the set must
match the graph document's affine descriptors exactly. Request order is
immaterial. Numeric values are validated against the descriptor tensors
represented by PlanningWeight.

### tensor_name (1)

`string`.

Protobuf default: `""`.

### parameters (2)

`volvoxai.v1.AffineQuantizationParameters`.

Protobuf default: `null`.

## volvoxai.v1.PlanningWeight

Weight descriptors are separate because volvox-graph/v1 deliberately does
not repeat fixed weight dtype and shape metadata from SafeTensors. Names are
unique, contain no U+0000, and occupy 1..127 UTF-8 bytes. dtype is one of
F16, F32, I32, I8, or U8; F16 storage is normalized to logical F32. A shape
has at most eight axes, an empty shape denotes a scalar, and every present
extent is in [1, 2,147,483,647]. One tensor's packed storage span is at most
4,294,967,295 bytes. Request order is immaterial and the resulting plan uses
canonical tensor order.

### name (1)

`string`.

Protobuf default: `""`.

### dtype (2)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### shape (3)

`int64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphPlanningSource

Standalone authoring source. graph_document contains 1..67,108,864 exact
UTF-8 JSON bytes of one volvox-graph/v1 document. The sum of every
PlanningWeight packed storage span is at most 9,223,372,036,854,775,807
bytes. The request snapshots every field before return; no caller buffer or
object is retained.

### graph_document (1)

`bytes`; oneof `definition_source`.

Protobuf default: `""`.

### definition (4)

`volvoxai.v1.GraphDefinition`; oneof `definition_source`.

Protobuf default: `null`.

### weights (2)

`volvoxai.v1.PlanningWeight` repeated.

Protobuf default: `[]`.

### quantization (3)

`volvoxai.v1.TensorAffineQuantization` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphDefinition

Typed construction of the same logical graph accepted as graph_document.
C owns normalization, generated node IDs, validation and fingerprinting.
Missing multiple_of means one. An omitted node ID selects the first unused
node_N, reserving all explicit IDs before assigning any generated ID.

### dimensions (1)

`volvoxai.v1.GraphDimension` repeated.

Protobuf default: `[]`.

### inputs (2)

`volvoxai.v1.GraphTensorDefinition` repeated.

Protobuf default: `[]`.

### nodes (3)

`volvoxai.v1.GraphNodeDefinition` repeated.

Protobuf default: `[]`.

### outputs (4)

`string` repeated.

Protobuf default: `[]`.

### quantization (5)

`volvoxai.v1.GraphQuantizationReference` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphTensorDefinition



### name (1)

`string`.

Protobuf default: `""`.

### dtype (2)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### shape (3)

`volvoxai.v1.GraphAxis` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphInputBinding



### port (1)

`string`.

Protobuf default: `""`.

### tensor_name (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.GraphOutputDefinition



### port (1)

`string`.

Protobuf default: `""`.

### tensor (2)

`volvoxai.v1.GraphTensorDefinition`.

Protobuf default: `null`.

## volvoxai.v1.GraphParameter

name is the source parameter spelling in the operator registry. Semantic
plans return its normalized NodeParameter; these are deliberately distinct.

### name (1)

`string`.

Protobuf default: `""`.

### value (2)

`volvoxai.v1.NodeParameterValue`.

Protobuf default: `null`.

## volvoxai.v1.GraphNodeDefinition



### id (1)

`string`.

Protobuf default: `""`.

### operator_name (2)

`string`.

Protobuf default: `""`.

### inputs (3)

`volvoxai.v1.GraphInputBinding` repeated.

Protobuf default: `[]`.

### outputs (4)

`volvoxai.v1.GraphOutputDefinition` repeated.

Protobuf default: `[]`.

### parameters (5)

`volvoxai.v1.GraphParameter` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphQuantizationReference



### tensor_name (1)

`string`.

Protobuf default: `""`.

### scale_tensor (2)

`string`.

Protobuf default: `""`.

### zero_point_tensor (3)

`string`.

Protobuf default: `""`.

### axis (4)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.GraphOutputSelection



### names (1)

`string` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphNodeReplacement



### id (1)

`string`.

Protobuf default: `""`.

### node (2)

`volvoxai.v1.GraphNodeDefinition`.

Protobuf default: `null`.

## volvoxai.v1.GraphQuantizationReplacement



### references (1)

`volvoxai.v1.GraphQuantizationReference` repeated.

Protobuf default: `[]`.

### values (2)

`volvoxai.v1.TensorAffineQuantization` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphEdit



### set_dimension (1)

`volvoxai.v1.GraphDimension`; oneof `operation`.

Protobuf default: `null`.

### remove_dimension (2)

`string`; oneof `operation`.

Protobuf default: `""`.

### set_input (3)

`volvoxai.v1.GraphTensorDefinition`; oneof `operation`.

Protobuf default: `null`.

### remove_input (4)

`string`; oneof `operation`.

Protobuf default: `""`.

### set_weight (5)

`volvoxai.v1.PlanningWeight`; oneof `operation`.

Protobuf default: `null`.

### remove_weight (6)

`string`; oneof `operation`.

Protobuf default: `""`.

### add_node (7)

`volvoxai.v1.GraphNodeDefinition`; oneof `operation`.

Protobuf default: `null`.

### replace_node (8)

`volvoxai.v1.GraphNodeReplacement`; oneof `operation`.

Protobuf default: `null`.

### remove_node (9)

`string`; oneof `operation`.

Protobuf default: `""`.

### select_outputs (10)

`volvoxai.v1.GraphOutputSelection`; oneof `operation`.

Protobuf default: `null`.

### replace_quantization (11)

`volvoxai.v1.GraphQuantizationReplacement`; oneof `operation`.

Protobuf default: `null`.

## volvoxai.v1.EditGraphPlanRequest

One immutable topology transaction. Edits apply in order to a private C
draft; references may be temporarily broken. C validates the final graph
once and publishes a new standalone plan only on success. The source plan
and all existing readers remain unchanged on success or failure. Removing
an absent item is a no-op; replacing an unknown node is an error. An omitted
replacement ID keeps the selected ID. An empty edit list clones the source.

### graph_plan_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `GraphPlan`.

### edits (2)

`volvoxai.v1.GraphEdit` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ExportedGraphPlan



### source (1)

`volvoxai.v1.GraphPlanningSource`.

Protobuf default: `null`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.SerializedGraph



### data (1)

`bytes`.

Protobuf default: `""`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.CreateGraphPlanRequest

Exactly one source is required. A model source pins that exact immutable
graph, weight, and adapter revision. A standalone source is independent of
any Runtime and is suitable for offline authoring. Definition normalization
is a constructor invariant: an invalid source returns a failed report with
no handle or partial GraphPlan. A well-formed definition whose bounded shape
domain is unsupported still creates a readable plan with a typed refusal.

### model_id (1)

`int64`; oneof `source`.

Protobuf default: `"0"`.

Handle kind: `Model`.

### graph (2)

`volvoxai.v1.GraphPlanningSource`; oneof `source`.

Protobuf default: `null`.

## volvoxai.v1.GraphDimension

Names are nonempty, unique, and emitted in UTF-8 byte order. Bounds and
multiple_of are positive integers no greater than 9,007,199,254,740,991,
minimum <= maximum, and the closed interval contains at least one multiple
of multiple_of. Those multiples, rather than every integer in the interval,
are the dimension's legal values.

### name (1)

`string`.

Protobuf default: `""`.

### minimum (2)

`int64`.

Protobuf default: `"0"`.

### maximum (3)

`int64`.

Protobuf default: `"0"`.

### multiple_of (4)

`int64`.

Protobuf default: `"0"`.

## volvoxai.v1.GraphAxis

One declared logical axis. Fixed extents are positive. A symbolic axis has a
nonempty name that joins exactly one GraphDimension; bounds live only in
that canonical table.

### fixed_extent (1)

`int64`; oneof `extent`.

Protobuf default: `"0"`.

### dimension (2)

`string`; oneof `extent`.

Protobuf default: `""`.

## volvoxai.v1.GraphTensorRef

References repeat the canonical name beside the dense index so generated
consumers remain readable while still detecting a stale or mismatched join.

### tensor_index (1)

`uint32`.

Protobuf default: `0`.

### name (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.GraphNodeRef



### schedule_index (1)

`uint32`.

Protobuf default: `0`.

### id (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.GraphTensorProducer



### node (1)

`volvoxai.v1.GraphNodeRef`.

Protobuf default: `null`.

### port (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.GraphPlanOutputRef

Reverse reference into GraphPlan.outputs. Message presence distinguishes a
public output at index zero from a tensor that is not a public output.

### output_index (1)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.GraphTensor

A present storage_alias_root says the planner proved this tensor is a
byte-identical logical view of that canonical root over the complete
accepted domain. The root itself has no storage_alias_root, so alias chains
are already collapsed. A provider may share storage or materialize a copy;
no physical address or allocation requirement is implied. producer and
public_output are reverse indexes into GraphNode.outputs and
GraphPlan.outputs respectively; both directions must agree exactly.

### tensor_index (1)

`uint32`.

Protobuf default: `0`.

### name (2)

`string`.

Protobuf default: `""`.

### kind (3)

`volvoxai.v1.GraphTensorKind`.

Protobuf default: `"GRAPH_TENSOR_KIND_UNSPECIFIED"`.

### dtype (4)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### shape (5)

`volvoxai.v1.GraphAxis` repeated.

Protobuf default: `[]`.

### quantization (6)

`volvoxai.v1.AffineQuantizationParameters`.

Protobuf default: `null`.

### producer (7)

`volvoxai.v1.GraphTensorProducer`.

Protobuf default: `null`.

### canonical_birth_step (8)

`int32`.

Protobuf default: `0`.

INPUT and WEIGHT birth is -1; VALUE birth is its producer's schedule
index. WEIGHT last use is -1 because weight residency is described
separately. For every other tensor, last use is its greatest consuming
schedule index, -1 for an unused non-output INPUT, or node_count when the
tensor is a public output retained through the caller boundary.

### canonical_last_use_step (9)

`int32`.

Protobuf default: `0`.

### public_output (10)

`volvoxai.v1.GraphPlanOutputRef`.

Protobuf default: `null`.

### storage_alias_root (11)

`volvoxai.v1.GraphTensorRef`.

Protobuf default: `null`.

## volvoxai.v1.GraphPortBinding



### port (1)

`string`.

Protobuf default: `""`.

### tensor (2)

`volvoxai.v1.GraphTensorRef`.

Protobuf default: `null`.

## volvoxai.v1.IntegerParameterList



### values (1)

`int64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ShapeExpressionAxis

One source-level axis in an operator shape parameter. Unlike GraphAxis, a
literal is signed and may be zero because operators such as Reshape assign
meaning to 0 and -1 sentinels. A symbolic value joins one GraphDimension;
the operator registry and its shape function define which literals are
legal for each parameter.

### literal (1)

`int64`; oneof `value`.

Protobuf default: `"0"`.

### dimension (2)

`string`; oneof `value`.

Protobuf default: `""`.

## volvoxai.v1.ShapeParameterList



### values (1)

`volvoxai.v1.ShapeExpressionAxis` repeated.

Protobuf default: `[]`.

## volvoxai.v1.NodeParameterValue

Canonical typed source-declared operator parameter. The generated registry
selects the arm, rather than the lexical JSON representation: accepted
boolean/integer and scalar/list aliases therefore converge to one value.
number_value is the registry's exact F32 value. JSON objects are not a
supported parameter form.

### bool_value (1)

`bool`; oneof `value`.

Protobuf default: `false`.

### integer_value (2)

`int64`; oneof `value`.

Protobuf default: `"0"`.

### number_value (3)

`float`; oneof `value`.

Protobuf default: `0`.

### string_value (4)

`string`; oneof `value`.

Protobuf default: `""`.

### integer_list (5)

`volvoxai.v1.IntegerParameterList`; oneof `value`.

Protobuf default: `null`.

### shape_list (6)

`volvoxai.v1.ShapeParameterList`; oneof `value`.

Protobuf default: `null`.

## volvoxai.v1.NodeParameter

canonical_name is the registry json_name for an ordinary parameter and the
stable alias-group id for mutually exclusive source aliases. It therefore
describes semantic meaning and is not necessarily the member spelling that
appeared in the source JSON. A registry-authorized JSON null has the same
meaning as omission and is therefore absent from GraphNode.parameters.

### canonical_name (1)

`string`.

Protobuf default: `""`.

### value (2)

`volvoxai.v1.NodeParameterValue`.

Protobuf default: `null`.

## volvoxai.v1.GraphNode

Ports and parameters use canonical UTF-8 port/canonical_name order.
schedule_index is dense execution order; definition_index points back to
the source graph.

### definition_index (1)

`uint32`.

Protobuf default: `0`.

### schedule_index (2)

`uint32`.

Protobuf default: `0`.

### id (3)

`string`.

Protobuf default: `""`.

### operator_name (4)

`string`.

Protobuf default: `""`.

### shape_function_id (5)

`string`.

Protobuf default: `""`.

### inputs (6)

`volvoxai.v1.GraphPortBinding` repeated.

Protobuf default: `[]`.

### outputs (7)

`volvoxai.v1.GraphPortBinding` repeated.

Protobuf default: `[]`.

### parameters (8)

`volvoxai.v1.NodeParameter` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphWeightBank

One logical bank declaration. tensor is a rank-two-or-greater WEIGHT whose
first axis is the bank's current declared slot count. That fixed extent must
be a legal value of the joined GraphDimension, whose maximum is at most
2,147,483,647. The dimension's maximum is append capacity, not current
extent; adding slots within it may preserve graph topology but creates a new
planning snapshot. Valid global slot ids are [0, tensor.shape[0]). Residency
is deliberately absent: it is a concrete binding choice reported by
ResolvedWeightBank, not graph topology.

### tensor (1)

`volvoxai.v1.GraphTensorRef`.

Protobuf default: `null`.

### dimension (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ResidentSlotSubset

The partial form is a nonempty strict subset of the bank's current declared
slots. Slots are ascending, unique, in [0, GraphWeightBank.tensor.shape[0]),
and remain in the graph's global slot space.

### slots (1)

`uint32` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ResolvedWeightBank

Concrete residency for exactly one GraphWeightBank. tensor and dimension
must match that declaration. Exactly one residency form is present. Full
residency means every currently declared slot and is represented without
enumerating them. For a partial bank, resident tensor row i represents
global slot partial.slots[i].

### tensor (1)

`volvoxai.v1.GraphTensorRef`.

Protobuf default: `null`.

### dimension (2)

`string`.

Protobuf default: `""`.

### fully_resident (3)

`volvoxai.v1.Empty`; oneof `residency`.

Protobuf default: `null`.

### partial (4)

`volvoxai.v1.ResidentSlotSubset`; oneof `residency`.

Protobuf default: `null`.

## volvoxai.v1.ShapeDiagnostic



### code (1)

`volvoxai.v1.ShapeDiagnosticCode`.

Protobuf default: `"SHAPE_DIAGNOSTIC_CODE_UNSPECIFIED"`.

### message (2)

`string`.

Protobuf default: `""`.

### node (3)

`volvoxai.v1.GraphNodeRef`.

Protobuf default: `null`.

Both references are optional. Their absence denotes a graph-wide
diagnostic; when present they join this GraphPlan exactly.

### tensor (4)

`volvoxai.v1.GraphTensorRef`.

Protobuf default: `null`.

## volvoxai.v1.AffineDimensionRelation



### source (1)

`string`.

Protobuf default: `""`.

### offset (2)

`int64`.

Protobuf default: `"0"`.

## volvoxai.v1.ShapeDimensionRelation

The target is defined by exactly one constant, source symbol, or affine
source-plus-offset expression. A self witness sets symbol == target.

### target (1)

`string`.

Protobuf default: `""`.

### constant (2)

`int64`; oneof `relation`.

Protobuf default: `"0"`.

### symbol (3)

`string`; oneof `relation`.

Protobuf default: `""`.

### affine (4)

`volvoxai.v1.AffineDimensionRelation`; oneof `relation`.

Protobuf default: `null`.

## volvoxai.v1.ShapeDomainNodeProof



### node (1)

`volvoxai.v1.GraphNodeRef`.

Protobuf default: `null`.

### facts (2)

`volvoxai.v1.ShapeDomainFact` repeated.

Protobuf default: `[]`.

Nonempty, ascending, unique, and never UNSPECIFIED.

## volvoxai.v1.ShapeDomainProof



### proof_identity (1)

`string`.

Protobuf default: `""`.

Stable identity of this exact complete-domain proof, independent of the
process-local GraphPlan handle that exposes it. Within one plan_identity,
equal non-identity proof fields produce the same proof_identity.

### kind (2)

`volvoxai.v1.ShapeDomainProofKind`.

Protobuf default: `"SHAPE_DOMAIN_PROOF_KIND_UNSPECIFIED"`.

### relations (3)

`volvoxai.v1.ShapeDimensionRelation` repeated.

Protobuf default: `[]`.

Unique targets in GraphPlan dimension order.

### nodes (4)

`volvoxai.v1.ShapeDomainNodeProof` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ShapeDomainRefusal

A refusal is emitted only after definition normalization succeeded and the
bounded-domain proof reached a canonical semantic terminal.

### proof_identity (1)

`string`.

Protobuf default: `""`.

Within one plan_identity, equal diagnostic evidence produces the same
proof_identity.

### diagnostic (2)

`volvoxai.v1.ShapeDiagnostic`.

Protobuf default: `null`.

## volvoxai.v1.ShapeDomainAnalysis



### supported (1)

`volvoxai.v1.ShapeDomainProof`; oneof `outcome`.

Protobuf default: `null`.

### unsupported (2)

`volvoxai.v1.ShapeDomainRefusal`; oneof `outcome`.

Protobuf default: `null`.

## volvoxai.v1.IndependentBatchAxis

dimension joins exactly one GraphDimension. tensor_axis is the single
occurrence of that dimension shared by every public input and output.

### dimension (1)

`string`.

Protobuf default: `""`.

### tensor_axis (2)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.IndependentBatchContract

batch_axis is absent for a singleton graph whose independence proof is
exhaustive but has no public symbolic batch dimension.

### proof_identity (1)

`string`.

Protobuf default: `""`.

Within one plan_identity, equal non-identity contract fields produce the
same proof_identity.

### batch_axis (2)

`volvoxai.v1.IndependentBatchAxis`.

Protobuf default: `null`.

### covered_nodes (3)

`uint32`.

Protobuf default: `0`.

Equal to GraphPlan.nodes.size.

## volvoxai.v1.IndependentBatchRefusal



### proof_identity (1)

`string`.

Protobuf default: `""`.

Present because UNSUPPORTED means the proof ran to a canonical semantic
terminal. NOT_EVALUATED below deliberately has no batch proof identity.
Within one plan_identity, equal non-identity refusal fields produce the
same proof_identity.

### reason (2)

`volvoxai.v1.IndependentBatchRefusalReason`.

Protobuf default: `"INDEPENDENT_BATCH_REFUSAL_REASON_UNSPECIFIED"`.

### covered_nodes (3)

`uint32`.

Protobuf default: `0`.

Number of leading schedule nodes accepted before the refusal terminal.

### node (4)

`volvoxai.v1.GraphNodeRef`.

Protobuf default: `null`.

### tensor (5)

`volvoxai.v1.GraphTensorRef`.

Protobuf default: `null`.

### message (6)

`string`.

Protobuf default: `""`.

## volvoxai.v1.IndependentBatchNotEvaluated

A prerequisite failure is different from a proof that ran and refused the
graph. In particular, bounded-domain rejection makes batch proof
NOT_EVALUATED rather than UNSUPPORTED. The identity is nonempty and must
equal GraphPlan.shape_domain.unsupported.proof_identity.

### prerequisite_shape_proof_identity (1)

`string`.

Protobuf default: `""`.

## volvoxai.v1.IndependentBatchAnalysis



### supported (1)

`volvoxai.v1.IndependentBatchContract`; oneof `outcome`.

Protobuf default: `null`.

### unsupported (2)

`volvoxai.v1.IndependentBatchRefusal`; oneof `outcome`.

Protobuf default: `null`.

### not_evaluated (3)

`volvoxai.v1.IndependentBatchNotEvaluated`; oneof `outcome`.

Protobuf default: `null`.

## volvoxai.v1.GraphPlan

Canonical order is explicit: dimensions and parameter canonical_name values
use UTF-8 byte order, tensors and weight banks use tensor_index, nodes and
their proof rows use schedule_index, and outputs keep graph declaration
order. Consumers must reject duplicate names or indices and inconsistent
forward/reverse joins.

### graph_fingerprint (1)

`string`.

Protobuf default: `""`.

Versioned identity of the exact graph-document byte revision. Treat this
string as opaque: graph_fingerprint distinguishes source revisions, while
plan_identity below identifies their canonicalized declared plan.

### plan_identity (2)

`string`.

Protobuf default: `""`.

Stable identity of the complete canonicalized source-declared plan and its
analysis. Registry-declared aliases and nullable nulls converge; an
explicitly declared runtime-default value remains distinct from omission.
Domain and batch evidence carry narrower proof identities below. Bank
declarations participate; concrete residency does not. Two canonical
GraphPlan payloads with equal non-identity fields have the same identity;
this comparison ignores graph_fingerprint and every *_identity field.

### dimensions (3)

`volvoxai.v1.GraphDimension` repeated.

Protobuf default: `[]`.

### tensors (4)

`volvoxai.v1.GraphTensor` repeated.

Protobuf default: `[]`.

### nodes (5)

`volvoxai.v1.GraphNode` repeated.

Protobuf default: `[]`.

### outputs (6)

`volvoxai.v1.GraphTensorRef` repeated.

Protobuf default: `[]`.

### shape_domain (7)

`volvoxai.v1.ShapeDomainAnalysis`.

Protobuf default: `null`.

### independent_batch (8)

`volvoxai.v1.IndependentBatchAnalysis`.

Protobuf default: `null`.

### weight_banks (9)

`volvoxai.v1.GraphWeightBank` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GraphPlanHandle

Success carries a positive id and a complete plan. Failure carries neither
an id nor a partial plan; its report is the only populated result evidence.

### graph_plan_id (1)

`int64`.

Protobuf default: `"0"`.

### source_kind (2)

`volvoxai.v1.GraphPlanSourceKind`.

Protobuf default: `"GRAPH_PLAN_SOURCE_KIND_UNSPECIFIED"`.

### plan (3)

`volvoxai.v1.GraphPlan`.

Protobuf default: `null`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.GraphPlanInfo

A successful lookup carries the complete immutable plan. A stale, released,
or invalid id carries no plan and is distinguished by report.status.

### graph_plan_id (1)

`int64`.

Protobuf default: `"0"`.

### source_kind (2)

`volvoxai.v1.GraphPlanSourceKind`.

Protobuf default: `"GRAPH_PLAN_SOURCE_KIND_UNSPECIFIED"`.

### plan (3)

`volvoxai.v1.GraphPlan`.

Protobuf default: `null`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.ConcreteInputShape



### name (1)

`string`.

Protobuf default: `""`.

### dtype (2)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### shape (3)

`int64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ExactShapeBinding



### inputs (1)

`volvoxai.v1.ConcreteInputShape` repeated.

Protobuf default: `[]`.

Request order is immaterial; every public input appears exactly once.

## volvoxai.v1.CallIntent

Absence of CallIntent in ResolveGraphPlan means FORWARD. When present, kind
must not be UNSPECIFIED. changed_inputs is allowed only for
FIXED_INPUT_DEPENDENCY and is canonicalized by tensor name.

### kind (1)

`volvoxai.v1.CallIntentKind`.

Protobuf default: `"CALL_INTENT_KIND_UNSPECIFIED"`.

### changed_inputs (2)

`string` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ResolveGraphPlanRequest



### graph_plan_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `GraphPlan`.

### minimum (2)

`volvoxai.v1.Empty`; oneof `binding`.

Protobuf default: `null`.

Selects the smallest legal value of every symbolic public-input axis,
preserving repeated-symbol equality.

### exact (3)

`volvoxai.v1.ExactShapeBinding`; oneof `binding`.

Protobuf default: `null`.

Supplies every public input exactly once. Dtype and rank participate in
validation as well as in the resulting canonical signature.

### call (4)

`volvoxai.v1.CallIntent`.

Protobuf default: `null`.

### bank_residency (5)

`volvoxai.v1.BankResidency` repeated.

Protobuf default: `[]`.

Allowed only for a standalone plan. A model-derived plan has immutable
residency selected by LoadModel and rejects a competing override. Request
order is immaterial; entries are matched by bank name.

## volvoxai.v1.ResolvedSymbol



### name (1)

`string`.

Protobuf default: `""`.

### value (2)

`int64`.

Protobuf default: `"0"`.

## volvoxai.v1.ResolvedGraphTensor

kind and dtype exactly match the same-index GraphTensor. shape is its exact
concrete specialization. A partial weight bank is the one exception to a
fixed logical extent: axis 0 equals its partial slot count, and per-axis
quantization on axis 0 is selected in that same slot order. Every other
fixed axis and quantization value is unchanged. element_count and byte_size
are checked derivations of shape and dtype, not independent estimates.

### tensor_index (1)

`uint32`.

Protobuf default: `0`.

### name (2)

`string`.

Protobuf default: `""`.

### kind (3)

`volvoxai.v1.GraphTensorKind`.

Protobuf default: `"GRAPH_TENSOR_KIND_UNSPECIFIED"`.

### dtype (4)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### shape (5)

`int64` repeated.

Protobuf default: `[]`.

### element_count (6)

`uint64`.

Protobuf default: `"0"`.

### byte_size (7)

`uint64`.

Protobuf default: `"0"`.

### quantization (8)

`volvoxai.v1.AffineQuantizationParameters`.

Protobuf default: `null`.

### storage_alias_root (9)

`volvoxai.v1.GraphTensorRef`.

Protobuf default: `null`.

## volvoxai.v1.ExecutionSlice

ALL carries every graph node in schedule order. EXPLICIT carries the exact
fixed-input dependency closure in schedule order and may therefore be empty
or equal to the full schedule. cross_call_live_tensors are in canonical
tensor order. These are semantic liveness constraints, not a provider
allocation plan.

### kind (1)

`volvoxai.v1.CallIntentKind`.

Protobuf default: `"CALL_INTENT_KIND_UNSPECIFIED"`.

### selection (2)

`volvoxai.v1.NodeSelection`.

Protobuf default: `"NODE_SELECTION_UNSPECIFIED"`.

### changed_inputs (3)

`volvoxai.v1.GraphTensorRef` repeated.

Protobuf default: `[]`.

### selected_nodes (4)

`volvoxai.v1.GraphNodeRef` repeated.

Protobuf default: `[]`.

### cross_call_live_tensors (5)

`volvoxai.v1.GraphTensorRef` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ResolvedGraphPlan

symbols contains exactly the dimensions bound during concrete
specialization, as a subsequence of GraphPlan dimension order. Every
symbolic GraphTensor axis is represented; an otherwise-unused declaration
may remain absent. Tensors use tensor_index, outputs retain graph declaration
order, and weight banks use their tensor order. A successful resolution
carries every field below. Failure carries report only; its lineage
identifies the requested GraphPlan when that handle was valid.

### graph_plan_id (1)

`int64`.

Protobuf default: `"0"`.

### graph_fingerprint (2)

`string`.

Protobuf default: `""`.

### plan_identity (3)

`string`.

Protobuf default: `""`.

### source_kind (4)

`volvoxai.v1.GraphPlanSourceKind`.

Protobuf default: `"GRAPH_PLAN_SOURCE_KIND_UNSPECIFIED"`.

### signature (5)

`string`.

Protobuf default: `""`.

Opaque canonical binding identity scoped to plan_identity. Caches and
joins use (plan_identity, signature); signature alone is not a graph id.

### signature_digest (6)

`fixed64`.

Protobuf default: `"0"`.

FNV-1a-64 of exactly the UTF-8 bytes carried by signature, with offset
basis 14695981039346656037 and prime 1099511628211. No C terminator or
protobuf framing participates, so every language can reproduce it. This
is a compact integrity check, not a globally unique identity.

### symbols (7)

`volvoxai.v1.ResolvedSymbol` repeated.

Protobuf default: `[]`.

### tensors (8)

`volvoxai.v1.ResolvedGraphTensor` repeated.

Protobuf default: `[]`.

### outputs (9)

`volvoxai.v1.GraphTensorRef` repeated.

Protobuf default: `[]`.

### weight_banks (10)

`volvoxai.v1.ResolvedWeightBank` repeated.

Protobuf default: `[]`.

One entry per GraphPlan.weight_banks declaration, in the same tensor order.

### logical_activation_bytes (11)

`uint64`.

Protobuf default: `"0"`.

Sum of every non-WEIGHT logical tensor's byte_size. Exact aliases are
counted independently; this is deliberately not a physical arena size.

### weight_bytes (12)

`uint64`.

Protobuf default: `"0"`.

Sum of effective resident WEIGHT bytes after bank residency is applied;
equivalently, the sum of byte_size for every resolved WEIGHT tensor.

### execution_slice (13)

`volvoxai.v1.ExecutionSlice`.

Protobuf default: `null`.

### report (14)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.SafetensorsPathSource



### path (1)

`string`.

Protobuf default: `""`.

Native-host path to a complete SafeTensors file.

## volvoxai.v1.SafetensorsInlineHeaderSource



### header_prefix (1)

`bytes`.

Protobuf default: `""`.

Complete eight-byte little-endian length prefix followed by exactly the
declared JSON header bytes.

### file_size (2)

`uint64`.

Protobuf default: `"0"`.

Size of the complete logical file: prefix, JSON header, and tensor data.

## volvoxai.v1.SafetensorsHeaderViewSource



### header_prefix (1)

`volvoxai.v1.BorrowedBuffer`.

Protobuf default: `null`.

On a native host, only HOST, NATIVE_HEAP, and MAPPED_FILE are accepted.

### file_size (2)

`uint64`.

Protobuf default: `"0"`.

Size of the complete logical file: prefix, JSON header, and tensor data.

## volvoxai.v1.InspectSafetensorsRequest

Exactly one self-contained source is required. Its complete header prefix is
at most 67,108,864 bytes. path is native-host only; inline_header is portable
across every transport; header_view is native in-process only. WASM and
REMOTE transports reject path and header_view without dereferencing them.

### path (1)

`volvoxai.v1.SafetensorsPathSource`; oneof `source`.

Protobuf default: `null`.

### inline_header (2)

`volvoxai.v1.SafetensorsInlineHeaderSource`; oneof `source`.

Protobuf default: `null`.

### header_view (3)

`volvoxai.v1.SafetensorsHeaderViewSource`; oneof `source`.

Protobuf default: `null`.

## volvoxai.v1.SafetensorsMetadataEntry



### key (1)

`string`.

Protobuf default: `""`.

### value (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.SafetensorsDiagnosticEntry

An optional section-relative coordinate: a root member, metadata entry,
tensor declaration, or coverage tensor. Message presence distinguishes the
first entry from a diagnostic that cannot identify one entry.

### index (1)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.SafetensorsDiagnostic

Present only when a supplied header reached the parser and was refused.
entry is present only when the parser identifies one section member.
byte_offset is absolute from the logical file's first byte; zero is also the
meaningful coordinate for prefix/file-wide failures.

### code (1)

`volvoxai.v1.SafetensorsDiagnosticCode`.

Protobuf default: `"SAFETENSORS_DIAGNOSTIC_CODE_UNSPECIFIED"`.

### section (2)

`volvoxai.v1.SafetensorsDiagnosticSection`.

Protobuf default: `"SAFETENSORS_DIAGNOSTIC_SECTION_UNSPECIFIED"`.

### entry (3)

`volvoxai.v1.SafetensorsDiagnosticEntry`.

Protobuf default: `null`.

### byte_offset (4)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.SafetensorsTensorInfo



### name (1)

`string`.

Protobuf default: `""`.

### dtype (2)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### shape (3)

`int64` repeated.

Protobuf default: `[]`.

### file_offset (4)

`uint64`.

Protobuf default: `"0"`.

Absolute byte offset from the beginning of the complete logical file.

### byte_size (5)

`uint64`.

Protobuf default: `"0"`.

### declaration_index (6)

`uint32`.

Protobuf default: `0`.

Zero-based position of this tensor in the root JSON object; a preceding
__metadata__ member therefore participates in the index.

## volvoxai.v1.SafetensorsInfo

Metadata entries retain their order inside __metadata__. Tensors retain
their root-object declaration order. The parser validates unique member
names and exact, gap-free coverage of the complete tensor data region. A
successful result has no diagnostic. A parser refusal carries diagnostic
and report only; failures that occur before a parser terminal carry report
only.

### header_json_bytes (1)

`uint64`.

Protobuf default: `"0"`.

JSON byte count encoded by the file's leading eight-byte length prefix.

### data_region_offset (2)

`uint64`.

Protobuf default: `"0"`.

Absolute file offset of the first tensor-data byte. A valid response has
data_region_offset == 8 + header_json_bytes.

### data_region_bytes (3)

`uint64`.

Protobuf default: `"0"`.

A valid response has file_bytes == data_region_offset + data_region_bytes.

### file_bytes (4)

`uint64`.

Protobuf default: `"0"`.

### has_metadata (5)

`bool`.

Protobuf default: `false`.

### metadata (6)

`volvoxai.v1.SafetensorsMetadataEntry` repeated.

Protobuf default: `[]`.

### tensors (7)

`volvoxai.v1.SafetensorsTensorInfo` repeated.

Protobuf default: `[]`.

### diagnostic (8)

`volvoxai.v1.SafetensorsDiagnostic`.

Protobuf default: `null`.

### report (9)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.ReadSafetensorsRequest

Storage operations are immutable byte transactions and acquire no runtime or
GPU. They retain the complete SafeTensors dtype vocabulary, empty tensors,
arbitrary rank and UTF-8 names/metadata. Runtime graph restrictions apply only
when those tensors are used to construct a Model.

### source (1)

`bytes`.

Protobuf default: `""`.

### names (2)

`string` repeated.

Protobuf default: `[]`.

Empty selects all tensors in file order. Otherwise order follows names;
duplicates and missing names fail the whole call.

### normalize_f16 (3)

`bool`.

Protobuf default: `false`.

Convert F16 payloads to F32 in C; other dtypes remain byte-exact.

## volvoxai.v1.SafetensorsMetadata



### entries (1)

`volvoxai.v1.SafetensorsMetadataEntry` repeated.

Protobuf default: `[]`.

## volvoxai.v1.SafetensorsContent



### tensors (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### metadata (2)

`volvoxai.v1.SafetensorsMetadata`.

Protobuf default: `null`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.SafetensorsEdit



### set_tensor (1)

`volvoxai.v1.Tensor`; oneof `operation`.

Protobuf default: `null`.

### remove_tensor (2)

`string`; oneof `operation`.

Protobuf default: `""`.

## volvoxai.v1.WriteSafetensorsRequest



### source (1)

`bytes`.

Protobuf default: `""`.

Empty starts a new file. All source validation precedes editing.

### edits (2)

`volvoxai.v1.SafetensorsEdit` repeated.

Protobuf default: `[]`.

Apply in order. Set adds or replaces exact descriptor+bytes (an omitted
Tensor.payload allocates zero-filled storage); removal of
an absent name is a no-op. No partial artifact escapes a failed call.

### metadata (3)

`volvoxai.v1.SafetensorsMetadata`; oneof `metadata_update`.

Protobuf default: `null`.

### remove_metadata (4)

`volvoxai.v1.Empty`; oneof `metadata_update`.

Protobuf default: `null`.

## volvoxai.v1.SafetensorsArtifact



### data (1)

`bytes`.

Protobuf default: `""`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.NormalInitializer



### mean (1)

`double`.

Protobuf default: `0`.

### stddev (2)

`double`.

Protobuf default: `0`.

## volvoxai.v1.XavierInitializer



### gain (1)

`double`.

Protobuf default: `0`.

## volvoxai.v1.TensorInitializer

F32 initializers use the former deterministic Mulberry32 stream and
Box-Muller normal transform. Omitted distribution is zeros. Numeric seed
defaults to zero; text seeds use FNV-1a over UTF-16 code units, including
surrogate pairs. Missing mean/stddev/gain are 0/0.02/1 respectively.

### zeros (1)

`volvoxai.v1.Empty`; oneof `distribution`.

Protobuf default: `null`.

### ones (2)

`volvoxai.v1.Empty`; oneof `distribution`.

Protobuf default: `null`.

### normal (3)

`volvoxai.v1.NormalInitializer`; oneof `distribution`.

Protobuf default: `null`.

### xavier_uniform (4)

`volvoxai.v1.XavierInitializer`; oneof `distribution`.

Protobuf default: `null`.

### xavier_normal (5)

`volvoxai.v1.XavierInitializer`; oneof `distribution`.

Protobuf default: `null`.

### seed (6)

`uint32`; oneof `seed_source`.

Protobuf default: `0`.

### seed_text (7)

`string`; oneof `seed_source`.

Protobuf default: `""`.

## volvoxai.v1.InitializeTensorRequest



### tensor (1)

`volvoxai.v1.PlanningWeight`.

Protobuf default: `null`.

Nonempty positive concrete shape and F32 dtype are required.

### initializer (2)

`volvoxai.v1.TensorInitializer`.

Protobuf default: `null`.

## volvoxai.v1.InitializedTensor



### tensor (1)

`volvoxai.v1.Tensor`.

Protobuf default: `null`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BuildLoraLinearRequest

Append an explicit base + (input @ A @ B) * scale branch to an immutable
graph. No model or GPU is acquired. The original plan stays unchanged.
C validates the complete successor and returns its initialized A/B/scale
weights. Serialize them with WriteSafetensors and load the resulting model
for ordinary training; trainable_names selects A and B only.

### graph_plan_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `GraphPlan`.

### input_name (2)

`string`.

Protobuf default: `""`.

### base_weight_name (3)

`string`.

Protobuf default: `""`.

### bias_name (4)

`string`.

Protobuf default: `""`.

### rank (5)

`uint32`.

Protobuf default: `0`.

### alpha (6)

`double`; oneof `scaling`.

Protobuf default: `0`.

### scale (7)

`double`; oneof `scaling`.

Protobuf default: `0`.

### transpose_base_weight (8)

`bool`.

Protobuf default: `false`.

False means base weight [input, output]; true means [output, input].

### base_operator (9)

`string`.

Protobuf default: `""`.

Empty selects Linear; Linear and MatMul are the supported base operators.

### name (10)

`string`.

Protobuf default: `""`.

Empty derives lora_linear_<source node count>. All derived tensor/node
names must be unused. The name is a prefix, never a backend resource ID.

### a_initializer (11)

`volvoxai.v1.TensorInitializer`.

Protobuf default: `null`.

### b_initializer (12)

`volvoxai.v1.TensorInitializer`.

Protobuf default: `null`.

### select_output (13)

`bool`.

Protobuf default: `false`.

Replace public outputs with the new branch output when true.

## volvoxai.v1.LoraLinearPlan



### graph_plan_id (1)

`int64`.

Protobuf default: `"0"`.

### plan (2)

`volvoxai.v1.GraphPlan`.

Protobuf default: `null`.

### initialized_weights (3)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### trainable_names (4)

`string` repeated.

Protobuf default: `[]`.

### output_name (5)

`string`.

Protobuf default: `""`.

### scale (6)

`float`.

Protobuf default: `0`.

### report (7)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BuildRoutedAdapterRequest

Append routed down projection -> GELU -> up projection -> optional Dropout
-> residual Add. Existing weights/routes are named in the source plan.
The result is an ordinary graph, with the same C forward/backward execution.

### graph_plan_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `GraphPlan`.

### input_name (2)

`string`.

Protobuf default: `""`.

### down_weight_name (3)

`string`.

Protobuf default: `""`.

[experts, input, hidden], F32

### up_weight_name (4)

`string`.

Protobuf default: `""`.

[experts, hidden, input], F32

### route_indices_name (5)

`string`.

Protobuf default: `""`.

### route_weights_name (6)

`string`.

Protobuf default: `""`.

### down_bias_name (7)

`string`.

Protobuf default: `""`.

### up_bias_name (8)

`string`.

Protobuf default: `""`.

### dropout (9)

`double`.

Protobuf default: `0`.

finite [0,1); zero omits the node

### seed (10)

`uint32`.

Protobuf default: `0`.

### name (11)

`string`.

Protobuf default: `""`.

default routed_adapter_<source node count>

### select_output (12)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.RoutedAdapterPlan



### graph_plan_id (1)

`int64`.

Protobuf default: `"0"`.

### plan (2)

`volvoxai.v1.GraphPlan`.

Protobuf default: `null`.

### output_name (3)

`string`.

Protobuf default: `""`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.PreservedWeightScales

Pure matrix conversion for persistent F32 training masters and W8 inference
snapshots. This never mutates a Trainer or publishes a Model revision.
Keep masters in F32 across updates; convert only when exporting/publishing.

### scales (1)

`float` repeated.

Protobuf default: `[]`.

## volvoxai.v1.QuantizeWeightRequest



### weight (1)

`volvoxai.v1.Tensor`.

Protobuf default: `null`.

rank-two F32, positive extents, host payload

### transpose_source (2)

`bool`.

Protobuf default: `false`.

### recompute (3)

`volvoxai.v1.Empty`; oneof `scale_policy`.

Protobuf default: `null`.

default: max(abs(row))/127, one for an all-zero row

### preserve (4)

`volvoxai.v1.PreservedWeightScales`; oneof `scale_policy`.

Protobuf default: `null`.

## volvoxai.v1.QuantizedWeight



### weight (1)

`volvoxai.v1.Tensor`.

Protobuf default: `null`.

canonical [output, input] I8, symmetric [-127,127]

### scales (2)

`float` repeated.

Protobuf default: `[]`.

### saturation_count (3)

`uint64`.

Protobuf default: `"0"`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.DequantizeWeightRequest



### weight (1)

`volvoxai.v1.Tensor`.

Protobuf default: `null`.

rank-two I8, canonical [output, input]

### scales (2)

`float` repeated.

Protobuf default: `[]`.

one finite positive scale per output row

### transpose_destination (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.TrainerQuantizedWeightBinding

Atomic inference-weight snapshot from persistent F32 training masters.
Template shards retain all unbound tensors and metadata. Bound destinations
must already exist: I8 [output,input], F32 scale [output], I8 zero [output].
Zero points must be symmetric zero. Each master and each destination name
occurs once across the transaction. The transpose and all math execute in C.

### master_name (1)

`string`.

Protobuf default: `""`.

### weight_name (2)

`string`.

Protobuf default: `""`.

### scale_name (3)

`string`.

Protobuf default: `""`.

### zero_point_name (4)

`string`.

Protobuf default: `""`.

### transpose_master (5)

`bool`.

Protobuf default: `false`.

### preserve_scales (6)

`bool`.

Protobuf default: `false`.

false recomputes max-abs/127; all-zero rows use 1

## volvoxai.v1.ExportQuantizedTrainerWeightsRequest



### trainer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trainer`.

### template_shards (2)

`bytes` repeated.

Protobuf default: `[]`.

### bindings (3)

`volvoxai.v1.TrainerQuantizedWeightBinding` repeated.

Protobuf default: `[]`.

## volvoxai.v1.QuantizedTrainerWeights



### shards (1)

`bytes` repeated.

Protobuf default: `[]`.

### saturation_count (2)

`uint64`.

Protobuf default: `"0"`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.CreateTrainerRequest

A Trainer retains the exact Model revision current at creation. An empty
backend selects the host's default training backend; any named backend is an
exact requirement and is never retried on another backend.

### model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Model`.

### backend (2)

`string`.

Protobuf default: `""`.

### rng_seed (3)

`uint64`.

Protobuf default: `"0"`.

### checkpoint (4)

`volvoxai.v1.TrainerCheckpoint`.

Protobuf default: `null`.

Optional private starting state. The checkpoint must match this Model's
exact logical graph and weight descriptors. It supplies its own RNG seed.
Creation validates everything before issuing a Trainer ID. CommitTrainer
publishes restored weights; rollback restores this starting checkpoint
until the first commit. The Model itself is unchanged by restoration.

### shape_options (5)

`volvoxai.v1.TrainerShapeOptions`.

Protobuf default: `null`.

## volvoxai.v1.TrainerShapeOptions

Immutable per-Trainer policy. Limits are positive; omitted fields use the
defaults below. Oversized cache entries execute without being retained.
Capacity covers C activation storage; parameters, gradients, optimizer
slots and backend working memory are reported separately.

### plan_cache_entries (1)

`uint32`.

Protobuf default: `0`.

8; LRU

### plan_cache_metadata_bytes (2)

`uint64`.

Protobuf default: `"0"`.

1 MiB of live C plan metadata

### max_activation_capacity_bytes (3)

`uint64`.

Protobuf default: `"0"`.

512 MiB

### capacity_growth_factor (4)

`double`.

Protobuf default: `0`.

2; finite and greater than 1

## volvoxai.v1.TrainerHandle



### trainer_id (1)

`int64`.

Protobuf default: `"0"`.

### inputs (2)

`volvoxai.v1.TensorSpec` repeated.

Protobuf default: `[]`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.CrossEntropyLoss



### name (1)

`string`.

Protobuf default: `""`.

### logits_name (2)

`string`.

Protobuf default: `""`.

### targets (3)

`volvoxai.v1.Tensor`.

Protobuf default: `null`.

Dense I32 targets; inline, retained, or transient local storage.

### ignore_index (4)

`int32`.

Protobuf default: `0`.

### row_index (5)

`int32`.

Protobuf default: `0`.

-1 consumes the complete logits tensor. A non-negative value selects a
row-aware loss when the graph supports that execution shape.

### weight (6)

`float`.

Protobuf default: `0`.

### normalizer (7)

`float`.

Protobuf default: `0`.

Zero selects the active-label count for a one-microbatch update.
Accumulation windows larger than one require an explicit positive
denominator.

## volvoxai.v1.ReadTrainerParametersRequest



### trainer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trainer`.

### names (2)

`string` repeated.

Protobuf default: `[]`.

### mode (3)

`volvoxai.v1.ParameterExportMode`.

Protobuf default: `"PARAMETER_EXPORT_MODE_SNAPSHOT"`.

## volvoxai.v1.TrainerOptimizerOptions



### kind (1)

`volvoxai.v1.TrainingOptimizerKind`.

Protobuf default: `"TRAINING_OPTIMIZER_KIND_ADAMW"`.

Omitted fields keep the Trainer's last successful configuration. A new
Trainer starts with AdamW, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
zero weight decay and no clipping. Changing kind starts from those
defaults before applying the supplied fields. A rejected step changes none.

### learning_rate (2)

`float`.

Protobuf default: `0`.

### beta1 (3)

`float`.

Protobuf default: `0`.

### beta2 (4)

`float`.

Protobuf default: `0`.

### epsilon (5)

`float`.

Protobuf default: `0`.

### weight_decay (6)

`float`.

Protobuf default: `0`.

### max_gradient_norm (7)

`float`.

Protobuf default: `0`.

## volvoxai.v1.TrainStepRequest

A step changes only Trainer-owned working weights, gradients, optimizer
slots, accumulation, and RNG state. It never publishes a Model revision.
One step may be pending per Trainer. While pending, TrainStep, commit,
rollback and export return BUSY. ReleaseTrainer may discard pending work.
Poll GetTrainStep until READY or FAILED before reading metrics or weights.

### trainer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trainer`.

### inputs (2)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

One atomic complete shape-bearing batch. No Trainer input state exists
outside this request.

### losses (3)

`volvoxai.v1.CrossEntropyLoss` repeated.

Protobuf default: `[]`.

### trainable_names (4)

`string` repeated.

Protobuf default: `[]`.

### optimizer (5)

`volvoxai.v1.TrainerOptimizerOptions`.

Protobuf default: `null`.

### accumulation_steps (6)

`uint32`.

Protobuf default: `0`.

### flush_accumulation (7)

`bool`.

Protobuf default: `false`.

### reset_accumulation (8)

`bool`.

Protobuf default: `false`.

### outputs (9)

`volvoxai.v1.TensorOutputSelection`.

Protobuf default: `null`.

Native CPU/GPU only. Capture selected forward outputs before backward or
optimizer mutation. Absent or empty selects none. WebGPU rejects a
nonempty selection before submission; no hidden CPU fallback.

## volvoxai.v1.TrainingMetric



### name (1)

`string`.

Protobuf default: `""`.

### loss (2)

`float`.

Protobuf default: `0`.

### correct (3)

`int32`.

Protobuf default: `0`.

### examples (4)

`int32`.

Protobuf default: `0`.

### normalizer (5)

`float`.

Protobuf default: `0`.

## volvoxai.v1.TrainStepResult



### microbatch_id (1)

`uint64`.

Protobuf default: `"0"`.

### optimizer_step (2)

`uint64`.

Protobuf default: `"0"`.

### accumulated_microbatches (3)

`uint32`.

Protobuf default: `0`.

### update_applied (4)

`bool`.

Protobuf default: `false`.

False for an unfinished accumulation window; the Model is unchanged in
either case.

### loss (5)

`float`.

Protobuf default: `0`.

### metrics (6)

`volvoxai.v1.TrainingMetric` repeated.

Protobuf default: `[]`.

### backend (7)

`string`.

Protobuf default: `""`.

### report (8)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### state (9)

`volvoxai.v1.ResultState`.

Protobuf default: `"RESULT_STATE_UNSPECIFIED"`.

### outputs (10)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

Owned snapshots are published only by the submitting TrainStep response.
GetTrainStep returns metrics/state only; it never mints duplicate handles.

## volvoxai.v1.TrainStepRef

Queries the most recently accepted step. A later accepted TrainStep retires
the previous result; querying a different microbatch_id returns NOT_FOUND.
Repeated queries are nonblocking and never submit the step again.

### trainer_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Trainer`.

### microbatch_id (2)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.ExportTrainerWeightsRequest

Export copies the current private Trainer weight shards without publishing
them. Empty output_paths returns SafeTensors bytes in package shard order.
Native callers may instead supply exactly one destination per loaded shard;
then only the report is returned. Export rejects unfinished accumulation.

### trainer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trainer`.

### output_paths (2)

`string` repeated.

Protobuf default: `[]`.

## volvoxai.v1.TrainerWeights



### shards (1)

`bytes` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.TrainerState

A read-only snapshot. Reading state never polls or submits GPU work.
Reset refuses a pending GPU step, and discards only unfinished gradients.

### optimizer_step (1)

`uint64`.

Protobuf default: `"0"`.

### rng_seed (2)

`uint64`.

Protobuf default: `"0"`.

### accumulated_microbatches (3)

`uint32`.

Protobuf default: `0`.

### accumulation_steps (4)

`uint32`.

Protobuf default: `0`.

### accumulated_metrics (5)

`volvoxai.v1.TrainingMetric` repeated.

Protobuf default: `[]`.

### step_pending (6)

`bool`.

Protobuf default: `false`.

### has_private_updates (7)

`bool`.

Protobuf default: `false`.

### activation_signature (8)

`string`.

Protobuf default: `""`.

### optimizer_bytes (9)

`uint64`.

Protobuf default: `"0"`.

### gradient_bytes (10)

`uint64`.

Protobuf default: `"0"`.

### report (11)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### optimizer_config (12)

`volvoxai.v1.TrainerOptimizerOptions`.

Protobuf default: `null`.

### shape (13)

`volvoxai.v1.TrainerShapeState`.

Protobuf default: `null`.

## volvoxai.v1.TrainerShapeState

The actual private C binding and retained memory, independent of backend.
No device work is submitted. Before the first input batch, signature is empty.
Tensor descriptors carry logical extents; arena capacity may be larger.

### signature (1)

`string`.

Protobuf default: `""`.

### activations (2)

`volvoxai.v1.TensorInfo` repeated.

Protobuf default: `[]`.

### parameter_bytes (3)

`uint64`.

Protobuf default: `"0"`.

### activation_capacity_bytes (4)

`uint64`.

Protobuf default: `"0"`.

### activation_high_water_bytes (5)

`uint64`.

Protobuf default: `"0"`.

### activation_grow_count (6)

`uint64`.

Protobuf default: `"0"`.

### plan_cache_entries (7)

`uint32`.

Protobuf default: `0`.

### plan_cache_metadata_bytes (8)

`uint64`.

Protobuf default: `"0"`.

### plan_cache_hits (9)

`uint64`.

Protobuf default: `"0"`.

### plan_cache_misses (10)

`uint64`.

Protobuf default: `"0"`.

### plan_cache_evictions (11)

`uint64`.

Protobuf default: `"0"`.

### options (12)

`volvoxai.v1.TrainerShapeOptions`.

Protobuf default: `null`.

complete resolved policy

### activation_gradients (13)

`volvoxai.v1.TensorInfo` repeated.

Protobuf default: `[]`.

F32 logical gradient shapes

### parameter_gradients (14)

`volvoxai.v1.TensorInfo` repeated.

Protobuf default: `[]`.

F32 logical gradient shapes

### backend (15)

`string`.

Protobuf default: `""`.

### cached_plans (16)

`volvoxai.v1.TrainerCachedPlan` repeated.

Protobuf default: `[]`.

Least to most recently used. C caches resolved tensor layouts; compiled C
operator code and device pipeline objects are not JavaScript recipes.
Model lineage + backend + signature identify the binding unambiguously.

### plan_cache_oversize_skips (17)

`uint64`.

Protobuf default: `"0"`.

### plan_cache_bypasses (18)

`uint64`.

Protobuf default: `"0"`.

all admitted bindings left uncached

### plan_cache_storage_bytes (19)

`uint64`.

Protobuf default: `"0"`.

Actual entry storage including spare slots, separately from the budgeted
metadata of live cache entries. Neither value includes tensor payloads.

## volvoxai.v1.TrainerCachedPlan



### signature (1)

`string`.

Protobuf default: `""`.

### metadata_bytes (2)

`uint64`.

Protobuf default: `"0"`.

### logical_bytes (3)

`uint64`.

Protobuf default: `"0"`.

### arena_bytes (4)

`uint64`.

Protobuf default: `"0"`.

### last_used (5)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.TrainerCheckpoint

Portable, self-contained training state, serialized with the generated
protobuf codec. Weight and optimizer bytes use SafeTensors. Graph bytes are
the exact source document, including dimensions and quantization metadata.
No pointers, GPU objects or executable backend plans are persisted.
Version 1 requires optimizer bytes, including an empty state at step zero.

### version (1)

`uint32`.

Protobuf default: `0`.

### graph (2)

`bytes`.

Protobuf default: `""`.

### weight_shards (3)

`bytes` repeated.

Protobuf default: `[]`.

### optimizer (4)

`bytes`.

Protobuf default: `""`.

### optimizer_step (5)

`uint64`.

Protobuf default: `"0"`.

### rng_seed (6)

`uint64`.

Protobuf default: `"0"`.

### metadata (7)

`bytes`.

Protobuf default: `""`.

Opaque application metadata, such as tokenizer configuration. C copies it
exactly and does not interpret it as an executable engine instruction.

### optimizer_config (8)

`volvoxai.v1.TrainerOptimizerOptions`.

Protobuf default: `null`.

Required complete configuration. Omitted per-step fields resume these
values after restoration, including when changing execution backend.

## volvoxai.v1.ExportTrainerCheckpointRequest

Export requires no unfinished accumulation or pending GPU step. Omitted
metadata preserves the imported checkpoint metadata; present bytes replace it.

### trainer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trainer`.

### metadata (2)

`bytes`.

Protobuf default: `""`.

## volvoxai.v1.TrainerCheckpointInfo



### checkpoint (1)

`volvoxai.v1.TrainerCheckpoint`.

Protobuf default: `null`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.PtqObserverSpec



### tensor_name (1)

`string`.

Protobuf default: `""`.

The float tensor calibration observes, named as the source graph names it.

### dtype (2)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### scheme (3)

`volvoxai.v1.PtqScheme`.

Protobuf default: `"PTQ_SCHEME_SYMMETRIC"`.

### quantized_tensor_name (4)

`string`.

Protobuf default: `""`.

The byte tensor it becomes in the template, and therefore the tensor whose
affine the writer fills in. Authoring chooses this name; leaving it empty
says the template did not rename the value, and the writer falls back to
naming the affine itself.

## volvoxai.v1.PtqLayerSpec



### mode (1)

`volvoxai.v1.PtqMode`.

Protobuf default: `"PTQ_MODE_W8A8"`.

### kind (2)

`volvoxai.v1.PtqLayerKind`.

Protobuf default: `"PTQ_LAYER_KIND_QLINEAR"`.

### node_index (3)

`int32`.

Protobuf default: `0`.

Position in the *source* graph, which is what binds the layer to the
loaded Model's node.

### weight_axis (4)

`int32`.

Protobuf default: `0`.

### input_tensor_name (5)

`string`.

Protobuf default: `""`.

### output_tensor_name (6)

`string`.

Protobuf default: `""`.

### source_weight_name (7)

`string`.

Protobuf default: `""`.

### packed_weight_name (8)

`string`.

Protobuf default: `""`.

### source_bias_name (9)

`string`.

Protobuf default: `""`.

### packed_bias_name (10)

`string`.

Protobuf default: `""`.

### node_id (11)

`string`.

Protobuf default: `""`.

The same node's id, which is what binds it to the *template*. Authoring
inserts the affine boundaries, so the two graphs no longer agree about
positions; they do agree about ids, because a quantized node keeps the id
it had.

## volvoxai.v1.PtqAuthoringConfig

What authoring is allowed to produce.

Every field is a decision, never an inference from graph topology. A caller
that wants a different precision says so; authoring does not pick one by
looking at the weights.

### activation_dtype (1)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

Activation storage. I8 with SYMMETRIC is the conservative default and
makes every activation zero point zero; U8 with ASYMMETRIC represents an
off-centre range without spending a sign bit on the side that is empty.

### activation_scheme (2)

`volvoxai.v1.PtqScheme`.

Protobuf default: `"PTQ_SCHEME_SYMMETRIC"`.

### weight_dtype (3)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

Weight storage. Weights are per-output-channel and always symmetric: a
zero point on a weight would survive into the accumulator, where it costs
a correction term on every output element.

### reduce_range (4)

`bool`.

Protobuf default: `false`.

Quantize weights into 7 bits instead of 8, leaving headroom that keeps a
32-bit accumulator from saturating on backends without a widening
multiply. It costs accuracy and is off by default.

### float_operators (5)

`string` repeated.

Protobuf default: `[]`.

Operator kinds to leave in F32 even when they qualify. An operator named
here is not a failure — it is a region the caller has decided to keep in
float, and the surrounding affine boundaries move to match.

### float_nodes (6)

`string` repeated.

Protobuf default: `[]`.

Node ids to leave in F32, for the same reason at finer grain.

### selected_nodes (7)

`string` repeated.

Protobuf default: `[]`.

Node ids to consider. Empty means every node that qualifies, which is the
usual case; naming a subset quantizes only that region.

## volvoxai.v1.AuthorPtqTemplateRequest

Rewrites an FP32 graph into the quantized template the rest of this service
consumes, and returns the plan that describes it.

The graph and its weights arrive either as paths or as bytes. Paths suit a
native caller who already has files and wants the next call to take one;
bytes suit a host that has no filesystem to name — a browser, or anything
else running the WebAssembly build, where there is no `fopen` to call. The
authoring itself is the same code either way: it works on bytes, and the
file form only reads and writes around it.

Bytes win where both are given. Weights are read but never modified —
packing happens at WritePtqPackage, after calibration has decided the
scales.

### source_graph_path (1)

`string`.

Protobuf default: `""`.

### weight_paths (2)

`string` repeated.

Protobuf default: `[]`.

### template_graph_path (3)

`string`.

Protobuf default: `""`.

Where to write the template. Leave empty to get it back in
PtqTemplateInfo.template_graph instead.

### config (4)

`volvoxai.v1.PtqAuthoringConfig`.

Protobuf default: `null`.

### source_graph (5)

`bytes`.

Protobuf default: `""`.

### weight_shards (6)

`bytes` repeated.

Protobuf default: `[]`.

## volvoxai.v1.PtqTemplateInfo

The derived plan, in exactly the shape CreatePtqPlan takes it.

Deriving these is authoring's whole job, so they are returned rather than
held: a caller can inspect them, log them, edit one, or hand them straight
on. `required_observations` names every tensor calibration must observe
before the plan can be written, which is what a calibrator loops over.

### observers (1)

`volvoxai.v1.PtqObserverSpec` repeated.

Protobuf default: `[]`.

### layers (2)

`volvoxai.v1.PtqLayerSpec` repeated.

Protobuf default: `[]`.

### required_observations (3)

`string` repeated.

Protobuf default: `[]`.

### quantized_nodes (4)

`uint64`.

Protobuf default: `"0"`.

### retained_float_nodes (5)

`uint64`.

Protobuf default: `"0"`.

Nodes that stayed in F32, whether by configuration or because they did
not qualify. A count of zero alongside a low quantized_nodes means
authoring declined the graph, not that it converted all of it.

### report (6)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### template_graph (7)

`bytes`.

Protobuf default: `""`.

The authored template, present when no `template_graph_path` was given.
A caller that asked for a file gets an empty field and the file.

## volvoxai.v1.CreatePtqPlanRequest

Every declared profile must receive at least one successful calibration
batch before package materialization is allowed.

### model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Model`.

### template_graph_path (2)

`string`.

Protobuf default: `""`.

### profile_names (3)

`string` repeated.

Protobuf default: `[]`.

### observers (4)

`volvoxai.v1.PtqObserverSpec` repeated.

Protobuf default: `[]`.

### layers (5)

`volvoxai.v1.PtqLayerSpec` repeated.

Protobuf default: `[]`.

### template_graph (6)

`bytes`.

Protobuf default: `""`.

Exactly one of template_graph_path or template_graph must be supplied.
Bytes are snapshotted by C and work identically in native and WASM hosts.

## volvoxai.v1.PtqPlanHandle



### ptq_plan_id (1)

`int64`.

Protobuf default: `"0"`.

### inputs (2)

`volvoxai.v1.TensorSpec` repeated.

Protobuf default: `[]`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.CalibratePtqPlanRequest

One calibration call is atomic: it supplies every declared graph input
exactly once, executes one complete private-engine forward pass, and
commits observer ranges only if every input and observed tensor succeeds.

### ptq_plan_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `PtqPlan`.

### profile_name (2)

`string`.

Protobuf default: `""`.

### sample_name (3)

`string`.

Protobuf default: `""`.

### sample_count (4)

`uint64`.

Protobuf default: `"0"`.

Number of logical examples represented by this batch.

### inputs (5)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

## volvoxai.v1.PtqCalibrationInfo



### calibration_batches (1)

`uint64`.

Protobuf default: `"0"`.

### calibration_samples (2)

`uint64`.

Protobuf default: `"0"`.

### coverage_complete (3)

`bool`.

Protobuf default: `false`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.PtqTensorParameters



### tensor_name (1)

`string`.

Protobuf default: `""`.

### dtype (2)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### scheme (3)

`volvoxai.v1.PtqScheme`.

Protobuf default: `"PTQ_SCHEME_SYMMETRIC"`.

### scale (4)

`float`.

Protobuf default: `0`.

### zero_point (5)

`int32`.

Protobuf default: `0`.

### observed_min (6)

`float`.

Protobuf default: `0`.

### observed_max (7)

`float`.

Protobuf default: `0`.

### observed_values (8)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.PtqSymbolCoverage



### symbol (1)

`string`.

Protobuf default: `""`.

### minimum (2)

`int64`.

Protobuf default: `"0"`.

### maximum (3)

`int64`.

Protobuf default: `"0"`.

## volvoxai.v1.PtqActivationCoverage



### tensor_name (1)

`string`.

Protobuf default: `""`.

### observed_values (2)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.PtqProfileCoverage



### name (1)

`string`.

Protobuf default: `""`.

### batches (2)

`uint64`.

Protobuf default: `"0"`.

### samples (3)

`uint64`.

Protobuf default: `"0"`.

### signatures (4)

`string` repeated.

Protobuf default: `[]`.

### symbols (5)

`volvoxai.v1.PtqSymbolCoverage` repeated.

Protobuf default: `[]`.

### activations (6)

`volvoxai.v1.PtqActivationCoverage` repeated.

Protobuf default: `[]`.

## volvoxai.v1.PtqCoverage

The typed form of the shared volvox.ptq-coverage/v1 record. This replaces
the previous JSON string projection; there is no JSON escape hatch.

### format (1)

`string`.

Protobuf default: `""`.

### logical_fingerprint (2)

`string`.

Protobuf default: `""`.

### complete (3)

`bool`.

Protobuf default: `false`.

### total_batches (4)

`uint64`.

Protobuf default: `"0"`.

### total_samples (5)

`uint64`.

Protobuf default: `"0"`.

### profiles (6)

`volvoxai.v1.PtqProfileCoverage` repeated.

Protobuf default: `[]`.

## volvoxai.v1.PtqPlanInfo



### calibration_batches (1)

`uint64`.

Protobuf default: `"0"`.

### calibration_samples (2)

`uint64`.

Protobuf default: `"0"`.

### tensors (3)

`volvoxai.v1.PtqTensorParameters` repeated.

Protobuf default: `[]`.

### coverage (4)

`volvoxai.v1.PtqCoverage`.

Protobuf default: `null`.

### revision (5)

`volvoxai.v1.RevisionInfo`.

Protobuf default: `null`.

### report (6)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.WritePtqPackageRequest



### ptq_plan_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `PtqPlan`.

### output_graph_path (2)

`string`.

Protobuf default: `""`.

Empty paths return package bytes. Native paths must be supplied together;
the graph basename must be exactly graph.json.

### output_weights_path (3)

`string`.

Protobuf default: `""`.

Must be a sibling *.safetensors file.

## volvoxai.v1.PtqPackageInfo



### graph_path (1)

`string`.

Protobuf default: `""`.

### safetensors_path (2)

`string`.

Protobuf default: `""`.

### plan (3)

`volvoxai.v1.PtqPlanInfo`.

Protobuf default: `null`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### graph (5)

`bytes`.

Protobuf default: `""`.

Populated for byte export; filenames are graph.json and weights.safetensors.

### weights (6)

`bytes`.

Protobuf default: `""`.

## volvoxai.v1.MemoryOwnerRef

Opaque owner identity. Public handles use decimal spelling; process scope uses "0".

### kind (1)

`volvoxai.v1.MemoryOwnerKind`.

Protobuf default: `"MEMORY_OWNER_KIND_UNSPECIFIED"`.

### owner_id (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.MemoryByteSize

Message presence, unlike a proto3 scalar default, distinguishes an omitted
measurement from an exact zero in every generated language.

### bytes (1)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.MemoryBoundTerm

One disjoint accounting unit in a compile-time peak case. term_id is unique
within its case. Shared/aliased physical storage appears once in a case
even when multiple objects consume it.

### term_id (1)

`string`.

Protobuf default: `""`.

### owner_kind (2)

`volvoxai.v1.MemoryOwnerKind`.

Protobuf default: `"MEMORY_OWNER_KIND_UNSPECIFIED"`.

### role (3)

`volvoxai.v1.MemoryResourceRole`.

Protobuf default: `"MEMORY_RESOURCE_ROLE_UNSPECIFIED"`.

### space (4)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### upper_bound_bytes (5)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

## volvoxai.v1.MemoryPeakCase

Every term in one case is simultaneously resident and physically disjoint.
A present case contains at least one term; exact-zero cases use a present
zero-byte term. total_bytes must equal the checked sum of upper_bound_bytes.
Cases are alternatives: callers take their maximum and never sum case
totals.

### case_id (1)

`string`.

Protobuf default: `""`.

### terms (2)

`volvoxai.v1.MemoryBoundTerm` repeated.

Protobuf default: `[]`.

### total_bytes (3)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

## volvoxai.v1.MemoryBoundProof

maximum_bytes must equal max(cases.total_bytes) when cases are present.
Empty cases allow a provider to publish its existing coarse attestation
before it implements detailed terms. limit_bytes is absent when the backend
or device exposes no meaningful limit.

### budget_domain_id (1)

`string`.

Protobuf default: `""`.

Stable provider-defined accounting domain, for example
"provider-resident", "activation-capacity", or "wasm-linear". It
disambiguates independent limits with the same canonical kind.

### kind (2)

`volvoxai.v1.MemoryBoundKind`.

Protobuf default: `"MEMORY_BOUND_KIND_UNSPECIFIED"`.

### cases (3)

`volvoxai.v1.MemoryPeakCase` repeated.

Protobuf default: `[]`.

### maximum_bytes (4)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

### limit_bytes (5)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

## volvoxai.v1.MemoryDomainAttestation

Compile-time evidence for the complete bounded shape domain. Bounds are
computed proof, not observed allocation or RSS/VRAM measurements. Consumers
accept only the canonical proof/resource protocols named below and require
at least one bound. They reject missing maxima, missing case/term byte
presence, duplicate (budget_domain_id, kind), duplicate case/term IDs,
checked-sum or max mismatches, and a maximum above its present limit. Empty
cases are allowed only for a coarse maximum supplied by an existing
provider attestation.

### proof_protocol (1)

`string`.

Protobuf default: `""`.

Exactly "canonical-symbolic-domain-proof/v1" in format v1.

### resource_protocol (2)

`string`.

Protobuf default: `""`.

Exactly "bounded-resource-maxima/v1" in format v1.

### graph_fingerprint (3)

`string`.

Protobuf default: `""`.

### shape_domain_proof_identity (4)

`string`.

Protobuf default: `""`.

### bounds (5)

`volvoxai.v1.MemoryBoundProof` repeated.

Protobuf default: `[]`.

## volvoxai.v1.MemoryMeasurement



### metric (1)

`volvoxai.v1.MemoryMetric`.

Protobuf default: `"MEMORY_METRIC_UNSPECIFIED"`.

### bytes (2)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

### source (3)

`volvoxai.v1.MemoryEvidenceSource`.

Protobuf default: `"MEMORY_EVIDENCE_SOURCE_UNSPECIFIED"`.

### value_relation (4)

`volvoxai.v1.MemoryValueRelation`.

Protobuf default: `"MEMORY_VALUE_RELATION_UNSPECIFIED"`.

### temporal_coverage (5)

`volvoxai.v1.MemoryTemporalCoverage`.

Protobuf default: `"MEMORY_TEMPORAL_COVERAGE_UNSPECIFIED"`.

## volvoxai.v1.MemoryResourceEvidence

resource_id is opaque, contains no address, and is stable for the
resource's lifetime within one capture. ALIAS/SUBALLOCATION records name
their physical range within backing_resource_id. Backing links are acyclic
and resolve in the same capture. Physical totals count an independent root
or a proved set of disjoint descendant ranges, never both. Equal content is
not allocation identity: physical copies always have different resource_id
values. addressable_bytes is required even when it is exact zero. It is
immutable range geometry, not a live/reserved measurement and not
independently additive. For ALIAS/SUBALLOCATION it equals
backing_length_bytes; for an INDEPENDENT root it supplies the extent needed
to validate child ranges. allocator is non-empty and every resource carries
at least one measurement. Measurements use allocator/runtime/API sources
only; API_REQUEST and REQUESTED occur together. High-water/cumulative
metrics never use INSTANT.

### resource_id (1)

`string`.

Protobuf default: `""`.

### backing_resource_id (2)

`string`.

Protobuf default: `""`.

### backing_relation (3)

`volvoxai.v1.MemoryBackingRelation`.

Protobuf default: `"MEMORY_BACKING_RELATION_UNSPECIFIED"`.

### backing_offset_bytes (4)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

### backing_length_bytes (5)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

### owner (6)

`volvoxai.v1.MemoryOwnerRef`.

Protobuf default: `null`.

### role (7)

`volvoxai.v1.MemoryResourceRole`.

Protobuf default: `"MEMORY_RESOURCE_ROLE_UNSPECIFIED"`.

### space (8)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### allocator (9)

`string`.

Protobuf default: `""`.

### consumers (10)

`volvoxai.v1.MemoryOwnerRef` repeated.

Protobuf default: `[]`.

### measurements (11)

`volvoxai.v1.MemoryMeasurement` repeated.

Protobuf default: `[]`.

### addressable_bytes (12)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

## volvoxai.v1.MemoryEnvelopeEvidence

sampler is the non-empty provider/method identity. Process RSS/PSS/private
envelopes use an OS sampler, managed-heap/external/ArrayBuffer envelopes
use a runtime counter, and device envelopes use a driver sampler. Envelopes
never use REQUESTED. PROCESS_PEAK_RSS is EXACT over PROCESS_LIFETIME (or is
UNAVAILABLE), and available SAMPLED_WINDOW values are LOWER_BOUND.

### kind (1)

`volvoxai.v1.MemoryEnvelopeKind`.

Protobuf default: `"MEMORY_ENVELOPE_KIND_UNSPECIFIED"`.

### bytes (2)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

### source (3)

`volvoxai.v1.MemoryEvidenceSource`.

Protobuf default: `"MEMORY_EVIDENCE_SOURCE_UNSPECIFIED"`.

### value_relation (4)

`volvoxai.v1.MemoryValueRelation`.

Protobuf default: `"MEMORY_VALUE_RELATION_UNSPECIFIED"`.

### temporal_coverage (5)

`volvoxai.v1.MemoryTemporalCoverage`.

Protobuf default: `"MEMORY_TEMPORAL_COVERAGE_UNSPECIFIED"`.

### sampler (6)

`string`.

Protobuf default: `""`.

## volvoxai.v1.MemorySnapshot

An on-demand memory observation, or one of a trace's start/stop observations.
Inventories explicitly state coverage. Partial records never prove a total.
Direct queries use the host monotonic clock. In TracePage, times use the
same capture-relative origin as TraceEvent. Sequence is 1/2 for trace samples,
and zero for a standalone query. No globally atomic snapshot is implied.

### sequence (1)

`uint64`.

Protobuf default: `"0"`.

### subject (2)

`volvoxai.v1.MemoryOwnerRef`.

Protobuf default: `null`.

### backend (3)

`string`.

Protobuf default: `""`.

### device (4)

`string`.

Protobuf default: `""`.

### resources (5)

`volvoxai.v1.MemoryResourceEvidence` repeated.

Protobuf default: `[]`.

### envelopes (6)

`volvoxai.v1.MemoryEnvelopeEvidence` repeated.

Protobuf default: `[]`.

### resource_inventory (7)

`volvoxai.v1.MemoryInventoryKind`.

Protobuf default: `"MEMORY_INVENTORY_KIND_UNSPECIFIED"`.

### observation_start_ns (8)

`uint64`.

Protobuf default: `"0"`.

### observation_end_ns (9)

`uint64`.

Protobuf default: `"0"`.

### counters (10)

`volvoxai.v1.MemoryCounter` repeated.

Protobuf default: `[]`.

Scoped counters are not an allocation inventory or an additive total.

## volvoxai.v1.ExecutionMetrics



### host_time_ns (1)

`uint64`.

Protobuf default: `"0"`.

Host execution and output submission; excludes asynchronous device completion.
Includes synchronous device waits performed by the backend. Not kernel time.

### output_bytes (2)

`uint64`.

Protobuf default: `"0"`.

Logical output payload; never process RSS or total allocator residency.

## volvoxai.v1.StartTraceRequest



### runtime_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Runtime`.

### detail (2)

`volvoxai.v1.TraceDetail`.

Protobuf default: `"TRACE_DETAIL_BASIC"`.

One detail level for host and device observations.

### device_timing (3)

`bool`.

Protobuf default: `false`.

Opt-in GPU elapsed time; the engine chooses the timestamp mechanism.
False creates no GPU timing resources, even when detail is NODES.
Device node timing may split passes or add barriers; see devices in TraceInfo.

### memory (4)

`bool`.

Protobuf default: `false`.

Opt-in bounded allocator history and two process RSS observations.
Allocator coverage is partial; RSS is not an allocator total or window peak.

### capacity_bytes (5)

`uint64`.

Protobuf default: `"0"`.

Engine default: 4194304

Allocated once at start. Includes fixed event metadata; [4 KiB, 64 MiB].

## volvoxai.v1.TraceRef



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

## volvoxai.v1.TraceDeviceCoverage

One row per encountered built-in backend. No device is probed by StartTrace.
Support describes timestamp availability; successful records and failures
describe this capture. Unsupported timing does not prevent host collection.

### backend (1)

`string`.

Protobuf default: `""`.

### support (2)

`volvoxai.v1.TraceSupport`.

Protobuf default: `"TRACE_SUPPORT_UNOBSERVED"`.

### node_timing_available (3)

`bool`.

Protobuf default: `false`.

Node timestamp support was observed on at least one encountered device
path. This does not promise node coverage for every operation. Counts
below describe successful observations at the requested detail.

### pass_intervals (4)

`uint64`.

Protobuf default: `"0"`.

### node_intervals (5)

`uint64`.

Protobuf default: `"0"`.

### failed_intervals (6)

`uint64`.

Protobuf default: `"0"`.

### unavailable_passes (7)

`uint64`.

Protobuf default: `"0"`.

### splits_passes (8)

`bool`.

Protobuf default: `false`.

Instrumentation changes that may alter the measured execution schedule.

### adds_barriers (9)

`bool`.

Protobuf default: `false`.

### program_timing_available (10)

`bool`.

Protobuf default: `false`.

Program timing support was observed on an encountered path. Programs are
engine-owned dispatches, not a complete driver-level kernel inventory.

### program_intervals (11)

`uint64`.

Protobuf default: `"0"`.

### calibrated_intervals (12)

`uint64`.

Protobuf default: `"0"`.

### bounded_intervals (13)

`uint64`.

Protobuf default: `"0"`.

### copy_intervals (14)

`uint64`.

Protobuf default: `"0"`.

Device copies, separate from pass/node/program.

### host_copy_calls (15)

`uint64`.

Protobuf default: `"0"`.

### host_wait_calls (16)

`uint64`.

Protobuf default: `"0"`.

Blocking waits; asynchronous awaits are separate.

### host_submit_calls (17)

`uint64`.

Protobuf default: `"0"`.

### host_awaits (18)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.TraceInfo



### trace_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### state (3)

`volvoxai.v1.TraceState`.

Protobuf default: `"TRACE_STATE_COLLECTING"`.

### detail (4)

`volvoxai.v1.TraceDetail`.

Protobuf default: `"TRACE_DETAIL_BASIC"`.

### device_timing (5)

`bool`.

Protobuf default: `false`.

### memory (6)

`bool`.

Protobuf default: `false`.

### capacity_bytes (7)

`uint64`.

Protobuf default: `"0"`.

### event_count (8)

`uint64`.

Protobuf default: `"0"`.

Includes reserved pending slots until READY; finalized count is immutable.

### dropped_events (9)

`uint64`.

Protobuf default: `"0"`.

Events lost to capacity or device errors. Unsupported timing is separate.

### active_operations (10)

`uint64`.

Protobuf default: `"0"`.

Host scopes and pending asynchronous host completion observations.
Device observations are counted separately below.

### pending_device_intervals (11)

`uint64`.

Protobuf default: `"0"`.

### host_clock_resolution_ns (12)

`uint64`.

Protobuf default: `"0"`.

Zero means unknown, including browser quantization. Storage in nanoseconds
does not imply nanosecond precision. Device precision is not inferred here.

### devices (13)

`volvoxai.v1.TraceDeviceCoverage` repeated.

Protobuf default: `[]`.

### allocators (14)

`volvoxai.v1.TraceAllocatorMemory` repeated.

Protobuf default: `[]`.

## volvoxai.v1.TraceHostSpan



### start_ns (1)

`uint64`.

Protobuf default: `"0"`.

Monotonic host time relative to capture start. An AWAIT activity measures
asynchronous completion latency, not time occupying the submitting thread.

### duration_ns (2)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.TraceDeviceInterval



### host_observed_ns (1)

`uint64`.

Protobuf default: `"0"`.

Host marker observed before device work is encoded/submitted, not GPU start.

### elapsed_ns (2)

`uint64`.

Protobuf default: `"0"`.

### correlation (3)

`volvoxai.v1.TraceClockCorrelation`.

Protobuf default: `null`.

Absent when no valid relation to the capture-relative host clock exists.

## volvoxai.v1.TraceClockCorrelation



### method (1)

`volvoxai.v1.TraceClockMethod`.

Protobuf default: `"TRACE_CLOCK_METHOD_UNSPECIFIED"`.

### earliest_start_ns (2)

`uint64`.

Protobuf default: `"0"`.

Range for the device interval's start on the capture-relative host clock.
Includes observed calibration uncertainty; it is not a precise start time.
Do not infer queue delay or overlap more precisely than this range permits.

### latest_start_ns (3)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.TraceQueue



### device_id (1)

`uint64`.

Protobuf default: `"0"`.

Opaque identities, meaningful only within this capture. No native handles.
Devices from different backend APIs are not asserted to be the same GPU.

### queue_id (2)

`uint64`.

Protobuf default: `"0"`.

### submission_id (3)

`uint64`.

Protobuf default: `"0"`.

Zero when the observation is not tied to one known submission/batch.

## volvoxai.v1.TraceCopy



### source (1)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### destination (2)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### bytes (3)

`uint64`.

Protobuf default: `"0"`.

Bytes passed to this copy, including required padding.

## volvoxai.v1.TraceNode



### schedule_index (1)

`uint32`.

Protobuf default: `0`.

Index in the executable schedule. Fused duration belongs to this work;
it must never be divided among eliminated source nodes.

### output_name (2)

`string`.

Protobuf default: `""`.

### fused (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.TraceProgram

A measured engine program invocation. Names describe the selected program,
not every candidate considered during planning. A CUDA program can launch
several internal kernels; this is not a CUPTI kernel activity record.

### name (1)

`string`.

Protobuf default: `""`.

### entry_point (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.TraceEvent



### sequence (1)

`uint64`.

Protobuf default: `"0"`.

### name (2)

`string`.

Protobuf default: `""`.

### track_id (3)

`uint64`.

Protobuf default: `"0"`.

Host thread; device intervals annotate this submitter.

### lineage (4)

`volvoxai.v1.Lineage`.

Protobuf default: `null`.

### backend (5)

`string`.

Protobuf default: `""`.

### node (6)

`volvoxai.v1.TraceNode`.

Protobuf default: `null`.

With program: the owning executable node, when known. Without program:
the node measured by this event. A node's duration includes its programs;
these nested observations must not be added together as independent work.

### metadata_truncated (7)

`bool`.

Protobuf default: `false`.

### host (8)

`volvoxai.v1.TraceHostSpan`; oneof `observation`.

Protobuf default: `null`.

### device (9)

`volvoxai.v1.TraceDeviceInterval`; oneof `observation`.

Protobuf default: `null`.

### memory (10)

`volvoxai.v1.TraceMemoryEvent`; oneof `observation`.

Protobuf default: `null`.

### phase (11)

`volvoxai.v1.TracePhase`.

Protobuf default: `"TRACE_PHASE_UNSPECIFIED"`.

### program (12)

`volvoxai.v1.TraceProgram`.

Protobuf default: `null`.

Presence distinguishes a program interval from its enclosing node/pass.

### tensor_name (13)

`string`.

Protobuf default: `""`.

A known work target, e.g. the parameter updated by an optimizer. Empty when
unknown or work spans multiple tensors. This does not invent a node owner.

### activity (14)

`volvoxai.v1.TraceActivity`.

Protobuf default: `"TRACE_ACTIVITY_WORK"`.

### queue (15)

`volvoxai.v1.TraceQueue`.

Protobuf default: `null`.

### copy (16)

`volvoxai.v1.TraceCopy`.

Protobuf default: `null`.

Present on COPY events; host and device may overlap.

## volvoxai.v1.TraceMemoryEvent



### timestamp_ns (1)

`uint64`.

Protobuf default: `"0"`.

### allocation_id (2)

`uint64`.

Protobuf default: `"0"`.

### allocator (3)

`string`.

Protobuf default: `""`.

### space (4)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### action (5)

`volvoxai.v1.TraceMemoryAction`.

Protobuf default: `"TRACE_MEMORY_ACTION_EXISTING"`.

### bytes (6)

`uint64`.

Protobuf default: `"0"`.

### live_bytes (7)

`uint64`.

Protobuf default: `"0"`.

This allocator's tracked requested capacity after the event.

## volvoxai.v1.TraceAllocatorMemory

Requested capacities of observed owned allocations, not physical residency.
Inventories are PARTIAL: metadata, driver allocations and unvisited owners
can be absent. Counters are additive only within their stated allocator.
Existing resources are inventoried when an owner first participates. A peak
covers observed allocator transitions from that point until StopTrace.

### allocator (1)

`string`.

Protobuf default: `""`.

### backend (2)

`string`.

Protobuf default: `""`.

### space (3)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### inventory (4)

`volvoxai.v1.MemoryInventoryKind`.

Protobuf default: `"MEMORY_INVENTORY_KIND_UNSPECIFIED"`.

### observation_start_ns (5)

`uint64`.

Protobuf default: `"0"`.

### existing_bytes (6)

`uint64`.

Protobuf default: `"0"`.

### live_bytes (7)

`uint64`.

Protobuf default: `"0"`.

### peak_bytes (8)

`uint64`.

Protobuf default: `"0"`.

### allocated_bytes (9)

`uint64`.

Protobuf default: `"0"`.

### freed_bytes (10)

`uint64`.

Protobuf default: `"0"`.

### dropped_events (11)

`uint64`.

Protobuf default: `"0"`.

### accounting_complete (12)

`bool`.

Protobuf default: `false`.

False if the bounded live-identity table overflowed. Byte counters then
describe only the tracked subset. Event loss alone does not lose counters.

### scope (13)

`volvoxai.v1.MemoryOwnerKind`.

Protobuf default: `"MEMORY_OWNER_KIND_UNSPECIFIED"`.

RUNTIME aggregates observed owners of the captured runtime. BACKEND_SHARED
includes every runtime using the same host bridge (currently WebGPU).
Never add BACKEND_SHARED counters from simultaneous runtime captures.

## volvoxai.v1.ReadTraceRequest



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

### offset (2)

`uint64`.

Protobuf default: `"0"`.

### limit (3)

`uint32`.

Protobuf default: `0`.

Engine default: 256

[1, 4096]

## volvoxai.v1.TracePage



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### events (2)

`volvoxai.v1.TraceEvent` repeated.

Protobuf default: `[]`.

### next_offset (3)

`uint64`.

Protobuf default: `"0"`.

### eof (4)

`bool`.

Protobuf default: `false`.

### process_memory (5)

`volvoxai.v1.MemorySnapshot` repeated.

Protobuf default: `[]`.

At most two process snapshots, returned on the first page only.

## volvoxai.v1.ExportChromeTraceRequest



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

### offset (2)

`uint64`.

Protobuf default: `"0"`.

Zero-based source event offset, as in ReadTrace.

### limit (3)

`uint32`.

Protobuf default: `0`.

Engine default: 128

Maximum source events serialized per call. Fixed trace metadata is added
on the first/last page; JSON bytes are variable-sized and are not cached.
[1, 1024]

## volvoxai.v1.TraceChunk



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### data (2)

`bytes`.

Protobuf default: `""`.

A UTF-8 JSON fragment. Only the concatenation of all pages is a document.

### next_offset (3)

`uint64`.

Protobuf default: `"0"`.

Next source event offset. Repeating the same request returns the same bytes.

### eof (4)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.GetMemorySnapshotRequest

Select exactly one scope. Process envelopes stay process-scoped even when
requested alongside a runtime/context inventory; they are never additive.

### process (1)

`bool`; oneof `scope`.

Protobuf default: `false`.

must be true

### runtime_id (2)

`int64`; oneof `scope`.

Protobuf default: `"0"`.

### context_id (3)

`int64`; oneof `scope`.

Protobuf default: `"0"`.

### include_process (4)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.MemorySnapshotResponse



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### snapshot (2)

`volvoxai.v1.MemorySnapshot`.

Protobuf default: `null`.

## volvoxai.v1.MemoryCounter



### name (1)

`string`.

Protobuf default: `""`.

### owner (2)

`volvoxai.v1.MemoryOwnerRef`.

Protobuf default: `null`.

### measurement (3)

`volvoxai.v1.MemoryMeasurement`.

Protobuf default: `null`.

## volvoxai.v1.TokenizationMode



- `TOKENIZATION_MODE_AUTO = 0`: Greedy byte matching when no valid merges exist; otherwise Unicode
letter/number/punctuation/whitespace pretokenization and ranked BPE.
- `TOKENIZATION_MODE_GREEDY = 1`
- `TOKENIZATION_MODE_BPE_WORD = 2`: Apply BPE to the whole text as one chunk, without pretokenization.

## volvoxai.v1.BatchWorkKind

C owns compatibility grouping, admission, token budgets, padding, lane/page
reservations, prefix reuse and retirement. The application supplies work and
executes each returned dispatch using its chosen worker. There is no product
TypeScript scheduler. Inference workers can use VxInferenceService; page plans
here describe worker-owned storage and are not an ExecutionContext binding.

- `BATCH_WORK_KIND_STATELESS = 0`
- `BATCH_WORK_KIND_PREFILL = 1`
- `BATCH_WORK_KIND_DECODE = 2`

## volvoxai.v1.BatchWorkState



- `BATCH_WORK_STATE_UNKNOWN = 0`: invalid ID or evicted terminal record
- `BATCH_WORK_STATE_QUEUED = 1`
- `BATCH_WORK_STATE_PREFILL = 2`
- `BATCH_WORK_STATE_DECODING = 3`
- `BATCH_WORK_STATE_COMPLETED = 4`
- `BATCH_WORK_STATE_CANCELLED = 5`
- `BATCH_WORK_STATE_FAILED = 6`

## volvoxai.v1.BatchQueuePolicy



- `BATCH_QUEUE_POLICY_WORK_CONSERVING = 0`
- `BATCH_QUEUE_POLICY_FILL_FIRST = 1`

## volvoxai.v1.DataType

Canonical tensor element/storage vocabulary. Enum membership describes a
representation, not execution support: each operation validates the subset
it can execute. Keep existing numbers stable and append future dtypes.

- `DATA_TYPE_UNSPECIFIED = 0`
- `DATA_TYPE_BOOL = 1`
- `DATA_TYPE_F4 = 2`
- `DATA_TYPE_F6_E2M3 = 3`
- `DATA_TYPE_F6_E3M2 = 4`
- `DATA_TYPE_U8 = 5`
- `DATA_TYPE_I8 = 6`
- `DATA_TYPE_F8_E5M2 = 7`
- `DATA_TYPE_F8_E4M3 = 8`
- `DATA_TYPE_F8_E8M0 = 9`
- `DATA_TYPE_F8_E4M3FNUZ = 10`
- `DATA_TYPE_F8_E5M2FNUZ = 11`
- `DATA_TYPE_I16 = 12`
- `DATA_TYPE_U16 = 13`
- `DATA_TYPE_F16 = 14`
- `DATA_TYPE_BF16 = 15`
- `DATA_TYPE_I32 = 16`
- `DATA_TYPE_U32 = 17`
- `DATA_TYPE_F32 = 18`
- `DATA_TYPE_C64 = 19`
- `DATA_TYPE_F64 = 20`
- `DATA_TYPE_I64 = 21`
- `DATA_TYPE_U64 = 22`

## volvoxai.v1.MemoryLocation



- `MEMORY_LOCATION_HOST = 0`
- `MEMORY_LOCATION_DEVICE = 1`

## volvoxai.v1.BackendPolicyMode



- `BACKEND_POLICY_MODE_PREFER = 0`
- `BACKEND_POLICY_MODE_REQUIRE = 1`

## volvoxai.v1.OperatorFallback



- `OPERATOR_FALLBACK_ALLOW = 0`
- `OPERATOR_FALLBACK_FORBID = 1`

## volvoxai.v1.DecodeRowMode



- `DECODE_ROW_MODE_DISABLED = 0`
- `DECODE_ROW_MODE_AUTO = 1`
- `DECODE_ROW_MODE_REQUIRED = 2`

## volvoxai.v1.ResultState

Execution acceptance and device completion are separate. A pending result
already owns an immutable output snapshot; later executions cannot change it.

- `RESULT_STATE_UNSPECIFIED = 0`
- `RESULT_STATE_PENDING = 1`
- `RESULT_STATE_READY = 2`
- `RESULT_STATE_FAILED = 3`

## volvoxai.v1.ExecutionMode

Runtime dispatch ownership. DIRECT bypasses Runtime scheduling entirely and
rejects every VxSchedulerService operation. SCHEDULED permits both direct
Run calls and scheduled Submit calls. CreateRuntime defaults to DIRECT when
execution_mode is absent.

- `EXECUTION_MODE_DIRECT = 0`
- `EXECUTION_MODE_SCHEDULED = 1`

## volvoxai.v1.BuildProfile

Which build profile a library was composed with. An inference build does
not register VxTrainingService or VxQuantizationService.

- `BUILD_PROFILE_UNSPECIFIED = 0`
- `BUILD_PROFILE_INFERENCE = 1`
- `BUILD_PROFILE_FULL = 2`

## volvoxai.v1.TransportProfile

IN_PROCESS accepts BorrowedBuffer descriptors in its local address space.
REMOTE accepts inline bytes and server-owned BufferView capabilities, but
rejects local pointers, native resources, access mappings and DLPack. A
network adapter must enforce this boundary before forwarding native calls.

- `TRANSPORT_PROFILE_UNSPECIFIED = 0`
- `TRANSPORT_PROFILE_IN_PROCESS = 1`
- `TRANSPORT_PROFILE_REMOTE = 2`

## volvoxai.v1.TrainingOptimizerKind



- `TRAINING_OPTIMIZER_KIND_ADAMW = 0`
- `TRAINING_OPTIMIZER_KIND_SGD = 1`

## volvoxai.v1.PtqMode

The full native authoring profile currently emits physical W8A8
QLinear/QConv2D packages. Other precision modes are deliberately absent
until their package writers and end-to-end execution contracts exist.

- `PTQ_MODE_W8A8 = 0`

## volvoxai.v1.PtqScheme



- `PTQ_SCHEME_SYMMETRIC = 0`
- `PTQ_SCHEME_ASYMMETRIC = 1`

## volvoxai.v1.PtqLayerKind



- `PTQ_LAYER_KIND_QLINEAR = 0`
- `PTQ_LAYER_KIND_QCONV2D = 1`

## volvoxai.v1.NativeStatus



- `NATIVE_STATUS_OK = 0`
- `NATIVE_STATUS_INVALID_ARGUMENT = -1`
- `NATIVE_STATUS_OUT_OF_MEMORY = -2`
- `NATIVE_STATUS_HANDLE_DISPOSED = -3`
- `NATIVE_STATUS_IO_ERROR = -4`
- `NATIVE_STATUS_INVALID_GRAPH = -5`
- `NATIVE_STATUS_BACKEND_UNAVAILABLE = -6`
- `NATIVE_STATUS_BACKEND_UNSUPPORTED = -7`
- `NATIVE_STATUS_BACKEND_REQUIRED = -8`
- `NATIVE_STATUS_OPERATOR_FALLBACK_FORBIDDEN = -9`
- `NATIVE_STATUS_EXECUTION_FAILED = -10`
- `NATIVE_STATUS_RESULT_DISPOSED = -11`
- `NATIVE_STATUS_DEVICE_LOST = -12`
- `NATIVE_STATUS_ABI_UNSUPPORTED = -13`
- `NATIVE_STATUS_NOT_FOUND = -14`
- `NATIVE_STATUS_BUFFER_TOO_SMALL = -15`
- `NATIVE_STATUS_INTERNAL = -16`
- `NATIVE_STATUS_REVISION_CONFLICT = -17`
- `NATIVE_STATUS_BUSY = -18`
- `NATIVE_STATUS_OVERLOADED = -19`
- `NATIVE_STATUS_CANCELLED = -20`
- `NATIVE_STATUS_DEADLINE_EXCEEDED = -21`
- `NATIVE_STATUS_SUPERSEDED = -22`
- `NATIVE_STATUS_CONTEXT_RESET_REQUIRED = -23`
- `NATIVE_STATUS_TRANSPORT_UNSUPPORTED = -24`: A BorrowedBuffer or local interop request reached an incompatible transport.

## volvoxai.v1.OperationCode

Machine-readable detail. Status describes the broad outcome; code preserves
the specific cause independently of human-readable message text.

- `OPERATION_CODE_NONE = 0`
- `OPERATION_CODE_ABI_UNSUPPORTED = 1`
- `OPERATION_CODE_ACCUMULATION_PENDING = 2`
- `OPERATION_CODE_ACTIVATION_CAPACITY_EXCEEDED = 3`
- `OPERATION_CODE_ACTIVATION_PLAN_INVALID = 4`
- `OPERATION_CODE_ACTIVATION_SIGNATURE_MISMATCH = 5`
- `OPERATION_CODE_ADAPTER_NOT_READABLE = 6`
- `OPERATION_CODE_ADAPTER_PACKAGE_REJECTED = 7`
- `OPERATION_CODE_ADAPTER_REVISION_MISSING = 8`
- `OPERATION_CODE_ADAPTER_REVISION_NOT_FOUND = 9`
- `OPERATION_CODE_ADAPTER_UNSUPPORTED = 10`
- `OPERATION_CODE_BACKEND_CONTEXT_CREATE_FAILED = 11`
- `OPERATION_CODE_BACKEND_ENUMERATION_FAILED = 12`
- `OPERATION_CODE_BACKEND_REQUIRED = 13`
- `OPERATION_CODE_BACKEND_UNAVAILABLE = 14`
- `OPERATION_CODE_BACKEND_UNSUPPORTED = 15`
- `OPERATION_CODE_BASE_ADAPTER_MISSING = 16`
- `OPERATION_CODE_BATCH_AXIS_INVALID = 17`
- `OPERATION_CODE_BATCH_AXIS_MISMATCH = 18`
- `OPERATION_CODE_BATCH_CONTRACT_INVALID = 19`
- `OPERATION_CODE_BATCH_EVIDENCE_UNAVAILABLE = 20`
- `OPERATION_CODE_BATCH_OUTPUT_AXIS_INVALID = 21`
- `OPERATION_CODE_BATCH_PROOF_MISMATCH = 22`
- `OPERATION_CODE_BATCH_RESULT_MISSING = 23`
- `OPERATION_CODE_BOUNDED_DOMAIN_UNSUPPORTED = 24`
- `OPERATION_CODE_BUFFER_TOO_SMALL = 25`
- `OPERATION_CODE_BUSY = 26`
- `OPERATION_CODE_CALIBRATION_EXECUTION_FAILED = 27`
- `OPERATION_CODE_CALL_SEQUENCE_POLICY_INVALID = 28`
- `OPERATION_CODE_CALL_SEQUENCE_SELECTION_INVALID = 29`
- `OPERATION_CODE_CANCELLED = 30`
- `OPERATION_CODE_CANONICAL_SHAPE_CONTRACT_UNSUPPORTED = 31`
- `OPERATION_CODE_CHECKPOINT_EXPORT_FAILED = 32`
- `OPERATION_CODE_CLOCK_FAILED = 33`
- `OPERATION_CODE_COMMIT_FAILED = 34`
- `OPERATION_CODE_CONDITION_INIT_FAILED = 35`
- `OPERATION_CODE_CONTEXT_RESET_REQUIRED = 36`
- `OPERATION_CODE_COORDINATOR_INIT_FAILED = 37`
- `OPERATION_CODE_DEADLINE_EXCEEDED = 38`
- `OPERATION_CODE_DECODE_CACHE_CAPACITY = 39`
- `OPERATION_CODE_DECODE_CACHE_REJECTED = 40`
- `OPERATION_CODE_DECODE_MODE_UNSUPPORTED = 41`
- `OPERATION_CODE_DECODE_NOT_ENABLED = 42`
- `OPERATION_CODE_DECODE_PREFILL_REQUIRED = 43`
- `OPERATION_CODE_DECODE_RESET_FAILED = 44`
- `OPERATION_CODE_DECODE_RESOURCE_DOMAIN_UNSUPPORTED = 45`
- `OPERATION_CODE_DECODE_ROW_DOMAIN_UNSUPPORTED = 46`
- `OPERATION_CODE_DECODE_STATE_WRITE_FAILED = 47`
- `OPERATION_CODE_DEVICE_EXECUTION_FAILED = 48`
- `OPERATION_CODE_DEVICE_LOST = 49`
- `OPERATION_CODE_DUPLICATE_DECODE_INPUT = 50`
- `OPERATION_CODE_DUPLICATE_EXPORT_PATH = 51`
- `OPERATION_CODE_DUPLICATE_PROVIDER = 52`
- `OPERATION_CODE_DYNAMIC_CUDA_ADAPTER_UNSUPPORTED = 53`
- `OPERATION_CODE_DYNAMIC_CUDA_DECODE_UNSUPPORTED = 54`
- `OPERATION_CODE_DYNAMIC_DOMAIN_OUT_OF_MEMORY = 55`
- `OPERATION_CODE_DYNAMIC_DOMAIN_RESERVATION_FAILED = 56`
- `OPERATION_CODE_EXECUTION_EVIDENCE_UNAVAILABLE = 57`
- `OPERATION_CODE_EXECUTION_FAILED = 58`
- `OPERATION_CODE_EXPORT_FAILED = 59`
- `OPERATION_CODE_EXPORT_MISMATCH = 60`
- `OPERATION_CODE_FETCH_UNAVAILABLE = 61`
- `OPERATION_CODE_GRAPH_INITIALIZATION_FAILED = 62`
- `OPERATION_CODE_GRAPH_LOWERING_FAILED = 63`
- `OPERATION_CODE_GRAPH_NOT_READABLE = 64`
- `OPERATION_CODE_HANDLE_DISPOSED = 65`
- `OPERATION_CODE_INPUT_BINDING_FAILED = 66`
- `OPERATION_CODE_INPUT_NOT_FOUND = 67`
- `OPERATION_CODE_INPUT_QUANTIZATION_INTROSPECTION_UNSUPPORTED = 68`
- `OPERATION_CODE_INPUT_QUERY_FAILED = 69`
- `OPERATION_CODE_INTERNAL = 70`
- `OPERATION_CODE_INVALID_ADAPTER_REVISION = 71`
- `OPERATION_CODE_INVALID_ADAPTER_SOURCE = 72`
- `OPERATION_CODE_INVALID_AFFINE_QUANTIZATION_REQUEST = 73`
- `OPERATION_CODE_INVALID_ARGUMENT = 74`
- `OPERATION_CODE_INVALID_AUTHORING_REVISION = 75`
- `OPERATION_CODE_INVALID_BANK_RESIDENCY = 76`
- `OPERATION_CODE_INVALID_CALIBRATION_SAMPLE = 77`
- `OPERATION_CODE_INVALID_CHECKPOINT = 78`
- `OPERATION_CODE_INVALID_COMMIT = 79`
- `OPERATION_CODE_INVALID_CONTEXT = 80`
- `OPERATION_CODE_INVALID_CONTEXT_OPTIONS = 81`
- `OPERATION_CODE_INVALID_DECODE_DEPENDENCY = 82`
- `OPERATION_CODE_INVALID_DECODE_FEEDBACK = 83`
- `OPERATION_CODE_INVALID_DECODE_INPUT = 84`
- `OPERATION_CODE_INVALID_DECODE_POSITION = 85`
- `OPERATION_CODE_INVALID_EXPORT = 86`
- `OPERATION_CODE_INVALID_GRAPH = 87`
- `OPERATION_CODE_INVALID_GRAPH_CONTRACT = 88`
- `OPERATION_CODE_INVALID_GRAPH_PATH = 89`
- `OPERATION_CODE_INVALID_INPUT_BATCH = 90`
- `OPERATION_CODE_INVALID_INPUT_QUERY = 91`
- `OPERATION_CODE_INVALID_INPUT_VALUES = 92`
- `OPERATION_CODE_INVALID_LOGICAL_DOMAIN = 93`
- `OPERATION_CODE_INVALID_MODEL_SOURCE = 94`
- `OPERATION_CODE_INVALID_MODEL_SOURCE_RESOLUTION = 95`
- `OPERATION_CODE_INVALID_OPTIONS = 96`
- `OPERATION_CODE_INVALID_OUTPUT_INFO = 97`
- `OPERATION_CODE_INVALID_POLICY = 98`
- `OPERATION_CODE_INVALID_PREFIX_ROW_COUNT = 99`
- `OPERATION_CODE_INVALID_PROVIDER = 100`
- `OPERATION_CODE_INVALID_PTQ_COVERAGE_QUERY = 101`
- `OPERATION_CODE_INVALID_PTQ_INFO = 102`
- `OPERATION_CODE_INVALID_PTQ_OUTPUT_PATHS = 103`
- `OPERATION_CODE_INVALID_PTQ_PLAN = 104`
- `OPERATION_CODE_INVALID_PTQ_PROFILE_QUERY = 105`
- `OPERATION_CODE_INVALID_PTQ_SPEC = 106`
- `OPERATION_CODE_INVALID_PTQ_TENSOR_QUERY = 107`
- `OPERATION_CODE_INVALID_READ = 108`
- `OPERATION_CODE_INVALID_REQUEST = 109`
- `OPERATION_CODE_INVALID_REQUEST_INFO = 110`
- `OPERATION_CODE_INVALID_RESULT_POINTER = 111`
- `OPERATION_CODE_INVALID_REVISION_INFO = 112`
- `OPERATION_CODE_INVALID_RUN = 113`
- `OPERATION_CODE_INVALID_RUNTIME = 114`
- `OPERATION_CODE_INVALID_SHAPE_OPTIONS = 115`
- `OPERATION_CODE_INVALID_SUBMISSION = 116`
- `OPERATION_CODE_INVALID_TEMPLATE_GRAPH = 117`
- `OPERATION_CODE_INVALID_TENSOR_SPEC = 118`
- `OPERATION_CODE_INVALID_TRAINER = 119`
- `OPERATION_CODE_INVALID_TRAINER_OPTIONS = 120`
- `OPERATION_CODE_INVALID_TRAINING_BACKEND = 121`
- `OPERATION_CODE_INVALID_TRAINING_ENGINE = 122`
- `OPERATION_CODE_INVALID_TRAIN_STEP = 123`
- `OPERATION_CODE_INVALID_WEIGHT_REVISION = 124`
- `OPERATION_CODE_IO_ERROR = 125`
- `OPERATION_CODE_LOGICAL_GRAPH_INVALID = 126`
- `OPERATION_CODE_MODEL_SOURCE_RESOLUTION_FAILED = 127`
- `OPERATION_CODE_MUTEX_INIT_FAILED = 128`
- `OPERATION_CODE_NOT_FOUND = 129`
- `OPERATION_CODE_NO_APPLIED_UPDATE = 130`
- `OPERATION_CODE_OPERATOR_FALLBACK_FORBIDDEN = 131`
- `OPERATION_CODE_OUTPUT_NOT_FOUND = 132`
- `OPERATION_CODE_OUT_OF_MEMORY = 133`
- `OPERATION_CODE_OVERLOADED = 134`
- `OPERATION_CODE_PACKAGE_BODY_UNAVAILABLE = 135`
- `OPERATION_CODE_PACKAGE_FETCH_FAILED = 136`
- `OPERATION_CODE_PACKAGE_READ_FAILED = 137`
- `OPERATION_CODE_PACKAGE_TOO_LARGE = 138`
- `OPERATION_CODE_PENDING = 139`
- `OPERATION_CODE_PRIVATE_VFS_MOUNT_FAILED = 140`
- `OPERATION_CODE_PROVIDER_ALREADY_REGISTERED = 141`
- `OPERATION_CODE_PROVIDER_RUNTIME_CREATE_FAILED = 142`
- `OPERATION_CODE_PTQ_COVERAGE_BUFFER_TOO_SMALL = 143`
- `OPERATION_CODE_PTQ_COVERAGE_QUERY_FAILED = 144`
- `OPERATION_CODE_PTQ_INSPECT_FAILED = 145`
- `OPERATION_CODE_PTQ_PACKAGE_WRITE_FAILED = 146`
- `OPERATION_CODE_PTQ_PLAN_CREATE_FAILED = 147`
- `OPERATION_CODE_PTQ_PROFILE_COVERAGE_REQUIRED = 148`
- `OPERATION_CODE_PTQ_PROFILE_NOT_FOUND = 149`
- `OPERATION_CODE_PTQ_PROFILE_QUERY_FAILED = 150`
- `OPERATION_CODE_PTQ_SINGLE_SHARD_REQUIRED = 151`
- `OPERATION_CODE_PTQ_TENSOR_NOT_FOUND = 152`
- `OPERATION_CODE_PTQ_TENSOR_QUERY_FAILED = 153`
- `OPERATION_CODE_PTQ_WRITE_FAILED = 154`
- `OPERATION_CODE_QUEUE_FULL = 155`
- `OPERATION_CODE_QUEUE_INIT_FAILED = 156`
- `OPERATION_CODE_REQUEST_INIT_FAILED = 157`
- `OPERATION_CODE_RESULT_BUDGET_INVALID = 158`
- `OPERATION_CODE_RESULT_DISPOSED = 159`
- `OPERATION_CODE_RESULT_PENDING = 160`
- `OPERATION_CODE_REVISION_CONFLICT = 161`
- `OPERATION_CODE_REVISION_MISSING = 162`
- `OPERATION_CODE_REVISION_OVERFLOW = 163`
- `OPERATION_CODE_ROLLBACK_FAILED = 164`
- `OPERATION_CODE_ROUTE_LEASE_FAILED = 165`
- `OPERATION_CODE_SCHEDULED_MODE_DISABLED = 166`
- `OPERATION_CODE_SCHEDULER_CLOCK_UNAVAILABLE = 167`
- `OPERATION_CODE_SHAPE_DOMAIN_ATTESTATION_INVALID = 168`
- `OPERATION_CODE_SHAPE_DOMAIN_UNSUPPORTED = 169`
- `OPERATION_CODE_STATE_ALLOCATION_FAILED = 170`
- `OPERATION_CODE_STATE_ALLOCATION_OVERFLOW = 171`
- `OPERATION_CODE_SUPERSEDED = 172`
- `OPERATION_CODE_TEMPLATE_SNAPSHOT_FAILED = 173`
- `OPERATION_CODE_TRAINER_CREATE_FAILED = 174`
- `OPERATION_CODE_TRAINER_POISONED = 175`
- `OPERATION_CODE_TRAINER_STATE_FAILED = 176`
- `OPERATION_CODE_TRAINING_BACKEND_UNAVAILABLE = 177`
- `OPERATION_CODE_TRAINING_BACKEND_UNSUPPORTED = 178`
- `OPERATION_CODE_TRAINING_GRAPH_UNSUPPORTED = 179`
- `OPERATION_CODE_TRAINING_PREFLIGHT_FAILED = 180`
- `OPERATION_CODE_TRAIN_STEP_FAILED = 181`
- `OPERATION_CODE_TRAIN_STEP_NOT_FOUND = 182`
- `OPERATION_CODE_TRANSPORT_UNSUPPORTED = 183`
- `OPERATION_CODE_WAIT_FAILED = 184`
- `OPERATION_CODE_WEIGHTS_NOT_READABLE = 185`
- `OPERATION_CODE_WEIGHT_STORE_INVALID = 186`
- `OPERATION_CODE_WEIGHT_STORE_MISSING = 187`
- `OPERATION_CODE_WORKSPACE_ALLOCATION_FAILED = 188`

## volvoxai.v1.OperationStage



- `OPERATION_STAGE_NONE = 0`
- `OPERATION_STAGE_RUNTIME_CREATE = 1`
- `OPERATION_STAGE_MODEL_LOAD = 2`
- `OPERATION_STAGE_COMPILE = 3`
- `OPERATION_STAGE_CONTEXT_CREATE = 4`
- `OPERATION_STAGE_INPUT = 5`
- `OPERATION_STAGE_EXECUTE = 6`
- `OPERATION_STAGE_READBACK = 7`
- `OPERATION_STAGE_CLOSE = 8`
- `OPERATION_STAGE_DECODE = 9`
- `OPERATION_STAGE_ADAPTER = 10`
- `OPERATION_STAGE_TRAINER_CREATE = 11`
- `OPERATION_STAGE_TRAINER_INPUT = 12`
- `OPERATION_STAGE_TRAINER_STEP = 13`
- `OPERATION_STAGE_TRAINER_COMMIT = 14`
- `OPERATION_STAGE_TRAINER_ROLLBACK = 15`
- `OPERATION_STAGE_TRAINER_EXPORT = 16`
- `OPERATION_STAGE_PTQ_CREATE = 17`
- `OPERATION_STAGE_PTQ_CALIBRATE = 18`
- `OPERATION_STAGE_PTQ_INSPECT = 19`
- `OPERATION_STAGE_PTQ_WRITE = 20`
- `OPERATION_STAGE_PTQ_CLOSE = 21`
- `OPERATION_STAGE_SUBMIT = 22`
- `OPERATION_STAGE_ADMISSION = 23`
- `OPERATION_STAGE_PTQ_AUTHOR = 24`: Deriving the quantized template, which happens before a plan exists and
so cannot report itself as PTQ_CREATE.
- `OPERATION_STAGE_GRAPH_PLAN_CREATE = 25`
- `OPERATION_STAGE_GRAPH_PLAN_GET = 26`
- `OPERATION_STAGE_GRAPH_PLAN_RESOLVE = 27`
- `OPERATION_STAGE_GRAPH_PLAN_CLOSE = 28`
- `OPERATION_STAGE_SAFETENSORS_INSPECT = 29`
- `OPERATION_STAGE_TOKENIZER_CREATE = 30`
- `OPERATION_STAGE_TOKENIZE = 31`
- `OPERATION_STAGE_DETOKENIZE = 32`
- `OPERATION_STAGE_GRAPH_PLAN_EDIT = 33`
- `OPERATION_STAGE_GRAPH_PLAN_EXPORT = 34`
- `OPERATION_STAGE_SAFETENSORS_READ = 35`
- `OPERATION_STAGE_SAFETENSORS_WRITE = 36`
- `OPERATION_STAGE_TENSOR_INITIALIZE = 37`
- `OPERATION_STAGE_LORA_AUTHOR = 38`
- `OPERATION_STAGE_WEIGHT_QUANTIZE = 39`
- `OPERATION_STAGE_WEIGHT_DEQUANTIZE = 40`

## volvoxai.v1.NativeResourceKind

Resource representation is independent of physical placement and mapping.

- `NATIVE_RESOURCE_KIND_UNSPECIFIED = 0`
- `NATIVE_RESOURCE_KIND_HOST = 1`
- `NATIVE_RESOURCE_KIND_CUDA = 2`
- `NATIVE_RESOURCE_KIND_VULKAN = 3`
- `NATIVE_RESOURCE_KIND_OPENGL = 4`
- `NATIVE_RESOURCE_KIND_METAL = 5`

## volvoxai.v1.BufferAccessMode



- `BUFFER_ACCESS_MODE_READ = 0`
- `BUFFER_ACCESS_MODE_WRITE = 1`

## volvoxai.v1.DimensionKind

One axis of a logical bounded tensor. A FIXED axis has an empty symbol,
min == max, and multiple_of == 1. A SYMBOLIC axis has a non-empty canonical
symbol and positive min/max/multiple_of values. Mixed states are rejected.

Zero is deliberately UNSPECIFIED rather than FIXED so a zero-initialized
descriptor fails validation instead of silently reading as a fixed axis of
extent zero.

- `DIMENSION_KIND_UNSPECIFIED = 0`
- `DIMENSION_KIND_FIXED = 1`
- `DIMENSION_KIND_SYMBOLIC = 2`

## volvoxai.v1.CandidateOutcome



- `CANDIDATE_OUTCOME_UNSPECIFIED = 0`
- `CANDIDATE_OUTCOME_SELECTED = 1`
- `CANDIDATE_OUTCOME_UNAVAILABLE = 2`
- `CANDIDATE_OUTCOME_DOMAIN_UNSUPPORTED = 3`
- `CANDIDATE_OUTCOME_COMPILE_FAILED = 4`
- `CANDIDATE_OUTCOME_OPERATOR_UNSUPPORTED = 5`
- `CANDIDATE_OUTCOME_NOT_EVALUATED = 6`

## volvoxai.v1.ShapePlanOrigin



- `SHAPE_PLAN_ORIGIN_UNSPECIFIED = 0`
- `SHAPE_PLAN_ORIGIN_CACHE_HIT = 1`
- `SHAPE_PLAN_ORIGIN_COLD = 2`

## volvoxai.v1.DecodeMode



- `DECODE_MODE_UNSPECIFIED = 0`
- `DECODE_MODE_DISABLED = 1`
- `DECODE_MODE_PROVIDER = 2`
- `DECODE_MODE_ROW = 3`
- `DECODE_MODE_DEPENDENCY = 4`

## volvoxai.v1.InputValidationCode



- `INPUT_VALIDATION_CODE_UNSPECIFIED = 0`
- `INPUT_VALIDATION_CODE_MISSING_INPUT = 1`
- `INPUT_VALIDATION_CODE_UNKNOWN_INPUT = 2`
- `INPUT_VALIDATION_CODE_DUPLICATE_INPUT = 3`
- `INPUT_VALIDATION_CODE_INVALID_NAME = 4`
- `INPUT_VALIDATION_CODE_DTYPE_MISMATCH = 5`
- `INPUT_VALIDATION_CODE_RANK_MISMATCH = 6`
- `INPUT_VALIDATION_CODE_DIMENSION_OUT_OF_RANGE = 7`
- `INPUT_VALIDATION_CODE_SYMBOL_MISMATCH = 8`
- `INPUT_VALIDATION_CODE_BYTE_SIZE_MISMATCH = 9`
- `INPUT_VALIDATION_CODE_PAYLOAD_REQUIRED = 10`
- `INPUT_VALIDATION_CODE_LOCATION_UNSUPPORTED = 11`
- `INPUT_VALIDATION_CODE_TRANSPORT_UNSUPPORTED = 12`
- `INPUT_VALIDATION_CODE_DECODE_SHAPE_MISMATCH = 13`
- `INPUT_VALIDATION_CODE_SIZE_OVERFLOW = 14`
- `INPUT_VALIDATION_CODE_INVALID_BUFFER_VIEW = 15`

## volvoxai.v1.ApiEffect



- `API_EFFECT_UNSPECIFIED = 0`
- `API_EFFECT_READ_ONLY = 1`
- `API_EFFECT_CREATE = 2`
- `API_EFFECT_RELEASE = 3`
- `API_EFFECT_MUTATE = 4`
- `API_EFFECT_EXECUTE = 5`

## volvoxai.v1.ApiRuleKind



- `API_RULE_KIND_UNSPECIFIED = 0`
- `API_RULE_KIND_EXACTLY_ONE = 1`
- `API_RULE_KIND_REQUIRES = 2`
- `API_RULE_KIND_LIVE_HANDLE = 3`
- `API_RULE_KIND_SCHEDULED_RUNTIME = 4`
- `API_RULE_KIND_PREFILLED_CONTEXT = 5`
- `API_RULE_KIND_COMPLETE_INPUT_BATCH = 6`
- `API_RULE_KIND_IN_PROCESS_ONLY = 7`

## volvoxai.v1.DecodeCachePolicy



- `DECODE_CACHE_POLICY_PAGED = 0`
- `DECODE_CACHE_POLICY_CONTIGUOUS = 1`

## volvoxai.v1.RequestFreshness



- `REQUEST_FRESHNESS_ALL = 0`: Every admitted request remains eligible until cancelled or completed.
- `REQUEST_FRESHNESS_LATEST = 1`: A request with the same compiled route and nonzero stream key is
superseded by a newer admitted LATEST request. Queued replacement is
transactional: equal payload storage may be reused across dynamic shapes,
while a larger payload requires temporary byte-budget headroom. Already
submitted work remains physical but its result is discarded at
completion.
- `REQUEST_FRESHNESS_DROP_IF_LATE = 2`: Refuse or retire queued/completed work whose monotonic deadline elapsed.

## volvoxai.v1.RequestState



- `REQUEST_STATE_UNSPECIFIED = 0`
- `REQUEST_STATE_QUEUED = 1`
- `REQUEST_STATE_RUNNING = 2`
- `REQUEST_STATE_SUCCEEDED = 3`
- `REQUEST_STATE_CANCELLED = 4`
- `REQUEST_STATE_FAILED = 5`
- `REQUEST_STATE_SUPERSEDED = 6`

## volvoxai.v1.GraphPlanSourceKind



- `GRAPH_PLAN_SOURCE_KIND_UNSPECIFIED = 0`
- `GRAPH_PLAN_SOURCE_KIND_MODEL = 1`
- `GRAPH_PLAN_SOURCE_KIND_STANDALONE_GRAPH = 2`

## volvoxai.v1.GraphTensorKind



- `GRAPH_TENSOR_KIND_UNSPECIFIED = 0`
- `GRAPH_TENSOR_KIND_INPUT = 1`
- `GRAPH_TENSOR_KIND_WEIGHT = 2`
- `GRAPH_TENSOR_KIND_VALUE = 3`

## volvoxai.v1.ShapeDiagnosticCode



- `SHAPE_DIAGNOSTIC_CODE_UNSPECIFIED = 0`
- `SHAPE_DIAGNOSTIC_CODE_INVALID_GRAPH = 1`
- `SHAPE_DIAGNOSTIC_CODE_INPUT_BINDING_FAILED = 2`
- `SHAPE_DIAGNOSTIC_CODE_UNBOUND_SYMBOL = 3`
- `SHAPE_DIAGNOSTIC_CODE_UNKNOWN_OPERATOR = 4`
- `SHAPE_DIAGNOSTIC_CODE_OPERATOR_INFERENCE_FAILED = 5`
- `SHAPE_DIAGNOSTIC_CODE_OPERATOR_DOMAIN_UNSUPPORTED = 6`
- `SHAPE_DIAGNOSTIC_CODE_INPUT_PORT_MISMATCH = 7`
- `SHAPE_DIAGNOSTIC_CODE_OUTPUT_PORT_MISMATCH = 8`
- `SHAPE_DIAGNOSTIC_CODE_OUTPUT_DTYPE_MISMATCH = 9`
- `SHAPE_DIAGNOSTIC_CODE_OUTPUT_SHAPE_MISMATCH = 10`
- `SHAPE_DIAGNOSTIC_CODE_SYMBOL_CONFLICT = 11`
- `SHAPE_DIAGNOSTIC_CODE_BOUND_VIOLATION = 12`
- `SHAPE_DIAGNOSTIC_CODE_MULTIPLE_OF_VIOLATION = 13`
- `SHAPE_DIAGNOSTIC_CODE_QUANTIZATION_METADATA_MISSING = 14`
- `SHAPE_DIAGNOSTIC_CODE_QUANTIZATION_METADATA_UNEXPECTED = 15`
- `SHAPE_DIAGNOSTIC_CODE_QUANTIZATION_MISMATCH = 16`
- `SHAPE_DIAGNOSTIC_CODE_ARITHMETIC_OVERFLOW = 17`

## volvoxai.v1.ShapeDomainProofKind



- `SHAPE_DOMAIN_PROOF_KIND_UNSPECIFIED = 0`
- `SHAPE_DOMAIN_PROOF_KIND_SINGLETON_EXHAUSTIVE = 1`
- `SHAPE_DOMAIN_PROOF_KIND_BOUNDED_SYMBOLIC = 2`

## volvoxai.v1.ShapeDomainFact



- `SHAPE_DOMAIN_FACT_UNSPECIFIED = 0`
- `SHAPE_DOMAIN_FACT_SINGLETON_PUBLIC_DOMAIN = 1`
- `SHAPE_DOMAIN_FACT_CONCRETE_NODE_ACCEPTED = 2`
- `SHAPE_DOMAIN_FACT_BOUNDED_SYMBOLIC_DOMAIN = 3`
- `SHAPE_DOMAIN_FACT_SYMBOLIC_NODE_ACCEPTED = 4`
- `SHAPE_DOMAIN_FACT_DIRECT_PRESERVE = 5`
- `SHAPE_DOMAIN_FACT_EXACT_BINARY = 6`
- `SHAPE_DOMAIN_FACT_BROADCAST = 7`
- `SHAPE_DOMAIN_FACT_STRUCTURAL = 8`
- `SHAPE_DOMAIN_FACT_AFFINE = 9`
- `SHAPE_DOMAIN_FACT_DIRECT_PROJECT = 10`

## volvoxai.v1.IndependentBatchRefusalReason



- `INDEPENDENT_BATCH_REFUSAL_REASON_UNSPECIFIED = 0`
- `INDEPENDENT_BATCH_REFUSAL_REASON_PUBLIC_AXIS_INVALID = 1`
- `INDEPENDENT_BATCH_REFUSAL_REASON_DIMENSION_DOMAIN = 2`
- `INDEPENDENT_BATCH_REFUSAL_REASON_WEIGHT_DEPENDENCY = 3`
- `INDEPENDENT_BATCH_REFUSAL_REASON_REPEATED_AXIS = 4`
- `INDEPENDENT_BATCH_REFUSAL_REASON_QUANTIZATION_DEPENDENCY = 5`
- `INDEPENDENT_BATCH_REFUSAL_REASON_AXIS_MAPPING = 6`
- `INDEPENDENT_BATCH_REFUSAL_REASON_CROSS_LANE_REDUCTION = 7`
- `INDEPENDENT_BATCH_REFUSAL_REASON_CROSS_LANE_OPERATOR = 8`
- `INDEPENDENT_BATCH_REFUSAL_REASON_VALUE_DEPENDENT_CARDINALITY = 9`
- `INDEPENDENT_BATCH_REFUSAL_REASON_OPERATOR_UNSUPPORTED = 10`

## volvoxai.v1.CallIntentKind



- `CALL_INTENT_KIND_UNSPECIFIED = 0`
- `CALL_INTENT_KIND_FORWARD = 1`: Execute the complete graph; retain no activation across calls.
- `CALL_INTENT_KIND_FIXED_INPUT_DEPENDENCY = 2`: Recompute the exact dependency closure of changed_inputs and report the
non-weight tensors that selected work must read or preserve across calls.
- `CALL_INTENT_KIND_ALL_CROSS_CALL_LIVE = 3`: Execute the complete graph while treating every non-weight tensor as
cross-call live.

## volvoxai.v1.NodeSelection



- `NODE_SELECTION_UNSPECIFIED = 0`
- `NODE_SELECTION_ALL = 1`
- `NODE_SELECTION_EXPLICIT = 2`

## volvoxai.v1.SafetensorsDiagnosticCode



- `SAFETENSORS_DIAGNOSTIC_CODE_UNSPECIFIED = 0`
- `SAFETENSORS_DIAGNOSTIC_CODE_INVALID_PREFIX = 1`
- `SAFETENSORS_DIAGNOSTIC_CODE_LENGTH_MISMATCH = 2`
- `SAFETENSORS_DIAGNOSTIC_CODE_INVALID_UTF8 = 3`
- `SAFETENSORS_DIAGNOSTIC_CODE_INVALID_JSON = 4`
- `SAFETENSORS_DIAGNOSTIC_CODE_DUPLICATE_KEY = 5`
- `SAFETENSORS_DIAGNOSTIC_CODE_ROOT_NOT_OBJECT = 6`
- `SAFETENSORS_DIAGNOSTIC_CODE_INVALID_METADATA = 7`
- `SAFETENSORS_DIAGNOSTIC_CODE_UNKNOWN_DTYPE = 8`
- `SAFETENSORS_DIAGNOSTIC_CODE_INVALID_TENSOR = 9`
- `SAFETENSORS_DIAGNOSTIC_CODE_INVALID_SHAPE = 10`
- `SAFETENSORS_DIAGNOSTIC_CODE_NON_BYTE_ALIGNED = 11`
- `SAFETENSORS_DIAGNOSTIC_CODE_INVALID_OFFSETS = 12`
- `SAFETENSORS_DIAGNOSTIC_CODE_BYTE_LENGTH_MISMATCH = 13`
- `SAFETENSORS_DIAGNOSTIC_CODE_INVALID_COVERAGE = 14`
- `SAFETENSORS_DIAGNOSTIC_CODE_INTEGER_OVERFLOW = 15`
- `SAFETENSORS_DIAGNOSTIC_CODE_DEPTH_LIMIT = 16`
- `SAFETENSORS_DIAGNOSTIC_CODE_RESERVED_NONZERO = 17`
- `SAFETENSORS_DIAGNOSTIC_CODE_INTERNAL = 18`

## volvoxai.v1.SafetensorsDiagnosticSection



- `SAFETENSORS_DIAGNOSTIC_SECTION_UNSPECIFIED = 0`
- `SAFETENSORS_DIAGNOSTIC_SECTION_PREFIX = 1`
- `SAFETENSORS_DIAGNOSTIC_SECTION_JSON = 2`
- `SAFETENSORS_DIAGNOSTIC_SECTION_ROOT = 3`
- `SAFETENSORS_DIAGNOSTIC_SECTION_METADATA = 4`
- `SAFETENSORS_DIAGNOSTIC_SECTION_TENSOR = 5`
- `SAFETENSORS_DIAGNOSTIC_SECTION_COVERAGE = 6`

## volvoxai.v1.ParameterExportMode

CPU shared read views retain the trainer and block trainer mutation until
all handles and access leases are released. GPU shared views are rejected;
native GPU parameter snapshots remain on the backend. Snapshots do not block
mutation. names must be a nonempty list of model weight tensor names.

- `PARAMETER_EXPORT_MODE_SNAPSHOT = 0`
- `PARAMETER_EXPORT_MODE_SHARED_READ = 1`

## volvoxai.v1.MemorySpace



- `MEMORY_SPACE_UNSPECIFIED = 0`
- `MEMORY_SPACE_HOST = 1`
- `MEMORY_SPACE_JS_HEAP = 2`
- `MEMORY_SPACE_JS_EXTERNAL = 3`
- `MEMORY_SPACE_JS_ARRAY_BUFFER = 4`
- `MEMORY_SPACE_WASM_LINEAR = 5`
- `MEMORY_SPACE_NATIVE_HEAP = 6`
- `MEMORY_SPACE_MAPPED_FILE = 7`
- `MEMORY_SPACE_DEVICE = 8`: An API-visible device allocation whose physical placement is unknown.
- `MEMORY_SPACE_DEVICE_LOCAL = 9`
- `MEMORY_SPACE_HOST_VISIBLE_DEVICE = 10`
- `MEMORY_SPACE_UNIFIED = 11`

## volvoxai.v1.MemoryOwnerKind

The lifetime owner is not necessarily the object currently using a
resource. In particular, contexts can consume one immutable
COMPILED_MODEL-owned weight allocation without owning copies of it.

- `MEMORY_OWNER_KIND_UNSPECIFIED = 0`
- `MEMORY_OWNER_KIND_PROCESS = 1`
- `MEMORY_OWNER_KIND_RUNTIME = 2`
- `MEMORY_OWNER_KIND_MODEL = 3`
- `MEMORY_OWNER_KIND_COMPILED_MODEL = 4`
- `MEMORY_OWNER_KIND_EXECUTION_CONTEXT = 5`
- `MEMORY_OWNER_KIND_RESULT = 6`
- `MEMORY_OWNER_KIND_BACKEND_SHARED = 7`
- `MEMORY_OWNER_KIND_TRAINER = 8`: Full-profile private authoring owners. Inference projections omit these
names while the shared protobuf wire vocabulary retains them.
- `MEMORY_OWNER_KIND_PTQ_PLAN = 9`
- `MEMORY_OWNER_KIND_GRAPH_PLAN = 10`: Logical planning runs in every profile; VxPlanningService, which publishes
plan handles, is full-only.

## volvoxai.v1.MemoryResourceRole



- `MEMORY_RESOURCE_ROLE_UNSPECIFIED = 0`
- `MEMORY_RESOURCE_ROLE_WEIGHTS = 1`
- `MEMORY_RESOURCE_ROLE_PACKED_WEIGHTS = 2`
- `MEMORY_RESOURCE_ROLE_INPUT = 3`
- `MEMORY_RESOURCE_ROLE_ACTIVATION = 4`
- `MEMORY_RESOURCE_ROLE_KV_CACHE = 5`
- `MEMORY_RESOURCE_ROLE_SCRATCH = 6`
- `MEMORY_RESOURCE_ROLE_RESULT_SNAPSHOT = 7`
- `MEMORY_RESOURCE_ROLE_ARENA = 8`
- `MEMORY_RESOURCE_ROLE_PLAN_CACHE = 9`
- `MEMORY_RESOURCE_ROLE_KERNEL_CACHE = 10`
- `MEMORY_RESOURCE_ROLE_METADATA = 11`
- `MEMORY_RESOURCE_ROLE_RUNTIME_OVERHEAD = 12`
- `MEMORY_RESOURCE_ROLE_ALLOCATOR_OVERHEAD = 13`
- `MEMORY_RESOURCE_ROLE_DRIVER_OVERHEAD = 14`
- `MEMORY_RESOURCE_ROLE_OTHER = 15`
- `MEMORY_RESOURCE_ROLE_TRAINING_WORKING_WEIGHTS = 16`: Full-profile private authoring state. These are distinct from immutable
published WEIGHTS and ordinary inference ACTIVATION/SCRATCH resources.
- `MEMORY_RESOURCE_ROLE_TRAINING_GRADIENTS = 17`
- `MEMORY_RESOURCE_ROLE_TRAINING_OPTIMIZER_SLOTS = 18`
- `MEMORY_RESOURCE_ROLE_TRAINING_ACCUMULATION = 19`
- `MEMORY_RESOURCE_ROLE_PTQ_OBSERVER_STATE = 20`

## volvoxai.v1.MemoryBackingRelation

A resource without a backing resource is INDEPENDENT. ALIAS and
SUBALLOCATION both refer to a range of backing_resource_id and add no bytes
on top of that backing allocation. SUBALLOCATION ranges may be summed only
after proving that they are disjoint; ALIAS ranges may overlap.

- `MEMORY_BACKING_RELATION_UNSPECIFIED = 0`
- `MEMORY_BACKING_RELATION_INDEPENDENT = 1`
- `MEMORY_BACKING_RELATION_ALIAS = 2`
- `MEMORY_BACKING_RELATION_SUBALLOCATION = 3`

## volvoxai.v1.MemoryBoundKind

Each bound kind is an independent number. Bounds of different kinds are
never summed. A decode bound, for example, is an alternative to the
ordinary resident bound rather than an additional allocation.

- `MEMORY_BOUND_KIND_UNSPECIFIED = 0`
- `MEMORY_BOUND_KIND_MAXIMUM_TENSOR = 1`
- `MEMORY_BOUND_KIND_CONTEXT_CAPACITY = 2`
- `MEMORY_BOUND_KIND_ORDINARY_RESIDENT = 3`
- `MEMORY_BOUND_KIND_DECODE_RESIDENT = 4`

## volvoxai.v1.MemoryMetric

Resource measurements describe allocator or API accounting. Proved maxima
and external resident envelopes have separate messages so a consumer cannot
accidentally add them to current allocations.

- `MEMORY_METRIC_UNSPECIFIED = 0`
- `MEMORY_METRIC_LOGICAL = 1`: Exact payload required by the logical request, excluding reusable slack.
- `MEMORY_METRIC_LIVE = 2`: Allocations not yet released at this snapshot.
- `MEMORY_METRIC_RESERVED = 3`: Bytes reserved from a backing allocator, including unusable slack.
- `MEMORY_METRIC_CAPACITY = 4`: Reusable payload capacity exposed by the resource.
- `MEMORY_METRIC_LOGICAL_HIGH_WATER = 5`: MemoryTemporalCoverage names the epoch represented by high-water values.
- `MEMORY_METRIC_LIVE_HIGH_WATER = 6`
- `MEMORY_METRIC_RESERVED_HIGH_WATER = 7`
- `MEMORY_METRIC_CAPACITY_HIGH_WATER = 8`
- `MEMORY_METRIC_CUMULATIVE_ALLOCATED = 9`

## volvoxai.v1.MemoryEvidenceSource



- `MEMORY_EVIDENCE_SOURCE_UNSPECIFIED = 0`
- `MEMORY_EVIDENCE_SOURCE_ALLOCATOR_COUNTER = 1`
- `MEMORY_EVIDENCE_SOURCE_RUNTIME_COUNTER = 2`
- `MEMORY_EVIDENCE_SOURCE_API_REQUEST = 3`
- `MEMORY_EVIDENCE_SOURCE_OS_SAMPLER = 4`
- `MEMORY_EVIDENCE_SOURCE_DRIVER_SAMPLER = 5`

## volvoxai.v1.MemoryValueRelation

Relation between the reported number and the quantity it names. REQUESTED
is exact API-requested storage but makes no claim about physical residency.
UNAVAILABLE carries no MemoryByteSize; every other relation does.

- `MEMORY_VALUE_RELATION_UNSPECIFIED = 0`
- `MEMORY_VALUE_RELATION_EXACT = 1`
- `MEMORY_VALUE_RELATION_UPPER_BOUND = 2`
- `MEMORY_VALUE_RELATION_LOWER_BOUND = 3`
- `MEMORY_VALUE_RELATION_REQUESTED = 4`
- `MEMORY_VALUE_RELATION_ESTIMATED = 5`
- `MEMORY_VALUE_RELATION_UNAVAILABLE = 6`

## volvoxai.v1.MemoryTemporalCoverage

Temporal coverage is independent from value relation and acquisition
source. For example, one RSS sample is EXACT at an INSTANT, while the
maximum of periodic RSS samples is only a LOWER_BOUND over SAMPLED_WINDOW.
SAMPLED_WINDOW always names that observed maximum, never a mean, last
sample, integral, or other statistic.

- `MEMORY_TEMPORAL_COVERAGE_UNSPECIFIED = 0`
- `MEMORY_TEMPORAL_COVERAGE_INSTANT = 1`
- `MEMORY_TEMPORAL_COVERAGE_OPERATION_WINDOW = 2`
- `MEMORY_TEMPORAL_COVERAGE_RESOURCE_LIFETIME = 3`
- `MEMORY_TEMPORAL_COVERAGE_CAPTURE_WINDOW = 4`
- `MEMORY_TEMPORAL_COVERAGE_PROCESS_LIFETIME = 5`
- `MEMORY_TEMPORAL_COVERAGE_SAMPLED_WINDOW = 6`

## volvoxai.v1.MemoryEnvelopeKind

Envelopes overlap resource accounting and can overlap one another. For
example RSS can contain JS heap, ArrayBuffers, WASM linear memory, mapped
host-visible device memory, and driver allocations. Envelope values are
compared as separate series and are never added to resources or each other.

- `MEMORY_ENVELOPE_KIND_UNSPECIFIED = 0`
- `MEMORY_ENVELOPE_KIND_PROCESS_RSS = 1`
- `MEMORY_ENVELOPE_KIND_PROCESS_PEAK_RSS = 2`
- `MEMORY_ENVELOPE_KIND_PROCESS_PSS = 3`
- `MEMORY_ENVELOPE_KIND_PROCESS_PRIVATE_BYTES = 4`
- `MEMORY_ENVELOPE_KIND_PROCESS_MANAGED_HEAP_USED = 5`
- `MEMORY_ENVELOPE_KIND_PROCESS_EXTERNAL_BYTES = 6`
- `MEMORY_ENVELOPE_KIND_PROCESS_ARRAY_BUFFER_BYTES = 7`
- `MEMORY_ENVELOPE_KIND_DEVICE_PROCESS_USED = 8`
- `MEMORY_ENVELOPE_KIND_DEVICE_TOTAL_USED = 9`
- `MEMORY_ENVELOPE_KIND_DEVICE_TOTAL_CAPACITY = 10`

## volvoxai.v1.MemoryInventoryKind

Whether a snapshot's resources are merely the records available to one
collector or the complete live inventory for its subject/backend/device
scope. Only COMPLETE inventories may prove physical totals or disjoint
active SUBALLOCATION ranges. An empty COMPLETE inventory means exact zero;
an empty PARTIAL inventory means that no resource records were observed.

- `MEMORY_INVENTORY_KIND_UNSPECIFIED = 0`
- `MEMORY_INVENTORY_KIND_PARTIAL = 1`
- `MEMORY_INVENTORY_KIND_COMPLETE = 2`

## volvoxai.v1.TraceDetail



- `TRACE_DETAIL_BASIC = 0`: Host operations and, when device timing is enabled, existing device passes.
- `TRACE_DETAIL_NODES = 1`: Also collect executable nodes, training work and supported GPU programs.
Actual pass/node/program coverage is reported in the result.

## volvoxai.v1.TracePhase

The work that actually executed, independently of its host/device clock.
UNSPECIFIED includes outer operations and observations with no known phase.

- `TRACE_PHASE_UNSPECIFIED = 0`
- `TRACE_PHASE_FORWARD = 1`
- `TRACE_PHASE_LOSS = 2`
- `TRACE_PHASE_BACKWARD = 3`
- `TRACE_PHASE_GRADIENT = 4`: Initialization, accumulation, norm and clipping.
- `TRACE_PHASE_OPTIMIZER = 5`

## volvoxai.v1.TraceState



- `TRACE_STATE_COLLECTING = 0`
- `TRACE_STATE_DRAINING = 1`
- `TRACE_STATE_READY = 2`

## volvoxai.v1.TraceSupport

Observed adapter support, not inferred from the number of collected events.

- `TRACE_SUPPORT_UNOBSERVED = 0`
- `TRACE_SUPPORT_AVAILABLE = 1`
- `TRACE_SUPPORT_UNAVAILABLE = 2`
- `TRACE_SUPPORT_MIXED = 3`: The same backend encountered devices/passes with different support.

## volvoxai.v1.TraceClockMethod



- `TRACE_CLOCK_METHOD_UNSPECIFIED = 0`
- `TRACE_CLOCK_METHOD_CALIBRATED = 1`: A device clock sample is related to host time with measured uncertainty.
Mapping is local to this pass; it is not a permanent clock conversion.
- `TRACE_CLOCK_METHOD_BOUNDED = 2`: Causal submission/completion bounds only, not clock calibration.

## volvoxai.v1.TraceActivity



- `TRACE_ACTIVITY_WORK = 0`
- `TRACE_ACTIVITY_COPY = 1`
- `TRACE_ACTIVITY_SUBMIT = 2`
- `TRACE_ACTIVITY_WAIT = 3`: Host blocking call, including scheduling overhead.
- `TRACE_ACTIVITY_AWAIT = 4`: Asynchronous completion, not CPU blocking time.

## volvoxai.v1.TraceMemoryAction

EXISTING introduces an allocation at the first inventory of its owner.
It does not claim that allocation happened during this trace. Identities are
opaque and unique within the trace, including after allocator address reuse.

- `TRACE_MEMORY_ACTION_EXISTING = 0`
- `TRACE_MEMORY_ACTION_ALLOCATE = 1`
- `TRACE_MEMORY_ACTION_FREE = 2`
