// Browser-only, WASM-backend-only entry for deployments that package one JS
// module and the adjacent volvoxai.full.wasm sidecar. It supports inference
// plus strict WASM training without Node filesystem, CPU, WebGPU, WGSL,
// or the duplicate JavaScript PTQ implementation. Generic PTQ math and narrow
// F32-master weight synchronization both reuse the full sidecar's C routines.
export type {
  RuntimeDType,
  RuntimeTypedArray,
  PerTensorQuantization,
  PerAxisQuantization,
  TensorQuantization,
  ExecutionInputs,
  ExecutionOptions,
  DecodeExecutionOptions,
} from './types.js';
export {
  BACKEND_MEMORY_SNAPSHOT_PROTOCOL,
  MEMORY_CAPTURE_PROTOCOL,
  MEMORY_EVIDENCE_FORMAT,
  MEMORY_EVIDENCE_PROOF_PROTOCOL,
  MEMORY_EVIDENCE_RESOURCE_PROTOCOL,
} from './core/MemoryCapture.js';
export type {
  MemoryCaptureOptions,
  BackendMemoryCaptureRequest,
  BackendMemorySnapshot,
  RuntimeMemoryByteSize,
  RuntimeMemoryMonotonicTime,
  RuntimeMemoryOwnerRef,
  RuntimeMemoryBoundTerm,
  RuntimeMemoryPeakCase,
  RuntimeMemoryBoundProof,
  RuntimeMemoryDomainAttestation,
  RuntimeMemoryMeasurement,
  RuntimeMemoryResourceEvidence,
  RuntimeMemoryEnvelopeEvidence,
  RuntimeMemorySnapshot,
  RuntimeMemoryEvidence,
} from './core/MemoryCapture.js';
export {
  MemorySpace,
  MemoryOwnerKind,
  MemoryResourceRole,
  MemoryBackingRelation,
  MemoryBoundKind,
  MemorySnapshotPoint,
  MemoryMetric,
  MemoryEvidenceSource,
  MemoryValueRelation,
  MemoryTemporalCoverage,
  MemoryEnvelopeKind,
  MemoryInventoryKind,
  OperationStage,
  ExecutionMode,
  executionModes,
} from './generated/volvoxaiEnums.js';
export type { ExecutionModeValue } from './generated/volvoxaiEnums.js';
export * from './core/Graph.js';
export * from './core/ModelLoader.js';
export * from './core/ModelBuilder.js';
export * from './core/Model.js';
export * from './core/ResolvedShapePlan.js';
export * from './ops/shapeSystem.js';
export {
  Runtime,
  CompiledModel,
  ExecutionContext,
} from './core/ContextRuntime.js';
export type {
  BackendPolicy,
  RuntimeOptions,
  ModelCompileOptions,
  ExecutionContextOptions,
  CompilationCandidateReport,
  CompilationReport,
  ExecutionFailureReport,
  RuntimeDiagnostic,
  ExecutionContextDecode,
} from './core/ContextRuntime.js';
export {
  RuntimeRequestHandle,
} from './core/RuntimeScheduler.js';
export type {
  RequestFreshness,
  RuntimeRequestState,
  RuntimeSchedulerOptions,
  RuntimeResultBudgetOptions,
  RuntimeExecutionConfiguration,
  RuntimeRunOptions,
  RuntimeSubmitOptions,
} from './core/RuntimeScheduler.js';
export {
  ExecutionResult,
  TensorResult,
} from './core/ExecutionResult.js';
export type {
  BackendHostTensorSnapshot,
  BackendDeviceTensorSnapshot,
  BackendTensorSnapshot,
  BackendExecutionSnapshot,
  OperatorRouteEvidence,
  ExecutionRouteEvidence,
  ExecutionDecodeState,
  ExecutionReport,
} from './core/ExecutionResult.js';
export { VolvoxAIError } from './core/RuntimeErrors.js';
export type {
  VolvoxAIErrorCode,
  RuntimeFailurePhase,
  VolvoxAIErrorOptions,
} from './core/RuntimeErrors.js';
export {
  SAFETENSORS_DTYPE_INFO,
  SAFETENSORS_TENSOR_READABLE,
  SAFETENSORS_TENSOR_WRITABLE,
  SAFETENSORS_OPEN_READ_ONLY,
  SAFETENSORS_OPEN_READ_WRITE,
  SafetensorsTensor,
  SafetensorsFile,
} from './core/Safetensors.js';
export type {
  SafetensorsDType,
  SafetensorsMetadata,
  SafetensorsTensorInfo,
  SafetensorsFileOptions,
  SafetensorsOpenOptions,
} from './core/Safetensors.js';
export * from './core/Tokenizer.js';
export { Trainer } from './training/WasmTrainer.js';
export type { TrainerOptions } from './training/WasmTrainer.js';
export {
  PTQ,
  PTQObserver,
  createPTQ,
} from './training/WasmPTQ.js';
export type {
  PTQDType,
  PTQScheme,
  PTQRange,
  PTQParameterOptions,
  PTQParameters,
  PTQQuantized,
  PTQWeightOptions,
  PTQPackedWeight,
  PTQOptions,
} from './training/WasmPTQ.js';
export * from './training/ModelCheckpoint.js';
export type {
  CrossEntropyLossInput,
  CrossEntropyTrainingOptions,
  CrossEntropyLossDescriptor,
} from './training/TrainingLosses.js';
export type {
  TrainingUpdateMode,
  TrainingOptimizerOptions,
  TrainingOptimizerValues,
  TrainingOptimizerDescriptor,
} from './training/TrainingOptimizer.js';
export type * from './training/TrainingStep.js';
export * from './WasmProfile.js';
