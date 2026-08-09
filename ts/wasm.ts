// Browser-only, WASM-backend-only entry for deployments that package one JS
// module and the adjacent volvoxai.full.wasm sidecar. It supports inference
// plus strict WASM training without Node filesystem, CPU, WebNN, WebGPU, WGSL,
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
