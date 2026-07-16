// Browser-only, WASM-backend-only entry for deployments that package one JS
// module and the adjacent volvoxai.full.wasm sidecar. It supports inference
// plus strict WASM training without Node filesystem, CPU, WebNN, WebGPU, WGSL,
// or the duplicate JavaScript PTQ implementation. Generic PTQ math and narrow
// F32-master weight synchronization both reuse the full sidecar's C routines.
export * from './core/Tensor.js';
export type * from './types.js';
export * from './backends/BackendEngine.js';
export * from './backends/DecodeSession.js';
export * from './backends/WasmEngine.js';
export * from './core/GraphLoader.js';
export * from './core/Safetensors.js';
export * from './core/AdapterManager.js';
export * from './core/Tokenizer.js';
export * from './training/TrainingGraph.js';
export * from './training/TrainingModelBuilder.js';
export * from './training/WasmAutograd.js';
export * from './training/WasmPTQ.js';
export * from './training/WasmQuantizedLoRATrainer.js';
export * from './training/GradientAccumulation.js';
export * from './training/ModelCheckpoint.js';
export * from './training/TrainingLosses.js';
export * from './training/TrainingOptimizer.js';
export type * from './training/TrainingStep.js';
export { TrainingGraph as Graph } from './training/TrainingGraph.js';
export { TrainingModelBuilder as ModelBuilder } from './training/TrainingModelBuilder.js';
export { WasmVolvoxAI, WasmVolvoxAI as VolvoxAI } from './WasmVolvoxAI.js';
