// Full training surface. The inference entry never imports this module.
//
// Autograd backends:
//   CPUAutograd     reference forward/backward pass in plain JavaScript.
//   WasmAutograd    strict strategy over AcceleratedAutograd: supplies the
//                   complete WasmTrainingKernels numerical contract and never
//                   imports or falls back to CPUAutograd/CPUEngine.
//   WebGPUAutograd  intentionally a separate driver: WebGPU records one deferred
//                   command buffer with GPU-resident gradients, which does not
//                   fit the CPU driver's immediate, JS-array grad map. Shared,
//                   backend-independent pieces (optimizer/update-mode validation,
//                   gradient accumulation, losses) live in the modules below.
export * from './TrainingVolvoxAI.js';
export * from './TrainingGraph.js';
export * from './TrainingModelBuilder.js';
export * from './Initializers.js';
export * from './CPUAutograd.js';
export * from './WebGPUAutograd.js';
export * from './WasmAutograd.js';
export * from './WasmPTQ.js';
export * from './WasmQuantizedLoRATrainer.js';
export * from './GradientAccumulation.js';
export * from './ModelCheckpoint.js';
export * from './TrainingLosses.js';
export * from './TrainingOptimizer.js';
export * from './Quantization.js';
