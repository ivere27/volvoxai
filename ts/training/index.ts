// Deliberate full-profile public training surface. Backend drivers, mutable
// graph-state helpers, optimizer plumbing, and accumulation storage stay
// implementation details behind Trainer.
export * from './Trainer.js';
export type {
  LoRALinearOptions,
  LoRALinearResult,
} from './TrainingModelBuilder.js';
export * from './Initializers.js';
export * from './ModelCheckpoint.js';
export type {
  CrossEntropyLossInput,
  CrossEntropyTrainingOptions,
  CrossEntropyLossDescriptor,
} from './TrainingLosses.js';
export type {
  TrainingUpdateMode,
  TrainingOptimizerOptions,
  TrainingOptimizerValues,
  TrainingOptimizerDescriptor,
} from './TrainingOptimizer.js';
export type * from './TrainingStep.js';
export * from './Quantization.js';
