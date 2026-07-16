// Full monolithic entry: the familiar VolvoxAI name resolves to the training
// subclass, while TrainingVolvoxAI remains available as an explicit name.
export * from './index.js';
export * from './training/index.js';
export { TrainingVolvoxAI as VolvoxAI } from './training/TrainingVolvoxAI.js';
export { TrainingGraph as Graph } from './training/TrainingGraph.js';
export { TrainingModelBuilder as ModelBuilder } from './training/TrainingModelBuilder.js';
