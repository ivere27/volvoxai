import { WasmEngine } from './backends/WasmEngine.js';
import { InferenceRuntime } from './core/InferenceRuntime.js';
import type { Graph } from './core/Graph.js';
import { TrainingGraph } from './training/TrainingGraph.js';
import { TrainingModelBuilder } from './training/TrainingModelBuilder.js';
import { WasmAutograd } from './training/WasmAutograd.js';
import {
  WasmQuantizedLoRATrainer,
  type WasmQuantizedLoRATrainerOptions,
} from './training/WasmQuantizedLoRATrainer.js';
import {
  createWasmPTQ as createWasmPTQInstance,
  type WasmPTQ,
} from './training/WasmPTQ.js';
import {
  exportModelCheckpoint,
  importModelCheckpoint,
  type ModelCheckpoint,
  type ModelCheckpointExportOptions,
} from './training/ModelCheckpoint.js';
import {
  getGradientAccumulationState,
  resetGradientAccumulation,
} from './training/GradientAccumulation.js';
import { resolveTrainingOptimizer } from './training/TrainingOptimizer.js';
import type { TrainingStepOptions } from './training/TrainingStep.js';
import type { BackendSelection } from './types.js';

const WASM_ONLY_BACKENDS = Object.freeze(['wasm']);

export interface WasmOnlyTrainStepOptions extends TrainingStepOptions {
  backend?: 'wasm';
  executor?: WasmEngine | null;
}

export interface WasmOnlyTrainerOptions {
  executor?: WasmEngine | null;
}

export interface WasmOnlyQuantizedLoRATrainerOptions extends WasmQuantizedLoRATrainerOptions {
  executor?: WasmEngine | null;
}

export interface WasmOnlyPTQOptions {
  executor?: WasmEngine | null;
}

function assertWasmSelection(selection: BackendSelection): void {
  const requested = typeof selection === 'string' ? [selection] : [...selection];
  if (requested.length === 0 || requested.some((backend) => backend !== 'wasm' && backend !== 'auto')) {
    throw new Error(
      `[VolvoxAI] The WASM-only entry supports only 'wasm'; received ${JSON.stringify(selection)}.`,
    );
  }
}

/** Inference and training APIs composed with exactly one selectable backend: WASM. */
export class WasmVolvoxAI extends InferenceRuntime {
  static listBackends(): readonly string[] {
    return WASM_ONLY_BACKENDS;
  }

  static async init(
    preferredBackend: BackendSelection = 'wasm',
    wasmUrl: string | URL = new URL('./volvoxai.full.wasm', import.meta.url),
  ): Promise<WasmVolvoxAI> {
    assertWasmSelection(preferredBackend);
    const engine = await WasmEngine.init(wasmUrl);
    if (!engine) {
      throw new Error('[VolvoxAI] WASM-only initialization failed.');
    }
    const instance = new this();
    instance.engines.push({ type: 'wasm', engine });
    console.log('[VolvoxAI] WASM Engine Initialized successfully.');
    return instance;
  }

  createGraph(): TrainingGraph {
    return new TrainingGraph();
  }

  createModel(graph: TrainingGraph = this.createGraph()): TrainingModelBuilder {
    return new TrainingModelBuilder(graph);
  }

  exportCheckpoint(graph: Graph, options: ModelCheckpointExportOptions = {}) {
    return exportModelCheckpoint(graph, options);
  }

  importCheckpoint(checkpoint: ModelCheckpoint) {
    return importModelCheckpoint(checkpoint);
  }

  async trainLoRAStep(graph: Graph, options: WasmOnlyTrainStepOptions = {}) {
    if (options.backend != null && options.backend !== 'wasm') {
      throw new Error(`[VolvoxAI] The WASM-only entry cannot train with '${options.backend}'.`);
    }
    const resolved = resolveTrainingOptimizer(
      graph as TrainingGraph,
      options.updateMode,
      options.optimizer,
    );
    const {
      backend: _backend,
      executor,
      ...trainOptions
    } = {
      ...options,
      updateMode: resolved.updateMode,
      optimizer: resolved.optimizer,
    };
    const trainer = await this.createWasmTrainer(graph, { executor });
    try {
      return await trainer.trainStep(trainOptions);
    } finally {
      trainer.dispose();
    }
  }

  async trainStep(graph: Graph, options: WasmOnlyTrainStepOptions = {}) {
    return this.trainLoRAStep(graph, options);
  }

  resetGradientAccumulation(graph: Graph): void {
    resetGradientAccumulation(graph);
  }

  getGradientAccumulationState(graph: Graph) {
    return getGradientAccumulationState(graph);
  }

  async createWasmTrainer(
    graph: Graph,
    { executor = null }: WasmOnlyTrainerOptions = {},
  ): Promise<WasmAutograd> {
    const initialized = this.engines.find((entry) => entry.type === 'wasm')?.engine as
      WasmEngine | undefined;
    const selected = executor || initialized;
    if (!selected) {
      throw new Error('[VolvoxAI] No full-profile WASM engine is initialized for training.');
    }
    return WasmAutograd.create(selected, graph);
  }

  async createWasmPTQ(
    { executor = null }: WasmOnlyPTQOptions = {},
  ): Promise<WasmPTQ> {
    const initialized = this.engines.find((entry) => entry.type === 'wasm')?.engine as
      WasmEngine | undefined;
    const selected = executor || initialized;
    if (!selected) {
      throw new Error('[VolvoxAI] No full-profile WASM engine is initialized for PTQ.');
    }
    return createWasmPTQInstance(selected);
  }

  async createPTQ(options: WasmOnlyPTQOptions = {}): Promise<WasmPTQ> {
    return this.createWasmPTQ(options);
  }

  async createQuantizedLoRATrainer(
    trainingGraph: Graph,
    inferenceGraph: Graph,
    {
      executor = null,
      bindings,
    }: WasmOnlyQuantizedLoRATrainerOptions,
  ): Promise<WasmQuantizedLoRATrainer> {
    const initialized = this.engines.find((entry) => entry.type === 'wasm')?.engine as
      WasmEngine | undefined;
    const selected = executor || initialized;
    if (!selected) {
      throw new Error('[VolvoxAI] No full-profile WASM engine is initialized for quantized LoRA training.');
    }
    return WasmQuantizedLoRATrainer.create(
      selected,
      trainingGraph,
      inferenceGraph,
      { bindings },
    );
  }
}
