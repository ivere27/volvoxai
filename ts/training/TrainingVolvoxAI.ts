import { VolvoxAI } from '../VolvoxAI.js';
import type { Graph } from '../core/Graph.js';
import type { WasmEngine } from '../backends/WasmEngine.js';
import type { GraphExecutor } from '../backends/GraphExecutor.js';
import { TrainingModelBuilder } from './TrainingModelBuilder.js';
import { TrainingGraph } from './TrainingGraph.js';
import {
  exportModelCheckpoint,
  importModelCheckpoint,
  type ModelCheckpoint,
  type ModelCheckpointExportOptions,
} from './ModelCheckpoint.js';
import { CPUAutograd } from './CPUAutograd.js';
import type { CPUTrainStepOptions } from './CPUAutograd.js';
import { WebGPUAutograd } from './WebGPUAutograd.js';
import { WasmAutograd } from './WasmAutograd.js';
import {
  getGradientAccumulationState,
  resetGradientAccumulation,
} from './GradientAccumulation.js';
import { resolveTrainingOptimizer } from './TrainingOptimizer.js';

export interface WebGPUTrainingExecutor {
  backendName?: string;
  device?: GPUDevice;
  executor?: WebGPUTrainingExecutor;
  prepareForTraining?(): void | Promise<void>;
  [name: string]: any;
}

export interface TrainingTrainStepOptions extends CPUTrainStepOptions {
  backend?: 'cpu' | 'wasm' | 'webgpu' | (string & {});
  executor?: WebGPUTrainingExecutor | WasmEngine | null;
  device?: GPUDevice | null;
}

export interface WebGPUTrainerOptions {
  executor?: WebGPUTrainingExecutor | null;
  device?: GPUDevice | null;
}

export interface WasmTrainerOptions {
  executor?: WasmEngine | null;
}

/** Inference API plus opt-in training, checkpoint, and accumulation methods. */
export class TrainingVolvoxAI extends VolvoxAI {
  static async init(
    preferredBackend = 'auto',
    wasmUrl = new URL('./volvoxai.full.wasm', import.meta.url),
  ) {
    return super.init(preferredBackend, wasmUrl);
  }

  createGraph() {
    return new TrainingGraph();
  }

  createModel(graph: TrainingGraph = this.createGraph()) {
    return new TrainingModelBuilder(graph);
  }

  exportCheckpoint(graph: Graph, options: ModelCheckpointExportOptions = {}) {
    return exportModelCheckpoint(graph, options);
  }

  importCheckpoint(checkpoint: ModelCheckpoint) {
    return importModelCheckpoint(checkpoint);
  }
  async trainLoRAStep(graph: Graph, options: TrainingTrainStepOptions = {}) {
    const resolved = resolveTrainingOptimizer(graph as TrainingGraph, options.updateMode, options.optimizer);
    const effectiveOptions = {
      ...options,
      updateMode: resolved.updateMode,
      optimizer: resolved.optimizer,
    };
    if (options.backend === 'webgpu') {
      const { backend: _backend, executor, device, ...trainOptions } = effectiveOptions;
      const trainer = await this.createWebGPUTrainer(graph, { executor, device });
      try {
        return await trainer.trainStep(trainOptions);
      } finally {
        trainer.dispose();
      }
    }
    if (options.backend === 'wasm') {
      const { backend: _backend, executor, ...trainOptions } = effectiveOptions;
      const trainer = await this.createWasmTrainer(graph, { executor: executor as WasmEngine | null });
      try {
        return await trainer.trainStep(trainOptions);
      } finally {
        trainer.dispose();
      }
    }
    if (options.backend != null && options.backend !== 'cpu') {
      throw new Error(`[VolvoxAI] Unsupported training backend '${options.backend}'.`);
    }
    return CPUAutograd.trainStep(graph, effectiveOptions);
  }

  async trainStep(graph: Graph, options: TrainingTrainStepOptions = {}) {
    return this.trainLoRAStep(graph, options);
  }

  async resetGradientAccumulation(graph: Graph) {
    resetGradientAccumulation(graph);
  }

  async getGradientAccumulationState(graph: Graph) {
    return getGradientAccumulationState(graph);
  }

  async createWebGPUTrainer(
    graph: Graph,
    { executor = null, device = null }: WebGPUTrainerOptions = {},
  ) {
    const selectedDevice = device || executor?.device ||
      this.engines.find((entry) => entry.type === 'webgpu')?.device as GPUDevice | undefined;
    if (!selectedDevice) throw new Error('[VolvoxAI] No WebGPU device is available for training.');
    // The public WebGPUEngine composes the lower-level GraphExecutor. Training
    // keeps consuming that executor directly so inference owns no training API,
    // but the wrapper must first invalidate its decode owner and mirror the
    // inner executor's training compilation state.
    if (executor?.backendName === 'webgpu' && executor.executor) {
      if (typeof executor.prepareForTraining !== 'function') {
        throw new Error('[VolvoxAI] WebGPUEngine does not expose prepareForTraining().');
      }
      await executor.prepareForTraining();
    }
    const selectedExecutor = executor?.backendName === 'webgpu' && executor.executor
      ? executor.executor
      : executor;
    return new WebGPUAutograd(selectedDevice, graph, selectedExecutor as GraphExecutor | null);
  }

  async createWasmTrainer(graph: Graph, { executor = null }: WasmTrainerOptions = {}) {
    const initialized = this.engines.find((entry) => entry.type === 'wasm')?.engine as WasmEngine | undefined;
    const selected = executor || initialized;
    if (!selected) {
      throw new Error('[VolvoxAI] No full-profile WASM engine is initialized for training.');
    }
    return WasmAutograd.create(selected, graph);
  }
}
