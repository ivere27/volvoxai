import { WasmEngine } from '../backends/WasmEngine.js';
import { acceleratedTrainStep } from './AcceleratedAutograd.js';
import { WasmTrainingKernels } from './WasmTrainingKernels.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import type { TrainingKernelStepOptions } from './TrainingStep.js';
import type {
  BackendPlanCacheInspection,
  BackendPlanCacheOptions,
} from './BackendPlanCache.js';

/** Strict full-profile WASM trainer; explicit WASM requests never fall back. */
export class WasmAutograd {
  graph: RuntimeGraph;
  readonly engine: WasmEngine;
  kernels: WasmTrainingKernels | null;
  private _training: boolean;
  private _disposed: boolean;

  static async create(
    wasmEngine: WasmEngine,
    graph: RuntimeGraph,
    cacheOptions: BackendPlanCacheOptions = {},
  ): Promise<WasmAutograd> {
    if (!(wasmEngine instanceof WasmEngine)) {
      throw new Error('WasmAutograd requires an initialized WasmEngine.');
    }
    if (!graph) throw new Error('WasmAutograd requires a graph.');
    const kernels = await WasmTrainingKernels.create(wasmEngine, cacheOptions);
    await kernels.preflight(graph);
    return new WasmAutograd(wasmEngine, graph, kernels);
  }

  constructor(wasmEngine: WasmEngine, graph: RuntimeGraph, kernels: WasmTrainingKernels) {
    this.engine = wasmEngine;
    this.graph = graph;
    this.kernels = kernels;
    this._training = false;
    this._disposed = false;
  }

  async trainStep(options: TrainingKernelStepOptions = {}) {
    if (this._disposed) throw new Error('WasmAutograd has been disposed.');
    if (this._training) throw new Error('WasmAutograd does not support concurrent trainStep calls.');
    const kernels = this.kernels;
    if (!kernels) throw new Error('WasmAutograd has been disposed.');
    this._training = true;
    try {
      const result = await acceleratedTrainStep(this.graph, options, kernels);
      // Inference compilation may retain packed F32 Linear weights. Rebuild
      // those copies immediately after an applied optimizer update so an
      // online correction is visible to the next inference call.
      if (result.updatedTensors.length > 0 && this.engine.graph === this.graph) {
        await this.engine.allocateGraph(this.graph);
      }
      return result;
    } finally {
      this._training = false;
    }
  }

  async rebind(
    graph: RuntimeGraph,
    binding: { readonly shapeSignature?: string; readonly tacticSignature?: string } = {},
  ) {
    if (this._disposed) throw new Error('WasmAutograd has been disposed.');
    if (this._training) throw new Error('WasmAutograd cannot rebind during a trainStep.');
    const kernels = this.kernels;
    if (!kernels) throw new Error('WasmAutograd has been disposed.');
    await kernels.preflight(graph, binding);
    this.graph = graph;
  }

  inspectPlanCache(): Readonly<BackendPlanCacheInspection> {
    const kernels = this.kernels;
    if (!kernels) throw new Error('WasmAutograd has been disposed.');
    return kernels.inspectPlanCache();
  }

  dispose() {
    if (this._training) throw new Error('Cannot dispose WasmAutograd during a trainStep.');
    this.kernels?.dispose();
    this._disposed = true;
    this.kernels = null;
  }
}
