import { WasmEngine } from '../backends/WasmEngine.js';
import { acceleratedTrainStep } from './AcceleratedAutograd.js';
import { WasmTrainingKernels } from './WasmTrainingKernels.js';
import type { Graph } from '../core/Graph.js';
import type { TrainingStepOptions } from './TrainingStep.js';

/** Strict full-profile WASM trainer; explicit WASM requests never fall back. */
export class WasmAutograd {
  readonly graph: Graph;
  readonly engine: WasmEngine;
  kernels: WasmTrainingKernels | null;
  private _training: boolean;
  private _disposed: boolean;

  static async create(wasmEngine: WasmEngine, graph: Graph): Promise<WasmAutograd> {
    if (!(wasmEngine instanceof WasmEngine)) {
      throw new Error('WasmAutograd requires an initialized WasmEngine.');
    }
    if (!graph) throw new Error('WasmAutograd requires a graph.');
    const kernels = await WasmTrainingKernels.create(wasmEngine);
    await kernels.preflight(graph);
    return new WasmAutograd(wasmEngine, graph, kernels);
  }

  constructor(wasmEngine: WasmEngine, graph: Graph, kernels: WasmTrainingKernels) {
    this.engine = wasmEngine;
    this.graph = graph;
    this.kernels = kernels;
    this._training = false;
    this._disposed = false;
  }

  async trainStep(options: TrainingStepOptions = {}) {
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

  dispose() {
    if (this._training) throw new Error('Cannot dispose WasmAutograd during a trainStep.');
    this._disposed = true;
    this.kernels = null;
  }
}
