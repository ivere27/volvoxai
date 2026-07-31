import {
  VolvoxAI,
  exportModelCheckpoint,
  importModelCheckpoint,
} from '../../ts/full.js';

/**
 * Test harness that exercises the public Runtime -> Model -> Trainer ownership
 * chain while allowing parity suites to create many independent graphs.
 */
export class TrainingSessionHarness {
  #backend;
  #wasmUrl;
  #runtimePromise;
  #records = new Map();
  #closed = false;

  constructor({ backend = 'cpu', wasmUrl } = {}) {
    this.#backend = backend;
    this.#wasmUrl = wasmUrl;
    this.#runtimePromise = VolvoxAI.createRuntime({
      backends: [backend],
      ...(wasmUrl == null ? {} : { wasmUrl }),
    });
  }

  async #record(graph) {
    if (this.#closed) throw new Error('TrainingSessionHarness is closed.');
    let record = this.#records.get(graph);
    if (record) return record;
    const runtime = await this.#runtimePromise;
    const initialCheckpoint = exportModelCheckpoint(graph);
    const model = runtime.createModel(importModelCheckpoint(initialCheckpoint).graph);
    let trainer;
    try {
      trainer = await VolvoxAI.createTrainer(model, {
        backend: this.#backend,
        checkpoint: initialCheckpoint,
        ...(this.#wasmUrl == null ? {} : { wasmUrl: this.#wasmUrl }),
      });
    } catch (error) {
      await model.close();
      throw error;
    }
    record = { model, trainer };
    this.#records.set(graph, record);
    return record;
  }

  async runStep(graph, options = {}) {
    const { backend: _ignoredBackend, ...stepOptions } = options;
    const { trainer } = await this.#record(graph);
    return trainer.trainStep(stepOptions);
  }

  async exportCheckpoint(graph, options = {}) {
    const { trainer } = await this.#record(graph);
    return trainer.exportCheckpoint(options);
  }

  async close() {
    if (this.#closed) return;
    this.#closed = true;
    const records = [...this.#records.values()];
    await Promise.allSettled(records.map(({ trainer }) => trainer.close()));
    await Promise.allSettled(records.map(({ model }) => model.close()));
    const runtime = await this.#runtimePromise;
    await runtime.close();
    this.#records.clear();
  }
}

export function createCPUTrainingHarness() {
  return new TrainingSessionHarness({ backend: 'cpu' });
}

export function createWasmTrainingHarness(wasmUrl) {
  return new TrainingSessionHarness({ backend: 'wasm', wasmUrl });
}

/** Create a one-step runner that deterministically closes every public handle. */
export function createWasmStepRunner(wasmUrl) {
  return async (graph, options = {}) => {
    const training = createWasmTrainingHarness(wasmUrl);
    try {
      const result = await training.runStep(graph, options);
      const checkpoint = await training.exportCheckpoint(graph);
      return {
        ...result,
        publishedGraph: importModelCheckpoint(checkpoint).graph,
      };
    } finally {
      await training.close();
    }
  };
}
