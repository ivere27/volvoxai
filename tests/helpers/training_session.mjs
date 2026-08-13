import { Trainer } from '../../ts/training/Trainer.js';
import { exportModelCheckpoint, importModelCheckpoint } from '../../ts/training/ModelCheckpoint.js';
import {
  logicalSnapshotFromTrainingGraph,
  checkpointForTrainingGraph,
  shapedFixtureInputs,
  trainingGraphFromCheckpoint,
} from './training_fixture.mjs';

/**
 * Internal numerical-test harness. It converts each legacy fixed concrete
 * kernel fixture once at the boundary; the Trainer itself sees only an
 * immutable Model and complete shaped inputs.
 */
export class TrainingSessionHarness {
  #backend;
  #wasmUrl;
  #records = new Map();
  #closed = false;

  constructor({ backend = 'cpu-js', wasmUrl } = {}) {
    this.#backend = backend;
    this.#wasmUrl = wasmUrl;
  }

  async #record(graph) {
    if (this.#closed) throw new Error('TrainingSessionHarness is closed.');
    let record = this.#records.get(graph);
    if (record) return record;
    const checkpoint = checkpointForTrainingGraph(graph);
    const snapshot = checkpoint === null
      ? logicalSnapshotFromTrainingGraph(graph)
      : importModelCheckpoint(checkpoint).snapshot;
    const initialCheckpoint = checkpoint ?? (graph.trainingMetadata == null
      ? null
      : {
        ...exportModelCheckpoint(snapshot),
        trainingMetadata: structuredClone(graph.trainingMetadata),
      });
    const trainer = await Trainer.create(snapshot, {
      backend: this.#backend,
      ...(initialCheckpoint === null ? {} : { checkpoint: initialCheckpoint }),
      ...(this.#wasmUrl == null ? {} : { wasmUrl: this.#wasmUrl }),
    });
    record = { snapshot, trainer };
    this.#records.set(graph, record);
    return record;
  }

  async runStep(graph, options = {}) {
    const { backend: _ignoredBackend, inputs, ...stepOptions } = options;
    const { snapshot, trainer } = await this.#record(graph);
    return trainer.trainStep({
      ...stepOptions,
      inputs: shapedFixtureInputs(snapshot, inputs),
    });
  }

  async exportCheckpoint(graph, options = {}) {
    const { trainer } = await this.#record(graph);
    return trainer.exportCheckpoint(options);
  }

  async close() {
    if (this.#closed) return;
    this.#closed = true;
    await Promise.allSettled([...this.#records.values()].map(({ trainer }) => trainer.close()));
    this.#records.clear();
  }
}

export function createCPUTrainingHarness() {
  return new TrainingSessionHarness({ backend: 'cpu-js' });
}

export function createWasmTrainingHarness(wasmUrl) {
  return new TrainingSessionHarness({ backend: 'wasm', wasmUrl });
}

/** Create a one-step runner that deterministically closes every Trainer. */
export function createWasmStepRunner(wasmUrl) {
  return async (graph, options = {}) => {
    const training = createWasmTrainingHarness(wasmUrl);
    try {
      const result = await training.runStep(graph, options);
      const checkpoint = await training.exportCheckpoint(graph);
      return {
        ...result,
        publishedGraph: trainingGraphFromCheckpoint(checkpoint),
      };
    } finally {
      await training.close();
    }
  };
}

export { trainingGraphFromCheckpoint } from './training_fixture.mjs';
