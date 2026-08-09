import { WasmEngine } from '../backends/WasmEngine.js';
import { RuntimeGraph } from '../core/RuntimeGraph.js';
import { Tensor } from '../core/Tensor.js';
import type { PerAxisQuantization } from '../types.js';
import type { TrainingKernelStepOptions } from './TrainingStep.js';
import { ensureTrainingGraphState } from './TrainingGraph.js';
import { WasmAutograd } from './WasmAutograd.js';
import { resolveTrainingOptimizer } from './TrainingOptimizer.js';
import type {
  WasmQuantizedWeight,
  WasmWeightScalePolicy,
} from './WasmTrainingKernels.js';

const QUANTIZED_LINEAR_OPS = new Set(['QLinear', 'QMatMul', 'QGemm']);
const QUANTIZED_LORA_OWNER: unique symbol = Symbol('volvoxai.quantizedLoRAOwner');

type ReservableWasmEngine = WasmEngine & {
  [QUANTIZED_LORA_OWNER]?: boolean;
};

export interface WasmQuantizedLoRABinding {
  /** Persistent initialized F32 weight in the training graph. */
  master: string;
  /** Existing canonical OUT_IN I8 weight in the inference graph. */
  target: string;
  /** Map an IN_OUT master to the target without a JavaScript transpose copy. */
  transpose?: boolean;
  /** Recompute per-row scales by default, or retain the target's current scales. */
  scalePolicy?: WasmWeightScalePolicy;
}

export interface WasmQuantizedLoRATrainerOptions {
  bindings: readonly WasmQuantizedLoRABinding[];
}

export interface WasmQuantizedLoRAWeightReport {
  master: string;
  target: string;
  scalePolicy: WasmWeightScalePolicy;
  scales: Float32Array;
  saturationCount: number;
}

export interface WasmQuantizedLoRASyncResult {
  quantizedWeights: readonly WasmQuantizedLoRAWeightReport[];
  weightRevision: number;
}

interface StagedBinding {
  readonly masterName: string;
  readonly targetName: string;
  readonly master: Tensor & { buffer: Float32Array };
  readonly target: Tensor & { buffer: Int8Array; quantization: PerAxisQuantization };
  readonly transpose: boolean;
  readonly scalePolicy: WasmWeightScalePolicy;
  readonly rows: number;
  readonly columns: number;
}

interface PackedBinding {
  binding: StagedBinding;
  packed: WasmQuantizedWeight;
  quantization: PerAxisQuantization;
}

interface TargetSnapshot {
  binding: StagedBinding;
  buffer: Int8Array;
  quantization: PerAxisQuantization;
  sizeBytes: number;
}

type WasmTrainStepResult = Awaited<ReturnType<WasmAutograd['trainStep']>>;

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function stageBindings(
  trainingGraph: RuntimeGraph,
  inferenceGraph: RuntimeGraph,
  inputs: readonly WasmQuantizedLoRABinding[],
): StagedBinding[] {
  if (!(trainingGraph instanceof RuntimeGraph) || !(inferenceGraph instanceof RuntimeGraph)) {
    throw new Error('Quantized LoRA training requires VolvoxAI training and inference graphs.');
  }
  if (trainingGraph === inferenceGraph) {
    throw new Error('Quantized LoRA training requires separate F32 training and W8 inference graphs.');
  }
  if (!Array.isArray(inputs) || inputs.length === 0) {
    throw new Error('Quantized LoRA training requires at least one master-to-target binding.');
  }
  const masterNames = new Set<string>();
  const targetNames = new Set<string>();
  return inputs.map((input, index) => {
    if (!input || typeof input !== 'object' || Array.isArray(input)) {
      throw new Error(`Quantized LoRA binding ${index} must be an object.`);
    }
    const masterName = input.master;
    const targetName = input.target;
    if (typeof masterName !== 'string' || !masterName ||
        typeof targetName !== 'string' || !targetName) {
      throw new Error(`Quantized LoRA binding ${index} requires master and target names.`);
    }
    if (masterNames.has(masterName)) {
      throw new Error(`Quantized LoRA master '${masterName}' is bound more than once.`);
    }
    if (targetNames.has(targetName)) {
      throw new Error(`Quantized LoRA target '${targetName}' is bound more than once.`);
    }
    masterNames.add(masterName);
    targetNames.add(targetName);

    const master = trainingGraph.getTensor(masterName);
    if (!master?.isWeight || master.dtype !== 'float32' || master.shape.length !== 2 ||
        !(master.buffer instanceof Float32Array)) {
      throw new Error(`Quantized LoRA master '${masterName}' must be an initialized rank-2 F32 weight.`);
    }
    const target = inferenceGraph.getTensor(targetName);
    if (!target?.isWeight || target.dtype !== 'int8' || target.shape.length !== 2 ||
        !(target.buffer instanceof Int8Array)) {
      throw new Error(`Quantized LoRA target '${targetName}' must be an initialized rank-2 I8 weight.`);
    }
    const descriptor = target.quantization;
    if (!descriptor || descriptor.scheme !== 'per_axis' || descriptor.axis !== 0 ||
        descriptor.scales.length !== target.shape[0] ||
        descriptor.zero_points.length !== target.shape[0] ||
        descriptor.zero_points.some((value) => value !== 0)) {
      throw new Error(
        `Quantized LoRA target '${targetName}' requires symmetric axis-0 per-row quantization.`,
      );
    }
    const transpose = input.transpose === true;
    if (input.transpose != null && typeof input.transpose !== 'boolean') {
      throw new Error(`Quantized LoRA binding '${masterName}' transpose must be boolean.`);
    }
    const expectedMasterShape = transpose
      ? [target.shape[1], target.shape[0]]
      : [...target.shape];
    if (!sameShape(master.shape, expectedMasterShape)) {
      throw new Error(
        `Quantized LoRA master '${masterName}' shape [${master.shape}] does not match ` +
        `target '${targetName}' shape [${target.shape}]${transpose ? ' with transpose' : ''}.`,
      );
    }
    const scalePolicy = input.scalePolicy ?? 'recompute';
    if (scalePolicy !== 'recompute' && scalePolicy !== 'preserve') {
      throw new Error(`Quantized LoRA binding '${masterName}' has unsupported scale policy '${scalePolicy}'.`);
    }

    const consumers = inferenceGraph.nodes.filter((node) =>
      Object.values(node.inputs || {}).includes(target));
    if (consumers.length === 0) {
      throw new Error(`Quantized LoRA target '${targetName}' is not consumed by the inference graph.`);
    }
    for (const node of consumers) {
      if (!QUANTIZED_LINEAR_OPS.has(node.opType) || node.inputs?.weight !== target) {
        throw new Error(
          `Quantized LoRA target '${targetName}' may only be the weight input of QLinear/QMatMul/QGemm.`,
        );
      }
      const bias = node.inputs?.bias;
      if (!bias || bias.dtype !== 'int32' || !(bias.buffer instanceof Int32Array) ||
          bias.buffer.length !== target.shape[0] || bias.buffer.some((value) => value !== 0)) {
        throw new Error(
          `Quantized LoRA target '${targetName}' requires an initialized all-zero I32 bias ` +
          `for node '${String(node.id)}'.`,
        );
      }
    }

    return {
      masterName,
      targetName,
      master: master as StagedBinding['master'],
      target: target as StagedBinding['target'],
      transpose,
      scalePolicy,
      rows: target.shape[0],
      columns: target.shape[1],
    };
  });
}

/**
 * Train persistent F32 LoRA masters, then synchronize them into an existing
 * W8 inference graph without changing its topology or activation descriptors.
 *
 * This deliberately does not make a deep W8A8 graph differentiable. The
 * training graph must already provide the supported F32 backward path; the
 * separate W8 graph is an inference snapshot refreshed after applied updates.
 */
export class WasmQuantizedLoRATrainer {
  readonly trainingGraph: RuntimeGraph;
  readonly inferenceGraph: RuntimeGraph;
  readonly engine: WasmEngine;
  readonly trainer: WasmAutograd;
  readonly bindings: readonly Readonly<Required<WasmQuantizedLoRABinding>>[];
  readonly trainingTopologyRevision: number;
  readonly inferenceTopologyRevision: number;
  private _queue: Promise<void>;
  private _disposed: boolean;
  private _trained: boolean;
  private _initializedFromQuantized: boolean;
  private readonly _bindings: readonly StagedBinding[];

  static async create(
    engine: WasmEngine,
    trainingGraph: RuntimeGraph,
    inferenceGraph: RuntimeGraph,
    options: WasmQuantizedLoRATrainerOptions,
  ): Promise<WasmQuantizedLoRATrainer> {
    if (!(engine instanceof WasmEngine)) {
      throw new Error('WasmQuantizedLoRATrainer requires an initialized WasmEngine.');
    }
    // Stage and validate every binding before either graph or backend state is
    // mutated. WasmAutograd owns a separate scratch WASM instance.
    const bindings = stageBindings(trainingGraph, inferenceGraph, options?.bindings);
    const reservableEngine = engine as ReservableWasmEngine;
    if (reservableEngine[QUANTIZED_LORA_OWNER]) {
      throw new Error('This WasmEngine already belongs to a live quantized LoRA trainer.');
    }
    reservableEngine[QUANTIZED_LORA_OWNER] = true;
    let trainer: WasmAutograd | null = null;
    try {
      trainer = await WasmAutograd.create(engine, trainingGraph);
      if (!trainer.kernels) throw new Error('Quantized LoRA trainer initialization lost its WASM kernels.');
      trainer.kernels.requireQuantizedWeightSync();
      const current = engine.graph === inferenceGraph &&
        engine.compiledTopologyRevision === inferenceGraph.topologyRevision &&
        engine.compiledWeightRevision === inferenceGraph.weightRevision;
      if (!current) await engine.allocateGraph(inferenceGraph);
      return new WasmQuantizedLoRATrainer(
        engine, trainingGraph, inferenceGraph, trainer, bindings,
      );
    } catch (error) {
      trainer?.dispose();
      delete reservableEngine[QUANTIZED_LORA_OWNER];
      throw error;
    }
  }

  private constructor(
    engine: WasmEngine,
    trainingGraph: RuntimeGraph,
    inferenceGraph: RuntimeGraph,
    trainer: WasmAutograd,
    bindings: readonly StagedBinding[],
  ) {
    this.engine = engine;
    this.trainingGraph = trainingGraph;
    this.inferenceGraph = inferenceGraph;
    this.trainer = trainer;
    this.bindings = Object.freeze(bindings.map((binding) => Object.freeze({
      master: binding.masterName,
      target: binding.targetName,
      transpose: binding.transpose,
      scalePolicy: binding.scalePolicy,
    })));
    this._bindings = Object.freeze([...bindings]);
    this.trainingTopologyRevision = trainingGraph.topologyRevision;
    this.inferenceTopologyRevision = inferenceGraph.topologyRevision;
    this._queue = Promise.resolve();
    this._disposed = false;
    this._trained = false;
    this._initializedFromQuantized = false;
  }

  private _enqueue<T>(operation: () => Promise<T> | T): Promise<T> {
    if (this._disposed) return Promise.reject(new Error('WasmQuantizedLoRATrainer has been disposed.'));
    const result = this._queue.then(operation);
    this._queue = result.then(() => undefined, () => undefined);
    return result;
  }

  private _assertBindingsCurrent(): void {
    this.trainingGraph.assertTopologyRevision(
      this.trainingTopologyRevision, 'Quantized LoRA training',
    );
    this.inferenceGraph.assertTopologyRevision(
      this.inferenceTopologyRevision, 'Quantized LoRA inference',
    );
    for (const binding of this._bindings) {
      if (this.trainingGraph.getTensor(binding.masterName) !== binding.master) {
        throw new Error(`Quantized LoRA master '${binding.masterName}' identity changed.`);
      }
      if (this.inferenceGraph.getTensor(binding.targetName) !== binding.target) {
        throw new Error(`Quantized LoRA target '${binding.targetName}' identity changed.`);
      }
      const expectedMasterShape = binding.transpose
        ? [binding.columns, binding.rows]
        : [binding.rows, binding.columns];
      if (!binding.master.isWeight || binding.master.dtype !== 'float32' ||
          !(binding.master.buffer instanceof Float32Array) ||
          binding.master.buffer.length !== binding.rows * binding.columns ||
          !sameShape(binding.master.shape, expectedMasterShape)) {
        throw new Error(`Quantized LoRA master '${binding.masterName}' storage or shape changed.`);
      }
      const descriptor = binding.target.quantization;
      if (!binding.target.isWeight || binding.target.dtype !== 'int8' ||
          !(binding.target.buffer instanceof Int8Array) ||
          binding.target.buffer.length !== binding.rows * binding.columns ||
          !sameShape(binding.target.shape, [binding.rows, binding.columns]) ||
          !descriptor || descriptor.scheme !== 'per_axis' || descriptor.axis !== 0 ||
          !Object.isFrozen(descriptor) || !Object.isFrozen(descriptor.scales) ||
          !Object.isFrozen(descriptor.zero_points) ||
          descriptor.scales.length !== binding.rows ||
          descriptor.scales.some((value) => !Number.isFinite(value) || value <= 0) ||
          descriptor.zero_points.length !== binding.rows ||
          descriptor.zero_points.some((value) => value !== 0)) {
        throw new Error(
          `Quantized LoRA target '${binding.targetName}' no longer has canonical symmetric axis-0 I8 storage.`,
        );
      }
      const consumers = this.inferenceGraph.nodes.filter((node) =>
        Object.values(node.inputs || {}).includes(binding.target));
      if (consumers.length === 0 || consumers.some((node) => {
        const bias = node.inputs?.bias;
        return !QUANTIZED_LINEAR_OPS.has(node.opType) || node.inputs?.weight !== binding.target ||
          !bias || bias.dtype !== 'int32' || !(bias.buffer instanceof Int32Array) ||
          bias.buffer.length !== binding.rows || bias.buffer.some((value) => value !== 0);
      })) {
        throw new Error(
          `Quantized LoRA target '${binding.targetName}' consumers or zero biases changed.`,
        );
      }
    }
  }

  private _stagePacked(bindings: readonly StagedBinding[]): PackedBinding[] {
    const kernels = this.trainer.kernels;
    if (!kernels) throw new Error('WasmQuantizedLoRATrainer has been disposed.');
    return bindings.map((binding) => {
      const currentScales = Float32Array.from(binding.target.quantization.scales);
      const packed = kernels.quantizeWeightI8(
        binding.master.buffer,
        binding.rows,
        binding.columns,
        {
          transposeSource: binding.transpose,
          scalePolicy: binding.scalePolicy,
          scales: currentScales,
        },
      );
      const quantization = binding.scalePolicy === 'preserve'
        ? binding.target.quantization
        : Tensor.normalizeQuantization('int8', binding.target.shape, {
          scheme: 'per_axis',
          axis: 0,
          scales: Array.from(packed.scales),
          zero_points: new Array(binding.rows).fill(0),
        }, `Quantized LoRA target '${binding.targetName}' quantization`);
      return { binding, packed, quantization };
    });
  }

  private async _syncUnlocked(bindings: readonly StagedBinding[]): Promise<WasmQuantizedLoRASyncResult> {
    this._assertBindingsCurrent();
    if (bindings.length === 0) {
      return Object.freeze({
        quantizedWeights: Object.freeze([]),
        weightRevision: this.inferenceGraph.weightRevision,
      });
    }
    const packed = this._stagePacked(bindings);
    const snapshots: TargetSnapshot[] = packed.map(({ binding }) => ({
      binding,
      buffer: new Int8Array(binding.target.buffer),
      quantization: binding.target.quantization,
      sizeBytes: binding.target.sizeBytes,
    }));
    const previousWeightRevision = this.inferenceGraph.weightRevision;
    const previousNextWeightRevision = this.inferenceGraph._nextWeightRevision;

    const restoreGraph = (): void => {
      for (const snapshot of snapshots) {
        snapshot.binding.target.buffer = new Int8Array(snapshot.buffer);
        snapshot.binding.target.quantization = snapshot.quantization;
        snapshot.binding.target.sizeBytes = snapshot.sizeBytes;
      }
      this.inferenceGraph.weightRevision = previousWeightRevision;
      this.inferenceGraph._nextWeightRevision = previousNextWeightRevision;
    };

    for (const entry of packed) {
      entry.binding.target.buffer = new Int8Array(entry.packed.data);
      entry.binding.target.quantization = entry.quantization;
      entry.binding.target.sizeBytes = entry.packed.data.byteLength;
    }
    const weightRevision = this.inferenceGraph._advanceWeightRevision();
    try {
      await this.engine.allocateGraph(this.inferenceGraph);
    } catch (compileError) {
      restoreGraph();
      try {
        await this.engine.allocateGraph(this.inferenceGraph);
      } catch (restoreError) {
        throw new AggregateError(
          [compileError, restoreError],
          'Quantized LoRA sync failed and the prior WASM inference allocation could not be restored.',
        );
      }
      throw compileError;
    }

    return Object.freeze({
      quantizedWeights: Object.freeze(packed.map(({ binding, packed: result }) => Object.freeze({
        master: binding.masterName,
        target: binding.targetName,
        scalePolicy: binding.scalePolicy,
        scales: new Float32Array(result.scales),
        saturationCount: result.saturationCount,
      }))),
      weightRevision,
    });
  }

  /** Quantize every bound F32 master and refresh the W8 inference allocation once. */
  sync(): Promise<WasmQuantizedLoRASyncResult> {
    return this._enqueue(() => this._syncUnlocked(this._bindings));
  }

  /**
   * Explicit one-time initialization of the F32 masters from current I8 bytes.
   * Training never dequantizes after an optimizer update; F32 remains the
   * persistent source of truth until the caller exports or disposes it.
   */
  initializeMastersFromQuantized(): Promise<readonly Tensor[]> {
    return this._enqueue(() => {
      this._assertBindingsCurrent();
      if (this._trained || this._initializedFromQuantized) {
        throw new Error('Quantized LoRA masters can only be initialized once before this trainer updates them.');
      }
      const stateful = this.trainingGraph as RuntimeGraph & {
        trainingStep?: number;
        optimizerState?: Map<string, unknown> | null;
        _pendingGradientAccumulation?: boolean;
      };
      if ((stateful.trainingStep ?? 0) !== 0 || (stateful.optimizerState?.size ?? 0) !== 0 ||
          stateful._pendingGradientAccumulation === true) {
        throw new Error(
          'Quantized LoRA master initialization requires an untrained graph without optimizer or pending-gradient state.',
        );
      }
      const kernels = this.trainer.kernels;
      if (!kernels) throw new Error('WasmQuantizedLoRATrainer has been disposed.');
      const values = this._bindings.map((binding) => kernels.dequantizeWeightI8(
        new Int8Array(binding.target.buffer),
        Float32Array.from(binding.target.quantization.scales),
        binding.rows,
        binding.columns,
        { transposeDestination: binding.transpose },
      ));
      for (let index = 0; index < this._bindings.length; index++) {
        this._bindings[index].master.buffer.set(values[index]);
        this._bindings[index].master.sizeBytes = values[index].byteLength;
      }
      this.trainingGraph._advanceWeightRevision();
      this._initializedFromQuantized = true;
      return Object.freeze(this._bindings.map((binding) => binding.master));
    });
  }

  /** Train selected bound masters and synchronize only when an update applied. */
  trainStep(
    options: TrainingKernelStepOptions = {},
  ): Promise<WasmTrainStepResult & WasmQuantizedLoRASyncResult> {
    return this._enqueue(async () => {
      this._assertBindingsCurrent();
      const boundNames = new Set(this._bindings.map((binding) => binding.masterName));
      for (const name of options.trainableTensors || []) {
        if (!boundNames.has(name)) {
          throw new Error(`Quantized LoRA trainable tensor '${name}' has no W8 inference binding.`);
        }
      }
      const resolved = resolveTrainingOptimizer(
        ensureTrainingGraphState(this.trainingGraph),
        options.updateMode,
        options.optimizer,
      );
      const training = await this.trainer.trainStep({
        ...options,
        updateMode: resolved.updateMode,
        optimizer: resolved.optimizer,
      });
      this._trained = true;
      const updated = new Set(training.updatedTensors.map((tensor) => tensor.name));
      const changedBindings = this._bindings.filter((binding) => updated.has(binding.masterName));
      const sync = await this._syncUnlocked(changedBindings);
      return { ...training, ...sync };
    });
  }

  async dispose(): Promise<void> {
    if (this._disposed) return;
    this._disposed = true;
    await this._queue;
    try {
      this.trainer.dispose();
    } finally {
      delete (this.engine as ReservableWasmEngine)[QUANTIZED_LORA_OWNER];
    }
  }
}
