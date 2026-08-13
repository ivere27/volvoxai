import { RuntimeGraphBuilder } from '../core/RuntimeGraphBuilder.js';
import { initializeTensor } from './Initializers.js';
import type { TensorInitializer } from './Initializers.js';
import { ensureTrainingGraphState, TrainingGraph } from './TrainingGraph.js';
import type { Tensor } from '../core/Tensor.js';
import type { AddTensorOptions, RuntimeDType, TensorStorage } from '../types.js';

interface TrainableWeightOptions extends AddTensorOptions {
  initializer?: TensorInitializer;
}

interface DropoutBuilderOptions {
  probability?: number;
  seed?: number;
  name?: string;
}

interface RoutedAdapterOptions {
  downBias?: Tensor | null;
  upBias?: Tensor | null;
  dropout?: number;
  seed?: number;
  name?: string;
}

export interface LoRALinearOptions {
  rank: number;
  alpha?: number;
  scale?: number;
  bias?: Tensor | null;
  layout?: 'din' | 'dout';
  opType?: 'Linear' | 'MatMul';
  aInitializer?: TensorInitializer;
  bInitializer?: TensorInitializer;
  name?: string;
}

export interface LoRALinearResult {
  out: Tensor;
  A: Tensor;
  B: Tensor;
  trainableTensors: [string, string];
  scale: number;
}

/** Model authoring extensions that create training metadata and trainable graphs. */
export class TrainingModelBuilder extends RuntimeGraphBuilder {
  constructor(graph: TrainingGraph = new TrainingGraph()) {
    super(graph);
    ensureTrainingGraphState(this.graph);
  }

  /** Add a trainable weight, optionally allocating it with a full-profile initializer. */
  weight(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    dataOrOptions: TrainableWeightOptions | TensorStorage = {},
  ): Tensor {
    const options: TrainableWeightOptions = ArrayBuffer.isView(dataOrOptions) || dataOrOptions instanceof ArrayBuffer
      ? { buffer: dataOrOptions }
      : { ...(dataOrOptions || {}) };
    if (options.buffer == null && options.initializer != null) {
      options.buffer = initializeTensor(shape, dtype, options.initializer);
    }
    delete options.initializer;
    return super.weight(name, shape, dtype, options);
  }

  /** Add train-only inverted Dropout. Normal inference executes this node as identity. */
  dropout(input: Tensor, {
    probability = 0.5,
    seed = 0,
    name = `dropout_${this.graph.nodes.length}`,
  }: DropoutBuilderOptions = {}): Tensor {
    if (!input || input.dtype !== "float32") {
      throw new Error("dropout input must be a float32 tensor.");
    }
    if (typeof probability !== "number" || !Number.isFinite(probability) ||
        probability < 0 || probability >= 1) {
      throw new Error("dropout probability must be finite and in [0, 1).");
    }
    if (!Number.isSafeInteger(seed) || seed < 0) {
      throw new Error("dropout seed must be a non-negative safe integer.");
    }
    return this.addOp(
      "Dropout",
      { input },
      { out: { name: `${name}.out`, shape: [...input.shape] } },
      { ratio: probability, seed },
      { id: name },
    ).out;
  }

  /** Add an explicit F32 LoRA branch whose A/B weights can be trained independently. */
  loraLinear(
    input: Tensor,
    baseWeight: Tensor,
    {
      rank,
      alpha,
      scale,
      bias = null,
      layout = 'din',
      opType = 'Linear',
      aInitializer = { type: 'xavierUniform', seed: 0 },
      bInitializer = 'zeros',
      name = `lora_linear_${this.graph.nodes.length}`,
    }: LoRALinearOptions,
  ): LoRALinearResult {
    if (!input || input.dtype !== 'float32' || input.shape.length === 0) {
      throw new Error('loraLinear input must be a non-scalar float32 tensor.');
    }
    if (!baseWeight || baseWeight.dtype !== 'float32' || baseWeight.shape.length !== 2) {
      throw new Error('loraLinear base weight must be a rank-2 float32 tensor.');
    }
    if (layout !== 'din' && layout !== 'dout') {
      throw new Error("loraLinear layout must be 'din' or 'dout'.");
    }
    if (opType !== 'Linear' && opType !== 'MatMul') {
      throw new Error("loraLinear opType must be 'Linear' or 'MatMul'.");
    }
    if (!Number.isSafeInteger(rank) || rank <= 0) {
      throw new Error('loraLinear rank must be a positive safe integer.');
    }
    if (alpha != null && scale != null) {
      throw new Error('loraLinear accepts alpha or scale, not both.');
    }
    if (alpha != null && (typeof alpha !== 'number' || !Number.isFinite(alpha) || alpha <= 0)) {
      throw new Error('loraLinear alpha must be positive and finite.');
    }
    if (scale != null && (typeof scale !== 'number' || !Number.isFinite(scale) || scale <= 0)) {
      throw new Error('loraLinear scale must be positive and finite.');
    }

    const inputWidth = input.shape.at(-1)!;
    const [weightInputWidth, outputWidth] = layout === 'din'
      ? baseWeight.shape
      : [baseWeight.shape[1], baseWeight.shape[0]];
    if (weightInputWidth !== inputWidth) {
      const expected = layout === 'din'
        ? `[${inputWidth},d_out]`
        : `[d_out,${inputWidth}]`;
      throw new Error(`loraLinear ${layout} base weight must be shaped ${expected}.`);
    }
    if (bias && (bias.dtype !== 'float32' || bias.shape.length !== 1 || bias.shape[0] !== outputWidth)) {
      throw new Error(`loraLinear bias must be a float32 tensor shaped [${outputWidth}].`);
    }

    const effectiveScale = Math.fround(scale ?? (alpha ?? rank) / rank);
    if (!Number.isFinite(effectiveScale) || effectiveScale <= 0) {
      throw new Error('loraLinear effective scale must be positive and representable as float32.');
    }
    const outputShape = [...input.shape.slice(0, -1), outputWidth];

    return this.topologyTransaction(() => {
      const A = this.weight(`${name}.lora_a`, [inputWidth, rank], 'float32', {
        initializer: aInitializer,
      });
      const B = this.weight(`${name}.lora_b`, [rank, outputWidth], 'float32', {
        initializer: bInitializer,
      });
      const scaleTensor = this.weight(`${name}.lora_scale`, [1], 'float32',
        Float32Array.of(effectiveScale));
      const base = this.addOp(
        opType,
        { input, weight: baseWeight, ...(bias ? { bias } : {}) },
        { out: { name: `${name}.base_out`, shape: outputShape } },
        {},
        { id: `${name}.base`, wLayout: layout },
      ).out;
      const reduced = this.addOp(
        'MatMul',
        { input, weight: A },
        { out: { name: `${name}.lora_hidden`, shape: [...input.shape.slice(0, -1), rank] } },
        {},
        { id: `${name}.lora_down`, wLayout: 'din' },
      ).out;
      const delta = this.addOp(
        'MatMul',
        { input: reduced, weight: B },
        { out: { name: `${name}.lora_delta`, shape: outputShape } },
        {},
        { id: `${name}.lora_up`, wLayout: 'din' },
      ).out;
      const scaled = this.addOp(
        'Mul',
        { a: delta, b: scaleTensor },
        { out: { name: `${name}.lora_scaled`, shape: outputShape } },
        {},
        { id: `${name}.lora_scale` },
      ).out;
      const out = this.addOp(
        'Add',
        { a: base, b: scaled },
        { out: { name: `${name}.out`, shape: outputShape } },
        {},
        { id: name },
      ).out;
      return {
        out,
        A,
        B,
        trainableTensors: [A.name, B.name],
        scale: effectiveScale,
      };
    });
  }

  /** Add a top-k routed bottleneck MLP adapter as explicit trainable graph tensors. */
  routedBottleneckAdapter(
    input: Tensor,
    downExpertWeight: Tensor,
    upExpertWeight: Tensor,
    routes: any,
    {
    downBias = null,
    upBias = null,
    dropout = 0,
    seed = 0,
    name = `routed_adapter_${this.graph.nodes.length}`,
    }: RoutedAdapterOptions = {},
  ): Tensor {
    const inputWidth = input?.shape?.at(-1);
    const downShape = downExpertWeight?.shape;
    const upShape = upExpertWeight?.shape;
    if (input?.dtype !== "float32" || !Array.isArray(downShape) || downShape.length !== 3 ||
        !Array.isArray(upShape) || upShape.length !== 3 ||
        downExpertWeight.dtype !== "float32" || upExpertWeight.dtype !== "float32" ||
        downShape[1] !== inputWidth || upShape[0] !== downShape[0] ||
        upShape[1] !== downShape[2] || upShape[2] !== inputWidth) {
      throw new Error("routedBottleneckAdapter requires matching [E,D,H] and [E,H,D] F32 expert weights.");
    }
    if (typeof dropout !== "number" || !Number.isFinite(dropout) || dropout < 0 || dropout >= 1) {
      throw new Error("routedBottleneckAdapter dropout must be finite and in [0, 1).");
    }
    const reduced = this.moeLinear(input, downExpertWeight, routes, {
      bias: downBias,
      name: `${name}.down`,
    }).out;
    const activated = this.addOp(
      "GELU",
      { input: reduced },
      { out: { name: `${name}.gelu`, shape: [...reduced.shape] } },
      {},
      { id: `${name}.gelu` },
    ).out;
    const expanded = this.moeLinear(activated, upExpertWeight, routes, {
      bias: upBias,
      name: `${name}.up`,
    }).out;
    const branch = dropout === 0
      ? expanded
      : this.dropout(expanded, { probability: dropout, seed, name: `${name}.dropout` });
    return this.addOp(
      "Add",
      { a: input, b: branch },
      { out: { name: `${name}.out`, shape: [...input.shape] } },
      {},
      { id: name },
    ).out;
  }
}
