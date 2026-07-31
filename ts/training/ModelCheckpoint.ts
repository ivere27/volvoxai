import { Graph as CoreGraph } from '../core/Graph.js';
import { GraphLoader } from '../core/GraphLoader.js';
import { SafetensorsFile } from '../core/Safetensors.js';
import { TrainingModelBuilder } from './TrainingModelBuilder.js';
import {
  ensureTrainingGraphState,
  synchronizeTrainingGraphState,
  TrainingGraph,
} from './TrainingGraph.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import type { StatefulTrainingGraph } from './TrainingGraph.js';
import {
  canonicalOptimizerDescriptor,
  type TrainingOptimizerDescriptor,
} from './TrainingOptimizer.js';

export const VOLVOX_CHECKPOINT_FORMAT = "volvox.checkpoint.v1";

export interface CheckpointOptimizerEntry {
  name: string;
  m: string;
  v: string;
  step: number;
}

export interface StandaloneTensorConfig {
  shape: number[];
  dtype: RuntimeDType;
  hasBuffer: boolean;
}

export interface CheckpointGraph {
  format: 'volvox-graph/v1';
  inputs: Record<string, any>;
  nodes: any[];
  outputs: string[];
  outputsExplicit: boolean;
  standaloneTensors: Record<string, StandaloneTensorConfig>;
  [name: string]: unknown;
}

type CheckpointTrainingGraph = StatefulTrainingGraph & {
  _pendingGradientAccumulation?: boolean;
  _outputsExplicit?: boolean;
  _autoOutputNames?: string[];
  tokenizerMetadata?: unknown;
  checkpointMetadata?: unknown;
};

export interface ModelCheckpointExportOptions {
  includeOptimizerState?: boolean;
  tokenizerMetadata?: unknown;
  metadata?: unknown;
}

export interface ModelCheckpoint {
  format: typeof VOLVOX_CHECKPOINT_FORMAT;
  graph: CheckpointGraph;
  weights: ArrayBuffer;
  optimizer: ArrayBuffer | null;
  optimizerEntries: CheckpointOptimizerEntry[];
  optimizerDescriptor: TrainingOptimizerDescriptor | null;
  trainingStep: number;
  tokenizerMetadata: unknown;
  trainingMetadata: unknown;
  metadata: unknown;
}

export interface ImportedModelCheckpoint {
  graph: TrainingGraph;
  model: TrainingModelBuilder;
  trainingStep: number;
  optimizerDescriptor: TrainingOptimizerDescriptor | null;
  tokenizerMetadata: unknown;
  trainingMetadata: unknown;
  metadata: unknown;
}

function cloneValue<T>(value: T): T {
  if (value == null) return value;
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function safetensorsDType(dtype: RuntimeDType): "F32" | "I32" | "I8" | "U8" {
  if (dtype === "float32") return "F32";
  if (dtype === "int32") return "I32";
  if (dtype === "int8") return "I8";
  if (dtype === "uint8") return "U8";
  throw new Error(`Checkpoint does not support graph dtype '${dtype}'.`);
}

function bytesOf(buffer: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (buffer instanceof ArrayBuffer) return new Uint8Array(buffer);
  if (ArrayBuffer.isView(buffer)) return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  throw new Error("Checkpoint tensor storage must be an ArrayBuffer or typed array.");
}

function sameShape(left: unknown, right: unknown): boolean {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function assertCheckpointFile(file: SafetensorsFile, kind: string): void {
  if (file.metadata?.format !== VOLVOX_CHECKPOINT_FORMAT || file.metadata?.kind !== kind) {
    throw new Error(`Checkpoint ${kind} safetensors metadata is missing or invalid.`);
  }
}

function assertFiniteArray(values: ArrayLike<number>, label: string): void {
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) throw new Error(`${label} is non-finite at index ${index}.`);
  }
}

function assertNonnegativeArray(values: ArrayLike<number>, label: string): void {
  assertFiniteArray(values, label);
  for (let index = 0; index < values.length; index++) {
    if (values[index] < 0) throw new Error(`${label} is negative at index ${index}.`);
  }
}

function storedOptimizerDescriptor(value: any): TrainingOptimizerDescriptor {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      !value.optimizer || typeof value.optimizer !== "object" || Array.isArray(value.optimizer)) {
    throw new Error("Checkpoint optimizerDescriptor is invalid.");
  }
  const canonical = canonicalOptimizerDescriptor(value.updateMode, value.optimizer);
  const expected = canonical.updateMode === "adamw"
    ? ["learningRate", "weightDecay", "maxGradNorm", "beta1", "beta2", "epsilon"]
    : ["learningRate", "weightDecay", "maxGradNorm"];
  const actual = Object.keys(value.optimizer).sort();
  if (actual.length !== expected.length || expected.some((name) => !actual.includes(name))) {
    throw new Error("Checkpoint optimizerDescriptor must contain exactly the persisted optimizer hyperparameters.");
  }
  return canonical;
}

/** Export a self-contained JS checkpoint object with graph document, weights and optimizer state. */
export function exportModelCheckpoint(
  graph: CoreGraph,
  options: ModelCheckpointExportOptions = {},
): ModelCheckpoint {
  if (!(graph instanceof CoreGraph)) throw new Error("exportModelCheckpoint expects a Graph instance.");
  ensureTrainingGraphState(graph);
  const trainingGraph = synchronizeTrainingGraphState(graph) as CheckpointTrainingGraph;
  if (trainingGraph._pendingGradientAccumulation === true) {
    throw new Error("Cannot checkpoint while gradient accumulation is pending; flush or reset it first.");
  }
  trainingGraph.assertValid();
  for (const name of ["graph", "trainingStep", "optimizerDescriptor", "trainingMetadata"]) {
    if (Object.prototype.hasOwnProperty.call(options, name)) {
      throw new Error(`Checkpoint ${name} is generated from graph state and cannot be overridden.`);
    }
  }
  const trainingStep = trainingGraph.trainingStep ?? 0;
  if (!Number.isSafeInteger(trainingStep) || trainingStep < 0) {
    throw new Error("Checkpoint trainingStep must be a non-negative safe integer.");
  }
  const graphPackage = new TrainingModelBuilder(
    trainingGraph as TrainingGraph,
  ).toGraphPackage();
  const graphDocument = cloneValue(graphPackage.graph) as CheckpointGraph;
  graphDocument.outputsExplicit = trainingGraph._outputsExplicit === true ||
    !sameShape(trainingGraph.outputNames, trainingGraph._autoOutputNames || []);
  for (let index = 0; index < (graphDocument.nodes || []).length; index++) {
    const graphNode = trainingGraph.nodes[index];
    if (!graphNode) throw new Error(`Checkpoint graph contains unexpected node ${index}.`);
    graphDocument.nodes[index].outputs_dtype = Object.fromEntries(
      Object.entries(graphNode.outputs || {}).map(([key, tensor]) => [key, tensor.dtype]),
    );
  }
  const weightsFile = SafetensorsFile.empty({
    metadata: { format: VOLVOX_CHECKPOINT_FORMAT, kind: "weights" },
  });
  const quantizationParameterNames = new Set(
    graphPackage.quantizationParameters.listTensorNames(),
  );
  const produced = new Set(trainingGraph.nodes.flatMap((node) =>
    Object.values(node.outputs || {}).map((tensor) => tensor.name)));
  graphDocument.standaloneTensors = {};
  for (const tensor of trainingGraph.tensors.values()) {
    const standalone = !tensor.isWeight && !tensor.isInput && !produced.has(tensor.name);
    if (!tensor.isWeight && !standalone) continue;
    if (tensor.isWeight && !tensor.buffer) {
      throw new Error(`Cannot checkpoint weight '${tensor.name}' without CPU storage.`);
    }
    if (standalone) {
      graphDocument.standaloneTensors[tensor.name] = {
        shape: [...tensor.shape],
        dtype: tensor.dtype,
        hasBuffer: tensor.buffer != null,
      };
    }
    if (tensor.buffer && !quantizationParameterNames.has(tensor.name)) {
      weightsFile.addTensor(tensor.name, safetensorsDType(tensor.dtype), tensor.shape, bytesOf(tensor.buffer));
    }
  }
  for (const [name, tensor] of graphPackage.quantizationParameters.tensorEntries()) {
    weightsFile.addTensor(name, tensor.dtype, tensor.shape, tensor.getBytes());
  }

  let optimizer: ArrayBuffer | null = null;
  const optimizerEntries: CheckpointOptimizerEntry[] = [];
  if (options.includeOptimizerState !== false && trainingGraph.optimizerState?.size) {
    const optimizerFile = SafetensorsFile.empty({
      metadata: { format: VOLVOX_CHECKPOINT_FORMAT, kind: "adamw" },
    });
    let index = 0;
    for (const [name, state] of trainingGraph.optimizerState) {
      const tensor = trainingGraph.getTensor(name);
      if (!tensor?.isWeight || tensor.dtype !== "float32" || !(state?.m instanceof Float32Array) ||
          !(state?.v instanceof Float32Array) || !(tensor.buffer instanceof Float32Array) ||
          state.m.length !== state.v.length || state.m.length !== tensor.buffer.length) {
        throw new Error(`Optimizer state for '${name}' is incompatible with its graph tensor.`);
      }
      if (!Number.isSafeInteger(state.step) || state.step <= 0 || state.step > trainingStep) {
        throw new Error(`Optimizer state for '${name}' has an invalid step.`);
      }
      assertFiniteArray(state.m, `Optimizer first moment '${name}'`);
      assertNonnegativeArray(state.v, `Optimizer second moment '${name}'`);
      const mName = `state.${index}.m`;
      const vName = `state.${index}.v`;
      optimizerFile.addTensor(mName, "F32", tensor.shape, bytesOf(state.m));
      optimizerFile.addTensor(vName, "F32", tensor.shape, bytesOf(state.v));
      optimizerEntries.push({ name, m: mName, v: vName, step: state.step });
      index++;
    }
    optimizer = optimizerFile.toArrayBuffer();
  }

  const trainingMetadata = cloneValue(trainingGraph.trainingMetadata ?? null);
  const rawOptimizerDescriptor = trainingGraph.optimizerDescriptor ?? null;
  const optimizerDescriptor = rawOptimizerDescriptor == null ? null : storedOptimizerDescriptor(rawOptimizerDescriptor);
  if (trainingStep > 0 && optimizerDescriptor == null) {
    throw new Error("A trained checkpoint requires an optimizerDescriptor for self-contained resume.");
  }
  if (optimizer != null && optimizerDescriptor?.updateMode !== "adamw") {
    throw new Error("AdamW optimizer moments require an AdamW optimizerDescriptor.");
  }

  return {
    format: VOLVOX_CHECKPOINT_FORMAT,
    graph: graphDocument,
    weights: weightsFile.toArrayBuffer(),
    optimizer,
    optimizerEntries,
    optimizerDescriptor,
    trainingStep,
    tokenizerMetadata: cloneValue(options.tokenizerMetadata ?? trainingGraph.tokenizerMetadata ?? null),
    trainingMetadata,
    metadata: cloneValue(options.metadata ?? trainingGraph.checkpointMetadata ?? {}),
  };
}

/** Import an object returned by exportModelCheckpoint without network access. */
export function importModelCheckpoint(checkpoint: any): ImportedModelCheckpoint {
  if (!checkpoint || checkpoint.format !== VOLVOX_CHECKPOINT_FORMAT) {
    throw new Error(`Unsupported checkpoint format '${checkpoint?.format}'.`);
  }
  for (const name of [
    "graph", "weights", "optimizer", "optimizerEntries", "optimizerDescriptor",
    "trainingStep", "tokenizerMetadata", "trainingMetadata", "metadata",
  ]) {
    if (!Object.prototype.hasOwnProperty.call(checkpoint, name)) {
      throw new Error(`Checkpoint v1 is missing required field '${name}'.`);
    }
  }
  if (!checkpoint.graph || !(checkpoint.weights instanceof ArrayBuffer)) {
    throw new Error("Checkpoint requires a canonical graph document and safetensors weight bytes.");
  }
  const trainingStep = checkpoint.trainingStep;
  if (!Number.isSafeInteger(trainingStep) || trainingStep < 0) {
    throw new Error("Checkpoint trainingStep must be a non-negative safe integer.");
  }
  if (!Array.isArray(checkpoint.optimizerEntries)) {
    throw new Error("Checkpoint optimizerEntries must be an array.");
  }
  const optimizerDescriptor = checkpoint.optimizerDescriptor == null
    ? null
    : storedOptimizerDescriptor(checkpoint.optimizerDescriptor);
  if (trainingStep > 0 && optimizerDescriptor == null) {
    throw new Error("A trained checkpoint requires an optimizerDescriptor for self-contained resume.");
  }
  const graphDocument = cloneValue(checkpoint.graph) as CheckpointGraph;
  if (!graphDocument || typeof graphDocument !== "object" || Array.isArray(graphDocument) ||
      graphDocument.format !== 'volvox-graph/v1') {
    throw new Error("Checkpoint graph must be a canonical volvox-graph/v1 document.");
  }
  const standaloneTensors = graphDocument.standaloneTensors;
  if (!isRecord(graphDocument.inputs) || !Array.isArray(graphDocument.nodes) ||
      !isRecord(standaloneTensors)) {
    throw new Error("Checkpoint graph inputs, nodes, and standaloneTensors are invalid.");
  }
  if (!Array.isArray(graphDocument.outputs) ||
      typeof graphDocument.outputsExplicit !== "boolean") {
    throw new Error("Checkpoint graph outputs and outputsExplicit are invalid.");
  }
  const graph = new TrainingGraph() as CheckpointTrainingGraph & TrainingGraph;
  const builder = new TrainingModelBuilder(graph);
  const tensors = new Map<string, any>();
  const weightsFile = SafetensorsFile.fromArrayBuffer(checkpoint.weights);
  assertCheckpointFile(weightsFile, "weights");
  const affine = GraphLoader._parseSafetensorsAffineQuantization(
    graphDocument,
    [weightsFile],
    ["checkpoint.weights"],
  );
  for (const [name, entry] of weightsFile.tensorEntries()) {
    const stored = weightsFile.toRuntimeTypedArray(entry);
    const runtimeBuffer = stored.slice() as RuntimeTypedArray;
    const dtype = SafetensorsFile.toGraphDType(entry.dtype);
    const standalone = standaloneTensors[name];
    if (standalone && (!sameShape(standalone.shape, entry.shape) || standalone.dtype !== dtype ||
        standalone.hasBuffer !== true)) {
      throw new Error(`Checkpoint standalone tensor '${name}' does not match its stored payload.`);
    }
    const tensor = standalone
      ? graph.addTensor(name, entry.shape, dtype, {
        buffer: runtimeBuffer,
        quantization: affine.hydrated[name],
      })
      : graph.addWeight(name, entry.shape, dtype, {
        buffer: runtimeBuffer,
        quantization: affine.hydrated[name],
      });
    tensors.set(name, tensor);
  }
  for (const [name, info] of Object.entries(standaloneTensors)) {
    if (!info || typeof info !== "object" || Array.isArray(info) || !Array.isArray(info.shape) ||
        typeof info.dtype !== "string" || typeof info.hasBuffer !== "boolean") {
      throw new Error(`Checkpoint standalone tensor '${name}' has invalid metadata.`);
    }
    if (tensors.has(name)) continue;
    if (info.hasBuffer) throw new Error(`Checkpoint standalone tensor '${name}' is missing its payload.`);
    const tensor = graph.addTensor(name, info.shape, info.dtype, {
      quantization: affine.hydrated[name],
    });
    tensors.set(name, tensor);
  }
  graph.weightFiles.push(weightsFile);

  for (const [name, info] of Object.entries(graphDocument.inputs)) {
    if (!isRecord(info) || !Array.isArray(info.shape) || typeof info.dtype !== "string" || !info.dtype) {
      throw new Error(`Checkpoint input '${name}' requires explicit shape and dtype.`);
    }
    const tensor = builder.input(name, info.shape, info.dtype as RuntimeDType, {
      quantization: affine.hydrated[name],
    });
    tensors.set(name, tensor);
  }
  for (const node of graphDocument.nodes) {
    if (!isRecord(node) || !isRecord(node.inputs) || !isRecord(node.outputs) ||
        !isRecord(node.outputs_shape) || !isRecord(node.outputs_dtype) || !isRecord(node.params)) {
      throw new Error("Checkpoint nodes require input/output/shape/dtype/params objects.");
    }
    if ((typeof node.id !== "string" && typeof node.id !== "number") || String(node.id).length === 0 ||
        typeof node.opType !== "string" || !node.opType) {
      throw new Error("Checkpoint nodes require explicit id and opType fields.");
    }
    const inputs: Record<string, any> = {};
    for (const [key, name] of Object.entries(node.inputs || {})) {
      const tensor = tensors.get(name) || graph.getTensor(name);
      if (!tensor) throw new Error(`Checkpoint node '${node.id}' references missing tensor '${name}'.`);
      inputs[key] = tensor;
    }
    const outputs: Record<string, any> = {};
    for (const [key, name] of Object.entries(node.outputs || {})) {
      const shape = node.outputs_shape?.[key];
      const dtype = node.outputs_dtype[key];
      if (!Array.isArray(shape) || typeof dtype !== "string" || !dtype) {
        throw new Error(`Checkpoint node '${node.id}' output '${key}' requires explicit shape and dtype.`);
      }
      outputs[key] = {
        name,
        shape,
        dtype,
        ...(affine.hydrated[name] != null
          ? { quantization: affine.hydrated[name] }
          : {}),
      };
    }
    const created = builder.addNode({
      id: node.id,
      opType: node.opType,
      inputs,
      outputs,
      params: cloneValue(node.params || {}),
    });
    for (const tensor of Object.values(created.outputs)) tensors.set(tensor.name, tensor);
  }
  if (graphDocument.outputsExplicit) {
    builder.outputs(...graphDocument.outputs);
  } else {
    builder.autoOutputs();
    if (!sameShape(graph.outputNames, graphDocument.outputs)) {
      throw new Error("Checkpoint auto-selected outputs do not match its graph topology.");
    }
  }
  GraphLoader._resolveMatMulLayouts(graph);

  graph.optimizerState = null;
  if (checkpoint.optimizer != null) {
    if (!(checkpoint.optimizer instanceof ArrayBuffer)) throw new Error("Checkpoint optimizer bytes must be an ArrayBuffer.");
    const optimizerFile = SafetensorsFile.fromArrayBuffer(checkpoint.optimizer);
    assertCheckpointFile(optimizerFile, "adamw");
    const entries = checkpoint.optimizerEntries || [];
    if (entries.length === 0) throw new Error("Checkpoint optimizer state has no entries.");
    const state = new Map();
    const usedOptimizerTensors = new Set();
    for (const entry of entries) {
      if (!entry || typeof entry.name !== "string" || !entry.name || state.has(entry.name)) {
        throw new Error("Checkpoint optimizer entries require unique non-empty tensor names.");
      }
      if (typeof entry.m !== "string" || !entry.m || typeof entry.v !== "string" || !entry.v ||
          entry.m === entry.v || usedOptimizerTensors.has(entry.m) || usedOptimizerTensors.has(entry.v)) {
        throw new Error(`Checkpoint optimizer entry '${entry.name}' has duplicate or invalid moment names.`);
      }
      if (!Number.isSafeInteger(entry.step) || entry.step <= 0 || entry.step > trainingStep) {
        throw new Error(`Checkpoint optimizer entry '${entry.name}' has an invalid step.`);
      }
      const tensor = graph.getTensor(entry.name);
      const mTensor = optimizerFile.getTensor(entry.m);
      const vTensor = optimizerFile.getTensor(entry.v);
      if (!tensor || !mTensor || !vTensor) throw new Error(`Checkpoint optimizer entry '${entry.name}' is incomplete.`);
      if (!tensor.isWeight || tensor.dtype !== "float32" || mTensor.dtype !== "F32" || vTensor.dtype !== "F32" ||
          !sameShape(mTensor.shape, tensor.shape) || !sameShape(vTensor.shape, tensor.shape)) {
        throw new Error(`Checkpoint optimizer entry '${entry.name}' has incompatible dtype or shape.`);
      }
      const m = new Float32Array(optimizerFile.toRuntimeTypedArray(mTensor));
      const v = new Float32Array(optimizerFile.toRuntimeTypedArray(vTensor));
      if (!(tensor.buffer instanceof Float32Array) ||
          m.length !== tensor.buffer.length || v.length !== m.length) {
        throw new Error(`Checkpoint optimizer entry '${entry.name}' has incompatible dimensions.`);
      }
      assertFiniteArray(m, `Checkpoint optimizer first moment '${entry.name}'`);
      assertNonnegativeArray(v, `Checkpoint optimizer second moment '${entry.name}'`);
      usedOptimizerTensors.add(entry.m);
      usedOptimizerTensors.add(entry.v);
      state.set(entry.name, { m, v, step: entry.step });
    }
    const storedNames = optimizerFile.listTensorNames();
    if (storedNames.length !== usedOptimizerTensors.size ||
        storedNames.some((name) => !usedOptimizerTensors.has(name))) {
      throw new Error("Checkpoint optimizer safetensors contains unreferenced or missing moments.");
    }
    graph.optimizerState = state;
  } else if ((checkpoint.optimizerEntries || []).length !== 0) {
    throw new Error("Checkpoint optimizer entries were provided without optimizer bytes.");
  }
  if (graph.optimizerState && optimizerDescriptor?.updateMode !== "adamw") {
    throw new Error("Checkpoint AdamW moments require an AdamW optimizerDescriptor.");
  }
  graph.trainingStep = trainingStep;
  graph.optimizerDescriptor = optimizerDescriptor;
  graph.tokenizerMetadata = cloneValue(checkpoint.tokenizerMetadata ?? null);
  graph.trainingMetadata = cloneValue(checkpoint.trainingMetadata ?? null);
  graph.checkpointMetadata = cloneValue(checkpoint.metadata ?? {});
  synchronizeTrainingGraphState(graph);
  graph.assertValid();
  return {
    graph,
    model: builder,
    trainingStep: graph.trainingStep,
    optimizerDescriptor: graph.optimizerDescriptor,
    tokenizerMetadata: graph.tokenizerMetadata,
    trainingMetadata: graph.trainingMetadata,
    metadata: graph.checkpointMetadata,
  };
}
