import { parseGraphDocument, type Graph } from '../core/Graph.js';
import { Model } from '../core/Model.js';
import { SafetensorsFile } from '../core/Safetensors.js';
import type { RuntimeDType, RuntimeTypedArray, TensorQuantization } from '../types.js';
import {
  canonicalOptimizerDescriptor,
  type TrainingOptimizerDescriptor,
} from './TrainingOptimizer.js';
import {
  cloneTrainingMutableState,
  createTrainingMutableState,
  type TrainingMutableState,
} from './TrainingShapeContext.js';

/** Logical, bounded-shape checkpoint. Concrete pre-dynamic checkpoints are not this format. */
export const VOLVOX_CHECKPOINT_FORMAT = 'volvox.training-checkpoint/v1' as const;
const LEGACY_CHECKPOINT_FORMAT = 'volvox.checkpoint.v1';

export interface CheckpointOptimizerEntry {
  readonly name: string;
  readonly m: string;
  readonly v: string;
  readonly step: number;
  readonly shape: readonly number[];
}

export interface CheckpointParameterDescriptor {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly role: 'model_weight' | 'quantization_parameter';
}

export interface ModelCheckpointExportOptions {
  readonly includeOptimizerState?: boolean;
  readonly tokenizerMetadata?: unknown;
  readonly metadata?: unknown;
}

export interface ModelCheckpoint {
  readonly format: typeof VOLVOX_CHECKPOINT_FORMAT;
  readonly logicalGraph: Readonly<Record<string, unknown>>;
  readonly logicalFingerprint: string;
  readonly sourceDefinitionId: string;
  readonly sourceWeightRevisionId: string;
  readonly parameterDescriptors: readonly CheckpointParameterDescriptor[];
  readonly parameters: ArrayBuffer;
  readonly quantizationByTensor: Readonly<Record<string, TensorQuantization>>;
  readonly optimizer: ArrayBuffer | null;
  readonly optimizerEntries: readonly CheckpointOptimizerEntry[];
  readonly optimizerDescriptor: TrainingOptimizerDescriptor | null;
  readonly trainingStep: number;
  readonly tokenizerMetadata: unknown;
  readonly trainingMetadata: unknown;
  readonly metadata: unknown;
}

export interface ImportedModelCheckpoint {
  readonly snapshot: Model;
  readonly trainingStep: number;
  readonly optimizerDescriptor: TrainingOptimizerDescriptor | null;
  readonly tokenizerMetadata: unknown;
  readonly trainingMetadata: unknown;
  readonly metadata: unknown;
}

export interface RestoredTrainingCheckpoint extends ImportedModelCheckpoint {
  readonly state: TrainingMutableState;
}

function cloneValue<T>(value: T): T {
  if (value == null) return value;
  if (typeof structuredClone === 'function') return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) &&
    !ArrayBuffer.isView(value) && !(value instanceof ArrayBuffer);
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function safetensorsDType(dtype: RuntimeDType): 'F32' | 'I32' | 'I8' | 'U8' {
  if (dtype === 'float32') return 'F32';
  if (dtype === 'int32') return 'I32';
  if (dtype === 'int8') return 'I8';
  if (dtype === 'uint8') return 'U8';
  throw new Error(`Checkpoint does not support parameter dtype '${dtype}'.`);
}

function bytesOf(buffer: RuntimeTypedArray): Uint8Array {
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

function assertFinite(values: Float32Array, label: string, nonnegative = false): void {
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index]) || (nonnegative && values[index] < 0)) {
      throw new Error(`${label} is ${nonnegative ? 'negative or ' : ''}non-finite at index ${index}.`);
    }
  }
}

function cloneQuantization(
  source: Readonly<Record<string, TensorQuantization>>,
): Readonly<Record<string, TensorQuantization>> {
  return Object.freeze(Object.fromEntries(Object.entries(source).map(([name, value]) => [
    name,
    value.scheme === 'per_tensor'
      ? Object.freeze({ ...value })
      : Object.freeze({
        scheme: value.scheme,
        axis: value.axis,
        scales: Object.freeze([...value.scales]),
        zero_points: Object.freeze([...value.zero_points]),
      }),
  ])));
}

/** Canonical decoded graph document; symbolic dimensions are retained verbatim. */
export function checkpointGraphDocument(graph: Graph): Readonly<Record<string, unknown>> {
  const dimensions = Object.fromEntries(Object.entries(graph.dimensions).map(([name, value]) => [
    name,
    { min: value.min, max: value.max, multiple_of: value.multiple_of },
  ]));
  const inputs = Object.fromEntries(Object.entries(graph.inputs).map(([name, value]) => [
    name,
    { dtype: value.dtype, shape: [...value.shape] },
  ]));
  const nodes = graph.nodes.map((node) => ({
    id: node.id,
    opType: node.opType,
    inputs: { ...node.inputs },
    outputs: Object.fromEntries(Object.entries(node.outputs).map(([port, value]) => [port, {
      tensor: value.tensor,
      dtype: value.dtype,
      shape: [...value.shape],
    }])),
    params: cloneValue(node.params),
  }));
  return Object.freeze({
    format: graph.format,
    dimensions,
    inputs,
    nodes,
    outputs: [...graph.outputs],
    ...(graph.quantization === null ? {} : { quantization: cloneValue(graph.quantization) }),
  });
}

function canonicalStoredOptimizer(value: unknown): TrainingOptimizerDescriptor {
  if (!isRecord(value) || !isRecord(value.optimizer)) {
    throw new Error('Checkpoint optimizerDescriptor is invalid.');
  }
  return canonicalOptimizerDescriptor(
    value.updateMode,
    value.optimizer,
  );
}

function assertModel(value: unknown): asserts value is Model {
  if (!(value instanceof Model)) {
    throw new Error(
      'Dynamic training v1 checkpoints require a Model; concrete Graph checkpoints ' +
      'must be converted to the bounded logical schema and retrained or resumed from converted parameters.',
    );
  }
}

/** Export an untrained or externally supplied immutable logical revision. */
export function exportModelCheckpoint(
  snapshot: Model,
  options: ModelCheckpointExportOptions = {},
): ModelCheckpoint {
  assertModel(snapshot);
  return exportTrainingCheckpoint(snapshot, createTrainingMutableState(snapshot), options);
}

/** @internal Export the Trainer's private fixed parameter and optimizer state. */
export function exportTrainingCheckpoint(
  snapshot: Model,
  sourceState: TrainingMutableState,
  options: ModelCheckpointExportOptions = {},
): ModelCheckpoint {
  assertModel(snapshot);
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new Error('Checkpoint export options must be an object.');
  }
  const allowedOptions = new Set(['includeOptimizerState', 'tokenizerMetadata', 'metadata']);
  const unknownOptions = Object.keys(options).filter((name) => !allowedOptions.has(name));
  if (unknownOptions.length !== 0) {
    throw new Error(`Checkpoint-derived fields cannot be overridden: ${unknownOptions.join(', ')}.`);
  }
  const state = cloneTrainingMutableState(sourceState);
  const parametersFile = SafetensorsFile.empty({
    metadata: { format: VOLVOX_CHECKPOINT_FORMAT, kind: 'fixed-parameters' },
  });
  const descriptors = snapshot.weightDescriptors.map((descriptor) => {
    const data = state.parameters.get(descriptor.name);
    if (!data || data.byteLength !== descriptor.sizeBytes) {
      throw new Error(`Checkpoint parameter '${descriptor.name}' changed fixed storage size.`);
    }
    parametersFile.addTensor(
      descriptor.name,
      safetensorsDType(descriptor.dtype),
      descriptor.shape,
      bytesOf(data),
    );
    return Object.freeze({
      name: descriptor.name,
      dtype: descriptor.dtype,
      shape: Object.freeze([...descriptor.shape]),
      role: descriptor.role,
    });
  });

  let optimizer: ArrayBuffer | null = null;
  const optimizerEntries: CheckpointOptimizerEntry[] = [];
  if (options.includeOptimizerState !== false && state.optimizerState?.size) {
    const optimizerFile = SafetensorsFile.empty({
      metadata: { format: VOLVOX_CHECKPOINT_FORMAT, kind: 'adamw-fixed-state' },
    });
    let index = 0;
    for (const [name, moments] of state.optimizerState) {
      const descriptor = snapshot.weightDescriptors.find((candidate) => candidate.name === name);
      const parameter = state.parameters.get(name);
      if (!descriptor || descriptor.dtype !== 'float32' || !(parameter instanceof Float32Array) ||
          !(moments.m instanceof Float32Array) || !(moments.v instanceof Float32Array) ||
          moments.m.length !== parameter.length || moments.v.length !== parameter.length) {
        throw new Error(`Optimizer state for '${name}' is incompatible with its fixed parameter.`);
      }
      if (!Number.isSafeInteger(moments.step) || moments.step <= 0 ||
          moments.step > state.trainingStep) {
        throw new Error(`Optimizer state for '${name}' has an invalid step.`);
      }
      assertFinite(moments.m, `Optimizer first moment '${name}'`);
      assertFinite(moments.v, `Optimizer second moment '${name}'`, true);
      const m = `state.${index}.m`;
      const v = `state.${index}.v`;
      optimizerFile.addTensor(m, 'F32', descriptor.shape, bytesOf(moments.m));
      optimizerFile.addTensor(v, 'F32', descriptor.shape, bytesOf(moments.v));
      optimizerEntries.push(Object.freeze({
        name,
        m,
        v,
        step: moments.step,
        shape: Object.freeze([...descriptor.shape]),
      }));
      index++;
    }
    optimizer = optimizerFile.toArrayBuffer();
  }
  const optimizerDescriptor = state.optimizerDescriptor === null
    ? null
    : canonicalStoredOptimizer(state.optimizerDescriptor);
  if (state.trainingStep > 0 && optimizerDescriptor === null) {
    throw new Error('A trained checkpoint requires an optimizerDescriptor.');
  }
  if (optimizer !== null && optimizerDescriptor?.updateMode !== 'adamw') {
    throw new Error('AdamW moments require an AdamW optimizer descriptor.');
  }

  return Object.freeze({
    format: VOLVOX_CHECKPOINT_FORMAT,
    logicalGraph: checkpointGraphDocument(snapshot.graph),
    logicalFingerprint: snapshot.definitionFingerprint,
    sourceDefinitionId: snapshot.definitionId,
    sourceWeightRevisionId: snapshot.weightRevisionId,
    parameterDescriptors: Object.freeze(descriptors),
    parameters: parametersFile.toArrayBuffer(),
    quantizationByTensor: cloneQuantization(snapshot.quantizationByTensor),
    optimizer,
    optimizerEntries: Object.freeze(optimizerEntries),
    optimizerDescriptor,
    trainingStep: state.trainingStep,
    tokenizerMetadata: cloneValue(options.tokenizerMetadata ?? null),
    trainingMetadata: cloneValue(state.trainingMetadata ?? null),
    metadata: cloneValue(options.metadata ?? {}),
  });
}

function assertCheckpoint(value: unknown): asserts value is ModelCheckpoint {
  if (isRecord(value) && value.format === LEGACY_CHECKPOINT_FORMAT) {
    throw new Error(
      'Concrete volvox.checkpoint.v1 files are not loadable by dynamic training v1. ' +
      'Convert the model to a bounded logical package and retrain, or explicitly convert its fixed parameters.',
    );
  }
  if (!isRecord(value) || value.format !== VOLVOX_CHECKPOINT_FORMAT) {
    throw new Error(`Unsupported logical checkpoint format '${String((value as any)?.format)}'.`);
  }
  for (const field of [
    'logicalGraph', 'logicalFingerprint', 'sourceDefinitionId', 'sourceWeightRevisionId',
    'parameterDescriptors', 'parameters', 'quantizationByTensor', 'optimizer',
    'optimizerEntries', 'optimizerDescriptor', 'trainingStep', 'tokenizerMetadata',
    'trainingMetadata', 'metadata',
  ]) {
    if (!Object.prototype.hasOwnProperty.call(value, field)) {
      throw new Error(`Logical checkpoint v1 is missing required field '${field}'.`);
    }
  }
}

/** @internal Restore both immutable logical model and private fixed training state. */
export function restoreTrainingCheckpoint(
  checkpoint: unknown,
  expectedDefinition: Model | null = null,
): RestoredTrainingCheckpoint {
  assertCheckpoint(checkpoint);
  if (!(checkpoint.parameters instanceof ArrayBuffer) ||
      !Array.isArray(checkpoint.parameterDescriptors) ||
      !Array.isArray(checkpoint.optimizerEntries) ||
      !Number.isSafeInteger(checkpoint.trainingStep) || checkpoint.trainingStep < 0 ||
      typeof checkpoint.logicalFingerprint !== 'string' || !checkpoint.logicalFingerprint) {
    throw new Error('Logical checkpoint v1 has invalid parameter or training metadata.');
  }
  const descriptorNames = new Set<string>();
  const descriptors = checkpoint.parameterDescriptors.map((raw, index) => {
    if (!isRecord(raw) || typeof raw.name !== 'string' || !raw.name ||
        typeof raw.dtype !== 'string' || !Array.isArray(raw.shape) ||
        (raw.role !== 'model_weight' && raw.role !== 'quantization_parameter') ||
        descriptorNames.has(raw.name)) {
      throw new Error(`Logical checkpoint parameter descriptor ${index} is invalid or duplicated.`);
    }
    descriptorNames.add(raw.name);
    return {
      name: raw.name,
      dtype: raw.dtype as RuntimeDType,
      shape: raw.shape as number[],
      role: raw.role as 'model_weight' | 'quantization_parameter',
    };
  });
  const graph = parseGraphDocument(checkpoint.logicalGraph, descriptors.map(({ name, dtype, shape }) => ({
    name,
    dtype,
    shape,
  })));
  if (graph.fingerprint !== checkpoint.logicalFingerprint) {
    throw new Error('Checkpoint logical graph fingerprint does not match its graph and bounds.');
  }
  if (expectedDefinition && expectedDefinition.definitionFingerprint !== graph.fingerprint) {
    throw new Error(
      'Checkpoint logical graph fingerprint or symbolic constraints do not match the requested Trainer model.',
    );
  }

  const parameterFile = SafetensorsFile.fromArrayBuffer(checkpoint.parameters);
  if (parameterFile.metadata?.format !== VOLVOX_CHECKPOINT_FORMAT ||
      parameterFile.metadata?.kind !== 'fixed-parameters') {
    throw new Error('Checkpoint fixed-parameter safetensors metadata is invalid.');
  }
  const storedNames = parameterFile.listTensorNames();
  if (storedNames.length !== descriptors.length ||
      storedNames.some((name) => !descriptorNames.has(name))) {
    throw new Error('Checkpoint fixed-parameter payload does not match its descriptors.');
  }
  const weights: Record<string, {
    name: string;
    dtype: RuntimeDType;
    shape: readonly number[];
    data: RuntimeTypedArray;
    role: 'model_weight' | 'quantization_parameter';
  }> = {};
  for (const descriptor of descriptors) {
    const tensor = parameterFile.getTensor(descriptor.name);
    if (!tensor || SafetensorsFile.toGraphDType(tensor.dtype) !== descriptor.dtype ||
        !sameShape(tensor.shape, descriptor.shape)) {
      throw new Error(`Checkpoint parameter '${descriptor.name}' dtype or shape changed.`);
    }
    weights[descriptor.name] = {
      ...descriptor,
      data: parameterFile.toRuntimeTypedArray(tensor).slice() as RuntimeTypedArray,
    };
  }
  if (!isRecord(checkpoint.quantizationByTensor)) {
    throw new Error('Checkpoint quantizationByTensor must be a descriptor record.');
  }
  const snapshot = Model.capture({
    graph,
    weights,
    quantizationByTensor: checkpoint.quantizationByTensor,
  });
  if (snapshot.definitionFingerprint !== checkpoint.logicalFingerprint) {
    throw new Error('Restored checkpoint changed the logical graph fingerprint.');
  }
  const state = createTrainingMutableState(snapshot);
  state.trainingStep = checkpoint.trainingStep;
  state.trainingMetadata = cloneValue(checkpoint.trainingMetadata);
  state.optimizerDescriptor = checkpoint.optimizerDescriptor === null
    ? null
    : canonicalStoredOptimizer(checkpoint.optimizerDescriptor);
  if (state.trainingStep > 0 && state.optimizerDescriptor === null) {
    throw new Error('A trained checkpoint requires an optimizerDescriptor.');
  }

  if (checkpoint.optimizer !== null) {
    if (!(checkpoint.optimizer instanceof ArrayBuffer) || checkpoint.optimizerEntries.length === 0 ||
        state.optimizerDescriptor?.updateMode !== 'adamw') {
      throw new Error('Checkpoint AdamW state is incomplete or has the wrong optimizer descriptor.');
    }
    const file = SafetensorsFile.fromArrayBuffer(checkpoint.optimizer);
    if (file.metadata?.format !== VOLVOX_CHECKPOINT_FORMAT ||
        file.metadata?.kind !== 'adamw-fixed-state') {
      throw new Error('Checkpoint optimizer safetensors metadata is invalid.');
    }
    const optimizerState = new Map<string, { m: Float32Array; v: Float32Array; step: number }>();
    const used = new Set<string>();
    for (const entry of checkpoint.optimizerEntries) {
      const entryStep = (entry as { readonly step?: unknown }).step;
      if (!isRecord(entry) || typeof entry.name !== 'string' || optimizerState.has(entry.name) ||
          typeof entry.m !== 'string' || typeof entry.v !== 'string' || entry.m === entry.v ||
          !Array.isArray(entry.shape) || !Number.isSafeInteger(entryStep) || (entryStep as number) <= 0 ||
          (entryStep as number) > state.trainingStep || used.has(entry.m) || used.has(entry.v)) {
        throw new Error('Checkpoint optimizer entry is invalid or duplicated.');
      }
      const parameter = state.parameters.get(entry.name);
      const descriptor = descriptors.find((candidate) => candidate.name === entry.name);
      const mTensor = file.getTensor(entry.m);
      const vTensor = file.getTensor(entry.v);
      if (!(parameter instanceof Float32Array) || !descriptor || descriptor.dtype !== 'float32' ||
          !sameShape(descriptor.shape, entry.shape as number[]) ||
          !mTensor || !vTensor || mTensor.dtype !== 'F32' || vTensor.dtype !== 'F32' ||
          !sameShape(mTensor.shape, descriptor.shape) || !sameShape(vTensor.shape, descriptor.shape)) {
        throw new Error(`Checkpoint optimizer state for '${entry.name}' changed fixed shape.`);
      }
      const m = new Float32Array(file.toRuntimeTypedArray(mTensor));
      const v = new Float32Array(file.toRuntimeTypedArray(vTensor));
      if (m.length !== parameter.length || v.length !== parameter.length) {
        throw new Error(`Checkpoint optimizer state for '${entry.name}' changed fixed size.`);
      }
      assertFinite(m, `Checkpoint first moment '${entry.name}'`);
      assertFinite(v, `Checkpoint second moment '${entry.name}'`, true);
      used.add(entry.m);
      used.add(entry.v);
      optimizerState.set(entry.name, { m, v, step: entryStep as number });
    }
    if (file.listTensorNames().length !== used.size ||
        file.listTensorNames().some((name) => !used.has(name))) {
      throw new Error('Checkpoint optimizer payload contains unreferenced state.');
    }
    state.optimizerState = optimizerState;
  } else if (checkpoint.optimizerEntries.length !== 0) {
    throw new Error('Checkpoint optimizer entries were provided without optimizer bytes.');
  }

  return Object.freeze({
    snapshot,
    state,
    trainingStep: state.trainingStep,
    optimizerDescriptor: state.optimizerDescriptor,
    tokenizerMetadata: cloneValue(checkpoint.tokenizerMetadata),
    trainingMetadata: cloneValue(checkpoint.trainingMetadata),
    metadata: cloneValue(checkpoint.metadata),
  });
}

/** Import without exposing mutable optimizer or parameter state. */
export function importModelCheckpoint(checkpoint: unknown): ImportedModelCheckpoint {
  const restored = restoreTrainingCheckpoint(checkpoint);
  return Object.freeze({
    snapshot: restored.snapshot,
    trainingStep: restored.trainingStep,
    optimizerDescriptor: restored.optimizerDescriptor,
    tokenizerMetadata: restored.tokenizerMetadata,
    trainingMetadata: restored.trainingMetadata,
    metadata: restored.metadata,
  });
}
