import { RuntimeGraph, isValidGraphName } from './RuntimeGraph.js';
import {
  assertLosslessJSONValue,
  parseStrictJSON,
  SAFETENSORS_OPEN_READ_WRITE,
  SafetensorsFile,
} from './Safetensors.js';
import type {
  SafetensorsOpenOptions,
  SafetensorsTensor,
} from './Safetensors.js';
import { Tensor } from './Tensor.js';
import { validatePortableQuantizedGraph } from '../ops/quantizedGraphValidation.js';
import { GraphOperatorNormalizer } from '../ops/graphOperatorNormalization.js';
import { runtimeDTypes, runtimeOperatorNames } from '../generated/volvoxaiEnums.js';
import type {
  GraphNode,
  NodeOutputSpec,
  NodeParameters,
  RuntimeDType,
  TensorQuantizationInput,
} from '../types.js';

type UnknownRecord = Record<string, unknown>;
type ConcreteNode = GraphNode<Tensor>;

const RUNTIME_DTYPES = new Set<unknown>(runtimeDTypes);
const RUNTIME_OPERATOR_NAMES = new Set<unknown>(runtimeOperatorNames);

interface RuntimeGraphFetchResponse {
  ok: boolean;
  statusText: string;
  text?: () => Promise<string>;
  json?: () => Promise<unknown>;
  arrayBuffer?: () => Promise<ArrayBuffer>;
}

export type RuntimeGraphFetch = (source: string) => Promise<RuntimeGraphFetchResponse>;

export interface RuntimeGraphLoaderOptions {
  graphUrl?: string;
  fetch?: RuntimeGraphFetch;
  safetensors?: SafetensorsOpenOptions;
  safetensorsCache?: ReadOnlySafetensorsCache;
}

function isCanonicalGraphDocumentUrl(value: string): boolean {
  const suffix = value.search(/[?#]/);
  const path = suffix === -1 ? value : value.slice(0, suffix);
  const basename = path.slice(Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\')) + 1);
  return basename === 'graph.json' ||
    (basename.length > '.graph.json'.length && basename.endsWith('.graph.json'));
}

interface GraphBuildOptions {
  baseTensors?: Map<string, Tensor>;
  reservedNames?: ReadonlySet<string>;
  quantizationByTensor?: Readonly<Record<string, TensorQuantizationInput>>;
}

function isRecord(value: unknown): value is UnknownRecord {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

const SAFETENSORS_AFFINE_QUANTIZATION_FORMAT = 'volvox-affine-safetensors/v1';
export const VOLVOX_RUNTIME_GRAPH_FORMAT = 'volvox-graph/v1';
const RETIRED_AFFINE_PARAM_FIELDS = [
  'quantization',
  'zero_point',
  'input_scale', 'input_zero_point',
  'output_scale', 'output_zero_point',
  'weight_scale', 'weight_zero_point',
  'scales', 'zero_points',
  'scale_tensor', 'zero_point_tensor',
] as const;

function hasOwn(record: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function findRetiredAffineParamPath(value: unknown): string | null {
  const seen = new WeakSet<object>();
  const visit = (item: unknown, path: string): string | null => {
    if (Array.isArray(item)) {
      if (seen.has(item)) return null;
      seen.add(item);
      for (let index = 0; index < item.length; index++) {
        const found = visit(item[index], `${path}[${index}]`);
        if (found != null) return found;
      }
      return null;
    }
    if (!isRecord(item)) return null;
    if (seen.has(item)) return null;
    seen.add(item);
    for (const field of RETIRED_AFFINE_PARAM_FIELDS) {
      if (hasOwn(item, field)) return `${path}.${field}`;
    }
    for (const key of Object.keys(item).sort()) {
      const found = visit(item[key], `${path}.${key}`);
      if (found != null) return found;
    }
    return null;
  };
  return visit(value, 'params');
}

function declaredOutputNames(graphDocument: UnknownRecord): string[] {
  if (!Array.isArray(graphDocument.outputs) ||
      graphDocument.outputs.length === 0 ||
      new Set(graphDocument.outputs).size !== graphDocument.outputs.length) {
    throw new Error(
      '[RuntimeGraphLoader] graph outputs must be a non-empty array of unique, non-empty graph tensor names.',
    );
  }
  if (graphDocument.outputs.some((name) => !isValidGraphName(name))) {
    throw new Error('[RuntimeGraphLoader] graph outputs contain a tensor name that is not a valid current-v1 name.');
  }
  return [...graphDocument.outputs] as string[];
}

function validateRuntimeShape(
  shape: readonly unknown[],
  dtype: RuntimeDType,
  label: string,
): void {
  let elements = 1;
  for (const [index, dimension] of shape.entries()) {
    if (!Number.isSafeInteger(dimension) || (dimension as number) <= 0) {
      throw new Error(`${label} shape dimension ${index} must be a positive safe integer.`);
    }
    elements *= dimension as number;
    if (!Number.isSafeInteger(elements)) {
      throw new Error(`${label} element count exceeds JSON's safe integer range.`);
    }
  }
  const bytes = dtype === 'float32' || dtype === 'int32' ? 4 : 1;
  if (!Number.isSafeInteger(elements * bytes)) {
    throw new Error(`${label} byte size exceeds JSON's safe integer range.`);
  }
}

function assertOnlyFields(
  record: UnknownRecord,
  fields: readonly string[],
  label: string,
): void {
  const allowed = new Set(fields);
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) throw new Error(`[RuntimeGraphLoader] ${label} has unsupported field '${key}'.`);
  }
  for (const key of fields) {
    if (!hasOwn(record, key)) throw new Error(`[RuntimeGraphLoader] ${label} requires field '${key}'.`);
  }
}

interface SafetensorsAffineState {
  hydrated: Record<string, TensorQuantizationInput>;
  parameterNames: Set<string>;
}

interface DeclaredTensorMetadata {
  dtype: RuntimeDType;
  shape: readonly number[];
}

function rejectLegacyQuantizationFields(graphDocument: UnknownRecord): void {
  for (const forbidden of ['weights_quantization', 'weights_quantization_storage']) {
    if (hasOwn(graphDocument, forbidden)) {
      throw new Error(`[RuntimeGraphLoader] ${VOLVOX_RUNTIME_GRAPH_FORMAT} forbids legacy field '${forbidden}'.`);
    }
  }
  const inputs = graphDocument.inputs;
  if (isRecord(inputs)) {
    for (const [name, descriptor] of Object.entries(inputs)) {
      if (isRecord(descriptor) && hasOwn(descriptor, 'quantization')) {
        throw new Error(`[RuntimeGraphLoader] Input '${name}' contains forbidden inline quantization.`);
      }
    }
  }
  const nodes = graphDocument.nodes;
  if (Array.isArray(nodes)) {
    for (const [index, node] of nodes.entries()) {
      if (isRecord(node)) {
        const label = String(node.id ?? node.opType ?? index);
        if (hasOwn(node, 'outputs_quantization')) {
          throw new Error(`[RuntimeGraphLoader] Node ${label} contains forbidden inline quantization.`);
        }
        const retiredPath = findRetiredAffineParamPath(node.params);
        if (retiredPath != null) {
          throw new Error(
            `[RuntimeGraphLoader] Node ${label} contains retired affine payload at ${retiredPath}.`,
          );
        }
      }
    }
  }
  const standaloneTensors = graphDocument.standaloneTensors;
  if (isRecord(standaloneTensors)) {
    for (const [name, descriptor] of Object.entries(standaloneTensors)) {
      if (isRecord(descriptor) && hasOwn(descriptor, 'quantization')) {
        throw new Error(`[RuntimeGraphLoader] Standalone tensor '${name}' contains forbidden inline quantization.`);
      }
    }
  }
}

function sameFields(record: UnknownRecord, expected: readonly string[]): boolean {
  const actual = Object.keys(record).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length &&
    actual.every((field, index) => field === wanted[index]);
}

/** Validate the persisted v1 shape before fetching immutable tensor payloads. */
function validatePersistedGraphDocument(graphDocument: UnknownRecord): void {
  if (!isRecord(graphDocument.inputs)) {
    throw new Error('[RuntimeGraphLoader] graph.json must contain an inputs object.');
  }
  for (const [name, descriptor] of Object.entries(graphDocument.inputs)) {
    if (!isValidGraphName(name)) {
      throw new Error(`[RuntimeGraphLoader] Input tensor name '${name}' is not a valid current-v1 name.`);
    }
    if (!isRecord(descriptor) || !Array.isArray(descriptor.shape)) {
      throw new Error(`[RuntimeGraphLoader] Input '${name}' requires an explicit shape array.`);
    }
    if (typeof descriptor.dtype !== 'string' || !RUNTIME_DTYPES.has(descriptor.dtype)) {
      throw new Error(
        `[RuntimeGraphLoader] Input '${name}' requires an explicit canonical dtype ` +
        "('float32', 'int32', 'int8', or 'uint8').",
      );
    }
    validateRuntimeShape(
      descriptor.shape,
      descriptor.dtype as RuntimeDType,
      `[RuntimeGraphLoader] Input '${name}'`,
    );
  }
  if (!Array.isArray(graphDocument.nodes)) {
    throw new Error('[RuntimeGraphLoader] graph.json must contain a nodes array.');
  }
  for (const [index, node] of graphDocument.nodes.entries()) {
    if (!isRecord(node)) {
      throw new Error(`[RuntimeGraphLoader] Node at index ${index} must be an object.`);
    }
    const label = String(node.id ?? node.opType ?? index);
    if (hasOwn(node, 'op')) {
      throw new Error(`[RuntimeGraphLoader] Node ${label} contains unsupported field 'op'; use 'opType'.`);
    }
    if (typeof node.opType !== 'string' || node.opType.trim().length === 0) {
      throw new Error(`[RuntimeGraphLoader] Node ${label} opType must be a non-empty string.`);
    }
    if (!RUNTIME_OPERATOR_NAMES.has(node.opType)) {
      throw new Error(
        `[RuntimeGraphLoader] Node ${label} uses unknown or offline-only current-v1 ` +
        `opType '${node.opType}'.`,
      );
    }
    if (!isRecord(node.inputs)) {
      throw new Error(`[RuntimeGraphLoader] Node ${label} inputs must be an object.`);
    }
    if (!isRecord(node.outputs) || Object.keys(node.outputs).length === 0) {
      throw new Error(`[RuntimeGraphLoader] Node ${label} requires at least one output.`);
    }
    for (const [port, tensorName] of Object.entries(node.inputs)) {
      if (!isValidGraphName(port) || !isValidGraphName(tensorName)) {
        throw new Error(`[RuntimeGraphLoader] Node ${label} has an invalid input port or tensor name.`);
      }
    }
    const outputPorts = Object.keys(node.outputs);
    for (const [port, tensorName] of Object.entries(node.outputs)) {
      if (!isValidGraphName(port) || !isValidGraphName(tensorName)) {
        throw new Error(`[RuntimeGraphLoader] Node ${label} has an invalid output port or tensor name.`);
      }
    }
    if (!isRecord(node.outputs_shape) || !sameFields(node.outputs_shape, outputPorts)) {
      throw new Error(
        `[RuntimeGraphLoader] Node ${label} outputs_shape must exactly describe every output port.`,
      );
    }
    if (!isRecord(node.outputs_dtype) || !sameFields(node.outputs_dtype, outputPorts)) {
      throw new Error(
        `[RuntimeGraphLoader] Node ${label} outputs_dtype must exactly describe every output port.`,
      );
    }
    for (const port of outputPorts) {
      if (!Array.isArray(node.outputs_shape[port])) {
        throw new Error(`[RuntimeGraphLoader] Node ${label} output '${port}' requires a shape array.`);
      }
      const dtype = node.outputs_dtype[port];
      if (typeof dtype !== 'string' || !RUNTIME_DTYPES.has(dtype)) {
        throw new Error(
          `[RuntimeGraphLoader] Node ${label} output '${port}' has unsupported dtype '${String(dtype)}'.`,
        );
      }
      validateRuntimeShape(
        node.outputs_shape[port] as readonly unknown[],
        dtype as RuntimeDType,
        `[RuntimeGraphLoader] Node ${label} output '${port}'`,
      );
    }
    if (hasOwn(node, 'params') && !isRecord(node.params)) {
      throw new Error(`[RuntimeGraphLoader] Node ${label} params must be an object.`);
    }
  }
}

function parseSafetensorsAffineQuantization(
  graphDocument: UnknownRecord,
  weightFiles: readonly SafetensorsFile[],
  sourceList: readonly string[],
): SafetensorsAffineState {
  const inputs = graphDocument.inputs;
  const nodes = graphDocument.nodes;
  const entries = new Map<string, { file: SafetensorsFile; entry: SafetensorsTensor }>();
  const declared = new Map<string, DeclaredTensorMetadata>();
  for (let fileIndex = 0; fileIndex < weightFiles.length; fileIndex++) {
    const file = weightFiles[fileIndex];
    const source = sourceList[fileIndex];
    for (const forbidden of ['weights_quantization_storage', 'weights_quantization']) {
      if (hasOwn(file.metadata, forbidden)) {
        throw new Error(
          `[RuntimeGraphLoader] Safetensors source '${source}' contains forbidden legacy metadata '${forbidden}'.`,
        );
      }
    }
    for (const [name, entry] of file.tensorEntries()) {
      if (entries.has(name)) {
        throw new Error(`[RuntimeGraphLoader] Tensor '${name}' occurs in more than one safetensors source.`);
      }
      entries.set(name, { file, entry });
      declared.set(name, {
        dtype: SafetensorsFile.toGraphDType(entry.dtype),
        shape: entry.shape,
      });
    }
  }
  if (isRecord(inputs)) {
    for (const [name, descriptor] of Object.entries(inputs)) {
      if (!isRecord(descriptor) || typeof descriptor.dtype !== 'string' ||
          !RUNTIME_DTYPES.has(descriptor.dtype) || !Array.isArray(descriptor.shape)) continue;
      declared.set(name, {
        dtype: descriptor.dtype as RuntimeDType,
        shape: descriptor.shape as number[],
      });
    }
  }
  if (Array.isArray(nodes)) {
    for (const node of nodes) {
      if (!isRecord(node) || !isRecord(node.outputs)) continue;
      const outputDtypes = isRecord(node.outputs_dtype) ? node.outputs_dtype : {};
      const outputShapes = isRecord(node.outputs_shape) ? node.outputs_shape : {};
      for (const [port, name] of Object.entries(node.outputs)) {
        if (typeof name !== 'string' || !Array.isArray(outputShapes[port])) continue;
        const dtype = outputDtypes[port];
        if (typeof dtype !== 'string' || !RUNTIME_DTYPES.has(dtype)) continue;
        declared.set(name, {
          dtype: dtype as RuntimeDType,
          shape: outputShapes[port] as number[],
        });
      }
    }
  }
  const standaloneTensors = graphDocument.standaloneTensors;
  if (isRecord(standaloneTensors)) {
    for (const [name, descriptor] of Object.entries(standaloneTensors)) {
      if (!isRecord(descriptor) || typeof descriptor.dtype !== 'string' ||
          !RUNTIME_DTYPES.has(descriptor.dtype) || !Array.isArray(descriptor.shape)) continue;
      declared.set(name, {
        dtype: descriptor.dtype as RuntimeDType,
        shape: descriptor.shape as number[],
      });
    }
  }

  if (!hasOwn(graphDocument, 'quantization')) return {
    hydrated: Object.create(null) as Record<string, TensorQuantizationInput>,
    parameterNames: new Set<string>(),
  };
  const root = graphDocument.quantization;
  if (!isRecord(root)) throw new Error('[RuntimeGraphLoader] quantization must be an object.');
  assertOnlyFields(root, ['format', 'tensors'], 'quantization');
  if (root.format !== SAFETENSORS_AFFINE_QUANTIZATION_FORMAT) {
    throw new Error(`[RuntimeGraphLoader] Unsupported quantization format '${String(root.format)}'.`);
  }
  if (!isRecord(root.tensors) || Object.keys(root.tensors).length === 0) {
    throw new Error('[RuntimeGraphLoader] quantization.tensors must be a non-empty object.');
  }

  const hydrated: Record<string, TensorQuantizationInput> = Object.create(null) as
    Record<string, TensorQuantizationInput>;
  const parameterNames = new Set<string>();
  for (const [name, rawDescriptor] of Object.entries(root.tensors)) {
    const target = declared.get(name);
    if (!target) throw new Error(`[RuntimeGraphLoader] Quantization declares unknown tensor '${name}'.`);
    if (target.dtype !== 'int8' && target.dtype !== 'uint8') {
      throw new Error(`[RuntimeGraphLoader] Quantization target '${name}' must have int8 or uint8 storage.`);
    }
    if (!isRecord(rawDescriptor)) {
      throw new Error(`[RuntimeGraphLoader] Quantization descriptor for '${name}' must be an object.`);
    }
    const perAxis = rawDescriptor.scheme === 'per_axis';
    assertOnlyFields(
      rawDescriptor,
      perAxis
        ? ['scheme', 'axis', 'scale_tensor', 'zero_point_tensor']
        : ['scheme', 'scale_tensor', 'zero_point_tensor'],
      `Quantization descriptor for '${name}'`,
    );
    if (rawDescriptor.scheme !== 'per_tensor' && !perAxis) {
      throw new Error(`[RuntimeGraphLoader] Quantization descriptor for '${name}' has unsupported scheme '${String(rawDescriptor.scheme)}'.`);
    }
    if (typeof rawDescriptor.scale_tensor !== 'string' || rawDescriptor.scale_tensor.length === 0 ||
        typeof rawDescriptor.zero_point_tensor !== 'string' || rawDescriptor.zero_point_tensor.length === 0 ||
        rawDescriptor.scale_tensor === rawDescriptor.zero_point_tensor) {
      throw new Error(`[RuntimeGraphLoader] Quantization descriptor for '${name}' requires distinct parameter tensor names.`);
    }
    const scaleName = rawDescriptor.scale_tensor;
    const zeroName = rawDescriptor.zero_point_tensor;
    const scaleRecord = entries.get(scaleName);
    const zeroRecord = entries.get(zeroName);
    if (!scaleRecord || !zeroRecord) {
      throw new Error(`[RuntimeGraphLoader] Quantization parameters for '${name}' must exist in safetensors.`);
    }
    if (scaleRecord.entry.dtype !== 'F32' || scaleRecord.entry.shape.length !== 1) {
      throw new Error(`[RuntimeGraphLoader] Scale tensor '${scaleName}' for '${name}' must be rank-1 F32.`);
    }
    const expectedZeroDtype = target.dtype === 'int8' ? 'I8' : 'U8';
    if (zeroRecord.entry.dtype !== expectedZeroDtype || zeroRecord.entry.shape.length !== 1) {
      throw new Error(`[RuntimeGraphLoader] Zero-point tensor '${zeroName}' for '${name}' must be rank-1 ${expectedZeroDtype}.`);
    }
    let count = 1;
    let axis: number | undefined;
    if (perAxis) {
      if (!Number.isInteger(rawDescriptor.axis) || target.shape.length === 0) {
        throw new Error(`[RuntimeGraphLoader] Per-axis quantization for '${name}' requires an integer axis.`);
      }
      const rawAxis = rawDescriptor.axis as number;
      axis = rawAxis < 0 ? rawAxis + target.shape.length : rawAxis;
      if (axis < 0 || axis >= target.shape.length ||
          !Number.isInteger(target.shape[axis]) || target.shape[axis] <= 0) {
        throw new Error(`[RuntimeGraphLoader] Per-axis quantization axis for '${name}' is outside its concrete rank.`);
      }
      count = target.shape[axis];
    }
    if (scaleRecord.entry.shape[0] !== count || zeroRecord.entry.shape[0] !== count) {
      throw new Error(`[RuntimeGraphLoader] Quantization parameters for '${name}' must have shape [${count}].`);
    }
    const scales = Array.from(scaleRecord.file.toRuntimeTypedArray(scaleRecord.entry));
    const zeroPoints = Array.from(zeroRecord.file.toRuntimeTypedArray(zeroRecord.entry));
    if (scales.some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new Error(`[RuntimeGraphLoader] Quantization scales for '${name}' must be finite and positive.`);
    }
    const minimum = target.dtype === 'int8' ? -128 : 0;
    const maximum = target.dtype === 'int8' ? 127 : 255;
    if (zeroPoints.some((value) => !Number.isInteger(value) || value < minimum || value > maximum)) {
      throw new Error(`[RuntimeGraphLoader] Quantization zero points for '${name}' are outside ${target.dtype} range.`);
    }
    hydrated[name] = perAxis
      ? { scheme: 'per_axis', axis: axis!, scales, zero_points: zeroPoints }
      : { scheme: 'per_tensor', scale: scales[0], zero_point: zeroPoints[0] };
    parameterNames.add(scaleName);
    parameterNames.add(zeroName);
  }
  for (const parameterName of parameterNames) {
    if (hasOwn(root.tensors, parameterName)) {
      throw new Error(`[RuntimeGraphLoader] Quantization parameter '${parameterName}' cannot itself be quantized.`);
    }
  }
  if (Array.isArray(nodes)) {
    for (const [index, node] of nodes.entries()) {
      if (!isRecord(node) || !isRecord(node.inputs) || !isRecord(node.outputs)) continue;
      const label = String(node.id ?? node.opType ?? index);
      if (node.opType === 'QuantizeLinear') {
        for (const outputName of Object.values(node.outputs)) {
          if (typeof outputName !== 'string') continue;
          const descriptor = root.tensors[outputName];
          if (!isRecord(descriptor) || descriptor.scale_tensor !== node.inputs.scale ||
              descriptor.zero_point_tensor !== node.inputs.zero_point) {
            throw new Error(
              `[RuntimeGraphLoader] QuantizeLinear node ${label} parameter inputs must match ` +
              `the output tensor '${outputName}' quantization references.`,
            );
          }
        }
      } else if (node.opType === 'DequantizeLinear') {
        const inputName = node.inputs.input;
        const descriptor = typeof inputName === 'string' ? root.tensors[inputName] : undefined;
        if (!isRecord(descriptor) || descriptor.scale_tensor !== node.inputs.scale ||
            descriptor.zero_point_tensor !== node.inputs.zero_point) {
          throw new Error(
            `[RuntimeGraphLoader] DequantizeLinear node ${label} parameter inputs must match ` +
            `the input tensor '${String(inputName)}' quantization references.`,
          );
        }
      }
    }
  }
  return { hydrated, parameterNames };
}

/**
 * Session-scoped cache for immutable model files used by more than one graph.
 * A cache entry owns one fetch and one SafeTensors header parse. Every caller
 * receives a fresh read-only file wrapper so graph/tensor topology remains
 * isolated while the immutable model bytes are shared.
 */
export class ReadOnlySafetensorsCache {
  _entries: Map<string, Promise<SafetensorsFile>>;

  constructor() {
    this._entries = new Map<string, Promise<SafetensorsFile>>();
  }

  get size(): number {
    return this._entries.size;
  }

  clear(): void {
    this._entries.clear();
  }

  async load(
    source: string,
    loader: () => SafetensorsFile | Promise<SafetensorsFile>,
  ): Promise<SafetensorsFile> {
    if (typeof source !== 'string' || source.length === 0) {
      throw new Error('[RuntimeGraphLoader] A read-only safetensors cache key must be a non-empty string.');
    }
    if (typeof loader !== 'function') {
      throw new Error('[RuntimeGraphLoader] A read-only safetensors cache loader is required.');
    }
    let pending = this._entries.get(source);
    if (!pending) {
      pending = Promise.resolve().then(loader).then((file) => {
        if (!(file instanceof SafetensorsFile)) {
          throw new Error('[RuntimeGraphLoader] A safetensors cache loader must return a SafetensorsFile.');
        }
        if (file.flags & SAFETENSORS_OPEN_READ_WRITE) {
          throw new Error('[RuntimeGraphLoader] Writable safetensors files cannot be cached for shared inference.');
        }
        return file;
      });
      this._entries.set(source, pending);
      void pending.catch(() => {
        if (this._entries.get(source) === pending) this._entries.delete(source);
      });
    }
    return (await pending).cloneReadOnly();
  }
}


/**
 * RuntimeGraphLoader.load() phase breakdown, opt-in via
 * `globalThis.__VOLVOX_GRAPH_LOAD_PROFILE`. The phase fields partition load();
 * `unattributedMs` is whatever they did not claim (code between the timers).
 */
interface GraphLoadProfile {
  graphFetchMs: number;
  graphParseMs: number;
  graphValidateMs: number;
  weightFetchMs: number;
  weightDecodeMs: number;
  tensorBuildMs: number;
  nodeBuildMs: number;
  postProcessMs: number;
  totalMs: number;
  unattributedMs: number;
  graphBytes: number;
  weightBytes: number;
  tensorCount: number;
  nodeCount: number;
}

export class RuntimeGraphLoader {
  /**
   * Loads a canonical graph.json document and one or more Safetensors files.
   * @param {RuntimeGraph} graph - An empty RuntimeGraph instance
   * @param {string|string[]} sources - URL(s) to .safetensors files
   * @param {Object} options - Loader and safetensors options. A custom fetch
   *   response exposing only json() supplies a trusted, already-decoded graph;
   *   its provider owns raw JSON duplicate-key validation.
   * @returns {Promise<RuntimeGraph>} Populated graph
   */
  static async load(
    graph: RuntimeGraph,
    sources: string | readonly string[],
    options: RuntimeGraphLoaderOptions = {},
  ): Promise<RuntimeGraph> {
    if (!(graph instanceof RuntimeGraph) || graph.nodes.length !== 0 || graph.tensors.size !== 0) {
      throw new Error('[RuntimeGraphLoader] load requires an empty RuntimeGraph.');
    }
    const sourceList: readonly string[] = typeof sources === 'string' ? [sources] : sources;
    if (sourceList.length === 0 || sourceList.some((source) => typeof source !== "string" || source.length === 0)) {
      throw new Error("[VolvoxAI] At least one safetensors source is required.");
    }
    /* Opt-in phase breakdown. Reading the global once here is the only cost
      * when it is unset; every site below then tests one null local. */
    const profile: GraphLoadProfile | null =
      (globalThis as any).__VOLVOX_GRAPH_LOAD_PROFILE ? {
        graphFetchMs: 0, graphParseMs: 0, graphValidateMs: 0,
        weightFetchMs: 0, weightDecodeMs: 0, tensorBuildMs: 0,
        nodeBuildMs: 0, postProcessMs: 0, totalMs: 0, unattributedMs: 0,
        graphBytes: 0, weightBytes: 0, tensorCount: 0, nodeCount: 0,
      } : null;
    const loadStart = profile ? performance.now() : 0;
    const mark = () => (profile ? performance.now() : 0);
    const primarySource = sourceList[0];
    const sourceSuffix = primarySource.search(/[?#]/);
    const sourcePath = sourceSuffix === -1 ? primarySource : primarySource.slice(0, sourceSuffix);
    const sourceDirectoryEnd = sourcePath.lastIndexOf('/') + 1;
    const graphUrl = options.graphUrl ?? `${sourcePath.slice(0, sourceDirectoryEnd)}graph.json`;
    if (typeof graphUrl !== 'string' || !isCanonicalGraphDocumentUrl(graphUrl)) {
      throw new Error(
        "[RuntimeGraphLoader] graphUrl must name graph.json or a named *.graph.json document.",
      );
    }
    const fetchImpl: RuntimeGraphFetch | undefined = options.fetch ||
      (typeof globalThis.fetch === 'function'
        ? async (source: string): Promise<RuntimeGraphFetchResponse> => globalThis.fetch(source)
        : undefined);
    if (typeof fetchImpl !== 'function') throw new Error('[RuntimeGraphLoader] No fetch implementation is available.');
    console.log(`[VolvoxAI] Loading graph from ${graphUrl}...`);
    const graphFetchStart = mark();
    const graphResponse = await fetchImpl(graphUrl);
    if (!graphResponse.ok) throw new Error(`Failed to load graph.json: ${graphResponse.statusText}`);
    if (profile) profile.graphFetchMs += performance.now() - graphFetchStart;
    const graphParseStart = mark();
    let graphDocument: unknown;
    if (typeof graphResponse.text === 'function') {
      const graphText = await graphResponse.text();
      if (profile) profile.graphBytes += graphText.length;
      graphDocument = parseStrictJSON(graphText, `RuntimeGraph '${graphUrl}'`);
    } else if (typeof graphResponse.json === 'function') {
      // json()-only injected responses are already-parsed trusted objects; only
      // a standard response exposing text() can retain duplicate-key evidence.
      graphDocument = await graphResponse.json();
    } else {
      throw new Error('[RuntimeGraphLoader] RuntimeGraph response must provide text() or json().');
    }
    if (profile) profile.graphParseMs += performance.now() - graphParseStart;
    const graphValidateStart = mark();
    if (!isRecord(graphDocument)) {
      throw new Error('[RuntimeGraphLoader] graph.json must contain a JSON object.');
    }
    assertLosslessJSONValue(graphDocument, `[RuntimeGraphLoader] graph '${graphUrl}'`);
    if (graphDocument.format !== VOLVOX_RUNTIME_GRAPH_FORMAT) {
      throw new Error(
        `[RuntimeGraphLoader] graph.json format must be '${VOLVOX_RUNTIME_GRAPH_FORMAT}', got '${String(graphDocument.format)}'.`,
      );
    }
    declaredOutputNames(graphDocument);
    rejectLegacyQuantizationFields(graphDocument);
    validatePersistedGraphDocument(graphDocument);
    if (profile) profile.graphValidateMs += performance.now() - graphValidateStart;

    const safetensorsOptions = options.safetensors || {};
    const safetensorsFlags = safetensorsOptions.flags ??
      (safetensorsOptions.writable ? SAFETENSORS_OPEN_READ_WRITE : 0);
    const safetensorsCache = options.safetensorsCache;
    if (safetensorsCache != null && !(safetensorsCache instanceof ReadOnlySafetensorsCache)) {
      throw new Error('[RuntimeGraphLoader] safetensorsCache must be a ReadOnlySafetensorsCache.');
    }
    if (safetensorsCache && (safetensorsFlags & SAFETENSORS_OPEN_READ_WRITE)) {
      throw new Error('[RuntimeGraphLoader] safetensorsCache cannot be used with writable safetensors options.');
    }

    const weightFiles: SafetensorsFile[] = [];
    for (const source of sourceList) {
      console.log(`[VolvoxAI] Loading Safetensors model from ${source}...`);
      const loadSafetensors = async () => {
        const fetchStart = mark();
        const response = await fetchImpl(source);
        if (!response.ok) throw new Error(`Failed to load safetensors: ${response.statusText}`);
        if (typeof response.arrayBuffer !== 'function') {
          throw new Error('[RuntimeGraphLoader] Safetensors response must provide arrayBuffer().');
        }
        const buffer = await response.arrayBuffer();
        if (profile) {
          profile.weightFetchMs += performance.now() - fetchStart;
          profile.weightBytes += buffer.byteLength;
        }
        const decodeStart = mark();
        const file = SafetensorsFile.fromArrayBuffer(buffer, safetensorsOptions);
        if (profile) profile.weightDecodeMs += performance.now() - decodeStart;
        return file;
      };
      const safetensors = safetensorsCache
        ? await safetensorsCache.load(source, loadSafetensors)
        : await loadSafetensors();
      weightFiles.push(safetensors);
    }
    const affineStart = mark();
    const safetensorsAffineState = parseSafetensorsAffineQuantization(
      graphDocument,
      weightFiles,
      sourceList,
    );
    if (profile) profile.weightDecodeMs += performance.now() - affineStart;

    const tensorBuildStart = mark();
    const tensorsMap = /* @__PURE__ */ new Map<string, Tensor>();
    const stagedSourceTensors = /* @__PURE__ */ new Map<string, Tensor>();
    graph._assertTopologyMutationAllowed();
    for (const safetensors of weightFiles) {
      for (const [name, entry] of safetensors.tensorEntries()) {
        const dtype = SafetensorsFile.toGraphDType(entry.dtype);
        if (stagedSourceTensors.has(name)) {
          throw new Error(`[RuntimeGraphLoader] Tensor '${name}' occurs more than once in the loaded sources.`);
        }
        const tensor = graph._createTensor(name, entry.shape, dtype, {
          isWeight: true,
          buffer: safetensors.toRuntimeTypedArray(entry),
          quantization: safetensorsAffineState.hydrated[name],
        });
        stagedSourceTensors.set(name, tensor);
        tensorsMap.set(name, tensor);
      }
    }
    if (graphDocument.inputs) {
      const inputsDef = graphDocument.inputs;
      if (!isRecord(inputsDef)) {
        throw new Error('[RuntimeGraphLoader] graph inputs must be an object.');
      }
      for (const [name, info] of Object.entries(inputsDef)) {
        if (safetensorsAffineState.parameterNames.has(name)) {
          throw new Error(
            `[RuntimeGraphLoader] Quantization parameter '${name}' is reserved storage and cannot be exposed as a graph input.`,
          );
        }
        if (stagedSourceTensors.has(name)) throw new Error(`Tensor '${name}' already exists.`);
        if (!isRecord(info)) {
          throw new Error(`[RuntimeGraphLoader] Input '${name}' must be an object.`);
        }
        if (typeof info.dtype !== 'string' ||
            !RUNTIME_DTYPES.has(info.dtype)) {
          throw new Error(
            `[RuntimeGraphLoader] Input '${name}' requires an explicit canonical dtype ` +
            "('float32', 'int32', 'int8', or 'uint8').",
          );
        }
        const tensor = graph._createTensor(
          name,
          info.shape as readonly number[],
          info.dtype as RuntimeDType,
          {
            isInput: true,
            quantization: safetensorsAffineState.hydrated[name],
          },
        );
        stagedSourceTensors.set(name, tensor);
        tensorsMap.set(name, tensor);
      }
    }

    if (profile) {
      profile.tensorBuildMs += performance.now() - tensorBuildStart;
      profile.tensorCount = tensorsMap.size;
    }
    if (!Array.isArray(graphDocument.nodes)) {
      throw new Error("[RuntimeGraphLoader] graph.json must contain a nodes array.");
    }
    const nodeBuildStart = mark();
    RuntimeGraphLoader._buildFromGraphDocument(graph, graphDocument, tensorsMap, {
      baseTensors: stagedSourceTensors,
      reservedNames: safetensorsAffineState.parameterNames,
      quantizationByTensor: safetensorsAffineState.hydrated,
    });
    if (profile) {
      profile.nodeBuildMs += performance.now() - nodeBuildStart;
      profile.nodeCount = graph.nodes.length;
    }

    const postProcessStart = mark();
    graph.weightFiles = weightFiles;

    RuntimeGraphLoader._assertBrowserQuantizationSupported(graph);
    RuntimeGraphLoader._dequantizeConvWeights(graph);
    RuntimeGraphLoader._normalizeConvWeightsForImageLayout(graph);
    
    // `_buildFromGraphDocument` applies the package's required explicit
    // `outputs` selection. Do not recompute leaves here: doing so would
    // silently discard the package author's public result contract.
    const outputNames = [...graph.outputNames];
    RuntimeGraphLoader._resolveMatMulLayouts(graph);
    graph.assertValid();
    if (profile) {
      profile.postProcessMs += performance.now() - postProcessStart;
      profile.totalMs = performance.now() - loadStart;
      profile.unattributedMs = profile.totalMs -
        (profile.graphFetchMs + profile.graphParseMs + profile.graphValidateMs +
         profile.weightFetchMs + profile.weightDecodeMs + profile.tensorBuildMs +
         profile.nodeBuildMs + profile.postProcessMs);
      const sink = (globalThis as any);
      sink.__VOLVOX_GRAPH_LOAD_PROFILE_RESULT = profile;
      (sink.__VOLVOX_GRAPH_LOAD_PROFILE_RESULTS ||= []).push(profile);
    }
    console.log(`[VolvoxAI] Successfully assembled graph. Outputs:`, outputNames);
    return graph;
  }

  static _resolveMatMulLayouts(graph: RuntimeGraph): unknown {
    return GraphOperatorNormalizer.resolveMatMulLayouts(graph);
  }

  static _parseSafetensorsAffineQuantization(
    graphDocument: UnknownRecord,
    weightFiles: readonly SafetensorsFile[],
    sourceList: readonly string[],
  ): SafetensorsAffineState {
    rejectLegacyQuantizationFields(graphDocument);
    return parseSafetensorsAffineQuantization(graphDocument, weightFiles, sourceList);
  }

  static _assertBrowserQuantizationSupported(
    graph: RuntimeGraph | { nodes: ConcreteNode[] },
  ): unknown {
    return validatePortableQuantizedGraph(graph);
  }

  static _buildFromGraphDocument(
    graph: RuntimeGraph,
    graphDocument: UnknownRecord,
    tensorsMap: Map<string, Tensor>,
    options: GraphBuildOptions = {},
  ): void {
    rejectLegacyQuantizationFields(graphDocument);
    const nodesDef = graphDocument.nodes;
    if (!Array.isArray(nodesDef)) {
      throw new Error('[RuntimeGraphLoader] graph nodes must be an array.');
    }

    // Validate the complete name topology before mutating the graph. Document
    // outputs are definitions, never aliases: they cannot overwrite an input,
    // weight, or earlier node output, and inputs may only reference sources or
    // values produced by an earlier node.
    const baseTensors = options.baseTensors || graph.tensors;
    if (!(baseTensors instanceof Map)) {
      throw new Error('[RuntimeGraphLoader] RuntimeGraph document baseTensors must be a Map.');
    }
    const reservedNames = options.reservedNames || new Set();
    const availableNames = new Set(baseTensors.keys());
    for (const [nodeIndex, nodeDef] of nodesDef.entries()) {
      if (!isRecord(nodeDef)) {
        throw new Error(`[RuntimeGraphLoader] Node at index ${nodeIndex} must be an object.`);
      }
      const nodeLabel = String(nodeDef.id ?? nodeDef.opType ?? '<unnamed>');
      if (hasOwn(nodeDef, 'op')) {
        throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} contains unsupported field 'op'; use 'opType'.`);
      }
      if (typeof nodeDef.opType !== 'string' || nodeDef.opType.trim().length === 0) {
        throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} opType must be a non-empty string.`);
      }
      if (!RUNTIME_OPERATOR_NAMES.has(nodeDef.opType)) {
        throw new Error(
          `[RuntimeGraphLoader] Node ${nodeLabel} uses unknown or offline-only current-v1 ` +
          `opType '${nodeDef.opType}'.`,
        );
      }
      if (!isRecord(nodeDef.inputs)) {
        throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} inputs must be an object.`);
      }
      if (!isRecord(nodeDef.outputs) || Object.keys(nodeDef.outputs).length === 0) {
        throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} requires at least one output.`);
      }
      const nodeInputs = nodeDef.inputs;
      const nodeOutputs = nodeDef.outputs;
      for (const [key, tensorName] of Object.entries(nodeInputs)) {
        if (typeof tensorName !== 'string' || !availableNames.has(tensorName)) {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} input '${key}' references undeclared tensor '${String(tensorName)}'.`);
        }
      }
      for (const [key, tensorName] of Object.entries(nodeOutputs)) {
        if (typeof tensorName !== 'string' || tensorName.trim().length === 0) {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} output '${key}' requires a tensor name.`);
        }
        if (reservedNames.has(tensorName)) {
          throw new Error(
            `[RuntimeGraphLoader] Quantization parameter '${tensorName}' is reserved storage and cannot be produced by a graph node.`,
          );
        }
        if (availableNames.has(tensorName)) {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} output '${key}' collides with existing tensor '${tensorName}'.`);
        }
        availableNames.add(tensorName);
      }
    }

    const explicitOutputNames = declaredOutputNames(graphDocument);
    if (explicitOutputNames.some((name) => !availableNames.has(name))) {
      throw new Error(
        '[RuntimeGraphLoader] graph outputs must name declared graph tensors.',
      );
    }
    if (explicitOutputNames.some((name) => reservedNames.has(name))) {
      throw new Error(
        '[RuntimeGraphLoader] quantization parameter tensors cannot be public graph outputs.',
      );
    }

    // Stage every node against local collections, validate the complete graph
    // once, then commit once. Per-node graph mutation would repeatedly validate
    // loaded weights and nodes, making package import O(nodes * tensors).
    graph._assertTopologyMutationAllowed();
    const stagedNodes = [...graph.nodes];
    const stagedTensors = new Map<string, Tensor>(baseTensors);
    const stagedAliases = new Map<string, Tensor>(tensorsMap);
    const stagedOutputs: Array<[string, Tensor]> = [];
    for (const nodeDef of nodesDef) {
      // The first validation pass above establishes these object shapes.
      if (!isRecord(nodeDef) || !isRecord(nodeDef.inputs) || !isRecord(nodeDef.outputs)) {
        throw new Error('[RuntimeGraphLoader] RuntimeGraph document node shape changed during validation.');
      }
      const nodeLabel = String(nodeDef.id ?? nodeDef.opType ?? '<unnamed>');
      const nodeInputs = nodeDef.inputs;
      const nodeOutputs = nodeDef.outputs;
      const inputs: Record<string, Tensor> = {};
      for (const [key, tName] of Object.entries(nodeDef.inputs)) {
        if (typeof tName !== 'string') {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} input '${key}' requires a tensor name.`);
        }
        const t = stagedAliases.get(tName) || stagedTensors.get(tName);
        if (!t) {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} input '${key}' references undeclared tensor '${tName}'.`);
        }
        inputs[key] = t;
      }
      const outputsShape = nodeDef.outputs_shape;
      const outputsDtype = nodeDef.outputs_dtype;
      const quantizationByTensor = options.quantizationByTensor;
      const outputPorts = Object.keys(nodeOutputs);
      if (!isRecord(outputsShape) || !sameFields(outputsShape, outputPorts)) {
        throw new Error(
          `[RuntimeGraphLoader] Node ${nodeLabel} outputs_shape must exactly describe every output port.`,
        );
      }
      if (!isRecord(outputsDtype) || !sameFields(outputsDtype, outputPorts)) {
        throw new Error(
          `[RuntimeGraphLoader] Node ${nodeLabel} outputs_dtype must exactly describe every output port.`,
        );
      }
      const outputDescriptors: Record<string, NodeOutputSpec<Tensor>> = {};
      for (const [key, tensorNameValue] of Object.entries(nodeOutputs)) {
        if (typeof tensorNameValue !== 'string') {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} output '${key}' requires a tensor name.`);
        }
        const tensorName = tensorNameValue;
        const shape = outputsShape[key];
        if (!Array.isArray(shape)) {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} output '${key}' requires a shape array.`);
        }
        const dtype = outputsDtype[key];
        if (typeof dtype !== 'string' || !dtype) {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} output '${key}' requires a declared dtype.`);
        }
        try {
          Tensor.dtypeBytes(dtype as RuntimeDType);
        } catch {
          throw new Error(`[RuntimeGraphLoader] Node ${nodeLabel} output '${key}' has unsupported dtype '${dtype}'.`);
        }
        outputDescriptors[key] = {
          name: tensorName,
          shape: shape as number[],
          dtype: dtype as RuntimeDType,
          ...(quantizationByTensor?.[tensorName] != null
            ? { quantization: quantizationByTensor[tensorName] }
            : {}),
        } as NodeOutputSpec<Tensor>;
      }
      const node = graph._prepareNode({
        opType: nodeDef.opType as string,
        inputs,
        outputs: outputDescriptors,
        params: (isRecord(nodeDef.params) ? nodeDef.params : {}) as NodeParameters,
      }, stagedTensors, stagedNodes);
      stagedNodes.push(node);
      for (const [key, tName] of Object.entries(nodeOutputs)) {
        if (typeof tName !== 'string') continue;
        const t = node.outputs[key];
        if (!t) continue;
        stagedAliases.set(tName, t);
        stagedOutputs.push([tName, t]);
      }
    }
    graph._assertValidState(stagedNodes, stagedTensors, explicitOutputNames || []);
    RuntimeGraphLoader._assertBrowserQuantizationSupported({ nodes: stagedNodes });
    graph._runTopologyTransaction(() => {
      // Install an explicit public-output selection before the only commit so
      // _refreshOutputNames preserves it without a second topology revision.
      if (explicitOutputNames) {
        graph._setOutputNames(explicitOutputNames);
        graph._outputsExplicit = true;
      }
      graph._commitTopology(stagedNodes, stagedTensors, 'load graph document');
      for (const node of stagedNodes) {
        if (typeof node.id === 'number' && node.id >= graph._nextNodeId) {
          graph._nextNodeId = node.id + 1;
        }
      }
    });
    for (const [name, value] of stagedOutputs) tensorsMap.set(name, value);
  }

  static _dequantizeConvWeights(graph: RuntimeGraph): unknown {
    return GraphOperatorNormalizer.dequantizeConvWeights(graph);
  }

  static _normalizeConvWeightsForImageLayout(graph: RuntimeGraph): unknown {
    return GraphOperatorNormalizer.normalizeConvWeightsForImageLayout(graph);
  }

};
