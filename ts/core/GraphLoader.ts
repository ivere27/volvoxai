import { Graph } from './Graph.js';
import {
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
import type {
  GraphNode,
  NodeOutputSpec,
  NodeParameters,
  RuntimeDType,
  TensorQuantization,
  TensorQuantizationInput,
} from '../types.js';

type UnknownRecord = Record<string, unknown>;
type QuantizationValue = TensorQuantizationInput | TensorQuantization | null | undefined;
type WeightQuantizationRecord = Record<string, unknown>;
type ConcreteNode = GraphNode<Tensor>;

interface GraphFetchResponse {
  ok: boolean;
  statusText: string;
  text?: () => Promise<string>;
  json?: () => Promise<unknown>;
  arrayBuffer?: () => Promise<ArrayBuffer>;
}

export type GraphFetch = (source: string) => Promise<GraphFetchResponse>;

export interface GraphLoaderOptions {
  configUrl?: string;
  fetch?: GraphFetch;
  safetensors?: SafetensorsOpenOptions;
  safetensorsCache?: ReadOnlySafetensorsCache;
}

interface BlueprintBuildOptions {
  baseTensors?: Map<string, Tensor>;
  reservedNames?: ReadonlySet<string>;
}

interface CompanionScaleState {
  hydrated: Record<string, TensorQuantizationInput>;
  companionNames: Set<string>;
}

type ModelBuilder = (
  graph: Graph,
  config: UnknownRecord,
  tensors: Map<string, Tensor>,
) => unknown;

function isRecord(value: unknown): value is UnknownRecord {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

const F32_COMPANION_SCALES_FORMAT = 'volvoxai-f32-companion-scales-v1';

function hasOwn(record: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function assertOnlyFields(
  record: UnknownRecord,
  fields: readonly string[],
  label: string,
): void {
  const allowed = new Set(fields);
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) throw new Error(`[GraphLoader] ${label} has unsupported field '${key}'.`);
  }
  for (const key of fields) {
    if (!hasOwn(record, key)) throw new Error(`[GraphLoader] ${label} requires field '${key}'.`);
  }
}

function parseWeightQuantizationStorage(config: UnknownRecord): boolean {
  const hasStorage = hasOwn(config, 'weights_quantization_storage');
  if (!hasStorage) {
    const weightsQuantization = config.weights_quantization;
    if (weightsQuantization != null && !isRecord(weightsQuantization)) {
      throw new Error('[GraphLoader] weights_quantization must be an object when present.');
    }
    for (const [name, descriptor] of Object.entries(weightsQuantization || {})) {
      if (isRecord(descriptor) && (hasOwn(descriptor, 'scales_offset') || hasOwn(descriptor, 'scales_count'))) {
        throw new Error(
          `[GraphLoader] Weight '${name}' uses unsupported binary scale offset fields.`,
        );
      }
    }
    return false;
  }

  const storage = config.weights_quantization_storage;
  if (!isRecord(storage)) {
    throw new Error('[GraphLoader] weights_quantization_storage must be an object.');
  }
  assertOnlyFields(
    storage,
    ['format'],
    'weights_quantization_storage',
  );
  if (storage.format !== F32_COMPANION_SCALES_FORMAT) {
    throw new Error(`[GraphLoader] Unsupported weights_quantization_storage format '${String(storage.format)}'.`);
  }
  if (hasOwn(config, 'weights_quantization')) {
    throw new Error(
      '[GraphLoader] companion-scale storage requires config.json to omit weights_quantization; ' +
      'the safetensors metadata is authoritative.',
    );
  }
  return true;
}

function parseCompanionScaleMetadata(
  file: SafetensorsFile,
  source: string,
): WeightQuantizationRecord | null {
  const label = `Safetensors source '${source}'`;
  const metadata = file.metadata;
  if (!isRecord(metadata)) {
    throw new Error(`[GraphLoader] ${label} __metadata__ must be an object.`);
  }
  const hasStorage = hasOwn(metadata, 'weights_quantization_storage');
  const hasQuantization = hasOwn(metadata, 'weights_quantization');
  if (!hasStorage && !hasQuantization) return null;
  if (!hasStorage || !hasQuantization) {
    throw new Error(
      `[GraphLoader] ${label} companion-scale metadata must contain both ` +
      'weights_quantization_storage and weights_quantization.',
    );
  }
  if (metadata.weights_quantization_storage !== F32_COMPANION_SCALES_FORMAT) {
    throw new Error(
      `[GraphLoader] ${label} weights_quantization_storage must equal ` +
      `'${F32_COMPANION_SCALES_FORMAT}'.`,
    );
  }
  const encoded = metadata.weights_quantization;
  if (typeof encoded !== 'string') {
    throw new Error(`[GraphLoader] ${label} weights_quantization metadata must be a JSON string.`);
  }
  let parsed: unknown;
  try {
    parsed = parseStrictJSON(encoded, `${label} weights_quantization metadata`);
  } catch (error) {
    if (isRecord(error) && error.code === 'ERR_STRICT_JSON_DUPLICATE_KEY') throw error;
    throw new Error(`[GraphLoader] ${label} weights_quantization metadata is not valid JSON.`);
  }
  if (!isRecord(parsed)) {
    throw new Error(`[GraphLoader] ${label} weights_quantization metadata must decode to an object.`);
  }
  return parsed;
}

function mergeCompanionScaleQuantization(
  weightFiles: readonly SafetensorsFile[],
  sourceList: readonly string[],
): WeightQuantizationRecord {
  const merged: WeightQuantizationRecord = Object.create(null) as WeightQuantizationRecord;
  const owners = new Map<string, string>();
  const tensorOwners = new Map<string, string>();
  for (let index = 0; index < weightFiles.length; index++) {
    const file = weightFiles[index];
    const source = sourceList[index];
    const local = parseCompanionScaleMetadata(file, source);
    if (local != null) {
      for (const [name, descriptor] of Object.entries(local)) {
        const previousSource = owners.get(name);
        if (previousSource != null) {
          throw new Error(
            `[GraphLoader] weights_quantization declares weight '${name}' in both ` +
            `'${previousSource}' and '${source}'.`,
          );
        }
        const companionName = `${name}_scale`;
        if (!file.getTensor(name) || !file.getTensor(companionName)) {
          throw new Error(
            `[GraphLoader] ${source} must store declared weight '${name}' and its companion ` +
            `'${companionName}' in the same safetensors file.`,
          );
        }
        merged[name] = descriptor;
        owners.set(name, source);
      }
    }
    for (const [name] of file.tensorEntries()) {
      const previousSource = tensorOwners.get(name);
      if (previousSource != null) {
        throw new Error(
          `[GraphLoader] Tensor '${name}' occurs in both '${previousSource}' and '${source}'.`,
        );
      }
      tensorOwners.set(name, source);
    }
  }
  if (owners.size === 0) {
    throw new Error(
      '[GraphLoader] companion-scale storage requires safetensors metadata to declare at least one weight.',
    );
  }
  return merged;
}

function hydrateCompanionScaleQuantization(
  weightsQuantization: WeightQuantizationRecord,
  weightFiles: readonly SafetensorsFile[],
): CompanionScaleState {
  const loadedEntries = new Map<string, Array<{
    file: SafetensorsFile;
    entry: SafetensorsTensor;
  }>>();
  for (const file of weightFiles) {
    for (const [name, entry] of file.tensorEntries()) {
      const entries = loadedEntries.get(name) || [];
      entries.push({ file, entry });
      loadedEntries.set(name, entries);
    }
  }
  const hydrated: Record<string, TensorQuantizationInput> = Object.create(null) as
    Record<string, TensorQuantizationInput>;
  const companionNames = new Set<string>();

  for (const [name, descriptor] of Object.entries(weightsQuantization)) {
    const weightEntries = loadedEntries.get(name) || [];
    if (weightEntries.length !== 1) {
      throw new Error(
        `[GraphLoader] Declared weight '${name}' must occur exactly once across the loaded safetensors files.`,
      );
    }
    const weight = weightEntries[0].entry;
    if (weight.dtype !== 'I8') {
      throw new Error(`[GraphLoader] Declared weight '${name}' must have safetensors dtype I8.`);
    }
    const companionName = `${name}_scale`;
    const companions = loadedEntries.get(companionName) || [];
    if (companions.length !== 1) {
      throw new Error(
        `[GraphLoader] Companion scale '${companionName}' for weight '${name}' must occur exactly once ` +
        'across the loaded safetensors files.',
      );
    }
    const { file, entry: companion } = companions[0];
    if (companion.dtype !== 'F32') {
      throw new Error(`[GraphLoader] Companion scale '${companionName}' for weight '${name}' must have safetensors dtype F32.`);
    }
    if (!isRecord(descriptor)) {
      throw new Error(`[GraphLoader] Weight '${name}' companion-scale quantization descriptor must be an object.`);
    }
    let expectedCount: number;
    let axis: 0 | undefined;
    if (descriptor.scheme === 'per_axis') {
      assertOnlyFields(
        descriptor,
        ['scheme', 'axis'],
        `Weight '${name}' companion-scale per_axis descriptor`,
      );
      if (descriptor.axis !== 0) {
        throw new Error(`[GraphLoader] Weight '${name}' companion-scale axis must be 0.`);
      }
      axis = descriptor.axis;
      if (weight.shape.length === 0) {
        throw new Error(`[GraphLoader] Weight '${name}' companion-scale axis 0 is outside tensor rank 0.`);
      }
      if (weight.shape[0] <= 0) {
        throw new Error(`[GraphLoader] Weight '${name}' companion-scale axis-0 dimension must be positive.`);
      }
      expectedCount = weight.shape[axis];
    } else if (descriptor.scheme === 'per_tensor') {
      assertOnlyFields(
        descriptor,
        ['scheme'],
        `Weight '${name}' companion-scale per_tensor descriptor`,
      );
      expectedCount = 1;
    } else {
      throw new Error(
        `[GraphLoader] Weight '${name}' companion-scale quantization has unsupported scheme '${String(descriptor.scheme)}'.`,
      );
    }
    if (companion.shape.length !== 1 || companion.shape[0] !== expectedCount) {
      throw new Error(
        `[GraphLoader] Companion scale '${companionName}' for weight '${name}' must have shape [${expectedCount}].`,
      );
    }
    const values: number[] = Array.from(file.toRuntimeTypedArray(companion));
    for (const [index, value] of values.entries()) {
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error(
          `[GraphLoader] Companion scale '${companionName}' value ${index} must be finite and positive.`,
        );
      }
    }
    hydrated[name] = descriptor.scheme === 'per_axis'
      ? { scheme: 'per_axis', axis: axis as 0, scales: values }
      : { scheme: 'per_tensor', scale: values[0] };
    companionNames.add(companionName);
  }
  return { hydrated, companionNames };
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
      throw new Error('[GraphLoader] A read-only safetensors cache key must be a non-empty string.');
    }
    if (typeof loader !== 'function') {
      throw new Error('[GraphLoader] A read-only safetensors cache loader is required.');
    }
    let pending = this._entries.get(source);
    if (!pending) {
      pending = Promise.resolve().then(loader).then((file) => {
        if (!(file instanceof SafetensorsFile)) {
          throw new Error('[GraphLoader] A safetensors cache loader must return a SafetensorsFile.');
        }
        if (file.flags & SAFETENSORS_OPEN_READ_WRITE) {
          throw new Error('[GraphLoader] Writable safetensors files cannot be cached for shared inference.');
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


export class GraphLoader {
  /**
   * Loads a graph from one or more Safetensors files.
   * The Safetensors __metadata__ field must contain a 'volvox_nodes' JSON string.
   * @param {Graph} graph - An empty Graph instance from VolvoxAI.createGraph()
   * @param {string|string[]} sources - URL(s) to .safetensors files
   * @param {Object} options - Loader and safetensors options. A custom fetch
   *   response exposing only json() supplies a trusted, already-decoded config;
   *   its provider owns raw JSON duplicate-key validation.
   * @returns {Promise<Graph>} Populated graph
   */
  static async load(
    graph: Graph,
    sources: string | readonly string[],
    options: GraphLoaderOptions = {},
  ): Promise<Graph> {
    if (!(graph instanceof Graph) || graph.nodes.length !== 0 || graph.tensors.size !== 0) {
      throw new Error('[GraphLoader] load requires an empty Graph.');
    }
    const sourceList: readonly string[] = typeof sources === 'string' ? [sources] : sources;
    if (sourceList.length === 0 || sourceList.some((source) => typeof source !== "string" || source.length === 0)) {
      throw new Error("[VolvoxAI] At least one safetensors source is required.");
    }
    const primarySource = sourceList[0];
    let configUrl = options.configUrl;
    if (!configUrl && primarySource.endsWith('_weights.safetensors')) {
      configUrl = primarySource.replace('_weights.safetensors', '_config.json');
    } else {
      configUrl = configUrl || primarySource.replace(/\/[^\/]+$/, '/config.json');
    }
    if (!configUrl) throw new Error('[GraphLoader] Could not resolve config.json URL.');
    const fetchImpl: GraphFetch | undefined = options.fetch ||
      (typeof globalThis.fetch === 'function'
        ? async (source: string): Promise<GraphFetchResponse> => globalThis.fetch(source)
        : undefined);
    if (typeof fetchImpl !== 'function') throw new Error('[GraphLoader] No fetch implementation is available.');
    console.log(`[VolvoxAI] Loading config from ${configUrl}...`);
    const configResponse = await fetchImpl(configUrl);
    if (!configResponse.ok) throw new Error(`Failed to load config.json: ${configResponse.statusText}`);
    let config: unknown;
    if (typeof configResponse.text === 'function') {
      const configText = await configResponse.text();
      config = parseStrictJSON(configText, `Config '${configUrl}'`);
    } else if (typeof configResponse.json === 'function') {
      // json()-only injected responses are already-parsed trusted objects; only
      // a standard response exposing text() can retain duplicate-key evidence.
      config = await configResponse.json();
    } else {
      throw new Error('[GraphLoader] Config response must provide text() or json().');
    }
    if (!isRecord(config)) {
      throw new Error('[GraphLoader] config.json must contain a JSON object.');
    }

    const usesCompanionScales = parseWeightQuantizationStorage(config);
    const inlineWeightsQuantization: WeightQuantizationRecord | null = usesCompanionScales
      ? null
      : isRecord(config.weights_quantization)
        ? config.weights_quantization
        : null;

    const safetensorsOptions = options.safetensors || {};
    const safetensorsFlags = safetensorsOptions.flags ??
      (safetensorsOptions.writable ? SAFETENSORS_OPEN_READ_WRITE : 0);
    const safetensorsCache = options.safetensorsCache;
    if (safetensorsCache != null && !(safetensorsCache instanceof ReadOnlySafetensorsCache)) {
      throw new Error('[GraphLoader] safetensorsCache must be a ReadOnlySafetensorsCache.');
    }
    if (safetensorsCache && (safetensorsFlags & SAFETENSORS_OPEN_READ_WRITE)) {
      throw new Error('[GraphLoader] safetensorsCache cannot be used with writable safetensors options.');
    }

    const weightFiles: SafetensorsFile[] = [];
    for (const source of sourceList) {
      console.log(`[VolvoxAI] Loading Safetensors model from ${source}...`);
      const loadSafetensors = async () => {
        const response = await fetchImpl(source);
        if (!response.ok) throw new Error(`Failed to load safetensors: ${response.statusText}`);
        if (typeof response.arrayBuffer !== 'function') {
          throw new Error('[GraphLoader] Safetensors response must provide arrayBuffer().');
        }
        const buffer = await response.arrayBuffer();
        return SafetensorsFile.fromArrayBuffer(buffer, safetensorsOptions);
      };
      const safetensors = safetensorsCache
        ? await safetensorsCache.load(source, loadSafetensors)
        : await loadSafetensors();
      weightFiles.push(safetensors);
    }
    const companionWeightsQuantization = usesCompanionScales
      ? mergeCompanionScaleQuantization(weightFiles, sourceList)
      : null;
    const companionScaleState = usesCompanionScales
      ? hydrateCompanionScaleQuantization(companionWeightsQuantization!, weightFiles)
      : null;
    const resolvedWeightsQuantization = companionScaleState?.hydrated ?? inlineWeightsQuantization;

    const tensorsMap = /* @__PURE__ */ new Map<string, Tensor>();
    const stagedSourceTensors = /* @__PURE__ */ new Map<string, Tensor>();
    const loadedWeightNames = new Set<string>();
    graph._assertTopologyMutationAllowed();
    for (const safetensors of weightFiles) {
      for (const [name, entry] of safetensors.tensorEntries()) {
        if (companionScaleState?.companionNames.has(name)) continue;
        const dtype = SafetensorsFile.toGraphDType(entry.dtype);
        if (stagedSourceTensors.has(name)) {
          throw new Error(`[GraphLoader] Tensor '${name}' occurs more than once in the loaded sources.`);
        }
        const tensor = graph._createTensor(name, entry.shape, dtype, {
          isWeight: true,
          buffer: safetensors.toRuntimeTypedArray(entry),
          quantization: resolvedWeightsQuantization?.[name] as QuantizationValue,
        });
        stagedSourceTensors.set(name, tensor);
        tensorsMap.set(name, tensor);
        loadedWeightNames.add(name);
      }
    }
    for (const name of Object.keys(resolvedWeightsQuantization || {})) {
      if (!loadedWeightNames.has(name)) {
        throw new Error(`[GraphLoader] weights_quantization declares unknown loaded weight '${name}'.`);
      }
    }
    if (config.inputs) {
      const inputsDef = config.inputs;
      if (!isRecord(inputsDef)) {
        throw new Error('[GraphLoader] config.inputs must be an object.');
      }
      for (const [name, info] of Object.entries(inputsDef)) {
        if (companionScaleState?.companionNames.has(name)) {
          throw new Error(
            `[GraphLoader] Companion scale '${name}' is reserved storage and cannot be exposed as a graph tensor.`,
          );
        }
        if (stagedSourceTensors.has(name)) throw new Error(`Tensor '${name}' already exists.`);
        if (!isRecord(info)) {
          throw new Error(`[GraphLoader] Input '${name}' must be an object.`);
        }
        const tensor = graph._createTensor(
          name,
          info.shape as readonly number[],
          (info.dtype || "float32") as RuntimeDType,
          {
          isInput: true,
          quantization: info.quantization as QuantizationValue,
          },
        );
        stagedSourceTensors.set(name, tensor);
        tensorsMap.set(name, tensor);
      }
    }

    if (config.nodes) {
      GraphLoader._buildFromBlueprint(graph, config, tensorsMap, {
        baseTensors: stagedSourceTensors,
        reservedNames: companionScaleState?.companionNames,
      });
    } else if (config.model_type) {
      const modelType = config.model_type;
      if (typeof modelType !== 'string' || !GraphLoader.ModelBuilders[modelType]) {
        throw new Error(`[VolvoxAI] Unsupported Hugging Face model_type: '${String(modelType)}'. No builder registered.`);
      }
      graph._assertValidState([], stagedSourceTensors, []);
      graph._commitTopology([], stagedSourceTensors, 'load graph sources');
      console.log(`[VolvoxAI] Building graph on the fly using model builder for '${modelType}'...`);
      GraphLoader.ModelBuilders[modelType](graph, config, tensorsMap);
    } else {
      throw new Error("[VolvoxAI] config.json must contain either 'nodes' (Volvox blueprint) or 'model_type' (Hugging Face).");
    }

    for (const name of companionScaleState?.companionNames || []) {
      if (graph.tensors.has(name)) {
        throw new Error(
          `[GraphLoader] Companion scale '${name}' is reserved storage and cannot be exposed as a graph tensor.`,
        );
      }
    }

    graph.weightFiles = weightFiles;

    GraphLoader._assertBrowserQuantizationSupported(graph);
    GraphLoader._dequantizeConvWeights(graph);
    GraphLoader._normalizeConvWeightsForImageLayout(graph);
    
    // `_buildFromBlueprint` refreshes inferred leaves and, when present,
    // applies the blueprint's explicit `outputs` selection.  Do not recompute
    // the leaves here: doing so would silently discard a package author's
    // requested public outputs.
    const outputNames = [...graph.outputNames];
    GraphLoader._resolveMatMulLayouts(graph);
    graph.assertValid();
    console.log(`[VolvoxAI] Successfully assembled graph. Outputs:`, outputNames);
    return graph;
  }

  static _resolveMatMulLayouts(graph: Graph): unknown {
    return GraphOperatorNormalizer.resolveMatMulLayouts(graph);
  }

  static _assertBrowserQuantizationSupported(
    graph: Graph | { nodes: ConcreteNode[] },
  ): unknown {
    return validatePortableQuantizedGraph(graph);
  }

  static _buildFromBlueprint(
    graph: Graph,
    config: UnknownRecord,
    tensorsMap: Map<string, Tensor>,
    options: BlueprintBuildOptions = {},
  ): void {
    const nodesDef = config.nodes;
    if (!Array.isArray(nodesDef)) {
      throw new Error('[GraphLoader] config.nodes must be an array.');
    }

    // Validate the complete name topology before mutating the graph. Blueprint
    // outputs are definitions, never aliases: they cannot overwrite an input,
    // weight, or earlier node output, and inputs may only reference sources or
    // values produced by an earlier node.
    const baseTensors = options.baseTensors || graph.tensors;
    if (!(baseTensors instanceof Map)) {
      throw new Error('[GraphLoader] Blueprint baseTensors must be a Map.');
    }
    const reservedNames = options.reservedNames || new Set();
    const availableNames = new Set(baseTensors.keys());
    for (const [nodeIndex, nodeDef] of nodesDef.entries()) {
      if (!isRecord(nodeDef)) {
        throw new Error(`[GraphLoader] Node at index ${nodeIndex} must be an object.`);
      }
      const nodeLabel = String(nodeDef.id ?? nodeDef.opType ?? nodeDef.op ?? '<unnamed>');
      if (!isRecord(nodeDef.inputs)) {
        throw new Error(`[GraphLoader] Node ${nodeLabel} inputs must be an object.`);
      }
      if (!isRecord(nodeDef.outputs) || Object.keys(nodeDef.outputs).length === 0) {
        throw new Error(`[GraphLoader] Node ${nodeLabel} requires at least one output.`);
      }
      const nodeInputs = nodeDef.inputs;
      const nodeOutputs = nodeDef.outputs;
      for (const [key, tensorName] of Object.entries(nodeInputs)) {
        if (typeof tensorName !== 'string' || !availableNames.has(tensorName)) {
          throw new Error(`[GraphLoader] Node ${nodeLabel} input '${key}' references undeclared tensor '${String(tensorName)}'.`);
        }
      }
      for (const [key, tensorName] of Object.entries(nodeOutputs)) {
        if (typeof tensorName !== 'string' || tensorName.trim().length === 0) {
          throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' requires a tensor name.`);
        }
        if (availableNames.has(tensorName)) {
          throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' collides with existing tensor '${tensorName}'.`);
        }
        if (reservedNames.has(tensorName)) {
          throw new Error(
            `[GraphLoader] Companion scale '${tensorName}' is reserved storage and cannot be exposed as a graph tensor.`,
          );
        }
        availableNames.add(tensorName);
      }
    }

    let explicitOutputNames: string[] | null = null;
    if (Object.prototype.hasOwnProperty.call(config, 'outputs')) {
      const outputCandidates = Array.isArray(config.outputs)
        ? [...config.outputs]
        : isRecord(config.outputs)
          ? Object.values(config.outputs)
          : null;
      if (!outputCandidates ||
          outputCandidates.some((name) => typeof name !== 'string' || !availableNames.has(name)) ||
          new Set(outputCandidates).size !== outputCandidates.length) {
        throw new Error(
          '[GraphLoader] config.outputs must be an array of unique graph tensor names ' +
          'or a legacy alias object with unique graph tensor-name values.',
        );
      }
      explicitOutputNames = outputCandidates as string[];
    }

    // Stage every node against local collections, validate the complete graph
    // once, then commit once. Calling graph.addOp() here used to validate every
    // previously loaded weight and node after each insertion, turning a model
    // import into O(nodes * tensors) quantization validation.
    graph._assertTopologyMutationAllowed();
    const stagedNodes = [...graph.nodes];
    const stagedTensors = new Map<string, Tensor>(baseTensors);
    const stagedAliases = new Map<string, Tensor>(tensorsMap);
    const stagedOutputs: Array<[string, Tensor]> = [];
    for (const nodeDef of nodesDef) {
      // The first validation pass above establishes these object shapes.
      if (!isRecord(nodeDef) || !isRecord(nodeDef.inputs) || !isRecord(nodeDef.outputs)) {
        throw new Error('[GraphLoader] Blueprint node shape changed during validation.');
      }
      const nodeLabel = String(nodeDef.id ?? nodeDef.opType ?? nodeDef.op ?? '<unnamed>');
      const nodeInputs = nodeDef.inputs;
      const nodeOutputs = nodeDef.outputs;
      const inputs: Record<string, Tensor> = {};
      for (const [key, tName] of Object.entries(nodeDef.inputs)) {
        if (typeof tName !== 'string') {
          throw new Error(`[GraphLoader] Node ${nodeLabel} input '${key}' requires a tensor name.`);
        }
        const t = stagedAliases.get(tName) || stagedTensors.get(tName);
        if (!t) {
          throw new Error(`[GraphLoader] Node ${nodeLabel} input '${key}' references undeclared tensor '${tName}'.`);
        }
        inputs[key] = t;
      }
      const outputsShape = isRecord(nodeDef.outputs_shape) ? nodeDef.outputs_shape : {};
      const outputsDtype = nodeDef.outputs_dtype;
      const outputsQuantization = nodeDef.outputs_quantization;
      const hasOutputsDtype = Object.prototype.hasOwnProperty.call(nodeDef, 'outputs_dtype');
      const hasOutputsQuantization = Object.prototype.hasOwnProperty.call(nodeDef, 'outputs_quantization');
      if (hasOutputsDtype && !isRecord(outputsDtype)) {
        throw new Error(`[GraphLoader] Node ${nodeLabel} outputs_dtype must be an object.`);
      }
      if (hasOutputsDtype) {
        if (!isRecord(outputsDtype)) {
          throw new Error(`[GraphLoader] Node ${nodeLabel} outputs_dtype must be an object.`);
        }
        for (const key of Object.keys(outputsDtype)) {
          if (!Object.prototype.hasOwnProperty.call(nodeOutputs, key)) {
            throw new Error(`[GraphLoader] Node ${nodeLabel} outputs_dtype declares unknown output '${key}'.`);
          }
        }
      }
      if (hasOutputsQuantization && !isRecord(outputsQuantization)) {
        throw new Error(`[GraphLoader] Node ${nodeLabel} outputs_quantization must be an object.`);
      }
      if (hasOutputsQuantization) {
        if (!isRecord(outputsQuantization)) {
          throw new Error(`[GraphLoader] Node ${nodeLabel} outputs_quantization must be an object.`);
        }
        for (const key of Object.keys(outputsQuantization)) {
          if (!Object.prototype.hasOwnProperty.call(nodeOutputs, key)) {
            throw new Error(`[GraphLoader] Node ${nodeLabel} outputs_quantization declares unknown output '${key}'.`);
          }
        }
      }
      const outputDescriptors: Record<string, NodeOutputSpec<Tensor>> = {};
      for (const [key, tensorNameValue] of Object.entries(nodeOutputs)) {
        if (typeof tensorNameValue !== 'string') {
          throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' requires a tensor name.`);
        }
        const tensorName = tensorNameValue;
        const shape = outputsShape[key];
        if (!hasOutputsDtype) {
          if (isRecord(outputsQuantization) && outputsQuantization[key] != null) {
            throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' quantization requires an explicit outputs_dtype.`);
          }
          // Existing blueprints only declare shapes. Preserve their historic F32
          // default while still allowing the descriptor form Graph already accepts.
          if (Array.isArray(shape)) {
            outputDescriptors[key] = { name: tensorName, shape: shape as number[] };
          } else if (isRecord(shape)) {
            if (shape.name != null && shape.name !== tensorName) {
              throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' has conflicting tensor names.`);
            }
            outputDescriptors[key] = { ...shape, name: tensorName } as NodeOutputSpec<Tensor>;
          } else {
            throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' requires a shape.`);
          }
          continue;
        }
        const dtype = isRecord(outputsDtype) ? outputsDtype[key] : undefined;
        if (typeof dtype !== 'string' || !dtype) {
          throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' requires a declared dtype.`);
        }
        try {
          Tensor.dtypeBytes(dtype as RuntimeDType);
        } catch {
          throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' has unsupported dtype '${dtype}'.`);
        }
        if (Array.isArray(shape)) {
          outputDescriptors[key] = {
            name: tensorName,
            shape: shape as number[],
            dtype: dtype as RuntimeDType,
            ...(isRecord(outputsQuantization) && outputsQuantization[key] != null
              ? { quantization: outputsQuantization[key] as QuantizationValue }
              : {}),
          } as NodeOutputSpec<Tensor>;
        } else if (isRecord(shape)) {
          if (shape.name != null && shape.name !== tensorName) {
            throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' has conflicting tensor names.`);
          }
          if (shape.dtype != null && shape.dtype !== dtype) {
            throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' has conflicting shape and outputs_dtype dtypes.`);
          }
          if (shape.quantization != null &&
              isRecord(outputsQuantization) && outputsQuantization[key] != null) {
            throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' cannot declare quantization in both its shape descriptor and outputs_quantization.`);
          }
          outputDescriptors[key] = {
            ...shape,
            name: tensorName,
            dtype: dtype as RuntimeDType,
            ...(isRecord(outputsQuantization) && outputsQuantization[key] != null
              ? { quantization: outputsQuantization[key] as QuantizationValue }
              : {}),
          } as NodeOutputSpec<Tensor>;
        } else {
          throw new Error(`[GraphLoader] Node ${nodeLabel} output '${key}' requires a shape.`);
        }
      }
      const opName = nodeDef.opType || nodeDef.op;
      const node = graph._prepareNode({
        opType: opName as string,
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
    GraphLoader._assertBrowserQuantizationSupported({ nodes: stagedNodes });
    graph._runTopologyTransaction(() => {
      // Install an explicit public-output selection before the only commit so
      // _refreshOutputNames preserves it without a second topology revision.
      if (explicitOutputNames) {
        graph.outputNames = [...explicitOutputNames];
        graph._outputsExplicit = true;
      }
      graph._commitTopology(stagedNodes, stagedTensors, 'load graph blueprint');
      for (const node of stagedNodes) {
        if (typeof node.id === 'number' && node.id >= graph._nextNodeId) {
          graph._nextNodeId = node.id + 1;
        }
      }
    });
    for (const [name, value] of stagedOutputs) tensorsMap.set(name, value);
  }

  // Registry for Hugging Face model builders
  static ModelBuilders: Record<string, ModelBuilder> = {};

  static _dequantizeConvWeights(graph: Graph): unknown {
    return GraphOperatorNormalizer.dequantizeConvWeights(graph);
  }

  static _normalizeConvWeightsForImageLayout(graph: Graph): unknown {
    return GraphOperatorNormalizer.normalizeConvWeightsForImageLayout(graph);
  }

};
