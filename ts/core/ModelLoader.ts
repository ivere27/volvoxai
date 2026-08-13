import type { RuntimeTypedArray, TensorQuantization } from '../types.js';
import {
  assertLosslessJSONValue,
  parseStrictJSON,
  SAFETENSORS_OPEN_READ_WRITE,
  SafetensorsFile,
} from './Safetensors.js';
import type {
  SafetensorsDType,
  SafetensorsOpenOptions,
} from './Safetensors.js';
import {
  VOLVOX_LOGICAL_GRAPH_FORMAT,
  parseGraphDocument,
} from './Graph.js';
import type {
  AffineQuantizationReference,
  Graph,
  WeightDescriptorInput,
} from './Graph.js';

interface GraphFetchResponse {
  readonly ok: boolean;
  readonly statusText?: string;
  readonly text?: () => Promise<string>;
  readonly json?: () => Promise<unknown>;
  readonly arrayBuffer?: () => Promise<ArrayBuffer>;
}

/** Fetch subset used by the package loader and by injected test/storage adapters. */
export type GraphFetch = (source: string) => Promise<GraphFetchResponse>;

/**
 * Structural form of ReadOnlySafetensorsCache. Keeping this contract local
 * avoids importing the legacy GraphLoader module (and therefore Graph/Tensor)
 * into the logical-model loading path. Existing ReadOnlySafetensorsCache
 * instances satisfy it directly.
 */
export interface SafetensorsCache {
  load(
    source: string,
    loader: () => SafetensorsFile | Promise<SafetensorsFile>,
  ): Promise<SafetensorsFile>;
}

export interface ModelLoadOptions {
  readonly graphUrl?: string;
  readonly fetch?: GraphFetch;
  readonly safetensors?: SafetensorsOpenOptions;
  readonly safetensorsCache?: SafetensorsCache;
}

export type WeightRole = 'model_weight' | 'quantization_parameter';

/**
 * Immutable fixed-weight descriptor. `data` and both copy methods always
 * return defensive copies; the loader's owned storage is never exposed.
 */
export interface LoadedWeight {
  readonly name: string;
  readonly dtype: WeightDescriptorInput['dtype'];
  readonly shape: readonly number[];
  readonly sizeBytes: number;
  readonly safetensorsSizeBytes: number;
  readonly safetensorsDtype: SafetensorsDType;
  readonly source: string;
  readonly sourceIndex: number;
  readonly role: WeightRole;
  readonly data: RuntimeTypedArray;
  copyData(): RuntimeTypedArray;
  copyBytes(): Uint8Array;
}

export interface LoadedWeightFile {
  readonly source: string;
  readonly sourceIndex: number;
  readonly sizeBytes: number;
  readonly metadata: Readonly<Record<string, string>>;
  readonly tensorNames: readonly string[];
}

export interface ModelPackageSource {
  readonly graphUrl: string;
  readonly weightSources: readonly string[];
}

export interface ModelPackage {
  readonly graph: Graph;
  readonly weights: Readonly<Record<string, LoadedWeight>>;
  readonly quantizationByTensor: Readonly<Record<string, TensorQuantization>>;
  readonly quantizationParameterNames: readonly string[];
  readonly weightFiles: readonly LoadedWeightFile[];
  readonly source: ModelPackageSource;
}

interface StagedWeight {
  readonly name: string;
  readonly dtype: WeightDescriptorInput['dtype'];
  readonly shape: readonly number[];
  readonly safetensorsDtype: SafetensorsDType;
  readonly safetensorsSizeBytes: number;
  readonly source: string;
  readonly sourceIndex: number;
  readonly ownedData: RuntimeTypedArray;
}

const UTF8_ENCODER = new TextEncoder();

function compareCanonicalNames(left: string, right: string): number {
  const leftBytes = UTF8_ENCODER.encode(left);
  const rightBytes = UTF8_ENCODER.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  return leftBytes.length - rightBytes.length || (left < right ? -1 : left > right ? 1 : 0);
}

function hasOwn(value: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isCanonicalGraphDocumentUrl(value: string): boolean {
  const suffix = value.search(/[?#]/);
  const path = suffix === -1 ? value : value.slice(0, suffix);
  const basename = path.slice(Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\')) + 1);
  return basename === 'graph.json' ||
    (basename.length > '.graph.json'.length && basename.endsWith('.graph.json'));
}

function cloneRuntimeData(data: RuntimeTypedArray): RuntimeTypedArray {
  if (data instanceof Float32Array) return new Float32Array(data);
  if (data instanceof Int32Array) return new Int32Array(data);
  if (data instanceof Int8Array) return new Int8Array(data);
  if (data instanceof Uint8ClampedArray) return new Uint8ClampedArray(data);
  return new Uint8Array(data);
}

function copyRuntimeBytes(data: RuntimeTypedArray): Uint8Array {
  return new Uint8Array(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
}

function canonicalScale(value: number, targetName: string, index: number): number {
  const f32 = Math.fround(value);
  if (!Number.isFinite(value) || value <= 0 || !Number.isFinite(f32) || f32 <= 0 || f32 !== value) {
    throw new Error(
      `[ModelLoader] Quantization scale ${index} for '${targetName}' ` +
      'must be finite, positive, and canonically representable as F32.',
    );
  }
  return f32;
}

function canonicalZeroPoint(
  value: number,
  targetName: string,
  index: number,
  dtype: 'int8' | 'uint8',
): number {
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(
      `[ModelLoader] Quantization zero point ${index} for '${targetName}' ` +
      `must be an integer in [${minimum}, ${maximum}].`,
    );
  }
  return value;
}

function requireParameter(
  weights: ReadonlyMap<string, StagedWeight>,
  name: string,
  targetName: string,
  kind: 'scale' | 'zero-point',
): StagedWeight {
  const weight = weights.get(name);
  if (!weight) {
    throw new Error(
      `[ModelLoader] Quantization ${kind} tensor '${name}' for '${targetName}' is missing.`,
    );
  }
  return weight;
}

function hydrateQuantization(
  graph: Graph,
  weights: ReadonlyMap<string, StagedWeight>,
): {
  readonly quantizationByTensor: Readonly<Record<string, TensorQuantization>>;
  readonly parameterNames: ReadonlySet<string>;
} {
  if (graph.quantization === null) {
    return Object.freeze({
      quantizationByTensor: Object.freeze({}) as Readonly<Record<string, TensorQuantization>>,
      parameterNames: new Set<string>(),
    });
  }

  const parameterNames = new Set<string>();
  const entries: Array<readonly [string, TensorQuantization]> = [];
  for (const targetName of Object.keys(graph.quantization.tensors)) {
    const reference: AffineQuantizationReference = graph.quantization.tensors[targetName];
    const target = graph.tensors[targetName];
    if (target.dtype !== 'int8' && target.dtype !== 'uint8') {
      // Graph already establishes this invariant. Keep this check local
      // so hydration never trusts a malformed programmatic object.
      throw new Error(`[ModelLoader] Quantization target '${targetName}' has invalid storage.`);
    }
    const scale = requireParameter(weights, reference.scale_tensor, targetName, 'scale');
    const zeroPoint = requireParameter(
      weights,
      reference.zero_point_tensor,
      targetName,
      'zero-point',
    );
    if (scale.safetensorsDtype !== 'F32' || !(scale.ownedData instanceof Float32Array)) {
      throw new Error(
        `[ModelLoader] Scale tensor '${scale.name}' for '${targetName}' must use canonical F32 storage.`,
      );
    }
    const expectedZeroDtype = target.dtype === 'int8' ? 'I8' : 'U8';
    const zeroDataMatches = target.dtype === 'int8'
      ? zeroPoint.ownedData instanceof Int8Array
      : zeroPoint.ownedData instanceof Uint8Array &&
        !(zeroPoint.ownedData instanceof Uint8ClampedArray);
    if (zeroPoint.safetensorsDtype !== expectedZeroDtype || !zeroDataMatches) {
      throw new Error(
        `[ModelLoader] Zero-point tensor '${zeroPoint.name}' for '${targetName}' ` +
        `must use ${expectedZeroDtype} storage.`,
      );
    }

    const scales = Array.from(scale.ownedData, (value, index) =>
      canonicalScale(value, targetName, index));
    const zeroPoints = Array.from(zeroPoint.ownedData, (value, index) =>
      canonicalZeroPoint(value, targetName, index, target.dtype as 'int8' | 'uint8'));
    let hydrated: TensorQuantization;
    if (reference.scheme === 'per_tensor') {
      if (scales.length !== 1 || zeroPoints.length !== 1) {
        throw new Error(
          `[ModelLoader] Per-tensor parameters for '${targetName}' must contain one value.`,
        );
      }
      hydrated = Object.freeze({
        scheme: 'per_tensor',
        scale: scales[0],
        zero_point: zeroPoints[0],
      });
    } else {
      const extent = target.shape[reference.axis];
      if (typeof extent !== 'number' || scales.length !== extent || zeroPoints.length !== extent) {
        throw new Error(
          `[ModelLoader] Per-axis parameters for '${targetName}' must contain ${String(extent)} values.`,
        );
      }
      hydrated = Object.freeze({
        scheme: 'per_axis',
        axis: reference.axis,
        scales: Object.freeze(scales),
        zero_points: Object.freeze(zeroPoints),
      });
    }
    entries.push([targetName, hydrated]);
    parameterNames.add(scale.name);
    parameterNames.add(zeroPoint.name);
  }

  return Object.freeze({
    quantizationByTensor: Object.freeze(Object.fromEntries(entries)),
    parameterNames,
  });
}

function createLoadedWeight(weight: StagedWeight, role: WeightRole): LoadedWeight {
  const ownedData = cloneRuntimeData(weight.ownedData);
  const shape = Object.freeze([...weight.shape]);
  return Object.freeze({
    name: weight.name,
    dtype: weight.dtype,
    shape,
    sizeBytes: ownedData.byteLength,
    safetensorsSizeBytes: weight.safetensorsSizeBytes,
    safetensorsDtype: weight.safetensorsDtype,
    source: weight.source,
    sourceIndex: weight.sourceIndex,
    role,
    get data(): RuntimeTypedArray {
      return cloneRuntimeData(ownedData);
    },
    copyData(): RuntimeTypedArray {
      return cloneRuntimeData(ownedData);
    },
    copyBytes(): Uint8Array {
      return copyRuntimeBytes(ownedData);
    },
  });
}

const GRAPH_ROOT_FIELDS = new Set([
  'format', 'dimensions', 'inputs', 'nodes', 'outputs', 'banks', 'quantization',
]);

function assertGraphDocumentContract(document: unknown): void {
  const required = ['format', 'dimensions', 'inputs', 'nodes', 'outputs'];
  if (!isRecord(document) || document.format !== VOLVOX_LOGICAL_GRAPH_FORMAT ||
      required.some((field) => !hasOwn(document, field)) ||
      Object.keys(document).some((field) => !GRAPH_ROOT_FIELDS.has(field))) {
    // Delegate exact diagnostics to the authoritative parser before weights
    // are fetched. A valid weighted graph cannot be fully parsed yet.
    parseGraphDocument(document);
  }
}

async function decodeGraphDocument(
  response: GraphFetchResponse,
  graphUrl: string,
): Promise<unknown> {
  if (typeof response.text === 'function') {
    const source = await response.text();
    const parsed = parseStrictJSON(source, `Graph '${graphUrl}'`);
    assertLosslessJSONValue(parsed, `Graph '${graphUrl}'`);
    return parsed;
  }
  if (typeof response.json === 'function') {
    // A json()-only adapter has already discarded duplicate-key evidence and
    // is therefore responsible for strict raw parsing. We still reject lossy,
    // cyclic, or otherwise non-JSON decoded values here.
    const parsed = await response.json();
    assertLosslessJSONValue(parsed, `Graph '${graphUrl}'`);
    return parsed;
  }
  throw new Error('[ModelLoader] Graph response must provide text() or json().');
}

/** Dynamic-v1 package loader that creates no Graph, Tensor, or backend state. */
export class ModelLoader {
  static async load(
    sources: string | readonly string[],
    options: ModelLoadOptions = {},
  ): Promise<ModelPackage> {
    const weightSources = Object.freeze(
      typeof sources === 'string' ? [sources] : [...sources],
    );
    if (weightSources.some((source) => typeof source !== 'string' || source.length === 0)) {
      throw new Error('[ModelLoader] Safetensors sources must be non-empty strings.');
    }

    let graphUrl = options.graphUrl;
    if (graphUrl === undefined) {
      if (weightSources.length === 0) {
        throw new Error(
          '[ModelLoader] graphUrl is required when no safetensors sources are supplied.',
        );
      }
      const primarySource = weightSources[0];
      const sourceSuffix = primarySource.search(/[?#]/);
      const sourcePath = sourceSuffix === -1
        ? primarySource
        : primarySource.slice(0, sourceSuffix);
      const sourceDirectoryEnd = sourcePath.lastIndexOf('/') + 1;
      graphUrl = `${sourcePath.slice(0, sourceDirectoryEnd)}graph.json`;
    }
    if (typeof graphUrl !== 'string' || !isCanonicalGraphDocumentUrl(graphUrl)) {
      throw new Error(
        '[ModelLoader] graphUrl must name graph.json or a named *.graph.json document.',
      );
    }

    const fetchImpl: GraphFetch | undefined = options.fetch ??
      (typeof globalThis.fetch === 'function'
        ? async (source: string): Promise<GraphFetchResponse> => globalThis.fetch(source)
        : undefined);
    if (fetchImpl === undefined) {
      throw new Error('[ModelLoader] No fetch implementation is available.');
    }

    const graphResponse = await fetchImpl(graphUrl);
    if (!graphResponse.ok) {
      throw new Error(
        `[ModelLoader] Failed to load graph '${graphUrl}': ` +
        `${graphResponse.statusText ?? 'unknown error'}`,
      );
    }
    const graphDocument = await decodeGraphDocument(graphResponse, graphUrl);
    assertGraphDocumentContract(graphDocument);

    const safetensorsOptions: SafetensorsOpenOptions = Object.freeze({
      ...(options.safetensors?.flags === undefined
        ? {}
        : { flags: options.safetensors.flags }),
      ...(options.safetensors?.writable === undefined
        ? {}
        : { writable: options.safetensors.writable }),
    });
    const safetensorsFlags = safetensorsOptions.flags ??
      (safetensorsOptions.writable ? SAFETENSORS_OPEN_READ_WRITE : 0);
    const cache = options.safetensorsCache;
    if (cache !== undefined &&
        (cache === null || typeof cache !== 'object' || typeof cache.load !== 'function')) {
      throw new Error('[ModelLoader] safetensorsCache must provide a read-only load() method.');
    }
    if (cache !== undefined && (safetensorsFlags & SAFETENSORS_OPEN_READ_WRITE)) {
      throw new Error(
        '[ModelLoader] safetensorsCache cannot be used with writable safetensors options.',
      );
    }

    const stagedWeights = new Map<string, StagedWeight>();
    const weightFiles: LoadedWeightFile[] = [];
    for (let sourceIndex = 0; sourceIndex < weightSources.length; sourceIndex++) {
      const source = weightSources[sourceIndex];
      const loadFile = async (): Promise<SafetensorsFile> => {
        const response = await fetchImpl(source);
        if (!response.ok) {
          throw new Error(
            `[ModelLoader] Failed to load safetensors '${source}': ` +
            `${response.statusText ?? 'unknown error'}`,
          );
        }
        if (typeof response.arrayBuffer !== 'function') {
          throw new Error('[ModelLoader] Safetensors response must provide arrayBuffer().');
        }
        return SafetensorsFile.fromArrayBuffer(
          await response.arrayBuffer(),
          safetensorsOptions,
        );
      };
      const file = cache === undefined ? await loadFile() : await cache.load(source, loadFile);
      if (!(file instanceof SafetensorsFile)) {
        throw new Error('[ModelLoader] Safetensors cache returned an invalid file.');
      }
      if (cache !== undefined && (file.flags & SAFETENSORS_OPEN_READ_WRITE)) {
        throw new Error('[ModelLoader] Safetensors cache returned a writable file.');
      }
      for (const forbidden of ['weights_quantization', 'weights_quantization_storage']) {
        if (hasOwn(file.metadata, forbidden)) {
          throw new Error(
            `[ModelLoader] Safetensors source '${source}' contains forbidden ` +
            `legacy metadata '${forbidden}'.`,
          );
        }
      }

      const tensorNames: string[] = [];
      for (const [name, entry] of file.tensorEntries()) {
        if (stagedWeights.has(name)) {
          throw new Error(
            `[ModelLoader] Tensor '${name}' occurs in more than one safetensors source.`,
          );
        }
        const dtype = SafetensorsFile.toGraphDType(entry.dtype);
        const runtimeData = file.toRuntimeTypedArray(entry);
        const ownedData = cloneRuntimeData(runtimeData);
        const staged: StagedWeight = Object.freeze({
          name,
          dtype,
          shape: Object.freeze([...entry.shape]),
          safetensorsDtype: entry.dtype,
          safetensorsSizeBytes: entry.sizeBytes,
          source,
          sourceIndex,
          ownedData,
        });
        stagedWeights.set(name, staged);
        tensorNames.push(name);
      }
      tensorNames.sort(compareCanonicalNames);
      const inspection = file.inspect();
      weightFiles.push(Object.freeze({
        source,
        sourceIndex,
        sizeBytes: inspection.sizeBytes,
        metadata: Object.freeze({ ...file.metadata }),
        tensorNames: Object.freeze(tensorNames),
      }));
    }

    const weightDescriptors = [...stagedWeights.values()].map((weight) => Object.freeze({
      name: weight.name,
      dtype: weight.dtype,
      shape: Object.freeze([...weight.shape]),
    }));
    const graph = parseGraphDocument(graphDocument, weightDescriptors);
    const { quantizationByTensor, parameterNames } = hydrateQuantization(graph, stagedWeights);

    const weightEntries: Array<readonly [string, LoadedWeight]> = [];
    for (const name of Object.keys(graph.weights)) {
      const staged = stagedWeights.get(name);
      if (!staged) {
        throw new Error(`[ModelLoader] Logical weight '${name}' has no owned payload.`);
      }
      weightEntries.push([
        name,
        createLoadedWeight(
          staged,
          parameterNames.has(name) ? 'quantization_parameter' : 'model_weight',
        ),
      ]);
    }
    const quantizationParameterNames = Object.freeze(
      [...parameterNames].sort(compareCanonicalNames),
    );

    return Object.freeze({
      graph,
      weights: Object.freeze(Object.fromEntries(weightEntries)),
      quantizationByTensor,
      quantizationParameterNames,
      weightFiles: Object.freeze(weightFiles),
      source: Object.freeze({
        graphUrl,
        weightSources,
      }),
    });
  }
}
