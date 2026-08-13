import type {
  RuntimeDType,
  RuntimeTypedArray,
  TensorQuantization,
} from '../types.js';
import {
  checkedTensorByteLength,
  type ShapedExecutionTensorView,
} from '../ops/shapeSystem.js';
import { inspectDeviceTensorReference } from '../ops/deviceTensorReference.js';
import { runtimeIdentity } from './Identity.js';
import {
  parseGraphDocument,
  type AffineQuantizationReference,
  type Graph,
  type InputDescriptor,
  type WeightBankSpec,
  type JsonValue,
  type TensorDescriptor,
} from './Graph.js';
import { ModelLoader } from './ModelLoader.js';
import type {
  ModelLoadOptions,
  ModelPackage,
  WeightRole,
} from './ModelLoader.js';
import {
  ResolvedShapePlanError,
  proveGraphShapeDomain,
  resolveGraphShapes,
  resolveStaticGraphShapes,
  type AcceptedGraphShapeDomainProof,
  type HydratedTensorQuantization,
  type ResolvedShapePlan,
} from './ResolvedShapePlan.js';

export type ModelErrorCode =
  | 'INVALID_SOURCE'
  | 'WEIGHT_SET_MISMATCH'
  | 'WEIGHT_PAYLOAD_MISMATCH'
  | 'SHAPE_DOMAIN_UNSUPPORTED'
  | 'STATIC_PREBIND_FAILED';

/** A model-load/capture failure raised before Tensor or backend allocation. */
export class ModelError extends Error {
  readonly code: ModelErrorCode;
  readonly path: string;

  constructor(
    code: ModelErrorCode,
    path: string,
    message: string,
    cause?: unknown,
  ) {
    super(`[Model] ${path}: ${message}`, cause === undefined ? undefined : { cause });
    this.name = 'ModelError';
    this.code = code;
    this.path = path;
  }
}

/** Minimal fixed-weight source accepted in addition to a loaded package. */
export interface ModelWeightSource {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly data: RuntimeTypedArray;
  readonly role?: WeightRole;
  copyData?(): RuntimeTypedArray;
}

/**
 * Structural capture source for programmatic authoring. A
 * ModelPackage satisfies this interface directly.
 */
export interface ModelSource {
  readonly graph: Graph;
  readonly weights: Readonly<Record<string, ModelWeightSource>>;
  readonly quantizationByTensor?: HydratedTensorQuantization;
}

export interface ModelWeightDescriptor {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly sizeBytes: number;
  readonly role: WeightRole;
  /** Non-null when axis 0 indexes runtime-selected slots (LoRA/MoE bank). */
  readonly bank: WeightBankSpec | null;
}

interface CapturedLogicalWeight extends ModelWeightDescriptor {
  readonly ownedData: RuntimeTypedArray;
}

const UTF8_ENCODER = new TextEncoder();

function compareCanonicalNames(left: string, right: string): number {
  if (left === right) return 0;
  const leftBytes = UTF8_ENCODER.encode(left);
  const rightBytes = UTF8_ENCODER.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  return leftBytes.length - rightBytes.length;
}

function sortedNames(value: object): string[] {
  return Object.getOwnPropertyNames(value).sort(compareCanonicalNames);
}

function hasOwn(value: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) &&
    !ArrayBuffer.isView(value);
}

function fail(
  code: ModelErrorCode,
  path: string,
  message: string,
  cause?: unknown,
): never {
  throw new ModelError(code, path, message, cause);
}

function cloneRuntimeData(data: RuntimeTypedArray): RuntimeTypedArray {
  if (data instanceof Float32Array) return new Float32Array(data);
  if (data instanceof Int32Array) return new Int32Array(data);
  if (data instanceof Int8Array) return new Int8Array(data);
  if (data instanceof Uint8ClampedArray) return new Uint8ClampedArray(data);
  return new Uint8Array(data);
}

function runtimeDataDType(data: unknown): RuntimeDType | null {
  if (data instanceof Float32Array) return 'float32';
  if (data instanceof Int32Array) return 'int32';
  if (data instanceof Int8Array) return 'int8';
  if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) return 'uint8';
  return null;
}

function copyRuntimeBytes(data: RuntimeTypedArray): Uint8Array {
  return new Uint8Array(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
}

function cloneJson(value: JsonValue): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'number' ||
      typeof value === 'string') {
    return value;
  }
  if (Array.isArray(value)) return value.map(cloneJson);
  return Object.fromEntries(
    sortedNames(value).map((name) => [name, cloneJson(value[name])]),
  ) as JsonValue;
}

function cloneQuantizationReference(
  reference: AffineQuantizationReference,
): AffineQuantizationReference {
  return reference.scheme === 'per_tensor'
    ? {
      scheme: 'per_tensor',
      scale_tensor: reference.scale_tensor,
      zero_point_tensor: reference.zero_point_tensor,
    }
    : {
      scheme: 'per_axis',
      axis: reference.axis,
      scale_tensor: reference.scale_tensor,
      zero_point_tensor: reference.zero_point_tensor,
    };
}

/** Reparse into a snapshot-owned immutable Graph and recheck its fingerprint. */
function cloneGraph(source: Graph): Graph {
  if (!isRecord(source)) fail('INVALID_SOURCE', 'graph', 'must be a Graph.');
  try {
    const document = {
      format: source.format,
      dimensions: Object.fromEntries(sortedNames(source.dimensions).map((name) => {
        const dimension = source.dimensions[name];
        return [name, {
          min: dimension.min,
          max: dimension.max,
          multiple_of: dimension.multiple_of,
        }];
      })),
      inputs: Object.fromEntries(sortedNames(source.inputs).map((name) => {
        const input = source.inputs[name];
        return [name, { dtype: input.dtype, shape: [...input.shape] }];
      })),
      nodes: source.nodes.map((node) => ({
        id: node.id,
        opType: node.opType,
        inputs: Object.fromEntries(sortedNames(node.inputs).map((port) =>
          [port, node.inputs[port]])),
        outputs: Object.fromEntries(sortedNames(node.outputs).map((port) => {
          const output = node.outputs[port];
          return [port, {
            tensor: output.tensor,
            dtype: output.dtype,
            shape: [...output.shape],
          }];
        })),
        params: cloneJson(node.params),
      })),
      outputs: [...source.outputs],
      ...(Object.keys(source.banks).length === 0
        ? {}
        : {
          banks: Object.fromEntries(sortedNames(source.banks).map((name) =>
            [name, source.banks[name].dimension])),
        }),
      ...(source.quantization === null
        ? {}
        : {
          quantization: {
            format: source.quantization.format,
            tensors: Object.fromEntries(
              sortedNames(source.quantization.tensors).map((name) => [
                name,
                cloneQuantizationReference(source.quantization!.tensors[name]),
              ]),
            ),
          },
        }),
    };
    const weights = sortedNames(source.weights).map((name) => {
      const weight = source.weights[name];
      return { name, dtype: weight.dtype, shape: [...weight.shape] };
    });
    const graph = parseGraphDocument(document, weights);
    if (graph.fingerprint !== source.fingerprint) {
      fail(
        'INVALID_SOURCE',
        'graph.fingerprint',
        'does not match the canonical logical definition.',
      );
    }
    return graph;
  } catch (error) {
    if (error instanceof ModelError) throw error;
    fail(
      'INVALID_SOURCE',
      'graph',
      error instanceof Error ? error.message : String(error),
      error,
    );
  }
}

function assertExactFields(
  source: Readonly<Record<string, unknown>>,
  expected: readonly string[],
  path: string,
): void {
  if (Object.getOwnPropertySymbols(source).length !== 0) {
    fail('INVALID_SOURCE', path, 'must not contain symbol-keyed fields.');
  }
  const expectedSet = new Set(expected);
  const actual = sortedNames(source);
  const missing = expected.filter((name) => !hasOwn(source, name));
  const unexpected = actual.filter((name) => !expectedSet.has(name));
  if (missing.length === 0 && unexpected.length === 0) return;
  const details: string[] = [];
  if (missing.length !== 0) details.push(`missing [${missing.join(', ')}]`);
  if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
  fail('INVALID_SOURCE', path, details.join('; '));
}

function cloneScale(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    fail('INVALID_SOURCE', path, 'must be finite and positive.');
  }
  const scale = Math.fround(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    fail('INVALID_SOURCE', path, 'must be positive and representable as float32.');
  }
  return scale;
}

function cloneZeroPoint(
  value: unknown,
  dtype: 'int8' | 'uint8',
  path: string,
): number {
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  if (!Number.isInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    fail('INVALID_SOURCE', path, `must be an integer in [${minimum}, ${maximum}].`);
  }
  return value === 0 ? 0 : value as number;
}

function denseArray(source: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(source) || source.length === 0) {
    fail('INVALID_SOURCE', path, 'must be a non-empty dense array.');
  }
  for (let index = 0; index < source.length; index++) {
    if (!hasOwn(source, index)) fail('INVALID_SOURCE', `${path}[${index}]`, 'must not be an array hole.');
  }
  return source;
}

function cloneTensorQuantization(
  source: Readonly<Record<string, unknown>>,
  reference: AffineQuantizationReference,
  target: TensorDescriptor,
  path: string,
): TensorQuantization {
  if (target.dtype !== 'int8' && target.dtype !== 'uint8') {
    fail('INVALID_SOURCE', path, `target dtype '${target.dtype}' is not quantized storage.`);
  }
  if (source.scheme !== reference.scheme) {
    fail(
      'INVALID_SOURCE',
      `${path}.scheme`,
      `must equal the logical scheme '${reference.scheme}'.`,
    );
  }
  if (source.scheme === 'per_tensor') {
    assertExactFields(source, ['scheme', 'scale', 'zero_point'], path);
    return Object.freeze({
      scheme: 'per_tensor',
      scale: cloneScale(source.scale, `${path}.scale`),
      zero_point: cloneZeroPoint(source.zero_point, target.dtype, `${path}.zero_point`),
    });
  }
  if (source.scheme !== 'per_axis' || reference.scheme !== 'per_axis') {
    fail(
      'INVALID_SOURCE',
      `${path}.scheme`,
      "must be 'per_tensor' or 'per_axis'.",
    );
  }
  assertExactFields(source, ['scheme', 'axis', 'scales', 'zero_points'], path);
  if (!Number.isInteger(source.axis) || source.axis !== reference.axis) {
    fail('INVALID_SOURCE', `${path}.axis`, `must equal ${reference.axis}.`);
  }
  const extent = target.shape[reference.axis];
  if (typeof extent !== 'number') {
    fail('INVALID_SOURCE', `${path}.axis`, 'cannot target a symbolic extent.');
  }
  const sourceScales = denseArray(source.scales, `${path}.scales`);
  const sourceZeroPoints = denseArray(source.zero_points, `${path}.zero_points`);
  if (sourceScales.length !== extent || sourceZeroPoints.length !== extent) {
    fail(
      'INVALID_SOURCE',
      path,
      `must contain exactly ${extent} scales and zero points.`,
    );
  }
  return Object.freeze({
    scheme: 'per_axis',
    axis: reference.axis,
    scales: Object.freeze(sourceScales.map((scale, index) =>
      cloneScale(scale, `${path}.scales[${index}]`))),
    zero_points: Object.freeze(sourceZeroPoints.map((zeroPoint, index) =>
      cloneZeroPoint(zeroPoint, target.dtype as 'int8' | 'uint8', `${path}.zero_points[${index}]`))),
  });
}

function cloneHydratedQuantization(
  source: unknown,
  graph: Graph,
): HydratedTensorQuantization {
  if (source === undefined) source = {};
  if (!isRecord(source) || Object.getOwnPropertySymbols(source).length !== 0) {
    fail('INVALID_SOURCE', 'quantizationByTensor', 'must be a tensor-name record.');
  }
  const references = graph.quantization?.tensors ?? {};
  const expectedNames = sortedNames(references);
  const actualNames = sortedNames(source);
  const expected = new Set(expectedNames);
  const actual = new Set(actualNames);
  const missing = expectedNames.filter((name) => !actual.has(name));
  const unexpected = actualNames.filter((name) => !expected.has(name));
  if (missing.length !== 0 || unexpected.length !== 0) {
    const details: string[] = [];
    if (missing.length !== 0) details.push(`missing [${missing.join(', ')}]`);
    if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
    fail('INVALID_SOURCE', 'quantizationByTensor', details.join('; '));
  }
  return Object.freeze(Object.fromEntries(expectedNames.map((name) => {
    const quantization = source[name];
    if (!isRecord(quantization) ||
        (quantization.scheme !== 'per_tensor' && quantization.scheme !== 'per_axis')) {
      fail(
        'INVALID_SOURCE',
        `quantizationByTensor.${name}`,
        'must be hydrated per_tensor or per_axis metadata.',
      );
    }
    const target = graph.tensors[name];
    if (target === undefined) {
      fail('INVALID_SOURCE', `quantizationByTensor.${name}`, 'targets a missing tensor.');
    }
    return [
      name,
      cloneTensorQuantization(
        quantization,
        references[name],
        target,
        `quantizationByTensor.${name}`,
      ),
    ];
  })));
}

function sourceWeightData(source: ModelWeightSource, path: string): RuntimeTypedArray {
  const candidate = typeof source.copyData === 'function'
    ? source.copyData()
    : source.data;
  if (runtimeDataDType(candidate) === null) {
    fail('WEIGHT_PAYLOAD_MISMATCH', `${path}.data`, 'must be a supported runtime typed array.');
  }
  return candidate;
}

function captureWeights(
  graph: Graph,
  source: unknown,
): readonly CapturedLogicalWeight[] {
  if (!isRecord(source) || Object.getOwnPropertySymbols(source).length !== 0) {
    fail('INVALID_SOURCE', 'weights', 'must be a tensor-name record.');
  }
  const expectedNames = sortedNames(graph.weights);
  const actualNames = sortedNames(source);
  const expected = new Set(expectedNames);
  const actual = new Set(actualNames);
  const missing = expectedNames.filter((name) => !actual.has(name));
  const unexpected = actualNames.filter((name) => !expected.has(name));
  if (missing.length !== 0 || unexpected.length !== 0) {
    const details: string[] = [];
    if (missing.length !== 0) details.push(`missing [${missing.join(', ')}]`);
    if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
    fail(
      'WEIGHT_SET_MISMATCH',
      'weights',
      `must contain exactly the logical fixed weights; ${details.join('; ')}.`,
    );
  }

  const quantizationParameterNames = new Set<string>();
  for (const reference of Object.values(graph.quantization?.tensors ?? {})) {
    quantizationParameterNames.add(reference.scale_tensor);
    quantizationParameterNames.add(reference.zero_point_tensor);
  }

  return Object.freeze(expectedNames.map((name) => {
    const path = `weights.${name}`;
    const logical = graph.weights[name];
    const candidate = source[name];
    if (!isRecord(candidate)) {
      fail('WEIGHT_PAYLOAD_MISMATCH', path, 'must be a fixed-weight source.');
    }
    const weight = candidate as unknown as ModelWeightSource;
    if (weight.name !== name) {
      fail('WEIGHT_PAYLOAD_MISMATCH', `${path}.name`, `must equal '${name}'.`);
    }
    if (weight.dtype !== logical.dtype) {
      fail(
        'WEIGHT_PAYLOAD_MISMATCH',
        `${path}.dtype`,
        `is '${String(weight.dtype)}', but the logical descriptor requires '${logical.dtype}'.`,
      );
    }
    if (!Array.isArray(weight.shape) || weight.shape.length !== logical.shape.length ||
        weight.shape.some((dimension, axis) => dimension !== logical.shape[axis])) {
      fail(
        'WEIGHT_PAYLOAD_MISMATCH',
        `${path}.shape`,
        `must equal [${logical.shape.join(', ')}].`,
      );
    }
    const role: WeightRole = quantizationParameterNames.has(name)
      ? 'quantization_parameter'
      : 'model_weight';
    if (weight.role !== undefined && weight.role !== role) {
      fail(
        'WEIGHT_PAYLOAD_MISMATCH',
        `${path}.role`,
        `must be '${role}' for the logical definition.`,
      );
    }
    const candidateData = sourceWeightData(weight, path);
    const actualDType = runtimeDataDType(candidateData);
    if (actualDType !== logical.dtype) {
      fail(
        'WEIGHT_PAYLOAD_MISMATCH',
        `${path}.data`,
        `has dtype '${String(actualDType)}', but the logical descriptor requires '${logical.dtype}'.`,
      );
    }
    let sizeBytes: number;
    try {
      sizeBytes = checkedTensorByteLength(logical.shape, logical.dtype, path);
    } catch (error) {
      fail(
        'WEIGHT_PAYLOAD_MISMATCH',
        path,
        error instanceof Error ? error.message : String(error),
        error,
      );
    }
    if (candidateData.byteLength !== sizeBytes) {
      fail(
        'WEIGHT_PAYLOAD_MISMATCH',
        `${path}.data`,
        `has ${candidateData.byteLength} bytes, but the logical descriptor requires ${sizeBytes}.`,
      );
    }
    return Object.freeze({
      name,
      dtype: logical.dtype,
      shape: Object.freeze([...logical.shape]),
      sizeBytes,
      role,
      bank: logical.bank,
      ownedData: cloneRuntimeData(candidateData),
    });
  }));
}

function firstAndLastLegalValue(
  graph: Graph,
  symbol: string,
): readonly [number, number] {
  const constraint = graph.environment.get(symbol);
  if (constraint === undefined) fail('INVALID_SOURCE', 'graph', `references unknown symbol '${symbol}'.`);
  const multiple = constraint.multiple_of ?? 1;
  const remainder = constraint.min % multiple;
  const first = constraint.min + (remainder === 0 ? 0 : multiple - remainder);
  const last = constraint.max - (constraint.max % multiple);
  return [first, last];
}

function hasSinglePublicShape(graph: Graph): boolean {
  for (const input of Object.values(graph.inputs)) {
    for (const dimension of input.shape) {
      if (typeof dimension !== 'string') continue;
      const [first, last] = firstAndLastLegalValue(graph, dimension);
      if (first !== last) return false;
    }
  }
  return true;
}

function staticInputFailure(path: string, message: string): never {
  throw new ResolvedShapePlanError('INPUT_BINDING_FAILED', path, message);
}

/** Allocation-light validation for a snapshot whose complete plan is fixed. */
function validateStaticInputViews(
  inputNames: readonly string[],
  plan: ResolvedShapePlan,
  values: Readonly<Record<string, ShapedExecutionTensorView>>,
): void {
  if (!isRecord(values) || Object.getOwnPropertySymbols(values).length !== 0) {
    staticInputFailure('execution inputs', 'must be a named tensor-view record.');
  }
  const names = Object.getOwnPropertyNames(values);
  if (names.length !== inputNames.length ||
      inputNames.some((name) => !hasOwn(values, name))) {
    staticInputFailure('execution inputs', 'must contain exactly the declared public inputs.');
  }
  for (const name of inputNames) {
    const path = `execution input '${name}'`;
    const value = values[name];
    const descriptor = plan.tensors[name];
    if (!isRecord(value) || !descriptor || descriptor.kind !== 'input') {
      staticInputFailure(path, 'must be an object containing data and shape.');
    }
    const deviceReference = inspectDeviceTensorReference(value.data);
    const dtype = runtimeDataDType(value.data) ?? deviceReference?.dtype ?? null;
    if (dtype !== descriptor.dtype) {
      staticInputFailure(`${path}.data`, `must expose exact ${descriptor.dtype} storage.`);
    }
    if (!Array.isArray(value.shape) || value.shape.length !== descriptor.shape.length) {
      staticInputFailure(`${path}.shape`, `must have rank ${descriptor.shape.length}.`);
    }
    for (let axis = 0; axis < descriptor.shape.length; axis++) {
      if (value.shape[axis] !== descriptor.shape[axis]) {
        staticInputFailure(
          `${path}.shape[${axis}]`,
          `must equal fixed dimension ${descriptor.shape[axis]}.`,
        );
      }
    }
    if (deviceReference && (deviceReference.shape.length !== descriptor.shape.length ||
        deviceReference.shape.some((dimension, axis) =>
          dimension !== descriptor.shape[axis]))) {
      staticInputFailure(`${path}.shape`, 'must match the device result shape exactly.');
    }
    const actualBytes = deviceReference?.logicalSizeBytes ??
      (value.data as RuntimeTypedArray).byteLength;
    if (actualBytes !== descriptor.sizeBytes) {
      staticInputFailure(`${path}.data`, `must contain exactly ${descriptor.sizeBytes} bytes.`);
    }
  }
}

function nextDefinitionId(): string {
  return runtimeIdentity('model-definition');
}

function nextWeightRevisionId(definitionId: string, revision: number): string {
  return `${definitionId}:weight:${revision}:${runtimeIdentity('weight-revision')}`;
}

/**
 * Immutable logical topology plus one exact privately owned fixed-weight
 * revision. It contains no request buffers, Tensor objects, or backend state.
 */
export class Model {
  readonly graph: Graph;
  readonly definitionFingerprint: string;
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly inputNames: readonly string[];
  readonly inputDescriptors: readonly InputDescriptor[];
  readonly outputNames: readonly string[];
  readonly outputDescriptors: readonly TensorDescriptor[];
  readonly weightNames: readonly string[];
  readonly weightDescriptors: readonly ModelWeightDescriptor[];
  readonly tensorCount: number;
  readonly nodeCount: number;
  readonly isStatic: boolean;
  readonly staticShapePlan: ResolvedShapePlan | null;
  readonly shapeDomainProof: AcceptedGraphShapeDomainProof;
  readonly quantizationByTensor: HydratedTensorQuantization;

  readonly #weights: ReadonlyMap<string, CapturedLogicalWeight>;

  private constructor(
    source: ModelSource,
    previous: Model | null,
  ) {
    if (!isRecord(source)) fail('INVALID_SOURCE', 'source', 'must be a logical model source.');
    this.graph = cloneGraph(source.graph);
    this.definitionFingerprint = this.graph.fingerprint;
    const sameDefinition = previous !== null &&
      previous.definitionFingerprint === this.definitionFingerprint;
    this.definitionId = sameDefinition ? previous.definitionId : nextDefinitionId();
    this.topologyRevision = previous === null
      ? 1
      : sameDefinition
        ? previous.topologyRevision
        : previous.topologyRevision + 1;
    this.weightRevision = previous === null ? 1 : previous.weightRevision + 1;
    this.weightRevisionId = nextWeightRevisionId(this.definitionId, this.weightRevision);

    const capturedWeights = captureWeights(this.graph, source.weights);
    this.#weights = new Map(capturedWeights.map((weight) => [weight.name, weight]));
    this.weightNames = Object.freeze(capturedWeights.map((weight) => weight.name));
    this.weightDescriptors = Object.freeze(capturedWeights.map((weight) => Object.freeze({
      name: weight.name,
      dtype: weight.dtype,
      shape: Object.freeze([...weight.shape]),
      sizeBytes: weight.sizeBytes,
      role: weight.role,
      bank: this.graph.weights[weight.name]?.bank ?? null,
    })));
    this.quantizationByTensor = cloneHydratedQuantization(
      source.quantizationByTensor,
      this.graph,
    );

    const proof = proveGraphShapeDomain(this.graph, this.quantizationByTensor);
    if (!proof.supported) {
      fail(
        'SHAPE_DOMAIN_UNSUPPORTED',
        proof.path,
        `${proof.code}: ${proof.reason}`,
      );
    }
    this.shapeDomainProof = proof;

    this.inputNames = Object.freeze(sortedNames(this.graph.inputs));
    this.inputDescriptors = Object.freeze(
      this.inputNames.map((name) => this.graph.inputs[name]),
    );
    this.outputNames = Object.freeze([...this.graph.outputs]);
    this.outputDescriptors = Object.freeze(this.outputNames.map((name) => {
      const descriptor = this.graph.tensors[name];
      if (descriptor === undefined) {
        fail('INVALID_SOURCE', 'graph.outputs', `references missing tensor '${name}'.`);
      }
      return descriptor;
    }));
    this.tensorCount = sortedNames(this.graph.tensors).length;
    this.nodeCount = this.graph.nodes.length;

    this.isStatic = hasSinglePublicShape(this.graph);
    if (this.isStatic) {
      try {
        this.staticShapePlan = resolveStaticGraphShapes(
          this.graph,
          this.quantizationByTensor,
        );
      } catch (error) {
        fail(
          'STATIC_PREBIND_FAILED',
          error instanceof ResolvedShapePlanError ? error.path : 'graph',
          error instanceof Error ? error.message : String(error),
          error,
        );
      }
    } else {
      this.staticShapePlan = null;
    }
    Object.freeze(this);
  }

  /**
   * Fetch a `volvox-graph/v1` package and capture it. This is the ordinary
   * entry point; `capture` stays available for callers that already hold a
   * package or build one in memory.
   */
  static async load(
    sources: string | readonly string[],
    options: ModelLoadOptions = {},
  ): Promise<Model> {
    return Model.capture(await ModelLoader.load(sources, options));
  }

  static capture(source: ModelPackage): Model;
  static capture(source: ModelSource): Model;
  static capture(source: ModelSource): Model {
    return new Model(source, null);
  }

  /**
   * Capture a successor weight revision. Definition identity is retained only
   * when the canonical logical fingerprint, including bounds, is unchanged.
   */
  static derive(
    source: ModelPackage,
    previous: Model,
  ): Model;
  static derive(
    source: ModelSource,
    previous: Model,
  ): Model;
  static derive(
    source: ModelSource,
    previous: Model,
  ): Model {
    if (!(previous instanceof Model)) {
      fail('INVALID_SOURCE', 'previous', 'must be a Model.');
    }
    return new Model(source, previous);
  }

  matchesDefinition(graph: Graph): boolean {
    return isRecord(graph) && graph.fingerprint === this.definitionFingerprint;
  }

  hasWeight(name: string): boolean {
    return this.#weights.has(name);
  }

  /** Return a mutable caller-owned copy; snapshot bytes are never exposed. */
  copyWeightData(name: string): RuntimeTypedArray {
    const weight = this.#weights.get(name);
    if (weight === undefined) throw new Error(`Logical weight '${name}' does not exist.`);
    return cloneRuntimeData(weight.ownedData);
  }

  /** Return a byte-exact mutable caller-owned copy. */
  copyWeightBytes(name: string): Uint8Array {
    const weight = this.#weights.get(name);
    if (weight === undefined) throw new Error(`Logical weight '${name}' does not exist.`);
    return copyRuntimeBytes(weight.ownedData);
  }

  /**
   * Validate and resolve a complete shaped-input set. Static snapshots reuse
   * their pre-resolved metadata after the same dtype/rank/shape/byte checks.
   */
  bindShapes(
    values: Readonly<Record<string, ShapedExecutionTensorView>>,
    bankResidency?: Readonly<Record<string, readonly number[]>>,
  ): ResolvedShapePlan {
    if (!this.isStatic || bankResidency !== undefined) {
      return resolveGraphShapes(
        this.graph, values, this.quantizationByTensor, bankResidency,
      );
    }
    validateStaticInputViews(this.inputNames, this.staticShapePlan!, values);
    return this.staticShapePlan!;
  }
}
