import type { NodeDescriptor, JsonValue } from '../core/Graph.js';
import type {
  LogicalDomainTensorDescriptor,
  LogicalShapeDomainNodeProof,
} from '../core/ResolvedShapePlan.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';
import {
  kernelRoute,
  operatorShapeContract,
  runtimeOperatorsByBackend,
} from '../generated/kernelRegistry.js';
import type { RuntimeDType, TensorQuantization } from '../types.js';
import type { BackendLogicalCompileInput } from './BackendProvider.js';

export interface WebGPUPhysicalDomainProof {
  readonly requiredShaderMethods: readonly string[];
  readonly optionalShaderMethods: readonly string[];
  readonly packedDot4OptionalShaderMethods: readonly string[];
  readonly maximumAuxiliaryBytes: number;
}

interface WebGPULimitsLike {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
  readonly maxUniformBufferBindingSize?: number;
  readonly maxComputeWorkgroupsPerDimension?: number;
  readonly maxComputeInvocationsPerWorkgroup?: number;
  readonly maxComputeWorkgroupSizeX?: number;
  readonly maxComputeWorkgroupSizeY?: number;
  readonly maxComputeWorkgroupSizeZ?: number;
  readonly maxComputeWorkgroupStorageSize?: number;
  readonly maxBindingsPerBindGroup?: number;
  readonly maxBindGroups?: number;
  readonly maxStorageBuffersPerShaderStage?: number;
  readonly maxUniformBuffersPerShaderStage?: number;
}

type PhysicalRouteFamily =
  | 'dense'
  | 'qdense'
  | 'batch-matmul'
  | 'qbatch-matmul'
  | 'moe-router'
  | 'moe-linear'
  | 'sdpa'
  | 'qsdpa'
  | 'cross-sdpa'
  | 'cross-attention'
  | 'feature-norm'
  | 'q-layer-norm'
  | 'group-norm'
  | 'q-group-norm'
  | 'batch-norm-2d'
  | 'conv-2d'
  | 'qconv-2d'
  | 'conv-1d'
  | 'conv-transpose-2d'
  | 'pool-2d'
  | 'global-average-pool'
  | 'resize'
  | 'interpolate-1d'
  | 'upsample-nearest-2d'
  | 'uniform-elementwise'
  | 'prelu'
  | 'gelu'
  | 'q-activation'
  | 'clip'
  | 'binary'
  | 'qadd'
  | 'softmax'
  | 'reduction'
  | 'q-masked-mean'
  | 'argmax'
  | 'qargmax'
  | 'comparison'
  | 'not'
  | 'where'
  | 'shape-copy'
  | 'transpose'
  | 'concat'
  | 'split'
  | 'slice'
  | 'pad'
  | 'expand'
  | 'gather'
  | 'gather-elements'
  | 'quantize'
  | 'dequantize'
  | 'requantize'
  | 'cast'
  | 'embedding'
  | 'qembedding'
  | 'vision-profile';

/**
 * Every canonical operator advertised by kernelRoute('webgpu', ...) has one
 * explicit physical proof family.  The table is intentionally independent of
 * exporter qualification: adding a registry route without adding a physical
 * proof must fail closed before a CompiledModel can be published.
 */
const WEBGPU_PHYSICAL_ROUTE_FAMILIES: Readonly<Record<string, PhysicalRouteFamily>> =
  Object.freeze({
    MatMul: 'dense',
    Linear: 'dense',
    QLinear: 'qdense',
    BatchMatMul: 'batch-matmul',
    QBatchMatMul: 'qbatch-matmul',
    QGemm: 'qdense',
    Gemm: 'dense',
    QMatMul: 'qdense',
    MoERouter: 'moe-router',
    MoELinear: 'moe-linear',
    SDPA: 'sdpa',
    QSDPA: 'qsdpa',
    CrossSDPA: 'cross-sdpa',
    CrossAttention: 'cross-attention',
    LayerNorm: 'feature-norm',
    QLayerNorm: 'q-layer-norm',
    GroupNorm: 'group-norm',
    QGroupNorm: 'q-group-norm',
    RMSNorm: 'feature-norm',
    BatchNorm2D: 'batch-norm-2d',
    Conv2D: 'conv-2d',
    QConv2D: 'qconv-2d',
    Conv1D: 'conv-1d',
    ConvTranspose2D: 'conv-transpose-2d',
    MaxPool2D: 'pool-2d',
    AveragePool2D: 'pool-2d',
    GlobalAveragePool: 'global-average-pool',
    Resize: 'resize',
    Interpolate1D: 'interpolate-1d',
    ResizeNearest2D: 'resize',
    UpsampleNearest2D: 'upsample-nearest-2d',
    ReLU: 'uniform-elementwise',
    LeakyReLU: 'uniform-elementwise',
    PReLU: 'prelu',
    GELU: 'gelu',
    QGELU: 'q-activation',
    SiLU: 'uniform-elementwise',
    QSiLU: 'q-activation',
    Sigmoid: 'uniform-elementwise',
    HardSwish: 'uniform-elementwise',
    HardSigmoid: 'uniform-elementwise',
    Tanh: 'uniform-elementwise',
    Clip: 'clip',
    Add: 'binary',
    QAdd: 'qadd',
    Sub: 'binary',
    Mul: 'binary',
    Div: 'binary',
    Softmax: 'softmax',
    LogSoftmax: 'softmax',
    ReduceSum: 'reduction',
    ReduceMean: 'reduction',
    QMaskedMean: 'q-masked-mean',
    ArgMax: 'argmax',
    QArgMax: 'qargmax',
    Equal: 'comparison',
    GreaterOrEqual: 'comparison',
    Not: 'not',
    Where: 'where',
    Mask: 'where',
    Reshape: 'shape-copy',
    Flatten: 'shape-copy',
    Squeeze: 'shape-copy',
    Unsqueeze: 'shape-copy',
    Transpose: 'transpose',
    Concat: 'concat',
    Split: 'split',
    Slice: 'slice',
    Pad: 'pad',
    Expand: 'expand',
    Gather: 'gather',
    GatherElements: 'gather-elements',
    Identity: 'shape-copy',
    Broadcast: 'expand',
    Concat2: 'concat',
    QuantizeLinear: 'quantize',
    DequantizeLinear: 'dequantize',
    RequantizeLinear: 'requantize',
    Cast: 'cast',
    Embedding: 'embedding',
    QEmbedding: 'qembedding',
    SpatialSoftargmaxY: 'vision-profile',
    MeanHeight: 'vision-profile',
    ProfileX: 'vision-profile',
    ProfileY: 'vision-profile',
    Dropout: 'shape-copy',
  });

export const WEBGPU_CANONICAL_PHYSICAL_OPERATORS: readonly string[] = Object.freeze(
  Object.keys(WEBGPU_PHYSICAL_ROUTE_FAMILIES).sort(),
);

const WEBGPU_UNIFORM_ELEMENTWISE_ROUTES: Readonly<Record<string, string>> = Object.freeze({
  ReLU: 'getReLUShader',
  Sigmoid: 'getSigmoidShader',
  HardSwish: 'getHardSwishShader',
  HardSigmoid: 'getHardSigmoidShader',
  SiLU: 'getSiLUShader',
  Tanh: 'getTanhShader',
  LeakyReLU: 'getLeakyReLUShader',
});

const BYTE_DTYPES = new Set<RuntimeDType>(['int8', 'uint8']);
const COPY_DTYPES = new Set<RuntimeDType>(['float32', 'int32', 'int8', 'uint8']);
const U32_MAX = 0xffffffffn;
const I32_MAX = 0x7fffffffn;

export interface WebGPUExtentBounds {
  readonly minimum: bigint;
  readonly maximum: bigint;
}

function unsupportedPhysicalDomain(node: { id: string; opType: string }, reason: string): never {
  throw new VolvoxAIError('BACKEND_UNSUPPORTED',
    `WebGPU cannot prove the complete physical domain of '${node.opType}' at ` +
    `node '${node.id}': ${reason}`, {
      phase: 'compilation', backend: 'webgpu', node: node.id,
    });
}

function requiredDeviceLimit(value: unknown, name: string): bigint {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WebGPU device does not expose a valid ${name} limit.`, {
        phase: 'compilation', backend: 'webgpu',
      });
  }
  return BigInt(value as number);
}

function exactNames(value: object): readonly string[] {
  return Object.getOwnPropertyNames(value).sort();
}

function assertPhysicalPorts(
  node: { id: string; opType: string },
  actual: object,
  expected: readonly string[],
  kind: string,
): void {
  const names = exactNames(actual);
  const wanted = [...expected].sort();
  if (names.length !== wanted.length || names.some((name, index) => name !== wanted[index])) {
    unsupportedPhysicalDomain(node,
      `${kind} ports [${names.join(', ')}] are not the proved [${wanted.join(', ')}].`);
  }
}

/**
 * Resolve one logical extent through the exact canonical-domain relations.
 *
 * Output-only symbols are frequently declared with a wider containing range
 * than the value inferred by their producer (for example an Embedding hidden
 * width or an affine Concat extent).  Reading only graph.dimensions would make
 * a fixed physical width appear dynamic and would also overstate dispatch and
 * allocation maxima.  The accepted proof is authoritative for those aliases
 * and affine definitions; public symbols terminate at their declared legal
 * multiple-of interval.
 */
export function checkedWebGPUExtentBounds(
  input: BackendLogicalCompileInput,
  dimension: number | string,
  label = `dimension '${String(dimension)}'`,
  visiting = new Set<string>(),
): WebGPUExtentBounds {
  if (typeof dimension === 'number') {
    if (!Number.isSafeInteger(dimension) || dimension <= 0) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebGPU cannot bound ${label}.`, {
          phase: 'compilation', backend: 'webgpu',
        });
    }
    const value = BigInt(dimension);
    return Object.freeze({ minimum: value, maximum: value });
  }
  if (visiting.has(dimension)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WebGPU found a cyclic symbolic extent while proving ${label}.`, {
        phase: 'compilation', backend: 'webgpu',
      });
  }
  visiting.add(dimension);
  try {
    const affine = input.shapeDomainProof.affineSymbolRelations[dimension];
    if (affine !== undefined) {
      if (!Number.isSafeInteger(affine.offset)) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `WebGPU cannot represent the affine offset for ${label}.`, {
            phase: 'compilation', backend: 'webgpu',
          });
      }
      const source = checkedWebGPUExtentBounds(input, affine.source, label, visiting);
      const offset = BigInt(affine.offset);
      const minimum = source.minimum + offset;
      const maximum = source.maximum + offset;
      if (minimum <= 0n || maximum < minimum) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `WebGPU cannot represent the affine bounded extent of ${label}.`, {
            phase: 'compilation', backend: 'webgpu',
          });
      }
      return Object.freeze({ minimum, maximum });
    }

    const relation = input.shapeDomainProof.symbolRelations[dimension];
    if (relation !== undefined && relation !== dimension) {
      return checkedWebGPUExtentBounds(input, relation, label, visiting);
    }

    const constraint = input.graph.dimensions[dimension];
    if (constraint === undefined || !Number.isSafeInteger(constraint.min) ||
        !Number.isSafeInteger(constraint.max) ||
        !Number.isSafeInteger(constraint.multiple_of) || constraint.min <= 0 ||
        constraint.max < constraint.min || constraint.multiple_of <= 0) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebGPU cannot bound ${label}.`, {
          phase: 'compilation', backend: 'webgpu',
        });
    }
    const multiple = BigInt(constraint.multiple_of);
    const rawMinimum = BigInt(constraint.min);
    const rawMaximum = BigInt(constraint.max);
    const minimum = ((rawMinimum + multiple - 1n) / multiple) * multiple;
    const maximum = (rawMaximum / multiple) * multiple;
    if (minimum <= 0n || maximum < minimum) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebGPU ${label} has no legal multiple_of binding.`, {
          phase: 'compilation', backend: 'webgpu',
        });
    }
    return Object.freeze({ minimum, maximum });
  } finally {
    visiting.delete(dimension);
  }
}

function dimensionBound(
  input: BackendLogicalCompileInput,
  dimension: number | string,
  maximum: boolean,
): bigint {
  const bounds = checkedWebGPUExtentBounds(input, dimension);
  return maximum ? bounds.maximum : bounds.minimum;
}

function descriptorElements(
  input: BackendLogicalCompileInput,
  descriptor: LogicalDomainTensorDescriptor,
  maximum: boolean,
  prefixAxes = descriptor.shape.length,
): bigint {
  let result = 1n;
  for (const dimension of descriptor.shape.slice(0, prefixAxes)) {
    result *= dimensionBound(input, dimension, maximum);
  }
  return result;
}

function axisBound(
  input: BackendLogicalCompileInput,
  descriptor: LogicalDomainTensorDescriptor,
  axis: number,
  maximum = true,
): bigint {
  const normalized = axis < 0 ? axis + descriptor.shape.length : axis;
  if (normalized < 0 || normalized >= descriptor.shape.length) return 0n;
  return dimensionBound(input, descriptor.shape[normalized], maximum);
}

function fixedExtent(
  input: BackendLogicalCompileInput,
  dimension: number | string,
  node: { id: string; opType: string },
  label: string,
): number {
  const minimum = dimensionBound(input, dimension, false);
  const maximum = dimensionBound(input, dimension, true);
  if (minimum !== maximum || maximum > U32_MAX) {
    unsupportedPhysicalDomain(node, `${label} is not one fixed u32 extent.`);
  }
  return Number(maximum);
}

function ceilDivide(value: bigint, divisor: bigint): bigint {
  return (value + divisor - 1n) / divisor;
}

function assertDispatchBound(
  node: { id: string; opType: string },
  label: string,
  counts: readonly bigint[],
  limit: bigint,
): void {
  if (counts.length !== 3 || counts.some((count) => count <= 0n || count > U32_MAX || count > limit)) {
    unsupportedPhysicalDomain(node,
      `${label} dispatch [${counts.join(',')}] exceeds maxComputeWorkgroupsPerDimension ${limit}.`);
  }
}

function assertDType(
  node: { id: string; opType: string },
  descriptor: LogicalDomainTensorDescriptor,
  allowed: ReadonlySet<RuntimeDType>,
  label: string,
): void {
  if (!allowed.has(descriptor.dtype)) {
    unsupportedPhysicalDomain(node,
      `${label} dtype '${descriptor.dtype}' is outside [${[...allowed].join(', ')}].`);
  }
}

function dimensionsProvablyEqual(
  input: BackendLogicalCompileInput,
  left: number | string,
  right: number | string,
): boolean {
  if (left === right) return true;
  return dimensionBound(input, left, false) === dimensionBound(input, left, true) &&
    dimensionBound(input, right, false) === dimensionBound(input, right, true) &&
    dimensionBound(input, left, true) === dimensionBound(input, right, true);
}

function shapesProvablyEqual(
  input: BackendLogicalCompileInput,
  left: LogicalDomainTensorDescriptor,
  right: LogicalDomainTensorDescriptor,
): boolean {
  return left.shape.length === right.shape.length && left.shape.every((dimension, axis) =>
    dimensionsProvablyEqual(input, dimension, right.shape[axis]));
}

function sameQuantization(left: TensorQuantization | undefined, right: TensorQuantization | undefined): boolean {
  if (left === undefined || right === undefined || left.scheme !== right.scheme) return false;
  if (left.scheme === 'per_tensor' && right.scheme === 'per_tensor') {
    return Math.fround(left.scale) === Math.fround(right.scale) &&
      left.zero_point === right.zero_point;
  }
  if (left.scheme !== 'per_axis' || right.scheme !== 'per_axis') return false;
  return left.axis === right.axis && left.scales.length === right.scales.length &&
    left.zero_points.length === right.zero_points.length &&
    left.scales.every((value, index) => Math.fround(value) === Math.fround(right.scales[index])) &&
    left.zero_points.every((value, index) => value === right.zero_points[index]);
}

function assertPerTensorByte(
  node: { id: string; opType: string },
  descriptor: LogicalDomainTensorDescriptor,
  label: string,
): Extract<TensorQuantization, { readonly scheme: 'per_tensor' }> {
  if (!BYTE_DTYPES.has(descriptor.dtype) || descriptor.quantization?.scheme !== 'per_tensor') {
    unsupportedPhysicalDomain(node, `${label} requires I8/U8 per-tensor affine storage.`);
  }
  return descriptor.quantization;
}

function numericParam(
  node: NodeDescriptor,
  name: string,
  fallback: number,
): number {
  const value = node.params[name];
  return typeof value === 'number' ? value : fallback;
}

function arrayParam(node: NodeDescriptor, name: string): readonly JsonValue[] {
  const value = node.params[name];
  return Array.isArray(value) ? value : [];
}

function positiveIntegerParam(node: NodeDescriptor, name: string, fallback: number): number {
  const value = numericParam(node, name, fallback);
  if (!Number.isSafeInteger(value) || value <= 0) {
    unsupportedPhysicalDomain(node, `parameter '${name}' is not a positive safe integer.`);
  }
  return value;
}

function dtypeRange(dtype: RuntimeDType): readonly [number, number] {
  return dtype === 'int8' ? [-128, 127] : [0, 255];
}

function physicalMethodForBinary(node: NodeDescriptor): string {
  if (node.opType === 'Mul') return 'getBroadcastMulShader';
  if (node.opType === 'Sub') return 'getBroadcastSubShader';
  if (node.opType === 'Div') return 'getBroadcastDivShader';
  const relu = numericParam(node, 'relu', 0);
  if (relu === 1) return 'getBroadcastAddReLUShader';
  if (relu === 2) return 'getBroadcastAddReLU6Shader';
  return 'getBroadcastAddShader';
}

function validateCoverageTable(): void {
  const advertised = runtimeOperatorsByBackend.webgpu.filter((operator) =>
    operatorShapeContract(operator)?.classification === 'canonical');
  const advertisedSet = new Set(advertised);
  const missing = advertised.filter((operator) => WEBGPU_PHYSICAL_ROUTE_FAMILIES[operator] === undefined);
  const stale = Object.keys(WEBGPU_PHYSICAL_ROUTE_FAMILIES).filter((operator) =>
    !advertisedSet.has(operator));
  if (missing.length !== 0 || stale.length !== 0) {
    throw new Error(
      `WebGPU physical proof registry mismatch; missing [${missing.join(', ')}], stale [${stale.join(', ')}].`,
    );
  }
}

validateCoverageTable();

function assertDeviceStructuralLimits(limits: WebGPULimitsLike): void {
  const requirements: readonly [keyof WebGPULimitsLike, bigint][] = [
    ['maxComputeInvocationsPerWorkgroup', 64n],
    ['maxComputeWorkgroupSizeX', 64n],
    ['maxComputeWorkgroupSizeY', 8n],
    ['maxComputeWorkgroupSizeZ', 1n],
    // QSDPA is the largest advertised portable route: 2,048 F32 partial
    // outputs plus three 32-lane reduction arrays.
    ['maxComputeWorkgroupStorageSize', 8576n],
    // MoELinear binds six data tensors, one uniform, and the resident-slot
    // translation table.
    ['maxBindingsPerBindGroup', 8n],
    ['maxBindGroups', 1n],
    ['maxStorageBuffersPerShaderStage', 7n],
    ['maxUniformBuffersPerShaderStage', 1n],
  ];
  for (const [name, minimum] of requirements) {
    const actual = requiredDeviceLimit(limits[name], name);
    if (actual < minimum) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebGPU ${String(name)} ${actual} is below the proved route requirement ${minimum}.`, {
          phase: 'compilation', backend: 'webgpu',
        });
    }
  }
}

function invariantWeightData(
  input: BackendLogicalCompileInput,
  node: NodeDescriptor,
  descriptor: LogicalDomainTensorDescriptor,
  label: string,
): ArrayBufferView {
  if (!Object.prototype.hasOwnProperty.call(input.graph.weights, descriptor.name)) {
    unsupportedPhysicalDomain(node, `${label} must be invariant for compile-time value preflight.`);
  }
  return input.snapshot.copyWeightData(descriptor.name);
}

function assertFiniteQuantizedAffinePreflight(
  input: BackendLogicalCompileInput,
  node: NodeDescriptor,
  descriptor: LogicalDomainTensorDescriptor,
  label: string,
): void {
  const logical = input.graph.tensors[descriptor.name];
  if (logical?.kind === 'input') return;
  if (logical?.kind !== 'weight') {
    unsupportedPhysicalDomain(node,
      `${label} is produced on device and has no finite-value preflight route.`);
  }
  const data = invariantWeightData(input, node, descriptor, label);
  const minimumElements = descriptorElements(input, descriptor, false);
  const maximumElements = descriptorElements(input, descriptor, true);
  if (!(data instanceof Float32Array) || minimumElements !== maximumElements ||
      BigInt(data.length) !== maximumElements || !data.every(Number.isFinite)) {
    unsupportedPhysicalDomain(node,
      `${label} must be one finite invariant F32 tensor over its complete extent.`);
  }
}

interface StaticIntegerValueDomain {
  readonly minimum: bigint;
  readonly maximum: bigint;
}

function exactProducer(
  input: BackendLogicalCompileInput,
  descriptor: LogicalDomainTensorDescriptor,
  beforeIndex: number,
): readonly [NodeDescriptor, LogicalShapeDomainNodeProof, number] | null {
  const logical = input.graph.tensors[descriptor.name];
  if (logical?.kind !== 'value') return null;
  const index = input.graph.nodes.findIndex((candidate, candidateIndex) =>
    candidateIndex < beforeIndex && candidate.id === logical.producerNodeId &&
    Object.values(candidate.outputs).some((output) => output.tensor === descriptor.name));
  const node = index < 0 ? undefined : input.graph.nodes[index];
  const proof = index < 0 ? undefined : input.shapeDomainProof.nodes[index];
  if (!node || !proof || proof.id !== node.id || proof.opType !== node.opType) return null;
  return [node, proof, index] as const;
}

/** Conservative value facts that require no device readback. */
function staticIntegerValueDomain(
  input: BackendLogicalCompileInput,
  descriptor: LogicalDomainTensorDescriptor,
  beforeIndex: number,
  visiting = new Set<string>(),
): StaticIntegerValueDomain | null {
  if (visiting.has(descriptor.name)) return null;
  visiting.add(descriptor.name);
  try {
    const producerEntry = exactProducer(input, descriptor, beforeIndex);
    if (producerEntry === null) return null;
    const [producer, proof, producerIndex] = producerEntry;

    if (producer.opType === 'Clip' && descriptor.dtype === 'int32') {
      const minimum = numericParam(producer, 'min', -2147483648);
      const maximum = numericParam(producer, 'max', 2147483647);
      return Number.isSafeInteger(minimum) && Number.isSafeInteger(maximum) && minimum <= maximum
        ? Object.freeze({ minimum: BigInt(minimum), maximum: BigInt(maximum) })
        : null;
    }

    if ((producer.opType === 'ArgMax' || producer.opType === 'QArgMax') &&
        descriptor.dtype === 'int32') {
      const source = proof.inputs.input;
      if (!source) return null;
      let axis = numericParam(producer, 'axis', producer.opType === 'QArgMax' ? -1 : 0);
      if (!Number.isSafeInteger(axis)) return null;
      if (axis < 0) axis += source.shape.length;
      if (axis < 0 || axis >= source.shape.length) return null;
      return Object.freeze({
        minimum: 0n,
        maximum: dimensionBound(input, source.shape[axis], true) - 1n,
      });
    }

    if (descriptor.dtype === 'int32' &&
        (producer.opType === 'Equal' || producer.opType === 'GreaterOrEqual' ||
         producer.opType === 'Not')) {
      return Object.freeze({ minimum: 0n, maximum: 1n });
    }

    if (producer.opType === 'Cast' && descriptor.dtype === 'int32') {
      const source = proof.inputs.input;
      if (!source) return null;
      if (source.dtype === 'uint8') return Object.freeze({ minimum: 0n, maximum: 255n });
      if (source.dtype === 'int8') return Object.freeze({ minimum: -128n, maximum: 127n });
      if (source.dtype === 'int32') {
        return staticIntegerValueDomain(input, source, producerIndex, visiting);
      }
      return null;
    }

    const preserving = new Set([
      'Identity', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose',
      'Slice', 'Expand', 'Broadcast', 'Dropout',
    ]);
    if (descriptor.dtype === 'int32' && preserving.has(producer.opType)) {
      const source = proof.inputs.input;
      return source
        ? staticIntegerValueDomain(input, source, producerIndex, visiting)
        : null;
    }
    return null;
  } finally {
    visiting.delete(descriptor.name);
  }
}

function assertI32ValuePreflight(
  input: BackendLogicalCompileInput,
  node: NodeDescriptor,
  descriptor: LogicalDomainTensorDescriptor,
  minimum: bigint,
  maximum: bigint,
  label: string,
): void {
  const logical = input.graph.tensors[descriptor.name];
  if (logical?.kind === 'input') return;
  if (logical?.kind === 'weight') {
    const data = invariantWeightData(input, node, descriptor, label);
    const elements = descriptorElements(input, descriptor, true);
    if (!(data instanceof Int32Array) || BigInt(data.length) !== elements ||
        data.some((value) => BigInt(value) < minimum || BigInt(value) > maximum)) {
      unsupportedPhysicalDomain(node,
        `${label} invariant I32 payload is outside [${minimum}, ${maximum}].`);
    }
    return;
  }
  const consumerIndex = input.graph.nodes.indexOf(node);
  const domain = staticIntegerValueDomain(input, descriptor, consumerIndex);
  if (domain === null || domain.minimum < minimum || domain.maximum > maximum) {
    unsupportedPhysicalDomain(node,
      `${label} has no complete host or static value-range preflight within ` +
      `[${minimum}, ${maximum}].`);
  }
}

function assertF32RoutePreflight(
  input: BackendLogicalCompileInput,
  node: NodeDescriptor,
  descriptor: LogicalDomainTensorDescriptor,
  label: string,
  maximumExpert?: number,
): void {
  const logical = input.graph.tensors[descriptor.name];
  if (logical?.kind === 'input') return;
  if (logical?.kind !== 'weight') {
    unsupportedPhysicalDomain(node,
      `${label} is produced on device and has no pre-dispatch value validation route.`);
  }
  const data = invariantWeightData(input, node, descriptor, label);
  const elements = descriptorElements(input, descriptor, true);
  const valid = data instanceof Float32Array && BigInt(data.length) === elements &&
    data.every((value) => maximumExpert === undefined
      ? Number.isFinite(value)
      : Number.isInteger(value) && value >= 0 && value < maximumExpert);
  if (!valid) {
    unsupportedPhysicalDomain(node,
      `${label} invariant F32 payload fails its complete value-domain preflight.`);
  }
}

function assertInvariantQuantizationParameters(
  input: BackendLogicalCompileInput,
  node: NodeDescriptor,
  scale: LogicalDomainTensorDescriptor,
  zeroPoint: LogicalDomainTensorDescriptor | undefined,
  quantization: TensorQuantization,
): void {
  if (quantization.scheme !== 'per_tensor') {
    unsupportedPhysicalDomain(node,
      'the WebGPU scalar quantize/dequantize route does not implement per-axis parameters.');
  }
  const scaleData = invariantWeightData(input, node, scale, 'scale');
  if (!(scaleData instanceof Float32Array) || scaleData.length !== 1 ||
      Math.fround(scaleData[0]) !== Math.fround(quantization.scale)) {
    unsupportedPhysicalDomain(node, 'invariant F32 scale does not match affine tensor metadata.');
  }
  if (zeroPoint === undefined) {
    if (quantization.zero_point !== 0) {
      unsupportedPhysicalDomain(node, 'an omitted zero_point requires affine zero_point 0.');
    }
    return;
  }
  const zeroPointData = invariantWeightData(input, node, zeroPoint, 'zero_point');
  const validStorage = zeroPoint.dtype === 'int8'
    ? zeroPointData instanceof Int8Array
    : zeroPointData instanceof Uint8Array;
  if (!validStorage || (zeroPointData as Int8Array | Uint8Array).length !== 1 ||
      (zeroPointData as Int8Array | Uint8Array)[0] !== quantization.zero_point) {
    unsupportedPhysicalDomain(node,
      'invariant typed zero_point does not match affine tensor metadata.');
  }
}

/**
 * Prove every physical WebGPU predicate over the complete bounded domain.
 * Canonical shape inference has already proved semantic output relationships;
 * this layer mirrors the concrete compiler's storage, rank, tactic, dispatch,
 * metadata, accumulator, and auxiliary-buffer restrictions.
 */
export function checkedWebGPUPhysicalDomain(
  input: BackendLogicalCompileInput,
  device: GPUDevice,
): WebGPUPhysicalDomainProof {
  if (input.shapeDomainProof.nodes.length !== input.graph.nodes.length) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      "Backend 'webgpu' received incomplete canonical domain evidence.", {
        phase: 'compilation', backend: 'webgpu',
      });
  }
  const limits = (device as GPUDevice & { limits?: WebGPULimitsLike }).limits || {};
  assertDeviceStructuralLimits(limits);
  const workgroupLimit = requiredDeviceLimit(
    limits.maxComputeWorkgroupsPerDimension,
    'maxComputeWorkgroupsPerDimension',
  );
  const bufferLimit = requiredDeviceLimit(limits.maxBufferSize, 'maxBufferSize');
  const storageLimit = requiredDeviceLimit(
    limits.maxStorageBufferBindingSize,
    'maxStorageBufferBindingSize',
  );
  const uniformLimit = requiredDeviceLimit(
    limits.maxUniformBufferBindingSize,
    'maxUniformBufferBindingSize',
  );
  const auxiliaryStorageLimit = bufferLimit < storageLimit ? bufferLimit : storageLimit;
  const auxiliaryUniformLimit = bufferLimit < uniformLimit ? bufferLimit : uniformLimit;
  const required = new Set<string>();
  const optional = new Set<string>();
  const packedDot4 = new Set<string>();
  let maximumAuxiliaryBytes = 0n;

  const charge = (
    node: NodeDescriptor,
    bytes: bigint,
    kind: 'storage' | 'uniform',
    label: string,
  ): void => {
    const limit = kind === 'storage' ? auxiliaryStorageLimit : auxiliaryUniformLimit;
    if (bytes <= 0n || bytes > U32_MAX || bytes > limit) {
      unsupportedPhysicalDomain(node,
        `${label} ${kind} allocation ${bytes} exceeds its proved binding limit ${limit}.`);
    }
    maximumAuxiliaryBytes += bytes < 4n ? 4n : ceilDivide(bytes, 4n) * 4n;
  };

  const dispatch1D = (
    node: NodeDescriptor,
    descriptor: LogicalDomainTensorDescriptor,
    divisor: bigint,
    label = node.opType,
  ): void => assertDispatchBound(node, label, [
    ceilDivide(descriptorElements(input, descriptor, true), divisor), 1n, 1n,
  ], workgroupLimit);

  const dispatchLinear2D = (
    node: NodeDescriptor,
    descriptor: LogicalDomainTensorDescriptor,
    divisor: bigint,
    label = node.opType,
  ): void => {
    const groups = ceilDivide(descriptorElements(input, descriptor, true), divisor);
    const strideLimit = U32_MAX / divisor;
    const x = groups < workgroupLimit
      ? (groups < strideLimit ? groups : strideLimit)
      : (workgroupLimit < strideLimit ? workgroupLimit : strideLimit);
    const y = ceilDivide(groups, x);
    assertDispatchBound(node, label, [x, y, 1n], workgroupLimit);
  };

  for (let index = 0; index < input.graph.nodes.length; index++) {
    const node = input.graph.nodes[index];
    const proof = input.shapeDomainProof.nodes[index];
    const contract = operatorShapeContract(node.opType);
    const family = WEBGPU_PHYSICAL_ROUTE_FAMILIES[node.opType];
    if (!contract || contract.classification !== 'canonical' ||
        !kernelRoute('webgpu', node.opType) || family === undefined || proof?.id !== node.id ||
        proof.opType !== node.opType || proof.shapeFunctionId !== contract.shapeFunctionId) {
      unsupportedPhysicalDomain(node, 'canonical registry route evidence is absent or inconsistent.');
    }
    assertPhysicalPorts(node, proof.inputs, Object.keys(node.inputs), 'input');
    assertPhysicalPorts(node, proof.outputs, Object.keys(node.outputs), 'output');

    // Concrete WebGPU descriptors encode extents, element counts, offsets, and
    // strides as u32. Prove that every tensor touched by this node remains
    // representable before considering its operator-specific dispatch.
    for (const [port, descriptor] of [
      ...Object.entries(proof.inputs),
      ...Object.entries(proof.outputs),
    ]) {
      if (descriptor.shape.some((dimension) => dimensionBound(input, dimension, true) > U32_MAX) ||
          descriptorElements(input, descriptor, true) > U32_MAX) {
        unsupportedPhysicalDomain(node,
          `tensor port '${port}' can exceed WebGPU u32 descriptor arithmetic.`);
      }
    }

    const inputs = proof.inputs;
    const outputs = proof.outputs;
    const output = outputs.out;

    if (family === 'uniform-elementwise') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      required.add(WEBGPU_UNIFORM_ELEMENTWISE_ROUTES[node.opType]);
      dispatchLinear2D(node, output, 64n, 'uniform elementwise');
      charge(node, 16n, 'uniform', 'elementwise parameters');
      continue;
    }

    if (family === 'gelu') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      required.add('getGELUShader');
      dispatch1D(node, output, 64n);
      charge(node, 16n, 'uniform', 'GELU parameters');
      continue;
    }

    if (family === 'prelu') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, inputs.slope, new Set(['float32']), 'slope');
      assertDType(node, output, new Set(['float32']), 'output');
      required.add('getPReLUShader');
      dispatch1D(node, output, 64n);
      charge(node, 16n, 'uniform', 'PReLU parameters');
      continue;
    }

    if (family === 'shape-copy') {
      const source = inputs.input;
      if (!COPY_DTYPES.has(source.dtype) || output.dtype !== source.dtype ||
          descriptorElements(input, source, false) !== descriptorElements(input, output, false) ||
          descriptorElements(input, source, true) !== descriptorElements(input, output, true)) {
        unsupportedPhysicalDomain(node,
          'shape-copy requires equal-element F32/I32/I8/U8 input and output storage over the domain.');
      }
      const packed = BYTE_DTYPES.has(source.dtype);
      if (packed) {
        if (!sameQuantization(source.quantization, output.quantization)) {
          unsupportedPhysicalDomain(node, 'byte shape-copy must preserve the exact affine descriptor.');
        }
        if (source.quantization?.scheme === 'per_axis' &&
            !shapesProvablyEqual(input, source, output)) {
          unsupportedPhysicalDomain(node,
            'the byte-copy route cannot remap per-axis metadata across a shape change.');
        }
      }
      required.add(packed ? 'getTypedCopyShader' : 'getCopy32Shader');
      dispatchLinear2D(node, output, packed ? 256n : 64n, node.opType);
      charge(node, 16n, 'uniform', 'shape-copy parameters');
      continue;
    }

    if (family === 'dense') {
      const activation = inputs.input;
      const weight = inputs.weight;
      assertDType(node, activation, new Set(['float32']), 'input');
      assertDType(node, weight, new Set(['float32']), 'weight');
      assertDType(node, output, new Set(['float32']), 'output');
      if (inputs.bias) assertDType(node, inputs.bias, new Set(['float32']), 'bias');
      const dIn = fixedExtent(input, activation.shape.at(-1)!, node, 'contracted dimension');
      const dOut = fixedExtent(input, output.shape.at(-1)!, node, 'output-feature dimension');
      const outputMajorWeight = node.params.weight_layout === 'dout_din' ||
        node.params.transB === true ||
        (node.params.weight_layout === undefined && node.params.transB === undefined &&
          node.opType === 'Linear');
      const inputMajorWeight = node.params.weight_layout === 'din_dout' ||
        node.params.transB === false ||
        (node.params.weight_layout === undefined && node.params.transB === undefined &&
          node.opType !== 'Linear');
      const expectedWeightRows = outputMajorWeight ? dOut : dIn;
      const expectedWeightColumns = outputMajorWeight ? dIn : dOut;
      if ((!outputMajorWeight && !inputMajorWeight) || weight.shape.length !== 2 ||
          fixedExtent(input, weight.shape[0], node, 'weight row dimension') !==
            expectedWeightRows ||
          fixedExtent(input, weight.shape[1], node, 'weight column dimension') !==
            expectedWeightColumns) {
        unsupportedPhysicalDomain(node,
          'the WebGPU dense route requires canonical [d_out,d_in] output-major or ' +
          '[d_in,d_out] input-major F32 weights.');
      }
      const minimumRows = descriptorElements(input, activation, false, activation.shape.length - 1);
      const maximumRows = descriptorElements(input, activation, true, activation.shape.length - 1);
      const tiled = dIn >= 16 && dOut >= 16;
      if (!tiled || minimumRows === 1n) {
        required.add(outputMajorWeight ? 'getLinearF32Shader' : 'getLinearF32RowMajorShader');
        assertDispatchBound(node, 'dense scalar', [
          ceilDivide(BigInt(dOut), 64n), tiled ? 1n : maximumRows, 1n,
        ], workgroupLimit);
      }
      if (tiled && maximumRows > 1n) {
        required.add(outputMajorWeight
          ? 'getLinearF32TiledShader'
          : 'getLinearF32RowMajorTiledShader');
        assertDispatchBound(node, 'dense tiled', [
          ceilDivide(BigInt(dOut), 16n), ceilDivide(maximumRows, 16n), 1n,
        ], workgroupLimit);
      }
      charge(node, BigInt(Math.max(4, dOut * 4)), 'storage', 'dense dummy bias');
      charge(node, 16n, 'uniform', 'dense parameters');
      continue;
    }

    if (family === 'qdense') {
      const activation = inputs.input;
      const weight = inputs.weight;
      const bias = inputs.bias;
      if (activation.shape.length < 1 || output.shape.length !== activation.shape.length ||
          weight.shape.length !== 2 || bias.shape.length !== 1 ||
          !BYTE_DTYPES.has(activation.dtype) || !BYTE_DTYPES.has(weight.dtype) ||
          !BYTE_DTYPES.has(output.dtype) || bias.dtype !== 'int32') {
        unsupportedPhysicalDomain(node, 'QLinear dtype/rank contract is not canonical W8A8.');
      }
      const dIn = fixedExtent(input, activation.shape.at(-1)!, node, 'contracted dimension');
      const dOut = fixedExtent(input, output.shape.at(-1)!, node, 'output-feature dimension');
      if (fixedExtent(input, weight.shape[0], node, 'weight output dimension') !== dOut ||
          fixedExtent(input, weight.shape[1], node, 'weight contracted dimension') !== dIn ||
          fixedExtent(input, bias.shape[0], node, 'bias extent') !== dOut) {
        unsupportedPhysicalDomain(node, 'QLinear fixed weight/bias dimensions disagree.');
      }
      const biasData = invariantWeightData(input, node, bias, 'I32 bias');
      if (!(biasData instanceof Int32Array) || biasData.length !== dOut) {
        unsupportedPhysicalDomain(node, 'I32 bias payload is not exact.');
      }
      const inputQuantization = assertPerTensorByte(node, activation, 'input');
      const outputQuantization = assertPerTensorByte(node, output, 'output');
      const weightQuantization = weight.quantization;
      if (weightQuantization?.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
          weightQuantization.scales.length !== dOut ||
          weightQuantization.zero_points.length !== dOut) {
        unsupportedPhysicalDomain(node, 'QLinear hydrated affine metadata is incomplete.');
      }
      const [inputMinimum, inputMaximum] = dtypeRange(activation.dtype);
      const [weightMinimum, weightMaximum] = dtypeRange(weight.dtype);
      const inputMagnitude = Math.max(
        Math.abs(inputMinimum - inputQuantization.zero_point),
        Math.abs(inputMaximum - inputQuantization.zero_point),
      );
      const inputScale = Math.fround(inputQuantization.scale);
      const outputScale = Math.fround(outputQuantization.scale);
      for (let channel = 0; channel < dOut; channel++) {
        const scale = weightQuantization.scales[channel];
        const zeroPoint = weightQuantization.zero_points[channel];
        const multiplier = Math.fround(Math.fround(inputScale * Math.fround(scale)) / outputScale);
        const weightMagnitude = Math.max(
          Math.abs(weightMinimum - zeroPoint), Math.abs(weightMaximum - zeroPoint),
        );
        const accumulator = BigInt(inputMagnitude) * BigInt(weightMagnitude) * BigInt(dIn) +
          BigInt(Math.abs(biasData[channel]));
        if (!Number.isFinite(scale) || scale <= 0 || !Number.isInteger(zeroPoint) ||
            zeroPoint < weightMinimum || zeroPoint > weightMaximum ||
            !Number.isFinite(multiplier) || multiplier <= 0 || accumulator > I32_MAX) {
          unsupportedPhysicalDomain(node,
            `QLinear channel ${channel} affine or I32 accumulator bound is not representable.`);
        }
      }
      const minimumRows = descriptorElements(input, activation, false, activation.shape.length - 1);
      const maximumRows = descriptorElements(input, activation, true, activation.shape.length - 1);
      const tiledGeometry = dIn >= 16 && dOut >= 32 && dOut % 4 === 0;
      if (!tiledGeometry || minimumRows === 1n) {
        const scalarRows = tiledGeometry ? 1n : maximumRows;
        assertDispatchBound(node, 'QLinear scalar', [
          ceilDivide(scalarRows * BigInt(dOut), 256n), 1n, 1n,
        ], workgroupLimit);
        required.add('getQLinearShader');
        packedDot4.add('getQLinearDotShader');
      }
      if (tiledGeometry && maximumRows > 1n) {
        assertDispatchBound(node, 'QLinear tiled', [
          ceilDivide(BigInt(dOut), 32n), ceilDivide(maximumRows, 8n), 1n,
        ], workgroupLimit);
        required.add('getQLinearTiledShader');
        packedDot4.add('getQLinearDotTiledShader');
      }
      charge(node, BigInt(Math.max(4, dOut * 4)), 'storage', 'QLinear multipliers');
      charge(node, BigInt(Math.max(4, dOut * 4)), 'storage', 'QLinear zero points');
      charge(node, 64n, 'uniform', 'QLinear parameters');
      continue;
    }

    if (family === 'batch-matmul' || family === 'qbatch-matmul') {
      const a = inputs.a;
      const b = inputs.b;
      if (family === 'batch-matmul') {
        assertDType(node, a, new Set(['float32']), 'a');
        assertDType(node, b, new Set(['float32']), 'b');
        assertDType(node, output, new Set(['float32']), 'output');
        const batches = descriptorElements(input, output, true, output.shape.length - 2);
        const portableLimit = workgroupLimit < 65535n ? workgroupLimit : 65535n;
        assertDispatchBound(node, 'BatchMatMul', [
          ceilDivide(axisBound(input, output, -1), 8n),
          ceilDivide(axisBound(input, output, -2), 8n), batches,
        ], portableLimit);
        required.add('getBatchMatMulShader');
        charge(node, BigInt((5 + Math.max(0, output.shape.length - 2) * 3) * 4),
          'storage', 'BatchMatMul stride metadata');
      } else {
        assertPerTensorByte(node, a, 'a');
        assertPerTensorByte(node, b, 'b');
        assertPerTensorByte(node, output, 'output');
        const portableLimit = workgroupLimit < 65535n ? workgroupLimit : 65535n;
        assertDispatchBound(node, 'QBatchMatMul', [
          ceilDivide(descriptorElements(input, output, true), 256n), 1n, 1n,
        ], portableLimit);
        required.add('getQBatchMatMulShader');
        packedDot4.add('getQBatchMatMulDotShader');
        charge(node, BigInt(Math.max(4, Math.max(0, output.shape.length - 2) * 12)),
          'storage', 'QBatchMatMul stride metadata');
        charge(node, 64n, 'uniform', 'QBatchMatMul parameters');
      }
      continue;
    }

    if (family === 'moe-router') {
      const activation = inputs.input;
      const weight = inputs.weight;
      const indices = outputs.indices;
      const weights = outputs.weights;
      for (const [label, descriptor] of Object.entries({ activation, weight, indices, weights })) {
        assertDType(node, descriptor, new Set(['float32']), label);
      }
      if (inputs.bias) assertDType(node, inputs.bias, new Set(['float32']), 'bias');
      const rows = descriptorElements(input, activation, true, activation.shape.length - 1);
      const experts = fixedExtent(input, weight.shape.at(-1)!, node, 'expert count');
      const topK = fixedExtent(input, indices.shape.at(-1)!, node, 'MoERouter top_k');
      if (topK > 8) {
        unsupportedPhysicalDomain(node,
          `MoERouter top_k ${topK} exceeds the shader's proved maximum 8.`);
      }
      required.add('getMoERouterShader');
      assertDispatchBound(node, 'MoERouter', [ceilDivide(rows, 64n), 1n, 1n], workgroupLimit);
      charge(node, BigInt(Math.max(4, experts * 4)), 'storage', 'MoERouter dummy bias');
      charge(node, 32n, 'uniform', 'MoERouter parameters');
      continue;
    }

    if (family === 'moe-linear') {
      const activation = inputs.input;
      const expertWeight = inputs.expert_weight;
      for (const [label, descriptor] of Object.entries(inputs)) {
        assertDType(node, descriptor, new Set(['float32']), label);
      }
      assertDType(node, output, new Set(['float32']), 'output');
      const rows = descriptorElements(input, activation, true, activation.shape.length - 1);
      const experts = fixedExtent(input, expertWeight.shape[0], node, 'expert count');
      const dOut = fixedExtent(input, output.shape.at(-1)!, node, 'expert output width');
      assertF32RoutePreflight(
        input, node, inputs.route_indices, 'MoELinear route indices', experts,
      );
      assertF32RoutePreflight(
        input, node, inputs.route_weights, 'MoELinear route weights',
      );
      required.add('getMoELinearShader');
      assertDispatchBound(node, 'MoELinear', [
        ceilDivide(BigInt(dOut), 64n), rows, 1n,
      ], workgroupLimit);
      charge(node, BigInt(Math.max(4, experts * dOut * 4)), 'storage', 'MoELinear dummy bias');
      charge(node, 32n, 'uniform', 'MoELinear parameters');
      // Global-slot -> staged-row table for a partially resident expert bank.
      // residentSlots is request-local bound-graph state and is intentionally
      // absent from the immutable logical node. Any legal resident subset may
      // include the final global expert, so the complete bank extent is the
      // only compile-time upper bound for this table.
      charge(
        node,
        BigInt(Math.max(4, experts * 4)),
        'storage',
        'MoELinear resident slot table',
      );
      continue;
    }

    if (family === 'sdpa' || family === 'cross-sdpa' || family === 'cross-attention') {
      const query = family === 'sdpa' ? inputs.qkv : inputs.q;
      const queryOutput = output;
      const rank = query.shape.length;
      const batch = rank === 2 ? 1n : axisBound(input, query, 0);
      const querySequence = axisBound(input, queryOutput, -2);
      const dModel = fixedExtent(input, queryOutput.shape.at(-1)!, node, 'attention feature width');
      const heads = positiveIntegerParam(node, 'heads', 8);
      const headDimension = dModel / heads;
      if (!Number.isInteger(headDimension) || headDimension <= 0 || headDimension > 64) {
        unsupportedPhysicalDomain(node,
          `attention head dimension ${headDimension} is outside the WebGPU kernel range [1, 64].`);
      }
      for (const [label, descriptor] of Object.entries(inputs)) {
        if (label === 'mask') assertDType(node, descriptor, new Set(['int32']), label);
        else assertDType(node, descriptor, new Set(['float32']), label);
      }
      assertDType(node, output, new Set(['float32']), 'output');
      if (family === 'sdpa') required.add('getSDPAShader');
      else if (family === 'cross-sdpa') required.add('getCrossSDPAShader');
      else {
        // Bounded-shape CrossAttention is canonical F32; the legacy packed-I8
        // projection route is not reachable through its closed shape contract.
        if (dModel > 64) {
          unsupportedPhysicalDomain(node, 'F32 CrossAttention requires d_model <= 64.');
        }
        required.add('getCrossAttentionF32Shader');
        charge(node, 4096n, 'storage', 'CrossAttention dummy scale');
        charge(node, 4096n, 'storage', 'CrossAttention dummy bias');
      }
      assertDispatchBound(node, node.opType, [
        ceilDivide(querySequence, 64n), BigInt(heads), batch,
      ], workgroupLimit);
      charge(node, family === 'sdpa' ? 32n : 48n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'qsdpa') {
      for (const label of ['q', 'k', 'v'] as const) assertPerTensorByte(node, inputs[label], label);
      assertPerTensorByte(node, output, 'output');
      if (inputs.mask) assertDType(node, inputs.mask, new Set(['int32']), 'mask');
      const rank = inputs.q.shape.length;
      const batch = rank === 2 ? 1n : axisBound(input, inputs.q, 0);
      const sequence = axisBound(input, inputs.q, -2);
      const feature = fixedExtent(input, inputs.q.shape.at(-1)!, node, 'QSDPA feature width');
      const heads = positiveIntegerParam(node, 'heads', 1);
      const headDimension = feature / heads;
      if (feature % 4 !== 0 || !Number.isInteger(headDimension) ||
          headDimension <= 0 || headDimension > 64 || headDimension % 4 !== 0 ||
          typeof node.params.causal !== 'boolean') {
        unsupportedPhysicalDomain(node,
          'QSDPA requires explicit causal and D/head dimensions divisible by 4 with head_dim <= 64.');
      }
      required.add('getQSDPAShader');
      assertDispatchBound(node, 'QSDPA', [sequence, BigInt(heads), batch], workgroupLimit);
      charge(node, 80n, 'uniform', 'QSDPA parameters');
      if (!inputs.mask) charge(node, 4n, 'storage', 'QSDPA dummy mask');
      continue;
    }

    if (family === 'feature-norm') {
      const source = inputs.input;
      assertDType(node, source, new Set(['float32']), 'input');
      assertDType(node, inputs.weight, new Set(['float32']), 'weight');
      if (inputs.bias) assertDType(node, inputs.bias, new Set(['float32']), 'bias');
      assertDType(node, output, new Set(['float32']), 'output');
      const feature = fixedExtent(input, source.shape.at(-1)!, node, 'normalization feature width');
      const rows = descriptorElements(input, source, true, source.shape.length - 1);
      required.add(node.opType === 'LayerNorm' ? 'getLayerNormShader' : 'getRMSNormShader');
      assertDispatchBound(node, node.opType, [ceilDivide(rows, 64n), 1n, 1n], workgroupLimit);
      if (node.opType === 'LayerNorm') {
        charge(node, BigInt(Math.max(4, feature * 4)), 'storage', 'LayerNorm dummy bias');
      }
      charge(node, 16n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'q-layer-norm') {
      const source = inputs.input;
      assertPerTensorByte(node, source, 'input');
      assertPerTensorByte(node, output, 'output');
      assertDType(node, inputs.weight, new Set(['float32']), 'weight');
      assertDType(node, inputs.bias, new Set(['float32']), 'bias');
      assertFiniteQuantizedAffinePreflight(input, node, inputs.weight, 'QLayerNorm weight');
      assertFiniteQuantizedAffinePreflight(input, node, inputs.bias, 'QLayerNorm bias');
      const feature = fixedExtent(input, source.shape.at(-1)!, node, 'QLayerNorm feature width');
      const rows = descriptorElements(input, source, true, source.shape.length - 1);
      const statsBytes = rows * 8n;
      required.add('getQLayerNormStatsShader');
      required.add('getQLayerNormApplyShader');
      assertDispatchBound(node, 'QLayerNorm stats', [rows, 1n, 1n], workgroupLimit);
      assertDispatchBound(node, 'QLayerNorm apply', [
        ceilDivide(descriptorElements(input, source, true), 256n), 1n, 1n,
      ], workgroupLimit);
      if (feature <= 0) unsupportedPhysicalDomain(node, 'QLayerNorm feature width is empty.');
      charge(node, statsBytes, 'storage', 'QLayerNorm row statistics');
      charge(node, 48n, 'uniform', 'QLayerNorm parameters');
      continue;
    }

    if (family === 'group-norm' || family === 'q-group-norm') {
      const source = inputs.input;
      const batch = axisBound(input, source, 0);
      const channels = fixedExtent(input, source.shape[3], node, 'group-normalization channels');
      const groups = positiveIntegerParam(node, 'num_groups', 0);
      if (channels % groups !== 0) {
        unsupportedPhysicalDomain(node, `num_groups ${groups} does not divide ${channels} channels.`);
      }
      assertDType(node, inputs.weight, new Set(['float32']), 'weight');
      assertDType(node, inputs.bias, new Set(['float32']), 'bias');
      if (family === 'group-norm') {
        assertDType(node, source, new Set(['float32']), 'input');
        assertDType(node, output, new Set(['float32']), 'output');
        required.add('getGroupNormShader');
        assertDispatchBound(node, 'GroupNorm', [
          ceilDivide(batch * BigInt(groups), 64n), 1n, 1n,
        ], workgroupLimit);
        charge(node, 32n, 'uniform', 'GroupNorm parameters');
      } else {
        assertFiniteQuantizedAffinePreflight(input, node, inputs.weight, 'QGroupNorm weight');
        assertFiniteQuantizedAffinePreflight(input, node, inputs.bias, 'QGroupNorm bias');
        assertPerTensorByte(node, source, 'input');
        assertPerTensorByte(node, output, 'output');
        required.add('getQGroupNormStatsShader');
        required.add('getQGroupNormApplyShader');
        const groupCount = batch * BigInt(groups);
        assertDispatchBound(node, 'QGroupNorm stats', [groupCount, 1n, 1n], workgroupLimit);
        assertDispatchBound(node, 'QGroupNorm apply', [
          ceilDivide(descriptorElements(input, source, true), 256n), 1n, 1n,
        ], workgroupLimit);
        charge(node, groupCount * 8n, 'storage', 'QGroupNorm group statistics');
        charge(node, 64n, 'uniform', 'QGroupNorm parameters');
      }
      continue;
    }

    if (family === 'batch-norm-2d') {
      for (const [label, descriptor] of Object.entries(inputs)) {
        assertDType(node, descriptor, new Set(['float32']), label);
      }
      assertDType(node, output, new Set(['float32']), 'output');
      required.add('getBatchNorm2DShader');
      dispatch1D(node, output, 64n);
      charge(node, 32n, 'uniform', 'BatchNorm2D parameters');
      continue;
    }

    if (family === 'conv-2d') {
      const source = inputs.input;
      const weight = inputs.weight;
      assertDType(node, source, new Set(['float32']), 'input');
      assertDType(node, weight, new Set(['float32']), 'weight');
      assertDType(node, output, new Set(['float32']), 'output');
      if (inputs.bias) assertDType(node, inputs.bias, new Set(['float32']), 'bias');
      const batch = axisBound(input, output, 0);
      const outHeight = axisBound(input, output, 1);
      const outWidth = axisBound(input, output, 2);
      const outChannels = fixedExtent(input, output.shape[3], node, 'Conv2D output channels');
      assertDispatchBound(node, 'Conv2D portable', [
        ceilDivide(outWidth, 8n), ceilDivide(outHeight, 8n), batch * BigInt(outChannels),
      ], workgroupLimit);
      required.add('getConv2DShader');
      optional.add('getConv2DRegularC3Out16Shader');
      optional.add('getConv2DRegularOut16Shader');
      optional.add('getConv2DDepthwise8Shader');
      optional.add('getConv2DPointwise16TileShader');
      optional.add('getConv2DPointwise8Vec4Shader');
      optional.add('getConv2DPointwise8Vec2Shader');
      charge(node, BigInt(Math.max(4, outChannels * 4)), 'storage', 'Conv2D dummy bias');
      charge(node, 80n, 'uniform', 'Conv2D parameters');
      continue;
    }

    if (family === 'qconv-2d') {
      const source = inputs.input;
      const weight = inputs.weight;
      assertPerTensorByte(node, source, 'input');
      assertPerTensorByte(node, output, 'output');
      if (!BYTE_DTYPES.has(weight.dtype) || weight.quantization?.scheme !== 'per_axis' ||
          weight.quantization.axis !== 0 || weight.shape.length !== 4) {
        unsupportedPhysicalDomain(node, 'QConv2D requires OHWI I8/U8 axis-0 per-channel weights.');
      }
      const outChannels = fixedExtent(input, output.shape[3], node, 'QConv2D output channels');
      const kernelHeight = fixedExtent(input, weight.shape[1], node, 'QConv2D kernel height');
      const kernelWidth = fixedExtent(input, weight.shape[2], node, 'QConv2D kernel width');
      const inputPerGroup = fixedExtent(input, weight.shape[3], node, 'QConv2D input channels per group');
      const groups = positiveIntegerParam(node, 'groups', 1);
      if (weight.quantization.scales.length !== outChannels ||
          weight.quantization.zero_points.length !== outChannels) {
        unsupportedPhysicalDomain(node, 'QConv2D per-channel affine table does not match output channels.');
      }
      let biasData: Int32Array;
      if (inputs.bias) {
        const data = invariantWeightData(input, node, inputs.bias, 'QConv2D I32 bias');
        if (!(data instanceof Int32Array) || data.length !== outChannels) {
          unsupportedPhysicalDomain(node, 'QConv2D I32 bias payload is not exact.');
        }
        biasData = data;
      } else {
        biasData = new Int32Array(outChannels);
      }
      const inputQuantization = source.quantization as Extract<TensorQuantization, { scheme: 'per_tensor' }>;
      const [inputMinimum, inputMaximum] = dtypeRange(source.dtype);
      const [weightMinimum, weightMaximum] = dtypeRange(weight.dtype);
      const inputMagnitude = Math.max(
        Math.abs(inputMinimum - inputQuantization.zero_point),
        Math.abs(inputMaximum - inputQuantization.zero_point),
      );
      const terms = BigInt(kernelHeight) * BigInt(kernelWidth) * BigInt(inputPerGroup);
      for (let channel = 0; channel < outChannels; channel++) {
        const scale = weight.quantization.scales[channel];
        const zeroPoint = weight.quantization.zero_points[channel];
        const weightMagnitude = Math.max(
          Math.abs(weightMinimum - zeroPoint), Math.abs(weightMaximum - zeroPoint),
        );
        const accumulator = BigInt(inputMagnitude) * BigInt(weightMagnitude) * terms +
          BigInt(Math.abs(biasData[channel]));
        if (!Number.isFinite(scale) || scale <= 0 || !Number.isInteger(zeroPoint) ||
            zeroPoint < weightMinimum || zeroPoint > weightMaximum || accumulator > I32_MAX) {
          unsupportedPhysicalDomain(node,
            `QConv2D channel ${channel} affine or I32 accumulator bound is not representable.`);
        }
      }
      const batch = axisBound(input, output, 0);
      const outHeight = axisBound(input, output, 1);
      const outWidth = axisBound(input, output, 2);
      const outputElements = descriptorElements(input, output, true);
      dispatchLinear2D(node, output, 256n, 'QConv2D scalar');
      if (groups === 1 && outChannels % 4 === 0 && terms >= 16n) {
        const tiled = [
          ceilDivide(BigInt(outChannels), 32n),
          ceilDivide(batch * outHeight * outWidth, 4n), 1n,
        ] as const;
        // The concrete compiler deliberately falls back to the scalar route
        // when a cooperative tile exceeds a device axis. The complete-domain
        // proof must attest that same choice instead of rejecting a graph for
        // an optional tactic that will never be selected.
        const tiledDispatchSupported = tiled.every((count) =>
          count > 0n && count <= workgroupLimit && count <= U32_MAX);
        if (tiledDispatchSupported) {
          if (outChannels >= 32) required.add('getQConv2DTiledShader');
          if (outChannels >= 16) packedDot4.add('getQConv2DDotTiledShader');
        }
      }
      required.add('getQConv2DShader');
      charge(node, BigInt(Math.max(4, outChannels * 4)), 'storage', 'QConv2D scales');
      charge(node, BigInt(Math.max(4, outChannels * 4)), 'storage', 'QConv2D zero points');
      charge(node, BigInt(Math.max(4, outChannels * 4)), 'storage', 'QConv2D bias');
      charge(node, 112n, 'uniform', 'QConv2D parameters');
      continue;
    }

    if (family === 'conv-1d') {
      const source = inputs.input;
      for (const [label, descriptor] of Object.entries(inputs)) {
        assertDType(node, descriptor, new Set(['float32']), label);
      }
      assertDType(node, output, new Set(['float32']), 'output');
      const batch = axisBound(input, output, 0);
      const outLength = axisBound(input, output, 1);
      const outChannels = fixedExtent(input, output.shape[2], node, 'Conv1D output channels');
      required.add('getConv1DShader');
      assertDispatchBound(node, 'Conv1D', [
        ceilDivide(outLength, 64n), BigInt(outChannels), batch,
      ], workgroupLimit);
      charge(node, BigInt(Math.max(4, outChannels * 4)), 'storage', 'Conv1D dummy bias');
      charge(node, 48n, 'uniform', 'Conv1D parameters');
      continue;
    }

    if (family === 'conv-transpose-2d') {
      for (const [label, descriptor] of Object.entries(inputs)) {
        assertDType(node, descriptor, new Set(['float32']), label);
      }
      assertDType(node, output, new Set(['float32']), 'output');
      const batch = axisBound(input, output, 0);
      const outHeight = axisBound(input, output, 1);
      const outWidth = axisBound(input, output, 2);
      const outChannels = fixedExtent(input, output.shape[3], node, 'ConvTranspose2D output channels');
      required.add('getConvTranspose2DShader');
      assertDispatchBound(node, 'ConvTranspose2D', [
        ceilDivide(outWidth, 8n), ceilDivide(outHeight, 8n), batch * BigInt(outChannels),
      ], workgroupLimit);
      charge(node, 16n, 'storage', 'ConvTranspose2D dummy bias');
      charge(node, 64n, 'uniform', 'ConvTranspose2D parameters');
      continue;
    }

    if (family === 'pool-2d') {
      const source = inputs.input;
      const packed = BYTE_DTYPES.has(source.dtype);
      if (packed) {
        if (node.opType !== 'MaxPool2D') {
          unsupportedPhysicalDomain(node, 'AveragePool2D has no raw byte WebGPU route.');
        }
        assertPerTensorByte(node, source, 'input');
        assertPerTensorByte(node, output, 'output');
        if (!sameQuantization(source.quantization, output.quantization)) {
          unsupportedPhysicalDomain(node, 'byte MaxPool2D must preserve its affine descriptor.');
        }
        required.add('getTypedMaxPool2DShader');
        dispatch1D(node, output, 256n, 'typed MaxPool2D');
        charge(node, 64n, 'uniform', 'typed MaxPool2D parameters');
      } else {
        assertDType(node, source, new Set(['float32']), 'input');
        assertDType(node, output, new Set(['float32']), 'output');
        const batch = axisBound(input, output, 0);
        const outHeight = axisBound(input, output, 1);
        const outWidth = axisBound(input, output, 2);
        const channels = axisBound(input, output, 3);
        required.add(node.opType === 'MaxPool2D'
          ? 'getMaxPool2DShader' : 'getAveragePool2DShader');
        assertDispatchBound(node, node.opType, [
          ceilDivide(outWidth, 8n), ceilDivide(outHeight, 8n), batch * channels,
        ], workgroupLimit);
        charge(node, 48n, 'uniform', `${node.opType} parameters`);
      }
      continue;
    }

    if (family === 'global-average-pool') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      required.add('getGlobalAveragePoolShader');
      assertDispatchBound(node, 'GlobalAveragePool', [
        ceilDivide(axisBound(input, inputs.input, 3), 64n),
        axisBound(input, inputs.input, 0), 1n,
      ], workgroupLimit);
      charge(node, 16n, 'uniform', 'GlobalAveragePool parameters');
      continue;
    }

    if (family === 'resize') {
      const source = inputs.input;
      const packed = BYTE_DTYPES.has(source.dtype);
      if (packed) {
        assertPerTensorByte(node, source, 'input');
        assertPerTensorByte(node, output, 'output');
        if (source.dtype !== output.dtype ||
            !sameQuantization(source.quantization, output.quantization)) {
          unsupportedPhysicalDomain(node, 'typed resize must preserve its byte affine domain.');
        }
        required.add('getTypedResizeNearestShader');
        dispatch1D(node, output, 256n, 'typed resize');
      } else {
        assertDType(node, source, new Set(['float32']), 'input');
        assertDType(node, output, new Set(['float32']), 'output');
        required.add('getResizeShader');
        assertDispatchBound(node, node.opType, [
          ceilDivide(axisBound(input, output, 2), 8n),
          ceilDivide(axisBound(input, output, 1), 8n),
          axisBound(input, output, 0) * axisBound(input, output, 3),
        ], workgroupLimit);
      }
      charge(node, 32n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'interpolate-1d') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      required.add('getInterp1DShader');
      assertDispatchBound(node, 'Interpolate1D', [
        ceilDivide(axisBound(input, output, 2), 64n),
        axisBound(input, output, 1), axisBound(input, output, 0),
      ], workgroupLimit);
      charge(node, 16n, 'uniform', 'Interpolate1D parameters');
      continue;
    }

    if (family === 'upsample-nearest-2d') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      required.add('getUpsample2xShader');
      assertDispatchBound(node, 'UpsampleNearest2D', [
        ceilDivide(axisBound(input, output, 2), 8n),
        ceilDivide(axisBound(input, output, 1), 8n),
        axisBound(input, output, 0) * axisBound(input, output, 3),
      ], workgroupLimit);
      charge(node, 16n, 'uniform', 'UpsampleNearest2D parameters');
      continue;
    }

    if (family === 'vision-profile') {
      const source = inputs.input;
      assertDType(node, source, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      const shader = {
        SpatialSoftargmaxY: 'getSpatialSoftargmaxYShader',
        MeanHeight: 'getMeanHeightShader',
        ProfileX: 'getProfileXShader',
        ProfileY: 'getProfileYShader',
      }[node.opType];
      if (shader === undefined) {
        unsupportedPhysicalDomain(node, 'vision-profile shader route is not registered.');
      }
      required.add(shader);
      const x = node.opType === 'ProfileY'
        ? ceilDivide(axisBound(input, source, 1), 64n)
        : ceilDivide(axisBound(input, source, 2), 64n);
      assertDispatchBound(node, node.opType, [
        x, axisBound(input, source, 3), axisBound(input, source, 0),
      ], workgroupLimit);
      charge(node, 16n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'q-activation') {
      assertPerTensorByte(node, inputs.input, 'input');
      assertPerTensorByte(node, output, 'output');
      required.add(node.opType === 'QGELU' ? 'getQGELUShader' : 'getQSiLUShader');
      dispatch1D(node, output, 256n);
      charge(node, 48n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'clip') {
      const source = inputs.input;
      assertDType(node, source, new Set(['float32', 'int32']), 'input');
      if (output.dtype !== source.dtype) {
        unsupportedPhysicalDomain(node, 'Clip input/output storage dtype must match.');
      }
      required.add('getTypedClipShader');
      dispatch1D(node, output, 64n);
      charge(node, 48n, 'uniform', 'Clip parameters');
      continue;
    }

    if (family === 'binary') {
      assertDType(node, inputs.a, new Set(['float32']), 'a');
      assertDType(node, inputs.b, new Set(['float32']), 'b');
      assertDType(node, output, new Set(['float32']), 'output');
      required.add(physicalMethodForBinary(node));
      dispatchLinear2D(node, output, 64n);
      charge(node, BigInt((3 + output.shape.length * 3) * 4),
        'storage', `${node.opType} broadcast strides`);
      continue;
    }

    if (family === 'qadd') {
      assertPerTensorByte(node, inputs.a, 'a');
      assertPerTensorByte(node, inputs.b, 'b');
      assertPerTensorByte(node, output, 'output');
      required.add('getQAddShader');
      dispatch1D(node, output, 256n);
      charge(node, 48n, 'uniform', 'QAdd parameters');
      continue;
    }

    if (family === 'softmax') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      const rows = descriptorElements(input, inputs.input, true, inputs.input.shape.length - 1);
      required.add(node.opType === 'Softmax' ? 'getSoftmaxShader' : 'getLogSoftmaxShader');
      assertDispatchBound(node, node.opType, [ceilDivide(rows, 64n), 1n, 1n], workgroupLimit);
      charge(node, 16n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'reduction') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      const rows = descriptorElements(input, inputs.input, true, inputs.input.shape.length - 1);
      required.add('getReduceShader');
      assertDispatchBound(node, node.opType, [ceilDivide(rows, 64n), 1n, 1n], workgroupLimit);
      charge(node, 16n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'q-masked-mean') {
      const source = inputs.input;
      const quantization = assertPerTensorByte(node, source, 'input');
      assertPerTensorByte(node, output, 'output');
      assertDType(node, inputs.mask, new Set(['int32']), 'mask');
      const maximumSequence = axisBound(input, source, 1);
      const [minimum, maximum] = dtypeRange(source.dtype);
      const centered = Math.max(
        Math.abs(minimum - quantization.zero_point),
        Math.abs(maximum - quantization.zero_point),
      );
      if (maximumSequence * BigInt(centered) > I32_MAX) {
        unsupportedPhysicalDomain(node, 'QMaskedMean maximum centered sequence sum exceeds I32.');
      }
      required.add('getQMaskedMeanShader');
      dispatch1D(node, output, 256n);
      charge(node, 48n, 'uniform', 'QMaskedMean parameters');
      continue;
    }

    if (family === 'argmax' || family === 'qargmax') {
      const source = inputs.input;
      if (family === 'argmax') {
        assertDType(node, source, new Set(['float32']), 'input');
        assertDType(node, output, new Set(['int32']), 'output');
        required.add('getArgMaxI32Shader');
      } else {
        assertPerTensorByte(node, source, 'input');
        assertDType(node, output, new Set(['int32']), 'output');
        if (source.shape.length < 2 || source.shape.length > 8) {
          unsupportedPhysicalDomain(node, 'QArgMax input rank must remain in [2, 8].');
        }
        required.add('getQArgMaxShader');
      }
      const axisRaw = numericParam(node, 'axis', family === 'qargmax' ? -1 : 0);
      const axis = axisRaw < 0 ? axisRaw + source.shape.length : axisRaw;
      const axisSize = axisBound(input, source, axis);
      if (axisSize > I32_MAX) {
        unsupportedPhysicalDomain(node, 'ArgMax axis extent exceeds its signed index range.');
      }
      dispatch1D(node, output, 64n);
      charge(node, 16n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'comparison') {
      assertDType(node, inputs.a, new Set(['int32']), 'a');
      assertDType(node, inputs.b, new Set(['int32']), 'b');
      assertDType(node, output, new Set(['int32']), 'output');
      required.add('getCompareI32Shader');
      dispatch1D(node, output, 64n);
      charge(node, BigInt((3 + output.shape.length * 3) * 4),
        'storage', `${node.opType} broadcast strides`);
      continue;
    }

    if (family === 'not') {
      assertDType(node, inputs.input, new Set(['int32']), 'input');
      assertDType(node, output, new Set(['int32']), 'output');
      required.add('getNotI32Shader');
      dispatch1D(node, output, 64n);
      charge(node, 16n, 'uniform', 'Not parameters');
      continue;
    }

    if (family === 'where') {
      const condition = inputs.condition ?? inputs.mask;
      assertDType(node, condition, new Set(['float32', 'int32']), 'condition');
      assertDType(node, inputs.a, new Set(['float32', 'int32']), 'a');
      if (inputs.b.dtype !== inputs.a.dtype || output.dtype !== inputs.a.dtype ||
          !shapesProvablyEqual(input, condition, output) ||
          !shapesProvablyEqual(input, inputs.a, output) ||
          !shapesProvablyEqual(input, inputs.b, output)) {
        unsupportedPhysicalDomain(node,
          'the WebGPU Where/Mask route requires exact-shape same-dtype data and condition storage.');
      }
      required.add('getWhereTypedShader');
      dispatch1D(node, output, 64n);
      charge(node, 16n, 'uniform', `${node.opType} parameters`);
      continue;
    }

    if (family === 'transpose') {
      const source = inputs.input;
      if (!COPY_DTYPES.has(source.dtype) || output.dtype !== source.dtype ||
          source.shape.length < 1 || source.shape.length > 8 ||
          descriptorElements(input, source, false) !== descriptorElements(input, output, false) ||
          descriptorElements(input, source, true) !== descriptorElements(input, output, true)) {
        unsupportedPhysicalDomain(node,
          'Transpose requires equal-element same-dtype rank-1..8 F32/I32/I8/U8 storage.');
      }
      const packed = BYTE_DTYPES.has(source.dtype);
      if (packed) {
        assertPerTensorByte(node, source, 'input');
        assertPerTensorByte(node, output, 'output');
        if (!sameQuantization(source.quantization, output.quantization)) {
          unsupportedPhysicalDomain(node,
            'the packed Transpose route requires identical per-tensor affine metadata.');
        }
      }
      required.add(packed ? 'getTypedTransposeShader' : 'getGeneralTransposeShader');
      dispatchLinear2D(node, output, packed ? 256n : 64n);
      charge(node, BigInt((3 + source.shape.length * 2) * 4),
        'storage', 'Transpose stride metadata');
      continue;
    }

    if (family === 'concat') {
      const sources = Object.values(inputs);
      if (sources.length < 2 || output.shape.length < 1 || output.shape.length > 8 ||
          sources.some((source) => source.dtype !== output.dtype ||
            source.shape.length !== output.shape.length) || !COPY_DTYPES.has(output.dtype)) {
        unsupportedPhysicalDomain(node,
          'Concat requires two or more same-rank same-dtype F32/I32/I8/U8 inputs.');
      }
      const packed = BYTE_DTYPES.has(output.dtype);
      const sigmoid = node.params.sigmoid === true;
      if ((packed || output.dtype === 'int32') && sigmoid) {
        unsupportedPhysicalDomain(node, 'Concat sigmoid fusion requires F32 storage.');
      }
      if (packed && sources.some((source) =>
        !sameQuantization(source.quantization, output.quantization))) {
        unsupportedPhysicalDomain(node,
          'the packed Concat copy route requires identical affine metadata on every tensor.');
      }
      required.add(packed
        ? 'getTypedConcatCopyShader'
        : output.dtype === 'int32'
          ? 'getConcatCopy32Shader'
          : 'getConcatCopyShader');
      for (const source of sources) {
        dispatch1D(node, source, 64n, `${node.opType} input copy`);
        charge(node, 32n, 'uniform', `${node.opType} input-copy parameters`);
      }
      continue;
    }

    if (family === 'split') {
      const source = inputs.input;
      const splitOutputs = Object.values(outputs);
      assertDType(node, source, new Set(['float32', 'int32']), 'input');
      if (source.shape.length < 1 || source.shape.length > 8 || splitOutputs.length === 0 ||
          splitOutputs.some((candidate) => candidate.dtype !== source.dtype ||
            candidate.shape.length !== source.shape.length)) {
        unsupportedPhysicalDomain(node,
          'Split requires rank-1..8 F32/I32 input and same-rank same-dtype outputs.');
      }
      const rawAxis = numericParam(node, 'axis', 0);
      const axis = rawAxis < 0 ? rawAxis + source.shape.length : rawAxis;
      const sourceAxis = axisBound(input, source, axis);
      const firstOutputAxis = axisBound(input, splitOutputs[0], axis);
      if (axis < 0 || axis >= source.shape.length || sourceAxis % BigInt(splitOutputs.length) !== 0n ||
          splitOutputs.some((candidate) =>
            axisBound(input, candidate, axis, false) !== firstOutputAxis ||
            axisBound(input, candidate, axis, true) !== firstOutputAxis)) {
        unsupportedPhysicalDomain(node,
          'the WebGPU Split route supports only equal-sized output slices.');
      }
      required.add('getSplitShader');
      for (const candidate of splitOutputs) {
        dispatch1D(node, candidate, 64n, 'Split output copy');
        charge(node, 32n, 'uniform', 'Split output-copy parameters');
      }
      continue;
    }

    if (family === 'slice') {
      const source = inputs.input;
      assertDType(node, source, new Set(['float32', 'int32']), 'input');
      if (output.dtype !== source.dtype || source.shape.length < 1 ||
          source.shape.length > 8 || output.shape.length !== source.shape.length) {
        unsupportedPhysicalDomain(node,
          'Slice requires rank-1..8 same-dtype F32/I32 input/output tensors.');
      }
      required.add('getSliceNdShader');
      dispatch1D(node, output, 64n);
      charge(node, 144n, 'uniform', 'Slice rank/coordinate/stride parameters');
      continue;
    }

    if (family === 'pad') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      if (inputs.input.shape.length < 1 || inputs.input.shape.length > 4 ||
          output.shape.length !== inputs.input.shape.length) {
        unsupportedPhysicalDomain(node,
          'the WebGPU Pad shader supports only rank-1..4 F32 tensors.');
      }
      required.add('getPadShader');
      dispatch1D(node, output, 64n);
      charge(node, 64n, 'uniform', 'Pad parameters');
      continue;
    }

    if (family === 'expand') {
      const source = inputs.input;
      if (!COPY_DTYPES.has(source.dtype) || output.dtype !== source.dtype ||
          source.shape.length < 1 || source.shape.length > output.shape.length ||
          output.shape.length > 8) {
        unsupportedPhysicalDomain(node,
          'Expand/Broadcast requires same-dtype rank-1..8 F32/I32/I8/U8 storage.');
      }
      const packed = BYTE_DTYPES.has(source.dtype);
      if (packed) {
        assertPerTensorByte(node, source, 'input');
        assertPerTensorByte(node, output, 'output');
        if (!sameQuantization(source.quantization, output.quantization)) {
          unsupportedPhysicalDomain(node,
            'the packed Expand route requires identical per-tensor affine metadata.');
        }
      }
      required.add(packed ? 'getTypedExpandShader' : 'getExpandShader');
      dispatchLinear2D(node, output, packed ? 256n : 64n);
      charge(node, 80n, 'uniform', `${node.opType} shape/stride parameters`);
      continue;
    }

    if (family === 'gather') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, inputs.indices, new Set(['int32']), 'indices');
      assertDType(node, output, new Set(['float32']), 'output');
      if (inputs.input.shape.length < 1 || inputs.input.shape.length > 8 ||
          inputs.indices.shape.length > 8 || output.shape.length > 8) {
        unsupportedPhysicalDomain(node,
          'Gather requires rank-1..8 F32 data, rank-0..8 I32 indices, and output rank at most 8.');
      }
      const bank = input.graph.weights[inputs.input.name]?.bank;
      let axis = numericParam(node, 'axis', 0);
      if (!Number.isSafeInteger(axis)) {
        unsupportedPhysicalDomain(node, 'Gather axis is not a safe integer.');
      }
      if (axis < 0) axis += inputs.input.shape.length;
      if (axis < 0 || axis >= inputs.input.shape.length) {
        unsupportedPhysicalDomain(node, 'Gather axis is outside the input rank.');
      }
      if (bank !== null && bank !== undefined && axis !== 0) {
        unsupportedPhysicalDomain(node,
          'a partially resident bank can be gathered only along its slot axis 0.');
      }
      const minimumAxisExtent = axisBound(input, inputs.input, axis, false);
      const maximumAxisExtent = axisBound(input, inputs.input, axis, true);
      if (maximumAxisExtent > I32_MAX) {
        unsupportedPhysicalDomain(node,
          'Gather indexed-axis extent exceeds its signed index representation.');
      }
      assertI32ValuePreflight(
        input,
        node,
        inputs.indices,
        -minimumAxisExtent,
        minimumAxisExtent - 1n,
        'Gather indices',
      );
      // Bank residency belongs to an ExecutionContext, not the immutable
      // model compile view. The value-domain proof above is sufficient for a
      // fully resident bank. A bound graph that requests only some slots
      // carries residentSlots metadata and must preflight device-produced
      // indices against that exact context before dispatch.
      required.add('getGatherInt32Shader');
      dispatch1D(node, output, 64n);
      charge(node, 96n, 'uniform', 'Gather shape/stride/bank parameters');
      charge(
        node,
        bank === null || bank === undefined
          ? 4n
          : BigInt(input.graph.weights[inputs.input.name].shape[0]) * 4n,
        'storage',
        'Gather resident slot table',
      );
      continue;
    }

    if (family === 'gather-elements') {
      assertDType(node, inputs.input, new Set(['float32']), 'input');
      assertDType(node, inputs.indices, new Set(['int32']), 'indices');
      assertDType(node, output, new Set(['float32']), 'output');
      if (inputs.input.shape.length < 1 || inputs.input.shape.length > 8 ||
          inputs.indices.shape.length !== inputs.input.shape.length ||
          output.shape.length !== inputs.input.shape.length) {
        unsupportedPhysicalDomain(node,
          'GatherElements requires rank-1..8 F32 data/output and equal-rank I32 indices.');
      }
      if (input.graph.weights[inputs.input.name]?.bank != null) {
        unsupportedPhysicalDomain(node,
          'GatherElements has no global-slot translation route for a partially resident bank.');
      }
      let axis = numericParam(node, 'axis', 0);
      if (!Number.isSafeInteger(axis)) {
        unsupportedPhysicalDomain(node, 'GatherElements axis is not a safe integer.');
      }
      if (axis < 0) axis += inputs.input.shape.length;
      if (axis < 0 || axis >= inputs.input.shape.length) {
        unsupportedPhysicalDomain(node, 'GatherElements axis is outside the input rank.');
      }
      const minimumAxisExtent = axisBound(input, inputs.input, axis, false);
      const maximumAxisExtent = axisBound(input, inputs.input, axis, true);
      if (maximumAxisExtent > I32_MAX) {
        unsupportedPhysicalDomain(node,
          'GatherElements indexed-axis extent exceeds its signed index representation.');
      }
      assertI32ValuePreflight(
        input,
        node,
        inputs.indices,
        -minimumAxisExtent,
        minimumAxisExtent - 1n,
        'GatherElements indices',
      );
      required.add('getGatherElementsShader');
      dispatch1D(node, output, 64n);
      charge(node, 80n, 'uniform', 'GatherElements shape/stride parameters');
      continue;
    }

    if (family === 'quantize') {
      const source = inputs.input;
      assertDType(node, source, new Set(['float32']), 'input');
      const quantization = assertPerTensorByte(node, output, 'output');
      assertInvariantQuantizationParameters(
        input, node, inputs.scale, inputs.zero_point, quantization,
      );
      required.add('getQuantizeLinearShader');
      dispatchLinear2D(node, output, 256n);
      charge(node, 16n, 'uniform', 'QuantizeLinear parameters');
      continue;
    }

    if (family === 'dequantize') {
      const source = inputs.input;
      const quantization = assertPerTensorByte(node, source, 'input');
      assertDType(node, output, new Set(['float32']), 'output');
      assertInvariantQuantizationParameters(
        input, node, inputs.scale, inputs.zero_point, quantization,
      );
      required.add('getDequantizeLinearShader');
      dispatchLinear2D(node, output, 64n);
      charge(node, 32n, 'uniform', 'DequantizeLinear parameters');
      continue;
    }

    if (family === 'requantize') {
      assertPerTensorByte(node, inputs.input, 'input');
      assertPerTensorByte(node, output, 'output');
      required.add('getRequantizeLinearShader');
      dispatch1D(node, output, 256n);
      charge(node, 48n, 'uniform', 'RequantizeLinear parameters');
      continue;
    }

    if (family === 'cast') {
      assertDType(node, inputs.input, COPY_DTYPES, 'input');
      assertDType(node, output, COPY_DTYPES, 'output');
      required.add('getCastShader');
      dispatch1D(node, output, BYTE_DTYPES.has(output.dtype) ? 256n : 64n);
      charge(node, 16n, 'uniform', 'Cast parameters');
      continue;
    }

    if (family === 'embedding') {
      const ids = inputs.input;
      const weight = inputs.weight;
      assertDType(node, ids, new Set(['int32']), 'input');
      assertDType(node, weight, new Set(['float32']), 'weight');
      assertDType(node, output, new Set(['float32']), 'output');
      if (ids.shape.length < 1 || weight.shape.length !== 2 ||
          fixedExtent(input, weight.shape[0], node, 'vocabulary') <= 0 ||
          fixedExtent(input, weight.shape[1], node, 'embedding width') <= 0) {
        unsupportedPhysicalDomain(node,
          'Embedding requires rank-positive I32 IDs and a fixed rank-2 F32 table.');
      }
      if (input.graph.weights[weight.name]?.bank != null) {
        unsupportedPhysicalDomain(node,
          'Embedding has no global-slot translation route for a partially resident table.');
      }
      const vocabulary = fixedExtent(input, weight.shape[0], node, 'vocabulary');
      assertI32ValuePreflight(
        input, node, ids, 0n, BigInt(vocabulary - 1), 'Embedding IDs',
      );
      required.add('getEmbeddingShader');
      assertDispatchBound(node, 'Embedding token rows', [
        ceilDivide(descriptorElements(input, ids, true), 64n), 1n, 1n,
      ], workgroupLimit);
      charge(node, 16n, 'uniform', 'Embedding parameters');
      continue;
    }

    if (family === 'qembedding') {
      const ids = inputs.input;
      const weight = inputs.weight;
      assertDType(node, ids, new Set(['int32']), 'input');
      if (ids.shape.length < 1 || weight.shape.length !== 2 ||
          !BYTE_DTYPES.has(weight.dtype)) {
        unsupportedPhysicalDomain(node,
          'QEmbedding requires rank-positive I32 IDs and a rank-2 I8/U8 table.');
      }
      const vocabulary = fixedExtent(input, weight.shape[0], node, 'vocabulary');
      const hidden = fixedExtent(input, weight.shape[1], node, 'embedding width');
      if (input.graph.weights[weight.name]?.bank != null) {
        unsupportedPhysicalDomain(node,
          'QEmbedding has no global-slot translation route for a partially resident table.');
      }
      const weightQuantization = weight.quantization;
      assertPerTensorByte(node, output, 'output');
      if (weightQuantization?.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
          weightQuantization.scales.length !== vocabulary ||
          weightQuantization.zero_points.length !== vocabulary) {
        unsupportedPhysicalDomain(node,
          'QEmbedding requires axis-0 table metadata and complete ID-range preflight.');
      }
      assertI32ValuePreflight(
        input, node, ids, 0n, BigInt(vocabulary - 1), 'QEmbedding IDs',
      );
      const [minimum, maximum] = dtypeRange(weight.dtype);
      for (let row = 0; row < vocabulary; row++) {
        const scale = Math.fround(weightQuantization.scales[row]);
        const zeroPoint = weightQuantization.zero_points[row];
        if (!Number.isFinite(scale) || scale <= 0 || !Number.isInteger(zeroPoint) ||
            zeroPoint < minimum || zeroPoint > maximum) {
          unsupportedPhysicalDomain(node,
            `QEmbedding row ${row} has an unrepresentable affine descriptor.`);
        }
      }
      required.add('getQEmbeddingShader');
      dispatch1D(node, output, 256n);
      charge(node, BigInt(vocabulary * 4), 'storage', 'QEmbedding row scales');
      charge(node, BigInt(vocabulary * 4), 'storage', 'QEmbedding row zero points');
      charge(node, 32n, 'uniform', 'QEmbedding parameters');
      if (hidden <= 0) {
        unsupportedPhysicalDomain(node, 'QEmbedding hidden width must be positive.');
      }
      continue;
    }

    unsupportedPhysicalDomain(node, `physical proof family '${family}' is not implemented.`);
  }

  if (maximumAuxiliaryBytes > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WebGPU auxiliary allocation maximum ${maximumAuxiliaryBytes} exceeds safe integer accounting.`, {
        phase: 'compilation', backend: 'webgpu',
      });
  }
  return Object.freeze({
    requiredShaderMethods: Object.freeze([...required].sort()),
    optionalShaderMethods: Object.freeze([...optional].sort()),
    packedDot4OptionalShaderMethods: Object.freeze([...packedDot4].sort()),
    maximumAuxiliaryBytes: Number(maximumAuxiliaryBytes),
  });
}
