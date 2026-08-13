import { createBoundExecutionGraph } from '../core/BoundExecutionGraph.js';
import type { BackendExecutionSnapshot } from '../core/ExecutionResult.js';
import { Model } from '../core/Model.js';
import type { LogicalDomainTensorDescriptor } from '../core/ResolvedShapePlan.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';
import { kernelRoute, operatorShapeContract } from '../generated/kernelRegistry.js';
import {
  runtimeDTypeBytes,
} from '../ops/shapeSystem.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import {
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  createBackendProviderCapabilities,
  requireHostExecutionInputs,
  type BackendDeviceIdentity,
  type BackendLogicalCompileInput,
  type BackendProvider,
  type BackendProviderCapabilities,
  type BackendProviderCompilationEvidence,
  type BackendProviderCompiledModel,
  type BackendProviderCompileOptions,
  type BackendProviderContextOptions,
  type BackendProviderExecutionContext,
  type BackendResolvedExecutionRequest,
} from './BackendProvider.js';
import { WebNNEngine } from './WebNNEngine.js';

const WEBNN_MAX_RANK = 8;
// WebNN defines both every dimension and a descriptor's complete element
// count as a positive WebIDL `long`, even though the exposed shape members are
// `unsigned long` values.
const WEBNN_MAX_VALID_DIMENSION = 0x7fffffff;
const WEBNN_MAX_UNSIGNED_LONG = 0xffffffff;
const WEBNN_VARIANT_CACHE_ENTRIES = 8;
const WEBNN_VARIANT_CACHE_METADATA_BYTES = 1024 * 1024;
const UTF8_ENCODER = new TextEncoder();

interface WebNNResourceDomainProof {
  readonly maximumTensorBytes: number;
  readonly maximumResidentBytes: number;
  readonly resourceLimitBytes: null;
}

interface WebNNAuxiliaryDomainProof {
  readonly totalBytes: bigint;
  readonly maximumTensorBytes: bigint;
}

interface WebNNVariant {
  readonly signature: string;
  readonly engine: WebNNEngine;
  readonly metadataBytes: number;
  readonly graphBuildTimeMs: number;
}

interface VariantSelection {
  readonly variant: WebNNVariant;
  readonly cacheHit: boolean;
  readonly coalesced: boolean;
  readonly transient: boolean;
  readonly evictions: number;
}

interface PublishedVariant {
  readonly variant: WebNNVariant;
  readonly evictions: number;
}

function storageDType(value: unknown): RuntimeDType | null {
  if (value instanceof Float32Array) return 'float32';
  if (value instanceof Int32Array) return 'int32';
  if (value instanceof Int8Array) return 'int8';
  if (value instanceof Uint8Array || value instanceof Uint8ClampedArray) return 'uint8';
  return null;
}

function checkedAdd(left: bigint, right: bigint, label: string): bigint {
  const value = left + right;
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED', `WebNN ${label} exceeds exact arithmetic.`, {
      phase: 'compilation', backend: 'webnn',
    });
  }
  return value;
}

function checkedMultiply(left: bigint, right: bigint, label: string): bigint {
  const value = left * right;
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED', `WebNN ${label} exceeds exact arithmetic.`, {
      phase: 'compilation', backend: 'webnn',
    });
  }
  return value;
}

function maximumDimension(
  input: BackendLogicalCompileInput,
  dimension: number | string,
  label: string,
): bigint {
  const value = typeof dimension === 'number'
    ? dimension
    : input.graph.dimensions[dimension]?.max;
  if (!Number.isSafeInteger(value) || (value as number) <= 0 ||
      (value as number) > WEBNN_MAX_VALID_DIMENSION) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WebNN cannot represent ${label}.`, { phase: 'compilation', backend: 'webnn' });
  }
  return BigInt(value as number);
}

function descriptorMaximumElements(
  input: BackendLogicalCompileInput,
  descriptor: LogicalDomainTensorDescriptor,
  label: string,
): bigint {
  if (descriptor.shape.length > WEBNN_MAX_RANK) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WebNN ${label} rank ${descriptor.shape.length} exceeds ${WEBNN_MAX_RANK}.`, {
        phase: 'compilation', backend: 'webnn',
      });
  }
  let elements = 1n;
  for (let axis = 0; axis < descriptor.shape.length; axis++) {
    elements *= maximumDimension(input, descriptor.shape[axis], `${label} axis ${axis}`);
    if (elements > BigInt(WEBNN_MAX_VALID_DIMENSION)) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebNN ${label} element count is not a valid WebNN dimension.`, {
          phase: 'compilation', backend: 'webnn',
        });
    }
  }
  return elements;
}

function requireUnsignedLongValues(
  node: { id: string; opType: string },
  value: unknown,
  label: string,
  fallback: number,
  lengths: readonly number[],
  minimum: number,
): void {
  const values = value === undefined ? [fallback] : Array.isArray(value) ? value : [value];
  if (!lengths.includes(values.length) || values.some((entry) =>
    !Number.isSafeInteger(entry) || (entry as number) < minimum ||
    (entry as number) > WEBNN_MAX_UNSIGNED_LONG)) {
    physicalFailure(node,
      `${label} must contain ${lengths.join(' or ')} WebIDL unsigned-long value(s).`);
  }
}

function physicalFailure(node: { id: string; opType: string }, reason: string): never {
  throw new VolvoxAIError('BACKEND_UNSUPPORTED',
    `WebNN physical route for '${node.opType}' at node '${node.id}' is unproved: ${reason}`, {
      phase: 'compilation', backend: 'webnn', node: node.id,
    });
}

function requirePorts(
  node: { id: string; opType: string },
  value: Readonly<Record<string, LogicalDomainTensorDescriptor>>,
  required: readonly string[],
  optional: readonly string[] = [],
): void {
  const names = Object.keys(value);
  if (required.some((name) => value[name] === undefined) ||
      names.some((name) => !required.includes(name) && !optional.includes(name))) {
    physicalFailure(node, `ports [${names.join(',')}] do not match the WebNN route.`);
  }
}

function requireDType(
  node: { id: string; opType: string },
  descriptors: readonly LogicalDomainTensorDescriptor[],
  allowed: readonly RuntimeDType[],
): void {
  if (descriptors.some((descriptor) => !allowed.includes(descriptor.dtype))) {
    physicalFailure(node, `dtype route requires ${allowed.join(' or ')} storage.`);
  }
}

function normalizedAxis(value: unknown, rank: number): number | null {
  if (value === undefined) return rank - 1;
  if (!Number.isSafeInteger(value)) return null;
  const axis = (value as number) < 0 ? (value as number) + rank : value as number;
  return axis >= 0 && axis < rank ? axis : null;
}

type WebNNTensorDescriptorLike = Pick<LogicalDomainTensorDescriptor, 'dtype' | 'shape'>;
type WebNNSupportRecord = Readonly<Record<string, unknown>>;

function supportRecord(value: unknown): WebNNSupportRecord | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as WebNNSupportRecord
    : null;
}

function supportEntry(
  node: { id: string; opType: string },
  limits: WebNNSupportRecord,
  operation: string,
): WebNNSupportRecord {
  const entry = supportRecord(limits[operation]);
  if (!entry) physicalFailure(node, `MLContext does not report '${operation}' support limits.`);
  return entry;
}

function assertTensorLimit(
  node: { id: string; opType: string },
  limit: unknown,
  descriptor: WebNNTensorDescriptorLike,
  label: string,
): void {
  const record = supportRecord(limit);
  const rankRange = supportRecord(record?.rankRange);
  const dataTypes = record?.dataTypes;
  const minimum = rankRange?.min;
  const maximum = rankRange?.max;
  if (!Array.isArray(dataTypes) || !dataTypes.includes(descriptor.dtype) ||
      !Number.isSafeInteger(minimum) || !Number.isSafeInteger(maximum) ||
      (minimum as number) < 0 || (maximum as number) < (minimum as number) ||
      descriptor.shape.length < (minimum as number) ||
      descriptor.shape.length > (maximum as number)) {
    physicalFailure(node,
      `${label} dtype '${descriptor.dtype}' and rank ${descriptor.shape.length} are outside ` +
      'the MLContext support limits.');
  }
}

function assertUnarySupport(
  node: { id: string; opType: string },
  limits: WebNNSupportRecord,
  operation: string,
  input: WebNNTensorDescriptorLike,
  output: WebNNTensorDescriptorLike,
): void {
  const entry = supportEntry(node, limits, operation);
  assertTensorLimit(node, entry.input, input, `${operation}.input`);
  assertTensorLimit(node, entry.output, output, `${operation}.output`);
}

function assertBinarySupport(
  node: { id: string; opType: string },
  limits: WebNNSupportRecord,
  operation: string,
  a: WebNNTensorDescriptorLike,
  b: WebNNTensorDescriptorLike,
  output: WebNNTensorDescriptorLike,
): void {
  const entry = supportEntry(node, limits, operation);
  assertTensorLimit(node, entry.a, a, `${operation}.a`);
  assertTensorLimit(node, entry.b, b, `${operation}.b`);
  assertTensorLimit(node, entry.output, output, `${operation}.output`);
}

function denseUsesOutputMajorWeights(node: {
  opType: string;
  params: Readonly<Record<string, unknown>>;
}): boolean {
  if (node.params.weight_layout !== undefined) return node.params.weight_layout === 'dout_din';
  if (node.params.transB !== undefined) return node.params.transB === true;
  return node.opType === 'Linear';
}

function assertWebNNPrimitiveSupport(
  node: { id: string; opType: string; params: Readonly<Record<string, unknown>> },
  proof: BackendLogicalCompileInput['shapeDomainProof']['nodes'][number],
  limits: WebNNSupportRecord,
): void {
  const input = proof.inputs;
  const output = proof.outputs.out;
  if (['MatMul', 'Linear', 'Gemm'].includes(node.opType)) {
    if (denseUsesOutputMajorWeights(node)) {
      assertUnarySupport(node, limits, 'transpose', input.weight, input.weight);
    }
    assertBinarySupport(node, limits, 'matmul', input.input, input.weight, output);
    if (input.bias) assertBinarySupport(node, limits, 'add', output, input.bias, output);
    return;
  }
  const unaryOperation: Readonly<Record<string, string>> = {
    ReLU: 'relu', GELU: 'gelu', Sigmoid: 'sigmoid', Softmax: 'softmax',
    Reshape: 'reshape', Flatten: 'reshape',
  };
  if (unaryOperation[node.opType]) {
    assertUnarySupport(node, limits, unaryOperation[node.opType], input.input, output);
    return;
  }
  if (node.opType === 'SiLU') {
    assertUnarySupport(node, limits, 'sigmoid', input.input, output);
    assertBinarySupport(node, limits, 'mul', input.input, output, output);
    return;
  }
  if (node.opType === 'Add' || node.opType === 'Mul') {
    assertBinarySupport(node, limits, node.opType.toLowerCase(), input.a, input.b, output);
    if (node.opType === 'Add' && node.params.relu === 1) {
      assertUnarySupport(node, limits, 'relu', output, output);
    } else if (node.opType === 'Add' && node.params.relu === 2) {
      assertUnarySupport(node, limits, 'clamp', output, output);
    }
    return;
  }
  if (node.opType === 'LayerNorm') {
    const entry = supportEntry(node, limits, 'layerNormalization');
    assertTensorLimit(node, entry.input, input.input, 'layerNormalization.input');
    assertTensorLimit(node, entry.scale, input.weight, 'layerNormalization.scale');
    if (input.bias) assertTensorLimit(node, entry.bias, input.bias, 'layerNormalization.bias');
    assertTensorLimit(node, entry.output, output, 'layerNormalization.output');
    return;
  }
  if (node.opType === 'Conv2D') {
    const entry = supportEntry(node, limits, 'conv2d');
    assertTensorLimit(node, entry.input, input.input, 'conv2d.input');
    assertTensorLimit(node, entry.filter, input.weight, 'conv2d.filter');
    if (input.bias) assertTensorLimit(node, entry.bias, input.bias, 'conv2d.bias');
    assertTensorLimit(node, entry.output, output, 'conv2d.output');
    if (node.params.relu === 1) assertUnarySupport(node, limits, 'relu', output, output);
    if (node.params.relu === 2) assertUnarySupport(node, limits, 'clamp', output, output);
    return;
  }
  if (node.opType === 'Embedding') {
    const entry = supportEntry(node, limits, 'gather');
    assertTensorLimit(node, entry.input, input.weight, 'gather.input');
    assertTensorLimit(node, entry.indices, input.input, 'gather.indices');
    assertTensorLimit(node, entry.output, output, 'gather.output');
    return;
  }
  if (node.opType === 'SDPA') {
    const rank3 = { dtype: 'float32' as const, shape: [1, 1, 1] };
    const rank4 = { dtype: 'float32' as const, shape: [1, 1, 1, 1] };
    const scalar = { dtype: 'float32' as const, shape: [1] };
    const matrix = { dtype: 'float32' as const, shape: [1, 1] };
    assertUnarySupport(node, limits, 'reshape', input.qkv, rank3);
    assertUnarySupport(node, limits, 'slice', rank3, rank3);
    assertUnarySupport(node, limits, 'reshape', rank3, rank4);
    assertUnarySupport(node, limits, 'transpose', rank4, rank4);
    assertBinarySupport(node, limits, 'matmul', rank4, rank4, rank4);
    assertTensorLimit(node, limits.constant, scalar, 'constant scale');
    assertBinarySupport(node, limits, 'mul', rank4, scalar, rank4);
    if (node.params.causal === true) {
      assertTensorLimit(node, limits.constant, matrix, 'constant causal mask');
      assertBinarySupport(node, limits, 'add', rank4, matrix, rank4);
    }
    assertUnarySupport(node, limits, 'softmax', rank4, rank4);
    assertUnarySupport(node, limits, 'reshape', rank4, output);
    return;
  }
  physicalFailure(node, 'the WebNN primitive decomposition is absent.');
}

function checkedWebNNPhysicalDomain(
  input: BackendLogicalCompileInput,
  limits: WebNNSupportRecord,
): WebNNAuxiliaryDomainProof {
  if (input.shapeDomainProof.nodes.length !== input.graph.nodes.length) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      "Backend 'webnn' received incomplete canonical domain evidence.", {
        phase: 'compilation', backend: 'webnn',
      });
  }
  let auxiliaryBytes = 0n;
  let maximumAuxiliaryTensorBytes = 0n;
  const chargeAuxiliary = (bytes: bigint, label: string, copies = 1n): void => {
    if (bytes > maximumAuxiliaryTensorBytes) maximumAuxiliaryTensorBytes = bytes;
    auxiliaryBytes = checkedAdd(
      auxiliaryBytes,
      checkedMultiply(bytes, copies, `${label} byte copies`),
      `${label} auxiliary bytes`,
    );
  };
  for (let index = 0; index < input.graph.nodes.length; index++) {
    const node = input.graph.nodes[index];
    const proof = input.shapeDomainProof.nodes[index];
    const contract = operatorShapeContract(node.opType);
    if (!contract || contract.classification !== 'canonical' ||
        !kernelRoute('webnn', node.opType) || proof?.id !== node.id ||
        proof.opType !== node.opType || proof.shapeFunctionId !== contract.shapeFunctionId) {
      physicalFailure(node, 'canonical registry route evidence is absent or inconsistent.');
    }
    const inputs = Object.values(proof.inputs);
    const outputs = Object.values(proof.outputs);
    for (const descriptor of [...inputs, ...outputs]) {
      descriptorMaximumElements(input, descriptor, `node '${node.id}' tensor '${descriptor.name}'`);
    }

    if (['MatMul', 'Linear', 'Gemm'].includes(node.opType)) {
      requirePorts(node, proof.inputs, ['input', 'weight'], ['bias', 'a', 'b']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [...inputs, ...outputs], ['float32']);
      if (proof.inputs.input.shape.length < 2) {
        physicalFailure(node, 'the current WebNN MatMul route requires rank 2 or greater.');
      }
      if (denseUsesOutputMajorWeights(node)) {
        chargeAuxiliary(
          descriptorMaximumElements(input, proof.inputs.weight, `node '${node.id}' transpose`) * 4n,
          `node '${node.id}' transposed weight`,
        );
      }
      if (proof.inputs.bias) {
        chargeAuxiliary(
          descriptorMaximumElements(input, proof.outputs.out, `node '${node.id}' matmul result`) * 4n,
          `node '${node.id}' pre-bias result`,
        );
      }
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    if (['ReLU', 'GELU', 'SiLU', 'Sigmoid'].includes(node.opType)) {
      requirePorts(node, proof.inputs, ['input']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [...inputs, ...outputs], ['float32']);
      if (node.opType === 'GELU' && node.params.approximate !== undefined &&
          node.params.approximate !== 'none') {
        physicalFailure(node, "GELU requires approximate='none'.");
      }
      if (node.opType === 'SiLU') {
        chargeAuxiliary(
          descriptorMaximumElements(input, proof.outputs.out, `node '${node.id}' sigmoid`) * 4n,
          `node '${node.id}' sigmoid result`,
        );
      }
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    if (node.opType === 'Add' || node.opType === 'Mul') {
      requirePorts(node, proof.inputs, ['a', 'b']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [...inputs, ...outputs], ['float32']);
      if (node.opType === 'Add' && (node.params.relu === 1 || node.params.relu === 2)) {
        chargeAuxiliary(
          descriptorMaximumElements(input, proof.outputs.out, `node '${node.id}' add result`) * 4n,
          `node '${node.id}' pre-activation result`,
        );
      }
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    if (node.opType === 'Softmax') {
      requirePorts(node, proof.inputs, ['input']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [...inputs, ...outputs], ['float32']);
      if (normalizedAxis(node.params.axis, proof.inputs.input.shape.length) !==
          proof.inputs.input.shape.length - 1) {
        physicalFailure(node, 'the current WebNN route supports only the final Softmax axis.');
      }
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    if (node.opType === 'LayerNorm') {
      requirePorts(node, proof.inputs, ['input', 'weight'], ['bias']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [...inputs, ...outputs], ['float32']);
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    if (node.opType === 'Reshape' || node.opType === 'Flatten') {
      requirePorts(node, proof.inputs, ['input']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [...inputs, ...outputs], ['float32', 'int32', 'int8', 'uint8']);
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    if (node.opType === 'Conv2D') {
      requirePorts(node, proof.inputs, ['input', 'weight'], ['bias']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [...inputs, ...outputs], ['float32']);
      if (proof.inputs.input.shape.length !== 4 || proof.inputs.weight.shape.length !== 4 ||
          proof.outputs.out.shape.length !== 4) {
        physicalFailure(node, 'Conv2D requires rank-4 input, weight, and output tensors.');
      }
      const dataLayout = node.params.data_layout ?? 'NHWC';
      const weightLayout = node.params.weight_layout ?? 'HWIO';
      if (dataLayout !== 'NHWC' || weightLayout !== 'HWIO') {
        physicalFailure(node, 'the current WebNN Conv2D route requires NHWC/HWIO.');
      }
      requireUnsignedLongValues(node, node.params.stride, 'stride', 1, [1, 2], 1);
      requireUnsignedLongValues(node, node.params.dilation, 'dilation', 1, [1, 2], 1);
      if (node.params.pads === undefined) {
        requireUnsignedLongValues(node, node.params.padding, 'padding', 0, [1, 2], 0);
      } else {
        requireUnsignedLongValues(node, node.params.pads, 'pads', 0, [4], 0);
      }
      requireUnsignedLongValues(node, node.params.groups, 'groups', 1, [1], 1);
      if (node.params.relu === 1 || node.params.relu === 2) {
        chargeAuxiliary(
          descriptorMaximumElements(input, proof.outputs.out, `node '${node.id}' convolution result`) * 4n,
          `node '${node.id}' pre-activation result`,
        );
      }
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    if (node.opType === 'Embedding') {
      requirePorts(node, proof.inputs, ['input', 'weight']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [proof.inputs.input], ['int32']);
      requireDType(node, [proof.inputs.weight, proof.outputs.out], ['float32']);
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    if (node.opType === 'SDPA') {
      requirePorts(node, proof.inputs, ['qkv']);
      requirePorts(node, proof.outputs, ['out']);
      requireDType(node, [...inputs, ...outputs], ['float32']);
      const rank = proof.inputs.qkv.shape.length;
      if (rank !== 2 && rank !== 3) physicalFailure(node, 'SDPA requires rank 2 or 3.');
      const sequenceAxis = rank - 2;
      const sequence = maximumDimension(
        input, proof.inputs.qkv.shape[sequenceAxis], `node '${node.id}' sequence extent`,
      );
      const batch = rank === 2
        ? 1n
        : maximumDimension(input, proof.inputs.qkv.shape[0], `node '${node.id}' batch extent`);
      const heads = BigInt(node.params.heads as number);
      const qkvBytes = descriptorMaximumElements(
        input, proof.inputs.qkv, `node '${node.id}' packed QKV`,
      ) * 4n;
      // The current WebNN route explicitly decomposes QKV slicing/head
      // transforms and output transforms. Five packed-QKV-sized buffers are a
      // conservative upper bound for all of those operands without assuming
      // implementation-private aliasing.
      chargeAuxiliary(qkvBytes, `node '${node.id}' QKV decomposition`, 5n);
      const scoreElements = checkedMultiply(
        checkedMultiply(batch, heads, `node '${node.id}' batch/head product`),
        checkedMultiply(sequence, sequence, `node '${node.id}' score square`),
        `node '${node.id}' score elements`,
      );
      if (scoreElements > BigInt(WEBNN_MAX_VALID_DIMENSION)) {
        physicalFailure(node, 'the decomposed attention score element count exceeds WebNN limits.');
      }
      const scoreBytes = checkedMultiply(scoreElements, 4n, `node '${node.id}' score bytes`);
      chargeAuxiliary(
        scoreBytes,
        `node '${node.id}' score decomposition`,
        node.params.causal === true ? 4n : 3n,
      );
      chargeAuxiliary(4n, `node '${node.id}' scale constant`);
      if (node.params.causal === true) {
        const maskElements = checkedMultiply(
          sequence,
          sequence,
          `node '${node.id}' mask square`,
        );
        if (maskElements > BigInt(WEBNN_MAX_VALID_DIMENSION)) {
          physicalFailure(node, 'the causal-mask element count exceeds WebNN limits.');
        }
        const maskBytes = checkedMultiply(
          maskElements,
          4n,
          `node '${node.id}' mask bytes`,
        );
        chargeAuxiliary(maskBytes, `node '${node.id}' causal mask`);
      }
      assertWebNNPrimitiveSupport(node, proof, limits);
      continue;
    }
    physicalFailure(node, 'no complete physical WebNN predicate is implemented.');
  }
  return Object.freeze({
    totalBytes: auxiliaryBytes,
    maximumTensorBytes: maximumAuxiliaryTensorBytes,
  });
}

function checkedWebNNResources(
  input: BackendLogicalCompileInput,
  limits: WebNNSupportRecord,
): WebNNResourceDomainProof {
  const auxiliary = checkedWebNNPhysicalDomain(input, limits);
  let maximumTensorBytes = auxiliary.maximumTensorBytes;
  let graphBytes = auxiliary.totalBytes;
  for (const descriptor of Object.values(input.graph.tensors)) {
    const logical = {
      name: descriptor.name,
      dtype: descriptor.dtype,
      shape: descriptor.shape,
      ...(descriptor.quantization === undefined ? {} : { quantization: descriptor.quantization }),
    } as LogicalDomainTensorDescriptor;
    const bytes = descriptorMaximumElements(input, logical, `tensor '${descriptor.name}'`) *
      BigInt(runtimeDTypeBytes(descriptor.dtype));
    maximumTensorBytes = bytes > maximumTensorBytes ? bytes : maximumTensorBytes;
    graphBytes = checkedAdd(graphBytes, bytes, 'graph byte maximum');
  }
  const graphNode = { id: '<graph>', opType: 'Graph' };
  for (const descriptor of Object.values(input.graph.inputs)) {
    assertTensorLimit(graphNode, limits.input, descriptor, `graph input '${descriptor.name}'`);
  }
  for (const descriptor of Object.values(input.graph.weights)) {
    assertTensorLimit(graphNode, limits.constant, descriptor, `graph constant '${descriptor.name}'`);
  }
  for (const name of input.graph.outputs) {
    assertTensorLimit(graphNode, limits.output, input.graph.tensors[name], `graph output '${name}'`);
  }
  const maxTensorByteLength = limits.maxTensorByteLength;
  const maximumAllowed = typeof maxTensorByteLength === 'bigint'
    ? maxTensorByteLength
    : Number.isSafeInteger(maxTensorByteLength) && (maxTensorByteLength as number) >= 0
      ? BigInt(maxTensorByteLength as number)
      : null;
  if (maximumAllowed === null || maximumTensorBytes > maximumAllowed) {
    physicalFailure(graphNode,
      `maximum tensor bytes ${maximumTensorBytes} exceed or cannot be checked against MLContext.maxTensorByteLength.`);
  }
  // A context may own eight cached exact graphs plus one transactionally built
  // candidate. This is a conservative logical-resource charge; the WebNN API
  // does not expose implementation-private compiler memory or a device limit.
  const copies = input.snapshot.isStatic ? 1n : BigInt(WEBNN_VARIANT_CACHE_ENTRIES + 1);
  const maximumResidentBytes = graphBytes * copies;
  if (maximumResidentBytes > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      'WebNN bounded graph-cache resources exceed exact JavaScript arithmetic.', {
        phase: 'compilation', backend: 'webnn',
      });
  }
  return Object.freeze({
    maximumTensorBytes: Number(maximumTensorBytes),
    maximumResidentBytes: Number(maximumResidentBytes),
    resourceLimitBytes: null,
  });
}

function variantMetadataBytes(request: BackendResolvedExecutionRequest): number {
  let bytes = UTF8_ENCODER.encode(request.signature).byteLength + 128;
  for (const descriptor of Object.values(request.tensors)) {
    bytes += UTF8_ENCODER.encode(descriptor.name).byteLength + 64 + descriptor.shape.length * 8;
  }
  return Number.isSafeInteger(bytes) ? bytes : Number.MAX_SAFE_INTEGER;
}

class WebNNProviderExecutionContext implements BackendProviderExecutionContext {
  readonly backendName = 'webnn';
  readonly #snapshot: Model;
  readonly #contextLostPromise: Promise<unknown> | null;
  #source: WebNNEngine | null;
  #variants = new Map<string, WebNNVariant>();
  #inFlight = new Map<string, Promise<PublishedVariant>>();
  #metadataBytes = 0;
  #hits = 0;
  #misses = 0;
  #evictions = 0;
  #oversizeVariants = 0;
  #coalescedBuilds = 0;
  #graphBuildCount = 0;
  #graphBuildTimeMs = 0;
  #contextLost: { readonly message: string } | null = null;
  #closed = false;

  constructor(snapshot: Model, source: WebNNEngine) {
    this.#snapshot = snapshot;
    this.#source = source;
    const lost = source.context?.lost;
    if (lost && typeof lost.then === 'function') {
      const contextLostPromise = Promise.resolve(lost);
      this.#contextLostPromise = contextLostPromise;
      void contextLostPromise.then((info) => {
        const record = supportRecord(info);
        this.#contextLost = Object.freeze({
          message: typeof record?.message === 'string' && record.message
            ? record.message
            : 'WebNN context was lost.',
        });
        for (const variant of this.#variants.values()) variant.engine.dispose();
        this.#variants.clear();
        this.#metadataBytes = 0;
      }, (error) => {
        this.#contextLost = Object.freeze({
          message: error instanceof Error ? error.message : String(error),
        });
        for (const variant of this.#variants.values()) variant.engine.dispose();
        this.#variants.clear();
        this.#metadataBytes = 0;
      });
    } else {
      this.#contextLostPromise = null;
    }
  }

  #assertContextAvailable(): void {
    if (!this.#contextLost) return;
    throw new VolvoxAIError('DEVICE_LOST', `WebNN context was lost: ${this.#contextLost.message}`, {
      phase: 'execution', backend: this.backendName,
    });
  }

  async #buildVariant(request: BackendResolvedExecutionRequest): Promise<WebNNVariant> {
    this.#assertContextAvailable();
    const source = this.#source;
    if (this.#closed || !source) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WebNN execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    const engine = source.fork();
    const started = performance.now();
    try {
      const bound = createBoundExecutionGraph(this.#snapshot, request.plan);
      await engine.allocateGraph(bound.graph);
      this.#assertContextAvailable();
      const graphBuildTimeMs = performance.now() - started;
      return Object.freeze({
        signature: request.signature,
        engine,
        metadataBytes: variantMetadataBytes(request),
        graphBuildTimeMs,
      });
    } catch (error) {
      engine.dispose();
      throw error;
    }
  }

  async #buildAndPublish(request: BackendResolvedExecutionRequest): Promise<PublishedVariant> {
    const variant = await this.#buildVariant(request);
    if (this.#closed) {
      variant.engine.dispose();
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WebNN context closed during graph build.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    this.#graphBuildCount++;
    this.#graphBuildTimeMs += variant.graphBuildTimeMs;
    this.#variants.set(request.signature, variant);
    this.#metadataBytes += variant.metadataBytes;
    let evictions = 0;
    while (this.#variants.size > WEBNN_VARIANT_CACHE_ENTRIES ||
           this.#metadataBytes > WEBNN_VARIANT_CACHE_METADATA_BYTES) {
      const oldest = this.#variants.keys().next().value;
      if (oldest === undefined) break;
      const removed = this.#variants.get(oldest)!;
      this.#variants.delete(oldest);
      this.#metadataBytes -= removed.metadataBytes;
      removed.engine.dispose();
      evictions++;
    }
    this.#evictions += evictions;
    return Object.freeze({ variant, evictions });
  }

  async #selectVariant(request: BackendResolvedExecutionRequest): Promise<VariantSelection> {
    const cached = this.#variants.get(request.signature);
    if (cached) {
      this.#variants.delete(request.signature);
      this.#variants.set(request.signature, cached);
      this.#hits++;
      return { variant: cached, cacheHit: true, coalesced: false, transient: false, evictions: 0 };
    }
    const pending = this.#inFlight.get(request.signature);
    if (pending) {
      this.#coalescedBuilds++;
      const published = await pending;
      return {
        variant: published.variant,
        cacheHit: true,
        coalesced: true,
        transient: false,
        evictions: published.evictions,
      };
    }
    this.#misses++;
    const metadataBytes = variantMetadataBytes(request);
    if (metadataBytes > WEBNN_VARIANT_CACHE_METADATA_BYTES) {
      const variant = await this.#buildVariant(request);
      this.#graphBuildCount++;
      this.#graphBuildTimeMs += variant.graphBuildTimeMs;
      this.#oversizeVariants++;
      return { variant, cacheHit: false, coalesced: false, transient: true, evictions: 0 };
    }
    const build = this.#buildAndPublish(request);
    this.#inFlight.set(request.signature, build);
    let published: PublishedVariant;
    try {
      published = await build;
    } finally {
      if (this.#inFlight.get(request.signature) === build) this.#inFlight.delete(request.signature);
    }
    return {
      variant: published.variant,
      cacheHit: false,
      coalesced: false,
      transient: false,
      evictions: published.evictions,
    };
  }

  async execute(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    if (this.#closed || !this.#source) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WebNN execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    this.#assertContextAvailable();
    if (request.operation !== 'execute') {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebNN ordinary execute does not implement '${request.operation}'.`, {
          phase: 'execution', backend: this.backendName,
        });
    }
    if (request.adapters.selectors.some((selector) => selector !== null)) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        'Logical dynamic-v1 WebNN models do not contain adapter revisions.', {
          phase: 'execution', backend: this.backendName,
      });
    }
    const hostInputs = requireHostExecutionInputs(request, this.backendName);
    const selected = await this.#selectVariant(request);
    this.#assertContextAvailable();
    try {
      const rawInputs = Object.fromEntries(this.#snapshot.inputNames.map((name) => [
        name, hostInputs[name].data,
      ]));
      let rawOutputs: Record<string, RuntimeTypedArray>;
      try {
        rawOutputs = await selected.variant.engine.execute(rawInputs);
      } catch (error) {
        await Promise.resolve();
        this.#assertContextAvailable();
        throw error;
      }
      const outputs = request.outputDescriptors.map((descriptor) => {
        const data = rawOutputs[descriptor.name];
        if (!data || storageDType(data) !== descriptor.dtype ||
            data.byteLength !== descriptor.sizeBytes) {
          throw new VolvoxAIError('EXECUTION_FAILED',
            `WebNN output '${descriptor.name}' disagrees with its resolved descriptor.`, {
              phase: 'execution', backend: this.backendName,
            });
        }
        return Object.freeze({
          name: descriptor.name,
          shape: descriptor.shape,
          dtype: descriptor.dtype,
          location: 'host' as const,
          ownership: 'transfer' as const,
          data: data as RuntimeTypedArray,
        });
      });
      return Object.freeze({
        outputs: Object.freeze(outputs),
        backendReport: Object.freeze({
          shapeSignature: request.signature,
          specializationCacheHit: selected.cacheHit,
          graphBuildCoalesced: selected.coalesced,
          graphBuildTimeMs: selected.cacheHit && !selected.coalesced
            ? 0
            : selected.variant.graphBuildTimeMs,
          graphBuildCount: this.#graphBuildCount,
          graphBuildTotalTimeMs: this.#graphBuildTimeMs,
          specializationCacheEntries: this.#variants.size,
          specializationCacheMetadataBytes: this.#metadataBytes,
          specializationCacheHits: this.#hits,
          specializationCacheMisses: this.#misses,
          specializationCacheEvictions: this.#evictions,
          specializationCacheEvictionsThisExecution: selected.evictions,
          specializationCacheOversizeVariants: this.#oversizeVariants,
          coalescedGraphBuildCount: this.#coalescedBuilds,
        }),
      });
    } finally {
      if (selected.transient) selected.variant.engine.dispose();
    }
  }

  async decodeSeed(_request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      'WebNN does not support retained dynamic-v1 decode seed/step execution.', {
        phase: 'execution', backend: this.backendName,
      });
  }

  async decodeStep(_request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      'WebNN does not support retained dynamic-v1 decode seed/step execution.', {
        phase: 'execution', backend: this.backendName,
      });
  }

  async decodeReset(): Promise<void> {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      'WebNN does not support retained dynamic-v1 decode state.', {
        phase: 'execution', backend: this.backendName,
      });
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    await Promise.allSettled(this.#inFlight.values());
    this.#inFlight.clear();
    for (const variant of this.#variants.values()) variant.engine.dispose();
    this.#variants.clear();
    this.#metadataBytes = 0;
    this.#source?.dispose();
    this.#source = null;
  }
}

class WebNNProviderCompiledModel implements BackendProviderCompiledModel {
  readonly backendName = 'webnn';
  readonly compilationEvidence: Readonly<BackendProviderCompilationEvidence>;
  readonly #snapshot: Model;
  #source: WebNNEngine | null;
  #closed = false;

  constructor(
    input: BackendLogicalCompileInput,
    source: WebNNEngine,
    deviceIdentity: BackendDeviceIdentity | null,
    resources: WebNNResourceDomainProof,
  ) {
    this.#snapshot = input.snapshot;
    this.#source = source;
    this.compilationEvidence = Object.freeze({
      device: deviceIdentity,
      allocationBytes: 0,
      operatorFallbackUsed: false,
      offendingNode: null,
      shapeDomain: Object.freeze({
        proofProtocol: 'canonical-symbolic-domain-proof/v1' as const,
        resourceProtocol: 'bounded-resource-maxima/v1' as const,
        support: 'full' as const,
        graphFingerprint: input.graphFingerprint,
        proof: input.shapeDomainProof,
        maximumTensorBytes: resources.maximumTensorBytes,
        maximumResidentBytes: resources.maximumResidentBytes,
        resourceLimitBytes: resources.resourceLimitBytes,
      }),
    });
  }

  createContext(options?: BackendProviderContextOptions): BackendProviderExecutionContext {
    if (this.#closed || !this.#source) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Compiled WebNN model is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (options?.decode !== undefined) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        'WebNN context binding does not support retained dynamic-v1 decode.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    return new WebNNProviderExecutionContext(this.#snapshot, this.#source.fork());
  }

  close(): void {
    this.#closed = true;
    this.#source = null;
  }
}

/** Exact-signature WebNN graph cache for bounded dynamic-v1 models. */
export class WebNNBackendProvider implements BackendProvider {
  readonly providerVersion = VOLVOXAI_BACKEND_PROVIDER_VERSION;
  readonly backendName = 'webnn';
  readonly capabilities: Readonly<BackendProviderCapabilities>;
  readonly deviceIdentity: BackendDeviceIdentity | null;
  #source: WebNNEngine | null;
  #closed = false;

  constructor(source: WebNNEngine) {
    if (!(source instanceof WebNNEngine)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'WebNN provider requires a WebNNEngine.', {
        phase: 'initialization', backend: this.backendName,
      });
    }
    this.#source = source;
    this.capabilities = createBackendProviderCapabilities({
      operatorFallback: 'none', outputLocation: 'host', dynamicShapeDomain: 'full',
    });
    // Unlike WebGPUAdapter, the WebNN MLContext currently exposes no portable
    // adapter identity.  Keep compilation evidence stable and avoid probing
    // implementation-specific context fields.
    const accelerated = source.context?.accelerated;
    this.deviceIdentity = Object.freeze({
      backend: 'webnn',
      device: accelerated === true
        ? 'accelerated'
        : accelerated === false ? 'cpu' : 'implementation-selected',
    });
  }

  async compile(
    input: BackendLogicalCompileInput,
    options: BackendProviderCompileOptions,
  ): Promise<BackendProviderCompiledModel> {
    if (this.#closed || !this.#source) {
      throw new VolvoxAIError('HANDLE_DISPOSED', "Backend provider 'webnn' is closed.", {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (!(input?.snapshot instanceof Model) ||
        input.shapeDomainProof !== input.snapshot.shapeDomainProof ||
        input.graphFingerprint !== input.snapshot.definitionFingerprint) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'WebNN provider compile requires the exact immutable logical compile view.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    if (options.operatorFallback === 'forbid' && this.capabilities.operatorFallback !== 'none') {
      throw new VolvoxAIError('OPERATOR_FALLBACK_FORBIDDEN',
        "Backend 'webnn' cannot attest strict operator routing.", {
          phase: 'compilation', backend: this.backendName,
        });
    }
    let limits: WebNNSupportRecord;
    try {
      if (typeof this.#source.context?.opSupportLimits !== 'function') {
        throw new Error('MLContext.opSupportLimits() is unavailable.');
      }
      const candidate = supportRecord(this.#source.context.opSupportLimits());
      if (!candidate) throw new Error('MLContext.opSupportLimits() returned an invalid value.');
      limits = candidate;
    } catch (error) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebNN support limits are unavailable: ${error instanceof Error ? error.message : String(error)}`, {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const resources = checkedWebNNResources(input, limits);
    return new WebNNProviderCompiledModel(input, this.#source, this.deviceIdentity, resources);
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    this.#source?.destroyContext();
    this.#source = null;
  }
}
