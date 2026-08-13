import type {
  GraphNodeSpec,
  RuntimeDType,
  RuntimeTypedArray,
  TensorQuantization,
} from '../types.js';
import {
  canonicalShapeSignature,
  checkedShapeAdd,
  checkedShapeElementCount,
  checkedTensorByteLength,
  ShapeContractError,
} from '../ops/shapeSystem.js';
import {
  getOperatorShapeContract,
  OperatorShapeContractError,
} from '../ops/operatorShapeContracts.js';
import { RuntimeGraph } from './RuntimeGraph.js';
import {
  VOLVOX_LOGICAL_GRAPH_FORMAT,
  type Graph,
  type JsonValue,
  type NodeDescriptor,
  type TensorDescriptor,
} from './Graph.js';
import type {
  ModelPackage,
  WeightRole,
} from './ModelLoader.js';
import { Model } from './Model.js';
import type {
  ResolvedShapePlan,
  ResolvedTensorDescriptor,
} from './ResolvedShapePlan.js';
import {
  canonicalBankResidencySuffix,
  hasCanonicalResolvedShapePlanProvenance,
} from './ResolvedShapePlan.js';
import { Tensor } from './Tensor.js';

export type BoundExecutionGraphErrorCode =
  | 'INVALID_PACKAGE'
  | 'PLAN_FINGERPRINT_MISMATCH'
  | 'PLAN_DESCRIPTOR_MISMATCH'
  | 'PLAN_TOPOLOGY_MISMATCH'
  | 'WEIGHT_PAYLOAD_MISMATCH'
  | 'ACTIVATION_STORAGE_MISMATCH'
  | 'CONSTRUCTION_FAILED';

/** Stable materialization failure raised before a concrete graph is published. */
export class BoundExecutionGraphError extends Error {
  readonly code: BoundExecutionGraphErrorCode;
  readonly path: string;

  constructor(
    code: BoundExecutionGraphErrorCode,
    path: string,
    message: string,
    cause?: unknown,
  ) {
    super(`[BoundExecutionGraph] ${path}: ${message}`,
      cause === undefined ? undefined : { cause });
    this.name = 'BoundExecutionGraphError';
    this.code = code;
    this.path = path;
  }
}

export interface BoundActivationStorageRequest {
  readonly graphFingerprint: string;
  readonly signature: string;
  readonly name: string;
  readonly kind: 'input' | 'value';
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly elementCount: number;
  readonly sizeBytes: number;
  readonly isPublicInput: boolean;
  readonly isPublicOutput: boolean;
  readonly outputIndex: number | null;
}

/**
 * Context capacity providers return an exact logical typed-array view. The
 * backing allocation may be larger, but the returned view may not expose its
 * capacity tail. Returning undefined leaves allocation to the backend.
 */
export type BoundActivationStorageFactory = (
  request: BoundActivationStorageRequest,
) => RuntimeTypedArray | undefined;

export interface BoundActivationLivenessRegion {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly arena: string;
  readonly offsetBytes: number;
  readonly sizeBytes: number;
  /** Public inputs are born at -1; node outputs are born at their node index. */
  readonly birth: number;
  /** Public outputs remain live through the sentinel index `nodes.length`. */
  readonly lastUse: number;
}

/**
 * An opt-in storage-alias proof. It is accepted only when it is immutable and
 * bound to the exact graph fingerprint and concrete shape-plan signature.
 * Ordinary activation factories remain strictly non-overlapping.
 */
export interface BoundActivationLivenessLayout {
  readonly protocol: 'logical-topology-liveness/v1';
  readonly graphFingerprint: string;
  readonly signature: string;
  readonly capacityByArena: Readonly<Record<string, number>>;
  readonly capacityBytes: number;
  readonly logicalBytes: number;
  readonly regions: Readonly<Record<string, BoundActivationLivenessRegion>>;
}

export interface BoundInvariantWeightStorageRequest {
  readonly graphFingerprint: string;
  readonly definitionId: string;
  readonly weightRevisionId: string;
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly elementCount: number;
  readonly sizeBytes: number;
  readonly role: WeightRole;
}

/**
 * A context may reuse a private, immutable weight revision across concrete
 * shape materializations. Returning undefined performs the default defensive
 * snapshot copy. This hook is accepted only for Model sources.
 */
export type BoundInvariantWeightStorageFactory = (
  request: BoundInvariantWeightStorageRequest,
) => RuntimeTypedArray | undefined;

export interface BoundExecutionGraphOptions {
  readonly activationStorageFactory?: BoundActivationStorageFactory;
  readonly activationStorageLayout?: BoundActivationLivenessLayout;
  readonly invariantWeightStorageFactory?: BoundInvariantWeightStorageFactory;
}

/**
 * Context-private materializer for repeated concrete shapes of one exact
 * immutable model revision. Invariant weight storage is validated once per
 * resident-bank selection; every plan and activation view remains exact.
 */
export interface BoundExecutionGraphMaterializer {
  readonly graphFingerprint: string;
  materialize(plan: ResolvedShapePlan): BoundExecutionGraph;
  /** Release context-owned staged weight views during context close. */
  clearInvariantStorage(): void;
}

export interface BoundTensorInspection {
  readonly name: string;
  readonly kind: 'input' | 'weight' | 'value';
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly elementCount: number;
  readonly sizeBytes: number;
  readonly quantization?: TensorQuantization;
}

export interface BoundLogicalByteTotals {
  readonly activations: number;
  readonly weights: number;
  readonly total: number;
}

export interface BoundLogicalRevision {
  readonly definitionFingerprint: string;
  readonly definitionId: string | null;
  readonly topologyRevision: number | null;
  readonly weightRevision: number | null;
  readonly weightRevisionId: string | null;
}

/**
 * Frozen context-local publication. `graph` is intentionally the one mutable
 * field target: it is private to an ExecutionContext and is consumed by the
 * existing concrete kernels. The wrapper and all introspection remain stable.
 */
export interface BoundExecutionGraph {
  readonly graphFingerprint: string;
  readonly signature: string;
  /** Retained when materialized from the exact immutable model revision. */
  readonly snapshot: Model | null;
  readonly revision: BoundLogicalRevision;
  readonly plan: ResolvedShapePlan;
  readonly graph: RuntimeGraph;
  /** Public inputs use canonical unsigned-UTF-8 name order. */
  readonly inputs: readonly BoundTensorInspection[];
  /** Public outputs retain the exact selected graph order. */
  readonly outputs: readonly BoundTensorInspection[];
  readonly logicalBytes: BoundLogicalByteTotals;
}

export type BoundExecutionGraphSource = ModelPackage | Model;

interface BoundWeightSource {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly sizeBytes: number;
  readonly role: WeightRole;
  copyData(): RuntimeTypedArray;
}

interface NormalizedBoundGraphSource {
  readonly graph: Graph;
  readonly weights: Readonly<Record<string, BoundWeightSource>>;
  readonly quantizationByTensor: Readonly<Record<string, TensorQuantization>>;
  readonly quantizationParameterNames: readonly string[];
  readonly snapshot: Model | null;
}

type UnknownRecord = Readonly<Record<string, unknown>>;

const UTF8_ENCODER = new TextEncoder();
const VALIDATED_DEEP_FROZEN_PLAN_VALUES = new WeakSet<object>();
const VALIDATED_QUANTIZATION_FIELDS = new WeakSet<object>();
const VALIDATED_SNAPSHOT_QUANTIZATION_PACKAGES = new WeakSet<Model>();
const EQUAL_QUANTIZATION_PAIRS = new WeakMap<object, WeakSet<object>>();

function fail(
  code: BoundExecutionGraphErrorCode,
  path: string,
  message: string,
  cause?: unknown,
): never {
  throw new BoundExecutionGraphError(code, path, message, cause);
}

function hasOwn(value: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value) &&
    !ArrayBuffer.isView(value) && !(value instanceof ArrayBuffer);
}

function hasFrozenOwnDataFields(value: object): boolean {
  if (!Object.isFrozen(value) || Object.getOwnPropertySymbols(value).length !== 0) return false;
  return Object.getOwnPropertyNames(value).every((name) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, name);
    return descriptor !== undefined && hasOwn(descriptor, 'value');
  });
}

function compareCanonicalNames(left: string, right: string): number {
  if (left === right) return 0;
  const leftBytes = UTF8_ENCODER.encode(left);
  const rightBytes = UTF8_ENCODER.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  return leftBytes.length - rightBytes.length || (left < right ? -1 : 1);
}

function sortedNames(value: object): string[] {
  return Object.getOwnPropertyNames(value).sort(compareCanonicalNames);
}

function sameNames(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((name, index) => name === right[index]);
}

function assertRecord(
  value: unknown,
  code: BoundExecutionGraphErrorCode,
  path: string,
): asserts value is UnknownRecord {
  if (!isRecord(value) || Object.getOwnPropertySymbols(value).length !== 0) {
    fail(code, path, 'must be an object record without symbol-keyed fields.');
  }
}

function assertExactFields(
  value: UnknownRecord,
  expected: readonly string[],
  code: BoundExecutionGraphErrorCode,
  path: string,
): void {
  const actual = sortedNames(value);
  const canonicalExpected = [...expected].sort(compareCanonicalNames);
  if (!sameNames(actual, canonicalExpected)) {
    fail(
      code,
      path,
      `has fields [${actual.join(', ')}], expected exactly [${canonicalExpected.join(', ')}].`,
    );
  }
}

function assertExactNames(
  actual: object,
  expected: object,
  code: BoundExecutionGraphErrorCode,
  path: string,
): void {
  const actualNames = sortedNames(actual);
  const expectedNames = sortedNames(expected);
  if (!sameNames(actualNames, expectedNames)) {
    fail(
      code,
      path,
      `has names [${actualNames.join(', ')}], expected exactly [${expectedNames.join(', ')}].`,
    );
  }
}

function assertBankResidency(graph: Graph, plan: ResolvedShapePlan): void {
  assertRecord(
    plan.bankResidency,
    'PLAN_DESCRIPTOR_MISMATCH',
    'plan.bankResidency',
  );
  for (const name of sortedNames(plan.bankResidency)) {
    const path = `plan.bankResidency.${name}`;
    if (!hasOwn(graph.banks, name) || !hasOwn(graph.weights, name)) {
      fail('PLAN_DESCRIPTOR_MISMATCH', path, 'does not name a declared weight bank.');
    }
    const slots = plan.bankResidency[name];
    if (!Array.isArray(slots) || slots.length === 0) {
      fail('PLAN_DESCRIPTOR_MISMATCH', path, 'must be a non-empty slot-id array.');
    }
    const available = graph.weights[name].shape[0];
    let previous = -1;
    for (let index = 0; index < slots.length; index++) {
      const slot = slots[index];
      if (!Number.isSafeInteger(slot) || slot < 0 || slot >= available) {
        fail(
          'PLAN_DESCRIPTOR_MISMATCH',
          `${path}[${index}]`,
          `must be a slot id in [0, ${available - 1}].`,
        );
      }
      if (slot <= previous) {
        fail(
          'PLAN_DESCRIPTOR_MISMATCH',
          `${path}[${index}]`,
          'must be strictly ascending and unique.',
        );
      }
      previous = slot;
    }
  }
}

function assertDeepFrozenPlan(
  value: unknown,
  path: string,
  seen = new WeakSet<object>(),
): void {
  if (value === null || typeof value !== 'object') return;
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    fail('PLAN_DESCRIPTOR_MISMATCH', path, 'must not contain execution storage.');
  }
  if (VALIDATED_DEEP_FROZEN_PLAN_VALUES.has(value)) return;
  if (seen.has(value)) return;
  seen.add(value);
  if (!Object.isFrozen(value)) {
    fail('PLAN_DESCRIPTOR_MISMATCH', path, 'must be immutable.');
  }
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    fail('PLAN_DESCRIPTOR_MISMATCH', path, 'must not contain symbol-keyed fields.');
  }
  for (const key of Object.getOwnPropertyNames(value)) {
    if (Array.isArray(value) && key === 'length') continue;
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (!hasOwn(descriptor, 'value')) {
      fail('PLAN_DESCRIPTOR_MISMATCH', `${path}.${key}`, 'must be a data field.');
    }
    assertDeepFrozenPlan(descriptor.value, `${path}.${key}`, seen);
  }
  VALIDATED_DEEP_FROZEN_PLAN_VALUES.add(value);
}

function assertConcreteShape(
  shape: unknown,
  dtype: RuntimeDType,
  path: string,
): readonly [number, number] {
  if (!Array.isArray(shape)) {
    fail('PLAN_DESCRIPTOR_MISMATCH', path, 'must be a concrete shape array.');
  }
  try {
    return [
      checkedShapeElementCount(shape as readonly number[], path),
      checkedTensorByteLength(shape as readonly number[], dtype, path),
    ];
  } catch (error) {
    if (error instanceof ShapeContractError) {
      fail('PLAN_DESCRIPTOR_MISMATCH', path, error.message, error);
    }
    throw error;
  }
}

function shapeEquals(left: readonly unknown[], right: readonly unknown[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function assertQuantizationFields(
  value: TensorQuantization,
  path: string,
  code: BoundExecutionGraphErrorCode = 'PLAN_DESCRIPTOR_MISMATCH',
): void {
  if (VALIDATED_QUANTIZATION_FIELDS.has(value)) return;
  assertRecord(value, code, path);
  if (value.scheme === 'per_tensor') {
    assertExactFields(
      value,
      ['scheme', 'scale', 'zero_point'],
      code,
      path,
    );
    if (hasFrozenOwnDataFields(value)) VALIDATED_QUANTIZATION_FIELDS.add(value);
    return;
  }
  if (value.scheme === 'per_axis') {
    assertExactFields(
      value,
      ['scheme', 'axis', 'scales', 'zero_points'],
      code,
      path,
    );
    if (hasFrozenOwnDataFields(value) && hasFrozenOwnDataFields(value.scales) &&
        hasFrozenOwnDataFields(value.zero_points)) {
      VALIDATED_QUANTIZATION_FIELDS.add(value);
    }
    return;
  }
  fail(code, `${path}.scheme`, 'has an unsupported scheme.');
}

function quantizationEquals(
  left: TensorQuantization | null | undefined,
  right: TensorQuantization | null | undefined,
): boolean {
  if (left == null || right == null) return left == null && right == null;
  if (left === right || EQUAL_QUANTIZATION_PAIRS.get(left)?.has(right)) return true;
  if (left.scheme !== right.scheme) return false;
  let equal: boolean;
  if (left.scheme === 'per_tensor' && right.scheme === 'per_tensor') {
    equal = left.scale === right.scale && left.zero_point === right.zero_point;
  } else if (left.scheme !== 'per_axis' || right.scheme !== 'per_axis') {
    return false;
  } else {
    equal = left.axis === right.axis && shapeEquals(left.scales, right.scales) &&
      shapeEquals(left.zero_points, right.zero_points);
  }
  if (equal && hasFrozenOwnDataFields(left) && hasFrozenOwnDataFields(right) &&
      (left.scheme !== 'per_axis' ||
       (hasFrozenOwnDataFields(left.scales) && hasFrozenOwnDataFields(left.zero_points))) &&
      (right.scheme !== 'per_axis' ||
       (hasFrozenOwnDataFields(right.scales) && hasFrozenOwnDataFields(right.zero_points)))) {
    let matches = EQUAL_QUANTIZATION_PAIRS.get(left);
    if (matches === undefined) {
      matches = new WeakSet<object>();
      EQUAL_QUANTIZATION_PAIRS.set(left, matches);
    }
    matches.add(right);
  }
  return equal;
}

function logicalJsonEquals(left: JsonValue, right: JsonValue): boolean {
  if (left === null || right === null || typeof left !== 'object' || typeof right !== 'object') {
    return Object.is(left, right);
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left) && Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => logicalJsonEquals(value, right[index]));
  }
  if (Object.getOwnPropertySymbols(left).length !== 0 ||
      Object.getOwnPropertySymbols(right).length !== 0) return false;
  const leftNames = sortedNames(left);
  const rightNames = sortedNames(right);
  return sameNames(leftNames, rightNames) && leftNames.every((name) =>
    logicalJsonEquals(left[name], right[name]));
}

function cloneLogicalJson(value: JsonValue): JsonValue {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(cloneLogicalJson);
  return Object.fromEntries(sortedNames(value).map((name) => [name, cloneLogicalJson(value[name])])) as
    { [name: string]: JsonValue };
}

function expectedConcreteShape(
  logical: TensorDescriptor,
  symbols: Readonly<Record<string, number>>,
  path: string,
): readonly number[] {
  return logical.shape.map((dimension, axis) => {
    if (typeof dimension === 'number') return dimension;
    if (!hasOwn(symbols, dimension)) {
      fail(
        'PLAN_DESCRIPTOR_MISMATCH',
        `${path}[${axis}]`,
        `has no concrete binding for symbol '${dimension}'.`,
      );
    }
    return symbols[dimension];
  });
}

function expectedDescriptorFields(
  logical: TensorDescriptor,
  quantization: TensorQuantization | undefined,
): string[] {
  const fields = ['name', 'kind', 'dtype', 'shape', 'elementCount', 'sizeBytes'];
  if (quantization !== undefined) fields.push('quantization');
  if (logical.kind === 'value') fields.push('producerNodeId', 'producerPort');
  return fields;
}

function assertTensorDescriptor(
  graph: Graph,
  plan: ResolvedShapePlan,
  name: string,
  expectedQuantization: TensorQuantization | undefined,
): void {
  const path = `plan.tensors.${name}`;
  const logical = graph.tensors[name];
  const concrete = plan.tensors[name];
  assertRecord(concrete, 'PLAN_DESCRIPTOR_MISMATCH', path);
  assertExactFields(
    concrete,
    expectedDescriptorFields(logical, expectedQuantization),
    'PLAN_DESCRIPTOR_MISMATCH',
    path,
  );
  if (concrete.name !== name || concrete.kind !== logical.kind || concrete.dtype !== logical.dtype) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      path,
      'name, kind, or dtype disagrees with the logical tensor descriptor.',
    );
  }
  const declaredShape = expectedConcreteShape(logical, plan.symbols, `${path}.shape`);
  // A partially resident bank stages only its resident slots, so axis 0 is the
  // resident count rather than the bank's full extent.
  const residentSlots = hasOwn(plan.bankResidency, name)
    ? plan.bankResidency[name]
    : undefined;
  const expectedShape = residentSlots === undefined
    ? declaredShape
    : [residentSlots.length, ...declaredShape.slice(1)];
  if (!Array.isArray(concrete.shape) || !shapeEquals(concrete.shape, expectedShape)) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      `${path}.shape`,
      `is [${Array.isArray(concrete.shape) ? concrete.shape.join(',') : ''}], ` +
      `expected [${expectedShape.join(',')}].`,
    );
  }
  const [elementCount, sizeBytes] = assertConcreteShape(concrete.shape, concrete.dtype, `${path}.shape`);
  if (concrete.elementCount !== elementCount || concrete.sizeBytes !== sizeBytes) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      path,
      `has element/byte totals ${concrete.elementCount}/${concrete.sizeBytes}, ` +
      `expected ${elementCount}/${sizeBytes}.`,
    );
  }
  if (logical.kind === 'value' &&
      (concrete.producerNodeId !== logical.producerNodeId ||
       concrete.producerPort !== logical.producerPort)) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      path,
      'producer identity disagrees with the logical topology.',
    );
  }
  if (!quantizationEquals(concrete.quantization, expectedQuantization)) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      `${path}.quantization`,
      'does not match the package-hydrated quantization descriptor.',
    );
  }
  if (concrete.quantization !== undefined) {
    assertQuantizationFields(concrete.quantization, `${path}.quantization`);
  }
}

function assertSymbols(graph: Graph, plan: ResolvedShapePlan): void {
  assertRecord(plan.symbols, 'PLAN_DESCRIPTOR_MISMATCH', 'plan.symbols');
  const used = new Set<string>();
  for (const descriptor of Object.values(graph.tensors)) {
    for (const dimension of descriptor.shape) {
      if (typeof dimension === 'string') used.add(dimension);
    }
  }
  const expectedNames = [...used].sort(compareCanonicalNames);
  const actualNames = sortedNames(plan.symbols);
  if (!sameNames(actualNames, expectedNames)) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      'plan.symbols',
      `binds [${actualNames.join(', ')}], expected exactly [${expectedNames.join(', ')}].`,
    );
  }
  for (const name of expectedNames) {
    const value = plan.symbols[name];
    const constraint = graph.environment.get(name);
    if (!constraint || !Number.isSafeInteger(value) || value <= 0 ||
        value < constraint.min || value > constraint.max ||
        value % (constraint.multiple_of ?? 1) !== 0) {
      fail(
        'PLAN_DESCRIPTOR_MISMATCH',
        `plan.symbols.${name}`,
        `binding ${String(value)} violates its declared bounded-domain constraint.`,
      );
    }
  }
}

function assertPlanTopology(graph: Graph, plan: ResolvedShapePlan): void {
  if (plan.nodes.length !== graph.nodes.length) {
    fail(
      'PLAN_TOPOLOGY_MISMATCH',
      'plan.nodes',
      `contains ${plan.nodes.length} nodes, expected ${graph.nodes.length}.`,
    );
  }
  for (let index = 0; index < graph.nodes.length; index++) {
    const logical = graph.nodes[index];
    const concrete = plan.nodes[index];
    const path = `plan.nodes[${index}]`;
    assertRecord(concrete, 'PLAN_TOPOLOGY_MISMATCH', path);
    assertExactFields(
      concrete,
      ['id', 'opType', 'shapeFunctionId', 'inputs', 'outputs', 'params'],
      'PLAN_TOPOLOGY_MISMATCH',
      path,
    );
    if (concrete.id !== logical.id || concrete.opType !== logical.opType) {
      fail(
        'PLAN_TOPOLOGY_MISMATCH',
        path,
        'node identity or operator disagrees with the logical topology.',
      );
    }
    let shapeFunctionId: string;
    try {
      shapeFunctionId = getOperatorShapeContract(logical.opType).shapeFunctionId;
    } catch (error) {
      if (error instanceof OperatorShapeContractError) {
        fail('PLAN_TOPOLOGY_MISMATCH', `${path}.opType`, error.message, error);
      }
      throw error;
    }
    if (concrete.shapeFunctionId !== shapeFunctionId) {
      fail(
        'PLAN_TOPOLOGY_MISMATCH',
        `${path}.shapeFunctionId`,
        `is '${concrete.shapeFunctionId}', expected '${shapeFunctionId}'.`,
      );
    }
    assertRecord(concrete.inputs, 'PLAN_TOPOLOGY_MISMATCH', `${path}.inputs`);
    assertRecord(concrete.outputs, 'PLAN_TOPOLOGY_MISMATCH', `${path}.outputs`);
    assertExactNames(concrete.inputs, logical.inputs, 'PLAN_TOPOLOGY_MISMATCH', `${path}.inputs`);
    assertExactNames(concrete.outputs, logical.outputs, 'PLAN_TOPOLOGY_MISMATCH', `${path}.outputs`);
    for (const port of sortedNames(logical.inputs)) {
      const expected = plan.tensors[logical.inputs[port]];
      if (concrete.inputs[port] !== expected) {
        fail(
          'PLAN_TOPOLOGY_MISMATCH',
          `${path}.inputs.${port}`,
          `does not reference canonical plan tensor '${logical.inputs[port]}'.`,
        );
      }
    }
    for (const port of sortedNames(logical.outputs)) {
      const expected = plan.tensors[logical.outputs[port].tensor];
      if (concrete.outputs[port] !== expected) {
        fail(
          'PLAN_TOPOLOGY_MISMATCH',
          `${path}.outputs.${port}`,
          `does not reference canonical plan tensor '${logical.outputs[port].tensor}'.`,
        );
      }
    }
    if (!logicalJsonEquals(
      concrete.params as Readonly<Record<string, JsonValue>>,
      logical.params,
    )) {
      fail('PLAN_TOPOLOGY_MISMATCH', `${path}.params`, 'disagrees with logical node params.');
    }
  }
}

function assertPlanOutputs(graph: Graph, plan: ResolvedShapePlan): void {
  if (plan.outputs.length !== graph.outputs.length) {
    fail(
      'PLAN_TOPOLOGY_MISMATCH',
      'plan.outputs',
      `contains ${plan.outputs.length} outputs, expected ${graph.outputs.length}.`,
    );
  }
  for (let index = 0; index < graph.outputs.length; index++) {
    const name = graph.outputs[index];
    if (plan.outputs[index] !== plan.tensors[name]) {
      fail(
        'PLAN_TOPOLOGY_MISMATCH',
        `plan.outputs[${index}]`,
        `does not reference canonical selected tensor '${name}'.`,
      );
    }
  }
}

function addBytes(
  total: number,
  value: number,
  code: BoundExecutionGraphErrorCode,
  path: string,
): number {
  try {
    return checkedShapeAdd(total, value, path);
  } catch (error) {
    if (error instanceof ShapeContractError) fail(code, path, error.message, error);
    throw error;
  }
}

function assertPlanTotals(graph: Graph, plan: ResolvedShapePlan): void {
  let activationBytes = 0;
  let weightBytes = 0;
  for (const name of sortedNames(graph.tensors)) {
    const descriptor = plan.tensors[name];
    if (descriptor.kind === 'weight') {
      weightBytes = addBytes(
        weightBytes,
        descriptor.sizeBytes,
        'PLAN_DESCRIPTOR_MISMATCH',
        `plan weight bytes at '${name}'`,
      );
    } else {
      activationBytes = addBytes(
        activationBytes,
        descriptor.sizeBytes,
        'PLAN_DESCRIPTOR_MISMATCH',
        `plan activation bytes at '${name}'`,
      );
    }
  }
  if (plan.logicalActivationBytes !== activationBytes || plan.weightBytes !== weightBytes) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      'plan byte totals',
      `are ${plan.logicalActivationBytes}/${plan.weightBytes}, ` +
      `expected ${activationBytes}/${weightBytes}.`,
    );
  }
}

function expectedQuantizationTargets(graph: Graph): readonly string[] {
  return graph.quantization === null ? [] : sortedNames(graph.quantization.tensors);
}

function expectedQuantizationParameterNames(graph: Graph): readonly string[] {
  if (graph.quantization === null) return [];
  const names = new Set<string>();
  for (const reference of Object.values(graph.quantization.tensors)) {
    names.add(reference.scale_tensor);
    names.add(reference.zero_point_tensor);
  }
  return [...names].sort(compareCanonicalNames);
}

function normalizeSource(source: BoundExecutionGraphSource): NormalizedBoundGraphSource {
  if (source instanceof Model) {
    const weights = Object.freeze(Object.fromEntries(source.weightDescriptors.map((descriptor) => [
      descriptor.name,
      Object.freeze({
        name: descriptor.name,
        dtype: descriptor.dtype,
        shape: descriptor.shape,
        sizeBytes: descriptor.sizeBytes,
        role: descriptor.role,
        copyData: (): RuntimeTypedArray => source.copyWeightData(descriptor.name),
      }),
    ]))) as Readonly<Record<string, BoundWeightSource>>;
    return Object.freeze({
      graph: source.graph,
      weights,
      quantizationByTensor: source.quantizationByTensor,
      quantizationParameterNames: Object.freeze(expectedQuantizationParameterNames(source.graph)),
      snapshot: source,
    });
  }
  if (!isRecord(source)) fail('INVALID_PACKAGE', 'source', 'must be a loaded package or logical snapshot.');
  return Object.freeze({
    graph: source.graph,
    weights: source.weights,
    quantizationByTensor: source.quantizationByTensor,
    quantizationParameterNames: source.quantizationParameterNames,
    snapshot: null,
  });
}

function assertQuantizationPackage(
  source: NormalizedBoundGraphSource,
  graph: Graph,
): void {
  if (source.snapshot !== null &&
      VALIDATED_SNAPSHOT_QUANTIZATION_PACKAGES.has(source.snapshot)) return;
  assertRecord(
    source.quantizationByTensor,
    'INVALID_PACKAGE',
    'package.quantizationByTensor',
  );
  const targetNames = expectedQuantizationTargets(graph);
  const actualTargetNames = sortedNames(source.quantizationByTensor);
  if (!sameNames(actualTargetNames, targetNames)) {
    fail(
      'INVALID_PACKAGE',
      'package.quantizationByTensor',
      `contains [${actualTargetNames.join(', ')}], expected [${targetNames.join(', ')}].`,
    );
  }
  for (const name of targetNames) {
    const quantization = source.quantizationByTensor[name];
    assertQuantizationFields(
      quantization,
      `package.quantizationByTensor.${name}`,
      'INVALID_PACKAGE',
    );
    const logical = graph.tensors[name];
    try {
      const validationShape = logical.shape.map((dimension) =>
        typeof dimension === 'number'
          ? dimension
          : graph.environment.get(dimension)?.min ?? 1);
      Tensor.normalizeQuantization(
        logical.dtype,
        validationShape,
        quantization,
        `package.quantizationByTensor.${name}`,
      );
    } catch (error) {
      // Per-axis quantization is restricted to constant extents by the logical
      // parser, so any arbitrary legal symbol values above are unobservable.
      fail('INVALID_PACKAGE', `package.quantizationByTensor.${name}`, String(error), error);
    }
  }
  if (!Array.isArray(source.quantizationParameterNames)) {
    fail('INVALID_PACKAGE', 'package.quantizationParameterNames', 'must be an array.');
  }
  const parameterNames = expectedQuantizationParameterNames(graph);
  if (!sameNames(source.quantizationParameterNames, parameterNames)) {
    fail(
      'INVALID_PACKAGE',
      'package.quantizationParameterNames',
      `is [${source.quantizationParameterNames.join(', ')}], ` +
      `expected [${parameterNames.join(', ')}].`,
    );
  }
  if (source.snapshot !== null) {
    VALIDATED_SNAPSHOT_QUANTIZATION_PACKAGES.add(source.snapshot);
  }
}

function runtimeStorageMatches(dtype: RuntimeDType, value: unknown): value is RuntimeTypedArray {
  if (!ArrayBuffer.isView(value) || value instanceof DataView) return false;
  if (dtype === 'float32') return value instanceof Float32Array;
  if (dtype === 'int32') return value instanceof Int32Array;
  if (dtype === 'int8') return value instanceof Int8Array;
  return dtype === 'uint8' && (value instanceof Uint8Array || value instanceof Uint8ClampedArray);
}

function cloneRuntimeStorage(value: RuntimeTypedArray): RuntimeTypedArray {
  if (value instanceof Float32Array) return new Float32Array(value);
  if (value instanceof Int32Array) return new Int32Array(value);
  if (value instanceof Int8Array) return new Int8Array(value);
  if (value instanceof Uint8ClampedArray) return new Uint8ClampedArray(value);
  return new Uint8Array(value);
}

function stageWeight(
  loaded: BoundWeightSource,
  descriptor: ResolvedTensorDescriptor,
  expectedRole: 'model_weight' | 'quantization_parameter',
  path: string,
): RuntimeTypedArray {
  if (!isRecord(loaded)) fail('WEIGHT_PAYLOAD_MISMATCH', path, 'must be a loaded weight object.');
  if (loaded.name !== descriptor.name || loaded.dtype !== descriptor.dtype ||
      !Array.isArray(loaded.shape) || !shapeEquals(loaded.shape, descriptor.shape) ||
      loaded.sizeBytes !== descriptor.sizeBytes || loaded.role !== expectedRole) {
    fail(
      'WEIGHT_PAYLOAD_MISMATCH',
      path,
      'descriptor name, dtype, shape, byte length, or role disagrees with the resolved plan.',
    );
  }
  if (typeof loaded.copyData !== 'function') {
    fail('WEIGHT_PAYLOAD_MISMATCH', `${path}.copyData`, 'must be a function.');
  }
  let payload: unknown;
  try {
    payload = loaded.copyData();
  } catch (error) {
    fail('WEIGHT_PAYLOAD_MISMATCH', `${path}.copyData`, 'failed to copy weight payload.', error);
  }
  if (!runtimeStorageMatches(descriptor.dtype, payload) ||
      payload.byteLength !== descriptor.sizeBytes) {
    fail(
      'WEIGHT_PAYLOAD_MISMATCH',
      `${path}.copyData()`,
      `must return exact ${descriptor.dtype} storage with ${descriptor.sizeBytes} bytes.`,
    );
  }
  // This second copy is the context-private invariant storage. It deliberately
  // does not trust a package implementation to return fresh data from copyData.
  return cloneRuntimeStorage(payload);
}

/**
 * Copy only the resident slots of a bank into contiguous storage. The payload
 * is validated against the bank's full extent, then sliced, so a context pays
 * memory for the families it actually routes to.
 */
function stageBankSlots(
  loaded: BoundWeightSource,
  fullShape: readonly number[],
  descriptor: ResolvedTensorDescriptor,
  residentSlots: readonly number[],
  expectedRole: 'model_weight' | 'quantization_parameter',
  path: string,
): RuntimeTypedArray {
  const slotElements = fullShape.slice(1).reduce((total, extent) => total * extent, 1);
  const slotBytes = descriptor.sizeBytes / residentSlots.length;
  const full = stageWeight(
    loaded,
    Object.freeze({
      ...descriptor,
      shape: Object.freeze([...fullShape]),
      elementCount: fullShape[0] * slotElements,
      sizeBytes: slotBytes * fullShape[0],
    }) as ResolvedTensorDescriptor,
    expectedRole,
    path,
  );
  // A fresh buffer: the source and destination slots overlap in general.
  const resident = new (full.constructor as new (length: number) => RuntimeTypedArray)(
    residentSlots.length * slotElements,
  );
  for (let index = 0; index < residentSlots.length; index++) {
    const start = residentSlots[index] * slotElements;
    resident.set(
      full.subarray(start, start + slotElements) as never,
      index * slotElements,
    );
  }
  return resident;
}

interface StagedWeightStorageRegion {
  readonly name: string;
  readonly start: number;
  readonly end: number;
}

function recordStagedWeightStorage(
  regionsByBuffer: WeakMap<object, StagedWeightStorageRegion[]>,
  name: string,
  storage: RuntimeTypedArray,
  rejectOverlap: boolean,
): void {
  const buffer = storage.buffer as object;
  let regions = regionsByBuffer.get(buffer);
  if (regions === undefined) {
    regions = [];
    regionsByBuffer.set(buffer, regions);
  }
  const start = storage.byteOffset;
  const end = start + storage.byteLength;
  if (rejectOverlap) {
    // The list retains canonical staging order, so the first match preserves
    // the prior error identity even when one view encloses another. Work is
    // linear only in views sharing this exact ArrayBuffer/SharedArrayBuffer;
    // the overwhelmingly common distinct-buffer case remains O(1).
    for (const region of regions) {
      if (start < region.end && region.start < end) {
        fail(
          'WEIGHT_PAYLOAD_MISMATCH',
          `invariant weight storage '${name}'`,
          `overlaps storage for '${region.name}'.`,
        );
      }
    }
  }
  regions.push({ name, start, end });
}

function stageWeights(
  source: NormalizedBoundGraphSource,
  graph: Graph,
  plan: ResolvedShapePlan,
  factory: BoundInvariantWeightStorageFactory | undefined,
): ReadonlyMap<string, RuntimeTypedArray> {
  assertRecord(source.weights, 'INVALID_PACKAGE', 'package.weights');
  assertExactNames(source.weights, graph.weights, 'INVALID_PACKAGE', 'package.weights');
  if (factory !== undefined && source.snapshot === null) {
    fail(
      'INVALID_PACKAGE',
      'invariantWeightStorageFactory',
      'requires an exact Model source revision.',
    );
  }
  if (factory !== undefined && typeof factory !== 'function') {
    fail('WEIGHT_PAYLOAD_MISMATCH', 'invariantWeightStorageFactory', 'must be a function.');
  }
  const quantizationParameters = new Set(expectedQuantizationParameterNames(graph));
  const staged = new Map<string, RuntimeTypedArray>();
  const stagedRegions = new WeakMap<object, StagedWeightStorageRegion[]>();
  for (const name of sortedNames(graph.weights)) {
    const descriptor = plan.tensors[name];
    const role = quantizationParameters.has(name)
      ? 'quantization_parameter'
      : 'model_weight';
    const residentSlots = hasOwn(plan.bankResidency, name)
      ? plan.bankResidency[name]
      : undefined;
    let storage: RuntimeTypedArray | undefined;
    // A partially resident bank holds different bytes per context, so it can
    // never come from the CompiledModel's shared invariant weight storage.
    if (factory !== undefined && residentSlots === undefined) {
      const snapshot = source.snapshot!;
      try {
        storage = factory(Object.freeze({
          graphFingerprint: graph.fingerprint,
          definitionId: snapshot.definitionId,
          weightRevisionId: snapshot.weightRevisionId,
          name,
          dtype: descriptor.dtype,
          shape: Object.freeze([...descriptor.shape]),
          elementCount: descriptor.elementCount,
          sizeBytes: descriptor.sizeBytes,
          role,
        }));
      } catch (error) {
        fail(
          'WEIGHT_PAYLOAD_MISMATCH',
          `invariant weight storage '${name}'`,
          'factory failed.',
          error,
        );
      }
      if (storage !== undefined &&
          (!runtimeStorageMatches(descriptor.dtype, storage) ||
           storage.byteLength !== descriptor.sizeBytes)) {
        fail(
          'WEIGHT_PAYLOAD_MISMATCH',
          `invariant weight storage '${name}'`,
          `must be exact ${descriptor.dtype} storage with ${descriptor.sizeBytes} bytes.`,
        );
      }
      if (storage !== undefined) {
        recordStagedWeightStorage(stagedRegions, name, storage, true);
      }
    }
    const stagedStorage = storage ?? (residentSlots === undefined
      ? stageWeight(
        source.weights[name],
        descriptor,
        role,
        `package.weights.${name}`,
      )
      : stageBankSlots(
        source.weights[name],
        graph.weights[name].shape,
        descriptor,
        residentSlots,
        role,
        `package.weights.${name}`,
      ));
    if (storage === undefined) {
      recordStagedWeightStorage(stagedRegions, name, stagedStorage, false);
    }
    staged.set(name, stagedStorage);
  }
  return staged;
}

function assertPackageAndPlan(
  source: NormalizedBoundGraphSource,
  plan: ResolvedShapePlan,
): Graph {
  assertRecord(source, 'INVALID_PACKAGE', 'package');
  const graph = source.graph;
  /* Model.bindShapes produced this exact immutable plan against this exact
   * snapshot graph and hydrated quantization identity. Rewalking every tensor,
   * node, JSON parameter, and byte total here duplicates the canonical
   * resolver's proof on the hottest dynamic-shape path. Provenance is held in
   * a private WeakMap, so copied/forged plain-data plans cannot enter this
   * branch and continue through the complete validation below. */
  if (source.snapshot !== null && hasCanonicalResolvedShapePlanProvenance(
    plan,
    graph,
    source.quantizationByTensor,
  )) {
    return graph;
  }
  if (!isRecord(graph) || graph.format !== VOLVOX_LOGICAL_GRAPH_FORMAT ||
      typeof graph.fingerprint !== 'string' || graph.fingerprint.length === 0 ||
      !isRecord(graph.inputs) || !isRecord(graph.weights) || !isRecord(graph.tensors) ||
      !isRecord(graph.banks) ||
      !Array.isArray(graph.nodes) || !Array.isArray(graph.outputs)) {
    fail('INVALID_PACKAGE', 'package.graph', 'must be a parsed logical graph.');
  }
  assertRecord(plan, 'PLAN_DESCRIPTOR_MISMATCH', 'plan');
  assertDeepFrozenPlan(plan, 'plan');
  assertExactFields(
    plan,
    [
      'graphFingerprint', 'signature', 'symbols', 'tensors', 'nodes', 'outputs',
      'logicalActivationBytes', 'weightBytes', 'bankResidency',
    ],
    'PLAN_DESCRIPTOR_MISMATCH',
    'plan',
  );
  assertBankResidency(graph, plan);
  if (plan.graphFingerprint !== graph.fingerprint) {
    fail(
      'PLAN_FINGERPRINT_MISMATCH',
      'plan.graphFingerprint',
      'does not match the loaded logical graph fingerprint.',
    );
  }
  assertSymbols(graph, plan);
  assertRecord(plan.tensors, 'PLAN_DESCRIPTOR_MISMATCH', 'plan.tensors');
  assertExactNames(plan.tensors, graph.tensors, 'PLAN_DESCRIPTOR_MISMATCH', 'plan.tensors');
  assertQuantizationPackage(source, graph);
  for (const name of sortedNames(graph.tensors)) {
    assertTensorDescriptor(
      graph,
      plan,
      name,
      hasOwn(source.quantizationByTensor, name)
        ? source.quantizationByTensor[name]
        : undefined,
    );
  }
  const inputNames = sortedNames(graph.inputs);
  const shapeSignature = canonicalShapeSignature(inputNames.map((name) => ({
    name,
    shape: plan.tensors[name].shape,
  })));
  // Residency is part of plan identity: contexts holding different slots must
  // not share a plan-cache entry even at the same input shapes.
  const signature = `${shapeSignature}` +
    canonicalBankResidencySuffix(plan.bankResidency);
  if (plan.signature !== signature) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      'plan.signature',
      `is '${plan.signature}', expected '${signature}'.`,
    );
  }
  assertPlanTopology(graph, plan);
  assertPlanOutputs(graph, plan);
  assertPlanTotals(graph, plan);
  return graph;
}

function activationStorageRequest(
  graph: Graph,
  plan: ResolvedShapePlan,
  descriptor: ResolvedTensorDescriptor,
): BoundActivationStorageRequest {
  const outputIndex = graph.outputs.indexOf(descriptor.name);
  return Object.freeze({
    graphFingerprint: graph.fingerprint,
    signature: plan.signature,
    name: descriptor.name,
    kind: descriptor.kind as 'input' | 'value',
    dtype: descriptor.dtype,
    shape: Object.freeze([...descriptor.shape]),
    elementCount: descriptor.elementCount,
    sizeBytes: descriptor.sizeBytes,
    isPublicInput: descriptor.kind === 'input',
    isPublicOutput: outputIndex !== -1,
    outputIndex: outputIndex === -1 ? null : outputIndex,
  });
}

function assertDeepFrozenActivationLayout(
  value: unknown,
  path: string,
  seen = new WeakSet<object>(),
): void {
  if (value === null || typeof value !== 'object') return;
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    fail('ACTIVATION_STORAGE_MISMATCH', path, 'must not contain execution storage.');
  }
  if (seen.has(value)) return;
  seen.add(value);
  if (!Object.isFrozen(value)) {
    fail('ACTIVATION_STORAGE_MISMATCH', path, 'must be immutable.');
  }
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    fail('ACTIVATION_STORAGE_MISMATCH', path, 'must not contain symbol-keyed fields.');
  }
  for (const key of Object.getOwnPropertyNames(value)) {
    if (Array.isArray(value) && key === 'length') continue;
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (!hasOwn(descriptor, 'value')) {
      fail('ACTIVATION_STORAGE_MISMATCH', `${path}.${key}`, 'must be a data field.');
    }
    assertDeepFrozenActivationLayout(descriptor.value, `${path}.${key}`, seen);
  }
}

function activationTopologyLifetimes(
  graph: Graph,
  plan: ResolvedShapePlan,
): Readonly<Record<string, Readonly<{ birth: number; lastUse: number }>>> {
  const birth = new Map<string, number>();
  const lastUse = new Map<string, number>();
  for (const name of sortedNames(plan.tensors)) {
    const descriptor = plan.tensors[name];
    if (descriptor.kind === 'input') {
      birth.set(name, -1);
      lastUse.set(name, -1);
    }
  }
  for (const [nodeIndex, node] of graph.nodes.entries()) {
    for (const name of Object.values(node.inputs)) {
      const descriptor = plan.tensors[name];
      if (descriptor.kind === 'weight') continue;
      if (!birth.has(name)) {
        fail(
          'ACTIVATION_STORAGE_MISMATCH',
          `activationStorageLayout node '${node.id}'`,
          `consumes '${name}' before its lifetime begins.`,
        );
      }
      lastUse.set(name, Math.max(lastUse.get(name)!, nodeIndex));
    }
    for (const output of Object.values(node.outputs)) {
      const name = output.tensor;
      if (birth.has(name)) {
        fail(
          'ACTIVATION_STORAGE_MISMATCH',
          `activationStorageLayout node '${node.id}'`,
          `produces '${name}' more than once.`,
        );
      }
      birth.set(name, nodeIndex);
      lastUse.set(name, nodeIndex);
    }
  }
  for (const name of graph.outputs) {
    if (plan.tensors[name].kind === 'weight') continue;
    if (!birth.has(name)) {
      fail('ACTIVATION_STORAGE_MISMATCH', 'activationStorageLayout',
        `public output '${name}' has no lifetime.`);
    }
    lastUse.set(name, graph.nodes.length);
  }
  const activationNames = sortedNames(plan.tensors)
    .filter((name) => plan.tensors[name].kind !== 'weight');
  if (activationNames.some((name) => !birth.has(name) || !lastUse.has(name))) {
    fail('ACTIVATION_STORAGE_MISMATCH', 'activationStorageLayout',
      'does not cover the complete logical activation topology.');
  }
  return Object.freeze(Object.fromEntries(activationNames.map((name) => [
    name,
    Object.freeze({ birth: birth.get(name)!, lastUse: lastUse.get(name)! }),
  ])));
}

function validateActivationStorageLayout(
  graph: Graph,
  plan: ResolvedShapePlan,
  layout: BoundActivationLivenessLayout | undefined,
): BoundActivationLivenessLayout | null {
  if (layout === undefined) return null;
  assertRecord(layout, 'ACTIVATION_STORAGE_MISMATCH', 'activationStorageLayout');
  assertDeepFrozenActivationLayout(layout, 'activationStorageLayout');
  assertExactFields(
    layout,
    [
      'protocol', 'graphFingerprint', 'signature', 'capacityByArena',
      'capacityBytes', 'logicalBytes', 'regions',
    ],
    'ACTIVATION_STORAGE_MISMATCH',
    'activationStorageLayout',
  );
  if (layout.protocol !== 'logical-topology-liveness/v1' ||
      layout.graphFingerprint !== graph.fingerprint || layout.signature !== plan.signature) {
    fail(
      'ACTIVATION_STORAGE_MISMATCH',
      'activationStorageLayout',
      'must use the liveness-v1 protocol and match the exact graph fingerprint and signature.',
    );
  }
  assertRecord(
    layout.capacityByArena,
    'ACTIVATION_STORAGE_MISMATCH',
    'activationStorageLayout.capacityByArena',
  );
  assertRecord(layout.regions, 'ACTIVATION_STORAGE_MISMATCH', 'activationStorageLayout.regions');
  const activationNames = sortedNames(plan.tensors)
    .filter((name) => plan.tensors[name].kind !== 'weight');
  const regionNames = sortedNames(layout.regions);
  if (!sameNames(activationNames, regionNames)) {
    fail(
      'ACTIVATION_STORAGE_MISMATCH',
      'activationStorageLayout.regions',
      `has names [${regionNames.join(', ')}], expected [${activationNames.join(', ')}].`,
    );
  }
  const lifetimes = activationTopologyLifetimes(graph, plan);
  let logicalBytes = 0;
  const arenas = new Map<string, BoundActivationLivenessRegion[]>();
  for (const name of activationNames) {
    const region = layout.regions[name];
    assertRecord(
      region,
      'ACTIVATION_STORAGE_MISMATCH',
      `activationStorageLayout.regions.${name}`,
    );
    assertExactFields(
      region,
      ['name', 'dtype', 'arena', 'offsetBytes', 'sizeBytes', 'birth', 'lastUse'],
      'ACTIVATION_STORAGE_MISMATCH',
      `activationStorageLayout.regions.${name}`,
    );
    const descriptor = plan.tensors[name];
    const lifetime = lifetimes[name];
    if (region.name !== name || region.dtype !== descriptor.dtype ||
        region.arena !== descriptor.dtype || region.sizeBytes !== descriptor.sizeBytes ||
        region.birth !== lifetime.birth || region.lastUse !== lifetime.lastUse ||
        !Number.isSafeInteger(region.offsetBytes) || region.offsetBytes < 0) {
      fail(
        'ACTIVATION_STORAGE_MISMATCH',
        `activationStorageLayout.regions.${name}`,
        'does not match the exact tensor descriptor and topology lifetime.',
      );
    }
    logicalBytes = addBytes(
      logicalBytes,
      region.sizeBytes,
      'ACTIVATION_STORAGE_MISMATCH',
      'activationStorageLayout.logicalBytes',
    );
    const values = arenas.get(region.arena) ?? [];
    values.push(region);
    arenas.set(region.arena, values);
  }
  if (logicalBytes !== plan.logicalActivationBytes || layout.logicalBytes !== logicalBytes) {
    fail(
      'ACTIVATION_STORAGE_MISMATCH',
      'activationStorageLayout.logicalBytes',
      `is ${layout.logicalBytes}, expected ${plan.logicalActivationBytes}.`,
    );
  }
  const arenaNames = [...arenas.keys()].sort(compareCanonicalNames);
  if (!sameNames(sortedNames(layout.capacityByArena), arenaNames)) {
    fail(
      'ACTIVATION_STORAGE_MISMATCH',
      'activationStorageLayout.capacityByArena',
      `must contain exactly [${arenaNames.join(', ')}].`,
    );
  }
  let capacityBytes = 0;
  for (const arena of arenaNames) {
    const capacity = layout.capacityByArena[arena];
    if (!Number.isSafeInteger(capacity) || capacity <= 0) {
      fail(
        'ACTIVATION_STORAGE_MISMATCH',
        `activationStorageLayout.capacityByArena.${arena}`,
        'must be a positive safe integer.',
      );
    }
    capacityBytes = addBytes(
      capacityBytes,
      capacity,
      'ACTIVATION_STORAGE_MISMATCH',
      'activationStorageLayout.capacityBytes',
    );
    const regions = arenas.get(arena)!;
    for (const region of regions) {
      if (region.offsetBytes > capacity - region.sizeBytes) {
        fail(
          'ACTIVATION_STORAGE_MISMATCH',
          `activationStorageLayout.regions.${region.name}`,
          `exceeds arena '${arena}' capacity ${capacity}.`,
        );
      }
    }
    const byBirth = [...regions].sort((left, right) =>
      left.birth - right.birth || compareCanonicalNames(left.name, right.name));
    const live: BoundActivationLivenessRegion[] = [];
    for (const region of byBirth) {
      for (let index = live.length - 1; index >= 0; index--) {
        if (live[index].lastUse < region.birth) live.splice(index, 1);
      }
      for (const other of live) {
        const overlap = region.offsetBytes < other.offsetBytes + other.sizeBytes &&
          other.offsetBytes < region.offsetBytes + region.sizeBytes;
        if (overlap) {
          fail(
            'ACTIVATION_STORAGE_MISMATCH',
            `activationStorageLayout.regions.${region.name}`,
            `overlaps simultaneously live activation '${other.name}'.`,
          );
        }
      }
      live.push(region);
    }
  }
  if (layout.capacityBytes !== capacityBytes) {
    fail(
      'ACTIVATION_STORAGE_MISMATCH',
      'activationStorageLayout.capacityBytes',
      `is ${layout.capacityBytes}, expected ${capacityBytes}.`,
    );
  }
  return layout;
}

function stageActivationStorage(
  graph: Graph,
  plan: ResolvedShapePlan,
  factory: BoundActivationStorageFactory | undefined,
  storageLayout: BoundActivationLivenessLayout | undefined,
): ReadonlyMap<string, RuntimeTypedArray> {
  const staged = new Map<string, RuntimeTypedArray>();
  const regions: Array<{
    readonly name: string;
    readonly buffer: ArrayBufferLike;
    readonly start: number;
    readonly end: number;
  }> = [];
  const layout = validateActivationStorageLayout(graph, plan, storageLayout);
  if (factory === undefined) {
    if (layout !== null) {
      fail(
        'ACTIVATION_STORAGE_MISMATCH',
        'activationStorageLayout',
        'requires an activationStorageFactory.',
      );
    }
    return staged;
  }
  if (typeof factory !== 'function') {
    fail('ACTIVATION_STORAGE_MISMATCH', 'activationStorageFactory', 'must be a function.');
  }
  for (const name of sortedNames(plan.tensors)) {
    const descriptor = plan.tensors[name];
    if (descriptor.kind === 'weight') continue;
    const request = activationStorageRequest(graph, plan, descriptor);
    let storage: unknown;
    try {
      storage = factory(request);
    } catch (error) {
      fail(
        'ACTIVATION_STORAGE_MISMATCH',
        `activation storage '${name}'`,
        'factory failed while staging exact logical storage.',
        error,
      );
    }
    if (storage === undefined) {
      if (layout !== null) {
        fail(
          'ACTIVATION_STORAGE_MISMATCH',
          `activation storage '${name}'`,
          'must be supplied when a liveness layout is present.',
        );
      }
      continue;
    }
    if (!runtimeStorageMatches(descriptor.dtype, storage) ||
        storage.byteLength !== descriptor.sizeBytes) {
      fail(
        'ACTIVATION_STORAGE_MISMATCH',
        `activation storage '${name}'`,
        `must be an exact ${descriptor.dtype} typed-array view with ` +
        `${descriptor.sizeBytes} bytes.`,
      );
    }
    const start = storage.byteOffset;
    const end = start + storage.byteLength;
    if (layout === null) {
      const overlap = regions.find((region) => region.buffer === storage.buffer &&
        start < region.end && region.start < end);
      if (overlap !== undefined) {
        fail(
          'ACTIVATION_STORAGE_MISMATCH',
          `activation storage '${name}'`,
          `overlaps supplied storage for '${overlap.name}'; bounded-shape v1 ` +
          'requires non-aliased logical tensor views.',
        );
      }
    }
    regions.push({ name, buffer: storage.buffer, start, end });
    staged.set(name, storage);
  }
  if (layout !== null) {
    const physicalArenas = new Map<string, {
      readonly buffer: ArrayBufferLike;
      readonly base: number;
      readonly capacity: number;
    }>();
    for (const name of sortedNames(layout.regions)) {
      const region = layout.regions[name];
      const storage = staged.get(name)!;
      const base = storage.byteOffset - region.offsetBytes;
      const capacity = layout.capacityByArena[region.arena];
      if (!Number.isSafeInteger(base) || base < 0 || base > storage.buffer.byteLength - capacity) {
        fail(
          'ACTIVATION_STORAGE_MISMATCH',
          `activation storage '${name}'`,
          `does not map to the declared '${region.arena}' arena offset.`,
        );
      }
      const previous = physicalArenas.get(region.arena);
      if (previous === undefined) {
        physicalArenas.set(region.arena, { buffer: storage.buffer, base, capacity });
      } else if (previous.buffer !== storage.buffer || previous.base !== base ||
                 previous.capacity !== capacity) {
        fail(
          'ACTIVATION_STORAGE_MISMATCH',
          `activation storage '${name}'`,
          `does not share the exact declared '${region.arena}' arena.`,
        );
      }
    }
    const arenas = [...physicalArenas.entries()];
    for (let leftIndex = 0; leftIndex < arenas.length; leftIndex++) {
      const [leftName, left] = arenas[leftIndex];
      for (let rightIndex = leftIndex + 1; rightIndex < arenas.length; rightIndex++) {
        const [rightName, right] = arenas[rightIndex];
        if (left.buffer === right.buffer && left.base < right.base + right.capacity &&
            right.base < left.base + left.capacity) {
          fail(
            'ACTIVATION_STORAGE_MISMATCH',
            'activationStorageLayout',
            `physical arenas '${leftName}' and '${rightName}' overlap.`,
          );
        }
      }
    }
  }
  return staged;
}

function tensorOptions(
  descriptor: ResolvedTensorDescriptor,
  storage: RuntimeTypedArray | undefined,
): {
  readonly buffer?: RuntimeTypedArray;
  readonly quantization?: TensorQuantization;
} {
  return {
    ...(storage === undefined ? {} : { buffer: storage }),
    ...(descriptor.quantization === undefined ? {} : { quantization: descriptor.quantization }),
  };
}

function concreteWeightLayout(
  opType: string,
  params: Readonly<Record<string, JsonValue>>,
): 'din' | 'dout' | undefined {
  if (opType !== 'Linear' && opType !== 'MatMul' && opType !== 'Gemm') return undefined;
  if (params.weight_layout === 'dout_din') return 'dout';
  if (params.weight_layout === 'din_dout') return 'din';
  if (params.transB === true) return 'dout';
  if (params.transB === false) return 'din';
  // Canonical Wave-A semantics distinguish Linear's output-major default from
  // MatMul/Gemm's input-major default even when square weights cannot reveal it.
  return opType === 'Linear' ? 'dout' : 'din';
}

/**
 * Expose the resident slot ids of any bank this node reads. Only the ports that
 * index a bank by slot are considered, so an ordinary weight adds nothing.
 */
function bankSlotTable(
  graph: Graph,
  node: NodeDescriptor,
  plan: ResolvedShapePlan,
): { residentSlots?: readonly number[]; residentSlotDomain?: number } {
  const routedPort = node.opType === 'Gather'
    ? (hasOwn(node.inputs, 'input') ? 'input' : 'data')
    : node.opType === 'MoELinear'
      ? (hasOwn(node.inputs, 'expert_weight') ? 'expert_weight' : 'weight')
      : null;
  if (routedPort === null || !hasOwn(node.inputs, routedPort)) return {};
  const routedTensor = node.inputs[routedPort];
  const routedDescriptor = graph.tensors[routedTensor];
  if (routedDescriptor === undefined) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      `nodes.${node.id}.inputs.${routedPort}`,
      `routed tensor '${routedTensor}' has no logical descriptor.`,
    );
  }
  const routedShape = expectedConcreteShape(
    routedDescriptor,
    plan.symbols,
    `graph.tensors.${routedTensor}.shape`,
  );
  const routedSlots = hasOwn(plan.bankResidency, routedTensor)
    ? plan.bankResidency[routedTensor]
    : null;

  if (node.opType === 'MoELinear') {
    /* One routed expert id selects the same row from expert_weight and the
     * optional expert_bias. Runtime nodes carry one slot table, so allowing
     * those tensors to stage different global rows would silently pair a
     * weight from one expert with a bias from another. A bank omitted from the
     * request is fully resident and therefore has the identity slot order. */
    for (const port of ['expert_weight', 'weight', 'expert_bias', 'bias']) {
      if (!hasOwn(node.inputs, port)) continue;
      const tensor = node.inputs[port];
      const descriptor = graph.tensors[tensor];
      if (descriptor === undefined) {
        fail(
          'PLAN_DESCRIPTOR_MISMATCH',
          `nodes.${node.id}.inputs.${port}`,
          `routed expert tensor '${tensor}' has no logical descriptor.`,
        );
      }
      const declaredShape = expectedConcreteShape(
        descriptor,
        plan.symbols,
        `graph.tensors.${tensor}.shape`,
      );
      const slots = hasOwn(plan.bankResidency, tensor)
        ? plan.bankResidency[tensor]
        : null;
      const count = slots === null ? declaredShape[0] : slots.length;
      const routedCount = routedSlots === null ? routedShape[0] : routedSlots.length;
      let aligned = declaredShape[0] === routedShape[0] && count === routedCount;
      for (let index = 0; aligned && index < routedCount; index++) {
        aligned = (routedSlots === null ? index : routedSlots[index]) ===
          (slots === null ? index : slots[index]);
      }
      if (!aligned) {
        fail(
          'PLAN_DESCRIPTOR_MISMATCH',
          `nodes.${node.id}.inputs.${port}`,
          `routed expert tensor '${tensor}' must use the same effective slot ` +
          `residency as '${routedTensor}'.`,
        );
      }
    }
  }

  if (routedSlots === null) return {};
  if (!hasOwn(graph.weights, routedTensor)) {
    fail(
      'PLAN_DESCRIPTOR_MISMATCH',
      `nodes.${node.id}.inputs.${routedPort}`,
      `resident routed tensor '${routedTensor}' is not an invariant weight.`,
    );
  }
  // An explicitly requested identity list is semantically the same as an
  // omitted (fully resident) bank. The staged rows remain in global order, so
  // exposing a partial-residency translation table would only make kernels
  // treat a complete bank as partial.
  const fullyResident = routedSlots.length === routedShape[0] &&
    routedSlots.every((slot, index) => slot === index);
  if (fullyResident) return {};

  return {
    residentSlots: routedSlots,
    // Keep negative indices and missing-slot checks in the model's global
    // bank coordinate system. The staged tensor's axis 0 is only the
    // resident-row count and cannot recover this complete extent.
    residentSlotDomain: routedShape[0],
  };
}

function buildGraph(
  graph: Graph,
  plan: ResolvedShapePlan,
  weights: ReadonlyMap<string, RuntimeTypedArray>,
  activations: ReadonlyMap<string, RuntimeTypedArray>,
  canonicalProvenance: boolean,
): RuntimeGraph {
  const concrete = new RuntimeGraph();
  try {
    const sourceTensorSpecs = [
      ...sortedNames(graph.inputs).map((name) => {
        const descriptor = plan.tensors[name];
        return {
          name,
          shape: descriptor.shape,
          dtype: descriptor.dtype,
          options: {
            ...tensorOptions(descriptor, activations.get(name)),
            role: 'input' as const,
          },
        };
      }),
      ...sortedNames(graph.weights).map((name) => {
        const descriptor = plan.tensors[name];
        return {
          name,
          shape: descriptor.shape,
          dtype: descriptor.dtype,
          options: {
            ...tensorOptions(descriptor, weights.get(name)),
            role: 'weight' as const,
          },
        };
      }),
    ];
    const valueTensorSpecs = graph.nodes.flatMap((logical, index) =>
      sortedNames(logical.outputs).map((port) => {
        const descriptor = plan.nodes[index].outputs[port];
        return {
          name: descriptor.name,
          shape: descriptor.shape,
          dtype: descriptor.dtype,
          options: tensorOptions(descriptor, activations.get(descriptor.name)),
        };
      }));
    /* Batch every tensor before wiring nodes. RuntimeGraph then validates the
     * complete topology once instead of constructing output tensors inside the
     * node loop and validating the same graph again after output selection. */
    const tensorSpecs = [...sourceTensorSpecs, ...valueTensorSpecs];
    const nodeSpecs: GraphNodeSpec<Tensor>[] = [];
    for (let index = 0; index < graph.nodes.length; index++) {
      const logical = graph.nodes[index];
      const resolved = plan.nodes[index];
      const wLayout = concreteWeightLayout(logical.opType, logical.params);
      nodeSpecs.push({
        id: logical.id,
        opType: logical.opType,
        inputs: Object.fromEntries(sortedNames(logical.inputs).map((port) =>
          [port, logical.inputs[port]])),
        outputs: Object.fromEntries(sortedNames(logical.outputs).map((port) =>
          [port, resolved.outputs[port].name])),
        params: cloneLogicalJson(logical.params) as Record<string, JsonValue>,
        ...(wLayout === undefined ? {} : { wLayout }),
        // Kernels reading a partially resident bank need the global slot ids
        // backing the staged rows so routes stay expressed in global terms.
        ...bankSlotTable(graph, logical, plan),
      });
    }
    if (canonicalProvenance) {
      concrete._addResolvedTopologyBatch(tensorSpecs, nodeSpecs, graph.outputs);
    } else {
      concrete._addTensorsBatch(tensorSpecs);
      concrete._addNodesBatch(nodeSpecs);
      concrete.setOutputs(graph.outputs);
    }
    return concrete;
  } catch (error) {
    fail(
      'CONSTRUCTION_FAILED',
      'concrete graph',
      'failed after complete plan, payload, and storage preflight.',
      error,
    );
  }
}

function boundWeightSelectionKey(plan: ResolvedShapePlan): string {
  /* Tensor names are arbitrary graph identifiers and may contain every
   * delimiter used by a hand-built string key. A canonical JSON tuple list is
   * unambiguous for both names and integer slot sequences, so two different
   * bank selections can never reuse the same staged payload accidentally. */
  return canonicalBankResidencySuffix(plan.bankResidency);
}

function materializeBoundExecutionGraph(
  normalized: NormalizedBoundGraphSource,
  plan: ResolvedShapePlan,
  options: BoundExecutionGraphOptions,
  weights: ReadonlyMap<string, RuntimeTypedArray> | undefined,
): BoundExecutionGraph {
  const canonicalProvenance = normalized.snapshot !== null &&
    hasCanonicalResolvedShapePlanProvenance(
      plan,
      normalized.graph,
      normalized.quantizationByTensor,
    );
  const graph = assertPackageAndPlan(normalized, plan);
  const stagedWeights = weights ?? stageWeights(
    normalized,
    graph,
    plan,
    options.invariantWeightStorageFactory,
  );
  const activations = stageActivationStorage(
    graph,
    plan,
    options.activationStorageFactory,
    options.activationStorageLayout,
  );
  const concrete = buildGraph(
    graph,
    plan,
    stagedWeights,
    activations,
    canonicalProvenance,
  );
  const inputs = Object.freeze(sortedNames(graph.inputs).map((name) =>
    inspectTensor(plan.tensors[name])));
  const outputs = Object.freeze(plan.outputs.map(inspectTensor));
  const total = addBytes(
    plan.logicalActivationBytes,
    plan.weightBytes,
    'PLAN_DESCRIPTOR_MISMATCH',
    'logical byte total',
  );
  return Object.freeze({
    graphFingerprint: graph.fingerprint,
    signature: plan.signature,
    snapshot: normalized.snapshot,
    revision: Object.freeze(normalized.snapshot === null
      ? {
        definitionFingerprint: graph.fingerprint,
        definitionId: null,
        topologyRevision: null,
        weightRevision: null,
        weightRevisionId: null,
      }
      : {
        definitionFingerprint: normalized.snapshot.definitionFingerprint,
        definitionId: normalized.snapshot.definitionId,
        topologyRevision: normalized.snapshot.topologyRevision,
        weightRevision: normalized.snapshot.weightRevision,
        weightRevisionId: normalized.snapshot.weightRevisionId,
      }),
    plan,
    // This is the mutable context-private kernel graph.
    graph: concrete,
    inputs,
    outputs,
    logicalBytes: Object.freeze({
      activations: plan.logicalActivationBytes,
      weights: plan.weightBytes,
      total,
    }),
  });
}

/**
 * Prepare one exact Model revision for repeated context-local materialization.
 * This is intentionally not available for mutable ModelPackage sources.
 */
export function createBoundExecutionGraphMaterializer(
  source: Model,
  options: BoundExecutionGraphOptions = {},
): BoundExecutionGraphMaterializer {
  if (!(source instanceof Model)) {
    fail('INVALID_PACKAGE', 'source', 'must be an exact Model revision.');
  }
  const normalized = normalizeSource(source);
  let weightSelectionKey: string | null = null;
  let stagedWeights: ReadonlyMap<string, RuntimeTypedArray> | undefined;
  return Object.freeze({
    graphFingerprint: source.definitionFingerprint,
    materialize(plan: ResolvedShapePlan): BoundExecutionGraph {
      /* Validate before even deriving the storage-selection key. A forged
       * plain-data lookalike must never make malformed bank metadata observable
       * through an incidental TypeError or reach the invariant storage cache. */
      const graph = assertPackageAndPlan(normalized, plan);
      const selectionKey = boundWeightSelectionKey(plan);
      if (stagedWeights === undefined || selectionKey !== weightSelectionKey) {
        stagedWeights = stageWeights(
          normalized,
          graph,
          plan,
          options.invariantWeightStorageFactory,
        );
        weightSelectionKey = selectionKey;
      }
      return materializeBoundExecutionGraph(normalized, plan, options, stagedWeights);
    },
    clearInvariantStorage(): void {
      stagedWeights = undefined;
      weightSelectionKey = null;
    },
  });
}

function cloneQuantizationForInspection(
  quantization: TensorQuantization | undefined,
): TensorQuantization | undefined {
  if (quantization === undefined) return undefined;
  if (quantization.scheme === 'per_tensor') {
    return Object.freeze({
      scheme: 'per_tensor',
      scale: quantization.scale,
      zero_point: quantization.zero_point,
    });
  }
  return Object.freeze({
    scheme: 'per_axis',
    axis: quantization.axis,
    scales: Object.freeze([...quantization.scales]),
    zero_points: Object.freeze([...quantization.zero_points]),
  });
}

function inspectTensor(descriptor: ResolvedTensorDescriptor): BoundTensorInspection {
  const quantization = cloneQuantizationForInspection(descriptor.quantization);
  return Object.freeze({
    name: descriptor.name,
    kind: descriptor.kind,
    dtype: descriptor.dtype,
    shape: Object.freeze([...descriptor.shape]),
    elementCount: descriptor.elementCount,
    sizeBytes: descriptor.sizeBytes,
    ...(quantization === undefined ? {} : { quantization }),
  });
}

/**
 * Materialize one context-private concrete graph for an already resolved
 * signature. Every mismatch is rejected before the activation factory runs;
 * every supplied activation view is staged and validated before RuntimeGraph/Tensor
 * construction begins.
 */
export function createBoundExecutionGraph(
  source: Model,
  plan: ResolvedShapePlan,
  options?: BoundExecutionGraphOptions,
): BoundExecutionGraph;
export function createBoundExecutionGraph(
  source: ModelPackage,
  plan: ResolvedShapePlan,
  options?: BoundExecutionGraphOptions,
): BoundExecutionGraph;
export function createBoundExecutionGraph(
  source: BoundExecutionGraphSource,
  plan: ResolvedShapePlan,
  options: BoundExecutionGraphOptions = {},
): BoundExecutionGraph {
  const normalized = normalizeSource(source);
  return materializeBoundExecutionGraph(normalized, plan, options, undefined);
}
