import {
  VOLVOX_AFFINE_QUANTIZATION_FORMAT,
  VOLVOX_LOGICAL_GRAPH_FORMAT,
  parseGraphDocument,
} from './Graph.js';
import type {
  AffineQuantizationReference,
  Graph,
  JsonValue,
  WeightDescriptorInput,
} from './Graph.js';
import type { ShapeDimensionSpec } from '../ops/shapeSystem.js';
import type { RuntimeDType } from '../types.js';

export interface ModelDimensionSpec {
  readonly min: number;
  readonly max: number;
  readonly multiple_of?: number;
}

export interface ModelTensorSpec {
  readonly dtype: RuntimeDType;
  readonly shape: readonly ShapeDimensionSpec[];
}

export interface ModelNodeOutputSpec extends ModelTensorSpec {
  readonly tensor: string;
}

export interface ModelNodeSpec {
  /** Omit only when the builder should derive a deterministic topology-local ID. */
  readonly id?: string;
  readonly opType: string;
  readonly inputs: Readonly<Record<string, string>>;
  readonly outputs: Readonly<Record<string, ModelNodeOutputSpec>>;
  readonly params?: Readonly<Record<string, JsonValue>>;
}

export interface ModelQuantizationTable {
  readonly format: typeof VOLVOX_AFFINE_QUANTIZATION_FORMAT;
  readonly tensors: Readonly<Record<string, AffineQuantizationReference>>;
}

/** Canonical decoded JSON document returned by {@link ModelBuilder}. */
export interface ModelGraphDocument {
  readonly format: typeof VOLVOX_LOGICAL_GRAPH_FORMAT;
  readonly dimensions: Readonly<Record<string, Required<ModelDimensionSpec>>>;
  readonly inputs: Readonly<Record<string, ModelTensorSpec>>;
  readonly nodes: readonly (ModelNodeSpec & { readonly id: string })[];
  readonly outputs: readonly string[];
  readonly quantization?: ModelQuantizationTable;
}

/**
 * Programmatic source for one initially valid logical model.
 *
 * A builder deliberately has no invalid empty state. Start with at least one
 * selected tensor, then use a topology transaction for edits that temporarily
 * break references while they are being assembled.
 */
export interface ModelBuilderDefinition {
  readonly dimensions: Readonly<Record<string, ModelDimensionSpec>>;
  readonly inputs: Readonly<Record<string, ModelTensorSpec>>;
  readonly weights?: readonly WeightDescriptorInput[];
  readonly nodes?: readonly ModelNodeSpec[];
  readonly outputs: readonly string[];
  readonly quantization?: ModelQuantizationTable | null;
}

interface MutableModelGraphDocument {
  format: typeof VOLVOX_LOGICAL_GRAPH_FORMAT;
  dimensions: Record<string, ModelDimensionSpec>;
  inputs: Record<string, ModelTensorSpec>;
  nodes: Array<ModelNodeSpec & { id: string }>;
  outputs: string[];
  quantization?: ModelQuantizationTable;
}

interface BuilderDraft {
  document: MutableModelGraphDocument;
  weights: WeightDescriptorInput[];
}

interface BuilderState {
  readonly graph: Graph;
  readonly document: ModelGraphDocument;
  readonly weights: readonly WeightDescriptorInput[];
}

export class ModelBuilderError extends Error {
  constructor(message: string) {
    super(`[ModelBuilder] ${message}`);
    this.name = 'ModelBuilderError';
  }
}

function defineRecordValue<T>(record: Record<string, T>, name: string, value: T): void {
  Object.defineProperty(record, name, {
    configurable: true,
    enumerable: true,
    value,
    writable: true,
  });
}

function cloneRecord<T>(
  source: Readonly<Record<string, T>>,
  cloneValue: (value: T) => T,
): Record<string, T> {
  const result = Object.create(null) as Record<string, T>;
  for (const name of Object.keys(source)) defineRecordValue(result, name, cloneValue(source[name]));
  return result;
}

function cloneLogicalJson(value: JsonValue): JsonValue {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(cloneLogicalJson);
  return cloneRecord(value as Readonly<Record<string, JsonValue>>, cloneLogicalJson);
}

function cloneShape(shape: readonly ShapeDimensionSpec[]): ShapeDimensionSpec[] {
  return [...shape];
}

function cloneDimensionSpec(value: ModelDimensionSpec): ModelDimensionSpec {
  return {
    min: value.min,
    max: value.max,
    ...(value.multiple_of === undefined ? {} : { multiple_of: value.multiple_of }),
  };
}

function cloneTensorSpec(value: ModelTensorSpec): ModelTensorSpec {
  return { dtype: value.dtype, shape: cloneShape(value.shape) };
}

function cloneNodeOutputSpec(value: ModelNodeOutputSpec): ModelNodeOutputSpec {
  return { tensor: value.tensor, dtype: value.dtype, shape: cloneShape(value.shape) };
}

function cloneQuantizationReference(
  value: AffineQuantizationReference,
): AffineQuantizationReference {
  return value.scheme === 'per_axis'
    ? {
        scheme: 'per_axis',
        axis: value.axis,
        scale_tensor: value.scale_tensor,
        zero_point_tensor: value.zero_point_tensor,
      }
    : {
        scheme: 'per_tensor',
        scale_tensor: value.scale_tensor,
        zero_point_tensor: value.zero_point_tensor,
      };
}

function cloneQuantizationTable(
  table: ModelQuantizationTable,
): ModelQuantizationTable {
  return {
    format: table.format,
    tensors: cloneRecord(table.tensors, cloneQuantizationReference),
  };
}

function cloneNode(
  node: ModelNodeSpec & { readonly id: string },
): ModelNodeSpec & { id: string } {
  return {
    id: node.id,
    opType: node.opType,
    inputs: cloneRecord(node.inputs, (value) => value),
    outputs: cloneRecord(node.outputs, cloneNodeOutputSpec),
    params: node.params === undefined
      ? Object.create(null) as Record<string, JsonValue>
      : cloneRecord(node.params, cloneLogicalJson),
  };
}

function cloneWeight(value: WeightDescriptorInput): WeightDescriptorInput {
  return { name: value.name, dtype: value.dtype, shape: [...value.shape] };
}

function deepFreeze<T>(value: T): T {
  if (value === null || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const name of Object.getOwnPropertyNames(value)) {
    deepFreeze((value as Record<string, unknown>)[name]);
  }
  return Object.freeze(value);
}

function canonicalDocumentFromGraph(graph: Graph): ModelGraphDocument {
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
    inputs: Object.fromEntries(Object.entries(node.inputs)),
    outputs: Object.fromEntries(Object.entries(node.outputs).map(([port, value]) => [
      port,
      { tensor: value.tensor, dtype: value.dtype, shape: [...value.shape] },
    ])),
    params: cloneLogicalJson(node.params) as Readonly<Record<string, JsonValue>>,
  }));
  const quantization = graph.quantization === null
    ? {}
    : {
        quantization: {
          format: graph.quantization.format,
          tensors: Object.fromEntries(Object.entries(graph.quantization.tensors).map(
            ([name, reference]) => [name, cloneQuantizationReference(reference)],
          )),
        },
      };
  return deepFreeze({
    format: VOLVOX_LOGICAL_GRAPH_FORMAT,
    dimensions,
    inputs,
    nodes,
    outputs: [...graph.outputs],
    ...quantization,
  });
}

function canonicalWeightsFromGraph(graph: Graph): readonly WeightDescriptorInput[] {
  return deepFreeze(Object.values(graph.weights).map((weight) => ({
    name: weight.name,
    dtype: weight.dtype,
    shape: [...weight.shape],
  })));
}

function validateState(
  document: unknown,
  weights: readonly WeightDescriptorInput[],
): BuilderState {
  const graph = parseGraphDocument(document, weights);
  return Object.freeze({
    graph,
    document: canonicalDocumentFromGraph(graph),
    weights: canonicalWeightsFromGraph(graph),
  });
}

function mutableDraftFromState(state: BuilderState): BuilderDraft {
  const document = state.document;
  return {
    document: {
      format: document.format,
      dimensions: cloneRecord(document.dimensions, cloneDimensionSpec),
      inputs: cloneRecord(document.inputs, cloneTensorSpec),
      nodes: document.nodes.map(cloneNode),
      outputs: [...document.outputs],
      ...(document.quantization === undefined
        ? {}
        : { quantization: cloneQuantizationTable(document.quantization) }),
    },
    weights: state.weights.map(cloneWeight),
  };
}

function nextGeneratedNodeId(nodes: readonly { readonly id?: unknown }[]): string {
  const occupied = new Set(nodes
    .map((node) => node.id)
    .filter((id): id is string => typeof id === 'string'));
  for (let ordinal = 0; ; ordinal++) {
    const candidate = `node_${ordinal}`;
    if (!occupied.has(candidate)) return candidate;
  }
}

function assignInitialNodeIds(
  nodes: readonly ModelNodeSpec[],
): Array<ModelNodeSpec & { id: string }> {
  const explicitIds = new Set(nodes
    .map((node) => node.id)
    .filter((id): id is string => typeof id === 'string'));
  let ordinal = 0;
  return nodes.map((node) => {
    let id = node.id;
    if (id === undefined) {
      do id = `node_${ordinal++}`; while (explicitIds.has(id));
      explicitIds.add(id);
    }
    return {
      ...node,
      id,
      params: node.params ?? {},
    } as ModelNodeSpec & { id: string };
  });
}

function isThenable(value: unknown): value is PromiseLike<unknown> {
  return value !== null && (typeof value === 'object' || typeof value === 'function') &&
    typeof (value as { then?: unknown }).then === 'function';
}

const TRANSACTION_FACADE_PROPERTIES = new Set<PropertyKey>([
  'snapshot',
  'documentSnapshot',
  'weightDescriptorsSnapshot',
  'fingerprint',
  'topologyTransaction',
  'setDimension',
  'removeDimension',
  'setInput',
  'removeInput',
  'setWeight',
  'removeWeight',
  'addNode',
  'replaceNode',
  'removeNode',
  'selectOutputs',
  'setQuantization',
]);

/**
 * Transactional authoring facade for the redesigned logical graph only.
 *
 * The builder imports no concrete Graph, Tensor, provider, or backend type.
 * Every committed edit is reparsed through `parseGraphDocument`, and a
 * failed edit cannot replace any previously published immutable snapshot.
 */
export class ModelBuilder {
  private state: BuilderState;
  private activeDraft: BuilderDraft | null = null;
  private activeTransactionToken: object | null = null;

  constructor(definition: ModelBuilderDefinition) {
    const document = {
      format: VOLVOX_LOGICAL_GRAPH_FORMAT,
      dimensions: definition.dimensions,
      inputs: definition.inputs,
      nodes: assignInitialNodeIds(definition.nodes ?? []),
      outputs: definition.outputs,
      ...(definition.quantization == null ? {} : { quantization: definition.quantization }),
    };
    this.state = validateState(document, definition.weights ?? []);
  }

  /** Strictly adopt an already-decoded new-v1 document and fixed weight metadata. */
  static fromDocument(
    document: unknown,
    weights: readonly WeightDescriptorInput[] = [],
  ): ModelBuilder {
    const state = validateState(document, weights);
    const builder = Object.create(ModelBuilder.prototype) as ModelBuilder;
    builder.state = state;
    builder.activeDraft = null;
    builder.activeTransactionToken = null;
    return builder;
  }

  /** Return the current immutable, allocation-free logical model snapshot. */
  snapshot(): Graph {
    return this.state.graph;
  }

  /** Return the current immutable canonical decoded `graph.json` snapshot. */
  documentSnapshot(): ModelGraphDocument {
    return this.state.document;
  }

  /** Fixed descriptor metadata only; payloads and concrete Tensor objects are absent. */
  weightDescriptorsSnapshot(): readonly WeightDescriptorInput[] {
    return this.state.weights;
  }

  get fingerprint(): string {
    return this.state.graph.fingerprint;
  }

  /**
   * Atomically validate and publish a group of topology edits.
   *
   * Transactions are intentionally synchronous and non-nested. During the
   * callback, public snapshot methods continue to expose the prior committed
   * state. The candidate becomes visible only after the logical parser accepts
   * the complete graph.
   */
  topologyTransaction<T>(callback: (builder: this) => T): T {
    if (typeof callback !== 'function') {
      throw new ModelBuilderError('topologyTransaction expects a callback.');
    }
    if (this.activeDraft !== null) {
      throw new ModelBuilderError('nested topology transactions are not supported.');
    }

    const draft = mutableDraftFromState(this.state);
    const token = Object.freeze({});
    this.activeDraft = draft;
    this.activeTransactionToken = token;
    const transactionBuilder = this.createTransactionFacade(token, draft);
    try {
      const result = callback(transactionBuilder);
      const returnsFacade = Object.is(result, transactionBuilder);
      if (!returnsFacade && isThenable(result)) {
        // The public transaction fails synchronously, but the callback already
        // created a promise. Observe its eventual rejection so a blocked late
        // edit cannot become an unhandled rejection.
        void Promise.resolve(result).catch(() => undefined);
        throw new ModelBuilderError('topologyTransaction callback must be synchronous.');
      }
      const nextState = validateState(draft.document, draft.weights);
      this.state = nextState;
      // Fluent editor methods return the guarded facade inside the callback.
      // Do not leak that now-inactive object to the caller after commit.
      return (returnsFacade ? this : result) as T;
    } finally {
      this.activeDraft = null;
      this.activeTransactionToken = null;
    }
  }

  setDimension(name: string, constraint: ModelDimensionSpec): this {
    return this.edit((draft) => {
      defineRecordValue(draft.document.dimensions, name, constraint);
      return this;
    });
  }

  removeDimension(name: string): boolean {
    return this.edit((draft) => delete draft.document.dimensions[name]);
  }

  setInput(name: string, descriptor: ModelTensorSpec): this {
    return this.edit((draft) => {
      defineRecordValue(draft.document.inputs, name, descriptor);
      return this;
    });
  }

  removeInput(name: string): boolean {
    return this.edit((draft) => delete draft.document.inputs[name]);
  }

  setWeight(descriptor: WeightDescriptorInput): this {
    return this.edit((draft) => {
      const index = draft.weights.findIndex((weight) => weight.name === descriptor.name);
      if (index < 0) draft.weights.push(descriptor);
      else draft.weights[index] = descriptor;
      return this;
    });
  }

  removeWeight(name: string): boolean {
    return this.edit((draft) => {
      const index = draft.weights.findIndex((weight) => weight.name === name);
      if (index < 0) return false;
      draft.weights.splice(index, 1);
      return true;
    });
  }

  /** Add one node using only the unified output-descriptor map. */
  addNode(spec: ModelNodeSpec): string {
    return this.edit((draft) => {
      const id = spec.id === undefined ? nextGeneratedNodeId(draft.document.nodes) : spec.id;
      draft.document.nodes.push({ ...spec, id, params: spec.params ?? {} });
      return id;
    });
  }

  /** Replace a node; an omitted replacement ID preserves the selected node ID. */
  replaceNode(id: string, spec: ModelNodeSpec): this {
    return this.edit((draft) => {
      const index = draft.document.nodes.findIndex((node) => node.id === id);
      if (index < 0) throw new ModelBuilderError(`unknown node '${id}'.`);
      draft.document.nodes[index] = {
        ...spec,
        id: spec.id === undefined ? id : spec.id,
        params: spec.params ?? {},
      };
      return this;
    });
  }

  removeNode(id: string): boolean {
    return this.edit((draft) => {
      const index = draft.document.nodes.findIndex((node) => node.id === id);
      if (index < 0) return false;
      draft.document.nodes.splice(index, 1);
      return true;
    });
  }

  selectOutputs(outputs: readonly string[]): this {
    return this.edit((draft) => {
      draft.document.outputs = [...outputs];
      return this;
    });
  }

  setQuantization(table: ModelQuantizationTable | null): this {
    return this.edit((draft) => {
      if (table === null) delete draft.document.quantization;
      else draft.document.quantization = table;
      return this;
    });
  }

  private edit<T>(operation: (draft: BuilderDraft) => T): T {
    if (this.activeDraft !== null) return operation(this.activeDraft);
    let result!: T;
    this.topologyTransaction(() => {
      result = operation(this.activeDraft!);
    });
    return result;
  }

  private createTransactionFacade(token: object, draft: BuilderDraft): this {
    const target = this;
    let facade!: this;
    const assertActive = (): void => {
      if (target.activeTransactionToken !== token || target.activeDraft !== draft) {
        throw new ModelBuilderError(
          'transaction editor is no longer active; late asynchronous edits are forbidden.',
        );
      }
    };
    facade = new Proxy(target, {
      get(current, property) {
        assertActive();
        if (!TRANSACTION_FACADE_PROPERTIES.has(property)) {
          throw new ModelBuilderError(
            `transaction editor does not expose '${String(property)}'.`,
          );
        }
        const value = Reflect.get(current, property, current) as unknown;
        if (typeof value !== 'function') return value;
        return (...args: unknown[]) => {
          assertActive();
          const result = Reflect.apply(value, current, args) as unknown;
          return result === current ? facade : result;
        };
      },
      set() {
        assertActive();
        throw new ModelBuilderError('transaction editor properties are read-only.');
      },
      defineProperty() {
        assertActive();
        throw new ModelBuilderError('transaction editor properties are read-only.');
      },
      deleteProperty() {
        assertActive();
        throw new ModelBuilderError('transaction editor properties are read-only.');
      },
    }) as this;
    return facade;
  }
}
