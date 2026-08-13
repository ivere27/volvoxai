import { Tensor } from './Tensor.js';
import type { SafetensorsFile } from './Safetensors.js';
import type {
  AddTensorOptions,
  GraphInspection,
  GraphInspectionOptions,
  GraphNode,
  GraphNodePatch,
  GraphNodeSpec,
  GraphValidationReport,
  NodeOutputSpec,
  NodeParameters,
  RuntimeDType,
  TensorPatch,
  TensorReference,
  TensorStorage,
} from '../types.js';

type ConcreteNode = GraphNode<Tensor>;
type ConcreteNodeSpec = GraphNodeSpec<Tensor>;
type ConcreteNodePatch = GraphNodePatch<Tensor>;
type ConcreteTensorReference = TensorReference<Tensor>;

interface ConcreteTensorBatchSpec {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: RuntimeDType;
  readonly options?: AddTensorOptions;
}

const REPLACE_EXISTING_OUTPUT: unique symbol = Symbol('replaceExistingOutput');
type PreparedOutputDescriptor = Exclude<
  NodeOutputSpec<Tensor>,
  readonly number[] | ConcreteTensorReference
> & { [REPLACE_EXISTING_OUTPUT]?: boolean };

interface TopologySnapshot {
  nodes: ConcreteNode[];
  tensors: Map<string, Tensor>;
  outputNames: string[];
  autoOutputNames: string[];
  outputsExplicit: boolean;
  topologyRevision: number;
  nextTopologyRevision: number;
  nextNodeId: number;
  lastTopologyMutation: string | null;
  weightRevision: number;
  nextWeightRevision: number;
  nodeDescriptors: Map<ConcreteNode, PropertyDescriptorMap>;
  tensorDescriptors: Map<Tensor, PropertyDescriptorMap>;
}

interface RemoveTensorOptions {
  cascade?: boolean;
}

interface RemoveNodeOptions {
  cascade?: boolean;
  removeOutputs?: boolean;
  preserveOutputsAsInputs?: boolean;
  rewire?: Record<string, ConcreteTensorReference>;
}

interface TensorUpdateOptions {
  mode?: 'assign' | 'add' | string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

export function isValidGraphName(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0 &&
    value !== "__metadata__" &&
    !Object.prototype.hasOwnProperty.call(Object.prototype, value);
}

function sameNames<T>(a: readonly T[], b: readonly T[]): boolean {
  return a.length === b.length && a.every((name, index) => name === b[index]);
}

function validateShape(shape: unknown, label: string): number[] {
  if (!Array.isArray(shape)) throw new Error(`${label} shape must be an array.`);
  for (const [index, dim] of shape.entries()) {
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new Error(`${label} shape dimension ${index} must be a positive integer.`);
    }
  }
  return [...shape];
}

function nodeLabel(node: Partial<ConcreteNode> | null | undefined, index: number): string {
  return `node '${String(node?.id ?? index)}' at index ${index}`;
}

function explicitLinearLayout(params: NodeParameters | undefined, id: string | number): 'din' | 'dout' | null {
  const layout = params?.weight_layout;
  if (layout == null || layout === "") return params?.transB ? "dout" : null;
  if (["OUT_IN", "out_in", "OI", "peft", "dout_din"].includes(layout)) return "dout";
  if (["IN_OUT", "in_out", "IO", "din_dout"].includes(layout)) return "din";
  throw new Error(`Unsupported linear weight_layout '${layout}' at node ${String(id)}.`);
}

export class RuntimeGraph {
  nodes: ConcreteNode[];
  tensors: Map<string, Tensor>;
  weightFiles: SafetensorsFile[];
  private _outputNames: readonly string[];
  topologyRevision: number;
  weightRevision: number;
  _nextTopologyRevision: number;
  _nextWeightRevision: number;
  _nextNodeId: number;
  _autoOutputNames: string[];
  _outputsExplicit: boolean;
  _lastTopologyMutation: string | null;

  constructor() {
    this.nodes = [];
    this.tensors = /* @__PURE__ */ new Map();
    this.weightFiles = [];
    this._outputNames = Object.freeze([]);
    this.topologyRevision = 0;
    this.weightRevision = 0;
    this._nextTopologyRevision = 1;
    this._nextWeightRevision = 1;
    this._nextNodeId = 0;
    this._autoOutputNames = [];
    this._outputsExplicit = false;
    this._lastTopologyMutation = null;
  }

  get outputNames(): readonly string[] {
    return this._outputNames;
  }

  _setOutputNames(names: readonly string[]): void {
    this._outputNames = Object.freeze([...names]);
  }

  _assertTopologyMutationAllowed(): void {
  }

  _resolveTensorRef(
    value: ConcreteTensorReference,
    label = "Tensor reference",
    tensors: Map<string, Tensor> = this.tensors,
  ): Tensor {
    const name = typeof value === "string" ? value : value?.name;
    if (!isValidGraphName(name)) throw new Error(`${label} must reference a named tensor.`);
    const tensor = tensors.get(name);
    if (!tensor) throw new Error(`${label} references missing tensor '${name}'.`);
    if (typeof value !== "string" && value !== tensor) {
      throw new Error(`${label} references a non-canonical Tensor object for '${name}'.`);
    }
    return tensor;
  }

  _commitTopology(nodes: readonly ConcreteNode[], tensors: Map<string, Tensor>, reason: string): number {
    this.nodes.splice(0, this.nodes.length, ...nodes);
    this.tensors.clear();
    for (const [name, tensor] of tensors) this.tensors.set(name, tensor);
    this._refreshOutputNames();
    this.topologyRevision = this._nextTopologyRevision++;
    this._lastTopologyMutation = reason;
    return this.topologyRevision;
  }

  _refreshOutputNames(): readonly string[] {
    if (this._outputsExplicit) {
      this._setOutputNames([...new Set(this.outputNames.filter((name) => this.tensors.has(name)))]);
      return this.outputNames;
    }
    const consumed = new Set<string>();
    for (const node of this.nodes) {
      for (const tensor of Object.values(node.inputs || {})) if (tensor?.name) consumed.add(tensor.name);
    }
    const inferred: string[] = [];
    for (const node of this.nodes) {
      for (const tensor of Object.values(node.outputs || {})) {
        if (tensor?.name && !consumed.has(tensor.name) && !inferred.includes(tensor.name)) inferred.push(tensor.name);
      }
    }
    this._setOutputNames(inferred);
    this._autoOutputNames = [...inferred];
    return this.outputNames;
  }

  _topologySnapshot(): TopologySnapshot {
    return {
      nodes: [...this.nodes],
      tensors: new Map(this.tensors),
      outputNames: [...this.outputNames],
      autoOutputNames: [...this._autoOutputNames],
      outputsExplicit: this._outputsExplicit,
      topologyRevision: this.topologyRevision,
      nextTopologyRevision: this._nextTopologyRevision,
      nextNodeId: this._nextNodeId,
      lastTopologyMutation: this._lastTopologyMutation,
      weightRevision: this.weightRevision,
      nextWeightRevision: this._nextWeightRevision,
      nodeDescriptors: new Map(this.nodes.map((node) => [
        node, Object.getOwnPropertyDescriptors(node),
      ])),
      tensorDescriptors: new Map(Array.from(this.tensors.values(), (tensor) => [
        tensor, Object.getOwnPropertyDescriptors(tensor),
      ])),
    };
  }

  _restoreTopologySnapshot(snapshot: TopologySnapshot): void {
    const restoreProperties = (target: object, descriptors: PropertyDescriptorMap): void => {
      for (const key of Reflect.ownKeys(target)) {
        if (!Object.prototype.hasOwnProperty.call(descriptors, key)) Reflect.deleteProperty(target, key);
      }
      Object.defineProperties(target, descriptors);
    };
    for (const [node, descriptors] of snapshot.nodeDescriptors) restoreProperties(node, descriptors);
    for (const [tensor, descriptors] of snapshot.tensorDescriptors) restoreProperties(tensor, descriptors);
    this.nodes.splice(0, this.nodes.length, ...snapshot.nodes);
    this.tensors.clear();
    for (const [name, tensor] of snapshot.tensors) this.tensors.set(name, tensor);
    this._setOutputNames(snapshot.outputNames);
    this._autoOutputNames = [...snapshot.autoOutputNames];
    this._outputsExplicit = snapshot.outputsExplicit;
    this.topologyRevision = snapshot.topologyRevision;
    this._nextTopologyRevision = snapshot.nextTopologyRevision;
    this._nextNodeId = snapshot.nextNodeId;
    this._lastTopologyMutation = snapshot.lastTopologyMutation;
    this.weightRevision = snapshot.weightRevision;
    this._nextWeightRevision = snapshot.nextWeightRevision;
  }

  _runTopologyTransaction<T>(callback: () => T): T {
    const snapshot = this._topologySnapshot();
    const rollback = (error: unknown): never => {
      this._restoreTopologySnapshot(snapshot);
      throw error;
    };
    try {
      const result = callback();
      return result && typeof (result as { then?: unknown }).then === "function"
        ? Promise.resolve(result).catch(rollback) as T
        : result;
    } catch (error) {
      return rollback(error);
    }
  }

  _createTensor(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType,
    {
      isWeight = false,
      isInput = false,
      buffer,
      quantization,
    }: AddTensorOptions = {},
  ): Tensor {
    if (!isValidGraphName(name)) throw new Error("Tensor name must be a non-empty string.");
    if (!isValidGraphName(dtype)) throw new Error(`Tensor '${name}' dtype must be a non-empty string.`);
    if (isWeight && isInput) throw new Error(`Tensor '${name}' cannot be both an input and a weight.`);
    const tensor = new Tensor(name, validateShape(shape, `Tensor '${name}'`), dtype, isWeight, { isInput, quantization });
    if (buffer != null) {
      Tensor.assertCompatibleBuffer(dtype, buffer, tensor.sizeBytes, `Tensor '${name}' buffer`);
      tensor.buffer = buffer;
    }
    return tensor;
  }

  /** Define a standalone tensor. Intermediate values must later be produced by a node. */
  addTensor(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    options: AddTensorOptions = {},
  ): Tensor {
    this._assertTopologyMutationAllowed();
    if (Object.prototype.hasOwnProperty.call(options, 'data')) {
      throw new Error(`Tensor '${name}' options contain unsupported field 'data'; use 'buffer'.`);
    }
    if (this.tensors.has(name)) throw new Error(`Tensor '${name}' already exists.`);
    const role = options.role || "value";
    if (!new Set(["value", "input", "weight"]).has(role)) {
      throw new Error(`Tensor '${name}' has unsupported role '${role}'.`);
    }
    const tensor = this._createTensor(name, shape, dtype, {
      isInput: role === "input" || options.isInput === true,
      isWeight: role === "weight" || options.isWeight === true,
      buffer: options.buffer,
      quantization: options.quantization,
    });
    const tensors = new Map(this.tensors);
    tensors.set(name, tensor);
    this._commitTopology([...this.nodes], tensors, `add tensor '${name}'`);
    return tensor;
  }

  /**
   * Internal all-or-nothing source-tensor construction for bound snapshots.
   * addTensor() publishes a topology revision immediately; repeating it for a
   * large immutable weight table otherwise copies the growing tensor map on
   * every insertion. This path creates the same tensors privately and commits
   * the complete table once.
   */
  _addTensorsBatch(specs: readonly ConcreteTensorBatchSpec[]): readonly Tensor[] {
    this._assertTopologyMutationAllowed();
    if (!Array.isArray(specs)) throw new Error('Tensor batch must be an array.');
    if (specs.length === 0) return Object.freeze([]);
    const tensors = new Map(this.tensors);
    const added: Tensor[] = [];
    for (const spec of specs) {
      const options = spec.options ?? {};
      if (Object.prototype.hasOwnProperty.call(options, 'data')) {
        throw new Error(`Tensor '${spec.name}' options contain unsupported field 'data'; use 'buffer'.`);
      }
      if (tensors.has(spec.name)) throw new Error(`Tensor '${spec.name}' already exists.`);
      const role = options.role || 'value';
      if (role !== 'value' && role !== 'input' && role !== 'weight') {
        throw new Error(`Tensor '${spec.name}' has unsupported role '${role}'.`);
      }
      const tensor = this._createTensor(spec.name, spec.shape, spec.dtype, {
        isInput: role === 'input' || options.isInput === true,
        isWeight: role === 'weight' || options.isWeight === true,
        buffer: options.buffer,
        quantization: options.quantization,
      });
      tensors.set(spec.name, tensor);
      added.push(tensor);
    }
    this._commitTopology(
      [...this.nodes],
      tensors,
      `add ${added.length} tensor${added.length === 1 ? '' : 's'} as one batch`,
    );
    return Object.freeze(added);
  }

  /**
   * Internal single-publication constructor for a canonically resolved bound
   * graph. The bound-graph layer has already proved the complete tensor/node
   * topology; this method still constructs every Tensor and resolves every
   * node/output reference, but deliberately omits the redundant whole-graph
   * validation scan performed by the public mutation APIs.
   */
  _addResolvedTopologyBatch(
    tensorSpecs: readonly ConcreteTensorBatchSpec[],
    nodeSpecs: readonly ConcreteNodeSpec[],
    outputReferences: readonly ConcreteTensorReference[],
  ): readonly ConcreteNode[] {
    this._assertTopologyMutationAllowed();
    if (this.tensors.size !== 0 || this.nodes.length !== 0 || this.outputNames.length !== 0) {
      throw new Error('Resolved topology construction requires an empty RuntimeGraph.');
    }
    if (!Array.isArray(tensorSpecs) || !Array.isArray(nodeSpecs) ||
        !Array.isArray(outputReferences)) {
      throw new Error('Resolved topology batches must be arrays.');
    }
    const tensors = new Map<string, Tensor>();
    for (const spec of tensorSpecs) {
      const options = spec.options ?? {};
      if (Object.prototype.hasOwnProperty.call(options, 'data')) {
        throw new Error(`Tensor '${spec.name}' options contain unsupported field 'data'; use 'buffer'.`);
      }
      if (tensors.has(spec.name)) throw new Error(`Tensor '${spec.name}' already exists.`);
      const role = options.role || 'value';
      if (role !== 'value' && role !== 'input' && role !== 'weight') {
        throw new Error(`Tensor '${spec.name}' has unsupported role '${role}'.`);
      }
      tensors.set(spec.name, this._createTensor(spec.name, spec.shape, spec.dtype, {
        isInput: role === 'input' || options.isInput === true,
        isWeight: role === 'weight' || options.isWeight === true,
        buffer: options.buffer,
        quantization: options.quantization,
      }));
    }

    const nodes: ConcreteNode[] = [];
    const nodeIds = new Set<string>();
    for (const spec of nodeSpecs) {
      const node = this._prepareNode(spec, tensors, nodes, nodeIds);
      nodes.push(node);
      nodeIds.add(String(node.id));
    }
    const outputNames = outputReferences.map((reference, index) =>
      this._resolveTensorRef(reference, `RuntimeGraph output ${index}`, tensors).name);
    if (new Set(outputNames).size !== outputNames.length) {
      throw new Error('RuntimeGraph outputs must be unique.');
    }

    const consumed = new Set<string>();
    for (const node of nodes) {
      for (const tensor of Object.values(node.inputs)) consumed.add(tensor.name);
    }
    this._autoOutputNames = nodes.flatMap((node) => Object.values(node.outputs))
      .map((tensor) => tensor.name)
      .filter((name, index, values) => !consumed.has(name) && values.indexOf(name) === index);
    this._outputsExplicit = true;
    this._setOutputNames(outputNames);
    this._commitTopology(nodes, tensors, `construct resolved ${nodes.length}-node topology`);
    for (const node of nodes) {
      if (typeof node.id === 'number' && node.id >= this._nextNodeId) {
        this._nextNodeId = node.id + 1;
      }
    }
    return Object.freeze(nodes);
  }

  /**
   * Define an input tensor
   */
  addInput(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    options: AddTensorOptions = {},
  ): Tensor {
    return this.addTensor(name, shape, dtype, { ...options, role: "input" });
  }
  /**
   * Define a weight tensor (learned parameter)
   */
  addWeight(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    options: AddTensorOptions | TensorStorage = {},
  ): Tensor {
    const normalized = ArrayBuffer.isView(options) || options instanceof ArrayBuffer
      ? { buffer: options }
      : { ...options };
    if ('initializer' in normalized && normalized.initializer != null) {
      throw new Error("Weight initializers are available only from the full-profile RuntimeGraphBuilder.");
    }
    delete (normalized as AddTensorOptions & { initializer?: unknown }).initializer;
    return this.addTensor(name, shape, dtype, { ...normalized, role: "weight" });
  }

  _nextAvailableNodeId(
    requested: string | number | null | undefined,
    nodes: readonly ConcreteNode[] = this.nodes,
    knownIds?: ReadonlySet<string>,
  ): string | number {
    const ids = knownIds ?? new Set(nodes.map((node) => String(node.id)));
    if (requested != null) {
      if ((typeof requested !== "string" && typeof requested !== "number") || String(requested).length === 0) {
        throw new Error("Node id must be a non-empty string or number.");
      }
      if (ids.has(String(requested))) throw new Error(`Node id '${String(requested)}' already exists.`);
      return requested;
    }
    let id = this._nextNodeId;
    while (ids.has(String(id))) id++;
    return id;
  }

  _prepareNode(
    spec: ConcreteNodeSpec,
    tensors: Map<string, Tensor>,
    nodes: readonly ConcreteNode[],
    knownIds?: ReadonlySet<string>,
  ): ConcreteNode {
    if (!isRecord(spec)) throw new Error("Node specification must be an object.");
    if (Object.prototype.hasOwnProperty.call(spec, 'op')) {
      throw new Error("Node specification contains unsupported field 'op'; use 'opType'.");
    }
    const opType = spec.opType;
    if (!isValidGraphName(opType)) throw new Error("Node opType must be a non-empty string.");
    if (!isRecord(spec.inputs)) throw new Error(`Node '${opType}' inputs must be an object.`);
    if (!isRecord(spec.outputs) || Object.keys(spec.outputs).length === 0) {
      throw new Error(`Node '${opType}' requires at least one output.`);
    }
    const id = this._nextAvailableNodeId(spec.id, nodes, knownIds);
    const inputs: Record<string, Tensor> = {};
    for (const [key, value] of Object.entries(spec.inputs)) {
      if (!isValidGraphName(key)) throw new Error(`Node '${opType}' has an invalid input key.`);
      inputs[key] = this._resolveTensorRef(value, `Node '${opType}' input '${key}'`, tensors);
    }
    const outputs: Record<string, Tensor> = {};
    for (const [key, value] of Object.entries(spec.outputs)) {
      if (!isValidGraphName(key)) throw new Error(`Node '${opType}' has an invalid output key.`);
      if (typeof value === "string" || value instanceof Tensor) {
        outputs[key] = this._resolveTensorRef(value, `Node '${opType}' output '${key}'`, tensors);
        continue;
      }
      const descriptor = (Array.isArray(value) ? { shape: value } : value) as
        PreparedOutputDescriptor;
      if (!isRecord(descriptor)) {
        throw new Error(`Node '${opType}' output '${key}' must be a shape, tensor name, Tensor, or descriptor.`);
      }
      for (const unsupportedField of ['data', 'replace', 'reuse']) {
        if (Object.prototype.hasOwnProperty.call(descriptor, unsupportedField)) {
          throw new Error(
            `Node '${opType}' output '${key}' contains unsupported field '${unsupportedField}'.`,
          );
        }
      }
      const outName = descriptor.name || `${opType}_${String(id)}_out_${key}`;
      if (tensors.has(outName)) {
        if (descriptor[REPLACE_EXISTING_OUTPUT] === true) {
          const existing = tensors.get(outName)!;
          if (existing.isInput || existing.isWeight) {
            throw new Error(`Node '${opType}' cannot replace source tensor '${outName}'.`);
          }
          const tensor = this._createTensor(outName, descriptor.shape, descriptor.dtype || existing.dtype, {
            buffer: descriptor.buffer,
            quantization: descriptor.quantization,
          });
          tensors.set(outName, tensor);
          outputs[key] = tensor;
          continue;
        }
        throw new Error(`Tensor '${outName}' already exists.`);
      }
      const tensor = this._createTensor(outName, descriptor.shape, descriptor.dtype || "float32", {
        buffer: descriptor.buffer,
        quantization: descriptor.quantization,
      });
      tensors.set(outName, tensor);
      outputs[key] = tensor;
    }
    const node: ConcreteNode = {
      ...spec,
      id,
      opType,
      inputs,
      outputs,
      params: { ...(spec.params || {}) },
    };
    if (["MatMul", "Linear", "Gemm"].includes(opType) && node.wLayout == null) {
      node.wLayout = explicitLinearLayout(node.params, id) || undefined;
    }
    return node;
  }

  /** Add a fully described node and return the node object. */
  addNode(spec: ConcreteNodeSpec): ConcreteNode {
    return this.insertNodeAt(this.nodes.length, spec);
  }

  /**
   * Internal bulk-construction path for already preflighted graph snapshots.
   * Every node is prepared against one private candidate topology, then the
   * complete graph is validated and published once. This preserves the same
   * all-or-nothing mutation boundary as addNode() without its O(N^2) prefix
   * validation cost when materializing hundreds of immutable logical nodes.
   */
  _addNodesBatch(specs: readonly ConcreteNodeSpec[]): readonly ConcreteNode[] {
    this._assertTopologyMutationAllowed();
    if (!Array.isArray(specs)) throw new Error('Node batch must be an array.');
    if (specs.length === 0) return Object.freeze([]);
    const tensors = new Map(this.tensors);
    const nodes = [...this.nodes];
    const nodeIds = new Set(nodes.map((node) => String(node.id)));
    const added: ConcreteNode[] = [];
    for (const spec of specs) {
      const node = this._prepareNode(spec, tensors, nodes, nodeIds);
      nodes.push(node);
      nodeIds.add(String(node.id));
      added.push(node);
    }
    // Outputs are selected separately after construction, matching addNode().
    this._assertValidState(nodes, tensors, []);
    this._commitTopology(
      nodes,
      tensors,
      `append ${added.length} node${added.length === 1 ? '' : 's'} as one batch`,
    );
    for (const node of added) {
      if (typeof node.id === 'number' && node.id >= this._nextNodeId) {
        this._nextNodeId = node.id + 1;
      }
    }
    return Object.freeze(added);
  }

  /** Insert a fully described node at an execution-order index. */
  insertNodeAt(index: number, spec: ConcreteNodeSpec): ConcreteNode {
    this._assertTopologyMutationAllowed();
    if (!Number.isInteger(index) || index < 0 || index > this.nodes.length) {
      throw new Error(`Node insertion index ${index} is out of range.`);
    }
    const tensors = new Map(this.tensors);
    const nodes = [...this.nodes];
    const node = this._prepareNode(spec, tensors, nodes);
    nodes.splice(index, 0, node);
    // Outputs are refreshed atomically at commit time. The pre-commit output
    // list is intentionally excluded because staged builders may rename a new
    // output before appending the next node.
    this._assertValidState(nodes, tensors, []);
    this._commitTopology(nodes, tensors, `insert node '${String(node.id)}'`);
    if (typeof node.id === "number" && node.id >= this._nextNodeId) this._nextNodeId = node.id + 1;
    else if (spec.id == null) this._nextNodeId++;
    return node;
  }

  insertNodeBefore(index: string | number | ConcreteNode, spec: ConcreteNodeSpec): ConcreteNode {
    return this.insertNodeAt(this._nodeIndex(index), spec);
  }

  insertNodeAfter(index: string | number | ConcreteNode, spec: ConcreteNodeSpec): ConcreteNode {
    return this.insertNodeAt(this._nodeIndex(index) + 1, spec);
  }

  /**
   * Add a computation operation to the graph
   * @param {string} opType - e.g., 'MatMul', 'Conv2D', 'LayerNorm'
   * @param {Object} inputs - Key-value pair of input names to Tensor objects
   * @param {Object} outputs - Key-value pair of output names to Tensor shapes
   * @param {Object} params - Uniform parameters for the shader (e.g., stride, kernel size)
   */
  addOp(
    opType: string,
    inputs: Record<string, ConcreteTensorReference>,
    outputs: Record<string, NodeOutputSpec<Tensor>>,
    params: NodeParameters = {},
  ): Record<string, Tensor> {
    return this.addNode({ opType, inputs, outputs, params }).outputs;
  }

  insertOpAt(
    index: number,
    opType: string,
    inputs: Record<string, ConcreteTensorReference>,
    outputs: Record<string, NodeOutputSpec<Tensor>>,
    params: NodeParameters = {},
    options: Omit<ConcreteNodeSpec, 'opType' | 'inputs' | 'outputs' | 'params'> = {},
  ): Record<string, Tensor> {
    return this.insertNodeAt(index, { ...options, opType, inputs, outputs, params }).outputs;
  }

  getTensor(name: string): Tensor | undefined {
    return this.tensors.get(name);
  }

  getNodeById(id: string | number): ConcreteNode | undefined {
    return this.nodes.find((node) => String(node.id) === String(id));
  }

  _nodeIndex(reference: string | number | ConcreteNode): number {
    if (reference && typeof reference === "object") {
      const index = this.nodes.indexOf(reference);
      if (index < 0) throw new Error("Node object does not belong to this graph.");
      return index;
    }
    const index = this.nodes.findIndex((node) => String(node.id) === String(reference));
    if (index >= 0) return index;
    // Numeric references primarily address node ids. Fall back to an execution
    // index for graphs imported with non-numeric ids; explicit *NodeAt methods
    // always use an index and remain unambiguous.
    if (typeof reference === "number" && Number.isInteger(reference) && reference >= 0 && reference < this.nodes.length) {
      return reference;
    }
    throw new Error(`Node id '${String(reference)}' not found.`);
  }

  getNode(reference: string | number | ConcreteNode): ConcreteNode {
    return this.nodes[this._nodeIndex(reference)];
  }

  _validationReport(
    nodes: readonly ConcreteNode[] = this.nodes,
    tensors: Map<string, Tensor> = this.tensors,
    outputNames: readonly string[] = this.outputNames,
  ): GraphValidationReport {
    const errors: string[] = [];
    const warnings: string[] = [];
    const ids = new Set<string>();
    const producers = new Map<string, number>();
    const consumed = new Set<string>();

    for (const [name, tensor] of tensors) {
      if (!isValidGraphName(name)) errors.push("Tensor map contains an invalid name.");
      if (!(tensor instanceof Tensor)) errors.push(`Tensor '${name}' is not a Tensor instance.`);
      if (tensor?.name !== name) errors.push(`Tensor map key '${name}' does not match tensor name '${tensor?.name}'.`);
      if (!Array.isArray(tensor?.shape) || tensor.shape.some((dim) => !Number.isInteger(dim) || dim <= 0)) {
        errors.push(`Tensor '${name}' has an invalid shape.`);
      }
      if (!isValidGraphName(tensor?.dtype)) errors.push(`Tensor '${name}' has an invalid dtype.`);
      else {
        try {
          const expected = tensor._calculateByteSize();
          if (tensor.sizeBytes !== expected) errors.push(`Tensor '${name}' sizeBytes does not match its shape/dtype.`);
          if (tensor.buffer != null) Tensor.assertCompatibleBuffer(tensor.dtype, tensor.buffer, expected, `Tensor '${name}' buffer`);
          Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
            `Tensor '${name}' quantization`);
        } catch (error) {
          errors.push(error instanceof Error ? error.message : String(error));
        }
      }
      if (tensor?.isWeight && tensor?.isInput) errors.push(`Tensor '${name}' cannot be both input and weight.`);
    }

    nodes.forEach((node, index) => {
      const label = nodeLabel(node, index);
      const id = String(node?.id ?? "");
      if (!id.length) errors.push(`${label} has no id.`);
      else if (ids.has(id)) errors.push(`Duplicate node id '${id}'.`);
      else ids.add(id);
      if (!isValidGraphName(node?.opType)) errors.push(`${label} has an invalid opType.`);
      if (!isRecord(node?.inputs)) errors.push(`${label} inputs must be an object.`);
      if (!isRecord(node?.outputs) || Object.keys(node.outputs).length === 0) {
        errors.push(`${label} requires at least one output.`);
      }
      const localOutputs = new Set<string>();
      for (const [key, tensor] of Object.entries(node?.outputs || {})) {
        const canonical = tensor?.name ? tensors.get(tensor.name) : undefined;
        if (!isValidGraphName(key)) errors.push(`${label} has an invalid output key.`);
        if (!canonical) errors.push(`${label} output '${key}' references a missing tensor.`);
        else if (canonical !== tensor) errors.push(`${label} output '${key}' is not the canonical tensor '${tensor.name}'.`);
        if (canonical?.isInput || canonical?.isWeight) {
          errors.push(`${label} output '${key}' cannot produce source tensor '${canonical.name}'.`);
        }
        if (canonical && localOutputs.has(canonical.name)) {
          errors.push(`${label} produces tensor '${canonical.name}' more than once.`);
        }
        if (canonical) {
          localOutputs.add(canonical.name);
          if (producers.has(canonical.name)) {
            const producerIndex = producers.get(canonical.name)!;
            errors.push(`Tensor '${canonical.name}' is produced by both ${nodeLabel(nodes[producerIndex], producerIndex)} and ${label}.`);
          } else {
            producers.set(canonical.name, index);
          }
        }
      }
    });

    nodes.forEach((node, index) => {
      const label = nodeLabel(node, index);
      for (const [key, tensor] of Object.entries(node?.inputs || {})) {
        const canonical = tensor?.name ? tensors.get(tensor.name) : undefined;
        if (!isValidGraphName(key)) errors.push(`${label} has an invalid input key.`);
        if (!canonical) {
          errors.push(`${label} input '${key}' references a missing tensor.`);
          continue;
        }
        if (canonical !== tensor) errors.push(`${label} input '${key}' is not the canonical tensor '${tensor.name}'.`);
        consumed.add(canonical.name);
        const producer = producers.get(canonical.name);
        if (producer == null) {
          if (!canonical.isInput && !canonical.isWeight) {
            errors.push(`${label} input '${key}' references detached value tensor '${canonical.name}'.`);
          }
        } else if (producer >= index) {
          errors.push(`${label} input '${key}' uses '${canonical.name}' before it is produced.`);
        }
      }
    });

    const seenOutputs = new Set();
    for (const name of outputNames || []) {
      if (!isValidGraphName(name) || !tensors.has(name)) errors.push(`RuntimeGraph output '${String(name)}' does not exist.`);
      else if (seenOutputs.has(name)) errors.push(`RuntimeGraph output '${name}' is listed more than once.`);
      seenOutputs.add(name);
    }
    for (const [name, tensor] of tensors) {
      if (!tensor.isInput && !tensor.isWeight && !producers.has(name)) {
        warnings.push(`Value tensor '${name}' is detached.`);
      }
      if (tensor.isWeight && !consumed.has(name)) warnings.push(`Weight tensor '${name}' is unused.`);
    }
    if (nodes.length > 0 && (outputNames || []).length === 0) warnings.push("RuntimeGraph has nodes but no selected outputs.");
    return Object.freeze({
      valid: errors.length === 0,
      errors: Object.freeze(errors),
      warnings: Object.freeze(warnings),
      nodeCount: nodes.length,
      tensorCount: tensors.size,
      topologyRevision: this.topologyRevision,
    });
  }

  _assertValidState(
    nodes: readonly ConcreteNode[],
    tensors: Map<string, Tensor>,
    outputNames: readonly string[] = this.outputNames,
  ): GraphValidationReport {
    const report = this._validationReport(nodes, tensors, outputNames);
    if (!report.valid) throw new Error(`Invalid graph: ${report.errors.join(" ")}`);
    return report;
  }

  validate({ throwOnError = false }: { throwOnError?: boolean } = {}): GraphValidationReport {
    const report = this._validationReport();
    if (throwOnError && !report.valid) throw new Error(`Invalid graph: ${report.errors.join(" ")}`);
    return report;
  }

  assertValid(): this {
    this.validate({ throwOnError: true });
    return this;
  }

  assertTopologyRevision(compiledRevision: number, backend = "Compiled backend"): void {
    if (compiledRevision !== this.topologyRevision) {
      const reason = this._lastTopologyMutation ? ` Last mutation: ${this._lastTopologyMutation}.` : "";
      throw new Error(`${backend} graph topology changed after compilation; recompile before execution.${reason}`);
    }
  }

  setOutputs(...references: Array<ConcreteTensorReference | readonly ConcreteTensorReference[]>): this {
    this._assertTopologyMutationAllowed();
    const values: readonly ConcreteTensorReference[] = references.length === 1 && Array.isArray(references[0])
      ? references[0]
      : references as ConcreteTensorReference[];
    const names = values.map((value, index) => this._resolveTensorRef(value, `RuntimeGraph output ${index}`).name);
    if (new Set(names).size !== names.length) throw new Error("RuntimeGraph outputs must be unique.");
    this._setOutputNames(names);
    this._outputsExplicit = true;
    this._commitTopology([...this.nodes], new Map(this.tensors), "select graph outputs");
    return this;
  }

  inferOutputs(): string[] {
    this._assertTopologyMutationAllowed();
    this._outputsExplicit = false;
    this._setOutputNames(this._autoOutputNames);
    this._commitTopology([...this.nodes], new Map(this.tensors), "infer graph outputs");
    return [...this.outputNames];
  }

  renameTensor(name: string, nextName: string): Tensor {
    this._assertTopologyMutationAllowed();
    const tensor = this.getTensor(name);
    if (!tensor) throw new Error(`Tensor '${name}' not found.`);
    if (!isValidGraphName(nextName)) throw new Error("Tensor name must be a non-empty string.");
    if (this.tensors.has(nextName)) throw new Error(`Tensor '${nextName}' already exists.`);
    const replacement = Object.assign(Object.create(Object.getPrototypeOf(tensor)), tensor, { name: nextName });
    const tensors = new Map();
    for (const [entryName, entry] of this.tensors) tensors.set(entryName === name ? nextName : entryName, entry === tensor ? replacement : entry);
    const nodes = this.nodes.map((node) => ({
      ...node,
      inputs: Object.fromEntries(Object.entries(node.inputs || {}).map(([key, value]) => [key, value === tensor ? replacement : value])),
      outputs: Object.fromEntries(Object.entries(node.outputs || {}).map(([key, value]) => [key, value === tensor ? replacement : value])),
      params: { ...(node.params || {}) },
    }));
    const mappedOutputs = this.outputNames.map((value) => value === name ? nextName : value);
    this._assertValidState(nodes, tensors, mappedOutputs);
    tensor.name = nextName;
    tensors.set(nextName, tensor);
    const committedNodes = nodes.map((node) => ({
      ...node,
      inputs: Object.fromEntries(Object.entries(node.inputs || {}).map(([key, value]) => [key, value === replacement ? tensor : value])),
      outputs: Object.fromEntries(Object.entries(node.outputs || {}).map(([key, value]) => [key, value === replacement ? tensor : value])),
      params: { ...(node.params || {}) },
    }));
    this._setOutputNames(mappedOutputs);
    if (!this._outputsExplicit) this._autoOutputNames = [...mappedOutputs];
    this._commitTopology(committedNodes, tensors, `rename tensor '${name}' to '${nextName}'`);
    return tensor;
  }

  updateTensor(name: string, patch: TensorPatch = {}): Tensor {
    this._assertTopologyMutationAllowed();
    const tensor = this.getTensor(name);
    if (!tensor) throw new Error(`Tensor '${name}' not found.`);
    if (!isRecord(patch as unknown)) throw new Error("Tensor patch must be an object.");
    if (Object.prototype.hasOwnProperty.call(patch, 'data')) {
      throw new Error(`Tensor '${name}' patch contains unsupported field 'data'; use 'buffer'.`);
    }
    if (patch.name != null && patch.name !== name) return this.renameTensor(name, patch.name);
    const next = Object.assign(Object.create(Object.getPrototypeOf(tensor)), tensor);
    if (patch.shape != null) next.shape = validateShape(patch.shape, `Tensor '${name}'`);
    if (patch.dtype != null) {
      if (!isValidGraphName(patch.dtype)) throw new Error(`Tensor '${name}' dtype must be a non-empty string.`);
      next.dtype = patch.dtype;
    }
    if (Object.prototype.hasOwnProperty.call(patch, "quantization")) {
      next.quantization = Tensor.normalizeQuantization(next.dtype, next.shape, patch.quantization,
        `Tensor '${name}' quantization`);
    } else if (next.quantization) {
      // A shape/dtype update can invalidate a per-axis descriptor. Revalidate
      // it rather than leaving an inconsistent quantized edge in the graph.
      next.quantization = Tensor.normalizeQuantization(next.dtype, next.shape, next.quantization,
        `Tensor '${name}' quantization`);
    }
    if (patch.role != null) {
      if (!["value", "input", "weight"].includes(patch.role)) throw new Error(`Unsupported tensor role '${patch.role}'.`);
      next.isWeight = patch.role === "weight";
      next.isInput = patch.role === "input";
    }
    const buffer = patch.buffer;
    if (buffer != null) {
      const expected = next._calculateByteSize();
      Tensor.assertCompatibleBuffer(next.dtype, buffer, expected, `Tensor '${name}' buffer`);
      next.buffer = buffer;
      next.sizeBytes = expected;
    } else if (patch.shape != null || patch.dtype != null) {
      next.sizeBytes = next._calculateByteSize();
      if (next.buffer && next.buffer.byteLength !== next.sizeBytes) next.buffer = undefined;
    }
    const tensors = new Map(this.tensors);
    tensors.set(name, next);
    const nodes = this.nodes.map((node) => ({
      ...node,
      inputs: Object.fromEntries(Object.entries(node.inputs || {}).map(([key, value]) => [key, value === tensor ? next : value])),
      outputs: Object.fromEntries(Object.entries(node.outputs || {}).map(([key, value]) => [key, value === tensor ? next : value])),
      params: { ...(node.params || {}) },
    }));
    this._assertValidState(nodes, tensors);
    this._commitTopology(nodes, tensors, `update tensor '${name}'`);
    return next;
  }

  removeTensor(name: string, { cascade = false }: RemoveTensorOptions = {}): boolean {
    this._assertTopologyMutationAllowed();
    if (!this.tensors.has(name)) return false;
    const direct = new Set<number>();
    this.nodes.forEach((node, index) => {
      if (Object.values(node.inputs || {}).some((tensor) => tensor?.name === name) ||
          Object.values(node.outputs || {}).some((tensor) => tensor?.name === name)) direct.add(index);
    });
    if (direct.size && !cascade) {
      throw new Error(`Tensor '${name}' is referenced by ${direct.size} node(s); pass { cascade: true } to remove dependent nodes.`);
    }
    const remove = new Set<number>(direct);
    if (cascade) {
      let changed = true;
      while (changed) {
        changed = false;
        const removedOutputs = new Set<string>();
        for (const index of remove) {
          for (const tensor of Object.values(this.nodes[index]?.outputs || {})) removedOutputs.add(tensor.name);
        }
        this.nodes.forEach((node, index) => {
          if (!remove.has(index) && Object.values(node.inputs || {}).some((tensor) => removedOutputs.has(tensor.name))) {
            remove.add(index);
            changed = true;
          }
        });
      }
    }
    const tensors = new Map(this.tensors);
    tensors.delete(name);
    for (const index of remove) {
      for (const tensor of Object.values(this.nodes[index].outputs || {})) tensors.delete(tensor.name);
    }
    const nodes = this.nodes.filter((_, index) => !remove.has(index));
    this._assertValidState(nodes, tensors, []);
    this._commitTopology(nodes, tensors, `remove tensor '${name}'`);
    return true;
  }

  replaceNodeAt(index: number, replacement: ConcreteNodePatch = {}): ConcreteNode {
    this._assertTopologyMutationAllowed();
    if (!Number.isInteger(index) || index < 0 || index >= this.nodes.length) {
      throw new Error(`Node index ${index} is out of range.`);
    }
    if (!isRecord(replacement)) throw new Error("Replacement node specification must be an object.");
    const previous = this.nodes[index];
    const spec: ConcreteNodeSpec = {
      ...previous,
      ...replacement,
      id: replacement.id ?? previous.id,
      opType: replacement.opType || previous.opType,
      inputs: replacement.inputs ?? previous.inputs,
      outputs: replacement.outputs ?? previous.outputs,
      params: replacement.params ?? previous.params,
    };
    if (replacement.outputs) {
      spec.outputs = Object.fromEntries(Object.entries(replacement.outputs).map(([key, value]) => {
        const oldTensor = previous.outputs?.[key];
        if (Array.isArray(value) && oldTensor) {
          return [key, {
            name: oldTensor.name,
            shape: value,
            dtype: oldTensor.dtype,
            [REPLACE_EXISTING_OUTPUT]: true,
          }];
        }
        if (isRecord(value) && !(value instanceof Tensor) && oldTensor && !value.name) {
          const descriptor = value as Record<string, unknown>;
          return [key, {
            ...descriptor,
            name: oldTensor.name,
            [REPLACE_EXISTING_OUTPUT]: true,
          }];
        }
        return [key, value];
      }));
    }
    const tensors = new Map(this.tensors);
    const nodesWithoutPrevious = this.nodes.filter((_, nodeIndex) => nodeIndex !== index);
    const next = this._prepareNode(spec, tensors, nodesWithoutPrevious);
    const rewrites = new Map();
    for (const [key, oldTensor] of Object.entries(previous.outputs || {})) {
      const nextTensor = next.outputs?.[key];
      if (nextTensor && nextTensor !== oldTensor) rewrites.set(oldTensor, nextTensor);
    }
    const nodes = this.nodes.map((node, nodeIndex) => {
      if (nodeIndex === index) return next;
      if (nodeIndex < index || rewrites.size === 0) return node;
      return {
        ...node,
        inputs: Object.fromEntries(Object.entries(node.inputs || {}).map(([key, tensor]) => [key, rewrites.get(tensor) || tensor])),
        outputs: { ...(node.outputs || {}) },
        params: { ...(node.params || {}) },
      };
    });
    const retained = new Set();
    for (const node of nodes) {
      for (const tensor of Object.values(node.inputs || {})) retained.add(tensor.name);
      for (const tensor of Object.values(node.outputs || {})) retained.add(tensor.name);
    }
    for (const oldTensor of Object.values(previous.outputs || {})) {
      if (!retained.has(oldTensor.name)) tensors.delete(oldTensor.name);
    }
    const nextOutputNames = this.outputNames.map((name) => {
      const oldTensor = Object.values(previous.outputs || {}).find((tensor) => tensor.name === name);
      return oldTensor && rewrites.has(oldTensor) ? rewrites.get(oldTensor).name : name;
    }).filter((name) => tensors.has(name));
    this._assertValidState(nodes, tensors, nextOutputNames);
    if (this._outputsExplicit) this._setOutputNames(nextOutputNames);
    this._commitTopology(nodes, tensors, `replace node '${String(previous.id)}'`);
    if (typeof next.id === "number" && next.id >= this._nextNodeId) this._nextNodeId = next.id + 1;
    return next;
  }

  removeNodeAt(index: number, options: RemoveNodeOptions = {}): ConcreteNode[] {
    this._assertTopologyMutationAllowed();
    if (!Number.isInteger(index) || index < 0 || index >= this.nodes.length) {
      throw new Error(`Node index ${index} is out of range.`);
    }
    const cascade = options.cascade === true;
    const preserveOutputs = options.removeOutputs === false || options.preserveOutputsAsInputs === true;
    const rewireSpec = options.rewire || {};
    if (!isRecord(rewireSpec)) throw new Error("Node rewire option must be an object.");
    const remove = new Set([index]);
    const rewrites = new Map();
    const start = this.nodes[index];
    for (const [key, tensor] of Object.entries(start.outputs || {})) {
      const value = rewireSpec[tensor.name] ?? rewireSpec[key];
      if (value != null) rewrites.set(tensor, this._resolveTensorRef(value, `Rewire for '${tensor.name}'`));
    }

    let changed = true;
    while (changed) {
      changed = false;
      const removedOutputs = new Set();
      for (const nodeIndex of remove) {
        for (const tensor of Object.values(this.nodes[nodeIndex].outputs || {})) removedOutputs.add(tensor);
      }
      this.nodes.forEach((node, nodeIndex) => {
        if (remove.has(nodeIndex)) return;
        for (const tensor of Object.values(node.inputs || {})) {
          if (!removedOutputs.has(tensor) || rewrites.has(tensor) || preserveOutputs) continue;
          if (!cascade) {
            throw new Error(`Cannot remove node '${String(start.id)}'; tensor '${tensor.name}' is consumed by node '${String(node.id)}'. Rewire it or pass { cascade: true }.`);
          }
          remove.add(nodeIndex);
          changed = true;
          break;
        }
      });
    }
    const removedNodes = [...remove].sort((a, b) => a - b).map((nodeIndex) => this.nodes[nodeIndex]);

    const tensors = new Map(this.tensors);
    if (preserveOutputs) {
      for (const nodeIndex of remove) {
        for (const tensor of Object.values(this.nodes[nodeIndex].outputs || {})) {
          if (rewrites.has(tensor)) continue;
          const input = Object.assign(Object.create(Object.getPrototypeOf(tensor)), tensor, { isInput: true, isWeight: false });
          tensors.set(input.name, input);
          rewrites.set(tensor, input);
        }
      }
    } else {
      for (const nodeIndex of remove) {
        for (const tensor of Object.values(this.nodes[nodeIndex].outputs || {})) {
          tensors.delete(tensor.name);
        }
      }
    }
    const nodes = this.nodes.filter((_, nodeIndex) => !remove.has(nodeIndex)).map((node) => ({
      ...node,
      inputs: Object.fromEntries(Object.entries(node.inputs || {}).map(([key, tensor]) => [key, rewrites.get(tensor) || tensor])),
      outputs: Object.fromEntries(Object.entries(node.outputs || {}).map(([key, tensor]) => [key, rewrites.get(tensor) || tensor])),
      params: { ...(node.params || {}) },
    }));
    const nextOutputNames = this.outputNames.map((name) => {
      const oldTensor = this.tensors.get(name);
      return oldTensor && rewrites.has(oldTensor) ? rewrites.get(oldTensor).name : name;
    }).filter((name) => tensors.has(name));
    this._assertValidState(nodes, tensors, nextOutputNames);
    if (this._outputsExplicit) this._setOutputNames(nextOutputNames);
    this._commitTopology(nodes, tensors, `remove node '${String(start.id)}'`);
    return removedNodes;
  }

  removeNode(reference: string | number | ConcreteNode, options: RemoveNodeOptions = {}): ConcreteNode {
    const index = this._nodeIndex(reference);
    const removed = this.nodes[index];
    this.removeNodeAt(index, options);
    return removed;
  }

  setTensorBuffer(name: string, buffer: ArrayBuffer | ArrayBufferView): Tensor {
    const tensor = this.getTensor(name);
    if (!tensor) throw new Error(`Tensor '${name}' not found.`);
    Tensor.assertCompatibleBuffer(tensor.dtype, buffer, tensor._calculateByteSize(), `Tensor '${name}' buffer`);
    tensor.buffer = buffer;
    tensor.sizeBytes = buffer.byteLength;
    if (tensor.isWeight) this._advanceWeightRevision();
    return tensor;
  }

  applyTensorUpdate(
    name: string,
    update: Float32Array | ArrayBuffer | ArrayBufferView | ArrayLike<number>,
    options: TensorUpdateOptions = {},
  ): Tensor {
    const tensor = this.getTensor(name);
    if (!tensor) throw new Error(`Tensor '${name}' not found.`);
    if (tensor.dtype !== "float32") {
      throw new Error(`Tensor '${name}' update requires float32, got '${tensor.dtype}'.`);
    }
    if (!tensor.buffer) throw new Error(`Tensor '${name}' has no CPU buffer to update.`);
    const target = tensor.buffer instanceof Float32Array
      ? tensor.buffer
      : new Float32Array(tensor.buffer instanceof ArrayBuffer ? tensor.buffer : tensor.buffer.buffer);
    const source = update instanceof Float32Array
      ? update
      : ArrayBuffer.isView(update)
        ? new Float32Array(update.buffer, update.byteOffset, update.byteLength / 4)
        : new Float32Array(update);
    if (source.length !== target.length) {
      throw new Error(`Tensor '${name}' update length mismatch: got ${source.length}, want ${target.length}.`);
    }

    const mode = options.mode || "assign";

    if (mode === "assign") {
      target.set(source);
    } else if (mode === "add") {
      for (let i = 0; i < target.length; i++) target[i] += source[i];
    } else {
      throw new Error(`Unsupported generic tensor update mode '${mode}'.`);
    }

    tensor.buffer = target;
    tensor.sizeBytes = target.byteLength;
    if (tensor.isWeight) this._advanceWeightRevision();
    return tensor;
  }

  _advanceWeightRevision(): number {
    this.weightRevision = this._nextWeightRevision++;
    return this.weightRevision;
  }

  inspect({ includeWeightFiles = true, includeParams = true }: GraphInspectionOptions = {}): GraphInspection {
    const tensors: GraphInspection['tensors'] = [];
    for (const tensor of this.tensors.values()) {
      tensors.push({
        name: tensor.name,
        shape: [...tensor.shape],
        dtype: tensor.dtype,
        isWeight: !!tensor.isWeight,
        isInput: !!tensor.isInput,
        role: tensor.isWeight ? "weight" : tensor.isInput ? "input" : "value",
        sizeBytes: tensor.sizeBytes || tensor.buffer?.byteLength || 0,
      });
    }
    const nodes = this.nodes.map((node, index) => ({
      index,
      id: String(node.id ?? index),
      opType: node.opType,
      inputs: Object.fromEntries(Object.entries(node.inputs || {}).map(([k, t]) => [k, t?.name || ""])),
      outputs: Object.fromEntries(Object.entries(node.outputs || {}).map(([k, t]) => [k, t?.name || ""])),
      outputShapes: Object.fromEntries(Object.entries(node.outputs || {}).map(([k, t]) => [k, [...(t?.shape || [])]])),
      params: includeParams ? { ...(node.params || {}) } : undefined,
    }));
    return {
      tensorCount: tensors.length,
      nodeCount: nodes.length,
      tensors,
      nodes,
      weightFiles: includeWeightFiles ? this.weightFiles.map((file) => file.inspect()) : [],
      outputNames: [...(this.outputNames || [])],
      topologyRevision: this.topologyRevision,
      weightRevision: this.weightRevision,
    };
  }

  patchNodeAt(index: number, patch: ConcreteNodePatch = {}): ConcreteNode {
    const node = this.nodes[index];
    if (!node) throw new Error(`Node ${index} not found.`);
    if (!isRecord(patch)) throw new Error("Node patch must be an object.");
    const replacement = this.replaceNodeAt(index, {
      ...patch,
      id: node.id,
      opType: patch.opType || node.opType,
      inputs: patch.inputs ? { ...(node.inputs || {}), ...patch.inputs } : node.inputs,
      outputs: patch.outputs ? { ...(node.outputs || {}), ...patch.outputs } : node.outputs,
      params: patch.params
        ? (patch.replaceParams ? { ...patch.params } : { ...(node.params || {}), ...patch.params })
        : node.params,
    });
    // patchNodeAt keeps its node handle stable while using replaceNodeAt's
    // transactional validation and downstream tensor rewrites.
    Object.assign(node, replacement);
    this.nodes[index] = node;
    return node;
  }
};
