import { Graph } from './Graph.js';
import type { Tensor } from './Tensor.js';
import type {
  AdapterDescription,
  AdapterSpec,
  AdapterStageOptions,
  AdapterVersion,
  AddTensorOptions,
  BlueprintConfig,
  GraphNode,
  GraphNodePatch,
  GraphNodeSpec,
  GraphValidationReport,
  NodeOutputSpec,
  NodeParameters,
  RuntimeDType,
  SerializedQuantization,
  TensorPatch,
  TensorQuantization,
  TensorReference,
  TensorStorage,
} from '../types.js';

type ConcreteNode = GraphNode<Tensor>;
type ConcreteNodeSpec = GraphNodeSpec<Tensor>;
type ConcreteNodePatch = GraphNodePatch<Tensor>;
type ConcreteTensorReference = TensorReference<Tensor>;
type NodeOptions = Partial<Pick<ConcreteNodeSpec, 'id' | 'wLayout'>> & Record<string, unknown>;

interface GroupNormOptions {
  numGroups?: number;
  epsilon?: number;
  name?: string;
}

interface MoeRouterOptions {
  bias?: Tensor | null;
  topK?: number;
  numExperts?: number;
  normalize?: boolean;
  temperature?: number;
  name?: string;
}

interface MoeRoutes {
  indices?: Tensor;
  expert_indices?: Tensor;
  weights?: Tensor;
  expert_weights?: Tensor;
}

interface MoeLinearOptions {
  bias?: Tensor | null;
  name?: string;
}

interface AdapterRoute {
  name: string;
  version?: AdapterVersion;
  scale: number;
}

function serializeQuantization(quantization: null): null;
function serializeQuantization(quantization: TensorQuantization): SerializedQuantization;
function serializeQuantization(quantization: TensorQuantization | null): SerializedQuantization | null {
  if (!quantization) return null;
  if (quantization.scheme === 'per_axis') {
    return {
      scheme: quantization.scheme,
      axis: quantization.axis,
      scales: [...quantization.scales],
      zero_points: [...quantization.zero_points],
    };
  }
  return {
    scheme: quantization.scheme,
    scale: quantization.scale,
    zero_point: quantization.zero_point,
  };
}

/**
 * Programmatic model construction and editing facade.
 *
 * ModelBuilder never owns a separate graph representation: every successful
 * operation is committed directly to `graph`, and every failed operation leaves
 * it untouched. This makes a builder equally useful for a new model and for an
 * already-loaded model that is being edited.
 */
export class ModelBuilder {
  graph: Graph;

  constructor(graph: Graph = new Graph()) {
    if (!(graph instanceof Graph)) throw new Error("ModelBuilder expects a Graph instance.");
    this.graph = graph;
  }

  /**
   * Run a group of structural builder edits atomically.
   *
   * A thrown error or rejected promise restores the exact prior topology,
   * selected outputs, object identities, and revision counters. Tensor-value
   * updates and adapter lifecycle changes are intentionally outside this scope.
   */
  topologyTransaction<T>(callback: (builder: this) => T): T {
    if (typeof callback !== "function") {
      throw new Error("topologyTransaction expects a callback.");
    }
    return this.graph._runTopologyTransaction(() => callback(this));
  }

  tensor(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    options: AddTensorOptions = {},
  ): Tensor {
    return this.graph.addTensor(name, shape, dtype, options);
  }

  addTensor(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    options: AddTensorOptions = {},
  ): Tensor {
    return this.tensor(name, shape, dtype, options);
  }

  input(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    options: AddTensorOptions = {},
  ): Tensor {
    return this.graph.addInput(name, shape, dtype, options);
  }

  addInput(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    options: AddTensorOptions = {},
  ): Tensor {
    return this.input(name, shape, dtype, options);
  }

  weight(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    dataOrOptions: AddTensorOptions | TensorStorage = {},
  ): Tensor {
    const options = ArrayBuffer.isView(dataOrOptions) || dataOrOptions instanceof ArrayBuffer
      ? { buffer: dataOrOptions }
      : dataOrOptions;
    return this.graph.addWeight(name, shape, dtype, options);
  }

  addWeight(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    dataOrOptions: AddTensorOptions | TensorStorage = {},
  ): Tensor {
    return this.weight(name, shape, dtype, dataOrOptions);
  }

  getTensor(name: string): Tensor | undefined {
    return this.graph.getTensor(name);
  }

  updateTensor(name: string, patch: TensorPatch = {}): Tensor {
    return this.graph.updateTensor(name, patch);
  }

  renameTensor(name: string, nextName: string): Tensor {
    return this.graph.renameTensor(name, nextName);
  }

  removeTensor(name: string, options: { cascade?: boolean } = {}): boolean {
    return this.graph.removeTensor(name, options);
  }

  addNode(spec: ConcreteNodeSpec): ConcreteNode {
    return this.graph.addNode(spec);
  }

  node(spec: ConcreteNodeSpec): ConcreteNode {
    return this.addNode(spec);
  }

  /** Existing Graph.addOp-compatible convenience that returns output tensors. */
  addOp(
    opType: string,
    inputs: Record<string, ConcreteTensorReference>,
    outputs: Record<string, NodeOutputSpec<Tensor>>,
    params: NodeParameters = {},
    options: NodeOptions = {},
  ): Record<string, Tensor> {
    return this.graph.addNode({ ...options, opType, inputs, outputs, params }).outputs;
  }

  op(
    opType: string,
    inputs: Record<string, ConcreteTensorReference>,
    outputs: Record<string, NodeOutputSpec<Tensor>>,
    params: NodeParameters = {},
    options: NodeOptions = {},
  ): Record<string, Tensor> {
    return this.addOp(opType, inputs, outputs, params, options);
  }

  /** Add NHWC GroupNorm with trainable per-channel affine tensors supplied by the caller. */
  groupNorm(input: Tensor, weight: Tensor, bias: Tensor, {
    numGroups,
    epsilon = 1e-5,
    name = `group_norm_${this.graph.nodes.length}`,
  }: GroupNormOptions = {}): Tensor {
    if (!input || input.dtype !== "float32" || input.shape?.length !== 4) {
      throw new Error("groupNorm input must be a rank-4 NHWC float32 tensor.");
    }
    const channels = input.shape[3];
    if (typeof numGroups !== 'number' || !Number.isSafeInteger(numGroups) ||
        numGroups <= 0 || channels % numGroups !== 0) {
      throw new Error(`groupNorm numGroups must be a positive divisor of ${channels}.`);
    }
    if (!weight || !bias || weight.dtype !== "float32" || bias.dtype !== "float32" ||
        weight.shape?.length !== 1 || bias.shape?.length !== 1 ||
        weight.shape[0] !== channels || bias.shape[0] !== channels) {
      throw new Error(`groupNorm weight and bias must be float32 tensors shaped [${channels}].`);
    }
    if (typeof epsilon !== "number" || !Number.isFinite(epsilon) || epsilon <= 0) {
      throw new Error("groupNorm epsilon must be positive and finite.");
    }
    return this.addOp(
      "GroupNorm",
      { input, weight, bias },
      { out: { name: `${name}.out`, shape: [...input.shape] } },
      { num_groups: numGroups, eps: epsilon },
      { id: name },
    ).out;
  }

  /** Add a differentiable top-k Mixture-of-Experts router. */
  moeRouter(input: Tensor, routerWeight: Tensor, {
    bias = null,
    topK = 2,
    numExperts = routerWeight?.shape?.[routerWeight.shape.length - 1],
    normalize = true,
    temperature = 1,
    name = `moe_router_${this.graph.nodes.length}`,
  }: MoeRouterOptions = {}): Record<string, Tensor> {
    const routeShape = [...input.shape.slice(0, -1), topK];
    return this.addOp('MoERouter', {
      input,
      weight: routerWeight,
      ...(bias ? { bias } : {}),
    }, {
      indices: { name: `${name}.indices`, shape: routeShape },
      weights: { name: `${name}.weights`, shape: routeShape },
    }, { num_experts: numExperts, top_k: topK, normalize, temperature }, { id: name });
  }

  /** Add a routed expert linear layer using MoERouter outputs. */
  moeLinear(input: Tensor, expertWeight: Tensor, routes: MoeRoutes, {
    bias = null,
    name = `moe_linear_${this.graph.nodes.length}`,
  }: MoeLinearOptions = {}): Record<string, Tensor> {
    const routeIndices = (routes.indices || routes.expert_indices) as Tensor;
    const routeWeights = (routes.weights || routes.expert_weights) as Tensor;
    const dOut = expertWeight.shape[expertWeight.shape.length - 1];
    return this.addOp('MoELinear', {
      input,
      expert_weight: expertWeight,
      route_indices: routeIndices,
      route_weights: routeWeights,
      ...(bias ? { expert_bias: bias } : {}),
    }, {
      out: { name: `${name}.out`, shape: [...input.shape.slice(0, -1), dOut] },
    }, {}, { id: name });
  }

  /** Pool [B,T,D] tokens with caller-supplied normalized F32 weights [B,T]. */
  maskedMean(input: Tensor, normalizedWeights: Tensor, {
    name = `masked_mean_${this.graph.nodes.length}`,
  }: { name?: string } = {}): Tensor {
    if (input?.dtype !== "float32" || input.shape?.length !== 3) {
      throw new Error("maskedMean input must be a [B,T,D] float32 tensor.");
    }
    const [batch, length, width] = input.shape;
    if (normalizedWeights?.dtype !== "float32" || normalizedWeights.shape?.length !== 2 ||
        normalizedWeights.shape[0] !== batch || normalizedWeights.shape[1] !== length) {
      throw new Error(`maskedMean weights must be a float32 tensor shaped [${batch},${length}].`);
    }
    const transposed = this.addOp(
      "Transpose",
      { input },
      { out: { name: `${name}.transposed`, shape: [batch, width, length] } },
      { perm: [0, 2, 1] },
      { id: `${name}.transpose` },
    ).out;
    const expandedWeights = this.addOp(
      "Unsqueeze",
      { input: normalizedWeights },
      { out: { name: `${name}.weights`, shape: [batch, 1, length] } },
      { axes: [1] },
      { id: `${name}.weights` },
    ).out;
    const weighted = this.addOp(
      "Mul",
      { a: transposed, b: expandedWeights },
      { out: { name: `${name}.weighted`, shape: [batch, width, length] } },
      {},
      { id: `${name}.weighted` },
    ).out;
    return this.addOp(
      "ReduceSum",
      { input: weighted },
      { out: { name: `${name}.out`, shape: [batch, width] } },
      { axis: -1, keepdims: false },
      { id: name },
    ).out;
  }

  stageAdapter(name: string, spec: AdapterSpec, options: AdapterStageOptions = {}): AdapterDescription {
    return this.graph.stageAdapter(name, spec, options);
  }

  activateAdapter(name: string | null, version?: AdapterVersion): AdapterDescription | null {
    return this.graph.activateAdapter(name, version);
  }

  removeAdapter(name: string, version?: AdapterVersion): boolean {
    return this.graph.removeAdapter(name, version);
  }

  /** Build an execution selector accepted by CPU and WebGPU execute(). */
  adapterRoute(name: string | null, version?: AdapterVersion, scale = 1): AdapterRoute | null {
    if (name == null) return null;
    return { name, ...(version == null ? {} : { version }), scale };
  }

  /** Build per-request or per-batch adapter routing options. */
  adapterRouting(...routes: Array<AdapterRoute | null | readonly (AdapterRoute | null)[]>): {
    adapters: readonly (AdapterRoute | null)[];
  } {
    const list: readonly (AdapterRoute | null)[] = routes.length === 1 && Array.isArray(routes[0])
      ? routes[0]
      : routes as Array<AdapterRoute | null>;
    return { adapters: list };
  }

  insertNodeAt(index: number, spec: ConcreteNodeSpec): ConcreteNode {
    return this.graph.insertNodeAt(index, spec);
  }

  insertNode(
    reference: string | number | ConcreteNode,
    spec: ConcreteNodeSpec,
    { position = "before" }: { position?: 'before' | 'after' | string } = {},
  ): ConcreteNode {
    if (position === "before") return this.graph.insertNodeBefore(reference, spec);
    if (position === "after") return this.graph.insertNodeAfter(reference, spec);
    throw new Error(`Unsupported insertion position '${position}'. Use 'before' or 'after'.`);
  }

  replaceNode(reference: string | number | ConcreteNode, spec: ConcreteNodePatch): ConcreteNode {
    return this.graph.replaceNodeAt(this.graph._nodeIndex(reference), spec);
  }

  patchNode(reference: string | number | ConcreteNode, patch: ConcreteNodePatch): ConcreteNode {
    return this.graph.patchNodeAt(this.graph._nodeIndex(reference), patch);
  }

  removeNode(
    reference: string | number | ConcreteNode,
    options: {
      cascade?: boolean;
      removeOutputs?: boolean;
      preserveOutputsAsInputs?: boolean;
      rewire?: Record<string, ConcreteTensorReference>;
    } = {},
  ): ConcreteNode {
    return this.graph.removeNode(reference, options);
  }

  getNode(reference: string | number | ConcreteNode): ConcreteNode {
    return this.graph.getNode(reference);
  }

  outputs(...references: Array<ConcreteTensorReference | readonly ConcreteTensorReference[]>): this {
    this.graph.setOutputs(...references);
    return this;
  }

  autoOutputs(): this {
    this.graph.inferOutputs();
    return this;
  }

  validate(options: { throwOnError?: boolean } = {}): GraphValidationReport {
    return this.graph.validate(options);
  }

  build({ validate = true }: { validate?: boolean } = {}): Graph {
    if (validate) this.graph.assertValid();
    return this.graph;
  }

  /** Export the graph as a Volvox blueprint config (weights remain external). */
  toConfig({ includeOutputs = true }: { includeOutputs?: boolean } = {}): BlueprintConfig {
    const inputs: BlueprintConfig['inputs'] = {};
    for (const tensor of this.graph.tensors.values()) {
      if (tensor.isInput) {
        inputs[tensor.name] = {
          shape: [...tensor.shape],
          dtype: tensor.dtype,
          ...(tensor.quantization ? { quantization: serializeQuantization(tensor.quantization) } : {}),
        };
      }
    }
    const nodes = this.graph.nodes.map((node) => {
      const params = { ...(node.params || {}) };
      if (["MatMul", "Linear", "Gemm"].includes(node.opType) && node.wLayout && !params.weight_layout) {
        params.weight_layout = node.wLayout === "din" ? "IN_OUT" : "OUT_IN";
      }
      if (node.opType === "Conv2D" && !params.weight_layout) {
        const inputChannels = (node.inputs.input || node.inputs.x)?.shape?.at(-1);
        params.weight_layout = params.groups === inputChannels ? "HWCM" : "HWIO";
      }
      const outputsQuantization = Object.fromEntries(
        Object.entries(node.outputs || {})
          .filter(([, tensor]) => tensor.quantization)
          .map(([key, tensor]) => [key, serializeQuantization(tensor.quantization!)]),
      );
      return {
        id: node.id,
        opType: node.opType,
        inputs: Object.fromEntries(Object.entries(node.inputs || {}).map(([key, tensor]) => [key, tensor.name])),
        outputs: Object.fromEntries(Object.entries(node.outputs || {}).map(([key, tensor]) => [key, tensor.name])),
        outputs_shape: Object.fromEntries(Object.entries(node.outputs || {}).map(([key, tensor]) => [key, [...tensor.shape]])),
        outputs_dtype: Object.fromEntries(Object.entries(node.outputs || {}).map(([key, tensor]) => [key, tensor.dtype])),
        ...(Object.keys(outputsQuantization).length ? { outputs_quantization: outputsQuantization } : {}),
        params,
      };
    });
    const weightsQuantization = Object.fromEntries(
      [...this.graph.tensors.values()]
        .filter((tensor) => tensor.isWeight && tensor.quantization)
        .map((tensor) => [tensor.name, serializeQuantization(tensor.quantization!)]),
    );
    return {
      inputs,
      nodes,
      ...(Object.keys(weightsQuantization).length ? { weights_quantization: weightsQuantization } : {}),
      ...(includeOutputs ? { outputs: [...this.graph.outputNames] } : {}),
    };
  }
}
