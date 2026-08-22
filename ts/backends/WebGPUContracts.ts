import type { Tensor } from '../core/Tensor.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import type { BackendExecutionOptions } from './BackendEngine.js';
import type { ShaderLibrary as ShaderLibraryClass } from './ShaderLibrary.js';
import type { WebGPUDeviceState } from './WebGPUDeviceState.js';
import type { RuntimeTypedArray } from '../types.js';
import type { DeviceTensorInputLease } from '../ops/deviceTensorReference.js';
import type { RowSpanGeometry } from './decodeRowSet.js';
import type { KVPagePlan } from './kvPageAddressing.js';

export type ShaderLibraryConstructor = typeof ShaderLibraryClass;

export interface WebGPULanguageFeatures {
  has(feature: string): boolean;
}

/**
 * How one operand of a compiled row pipeline is addressed each step.
 *
 * This used to be a byte stride back-derived from a position-one sample, on the
 * premise that "a row's address is a linear function of its position". A page
 * table falsifies that premise, and a second lane falsifies it again: the rows a
 * step touches are `b * S + position[b]`, which is a *set* of spans. So the
 * descriptor now declares geometry the tensor's own shape already fixes, and
 * the rows come from the shared `DecodeRowSet` resolver every runtime uses.
 *
 * Nothing here is derived from a sample execution, which is why there is no
 * position in it.
 */
export interface RowStorageDescriptor {
  /** Published by the shared row builder; see `RowSpanGeometry`. */
  readonly geometry: RowSpanGeometry;
  /** Bytes one row occupies. */
  readonly rowBytes: number;
  /** Dense lane count this pipeline was compiled for. */
  readonly lanes: number;
  /**
   * The staged bytes are each lane's causal prefix, not a whole row.
   *
   * Only a per-query keep mask of a causal attention node: the row moves with
   * the position *and* only its first `kvLength` entries are visible.
   */
  readonly queryMaskPrefix: boolean;
}

export type ExecutorTensor = Tensor & { _webgpuRow?: RowStorageDescriptor };

export type SpatialParameter = number | number[];

export interface ExecutorNodeParams {
  [name: string]: unknown;
  align_corners?: boolean | number;
  alpha?: number;
  antialias?: boolean | number;
  approximate?: string;
  axes?: number[];
  axis?: number;
  causal?: boolean;
  ceil_mode?: boolean | number;
  coordinate_transformation_mode?: string;
  d_model?: number;
  data_layout?: string;
  dilation?: SpatialParameter;
  eps?: number;
  groups?: number;
  heads?: number;
  kernel?: SpatialParameter;
  max?: number;
  min?: number;
  mode?: string;
  nearest_mode?: string;
  normalize?: boolean;
  num_experts?: number;
  num_groups?: number;
  padding?: SpatialParameter;
  pads?: number[];
  perm?: number[];
  relu?: number;
  scale?: number;
  sigmoid?: boolean;
  size?: number;
  starts?: number[];
  steps?: number[];
  stride?: SpatialParameter;
  temperature?: number;
  top_k?: number;
  value?: number;
  weight_layout?: string;
}

export interface ExecutorNode {
  id: string | number;
  opType: string;
  inputs: Record<string, ExecutorTensor>;
  outputs: Record<string, ExecutorTensor>;
  params: ExecutorNodeParams;
  wLayout?: 'din' | 'dout';
  /** Global slot ids backing the staged rows of a partially resident bank. */
  residentSlots?: readonly number[];
  /** Complete global slot extent; distinct from the staged row count. */
  residentSlotDomain?: number;
}

export type ExecutorGraph = Omit<RuntimeGraph, 'nodes'> & { nodes: ExecutorNode[] };
export type AdapterExecutionPlan = null;

/** Compiled-model-owned immutable device weight returned through a borrow-only view. */
export interface WebGPUInvariantDeviceWeight {
  readonly name: string;
  readonly buffer: GPUBuffer;
  readonly capacityBytes: number;
  readonly usage: GPUBufferUsageFlags;
}

export interface WebGPUInvariantWeightBorrow {
  /** Exact compiled-owned immutable device storage; absence is not a fallback. */
  borrowDeviceWeight(name: string): WebGPUInvariantDeviceWeight;
  /** Declared weight banks use context-private residency generations. */
  isContextPrivateWeight(name: string): boolean;
}

export interface GraphExecutorOptions {
  shaderLibrary?: ShaderLibraryConstructor | null;
  wgslLanguageFeatures?: WebGPULanguageFeatures | null;
  deviceState?: WebGPUDeviceState | null;
  /** @internal Compiled-model-owned immutable device weight lease. */
  invariantWeightBorrow?: WebGPUInvariantWeightBorrow | null;
}

export interface WebGPURebindOptions {
  /** Canonical logical public-input signature for telemetry and cache ownership. */
  readonly shapeSignature: string;
  /**
   * Exact resident slots whose bytes back each partially resident weight bank.
   * Rebind uses this identity to keep bank payload replacement transactional.
   * @internal Provider-authored from the resolved bounded-shape plan.
   */
  readonly bankResidency?: Readonly<Record<string, readonly number[]>>;
  /** @internal No-throw callback immediately before candidate publication. */
  readonly beforeCommit?: (() => void);
  /** Per-tensor aligned upper bound proved during provider compilation. */
  readonly tensorMaximumBytes?: ReadonlyMap<string, number>;
  /** Geometric capacity multiplier used only when a committed tensor grows. */
  readonly capacityGrowthFactor?: number;
}

export interface WebGPUTensorCapacityInspection {
  readonly name: string;
  readonly logicalBytes: number;
  readonly capacityBytes: number;
}

export interface WebGPUResourceInspection {
  readonly shapeSignature: string | null;
  readonly logicalActivationBytes: number;
  readonly activationCapacityBytes: number;
  readonly activationCapacityHighWaterBytes: number;
  readonly activationGrowCount: number;
  readonly specializationRebindCount: number;
  readonly specializationBufferCreateCount: number;
  readonly specializationBufferReuseCount: number;
  readonly liveSpecializationBufferCount: number;
  readonly bindGroupCreateCount: number;
  readonly bindGroupReuseCount: number;
  readonly liveBindGroupCount: number;
  /** Queue writes actually committed while specializing concrete shapes. */
  readonly specializationWriteCount: number;
  readonly specializationWriteBytes: number;
  /** Byte-identical writes elided using the context-owned exact content mirror. */
  readonly specializationWriteSkipCount: number;
  readonly specializationWriteSkipBytes: number;
  /** Exact host bytes retained to make write elision collision-free. */
  readonly specializationContentBytes: number;
  readonly pendingRetiredBufferCount: number;
  readonly selectedTactics: readonly string[];
  readonly tensorCapacities: readonly WebGPUTensorCapacityInspection[];
}

export type WebGPUExecutionInput = RuntimeTypedArray | DeviceTensorInputLease;

export interface WebGPUExecutionInputs {
  [name: string]: WebGPUExecutionInput;
}

export interface WebGPUAdapterSelector {
  name: string;
  version?: number;
  scale?: number;
}

export interface WebGPUExecutionOptions extends BackendExecutionOptions {
  adapter?: WebGPUAdapterSelector | null;
  adapters?: Array<WebGPUAdapterSelector | null>;
}

export interface DeviceFeedbackDecodeOptions {
  tokenInput?: string;
  keepInput?: string;
  output?: string;
  startPosition?: number;
  endPosition?: number;
  tokenCount?: number;
  rowsPerSubmission?: number;
}

export interface CompiledWebGPUPipeline {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroupCount: number[];
  nodeName?: string | number;
  graphNodeIndex?: number;
  paramsBuffer?: GPUBuffer;
  resources?: GPUBuffer[];
  tacticId?: string;
}

/**
 * A `[lanes, keys]` keep mask the host computes and uploads each step.
 *
 * Every other staged operand is a region of a tensor the device already holds,
 * so the step copies it buffer-to-buffer. This one is not a region of anything:
 * the current dense-row contract publishes each lane's active length *through*
 * the keep mask, so its contents depend on the step's `kvLengths` and exist
 * nowhere until the host builds them. It is therefore uploaded rather than
 * copied, and it is materialised even when the graph carries no mask at all --
 * which is where the lengths would otherwise have no way to reach the kernel.
 */
export interface IncrementalRowKeepMask {
  /** Name the pipeline binds. The graph's mask, or a synthesised name. */
  readonly name: string;
  /** The graph input whose values the mask reads, or null for length-only. */
  readonly sourceName: string | null;
  readonly sourceTensor: ExecutorTensor | null;
  readonly queryLength: number;
  readonly keyLength: number;
  readonly causal: boolean;
}

export interface IncrementalRowCandidate {
  nodeIndex: number;
  node: ExecutorNode;
  sampleNode: ExecutorNode;
  /** Dense lane count this candidate's pipeline covers. */
  lanes: number;
  scratchInputs: Set<string>;
  scratchOutputs: Set<string>;
  scratchCapacities: Map<string, number>;
  byteCopyInputs: Set<string>;
  byteCopyOutputs: Set<string>;
  invariantInputs: Set<string>;
  keepMask: IncrementalRowKeepMask | null;
}

/**
 * Packed-byte copy resources, one entry per staged row.
 *
 * One entry sufficed while a step issued a single unaligned copy per operand.
 * A batched step issues one per lane, and `queue.writeBuffer` on a shared
 * params buffer is ordered against *submission*, not against the passes encoded
 * before it -- so every lane would read whichever params were written last.
 * Giving each row its own buffer is what keeps the copies independent.
 */
export interface IncrementalRowByteCopy {
  paramsBuffers: GPUBuffer[];
  bindGroups: GPUBindGroup[];
}

export interface IncrementalRowPlan extends IncrementalRowCandidate {
  buffers: Map<string, GPUBuffer>;
  scratchByName: Map<string, GPUBuffer>;
  pipelines: CompiledWebGPUPipeline[];
  qsdpaParamsBuffer: GPUBuffer | null;
  byteCopyPipeline: GPUComputePipeline | null;
  inputByteCopies: Map<string, IncrementalRowByteCopy>;
  outputByteCopies: Map<string, IncrementalRowByteCopy>;
  /**
   * A second binding of the same attention pipeline whose K/V operands point
   * at contiguous staging buffers instead of the page pool.
   *
   * A bind group is built once and reads its operand from offset zero, so a
   * lane whose pages are scattered — or merely based somewhere other than slot
   * zero — cannot be read through the ordinary binding at all. Staging the
   * lane's active prefix into logical order and binding *that* is what makes
   * an arbitrary page table executable without changing a shader ABI five
   * backends share. Null until a paged step asks for it.
   */
  pagedStaging: Map<string, GPUBuffer> | null;
  pagedPipelines: CompiledWebGPUPipeline[] | null;
  pagedQsdpaParamsBuffer: GPUBuffer | null;
}

/** Everything a decode step needs from a caller that reaches the engine directly. */
export interface WebGPURowStepOptions {
  qsdpaControlBuffer?: GPUBuffer | null;
  /** The one-lane page table, or null. Batched pages ride on the row set. */
  kvPages?: KVPagePlan | null;
}

export interface AdapterTarget {
  weight: string;
  A: Float32Array;
  B: Float32Array;
  din: number;
  dout: number;
  rank: number;
  scale: number;
}

export interface DeviceFeedbackState {
  readonly tokenInputName: string;
  readonly keepInputName: string;
  readonly outputName: string;
  readonly sequenceLength: number;
  readonly nextPosition: number;
  readonly cacheGeneration: number;
}

export interface DeviceFeedbackDescriptor {
  tokenInputName: string;
  keepInputName: string;
  outputName: string;
  tokenInput: ExecutorTensor;
  keepInput: ExecutorTensor;
  output: ExecutorTensor;
  startPosition: number;
  endPosition: number;
  rowsPerSubmission: number;
  sequenceLength: number;
  selectedNodes: Set<number>;
}
