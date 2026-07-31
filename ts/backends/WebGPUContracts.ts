import type { Tensor } from '../core/Tensor.js';
import type { Graph } from '../core/Graph.js';
import type { BackendExecutionOptions } from './BackendEngine.js';
import type { ShaderLibrary as ShaderLibraryClass } from './ShaderLibrary.js';
import type { WebGPUDeviceState } from './WebGPUDeviceState.js';
import type { RuntimeTypedArray } from '../types.js';

export type ShaderLibraryConstructor = typeof ShaderLibraryClass;

export interface WebGPULanguageFeatures {
  has(feature: string): boolean;
}

export interface RowStorageDescriptor {
  readonly offsetStride: number;
  readonly constantSize: number;
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
}

export type ExecutorGraph = Omit<Graph, 'nodes'> & { nodes: ExecutorNode[] };
export type AdapterExecutionPlan = ReturnType<Graph['adapters']['_pinExecution']>;

export interface GraphExecutorOptions {
  shaderLibrary?: ShaderLibraryConstructor | null;
  wgslLanguageFeatures?: WebGPULanguageFeatures | null;
  deviceState?: WebGPUDeviceState | null;
}

export interface WebGPUExecutionInputs {
  [name: string]: RuntimeTypedArray;
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
}

export interface IncrementalRowCandidate {
  nodeIndex: number;
  node: ExecutorNode;
  sampleNode: ExecutorNode;
  scratchInputs: Set<string>;
  scratchOutputs: Set<string>;
  scratchCapacities: Map<string, number>;
  byteCopyInputs: Set<string>;
  byteCopyOutputs: Set<string>;
  invariantInputs: Set<string>;
}

export interface IncrementalRowByteCopy {
  paramsBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
}

export interface IncrementalRowPlan extends IncrementalRowCandidate {
  buffers: Map<string, GPUBuffer>;
  scratchByName: Map<string, GPUBuffer>;
  pipelines: CompiledWebGPUPipeline[];
  qsdpaParamsBuffer: GPUBuffer | null;
  byteCopyPipeline: GPUComputePipeline | null;
  inputByteCopies: Map<string, IncrementalRowByteCopy>;
  outputByteCopies: Map<string, IncrementalRowByteCopy>;
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
