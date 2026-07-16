/** Public inference contracts shared by the TypeScript core and facade. */

export type RuntimeDType = 'float32' | 'int32' | 'int8' | 'uint8';

export type RuntimeTypedArray =
  | Float32Array
  | Int32Array
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray;

export type TensorStorage = ArrayBuffer | RuntimeTypedArray;
export type TensorShape = number[];

export interface PerTensorQuantization {
  readonly scheme: 'per_tensor';
  readonly scale: number;
  readonly zero_point: number;
}

export interface PerAxisQuantization {
  readonly scheme: 'per_axis';
  readonly axis: number;
  readonly scales: readonly number[];
  readonly zero_points: readonly number[];
}

export type TensorQuantization = PerTensorQuantization | PerAxisQuantization;

export interface PerTensorQuantizationInput {
  scheme?: 'per_tensor';
  scale: number;
  zero_point?: number;
}

export interface PerAxisQuantizationInput {
  scheme: 'per_axis';
  axis: number;
  scales: readonly number[];
  zero_points?: readonly number[];
}

export type TensorQuantizationInput =
  | PerTensorQuantizationInput
  | PerAxisQuantizationInput;

export interface TensorOptions {
  isInput?: boolean;
  quantization?: TensorQuantizationInput | TensorQuantization | null;
}

export interface TensorLike {
  name: string;
  shape: TensorShape;
  dtype: RuntimeDType;
  isWeight: boolean;
  isInput: boolean;
  quantization: TensorQuantization | null;
  sizeBytes: number;
  buffer?: TensorStorage;
}

export type TensorReference<TTensor extends TensorLike = TensorLike> = string | TTensor;

export interface TensorDescriptor {
  name?: string;
  shape: readonly number[];
  dtype?: RuntimeDType;
  buffer?: TensorStorage;
  data?: TensorStorage;
  quantization?: TensorQuantizationInput | TensorQuantization | null;
  replace?: boolean;
  reuse?: boolean;
}

export type NodeOutputSpec<TTensor extends TensorLike = TensorLike> =
  | readonly number[]
  | TensorReference<TTensor>
  | TensorDescriptor;

export interface NodeParameters {
  weight_layout?: string;
  transB?: boolean;
  [name: string]: unknown;
}

export interface GraphNode<TTensor extends TensorLike = TensorLike> {
  id: string | number;
  opType: string;
  inputs: Record<string, TTensor>;
  outputs: Record<string, TTensor>;
  params: NodeParameters;
  wLayout?: 'din' | 'dout';
  [name: string]: unknown;
}

export interface GraphNodeSpec<TTensor extends TensorLike = TensorLike> {
  id?: string | number;
  opType?: string;
  op?: string;
  inputs: Record<string, TensorReference<TTensor>>;
  outputs: Record<string, NodeOutputSpec<TTensor>>;
  params?: NodeParameters;
  wLayout?: 'din' | 'dout';
  [name: string]: unknown;
}

export interface GraphNodePatch<TTensor extends TensorLike = TensorLike> {
  id?: string | number;
  opType?: string;
  op?: string;
  inputs?: Record<string, TensorReference<TTensor>>;
  outputs?: Record<string, NodeOutputSpec<TTensor>>;
  params?: NodeParameters;
  replaceParams?: boolean;
  [name: string]: unknown;
}

export type TensorRole = 'value' | 'input' | 'weight';

export interface AddTensorOptions {
  role?: TensorRole;
  isInput?: boolean;
  isWeight?: boolean;
  buffer?: TensorStorage;
  data?: TensorStorage;
  quantization?: TensorQuantizationInput | TensorQuantization | null;
}

export interface TensorPatch extends Partial<AddTensorOptions> {
  name?: string;
  shape?: readonly number[];
  dtype?: RuntimeDType;
}

export interface GraphValidationReport {
  readonly valid: boolean;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
  readonly nodeCount: number;
  readonly tensorCount: number;
  readonly topologyRevision: number;
}

export interface GraphInspectionOptions {
  includeWeightFiles?: boolean;
  includeParams?: boolean;
}

export interface GraphInspection {
  tensorCount: number;
  nodeCount: number;
  tensors: Array<{
    name: string;
    shape: number[];
    dtype: RuntimeDType;
    isWeight: boolean;
    isInput: boolean;
    role: TensorRole;
    sizeBytes: number;
  }>;
  nodes: Array<{
    index: number;
    id: string;
    opType: string;
    inputs: Record<string, string>;
    outputs: Record<string, string>;
    outputShapes: Record<string, number[]>;
    params?: NodeParameters;
  }>;
  weightFiles: unknown[];
  outputNames: string[];
  topologyRevision: number;
  weightRevision: number;
  adapters: AdapterDescription[];
  activeAdapter: AdapterDescription | null;
}

export type SerializedQuantization =
  | {
      scheme: 'per_tensor';
      scale: number;
      zero_point: number;
    }
  | {
      scheme: 'per_axis';
      axis: number;
      scales: number[];
      zero_points: number[];
    };

export interface BlueprintConfig {
  inputs: Record<string, {
    shape: number[];
    dtype: RuntimeDType;
    quantization?: SerializedQuantization;
  }>;
  nodes: Array<{
    id: string | number;
    opType: string;
    inputs: Record<string, string>;
    outputs: Record<string, string>;
    outputs_shape: Record<string, number[]>;
    outputs_dtype: Record<string, RuntimeDType>;
    outputs_quantization?: Record<string, SerializedQuantization | null>;
    params: NodeParameters;
  }>;
  weights_quantization?: Record<string, SerializedQuantization | null>;
  outputs?: string[];
}

export interface AdapterTensorValue {
  data?: ArrayLike<number> | ArrayBuffer | ArrayBufferView;
  values?: ArrayLike<number> | ArrayBuffer | ArrayBufferView;
  buffer?: ArrayLike<number> | ArrayBuffer | ArrayBufferView;
  shape?: readonly number[];
}

export type AdapterTensorInput =
  | ArrayLike<number>
  | ArrayBuffer
  | ArrayBufferView
  | AdapterTensorValue;

export interface AdapterTargetSpec {
  weight?: string;
  baseTensor?: string;
  target?: string;
  kind?: 'lora';
  layout?: string;
  rank?: number;
  alpha?: number;
  scale?: number;
  A?: AdapterTensorInput;
  a?: AdapterTensorInput;
  B?: AdapterTensorInput;
  b?: AdapterTensorInput;
  adapterB?: AdapterTensorInput;
}

export interface AdapterSpec {
  kind?: 'lora';
  type?: 'lora';
  layout?: string;
  rank?: number;
  alpha?: number;
  scale?: number;
  activate?: boolean;
  sourceVersion?: string | number | null;
  metadata?: Record<string, string>;
  targets: readonly AdapterTargetSpec[] | Record<string, AdapterTargetSpec>;
}

export interface AdapterDescription {
  readonly id: string;
  readonly name: string;
  readonly version: number;
  readonly sourceVersion: string | number | null;
  readonly kind: 'lora';
  readonly metadata: Readonly<Record<string, string>>;
  readonly active: boolean;
  readonly merged: boolean;
  readonly targets: readonly {
    readonly weight: string;
    readonly rank: number;
    readonly din: number;
    readonly dout: number;
    readonly alpha: number;
    readonly scale: number;
    readonly layout: 'din_r_r_dout';
  }[];
}

export type AdapterVersion = number | string | null | undefined;

export interface AdapterStageOptions {
  activate?: boolean;
}

export interface AdapterUpdateOptions extends AdapterStageOptions {
  version?: AdapterVersion;
  mode?: 'assign' | 'add';
}

export interface AdapterLoadOptions extends AdapterStageOptions {
  name?: string;
}

export interface AdapterExportOptions {
  as?: 'arraybuffer' | 'file' | 'blob';
}

export interface BackendCapabilityContract {
  readonly incrementalExecution: boolean;
  readonly incrementalRows: boolean;
  readonly outputLocation: 'host' | 'device';
}

export type ExecutionInputs = Record<string, RuntimeTypedArray>;
export type ExecutionOptions = Record<string, unknown>;

export interface BackendEngineContract {
  readonly backendApiVersion: number;
  readonly backendName: string;
  readonly capabilities: BackendCapabilityContract;
  readonly decodeCacheGeneration?: number;
  readonly device?: unknown;
  allocateGraph(graph: GraphContract): Promise<unknown> | unknown;
  execute(inputs: ExecutionInputs, options?: ExecutionOptions): Promise<unknown> | unknown;
  createDecodeSession(options?: ExecutionOptions): unknown;
  resetDecodeCache?(): void;
  fork?(): Promise<BackendEngineContract> | BackendEngineContract;
  dispose?(): void;
}

export interface GraphContract {
  nodes: readonly GraphNode[];
  tensors: ReadonlyMap<string, TensorLike>;
  outputNames: readonly string[];
  topologyRevision: number;
  weightRevision: number;
}

export type BackendFactory = (context: {
  name: string;
  runtime: unknown;
  wasmUrl: string | URL;
}) => BackendEngineContract | null | Promise<BackendEngineContract | null>;

export type BackendSelection = string | readonly string[];
