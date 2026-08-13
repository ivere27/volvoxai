/** Public inference contracts shared by the TypeScript core and facade. */

import type {
  MemoryLocationValue,
  RuntimeDType as ProtoRuntimeDType,
} from './generated/volvoxaiEnums.js';
import type { ShapedExecutionTensorView } from './ops/shapeSystem.js';

export type RuntimeDType = ProtoRuntimeDType;

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

export interface RuntimeTensorDescriptor {
  name?: string;
  shape: readonly number[];
  dtype?: RuntimeDType;
  buffer?: TensorStorage;
  quantization?: TensorQuantizationInput | TensorQuantization | null;
}

export type NodeOutputSpec<TTensor extends TensorLike = TensorLike> =
  | readonly number[]
  | TensorReference<TTensor>
  | RuntimeTensorDescriptor;

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
  opType: string;
  inputs: Record<string, TensorReference<TTensor>>;
  outputs: Record<string, NodeOutputSpec<TTensor>>;
  params?: NodeParameters;
  wLayout?: 'din' | 'dout';
  [name: string]: unknown;
}

export interface GraphNodePatch<TTensor extends TensorLike = TensorLike> {
  id?: string | number;
  opType?: string;
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
}

export type SerializedAffineQuantizationReference =
  | {
      scheme: 'per_tensor';
      scale_tensor: string;
      zero_point_tensor: string;
    }
  | {
      scheme: 'per_axis';
      axis: number;
      scale_tensor: string;
      zero_point_tensor: string;
    };

export interface GraphDocument {
  format: 'volvox-graph/v1';
  inputs: Record<string, {
    shape: number[];
    dtype: RuntimeDType;
  }>;
  nodes: Array<{
    id?: string | number;
    opType: string;
    inputs: Record<string, string>;
    outputs: Record<string, string>;
    outputs_shape: Record<string, number[]>;
    outputs_dtype: Record<string, RuntimeDType>;
    params?: NodeParameters;
  }>;
  quantization?: {
    format: 'volvox-affine-safetensors/v1';
    tensors: Record<string, SerializedAffineQuantizationReference>;
  };
  outputs: string[];
}

/** Canonical execution-time adapter selector. */
export interface AdapterSelector {
  name: string;
  version?: number;
  scale?: number;
}

export interface BackendCapabilityContract {
  readonly incrementalExecution: boolean;
  readonly incrementalRows: boolean;
  readonly outputLocation: MemoryLocationValue;
}

/**
 * Dynamic-v1 execution never infers logical shape from storage length. Every
 * public input, including inputs to an otherwise constant graph, carries an
 * explicit concrete shape alongside its typed storage.
 */
export type ExecutionInputs = Readonly<Record<string, ShapedExecutionTensorView>>;
export interface ExecutionOptions {
  readonly adapter?: Readonly<AdapterSelector> | null;
  readonly adapters?: readonly (Readonly<AdapterSelector> | null)[];
}

export interface DecodeExecutionOptions extends ExecutionOptions {
  readonly changedInputs?: readonly string[];
  readonly position?: number;
  /**
   * One row position per declared decode lane, or `null` for a lane that
   * produces no new token this step.
   *
   * The batched spelling of `position`, for a context that owns more than one
   * slot. Lanes are independent: a lane's position is its own next token, so
   * two requests admitted a step apart stay a step apart instead of one of them
   * padding to the other's length. A context with one lane uses `position`;
   * declaring both is refused rather than resolved by precedence, because the
   * two would disagree about how many slots exist.
   *
   * A `[B,S,D]` batch cannot drop a lane without changing every operand's
   * shape, so a lane that produces no token this step still occupies one. There
   * are two ways to say that and they differ:
   *
   *   `null`  idles a lane that still holds its request. Its active length stops
   *           moving and its row is left exactly as it was.
   *   `-1`    parks a lane that holds no request. It has no row to preserve, so
   *           it occupies a row for shape and its output is discarded -- which
   *           is what lets a slot whose request retired, and whose pages went
   *           back to the pool, stay in the batch at all.
   *
   * At least one lane must advance, since a step in which nothing advances is
   * not a step.
   */
  readonly positions?: readonly (number | null)[];
}

export interface GraphContract {
  nodes: readonly GraphNode[];
  tensors: ReadonlyMap<string, TensorLike>;
  outputNames: readonly string[];
  topologyRevision: number;
  weightRevision: number;
}
