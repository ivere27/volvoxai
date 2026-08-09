import type { Model } from '../core/Model.js';
import type { BackendResolvedExecutionRequest } from './BackendProvider.js';
import type { BackendExecutionOptions } from './BackendEngine.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';
import type { DecodeRowModeValue } from '../generated/volvoxaiEnums.js';
import { incrementalRowDomainSupported } from './quantizedRowExecution.js';

export type ProviderDecodeResetReason =
  | 'reseed'
  | 'batch-changed'
  | 'layout-changed'
  | 'feature-head-geometry-changed'
  | 'kv-capacity-class-changed'
  | 'prompt-geometry-changed'
  | 'ordinary-execution'
  | 'decode-execution-failed'
  | 'explicit-reset'
  | 'context-close';

export interface ProviderDecodeOptions {
  readonly changedInputs?: readonly string[] | null;
  readonly rowMode?: DecodeRowModeValue;
  readonly requireIncremental?: boolean;
}

export interface ProviderDecodeCapabilities {
  readonly incrementalExecution: boolean;
  readonly incrementalRows: boolean;
  /**
   * Whether this provider can execute one row of a sequence-major `[S,...]`
   * tensor, not only the batch-major `[B,S,...]` spelling. Defaults to false:
   * the row attestation is shared by every provider, and WebGPU has no row
   * candidate for that layout.
   */
  readonly sequenceMajorRows?: boolean;
}

interface DecodeGeometry {
  readonly batch: number;
  readonly layout: string;
  readonly fixedFeatureHeadGeometry: string;
  readonly promptGeometry: string;
  readonly sequenceCapacity: number;
  readonly sequenceMaximum: number;
  readonly kvCapacityClass: number;
}

interface DecodeSeedState extends DecodeGeometry {
  readonly requestSignature: string;
  readonly semanticSeedSignature: string;
  readonly activeSequenceLength: number;
  readonly cacheGeneration: number;
}

export interface PreparedProviderDecode {
  readonly operation: 'seed' | 'step';
  readonly requestSignature: string;
  readonly engineOptions: BackendExecutionOptions;
  readonly mode: 'incremental-seed' | 'incremental-row' | 'incremental-dependency';
  readonly position: number | null;
  readonly nextActiveSequenceLength: number;
  readonly automaticResetReason: ProviderDecodeResetReason | null;
  readonly geometry: DecodeGeometry;
  readonly semanticSeedSignature: string;
}

export interface ProviderDecodeTelemetry {
  readonly mode: PreparedProviderDecode['mode'];
  readonly cacheGeneration: number;
  readonly position: number | null;
  readonly activeSequenceLength: number;
  readonly kvCapacity: number;
  readonly kvCapacityClass: number;
  readonly semanticSeedSignature: string;
  readonly automaticReset: boolean;
  readonly automaticResetReason: ProviderDecodeResetReason | null;
  readonly automaticResetCount: number;
}

const ROW_MODES = new Set<DecodeRowModeValue>(['disabled', 'auto', 'required']);

function fail(backend: string, code: 'INVALID_ARGUMENT' | 'BACKEND_UNSUPPORTED', message: string): never {
  throw new VolvoxAIError(code, message, { phase: 'execution', backend });
}

function stableShape(shape: readonly (number | string)[]): string {
  return `[${shape.map((dimension) =>
    typeof dimension === 'number' ? `#${dimension}` : `$${dimension}`).join(',')}]`;
}

function fixedFeatureHeadGeometry(
  snapshot: Model,
  request: BackendResolvedExecutionRequest,
): string {
  const fixedInputs = snapshot.inputNames.map((name) => {
    const logical = snapshot.graph.inputs[name];
    const fixed = logical.shape.map((dimension, axis) =>
      typeof dimension === 'number' ? `${axis}:${dimension}` : null).filter(Boolean);
    return `${name}{${fixed.join(',')}}`;
  });
  const attention = request.plan.nodes
    .filter((node) => ['SDPA', 'QSDPA', 'CrossSDPA', 'CrossAttention'].includes(node.opType))
    .map((node) => {
      const q = node.inputs.q ?? node.inputs.qkv;
      const k = node.inputs.k ?? node.inputs.kv ?? node.inputs.qkv;
      const heads = node.params.heads;
      const qFeature = q?.shape.at(-1) ?? null;
      const kFeature = k?.shape.at(-1) ?? null;
      return `${node.id}:${node.opType}:h=${String(heads ?? '')}:q=${String(qFeature)}:k=${String(kFeature)}`;
    });
  return `${fixedInputs.join('|')}::${attention.join('|')}`;
}

function batchSize(
  snapshot: Model,
  request: BackendResolvedExecutionRequest,
  backend: string,
): number {
  const candidates = new Set<number>();
  for (const name of snapshot.inputNames) {
    const logical = snapshot.graph.inputs[name];
    const concrete = request.plan.tensors[name];
    if (!logical || !concrete || concrete.shape.length === 0) continue;
    const leading = logical.shape[0];
    const explicitBatchSymbol = typeof leading === 'string' && /^(?:b|batch)$/i.test(leading);
    const sequenceMajor = concrete.shape.length === 3 && concrete.shape[1] === 1 &&
      concrete.shape[0] > 1;
    if (explicitBatchSymbol || (!sequenceMajor && concrete.shape.length >= 3) ||
        (concrete.shape.length >= 2 && concrete.shape[0] === 1)) {
      candidates.add(concrete.shape[0]);
    }
  }
  for (const node of request.plan.nodes) {
    if (!['SDPA', 'QSDPA', 'CrossSDPA', 'CrossAttention'].includes(node.opType)) continue;
    for (const tensor of Object.values(node.inputs)) {
      if (tensor.shape.length >= 3) candidates.add(tensor.shape[0]);
    }
  }
  if (candidates.size === 0) return 1;
  if (candidates.size !== 1) {
    fail(backend, 'BACKEND_UNSUPPORTED',
      'Dynamic decode requires one unambiguous batch geometry.');
  }
  return candidates.values().next().value as number;
}

function sequenceGeometry(
  snapshot: Model,
  request: BackendResolvedExecutionRequest,
  backend: string,
): { readonly capacity: number; readonly maximum: number } {
  type Candidate = { readonly capacity: number; readonly maximum: number };
  const causalQueries: Candidate[] = [];
  const tokenInputs: Candidate[] = [];
  const publicOutputs: Candidate[] = [];
  const integerInputs: Candidate[] = [];
  const fallbackInputs: Candidate[] = [];
  const add = (
    name: string,
    concrete: { readonly shape: readonly number[] },
    axis: number,
    target: Candidate[],
  ) => {
    const logical = snapshot.graph.tensors[name];
    const capacity = concrete.shape[axis];
    const logicalDimension = logical?.shape[axis];
    const maximum = typeof logicalDimension === 'string'
      ? snapshot.graph.dimensions[logicalDimension]?.max
      : capacity;
    if (Number.isSafeInteger(capacity) && capacity > 0 &&
        Number.isSafeInteger(maximum) && (maximum as number) >= capacity) {
      target.push({ capacity, maximum: maximum as number });
    }
  };
  const sequenceAxis = (shape: readonly number[], allowRankTwoLeadingOne: boolean): number => {
    if (shape.length >= 3 && shape[0] === 1) return 1;
    if (shape.length === 3 && shape[0] > 1 && shape[1] === 1) return 0;
    if (shape.length === 2 && shape[0] > 1) return 0;
    if (allowRankTwoLeadingOne && shape.length === 2 && shape[0] === 1) return 1;
    if (shape.length === 1) return 0;
    return -1;
  };

  // A causal attention query is the strongest semantic source. In particular,
  // its length must win over a longer cross-attention memory mask, such as a
  // decoder capacity of 192 paired with an encoder memory length of 402.
  for (const node of request.plan.nodes) {
    if (!['SDPA', 'QSDPA', 'CrossSDPA', 'CrossAttention'].includes(node.opType) ||
        node.params.causal !== true) continue;
    const query = node.inputs.q ?? node.inputs.qkv;
    if (!query) continue;
    const axis = sequenceAxis(query.shape, false);
    if (axis >= 0) add(query.name, query, axis, causalQueries);
  }

  for (const name of snapshot.inputNames) {
    const concrete = request.plan.tensors[name];
    if (!concrete) continue;
    const axis = sequenceAxis(concrete.shape, true);
    if (axis < 0) continue;
    if (/(?:^|[._-])(?:ids|tokens)(?:$|[._-])/i.test(name)) {
      add(name, concrete, axis, tokenInputs);
    }
    if (concrete.dtype === 'int32') add(name, concrete, axis, integerInputs);
    add(name, concrete, axis, fallbackInputs);
  }
  for (const output of request.plan.outputs) {
    const axis = sequenceAxis(output.shape, true);
    if (axis >= 0) add(output.name, output, axis, publicOutputs);
  }

  const candidates = causalQueries.length ? causalQueries
    : tokenInputs.length ? tokenInputs
      : publicOutputs.length ? publicOutputs
        : integerInputs.length ? integerInputs
          : fallbackInputs;
  if (candidates.length === 0) return { capacity: 1, maximum: 1 };
  const capacity = candidates[0].capacity;
  const compatible = candidates.filter((candidate) => candidate.capacity === capacity);
  if (compatible.length !== candidates.length) {
    fail(backend, 'BACKEND_UNSUPPORTED',
      'Dynamic decode requires one unambiguous active query sequence geometry.');
  }
  // When equivalent tensor views use independently declared bounds, retain the
  // strictest maximum so the semantic capacity class is valid for all of them.
  return {
    capacity,
    maximum: compatible.reduce(
      (maximum, candidate) => Math.min(maximum, candidate.maximum),
      compatible[0].maximum,
    ),
  };
}

function capacityClass(required: number, maximum: number): number {
  let value = 1;
  while (value < required && value < maximum) {
    value = Math.min(maximum, value * 2);
  }
  return value;
}

function decodeGeometry(
  snapshot: Model,
  request: BackendResolvedExecutionRequest,
  backend: string,
): DecodeGeometry {
  const batch = batchSize(snapshot, request, backend);
  const sequence = sequenceGeometry(snapshot, request, backend);
  const layout = snapshot.inputNames.map((name) => {
    const logical = snapshot.graph.inputs[name];
    return `${name}:${logical.dtype}:r${logical.shape.length}:${stableShape(logical.shape)}`;
  }).join('|');
  const promptGeometry = snapshot.inputNames.map((name) => {
    const descriptor = request.plan.tensors[name];
    return `${name}:${descriptor.dtype}:[${descriptor.shape.join(',')}]`;
  }).join('|');
  return Object.freeze({
    batch,
    layout,
    fixedFeatureHeadGeometry: fixedFeatureHeadGeometry(snapshot, request),
    promptGeometry,
    sequenceCapacity: sequence.capacity,
    sequenceMaximum: sequence.maximum,
    kvCapacityClass: capacityClass(sequence.capacity, sequence.maximum),
  });
}

function semanticSignature(geometry: DecodeGeometry, activeSequenceLength: number): string {
  return JSON.stringify({
    version: 'volvox-decode-seed/v1',
    batch: geometry.batch,
    layout: geometry.layout,
    fixedFeatureHeadGeometry: geometry.fixedFeatureHeadGeometry,
    promptGeometry: geometry.promptGeometry,
    promptActiveLength: activeSequenceLength,
    kvCapacityClass: geometry.kvCapacityClass,
  });
}

function resetReason(
  previous: DecodeSeedState,
  geometry: DecodeGeometry,
): ProviderDecodeResetReason {
  if (previous.batch !== geometry.batch) return 'batch-changed';
  if (previous.layout !== geometry.layout) return 'layout-changed';
  if (previous.fixedFeatureHeadGeometry !== geometry.fixedFeatureHeadGeometry) {
    return 'feature-head-geometry-changed';
  }
  if (previous.kvCapacityClass !== geometry.kvCapacityClass) {
    return 'kv-capacity-class-changed';
  }
  if (previous.promptGeometry !== geometry.promptGeometry) return 'prompt-geometry-changed';
  return 'reseed';
}

/** Context-local semantic owner layered over a provider's retained engine cache. */
export class ProviderDecodeLifecycle {
  readonly #snapshot: Model;
  readonly #backend: string;
  readonly #capabilities: ProviderDecodeCapabilities;
  readonly #changedInputs: readonly string[] | null;
  readonly #rowMode: DecodeRowModeValue;
  #seed: DecodeSeedState | null = null;
  #generation = 0;
  #automaticResetCount = 0;

  constructor(
    snapshot: Model,
    backend: string,
    capabilities: ProviderDecodeCapabilities,
    options: ProviderDecodeOptions | undefined = undefined,
  ) {
    const rowMode = options?.rowMode ?? 'auto';
    const changedInputs = options?.changedInputs == null
      ? snapshot.inputNames
      : options.changedInputs;
    const resolvedCapabilities = Object.freeze({
      incrementalExecution: capabilities.incrementalExecution,
      incrementalRows: capabilities.incrementalRows &&
        incrementalRowDomainSupported(snapshot.graph, changedInputs, {
          /* Only a provider that can execute a row of a sequence-major tensor
           * may attest one. WebGPU has no row candidate for that layout, so
           * the default stays batch-major and each provider opts in. */
          sequenceMajorRows: capabilities.sequenceMajorRows === true,
        }),
    });
    /* What the provider offered, what survived attestation, and over which
     * inputs. A caller that names the wrong changed inputs and one whose graph
     * refuses both decode at full cost, and only this tells them apart --
     * `incrementalRows` short-circuits, so a provider that offers nothing
     * produces no attestation output at all. Matches the native
     * VOLVOXAI_ROW_DEBUG line so one flag reads both runtimes. */
    if ((globalThis as any)?.process?.env?.VOLVOXAI_ROW_DEBUG) {
      console.error(`[row] lifecycle ${backend} offered=${capabilities.incrementalRows}` +
        ` resolved=${resolvedCapabilities.incrementalRows} changed=[${changedInputs}]`);
    }
    if (!ROW_MODES.has(rowMode)) {
      fail(backend, 'INVALID_ARGUMENT', `Decode row mode '${String(rowMode)}' is invalid.`);
    }
    if (options?.requireIncremental === true && !resolvedCapabilities.incrementalExecution) {
      fail(backend, 'BACKEND_UNSUPPORTED',
        `Backend '${backend}' does not support incremental execution.`);
    }
    if (rowMode === 'required' && !resolvedCapabilities.incrementalRows) {
      fail(backend, 'BACKEND_UNSUPPORTED',
        `Backend '${backend}' cannot attest fixed-row execution for this graph and shape domain.`);
    }
    this.#snapshot = snapshot;
    this.#backend = backend;
    this.#capabilities = resolvedCapabilities;
    this.#rowMode = rowMode;
    this.#changedInputs = options?.changedInputs == null
      ? null
      : Object.freeze([...options.changedInputs]);
  }

  get seeded(): boolean { return this.#seed !== null; }

  prepareSeed(request: BackendResolvedExecutionRequest): PreparedProviderDecode {
    if (request.operation !== 'seed') {
      fail(this.#backend, 'INVALID_ARGUMENT', 'Decode seed received a non-seed request.');
    }
    if (!this.#capabilities.incrementalExecution) {
      fail(this.#backend, 'BACKEND_UNSUPPORTED',
        `Backend '${this.#backend}' does not support incremental execution.`);
    }
    const geometry = decodeGeometry(this.#snapshot, request, this.#backend);
    if (geometry.batch !== 1) {
      fail(this.#backend, 'BACKEND_UNSUPPORTED',
        `Dynamic decode currently requires B=1; received B=${geometry.batch}.`);
    }
    const position = request.options.position ?? 0;
    if (!Number.isSafeInteger(position) || position < 0 || position >= geometry.sequenceCapacity) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        `Decode seed position must be in [0, ${geometry.sequenceCapacity - 1}].`);
    }
    const activeSequenceLength = position + 1;
    const signature = semanticSignature(geometry, activeSequenceLength);
    return Object.freeze({
      operation: 'seed',
      requestSignature: request.signature,
      engineOptions: {
        incremental: true,
        incrementalReset: true,
        changedInputs: [...this.#snapshot.inputNames],
      },
      mode: 'incremental-seed',
      position: null,
      nextActiveSequenceLength: activeSequenceLength,
      automaticResetReason: this.#seed ? resetReason(this.#seed, geometry) : null,
      geometry,
      semanticSeedSignature: signature,
    });
  }

  prepareStep(request: BackendResolvedExecutionRequest): PreparedProviderDecode {
    if (request.operation !== 'step') {
      fail(this.#backend, 'INVALID_ARGUMENT', 'Decode step received a non-step request.');
    }
    const seed = this.#seed;
    if (!seed) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        'Decode step requires a successful seed on this execution context.');
    }
    const geometry = decodeGeometry(this.#snapshot, request, this.#backend);
    if (geometry.batch !== 1) {
      fail(this.#backend, 'BACKEND_UNSUPPORTED',
        `Dynamic decode currently requires B=1; received B=${geometry.batch}.`);
    }
    if (request.signature !== seed.requestSignature ||
        geometry.layout !== seed.layout ||
        geometry.fixedFeatureHeadGeometry !== seed.fixedFeatureHeadGeometry ||
        geometry.promptGeometry !== seed.promptGeometry ||
        geometry.kvCapacityClass !== seed.kvCapacityClass) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        'Decode step geometry is incompatible with the active seed; reset and seed the new shape first.');
    }
    const requestedPosition = request.options.position;
    const changedInputs = request.options.changedInputs ?? this.#changedInputs ??
      this.#snapshot.inputNames;
    const attestedChangedInputs = this.#changedInputs === null || (
      changedInputs.length === this.#changedInputs.length &&
      changedInputs.every((name) => this.#changedInputs!.includes(name))
    );
    if (this.#rowMode === 'required' && requestedPosition === undefined) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        "Decode rowMode 'required' needs a position on every step.");
    }
    if (this.#rowMode === 'disabled' && requestedPosition !== undefined) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        "Decode rowMode 'disabled' does not accept row positions.");
    }
    if (requestedPosition !== undefined && (!Number.isSafeInteger(requestedPosition) ||
        requestedPosition < 1 || requestedPosition !== seed.activeSequenceLength ||
        requestedPosition >= geometry.sequenceCapacity)) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        `Decode position must equal the next active position ${seed.activeSequenceLength} ` +
        `and remain below capacity ${geometry.sequenceCapacity}.`);
    }
    if (this.#rowMode === 'required' && !attestedChangedInputs) {
      fail(this.#backend, 'BACKEND_UNSUPPORTED',
        'Decode changedInputs are outside the context-attested fixed-row dependency domain.');
    }
    const useRow = requestedPosition !== undefined && this.#rowMode !== 'disabled' &&
      this.#capabilities.incrementalRows && attestedChangedInputs;
    const position = useRow ? requestedPosition as number : null;
    const nextActive = requestedPosition === undefined
      ? Math.min(geometry.sequenceCapacity, seed.activeSequenceLength + 1)
      : requestedPosition + 1;
    const engineOptions: BackendExecutionOptions = {
      incremental: true,
      changedInputs: [...changedInputs],
    };
    if (position !== null) engineOptions.incrementalRowPosition = position;
    return Object.freeze({
      operation: 'step',
      requestSignature: request.signature,
      engineOptions,
      mode: position === null ? 'incremental-dependency' : 'incremental-row',
      position,
      nextActiveSequenceLength: nextActive,
      automaticResetReason: null,
      geometry,
      semanticSeedSignature: seed.semanticSeedSignature,
    });
  }

  commit(prepared: PreparedProviderDecode): Readonly<ProviderDecodeTelemetry> {
    if (prepared.operation === 'seed') {
      this.#generation++;
      if (prepared.automaticResetReason !== null) this.#automaticResetCount++;
      this.#seed = Object.freeze({
        ...prepared.geometry,
        requestSignature: prepared.requestSignature,
        semanticSeedSignature: prepared.semanticSeedSignature,
        activeSequenceLength: prepared.nextActiveSequenceLength,
        cacheGeneration: this.#generation,
      });
    } else {
      if (!this.#seed) {
        fail(this.#backend, 'INVALID_ARGUMENT', 'Decode state was reset before step commit.');
      }
      this.#seed = Object.freeze({
        ...this.#seed,
        activeSequenceLength: prepared.nextActiveSequenceLength,
      });
    }
    const seed = this.#seed!;
    return Object.freeze({
      mode: prepared.mode,
      cacheGeneration: seed.cacheGeneration,
      position: prepared.position,
      activeSequenceLength: seed.activeSequenceLength,
      kvCapacity: seed.sequenceCapacity,
      kvCapacityClass: seed.kvCapacityClass,
      semanticSeedSignature: seed.semanticSeedSignature,
      automaticReset: prepared.automaticResetReason !== null,
      automaticResetReason: prepared.automaticResetReason,
      automaticResetCount: this.#automaticResetCount,
    });
  }

  invalidate(_reason: ProviderDecodeResetReason): void {
    if (this.#seed) this.#generation++;
    this.#seed = null;
  }

  failExecution(): void {
    this.invalidate('decode-execution-failed');
  }

  reset(): void {
    this.invalidate('explicit-reset');
  }

  close(): void {
    this.invalidate('context-close');
  }
}
