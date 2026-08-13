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
  /**
   * Dense slot capacity this context owns. Defaults to one.
   *
   * Declared, never inferred. `batchSize` used to read the leading extent of
   * whichever operands happened to carry one, which makes the lane count a
   * property of the sample binding rather than of the context -- so a
   * two-slot scheduler handed `[1,S,D]` activations would decode one lane and
   * report success. The lane count is owned by whoever owns the slots, and the
   * geometry check below verifies the graph agrees rather than asking it.
   */
  readonly lanes?: number;
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
  /**
   * Whether this provider can execute a row step covering more than one lane.
   *
   * Defaults to false, and the default is the safe direction. A provider that
   * ignores the lane list does not produce wrong answers -- it falls back to
   * recomputing the whole prefix -- but it *reports* a row step it did not
   * execute, which is the "rejected in the browser, accepted on the robot"
   * divergence this contract exists to prevent. So a context that declares more
   * than one lane refuses at creation on a provider that cannot serve it,
   * rather than discovering it a step later or never.
   */
  readonly batchedRows?: boolean;
}

interface DecodeGeometry {
  /** The declared lane count, echoed here so a reset can compare against it. */
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
  /**
   * `I32[B]` active key/value length per lane.
   *
   * Per lane because independently scheduled sequences can disagree about it,
   * and a value rather than a shape because the alternative is a ragged tensor.
   * The one-lane case is this array with one entry.
   */
  readonly activeSequenceLengths: Int32Array;
  readonly cacheGeneration: number;
}

/** One lane's step. The scalar `position` spelling is the one-lane list. */
export interface ProviderDecodeLaneStep {
  readonly position: number;
}

export interface PreparedProviderDecode {
  readonly operation: 'seed' | 'step';
  readonly requestSignature: string;
  readonly engineOptions: BackendExecutionOptions;
  readonly mode: 'incremental-seed' | 'incremental-row' | 'incremental-dependency';
  /** Lane zero's row, retained for the one-lane telemetry contract. */
  readonly position: number | null;
  /** Every lane's row this step, or null when the step is not a row step. */
  readonly positions: readonly number[] | null;
  readonly nextActiveSequenceLengths: Int32Array;
  readonly automaticResetReason: ProviderDecodeResetReason | null;
  readonly geometry: DecodeGeometry;
  readonly semanticSeedSignature: string;
}

export interface ProviderDecodeTelemetry {
  readonly mode: PreparedProviderDecode['mode'];
  readonly cacheGeneration: number;
  readonly position: number | null;
  readonly lanes: number;
  readonly activeSequenceLength: number;
  /** Every lane's active length. `activeSequenceLength` is lane zero's. */
  readonly activeSequenceLengths: readonly number[];
  readonly kvCapacity: number;
  readonly kvCapacityClass: number;
  readonly semanticSeedSignature: string;
  readonly automaticReset: boolean;
  readonly automaticResetReason: ProviderDecodeResetReason | null;
  readonly automaticResetCount: number;
}

const ROW_MODES = new Set<DecodeRowModeValue>(['disabled', 'auto', 'required']);

/**
 * The `positions` entry for a lane that holds no request.
 *
 * A negative row cannot be a real position, which is what makes it usable as
 * the sentinel; `null` is already taken by the *idle* lane, and the two mean
 * different things -- see `prepareStep`.
 */
const PARKED = -1;

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

/**
 * Confirm every batched operand agrees with the declared lane count.
 *
 * This replaces the inference that used to live here. The difference is not
 * stylistic: an inferred lane count is whatever the sample binding happened to
 * carry, so a two-slot context handed one-lane activations would decode lane
 * zero and report success, with every downstream shape still agreeing. The
 * batched row-decode state contract requires exact B equality across decoder
 * tokens, masks, memory, KV state and outputs, and that is a statement you can
 * only check against a number somebody declared.
 */
function assertDeclaredBatch(
  snapshot: Model,
  request: BackendResolvedExecutionRequest,
  backend: string,
  lanes: number,
): number {
  const disagreeing = new Set<number>();
  const note = (extent: number) => {
    /* One is the batch-broadcast extent every spelling allows, so it is not a
     * disagreement -- an operand that carries no lane of its own is shared. */
    if (extent !== 1 && extent !== lanes) disagreeing.add(extent);
  };
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
      note(concrete.shape[0]);
    }
  }
  for (const node of request.plan.nodes) {
    if (!['SDPA', 'QSDPA', 'CrossSDPA', 'CrossAttention'].includes(node.opType)) continue;
    for (const tensor of Object.values(node.inputs)) {
      if (tensor.shape.length >= 3) note(tensor.shape[0]);
    }
  }
  if (disagreeing.size > 0) {
    fail(backend, 'BACKEND_UNSUPPORTED',
      `Dynamic decode declares ${lanes} lane(s) but this graph binds batch ` +
      `${[...disagreeing].sort((a, b) => a - b).join('/')}.`);
  }
  return lanes;
}

function sequenceGeometry(
  snapshot: Model,
  request: BackendResolvedExecutionRequest,
  backend: string,
  lanes: number,
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
    /* With more than one lane the spelling is batch-major and the leading extent
     * is the declared capacity, so the token axis is unambiguously the next one.
     * Falling through to the one-lane rules instead would read a `[2,S,D]`
     * activation as having no token axis at all and report a capacity of one --
     * which is how a staggered two-lane seed was refused for being "outside
     * [0, 0]". */
    if (lanes > 1) {
      if (shape.length >= 2 && shape[0] === lanes) return 1;
      return -1;
    }
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
  lanes: number,
): DecodeGeometry {
  const batch = assertDeclaredBatch(snapshot, request, backend, lanes);
  const sequence = sequenceGeometry(snapshot, request, backend, lanes);
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

function semanticSignature(
  geometry: DecodeGeometry, activeSequenceLengths: Int32Array,
): string {
  return JSON.stringify({
    version: 'volvox-decode-seed/v1',
    batch: geometry.batch,
    layout: geometry.layout,
    fixedFeatureHeadGeometry: geometry.fixedFeatureHeadGeometry,
    promptGeometry: geometry.promptGeometry,
    promptActiveLength: [...activeSequenceLengths],
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
  readonly #lanes: number;
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
    const lanes = options?.lanes ?? 1;
    if (!Number.isSafeInteger(lanes) || lanes < 1) {
      fail(backend, 'INVALID_ARGUMENT',
        `Decode lane capacity must be a positive integer; received ${String(lanes)}.`);
    }
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
          /* The declared capacity, so the proof covers the lane count the
           * context will actually bind rather than assuming one. */
          lanes,
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
    /* The B>1 capability proof: a legal B binding with no deterministic
     * route fails at context creation, not at the step that needs it. */
    if (lanes > 1 && rowMode !== 'disabled' && capabilities.batchedRows !== true) {
      fail(backend, 'BACKEND_UNSUPPORTED',
        `Backend '${backend}' cannot execute a decode step covering ${lanes} lanes.`);
    }
    this.#snapshot = snapshot;
    this.#backend = backend;
    this.#capabilities = resolvedCapabilities;
    this.#rowMode = rowMode;
    this.#lanes = lanes;
    this.#changedInputs = options?.changedInputs == null
      ? null
      : Object.freeze([...options.changedInputs]);
  }

  get seeded(): boolean { return this.#seed !== null; }

  /** The declared dense slot capacity of this context. */
  get lanes(): number { return this.#lanes; }

  prepareSeed(request: BackendResolvedExecutionRequest): PreparedProviderDecode {
    if (request.operation !== 'seed') {
      fail(this.#backend, 'INVALID_ARGUMENT', 'Decode seed received a non-seed request.');
    }
    if (!this.#capabilities.incrementalExecution) {
      fail(this.#backend, 'BACKEND_UNSUPPORTED',
        `Backend '${this.#backend}' does not support incremental execution.`);
    }
    const geometry = decodeGeometry(this.#snapshot, request, this.#backend, this.#lanes);
    /* A seed prompt may be staggered exactly as a step may be: two requests
     * admitted into one context rarely arrive with the same prompt length, and
     * refusing that would make continuous batching wait for equal prompts. */
    const positions = this.#seedPositions(request, geometry);
    const activeSequenceLengths = Int32Array.from(positions, (position) => position + 1);
    const signature = semanticSignature(geometry, activeSequenceLengths);
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
      positions: null,
      nextActiveSequenceLengths: activeSequenceLengths,
      automaticResetReason: this.#seed ? resetReason(this.#seed, geometry) : null,
      geometry,
      semanticSeedSignature: signature,
    });
  }

  /** Per-lane seed positions, defaulting every lane to the scalar spelling. */
  #seedPositions(
    request: BackendResolvedExecutionRequest, geometry: DecodeGeometry,
  ): number[] {
    const declared = request.options.positions;
    if (declared !== undefined) {
      if (!Array.isArray(declared) || declared.length !== this.#lanes) {
        fail(this.#backend, 'INVALID_ARGUMENT',
          `Decode seed needs one position per declared lane (${this.#lanes}).`);
      }
      if (request.options.position !== undefined) {
        fail(this.#backend, 'INVALID_ARGUMENT',
          'Decode accepts a scalar position or per-lane positions, not both.');
      }
    } else if (this.#lanes !== 1 && request.options.position !== undefined) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        `Decode declares ${this.#lanes} lanes, so it needs per-lane positions.`);
    }
    const scalar = request.options.position ?? 0;
    const positions = declared === undefined
      ? new Array(this.#lanes).fill(scalar)
      : [...declared];
    for (const [lane, position] of positions.entries()) {
      if (!Number.isSafeInteger(position) || position < 0 ||
          position >= geometry.sequenceCapacity) {
        fail(this.#backend, 'INVALID_ARGUMENT',
          `Decode seed position for lane ${lane} must be in ` +
          `[0, ${geometry.sequenceCapacity - 1}].`);
      }
    }
    return positions;
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
    const geometry = decodeGeometry(this.#snapshot, request, this.#backend, this.#lanes);
    if (request.signature !== seed.requestSignature ||
        geometry.layout !== seed.layout ||
        geometry.fixedFeatureHeadGeometry !== seed.fixedFeatureHeadGeometry ||
        geometry.promptGeometry !== seed.promptGeometry ||
        geometry.kvCapacityClass !== seed.kvCapacityClass) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        'Decode step geometry is incompatible with the active seed; reset and seed the new shape first.');
    }
    /* One row per lane. The scalar spelling is the one-lane list, so there is
     * one representation of "which rows does this step write" and a one-lane
     * step cannot drift from a batched one. */
    const declaredPositions = request.options.positions;
    if (declaredPositions !== undefined && request.options.position !== undefined) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        'Decode accepts a scalar position or per-lane positions, not both.');
    }
    if (declaredPositions !== undefined &&
        (!Array.isArray(declaredPositions) || declaredPositions.length !== this.#lanes)) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        `Decode step needs one position per declared lane (${this.#lanes}).`);
    }
    if (declaredPositions === undefined && this.#lanes !== 1 &&
        request.options.position !== undefined) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        `Decode declares ${this.#lanes} lanes, so it needs per-lane positions.`);
    }
    /* Two ways for a lane not to advance, and they are not the same thing.
     *
     * `null` is *idle*: the lane still holds its request, so it repeats its own
     * last row. That is byte-for-byte what is already there -- the recompute
     * reads the same unchanged inputs and writes the same bytes to the same
     * slot -- so no other lane can observe it even when that row sits in a page
     * shared with one. The identity is asserted rather than assumed.
     *
     * `-1` is *parked*: the lane holds no request. It has no row to repeat and,
     * once its pages went back to the pool, nowhere to write one. So it
     * occupies a dense row for shape and its output is discarded.
     *
     * Neither may be dropped from the batch: the activations are `[B,S,D]` and
     * the row set carries the declared capacity, so removing a lane changes
     * every operand's shape and forces the reseed continuous batching exists to
     * avoid. */
    const requestedLanes: readonly (number | null)[] | undefined =
      declaredPositions !== undefined
        ? [...declaredPositions]
        : request.options.position === undefined
          ? undefined
          : [request.options.position];
    const anyAdvance = requestedLanes === undefined
      ? false
      : requestedLanes.some((position) => position !== null && position !== PARKED);
    if (requestedLanes !== undefined && !anyAdvance) {
      fail(this.#backend, 'INVALID_ARGUMENT',
        'Decode step needs at least one advancing lane; every lane was idle.');
    }
    const requestedPosition = requestedLanes === undefined
      ? undefined
      : requestedLanes[0];
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
    if (requestedLanes !== undefined) {
      for (const [lane, position] of requestedLanes.entries()) {
        if (position === null) {
          /* An idle lane needs a row to occupy, and its own last one is the
           * only row whose recompute changes nothing. A lane that never
           * advanced has no such row, so it cannot be idle -- it parks. */
          if (seed.activeSequenceLengths[lane] < 1) {
            fail(this.#backend, 'INVALID_ARGUMENT',
              `Decode lane ${lane} has produced no row and cannot idle; park it instead.`);
          }
          continue;
        }
        if (position === PARKED) continue;
        /* Each lane's next position is its own. Comparing every lane against
         * one scalar is what made staggered lengths impossible; the batched
         * row-decode state contract therefore checks each lane independently. */
        if (!Number.isSafeInteger(position) || position < 1 ||
            position !== seed.activeSequenceLengths[lane] ||
            position >= geometry.sequenceCapacity) {
          fail(this.#backend, 'INVALID_ARGUMENT',
            `Decode position for lane ${lane} must equal its next active position ` +
            `${seed.activeSequenceLengths[lane]} and remain below capacity ` +
            `${geometry.sequenceCapacity}.`);
        }
      }
    }
    if (this.#rowMode === 'required' && !attestedChangedInputs) {
      fail(this.#backend, 'BACKEND_UNSUPPORTED',
        'Decode changedInputs are outside the context-attested fixed-row dependency domain.');
    }
    const useRow = requestedLanes !== undefined && this.#rowMode !== 'disabled' &&
      this.#capabilities.incrementalRows && attestedChangedInputs;
    /* The engine takes concrete rows: an idle lane resolves to its own last
     * row, so the row set stays dense and the executor needs no idea that a
     * lane finished. */
    const lanePlan = useRow
      ? requestedLanes!.map((position, lane) => position === PARKED
        ? { parked: true }
        : {
          position: position === null
            ? seed.activeSequenceLengths[lane] - 1
            : position,
        })
      : null;
    const positions = lanePlan === null
      ? null
      : lanePlan.map((entry, lane) => 'position' in entry
        ? (entry as { position: number }).position
        : seed.activeSequenceLengths[lane]);
    const nextActive = requestedLanes === undefined
      ? Int32Array.from(seed.activeSequenceLengths, (length) =>
          Math.min(geometry.sequenceCapacity, length + 1))
      /* An idle lane's active length does not move. That is the whole state a
       * finished lane carries, and keeping it here rather than in the executor
       * is why the executor has one shape of step instead of two. */
      /* An idle or parked lane's active length does not move. That is the whole
       * state a finished or empty lane carries, and keeping it here rather than
       * in the executor is why the executor has one shape of step, not three. */
      : Int32Array.from(requestedLanes, (position, lane) =>
          position === null || position === PARKED
            ? seed.activeSequenceLengths[lane]
            : position + 1);
    const engineOptions: BackendExecutionOptions = {
      incremental: true,
      changedInputs: [...changedInputs],
    };
    if (lanePlan !== null) {
      /* One lane keeps the scalar option the engines have always taken; more
       * than one publishes the lane list. Both build the same row set inside
       * the engine, so this is a spelling and not a second contract. */
      if (lanePlan.length === 1 && 'position' in lanePlan[0]) {
        engineOptions.incrementalRowPosition = (lanePlan[0] as { position: number }).position;
      } else {
        engineOptions.incrementalRowLanes = lanePlan;
      }
    }
    return Object.freeze({
      operation: 'step',
      requestSignature: request.signature,
      engineOptions,
      mode: positions === null ? 'incremental-dependency' : 'incremental-row',
      position: positions === null ? null : positions[0],
      positions,
      nextActiveSequenceLengths: nextActive,
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
        activeSequenceLengths: prepared.nextActiveSequenceLengths,
        cacheGeneration: this.#generation,
      });
    } else {
      if (!this.#seed) {
        fail(this.#backend, 'INVALID_ARGUMENT', 'Decode state was reset before step commit.');
      }
      this.#seed = Object.freeze({
        ...this.#seed,
        activeSequenceLengths: prepared.nextActiveSequenceLengths,
      });
    }
    const seed = this.#seed!;
    return Object.freeze({
      mode: prepared.mode,
      cacheGeneration: seed.cacheGeneration,
      position: prepared.position,
      lanes: this.#lanes,
      activeSequenceLength: seed.activeSequenceLengths[0],
      activeSequenceLengths: [...seed.activeSequenceLengths],
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
