const DEFAULT_BACKEND_PLAN_CACHE_ENTRIES = 8;
const DEFAULT_BACKEND_PLAN_CACHE_METADATA_BYTES = 1024 * 1024;
const PLAN_CACHE_UTF8_ENCODER = new TextEncoder();

type ConcreteTrainingGraph = {
  readonly tensors: Map<string, {
    readonly name: string;
    readonly dtype: string;
    readonly shape: readonly number[];
    readonly isWeight?: boolean;
    readonly isInput?: boolean;
  }>;
  readonly nodes: readonly {
    readonly id: string | number;
    readonly opType: string;
    readonly inputs?: Record<string, { readonly name: string }>;
    readonly outputs?: Record<string, { readonly name: string }>;
    readonly params?: unknown;
    readonly wLayout?: unknown;
  }[];
  readonly outputNames?: readonly string[];
  readonly trainingShapeSignature?: string;
  readonly trainingTacticSignature?: string;
};

interface BackendPlanCacheEntry<TPlan> {
  readonly plan: TPlan;
  readonly metadataBytes: number;
}

export interface BackendPlanCacheOptions {
  readonly planCacheEntries?: number;
  readonly planCacheMetadataBytes?: number;
}

export interface BackendPlanCacheInspection {
  readonly backend: 'wasm' | 'webgpu';
  readonly entryLimit: number;
  readonly metadataLimitBytes: number;
  readonly entries: number;
  readonly metadataBytes: number;
  readonly hits: number;
  readonly misses: number;
  readonly evictions: number;
  readonly oversizeSkips: number;
  readonly recipeBuilds: number;
  readonly forwardPlanBuilds: number;
  readonly backwardPlanBuilds: number;
  readonly materializations: number;
  readonly forwardMaterializations: number;
  readonly backwardMaterializations: number;
  readonly keys: readonly string[];
}

function positiveSafeInteger(value: unknown, fallback: number, label: string): number {
  const selected = value === undefined ? fallback : value;
  if (!Number.isSafeInteger(selected) || (selected as number) <= 0) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return selected as number;
}

/** Exact cache key. The owning backend invalidates the cache on topology changes. */
export function backendPlanCacheKey(shapeSignature: unknown, tacticSignature: unknown): string {
  if (typeof shapeSignature !== 'string' || shapeSignature.length === 0 ||
      typeof tacticSignature !== 'string' || tacticSignature.length === 0) {
    throw new Error('Backend training plans require non-empty shape and tactic signatures.');
  }
  return `${shapeSignature.length}:${shapeSignature}|${tacticSignature.length}:${tacticSignature}`;
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, entry]) => [name, stableValue(entry)]));
  }
  return value;
}

function namedPorts(ports: Record<string, { readonly name: string }> | undefined): unknown {
  return Object.fromEntries(Object.entries(ports || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, tensor]) => [name, tensor.name]));
}

/** Shape-independent owner identity used to invalidate a context on topology change. */
export function trainingGraphTopologyIdentity(graph: ConcreteTrainingGraph): string {
  return JSON.stringify({
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      opType: node.opType,
      inputs: namedPorts(node.inputs),
      outputs: namedPorts(node.outputs),
      params: stableValue(node.params || {}),
      wLayout: node.wLayout ?? null,
    })),
    tensors: [...graph.tensors.values()]
      .sort((left, right) => left.name.localeCompare(right.name))
      .map((tensor) => ({
        name: tensor.name,
        dtype: tensor.dtype,
        rank: tensor.shape.length,
        kind: tensor.isWeight ? 'weight' : tensor.isInput ? 'input' : 'activation',
        ...(tensor.isWeight ? { shape: tensor.shape } : {}),
      })),
    outputs: [...(graph.outputNames || [])],
  });
}

/** Canonical fallback for direct backend tests that do not use TrainerCore. */
export function concreteTrainingSignatures(
  graph: ConcreteTrainingGraph,
  binding: { readonly shapeSignature?: string; readonly tacticSignature?: string } = {},
): Readonly<{ shapeSignature: string; tacticSignature: string }> {
  const tensors = [...graph.tensors.values()]
    .sort((left, right) => left.name.localeCompare(right.name))
    .map((tensor) => ({ name: tensor.name, dtype: tensor.dtype, shape: tensor.shape }));
  const shapeSignature = binding.shapeSignature || graph.trainingShapeSignature ||
    `volvox-training-concrete-shape/v1|${JSON.stringify(tensors)}`;
  const tacticSignature = binding.tacticSignature || graph.trainingTacticSignature ||
    `volvox-training-concrete-tactic/v1|${trainingGraphTopologyIdentity(graph)}|${JSON.stringify(tensors)}`;
  return Object.freeze({ shapeSignature, tacticSignature });
}

/**
 * Context-private metadata-only LRU. Callers build detached immutable recipes,
 * materialize current resource bindings separately, and publish only after the
 * complete backend rebind succeeds.
 */
export class BackendPlanCache<TPlan> {
  readonly backend: 'wasm' | 'webgpu';
  readonly #entryLimit: number;
  readonly #metadataLimit: number;
  #entries = new Map<string, BackendPlanCacheEntry<TPlan>>();
  #metadataBytes = 0;
  #hits = 0;
  #misses = 0;
  #evictions = 0;
  #oversizeSkips = 0;
  #recipeBuilds = 0;
  #forwardPlanBuilds = 0;
  #backwardPlanBuilds = 0;
  #materializations = 0;
  #forwardMaterializations = 0;
  #backwardMaterializations = 0;

  constructor(
    backend: 'wasm' | 'webgpu',
    options: BackendPlanCacheOptions = {},
  ) {
    this.backend = backend;
    this.#entryLimit = positiveSafeInteger(
      options.planCacheEntries,
      DEFAULT_BACKEND_PLAN_CACHE_ENTRIES,
      `${backend.toUpperCase()} backend planCacheEntries`,
    );
    this.#metadataLimit = positiveSafeInteger(
      options.planCacheMetadataBytes,
      DEFAULT_BACKEND_PLAN_CACHE_METADATA_BYTES,
      `${backend.toUpperCase()} backend planCacheMetadataBytes`,
    );
  }

  /** Read without mutating LRU order or telemetry; commit after materialization. */
  peek(key: string): TPlan | null {
    return this.#entries.get(key)?.plan ?? null;
  }

  recordHit(key: string): void {
    const entry = this.#entries.get(key);
    if (!entry) throw new Error(`${this.backend.toUpperCase()} backend plan cache hit disappeared.`);
    this.#entries.delete(key);
    this.#entries.set(key, entry);
    this.#hits++;
  }

  recordMiss(): void {
    this.#misses++;
  }

  recordBuild({ forward = true, backward = true } = {}): void {
    this.#recipeBuilds++;
    if (forward) this.#forwardPlanBuilds++;
    if (backward) this.#backwardPlanBuilds++;
  }

  recordMaterialization({ forward = true, backward = true } = {}): void {
    this.#materializations++;
    if (forward) this.#forwardMaterializations++;
    if (backward) this.#backwardMaterializations++;
  }

  /** Publish a fully built recipe. Oversized recipes execute uncached. */
  publish(key: string, plan: TPlan, metadataBytes: number): boolean {
    if (!Number.isSafeInteger(metadataBytes) || metadataBytes <= 0) {
      throw new Error(`${this.backend.toUpperCase()} backend plan metadata size is invalid.`);
    }
    const keyBytes = PLAN_CACHE_UTF8_ENCODER.encode(key).byteLength;
    const retainedBytes = metadataBytes + keyBytes;
    if (!Number.isSafeInteger(retainedBytes) || retainedBytes > this.#metadataLimit) {
      this.#oversizeSkips++;
      return false;
    }

    const prior = this.#entries.get(key);
    if (prior) {
      this.#entries.delete(key);
      this.#metadataBytes -= prior.metadataBytes;
    }
    while (this.#entries.size >= this.#entryLimit ||
           this.#metadataBytes + retainedBytes > this.#metadataLimit) {
      const oldest = this.#entries.entries().next().value as
        [string, BackendPlanCacheEntry<TPlan>] | undefined;
      if (!oldest) break;
      this.#entries.delete(oldest[0]);
      this.#metadataBytes -= oldest[1].metadataBytes;
      this.#evictions++;
    }
    this.#entries.set(key, Object.freeze({ plan, metadataBytes: retainedBytes }));
    this.#metadataBytes += retainedBytes;
    return true;
  }

  /** Invalidate recipes when a driver is rebound to another logical topology. */
  invalidateTopology(): void {
    this.#evictions += this.#entries.size;
    this.#entries.clear();
    this.#metadataBytes = 0;
  }

  clear(): void {
    this.#entries.clear();
    this.#metadataBytes = 0;
  }

  inspect(): Readonly<BackendPlanCacheInspection> {
    return Object.freeze({
      backend: this.backend,
      entryLimit: this.#entryLimit,
      metadataLimitBytes: this.#metadataLimit,
      entries: this.#entries.size,
      metadataBytes: this.#metadataBytes,
      hits: this.#hits,
      misses: this.#misses,
      evictions: this.#evictions,
      oversizeSkips: this.#oversizeSkips,
      recipeBuilds: this.#recipeBuilds,
      forwardPlanBuilds: this.#forwardPlanBuilds,
      backwardPlanBuilds: this.#backwardPlanBuilds,
      materializations: this.#materializations,
      forwardMaterializations: this.#forwardMaterializations,
      backwardMaterializations: this.#backwardMaterializations,
      keys: Object.freeze([...this.#entries.keys()]),
    });
  }
}
