import { Graph } from './Graph.js';
import { Tensor } from './Tensor.js';
import { runtimeIdentity } from './Identity.js';
import { GraphOperatorNormalizer } from '../ops/graphOperatorNormalization.js';
import type {
  GraphNode,
  NodeParameters,
  RuntimeDType,
  RuntimeTypedArray,
  TensorQuantization,
  TensorStorage,
} from '../types.js';

interface CapturedTensor {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: RuntimeDType;
  readonly isWeight: boolean;
  readonly isInput: boolean;
  readonly quantization: TensorQuantization | null;
  readonly sizeBytes: number;
  readonly initialBytes: Uint8Array | null;
}

export interface ModelOutputDescriptor {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: RuntimeDType;
}

interface CapturedNode {
  readonly id: string | number;
  readonly opType: string;
  readonly inputs: Readonly<Record<string, string>>;
  readonly outputs: Readonly<Record<string, string>>;
  readonly params: NodeParameters;
  readonly wLayout?: 'din' | 'dout';
  readonly extras: Readonly<Record<string, unknown>>;
}

interface CapturedAdapterTarget {
  readonly weight: string;
  readonly kind: 'lora';
  readonly din: number;
  readonly dout: number;
  readonly rank: number;
  readonly alpha: number;
  readonly scale: number;
  readonly A: Float32Array;
  readonly B: Float32Array;
}

interface CapturedAdapter {
  readonly id: string;
  readonly name: string;
  readonly version: number;
  readonly sourceVersion: string | number | null;
  readonly kind: 'lora';
  readonly metadata: Readonly<Record<string, string>>;
  readonly targets: readonly CapturedAdapterTarget[];
}

interface CapturedAdapters {
  readonly versions: readonly CapturedAdapter[];
  readonly nextVersions: ReadonlyMap<string, number>;
  readonly activeId: string | null;
  readonly mergedId: string | null;
  readonly mergedPriorRevision: number | null;
  readonly mergedPreActiveId: string | null;
}

function canonicalDefinitionValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalDefinitionValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).sort(([left], [right]) =>
      left.localeCompare(right)).map(([name, entry]) => [name, canonicalDefinitionValue(entry)]));
  }
  return value;
}

function definitionFingerprint(graph: Graph): string {
  return JSON.stringify(canonicalDefinitionValue({
    tensors: Array.from(graph.tensors.values(), (tensor) => ({
      name: tensor.name,
      shape: tensor.shape,
      dtype: tensor.dtype,
      isWeight: tensor.isWeight === true,
      isInput: tensor.isInput === true,
      quantization: tensor.quantization ?? null,
    })).sort((left, right) => left.name.localeCompare(right.name)),
    nodes: graph.nodes.map((node) => {
      const { id, opType, inputs, outputs, params, wLayout, ...extras } = node;
      return {
        id,
        opType,
        inputs: Object.fromEntries(Object.entries(inputs || {}).map(([name, tensor]) => [name, tensor.name])),
        outputs: Object.fromEntries(Object.entries(outputs || {}).map(([name, tensor]) => [name, tensor.name])),
        params: params || {},
        wLayout: wLayout ?? null,
        extras,
      };
    }),
    outputs: graph.outputNames,
  }));
}

function nextIdentity(): string {
  return runtimeIdentity('model-definition');
}

function nextWeightIdentity(definitionId: string): string {
  return `${definitionId}:${runtimeIdentity('weight-revision')}`;
}

function cloneValue<T>(value: T): T {
  if (value instanceof ArrayBuffer) return value.slice(0) as T;
  if (ArrayBuffer.isView(value)) {
    if (value instanceof DataView) {
      return new DataView(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength)) as T;
    }
    const constructor = value.constructor as {
      new (source: ArrayLike<number>): ArrayBufferView;
    };
    return new constructor(value as unknown as ArrayLike<number>) as T;
  }
  if (Array.isArray(value)) return value.map((entry) => cloneValue(entry)) as T;
  if (value && typeof value === 'object') {
    const prototype = Object.getPrototypeOf(value);
    if (prototype === Object.prototype || prototype === null) {
      return Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([key, entry]) => [key, cloneValue(entry)]),
      ) as T;
    }
  }
  return value;
}

function bytesOf(storage: TensorStorage | undefined): Uint8Array | null {
  if (storage == null) return null;
  if (storage instanceof ArrayBuffer) return new Uint8Array(storage.slice(0));
  return new Uint8Array(
    storage.buffer.slice(storage.byteOffset, storage.byteOffset + storage.byteLength),
  );
}

function typedStorage(dtype: RuntimeDType, bytes: Uint8Array): RuntimeTypedArray {
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  if (dtype === 'float32') return new Float32Array(copy);
  if (dtype === 'int32') return new Int32Array(copy);
  if (dtype === 'int8') return new Int8Array(copy);
  if (dtype === 'uint8') return new Uint8Array(copy);
  throw new Error(`Unsupported snapshot dtype '${dtype}'.`);
}

function captureAdapters(graph: Graph): CapturedAdapters {
  const manager = graph.adapters as any;
  const versions: CapturedAdapter[] = [];
  for (const versionMap of manager?._versions?.values?.() || []) {
    for (const snapshot of versionMap.values()) {
      versions.push(Object.freeze({
        id: snapshot.id,
        name: snapshot.name,
        version: snapshot.version,
        sourceVersion: snapshot.sourceVersion ?? null,
        kind: snapshot.kind,
        metadata: Object.freeze({ ...(snapshot.metadata || {}) }),
        targets: Object.freeze(snapshot.targets.map((target) => Object.freeze({
          weight: target.weight,
          kind: target.kind,
          din: target.din,
          dout: target.dout,
          rank: target.rank,
          alpha: target.alpha,
          scale: target.scale,
          A: new Float32Array(target.A),
          B: new Float32Array(target.B),
        }))),
      }));
    }
  }
  versions.sort((left, right) => left.name.localeCompare(right.name) || left.version - right.version);
  return Object.freeze({
    versions: Object.freeze(versions),
    nextVersions: new Map<string, number>(
      (manager?._nextVersions || []) as Iterable<readonly [string, number]>,
    ),
    activeId: manager?._active?.id || null,
    mergedId: manager?._merged?.snapshot?.id || null,
    mergedPriorRevision: manager?._merged?.priorRevision ?? null,
    mergedPreActiveId: manager?._merged?.preMergeActive?.id || null,
  });
}

function restoreAdapters(graph: Graph, captured: CapturedAdapters): void {
  const manager = graph.adapters as any;
  const snapshotsById = new Map<string, any>();
  const versions = new Map<string, Map<number, any>>();
  for (const capturedSnapshot of captured.versions) {
    const targets = Object.freeze(capturedSnapshot.targets.map((target) => Object.freeze({
      ...target,
      A: new Float32Array(target.A),
      B: new Float32Array(target.B),
    })));
    const snapshot = Object.freeze({
      id: capturedSnapshot.id,
      name: capturedSnapshot.name,
      version: capturedSnapshot.version,
      sourceVersion: capturedSnapshot.sourceVersion,
      kind: capturedSnapshot.kind,
      metadata: Object.freeze({ ...capturedSnapshot.metadata }),
      targets,
      targetsByWeight: new Map(targets.map((target) => [target.weight, target])),
    });
    let namedVersions = versions.get(snapshot.name);
    if (!namedVersions) {
      namedVersions = new Map();
      versions.set(snapshot.name, namedVersions);
    }
    namedVersions.set(snapshot.version, snapshot);
    snapshotsById.set(snapshot.id, snapshot);
  }
  manager._versions = versions;
  manager._nextVersions = new Map(captured.nextVersions);
  manager._active = captured.activeId ? snapshotsById.get(captured.activeId) || null : null;
  manager._acceleratedBackends = new Set();
  manager._resourceOwners = new Set();
  if (captured.mergedId) {
    manager._merged = {
      snapshot: snapshotsById.get(captured.mergedId),
      plans: [],
      revision: graph.weightRevision,
      priorRevision: captured.mergedPriorRevision ?? graph.weightRevision,
      preMergeActive: captured.mergedPreActiveId
        ? snapshotsById.get(captured.mergedPreActiveId) || null
        : null,
    };
  } else {
    manager._merged = null;
  }
}

/**
 * Immutable, execution-safe capture of one graph topology and weight/adapter revision.
 * Mutable byte storage is private and copied again for every execution graph.
 */
export class ModelSnapshot {
  readonly definitionId: string;
  readonly weightRevisionId: string;
  readonly adapterRevisionId: string | null;
  readonly adapterRevisionIds: readonly string[];
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly outputNames: readonly string[];
  readonly outputDescriptors: readonly ModelOutputDescriptor[];
  readonly tensorCount: number;
  readonly nodeCount: number;

  readonly #tensors: readonly CapturedTensor[];
  readonly #nodes: readonly CapturedNode[];
  readonly #adapters: CapturedAdapters;
  readonly #definitionFingerprint: string;

  private constructor(graph: Graph, previous: ModelSnapshot | null) {
    this.#definitionFingerprint = definitionFingerprint(graph);
    this.definitionId = previous && previous.#definitionFingerprint === this.#definitionFingerprint
      ? previous.definitionId
      : nextIdentity();
    this.topologyRevision = graph.topologyRevision || 0;
    this.weightRevision = graph.weightRevision || 0;
    this.weightRevisionId = nextWeightIdentity(this.definitionId);
    this.outputNames = Object.freeze([...graph.outputNames]);
    this.outputDescriptors = Object.freeze(this.outputNames.map((name) => {
      const tensor = graph.getTensor(name);
      if (!tensor) throw new Error(`Model output '${name}' is not a declared tensor.`);
      return Object.freeze({
        name,
        shape: Object.freeze([...tensor.shape]),
        dtype: tensor.dtype,
      });
    }));
    const produced = new Set(graph.nodes.flatMap((node) =>
      Object.values(node.outputs || {}).map((tensor) => tensor.name)));
    this.#tensors = Object.freeze(Array.from(graph.tensors.values(), (tensor) => Object.freeze({
      name: tensor.name,
      shape: Object.freeze([...tensor.shape]),
      dtype: tensor.dtype,
      isWeight: tensor.isWeight === true,
      isInput: tensor.isInput === true,
      quantization: tensor.quantization ? cloneValue(tensor.quantization) : null,
      sizeBytes: tensor.sizeBytes,
      initialBytes: tensor.isWeight || (!tensor.isInput && !produced.has(tensor.name))
        ? bytesOf(tensor.buffer)
        : null,
    })));
    this.#nodes = Object.freeze(graph.nodes.map((node) => {
      const { id, opType, inputs, outputs, params, wLayout, ...extras } = node;
      return Object.freeze({
        id,
        opType,
        inputs: Object.freeze(Object.fromEntries(
          Object.entries(inputs || {}).map(([name, tensor]) => [name, tensor.name]),
        )),
        outputs: Object.freeze(Object.fromEntries(
          Object.entries(outputs || {}).map(([name, tensor]) => [name, tensor.name]),
        )),
        params: cloneValue(params || {}),
        ...(wLayout ? { wLayout } : {}),
        extras: Object.freeze(cloneValue(extras)),
      });
    }));
    this.#adapters = captureAdapters(graph);
    this.adapterRevisionId = this.#adapters.mergedId || this.#adapters.activeId;
    this.adapterRevisionIds = Object.freeze(this.#adapters.versions.map((adapter) => adapter.id));
    this.tensorCount = this.#tensors.length;
    this.nodeCount = this.#nodes.length;
  }

  static capture(graph: Graph): ModelSnapshot {
    if (!(graph instanceof Graph)) throw new Error('ModelSnapshot.capture requires a Graph.');
    graph.assertValid();
    return new ModelSnapshot(graph, null);
  }

  /** Derive a successor; weight-only revisions retain their model-definition identity. */
  static derive(graph: Graph, previous: ModelSnapshot): ModelSnapshot {
    if (!(graph instanceof Graph)) throw new Error('ModelSnapshot.derive requires a Graph.');
    if (!(previous instanceof ModelSnapshot)) {
      throw new Error('ModelSnapshot.derive requires a previous ModelSnapshot.');
    }
    graph.assertValid();
    return new ModelSnapshot(graph, previous);
  }

  /** Compare a candidate graph with the normalized definition retained by this snapshot. */
  matchesDefinition(graph: Graph): boolean {
    if (!(graph instanceof Graph)) return false;
    graph.assertValid();
    return definitionFingerprint(graph) === definitionFingerprint(this.createExecutionGraph());
  }

  /** Resolve an execution selector against the immutable adapter versions in this snapshot. */
  resolveAdapterRevisionId(selector: unknown = undefined): string | null {
    if (this.#adapters.mergedId) return this.#adapters.mergedId;
    if (selector === undefined) return this.#adapters.activeId;
    if (selector == null) return null;
    let requestedVersion: number | null = null;
    if (typeof selector === 'object' && !Array.isArray(selector)) {
      const record = selector as Readonly<Record<string, unknown>>;
      for (const field of Object.keys(record)) {
        if (field !== 'name' && field !== 'version' && field !== 'scale') {
          throw new Error(`Adapter selector contains unsupported field '${field}'.`);
        }
      }
      const candidateName = record.name;
      if (typeof candidateName !== 'string') throw new Error('Adapter selector requires a name.');
      const name = candidateName;
      const candidateVersion = record.version;
      if (candidateVersion != null) {
        if (typeof candidateVersion !== 'number' ||
            !Number.isSafeInteger(candidateVersion) || candidateVersion <= 0) {
          throw new Error('Adapter selector version must be a positive integer.');
        }
        requestedVersion = candidateVersion;
      }
      if (record.scale != null &&
          (typeof record.scale !== 'number' || !Number.isFinite(record.scale))) {
        throw new Error('Adapter selector scale must be finite.');
      }
      if (record.scale === 0) return null;
      const versions = this.#adapters.versions.filter((adapter) => adapter.name === name);
      if (versions.length === 0) throw new Error(`Adapter '${name}' is not staged.`);
      const selected = requestedVersion == null
        ? versions.at(-1)
        : versions.find((adapter) => adapter.version === requestedVersion);
      if (!selected) throw new Error(`Adapter '${name}' version '${requestedVersion}' is not staged.`);
      return selected.id;
    } else {
      throw new Error('Adapter selector must be an object or null.');
    }
  }

  /** Build a private mutable graph binding for exactly one backend context. */
  createExecutionGraph(): Graph {
    const graph = new Graph();
    const tensors = new Map<string, Tensor>();
    for (const captured of this.#tensors) {
      const tensor = new Tensor(captured.name, captured.shape, captured.dtype, captured.isWeight, {
        isInput: captured.isInput,
        quantization: captured.quantization,
      });
      tensor.sizeBytes = captured.sizeBytes;
      if (captured.initialBytes) tensor.buffer = typedStorage(captured.dtype, captured.initialBytes);
      tensors.set(tensor.name, tensor);
    }
    const nodes = this.#nodes.map((captured) => ({
      ...cloneValue(captured.extras),
      id: captured.id,
      opType: captured.opType,
      inputs: Object.fromEntries(Object.entries(captured.inputs).map(([name, tensorName]) => {
        const tensor = tensors.get(tensorName);
        if (!tensor) throw new Error(`Snapshot node '${String(captured.id)}' references missing input '${tensorName}'.`);
        return [name, tensor];
      })),
      outputs: Object.fromEntries(Object.entries(captured.outputs).map(([name, tensorName]) => {
        const tensor = tensors.get(tensorName);
        if (!tensor) throw new Error(`Snapshot node '${String(captured.id)}' references missing output '${tensorName}'.`);
        return [name, tensor];
      })),
      params: cloneValue(captured.params),
      ...(captured.wLayout ? { wLayout: captured.wLayout } : {}),
    })) as GraphNode<Tensor>[];

    graph.nodes = nodes;
    graph.tensors = tensors;
    graph._setOutputNames(this.outputNames);
    graph._autoOutputNames = [...this.outputNames];
    graph._outputsExplicit = true;
    graph.topologyRevision = this.topologyRevision;
    graph._nextTopologyRevision = Math.max(this.topologyRevision + 1, 1);
    graph.weightRevision = this.weightRevision;
    graph._nextWeightRevision = Math.max(this.weightRevision + 1, 1);
    graph._nextNodeId = nodes.reduce((next, node) =>
      typeof node.id === 'number' && Number.isSafeInteger(node.id) ? Math.max(next, node.id + 1) : next, 0);
    restoreAdapters(graph, this.#adapters);
    GraphOperatorNormalizer.ensureMatMulLayouts(graph);
    graph.assertValid();
    return graph;
  }
}
