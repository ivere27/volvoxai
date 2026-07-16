import { SafetensorsFile } from './Safetensors.js';
import type { Graph } from './Graph.js';
import type { Tensor } from './Tensor.js';
import type {
  AdapterDescription,
  AdapterExportOptions,
  AdapterLoadOptions,
  AdapterSpec,
  AdapterStageOptions,
  AdapterTargetSpec,
  AdapterTensorInput,
  AdapterUpdateOptions,
  AdapterVersion,
  GraphNode,
} from '../types.js';

type ConcreteNode = GraphNode<Tensor>;
type AdapterKind = 'lora';

interface NormalizedAdapterTarget {
  readonly weight: string;
  readonly kind: AdapterKind;
  readonly din: number;
  readonly dout: number;
  readonly rank: number;
  readonly alpha: number;
  readonly scale: number;
  readonly A: Float32Array;
  readonly B: Float32Array;
}

interface AdapterSnapshot {
  readonly id: string;
  readonly name: string;
  readonly version: number;
  readonly sourceVersion: string | number | null;
  readonly kind: AdapterKind;
  readonly metadata: Readonly<Record<string, string>>;
  readonly targets: readonly NormalizedAdapterTarget[];
  readonly targetsByWeight: Map<string, NormalizedAdapterTarget>;
}

interface MergePlan {
  weight: Tensor & { buffer: Float32Array };
  backup: Float32Array;
  merged: Float32Array;
}

interface MergeState {
  snapshot: AdapterSnapshot;
  plans: MergePlan[];
  revision: number;
  priorRevision: number;
  preMergeActive: AdapterSnapshot | null;
}

interface AdapterResourceOwner {
  releaseAdapterTargets?(targets: readonly NormalizedAdapterTarget[]): void;
}

interface AdapterSelector {
  name?: string;
  adapterId?: string;
  adapter_id?: string;
  version?: AdapterVersion;
  versionId?: AdapterVersion;
  version_id?: AdapterVersion;
  scale?: number;
}

interface PinnedAdapterRoute {
  readonly snapshot: AdapterSnapshot;
  readonly scale: number;
}

interface AdapterExecutionOptions {
  adapter?: string | AdapterSelector | null;
  adapters?: Array<string | AdapterSelector | null>;
}

interface AdapterManifestTensorSpec {
  role: 'a' | 'b';
  name: string;
  shape: number[];
  dtype: 'F16' | 'F32';
}

interface AdapterManifestTarget {
  weight: string;
  a?: string;
  b?: string;
  layout: string;
  rank?: number;
  alpha: number;
  scale?: number;
  kind?: AdapterKind;
  tensors?: AdapterManifestTensorSpec[];
}

interface AdapterManifest {
  format: string;
  name?: string;
  adapter_id: string;
  version_id: string;
  kind: AdapterKind;
  targets: AdapterManifestTarget[];
  metadata?: Record<string, string>;
}

export const VOLVOX_ADAPTER_FORMAT = "volvox.adapter.v1";
export const VOLVOX_ADAPTER_MANIFEST_KEY = "volvox_adapter_manifest";

const LINEAR_OPS = new Set(["MatMul", "Linear", "Gemm"]);
const CANONICAL_LAYOUT = "din_r_r_dout";
const PEFT_LAYOUT = "peft";
const MANIFEST_FIELDS = new Set(["format", "name", "adapter_id", "version_id", "kind", "targets", "metadata"]);
const TARGET_FIELDS = new Set(["id", "op", "weight", "kind", "a", "b", "layout", "rank", "alpha", "scale", "tensors"]);
const TENSOR_SPEC_FIELDS = new Set(["role", "name", "shape", "dtype"]);

function last(values: readonly number[] | null | undefined): number | undefined {
  return values?.[values.length - 1];
}

function firstOutput(node: ConcreteNode): Tensor | undefined {
  return node.outputs?.out || Object.values(node.outputs || {})[0];
}

function linearInput(node: ConcreteNode): Tensor | undefined {
  return node.inputs?.input || node.inputs?.x || node.inputs?.data;
}

function asF32(value: unknown, label: string): Float32Array {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : null;
  const source = ArrayBuffer.isView(value) || Array.isArray(value)
    ? value
    : record?.data ?? record?.values ?? record?.buffer ?? value;
  let result: Float32Array;
  if (source instanceof Float32Array) {
    result = new Float32Array(source);
  } else if (ArrayBuffer.isView(source) && !(source instanceof DataView)) {
    try {
      result = Float32Array.from(source as unknown as ArrayLike<number>);
    } catch {
      throw new Error(`${label} must contain numeric values.`);
    }
  } else if (source instanceof ArrayBuffer) {
    if (source.byteLength % 4 !== 0) throw new Error(`${label} is not F32-aligned.`);
    result = new Float32Array(source.slice(0));
  } else if (Array.isArray(source)) {
    result = Float32Array.from(source);
  } else {
    throw new Error(`${label} must contain F32 values.`);
  }
  for (let i = 0; i < result.length; i++) {
    if (!Number.isFinite(result[i])) throw new Error(`${label} contains a non-finite value at index ${i}.`);
  }
  return result;
}

function valueShape(value: unknown): number[] | null {
  const shape = value && typeof value === 'object'
    ? (value as { shape?: unknown }).shape
    : undefined;
  return Array.isArray(shape) ? [...shape] as number[] : null;
}

function transpose2D(data: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) out[col * rows + row] = data[row * cols + col];
  }
  return out;
}

function normalizeKind(value: unknown): AdapterKind {
  const kind = String(value || "lora");
  if (kind !== "lora") throw new Error(`Unsupported adapter kind '${value}'. Only LoRA is supported.`);
  return kind;
}

function normalizeTargets(
  targets: AdapterSpec['targets'] | undefined,
): AdapterTargetSpec[] {
  if (Array.isArray(targets)) return targets.map((target) => ({ ...target }));
  if (targets && typeof targets === "object") {
    return Object.entries(targets).map(([weight, target]) => ({ ...target, weight: target.weight || weight }));
  }
  throw new Error("Adapter spec requires at least one target.");
}

function f32Bytes(array: Float32Array): Uint8Array {
  return new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
}

function sameSnapshot(a: AdapterSnapshot | null | undefined, b: AdapterSnapshot | null | undefined): boolean {
  return !!a && !!b && a.name === b.name && a.version === b.version;
}

function immutableMetadata(value: unknown): Readonly<Record<string, string>> {
  const metadata = value ?? {};
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error("Adapter metadata must be a string-valued object.");
  }
  const normalized: Record<string, string> = {};
  for (const [key, entry] of Object.entries(metadata)) {
    if (typeof entry !== "string") throw new Error(`Adapter metadata '${key}' must be a string.`);
    normalized[key] = entry;
  }
  return Object.freeze(normalized);
}

function assertOnlyFields(
  value: unknown,
  allowed: ReadonlySet<string>,
  label: string,
): asserts value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) throw new Error(`${label} contains unsupported field '${key}'.`);
  }
}

export class AdapterManager {
  graph: Graph;
  _versions: Map<string, Map<number, AdapterSnapshot>>;
  _nextVersions: Map<string, number>;
  _active: AdapterSnapshot | null;
  _merged: MergeState | null;
  _acceleratedBackends: Set<string>;
  _resourceOwners: Set<AdapterResourceOwner>;

  constructor(graph: Graph) {
    this.graph = graph;
    this._versions = new Map();
    this._nextVersions = new Map();
    this._active = null;
    this._merged = null;
    this._acceleratedBackends = new Set();
    this._resourceOwners = new Set();
  }

  hasActive(): boolean {
    return this._active != null;
  }

  stage(name: string, spec: AdapterSpec, options: AdapterStageOptions = {}): AdapterDescription {
    if (typeof name !== "string" || !name.trim()) throw new Error("Adapter name must be a non-empty string.");
    const adapterName = name.trim();
    const kind = normalizeKind(spec?.kind || spec?.type);
    const targets = normalizeTargets(spec?.targets);
    if (targets.length === 0) throw new Error("Adapter spec requires at least one target.");
    const activate = options.activate === true || spec?.activate === true;
    if (activate && this._merged) {
      throw new Error(`Unmerge '${this._merged.snapshot.id}' before activating an adapter.`);
    }

    const normalizedTargets = targets.map((target, index) => this._normalizeTarget(kind, spec, target, index));
    const seen = new Set();
    for (const target of normalizedTargets) {
      if (seen.has(target.weight)) throw new Error(`Adapter '${adapterName}' targets '${target.weight}' more than once.`);
      seen.add(target.weight);
    }

    const version = this._nextVersions.get(adapterName) || 1;
    this._nextVersions.set(adapterName, version + 1);
    const snapshot = Object.freeze({
      id: `${adapterName}@${version}`,
      name: adapterName,
      version,
      sourceVersion: spec?.sourceVersion ?? null,
      kind,
      metadata: immutableMetadata(spec?.metadata),
      targets: Object.freeze(normalizedTargets),
      targetsByWeight: new Map(normalizedTargets.map((target) => [target.weight, target])),
    });

    let versions = this._versions.get(adapterName);
    if (!versions) {
      versions = new Map();
      this._versions.set(adapterName, versions);
    }
    versions.set(version, snapshot);
    if (activate) this._activateSnapshot(snapshot);
    return this._describe(snapshot);
  }

  update(
    name: string,
    updates: Record<string, Record<string, unknown>>,
    options: AdapterUpdateOptions = {},
  ): AdapterDescription {
    const source = this._resolve(name, options.version);
    const mode = options.mode || "assign";
    if (mode !== "assign" && mode !== "add") throw new Error(`Unsupported adapter update mode '${mode}'.`);
    if (!updates || typeof updates !== "object") throw new Error("Adapter updates must be keyed by base tensor name.");

    const targetSpecs: AdapterTargetSpec[] = source.targets.map((target) => ({
      weight: target.weight,
      layout: "din_r_r_dout",
      rank: target.rank,
      alpha: target.alpha,
      scale: target.scale,
      A: new Float32Array(target.A),
      B: new Float32Array(target.B),
    }));
    const byWeight = new Map(targetSpecs.map((target) => [target.weight, target]));
    for (const [weight, patch] of Object.entries(updates)) {
      const target = byWeight.get(weight);
      if (!target) throw new Error(`Adapter '${source.id}' does not target '${weight}'.`);
      for (const [field, value] of Object.entries(patch || {})) {
        const canonical = field === "a" ? "A" : field === "b" ? "B" : field;
        if (!["A", "B"].includes(canonical)) {
          throw new Error(`Unsupported adapter tensor role '${field}'.`);
        }
        const next = asF32(value, `${weight}.${canonical}`);
        const tensor = target[canonical as 'A' | 'B'] as Float32Array | undefined;
        if (!tensor || next.length !== tensor.length) {
          throw new Error(`Adapter tensor '${weight}.${canonical}' has an invalid length.`);
        }
        if (mode === "assign") tensor.set(next);
        else for (let i = 0; i < next.length; i++) tensor[i] += next[i];
      }
    }
    return this.stage(source.name, {
      kind: source.kind,
      targets: targetSpecs,
      metadata: { ...source.metadata },
      sourceVersion: source.version,
    }, { activate: options.activate === true });
  }

  load(
    source: SafetensorsFile | ArrayBuffer | ArrayBufferView,
    options: AdapterLoadOptions = {},
  ): AdapterDescription {
    let file: SafetensorsFile | null = source instanceof SafetensorsFile ? source : null;
    if (!file) {
      let buffer: ArrayBuffer;
      if (source instanceof ArrayBuffer) buffer = source;
      else if (ArrayBuffer.isView(source)) {
        buffer = new Uint8Array(source.buffer, source.byteOffset, source.byteLength).slice().buffer;
      }
      else throw new Error("Adapter load expects a SafetensorsFile, ArrayBuffer, or byte view.");
      file = SafetensorsFile.fromArrayBuffer(buffer);
    }
    const encoded = file.metadata?.[VOLVOX_ADAPTER_MANIFEST_KEY];
    if (typeof encoded !== "string") throw new Error(`Missing '${VOLVOX_ADAPTER_MANIFEST_KEY}' metadata.`);
    let manifestValue: unknown;
    try { manifestValue = JSON.parse(encoded); }
    catch { throw new Error("Adapter manifest is not valid JSON."); }
    assertOnlyFields(manifestValue, MANIFEST_FIELDS, "Adapter manifest");
    const manifest = manifestValue as unknown as AdapterManifest;
    if (manifest?.format !== VOLVOX_ADAPTER_FORMAT) throw new Error(`Unsupported adapter format '${manifest?.format}'.`);
    if (typeof manifest.adapter_id !== "string" || !manifest.adapter_id.trim()) throw new Error("Adapter manifest requires adapter_id.");
    if (typeof manifest.version_id !== "string" || !manifest.version_id.trim()) throw new Error("Adapter manifest requires version_id.");
    if (options.name != null && options.name !== manifest.adapter_id) {
      throw new Error("Adapter load name must match the checkpoint adapter_id.");
    }
    if (manifest.kind !== "lora") {
      throw new Error(`Unsupported adapter kind '${manifest.kind}'. Only LoRA is supported.`);
    }
    const manifestKind = normalizeKind(manifest.kind);
    if (!Array.isArray(manifest.targets) || manifest.targets.length === 0) throw new Error("Adapter manifest has no targets.");

    const read = (tensorName: string | undefined, role: string): AdapterTensorInput | undefined => {
      if (!tensorName) return undefined;
      const tensor = file.getTensor(tensorName);
      if (!tensor) throw new Error(`Adapter manifest ${role} tensor '${tensorName}' is missing.`);
      if (tensor.dtype !== "F16" && tensor.dtype !== "F32") {
        throw new Error(`Adapter tensor '${tensorName}' must be F16 or F32.`);
      }
      return { data: file.toRuntimeTypedArray(tensor), shape: [...tensor.shape] };
    };
    const targets: AdapterTargetSpec[] = manifest.targets.map((target, index) => {
      assertOnlyFields(target, TARGET_FIELDS, `Adapter manifest target ${index}`);
      if (typeof target?.layout !== "string" || !target.layout) {
        throw new Error(`Adapter manifest target ${index} requires an explicit matrix layout.`);
      }
      if (typeof target.alpha !== "number" || !Number.isFinite(target.alpha) || target.alpha <= 0) {
        throw new Error(`Adapter manifest target ${index} requires a positive finite alpha.`);
      }
      if (target.kind != null && normalizeKind(target.kind) !== manifestKind) {
        throw new Error(`Adapter manifest target ${index} kind does not match the global kind.`);
      }
      if (target.tensors != null) {
        if (!Array.isArray(target.tensors) || target.tensors.length !== 2) {
          throw new Error(`Adapter manifest target ${index} tensor specs must contain A and B.`);
        }
        const roles = new Set();
        for (const [specIndex, tensorSpec] of target.tensors.entries()) {
          assertOnlyFields(tensorSpec, TENSOR_SPEC_FIELDS,
            `Adapter manifest target ${index} tensor spec ${specIndex}`);
          const role = tensorSpec.role;
          if ((role !== "a" && role !== "b") || roles.has(role)) {
            throw new Error(`Adapter manifest target ${index} tensor specs require unique A and B roles.`);
          }
          roles.add(role);
          if (tensorSpec.name !== target[role]) {
            throw new Error(`Adapter manifest target ${index} ${role.toUpperCase()} tensor spec name does not match.`);
          }
          const tensor = file.getTensor(tensorSpec.name);
          if (!tensor || tensor.dtype !== tensorSpec.dtype || !Array.isArray(tensorSpec.shape) ||
              tensorSpec.shape.length !== tensor.shape.length ||
              tensorSpec.shape.some((dim, dimIndex) => !Number.isInteger(dim) || dim <= 0 || dim !== tensor.shape[dimIndex])) {
            throw new Error(`Adapter manifest target ${index} ${role.toUpperCase()} tensor spec does not match its payload.`);
          }
        }
      }
      return {
        weight: target.weight,
        layout: target.layout,
        rank: target.rank,
        alpha: target.alpha,
        scale: target.scale,
        A: read(target.a, "A"),
        B: read(target.b, "B"),
        kind: target.kind,
      };
    });
    return this.stage(manifest.adapter_id, {
      kind: manifestKind,
      targets,
      metadata: manifest.metadata || {},
      sourceVersion: manifest.version_id,
    }, { activate: options.activate === true });
  }

  export(
    name: string,
    version: AdapterVersion | AdapterExportOptions = undefined,
    options: AdapterExportOptions = {},
  ): ArrayBuffer | Blob | SafetensorsFile {
    if (version && typeof version === "object" && Object.prototype.hasOwnProperty.call(version, "as") &&
        !Object.prototype.hasOwnProperty.call(version, "version")) {
      options = version;
      version = undefined;
    }
    const resolvedVersion: AdapterVersion = typeof version === 'object' ? undefined : version;
    const snapshot = this._resolve(name, resolvedVersion);
    const file = SafetensorsFile.empty();
    const manifestTargets: AdapterManifestTarget[] = [];
    snapshot.targets.forEach((target, index) => {
      const prefix = `adapter.${index}`;
      const aName = `${prefix}.a`;
      const bName = `${prefix}.b`;
      file.addTensor(aName, "F32", [target.din, target.rank], f32Bytes(target.A));
      file.addTensor(bName, "F32", [target.rank, target.dout], f32Bytes(target.B));
      const entry = {
        weight: target.weight,
        a: aName,
        b: bName,
        layout: "din_r_r_dout",
        rank: target.rank,
        alpha: target.alpha,
        scale: target.scale,
      };
      manifestTargets.push(entry);
    });
    const manifest: Omit<AdapterManifest, 'format' | 'version_id'> & {
      format: typeof VOLVOX_ADAPTER_FORMAT;
      version_id: string;
    } = {
      format: VOLVOX_ADAPTER_FORMAT,
      adapter_id: snapshot.name,
      version_id: String(snapshot.version),
      kind: snapshot.kind,
      targets: manifestTargets,
    };
    if (Object.keys(snapshot.metadata).length) manifest.metadata = { ...snapshot.metadata };
    file.metadata = { [VOLVOX_ADAPTER_MANIFEST_KEY]: JSON.stringify(manifest) };
    file.hasMetadata = true;
    if (options.as != null && !["arraybuffer", "file", "blob"].includes(options.as)) {
      throw new Error(`Unsupported adapter export type '${options.as}'.`);
    }
    if (options.as === "file") return file;
    if (options.as === "blob") return file.toBlob();
    return file.toArrayBuffer();
  }

  activate(name: string | null, version?: AdapterVersion): AdapterDescription | null {
    if (this._merged) {
      throw new Error(`Unmerge '${this._merged.snapshot.id}' before activating an adapter.`);
    }
    if (name == null) {
      this._active = null;
      return null;
    }
    const snapshot = this._resolve(name, version);
    this._activateSnapshot(snapshot);
    return this._describe(snapshot);
  }

  remove(name: string, version?: AdapterVersion): boolean {
    const versions = this._versions.get(name);
    if (!versions) return false;
    if (version == null) {
      if (this._merged?.snapshot.name === name) throw new Error(`Unmerge '${this._merged.snapshot.id}' before removing it.`);
      if (this._merged?.preMergeActive?.name === name) {
        throw new Error(`Unmerge '${this._merged.snapshot.id}' before removing its saved active adapter.`);
      }
      if (this._active?.name === name) {
        throw new Error(`Activate the base model before removing active adapter '${this._active.id}'.`);
      }
      for (const snapshot of versions.values()) this._releaseSnapshotResources(snapshot);
      this._versions.delete(name);
      return true;
    }
    const numeric = Number(version);
    const snapshot = versions.get(numeric);
    if (!snapshot) return false;
    if (sameSnapshot(this._merged?.snapshot, snapshot)) throw new Error(`Unmerge '${snapshot.id}' before removing it.`);
    const merged = this._merged;
    if (sameSnapshot(merged?.preMergeActive, snapshot) && merged) {
      throw new Error(`Unmerge '${merged.snapshot.id}' before removing its saved active adapter.`);
    }
    if (sameSnapshot(this._active, snapshot)) {
      throw new Error(`Activate the base model before removing active adapter '${snapshot.id}'.`);
    }
    this._releaseSnapshotResources(snapshot);
    versions.delete(numeric);
    if (versions.size === 0) this._versions.delete(name);
    return true;
  }

  list(): AdapterDescription[] {
    const result: AdapterDescription[] = [];
    for (const versions of this._versions.values()) {
      for (const snapshot of versions.values()) result.push(this._describe(snapshot));
    }
    return result.sort((a, b) => a.name.localeCompare(b.name) || a.version - b.version);
  }

  active(): AdapterDescription | null {
    return this._active ? this._describe(this._active) : null;
  }

  merge(name: string, version?: AdapterVersion): AdapterDescription {
    if (this._merged) throw new Error(`Adapter '${this._merged.snapshot.id}' is already merged; unmerge it first.`);
    if (this._acceleratedBackends.size) {
      throw new Error(`Cannot merge after accelerated compilation (${[...this._acceleratedBackends].join(", ")}); recompile from an unbound graph.`);
    }
    const snapshot = this._resolve(name, version);
    const plans: MergePlan[] = [];
    for (const target of snapshot.targets) {
      const weight = this.graph.getTensor(target.weight);
      if (!weight || weight.dtype !== "float32" || !(weight.buffer instanceof Float32Array)) {
        throw new Error(`Merge requires F32 base tensor '${target.weight}'.`);
      }
      const writableWeight = weight as Tensor & { buffer: Float32Array };
      const backup = new Float32Array(writableWeight.buffer);
      const mergedValues = new Float32Array(writableWeight.buffer);
      this._mergeTarget(target, mergedValues);
      plans.push({ weight: writableWeight, backup, merged: mergedValues });
    }
    const priorRevision = this.graph.weightRevision || 0;
    for (const plan of plans) plan.weight.buffer.set(plan.merged);
    const revision = this.graph._advanceWeightRevision
      ? this.graph._advanceWeightRevision()
      : (this.graph.weightRevision = priorRevision + 1);
    this._merged = { snapshot, plans, revision, priorRevision, preMergeActive: this._active };
    this._active = null;
    return this._describe(snapshot);
  }

  unmerge(): AdapterDescription | null {
    if (!this._merged) return null;
    if ((this.graph.weightRevision || 0) !== this._merged.revision) {
      throw new Error("Base tensors changed after merge; refusing to overwrite them during unmerge.");
    }
    for (const plan of this._merged.plans) {
      if (plan.weight.buffer.length !== plan.merged.length ||
          plan.weight.buffer.some((value, index) => !Object.is(value, plan.merged[index]))) {
        throw new Error(`Merged base tensor '${plan.weight.name}' changed; refusing to overwrite it during unmerge.`);
      }
    }
    const { snapshot, preMergeActive } = this._merged;
    for (const plan of this._merged.plans) plan.weight.buffer.set(plan.backup);
    this.graph.weightRevision = this._merged.priorRevision;
    this._merged = null;
    this._active = preMergeActive;
    return this._describe(snapshot);
  }

  _normalizeTarget(
    defaultKind: AdapterKind,
    spec: AdapterSpec,
    input: AdapterTargetSpec,
    index: number,
  ): NormalizedAdapterTarget {
    const kind = input.kind ? normalizeKind(input.kind) : defaultKind;
    if (kind !== defaultKind) throw new Error("One adapter version cannot mix adapter kinds.");
    const weightName = input.weight || input.baseTensor || input.target;
    if (typeof weightName !== 'string' || !weightName) {
      throw new Error(`Adapter target ${index} requires a base tensor name.`);
    }
    const weight = this.graph.getTensor(weightName);
    if (!weight) throw new Error(`Adapter target ${index} references missing weight '${weightName}'.`);
    if (weight.dtype !== "float32" || !(weight.buffer instanceof Float32Array)) {
      throw new Error(`LoRA target '${weightName}' requires an F32 runtime base tensor.`);
    }
    const nodes = this.graph.nodes.filter((node) => LINEAR_OPS.has(node.opType) && node.inputs?.weight?.name === weightName);
    if (nodes.length === 0) throw new Error(`Adapter target '${weightName}' is not used by a linear node.`);
    const first = nodes[0];
    const din = last(linearInput(first)?.shape);
    const dout = last(firstOutput(first)?.shape);
    if (typeof din !== 'number' || typeof dout !== 'number' ||
        !Number.isInteger(din) || !Number.isInteger(dout)) {
      throw new Error(`Cannot resolve dimensions for '${weightName}'.`);
    }
    for (const node of nodes) {
      if (last(linearInput(node)?.shape) !== din || last(firstOutput(node)?.shape) !== dout) {
        throw new Error(`Shared adapter target '${weightName}' has inconsistent linear dimensions.`);
      }
      if (this._isDoutFirst(node, weight, din, dout) !== this._isDoutFirst(first, weight, din, dout)) {
        throw new Error(`Shared adapter target '${weightName}' has inconsistent weight layouts.`);
      }
    }
    let A = asF32(input.A ?? input.a, `${weightName}.A`);
    let B = asF32(input.B ?? input.adapterB, `${weightName}.B`);
    const aShape = valueShape(input.A ?? input.a);
    const bShape = valueShape(input.B ?? input.adapterB);
    const layout = String(input.layout || spec.layout || CANONICAL_LAYOUT);
    if (layout !== CANONICAL_LAYOUT && layout !== PEFT_LAYOUT) {
      throw new Error(`Adapter target '${weightName}' has unsupported matrix layout '${layout}'.`);
    }
    const peft = layout === PEFT_LAYOUT;
    let rank = input.rank ?? spec.rank;
    if (rank == null && aShape) rank = peft ? aShape[0] : aShape[1];
    if (rank == null && din > 0 && A.length % din === 0) rank = A.length / din;
    if (typeof rank !== 'number' || !Number.isInteger(rank) || rank <= 0) {
      throw new Error(`Adapter target '${weightName}' has an invalid rank.`);
    }
    if (A.length !== din * rank || B.length !== rank * dout) {
      throw new Error(`Adapter target '${weightName}' expects A=[${din},${rank}] and B=[${rank},${dout}] (${peft ? "after PEFT normalization" : "canonical"}).`);
    }
    if (peft) {
      if (aShape && (aShape[0] !== rank || aShape[1] !== din)) throw new Error(`PEFT A for '${weightName}' must be [rank,din].`);
      if (bShape && (bShape[0] !== dout || bShape[1] !== rank)) throw new Error(`PEFT B for '${weightName}' must be [dout,rank].`);
      A = transpose2D(A, rank, din);
      B = transpose2D(B, dout, rank);
    } else {
      if (aShape && (aShape[0] !== din || aShape[1] !== rank)) throw new Error(`Canonical A for '${weightName}' must be [din,rank].`);
      if (bShape && (bShape[0] !== rank || bShape[1] !== dout)) throw new Error(`Canonical B for '${weightName}' must be [rank,dout].`);
    }

    const alpha = input.alpha ?? spec.alpha ?? rank;
    if (typeof alpha !== "number" || !Number.isFinite(alpha) || alpha <= 0) {
      throw new Error(`Adapter target '${weightName}' requires a positive finite alpha.`);
    }
    const defaultScale = alpha / rank;
    const scale = input.scale ?? spec.scale ?? defaultScale;
    if (typeof scale !== "number" || !Number.isFinite(scale)) {
      throw new Error(`Adapter target '${weightName}' has an invalid scale.`);
    }

    return Object.freeze({
      weight: weightName,
      kind,
      din,
      dout,
      rank,
      alpha,
      scale,
      A,
      B,
    });
  }

  _baseValue(
    weight: Tensor & { buffer: Float32Array },
    node: ConcreteNode,
    k: number,
    j: number,
    din: number,
    dout: number,
    buffer: Float32Array = weight.buffer,
  ): number {
    const doutFirst = this._isDoutFirst(node, weight, din, dout);
    return doutFirst ? buffer[j * din + k] : buffer[k * dout + j];
  }

  _setBaseValue(
    weight: Tensor & { buffer: Float32Array },
    node: ConcreteNode,
    buffer: Float32Array,
    k: number,
    j: number,
    value: number,
    din: number,
    dout: number,
  ): void {
    const doutFirst = this._isDoutFirst(node, weight, din, dout);
    buffer[doutFirst ? j * din + k : k * dout + j] = value;
  }

  _isDoutFirst(node: ConcreteNode, weight: Tensor, din: number, dout: number): boolean {
    return (node.inputs.scale || node.inputs.weight_scale)
      ? true
      : (node.wLayout ? node.wLayout === "dout" : weight.shape[0] === dout && weight.shape[1] === din);
  }

  _mergeTarget(target: NormalizedAdapterTarget, buffer: Float32Array): void {
    const weight = this.graph.getTensor(target.weight);
    const node = this.graph.nodes.find((candidate) => LINEAR_OPS.has(candidate.opType) && candidate.inputs?.weight?.name === target.weight);
    if (!weight || !(weight.buffer instanceof Float32Array) || !node) {
      throw new Error(`Merge requires a compatible linear target '${target.weight}'.`);
    }
    const writableWeight = weight as Tensor & { buffer: Float32Array };
    for (let k = 0; k < target.din; k++) {
      for (let j = 0; j < target.dout; j++) {
        let lowRank = 0;
        for (let r = 0; r < target.rank; r++) {
          lowRank += target.A[k * target.rank + r] * target.B[r * target.dout + j];
        }
        const base = this._baseValue(writableWeight, node, k, j, target.din, target.dout, buffer);
        this._setBaseValue(writableWeight, node, buffer, k, j,
          base + target.scale * lowRank, target.din, target.dout);
      }
    }
  }

  _resolve(name: string | AdapterSelector, version?: AdapterVersion): AdapterSnapshot {
    if (typeof name === "object" && name) {
      version = name.version ?? name.versionId ?? name.version_id;
      name = name.name ?? name.adapterId ?? name.adapter_id ?? '';
    }
    const versions = this._versions.get(name);
    if (!versions || versions.size === 0) throw new Error(`Adapter '${name}' is not staged.`);
    const numeric = version == null ? Math.max(...versions.keys()) : Number(version);
    const snapshot = versions.get(numeric);
    if (!snapshot) throw new Error(`Adapter '${name}' version '${version}' is not staged.`);
    return snapshot;
  }

  _activateSnapshot(snapshot: AdapterSnapshot): void {
    if (this._merged) throw new Error(`Unmerge '${this._merged.snapshot.id}' before activating an adapter.`);
    this._active = snapshot;
  }

  _describe(snapshot: AdapterSnapshot): AdapterDescription {
    return Object.freeze({
      id: snapshot.id,
      name: snapshot.name,
      version: snapshot.version,
      sourceVersion: snapshot.sourceVersion,
      kind: snapshot.kind,
      metadata: snapshot.metadata,
      active: sameSnapshot(this._active, snapshot),
      merged: sameSnapshot(this._merged?.snapshot, snapshot),
      targets: Object.freeze(snapshot.targets.map((target) => Object.freeze({
        weight: target.weight,
        rank: target.rank,
        din: target.din,
        dout: target.dout,
        alpha: target.alpha,
        scale: target.scale,
        layout: "din_r_r_dout",
      }))),
    });
  }

  _assertBaseMutationAllowed(): void {
    if (this._merged) throw new Error(`Unmerge '${this._merged.snapshot.id}' before updating base tensors.`);
  }

  _assertGraphPatchAllowed(): void {
    if (this._versions.size) throw new Error("Remove all adapter versions before patching the graph.");
  }

  _pinSelector(
    selector: string | AdapterSelector | null | undefined,
    useActive = false,
  ): PinnedAdapterRoute | null {
    const snapshot = selector === undefined && useActive ? this._active : (selector == null ? null : this._resolve(selector));
    if (!snapshot) return null;
    const scale = typeof selector === "object" && selector ? selector.scale ?? 1 : 1;
    if (typeof scale !== "number" || !Number.isFinite(scale)) {
      throw new Error("Execution adapter scale must be finite.");
    }
    if (this._merged) {
      throw new Error(`Adapter '${snapshot.id}' cannot run unmerged while '${this._merged.snapshot.id}' is merged.`);
    }
    if (scale === 0) return null;
    return Object.freeze({ snapshot, scale });
  }

  _pinExecution(options: AdapterExecutionOptions = {}):
    | { readonly kind: 'single'; readonly route: PinnedAdapterRoute }
    | { readonly kind: 'batch'; readonly routes: readonly (PinnedAdapterRoute | null)[] }
    | null {
    if (Object.prototype.hasOwnProperty.call(options, "adapters")) {
      if (!Array.isArray(options.adapters)) throw new Error("Execution adapters must be an array.");
      if (options.adapters.length === 0) return null;
      const routes = Object.freeze(options.adapters.map((selector) => this._pinSelector(selector, false)));
      if (routes.length === 1) return routes[0] ? Object.freeze({ kind: "single", route: routes[0] }) : null;
      return Object.freeze({ kind: "batch", routes });
    }
    const selector = Object.prototype.hasOwnProperty.call(options, "adapter") ? options.adapter : undefined;
    const route = this._pinSelector(selector, true);
    return route ? Object.freeze({ kind: "single", route }) : null;
  }

  _markAcceleratedBackend(name: string): void {
    this._acceleratedBackends.add(name);
  }

  _registerResourceOwner(owner: AdapterResourceOwner): void {
    this._resourceOwners.add(owner);
  }

  _unregisterResourceOwner(owner: AdapterResourceOwner): void {
    this._resourceOwners.delete(owner);
  }

  _releaseSnapshotResources(snapshot: AdapterSnapshot): void {
    for (const owner of this._resourceOwners) owner.releaseAdapterTargets?.(snapshot.targets);
  }
}
