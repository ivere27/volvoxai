import { Tensor } from './Tensor.js';
import { runtimeIdentity } from './Identity.js';
import { VolvoxAIError } from './RuntimeErrors.js';
import type { ResolvedTensorDescriptor } from './ResolvedShapePlan.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import { memoryLocations, runtimeDTypes } from '../generated/volvoxaiEnums.js';
import type { MemoryLocationValue } from '../generated/volvoxaiEnums.js';
import {
  DEVICE_TENSOR_REFERENCE_BRAND,
  registerDeviceTensorReference,
  type DeviceTensorReference,
} from '../ops/deviceTensorReference.js';

const RUNTIME_DTYPES = new Set<unknown>(runtimeDTypes);
const MEMORY_LOCATIONS = new Set<unknown>(memoryLocations);
const INTERNAL_RESULT_CONSTRUCTION_TOKEN = Symbol('volvoxai.internal-result-construction');

interface BackendTensorSnapshotBase {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: RuntimeDType;
}

export interface BackendHostTensorSnapshot extends BackendTensorSnapshotBase {
  readonly location: 'host';
  /** `transfer` hands exact storage to the result; `borrowed` is cloned once. */
  readonly ownership: 'transfer' | 'borrowed';
  readonly data: RuntimeTypedArray;
}

export interface BackendDeviceTensorSnapshot extends BackendTensorSnapshotBase {
  readonly location: 'device';
  /** Exact logical bytes; deviceBuffer may include only mandatory API alignment padding. */
  readonly logicalSizeBytes: number;
  readonly deviceBuffer: GPUBuffer;
  /** Present only when this snapshot can be handed to the built-in WebGPU provider. */
  readonly deviceType?: 'webgpu';
  /** Exact physical owner used for same-device input validation. */
  readonly device?: GPUDevice;
  read(): Promise<RuntimeTypedArray>;
  release(): void;
}

export type BackendTensorSnapshot = BackendHostTensorSnapshot | BackendDeviceTensorSnapshot;

export interface BackendExecutionSnapshot {
  readonly outputs: readonly BackendTensorSnapshot[];
  readonly backendReport?: Readonly<Record<string, unknown>> | null;
}

export interface OperatorRouteEvidence {
  readonly attestation: 'none' | 'reported' | 'unknown';
  readonly used: boolean | null;
  readonly offendingNode: string | number | null;
}

export interface ExecutionRouteEvidence {
  readonly tierFallback: boolean;
  readonly operator: OperatorRouteEvidence;
}

export interface ExecutionDecodeState {
  readonly operation: 'execute' | 'seed' | 'step';
  readonly mode: string | null;
  readonly cacheState: 'not-applicable' | 'seeded' | 'advanced';
  readonly cacheGeneration: number | null;
  readonly position: number | null;
  readonly activeSequenceLength: number | null;
  readonly kvCapacity: number | null;
  readonly kvCapacityClass: number | null;
  readonly semanticSeedSignature: string | null;
  readonly automaticReset: boolean;
  readonly automaticResetReason: string | null;
  readonly automaticResetCount: number;
}

/** @internal Release an unclaimed provider snapshot after validation or construction fails. */
export function releaseBackendExecutionSnapshot(snapshot: unknown): void {
  const outputs = (snapshot as { outputs?: unknown } | null)?.outputs;
  if (!Array.isArray(outputs)) return;
  const released = new Set<object>();
  for (const candidate of outputs) {
    if (!candidate || typeof candidate !== 'object' || released.has(candidate)) continue;
    const output = candidate as Partial<BackendDeviceTensorSnapshot>;
    if (output.location !== 'device' || typeof output.release !== 'function') continue;
    released.add(candidate);
    try {
      output.release.call(candidate);
    } catch {
      // Cleanup is best-effort and must not replace the original execution failure.
    }
  }
}

export interface ExecutionReport {
  readonly executionId: string;
  readonly contextId: string;
  readonly backend: string;
  readonly device: Readonly<Record<string, string>> | null;
  readonly outcome: 'success';
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly shapeSignature: string;
  readonly adapterRevisionId: string | null;
  readonly adapterRevisionIds: readonly (string | null)[];
  /** Complete logical graph binding only; excludes provider specialization and dispatch. */
  readonly shapeBindTimeMs: number;
  /** Provider call including any cold specialization, dispatch, and result snapshot staging. */
  readonly providerTimeMs: number;
  readonly executionTimeMs: number;
  readonly routeEvidence: ExecutionRouteEvidence;
  readonly decodeState: ExecutionDecodeState;
  readonly operatorFallback: 'none' | 'reported' | 'unknown';
  readonly backendReport: Readonly<Record<string, unknown>> | null;
}

function cloneSerializable(
  value: unknown,
  path: string,
  visiting: Set<object>,
): unknown {
  if (value == null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (Number.isFinite(value)) return value;
    throw new VolvoxAIError('EXECUTION_FAILED', `${path} contains a non-finite number.`, {
      phase: 'execution',
    });
  }
  if (typeof value !== 'object') {
    throw new VolvoxAIError('EXECUTION_FAILED', `${path} is not JSON-serializable.`, {
      phase: 'execution',
    });
  }
  if (visiting.has(value)) {
    throw new VolvoxAIError('EXECUTION_FAILED', `${path} contains a cycle.`, {
      phase: 'execution',
    });
  }
  visiting.add(value);
  try {
    if (Array.isArray(value)) {
      return Object.freeze(value.map((entry, index) =>
        cloneSerializable(entry, `${path}[${index}]`, visiting)));
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new VolvoxAIError('EXECUTION_FAILED', `${path} must contain only plain objects.`, {
        phase: 'execution',
      });
    }
    return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, entry]) => [
      key,
      cloneSerializable(entry, `${path}.${key}`, visiting),
    ])));
  } finally {
    visiting.delete(value);
  }
}

/** @internal Validate, copy, and freeze provider evidence before publishing a report. */
export function normalizeBackendReport(
  value: Readonly<Record<string, unknown>> | null | undefined,
): Readonly<Record<string, unknown>> | null {
  if (value == null) return null;
  if (Array.isArray(value) || typeof value !== 'object') {
    throw new VolvoxAIError('EXECUTION_FAILED', 'Backend report must be a plain object or null.', {
      phase: 'execution',
    });
  }
  return cloneSerializable(value, 'Backend report', new Set()) as Readonly<Record<string, unknown>>;
}

export function executionIdentity(): string {
  return runtimeIdentity('execution');
}

export function cloneRuntimeArray(value: RuntimeTypedArray): RuntimeTypedArray {
  if (value instanceof Float32Array) return new Float32Array(value);
  if (value instanceof Int32Array) return new Int32Array(value);
  if (value instanceof Int8Array) return new Int8Array(value);
  if (value instanceof Uint8ClampedArray) return new Uint8ClampedArray(value);
  return new Uint8Array(value);
}

function expectedBytes(snapshot: BackendTensorSnapshot): number {
  let elements = 1;
  if (!Array.isArray(snapshot.shape)) {
    throw new VolvoxAIError('EXECUTION_FAILED', `Result tensor '${snapshot.name}' has no shape.`, {
      phase: 'execution',
    });
  }
  for (const [index, dimension] of snapshot.shape.entries()) {
    if (!Number.isSafeInteger(dimension) || dimension <= 0 ||
        !Number.isSafeInteger(elements * dimension)) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        `Result tensor '${snapshot.name}' has invalid shape dimension ${index}.`, {
          phase: 'execution',
        });
    }
    elements *= dimension;
  }
  const bytes = elements * Tensor.dtypeBytes(snapshot.dtype);
  if (!Number.isSafeInteger(bytes)) {
    throw new VolvoxAIError('EXECUTION_FAILED', `Result tensor '${snapshot.name}' is too large.`, {
      phase: 'execution',
    });
  }
  return bytes;
}

function alignedDeviceBytes(logicalBytes: number, name: string): number {
  if (logicalBytes > Number.MAX_SAFE_INTEGER - 3) {
    throw new VolvoxAIError('EXECUTION_FAILED',
      `Result tensor '${name}' cannot be aligned safely for device storage.`, {
        phase: 'execution',
      });
  }
  return Math.ceil(logicalBytes / 4) * 4;
}

function validateSnapshotOutput(value: unknown): BackendTensorSnapshot {
  const output = value as Partial<BackendTensorSnapshot> | null;
  if (!output || typeof output !== 'object' ||
      typeof output.name !== 'string' || output.name.length === 0 ||
      !Array.isArray(output.shape) ||
      !RUNTIME_DTYPES.has(output.dtype) ||
      !MEMORY_LOCATIONS.has(output.location)) {
    throw new VolvoxAIError('EXECUTION_FAILED', 'Backend returned an invalid result tensor descriptor.', {
      phase: 'execution',
    });
  }
  expectedBytes(output as BackendTensorSnapshot);
  if (output.location === 'host') {
    if ((output.ownership !== 'transfer' && output.ownership !== 'borrowed') ||
        !ArrayBuffer.isView(output.data) || output.data instanceof DataView) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        `Host result tensor '${output.name}' has no typed-array data.`, {
          phase: 'execution',
        });
    }
  } else {
    const deviceOutput = output as Partial<BackendDeviceTensorSnapshot>;
    if (!deviceOutput.deviceBuffer || typeof deviceOutput.deviceBuffer.size !== 'number' ||
        !Number.isSafeInteger(deviceOutput.logicalSizeBytes) ||
        typeof deviceOutput.read !== 'function' || typeof deviceOutput.release !== 'function') {
      throw new VolvoxAIError('EXECUTION_FAILED',
        `Device result tensor '${output.name}' has an invalid ownership contract.`, {
          phase: 'execution',
        });
    }
    const hasDeviceType = Object.prototype.hasOwnProperty.call(deviceOutput, 'deviceType');
    const hasDevice = Object.prototype.hasOwnProperty.call(deviceOutput, 'device');
    if (hasDeviceType !== hasDevice ||
        (hasDeviceType && (deviceOutput.deviceType !== 'webgpu' ||
          !deviceOutput.device || typeof deviceOutput.device !== 'object'))) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        `Device result tensor '${output.name}' has invalid handoff ownership metadata.`, {
          phase: 'execution',
        });
    }
  }
  return output as BackendTensorSnapshot;
}

function assertHostSnapshot(snapshot: BackendHostTensorSnapshot): RuntimeTypedArray {
  const sizeBytes = expectedBytes(snapshot);
  Tensor.assertCompatibleBuffer(snapshot.dtype, snapshot.data, sizeBytes,
    `Result tensor '${snapshot.name}'`);
  // The built-in CPU has already created an exact owned clone and transfers
  // it without a second full-output copy. External providers may retain their
  // storage by declaring it borrowed, in which case the result snapshots once.
  return snapshot.ownership === 'transfer'
    ? snapshot.data
    : cloneRuntimeArray(snapshot.data);
}

export class TensorResult implements DeviceTensorReference {
  readonly [DEVICE_TENSOR_REFERENCE_BRAND] = true as const;
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: RuntimeDType;
  readonly location: MemoryLocationValue;
  readonly logicalSizeBytes: number;
  readonly deviceBuffer?: GPUBuffer;
  #data: RuntimeTypedArray | null;
  readonly #deviceRead: (() => Promise<RuntimeTypedArray>) | null;
  readonly #release: (() => void) | null;
  readonly #isDisposed: () => boolean;
  #tail: Promise<void> = Promise.resolve();
  #released = false;
  #deviceInputLeases = 0;
  #deviceInputDrain: Promise<void> | null = null;
  #resolveDeviceInputDrain: (() => void) | null = null;

  constructor(
    snapshot: BackendTensorSnapshot,
    isDisposed: () => boolean,
    constructionToken?: symbol,
  ) {
    this.name = snapshot.name;
    this.shape = Object.freeze([...snapshot.shape]);
    this.dtype = snapshot.dtype;
    this.location = snapshot.location;
    const logicalBytes = expectedBytes(snapshot);
    this.logicalSizeBytes = logicalBytes;
    if (snapshot.location === 'device' &&
        (snapshot.logicalSizeBytes !== logicalBytes ||
          snapshot.deviceBuffer.size !== alignedDeviceBytes(logicalBytes, snapshot.name))) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        `Device result tensor '${snapshot.name}' must expose its exact logical length with only required alignment padding.`, {
          phase: 'execution',
        });
    }
    this.#data = snapshot.location === 'host' ? assertHostSnapshot(snapshot) : null;
    this.#deviceRead = snapshot.location === 'device' ? snapshot.read.bind(snapshot) : null;
    this.#release = snapshot.location === 'device' ? snapshot.release.bind(snapshot) : null;
    if (snapshot.location === 'device') {
      this.deviceBuffer = snapshot.deviceBuffer;
      // TensorResult is a public runtime value for compatibility, but only a
      // result constructed by the runtime may issue an opaque device input.
      // A caller-created result still owns/readbacks its supplied snapshot; it
      // cannot mint a registry entry for an arbitrary raw GPUBuffer.
      if (constructionToken === INTERNAL_RESULT_CONSTRUCTION_TOKEN &&
          snapshot.deviceType === 'webgpu' && snapshot.device) {
        registerDeviceTensorReference(this, {
          kind: 'webgpu',
          owner: snapshot.device,
          resource: snapshot.deviceBuffer,
          dtype: snapshot.dtype,
          shape: this.shape,
          logicalSizeBytes: logicalBytes,
          acquire: () => this.#acquireDeviceInputLease(),
        });
      }
    }
    this.#isDisposed = isDisposed;
  }

  #acquireDeviceInputLease(): () => void {
    if (this.#isDisposed() || this.#released) {
      throw new VolvoxAIError('RESULT_DISPOSED',
        `Result tensor '${this.name}' is disposed and cannot be used as device input.`, {
          phase: 'execution',
        });
    }
    this.#deviceInputLeases++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.#deviceInputLeases--;
      if (this.#deviceInputLeases === 0 && this.#resolveDeviceInputDrain) {
        const resolve = this.#resolveDeviceInputDrain;
        this.#resolveDeviceInputDrain = null;
        this.#deviceInputDrain = null;
        resolve();
      }
    };
  }

  #drainDeviceInputLeases(): Promise<void> {
    if (this.#deviceInputLeases === 0) return Promise.resolve();
    if (!this.#deviceInputDrain) {
      this.#deviceInputDrain = new Promise((resolve) => {
        this.#resolveDeviceInputDrain = resolve;
      });
    }
    return this.#deviceInputDrain;
  }

  read(): Promise<RuntimeTypedArray> {
    if (this.#isDisposed()) {
      return Promise.reject(new VolvoxAIError('RESULT_DISPOSED', `Result tensor '${this.name}' is disposed.`, {
        phase: 'readback',
      }));
    }
    // Host snapshots are already immutable result-owned storage. Clone them before
    // returning so an accepted read no longer depends on the result lifecycle, and
    // avoid retaining each caller-owned clone in the device-read drain chain.
    if (this.#data) return Promise.resolve(cloneRuntimeArray(this.#data));
    const operation = async (): Promise<RuntimeTypedArray> => {
      const data = await this.#deviceRead!();
      Tensor.assertCompatibleBuffer(this.dtype, data, expectedBytes({
        name: this.name,
        shape: this.shape,
        dtype: this.dtype,
        location: 'host',
        ownership: 'transfer',
        data,
      }), `Result tensor '${this.name}' readback`);
      return cloneRuntimeArray(data);
    };
    const pending = this.#tail.then(operation, operation);
    this.#tail = pending.then(() => undefined, () => undefined);
    return pending;
  }

  /** @internal Drain accepted reads before releasing result-owned storage. */
  async _close(): Promise<void> {
    await this.#tail;
    await this.#drainDeviceInputLeases();
    if (this.#released) return;
    this.#released = true;
    this.#data = null;
    this.#release?.();
  }
}

class ResultOutputMap implements ReadonlyMap<string, TensorResult> {
  readonly #outputs: ReadonlyMap<string, TensorResult>;

  constructor(outputs: ReadonlyMap<string, TensorResult>) {
    this.#outputs = outputs;
    Object.freeze(this);
  }

  get size(): number { return this.#outputs.size; }
  get(name: string): TensorResult | undefined { return this.#outputs.get(name); }
  has(name: string): boolean { return this.#outputs.has(name); }
  entries(): MapIterator<[string, TensorResult]> { return this.#outputs.entries(); }
  keys(): MapIterator<string> { return this.#outputs.keys(); }
  values(): MapIterator<TensorResult> { return this.#outputs.values(); }
  forEach(
    callback: (value: TensorResult, key: string, map: ReadonlyMap<string, TensorResult>) => void,
    thisArg?: unknown,
  ): void {
    for (const [name, output] of this.#outputs) callback.call(thisArg, output, name, this);
  }
  [Symbol.iterator](): MapIterator<[string, TensorResult]> { return this.entries(); }
}

/** Stable host/device snapshot owned independently from its execution context. */
export class ExecutionResult {
  readonly backend: string;
  readonly report: ExecutionReport;
  readonly outputs: ReadonlyMap<string, TensorResult>;
  readonly #ownedOutputs: ReadonlyMap<string, TensorResult>;
  #state: 'open' | 'closing' | 'closed' = 'open';
  #closePromise: Promise<void> | null = null;

  /** @internal Provider snapshots transfer ownership only after this constructor succeeds. */
  constructor(
    backend: string,
    snapshot: BackendExecutionSnapshot,
    report: ExecutionReport,
    expectedOutputs: readonly ResolvedTensorDescriptor[],
    constructionToken?: symbol,
  ) {
    this.backend = backend;
    this.report = Object.freeze({ ...report });
    if (!snapshot || typeof snapshot !== 'object' || !Array.isArray(snapshot.outputs)) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        `Backend '${backend}' returned an invalid execution snapshot.`, {
          phase: 'execution', backend,
        });
    }
    const validated = snapshot.outputs.map(validateSnapshotOutput);
    const outputsByName = new Map<string, BackendTensorSnapshot>();
    for (const output of validated) {
      if (outputsByName.has(output.name)) {
        throw new VolvoxAIError('EXECUTION_FAILED', `Backend '${backend}' returned output '${output.name}' twice.`, {
          phase: 'execution', backend,
        });
      }
      outputsByName.set(output.name, output);
    }
    if (outputsByName.size !== expectedOutputs.length ||
        expectedOutputs.some(({ name }) => !outputsByName.has(name))) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        `Backend '${backend}' did not return every declared graph output.`, {
          phase: 'execution', backend,
        });
    }
    for (const expected of expectedOutputs) {
      const output = outputsByName.get(expected.name)!;
      const sameShape = output.shape.length === expected.shape.length &&
        expected.shape.every((dimension, index) => output.shape[index] === dimension);
      if (output.dtype !== expected.dtype || !sameShape) {
        throw new VolvoxAIError('EXECUTION_FAILED',
          `Backend '${backend}' output '${expected.name}' does not match its declared dtype and shape.`, {
            phase: 'execution', backend,
          });
      }
    }
    const outputs = new Map<string, TensorResult>();
    for (const expected of expectedOutputs) {
      const tensor = outputsByName.get(expected.name)!;
      outputs.set(tensor.name, new TensorResult(
        tensor,
        () => this.#state !== 'open',
        constructionToken === INTERNAL_RESULT_CONSTRUCTION_TOKEN
          ? INTERNAL_RESULT_CONSTRUCTION_TOKEN
          : undefined,
      ));
    }
    this.#ownedOutputs = outputs;
    this.outputs = new ResultOutputMap(outputs);
  }

  get closed(): boolean {
    return this.#state === 'closed';
  }

  output(name: string): TensorResult {
    if (this.#state !== 'open') {
      throw new VolvoxAIError('RESULT_DISPOSED', 'ExecutionResult is disposed.', {
        phase: 'readback', backend: this.backend,
      });
    }
    const output = this.#ownedOutputs.get(name);
    if (!output) {
      throw new VolvoxAIError('INVALID_ARGUMENT', `Execution result has no output named '${name}'.`, {
        phase: 'readback', backend: this.backend,
      });
    }
    return output;
  }

  close(): Promise<void> {
    if (this.#closePromise) return this.#closePromise;
    this.#state = 'closing';
    this.#closePromise = Promise.all(
      Array.from(this.#ownedOutputs.values(), (output) => output._close()),
    ).then(() => {
      this.#state = 'closed';
    }, (error) => {
      this.#state = 'closed';
      throw error;
    });
    return this.#closePromise;
  }

  dispose(): Promise<void> {
    return this.close();
  }
}

/** @internal Construct the only ExecutionResult allowed to issue device inputs. */
export function createExecutionResult(
  backend: string,
  snapshot: BackendExecutionSnapshot,
  report: ExecutionReport,
  expectedOutputs: readonly ResolvedTensorDescriptor[],
): ExecutionResult {
  return new ExecutionResult(
    backend,
    snapshot,
    report,
    expectedOutputs,
    INTERNAL_RESULT_CONSTRUCTION_TOKEN,
  );
}
