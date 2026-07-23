import { assertBuiltInEngine } from './BackendEngine.js';
import { Tensor } from '../core/Tensor.js';
import { ModelSnapshot, type ModelOutputDescriptor } from '../core/ModelSnapshot.js';
import { cloneRuntimeArray } from '../core/ExecutionResult.js';
import { VolvoxAIError, runtimeError } from '../core/RuntimeErrors.js';
import type { Graph } from '../core/Graph.js';
import type {
  BackendExecutionSnapshot,
  BackendTensorSnapshot,
} from '../core/ExecutionResult.js';
import type {
  BackendCapabilityContract,
  DecodeExecutionOptions,
  ExecutionInputs,
  ExecutionOptions,
  RuntimeDType,
  RuntimeTypedArray,
} from '../types.js';
import { memoryLocations } from '../generated/volvoxaiEnums.js';
import {
  runtimeSupportsOperator,
  type KernelBackend,
} from '../generated/kernelRegistry.js';
import type {
  DecodeRowModeValue,
  MemoryLocationValue,
  OperatorFallbackValue,
} from '../generated/volvoxaiEnums.js';

/** Version of the context-aware JavaScript provider composition SPI. */
export const VOLVOXAI_BACKEND_PROVIDER_VERSION = 1;

export type OperatorFallbackAttestation = 'none' | 'reported' | 'unknown';

const MEMORY_LOCATIONS = new Set<unknown>(memoryLocations);

export type BackendDeviceIdentity = Readonly<Record<string, string>>;

export interface BackendProviderCompilationEvidence {
  readonly device: BackendDeviceIdentity | null;
  readonly allocationBytes: number | null;
  readonly operatorFallbackUsed?: boolean | null;
  readonly offendingNode?: string | number | null;
}

/** Copy provider-owned device metadata into a stable serializable identity. */
export function createBackendDeviceIdentity(value: unknown): BackendDeviceIdentity | null {
  if (value == null) return null;
  if (Array.isArray(value) || typeof value !== 'object') {
    throw new VolvoxAIError('ABI_UNSUPPORTED', 'Backend device identity must be a plain object or null.', {
      phase: 'initialization',
    });
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new VolvoxAIError('ABI_UNSUPPORTED', 'Backend device identity must be a plain object or null.', {
      phase: 'initialization',
    });
  }
  const identity: Record<string, string> = {};
  for (const [key, candidate] of Object.entries(value as Record<string, unknown>)) {
    if (typeof candidate === 'string' && candidate.trim()) identity[key] = candidate.trim();
  }
  return Object.keys(identity).length ? Object.freeze(identity) : null;
}

export interface BackendProviderCapabilityOptions {
  operatorFallback?: OperatorFallbackAttestation;
  outputLocation?: MemoryLocationValue;
}

export interface BackendProviderCapabilities {
  readonly contextIsolation: true;
  readonly operatorFallback: OperatorFallbackAttestation;
  readonly outputLocation: MemoryLocationValue;
}

export function createBackendProviderCapabilities({
  operatorFallback = 'unknown',
  outputLocation = 'host',
}: BackendProviderCapabilityOptions = {}): Readonly<BackendProviderCapabilities> {
  if (!['none', 'reported', 'unknown'].includes(operatorFallback)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      `Invalid provider operatorFallback attestation '${operatorFallback}'.`, {
        phase: 'initialization',
      });
  }
  if (!MEMORY_LOCATIONS.has(outputLocation)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      `Invalid provider outputLocation '${outputLocation}'.`, {
        phase: 'initialization',
      });
  }
  return Object.freeze({ contextIsolation: true, operatorFallback, outputLocation });
}

export interface BackendProviderCompileOptions {
  readonly operatorFallback: OperatorFallbackValue;
}

/** Immutable model view supplied to one provider compilation. */
export interface BackendModelSnapshot {
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
  createExecutionGraph(): Graph;
}

export interface BackendProviderExecutionContext {
  readonly backendName: string;
  execute(inputs: ExecutionInputs, options?: ExecutionOptions): Promise<BackendExecutionSnapshot>;
  decodeSeed?(inputs: ExecutionInputs, options?: DecodeExecutionOptions): Promise<BackendExecutionSnapshot>;
  decodeStep?(inputs: ExecutionInputs, options?: DecodeExecutionOptions): Promise<BackendExecutionSnapshot>;
  decodeReset?(): Promise<void>;
  close(): Promise<void> | void;
}

export interface BackendProviderContextOptions extends ExecutionOptions {
  readonly decode?: Readonly<{
    changedInputs?: readonly string[] | null;
    rowMode?: DecodeRowModeValue;
    requireIncremental?: boolean;
  }>;
}

export interface BackendProviderCompiledModel {
  readonly backendName: string;
  readonly compilationEvidence?: Readonly<BackendProviderCompilationEvidence>;
  createContext(options?: BackendProviderContextOptions):
    Promise<BackendProviderExecutionContext> | BackendProviderExecutionContext;
  close(): Promise<void> | void;
}

export interface BackendProvider {
  readonly providerVersion: number;
  readonly backendName: string;
  readonly deviceIdentity?: BackendDeviceIdentity | null;
  readonly capabilities: Readonly<BackendProviderCapabilities>;
  compile(snapshot: BackendModelSnapshot, options: BackendProviderCompileOptions):
    Promise<BackendProviderCompiledModel> | BackendProviderCompiledModel;
  close(): Promise<void> | void;
}

export interface BackendProviderFactoryContext {
  readonly name: string;
  readonly runtime: unknown;
  readonly wasmUrl: string | URL;
}

export type BackendProviderFactory = (
  context: BackendProviderFactoryContext,
) => BackendProvider | null | Promise<BackendProvider | null>;

export function assertBackendProvider(value: unknown, label = 'Backend provider'): BackendProvider {
  const provider = value as Partial<BackendProvider> | null;
  const capabilities = provider?.capabilities;
  const validCapabilities = capabilities && Object.isFrozen(capabilities) &&
    capabilities.contextIsolation === true &&
    ['none', 'reported', 'unknown'].includes(capabilities.operatorFallback) &&
    MEMORY_LOCATIONS.has(capabilities.outputLocation);
  if (!provider || provider.providerVersion !== VOLVOXAI_BACKEND_PROVIDER_VERSION ||
      typeof provider.backendName !== 'string' || provider.backendName.length === 0 ||
      !validCapabilities || typeof provider.compile !== 'function' ||
      typeof provider.close !== 'function') {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `${label} must implement BackendProvider v${VOLVOXAI_BACKEND_PROVIDER_VERSION}.`, {
        phase: 'initialization',
      });
  }
  if (provider.deviceIdentity !== undefined) createBackendDeviceIdentity(provider.deviceIdentity);
  return provider as BackendProvider;
}

export function assertProviderCompiledModel(
  value: unknown,
  backendName: string,
): BackendProviderCompiledModel {
  const compiled = value as Partial<BackendProviderCompiledModel> | null;
  if (!compiled || compiled.backendName !== backendName ||
      typeof compiled.createContext !== 'function' || typeof compiled.close !== 'function') {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid compiled model.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  const evidence = compiled.compilationEvidence;
  if (evidence !== undefined) {
    if (!evidence || typeof evidence !== 'object' || Array.isArray(evidence) ||
        (evidence.allocationBytes !== null &&
          (!Number.isSafeInteger(evidence.allocationBytes) || evidence.allocationBytes < 0)) ||
        (evidence.operatorFallbackUsed !== undefined &&
          evidence.operatorFallbackUsed !== null &&
          typeof evidence.operatorFallbackUsed !== 'boolean') ||
        (evidence.offendingNode !== undefined && evidence.offendingNode !== null &&
          typeof evidence.offendingNode !== 'string' &&
          typeof evidence.offendingNode !== 'number')) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `Backend provider '${backendName}' returned invalid compilation evidence.`, {
          phase: 'compilation', backend: backendName,
        });
    }
    createBackendDeviceIdentity(evidence.device);
  }
  return compiled as BackendProviderCompiledModel;
}

export function assertProviderExecutionContext(
  value: unknown,
  backendName: string,
): BackendProviderExecutionContext {
  const context = value as Partial<BackendProviderExecutionContext> | null;
  if (!context || context.backendName !== backendName ||
      typeof context.execute !== 'function' || typeof context.close !== 'function') {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `Backend provider '${backendName}' returned an invalid execution context.`, {
        phase: 'compilation', backend: backendName,
      });
  }
  return context as BackendProviderExecutionContext;
}

type BuiltInEngine = {
  readonly backendName: string;
  readonly capabilities: BackendCapabilityContract;
  readonly adapterInfo?: Readonly<Record<string, string>> | null;
  readonly decodeCacheGeneration?: number;
  /**
   * Build a pointer-free backend execution blueprint once per CompiledModel.
   * Execution contexts may share this immutable object, but never their graph
   * tensors, device allocations, scratch storage, or decode state.
   */
  prepareGraph?(graph: Graph): Promise<Readonly<object>> | Readonly<object>;
  allocateGraph(
    graph: Graph,
    preparedGraph?: Readonly<object>,
  ): Promise<unknown> | unknown;
  execute(inputs: ExecutionInputs, options?: ExecutionOptions): Promise<unknown> | unknown;
  createDecodeSession(options?: ExecutionOptions): any;
  fork?(): Promise<BuiltInEngine> | BuiltInEngine;
  dispose?(): void;
  gpuBuffers?: Map<string, unknown>;
  readBuffer?(buffer: unknown, sizeBytes: number, dtype: RuntimeDType): Promise<RuntimeTypedArray>;
  executor?: {
    readBuffer?(buffer: GPUBuffer, sizeBytes: number, dtype: RuntimeDType): Promise<RuntimeTypedArray>;
  } | null;
  snapshotOutputs?(): ReadonlyMap<string, {
    readonly sizeBytes: number;
    readonly dtype: RuntimeDType;
    readonly deviceBuffer: GPUBuffer;
  }>;
};

function builtInRegistryBackend(backendName: string): KernelBackend | null {
  switch (backendName) {
    case 'cpu': return 'cpu-js';
    case 'wasm': return 'wasm';
    case 'webgpu': return 'webgpu';
    case 'webnn': return 'webnn';
    default: return null;
  }
}

function preflightBuiltIn(engine: BuiltInEngine, graph: Graph): void {
  const registryBackend = builtInRegistryBackend(engine.backendName);
  if (!registryBackend) return;
  for (const node of graph.nodes) {
    if (!runtimeSupportsOperator(registryBackend, node.opType)) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `Backend '${engine.backendName}' does not support operator '${node.opType}' at node '${String(node.id)}'.`, {
          phase: 'compilation', backend: engine.backendName, node: node.id,
        });
    }
  }
}

function typedArrayFromBuffer(
  dtype: RuntimeDType,
  value: ArrayBuffer,
): RuntimeTypedArray {
  const copy = value.slice(0);
  if (dtype === 'float32') return new Float32Array(copy);
  if (dtype === 'int32') return new Int32Array(copy);
  if (dtype === 'int8') return new Int8Array(copy);
  return new Uint8Array(copy);
}

function hostOutput(
  value: unknown,
  dtype: RuntimeDType,
  sizeBytes: number,
  label: string,
): RuntimeTypedArray | null {
  if (value instanceof ArrayBuffer) {
    Tensor.assertCompatibleBuffer(dtype, value, sizeBytes, label);
    return typedArrayFromBuffer(dtype, value);
  }
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    Tensor.assertCompatibleBuffer(dtype, value, sizeBytes, label);
    return cloneRuntimeArray(value as RuntimeTypedArray);
  }
  return null;
}

function resultValue(result: unknown, name: string): unknown {
  if (result instanceof Map) return result.get(name);
  if (result && typeof result === 'object') return (result as Record<string, unknown>)[name];
  return undefined;
}

class BuiltInExecutionContext implements BackendProviderExecutionContext {
  readonly backendName: string;
  #engine: BuiltInEngine | null;
  #graph: Graph | null;
  #decode: any | null;
  readonly #deviceIdentity: BackendDeviceIdentity | null;
  #closed = false;

  constructor(
    engine: BuiltInEngine,
    graph: Graph,
    deviceIdentity: BackendDeviceIdentity | null,
    options: BackendProviderContextOptions = {},
  ) {
    this.backendName = engine.backendName;
    this.#engine = engine;
    this.#graph = graph;
    const decodeOptions = options.decode && typeof options.decode === 'object' &&
        !Array.isArray(options.decode)
      ? options.decode
      : {};
    this.#decode = engine.createDecodeSession(decodeOptions as ExecutionOptions);
    this.#deviceIdentity = deviceIdentity;
  }

  async #capture(
    result: unknown,
    operation: 'execute' | 'seed' | 'step',
    options: DecodeExecutionOptions,
    engine: BuiltInEngine,
    graph: Graph,
    decode: any,
  ): Promise<BackendExecutionSnapshot> {
    const outputs: BackendTensorSnapshot[] = [];
    const deviceSnapshots = engine.snapshotOutputs?.() || null;
    try {
      for (const name of graph.outputNames) {
        const tensor = graph.tensors.get(name);
        if (!tensor) {
          throw new VolvoxAIError('EXECUTION_FAILED',
            `Compiled graph no longer contains declared output '${name}'.`, {
              phase: 'execution', backend: this.backendName,
            });
        }
        const deviceSnapshot = deviceSnapshots?.get(name);
        if (deviceSnapshot) {
          const executor = engine.executor;
          const reader = typeof executor?.readBuffer === 'function'
            ? executor.readBuffer.bind(executor)
            : typeof engine.readBuffer === 'function'
              ? engine.readBuffer.bind(engine)
              : null;
          if (!reader) {
            throw new VolvoxAIError('EXECUTION_FAILED',
              `Device backend '${this.backendName}' cannot read output '${name}'.`, {
                phase: 'readback', backend: this.backendName,
              });
          }
          let released = false;
          outputs.push(Object.freeze({
            name,
            shape: Object.freeze([...tensor.shape]),
            dtype: tensor.dtype,
            location: 'device' as const,
            deviceBuffer: deviceSnapshot.deviceBuffer,
            read: () => reader(
              deviceSnapshot.deviceBuffer,
              deviceSnapshot.sizeBytes,
              deviceSnapshot.dtype,
            ),
            release: () => {
              if (released) return;
              released = true;
              deviceSnapshot.deviceBuffer.destroy?.();
            },
          }));
          continue;
        }

        let data = hostOutput(resultValue(result, name), tensor.dtype, tensor.sizeBytes,
          `Backend output '${name}'`);
        if (!data) {
          data = hostOutput(tensor.buffer, tensor.dtype, tensor.sizeBytes, `Backend output '${name}'`);
        }
        if (!data && engine.gpuBuffers && typeof engine.readBuffer === 'function') {
          const buffer = engine.gpuBuffers.get(name);
          if (buffer) {
            data = hostOutput(
              await engine.readBuffer(buffer, tensor.sizeBytes, tensor.dtype),
              tensor.dtype,
              tensor.sizeBytes,
              `Backend output '${name}'`,
            );
          }
        }
        if (!data) {
          throw new VolvoxAIError('EXECUTION_FAILED',
            `Backend '${this.backendName}' did not publish declared output '${name}'.`, {
              phase: 'execution', backend: this.backendName,
            });
        }
        outputs.push(Object.freeze({
          name,
          shape: Object.freeze([...tensor.shape]),
          dtype: tensor.dtype,
          location: 'host' as const,
          data,
        }));
      }
    } catch (error) {
      for (const snapshot of deviceSnapshots?.values() || []) {
        snapshot.deviceBuffer.destroy?.();
      }
      throw error;
    }
    const cacheGeneration = Number.isSafeInteger(engine.decodeCacheGeneration)
      ? engine.decodeCacheGeneration!
      : null;
    const position = operation === 'step' && Number.isSafeInteger(options.position)
      ? options.position as number
      : null;
    return Object.freeze({
      outputs: Object.freeze(outputs),
      backendReport: Object.freeze({
        route: Object.freeze({
          backend: this.backendName,
          operatorFallbackUsed: false,
          offendingNode: null,
        }),
        device: this.#deviceIdentity,
        decode: Object.freeze({
          operation,
          mode: operation === 'execute'
            ? null
            : decode.lastExecutionMode || decode.mode || null,
          cacheState: operation === 'seed'
            ? 'seeded'
            : operation === 'step' ? 'advanced' : 'not-applicable',
          cacheGeneration,
          position,
        }),
      }),
    });
  }

  async execute(inputs: ExecutionInputs, options: ExecutionOptions = {}): Promise<BackendExecutionSnapshot> {
    const { engine, graph, decode } = this.#openResources();
    try {
      return await this.#capture(
        await engine.execute(inputs, options), 'execute', options, engine, graph, decode,
      );
    } catch (error) {
      throw runtimeError(error, 'EXECUTION_FAILED',
        `Backend '${this.backendName}' execution failed.`, {
          phase: 'execution', backend: this.backendName,
        });
    }
  }

  async decodeSeed(inputs: ExecutionInputs, options: DecodeExecutionOptions = {}): Promise<BackendExecutionSnapshot> {
    const { engine, graph, decode } = this.#openResources();
    return this.#capture(
      await decode.seed(inputs, options), 'seed', options, engine, graph, decode,
    );
  }

  async decodeStep(inputs: ExecutionInputs, options: DecodeExecutionOptions = {}): Promise<BackendExecutionSnapshot> {
    const { engine, graph, decode } = this.#openResources();
    return this.#capture(
      await decode.step(inputs, options), 'step', options, engine, graph, decode,
    );
  }

  async decodeReset(): Promise<void> {
    const { decode } = this.#openResources();
    await decode.reset();
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    let engine = this.#engine;
    let decode = this.#decode;
    this.#engine = null;
    this.#graph = null;
    this.#decode = null;
    try {
      await decode?.close();
    } finally {
      engine?.dispose?.();
      engine = null;
      decode = null;
    }
  }

  #openResources(): { engine: BuiltInEngine; graph: Graph; decode: any } {
    const engine = this.#engine;
    const graph = this.#graph;
    const decode = this.#decode;
    if (this.#closed || !engine || !graph || !decode) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Backend execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    return { engine, graph, decode };
  }
}

class BuiltInCompiledModel implements BackendProviderCompiledModel {
  readonly backendName: string;
  readonly compilationEvidence: Readonly<BackendProviderCompilationEvidence>;
  readonly #source: BuiltInEngine;
  readonly #snapshot: ModelSnapshot;
  readonly #preparedGraph: Readonly<object> | null;
  #seed: { engine: BuiltInEngine; graph: Graph } | null;
  #closed = false;

  constructor(
    source: BuiltInEngine,
    snapshot: ModelSnapshot,
    preparedGraph: Readonly<object> | null,
    seed: { engine: BuiltInEngine; graph: Graph },
    deviceIdentity: BackendDeviceIdentity | null,
    allocationBytes: number | null,
  ) {
    this.backendName = source.backendName;
    this.#source = source;
    this.#snapshot = snapshot;
    this.#preparedGraph = preparedGraph;
    this.#seed = seed;
    this.compilationEvidence = Object.freeze({
      device: deviceIdentity,
      allocationBytes,
      operatorFallbackUsed: false,
      offendingNode: null,
    });
  }

  async createContext(options: BackendProviderContextOptions = {}): Promise<BackendProviderExecutionContext> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Compiled backend model is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    const seed = this.#seed;
    if (seed) {
      this.#seed = null;
      return new BuiltInExecutionContext(
        seed.engine,
        seed.graph,
        this.compilationEvidence.device,
        options,
      );
    }
    if (typeof this.#source.fork !== 'function') {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `Backend '${this.backendName}' cannot create an independent context.`, {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const graph = this.#snapshot.createExecutionGraph();
    let engine: BuiltInEngine | null = null;
    try {
      engine = assertBuiltInEngine(await this.#source.fork(),
        `Built-in '${this.backendName}' context`) as BuiltInEngine;
      preflightBuiltIn(engine, graph);
      if (this.#preparedGraph) {
        await engine.allocateGraph(graph, this.#preparedGraph);
      } else {
        await engine.allocateGraph(graph);
      }
      return new BuiltInExecutionContext(
        engine,
        graph,
        this.compilationEvidence.device,
        options,
      );
    } catch (error) {
      engine?.dispose?.();
      throw runtimeError(error, 'BACKEND_UNSUPPORTED',
        `Backend '${this.backendName}' could not create a context.`, {
          phase: 'compilation', backend: this.backendName,
        });
    }
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    this.#seed?.engine.dispose?.();
    this.#seed = null;
  }
}

/** Internal context-aware provider for a built-in engine implementation. */
export class BuiltInBackendProvider implements BackendProvider {
  readonly providerVersion = VOLVOXAI_BACKEND_PROVIDER_VERSION;
  readonly backendName: string;
  readonly capabilities: Readonly<BackendProviderCapabilities>;
  readonly deviceIdentity: BackendDeviceIdentity | null;
  readonly #source: BuiltInEngine;
  #closed = false;

  constructor(source: BuiltInEngine) {
    this.#source = assertBuiltInEngine(source, 'Built-in backend provider source') as BuiltInEngine;
    this.backendName = this.#source.backendName;
    this.capabilities = createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: this.#source.capabilities.outputLocation,
    });
    this.deviceIdentity = createBackendDeviceIdentity(this.#source.adapterInfo) ||
      Object.freeze({
        backend: this.backendName,
        device: this.backendName === 'cpu' ? 'host' : this.backendName,
      });
  }

  async compile(
    snapshot: BackendModelSnapshot,
    options: BackendProviderCompileOptions,
  ): Promise<BackendProviderCompiledModel> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', `Backend provider '${this.backendName}' is closed.`, {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (!(snapshot instanceof ModelSnapshot)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'Backend provider compile requires a ModelSnapshot.', {
        phase: 'compilation', backend: this.backendName,
      });
    }
    if (options.operatorFallback === 'forbid' &&
        this.capabilities.operatorFallback !== 'none') {
      throw new VolvoxAIError('OPERATOR_FALLBACK_FORBIDDEN',
        `Backend '${this.backendName}' cannot attest strict operator routing.`, {
          phase: 'compilation', backend: this.backendName,
        });
    }
    if (typeof this.#source.fork !== 'function') {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `Backend '${this.backendName}' has no independent context factory.`, {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const graph = snapshot.createExecutionGraph();
    let preparedGraph: Readonly<object> | null = null;
    let engine: BuiltInEngine | null = null;
    try {
      if (typeof this.#source.prepareGraph === 'function') {
        const candidate = await this.#source.prepareGraph(graph);
        if (!candidate || typeof candidate !== 'object' || !Object.isFrozen(candidate)) {
          throw new VolvoxAIError('ABI_UNSUPPORTED',
            `Built-in '${this.backendName}' returned a mutable prepared graph.`, {
              phase: 'compilation', backend: this.backendName,
            });
        }
        preparedGraph = candidate;
      }
      engine = assertBuiltInEngine(await this.#source.fork(),
        `Built-in '${this.backendName}' provider`) as BuiltInEngine;
      preflightBuiltIn(engine, graph);
      if (preparedGraph) {
        await engine.allocateGraph(graph, preparedGraph);
      } else {
        await engine.allocateGraph(graph);
      }
      let allocationBytes = 0;
      let countedBuffers = false;
      const buffers = engine.gpuBuffers;
      if (buffers instanceof Map && buffers.size > 0) {
        const seen = new Set<object>();
        for (const buffer of buffers.values()) {
          if (!buffer || typeof buffer !== 'object' || seen.has(buffer)) continue;
          seen.add(buffer);
          const size = (buffer as { size?: unknown }).size;
          if (Number.isSafeInteger(size) && (size as number) >= 0) {
            allocationBytes += size as number;
            countedBuffers = true;
          }
        }
      }
      if (!countedBuffers) {
        allocationBytes = 0;
        for (const tensor of graph.tensors.values()) {
          if (Number.isSafeInteger(tensor.sizeBytes) && tensor.sizeBytes >= 0) {
            allocationBytes += tensor.sizeBytes;
          }
        }
      }
      return new BuiltInCompiledModel(
        this.#source,
        snapshot,
        preparedGraph,
        { engine, graph },
        this.deviceIdentity,
        Number.isSafeInteger(allocationBytes) ? allocationBytes : null,
      );
    } catch (error) {
      engine?.dispose?.();
      throw runtimeError(error, 'BACKEND_UNSUPPORTED',
        `Backend '${this.backendName}' could not compile the model.`, {
          phase: 'compilation', backend: this.backendName,
      });
    }
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    this.#source.dispose?.();
  }
}
