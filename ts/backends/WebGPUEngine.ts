import { BackendEngine } from './BackendEngine.js';
import { GraphExecutor } from './GraphExecutor.js';
import { preflightWebGPUExecutionInputs } from './WebGPUDispatch.js';
import { WebGPUDeviceState } from './WebGPUDeviceState.js';
import type { WebGPUOutputSnapshot } from './WebGPUResults.js';
import type {
  DeviceFeedbackDecodeOptions,
  GraphExecutorOptions,
  WebGPURebindOptions,
  WebGPUResourceInspection,
  WebGPUExecutionInputs,
  WebGPUExecutionOptions,
} from './GraphExecutor.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import type { ExecutorGraph } from './WebGPUContracts.js';

export interface WebGPUEngineOptions {
  shaderLibrary?: GraphExecutorOptions['shaderLibrary'];
  adapterInfo?: WebGPUAdapterIdentity | null;
  /** @internal Shared model-independent state for context forks. */
  deviceState?: WebGPUDeviceState | null;
}

export interface WebGPUAdapterIdentity {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  backend?: string;
  deviceType?: string;
  driver?: string;
}

export interface WebGPUInitOptions {
  adapterOptions?: GPURequestAdapterOptions;
  deviceDescriptor?: GPUDeviceDescriptor;
}

function normalizeAdapterIdentity(info: WebGPUAdapterIdentity | null | undefined): Readonly<WebGPUAdapterIdentity> | null {
  if (!info || typeof info !== 'object') return null;
  if (Object.isFrozen(info) && Object.keys(info).length > 0) return info;
  const result: WebGPUAdapterIdentity = {};
  for (const key of ['vendor', 'architecture', 'device', 'description', 'backend', 'deviceType', 'driver'] as const) {
    if (typeof info[key] === 'string' && info[key]!.trim()) result[key] = info[key]!.trim();
  }
  return Object.keys(result).length ? Object.freeze(result) : null;
}

function adapterIdentity(adapter: GPUAdapter): Readonly<WebGPUAdapterIdentity> | null {
  return normalizeAdapterIdentity((adapter as GPUAdapter & { info?: WebGPUAdapterIdentity }).info);
}

/** Internal WebGPU engine. GraphExecutor owns context resources while the
 * retained WebGPUDeviceState owns model-independent device caches. */
export class WebGPUEngine extends BackendEngine {
  declare device: GPUDevice;
  declare shaderLibrary: GraphExecutorOptions['shaderLibrary'];
  declare deviceState: WebGPUDeviceState;
  declare executor: GraphExecutor | null;
  declare _graph: RuntimeGraph | null;
  declare _disposed: boolean;
  declare readonly adapterInfo: Readonly<WebGPUAdapterIdentity> | null;

  constructor(
    device: GPUDevice,
    {
      shaderLibrary = null,
      adapterInfo = null,
      deviceState = null,
    }: WebGPUEngineOptions = {},
  ) {
    super('webgpu', {
      incrementalExecution: true,
      incrementalRows: true,
      outputLocation: 'device',
    });
    if (!device) throw new Error('WebGPUEngine requires a GPUDevice.');
    this.device = device;
    this.deviceState = (deviceState || new WebGPUDeviceState(device)).retain();
    if (this.deviceState.device !== device) {
      throw new Error('WebGPUEngine deviceState belongs to a different GPUDevice.');
    }
    this.shaderLibrary = shaderLibrary;
    // GPUAdapterInfo fields are prototype getters in several implementations
    // (including Deno/wgpu), so an object spread would silently erase them.
    this.adapterInfo = normalizeAdapterIdentity(adapterInfo);
    this.executor = null;
    this._graph = null;
    this._disposed = false;
  }

  static async init({
    adapterOptions,
    deviceDescriptor,
  }: WebGPUInitOptions = {}): Promise<WebGPUEngine | null> {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter(adapterOptions);
    if (!adapter) return null;
    const device = await adapter.requestDevice(deviceDescriptor);
    return new WebGPUEngine(device, { adapterInfo: adapterIdentity(adapter) });
  }

  /** Create an unallocated peer that shares the device but owns independent graph state. */
  fork(): WebGPUEngine {
    if (this._disposed) throw new Error('WebGPUEngine is disposed.');
    return new WebGPUEngine(this.device, {
      shaderLibrary: this.shaderLibrary,
      adapterInfo: this.adapterInfo,
      deviceState: this.deviceState,
    });
  }

  _createExecutor(graph: RuntimeGraph): GraphExecutor {
    return new GraphExecutor(this.device, graph, {
      shaderLibrary: this.shaderLibrary,
      deviceState: this.deviceState,
    });
  }

  /**
   * Compile every shader route named by a symbolic bounded-domain proof before
   * a CompiledModel is published. Optional specialized routes are allowed to fail
   * only when their portable route has already compiled; the rejected source
   * is then remembered so contexts select the portable route directly.
   */
  async precompileDynamicPipelines(
    requiredMethods: readonly string[],
    optionalMethods: readonly string[] = [],
  ): Promise<void> {
    if (this._disposed) throw new Error('WebGPUEngine is disposed.');
    const library = this.shaderLibrary || (await import('./ShaderLibrary.js')).ShaderLibrary;
    const methods = library as unknown as Record<string, unknown>;
    const source = (method: string): string => {
      const candidate = methods[method];
      if (typeof candidate !== 'function') {
        throw new Error(`WebGPU shader library does not expose '${method}'.`);
      }
      const code = candidate.call(library);
      if (typeof code !== 'string' || code.length === 0) {
        throw new Error(`WebGPU shader library '${method}' returned no WGSL source.`);
      }
      return code;
    };
    for (const method of new Set(requiredMethods)) {
      await this.deviceState.computePipeline(source(method));
    }
    for (const method of new Set(optionalMethods)) {
      const code = source(method);
      try {
        await this.deviceState.computePipeline(code);
      } catch {
        this.deviceState.rejectedSpecializedShaders.add(code);
      }
    }
  }

  get packedDot4Available(): boolean {
    return globalThis.navigator?.gpu?.wgslLanguageFeatures
      ?.has?.('packed_4x8_integer_dot_product') === true;
  }

  async allocateGraph(graph: RuntimeGraph | null): Promise<this> {
    if (this._disposed) throw new Error('WebGPUEngine is disposed.');
    if (!graph || !(graph.tensors instanceof Map) || !Array.isArray(graph.nodes)) {
      throw new Error('WebGPUEngine.allocateGraph requires a RuntimeGraph.');
    }
    this._assertPortableQuantizedGraph(graph);
    const previousExecutor = this.executor;
    this.resetDecodeCache();
    previousExecutor?._setDecodeCacheGenerationListener?.(null);
    previousExecutor?.dispose?.();
    this.executor = null;
    this._graph = null;
    const executor = this._createExecutor(graph);
    executor._setDecodeCacheGenerationListener?.(
      () => this._advanceDecodeCacheGeneration(),
    );
    try {
      await executor.compile();
    } catch (error) {
      executor._setDecodeCacheGenerationListener?.(null);
      executor.dispose?.();
      throw error;
    }
    this.executor = executor;
    this._graph = graph;
    return this;
  }

  /** Publish one concrete bounded-shape generation without replacing the executor. */
  async rebindGraph(graph: RuntimeGraph, options: WebGPURebindOptions): Promise<this> {
    if (this._disposed) throw new Error('WebGPUEngine is disposed.');
    if (!graph || !(graph.tensors instanceof Map) || !Array.isArray(graph.nodes)) {
      throw new Error('WebGPUEngine.rebindGraph requires a RuntimeGraph.');
    }
    this._assertPortableQuantizedGraph(graph);
    if (this.decodeCacheGeneration >= Number.MAX_SAFE_INTEGER ||
        (this.executor?.decodeCacheGeneration ?? 0) >= Number.MAX_SAFE_INTEGER) {
      throw new Error('WebGPU decode-cache generation is exhausted before shape rebinding.');
    }
    if (!this.executor) {
      const executor = this._createExecutor(graph);
      executor._setDecodeCacheGenerationListener?.(
        () => this._advanceDecodeCacheGeneration(),
      );
      try {
        await executor.rebindGraph(graph, options);
      } catch (error) {
        executor._setDecodeCacheGenerationListener?.(null);
        executor.dispose?.();
        throw error;
      }
      this.executor = executor;
      this._graph = graph;
      return this;
    }
    await this.executor.rebindGraph(graph, options);
    this._graph = graph;
    return this;
  }

  inspectDynamicResources(): Readonly<WebGPUResourceInspection> {
    if (!this.executor) throw new Error('WebGPUEngine has no bound graph.');
    return this.executor.inspectDynamicResources();
  }

  async execute(
    inputs: WebGPUExecutionInputs,
    options: WebGPUExecutionOptions = {},
  ): Promise<GPUBuffer | undefined> {
    if (!this.executor) throw new Error('WebGPUEngine.execute requires an allocated graph.');
    this.preflightExecutionInputs(inputs);
    return this._executePreflighted(inputs, options);
  }

  /** Pure validation usable before a candidate graph generation is committed. */
  preflightExecutionInputs(
    inputs: WebGPUExecutionInputs,
    graph: RuntimeGraph | null = null,
  ): void {
    if (graph !== null) {
      preflightWebGPUExecutionInputs(graph as ExecutorGraph, inputs);
      return;
    }
    if (!this.executor) throw new Error('WebGPUEngine has no graph to preflight.');
    const hook = this.executor._preflightExecutionInputs;
    if (typeof hook === 'function') hook.call(this.executor, inputs);
    else preflightWebGPUExecutionInputs(this.executor.graph as ExecutorGraph, inputs);
  }

  /** @internal The caller preflighted these inputs against the current graph. */
  async _executePreflighted(
    inputs: WebGPUExecutionInputs,
    options: WebGPUExecutionOptions = {},
  ): Promise<GPUBuffer | undefined> {
    if (!this.executor) throw new Error('WebGPUEngine.execute requires an allocated graph.');
    const sessionExecution = this._beginDecodeExecution(options);
    let executorOptions = options;
    if (sessionExecution && typeof this.executor._claimDecodeSessionExecution === 'function') {
      // The wrapper and its composed executor own separate monotonic counters.
      // Claim both without exposing either counter in the public option object.
      executorOptions = { ...options };
      this.executor._claimDecodeSessionExecution(
        executorOptions, this.executor.decodeCacheGeneration,
      );
    }
    const hook = this.executor._executePreflighted;
    return typeof hook === 'function'
      ? hook.call(this.executor, inputs, executorOptions)
      : this.executor.execute(inputs, executorOptions);
  }

  resetDecodeCache(): void {
    super.resetDecodeCache();
    this.executor?.resetDecodeCache?.();
  }

  /** @internal Clear seed-specific row plans and device-feedback replay state. */
  resetDecodeBinding(): void {
    this.resetDecodeCache();
    this.executor?.resetDecodeBindingState();
  }

  readBuffer(
    gpuBuffer: GPUBuffer,
    sizeBytes: number,
    dtype: RuntimeDType = 'float32',
  ): Promise<RuntimeTypedArray> {
    if (!this.executor) throw new Error('WebGPUEngine.readBuffer requires an allocated graph.');
    return this.executor.readBuffer(gpuBuffer, sizeBytes, dtype);
  }

  readBufferRange(
    gpuBuffer: GPUBuffer,
    byteOffset: number,
    sizeBytes: number,
    dtype: RuntimeDType = 'float32',
  ): Promise<RuntimeTypedArray> {
    if (!this.executor) throw new Error('WebGPUEngine.readBufferRange requires an allocated graph.');
    return this.executor.readBufferRange(gpuBuffer, byteOffset, sizeBytes, dtype);
  }

  snapshotOutputs(): ReadonlyMap<string, WebGPUOutputSnapshot> {
    if (!this.executor) throw new Error('WebGPUEngine.snapshotOutputs requires an allocated graph.');
    return this.executor.snapshotOutputs();
  }

  executeDeviceFeedbackDecode(
    inputs: WebGPUExecutionInputs,
    options: DeviceFeedbackDecodeOptions = {},
  ): Promise<GPUBuffer> {
    if (!this.executor) {
      throw new Error('WebGPUEngine.executeDeviceFeedbackDecode requires an allocated graph.');
    }
    this._beginDecodeExecution({});
    return this.executor.executeDeviceFeedbackDecode(inputs, options);
  }

  releaseAdapterTargets(
    targets: Parameters<GraphExecutor['releaseAdapterTargets']>[0],
  ): void {
    return this.executor?.releaseAdapterTargets(targets);
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.resetDecodeCache();
    const executor = this.executor;
    executor?._setDecodeCacheGenerationListener?.(null);
    executor?.dispose?.();
    this.executor = null;
    this._graph = null;
    this.deviceState.release();
  }

  get graph(): RuntimeGraph | null { return (this.executor?.graph as RuntimeGraph | undefined) || this._graph; }
  get gpuBuffers(): Map<string, GPUBuffer> | undefined { return this.executor?.gpuBuffers; }
  get pipelines(): GraphExecutor['pipelines'] | undefined { return this.executor?.pipelines; }
  get adapterTargetBuffers(): GraphExecutor['adapterTargetBuffers'] | undefined {
    return this.executor?.adapterTargetBuffers;
  }
  get auxiliaryBuffers(): Set<GPUBuffer> | undefined { return this.executor?.auxiliaryBuffers; }
  get compiledWeightRevision(): number | undefined { return this.executor?.compiledWeightRevision; }
  set compiledWeightRevision(value: number | undefined) {
    if (this.executor) this.executor.compiledWeightRevision = value;
  }
  get compiledTopologyRevision(): number | undefined { return this.executor?.compiledTopologyRevision; }
}
