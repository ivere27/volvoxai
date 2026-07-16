import { BackendEngine } from './BackendEngine.js';
import { GraphExecutor } from './GraphExecutor.js';
import type {
  DeviceFeedbackDecodeOptions,
  GraphExecutorOptions,
  WebGPUExecutionInputs,
  WebGPUExecutionOptions,
} from './GraphExecutor.js';
import type { Graph } from '../core/Graph.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';

export interface WebGPUEngineOptions {
  shaderLibrary?: GraphExecutorOptions['shaderLibrary'];
  training?: boolean;
}

export interface WebGPUInitOptions {
  adapterOptions?: GPURequestAdapterOptions;
  deviceDescriptor?: GPUDeviceDescriptor;
}

/**
 * WebGPU backend peer for CPUEngine, WasmEngine, and WebNNEngine.
 *
 * GraphExecutor remains the lower-level scheduler/resource owner. This class
 * supplies the shared backend lifecycle (`allocateGraph`, `execute`, decode
 * sessions) while forwarding the established WebGPU readback and training
 * integration points for compatibility.
 */
export class WebGPUEngine extends BackendEngine {
  declare device: GPUDevice;
  declare shaderLibrary: GraphExecutorOptions['shaderLibrary'];
  declare training: boolean;
  declare executor: GraphExecutor | null;
  declare _graph: Graph | null;

  constructor(
    device: GPUDevice,
    { shaderLibrary = null, training = false }: WebGPUEngineOptions = {},
  ) {
    super('webgpu', {
      incrementalExecution: true,
      incrementalRows: true,
      outputLocation: 'device',
    });
    if (!device) throw new Error('WebGPUEngine requires a GPUDevice.');
    this.device = device;
    this.shaderLibrary = shaderLibrary;
    this.training = training === true;
    this.executor = null;
    this._graph = null;
  }

  static async init({
    adapterOptions,
    deviceDescriptor,
  }: WebGPUInitOptions = {}): Promise<WebGPUEngine | null> {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter(adapterOptions);
    if (!adapter) return null;
    const device = await adapter.requestDevice(deviceDescriptor);
    return new WebGPUEngine(device);
  }

  /** Create an unallocated peer that shares the device but owns independent graph state. */
  fork(): WebGPUEngine {
    return new WebGPUEngine(this.device, {
      shaderLibrary: this.shaderLibrary,
      training: this.training,
    });
  }

  _createExecutor(graph: Graph): GraphExecutor {
    return new GraphExecutor(this.device, graph, {
      shaderLibrary: this.shaderLibrary,
      training: this.training,
    });
  }

  async allocateGraph(graph: Graph | null): Promise<this> {
    if (!graph || !(graph.tensors instanceof Map) || !Array.isArray(graph.nodes)) {
      throw new Error('WebGPUEngine.allocateGraph requires a Graph.');
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

  /** Compatibility alias for callers that previously compiled WebGPU directly. */
  compile(graph: Graph | null = this._graph): Promise<this> {
    return this.allocateGraph(graph);
  }

  async execute(
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
    return this.executor.execute(inputs, executorOptions);
  }

  resetDecodeCache(): void {
    super.resetDecodeCache();
    this.executor?.resetDecodeCache?.();
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

  async prepareForTraining(): Promise<this> {
    if (!this.executor) throw new Error('WebGPUEngine.prepareForTraining requires an allocated graph.');
    this.resetDecodeCache();
    await this.executor.prepareForTraining();
    this.training = true;
    return this;
  }

  releaseAdapterTargets(
    targets: Parameters<GraphExecutor['releaseAdapterTargets']>[0],
  ): void {
    return this.executor?.releaseAdapterTargets(targets);
  }

  dispose(): void {
    this.resetDecodeCache();
    const executor = this.executor;
    executor?._setDecodeCacheGenerationListener?.(null);
    executor?.dispose?.();
    this.executor = null;
    this._graph = null;
  }

  get graph(): Graph | null { return (this.executor?.graph as Graph | undefined) || this._graph; }
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
  get _dinWeights(): GraphExecutor['_dinWeights'] { return this.executor?._dinWeights; }
}
