import type { Graph } from '../core/Graph.js';
import { assertInferenceExecutionOptions, BackendEngine } from './BackendEngine.js';
import type { BackendExecutionOptions } from './BackendEngine.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import { WebGPUDeviceState } from './WebGPUDeviceState.js';
import {
  WebGPUGraphCompiler,
  compileWebGPUGraphPlan,
  type WebGPUCompiledGraphPlan,
} from './WebGPUGraphCompiler.js';
import { WebGPUResources } from './WebGPUResources.js';
import { WebGPUDispatch } from './WebGPUDispatch.js';
import { WebGPUDecodeState } from './WebGPUDecodeState.js';
import { WebGPUResults, type WebGPUOutputSnapshot } from './WebGPUResults.js';
import type {
  AdapterExecutionPlan,
  AdapterTarget,
  CompiledWebGPUPipeline,
  DeviceFeedbackDecodeOptions,
  DeviceFeedbackDescriptor,
  DeviceFeedbackState,
  ExecutorGraph,
  ExecutorNode,
  ExecutorTensor,
  GraphExecutorOptions,
  IncrementalRowByteCopy,
  IncrementalRowCandidate,
  IncrementalRowPlan,
  WebGPUExecutionInputs,
  WebGPUExecutionOptions,
} from './WebGPUContracts.js';

export type {
  CompiledWebGPUPipeline,
  DeviceFeedbackDecodeOptions,
  GraphExecutorOptions,
  WebGPUAdapterSelector,
  WebGPUExecutionInputs,
  WebGPUExecutionOptions,
} from './WebGPUContracts.js';

/**
 * Graph-bound WebGPU orchestration facade. Operator compilation, allocation,
 * dispatch, decode mutation, and result snapshots are owned by dedicated
 * collaborators; this class only composes their lifecycle and preserves the
 * internal engine hooks used by training and backend integration.
 */
export class GraphExecutor extends BackendEngine {
  declare readonly allocateGraph: never;

  declare device: GPUDevice;
  declare graph: ExecutorGraph;
  declare hasPackedDot4: boolean;
  declare deviceState: WebGPUDeviceState;
  declare _ownsDeviceState: boolean;
  declare pipelines: CompiledWebGPUPipeline[];
  declare computePipelineCache: Map<string, Map<string, Promise<GPUComputePipeline>>>;
  declare rejectedSpecializedShaders: Set<string>;
  declare gpuBuffers: Map<string, GPUBuffer>;
  declare adapterTargetBuffers: Map<AdapterTarget, { a: GPUBuffer; b: GPUBuffer }>;
  declare auxiliaryBuffers: Set<GPUBuffer>;
  declare incrementalRowPlans: Map<number, IncrementalRowPlan>;
  declare incrementalRowCandidates: Map<number, IncrementalRowCandidate>;
  declare incrementalRowCopyTensorNames: Set<string>;
  declare decodeState: WebGPUDecodeState;
  declare adapterPipeline: GPUComputePipeline | null;
  declare compiledWeightRevision: number | undefined;
  declare compiledTopologyRevision: number | undefined;
  declare _dinWeights: Map<string, { din: number; dout: number }> | undefined;
  declare _activePipelineTarget: CompiledWebGPUPipeline[] | undefined;
  declare _activePipelineBuffers: Map<string, GPUBuffer> | undefined;
  declare compiledGraphPlan: WebGPUCompiledGraphPlan | null;

  readonly graphCompiler: WebGPUGraphCompiler;
  readonly resources: WebGPUResources;
  readonly dispatch: WebGPUDispatch;
  readonly results: WebGPUResults;

  constructor(device: GPUDevice, graph: Graph, {
    shaderLibrary = null,
    wgslLanguageFeatures = globalThis.navigator?.gpu?.wgslLanguageFeatures,
    deviceState = null,
  }: GraphExecutorOptions = {}) {
    super('webgpu', {
      incrementalExecution: true,
      incrementalRows: true,
      outputLocation: 'device',
    });
    this.device = device;
    if (deviceState && deviceState.device !== device) {
      throw new Error('GraphExecutor deviceState belongs to a different GPUDevice.');
    }
    this.deviceState = deviceState || new WebGPUDeviceState(device);
    this._ownsDeviceState = deviceState == null;
    this.graph = graph as ExecutorGraph;
    this.hasPackedDot4 = wgslLanguageFeatures?.has?.('packed_4x8_integer_dot_product') === true;
    this._incrementalCacheValid = false;
    this.pipelines = [];
    this.computePipelineCache = this.deviceState.computePipelineCache;
    this.rejectedSpecializedShaders = this.deviceState.rejectedSpecializedShaders;
    this.gpuBuffers = new Map();
    this.adapterTargetBuffers = new Map();
    this.auxiliaryBuffers = new Set();
    this.adapterPipeline = null;
    this.compiledGraphPlan = null;

    this.graphCompiler = new WebGPUGraphCompiler(this);
    this.resources = new WebGPUResources(this);
    this.dispatch = new WebGPUDispatch(this);
    this.decodeState = new WebGPUDecodeState(this);
    this.results = new WebGPUResults(this);
    this.incrementalRowPlans = this.decodeState.rowPlans;
    this.incrementalRowCandidates = this.decodeState.rowCandidates;
    this.incrementalRowCopyTensorNames = this.decodeState.rowCopyTensorNames;
    this.graphCompiler.installShaderLibrary(shaderLibrary);
    this.graph.adapters?._registerResourceOwner(this);
    console.log('[VolvoxAI WebGPU] Starting Graph Compilation...');
  }

  resetDecodeCache(): void {
    super.resetDecodeCache();
    this.decodeState.resetExecution();
  }

  get _webGPUIncrementalCacheValid(): boolean {
    return this._incrementalCacheValid;
  }

  set _webGPUIncrementalCacheValid(value: boolean) {
    this._incrementalCacheValid = value;
  }

  _beginWebGPUDecodeExecution(options: BackendExecutionOptions = {}): boolean {
    return this._beginDecodeExecution(options);
  }

  get deviceFeedbackControlBuffer(): GPUBuffer | null {
    return this.decodeState.controlBuffer;
  }

  set deviceFeedbackControlBuffer(value: GPUBuffer | null) {
    this.decodeState.controlBuffer = value;
  }

  get deviceFeedbackSequenceLength(): number {
    return this.decodeState.sequenceLength;
  }

  set deviceFeedbackSequenceLength(value: number) {
    this.decodeState.sequenceLength = value;
  }

  get deviceFeedbackState(): Readonly<DeviceFeedbackState> | null {
    return this.decodeState.feedback;
  }

  set deviceFeedbackState(value: Readonly<DeviceFeedbackState> | null) {
    this.decodeState.feedback = value;
  }

  _buffer(tensor: ExecutorTensor | null | undefined): GPUBuffer {
    return (tensor
      ? (this._activePipelineBuffers || this.gpuBuffers).get(tensor.name)
      : undefined) as GPUBuffer;
  }

  async compile(): Promise<void> {
    this._assertPortableQuantizedGraph(this.graph);
    this.resetDecodeCache();
    await this.graphCompiler.ensureShaderLibrary();
    this.resources.resetCompilationResources();
    this.pipelines = [];
    this.decodeState.resetCompilation();
    this.compiledGraphPlan = compileWebGPUGraphPlan(this.graph as Graph);
    this._dinWeights = new Map(this.compiledGraphPlan.dinWeights);
    this._analyzeIncrementalRows();
    this._allocateBuffers();
    this._aliasInferenceDropoutBuffers();
    for (let nodeIndex = 0; nodeIndex < this.graph.nodes.length; nodeIndex++) {
      const node = this.graph.nodes[nodeIndex];
      if (node.opType === 'Dropout') continue;
      const start = this.pipelines.length;
      await this._buildNodePipeline(node);
      for (let i = start; i < this.pipelines.length; i++) {
        this.pipelines[i].graphNodeIndex = nodeIndex;
      }
    }
    this.compiledWeightRevision = this.graph.weightRevision || 0;
    this.compiledTopologyRevision = this.graph.topologyRevision || 0;
    this.graph.adapters?._markAcceleratedBackend('webgpu');
    console.log(`[VolvoxAI WebGPU] Compilation complete. Allocated ${this.gpuBuffers.size} VRAM buffers.`);
  }

  dispose(): void {
    this.resetDecodeCache();
    this.graph.adapters?._unregisterResourceOwner(this);
    this.resources.resetCompilationResources();
    this.decodeState.resetCompilation();
    this.pipelines = [];
    if (this._ownsDeviceState) this.deviceState.clear();
    this.adapterPipeline = null;
    this.compiledGraphPlan = null;
  }

  _rowTypedStorage(tensor: ExecutorTensor, sizeBytes = tensor.sizeBytes): RuntimeTypedArray {
    return this.decodeState._rowTypedStorage(tensor, sizeBytes);
  }

  _prepareIncrementalRowNode(node: ExecutorNode, position: number): ExecutorNode {
    return this.decodeState._prepareIncrementalRowNode(node, position);
  }

  _incrementalRowCandidate(node: ExecutorNode, nodeIndex: number): IncrementalRowCandidate | null {
    return this.decodeState._incrementalRowCandidate(node, nodeIndex);
  }

  _assertIncrementalRowInvariants(
    selectedNodes: Iterable<number>,
    changedInputs: readonly string[],
  ): void {
    return this.decodeState._assertIncrementalRowInvariants(selectedNodes, changedInputs);
  }

  _analyzeIncrementalRows(): void {
    return this.decodeState._analyzeIncrementalRows();
  }

  _compileIncrementalRowPipelines(
    nodeIndices: Iterable<number> = this.incrementalRowCandidates.keys(),
  ): Promise<void> {
    return this.graphCompiler._compileIncrementalRowPipelines(nodeIndices);
  }

  _ensureAdapterPipeline(): Promise<GPUComputePipeline> {
    return this.graphCompiler._ensureAdapterPipeline();
  }

  _cachedComputePipeline(
    code: string,
    options: { entryPoint?: string; constants?: Record<string, number | boolean> | null } = {},
  ): Promise<GPUComputePipeline> {
    return this.graphCompiler._cachedComputePipeline(code, options);
  }

  _adapterBuffers(target: AdapterTarget): { a: GPUBuffer; b: GPUBuffer } {
    return this.resources._adapterBuffers(target);
  }

  _createAuxiliaryStorageBuffer(label: string, values: ArrayBufferView): GPUBuffer {
    return this.resources._createAuxiliaryStorageBuffer(label, values);
  }

  releaseAdapterTargets(targets: readonly AdapterTarget[] | null | undefined): void {
    return this.resources.releaseAdapterTargets(targets);
  }

  _prepareAdapterDispatches(
    plan: AdapterExecutionPlan,
  ): Promise<Map<number, CompiledWebGPUPipeline[]>> {
    return this.dispatch._prepareAdapterDispatches(plan);
  }

  _allocateBuffers(): void {
    return this.resources._allocateBuffers();
  }

  _aliasInferenceDropoutBuffers(): void {
    return this.resources._aliasInferenceDropoutBuffers();
  }

  _buildNodePipeline(
    node: ExecutorNode,
    options: {
      pipelines?: CompiledWebGPUPipeline[];
      buffers?: Map<string, GPUBuffer>;
    } = {},
  ): Promise<void> {
    return this.graphCompiler._buildNodePipeline(node, options);
  }

  _buildNodePipelineImpl(node: ExecutorNode): Promise<void> {
    return this.graphCompiler._buildNodePipelineImpl(node);
  }

  _preflightQGroupNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQGroupNormDynamicAffines(inputs);
  }

  _preflightQLayerNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQLayerNormDynamicAffines(inputs);
  }

  _preflightQSDPAMasks(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQSDPAMasks(inputs);
  }

  _preflightQMaskedMeanMasks(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQMaskedMeanMasks(inputs);
  }

  _preflightQEmbeddingIds(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQEmbeddingIds(inputs);
  }

  _incrementalRowRange(
    rowTensor: ExecutorTensor | null,
    fullTensor: ExecutorTensor,
    position: number,
    label: string,
  ): { offset: number; size: number } {
    return this.decodeState._incrementalRowRange(rowTensor, fullTensor, position, label);
  }

  _uploadExecutionInputs(
    inputs: WebGPUExecutionInputs,
    rowPosition: number | null,
    changedInputs?: readonly string[],
  ): void {
    return this.dispatch._uploadExecutionInputs(inputs, rowPosition, changedInputs);
  }

  _encodeIncrementalRowByteCopy(
    commandEncoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline | null,
    copy: IncrementalRowByteCopy | undefined,
    sourceOffset: number,
    destinationOffset: number,
    size: number,
    label: string,
  ): void {
    return this.decodeState._encodeIncrementalRowByteCopy(
      commandEncoder, pipeline, copy, sourceOffset, destinationOffset, size, label,
    );
  }

  _rowTensorByName(
    tensors: Record<string, ExecutorTensor>,
    name: string,
  ): ExecutorTensor | null {
    return this.decodeState._rowTensorByName(tensors, name);
  }

  _encodeIncrementalRowsInto(
    commandEncoder: GPUCommandEncoder,
    selectedNodes: Iterable<number>,
    rowPosition: number,
    options: { qsdpaControlBuffer?: GPUBuffer | null } = {},
  ): void {
    return this.decodeState._encodeIncrementalRowsInto(
      commandEncoder, selectedNodes, rowPosition, options,
    );
  }

  _encodeIncrementalRows(selectedNodes: Iterable<number>, rowPosition: number): void {
    return this.decodeState._encodeIncrementalRows(selectedNodes, rowPosition);
  }

  _deviceFeedbackDescriptor(
    inputs: WebGPUExecutionInputs,
    options: DeviceFeedbackDecodeOptions = {},
  ): DeviceFeedbackDescriptor {
    return this.decodeState._deviceFeedbackDescriptor(inputs, options);
  }

  _deviceFeedbackControl(sequenceLength: number): GPUBuffer {
    return this.decodeState._deviceFeedbackControl(sequenceLength);
  }

  executeDeviceFeedbackDecode(
    inputs: WebGPUExecutionInputs,
    options: DeviceFeedbackDecodeOptions = {},
  ): Promise<GPUBuffer> {
    return this.decodeState.executeDeviceFeedbackDecode(inputs, options);
  }

  execute(
    inputs: WebGPUExecutionInputs,
    options: WebGPUExecutionOptions = {},
  ): Promise<GPUBuffer | undefined> {
    assertInferenceExecutionOptions(options, 'WebGPU inference');
    return this.dispatch.execute(inputs, options);
  }

  snapshotOutputs(): ReadonlyMap<string, WebGPUOutputSnapshot> {
    return this.results.snapshotOutputs();
  }

  readBuffer(
    gpuBuffer: GPUBuffer,
    sizeBytes: number,
    dtype: RuntimeDType = 'float32',
  ): Promise<RuntimeTypedArray> {
    return this.results.readBuffer(gpuBuffer, sizeBytes, dtype);
  }

  readBufferRange(
    gpuBuffer: GPUBuffer,
    byteOffset: number,
    sizeBytes: number,
    dtype: RuntimeDType = 'float32',
  ): Promise<RuntimeTypedArray> {
    return this.results.readBufferRange(gpuBuffer, byteOffset, sizeBytes, dtype);
  }
}
