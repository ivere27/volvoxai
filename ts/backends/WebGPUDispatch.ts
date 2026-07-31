import { Tensor } from '../core/Tensor.js';
import type { Graph } from '../core/Graph.js';
import { incrementalExecutionEnabled, incrementalNodeSelection } from './incrementalExecution.js';
import { incrementalRowPosition } from './quantizedRowExecution.js';
import type { GraphExecutor } from './GraphExecutor.js';
import type {
  AdapterExecutionPlan,
  CompiledWebGPUPipeline,
  WebGPUExecutionInputs,
  WebGPUExecutionOptions,
} from './WebGPUContracts.js';

export interface WebGPUBufferCopy {
  readonly source: GPUBuffer;
  readonly sourceOffset?: number;
  readonly destination: GPUBuffer;
  readonly destinationOffset?: number;
  readonly size: number;
}

/** Encode an ordered set of copies into one submission. */
export function submitWebGPUBufferCopies(
  device: GPUDevice,
  copies: readonly WebGPUBufferCopy[],
): void {
  if (copies.length === 0) return;
  const commandEncoder = device.createCommandEncoder();
  for (const copy of copies) {
    commandEncoder.copyBufferToBuffer(
      copy.source,
      copy.sourceOffset || 0,
      copy.destination,
      copy.destinationOffset || 0,
      copy.size,
    );
  }
  device.queue.submit([commandEncoder.finish()]);
}

/** Host upload validation, command encoding, and ordered queue submission. */
export class WebGPUDispatch {
  constructor(readonly host: GraphExecutor) {}

    async _prepareAdapterDispatches(
      plan: AdapterExecutionPlan,
    ): Promise<Map<number, CompiledWebGPUPipeline[]>> {
      const byNode = new Map<number, CompiledWebGPUPipeline[]>();
      if (!plan) return byNode;
      const pipeline = await this.host._ensureAdapterPipeline();
      for (let nodeIndex = 0; nodeIndex < this.host.graph.nodes.length; nodeIndex++) {
        const node = this.host.graph.nodes[nodeIndex];
        if (node.opType !== "MatMul" && node.opType !== "Linear" && node.opType !== "Gemm") continue;
        const input = node.inputs.input || node.inputs.x || node.inputs.a;
        const output = node.outputs.out || Object.values(node.outputs)[0];
        const weight = node.inputs.weight || node.inputs.b;
        if (!input || !output || !weight) continue;
        const rows = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const batchSize = input.shape.length > 1 ? input.shape[0] : 1;
        const routes = plan.kind === "batch" ? plan.routes : [plan.route];
        if (routes.length !== 1 && routes.length !== batchSize) {
          throw new Error(`WebGPU adapter execution requires one route or ${batchSize} routes; received ${routes.length}.`);
        }
        if (rows % batchSize !== 0) throw new Error(`Cannot divide ${rows} linear rows across batch size ${batchSize}.`);
        const rowsPerRoute = routes.length === 1 ? rows : rows / batchSize;
        const dispatches: CompiledWebGPUPipeline[] = [];
        for (let routeIndex = 0; routeIndex < routes.length; routeIndex++) {
          const route = routes[routeIndex];
          if (!route) continue;
          const target = route.snapshot.targetsByWeight.get(weight.name);
          if (!target) continue;
          const rowStart = routes.length === 1 ? 0 : routeIndex * rowsPerRoute;
          const rowCount = rowsPerRoute;
          const params = new ArrayBuffer(32);
          const pu = new Uint32Array(params);
          const pf = new Float32Array(params);
          pu[0] = rows; pu[1] = target.din; pu[2] = target.dout; pu[3] = target.rank;
          pu[4] = rowStart; pu[5] = rowCount;
          pf[6] = route.scale * target.scale;
          const paramsBuffer = this.host.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          this.host.device.queue.writeBuffer(paramsBuffer, 0, params);
          const factors = this.host._adapterBuffers(target);
          const bindGroup = this.host.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.host._buffer(input) } },
              { binding: 1, resource: { buffer: factors.a } },
              { binding: 2, resource: { buffer: factors.b } },
              { binding: 3, resource: { buffer: this.host._buffer(output) } },
              { binding: 4, resource: { buffer: paramsBuffer } },
            ],
          });
          dispatches.push({
            pipeline,
            bindGroup,
            workgroupCount: [Math.ceil(target.dout / 64), rowCount, 1],
            resources: [paramsBuffer],
          });
        }
        if (dispatches.length) byNode.set(nodeIndex, dispatches);
      }
      return byNode;
    }

    _preflightQGroupNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
      for (const node of this.host.graph.nodes) {
        if (node.opType !== 'QGroupNorm') continue;
        for (const [label, tensor] of
          [['weight', node.inputs.weight], ['bias', node.inputs.bias]] as const) {
          if (!tensor || tensor.isWeight === true || tensor.isInput !== true) continue;
          const supplied = Object.prototype.hasOwnProperty.call(inputs, tensor.name)
            ? inputs[tensor.name]
            : null;
          if (!supplied) {
            throw new Error(`WebGPU QGroupNorm node ${node.id} requires graph-input F32 ${label} supplied on every execution.`);
          }
          Tensor.assertCompatibleInput('float32', supplied, tensor.sizeBytes,
            `QGroupNorm node ${node.id} ${label}`);
          if (!supplied.every(Number.isFinite)) {
            throw new Error(`WebGPU QGroupNorm node ${node.id} ${label} must contain finite F32 values before dispatch.`);
          }
        }
      }
    }

    _preflightQLayerNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
      for (const node of this.host.graph.nodes) {
        if (node.opType !== 'QLayerNorm') continue;
        for (const [label, tensor] of
          [['weight', node.inputs.weight], ['bias', node.inputs.bias]] as const) {
          if (!tensor || tensor.isWeight === true || tensor.isInput !== true) continue;
          const supplied = Object.prototype.hasOwnProperty.call(inputs, tensor.name)
            ? inputs[tensor.name]
            : null;
          if (!supplied) {
            throw new Error(`WebGPU QLayerNorm node ${node.id} requires graph-input F32 ${label} supplied on every execution.`);
          }
          Tensor.assertCompatibleInput('float32', supplied, tensor.sizeBytes,
            `QLayerNorm node ${node.id} ${label}`);
          if (!supplied.every(Number.isFinite)) {
            throw new Error(`WebGPU QLayerNorm node ${node.id} ${label} must contain finite F32 values before dispatch.`);
          }
        }
      }
    }

    _preflightQSDPAMasks(inputs: WebGPUExecutionInputs): void {
      for (const node of this.host.graph.nodes) {
        if (node.opType !== 'QSDPA') continue;
        const mask = node.inputs.mask;
        if (!mask?.isInput) continue;
        const supplied = Object.prototype.hasOwnProperty.call(inputs, mask.name)
          ? inputs[mask.name]
          : null;
        if (!supplied) {
          throw new Error(`WebGPU QSDPA node ${node.id} requires graph-input I32 mask supplied on every execution.`);
        }
        Tensor.assertCompatibleInput('int32', supplied, mask.sizeBytes,
          `QSDPA node ${node.id} mask`);
      }
    }

    _preflightQMaskedMeanMasks(inputs: WebGPUExecutionInputs): void {
      for (const node of this.host.graph.nodes) {
        if (node.opType !== 'QMaskedMean') continue;
        const mask = node.inputs.mask;
        if (!mask?.isInput) continue;
        const supplied = Object.prototype.hasOwnProperty.call(inputs, mask.name)
          ? inputs[mask.name]
          : null;
        if (!supplied) {
          throw new Error(`WebGPU QMaskedMean node ${node.id} requires graph-input I32 mask supplied on every execution.`);
        }
        Tensor.assertCompatibleInput('int32', supplied, mask.sizeBytes,
          `QMaskedMean node ${node.id} mask`);
      }
    }

    _preflightQEmbeddingIds(inputs: WebGPUExecutionInputs): void {
      for (const node of this.host.graph.nodes) {
        if (node.opType !== 'QEmbedding') continue;
        const ids = node.inputs.input;
        const weight = node.inputs.weight;
        if (!ids || ids.dtype !== 'int32' || !weight || weight.shape?.length !== 2) {
          throw new Error(`WebGPU QEmbedding node ${node.id} requires preflight-complete I32 IDs.`);
        }
        if (!ids.isInput) continue;
        const supplied = Object.prototype.hasOwnProperty.call(inputs, ids.name)
          ? inputs[ids.name]
          : null;
        if (!supplied) {
          throw new Error(`WebGPU QEmbedding node ${node.id} requires graph-input I32 IDs supplied on every execution for preflight.`);
        }
        Tensor.assertCompatibleInput(ids.dtype, supplied, ids.sizeBytes,
          `QEmbedding node ${node.id} input IDs`);
        const vocabularySize = weight.shape[0];
        for (let token = 0; token < supplied.length; token++) {
          const id = supplied[token];
          if (id < 0 || id >= vocabularySize) {
            throw new Error(`WebGPU QEmbedding node ${node.id} token id ${id} is outside vocabulary size ${vocabularySize}.`);
          }
        }
      }
    }

    _uploadExecutionInputs(
      inputs: WebGPUExecutionInputs,
      rowPosition: number | null,
      changedInputs?: readonly string[],
    ): void {
      const changed = rowPosition == null ? null : new Set(changedInputs || Object.keys(inputs));
      for (const [name, data] of Object.entries(inputs)) {
        const tensor = this.host.graph.getTensor(name);
        const buffer = this.host.gpuBuffers.get(name);
        if (!tensor?.isInput || !buffer) throw new Error(`Unknown graph input '${name}'.`);
        Tensor.assertCompatibleInput(tensor.dtype, data, tensor.sizeBytes, `Input '${name}'`);
        let destinationOffset = 0;
        let sourceOffset = 0;
        let sourceBytes = data.byteLength;
        if (rowPosition != null) {
          if (!changed!.has(name)) continue;
          const sequence = tensor.shape?.[1];
          if (tensor.shape?.[0] !== 1 || typeof sequence !== 'number' ||
              !Number.isInteger(sequence) || sequence <= rowPosition ||
              tensor.sizeBytes % sequence !== 0) {
            throw new Error(`WebGPU incremental row input '${name}' requires B=1 fixed-sequence storage.`);
          }
          sourceBytes = tensor.sizeBytes / sequence;
          destinationOffset = rowPosition * sourceBytes;
          sourceOffset = destinationOffset;
        }
        if (sourceBytes % 4 === 0 && destinationOffset % 4 === 0) {
          this.host.device.queue.writeBuffer(
            buffer, destinationOffset, data.buffer, data.byteOffset + sourceOffset, sourceBytes,
          );
        } else {
          // writeBuffer requires a four-byte destination offset and write size.
          // The caller supplies the complete logical input, so include the few
          // neighboring host bytes needed to form a minimally aligned span. This
          // preserves adjacent packed rows without uploading the full sequence.
          const alignedStart = Math.floor(destinationOffset / 4) * 4;
          const logicalEnd = destinationOffset + sourceBytes;
          const alignedEnd = Math.ceil(logicalEnd / 4) * 4;
          const aligned = new Uint8Array(alignedEnd - alignedStart);
          const availableEnd = Math.min(alignedEnd, data.byteLength);
          if (availableEnd > alignedStart) {
            aligned.set(new Uint8Array(
              data.buffer, data.byteOffset + alignedStart, availableEnd - alignedStart,
            ));
          }
          this.host.device.queue.writeBuffer(buffer, alignedStart, aligned);
        }
      }
    }

    async execute(
      inputs: WebGPUExecutionInputs,
      options: WebGPUExecutionOptions = {},
      pipelineSelection: ReadonlyMap<number, CompiledWebGPUPipeline> | null = null,
    ): Promise<GPUBuffer | undefined> {
      this.host.graph.assertTopologyRevision?.(this.host.compiledTopologyRevision!, "WebGPU");
      this.host.deviceFeedbackState = null;
      this.host._beginWebGPUDecodeExecution(options);
      const hasAdapterSelector = Object.prototype.hasOwnProperty.call(options, "adapter") ||
        Object.prototype.hasOwnProperty.call(options, "adapters");
      const adapterPlan = (hasAdapterSelector || this.host.graph.adapters?.hasActive())
        ? this.host.graph.adapters._pinExecution(options)
        : null;
      const incremental = incrementalExecutionEnabled(options, adapterPlan);
      const cacheWasValid = this.host._webGPUIncrementalCacheValid;
      const selectedNodes = incremental
        ? incrementalNodeSelection(this.host.graph as Graph, inputs, options, cacheWasValid)
        : null;
      this.host._webGPUIncrementalCacheValid = false;
      const rowPosition = incrementalRowPosition(options, selectedNodes, cacheWasValid);
      if (rowPosition != null) {
        const rowNodes = selectedNodes!;
        this.host._assertIncrementalRowInvariants(
          rowNodes, options.changedInputs ?? Object.keys(inputs),
        );
        await this.host._compileIncrementalRowPipelines(rowNodes);
        for (const nodeIndex of rowNodes) {
          if (!this.host.incrementalRowPlans.has(nodeIndex)) {
            const node = this.host.graph.nodes[nodeIndex];
            throw new Error(`WebGPU W8A8 incremental row node ${String(node?.id ?? nodeIndex)} has no compiled row pipeline.`);
          }
          const sequence = (this.host.graph.nodes[nodeIndex].outputs?.out ||
            Object.values(this.host.graph.nodes[nodeIndex].outputs || {})[0])?.shape?.[1];
          if (typeof sequence !== 'number' || !Number.isInteger(sequence) || rowPosition >= sequence) {
            throw new Error(`WebGPU W8A8 incremental row node ${this.host.graph.nodes[nodeIndex].id} position ${rowPosition} is outside its fixed sequence.`);
          }
        }
      }
      if (pipelineSelection != null && !(pipelineSelection instanceof Map)) {
        throw new Error('WebGPU pipeline selection must be a Map.');
      }
      if ((this.host.graph.weightRevision || 0) !== this.host.compiledWeightRevision) {
        throw new Error("WebGPU weights changed after compilation; recompile the graph before execution.");
      }
      // A packed compute shader cannot abort an entire dispatch after one lane
      // discovers a bad ID. Keep QEmbedding IDs as host graph inputs and reject
      // them before any GPU upload or dispatch, preserving the no-partial-write
      // guarantee of the CPU and WASM canonical kernels.
      this.host._preflightQGroupNormDynamicAffines(inputs);
      this.host._preflightQLayerNormDynamicAffines(inputs);
      this.host._preflightQSDPAMasks(inputs);
      this.host._preflightQMaskedMeanMasks(inputs);
      this.host._preflightQEmbeddingIds(inputs);
      const adapterDispatches = adapterPlan ? await this.host._prepareAdapterDispatches(adapterPlan) : null;
      const adapterResources: GPUBuffer[] = [];
      this.host._uploadExecutionInputs(inputs, rowPosition, options.changedInputs);
      if (rowPosition != null) {
        this.host._encodeIncrementalRows(selectedNodes!, rowPosition);
        this.host._webGPUIncrementalCacheValid = true;
        const lastNode = this.host.graph.nodes[this.host.graph.nodes.length - 1];
        const outName = Object.keys(lastNode.outputs)[0];
        return this.host.gpuBuffers.get(lastNode.outputs[outName].name);
      }
      let commandEncoder = this.host.device.createCommandEncoder();
      let passEncoder = commandEncoder.beginComputePass();
      let dispatched = 0;
      for (let i = 0; i < this.host.pipelines.length; i++) {
        const compiledPipeline = this.host.pipelines[i];
        if (selectedNodes && !selectedNodes.has(compiledPipeline.graphNodeIndex!)) continue;
        const p = pipelineSelection?.get(i) || compiledPipeline;
        passEncoder.setPipeline(p.pipeline);
        passEncoder.setBindGroup(0, p.bindGroup);
        passEncoder.dispatchWorkgroups(p.workgroupCount[0], p.workgroupCount[1], p.workgroupCount[2]);
        dispatched++;
        const nextNodeIndex = i + 1 < this.host.pipelines.length ? this.host.pipelines[i + 1].graphNodeIndex : -1;
        if (compiledPipeline.graphNodeIndex !== nextNodeIndex) {
          for (const adapter of adapterDispatches?.get(compiledPipeline.graphNodeIndex!) || []) {
            adapterResources.push(...(adapter.resources || []));
            passEncoder.setPipeline(adapter.pipeline);
            passEncoder.setBindGroup(0, adapter.bindGroup);
            passEncoder.dispatchWorkgroups(
              adapter.workgroupCount[0], adapter.workgroupCount[1], adapter.workgroupCount[2]);
          }
        }
        if (dispatched % 20 === 0) {
          passEncoder.end();
          this.host.device.queue.submit([commandEncoder.finish()]);
          commandEncoder = this.host.device.createCommandEncoder();
          passEncoder = commandEncoder.beginComputePass();
        }
      }
      passEncoder.end();
      this.host.device.queue.submit([commandEncoder.finish()]);
      if (incremental) this.host._webGPUIncrementalCacheValid = true;
      if (adapterResources.length && this.host.device.queue.onSubmittedWorkDone) {
        this.host.device.queue.onSubmittedWorkDone().then(
          () => { for (const resource of adapterResources) resource.destroy?.(); },
          () => { for (const resource of adapterResources) resource.destroy?.(); },
        );
      }
      const lastNode = this.host.graph.nodes[this.host.graph.nodes.length - 1];
      const outName = Object.keys(lastNode.outputs)[0];
      return this.host.gpuBuffers.get(lastNode.outputs[outName].name);
    }
    /**
     * Read a logical GPU tensor allocation back to CPU. WebGPU copies are
     * four-byte aligned, so byte tensors use their padded GPU allocation while
     * the returned typed array is trimmed to the requested logical size.
     */
}
