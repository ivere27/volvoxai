import { Tensor } from '../core/Tensor.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import { incrementalExecutionEnabled, incrementalNodeSelection } from './incrementalExecution.js';
import { incrementalRowPosition } from './quantizedRowExecution.js';
import type { GraphExecutor } from './GraphExecutor.js';
import {
  isDeviceTensorInputLease,
  resolveDeviceTensorInputResource,
} from '../ops/deviceTensorReference.js';
import type {
  AdapterExecutionPlan,
  CompiledWebGPUPipeline,
  ExecutorGraph,
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

/** Pure request-value validation over one candidate concrete graph. */
class WebGPUExecutionPreflight {
  constructor(readonly graph: ExecutorGraph) {}

    _preflightQGroupNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
      for (const node of this.graph.nodes) {
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
          if (isDeviceTensorInputLease(supplied)) {
            throw new Error(`WebGPU QGroupNorm node ${node.id} requires host-visible ${label} value preflight.`);
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
      for (const node of this.graph.nodes) {
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
          if (isDeviceTensorInputLease(supplied)) {
            throw new Error(`WebGPU QLayerNorm node ${node.id} requires host-visible ${label} value preflight.`);
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
      for (const node of this.graph.nodes) {
        if (node.opType !== 'QSDPA') continue;
        const mask = node.inputs.mask;
        if (!mask?.isInput) continue;
        const supplied = Object.prototype.hasOwnProperty.call(inputs, mask.name)
          ? inputs[mask.name]
          : null;
        if (!supplied) {
          throw new Error(`WebGPU QSDPA node ${node.id} requires graph-input I32 mask supplied on every execution.`);
        }
        if (isDeviceTensorInputLease(supplied)) {
          if (supplied.dtype !== 'int32' || supplied.logicalSizeBytes !== mask.sizeBytes) {
            throw new Error(`WebGPU QSDPA node ${node.id} mask has incompatible device storage.`);
          }
        } else {
          Tensor.assertCompatibleInput('int32', supplied, mask.sizeBytes,
            `QSDPA node ${node.id} mask`);
        }
      }
    }

    _preflightQMaskedMeanMasks(inputs: WebGPUExecutionInputs): void {
      for (const node of this.graph.nodes) {
        if (node.opType !== 'QMaskedMean') continue;
        const mask = node.inputs.mask;
        if (!mask?.isInput) continue;
        const supplied = Object.prototype.hasOwnProperty.call(inputs, mask.name)
          ? inputs[mask.name]
          : null;
        if (!supplied) {
          throw new Error(`WebGPU QMaskedMean node ${node.id} requires graph-input I32 mask supplied on every execution.`);
        }
        if (isDeviceTensorInputLease(supplied)) {
          if (supplied.dtype !== 'int32' || supplied.logicalSizeBytes !== mask.sizeBytes) {
            throw new Error(`WebGPU QMaskedMean node ${node.id} mask has incompatible device storage.`);
          }
        } else {
          Tensor.assertCompatibleInput('int32', supplied, mask.sizeBytes,
            `QMaskedMean node ${node.id} mask`);
        }
      }
    }

    _preflightQEmbeddingIds(inputs: WebGPUExecutionInputs): void {
      for (const node of this.graph.nodes) {
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
        if (isDeviceTensorInputLease(supplied)) {
          throw new Error(`WebGPU QEmbedding node ${node.id} requires host-visible token ID preflight.`);
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

    _preflightCanonicalValueDomains(inputs: WebGPUExecutionInputs): void {
      const suppliedGraphInput = (tensor, dtype, label) => {
        const supplied = Object.prototype.hasOwnProperty.call(inputs, tensor.name)
          ? inputs[tensor.name]
          : null;
        if (!supplied) {
          throw new Error(`WebGPU ${label} requires graph input '${tensor.name}' on every execution for value preflight.`);
        }
        if (isDeviceTensorInputLease(supplied)) {
          throw new Error(`WebGPU ${label} requires host-visible value preflight.`);
        }
        Tensor.assertCompatibleInput(dtype, supplied, tensor.sizeBytes, label);
        return supplied;
      };
      const invariantValue = (tensor, dtype, label) => {
        if (!tensor.isWeight || tensor.buffer === undefined) {
          throw new Error(`WebGPU ${label} has no context-local invariant payload for value preflight.`);
        }
        return Tensor.assertCompatibleInput(dtype, tensor.buffer, tensor.sizeBytes, label);
      };
      const normalizedIndex = (raw, domain) => raw < 0 ? raw + domain : raw;

      for (const node of this.graph.nodes) {
        if (node.opType === 'Embedding') {
          const ids = node.inputs.input;
          const weight = node.inputs.weight;
          if (!ids?.isInput) continue;
          if (ids.dtype !== 'int32' || weight?.shape?.length !== 2) {
            throw new Error(`WebGPU Embedding node ${node.id} has no complete ID preflight descriptor.`);
          }
          const supplied = suppliedGraphInput(
            ids, 'int32', `Embedding node ${node.id} input IDs`,
          );
          const vocabulary = weight.shape[0];
          for (let offset = 0; offset < supplied.length; offset++) {
            const id = supplied[offset];
            if (id < 0 || id >= vocabulary) {
              throw new Error(
                `WebGPU Embedding node ${node.id} token id ${id} is outside vocabulary size ${vocabulary}.`,
              );
            }
          }
          continue;
        }

        if (node.opType === 'Gather' || node.opType === 'GatherElements') {
          const data = node.inputs.input || node.inputs.data;
          const indices = node.inputs.indices;
          if (!data || !indices || indices.dtype !== 'int32') {
            throw new Error(`WebGPU ${node.opType} node ${node.id} has no complete index preflight descriptor.`);
          }
          let axis = node.params.axis ?? 0;
          if (typeof axis !== 'number' || !Number.isInteger(axis)) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} has an invalid preflight axis.`);
          }
          if (axis < 0) axis += data.shape.length;
          if (axis < 0 || axis >= data.shape.length) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} has an invalid preflight axis.`);
          }
          const residentSlots = node.opType === 'Gather' && axis === 0 &&
            Array.isArray((node as { residentSlots?: unknown }).residentSlots)
            ? (node as { residentSlots: readonly number[] }).residentSlots
            : null;
          if (residentSlots !== null && !indices.isInput && !indices.isWeight) {
            throw new Error(
              `WebGPU Gather node ${node.id} cannot preflight device-produced indices ` +
              `against this context's partial bank residency.`,
            );
          }
          const supplied = indices.isInput
            ? suppliedGraphInput(
              indices, 'int32', `${node.opType} node ${node.id} indices`,
            )
            : residentSlots !== null
              ? invariantValue(
                indices, 'int32', `${node.opType} node ${node.id} invariant indices`,
              )
              : null;
          if (supplied === null) continue;
          const domain = residentSlots === null
            ? data.shape[axis]
            : (node as { residentSlotDomain?: number }).residentSlotDomain;
          if (!Number.isSafeInteger(domain) || (domain as number) <= 0) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} has an invalid index domain.`);
          }
          const resident = residentSlots === null ? null : new Set(residentSlots);
          for (let offset = 0; offset < supplied.length; offset++) {
            const raw = supplied[offset];
            const normalized = normalizedIndex(raw, domain as number);
            if (normalized < 0 || normalized >= (domain as number)) {
              throw new Error(
                `WebGPU ${node.opType} node ${node.id} index ${raw} is outside axis extent ${domain}.`,
              );
            }
            if (resident !== null && !resident.has(normalized)) {
              throw new Error(
                `WebGPU Gather node ${node.id} slot ${normalized} is not resident in this context.`,
              );
            }
          }
          continue;
        }

        if (node.opType === 'MoELinear') {
          const expertWeight = node.inputs.expert_weight || node.inputs.weight;
          const routeIndices = node.inputs.route_indices || node.inputs.indices;
          const routeWeights = node.inputs.route_weights || node.inputs.weights;
          if (!expertWeight || !routeIndices || !routeWeights) {
            throw new Error(`WebGPU MoELinear node ${node.id} has no complete route preflight descriptor.`);
          }
          const residentSlots = Array.isArray((node as { residentSlots?: unknown }).residentSlots)
            ? (node as { residentSlots: readonly number[] }).residentSlots
            : null;
          const resident = residentSlots === null ? null : new Set(residentSlots);
          const suppliedRouteIndices = routeIndices.isInput
            ? suppliedGraphInput(
              routeIndices, 'float32', `MoELinear node ${node.id} route indices`,
            )
            : resident !== null
              ? invariantValue(
                routeIndices, 'float32', `MoELinear node ${node.id} invariant route indices`,
              )
              : null;
          if (suppliedRouteIndices !== null) {
            const supplied = suppliedRouteIndices;
            for (let offset = 0; offset < supplied.length; offset++) {
              const expert = supplied[offset];
              const valid = Number.isInteger(expert) && expert >= 0 &&
                (resident === null ? expert < expertWeight.shape[0] : resident.has(expert));
              if (!valid) {
                throw new Error(
                  `WebGPU MoELinear node ${node.id} has invalid route index ${expert} at offset ${offset}.`,
                );
              }
            }
          }
          if (routeWeights.isInput) {
            const supplied = suppliedGraphInput(
              routeWeights, 'float32', `MoELinear node ${node.id} route weights`,
            );
            for (let offset = 0; offset < supplied.length; offset++) {
              if (!Number.isFinite(supplied[offset])) {
                throw new Error(
                  `WebGPU MoELinear node ${node.id} has non-finite route weight at offset ${offset}.`,
                );
              }
            }
          }
        }
      }
    }

    run(inputs: WebGPUExecutionInputs): void {
      this._preflightQGroupNormDynamicAffines(inputs);
      this._preflightQLayerNormDynamicAffines(inputs);
      this._preflightQSDPAMasks(inputs);
      this._preflightQMaskedMeanMasks(inputs);
      this._preflightQEmbeddingIds(inputs);
      this._preflightCanonicalValueDomains(inputs);
    }
}

export function preflightWebGPUExecutionInputs(
  graph: ExecutorGraph,
  inputs: WebGPUExecutionInputs,
): void {
  new WebGPUExecutionPreflight(graph).run(inputs);
}

/** Host upload validation, command encoding, and ordered queue submission. */
export class WebGPUDispatch {
  constructor(readonly host: GraphExecutor) {}

    _preflightQGroupNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
      return new WebGPUExecutionPreflight(this.host.graph)
        ._preflightQGroupNormDynamicAffines(inputs);
    }

    _preflightQLayerNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
      return new WebGPUExecutionPreflight(this.host.graph)
        ._preflightQLayerNormDynamicAffines(inputs);
    }

    _preflightQSDPAMasks(inputs: WebGPUExecutionInputs): void {
      return new WebGPUExecutionPreflight(this.host.graph)._preflightQSDPAMasks(inputs);
    }

    _preflightQMaskedMeanMasks(inputs: WebGPUExecutionInputs): void {
      return new WebGPUExecutionPreflight(this.host.graph)._preflightQMaskedMeanMasks(inputs);
    }

    _preflightQEmbeddingIds(inputs: WebGPUExecutionInputs): void {
      return new WebGPUExecutionPreflight(this.host.graph)._preflightQEmbeddingIds(inputs);
    }

    _preflightCanonicalValueDomains(inputs: WebGPUExecutionInputs): void {
      return new WebGPUExecutionPreflight(this.host.graph)
        ._preflightCanonicalValueDomains(inputs);
    }

    _preflightExecutionInputs(inputs: WebGPUExecutionInputs): void {
      return preflightWebGPUExecutionInputs(this.host.graph, inputs);
    }

    _uploadExecutionInputs(
      inputs: WebGPUExecutionInputs,
      rowPosition: number | null,
      changedInputs?: readonly string[],
      reuseRetainedInputs = false,
    ): void {
      const changed = reuseRetainedInputs
        ? new Set(changedInputs || Object.keys(inputs))
        : null;
      const deviceCopies: WebGPUBufferCopy[] = [];
      for (const [name, data] of Object.entries(inputs)) {
        const tensor = this.host.graph.getTensor(name);
        const buffer = this.host.gpuBuffers.get(name);
        if (!tensor?.isInput || !buffer) throw new Error(`Unknown graph input '${name}'.`);
        if (isDeviceTensorInputLease(data)) {
          if (rowPosition != null) {
            throw new Error(`WebGPU incremental row input '${name}' does not accept device tensor storage.`);
          }
          if (data.dtype !== tensor.dtype || data.logicalSizeBytes !== tensor.sizeBytes) {
            throw new Error(`WebGPU device input '${name}' disagrees with its bound tensor descriptor.`);
          }
          const source = resolveDeviceTensorInputResource(data, 'webgpu', this.host.device) as GPUBuffer;
          if ((source as { destroyed?: boolean }).destroyed === true ||
              source.size !== Math.max(4, Math.ceil(tensor.sizeBytes / 4) * 4)) {
            throw new Error(`WebGPU device input '${name}' has invalid physical storage.`);
          }
          deviceCopies.push({
            source,
            destination: buffer,
            size: Math.ceil(tensor.sizeBytes / 4) * 4,
          });
          continue;
        }
        Tensor.assertCompatibleInput(tensor.dtype, data, tensor.sizeBytes, `Input '${name}'`);
        if (changed !== null && !changed.has(name)) continue;
        let destinationOffset = 0;
        let sourceOffset = 0;
        let sourceBytes = data.byteLength;
        if (rowPosition != null) {
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
      // GPUQueue.writeBuffer calls above and this one copy submission are
      // ordered on the same queue before the following compute submission.
      // Grouping every device input into one encoder avoids one submit per KV.
      submitWebGPUBufferCopies(this.host.device, deviceCopies);
    }

    async execute(
      inputs: WebGPUExecutionInputs,
      options: WebGPUExecutionOptions = {},
      pipelineSelection: ReadonlyMap<number, CompiledWebGPUPipeline> | null = null,
      preflightComplete = false,
    ): Promise<GPUBuffer | undefined> {
      this.host.graph.assertTopologyRevision?.(this.host.compiledTopologyRevision!, "WebGPU");
      if (pipelineSelection != null && !(pipelineSelection instanceof Map)) {
        throw new Error('WebGPU pipeline selection must be a Map.');
      }
      if ((this.host.graph.weightRevision || 0) !== this.host.compiledWeightRevision) {
        throw new Error("WebGPU weights changed after compilation; recompile the graph before execution.");
      }
      // Value-dependent kernels cannot abort a whole GPU dispatch after one
      // lane discovers an invalid input. This pure pass must precede every
      // decode mutation, lazy row-plan publication, input upload, and submit.
      if (!preflightComplete) this._preflightExecutionInputs(inputs);
      this.host.deviceFeedbackState = null;
      this.host._beginWebGPUDecodeExecution(options);
      const adapterPlan = null;
      const incremental = incrementalExecutionEnabled(options, adapterPlan);
      const cacheWasValid = this.host._webGPUIncrementalCacheValid;
      const selectedNodes = incremental
        ? incrementalNodeSelection(this.host.graph as RuntimeGraph, inputs, options, cacheWasValid)
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
      const adapterResources: GPUBuffer[] = [];
      this.host._uploadExecutionInputs(
        inputs,
        rowPosition,
        options.changedInputs,
        selectedNodes !== null,
      );
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
