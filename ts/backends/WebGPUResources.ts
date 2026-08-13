import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';
import type { Tensor } from '../core/Tensor.js';
import type { GraphExecutor } from './GraphExecutor.js';
import type { AdapterTarget } from './WebGPUContracts.js';
import { compileWebGPUGraphPlan } from './WebGPUGraphCompiler.js';

/** Classify synchronous dynamic-provider device allocation failures uniformly. */
export function createWebGPUBufferOrOOM(
  device: GPUDevice,
  descriptor: GPUBufferDescriptor,
  allocation: string,
): GPUBuffer {
  try {
    return device.createBuffer(descriptor);
  } catch (error) {
    const detail = error instanceof Error && error.message
      ? ` ${error.message}`
      : '';
    throw new VolvoxAIError('OUT_OF_MEMORY',
      `${allocation} allocation of ${String(descriptor.size)} bytes failed.${detail}`, {
        phase: 'execution', backend: 'webgpu', cause: error,
      });
  }
}

export interface StagedWebGPUTensorResources {
  readonly buffers: Map<string, GPUBuffer>;
  readonly usages: Map<string, GPUBufferUsageFlags>;
  readonly capacities: Map<string, number>;
  /** Buffers created by this candidate and safe to destroy on rollback. */
  readonly created: Set<GPUBuffer>;
  readonly logicalActivationBytes: number;
  readonly activationCapacityBytes: number;
  readonly grew: boolean;
}

interface WebGPUResourcePlanOptions {
  graph: RuntimeGraph;
  tensor: Tensor;
  incrementalCopyTensorNames?: ReadonlySet<string>;
  resultCopyTensorNames?: ReadonlySet<string>;
}

export function webGPUBufferUsage({
  graph,
  tensor,
  incrementalCopyTensorNames = new Set(),
  resultCopyTensorNames = new Set(),
}: WebGPUResourcePlanOptions): GPUBufferUsageFlags {
  const name = tensor.name;
  const consumed = graph.nodes.some((node) =>
    Object.values(node.inputs || {}).some((value) => value.name === name));
  const produced = graph.nodes.some((node) =>
    Object.values(node.outputs || {}).some((value) => value.name === name));
  let usage = GPUBufferUsage.STORAGE;
  if (consumed) usage |= GPUBufferUsage.COPY_DST;
  if (produced || resultCopyTensorNames.has(name)) usage |= GPUBufferUsage.COPY_SRC;
  if (!tensor.isWeight && !produced) usage |= GPUBufferUsage.COPY_DST;
  if (tensor.isWeight && tensor.buffer) usage |= GPUBufferUsage.COPY_DST;
  if (incrementalCopyTensorNames.has(name)) {
    usage |= GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  }
  return usage;
}

export function aliasInferenceDropoutBuffers(
  graph: RuntimeGraph,
  buffers: Map<string, GPUBuffer>,
): void {
  for (const node of graph.nodes) {
    if (node.opType !== 'Dropout') continue;
    const input = node.inputs.input || node.inputs.x || node.inputs.data;
    const output = node.outputs.out || Object.values(node.outputs || {})[0];
    const source = input && buffers.get(input.name);
    const allocated = output && buffers.get(output.name);
    if (!input || !output || !source || !allocated || input.dtype !== output.dtype ||
        input.sizeBytes !== output.sizeBytes) {
      throw new Error(`Dropout node ${String(node.id)} requires equal-size input/output tensors.`);
    }
    if (allocated !== source) allocated.destroy?.();
    buffers.set(output.name, source);
  }
}

export function destroyDistinctWebGPUBuffers(buffers: Iterable<GPUBuffer>): void {
  for (const buffer of new Set(buffers)) buffer.destroy?.();
}

function alignedBufferBytes(sizeBytes: number, label: string): number {
  if (!Number.isSafeInteger(sizeBytes) || sizeBytes <= 0 || sizeBytes > Number.MAX_SAFE_INTEGER - 3) {
    throw new Error(`${label} must be a positive safe byte size.`);
  }
  return Math.max(4, Math.ceil(sizeBytes / 4) * 4);
}

function checkedCapacityTarget(
  requiredBytes: number,
  previousBytes: number,
  maximumBytes: number,
  growthFactor: number,
  label: string,
): number {
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes < requiredBytes) {
    throw new Error(`${label} exceeds its compiled bounded-domain capacity.`);
  }
  if (previousBytes >= requiredBytes) return previousBytes;
  if (previousBytes === 0) return requiredBytes;
  const grown = Math.ceil(previousBytes * growthFactor / 4) * 4;
  if (!Number.isSafeInteger(grown)) return maximumBytes;
  return Math.min(maximumBytes, Math.max(requiredBytes, grown));
}

function dropoutAliases(graph: RuntimeGraph): Map<string, string> {
  const aliases = new Map<string, string>();
  for (const node of graph.nodes) {
    if (node.opType !== 'Dropout') continue;
    const input = node.inputs.input || node.inputs.x || node.inputs.data;
    const output = node.outputs.out || Object.values(node.outputs || {})[0];
    if (!input || !output || input.dtype !== output.dtype || input.sizeBytes !== output.sizeBytes) {
      throw new Error(`Dropout node ${String(node.id)} requires equal-size input/output tensors.`);
    }
    aliases.set(output.name, input.name);
  }
  return aliases;
}

function aliasRoot(aliases: ReadonlyMap<string, string>, name: string): string {
  const seen = new Set<string>();
  let current = name;
  while (aliases.has(current)) {
    if (seen.has(current)) throw new Error(`WebGPU Dropout aliases contain a cycle at '${name}'.`);
    seen.add(current);
    current = aliases.get(current)!;
  }
  return current;
}


/** Context-owned allocation, upload, aliasing, and destruction policy. */
export class WebGPUResources {
  constructor(readonly host: GraphExecutor) {}

  resetCompilationResources(): void {
    destroyDistinctWebGPUBuffers(this.host.gpuBuffers.values());
    for (const buffers of this.host.adapterTargetBuffers.values()) {
      buffers.a?.destroy?.();
      buffers.b?.destroy?.();
    }
    for (const buffer of this.host.auxiliaryBuffers) buffer.destroy?.();
    this.host.gpuBuffers.clear();
    this.host.tensorBufferUsages.clear();
    this.host.tensorCapacityBytes.clear();
    this.host.adapterTargetBuffers.clear();
    this.host.auxiliaryBuffers.clear();
  }

    /**
     * Stage one complete concrete tensor generation without mutating the
     * committed graph. Existing buffers are reused only when both capacity and
     * usage are sufficient. Newly allocated buffers remain rollback-owned by
     * the returned candidate until GraphExecutor publishes the generation.
     */
    stageTensorBuffers(
      graph: RuntimeGraph,
      tensorMaximumBytes: ReadonlyMap<string, number> | undefined,
      growthFactor = 2,
      aliasDropout = true,
      plannedResultCopyTensorNames?: readonly string[],
      plannedIncrementalCopyTensorNames: ReadonlySet<string> = new Set(),
      replaceWeightNames: ReadonlySet<string> = new Set(),
    ): StagedWebGPUTensorResources {
      if (!Number.isFinite(growthFactor) || growthFactor < 1) {
        throw new Error('WebGPU capacityGrowthFactor must be finite and at least 1.');
      }
      const resultCopyTensorNames = new Set(
        plannedResultCopyTensorNames ?? compileWebGPUGraphPlan(graph).resultCopyTensorNames,
      );
      const aliases = aliasDropout ? dropoutAliases(graph) : new Map<string, string>();
      const usageByRoot = new Map<string, GPUBufferUsageFlags>();
      for (const [name, tensor] of graph.tensors.entries()) {
        const root = aliasRoot(aliases, name);
        const usage = webGPUBufferUsage({
          graph,
          tensor,
          incrementalCopyTensorNames: plannedIncrementalCopyTensorNames,
          resultCopyTensorNames,
        });
        usageByRoot.set(root, (usageByRoot.get(root) || 0) | usage);
      }

      const buffers = new Map<string, GPUBuffer>();
      const usages = new Map<string, GPUBufferUsageFlags>();
      const capacities = new Map<string, number>();
      const created = new Set<GPUBuffer>();
      let logicalActivationBytes = 0;
      for (const tensor of graph.tensors.values()) {
        if (tensor.isWeight) continue;
        if (!Number.isSafeInteger(logicalActivationBytes + tensor.sizeBytes)) {
          throw new Error('WebGPU logical activation bytes exceed exact JavaScript arithmetic.');
        }
        logicalActivationBytes += tensor.sizeBytes;
      }
      let activationCapacityBytes = 0;
      let grew = false;
      try {
        for (const [name, tensor] of graph.tensors.entries()) {
          if (aliases.has(name)) continue;
          const requiredBytes = alignedBufferBytes(tensor.sizeBytes, `Tensor '${name}'`);
          const maximumBytes = tensorMaximumBytes?.get(name) ?? requiredBytes;
          const usage = usageByRoot.get(name) ?? webGPUBufferUsage({
            graph,
            tensor,
            incrementalCopyTensorNames: plannedIncrementalCopyTensorNames,
            resultCopyTensorNames,
          });
          const previous = this.host.gpuBuffers.get(name);
          const previousUsage = this.host.tensorBufferUsages.get(name);
          const previousCapacity = this.host.tensorCapacityBytes.get(name) || 0;
          // A bank residency change means equal-capacity storage has different
          // bytes. Allocate and upload a candidate instead of overwriting the
          // committed buffer, which may still be referenced by submitted work.
          const replaceWeight = tensor.isWeight && replaceWeightNames.has(name);
          const canReuse = !replaceWeight && previous !== undefined && previousUsage === usage &&
            previousCapacity >= requiredBytes;
          let buffer: GPUBuffer;
          let capacityBytes: number;
          if (canReuse) {
            buffer = previous;
            capacityBytes = previousCapacity;
          } else {
            capacityBytes = checkedCapacityTarget(
              requiredBytes,
              previousUsage === usage ? previousCapacity : 0,
              alignedBufferBytes(maximumBytes, `Tensor '${name}' maximum`),
              growthFactor,
              `Tensor '${name}'`,
            );
            buffer = createWebGPUBufferOrOOM(
              this.host.device,
              {
                label: `Tensor_${name}`,
                size: capacityBytes,
                usage,
              },
              `WebGPU tensor '${name}' capacity`,
            );
            created.add(buffer);
            if (!tensor.isWeight && previous !== undefined && capacityBytes > previousCapacity) {
              grew = true;
            }
            if (tensor.isWeight && tensor.buffer) this.uploadWeight(buffer, tensor);
          }
          buffers.set(name, buffer);
          usages.set(name, usage);
          capacities.set(name, capacityBytes);
          if (!tensor.isWeight) {
            if (!Number.isSafeInteger(activationCapacityBytes + capacityBytes)) {
              throw new Error('WebGPU activation capacity exceeds exact JavaScript arithmetic.');
            }
            activationCapacityBytes += capacityBytes;
          }
        }

        for (const [name] of aliases) {
          const root = aliasRoot(aliases, name);
          const buffer = buffers.get(root);
          const capacity = capacities.get(root);
          const usage = usages.get(root);
          if (!buffer || capacity === undefined || usage === undefined) {
            throw new Error(`WebGPU Dropout alias '${name}' has no root buffer '${root}'.`);
          }
          buffers.set(name, buffer);
          capacities.set(name, capacity);
          usages.set(name, usage);
        }
        return {
          buffers,
          usages,
          capacities,
          created,
          logicalActivationBytes,
          activationCapacityBytes,
          grew,
        };
      } catch (error) {
        destroyDistinctWebGPUBuffers(created);
        throw error;
      }
    }

    private uploadWeight(buffer: GPUBuffer, tensor: Tensor): void {
      const weight = tensor.buffer!;
      const source = weight instanceof ArrayBuffer
        ? new Uint8Array(weight)
        : new Uint8Array(weight.buffer, weight.byteOffset, weight.byteLength);
      const paddedBytes = Math.ceil(source.byteLength / 4) * 4;
      if (paddedBytes === source.byteLength) {
        this.host.device.queue.writeBuffer(buffer, 0, source);
      } else {
        const padded = new Uint8Array(paddedBytes);
        padded.set(source);
        this.host.device.queue.writeBuffer(buffer, 0, padded);
      }
    }

    _adapterBuffers(target: AdapterTarget): { a: GPUBuffer; b: GPUBuffer } {
      let buffers = this.host.adapterTargetBuffers.get(target);
      if (buffers) return buffers;
      const create = (label: string, values: Float32Array): GPUBuffer => {
        const buffer = this.host.device.createBuffer({
          label,
          size: Math.max(4, Math.ceil(values.byteLength / 4) * 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(buffer, 0, values.buffer, values.byteOffset, values.byteLength);
        return buffer;
      };
      buffers = {
        a: create(`LoRA_A_${target.weight}`, target.A),
        b: create(`LoRA_B_${target.weight}`, target.B),
      };
      this.host.adapterTargetBuffers.set(target, buffers);
      return buffers;
    }

    _createAuxiliaryStorageBuffer(label: string, values: ArrayBufferView): GPUBuffer {
      const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
      const buffer = this.host._createSpecializationBuffer({
        label,
        size: Math.max(4, Math.ceil(bytes.byteLength / 4) * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      // These tables are immutable shader inputs for the executor's exact
      // model revision. They are never dispatch outputs or copy destinations;
      // a weight/topology revision requires a new executor before dispatch.
      this.host._markSpecializationBufferInvariant(buffer);
      this.host._writeSpecializationBuffer(buffer, 0, bytes);
      return buffer;
    }

    releaseAdapterTargets(targets: readonly AdapterTarget[] | null | undefined): void {
      const retired: GPUBuffer[] = [];
      for (const target of targets || []) {
        const buffers = this.host.adapterTargetBuffers.get(target);
        if (!buffers) continue;
        this.host.adapterTargetBuffers.delete(target);
        retired.push(buffers.a, buffers.b);
      }
      const destroy = () => { for (const buffer of retired) buffer?.destroy?.(); };
      if (retired.length && this.host.device.queue.onSubmittedWorkDone) {
        this.host.device.queue.onSubmittedWorkDone().then(destroy, destroy);
      } else {
        destroy();
      }
    }

    _allocateBuffers(): void {
      const resultCopyTensorNames = new Set(
        this.host.compiledGraphPlan?.resultCopyTensorNames || [],
      );
      for (const [name, tensor] of this.host.graph.tensors.entries()) {
        const usage = webGPUBufferUsage({
          graph: this.host.graph as RuntimeGraph,
          tensor,
          incrementalCopyTensorNames: this.host.incrementalRowCopyTensorNames,
          resultCopyTensorNames,
        });
        const buffer = this.host.device.createBuffer({
          label: `Tensor_${name}`,
          size: alignedBufferBytes(tensor.sizeBytes, `Tensor '${name}'`),
          // Align to 4 bytes
          usage
        });
        this.host.gpuBuffers.set(name, buffer);
        this.host.tensorBufferUsages.set(name, usage);
        this.host.tensorCapacityBytes.set(
          name,
          alignedBufferBytes(tensor.sizeBytes, `Tensor '${name}'`),
        );
        // Upload learned-parameter data now. Weight buffers are consumed as node
        // inputs (so they carry COPY_DST); without this, Conv/MatMul/BatchNorm read
        // zeros on the GPU. Input tensors are written later at execute() time.
        if (tensor.isWeight && tensor.buffer) {
          this.uploadWeight(buffer, tensor);
        }
      }
    }

    _aliasInferenceDropoutBuffers(): void {
      aliasInferenceDropoutBuffers(this.host.graph as RuntimeGraph, this.host.gpuBuffers);
      for (const node of this.host.graph.nodes) {
        if (node.opType !== 'Dropout') continue;
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        if (!input || !output) continue;
        const capacity = this.host.tensorCapacityBytes.get(input.name);
        const usage = this.host.tensorBufferUsages.get(input.name);
        if (capacity !== undefined) this.host.tensorCapacityBytes.set(output.name, capacity);
        if (usage !== undefined) this.host.tensorBufferUsages.set(output.name, usage);
      }
    }
}
