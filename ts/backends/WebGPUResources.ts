import type { Graph } from '../core/Graph.js';
import type { Tensor } from '../core/Tensor.js';
import type { GraphExecutor } from './GraphExecutor.js';
import type { AdapterTarget } from './WebGPUContracts.js';

interface WebGPUResourcePlanOptions {
  graph: Graph;
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
  graph: Graph,
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
    this.host.adapterTargetBuffers.clear();
    this.host.auxiliaryBuffers.clear();
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
      const buffer = this.host.device.createBuffer({
        label,
        size: Math.max(4, Math.ceil(bytes.byteLength / 4) * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.host.device.queue.writeBuffer(buffer, 0, bytes);
      this.host.auxiliaryBuffers.add(buffer);
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
          graph: this.host.graph as Graph,
          tensor,
          incrementalCopyTensorNames: this.host.incrementalRowCopyTensorNames,
          resultCopyTensorNames,
        });
        const buffer = this.host.device.createBuffer({
          label: `Tensor_${name}`,
          size: Math.ceil(tensor.sizeBytes / 4) * 4,
          // Align to 4 bytes
          usage
        });
        this.host.gpuBuffers.set(name, buffer);
        // Upload learned-parameter data now. Weight buffers are consumed as node
        // inputs (so they carry COPY_DST); without this, Conv/MatMul/BatchNorm read
        // zeros on the GPU. Input tensors are written later at execute() time.
        if (tensor.isWeight && tensor.buffer) {
          let wbuf = tensor.buffer;
          const dw = this.host._dinWeights && this.host._dinWeights.get(name);
          if (dw) {   // transpose [d_in, d_out] -> [d_out, d_in] for the shader
            const { din, dout } = dw;
            const t = new Float32Array(din * dout);
            for (let k = 0; k < din; k++) for (let j = 0; j < dout; j++) t[j * din + k] = tensor.buffer[k * dout + j];
            wbuf = t;
          }
          const src = wbuf instanceof ArrayBuffer
            ? new Uint8Array(wbuf)
            : new Uint8Array(wbuf.buffer, wbuf.byteOffset, wbuf.byteLength);
          const padded = Math.ceil(src.byteLength / 4) * 4;
          if (padded === src.byteLength) {
            this.host.device.queue.writeBuffer(buffer, 0, src);
          } else {
            const tmp = new Uint8Array(padded);
            tmp.set(src);
            this.host.device.queue.writeBuffer(buffer, 0, tmp);
          }
        }
      }
    }

    _aliasInferenceDropoutBuffers(): void {
      aliasInferenceDropoutBuffers(this.host.graph as Graph, this.host.gpuBuffers);
    }
}
