import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import { Tensor } from '../core/Tensor.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import type { GraphExecutor } from './GraphExecutor.js';
import { submitWebGPUBufferCopies, type WebGPUBufferCopy } from './WebGPUDispatch.js';
import { captureWebGPUErrorScopesSync } from './WebGPUErrorScopes.js';
import { createWebGPUBufferOrOOM } from './WebGPUResources.js';

export interface WebGPUOutputSnapshot {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: RuntimeDType;
  readonly sizeBytes: number;
  readonly deviceBuffer: GPUBuffer;
}

function destroyBuffers(buffers: Iterable<GPUBuffer>): void {
  for (const buffer of new Set(buffers)) buffer.destroy?.();
}

function retireSubmittedBuffers(device: GPUDevice, buffers: Iterable<GPUBuffer>): void {
  const retained = [...new Set(buffers)];
  if (retained.length === 0) return;
  const destroy = () => destroyBuffers(retained);
  try {
    const fence = device.queue.onSubmittedWorkDone?.();
    if (!fence || typeof fence.then !== 'function') return;
    void fence.then(destroy, destroy);
  } catch {
    // A broken/lost queue cannot prove retirement. Keep the objects alive for
    // the device lifetime rather than destroying storage submitted work may
    // still reference.
  }
}

export async function snapshotWebGPUOutputs(
  device: GPUDevice,
  graph: RuntimeGraph,
  buffers: ReadonlyMap<string, GPUBuffer>,
): Promise<ReadonlyMap<string, WebGPUOutputSnapshot>> {
  const snapshots = new Map<string, WebGPUOutputSnapshot>();
  const created: GPUBuffer[] = [];
  const copies: WebGPUBufferCopy[] = [];
  let submissionAttempted = false;
  try {
    const captured = captureWebGPUErrorScopesSync(
      device,
      { label: 'WebGPU result snapshot allocation' },
      () => {
        for (const name of graph.outputNames || []) {
          const tensor = graph.tensors.get(name);
          const source = buffers.get(name);
          if (!tensor || !source) {
            throw new Error(`WebGPU declared output '${name}' has no allocated device tensor.`);
          }
          const paddedSize = Math.ceil(tensor.sizeBytes / 4) * 4;
          const deviceBuffer = createWebGPUBufferOrOOM(
            device,
            {
              label: `Result_${name}`,
              size: paddedSize,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            },
            `WebGPU result '${name}' snapshot`,
          );
          created.push(deviceBuffer);
          copies.push({ source, destination: deviceBuffer, size: paddedSize });
          snapshots.set(name, Object.freeze({
            name,
            shape: Object.freeze([...tensor.shape]),
            dtype: tensor.dtype,
            sizeBytes: tensor.sizeBytes,
            deviceBuffer,
          }));
        }
      },
    );
    await captured.check;
    // Every source, destination, range and usage was checked before this
    // point. Do not add a second per-result error-scope round trip to the hot
    // path; allocation is the fallible resource boundary we must resolve.
    submissionAttempted = true;
    submitWebGPUBufferCopies(device, copies);
    return snapshots;
  } catch (error) {
    if (submissionAttempted) retireSubmittedBuffers(device, created);
    else destroyBuffers(created);
    throw error;
  }
}

export function destroyWebGPUOutputSnapshots(
  snapshots: ReadonlyMap<string, WebGPUOutputSnapshot>,
): void {
  for (const snapshot of snapshots.values()) snapshot.deviceBuffer.destroy?.();
}

export async function readWebGPUBufferRange(
  device: GPUDevice,
  gpuBuffer: GPUBuffer,
  byteOffset: number,
  sizeBytes: number,
  dtype: RuntimeDType = 'float32',
): Promise<RuntimeTypedArray> {
  const elementBytes = Tensor.dtypeBytes(dtype);
  if (!gpuBuffer || !Number.isSafeInteger(byteOffset) || byteOffset < 0 ||
      byteOffset % elementBytes !== 0 || !Number.isSafeInteger(sizeBytes) ||
      sizeBytes <= 0 || sizeBytes % elementBytes !== 0) {
    throw new Error(`WebGPU readBufferRange requires a non-negative ${dtype}-aligned offset and positive aligned logical byte size.`);
  }
  const copyOffset = Math.floor(byteOffset / 4) * 4;
  const prefixBytes = byteOffset - copyOffset;
  const paddedSize = Math.ceil((prefixBytes + sizeBytes) / 4) * 4;
  let stagingBuffer: GPUBuffer | null = null;
  let mapped = false;
  let submissionAttempted = false;
  try {
    const captured = captureWebGPUErrorScopesSync(
      device,
      { label: 'WebGPU result readback allocation', phase: 'readback' },
      () => {
        const candidate = createWebGPUBufferOrOOM(
          device,
          {
            size: paddedSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          },
          'WebGPU result readback staging buffer',
        );
        stagingBuffer = candidate;
        return candidate;
      },
    );
    stagingBuffer = captured.value;
    await captured.check;
    // The public range checks above and the exact staging allocation make this
    // copy structurally valid. Scope only the resource allocation so ordinary
    // readback does not pay another asynchronous scope resolution.
    submissionAttempted = true;
    submitWebGPUBufferCopies(device, [{
      source: gpuBuffer,
      sourceOffset: copyOffset,
      destination: stagingBuffer,
      size: paddedSize,
    }]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    mapped = true;
    const logicalBytes = stagingBuffer.getMappedRange().slice(
      prefixBytes, prefixBytes + sizeBytes,
    );
    if (dtype === 'float32') return new Float32Array(logicalBytes);
    if (dtype === 'int32') return new Int32Array(logicalBytes);
    if (dtype === 'int8') return new Int8Array(logicalBytes);
    if (dtype === 'uint8') return new Uint8Array(logicalBytes);
    throw new Error(`WebGPU readBuffer does not support dtype '${dtype}'.`);
  } finally {
    if (mapped) stagingBuffer?.unmap();
    if (stagingBuffer) {
      if (!mapped && submissionAttempted) retireSubmittedBuffers(device, [stagingBuffer]);
      else stagingBuffer.destroy?.();
    }
  }
}

/** Stable result snapshots and aligned host readback. */
export class WebGPUResults {
  constructor(readonly host: GraphExecutor) {}

  snapshotOutputs(): Promise<ReadonlyMap<string, WebGPUOutputSnapshot>> {
    if (this.host.compiledTopologyRevision == null) {
      throw new Error('WebGPU snapshotOutputs requires a compiled graph.');
    }
    return snapshotWebGPUOutputs(this.host.device, this.host.graph as RuntimeGraph, this.host.gpuBuffers);
  }

  readBuffer(
    gpuBuffer: GPUBuffer,
    sizeBytes: number,
    dtype: RuntimeDType = 'float32',
  ): Promise<RuntimeTypedArray> {
    return this.readBufferRange(gpuBuffer, 0, sizeBytes, dtype);
  }

  readBufferRange(
    gpuBuffer: GPUBuffer,
    byteOffset: number,
    sizeBytes: number,
    dtype: RuntimeDType = 'float32',
  ): Promise<RuntimeTypedArray> {
    return readWebGPUBufferRange(this.host.device, gpuBuffer, byteOffset, sizeBytes, dtype);
  }
}
