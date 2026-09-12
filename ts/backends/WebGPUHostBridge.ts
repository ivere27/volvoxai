/** Device objects and asynchronous completion for C-owned graph execution.
 * Copies belong to result tickets, never to a mutable host address. */
import {
  GPU_BRIDGE_ABI_HASH, GPU_PENDING, GPU_ERROR, GPU_DEVICE_LOST, GPU_OK, GPU_NO_UNIFORM,
  GPU_DISPATCH_WORDS, GPU_VARIANT_WORDS, GPU_BINDING_WORDS,
  GPU_LIMIT_NAMES,
  type WasmGpuBridge,
} from '../generated/gpuBridge.js';
import type { WasmGpuBridgeHost } from '../core/WasmReleaseModule.js';

/** Full can execute on CPU before a device is prepared or when none is available. */
export const REFUSING_GPU_BRIDGE: WasmGpuBridge = Object.freeze({
  vx_gpu_available: () => 0,
  vx_gpu_limits: () => 0,
  vx_gpu_ensure: () => 0,
  vx_gpu_release: () => {},
  vx_gpu_invalidate: () => {},
  vx_gpu_begin: () => {},
  vx_gpu_encode: () => GPU_ERROR,
  vx_gpu_end: () => GPU_ERROR,
  vx_gpu_snapshot: () => GPU_ERROR,
  vx_gpu_readback: () => GPU_ERROR,
  vx_gpu_readback_release: () => {},
});

interface Span {
  readonly buffer: GPUBuffer;
  readonly bytes: number;
  deviceDirty: boolean;
  readonly isWeight: boolean;
  uploaded: boolean;
}
interface Completion { failed: boolean; done: Promise<void> }
interface Readback extends Completion {
  readonly staging: GPUBuffer;
  readonly bytes: number;
  readonly offset: number;
  ready: boolean;
}
/** The full release entry supplies its generated shader closure. */
export interface ShaderCatalog {
  readonly SHADER_CATALOG_HASH: string;
  readonly SHADER_NAMES: readonly string[];
  readonly SHADER_ENTRY_POINTS: readonly string[];
  readonly SHADER_LAYOUTS: readonly {
    readonly slots: readonly number[];
    readonly writesMask: number;
    readonly paramsBinding: number;
  }[];
  loadShaderPack(): Promise<readonly string[]>;
}
export interface WebGPUHostBridgeOptions {
  readonly device: GPUDevice;
  readonly catalog: ShaderCatalog;
  readonly onDiagnostic?: (message: string) => void;
}

/** Fixed WASM imports can delegate to a device prepared later at C's request. */
export function createDeferredWebGPUHostBridge(
  catalog: ShaderCatalog,
  onDiagnostic?: (message: string) => void,
): WasmGpuBridgeHost {
  let memory: WebAssembly.Memory | null = null;
  let delegate: WasmGpuBridgeHost | undefined;
  let wakeup = () => {};
  let preparing: Promise<void> | undefined;
  let closing: Promise<void> | undefined;
  let closed = false;
  const imports = Object.fromEntries(Object.entries(REFUSING_GPU_BRIDGE).map(([name, refuse]) => [
    name, (...args: number[]) => {
      const call = (closed ? undefined : delegate?.imports[name as keyof WasmGpuBridge]) ?? refuse;
      return (call as (...values: number[]) => number | void)(...args);
    },
  ])) as unknown as WasmGpuBridge;
  return {
    imports,
    setWakeup(callback): void { wakeup = callback; delegate?.setWakeup?.(callback); },
    attach(ownerMemory): void {
      if (closed) throw new Error('GPU bridge is closed.');
      if (memory && memory !== ownerMemory) throw new Error('GPU bridge already belongs to another owner.');
      memory = ownerMemory;
    },
    async prepare(): Promise<void> {
      if (closed) throw new Error('GPU bridge is closed.');
      if (!memory) throw new Error('GPU bridge memory is not attached.');
      preparing ??= (async () => {
        const ready = await acquireWebGPUHostBridge(catalog, onDiagnostic);
        if (!ready) return;
        if (closed) {
          ready.close?.();
          await ready.waitForCompletion();
          return;
        }
        try {
          ready.attach(memory!);
          ready.setWakeup?.(wakeup);
          delegate = ready;
        } catch (error) {
          ready.close?.();
          await ready.waitForCompletion();
          throw error;
        }
      })();
      await preparing;
    },
    waitForCompletion: () => closing ?? delegate?.waitForCompletion() ?? Promise.resolve(),
    close(): void {
      if (closed) return;
      closed = true;
      memory = null;
      wakeup = () => {};
      closing = (async () => {
        try { await preparing; } finally {
          const ready = delegate;
          delegate = undefined;
          ready?.close?.();
          await ready?.waitForCompletion();
        }
      })();
    },
  };
}

/** Browser device acquisition; backend admission remains a C compile decision. */
export async function acquireWebGPUHostBridge(
  catalog: ShaderCatalog,
  onDiagnostic?: (message: string) => void,
): Promise<WasmGpuBridgeHost | undefined> {
  const gpu = globalThis.navigator?.gpu;
  if (!gpu) return undefined;
  let device: GPUDevice | undefined;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return undefined;
    device = await adapter.requestDevice();
    const bridge = await createWebGPUHostBridge({ device, catalog, onDiagnostic });
    const ownedDevice = device;
    let closing: Promise<void> | undefined;
    return {
      imports: bridge.imports,
      attach: bridge.attach,
      setWakeup: bridge.setWakeup,
      waitForCompletion: () => closing ?? bridge.waitForCompletion(),
      close(): void {
        if (closing) return;
        bridge.close?.();
        closing = bridge.waitForCompletion().finally(async () => {
          ownedDevice.destroy();
          await ownedDevice.lost;
        });
      },
    };
  } catch (error) {
    device?.destroy();
    onDiagnostic?.(`WebGPU device unavailable: ${String(error)}`);
    return undefined;
  }
}

export async function createWebGPUHostBridge(
  options: WebGPUHostBridgeOptions,
): Promise<WasmGpuBridgeHost> {
  const { device } = options;
  const { loadShaderPack, SHADER_NAMES, SHADER_CATALOG_HASH, SHADER_ENTRY_POINTS, SHADER_LAYOUTS } = options.catalog;
  const diagnostic = (message: string) => {
    // C can call a GPU import while its call-runtime locks are held. User
    // diagnostics may cancel/close calls, so deliver them after that entry ends.
    if (options.onDiagnostic) queueMicrotask(() => options.onDiagnostic!(message));
  };
  const shaders = await loadShaderPack();
  let memory: WebAssembly.Memory | null = null;
  let lost = false;
  let closed = false;
  let passScopes = false;
  let mismatchReported = false;
  const spans = new Map<number, Span>();
  const pipelines = new Map<number, GPUComputePipeline>();
  const readbacks = new Map<number, Readback>();
  const retiring = new Set<Promise<void>>();
  let wakeup = () => {};
  let nextTicket = 1;
  let encoder: GPUCommandEncoder | null = null;
  let pass: GPUComputePassEncoder | null = null;
  let uniforms: GPUBuffer[] = [];
  let passFailed = false;
  let submission: Completion = { failed: false, done: Promise.resolve() };
  const deviceLost = device.lost.then((info) => {
    lost = true;
    wakeup();
    if (!closed || info.reason !== 'destroyed') {
      diagnostic(`GPU bridge: device lost. ${info.message}`);
    }
  });

  function linear(): WebAssembly.Memory {
    if (!memory) throw new Error('GPU bridge memory is not attached.');
    return memory;
  }
  const words = () => new Uint32Array(linear().buffer);
  const bytesAt = (pointer: number, length: number) =>
    new Uint8Array(linear().buffer, pointer, length);
  const aligned = (bytes: number) => Math.ceil(bytes / 4) * 4;

  function openScopes(): void {
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
  }
  function closeScopes(completion: Completion): Promise<void> {
    // Pop synchronously: a later C dispatch must not enter this operation's scopes.
    const pop = async () => device.popErrorScope();
    return Promise.allSettled([pop(), pop()]).then((errors) => {
      for (const outcome of errors) {
        const error = outcome.status === 'fulfilled' ? outcome.value : outcome.reason;
        if (!error) continue;
        completion.failed = true;
        diagnostic(`GPU bridge: ${String(error.message ?? error)}`);
      }
    });
  }
  async function queueDone(): Promise<void> {
    await device.queue.onSubmittedWorkDone();
  }
  function complete(completion: Completion, work: readonly Promise<unknown>[]): Promise<void> {
    return Promise.race([
      Promise.allSettled(work).then((outcomes) => {
        for (const outcome of outcomes) if (outcome.status === 'rejected') {
          completion.failed = true;
          diagnostic(`GPU bridge: completion failed. ${String(outcome.reason)}`);
        }
      }), deviceLost,
    ]).then(() => { completion.failed ||= lost || closed; wakeup(); });
  }
  function retire(buffer: GPUBuffer, after?: Promise<void>): void {
    if (!after && pass) { uniforms.push(buffer); return; }
    const released = Promise.race([after ?? queueDone(), deviceLost])
      .catch(() => {}).then(() => {
        try { buffer.destroy(); } finally { retiring.delete(released); }
      }).catch((error) => { diagnostic(`GPU bridge: retirement failed. ${String(error)}`); });
    retiring.add(released);
  }
  function pipelineFor(shaderId: number): GPUComputePipeline | null {
    const cached = pipelines.get(shaderId);
    if (cached) return cached;
    const code = shaders[shaderId];
    if (code === undefined) return null;
    try {
      const declared = SHADER_LAYOUTS[shaderId]!;
      const entries: GPUBindGroupLayoutEntry[] = declared.slots.map((binding, index) => ({
        binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: (declared.writesMask >> index) & 1 ? 'storage' : 'read-only-storage' },
      }));
      if (declared.paramsBinding !== GPU_NO_UNIFORM) entries.push({
        binding: declared.paramsBinding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      });
      const pipeline = device.createComputePipeline({
        label: SHADER_NAMES[shaderId],
        layout: device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries })] }),
        compute: { module: device.createShaderModule({ code }), entryPoint: SHADER_ENTRY_POINTS[shaderId] },
      });
      pipelines.set(shaderId, pipeline);
      return pipeline;
    } catch (error) {
      diagnostic(`GPU bridge: pipeline failed. ${String(error)}`);
      return null;
    }
    // Asynchronous validation is part of submission.done. An invalid pipeline
    // cannot turn into a successful result even if creation returned an object.
  }

  const imports: WasmGpuBridge = Object.freeze({
    vx_gpu_available(abiPointer: number, hashPointer: number): number {
      if (closed || lost) return 0;
      const hash = new TextDecoder().decode(bytesAt(hashPointer, 64));
      const abi = new TextDecoder().decode(bytesAt(abiPointer, 64));
      const matches = abi === GPU_BRIDGE_ABI_HASH && hash === SHADER_CATALOG_HASH;
      if (!matches && !mismatchReported) {
        mismatchReported = true;
        diagnostic('GPU bridge: WASM/host ABI or shader catalogue mismatch.');
      }
      return matches && !lost && !closed ? 1 : 0;
    },
    vx_gpu_limits(destination: number, bytes: number): number {
      if (lost || closed || bytes !== GPU_LIMIT_NAMES.length * 4) return 0;
      const limits = GPU_LIMIT_NAMES.map((name) => device.limits[name]);
      if (limits.some((value) => !Number.isSafeInteger(value) || value < 0)) return 0;
      new Uint32Array(linear().buffer, destination, limits.length)
        .set(limits.map((value) => Math.min(value, 0xfffffffc)));
      return 1;
    },
    vx_gpu_ensure(hostPointer: number, bytes: number, isWeight: number): number {
      if (lost || closed || !hostPointer || bytes <= 0) return 0;
      let span = spans.get(hostPointer);
      if (span && span.bytes !== bytes) {
        retire(span.buffer);
        spans.delete(hostPointer);
        span = undefined;
      }
      try {
        if (!span) {
          span = {
            buffer: device.createBuffer({
              size: aligned(bytes), usage: GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            }),
            bytes, deviceDirty: false, isWeight: isWeight !== 0, uploaded: false,
          };
          spans.set(hostPointer, span);
        }
        if (!span.deviceDirty && !(span.isWeight && span.uploaded)) {
          const source = bytesAt(hostPointer, bytes);
          if (bytes % 4 === 0) device.queue.writeBuffer(span.buffer, 0, source);
          else {
            const padded = new Uint8Array(aligned(bytes));
            padded.set(source);
            device.queue.writeBuffer(span.buffer, 0, padded);
          }
          span.uploaded = true;
        }
        return 1;
      } catch (error) {
        passFailed = true;
        diagnostic(`GPU bridge: residency failed. ${String(error)}`);
        if (span) retire(span.buffer);
        spans.delete(hostPointer);
        return 0;
      }
    },
    vx_gpu_invalidate(hostPointer: number): void {
      const span = spans.get(hostPointer);
      if (span) { span.deviceDirty = false; span.uploaded = false; }
    },
    vx_gpu_release(hostPointer: number): void {
      const span = spans.get(hostPointer);
      if (span) retire(span.buffer);
      spans.delete(hostPointer);
    },
    vx_gpu_begin(): void {
      if (closed) { passFailed = true; return; }
      uniforms = [];
      passFailed = lost || closed;
      try {
        openScopes();
        passScopes = true;
        encoder = device.createCommandEncoder();
        pass = encoder.beginComputePass();
      } catch (error) {
        passFailed = true;
        diagnostic(`GPU bridge: pass creation failed. ${String(error)}`);
      }
    },
    vx_gpu_encode(dispatch: number): number {
      if (!pass || lost || closed) return GPU_ERROR;
      try {
        const view = words();
        const base = dispatch >>> 2;
        const variantCount = view[base]!;
        const bindingCount = view[base + 1]!;
        const paramsBytes = view[base + 2]!;
        const paramsSlot = view[base + 4]!;
        const variantBase = base + GPU_DISPATCH_WORDS;
        const bindingBase = variantBase + variantCount * GPU_VARIANT_WORDS;
        const paramsBase = bindingBase + bindingCount * GPU_BINDING_WORDS;

        const entries: GPUBindGroupEntry[] = [];
        /* Spans this dispatch stores into. The engine says which; nothing here
         * guesses, because a guess is what leaves inputs looking device-owned. */
        const written: Span[] = [];
        for (let index = 0; index < bindingCount; index++) {
          const at = bindingBase + index * GPU_BINDING_WORDS;
          const span = spans.get(view[at + 1]!);
          if (!span) {
            diagnostic('GPU bridge: a binding named a span with no device buffer.');
            passFailed = true;
            return GPU_ERROR;
          }
          /* Storage bindings are measured in whole words. A byte tensor whose
           * element count is not a multiple of four still occupies a padded
           * buffer, and binding its exact length would be rejected. */
          entries.push({
            binding: view[at]!,
            resource: {
              buffer: span.buffer,
              offset: view[at + 2]!,
              size: (view[at + 3]! + 3) & ~3,
            },
          });
          if (view[at + 4]) written.push(span);
        }

        /* Uniform bytes travel inline with the descriptor: they are this node's
         * shape, not a tensor, and exist nowhere the engine could name. */
        let params: GPUBuffer | null = null;
        if (paramsBytes > 0) {
          /* Rounded to 16: WGSL lays uniform structures out on that boundary,
           * and a device is free to refuse a binding that is smaller than the
           * structure the shader declares. The extra bytes are never read. */
          params = device.createBuffer({
            size: (paramsBytes + 15) & ~15,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          uniforms.push(params);
          device.queue.writeBuffer(params, 0, bytesAt(paramsBase << 2, paramsBytes));
          if (paramsSlot === GPU_NO_UNIFORM) {
            diagnostic('GPU bridge: uniform bytes arrived with no slot to bind them.');
            passFailed = true;
            return GPU_ERROR;
          }
          /* The slot the shader declares, not the next free one: a uniform is
           * not always numbered after the storage buffers. */
          entries.push({ binding: paramsSlot, resource: { buffer: params } });
        }

        for (let variant = 0; variant < variantCount; variant++) {
          const at = variantBase + variant * GPU_VARIANT_WORDS;
          const pipeline = pipelineFor(view[at]!);
          if (!pipeline) continue;
          try {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries,
            }));
            pass.dispatchWorkgroups(view[at + 1]!, view[at + 2]!, view[at + 3]!);
          } catch (error) {
            diagnostic(`GPU bridge: encode failed. ${String(error)}`);
            passFailed = true;
            return GPU_ERROR;
          }
          for (const span of written) span.deviceDirty = true;
          return variant;
        }
        diagnostic('GPU bridge: no candidate shader built on this device.');
        passFailed = true;
        return GPU_ERROR;
      } catch (error) {
        passFailed = true;
        diagnostic(`GPU bridge: encoding failed. ${String(error)}`);
        return GPU_ERROR;
      }
    },

    vx_gpu_end(): number {
      const current: Completion = { failed: passFailed || lost || closed || !encoder || !pass,
        done: Promise.resolve() };
      try {
        pass?.end();
        if (!current.failed && encoder) device.queue.submit([encoder.finish()]);
      } catch (error) {
        current.failed = true;
        diagnostic(`GPU bridge: submission failed. ${String(error)}`);
      }
      encoder = null;
      pass = null;
      current.done = complete(current, [
        ...(passScopes ? [closeScopes(current)] : []), queueDone(),
      ]);
      passScopes = false;
      for (const uniform of uniforms) retire(uniform, current.done);
      uniforms = [];
      submission = current;
      return current.failed ? GPU_ERROR : GPU_OK;
    },
    vx_gpu_snapshot(hostPointer: number, offset: number, bytes: number): number {
      const span = spans.get(hostPointer);
      if (lost || closed || !span || bytes <= 0 || offset + bytes > span.bytes || nextTicket > 0x7fffffff)
        return GPU_ERROR;
      // WebGPU copies aligned words. The ticket retains any leading bytes so
      // Readback can expose an unaligned int8 tensor slice without changing it.
      const start = Math.floor(offset / 4) * 4;
      const length = aligned(offset + bytes) - start;
      openScopes();
      let staging: GPUBuffer;
      try {
        staging = device.createBuffer({ size: length,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      } catch (error) {
        const failed: Completion = { failed: true, done: Promise.resolve() };
        const done = complete(failed, [closeScopes(failed), queueDone()]);
        retiring.add(done);
        void done.then(() => retiring.delete(done));
        diagnostic(`GPU bridge: staging allocation failed. ${String(error)}`);
        return GPU_ERROR;
      }
      const job: Readback = {
        staging,
        bytes, offset: offset - start, failed: false, ready: false, done: Promise.resolve(),
      };
      const owner = submission;
      let mapped: Promise<void>;
      try {
        const copy = device.createCommandEncoder();
        copy.copyBufferToBuffer(span.buffer, start, job.staging, 0, length);
        device.queue.submit([copy.finish()]);
        mapped = job.staging.mapAsync(GPUMapMode.READ);
      } catch (error) {
        job.failed = true;
        mapped = Promise.resolve();
        diagnostic(`GPU bridge: snapshot failed. ${String(error)}`);
      }
      job.done = complete(job, [mapped, owner.done, closeScopes(job), queueDone()]).then(() => {
        job.failed ||= owner.failed;
        job.ready = !job.failed;
        wakeup();
      });
      const ticket = nextTicket++;
      readbacks.set(ticket, job);
      return ticket;
    },
    vx_gpu_readback(ticket: number, destination: number, bytes: number): number {
      const job = readbacks.get(ticket);
      if (lost) return GPU_DEVICE_LOST;
      if (closed || !job || job.failed || job.bytes !== bytes) return GPU_ERROR;
      if (!job.ready) return GPU_PENDING;
      // This is the only write into WASM. No promise closes over a host pointer.
      try {
        bytesAt(destination, bytes).set(new Uint8Array(job.staging.getMappedRange(), job.offset, bytes));
        return GPU_OK;
      } catch (error) {
        job.failed = true;
        diagnostic(`GPU bridge: readback failed. ${String(error)}`);
        return GPU_ERROR;
      }
    },
    vx_gpu_readback_release(ticket: number): void {
      const job = readbacks.get(ticket);
      if (!job) return;
      readbacks.delete(ticket);
      retire(job.staging, job.done);
    },
  });
  return Object.freeze({
    imports,
    setWakeup(callback: () => void): void { wakeup = callback; },
    attach(attached: WebAssembly.Memory): void {
      if (memory || closed) throw new Error('GPU bridge already belongs to a WASM instance.');
      memory = attached;
    },
    close(): void {
      if (closed) return;
      closed = true;
      if (passScopes) imports.vx_gpu_end();
      for (const span of spans.values()) retire(span.buffer);
      spans.clear();
      for (const ticket of readbacks.keys()) imports.vx_gpu_readback_release(ticket);
      pipelines.clear();
      memory = null;
      wakeup = () => {};
    },
    async waitForCompletion(): Promise<void> {
      await Promise.all([submission.done, ...retiring, ...[...readbacks.values()].map((job) => job.done)]);
    },
  });
}
