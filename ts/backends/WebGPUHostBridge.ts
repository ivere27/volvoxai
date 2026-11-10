/** Device objects and asynchronous completion for C-owned graph execution.
 * Copies belong to result tickets, never to a mutable host address. */
import {
  GPU_BRIDGE_ABI_HASH, GPU_PENDING, GPU_ERROR, GPU_DEVICE_LOST, GPU_OK, GPU_NO_UNIFORM,
  GPU_RESIDENCY_WEIGHT, GPU_RESIDENCY_OVERWRITE,
  GPU_DISPATCH_WORDS, GPU_VARIANT_WORDS, GPU_BINDING_WORDS,
  GPU_LIMIT_NAMES,
  type WasmGpuBridge,
} from '../generated/gpuBridge.js';
import { TraceActivity, MemorySpace } from '../generated/volvoxaiEnums.js';
import type { WasmGpuBridgeHost } from '../core/WasmReleaseModule.js';

/** Full can execute on CPU before a device is prepared or when none is available. */
export const REFUSING_GPU_BRIDGE: WasmGpuBridge = Object.freeze({
  vx_gpu_available: () => 0,
  vx_gpu_limits: () => 0,
  vx_gpu_ensure: () => 0,
  vx_gpu_release: () => {},
  vx_gpu_invalidate: () => {},
  vx_gpu_memory_start: () => 0,
  vx_gpu_memory_stop: () => {},
  vx_gpu_begin: () => {},
  vx_gpu_begin_activity: () => {},
  vx_gpu_begin_trace: () => 0,
  vx_gpu_trace_node_begin: () => 0,
  vx_gpu_trace_node_end: () => {},
  vx_gpu_trace_program_begin: () => 0,
  vx_gpu_trace_program_end: () => {},
  vx_gpu_debug_label: () => 0,
  vx_gpu_trace_read: () => GPU_ERROR,
  vx_gpu_trace_release: () => {},
  vx_gpu_await_read: () => GPU_ERROR,
  vx_gpu_await_release: () => {},
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
  lastUse?: Completion;
}
interface Completion { failed: boolean; done: Promise<void>; settled?: boolean }
interface TimestampBuffers {
  readonly queries: GPUQuerySet;
  readonly resolve: GPUBuffer;
  readonly staging: GPUBuffer;
}
interface Timestamp extends Completion, TimestampBuffers {
  readonly capacity: number;
  readonly nodes: boolean;
  count: number;
  references: number;
  retired: boolean;
  values?: BigUint64Array;
  hostBefore?: bigint;
  hostAfter?: bigint;
  readonly finish: () => void;
}
interface Readback extends Completion {
  // Absent for a zero-byte completion fence, which owns no GPU allocation.
  readonly staging?: GPUBuffer;
  readonly bytes: number;
  readonly offset: number;
  ready: boolean;
}
// Queue writes retain host-visible staging storage until a submission. Large
// graphs can exhaust that heap before their first compute pass is submitted.
const MAX_PENDING_UPLOAD_BYTES = 16 * 1024 * 1024;
// Only completed submissions contribute reusable parameter storage. Bound both
// the driver objects and their payload; larger graphs still execute normally.
const MAX_POOLED_PARAMS = 4;
const MAX_POOLED_PARAMS_BYTES = 4 * 1024 * 1024;
const PARAMS_CHUNK_BYTES = 1024 * 1024;
const MAX_INACTIVE_SPANS_PER_POINTER = 4;
const MAX_INACTIVE_SPAN_BYTES = 4 * 1024 * 1024;
const MAX_CACHED_BIND_GROUPS = 4096;
interface ParamsChunk {
  readonly buffer: GPUBuffer;
  readonly staging: Uint8Array;
  usedBytes: number;
}
// Prepared only on explicit device-timing admission. Reused after mapping and
// ticket retirement; an exhausted pool loses observations, never numerical work.
const TIMESTAMP_BATCHES = 4;
const TIMESTAMP_INTERVALS = 1024;
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
  let memoryEvent: Parameters<WasmGpuBridgeHost['attach']>[1];
  let activityEvent: Parameters<WasmGpuBridgeHost['attach']>[2];
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
    attach(ownerMemory, callback, activityCallback): void {
      memoryEvent = callback; activityEvent = activityCallback;
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
          ready.attach(memory!, memoryEvent, activityEvent);
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
    async prepareTracing(): Promise<void> {
      await preparing;
      await delegate?.prepareTracing?.();
    },
    waitForCompletion: () => closing ?? delegate?.waitForCompletion() ?? Promise.resolve(),
    close(): void {
      if (closed) return;
      closed = true;
      memory = null;
      memoryEvent = undefined;
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
    // Request the adapter's supported storage limits so a model is admitted
    // against the actual device, rather than an unnecessarily smaller default.
    // This requests capabilities; it does not allocate or reserve model memory.
    const requiredLimits: Record<string, number> = {};
    for (const name of ['maxBufferSize', 'maxStorageBufferBindingSize'] as const) {
      const value = adapter.limits?.[name];
      if (Number.isSafeInteger(value) && value > 0) requiredLimits[name] = value;
    }
    device = await adapter.requestDevice({ requiredLimits, requiredFeatures:
      adapter.features.has('timestamp-query') ? ['timestamp-query'] : [] });
    const bridge = await createWebGPUHostBridge({ device, catalog, onDiagnostic });
    const ownedDevice = device;
    let closing: Promise<void> | undefined;
    return {
      imports: bridge.imports,
      attach: bridge.attach,
      prepareTracing: bridge.prepareTracing,
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
  const inactiveSpans = new Map<number, Map<number, Span>>();
  let inactiveSpanBytes = 0;
  interface PipelineRecord {
    readonly pipeline: GPUComputePipeline;
    readonly layout: GPUBindGroupLayout;
  }
  const pipelines = new Map<number, PipelineRecord>();
  const bindGroupCache = new Map<string, {group: GPUBindGroup; buffers: GPUBuffer[]}>();
  const bufferToKeys = new Map<GPUBuffer, Set<string>>();
  let nextBufferId = 1;
  const bufferIds = new WeakMap<GPUBuffer, number>();
  function bufferId(b: GPUBuffer): number {
    let id = bufferIds.get(b);
    if (!id) { id = nextBufferId++; bufferIds.set(b, id); }
    return id;
  }
  let currentPipeline: GPUComputePipeline | null = null;
  let currentBindGroup: GPUBindGroup | null = null;
  const readbacks = new Map<number, Readback>();
  interface Interval {batch: Timestamp; index: number; endIndex: number; incomplete?: boolean}
  const timestamps = new Map<number, Interval>();
  let activeNode: Interval | undefined;
  // Released tickets still own GPU storage until their submission/map drains.
  let timestampResources = 0;
  const timestampPool: TimestampBuffers[] = [];
  let preparingTimestamps: Promise<void> | undefined;
  let timestamp: Timestamp | undefined;
  const retiring = new Map<Promise<void>, GPUBuffer | Timestamp | undefined>();
  let wakeup = () => {};
  let nextTicket = 1;
  let encoder: GPUCommandEncoder | null = null;
  let pass: GPUComputePassEncoder | null = null;
  // Set only by an annotating trace (TraceOptions.external): passes and the
  // dispatches encoded while it is set carry the node's label, so Dawn/wgpu
  // validation messages and native capture tools name the node.
  let debugLabel: string | undefined;
  let uniforms: GPUBuffer[] = [];
  let activeChunk: ParamsChunk | null = null;
  const activeParamsChunks: ParamsChunk[] = [];
  const paramsPool: ParamsChunk[] = [];
  let pooledParamsBytes = 0;
  const pendingParams = new Map<Promise<void>, GPUBuffer[]>();
  let passFailed = false;
  let pendingUploadBytes = 0;
  let activityEnabled = false;
  const awaits = new Map<number, {start: bigint; end?: bigint; failed: boolean}>();
  const hostNs = () => BigInt(Math.floor(performance.now() * 1000)) * 1000n;
  function activity(kind: number, source: number, destination: number, bytes: number,
    start: number, end: number, ticket = 0): void {
    activityEvent?.(kind, source, destination, bytes, ticket, start, end);
  }
  function resultView(destination: number): DataView {
    const view = new DataView(linear().buffer, destination, 40);
    for (let at = 0; at < 40; at += 8) view.setBigUint64(at, 0n, true);
    return view;
  }
  let submission: Completion = { failed: false, done: Promise.resolve() };
  let encoding: Completion | null = null;
  const lossWaiters = new Set<() => void>();
  void device.lost.then((info) => {
    lost = true;
    destroyParamsPool();
    drainInactive();
    bindGroupCache.clear();
    bufferToKeys.clear();
    currentPipeline = null;
    currentBindGroup = null;
    for (const finish of lossWaiters) finish();
    lossWaiters.clear();
    wakeup();
    if (!closed || info.reason !== 'destroyed') {
      diagnostic(`GPU bridge: device lost. ${info.message}`);
    }
  });

  let memoryEvent: Parameters<WasmGpuBridgeHost['attach']>[1];
  let activityEvent: Parameters<WasmGpuBridgeHost['attach']>[2];
  interface MemoryObserver { capacity: number; next: number; buffers: Map<GPUBuffer, number> }
  const memoryObservers = new Map<number, MemoryObserver>();
  function memoryCreated(buffer: GPUBuffer, action: number, only?: number): void {
    for (const [id, observer] of memoryObservers) {
      if (only !== undefined && id !== only || observer.buffers.has(buffer)) continue;
      if (observer.buffers.size >= observer.capacity || observer.next === 0xffffffff) {
        memoryEvent?.(id, 3, 0, 0, 0);
        continue;
      }
      const resource = ++observer.next;
      observer.buffers.set(buffer, resource);
      memoryEvent?.(id, action, resource, buffer.size >>> 0, Math.floor(buffer.size / 0x100000000));
    }
  }
  function createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    if (memoryObservers.size) memoryCreated(buffer, 1);
    return buffer;
  }
  function destroyBuffer(buffer: GPUBuffer): void {
    const keys = bufferToKeys.get(buffer);
    if (keys) {
      for (const key of keys) evictBindGroup(key);
    }
    buffer.destroy();
    if (memoryObservers.size) for (const [id, observer] of memoryObservers) {
      const resource = observer.buffers.get(buffer);
      if (resource === undefined) continue;
      observer.buffers.delete(buffer);
      memoryEvent?.(id, 2, resource, buffer.size >>> 0, Math.floor(buffer.size / 0x100000000));
    }
  }

  function evictBindGroup(key: string): void {
    const entry = bindGroupCache.get(key);
    if (!entry) return;
    bindGroupCache.delete(key);
    for (const buffer of entry.buffers) {
      const keys = bufferToKeys.get(buffer);
      keys?.delete(key);
      if (keys?.size === 0) bufferToKeys.delete(buffer);
    }
  }

  function removeInactive(pointer: number, bytes: number): Span | undefined {
    const pool = inactiveSpans.get(pointer);
    const span = pool?.get(bytes);
    if (span) {
      pool!.delete(bytes);
      inactiveSpanBytes -= span.buffer.size;
      if (pool!.size === 0) inactiveSpans.delete(pointer);
    }
    return span;
  }

  function drainInactive(pointer?: number): void {
    for (const [at, pool] of inactiveSpans) {
      if (pointer !== undefined && pointer !== at) continue;
      for (const [bytes, span] of pool) {
        removeInactive(at, bytes);
        retireInactive(span);
      }
    }
  }

  function retireInactive(span: Span): void {
    if (span.lastUse?.settled) destroyBuffer(span.buffer);
    else retire(span.buffer);
  }

  function retainInactive(pointer: number, span: Span): void {
    if (span.buffer.size > MAX_INACTIVE_SPAN_BYTES) {
      spans.delete(pointer);
      retireInactive(span);
      return;
    }
    let pool = inactiveSpans.get(pointer);
    if (!pool) {
      pool = new Map();
      inactiveSpans.set(pointer, pool);
    }
    const replaced = pool.get(span.bytes);
    // Register the destination before relinquishing the active owner.
    pool.set(span.bytes, span);
    inactiveSpanBytes += span.buffer.size - (replaced?.buffer.size ?? 0);
    spans.delete(pointer);
    if (replaced && replaced !== span) retireInactive(replaced);
  }

  function trimInactive(pointer: number): void {
    const pool = inactiveSpans.get(pointer);
    while (pool && pool.size > MAX_INACTIVE_SPANS_PER_POINTER) {
      const bytes = pool.keys().next().value!;
      retireInactive(removeInactive(pointer, bytes)!);
    }
    while (inactiveSpanBytes > MAX_INACTIVE_SPAN_BYTES) {
      const [at, oldest] = inactiveSpans.entries().next().value!;
      retireInactive(removeInactive(at, oldest.keys().next().value!)!);
    }
  }

  function linear(): WebAssembly.Memory {
    if (!memory) throw new Error('GPU bridge memory is not attached.');
    return memory;
  }
  const words = () => new Uint32Array(linear().buffer);
  const bytesAt = (pointer: number, length: number) =>
    new Uint8Array(linear().buffer, pointer, length);
  const aligned = (bytes: number) => Math.ceil(bytes / 4) * 4;

  function submit(commands: GPUCommandBuffer[]): void {
    const start = activityEnabled ? performance.now() * 1000 : 0;
    device.queue.submit(commands);
    if (activityEnabled) activity(TraceActivity.Submit, 0, 0, 0, start, performance.now() * 1000);
    pendingUploadBytes = 0;
  }
  function upload(buffer: GPUBuffer, source: Uint8Array): void {
    for (let offset = 0; offset < source.byteLength;) {
      const length = Math.min(MAX_PENDING_UPLOAD_BYTES, source.byteLength - offset);
      if (pendingUploadBytes + length > MAX_PENDING_UPLOAD_BYTES) submit([]);
      const start = activityEnabled ? performance.now() * 1000 : 0;
      device.queue.writeBuffer(buffer, offset, source.subarray(offset, offset + length));
      if (activityEnabled) activity(TraceActivity.Copy, MemorySpace.Host,
        MemorySpace.Device, length, start, performance.now() * 1000);
      pendingUploadBytes += length;
      offset += length;
    }
  }

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
    if (!activityEnabled || !activityEvent) { await device.queue.onSubmittedWorkDone(); return; }
    const start = hostNs();
    const work = device.queue.onSubmittedWorkDone();
    if (awaits.size >= 128 || nextTicket > 0x7fffffff) {
      activity(TraceActivity.Completion, 0, 0, 0, Number(start / 1000n), 0);
      await work; return;
    }
    const ticket = nextTicket++;
    const job: {start: bigint; end?: bigint; failed: boolean} = {start, failed: false};
    awaits.set(ticket, job);
    activity(TraceActivity.Completion, 0, 0, 0, Number(start / 1000n), 0, ticket);
    try { await work; } catch (error) { job.failed = true; throw error; }
    finally { job.end = hostNs(); wakeup(); }
  }
  // Racing every operation against device.lost retains a reaction on that
  // lifetime-long promise even after the operation completes. Register only
  // live operations and remove their loss notification when work settles.
  function untilDeviceLost(work: Promise<void>): Promise<void> {
    return new Promise((resolve, reject) => {
      const finish = () => { lossWaiters.delete(finish); resolve(); };
      if (lost) finish();
      else lossWaiters.add(finish);
      void work.then(finish, (error) => { lossWaiters.delete(finish); reject(error); });
    });
  }
  function complete(completion: Completion, work: readonly Promise<unknown>[]): Promise<void> {
    return untilDeviceLost(
      Promise.allSettled(work).then((outcomes) => {
        for (const outcome of outcomes) if (outcome.status === 'rejected') {
          completion.failed = true;
          diagnostic(`GPU bridge: completion failed. ${String(outcome.reason)}`);
        }
      }),
    ).then(() => { completion.failed ||= lost || closed; wakeup(); });
  }
  function retire(buffer: GPUBuffer, after?: Promise<void>): void {
    if (!after && pass) { uniforms.push(buffer); return; }
    const released = untilDeviceLost(after ?? queueDone())
      .catch(() => {}).then(() => {
        try { destroyBuffer(buffer); } finally { retiring.delete(released); }
      }).catch((error) => { diagnostic(`GPU bridge: retirement failed. ${String(error)}`); });
    retiring.set(released, buffer);
  }
  function destroyParamsPool(): void {
    for (const chunk of paramsPool) destroyBuffer(chunk.buffer);
    paramsPool.length = 0;
    pooledParamsBytes = 0;
    for (const chunk of activeParamsChunks) destroyBuffer(chunk.buffer);
    activeParamsChunks.length = 0;
    activeChunk = null;
  }
  function takeParamsChunk(requiredSize: number): ParamsChunk {
    for (let i = paramsPool.length - 1; i >= 0; i--) {
      const cached = paramsPool[i]!;
      if (cached.buffer.size >= requiredSize) {
        // Keep the pool's ownership until registration succeeds. Removing the
        // entry afterwards needs no allocation, including for a nonlast match.
        activeParamsChunks.push(cached);
        paramsPool[i] = paramsPool[paramsPool.length - 1]!;
        paramsPool.pop();
        pooledParamsBytes -= cached.buffer.size;
        cached.usedBytes = 0;
        return cached;
      }
    }
    const size = Math.max(PARAMS_CHUNK_BYTES, requiredSize);
    const staging = new Uint8Array(size);
    const buffer = createBuffer({label: 'vx.params', size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    try {
      const chunk: ParamsChunk = {
        buffer,
        staging,
        usedBytes: 0,
      };
      activeParamsChunks.push(chunk);
      return chunk;
    } catch (error) {
      destroyBuffer(buffer);
      throw error;
    }
  }
  function finishParams(owner: Completion): void {
    const chunks = activeParamsChunks.slice();
    activeParamsChunks.length = 0;
    activeChunk = null;
    if (!chunks.length) return;
    const buffers = chunks.map(c => c.buffer);
    // One completion owns the whole batch. Neither another pass nor another
    // execution context can overwrite these uniforms while GPU work uses them.
    const released = owner.done.catch(() => { owner.failed = true; }).then(() => {
      for (const chunk of chunks) {
        if (owner.failed || lost || closed || paramsPool.length >= MAX_POOLED_PARAMS ||
            pooledParamsBytes + chunk.buffer.size > MAX_POOLED_PARAMS_BYTES) {
          destroyBuffer(chunk.buffer);
        } else {
          chunk.usedBytes = 0;
          paramsPool.push(chunk);
          pooledParamsBytes += chunk.buffer.size;
        }
      }
    }).catch((error) => {
      diagnostic(`GPU bridge: parameter retirement failed. ${String(error)}`);
    }).finally(() => pendingParams.delete(released));
    pendingParams.set(released, buffers);
  }
  function pipelineFor(shaderId: number): PipelineRecord | null {
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
      const layout = device.createBindGroupLayout({ entries });
      const pipeline = device.createComputePipeline({
        label: SHADER_NAMES[shaderId],
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ code }), entryPoint: SHADER_ENTRY_POINTS[shaderId] },
      });
      const record: PipelineRecord = { pipeline, layout };
      pipelines.set(shaderId, record);
      return record;
    } catch (error) {
      diagnostic(`GPU bridge: pipeline failed. ${String(error)}`);
      return null;
    }
    // Asynchronous validation is part of submission.done. An invalid pipeline
    // cannot turn into a successful result even if creation returned an object.
  }

  function timestampTicket(batch: Timestamp, index: number): number {
    if (nextTicket > 0x7fffffff) return GPU_ERROR;
    const ticket = nextTicket++;
    timestamps.set(ticket, {batch, index, endIndex: index}); batch.references++;
    return ticket;
  }
  function destroyTimestamp(buffers: TimestampBuffers): void {
    buffers.queries.destroy(); destroyBuffer(buffers.resolve); destroyBuffer(buffers.staging);
    timestampResources--;
  }
  function prepareTracing(): Promise<void> {
    if (closed || lost || !device.features.has('timestamp-query')) return Promise.resolve();
    preparingTimestamps ??= (async () => {
      while (!closed && !lost && timestampResources < TIMESTAMP_BATCHES) {
        let queries: GPUQuerySet | undefined, resolve: GPUBuffer | undefined, staging: GPUBuffer | undefined;
        const completion: Completion = {failed: false, done: Promise.resolve()};
        // Pop both scopes before yielding so subsequent engine calls cannot
        // enter them. Invalid resources never reach a numerical command buffer.
        openScopes();
        try {
          queries = device.createQuerySet({label: 'vx.timestamp', type: 'timestamp', count: TIMESTAMP_INTERVALS * 2});
          resolve = createBuffer({label: 'vx.timestamp.resolve', size: TIMESTAMP_INTERVALS * 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC});
          staging = createBuffer({label: 'vx.timestamp.staging', size: TIMESTAMP_INTERVALS * 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        } catch (error) {
          completion.failed = true;
          diagnostic(`GPU bridge: timestamp allocation failed. ${String(error)}`);
        }
        await untilDeviceLost(closeScopes(completion));
        if (completion.failed || closed || lost || !queries || !resolve || !staging) {
          queries?.destroy(); if (resolve) destroyBuffer(resolve); if (staging) destroyBuffer(staging);
          break;
        }
        timestampResources++;
        timestampPool.push({queries, resolve, staging});
      }
    })().finally(() => { preparingTimestamps = undefined; });
    return preparingTimestamps;
  }
  function timestampPass(batch: Timestamp, index: number): GPUComputePassEncoder {
    return encoder!.beginComputePass({label: debugLabel, timestampWrites: {
      querySet: batch.queries, beginningOfPassWriteIndex: index * 2, endOfPassWriteIndex: index * 2 + 1,
    }});
  }
  // The ordinary dispatch guard stays unchanged. Node mode enters and leaves
  // passes explicitly from the C instrumented loop, sharing one query batch.
  function begin(capacity = 0, nodes = false, observe = false): number {
    activityEnabled = observe || (capacity > 0 && nodes);
    if (closed) { passFailed = true; return GPU_ERROR; }
    uniforms = [];
    encoding = {failed: false, done: Promise.resolve()};
    passFailed = lost || closed;
    timestamp = undefined;
    activeNode = undefined;
    pass = null;
    currentPipeline = null;
    currentBindGroup = null;
    let ticket = 0;
    try {
      openScopes();
      passScopes = true;
      encoder = device.createCommandEncoder({label: 'vx.graph'});
      if (Number.isInteger(capacity) && capacity > 0 && capacity <= TIMESTAMP_INTERVALS &&
          nextTicket <= 0x7fffffff) {
        const buffers = timestampPool.pop();
        if (buffers) {
          let finish!: () => void;
          const done = new Promise<void>(resolve => { finish = resolve; });
          timestamp = {...buffers, failed: false, done, finish,
            capacity, nodes, count: nodes ? 0 : 1, references: 0, retired: false};
          ticket = timestampTicket(timestamp, nodes ? -1 : 0);
        }
      }
      if (capacity && device.features.has('timestamp-query') && !timestamp && !ticket) ticket = GPU_ERROR;
      if (!timestamp) pass = encoder.beginComputePass();
      else if (!nodes) pass = timestampPass(timestamp, 0);
    } catch (error) {
      passFailed = true;
      diagnostic(`GPU bridge: pass creation failed. ${String(error)}`);
    }
    return ticket;
  }

  function beginInterval(): number {
    if (!timestamp?.nodes || !encoder) return 0;
    try {
      pass?.end(); pass = null;
      currentPipeline = null;
      currentBindGroup = null;
      if (timestamp.count === timestamp.capacity || nextTicket > 0x7fffffff) {
        if (activeNode) activeNode.incomplete = true;
        pass = encoder.beginComputePass({label: debugLabel}); return GPU_ERROR;
      }
      const index = timestamp.count++;
      pass = timestampPass(timestamp, index);
      return timestampTicket(timestamp, index);
    } catch (error) {
      passFailed = true;
      if (activeNode) activeNode.incomplete = true;
      diagnostic(`GPU bridge: timestamp pass failed. ${String(error)}`);
      return GPU_ERROR;
    }
  }
  function endInterval(): void {
    if (!timestamp?.nodes) return;
    try { pass?.end(); } catch { passFailed = true; }
    pass = null;
    currentPipeline = null;
    currentBindGroup = null;
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
    vx_gpu_memory_start(id: number, capacity: number): number {
      if (!memoryEvent || closed || lost || !id || capacity < 1 || memoryObservers.has(id) || memoryObservers.size >= 64) return 0;
      const observer: MemoryObserver = { capacity: Math.min(capacity, 16384), next: 0, buffers: new Map() };
      memoryObservers.set(id, observer);
      // The observer's bounded map deduplicates inventory; do not construct an
      // additional unbounded copy of every bridge resource just for tracing.
      const existing = (buffer: GPUBuffer) => memoryCreated(buffer, 0, id);
      for (const span of spans.values()) existing(span.buffer);
      for (const pool of inactiveSpans.values()) for (const span of pool.values()) existing(span.buffer);
      for (const job of readbacks.values()) if (job.staging) existing(job.staging);
      for (const buffer of uniforms) existing(buffer);
      for (const chunk of activeParamsChunks) existing(chunk.buffer);
      for (const chunk of paramsPool) existing(chunk.buffer);
      for (const buffers of pendingParams.values()) for (const buffer of buffers) existing(buffer);
      for (const resource of retiring.values()) {
        if (!resource) continue;
        if ('queries' in resource) { existing(resource.resolve); existing(resource.staging); }
        else existing(resource);
      }
      for (const interval of timestamps.values()) {
        existing(interval.batch.resolve); existing(interval.batch.staging);
      }
      for (const buffers of timestampPool) { existing(buffers.resolve); existing(buffers.staging); }
      return 1;
    },
    vx_gpu_memory_stop(id: number): void { memoryObservers.delete(id); },
    vx_gpu_ensure(hostPointer: number, bytes: number, residencyFlags: number): number {
      if (lost || closed || !hostPointer || bytes <= 0) return 0;
      if (residencyFlags !== 0 && residencyFlags !== GPU_RESIDENCY_WEIGHT &&
          residencyFlags !== GPU_RESIDENCY_OVERWRITE) return 0;
      const isWeight = residencyFlags === GPU_RESIDENCY_WEIGHT;
      const overwrite = residencyFlags === GPU_RESIDENCY_OVERWRITE;
      // Byte-packed outputs and immutable weights cannot discard host content.
      if (overwrite && bytes % 4 !== 0) return 0;
      let span = spans.get(hostPointer);
      if (overwrite && span?.isWeight) return 0;
      try {
        if (span && span.bytes !== bytes) {
          retainInactive(hostPointer, span);
          span = undefined;
        }
        let cached = !span ? inactiveSpans.get(hostPointer)?.get(bytes) : undefined;
        if (cached?.lastUse?.failed) {
          retireInactive(removeInactive(hostPointer, bytes)!);
          cached = undefined;
        }
        if (cached && (!cached.lastUse || cached.lastUse.settled)) {
          // Reclassify storage and discard old activation validity. Canonical
          // immutable bytes remain current unless host invalidation cleared them.
          span = {buffer: cached.buffer, bytes, deviceDirty: false, isWeight,
            uploaded: isWeight && cached.isWeight && cached.uploaded};
          spans.set(hostPointer, span);
          removeInactive(hostPointer, bytes);
        }
        // Checkout before trimming: exchanging two retained extents must not
        // evict the very buffer that is about to become active.
        trimInactive(hostPointer);
        if (!span) {
          const descriptor = {
            label: isWeight ? 'vx.weight' : 'vx.tensor',
            size: aligned(bytes), usage: GPUBufferUsage.STORAGE |
              GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          };
          let buffer: GPUBuffer;
          try { buffer = createBuffer(descriptor); }
          catch (error) {
            // Retire optional idle storage on synchronous allocation pressure.
            // Queue-owned buffers remain deferred; asynchronous validation OOM
            // still fails the request through its original error scopes.
            if (!inactiveSpanBytes) throw error;
            drainInactive();
            buffer = createBuffer(descriptor);
          }
          span = {
            buffer,
            bytes, deviceDirty: false, isWeight, uploaded: false,
          };
          spans.set(hostPointer, span);
        }
        // Reservation is not a write: only successful encode marks deviceDirty.
        // An intervening read must still upload the current host bytes.
        if (!overwrite && !span.deviceDirty && !(span.isWeight && span.uploaded)) {
          span.lastUse = encoding ?? undefined;
          const source = bytesAt(hostPointer, bytes);
          if (bytes % 4 === 0) upload(span.buffer, source);
          else {
            const padded = new Uint8Array(aligned(bytes));
            padded.set(source);
            upload(span.buffer, padded);
          }
          span.uploaded = true;
        }
        return 1;
      } catch (error) {
        passFailed = true;
        diagnostic(`GPU bridge: residency failed. ${String(error)}`);
        const active = spans.get(hostPointer);
        spans.delete(hostPointer);
        if (active) retire(active.buffer);
        // A failed checkout remains owned by the idle pool. A newly allocated
        // buffer whose registration failed has only this local owner.
        if (span && span.buffer !== active?.buffer &&
            inactiveSpans.get(hostPointer)?.get(span.bytes)?.buffer !== span.buffer) retire(span.buffer);
        if (inactiveSpans.get(hostPointer)?.size === 0) inactiveSpans.delete(hostPointer);
        trimInactive(hostPointer);
        return 0;
      }
    },
    vx_gpu_invalidate(hostPointer: number): void {
      const span = spans.get(hostPointer);
      if (span) { span.deviceDirty = false; span.uploaded = false; }
      const pool = inactiveSpans.get(hostPointer);
      if (pool) {
        for (const s of pool.values()) { s.deviceDirty = false; s.uploaded = false; }
      }
    },
    vx_gpu_release(hostPointer: number): void {
      const span = spans.get(hostPointer);
      if (span) retire(span.buffer);
      spans.delete(hostPointer);
      drainInactive(hostPointer);
    },
    vx_gpu_begin(): void { begin(); },
    vx_gpu_begin_activity(): void { begin(0, false, true); },
    vx_gpu_begin_trace(capacity: number, nodes: number): number { return begin(capacity, nodes !== 0); },
    vx_gpu_trace_node_begin(): number {
      const ticket = beginInterval();
      activeNode = ticket > 0 ? timestamps.get(ticket) : undefined;
      return ticket;
    },
    vx_gpu_trace_node_end(): void {
      endInterval();
      if (activeNode && timestamp) {
        activeNode.endIndex = timestamp.count - 1;
        activeNode = undefined;
      }
    },
    vx_gpu_trace_program_begin(): number { return beginInterval(); },
    vx_gpu_trace_program_end(): void { endInterval(); },
    vx_gpu_debug_label(text: number, bytes: number): number {
      if (lost || closed) return 0;
      debugLabel = bytes ? new TextDecoder().decode(bytesAt(text, bytes)) : undefined;
      return 1;
    },
    vx_gpu_trace_read(ticket: number, destination: number): number {
      const interval = timestamps.get(ticket);
      if (lost) return GPU_DEVICE_LOST;
      if (!interval || interval.batch.failed || closed || interval.index < 0 || interval.incomplete) return GPU_ERROR;
      const values = interval.batch.values;
      if (!values) return GPU_PENDING;
      const start = values[interval.index * 2], end = values[interval.endIndex * 2 + 1];
      if (start === undefined || end === undefined || end < start) return GPU_ERROR;
      for (let index = interval.index; index <= interval.endIndex; index++) {
        if (values[index * 2 + 1]! < values[index * 2]!) return GPU_ERROR;
      }
      const out = resultView(destination);
      out.setBigUint64(0, end - start, true);
      const first = values[0], last = values[interval.batch.count * 2 - 1];
      const {hostBefore: before, hostAfter: after} = interval.batch;
      if (before !== undefined && after !== undefined && first !== undefined && last !== undefined &&
          after >= before && start >= first && end <= last && last - first <= after - before) {
        out.setBigUint64(8, before + start - first, true);
        out.setBigUint64(16, after - (last - start), true);
        out.setUint32(24, 2, true); // BOUNDED: no host/device calibration in WebGPU.
      }
      return GPU_OK;
    },
    vx_gpu_trace_release(ticket: number): void {
      const interval = timestamps.get(ticket);
      if (!interval) return;
      timestamps.delete(ticket);
      const job = interval.batch;
      job.references--;
      const released = job.done.finally(() => {
        if (!job.references && !job.retired) {
          job.retired = true;
          if (closed || lost || job.failed) destroyTimestamp(job);
          else timestampPool.push({queries: job.queries, resolve: job.resolve, staging: job.staging});
        }
        retiring.delete(released);
      });
      retiring.set(released, job);
    },
    vx_gpu_await_read(ticket: number, destination: number): number {
      const job = awaits.get(ticket);
      if (!job || job.failed || closed || lost) return GPU_ERROR;
      if (job.end === undefined) return GPU_PENDING;
      if (job.end < job.start) return GPU_ERROR;
      const out = resultView(destination);
      out.setBigUint64(0, job.end - job.start, true);
      out.setBigUint64(32, job.start, true);
      return GPU_OK;
    },
    vx_gpu_await_release(ticket: number): void { awaits.delete(ticket); },
    vx_gpu_encode(dispatch: number): number {
      if (!pass || lost || closed) {
        // The ordinary dispatch guard is unchanged. Instrumented node passes
        // may leave auxiliary decode/row-transfer commands between nodes;
        // open an untimed pass only when those commands actually arrive.
        if (lost || closed || !encoder || !timestamp?.nodes) return GPU_ERROR;
        try {
          currentPipeline = null;
          currentBindGroup = null;
          pass = encoder.beginComputePass({label: debugLabel});
        }
        catch { passFailed = true; return GPU_ERROR; }
      }
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
          span.lastUse = encoding ?? undefined;
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
        if (paramsBytes > 0) {
          /* Rounded to 16: WGSL lays uniform structures out on that boundary,
           * and a device is free to refuse a binding that is smaller than the
           * structure the shader declares. The extra bytes are never read. */
          const align = device.limits?.minUniformBufferOffsetAlignment ?? 256;
          const roundedSize = (paramsBytes + 15) & ~15;
          if (!activeChunk) activeChunk = takeParamsChunk(PARAMS_CHUNK_BYTES);
          let offset = (activeChunk.usedBytes + align - 1) & ~(align - 1);
          if (offset + roundedSize > activeChunk.buffer.size) {
            activeChunk = takeParamsChunk(Math.max(PARAMS_CHUNK_BYTES, roundedSize));
            offset = 0;
          }
          activeChunk.staging.set(bytesAt(paramsBase << 2, paramsBytes), offset);
          if (paramsSlot === GPU_NO_UNIFORM) {
            diagnostic('GPU bridge: uniform bytes arrived with no slot to bind them.');
            passFailed = true;
            return GPU_ERROR;
          }
          /* The slot the shader declares, not the next free one: a uniform is
           * not always numbered after the storage buffers. */
          entries.push({
            binding: paramsSlot,
            resource: {
              buffer: activeChunk.buffer,
              offset,
              size: roundedSize,
            },
          });
          activeChunk.usedBytes = offset + Math.max(roundedSize, align);
        }

        for (let variant = 0; variant < variantCount; variant++) {
          const at = variantBase + variant * GPU_VARIANT_WORDS;
          const record = pipelineFor(view[at]!);
          if (!record) continue;
          try {
            // A group per dispatch stays balanced however passes are split.
            if (debugLabel) pass.pushDebugGroup(debugLabel);
            if (currentPipeline !== record.pipeline) {
              pass.setPipeline(record.pipeline);
              currentPipeline = record.pipeline;
            }
            let bindKey = `${view[at]!}`;
            for (let i = 0; i < entries.length; i++) {
              const res = entries[i]!.resource as GPUBufferBinding;
              bindKey += `_${entries[i]!.binding}:${bufferId(res.buffer)}:${res.offset ?? 0}:${res.size ?? 0}`;
            }
            let bg = bindGroupCache.get(bindKey)?.group;
            if (!bg) {
              while (bindGroupCache.size >= MAX_CACHED_BIND_GROUPS) {
                evictBindGroup(bindGroupCache.keys().next().value!);
              }
              bg = device.createBindGroup({
                layout: record.layout,
                entries,
              });
              const buffers = entries.map(entry => (entry.resource as GPUBufferBinding).buffer);
              bindGroupCache.set(bindKey, {group: bg, buffers});
              try {
                for (const buf of buffers) {
                  let set = bufferToKeys.get(buf);
                  if (!set) bufferToKeys.set(buf, set = new Set());
                  set.add(bindKey);
                }
              } catch (error) {
                evictBindGroup(bindKey);
                throw error;
              }
            }
            if (currentBindGroup !== bg) {
              pass.setBindGroup(0, bg);
              currentBindGroup = bg;
            }
            pass.dispatchWorkgroups(view[at + 1]!, view[at + 2]!, view[at + 3]!);
            if (debugLabel) pass.popDebugGroup();
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
      const current: Completion = encoding ?? {failed: false, done: Promise.resolve()};
      current.failed = passFailed || lost || closed || !encoder || (!pass && !timestamp?.nodes);
      try {
        pass?.end();
        if (!current.failed && encoder) {
          for (const chunk of activeParamsChunks) {
            if (chunk.usedBytes > 0) {
              upload(chunk.buffer, chunk.staging.subarray(0, chunk.usedBytes));
            }
          }
          if (timestamp?.count) {
            encoder.resolveQuerySet(timestamp.queries, 0, timestamp.count * 2, timestamp.resolve, 0);
            encoder.copyBufferToBuffer(timestamp.resolve, 0, timestamp.staging, 0, timestamp.count * 16);
          }
          const commands = encoder.finish();
          if (timestamp) timestamp.hostBefore = hostNs();
          submit([commands]);
        }
      } catch (error) {
        current.failed = true;
        diagnostic(`GPU bridge: submission failed. ${String(error)}`);
      }
      encoder = null;
      pass = null;
      currentPipeline = null;
      currentBindGroup = null;
      current.done = complete(current, [
        ...(passScopes ? [closeScopes(current)] : []), queueDone(),
      ]).then(() => { current.settled = true; });
      encoding = null;
      passScopes = false;
      if (timestamp) {
        const job = timestamp;
        timestamp = undefined;
        job.failed = current.failed;
        void current.done.then(() => { job.hostAfter = hostNs(); });
        let mapped: Promise<void>;
        try { mapped = current.failed ? Promise.resolve() : job.staging.mapAsync(GPUMapMode.READ); }
        catch { job.failed = true; mapped = Promise.resolve(); }
        void complete(job, [current.done, mapped]).then(() => {
          job.failed ||= current.failed;
          if (!job.failed) try {
            const values = new BigUint64Array(job.staging.getMappedRange());
            // Copy before unmapping; only polling writes the current WASM memory.
            // Validate each interval separately so one invalid pair cannot hide
            // valid neighbours. Equal, quantized timestamps are a valid zero.
            job.values = values.slice(0, job.count * 2);
            job.staging.unmap();
          } catch { job.failed = true; }
          wakeup();
        }).finally(job.finish);
      }
      for (const uniform of uniforms) retire(uniform, current.done);
      uniforms = [];
      finishParams(current);
      submission = current;
      activityEnabled = false;
      return current.failed ? GPU_ERROR : GPU_OK;
    },
    vx_gpu_snapshot(hostPointer: number, offset: number, bytes: number, observe: number): number {
      activityEnabled = !!observe;
      try {
        if (lost || closed || nextTicket > 0x7fffffff) return GPU_ERROR;
        // The explicit null/zero command fences preceding work even when the
        // caller selects no outputs. It never creates/maps a staging buffer.
        if (hostPointer === 0 && offset === 0 && bytes === 0) {
          const owner = submission;
          const job: Readback = {bytes: 0, offset: 0, failed: false, ready: false,
            done: Promise.resolve()};
          job.done = complete(job, [owner.done, queueDone()]).then(() => {
            job.failed ||= owner.failed;
            job.ready = !job.failed;
            wakeup();
          });
          const ticket = nextTicket++;
          readbacks.set(ticket, job);
          return ticket;
        }
        const span = spans.get(hostPointer);
        if (!span || bytes <= 0 || offset + bytes > span.bytes)
          return GPU_ERROR;
        // WebGPU copies aligned words. The ticket retains any leading bytes so
        // Readback can expose an unaligned int8 tensor slice without changing it.
        const start = Math.floor(offset / 4) * 4;
        const length = aligned(offset + bytes) - start;
        openScopes();
        let staging: GPUBuffer;
        try {
          staging = createBuffer({ label: 'vx.readback', size: length,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        } catch (error) {
          const failed: Completion = { failed: true, done: Promise.resolve() };
          const done = complete(failed, [closeScopes(failed), queueDone()]);
          retiring.set(done, undefined);
          void done.then(() => retiring.delete(done));
          diagnostic(`GPU bridge: staging allocation failed. ${String(error)}`);
          return GPU_ERROR;
        }
        const job: Readback = {
          staging,
          bytes, offset: offset - start, failed: false, ready: false, done: Promise.resolve(),
        };
        span.lastUse = job;
        const owner = submission;
        let mapped: Promise<void>;
        try {
          const copy = device.createCommandEncoder({label: 'vx.readback'});
          const copyStart = activityEnabled ? performance.now() * 1000 : 0;
          copy.copyBufferToBuffer(span.buffer, start, staging, 0, length);
          if (activityEnabled) activity(TraceActivity.Copy, MemorySpace.Device,
            MemorySpace.HostVisibleDevice, length, copyStart, performance.now() * 1000);
          submit([copy.finish()]);
          mapped = staging.mapAsync(GPUMapMode.READ);
        } catch (error) {
          job.failed = true;
          mapped = Promise.resolve();
          diagnostic(`GPU bridge: snapshot failed. ${String(error)}`);
        }
        job.done = complete(job, [mapped, owner.done, closeScopes(job), queueDone()]).then(() => {
          job.failed ||= owner.failed;
          job.ready = !job.failed;
          job.settled = true;
          wakeup();
        });
        const ticket = nextTicket++;
        readbacks.set(ticket, job);
        return ticket;
      } finally { activityEnabled = false; }
    },
    vx_gpu_readback(ticket: number, destination: number, bytes: number): number {
      const job = readbacks.get(ticket);
      if (lost) return GPU_DEVICE_LOST;
      if (closed || !job || job.failed || job.bytes !== bytes) return GPU_ERROR;
      if (!job.ready) return GPU_PENDING;
      if (!job.staging) return bytes === 0 && destination === 0 ? GPU_OK : GPU_ERROR;
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
      if (job.staging) retire(job.staging, job.done);
      else {
        // A retired fence still drains on owner close even after a later
        // submission replaces the bridge's current completion record.
        retiring.set(job.done, undefined);
        void job.done.then(() => retiring.delete(job.done));
      }
    },
  });
  return Object.freeze({
    imports,
    prepareTracing,
    setWakeup(callback: () => void): void { wakeup = callback; },
    attach(attached: WebAssembly.Memory, callback, activityCallback): void {
      memoryEvent = callback; activityEvent = activityCallback;
      if (memory || closed) throw new Error('GPU bridge already belongs to a WASM instance.');
      memory = attached;
    },
    close(): void {
      if (closed) return;
      closed = true;
      if (passScopes) imports.vx_gpu_end();
      for (const span of spans.values()) retire(span.buffer);
      spans.clear();
      drainInactive();
      for (const ticket of readbacks.keys()) imports.vx_gpu_readback_release(ticket);
      for (const ticket of timestamps.keys()) imports.vx_gpu_trace_release(ticket);
      for (const buffers of timestampPool) destroyTimestamp(buffers);
      timestampPool.length = 0;
      destroyParamsPool();
      pipelines.clear();
      bindGroupCache.clear();
      bufferToKeys.clear();
      currentPipeline = null;
      currentBindGroup = null;
      memoryObservers.clear();
      memoryEvent = undefined;
      memory = null;
      wakeup = () => {};
    },
    async waitForCompletion(): Promise<void> {
      await Promise.all([preparingTimestamps, submission.done, ...retiring.keys(), ...pendingParams.keys(), ...[...timestamps.values()].map((interval) => interval.batch.done), ...[...readbacks.values()].map((job) => job.done)]);
    },
  });
}
