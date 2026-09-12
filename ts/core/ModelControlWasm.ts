import { PROTO_PACKAGE, ProtoService } from '../generated/protoMethods.js';
import type { WasmGpuBridgeHost } from './WasmReleaseModule.js';
import {
  instantiateWasmReleaseModule,
  loadWasmReleaseModule,
} from './WasmReleaseModule.js';
import {
  MODEL_CONTROL_WASM_REQUIRED_EXPORTS,
  WASM_PROFILE_IMPORTS,
  type WasmInferenceParentExports,
  type WasmFullParentExports,
} from '../generated/wasmInternalAbi.js';
import { RpcError, type ByteCall, type CallOptions, type Host, type Method, type Transport } from
  '../../runtime/generated/typescript/inference/synurang_runtime.js';
import { createWasmHost, type WasmModule } from
  '../../runtime/generated/typescript/inference/synurang_wasm.js';
const PAGE_BYTES = 65_536;
const WASM32_LINEAR_MEMORY_BYTES = 0x1_0000_0000;
const CONTROL_ALIGNMENT = 16;
const I32_MINIMUM = -0x80000000;
const I32_MAXIMUM = 0x7fffffff;
/* alloc_bytes accepts a signed i32 and rounds it up by 15 in C. */
const MAXIMUM_DISPATCH_MAILBOX_BYTES = 0x7ffffff0;
const MAXIMUM_PROTO_BYTES = 72 * 1024 * 1024;
const MINIMUM_DISPATCH_MAILBOX_BYTES = 256;
const VFS_UPLOAD_CHUNK_BYTES = 1024 * 1024;
const MAXIMUM_VFS_PATH_BYTES = 255;
const UTF8_ENCODER = new TextEncoder();



type ModelControlServiceName = `${ProtoService}`;

type ModelControlExportName = typeof MODEL_CONTROL_WASM_REQUIRED_EXPORTS[number]['name'];
type ModelControlExports = Omit<Pick<WasmInferenceParentExports, ModelControlExportName>, keyof WasmModule> &
  Omit<WasmModule, 'registerWakeup' | 'unregisterWakeup'> &
  Partial<Pick<WasmFullParentExports, 'vx_wasm_prepare_gpu_v1'>>;

const arrayBufferByteLengthGetter = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength',
)?.get;

function isOrdinaryArrayBuffer(value: unknown): value is ArrayBuffer {
  if (value === null || typeof value !== 'object' ||
      arrayBufferByteLengthGetter === undefined) return false;
  try {
    Reflect.apply(arrayBufferByteLengthGetter, value, []);
    return true;
  } catch {
    return false;
  }
}

function fail(message: string, cause?: unknown): never {
  throw new Error(
    `[ModelControlWasm] ${message}`,
    cause === undefined ? undefined : { cause },
  );
}

function validateReleaseModule(module: WebAssembly.Module, profile: 'inference' | 'full'): void {
  const imports = WebAssembly.Module.imports(module)
    .map(({ module: namespace, name, kind }) => `${namespace}.${name}:${kind}`)
    .sort();
  // The full host supplies the device bridge; inference admits only CPU WASM.
  const expected = WASM_PROFILE_IMPORTS[profile];
  const matches = imports.length === expected.length &&
    imports.every((value, index) => value === expected[index]);
  if (!matches) {
    fail(
      `Release parent imports do not match the ${profile} profile: ` +
      (imports.length === 0 ? '(none)' : imports.join(', ')),
    );
  }
  const exports = new Map(
    WebAssembly.Module.exports(module).map(({ name, kind }) => [name, kind]),
  );
  for (const { name, kind } of MODEL_CONTROL_WASM_REQUIRED_EXPORTS) {
    if (exports.get(name) !== kind) {
      fail(`Release parent export '${name}' must have kind '${kind}'.`);
    }
  }
}

function checkedReleaseModule(module: WebAssembly.Module, profile: 'inference' | 'full'): WebAssembly.Module {
  if (!(module instanceof WebAssembly.Module)) {
    fail('Factory requires a compiled release WebAssembly.Module.');
  }
  validateReleaseModule(module, profile);
  return module;
}

/** Internal proof that every owner factory passed the same module ABI checks. */
abstract class ValidatedModelControlFactory {
  readonly module: WebAssembly.Module;
  /*
   * Full owns a device bridge; inference has no GPU imports or bridge.
   *
   * One bridge belongs to one instance. It attaches to the linear memory it is
   * given, and it remembers exactly one, so a factory that makes several
   * instances must be given a bridge only when it makes exactly one -- a
   * second instantiation moves the bridge to the newer memory and leaves the
   * first instance dispatching against someone else's heap.
   */
  readonly gpuBridge: WasmGpuBridgeHost | undefined;

  protected constructor(module: WebAssembly.Module, gpuBridge?: WasmGpuBridgeHost) {
    this.module = checkedReleaseModule(module, gpuBridge ? 'full' : 'inference');
    this.gpuBridge = gpuBridge;
  }

  create(): ModelControlWasm {
    return new ModelControlWasm(this);
  }

  /** @internal Instantiate the exact validated release parent. */
  instantiate(wakeup?: (token: number) => void): WebAssembly.Instance {
    return instantiateWasmReleaseModule(this.module, this.gpuBridge, wakeup);
  }
}

/**
 * Immutable factory for callers that only forward generated protobuf RPCs.
 * It deliberately carries no TypeScript Planning codec or operator registry.
 */
export class ModelControlWasmDispatchFactory extends ValidatedModelControlFactory {
  constructor(module: WebAssembly.Module, gpuBridge?: WasmGpuBridgeHost) {
    super(module, gpuBridge);
    Object.freeze(this);
  }
}

/** Load/compile a release parent without retaining a TS Planning registry. */
export async function loadModelControlWasmDispatchFactory(
  wasmUrl: string | URL,
  gpuBridge?: WasmGpuBridgeHost,
): Promise<ModelControlWasmDispatchFactory> {
  return new ModelControlWasmDispatchFactory(
    await loadWasmReleaseModule(wasmUrl),
    gpuBridge,
  );
}

function checkedMailboxAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(left) || left < 0 || !Number.isSafeInteger(right) ||
      right < 0 || !Number.isSafeInteger(result) ||
      result > MAXIMUM_DISPATCH_MAILBOX_BYTES) {
    fail(`${label} is not representable by the signed wasm32 allocator ABI.`);
  }
  return result;
}

function alignDispatch16(value: number, label: string): number {
  const remainder = value % CONTROL_ALIGNMENT;
  return remainder === 0
    ? value
    : checkedMailboxAdd(value, CONTROL_ALIGNMENT - remainder, label);
}

function checkedWasmAddress(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > I32_MAXIMUM) {
    fail(`${label} is not a positive signed wasm32 address.`);
  }
  return value;
}

function checkedWasmRange(
  offset: number,
  bytes: number,
  memory: WebAssembly.Memory,
  label: string,
): number {
  const end = offset + bytes;
  if (!Number.isSafeInteger(offset) || offset < 0 ||
      !Number.isSafeInteger(bytes) || bytes < 0 ||
      !Number.isSafeInteger(end) || end > WASM32_LINEAR_MEMORY_BYTES ||
      end > memory.buffer.byteLength) {
    fail(`${label} exceeds the current wasm32 linear-memory extent.`);
  }
  return end;
}

function checkedStatus(value: number, label: string): number {
  if (!Number.isInteger(value) || value < I32_MINIMUM || value > I32_MAXIMUM) {
    fail(`${label} returned a non-i32 status.`);
  }
  return value;
}

function checkedProtoData(value: unknown): Uint8Array {
  if (!(value instanceof Uint8Array)) {
    fail('Proto dispatch data must be a Uint8Array.');
  }
  if (value.byteLength > MAXIMUM_PROTO_BYTES) {
    fail('Proto dispatch data exceeds the 72-MiB private transport limit.');
  }
  if (!isOrdinaryArrayBuffer(value.buffer)) {
    fail('Proto dispatch data must use a non-shared ArrayBuffer.');
  }
  return value;
}

function checkedService(
  serviceName: string,
  methodName: string,
): readonly [ModelControlServiceName, Uint8Array] {
  if (!Object.values(ProtoService).includes(serviceName as ProtoService)) {
    fail(`Unknown generated service '${serviceName}'.`);
  }
  const service = serviceName as ModelControlServiceName;
  const expectedPrefix = `/${PROTO_PACKAGE}.${service}/`;
  if (typeof methodName !== 'string' || !methodName.startsWith(expectedPrefix) ||
      methodName.length === expectedPrefix.length || methodName.includes('\0')) {
    fail(`Method '${methodName}' does not belong to generated service '${service}'.`);
  }
  const method = UTF8_ENCODER.encode(methodName);
  if (method.byteLength >= I32_MAXIMUM) {
    fail('Generated method name is not representable by the signed wasm32 byte-length ABI.');
  }
  return Object.freeze([service, method]);
}

function checkedExports(instance: WebAssembly.Instance): ModelControlExports {
  const values = instance.exports as Readonly<Record<string, WebAssembly.ExportValue>>;
  const memory = values.memory;
  if (!(memory instanceof WebAssembly.Memory)) fail("Export 'memory' is not WebAssembly memory.");
  for (const { name, kind } of MODEL_CONTROL_WASM_REQUIRED_EXPORTS) {
    if (kind === 'function' && typeof values[name] !== 'function') {
      fail(`Export '${name}' is not a function.`);
    }
  }
  return values as unknown as ModelControlExports;
}

/**
 * Explicit owner for one model-definition, short-lived loader transaction, or
 * one Runtime's persistent lifecycle tree.
 *
 * The factory shares one already-compiled release parent. Every owner still
 * has a distinct persistent Instance and Memory; C registries and handles stay
 * private to that owner until close drops the complete instance graph.
 */
export class ModelControlWasm implements Transport {
  #exports: ModelControlExports | null;
  #memory: WebAssembly.Memory | null;
  #host: Promise<Host> | null;
  #instanceId = 0;
  #closePromise: Promise<void> | null = null;
  #dispatchMailboxOffset = 0;
  #dispatchMailboxBytes = 0;
  #busy = false;
  #closed = false;
  #gpuBridge: WasmGpuBridgeHost | undefined;

  constructor(
    factory: ModelControlWasmDispatchFactory,
  ) {
    if (!(factory instanceof ValidatedModelControlFactory)) {
      fail('Control owner requires its validated release-module factory.');
    }
    this.#gpuBridge = factory.gpuBridge;
    const wakeups = new Map<number, () => void>();
    let nextToken = 0;
    let instance: WebAssembly.Instance;
    try {
      instance = factory.instantiate(token => wakeups.get(token)?.());
    } catch (error) {
      fail('Release parent could not be instantiated with its declared host imports.', error);
    }
    const exports = checkedExports(instance);
    const initialMemoryBytes = exports.memory.buffer.byteLength;
    if (!Number.isSafeInteger(initialMemoryBytes) || initialMemoryBytes <= 0 ||
        initialMemoryBytes % PAGE_BYTES !== 0 ||
        initialMemoryBytes > WASM32_LINEAR_MEMORY_BYTES) {
      fail('Release parent memory is incompatible.');
    }
    const heapBase = Number(exports.__heap_base.value);
    const heapMark = Number(exports.heap_mark());
    const alignedHeapBase = Math.ceil(heapBase / CONTROL_ALIGNMENT) * CONTROL_ALIGNMENT;
    if (!Number.isSafeInteger(heapBase) || heapBase <= 0 ||
        heapBase > I32_MAXIMUM || heapBase > initialMemoryBytes ||
        !Number.isSafeInteger(heapMark) || heapMark !== heapBase ||
        heapMark % CONTROL_ALIGNMENT !== 0 || alignedHeapBase !== heapBase) {
      fail('Release parent did not start with a fresh aligned C heap.');
    }
    this.#exports = exports;
    this.#memory = exports.memory;
    if (exports.synurang_module_abi_version() !== 1) {
      fail('Release parent uses an unsupported Synurang module ABI.');
    }
    const module = {
      ...exports,
      synurang_module_create: (capacity: number, token: number) => {
        this.#instanceId = exports.synurang_module_create(capacity, token);
        return this.#instanceId;
      },
      registerWakeup: (callback: () => void) => {
        const token = ++nextToken;
        wakeups.set(token, callback);
        return token;
      },
      unregisterWakeup: (token: number) => { wakeups.delete(token); },
    } as unknown as WasmModule;
    this.#host = createWasmHost(() => module);
    this.#gpuBridge?.setWakeup?.(() => {
      if (this.#exports !== null) void this.#host?.then(host => host.wake());
    });
    void this.#host.catch(() => {});
    Object.freeze(this);
  }

  get closed(): boolean {
    return this.#closed;
  }

  /** Cancel open calls and drain the C instance before releasing its memory. */
  close(): Promise<void> {
    if (this.#closePromise !== null) return this.#closePromise;
    this.#closed = true;
    return this.#closePromise = (async () => {
      let host: Host | undefined;
      try { host = await this.#host ?? undefined; } catch { /* Failed initialization owns no C instance. */ }
      await host?.close();
      if (this.#dispatchMailboxOffset !== 0) {
        this.#exports!.synurang_module_free(this.#dispatchMailboxOffset);
      }
      this.#gpuBridge?.close?.();
      await this.#gpuBridge?.waitForCompletion();
      this.#gpuBridge = undefined;
      this.#host = null;
      this.#exports = null;
      this.#memory = null;
      this.#instanceId = 0;
      this.#dispatchMailboxOffset = 0;
      this.#dispatchMailboxBytes = 0;
    })();
  }

  async open(method: Method, options?: CallOptions): Promise<ByteCall> {
    if (this.#closed) fail('Call requires an open owner.');
    return (await this.#host!).open(method, options);
  }

  /** C decides transport preparation without executing the public request. */
  requiresGpuPreparation(serviceName: string, methodName: string, data: Uint8Array): boolean {
    const response = this.#invokeProto(serviceName, methodName, data, 'vx_wasm_prepare_gpu_v1');
    if (response.byteLength !== 1 || response[0]! > 1) {
      fail('C returned invalid GPU preparation requirements.');
    }
    return response[0] === 1;
  }

  /**
   * Ask C whether a path-backed inference request can reach its first file
   * read. The returned bytes use the operation's declared response type: an
   * OK report is fetch-ready, while any failure is returned to the generated
   * caller unchanged. The threadless preflight mutates no lifecycle state.
   */
  preflightInferencePathRequest(
    methodName: string,
    data: Uint8Array,
  ): Uint8Array {
    return this.#invokeProto(
      'VxInferenceService',
      methodName,
      data,
      'vx_wasm_inference_path_preflight_v1',
    );
  }

  #invokeProto(
    serviceName: string,
    methodName: string,
    data: Uint8Array,
    dispatchExport: 'vx_wasm_inference_path_preflight_v1' | 'vx_wasm_prepare_gpu_v1',
  ): Uint8Array {
    if (this.#closed || this.#exports === null || this.#memory === null) {
      fail('Proto dispatch requires an open explicit owner.');
    }
    if (this.#busy) fail('Proto dispatch cannot reenter one explicit owner.');
    const [service, method] = checkedService(serviceName, methodName);
    const request = checkedProtoData(data);
    const methodBytes = checkedMailboxAdd(method.byteLength, 1, 'Proto method range');
    const requestOffsetInMailbox = alignDispatch16(
      methodBytes,
      'Proto request alignment',
    );
    const responseLengthOffsetInMailbox = alignDispatch16(
      checkedMailboxAdd(
        requestOffsetInMailbox,
        Math.max(1, request.byteLength),
        'Proto request range',
      ),
      'Proto response-length alignment',
    );
    const usedBytes = checkedMailboxAdd(
      responseLengthOffsetInMailbox,
      4,
      'Proto dispatch mailbox range',
    );

    const exports = this.#exports;
    const memory = this.#memory;
    this.#busy = true;
    let responsePointer = 0;
    let responseBytes = 0;
    try {
      this.#ensureDispatchMailbox(exports, memory, usedBytes);
      const mailboxOffset = checkedWasmAddress(
        this.#dispatchMailboxOffset,
        'Proto dispatch mailbox address',
      );
      checkedWasmRange(mailboxOffset, usedBytes, memory, 'Proto dispatch mailbox');
      const methodOffset = mailboxOffset;
      const requestOffset = checkedWasmAddress(
        mailboxOffset + requestOffsetInMailbox,
        'Proto request address',
      );
      const responseLengthOffset = checkedWasmAddress(
        mailboxOffset + responseLengthOffsetInMailbox,
        'Proto response-length address',
      );
      const mailbox = new Uint8Array(memory.buffer, mailboxOffset, usedBytes);
      mailbox.fill(0);
      new Uint8Array(memory.buffer, methodOffset, method.byteLength).set(method);
      if (request.byteLength !== 0) {
        new Uint8Array(memory.buffer, requestOffset, request.byteLength).set(request);
      }
      new DataView(memory.buffer).setInt32(responseLengthOffset, 0, true);

      const dispatch = exports[dispatchExport];
      if (typeof dispatch !== 'function') {
        fail(`Service '${service}' is unavailable in this WASM profile.`);
      }
      responsePointer = checkedStatus(
        dispatch(this.#instanceId, methodOffset, requestOffset, request.byteLength, responseLengthOffset),
        `${service} proto dispatch pointer`,
      );
      const signedResponseBytes = new DataView(memory.buffer)
        .getInt32(responseLengthOffset, true);
      if (signedResponseBytes === I32_MINIMUM) {
        fail(`${service} proto dispatch returned an invalid response length.`);
      }
      if (responsePointer === 0 && signedResponseBytes < 0) {
        throw new RpcError(-signedResponseBytes, 'C transport preflight rejected the request');
      }
      const transportFailed = signedResponseBytes < 0;
      responseBytes = Math.abs(signedResponseBytes);
      if (responsePointer <= 0 || responsePointer >= memory.buffer.byteLength ||
          !Number.isSafeInteger(responseBytes)) {
        fail(`${service} proto dispatch returned an invalid response range.`);
      }
      checkedWasmRange(
        responsePointer,
        responseBytes,
        memory,
        `${service} proto dispatch response`,
      );
      const response = new Uint8Array(
        memory.buffer,
        responsePointer,
        responseBytes,
      ).slice();
      if (transportFailed) {
        fail(
          `${service} rejected '${methodName}' with a ${responseBytes}-byte ` +
          'Synurang transport error.',
        );
      }
      return response;
    } finally {
      try {
        if (responsePointer > 0 && responsePointer < memory.buffer.byteLength) {
          if (responseBytes > 0 && Number.isSafeInteger(responseBytes) &&
              responsePointer + responseBytes <= memory.buffer.byteLength) {
            new Uint8Array(memory.buffer, responsePointer, responseBytes).fill(0);
          }
          exports.synurang_module_free(responsePointer);
        }
        if (this.#dispatchMailboxOffset !== 0) {
          const clearBytes = Math.min(
            usedBytes,
            memory.buffer.byteLength - this.#dispatchMailboxOffset,
          );
          if (clearBytes > 0) {
            new Uint8Array(
              memory.buffer,
              this.#dispatchMailboxOffset,
              clearBytes,
            ).fill(0);
          }
        }
      } finally {
        this.#busy = false;
      }
    }
  }

  /** Copy one browser-owned package file into this C runtime's private VFS. */
  mountFile(path: string, data: Uint8Array): void {
    this.mountFileChunks(path, [data]);
  }

  /**
   * Copy an already bounded response body without first joining its transport
   * chunks into another model-sized JavaScript allocation.
   *
   * @internal This is package transport, not an application-facing operation.
   */
  mountFileChunks(path: string, chunks: readonly Uint8Array[]): void {
    if (this.#closed || this.#exports === null || this.#memory === null) {
      fail('VFS mount requires an open explicit owner.');
    }
    if (this.#busy) fail('VFS mount cannot reenter one explicit owner.');
    const encodedPath = this.#checkedVfsPath(path);
    if (!Array.isArray(chunks)) fail('VFS mount chunks must be an array.');
    const sources = chunks.map((chunk) => checkedProtoData(chunk));
    let sourceBytes = 0;
    let largestChunkBytes = 0;
    for (const source of sources) {
      sourceBytes = checkedMailboxAdd(
        sourceBytes,
        source.byteLength,
        'VFS source byte count',
      );
      largestChunkBytes = Math.max(largestChunkBytes, source.byteLength);
    }
    const pathBytes = checkedMailboxAdd(encodedPath.byteLength, 1, 'VFS path range');
    const chunkBytes = Math.max(
      1,
      Math.min(VFS_UPLOAD_CHUNK_BYTES, largestChunkBytes),
    );
    const usedBytes = Math.max(pathBytes, chunkBytes);
    const exports = this.#exports;
    const memory = this.#memory;
    this.#busy = true;
    let uploadNeedsAbort = false;
    try {
      this.#ensureDispatchMailbox(exports, memory, usedBytes);
      const mailboxOffset = checkedWasmAddress(
        this.#dispatchMailboxOffset,
        'VFS upload mailbox address',
      );
      checkedWasmRange(mailboxOffset, usedBytes, memory, 'VFS upload mailbox');
      const mailbox = new Uint8Array(memory.buffer, mailboxOffset, usedBytes);
      mailbox.fill(0);
      mailbox.set(encodedPath);
      uploadNeedsAbort = true;
      const beginStatus = checkedStatus(
        exports.vx_wasm_mount_file_begin(mailboxOffset, sourceBytes),
        'VFS mount begin',
      );
      if (beginStatus !== 0) fail(`C VFS rejected mount path '${path}'.`);

      for (const source of sources) {
        for (let sourceOffset = 0; sourceOffset < source.byteLength;) {
          const length = Math.min(
            VFS_UPLOAD_CHUNK_BYTES,
            source.byteLength - sourceOffset,
          );
          new Uint8Array(memory.buffer, mailboxOffset, length).set(
            source.subarray(sourceOffset, sourceOffset + length),
          );
          const writeStatus = checkedStatus(
            exports.vx_wasm_mount_file_write(mailboxOffset, length),
            'VFS mount chunk write',
          );
          if (writeStatus !== 0) {
            fail(`C VFS rejected a mount chunk for path '${path}'.`);
          }
          sourceOffset += length;
        }
      }

      const finishStatus = checkedStatus(
        exports.vx_wasm_mount_file_finish(),
        'VFS mount finish',
      );
      if (finishStatus !== 0) fail(`C VFS could not publish mount path '${path}'.`);
      uploadNeedsAbort = false;
    } finally {
      try {
        if (uploadNeedsAbort) {
          const abortStatus = checkedStatus(
            exports.vx_wasm_mount_file_abort(),
            'VFS mount abort',
          );
          if (abortStatus !== 0) fail(`C VFS could not abort mount path '${path}'.`);
        }
        if (this.#dispatchMailboxOffset !== 0) {
          const clearBytes = Math.min(
            usedBytes,
            memory.buffer.byteLength - this.#dispatchMailboxOffset,
          );
          if (clearBytes > 0) {
            new Uint8Array(
              memory.buffer,
              this.#dispatchMailboxOffset,
              clearBytes,
            ).fill(0);
          }
        }
      } finally {
        this.#busy = false;
      }
    }
  }

  /** Remove one copied package file from this C runtime's private VFS. */
  unmountFile(path: string): void {
    if (this.#closed || this.#exports === null || this.#memory === null) {
      fail('VFS unmount requires an open explicit owner.');
    }
    if (this.#busy) fail('VFS unmount cannot reenter one explicit owner.');
    const encodedPath = this.#checkedVfsPath(path);
    const usedBytes = checkedMailboxAdd(encodedPath.byteLength, 1, 'VFS path range');
    const exports = this.#exports;
    const memory = this.#memory;
    this.#busy = true;
    try {
      this.#ensureDispatchMailbox(exports, memory, usedBytes);
      const pathOffset = checkedWasmAddress(
        this.#dispatchMailboxOffset,
        'VFS unmount path address',
      );
      checkedWasmRange(pathOffset, usedBytes, memory, 'VFS unmount path');
      const mailbox = new Uint8Array(memory.buffer, pathOffset, usedBytes);
      mailbox.fill(0);
      mailbox.set(encodedPath);
      const status = checkedStatus(
        exports.vx_wasm_unmount_file(pathOffset),
        'VFS unmount',
      );
      if (status !== 0) fail(`C VFS does not own mount path '${path}'.`);
    } finally {
      try {
        if (this.#dispatchMailboxOffset !== 0) {
          const clearBytes = Math.min(
            usedBytes,
            memory.buffer.byteLength - this.#dispatchMailboxOffset,
          );
          if (clearBytes > 0) {
            new Uint8Array(
              memory.buffer,
              this.#dispatchMailboxOffset,
              clearBytes,
            ).fill(0);
          }
        }
      } finally {
        this.#busy = false;
      }
    }
  }

  #checkedVfsPath(path: string): Uint8Array {
    if (typeof path !== 'string' || path.length === 0 || path.includes('\0')) {
      fail('VFS path must be a non-empty string without NUL bytes.');
    }
    const encoded = UTF8_ENCODER.encode(path);
    if (encoded.byteLength === 0 || encoded.byteLength > MAXIMUM_VFS_PATH_BYTES) {
      fail(`VFS path must encode to at most ${MAXIMUM_VFS_PATH_BYTES} UTF-8 bytes.`);
    }
    return encoded;
  }

  #ensureDispatchMailbox(
    exports: ModelControlExports,
    memory: WebAssembly.Memory,
    usedBytes: number,
  ): void {
    if (usedBytes <= this.#dispatchMailboxBytes) return;
    let capacity = Math.max(
      MINIMUM_DISPATCH_MAILBOX_BYTES,
      this.#dispatchMailboxBytes,
    );
    while (capacity < usedBytes) {
      if (capacity > Math.floor(MAXIMUM_DISPATCH_MAILBOX_BYTES / 2)) {
        capacity = usedBytes;
        break;
      }
      capacity *= 2;
    }
    capacity = alignDispatch16(capacity, 'Proto dispatch mailbox capacity');
    if (capacity > MAXIMUM_DISPATCH_MAILBOX_BYTES) {
      fail('Proto dispatch mailbox exceeds the signed wasm32 allocator ABI.');
    }
    const offset = exports.synurang_module_alloc(capacity);
    checkedWasmAddress(offset, 'Transport mailbox allocation');
    checkedWasmRange(offset, capacity, memory, 'Transport mailbox allocation');
    if (this.#dispatchMailboxOffset !== 0) {
      exports.synurang_module_free(this.#dispatchMailboxOffset);
    }
    this.#dispatchMailboxOffset = offset;
    this.#dispatchMailboxBytes = capacity;
  }

}
