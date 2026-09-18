/**
 * Thin asynchronous transport for the inference profile's generated C/WASM
 * protobuf dispatcher.
 *
 * The host owns one persistent module instance, so native handles published by
 * one RPC remain valid for every later RPC. Browser package sources are the
 * only operation that cannot be forwarded byte-for-byte: they are fetched,
 * copied into that instance's private VFS, and rewritten to private paths until
 * the call has finished.
 */
import { RpcError, type ByteCall, type CallOptions, type Method, type Transport } from
  '../../runtime/generated/typescript/inference/synurang_runtime.js';
import type * as pb from '../../runtime/generated/typescript/inference/volvoxai_lite.js';
import type { WasmGpuBridgeHost } from '../core/WasmReleaseModule.js';
import { assertOperationResponse } from './OperationReports.js';
import { CallScope } from './CallScope.js';
import {
  loadModelControlWasmDispatchFactory,
  type ModelControlWasm,
} from '../core/ModelControlWasm.js';

export type ProtoTransport = Pick<typeof pb,
  'LoadModelRequest' | 'ModelHandle' | 'PublishAdapterRequest' | 'AdapterRevision' |
  'OperationReport' | 'OperationCode' | 'OperationStage' | 'NativeStatus'> &
  Readonly<Record<string, unknown>>;

interface EngineHostFetchResponse {
  readonly ok: boolean;
  readonly status?: number;
  readonly statusText?: string;
  readonly headers?: { get(name: string): string | null };
  readonly body?: ReadableStream<Uint8Array> | null;
  readonly arrayBuffer?: () => Promise<ArrayBuffer>;
  readonly text?: () => Promise<string>;
}

export type EngineHostFetch = (
  source: string,
  options?: { readonly signal?: AbortSignal },
) => Promise<EngineHostFetchResponse>;

export interface ResolvedModelSource {
  readonly graphUrl: string;
  readonly weightSources: readonly string[];
}

/** Transport configuration shared by inference and full hosts. */
export interface EngineHostOptions {
  /** Profile-matched release module. Defaults beside the JavaScript bundle. */
  readonly wasmUrl?: string | URL;
  /** Fetch implementation used only for path-named package inputs. */
  readonly fetch?: EngineHostFetch;
  /** Maps public model paths to fetchable package sources. */
  readonly resolveModelSource?: (
    graphPath: string,
    weightPaths: readonly string[],
  ) => ResolvedModelSource;
  /**
   * Maximum source bytes in one model or adapter package, including inline
   * ModelPackage bytes. May tighten, but
   * cannot raise, the ordinary browser profile's fixed 64-MiB ceiling.
   */
  readonly maxPackageBytes?: number;
}

class SourceTransferError extends Error {
  readonly status: pb.NativeStatus;
  readonly code: pb.OperationCode;

  constructor(status: pb.NativeStatus, code: pb.OperationCode, message: string) {
    super(message);
    this.name = 'SourceTransferError';
    this.status = status;
    this.code = code;
  }
}

interface PackageSource {
  readonly path: string;
  readonly source: string;
  readonly label: string;
  readonly allowText: boolean;
}

interface PreparedRequest {
  readonly data?: Uint8Array;
  readonly response?: Uint8Array;
}

interface Preparation {
  readonly scope: CallScope;
  own(dispose: () => void): void;
}

interface PackageBody {
  readonly chunks: readonly Uint8Array[];
  readonly byteLength: number;
}

const DEFAULT_MAX_PACKAGE_BYTES = 64 * 1024 * 1024;
const STREAM_BLOCK_BYTES = 1024 * 1024;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function failureReport(
  codec: ProtoTransport,
  error: SourceTransferError,
  stage: pb.OperationStage,
  lineage?: pb.Lineage,
): pb.OperationReport {
  return new codec.OperationReport({
    status: error.status,
    stage,
    code: error.code,
    message: error.message,
    lineage,
  });
}

/**
 * Generated service clients share one C/WASM module instance. Synurang owns
 * call progress; this transport only prepares browser resources and reports.
 */
export class InferenceWasmHost implements Transport {
  readonly #codec: ProtoTransport;
  readonly #responses: Readonly<Record<string, string>>;
  readonly #methods: { LoadModel: string; PublishAdapter: string };
  readonly #wasmUrl: string | URL;
  readonly #fetch: EngineHostFetch | undefined;
  readonly #resolveModelSource: EngineHostOptions['resolveModelSource'];
  readonly #maxPackageBytes: number;
  #gpuBridge: WasmGpuBridgeHost | undefined;
  #ownerPromise: Promise<ModelControlWasm> | null = null;
  #ownerValue: ModelControlWasm | null = null;
  readonly #shutdown = new AbortController();
  #closePromise: Promise<void> | null = null;
  #closed = false;
  #nextTransferId = 1n;

  constructor(
    options: EngineHostOptions,
    codec: ProtoTransport,
    responses: Readonly<Record<string, string>>,
    gpuBridge?: WasmGpuBridgeHost,
  ) {
    this.#codec = codec;
    this.#responses = responses;
    const paths = Object.keys(responses);
    this.#methods = {
      LoadModel: paths.find(path => path.endsWith('.VxInferenceService/LoadModel'))!,
      PublishAdapter: paths.find(path => path.endsWith('.VxInferenceService/PublishAdapter'))!,
    };
    this.#wasmUrl = options.wasmUrl ?? new URL('./volvoxai.lite.wasm', import.meta.url);
    this.#fetch = options.fetch;
    this.#resolveModelSource = options.resolveModelSource;
    this.#maxPackageBytes = options.maxPackageBytes ?? DEFAULT_MAX_PACKAGE_BYTES;
    this.#gpuBridge = gpuBridge;
    if (!Number.isSafeInteger(this.#maxPackageBytes) ||
        this.#maxPackageBytes <= 0 ||
        this.#maxPackageBytes > DEFAULT_MAX_PACKAGE_BYTES) {
      throw new TypeError(
        'EngineHost maxPackageBytes must be a positive integer no larger than 64 MiB.',
      );
    }
  }

  /** Snapshot the generated request before initialization or browser I/O. */
  openWithRequest(method: Method, encode: () => Uint8Array, options?: CallOptions): Promise<ByteCall> {
    let scope: CallScope | undefined;
    try {
      if (this.#closed) throw new RpcError(14, 'EngineHost is closed.');
      scope = new CallScope(options, this.#shutdown.signal);
      scope.signal.throwIfAborted();
      const data = encode();
      return this.#openWithRequest(method, data, scope);
    } catch (error) { scope?.close(); return Promise.reject(error); }
  }

  async #openWithRequest(method: Method, data: Uint8Array, scope: CallScope): Promise<ByteCall> {
    const call = await this.#open(method, scope);
    try {
      await call.send(data);
      await call.halfClose();
      return call;
    } catch (error) {
      await call.close();
      throw error;
    }
  }

  open(method: Method, options?: CallOptions): Promise<ByteCall> {
    try { return this.#open(method, new CallScope(options, this.#shutdown.signal)); }
    catch (error) { return Promise.reject(error); }
  }

  async #open(method: Method, scope: CallScope): Promise<ByteCall> {
    let call: ByteCall;
    let owner: ModelControlWasm;
    try {
      if (this.#closed) throw new RpcError(14, 'EngineHost is closed.');
      if (method.requestStream || method.responseStream) {
        throw new RpcError(12, 'The VolvoxAI schema declares unary calls only.');
      }
      owner = await scope.wait(this.#owner());
      call = await owner.open(method, scope.remainingOptions);
    } catch (error) { scope.close(); throw error; }
    let dispose = () => {};
    const preparation: Preparation = {
      scope,
      own(cleanup) {
        dispose = cleanup;
        if (scope.signal.aborted) cleanup();
      },
    };
    const cleanup = () => dispose();
    scope.signal.addEventListener('abort', cleanup, { once: true });
    const firstRead = call.recv();
    void firstRead.catch(error => {
      scope.abort(error instanceof RpcError ? error : new RpcError(13, errorMessage(error)));
      cleanup();
    });
    let firstReadPending = true;
    let prepared: PreparedRequest | undefined;
    let response: Uint8Array | undefined;
    let responseReturned = false;
    let released = false;
    return {
      send: async (data) => {
        if (released || this.#closed) throw new RpcError(1, 'Call cancelled');
        if (prepared) throw new RpcError(3, 'Unary call accepts one request');
        prepared = await scope.wait(Promise.race([
          this.#prepare(owner, method, data.slice(), preparation),
          firstRead.then(() => { throw new RpcError(13, 'Provider finished before accepting its request'); }),
        ]));
        if (prepared.response === undefined) await scope.wait(call.send(prepared.data!));
      },
      halfClose: () => prepared?.response === undefined ? scope.wait(call.halfClose()) : Promise.resolve(),
      recv: async () => {
        scope.signal.throwIfAborted();
        const data = prepared?.response === undefined
          ? await scope.wait(firstReadPending ? (firstReadPending = false, firstRead) : call.recv())
          : responseReturned ? null : prepared.response;
        if (data !== null) {
          responseReturned = true;
          response = data;
        } else if (response !== undefined) {
          // A response becomes successful only after the provider's terminal status.
          assertOperationResponse(this.#codec, this.#responses, method.path, response);
          response = undefined;
        }
        return data;
      },
      cancel: (code = 1) => { released = true; scope.cancel(code); call.cancel(code); },
      close: async () => {
        released = true;
        scope.cancel();
        try { await call.close(); } finally {
          dispose();
          scope.signal.removeEventListener('abort', cleanup);
          scope.close();
        }
      },
    };
  }

  /** Cancel calls, drain C-owned resources, then release the device and memory. */
  close(): Promise<void> {
    if (this.#closePromise !== null) return this.#closePromise;
    this.#closed = true;
    this.#shutdown.abort(new RpcError(1, 'EngineHost is closed.'));
    const owner = this.#ownerValue;
    this.#ownerValue = null;
    this.#ownerPromise = null;
    return this.#closePromise = (async () => {
      await owner?.close();
      const bridge = this.#gpuBridge;
      this.#gpuBridge = undefined;
      bridge?.close?.();
      await bridge?.waitForCompletion();
    })();
  }

  #owner(): Promise<ModelControlWasm> {
    if (this.#ownerPromise !== null) return this.#ownerPromise;
    const pending = loadModelControlWasmDispatchFactory(
      this.#wasmUrl, this.#gpuBridge,
    ).then(factory => {
      if (this.#closed) throw new RpcError(14, 'EngineHost is closed.');
      return this.#ownerValue = factory.create();
    });
    const owned = pending.catch((error) => {
      if (this.#ownerPromise === owned) this.#ownerPromise = null;
      throw error;
    });
    this.#ownerPromise = owned;
    return owned;
  }

  async #prepare(owner: ModelControlWasm, method: Method, data: Uint8Array, preparation: Preparation): Promise<PreparedRequest> {
    const serviceName = method.path.slice(method.path.lastIndexOf('.') + 1, method.path.lastIndexOf('/'));
    const bridge = this.#gpuBridge;
    if (bridge?.prepare && owner.requiresGpuPreparation(serviceName, method.path, data)) {
      await preparation.scope.wait(bridge.prepare());
    }
    if (method.path === this.#methods.LoadModel) return this.#loadModel(owner, data, preparation);
    if (method.path === this.#methods.PublishAdapter) return this.#publishAdapter(owner, data, preparation);
    return { data };
  }

  async #loadModel(
    owner: ModelControlWasm,
    data: Uint8Array,
    preparation: Preparation,
  ): Promise<PreparedRequest> {
    let request: pb.LoadModelRequest;
    try { request = this.#codec.LoadModelRequest.fromBinary(data); } catch {
      // The generated C call publishes the authoritative malformed-request status.
      return { data };
    }
    const preflightBytes = owner.preflightInferencePathRequest(
      this.#methods.LoadModel,
      data,
    );
    const preflight = this.#codec.ModelHandle.fromBinary(preflightBytes);
    if (preflight.report === undefined) {
      throw new Error('C LoadModel path preflight returned no OperationReport.');
    }
    if (preflight.report.status !== this.#codec.NativeStatus.NATIVE_STATUS_OK) {
      return { response: preflightBytes };
    }
    if (preflight.modelId !== 0n) {
      throw new Error('C LoadModel path preflight published a model handle.');
    }

    // C's generated decoder and native validators are authoritative. Decode
    // in TypeScript only after they approve the request, solely to stage and
    // rewrite browser-owned package paths.
    try {
      if (request.package !== undefined) {
        const packageBytes = request.package.weightShards.reduce(
          (total, shard) => total + shard.byteLength,
          request.package.graphDocument!.byteLength,
        );
        if (packageBytes > this.#maxPackageBytes) {
          throw new SourceTransferError(
            this.#codec.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY,
            this.#codec.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE,
            `ModelPackage exceeds the ${this.#maxPackageBytes}-byte package transport budget.`,
          );
        }
        return { data };
      }
      const source = this.#resolveModel(request);
      const transfer = this.#allocateTransfer('model');
      const graphPath = `${transfer}/graph.json`;
      const weightPaths = source.weightSources.map(
        (_unused, index) => `${transfer}/weight-${index}.safetensors`,
      );
      const sources: PackageSource[] = [{
        path: graphPath,
        source: source.graphUrl,
        label: 'model graph',
        allowText: true,
      }];
      for (let index = 0; index < source.weightSources.length; index++) {
        sources.push({
          path: weightPaths[index],
          source: source.weightSources[index],
          label: `model weight ${index}`,
          allowText: false,
        });
      }
      const forwarded = new this.#codec.LoadModelRequest({
        runtimeId: request.runtimeId,
        graphPath,
        weightPaths,
        bankResidency: request.bankResidency,
      });
      return await this.#invokeFetched(
        owner,
        sources,
        forwarded.toBinary(),
        preparation,
      );
    } catch (error) {
      if (!(error instanceof SourceTransferError)) throw error;
      return { response: new this.#codec.ModelHandle({
        report: failureReport(
          this.#codec,
          error,
          this.#codec.OperationStage.OPERATION_STAGE_MODEL_LOAD,
          preflight.report.lineage,
        ),
      }).toBinary() };
    }
  }

  async #publishAdapter(
    owner: ModelControlWasm,
    data: Uint8Array,
    preparation: Preparation,
  ): Promise<PreparedRequest> {
    let request: pb.PublishAdapterRequest;
    try { request = this.#codec.PublishAdapterRequest.fromBinary(data); } catch {
      // The generated C call publishes the authoritative malformed-request status.
      return { data };
    }
    const preflightBytes = owner.preflightInferencePathRequest(
      this.#methods.PublishAdapter,
      data,
    );
    const preflight = this.#codec.AdapterRevision.fromBinary(preflightBytes);
    if (preflight.report === undefined) {
      throw new Error('C PublishAdapter path preflight returned no OperationReport.');
    }
    if (preflight.report.status !== this.#codec.NativeStatus.NATIVE_STATUS_OK) {
      return { response: preflightBytes };
    }
    if (preflight.adapterId !== 0n || preflight.adapterRevision !== 0n) {
      throw new Error('C PublishAdapter path preflight published an adapter revision.');
    }

    // package_path is optional. C still owns decoding and validation; this
    // decode happens only after its fetch-ready decision.
    if (request.packagePath.length === 0) {
      return { data };
    }

    try {
      const transfer = this.#allocateTransfer('adapter');
      const packagePath = `${transfer}/package.bin`;
      const forwarded = new this.#codec.PublishAdapterRequest({
        modelId: request.modelId,
        adapterName: request.adapterName,
        packagePath,
        versionName: request.versionName,
      });
      return await this.#invokeFetched(
        owner,
        [{
          path: packagePath,
          source: request.packagePath,
          label: 'adapter package',
          allowText: false,
        }],
        forwarded.toBinary(),
        preparation,
      );
    } catch (error) {
      if (!(error instanceof SourceTransferError)) throw error;
      return { response: new this.#codec.AdapterRevision({
        report: failureReport(
          this.#codec,
          error,
          this.#codec.OperationStage.OPERATION_STAGE_ADAPTER,
          preflight.report.lineage,
        ),
      }).toBinary() };
    }
  }

  #resolveModel(request: pb.LoadModelRequest): ResolvedModelSource {
    let resolved: ResolvedModelSource;
    try {
      resolved = this.#resolveModelSource?.(
        request.graphPath,
        [...request.weightPaths],
      ) ?? {
        graphUrl: request.graphPath,
        weightSources: [...request.weightPaths],
      };
    } catch (error) {
      throw new SourceTransferError(
        this.#codec.NativeStatus.NATIVE_STATUS_IO_ERROR,
        this.#codec.OperationCode.OPERATION_CODE_MODEL_SOURCE_RESOLUTION_FAILED,
        `Model source resolution failed: ${errorMessage(error)}`,
      );
    }
    if (resolved === null || typeof resolved !== 'object' ||
        typeof resolved.graphUrl !== 'string' || resolved.graphUrl.length === 0 ||
        resolved.graphUrl.includes('\0') ||
        !Array.isArray(resolved.weightSources) ||
        resolved.weightSources.length !== request.weightPaths.length ||
        resolved.weightSources.some((source) =>
          typeof source !== 'string' || source.length === 0 || source.includes('\0'))) {
      throw new SourceTransferError(
        this.#codec.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
        this.#codec.OperationCode.OPERATION_CODE_INVALID_MODEL_SOURCE_RESOLUTION,
        'resolveModelSource must return one non-empty source for the graph and each weight path.',
      );
    }
    return {
      graphUrl: resolved.graphUrl,
      weightSources: [...resolved.weightSources],
    };
  }

  async #readSource(
    source: string,
    label: string,
    allowText: boolean,
    maximumBytes: number,
    scope: CallScope,
  ): Promise<PackageBody> {
    const transport = this.#fetch ?? this.#defaultFetch();
    let response: EngineHostFetchResponse;
    try {
      scope.signal.throwIfAborted();
      response = await scope.wait(transport(source, { signal: scope.signal }).then(value => {
        if (scope.signal.aborted) {
          void value.body?.cancel(scope.signal.reason).catch(() => {});
          scope.signal.throwIfAborted();
        }
        return value;
      }));
    } catch (error) {
      if (scope.signal.aborted) throw scope.signal.reason;
      throw new SourceTransferError(
        this.#codec.NativeStatus.NATIVE_STATUS_IO_ERROR,
        this.#codec.OperationCode.OPERATION_CODE_PACKAGE_FETCH_FAILED,
        `Could not fetch ${label} '${source}': ${errorMessage(error)}`,
      );
    }
    if (response === null || typeof response !== 'object' || response.ok !== true) {
      const detail = typeof response?.status === 'number'
        ? `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ''}`
        : response?.statusText || 'transport returned no successful response';
      throw new SourceTransferError(
        this.#codec.NativeStatus.NATIVE_STATUS_IO_ERROR,
        this.#codec.OperationCode.OPERATION_CODE_PACKAGE_FETCH_FAILED,
        `Could not fetch ${label} '${source}': ${detail}.`,
      );
    }
    const declaredLength = response.headers?.get('content-length');
    if (declaredLength !== undefined && declaredLength !== null &&
        /^\d+$/.test(declaredLength) && Number(declaredLength) > maximumBytes) {
      try {
        await response.body?.cancel();
      } catch {
        // The typed size failure remains authoritative if cancellation fails.
      }
      throw new SourceTransferError(
        this.#codec.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY,
        this.#codec.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE,
        `${label} exceeds the ${this.#maxPackageBytes}-byte package transport budget.`,
      );
    }
    try {
      if (response.body !== undefined && response.body !== null) {
        const reader = response.body.getReader();
        const cancel = () => { void reader.cancel(scope.signal.reason).catch(() => {}); };
        scope.signal.addEventListener('abort', cancel, { once: true });
        const chunks: Uint8Array[] = [];
        let byteLength = 0;
        let block: Uint8Array | undefined;
        let blockBytes = 0;
        let finished = false;
        try {
          while (true) {
            const item = await scope.wait(reader.read());
            if (item.done) {
              finished = true;
              break;
            }
            if (!(item.value instanceof Uint8Array)) {
              throw new TypeError('response body yielded a non-Uint8Array chunk');
            }
            if (item.value.byteLength > maximumBytes - byteLength) {
              try {
                await reader.cancel();
              } catch {
                // Preserve PACKAGE_TOO_LARGE if transport cancellation fails.
              }
              finished = true;
              throw new SourceTransferError(
                this.#codec.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY,
                this.#codec.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE,
                `${label} exceeds the ${this.#maxPackageBytes}-byte package transport budget.`,
              );
            }
            /* Never retain a transport-owned view. Besides making the bytes
             * stable, fixed blocks bound metadata and backing-store retention
             * when a custom stream yields tiny slices or views over huge
             * buffers. Each source view is discarded before the next read. */
            for (let sourceOffset = 0; sourceOffset < item.value.byteLength;) {
              if (block === undefined || blockBytes === block.byteLength) {
                block = new Uint8Array(Math.min(
                  STREAM_BLOCK_BYTES,
                  maximumBytes - byteLength,
                ));
                blockBytes = 0;
                chunks.push(block);
              }
              const copied = Math.min(
                block.byteLength - blockBytes,
                item.value.byteLength - sourceOffset,
              );
              block.set(
                item.value.subarray(sourceOffset, sourceOffset + copied),
                blockBytes,
              );
              blockBytes += copied;
              byteLength += copied;
              sourceOffset += copied;
            }
          }
        } finally {
          if (!finished) {
            try {
              await reader.cancel();
            } catch {
              // The original read or size failure remains authoritative.
            }
          }
          scope.signal.removeEventListener('abort', cancel);
          reader.releaseLock();
        }
        if (block !== undefined && blockBytes !== block.byteLength) {
          chunks[chunks.length - 1] = block.subarray(0, blockBytes);
        }
        return { chunks, byteLength };
      }
      if (typeof response.arrayBuffer === 'function') {
        const buffer = await scope.wait(response.arrayBuffer());
        if (!(buffer instanceof ArrayBuffer)) {
          throw new TypeError('arrayBuffer() did not return an ArrayBuffer');
        }
        if (buffer.byteLength > maximumBytes) {
          throw new SourceTransferError(
            this.#codec.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY,
            this.#codec.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE,
            `${label} exceeds the ${this.#maxPackageBytes}-byte package transport budget.`,
          );
        }
        /* Fetch already hands ownership of an ordinary ArrayBuffer to this
         * response. mountFile() copies it synchronously into WASM before the
         * next await, so retaining this view avoids a second 30--40 MiB JS
         * allocation without allowing the VFS to alias browser memory. */
        return {
          chunks: [new Uint8Array(buffer)],
          byteLength: buffer.byteLength,
        };
      }
      if (allowText && typeof response.text === 'function') {
        const bytes = new TextEncoder().encode(await scope.wait(response.text()));
        if (bytes.byteLength > maximumBytes) {
          throw new SourceTransferError(
            this.#codec.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY,
            this.#codec.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE,
            `${label} exceeds the ${this.#maxPackageBytes}-byte package transport budget.`,
          );
        }
        return { chunks: [bytes], byteLength: bytes.byteLength };
      }
    } catch (error) {
      if (scope.signal.aborted) throw scope.signal.reason;
      if (error instanceof SourceTransferError) throw error;
      throw new SourceTransferError(
        this.#codec.NativeStatus.NATIVE_STATUS_IO_ERROR,
        this.#codec.OperationCode.OPERATION_CODE_PACKAGE_READ_FAILED,
        `Could not read ${label} '${source}': ${errorMessage(error)}`,
      );
    }
    throw new SourceTransferError(
      this.#codec.NativeStatus.NATIVE_STATUS_IO_ERROR,
      this.#codec.OperationCode.OPERATION_CODE_PACKAGE_BODY_UNAVAILABLE,
      `${label} '${source}' has no readable ${allowText ? 'binary or text' : 'binary'} body.`,
    );
  }

  #defaultFetch(): EngineHostFetch {
    if (typeof globalThis.fetch !== 'function') {
      throw new SourceTransferError(
        this.#codec.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED,
        this.#codec.OperationCode.OPERATION_CODE_FETCH_UNAVAILABLE,
        'Path-named package inputs require an EngineHost fetch transport.',
      );
    }
    return (source, options) => globalThis.fetch(source, options);
  }

  #allocateTransfer(kind: 'model' | 'adapter'): string {
    const id = this.#nextTransferId;
    this.#nextTransferId++;
    return `/__volvoxai_host/${kind}-${id}`;
  }

  async #invokeFetched(
    owner: ModelControlWasm,
    sources: readonly PackageSource[],
    data: Uint8Array,
    preparation: Preparation,
  ): Promise<PreparedRequest> {
    const mounted: string[] = [];
    let remainingBytes = this.#maxPackageBytes;
    const dispose = () => {
      for (const path of mounted.splice(0).reverse()) {
        try { owner.unmountFile(path); } catch { /* Owner close retires its VFS. */ }
      }
    };
    preparation.own(dispose);
    try {
      for (const source of sources) {
        preparation.scope.signal.throwIfAborted();
        const body = await this.#readSource(
          source.source,
          source.label,
          source.allowText,
          remainingBytes,
          preparation.scope,
        );
        remainingBytes -= body.byteLength;
        preparation.scope.signal.throwIfAborted();
        try {
          owner.mountFileChunks(source.path, body.chunks);
        } catch (error) {
          throw new SourceTransferError(
            this.#codec.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY,
            this.#codec.OperationCode.OPERATION_CODE_PRIVATE_VFS_MOUNT_FAILED,
            `Could not stage a package source in the private WASM VFS: ${errorMessage(error)}`,
          );
        }
        mounted.push(source.path);
      }
      return { data };
    } catch (error) {
      dispose();
      throw error;
    }
  }
}
