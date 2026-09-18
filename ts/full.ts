import * as fullProto from '../runtime/generated/typescript/volvoxai_lite.js';
import { PROTO_METHOD_RESPONSES } from './generated/protoMethodsFull.js';
/** Full web profile: the proto API with inference, training, and PTQ. */

import { InferenceWasmHost, type EngineHostOptions, type ProtoTransport } from './host/InferenceWasmHost.js';
import { createDeferredWebGPUHostBridge } from './backends/WebGPUHostBridge.js';
import type { WasmGpuBridgeHost } from './core/WasmReleaseModule.js';
import * as catalog from './generated/shaderCatalog.js';

export interface FullEngineHostOptions extends EngineHostOptions {
  /** Device transport diagnostics; engine reports are returned by proto RPCs. */
  readonly onDiagnostic?: (message: string) => void;
  /** Optional device bridge owned by this host; otherwise acquired lazily at C's request. */
  readonly gpuBridge?: WasmGpuBridgeHost;
}

/** Promise-based service clients generated from the public proto. */
export {
  VxPlatformServiceClient,
  VxInferenceServiceClient,
  VxBufferServiceClient,
  VxSchedulerServiceClient,
  VxPlanningServiceClient,
  VxTextServiceClient,
  VxTrainingServiceClient,
  VxQuantizationServiceClient,
} from '../runtime/generated/typescript/volvoxai_ffi.js';
export { RpcError } from '../runtime/generated/typescript/inference/synurang_runtime.js';
export type { CallOptions, Transport, ByteCall, Method } from
  '../runtime/generated/typescript/inference/synurang_runtime.js';

export * as pb from '../runtime/generated/typescript/volvoxai_lite.js';
export { VolvoxAIError } from './core/RuntimeErrors.js';
export type { OperationReportLike } from './host/OperationReports.js';
export type {
  VolvoxAIErrorCode,
  VolvoxAIErrorOptions,
} from './core/RuntimeErrors.js';
/** Inference, scheduler, training and PTQ share one persistent C owner. */
export class FullEngineHost extends InferenceWasmHost {
  constructor(options: FullEngineHostOptions = {}) {
    super({
      ...options,
      wasmUrl: options.wasmUrl ?? new URL('./volvoxai.wasm', import.meta.url),
    // These transport messages come from the same proto. The full projection
    // adds enum members, which TypeScript treats as a different enum type.
    }, fullProto as unknown as ProtoTransport, PROTO_METHOD_RESPONSES,
    options.gpuBridge ?? createDeferredWebGPUHostBridge(catalog, options.onDiagnostic));
  }
}

export { FullEngineHost as EngineHost };
