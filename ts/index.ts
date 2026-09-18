/** Inference profile: generated proto services over the C/WASM owner. */
export { EngineHost } from './host/EngineHost.js';
export type { EngineHostOptions } from './host/EngineHost.js';

/** Promise-based service clients generated from the public proto. */
export {
  VxPlatformServiceClient,
  VxInferenceServiceClient,
  VxBufferServiceClient,
  VxSchedulerServiceClient,
  VxTextServiceClient,
} from '../runtime/generated/typescript/inference/volvoxai_ffi.js';
export { RpcError } from '../runtime/generated/typescript/inference/synurang_runtime.js';
export type { CallOptions, Transport, ByteCall, Method } from
  '../runtime/generated/typescript/inference/synurang_runtime.js';

/** Every inference-profile request, response, enum, and evidence message. */
export * as pb from '../runtime/generated/typescript/inference/volvoxai_lite.js';

export { VolvoxAIError } from './core/RuntimeErrors.js';
export type { OperationReportLike } from './host/OperationReports.js';
export type {
  VolvoxAIErrorCode,
  VolvoxAIErrorOptions,
} from './core/RuntimeErrors.js';
