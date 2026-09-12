export { FullEngineHost } from '../../../ts/full.js';
import { createWebGPUHostBridge as createBridge, type WebGPUHostBridgeOptions } from '../../../ts/backends/WebGPUHostBridge.js';
import * as catalog from '../../../ts/generated/shaderCatalog.js';
export const createWebGPUHostBridge = (options: Omit<WebGPUHostBridgeOptions, 'catalog'>) => createBridge({ ...options, catalog });
export * as pb from '../../../runtime/generated/typescript/volvoxai_lite.js';
export { VxInferenceServiceClient }
  from '../../../runtime/generated/typescript/volvoxai_ffi.js';
