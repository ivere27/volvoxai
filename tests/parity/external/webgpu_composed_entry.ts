/** Full browser transport: every operation reaches the same C owner. */
export { FullEngineHost } from '../../../ts/full.js';
export * as pb from '../../../runtime/generated/typescript/volvoxai_lite.js';
export { VxInferenceServiceClient, VxSchedulerServiceClient, VxTrainingServiceClient,
  VxQuantizationServiceClient, VxPlanningServiceClient, VxTextServiceClient }
  from '../../../runtime/generated/typescript/volvoxai_ffi.js';
import { createWebGPUHostBridge as createBridge, type WebGPUHostBridgeOptions } from '../../../ts/backends/WebGPUHostBridge.js';
import * as catalog from '../../../ts/generated/shaderCatalog.js';
export const createWebGPUHostBridge = (options: Omit<WebGPUHostBridgeOptions, 'catalog'>) => createBridge({ ...options, catalog });
