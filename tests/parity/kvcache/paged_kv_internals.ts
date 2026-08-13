/* Bundle entry for the paged-KV GPU harness.
 *
 * Deliberately not part of the published surface. `ts/index.ts` exports the
 * public runtime; a page table is a backend-internal object, and this campaign
 * needs the engines and the allocator directly to drive a mapping the public
 * API has no way to ask for.
 *
 * It exists as a bundle rather than a direct TypeScript import because
 * `ShaderLibrary` imports `.wgsl` files, which only esbuild's text loader
 * resolves. Bundling here means the harness executes exactly the code the
 * shipped bundle executes.
 */
export { RuntimeGraph } from '../../../ts/core/RuntimeGraph.js';
export { CPUEngine } from '../../../ts/backends/CPUEngine.js';
export { WebGPUEngine } from '../../../ts/backends/WebGPUEngine.js';
export { PagedKVCache } from '../../../ts/core/PagedKVCache.js';
export { ContinuousBatchScheduler } from '../../../ts/core/ContinuousBatchScheduler.js';
export {
  classifyPlan,
  kvPageCopyRanges,
  kvPagePlanForLane,
} from '../../../ts/backends/kvPageAddressing.js';
