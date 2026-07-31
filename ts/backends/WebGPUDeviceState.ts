/**
 * Model-independent state shared by WebGPU engines created for one device.
 *
 * Pipelines are safe to share because their layouts depend only on shader
 * source, entry point, and constants. Bind groups and tensor buffers remain
 * context-owned in GraphExecutor.
 */
export class WebGPUDeviceState {
  readonly device: GPUDevice;
  readonly computePipelineCache: Map<string, Map<string, Promise<GPUComputePipeline>>>;
  readonly rejectedSpecializedShaders: Set<string>;
  #references = 0;

  constructor(device: GPUDevice) {
    if (!device) throw new Error('WebGPUDeviceState requires a GPUDevice.');
    this.device = device;
    this.computePipelineCache = new Map();
    this.rejectedSpecializedShaders = new Set();
  }

  retain(): this {
    this.#references++;
    return this;
  }

  release(): void {
    if (this.#references <= 0) {
      throw new Error('WebGPUDeviceState release has no matching retain.');
    }
    this.#references--;
    if (this.#references === 0) this.clear();
  }

  get referenceCount(): number {
    return this.#references;
  }

  clear(): void {
    this.computePipelineCache.clear();
    this.rejectedSpecializedShaders.clear();
  }
}
