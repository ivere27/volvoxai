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

  /** Device-wide interning for model-independent compute pipelines. */
  async computePipeline(
    code: string,
    {
      entryPoint = 'main',
      constants = null,
    }: {
      entryPoint?: string;
      constants?: Record<string, number | boolean> | null;
    } = {},
  ): Promise<GPUComputePipeline> {
    const constantEntries = constants
      ? Object.entries(constants).sort(([left], [right]) => left.localeCompare(right))
      : [];
    const constantKeyEntries = constantEntries.map(([name, value]) => {
      let encoded: string;
      if (typeof value === 'boolean') encoded = value ? 'boolean:true' : 'boolean:false';
      else if (typeof value !== 'number') {
        throw new TypeError(`WebGPU pipeline constant '${name}' must be numeric or boolean.`);
      } else if (Number.isNaN(value)) encoded = 'number:NaN';
      else if (Object.is(value, -0)) encoded = 'number:-0';
      else encoded = `number:${String(value)}`;
      return [name, encoded] as const;
    });
    const variantKey = JSON.stringify([entryPoint, constantKeyEntries]);
    let variants = this.computePipelineCache.get(code);
    if (!variants) {
      variants = new Map();
      this.computePipelineCache.set(code, variants);
    }
    let pending = variants.get(variantKey);
    if (!pending) {
      const module = this.device.createShaderModule({ code });
      const compute: GPUProgrammableStage = { module, entryPoint };
      if (constantEntries.length) {
        compute.constants = Object.fromEntries(constantEntries) as Record<string, number>;
      }
      try {
        pending = Promise.resolve(
          this.device.createComputePipelineAsync({ layout: 'auto', compute }),
        );
      } catch (error) {
        if (variants.size === 0) this.computePipelineCache.delete(code);
        throw error;
      }
      variants.set(variantKey, pending);
    }
    try {
      return await pending;
    } catch (error) {
      if (variants.get(variantKey) === pending) variants.delete(variantKey);
      if (variants.size === 0) this.computePipelineCache.delete(code);
      throw error;
    }
  }

  clear(): void {
    this.computePipelineCache.clear();
    this.rejectedSpecializedShaders.clear();
  }
}
