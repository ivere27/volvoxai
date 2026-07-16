import { CPUEngine } from './backends/CPUEngine.js';
import { WasmEngine } from './backends/WasmEngine.js';
import { WebGPUEngine } from './backends/WebGPUEngine.js';
import { WebNNEngine } from './backends/WebNNEngine.js';
import {
  InferenceRuntime,
  caughtMessage,
  checkedEngine,
  type RuntimeEngine,
} from './core/InferenceRuntime.js';
import type {
  BackendFactory,
  BackendSelection,
} from './types.js';

const BUILTIN_BACKENDS = Object.freeze(['webnn', 'webgpu', 'wasm', 'cpu']);
const BUILTIN_BACKEND_SET = new Set(BUILTIN_BACKENDS);
const registeredBackendFactories = new Map<string, BackendFactory>();

interface BrowserMLContextFactory {
  createContext(options: { deviceType: string }): Promise<unknown>;
}

interface BrowserGPUAdapter {
  requestDevice(): Promise<GPUDevice>;
}

interface BrowserGPUFactory {
  requestAdapter(): Promise<BrowserGPUAdapter | null>;
}

interface BrowserBackendNavigator {
  ml?: BrowserMLContextFactory;
  gpu?: BrowserGPUFactory;
}

function browserNavigator(): BrowserBackendNavigator | undefined {
  return typeof navigator === 'undefined'
    ? undefined
    : navigator as Navigator & BrowserBackendNavigator;
}

function backendName(name: string): string {
  if (typeof name !== 'string' || !/^[a-z][a-z0-9._-]*$/.test(name)) {
    throw new Error('[VolvoxAI] Backend names must begin with a lowercase letter and contain only lowercase letters, digits, dot, underscore, or dash.');
  }
  return name;
}

export class VolvoxAI extends InferenceRuntime {

  /** Register an out-of-tree browser backend without changing the core enum. */
  static registerBackend(name: string, factory: BackendFactory): () => boolean {
    name = backendName(name);
    if (name === 'auto' || BUILTIN_BACKEND_SET.has(name)) {
      throw new Error(`[VolvoxAI] Built-in backend '${name}' cannot be replaced.`);
    }
    if (typeof factory !== 'function') {
      throw new Error(`[VolvoxAI] Backend '${name}' factory must be a function.`);
    }
    if (registeredBackendFactories.has(name)) {
      throw new Error(`[VolvoxAI] Backend '${name}' is already registered.`);
    }
    registeredBackendFactories.set(name, factory);
    return () => registeredBackendFactories.get(name) === factory &&
      registeredBackendFactories.delete(name);
  }

  static unregisterBackend(name: string): boolean {
    return registeredBackendFactories.delete(backendName(name));
  }

  static listBackends(): readonly string[] {
    return Object.freeze([...BUILTIN_BACKENDS, ...registeredBackendFactories.keys()]);
  }
  
  static async init(
    preferredBackend: BackendSelection = "auto",
    wasmUrl: string | URL = new URL("./volvoxai.wasm", import.meta.url),
  ): Promise<VolvoxAI> {
    const instance = new this();
    const stringOrders: Record<string, readonly string[]> = {
      auto: ["webnn", "webgpu", "wasm", "cpu"],
      webnn: ["webnn", "wasm", "cpu"],
      webgpu: ["webgpu", "wasm", "cpu"],
      wasm: ["wasm", "cpu"],
      cpu: ["cpu"],
    };
    const strictList = typeof preferredBackend !== 'string';
    const order: string[] | null = typeof preferredBackend !== 'string'
      ? [...preferredBackend]
      : (stringOrders[preferredBackend] ? [...stringOrders[preferredBackend]] : null) ||
        (registeredBackendFactories.has(preferredBackend) ? [preferredBackend] : null);

    if (!order || order.length === 0) {
      throw new Error(`[VolvoxAI] Invalid backend selection: ${JSON.stringify(preferredBackend)}`);
    }

    for (const backend of order) {
      if (!BUILTIN_BACKEND_SET.has(backend) && !registeredBackendFactories.has(backend)) {
        throw new Error(`[VolvoxAI] Unknown backend '${backend}'. Available backends: ${this.listBackends().join(', ')}.`);
      }
    }

    const added = new Set();
    for (const backend of order) {
      if (added.has(backend)) continue;

      if (backend === "webnn") {
        const runtimeNavigator = browserNavigator();
        if (runtimeNavigator?.ml) {
          try {
            const context = await runtimeNavigator.ml.createContext({ deviceType: 'npu' });
            console.log("[VolvoxAI] WebNN Engine (NPU) Initialized successfully.");
            instance.engines.push({
              type: 'webnn',
              engine: new WebNNEngine(context) as unknown as RuntimeEngine,
            });
            added.add(backend);
          } catch (e) {
            console.warn("[VolvoxAI] WebNN initialization failed.", caughtMessage(e));
          }
        } else if (strictList || preferredBackend === "webnn") {
          console.warn("[VolvoxAI] WebNN is not supported.");
        }
      } else if (backend === "webgpu") {
        const runtimeNavigator = browserNavigator();
        if (runtimeNavigator?.gpu) {
          try {
            const adapter = await runtimeNavigator.gpu.requestAdapter();
            if (adapter) {
              const device = await adapter.requestDevice();
              console.log("[VolvoxAI] WebGPU Engine Initialized successfully.");
              instance.engines.push({
                type: 'webgpu',
                engine: new WebGPUEngine(device) as unknown as RuntimeEngine,
                // Retain the old entry.device integration point for training.
                device,
              });
              added.add(backend);
            } else {
              console.warn("[VolvoxAI] WebGPU adapter is not available.");
            }
          } catch (e) {
            console.warn("[VolvoxAI] WebGPU initialization failed.", caughtMessage(e));
          }
        } else if (strictList || preferredBackend === "webgpu") {
          console.warn("[VolvoxAI] WebGPU is not supported.");
        }
      } else if (backend === "wasm") {
        const wasmEngine = await WasmEngine.init(wasmUrl);
        if (wasmEngine) {
          console.log("[VolvoxAI] WASM Engine Initialized successfully.");
          instance.engines.push({
            type: 'wasm',
            engine: wasmEngine as unknown as RuntimeEngine,
          });
          added.add(backend);
        } else {
          console.warn("[VolvoxAI] WASM initialization failed.");
        }
      } else if (backend === "cpu") {
        console.log("[VolvoxAI] Pure JS CPU Engine Initialized.");
        instance.engines.push({
          type: 'cpu',
          engine: new CPUEngine() as unknown as RuntimeEngine,
        });
        added.add(backend);
      } else {
        try {
          const factory = registeredBackendFactories.get(backend)!;
          const candidate = await factory({
            name: backend,
            runtime: instance,
            wasmUrl,
          });
          if (candidate == null) {
            console.warn(`[VolvoxAI] Registered backend '${backend}' is not available.`);
            continue;
          }
          const engine = checkedEngine(candidate, `[VolvoxAI] Registered backend '${backend}'`);
          if (engine.backendName !== backend) {
            throw new Error(`[VolvoxAI] Registered backend '${backend}' returned backendName '${engine.backendName}'.`);
          }
          instance.engines.push({ type: backend, engine, device: engine.device });
          added.add(backend);
        } catch (e) {
          console.warn(
            `[VolvoxAI] Registered backend '${backend}' initialization failed.`,
            caughtMessage(e),
          );
        }
      }
    }

    if (instance.engines.length === 0) {
      throw new Error(`[VolvoxAI] None of the requested backends initialized: ${order.join(", ")}`);
    }

    return instance;
  }

}
