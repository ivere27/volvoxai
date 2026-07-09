import { Tensor } from './Tensor.js';
import { Graph } from './Graph.js';
import { CPUEngine } from './CPUEngine.js';
import { WasmEngine } from './WasmEngine.js';
import { GraphExecutor } from './GraphExecutor.js';
import { GraphLoader } from './GraphLoader.js';
import { WebNNEngine } from './WebNNEngine.js';

export class VolvoxAI {
  constructor() {
    this.engines = [];
    this.weightsBaseUrl = null;
  }
  
  static async init(preferredBackend = "auto", wasmUrl = "./volvoxai.wasm") {
    const instance = new VolvoxAI();
    const stringOrders = {
      auto: ["webnn", "webgpu", "wasm", "cpu"],
      webnn: ["webnn", "wasm", "cpu"],
      webgpu: ["webgpu", "wasm", "cpu"],
      wasm: ["wasm", "cpu"],
      cpu: ["cpu"],
    };
    const validBackends = new Set(["webnn", "webgpu", "wasm", "cpu"]);
    const strictList = Array.isArray(preferredBackend);
    const order = strictList ? [...preferredBackend] : stringOrders[preferredBackend];

    if (!order || order.length === 0) {
      throw new Error(`[VolvoxAI] Invalid backend selection: ${JSON.stringify(preferredBackend)}`);
    }

    for (const backend of order) {
      if (!validBackends.has(backend)) {
        throw new Error(`[VolvoxAI] Unknown backend '${backend}'. Use 'auto', 'webnn', 'webgpu', 'wasm', 'cpu', or an array of those backend names.`);
      }
    }

    const added = new Set();
    for (const backend of order) {
      if (added.has(backend)) continue;

      if (backend === "webnn") {
        if (typeof navigator !== 'undefined' && navigator.ml) {
          try {
            const context = await navigator.ml.createContext({ deviceType: 'npu' });
            console.log("[VolvoxAI] WebNN Engine (NPU) Initialized successfully.");
            instance.engines.push({ type: 'webnn', engine: new WebNNEngine(context) });
            added.add(backend);
          } catch (e) {
            console.warn("[VolvoxAI] WebNN initialization failed.", e.message);
          }
        } else if (strictList || preferredBackend === "webnn") {
          console.warn("[VolvoxAI] WebNN is not supported.");
        }
      } else if (backend === "webgpu") {
        if (typeof navigator !== 'undefined' && navigator.gpu) {
          try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
              const device = await adapter.requestDevice();
              console.log("[VolvoxAI] WebGPU Engine Initialized successfully.");
              instance.engines.push({ type: 'webgpu', device: device });
              added.add(backend);
            } else {
              console.warn("[VolvoxAI] WebGPU adapter is not available.");
            }
          } catch (e) {
            console.warn("[VolvoxAI] WebGPU initialization failed.", e.message);
          }
        } else if (strictList || preferredBackend === "webgpu") {
          console.warn("[VolvoxAI] WebGPU is not supported.");
        }
      } else if (backend === "wasm") {
        const wasmEngine = await WasmEngine.init(wasmUrl);
        if (wasmEngine) {
          console.log("[VolvoxAI] WASM Engine Initialized successfully.");
          instance.engines.push({ type: 'wasm', engine: wasmEngine });
          added.add(backend);
        } else {
          console.warn("[VolvoxAI] WASM initialization failed.");
        }
      } else if (backend === "cpu") {
        console.log("[VolvoxAI] Pure JS CPU Engine Initialized.");
        instance.engines.push({ type: 'cpu', engine: new CPUEngine() });
        added.add(backend);
      }
    }

    if (instance.engines.length === 0) {
      throw new Error(`[VolvoxAI] None of the requested backends initialized: ${order.join(", ")}`);
    }

    return instance;
  }

  createGraph() {
    return new Graph();
  }

  async loadGraph(safetensorsUrl) {
    const graph = this.createGraph();
    return await GraphLoader.load(graph, safetensorsUrl);
  }

  async compile(graph, weightsUrl) {
    this.weightsBaseUrl = weightsUrl;
    console.log(`[VolvoxAI] Compiling graph with ${graph.nodes.length} nodes...`);
    
    for (const entry of this.engines) {
      try {
        if (entry.type === 'webgpu') {
          console.log(`[VolvoxAI] Trying to allocate graph on WebGPU (Tier 2)...`);
          const executor = new GraphExecutor(entry.device, graph);
          await executor.compile();
          console.log(`[VolvoxAI] WebGPU Engine compiled successfully.`);
          return executor;
        } else {
          console.log(`[VolvoxAI] Trying to allocate graph on ${entry.engine.constructor.name}...`);
          await entry.engine.allocateGraph(graph);
          console.log(`[VolvoxAI] ${entry.engine.constructor.name} compiled successfully.`);
          return entry.engine;
        }
      } catch (e) {
        console.warn(`[VolvoxAI] Compilation failed on ${entry.type}. Falling back to next tier. Error: ${e.message}`);
      }
    }
    
    throw new Error("[VolvoxAI] All engine tiers failed to compile the graph.");
  }
}
