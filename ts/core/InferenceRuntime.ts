import { assertBackendEngine } from '../backends/BackendEngine.js';
import { Graph } from './Graph.js';
import { GraphLoader } from './GraphLoader.js';
import { ModelBuilder } from './ModelBuilder.js';
import type { GraphLoaderOptions } from './GraphLoader.js';
import type { SafetensorsFile } from './Safetensors.js';
import type {
  AdapterDescription,
  AdapterExportOptions,
  AdapterLoadOptions,
  AdapterSpec,
  AdapterStageOptions,
  AdapterUpdateOptions,
  AdapterVersion,
  BackendEngineContract,
  GraphInspection,
  GraphInspectionOptions,
} from '../types.js';

export type RuntimeEngine = BackendEngineContract;

export interface RuntimeEngineEntry {
  type: string;
  engine: RuntimeEngine;
  device?: unknown;
}

type AdapterSource = SafetensorsFile | ArrayBuffer | ArrayBufferView;
type AdapterUpdates = Record<string, Record<string, unknown>>;
type WeightsBaseUrl = string | URL | null | undefined;

export function caughtMessage(error: unknown): unknown {
  return error !== null && (typeof error === 'object' || typeof error === 'function')
    ? (error as { message?: unknown }).message
    : undefined;
}

export function checkedEngine(value: unknown, label: string): RuntimeEngine {
  assertBackendEngine(value, label);
  return value as RuntimeEngine;
}

/** Backend-neutral graph, model-loading, adapter, and compilation facade. */
export class InferenceRuntime {
  engines: RuntimeEngineEntry[];
  weightsBaseUrl: WeightsBaseUrl;

  constructor() {
    this.engines = [];
    this.weightsBaseUrl = null;
  }

  createGraph(): Graph {
    return new Graph();
  }

  /** Create an empty, API-designed model backed by a ModelBuilder. */
  createModel(graph: Graph = this.createGraph()): ModelBuilder {
    return new ModelBuilder(graph);
  }

  createModelBuilder(graph: Graph = this.createGraph()): ModelBuilder {
    return this.createModel(graph);
  }

  async loadGraph(
    sources: string | readonly string[],
    options: GraphLoaderOptions = {},
  ): Promise<Graph> {
    const graph = this.createGraph();
    return await GraphLoader.load(graph, sources, options);
  }

  inspectGraph(graph: Graph, options: GraphInspectionOptions = {}): GraphInspection {
    if (!graph || typeof graph.inspect !== 'function') {
      throw new Error('[VolvoxAI] inspectGraph expects a Graph instance.');
    }
    return graph.inspect(options);
  }

  stageAdapter(
    graph: Graph,
    name: string,
    spec: AdapterSpec,
    options: AdapterStageOptions = {},
  ): AdapterDescription {
    return graph.stageAdapter(name, spec, options);
  }

  updateAdapter(
    graph: Graph,
    name: string,
    updates: AdapterUpdates,
    options: AdapterUpdateOptions = {},
  ): AdapterDescription {
    return graph.updateAdapter(name, updates, options);
  }

  loadAdapter(
    graph: Graph,
    source: AdapterSource,
    options: AdapterLoadOptions = {},
  ): AdapterDescription {
    return graph.loadAdapter(source, options);
  }

  activateAdapter(
    graph: Graph,
    name: string | null,
    version?: AdapterVersion,
  ): AdapterDescription | null {
    return graph.activateAdapter(name, version);
  }

  removeAdapter(graph: Graph, name: string, version?: AdapterVersion): boolean {
    return graph.removeAdapter(name, version);
  }

  listAdapters(graph: Graph): AdapterDescription[] {
    return graph.listAdapters();
  }

  exportAdapter(
    graph: Graph,
    name: string,
    version?: AdapterVersion | AdapterExportOptions,
    options: AdapterExportOptions = {},
  ): ArrayBuffer | Blob | SafetensorsFile {
    return graph.exportAdapter(name, version, options);
  }

  mergeAdapter(graph: Graph, name: string, version?: AdapterVersion): AdapterDescription {
    return graph.mergeAdapter(name, version);
  }

  unmergeAdapter(graph: Graph): AdapterDescription | null {
    return graph.unmergeAdapter();
  }

  async compile(graph: Graph, weightsUrl?: string | URL | null): Promise<BackendEngineContract> {
    this.weightsBaseUrl = weightsUrl;
    console.log(`[VolvoxAI] Compiling graph with ${graph.nodes.length} nodes...`);

    for (const entry of this.engines) {
      try {
        console.log(`[VolvoxAI] Trying to allocate graph on ${entry.engine.constructor.name}...`);
        await entry.engine.allocateGraph(graph);
        console.log(`[VolvoxAI] ${entry.engine.constructor.name} compiled successfully.`);
        return entry.engine;
      } catch (error) {
        console.warn(
          `[VolvoxAI] Compilation failed on ${entry.type}. Falling back to next tier. ` +
          `Error: ${String(caughtMessage(error))}`,
        );
      }
    }

    throw new Error('[VolvoxAI] All engine tiers failed to compile the graph.');
  }

  /** Compile into a fresh backend owner without replacing current allocations. */
  async compileDetached(
    graph: Graph,
    weightsUrl?: string | URL | null,
  ): Promise<BackendEngineContract> {
    this.weightsBaseUrl = weightsUrl;
    console.log(`[VolvoxAI] Compiling detached graph with ${graph.nodes.length} nodes...`);
    for (const entry of this.engines) {
      if (typeof entry.engine.fork !== 'function') continue;
      let engine: RuntimeEngine | null = null;
      try {
        const candidate = await entry.engine.fork();
        engine = checkedEngine(candidate, `[VolvoxAI] Detached ${entry.type} backend`);
        await engine.allocateGraph(graph);
        console.log(`[VolvoxAI] Detached ${engine.constructor.name} compiled successfully.`);
        return engine;
      } catch (error) {
        engine?.dispose?.();
        console.warn(
          `[VolvoxAI] Detached compilation failed on ${entry.type}. Falling back to next tier. ` +
          `Error: ${String(caughtMessage(error))}`,
        );
      }
    }
    throw new Error('[VolvoxAI] No detachable engine tier could compile the graph.');
  }
}
