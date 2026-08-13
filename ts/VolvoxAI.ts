import { CPUEngine } from './backends/CPUEngine.js';
import { WasmEngine } from './backends/WasmEngine.js';
import { WebGPUEngine } from './backends/WebGPUEngine.js';
import {
  BuiltInBackendProvider,
  assertBackendProvider,
  type BackendProvider,
  type BackendProviderFactory,
} from './backends/BackendProvider.js';
import { CPUBackendProvider } from './backends/CPUBackendProvider.js';
import { WasmBackendProvider } from './backends/WasmBackendProvider.js';
import { WebGPUBackendProvider } from './backends/WebGPUBackendProvider.js';
import { Runtime, type RuntimeOptions } from './core/ContextRuntime.js';
import { VolvoxAIError } from './core/RuntimeErrors.js';

const BUILTIN_BACKENDS = Object.freeze(['webgpu', 'wasm', 'cpu-js']);

export type RuntimeProviderSource = BackendProvider | BackendProviderFactory;

export interface CreateRuntimeOptions extends RuntimeOptions {
  readonly backends?: readonly string[];
  readonly providers?: Readonly<Record<string, RuntimeProviderSource>>;
  readonly wasmUrl?: string | URL;
}

interface BrowserGPUAdapter {
  info?: import('./backends/WebGPUEngine.js').WebGPUAdapterIdentity;
  requestDevice(): Promise<GPUDevice>;
}

interface BrowserGPUFactory {
  requestAdapter(): Promise<BrowserGPUAdapter | null>;
}

interface BrowserBackendNavigator {
  gpu?: BrowserGPUFactory;
}

function browserNavigator(): BrowserBackendNavigator | undefined {
  return typeof navigator === 'undefined'
    ? undefined
    : navigator as Navigator & BrowserBackendNavigator;
}

function checkedName(name: string): string {
  if (typeof name !== 'string' || !/^[a-z][a-z0-9._-]*$/.test(name)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] Backend names must begin with a lowercase letter and contain only lowercase letters, digits, dot, underscore, or dash.', {
        phase: 'initialization',
      });
  }
  return name;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function requestWebGPUAdapter(factory: BrowserGPUFactory): Promise<BrowserGPUAdapter | null> {
  // A freshly launched Chromium/Vulkan process can transiently return null
  // while its surfaceless adapter finishes initializing. Bound the retry so a
  // genuinely unavailable device still falls through promptly.
  for (let attempt = 0; attempt < 3; attempt++) {
    const adapter = await factory.requestAdapter();
    if (adapter) return adapter;
    if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return null;
}

async function initializeBuiltin(
  name: string,
  wasmUrl: string | URL,
): Promise<BackendProvider> {
  if (name === 'cpu-js') return new CPUBackendProvider(new CPUEngine());
  if (name === 'wasm') {
    const engine = await WasmEngine.init(wasmUrl);
    if (!engine) throw new Error(`WASM could not load '${String(wasmUrl)}'.`);
    return new WasmBackendProvider(engine);
  }
  if (name === 'webgpu') {
    const runtimeNavigator = browserNavigator();
    if (!runtimeNavigator?.gpu) throw new Error('WebGPU is unavailable.');
    const adapter = await requestWebGPUAdapter(runtimeNavigator.gpu);
    if (!adapter) throw new Error('WebGPU adapter is unavailable.');
    const device = await adapter.requestDevice();
    return new WebGPUBackendProvider(new WebGPUEngine(device, { adapterInfo: adapter.info }));
  }
  throw new Error(`Unknown built-in backend '${name}'.`);
}

function runtimeProviderSources(
  providers: Readonly<Record<string, RuntimeProviderSource>>,
): Map<string, RuntimeProviderSource> {
  if (!providers || typeof providers !== 'object' || Array.isArray(providers)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] Runtime providers must be a name-to-provider object.', {
        phase: 'initialization',
      });
  }
  const prototype = Object.getPrototypeOf(providers);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] Runtime providers must be a plain name-to-provider object.', {
        phase: 'initialization',
      });
  }
  const result = new Map<string, RuntimeProviderSource>();
  for (const [rawName, source] of Object.entries(providers)) {
    const name = checkedName(rawName);
    if (BUILTIN_BACKENDS.includes(name)) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        `[VolvoxAI] Built-in backend provider '${name}' cannot be replaced.`, {
          phase: 'initialization', backend: name,
        });
    }
    if (typeof source !== 'function' && (!source || typeof source !== 'object')) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `[VolvoxAI] Backend provider '${name}' must be a provider instance or factory.`, {
          phase: 'initialization', backend: name,
        });
    }
    result.set(name, source);
  }
  return result;
}

/** Create the root handle for the Runtime -> Model -> CompiledModel -> ExecutionContext lifecycle. */
export async function createRuntime(options: CreateRuntimeOptions = {}): Promise<Runtime> {
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] Runtime options must be an object.', { phase: 'initialization' });
  }
  const {
    backends = BUILTIN_BACKENDS,
    providers = {},
    wasmUrl = new URL('./volvoxai.wasm', import.meta.url),
    onDiagnostic = null,
    memoryCapture,
    execution,
  } = options;
  if (!Array.isArray(backends) || backends.length === 0) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] Runtime backends must be a non-empty ordered array.', {
        phase: 'initialization',
      });
  }
  if (!((typeof wasmUrl === 'string' && wasmUrl.length > 0) || wasmUrl instanceof URL)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] wasmUrl must be a non-empty string or URL.', {
        phase: 'initialization',
      });
  }
  const order = [...backends];
  const providerSources = runtimeProviderSources(providers);
  const seen = new Set<string>();
  for (const name of order) {
    checkedName(name);
    if (seen.has(name)) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        `[VolvoxAI] Runtime backend '${name}' occurs more than once.`, {
          phase: 'initialization', backend: name,
        });
    }
    seen.add(name);
    if (!BUILTIN_BACKENDS.includes(name) && !providerSources.has(name)) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        `[VolvoxAI] Unknown backend provider '${name}'.`, {
          phase: 'initialization', backend: name,
        });
    }
  }

  const runtime = new Runtime({ onDiagnostic, memoryCapture, execution });
  for (const name of order) {
    let candidate: unknown = null;
    let retained = false;
    try {
      const source = providerSources.get(name);
      candidate = BUILTIN_BACKENDS.includes(name)
        ? await initializeBuiltin(name, wasmUrl)
        : typeof source === 'function'
          ? await source({ name, runtime, wasmUrl })
          : source;
      if (!candidate) {
        runtime._addInitializationFailure(name, `Backend provider '${name}' is unavailable.`);
        continue;
      }
      const provider = assertBackendProvider(candidate, `Backend provider '${name}'`);
      runtime._addProvider(name, provider);
      retained = true;
    } catch (error) {
      const close = (candidate as Partial<BackendProvider> | null)?.close;
      if (!retained && typeof close === 'function') {
        try { await close.call(candidate); } catch { /* Preserve initialization evidence. */ }
      }
      runtime._addInitializationFailure(
        name,
        `Backend provider '${name}' initialization failed: ${errorMessage(error)}`,
      );
    }
  }
  return runtime;
}

/** Stateless namespace; concrete state starts at Runtime. */
export const VolvoxAI = Object.freeze({
  createRuntime,
});
