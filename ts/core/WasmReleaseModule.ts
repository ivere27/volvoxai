import type { WasmGpuBridge } from '../generated/gpuBridge.js';
export type { WasmGpuBridge } from '../generated/gpuBridge.js';
declare const __VOLVOXAI_BROWSER_ONLY__: boolean | undefined;

// Keep the Node-only transport unresolved in browser bundles. The runtime
// branch below is the same boundary used by the full-profile authoring loader.
const importRuntimeModule = (specifier: string) => import(specifier);

function hostMonotonicMicros(): number {
  const clock = globalThis.performance;
  if (!clock || typeof clock.now !== 'function') return -1;
  const micros = Math.floor(clock.now() * 1000);
  return Number.isSafeInteger(micros) && micros >= 0 ? micros : -1;
}

const compiledModules = new Map<string, Promise<WebAssembly.Module>>();

/**
 * A bridge and the moment it learns where memory is.
 *
 * Imports must exist before `new WebAssembly.Instance`, but the linear memory
 * they read is exported by that same instance. The circle is real, so it is
 * stated rather than worked around: `imports` is supplied at instantiation and
 * `attach` is called immediately afterwards, before any engine call can run.
 */
export interface WasmGpuBridgeHost {
  readonly imports: WasmGpuBridge;
  attach(memory: WebAssembly.Memory): void;
  /** Prepare a device only after C requests it for a pending proto call. */
  prepare?(): Promise<void>;
  /** Notify the call executor when device completion changes C-visible state. */
  setWakeup?(callback: () => void): void;
  /** Wait for submitted work and validation scopes without polling timers. */
  waitForCompletion(): Promise<void>;
  /** Retire device objects when the attached WASM owner is closed. */
  close?(): void;
}

/**
 * Both release parents import monotonic time and entropy. Only full receives
 * the device imports, supplied by its host even before a device is prepared.
 */
export function wasmReleaseModuleImports(
  gpuBridge?: WasmGpuBridge,
  wakeup: (token: number) => void = () => {},
): WebAssembly.Imports {
  return {
    synurang: { wakeup },
    host: {
      vx_host_monotonic_micros_v1: hostMonotonicMicros,
      vx_host_random_u64_v1: () => {
        const words = globalThis.crypto.getRandomValues(new Uint32Array(2));
        return (BigInt(words[0]) << 32n) | BigInt(words[1]);
      },
    },
    ...(gpuBridge ? { gpu: gpuBridge as unknown as WebAssembly.ModuleImports } : {}),
  };
}

function checkedWasmUrl(value: unknown): string | URL {
  if ((typeof value === 'string' && value.length > 0) || value instanceof URL) {
    return value;
  }
  throw new Error('Release WASM URL must be a non-empty string or URL.');
}

async function readReleaseWasm(wasmUrl: string | URL): Promise<BufferSource> {
  const browserOnly = typeof __VOLVOXAI_BROWSER_ONLY__ !== 'undefined' &&
    __VOLVOXAI_BROWSER_ONLY__ === true;
  const nodeProcess = (globalThis as typeof globalThis & {
    process?: { versions?: { node?: string } };
  }).process;
  if (!browserOnly && nodeProcess?.versions?.node) {
    const isUrl = typeof URL !== 'undefined' && wasmUrl instanceof URL;
    const href = isUrl ? wasmUrl.href : String(wasmUrl);
    if (!((isUrl && wasmUrl.protocol !== 'file:') || /^(?:https?|data):/i.test(href))) {
      const fs = await importRuntimeModule('fs');
      const file = href.startsWith('file:') && !isUrl ? new URL(href) : wasmUrl;
      return fs.promises.readFile(file);
    }
  }
  if (typeof globalThis.fetch !== 'function') {
    throw new Error(`Release WASM '${String(wasmUrl)}' requires a fetch transport.`);
  }
  const response = await globalThis.fetch(wasmUrl);
  if (!response.ok) throw new Error(`Release WASM '${String(wasmUrl)}' was not found.`);
  return response.arrayBuffer();
}

/** Read and compile one immutable release parent once per URL in this JS realm. */
export async function loadWasmReleaseModule(
  wasmUrl: string | URL,
): Promise<WebAssembly.Module> {
  const source = checkedWasmUrl(wasmUrl);
  const key = String(source);
  const cached = compiledModules.get(key);
  if (cached) return cached;

  const pending = readReleaseWasm(source).then((bytes) => WebAssembly.compile(bytes));
  compiledModules.set(key, pending);
  try {
    return await pending;
  } catch (error) {
    if (compiledModules.get(key) === pending) compiledModules.delete(key);
    throw error;
  }
}

/**
 * Give one owner an independent Instance and defined linear Memory.
 *
 * The device bridge is per-instance because the buffers it names are: two
 * owners share a compiled module but never a linear memory, so they cannot
 * share the mapping from host pointer to device buffer either. The inference
 * parent has no device imports; full requires its deferred device bridge.
 */
export function instantiateWasmReleaseModule(
  module: WebAssembly.Module,
  gpuBridge?: WasmGpuBridgeHost,
  wakeup?: (token: number) => void,
): WebAssembly.Instance {
  if (!(module instanceof WebAssembly.Module)) {
    throw new Error('Release WASM instantiation requires a compiled WebAssembly.Module.');
  }
  const instance = new WebAssembly.Instance(
    module, wasmReleaseModuleImports(gpuBridge?.imports, wakeup));
  gpuBridge?.attach(instance.exports.memory as WebAssembly.Memory);
  (instance.exports.__wasm_call_ctors as (() => void) | undefined)?.();
  return instance;
}
