import { WASM_PROFILE_IMPORTS } from './generated/wasmInternalAbi.mjs';

/** Host transport for release checks and standalone C/WASM kernel tools. */
export function wasmToolImports() {
  const env = {
    expf: Math.exp, logf: Math.log, powf: Math.pow, sqrtf: Math.sqrt,
    tanhf: Math.tanh, sinf: Math.sin, cosf: Math.cos,
  };
  return {
    env,
    synurang: { wakeup: () => {} },
    math: env,
    host: {
      vx_host_monotonic_micros_v1: () => Math.floor(performance.now() * 1000),
      vx_host_random_u64_v1: () => {
        const words = crypto.getRandomValues(new Uint32Array(2));
        return (BigInt(words[0]) << 32n) | BigInt(words[1]);
      },
    },
    gpu: Object.fromEntries(WASM_PROFILE_IMPORTS.full
      .filter((entry) => entry.startsWith('gpu.'))
      .map((entry) => {
        const name = entry.slice(4).split(':')[0];
        return [name, () => name === 'vx_gpu_available' || name === 'vx_gpu_ensure' ? 0 : -1];
      })),
  };
}
