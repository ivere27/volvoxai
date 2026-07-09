# Browser and Node Runtime

VolvoxAI exposes one browser API and chooses the best available execution tier at
initialization and compilation time.

## Backend Selection

`VolvoxAI.init(preferredBackend = "auto", wasmUrl = "./volvoxai.wasm")` accepts a
backend string or a strict ordered backend array.

| Call | Behavior |
| --- | --- |
| `VolvoxAI.init()` or `VolvoxAI.init("auto")` | Try WebNN, then WebGPU, then WASM, then CPU. |
| `VolvoxAI.init("webnn")` | Prefer WebNN; fall back to WASM then CPU. |
| `VolvoxAI.init("webgpu")` | Prefer WebGPU; fall back to WASM then CPU. |
| `VolvoxAI.init("wasm")` | Skip WebNN/WebGPU and use WASM with CPU fallback. |
| `VolvoxAI.init("cpu")` | CPU only. |
| `VolvoxAI.init(["wasm", "webgpu"])` | Strict allow-list. No CPU fallback unless `"cpu"` is included. |

Array mode is strict. `init()` throws if none of the listed backends initializes,
and `compile()` throws if the initialized backends cannot compile the graph.

## Tier 1: WebNN

`WebNNEngine` builds an `MLGraphBuilder` graph and dispatches through
`navigator.ml`. The browser may route work to NPU, GPU, or CPU depending on the
platform, browser flags, drivers, and supported ops.

Current WebNN coverage:

`MatMul`/`Linear`/`Gemm`, `Add`, `Mul`, `ReLU`, `GELU`, `SiLU`/`Swish`,
`Sigmoid`, `Softmax`, `Reshape`/`Flatten`, `LayerNorm`, `Conv2D`,
`Embedding`, and `SDPA` decomposed into lower-level WebNN ops.

Missing low-cost mappings include `Sub`, `Div`, `Tanh`, `Clip`, `LeakyReLU`,
`PReLU`, `HardSwish`, `HardSigmoid`, `Transpose`, `Concat`, `Split`, `Slice`,
`Pad`, `Where`, `Gather`, `Cast`, `Expand`, reductions, pooling, batch norm,
resampling, and `ConvTranspose2D`.

Missing composite formulas include `RMSNorm`, `CrossSDPA`, `CrossAttention`,
`LogSoftmax`, `DequantizeLinear`, and `Conv1D` as a reshape plus `conv2d`.

WebNN requires a secure context such as HTTPS or localhost. Chromium WebNN support
is evolving and often needs `--enable-features=WebMachineLearningNeuralNetwork`.
`deviceType: "npu"` is not proof that an NPU executed the work; unsupported
accelerators can fall back to CPU.

Useful references:

- W3C WebNN: <https://www.w3.org/TR/webnn/>
- WebNN compatibility: <https://webnn.io/en/api-reference/browser-compatibility/api>
- Chromium flags: <https://webnn.io/en/api-reference/browser-compatibility/chrome-flags>

## Tier 2: WebGPU

`GraphExecutor` allocates one GPU buffer per tensor, uploads weights once, and
builds one compute pipeline per node. `execute()` replays the prebuilt pipelines in
one command encoder, flushing every 20 dispatches.

The WebGPU path keeps the model resident on the GPU and returns a `GPUBuffer`.
Current limitation: it returns the final node's first output rather than a map of
all `graph.outputNames`. Multi-output models should use WASM/CPU until WebGPU
multi-output readback is implemented.

Unsupported WebGPU shader nodes currently warn and skip, leaving the output buffer
unwritten. Use [operation_list.md](operation_list.md) to check whether a model's
ops are safe on WebGPU.

## Tier 3: WASM SIMD

`WasmEngine` loads the freestanding `volvoxai.wasm` module built from
`native/kernels.c`. It uses a bump allocator over `__heap_base` and places graph
tensors in WASM linear memory. Some pointwise, broadcast, and gather operations are
handled in JS over typed-array views into the same heap.

## Tier 4: CPU

`CPUEngine` is the pure-JS reference backend. It is slower, but dependency-free and
useful for debugging, fallback behavior, and cross-tier correctness checks.

## Node Import Behavior

`ShaderLibrary.js` statically imports every `shaders/*.wgsl` file as text, which
only works in the bundled browser build. It is not re-exported from `js/index.js`.
`GraphExecutor` lazy-loads it only when the WebGPU path compiles, so plain Node
imports can use WASM/CPU without a bundler.

