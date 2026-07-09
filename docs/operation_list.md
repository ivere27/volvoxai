# Operation List and Status

This document describes the VolvoxAI operation list, what each operation means, where
its logic lives in the repository, and the current implementation status across browser,
Node, and native backends.

## What the Operation List Is

The operation list is the set of `opType` names that can appear in a Volvox blueprint
graph. A graph node uses one of these names, references input and output tensors, and
passes op-specific parameters through `params`.

The list is used by three parts of the system:

1. **Exporters and converters** decide which ONNX, TFLite, PyTorch, or custom graph
   operations can be emitted into Volvox format.
2. **Graph loading** resolves tensor shapes, weight layouts, quantized weights, and
   aliases before execution.
3. **Runtime backends** dispatch each node to the best available implementation.

The operation list is not a full ONNX or TFLite compatibility promise. It is the smaller
runtime contract that VolvoxAI currently implements for inference graphs.

## Status Legend

| Status | Meaning |
| --- | --- |
| Full | Implemented directly for the normal runtime path. |
| Partial | Implemented with shape, axis, dtype, layout, size, or execution-mode limits. |
| Fallback | The backend can run the op by delegating to a lower-level/reference helper. |
| Missing | Not wired on that backend. |
| Alias | Shape-only or alternate spelling handled by another operation. |
| Init only | Backend can initialize but has no per-op runtime dispatch. |

## Backend Names

| Backend | Runtime role | Main files |
| --- | --- | --- |
| CPU(JS) | Pure JavaScript reference backend used by the browser/Node ES module runtime. | [CPUEngine.js](../js/CPUEngine.js), [js/ops](../js/ops) |
| WASM | Browser/Node WebAssembly tier. Some ops call C kernels; some fall back to CPU(JS) helpers over WASM heap views. | [WasmEngine.js](../js/WasmEngine.js), [native/kernels](../native/kernels) |
| WebGPU | Browser WebGPU compute-shader backend. | [GraphExecutor.js](../js/GraphExecutor.js), [shaders](../shaders) |
| WebNN | Opportunistic browser accelerator through `navigator.ml`. Unsupported ops throw during build so lower tiers can run. | [WebNNEngine.js](../js/WebNNEngine.js) |
| CPU(Native) | Freestanding native runtime CPU path. This is separate from CPU(JS). | [engine_runtime.c](../native/engine_runtime.c), [native/kernels](../native/kernels), [conv_f32_opt.c](../native/conv_f32_opt.c), [quant_cpu_opt.c](../native/quant_cpu_opt.c) |
| Vulkan(Native) | Native Vulkan compute path. It is called from the native dispatcher and is separate from browser WebGPU. | [engine_runtime.c](../native/engine_runtime.c), [vulkan_engine.c](../native/vulkan_engine.c) |
| OpenGL(Native) | Native OpenGL ES compute path. It is called from the native dispatcher and is separate from browser WebGPU. | [engine_runtime.c](../native/engine_runtime.c), [opengl_engine.c](../native/opengl_engine.c) |
| Metal(Native) | Native Metal compute path on Apple platforms. It loads Naga-generated MSL and is called from the native dispatcher for selected F32 graph ops. | [engine_runtime.c](../native/engine_runtime.c), [metal_engine.m](../native/metal_engine.m) |

Note: [native/gpu](../native/gpu) contains older scaffold files. The active native
runtime paths are the files listed above.

## Browser and Node Status

This table covers the ES module runtime: CPU(JS), WASM, browser WebGPU, and WebNN.

| Operation | What it does | CPU(JS) | WASM | WebGPU | WebNN |
| --- | --- | --- | --- | --- | --- |
| `MatMul` | Dense matrix multiply with optional bias; FP32 and INT8-packed weight paths. | [Full](../js/ops/matMul.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/linearF32.wgsl) | [Full](../js/WebNNEngine.js) |
| `Linear`, `Gemm` | Dense-layer aliases used by exporters. | [Missing](../js/CPUEngine.js) | [Missing](../js/WasmEngine.js) | [Missing except layout pre-scan](../js/GraphExecutor.js) | [Full](../js/WebNNEngine.js) |
| `Conv2D` | NHWC 2D convolution, including grouped/depthwise and dilation. | [Full](../js/ops/conv2D.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/conv2D.wgsl) | [Full](../js/WebNNEngine.js) |
| `Conv1D` | 1D convolution for sequence/audio tensors. | [Full](../js/ops/conv1D.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/conv1D.wgsl) | [Missing](../js/WebNNEngine.js) |
| `ConvTranspose2D` | Transposed convolution / deconvolution. | [Full](../js/ops/convTranspose2D.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/convTranspose2D.wgsl) | [Missing](../js/WebNNEngine.js) |
| `QConv2D` | Quantized Conv2D node from native/direct TFLite exports. | [Loaded graphs convert to `Conv2D`](../js/GraphLoader.js) | [Loaded graphs convert to `Conv2D`](../js/GraphLoader.js) | [Loaded graphs convert to `Conv2D`](../js/GraphLoader.js) | [Loaded graphs convert to `Conv2D`](../js/GraphLoader.js) |
| `LayerNorm` | Layer normalization over the last feature axis. | [Full](../js/ops/layerNorm.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/layerNorm.wgsl) | [Full](../js/WebNNEngine.js) |
| `RMSNorm` | RMS normalization over the last feature axis. | [Full](../js/ops/rMSNorm.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/rMSNorm.wgsl) | [Missing](../js/WebNNEngine.js) |
| `BatchNorm2D` | Per-channel image batch normalization. | [Full](../js/ops/batchNorm2D.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/batchNorm2D.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Embedding` | Row lookup from token ids into an embedding table. | [Full](../js/ops/embedding.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/embedding.wgsl) | [Full](../js/WebNNEngine.js) |
| `SDPA` | Self-attention over packed QKV with causal behavior. | [Full](../js/ops/sDPA.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/sDPA.wgsl) | [Partial](../js/WebNNEngine.js) |
| `CrossSDPA` | Cross-attention over separate Q, K, and V tensors. | [Full](../js/ops/crossSDPA.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/crossSDPA.wgsl) | [Missing](../js/WebNNEngine.js) |
| `CrossAttention` | Cross-attention variant using Q plus packed KV and optional scale/bias. | [Full](../js/ops/crossAttention.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/crossAttention.wgsl) | [Missing](../js/WebNNEngine.js) |
| `MaxPool2D` | 2D max pooling. | [Full](../js/ops/maxPool2D.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/maxPool2D.wgsl) | [Missing](../js/WebNNEngine.js) |
| `AveragePool`, `AveragePool2D` | 2D average pooling. | [Full](../js/ops/averagePool2D.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/averagePool2D.wgsl) | [Missing](../js/WebNNEngine.js) |
| `GlobalAveragePool` | Global spatial average pooling. | [Full](../js/ops/globalAveragePool.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/globalAveragePool.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Resize` | Image resize. Bilinear is the normal path. | [Full](../js/ops/resize.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/resize.wgsl) | [Missing](../js/WebNNEngine.js) |
| `ResizeNearest2D` | Nearest-neighbor image resize. | [Full](../js/ops/resize.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/resize.wgsl) | [Missing](../js/WebNNEngine.js) |
| `UpsampleNearest2D`, `Upsample2x` | Nearest-neighbor 2x upsampling aliases. | [Full](../js/ops/upsample2x.js) | [Full for `UpsampleNearest2D`](../js/WasmEngine.js) | [Full for `UpsampleNearest2D`](../shaders/upsample2x.wgsl) | [Missing](../js/WebNNEngine.js) |
| `InterpLinear1D`, `Interp1D` | Linear 1D interpolation aliases. | [Full](../js/ops/interp1D.js) | [Full for `InterpLinear1D`](../js/WasmEngine.js) | [Full for `InterpLinear1D`](../shaders/interp1D.wgsl) | [Missing](../js/WebNNEngine.js) |
| `ReLU` | Rectified linear activation. | [Full](../js/ops/reLU.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/reLU.wgsl) | [Full](../js/WebNNEngine.js) |
| `LeakyReLU` | Leaky ReLU activation with `alpha`. | [Full](../js/ops/leakyReLU.js) | [Full or fallback](../js/WasmEngine.js) | [Full](../shaders/leakyReLU.wgsl) | [Missing](../js/WebNNEngine.js) |
| `PReLU` | Per-channel parametric ReLU. | [Full](../js/ops/pReLU.js) | [Full or fallback](../js/WasmEngine.js) | [Full](../shaders/pReLU.wgsl) | [Missing](../js/WebNNEngine.js) |
| `GELU` | Gaussian error linear unit. | [Full](../js/ops/gELU.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/gELU.wgsl) | [Full](../js/WebNNEngine.js) |
| `SiLU`, `Swish` | `x * sigmoid(x)` activation aliases. | [Full](../js/ops/siLU.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/siLU.wgsl) | [Full](../js/WebNNEngine.js) |
| `Sigmoid` | Logistic activation. | [Full](../js/ops/sigmoid.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/sigmoid.wgsl) | [Full](../js/WebNNEngine.js) |
| `HardSwish` | MobileNet-style hard swish activation. | [Full](../js/ops/hardSwish.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/hardSwish.wgsl) | [Missing](../js/WebNNEngine.js) |
| `HardSigmoid` | Piecewise-linear sigmoid approximation. | [Full](../js/ops/hardSigmoid.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/hardSigmoid.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Tanh` | Hyperbolic tangent activation. | [Full](../js/ops/tanh.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/tanh.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Clip` | Clamp values to min/max. | [Full](../js/ops/clip.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/clip.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Add` | Elementwise add with broadcasting. | [Full](../js/ops/add.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/broadcastBinary.wgsl) | [Full](../js/WebNNEngine.js) |
| `Mul` | Elementwise multiply with broadcasting. | [Full](../js/ops/mul.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/broadcastBinary.wgsl) | [Full](../js/WebNNEngine.js) |
| `Sub` | Elementwise subtract with broadcasting. | [Full](../js/ops/sub.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/broadcastBinary.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Div` | Elementwise divide with broadcasting. | [Full](../js/ops/div.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/broadcastBinary.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Softmax` | Softmax over the last axis. | [Full](../js/ops/softmax.js) | [Fallback](../js/WasmEngine.js) | [Partial, last axis only](../shaders/softmax.wgsl) | [Full](../js/WebNNEngine.js) |
| `LogSoftmax` | Log-softmax over the last axis. | [Full](../js/ops/logSoftmax.js) | [Fallback](../js/WasmEngine.js) | [Partial, last axis only](../shaders/logSoftmax.wgsl) | [Missing](../js/WebNNEngine.js) |
| `ReduceSum` | Reduction sum. | [Full](../js/ops/reduceSum.js) | [Full](../js/WasmEngine.js) | [Partial, non-2D input is flattened to 1-by-N](../shaders/reduce.wgsl) | [Missing](../js/WebNNEngine.js) |
| `ReduceMean` | Reduction mean. | [Full](../js/ops/reduceMean.js) | [Full](../js/WasmEngine.js) | [Partial, non-2D input is flattened to 1-by-N](../shaders/reduce.wgsl) | [Missing](../js/WebNNEngine.js) |
| `ArgMax` | Index of maximum value. | [Full](../js/ops/argMax.js) | [Fallback](../js/WasmEngine.js) | [Missing](../js/GraphExecutor.js) | [Missing](../js/WebNNEngine.js) |
| `Transpose` | General N-D tensor permutation. | [Full](../js/ops/transpose.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/generalTranspose.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Concat`, `Concat2` | Tensor concatenation. | [Full](../js/ops/concat2.js) | [Fallback](../js/WasmEngine.js) | [Partial, `Concat` only](../shaders/concatCopy.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Split` | Split a tensor into multiple outputs. | [Full](../js/ops/split.js) | [Fallback](../js/WasmEngine.js) | [Partial, equal-sized output slices](../shaders/split.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Slice` | Strided slice with starts, steps, and axes. | [Full](../js/ops/slice.js) | [Full](../js/WasmEngine.js) | [Partial, up to 4D-padded metadata](../shaders/slice.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Pad` | Constant padding. | [Full](../js/ops/pad.js) | [Full](../js/WasmEngine.js) | [Partial, image/4D-oriented metadata](../shaders/pad.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Expand`, `Broadcast` | Broadcast a tensor to a larger shape. | [Full](../js/ops/expand.js) | [Fallback](../js/WasmEngine.js) | [Partial, 4D-padded broadcast](../shaders/expand.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Gather` | Gather values by index. | [Full](../js/ops/gather.js) | [Full](../js/WasmEngine.js) | [Partial, axis 0 only](../shaders/gather.wgsl) | [Missing](../js/WebNNEngine.js) |
| `GatherElements` | Elementwise indexed gather. | [Full](../js/ops/gatherElements.js) | [Fallback](../js/WasmEngine.js) | [Missing](../js/GraphExecutor.js) | [Missing](../js/WebNNEngine.js) |
| `Where`, `Mask` | Select values by condition. | [Full](../js/ops/where.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/where.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Cast` | Dtype conversion. | [Full](../js/ops/cast.js) | [Fallback](../js/WasmEngine.js) | [Partial, FP32 copy](../shaders/copy.wgsl) | [Missing](../js/WebNNEngine.js) |
| `DequantizeLinear` | Convert quantized values to FP32 with scale and optional zero point. | [Full](../js/ops/dequantizeLinear.js) | [Fallback](../js/WasmEngine.js) | [Full](../shaders/dequantizeLinear.wgsl) | [Missing](../js/WebNNEngine.js) |
| `QuantizeLinear` | Quantize FP32 values. | [Missing](../js/CPUEngine.js) | [Missing](../js/WasmEngine.js) | [Missing](../js/GraphExecutor.js) | [Missing](../js/WebNNEngine.js) |
| `NonMaxSuppression` | NMS for object-detection boxes. | [Full](../js/ops/nonMaxSuppression.js) | [Fallback](../js/WasmEngine.js) | [Missing](../js/GraphExecutor.js) | [Missing](../js/WebNNEngine.js) |
| `Reshape` | Shape-only tensor view/copy. | [Alias](../js/ops/reshape.js) | [Alias](../js/WasmEngine.js) | [Alias](../shaders/copy.wgsl) | [Full](../js/WebNNEngine.js) |
| `Flatten` | Shape-only flatten. | [Alias](../js/ops/reshape.js) | [Alias](../js/WasmEngine.js) | [Alias](../shaders/copy.wgsl) | [Full](../js/WebNNEngine.js) |
| `Squeeze` | Remove size-1 dimensions. | [Alias](../js/ops/reshape.js) | [Alias](../js/WasmEngine.js) | [Alias](../shaders/copy.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Unsqueeze` | Add size-1 dimensions. | [Alias](../js/ops/reshape.js) | [Alias](../js/WasmEngine.js) | [Alias](../shaders/copy.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Dropout` | Inference-time pass-through. | [Alias](../js/ops/reshape.js) | [Alias](../js/WasmEngine.js) | [Alias](../shaders/copy.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Identity` | Pass-through copy. | [Alias](../js/ops/reshape.js) | [Alias](../js/WasmEngine.js) | [Alias](../shaders/copy.wgsl) | [Missing](../js/WebNNEngine.js) |
| `SpatialSoftargmaxY` | Custom vertical soft-argmax primitive for vision postprocessing. | [Full](../js/ops/spatialSoftargmaxY.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/spatialSoftargmaxY.wgsl) | [Missing](../js/WebNNEngine.js) |
| `ProfileX` | Custom horizontal profile/reduction primitive. | [Full](../js/ops/profileX.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/profileX.wgsl) | [Missing](../js/WebNNEngine.js) |
| `ProfileY` | Custom vertical profile/reduction primitive. | [Full](../js/ops/profileY.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/profileY.wgsl) | [Missing](../js/WebNNEngine.js) |
| `MeanHeight` | Custom height mean primitive. | [Full](../js/ops/meanHeight.js) | [Full](../js/WasmEngine.js) | [Full](../shaders/meanHeight.wgsl) | [Missing](../js/WebNNEngine.js) |
| `Shape` | Emit tensor shape. | [Missing](../js/CPUEngine.js) | [Missing](../js/WasmEngine.js) | [Missing](../js/GraphExecutor.js) | [Missing](../js/WebNNEngine.js) |
| `Size` | Emit tensor element count. | [Missing](../js/CPUEngine.js) | [Missing](../js/WasmEngine.js) | [Missing](../js/GraphExecutor.js) | [Missing](../js/WebNNEngine.js) |
| `TopK` | Top-k values/indices. | [Missing](../js/CPUEngine.js) | [Missing](../js/WasmEngine.js) | [Missing](../js/GraphExecutor.js) | [Missing](../js/WebNNEngine.js) |

## Native Backend Status

This table separates CPU(Native), Vulkan(Native), OpenGL(Native), and Metal(Native).
Vulkan and OpenGL graph ops are attempted before CPU(Native), but only for selected F32
nodes and only outside native decode/prefill modes. If a Vulkan/OpenGL graph op is not
accepted, the native dispatcher continues to CPU(Native); rows with CPU(Native)
`Missing` still require the GPU path to accept the node. Large native `MatMul` can also
use one-shot Vulkan/OpenGL offload when enabled.

Native shader generation is separate from native runtime support. The tracked
`shaders/*.wgsl` files are translated by `make compile_shaders` / Naga into ignored
build outputs under `native/shaders/{spv,glsl,gles,metal}`. An op is marked supported
for Vulkan/OpenGL only when `engine_runtime.c` has a dispatcher path and the native
backend has a C wrapper for that shader. Metal follows the same rule on Apple builds:
the MSL may be generated for many ops, but a row is marked supported only when
`engine_runtime.c` dispatches to a `metal_graph_*` wrapper.

| Operation | CPU(Native) | Vulkan(Native) | OpenGL(Native) | Metal(Native) |
| --- | --- | --- | --- | --- |
| `MatMul` | [Full](../native/kernels/broadcast_ops.c) | [Partial, large one-shot `linearF32` offload only](../native/vulkan_engine.c) | [Partial, large one-shot `linearF32RowMajor` offload only](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Linear`, `Gemm` | [Full](../native/kernels/broadcast_ops.c) | [Partial, large one-shot `linearF32` offload only](../native/vulkan_engine.c) | [Partial, large one-shot `linearF32RowMajor` offload only](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Conv2D` | [Full](../native/conv_f32_opt.c) | [Partial, selected F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, selected F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `QConv2D` | [Full with FP32 fallback](../native/quant_cpu_opt.c) | [Missing](../native/vulkan_engine.c) | [Missing](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Conv1D` | [Missing](../native/engine_runtime.c) | [Partial, F32 batch-1/channel-length graph path](../native/vulkan_engine.c) | [Partial, F32 batch-1/channel-length graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 batch-1/channel-length graph path](../native/metal_engine.m) |
| `ConvTranspose2D` | [Missing](../native/engine_runtime.c) | [Partial, F32 NHWC graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `LayerNorm` | [Full](../native/kernels/layernorm.c) | [Partial, F32 with weight+bias graph path](../native/vulkan_engine.c) | [Partial, F32 with weight+bias graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `RMSNorm` | [Full](../native/kernels/math_nlp.c) | [Partial, F32 graph path](../native/vulkan_engine.c) | [Partial, F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `BatchNorm2D` | [Full](../native/kernels/edge_primitives.c) | [Partial, F32 NHWC graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Embedding` | [Full](../native/kernels/embedding.c) | [Partial, F32 token-id graph path](../native/vulkan_engine.c) | [Partial, F32 token-id graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `SDPA` | [Full, includes decode KV-cache handling](../native/kernels/sdpa.c) | [Partial, full-forward F32 causal graph path, head_dim <= 64](../native/vulkan_engine.c) | [Partial, full-forward F32 causal graph path, head_dim <= 64](../native/opengl_engine.c) | [Partial, Apple-only full-forward F32 causal graph path, head_dim <= 64](../native/metal_engine.m) |
| `CrossSDPA` | [Full](../native/kernels/cross_sdpa.c) | [Partial, full-forward F32 graph path, head_dim <= 64](../native/vulkan_engine.c) | [Partial, full-forward F32 graph path, head_dim <= 64](../native/opengl_engine.c) | [Partial, Apple-only full-forward F32 graph path, head_dim <= 64](../native/metal_engine.m) |
| `CrossAttention` | [Missing](../native/engine_runtime.c) | [Partial, F32 Q/KV/weight graph path, d_model/head_dim <= 64](../native/vulkan_engine.c) | [Partial, F32 Q/KV/weight graph path, d_model/head_dim <= 64](../native/opengl_engine.c) | [Partial, Apple-only F32 Q/KV/weight graph path, d_model/head_dim <= 64](../native/metal_engine.m) |
| `MaxPool2D` | [Full](../native/engine_runtime.c) | [Partial, F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `AveragePool`, `AveragePool2D` | [Missing](../native/engine_runtime.c) | [Partial, F32 NHWC graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `GlobalAveragePool` | [Full](../native/kernels/edge_primitives.c) | [Partial, F32 NHWC graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Resize` | [Missing](../native/engine_runtime.c) | [Partial, F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `ResizeNearest2D` | [Full](../native/engine_runtime.c) | [Partial, F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `UpsampleNearest2D` | [Full](../native/kernels/vision_ops.c) | [Partial, F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `InterpLinear1D`, `Interp1D` | [Missing](../native/engine_runtime.c) | [Partial, F32 2D channels-by-length graph path](../native/vulkan_engine.c) | [Partial, F32 2D channels-by-length graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `ReLU` | [Full](../native/kernels/activations.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `LeakyReLU` | [Missing](../native/engine_runtime.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `PReLU` | [Missing](../native/engine_runtime.c) | [Partial, F32 last-channel graph path](../native/vulkan_engine.c) | [Partial, F32 last-channel graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `GELU` | [Full](../native/kernels/activations.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `SiLU`, `Swish` | [Full](../native/kernels/math_nlp.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Sigmoid` | [Full](../native/kernels/edge_primitives.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `HardSwish` | [Full](../native/kernels/edge_primitives.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `HardSigmoid` | [Full](../native/kernels/edge_primitives.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Tanh` | [Missing](../native/engine_runtime.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Clip` | [Full](../native/kernels/math_nlp.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Add` | [Full](../native/kernels/broadcast_ops.c) | [Partial, same-shape F32 graph path](../native/vulkan_engine.c) | [Partial, same-shape F32 graph path plus Add3 fusion](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Mul` | [Full](../native/kernels/broadcast_ops.c) | [Partial, F32 scalar/modulo-broadcast graph path](../native/vulkan_engine.c) | [Partial, F32 scalar/modulo-broadcast graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 scalar/modulo-broadcast graph path](../native/metal_engine.m) |
| `Sub` | [Missing](../native/engine_runtime.c) | [Partial, F32 scalar/modulo-broadcast graph path](../native/vulkan_engine.c) | [Partial, F32 scalar/modulo-broadcast graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 scalar/modulo-broadcast graph path](../native/metal_engine.m) |
| `Div` | [Missing](../native/engine_runtime.c) | [Partial, F32 scalar/modulo-broadcast graph path](../native/vulkan_engine.c) | [Partial, F32 scalar/modulo-broadcast graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 scalar/modulo-broadcast graph path](../native/metal_engine.m) |
| `Softmax` | [Full](../native/kernels/math_nlp.c) | [Partial, last-axis F32 graph path](../native/vulkan_engine.c) | [Partial, last-axis F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `LogSoftmax` | [Missing](../native/engine_runtime.c) | [Partial, last-axis F32 graph path](../native/vulkan_engine.c) | [Partial, last-axis F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `ReduceSum` | [Missing](../native/engine_runtime.c) | [Partial, F32 2D rows or flattened 1-by-N](../native/vulkan_engine.c) | [Partial, F32 2D rows or flattened 1-by-N](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `ReduceMean` | [Missing](../native/engine_runtime.c) | [Partial, F32 2D rows or flattened 1-by-N](../native/vulkan_engine.c) | [Partial, F32 2D rows or flattened 1-by-N](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `ArgMax` | [Missing](../native/engine_runtime.c) | [Missing](../native/vulkan_engine.c) | [Missing](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Transpose` | [Full](../native/tensor_f32_opt.c) | [Partial, F32 graph path](../native/vulkan_engine.c) | [Partial, F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Concat` | [Full](../native/engine_runtime.c) | [Partial, flat concat axis 0 or axis 1 with batch 1](../native/vulkan_engine.c) | [Partial, flat concat axis 0 or axis 1 with batch 1](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Concat2` | [Missing](../native/engine_runtime.c) | [Missing](../native/vulkan_engine.c) | [Missing](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Split` | [Missing](../native/engine_runtime.c) | [Partial, F32 multi-output axis slices](../native/vulkan_engine.c) | [Partial, F32 multi-output axis slices](../native/opengl_engine.c) | [Partial, Apple-only F32 multi-output axis slices](../native/metal_engine.m) |
| `Slice` | [Missing](../native/engine_runtime.c) | [Partial, F32 up to 4D with positive steps](../native/vulkan_engine.c) | [Partial, F32 up to 4D with positive steps](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Pad` | [Missing](../native/engine_runtime.c) | [Partial, F32 up to 4D top/left constant pad](../native/vulkan_engine.c) | [Partial, F32 up to 4D top/left constant pad](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Expand`, `Broadcast` | [Missing](../native/engine_runtime.c) | [Partial, F32 up to 4D graph path](../native/vulkan_engine.c) | [Partial, F32 up to 4D graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Gather` | [Missing](../native/engine_runtime.c) | [Partial, F32 axis-0 graph path with F32 indices](../native/vulkan_engine.c) | [Partial, F32 axis-0 graph path with F32 indices](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `GatherElements` | [Missing](../native/engine_runtime.c) | [Missing](../native/vulkan_engine.c) | [Missing](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Where`, `Mask` | [Missing](../native/engine_runtime.c) | [Partial, same-size F32 graph path](../native/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Cast` | [Missing](../native/engine_runtime.c) | [Partial, FP32 copy graph path](../native/vulkan_engine.c) | [Partial, FP32 copy graph path](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `DequantizeLinear` | [Partial, materializes/copies to F32](../native/quant_cpu_opt.c) | [Partial, F32-buffer scale/zero-point shader path](../native/vulkan_engine.c) | [Partial, F32-buffer scale/zero-point shader path](../native/opengl_engine.c) | [Partial, Apple-only F32-buffer scale/zero-point shader path](../native/metal_engine.m) |
| `QuantizeLinear` | [Full](../native/quant_cpu_opt.c) | [Partial, F32-to-int8 QTensor sidecar graph path](../native/vulkan_engine.c) | [Partial, F32-to-int8 QTensor sidecar graph path](../native/opengl_engine.c) | [Partial, Apple-only F32-to-int8 QTensor sidecar graph path](../native/metal_engine.m) |
| `NonMaxSuppression` | [Missing](../native/engine_runtime.c) | [Partial, F32 sequential greedy graph path](../native/vulkan_engine.c) | [Partial, F32 sequential greedy graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 sequential greedy graph path](../native/metal_engine.m) |
| `Reshape` | [Alias](../native/engine_runtime.c) | [Alias, same-numel F32 graph alias](../native/vulkan_engine.c) | [Alias, same-numel F32 graph alias](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Flatten` | [Alias](../native/engine_runtime.c) | [Alias, same-numel F32 graph alias](../native/vulkan_engine.c) | [Alias, same-numel F32 graph alias](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Squeeze` | [Alias](../native/engine_runtime.c) | [Alias, same-numel F32 graph alias](../native/vulkan_engine.c) | [Alias, same-numel F32 graph alias](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Unsqueeze` | [Alias](../native/engine_runtime.c) | [Alias, same-numel F32 graph alias](../native/vulkan_engine.c) | [Alias, same-numel F32 graph alias](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Dropout` | [Alias](../native/engine_runtime.c) | [Alias, same-numel F32 graph alias](../native/vulkan_engine.c) | [Alias, same-numel F32 graph alias](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Identity` | [Alias](../native/engine_runtime.c) | [Alias, same-numel F32 graph alias](../native/vulkan_engine.c) | [Alias, same-numel F32 graph alias](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `SpatialSoftargmaxY` | [Missing](../native/engine_runtime.c) | [Partial, F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 NHWC batch-1 graph path](../native/metal_engine.m) |
| `ProfileX` | [Missing](../native/engine_runtime.c) | [Partial, F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 NHWC batch-1 graph path](../native/metal_engine.m) |
| `ProfileY` | [Missing](../native/engine_runtime.c) | [Partial, F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 NHWC batch-1 graph path](../native/metal_engine.m) |
| `MeanHeight` | [Missing](../native/engine_runtime.c) | [Partial, F32 NHWC batch-1 graph path](../native/vulkan_engine.c) | [Partial, F32 NHWC batch-1 graph path](../native/opengl_engine.c) | [Partial, Apple-only F32 NHWC batch-1 graph path](../native/metal_engine.m) |
| `Shape` | [Missing](../native/engine_runtime.c) | [Missing](../native/vulkan_engine.c) | [Missing](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `Size` | [Missing](../native/engine_runtime.c) | [Missing](../native/vulkan_engine.c) | [Missing](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |
| `TopK` | [Missing](../native/engine_runtime.c) | [Missing](../native/vulkan_engine.c) | [Missing](../native/opengl_engine.c) | [Init only](../native/metal_engine.m) |

## Known Gaps

1. **WebGPU output contract:** [GraphExecutor.execute()](../js/GraphExecutor.js)
   returns the first output buffer of the last node, not a map of all
   `graph.outputNames`. Multi-output read-back is a known WebGPU API gap.
2. **WebGPU partial ops:** `Gather` supports only axis 0. `Softmax` and `LogSoftmax`
   run over the last axis. `Split` assumes equal-sized outputs. Some
   shape/broadcast/pad kernels use 4D-padded metadata.
3. **WebGPU unsupported op behavior:** Missing shaders log a warning and skip
   the node during pipeline build, leaving that output buffer unwritten. Product code
   should prefer compile-time validation for strict GPU-only deployments.
4. **WebNN coverage:** WebNN is intentionally narrow. It can run common dense,
   activation, embedding, conv, reshape, layer norm, and decomposed SDPA graphs, but
   most vision postprocessing and shape ops are not mapped yet.
5. **Browser quantized Conv:** Browser [GraphLoader](../js/GraphLoader.js)
   dequantizes `QConv2D` weights to `Conv2D`. Native keeps a real quantized Conv path.
6. **Native GPU coverage:** Vulkan(Native), OpenGL(Native), and Metal(Native) are graph
   accelerators for selected F32 nodes; Vulkan/OpenGL also have large one-shot `MatMul`
   offload. They are not equivalent to CPU(Native). Naga may generate native shader files
   for more WGSL kernels than the dispatcher wires. Metal paths are Apple-only and load
   ignored generated MSL from `native/shaders/metal`.
7. **Alias consistency:** Some aliases are not equally wired across all tiers. For
   example, `Linear` and `Gemm` are handled by WebNN and CPU(Native), but browser
   CPU(JS)/WASM/WebGPU dispatch primarily expects `MatMul`.
