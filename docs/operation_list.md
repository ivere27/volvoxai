# Operation List and Status

This document describes the VolvoxAI operation list, what each operation means, where
its logic lives in the repository, and the current implementation status across browser,
Node, and native backends.

The machine-readable inventory and exporter qualification matrix is generated
from `proto/kernel_registry.proto` at
[`generated/kernel-registry.md`](generated/kernel-registry.md). That generated
matrix is authoritative for membership and route IDs; this page supplies the
operator semantics, limits, and implementation notes that do not fit in the
registry.

## What the Operation List Is

The operation list is the set of `opType` names that can appear in a
`volvox-graph/v1` document. A graph node uses one of these names, references input and output tensors, and
passes op-specific parameters through `params`.

The list is used by three parts of the system:

1. **Exporters and converters** decide which ONNX, TFLite, PyTorch, or custom graph
   operations can be emitted into Volvox format.
2. **Graph loading** resolves tensor shapes, weight layouts, quantized weights, and
   aliases before execution.
3. **Runtime backends** dispatch each node to the best available implementation.

The operation list is smaller than the complete ONNX or TFLite operator sets.
It is the runtime contract that VolvoxAI implements for inference and its
opt-in training paths. Forward support does not automatically imply backward
support; see the training section below.

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
| CPU(JS) | Pure JavaScript reference backend used by the browser/Node ES module runtime. | [CPUEngine.ts](../ts/backends/CPUEngine.ts), [ts/ops](../ts/ops) |
| WASM | Browser/Node WebAssembly tier. Direct C kernels cover every documented portable inference contract; unsupported inputs reject instead of changing backend tier. The full-WASM training sidecar has a separate strict C ABI. | [WasmEngine.ts](../ts/backends/WasmEngine.ts), [native kernels](../native/src/kernels) |
| WebGPU | Browser WebGPU compute-shader backend. | [GraphExecutor.ts](../ts/backends/GraphExecutor.ts), [shaders](../shaders) |
| WebNN | Opportunistic browser accelerator through `navigator.ml`. Unsupported ops throw during build so lower tiers can run. | [WebNNEngine.ts](../ts/backends/WebNNEngine.ts) |
| CPU(Native) | Freestanding native runtime CPU path. This is separate from CPU(JS). | [engine_runtime.c](../native/src/runtime/engine_runtime.c), [native kernels](../native/src/kernels), [conv_f32_opt.c](../native/src/kernels/conv_f32_opt.c), [quant_cpu_opt.c](../native/src/kernels/quant_cpu_opt.c) |
| Vulkan(Native) | Native Vulkan compute path. It is called from the native dispatcher and is separate from browser WebGPU. | [engine_runtime.c](../native/src/runtime/engine_runtime.c), [vulkan_engine.c](../native/src/backends/vulkan_engine.c) |
| OpenGL(Native) | Native desktop OpenGL/OpenGL ES compute path. It is called from the native dispatcher and is separate from browser WebGPU. | [engine_runtime.c](../native/src/runtime/engine_runtime.c), [opengl_engine.c](../native/src/backends/opengl_engine.c) |
| CUDA(Native) | Opt-in manual-PTX path. Inference is forward-only; the full profile adds the current native F32 backward/loss/optimizer and W8-authoring contracts. Explicit/required selection is strict. | [cuda.md](cuda.md), [cuda_engine.c](../native/src/backends/cuda_engine.c), [cuda_kernels.cu](../native/src/backends/cuda_kernels.cu), [cuda_training_kernels.cu](../native/src/backends/cuda_training_kernels.cu) |
| Metal(Native) | Native Metal compute path on Apple platforms. It loads Naga-generated MSL and is called from the native dispatcher for selected F32 and canonical packed-byte W8A8 graph ops. | [engine_runtime.c](../native/src/runtime/engine_runtime.c), [metal_engine.m](../native/src/backends/metal_engine.m) |

## Browser and Node Status

This table covers the ES module runtime: CPU(JS), WASM, browser WebGPU, and WebNN,
and is the release reference for documented canonical shapes and dtypes. The WASM
column below describes ordinary `WasmEngine` inference dispatch; strict full-WASM
training has separate C-backed dispatch, described in the training section.
A WASM entry marked `Full` invokes a C/WASM kernel on this ordinary inference
path rather than a CPU(JS) data loop.

| Operation | What it does | CPU(JS) | WASM | WebGPU | WebNN |
| --- | --- | --- | --- | --- | --- |
| `MatMul` | Dense matrix multiply with optional bias. INT8-packed weights use W8A32: FP32 activations and accumulation. | [Full, FP32/W8A32](../ts/ops/matMul.ts) | [Full, FP32/W8A32](../ts/backends/WasmEngine.ts) | [Full, canonical W8A32 I8/U8 weights](../shaders/inference/linearInt8.wgsl) | [Full, FP32](../ts/backends/WebNNEngine.ts) |
| `Linear`, `Gemm` | Dense-layer aliases used by exporters. | [Full, FP32/W8A32](../ts/backends/CPUEngine.ts) | [Full, FP32/W8A32](../ts/backends/WasmEngine.ts) | [Full, canonical W8A32 I8/U8 weights](../ts/backends/GraphExecutor.ts) | [Full, FP32](../ts/backends/WebNNEngine.ts) |
| `Conv2D` | NHWC 2D convolution, including grouped/depthwise and dilation. HWIO weights (HWCM when depthwise). | [Full](../ts/ops/conv2D.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/conv2D.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `Conv1D` | 1D convolution for sequence/audio tensors. NLC activations, WIO weights `[k, in_per_group, out_c]`. | [Full](../ts/ops/conv1D.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/conv1D.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `ConvTranspose2D` | Transposed convolution / deconvolution. NHWC activations, HWIO weights `[kh, kw, in_c, out_c]`. | [Full](../ts/ops/convTranspose2D.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/convTranspose2D.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `QConv2D` | Canonical W8A8 Conv2D: NHWC I8/U8 activation, OHWI I8/U8 weight with axis-0 per-channel metadata, optional I32 accumulator bias, groups/stride/padding/dilation, and fused ReLU/ReLU6. | [Full reference](../ts/ops/qConv2D.ts) | [Full, im2col plus packed SIMD128 for eligible groups=1 calls; portable C fallback](../ts/backends/WasmEngine.ts) | [Full, packed-byte baseline plus portable cooperative tile and feature-gated DP4a tile](../shaders/inference/qConv2DInt8Tiled.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `QLinear`, `QMatMul`, `QGemm` | Canonical W8A8 dense layer: `[...,d_in]` bytes, `[d_out,d_in]` per-axis weight, I32 bias, and typed output. | [Full reference](../ts/ops/qLinear.ts) | [Full, packed/raw baseline plus optional Relaxed-SIMD M=1 dot child](../ts/backends/WasmEngine.ts) | [Full, packed-byte baseline plus feature-gated scalar/tiled DP4a](../shaders/inference/qLinearInt8Dot.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `QBatchMatMul` | Canonical physical-byte ONNX MatMul: rank-2–8 `[...,M,K] @ [...,K,N]` with right-aligned batch broadcasting, independent per-tensor I8/U8 operands and output, and preflighted I32-safe centered accumulation. | [Full reference](../ts/ops/qBatchMatMul.ts) | [Full, standard SIMD128 across N with exact scalar fallback/tails](../ts/backends/WasmEngine.ts) | [Full implementation, packed-byte WGSL within portable dispatch bounds; physical execution requires a WebGPU adapter](../shaders/inference/qBatchMatMul.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `LayerNorm` | Layer normalization over the last feature axis. | [Full](../ts/ops/layerNorm.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/layerNorm.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `QLayerNorm` | Canonical W8A8 last-axis LayerNorm: same-shape rank-at-least-1 per-tensor I8/U8 input/output, F32 `[D]` gamma/beta, optional positive `eps` (default `1e-5`), and optional `d_model` matching D; row statistics are F32 scratch, not an F32 activation tensor. | [Full reference](../ts/ops/qLayerNorm.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, two-pass packed-byte stats/apply](../ts/backends/GraphExecutor.ts) | [Missing](../ts/backends/WebNNEngine.ts) |
| `RMSNorm` | RMS normalization over the last feature axis. | [Full](../ts/ops/rMSNorm.ts) | [Full, F32 rows](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/rMSNorm.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `RoPE` | Rotary position embedding with GPT-NeoX half-split or GPT-J interleaved pairs. | [Full, canonical F32](../ts/ops/roPE.ts) | [Full, portable C](../native/src/kernels/sequence_ops.c) | [Missing](../ts/backends/GraphExecutor.ts) | [Missing](../ts/backends/WebNNEngine.ts) |
| `SSMScan`, `SelectiveScan` | Mamba-style selective state-space recurrence with optional initial/final state, skip, and gate. | [Full, canonical F32](../ts/ops/ssmScan.ts) | [Full, portable C](../native/src/kernels/sequence_ops.c) | [Missing](../ts/backends/GraphExecutor.ts) | [Missing](../ts/backends/WebNNEngine.ts) |
| `BatchNorm2D` | Per-channel image batch normalization. | [Full](../ts/ops/batchNorm2D.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/batchNorm2D.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `GroupNorm` | NHWC group normalization with per-channel affine scale and bias. | [Full](../ts/ops/groupNorm.ts) | [Full, F32 NHWC](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/groupNorm.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `QGroupNorm` | Canonical W8A8 GroupNorm: rank-4 NHWC per-tensor I8/U8 input/output, F32 `[C]` gamma/beta, required `num_groups` dividing C, optional positive `eps` (default `1e-5`), and optional `data_layout: "NHWC"`; group statistics are F32 scratch, not an F32 activation tensor. | [Full reference](../ts/ops/qGroupNorm.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, two-pass packed-byte stats/apply](../ts/backends/GraphExecutor.ts) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Embedding` | I32 row lookup into an F32 embedding table. | [Full](../ts/ops/embedding.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/embedding.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `QEmbedding` | Canonical W8A8 lookup: I32 IDs `[S...]` covered by complete public-input, invariant-payload, or closed canonical producer-range preflight, I8/U8 `[vocab,hidden]` table with axis-0 row metadata, and caller-supplied per-tensor I8/U8 output descriptor `[S...,hidden]`. | [Full reference](../ts/ops/qEmbedding.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, packed-byte baseline](../shaders/inference/qEmbeddingInt8.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `QMaskedMean` | Canonical TaskRouter reduction: I8/U8 per-tensor `[B,S,D]` input, unquantized I32 `[B,S]` nonzero-keep mask, and I8/U8 per-tensor `[B,D]` output. It accumulates centered bytes in I32, then performs one private scalar requantization; an all-masked row emits the output zero point. | [Full reference](../ts/ops/qMaskedMean.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, packed-byte word-owner kernel](../shaders/inference/qMaskedMeanInt8.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `SDPA` | Self-attention over F32 rank-2/3 packed QKV with full/causal attention, optional keep mask, and train-only probability dropout. | [Full](../ts/ops/sDPA.ts) | [Full, F32 rank-2/3 and documented keep-mask layouts](../ts/backends/WasmEngine.ts) | [Partial, F32 rank-2/3 and `head_dim <= 64`](../shaders/inference/sDPA.wgsl) | [Partial](../ts/backends/WebNNEngine.ts) |
| `CrossSDPA` | Cross-attention over separate F32 rank-2/3 Q, K, and V tensors, optional keep mask, and train-only probability dropout. | [Full](../ts/ops/crossSDPA.ts) | [Full, F32 rank-2/3 and documented keep-mask layouts](../ts/backends/WasmEngine.ts) | [Partial, F32 rank-2/3 and `head_dim <= 64`](../shaders/inference/crossSDPA.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `QSDPA` | Canonical W8A8 self/cross attention: separate rank-2/3 I8/U8 Q/K/V and output descriptors, optional I32 keep mask, causal mode, and positive scale. Raw QK accumulation is I32; stable softmax/value accumulation is backend-private F32 scratch, never a graph activation. | [Full reference](../ts/ops/qSDPA.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, packed-byte one-query/head workgroups; `head_dim <= 64`](../shaders/inference/qSDPAInt8.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `CrossAttention` | F32 rank-2/3 cross-attention using Q, packed KV, a `[3*d_model,d_model]` projection, and optional scale/bias. | [Full](../ts/ops/crossAttention.ts) | [Full, portable F32 contract](../ts/backends/WasmEngine.ts) | [Partial, F32 with `d_model` and `head_dim <= 64`](../shaders/inference/crossAttentionF32.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `MoERouter` | Temperature-scaled top-k expert routing. | [Full](../ts/ops/moeRouter.ts) | [Full, canonical F32 routes](../ts/backends/WasmEngine.ts) | [Partial, F32 routes and `top_k <= 8`](../shaders/inference/moeRouter.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `MoELinear` | Execute and mix selected expert matrices. | [Full](../ts/ops/moeLinear.ts) | [Full, canonical F32 expert and route tensors](../ts/backends/WasmEngine.ts) | [Full, F32 expert and route tensors](../shaders/inference/moeLinear.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `MaxPool2D` | 2D max pooling. | [Full, including descriptor-preserving I8/U8](../ts/ops/maxPool2D.ts) | [Full, typed C path](../ts/backends/WasmEngine.ts) | [Full, packed-byte I8/U8 and F32 paths](../shaders/inference/maxPool2DTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `AveragePool2D` | 2D average pooling. | [Full](../ts/ops/averagePool2D.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/averagePool2D.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `GlobalAveragePool` | Global spatial average pooling. | [Full](../ts/ops/globalAveragePool.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/globalAveragePool.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Resize` | Image resize. Bilinear is the normal path; typed byte activation storage is nearest-only. | [Full](../ts/ops/resize.ts) | [Full, typed path is nearest-only](../ts/backends/WasmEngine.ts) | [Full, typed path is nearest-only](../shaders/inference/resizeNearestTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `ResizeNearest2D` | Nearest-neighbor image resize. | [Full, descriptor-preserving I8/U8](../ts/ops/resize.ts) | [Full, typed NHWC C path](../ts/backends/WasmEngine.ts) | [Full, packed-byte I8/U8 and F32 paths](../shaders/inference/resizeNearestTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `UpsampleNearest2D` | Nearest-neighbor 2x upsampling. | [Full](../ts/ops/upsample2x.ts) | [Full, F32 NHWC](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/upsample2x.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Interpolate1D` | Linear 1D interpolation. | [Full](../ts/ops/interp1D.ts) | [Full, F32 batched channel-length](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/interp1D.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `ReLU` | Rectified linear activation. | [Full](../ts/ops/reLU.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/reLU.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `LeakyReLU` | Leaky ReLU activation with `alpha`. | [Full](../ts/ops/leakyReLU.ts) | [Full, F32](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/leakyReLU.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `PReLU` | Per-channel parametric ReLU. | [Full](../ts/ops/pReLU.ts) | [Full, F32 generic slope repetition](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/pReLU.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `GELU` | Erf GELU by default; optional `approximate: "tanh"`. | [Full](../ts/ops/gELU.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/gELU.wgsl) | [Exact mode](../ts/backends/WebNNEngine.ts) |
| `QGELU` | Canonical W8A8 GELU: a distinct same-shape per-tensor I8/U8 input/output pair; only omitted parameters or `approximate: "none"`; uses the fixed Abramowitz–Stegun 7.1.26 erf polynomial and ties-to-even saturating requantization. | [Full reference](../ts/ops/qGELU.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, packed-byte fixed-erf baseline](../shaders/inference/qGELUInt8.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `SiLU` | `x * sigmoid(x)` activation. | [Full](../ts/ops/siLU.ts) | [Full, F32](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/siLU.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `QSiLU` | Canonical W8A8 SiLU: one distinct same-shape per-tensor I8/U8 input/output pair, no auxiliary inputs or parameters; dequantize, apply `x / (1 + exp(-x))`, and ties-to-even saturating requantize directly into output bytes. | [Full reference](../ts/ops/qSiLU.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, packed-byte baseline](../shaders/inference/qSiLUInt8.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Sigmoid` | Logistic activation. | [Full](../ts/ops/sigmoid.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/sigmoid.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `HardSwish` | MobileNet-style hard swish activation. | [Full](../ts/ops/hardSwish.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/hardSwish.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `HardSigmoid` | Piecewise-linear sigmoid approximation. | [Full](../ts/ops/hardSigmoid.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/hardSigmoid.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Tanh` | Hyperbolic tangent activation. | [Full](../ts/ops/tanh.ts) | [Full, F32](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/tanh.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Sin`, `Cos` | Elementwise trigonometric functions over same-shape F32 tensors. | [Full](../ts/ops/sin.ts) | [Full, portable C](../native/src/kernels/sequence_ops.c) | [Missing](../ts/backends/GraphExecutor.ts) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Clip` | Clamp values to min/max. | [Full](../ts/ops/clip.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/clip.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Add` | Elementwise F32 add with right-aligned N-D broadcasting. | [Full](../ts/ops/add.ts) | [Full, F32 rank-1–8 broadcast](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/broadcastBinary.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `QAdd` | Exact-shape W8A8 add with per-tensor descriptors, requantization, and optional fused ReLU/ReLU6. | [Full reference](../ts/ops/qAdd.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, packed-byte baseline](../shaders/inference/qAdd.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Mul` | Elementwise multiply with right-aligned N-D broadcasting. | [Full](../ts/ops/mul.ts) | [Full, F32 rank-1–8 broadcast](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/broadcastBinary.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `Sub` | Elementwise subtract with broadcasting. | [Full](../ts/ops/sub.ts) | [Full, F32 rank-1–8 broadcast](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/broadcastBinary.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Div` | Elementwise divide with broadcasting. | [Full](../ts/ops/div.ts) | [Full, F32 rank-1–8 broadcast](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/broadcastBinary.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Softmax` | Softmax over the last axis. | [Full](../ts/ops/softmax.ts) | [Full, F32 rows](../ts/backends/WasmEngine.ts) | [Partial, last axis only](../shaders/inference/softmax.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `LogSoftmax` | Log-softmax over the last axis. | [Full](../ts/ops/logSoftmax.ts) | [Full, F32 rows](../ts/backends/WasmEngine.ts) | [Partial, last axis only](../shaders/inference/logSoftmax.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `ReduceSum` | Sum the last axis, preserving one value per outer row. | [Partial, last axis](../ts/ops/reduceSum.ts) | [Partial, rank-2 direct](../ts/backends/WasmEngine.ts) | [Partial, last axis](../shaders/inference/reduce.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `ReduceMean` | Mean of the last axis, preserving one value per outer row. | [Partial, last axis](../ts/ops/reduceMean.ts) | [Partial, rank-2 direct](../ts/backends/WasmEngine.ts) | [Partial, last axis](../shaders/inference/reduce.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `ArgMax` | Inference-only index of the maximum value; the axis may be removed or retained as size 1. | [Full](../ts/ops/argMax.ts) | [Full, typed F32/I32/I8/U8 input and output](../ts/backends/WasmEngine.ts) | [Partial, F32 input; F32/I32/I8/U8 output](../ts/backends/GraphExecutor.ts) | [Missing](../ts/backends/WebNNEngine.ts) |
| `QArgMax` | Canonical terminal byte-domain ArgMax: rank-2–8 per-tensor I8/U8 input, exact integer axis, first-tie semantics, and unquantized I32 output with that axis removed. Positive affine quantization preserves raw-byte order, so logits stay byte-typed. | [Full reference](../ts/ops/qArgMax.ts) | [Full, portable C](../ts/backends/WasmEngine.ts) | [Full, packed-byte I32-index kernel](../shaders/inference/qArgMaxInt8.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Transpose` | General N-D tensor permutation. | [Full](../ts/ops/transpose.ts) | [Full, F32 rank-1–8](../ts/backends/WasmEngine.ts) | [Full](../shaders/inference/generalTranspose.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Concat`, `Concat2` | Tensor concatenation. Raw byte concat requires identical dtype and immutable descriptor; fused sigmoid requires an explicit F32 boundary. | [Full](../ts/ops/concat2.ts) | [Full, typed raw concat and F32 rank-1–8](../ts/backends/WasmEngine.ts) | [Full, typed and F32/I32](../shaders/inference/concatCopyTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Split` | Split a tensor into multiple outputs. | [Full](../ts/ops/split.ts) | [Full, F32 equal-sized output slices](../ts/backends/WasmEngine.ts) | [Partial, equal-sized output slices](../shaders/inference/split.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Slice` | Canonical F32 rank-1–8 strided slice with normalized axes/starts, positive integer steps, and an in-bounds selection. | [Full](../ts/ops/slice.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full, canonical F32 rank-1–8 positive-step contract](../shaders/inference/sliceNd.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Pad` | Constant padding. | [Full](../ts/ops/pad.ts) | [Full](../ts/backends/WasmEngine.ts) | [Partial, image/4D-oriented metadata](../shaders/inference/pad.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Expand`, `Broadcast` | Canonical F32/I32 right-aligned broadcast plus descriptor-preserving per-tensor I8/U8 `Expand`, to a valid rank-1–8 target shape. | [Full](../ts/ops/expand.ts) | [Full, F32/I32/I8/U8 rank-1–8](../ts/backends/WasmEngine.ts) | [Full, F32/I32 plus packed descriptor-preserving I8/U8](../shaders/inference/expandTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Gather` | Canonical F32 gather with I32 indices, rank at most 8, any valid axis, and ONNX negative-index normalization. | [Full](../ts/ops/gather.ts) | [Full](../ts/backends/WasmEngine.ts) | [Full, F32 data/output and I32 indices](../shaders/inference/gatherInt32.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `GatherElements` | Canonical F32 elementwise indexed gather with I32 indices, rank at most 8, and normalized negative indices. | [Full](../ts/ops/gatherElements.ts) | [Full, F32 data/output and I32 indices](../ts/backends/WasmEngine.ts) | [Full, F32 data/output and I32 indices](../shaders/inference/gatherElements.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Where`, `Mask` | Exact-shape F32 selection with an F32 or I32 condition; `mask`, `cond`, and `condition` are aliases. | [Full](../ts/ops/where.ts) | [Full, exact-shape portable contract](../ts/backends/WasmEngine.ts) | [Full, exact-shape F32 operands/output and F32 or I32 condition](../shaders/inference/whereTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Cast` | Dtype conversion among F32/I32/I8/U8; F32-to-I32 truncates toward zero, wraps modulo 2^32, and maps NaN/infinities to zero; only F32-to-F32 has a backward. | [Full](../ts/ops/cast.ts) | [Full, F32/I32/I8/U8](../ts/backends/WasmEngine.ts) | [Full, F32/I32/I8/U8](../shaders/inference/cast.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `DequantizeLinear` | Convert F32/I32/I8/U8 values to F32 with scalar F32 scale and an optional scalar typed zero point. | [Full](../ts/ops/dequantizeLinear.ts) | [Full, scalar typed contract](../ts/backends/WasmEngine.ts) | [Full, scalar typed contract](../shaders/inference/dequantizeLinearTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `QuantizeLinear` | Canonical F32-to-I8/U8 boundary with scalar F32 scale, matching typed zero point, nearest-even rounding, saturation, and a required central per-tensor output descriptor whose numeric parameters live in safetensors. | [Full](../ts/ops/quantizeLinear.ts) | [Full, typed portable C kernel](../ts/backends/WasmEngine.ts) | [Full, packed-byte typed dispatch](../shaders/inference/quantizeLinearTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `RequantizeLinear` | I8/U8-to-I8/U8 conversion between central per-tensor descriptors, with nearest-even rounding and saturation. | [Full](../ts/ops/requantizeLinear.ts) | [Full, typed portable C kernel](../ts/backends/WasmEngine.ts) | [Full, packed-byte typed dispatch](../shaders/inference/requantizeLinearTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `NonMaxSuppression` | Inference-only greedy NMS for object-detection boxes. | [Full](../ts/ops/nonMaxSuppression.ts) | [Full, typed boxes/scores/output](../ts/backends/WasmEngine.ts) | [Shader/direct implementation only; public bounded-domain compilation does not advertise it until a complete bounded output-cardinality proof exists](../shaders/inference/nonMaxSuppression.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Reshape` | Shape-only tensor view/copy. Typed storage requires an unchanged descriptor. | [Alias](../ts/ops/reshape.ts) | [Alias, typed C copy](../ts/backends/WasmEngine.ts) | [Alias, packed-byte copy](../shaders/inference/copyTyped.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `Flatten` | Shape-only flatten. Typed storage requires an unchanged descriptor. | [Alias](../ts/ops/reshape.ts) | [Alias, typed C copy](../ts/backends/WasmEngine.ts) | [Alias, packed-byte copy](../shaders/inference/copyTyped.wgsl) | [Full](../ts/backends/WebNNEngine.ts) |
| `Squeeze` | Remove size-1 dimensions. Typed storage requires an unchanged descriptor. | [Alias](../ts/ops/reshape.ts) | [Alias, typed C copy](../ts/backends/WasmEngine.ts) | [Alias, packed-byte copy](../shaders/inference/copyTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Unsqueeze` | Add size-1 dimensions. Typed storage requires an unchanged descriptor. | [Alias](../ts/ops/reshape.ts) | [Alias, typed C copy](../ts/backends/WasmEngine.ts) | [Alias, packed-byte copy](../shaders/inference/copyTyped.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Dropout` | Deterministic inverted Dropout while training; exact identity during inference. | [Full training / alias inference](../ts/ops/dropout.ts) | [Alias inference](../ts/backends/WasmEngine.ts) | [Full training / alias inference](../shaders/training/dropout.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `Identity` | Pass-through copy. | [Alias](../ts/ops/reshape.ts) | [Alias](../ts/backends/WasmEngine.ts) | [Alias](../shaders/inference/copy.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `SpatialSoftargmaxY` | F32 NHWC vertical soft-argmax for vision postprocessing; output `[N,C,W]`. | [Full, F32 NHWC](../ts/ops/spatialSoftargmaxY.ts) | [Full, F32 NHWC](../ts/backends/WasmEngine.ts) | [Full, F32 NHWC](../shaders/inference/spatialSoftargmaxY.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `ProfileX` | F32 NHWC horizontal profile/reduction; output `[N,2C,W]`. | [Full, F32 NHWC](../ts/ops/profileX.ts) | [Full, F32 NHWC](../ts/backends/WasmEngine.ts) | [Full, F32 NHWC](../shaders/inference/profileX.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `ProfileY` | F32 NHWC vertical profile/reduction; output `[N,2C,H]`. | [Full, F32 NHWC](../ts/ops/profileY.ts) | [Full, F32 NHWC](../ts/backends/WasmEngine.ts) | [Full, F32 NHWC](../shaders/inference/profileY.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |
| `MeanHeight` | F32 NHWC height mean; output `[N,C,W]`. | [Full, F32 NHWC](../ts/ops/meanHeight.ts) | [Full, F32 NHWC](../ts/backends/WasmEngine.ts) | [Full, F32 NHWC](../shaders/inference/meanHeight.wgsl) | [Missing](../ts/backends/WebNNEngine.ts) |

### Sequence-model operator contracts

`RoPE` accepts an F32 `input` (or `x`)
with shape `[S,D]` or `[B,S,D]` and produce a same-shape F32 `out`. The
optional I32 `position_ids` input is `[S]` or `[B,S]`; without it, position is
`position_offset + sequence_index`. `rotary_dim` defaults to `D` and must be a
positive even integer no larger than `D`; dimensions after it are copied.
`theta` defaults to `10000` and must be finite and positive.
`interleaved: false` (the default) rotates GPT-NeoX half-split pairs
`(i, i + rotary_dim/2)`. `interleaved: true` rotates GPT-J adjacent pairs
`(2i, 2i+1)`. Position IDs and `position_offset` must be non-negative.

`SSMScan` and its exact alias `SelectiveScan` are a Mamba-style selective
scan, not the generic ONNX `Scan` control-flow operator. All storage is F32.
`input` (or `u`) and `delta` are `[S,D]` or `[B,S,D]`; `A` is `[D,N]`;
each of `B` and `C` independently broadcasts from `[N]`, `[S,N]`, or
`[B,S,N]`. Optional `D` is `[D]`, optional `z` matches the input, and optional
`initial_state` is `[B,D,N]` (including explicit `B=1` for rank-two input).
The required `out` matches the input and optional `state`/`final_state` emits
the final `[B,D,N]` state. For every batch, time, channel, and state lane:

```text
dt       = delta_softplus ? softplus(delta) : delta
state    = exp(dt * A) * state + dt * B * input
out      = sum(state * C) + D * input
out      = z ? out * silu(z) : out
```

`delta_softplus` is boolean and defaults to `true`. The portable kernel keeps
only `[B,D,N]` F32 scratch. It does not materialize a per-time state tensor.
An exporter must not spell this operation as `Scan`: VolvoxAI deliberately
reserves that name for a possible future generic body-graph recurrence.

`Sin` and `Cos` are same-shape F32 unary operations and use the platform's
IEEE/libm behavior, including NaN results for infinite inputs. The sequence
operators in this section currently have forward implementations on CPU(JS),
C/WASM, and CPU(Native); browser and native GPU backends fall back to CPU or
decline them as described in their tables.

## Quantized Execution

`W8A32` means INT8/U8 weights remain packed, while activations, accumulation,
and output are FP32. That is the browser/Node `MatMul`/`Linear` path today: it
can reduce weight storage and bandwidth, but it is not a full integer graph.

A canonical W8A8 edge is explicit: its tensor has physical `int8` or `uint8`
storage and a central `volvox-affine-safetensors/v1` descriptor. The descriptor
contains only immutable scale/zero-point tensor names; all numeric values live
in safetensors. Activations normally use `per_tensor`; weights may use
`per_axis` along the operator's required output dimension. See
[the model format](model-format.md#safetensors-backed-affine-quantization).
`QuantizeLinear` and `DequantizeLinear` operands must exactly match the central
references. Shape-preserving operators may carry an unchanged descriptor,
while an operation that changes it must explicitly requantize.

“W8A8” here means physical byte storage for quantized activation/weight edges,
not integer-only arithmetic. `QEmbedding` starts from I32 IDs;
`QSiLU`/`QGELU` have exact byte-to-byte transforms and requantization;
`QGroupNorm` and
`QLayerNorm` also have F32 affine inputs plus backend-private F32 statistics
scratch, without materializing an F32 activation tensor.

The browser/Node CPU, WASM, WebGPU, and native CPU execute the canonical W8A8
typed activation island directly: `QuantizeLinear`, `RequantizeLinear`, `QConv2D`,
`QLinear`/`QMatMul`/`QGemm`, `QBatchMatMul`, `QEmbedding`, `QAdd`, `QSiLU`, `QGELU`, `QGroupNorm`, `QLayerNorm`, `QMaskedMean`, `QSDPA`, max pool, nearest resize,
descriptor-preserving shape copies/concat, `QArgMax`, and `DequantizeLinear`. `QConv2D`
is NHWC/OHWI; `QLinear` is `[d_out,d_in]`; both use I32 bias and I32
accumulation. `QBatchMatMul` accepts rank-2–8 matrix operands, applies
right-aligned ONNX broadcasting only to their batch axes, and requires
independent per-tensor I8/U8 descriptors for both operands and the output. Its
worst-case centered dot product is rejected unless it fits I32.
The fixed WASM parent selects a standard SIMD128 kernel that accumulates four
adjacent N columns in I32 lanes. N tails and matrices narrower than four
columns use the same scalar implementation; both paths share staged-F32
requantization, ties-to-even rounding, and I8/U8 saturation.
`QEmbedding` has no byte activation input. Its I32 IDs require one complete
fail-closed range proof before any output write: full public-input preflight,
exact validation of an invariant payload, or a closed static proof through
canonical producers. The static proof admits bounded I32 `Clip`,
`ArgMax`/`QArgMax`, `Equal`/`GreaterOrEqual`/`Not`, safe byte-or-I32 `Cast`, and
value-preserving shape chains. Arbitrary integer graphs and application metadata
are not substitutes. The table is `[vocab,hidden]` with axis-0 metadata, and the
graph must supply the output's per-tensor descriptor.
`QAdd` deliberately has no broadcast form—an exporter
must lower broadcast to descriptor-preserving byte `Expand` plus exact-shape
`QAdd` before it reaches the typed island. `QSiLU`
has one typed activation input and no scale/zero-point tensor inputs: both
immutable descriptors are carried by its input and output tensors. `QGELU`
has the same byte-edge rule but accepts only omitted parameters or
`approximate: "none"` and uses the fixed A–S polynomial. `QGroupNorm` is
rank-4 NHWC with F32 `[C]` gamma/beta, required `num_groups`, and optional
positive `eps`; activations remain I8/U8 while the kernel uses compact
per-group F32 statistics and affine arithmetic. `QLayerNorm` applies the same
policy to every final-axis `[D]` row, with optional `d_model` only as a strict
check of D. Their centered raw-domain statistics are stable, but parallel GPU
reduction order can change the final F32 value near a requantization half-step;
`QSDPA` likewise uses only private F32 online-softmax/value accumulators.
`QMaskedMean` uses an I32 raw-byte reduction plus a private scalar
requantization, while `QArgMax` compares the ordered raw bytes and emits an I32 token index rather
than materializing F32 logits.
CPU and GPU are therefore not promised bit-identical for those boundary cases.

WebGPU detects the optional `packed_4x8_integer_dot_product` WGSL language
feature before compiling DP4a variants. The dense variants cover all canonical
I8/U8 type and zero-point combinations, including odd reduction tails. The
cooperative convolution variant is used for sufficiently large groups=1 layers
with output channels divisible by four; other convolution descriptors retain
the scalar word-owner kernel. Both variants preserve the same I32 accumulator,
bias, requantization, and fused-activation contract as the portable reference.
On the repository's AMD GCN-5 WebGPU host, the byte-exact harness measured the
DP4a decoder-row QLinear at 0.186 ms versus 0.803 ms portable (4.32x), the tiled
multi-row QLinear at 0.249 ms versus 0.345 ms portable (1.39x), and tiled QConv
at 0.480 ms versus 1.950 ms scalar portable (4.06x). The feature-free portable
QConv tile measured 0.760 ms, 2.57x faster than scalar and 1.59x slower than
DP4a. These are isolated kernel timings, not end-to-end VQA latency.

The end-to-end Chrome harness for the same AMD GCN-5 host loads the real 23 MB
TinyReceipt package and `00002.jpg`. For the 80-token `phone number last one`
case, per-token WebGPU mapping measured 2,444.5 ms hot; feeding QArgMax IDs
on-device and checking EOS every 16 tokens measured 1,920.2 ms hot (21.5%
faster). The device-feedback answer exactly matched the ordinary WebGPU row
path. It did not match WASM/native on this sample, so these timings establish
feedback-loop parity and speed only, not cross-backend accuracy qualification.

The graph loader treats descriptor-bearing I8/U8 activation edges as a
fail-closed contract. They may use only the operations above, or cross an
explicit quantize/dequantize boundary; a generic F32 operation cannot silently
reinterpret raw bytes. This does not change ordinary W8A32 `MatMul`/`Conv2D`:
their packed quantized weights remain valid with F32 activation/output edges.

Native CPU also accepts a fail-closed weight-only convolution spelling as
`QConv2D` with `params.weight_only: true`. It requires F32 rank-4 NHWC input
and output, an immutable rank-4 I8/U8 OHWI model weight, an immutable positive
F32 scale vector of length one or `O`, an optional immutable I8/U8/I32 zero
point vector of length one or `O`, and an optional immutable finite F32 bias
of length `O`. Group count, stride, dilation, padding, channel grouping, and
output geometry are validated when the model loads. This weight-only path does
not create an activation `QTensor` sidecar and is CPU-only; native GPU
`QConv2D` routes continue to mean canonical physical-byte W8A8. Every other
native `QConv2D` must declare physical I8/U8 input and output descriptors. The
loader rejects an untyped activation graph instead of inferring quantization
parameters from neighboring nodes.

CPU(Native) stores canonical W8A8 activations directly in physical I8/U8
buffers, without a sidecar or FP32 materialization.
For C `QSiLU` and `QGELU`, a call of at least 512 elements builds a thread-local
256-entry table keyed by the exact input/output descriptors; subsequent calls,
including smaller decoder rows, map raw bytes through that table. Small cold
calls stay scalar. Every table byte is produced by the same transform and
ties-to-even saturation as the scalar route, and exhaustive cold/build/warm
tests cover all I8/U8 input/output pairings.

The activation table lives in the portable C inference kernels, so it ships in
both `volvoxai.wasm` and `volvoxai.full.wasm`; WASM tests exercise all 256 input
bytes across cold, build, and warm reuse. In contrast, the hashed native graph
lookup, cached incremental plan/direct dispatch, and threaded `QLinear`,
`QConv2D`, and `QSDPA` routes are native-runtime-only. WASM `QLinear` dispatch
remains Relaxed-SIMD first for eligible `M=1` calls, baseline SIMD128 packed
second, and the canonical raw portable C kernel last. The baseline packed
kernel reuses each pair of packed K rows across a small M tile, accumulates
with the core `i32x4.dot_i16x8_s` instruction, and vectorizes ties-to-even
requantization; scalar K/N tails preserve the canonical result.

Eligible groups=1 WASM `QConv2D` calls with immutable weights, I32 bias, no
fused activation, and at least eight outputs use a reusable im2col buffer
followed by that packed kernel. The internal bridge repeats canonical QConv's
descriptor-wide I32 overflow proof, fills padding with the raw activation zero
point, caps scratch at 64 MiB, and falls back to canonical portable QConv
otherwise.

Vulkan(Native), OpenGL(Native), and Metal(Native) now dispatch the same
canonical subset as packed-byte graph kernels, keeping consecutive typed nodes
device-resident. The GPU route is intentionally limited to ordinary forward
execution (not native prefill/decode); a dynamic scale or zero point produced
by an earlier node is an explicit CPU synchronization boundary. Canonical W8A8
uses explicitly typed graph tensors and Q operators.

CUDA(Native) dispatches a broader manual-kernel F32/W8A8 forward allowlist and
also supports an eligible device-resident incremental row closure. Unlike the
ordered Vulkan/OpenGL/Metal registry route, an explicitly selected CUDA node
does not fall back to CPU. Its exact operation and shape restrictions are
maintained in [cuda.md](cuda.md#status).

The direct TFLite exporter emits this subset for its supported operators. Its
Logistic lowering is deliberately `DequantizeLinear → Sigmoid → QuantizeLinear`,
so that graph is hybrid W8A8/F32 rather than all-integer arithmetic.

### TinyReceipt split INT8 deployment

TinyReceipt uses one bounded-active encoder graph and one bounded-active
one-token decoder graph. The supported application pipeline imports either the
FP32 or producer-authored static INT8 variant from the cache-enabled split ONNX
directory. There is no second application PTQ or calibration pass. The complete
commands and qualification rules are in the
[TinyReceipt example](../examples/tiny_receipt_vqa/README.md).

The encoder binds the exact question extent `Q` and derives memory extent
`M=Q+210`. Each decoder call binds one current token, explicit cross K/V, and
self-attention past K/V with `R=P+1`. `TinyReceiptSplitSession` and
`tiny_receipt_split_w8a8` require the explicit-KV package-v1 manifest and begin
with the qualified blocked zero `P=1` sentinel.

```bash
examples/target/bin/tiny_receipt_split_w8a8 \
  build/tiny-receipt-kv-int8 \
  --image receipt.png --prompt "What is the phone number?" \
  --family phone --cpu --threads 1
```

The package records grayscale, bilinear `672x320`, `[-1,1]` F32 NCHW input,
canonical byte-fallback BPE tokenization, mask semantics, and exact graph asset
hashes. Provider qualification must preserve selected family, greedy token IDs,
structured text, and task results while forbidding backend fallback.

### W8A8 benchmark harness

Run `npx tsx tools/benchmark_w8a8.mjs --dry-run --json` to inspect the deterministic
operator workloads, or omit `--dry-run` to time the direct CPU(JS) reference
kernels. It compares QLinear W8A8 with the existing W8A32 and FP32 paths, and
QConv2D W8A8 with FP32 (there is no corresponding CPU(JS) W8A32 Conv2D
reference). The report includes raw typed-buffer footprint as well as warmup
and iteration timings. It is deliberately not a WASM, WebGPU, native CPU, or
native GPU performance claim.

For native CPU hot-path evidence, run `make benchmark_native`. It first
requires byte-identical portable/dispatched W8A8 results and validates the
packed W8A32 path. It then times the exact 51-QLinear call mix in one
TinyReceipt incremental decoder row using shared I8 weights and scales, as well
as exact `M=402`/`M=192` full-seed dense shapes, exact TinyReceipt activation
shapes, and a representative QConv2D W8A8 workload. Run
`make benchmark_native` for the exact encoder-self
`Q=K=402,D=320,H=8`, causal decoder-self `Q=K=192,D=320,H=8`, decoder-cross
`Q=192,K=402,D=320,H=8`, and incremental cross `Q=1` shapes. Both commands
verify byte equality before timing. These are matched kernel proxies, not an
end-to-end W8A32 model package or accuracy comparison.

Run `make test_wasm_relaxed_simd` for the WASM M=1 kernel check. In addition to
feature and parity validation, it times W8A8 Relaxed SIMD, packed baseline
W8A8, portable W8A8, and packed W8A32 for a shared `320x320` decoder-sized
row. The result is kernel-only and must not be read as whole-model latency.
The symmetric signed-I8 W8A32 row now uses compensated baseline-SIMD `f32x4`
accumulation when `K<=1280`, weight scales are at most `0.025`, and every
finite activation is in `[-35,35]`. Nonzero zero points, U8 weights, larger K,
larger scales, and out-of-envelope inputs use the scalar/double reference
fallback. Because F32 is not bit-exact to double, the fast path has direct
non-dyadic and maximum-envelope cancellation tests.

Run `make benchmark_wasm_w8a8_seed` for byte-checked TinyReceipt seed proxies.
It times the exact grayscale stem and the exact weighted `M=402`/`M=192`
base-dense call mix. Eligible convolution is reported as im2col layout plus
packed SIMD128 GEMM, with canonical portable QConv and an independent
JavaScript implementation as parity oracles. This benchmark does not use the
Relaxed-SIMD child because every measured seed matrix has `M>1`.

Run `make benchmark_wasm_qbatch_matmul` for deterministic mixed-I8/U8
QBatchMatMul proxies covering a `320`-wide decoder row, a multi-row encoder
matrix, and odd K/N tails. The harness first requires byte equality between
the exported scalar and standard-SIMD128 kernels, then reports their separate
times. It is a kernel benchmark, not an end-to-end model latency claim.

The browser/Node reference and WASM measurements below were taken on an AMD
Ryzen 5 5600U (AVX2; no AVX-VNNI). CPU(JS) QLinear measured 2.069 ms W8A8,
0.962 ms W8A32, and 1.067 ms FP32, with raw live-buffer footprints of 70,656,
84,992, and 279,552 bytes. CPU(JS) QConv2D measured 3.811 ms W8A8 versus
6.871 ms FP32 (17,024 versus 67,712 live bytes; no CPU(JS) W8A32 Conv
reference exists). In the fixed Clang-17 release artifact, the symmetric-I8
`320x320` row measured 0.0130 ms with packed SIMD128 versus 0.0228 ms with the
Relaxed-SIMD child. Runtime dispatch therefore tries the faster packed SIMD128
path first and keeps the ABI-compatible child as a fail-closed row fallback.
A five-pair real-model child-first/baseline ablation showed no separable
application-level difference. A multi-row Relaxed-SIMD prototype was about 2x
slower than packed SIMD128 and was not enabled. On the seed harness, the exact
grayscale stem measured 152.244 ms canonical versus 4.877 ms im2col+packed
(31.21x). The zero-point-zero symmetric fast path reduced the exact six-shape
weighted packed proxy from about 303.0 to 236.4 ms (22%); all compared output
bytes matched.

Current native kernel measurements are host-specific:

| Host and command | Workload | Established result |
| --- | --- | --- |
| AMD Ryzen 5 5600U, `make benchmark_native` | 51-call incremental `M=1` dense proxy | W8A8 selected raw/packed policy 0.696 ms; all-packed W8A8 0.777 ms; packed W8A32 3.581 ms (5.14x vs selected W8A8) |
| AMD Ryzen 5 5600U, Clang 17 native benchmark | Weighted `M=402`/`M=192` base-dense seed subset | exact signed-absolute K4/N16 W8A8 23.043 ms; raw SIMD 1-thread 230.136 ms; raw SIMD 4-thread 69.218 ms (3.00x vs raw 4-thread) |
| AMD Ryzen 5 5600U, Clang 17 native benchmark | Exact grayscale stem `[1,320,672,1] -> [1,160,336,48]` | portable 106.539 ms; persistent-pack im2col 1-thread 3.126 ms; output-row/strip-copy im2col 4-thread 1.084 ms (98.28x vs portable) |
| AMD Ryzen 5 5600U, `make benchmark_native` | `16x16`, 64-channel, 3x3 QConv proxy | portable 2.571 ms; native 1-thread 0.455 ms; native 4-thread 0.153 ms (16.84x vs portable) |
| AMD Ryzen 5 5600U, `make benchmark_native` | Exact activation shapes | QSiLU stem 23.31x; QGELU encoder 32.80x; decoder seed 29.74x; warm decoder row 31.28x; cold 160-element router 0.98x |
| AMD Ryzen 5 5600U, `make benchmark_native` | Encoder self / decoder self / decoder cross | 46.731→6.432 ms (7.27x) / 6.308→0.770 ms (8.19x) / 22.667→3.019 ms (7.51x), portable→4-thread |
| AMD Ryzen 5 5600U, `make benchmark_native` | Incremental cross `Q=1` | 0.119→0.053 ms (2.23x); no pool dispatch |
| Intel Core i3-1115G4, 2 cores/4 threads, AVX-512 VNNI | 51-call incremental `M=1` dense proxy | W8A8 0.547 ms; packed W8A32 2.896 ms (5.30x) |
| Intel Core i3-1115G4, 2 cores/4 threads, AVX-512 VNNI | Weighted seed dense subset | row-at-a-time 112.921 ms; four-row tiled 62.051 ms (1.82x) |
| Intel Core i3-1115G4, 2 cores/4 threads, AVX-512 VNNI | Exact grayscale stem | portable 97.235 ms; native 1-thread 23.431 ms; 4-thread 9.839 ms |

The current TinyReceipt explicit-KV v1 measurements and their exact workload,
affinity, hashes, and qualification limits are recorded in
[the BPE1536 benchmark](tiny-receipt-vqa-bpe1536-benchmark.md) and its tracked
reports. Those runs use ordinary bounded encoder/decoder calls with explicit
cache tensors; native GPU execution does not use a GPU-to-CPU row handoff.

For the separate Pillow-RGB-to-JavaScript preprocessed `00002.jpg` input, the
fixed-artifact same-context benchmark measures 2,168.763 ms cold and
1,596.581 ms hot. The hot seed is 1,356.306 ms and later rows take 2.996
ms/token. Against the immediately preceding packed kernel, the symmetric-I8
fast path reduced hot total by 21.5%, hot seed by 24.5%, and cold total by
20.1%. Atomic graph assembly reduced the cold family load from 1,611 ms to
about 169 ms, and the example's read-only SafeTensors cache fetches the
23,891,120-byte model exactly once. The 80-token output is identical between
cold and hot runs.
Native stb JPEG decoding is not byte-identical to Pillow decoding on this
sample; when both backends receive the same preprocessed tensor, native, WASM,
and CPU(JS) agree. These timings therefore measure execution speed, not answer
accuracy across different preprocessors.

### Native CPU W8A8 SIMD status

The portable C kernels remain the authoritative W8A8 implementation and the
fallback for every CPU. On GNU/Clang x86, canonical `QLinear` first uses a
target-attributed AVX-512 VNNI ZMM `VPDPBUSD` specialization for `d_in >= 64`.
Its runtime gate requires AVX2, AVX-512F/BW/VL/VNNI, and OS-enabled XMM, YMM,
opmask, and complete ZMM state. Hosts without that envelope next try AVX-VNNI
`VPDPBUSD` when CPUID leaf 7, subleaf 1, EAX bit 4 is present and `d_in >= 32`.
Both paths remap every I8/U8 input and weight pairing into the instruction's
U8 × I8 domain and apply exact zero-point compensation, so asymmetric mappings
remain correct. `QConv2D` uses the corresponding ZMM path for contiguous
NHWC/OHWI input groups of at least 64 channels and the YMM path for groups of
at least 32, counting only valid padded taps in its dot, sums, and correction.
Otherwise, eligible `QLinear` with `d_in >= 32` uses an exact AVX2
`VPMADDUBSW` path. Because that instruction saturates each adjacent pair to
I16, the kernel splits every unsigned activation byte into `min(a,127)` and
`a-min(a,127)`, evaluates the two non-saturating products separately, and
combines them only after widening to I32. This preserves arbitrary I8/U8 zero
points without the usual PMADDUBSW saturation error. Smaller eligible inputs
use target-attributed AVX2 widening and `VPMADDWD`. Seed-sized PMADDUBSW and
VNNI calls reuse each weight vector across four activation rows.

The persistent symmetric-I8 K4/N16 pack additionally records whether the
weight contains -128. If it does not, the AVX2 microkernel maps activations to
signed bytes and evaluates `abs(input) * sign(weight,input)`. The pair bound is
`2*128*127 = 32512`, so `VPMADDUBSW` remains exact with one product instead of
the two-part unsigned decomposition. Affine zero-point compensation is folded
into the initial I32 accumulator. A pack containing any -128 weight retains
the general split path, including its exact signed endpoint behavior.

Native `QBatchMatMul` uses a separate target-attributed AVX2 kernel for its
`[M,K] @ [K,N]` layout. It interleaves two K rows over sixteen N columns, then
uses the same split `VPMADDUBSW` proof and affine zero-point compensation as
QLinear. Scalar N tails and ineligible calls retain the canonical kernel.

For QConv2D input groups of at least 16 channels, the direct AVX2 path
traverses an activation block once for four output channels. Dense groups=1
3x3 calls with at least 16 output channels can instead expand zero-point-padded
NHWC patches and reuse the same exact QLinear PMADDUBSW hierarchy. The native
model prepares a persistent flattened K4/N8 companion pack for symmetric I8
weights, including narrow-input layers such as the grayscale stem; asymmetric
weights retain the raw/direct hierarchy. Dilation-one im2col partitions
batch/output rows and copies each fully in-bounds 3*C strip contiguously,
without changing zero-point padding at borders. Allocation, alias, or
shape-policy failures return to the direct or portable kernel. All paths retain
scalar tails, grouped/padded convolution geometry, and a conservative
I32-prefix-overflow proof before changing reduction order.

Canonical ARM `QLinear`, `QConv2D`, and `QBatchMatMul` have NEON widening
paths. Native Linux AArch64 and Android AArch64 builds also add separately
compiled `SDOT` specializations, each gated by `HWCAP_ASIMDDP`: `QLinear` for eligible 16-byte
reductions and `QConv2D` for eligible contiguous input groups of at least 16
channels. These kernels operate directly on the declared physical-byte tensors.

On eligible x86 calls above roughly one million products, native `QLinear` and
`QBatchMatMul` partition independent rows and `QConv2D` partitions independent
output locations through the shared pool. Alias-sensitive calls remain
ordered and serial. Whole-tensor native `QSDPA` similarly partitions independent
batch/query rows once `B*Q*K*D` reaches 524,288 products and there are enough
rows for the configured workers. On AVX2, each row also vectorizes the exact
centered-byte QK dot and the F32 value accumulation while retaining the
portable online-softmax key order. `QLinear M=1`, `QSDPA Q=1`, and other small
calls remain on the caller, so the steady incremental path does not wake the
pool.

Native `QSiLU` partitions only large contiguous byte ranges. `QLayerNorm`
partitions large row sets, while its runtime-gated AVX2 path preserves scalar
mean/variance order and vectorizes the independent affine/requantization pass.
`QGroupNorm` partitions independent groups and, on AVX2, evaluates four groups
in parallel SIMD lanes without changing the reduction order inside any group.
The thresholds keep decoder-sized prefixes serial; portable scalar execution
remains the fallback on other CPUs.

Native releases default to a baseline CPU target rather than global
`-mavx2 -mfma`. The canonical x86 dispatcher keeps AVX2, AVX-VNNI, and AVX-512
VNNI in target-attributed functions behind runtime capability checks. Linux
and Android AArch64 do the same compositionally: baseline NEON code and
separately compiled `+dotprod` objects coexist in one binary, and the SDOT
objects are entered only after the HWCAP check. `make verify_native_isa`
verifies both that the optimized x86 instructions are present and that public
baseline dispatchers contain no unguarded YMM/ZMM instructions. Build-time
AVX2 kernels remain available through `NATIVE_CPU_TARGET=avx2` for deployments
that guarantee that ISA.

## Native Backend Status

This table separates CPU(Native), Vulkan(Native), OpenGL(Native), and
Metal(Native). CUDA is deliberately not duplicated as a fifth fast-changing
column: [cuda.md](cuda.md#status) is its canonical forward and
full-profile training matrix and restrictions record.
For a bounded graph, Vulkan, OpenGL, and Metal accept only their qualified
operator subset after proving every legal shape, launch, slot, scratch, and
resident-resource bound. Context creation then reserves the maximum host/device
domain before execution; concrete shapes rebind within that fixed capacity.
An allow-fallback policy records a rejected accelerator node as an explicit CPU
operator route. A require-tier, forbid-operator-fallback policy accepts only an
attested all-device route and rejects the graph before execution otherwise.

Native shader generation is separate from native runtime support. The tracked
`shaders/{inference,training}/*.wgsl` files are translated by `make compile_shaders` / Naga into ignored
build outputs under `native/shaders/{spv,glsl,gles,metal}`. The Rust generator passes
the original WGSL binding numbers through Naga's structured GLSL/MSL resource maps;
generated source is not patched afterward. An op is marked supported
for Vulkan/OpenGL only when `engine_runtime.c` has a dispatcher path and the native
backend has a C wrapper for that shader. Metal follows the same rule on Apple builds:
the MSL may be generated for many ops, but a row is marked supported only when
`engine_runtime.c` dispatches to a `metal_graph_*` wrapper.

| Operation | CPU(Native) | Vulkan(Native) | OpenGL(Native) | Metal(Native) |
| --- | --- | --- | --- | --- |
| `MatMul`, `Gemm` | [Full FP32/W8A32; physical W8A8 uses QMatMul/QGemm](../native/src/runtime/engine_runtime.c) | [Partial, F32 graph path with both declared dense layouts plus the large static one-shot path; not exporter-qualified for public bounded dynamic](../native/src/backends/vulkan_engine.c) | [Partial, F32 graph path with both declared dense layouts plus the large static one-shot path; not exporter-qualified for public bounded dynamic](../native/src/backends/opengl_engine.c) | [Partial, Apple-only F32 graph path; not exporter-qualified for public bounded dynamic](../native/src/backends/metal_engine.m) |
| `Linear` | [Full FP32/W8A32; physical W8A8 uses QLinear](../native/src/runtime/engine_runtime.c) | [Partial, bounded F32 graph path with both declared weight layouts and optional invariant bias; dynamic F16 is rejected](../native/src/backends/vulkan_engine.c) | [Partial, bounded F32 graph path with both declared weight layouts and optional invariant bias; dynamic F16 is rejected](../native/src/backends/opengl_engine.c) | [Partial, Apple-only bounded F32 graph path with both declared weight layouts and optional invariant bias; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `BatchMatMul` | [Full, rank-2–8 F32 with right-aligned ONNX batch broadcasting](../native/src/runtime/engine_runtime.c) | [Partial, bounded F32 graph path with 8x8 dispatch-domain proof](../native/src/backends/vulkan_engine.c) | [Partial, bounded F32 graph path with 8x8 dispatch-domain proof](../native/src/backends/opengl_engine.c) | [Partial, Apple-only bounded F32 graph path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `QLinear`, `QMatMul`, `QGemm` | [Full, canonical physical-byte W8A8 with immutable cached I32 bias; large multi-row x86 pool path; B=1 decoder row stays serial](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte bounded-forward path with bias-inclusive I32 accumulator proof; no row decode](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte bounded-forward path with bias-inclusive I32 accumulator proof; no row decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte bounded-forward path with bias-inclusive I32 accumulator proof; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `QBatchMatMul` | [Full, canonical rank-2–8 physical-byte W8A8 with right-aligned ONNX batch broadcasting, I32 overflow preflight, runtime-gated AVX2/NEON N-column kernels, and large-M row pooling](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path; requires an available Vulkan device; no row decode](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path; requires an available OpenGL compute device; no row decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Conv2D` | [Full](../native/src/kernels/conv_f32_opt.c) | [Partial, bounded batched F32 NHWC path with canonical HWIO/HWCM weight; public F32 weight remains mutable](../native/src/backends/vulkan_engine.c) | [Partial, bounded batched F32 NHWC path with canonical HWIO/HWCM weight; public F32 weight remains mutable](../native/src/backends/opengl_engine.c) | [Partial, Apple-only bounded batched F32 NHWC path with canonical HWIO/HWCM weight; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `QConv2D` | [Full, canonical physical-byte NHWC/OHWI W8A8 with immutable optional I32 bias and large-call x86 output-location pool path; CPU-only fail-closed F32-activation weight-only path via `params.weight_only: true`; all other untyped forms rejected](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte bounded-forward path with per-channel bias-inclusive I32 accumulator proof](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte bounded-forward path with per-channel bias-inclusive I32 accumulator proof](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte bounded-forward path with per-channel bias-inclusive I32 accumulator proof; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Conv1D` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, batched F32 channel-length graph path](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 channel-length graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only batched F32 channel-length graph path](../native/src/backends/metal_engine.m) |
| `ConvTranspose2D` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `LayerNorm` | [Full](../native/src/kernels/layernorm.c) | [Partial, F32 with weight+bias graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 with weight+bias graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only F32 with weight+bias graph path](../native/src/backends/metal_engine.m) |
| `QLayerNorm` | [Full, canonical final-axis physical-byte W8A8 with F32 `[D]` affine; pooled large-row execution and runtime-gated AVX2 affine/requantization; serial decoder-row path](../native/src/runtime/engine_runtime.c) | [Partial, two-pass packed-byte ordinary-forward path; no row decode](../native/src/backends/vulkan_engine.c) | [Partial, two-pass packed-byte ordinary-forward path; no row decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only two-pass packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `RMSNorm` | [Full](../native/src/kernels/math_nlp.c) | [Partial, F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `RoPE` | [Full, canonical F32 runtime](../native/src/runtime/sequence_runtime.c), [portable kernel](../native/src/kernels/sequence_ops.c) | [Missing](../native/src/backends/vulkan_engine.c) | [Missing](../native/src/backends/opengl_engine.c) | [Missing](../native/src/backends/metal_engine.m) |
| `SSMScan`, `SelectiveScan` | [Full, canonical F32 runtime](../native/src/runtime/sequence_runtime.c), [portable kernel](../native/src/kernels/sequence_ops.c) | [Missing](../native/src/backends/vulkan_engine.c) | [Missing](../native/src/backends/opengl_engine.c) | [Missing](../native/src/backends/metal_engine.m) |
| `BatchNorm2D` | [Full](../native/src/kernels/edge_primitives.c) | [Partial, F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `GroupNorm` | [Partial, F32 NHWC](../native/src/runtime/engine_runtime.c) | [Partial, F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only F32 NHWC graph path](../native/src/backends/metal_engine.m) |
| `QGroupNorm` | [Full, canonical NHWC physical-byte W8A8 with F32 `[C]` affine; pooled groups and runtime-gated AVX2 four-group SIMD](../native/src/runtime/engine_runtime.c) | [Partial, two-pass packed-byte ordinary-forward path; no prefill/decode](../native/src/backends/vulkan_engine.c) | [Partial, two-pass packed-byte ordinary-forward path; no prefill/decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only two-pass packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Embedding` | [Full, I32 token IDs](../native/src/kernels/embedding.c) | [Partial, I32 IDs/F32 table graph path](../native/src/backends/vulkan_engine.c) | [Partial, I32 IDs/F32 table graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only I32 IDs/F32 table graph path](../native/src/backends/metal_engine.m) |
| `QEmbedding` | [Full, canonical physical-byte W8A8; B=1 CPU decoder-row path](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path; no row decode](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path; no row decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `QMaskedMean` | [Full, canonical I32 centered-byte reduction to physical-byte output](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path; no prefill/decode](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path; no prefill/decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `SDPA` | [Full, full/causal masks and decode KV cache](../native/src/kernels/sdpa.c) | [Partial, batched F32 full/causal masked graph path, head_dim <= 64](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 full/causal masked graph path, head_dim <= 64](../native/src/backends/opengl_engine.c) | [Partial, Apple-only batched F32 full/causal masked graph path, head_dim <= 64](../native/src/backends/metal_engine.m) |
| `CrossSDPA` | [Full, optional keep mask](../native/src/kernels/cross_sdpa.c) | [Partial, batched F32 masked graph path, head_dim <= 64](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 masked graph path, head_dim <= 64](../native/src/backends/opengl_engine.c) | [Partial, Apple-only batched F32 masked graph path, head_dim <= 64](../native/src/backends/metal_engine.m) |
| `QSDPA` | [Full, canonical physical-byte W8A8 with private F32 softmax/value scratch, large whole-tensor batch/query pool path, and B=1 CPU row/KV reuse without pool dispatch](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path; no row decode](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path; no row decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `CrossAttention` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, batched F32 Q/KV/weight graph path, d_model/head_dim <= 64](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 Q/KV/weight graph path, d_model/head_dim <= 64](../native/src/backends/opengl_engine.c) | [Partial, Apple-only batched F32 Q/KV/weight graph path, d_model/head_dim <= 64](../native/src/backends/metal_engine.m) |
| `MoERouter` | [Full, F32/I32 route indices](../native/src/runtime/engine_runtime.c) | [Missing](../native/src/backends/vulkan_engine.c) | [Missing](../native/src/backends/opengl_engine.c) | [Missing](../native/src/backends/metal_engine.m) |
| `MoELinear` | [Full](../native/src/runtime/engine_runtime.c) | [Missing](../native/src/backends/vulkan_engine.c) | [Missing](../native/src/backends/opengl_engine.c) | [Missing](../native/src/backends/metal_engine.m) |
| `MaxPool2D` | [Full, F32 plus canonical typed NHWC I8/U8 with unit dilation/no ceil mode](../native/src/runtime/engine_runtime.c) | [Partial, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Partial, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte typed ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `AveragePool2D` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `GlobalAveragePool` | [Full](../native/src/kernels/edge_primitives.c) | [Partial, F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `Resize` | [Partial, canonical typed nearest/asymmetric/floor path; generic F32 Resize remains unavailable](../native/src/runtime/engine_runtime.c) | [Partial, F32 plus packed-byte typed nearest ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Partial, F32 plus packed-byte typed nearest ordinary-forward path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte typed nearest ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `ResizeNearest2D` | [Full, F32 plus canonical typed nearest NHWC path](../native/src/runtime/engine_runtime.c) | [Partial, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Partial, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte typed ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `UpsampleNearest2D` | [Full](../native/src/kernels/vision_ops.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `Interpolate1D` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, F32 2D channels-by-length graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 2D channels-by-length graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `ReLU` | [Full](../native/src/kernels/activations.c) | [Partial, same-size F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `LeakyReLU` | [Full](../native/src/kernels/math_nlp.c) | [Partial, same-size F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `PReLU` | [Full, scalar or last-channel slope](../native/src/runtime/engine_runtime.c) | [Partial, F32 last-channel graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 last-channel graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `GELU` | [Full, erf default and optional tanh](../native/src/kernels/activations.c) | [Partial, same-size F32 erf/tanh graph path](../native/src/backends/vulkan_engine.c) | [Partial, same-size F32 erf/tanh graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only same-size F32 erf/tanh graph path](../native/src/backends/metal_engine.m) |
| `SiLU` | [Full](../native/src/kernels/math_nlp.c) | [Partial, same-size F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only same-size F32 graph path](../native/src/backends/metal_engine.m) |
| `QSiLU` | [Full, canonical same-shape parameter-free physical-byte W8A8; exact 256-byte portable-C LUT with warm reuse and large-range CPU pooling](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path; no prefill/decode](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path; no prefill/decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `QGELU` | [Full, canonical same-shape physical-byte W8A8; fixed A–S 7.1.26 erf polynomial, `approximate: "none"` only; exact 256-byte portable-C LUT with warm row reuse](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path; no row decode](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path; no row decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Sigmoid` | [Full](../native/src/kernels/edge_primitives.c) | [Partial, same-size F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `HardSwish` | [Full](../native/src/kernels/edge_primitives.c) | [Partial, same-size F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `HardSigmoid` | [Full](../native/src/kernels/edge_primitives.c) | [Partial, same-size F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `Tanh` | [Full](../native/src/kernels/math_nlp.c) | [Partial, same-size F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, same-size F32 graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `Sin`, `Cos` | [Full, same-shape F32 runtime](../native/src/runtime/sequence_runtime.c), [portable kernel](../native/src/kernels/sequence_ops.c) | [Missing](../native/src/backends/vulkan_engine.c) | [Missing](../native/src/backends/opengl_engine.c) | [Missing](../native/src/backends/metal_engine.m) |
| `Clip` | [Full, same-shape F32/I32](../native/src/runtime/engine_runtime.c) | [Partial, same-shape F32/I32 graph paths](../native/src/backends/vulkan_engine.c) | [Partial, same-shape F32/I32 graph paths](../native/src/backends/opengl_engine.c) | [Partial, Apple-only same-shape I32 graph path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Add` | [Full, right-aligned N-D broadcast](../native/src/runtime/engine_runtime.c) | [Partial, right-aligned F32 N-D broadcast](../native/src/backends/vulkan_engine.c) | [Partial, right-aligned F32 N-D broadcast plus Add3 fusion](../native/src/backends/opengl_engine.c) | [Partial, Apple-only right-aligned F32 N-D broadcast](../native/src/backends/metal_engine.m) |
| `QAdd` | [Full, canonical exact-shape physical-byte W8A8; B=1 CPU decoder-row path](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Mul` | [Full, right-aligned N-D broadcast](../native/src/runtime/engine_runtime.c) | [Partial, right-aligned F32 N-D broadcast](../native/src/backends/vulkan_engine.c) | [Partial, right-aligned F32 N-D broadcast](../native/src/backends/opengl_engine.c) | [Partial, Apple-only right-aligned F32 N-D broadcast](../native/src/backends/metal_engine.m) |
| `Sub` | [Full, right-aligned F32 N-D broadcast](../native/src/runtime/engine_runtime.c) | [Partial, right-aligned F32 N-D broadcast](../native/src/backends/vulkan_engine.c) | [Partial, right-aligned F32 N-D broadcast](../native/src/backends/opengl_engine.c) | [Partial, Apple-only right-aligned F32 N-D broadcast](../native/src/backends/metal_engine.m) |
| `Div` | [Full, right-aligned F32 N-D broadcast](../native/src/runtime/engine_runtime.c) | [Partial, right-aligned F32 N-D broadcast](../native/src/backends/vulkan_engine.c) | [Partial, right-aligned F32 N-D broadcast](../native/src/backends/opengl_engine.c) | [Partial, Apple-only right-aligned F32 N-D broadcast](../native/src/backends/metal_engine.m) |
| `Softmax` | [Full](../native/src/kernels/math_nlp.c) | [Partial, last-axis F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, last-axis F32 graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only last-axis F32 graph path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `LogSoftmax` | [Full, last axis](../native/src/runtime/engine_runtime.c) | [Partial, last-axis F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, last-axis F32 graph path](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `ReduceSum` | [Partial, last-axis F32](../native/src/runtime/engine_runtime.c) | [Partial, last-axis F32](../native/src/backends/vulkan_engine.c) | [Partial, last-axis F32](../native/src/backends/opengl_engine.c) | [Partial, Apple-only last-axis F32](../native/src/backends/metal_engine.m) |
| `ReduceMean` | [Partial, last-axis F32](../native/src/runtime/engine_runtime.c) | [Partial, last-axis F32](../native/src/backends/vulkan_engine.c) | [Partial, last-axis F32](../native/src/backends/opengl_engine.c) | [Partial, Apple-only last-axis F32](../native/src/backends/metal_engine.m) |
| `ArgMax` | [Full, F32 input to I32 index with first-tie semantics; B=1 CPU decoder-row path; nonzero `select_last_index` rejected](../native/src/runtime/engine_runtime.c) | [Partial, bounded F32-to-I32 graph path; `keepdims` 0/1 and first-tie only](../native/src/backends/vulkan_engine.c) | [Partial, bounded F32-to-I32 graph path; `keepdims` 0/1 and first-tie only](../native/src/backends/opengl_engine.c) | [Partial, Apple-only bounded F32-to-I32 graph path; `keepdims` 0/1, first-tie only, runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `QArgMax` | [Full, canonical byte-domain I8/U8 to I32 index; B=1 CPU decoder-row path](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path; no row decode](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path; no row decode](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Equal`, `GreaterOrEqual` | [Full, I32 right-aligned N-D broadcast](../native/src/runtime/engine_runtime.c) | [Partial, bounded I32 right-aligned N-D broadcast](../native/src/backends/vulkan_engine.c) | [Partial, bounded I32 right-aligned N-D broadcast](../native/src/backends/opengl_engine.c) | [Partial, Apple-only bounded I32 right-aligned N-D broadcast; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Not` | [Full, same-shape I32](../native/src/runtime/engine_runtime.c) | [Partial, bounded same-shape I32](../native/src/backends/vulkan_engine.c) | [Partial, bounded same-shape I32](../native/src/backends/opengl_engine.c) | [Partial, Apple-only bounded same-shape I32; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Transpose` | [Full](../native/src/kernels/tensor_f32_opt.c) | [Partial, F32 graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only F32 graph path](../native/src/backends/metal_engine.m) |
| `Concat` | [Full, F32 plus canonical same-domain physical-byte concat](../native/src/runtime/engine_runtime.c) | [Partial, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Partial, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte typed ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Concat2` | [Missing](../native/src/runtime/engine_runtime.c) | [Missing](../native/src/backends/vulkan_engine.c) | [Missing](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `Split` | [Full, multi-output axis slices](../native/src/runtime/engine_runtime.c) | [Partial, F32 multi-output axis slices](../native/src/backends/vulkan_engine.c) | [Partial, F32 multi-output axis slices](../native/src/backends/opengl_engine.c) | [Partial, Apple-only F32 multi-output axis slices](../native/src/backends/metal_engine.m) |
| `Slice` | [Full, same-dtype F32/I32 rank-1–8 positive-step slice](../native/src/runtime/engine_runtime.c) | [Partial, F32 up to 4D with positive steps](../native/src/backends/vulkan_engine.c) | [Partial, F32 up to 4D with positive steps](../native/src/backends/opengl_engine.c) | [Partial, Apple-only F32 up to 4D with positive steps; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Pad` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, F32 up to 4D top/left constant pad](../native/src/backends/vulkan_engine.c) | [Partial, F32 up to 4D top/left constant pad](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `Expand`, `Broadcast` | [Full, F32/I32 plus descriptor-preserving rank-1–8 I8/U8 `Expand`](../native/src/runtime/engine_runtime.c) | [Partial, bounded rank-1–8 F32/I32 graph path; byte `Expand` is not qualified](../native/src/backends/vulkan_engine.c) | [Partial, bounded rank-1–8 F32/I32 graph path; byte `Expand` is not qualified](../native/src/backends/opengl_engine.c) | [Partial, Apple-only bounded rank-1–8 F32/I32 graph path; byte `Expand` is not qualified](../native/src/backends/metal_engine.m) |
| `Gather` | [Full, F32 data/output with I32 indices, any axis through rank eight, and normalized negative indices](../native/src/runtime/engine_runtime.c) | [Partial, F32 axis-0 graph path with I32 indices](../native/src/backends/vulkan_engine.c) | [Partial, F32 axis-0 graph path with I32 indices](../native/src/backends/opengl_engine.c) | [Partial, Apple-only F32 axis-0 graph path with I32 indices; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `GatherElements` | [Missing](../native/src/runtime/engine_runtime.c) | [Missing](../native/src/backends/vulkan_engine.c) | [Missing](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `Where` | [Full, exact-shape I32 condition with F32/I32 values and output](../native/src/runtime/engine_runtime.c) | [Partial, exact-shape F32 values with F32/I32 condition, or I32 values with I32 condition](../native/src/backends/vulkan_engine.c) | [Partial, exact-shape F32 values with F32/I32 condition, or I32 values with I32 condition](../native/src/backends/opengl_engine.c) | [Partial, Apple-only exact-shape I32 condition with F32/I32 values; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Mask` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, exact-shape F32 condition/values; not exporter-qualified for public bounded dynamic](../native/src/backends/vulkan_engine.c) | [Partial, exact-shape F32 condition/values; not exporter-qualified for public bounded dynamic](../native/src/backends/opengl_engine.c) | [Init only](../native/src/backends/metal_engine.m) |
| `Cast` | [Full, F32/I32/I8/U8 conversions](../native/src/runtime/engine_runtime.c) | [Partial, bounded F32/I32 conversions](../native/src/backends/vulkan_engine.c) | [Partial, bounded F32/I32 conversions](../native/src/backends/opengl_engine.c) | [Partial, Apple-only bounded F32/I32 conversions; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `DequantizeLinear` | [Full, canonical I8/U8-to-F32 boundary](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path with static/graph-input scalar metadata](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path with static/graph-input scalar metadata](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `QuantizeLinear` | [Full, canonical F32-to-I8/U8 boundary](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path with static/graph-input scalar metadata](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path with static/graph-input scalar metadata](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `RequantizeLinear` | [Full, canonical physical-byte domain conversion](../native/src/runtime/engine_runtime.c) | [Partial, packed-byte ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Partial, packed-byte ordinary-forward path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only packed-byte ordinary-forward path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `NonMaxSuppression` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, F32 sequential greedy graph path](../native/src/backends/vulkan_engine.c) | [Partial, F32 sequential greedy graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only F32 sequential greedy graph path](../native/src/backends/metal_engine.m) |
| `Reshape` | [Alias, canonical typed copy when descriptors match](../native/src/runtime/engine_runtime.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/opengl_engine.c) | [Alias/copy, Apple-only typed path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Flatten` | [Alias, canonical typed copy when descriptors match](../native/src/runtime/engine_runtime.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/opengl_engine.c) | [Alias/copy, Apple-only typed path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Squeeze` | [Alias, canonical typed copy when descriptors match](../native/src/runtime/engine_runtime.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/opengl_engine.c) | [Alias/copy, Apple-only typed path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Unsqueeze` | [Alias, canonical typed copy when descriptors match](../native/src/runtime/engine_runtime.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/opengl_engine.c) | [Alias/copy, Apple-only typed path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `Dropout` | [Full training / alias inference](../native/src/runtime/engine_runtime.c) | [Full F32 training / alias inference](../native/src/backends/vulkan_engine.c) | [Full F32 training / alias inference](../native/src/backends/opengl_engine.c) | [Full Apple-only F32 training / alias inference](../native/src/backends/metal_engine.m) |
| `Identity` | [Alias, canonical typed copy when descriptors match](../native/src/runtime/engine_runtime.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/vulkan_engine.c) | [Alias/copy, F32 plus packed-byte typed ordinary-forward path](../native/src/backends/opengl_engine.c) | [Alias/copy, Apple-only typed path; runtime unverified on Linux](../native/src/backends/metal_engine.m) |
| `SpatialSoftargmaxY` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only batched F32 NHWC graph path](../native/src/backends/metal_engine.m) |
| `ProfileX` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only batched F32 NHWC graph path](../native/src/backends/metal_engine.m) |
| `ProfileY` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only batched F32 NHWC graph path](../native/src/backends/metal_engine.m) |
| `MeanHeight` | [Missing](../native/src/runtime/engine_runtime.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/vulkan_engine.c) | [Partial, batched F32 NHWC graph path](../native/src/backends/opengl_engine.c) | [Partial, Apple-only batched F32 NHWC graph path](../native/src/backends/metal_engine.m) |

## Training and backward status

Training is a separate full-profile path. Normal inference does not allocate
gradient buffers, optimizer state, random masks, or backward pipelines. The
public inference header exposes inference Vx* handles only. The full-only
`volvoxai_full.h` exposes an opaque `VxTrainer`, and the model-neutral
`native/volvoxai-full train` command is a client of that lifecycle. JavaScript
training is likewise owned by a retained Trainer created from a
`Model`.

Strict `backend: "wasm"` training in `volvoxai.full.wasm` accepts the
differentiable complete contracts documented here and rejects partial or
non-canonical contracts before model state changes. JavaScript CPU and browser
WebGPU implement the same portable contracts. `ArgMax` and
`NonMaxSuppression`, `RoPE`, `SSMScan`/`SelectiveScan`,
`Sin`, and `Cos` are intentionally forward-only. This browser/Node portability
gate does not change native coverage below. CUDA is intentionally kept out of
this four-column summary because its fast-changing full-profile F32 backward,
loss, optimizer, and W8-authoring contract is maintained in
[cuda.md](cuda.md#backward-operator-coverage). Its inference profile
remains free of all training code and PTX.

| Capability | JavaScript CPU | Browser WebGPU | Native CPU | Vulkan / OpenGL / Metal training |
| --- | --- | --- | --- | --- |
| NHWC `GroupNorm` | Forward and input/affine backward | Forward and input/affine WGSL backward | Forward and input/affine backward | Forward and input/affine GPU backward for supported F32 layouts |
| Standalone `Dropout` | Deterministic inverted forward/backward | Deterministic inverted WGSL forward/backward | Deterministic inverted forward/backward | Deterministic inverted GPU forward/backward |
| `SDPA` / `CrossSDPA` probability dropout | Exact training-only mask | Exact training-only mask | Deterministic after-softmax training mask | Dropout-aware forward and backward, with CPU fallback if GPU dispatch is unavailable |
| `Add` / `Mul` broadcasting | Right-aligned N-D forward and reduced backward | Right-aligned N-D forward and reduced WGSL backward | Right-aligned N-D forward and reduced backward | Right-aligned F32 forward and reduced GPU backward, up to rank 8 |
| `ReduceSum` / `ReduceMean` | Last-axis forward/backward | Last-axis WGSL forward/backward | Last-axis forward/backward | Last-axis F32 GPU forward/backward |
| Weighted CE losses and accumulation | `losses`, per-loss metrics, explicit normalizers, reset/flush | Same contract | Full-only opaque `VxTrainer` | Same Trainer contract; loss seeding and the optimizer remain coordinated by its private owner |

For JavaScript CPU, strict full-WASM, and WebGPU, attention probability `dropout` (or
`attention_dropout`) is applied after softmax with inverted scaling and no
renormalization. Forward and backward regenerate the same deterministic
`[B,H,Q,K]` mask. Inference remains ordinary deterministic attention. Native
CPU, Vulkan, OpenGL compute, Metal, and full-profile CUDA implement the same
after-softmax training semantics and regenerate their mask during backward.
Vulkan/OpenGL/Metal dispatch may fall back to the matching CPU forward/backward
path when its training shader or layout is unavailable; required-CUDA training
instead preflights the complete plan and fails closed. Standalone `Dropout` is
fully separate and supported on all of them. Determinism is per backend and
training state; the JS and native seed APIs are not a promise of bit-identical
masks to each other.

Multiple loss entries can target different or repeated logits tensors and are
summed after their independent weights and normalizers are applied. Explicit
full-window normalizers make gradient accumulation exact across heterogeneous
microbatches. SGD/AdamW updates use one global norm over the complete trainable
set when `maxGradNorm`/`max_grad_norm` is nonzero, and an optimizer step advances
only when the accumulation window is applied.

### Metal forward and backward coverage

The Apple-only Metal graph dispatcher now has explicit F32 forward paths for
MatMul/Linear, Conv2D/Conv1D, Embedding, SDPA/CrossSDPA/CrossAttention,
LayerNorm/GroupNorm, GELU/SiLU, right-aligned Add/Mul/Sub/Div, last-axis
reductions, Transpose, Concat/Split, standalone Dropout, shape aliases, and the
listed quantization and vision/postprocessing operations. Unsupported forward
nodes still use the native dispatcher contract rather than becoming an implied
Metal implementation.

Metal's lazy backward shader allowlist includes dense and Conv2D gradients;
LayerNorm, RMSNorm, BatchNorm2D, and GroupNorm gradients; embeddings;
SDPA/CrossSDPA including attention-probability dropout; activations, PReLU, and
softmax; broadcast Add/Mul; pooling and resize; MoE router/expert gradients;
concat/split/transpose/copy; standalone Dropout; and last-axis reductions.
Layouts outside an allowlist select the complete native CPU backward path during
preflight. Metal runtime validation requires macOS and an Apple GPU.

## Known Gaps

1. **WebGPU contract limits:** Canonical `Gather` uses I32 indices and
   supports arbitrary valid axes through rank eight. `Softmax` and
   `LogSoftmax` remain last-axis operations, `Split` uses equal-sized output
   slices, and `Pad` remains constant NHWC/4D-oriented. See the Browser and
   Node Status table above for rank, dtype, and shape contracts.
2. **WebGPU compilation:** Unsupported nodes reject compilation with a typed
   error. Every successful ExecutionResult contains a stable snapshot for all
   declared outputs, including input, weight, and transitive alias outputs.
3. **WebNN coverage:** WebNN is intentionally narrow. It can run common dense,
   activation, embedding, conv, reshape, layer norm, and decomposed SDPA graphs, but
   most vision postprocessing and shape ops are not mapped yet.
4. **Canonical W8A8 scope:** Browser/Node CPU, WASM, WebGPU, native CPU, and
   the native GPU graph routes accept the documented canonical subset. It is
   not a universal INT8-op set: unsupported raw I8/U8 activations fail closed
   until an explicit typed kernel is added. The direct TFLite exporter also
   retains an explicit F32 Sigmoid boundary. TinyReceiptVQA now has a
   development-calibrated materializer plus native and browser/Node example
   router/family sessions. It remains fixed B=1. The application executes each
   one-token decoder call through ordinary bounded context execution and owns
   cross K/V, past/present self K/V, and mask progression explicitly on every
   selected backend. `P/R` advance one token per call; backend-private tactics
   may reuse proved fixed capacity without changing that application ABI.
   Production-wide accuracy qualification remains open work.
5. **Native GPU coverage:** Vulkan(Native), OpenGL(Native), and Metal(Native)
   dispatch the canonical packed-byte subset during bounded graph forward
   execution. The explicit-KV TinyReceipt encoder and decoder remain on the
   selected device while their concrete dimensions rebind within the reserved
   domain. The calibrated TinyReceipt package has been exercised on Vulkan
   (RADV Renoir) and OpenGL (Mesa Renoir) here. Metal is Apple-only and remains
   runtime-unverified on Linux. OpenGL
   selects the portable cooperative QConv tile for eligible layers, while
   Vulkan can select packed integer-dot SPIR-V only when the device reports an
   accelerated signed 4x8-bit dot product; all other descriptors retain their
   portable packed-byte kernels. CUDA has separate broad, strict forward
   coverage and RTX 3090 validation documented in [cuda.md](cuda.md); it does
   not use the CPU-row handoff or per-node CPU fallback.
6. **Training coverage:** JavaScript CPU, strict full-WASM, and browser WebGPU
   implement the portable matrix's differentiable complete rows. Native CPU and
   Vulkan/OpenGL/Metal retain separate explicit backward allowlists; an
   unsupported native GPU operation either selects the complete native CPU
   backward path during preflight or fails, never silently stops a gradient.
   Batched SDPA/CrossSDPA are not restricted to batch one. WebGPU portable
   attention requires `head_dim <= 64` (`CrossAttention` F32 also requires
   `d_model <= 64`), and WebGPU MoE routing requires `top_k <= 8`. Exact
   after-softmax attention-probability dropout is supported by JavaScript CPU,
   strict full-WASM, browser WebGPU, native CPU, Vulkan, OpenGL compute, and
   Metal training. Full-profile CUDA implements the current native F32
   backward contract plus device loss, accumulation, clipping, SGD, and AdamW;
   its staged-attention implementation, split-axis checks, combined batch-one
   query/output-length-192 accumulation-24 TinyVQA update/checkpoint/resume
   evidence, and remaining residency/performance work are tracked in
   [cuda.md](cuda.md). Metal is Apple-only and cannot be
   runtime-tested on Linux.
   See [model construction, routing, and training](model_builder_training.md) for
   the API and current constraints.
