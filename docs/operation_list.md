# Operation List and Status

This document describes the VolvoxAI operation list, what each operation means, where
its logic lives in the repository, and the current implementation status across browser,
Node, and native backends.

Use the operator descriptions below to understand a graph and find its kernels.
For the current backend registrations, consult the
[backend matrix](generated/kernel-registry.md#operator-matrix); for tested
shape/dtype combinations, use the [validation map](c-runtime-validation.md).
Registration identifies an available route, while compilation decides whether
that route can execute your complete graph. Source links are reading aids;
a shader's existence alone does not establish runtime support.

## What the Operation List Is

The operation list is the set of `opType` names that can appear in a
`volvox-graph/v1` document. A graph node uses one of these names, references input and output tensors, and
passes op-specific parameters through `params`.

The list is used by three parts of the system:

1. **Exporters and converters** decide which ONNX, TFLite, PyTorch, or custom graph
   operations can be emitted into Volvox format.
2. **Graph loading** resolves tensor shapes, weight layouts, quantized weights, and
   aliases before execution.
3. **Runtime backends** dispatch each node through the qualified implementation
   fixed by the selected compiled route.

The operation list is smaller than the complete ONNX or TFLite operator sets.
It is the runtime contract that VolvoxAI implements for inference and its
opt-in training paths. Forward support does not automatically imply backward
support; see the training section below.

## How to read support information

An operator may have a CPU kernel, a GPU shader, and additional restrictions on
shapes, layouts, or values. Forward inference support also differs from backward
training support. Use the current registry and the relevant validation fixtures
together, then compile with an explicit backend policy for your model.
The notes here explain semantics and implementation techniques without keeping
a second, independently maintained copy of the generated backend matrix.

## Backend Names

| Backend | Runtime role | Main files |
| --- | --- | --- |
| WASM | Browser/Node CPU execution through the shared C runtime compiled to WASM. Both profiles include inference; full adds training/PTQ. | [C runtime](../native/src/runtime/engine_runtime.c), [native kernels](../native/src/kernels) |
| WebGPU | Browser GPU execution in the full profile; C plans work and the device bridge submits it. | [WebGPU planner](../native/src/backends/webgpu_backend.c), [shaders](../shaders) |
| CPU(Native) | Freestanding native runtime CPU path. | [engine_runtime.c](../native/src/runtime/engine_runtime.c), [native kernels](../native/src/kernels), [conv_f32_isa.c](../native/src/kernels/conv_f32_isa.c), [quant_cpu_isa.c](../native/src/kernels/quant_cpu_isa.c) |
| Vulkan(Native) | Native Vulkan compute path. It is called from the native dispatcher and is separate from browser WebGPU. | [engine_runtime.c](../native/src/runtime/engine_runtime.c), [vulkan_engine.c](../native/src/backends/vulkan_engine.c) |
| OpenGL(Native) | Native desktop OpenGL/OpenGL ES compute path. It is called from the native dispatcher and is separate from browser WebGPU. | [engine_runtime.c](../native/src/runtime/engine_runtime.c), [opengl_engine.c](../native/src/backends/opengl_engine.c) |
| CUDA(Native) | Opt-in manual-PTX path. Inference is forward-only; the full profile adds the current native F32 backward/loss/optimizer and W8-authoring contracts. Explicit/required selection is strict. | [cuda.md](cuda.md), [cuda_engine.c](../native/src/backends/cuda_engine.c), [cuda_kernels.cu](../native/src/backends/cuda_kernels.cu), [cuda_training_kernels.cu](../native/src/backends/cuda_training_kernels.cu) |
| Metal(Native) | Native Metal compute path on Apple platforms. It loads Naga-generated MSL and is called from the native dispatcher for selected F32 and canonical packed-byte W8A8 graph ops. | [engine_runtime.c](../native/src/runtime/engine_runtime.c), [metal_engine.m](../native/src/backends/metal_engine.m) |

## Operator guide

These operators appear in browser and native graphs. The table explains the
computation and points to CPU/WASM and GPU source. Read the
[current backend matrix](generated/kernel-registry.md#operator-matrix) before
selecting a route; individual source files can contain only part of its
validation and dispatch.

| Operation | What it does | Source reading |
| --- | --- | --- |
| `MatMul` | Dense matrix multiply with optional bias. INT8-packed weights use W8A32: FP32 activations and accumulation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/linearInt8.wgsl) |
| `Linear`, `Gemm` | Dense-layer aliases used by exporters. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../native/src/backends/webgpu_backend.c) |
| `Conv2D` | NHWC 2D convolution, including grouped/depthwise and dilation. HWIO weights (HWCM when depthwise). | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/conv2D.wgsl) |
| `Conv1D` | 1D convolution for sequence/audio tensors. NLC activations, WIO weights `[k, in_per_group, out_c]`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/conv1D.wgsl) |
| `ConvTranspose2D` | Transposed convolution / deconvolution. NHWC activations, HWIO weights `[kh, kw, in_c, out_c]`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/convTranspose2D.wgsl) |
| `QConv2D` | Canonical W8A8 Conv2D: NHWC I8/U8 activation, OHWI I8/U8 weight with axis-0 per-channel metadata, optional I32 accumulator bias, groups/stride/padding/dilation, and fused ReLU/ReLU6. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qConv2DInt8Tiled.wgsl) |
| `QLinear`, `QMatMul`, `QGemm` | Canonical W8A8 dense layer: `[...,d_in]` bytes, `[d_out,d_in]` per-axis weight, I32 bias, and typed output. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qLinearInt8Dot.wgsl) |
| `QBatchMatMul` | Canonical ONNX MatMul: rank-2–8 `[...,M,K] @ [...,K,N]` with right-aligned batch broadcasting, independent per-tensor I8/U8 operands and output, and preflighted I32-safe centered accumulation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qBatchMatMul.wgsl) |
| `LayerNorm` | Layer normalization over the last feature axis. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/layerNorm.wgsl) |
| `QLayerNorm` | Canonical W8A8 last-axis LayerNorm: same-shape rank-at-least-1 per-tensor I8/U8 input/output, F32 `[D]` gamma/beta, optional positive `eps` (default `1e-5`), and optional `d_model` matching D; row statistics are F32 scratch, not an F32 activation tensor. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../native/src/backends/webgpu_backend.c) |
| `RMSNorm` | RMS normalization over the last feature axis. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/rMSNorm.wgsl) |
| `RoPE` | Rotary position embedding with GPT-NeoX half-split or GPT-J interleaved pairs. | [CPU/WASM](../native/src/kernels/sequence_ops.c) · [GPU](../native/src/backends/webgpu_backend.c) |
| `SSMScan`, `SelectiveScan` | Mamba-style selective state-space recurrence with optional initial/final state, skip, and gate. | [CPU/WASM](../native/src/kernels/sequence_ops.c) · [GPU](../native/src/backends/webgpu_backend.c) |
| `BatchNorm2D` | Per-channel image batch normalization. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/batchNorm2D.wgsl) |
| `GroupNorm` | NHWC group normalization with per-channel affine scale and bias. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/groupNorm.wgsl) |
| `QGroupNorm` | Canonical W8A8 GroupNorm: rank-4 NHWC per-tensor I8/U8 input/output, F32 `[C]` gamma/beta, required `num_groups` dividing C, optional positive `eps` (default `1e-5`), and optional `data_layout: "NHWC"`; group statistics are F32 scratch, not an F32 activation tensor. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../native/src/backends/webgpu_backend.c) |
| `Embedding` | I32 row lookup into an F32 embedding table. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/embedding.wgsl) |
| `QEmbedding` | Canonical W8A8 lookup: I32 IDs `[S...]` covered by complete public-input, invariant-payload, or closed canonical producer-range preflight, I8/U8 `[vocab,hidden]` table with axis-0 row metadata, and caller-supplied per-tensor I8/U8 output descriptor `[S...,hidden]`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qEmbeddingInt8.wgsl) |
| `QMaskedMean` | Canonical TaskRouter reduction: I8/U8 per-tensor `[B,S,D]` input, unquantized I32 `[B,S]` nonzero-keep mask, and I8/U8 per-tensor `[B,D]` output. It accumulates centered bytes in I32, then performs one private scalar requantization; an all-masked row emits the output zero point. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qMaskedMeanInt8.wgsl) |
| `SDPA` | Self-attention over F32 rank-2/3 packed QKV with full/causal attention, optional keep mask, and train-only probability dropout. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/sDPA.wgsl) |
| `CrossSDPA` | Cross-attention over separate F32 rank-2/3 Q, K, and V tensors, optional keep mask, and train-only probability dropout. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/crossSDPA.wgsl) |
| `QSDPA` | Canonical W8A8 self/cross attention: separate rank-2/3 I8/U8 Q/K/V and output descriptors, optional I32 keep mask, causal mode, and positive scale. Raw QK accumulation is I32; stable softmax/value accumulation is backend-private F32 scratch, never a graph activation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qSDPAInt8.wgsl) |
| `CrossAttention` | F32 rank-2/3 cross-attention using Q, packed KV, a `[3*d_model,d_model]` projection, and optional scale/bias. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/crossAttentionF32.wgsl) |
| `MoERouter` | Temperature-scaled top-k expert routing. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/moeRouter.wgsl) |
| `MoELinear` | Execute and mix selected expert matrices. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/moeLinear.wgsl) |
| `MaxPool2D` | 2D max pooling. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/maxPool2DTyped.wgsl) |
| `AveragePool2D` | 2D average pooling. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/averagePool2D.wgsl) |
| `GlobalAveragePool` | Global spatial average pooling. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/globalAveragePool.wgsl) |
| `Resize` | Image resize. Bilinear is the normal path; typed byte activation storage is nearest-only. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/resizeNearestTyped.wgsl) |
| `ResizeNearest2D` | Nearest-neighbor image resize. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/resizeNearestTyped.wgsl) |
| `UpsampleNearest2D` | Nearest-neighbor 2x upsampling. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/upsample2x.wgsl) |
| `Interpolate1D` | Linear 1D interpolation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/interp1D.wgsl) |
| `ReLU` | Rectified linear activation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/reLU.wgsl) |
| `LeakyReLU` | Leaky ReLU activation with `alpha`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/leakyReLU.wgsl) |
| `PReLU` | Per-channel parametric ReLU. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/pReLU.wgsl) |
| `GELU` | Erf GELU by default; optional `approximate: "tanh"`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/gELU.wgsl) |
| `QGELU` | Canonical W8A8 GELU: a distinct same-shape per-tensor I8/U8 input/output pair; only omitted parameters or `approximate: "none"`; uses the fixed Abramowitz–Stegun 7.1.26 erf polynomial and ties-to-even saturating requantization. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qGELUInt8.wgsl) |
| `SiLU` | `x * sigmoid(x)` activation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/siLU.wgsl) |
| `QSiLU` | Canonical W8A8 SiLU: one distinct same-shape per-tensor I8/U8 input/output pair, no auxiliary inputs or parameters; dequantize, apply `x / (1 + exp(-x))`, and ties-to-even saturating requantize directly into output bytes. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qSiLUInt8.wgsl) |
| `Sigmoid` | Logistic activation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/sigmoid.wgsl) |
| `HardSwish` | MobileNet-style hard swish activation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/hardSwish.wgsl) |
| `HardSigmoid` | Piecewise-linear sigmoid approximation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/hardSigmoid.wgsl) |
| `Tanh` | Hyperbolic tangent activation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/tanh.wgsl) |
| `Sin`, `Cos` | Elementwise trigonometric functions over same-shape F32 tensors. | [CPU/WASM](../native/src/kernels/sequence_ops.c) · [GPU](../native/src/backends/webgpu_backend.c) |
| `Clip` | Clamp values to min/max. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/clip.wgsl) |
| `Add` | Elementwise F32 add with right-aligned N-D broadcasting. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/broadcastBinary.wgsl) |
| `QAdd` | Exact-shape W8A8 add with per-tensor descriptors, requantization, and optional fused ReLU/ReLU6. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qAdd.wgsl) |
| `Mul` | Elementwise multiply with right-aligned N-D broadcasting. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/broadcastBinary.wgsl) |
| `Sub` | Elementwise subtract with broadcasting. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/broadcastBinary.wgsl) |
| `Div` | Elementwise divide with broadcasting. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/broadcastBinary.wgsl) |
| `Softmax` | Softmax over the last axis. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/softmax.wgsl) |
| `LogSoftmax` | Log-softmax over the last axis. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/logSoftmax.wgsl) |
| `ReduceSum` | Sum the last axis, preserving one value per outer row. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/reduce.wgsl) |
| `ReduceMean` | Mean of the last axis, preserving one value per outer row. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/reduce.wgsl) |
| `ArgMax` | Inference-only index of the maximum value; the axis may be removed or retained as size 1. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../native/src/backends/webgpu_backend.c) |
| `QArgMax` | Canonical terminal byte-domain ArgMax: rank-2–8 per-tensor I8/U8 input, exact integer axis, first-tie semantics, and unquantized I32 output with that axis removed. Positive affine quantization preserves raw-byte order, so logits stay byte-typed. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/qArgMaxInt8.wgsl) |
| `Transpose` | General N-D tensor permutation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/generalTranspose.wgsl) |
| `Concat`, `Concat2` | Tensor concatenation. Raw byte concat requires identical dtype and immutable descriptor; fused sigmoid requires an explicit F32 boundary. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/concatCopyTyped.wgsl) |
| `Split` | Split a tensor into multiple outputs. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/split.wgsl) |
| `Slice` | Canonical F32 rank-1–8 strided slice with normalized axes/starts, positive integer steps, and an in-bounds selection. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/sliceNd.wgsl) |
| `Pad` | Constant padding. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/pad.wgsl) |
| `Expand`, `Broadcast` | Canonical F32/I32 right-aligned broadcast plus descriptor-preserving per-tensor I8/U8 `Expand`, to a valid rank-1–8 target shape. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/expandTyped.wgsl) |
| `Gather` | Canonical F32 gather with I32 indices, rank at most 8, any valid axis, and ONNX negative-index normalization. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/gatherInt32.wgsl) |
| `GatherElements` | Canonical F32 elementwise indexed gather with I32 indices, rank at most 8, and normalized negative indices. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/gatherElements.wgsl) |
| `Where`, `Mask` | Exact-shape F32 selection with an F32 or I32 condition; `mask`, `cond`, and `condition` are aliases. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/whereTyped.wgsl) |
| `Cast` | Dtype conversion among F32/I32/I8/U8; F32-to-I32 truncates toward zero, wraps modulo 2^32, and maps NaN/infinities to zero; only F32-to-F32 has a backward. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/cast.wgsl) |
| `DequantizeLinear` | Convert F32/I32/I8/U8 values to F32 with scalar F32 scale and an optional scalar typed zero point. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/dequantizeLinearTyped.wgsl) |
| `QuantizeLinear` | Canonical F32-to-I8/U8 boundary with scalar F32 scale, matching typed zero point, nearest-even rounding, saturation, and a required central per-tensor output descriptor whose numeric parameters live in safetensors. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/quantizeLinearTyped.wgsl) |
| `RequantizeLinear` | I8/U8-to-I8/U8 conversion between central per-tensor descriptors, with nearest-even rounding and saturation. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/requantizeLinearTyped.wgsl) |
| `NonMaxSuppression` | Inference-only greedy NMS for object-detection boxes. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/nonMaxSuppression.wgsl) |
| `Reshape` | Shape-only tensor view/copy. Typed storage requires an unchanged descriptor. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/copyTyped.wgsl) |
| `Flatten` | Shape-only flatten. Typed storage requires an unchanged descriptor. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/copyTyped.wgsl) |
| `Squeeze` | Remove size-1 dimensions. Typed storage requires an unchanged descriptor. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/copyTyped.wgsl) |
| `Unsqueeze` | Add size-1 dimensions. Typed storage requires an unchanged descriptor. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/copyTyped.wgsl) |
| `Dropout` | Deterministic inverted Dropout while training; exact identity during inference. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/training/dropout.wgsl) |
| `Identity` | Pass-through copy. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/copy.wgsl) |
| `SpatialSoftargmaxY` | F32 NHWC vertical soft-argmax for vision postprocessing; output `[N,C,W]`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/spatialSoftargmaxY.wgsl) |
| `ProfileX` | F32 NHWC horizontal profile/reduction; output `[N,2C,W]`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/profileX.wgsl) |
| `ProfileY` | F32 NHWC vertical profile/reduction; output `[N,2C,H]`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/profileY.wgsl) |
| `MeanHeight` | F32 NHWC height mean; output `[N,C,W]`. | [CPU/WASM](../native/src/runtime/engine_runtime.c) · [GPU](../shaders/inference/meanHeight.wgsl) |

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
operators in this section currently have forward implementations on C/WASM and
CPU(Native); browser and native GPU backends decline them as described in their
tables.

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

“W8A8” here means byte storage for quantized activation/weight edges,
not integer-only arithmetic. `QEmbedding` starts from I32 IDs;
`QSiLU`/`QGELU` have exact byte-to-byte transforms and requantization;
`QGroupNorm` and
`QLayerNorm` also have F32 affine inputs plus backend-private F32 statistics
scratch, without materializing an F32 activation tensor.

WASM, WebGPU, and native CPU execute the canonical W8A8
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
In a retained AMD GCN-5 WebGPU kernel campaign, the byte-exact harness measured the
DP4a decoder-row QLinear at 0.186 ms versus 0.803 ms portable (4.32x), the tiled
multi-row QLinear at 0.249 ms versus 0.345 ms portable (1.39x), and tiled QConv
at 0.480 ms versus 1.950 ms scalar portable (4.06x). The feature-free portable
QConv tile measured 0.760 ms, 2.57x faster than scalar and 1.59x slower than
DP4a. These are isolated kernel timings, not end-to-end VQA latency.

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
`QConv2D` routes continue to mean canonical W8A8. Every other
native `QConv2D` must declare I8/U8 input and output descriptors. The
loader rejects an untyped activation graph instead of inferring quantization
parameters from neighboring nodes.

CPU(Native) stores canonical W8A8 activations directly in I8/U8
buffers, without a sidecar or FP32 materialization.
For C `QSiLU` and `QGELU`, a call of at least 512 elements builds a thread-local
256-entry table keyed by the exact input/output descriptors; subsequent calls,
including smaller decoder rows, map raw bytes through that table. Small cold
calls stay scalar. Every table byte is produced by the same transform and
ties-to-even saturation as the scalar route, and exhaustive cold/build/warm
tests cover all I8/U8 input/output pairings.

The activation table lives in the portable C inference kernels, so it ships in
both `volvoxai.lite.wasm` and `volvoxai.wasm`. Native and WASM share C graph and
decode planning; native worker threading and ISA selection depend on the build.
WASM uses baseline SIMD128 and portable C kernels. Current release profiles
contain no Relaxed-SIMD child. The baseline packed
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
one-token decoder graph. The package pipeline imports either the
FP32 or producer-authored static INT8 variant from the cache-enabled split ONNX
directory. There is no second application PTQ or calibration pass. The complete
commands and qualification rules are in the
[TinyReceipt example](../examples/tiny_receipt_vqa/README.md).

The encoder binds the exact question extent `Q` and derives memory extent
`M=Q+210`. Each decoder call binds one current token, explicit cross K/V, and
self-attention past K/V with `R=P+1`. The explicit-KV package begins with the
qualified blocked zero `P=1` sentinel.

The package records grayscale, bilinear `672x320`, `[-1,1]` F32 NCHW input,
canonical byte-fallback BPE tokenization, mask semantics, and exact graph asset
hashes. Provider qualification must preserve selected family, greedy token IDs,
structured text, and task results while forbidding backend fallback.

### W8A8 benchmark harnesses

Run `make benchmark_wasm_w8a8_prefill` for byte-checked TinyReceipt prefill proxies.
It times the exact grayscale stem and the exact weighted `M=402`/`M=192`
base-dense call mix. Eligible convolution is reported as im2col layout plus
packed SIMD128 GEMM, with canonical portable QConv and an independent
JavaScript arithmetic oracle that is not a runtime backend. This benchmark
measures the current SIMD128 and portable routes. Release WASM contains no
Relaxed-SIMD child.

Run `make benchmark_wasm_qbatch_matmul` for deterministic mixed-I8/U8
QBatchMatMul proxies covering a `320`-wide decoder row, a multi-row encoder
matrix, and odd K/N tails. The harness first requires byte equality between
the exported scalar and standard-SIMD128 kernels, then reports their separate
times. It is a kernel benchmark, not an end-to-end model latency claim.

The [latest TinyReceipt benchmark](../examples/tiny_receipt_vqa/BENCHMARK.md)
records complete-answer latency with exact workload, runtime and model hashes.
The explicit-KV decoder reads the caches needed by the next step into caller
memory. Native BufferView payloads avoid protobuf tensor serialization; they
do not make that application loop device-resident.

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
use target-attributed AVX2 widening and `VPMADDWD`. Prefill-sized PMADDUBSW and
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
channels. These kernels operate directly on the declared byte tensors.

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
operator route. A generated `BackendPolicy` in REQUIRE mode with operator
fallback FORBID accepts only an attested all-device route and rejects the graph
before execution otherwise.

Native shader generation is separate from native runtime support. The tracked
`shaders/{inference,training}/*.wgsl` files are translated by `make compile_shaders` / Naga into ignored
build outputs under `native/shaders/{spv,glsl,gles,metal}`. The Rust generator passes
the original WGSL binding numbers through Naga's structured GLSL/MSL resource maps;
generated source is not patched afterward. An op is marked supported
for Vulkan/OpenGL only when `engine_runtime.c` has a dispatcher path and the native
backend has a C wrapper for that shader. Metal follows the same rule on Apple builds:
the MSL may be generated for many ops, but a row is marked supported only when
`engine_runtime.c` dispatches to a `metal_graph_*` wrapper.

The [current operator matrix](generated/kernel-registry.md#operator-matrix)
lists native CPU, Vulkan, OpenGL, Metal, and CUDA registrations beside the
browser routes. Use [native deployment](native-runtime.md) to enable/select a
backend and [CUDA](cuda.md) for its numerical modes and training coverage.
Physical-device validation is separate from shader generation or registration.

## Training and backward status

Training is a separate full-profile path. Normal inference does not allocate
gradient buffers, optimizer state, random masks, or backward pipelines. The
inference profile includes execution, planning, text, and scheduling. Full
adds training and quantization. Each `CreateTrainer` response is a retained handle
pinned to the exact loaded Model revision.

Strict `backend: "wasm"` training in `volvoxai.wasm` accepts the
differentiable complete contracts documented here and rejects partial or
non-canonical contracts before model state changes. Browser WebGPU implements
the same portable contracts. `ArgMax` and
`NonMaxSuppression`, `RoPE`, `SSMScan`/`SelectiveScan`,
`Sin`, and `Cos` are intentionally forward-only. This browser/Node portability
gate does not change native coverage below. CUDA is intentionally kept out of
this training summary because its fast-changing full-profile F32 backward,
loss, optimizer, and W8-authoring contract is maintained in
[cuda.md](cuda.md#backward-operator-coverage). Its inference profile
remains free of all training code and PTX.

| Capability | Strict full-WASM | Browser WebGPU | Native CPU | Vulkan / OpenGL / Metal training |
| --- | --- | --- | --- | --- |
| NHWC `GroupNorm` | Forward and input/affine backward | Forward and input/affine WGSL backward | Forward and input/affine backward | Forward and input/affine GPU backward for supported F32 layouts |
| Standalone `Dropout` | Deterministic inverted forward/backward | Deterministic inverted WGSL forward/backward | Deterministic inverted forward/backward | Deterministic inverted GPU forward/backward |
| `SDPA` / `CrossSDPA` probability dropout | Exact training-only mask | Exact training-only mask | Deterministic after-softmax training mask | Dropout-aware forward and backward, with CPU fallback if GPU dispatch is unavailable |
| `Add` / `Mul` broadcasting | Right-aligned N-D forward and reduced backward | Right-aligned N-D forward and reduced WGSL backward | Right-aligned N-D forward and reduced backward | Right-aligned F32 forward and reduced GPU backward, up to rank 8 |
| `ReduceSum` / `ReduceMean` | Last-axis forward/backward | Last-axis WGSL forward/backward | Last-axis forward/backward | Last-axis F32 GPU forward/backward |
| Weighted CE losses and accumulation | `losses`, per-loss metrics, explicit normalizers, reset/flush | Same contract | Generated full-profile Trainer handle | Same Trainer contract; loss seeding and the optimizer remain coordinated by its private owner |

For strict full-WASM and WebGPU, attention probability `dropout` (or
`attention_dropout`) is applied after softmax with inverted scaling and no
renormalization. Forward and backward regenerate the same deterministic
`[B,H,Q,K]` mask. Inference remains ordinary deterministic attention. Native
CPU, Vulkan, OpenGL compute, Metal, and full-profile CUDA implement the same
after-softmax training semantics and regenerate their mask during backward.
Vulkan/OpenGL/Metal dispatch may fall back to the matching CPU forward/backward
path when its training shader or layout is unavailable; required-CUDA training
instead preflights the complete plan and fails closed. Standalone `Dropout` is
fully separate and supported on all of them. Determinism is per backend and
training state; the WASM/WebGPU and native RNG-seed contracts are not a promise
of bit-identical masks to each other.

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
   error. Every successful generated result handle owns a stable snapshot for
   all declared outputs, including input, weight, and transitive alias outputs.
3. **Canonical W8A8 scope:** WASM, WebGPU, native CPU, and
   the native GPU graph routes accept the documented canonical subset. It is
   not a universal INT8-op set: unsupported raw I8/U8 activations fail closed
   until an explicit typed kernel is added. The direct TFLite exporter also
   retains an explicit F32 Sigmoid boundary. TinyReceiptVQA's
   [latest 2,000-case audit](../examples/tiny_receipt_vqa/BENCHMARK.md) covers
   imported INT8 and native/WASM PTQ across six backends. Its importer defaults
   to a static B=1 package and can opt into a proved symbolic B=1..N component
   graph. The generated-proto application owns cross K/V, past/present self K/V, and
   mask progression explicitly; `P/R` advance one token per call. Provider
   tactics may reuse proved fixed capacity without changing the graph ABI.
   Component physical-batch qualification, held-out answer accuracy and
   whole-output agreement with source ONNX Runtime are separate measurements.
4. **Native GPU coverage:** Vulkan(Native), OpenGL(Native), and Metal(Native)
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
5. **Training coverage:** Strict full-WASM and browser WebGPU
   implement the portable matrix's differentiable complete rows. Native CPU and
   Vulkan/OpenGL/Metal retain separate explicit backward allowlists; an
   unsupported native GPU operation either selects the complete native CPU
   backward path during preflight or fails, never silently stops a gradient.
   Batched SDPA/CrossSDPA are not restricted to batch one. WebGPU portable
   attention requires `head_dim <= 64` (`CrossAttention` F32 also requires
   `d_model <= 64`), and WebGPU MoE routing requires `top_k <= 8`. Exact
   after-softmax attention-probability dropout is supported by strict full-WASM,
   browser WebGPU, native CPU, Vulkan, OpenGL compute, and
   Metal training. Full-profile CUDA implements the current native F32
   backward contract plus device loss, accumulation, clipping, SGD, and AdamW;
   its staged-attention implementation, split-axis checks, combined batch-one
   query/output-length-192 accumulation-24 TinyVQA update/checkpoint/resume
   evidence, and remaining residency/performance work are tracked in
   [cuda.md](cuda.md). Metal is Apple-only and cannot be
   runtime-tested on Linux.
   See [model construction, routing, and training](model_builder_training.md) for
   the API and current constraints.
