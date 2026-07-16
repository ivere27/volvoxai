# Native Runtime

`native/` is a freestanding C runtime that runs the same Volvox blueprint package as
the browser runtime: `config.json` plus `.safetensors`. It has no static GPU SDK
dependency. Vulkan, OpenGL, Metal, and NNAPI are loaded dynamically when requested.

Device integrations are also compile-time composable. The Make variables
The CMake options `VOLVOXAI_ENABLE_VULKAN`, `VOLVOXAI_ENABLE_OPENGL`,
`VOLVOXAI_ENABLE_METAL`, and `VOLVOXAI_ENABLE_NNAPI` accept `ON`/`OFF`; a
disabled backend's source and vtable entries are omitted. Linux defaults to
Vulkan and OpenGL, macOS additionally defaults to Metal, and NNAPI is enabled by
the Android target. For example, configuring with
`-DVOLVOXAI_ENABLE_VULKAN=OFF -DVOLVOXAI_ENABLE_OPENGL=OFF` builds the CPU-only
profile. The `native_backend_composition` CTest case (run by `make test_native`)
performs that source-exclusion check explicitly.

The embedded shader pack follows the same composition. Vulkan contributes the
SPIR-V block, OpenGL contributes desktop GLSL and GLES blocks, and Metal
contributes the Metal block. Consequently, a default Linux binary contains no
Metal source or Metal shader bytes; an all-GPU-off build carries a valid empty
pack while preserving the shader-store lifecycle and override behavior for
builds that include a GPU backend.

Normal native releases use a baseline CPU target. W8A8 x86 AVX2, AVX-VNNI,
and AVX-512 VNNI plus ARM NEON/SDOT kernels live behind runtime capability
checks; unsupported CPUs retain the portable scalar implementation. Select
`NATIVE_CPU_TARGET=baseline` (the default) for a portable binary or
`NATIVE_CPU_TARGET=avx2` for an x86 deployment that guarantees AVX2/FMA.
`NATIVE_CPU_FLAGS` remains available as a lower-level override.

On native Linux AArch64 and Android AArch64, the build compiles QLinear and
QConv2D SDOT implementations as separate `armv8.2-a+dotprod` objects. The
baseline translation units remain deployable on ordinary Armv8 cores and call
those objects only after `HWCAP_ASIMDDP` succeeds. Other ARM targets retain
their baseline NEON or scalar route unless their build supplies an equivalent
runtime-gated dot-product object.

The target-attributed physical W8A8 QLinear and QConv2D implementations retain
runtime AVX2, AVX-VNNI, and AVX-512 VNNI dispatch even in a baseline build.
The AVX-512 gate requires CPUID AVX2/AVX-512F/BW/VL/VNNI and OS-enabled
XMM/YMM/opmask/ZMM state before entering a ZMM function. Older AVX2 blocks in
`quant_cpu_opt.c`, `conv_f32_opt.c`, and `tensor_f32_opt.c` are compile-time
intrinsics and therefore activate only for the `avx2` target (or equivalent
custom `NATIVE_CPU_FLAGS`); their baseline forms use the scalar/ARM paths.

Seed-sized x86 QLinear calls process four activation rows together, reusing each
loaded weight vector across the row tile before moving to the next output. This
applies to AVX2, AVX-VNNI, and AVX-512 VNNI without changing the accumulator or
requantization contract. QConv2D similarly reuses one activation block across
four output channels for ordinary channel counts. For narrow inputs such as the
TinyReceipt grayscale stem, it transiently repacks the immutable OHWI weights
and evaluates eight output channels together; the authoritative portable path
remains the fallback for allocation, aliasing, ISA, or overflow failures.

Large native W8A8 work also uses the shared kernel thread pool. Eligible x86
`QLinear` calls partition independent output rows and `QConv2D` partitions
independent NHWC output locations after their normal ISA, overflow, and alias
checks. Whole-tensor `QSDPA` partitions independent batch/query rows while
calling the portable kernel for each row, so each head retains the same key and
dimension arithmetic order. The pool gates are deliberately size-based:
incremental `QLinear` with `M=1` and `QSDPA` with `Q=1` stay on the caller, as do
small convolutions, avoiding worker wake-up in steady token decode.

Canonical C `QSiLU` and `QGELU` cache the exact 256 possible output bytes for an
immutable input/output quantization descriptor. A cold call below 512 elements
uses the original scalar transform; a larger call builds a thread-local table,
which later small decoder rows can reuse. Table entries are produced by the
same activation polynomial/libm and ties-to-even requantization code as the
scalar route, so this is a lookup optimization rather than a numerical
approximation. The portable C implementation is shared by native CPU and WASM.

## Build

```bash
make build_native
./native/volvoxai --version
./native/volvoxai --help
./native/volvoxai-full --help
```

The Docker build produces `native/volvoxai` for inference and
`native/volvoxai-full` for inference plus training. Both use clang, pthreads,
and the selected CPU/GPU backend sources. Each binary embeds package
version, git commit, and UTC build date from the CMake build. Run
`make build_native` to build both profiles (`native/volvoxai` and
`native/volvoxai-full`).

The fixed binaries deliberately have a narrow command surface. Both expose
`run`, `--help`, and `--version`; only the full profile additionally exposes
`train`. Image decoding, vocabulary selection, generation loops, and task
postprocessing are not linked into either release artifact.

The full executable exposes a generic cross-entropy training command:

```bash
./native/volvoxai-full train models/my_model \
  --input input=batch.f32 \
  --targets targets.i32 \
  --logits logits \
  --trainable classifier.weight \
  --trainable classifier.bias \
  --steps 10 \
  --learning-rate 0.001 \
  --output-weights trained.safetensors \
  --output-optimizer optimizer.safetensors
```

Targets are raw int32 class IDs. `--trainable` is repeatable, and
`--input-optimizer` resumes previously saved optimizer state and its training
step. The inference executable neither shows nor compiles this command.

Both executables embed deterministic XZ-compressed shader blocks. The inference
profile contains forward shaders only; the full profile adds separate training
blocks. A block is decompressed and cached only when its backend and scope are
first used, so normal CPU execution allocates no shader memory.

For shader development, `make compile_shaders` creates the ignored
`native/shaders/{spv,glsl,gles,metal}` tree. Set `VOLVOXAI_SHADER_DIR` to
`native/shaders` (or another tree with those four children) to override embedded
bytes. The runtime logs the selected environment override once. A missing file
warns once and falls back to the embedded copy.

On macOS, build on the host so clang can compile
`native/src/backends/metal_engine.m` and link the system Metal frameworks:

```bash
cmake -S . -B build/mac -DCMAKE_C_COMPILER=clang -DVOLVOXAI_ENABLE_METAL=ON
cmake --build build/mac
```

This target also compiles the runtime-loaded Vulkan backend, so Vulkan headers must
be installed; set `VULKAN_SDK` when using the LunarG SDK.

The library compiler under `tools/metal_shader_compiler` generates both GLSL and
MSL. It gives Naga 30 an explicit GLSL `Options.binding_map` for core430 and es310,
and an MSL `Options.per_entry_point_map`. Generated shaders therefore contain the
logical WGSL buffer indices directly; no Python, regex, or other post-generation
binding-number rewrite is used. Multi-entry backward modules are emitted as one
deterministically named GLSL file per entry point.

## Architecture

Key files:

- `native/src/runtime/engine.c`: parses the blueprint, builds tensors/nodes, and
  owns the global engine context.
- `native/src/runtime/backend.[ch]`: private `VxBackend` registry. It validates
  ordered backends, fans out graph-storage lifecycle/synchronization hooks, and
  selects the first backend that handles a node; CPU is always the final fallback.
- `native/src/runtime/engine_runtime.c`: private execution compilation boundary.
  Its ordered `engine_runtime_*.inc` sections separate model/graph state, F32 CPU,
  F32 GPU, physical W8A8, and dispatch routes while preserving runtime-static scope.
- `native/src/runtime/arena.c`: activation-arena planning and tensor-table
  mutation restoration.
- `native/src/runtime/incremental_runtime.[ch]`: opt-in dependency scheduling,
  cache invalidation, arena de-aliasing, and CPU row orchestration.
- `native/src/runtime/attention_mask.[ch]`: mask shape/indexing shared by forward
  execution and the guarded full-profile training implementation.
- `native/src/runtime/sequence_runtime.[ch]`: narrow native validation and CPU
  dispatch boundary for `Sin`, `Cos`, `RoPE`, and selective state-space scan;
  the scalar math remains in `native/src/kernels/sequence_ops.c` for native/WASM parity.
- `native/src/backends/w8a8_device_ops.[ch]`: immutable Vulkan/OpenGL/Metal W8A8
  kernel tables used by the runtime's device routes.
- `native/src/backends/backend_manager.[ch]`: runtime-owned backend selection,
  device initialization, and cleanup; applications never define backend flags.
- `native/src/kernels/thread_pool.[ch]`: shared restartable CPU worker pool owned
  by the kernel layer and shut down with the runtime.
- `native/src/kernels/qlinear_w8a8_x86.c` and `qconv_w8a8_x86.c`: runtime-gated
  x86 W8A8 SIMD plus large-call row/output-location scheduling.
- `native/src/kernels/qsdpa_w8a8_native.c`: exact native whole-tensor QSDPA
  scheduling over independent batch/query rows; the portable kernel remains
  the numerical implementation and the single-row fallback.
- `native/src/kernels/lora_linear.[ch]`: private LoRA numerical kernels shared by
  adapter materialization and routed inference.
- `native/src/training/training_runtime.inc`: full-profile backward planning,
  gradient accumulation, and train-step orchestration behind one compile boundary.
- `native/src/training/optimizer_runtime.inc`: full-profile optimizer state,
  tensor updates, and optimizer checkpoint persistence.
- `native/src/shader_store.c`: external development override plus lazy decoding
  of embedded backend/scope shader blocks.
- `native/src/runtime/graph_opt_fusion.inc`: private compile-time fusion pass for patterns such as Conv+ReLU6,
  chained Add, depthwise-to-pointwise, concat+sigmoid, and alias elision.
- `native/src/kernels/conv_f32_opt.c`: optimized FP32 Conv2D with weight prepacking and IGEMM.
- `native/src/kernels/quant_cpu_opt.c`: INT8 quantized CPU path for `QConv2D` and
  quantization helpers.
- `native/src/kernels/kernels.c`: shared portable kernels used by both native and wasm32.
- `native/cli/main.c`: fixed model-agnostic raw-tensor runner plus the guarded
  full-profile training command.
- `native/src/tokenization/tokenizer.c` and `native/include/volvoxai_tokenizer.h`:
  opaque public native BPE tokenizer available to applications and services.
- `examples/native_support/`: neutral native image decoding shared by opt-in
  example applications.
- `examples/native_task_cli/`: opt-in vocabulary policy, generation loops, and
  general task wrappers built on public runtime APIs.
- `examples/tiny_receipt_vqa/native/`: opt-in typed TinyReceiptVQA session and
  model-specific preprocessing, outside both fixed native release binaries.

## Raw Tensor Runner

```bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 \
  --row 4 \
  --debug
```

`--row` selects an explicit row from rank-2-or-higher outputs; the fixed runner
does not assign token or decoder semantics to that index. Row output is
F32-only because the public row-view API exposes F32 rows.

`run` accepts named raw tensor files. Input and output suffixes must match the
declared storage dtype: `.f32`, `.f16`, `.i32`, `.i8`, or `.u8`; byte sizes must
match exactly. Weightless graphs may omit `--weights`; strict graph preflight
still rejects unresolved weight names. The runner does not infer image shapes
or choose a normalization policy.
Applications that already have real F32 values can call
`volvoxai_engine_set_input_f32()`. It copies them into an F32 graph input or
quantizes them into a canonical I8/U8 graph input with that input's declared
per-tensor scale and zero point. Use `volvoxai_engine_set_input_raw()` when the
caller already owns the exact physical storage bytes.
Image, video, audio, camera, and streaming inputs should be decoded by a
frontend and supplied as tensors. The task example below provides one opt-in
PNG/JPEG frontend.

## Opt-in Task CLI Example

Build the separate example application when command-level image, vocabulary,
generation, or postprocessing policy is useful:

```bash
make -C examples native_task_cli
```

Its task wrappers sit on top of the same public runtime:

Image packages declare per-input `image_normalization` metadata. For an older
or ad hoc package without that metadata, add an explicit
`--image-normalize zero-one`, `minus-one-one`, or `raw-255` option.

```bash
examples/target/bin/volvoxai-tasks classify models/classifier \
  --image image=photo.jpg \
  --logits logits \
  --labels labels.txt \
  --top-k 5

examples/target/bin/volvoxai-tasks detect models/detector \
  --image image=receipt.jpg \
  --classes classes \
  --max-det 20

examples/target/bin/volvoxai-tasks ctc models/ocr_line \
  --image image=line.png \
  --logits logits \
  --labels labels.txt \
  --blank 0

examples/target/bin/volvoxai-tasks seq2seq models/encoder_decoder \
  --image image=receipt.jpg \
  --prompt "What is the first number of the store phone?" \
  --prompt-input q_tokens \
  --decoder-input y_tokens \
  --logits logits \
  --max-new 192

examples/target/bin/volvoxai-tasks chat models/multimodal_chat \
  --prompt "Summarize this receipt" \
  --image image=receipt.jpg
```

`classify` ranks logits, `ctc` performs greedy CTC collapse, and `detect` prints or
writes named raw tensors because detector output layouts vary by exporter. When
the detector model directory contains `labels.txt`, its ranked table includes a
`label` column; `--labels` overrides that file. The table retains the raw
`score` and adds `score_pct` (`score` multiplied by 100) as a human-readable
percentage. Detection uses output tensors named `boxes` and `scores` by default;
the corresponding options override those names for nonstandard packages.
`seq2seq` runs a greedy encoder-decoder loop. `chat` is a user-facing alias for
multimodal seq2seq packages. Image normalization metadata, explicit preprocessing
overrides, and default vocabulary and label filenames are application policy,
not engine behavior.
See [`examples/native_task_cli/README.md`](../examples/native_task_cli/README.md)
for its complete scope and command-line options.

## Typed TinyReceiptVQA W8A8

The materializer emits a package directory with `package_manifest.json`, one shared
`model.safetensors`, a router graph, eight explicit-family graphs, and a JSON
`CharVocab`. Build and run the opt-in native example:

```bash
make -C examples native_receipt_inference_example

examples/target/bin/tiny_receipt_w8a8 /path/to/materialized_tinyreceipt_w8a8 \
  --image receipt.jpg \
  --prompt "What is the phone number?" \
  --max-new 96 \
  --incremental
```

The example verifies the manifest format and keeps every referenced graph, weights,
and vocabulary file inside that package directory. It converts decoded RGB to rounded
8-bit grayscale **before** bilinear `672x320` resize, then applies
`(pixel / 255 - 0.5) / 0.5` into the F32 NHWC `image` input; the graph itself performs
its first `QuantizeLinear` operation. This preserves the source evaluator's preprocessing
order (image decoder implementations can still differ at the pixel level).

Question and decoder IDs plus all attention keep masks are passed with
`volvoxai_engine_set_input_raw(..., VOLVOXAI_DTYPE_I32, ...)`. The wrapper runs the
router graph first, uses its one-element I32 family ID unless `--family` overrides it,
then reads each `token_ids[step]` result from the terminal `QArgMax` output directly.
It does not read F32 logits or re-run ArgMax.

This model session is not linked into `native/volvoxai` or
`native/volvoxai-full`. Ordinary full-graph forward remains the default. The
opt-in `--incremental` path creates a `VolvoxAIDecodeSession`, seeds the
complete fixed-shape graph on the first decoder step, then lets the session use
row execution when supported or rerun only descendants of `y_ids`/`y_keep`.
This retains the image
stem, encoder memory, cross-attention K/V, and other input-independent tensors
for the image instead of recomputing them per token.

The row call has a strict cache contract: after the seed, every modified
row-shaped graph input may differ only at the selected row. A caller may submit
the complete buffer through `volvoxai_engine_set_input_raw()`, but all other
rows must retain their seeded values. After changing multiple rows, reset the
incremental cache and use `volvoxai_engine_forward_incremental()` so no cached
row remains stale.

On native CPU, later steps additionally refresh only the current B=1 decoder
row. The physical self-attention K/V projection tensors persist in the
dependency cache, so causal `QSDPA` reads prior rows and appends the new row;
cross-attention reads the already-cached encoder K/V. This row path covers the
materialized decoder's `QEmbedding`, `QAdd`, `QLayerNorm`, `QLinear`, `QGELU`,
`QSDPA`, and terminal `QArgMax` chain. A built-in Vulkan, OpenGL, or Metal
session may run the complete seed on the selected GPU and then make a one-way
handoff to this CPU row path. Before the first later token, the runtime
validates the entire changed closure and synchronizes its retained prefixes
and clean cross-attention boundaries exactly once. It falls back to ordinary
GPU dependency execution without changing ownership if the closure is not
canonical. Set `VOLVOXAI_DISABLE_GPU_CPU_ROW=1` to disable the handoff for an
A/B comparison. A selected public V1 backend also uses dependency mode because
that ABI cannot coordinate the runtime's private attention K/V rows. A failed seed/decode, model
weight change, adapter-targeted run, explicit reset, or ordinary forward
invalidates the retained row cache before it can be reused.

With `--debug`, the example prints both the existing total generation line and
an incremental timing split:

```text
[debug] tinyreceipt timing seed=847.765 ms steady_steps=99 steady_mean=1.469 ms steady_tok/s=680.81
```

`seed` is decoder step zero, including the complete fixed-shape dependency
seed. `steady_mean` and `steady_tok/s` cover only later row steps, from updating
the typed decoder inputs through copying the typed token output; they exclude
the seed. The total line still includes both phases. This distinction matters
because a short answer can have a fast steady decoder while the one-time image,
encoder, and full decoder seed remains the dominant latency.

The browser/Node CPU, WASM, and WebGPU executors mirror this fixed-B=1 row
contract for the same canonical decoder operator chain. WebGPU retains the K/V
prefixes in canonical device buffers, runs the selected decoder closure through
one-row scratch storage in one submission, and supports a four-byte ranged
readback for the generated I32 token. Native Vulkan/OpenGL/Metal kernels retain
their ordinary full-tensor contract; the native decode session avoids that
per-token cost by handing compatible later rows to the CPU implementation.

The lower-level `volvoxai_engine_forward_incremental*()` functions remain
available for schedulers with specialized needs. New autoregressive wrappers
should prefer the decode-session facade so row/dependency negotiation and cache
invalidation stay out of model-specific token loops:

```c
VolvoxAIDecodeSessionOptions options = VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
VolvoxAIDecodeSession* decode = volvoxai_engine_decode_session_create(&options);
if (!decode) return -1;
if (volvoxai_engine_decode_session_seed(decode) != 0) {
    volvoxai_engine_decode_session_destroy(decode);
    return -1;
}

/* After updating the row-shaped decoder inputs for position 1: */
if (volvoxai_engine_decode_session_step(decode, 1) != 0) {
    volvoxai_engine_decode_session_destroy(decode);
    return -1;
}
volvoxai_engine_decode_session_destroy(decode);
```

One native engine graph and incremental cache exist per process, so only one
decode session may be attached to a loaded graph at a time. Shutdown or reload
invalidates that session; its stale handle can only be queried or destroyed and
cannot reset a replacement session's cache.

## Task Example Generation

TinyStories generation uses the same loaded graph repeatedly:

```bash
examples/target/bin/volvoxai-tasks generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" \
  --max-new 50 \
  --debug
```

Weights and graph load once. The task example discovers the declared primary output,
runs the prompt through `volvoxai_engine_forward_prefix()`, advances with
`volvoxai_engine_forward_row()`, and reads each logits row through
`volvoxai_engine_tensor_row_f32()`.
For language models, `--vulkan` uses the MatMul-focused path: eligible multi-row
MatMul/Gemm/Linear nodes use a cooperative 16x16 tiled shader, while small decode-time
MatMuls stay on the packed CPU `M=1` microkernel to avoid dispatch overhead.

## Native GPU and NPU Backends

Driver loading is resolved at runtime:

- Vulkan (`--vulkan`): `libvulkan.so.1`, `libvulkan.so`, or `vulkan-1.dll`.
- OpenGL (`--opengl`): `libGL.so.1`, `opengl32.dll`, or macOS `OpenGL.framework`.
- Metal (`--metal`, macOS): default `MTLDevice` through the Objective-C runtime.
- NNAPI (`--nnapi`, Android compatibility build): legacy Android Neural Networks
  API offload for large MatMul/FullyConnected nodes.

Pass at most one accelerator flag. The fixed runner and task example each
translate it into a public `VolvoxAIEngineOptions` policy and call
`volvoxai_engine_configure()`; neither initializes backend devices itself. An
explicit backend request that is not
compiled or cannot be initialized fails instead of silently changing to CPU.

EfficientDet-style vision graphs have graph-resident Vulkan and OpenGL paths for
fused Conv2D/ReLU6, MaxPool2D, same-size Add, Clip, Sigmoid, 2x nearest upsample,
Concat, and reshape/identity aliasing. Typed physical W8A8 routes use the same
ordered backend registry for QLinear, QEmbedding, QConv2D, byte-domain
elementwise/norm/attention/reduction operations, and compatible shape operators;
an unsupported device kernel cleanly falls back to the CPU reference path.
For groups=1 W8A8 convolutions with at least 32 output channels and a reduction
width of at least 16, OpenGL uses a portable cooperative 8x4 tiled kernel when
the output channels are divisible by four, and falls back to the scalar shader
if compilation or dispatch is unavailable. Set
`VOLVOX_OPENGL_DISABLE_TILED_QCONV=1` only for regression A/B testing.

Out-of-tree devices register a versioned `VxBackendV1` by name and consume only
opaque node/tensor accessors. They can implement a useful operator subset while
the same registry keeps CPU as the correctness fallback; no core enum or
per-node dispatch branch is required. See [Custom backend SDK](backend-sdk.md).

NNAPI was deprecated in Android 15. It remains here for compatibility, while
new vendor NPU, QNN, LiteRT-delegate, or custom-driver integrations should use
the named backend SDK. See Android's
[NNAPI migration guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)
and Google's [LiteRT NPU delegate guidance](https://ai.google.dev/edge/litert/android/npu).

On software Vulkan/OpenGL stacks, dispatch synchronization can be slower than the
AVX2 CPU path. The GPU win is on real mobile or desktop GPUs.

TinyReceipt incremental decode is a CPU-row workload even when a built-in GPU
runs the seed. Vulkan, OpenGL, and Metal retain ordinary full-tensor kernels,
but a compatible decode session now synchronizes the retained prefix and
cross-attention boundary once and executes all later rows with the native CPU
row/KV cache. On the repository's AMD Renoir sample run (`00002.jpg`, `phone
number last one`, 79 later steps), native CPU and OpenGL hybrid returned the
same complete answer. Current paired timings were:

| Backend | Total | Seed | Later token mean |
| --- | ---: | ---: | ---: |
| CPU row/KV | 650.177 ms | 530.060 ms | 1.518 ms |
| OpenGL GPU seed -> CPU row/KV | 870.770 ms | 599.302 ms | 3.434 ms |
| OpenGL device dependency, handoff disabled | 6,651.519 ms | 607.653 ms | 76.488 ms |

A paired scalar-QConv fallback run measured 7,716.203 ms total, 1,731.571 ms
seed, and 75.742 ms later-token mean. Three alternating one-token A/B runs
measured a 1,721.468 ms median scalar seed and 596.624 ms with portable tiled
QConv (2.89x faster). Steady decode is unchanged because dependency scheduling
caches the CNN after the seed.

The GPU profiles attribute almost all of the old later-token time to `GPUWait`.
The automatic handoff reduced the paired full run by 7.64x and later-token mean
by 22.27x. It reported 43.7 KB of dirty prefixes and 1,005 KB of clean boundary
state. Native CPU is still faster on this integrated GPU because its seed is
also faster and it needs no ownership transition. These numbers diagnose this
graph and driver, not GPUs in general; a persistent, fused device decoder could
change the tradeoff for a larger token workload.

## Benchmark Flags

For TFLite-style timing:

```bash
examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=photo.jpg \
  --max-det 5 \
  --num_threads 4 \
  --warmup_runs 5 \
  --num_runs 20
```

`--num_threads` sets `VolvoxAIEngineOptions.cpu_threads` before engine
configuration; omitting it keeps the runtime's automatic worker policy.
`--debug` prints graph load, build, node backend, and per-op timing logs.

Direct W8A8 correctness-preflighted kernel benchmarks are available separately:

```bash
make benchmark_native   # w8a8, qsdpa and gemm_f32 kernel benchmarks
make test_native        # includes the qsdpa correctness test
```

The first command reports the 51-call TinyReceipt steady-row dense proxy, exact
`M=402`/`M=192` seed dense shapes, exact activation shapes, and a convolution
proxy. The second reports exact TinyReceipt encoder-self, decoder-self, and
decoder-cross QSDPA shapes plus the `Q=1` fallback. These are kernel benchmarks,
not model accuracy tests. Current host-labeled figures and the separate heldout
sample timing are recorded in
[the W8A8 benchmark harness](operation_list.md#w8a8-benchmark-harness).

See [efficientdet_tflite_vs_volvoxai.md](efficientdet_tflite_vs_volvoxai.md) for
the current EfficientDet benchmark methodology and results.

## Android Cross-Compile

```bash
export ANDROID_NDK="$HOME/Android/Sdk/ndk/<version>"
cmake -S . -B build/android -DCMAKE_C_COMPILER=clang \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29
cmake --build build/android
```

The Android profile is inference-only and uses the inference shader pack. The
target requires an Android NDK and defaults to arm64 API 29, the first API level
used by its NNAPI device discovery. Override `ANDROID_HOST_TAG`,
`ANDROID_TARGET`, `ANDROID_API`, or `ANDROID_CC` for another NDK layout or
target. It fails before compilation when the configured NDK compiler is absent.

Push only the executable and model directory; shader files are already embedded:

```bash
adb push volvoxai_android /data/local/tmp/volvoxai
adb shell "cd /data/local/tmp && VOLVOX_NUM_THREADS=2 ./volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 --vulkan --debug"
```

ARM/NEON builds default to two CPU threads. Use one thread for stable profiling and
test two threads for throughput; using all cores is often slower because of big/little
scheduling and thermal limits.

## Service Runtime

The Rust crate under `runtime/` builds a `libvolvoxai.so` Synurang service wrapper
around the C engine. See [../runtime/README.md](../runtime/README.md) for the FFI
service ABI, build commands, and current RPC coverage.
