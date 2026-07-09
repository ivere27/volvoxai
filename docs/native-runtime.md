# Native Runtime

`native/` is a freestanding C runtime that runs the same Volvox blueprint package as
the browser runtime: `config.json` plus `.safetensors`. It has no static GPU SDK
dependency. Vulkan, OpenGL, Metal, and NNAPI are loaded dynamically when requested.

## Build

```bash
make build_native
./native/volvoxai --version
./native/volvoxai --help
```

The Docker build compiles the native binary with clang, AVX2/FMA on x86, pthreads,
and the CPU/GPU backend sources. The binary embeds package version, git commit,
and UTC build date from the Makefile build.

## Architecture

Key files:

- `engine.c`: parses the blueprint, builds tensors/nodes, owns the global engine context.
- `engine_runtime.c`: dispatches each node to CPU, Vulkan, OpenGL, Metal, or NNAPI.
- `graph_opt_fusion.c`: compile-time fusion pass for patterns such as Conv+ReLU6,
  chained Add, depthwise-to-pointwise, concat+sigmoid, and alias elision.
- `conv_f32_opt.c`: optimized FP32 Conv2D with weight prepacking and IGEMM.
- `quant_cpu_opt.c`: INT8 quantized CPU path for `QConv2D`, quantized Add, and
  quantization helpers.
- `kernels.c`: shared portable kernels used by both native and wasm32.
- `image_io.c`: PNG/JPEG decode through `stb_image.h` into NHWC tensors.
- `tokenizer.c` and `kie_runtime.c`: native BPE tokenizer and receipt KIE task runtime.

## Raw Tensor Runner

```bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 \
  --last-token 4 \
  --debug
```

Image-shaped inputs can be decoded directly:

```bash
./native/volvoxai run models/vision_model \
  --image image=receipt.png \
  --image-normalize zero-one \
  --output logits=out.f32
```

`--image-normalize` supports `zero-one`, `minus-one-one`, and `raw-255`.
Video, audio, camera, and streaming input should be handled by a frontend that feeds
decoded tensors to `run`.

## Task Wrappers

Task wrappers sit on top of the tensor runner:

```bash
./native/volvoxai classify models/classifier \
  --image image=photo.jpg \
  --logits logits \
  --labels labels.txt \
  --top-k 5

./native/volvoxai detect models/detector \
  --image image=receipt.jpg \
  --boxes boxes \
  --scores scores \
  --classes classes \
  --max-det 20

./native/volvoxai ctc models/ocr_line \
  --image image=line.png \
  --logits logits \
  --labels labels.txt \
  --blank 0

./native/volvoxai seq2seq models/encoder_decoder \
  --image image=receipt.jpg \
  --prompt "What is the first number of the store phone?" \
  --prompt-input q_tokens \
  --decoder-input y_tokens \
  --logits logits \
  --max-new 192

./native/volvoxai chat models/multimodal_chat \
  --prompt "Summarize this receipt" \
  --image image=receipt.jpg
```

`classify` ranks logits, `ctc` performs greedy CTC collapse, and `detect` prints or
writes named raw tensors because detector output layouts vary by exporter. `seq2seq`
runs a greedy encoder-decoder loop. `chat` is a user-facing alias for multimodal
seq2seq packages.

## Generation

TinyStories generation uses the same loaded graph repeatedly:

```bash
./native/volvoxai generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" \
  --max-new 50 \
  --debug
```

Weights and graph load once. Generation updates token inputs and reruns the node loop.
For language models, `--vulkan` uses the MatMul-focused path: large MatMul/Gemm/Linear
nodes run through the `linearF32` shader, while small decode-time MatMuls stay on CPU
to avoid dispatch overhead.

## Native GPU and NPU Backends

Driver loading is resolved at runtime:

- Vulkan (`--vulkan`): `libvulkan.so.1`, `libvulkan.so`, or `vulkan-1.dll`.
- OpenGL (`--opengl`): `libGL.so.1`, `opengl32.dll`, or macOS `OpenGL.framework`.
- Metal (`--metal`, macOS): default `MTLDevice` through the Objective-C runtime.
- NNAPI (`--nnapi`, Android build with `-DUSE_NNAPI`): Android Neural Networks API
  for large MatMul/FullyConnected offload.

If multiple accelerator flags are passed, priority is `--nnapi`, then `--vulkan`,
then `--metal`, then `--opengl`.

EfficientDet-style vision graphs have graph-resident Vulkan and OpenGL paths for
fused Conv2D/ReLU6, MaxPool2D, same-size Add, Clip, Sigmoid, 2x nearest upsample,
Concat, and reshape/identity aliasing. The native GPU graph path is FP32-only;
quantized `QConv2D` runs on CPU.

On software Vulkan/OpenGL stacks, dispatch synchronization can be slower than the
AVX2 CPU path. The GPU win is on real mobile or desktop GPUs.

## Benchmark Flags

For TFLite-style timing:

```bash
./native/volvoxai detect models/efficientdet_lite0_int8 \
  --image input0=photo.jpg \
  --image-normalize raw-255 \
  --boxes boxes \
  --scores scores \
  --max-det 5 \
  --num_threads 4 \
  --warmup_runs 5 \
  --num_runs 20
```

`--num_threads` maps to `VOLVOX_NUM_THREADS`. `--debug` prints graph load, build,
node backend, and per-op timing logs.

See [efficientdet_tflite_vs_volvoxai.md](efficientdet_tflite_vs_volvoxai.md) for
the current EfficientDet benchmark methodology and results.

## Android Cross-Compile

```bash
NDK=$ANDROID_SDK/ndk/<version>
CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
cd native && "$CC" -O3 -DUSE_NNAPI -pthread -Wno-unknown-attributes -I. \
  cJSON.c safetensors.c kernels.c quant_cpu_opt.c conv_f32_opt.c tensor_f32_opt.c \
  engine_runtime.c engine.c image_io.c kie_runtime.c \
  vulkan_engine.c opengl_engine.c tokenizer.c nnapi_engine.c main.c \
  -o volvoxai_android -lm -ldl -lneuralnetworks
```

Then push the binary, model directory, and native shader outputs to the device:

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
