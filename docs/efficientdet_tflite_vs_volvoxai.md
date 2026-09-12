# EfficientDet Lite0: TFLite vs VolvoxAI Native

> **Historical measurement record.** The results and commands on this page are
> tied to the listed machines and artifacts. `volvoxai-tasks` is a
> model-specific generated-service client; it does not define another engine
> lifecycle or public C API.

This document compares MediaPipe EfficientDet Lite0 int8, float16-source, and
float32-source packages on the Linux CPU path, and records the Android
OpenGL GPU result for the float32 package.

This page covers the CPU/Vulkan/OpenGL comparison. CUDA build, numerical, and
hardware validation requirements are maintained in
[cuda.md](cuda.md#validation).

Machine/runtime:

- CPU: AMD Ryzen 5 5600U with Radeon Graphics, single-thread runs
- TFLite: `/tmp/benchmark_model`, XNNPACK enabled, `--num_threads=1`,
  `--warmup_runs=2`, `--num_runs=20`
- VolvoxAI task example: `examples/target/bin/volvoxai-tasks detect ...
  --num_threads 1 --warmup_runs 2 --num_runs 20`
- Test image for VolvoxAI: `/tmp/volvox_dog.jpg`

Android GPU runtime:

- Device: Samsung SM-A528N, Qualcomm `lahaina`, Adreno 642L, Android SDK 34
- TFLite: `/data/local/tmp/benchmark_model`, GPU delegate, `--gpu_backend=gl`,
  `--warmup_runs=2`, `--num_runs=20`
- VolvoxAI: Android arm64 build of `examples/native_task_cli/`, `--opengl`, OpenGL ES 3.2,
  `--warmup_runs 2`, `--num_runs 20`
- Test image for VolvoxAI: `/data/local/tmp/volvoxai_gltest/volvoxai_object_test.jpg`

Important caveats:

- TFLite `benchmark_model` measures generated tensor input. The VolvoxAI task
  example's `detect forward` timing is the graph forward after the image input
  is loaded.
- VolvoxAI prints raw detection tensors. This is a speed comparison, not a validation of
  MediaPipe postprocessing/NMS correctness.
- TFLite GPU defaults to allowing lower precision. The Android section includes both
  the default GPU delegate result and a `--gpu_precision_loss_allowed=false` rerun.
- The TFLite float16 model uses float32 input/output and XNNPACK reports F32 kernels on
  CPU. It is not an FP16 CPU-kernel speed test.
- Direct TFLite export keeps VolvoxAI tensors in NHWC and stores Conv weights in
  TFLite-native layouts: regular Conv2D as `OHWI`, depthwise Conv2D as `1HWO`. The
  native CPU path prepares a per-node HWIO/HWCM compute cache during model compilation, so the
  artifact stays TFLite-shaped without making the hot Conv loops stride through OHWI.
- The float16-source package stores floating weights as safetensors `F16`. Native CPU
  keeps the artifact half-sized and prepares a widened Conv weight cache during model compilation
  for the default `cpu-f16w-pack` path. On this AVX2 CPU this is not native FP16
  arithmetic; it is half-size storage plus one-time FP32 widening.

## Artifacts

| Package | Source TFLite | Optional ONNX | Volvox weights | graph.json | Volvox Graph contract |
| --- | ---: | ---: | ---: | ---: | --- |
| `models/efficientdet_lite0_int8` | `4.4M` | `4.0M` | `3.4M` | `148K` | direct TFLite NHWC, quantized `QConv2D`, `I8` OHWI/1HWO weights |
| `models/efficientdet_lite0_fp16` | `7.0M` | `13M` | `6.4M` | `108K` | direct TFLite NHWC, `F16` OHWI/1HWO weights |
| `models/efficientdet_lite0_fp32` | `14M` | `13M` | `13M` | `108K` | direct TFLite NHWC, FP32 OHWI/1HWO weights |

## Detection Score Parity

These scores use byte-identical inputs produced by the native task example's
`image_io.c` path. TFLite Runtime 2.14.0 used XNNPACK with one thread; ONNX
Runtime 1.23.2 used the CPU execution provider. Every run selected anchor 19011,
class 17 (`dog`) or anchor 19013, class 16 (`cat`), as appropriate.

| Package | Image | TFLite | ONNX | VolvoxAI CPU |
| --- | --- | ---: | ---: | ---: |
| int8 | dog | `91.796875%` | `91.796875%` | `91.7969%` |
| int8 | cat | `81.640625%` | `81.640625%` | `80.8594%` |
| fp16 | dog | `91.165912%` | `91.165906%` | `91.0401%` |
| fp16 | cat | `77.409363%` | `77.409363%` | `77.4408%` |
| fp32 | dog | `91.170549%` | `91.170549%` | `91.0444%` |
| fp32 | cat | `77.464628%` | `77.464622%` | `77.5092%` |

The float ONNX and TFLite score tensors agree within `9e-7`. The int8 VolvoxAI
cat score differs by one output quantization bin, while preserving the same
winning anchor and class. The ONNX files are local tf2onnx conversions of the
MediaPipe TFLite sources, not separately published vendor models.

The fp16-source package preserves half-precision storage in safetensors:

```text
models/efficientdet_lite0_fp16/model.safetensors  F16 tensors 253
Conv2D backend=cpu-f16w-pack
```

The int8 package preserves quantized Conv nodes and quantization metadata:

```text
models/efficientdet_lite0_int8/graph.json  QConv2D 182
models/efficientdet_lite0_fp16/graph.json  Conv2D  182
models/efficientdet_lite0_fp32/graph.json  Conv2D  182
models/efficientdet_lite0_fp16/graph.json  Transpose 0
models/efficientdet_lite0_fp32/graph.json  Transpose 0
```

The direct TFLite configs report:

```text
internal_layout=NHWC
conv_weight_layout=OHWI
depthwise_weight_layout=1HWO
```

The first Conv/QConv weights match TFLite shapes:

```text
w0 regular Conv2D   (32, 3, 3, 3)    # OHWI
w2 depthwise Conv2D (1, 3, 3, 32)    # 1HWO
```

The int8 VolvoxAI debug log confirms that the typed quantized path is active:

```text
QConv2D backend=cpu-qconv-w8a8
QAdd backend=cpu-qadd-w8a8
```

Conv nodes keep typed quantized NHWC activations and int8 weights with int32
accumulation/requantization. Output heads leave the quantized island only at an
explicit `DequantizeLinear` boundary.

## Export Commands

Download the float models:

```bash
mkdir -p models/efficientdet_lite0_fp16 models/efficientdet_lite0_fp32

curl -L -o models/efficientdet_lite0_fp16/efficientdet_lite0_float16.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite

curl -L -o models/efficientdet_lite0_fp32/efficientdet_lite0_float32.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite

cp models/efficientdet_lite0_int8/labels.txt models/efficientdet_lite0_fp16/labels.txt
cp models/efficientdet_lite0_int8/labels.txt models/efficientdet_lite0_fp32/labels.txt
```

Export TFLite directly to VolvoxAI:

```bash
python3 tools/export_safetensors.py \
  --model models/efficientdet_lite0_fp16/efficientdet_lite0_float16.tflite \
  --out models/efficientdet_lite0_fp16/model.safetensors \
  --image-normalization input0=zero-one \
  --output-name scores --output-name boxes

python3 tools/export_safetensors.py \
  --model models/efficientdet_lite0_fp32/efficientdet_lite0_float32.tflite \
  --out models/efficientdet_lite0_fp32/model.safetensors \
  --image-normalization input0=zero-one \
  --output-name scores --output-name boxes
```

The exporter auto-selects `F16` storage for the float16 TFLite model because its constant
weights are stored as half precision. Use `--weight-dtype float32` to force FP32 artifact
storage, or `--weight-dtype float16` to force half-precision storage.

The direct float exports reported:

```text
float16: TFLite ops=516 folded_dequantize=253 lowered_nodes=263 optimized_nodes=262 weights=253
float32: TFLite ops=263 folded_dequantize=0 lowered_nodes=263 optimized_nodes=262 weights=253
```

## Benchmark Commands

TFLite:

```bash
/tmp/benchmark_model \
  --graph=models/efficientdet_lite0_int8/efficientdet_lite0.tflite \
  --num_threads=1 \
  --use_xnnpack=true \
  --warmup_runs=2 \
  --num_runs=20 \
  --min_secs=0 \
  --max_secs=30
```

Add `--enable_op_profiling=true --op_profiling_output_mode=stdout` to reproduce
the TFLite op-sum tables below.

VolvoxAI:

```bash
make -C examples native_task_cli

examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=/tmp/volvox_dog.jpg \
  --max-det 5 \
  --num_threads 1 \
  --warmup_runs 2 \
  --num_runs 20
```

Add `--debug` to reproduce the VolvoxAI op-sum and load/init tables below.

For the float-source packages, replace only the model directory with
`models/efficientdet_lite0_fp16` or `models/efficientdet_lite0_fp32`. Generated
packages declare `raw-255` normalization for int8 and `zero-one` for fp16/fp32,
so the task CLI selects the correct preprocessing automatically. An explicit
`--image-normalize` remains available as an override.

Android TFLite OpenGL GPU:

```bash
adb shell 'cd /data/local/tmp && ./benchmark_model \
  --graph=/data/local/tmp/tflite_gltest/efficientdet_lite0_float32.tflite \
  --use_gpu=true \
  --gpu_backend=gl \
  --warmup_runs=2 \
  --warmup_min_secs=0 \
  --num_runs=20 \
  --min_secs=0 \
  --max_secs=60 \
  --num_threads=1'
```

Add `--gpu_precision_loss_allowed=false` for the strict fp32 GPU rerun.

Android VolvoxAI OpenGL ES task-example binary:

```bash
adb shell 'cd /data/local/tmp/volvoxai_gltest && ./volvoxai-tasks detect \
  models/efficientdet_lite0_fp32 \
  --image input0=volvoxai_object_test.jpg \
  --max-det 3 \
  --opengl \
  --warmup_runs 2 \
  --num_runs 20'
```

## High-Level Result

Headline timings are timed first/avg/min/max after two warmup runs. They omit
TFLite op profiling and VolvoxAI `--debug` logging.

| Source model | TFLite first | TFLite avg | TFLite min | TFLite max | VolvoxAI first | VolvoxAI avg | VolvoxAI min | VolvoxAI max | Gap by avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| int8 | `26.758 ms` | `25.943 ms` | `25.234 ms` | `28.078 ms` | `37.589 ms` | `38.299 ms` | `37.589 ms` | `39.431 ms` | `1.48x` |
| float16 | `25.816 ms` | `25.857 ms` | `25.115 ms` | `27.432 ms` | `29.486 ms` | `30.895 ms` | `29.486 ms` | `32.473 ms` | `1.19x` |
| float32 | `25.355 ms` | `26.024 ms` | `25.004 ms` | `26.950 ms` | `29.361 ms` | `30.805 ms` | `29.361 ms` | `33.251 ms` | `1.18x` |

TFLite op profiling and VolvoxAI `--debug` summaries from matching 20-run jobs.
For VolvoxAI, the op row averages the last 20 debug summary blocks and excludes
the two warmup forwards.

| Source model | TFLite profiled op avg | TFLite nodes | VolvoxAI debug op avg | VolvoxAI nodes |
| --- | ---: | ---: | ---: | ---: |
| int8 | `25.579 ms` | `196` | `37.078 ms` | `265` |
| float16 | `26.326 ms` | `193` | `30.770 ms` | `262` |
| float32 | `26.255 ms` | `193` | `30.987 ms` | `262` |

Read:

- TFLite/XNNPACK is faster on one CPU thread. The int8 gap is `1.48x`.
- VolvoxAI int8 uses quantized `QConv2D` execution, but it is slower than the
  float-source native path in this run.
- TFLite float16 and float32 are almost the same on CPU because XNNPACK reports F32 kernels
  for both.
- VolvoxAI float16-source stores `F16` weights and defaults to a prepared
  `cpu-f16w-pack` path. Direct half-weight consumption is not used for this
  comparison because this AVX2 CPU lacks native FP16 arithmetic.
- The direct TFLite paths have no ONNX-derived Transpose nodes. The float
  graph has 262 nodes, and the int8 graph is direct NHWC TFLite with 265 nodes.
- No-debug single-thread runs are `38.299 ms` for int8 and about `31 ms` for the
  float-source packages.

## TFLite Op Sums

These come from TFLite operator-profiling runs with the same thread/warmup/run counts.
The total row is the profiler's 20-run average, not the sum of rounded table rows.

| Op group | int8 | float16-source | float32-source |
| --- | ---: | ---: | ---: |
| Pointwise 1x1 / GEMM, 101 calls | `17.912 ms` | `18.009 ms` | `18.052 ms` |
| Depthwise Conv, 80 calls | `4.418 ms` | `5.197 ms` | `5.124 ms` |
| Stem Conv, 1 call | `1.758 ms` | `1.019 ms` | `0.951 ms` |
| Unary / activation | `0.871 ms` | `1.019 ms` | `1.043 ms` |
| Copy / tensor movement | `0.153 ms` | `0.711 ms` | `0.708 ms` |
| Binary Add | `0.103 ms` | `0.150 ms` | `0.159 ms` |
| ResizeNearest | `0.072 ms` | `0.085 ms` | `0.086 ms` |
| MaxPool | `0.017 ms` | `0.059 ms` | `0.056 ms` |
| Quantize | `0.208 ms` | n/a | n/a |
| Total profiled op avg | `25.579 ms` | `26.326 ms` | `26.255 ms` |

TFLite int8 uses XNNPACK quantized kernels:

```text
Fully Connected (NC, QS8, QC8W) GEMM
Convolution (NHWC, QC8) DWConv
Convolution (NHWC, QC8) IGEMM
```

The float16-source and float32-source TFLite runs both report:

```text
Fully Connected (NC, F32) GEMM
Convolution (NHWC, F32) DWConv
Convolution (NHWC, F32) IGEMM
```

## VolvoxAI Op Sums

These average the 20 timed `--debug` forwards; the two warmup forwards are excluded.

| Op group | int8 package | float16-source package | float32-source package |
| --- | ---: | ---: | ---: |
| Conv aggregate | `QConv2D 34.80 ms` | `Conv2D cpu-f16w-pack 28.85 ms` | `Conv2D cpu-pack 28.93 ms` |
| Concat | `1.25 ms` | `1.37 ms` | `1.49 ms` |
| Add | `0.42 ms` | `0.33 ms` | `0.34 ms` |
| MaxPool2D | `0.40 ms` | `0.14 ms` | `0.14 ms` |
| ResizeNearest2D | `0.04 ms` | `0.08 ms` | `0.08 ms` |
| QuantizeLinear | `0.18 ms` | n/a | n/a |
| DequantizeLinear | aliased after Concat | n/a | n/a |
| Total profiled op avg | `37.08 ms` | `30.77 ms` | `30.99 ms` |

VolvoxAI load/compile timings from representative debug runs:

| Package | load weights | validate Graph | compile Model |
| --- | ---: | ---: | ---: |
| int8 | `3.523 ms` | `8.906 ms` | `14.800 ms` |
| float16-source | `7.216 ms` | `7.503 ms` | `36.135 ms` |
| float32-source | `12.595 ms` | `6.758 ms` | `40.580 ms` |

Float-source compilation includes layout/precision preparation:

```text
fp16: prepack_conv weights=182 biases=182 pointwise_packs=101 igemm_indirs=1 19.720 ms
fp32: prepack_conv weights=182 biases=0   pointwise_packs=101 igemm_indirs=1 19.535 ms
```

## CPU Gap

The int8 VolvoxAI path is a real quantized graph island for most Conv nodes. It uses a
wide AVX2 pointwise tile plus an exact AVX2 byte-dot path for `input_zero_point = -128`
pointwise layers, vectorized same-quant MaxPool2D, vectorized contiguous QuantizeLinear,
and aliasing for final no-op DequantizeLinear nodes after Concat. It is not an
XNNPACK-class CPU implementation.

One-thread blockers:

1. Pointwise and depthwise microkernels are much simpler than XNNPACK. They need
   architecture-specific dot-product paths, better unrolling, and better cache blocking.
2. Depthwise Conv needs XNNPACK-style indirection/bounds handling to remove per-MAC
   padding math.
3. Weight packing exists, but it is not microkernel-specific enough for VNNI/AVX512 or
   ARM `sdot`/`udot`.
4. Some graph tails leave the quantized island and produce FP32 output tensors.
5. The fp16-source export preserves `F16` weight storage and prepares those weights once
   for CPU Conv2D. This follows the guide's weight-preparation principle on AVX2, but
   true hardware FP16 arithmetic requires ARM FP16, AVX512-FP16, or a GPU path.

## Native GPU Status

The figures below cover AMD Vulkan/OpenGL. For the CUDA backend and NVIDIA comparison, see
[Native CUDA Backend](cuda.md).

OpenGL waits before context execution completes, so reported execution time includes real
queued GPU work rather than only CPU-side dispatch/enqueue time.

Linux fp32 EfficientDet Lite0 timings on the Ryzen 5 5600U machine:

| Engine | Backend | Avg, warmup 2 / runs 20 | Notes |
| --- | --- | ---: | --- |
| VolvoxAI | CPU fp32 | `29.990 ms` | `--num_threads 1` |
| VolvoxAI | Vulkan graph | `45.523 ms` | AMD RADV Renoir, vector/tiled Conv2D shaders |
| VolvoxAI | OpenGL graph | `40.256 ms` | AMD Renoir Mesa, vector/tiled Conv2D shaders |

Representative corrected debug summaries show that most elapsed time is in the final GPU
wait, not in CPU enqueue:

```text
OpenGL: context execute 46.823 ms, GPUWait 40.46 ms, Conv2D enqueue 3.42 ms
Vulkan: context execute 43.100 ms, GPUWait 39.41 ms, Conv2D enqueue 1.67 ms
```

## Android OpenGL GPU Result

Run on the attached SM-A528N / Adreno 642L device with the float32 EfficientDet Lite0
package.

| Engine | Backend | First | Avg | Min | Max | Gap vs TFLite default |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TFLite | GPU delegate, OpenGL, default precision | `138.365 ms` | `127.211 ms` | `90.464 ms` | `139.163 ms` | `1.00x` |
| TFLite | GPU delegate, OpenGL, fp32 strict | `120.499 ms` | `130.962 ms` | `120.499 ms` | `139.626 ms` | `1.03x` |
| VolvoxAI | OpenGL ES 3.2, vector/tiled Conv2D | `118.918 ms` | `144.332 ms` | `118.918 ms` | `153.026 ms` | `1.13x` |

The OpenGL run uses Naga-generated GLES shaders for depthwise, pointwise, and the
EfficientDet stem Conv2D, and is `1.13x` slower than the TFLite OpenGL default-precision
baseline.

TFLite log confirmation:

```text
Replacing 263 out of 263 node(s) with delegate (TfLiteGpuDelegateV2)
yielding 1 partitions for subgraph 0
Initialized OpenGL-based API
Created 1 GPU delegate kernels
Inference (avg): 127211 us
```

VolvoxAI log confirmation:

```text
[VolvoxAI GPU] OpenGL Compute initialized:
Qualcomm / Adreno (TM) 642L / OpenGL ES 3.2
[bench] detect: first=118.918 avg=144.332 min=118.918 max=153.026 ms
```

The `--debug` run is not the benchmark average, but it identifies where the time goes:

```text
[debug] context execute nodes=262 159.649 ms
[debug] --- op time summary (by total) ---
[debug]   GPUWait            114.84 ms  (n=1)
[debug]   Conv2D              11.90 ms  (n=182)
[debug]   Add                  2.51 ms  (n=42)
[debug]   MaxPool2D            0.76 ms  (n=14)
[debug]   ResizeNearest2D      0.62 ms  (n=12)
[debug]   Concat               0.18 ms  (n=2)
[debug]   Reshape              0.13 ms  (n=10)
[debug]   TOTAL              130.94 ms
```

## Android OpenGL Gap Analysis

The OpenGL path selects vectorized and tiled Naga-generated GLES shaders for
EfficientDet's dominant Conv shapes:

```text
stem Conv2D:       conv2DRegularC3Out16
depthwise Conv2D:  conv2DDepthwise8
pointwise Conv2D:  conv2DPointwise16Tile
head pointwise:    conv2DPointwise8Vec4 / conv2DPointwise8Vec2
fallback Conv2D:   conv2D / conv2DPointwise8
```

The `1.13x` gap is small but not a measurement artifact. The VolvoxAI OpenGL path is a
per-node compute backend, while TFLite's GPU delegate compiles the whole graph into one
delegated partition.

Reasons for the gap:

1. TFLite delegates the whole EfficientDet graph into one GPU partition and creates one
   GPU delegate kernel. VolvoxAI executes the exported graph as 262 native nodes.
2. VolvoxAI dispatches each graph op separately. Each dispatch binds buffers, launches
   compute, and inserts a shader-storage memory barrier. That is correct but expensive for
   hundreds of small mobile-GPU kernels.
3. Each op creates a small uniform parameter buffer for that dispatch. This is minor
   compared with `GPUWait`, but it is unnecessary per-node overhead.
4. The OpenGL backend has only kernel-level specialization. It does not fuse larger
   EfficientDet blocks across Add/Resize/Conv boundaries the way a graph delegate can.
5. The Android OpenGL path runs fp32 GLES compute shaders. TFLite default precision can use
   lower precision, but the strict fp32 TFLite rerun is `130.962 ms`, so precision
   alone is not the reason.

The debug profile's `Conv2D 11.90 ms` row is CPU-side enqueue/profile time, not real GPU
compute time. The real queued GPU execution appears in `GPUWait 114.84 ms`; enqueue-only
OpenGL timing omits it.

Work to consistently match or beat the TFLite GPU delegate:

1. Fuse EfficientDet blocks at graph compile time instead of dispatching every op as a
   separate shader.
2. Remove per-dispatch memory barriers when a graph-level dependency schedule can prove
   ordering.
3. Reuse persistent parameter buffers or push constants/equivalent small-uniform storage.
4. Add a fp16/mobile GPU path and benchmark it separately from fp32.
