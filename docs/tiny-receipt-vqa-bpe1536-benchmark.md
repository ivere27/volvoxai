# TinyReceiptVQA BPE1536 explicit-KV benchmark

TinyReceiptVQA supports only the producer's cache-enabled split ABI:

```text
source:  tiny_receipt_vqa_split_kv_onnx_v1
package: volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v1
```

The FP32 and static INT8 packages are imported directly from the producer ONNX
directory and use explicit F32 KV
caches in ONNX Runtime and every VolvoxAI route in this note. Native JavaScript
CPU and NNAPI are deliberately excluded.

The canonical imported package manifests are:

| Package | Manifest SHA-256 | Encoder nodes | Decoder nodes |
| --- | --- | ---: | ---: |
| FP32 | `5c2c32eee248f0991f30e6c2688ea458792cc28f07712593307f961cc937a816` | 409 | 236 |
| INT8 | `be1b68bd3a4615bc8e6f61b6480c83d34cc9e8090243f83f9cc75511ba39d9ff` | 480 | 271 |

These hashes identify the current re-export after the redundant secondary
graph discriminator was removed. The timed reports below remain bound
to the immediately preceding manifests (`875d9f90...` FP32 and `7eb27743...`
INT8). After deleting only that discriminator, all four graph documents are
semantically identical, every safetensors payload is byte-identical, and the
CPU strict smoke emits the same family/token decision. Therefore the retained
latencies remain execution-path evidence, but their content-addressed reports
are not relabelled as fresh measurements of the new manifest bytes.

See the [TinyReceipt example](../examples/tiny_receipt_vqa/README.md) for the
import and runtime contracts.

## Workload and timing rules

All tables use the same generated 672x320 grayscale input, normalized F32 input
SHA-256 `7a6f7eb153434868b1685c4fd96fc63f1004cae356d15bd58404dccdc2c15963`,
prompt `phone number last one`, requested family `phone`, active shape
`B=1/Q=8/M=218/T=5`, and greedy four-token decode. Every measured entry selected
family ID 0, emitted `[4, 1038, 5, 6]`, and proved the explicit-cache transition
`P=1 -> R=2 -> 3 -> 4 -> 5`. The initial `P=1` row is the blocked zero sentinel.

`Component` is the median of each sample's encoder time plus its four decoder
calls. It is not formed by adding independently computed column medians.
Encoder and decoder timings include execution, synchronization, shape binding,
and owned output snapshotting where the runtime exposes those phases. Model
loading, context/session creation, graph compilation, image preprocessing,
tokenization, and process startup are excluded. Process-wall time is therefore
not an inference-latency comparison.

The CPU and GPU reports have intentionally different lifecycle questions:

- The current CPU report measures fresh-session/runtime first execution.
  `--warmup 1` discards
  one complete matrix, but every measured ORT session and native/WASM child is
  new. Before warmup it also runs an untimed canonical dynamic-rebind proof for
  native CPU and WASM at both precisions.
- The current GPU report performs one untimed full request on the same runtime and encoder /
  decoder contexts immediately before every measured request. It resets the
  cache to the sentinel and proves token/cache parity. This removes lazy device
  initialization and first-shape compilation from the comparison.

The CPU and GPU reports use their `v1` schemas. Dynamic qualification uses
`volvoxai.tiny-receipt-dynamic-shape-qualification/v1`. The `explicit_kv_v1`
artifact name is the producer model/package ABI.

The current CPU and AMD GPU harnesses counterbalance tier order with
deterministic forward/reverse pairs and rotation, and record the exact order
for every matrix. Their spawned children inherit non-Volvox environment
variables, remove every uppercase `VOLVOX*` runtime override, and force common
nested-library thread variables to the requested harness thread count (one for
the one-core and GPU matrices, six for the six-core matrix); no such override
was present in these runs. The CPU ORT sessions remain in the parent Python
process, while the current AMD GPU policy covers every benchmark and
dynamic-qualification child.

The browser runners use fresh origins/profiles and serve every artifact with
`Cache-Control: no-store`. VolvoxAI preloads both graphs and contexts before
warmup; ORT fetches each selected ONNX model once into immutable bytes before
creating both sessions. Those setup phases remain outside execution timing.

## Physical W8A8 graph

The INT8 package is deliberately described as hybrid W8A8 because its public
cache/logit interface and numerically sensitive regions remain F32;
`complete_w8a8_fusion=false` is correct. Hybrid does not mean the large compute
fell back to FP32.

| Runtime operator | Encoder | Decoder |
| --- | ---: | ---: |
| `QConv2D` | 13 | 0 |
| `QLinear` | 26 | 33 |
| `QGemm` | 8 | 0 |
| `QBatchMatMul` | 14 | 18 |
| F32 `Conv2D` / `Linear` / `Gemm` / `MatMul` / `BatchMatMul` | 0 | 0 |
| `QuantizeLinear` / `DequantizeLinear` | 62 / 61 | 53 / 48 |
| `GroupNorm` / `SiLU` | 13 / 13 | 0 / 0 |
| `LayerNorm` / `Softmax` / `GELU` | 12 / 6 / 8 | 13 / 8 / 5 |

The exporter migrated all four decoder BatchMatMul regions previously left at
the cache boundary; the decoder now has 18 physical `QBatchMatMul` nodes and no
F32 `BatchMatMul`. It also folded 26 encoder and 33 decoder immutable F32 biases
into I32 quantized accumulators. Direct, redundant `DequantizeLinear ->
QuantizeLinear` round trips are absent. Public memory, cross/self KV caches,
logits, LayerNorm, Softmax, GELU, and the encoder GroupNorm/SiLU islands remain
F32 by policy.

The generic GroupNorm/SiLU byte-island migration remains opt-in and is not
enabled for this package. A reproduced local 32-request qualification tied to
producer INT8 encoder/decoder SHA-256 values `a5be30f7...` / `75437bee...`
changed greedy tokens in 3 of 32 requests (family selection stayed 32/32), with
up to 32.46% pre-SiLU saturation and 39.66% terminal byte mismatch. This is a
local qualification, not a checked-in corpus accuracy report, so it supports
rejecting that migration but not a corpus-level accuracy claim.

## CPU: one physical core

The process is pinned to one logical CPU that maps to one physical core.
ONNX Runtime and native C use one requested execution thread. VolvoxAI WASM has
one engine thread and zero workers.

| Runtime | Precision | Encoder | Seed | Steady / token | Decoder total | Component | x ORT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU | FP32 | 247.790 | 2.618 | 2.468 | 9.834 | 257.520 | 1.000 |
| VolvoxAI native C CPU | FP32 | 272.356 | 5.184 | 3.299 | 14.984 | 286.967 | 1.114 |
| VolvoxAI WASM | FP32 | 1095.780 | 50.177 | 60.533 | 231.775 | 1325.460 | 5.147 |
| ONNX Runtime CPU | INT8 | 147.688 | 1.570 | 1.004 | 4.583 | 153.290 | 1.000 |
| VolvoxAI native C CPU | INT8 | 188.372 | 4.320 | 4.011 | 16.350 | 204.812 | 1.336 |
| VolvoxAI WASM | INT8 | 588.990 | 60.477 | 61.127 | 244.959 | 839.373 | 5.476 |

ONNX Runtime remains faster at one thread because the thread count limits
parallel workers, not graph quality or microkernel quality. ORT optimizes and
packs during session creation outside the execution timer and uses MLAS
packing/cache blocking. Native C includes binding commit and exact output
snapshot work and, on this pre-VNNI CPU, uses an exact full-range U8/S8 AVX2
route that avoids `VPMADDUBSW` I16 saturation. The latter costs more arithmetic
than a reduced-range path but preserves the authored quantization contract.

The WASM encoder gap is predominantly provider compute, not just binding. The
FP32 encoder medians split into 18.235 ms binding and 1076.945 ms provider work;
INT8 splits into 40.029 and 548.631 ms. WASM uses one thread and SIMD128, while
native uses wider AVX2 plus a multithread-capable engine. Its decoder also pays
for each new `P/R` signature, JavaScript/WASM boundary work, and exact host
snapshots. Explicit KV is active in all rows, so the gap is not a missing-cache
failure.

## CPU: six-physical-core process envelope

This report pins `0,2,4,6,8,10`, which topology maps to six physical cores, and
requests six ORT/native threads. WASM remains one engine thread; wider affinity
only gives Node/V8 auxiliary threads more scheduling room and is not WASM
inference-worker scaling.

| Runtime | Precision | Encoder | Seed | Steady / token | Decoder total | Component | x ORT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU | FP32 | 79.891 | 2.495 | 3.415 | 12.741 | 91.859 | 1.000 |
| VolvoxAI native C CPU | FP32 | 111.926 | 5.090 | 3.322 | 15.057 | 126.699 | 1.379 |
| VolvoxAI WASM | FP32 | 1048.073 | 25.434 | 22.522 | 93.427 | 1138.566 | 12.395 |
| ONNX Runtime CPU | INT8 | 62.889 | 1.811 | 2.565 | 9.506 | 72.395 | 1.000 |
| VolvoxAI native C CPU | INT8 | 95.437 | 4.351 | 3.902 | 16.080 | 111.517 | 1.540 |
| VolvoxAI WASM | INT8 | 465.993 | 31.331 | 28.683 | 117.395 | 584.616 | 8.075 |

From the one-core to six-core envelope, component latency improves 2.803x /
2.117x for ORT FP32/INT8 and 2.265x / 1.837x for native C. Three repeats are
enough to preserve a reproducible observation, not to characterize all host
noise or deployment tail latency.

## Physical AMD GPU observation

The current GPU report used a physical AMD Renoir GPU for Vulkan and OpenGL. The
isolated ORT and VolvoxAI browser processes both reported the normalized
WebGPU adapter class `{vendor: amd, architecture: gcn-5}`. WebGPU exposes no
stable physical-adapter identifier here, so the report attests the adapter
class match, not that both processes opened the same physical adapter.

The direct WebGPU comparison is ONNX Runtime Web 1.27.0 with WebGPU plus CPU
partitions against strict VolvoxAI WebGPU. All values are median milliseconds.

| Runtime route | Precision | Encoder | Seed | Steady / token | Decoder total | Component | x ORT Web |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime Web 1.27.0 WebGPU + CPU partitions | FP32 | 216.500 | 71.600 | 52.167 | 235.000 | 452.700 | 1.000 |
| VolvoxAI strict WebGPU | FP32 | 601.800 | 36.400 | 31.167 | 130.500 | 731.800 | 1.617 |
| ONNX Runtime Web 1.27.0 WebGPU + CPU partitions | INT8 | 1029.000 | 447.900 | 454.233 | 1813.200 | 2842.200 | 1.000 |
| VolvoxAI strict WebGPU | physical W8A8 | 615.900 | 51.500 | 55.800 | 219.100 | 832.300 | 0.293 |

ORT's optimized-session diagnostics report exact aggregate provider-assignment
counts, not an exact per-operation CPU attribution: FP32 encoder CPU 70 /
WebGPU 504 and decoder CPU 3 / WebGPU 244; INT8 encoder CPU 282 / WebGPU 1223
and decoder CPU 133 / WebGPU 676. The pinned ONNX Runtime Web 1.27.0 build
cannot instantiate any of these four graphs as strict all-WebGPU. Missing FP32
Conv or MatMul kernels are not the cause. The shipped ORT operator table itself
marks `Reshape` and `Shape` as having no GPU kernel, and the captured missing-
kernel events contain both operations plus other shape/control operations.
Inspection of the model's INT64/BOOL shape/control tensors explains additional
type-constrained CPU islands. INT8 additionally has no registered WebGPU
`QuantizeLinear` kernel. This is a report/table/model cross-check, not exact
per-node attribution. The reported `unsupportedKernelEvents` values are repeated capability-
probe log events, not counts of nodes, executions, transfers, or partition
boundaries. VolvoxAI requires WebGPU and reports fallback zero.

Both harnesses reuse public KV GPU buffers and perform zero KV readbacks in the
measured request. ORT exposes eight cross-cache and eight present-cache outputs
as `gpu-buffer` tensors and passes those tensor objects forward. Its internal
transfers across WebGPU/CPU execution-provider partitions are not attested.
Separate untimed qualification requests read caches back and prove token and
cache-transition correctness.

The former FP32 encoder anomaly was a model-neutral WebGPU selector defect.
This encoder has thirteen groups=1 regular 3x3 FP32 convolutions, about 8.190
GMAC in total, and every output-channel count is divisible by 16. They
previously used the scalar `conv2D` schedule: 155,520 workgroups and 9,953,280
invocations. The browser compiler now selects the shared regular-out16 kernel,
which emits 9,720 workgroups and 622,080 invocations and shares each input load
across sixteen adjacent output channels. Every one of the five measured FP32
samples records exactly thirteen `webgpu.conv2d.regular-out16` encoder tactics.
An actual hardware WebGPU correctness test covering asymmetric padding, stride,
and dilation agrees with the CPU reference within `2.98e-8`; bounded-domain
tests execute B=1 -> 2 -> 1 in one context and retain the same precompiled
regular-out16 pipeline while rewriting exact shape metadata.

Against the previous same-harness observation, the VolvoxAI FP32 encoder fell
from 6533.6 ms to 601.8 ms, a 10.86x speedup; component latency fell from
6661.2 ms to 731.8 ms. The remaining FP32 component gap is 1.617x, concentrated
in the encoder: 2.780x ORT, while VolvoxAI's decoder total is 0.555x ORT. Its
601.8 ms encoder is also close to the current strict Vulkan result, 594.359 ms.
Compilation is warmed and measured execution has no KV readback. Host-side
execution telemetry remains only 8.4 ms shape binding, 1.2 ms provider enqueue,
and 10.1 ms through submission, so the remaining encoder interval is queued GPU
compute plus the required small control-output readback, not dynamic rebind.
ORT retains mature graph fusion, layout planning, and kernel/tactic selection;
VolvoxAI still submits a fine-grained model-neutral graph without equivalent
fusion. Per-kernel GPU timestamps are still required for an exact residual
breakdown.

ORT Web INT8 is 6.278x its FP32 component time. Its much larger CPU/WebGPU
partition counts and unsupported-`QuantizeLinear` diagnostics (398 encoder,
260 decoder events) are consistent with graph fragmentation and boundary
overhead. Those repeated capability-probe events are not executed-node or
transfer counts, so this report does not claim that every event denotes a CPU
node or identify a measured internal-copy cost. VolvoxAI's packed-dot4 W8A8
route avoids that ORT partition pattern and records 0.293x the ORT Web INT8
component time. Its encoder is 0.599x and its decoder total is 0.121x the
corresponding ORT Web INT8 medians.

This INT8 reversal is therefore not evidence that VolvoxAI generally beats an
optimized ORT all-WebGPU W8A8 implementation. It compares ORT's producer QDQ
graph partitioned across WebGPU and CPU with VolvoxAI's strict physical W8A8
graph. VolvoxAI routes 99.72% of the encoder Conv2D MACs through packed-dot
tiled QConv; only the first 3x3x1 convolution is below the packed-dot reduction
threshold. ORT's strict WebGPU probes reject both graphs. A strict physical
W8A8 ORT WebGPU route is not available in this measurement.

Vulkan and OpenGL have no ONNX Runtime native peer in this harness, so these
strict VolvoxAI measurements are observations rather than ORT ratios:

| Runtime route | Precision | Encoder | Seed | Steady / token | Decoder total | Component | ORT native peer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| VolvoxAI strict Vulkan | FP32 | 594.359 | 16.221 | 16.451 | 65.598 | 660.074 | N/A |
| VolvoxAI strict OpenGL | FP32 | 3764.764 | 15.726 | 15.794 | 63.782 | 3828.865 | N/A |
| VolvoxAI strict Vulkan | physical W8A8 | 645.542 | 38.341 | 38.318 | 152.888 | 799.635 | N/A |
| VolvoxAI strict OpenGL | physical W8A8 | 647.283 | 42.104 | 42.754 | 170.266 | 820.772 | N/A |

Every native row requires its named backend and reports `fallback=0;missing=0`.
The Vulkan device reports `packedInt8Dot=false` and therefore uses its
tiled/scalar W8A8 route; the OpenGL evidence has no packed-dot tactic counter.
The slow OpenGL FP32 encoder is not a dynamic-shape or fallback result, but the
matrix does not isolate the responsible kernel or driver cost.

ONNX Runtime also offers a [native WebGPU plugin](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)
that can run through Dawn's Vulkan backend on Linux. That remains WebGPU over
Dawn rather than a direct Vulkan execution provider, just as ORT Web's WebGL
route is not a native OpenGL execution provider, so neither is used as a
same-backend denominator here.

## RTX 3090 CUDA observation

The values below are retained from an earlier RTX 3090 run and were not
remeasured after the current harness updates. They are not a fresh current
qualification; the report records the original settings and provenance.

| Runtime route | Precision | Encoder | Seed | Steady / token | Decoder total | Component | x ORT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CUDA-first + CPU fallback | FP32 | 5.427 | 1.774 | 2.096 | 8.054 | 13.482 | 1.000 |
| VolvoxAI strict CUDA | FP32 | 36.261 | 4.593 | 4.436 | 17.952 | 54.213 | 4.021 |
| ONNX Runtime CUDA-first + CPU fallback | INT8 artifact | 7.425 | 2.482 | 2.801 | 10.894 | 18.252 | 1.000 |
| VolvoxAI strict CUDA | physical W8A8 | 31.952 | 5.318 | 5.219 | 20.993 | 52.953 | 2.901 |

Only VolvoxAI's rows are strict all-CUDA: every selected encoder/decoder node
reports `fallback=0;missing=0`. The ORT reference is CUDA-first with explicit
CPU fallback. Separate untimed profiling sessions with the same provider and
session configuration recorded exact executed-node placement: FP32 encoder
CUDA 473 / CPU 48 and decoder CUDA 202 / CPU 0; INT8 encoder CUDA 943 / CPU 56
and decoder CUDA 530 / CPU 0. Separate strict probes reject both encoders and
accept both decoders. The measured sessions themselves are deliberately not
profiled, so this evidence attests the invariant configuration and canonical
request, not the exact execution instance. ORT INT8 is therefore the producer
QDQ artifact under CUDA-first partitioning, not proof of an all-CUDA
tensor-core INT8 path.

Both processes selected visible CUDA ordinal 0, but this retained report does
not expose a comparable physical-device UUID across runtimes. The pairing is
therefore by ordinal and reported RTX 3090 identity, not a cryptographic
same-device attestation.

VolvoxAI CUDA INT8 improves encoder latency by 11.9% over its FP32 route, but
its decoder total is 16.9% slower; component totals improve by only 2.3%
(52.953 vs 54.213 ms). Route counters prove DP4A execution, including
139,889,120 DP4A dot4 groups and 2,889,600 scalar-tail operations for the
representative encoder. This does not claim IMMA or tensor-core use: those tiers
are not implemented in this result.

ORT is still faster because it brings mature CUDA graph optimization, fusion,
library kernels, packing, and tactic selection. VolvoxAI is a model-neutral
engine executing a much more granular per-node schedule with manual generic
kernels; its FP32 path does not use cuBLAS/cuDNN, TF32, or tensor cores. DP4A
helps the large encoder, while the one-row decoder remains dispatch/binding
limited and crosses Q/DQ/F32 boundaries often. Explicit KV is active on both
runtimes and does not explain the remaining gap.

## Dynamic-shape qualification

Dynamic shape support is a bounded backend/operator contract, not a claim that
every conceivable shape of every kernel is accepted. For this model's declared
domain, FP32 and INT8 currently pass the complete dynamic-shape v1
qualification on native CPU and WASM in the current CPU reports and on strict
WebGPU, Vulkan, and OpenGL in the current AMD GPU report. Each backend/precision pair
reuses one runtime plus one encoder and decoder context for this untimed
sequence:

```text
active Q=2/M=212
active Q=8/M=218
maximum-padded Q=192/M=402 (logical Q=8/M=218)
active Q=2/M=212
```

Every current sequence also records the exact question-token input, grows
decoder cache `P=1..4`, returns exact logical output shapes, preserves cache
prefixes, produces finite appended rows, selects the same family, and emits the
same tokens after shrinking. Unsupported bounded domains fail
compilation/execution as unsupported; execution never silently switches
backend. The retained CUDA run records strict shape/cache/output-token evidence,
but a current complete qualification needs a fresh RTX 3090 run.

## Reproduction

Build with at most `CPU count - 2` parallel jobs (10 on a 12-logical-CPU host):

```bash
cmake --build build/gpu-dynamic-release \
  --target tiny_receipt_split_w8a8 --parallel 10
make -j10 build_wasm
npm run build:all
```

CPU first-execution matrix:

```bash
: "${KV_MODEL_SOURCE:?set KV_MODEL_SOURCE to the cache-enabled ONNX directory}"

taskset -c 0 python3 -m examples.tiny_receipt_vqa.tools.benchmark_explicit_kv \
  --source "$KV_MODEL_SOURCE" \
  --fp32-package build/tiny-receipt-kv-f32 \
  --int8-package build/tiny-receipt-kv-int8 \
  --native-binary build/gpu-dynamic-release/native/tiny_receipt_split_w8a8 \
  --max-new 4 --warmup 1 --repeat 3 --threads 1 \
  --report examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix.json

taskset -c 0,2,4,6,8,10 \
  python3 -m examples.tiny_receipt_vqa.tools.benchmark_explicit_kv \
  --source "$KV_MODEL_SOURCE" \
  --fp32-package build/tiny-receipt-kv-f32 \
  --int8-package build/tiny-receipt-kv-int8 \
  --native-binary build/gpu-dynamic-release/native/tiny_receipt_split_w8a8 \
  --max-new 4 --warmup 1 --repeat 3 --threads 6 \
  --report examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix_6c.json
```

Vulkan/OpenGL/WebGPU same-context warmed matrix:

```bash
: "${KV_MODEL_SOURCE:?set KV_MODEL_SOURCE to the cache-enabled ONNX directory}"

ORT_WEB_TMP="$(mktemp -d)"
npm install --prefix "$ORT_WEB_TMP" \
  --ignore-scripts --no-save --package-lock=false \
  onnxruntime-web@1.27.0
ORT_WEB_ROOT="$ORT_WEB_TMP/node_modules/onnxruntime-web"

taskset -c 0-9 \
  python3 -m examples.tiny_receipt_vqa.tools.benchmark_explicit_kv_gpu \
  --source "$KV_MODEL_SOURCE" \
  --fp32-package build/tiny-receipt-kv-f32 \
  --int8-package build/tiny-receipt-kv-int8 \
  --ort-web-root "$ORT_WEB_ROOT" \
  --native-binary build/gpu-dynamic-release/native/tiny_receipt_split_w8a8 \
  --native-backend vulkan --native-backend opengl \
  --warmup 1 --repeat 5 \
  --report examples/tiny_receipt_vqa/reports/explicit_kv_v1_gpu_matrix.json
```

RTX 3090 CUDA remeasurement with the current GPU report harness:

```bash
taskset -c 0-1 \
  python3 -m examples.tiny_receipt_vqa.tools.benchmark_explicit_kv_gpu \
  --source "$KV_MODEL_SOURCE" \
  --fp32-package build/tiny-receipt-kv-f32 \
  --int8-package build/tiny-receipt-kv-int8 \
  --native-binary build/gpu-dynamic-release/native/tiny_receipt_split_w8a8 \
  --native-backend cuda --no-webgpu \
  --ort-provider CUDAExecutionProvider --allow-ort-cpu-fallback \
  --warmup 1 --repeat 5 \
  --report examples/tiny_receipt_vqa/reports/explicit_kv_v1_cuda_matrix.json
```

Content-addressed reports:

- [one-core CPU/native/WASM](../examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix.json), SHA-256 `06d09752777b421ca9383d2c73740bf4afbe274b741488a374ff861eedfb7d7b`
- [six-core CPU/native/WASM](../examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix_6c.json), SHA-256 `be3d2df5399abe965b3e7bf4d1acca6aa425e5ccf67aa4596a96bde95cc26588`
- [Vulkan/OpenGL/WebGPU](../examples/tiny_receipt_vqa/reports/explicit_kv_v1_gpu_matrix.json), SHA-256 `a6707b8f9556c15503fb59c54cb73f2c53e8563d7f0ce0bf8408efa9f575685f`
- [RTX 3090 CUDA retained measurement](../examples/tiny_receipt_vqa/reports/explicit_kv_v1_cuda_matrix.json), SHA-256 `23fb02af2f9543ccab192342b190f083e6d65da15d90fb6081ef1cd80d13133c`

All four report paths are tracked publication evidence. Staging and publication
remain explicit maintainer decisions.
