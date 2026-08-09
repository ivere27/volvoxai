# TinyReceiptVQA explicit-KV example

TinyReceiptVQA is application code built on VolvoxAI's model-neutral inference
lifecycle. Receipt preprocessing, tokenizer/question policy, router selection,
explicit-cache generation, and answer postprocessing stay in this directory.
Reusable shape binding, graph optimization, kernels, and strict backend
execution stay in VolvoxAI.

Only the cache-enabled split ABI is supported:

```text
source:  tiny_receipt_vqa_split_kv_onnx_v1
package: volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v1
```

## Import

The importer consumes a local, self-contained ONNX directory, not a PyTorch
checkpoint. The canonical source directory is named
`tiny_receipt_vqa_structured_qa_d320_e6_d4_bpe1536_lora_router_direct_novalue_e100_onnx`.
Keep it outside this repository and pass its path explicitly.

```bash
KV_MODEL_SOURCE=/path/to/cache-enabled-tiny-receipt-split-onnx

python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source "$KV_MODEL_SOURCE" \
  --out-dir build/tiny-receipt-kv-f32 \
  --target portable \
  --target backend:vulkan \
  --target backend:opengl \
  --target backend:cuda

python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source "$KV_MODEL_SOURCE" \
  --out-dir build/tiny-receipt-kv-int8 \
  --variant int8-w8a8 \
  --target portable \
  --target backend:vulkan \
  --target backend:opengl \
  --target backend:cuda
```

`--target` is repeatable and uses the generic exporter's intersection
semantics. These commands resolve to CPU JS, WASM, WebGPU, native CPU, Vulkan,
OpenGL, and CUDA qualification. The package records the requested/resolved
sets under `validation.offline_target_attestation`; runtime compilation still
proves the complete bounded domain on the physical device.

The importer validates hashes, opset 18, closed graph signatures, the
1,536-entry NFC `byte_fallback_bpe` tokenizer, eight-family router order, and
the producer's static U8S8 QDQ declaration. It delegates model-neutral ONNX
conversion to `tools/export_safetensors.py`.

## Imported graphs and hybrid W8A8

The canonical FP32 encoder/decoder contain 409/236 nodes. INT8 contains 480/271
nodes and remains honestly marked hybrid with `complete_w8a8_fusion=false`.
Its large compute is nevertheless physical W8A8:

| Operator | Encoder | Decoder |
| --- | ---: | ---: |
| `QConv2D` | 13 | 0 |
| `QLinear` | 26 | 33 |
| `QGemm` | 8 | 0 |
| `QBatchMatMul` | 14 | 18 |
| F32 Conv/Linear/Gemm/MatMul/BatchMatMul | 0 | 0 |
| Quantize / Dequantize | 62 / 61 | 53 / 48 |

All four formerly residual decoder F32 BatchMatMul regions are migrated, and
26 encoder plus 33 decoder biases are folded into quantized I32 accumulators.
The public memory/logits/KV ABI remains F32. LayerNorm, Softmax, GELU, and the
encoder's 13 GroupNorm/SiLU pairs also remain F32 because moving them to the
byte domain is a numerical migration, not an exact backend optimization.

TinyReceipt enables the generic exact common-subexpression pass, canonical
F32 SiLU recognition, quantized bias folding, and safe static-QDQ
`QBatchMatMul` migration. The generic GroupNorm/SiLU byte-island pass exists
but is intentionally not selected: a reproduced local 32-request check changed
three greedy outputs. TinyReceipt-specific qualification policy stays here;
the pass and kernels remain model-neutral.

## Closed graph ABI

| Graph | Inputs | Outputs |
| --- | --- | --- |
| Encoder | image, question IDs, question positions, family IDs | memory, memory mask, router logits, selected family, 4 layers x cross K/V |
| Decoder | one token, one position, family IDs, memory mask, past mask, 4 layers x cross K/V, 4 layers x past K/V | current-token logits, present mask, 4 layers x present K/V |

The public counts are encoder 4 inputs / 12 outputs and decoder 21 inputs / 10
outputs. Every name is mapped explicitly in `package_manifest.json`.

```text
B = 1
Q = 1..192
M = Q + 210 = 211..402
P = 1..191
R = P + 1 = 2..192
heads = 8
head width = 40
decoder layers = 4
```

Cross caches are F32 `[B,8,M,40]`; past/present self-attention caches are F32
`[B,8,P,40]` / `[B,8,R,40]`.

The producer begins decoding at empty `P=0`, while Volvox bounded dimensions
are positive. The package therefore starts with an all-zero `P=1` row and
`past_padding_mask=[1]`; nonzero means blocked. Import-time ORT validation proves
for all eight family IDs that this sentinel preserves logits, greedy token, and
the appended cache row. Every decoder call consumes one token, preserves the
cache prefix, and appends exactly one row.

## Dynamic shape and session use

The encoder runs once and the decoder reuses ordinary execution contexts.
`kvTransferMode: 'host-validated'` is the default: the host materializes the
encoder memory, eight cross caches, eight self-attention caches, and the past
mask so the session can validate exact shapes, binary masks, unchanged cache
prefixes, finite values, and `R=P+1` on every step.

WebGPU also supports `kvTransferMode: 'device-resident'` and
`'device-qualified'`. Both pass cross-cache and successor self-cache
`TensorResult` outputs directly into the next execution on the same physical
WebGPU device. The measured `device-resident` path performs zero KV readbacks;
it synchronizes only the small host outputs required for control flow (encoder
mask/router selection, then decoder logits/present mask). Its component timing
ends after those required reads and excludes application validation, argmax,
and result retirement. `device-qualified` uses the same device handoff and
additionally reads encoder memory/cross caches and each present-cache
prefix/appended row to validate correctness outside the measured interval.

`shapeMode: 'maximum-padded'` pads only encoder `Q/M` to `192/402`; decoder
`P/R` still grows one row per token. Backends compile against the declared
bounded domain and bind concrete positive shapes per request. Exact shape plans
and capacity are context-owned; a failed or unsupported bind never falls back
to another backend.

Native CPU and WASM passed the same-context grow/shrink sequence for FP32 and
INT8 in the current CPU reports. Physical WebGPU, Vulkan, and OpenGL passed the
current dynamic-shape v1 qualification in the current AMD GPU report. The
retained RTX 3090 CUDA measurement is not a fresh current qualification:

```text
active Q=2/M=212 -> active Q=8/M=218
-> maximum-padded Q=192/M=402 -> active Q=2/M=212
```

That is qualification for this model's declared bounded domain, not a blanket
claim that every arbitrary shape is legal for every kernel.

The dynamic qualification contract is
`volvoxai.tiny-receipt-dynamic-shape-qualification/v1`. The CPU and GPU reports
are separate `v1` document types. `explicit_kv_v1` in artifact paths names the
producer's explicit-KV model/package ABI.

## JavaScript session

Build the inference profile with no more than `CPU count - 2` jobs:

```bash
make -j10 build_wasm
npm run build:all
```

Then load the imported package:

```javascript
import { Model, VolvoxAI } from '../../ts/index.ts';
import { TinyReceiptSplitSession } from './TinyReceiptSplitSession.js';

const runtime = await VolvoxAI.createRuntime({
  backends: ['webgpu', 'wasm', 'cpu'],
});

const session = await TinyReceiptSplitSession.load({
  runtime,
  packageUrl: 'models/tiny-receipt-kv/package_manifest.json',
  snapshotLoader: async ({ graphUrl, weightsUrl, fetch, safetensorsCache }) =>
    await Model.load(weightsUrl, { graphUrl, fetch, safetensorsCache }),
});

const answer = await session.generate({
  image,
  prompt: 'What is the phone number?',
  family: 'auto',
  maxNewTokens: 96,
});

await session.close();
await runtime.close();
```

## Smoke and package verification

There is no implicit reference. Pass an explicit package-bound v1 ORT reference
or `--no-reference`:

```bash
node examples/tiny_receipt_vqa/tools/run_split_e2e.mjs \
  --backend=cpu \
  --package=build/tiny-receipt-kv-int8 \
  --no-reference \
  --out=build/tiny-receipt-e2e/kv-int8-cpu-smoke.json

node --experimental-wasm-relaxed-simd --import tsx \
  examples/tiny_receipt_vqa/tools/verify_split_package.mjs \
  --package build/tiny-receipt-kv-int8 \
  --backend cpu
```

A no-reference run proves strict backend selection, forbidden operator
fallback, deterministic lifecycle, and runnability; it is not an accuracy
qualification. A reference must identify `tiny_receipt_vqa_split_kv_onnx_v1`
and exact graph, weight, and tokenizer hashes.

The portable initial-state fixture can be emitted without a backend:

```bash
node examples/tiny_receipt_vqa/tools/run_split_e2e.mjs \
  --package=build/tiny-receipt-kv-int8 \
  --fixtures-dir=build/tiny-receipt-e2e/fixtures \
  --fixtures-only
```

## Benchmarks

The current CPU report harness compares strict ORT CPU, native C CPU, and
single-threaded VolvoxAI WASM. Native JavaScript CPU is excluded. Before warmup, it runs an
untimed canonical short/grow/maximum-padded/shrink qualification for native CPU
and WASM at both precisions, proving the same contexts, tokens, cache lifecycle,
input hash, and fallback zero. Its `--warmup` count means discarded fresh
matrices, so measured rows retain first-execution shape binding behavior.
Warmup and measured matrices use a recorded counterbalanced tier order.

```bash
: "${KV_MODEL_SOURCE:?set KV_MODEL_SOURCE to the cache-enabled ONNX directory}"

cmake --build build/gpu-dynamic-release \
  --target tiny_receipt_split_w8a8 --parallel 10
make -j10 build_wasm
npm run build:all

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

The current GPU report harness instead runs each warmup on the same runtime/session/
contexts as the immediately following measured request, resets KV to `P=1`, and rejects
token/cache drift. It counterbalances and records per-repeat tier order, and
runs an untimed same-context dynamic grow/shrink qualification for every
selected VolvoxAI backend/precision pair.

```bash
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

The harness verifies the exact `onnxruntime-web@1.27.0` package and asset
hashes under `ORT_WEB_ROOT`. Fresh browser origins/profiles serve artifacts
with `Cache-Control: no-store`; VolvoxAI preloads both contexts, while ORT
fetches each selected model once before creating both sessions. The report
compares VolvoxAI WebGPU directly with ORT WebGPU on the same browser GPU API.
ORT's optimized sessions use WebGPU plus attested CPU partitions, and strict
no-CPU-fallback probes reject encoder and decoder at both precisions. With the
pinned ONNX Runtime Web 1.27.0 build, FP32 Conv and MatMul are not the missing
kernels. Its shipped operator table marks `Reshape` and `Shape` as having no
GPU kernel; captured missing-kernel events include both, and model inspection
also finds INT64/BOOL shape/control tensors that constrain other routes. INT8
additionally lacks a registered WebGPU `QuantizeLinear` kernel. This is an
operator-table/model cross-check rather than exact per-node attribution.
Aggregate provider-assignment counts are exact, but
the repeated unsupported-kernel diagnostics are capability-probe events, not
node, execution, transfer, or partition-boundary counts. Native Vulkan/OpenGL
are reported separately because official ONNX Runtime has no direct native
Vulkan or OpenGL execution provider. Its native WebGPU plugin may use Dawn over
Vulkan on Linux, but that is a different abstraction rather than a same-backend
Vulkan peer; ORT Web's WebGL route is likewise not native OpenGL.

The current CPU and AMD GPU harnesses remove inherited uppercase `VOLVOX*`
runtime overrides from every benchmark child and force common nested-library
thread limits to the requested harness count (one for the one-core and GPU
matrices, six for the six-core matrix); only override names, never values, are
recorded. Their exact sanitized environment and execution order are part of
their reports.

For a future RTX 3090 CUDA remeasurement with the current GPU report harness,
disable WebGPU and label ORT
honestly as CUDA-first with CPU fallback. Strict ORT CUDA rejects the encoder's
CPU-assigned shape/control partition, whereas VolvoxAI CUDA stays strict with
fallback zero.

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

The [benchmark note](../../docs/tiny-receipt-vqa-bpe1536-benchmark.md) records
actual CPU, WASM, WebGPU, Vulkan, OpenGL, and RTX 3090 CUDA results, along with
the strictness and lifecycle caveats. The RTX 3090 result is a retained
measurement, not a fresh current qualification.
All four report paths are tracked publication evidence; staging remains an
explicit maintainer decision.

## Focused tests

```bash
node --test \
  examples/tiny_receipt_vqa/tests/js_tiny_receipt_split_session.test.mjs \
  examples/tiny_receipt_vqa/tests/js_tiny_receipt_split_e2e.test.mjs

node --import tsx --test \
  examples/tiny_receipt_vqa/tests/js_verify_split_package.test.mjs

python3 -m unittest \
  examples.tiny_receipt_vqa.tests.test_import_hf_split_onnx \
  examples.tiny_receipt_vqa.tests.test_benchmark_explicit_kv \
  examples.tiny_receipt_vqa.tests.test_benchmark_explicit_kv_gpu

node --test \
  examples/tiny_receipt_vqa/tests/js_cdp_reply_timeout.test.mjs \
  examples/tiny_receipt_vqa/tests/js_benchmark_explicit_kv_runtime.test.mjs
```

The tests cover sentinel and PAD-mask behavior, one-token positions, cross-cache
reuse, cache-prefix preservation, non-finite rejection, graph-relation
witnesses, strict manifest/backend rejection, warmup parity, dynamic
grow/shrink, and lifecycle cleanup.
