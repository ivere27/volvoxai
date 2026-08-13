# TinyReceiptVQA explicit-KV example

TinyReceiptVQA is application code built on VolvoxAI's model-neutral inference
lifecycle. Receipt preprocessing, tokenizer/question policy, router selection,
explicit-cache generation, and answer postprocessing stay in this directory.
Reusable shape binding, graph optimization, kernels, and strict backend
execution stay in VolvoxAI.

Only the cache-enabled split ABI is supported:

```text
source:  tiny_receipt_vqa_split_kv_onnx_v2
package: volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2
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
  --max-batch-size 2 \
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

`--max-batch-size` defaults to 1 and accepts 1..8. The importer applies the
same exactly qualified encoder-attention normalization for B=1 and B=1..N;
the default authors a fixed B=1 domain, while a larger value preserves the
producer's leading `B` symbol and authors the exact B=1..N domain in the
manifest plus both graphs. Dynamic authoring is transactional: it verifies
distinct-lane source-ONNX parity, then validates both emitted public ABIs and
domains before publishing the package. An unfamiliar consumer, reshape,
permutation, quantization axis, shared edge, or namespace collision fails the
import instead of guessing batch semantics.

## Imported graphs and hybrid W8A8

The B1 baseline artifacts revalidated on 2026-08-21 were produced before the
final normalization was made unconditional: they contain 357/170 FP32
encoder/decoder nodes and 480/271 INT8 nodes. Current imports apply the exact
encoder layout normalization at B=1 as well as B=1..N; the fresh B=1..2
packages used for physical batching contain 325/170 FP32 and 472/271 INT8
nodes. INT8 remains honestly marked hybrid with
`complete_w8a8_fusion=false`.
Its large compute is nevertheless W8A8:

The operator counts below describe the revalidated pre-normalization B1
baseline INT8 artifact; the current normalized B=1..2 package's node counts,
hashes, and typed-proof fingerprints are recorded in the benchmark note.

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
B = 1                                  # one high-level session call
package B domain = 1..N, N defaults 1  # opt-in max 8
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

The graph's explicit B domain and the application session are intentionally
different contracts. The component graph can execute a caller-authored bulk
tensor and, after typed independence proof, can receive coalesced Runtime B1
requests. `TinyReceiptSplitSession` itself still owns private encoder/decoder
contexts and serializes each autoregressive session through `_exclusive`; it
does not yet share compiled targets or coalesce different sessions. Production
multi-stream VQA needs a shared compiled-model service with per-sequence FIFO
KV state, not merely a B=2 package.

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

The current Android Vulkan FP32 and INT8 routes pass this same-context
qualification with strict fallback count zero:

```text
active Q=2/M=212 -> active Q=8/M=218
-> maximum-padded Q=192/M=402 -> active Q=2/M=212
```

That is qualification for this model's declared bounded domain, not a blanket
claim that every arbitrary shape is legal for every kernel.

The qualification document schema is
`volvoxai.tiny-receipt-dynamic-shape-qualification/v1`; the imported model and
package ABI is v2 only.

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
  backends: ['webgpu', 'wasm', 'cpu-js'],
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

There is no implicit reference. Pass an explicit package-bound v2 ORT reference
or `--no-reference`:

```bash
node examples/tiny_receipt_vqa/tools/run_split_e2e.mjs \
  --backend=cpu-js \
  --package=build/tiny-receipt-kv-int8 \
  --no-reference \
  --out=build/tiny-receipt-e2e/kv-int8-cpu-js-smoke.json

node --experimental-wasm-relaxed-simd --import tsx \
  examples/tiny_receipt_vqa/tools/verify_split_package.mjs \
  --package build/tiny-receipt-kv-int8 \
  --backend cpu-js
```

A no-reference run proves strict backend selection, forbidden operator
fallback, deterministic lifecycle, and runnability; it is not an accuracy
qualification. A reference must identify `tiny_receipt_vqa_split_kv_onnx_v2`
and exact graph, weight, and tokenizer hashes.

The portable initial-state fixture can be emitted without a backend:

```bash
node examples/tiny_receipt_vqa/tools/run_split_e2e.mjs \
  --package=build/tiny-receipt-kv-int8 \
  --fixtures-dir=build/tiny-receipt-e2e/fixtures \
  --fixtures-only
```

## Benchmarks

The [benchmark note](../../docs/tiny-receipt-vqa-bpe1536-benchmark.md)
records a retained pre-final host B1 observation, component-level WASM B2 proof,
the hash-bound
RTX 3090 Deno WebGPU B4/B8 proof, and the historical Android ARM64 comparison
between official ONNX Runtime CPU, VolvoxAI native CPU, and strict Vulkan. The
batch tool compares every encoder/decoder output and KV tensor lane-wise and
records execution diagnostics in
`volvoxai.tiny-receipt-vqa-runtime-batches/v2` for the MJS worker and
`volvoxai.tiny-receipt-vqa-runtime-batch-audit/v2` for the Python ORT wrapper,
including the selected `mode` and scheduler size/delay;
elapsed time or concurrent promise count alone is not accepted as batching
evidence.
Default reports redact machine paths and stable fixture/output digests. The
worker's raw tensor output directory remains private, and the explicit
`--include-private-artifacts` option is only for a non-public audit ledger.
The Python wrapper writes its report before enforcing the ORT comparison, but
returns nonzero with status `failed_ort_reference_tolerance` if either the
independent or scheduled route misses the configured tolerance gate.

### RTX 3090 WebGPU component batching

The retained sweep used Deno 2.9.3, Vulkan, NVIDIA driver
535.309.01, and a physical GeForce RTX 3090. The API SHA-256 was
`3337273dd7b9e87e5e865f457f6b602f4769e84225ee21f28480f852e33f49b1`.
Fresh FP32 and INT8 packages used `--max-batch-size 8`, the maximum supported
by this producer contract. B8 is therefore the largest legal model batch for
these artifacts; it is not the GPU's generic lane maximum.
The importer adds no separate B8 ceiling: a future source manifest with a
larger producer-proved domain may be imported up to that declared maximum.

One warmup and five measured groups gave these whole-group medians. DIRECT
makes N physical B1 invocations; SCHEDULED makes one physical B=N invocation
for the same N distinct inputs and includes all required output readbacks and
result close.

These immutable reports predate the execution-mode rename. Their table labels
are retained verbatim: `SIMPLE` maps to DIRECT, `ADAPTIVE` maps to SCHEDULED
with zero batch delay, and `SERVICE` maps to SCHEDULED with a positive bounded
delay. Active Runtime and benchmark APIs provide no aliases for those retired
plan names.
The active harness default remains a positive 10 ms delay; a new run intended
to match an ADAPTIVE zero-delay row must pass
`--max-batch-delay-ms 0` explicitly.

| Variant | Component | B | SIMPLE B1 group ms | ADAPTIVE B group ms | Speedup | Max abs / rel |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | Encoder | 4 | 715.239 | 226.844 | **3.153x** | `5.84126e-6` / `6.83582e-6` |
| FP32 | Encoder | 8 | 1282.119 | 316.801 | **4.047x** | `7.62939e-6` / `7.62361e-6` |
| FP32 | Decoder | 4 | 218.852 | 84.390 | **2.593x** | `8.58307e-6` / `2.68457e-3` |
| FP32 | Decoder | 8 | 311.744 | 147.503 | **2.113x** | `9.05991e-6` / `2.68457e-3` |
| INT8 | Encoder | 4 | 711.418 | 252.909 | **2.813x** | `0` / `0` |
| INT8 | Encoder | 8 | 1301.312 | 355.037 | **3.665x** | `0` / `0` |
| INT8 | Decoder | 4 | 234.617 | 110.830 | **2.117x** | `0` / `0` |
| INT8 | Decoder | 8 | 337.868 | 194.603 | **1.736x** | `0` / `0` |

Every B4 route records 20 logical requests as 20 DIRECT versus five scheduled
physical invocations across the five groups; every B8 route records 40 as 40
versus five. Scheduled evidence also reports only batch size 4 or 8, five
dispatches, and five true backend invocations. Thus each measured group is
physically N-to-1. INT8 outputs are bit-exact and the FP32 lane-wise maxima are
shown above.

The B8 `typed-independent-batch-proof/v1` identities are:

| Variant | Component | Covered nodes | Graph fingerprint SHA-256 |
| --- | --- | ---: | --- |
| FP32 | Encoder | 325 / 325 | `ab55247f05249dea6f546620cbc501a8149001340be28d8631580d320b602adf` |
| FP32 | Decoder | 170 / 170 | `f9958caba2abd31ff6577b4c18d91c52e920c0eec9e3ad52ca23d9dc5a546241` |
| INT8 | Encoder | 472 / 472 | `d199c9bc7f811a9179566e0c1ecafdc5bf40b08b006e4203c2c4ff6b74545900` |
| INT8 | Decoder | 271 / 271 | `ed4207bf41d4b3d9f9c212a4dca7ae7094cbea89f5a1ea5838cf92a3c12e7b57` |

The FP32/INT8 manifest SHA-256 values are
`fb20832602cc1757ff1410f59d7e3e7525a5ce55fdad24a3a0c4ca69566dd337`
and
`bb8c0838f5247abcab75377d4026265550962624c6b90bd04a0947cbbcd5cdac`.
In encoder B4/B8 then decoder B4/B8 order, the immutable runtime report
SHA-256 values are:

```text
FP32  796ffae6e935c529cbb1870106ae53a9c167609201f794eb7306774dc55ac18c
      2927242ccef00a8bd2ebd7225f033c6d66234b0bfe2ee8b54442a76014caefff
      dba8e09b6f35c412dc49dd8a753a9f86754e06ce0e36536d1a576c8026c57d0d
      d28fc3d23273018845940ae3e156b829f8587260468e45a9be64869b20158331
INT8  fb1151c56d2063ac429a5ce885e3dfd4ca9d10f94aa885b8c643089b35a1a31c
      f57f83ce3e58389503ceb9a69908569003062d4e0eb409e7fd8b5a45a997117f
      8ac026dbb09fbc52504b075b6a3c9d131863349f1082eae2ca0222000a706940
      65e32a90532f8ac518b9bad5ee357810a6decb79b021b1460a03599547b8ed3e
```

The report files, B8 packages, and distinct-lane fixtures named by these hashes
are external archive evidence and are not tracked in this repository. After
restoring those exact inputs, or generating and hashing new inputs, remeasure a
full fixture-authoring audit with:

```bash
DENO_WEBGPU_BACKEND=vulkan \
python3 -m examples.tiny_receipt_vqa.tools.benchmark_runtime_batches \
  --source "$KV_MODEL_SOURCE" --package build/tiny-receipt-kv-int8-b8 \
  --role encoder --backend webgpu --mode scheduled --concurrency 8 \
  --max-batch-delay-ms 0 \
  --warmup 1 --repeat 5 --api dist/0.4.0/volvoxai.js \
  --deno "$(command -v deno)" --adapter high-performance \
  --require-adapter 'RTX 3090' \
  --report build/tiny-receipt-kv-int8-encoder-webgpu-b8.json
```

The table is a same-WebGPU-backend batching-invariance measurement. It does
not replace the benchmark note's separate original-ONNX-Runtime fidelity gate.
It is also component proof only: `TinyReceiptSplitSession` continues to own
private contexts and serialize each autoregressive session at B1.

### WASM B2 audit

```bash
python3 -m examples.tiny_receipt_vqa.tools.benchmark_runtime_batches \
  --source "$KV_MODEL_SOURCE" --package build/tiny-receipt-kv-int8 \
  --role decoder --backend wasm --mode scheduled --concurrency 2 \
  --max-batch-delay-ms 0 \
  --wasm dist/0.4.0/volvoxai.wasm \
  --report build/tiny-receipt-kv-int8-decoder-b2.json
```

This is a component audit, not a `TinyReceiptSplitSession` throughput command.
The retained actual-model record also shows why legal batching cannot select B
by itself: only the INT8 one-token decoder improved in the single-sample WASM
B2 direction; both encoders and the FP32 decoder slowed down. Same-backend
WASM B1/B2 tensors were byte-exact, but the retained INT8 Runtime B1 full outputs
still differ materially from original ONNX Runtime. That separate numerical
fidelity gate remains open even though the small end-to-end token fixture and
the batching-invariance gate pass.

## Focused tests

```bash
node --test \
  examples/tiny_receipt_vqa/tests/tiny_receipt_split_session.test.mjs \
  examples/tiny_receipt_vqa/tests/tiny_receipt_split_e2e.test.mjs

node --import tsx --test \
  examples/tiny_receipt_vqa/tests/verify_split_package.test.mjs

python3 -m unittest \
  examples.tiny_receipt_vqa.tests.test_import_hf_split_onnx \
  examples.tiny_receipt_vqa.tests.test_import_hf_split_onnx_attention \
  examples.tiny_receipt_vqa.tests.test_benchmark_explicit_kv \
  examples.tiny_receipt_vqa.tests.test_benchmark_explicit_kv_gpu

node --test \
  examples/tiny_receipt_vqa/tests/cdp_reply_timeout.test.mjs \
  examples/tiny_receipt_vqa/tests/benchmark_explicit_kv_runtime.test.mjs
```

The tests cover sentinel and PAD-mask behavior, one-token positions, cross-cache
reuse, cache-prefix preservation, non-finite rejection, graph-relation
witnesses, strict manifest/backend rejection, warmup parity, dynamic
grow/shrink, and lifecycle cleanup.
