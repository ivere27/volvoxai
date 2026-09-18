# TinyReceiptVQA explicit-KV example

[Hugging Face model](https://huggingface.co/ivere27/tiny-receipt-vqa-structured-qa-21m) · [Accuracy and benchmark results](BENCHMARK.md) · [Run benchmarks](benchmarks/README.md)

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

## Download the model

Run these commands from the VolvoxAI repository root. Download the published
[tiny-receipt-vqa-structured-qa-21m](https://huggingface.co/ivere27/tiny-receipt-vqa-structured-qa-21m)
release before importing or running PTQ:

```bash
python3 -m pip install huggingface_hub
KV_MODEL_SOURCE=examples/tiny_receipt_vqa/benchmarks/work/source
VQA_REVISION=f536c6ed181ca88929214f0bbc08f126f148c7ae

hf download ivere27/tiny-receipt-vqa-structured-qa-21m \
  --revision "$VQA_REVISION" --local-dir "$KV_MODEL_SOURCE"
```

The download includes the FP32 encoder/decoder, both `_int8.onnx` graphs,
`config.json`, `manifest.json`, `vocab.json`, `SHA256SUMS`, and the producer's
inference code and examples. The pinned revision keeps models, tokenizer and
cache contracts from the same export. The download directory is ignored by
Git. Keep `KV_MODEL_SOURCE` set for the import and PTQ commands below.

## Import

The importer consumes the downloaded self-contained ONNX directory. It checks
the producer contract instead of requiring a particular directory name.

```bash
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
semantics. These commands resolve to WASM, WebGPU, native CPU, Vulkan,
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

Current imports apply encoder layout normalization at B=1 and B=1..N. The
producer INT8 package is hybrid (`complete_w8a8_fusion=false`): its convolution
and dense compute use quantized operators while public memory/logits/KV ports
remain F32. LayerNorm, Softmax, GELU and encoder GroupNorm/SiLU also remain F32.
Exact graph and weight hashes are recorded with the
[latest validation](reports/validation.json).

TinyReceipt enables the generic exact common-subexpression pass, canonical
F32 SiLU recognition, quantized bias folding and safe static-QDQ
`QBatchMatMul` migration. Moving normalization and activation operators into a
byte domain requires separate numerical qualification. The generic
GroupNorm/SiLU byte-island pass is therefore not selected by this importer.
TinyReceipt-specific qualification policy stays here; the passes and kernels
remain model-neutral.

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
requests. Production multi-stream VQA needs shared compiled models with
per-sequence FIFO KV state; declaring a B=2 package alone does not provide
application session coordination.

The producer begins decoding at empty `P=0`, while Volvox bounded dimensions
are positive. The package therefore starts with an all-zero `P=1` row and
`past_padding_mask=[1]`; nonzero means blocked. Import-time ORT validation proves
for all eight family IDs that this sentinel preserves logits, greedy token, and
the appended cache row. Every decoder call consumes one token and appends one
row. FP32 preserves the cache prefix exactly. The producer's INT8 graph applies
Q/DQ to the public value cache; ORT's optional QDQ optimizations can also move
quantization across the key-cache concatenation. An arbitrary incoming F32
prefix can therefore change on the INT8 path. Compare those outputs against
the source ONNX, rather than treating all F32 cache ports as lossless copies.

## Dynamic shape contract

The encoder runs once and the decoder reuses an execution-context ID. Each
generated `Execute` request supplies the complete named tensor set with exact
dtypes, shapes, and bytes. The caller reads encoder memory, cross caches,
self-attention caches, masks, and logits from retained result IDs, then submits
the required values to the next decoder call. In-process transports may use a
`BufferView` for caller-owned memory, but the public schema does not yet expose
a provider-issued result as a later device input.

Current providers compile the declared bounded domain and bind
concrete positive shapes per request; `Q/M` and `P/R` may vary only within that
domain. Exact shape plans and capacity are context-owned, and a failed or
unsupported bind never switches to another provider.

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

## JavaScript integration

Build the web inference artifacts:

```bash
npm run build:all
make build_wasm
```

The current [JavaScript verification runner](tools/verify_proto_backends.mjs)
loads both graphs through the generated proto clients and performs complete
greedy decoding with explicit cache inputs. Public applications compose
`EngineHost` with the generated inference/scheduler clients and keep image
preprocessing, tokenizer policy, autoregressive
generation, and answer parsing downstream. See the repository
[README](../../README.md#web-inference).

The old browser/Node VolvoxAI harnesses in this directory were removed after
the public bundle dropped the handwritten `VolvoxAI`/`Model`/`ModelLoader`
surface in favor of generated proto clients over `EngineHost`. Those harnesses
also depended on private object-API diagnostics and device-resident transfer
paths that the public proto contract does not expose. The retained JS utilities
here are the image preprocessor, the standalone BPE tokenizer, the ORT WebGPU
comparison harness, and the wasm binary microbenchmark.

## PTQ and package verification

Complete [Download the model](#download-the-model) first. VolvoxAI PTQ starts
from the downloaded **`encoder_model.onnx` and `decoder_model.onnx` FP32
graphs**. The `_int8.onnx` files are the producer's separately quantized
comparison models.

The [Python runner](tools/verify_proto_backends.py) prepares a shared B1 corpus,
compares original ONNX execution, verifies native backends, and calibrates C
PTQ. The JavaScript runner uses the actual release bundles for WASM and physical
WebGPU. Both call the generated proto API, require the requested backend, forbid
operator fallback, inspect every execution's route evidence, and release each
result after copying the outputs needed by the next step.

Import both downloaded variants as B1 packages into the work directory used
by PTQ. Calibration requires separate training images and question annotations;
evaluation requires held-out images and annotations. These corpora are not
included in the model download. The bundled demonstration receipts do not
replace the 256-record calibration or the 2,000-case accuracy evaluation.

```bash
VQA_WORK=build/tiny-receipt-vqa-verification
VQA_EVAL=/path/to/heldout
VQA_CALIBRATION=/path/to/training-data

python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source "$KV_MODEL_SOURCE" --out-dir "$VQA_WORK/fp32" \
  --variant fp32 --max-batch-size 1 \
  --target portable --target backend:vulkan --target backend:opengl --target backend:cuda

python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source "$KV_MODEL_SOURCE" --out-dir "$VQA_WORK/int8" \
  --variant int8-w8a8 --max-batch-size 1 \
  --target portable --target backend:vulkan --target backend:opengl --target backend:cuda

python3 -m examples.tiny_receipt_vqa.tools.verify_proto_backends prepare \
  --work "$VQA_WORK" \
  --eval-images "$VQA_EVAL/images" --eval-annotations "$VQA_EVAL/annotations" \
  --calibration-images "$VQA_CALIBRATION/images" \
  --calibration-annotations "$VQA_CALIBRATION/annotations"

python3 -m examples.tiny_receipt_vqa.tools.verify_proto_backends fixtures \
  --work "$VQA_WORK" --source "$KV_MODEL_SOURCE"

python3 -m examples.tiny_receipt_vqa.tools.verify_proto_backends run \
  --work "$VQA_WORK" --source "$KV_MODEL_SOURCE" --backend ort \
  --variants fp32 int8 --limit 0 --out "$VQA_WORK/ort.json"

python3 -m examples.tiny_receipt_vqa.tools.verify_proto_backends calibrate \
  --work "$VQA_WORK"
node examples/tiny_receipt_vqa/tools/verify_proto_ptq_wasm.mjs "$VQA_WORK"

python3 -m examples.tiny_receipt_vqa.tools.verify_proto_backends run \
  --work "$VQA_WORK" --backend cpu --profile inference \
  --limit 2000 --out "$VQA_WORK/native-cpu.json"
node examples/tiny_receipt_vqa/tools/verify_proto_backends.mjs \
  --work "$VQA_WORK" --backend wasm --profile inference \
  --limit 2000 --out "$VQA_WORK/wasm.json"
```

`--limit 0` evaluates every annotation. Native `--backend` also accepts `cuda`,
`vulkan`, and `opengl`; repeat with `--profile full` to verify both native
profiles. The web inference profile supports WASM CPU; WebGPU requires
`--backend webgpu --profile full` and a WebGPU-capable host. The physical GPU
verification runner currently requires an NVIDIA adapter and rejects software
renderers. Run GPU routes sequentially per physical device and record the
device/driver and artifact hashes. These are correctness runs with output
readback on every token, not throughput measurements.

For the RTX 3090 WebGPU check, use the repository's
[pinned Deno test runner](../../tools/deno/README.md). It contains wgpu lifecycle
and memory-budget fixes and is separate from the eight VolvoxAI release files.
Its results qualify that runtime; individual browser products need their own
runs.

```bash
DENO_WEBGPU_BACKEND=vulkan build/deno/target/webgpu-fix/deno run \
  --no-config --allow-all --unstable-webgpu \
  examples/tiny_receipt_vqa/tools/verify_proto_backends.mjs \
  --work "$VQA_WORK" --backend webgpu --profile full \
  --limit 2000 --out "$VQA_WORK/webgpu.json"
```

The nine component fixtures cover automatic routing and all eight explicit
families, question lengths 1..192, and past lengths 1..191 with growth and
shrink in reused contexts. They use ORT with graph optimizations disabled to
compare the ONNX graph as authored. End-to-end ORT defaults to its normal
optimizations; `--ort-disable-optimizations` provides a separate reference.
Component reports retain per-output error and argmax agreement. Successful
execution does not imply numerical agreement, especially for an imported INT8
graph whose bias folding and integer compute introduce additional rounding.

Calibration selects 256 distinct normalized images, excluding the entire
evaluation corpus by normalized content. It uses the corresponding questions
and FP32-generated decoder prefixes at positions 0, 4, and 12 when reached;
heldout answers are never calibration inputs. C PTQ quantizes Conv2D and Linear
weights/activations with per-channel symmetric I8 weights and asymmetric I8
activations. Attention, embeddings, normalization, and activation functions
remain float. The output keeps the original named dynamic shape contract and
public F32/I32 input/output types. Serialized calibration requests let the
WASM C service repeat the same experiment. Native and WASM packages are saved
separately as `ptq` and `ptq-wasm`; their observed ranges and exported bytes can
differ slightly because the CPU implementations accumulate floats differently.

ONNX import is a Python development tool. C implements PTQ and execution,
accessible from C dispatch, generated Python clients, and generated JavaScript
clients. There is no C ONNX parser. PTQ author/calibrate/write belongs to the
full profile; either profile can load the resulting quantized package. See
[`proto/volvoxai.proto`](../../proto/volvoxai.proto) for the complete contract.

## Benchmarks

The [latest benchmark](BENCHMARK.md) reports complete-answer latency through
EOS and accuracy for all 2,000 held-out image/question pairs. It covers FP32,
producer INT8, native C PTQ and WASM C PTQ on native CPU, WASM, CUDA, Vulkan,
OpenGL and WebGPU.

Follow the [measurement workflow](benchmarks/README.md) for retained-session
measurements and host contention checks. The [latency record](reports/benchmark.json)
keeps one latest qualified row per backend/package, including per-input timing,
TTFT, decoder steps and generated tokens/s. The
[validation record](reports/validation.json) contains the full 44-cell accuracy
audit. Publication replaces these two files instead of appending history.
