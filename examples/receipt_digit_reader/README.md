# Receipt digit reader example

[Hugging Face model](https://huggingface.co/ivere27/tiny-receipt-reader-digit-slots-2m) · [Accuracy and benchmark results](BENCHMARK.md) · [Run benchmarks](benchmarks/README.md)

The tiny receipt digit reader is a 2.46M-parameter convolutional stem with an
iterative slot readout. It reads one receipt image and emits one record: a
phone number and a street number, each as left-aligned digit slots terminated
by a blank class. The graph never sees a question — question handling is
regex-only and lives in `questionRouter.js` — so several questions about one
receipt cost exactly one forward pass.

This directory is application code built on VolvoxAI's model-neutral inference
lifecycle. Image preprocessing, the producer contract, slot decoding, and
question routing stay here. Shape binding, graph optimization, kernels, PTQ
authoring, and strict backend execution stay in VolvoxAI.

```text
source:  receipt_digit_reader_onnx_v1
package: volvoxai-receipt-digit-reader-onnx-package-v1
```

## Download the model

Run these commands from the VolvoxAI repository root. Download the published
[tiny-receipt-reader-digit-slots-2m](https://huggingface.co/ivere27/tiny-receipt-reader-digit-slots-2m)
release before importing or running PTQ:

```bash
python3 -m pip install huggingface_hub
SOURCE=examples/receipt_digit_reader/benchmarks/work/source
READER_REVISION=c9fa07886f99334ea820bfe312d6b4771d9460b0

hf download ivere27/tiny-receipt-reader-digit-slots-2m \
  --revision "$READER_REVISION" --local-dir "$SOURCE"
```

The download includes `model.onnx`, `model_int8.onnx`, `config.json`,
`manifest.json`, `SHA256SUMS`, and the producer's question router and examples.
The pinned revision keeps these files from different exports from being mixed.
The download directory is ignored by Git. Keep `SOURCE` set for the commands
below; no private release directory is needed.

## Import

The importer consumes a local, self-contained release directory, not a PyTorch
checkpoint. It verifies the release `SHA256SUMS` for every artifact it reads,
requires opset 18, derives the per-request ABI from the producer manifest, and
delegates model-neutral ONNX conversion to `tools/export_safetensors.py`.

```bash
python3 -m examples.receipt_digit_reader.tools.import_hf_onnx \
  --source "$SOURCE" --out-dir build/receipt-digit-reader-fp32 --variant fp32

python3 -m examples.receipt_digit_reader.tools.import_hf_onnx \
  --source "$SOURCE" --out-dir build/receipt-digit-reader-int8 --variant int8 \
  --max-batch-size 4
```

`--target` is repeatable and uses the generic exporter's intersection
semantics; the default is `portable`, which resolves to WASM, WebGPU,
and native CPU.

### Per-request B1 and optional physical batching

`--max-batch-size` defaults to 1. At that default the graph is static B1. A
value greater than one preserves the producer's named leading batch symbol
and authors its exact graph domain, for example `batch=1:256:1`. The importer
does not impose an application-sized ceiling: backend compilation proves
element, launch, binding, and resident-memory safety for the requested domain.
The browser-facing `ReceiptDigitSession` keeps its separate application policy
limit on top of the generated `Run` contract; native Runtime packages are not
constrained by that helper. The
manifest still declares `abi.input.shape: [1, 1, 320, 672]` and
`abi.batch.per_request: 1`: one `ReceiptDigitSession.read()` is always one
receipt. A shared Runtime may coalesce several such B1 calls into one physical
B=N invocation in SCHEDULED mode.

This is distinct from a caller supplying a bulk B=N tensor through generated
`Run` or `Execute`. The example's application session exposes only a B1
per-request contract.

Dynamic import is allowed only when the producer manifest and ONNX public
input and outputs all carry the same leading symbol. The producer also leaves
a stale `dim_param` on the slot axis even though its manifest proves 16 slots;
the importer binds that symbol to the singleton domain `16:16`. Generic ONNX
shape programs are folded only from immutable integer shape values. A
data-dependent or noncanonical `Shape` consumer fails closed rather than
assuming batch 1.

## PTQ variant

Complete [Download the model](#download-the-model) first. VolvoxAI PTQ starts
from the downloaded **`model.onnx` FP32 graph**. The downloaded
`model_int8.onnx` is the producer's separately quantized comparison model.

The `ptq` variant is **not** a re-encoding of `int8`. The producer quantizes
convolution only (`quantized_op_types: ["Conv"]`), so its graph keeps float
GroupNorm and float SiLU that VolvoxAI would have quantized natively. The two
paths cannot converge numerically; see [typed PTQ](../../docs/typed-ptq.md).

The generated-API workflow below imports the downloaded files, prepares the
FP32 graph, calibrates C PTQ, writes the INT8 package, and evaluates it.
Provide a calibration image directory and a separate held-out image/annotation
pair. These corpora are not included in the model repository; its two demo
receipts are not a 256-image calibration set or a 2,000-case evaluation set.
Calibration images must be disjoint from the entire evaluation split.

### Verify the generated PTQ API

The following check imports both producer ONNX models, prepares the float graph,
and calls `AuthorPtqTemplate`, `CreatePtqPlan`, `CalibratePtqPlan`,
`InspectPtqPlan`, and `WritePtqPackage` through the generated Python clients.
It reloads the result in both native profiles and compares predictions with
ONNX Runtime and the evaluation annotations. Build the native libraries first
with `make build_native_libraries`.

```bash
python3 -m examples.receipt_digit_reader.tools.verify_proto_ptq \
  --source "$SOURCE" \
  --calibration-images /path/to/calibration/images \
  --eval-images /path/to/heldout/images \
  --eval-annotations /path/to/heldout/annotations \
  --limit 2000 \
  --out-dir build/receipt-proto-ptq

# After building both web profiles with make build_web:
node examples/receipt_digit_reader/tools/verify_proto_ptq_wasm.mjs \
  build/receipt-proto-ptq
```

Use a new output directory for fresh imports. `--reuse-imports` repeats native
calibration from the prepared package; `--reuse-packages` repeats evaluation.
Calibration uses 256 distinct images by default and excludes normalized inputs
present anywhere in the evaluation directory. `--limit` restricts scoring,
while preserving that separation. Annotations use the producer's
`receipt.store`, `question`, and `answer` fields.

The output contains `ptq/graph.json`, `ptq/model.safetensors`, `report.json`,
input hashes and raw calibration inputs. The WASM check consumes those same
inputs, writes `ptq-wasm/`, and compares the first eight evaluation images in
both WASM profiles. Reports distinguish logit differences, slot decisions,
whole records and task accuracy; successful export alone proves none of those
accuracy measures. The native acceptance checks allow at most one percentage
point of PTQ record-accuracy loss, and report imported INT8 numerical differences
separately. These scripts do not qualify GPU backends.

### Verify the same packages on CPU, WASM and GPU

The ONNX importer above is Python tooling. The runtime accepts VolvoxAI graph
and safetensors packages; it does not expose a C ONNX importer. PTQ authoring,
calibration and packing run in C through the generated
[quantization service](../../proto/volvoxai.proto). C, Python and JavaScript
callers use that same contract. PTQ requires the full profile, and calibration
currently uses the portable CPU engine without a GPU backend selector. Its
exported INT8 package can then run in either inference profile.

Prepare one corpus after the native and WASM PTQ checks above. This verifies
that all normalized evaluation inputs match the original evaluation and that
none occurs in the calibration set. It retains the four packages and their
native CPU reference logits alongside compact grayscale image bytes.

```bash
python3 examples/receipt_digit_reader/tools/verify_proto_backends.py prepare \
  --verification build/receipt-proto-ptq \
  --images /path/to/heldout/images \
  --annotations /path/to/heldout/annotations \
  --out build/receipt-backend-corpus

python3 examples/receipt_digit_reader/tools/verify_proto_backends.py run \
  --corpus build/receipt-backend-corpus --backend cpu --profile inference \
  --limit 2000 --out build/receipt-cpu.json

node examples/receipt_digit_reader/tools/verify_proto_backends.mjs \
  --corpus build/receipt-backend-corpus --backend wasm --profile inference \
  --limit 2000 --out build/receipt-wasm.json
```

Repeat with `--profile full`. Both native profiles admit compiled CUDA,
Vulkan and OpenGL backends; select one with `--backend cuda`, `vulkan` or
`opengl`. CUDA is optional at build time. For an RTX 3090 build:

```bash
make build_native_profiles build_native_libraries \
  CMAKE_BUILD_DIR=build/cmake-receipt-gpu \
  CMAKE_CONFIG='-DCMAKE_C_COMPILER=clang -DVOLVOXAI_ENABLE_CUDA=ON -DVOLVOXAI_CUDA_ARCH=86'

VOLVOXAI_TEST_NATIVE_GPU_BACKEND=vulkan \
  python3 python/tests/test_native_gpu_lifecycle.py
```

The lifecycle check uses independent Linear outputs and repeatedly closes
and recreates native owners in both profiles, including worker thread exit.
Run it for each selected native GPU backend. Device checks must run
sequentially on an idle GPU; follow the [hardware test policy](../../docs/testing.md).

WebGPU belongs to the full browser profile. On the RTX 3090 test device, use
the repository's [pinned Deno runner](../../tools/deno/README.md):

```bash
DENO_WEBGPU_BACKEND=vulkan build/deno/target/webgpu-fix/deno run \
  --no-config --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi \
  examples/receipt_digit_reader/tools/verify_proto_backends.mjs \
  --corpus build/receipt-backend-corpus --backend webgpu --profile full \
  --limit 2000 --out build/receipt-webgpu.json
```

The GPU checks forbid backend and operator fallback, require route attestation
for every execution, and verify that every selected node is active. The
WebGPU harness also inspects the actual adapter acquired by the product
bridge and requires physical NVIDIA hardware. Linux results do not qualify
Metal or a browser deployment. Each report records artifact hashes, execution
status, logit differences and decoded record agreement against the identical
CPU package; raw output logits are retained for held-out accuracy scoring.
Successful execution is separate from numerical parity. `--limit 0` runs the
entire corpus; always report the actual sample count.

### Coverage is convolution-only by default

The default PTQ profile quantizes the convolutional stem and keeps the slot
readout in float, matching the producer's INT8 coverage. The latest full
2,000-case accuracy results for FP32, imported INT8, native C PTQ and WASM C
PTQ are in the [benchmark record](BENCHMARK.md).

`--ptq-float-op` replaces the list of operators retained in float. Changing
that list to quantize the readout or normalization needs a separate full
accuracy evaluation because those operators change numerical behavior.

### Attention scores stay in float

The readout contracts every grid cell against a fixed slot query, so its score
`BatchMatMul` outputs reach four digits of dynamic range:

| tensor | role | observed range | int8 step |
| --- | --- | ---: | ---: |
| round 2 | Q·Kᵀ | ±1809 | 14.6 |
| round 3 | Q·Kᵀ | ±3757 | 29.1 |
| all rounds | attn·V | [−0.4, 23] | 0.05–0.09 |

Those scores feed `Div → Softmax`. An 8-bit step of 29 on a pre-softmax logit
erases the attention pattern the phone head depends on, so `BatchMatMul` is
retained in F32 in every coverage above.
`--ptq-quantize-attention-scores` removes that guard; it collapses
`phone_digit_exact` and exists so the cost is on the record, not folklore.

The default activation scheme is **asymmetric**. Symmetric int8 collapses this
model outright: SiLU outputs are bounded below at −0.2785 and softmax outputs
live in `[0, 1]`, so a symmetric range spends half its levels on values those
tensors never take.

### Why the FP32 source must keep its bias ports

`Linear` and `Conv2D` fold an immutable `[d_out]` bias into one I32
accumulator. If the importer had lowered the producer's `MatMul + bias[256]`
as two nodes, the bias would be broadcast to the full activation shape, its
per-output-channel structure would be invisible to bias folding, and PTQ would
quantize the pre-bias accumulator and the bias separately. That is exactly what
happened before `_recognize_linear_bias` was added: FP32 stayed bit-comparable
with ONNX Runtime while `phone_digit_exact` collapsed to roughly one percent.
FP32 parity does not imply a quantizable graph.

## Backends

One package runs on every portable backend. The session requires its backend
strictly and forbids operator fallback, so an unavailable provider fails loudly
instead of silently landing somewhere slower.

| Backend | Entry point |
| --- | --- |
| WASM | `tools/run_backends.mjs` |
| WebGPU | `../receipt_digit_reader.html` |
| Native CPU, Vulkan, OpenGL, CUDA | `native/main_receipt_digit_reader.c` |

```bash
# Strict WASM execution
node examples/receipt_digit_reader/tools/run_backends.mjs \
  --package build/receipt-digit-reader-fp32 \
  --raw receipt.f32 --backend wasm \
  --wasm-url dist/0.5.0/volvoxai.wasm
```

The default terminal and JSON reports omit receipt paths and decoded values.
`--include-private-records` is available only for local diagnosis; output
produced with it can contain personal data and must not be published.

For WebGPU, serve the repository root and open
`examples/receipt_digit_reader.html`. The page reads a receipt, shows the
record, and answers a typed question through the same regex router. Selecting
`webgpu` where no adapter exists reports that plainly rather than falling back.

### Native

`make build_native` builds `receipt_digit_reader` alongside the runtime:

```bash
./build/cmake/native/receipt_digit_reader \
  --package build/receipt-digit-reader-int8 \
  --image receipt.jpg --backend cpu
# The JSON response contains decoded digit strings and backend "cpu".
```

`--backend` takes a **runtime** identity — `cpu`, `vulkan`, `opengl`, `cuda` —
not the exporter target name (`native-cpu`). The C sources own image decoding
(stb_image), preprocessing, manifest policy, and slot decoding. Public C
applications should talk to the generated service contract; this example keeps
its application logic in C for benchmarking and packaging parity.

## Inference

`ReceiptDigitSession.js` is a thin application helper over `EngineHost` and
the generated inference client. Public applications can either use that helper
or issue the same `CreateRuntime -> LoadModel -> CompileModel -> Run ->
ReadOutput -> Release*` RPC sequence directly, then keep
`prepareReceiptImage`, slot decoding, and `answerFromRecord` downstream. See
the repository
[README](../../README.md#web-inference) for the generated handle lifecycle.

Create the Runtime without `executionMode` for a DIRECT one-shot `Run`. For
concurrent reads, explicitly create it as SCHEDULED and call `Submit` with the
compiled model ID. The task adapter requires its selected backend and forbids
operator fallback; it never changes backend after compilation.

The native CLI runs the same packages:

```bash
./native/volvoxai run build/receipt-digit-reader-fp32 \
  --input input0=receipt.f32 --output slot_logits=logits.f32
```

## Fine-tuning

Training uses `VxTrainingService` from the full profile. The selected backend
preflights the complete graph and fails before optimizer mutation when an
operator or layout is unsupported.

```javascript
const trainer = await training.createTrainer(new pb.CreateTrainerRequest({
  modelId: model.modelId,
  backend: 'wasm',
}));
const step = await training.trainStep(new pb.TrainStepRequest({
  trainerId: trainer.trainerId,
  inputs: [imageTensor],
  losses: [new pb.CrossEntropyLoss({
    name: 'slots', logitsName: 'slot_logits',
    targets: new pb.Tensor({ shape: [BigInt(targets.length)],
      dtype: pb.DataType.DATA_TYPE_I32,
      inline: new Uint8Array(Int32Array.from(targets).buffer) }),
  })],
  trainableNames: ['w36', 'w38', 'w41'],
  optimizer: new pb.TrainerOptimizerOptions({
    kind: pb.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW,
    learningRate: 5e-4,
  }),
}));
const revision = await training.commitTrainer(
  new pb.TrainerRef({ trainerId: trainer.trainerId }),
);
```

`TrainStep` mutates only the Trainer's private revision. `CommitTrainer`
publishes the successor revision on the retained Model handle;
`RollbackTrainer` restores the last baseline.

## Evaluation

```bash
python3 -m examples.receipt_digit_reader.tools.eval_heldout \
  --package build/receipt-digit-reader-fp32 \
  --images heldout.f32 --labels heldout.json \
  --onnx "$SOURCE/model.onnx"
```

`--onnx` adds an ONNX Runtime column computed on the identical preprocessed
batch, so the comparison is against the producer on the same inputs rather than
against a published number from a different pipeline.

## Benchmark

The [latest benchmark](BENCHMARK.md) reports per-receipt latency and all
2,000-case accuracy results for native CPU, WASM, CUDA, Vulkan, OpenGL and
WebGPU. It covers FP32, producer INT8, native C PTQ and WASM C PTQ.

Use the [benchmark workflow](benchmarks/README.md) to measure the selected
backend on the appropriate host. Native CPU/WASM measurements use the local
CPU host; GPU measurements require an idle GPU host.

The [latency record](reports/benchmark.json) contains one latest measurement
per backend/package, with every sample and runtime/model/caller hash. The
[validation record](reports/validation.json) retains all 44 cells of the
2,000-case audit. These files are replaced when new qualified results are
published; per-case distributions and timing boundaries remain explicit.
