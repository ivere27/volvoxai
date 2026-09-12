# Receipt digit reader example

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

## Import

The importer consumes a local, self-contained release directory, not a PyTorch
checkpoint. It verifies the release `SHA256SUMS` for every artifact it reads,
requires opset 18, derives the per-request ABI from the producer manifest, and
delegates model-neutral ONNX conversion to `tools/export_safetensors.py`.

```bash
SOURCE=/path/to/tiny-receipt-reader-digit-slots-2m

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

The `ptq` variant is **not** a re-encoding of `int8`. The producer quantizes
convolution only (`quantized_op_types: ["Conv"]`), so its graph keeps float
GroupNorm and float SiLU that VolvoxAI would have quantized natively. The two
paths cannot converge numerically; see [typed PTQ](../../docs/typed-ptq.md).

Calibration is a separate explicit stage because activation affines can only
come from running the graph on real receipts:

```bash
# 1. stage a prepared FP32 graph and calibrate it
python3 -m examples.receipt_digit_reader.tools.import_hf_onnx \
  --source "$SOURCE" --out-dir build/receipt-digit-reader-ptq --variant ptq \
  --calibration build/calibration.json
```

The importer stages `.build/receipt-digit-reader-ptq.import/prepared` on the
first run; calibrate against that exact revision, since a profile measured on
any other graph revision is rejected:

```bash
node --import tsx examples/receipt_digit_reader/tools/calibrate.mjs \
  --package build/.receipt-digit-reader-ptq.import/prepared \
  --raw calibration_batch.f32 \
  --out build/calibration.json
```

`--raw` takes an already-normalized `[N, 1, 320, 672]` F32 batch, which is what
a producer pipeline naturally emits and keeps this tool free of an image-decoder
dependency. `--images` decodes files instead when `sharp` is installed.

Calibration images must be disjoint from any split used for evaluation.

### Coverage is convolution-only by default

Quantizing more of the graph is not free. Measured on the 2,000-image held-out
split, every row sharing one calibration profile over 256 real receipts. These
use the corrected annotations published 2026-08-15; an earlier contaminated
label set understated every row by roughly five points of `target_exact`.

| PTQ coverage | `target_exact` | `answer_exact` | phone | street | weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Conv only (default)** | **0.9685** | **0.9880** | 0.9925 | 0.9760 | 4.10 MB |
| Conv + Linear | 0.9600 | 0.9860 | 0.9840 | 0.9755 | 2.92 MB |
| + GroupNorm/SiLU/Add/LayerNorm | 0.9420 | 0.9855 | 0.9690 | 0.9710 | 2.63 MB |
| *(FP32 reference)* | 0.9690 | 0.9875 | 0.9940 | 0.9750 | 9.91 MB |

The default matches what the producer's own INT8 graph quantizes and lands
within 0.05 points of FP32 on `target_exact` — one receipt in 2,000 — while
scoring the highest `answer_exact` of any route measured, FP32 included.

An independent re-measurement over the same 2,000 images reproduces the FP32
reference row exactly and puts a freshly calibrated Conv-only profile — a
different 256 receipts — at 0.9690 / 0.9880 / 0.9930 / 0.9760, one receipt
above the row here. The rows above share one profile and stay as measured;
[the benchmark](../../docs/receipt-digit-reader-benchmark.md#held-out-accuracy)
carries the cross-route comparison against ONNX Runtime.

Widen it with `--ptq-float-op` when package size matters more than accuracy: quantizing the readout as well trades
2.7 points of `target_exact` for 1.5 MB. The cost lands almost entirely on
`phone_digit_exact` (0.9925 to 0.9690) rather than on `street_exact` (0.9760 to
0.9710), because a phone number needs seven consecutive digits right while a
street number needs three.

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
  --wasm-url dist/0.4.0/volvoxai.wasm
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
    name: 'slots', logitsName: 'slot_logits', targets,
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

```bash
python3 -m examples.receipt_digit_reader.tools.benchmark_backends \
  --package build/receipt-digit-reader-ptq \
  --image receipt.jpg --onnx "$SOURCE/model_int8.onnx" \
  --native-binary build/cmake/native/receipt_digit_reader \
  --api dist/0.4.0/volvoxai.js \
  --repeat 30 --warmup 5 --pin-cpu 0
```

This publication-safe report contains route timing, record-agreement booleans,
exact package `manifest.json`, `graph.json`, and `model.safetensors` hashes, and
exact native/API/WASM/ONNX artifact hashes, but no machine or receipt path,
decoded digits, or captured route stderr. `--pin-cpu` temporarily pins the
parent (including in-process ONNX Runtime) and child routes, then restores the
parent's original affinity.

Each route holds one session open and times execution, synchronization, and the
owned output snapshot only — model load, compilation, preprocessing, and
process startup are excluded, and the reported statistic is the median. A route
whose record changes between runs is a failure, not a fast result.

The harness refuses to measure on a contended host: `taskset` does not isolate
a route from an unrelated job spread across every core, and that contention has
already moved the ONNX Runtime reference by 1.8×.

The legacy receipt Runtime-mode benchmark has been removed. It depended on
private object-API diagnostics that the public proto contract does not expose:
typed batch-semantics evidence, per-result scheduling evidence, execution
inspection, and structured WebGPU adapter identity. The generated
DIRECT/SCHEDULED lifecycle remains the supported API surface, but this example
no longer ships a receipt-specific harness that would imply physical-batch
proof without those typed public fields.

Current medians on one core are in
[the benchmark record](../../docs/receipt-digit-reader-benchmark.md). It records
the clean quiet-host backend run and retained historical Runtime batching
qualification captured before the generated-API cutover. PTQ is the fastest
VolvoxAI route, while native CPU still trails ONNX Runtime; historical
CPU-JS/WASM comparison data for the measured device found no benefit above B1.
For current WASM deployment, keep
`scheduler.maxBatchSize: 1` until a route-specific measured `T(B)` policy says otherwise.
SCHEDULED still provides global admission and fairness at that cap; one-shot
callers use DIRECT to bypass Runtime coordinator allocation.

### RTX 3090 WebGPU and CUDA qualification

A separate Deno 2.9.3/Vulkan WebGPU campaign ran the fresh FP32 and INT8 B=19
packages on an RTX 3090 with final API SHA-256
`3337273dd7b9e87e5e865f457f6b602f4769e84225ee21f28480f852e33f49b1`.
These immutable reports predate the execution-mode rename. Their table and
artifact labels are retained verbatim: `SIMPLE` maps to DIRECT, `ADAPTIVE` maps
to SCHEDULED with zero batch delay, and `SERVICE` maps to SCHEDULED with a
positive bounded delay. The SIMPLE B1 report remained scheduler-free
(`schedulerAllocated=false`); the ADAPTIVE B15 report turned 150 logical
requests into ten physical executions, and every lane was byte-exact against
an independent same-WebGPU SIMPLE B1 reference.

| Variant / retired plan label | B | Median group ms | Logical req/s | B15 throughput vs B1 |
| --- | ---: | ---: | ---: | ---: |
| FP32 SIMPLE | 1 | 86.787 | 11.382 | — |
| FP32 ADAPTIVE | 15 | 201.578 | 72.701 | **6.387x** |
| INT8 SIMPLE | 1 | 87.044 | 11.400 | — |
| INT8 ADAPTIVE | 15 | 231.141 | 64.655 | **5.672x** |

The full B4/B8/B12/B15 table and report hashes are in
[the benchmark record](../../docs/receipt-digit-reader-benchmark.md#rtx-3090-deno-webgpu-large-b-sweep).
The complete static WebGPU domain is legal through B=19; B=20 is rejected by
the adapter's 128-MiB storage-binding limit. Actual Deno/wgpu execution reached
B15 and normalized B16 to `OUT_OF_MEMORY` before publishing any output. GPU
memory telemetry was 4 MiB idle, 1,506 MiB at successful B15 peak, and 4 MiB
afterward; the failed B16 process peaked at only 226 MiB and returned to 4 MiB.
This was not 24-GiB device exhaustion. An isolated fresh-device probe confirms
the exact `v1` buffer succeeds at B15 (103,219,200 bytes) and fails at B16
(110,100,480 bytes) in 3/3 processes; a 109,051,904-byte (104-MiB) spot check
still succeeds. The effective single-buffer boundary is therefore
`>109,051,904` and `<=110,100,480` bytes, despite advertised 128-MiB binding
and 256-MiB buffer limits. This is an advertised-versus-effective allocator
policy mismatch in the measured Deno/wgpu/Vulkan/NVIDIA 535 stack, not graph
generation coexistence. B15 is its observed graph boundary, not a universal
maximum.

The reviewed campaign directory is
`/path/to/volvoxai-gpu-validation/results/webgpu-3337273dd7b9/`;
`campaign-summary.json` hashes to
`a58f07d39ab15c431a4c37cc4fb17acbb1ab6add8478ada5432b98ef7e7caaa0`.
The adjacent `isolated-v1-allocation/allocation-boundary-summary.json` hashes
to `8e8df356592252fa341fd6bfd326d26ae5b84c0811f4f745e0293827a4b93d96`.

That exact campaign depended on the now-removed pre-proto receipt Runtime-mode
harness. The archived reports remain valid historical evidence, but current
repository tooling does not reproduce them until equivalent typed public
evidence is added to `proto/volvoxai.proto`.

Native CUDA has a narrower qualification: an explicitly authored bulk B=4
tensor was byte-exact to four same-CUDA B1 lanes and provided about 1.30x
useful throughput for both FP32 and INT8. That retained run predates the
current proved symbolic-B coordinator path, so it does not claim Runtime
scheduler coalescing. Current Vulkan/OpenGL/CUDA coalescing has separate
one-forward qualification evidence. The fresh FP32/INT8 reports are
`results/native-cuda/digit-{fp32,int8}-cuda-b4-parity-fresh.json` under the
same remote validation root; their SHA-256 values are
`6ecf9d58f0b854bec6e7da9002a86547fea7cc09194ac1bc927c82d6dc945e6e`
and `dc4229f919a4582be77839dc7cebd07b9be06e639985ca94759cc7346c726711`.
