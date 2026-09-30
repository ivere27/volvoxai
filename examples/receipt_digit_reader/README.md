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

## Explore the browser workbench

Serve the repository root and open
[`receipt_digit_reader.html`](../receipt_digit_reader.html). Prepare the FP32 and
INT8 packages with the download/import workflow below. The persistent header
selects the model variant and backend; **Run inference** uses the current image.

**Input & results** shows the receipt, decoded digits, sample reference and
model context. Question routing is an optional host-side demonstration. **Debug**
explores execution and tensor values. **Performance & memory** measures an
ordinary inference. These are full workspace tabs; changing tabs neither runs
the model nor starts a capture. VolvoxAI is a unified edge engine for native C,
native GPU, WASM and WebGPU. This browser example offers WASM and WebGPU.

The [common workbench](../common/workbench/README.md) provides the shell,
debugger and profile analysis. The [vector example](../debugging.html) uses the
same components without receipt-specific controls. Image preparation, digit
decoding and spatial mapping remain in this example's adapter.

### Step through execution

Open **Debug**, then **Start session** to prepare an independent execution of
the current image. Opening the tab alone creates no engine session.

The **Execution** navigator searches by node ID, operator or tensor name. Click
a gutter dot to break before a node, or use the breakpoint and capture-issue
filters to narrow the list. The **Graph** follows actual tensor connections.
Drag its background to pan, use the zoom buttons or Ctrl/Command + scroll to
zoom, and **Center** to return to the selected node. Click an operator to inspect
it, or an edge to select the receiving node's input tensor. Double-click an
operator to toggle its breakpoint. Blue marks the selection; orange marks the
next node to run. Selection alone never moves the execution cursor.

**Step** (F10) executes one schedule entry and selects its observations.
**Continue** (F8) stops before the next breakpoint and selects that pending node;
Step then captures its inputs and outputs. **Run to selected** adds a temporary
stop before a future node while respecting earlier breakpoints. **Locate cursor**
returns to the execution position. **Restart** retains breakpoints when the new
execution has the same schedule. F9 toggles the selected node's breakpoint,
`/` searches, and arrow keys navigate the focused execution list.

The right-hand inspector lists the selected node's inputs and outputs, including
planned tensors that have not run yet. **Summary** shows shape, dtype, size,
capture status, full-snapshot statistics, nonfinite counts and quantization.
Producer/consumer links follow the selected tensor through the graph.
**Metadata** exposes its exact structured record. The central analysis area
switches between **Graph**, **Tensor values** and **Spatial / image**, giving
numbers and image maps the same space as the graph.
Drag the pane dividers to resize the workspace. On smaller screens, use the
Execution, Analysis and Inspector pane buttons. Left/right arrows focus tabs;
Enter or Space activates them, so moving keyboard focus never triggers a read.

**Capture settings** applies to the next session. The default keeps statistics
and original node boundaries. Choose **Optimized graph** to include fused and
skipped entries, or enable **Retain raw values** to retain tensor bytes. Exact
tensor names, separated by commas, restrict capture. The footer shows retained
capture memory and its budget, with loss marked explicitly; hover for peak and
dropped-record counts. This budget excludes execution buffers and shared weights.

In **Tensor values**, browse 64, 256 or 1024 elements at a time. Expand **Jump to offset /
coordinates** to enter a flat offset or coordinates such as `[0, 0, 123, 456]`.
Each row keeps the exact stored
value alongside its affine dequantization when available. The **Page heatmap**
shows the same flat page as a grid, with hover inspection and a color scale.
It is not a spatial tensor slice. Colors and the distribution chart describe
only the displayed page; the Summary statistics describe the entire captured
tensor. Nonfinite elements remain explicit and are excluded from histogram bins.
**Page CSV** saves the displayed rows, and **Raw tensor** downloads the full
captured `.bin`. Viewing pages never advances execution and reads at most 4 KiB
per page. The Summary and Metadata tabs do not read raw tensor bytes.

### Follow the image through the graph

Select a node's input or output and open **Spatial / image**. Choose **Capture this
tensor** if only statistics are available; this explicitly starts a new session
and runs through that node. **Step** executes the pending node and captures its
inputs and outputs. Select a channel to see its actual
activations, or **Mean |activation|** to reduce all channels into one map.
Quantized tensors offer stored integers and affine dequantized values.

The **Display** menu stays visible in **Spatial / image**, including before capture;
it becomes enabled when a map is available. **Receipt overlay**, the default,
aligns the map with the displayed input image. **Feature
map** shows its native grid, and **Input image** lets you inspect the receipt
without colors. Hover the image, or focus it and use arrow keys, to read exact
tensor coordinates, values and input sampling centers. Colors use the current
plane's minimum and maximum; magenta marks nonfinite values. **Pin for
comparison** keeps one map while you step, change channels or recapture another
tensor. **Lock color scale** shares one range between the current and pinned
maps; compare maps with compatible units. Each caption identifies the step,
tensor, input/output side and selection. Hover the caption for session provenance.

For digit localization, use **Capture attention**. It runs a new original-graph
session to completion, ignoring breakpoints for this explicit preset, and retains
only the three spatial Softmax outputs. Choose an attention pass and a **Phone**
or **Street** digit slot. There are 12 phone and four street slots, including blank
positions. Each slot has 420 weights on the verified 42 × 10 feature grid.
The three outputs contain 78.75 KiB of raw values; events, statistics and plan
metadata also consume capture storage. This preset does not retain final logits.

The overlay follows the graph's convolution stride, padding and kernel geometry,
Transpose axes, and contiguous spatial Reshape operations. It does not guess a
rectangle from element count. Coordinates refer to the resized receipt shown
on the page, not an uploaded file's original dimensions. Tensor weights, pooled
features and logits have no verified image mapping and are labeled accordingly.
Activations and attention show where features are represented or read; they are
**not causal attribution**. GroupNorm and attention can depend on the entire
image, so a grid cell is not a local receptive-field boundary.

Spatial data is read only when this tab displays a retained map, in chunks of
at most 64 KiB. A planar channel or attention slot can be read directly;
interleaved channels require scanning the enclosing tensor span. The view keeps
one float64 plane plus at most one pinned plane, not a full feature tensor. For
the first 336 × 160 feature grid this is 420 KiB per plane; image and canvas
storage are separate. Each plane request scans at most 64 MiB and maps are capped at
1,048,576 cells. Changing opacity, display mode or color scale uses cached data.
Releasing the session or changing the receipt/model clears the comparison;
changing the model also clears tensor-name filters from the previous graph.

**Map JSON** exports the selected plane in row-major `y,x` order, its graph-derived
coordinate mapping, source and snapshot identity, selector, value domain and
display range. Values are decimal strings to preserve nonfinite values and
negative zero for analysis tools. The workspace's **Export JSON** includes the
current spatial metadata, with values kept in the separate map export.

### Capture lifetime and export

For statistics-only, optimized-away or budget-lost snapshots, **Recapture this
tensor** explicitly replaces the session and reruns the same receipt through
the selected node. It retains only that tensor and input/output side, preserving
original node boundaries when necessary. The selected tensor must still fit
the budget. Historical values that were never captured cannot be read without
rerunning. A breakpoint before a node has no output snapshot until it executes.

**Export JSON** saves the source description, execution plan, events, tensor
metadata and statistics for people and analysis tools. Raw bytes are separate
downloads named by session and snapshot IDs. **Stop** preserves observations;
**Release** frees the session. Changing the receipt, variant or backend releases
the old capture. Normal inference and profiling remain independent; ordinary
inference can run while the debugger is paused. The model controls remain
available in the header; they lock while a debug operation is in flight.

Compilation and capture use the generated `VxDebugService` only on explicit
request. Debug execution is not a performance measurement; GPU debugging may
synchronize the device. See the [debugging guide](../../docs/debugging.md) for
engine semantics and the observation contract.

### Measure performance and memory

Open **Performance & memory** and press **Record inference**. The operator table
starts with the largest measured host duration. Search and sort by execution
order, host duration, GPU elapsed time or logical tensor size. Selecting a row
links the host timeline to its source nodes and tensor placement. **Inspect
tensor in debugger** opens that output in the independent debug session.

**Memory & resources** separates WASM allocated blocks, reusable free blocks,
allocator metadata, module prefix, page slack and untracked reservations. It
also shows the largest free block, allocator live/peak capacities and typed
CPU/GPU observations with their scope and availability. Unsupported readings
are never shown as zero. These overlapping accounting domains are not added
together. A profiled heap includes the capture collector and any retained
debug session; release the debugger before measuring inference alone.

**Pin baseline** retains wall time and module-heap size for comparison with the
next capture. Each result is one instrumented run, not a statistical benchmark;
use the [benchmark workflow](benchmarks/README.md) for repeated measurements.
**Analysis JSON** exports typed events, resource samples and execution plans;
**Chrome trace** and **Open Perfetto** retain the full trace and clock evidence.
Operator duration is not CPU/GPU utilization. Browser process CPU/RSS and
hardware GPU utilization remain explicitly unsupported by these APIs.

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
  --wasm-url dist/0.7.0/volvoxai.wasm
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
