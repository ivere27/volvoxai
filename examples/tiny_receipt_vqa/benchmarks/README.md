# Receipt accuracy and latency with the proto API

The [VQA benchmark](../BENCHMARK.md) and
[digit reader benchmark](../../receipt_digit_reader/BENCHMARK.md) retain the
latest qualified measurement for each backend/package. Each example publishes
`reports/benchmark.json` and `reports/validation.json`; publication
replaces these files and updates the tables without appending dated reports.

For isolated CPU/WASM PackedLinear comparisons, the
[automatic kernel benchmark tools](../../../docs/packed-linear-benchmarks.md)
extract all projection shapes directly from the encoder/decoder graphs and
retain the generated workload for replay.

The latest native tensor comparison is separate in `reports/native-tensors.json`:
it compares Python `run()` and `run_tensors()` against a preserved previous SDK.
Use `native_tensor_handoff.py --model vqa` here, or the
[reader entry point](../../receipt_digit_reader/benchmarks/native_tensor_handoff.py).
Both use the frozen inputs and output references below. They measure FP32 and
imported INT8 on native CUDA/Vulkan/OpenGL; they do not replace the separate
WASM/WebGPU/PTQ matrix or rerun the 2,000-case accuracy audit.

```sh
python /path/to/before/examples/tiny_receipt_vqa/benchmarks/native_tensor_handoff.py \
  --source-tree /path/to/before --data-root . \
  --spec build/receipt-proto-benchmark-head-29ffb7e/spec.json \
  --model vqa --backend cuda --label before --modes resident --warmup 10 --repeat 15 \
  --out examples/tiny_receipt_vqa/benchmarks/results/native-tensors/before.json

python examples/tiny_receipt_vqa/benchmarks/native_tensor_handoff.py \
  --source-tree . --data-root . \
  --spec build/receipt-proto-benchmark-head-29ffb7e/spec.json \
  --model vqa --backend cuda --label after --modes resident stateful --warmup 10 --repeat 15 \
  --out examples/tiny_receipt_vqa/benchmarks/results/native-tensors/after.json
```

Run each SDK with its matching caller in a separate process on the same idle
GPU host. The API has no compatibility path. The `resident` mode publishes
retained GPU tensors; `stateful` keeps decoder inputs and K/V inside the
execution context, exports only logits, and reads those logits for token
selection. Both run to EOS, include result release, and verify every request
against the frozen output reference. Compare each input separately: their
generated lengths differ. Native tensor timing uses ten warmups per input;
Reader takes 30 samples and VQA 15 samples per input in each measurement round.
The `--profile-api` request runs separately from latency collection.

Accuracy covers all 2,000 held-out cases in each backend/package/profile cell:
44 cells per model, 88 total. Latency uses held-out inputs 0 and 1, selected
before timing, with three warmups and fifteen measured requests per input.
Those 30 observations per row describe repeated requests for two fixed inputs.
They do not measure the latency distribution of the entire 2,000-case corpus.
Every measured output must match its saved accuracy reference.

VQA's `benchmarks/run.py` selects only VQA. The
[reader entry point](../../receipt_digit_reader/benchmarks/README.md) selects
only the digit reader. Both use the implementation in this directory. The
combined preparation and publication workflow below handles both models so
their timing, output validation and host checks stay consistent.

Keep sources, tests, guides and the two curated reports in Git. Use
`benchmarks/work/` for private corpora and packages, `benchmarks/results/` for
unpublished runs, and `benchmarks/.venv/` for a local Python environment. The
example's [.gitignore](../.gitignore) excludes these directories, caches,
logs and temporary files. Public reports retain samples and hashes without
decoded receipt values, account names or home-directory paths.

## Prepare the inputs

Run commands from the repository root. Install the pinned Python dependencies
on each measurement host:

```sh
python3 -m venv examples/tiny_receipt_vqa/benchmarks/.venv
BENCH_PYTHON=examples/tiny_receipt_vqa/benchmarks/.venv/bin/python
"$BENCH_PYTHON" -m pip install \
  -r examples/tiny_receipt_vqa/benchmarks/requirements-receipt-benchmark.txt
```

Preparation requires converted model packages and a completed 88-cell accuracy
audit from the [reader](../../receipt_digit_reader/README.md) and
[VQA](../README.md) workflows. It verifies the audit, freezes normalized inputs
and expected outputs, and copies the six original ONNX models. It performs no
inference and does not regenerate a missing accuracy audit.

Download the source models using the
[reader instructions](../../receipt_digit_reader/README.md#download-the-model)
and [VQA instructions](../README.md#download-the-model) before importing and
calibrating PTQ. Both workflows start from the published FP32 ONNX files.
The remaining audit paths below identify the current publication.
Use a fresh private work directory for new inputs; an existing specification
and the files it hashes must remain unchanged while collecting a matrix.

```sh
DIGIT_SOURCE=examples/receipt_digit_reader/benchmarks/work/source
VQA_SOURCE=examples/tiny_receipt_vqa/benchmarks/work/source
VALIDATION=build/receipt-all-2000-head-29ffb7e
DIGIT_WORK=build/receipt-backends-head-29ffb7e
VQA_WORK=build/receipt-vqa-ptq-head-29ffb7e
DIGIT_REFERENCE=build/receipt-ptq-head-29ffb7e/verification/logits.npz
BENCH_WORK=examples/tiny_receipt_vqa/benchmarks/work/current
BENCH_RESULTS=examples/tiny_receipt_vqa/benchmarks/results

"$BENCH_PYTHON" examples/tiny_receipt_vqa/benchmarks/prepare_receipt_benchmark.py \
  --validation "$VALIDATION" \
  --digit-work "$DIGIT_WORK" --vqa-work "$VQA_WORK" \
  --digit-source "$DIGIT_SOURCE" --vqa-source "$VQA_SOURCE" \
  --digit-reference "$DIGIT_REFERENCE" --out "$BENCH_WORK"
```

To reproduce the published measurements, use the existing frozen specification
at `build/receipt-proto-benchmark-head-29ffb7e/spec.json` and its exact inputs.
The measured runtime files are retained privately in that work directory's
`measured-runtime/`. A rebuilt runtime with a different hash requires its own
matching validation and specification; rebuilding alone creates no new timing
observation.

## Measure local CPU and WASM

CPU, WASM and ORT CPU run on the original local host,
**AMD Ryzen 5 5600U**. Set `NODE20` to Node 20.11.1, matching the published
WASM environment. Tools record executable hashes, runtime versions, host
roles, numerical threads and CPU affinity without collecting CPU hardware
identities.

```sh
NODE20=/path/to/node-v20.11.1-linux-x64/bin/node
"$BENCH_PYTHON" examples/tiny_receipt_vqa/benchmarks/run_receipt_proto_benchmarks.py \
  --spec "$BENCH_WORK/spec.json" --out "$BENCH_RESULTS/local" \
  --deno build/deno/target/webgpu-fix/deno --node "$NODE20" \
  --models digit vqa --backends ort cpu wasm --host-label cpu-wasm-host \
  --cpu 0 --warmup 3 --repeat 15
```

This produces 20 rows: per model, four VolvoxAI packages on native CPU and
WASM, plus FP32 and producer INT8 on ORT CPU. The VolvoxAI packages are FP32,
imported INT8, native C PTQ from FP32 and WASM C PTQ from FP32.

The published local measurements used `--allow-busy-host` because desktop
background activity remained. Their load observations and
`host_quiet_verified: false` are retained. Add that option only when accepting
and reporting such a local measurement; it is restricted to CPU/WASM routes.
No unrelated process is paused or terminated.

## Measure GPU backends on an idle remote host

Copy the specification, referenced input/model files, released `dist/<version>/`
artifacts, both native libraries, generated Python package and measurement
tools to the remote checkout, preserving repository-relative paths. Their
hashes must match the specification. Set the same work/result variables there
and install the pinned Python dependencies.

```sh
"$BENCH_PYTHON" examples/tiny_receipt_vqa/benchmarks/run_receipt_proto_benchmarks.py \
  --spec "$BENCH_WORK/spec.json" --out "$BENCH_RESULTS/gpu" \
  --deno build/deno/target/webgpu-fix/deno \
  --models digit vqa --backends cuda vulkan opengl webgpu \
  --host-label gpu-host --cpu 0 --warmup 3 --repeat 15
```

This produces 32 rows: four packages on four GPU backends for each model.
Native and WASM latency use inference; WebGPU uses full. The accuracy audit
covers both profiles wherever supported. CPU/WASM results from the GPU host
cannot enter the local table, and publication rejects contended GPU results.

The published GPU environment is an RTX 3090 with NVIDIA driver 535.309.01.
WebGPU uses a patched Deno 2.9.6 executable, SHA-256
`ae098ccefee95494a40c5711c6abdcd07c70b7bad33df444ec65d80e19c421fe`,
built with Cargo optimization level 1 and no LTO. These measurements qualify
that serial Deno route; browser products and simultaneous WebGPU jobs require
their own validation.

The controller runs routes sequentially, pins each child to CPU 0 and requests
one numerical thread. Before, during and after each child it checks unrelated
CPU activity against a one-core limit. GPU routes also require zero initial
GPU utilization and no other active compute owner. A failed gate stops the
suite. Completed rows are reused only when specification, tools, settings and
host role match.

For a partial refresh, use a separate unpublished result directory. For
example, on the idle GPU host:

```sh
"$BENCH_PYTHON" examples/tiny_receipt_vqa/benchmarks/run.py \
  --spec "$BENCH_WORK/spec.json" --out "$BENCH_RESULTS/cuda-refresh" \
  --deno build/deno/target/webgpu-fix/deno \
  --backends cuda --host-label gpu-host --cpu 0 --warmup 3 --repeat 15
```

## Timing and output handling

Models and execution contexts stay loaded across warmups and measured
requests. Request timing starts with preprocessed pixels and question token
IDs. It includes proto input construction, dispatch, completion, required
output reads, output materialization and result release. VQA includes the
encoder and the complete explicit-KV greedy decoder loop through EOS, token
selection and cache rebinding. VolvoxAI starts with a masked zero sentinel;
the original ORT full-answer path starts with an empty cache.

Native measurements use a generated Python client calling the native C
runtime. WASM/WebGPU use generated JavaScript calling the released C/WASM
owner. ORT uses Python CPU sessions. A separate C-only caller was not timed.
Loading, compilation, context creation, preprocessing, correctness comparisons
and report I/O are excluded.

The current drivers expose these intervals and call counts in each sample:

- `execute_ms` ends when the completed `Execute` packet arrives, before output copies or
  buffer release. The packet contains selected owned-buffer snapshots.
- `copy_tensors_ms` measures batched owned-output copies into the caller’s arrays.
- `release_buffers_ms` measures owned-output buffer release.
- `runtime_execution_ms` is the original Execute report's interval. Native
  CUDA includes runtime binding, synchronization and output snapshots. An
  asynchronous WebGPU report need not include subsequent completion polling.
  This is not a GPU kernel-only timer.

VQA also records encoder and decoder totals. The request timer includes every
needed read, token selection and cache handoff. Ten encoder outputs feed the
decoder. Intermediate decoder steps read logits, eight K/V tensors and a
padding mask; EOS needs only logits. Native payloads use caller-owned
`BufferView` memory. The explicit-KV model still hands caches through host
memory because the public API does not expose retained device outputs as
subsequent inputs. WASM/WebGPU use inline payloads through their transport.

Each published row retains its measured timestamp and caller hash. Only VQA
CUDA currently has the detailed execution/read/release breakdown; other rows
retain their measured component definitions. Use complete-request latency
across those rows. Short smoke checks do not replace the formal latency
matrix or the 2,000-case accuracy audit.

Reports include every sample, input index, median, mean, min/max and linearly
interpolated p95. VQA also includes time to first token, decoder calls and
generated tokens/s. Counts include EOS and exclude BOS. Serial requests/s is
requests divided by total request time; it does not measure concurrent serving.
Use `by_case` to compare a fixed input and generated length.

## Publish the latest measurements

Copy GPU results back to the documentation checkout. The publisher selects
the newest completed `ended_utc` for each model/backend/package, independent
of directory order. Matching timestamps with conflicting records fail. All
selected rows must share the specification, model/runtime hashes, case set,
warmup and repetition counts. The complete matrix must contain 20 local and
32 idle GPU rows, with validated outputs and statistics recomputed from samples.

```sh
"$BENCH_PYTHON" examples/tiny_receipt_vqa/benchmarks/publish_receipt_benchmarks.py \
  --spec "$BENCH_WORK/spec.json" --accuracy "$VALIDATION/verification-summary.json" \
  --local-results "$BENCH_RESULTS/local" \
  --gpu-results "$BENCH_RESULTS/gpu" "$BENCH_RESULTS/cuda-refresh"
```

Omit the refresh directory if only the complete GPU suite was run. For the
current publication, the frozen inputs are `build/receipt-proto-benchmark-head-29ffb7e`:
`results-local-node20-20260915`, the idle GPU result directory, and
`output-read-cleanup-20260915/cuda`. The last directory supplies the four
latest VQA CUDA rows; the selected matrix contains only one row per cell.

Publication overwrites each example's `reports/benchmark.json`,
`reports/validation.json` and current blocks in `BENCHMARK.md`. Per-row samples,
timestamps and caller hashes stay intact. The 44-cell accuracy audit per model
remains separate from the 26-row latency table. No dated report copies,
before/after tables or diagnostic history are published.

## Extension PTQ and packed arithmetic comparison

`extension_ptq_profile.py` and `extension_ptq_profile.mjs` profile the separately
built receipt extensions through native CPU and a single headless Chrome page.
They use the extension's public example images and exact preprocessing. Run
`--case-index 0` and `--case-index 1` separately; for VQA each selects the first
question of that public example and generates through EOS. This small repeated
latency sample does not replace the 2,000-image digit accuracy audit or all 16
VQA questions.

The native runner fixes `cpu_threads=1`. Chrome uses one isolated WASM owner,
without an extension UI, a second tab or a shared worker. Both retain compiled
sessions, record setup and the first request separately, then perform two warmup
requests and five measured requests. Cold means a fresh owner/context; neither
the OS file cache nor Chrome's compiled-module cache is flushed. Input preparation
and file reads precede request timing. Node/memory traces and output byte hashes
are collected in separate requests after timing. Native `run()` includes its
NumPy output copies; Chrome measures inline protobuf output transport. These
wall timings are not pure numerical kernel timings.
The separate native API probe times generated `Execute`, `CopyTensors` and
`ReleaseBuffers` calls inside the session. Its copy-service time includes
dispatch and codec work; the remaining wrapper time includes NumPy allocation
and input validation. Chrome retains the wall time of its traced request to
compare with C operation/node spans. Neither reports an invented memcpy-only
duration when the runtime has no such observation.

Freeze both release artifacts and models before modifying them. A model archive
has `digit/{fp32-prepared,ptq/...}` and `vqa/{fp32-prepared,ptq/...}` directories.
An archived wheel's `libvolvoxai.so` and `libsynurang_module_host.so` can be
extracted together to keep the original runtime available after wheel updates.
`VOLVOXAI_LIBRARY` selects that exact library; reports retain its SHA-256. Use
the same processor affinity and stop task-owned builds/calibration during the
final measurements. Record any remaining desktop/background activity.

```sh
export VOLVOXAI_SOURCE=/path/to/volvoxai
EXTENSIONS=/path/to/receipt_extensions
BENCH=examples/tiny_receipt_vqa/benchmarks

VOLVOXAI_LIBRARY=/path/to/retained/libvolvoxai.so \
  "$EXTENSIONS/.venv/bin/python" "$BENCH/extension_ptq_profile.py" \
  --extensions "$EXTENSIONS" --models-root /path/to/retained/models \
  --work build/receipt-profile/before-case0 --label before --case-index 0 \
  --warmup 2 --repeat 5 \
  --variants w8a8 w4a8 w4-codes-i8 w8-group32-mse

node "$BENCH/extension_ptq_profile.mjs" \
  --spec build/receipt-profile/before-case0/spec.json \
  --runtime-dir /path/to/retained/runtime \
  --puppeteer-root "$EXTENSIONS" \
  --out build/receipt-profile/before-case0/chrome.json
```

Repeat with the new library/release and the same archived models to isolate the
runtime change. Then omit `--models-root` to use newly generated extension
models, isolating the mixed Conv-I8/PackedLinear-I4 recipe. Add `--feedback` to
compare internal KV feedback with explicit KV export on identical models; check
complete token signatures, not just generated text. The scripts record per-token
execution time and input/output byte counts. `--prepare-only` writes a spec and
controls without running native measurements.

The first two controls preserve the exact Conv and other non-packed tensors;
the third preserves the Conv4 model's logical numerical weights:

- `w4-codes-i8` repacks the original signed I4 codes losslessly into I8 words,
  retaining every group scale. Matching output hashes verify that the only
  intended difference is storage width and unpacking. This is a numerical
  control, not an independently fitted W8 model.
- `w8-group32-mse` fits grouped I8 Linear weights from the original FP32 package,
  then replaces only the PackedLinear banks in the W4 graph. It measures a
  practical I8/group32 alternative under the same Conv precision and explicit
  `dynamic-i8` compilation. It needs its own accuracy qualification before use
  as a product artifact.
- `conv4-codes-i8` reads the opt-in `w4a8-conv4` package and expands only its
  packed Conv codes into ordinary I8 OHWI storage. It preserves Conv scales,
  centered integer values, activation affines and all PackedLinear tensors.
  Matching outputs isolate the cost of Conv packing and its runtime path from
  the accuracy effect of fitting four-bit weights. Include `w4a8-conv4` in
  `--variants` to measure the actual packed package alongside this control.

The inventory attributes stored tensor bytes to their consuming operator(s),
counting shared storage once. Node traces distinguish Conv, PackedLinear,
QuantizeLinear/DequantizeLinear and copy-like operators. Dynamic activation
preparation and group scaling remain inside PackedLinear's node duration.
The allocator trace covers declared arenas/result buffers, not every libc
scratch allocation; never interpret absent events as allocation-free execution.

`packed_i8_breakdown.c` supplies a bounded synthetic probe of activation
preparation, integer dots plus group scaling, the complete packed forward
function, and scratch malloc/free. It uses `[8,320] × [1280,320]`, group 32,
and reports medians over nine trials. Its independently measured blocks must
not be added as model self-times. Build both probes with identical flags:

```sh
cc -std=c11 -O3 -ffp-contract=off -I native/src/kernels \
  "$BENCH/packed_i8_breakdown.c" -o build/packed-i8-sse2
cc -std=c11 -O3 -ffp-contract=off -U__SSE2__ -I native/src/kernels \
  "$BENCH/packed_i8_breakdown.c" -o build/packed-i8-scalar
build/packed-i8-sse2 > build/packed-i8-sse2.json
build/packed-i8-scalar > build/packed-i8-scalar.json
```

On x86, the second build selects the previous scalar unpacking loop while
keeping the same compiler target. `test_packed_linear_i8` independently checks
all dot lengths through 256, code extrema, unaligned inputs, all supported group
sizes, group tails, per-row scales and partial output ranges.

For a retained matrix named `before-case0`, `runtime-case0`, `mixed-case0` (and
case 1), plus `controls-case0` and `explicit-case0`, run
`python summarize_extension_ptq_profile.py /path/to/matrix`. It writes
`summary.json` after checking unchanged inputs/models for the runtime control,
byte-identical lossless-I8 outputs, unchanged non-packed tensor banks, and exact
logits/tokens for the feedback control. The summary retains medians, nearest-rank
p95, cold/setup, decoder positions, operator shares, copy-service measurements,
storage and hashes. Add `--task-jobs-paused` only when the coordinator actually
stopped other task work during measurement; `host_quiet_verified` remains false.

`GetTraceSummary(OPERATOR)` merges operator totals across contexts, so its first
lineage cannot identify an encoder or decoder. The Chrome collector requests
up to 4096 `NODE` rows and the analyzer joins their preserved context IDs. A
truncated node summary is reported as runtime-aggregate coverage instead of
inventing per-role timings. To refresh only profiling in an existing report,
add `--refresh-profiles /path/to/report.json --profile-model vqa` to the Chrome
command with the same `--spec`, runtime and `--out` report path. It checks runtime
hashes and token signatures and replaces only profile records; cold and warm
latency samples remain unchanged.

`packed_i8_tiling.c` compares complete dynamic-I8 projections for one-token
and encoder-sized inputs, including a short K/N tail. It reports deterministic
output hashes and five-batch warm medians after two warmups. The CPU/WASM
4-by-4 tile uses the existing K-float scratch as four I8 input rows. Weight
unpacking is bounded to eight values per output; group integer dots and the
ascending F32 group accumulation remain exact. Partial rows and one-token
inputs retain the direct path. These projection timings exclude model I/O,
allocation, API transport and UI work.

Archive `native/src/kernels/packed_linear_ops.inc` before editing it. Compile
`packed_i8_tiling.c` with `-DPACKED_KERNEL_SOURCE='"/path/to/archive.inc"'`
for the old implementation and omit that definition for the candidate:

```sh
cc -std=c11 -O3 -ffp-contract=off -I native/src/kernels \
  "$BENCH/packed_i8_tiling.c" -o build/packed-i8-tiling
build/packed-i8-tiling > build/packed-i8-tiling.json
```

The same C probe exports `packed_bench_init`, `packed_bench_run` and
`packed_bench_hash` when compiled as freestanding WASM. Build with the release
container's clang-17, `--target=wasm32 -msimd128 -O3 -ffp-contract=off
-nostdlib`, the include paths `native/src/kernels` and
`native/src/runtime/wasm_freestanding/include`, and link
`native/src/kernels/wasm_heap_arena.c`. Pass `-Wl,--no-entry`,
`-Wl,--export-memory`, `-Wl,-z,stack-size=1048576` and one `-Wl,--export=NAME`
for each of those three functions. Build a test WASM with the same flags from
`native/tests/test_packed_linear_i8.c`, exporting `test_packed_linear_i8` and
`test_packed_linear_i8_tiles`. These developer modules are not release artifacts.

```sh
node "$BENCH/packed_i8_tiling.mjs" \
  --before /path/to/before.wasm --after /path/to/after.wasm \
  --test /path/to/test.wasm --puppeteer-root "$EXTENSIONS" \
  --out build/packed-i8-chrome.json
```

This runs the independent scalar oracle in Chrome, then alternates old/new
projection batches in one page. It requires matching output hashes. The oracle
covers both storage widths, every group size, SIMD and row/column tails,
nonfinite/zero/subnormal/large input rows, optional bias, and task ranges that
start or end inside a row. Native builds run it through the existing
`test_packed_linear_i8` target; ASan/UBSan can check the same source. Record
background load separately: this probe does not certify an idle machine.

For the subsequent Conv4 matrix, use `summarize_conv4_profile.py` instead of the
older matrix summarizer. It expects `before-case0/1` (old runtime and models),
`runtime-case0/1` (new runtime, identical old W4 models), `after-case0/1` (new
runtime and regenerated `w8a8`, `w4a8`, `w4a8-conv4`), and `controls-case0`.
All captures use the same two inputs, one thread, internal KV feedback, two
warmups and five samples. Snapshot the Python/JS collectors under the matrix's
`timing-callers/` before running them.

```sh
python "$BENCH/summarize_conv4_profile.py" build/receipt-conv4/final-profile \
  --task-jobs-paused --evidence /path/to/release-validation.json
```

Use `--task-jobs-paused` only when other task jobs were actually stopped.
The analyzer checks measured input/package hashes, exact runtime-only outputs,
and both lossless storage controls: `w4-codes-i8` changes only Linear storage;
`conv4-codes-i8` changes only Conv storage and its offset zero-point encoding.
It independently decodes the stored codes, verifies canonical padding, and
requires all scales, bias and other tensors to remain identical. The output
retains setup/cold/warm, per-token and per-context operator summaries, current
mode comparisons, source/runtime/model/input hashes and optional validation
report hashes. Contract/projection hashes record codegen scope; they do not
stand in for a successful codegen check. Failed validation leaves an existing
summary unchanged. This analyzer does not change the earlier PTQ matrix.

The collectors also accept VQA manifests with several weight shards per role.
`package_bytes_unique` counts the shared embedding file once for distribution
size; each role's inventory counts every shard that model loads. Shared storage
does not imply shared execution memory. The older matrix analyzers assume one
weight file per role and must not be used to total these packages.

For an explicit arithmetic comparison, add `--packed-arithmetic dynamic-i8`,
`f32` or `f16` to the Python collector and use its specification with Chrome.
This keeps the same packed weight storage and records the arithmetic returned
by compilation. Activation rounding can change outputs, so compare task answers
and numerical differences as well as latency. This developer override does not
change the extension's W4A8 recipe. Reserved dynamic-I4 has no executable kernel
and is excluded from these benchmarks.
