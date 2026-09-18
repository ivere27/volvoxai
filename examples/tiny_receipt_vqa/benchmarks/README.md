# Receipt accuracy and latency with the proto API

The [VQA benchmark](../BENCHMARK.md) and
[digit reader benchmark](../../receipt_digit_reader/BENCHMARK.md) retain the
latest qualified measurement for each backend/package. Each example publishes
`reports/benchmark.json` and `reports/validation.json`; publication
replaces these files and updates the tables without appending dated reports.

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

Copy the specification, referenced input/model files, released `dist/0.5.0`
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

- `execute_ms` ends when the result is READY, before `ReadOutput` or
  `ReleaseResult`. Only pending work needs completion polling.
- `read_output_ms` measures the output API calls needed by the caller.
- `release_result_ms` measures result release.
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
