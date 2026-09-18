# Native tensor handoff benchmark

This benchmark measures the common C tensor storage layer through Python and
the generated native C client. CPU, CUDA, Vulkan and OpenGL use the same public
operations. Metal has an adapter but has not been executed on a Mac in this
qualification.

The current comparison is retained in `latest.json`. It contains the original
samples, library and caller hashes, correctness checks and GPU process checks.
Private work trees, raw traces and unpublished runs belong in the ignored
`work/` and `results/` directories. Replace the current report on refresh.

## What is timed

Both graphs compute `y = x + x` in float32. Inputs have shape `[2, 4]` (32 bytes)
or `[256, 1024]` (1 MiB). Model loading and compilation happen before warmup.
Every measured output is checked exactly against NumPy outside the timer.

- `single_host`: one `run()` call returning NumPy.
- `single_resident`: one `run_tensors()` call, explicit `numpy()` and release.
- `host_pipeline`: two inference calls with an intermediate NumPy array.
- `resident_pipeline`: two inference calls sharing a retained tensor, followed
  by one explicit host read and tensor release.

Reported Python times include the wrapper, generated dispatch, computation,
transfers and result release. A pipeline sample contains **two** inferences;
it is not a single-inference measurement. The C client times one
`ExecuteTensors` call with an already resident input; its final read and handle
release are outside that timer. Do not compare C and Python columns as kernel
speed measurements.

GPU residency avoids the intermediate download and upload. Inputs still copy
into the execution arena and outputs copy into independent GPU snapshots.
DLPack export shares that snapshot without an additional copy. CUDA profiler
measurements record transfer counts/bytes separately from latency timing.

## Reproduce

Preserve the complete previous source tree, generated clients, native libraries
and matching `_dlpack` extension before changing the API. Build both versions
with the same compiler and backend settings. Never load both API versions in
one Python process. For a source checkout, build the capsule bridge with
`python python/build_dlpack_bridge.py`.

```sh
python /path/to/before/python/benchmarks/native_tensors.py \
  --source-tree /path/to/before --backend cuda --label before \
  --samples 200 --warmup 20 --profile-copies \
  --output python/benchmarks/results/before-cuda.json

python python/benchmarks/native_tensors.py \
  --source-tree . --backend cuda --label after \
  --samples 200 --warmup 20 --profile-copies \
  --output python/benchmarks/results/after-cuda.json
```

`--profile-copies` requires optional PyTorch with CUDA profiling support. Omit it
for CPU, Vulkan or OpenGL. Vulkan/OpenGL use the same source-tree commands with
their backend name. Run each version with its matching caller and generated
API; the product has no compatibility path for previous schemas.

Run GPU measurements serially on an idle GPU host. The runner rejects another
GPU compute owner before, during or after a measurement. Run both CPU versions
on the original local CPU host, after builds finish. Graphics driver selection
may require `VK_ICD_FILENAMES` or `EGL_PLATFORM=surfaceless`.

The [C client](../../native/tests/test_native_tensor_client.c) is also a CTest
target in both native profiles: `test_native_tensor_client_inference` and
`test_native_tensor_client_full`. Pass a backend and iteration count to its
executable, for example `test_native_tensor_client_inference cuda 1000`.

Real-model comparisons belong to the examples:
[Reader](../../examples/receipt_digit_reader/BENCHMARK.md) and
[VQA](../../examples/tiny_receipt_vqa/BENCHMARK.md). Their complete requests
include task-specific output handling; the Add fixture isolates tensor handoff.

<!-- latest-results -->

## Current results — 2026-09-16

Both versions use Clang 14, Release, CUDA compute 8.6 and strict FP32.
GPU pairs run on the same idle RTX 3090 with driver 535.309.01. CPU pairs
run on the original AMD Ryzen 5 5600U host after all builds stop. Desktop
background activity is not certified absent; tiny differences are observations.

**1 MiB float32 input; median milliseconds.** The single call includes
host readback. Each pipeline includes two inferences and final host readback.

| Backend | Single NumPy before | After | NumPy pipeline before | After | Retained pipeline before | After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu | 1.568 | 1.724 | 2.743 | 2.577 | 2.737 | 2.702 |
| cuda | 2.767 | 2.620 | 5.032 | 4.692 | 3.435 | 2.999 |
| vulkan | 6.511 | 6.481 | 13.897 | 12.751 | 9.998 | 8.553 |
| opengl | 2.903 | 2.889 | 5.873 | 5.864 | 5.747 | 5.521 |

The generated C caller measures 1,000 executions per sample with a 32-byte
resident input; the final host read and tensor releases are outside its
timer. Each cell is the mean of three process samples. Both callers link
to the matching shared library and use the same Clang flags.

| Backend | Profile | Before (ms/call) | After (ms/call) | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| cpu | inference | 0.016322 | 0.016385 | -0.4% |
| cpu | full | 0.016038 | 0.016459 | -2.6% |
| cuda | inference | 0.073119 | 0.072330 | +1.1% |
| cuda | full | 0.071970 | 0.070598 | +1.9% |
| vulkan | inference | 0.388383 | 0.255176 | +34.3% |
| vulkan | full | 0.392340 | 0.252866 | +35.5% |
| opengl | inference | 0.150872 | 0.151050 | -0.1% |
| opengl | full | 0.149993 | 0.151250 | -0.8% |

CUDA transfer traces are separate from latency samples. The generic
retained two-model pipeline still makes one 1 MiB upload, one 1 MiB
download and three 1 MiB device copies. Pooling removes repeated output
allocations; it does not eliminate the independent snapshot contract.
The real VQA benchmark additionally uses context-owned input reuse and
feedback to remove its intermediate snapshots and unchanged input copies.
Its per-input timings and API counts are linked above.

The Linux wheel is qualified separately on CPython 3.10–3.14 and with
36 CPU/CUDA/Vulkan/OpenGL tensor tests on the GPU host. The wheel targets
CUDA compute 7.5; the source-build timings above are not wheel timings.

## Qualify CUDA stream isolation and external completion

Use an idle CUDA host with optional PyTorch and CUPTI installed:

```sh
python python/benchmarks/cuda_stream_reuse.py \
  --source-tree . --samples 12 --require-isolation --external-completion \
  --output build/validation/cuda-external-completion.json
```

This caller verifies exact results and measures idle, independent-stream and
legacy-default-stream scenarios. `--external-completion` additionally compares
ordinary DLPack final release with `torch_access()` scope exit and final release.
Scope setup and numerical readback are outside that exit-latency measurement;
it is not total application throughput. Driver callbacks run separately from
timing and check context drains, event reuse and warmed allocation counts.
Explicit completion still adds Begin/Export/End control calls. Request execution
continues to share one native engine stream. Store run-specific JSON and host
notes under ignored `build/validation/`, rather than committing machine reports.
