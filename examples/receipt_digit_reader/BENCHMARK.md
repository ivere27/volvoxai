# Receipt digit reader benchmark

[Source model on Hugging Face](https://huggingface.co/ivere27/tiny-receipt-reader-digit-slots-2m) · [Download and run PTQ](README.md#download-the-model)

CPU/WASM measurement host: **AMD Ryzen 5 5600U**.

<!-- BEGIN CURRENT NATIVE TENSOR COMPARISON -->
## Native tensor comparison — 2026-09-16

[Latest samples, API call counts and binary/model/caller hashes](reports/native-tensors.json).
Both versions use the same idle RTX 3090 host, NVIDIA driver 535.309.01,
Clang 14, Release, CUDA compute 8.6 and strict FP32. Calls use one CPU
thread and the inference profile. C owns execution; Python drives the
generated API. No other GPU compute owner was present during any run.

The previous SDK is the immediately preceding native tensor implementation.
Each version uses its matching generated client; there is no compatibility
path. Two rounds use before/after/after/before order, ten warmups per input.
Each input has 60 measured requests per version (30 per round). Timing
includes one inference, its host output and release. It excludes model
loading, compilation, correctness checks and separate API profiling.

**Milliseconds per image; per-input medians.** Positive reduction means faster.

| Backend | Model | Input | NumPy before | NumPy after | Retained before | Retained after | Retained reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda | fp32 | 0 | 6.524 | 6.782 | 6.662 | 6.575 | +1.3% |
| cuda | fp32 | 1 | 6.487 | 6.476 | 6.571 | 6.654 | -1.3% |
| cuda | int8 | 0 | 6.712 | 6.780 | 6.789 | 6.819 | -0.4% |
| cuda | int8 | 1 | 6.759 | 6.717 | 6.897 | 6.790 | +1.6% |
| vulkan | fp32 | 0 | 68.960 | 68.913 | 70.126 | 69.702 | +0.6% |
| vulkan | fp32 | 1 | 68.913 | 68.762 | 70.149 | 69.711 | +0.6% |
| vulkan | int8 | 0 | 65.946 | 65.863 | 67.811 | 67.304 | +0.7% |
| vulkan | int8 | 1 | 65.909 | 66.045 | 67.640 | 67.414 | +0.3% |
| opengl | fp32 | 0 | 61.250 | 61.264 | 61.003 | 61.100 | -0.2% |
| opengl | fp32 | 1 | 61.190 | 61.337 | 61.032 | 61.055 | -0.0% |
| opengl | int8 | 0 | 58.852 | 58.923 | 58.584 | 58.611 | -0.0% |
| opengl | int8 | 1 | 58.776 | 58.900 | 58.609 | 58.718 | -0.2% |

Reader has one inference and one required output, with no recurrent K/V
handoff. The decoder optimizations therefore offer little benefit here.
Small changes should be read with the per-round medians and raw sample
spread; they are not evidence of a general speedup or regression.

Every measured request matches its frozen output signature. This is a
latency experiment on two fixed inputs, not a new 2,000-case accuracy audit.
The separate CPU/WASM/WebGPU/PTQ qualification below retains its own
artifact hashes and measurement scope. Metal was not executed on macOS.
<!-- END CURRENT NATIVE TENSOR COMPARISON -->

<!-- BEGIN CURRENT PROTO BENCHMARK -->
## Latency — 2026-09-15

Measured from HEAD `29ffb7e7c71703f8f2df1a66fb1801afef72e284` plus the staged PTQ/runtime fixes used by the completed
2,000-case audit. [Raw samples, output and artifact hashes](reports/benchmark.json) identify the measured bytes.

CPU, WASM and ORT CPU run on the local CPU/WASM host.
CUDA, Vulkan, OpenGL and WebGPU run on the remote GPU host with an RTX 3090
and driver 535.309.01. Every retained GPU row passed the before/during/after idle gates.
Measurement tools record host roles and settings without collecting CPU hardware identities.

Local desktop background activity remained; the local rows retain `host_quiet_verified: false`.
Unrelated CPU activity averaged 1.07–1.21 cores across these routes.
The two host groups are separate measurement environments; do not interpret the table as a
same-machine CPU-versus-GPU comparison. Every route requests one numerical thread and is pinned to CPU 0.

Held-out inputs 0, 1 each receive 3 warmups and 15 measured requests
(30 observations per row). All outputs reproduce their saved 2,000-case references
and remain identical across repetitions. The complete 2,000-case accuracy audit is separate
from this fixed-input latency experiment.

**Timing:** each retained session starts with preprocessed pixels and question token IDs.
The request timer includes proto input construction, dispatch, synchronization, owned output reads
and release. VQA includes the encoder and complete explicit-KV greedy decoding through EOS.
Loading, compilation, context creation, preprocessing, validation and report I/O are excluded.
Native uses a generated Python client calling C; WASM/WebGPU use the generated JavaScript client
and released C/WASM owner. ORT uses its Python CPU session API. This run does not separately time a C-only caller.

WASM uses Node 20.11.1. WebGPU uses the pinned patched
Deno 2.9.6 executable (Cargo optimization level 1, no LTO). Executable hashes are recorded;
these WebGPU timings do not qualify a browser product.

### Complete-request latency per item

Each cell is **median / p95 in milliseconds**. Native/WASM use inference; WebGPU uses full.
One item is one receipt image.

| Host | Backend / client | FP32 | Imported INT8 | Native C PTQ | WASM C PTQ |
| --- | --- | ---: | ---: | ---: | ---: |
| Local | ORT CPU / Python | 33.22 / 34.23 | 23.57 / 24.10 | — | — |
| Local | Native CPU / Python → C | 39.21 / 40.29 | 34.37 / 35.29 | 32.00 / 32.41 | 31.96 / 32.75 |
| Local | WASM / Node 20.11.1 | 182.33 / 190.31 | 81.01 / 88.90 | 67.40 / 72.81 | 67.91 / 77.16 |
| GPU host | CUDA / Python → C | 7.88 / 8.04 | 8.08 / 8.31 | 7.40 / 8.00 | 7.85 / 8.10 |
| GPU host | Vulkan / Python → C | 70.53 / 70.92 | 68.06 / 68.41 | 67.15 / 67.57 | 67.10 / 67.56 |
| GPU host | OpenGL / Python → C | 61.47 / 61.57 | 59.01 / 59.39 | 58.52 / 58.80 | 58.54 / 58.72 |
| GPU host | WebGPU / Deno 2.9.6 | 311.56 / 319.32 | 348.75 / 367.49 | 290.44 / 307.82 | 290.98 / 308.26 |

The JSON contains the latest qualified measurement for each backend/package cell,
including per-input statistics, component times, serial requests/s and every observation.
Each row retains its own measured timestamp and caller hash.

Follow the [benchmark workflow](benchmarks/README.md) to prepare inputs,
measure the two host groups and publish only the approved results.

<!-- END CURRENT PROTO BENCHMARK -->

<!-- BEGIN CURRENT PROTO VALIDATION -->
## Accuracy — 2026-09-14

The latest audit uses HEAD `29ffb7e7c71703f8f2df1a66fb1801afef72e284` plus the staged PTQ/runtime fixes,
with exact deployed artifact hashes in the [44-row validation record](reports/validation.json).
Every backend/variant/profile cell below completed **all 2,000 held-out cases**.
Native CPU, CUDA, Vulkan, OpenGL and WASM passed in both inference and full profiles;
WebGPU passed in full. Their paired profiles produce identical digit logits or VQA
tokens, answers and router choices across all 2,000 cases. Final routes are attested
with no operator fallback. This is 44 completed cells per model, 88 for both models.

Imported INT8 uses the producer's quantized ONNX. Native C PTQ and WASM C PTQ
are separately calibrated VolvoxAI packages made from the FP32 source.

Accuracy is strict `target_exact`: every phone and street digit must be correct on the receipt.

| Backend | FP32 | Imported INT8 | Native C PTQ | WASM C PTQ |
| --- | ---: | ---: | ---: | ---: |
| cpu | 1938/2000 (96.90%) | 1934/2000 (96.70%) | 1936/2000 (96.80%) | 1936/2000 (96.80%) |
| cuda | 1938/2000 (96.90%) | 1934/2000 (96.70%) | 1936/2000 (96.80%) | 1936/2000 (96.80%) |
| vulkan | 1938/2000 (96.90%) | 1934/2000 (96.70%) | 1935/2000 (96.75%) | 1935/2000 (96.75%) |
| opengl | 1938/2000 (96.90%) | 1934/2000 (96.70%) | 1935/2000 (96.75%) | 1935/2000 (96.75%) |
| wasm | 1938/2000 (96.90%) | 1934/2000 (96.70%) | 1936/2000 (96.80%) | 1936/2000 (96.80%) |
| webgpu | 1938/2000 (96.90%) | 1934/2000 (96.70%) | 1935/2000 (96.75%) | 1935/2000 (96.75%) |

The WebGPU result records successful serial execution with a patched Deno 2.9.6
host and a physical RTX 3090. Browser products and simultaneous WebGPU jobs
were not qualified by this audit.

Accuracy evaluation wall time is not inference latency. The separate speed
experiment retains sessions and repeats fixed inputs under the timing contract above.
See the [benchmark workflow](benchmarks/README.md) for reproduction.

<!-- END CURRENT PROTO VALIDATION -->
