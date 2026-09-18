# TinyReceiptVQA BPE1536 benchmark

[Source model on Hugging Face](https://huggingface.co/ivere27/tiny-receipt-vqa-structured-qa-21m) · [Download and run PTQ](README.md#download-the-model)

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
Each input has 30 measured requests per version (15 per round). Input 0
executes the decoder 36 times (33 for Vulkan/OpenGL INT8); input 1
executes it 20 times. Both include
the encoder, generation to EOS, required host reads and tensor release.
Model loading, compilation and separate API profiling are excluded.
Medians are reported per input rather than pooling different lengths.

**Milliseconds per complete question.** Positive reduction means faster.

| Backend | Model | Input / decoder steps | Before | After | Reduction |
| --- | --- | --- | ---: | ---: | ---: |
| cuda | fp32 | 0 / 36 | 246.30 | 163.81 | +33.5% |
| cuda | fp32 | 1 / 20 | 148.99 | 104.39 | +29.9% |
| cuda | int8 | 0 / 36 | 270.04 | 177.67 | +34.2% |
| cuda | int8 | 1 / 20 | 161.14 | 112.65 | +30.1% |
| vulkan | fp32 | 0 / 36 | 1041.66 | 789.61 | +24.2% |
| vulkan | fp32 | 1 / 20 | 630.15 | 496.68 | +21.2% |
| vulkan | int8 | 0 / 33 | 944.47 | 797.22 | +15.6% |
| vulkan | int8 | 1 / 20 | 622.77 | 539.35 | +13.4% |
| opengl | fp32 | 0 / 36 | 636.56 | 491.54 | +22.8% |
| opengl | fp32 | 1 / 20 | 408.82 | 328.80 | +19.6% |
| opengl | int8 | 0 / 33 | 624.39 | 486.60 | +22.1% |
| opengl | int8 | 1 / 20 | 428.36 | 345.29 | +19.4% |

After uses `reuse_inputs` for fixed decoder memory and `feedback` for K/V
and the growing padding mask. Only logits are exported from the decoder.
The unchanged input copies and intermediate K/V snapshots disappear;
feedback still performs one device copy into the next graph input.
The kernels and Python greedy token loop are unchanged.

To separate the changes, round 0 also measures the new pooled/batched
API while continuing to export K/V. These figures have 15 samples per
input, and are not combined with the two-round final measurement:

| Backend | Model | Input | Pooled/batched, exported K/V (ms) | Context feedback (ms, round 0) |
| --- | --- | ---: | ---: | ---: |
| cuda | fp32 | 0 | 209.28 | 163.38 |
| cuda | fp32 | 1 | 129.45 | 104.19 |
| cuda | int8 | 0 | 235.20 | 177.66 |
| cuda | int8 | 1 | 138.56 | 111.93 |
| vulkan | fp32 | 0 | 893.54 | 775.94 |
| vulkan | fp32 | 1 | 584.08 | 494.46 |
| vulkan | int8 | 0 | 859.23 | 804.53 |
| vulkan | int8 | 1 | 558.06 | 558.35 |
| opengl | fp32 | 0 | 555.28 | 491.36 |
| opengl | fp32 | 1 | 362.05 | 329.42 |
| opengl | int8 | 0 | 548.01 | 486.06 |
| opengl | int8 | 1 | 380.13 | 345.15 |

For the 36-step CUDA FP32 request, decoder public calls fall from 467
to 108: 36 executions, 36 logits reads and 36 batch releases.
API timings are profiled separately and must not be added to benchmark
latencies. `summary` also reports milliseconds per generated token,
excluding the first decoder step.

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
Unrelated CPU activity averaged 0.46–1.29 cores across these routes.
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
One item is one image/question pair through EOS.

| Host | Backend / client | FP32 | Imported INT8 | Native C PTQ | WASM C PTQ |
| --- | --- | ---: | ---: | ---: | ---: |
| Local | ORT CPU / Python | 288.13 / 308.04 | 181.70 / 198.41 | — | — |
| Local | Native CPU / Python → C | 657.31 / 796.08 | 704.01 / 882.01 | 663.59 / 820.53 | 733.63 / 826.29 |
| Local | WASM / Node 20.11.1 | 1676.18 / 1931.24 | 1546.89 / 1886.87 | 1271.23 / 1640.43 | 1270.28 / 1608.54 |
| GPU host | CUDA / Python → C | 288.70 / 367.08 | 314.86 / 402.30 | 301.98 / 382.60 | 301.62 / 383.84 |
| GPU host | Vulkan / Python → C | 742.10 / 920.41 | 754.19 / 914.45 | 739.62 / 923.21 | 738.70 / 919.58 |
| GPU host | OpenGL / Python → C | 654.52 / 807.94 | 689.68 / 828.55 | 663.29 / 823.01 | 665.01 / 821.88 |
| GPU host | WebGPU / Deno 2.9.6 | 8039.49 / 10361.55 | 5536.05 / 6851.89 | 5403.13 / 6847.61 | 5422.00 / 6839.02 |

The JSON contains the latest qualified measurement for each backend/package cell,
including per-input statistics, component times, serial requests/s and every observation.
Each row retains its own measured timestamp and caller hash.
Decoder calls including EOS: input 0: [33, 36, 38], input 1: [20].
VolvoxAI uses 20–36 calls; original ORT INT8 on the local host uses 38/20 and matches its saved reference.
The report also records TTFT and generated tokens/s. Use `by_case` for a fixed input and length.

### CUDA execution and output reads

Median milliseconds for one complete answer (encoder plus all decoder steps):

| Package | Execute through READY | ReadOutput | Complete request |
| --- | ---: | ---: | ---: |
| fp32 | 179.685 | 95.155 | 288.698 |
| int8 | 205.953 | 95.127 | 314.864 |
| ptq | 193.147 | 95.061 | 301.984 |
| ptq-wasm | 193.115 | 94.978 | 301.621 |

CUDA uses native BufferView inputs and output destinations. Intermediate steps
read logits, eight K/V tensors and a mask for the next step; EOS reads only logits.
READY results need no GetResult. Execute timing excludes ReadOutput and ReleaseResult,
but includes public dispatch, synchronization and runtime output snapshots. It is not
GPU kernel-only time. Complete-request timing includes every needed read and cache handoff.
Other rows retain the caller and timing contract identified by their recorded hashes.

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

Accuracy is exact agreement between the extracted answer and the held-out answer label.

| Backend | FP32 | Imported INT8 | Native C PTQ | WASM C PTQ |
| --- | ---: | ---: | ---: | ---: |
| cpu | 1945/2000 (97.25%) | 1944/2000 (97.20%) | 1943/2000 (97.15%) | 1940/2000 (97.00%) |
| cuda | 1945/2000 (97.25%) | 1943/2000 (97.15%) | 1943/2000 (97.15%) | 1942/2000 (97.10%) |
| vulkan | 1945/2000 (97.25%) | 1942/2000 (97.10%) | 1945/2000 (97.25%) | 1941/2000 (97.05%) |
| opengl | 1945/2000 (97.25%) | 1942/2000 (97.10%) | 1945/2000 (97.25%) | 1941/2000 (97.05%) |
| wasm | 1945/2000 (97.25%) | 1942/2000 (97.10%) | 1943/2000 (97.15%) | 1940/2000 (97.00%) |
| webgpu | 1945/2000 (97.25%) | 1942/2000 (97.10%) | 1945/2000 (97.25%) | 1941/2000 (97.05%) |

The WebGPU result records successful serial execution with a patched Deno 2.9.6
host and a physical RTX 3090. Browser products and simultaneous WebGPU jobs
were not qualified by this audit.

For imported INT8 on native CPU, whole-token agreement with source ORT is 1482/2000
(74.1%), while extracted-answer agreement is 1956/2000
(97.8%). These are output-equivalence metrics, separate from label accuracy.
Differences in the generated `<value>` field can lower whole-sequence agreement while
leaving the final answer correct. INT8 import changes bias folding and quantized arithmetic;
their separate numerical contributions have not been isolated by this audit.

Accuracy evaluation wall time is not inference latency. The separate speed
experiment retains sessions and repeats fixed inputs under the timing contract above.
See the [benchmark workflow](benchmarks/README.md) for reproduction.

<!-- END CURRENT PROTO VALIDATION -->
