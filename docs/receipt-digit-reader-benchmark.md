# Receipt digit reader benchmark

This document records a historical inference-latency campaign for the receipt
digit reader, next to ONNX Runtime on the same host and the same input. It
exists to direct optimization: the headline is that
**VolvoxAI native CPU was slower than ONNX Runtime on all three variants in
this campaign**, and the gap is the work item.

> **Historical record.** CPU-JS rows and the associated runtime harness in this
> report predate removal of that provider. They remain only as immutable
> measurement evidence. Retired commands and deleted validation target names are
> intentionally omitted; current deployment choices are WASM, WebGPU, and native
> providers.

```text
source:   receipt_digit_reader_onnx_v1
package:  volvoxai-receipt-digit-reader-onnx-package-v1
example:  examples/receipt_digit_reader
```

## Packages measured

Every package is published by
`examples/receipt_digit_reader/tools/import_hf_onnx.py` from the same producer
release. The PTQ package uses the default convolution-only coverage and one
calibration profile over 256 real receipts.

| Variant | Weights | Nodes | graph.json SHA-256 | model.safetensors SHA-256 |
| --- | ---: | ---: | --- | --- |
| FP32 | 9.91 MB | 104 | `fa10d88cb56e8659…` | `1aebb58a94fe40ab…` |
| INT8 (imported ORT QDQ) | 4.98 MB | 121 | `2704cd52bb2f6b99…` | `4cc706935c1d5477…` |
| PTQ (Conv only) | 4.10 MB | 102 | `0c52b56c578f82ad…` | `82d7291451262625…` |

Digests ending in an ellipsis are display prefixes, not sufficient standalone
integrity proofs. The complete package and calibration identities remain in
the private release ledger; the producer release itself is not tracked here.

The PTQ row is the package calibrated on the 256-real-receipt profile with
input digest `b97e9436a376f1c3…`. A fresh import repeated twice produced
byte-identical graph, weights, and manifest. The previous hashes in this table
pointed at an older profile containing 69 real and 187 synthetic receipts,
which contradicted the table's stated provenance. For the profile above, all
256 normalized sample identities resolve to the private calibration corpus,
and their intersection with the 2,000 held-out sample identities is empty.

Accuracy for all three is recorded in
[the example README](../examples/receipt_digit_reader/README.md). The
latency comparison below is between routes that agree on the answer.

## Held-out accuracy

Re-measured over the full 2,000-image corrected held-out split, next to ONNX
Runtime on the identical preprocessed batch. Labels come from the producer's
own `question_router.py`, so digit extraction matches the training loader
rather than a re-implementation. The route is native CPU.

| Route | `target_exact` | `answer_exact` | phone | street |
| --- | ---: | ---: | ---: | ---: |
| ONNX Runtime FP32 | 0.9690 | 0.9875 | 0.9940 | 0.9750 |
| VolvoxAI FP32 | **0.9690** | 0.9875 | 0.9940 | 0.9750 |
| ONNX Runtime INT8 | 0.9585 | 0.9845 | 0.9900 | 0.9670 |
| VolvoxAI INT8 | **0.9670** | 0.9870 | 0.9930 | 0.9740 |
| VolvoxAI PTQ | **0.9690** | **0.9880** | 0.9930 | 0.9760 |

Two results carry: **FP32 is an exact match to ONNX Runtime** — `record_exact`
1.0 across all 2,000 records with a maximum logit difference of `8.2e-05` — and
**VolvoxAI's INT8 beats ONNX Runtime on ONNX Runtime's own QDQ graph** by 0.85
points of `target_exact`. PTQ ties FP32 on `target_exact` and takes the best
`answer_exact` of any route, on 4.10 MB of weights against 9.91 MB.

The PTQ profile here was calibrated on 256 real receipts from a private corpus
that shares no image content with the held-out split (verified by SHA-256 over
every file in the private audit ledger). That is a different 256 than the profile
behind the example README's coverage table, which is why PTQ reads 0.9690 here
against 0.9685 there — one receipt in 2,000.

### The GPU routes score the same split

Re-scored on the first 300 of the same images with `--backend`, comparing the
decoded record image by image rather than the aggregate:

| Package | CPU vs Vulkan | CPU vs OpenGL |
| --- | ---: | ---: |
| FP32 | 300/300 | 300/300 |
| INT8 | 300/300 | 300/300 |
| PTQ | 299/300 | 299/300 |

FP32 and INT8 are record-identical on every route. PTQ disagrees on exactly one
image, and both GPU backends disagree with CPU *identically* — the same image
and a one-digit boundary decision, which is a boundary case in the
quantized kernel path rather than a device fault. It is the same divergence the
whole-logit comparison shows as `0.09`; on this subset the GPU record happens to
be the correct one. Aggregate scores alone would have hidden it: FP32's three
routes tie because they are identical, PTQ's differ.

## Timing contract

The current harness measures shape binding, dispatch, synchronization, and the
owned output snapshot. **Excluded:** model load, graph compilation, context or
session creation, image decoding, preprocessing, slot decoding, and process
startup. Each route holds one session open across its samples, so these are
inference latency rather than process wall time — an earlier draft of this work
quoted process-wall numbers, which inflated every route by roughly 90 ms of
startup and compared nothing useful.

The retained JS/WASM rows below predate the current `readForBenchmark` timing
seam. Their historical worker timed all of `ReceiptDigitSession.read()`, so
they additionally include the F32 input copy, slot decode, and result close;
native and ONNX Runtime stopped after the owned output snapshot. The native-vs-
ORT comparisons remain like-for-like. Historical WASM values are directional
only and their `vs ORT` ratios must not be presented as exact cross-route
slowdowns until the current harness remeasures them.

The reported statistic is the **median**. A cold page fault or a scheduler
migration moves a mean far more than a median, and neither is a property of the
kernel under comparison. Min and max are reported so an unstable route is
visible rather than averaged away.

Every route must reproduce the same record. A route whose record changes
between runs, or disagrees with the others, is reported as a failure and not as
a fast result. `records_agree` was true for every table below.

Reproduce with:

```bash
python3 -m examples.receipt_digit_reader.tools.benchmark_backends \
  --package build/receipt-digit-reader-ptq \
  --image receipt.jpg --onnx "$SOURCE/model_int8.onnx" \
  --native-binary build/cmake/native/receipt_digit_reader \
  --api dist/0.4.0/volvoxai.js \
  --repeat 30 --warmup 5 --pin-cpu 0
```

The benchmark report records per-route success, decoded-record agreement, and
the exact package `manifest.json`, `graph.json`, and `model.safetensors` hashes
as well as native/API/WASM/ONNX artifact hashes, but omits machine paths, the
receipt path, decoded digits, and captured route stderr. Detailed failures
remain local on stderr and must be reviewed before publishing a run.

## Host

| | |
| --- | --- |
| CPU | AMD Ryzen 5 5600U (6 cores / 12 threads) |
| Threads | 1 requested; parent ORT and every child route pinned to logical CPU 0 |
| Native toolchain | clang 17.0.2, `-O3`, matching the release build's compiler |
| Node.js | 20.11.1 |
| ONNX Runtime | 1.23.2, `CPUExecutionProvider`, `intra_op_num_threads=1` |
| Repeat / warmup | 30 measured / 5 discarded |
| Measurement | 2026-08-21 KST |

Nested math libraries (`OMP_NUM_THREADS` and friends) are forced to 1 for every
child, because otherwise a one-core comparison is silently multi-core for some
routes only.

`--pin-cpu 0` temporarily narrows the benchmark process affinity while each
route is measured, including in-process ONNX Runtime, and restores the original
affinity afterward. Child routes inherit that affinity and are also launched
through `taskset` when it is available.

The host must also be idle. Pinning with `taskset` does not isolate a route
from an unrelated job spread across every core — it contends for the pinned one
too. `benchmark_backends.py` takes three short live `/proc/stat` samples and
refuses to measure above `--max-load` busy cores (default 1.0);
`--allow-busy-host` records anyway and marks the report `host_contended`.

Building the native binary with gcc instead of clang changes these numbers; the
release build uses clang, so a gcc-built measurement is not evidence about the
shipped artifact.

## Results

These are retained quiet-host measurements taken with foreign work below 0.71
cores. An
earlier set, recorded before the harness gained its quiescence gate, was
discarded: a control run under an unrelated job saturating every core moved the
ONNX Runtime reference by 1.8×, more than any difference reported here.

Median milliseconds per receipt, one core. For native rows, **`vs ORT` is how
many times longer the route takes than ONNX Runtime** — `1.12× slower` means it
spends 1.12× the time, so lower is better and `1.00×` is parity. WASM ratios
retain the legacy mixed-boundary caveat above.

### FP32

| Route | Median | Min | Max | vs ORT |
| --- | ---: | ---: | ---: | ---: |
| ONNX Runtime | **33.33** | 32.27 | 41.15 | 1.00× |
| VolvoxAI native CPU | 36.63 | 35.99 | 43.67 | **1.10× slower** |
| VolvoxAI WASM | 168.74 | 163.88 | 178.24 | 5.06× slower |

### INT8 (imported ORT QDQ)

| Route | Median | Min | Max | vs ORT |
| --- | ---: | ---: | ---: | ---: |
| ONNX Runtime | **22.73** | 22.09 | 27.39 | 1.00× |
| VolvoxAI native CPU | 31.26 | 30.38 | 35.86 | **1.38× slower** |
| VolvoxAI WASM | 74.79 | 73.05 | 80.24 | 3.29× slower |

### PTQ (Conv only)

Compared against ONNX Runtime on the producer's INT8 graph, the closest
byte-domain reference.

| Route | Median | Min | Max | vs ORT |
| --- | ---: | ---: | ---: | ---: |
| ONNX Runtime (INT8) | **22.82** | 22.16 | 32.26 | 1.00× |
| VolvoxAI native CPU | 27.64 | 27.09 | 31.24 | **1.21× slower** |
| VolvoxAI WASM | 58.09 | 57.09 | 64.28 | 2.55× slower |

## Runtime modes and physical batching

The measurements in this section predate the execution-mode rename and
generated-API cutover. Immutable report filenames, hashes, and table labels
retain their original vocabulary:
`SIMPLE` maps to DIRECT, `ADAPTIVE` maps to SCHEDULED with zero batch delay,
and `SERVICE` maps to SCHEDULED with a positive bounded delay. Current public
applications use only the proto-defined DIRECT/SCHEDULED modes; no old
plan-name aliases exist. The removed pre-proto receipt Runtime-mode harness
wrote the privacy-redacted `volvoxai.receipt-digit-runtime-modes/v2` schema
with a `mode` field instead of overwriting any hash-bound artifact listed
below. Those retained public reports contain anonymized lane IDs and
parity/stability summaries, not input paths, decoded receipt values, or output
fingerprints; the explicit `--include-private-logits` diagnostic was local
only and must never be published. They bind the package without exposing its
location by recording exact SHA-256 digests for `manifest.json`, `graph.json`,
and `model.safetensors`.

The application ABI and graph batch domain are deliberately separate. One
`ReceiptDigitSession.read()` remains `[1, 1, 320, 672]`; the manifest records
`abi.batch.per_request: 1`. A default import (`--max-batch-size 1`) also authors
a static B1 graph. An opt-in dynamic import preserves the producer's leading
`batch` symbol with an exact `1..N` domain, allowing Runtime to coalesce N
independent B1 calls into one physical B=N execution. Supplying one explicit
bulk B=N tensor through generated `Run` or `Execute` is a different
contract and is not what the session benchmark below measures.

Those retained measurements used the older helper implementation. Current
applications call generated `Run` for DIRECT work or `Submit` for a
SCHEDULED Runtime, and the current receipt helper is a thin wrapper over that
public lifecycle rather than a separate runtime surface.

### Retained dynamic packages

Both packages were imported from the checksummed release named at the top of
this document with `--max-batch-size 4`:

| Variant | Nodes | graph.json SHA-256 | model.safetensors SHA-256 | manifest.json SHA-256 |
| --- | ---: | --- | --- | --- |
| INT8 | 126 | `fdd3ea5c9891e7a10e3bda359b2926cd58074bd11e15952c628bd68bc641caaa` | `6c647c1bfdb16e47a5db6c115ff7bb16e35f149f652e673d2dc482b7e13b172c` | `f762421c0474f42bce7730ea88703450d09b5a89893f78d3037dbf7df848050a` |
| FP32 | 111 | `36cef0c05636227c57f458d93d1a7aca87add9ca41d7bda78685f4aa3f354649` | `8e1a78e1ab1977a179d38ed85f783ba549eb608814bb851312e65ec982df1bb1` | `f8553fd3edd7434529c069c622cfee0ca1195d29916923378922dadd76717c94` |

The producer release and these generated package directories are not tracked
in this repository. Their hashes identify entries in the private audit ledger;
a clean checkout cannot independently reconstruct these rows without the
producer source.

The corresponding source ONNX hashes are
`0f30b70bf507b51ebe933275d920fd8f006f6187fad280f09e1a2ca9753ef73f`
for INT8 and
`b84c53d5282dfd0d572bb8ef48216c29928a42dd859e0f4d7bea466126d3d8e8`
for FP32. Both graphs expose input `['batch', 1, 320, 672]` and domain
`batch={min:1,max:4,multiple_of:1}` while their manifests retain the B1
per-request shape.

The INT8 operator multiset is `Add×7`, `BatchMatMul×6`, `Conv2D×1`,
`DequantizeLinear×15`, `Div×3`, `Expand×5`, `GroupNorm×11`, `LayerNorm×1`,
`Linear×7`, `Mul×14`, `QConv2D×10`, `QuantizeLinear×12`, `Reshape×2`,
`Sigmoid×14`, `Softmax×3`, and `Transpose×15`. The FP32 graph has `Add×8`,
`BatchMatMul×6`, `Conv2D×11`, `Div×3`, `Expand×6`, `GroupNorm×11`,
`LayerNorm×1`, `Linear×7`, `Mul×14`, `Reshape×2`, `Sigmoid×14`, `Softmax×3`,
and `Transpose×25`. No runtime-op fallback or manual graph override was used.

The import command form was:

```bash
SOURCE=/path/to/huggingface/tiny-receipt-reader-digit-slots-2m
RUN=/path/to/generated-digit-audit

taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m examples.receipt_digit_reader.tools.import_hf_onnx \
  --source "$SOURCE" --out-dir "$RUN/int8-b4-v3" \
  --variant int8 --max-batch-size 4

taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m examples.receipt_digit_reader.tools.import_hf_onnx \
  --source "$SOURCE" --out-dir "$RUN/fp32-b4-v2" \
  --variant fp32 --max-batch-size 4
```

The FP32 import initially stopped at a producer output whose slot dimension
was named `Addslot_logits_dim_1` while ONNX inference proved concrete extent
16. It now succeeds only because the producer manifest supplies the exact
singleton domain `16:16`; distinct or non-singleton symbols remain
non-interchangeable. Shape programs are evaluated only in the immutable
integer shape domain, and data-dependent/noncanonical consumers still reject.

### Inputs and correctness contract

Four distinct normalized B1 inputs were used: two producer-release fixtures
and two held-out private audit fixtures. Public lane IDs intentionally replace
the original filenames, per-image hashes, and decoded identifiers; those values
can fingerprint a real receipt without making this repository reproducible.
The private input ledger retains the exact values and was rechecked during this
audit, including correction of a lane-2 digest transcription error.

| Lane | Fixture class | Public identity | Result contract |
| ---: | --- | --- | --- |
| 0 | producer release | `release-a` | independent B1 reference established |
| 1 | producer release | `release-b` | independent B1 reference established |
| 2 | held-out audit | `audit-a` | independent B1 reference established |
| 3 | held-out audit | `audit-b` | independent B1 reference established |

For INT8, every historical CPU-JS and WASM B2/B4 lane was byte-identical to
that same backend's independent DIRECT B1 result. The private report retains
the complete output digests; the public result is the non-reversible summary
that all four lanes were stable across repeats and scheduled-vs-independent
bytes were exact.

Each zero-delay SCHEDULED B2/B4 group produced exactly one Runtime execution diagnostic,
one distinct `dispatchId`, `batchSize=N`, and additive
`trueBackendInvocations=1`, with shape signature
`v1|6:input0|4:N,1,320,672`. A separate positive-delay SCHEDULED WASM B4 smoke
did the same and returned byte-identical results for all four lanes. Thus both
scheduled delay configurations have measured physical B>1 evidence;
concurrency alone is not being relabelled as a batch.

FP32 was qualification-only, not a timing row. One zero-delay SCHEDULED WASM B4 group
produced one physical invocation/dispatch with `batchSize=4` and decoded all
four records correctly. Each lane was byte-identical to its independent WASM
B1 result; exact digests remain in the private report rather than exposing
stable fingerprints of the receipt inputs.

The producer ORT graph also returned exactly identical logits for physical
B2/B4 versus N independent ORT B1 calls. VolvoxAI's full 176-logit lanes were
then compared against those independent ORT B1 references:

| Lane | CPU max abs | CPU max relative¹ | WASM max abs | WASM max relative¹ | Decode |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.5752413 | 3.2592 | 0.5752367 | 3.2591 | matches reference |
| 1 | 0.2711363 | 1.4542 | 0.2711371 | 1.4542 | matches reference |
| 2 | 0.7708579 | 6.5098 | 0.7708536 | 6.5098 | matches reference |
| 3 | 1.1343807 | 1344.2555 | 0.9867835 | 1409.5254 | matches reference |

¹ Maximum elementwise `abs(error) / max(abs(reference), 1e-6)`. It is dominated
by near-zero reference logits, especially lane 3; the more stable maximum
absolute error divided by each lane's maximum absolute reference logit ranges
from 1.420% to 5.917% on CPU and 1.420% to 5.147% on WASM. B2 uses lanes 0–1
and has the same values. All decoded records agree. This comparison establishes
reference accuracy; the private byte-level same-backend B1/B2/B4 comparison is
the separate proof that batching itself preserves VolvoxAI results.

### Clean directional timing

The timing host was the Ryzen 5 5600U described above, pinned to CPU 0 with
governor `powersave`. Python was 3.10.12, ONNX Runtime 1.23.2, ONNX 1.22.0,
NumPy 2.2.6, and Node 20.11.1. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`MKL_NUM_THREADS` were all 1. The quiet gate was `max-load=1.0`; immediately
before the run, 1-minute load average was 0.81 and measured foreign work was
0.475991649 cores. No other model export or benchmark process was live.

The then-current TypeScript entry was bundled for the Node run with:

```bash
RUN=/path/to/generated-or-restored-digit-audit
npx esbuild ts/index.ts --bundle --platform=node --format=esm \
  --loader:.wgsl=text \
  --outfile="$RUN/volvoxai-measured.mjs"
```

That bundle's SHA-256 was
`04195ea9dd51e674d8ea35f08ead36c886b584fc3ae933bbf1e8e15835ad0663`;
the WASM artifact `dist/0.4.0/volvoxai.wasm` was
`ffaa402d483fe37cc709322411dfb3381b57bd6345df06d538a693d65237922d`.

The retired CPU-JS provider used 1 warmup and 3 measured groups; WASM used 5 warmups and 30
measured groups. DIRECT B1 ran lane 0 without a scheduler. SCHEDULED B2 used
lanes 0–1 and B4 used all four lanes. Rows were executed B1, then B2, then B4;
the order was not counterbalanced. These numbers are therefore a directional
qualification of batching behavior, not a fine-grained performance study. In
particular the ±1% CPU differences must not be over-interpreted.

The next table retains retired report labels: SIMPLE = DIRECT; ADAPTIVE =
SCHEDULED with zero batch delay. The current CLI defaults to the later 1 ms
collection-delay behavior, so reproducing these immutable rows requires the
explicit `--max-batch-delay-ms 0` shown below.

| Backend / retired plan label | B | Groups | Median group ms | Min | Max | Logical req/s | Logical / physical | Snapshot copies² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU-JS SIMPLE (retired) | 1 | 3 | 6607.922 | 6548.996 | 6677.027 | 0.15126 | 3 / 3 | 0 |
| CPU-JS ADAPTIVE (retired) | 2 | 3 | 13214.781 | 13123.207 | 13236.029 | 0.15161 | 6 / 3 | 8 |
| CPU-JS ADAPTIVE (retired) | 4 | 3 | 26732.047 | 26722.622 | 26773.883 | 0.14957 | 12 / 3 | 16 |
| WASM SIMPLE | 1 | 30 | 77.999 | 74.959 | 102.692 | 12.49431 | 30 / 30 | 0 |
| WASM ADAPTIVE | 2 | 30 | 173.454 | 166.884 | 183.654 | 11.46153 | 60 / 30 | 70 |
| WASM ADAPTIVE | 4 | 30 | 348.877 | 333.782 | 372.678 | 11.45397 | 120 / 30 | 140 |

² Runtime inspection is cumulative and includes warmups: `(warmup + measured)
× B` snapshots for scheduled plans. It counts staging copies but does not time
them separately. Relative to `B × B1` median, aggregate group latency changed
by −1.064 ms at retired CPU-JS B2, +300.357 ms at retired CPU-JS B4,
+17.456 ms at WASM B2, and
+36.881 ms at WASM B4. Those aggregate deltas include different lane data,
scheduler admission, snapshots, staging, and backend work; they are not an
isolated copy-cost estimate.

There is **no material throughput win** in this run. Retired CPU-JS B2 is directionally
0.23% above B1 and retired CPU-JS B4 is 1.12% below it, while WASM B2/B4 are
8.27%/8.33% below. True physical batching is working and reduces N backend
invocations to one, but this model/backend combination does not materially
amortize the scheduler and batched-kernel cost.

In this historical campaign, until the route-specific measured `T(B)` selector described in the
[scheduling design](scheduling-and-dynamic-batching-design.md#legal-b-operating-b-and-padding)
is implemented,
the measured default was `scheduler.maxBatchSize: 1`. Re-run the current WASM
route before treating that historical setting as deployment guidance. Use DIRECT
for a one-shot call when global admission/fairness is not
needed; use SCHEDULED with the same batch cap when concurrent streams still
need the Runtime-wide resource coordinator. B2/B4 remains an explicit
qualification/benchmark setting, not the recommended production default for
this measured device.

ONNX Runtime was run in the same fixed order; for each B the harness measured
independent B1 first and physical BN second. It too is directional rather than
counterbalanced:

| B | Independent-B1 median ms / req/s | Physical-BN median ms / req/s | Max abs vs independent |
| ---: | ---: | ---: | ---: |
| 1 | 23.376 / 41.127 | 23.129 / 41.735 | 0.0 |
| 2 | 47.563 / 41.488 | 52.971 / 36.656 | 0.0 |
| 4 | 96.452 / 41.140 | 113.686 / 35.194 | 0.0 |

Physical ORT B2/B4 was also slower than independent B1, so the negative
throughput result is not evidence of a VolvoxAI correctness failure.

### Timing command form

The reports and raw lanes for these historical CPU-JS/WASM rows are private,
untracked audit artifacts and cannot be reproduced from a clean checkout. The
retired harness command is omitted. A future campaign should use only current
providers and publish a self-contained invocation with anonymized inputs.

New reports record both scheduler values as
`scheduler={maxBatchSize,maxBatchDelayMs}`; do not compare a default 1 ms run
to the zero-delay rows above without labelling the contract change.

ORT used:

```bash
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m examples.receipt_digit_reader.tools.benchmark_onnx_batches \
  --onnx "$SOURCE/model_int8.onnx" --manifest "$RUN/int8-b4-v3/manifest.json" \
  --raw "$RUN/lanes/lane0.f32" --raw "$RUN/lanes/lane1.f32" \
  --raw "$RUN/lanes/lane2.f32" --raw "$RUN/lanes/lane3.f32" \
  --batch-size 1 --batch-size 2 --batch-size 4 \
  --threads 1 --repeat 30 --warmup 5 --max-load 1.0 \
  --out "$RUN/ort-batches-clean.json"
```

### RTX 3090: Deno WebGPU large-B sweep

This is a separate hardware campaign from the one-core retired-CPU-JS/WASM rows above.
It ran Deno 2.9.3 with its Vulkan WebGPU backend on an NVIDIA GeForce RTX 3090
(WebGPU vendor `4318`, device `8708`; driver 535.309.01; 24,576 MiB). The exact
JavaScript artifact was
`dist/0.4.0/volvoxai.js` with SHA-256
`3337273dd7b9e87e5e865f457f6b602f4769e84225ee21f28480f852e33f49b1`.
Here B is the model's request-batch dimension, not a CUDA warp-lane count, so
the useful ceiling has to be measured for this graph and execution stack.

The campaign used fresh packages authored through B=19:

| Variant | Nodes | manifest.json SHA-256 | graph.json SHA-256 | model.safetensors SHA-256 |
| --- | ---: | --- | --- | --- |
| FP32 | 111 | `afedc614692484c6e5b3cfbf21aad1888843ad596e7f8fd642ffb9175c09c54e` | `5402593fee5103b847bf67ccc109ecdda8daef54a6d454e2372dbc91f89b6d8d` | `8e1a78e1ab1977a179d38ed85f783ba549eb608814bb851312e65ec982df1bb1` |
| INT8 | 126 | `27d17c4fc35ed509e94330abd8cfb54a398f03811ceae301a84e3165519372c7` | `59df328de20aba3c5f1d5a26c4c0f175642256c0feba6f1ca6ad167df1e8678f` | `6c647c1bfdb16e47a5db6c115ff7bb16e35f149f652e673d2dc482b7e13b172c` |

The WebGPU physical-domain proof admits their complete B=1..19 domain on the
reported device limits. B=20 is rejected before execution: the F32
intermediate `[20, 160, 336, 32]` requires 137,625,600 bytes, greater than the
adapter's 134,217,728-byte (`128 MiB`) maximum storage-buffer binding. At B=19
the same tensor is 130,744,320 bytes. Thus B=19 is the static legal package
ceiling on this adapter, not a claim that it is allocatable or optimal in a
live process.

DIRECT B1 used no Runtime scheduler. Every SCHEDULED row used two warmup and ten
measured groups, and its timing boundary includes all required output readbacks
and result close. Logical req/s is the aggregate measured throughput; it is not
recomputed from the median row alone.

The next table retains retired report labels: SIMPLE = DIRECT; ADAPTIVE =
SCHEDULED with zero batch delay. These reports predate the current CLI; their
delay contract is historical and a new reproduction must pass
`--max-batch-delay-ms 0` explicitly.

| Variant / retired plan label | B | Median group ms | Logical req/s | B15 vs SIMPLE B1 throughput | Physical proof | Same-backend lane parity |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| FP32 SIMPLE | 1 | 86.787 | 11.382 | — | 10 logical / 10 physical | reference |
| FP32 ADAPTIVE | 4 | 119.930 | 33.550 | — | 40 logical / 10 physical | byte-exact |
| FP32 ADAPTIVE | 8 | 154.616 | 51.774 | — | 80 logical / 10 physical | byte-exact |
| FP32 ADAPTIVE | 12 | 183.743 | 65.456 | — | 120 logical / 10 physical | byte-exact |
| FP32 ADAPTIVE | 15 | 201.578 | **72.701** | **6.387x** | 150 logical / 10 physical | byte-exact |
| INT8 SIMPLE | 1 | 87.044 | 11.400 | — | 10 logical / 10 physical | reference |
| INT8 ADAPTIVE | 4 | 125.747 | 32.351 | — | 40 logical / 10 physical | byte-exact |
| INT8 ADAPTIVE | 8 | 157.422 | 49.578 | — | 80 logical / 10 physical | byte-exact |
| INT8 ADAPTIVE | 12 | 196.376 | 59.699 | — | 120 logical / 10 physical | byte-exact |
| INT8 ADAPTIVE | 15 | 231.141 | **64.655** | **5.672x** | 150 logical / 10 physical | byte-exact |

Both DIRECT reports record `schedulerAllocated=false`. Each scheduled group
records one `dispatchId`, `batchSize=N`, and one additive
`trueBackendInvocations`, so N logical calls really became one physical
provider invocation. The harness first executed every lane as an independent
same-WebGPU DIRECT B1 reference in the same process, then compared every F32
logit. All B4/B8/B12/B15 lanes were byte-exact (`max_abs=0`, `max_rel=0`), not
merely decode-equivalent.

The frozen typed proof is `typed-independent-batch-proof/v1`. FP32 covers all
111 nodes with graph-fingerprint SHA-256
`8ead1e66c03a2eadef1d4eb52b1e6e120a654ef6bb2f182aaee9680fd7b702bc`;
INT8 covers all 126 with
`a577b437eb26d3300774d1ffe7b4fd247b11235d37f858be517c0482bcaa4f98`.
The individual report SHA-256 ledger uses filenames relative to the campaign
directory recorded below:

```text
digit-fp32-simple-b1.json    40ebc09a1d76f4f8b750fa393d1a5afe41ea7f01df2954c2e5f8b2bc99efb084
digit-fp32-adaptive-b4.json  b43c64d405993271d25482151f141acc3a92f34df15de9076067a8fdd4c24127
digit-fp32-adaptive-b8.json  e614afb774dd5622d2623c68ccf94a316a9fc8bac3fc3389531b16c628b750f7
digit-fp32-adaptive-b12.json a70a77995eaef8e9332899348717b58266667a5403ff49cf4a91f78867b9d34a
digit-fp32-adaptive-b15.json 66b7b97d5a6b8a37d18580347e7248a6719bba6e0a4d1fce622bbaacb392fb7e
digit-int8-simple-b1.json    a7d76d37955b14ea2afaac3772d205c89f704bc7c6434732f34b3f964c1a90b7
digit-int8-adaptive-b4.json  c029d0277cb87d3a6344da1d0b1b1e815529ec4ccbf606ea3202e2c729e31ed6
digit-int8-adaptive-b8.json  a15b25c9b8e2c36cb0188ca5a438318e4bfc2fa3f1c16b4a70977754b1d57d3b
digit-int8-adaptive-b12.json f1b32c4b7e724d40b1e2c1966f3c1ad4c4d683710a2aa5a28eaa1527b8447179
digit-int8-adaptive-b15.json fd629f72bba878856bb332865ef2ae8be93ffb87d210220d14d19e4bee70b687
```

B=16 was then exercised as a fail-closed boundary. Both variants returned a
`VolvoxAIError` with `code=OUT_OF_MEMORY`, `phase=execution`, and
`backend=webgpu` while constructing the concrete tensor generation. No invalid
buffer, zero-filled logits, decoded record, or other corrupted output was
published. The FP32 failure report hashes to
`40ac54d20d8a939e53f04f0501e607753bb69ad56057e32218c0c190e36d51f9`;
the INT8 report hashes to
`6b11120a76d0ce0c88cf6ee5bcce14dda895bf6f88eb4792976acc74713b40dc`.

Process telemetry rules out exhaustion of the RTX 3090's 24 GiB as the
explanation by itself:

| FP32 probe | Idle before | Active peak | Idle after | Peak utilization | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| B15 | 4 MiB | 1,506 MiB | 4 MiB | 88% | success |
| B16 | 4 MiB | 226 MiB | 4 MiB | 89% | normalized `OUT_OF_MEMORY` |

The B16 run fails early, which explains its lower observed peak; neither run
approaches 24 GiB. A follow-up probe then isolated the exact failing graph
tensor, `v1` from `node_1` (`Conv2D`, F32, `[B, 160, 336, 32]`), on a fresh
device created with the same default `requestDevice()` call as the engine:

| Isolated buffer | Exact bytes | Result |
| --- | ---: | --- |
| Tensor `v1`, B15 | 103,219,200 (98.4375 MiB) | allocate, clear, submit, and queue completion succeeded |
| Threshold spot check | 109,051,904 (104 MiB) | allocate and use succeeded |
| Tensor `v1`, B16 | 110,100,480 (105 MiB) | scoped `GPUOutOfMemoryError`, 3/3 fresh processes |

The effective single-buffer boundary is therefore greater than 109,051,904
bytes and no greater than 110,100,480 bytes on this exact stack, despite the
advertised 134,217,728-byte storage-binding and 268,435,456-byte buffer limits.
The failure reproduces without old/new graph-generation coexistence. This
classifies it as an advertised-versus-effective allocator-policy mismatch in
Deno 2.9.3/wgpu/Vulkan/NVIDIA driver 535, not aggregate 24-GiB VRAM exhaustion.
B15 remains the observed graph operating boundary for this stack, not a
universal RTX 3090 maximum.

The isolated evidence is under the campaign's
`isolated-v1-allocation/` directory. Its
`allocation-boundary-summary.json` SHA-256 is
`8e8df356592252fa341fd6bfd326d26ae5b84c0811f4f745e0293827a4b93d96`;
the exact B15 and captured B16 reports hash to
`0343a57587b4a4799261e8f41d1ec884dfafc6936868f4b7cf6c85f6ec6a6a7c`
and `8d319b66bbfec68467f3746de3e35d8c2352605563508282635cbad8ffb2b56d`.
The executable Deno-eval commands are retained as `b15/command.txt` and
`b16-captured/command.txt` alongside those reports.

The reviewed campaign is at
`/path/to/volvoxai-gpu-validation/results/webgpu-3337273dd7b9/`.
Its `campaign-summary.json` SHA-256 is
`a58f07d39ab15c431a4c37cc4fb17acbb1ab6add8478ada5432b98ef7e7caaa0`;
the historical campaign copy of the receipt runtime harness (not the current
CLI bytes) has SHA-256
`57336c345ca2c10a2406d444b531584e678a259827e36cd221134d69277a3d40`.
The campaign reports and lane artifacts are not tracked in this repository;
the path is a locator for the private archive, not a self-contained public
reproduction source. The exact command path used the now-removed pre-proto
receipt Runtime-mode harness, so current repository tooling does not recreate
these archived WebGPU rows. The campaign wrapper also captured the expected
nonzero B16 process and persisted its normalized failure and GPU telemetry.

### RTX 3090 native Runtime dynamic-batch sweep

The native internal Runtime benchmark path was measured separately on Vulkan,
OpenGL, and CUDA at B16, B32, B64, B128, and B256. This is the coalescing path:
DIRECT executes N independent B1 calls, while SCHEDULED submits N B1 requests
and returns them from one physical B=N execution. It is not the Deno WebGPU
route above or the historical explicit-bulk CUDA experiment below.

Each precision uses one exact maxB256 package across every backend and batch
size. The report's comparison policy is
`same-package-dynamic-batch-sweep-only`, so within one precision/backend the
B rows are a real dynamic-batch sweep rather than comparisons between packages
with different compiled domains.

The canonical public report is
[`native_dynamic_batch_rtx3090.json`](../examples/receipt_digit_reader/reports/native_dynamic_batch_rtx3090.json)
(SHA-256
`3cf3bae88e1d7144fc31c66f7e495a05f436f3b6dbd425249db4bfe084bf2968`),
measured at `2026-08-22T11:49:35Z`. It binds source base commit
`064e30d2923351401e3fb654863905e2dd77a533` and the exact build-relevant
source snapshot SHA-256
`d0a3f2a8da0f8664659a26754ce33d1352a5fd580fd0750eaf1bf1a42975f4a9`.
Because the report records `sourceDirtyWithinSnapshotScope=true`, it attests
that snapshot, not the base commit by itself. The Release/O3/NDEBUG harness
SHA-256 is
`0be73f6af7f2f1d230eebb82257fdf271642ae53b6c28110d642a70f2d08d02c`;
the complete CMake, compiler, strict-no-FMA `sm_86` CUDA, shader-compiler,
locked Naga 30.0.0, host, and RTX 3090 driver/device evidence is retained under
`build`, `host`, and `device` in the report. The build contract SHA-256 is
`38f998e0337cefdf4f4cbdb1405c256acbddb3c9f80c1849469c949923f475a8`.

The exact shader toolchain identities are:

| Toolchain artifact | SHA-256 |
| --- | --- |
| Naga 30.0.0 binary | `84d8eaf638bb5f16883fd0270935a06ae90b6ea6e67f4174fe129a7620c905b4` |
| Native shader compiler 0.2.0 binary | `e0c8d63b32039fdfe68f9209cb8fbcf2e5cc45853001c3799f9699dd4898f2bd` |
| Native shader compiler Rust source | `faf0b0d7ea3cb430a9156926b45c5876fe78c1cb14dd24d1122b03e5c1c39282` |
| Native shader compiler `Cargo.lock` | `f3941bf10fcf1435a034b53c7c581b6310600ecb39df2e4c2f2e3cb4733f0353` |

The host was x86_64 Linux `6.8.0-136-generic` on
`11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz`; the measured device was
ordinal 0, an RTX 3090 with driver `535.309.01` and 24,576 MiB reported memory.

The wrapper re-attested the exact package bytes, declared max B, and typed
batch proof:

| Variant | Package max B | Manifest SHA-256 | Graph SHA-256 | Weights SHA-256 | Typed proof identity |
| --- | ---: | --- | --- | --- | --- |
| FP32 | 256 | `396ffd86141faa4901ed3a79997d82b04cf440b851b14081b741ac464644f50c` | `200cf0f0997ab9e24b10d1443d70bf44a5ce38bc873211f3f77e142bc0ea551c` | `8e1a78e1ab1977a179d38ed85f783ba549eb608814bb851312e65ec982df1bb1` | `typed-independent-batch-proof/v1:fnv1a64:315abb811c18b77e` |
| INT8 | 256 | `7fb71808b5ea8159d8d6247ce0b3d6f16e817729f991204cae799ce944b38b97` | `a02edeb88a89bc122a62a43619b53fd640c28d8a9d9aba8c60c6d619c4d9d002` | `6c647c1bfdb16e47a5db6c115ff7bb16e35f149f652e673d2dc482b7e13b172c` | `typed-independent-batch-proof/v1:fnv1a64:e5877dbc2a9f6b00` |

Every B cycles the same four distinct private fixture classes, with each class
used 4, 8, 16, 32, or 64 times at B16, B32, B64, B128, or B256 respectively.
The public report omits private paths, raw backend logs, input/output digests,
and decoded receipt values; the private matrix input and lane ledger are not
publication artifacts.

DIRECT timing starts immediately before the first `vx_runtime_run` and ends
after the Nth owned result is returned. SCHEDULED timing starts immediately
before the first `vx_runtime_submit`, submits all N requests before waiting,
and ends after the Nth `vx_request_result` returns its owned result. Both
boundaries include binding validation, backend input upload, provider dispatch,
device synchronization, and result snapshot publication. SCHEDULED additionally
includes owned submission snapshots, aggregate input stacking, and the
aggregate-to-B1 output split. They exclude `vx_result_read`, parity/decode,
request/result release, and JSON writing. Routes alternate per iteration,
starting with DIRECT on even iterations. This differs from the WebGPU table's
output-readback/result-close boundary, so no cross-table latency ratio or
backend ranking is valid.

Every case used three warmup groups and five measured groups, one CPU thread
pinned to CPU 0, a 5000 ms collection guard, Vulkan arena `6144` MiB, and CUDA
ordinal 0. The values below are whole-group milliseconds shown as median [min,
nearest-rank p95, max], rounded decimal half-up to three places. Speedup is
DIRECT median divided by SCHEDULED median; scheduled req/s is the aggregate
over all measured samples, not a reciprocal of the median.

The 6144-MiB Vulkan arena is this campaign's reproducibility setting, not an
engine or RTX 3090 capacity ceiling. Native Vulkan has no fixed arena-size or
batch-count clamp: acceptance follows overflow-checked sizing, queried device
limits, the selected kernel launch geometry, and actual Vulkan allocation.
OpenGL and CUDA likewise have no fixed total-memory or batch-count ceiling;
their kernel/index ABIs, queried allocation and launch limits, and actual
device allocation govern acceptance. The current out16 Vulkan/OpenGL proof
uses its selected tactic's real launch geometry, while CUDA uses its own launch
proof rather than a cross-backend grid bound. All three native backends passed
the official B256 executions below. The WebGPU 128-MiB binding limit discussed
above does not apply to these native paths.

| Variant | Backend | B | DIRECT independent B1 group ms | SCHEDULED B=N group ms | Speedup | SCHEDULED req/s | Same-backend max abs / rel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | Vulkan | 16 | 1042.749 [1038.323, 1044.225, 1044.225] | 111.860 [111.463, 111.923, 111.923] | 9.322x | 143.118 | `0 / 0` |
| FP32 | Vulkan | 32 | 2071.859 [2071.135, 2074.207, 2074.207] | 151.256 [150.502, 152.132, 152.132] | 13.698x | 211.595 | `0 / 0` |
| FP32 | Vulkan | 64 | 4147.997 [4142.546, 4151.831, 4151.831] | 247.570 [247.189, 251.448, 251.448] | 16.755x | 257.023 | `0 / 0` |
| FP32 | Vulkan | 128 | 8301.287 [8295.561, 8320.842, 8320.842] | 432.547 [431.898, 437.996, 437.996] | 19.192x | 294.687 | `0 / 0` |
| FP32 | Vulkan | 256 | 16691.671 [16690.187, 16692.818, 16692.818] | 823.797 [821.823, 827.798, 827.798] | 20.262x | 310.589 | `0 / 0` |
| INT8 | Vulkan | 16 | 1022.937 [1022.190, 1023.793, 1023.793] | 120.353 [120.169, 120.402, 120.402] | 8.499x | 132.996 | `0 / 0` |
| INT8 | Vulkan | 32 | 2046.375 [2041.141, 2051.718, 2051.718] | 169.709 [169.539, 170.805, 170.805] | 12.058x | 188.235 | `0 / 0` |
| INT8 | Vulkan | 64 | 4100.196 [4095.827, 4102.833, 4102.833] | 283.718 [281.946, 286.196, 286.196] | 14.452x | 225.445 | `0 / 0` |
| INT8 | Vulkan | 128 | 8198.038 [8190.519, 8199.384, 8199.384] | 578.594 [575.687, 582.693, 582.693] | 14.169x | 220.919 | `0 / 0` |
| INT8 | Vulkan | 256 | 16378.788 [16368.035, 16392.913, 16392.913] | 1108.550 [1103.739, 1111.708, 1111.708] | 14.775x | 231.085 | `0 / 0` |
| FP32 | OpenGL | 16 | 11503.872 [11501.920, 11509.134, 11509.134] | 766.772 [766.169, 767.213, 767.213] | 15.003x | 20.868 | `0 / 0` |
| FP32 | OpenGL | 32 | 23056.174 [23048.722, 23058.381, 23058.381] | 810.031 [809.513, 811.039, 811.039] | 28.463x | 39.498 | `0 / 0` |
| FP32 | OpenGL | 64 | 46183.079 [46176.272, 46202.547, 46202.547] | 899.424 [898.877, 900.242, 900.242] | 51.347x | 71.144 | `0 / 0` |
| FP32 | OpenGL | 128 | 91974.396 [91969.911, 91994.565, 91994.565] | 1080.949 [1077.560, 1082.492, 1082.492] | 85.087x | 118.438 | `0 / 0` |
| FP32 | OpenGL | 256 | 184195.134 [184149.521, 184233.725, 184233.725] | 1466.197 [1461.655, 1481.666, 1481.666] | 125.628x | 174.142 | `0 / 0` |
| INT8 | OpenGL | 16 | 2250.384 [2247.363, 2254.313, 2254.313] | 197.593 [196.910, 200.053, 200.053] | 11.389x | 80.830 | `0 / 0` |
| INT8 | OpenGL | 32 | 4497.830 [4492.654, 4501.261, 4501.261] | 247.315 [246.418, 248.716, 248.716] | 18.187x | 129.406 | `0 / 0` |
| INT8 | OpenGL | 64 | 8974.863 [8960.027, 8991.343, 8991.343] | 361.973 [361.115, 363.324, 363.324] | 24.794x | 176.819 | `0 / 0` |
| INT8 | OpenGL | 128 | 17968.060 [17959.046, 17977.524, 17977.524] | 647.420 [646.939, 648.934, 648.934] | 27.753x | 197.621 | `0 / 0` |
| INT8 | OpenGL | 256 | 35963.176 [35927.710, 36029.130, 36029.130] | 1171.915 [1169.818, 1172.873, 1172.873] | 30.688x | 218.536 | `0 / 0` |
| FP32 | CUDA | 16 | 102.899 [101.001, 103.228, 103.228] | 74.953 [74.148, 75.150, 75.150] | 1.373x | 214.147 | `0 / 0` |
| FP32 | CUDA | 32 | 204.475 [201.550, 204.926, 204.926] | 150.240 [150.074, 150.868, 150.868] | 1.361x | 212.872 | `0 / 0` |
| FP32 | CUDA | 64 | 407.817 [405.119, 408.191, 408.191] | 315.457 [314.799, 316.709, 316.709] | 1.293x | 202.766 | `0 / 0` |
| FP32 | CUDA | 128 | 811.800 [809.468, 819.339, 819.339] | 645.245 [643.896, 645.525, 645.525] | 1.258x | 198.472 | `0 / 0` |
| FP32 | CUDA | 256 | 1631.656 [1628.827, 1632.585, 1632.585] | 1321.835 [1320.765, 1325.571, 1325.571] | 1.234x | 193.599 | `0 / 0` |
| INT8 | CUDA | 16 | 90.732 [89.297, 91.523, 91.523] | 65.774 [65.341, 66.396, 66.396] | 1.379x | 242.993 | `0 / 0` |
| INT8 | CUDA | 32 | 180.337 [177.380, 180.778, 180.778] | 132.702 [132.311, 132.926, 132.926] | 1.359x | 241.303 | `0 / 0` |
| INT8 | CUDA | 64 | 358.829 [354.986, 359.157, 359.157] | 280.316 [280.005, 280.542, 280.542] | 1.280x | 228.372 | `0 / 0` |
| INT8 | CUDA | 128 | 714.618 [711.182, 715.279, 715.279] | 575.494 [575.251, 576.376, 576.376] | 1.242x | 222.357 | `0 / 0` |
| INT8 | CUDA | 256 | 1431.980 [1429.511, 1433.294, 1433.294] | 1179.566 [1178.617, 1181.210, 1181.210] | 1.214x | 216.976 | `0 / 0` |

All 30 cases passed strict provider-route attestation with zero missing or
fallback nodes. The measured-request and parity counts are:

| B | Logical requests per route | DIRECT physical invocations | SCHEDULED physical invocations | Warmup + measured parity tensors / elements |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 80 | 80 | 5 | 128 / 22,528 |
| 32 | 160 | 160 | 5 | 256 / 45,056 |
| 64 | 320 | 320 | 5 | 512 / 90,112 |
| 128 | 640 | 640 | 5 | 1,024 / 180,224 |
| 256 | 1,280 | 1,280 | 5 | 2,048 / 360,448 |

Every scheduled group has one shared nonzero process-local physical execution
ID, physical batch size N, exactly one true backend invocation, and a new ID in
the next group. All outputs and decoded lane results are exact. Every Vulkan
physical forward additionally attests a device-local 6-GiB compute arena, a
32-MiB host-visible staging buffer, and successful staged upload/download
operations. Every measured CUDA physical forward attests a successful cached
CUDA Graph launch and stream synchronization. All 30 rows speed up, but the
claim remains limited to these exact package bytes, shapes, timing boundary,
and hardware.

The report was produced by the checked-in public-API-only harness and matrix
wrapper. A reproduction uses an exact Naga 30.0.0 executable, the current
locked native shader compiler, and a private matrix input that supplies the
two maxB256 package roles and four receipt lane paths:

```bash
ROOT=/path/to/volvoxai
BUILD=/path/to/release-gpu-build
RUN=/path/to/private-native-benchmark
SHADER_COMPILER=/path/to/native-shader-compiler
NAGA_BIN=/path/to/naga-30/bin

PATH="$NAGA_BIN:$PATH" \
VOLVOXAI_NATIVE_SHADER_COMPILER="$SHADER_COMPILER" \
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_C_COMPILER=clang \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVOLVOXAI_ENABLE_VULKAN=ON -DVOLVOXAI_ENABLE_OPENGL=ON \
  -DVOLVOXAI_ENABLE_CUDA=ON -DVOLVOXAI_CUDA_ARCH=86 \
  -DVOLVOXAI_CUDA_FAST_FP32=OFF
PATH="$NAGA_BIN:$PATH" cmake --build "$BUILD" \
  --target native_dynamic_batch_benchmark

PATH="$NAGA_BIN:$PATH" python3 \
  "$ROOT/examples/native_dynamic_batch_benchmark/run_matrix.py" \
  --config "$RUN/receipt-native-matrix-input.json" \
  --output "$RUN/receipt-native-dynamic-batch.json"
sha256sum "$RUN/receipt-native-dynamic-batch.json"
```

The input schema is `volvoxai.native-dynamic-batch-matrix-input`. The wrapper
does not contain a B16/B32/B64/B128/B256 allowlist: it derives each workload's
batch set from the cases, requires the full Vulkan/OpenGL/CUDA x FP32/INT8
cross-product, accepts an executed B covered by the declared package domain,
and requires every B of one precision/workload to use the same exact package.
This campaign's config selects B16/B32/B64/B128/B256, warmup/repeat 3/5,
delay 5000 ms, and Vulkan arena 6144 MiB. The wrapper rejects inherited
performance/device overrides and injects the validated Vulkan arena and CUDA
device settings into child processes. Keep the input config and receipt lanes
private; only the path-free canonical output is suitable for publication. A
clean checkout still needs the two external, hash-bound maxB256 packages and
the private receipt lanes to remeasure.

### RTX 3090 native CUDA: explicit authored B4

A fresh `sm_86` native CUDA build also qualified B=4, but through a different
contract: the caller supplied one explicit `[4, 1, 320, 672]` bulk tensor to a
graph authored for B=4. At the time of this retained run, the built-in native
provider exposed only a B1 scheduler contract. The current native coordinator
can coalesce separately submitted B1 requests for proved symbolic-B graphs on
Vulkan, OpenGL, and CUDA, but this historical explicit-bulk report did not
exercise that path and is **not** its evidence.

| Variant | Four serial same-CUDA B1 | Explicit CUDA B4 | Useful throughput gain | Lane parity |
| --- | ---: | ---: | ---: | --- |
| FP32 | 26.248 ms | 20.176 ms | 1.301x | all four byte-exact |
| INT8 | 22.264 ms | 16.937 ms | 1.315x | all four byte-exact |

This small three-sample qualification is directional rather than an optimal-B
sweep. Both reports bind native binary SHA-256
`aee0fede14dafd970ad50aea966c35bf4f3e3add0b63836a6aed483d191b4d1b`.
The FP32 report is
`/path/to/volvoxai-gpu-validation/results/native-cuda/digit-fp32-cuda-b4-parity-fresh.json`
(SHA-256
`6ecf9d58f0b854bec6e7da9002a86547fea7cc09194ac1bc927c82d6dc945e6e`),
and the INT8 report is the adjacent
`digit-int8-cuda-b4-parity-fresh.json` (SHA-256
`dc4229f919a4582be77839dc7cebd07b9be06e639985ca94759cc7346c726711`).
These CUDA reports and their one-off explicit-bulk harness are also external
archive evidence, not files tracked by this repository.

### Historical static-B1 boundary

The earlier runtime-plan table used a default-imported package whose graph ABI
was statically authored at B1. Its retired ADAPTIVE/SERVICE-labelled groups
correctly produced N
physical B1 invocations, and those results remain useful as a fail-closed
boundary check. They do **not** describe the opt-in `--max-batch-size 4`
packages above. The former limitation came from package authoring, not from a
claim that this graph's convolution, normalization, or attention operators are
intrinsically non-batchable.

## What the numbers say

**VolvoxAI native CPU takes 1.10–1.38× as long as ONNX Runtime.** The gap is
widest on the imported INT8 graph (1.38×) and narrowest on FP32 (1.10×). This
is the primary optimization target and the reason this document exists.

**The PTQ path is the fastest VolvoxAI route in the aggregate table.** Against the
imported ORT QDQ package it is 1.13× faster on the host CPU (27.64 vs 31.26 ms)
and 1.29× faster on host WASM (58.09 vs 74.79 ms) — and 1.27× on the Android
CPU, its widest margin. Meanwhile
`target_exact` differs by 0.20 percentage points in the aggregate table:
0.9690 for PTQ against 0.9670 for imported INT8, or four additional exact
targets out of 2,000. Per-image discordant-pair outcomes are not present in a
tracked public report, so this document does not attach a McNemar p-value to
that aggregate difference.

Graph structure provides a concrete explanation for the speed difference. The
`--prepare-fp32-for-ptq` pipeline fuses `Sigmoid × Mul` into 14 `SiLU` kernels
while the graph is still F32; the imported QDQ graph cannot fuse across
quantization boundaries already inserted by the upstream tool, so it carries
28 separate nodes there. It also
quantizes 11 of 11 convolutions where the producer left one in float, and
crosses 5 fewer byte boundaries. Node count is 102 against 121.

**Quantization buys much more on WASM than on native CPU.** Host native goes
36.63 → 27.64 ms (1.33×) from FP32 to PTQ, while host WASM goes 168.74 → 58.09
ms (2.90×). The WASM F32 route is the weakest link by a wide margin.

## Android results

Measured 2026-08-14 KST on a Samsung SM-A566S (Galaxy A56, Exynos `s5e8855`,
Android 16). Every process is pinned to CPU 7 with `taskset 80` and requests one
runtime thread. The native binary is built with the NDK 28.2 toolchain,
`arm64-v8a`, `android-29`. Deno is the Android-native `aarch64-linux-android`
2.9.4 build already staged on the device; it runs with `--no-code-cache` so no
sample reuses a compiled module from a previous one.

Same timing contract as the host tables: one retained session, execution and
output snapshot only, median of the measured runs.

| Route | FP32 | INT8 | PTQ (Conv only) |
| --- | ---: | ---: | ---: |
| VolvoxAI native CPU | 88.86 | 43.02 | **33.85** |
| VolvoxAI Deno 2.9.4 WASM | 228.01 | 181.01 | 172.48 |
| VolvoxAI Deno 2.9.4 WebGPU | 207.08 | 175.57 | 175.23 |

Every route produced the same decoded record as the host FP32 reference. The
identifier itself is omitted because the input is a real receipt.

**PTQ is the best Android route by a wider margin than on the desktop.** Native
CPU goes 43.02 → 33.85 ms against the imported INT8 package, a 1.27× gain
where the desktop saw 1.13×. Against FP32 it is 2.63× faster. The Arm CPU
route is also the only one under 100 ms; both JavaScript routes sit above 170 ms
for every variant.

**WebGPU is not a win here.** At 175 ms for PTQ it is level with WASM (172 ms)
and 5.2× slower than the native CPU route on the same device. The model is small
enough that per-dispatch overhead dominates: 102 nodes at this size do not give
the GPU enough work to amortize submission. WebGPU's advantage over WASM shows
only at FP32 (207 vs 228 ms), where the arithmetic per dispatch is largest.

The WASM samples show a much wider spread than their medians — maxima of 778,
790, and 476 ms against medians of 228, 181, and 172 — which is why the median
is the reported statistic. The native and WebGPU routes are tight by comparison
(PTQ native ranged 33.74–34.26 ms).

## Native Vulkan/OpenGL: historical failure and resolution

The following records the failure sequence that led to the current fixes; it
is not the current backend status. Before those fixes, the built-in Vulkan and
OpenGL backends selected zero nodes on both the Linux host and Android device:

```text
BACKEND_UNSUPPORTED: built-in backend cannot attest the complete declared shape domain
provider=builtin:vulkan;nodes=102;selected=0;fallback=0;missing=102
```

The devices themselves initialized (`Samsung Xclipse 540` on Android, `RADV
RENOIR` on the host). Four facts located the problem:

- The committed `models/efficientdet_lite0_fp32` package failed identically on
  both machines, so this is not specific to the receipt reader.
- Exporting with `--target backend:vulkan` named the reason at authoring time:
  `Sigmoid at node_100 is not admitted by target backend:vulkan`. Re-exporting
  with `--allow-silu-numerical-migration` fuses all 14 `Sigmoid × Mul` pairs
  into `SiLU` and the capability check then passes for both GPU targets.
  (That workaround is no longer needed — see *Sigmoid was never qualified*
  below. `--target backend:vulkan` now admits the graph as authored.)
- That qualified package still selected zero nodes at runtime, and operator
  coverage is provably not the reason: every one of its eleven operators
  (`Add`, `BatchMatMul`, `Conv2D`, `Div`, `GroupNorm`, `LayerNorm`, `Linear`,
  `Reshape`, `SiLU`, `Softmax`, `Transpose`) is present in both
  `RUNTIME_OPERATORS_BY_BACKEND['opengl']` and `['vulkan']`. The exporter's
  capability table and the runtime's kernel selection therefore disagree.
- The refusal was not even uniform across packages. `models/tinystories_1m`
  failed on the same device with a *different* code,
  `CANONICAL_SHAPE_CONTRACT_UNSUPPORTED` ("node lacks a canonical native shape
  contract"), against `BOUNDED_DOMAIN_UNSUPPORTED` for this graph and
  EfficientDet. Three committed or published packages, two distinct refusals,
  zero nodes selected in every case.

At that stage there was consequently **no native OpenGL or Vulkan row to
report for any variant** — not a slow number, an absent one.

### Root cause, and the fix

`missing=102` was a consequence, not the cause. `runtime_route_backend[]` is
filled by the validation forward, which only runs after the bounded-domain
proof passes, so the proof failed first and every route stayed NULL. No kernel
was missing.

The proof's evidence came back empty because five of its failure paths exited
`VX_STATUS_BACKEND_UNSUPPORTED` without writing one. Giving them an evidence
clause identified the refusal immediately:

```text
required=native-gpu-static-domain-limits;
predicate_reason=validation=1,bounds_ready=1,limits_query=-1
```

`vx_native_gpu_query_domain_limits()` failed, inside a branch only **fully
static** graphs reach (`if (!dynamic && vx_native_gpu_backend(backend))`).

Device limits live on the engine state and are only readable while that state
is current. The dynamic path enters `vx_engine_state_scope_enter(validation)`
before its query; the static path called straight through, so it always read an
uninitialized device. **Entering the same scope fixes it**, and that is why
TinyReceiptVQA — which carries bounded symbols `B`, `Q`, `M`, `T` and therefore
takes the dynamic path — ran on these backends all along. Static shapes were
never the limitation; the missing scope was.

With the scope entered, both backends compiled and entered execution, exposing
the separate correctness defect below.

The scope was not EfficientDet's only blocker: it went on to fail at
`MaxPool2D` and then at `ResizeNearest2D`, both for the same reason `Sigmoid`
did (below). Both are now qualified, and `models/efficientdet_lite0_fp32` runs
on Vulkan and OpenGL, agreeing with native CPU to `5.1e-07` over all 1,728,540
score values and `8.6e-06` over the boxes, with the two backends bit-identical
to each other.

### The next defect this exposes, isolated to one tensor

Compilation was not yet sufficient for correct execution: the host CPU
reference decoded correctly while both GPU backends decoded the same input
incorrectly. Bisecting the graph localized it precisely, and **it was not an
operator bug**.

Promoting every intermediate to a graph output and diffing CPU against OpenGL
node by node shows all 80 tensors agreeing to within `2e-5` relative — and the
GPU result becomes *correct*. Promotion's only effect is to give each tensor
its own persistent buffer, so the fault is in buffer reuse, not arithmetic.

A prefix bisect over the 80 tensors narrows it to exactly one:

| Promoted to an output | OpenGL result |
| --- | --- |
| nothing | mismatches CPU reference |
| **`v90` alone** | **matches CPU reference** |
| `v91`, `v88`, or `v92` alone | still wrong |

`v90` is a `SiLU` output of shape `[1, 256, 10, 42]` whose only consumer is a
`Reshape` to `[1, 256, 420]` — the same element count, so the runtime aliases
the reshape onto the producer's buffer in place. It is also **the only
`SiLU → Reshape` adjacency in the graph**; the other thirteen SiLU outputs feed
`Conv2D`, `Transpose`, or `Linear` and none of them misbehave.

The wrong answers were also unstable: five consecutive runs of the identical
binary, package, and input produced four distinct wrong decodes. That rules out
a stable alternate decode formula without publishing receipt-derived values.

### Root cause: a device alias merges two activation spans

Disabling the alias — making the storage view copy instead — made both backends
match CPU exactly and repeat bit-identically, which isolates the mechanism to
`opengl_graph_alias_f32` and its Vulkan and Metal twins.

Device slots are keyed by **host pointer**, and the runtime pools activations so
that many tensors with disjoint lifetimes share one host span. Tracing the real
forward shows 80 tensors sharing exactly four host spans. `v90` and `v92` are
both planned into span C; `v91`, the reshape between them, is planned into
span A:

```text
alias in=<span C> out=<span A> n=107520 srcslot=13 dstslot=1 srcbuf=15 dstbuf=7
```

Pointing span A's slot at span C's buffer does not extend one tensor's lifetime.
It **merges the two spans on the device for the rest of the graph**: every later
tensor planned into either span now writes through both. The immediate casualty
is `v92 = Transpose(v91)`, which reads span A and writes span C — one buffer
bound as both operands of a single dispatch, which is why the answer changed
from run to run. The planner proved those lifetimes against distinct storage, so
a backend may not quietly make the storage shared.

The fix is that distinct host storage must be a real copy. Only a view the
runtime has *already* aliased on the host (`in == out`) may share a device
buffer; that case still costs nothing. All three GPU backends carried the same
defect and all three now take the copy path, which is what the domain-enforced
path had always done.

Both backends now match the native CPU decoded record to `4.7e-05`, are
**bit-identical to each other**, and repeat byte-for-byte across five runs. This
was verified on two independent driver stacks — NVIDIA 535 on an RTX 3090 and
Mesa RADV on this host's Cezanne iGPU — so it is not a driver quirk. Historical
focused validation pinned storage-view alias isolation and failed against the
previous code.

### Sigmoid was never qualified, which is why only PTQ ran

With the alias fixed, PTQ ran on both GPU backends but FP32 and INT8 were still
refused at `Sigmoid`. PTQ was not special: its authoring runs the graph
optimizer, which fuses `Sigmoid × Mul` into `SiLU`, so its graph has no
`Sigmoid` left. The other two keep 14 of them.

`Sigmoid` already carried the canonical `volvox.shape.activation-preserve.v1`
contract — the same one `SiLU` and `GELU` carry — and a real device kernel on
all four native GPU backends. It was missing from two places, with no recorded
reason:

- the `qualified-native-gpu` operator set in `proto/kernel_registry.proto`, so
  `exporter_qualified` generated as `0`;
- the shape-preserving branch of the native-GPU bounded-domain proof, so it
  fell through to `missing-native-gpu-domain-proof`.

`MaxPool2D` and `ResizeNearest2D`, EfficientDet's remaining blockers, were the
same omission in the same two places. They are not shape-preserving, so they
share a new NHWC proof instead: batch and channel carried through unchanged,
and the launch bounded either as Vulkan/OpenGL's 8x8 tile over output
width/height with one Z workgroup per batch/channel pair, or as CUDA's flat
one-thread-per-output. Both are qualified on Vulkan, OpenGL, and CUDA only —
**Metal has no F32 kernel for either**, so it stays unqualified rather than
qualified on a kernel it does not have. The byte domain reaches its pooling and
resize kernels through the separate W8A8 route, whose packed-lane launch this
proof does not cover, so `models/efficientdet_lite0_int8` still refuses, and
says so precisely: `predicate_reason=nhwc-spatial-launch-domain`.

Adding it to both makes all three variants compile and run on both backends as
authored. FP32 agrees with native CPU to `1.97e-05` and both GPU backends are
bit-identical to each other. The INT8 and PTQ CPU-vs-GPU deltas (`0.15`–`0.17`,
`0.09`) are the pre-existing quantized-path difference, not `Sigmoid`: a
control INT8 package with the `SiLU` migration applied — no `Sigmoid` anywhere
in the graph — reproduces the identical `0.168`. Every variant decodes both
producer-release fixtures correctly on NVIDIA 535 and Mesa RADV alike, and the
held-out record agreement above puts a number on what that delta costs: nothing
for FP32 and INT8, one record in 300 for PTQ.

A native GPU row is therefore no longer blocked. It is still unmeasured:
publishing a latency number needs the same receipt image and quiescent host as
the rest of the table.

## Optimization backlog

Ordered by the evidence above:

1. **Native CPU kernels, INT8 first.** 1.38× ONNX Runtime's time on the same
   graph is the largest single gap. Use a production-linked scratch profiler
   with a shared shape table and independent numerical oracle to attribute it per
   kernel before changing anything — an end-to-end median cannot say which
   operator is responsible.
2. **WASM F32.** 5.06× is far worse than the 2.55× the same route reaches once
   quantized, which points at the F32 convolution and GroupNorm paths rather
   than at framework overhead.
3. **Measure the native Vulkan/OpenGL B1 row.** All three variants now run on
   both backends as authored, so what remains is the measurement against the
   same image and host as the rows above. The separate CUDA explicit-B4
   qualification does not supply this row. A storage view now copies on every
   GPU backend, so this route pays one buffer copy per `Reshape`; if that shows
   up, the way to remove it is to teach the activation planner about device-side
   views, not to restore the alias.
4. **Android WebGPU dispatch overhead**, which leaves the GPU level with WASM
   and 5.2× behind the device's own CPU route on a 102-node graph.
5. **Extend PTQ's structural advantage.** SiLU fusion and boundary reuse are
   already worth 12–22%; `QGroupNorm`/`QSiLU` coverage would extend it, but
   costs accuracy on this model today, and needs the score-matmul range problem
   solved first; the example README records the measured coverage trade-off.

## Related

- [Example README](../examples/receipt_digit_reader/README.md) — accuracy,
  coverage trade-offs, and the import pipeline.
- [Typed PTQ](typed-ptq.md) — why an imported QDQ package and a VolvoxAI-authored
  one cannot converge numerically.
- [TinyReceiptVQA benchmark](tiny-receipt-vqa-bpe1536-benchmark.md) — the
  timing-contract conventions this document follows.
