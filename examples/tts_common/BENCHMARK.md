# Complete Piper and Kokoro comparison — 2026-10-29

The measurements in this document belong to the dated experiments described
in each section. They do not qualify the current working tree or establish a
current speed advantage over ONNX Runtime. A new comparison must retain the
model, request, runtime and library hashes, provider placement, thread count,
timing boundaries and raw samples for both engines. Earlier FP32, quantized,
native and web measurements have different scopes and baselines.

The 2026-11-05 review found that `benchmark_quantized.py` accepted references
produced by VolvoxAI while labeling them `against_original_ort`, and could
return success after runner cleanup failed. It now requires explicit ORT
reference provenance and deterministic requests, validates the first request
and every warmup as well as measured requests, and reports cleanup errors as
failures. Empty case lists and invalid reference waveforms are rejected.
Native and web comparisons now also require explicit ORT reference provenance;
native comparisons reject empty requests and empty/nonfinite references.
Older receipts need their reference provenance checked before they can support
an ORT accuracy claim. A same-model, same-backend output hash that
survives an optimization proves only equality for those prepared requests;
it does not establish FP32/ORT parity or speech quality. A retained
`numerical_difference` status alone also does not prove numerical regression
absence. Fresh validation results are recorded separately from the historical
timings below.

## Correctness recheck on 2026-11-05

The library at the end of this initial review stage has SHA-256
`289e8d64dbaabd4533637844d251e8c6a7999559988325d52a332f56eff82cfa`.
The ABBA recheck used fresh original-ORT references and checked the first request,
all warmups and every measured request. The retained before library includes the
initial review fixes; it is not an unmodified build of commit `29ade3b`.

| Prepared request | Maximum absolute error, before → after | RMSE, before → after | Strict mismatches, before → after | Final result |
| --- | ---: | ---: | ---: | --- |
| Piper, 14,592 samples | 1.2822449e-5 → 1.5817583e-5 | 1.2738393e-6 → 1.2707572e-6 | 0 → 2 | Declared PCM16 maximum/RMS limits pass; strict elementwise parity fails |
| Kokoro short, 33,600 samples / 1.4 s | 0.11359594 → 0.10633671 | 0.00577642 → 0.00601780 | 14,187 → 14,068 | `numerical_difference` |

Piper's final SNR is 99.2593 dB. Its maximum error and strict mismatch count
increase while RMSE decreases. Kokoro's RMSE increases even though its maximum
error and mismatch count decrease. These are changed waveforms, not evidence of
numerical non-regression. The Kokoro request differs from the 1.65-second request
in the historical tables. No error threshold was relaxed.

Local latency measurements used an AMD Ryzen 5 5600U, four threads on distinct
physical cores (`taskset -c 0,2,4,6`), three warmups and nine measurements in each
process, in before/after/after/before order. Each variant therefore has two
independent process trials and eighteen requests; eighteen requests are not
eighteen independent trials. Task-owned builds and tests had finished, but the
shared workstation and CPU affinity do not exclude unrelated workloads. ORT
always ran first in each process. Model, input, reference and library hashes
were checked between trials.

| Prepared-input-to-host-waveform latency | Before median | After median | Observed change |
| --- | ---: | ---: | ---: |
| Piper | 48.665 ms | 48.924 ms | +0.53% |
| Kokoro short | 560.381 ms | 564.126 ms | +0.67% |

Trial medians vary more than these pooled differences: Piper native
46.884–50.410 ms and Kokoro native 545.079–577.287 ms. The measurements establish
neither a dependable speedup nor a statistically resolved small slowdown.
Same-run ORT trial medians were 29.709–36.475 ms and 400.809–416.754 ms,
respectively. This environment supplies no evidence of a Volvox speed advantage
over ORT. Kokoro timings remain numerically failing diagnostic measurements.
Raw samples, load observations, hashes and trial results are in
`build/review-29ade3b-followup/benchmark-summary.json` and its referenced receipts.
The earlier `*-after-correctness.json` files were collected during builds and
are retained only as numerical diagnostics.

Fresh quantized checks also remain `numerical_difference`: Piper acoustic W8A8
has original-ORT RMSE 0.01370489; Kokoro decoder grouped W8/G128 with F32
activations/arithmetic has RMSE 0.02369172. These checks ran alongside validation
work, so their latency is not performance evidence. They use independently
regenerated ORT references and preserve both own-engine-FP32 and original-ORT
comparisons. They do not qualify perceptual quality or unseen speech.

An independent one-thread diagnostic used the retained pre-follow-up library
and the same live native encoder tensors for the original and lowered ONNX
decoders. Native encoder text/predictor features passed the existing strict
elementwise bounds, and durations were identical. Passing those features to
the **original ORT decoder** was sufficient to produce waveform RMSE 0.00566540
against that decoder's own ORT encoder features, with 126 phase jumps exceeding
six radians. The lowered and original ONNX decoders, given identical native
encoder features, differed by RMSE 0.0000174785. Their STFT inputs were
bit-identical; a maximum STFT difference of 0.0000000596046 became a maximum
phase difference of 0.0119095 radians.

The source network's sensitivity was also reproduced without Volvox execution:
moving each original ORT encoder feature one F32 value toward positive
infinity produced waveform RMSE 0.00552114 on the short request and 0.00297510
on the historical 1.65-second request, with 121 and 167 phase jumps respectively.
These are diagnostic perturbations, not inference corrections. They identify
an amplification mechanism in the original model; they do not qualify Kokoro
waveform parity, perceptual quality or every runtime kernel. No tolerance was
relaxed and no phase correction was applied. Source/library/input hashes,
intermediate comparisons and outputs are retained under
`build/review-29ade3b-followup/kokoro-diagnostic/`.

### Final artifact and regression checks

Native inference/full CLI and libraries, both WASM profiles and both JS profiles
build successfully. The public API corpus passes 90 cases in each native CPU
profile and 90 cases in each actual release WASM profile (360 profile/case
checks, each case executed twice). The Vulkan route passes 49 TTS cases with
backend attestation; this report does not identify a physical GPU model.

Focused CPU/benchmark tests pass 65 tests; JS/API/training integration passes
97; Python training/dynamic-fusion/PTQ passes 11. Native invariant/ISA/static-GPU
checks report 78 passes and four unsupported AVX-VNNI/AVX512-VNNI skips out of
82. CUDA host-emulation/dispatch/replay/PTX checks pass 50 tests. These suites
have overlapping coverage and must not be summed as independent model tests.
The sampled Tanh audit covers 2,000,006 values, with observed maximum error
3 ULP; it is not an exhaustive F32-domain bound. Native GEMM's caller-thread
leak reproduction also passes AddressSanitizer after the lifetime fix.

All eight fixed release artifacts pass provenance, ABI and inference/full
symbol-boundary checks after rebuilding JS against the final WASM. API
conformance and `git diff --check` pass. **Release size still fails six budgets:**

| Artifact / metric | Final bytes | Budget bytes |
| --- | ---: | ---: |
| lite WASM gzip | 541,301 | 531,457 |
| native-lite raw | 2,408,584 | 2,382,727 |
| native-lite gzip | 1,089,484 | 1,031,017 |
| native full gzip | 2,680,326 | 2,563,074 |
| inference browser pair gzip | 582,208 | 572,057 |
| full browser pair gzip | 933,297 | 927,195 |

Budgets were not raised. Four categories already failed the retained 10/29
report; native-lite raw and the full browser pair are additional categories
relative to that older report. The final artifact is not release-size-qualified.
At that stage, physical CUDA execution/sanitizer/performance and physical
ARM/NEON validation remained unperformed; the configured GPU host had been
unreachable before any remote files were created. The subsequent hardware
validation below supersedes that limitation. PTX compilation, host emulation
and NEON syntax checks do not replace device execution. That stage's hashes and
receipts are indexed in `build/review-29ade3b-followup/final-summary.json`.

## Physical CUDA and Android validation on 2026-11-05

The hardware pass found and fixed additional defects. CUDA Tanh lost tiny
values and negative zero through exponential cancellation; it now preserves
tiny inputs directly and uses the CPU approximation near zero. CUDA Conv could
overflow on finite cancelling operands. Only nonfinite accumulators now enter
an outlined recovery: finite participating operands are recomputed in F64 and
cast back before activation/residual addition, while actual NaN/Inf operands
retain the original arithmetic. Finite F32 results are unchanged by that repair.

On ARM, compiler contraction of accumulator scaling and zero-point addition
changed an exact half-integer into a value just above it. For example,
`4375 * 0.004f + (-17)` becomes `0.5f` with the specified separate F32 rounding,
but approximately `0.50000083f` when fused, changing the ties-to-even output
byte from zero to one. Packed QConv2/3/4 and ARM QLinear/QConv now share the
existing affine helper that preserves those rounding boundaries. Global
floating-point compiler flags were not changed.

An additional contraction-enabled x86 build exposed the same problem in AVX2
vector requantization, including the store helper shared by VNNI paths. The
vector product now has an explicit F32 rounding boundary, and scalar tails use
the canonical helper. The regression enables `-mfma -ffp-contract=fast` and
covers channel counts 1/8/17/33 and both packed/general byte-weight routes;
the ordinary contraction-disabled ISA test alone could not detect this defect.
The vector boundary adds stack traffic; the FP32 TTS timings do not qualify
its effect on complete quantized models.
A separate, quiet AVX2 microbenchmark used Clang 14, one pinned worker,
I8/U8 weight routes and four ABBA cycles. Small `M=1, K=32, N=32` calls cost
approximately 1–10 ns more (+0.47–4.60%); medium/large median changes range
from −2.94% to +0.42% and do not establish a speedup. Normal-input outputs
are bit-identical. The small-call cost is an observed correctness tradeoff,
not a claim of performance neutrality for all quantized workloads.

The padding zero-sign repair is also outlined and compiled for size. SIMD
ReLU lanes already have canonical positive zero and skip that repair; AVX2
scalar channel tails still receive it. Together with the affine fix, this
reduced the ARM native binary's text by 7,696 bytes. A separate pinned x86
kernel experiment reduced an all-zero padded
ReLU case from 1.178 to 0.177 ms, with identical output bits; ordinary dense
Conv stayed approximately 0.178 ms. This exceptional-input microbenchmark is
not a whole-model speed claim.

### CUDA correctness baseline, ORT placement and numerical results

The device was an RTX 3090 with 24 GiB memory and NVIDIA driver 535.309.01.
The first hardware-qualified native library was
`c51807b119023259897ad23041121cd7f780ae739dda7ee8190c78f7e5a1c1a6`.
The final optimized artifact is recorded in the next subsection. This baseline's
embedded PTX was built with Ubuntu Clang 14.0.0, `sm_75`, PTX 6.5,
`-O3 -ffp-contract=off`; execution was on the physical RTX 3090. The final
embedded PTX passed Compute Sanitizer 12.1 memcheck with **zero errors** in
seven native and 27 Python checks, including 65,537 Tanh oracle values.
The public dispatch corpus passed all 90 cases with no operator fallback.
LSTM passed 21 fixtures, including the finite-Tanh tail fixture, with two
different live requests each. Conv's grid-y boundary passed two live requests at width 1,048,561;
the test explicitly supplied the required output-memory budget.

ORT 1.23.2 refused strict CUDA-only execution because graph nodes require its
CPU provider. That refusal is retained. The successful comparison explicitly
allowed **CUDA+CPU placement**, with effective `use_tf32=0`; it is not a
pure-CUDA ORT result. The separately collected provider trace recorded
1,882 CUDA / 659 CPU node events for Piper and 1,950 / 381 for Kokoro.
Volvox required CUDA throughout both model stages and attested zero fallback.

Prepared host inputs through completed host waveform were timed, excluding
loading, preparation, file I/O and assertions. Each engine used one host
thread, three warmups and ten measured requests per process. The following
are same-run mixed-ORT comparisons; they are not independent multi-process
ABBA performance qualification:

| Request | Volvox CUDA median | ORT CUDA+CPU median | Volvox vs same-device ORT maximum error / RMSE |
| --- | ---: | ---: | ---: |
| Piper, 14,592 samples | 16.233 ms | 11.073 ms | 1.36830e-5 / 1.44895e-6; declared PCM16 bounds pass |
| Kokoro short, 33,600 samples | 64.713 ms | 63.390 ms | 0.0473227 / 0.00261069; strict comparison fails |

Against the canonical CPU ORT reference, current CUDA Kokoro has maximum error
0.114228 and RMSE 0.00678721. The earlier hardware baseline had RMSE 0.00555349;
the corrected Tanh changes the waveform, so this is not a bitwise regression
absence claim. ORT CUDA+CPU itself differs from the same CPU reference by
maximum 0.125252 / RMSE 0.00674743. Cross-provider sensitivity therefore does
not establish a Volvox kernel defect, and it does not qualify Kokoro quality.
Piper passes its declared waveform bounds against both references, while the
canonical CPU comparison still has two strict elementwise mismatches.

The current exporter removes redundant broadcast expansion: Kokoro decoder
nodes fall from 2,466 to 2,006, and Expand nodes from 456 to three. The candidate's
528 tensor entries reuse byte-identical values from the unchanged original
weight bank; only four scalar constants (16 data bytes) are added. On CUDA,
both complete pipelines produce exactly the same output
bytes and SHA-256, with zero fallback. The paired mixed-ORT runs observed about
64.71 ms with the old graph and 61.12 ms with the compact graph. This is a graph
optimization observation with one process per variant, not a resolved kernel
speedup or a remedy for Kokoro's oracle mismatch.

### Qualified CUDA reduction and final paired measurements

The final library is
`97fdb2aaf663b80dc27203d740488c1ed9bf764506a9550f9d4f6c3cc223dcdb`,
with embedded PTX
`d42999c8643abf7142ef2e0c164684585c8c0d82457d9dd536e904cbba1eb018`.
The new reduction loads an 8-row × 128-column tile into shared memory with a
padded row stride. One thread still adds each row's values in the original
ascending order and scales only after summation. It changes memory access,
not the reduction tree. Admission is limited to 128–1024 rows and 128–16384
columns with disjoint input/output spans; other shapes retain the scalar route.
A failed launch remains an error rather than silently switching implementations.

The physical shape sweep covered 192 shapes; all 77 measured shapes within
the chosen admission range improved, with the smallest observed improvement
13.5%. Smaller row counts sometimes regressed and were excluded. Standalone
direct/captured execution passed 51 checks, including changed inputs, against
an independent ascending-F32 oracle. The final embedded PTX passed memcheck
with zero errors. A permanent public API test passed 28 requests at admission
boundaries and outside them, checking changed inputs, cancellation, NaN/Inf
and signed zero. The final library also passed the 90-case CUDA corpus,
21 LSTM fixtures with two requests each and both large Conv grid-boundary
requests.

Whole-model measurements alternated **O/A/B/B/A/O**, where O is original ORT
CUDA+CPU with TF32 disabled, A is the preceding `c51807…` library and B is the
final `97fdb2…` library. A/B use the same compact graph and original inputs,
isolating the runtime change from graph compaction. Each process uses three
warmups and ten measurements: twenty requests from two independent processes
per variant, not twenty independent trials. Timing includes prepared host
inputs through completed host outputs. Profiling is separate.

| Request | A pooled median | B pooled median | ORT pooled median | B versus A |
| --- | ---: | ---: | ---: | --- |
| Piper | 16.093 ms | 16.214 ms | 11.024 ms | +0.75%; overlapping samples, no speedup claim |
| Kokoro short, compact graph | 61.242 ms | 50.482 ms | 63.648 ms | −17.57%; both B trials are consistently faster |

The two Kokoro A trial medians are 61.242/61.241 ms and B medians are
50.481/50.507 ms. Actual A/B output arrays and hashes are identical for Piper
and for both the original and compact Kokoro graphs. This optimization adds
no waveform difference to those requests; Kokoro's existing strict failure
remains. The profiled process records 528 tiled reduction launches with
34.42 ms aggregate device time, including setup and warmup work. That aggregate
is not a single-request latency or a count of complete measured requests.

Using the Python examples' `round(clip(audio) * 32767)` conversion, Piper's
same-device ORT comparison has maximum one integer PCM16 step and RMS
0.141218 steps. Kokoro has maximum 1,551 / RMS 85.540409 steps. These rounded-file
differences are separate from the continuous F32 waveform error limits above;
a declared waveform pass does not mean identical PCM16 files. Evidence is in
`cuda/full-paired-86a90657/`, `cuda/paired-latency-summary.json` and the sanitizer
collection receipts below the hardware evidence directory.

The final same-device F32 comparison is maximum 1.14739e-5 / RMSE 1.35588e-6
for Piper and maximum 0.0473227 / RMSE 0.00261069 for Kokoro. The differences
from earlier Piper comparison statistics reflect separate ORT observations;
the actual A/B Volvox arrays remain identical.

### ARM execution and optimization priorities

The attached ARM64 Android 11 / API 30 device has four Qualcomm Kryo cores,
NEON, and no dot-product or i8mm extension. Both inference/full profiles passed
the 90-case public corpus. Eight native binaries passed, including the expanded
math probe's 485,908 checks with one and four threads. Packed Conv2/3/4
passed 1,152 exact-output cases, including the affine rounding regression.
Dot-product paths were compiled but cannot be physically qualified on this
device. No NEON result is presented as a dot-product or i8mm execution result.

The interleaved four-thread order was ORT/A/B/C/C/B/A/ORT. A is the retained
hardware-pass baseline, which already contains earlier correctness fixes;
it is not the original unmodified commit. B adds the ARM affine and compact
zero-repair changes; C also uses the compact Kokoro graph. All A/B/C outputs
are bit-identical for both prepared requests.

The ARM timing boundary differs from the CUDA table: it sums resident public
`Execute` calls for the two native stages and compares them with original ORT
`Run`. Setup, file transfers and host work joining the stages are excluded;
profiling is collected separately. Each variant has two trials, with each
stage in its own resident process and one warmup plus three measured calls
per stage. This yields six stage-sum observations per variant.
Affinity includes all four CPUs. These are prepared execution costs, not
complete application request latency, and must not be mixed with the CUDA
host-to-host measurements. Timings overlap substantially:

| Request | A median | B median | C median | ORT CPU median |
| --- | ---: | ---: | ---: | ---: |
| Piper | 240.99 ms | 253.43 ms | 241.46 ms | 171.06 ms |
| Kokoro short | 3,367.80 ms | 3,325.16 ms | 3,357.53 ms | 3,642.71 ms |

For Piper, B and C have identical binary and graph, yet differ by about 12 ms.
These runs do not resolve a whole-model gain from either change. Piper passes
the declared PCM16 criterion against same-device ORT. Kokoro remains a strict
failure; its timings describe execution, not numerically qualified throughput.

Native times include dispatch, input binding/copy and inline output
serialization, but exclude request encoding, response decoding and copying
between stages. Each stage repeats with resident inputs before the next stage
is prepared from its actual outputs. ORT times cover `Session.Run` through
returned values, excluding host tensor preparation and output file writing.
ORT 1.23.2 uses only the CPU provider, sequential execution, full graph
optimization, the requested intra-op count and one inter-op thread. Separate
profiles confirm CPU placement for every ORT node.

Restricting execution to the two higher-frequency cores (2/3) with two workers
gave Piper 304.68 ms and Kokoro 4,951.76 ms, versus ORT 272.73/4,903.64 ms.
These later runs have different thermal conditions; they do not establish a
scaling law or support recommending two cores for these complete models.
An isolated one-thread Conv experiment on core 2 does establish the targeted
repair improvement: the all-zero ReLU boundary case changes from 4.636 to
3.140 ms, while a finite-input control stays 3.112/3.109 ms. Output bytes are
identical in both cases. Full model A/B/C outputs are also bit-identical, and
the final x86-only affine change leaves both rebuilt Android binaries byte-
identical to the device-qualified artifacts.

Separate profiles, with no dropped node events, identify useful next targets.
Conv contributes 78.4% of Piper acoustic and 77.7% of Kokoro decoder node time.
Kokoro's large `K=11, C=128, OC=128, W=6721` Conv calls at dilation 1/3/5 take
about 83–89 ms each. Its encoder spends 66.5% in Linear and 23.8% in LSTM;
LSTM is about 36.6/36.4 ms at one/four threads. Piper's short `W=31` duration
Conv was slower at four threads in a single profile. Short-work scheduling,
NEON Conv tiling and recurrent projection work merit separate controlled
experiments. Profile sums and individual observations are not latency speedup
proofs.

The prepared decoder's duration guard remains necessary. A changed Kokoro
speed produces a new extent that original ORT can execute dynamically; this
fixed prepared package must reject it before the decoder runs. Native
`synth.py` prepares a matching specialization as documented in the model guide.
The prepared-request benchmark does not demonstrate arbitrary-duration reuse.

Hardware receipts, compiler/model/input hashes, raw samples, output tensors,
profiles, numerical diagnostics and task-directory cleanup records are retained
under `build/review-29ade3b-hardware/`. No tolerance or release-size budget was
relaxed. The local inference/full builds and boundary checks pass, but the
release still exceeds six size budgets and Kokoro still lacks waveform parity.

Final local validation against the rebuilt artifacts passes 90 cases per
native CPU profile and 90 per actual release WASM profile, 68 focused tests,
97 JS/API/training tests and 11 Python training/dynamic-fusion/PTQ tests.
Native invariant/ISA/static checks report 78 passes and four unsupported ISA
skips out of 82; CUDA source/host checks pass 55. The final CPU library also
produces byte-identical Piper and original/compact Kokoro outputs to the
hardware-pass baseline at both one and four threads. These overlapping checks
do not prove regression absence for every model or input.

All eight fixed artifacts pass provenance, ABI and inference/full symbol
boundaries. Typecheck and API conformance pass. Current size failures are:

| Artifact / metric | Final bytes | Budget bytes |
| --- | ---: | ---: |
| lite WASM gzip | 540,674 | 531,457 |
| native-lite raw | 2,395,464 | 2,382,727 |
| native-lite gzip | 1,084,373 | 1,031,017 |
| native full gzip | 2,682,502 | 2,563,074 |
| inference browser pair gzip | 581,581 | 572,057 |
| full browser pair gzip | 932,406 | 927,195 |

An isolated cold-code size experiment saved at most 2,515 gzip bytes for the
lite executable or 4,775 for full, which did not resolve these failures. A
WASM helper outlining experiment saved at most 535 gzip bytes with a hot-call
overhead risk. Neither was adopted, and the size budgets were not raised.

After local report/artifact verification, all task-owned remote directories
were removed. Removal receipts total 4,340,432,896 bytes on the CUDA machine
and 2,436,759,552 bytes on Android. These are cumulative removed task allocations,
including successive experiments, not a net change from the initial free-space
reading. Shared models, ORT installations and existing services were preserved.
The complete index of final artifacts, source hashes, checks and cleanup
receipts is `build/review-29ade3b-hardware/final-summary.json`.

## Historical experiment overview

The latest [bind-group investigation](#bind-group-reuse-investigation) records
an unadopted layout-only candidate and the measured cause of buffer churn.
The retained [WebGPU parameter-buffer results](#webgpu-parameter-buffer-reuse)
measure a bounded storage pool on Piper and its shared-path Kokoro regression.
The preceding [runtime optimization results](#runtime-optimization-after-the-ort-comparison)
include FP32/quantized improvements, numerical regression checks,
memory costs and four failing release-size gates. The earlier
[profiling follow-up](#profiling-and-refactoring-follow-up) and
[ORT comparison](#onnx-runtime-profiling-and-quantized-comparison) retain their
own baselines and experiments. Original measurements remain here for comparison.

Both examples execute the entire pretrained phonemes-to-waveform computation
in VolvoxAI. The application reads the predicted duration between two Volvox
stages to select a concrete output shape and passes their live tensors onward.
ONNX Runtime runs only as an independent reference and timing baseline.

Piper passed its declared waveform criterion on native CPU, CUDA, Vulkan,
OpenGL, both WASM profiles and physical WebGPU. Kokoro executes on all those
routes but **does not pass strict waveform parity**; its timings below are
numerically failing diagnostic measurements. Every Volvox stage attested the
requested backend with zero fallback.

Raw JSON and diagnostic receipts are local run artifacts under
`build/tts-examples/reports/` and are excluded from Git. This document retains
the summarized measurements.

The matrix (`build/tts-examples/reports/full/matrix.json`) retains each trial's
exit status.
The JSON summary (`build/tts-examples/reports/full/summary.json`) indexes timings,
numerical results
and individual reports with artifact hashes and raw samples.

See [Piper](../piper/README.md), [Kokoro](../kokoro/README.md) and the
[shared workflow](README.md) for preparation, synthesis and reproduction.
Earlier component measurements had a different computation scope and must not
be substituted for these full-model results.

## Same-machine latency

Median milliseconds after a separate first execution, two warmups and five
measured requests. Hardware: Intel Core i3-1115G4 and NVIDIA GeForce RTX 3090,
driver 535.309.01, Linux. Native CPU and ONNX Runtime use four threads. ONNX
Runtime 1.23.2 uses **CPUExecutionProvider** and `ORT_ENABLE_ALL`; no ONNX CUDA
provider was installed in the original environment. A separate temporary
ONNX Runtime GPU 1.23.2 overlay supplies the CUDA comparison, with TF32 disabled
and CPU node placement explicitly allowed and counted. WASM uses Deno 2.9.6.

| Complete model | ONNX CPU | ONNX CUDA + CPU | Volvox CPU | CUDA | Vulkan | OpenGL | WASM full | WebGPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Piper, 0.912 s audio | 38.403 | 11.043 | 58.740 | 22.649 | 83.642 | 50.212 | 341.473 | 797.821 |
| Kokoro, 1.65 s audio, parity failed | 666.294 | 73.795 | 1378.918 | 263.049 | 519.252 | 414.161 | 8687.564 | 7340.191 |

Each request starts with prepared host inputs and ends with completed host
outputs. Timing includes both Volvox stages, synchronization, transfer and the
host duration check. Loading, compilation, phonemization, validation and WAV
encoding are excluded. Reports retain setup, first-request and raw repeated
measurements separately. Each native trial also measures its own ONNX CPU
baseline; the table uses the baseline from the corresponding native CPU trial.

The separate ONNX CUDA reports
(`build/tts-examples/reports/full/ort-cuda/run-summary.json`) record
1882 CUDA / 659 CPU node events for Piper and 1950 CUDA / 381 CPU events for
Kokoro. CPU placement includes arithmetic and, for Kokoro, STFT, Atan and Resize,
so these are mixed-provider ONNX runs. They are faster than the measured Volvox
CUDA runs. Piper passes its waveform criterion; Kokoro also fails strict
comparison in ONNX CUDA mode (RMSE `0.003614`). The original Python environment
was unchanged; cleanup (`build/tts-examples/reports/full/ort-cuda/cleanup.json`)
recovered 877,625,344
bytes after report verification.

The inference-profile medians were 57.505 ms native / 340.507 ms WASM for Piper
and 1392.146 ms native / 8463.914 ms WASM for Kokoro. CUDA was the fastest measured
Volvox route. For these requests, Piper WebGPU is slower than WASM and leaves
little real-time margin. Kokoro's real-time factors are 5.27 on WASM and 4.45 on
WebGPU, so neither configuration synthesizes at real-time throughput. These
prepared short requests and five samples do not establish performance for all
voices, durations or devices.

## Numerical results

All routes must preserve output shape and dtype, produce finite waveform
samples and attest the requested backend for every stage with zero fallback.
Strict elementwise comparison uses `abs(error) <= 1e-5 + 1e-4 * abs(reference)`.
Reports retain maximum absolute error, RMSE, strict mismatch counts and SNR.

Piper also declares a waveform criterion of at most one PCM16 quantization step
(LSB) of peak error and one tenth of a step in RMS error (`1/32767` and
`0.1/32767`). Its preparation checks each stage with the strict elementwise
criterion. The native and WebGPU routes passed
strict comparison in this matrix. WASM had one strict mismatch among 14,592
samples, maximum error `1.524e-5` and RMSE `1.193e-6`; it passed the declared
waveform criterion. This is not a claim of bit-identical audio. Both Gaussian
draws are original request inputs with seed 42 and ordinary voice scales
`[0.667, 1, 0.8]`.

Kokoro keeps strict comparison and records `numerical_difference` with exit
code 3. Its waveform error against the original ONNX model is:

| Volvox backend | Maximum absolute error | RMSE |
| --- | ---: | ---: |
| CPU | 0.070598 | 0.003846 |
| CUDA | 0.059519 | 0.003101 |
| Vulkan | 0.080685 | 0.004208 |
| OpenGL | 0.060400 | 0.003666 |
| WASM | 0.082211 | 0.003889 |
| WebGPU | 0.076916 | 0.004159 |

The independent stability diagnostic
(`build/tts-examples/reports/full/kokoro-oracle-stability.json`)
also finds that changing only the original ONNX optimizer setting produces
RMSE `0.002198`, and shifting the oracle's encoder features by one F32 ULP
produces RMSE `0.002975`. The generator feeds raw spectral phase into learned
convolutions; rounding near a phase branch can amplify small upstream errors.
Volvox CPU's spectral magnitude relative L2 error was 2.77%, compared with
1.72% for the oracle optimizer change. These sensitivity measurements explain
a numerical limitation; they do not establish perceptual equivalence or waive
the failed parity check.

## Resource and verification coverage

The complete Kokoro Vulkan trial explicitly uses `VOLVOX_VULKAN_MB=4096`.
Its decoder has approximately 3263 MiB of distinct output buffers and 244 MiB
of weights. Its package uses the public streaming model-source API. New token
counts or predicted durations require preparing a new concrete graph; runtime
checks reject a duration that differs from the prepared extent.

Both native executable/library profiles, both WASM profiles and all JS release
variants built successfully. Native API/profile checks passed. Physical CUDA,
Vulkan and OpenGL passed 219 operator cases; both WASM profiles passed 146 cases;
physical WebGPU passed 73 operator cases. The qualification receipt
(`build/tts-examples/reports/full/operators/qualification-summary.json`)
records all 438 results and artifact hashes. Shared shader tests covered 93 cases.
WebGPU stages share one host/device. Bootstrap allocation reuses storage by
tensor lifetime, including immutable views, and bounded-shape admission proves
real maximum-sized operand spans. Tanh and approximate GELU use stable F32 tails
to avoid device-dependent NaNs at large finite inputs; LSTM applies the same
rule. Regression tests cover storage ownership, allocation failure and aliases.

The browser workbenches executed both complete models with WASM and produced
playable audio, while displaying Kokoro's failed strict comparison. Metal
translation checks passed; Apple hardware was unavailable for execution tests.

The release check verified the eight fixed artifacts, provenance and API/profile
boundaries. Sharing immutable shader metadata reduced full JS to 688,561 bytes
and its minified variant to 427,579 bytes, within their existing limits. Two
native gzip limits remain exceeded: lite 1,043,971 versus 1,031,017 bytes and
full 2,609,387 versus 2,563,074 bytes. Size budgets have not been raised.

The remote collection verified report hashes before deleting its task-owned
directory. The cleanup receipt (`build/tts-examples/reports/full/cleanup.json`)
records 814,186,496
bytes recovered; shared model assets and existing services were preserved.

## Profiling and refactoring follow-up

The follow-up uses the same prepared requests, model files, FP32 policy and
Intel i3-1115G4 / RTX 3090 server. Native CPU and ORT use four threads. Each
latency trial has one separate first request, two warmups and five measured
requests, from prepared host inputs through both completed host stages. Loading,
compilation, phonemization and audio-file writing remain excluded. No model
activation or waveform is reused. Both stages require the selected Volvox
backend with zero fallback; ORT remains an independent full-model comparison.

Raw evidence is under `build/tts-perf-20261029/`: `reports/baseline/` and
`reports/final/` contain ordinary timings, `reports/profiles-before/` and
`reports/profiles-final/` contain diagnostic traces and ordinary waveform
snapshots, and `reports/qualification-final/` contains operator results.
`final-summary.json` joins timings, model/runtime hashes, numerical comparisons
and qualification. `profile-summary.json` separates timing domains and records
coverage, memory scopes and source-to-runtime mappings. Candidate and ablation
reports are retained separately; their timings are not the final release table.
The baseline artifacts were snapshotted before editing. The prior TTS changes
became commit `aa778ba` during this session and were preserved.

### Final whole-request measurements

Full-profile medians in milliseconds. Reduction is `(before - after) / before`;
negative values mean slower. These compare the reproduced baseline with the
final artifacts, not the earlier table at the top of this document.

| Backend | Piper before → after | Reduction | Kokoro before → after | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Native CPU | 58.609 → 57.348 | 2.2% | 1403.071 → 1049.950 | 25.2% |
| CUDA | 20.974 → 21.207 | −1.1% | 262.800 → 263.863 | −0.4% |
| Vulkan | 84.190 → 83.718 | 0.6% | 519.888 → 518.771 | 0.2% |
| OpenGL | 50.380 → 49.580 | 1.6% | 411.478 → 413.587 | −0.5% |
| WASM | 341.118 → 341.559 | −0.1% | 8735.782 → 8185.139 | 6.3% |
| WebGPU | 758.577 → 550.074 | 27.5% | 7355.829 → 3797.510 | 48.4% |

The inference-profile medians were Piper native 57.889 → 57.136 ms and WASM
340.562 → 341.303 ms; Kokoro native 1395.176 → 1046.843 ms and WASM
8484.867 → 7979.920 ms. The reproduced ORT CPU baseline was 38.532 ms for
Piper and 664.747 ms for Kokoro. GPU routes other than WebGPU were not changed;
their small timing differences do not establish improvements. Five samples
of these fixed short requests also do not establish general performance across
voices, lengths or devices. Piper WebGPU still loses to WASM. Kokoro WebGPU
remains above real time: 3.798 s to synthesize 1.65 s of audio, RTF about 2.30.

Final native stage medians show where the completed request is spent. Stage
one is duration for Piper and encoder for Kokoro; stage two is acoustic or
decoder. Stage time includes the completed output copy. Host input binding is
measured separately before each stage.

| Model / backend | Stage one ms | Stage two ms | Both stages' host input binding ms | Both stages' host output copy ms |
| --- | ---: | ---: | ---: | ---: |
| Piper CPU | 16.388 | 40.716 | 0.276 | 0.456 |
| Piper CUDA | 12.065 | 8.781 | 0.270 | 0.551 |
| Kokoro CPU | 45.729 | 1003.841 | 0.247 | 0.398 |
| Kokoro CUDA | 79.131 | 184.262 | 0.239 | 0.593 |

Output copy is nested within stage time; input binding is additional. Request
timing also includes host shape selection and orchestration between stages;
independently computed medians need not add exactly. GPU output-copy spans can
include synchronization. Detailed web stage timings are retained separately
because timestamp collection changes execution overhead.

### What the profiles established

Piper WebGPU's original problem is mostly host-side upload work. A warmed
diagnostic request made 3,363 `writeBuffer` calls, moving 107.914 MB, and spent
about 535 ms inside those calls. Its separately captured complete basic GPU
passes took about 103 ms. Creating 2,361 buffers cost about 45 ms and 2,092 bind
groups about 32 ms. No shader module or pipeline was created in the warmed
request: the existing per-device pipeline cache was already effective.
The measured WASM Conv total was about 290 ms. Upload and object-management
overhead therefore explains why the original WebGPU request could lose to WASM
despite faster device computation. GPU clocks and trace instrumentation affect
individual captures; ordinary, detailed and basic captures remain distinct.

Kokoro WebGPU moved 1,155.180 MB in 5,164 uploads per request, with about 5.44 s
spent in `writeBuffer`. Its basic GPU passes took about 1.91 s in a separate
capture. The C planner passed every operand as a nonweight, so the bridge's
existing immutable-weight upload cache was never selected. Buffer creation
was also substantial: 3,697 allocations totaling about 568 MB per request.
Readback payloads were only about 106 KB for Piper and 232 KB for Kokoro,
including stage outputs. The stage handoff is necessary and remains measured;
it does not account for the gigabyte-scale upload traffic. Mapping wait time
includes preceding GPU computation and is not an isolated PCIe readback cost.

Final warmed WebGPU API captures further separate host submission from upload
and readback waiting. These are single diagnostic requests with API wrappers,
not the five-sample latency medians above.

| Host API observation | Piper | Kokoro |
| --- | ---: | ---: |
| `writeBuffer` calls / host ms | 2828 / 217.142 | 4082 / 2734.497 |
| Buffer creation calls / host ms | 2361 / 45.178 | 3697 / 50.611 |
| Bind group creation calls / host ms | 2092 / 31.550 | 3098 / 48.538 |
| Queue submissions / host ms | 10 / 2.183 | 42 / 6.443 |
| `mapAsync` calls / completion ms | 6 / 114.471 | 4 / 55.224 |

These API measurements cover calls reached through the JS/WASM bridge, but do
not isolate every bridge transition or all dispatch encoding from runtime
bookkeeping. Subtracting overlapping device and host spans would not produce
that missing exclusive cost. Readback waits overlap submitted device execution.
The complete host node spans and sampled device programs remain available for
further investigation without presenting this residual as a precise measurement.

CUDA has a different bottleneck. Kokoro's six LSTMs consumed 132.48 ms of actual
device-program time: 69.41 ms in the encoder and 63.07 ms in the decoder.
They account for roughly half its observed kernel time; decoder Conv adds
about 74.11 ms. The current bidirectional, batch-one LSTM launches two blocks
and performs long reductions and recurrence inside them, exposing little
parallel work on the RTX 3090. LSTM alone exceeds the independent ORT CUDA+CPU
whole-request median of about 74 ms. Piper CUDA has many small launches;
observed Conv program time was 4.71 ms in duration and 5.82 ms in acoustic.
The independent ORT CUDA rerun used 1.23.2 in a task-owned overlay, TF32 off,
and explicitly retained mixed CPU placement: Piper 11.067 ms and Kokoro
74.029 ms. No part of Volvox inference was redirected to that runtime.

Host node attribution can be misleading on CUDA. For example, a Kokoro encoder
host upload span took 69.169 ms, including 69.129 ms waiting behind earlier
device work; the actual upload was 1,092 bytes and 0.050686 ms. Charging this
to the following Expand would misidentify the bottleneck. Traced CUDA requests
also bypass replay and were slower than ordinary requests. Host spans, device
programs, enclosing passes and asynchronous waits overlap; they are not summed
to produce the latency table.

On native CPU, Kokoro decoder costs included Conv 412.15 ms, Expand 388.52 ms,
Sin 125.04 ms, Mul 115.99 ms, Add 75.13 ms, Gather 68.26 ms and ReduceMean
62.22 ms. Encoder Linear and LSTM were about 17.54 and 15.75 ms. On WASM,
decoder Conv alone took 6,567.58 ms and Expand 567.98 ms; encoder Linear and
LSTM took about 214.76 and 142.30 ms. The active dense WASM Conv already uses
SIMD128, with sixteen output channels and two neighboring pixels per interior
block. It remains a large single-thread FP32 computation, rather than an
unimplemented SIMD path.

Kokoro's 20-point STFT is lowered to Bluestein FFT arithmetic, without a
dedicated FFT/STFT dispatch. Reconstructing exporter provenance in memory
reproduced the prepared graph exactly and identified its 238 runtime nodes:
55 Gather, 63 Expand, 59 Mul, 15 Sub, 39 Add, three Concat, two Div and two
Unsqueeze. That region took 103.024 ms on CPU, 258.337 ms on WASM and 2.806 ms
in CUDA device programs. It overlaps the operator totals above. WebGPU,
Vulkan and OpenGL detailed timestamp pools exhausted before this region;
their missing spectral timings are not reported as zero.

### Changes retained and experiments rejected

CPU/WASM Expand now fills trailing broadcast runs with the current input's
exact 32-bit pattern, advancing coordinates once per run. This preserves
signed zero, NaN payloads and integer values, introduces no FP arithmetic,
workspace or cross-request cache, and substantially reduces Kokoro's repeated
`[1,C,1]` to `[1,C,T]` expansion cost.
Final decoder Expand time was 43.71 ms on CPU versus 388.52 ms before, and
139.57 ms on WASM versus 567.98 ms. These operator improvements account for
most of the corresponding whole-model reductions; Conv remains dominant.

WebGPU now classifies an admitted span as a reusable weight only when its
exact base and checked extent identify canonical immutable compiled-model
storage. Classification belongs to the context and its model/backend owner.
Extent changes reclassify and recreate storage; host invalidation forces an
upload; release and teardown retire it. Mutable engines, training weights,
inputs and scratch retain ordinary uploads. This reduces warmed upload traffic
to about 40.046 MB for Piper and 577.485 MB for Kokoro. It caches only model
weights, not computed activations or outputs.

Eligible ungrouped HWIO WebGPU Conv now selects the existing sixteen-output
shader when input channels exceed one and output channels are divisible by
sixteen. Selection is recomputed from the concrete shape. The shader preserves
the scalar per-channel accumulation order and uses no new workspace. It
applies to 113/132 Piper and 85/94 Kokoro Conv nodes. A diagnostic scalar-kernel
ablation on the same candidate artifacts measured Kokoro at 4,782.030 ms
versus 3,795.971 ms with vector Conv, about 20.6% lower request latency.

Independent Piper ablation processes gave inconsistent results because of host
allocation/GC and session effects. A paired retained-session experiment warmed
all pipelines, alternated execution order, applied identical instrumentation
and checked equal waveform hashes. Its medians were 653.539 ms scalar,
544.610 ms vector and 572.309 ms for a proposed 131,072-output threshold.
That threshold did not establish an advantage and was not added. The report
does not assign a separate stable Piper speedup to Conv alone.
Indeed, separate basic traces measured Piper GPU passes at 102.76 → 125.08 ms
(duration 27.39 → 69.03 ms, acoustic 75.37 → 56.05 ms). Its net request gain
comes from lower upload cost, despite that device-time increase. Kokoro basic
passes fell from 1911.20 to 756.62 ms. These are individual diagnostic captures,
not a claim that the vector route improves every convolution shape.

Copying uploads into tightly sized host buffers also failed the full-model
test: Piper rose to 808.739 ms and Kokoro to 7,590.143 ms, so that workaround
was rejected. A SIMD patch to a generic WASM Conv function passed isolated
kernel tests but did not affect the actual release dispatch; it was removed.
These experiments illustrate why whole-request measurements and executed-route
evidence are required in addition to primitive tests.

### Export, fusion and remaining work

Exporter provenance records 67 typed-optimizer changes and 24 exact GELU
patterns for Piper duration, and three layout changes for acoustic. Kokoro
encoder records 31 typed changes and 75 Linear+bias source patterns; decoder
records 41 typed changes and one bias pattern. These are transformation counts,
not speedup estimates or counts of fused GPU dispatches.

The actual CPU/WASM plans skip 185/135 Piper nodes and 74/255 Kokoro nodes,
principally storage-only operations, with no numerical fusion flags. CUDA
records 0/6 fused steps for Piper and 1/23 for Kokoro. WebGPU, Vulkan and OpenGL
plans skip no nodes and record no fused steps for these packages. Thus an
export optimizer and a backend fusion registry do not imply arbitrary fusion
of the hundreds of normalization, spectral and elementwise operations here.

Remaining priorities are WebGPU nonweight span resizing/allocation and uploads,
WASM dense Conv, native CPU Conv and elementwise traffic, and CUDA LSTM
parallelism. Writable shader bindings alone do not prove a complete overwrite;
skipping their initialization globally would be unsafe. Normalization and FFT
fusion need source/shape and rounding proofs, especially for Kokoro's phase
sensitivity. No broader activation cache, lower precision or numerical-tolerance
change was introduced. Metal remains translation-only, without device execution.

### Numerical and memory results

All sixteen final model trials retain their expected result: Piper passes its
declared PCM16 criterion, and Kokoro reports `numerical_difference` with the
original strict tolerance. Every trial's manifest and model artifact hashes
match the reproduced baseline. For both models on CPU, CUDA, Vulkan, OpenGL,
WASM and WebGPU, the saved ordinary waveform before and after is byte-identical
(twelve comparisons). Thus the measured speedups add no waveform difference
to Kokoro's existing oracle failure. The stricter Piper WASM mismatch remains
visible; it was not hidden by changing the waveform criterion.

The immutable-weight fix reduces transfer traffic, not retained device storage:

| Shared WebGPU buffer inventory | Before | Final |
| --- | ---: | ---: |
| Piper live bytes | 75,773,008 | 75,773,008 |
| Piper observed peak bytes | 108,980,904 | 108,980,904 |
| Kokoro live bytes | 368,144,536 | 368,144,536 |
| Kokoro observed peak bytes | 921,710,472 | 921,710,472 |

These are bridge-owned GPUBuffer observations, not total driver VRAM or process
RSS. Both stage collectors observe the same shared bridge; their inventories
must not be added. Ordinary per-request buffer allocation counts and bytes
remain unchanged. Kokoro still creates roughly 568 MB of buffers and uploads
about 577 MB of nonweight data per request; its upload API time remains about
2.73 s. Pipeline creation stays zero after warmup, while bind groups and uniform
buffers are still created per dispatch. Expand and vector Conv add no scratch
workspace. Native/WASM allocator and RSS observations, including collector
overhead and partial coverage, are retained in the profile reports.

### Verification and release size

Native inference/full executables and libraries, both WASM profiles and all
four JS variants built. The final artifact check validated all eight fixed
release files, provenance, ABI and inference/training symbol boundaries.
API conformance checked all 113 RPCs; the public schema was unchanged. Native
ABI/profile checks passed. The public API suite passed 288 tests with two
environment-dependent skips, and bridge tests passed 90 tests. Focused tests
cover exact Expand bits with changed inputs, shape-dependent Conv selection,
immutable-weight classification, explicit invalidation and buffer replacement.

There were **700 passing numerical cases**: 98 native CPU results across
inference/full and 602 native GPU/WASM/WebGPU results. The latter comprise the
49-case TTS corpus (including six new bit-exact Expand cases), 14 LSTM cases,
16 comparison cases and the new 22-case independent Conv corpus. Four additional
grouped Conv cases—two inputs each on Vulkan/OpenGL—return
`NATIVE_STATUS_OPERATOR_FALLBACK_FORBIDDEN`, code 131, identically on preserved
baseline and final binaries. They are existing admission limits, not numerical
passes; the other twenty Conv cases on each of those backends pass. The refusal
receipts are in `reports/qualification-refusals/`. No operator was retried on CPU.

The two pre-existing native gzip gates remain exceeded. Their authoritative
release-check measurements are:

| Artifact | Before gzip bytes | Final gzip bytes | Change | Existing limit |
| --- | ---: | ---: | ---: | ---: |
| `native/volvoxai-lite` | 1,043,971 | 1,044,121 | +150 | 1,031,017 |
| `native/volvoxai` | 2,609,387 | 2,609,599 | +212 | 2,563,074 |

Consequently `check:release` still returns failure for these two size gates.
No budget was raised. The uncompressed native files grew by 448 and 432 bytes,
respectively. Other artifact size gates pass. Artifact manifests retain exact
hashes rather than identifying a build only by package version.

The final remote collection verified 237 sealed files (326,289,221 bytes)
against its hash manifest before deleting the task-owned reports, runtime and
ORT overlay. Cleanup recovered **1,997,836,288 bytes** across its two phases;
`final-summary.json` records both receipts and confirms removal of the task
directory. A local publication-manifest conflict required a verified recovery
into a clean destination before the second cleanup phase. All reports are
preserved locally. Shared weights, the original Python environment and existing
services were preserved; no GPU benchmark ran concurrently with another.

## ONNX Runtime profiling and quantized comparison

This follow-up compares the retained Volvox artifacts directly with ORT 1.23.2
on the same server and prepared requests. It makes no further runtime changes.
Ordinary trials retain the one-first/two-warmup/five-measurement contract and
four CPU threads. CUDA TF32 remains disabled; ORT CUDA explicitly permits CPU
nodes, while every successful Volvox stage attests zero fallback. Loading,
profiling, model authoring and validation stay outside ordinary latency.

Evidence is in `build/tts-ort-comparison-20261029/`. Start with
`comparison-summary.json`, `{piper,kokoro}-profile-comparison.json` and
`published-quant-analysis.json`. `reports/fp32/`, `reports/volvox-profiles/`,
`reports/quantized/` and `reports/nsys/` retain original measurements, untouched
ORT traces, current Volvox traces, waveform snapshots and Nsight SQLite/raw
captures. Runtime/model/input hashes, provider options and request output hashes
are retained. The CPU and CUDA FP32 model artifacts are the previous final
artifacts; quantized candidates have separate files and receipts.

### FP32: the remaining gap is measured kernel and tensor work

Fresh complete-request medians, in milliseconds:

| Model | ORT CPU | Volvox CPU | Ratio | ORT CUDA+CPU | Volvox CUDA | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Piper | 38.655 | 57.319 | 1.48× | 11.025 | 20.893 | 1.90× |
| Kokoro | 657.473 | 1022.418 | 1.56× | 73.733 | 262.838 | 3.56× |

The baseline models are FP32, not INT8: Piper contains 15,650,475 float
initializer elements, and Kokoro 81,151,936. The integer initializers in those
files are shape/index constants. Their ONNX files are 63,201,330 and 325,532,232
bytes. The precision audit is `model-storage-audit.json`.

Actual CUDA device intervals were captured with the installed Nsight Systems,
using five NVTX request ranges after warmup. These are instrumented diagnostic
captures, separate from the latency table:

| Model / engine | Kernel duration sum per request, median ms | Kernel launches per request |
| --- | ---: | ---: |
| Piper ORT | 7.028 | 1794 |
| Piper Volvox | 13.413 | 1766 |
| Kokoro ORT | 47.183 | 4069 |
| Kokoro Volvox | 251.241 | 2745 |

Kernel sums can overlap across streams. Kokoro ORT's device-activity interval
union, including copies/memsets, is 44.071 ms; Volvox's is 251.383 ms. Neither
number is an SM-occupancy estimate. All five requests have complete range
boundaries, with no crossing kernel/copy/memset events excluded. ORT launches
more kernels yet finishes faster, so reducing dispatch count alone does not
explain or resolve this native CUDA gap.

Volvox Kokoro spends 132.833 ms in six LSTM kernels, 74.754 ms in the generic
Conv kernel and 15.237 ms in reductions. Its bidirectional batch-one LSTM
launches two blocks of 256 threads and performs the time loop and compensated
input/recurrent reductions within those blocks. Its LSTM alone exceeds all
ORT device kernels combined. This gives CUDA LSTM parallelism higher priority
than optimizing the small stage handoff.

Indeed, ORT Kokoro transfers about 8.036 MB host-to-device and 6.868 MB
device-to-host per request, versus Volvox's 0.076 MB and 0.232 MB. ORT's mixed
placement includes CPU STFT and Resize. Volvox's measured copy durations are
only about 0.059/0.079 ms in those directions. Large host synchronization or
launch-call spans can wait for earlier device work; they are not independent
overheads to add to kernel time.

The ORT `Node/*_kernel_time` records are host intervals, including CUDA nodes.
They are not used as GPU execution time. ORT's own device profiling requires
explicit CUDA profiling support, as described in its
[profiling documentation](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
The additional Nsight cuDNN/cuBLAS trace emitted no library API spans, so full
ORT LSTM device time cannot be attributed separately from shared GEMM kernels.
The report does not mislabel ORT's approximately 1.6 ms LSTM **host** interval
as its complete GPU cost.

Current CPU traces expose different gaps. Exact source-name matching finds
all 455 operations in Kokoro's 65 centered-normalization cores:

| Kokoro CPU diagnostic region | ORT host ms | Volvox host ms |
| --- | ---: | ---: |
| Centered-normalization cores | 37.559 | 126.844 |
| STFT | 25.176 | 107.126 |
| Six LSTMs | 4.930 | 28.031 |
| Original Conv regions, including Volvox layout operations | 386.800 | 434.904 |

These are diagnostic samples with different instrumentation overheads, not an
additive reconciliation of ordinary latency. The normalization cores expand
455 source operations into 585 Volvox operations, including 130 materialized
broadcasts. The actual CPU ReduceMean/ReduceSum dispatch uses a serial row/column
loop with a double accumulator. Preserving each row's ordered sum while
parallelizing independent rows is a more targeted next experiment than assuming
that all native Conv is serial.

Most native Kokoro Conv cost is already eligible for the AVX2 parallel spatial
kernel; the portable serial SIMD128 dense path applies to WASM. Both engines
benefit from four threads: the clean one-thread Kokoro trials were ORT
1060.909 ms and Volvox 1421.174 ms. Source-route eligibility is not proof of a
particular microkernel's execution, but it rules out the earlier blanket
native-Conv-is-serial explanation.

Both STFT implementations use Bluestein for the 20-point transform, with
64-point forward/inverse FFT work. ORT computes each frame using small temporary
buffers and reuses chirp/twiddle data across frames inside one operator;
Volvox materializes 238 tensor operations over the frame
extent. Those operations declare 441,738,328 summed output bytes; that is a
tensor-extent sum, not measured DRAM traffic or peak memory. This is an execution
and intermediate-storage difference, not a claim
that ORT uses a different FFT length. See the pinned
[ORT DFT/STFT implementation](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/onnxruntime/core/providers/cpu/signal/dft.cc).

ORT optimization is active: Piper CPU has eight FusedConv, twenty FusedMatMul
and 24 Gelu nodes; Kokoro has twelve FastGelu and twelve SkipLayerNormalization
nodes. However, disabling all graph optimizations still yields Piper
38.270/11.291 ms and Kokoro 636.605/74.839 ms on CPU/CUDA respectively.
These separate trials are sensitive to run order and clocks, but the large
engine gap persists without ORT graph fusion. Fusion counts alone were an
insufficient explanation. Kokoro's optimizer-disabled outputs retain their
strict numerical difference and do not replace the accuracy baseline.
The original potentially transfer-overlapped Kokoro
ablation runs are retained separately under `reports/ablation-initial/`; the
table uses the clean reruns.

### INT8: smaller files did not make these requests faster

Quantized candidates execute complete live requests. The following table
compares each with its own engine's FP32 baseline. The schemes and quantized
operator subsets differ; they are not equivalent integer implementations.

| Model / engine / candidate | FP32 CPU → candidate ms | FP32 CUDA → candidate ms | Candidate scope |
| --- | ---: | ---: | --- |
| Piper ORT static INT8 | 38.655 → 41.178 | 11.025 → 12.460 | Acoustic Conv subset; duration FP32 |
| Piper Volvox static W8A8 | 57.319 → 119.944 | 20.893 → 23.120 | Acoustic 63 Conv; duration FP32 |
| Kokoro ORT published INT8 | 657.473 → 1628.333 | 73.733 → 1507.687 | MatMul, Conv, LSTM and embeddings |
| Kokoro Volvox grouped W8/G128/F32 | 1022.418 → 1353.460 | 262.838 → 263.928 | 149 PackedLinear; Conv/LSTM FP32 |
| Kokoro Volvox decoder W8A8 | 1022.418 → 1711.333 | 262.838 → 265.855 | Decoder 71 Linear + 88 Conv; encoder FP32 |

The published Kokoro candidate is
[`model_quantized.onnx` from the same pinned revision](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/1939ad2a8e416c0acfeecc08a694d14ef25f2231/onnx),
with verified SHA-256
`fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478`.
It has 148 MatMulInteger, 87 ConvInteger and six DynamicQuantizeLSTM nodes,
plus four quantized embedding tables. Six ConvTranspose and one final Conv
remain FP32. The unchanged published graph is not directly importable into
Volvox: an actual compiler construction refuses its `com.microsoft` domain.
The separately authored Volvox variants do not claim to run that same graph.

ORT's quantized slowdown is concentrated in ConvInteger. The same 87 source
Conv nodes take 384.759 ms in the FP32 CPU trace and 1325.386 ms in the INT8
CPU trace, about 3.45× longer. All 87 integer convolutions remain on CPU when
CUDA is requested, as do the quantized LSTMs and integer MatMuls. The mixed
session therefore spends about 1.33 s in CPU ConvInteger and also incurs
hundreds of transfer nodes. This is not an RTX 3090 INT8-kernel benchmark.
The pinned integer Conv uses generic im2col plus U8×U8→I32 GEMM, whereas FP32
uses the prepared MLAS Conv path. The
[pinned ConvInteger implementation](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/onnxruntime/core/providers/cpu/quantization/conv_integer.cc)
documents that route; the trace does not separate im2col from GEMM
cost, so neither is assigned an invented individual duration.

Authored Piper ORT QDQ covers sixty acoustic Conv nodes, but optimization
produces only nineteen CPU QLinearConv nodes; many convolutions execute as
float with quantize/dequantize boundaries. Its CUDA profile executes all 129
Conv nodes as float. Quantized weight storage alone therefore does not imply
integer device arithmetic. Volvox grouped W8 likewise keeps FP32 activation
and accumulation arithmetic. Volvox quantized operator-level timing was not
captured before remote cleanup; its CPU regressions are established by complete
request timings, but a specific kernel cause remains unmeasured.

Storage changes are real, even where latency gets worse:

| Candidate | Same engine's FP32 bytes → candidate bytes | Storage scope |
| --- | ---: | --- |
| Piper ORT acoustic static | 63,201,330 → 39,347,921 | Complete ONNX file |
| Piper Volvox acoustic static | 67,858,664 → 41,773,399 | Both stages' weight files |
| Kokoro ORT published INT8 | 325,532,232 → 92,361,116 | Complete ONNX file |
| Kokoro Volvox grouped W8 | 333,710,828 → 296,963,992 | Both stages' weight files |
| Kokoro Volvox decoder static | 333,710,828 → 155,027,940 | Both stages' weight files |

All ten runnable quantized paths have unchanged output lengths for these inputs
and report **numerical_difference**. Their own-backend FP32 RMSE values are:
Piper ORT 0.01454/0.01442, Piper Volvox 0.01369/0.01369, Kokoro published
0.09103/0.08992, Kokoro grouped W8 0.02310/0.02374 and Kokoro decoder W8A8
0.09574/0.09640 (CPU/CUDA). Piper candidates fail the unchanged PCM16 bounds.
Kokoro's additional quantization error is reported separately from its existing
FP32 oracle difference. These waveform errors are not perceptual quality scores.

Full-stage static Piper candidates cannot execute the fixed request: ORT
predicts 56 versus 57 frames; Volvox predicts 59 on CPU and 56 on CUDA versus
57. These are refusals, with no complete-request latency reported. A separate
local full-static Kokoro check similarly predicts 65 versus 66 frames.
These cases require newly prepared packages for the quantized duration before
a different-extent complete-request benchmark would be valid.
No duration is forced, no waveform is aligned or truncated, and no activation
is cached to conceal the mismatch. Static calibration uses the evaluation
request only; none of these authored candidates is qualified for unseen speech.

Memory reports retain process RSS with its scope. Volvox quantized RSS includes
both the candidate and an FP32 reference session loaded together. ORT RSS
snapshots include one session plus profiling/inspection stages; early peak
values use `getrusage` and may inherit a launcher's high-water mark. Later
quantized Volvox receipts use Linux `VmHWM`. These observations must not be
presented as a like-for-like per-model memory comparison or total GPU VRAM.

### Follow-up priorities and verification

The evidence prioritizes CUDA LSTM parallelism and convolution algorithms,
CPU normalization reductions/broadcast materialization, and executing STFT
with small local buffers while preserving numerical order. WebGPU's prior
nonweight uploads remain a separate host-transport problem; this follow-up
does not claim to have benchmarked ORT WebGPU or ORT WASM. Quantized kernel
coverage and actual operator cost must improve before INT8 can be presented
as a speed benefit for these requests.

This follow-up changes example measurement/authoring tools and documentation,
not runtime kernels or the public schema. All eleven retained release/library
artifact hashes match the previous final build. The previous native/WASM build and
size-gate results still apply to the unchanged runtime artifacts. Tests cover
run segmentation, host/device separation, exact source-region matching,
quantized numerical execution, unchanged PCM16 criteria, duration refusals,
storage accounting and Nsight request/process boundaries. All twenty tests,
Python compilation checks and `git diff --check` pass.

The final collection verified **140 files / 244,380,579 bytes** before removing
the task-owned runtime, overlay, transfer files and reports. Cleanup recovered
**2,222,764,032 bytes**. The task directory is absent, shared ORT remains 1.23.2
with its original CPU/Azure providers, and the GPU returned to its prior idle
state. Exact collection and cleanup receipts are beside the summary JSON.

## Runtime optimization after the ORT comparison

This pass implements the measured bottlenecks; it does not change the exported
models, the default FP32 policy, or the two-stage host protocol. Every request
still runs the complete model on its required Volvox backend with current
inputs. ONNX Runtime remains an independent reference. The preserved initial
artifacts and all intermediate/final receipts are under
`build/tts-opt2-20261029/`; `baseline-artifacts.json` and
`verification-local/summary.json` identify the initial and final binaries.
The source revision is `aa778bae52186dd14e03eb178f9f61f0c054aa30` plus the
preserved working changes. No existing work was discarded.

### Same-condition complete requests

The server remains the i3-1115G4 / RTX 3090 machine, with four native CPU/ORT
threads, a separate first request, two warmups and five measured requests.
These medians include both stages, actual intermediate tensor handoff,
synchronization and completed host output. Loading, compilation, phonemization
and WAV saving remain excluded. The output durations remain 0.912 s and 1.65 s.
Each cell is **before → final**, in milliseconds; full profiles are shown.
The inference CPU/WASM profiles are also retained in the 16-trial matrix.

| Model | CPU | CUDA | Vulkan | OpenGL | WASM | WebGPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Piper | 59.210 → 57.850 | 21.445 → 20.601 | 83.831 → 87.746 | 49.793 → 50.823 | 341.156 → 339.288 | 587.921 → 384.237 |
| Kokoro | 1094.279 → 1037.942 | 264.326 → 119.922 | 509.228 → 516.061 | 412.760 → 412.939 | 8329.746 → 8114.521 | 3798.568 → 1021.242 |

Kokoro improves **54.6% on CUDA**, **73.1% on WebGPU**, and **5.1% on CPU**.
Piper WebGPU improves **34.6%**. Small changes on other paths are not evidence
of a general speedup: intermediate Piper CUDA medians were 22.46 ms, and the
first WASM candidate did not improve the complete Kokoro request. The table
also retains slower final Vulkan/OpenGL observations rather than selecting
better runs. FP32 WASM still needs about 8.1 s for Kokoro; its Conv kernels
remain the main bottleneck.

The current CPU reference medians are ORT **39.231 ms / 668.367 ms** for
Piper/Kokoro. The separately retained CUDA comparison phase measured ORT
**11.025 ms / 73.733 ms** with CUDA+CPU placement and TF32 disabled. Those CUDA
reference numbers are not new measurements from this optimization pass.
Kokoro Volvox CUDA remains about **1.63×** that mixed-provider reference.
The final offline comparisons are `piper-final-ort-comparison.json` and
`kokoro-final-ort-comparison.json`, using the exact graph/source provenance
and the retained independent ORT profiles. ORT node host intervals are never
relabeled as GPU kernel time.

### What changed and why

**CPU/WASM reductions.** The large normalization reductions previously
processed all rows serially. Independent rows now occupy SIMD F64 lanes,
retaining each row's original double accumulation order and F32 cast before
scaling. Native execution uses the existing worker pool for large row sets.
Kokoro CPU reduction node time falls **62.100 → 14.585 ms**; isolated complete
request measurements confirm the roughly 5% CPU improvement. WASM reduction
time falls **63.746 → 32.824 ms**, but this is small beside roughly 6.6 s of
Conv work. No scratch or retained values are added.

**CUDA LSTM.** The original batch-one bidirectional kernel launched only two
blocks and serialized the full time/input/recurrent loops. The new tactic
transposes the live weights in device workspace, projects all time positions
in parallel, and executes ordered recurrent/state steps. It carries both the
FP32 sum and Kahan correction across the projection boundary. Actual warmed
GPU time for the six Kokoro LSTMs falls **132.194 → 4.759 ms**: transpose
0.166 ms, input projection 0.490 ms, recurrence 3.042 ms, update 1.061 ms.
Launch count grows from 6 to 310, yet the isolated complete request improves
**264.326 → 134.548 ms**. This is evidence for insufficient parallelism as the
old bottleneck, rather than a claim that fewer dispatches are always faster.

**CUDA Conv1D.** Large regular temporal convolutions now reuse each weight
across four spatial outputs per thread while preserving each output's
ic→kx accumulation and skipped-padding behavior. The tactic covers 51 Kokoro
nodes and 14 Piper nodes in these packages. Kokoro Conv GPU time falls
**74.706 → 62.007 ms**, giving a further complete-request improvement
**134.548 → 119.716 ms** in the isolated r3 comparison. Piper's selected kernels
improve modestly, but r1/r3 whole-request medians are effectively unchanged.
There is no new allocation or activation cache.

**WebGPU output residency.** A planner that proves complete word stores can
reserve an output buffer without uploading its obsolete host bytes. Writable
binding metadata alone is insufficient: partial writes, packed-byte outputs
and training retain ordinary residency. Reservation does not mark content
valid; successful dispatch does. Immutable weights still follow their existing
model ownership, host invalidation and changed-extent rules. The private C/JS
bridge contract is generated from ABI version 14; the public proto is unchanged.

In paired r1 untraced diagnostic requests, Piper uploads fall **40.05 → 5.58 MB**, and
Kokoro **577.48 → 46.27 MB**. Host `writeBuffer` time falls
**214.48 → 50.68 ms** and **2735.05 → 253.21 ms**, respectively. Kokoro submits
fall from 42 to 8 because fewer staging-budget flushes are needed. Buffer and
bind-group creation counts are unchanged. There is still no warm pipeline or
shader compilation. In the paired r1 traces, complete BASIC pass intervals fall about
133.49 → 118.34 ms for Piper and 741.97 → 452.64 ms for Kokoro; these intervals
include idle gaps and are not pure arithmetic time.

This explains both the former multi-second WebGPU behavior and the remaining
Piper gap: Piper WebGPU still takes longer than WASM. It still creates 1700
parameter buffers and 2092 bind groups per request, executes many small GPU
programs, and pays asynchronous completion/readback waits. Submission calls
alone are a small fraction of the request. Host write, submission and wait
spans can overlap device work and must not be added as disjoint costs.

### Quantized complete requests

Quantized execution is measured separately, with the same original inputs and
unchanged quantized model hashes. These candidates keep their previous
`numerical_difference` status; the default model remains FP32. Every cell is
**before → final**, in milliseconds, with the same first/warmup/repeat policy.

| Volvox candidate | CPU | CUDA | WASM |
| --- | ---: | ---: | ---: |
| Piper acoustic W8A8 | 121.605 → 55.623 | 25.220 → 25.022 | 1805.344 → 211.102 |
| Kokoro grouped W8/G128/F32 | 1412.098 → 1086.484 | 262.035 → 120.143 | 8936.317 → 8102.707 |
| Kokoro decoder W8A8 | 1708.156 → 1058.776 | 264.496 → 139.028 | 42181.051 → 3537.792 |

**QConv coverage, rather than conversion overhead, was the main INT8 issue.**
The existing packed im2col→QLinear route was restricted to 3×3 despite its
underlying patch builder supporting temporal geometry. Dense 1×K now uses
that route, with the specialized 3×3 worker still explicitly guarded. Padding
zero codes, integer sums, affine parameters, overflow guards and requantization
are preserved. Piper CPU QConv time falls **86.095 → 21.749 ms** in r2, and
Kokoro decoder QConv **1014.672 → 397.992 ms**. This yields **54.3% / 38.0%**
final CPU improvements and **88.3% / 91.6%** WASM improvements. The very slow
42.18 s Kokoro WASM baseline is a completed five-sample measurement, not a
projection or a timeout. The final quantized request is 3.54 s, still above
real time for its 1.65 s output.

**Grouped W8/F32.** Independent output columns are vectorized without changing
ascending-K accumulation. Four finite complete input rows then share each
weight decode and worker submission. Partial/nonfinite cases retain the direct
path, and input/output aliasing excludes the direct multirow reader. No full
dequantized weight bank or activation cache is retained. CPU PackedLinear
node time falls **348.449 → 177.120 → 71.777 ms**. The four-row step alone
improves the complete request **1199.572 → 1086.484 ms**; total improvement is
23.1%. CUDA quantized gains come primarily from the shared LSTM/Conv work,
not a newly introduced integer CUDA kernel.

The prior ORT quantization comparisons remain applicable to their own recorded
models: Piper acoustic INT8 CPU/CUDA+CPU 41.178/12.460 ms, and published Kokoro
INT8 1628.333/1507.687 ms. Published Kokoro uses ConvInteger, MatMulInteger and
DynamicQuantizeLSTM, unlike these Volvox variants; its 87 ConvInteger nodes
run on CPU even with CUDA enabled. These are different quantization policies,
so the table above does not claim a same-graph runtime victory over ORT INT8.
The original strict numerical errors and incompatible-duration refusals are
retained. One-request calibration and waveform comparisons do not establish
speech quality on unseen utterances.

### Memory, remaining work and rejected candidates

CUDA LSTM uses at most **8,431,616 B** inside the already admitted **32 MiB**
context workspace; it does not grow device allocation. Reduction and the
PackedLinear rewrite add no tensor-sized allocation. WebGPU's observed shared
bridge peaks remain **108,980,904 B** for Piper and **921,710,472 B** for Kokoro.
Skipping uploads changes neither allocation counts nor this scoped inventory;
these values are not total GPU VRAM.

QConv adds immutable packed weights under the existing compiled-model lifetime
and rebuilds the patch matrix from current input on every invocation. The
sampled K11/C128/O128 temporal node uses **361,008 B** of packed weights and
**11,152,768 B** of temporary patches. A K3/C256/O256 node uses **394,288 B**
of packed weights and **1,013,760 B** of patches. Patch allocation is capped at **128 MiB**;
larger geometry retains the existing same-backend kernel. Resident packed
weights and transient patch bounds are now both accounted for. These are
per-node costs; measured process RSS includes its recorded sessions and
profiling overhead and is not a per-model or GPU-memory measurement.

Native CPU quantized benchmark process peak RSS changes as follows. Each
process holds both FP32 and candidate sessions and includes setup, host outputs
and diagnostics; the complete RSS delta cannot be attributed solely to packed
weights. These measured increases are a cost of the faster QConv route.

| Candidate | Before peak RSS (B) | Final peak RSS (B) | Change (B) |
| --- | ---: | ---: | ---: |
| Piper acoustic W8A8 | 344,952,832 | 365,895,680 | +20,942,848 |
| Kokoro decoder W8A8 | 1,563,615,232 | 1,682,620,416 | +119,005,184 |
| Kokoro grouped W8/G128/F32 | 1,665,155,072 | 1,665,396,736 | +241,664 |

The same graph export/fusion decisions described above remain in force; this
pass changes execution kernels and residency, not model lowering. CPU still
materializes normalization broadcasts and executes the STFT lowering across
many tensor operations. In the final exact-source CPU profile, normalization
core work takes **93.140 ms** versus retained ORT **37.559 ms**, and STFT
**105.904 ms** versus **25.176 ms**. CUDA still spends about 62 ms in Conv and
roughly 15 ms in long-row reductions; the latter expose few independent rows.
ORT's complete GPU kernel time was about 47 ms in the independent Nsight
comparison, so the remaining Conv cost alone is material.

WASM remains dominated by strict single-thread SIMD128 Conv execution. Alternate
pixel/channel tiles did not establish a gain. A live weight-repacking proposal
looked promising on the local CPU but failed target-machine testing:
K11/dilation1 **108.859 → 109.350 ms**, dilation5 **110.562 → 109.660 ms**;
K3 regressed **20.181 → 21.256 ms**. Those candidates were discarded.
`reports/micro/wasm-conv-packing.json` retains the numerical and timing evidence.

WebGPU/Vulkan/OpenGL node-level traces have explicit interval/event limits.
Partial device-program sums are not presented as complete operator GPU costs.
Complete BASIC pass timings, ordinary request timings and the independent
CUDA Nsight captures remain separate evidence. Metal still has shader
translation coverage only, with no physical-device qualification.

An additional Nsight Vulkan attempt completed the Deno requests with unchanged
waveform hashes, but did not produce a usable timeline: its NVTX capture range
did not start and no `.nsys-rep` was emitted. The retained application/log files
under `reports/nsys/final-kokoro-webgpu.*` document the failed capture; missing
GPU intervals are not interpreted as zero time.

### Numerical, build and release validation

All 12 full-profile FP32 and nine quantized final waveforms are **bit-identical
to their respective same-model, same-backend pre-change results**, with zero
added maximum error and RMSE. This is not cross-backend or ORT equality. Piper
FP32 still passes its declared PCM16 error criterion. Kokoro FP32 and all
quantized candidates retain their original `numerical_difference` status;
quantized Piper does not meet the FP32 PCM16 criterion. All 16 FP32 execution
trials and nine quantized trials run their complete models with zero fallback,
unchanged model/input hashes, and current-input execution on every request.

The final physical-backend qualification passes **804 shared fixture runs**,
plus **66 live CUDA requests** covering the LSTM and Conv rewrites against both
the previous kernels and ORT. Four grouped-Conv Vulkan/OpenGL fixture refusals
remain exactly as before (`OPERATOR_FALLBACK_FORBIDDEN`, code 131); they are
reported separately and are not counted as passing executions. Local checks
pass 63 Python tests, 78 CTest targets and 396 Node tests, with their recorded
unsupported/optional skips. The 62 native invariant targets pass, including
the targeted rerun after correcting a stale test-only CUDA mock. API
conformance, generated ABI, strict CUDA compilation, and both WASM call-sequence
profiles pass. Both native and WASM inference/full builds complete, and all
eight fixed release artifacts pass provenance, ABI and symbol validation.

The release check still **fails on size**. The two preexisting native gzip
overruns remain, and this pass adds two inference WASM/browser-pair overruns.
No budget was raised. GNU `gzip -9n` measurements below compare the preserved
starting artifacts with the final candidate; gate baselines are separate from
this immediate before/after comparison.

| Artifact or pair | Before gzip (B) | Final gzip (B) | Gate limit (B) | Result |
| --- | ---: | ---: | ---: | --- |
| Native inference | 1,044,121 | 1,048,260 | 1,031,017 | Existing failure |
| Native full | 2,609,599 | 2,617,612 | 2,563,074 | Existing failure |
| Inference WASM | 531,109 | 533,446 | 531,457 | New failure: 1,989 B over |
| Inference minified JS + WASM | 572,014 | 574,357 | 572,057 | New failure: 2,300 B over |
| Full WASM | 782,066 | 784,031 | 795,463 | Pass |

Exact commands, failed and corrected development receipts, artifact SHA-256
values and gate output are retained in `verification-local/summary.json`,
`verification-local/size/release-size.json` and
`immediate-size-comparison.json`. Functional qualification does not imply a
passing release gate.

The sealed remote collection contains **362 files / 576,112,782 B**; every
file hash was verified locally, then independently checked again before handoff.
Cleanup removed the task-owned directory and recovered **1,939,087,360 B**.
No ORT overlay was created in this pass, and shared weights, the Python
environment and services were preserved. `collection.json`, `cleanup.json`,
`pre-cleanup-state.json` and `root-final-audit.json` retain the receipts beside
`final-summary.json` under `build/tts-opt2-20261029/`.

## WebGPU parameter buffer reuse

This section records the earlier exact-size buffer pool. The later chunk policy
and its separate measurements are described in
[parameter upload batching](#webgpu-parameter-upload-batching-and-resource-policy).

This pass reduces repeated host work for complete FP32 Piper requests. Each
dispatch still uploads its current parameters, but completed uniform buffers
can be reused by a later submission. Buffers remain exclusive while their
submission or validation scopes are pending. An exact-size pool retains at
most 4,096 buffers and 1 MiB of idle payload. Failed submissions, device loss
and owner close destroy their storage. General tensor retirement remains
separate; this pool stores no model activations or waveform results.

The same i3-1115G4 / RTX 3090 with driver 535.309.01 was used. Each A/B trial
starts a separate process and executes both live stages through completed host
outputs. Loading, compilation, phonemization and file writes are excluded.
Blocks alternate baseline/candidate order as AB, BA, AB. Candidate artifacts
are the final release bytes; two early report directory names contain
`provisional`, but all six staged hashes were verified identical to the final
files. ORT and native comparisons were not remeasured in this pass.

| Request and measurement policy | Baseline median (ms) | Final median (ms) | Reduction |
| --- | ---: | ---: | ---: |
| Piper, 2 warmups + 5 measurements per block, 15 total samples per variant | 319.248 | 292.863 | 8.3% |
| Piper, 10 warmups + 15 measurements per block, 45 total samples per variant | 405.621 | 286.169 | 29.4% |
| Kokoro, 2 warmups + 5 measurements per block, 15 total samples per variant | 1025.804 | 985.900 | 3.9% |

Each aggregate median uses all measurement samples, excluding the separate
first request and warmups. Piper's short-run block medians were
317.178→288.718, 319.248→342.866 and 377.401→292.863 ms: the second block
regressed. Requests sometimes acquire roughly 90 ms of extra latency, whose
cause is not established here. Longer measurement reduces the influence of
the transition after startup without eliminating variability. Its three block
medians were 409.711→287.169, 408.248→284.015 and 391.385→307.662 ms, all
improvements. The 45-sample ranges were 304.417–464.663 ms before and
271.659–378.911 ms after. Piper RTF is 0.350→0.321 under the short policy and
0.445→0.314 under the longer policy. These policies and independent earlier
baselines must not be mixed into one speedup claim.

Complete warmed host-API observations establish the intended mechanism:

| Per-request API observation | Piper before → final | Kokoro before → final |
| --- | ---: | ---: |
| Parameter buffer creations | 1700 → 0 | 2825 → 0 |
| Parameter writes | 1700 → 1700 | 2825 → 2825 |
| Bind group creations | 2092 → 2092 | 3098 → 3098 |

Piper's parameter payload remains 62,160 B and total upload remains 5.582 MB.
One baseline host diagnostic spent 33.074 ms creating parameter buffers;
the final observation eliminates those creations. Current parameter writes,
bind group creation and asynchronous completion remain costs. Their intervals
can overlap GPU work and do not add up to a latency explanation. Detailed
device traces remain truncated and are not used as complete kernel-cost sums.

Piper's observed shared bridge live inventory grows by 48,336 B, with peak
108,980,904→109,005,352 B. Kokoro live inventory grows by 107,232 B, with peak
921,710,472→921,710,568 B. These scoped GPU-buffer payload inventories exclude
driver object overhead and are not total VRAM or process RSS. The full JS grows
by 1,975 raw / 365 GNU gzip bytes; minified full JS by 731 raw / 211 GNU gzip
bytes. Both WASM files and all inference browser artifacts are byte-identical
to baseline. No model, input or quantization policy was changed.

Piper retains its declared PCM16 criterion. Acoustic-noise A→B→A requests use
live inputs and preserve the prepared duration; B has an independently computed
ORT reference. Both input variants pass, and baseline/final waveforms are
bit-identical for both models and the changed input. Kokoro retains its existing
`numerical_difference`; this change adds no waveform error. Every measured
stage attests zero fallback. The training smoke fixture was updated to declare
the required stateless execution contract; its original missing-contract
refusal also occurred with the baseline engine. Physical minified-bundle
training checks cover changing batch sizes, optimizers, finite-difference
updates, rollback, failure recovery, cancellation and dropout.

Native inference/full CLI and shared/static libraries, both JS/WASM profiles,
API conformance and 62 native invariants pass. WebGPU contracts pass 101 tests,
including ten new parameter ownership, live-value, extent, pool-bound and
failure/loss tests. The broader proto API run passes 299 tests with two skips;
its overlapping tests are not added to that count. The fixed eight artifacts
pass provenance, ABI and symbol validation. Release size still fails the same
four budgets; no budget was changed and no new size failure was introduced.

Evidence is retained under `build/tts-webgpu-params-20261029/`:
`final-summary.json`, `remote-results-summary.json`, `artifact-comparison.json`,
`reports/`, `verification/` and the collection/cleanup receipts. The next host
cost to investigate is bind-group creation. Arena and bind-group reuse remain
separate, unimplemented candidates requiring resource-identity and lifetime
proofs before another A/B experiment.

## Bind-group reuse investigation

The next pass investigated Piper's 2,092 bind-group creations per request.
An exact binding key includes the pipeline/layout and every buffer identity,
slot, offset and size, including uniform storage. Pipeline identity normalizes
the new layout wrappers returned by `getBindGroupLayout(0)`; wrapper identity
alone is not a useful cache key. Six complete request observations found no
within-request exact duplicates. A cache invalidated on buffer destruction had
only 0–3 eligible hits per Piper request and 0–44 per Kokoro request, out of
2,092 and 3,098 groups. This diagnostic measures eligibility, not an implemented
cache or an ordinary benchmark. Key construction and invalidation also have a
host cost. No general bind-group cache was adopted.

A smaller candidate retained the original explicit bind-group layout beside
each pipeline instead of querying a new wrapper at every dispatch. It removed
2,092 Piper / 3,098 Kokoro getter calls per warmed request while preserving all
bind-group creations and live parameter writes. The direct getter host spans
were only about 1.5 / 2.4 ms, respectively. Both models used 29 original layouts.

Fresh complete-request A/B trials used the same target, inputs and measurement
scope as the preceding pass, with separate processes and three alternating
AB/BA/AB blocks. These are results for the **rejected candidate**, not a new
runtime baseline:

| Measurement policy | Baseline → candidate median (ms) | Baseline → candidate mean (ms) |
| --- | ---: | ---: |
| Piper, 2 warmups / 5 measurements per block, 15 samples per variant | 287.209 → 259.565 | 293.036 → 269.713 |
| Piper, 10 warmups / 15 measurements per block, 45 samples per variant | 283.685 → 280.769 | 300.851 → 304.230 |
| Kokoro, 2 warmups / 5 measurements per block, 15 samples per variant | 985.826 → 983.940 | 986.094 → 985.207 |

The apparent 9.63% short-policy Piper median reduction depends on the timing
of the roughly 90 ms latency clusters. Longer trials gave 1.03% lower median
but 1.12% higher mean, with overlapping distributions. Their block medians were
282.993→279.391, 283.685→279.511 and 284.692→281.195 ms. This does not establish
a dependable whole-request gain. The layout-only runtime and its three new
tests were removed; the preceding parameter-pool implementation, its tests and
all six browser release files were restored byte-for-byte. Candidate source,
patches, artifacts and passing numerical tests remain in local evidence.

A final import-level diagnostic on the retained runtime identified the actual
buffer churn. All 5,502 ensures succeeded. All **655 tensor buffer creations /
39,275,212 B** came from changing the byte extent of an existing host pointer:
457 in duration and 198 in acoustic. These were not first-seen spans or explicit
release/recreate events. The diagnostic waveform remains bit-identical and
passes PCM16. Host memory can reuse an address for different tensor extents;
each replacement device buffer changes the bind-group key.

This observation does not prove that retaining capacity is safe. Earlier
commands may still consume the old storage, and queue uploads must not
overwrite those inputs before execution. Any physical storage pool needs a
bounded lifetime, fresh residency state, current inputs, actual GPU completion
and numerical tests. It must preserve C's logical span identity. A separate
next candidate is batching the 1,700 parameter uploads: the retained pipeline
still writes 62,160 parameter bytes in roughly 20 ms of host API spans. Its
offset alignment, live-byte staging and submission ordering remain unimplemented
work requiring a separate A/B experiment.

The rejected candidate passes the unchanged PCM16 criterion, A→B→A live-input
checks, bitwise comparison to both prior model waveforms and physical minified
training/profiling. Kokoro keeps its existing oracle difference; all stages
remain zero-fallback. GPU-buffer live/peak inventories are unchanged. Native
inference/full CLI/libraries and JS/WASM inference/full builds pass. The retained
runtime passes 101 WebGPU contracts and all eight release artifact provenance,
ABI and symbol checks; the same four size budgets still fail. Native libraries
match the earlier 62-invariant-qualified hashes, with eight targeted native
contracts rechecked here.

Evidence is under `build/tts-webgpu-bindgroups-20261029/`: `final-summary.json`,
`remote-results-summary.json`, the exact-key and span-churn diagnostic scripts,
candidate patches/artifacts, `reports/` and `verification/retained-*`. The sealed
collection contains 62 files / 75,331,280 B, all hash verified. Remote task-only
cleanup recovered 498,741,248 B and preserved shared assets and services.

## Kokoro waveform, threading and size recheck

This follow-up to `29ade3b` fixes reproducible threading, SIMD and STFT
conversion errors. **Kokoro still fails the original waveform criterion and
six release size gates still fail.** Passing operator tests does not establish
full-model quality. Evidence for this pass is under
`build/review-29ade3b-deep/`; earlier measurements above remain historical.

### Waveform conversion and error amplification

The prepared STFT graph previously calculated its roots and chirp with Python
double-precision sine/cosine, then rounded to F32. The source CPU Bluestein DFT
uses scalar F32 `sinf`/`cosf`. These evaluations can differ by one ULP even with
the same F32 argument. NumPy's vectorized F32 sine is also a different evaluator;
it disagreed with the scalar source on 14 of the 84 sine coefficients tested.
Preparation now calls scalar C99 F32 functions, with explicit argument/result
types and an error if unavailable. It neither invokes ORT to obtain constants
nor silently substitutes double-precision math. Host libm implementations can
still differ in their last bit across platforms.

Regenerate prepared decoder packages to obtain the corrected constants.
Updating the runtime cannot repair constants already stored in an exported
graph. On the measured short request, with identical decoder inputs, the
corrected lowered graph and original ONNX now produce **bit-identical ORT CPU
waveforms**; the previous lowering alone had RMSE `1.87403644e-5` and maximum
error `2.15440989e-4`. Independent coefficient and near-zero STFT tests cover
the conversion fix. This is a conversion result for the tested inputs and
provider, not a cross-platform bitwise guarantee.

Native decoder differences remain. A controlled experiment held the live
encoder outputs fixed and used the public debug tensor override to replace
one intermediate with its same-input ORT value. The unmodified debug run was
bit-identical to ordinary native execution. The overrides are diagnostic only;
production synthesis does not inject ORT activations.

| Decoder experiment | Waveform RMSE vs same-input original ORT | Samples failing `abs(error) <= 1e-5 + 1e-4*abs(reference)` |
| --- | ---: | ---: |
| Ordinary native control | 0.00214295454 | 13,713 / 33,600 |
| Replace source-network Tanh input (`v700`) | 0.00211484217 | 13,234 / 33,600 |
| Replace Tanh output (`v703`) | 0.00000109984 | 0 / 33,600 |
| Replace STFT input (`v730`) | 0.00000109984 | 0 / 33,600 |
| Replace phase (`v942`) | 0.00002276830 | 3,163 / 33,600 |

Replacing the STFT input makes all 147,862 measured STFT components
bit-identical. Thus the downstream decoder can meet the unchanged criterion
when supplied the same source signal. Tiny differences in that signal are
amplified by phase extraction near zero complex magnitude and by the network
that consumes the phase. Phase values equivalent modulo `2*pi` are not
necessarily interchangeable when raw phase is an input to learned layers.
Replacing branch-wrap differences alone did not remove the discrepancy.

This does **not** establish that native Tanh is mathematically less accurate.
For the isolated identical-input Tanh sample, its RMSE against double Tanh
rounded to F32 was `2.89927e-9`, versus ORT's `5.03780e-9`; both maximum errors
were `2.98023e-8`. Native and ORT differed by at most `4.47035e-8`. Substituting
double Tanh in a separate diagnostic also failed to repair the waveform.
Copying one provider's approximation into the generic kernel, clamping phase
or weakening the acceptance threshold is not a demonstrated fix.

The ordinary complete preparation/synthesis command was also run, producing
fresh encoder/decoder packages, `volvox.wav`, `reference.wav` and its validation
report in `prepared-kokoro-short/`. It exits with validation failure, rather
than declaring the package numerically qualified. With the final CPU library,
the 1.4-second request has maximum error **0.105982**, RMSE **0.00599464**,
SNR **20.46 dB**, and **14,047 / 33,600** strict failures against fresh CPU ORT.
Before the coefficient correction, the prior baseline had RMSE `0.00602089`.
The corrected PCM16 comparison has maximum 3,473 LSB and RMS 196.43 LSB error.
This is larger than a harmless last-bit difference in the final waveform.
No listening study or perceptual-quality claim follows from these numbers.

On RTX 3090, the corrected graph's ordinary complete request still fails
against ORT CUDA+CPU (`use_tf32=0`): RMSE **0.00261977**, maximum **0.0478242**,
13,461 strict failures. Both Volvox stages require CUDA and report zero
fallback. This model diagnostic used library `d71a2497…`; the final library
`0473a2d2…` additionally fixes CPU integer chunk arithmetic. CUDA code is
unchanged, and the final CLI `98007389…` separately passes all 90 public CUDA
corpus cases. These observations do not turn the earlier latency win into a
quality-qualified Kokoro result.

### Thread ownership and partition-independent arithmetic

The original commit's unbound kernel calls could create a process-global pool.
Nested callbacks could also observe different thread counts on the submitting
thread and workers. The corrected implementation uses only the context-owned
bound pool. Unbound calls execute serially without pool allocation; every
nested callback reports one thread and executes nested work inline. Shutdown
and reconfiguration act on the bound owner and cannot restart another owner's
pool. Tests exercise concurrent submissions, separate owners, nested work,
draining, restart and destruction.

Automatic CPU-count resolution now belongs to each pool and respects the
resolving thread's allowed CPU mask. Auto reconfiguration resolves it again;
a global topology cache cannot carry one owner's affinity into another.
Linux sibling lists with ranges are parsed correctly even when only later
siblings are allowed. ARM keeps its existing automatic cap of two, bounded
by the allowed CPUs. This is not a claim that two is optimal on every ARM SoC.

Sin/Cos previously switched an entire vector to scalar evaluation when one
lane needed a large-range fallback. Several unary tails also used different
arithmetic from full vectors. Consequently thread partitioning or an unrelated
neighboring value changed a result. Out-of-range fallback is now lane-local,
and AVX2/NEON/WASM tails use the same vector evaluation as full blocks. Tests
cover 1/4 threads, irregular partitions, permutations, singleton/full/tail
positions, in-place operation, signed zero and nonfinite inputs. The measured
Piper and Kokoro pipelines now produce bit-identical 1/4-thread outputs on both
x86 and the newly attached ARM device. Cross-ISA identity is not promised.

Large valid extents exposed another defect: `(n + chunks - 1) / chunks`
overflows signed `int` near `INT_MAX`. Positive-extent MatMul, Transpose,
activation, Conv and runtime partition branches now use `1 + (n - 1) / chunks`.
The original MatMul source reproduces the overflow under UBSan. Regression
tests execute the actual partition branches with a recording dispatcher at
`INT_MAX` boundaries, without allocating enormous tensors.

Host ASan/UBSan/leak checks and physical ARM lifecycle tests pass. Host TSAN
cannot start because of an unexpected memory mapping; this is **not** a TSAN
race-detector pass. Public API, inference/training boundaries and generated
dispatch remain intact. Core kernels are shared by other models, so focused
operator, training, PTQ and runtime tests are included; these tests do not
prove every untested model or input regression-free.

### Why native full grew by substantially more than 10 kB

Three isolated builds used the same pinned compiler, compile/link options,
object lists, fixed build date and normalized Git metadata. Sizes below use
deterministic `gzip -9 -n`; the intermediate dirty snapshot is the start of
this follow-up, not the final source.

| Native full snapshot | Raw bytes | gzip bytes | gzip change from previous row |
| --- | ---: | ---: | ---: |
| Parent `bda04f5` | 4,872,272 | 2,572,862 | — |
| Original `29ade3b` | 5,047,168 | 2,674,846 | +101,984 |
| Prior corrected snapshot (`97fdb2…` library) | 5,055,520 | 2,682,504 | +7,658 |

The original commit adds **174,896 raw bytes / 101,984 gzip bytes**. Its full
ELF growth includes 114,080 B of code, 39,456 B of read-only data, 9,144 B of
dynamic relocations, 6,784 B of unwind data/headers and 5,488 B of relocatable
read-only data, minus 56 B of other layout differences. It is a broad operator
and backend addition, not just a small thread-pool change. Input-object growth
includes CPU kernels 77,352 B, runtime 21,391 B, CUDA PTX 22,916 B, compiled
graphics shader packs 15,736 B, GPU host code 17,317 B and training/PTQ 4,146 B.
These input-object categories are diagnostic, not separate gzip contributions.

Backend-disabled rebuilds further isolate the cause. Removing CUDA changes
the original commit's gzip delta by 29,864 B; Vulkan by 8,812 B; OpenGL by
11,463 B. Removing all three changes it by 48,224 B, leaving **53,760 B** of
growth in the common/CPU build. These conditional differences cannot be added:
whole-file compression and shared code are not additive. The embedded CUDA
training PTX is byte-identical across all three snapshots. The graphics pack
contains compiled SPIR-V/GLSL/GLES artifacts. Neither a missed strip step nor
training translation units leaking into lite explains the increase.

The AVX2 spatial/output-channel Conv paths now share their six-pixel inner
kernel. This saves **2,944 raw / 1,441 gzip bytes** in the isolated full build,
while preserving operation order, tails, finite-weight guards and signed-zero
repair. All 96 pinned-compiler before/after numerical cases are bit-identical.
Its small microbenchmark changes were mixed; no speedup is claimed.

The final production full CLI is **5,061,856 raw / 2,683,568 gzip bytes**.
Its gzip budget is 2,563,074 B: **120,494 B (4.70%) over budget**. Budget excess
is different from commit growth; even the parent is already 9,788 B over this
budget. The final production artifact also has normal build metadata, so its
gzip difference from a normalized snapshot is not a pure code attribution.
The Conv saving does not offset all correctness fixes in this pass. The six
unchanged failing gates are:

| Gate | Actual bytes | Budget bytes | Excess bytes |
| --- | ---: | ---: | ---: |
| lite WASM gzip | 540,765 | 531,457 | 9,308 |
| lite native raw | 2,401,800 | 2,382,727 | 19,073 |
| lite native gzip | 1,085,745 | 1,031,017 | 54,728 |
| full native gzip | 2,683,568 | 2,563,074 | 120,494 |
| inference web pair gzip | 581,673 | 572,057 | 9,616 |
| full web pair gzip | 932,332 | 927,195 | 5,137 |

All eight fixed release filenames, provenance, WASM ABI and native profile
boundaries pass. The release checker nevertheless exits with failure because
of these size budgets. No budget was raised and no required backend was removed
from the production build to conceal the regression.

### Final physical-device and regression checks

The connected Android device changed during this pass. Final validation uses
Galaxy A56 / Android 16, separately from the earlier Android 11 device above.
Its NEON, dotprod and i8mm paths all executed successfully: raw ISA parity and
packed GEMM under each explicit ISA clamp, the packed GEMM contract and QConv4
contract (eight executable invocations, no ISA skips). Both Android release
profiles pass the 90-case public corpus with CPU route attestation and zero
fallback. Default-contraction and contraction-off unary probes each pass
2,904,282 checks; the pool lifecycle/affinity executable also passes. A source
comparison verifies that all 599 corresponding native/runtime/proto files in
the Android build snapshot match the final working tree.

The same-device full-model check uses original ONNX with ORT 1.23.2 CPU and
the freshly prepared native decoder, preserving original inputs and live
encoder outputs. Piper passes its declared PCM16 criterion: maximum error
`8.60915e-6`, RMSE `8.70487e-7`. Kokoro fails: maximum **0.0775548**, RMSE
**0.00434378**, 13,866 / 33,600 strict failures. Both models are bit-identical
across 1, 2 and 4 native threads. A56 ORT itself differs from saved x86 ORT on
Kokoro (RMSE `0.00763598`), which is why the native comparison uses the oracle
on the same device. Cross-provider variation is evidence of sensitivity, not
permission to relabel a native failure as success.

The following A56 timings are **single-trial diagnostics**, each with one
warmup and three measurements per stage/provider. Native is the sum of stage
medians around resident public Execute calls; ORT is a complete Session.Run.
Native excludes interstage host transfer/re-encoding, model setup and output
file I/O, so these are not equivalent end-to-end timings. The 2-thread trial
was performed later. No alternating A/B qualification or speedup claim is
made, and these results must not be combined with the earlier device's table.

| Model / threads | Native prepared stage median sum (ms) | ORT Session.Run median (ms) | Numerical result |
| --- | ---: | ---: | --- |
| Piper / 1 | 115.38 | 125.92 | PASS |
| Piper / 2 | 129.53 | 78.84 | PASS |
| Piper / 4 | 136.72 | 61.53 | PASS |
| Kokoro / 1 | 2,244.38 | 2,095.09 | FAIL |
| Kokoro / 2 | 1,901.14 | 1,254.92 | FAIL |
| Kokoro / 4 | 1,476.33 | 784.76 | FAIL |

Threading correctness therefore does not imply an optimal thread-count or
grain policy. In the extra untimed A56 4-thread profile, Kokoro decoder Conv
accounts for 844 ms of 1,396 ms of operator time. Mul/Add/Transpose account for
another 336 ms and Sin for 65 ms. Its encoder Linear/LSTM dominate at about
47/16 ms. Piper's small duration stage is slower with four native threads in
this diagnostic. These identify Conv tiling, memory traffic and small-task
dispatch thresholds as candidates for a controlled A/B experiment; no new
device-wide thread policy was inferred from three samples. The earlier CUDA
ordered-reduction optimization retains its measured result and numerical
scope; this pass introduces no reassociated reduction or unqualified ARM
speedup.

Final native and JS/WASM inference/full builds pass. The final native CLI and
actual release WASM each pass 90 public numerical cases per profile; CUDA
passes 90 on physical hardware. Native invariant/ISA/static checks report
78 passes and four unsupported x86 ISA skips out of 82; JS/API reports 301
passes and two skips; WASM training and PTQ report 7 and 48 passes; Python
training/threading/dynamic-transpose/PTQ reports 28 passes. Focused numerical,
geometry and benchmark tests passed earlier in this pass, with the final
integer-grain regression rerun after the last source change. Counts overlap
in coverage and are not a count of independently validated models.

Detailed receipts include `size/comparison-summary.json`,
`size/backend-ablation.json`, `waveform/final-injection/`,
`waveform-final-comparison.json`, `fixed-kokoro-report.json`,
`final-cpu-pipeline-parity.json`, `android/a56-final/`,
`android/a56-2threads/`, `threading/a56-final/`, `gpu/1bdaf165/`,
and `final-release-docker.log`. The final CUDA corpus receipt is
`build/review-29ade3b-hardware/cuda/corpus-c921ba6b/`.
GPU reports were collected and hash-verified locally before removing the
task directories (1,161,658,368 B and 14,159,872 B recovered for the model
and final corpus tasks). Android task directories and transferred model/ORT
assets were likewise removed; local cleanup receipts record their sizes and
before/after free space. Shared assets and unrelated tasks were preserved.
`final-summary.json` indexes the final artifacts and the remaining failures.

## Thread dispatch optimization and source-phase conditioning follow-up

The next pass accepts the user's requested size-budget increase and continues
thread and waveform analysis. Evidence is in `build/review-29ade3b-threads2/`.
The former six failing byte limits were raised to the measured artifact/pair
sizes. The existing margin formula was inverted to obtain its gate inputs:
these adjusted baseline numbers are not themselves measured artifact sizes.
Each adoption records the actual bytes, effective limit and artifact/report
hashes. Exact WASM ABI checks and all other gates remain enforced; no extra
headroom was added. Size acceptance does not change Kokoro's numerical status.

### Scheduling changes and measured scope

Workers and the submitting caller now retain the queue mutex between finishing
a chunk and deciding the next fair claim. Callbacks still execute unlocked;
condition waits release the mutex. The initial enqueue-to-caller decision also
shares its lock. Ready-job rotation, the single caller permit, context ownership
and drain/destroy rules remain intact. A structural 67-chunk probe records
136 to 68 mutex acquisitions with the same 67 callbacks. Four concurrent
producers at widths 1/2/4 pass exact-coverage and maximum-active-callback tests.
This is not a proof of absolute fairness from every POSIX scheduler.

The Linux affinity helper added in the preceding pass exposed Linux-only
types on other POSIX platforms. Both affinity and topology queries are now
guarded by platform, with the existing online-processor fallback elsewhere.
Non-Linux ARM/x86 compile shims and lifecycle checks pass; execution against an
actual Darwin SDK/runtime is not claimed.

Tanh now starts parallel work at **16,384 elements**, up from 2,048. Its SIMD
arithmetic and larger-input partitioning are unchanged. Direct kernel sweeps
on both x86 and A56 show that the old small-input threshold paid more dispatch
cost than it saved. Other activation/transpose thresholds were retained where
the evidence did not support a general change. New boundary probes include
8K/16K minus one, exact and plus one elements, exceptional values and in-place
operation.

The A56 microbenchmarks use cores 4–7, separate processes in forward/reverse
order, and nine timed batches per condition. Each number below is one process's
median, in microseconds; a batch repeats the warmed kernel for an approximately
three-millisecond target. These are kernel costs, not synthesis latency.

| Four-thread diagnostic | Before: two process medians | After: two process medians |
| --- | --- | --- |
| Empty dispatch, four chunks per thread; pool change only | 5.937 / 6.711 | 3.937 / 4.674 |
| Tanh, 2,048 elements; threshold change with the same pool | 21.504 / 17.450 | 5.516 / 6.019 |
| Tanh, 8,192 elements; threshold change with the same pool | 33.439 / 33.552 | 23.066 / 23.049 |

All 756 numerical rows across the 16 Android probe processes pass exact-bit
1T/2T/4T, in-place and canary checks. Hashes match between threshold variants
for ordinary and arbitrary-bit inputs. At and above 16K, the old/new threshold
variants use the same parallel path: timing differences there are not evidence
of an optimization effect. The independent x86 serial/parallel sweep supports
the same 16K transition. Identical-path 1T controls also show device variability.

### Whole-model A/B remains inconclusive

The frozen pre-optimization baseline and candidate use identical graph/weight
bytes and original inputs, with live encoder outputs fed to the decoder.
Both stages attest CPU execution with zero fallback. All stage outputs and
waveforms are bit-identical before/after. ORT 1.23.2 uses the original ONNX,
CPU execution, the same requested thread count, sequential execution and full
graph optimization. Its default spinning policy is not overridden.

Each experiment uses O/A/B/B/A/O, with two independent processes per native
variant and stage. Native timings surround resident public Execute, including
inline output serialization but excluding request encoding, response decoding
and interstage transfer. ORT timings surround Session.Run with its own output
handling. The table reports the mean of the two **sums of stage trial medians**;
it is a derived prepared-work comparison, not observed end-to-end latency.
Pairing stage samples by index cannot create an actual request distribution or
P95. Internal repetitions are not independent process trials.

| Experiment / model | Policy per stage | Derived A→B change | Interpretation |
| --- | --- | ---: | --- |
| Pool change only, all eight cores allowed / Piper | 3 warmups / 10 measurements | about −0.8% | Device variability; no established gain |
| Pool change only, all eight cores allowed / Kokoro | 3 / 10 | about −2.8% | ORT endpoints also drift +6.3% |
| Final pool + Tanh, cores 4–7 / Piper | 3 / 10 | +7.32% | Triggered the longer check below |
| Final pool + Tanh, cores 4–7 / Kokoro | 3 / 10 | −2.79% | ORT endpoints drift −33.8%; no established gain |
| Final pool + Tanh, cores 4–7 / Piper | 20 / 30 | −0.25% | Earlier slowdown is not consistently reproduced |

In the longer Piper check, A's stage-median sums are 89.200/100.362 ms and B's
are 100.256/88.832 ms. This crossover does not establish a model speedup.
Readable CPU-frequency snapshots do not control DVFS; no readable thermal-zone
samples were available. Core affinity is an experiment condition, not a new
runtime pinning policy. The earlier aborted harness attempt completed Piper
timings but then used incompatible explicit noise in an unrelated changed-
duration ORT diagnostic. It is excluded from final performance aggregation and
is not evidence of a host shape-refusal check. Its task directory was cleaned.

### Why one ULP matters in this source model

The new decoder experiment uses the same short 33,600-sample request and fixed
historical live encoder inputs. The pinned original graph has no stochastic
nodes. Its F0 gate closes below the model's voiced threshold, leaving the
source linear layer's bias followed by Tanh. This creates 4,078 exactly
constant 20-sample STFT frames out of 6,721. It is inherited source behavior,
not exporter removal of noise. Original/lowered phase expressions match after
normalizing Constant nodes versus initializers.

For an ideal periodic Hann window, a constant signal's DFT has zero bins
2–10. The actual stored F32 window has small nonzero real residues: an
independent 80-digit DFT gives about `1e-10..3e-9` after rounding the window
products to F32. The F32 Bluestein algorithm produces larger complex residues,
about `1e-8..1e-7`. Their angles are poorly conditioned. The model passes raw
phase channels into learned convolutions without a magnitude-dependent guard.

With identical Tanh inputs and the same source ORT suffix, changing only phase
reproduces waveform RMSE `0.00211476`; changing only magnitude gives
`1.091e-7` with zero strict failures. The 24 raw branch crossings alone give
RMSE `0.00082350`; the other phase changes give `0.00180858`. Exactly 20,390
wrapped phase differences exceed 0.1 radians in five near-zero bins of those
constant frames. They account for 99.9117% of squared **wrapped phase error**,
not a percentage attribution of waveform error. Nonlinear suffix errors are
not additive.

A source-only experiment requires no native kernel: changing the constant
plateau by one adjacent F32 value, **±1.862645149e-9**, produces waveform RMSE
**0.00143205 / 0.00143951** in the original ORT STFT/phase/suffix. Both changes
have **zero branch crossings**. Thus small local operator errors alone cannot
guarantee the current strict waveform tolerance for this computation.

Canonicalizing constant-frame phase and real-axis DC/Nyquist phase on both
sides reduces their residual RMSE to `6.720e-7`, with zero strict failures.
However, the modified reference itself differs from the original by **RMSE
0.01293174 / maximum 0.199885**. That changes the model's computation and is
not an adopted correction. Production phase, reference and tolerance remain
unchanged. This isolates a decoder sensitivity; it does not attribute every
remaining full-pipeline difference, qualify additional voices/lengths, or
replace audio-quality assessment.

The detailed derivation, standalone PNG/PDF, explicitly labeled diagnostic
F32 WAVs, control arrays and hashes are in `waveform/analysis.md`,
`waveform/phase-conditioning-report.json`,
`waveform/constant-frame-conditioning-report.json` and
`waveform/artifact-receipts.json`. New controls with the preceding final
`0473a2d2…` library reproduce the earlier `d71a2497…` probes byte for byte;
the new scheduling candidates likewise preserve the whole-model outputs in
their CPU/Android comparisons. Kokoro's original full-pipeline strict failure
remains the open numerical issue.

The final native library is `ef4fc811…`. Native inference/full, JS and WASM
inference/full compilation pass; the first package dry-run found a 10-byte
increase in the rebuilt full browser pair, whose ceiling was then set to that
final measured 932,342 B under the user's size-adoption instruction. The final
eight-artifact provenance/ABI/profile and size checks pass, as does the repeated
local package dry-run. No package was uploaded. Native full is 5,061,824 B raw /
2,683,480 B gzip, below the accepted 2,683,568 B gzip ceiling.

Final CPU and release WASM pass 90 public numerical cases per profile; physical
CUDA passes 90 (`build/review-29ade3b-hardware/cuda/corpus-071879a7/`). Android
passes both 90-case profiles plus two 4,673,754-check unary configurations and
the expanded lifecycle contract. Native checks report 78 passes/four unsupported
ISA skips; JS/API 301 passes/two skips; WASM training/PTQ 7/48; Python
training/threading/transpose/PTQ 28; size tests 50. The regenerated Kokoro
decoder's final CPU outputs are bit-identical to the preceding corrected
runtime, retaining RMSE `0.005994636` and 14,047 strict mismatches. Reports and
task-only cleanup receipts are indexed by
`build/review-29ade3b-threads2/final-summary.json`.

## Matched request timing, Conv candidate tests and broader waveform coverage

This follow-up retains the `ef4fc811…` production library. **No new kernel,
fusion or thread-policy candidate was adopted.** Evidence is in
`build/review-29ade3b-next/`; its `final-summary.json` indexes the receipts.
Existing ordered CUDA reductions and compact Kokoro graphs are part of this
baseline, not new improvements from this pass.

### Observed request latency

The Android device was rechecked as Galaxy A56 / Android 16. A private harness
uses generated public dispatch and retains compiled stages. Its single request
timer includes input encoding, live encoder-output transfer, the duration
check, decoder execution, and final host-output decoding. ORT Session.Run also
ends with completed host output. Both exclude initial token conversion,
phonemization, setup, file I/O and previous-output cleanup. Native I32 and ORT
I64 token arrays encode the same values. Stage diagnostics remain separate;
no synthetic sum of stage samples is presented as a request percentile.

Each model/thread configuration runs O/N/N/O: two independent processes per
engine, three warmups and eight internal measured requests per process.
Inherited affinity is retained. Actual task affinity, frequency snapshots,
readable battery/thermal data, provider and session options are recorded.
ORT 1.23.2 uses CPUExecutionProvider, sequential execution, full optimization,
inter-op width one and default spinning. Native retains its owner-pool sleep
policy. Arrival gaps, thermal state and DVFS were not controlled; spinning
versus sleeping is not an established cause. Eight correlated requests give
only a descriptive interpolated P95, not a production-tail estimate.

| Model / threads | Native process medians (ms) | ORT process medians (ms) |
| --- | ---: | ---: |
| Piper / 1 | 114.450 / 114.362 | 120.170 / 120.949 |
| Piper / 2 | 429.408 / 442.007 | 78.244 / 81.052 |
| Piper / 4 | 175.947 / 218.855 | 52.294 / 50.282 |
| Kokoro / 1 | 2405.009 / 2301.046 | 2122.108 / 2032.578 |
| Kokoro / 2 | 1934.152 / 1909.646 | 1285.940 / 1275.247 |
| Kokoro / 4 | 1416.377 / 1414.281 | 760.502 / 766.605 |

Auto was also measured, but native's resolved two-thread pools and ORT's
automatic worker policy are not matched actual thread counts. All native
waveforms are bit-identical across 1/2/4/auto and to the previous final Android
output. Piper passes its existing criterion. Kokoro still fails the same-device
original ORT comparison: RMSE 0.0043437764, 13,866 of 33,600 strict mismatches.
These observations do not establish a change relative to earlier pinned-core,
stage-sum measurements.

### Candidate decisions and memory attribution

The separate four-thread profile still identifies Kokoro decoder Conv as the
largest cost, 810.191 ms. Mul/Add/Transpose take 130.088/101.052/63.786 ms;
encoder Linear/LSTM take 74.629/19.805 ms. These are one profiled request's
operator totals, not untraced request percentiles.

ARM candidates compare spatial/output-channel splitting, grains, serial
admission and spatial/channel tiles for K11/C128/OC128/W6721 at dilation 1/3/5.
Piper checks use actual W31/K3 shapes, C192→768 and C768→192. Two reversed
process blocks cover 1/2/4/auto. A six-row tile improves some long cases but
regresses others: four-thread ratios against current are approximately
0.982/1.001/0.930 for dilation 1/3/5, while two-thread dilation 1 is 1.153.
Identical-path one-thread controls vary substantially. Numerical tests cover
4,320 configurations and 12,389,760 elements, plus Android owner-pool lifecycle.

The isolated six-row candidate then runs complete O/A/B/B/A/O requests.
Kokoro A medians are 1315.864/1327.424 ms; B medians are 1325.319/1282.229 ms.
The descriptive −1.35% change has opposite pairwise directions. Piper, where
no node enters the changed candidate route, shows an apparent −8.63% change.
Setup-stage outputs match in a separate comparison; per-repeat checks cover
final audio, which also matches exactly. This evidence does not
support adopting a kernel or blanket serial threshold.

`memory/` separates warmed copy, same-shape/scalar/channel broadcasting and
both transpose directions from static and executed graph tensor extents.
The executed target nodes have 1,269,560,124 logical output bytes in total;
this is not simultaneous allocation or DRAM traffic. The duplicate transposes
`v946/v951` each write 591,448 bytes from the same `v944`, with observed
0.289/0.285 ms in the separate profile. Mul→Add fusion candidates require
explicit preservation of the F32 multiply rounding boundary, operand order,
broadcasting, aliases and lifetimes. No fusion, materialization removal or
bandwidth-bottleneck claim follows from this analysis.

M9/K2048/N768 probes separate immutable B packing, dynamic A packing and the
complete packed GEMM. Packed B reuse belongs to private engine state; dynamic
A scratch belongs to one invocation. M1/K256/N1024 recurrent projection and
T9/B1/I256/H256 LSTM are measured separately. Empty pool dispatch is only a
control, not an exact measurement of GEMM's panel dispatch. These timings must
not be subtracted to infer model-stage contributions. No LSTM time dependency
or existing LeakyReLU/Transpose policy changes.

### CUDA and numerical limits

On the physical RTX 3090, the current corrected graph's crossed host-to-host
process medians are native 50.450/50.504 ms and ORT 63.340/63.582 ms. ORT uses
CUDA plus explicitly admitted CPU nodes, `use_tf32=0`, and one CPU thread.
These are baseline observations, not newly achieved speedups. Existing large
IGEMM tile comparisons are mixed; the faster small tile for W31 is already
selected by current admission. No CUDA source change was adopted.

A follow-up adds replay-node-visible Nsight tracing and same-device waveform
comparison, which still fails at RMSE 0.0026197687. Whole-process profiler totals
include setup, capture, warmup, replay and teardown; they do not establish
steady-request dispatch cost. `cuda/followup-contract-errata.json` corrects
inherited process-count metadata in that follow-up without changing its sealed
raw reports. The original two-process-per-engine crossed experiment is
unaffected. Public CUDA corpus checks and direct exceptional-value/overflow
checks pass on hardware.

CPU waveform analysis expands to six explicit phoneme requests: 9/16/31 tokens,
speed 0.8/1/1.2 and af_heart/af_bella. The longer case repeats a phrase; this is
not general long-form or multilingual coverage. Full-pipeline RMSE is
0.005994636 (short), 0.004381846 (medium), 0.002593599 (longer),
0.003838016 (speed 0.8), 0.001878089 (speed 1.2) and 0.001402454 (af_bella);
**all fail the unchanged strict criterion**.

For every case, original and lowered ORT decoders receiving the same native
encoder values have byte-identical waveforms and ten intermediate probes.
However, encoder predictor outputs retain one strict mismatch in medium and
three in longer. Diagnostic Tanh-output or STFT-output replacement leaves one waveform
mismatch in longer and three in slow. A bounded exact-STFT trace has its first nonzero floating-point difference
at Atan, but passes its local tolerance and does not establish a kernel defect
or attribute the final residual. Source-only plateau ±1 ULP tests yield RMSE
0.000601–0.001767 with zero branch crossings in all twelve perturbations.
Counterfactual injection residuals are not additive error contributions or
production parity results. No reference, tolerance, phase canonicalization or
model stabilization change was adopted.

Final native and JS/WASM profiles build successfully. All eight release
artifacts pass size, provenance, ABI and profile-boundary checks with unchanged
budgets; both npm packages pass local dry-run. Native CTest reports 83 passes
and four unsupported-ISA skips; public CPU and WASM corpora pass 90 cases per
profile, and physical CUDA passes 90. JS/API reports 301 passes/two skips,
WASM training/PTQ 7/48, and focused Python training/dynamic/PTQ 14 passes.
Additional actual x86 public-context concurrency checks cover two contexts,
1/2/4/auto, changing shapes/inputs and outputs retained after close. Android
owner-pool lifecycle does not qualify concurrent whole-model throughput.
Remote reports were hash-verified before task-directory cleanup; existing
shared assets and the Git index were preserved.

## Four-thread ARM Conv grain and follow-up attribution

`build/review-29ade3b-next2/` contains this follow-up. The adopted change is
limited to direct NEON Conv's output-channel route at **resolved four threads**
and `ow*C*K >= 16384`: split complete 16-channel blocks into up to roughly four
chunks per thread. The admission calculation avoids the three-factor integer
overflow. Arithmetic, accumulation order, FMA behavior, padding/tails and weight
cache lifetimes are unchanged. Other thread widths, A56's default auto width of
two, small jobs and spatial splitting retain their previous policy.

The fixed-binary confirmation uses two O/A/B/B/A/O blocks per configuration,
**four independent processes per engine/candidate**, each with three warmups
and eight measured requests. All timings use the established contiguous
prepared-host-input to completed-host-output boundary. The following figures
are means of the four process medians, not stage sums or production P95.

| Four-thread model | Baseline A, ms | Candidate B, ms | Descriptive change |
|---|---:|---:|---:|
| Piper | 147.489 | 129.284 | −12.34%; 3/4 mirrored pairs improve |
| Kokoro | 1412.936 | 1344.331 | −4.86%; 4/4 mirrored pairs improve |

Piper baseline dispersion is substantial. The timed candidate had the guard
at all thread widths; the final implementation retains the **same four-thread
grain and workers** and restricts the change to that width. Product source and
the final isolated candidate are token-equivalent except for comments/whitespace.
Two-thread independent-process results are approximately flat and mixed,
therefore that policy is preserved. A separate same-process/context ABBA
experiment supports finer grain, but its controller's mixed-policy median/P95
is excluded; only samples grouped by `conv_policy` are used. That design does
not replace independent-process evidence. Inherited affinity, frequency
snapshots and readable thermal data remain recorded. No busy-spin, pinning,
LeakyReLU or Transpose threshold change is part of this optimization.

K11/C128/OC128/W6721 dilation 1/3/5 spatial/K-block weight-reuse candidates are
retained as experiments: about 4% slower at one thread and mixed at four.
Neither W31-only nor all-OC forced serial execution was adopted. Diagnostic
pool tracing separates callback CPU/wall time and caller condition wait;
condition wait includes useful worker computation. Callback start CPU is not
continuous residency, and an output-channel callback total does not identify
W31 cost individually. Instrumented trace latency is not a speed result.

The actual Android dispatcher fixture checks 840 calls and 4,385,920 output
elements per baseline/guarded/final-four-thread executable. The final candidate
changes grain in 50 cases, exclusively at resolved four threads. Candidate and
baseline outputs match **all bytes, including NaN payloads**. Existing
serial-versus-threaded NaN payload differences are counted separately, not
waived in candidate comparisons. Coverage includes admission/OC-block
boundaries, changing inputs while reusing immutable packed weights, poisoned
outputs, sentinels, signed zero, Inf/NaN and finite overflow. The permanent Conv
numerical module passes 11 tests locally; this x86 regression is distinct from
the actual NEON qualification.

Two independent full TTS pipelines also execute concurrently in one Android
process, using four separate stage contexts/model instances rather than two
contexts sharing one compiled model. Setup/JSON parsing is serialized, and
request start/completion barriers separate timed work from housekeeping.
Baseline 1/2/4/auto and the final selected four-thread policy preserve both
models' complete audio bytes. One process per configuration and two measured
pairs are correctness diagnostics, not a controlled concurrent-throughput
comparison. The initial concurrent-setup harness is explicitly superseded
because of cJSON's shared parser error state and peer-housekeeping interference.

The Transpose CSE candidate removes exactly `v951`, one 591,448-byte logical
output write, and passes native/WASM/physical-CUDA dynamic and exceptional-value
checks. Full CPU and Android pipeline outputs remain byte-identical. However,
CPU requested `host.arena` capacity increases from **39,684,884 to 43,126,036 B**
(+8.67%). Peak union of live placement ranges, counting aliases once, remains
34,463,268 B; changed lifetimes affect placement/space reuse. These are plan and
allocation observations, not RSS or DRAM bandwidth measurements. On Android 4T,
A medians are 1305.670/1391.963 ms and B 1337.240/1361.271 ms, with mixed pair
directions and mean ratio 1.00033. Existing Expand CSE is held constant. The
Transpose extension is deferred; its source/tests/patch are preserved only as
task evidence. No arithmetic fusion was introduced.

Bounded waveform follow-up checks six LSTMs, six LayerNorms, two BERT output
projections and two final decoder Convs on identical actual native inputs.
All sixteen pass the unchanged local criteria versus ORT and independent F64
diagnostics. Matching the BERT projection output removes the medium/longer
predictor's 1/3 strict differences, while the first upstream BERT divergence
remains unidentified. Matching phase does not eliminate the longer/slow decoder
residuals; replacing the second generator merge does. These are diagnostic
interventions, not product/reference changes or additive error contributions.
Original CPU short full-pipeline RMSE remains **0.005994636, strict FAIL**.
The latest Android CSE comparison with the selected four-thread binary retains
RMSE **0.0043437764**, 13,866/33,600 strict mismatches against original ORT.

Final native and JS/WASM profiles were rebuilt. The rebuilt native library is
`d8485b4d00e367188f5c49d95f8f86244aead6086c40aba70151d596ca92ba6a`;
its x86 `.text`, `.rodata` and `.data` match the prior `ef4fc811…` library,
while its build ID differs. Both WASM executable payload hashes and raw sizes
are unchanged; provenance hashes changed. Final validation records native CTest
83 passes/four unsupported-ISA skips, JS/API 301 passes/two skips, WASM training 7
and PTQ 48, Python training/dynamic/PTQ 12, plus the new Conv checks above.

The final rebuild initially exceeded two gzip limits by **2 bytes**: lite WASM
540,767 versus 540,765 B, and the inference browser pair 581,675 versus 581,673 B.
Both executable payloads and raw sizes remain unchanged; the source-provenance
hash change accounts for this compressed-size difference. The user explicitly
approved adding exactly 2 bytes to those two limits, without extra headroom or
any other limit/ABI change. The complete eight-artifact release gate, fifty
size/ABI tests and both npm package dry-runs then pass. This size approval does
not affect Kokoro's unchanged strict numerical failure. Remote evidence is
hash-verified and all 11 Android plus one CUDA task directories are absent; staging
content is preserved without add, commit or reset. Final artifacts, approvals
and evidence hashes are indexed by `build/review-29ade3b-next2/final-summary.json`.

## ARM diagnostics, Transpose CSE placement and bounded waveform tracing

This section records `build/review-29ade3b-next3/` with corrected scope. Its original
claims of complete causal identification were not supported. The original evidence
and summaries remain preserved.
Prior four-thread OC grain changes, CUDA ordered reductions and compact graphs are
baseline changes, not new improvements.

### ARM observations

The source is `build/review-29ade3b-next2/threading/analysis.json`. These numbers
average callback-category sums from two instrumented requests; they are neither
one Conv shape's latency nor request P95.

- Kokoro 2T spatial callbacks: 50 jobs/request, 1016.458 ms job wall, 1881.837 ms
  summed callback CPU, 64.281 ms caller condition wait. Summed CPU spans multiple
  threads and can exceed wall time. Large K11/C128/OC128/W6721 Conv remains a target,
  but the entire category total cannot be assigned to that shape.
- Piper 2T OC callbacks: 37 jobs/request, 140.520 ms job wall and 75.176 ms caller
  wait. Only 12 jobs have W31. Recorded CPU is callback start CPU, not residency;
  wait includes useful worker computation. These data do not establish a Little
  core, lock or sleep-policy cause.
- Source access counts are logical accesses, not measured DRAM traffic. Low wait
  does not prove a bandwidth limit. Prior blocking candidates' regressions do not
  isolate loop overhead as their cause. Independent-process 2T grain results were
  flat/mixed, so the earlier Conv grain change was limited to 4T. No global spin, forced pinning,
  blanket threshold or W31 serial policy follows from these observations.

### Arena observations and correction

Duplicate Transpose CSE removes 591,448 B of logical writes (about 577.59 KiB),
while requested arena capacity rises from 39,684,884 to 43,126,036 B: +3,441,152 B
(about 3.28 MiB, +8.67%). Peak union of live placement ranges remains 34,463,268 B
(about 32.87 MiB). These are placement/allocation quantities, not RSS or DRAM traffic.

The earlier direct lifetime-collision explanation is contradicted by the trace:
`v922` lives at steps 893–897, `v926` at 897–910, `v929` at 900–901 and `v931` at
901–912. All end before `v946` begins at 916. Extending `v946` through step 921
therefore does not directly overlap those I32 tensors.

`build/review-29ade3b-next5/evidence/arena_observations.json` rechecks both complete
placement traces. The F32 maximum placed end itself rises from 37,958,232 to
41,399,384 B, the full 3,441,152 B delta. The minimum observed I32 offset equals
the 64-byte-aligned F32 endpoint in both plans; `v1812` occupies the highest F32
endpoint. The earlier explanation of only moving the I32 base is insufficient.
The allocator decision that changed F32 packing remains unisolated. Removing ranks
or introducing a new bin-packing algorithm is not established as necessary.
The observed capacity increase and lack of Android speed gain keep this CSE candidate deferred.

### Bounded waveform evidence

The next3 BERT report compares 426 mapped points in one medium request, rather than
all 500 graph nodes. It records first bit divergence at Embeddings LayerNorm
(max_abs 7.15e-7, strict PASS), first strict divergence at `attention/key_1/Add`
(1/12,288), and final projection differences of 8/8,192. Its script lacks an
unprobed-versus-probed ORT output control. The 16 same-input local operator checks
from the four-thread Conv follow-up qualify only their tested inputs and shapes.

A graph interval containing 349 nodes is not proof of 349 serial error-accumulation
steps. Longer/slow injection results narrow a generator investigation region;
they do not identify one defective kernel or explain every error. A different
short request passing a diagnostic does not refute the medium/longer observations.
Original CPU short RMSE 0.005994636 and Android 4T RMSE 0.0043437764 remain strict FAIL.

Small-M and M1 NEON paths are existing implementations. Weight-pack reuse depends
on immutable ownership and cache admission; dynamic activations and invocation
scratch have separate lifetimes. Not every B operand is permanently cached.
LSTM temporal dependencies remain intact. This section adds no Linear/LSTM or CUDA
speed result. Its historical x86 library is `d8485b4d00e367188f5c49d95f8f86244aead6086c40aba70151d596ca92ba6a`;
that hash does not qualify later source changes.

## Follow-up waveform diagnostics and ARM Conv candidates: corrected scope

`build/review-29ade3b-next4/` preserves additional diagnostics and an ARM candidate.
Claims that it completely identified the waveform cause, proved an engine fix
impossible, or resolved ARM bottlenecks are withdrawn.

### Short-request diagnostic limits

- The summary records 0/10,377 strict mismatches in the short encoder output,
  with max_abs 2.89e-6 for text features and 3.47e-5 for predictor features;
  durations are exact. Strict agreement is not bit identity. The decoder still
  fails with ORT encoder outputs (RMSE 0.006932, 13,976/33,600), proving a decoder
  discrepancy exists but not that encoder differences contribute nothing.
- ORT excitation injection yields final short audio RMSE 1.0998e-6 and strict
  0/33,600 in the recorded diagnostic. It does not check each of 349 intermediate
  nodes, refute longer/slow residuals, or turn full-pipeline parity into PASS.
- `next4/evidence/op_results.json` records 304 selected local operators, including
  23 Convs, from F0/N/decode on held ORT inputs with zero strict failures. It does
  not cover all generator residual Convs, inputs, shapes or ISAs, and does not
  establish FMA rounding as the sole explanation for all differences.
- `check_exact_intermediates.py` selects the lowest-RMSE tensor among candidates
  with the same element count. This cannot establish semantic identity and is
  excluded as boundary evidence. Replacement probes must use compiler name mapping,
  exact dtype/shape and unchanged-output controls for both ORT probes and native debug.
- Saved audio in `next4/kokoro_run_test/` has 34,800 samples from a different
  request. It must not inherit the 33,600-sample short metrics or reference identity.

The replacement `next5/evidence/check_mapped_intermediates.py` maps eight boundaries
by compiler identity and replays only eliminated Squeeze/Transpose layout nodes
specified by compiler IR before exact dtype/shape comparison. In
`mapped-short-final/report.json`, recompiled-versus-prepared native output,
unprobed-versus-probed ORT output, and ordinary-versus-debug native output are all
byte-identical. The same-native-encoder decoder comparison has RMSE 0.00216501048
and 13,719/33,600 strict failures; `decode3` already has 4/57,344 differences.
These are full-path boundary observations, not same-immediate-input isolated
kernel tests, and do not establish a defective kernel or a single cause. Three
earlier mapping/document-name admission rejections remain recorded separately.

### Source phase sensitivity, not a universal kernel diagnosis

The preserved `review-29ade3b-threads2/waveform/source-phase-provenance.json` and
analysis establish a narrower mechanism. In the short request, 4,078 of 6,721
STFT frames contain 20 identical excitation samples and coincide with the all-gate-false
frame mask. The source gate tests F0 > 10; a false gate does not establish F0 = 0,
absence of speech, perceptual silence or inaudibility.

The observed native/ORT F32 plateau difference does not compare SIMD against
scalar libm: ORT 1.23.2 CPU F32 Tanh uses MLAS. Neither a universal IEEE 754
"1–4 ULP allowance" nor a measured 5–10x inference slowdown for scalar math follows.
An ideal periodic length-20 Hann window gives constant-input DFT components at
bins 0 and ±1. Only bins 2–10 of the one-sided ideal spectrum cancel, and the
stored F32 window coefficients need not cancel exactly.

The source phase path is **imag/real → Atan → sign-controlled ±pi correction**.
It has no `Where(magnitude > 0, Atan2, 0)` gate. Magnitude and raw phase are
concatenated into learned Conv inputs. The earlier direct explanation of input
phase going through `sin` and multiplying `conv_post` magnitude does not match
this dependency path.

In the prior source-only ORT diagnostic, ±1 ULP plateau perturbations cause about
0.00143 waveform RMSE with **zero raw phase branch crossings**. This demonstrates
source sensitivity, not a complete explanation of full-pipeline error through
±90-degree jumps. Phase canonicalization changes the original reference by about
0.01293 RMSE and remains diagnostic only. Epsilon gating, a changed reference or
tolerance, and injected tensors are not repairs of existing-model parity. No
perceptual-quality conclusion is available without separate evaluation.

### ARM candidate admission and measurement

The next4 candidate broadens the qualified 4T OC grain guard to at least two
threads and adds a 64-output-position outer spatial tile. The former 4T guard
was deliberate after mixed 2T process results, not an established bug. Taking
unassigned chunks from the shared counter cannot preempt a chunk already running
on another worker; the code change alone does not establish removal of 75 ms wait.

`64*C*4` omits the kernel halo, stride/dilation and other cache users. For
K11/C128/stride1, dilation1/3/5 spans 74/94/114 input positions. No measured
cache-residency or 87.5% DRAM reduction accompanies this candidate. The summary's
USB-disconnected assertion lacks a preserved timestamped command log; a subsequent
review sees the same Galaxy A56 SM-A566S/Android16 connected. Each run must recheck it.

`next4/native_pipeline_arm` is an Android aarch64 ELF with `/system/bin/linker64`,
2,542,584 B, SHA256 `180c56433ef77694054e8bf9feaee19b196d9a4c3ee70ab8e351fd644d9a14ba`.
ELF validation and x86/WASM tests do not exercise changed NEON code. Adoption needs
actual device numerical/shape checks, model A/B, 1/2/4/auto and concurrent-context checks.

### Build and regression evidence

Next4 lists 11 Conv, 6 ISA/grain, 32 TTS, 25 Python, 301-pass/2-skip JS and release/npm
successes as literal strings in `write_summary.py`; the task inventory contains
no corresponding command logs/exit codes. These are historical assertions, not
newly audited passes. This evidence gap does not prove the commands never ran.
Current receipts match and JS/WASM provenance was refreshed; native binaries and
library retain their next2 hashes.

New checks use `next5/evidence/run_checked.py` to retain actual argv, cwd, relevant
environment, UTC times, exit status, log hash and source/artifact hashes, and
`audit_receipts.py` to audit receipts. Command success, bit identity, unchanged
ORT strict parity and measured speed remain separate conclusions. Historical
summaries stay unmodified and must be read with these corrections.

## Candidate revalidation: ARM policy and Kokoro strict waveform status

No new ARM policy or Tanh candidate was adopted. The qualified Conv baseline and production math remain in use, and Kokoro still fails the original strict waveform criterion. Evidence is retained in the [review index](../../build/review-29ade3b-next5/final-summary.json). Existing staging and the approved size budget were preserved. Earlier causal claims and verification scopes were corrected against the actual graph and raw evidence.

### ARM: output preservation and speed are separate decisions

The next4 tile64 loop could overflow at large extents; its index-corrected variant also changed NaN payloads in the actual dispatcher test. The next4 tile64 and broad grain change for two or more threads were therefore excluded, restoring the qualified existing 4T policy. Later candidates were built separately under task-owned paths. Cross-compilation, copied-worker tests, or tests that never enter a new branch do not establish production qualification.

Experiments on the separate LeEco development phone (`ZL1_CN/zl1`, `MSM8996pro`, Android11) are complete, with **none of the measured candidates adopted**. Results are not pooled with Galaxy A56 measurements. The device summary and hashes support these decisions.

| Candidate | Output scope established | LeEco timing and decision |
|---|---|---|
| 2T-only OC grain | Dispatcher output bits, model setup-stage snapshots and final audio preserved; timed-repeat identity covers final audio only | Whole-request latency increased by 7.95% for Piper and 1.56% for Kokoro. No global adoption |
| Original 8×8 | New worker entry confirmed; NaN payload differences | Failed numerical preservation; timing skipped |
| 8×8 with finite admission and 64-bit indices | All 69,936,000 output elements matched bit for bit across 776 cases, including nonfinite fallback and finite overflow | Slower in all 12 W6721/dilation1,3,5/threads1,2,4,auto combinations. Rejected without another whole-model run |
| 5×16 and 6×16 | Ordinary finite outputs matched; special values and finite overflow remain unqualified | 5×16 was slower in all nine combinations. 6×16 was only about 0.8–2.6% lower at dilation5 and slower elsewhere. Weak mixed evidence; neither adopted |

Timing changes in the table compare the means of the process-level measured medians. The 2T whole-model comparison timed contiguous prepared host input→completed host output in O/A/B/B/A/O order: two independent processes per engine, with one warmup and four measured requests inside each process. Spatial screens used the actual Conv dispatcher, including cache lookup, finite checks and thread-pool dispatch. The 8×8 schedule was ABBA and the 5×16/6×16 schedule ABCCBA, each with two independent processes per variant. Dynamic input versions and internal repetitions are correlated samples, not extra independent trials. Affinity, frequency and available thermal data were recorded. Stage samples were not added into request P95, and ORT spinning/native sleep differences were not established as a cause.

These were **whole-candidate comparisons, not isolated tile-size experiments**. The baseline interior processes two input channels per loop iteration; the earlier 5/6/8 candidates process one, and the candidates add finite-scan costs. An inner loop without stack accesses is insufficient to explain throughput. A paired-channel 5×16 refinement was therefore [built as a separate object](../../build/review-29ade3b-next5/arm5x16-paired-finite-safe/static-comparison.json) and tested afresh on the reconnected A56. It does not inherit the earlier candidates' results.

The earlier Galaxy A56 screen (SM-A566S/Android16, SoC property `s5e8855`) contains six complete blocks and an incomplete Kokoro4T block. The incomplete block is excluded from comparisons; an initial 2T improvement signal has not been promoted to confirmation. After reconnection, 168 result files from `/data/local/tmp/volvox-next5-pipeline-aa9a3cd69924` were preserved with verified hashes, followed by [cleanup and absence verification](../../build/review-29ade3b-next5/arm/pipeline-screen/interrupted-kokoro-task-cleanup.json). The LeEco summary's statement that A56 cleanup was pending is a snapshot from the end of that device's investigation; this later receipt establishes the current A56 state. The separate record confirming removal of the nine LeEco task paths and task processes remains intact.

The new paired5×16 candidate matched the A56 baseline bit for bit across **776 cases and 69,936,000 elements**. Tests verified actual worker entry, all modulo5 entry/interior-tail residues, nonfinite fallback and finite overflow. This is preservation at each identical thread setting; existing cross-thread NaN-payload differences in the baseline remain. The [independent array, coverage and SHA audit](../../build/review-29ade3b-next5/evidence/paired5-a56-independent-audit.json) records that distinction.

The A56 paired5×16 microbenchmark retained identical output bits over 48 independent processes covering 12 W6721/dilation1,3,5/threads1,2,4,auto configurations. Comparing the means of process-level measured medians gives six lower and six higher results, without a general speed improvement. One thread is broadly unchanged, 2T/auto is mixed, and 4T at dilation1/3 is slower. The numerical fixture and microbenchmark task have been cleaned up.

A representative whole-request comparison is also complete. Piper2T and Kokoro1/2/4T were tested in O/A/B/B/A/O order: 24 independent processes in total, each with two warmup and six measured requests. Comparing the means of process-level measured medians gives **+2.20%** for Piper2T and **−0.13%**, **−2.70%**, **+0.75%** for Kokoro1T/2T/4T. These mixed or near-flat results do not establish a general speedup. **The paired5×16 candidate was not adopted; the qualified baseline is retained.**

Native setup-stage outputs and final audio matched the baseline bit for bit at each identical setting; repeated timed-output identity checks cover final audio. Against same-device ORT, Piper passes strict comparison, while Kokoro still fails with RMSE **0.0043437764** and **13,866/33,600** mismatches. The [independent audit](../../build/review-29ade3b-next5/evidence/paired5-a56-full-independent-audit.json) rechecked raw samples, statistics, arrays, hashes and binary bindings. This is separate from the original x86-reference waveform analysis and does not resolve strict waveform parity or the ARM spatial bottleneck. The final A56 cleanup audit confirmed that all seven known task paths were absent and no owned process remained. The final device summary records this completed state alongside the measurement scope.

### Kokoro: matching Tanh alone did not resolve strict waveform failure

With the original model, references and tolerances preserved, **all 36 full-pipeline runs failed strict comparison**: six requests covering length, voice and speed, each run with the baseline and five Tanh candidates. All six baseline outputs reproduced the preceding saved audio byte for byte. The correct CPU short result is RMSE **0.00599463611** and **14,047/33,600** mismatches. The historical 14,055 count belongs to a different, older-geometry output; reference or tolerance changes do not explain it. The [saved-array audit](../../build/review-29ade3b-next5/evidence/short-strict-count-provenance.json) records a reporting correction, not a new improvement.

The last standalone-Tanh candidate matched ORT bit for bit on 234,000 actual captured inputs, but full waveforms still failed. It also violated the existing `|Tanh|≤1` invariant for some inputs and was not adopted. The [waveform report](../../build/review-29ade3b-next5/waveform/summary.json) separates candidates that changed the shared LSTM helper from those affecting only standalone Tanh. Provider compatibility does not imply greater mathematical accuracy. These results neither prove every kernel correct nor establish that an engine-side fix is impossible.

The [short fixed-encoder follow-up](../../build/review-29ade3b-next5/waveform/short-boundary-followup/analysis.md) found bit-identical STFT and magnitude on identical input, with phase passing local strict checks. Given identical F0, upsampling and accumulated oscillator phase were also bit-identical; local Sin and projection passed. The original predicted F0 differed within its local tolerance, but those differences accumulated into phase and then strict differences in Sin. F0 prediction upstream remains the next unresolved boundary; this observation alone does not establish another kernel bug. Substituting the original ORT STFT input produced a strict-passing diagnostic audio result. That intervention is not an unchanged production pipeline PASS or a resolution of original-model parity. No phase canonicalization, epsilon gate, reference replacement or tolerance relaxation was adopted.

The [F0 boundary follow-up](../../build/review-29ade3b-next5/waveform/f0-followup/analysis.md) tested 19 individual operators and two normalization components using actual native inputs from the same short fixed-encoder request; all passed local strict checks. It includes F0 projection and style Gemm, so its scope differs from the earlier 304 checks using ORT-captured inputs. In the observed full-path interval, norm2 differs strictly after receiving an already-different conv1 input, while norm2 passes when isolated on identical input. No new local kernel defect was demonstrated. F0.2 conv1 and its preceding norm1/F0.1 inputs remain unresolved. The original full waveform and the unmodified diagnostic candidate remain strict FAIL. This is a CPU1T investigation of one short request, not a new six-case PASS or performance result.

### Build, regression and remaining scope

Native inference/full and both JS/WASM profiles, native/API/numerical/dynamic-input/training/PTQ regressions, the eight-artifact size/provenance/ABI/profile checks and npm dry-runs qualify the **retained baseline product**. Actual commands, exit codes, logs and source/artifact SHA-256 bindings are retained in the [validation records](../../build/review-29ade3b-next5/validation/) and final evidence index. They do not qualify the rejected ARM or Tanh objects.

This investigation adds no CUDA change, execution or performance result; earlier ordered-reduction and compact-graph gains are not counted again. Speed, baseline output identity, original ORT strict parity and measurement scope remain separate. The outcome is exclusion of the incorrect next4 changes, corrected reporting and a documented measurement scope; neither original problem is claimed solved. The earlier pending A56 task and the new numerical, microbenchmark and representative full-screen tasks have all been removed, and final checks confirmed that the known task-owned paths were absent and no task-owned measurement process remained. An [independent SHA audit](../../build/review-29ade3b-next5/evidence/paired5-a56-final-retention-independent-audit.json) verified all 1,563 entries in the retained local artifact manifest.

### Subsequent known-issue classification

At the user's request, the existing discrepancy is now tracked as accepted known issue [`KOKORO-STRICT-001`](../kokoro/README.md#known-issue-strict-waveform-parity). Further waveform investigation is deferred; independent ARM optimization may proceed with baseline output preservation and its own correctness and performance checks. The measurements above, original strict FAIL results, references, tolerances and failure exit codes remain unchanged. ORT-only sensitivity does not prove an ORT kernel defect or explain every native residual. New output differences, increased error or newly failing local operators require separate regression review.

## WebGPU parameter upload batching and resource policy

Each dispatch now copies its live parameter bytes into an independent aligned
slice of host staging. The bridge uploads each used uniform chunk before
submitting its command buffer. Completion includes both GPU work and validation
before a chunk becomes reusable. The current policy retains at most four
completed chunks and **4 MiB of GPU storage**, with matching host staging of up
to 4 MiB. Active and pending submissions own separate storage; 4 MiB is not a
total bridge memory limit. The incremental ROW estimate accounts for that row's
GPU chunks and host staging, not the shared idle pool.

The earlier batching experiment in `build/tts-webgpu-params-batching/` used Deno
2.9.6 and an RTX 3090 with no numerical fallback. Its candidate JS SHA256 is
`b2390073b765e77378eb5104ff26fe21685ddf244f893901661d7b216a59ed48` and full WASM
SHA256 is `a08234866241ed4ec66c5214cbd67bc932e81d87194ec884bdf0e5a06dd25da1`.
Those artifacts precede the later failure-path, 4 MiB pool and C ROW-estimate
fixes; these measurements do not qualify the latest source or release files.

Piper used AB, BA, AB ordering, with three independent processes per variant,
two warmups and five measured requests inside each process. The mean of the
process medians decreased from **255.22 to 230.52 ms (about 9.68%)**. The 15
internal samples per variant are not 15 independent trials. Kokoro used only
one process per variant; its **944.03 to 908.07 ms (3.81%)** change is preliminary.
Saved Piper and Kokoro baseline/candidate waveforms were bit-identical. Kokoro's
original strict FAIL and failure exit code remain unchanged.

The profile records parameter `writeBuffer` counts of **1,700 to 2** for Piper
and **2,281 to 2** for Kokoro. Alignment increases the uploaded parameter bytes
from 62,160 to 435,200 B and from 85,608 to 583,936 B respectively. Reduced API
call overhead is distinct from reduced transfer volume. The profile's single
untraced request is separate from the repeated latency trials. Source/artifact
hashes and new execution receipts are required for each later qualification.

### Final 4 MiB pool ownership and resource verification

The [final verification record](../../build/fix-webgpu-pool-20261106/final-summary.json)
qualifies the subsequent ownership fix: a cached chunk stays owned by the idle
pool until registration with the active encode succeeds. A regression injects
registration failure, verifies reuse and checks exactly-once destruction. The
four-chunk / **4 MiB GPU idle pool and matching host staging remain unchanged**;
active and pending storage is additional. An independent C capacity oracle
checks supported parameter sizes, chunk boundaries, host staging, active chunks
beyond the idle limit and overflow rejection.

On the remote GPU host, Deno 2.9.6 / RTX 3090 compared the **prior batching
candidate** against the final artifacts in A/B/B/A order for each model. Each
variant had two independent processes, each with one first execution, two
warmups and five measured requests. Timers cover prepared host input through
completed host audio across all stages; snapshot copies and file writes are
outside them. All **32 audio captures per model were bit-identical**, with zero
fallback. Piper passes strict comparison. Kokoro retains **RMSE 0.0041591035 and
19,956/39,600 mismatches**, its original strict FAIL and failure exit code. Full
artifact/source hashes and raw trials are in the
[remote analysis](../../build/fix-webgpu-pool-20261106/remote/analysis.json).

This verifies output preservation for one input per model on one GPU. It does
not repeat the earlier unbatched-versus-batched experiment or establish a new
speedup: Piper's prior process medians alone ranged from 233.30 to 317.58 ms.
Internal repetitions are not independent trials, and stage samples or
percentiles are not summed. Inherited affinity and process-boundary GPU
frequency/temperature were recorded, not controlled. These fixtures do not
exhaust the 4 MiB pool; local boundary and failure tests provide separate coverage.

Native inference/full and both JS/WASM profile builds pass. The recorded local
regressions include native **83 pass / 4 unsupported-ISA skips**, JS **303 pass /
2 skips**, focused WebGPU **107 pass**, WASM training **7 pass** and PTQ **48 pass**.
Actual GPU composed inference and seven training cases also pass. Three original
Deno fault expectations (staging allocation, pass creation and close pending)
still fail identically for prior and final artifacts; separate observed-cleanup
diagnostics pass without replacing those failures. The multi-device failure
also reproduces on the second device in a pure Deno/Vulkan control without
Volvox. Its Deno, Vulkan or driver cause remains unisolated.

All eight release artifacts pass provenance, ABI and profile-boundary checks,
but the [original size gate](../../build/fix-webgpu-pool-20261106/size/release-size.json)
still **fails**: full browser-pair gzip is **932,671 B against 932,342 B**, an
excess of **329 B**. The proposed budget adjustment is unapproved and unapplied;
its separate successful check is not a release PASS. Remote reports and artifact
hashes were verified locally before all task-owned remote paths were removed,
recovering **464,633,856 B**. This task adds no ARM or CUDA performance result.

### WebGPU bind-group caching and inactive tensor reuse: initial experiment and review

The [initial reports](../../build/tts-webgpu-bindgroups/reports/) measure a candidate
that retained inactive tensor buffers by pointer and extent, cached explicit
bind-group layouts and groups, and omitted repeated state commands within a pass.
The layouts were created alongside the pipelines; the implementation did not
cache calls to `pipeline.getBindGroupLayout(0)`. The original extent replacement
path retired buffers after outstanding work completed, rather than destroying
in-flight storage immediately.

The [independent audit](../../build/review-webgpu33-20261106/evidence/evidence-audit.json)
separates the measurement types. On RTX 3090 / Deno 2.9.6, Piper had **three
independent processes per variant** in A/B/B/A/A/B order; Kokoro had **one per
variant** in A/B order. Each process used two warmups and five measured requests.
Those internal repetitions are not additional independent trials.

| Model | Observation | Baseline | Candidate |
| --- | --- | ---: | ---: |
| Piper | API-profile `createBuffer` calls / host time | 661 / 11.33 ms | 191 / 3.24 ms |
| | API-profile `writeBuffer` host time | 34.19 ms | 27.01 ms |
| | API-profile `createBindGroup` calls | 2,092 | 1,926 |
| | Single request with API wrappers | 239.12 ms | 223.17 ms |
| | Three process medians | 232.86 / 312.51 / 229.73 ms | 217.10 / 215.59 / 220.89 ms |
| Kokoro | API-profile `createBuffer` calls / host time | 798 / 18.25 ms | 199 / 3.46 ms |
| | API-profile `writeBuffer` host time | 227.43 ms | 148.73 ms |
| | API-profile `createBindGroup` calls | 2,826 | 2,264 |
| | Single request with API wrappers | 915.68 ms | 825.52 ms |
| | One process median | 906.23 ms | 821.18 ms |

The profile's `untraced_ms` means one request without device/node tracing after
two preparatory requests; **API instrumentation wrappers remain active**. Its
−6.67% Piper and −9.85% Kokoro differences are single-request observations, not
established general speedups or request P95. Host API durations are not added to
request timing. Piper's baseline process variation and Kokoro's single trial per
variant limit inference about performance. The retained run has no affinity,
frequency, temperature or competing-job record. Its shell wrapper applied
`env` and `timeout` to `select_runtime.py`, but the `&& deno ...` command ran
outside that scope. The requested Deno environment and timeout were therefore
not enforced by that command. Node captures also dropped events.

Four retained raw captures per model (profile/untraced × baseline/candidate)
were bit-identical: Piper SHA-256 `ea33dfc4aa293057b8db60bcb00b22a12d54e9a07b8acab397fa12f894d76415`
and Kokoro `f8c08e00505399aa0eaf1e5592514ada211e2f65d6e05c5023ec51106fcabe14`.
The repeated benchmark requests retained aggregate error metrics rather than
all raw audio; they do not have the same independent array-verification coverage.
All recorded routes have zero fallback. Piper passes strict comparison; Kokoro
retains **RMSE 0.0041591035, 19,956/39,600 mismatches and strict FAIL**. This is one
prepared input per model, not dynamic-input or training qualification.

Original collection receipts report a failed sealed-report comparison, and the
cleanup receipt says `task_removed=false`. A later
[read-only postflight](../../build/review-webgpu33-20261106/evidence/remote-postflight.json)
confirmed the exact task path absent and no process with that task cwd. The
missing intervening recovery/hash-verification receipt remains an evidence gap;
current absence does not establish a complete earlier preservation workflow.

The original candidate disabled inactive reuse during memory capture. Its memory
trace therefore did not measure ordinary execution's retained storage. The
[follow-up review](../../build/review-webgpu33-20261106/findings.json) reproduced
missing state initialization for auxiliary passes, a buffer leak across memory
capture transitions, stale reverse-index keys and lost ownership on inactive
registration failure. The initial 105 passing WebGPU tests and 305 passing API
tests (two skips) did not cover those failures.

The follow-up policy bounds inactive tensor storage to **4 MiB total**, separately
from the existing **4 MiB parameter GPU pool and up to 4 MiB matching host staging**.
Active and pending-retirement storage remains additional. Memory capture observes
the same pool policy as ordinary execution. Pool ownership is retained until
active registration succeeds; replacement safely retires the prior entry.
Storage is reused only after its last encoded work or snapshot copy and validation
complete successfully; prior activation validity is cleared and the current
residency classification is applied. The selected follow-up preserves upload
validity only when the same host pointer and exact extent were and remain
canonical immutable weights with a valid cached upload. Explicit invalidation
clears that validity in both active and inactive storage. Mutable activations
do not use this exception.
The bind-group cache holds at most 4,096 groups with oldest-inserted eviction.
Group eviction clears every participating reverse index, and every new compute
pass starts with empty tracked state. The initial measurements above do not
qualify the later implementation or its changed retention limit; new artifacts
and execution receipts are required.

The reviewed initial candidate's rebuilt full browser pair measured **933,177 B**.
The user approved raising only its gzip ceiling from **932,342 B to 933,177 B
(+835 B)**, with no additional headroom. This includes the earlier pending 329 B;
it is not added twice. The [size adoption record](../../build/review-webgpu33-20261106/size-adoption.json)
preserves the original failure and remaining limits. Size acceptance does not
establish runtime correctness or waive Kokoro's strict criterion.

### Validated bounded candidate and lifetime fixes

The [bounded candidate record](../../build/fix-webgpu33-20261106/remote/analysis.json)
qualifies source SHA-256 `1c6dc3d516ff3862fbb7953e80104e1b4cbee8e52cd9bdf6d056f18e809e511a`
separately from the initial experiment above. This is the validated reference
for subsequent performance diagnostics, not a final product selection.
The implementation preserves ownership until active registration succeeds,
clears reverse indices on eviction, resets state for each compute pass and
keeps pending encode/snapshot storage ineligible for reuse. Failed
completion storage is discarded. The separate 4 MiB inactive tensor limit and
4,096-group FIFO limit bound retained resources; the existing parameter pool
remains at 4 MiB GPU plus up to 4 MiB host staging. Memory capture uses the same
reuse policy as ordinary execution. Active and pending-retirement storage is
additional to those idle limits.

The [focused local regressions](../../build/fix-webgpu33-20261106/tests/focused-final.json)
pass **116 tests**, covering pass state, observer transitions, failed ownership
registration, byte/cache limits, mutable inputs, failed completion and pending
snapshots. This candidate's API run passes **316 tests with two skips**; WASM training
passes **7**, and PTQ **48**. Native inference/full and both JS/WASM profiles
build successfully. All eight candidate artifacts pass provenance, ABI,
profile-boundary and adopted size checks.

The [physical GPU record](../../build/fix-webgpu33-20261106/remote/analysis.json)
uses RTX 3090 / Deno 2.9.6. The previous source reproduced the missing-pipeline
validation error, observer-transition leak and excess idle retention; the
bounded source passes all four focused cases, including changing host input values.
The initial standalone harness used an incorrect no-uniform ABI sentinel;
that attempt is retained separately and does not qualify product behavior.
The corrected before/bounded-candidate harness uses the same generated ABI value.
The candidate's public-runtime composed check and seven SGD/AdamW training cases also
pass, with finite-difference, rollback, nonfinite recovery, cancellation,
dropout and profiling coverage. Earlier unrelated Deno multi-device OOM and
three public error-expectation mismatches were not rerun or reclassified here.

For each TTS model, **A/B/B/A** compared the frozen initial reuse candidate with
the bounded candidate. Each variant had **two independent processes**, each with one
first execution, two warmups and five measured requests. Timers cover prepared
host input through completed host audio across all stages; snapshot copies and
file writes are outside the timer. No stage samples or percentiles are summed.
All **32 retained audio captures per model are bit-identical**, with zero
fallback. Piper passes strict comparison; Kokoro retains **RMSE 0.0041591035 and
19,956/39,600 mismatches**, its original strict FAIL and failure exit code.

| Model | Initial reuse candidate: process medians | Bounded candidate: process medians | Change in mean of process medians |
| --- | ---: | ---: | ---: |
| Piper | 226.48 / 214.96 ms | 268.24 / 237.95 ms | 220.72→253.10 ms, **+14.67%** |
| Kokoro | 814.59 / 819.58 ms | 915.89 / 915.60 ms | 817.08→915.75 ms, **+12.07%** |

The bounded candidate is slower in these samples. This is a correctness and
output-preservation comparison against the initial reuse candidate, not a new
batching speedup. One prepared input per model, one device and two independent
processes per variant limit generalization. Affinity was recorded rather than
pinned, and GPU frequency/temperature were sampled at process boundaries rather
than controlled. No ablation separates the effects of bounded retention and
lifetime checks, so the regression's cause is not established. Internal
repetitions are not additional independent trials or a robust P95 estimate.

All four new remote task paths were removed after verified collection; the
postflight found no owned process and recorded **439,287,808 B recovered**.
This complete new receipt chain does not fill the earlier experiment's missing
recovery provenance.

The user's accepted bind-group/buffer-reuse size increase and request to fix its remaining issues
were applied to this candidate's measurements through
[three exact ceiling adjustments](../../build/fix-webgpu33-20261106/size-adoption.json):

| Size cell | Previous ceiling | Measured candidate ceiling | Increase |
| --- | ---: | ---: | ---: |
| Full readable JS raw | 697,688 B | 698,707 B | 1,019 B |
| Full readable JS gzip | 161,321 B | 161,553 B | 232 B |
| Full browser-pair gzip | 933,177 B | 933,580 B | 403 B |

The original failures remain saved. These changes add no prospective headroom
and preserve all other byte/ABI limits, the allowance formula and numerical
criteria. They do not change Kokoro's known strict discrepancy.

### Policy diagnostics and canonical immutable upload reuse

The [separate policy diagnostic](../../build/fix-webgpu33-20261106/remote/policy-diagnostic/analysis.json)
changed only task-owned readable JS copies. It compared tensor idle caps of
4, 16 and 64 MiB, plus a 4 MiB cap admitting individual buffers no larger than
256 KiB. Every variant retained the **4 MiB parameter cap**, original WASM,
shader payloads, model, references and tolerances. Each variant had two
independent processes per model, each with two warmups and five measured
requests; counter-instrumented requests ran separately from latency trials.

| Tensor policy | Piper mean of process medians | Kokoro mean of process medians |
| --- | ---: | ---: |
| 4 MiB aggregate | 242.19 ms | 917.16 ms |
| 16 MiB aggregate | 240.13 ms | 916.01 ms |
| 64 MiB aggregate | 240.46 ms | 908.08 ms |
| 4 MiB aggregate, 256 KiB admission | 237.59 ms | 913.37 ms |

Neither larger retention nor the admission filter resolved the preceding
regression, and those alternatives were not adopted. These are within-run
comparisons; times from separate experiments are not combined into a new
speedup claim. GPUBuffer live-byte counters exclude driver allocations and JS
metadata. API host durations include driver/scheduling and wrapper costs, so
they do not isolate memory bandwidth.

The selected follow-up source
`f43538f777291ae98a9ad6cdaecf5074943fcbf4edce3f713289bea12a0bbea8`
keeps both GPU pool limits at 4 MiB and the 4,096-group bound. It adds only the
canonical immutable-weight upload exception described above. The
[E diagnostic](../../build/fix-webgpu33-20261106/remote/weight-upload-diagnostic/analysis.json)
compared the prior bounded candidate with that exception in A/E/E/A order,
again with two independent processes per variant and separate API counters.
Piper's mean of process medians was **249.96→246.52 ms**; Kokoro's was
**918.42→918.37 ms**. The API call counts and upload byte counts were identical.
All **40 retained captures per model**, including the separate counter runs,
were bit-identical, with zero fallback and unchanged strict verdicts.
These fixtures did not demonstrate reduced upload traffic or an established
TTS speedup from the exception. Its local focused regression suite passes
**117 tests**, including mutable input, weight reclassification and explicit
invalidation. Diagnostic copies are separate from official rebuilt-artifact
qualification.

### Official final artifact verification

The [final evidence index](../../build/fix-webgpu33-20261106/final-summary.json)
binds the official rebuilt artifacts to source `f43538f…` and the policy selected
above. Native inference/full and both JS/WASM builds pass. Focused regressions
pass **117 tests**, API tests **317 with two skips**, WASM training **7** and PTQ
**48**. All eight artifacts pass provenance, ABI, profile and size checks.

On [RTX 3090 / Deno](../../build/fix-webgpu33-20261106/remote/official-final/analysis.json),
the exact final source passes five focused cases, including immutable-weight
reuse followed by explicit invalidation and fresh upload. Composed execution,
seven SGD/AdamW cases, rollback, failure recovery and profiling also pass.
Official TTS artifacts were compared with the original reuse candidate in
**A/B/B/A** order for each model: two independent processes per variant, each
with one first execution, two warmups and five measured requests. All **32 raw
audio captures per model are bit-identical**, with zero fallback. Piper passes
strict comparison; Kokoro retains RMSE **0.0041591035**, **19,956/39,600** mismatches
and the original strict failure exit code.

| Model | Original reuse candidate: mean of process medians | Official final: mean of process medians | Observed change |
| --- | ---: | ---: | ---: |
| Piper | 223.47 ms | 252.69 ms | **+13.08%** |
| Kokoro | 818.35 ms | 916.21 ms | **+11.96%** |

The latency regression remains in this final sample. Correctness and ownership
fixes do not establish a speedup. This comparison covers one prepared input per
model and one device, with two independent trials per variant; internal repeats
are not additional independent trials. Timers measure the completed whole
request directly, without adding stage samples or percentiles. Affinity and
frequency/temperature observations do not remove scheduling variability.

The [final size adoption](../../build/fix-webgpu33-20261106/size-adoption-final.json)
preserves the earlier candidate records and sets only measured ceilings: full
readable JS **698,818 B raw / 161,556 B gzip**, and the full browser pair
**933,598 B gzip**. No future headroom, ABI or numerical change was added.
The [complete cleanup chain](../../build/fix-webgpu33-20261106/remote/cleanup-all.json)
verifies collection before deletion for all **seven** tasks in this correction
and its diagnostics, with **1,731,739,648 B** recovered cumulatively. The final
postflight found no owned task path or process. The earlier initial experiment's
missing recovery record remains a separate historical limitation.

## Kokoro W8A8 WASM SIMD optimization and measurement correction

The decoder static INT8 fixture uses `af_heart`, speed 1 and the single text
`Hello!`, with an F32 encoder and LSTM. Its **33,600 samples at 24,000 Hz are
1.4 seconds of audio**. RTF divides the completed whole-request latency by
**1,400 ms**. The timer includes prepared host-input binding, the encoder,
duration feedback, decoder and completed host audio. Loading, compilation and
WAV encoding are excluded.

The [2026-11-06 review audit](../../build/review-kokoro-w8a8-20261106/evidence/audit.json)
withdraws the earlier **1.65-second denominator and RTF 0.913 achievement**.
Even the claimed decoder-only 1,506.24 ms would yield **RTF 1.076** for this
fixture; its exact observation was not found in the retained reports. The
2,569.90 ms baseline was an inference-profile **whole request**, not decoder
Execute. Mixing those boundaries and profiles does not support the earlier
**41.4% reduction, 1.71x throughput or per-stage speedup table**.

| Retained observation, 2026-11-06 | Scope and samples | Whole-request latency | RTF, 1.4-second audio |
| --- | --- | ---: | ---: |
| [Initial inference run](../../build/review-kokoro-w8a8-20261106/evidence/baseline-inference-request.json) | One independent process, one warmup, one measured request | 2,569.909 ms | 1.836 |
| [Full-profile review run](../../build/review-kokoro-w8a8-20261106/fresh-full-benchmark-result.json) | Before the QGEMM correction below; one independent process, one warmup, median of three internal repeats | 1,585.796 ms | 1.133 |

These are historical observations with corrected scope, not a controlled A/B
comparison. Internal repeats are not independent trials. Operator/stage trace
samples and percentiles must not be added to obtain whole-request latency or P95.

The implementation retains SIMD128 4×4 transposed quantization/dequantization,
vector Sin/Cos, contiguous broadcast and Expand copies, and paired packed-QGEMM
unrolling. The QGEMM multiplier restores **`F32(F32(input_scale * weight_scale)
/ output_scale)`**, removing the reassociation that divided input by output
scale first. Accumulator scaling and output-zero-point addition retain their
separate F32 rounding boundaries. The incorrect reassociation produced **5
instead of 4** at a halfway case and **127 instead of 8** for a finite-scale
intermediate-overflow case at `M5/K16/N8`.

Numerical qualification is separate from timing. The retained pre-correction
W8A8 waveform differed from an earlier W8A8 output by **RMSE 0.043338821,
maximum absolute difference 0.516633719, and 11,101/33,600 strict mismatches**.
Against the unchanged ORT reference, the earlier and pre-correction outputs had
RMSE **0.090531323** and **0.089861995**, respectively. These observations do not
establish a perceptual-quality regression or isolate a kernel's contribution;
they also do not establish output equivalence. The F32 Kokoro known issue
`KOKORO-STRICT-001` (short CPU RMSE **0.005994636**), W8A8 quantization differences
and the multiplier bug are distinct. Kernel checks or faster execution do not
turn the original strict waveform comparison into a pass.

The [new matched comparison](../../build/fix-kokoro-w8a8-20261106/benchmark/summary.json)
uses one AMD Ryzen 5 5600U host on Linux x86_64, Node.js **v24.19.0**, and
single-threaded SIMD128 in the **full profile**. The [environment record](../../build/fix-kokoro-w8a8-20261106/benchmark/environment.json)
retains runtime versions and artifact hashes. A is the
[baseline source copy](../../build/fix-kokoro-w8a8-20261106/baseline/summary.json)
with exactly six numerical kernel files restored from the git index; B retains
the SIMD changes with the corrected F32 multiplier order. Both use identical
JS, model, input and reference assets. WASM hashes are A `7c39d4bc…` and B
`ed4f0d9a…`. This measures the SIMD change set together, not the isolated timing
effect of the multiplier correction.

The order is **A/B/B/A/A/B/B/A**, with **four independent processes per
variant**. Each process records its first execution and two warmups separately,
then measures **five completed requests**. Values below are the median of the
four process medians; no stage times are summed.

| Variant | Whole request: median of process medians | RTF, 1.4-second audio |
| --- | ---: | ---: |
| A: full WASM before the SIMD changes | 2,579.166 ms | 1.842 |
| B: SIMD full WASM with corrected F32 order | 1,583.915 ms | 1.131 |

Whole-request latency falls **38.59%** in this sample, but **RTF < 1 remains
unachieved**. The [trial records](../../build/fix-kokoro-w8a8-20261106/benchmark/trials.json)
retain inherited CPU affinity, load, and available frequency/governor readings.
The inherited affinity was CPUs 0–11 and the governor was `powersave`; no
pinning or frequency change was applied. Thermal-zone entries were absent, so
temperature was unrecorded and is not assumed stable. This covers one short request, one CPU and four processes per variant;
it does not establish latency for other lengths, voices, speeds or devices,
or a request P95. ORT timing and spinning policy are outside this comparison.
A [separate stage profile](../../build/fix-kokoro-w8a8-20261106/stage-profile-result.json)
retains encoder, decoder and operator diagnostics. Trace timings and a successful
profiler exit do not replace whole-request timing or waveform qualification.

The [corrected waveform comparison](../../build/fix-kokoro-w8a8-20261106/waveform-comparison.json)
still finds A/B output differences: **RMSE 0.044365839, maximum absolute
difference 0.638194591, and 11,106/33,600 strict mismatches**. Within each
variant, the four retained process-output hashes agree. B remains a strict
FAIL against the original ORT reference: **RMSE 0.090532522 and 22,031/33,600
mismatches**. The reproducible multiplier bug is fixed; the remaining whole-model
output difference still needs separate localization and W8A8 quality evaluation.
It is not classified as the existing F32 known issue or justified by speed.

The correction was rebuilt for [both native profiles](../../build/fix-kokoro-w8a8-20261106/native-build.json),
[both JS profiles with typecheck](../../build/fix-kokoro-w8a8-20261106/js-build.json)
and [both WASM profiles](../../build/fix-kokoro-w8a8-20261106/wasm-build.json).
The fresh WASM fixture passes [27 symmetric I8 regression cases](../../build/fix-kokoro-w8a8-20261106/quant/summary.json)
covering signed halfway rounding, intermediate overflow with finite scales,
nonzero pair packing, MR tails, N8 admission and K panel boundaries.
[Dispatch tests](../../build/fix-kokoro-w8a8-20261106/wasm-baseline.log) pass **10**,
[API tests](../../build/fix-kokoro-w8a8-20261106/proto-api.log) **317 with two skips**,
[WASM training](../../build/fix-kokoro-w8a8-20261106/wasm-training.log) **7** and
[PTQ](../../build/fix-kokoro-w8a8-20261106/wasm-ptq.log) **48**. Relevant native
CTest cases pass **17**, with **four hardware-dependent skips**.

The [full release check](../../build/fix-kokoro-w8a8-20261106/release-check.log)
passes size, provenance, ABI and inference/full boundaries for **all eight
artifacts**. The [approved additional size adoption](../../build/fix-kokoro-w8a8-20261106/size-adoption.json)
changes only the native lite measured ceilings: raw **2,401,800→2,401,832 B
(+32 B)** and gzip **1,085,745→1,085,766 B (+21 B)**. Existing JS/WASM ceilings
remain unchanged, with no additional headroom. The preceding five adoption
records now reference frozen measurements and prior-budget snapshots with
SHA-256 hashes. The [final size report](../../build/fix-kokoro-w8a8-20261106/size-final/release-size.json)
binds each artifact to its build evidence.

## Kokoro W8A8 WASM follow-up: RTF below one

The [final matched A/B](../../build/rtf-kokoro-w8a8-20261106/benchmark/summary.json) compares the corrected
baseline WASM from the preceding section `ed4f0d9a…` with final `a9530e39…`. Both use the same final
JS, prepared model, input and reference. This is the AMD Ryzen 5 5600U Linux
host, Node v24.19.0, full profile, single-threaded WASM SIMD128. The request is
`Hello!`, `af_heart`, speed 1, with an F32 encoder and static INT8 decoder:
33,600 samples at 24 kHz, **1.4 seconds**. The measured boundary includes
prepared host-input binding, encoder, duration feedback, decoder and completed
host PCM; loading, compilation and WAV encoding are excluded.

The order is **A/B/B/A/A/B/B/A**, four independent processes per variant.
Each records first execution and two warmups separately, then measures five
requests. These are medians of the four process medians, not summed stages.

| Variant | Whole request | RTF |
| --- | ---: | ---: |
| Corrected baseline | 1,545.000 ms | 1.104 |
| Final follow-up | **1,292.929 ms** | **0.924** |

The observed reduction is **16.32%**. All four final process medians are
below real time, spanning 1,269.939–1,317.183 ms.
Baseline process medians are 3,481.324 / 1,546.502 / 1,543.497 / 1,541.219 ms.
The slow first baseline process remains included; its cause is unknown.
[Environment and trials](../../build/rtf-kokoro-w8a8-20261106/benchmark/trials.json) retain inherited affinity,
load and available frequency/governor observations. There was no forced pinning
or frequency change; temperature was unavailable. One short request on one host
does not qualify Android/ARM, browsers, other lengths/voices/speeds or request
P95. ORT timing and spinning policies were not compared.

Three implementation changes preserve arithmetic and existing fallback routes:

- Immutable F32 Linear/LSTM weights receive engine-owned SIMD panels. Current
  inputs and recurrent state are read afresh; time steps, gate functions,
  ascending-K accumulation and F32 rounding remain unchanged. Mutable/training
  routes keep the ordinary kernel. Cache invalidation and shutdown release the
  panels. Newly reachable GEMM tail bias reads and signed-zero initialization
  were also corrected.
- Symmetric I8 temporal QConv directly fills the existing 80 KiB MR5 panel from
  current NHWC input, avoiding full im2col allocation/copy. The shared MR5/N8
  dot/requantization handles partial output channels, with live-lane bias,
  scale and output access. Shape, padding, dilation, odd-channel and overflow
  proofs remain; admission conservatively retains the old 128 MiB/full-im2col
  fallback bound, including when the direct path runs.
- Proved immutable-weight Expand→binary edges omit materialization: 48 encoder
  and 181 decoder nodes, or 332,271,720 logical written bytes per request.
  Operand order and separate Mul/Add rounding remain unchanged. Arena capacity
  stays 540,844 / 40,140,564 B. The 272 decoder activation-source candidates
  remain materialized pending lifetime proofs. Logical bytes are not measured
  DRAM traffic or evidence of bandwidth saturation.

[Final inventory](../../build/rtf-kokoro-w8a8-20261106/f32/final-model-inventory.json) reports an additional
61,326,096 B of encoder panels/metadata and 7,340,144 B for the decoder.
The per-engine 64 MiB cap is an implementation bound, not a measured optimum.
Admission counts potential eligible packs independently of allocator success;
repeated uses can conservatively saturate the bound while actual storage is
smaller. Store/descriptor/panel allocation failure injection verifies this
[review correction](../../build/rtf-kokoro-w8a8-20261106/memory-review.json).

The [separate node traces](../../build/rtf-kokoro-w8a8-20261106/stage-summary.json) are diagnostics, not request
latency or per-change causal speedups. They show encoder Linear
83.525→42.486 ms, encoder LSTM
58.150→11.238 ms, decoder LSTM
74.128→11.247 ms, QConv
900.382→849.089 ms and Expand
64.741→16.758 ms.
[QConv micro results](../../build/rtf-kokoro-w8a8-20261106/qconv/summary.json) retain shape-specific comparisons,
five slower rejected tile variants and instrumented pack/dot/requantization
costs. The large OC22 gain includes SIMD tail handling, not just im2col removal.
No new small-shape threshold is justified by the limited W31/1×1 samples.

All eight [preserved process PCM outputs](../../build/rtf-kokoro-w8a8-20261106/waveform-comparison.json) are
byte-identical to the corrected W8A8 baseline (`1b45ce32…`). Separate Piper and
1.65-second F32 Kokoro requests also retain byte-identical output. The W8A8
original ORT comparison nevertheless remains **strict FAIL: RMSE 0.090532522,
22,031/33,600 mismatches**. This is distinct from the existing F32 known issue.
Neither faster execution nor unchanged output establishes ORT parity or
perceptual quality; model/reference/tolerances remain unchanged.

Final native inference/full, JS and WASM builds pass. F32 fixtures cover
200 Linear and 48 LSTM cases with 21,046 exact-bit checks and allocation-failure
bounds; QConv covers the existing 27 affine cases plus 13 direct shapes with
two dynamic inputs and rejected-path fallback. Public API checks pass **347**
with two skips, including both-profile Expand **18** and F32 **12** cases.
Training **7**, PTQ **48**, WASM dispatch **10**, and relevant native checks
**17** with two hardware skips pass. Native GEMM edge tests reproduce the old
signed-zero failure and pass the corrected baseline/AVX2 paths.

The [full release check](../../build/rtf-kokoro-w8a8-20261106/release-check.json) passes provenance, exact ABI,
profile boundaries and size for all eight artifacts. The approved
[size adoption](../../build/rtf-kokoro-w8a8-20261106/size-adoption.json) changes only seven failing byte cells,
with no future headroom. WASM gzip ceilings become **550,802 / 799,023 B**;
browser-pair gzip **591,706 / 939,455 B**; native lite raw/gzip
**2,403,704 / 1,087,454 B**; native full gzip **2,685,052 B**. Original budgets
and failing measurements remain preserved. The [final evidence index](../../build/rtf-kokoro-w8a8-20261106/final-summary.json)
binds these reports to their artifacts. Staging was preserved; no add, commit
or reset was performed.
