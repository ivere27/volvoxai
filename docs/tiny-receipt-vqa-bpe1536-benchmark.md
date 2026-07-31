# TinyReceiptVQA BPE1536 CPU and WASM benchmark

- Status: current local performance and correctness report
- Release: VolvoxAI `0.3.0`

This report covers the BPE1536 TinyReceiptVQA split encoder/decoder release at
[`ivere27/tiny-receipt-vqa-structured-qa-21m`](https://huggingface.co/ivere27/tiny-receipt-vqa-structured-qa-21m).
It compares ONNX Runtime CPU, VolvoxAI native CPU, and VolvoxAI WASM on one
content-addressed real receipt with full generation. It is a deployment-path
comparison on one machine, not an isolated-kernel or general hardware ranking.

ONNX Runtime below means the native `CPUExecutionProvider`. This repository
does not depend on `onnxruntime-web`, so the report does not invent an ONNX
Runtime Web/WASM result. The WASM rows are VolvoxAI's strict WASM backend.

## Executive result

Every measured run used the release-owned `receipt_en.jpg`, the question
`What is the store phone number?`, AUTO routing, batch one, and generation to
EOS or 191 new tokens. Every run selected `phone` and produced exactly:

~~~text
<field>phone</field><op>identity</op><answer>6533831</answer>
~~~

Primary latency results are medians. Session/model construction is excluded
from the three compute rows; native process wall and WASM cold generation are
shown separately.

| Runtime / deployment path | Timed scope | Samples per precision | FP32 ms | INT8 ms | FP32 / INT8 |
| --- | --- | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU, 1 thread | encoder + dynamic-prefix generation | 51 | 326.074 | 207.982 | 1.568x |
| VolvoxAI native CPU, 1 thread | encoder + retained-row generation | 11 | 888.641 | 438.217 | 2.028x |
| VolvoxAI native CPU, 1 thread | complete fresh process wall | 11 | 1,726.983 | 904.366 | 1.910x |
| VolvoxAI WASM, single-thread | hot retained-row generation | 5 | 16,223.514 | 1,610.283 | 10.075x |
| VolvoxAI WASM, single-thread | cold retained-row generation | 5 | 18,061.239 | 3,059.640 | 5.903x |

For the directly timed compute/generation rows, ONNX Runtime remains fastest:

| Precision | ONNX CPU | VolvoxAI native CPU | VolvoxAI WASM hot | Native / ONNX | WASM / ONNX | WASM / native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 326.074 ms | 888.641 ms | 16,223.514 ms | 2.725x | 49.754x | 18.257x |
| INT8 | 207.982 ms | 438.217 ms | 1,610.283 ms | 2.107x | 7.742x | 3.675x |

These cross-runtime ratios are not pure kernel comparisons. ONNX uses a
dynamic decoder prefix, while native and WASM seed their fixed decoder once,
retain dependency/KV state, and execute one new row for every later token.
WASM hot generation and native compute both exclude graph loading and
compilation; the native complete-process row shows the additional startup
boundary separately.

## Model and measured artifact identities

The producer contract is `tiny_receipt_vqa_split_onnx_v1`: `d_model=320`,
eight heads, six encoder layers, four decoder layers, input
`[1,1,320,672]`, question/decoder length 192, memory length 402, and vocabulary
size 1536.

The tokenizer is `byte_fallback_bpe` v1 with NFC normalization, 256 byte
fallback tokens, 517 ordered merges, and fingerprint
`5801826ac9092bdc984210af805d2ff20f0398b80adaf5963ef13b04771a8584`.

| Producer artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `4094a28d67604c646b035e9eb398fa7537090279686b09dd2b09105de057b066` |
| `config.json` | `e18e266da200b9b915628c1b8f633cb523520287e2d8f75c8093e3f4d6575f7e` |
| `vocab.json` | `ae19bf56556649d75e72450ff6c7d43895190404c15dffeb6412dce5cc4d2c80` |
| FP32 encoder ONNX | `0c6469ef75eeb266c3dbc77c414d658f98f704264b72bdc7dfeed9cf37217db7` |
| FP32 decoder ONNX | `3fff897777fa5e9805c01a6779812f4adf33e1294f4e2003dd823c04a3300b05` |
| INT8 encoder ONNX | `924a60cfa8172456131aedb2e672af2ad1de7e4fa975bf14f4837a49e09cec94` |
| INT8 decoder ONNX | `24dbf7411719fea928f1f249db9b981eedf4b3af6751bec0970f242ca3f99d0a` |

The measured VolvoxAI packages were generated afresh from those exact files:

| Package | Measured manifest SHA-256 | Encoder / decoder nodes | Decoder ABI |
| --- | --- | ---: | --- |
| Optimized FP32 | `c9e45078fb88868459a088aa1263688be31cf33b3242d76769f0eee1f66a93fe` | 197 / 128 | F32 logits, host first-index argmax |
| Canonical INT8, native copy | `c68550ccf2a2b0271f4aeb473c9933944d19ad032f79ca77ade3cd205fbe18b5` | 327 / 235 | I32 token IDs, in-graph `QArgMax` |
| Canonical INT8, WASM copy | `19fbd163ace123c720dfbe0a9818edf2c248abf6349c05f198ccf45ed40d4aac` | 327 / 235 | I32 token IDs, in-graph `QArgMax` |

The two INT8 manifest hashes differ only through non-executable optimizer
report fingerprints. Their runtime graph and weight assets are byte-identical:

| Runtime asset | SHA-256 |
| --- | --- |
| FP32 encoder graph / weights | `a2600527441d58c57ee49a078c74b8527f31d91b05822c433ce7eaea9e8ee71b` / `21be9f3c4611b3ee6585ef5d655c30c4cac866ce46953a8a7ed4a0ef8994eb0e` |
| FP32 decoder graph / weights | `544efd83ad73d31fe84de892a22c154a1442ce5c0616d65c68c6b29c18690403` / `ae4e6026903c70ec6e8fc64687429612d4c338f075d8095c2dd9ee13fc480a7b` |
| INT8 encoder graph / weights | `a89ae9e30499c9bda07fd6f675cb0f680da6387cfa6f1f2d93cd4a90354815a7` / `46af98fcd96d9644f5b8e0ee9886024c551f0a9effe6cf8590454fd6a09d3a18` |
| INT8 decoder graph / weights | `7410f71098e3c82763e4f0ea54bfd23c11ae104b73f1fc583dcc10352a6014a9` / `8f9bc097aab3b2573b4d166b54203baeae8984ea5b7df689f7ed6e5832e757e8` |

The direct INT8 package remains correctly classified as `hybrid`: quantized
regions use the producer's U8S8 affines while required floating-point
boundaries remain F32.

## Test machine and measurement policy

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 5 5600U, 6 cores / 12 hardware threads, AVX2/FMA |
| OS | Linux 6.8.0-124-generic x86-64 |
| VolvoxAI source | commit `290dd3db8a948157fb2ee221cfb6d72dc0f288a9` plus the runtime changes hashed below |
| ONNX Runtime | 1.23.2, `CPUExecutionProvider` only |
| Python / NumPy | 3.10.12 / 2.2.6 |
| Node.js | 20.11.1 |
| Native benchmark executable | 1,684,488 bytes, SHA-256 `29aa411313430e78701f822a8fe54c8af46e0478cdfd183a484a362becaf287c` |
| Native inference / full release executables | 1,389,640 / 1,970,320 bytes; SHA-256 `c218ec78ec2d84ac25d32122005212e6657685b6a1129f39f3db4e1fdd919149` / `3b91b4e69b0e5c08202fb6686e06b426c19cf0692b41f1fe22d359f06ec7e5e9` |
| WASM inference / full executables | 199,583 / 311,002 bytes; SHA-256 `99d6ac675cfc26d922c2a971603b3cdc020ae08a22c1978f8ceb332b0d286212` / `4a6da8ece1a53cf79a545f6472e845e0d42d76e429dfe7296b6f2b0c1fdd0fed` |
| CPU affinity | logical CPU 2 for every timed process |
| Frequency policy | `amd-pstate-epp`, `powersave`; boost/frequency scaling left enabled |

Runtime-affecting dirty-source identities were stable before and after the
final benchmark campaigns:

| Source | SHA-256 |
| --- | --- |
| `attention_f32_core.h` | `d59e929027cd2d61bccc33c1611bb1cb4ac26a43d44ca4cbd4af5e503f191db2` |
| `attention_f32_isa.h` | `7e6e6e00e91cdec024215aefbacf3930a2846966830b3d1c2aede0f3eca606c6` |
| `attention_f32_tiled.h` | `7ca61137e2213d225b7f87c1dbfa89db54763d313c7663f3a7a1dc17a12bdb41` |
| `cross_sdpa.c` | `97c0cca4ff8cb7e8a757af2cb2de838358cb55cad157458c11ee64f1c111bb60` |
| `sdpa.c` | `7076cb8c01689fb64b92cf49a98472f2a235e1f8f3f9b22ef83930d23cfd188b` |
| `packed_quant_gemm.c` / `.h` | `67481de6b3b80989c33744a4ed1e1646ca482b9e31139154057b600f5203b1bb` / `748d1b2294563eabab94750f5a61a7448fc3a2d3450c9ea96384b5e808adae08` |
| `qlinear_w8a8_wasm_relaxed.c` | `19d598eb12b1a32906f46d4338162e4736b50a1166db6541702b7d0b08ac211f` |
| `TinyReceiptSplitSession.js` | `2a2a384d8c10693cb3d25431778d862701806c4b1949d9c212baad29611544ae` |
| `CPUEngine.ts` / `WasmEngine.ts` | `e32ef16621920fe29e65b2af99f6677896a9d859f49b8f2854c397fafd9d92df` / `97992427c146eb24f8d540244b76594a1d05307c1662fac7225ac1ed19f6367d` |
| `quantizedRowExecution.ts` | `89b8922949eaf77958fba10347cff9390c33f4a0ef5d613431da55b7263b12a7` |

All paths were explicitly single-threaded:

- ONNX Runtime used `ORT_SEQUENTIAL`, `intra_op_num_threads=1`, and
  `inter_op_num_threads=1`.
- Native used `--cpu --threads 1 --incremental --require-row`.
- WASM required one full incremental seed followed by retained-row execution,
  was structurally single-threaded, and also received `VOLVOXAI_THREADS=1`.
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` were set to one for the
  ONNX and native runs. The WASM runs additionally set `BLIS_NUM_THREADS=1`
  and `OMP_DYNAMIC=FALSE`.

No other build or benchmark was allowed to overlap a timed campaign. CPU
frequency scaling and ordinary desktop services remained enabled, so small
differences should still be treated as noise.

## ONNX Runtime CPU result

Sessions, image preprocessing, and tokenization were completed before timing.
Each sample contains one encoder call and 16 dynamic-prefix decoder calls
through EOS. Ten complete warmups per precision were discarded, followed by
51 measured generations.

| Precision | Encoder median | Decoder median | Total median | Total mean | Total p05-p95 | Session creation, excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 216.519 ms | 109.712 ms | 326.074 ms | 326.944 ms | 322.341-333.282 ms | 272.246 ms |
| INT8 | 141.323 ms | 66.282 ms | 207.982 ms | 208.190 ms | 203.223-213.239 ms | 117.090 ms |

INT8 is 1.532x faster in the encoder, 1.655x in the decoder, and
1.568x end to end. All 102 measured generations were deterministic and FP32
and INT8 produced identical token IDs.

## VolvoxAI native CPU result

The native executable was built in a fresh CMake directory. Before timing,
the packed-GEMM, incremental-runtime, backend-composition, inference/full
profile-boundary, training-boundary, and real TinyReceipt example checks passed
6/6. Eleven fresh processes were measured per precision using one five-repeat
FP32-first campaign and one six-repeat INT8-first campaign.

`compute` is paired encoder plus generation time and excludes package
load/compile. `process wall` covers the complete fresh process. The median of
paired totals is reported, so it need not equal the sum of component medians.

| Precision | Encoder | First token | Steady token | Generation | Compute | Process wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 593.617 ms | 145.954 ms | 9.754 ms | 296.270 ms | 888.641 ms | 1,726.983 ms |
| INT8 | 364.178 ms | 59.364 ms | 0.937 ms | 73.801 ms | 438.217 ms | 904.366 ms |
| FP32 / INT8 | 1.630x | 2.459x | 10.410x | 4.014x | 2.028x | 1.910x |

Dispersion was narrow:

| Precision | Compute p05-p95 | Process wall p05-p95 |
| --- | ---: | ---: |
| FP32 | 860.491-899.074 ms | 1,689.738-1,744.927 ms |
| INT8 | 433.866-448.584 ms | 898.104-920.133 ms |

The retained-row decoder is especially effective for INT8: after the first
token, its median steady step is 0.937 ms. Native INT8 generation is therefore
close to ONNX Runtime's entire 66.282 ms dynamic-prefix decoder total
(73.801 versus 66.282 ms), even though native encoder and process-level
overhead keep total latency higher.

## VolvoxAI WASM result

Five fresh Node processes were measured per precision, with package order
alternated by round. Each process performed one `cold-graphs` generation and
then one `hot-graphs` generation. All ten fresh processes compiled successfully
with `mode=require`, `backend=wasm`, and `operatorFallback=forbid`; all 20
cold/hot generations completed without a reported tier or operator fallback.

Every generation executed exactly one full decoder seed and 15 retained rows;
no ordinary full-decoder step was accepted. The real optimized FP32 decoder
prepared all 102/102 selected dependency nodes, and the canonical INT8 decoder
prepared all 185/185.

`hot decoder total` is the directly recorded sum of seed and row execution
times inside each process, then medianed across processes. `hot generation` is
the directly measured application wall time and is the primary total.
Displayed component medians are independently medianed and therefore need not
add exactly.

| Precision | Cold generation | Hot generation | Hot encoder | Decoder seed | Retained row | Hot decoder total | Graph load | Compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 18,061.239 ms | 16,223.514 ms | 15,794.937 ms | 286.676 ms | 7.770 ms | 404.644 ms | 626.723 ms | 581.052 ms |
| INT8 | 3,059.640 ms | 1,610.283 ms | 1,292.071 ms | 229.222 ms | 6.023 ms | 321.354 ms | 324.348 ms | 762.123 ms |
| FP32 / INT8 | 5.903x | 10.075x | 12.225x | 1.251x | 1.290x | 1.259x | 1.932x | 0.762x |

Hot-generation dispersion:

| Precision | Mean | Sample SD | p05-p95 | Min-max |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 16,182.246 ms | 203.167 ms | 15,920.727-16,382.823 ms | 15,859.975-16,416.703 ms |
| INT8 | 1,609.970 ms | 8.343 ms | 1,599.122-1,617.646 ms | 1,596.631-1,618.128 ms |

The retained-row/KV change is visible in the decoder rather than the encoder:

| Precision and scope | Before retained row | Current | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: |
| FP32 steady decoder step | 266.697 ms | 7.770 ms | 34.324x | 97.09% |
| FP32 decoder total | 4,263.401 ms | 404.644 ms | 10.536x | 90.51% |
| FP32 hot generation | 18,592.195 ms | 16,223.514 ms | 1.146x | 12.74% |
| INT8 steady decoder step | 213.784 ms | 6.023 ms | 35.495x | 97.18% |
| INT8 decoder total | 3,410.801 ms | 321.354 ms | 10.614x | 90.58% |
| INT8 hot generation | 4,579.629 ms | 1,610.283 ms | 2.844x | 64.84% |

FP32 end-to-end improves less because its 15,794.937 ms encoder dominates the
remaining time. INT8 compilation is 1.312x slower than FP32 compilation
because the canonical hybrid graph is larger, partially offsetting its lower
graph-load and inference costs.

### Relaxed-SIMD validation and ablation

The optional M=1 Relaxed-SIMD child now consumes the exact V8Q2 packed-weight
header emitted by the parent. The shipped inference and full WASM artifacts
passed shared-memory, final-dot-opcode, all eight input/weight/output signedness
combinations, corrupt-header rejection, and fail-closed fallback tests.

This makes the path correct and usable, but it is not the source of the
end-to-end gain above. In the validation microbenchmark, Relaxed-SIMD measured
0.0228 ms, baseline packed SIMD128 0.0130 ms, and portable W8A8 0.1152 ms for
the 320x320 M=1 case. A separate five-pair real-INT8 ablation found essentially
the same application behavior with child-first and baseline-only dispatch:
steady-row medians were 5.676 versus 5.537 ms, while hot-generation medians
were 1,502.227 versus 1,497.412 ms. The small, inconsistent differences are
within the overlapping run-to-run distributions. Because the isolated child
also lost, current dispatch tries the faster packed SIMD128 kernel first and
uses the optional child only as a fail-closed row fallback. The measured
application speedup is therefore attributed to retained-row/KV execution, not
to the optional child.

## Why ONNX Runtime remains faster

The timing breakdown localizes most of the native gap to the encoder. Relative
to ONNX Runtime, the VolvoxAI native encoder is 2.742x slower in FP32
(593.617 versus 216.519 ms) and 2.577x slower in INT8 (364.178 versus
141.323 ms). The native INT8 retained-row generation path is already close to
the ONNX dynamic-prefix decoder total: 73.801 versus 66.282 ms, 1.113x slower.
Native process wall also includes graph loading, compilation, and host startup
that the compute row and ONNX timing exclude.

The likely implementation-level explanation is that ONNX Runtime's mature
graph optimizer, memory planner, and vectorized CPU operator kernels execute
this model's large dense encoder more efficiently. The current benchmark does
not contain per-operator ablations, so this is an inference from the component
timings rather than a causal kernel proof. ONNX is not intrinsically faster as
a format; these numbers compare the measured runtime implementations and their
different execution plans.

The earlier WASM structural disadvantage has been removed: native and WASM
now both retain dependency/KV state and execute one later decoder row. The
remaining native-versus-WASM difference is:

| Precision | Native compute | WASM hot generation | WASM / native compute | Native fresh-process wall | WASM hot / native fresh wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 888.641 ms | 16,223.514 ms | 18.257x | 1,726.983 ms | 9.394x |
| INT8 | 438.217 ms | 1,610.283 ms | 3.675x | 904.366 ms | 1.781x |

The compute-to-hot column is the closest available direct comparison. The last
two columns deliberately mix scopes—hot preprocessed WASM generation versus a
complete fresh native process—and are startup context, not a wall-to-wall
runtime ratio.

For INT8, the WASM encoder is 3.548x slower than native (1,292.071 versus
364.178 ms) and its retained-row decoder total is 4.354x slower (321.354
versus 73.801 ms). Seed is 3.861x slower, and each steady row is 6.428x slower.
The likely residual causes are native x86 AVX2-specialized quantized kernels
and lower native dispatch/memory overhead versus V8's single-threaded
SIMD128/WASM execution. This is a component-level inference, not a per-operator
profile.

FP32 is more extreme because its WASM encoder alone takes 15,794.937 ms,
26.608x the native FP32 encoder. Its retained-row decoder total is only 1.366x
native, so further FP32 WASM work should target the encoder's large dense and
attention operations rather than decoder caching.

## Interpretation and change from the previous report

The earlier revision of this document used one hot same-example observation
for several paths. The current report uses medians, fixed CPU affinity, explicit
thread environment variables, and order balancing for the native and WASM
campaigns. ONNX Runtime used one complete FP32 block followed by one INT8 block.
Direct deltas include that methodology change, but the native FP32 difference
is much larger than the observed variance:

| Path | Previous FP32 | Current FP32 | Previous / current | Current reduction |
| --- | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU | 355.764 ms | 326.074 ms | 1.091x | 8.35% |
| VolvoxAI native CPU compute | 9,196.749 ms | 888.641 ms | 10.349x | 90.34% |
| VolvoxAI WASM hot generation | 22,206.836 ms | 16,223.514 ms | 1.369x | 26.94% |

The native FP32 gain is consistent with the current shared online-softmax
attention implementation and x86 tiled AVX2/FMA path, which remove repeated
Q.K work and let full and retained-row execution share the optimized
recurrence. This campaign is not a controlled code-path ablation, so the full
gain cannot be attributed to those changes alone.

INT8 remains the best VolvoxAI deployment choice on this machine. Native INT8
cuts compute by 50.7% versus native FP32; WASM INT8 cuts hot generation by
90.1%. ONNX Runtime is still faster than VolvoxAI native for both precisions,
especially in the encoder.

## Correctness status and limitations

- ONNX Runtime: 102/102 measured generations passed; FP32 and INT8 token IDs,
  family, structured text, and answer were identical.
- Native: 22/22 measured fresh processes reached EOS, retained row execution,
  selected `phone`, and returned the exact answer.
- WASM: 20/20 cold/hot generations passed strict WASM execution with exact
  output and no reported operator fallback.
- This is a one-example latency report. It does not replace the release's
  2,000-record accuracy evaluation or a multi-case latency distribution.
- ONNX uses a dynamic prefix. Native and WASM both use a full seed plus
  retained rows, but their graph boundaries and runtime implementations still
  differ. The end-to-end rows describe shipping deployment paths, not
  identical operator workloads.
- WASM required retained-row execution and rejected ordinary decoder fallback.
  Dirty noncausal K/V/mask, invariant matrix, or broadcast inputs fail before
  input/output writes, invalidating the seed cache instead of reusing unsafe
  state.
- The optional Relaxed-SIMD M=1 QLinear child is now V8Q2-compatible and
  correctness-tested. Its real-model ablation was neutral within noise, so it
  is not credited for the reported KV speedup.
- No ONNX Runtime Web/WASM number is included because `onnxruntime-web` is not
  a project dependency in this repository.

## Content-addressed measurement evidence

The raw reports were retained in local temporary benchmark directories during
this run. Their digests identify the exact evidence used for this document:

| Evidence | SHA-256 |
| --- | --- |
| ONNX Runtime 51-sample report | `3e3101de86538291dbed00fdfdf6bdf94c79267cc1119b3f85b604bb8b5faa3d` |
| ONNX timing harness | `dc8cafdf832277460dd38427fa8ff683c273ce49ca29b3cbe79605416c230431` |
| Native report, FP32-first five samples | `3fcb65e54818db976673962063bebcade1237428403b5f4ef73272fb4d317395` |
| Native report, INT8-first six samples | `693c2f32550a27f3daa9b8f01b2064c8d9869d9232065a18b0d9f3af793114d8` |
| WASM five-round ordered raw-log checksum listing | `9ac6a235c21578a3efd1d83734226be3bf047ed7581740f89e5f84371b93894b` |
| Relaxed/baseline SIMD five-pair checksum listing | `8a34fafe534e4584b9d0adc66a60721fe8843965a348d32420ae93f1e50590eb` |
| Strict FP32 / INT8 smoke logs | `d888f0924f4dc762e1bbc8b1fc4433baaaeeec4050fdd569c12e2f71dfaf8553` / `c1f2d15c3ddaaeac21e3d6c31d5a3fdbb134ec58c1980a0d6993b205c221f804` |
| Versioned WASM benchmark harness | `859575a0c083373c01c831d88c8c72fa8296c566b53cd4153d83e42fea1e5ef2` |
| Receipt image | `24d7dde7d28bec844fee9134e6cc8d3180ee85315d6d493cc39c7cf0e1ebf328` |

The ordered-listing digests are the SHA-256 of the textual `sha256sum`
listing in round order (`round-{1..5}-{fp32,int8}` and
`ablation-{1..5}-{relaxed,baseline}` respectively), so all ten constituent
logs are covered without selecting a single favorable run.

## Representative commands and aggregation protocol

Use new output directories; package tools deliberately refuse to overwrite an
existing artifact. The commands below reproduce package construction and the
native/WASM runtime invocations. The published aggregate values use the sample
counts and order shown below, rather than the result of any single command.

~~~bash
MODEL_SOURCE=/path/to/tiny_receipt_vqa_bpe1536_onnx
EVAL_ROOT=/path/to/eval/heldout
FP32_IMPORTED_PACKAGE=build/tiny-receipt-bpe1536-fp32-imported
FP32_PACKAGE=build/tiny-receipt-bpe1536-fp32
INT8_IMPORTED_PACKAGE=build/tiny-receipt-bpe1536-int8-imported
INT8_PACKAGE=build/tiny-receipt-bpe1536-int8

python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source "$MODEL_SOURCE" --out-dir "$FP32_IMPORTED_PACKAGE" \
  --variant fp32 --target portable
python3 -m examples.tiny_receipt_vqa.tools.optimize_split_fp32 \
  --source "$FP32_IMPORTED_PACKAGE" --out "$FP32_PACKAGE" \
  --hoist-input v4_keep:int32

python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source "$MODEL_SOURCE" --out-dir "$INT8_IMPORTED_PACKAGE" \
  --variant int8-w8a8 --target portable
python3 -m examples.tiny_receipt_vqa.tools.optimize_direct_int8 \
  --source "$INT8_IMPORTED_PACKAGE" --out "$INT8_PACKAGE" \
  --fuse-attention --fuse-static-qdq-compute --canonical-deployment
~~~

Build the relevant VolvoxAI runtime and opt-in example targets:

~~~bash
make build_native
make build_tiny_receipt_split_native_example
make test_wasm_relaxed_simd
npm run build:all
~~~

Apply the single-thread and affinity policy before timing:

~~~bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE VOLVOXAI_THREADS=1
BENCH_CPU=2
~~~

Reproduce the native 5+6 campaign. The second command reverses package order;
pool the eleven per-run observations for each precision before calculating the
median and linear-interpolated p05/p95 values:

~~~bash
taskset -c "$BENCH_CPU" python3 -m \
  examples.tiny_receipt_vqa.tools.benchmark_native_heldout \
  --eval "$EVAL_ROOT" \
  --binary examples/target/bin/tiny_receipt_split_w8a8 \
  --ids receipt_en \
  --package fp32="$FP32_PACKAGE" --package int8="$INT8_PACKAGE" \
  --max-new 191 --repeat 5 --threads 1 \
  --report build/tiny-receipt-native-fp32-first.json

taskset -c "$BENCH_CPU" python3 -m \
  examples.tiny_receipt_vqa.tools.benchmark_native_heldout \
  --eval "$EVAL_ROOT" \
  --binary examples/target/bin/tiny_receipt_split_w8a8 \
  --ids receipt_en \
  --package int8="$INT8_PACKAGE" --package fp32="$FP32_PACKAGE" \
  --max-new 191 --repeat 6 --threads 1 \
  --report build/tiny-receipt-native-int8-first.json
~~~

For WASM, each invocation below is one fresh Node process containing a cold/hot
pair. The call order matches the measured five-round campaign:

~~~bash
run_wasm() {
  local package_path="$1" label="$2" round="$3"
  taskset -c "$BENCH_CPU" \
    node --experimental-wasm-relaxed-simd --import tsx \
    examples/tiny_receipt_vqa/tools/benchmark_wasm_cold_hot.mjs \
    "$package_path" "$MODEL_SOURCE/examples/receipt_en.jpg" \
    'What is the store phone number?' \
    dist/0.3.0/volvoxai.wasm 191 \
    > "build/round-${round}-${label}.log"
}

run_wasm "$FP32_PACKAGE" fp32 1
run_wasm "$INT8_PACKAGE" int8 1
run_wasm "$INT8_PACKAGE" int8 2
run_wasm "$FP32_PACKAGE" fp32 2
run_wasm "$FP32_PACKAGE" fp32 3
run_wasm "$INT8_PACKAGE" int8 3
run_wasm "$INT8_PACKAGE" int8 4
run_wasm "$FP32_PACKAGE" fp32 4
run_wasm "$FP32_PACKAGE" fp32 5
run_wasm "$INT8_PACKAGE" int8 5
~~~

Extract the final `WASM_TINYRECEIPT_COLD_HOT` JSON record from every log and
calculate each reported median over the five observations for that precision.
The harness itself requires strict WASM selection, exact cold/hot output,
one decoder seed, 15 monotonically positioned retained rows, zero ordinary
decoder executions, and no operator fallback. It reports encoder, seed,
steady-total/mean, decoder-total, and application end-to-end times separately.

The campaign evidence digest can be reproduced without discarding any raw log:

~~~bash
cd build
sha256sum round-{1..5}-{fp32,int8}.log | sha256sum
~~~

For the SIMD ablation, compile the same parent without embedding the optional
custom section, then run five order-balanced INT8 pairs with the shipped and
baseline artifacts:

~~~bash
clang-17 --target=wasm32 -O3 -msimd128 -nostdlib \
  -Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
  -o build/volvoxai.baseline-simd.wasm native/src/kernels/kernels.c
~~~

The exact ONNX measurement used the one-off timing harness identified by hash
above; that harness is not currently checked into the repository. Its exact
protocol creates FP32 or INT8 sessions once with
`CPUExecutionProvider`, `ORT_SEQUENTIAL`, and both thread counts set to one;
preprocesses and tokenizes once; discards ten complete generations; and then
times 51 encoder-plus-dynamic-prefix generations. Session construction and
preprocessing stay outside the timed loop. Future release automation should
promote this harness into a versioned tool before treating the ONNX aggregate
as independently command-reproducible.
