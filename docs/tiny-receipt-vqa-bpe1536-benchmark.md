# TinyReceiptVQA BPE1536 explicit-KV benchmark

> **Historical measurement record.** The tables below remain bound to their
> recorded artifacts and machines. The private JavaScript object-API harnesses
> that produced several browser and batching rows have been removed; those rows
> are evidence, not current public-API instructions.

This document records only the current producer and package ABI:

```text
source:           tiny_receipt_vqa_split_kv_onnx_v2
source KV cache:  tiny_receipt_vqa_default_kv_cache_v2
package:          volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2
shape mode:       bounded-explicit-kv-v2
runtime KV prefill format: masked-zero-sentinel-v1
```

The current importer, retained independent benchmark tools, and tests accept
the v2 source/package/shape ABI only. There is no manifest downgrade or
compatibility path. The private-lifecycle native split driver named by older
reports has been removed.

## 2026-08-21 retained pre-final host observation

This section is the latest locally retained B1 observation available during
the publication audit. It is not a clean-tree validation of the current
commit. The longer device tables below are retained historical evidence and
are not relabelled as results from these bytes.

The source was the checksummed local release:

```text
/path/to/huggingface/
  tiny_receipt_vqa_structured_qa_d320_e6_d4_bpe1536_lora_router_direct_novalue_e100_onnx
source manifest SHA-256: c43bb3f34779e929d556f6e786890568f4542cd7d354d1292e7ecc6eac2bb828
```

The producer ONNX hashes are unchanged from the table below. The following B1
baseline artifacts were generated and measured before the final
encoder-attention normalization was made unconditional:

| Variant | Manifest | Nodes enc/dec | Encoder graph / weights | Decoder graph / weights |
| --- | --- | ---: | --- | --- |
| FP32 | `373c5e4b955b91b79a726571baf1368ee79c73d3a94589902610b0ee0aaf5f7b` | 357 / 170 | `b87015ba277664be27d74304af6e130a8a8636f8011cd8be31cd601f80a10f91` / `d74a1a6c61dcbd427e98b1f528725c61fd61cf29c3c21060f7c654b917e742bc` | `eac88bcbfbfa44e38d6ee98014abaeb2e974ff1d1c3fc7988108a72f9d2fca2e` / `1d07de964ec3cdcb814cef9e055fb2fabc43dc0f6da83794be23e747baceaa98` |
| INT8 | `be1b68bd3a4615bc8e6f61b6480c83d34cc9e8090243f83f9cc75511ba39d9ff` | 480 / 271 | `318c48ad1f1b5539f98c15d936f4116d8636fa1ed4ca3c1cf854eca20c1792a8` / `92527d2b977f403b2432a15a0cfe8900c13cfe9d2365a64d94e2232f523e9e6e` | `a464046ff0dc071de81d032982a90ed9379917fa1f4b7ab686a0556bd1c66555` / `f34313c5ffeb9db040c80156a1ce7ea7ec5f492f88a87be9c92e75ea131b73f7` |

### End-to-end B1 correctness and latency

The Ryzen 5 5600U host was pinned to CPU 0; ONNX Runtime, native C, and WASM
used one thread. One full request was warmed up and three were measured. Every
one of the 18 measured samples selected family 0, emitted tokens
`[4,1038,5,6]`, and advanced `P=1 -> R=2 -> 3 -> 4 -> 5` with strict backend
selection and no fallback. Dynamic grow/max-padded/shrink qualification also
passed native and WASM for both precisions.

| Route | FP32 component median ms | vs ORT | INT8 component median ms | vs ORT |
| --- | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU | **253.637** | 1.000x | **147.949** | 1.000x |
| VolvoxAI native CPU | 286.611 | 1.130x | 214.076 | 1.447x |
| VolvoxAI WASM | 1239.831 | 4.888x | 854.062 | 5.773x |

`component` is each sample's encoder plus four decoder executions. Model load,
compile, preprocessing, and process startup are excluded. These values replace
neither the historical tables below nor their different lifecycle contracts.

The saved B1 report is bound to native binary SHA-256
`926696111d4a961244171f973f8495ab8da4df302ee2e34fc88d71d19488b3c3`,
API SHA-256
`7f55755c00665338c29dc95b0b4f9a2095d80a2b477b2764c27d831cfb04a486`,
and WASM SHA-256
`ffaa402d483fe37cc709322411dfb3381b57bd6345df06d538a693d65237922d`.
The untracked local report had SHA-256
`df997b4c6438c73a44024970a61573f3c6911e3732ed106401bbd7cdebd15395`.
Its embedded repository provenance names base commit
`9939bbe54597bf1c1ff541d37f552122355baaaf` and a dirty worktree. The report is
not committed publication evidence and cannot validate commit
`50cdf1af157d7272714d775c72303cf3b487daac`; that commit requires a new run.
That equivalent new-run command path used a removed Python wrapper around the
removed private-API JS runtime harness and is not reproduced in the current
checkout.

The mutable `dist/0.4.0/volvoxai.js` path has since been rebuilt; at publication
audit time its SHA-256 was
`1a3a74ce22be83e75ab32ca7de46dc1dc1663141d2befe35d2438d4676a3163d`,
not the report-bound API digest above. Running this command against the current
path creates a new measurement; it does not retroactively reproduce the saved
report.

### Opt-in B=1..2 component packages

`--max-batch-size 2` preserves the producer's leading B symbol. The encoder
importer rewrites only the exact qualified attention layout: Q/K/V 18, mask 1,
key transpose 6, and output projection 6 patterns. It keeps `[B,H,M,D]`
explicit instead of treating `B*H` or `B*M` as unrelated atomic dimensions.
Unknown consumers, shapes, permutations, shared edges, quantization axes, or
generated-name collisions reject the transaction.

| Variant | Import wall / peak RSS | Manifest | Nodes enc/dec | Encoder graph / weights | Decoder graph / weights |
| --- | ---: | --- | ---: | --- | --- |
| FP32 | 22.84 s / 683,828 KB | `a918e290820486f58b5e298575b663823ce796e22d15d4b907a877a0d1cee986` | 325 / 170 | `d933ee0090e3b85f5ce20bac3affa3c1ef98e454b9720978e1780175eb60bd6e` / `2f59b172b98a6b63007aa0f27e3676e77b66cfaba55e2ef996c7e2a9a8f6ce25` | `38aced0011d9af03acf7f10830775ebd20ed3f56f3c1bad96217f1a317582530` / `1d07de964ec3cdcb814cef9e055fb2fabc43dc0f6da83794be23e747baceaa98` |
| INT8 | 3:40.17 / 397,340 KB | `259e8dd9e4fcfa4ab7d5009f660c44bde04f9ed72a51b13d8536c6b64dade6c0` | 472 / 271 | `f21d8a6fb6c02d44ce0fa31c5fceb65a61cec8a342347622e2b4ac5dc83ba5bc` / `50d9297db830398cf7bc729ba05345baa92f5a4ca5fff0b84ba30bba69114972` | `da9cdae135af9843af5bf34a99792b61dd3742ae190c3af74bfc6bcb2dc48fd5` / `f34313c5ffeb9db040c80156a1ce7ea7ec5f492f88a87be9c92e75ea131b73f7` |

Manifest and both graph domains are exactly `B={min:1,max:2}`. Core's
`typed-independent-batch-proof/v1` covers every node: FP32 encoder/decoder
325/170 and INT8 encoder/decoder 472/271. A separate direct compile audit tied
the evidence to these exact canonical graph-fingerprint SHA-256 values:

| Component | Covered nodes | Canonical graph-fingerprint SHA-256 |
| --- | ---: | --- |
| FP32 encoder | 325 / 325 | `d553ed0dd3e5700d55d6ab1712a75d69fe00a0b7637ac11861d2afcbe0a6ea9a` |
| FP32 decoder | 170 / 170 | `6b1b5267c06eebb463701a11d0b8e74f0ebad02979067370fd1aa8f15a0c7e56` |
| INT8 encoder | 472 / 472 | `a532af23a3d8a7ffafc958b22d08f6cf9bf1fa414d09325d5e0ca8fca953eba5` |
| INT8 decoder | 271 / 271 | `84e56c96ac7600c486ac21f9b2fc56194c6dfea3d00f9a46eda0b768564eb379` |

The B2 packages and timed reports are untracked external audit artifacts. The
digests in this section identify the private ledger but do not make the inputs
available from a clean checkout.

The removed harness serialized a compact proof record. The four timed reports
below predate that field; a later, separate zero-warmup INT8-decoder rerun
confirmed its serialization. The four fingerprints above came from direct
loads of the exact package graphs.

Before package export, distinct-lane original-vs-normalized ONNX authoring
parity passed all short/representative/maximum/history cases. FP32 maximum
absolute error was `1.90735e-6`; INT8 was bit-exact.

### Physical B2 Runtime proof

The measurements in this section and the later WebGPU tables predate the
execution-mode rename. Immutable report labels retain their original
vocabulary: `SIMPLE` maps to DIRECT, `ADAPTIVE` maps to SCHEDULED with zero
batch delay, and `SERVICE` maps to SCHEDULED with a positive bounded delay.
Current public applications use only the proto-defined DIRECT/SCHEDULED modes;
no old plan-name aliases exist. The removed direct MJS worker wrote
`volvoxai.tiny-receipt-vqa-runtime-batches/v2`; its Python ORT wrapper wrote
`volvoxai.tiny-receipt-vqa-runtime-batch-audit/v2`. Both archived contracts
record the selected mode, scheduler size/delay, redacted artifact identities,
and exact parity outcome. Private raw tensor directories and machine paths were
never publication artifacts.

The audit procedure runs two independent B1 fixtures, then submits those
same fixtures concurrently. It reads every encoder output or decoder
logits/mask/eight present K/V outputs, writes their full bytes, and records both
Runtime execution diagnostics and additive per-result scheduling reports.

On WASM, FP32/INT8 encoder/decoder all had this exact result under both zero-
and positive-delay SCHEDULED configurations:

```text
2 logical requests -> 1 provider graph invocation
observedBatchSizes = [2]
dispatchCount = 1
additive trueBackendInvocations = 1
scheduled lane bytes = independent same-backend B1 lane bytes
```

As a fail-closed boundary, the default max-B1 FP32 packages were also run with
DIRECT and both SCHEDULED delay configurations. Every mode/configuration produced two physical graph
invocations for two logical lanes and no false batch report.

The timing rows below are **one warmup plus one measured group in fixed order**.
They qualify direction only; they are not a counterbalanced performance study.

| Component | Two independent B1 ms | Physical B2 ms | Group latency change | B2 throughput change |
| --- | ---: | ---: | ---: | ---: |
| FP32 encoder | 1855.683 | 1926.825 | +3.83% | -3.69% |
| FP32 decoder | 32.102 | 52.668 | +64.07% | -39.05% |
| INT8 encoder | 770.028 | 838.249 | +8.86% | -8.14% |
| INT8 decoder | 50.299 | 43.431 | **-13.65%** | **+15.81%** |

Only the INT8 one-token decoder improved. A legal B2 is therefore not a
portable production choice; the measured per-route/device `T(B)` selector in the
[scheduling design](scheduling-and-dynamic-batching-design.md#legal-b-operating-b-and-padding)
remains required.

Same-backend B1/B2 invariance is exact, but reference fidelity is a separate
gate. The FP32 Runtime comparison against independent original ORT B1 had max
absolute differences `0.0057725` (encoder) and `1.00136e-5` (decoder), within
the recorded mixed `atol/rtol=0.01` and `1e-4` gates. INT8 already differs on
independent Runtime B1: observed full-output maxima were `2.2094` for encoder
memory/KV and `0.6405` for decoder logits/KV. The two fixture argmax values and
the end-to-end token smoke still match, but that does **not** qualify INT8 full
numerical fidelity. The retained historical report status records this as
`runtime_batching_passed_ort_numerical_difference_recorded`.

That historical status is not current success semantics. The removed wrapper
recorded `failed_ort_reference_tolerance` and exited nonzero after saving the
report whenever either route missed the ORT tolerance gate; same-backend
batching invariance alone could not qualify a run.

The measured engine artifacts were `dist/0.4.0/volvoxai.js` SHA-256
`fc856cd21998d62d6fac221275db557f50e11f1b12ded909d3de0fdf8a3e61e3`
and `volvoxai.wasm` SHA-256
`ffaa402d483fe37cc709322411dfb3381b57bd6345df06d538a693d65237922d`.

These retained reports were produced by a removed private-API JS harness. They
remain valid archived evidence, but the current checkout does not remeasure
them from the public proto surface.

For the retained INT8 package described above, the archived audit failed the
ORT tolerance gate after writing its diagnostic report. Any future generated-
proto replacement must exit successfully and report `passed` before a
regenerated package qualifies.

This is component proof only. The historical JS session harness owned private
contexts and serialized each explicit-KV autoregressive session at B1; it did
not batch work across sessions.

### RTX 3090 Deno WebGPU B4/B8 component proof

The retained hash-bound WebGPU sweep used Deno 2.9.3 over Vulkan on an NVIDIA
GeForce RTX 3090 (vendor `4318`, device `8708`, driver `535.309.01`). It was
current at measurement time, but is not a validation of the present checkout.
The measured API was `dist/0.4.0/volvoxai.js` SHA-256
`3337273dd7b9e87e5e865f457f6b602f4769e84225ee21f28480f852e33f49b1`.
The packages were freshly imported with `--max-batch-size 8`; B8 is the
producer and package contract maximum, not a claim about an RTX 3090 hardware
lane limit. A B greater than 8 is illegal for these source artifacts and was
therefore not benchmarked.

The exact B8 package and typed-independence proof identities were:

| Variant | Component | Manifest SHA-256 | Graph / weights SHA-256 | Typed proof coverage / graph fingerprint SHA-256 |
| --- | --- | --- | --- | --- |
| FP32 | Encoder | `fb20832602cc1757ff1410f59d7e3e7525a5ce55fdad24a3a0c4ca69566dd337` | `a33d2f6b085da272b5547154aa9efd8a837ace14f0483afbbb10132422e0df23` / `2f59b172b98a6b63007aa0f27e3676e77b66cfaba55e2ef996c7e2a9a8f6ce25` | 325 / 325, `ab55247f05249dea6f546620cbc501a8149001340be28d8631580d320b602adf` |
| FP32 | Decoder | `fb20832602cc1757ff1410f59d7e3e7525a5ce55fdad24a3a0c4ca69566dd337` | `c9dc928732c77670619da15d8f9e3f0ff89e541674762a70d6cd7df71c6a6f8c` / `1d07de964ec3cdcb814cef9e055fb2fabc43dc0f6da83794be23e747baceaa98` | 170 / 170, `f9958caba2abd31ff6577b4c18d91c52e920c0eec9e3ad52ca23d9dc5a546241` |
| INT8 | Encoder | `bb8c0838f5247abcab75377d4026265550962624c6b90bd04a0947cbbcd5cdac` | `c43935b12c8d6751d24f239abd983280aaf8707a4785a08c2cf6beca8336316a` / `50d9297db830398cf7bc729ba05345baa92f5a4ca5fff0b84ba30bba69114972` | 472 / 472, `d199c9bc7f811a9179566e0c1ecafdc5bf40b08b006e4203c2c4ff6b74545900` |
| INT8 | Decoder | `bb8c0838f5247abcab75377d4026265550962624c6b90bd04a0947cbbcd5cdac` | `b3b9a7192ae853738c10b27b8a4d79a5eb8ee8af0956e17bfe85534aae9b8c5a` / `f34313c5ffeb9db040c80156a1ce7ea7ec5f492f88a87be9c92e75ea131b73f7` | 271 / 271, `ed4207bf41d4b3d9f9c212a4dca7ae7094cbea89f5a1ea5838cf92a3c12e7b57` |

Each route used one warmup group followed by five measured groups. The
independent route submits the same distinct fixtures through DIRECT and makes
N physical B1 graph invocations. The scheduled route submits those fixtures
through SCHEDULED and makes one physical B=N graph invocation. The medians are
therefore whole-group latencies, including all required output readbacks and
result close; `speedup` is independent-group median divided by scheduled-group
median.

The next table retains the retired SIMPLE label for DIRECT reference runs; the
scheduled column is the active SCHEDULED semantic.

| Variant | Component | B | Independent retired SIMPLE B1 group ms | Scheduled B group ms | Speedup | Same-backend max abs / rel |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | Encoder | 4 | 715.239 | 226.844 | **3.153x** | `5.84126e-6` / `6.83582e-6` |
| FP32 | Encoder | 8 | 1282.119 | 316.801 | **4.047x** | `7.62939e-6` / `7.62361e-6` |
| FP32 | Decoder | 4 | 218.852 | 84.390 | **2.593x** | `8.58307e-6` / `2.68457e-3` |
| FP32 | Decoder | 8 | 311.744 | 147.503 | **2.113x** | `9.05991e-6` / `2.68457e-3` |
| INT8 | Encoder | 4 | 711.418 | 252.909 | **2.813x** | `0` / `0` |
| INT8 | Encoder | 8 | 1301.312 | 355.037 | **3.665x** | `0` / `0` |
| INT8 | Decoder | 4 | 234.617 | 110.830 | **2.117x** | `0` / `0` |
| INT8 | Decoder | 8 | 337.868 | 194.603 | **1.736x** | `0` / `0` |

For every B4 row the five measured groups produced 20 logical requests,
20 DIRECT invocations, and five scheduled invocations. Every B8 row produced
40 logical requests, 40 DIRECT invocations, and five scheduled invocations.
The scheduled reports independently record `observedBatchSizes=[4]` or `[8]`,
`dispatchCount=5`, and `trueBackendInvocations=5`: each group is physically
N-to-1, rather than merely N concurrent promises. Full encoder outputs and
decoder logits, masks, and all present K/V tensors were compared lane by lane.
INT8 is bit-exact; the FP32 maxima are recorded per row above.

This is a same-WebGPU-backend batching-invariance result. It does not compare
against original ONNX Runtime and does not close or relax the separate
original-ORT numerical-fidelity boundary recorded in the B2 section above.
Likewise, it is component batching only: the removed historical JS session
harness owned private contexts and serialized each autoregressive session at
B1.

The original command path used a removed private-API Deno worker and is kept
here only as archived measurement provenance.

The report root was
`/path/to/volvoxai-gpu-validation/results/webgpu-3337273dd7b9`.
The immutable report suffixes and SHA-256 values are:

| Report suffix | SHA-256 |
| --- | --- |
| `vqa-fp32-encoder-b4/runtime-report.json` | `796ffae6e935c529cbb1870106ae53a9c167609201f794eb7306774dc55ac18c` |
| `vqa-fp32-encoder-b8/runtime-report.json` | `2927242ccef00a8bd2ebd7225f033c6d66234b0bfe2ee8b54442a76014caefff` |
| `vqa-fp32-decoder-b4/runtime-report.json` | `dba8e09b6f35c412dc49dd8a753a9f86754e06ce0e36536d1a576c8026c57d0d` |
| `vqa-fp32-decoder-b8/runtime-report.json` | `d28fc3d23273018845940ae3e156b829f8587260468e45a9be64869b20158331` |
| `vqa-int8-encoder-b4/runtime-report.json` | `fb1151c56d2063ac429a5ce885e3dfd4ca9d10f94aa885b8c643089b35a1a31c` |
| `vqa-int8-encoder-b8/runtime-report.json` | `f57f83ce3e58389503ceb9a69908569003062d4e0eb409e7fd8b5a45a997117f` |
| `vqa-int8-decoder-b4/runtime-report.json` | `8ac026dbb09fbc52504b075b6a3c9d131863349f1082eae2ca0222000a706940` |
| `vqa-int8-decoder-b8/runtime-report.json` | `65e32a90532f8ac518b9bad5ee357810a6decb79b021b1460a03599547b8ed3e` |

Those reports, their distinct-lane fixture, and the B8 packages are not
tracked in this repository. The paths and hashes above describe an external
archive ledger, not a clean-checkout reproduction source. Publication-grade
remeasurement requires restoring those exact inputs or generating and hashing
new ones, then publishing the resulting provenance separately.

### RTX 3090 native Runtime component coalescing (producer maxB8)

The native internal Runtime benchmark path was measured separately on Vulkan,
OpenGL, and CUDA.
DIRECT makes eight independent B1 calls; SCHEDULED submits eight B1 requests
and returns them from one physical B8 execution. This is distinct from the
Deno WebGPU route above and from an explicit authored-B8 bulk call.

B8 is not a native harness, wrapper, importer, or backend lane limit. The
current producer manifest declares max B=8, and the importer may not widen a
producer-owned shape domain. The importer itself adds no separate B8 ceiling,
and the native benchmark accepts case-driven B values covered by the selected
package. Testing B16 and above for this model therefore requires a new producer
artifact whose manifest legally declares that larger domain.

The report retains the wrapper's
`same-package-dynamic-batch-sweep-only` comparison policy. With only one measured
B in this report, it proves artifact consistency across backends but does not
claim a multi-B scaling curve.

The canonical public report is
[`native_dynamic_batch_rtx3090.json`](../examples/tiny_receipt_vqa/reports/native_dynamic_batch_rtx3090.json)
(SHA-256
`8492466d677c03c4d5145ccb2e3948476fcd0e635bb33be2788d61339ce1577c`),
measured at `2026-08-22T13:01:32Z`. It binds source base commit
`064e30d2923351401e3fb654863905e2dd77a533` and exact build-relevant source
snapshot SHA-256
`d0a3f2a8da0f8664659a26754ce33d1352a5fd580fd0750eaf1bf1a42975f4a9`.
The report records `sourceDirtyWithinSnapshotScope=true`, so the evidence is
for that exact snapshot, not the base commit alone. The Release/O3/NDEBUG
harness SHA-256 is
`0be73f6af7f2f1d230eebb82257fdf271642ae53b6c28110d642a70f2d08d02c`
and the build contract SHA-256 is
`38f998e0337cefdf4f4cbdb1405c256acbddb3c9f80c1849469c949923f475a8`.
The report also retains the exact compiler, strict-no-FMA `sm_86` CUDA, native
shader compiler, locked Naga 30.0.0, host, RTX 3090, and driver evidence.

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

The wrapper re-attested the exact maxB8 artifact bytes, declared package
domain, and compiled provider contract for every component:

| Variant | Component | Manifest SHA-256 | Graph SHA-256 | Weights SHA-256 | Typed proof identity |
| --- | --- | --- | --- | --- | --- |
| FP32 | Encoder | `fb20832602cc1757ff1410f59d7e3e7525a5ce55fdad24a3a0c4ca69566dd337` | `a33d2f6b085da272b5547154aa9efd8a837ace14f0483afbbb10132422e0df23` | `2f59b172b98a6b63007aa0f27e3676e77b66cfaba55e2ef996c7e2a9a8f6ce25` | `typed-independent-batch-proof/v1:fnv1a64:8613f2e82d50f7e4` |
| FP32 | Decoder | `fb20832602cc1757ff1410f59d7e3e7525a5ce55fdad24a3a0c4ca69566dd337` | `c9dc928732c77670619da15d8f9e3f0ff89e541674762a70d6cd7df71c6a6f8c` | `1d07de964ec3cdcb814cef9e055fb2fabc43dc0f6da83794be23e747baceaa98` | `typed-independent-batch-proof/v1:fnv1a64:203fb42ea32b3135` |
| INT8 | Encoder | `bb8c0838f5247abcab75377d4026265550962624c6b90bd04a0947cbbcd5cdac` | `c43935b12c8d6751d24f239abd983280aaf8707a4785a08c2cf6beca8336316a` | `50d9297db830398cf7bc729ba05345baa92f5a4ca5fff0b84ba30bba69114972` | `typed-independent-batch-proof/v1:fnv1a64:a096e63b0400ce02` |
| INT8 | Decoder | `bb8c0838f5247abcab75377d4026265550962624c6b90bd04a0947cbbcd5cdac` | `b3b9a7192ae853738c10b27b8a4d79a5eb8ee8af0956e17bfe85534aae9b8c5a` | `f34313c5ffeb9db040c80156a1ce7ea7ec5f492f88a87be9c92e75ea131b73f7` | `typed-independent-batch-proof/v1:fnv1a64:6d2b4d192b974a57` |

These are actual graph and weight bytes but deliberately small synthetic
component inputs. The harness generates eight deterministic, distinct lanes
with seed `20260821`. Encoder lanes use image `[1,1,320,672]` and the graph
domain minima Q=1 and M=211. Decoder lanes use M=211, P=1, and R=2. Q=1/M=211
is not the application's BOS/EOS-valid Q=2/M=212 request, and neither row
includes tokenization, image preprocessing, autoregressive session policy, or
end-to-end decoding. The table is therefore a graph-minimum component smoke,
not production-request or end-to-end VQA performance.

DIRECT timing starts immediately before the first `vx_runtime_run` and ends
after the eighth owned result returns. SCHEDULED timing starts immediately
before the first `vx_runtime_submit`, submits all eight requests before
waiting, and ends after the eighth `vx_request_result` returns its owned
result. Both include binding validation, backend input upload, provider
dispatch, device synchronization, and result snapshot publication. SCHEDULED
additionally includes owned submission snapshots, aggregate input stacking,
and the aggregate-to-B1 output split. They exclude `vx_result_read`, parity,
request/result release, and JSON writing. Routes alternate per iteration,
starting with DIRECT on even iterations. The WebGPU table includes output
readback/result close, so cross-table latency ratios and backend rankings are
invalid.

Every case used three warmup groups and five measured groups, one CPU thread
pinned to CPU 0, a 100 ms collection guard, Vulkan arena `1024` MiB, and CUDA
ordinal 0. Values are whole-group milliseconds shown as median [min,
nearest-rank p95, max], rounded decimal half-up to three places. Speedup is
DIRECT median divided by SCHEDULED median; scheduled req/s is the aggregate
over every measured sample.

The 1024-MiB Vulkan arena is this campaign's reproducibility setting, not an
engine or RTX 3090 capacity ceiling. Native Vulkan, OpenGL, and CUDA have no
fixed total-memory or batch-count clamp; overflow-checked sizes, each backend's
kernel/index ABI, queried allocation and launch limits, and actual device
allocation govern acceptance. The current package's producer-owned maxB8
domain is the only reason this component matrix stops at B8. The WebGPU
128-MiB binding limit discussed elsewhere does not apply to these native paths.

| Variant | Backend | Component | DIRECT independent B1 group ms | SCHEDULED B8 group ms | Speedup | SCHEDULED req/s | Same-backend max abs / rel |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | Vulkan | Encoder | 1171.543 [1168.191, 1172.276, 1172.276] | 320.986 [314.966, 321.583, 321.583] | 3.650x | 25.011 | `5.7220459e-6 / 5.08320936e-6` |
| FP32 | Vulkan | Decoder | 121.182 [120.387, 123.680, 123.680] | 28.193 [26.697, 29.203, 29.203] | 4.298x | 285.647 | `1.14440918e-5 / 3.10550916e-3` |
| INT8 | Vulkan | Encoder | 1215.542 [1213.125, 1217.848, 1217.848] | 358.557 [357.875, 359.669, 359.669] | 3.390x | 22.303 | `0 / 0` |
| INT8 | Vulkan | Decoder | 139.254 [138.355, 142.038, 142.038] | 30.742 [30.408, 31.810, 31.810] | 4.530x | 259.361 | `0 / 0` |
| FP32 | OpenGL | Encoder | 2074.054 [2072.662, 2075.190, 2075.190] | 366.687 [366.061, 367.349, 367.349] | 5.656x | 21.823 | `5.7220459e-6 / 5.48033291e-6` |
| FP32 | OpenGL | Decoder | 132.448 [129.945, 132.827, 132.827] | 27.344 [27.190, 28.769, 28.769] | 4.844x | 289.030 | `8.58306885e-6 / 3.62689097e-3` |
| INT8 | OpenGL | Encoder | 2338.927 [2327.050, 2346.438, 2346.438] | 427.277 [425.135, 427.946, 427.946] | 5.474x | 18.736 | `0 / 0` |
| INT8 | OpenGL | Decoder | 172.744 [170.394, 173.388, 173.388] | 35.922 [35.683, 37.296, 37.296] | 4.809x | 219.870 | `0 / 0` |
| FP32 | CUDA | Encoder | 258.440 [257.235, 258.683, 258.683] | 198.031 [196.574, 198.842, 198.842] | 1.305x | 40.438 | `0 / 0` |
| FP32 | CUDA | Decoder | 31.321 [31.101, 31.434, 31.434] | 18.345 [18.185, 18.509, 18.509] | 1.707x | 436.067 | `0 / 0` |
| INT8 | CUDA | Encoder | 237.727 [236.304, 239.990, 239.990] | 166.691 [166.148, 168.343, 168.343] | 1.426x | 47.867 | `0 / 0` |
| INT8 | CUDA | Decoder | 40.724 [40.652, 41.050, 41.050] | 19.580 [19.522, 19.767, 19.767] | 2.080x | 408.205 | `0 / 0` |

Every measured row contains 40 logical requests: DIRECT records 40 physical
invocations and SCHEDULED records five. Each scheduled group has one shared
nonzero process-local execution ID, physical batch size 8, exactly one true
backend invocation, and a new ID in the next group. Strict route attestation
records no missing or fallback nodes. Parity includes warmup and measured
groups: encoder compares 768 tensors / 38,905,600 elements and decoder compares
640 / 426,112. INT8 is exact. FP32 passes finite combined allclose with
`atol=rtol=1e-4`; a relative maximum above `1e-4` can still pass for near-zero
values when its absolute error satisfies the combined criterion. Every Vulkan
physical forward additionally attests a device-local 1-GiB compute arena, a
32-MiB host-visible staging buffer, and successful staged upload/download
operations. Every measured CUDA physical forward attests a successful cached
CUDA Graph launch and stream synchronization. All 12 measured rows speed up,
but this remains a graph-minimum component result rather than an end-to-end VQA
claim.

The canonical output contains no machine paths, raw backend logs, synthetic
input/output digests, or decoded receipt values. Its physical execution IDs
are process-local counters used only to prove N-to-1 execution. The matrix
input remains local because it selects external package paths; the synthetic
lanes themselves are regenerated by the harness and need no fixture files.

Reproduction uses an exact Naga 30.0.0 executable, the current locked native
shader compiler, and this producer-maxB8 12-case matrix:

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
  --config "$RUN/vqa-native-matrix-input.json" \
  --output "$RUN/vqa-native-dynamic-batch.json"
sha256sum "$RUN/vqa-native-dynamic-batch.json"
```

The input schema is `volvoxai.native-dynamic-batch-matrix-input`. The wrapper
does not require B8: it derives each workload's batch set from the configured
cases, requires the full Vulkan/OpenGL/CUDA x FP32/INT8 cross-product at every
configured B, verifies executed B is within the selected package domain, and
requires one exact package per precision/component across a multi-B sweep.
This campaign selects B8 because the producer manifest stops there, with
warmup/repeat 3/5, delay 100 ms, and Vulkan arena 1024 MiB. The wrapper rejects
inherited performance/device overrides and injects the validated Vulkan arena
and CUDA device settings into child processes. Only the path-free canonical
output is publication-safe. A clean checkout still needs the four external
hash-bound maxB8 package components to remeasure.

## Earlier published artifacts

The packages are imported directly from the producer ONNX directory. ONNX
Runtime and every VolvoxAI route use explicit F32 KV caches.

See the [TinyReceipt example](../examples/tiny_receipt_vqa/README.md) for the
import and runtime contracts.

The table below describes the earlier publication bytes, not the fresh section
above.

| Package | Manifest SHA-256 | Encoder nodes | Decoder nodes |
| --- | --- | ---: | ---: |
| FP32 | `5c2c32eee248f0991f30e6c2688ea458792cc28f07712593307f961cc937a816` | 409 | 236 |
| INT8 | `be1b68bd3a4615bc8e6f61b6480c83d34cc9e8090243f83f9cc75511ba39d9ff` | 480 | 271 |

Exact producer ONNX hashes:

| Model | SHA-256 |
| --- | --- |
| FP32 encoder | `0e2206c54756b15d697d2bbd8d22bb68d0d2475d0323965ca03a14fb186e6eb0` |
| FP32 decoder | `b918a9de531bad205990acff28a2a7e2244b14229c312ed6b2f75d0540d6a6a2` |
| INT8 encoder | `a5be30f7507f8e2988c4cf735a80fb6c09a4f1e93c9759b1fc5e12d04c4da45c` |
| INT8 decoder | `75437beed7cdac93b4a704e9a4ffbefbe0199855344a57cb8f78ae105e915419` |

The INT8 graph is hybrid W8A8: large convolution, linear, GEMM, and batch
matrix multiplication use byte kernels, while public memory, logits,
KV caches, normalization, softmax, GELU, and GroupNorm/SiLU islands remain
F32. `complete_w8a8_fusion=false` is therefore the correct manifest state.

| Runtime operator | Encoder | Decoder |
| --- | ---: | ---: |
| `QConv2D` | 13 | 0 |
| `QLinear` | 26 | 33 |
| `QGemm` | 8 | 0 |
| `QBatchMatMul` | 14 | 18 |
| F32 `Conv2D` / `Linear` / `Gemm` / `MatMul` / `BatchMatMul` | 0 | 0 |
| `QuantizeLinear` / `DequantizeLinear` | 62 / 61 | 53 / 48 |
| `GroupNorm` / `SiLU` | 13 / 13 | 0 / 0 |
| `LayerNorm` / `Softmax` / `GELU` | 12 / 6 / 8 | 13 / 8 / 5 |

The exporter migrated all four decoder BatchMatMul regions previously left at
the cache boundary; the decoder has 18 physical `QBatchMatMul` nodes and no F32
`BatchMatMul`. It also folded 26 encoder and 33 decoder immutable F32 biases
into I32 quantized accumulators. Direct redundant `DequantizeLinear ->
QuantizeLinear` round trips are absent. Public memory, cross/self KV caches,
logits, LayerNorm, Softmax, GELU, and encoder GroupNorm/SiLU islands remain F32
by policy.

The generic GroupNorm/SiLU byte-island migration remains opt-in and is not
enabled for this package. A local 32-request qualification tied to producer
INT8 encoder/decoder SHA-256 values `a5be30f7...` / `75437bee...` changed greedy
tokens in 3 of 32 requests, with up to 32.46% pre-SiLU saturation and 39.66%
terminal byte mismatch. This local qualification supports rejecting that
migration but is not a corpus-level accuracy claim.

## Workload and timing

All routes use the same generated 672x320 grayscale input, normalized F32 input
SHA-256 `7a6f7eb153434868b1685c4fd96fc63f1004cae356d15bd58404dccdc2c15963`,
prompt `phone number last one`, requested family `phone`, active shape
`B=1/Q=8/M=218/T=5`, and greedy four-token decode.

Every measured route must select family 0, emit `[4, 1038, 5, 6]`, and preserve
the explicit-cache transition `P=1 -> R=2 -> 3 -> 4 -> 5`. The initial `P=1`
row is the blocked zero sentinel. A route that fails this validation is not a
benchmark result.

`Component` is the median of each sample's encoder plus four decoder
executions; it is not formed by adding independently computed column medians.
The existing host reports and the current Android report use different
lifecycle and synchronization contracts. Those contracts are stated in their
own sections, so their rows remain publication evidence without being
presented as one directly interchangeable timing matrix.

## Existing host benchmark results

The CPU, AMD GPU, and RTX 3090 tables below are the existing host publication
evidence and are not Android measurements. Their immutable report filenames
and report schemas retain the `explicit_kv_v1` identifier used when those
reports were recorded. That identifier labels the evidence record; it is not a
runtime or package compatibility path. The current importer, packages, and
runtime accept only the v2 ABI declared at the top of this document.

The host timing reports are bound to manifest digests
`875d9f907dfd1544387761fa44e83d1c9e9508b7a50f4d5dd8b2c05018e4a580`
for FP32 and
`7eb277431c0779634b02f54641feb46847ed8c581493bbd7d2822753234db47f`
for INT8. They remain attached to those immutable report records and are not
relabelled as measurements of the current v2 manifest bytes. The current v2
artifact hashes and Android results are recorded separately; no manifest
downgrade or compatibility claim is made between the two report generations.

### Host timing contract

Host encoder and decoder timings include execution, synchronization, shape
binding, and owned output snapshotting where the runtime exposes those phases.
Model loading, context/session creation, graph compilation, image
preprocessing, tokenization, and process startup are excluded. Process-wall
time is therefore not an inference-latency comparison for these tables.

The CPU and GPU reports intentionally answer different lifecycle questions:

- The CPU report measures fresh-session/runtime first execution. `--warmup 1`
  discards one complete matrix, but every measured ORT session and native/WASM
  child is new. Before warmup it runs an untimed canonical dynamic-rebind proof
  for native CPU and WASM at both precisions.
- The AMD GPU report performs one untimed full request on the same runtime and
  encoder/decoder contexts immediately before every measured request. It resets
  the cache to the sentinel and proves token/cache parity, removing lazy device
  initialization and first-shape compilation from the comparison.

The CPU and AMD GPU harnesses counterbalanced provider order with deterministic
forward/reverse pairs and rotation, and recorded the exact order for every
matrix. Child processes remove uppercase `VOLVOX*` overrides and force common
nested-library thread variables to the requested count: one for the one-core
and GPU matrices, six for the six-core matrix. The browser runners use fresh
origins/profiles and serve every artifact with `Cache-Control: no-store`.
VolvoxAI preloads both graphs and contexts before warmup; ORT fetches each ONNX
model once into immutable bytes before creating both sessions.

### CPU: one physical core

The measured host is an AMD Ryzen 5 5600U with Node.js 20.11.1. The process is
pinned to one logical CPU that maps to one physical core. ONNX Runtime and
native C use one requested execution thread. VolvoxAI WASM has one engine
thread and zero workers.

| Runtime | Precision | Encoder | Prefill | Steady / token | Decoder total | Component | x ORT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU | FP32 | 247.790 | 2.618 | 2.468 | 9.834 | 257.520 | 1.000 |
| VolvoxAI native C CPU | FP32 | 272.356 | 5.184 | 3.299 | 14.984 | 286.967 | 1.114 |
| VolvoxAI WASM | FP32 | 1095.780 | 50.177 | 60.533 | 231.775 | 1325.460 | 5.147 |
| ONNX Runtime CPU | INT8 | 147.688 | 1.570 | 1.004 | 4.583 | 153.290 | 1.000 |
| VolvoxAI native C CPU | INT8 | 188.372 | 4.320 | 4.011 | 16.350 | 204.812 | 1.336 |
| VolvoxAI WASM | INT8 | 588.990 | 60.477 | 61.127 | 244.959 | 839.373 | 5.476 |

ONNX Runtime remains faster at one thread because the thread count limits
parallel workers, not graph quality or microkernel quality. ORT optimizes and
packs during session creation outside the execution timer and uses MLAS
packing/cache blocking. Native C includes binding commit and exact output
snapshot work and, on this pre-VNNI CPU, uses an exact full-range U8/S8 AVX2
route that avoids `VPMADDUBSW` I16 saturation. The latter costs more arithmetic
than a reduced-range path but preserves the authored quantization contract.

The WASM encoder gap is predominantly provider compute, not just binding. The
FP32 encoder medians split into 18.235 ms binding and 1076.945 ms provider work;
INT8 splits into 40.029 and 548.631 ms. WASM uses one thread and SIMD128, while
native uses wider AVX2 plus a multithread-capable engine. Its decoder also pays
for each new `P/R` signature, JavaScript/WASM boundary work, and exact host
snapshots. Explicit KV is active in all rows, so the gap is not a missing-cache
failure.

### CPU: six-physical-core process envelope

This report pins `0,2,4,6,8,10`, which topology maps to six physical cores, and
requests six ORT/native threads. WASM remains one engine thread; wider affinity
only gives Node/V8 auxiliary threads more scheduling room and is not WASM
inference-worker scaling.

| Runtime | Precision | Encoder | Prefill | Steady / token | Decoder total | Component | x ORT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU | FP32 | 79.891 | 2.495 | 3.415 | 12.741 | 91.859 | 1.000 |
| VolvoxAI native C CPU | FP32 | 111.926 | 5.090 | 3.322 | 15.057 | 126.699 | 1.379 |
| VolvoxAI WASM | FP32 | 1048.073 | 25.434 | 22.522 | 93.427 | 1138.566 | 12.395 |
| ONNX Runtime CPU | INT8 | 62.889 | 1.811 | 2.565 | 9.506 | 72.395 | 1.000 |
| VolvoxAI native C CPU | INT8 | 95.437 | 4.351 | 3.902 | 16.080 | 111.517 | 1.540 |
| VolvoxAI WASM | INT8 | 465.993 | 31.331 | 28.683 | 117.395 | 584.616 | 8.075 |

From the one-core to six-core envelope, component latency improves 2.803x /
2.117x for ORT FP32/INT8 and 2.265x / 1.837x for native C. Three repeats are
enough to preserve a reproducible observation, not to characterize all host
noise or deployment tail latency.

### Physical AMD GPU observation

The AMD GPU report used the Ryzen 5 5600U host, Node.js 20.11.1, Google Chrome
151.0.7922.137, and a physical AMD Renoir GPU for Vulkan and OpenGL. The
isolated ORT and VolvoxAI browser processes both reported the normalized
WebGPU adapter class `{vendor: amd, architecture: gcn-5}`. WebGPU exposes no
stable physical-adapter identifier here, so the report attests the
adapter-class match, not that both processes opened the same physical adapter.

The direct WebGPU comparison is ONNX Runtime Web 1.27.0 with WebGPU plus CPU
partitions against strict VolvoxAI WebGPU. All values are median milliseconds.

| Runtime route | Precision | Encoder | Prefill | Steady / token | Decoder total | Component | x ORT Web |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime Web 1.27.0 WebGPU + CPU partitions | FP32 | 216.500 | 71.600 | 52.167 | 235.000 | 452.700 | 1.000 |
| VolvoxAI strict WebGPU | FP32 | 601.800 | 36.400 | 31.167 | 130.500 | 731.800 | 1.617 |
| ONNX Runtime Web 1.27.0 WebGPU + CPU partitions | INT8 | 1029.000 | 447.900 | 454.233 | 1813.200 | 2842.200 | 1.000 |
| VolvoxAI strict WebGPU | W8A8 | 615.900 | 51.500 | 55.800 | 219.100 | 832.300 | 0.293 |

ORT's optimized-session diagnostics report aggregate provider-assignment
counts, not exact per-operation CPU attribution: FP32 encoder CPU 70 / WebGPU
504 and decoder CPU 3 / WebGPU 244; INT8 encoder CPU 282 / WebGPU 1223 and
decoder CPU 133 / WebGPU 676. The pinned ORT Web 1.27.0 build cannot instantiate
any graph as strict all-WebGPU. Its operator table marks `Reshape` and `Shape`
as having no GPU kernel, and the captured missing-kernel events contain those
operations plus other shape/control operations. The model's INT64/BOOL
shape/control tensors explain additional type-constrained CPU islands. INT8
also has no registered WebGPU `QuantizeLinear` kernel. Repeated capability-
probe events are not counts of executed nodes, transfers, or partition
boundaries. VolvoxAI requires WebGPU and reports fallback zero.

The measured harnesses reused WebGPU-resident KV buffers and performed zero KV
readbacks in the timed request. ORT exposed eight cross-cache and eight present-cache outputs
as `gpu-buffer` tensors and passes those tensor objects forward. Its internal
transfers across WebGPU/CPU execution-provider partitions are not attested.
Separate untimed qualification requests read caches back and prove token and
cache-transition correctness.

The FP32 encoder uses thirteen groups=1 regular 3x3 convolutions, about 8.190
GMAC in total, and every output-channel count is divisible by 16. The browser
compiler selects the shared regular-out16 kernel, emitting 9,720 workgroups and
622,080 invocations instead of the scalar schedule's 155,520 workgroups and
9,953,280 invocations. All five measured FP32 samples record exactly thirteen
`webgpu.conv2d.regular-out16` encoder tactics. A physical WebGPU correctness
test covering asymmetric padding, stride, and dilation agrees with the CPU
reference within `2.98e-8`; bounded-domain tests execute B=1 -> 2 -> 1 in one
context and retain the same precompiled pipeline while rewriting exact shape
metadata.

Against the preceding same-harness observation, the VolvoxAI FP32 encoder fell
from 6533.6 ms to 601.8 ms, a 10.86x speedup; component latency fell from
6661.2 ms to 731.8 ms. The remaining FP32 component gap is 1.617x,
concentrated in the encoder: 2.780x ORT, while VolvoxAI's decoder total is
0.555x ORT. Its 601.8 ms encoder is close to the strict Vulkan result, 594.359
ms. Compilation is warmed and measured execution has no KV readback. Host-side
telemetry remains only 8.4 ms shape binding, 1.2 ms provider enqueue, and 10.1
ms through submission, so the remaining interval is queued GPU compute plus
the required small control-output readback. ORT retains mature graph fusion,
layout planning, and tactic selection; VolvoxAI still submits a fine-grained
model-neutral graph. Per-kernel GPU timestamps are required for an exact
residual breakdown.

ORT Web INT8 is 6.278x its FP32 component time. Its larger CPU/WebGPU partition
counts and unsupported-`QuantizeLinear` diagnostics are consistent with graph
fragmentation and boundary overhead, but the capability-probe events do not
identify measured internal-copy cost. VolvoxAI's packed-dot4 W8A8 route avoids
that partition pattern and records 0.293x the ORT Web INT8 component time. This
does not establish a general advantage over an optimized all-WebGPU ORT W8A8
implementation: it compares ORT's producer QDQ graph partitioned across WebGPU
and CPU with VolvoxAI's strict W8A8 graph.

Vulkan and OpenGL have no ONNX Runtime native peer in this harness, so these
strict VolvoxAI rows are observations rather than ORT ratios:

| Runtime route | Precision | Encoder | Prefill | Steady / token | Decoder total | Component | ORT native peer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| VolvoxAI strict Vulkan | FP32 | 594.359 | 16.221 | 16.451 | 65.598 | 660.074 | N/A |
| VolvoxAI strict OpenGL | FP32 | 3764.764 | 15.726 | 15.794 | 63.782 | 3828.865 | N/A |
| VolvoxAI strict Vulkan | W8A8 | 645.542 | 38.341 | 38.318 | 152.888 | 799.635 | N/A |
| VolvoxAI strict OpenGL | W8A8 | 647.283 | 42.104 | 42.754 | 170.266 | 820.772 | N/A |

Every native row requires its named backend and reports
`fallback=0;missing=0`. Vulkan reports `packedInt8Dot=false` and uses the tiled
or scalar W8A8 route; OpenGL has no packed-dot tactic counter. The slow OpenGL
FP32 encoder is not a dynamic-shape or fallback result, but this matrix does
not isolate the responsible kernel or driver cost.

ONNX Runtime also offers a
[native WebGPU plugin](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)
that can run through Dawn's Vulkan backend on Linux. That remains WebGPU over
Dawn rather than a direct Vulkan execution provider, just as ORT Web's WebGL
route is not a native OpenGL execution provider, so neither is used as a
same-backend denominator here.

### RTX 3090 CUDA observation

These values are retained from the existing RTX 3090 report and were not
remeasured after that report's final harness revision. They are not a fresh
qualification; the original settings and provenance remain attached to the
table.

| Runtime route | Precision | Encoder | Prefill | Steady / token | Decoder total | Component | x ORT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CUDA-first + CPU fallback | FP32 | 5.427 | 1.774 | 2.096 | 8.054 | 13.482 | 1.000 |
| VolvoxAI strict CUDA | FP32 | 36.261 | 4.593 | 4.436 | 17.952 | 54.213 | 4.021 |
| ONNX Runtime CUDA-first + CPU fallback | INT8 artifact | 7.425 | 2.482 | 2.801 | 10.894 | 18.252 | 1.000 |
| VolvoxAI strict CUDA | W8A8 | 31.952 | 5.318 | 5.219 | 20.993 | 52.953 | 2.901 |

Only VolvoxAI's rows are strict all-CUDA: every selected encoder/decoder node
reports `fallback=0;missing=0`. The ORT reference is CUDA-first with explicit
CPU fallback. Separate untimed profiling sessions recorded exact executed-node
placement: FP32 encoder CUDA 473 / CPU 48 and decoder CUDA 202 / CPU 0; INT8
encoder CUDA 943 / CPU 56 and decoder CUDA 530 / CPU 0. Separate strict probes
reject both encoders and accept both decoders. The measured sessions are not
profiled, so this attests the invariant configuration and canonical request,
not the exact execution instance.

ORT INT8 is the producer QDQ artifact under CUDA-first partitioning, not proof
of an all-CUDA tensor-core INT8 path.

Both processes selected visible CUDA ordinal 0, but this report does not expose
a comparable physical-device UUID across runtimes. Pairing is by ordinal and
reported RTX 3090 identity, not a cryptographic same-device attestation.

VolvoxAI CUDA INT8 improves encoder latency by 11.9% over FP32, but decoder
total is 16.9% slower; component totals improve by 2.3% (52.953 vs 54.213 ms).
Route counters prove DP4A execution, including 139,889,120 DP4A dot4 groups and
2,889,600 scalar-tail operations for the representative encoder. This does not
claim IMMA or tensor-core use. ORT remains faster because it brings mature CUDA
graph optimization, fusion, library kernels, packing, and tactic selection;
VolvoxAI executes a more granular per-node schedule with generic kernels.

A separate 2026-08-21 current-source, untimed B1 qualification on the same RTX
3090 passed FP32 and INT8 with one runtime and reused encoder/decoder contexts
across active `Q=2/M=212`, active `Q=8/M=218`, maximum-padded
`Q=192/M=402`, and active `Q=2/M=212` again. Both routes emitted
`[4,1038,5,6]`, preserved the cache, and reported strict CUDA with zero
fallback and zero missing nodes. This refreshes the current-source B1
correctness evidence only; it does not remeasure the historical table. The
fixed-B1 packages used by that qualification remain scheduler B1, so it also
makes no native scheduler-coalescing claim. Current native Vulkan/OpenGL/CUDA
coalescing requires an authored symbolic leading-B graph and the core proof.

### Host dynamic-shape qualification

The qualification record schema is
`volvoxai.tiny-receipt-dynamic-shape-qualification/v1`. Like the immutable
report filenames, that is an evidence schema rather than a supported runtime
or package ABI.

For the package's declared bounded domain, FP32 and INT8 pass the complete
dynamic-shape qualification on native CPU and WASM in the CPU reports and on
strict WebGPU, Vulkan, and OpenGL in the AMD GPU report. Each backend/precision
pair reuses one runtime plus one encoder and decoder context for this untimed
sequence:

```text
active Q=2/M=212
active Q=8/M=218
maximum-padded Q=192/M=402 (logical Q=8/M=218)
active Q=2/M=212
```

Every sequence records exact question tokens, grows decoder cache `P=1..4`,
returns exact logical output shapes, preserves cache prefixes, produces finite
appended rows, selects the same family, and emits the same tokens after
shrinking. Unsupported bounded domains fail rather than switching backend. The
retained CUDA report contains strict shape/cache/token evidence, but not a
complete fresh qualification.

### Host report evidence

- [One-core CPU/native/WASM report](../examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix.json),
  SHA-256 `06d09752777b421ca9383d2c73740bf4afbe274b741488a374ff861eedfb7d7b`
- [Six-core CPU/native/WASM report](../examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix_6c.json),
  SHA-256 `be3d2df5399abe965b3e7bf4d1acca6aa425e5ccf67aa4596a96bde95cc26588`
- [AMD WebGPU/Vulkan/OpenGL report](../examples/tiny_receipt_vqa/reports/explicit_kv_v1_gpu_matrix.json),
  SHA-256 `a6707b8f9556c15503fb59c54cb73f2c53e8563d7f0ce0bf8408efa9f575685f`
- [RTX 3090 CUDA report](../examples/tiny_receipt_vqa/reports/explicit_kv_v1_cuda_matrix.json),
  SHA-256 `23fb02af2f9543ccab192342b190f083e6d65da15d90fb6081ef1cd80d13133c`

These tracked paths are immutable publication evidence. Their v1 report names
remain unchanged; current runtime/package ABI support remains v2-only.

## Current Android results

Each Android sample uses a fresh process, runtime, and encoder/decoder context
or ORT session. One untimed request warms the same contexts immediately before
the measured request. Model loading, graph compilation/optimization, context
creation, image preprocessing, tokenization, process startup, application
validation, and host argmax are outside `Component` and included in
`Process wall`.

Deno WASM uses the runtime execution diagnostic boundary. Deno WebGPU uses the
required small-output readback fence so asynchronous GPU work is complete, but
closes the interval before application validation, argmax, and optional cache
qualification reads. Native routes request one host runtime thread. The Deno
WASM module has no shared-memory or pthread import, and every native or Deno
process is pinned to CPU 7 with `taskset 80`.

These results were measured on 2026-08-14 KST. All values are milliseconds.
`Encoder`, `Decoder x4`, and `Component` are the medians of three independently
measured execution samples. `Process wall` is
the median device-side elapsed time for the complete fresh process, including
model and context preparation, one untimed warmup request, the measured
request, and validation. Every VolvoxAI invocation is configured with
`--threads 1` where that native option exists; ONNX Runtime uses one intra-op
thread and one inter-op thread. Deno WASM executes on its single WASM thread;
Deno/V8 auxiliary threads inherit the same one-CPU process affinity.

| Runtime | Backend | Precision | Encoder | Decoder x4 | Component | Process wall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| VolvoxAI native | CPU | FP32 | 529.820 | 15.095 | 544.915 | 2,224.337 |
| VolvoxAI native | CPU | INT8 | 179.290 | 22.545 | 201.708 | 1,016.108 |
| ONNX Runtime 1.23.2 | CPUExecutionProvider | FP32 | 543.209 | 8.370 | 551.579 | 1,529.073 |
| ONNX Runtime 1.23.2 | CPUExecutionProvider | INT8 | 211.572 | 6.812 | 218.384 | 843.005 |
| VolvoxAI Deno 2.9.4 | WASM | FP32 | 1,538.088 | 84.592 | 1,624.159 | 4,729.608 |
| VolvoxAI Deno 2.9.4 | WASM | INT8 | 1,103.073 | 137.068 | 1,240.552 | 4,147.304 |
| VolvoxAI native | Vulkan | FP32 | 1,216.703 | 119.186 | 1,335.889 | 7,038.000 |
| VolvoxAI native | Vulkan | INT8 | 920.499 | 153.678 | 1,075.362 | 6,363.000 |
| VolvoxAI native | OpenGL ES | FP32 | 1,127.616 | 175.482 | 1,270.110 | 6,259.000 |
| VolvoxAI native | OpenGL ES | INT8 | 1,065.115 | 257.236 | 1,322.351 | 6,586.000 |
| VolvoxAI Deno 2.9.4 | WebGPU over Vulkan | FP32 | 1,327.143 | 399.952 | 1,731.623 | 6,211.788 |
| VolvoxAI Deno 2.9.4 | WebGPU over Vulkan | INT8 | 739.633 | 549.650 | 1,289.282 | 6,295.258 |

All 12 native GPU samples selected every graph node, reported zero fallback
and zero missing kernels, emitted `[4, 1038, 5, 6]`, and preserved cache/reset
parity. The measured Android native binary SHA-256 is
`64d392e7feacbe5f957c08b1aa95006be115ede70bb2c9c1d5a24374b272cac7`.

The CPU INT8 component is 7.6% shorter than ORT INT8, while CPU FP32 is 1.2%
shorter than ORT FP32. The native process wall remains longer because its graph
and context preparation is outside `Component`; the table keeps that cost
visible instead of mixing it into operator execution.

The one-thread node profile explains the remaining Arm CPU time. The FP32
encoder spends 373.590 ms of its 538.881 ms node sum in 13 Conv2D nodes. The
INT8 encoder has a 183.227 ms node sum led by QConv2D at 76.915 ms; GroupNorm
plus SiLU account for 35.921 ms, and QLinear for 16.984 ms. FP32 convolution is
therefore the principal CPU limit rather than a missing ISA dispatch. At
2.9 GHz, the Cortex-A720's two 128-bit FP FMLA instructions per cycle imply a
46.4 GFLOP/s one-core ceiling, and this model's 13 convolutions contain about
16.38 GFLOP. Their profiled 373.590 ms is about 43.9 GFLOP/s, roughly 95% of
that arithmetic ceiling. The phone's SVE vector length is also 128 bits, so
SVE2 cannot supply an AVX2/AVX-512-style width gain. The instruction throughput
figures are from the
[Cortex-A720 Software Optimization Guide](https://documentation-service.arm.com/static/65f1692987f147198672973b).

Vulkan INT8 has the shortest GPU component time at 1,075.362 ms, followed by
OpenGL ES FP32 at 1,270.110 ms. It is nevertheless 5.3x the native CPU INT8
component time, so the one-core CPU INT8 route remains the latency choice on
this target. The GPU drivers expose no packed INT8 dot capability, while these
graphs contain hundreds of small, dependency-ordered dispatches; launch,
binding, synchronization, and intermediate-memory traffic therefore remain
large relative to useful work.

The current native GPU implementation applies operator-specific paths rather
than a single Arm-wide shader strategy:

- Common QLinear and QConv2D kernels accumulate four adjacent output channels
  while sharing input/address calculations. Aligned QLinear inputs and weights
  use packed 32-bit loads, and tiled QConv2D reuses each unpacked activation in
  a four-lane accumulator. Odd reductions, asymmetric zero points, requantized
  output tails, and generic shape fallbacks are covered by native correctness
  tests.
- OpenGL ES selects the 16-output regular FP32 Conv2D shader for eligible
  convolutions, reuses one context-owned dispatch-parameter UBO whose storage
  is replaced for each upload, and omits redundant workgroup-memory clearing
  only from the audited inference shaders that initialize their shared tiles.
- Vulkan contexts share a device pipeline cache, including the prepared-model
  pipeline path. This follows Khronos's recommendation to supply a cache during
  compute-pipeline creation so reusable compilation work can be recovered
  ([Khronos Vulkan pipeline-cache sample](https://github.khronos.org/Vulkan-Site/samples/latest/samples/performance/pipeline_cache/README.html)).

The current OpenGL ES FP32 diagnostic run, with an intentional `glFinish`
after every dispatch, attributes 717.49 ms across the 13
`conv2DRegularOut16` encoder dispatches, 45.3% of its synchronized encoder
kernel time. GroupNorm is next at 235.01 ms. The forced synchronization
perturbs scheduling, so those values are used only to identify the current
bottleneck and are not table measurements.

### Deno WASM and WebGPU measurement

Android-native Deno 2.9.4 (`aarch64-linux-android`) runs directly from
`/data/local/tmp`. The current workload-minimal directory contains the Deno
binary plus `libandroid.so`, `libtermux-platform-ns.so`, `libsqlite3.so`, and
`libz.so.1` from the
[Termux Deno build](https://github.com/termux/termux-packages/blob/master/packages/deno/build.sh)
dependency packages, with the relocated library directory supplied through
`LD_LIBRARY_PATH`. The generic upstream Linux AArch64 binary is not used
because it targets glibc rather than Android's Bionic runtime.

The tested Deno binary SHA-256 is
`4ef01eb50ecce8cf8a86c7c2125abf1fdd6bcafd61c821312852c80f56edc97e`;
the tested `volvoxai.js` and `volvoxai.wasm` SHA-256 values are
`b98b59697541d6b54ec054dc51b36e5db88e66a6296298129697ea068279195d` and
`ffaa402d483fe37cc709322411dfb3381b57bd6345df06d538a693d65237922d`.
The measured Deno runner SHA-256 is
`6171df29cecced40b8ee497a1171d5c2cd576d60ca496da2b98203229407b9a1`.

The Deno runner generates the canonical raw U8 grayscale pixels, applies the
same `x/255*2-1` preprocessing, and fails unless the normalized F32 SHA-256 is
`7a6f7eb153434868b1685c4fd96fc63f1004cae356d15bd58404dccdc2c15963`.
Each of the 12 samples used a new Deno process with the persistent V8 code cache
disabled, preloaded two strict compiled models, performed one untimed request
on the same runtime/session/contexts, and timed the next request. Every warmup
and measured request selected family 0, emitted `[4, 1038, 5, 6]`, preserved
`P=1 -> R=2 -> 3 -> 4 -> 5`, and reported one encoder plus four decoder
executions with zero fallback. Every measured execution also hit its warmed
shape specialization.

WASM uses host-validated KV; its table interval is the runtime execution
diagnostic, while `required-output-readback` is retained as completion evidence.
WebGPU qualifies device cache values during warmup, then uses device-resident
KV with no cache readback in the measured request and the
`required-small-output-readback` component boundary. Its process used
`DENO_WEBGPU_BACKEND=vulkan` and Deno's
[`--unstable-webgpu`](https://docs.deno.com/runtime/reference/cli/unstable_flags/#--unstable-webgpu)
flag. `GPUAdapterInfo.description` reported `Samsung Xclipse 540`, and Android
driver logs recorded `Samsung::Vulkan::OpenDevice` with driver 24.0.560 in all
six WebGPU samples. This is a physical-device result, not SwiftShader or another
software adapter. Deno exposed the WGSL packed-dot language feature; that API
capability alone is not a claim that the driver selected a native packed-dot
instruction.

WebGPU reduces the FP32 and INT8 encoder medians by 13.7% and 32.9% relative to
Deno WASM, but its four decoder calls are 4.7x and 4.0x longer respectively.
Consequently the WebGPU component is 6.6% longer than WASM for FP32 and 3.9%
longer for INT8. The fine-grained one-token decoder remains dominated by WebGPU
dispatch, binding, synchronization, and shape-specialization writes on this
device.

### Android validation target

Device:

```text
Samsung SM-A566S / a56x
SoC: Samsung s5e8855
CPU: 1x Cortex-A720 2.9 GHz, 3x Cortex-A720 2.6 GHz, 4x Cortex-A520 1.95 GHz
ABI: arm64-v8a
Android: 16 / API 36
CPU features: asimd, asimddp, sve, sve2, svei8mm, i8mm, bf16
Linux SVE vector length: 16 bytes (128 bits)
measurement affinity: CPU 7 (`taskset 80`)
GPU: Samsung Xclipse 540
Vulkan: 1.3.279
Vulkan driver: Samsung 24.0.560
packed Vulkan INT8 dot: unavailable
OpenGL ES: 3.2, ANGLE over Vulkan
OpenGL ES driver: 24.1.305 (`e94bc7a33c77`)
```

The CPU topology and frequencies agree with
[Samsung's Exynos 1580 specification](https://semiconductor.samsung.com/kr/processor/mobile-processor/exynos-1580/).
Every one of the 12 current native GPU samples reported Android thermal status
0. Cached AP temperature across those samples ranged from 36.6 C through
41.7 C. All 12 Deno samples also reported status 0; none of these measurements
reported thermal throttling.

The comparison uses the official `onnxruntime-android` 1.23.2 ARM64 library,
`CPUExecutionProvider`, sequential execution, one intra-op thread, one inter-op
thread, and `ORT_ENABLE_ALL`. The standalone
[Android ORT runner](../examples/tiny_receipt_vqa/native/benchmark_ort_android.cpp)
requests all public outputs and validates finite logits/caches, cache prefixes,
mask growth, family selection, token parity, and every cache transition.

Current ARM CPU code is single-threaded for this benchmark and contains these
one-core reuse optimizations:

- The dispatcher probes HWCAP/HWCAP2 once and resolves the current Arm ladder:
  baseline, NEON, DOTPROD, I8MM, then SVE2. Optional DOTPROD and I8MM objects
  are compiled separately, so the baseline binary never executes unsupported
  instructions. This phone exposes SVE2 but only at a 128-bit vector length;
  it therefore provides no SIMD-width increase over NEON on this device, and
  the 128-bit I8MM kernels remain the selected integer matrix path. Arm defines
  SVE register width as implementation-selected from 128 through 2048 bits, so
  SVE/SVE2 availability alone does not imply an AVX2- or AVX-512-width vector
  unit ([Arm's NEON/SVE/SME comparison](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/matrix-matrix-multiplication-neon-sve-and-sme-compared)).
- FP32 BatchMatMul uses shape-specific AArch64 NEON tiles for both attention
  contractions. FP32 Conv2D uses its packed indirect-GEMM/FMLA route.
- QLinear/QGemm stores an immutable I8MM pair layout beside the canonical
  packed weights. A 6x16 SMMLA tile supports U8/I8 input, odd-K zero tails, exact
  asymmetric compensation, and vector ties-to-even requantization.
- Dense NHWC/OHWI QConv2D materializes only a bounded activation panel and
  consumes the same packed SMMLA weights. Unit-dilation rows are copied as
  contiguous NHWC spans, output coordinates advance without integer division,
  and the grayscale `K=9` stem has a byte-copy path. No full-image im2col buffer
  is allocated.
- QBatchMatMul packs dynamic right-hand data into at most 64 KiB of
  context-owned scratch and applies a 4x8 SMMLA tile. Ragged `K=218` attention
  projections are zero-padded inside the final K8 block; one-row decoder GEMV
  keeps the lower-overhead NEON route.
- QuantizeLinear/DequantizeLinear, GroupNorm, SiLU, GELU, Softmax, and common
  F32/U8 transpose permutations have AArch64 NEON paths. Floating reductions
  preserve their scalar order where the operator contract requires it.
- All CPU measurements request exactly one runtime thread; there is no worker
  pool contribution. Each process is pinned to CPU 7 to prevent migration.

This design follows the production techniques documented by
[XNNPACK's indirect convolution work](https://arxiv.org/abs/1907.02129),
[XNNPACK's ARM64 SDOT/I8MM and KleidiAI composition](https://github.com/google/XNNPACK/blob/master/CMakeLists.txt),
and [Arm Compute Library's GEMM, direct, Winograd, and indirect-GEMM routes](https://github.com/ARM-software/ComputeLibrary).
[Arm's current KleidiAI integration](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/arm-kleidiai-in-xnnpack)
likewise selects SDOT or I8MM kernels and packs operands for the selected
microkernel rather than traversing model-native weight layouts directly.

The Android tier-parity tests force baseline, NEON, DOTPROD, I8MM, and SVE2 in
fresh processes. QLinear, QConv2D, QBatchMatMul, quantization, normalization,
transpose, activation, and full quantized-runtime results match their canonical
references; the end-to-end token digest is `3f5608ae5cf7158e`.

## Reproduction

### Retired host remeasurement path

The retained host tables are immutable evidence bound to their recorded
manifest hashes. CPU first-execution matrices were produced by a removed
Python wrapper around a removed JS runtime harness. The archived reports remain
valid evidence, but the current checkout has no generated-proto TinyReceipt
host harness with which to remeasure them.

The AMD Vulkan/OpenGL/WebGPU same-context warmed matrix likewise depended on a
removed Python wrapper and removed browser/JS harnesses.

The RTX 3090 CUDA remeasurement command path also depended on the removed GPU
wrapper/harness stack and is retained only through the archived reports.

Remeasurement does not add a v1 runtime/package compatibility path.

### Retained Android native and Deno evidence

The VolvoxAI native rows were produced by the removed private-lifecycle split
driver. Their report identities remain above, but there is no corresponding
current-checkout command. The independent ORT Android runner and image helper
remain available. To rerun only that reference path:

```bash
: "${ANDROID_NDK:?set ANDROID_NDK to an installed Android NDK}"
: "${KV_MODEL_SOURCE:?set KV_MODEL_SOURCE to the producer v2 ONNX directory}"

ORT_ANDROID=build/onnxruntime-android-1.23.2
mkdir -p "$ORT_ANDROID/aar"
curl -fL -o "$ORT_ANDROID/onnxruntime-android-1.23.2.aar" \
  https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar
unzip -oq "$ORT_ANDROID/onnxruntime-android-1.23.2.aar" \
  -d "$ORT_ANDROID/aar"

CC="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang"
CXX="${CC}++"
"$CC" -O3 -DNDEBUG -Iexamples/tiny_receipt_vqa/native \
  -Inative/third_party -c examples/tiny_receipt_vqa/native/tiny_receipt_image.c \
  -o "$ORT_ANDROID/tiny_receipt_image.o"
"$CXX" -std=c++17 -O3 -DNDEBUG -I"$ORT_ANDROID/aar/headers" \
  examples/tiny_receipt_vqa/native/benchmark_ort_android.cpp \
  "$ORT_ANDROID/tiny_receipt_image.o" \
  -L"$ORT_ANDROID/aar/jni/arm64-v8a" -lonnxruntime \
  -Wl,-rpath,'$ORIGIN' -o "$ORT_ANDROID/benchmark_ort_android"
```

Push the exact ONNX models and image. The canonical image is an external
benchmark input and is not tracked in this repository; set
`CANONICAL_RECEIPT` to the restored, non-sensitive fixture and record its hash
with the new report before running this block:

```bash
: "${CANONICAL_RECEIPT:?set CANONICAL_RECEIPT to the restored benchmark image}"
DEVICE_DIR=/data/local/tmp/volvoxai-tinyreceipt
adb shell "mkdir -p $DEVICE_DIR/onnxruntime-1.23.2"
adb push "$CANONICAL_RECEIPT" "$DEVICE_DIR/receipt.png"
adb push "$ORT_ANDROID/benchmark_ort_android" \
  "$ORT_ANDROID/aar/jni/arm64-v8a/libonnxruntime.so" \
  "$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
  "$DEVICE_DIR/onnxruntime-1.23.2/"
adb push "$KV_MODEL_SOURCE/encoder_model.onnx" \
  "$KV_MODEL_SOURCE/decoder_model.onnx" \
  "$KV_MODEL_SOURCE/encoder_model_int8.onnx" \
  "$KV_MODEL_SOURCE/decoder_model_int8.onnx" \
  "$DEVICE_DIR/onnxruntime-1.23.2/"
adb shell "chmod 755 $DEVICE_DIR/onnxruntime-1.23.2/benchmark_ort_android"
```

Run each reference command in a fresh process three times. `toybox time -p` reports
`real` in seconds; multiply it by 1,000 for the table's `Process wall`
milliseconds:

```bash
adb shell 'cd /data/local/tmp/volvoxai-tinyreceipt/onnxruntime-1.23.2; \
  toybox time -p taskset 80 ./benchmark_ort_android encoder_model.onnx decoder_model.onnx ../receipt.png fp32 1 1; \
  toybox time -p taskset 80 ./benchmark_ort_android encoder_model_int8.onnx decoder_model_int8.onnx ../receipt.png int8 1 1'
```

The retained Deno measurement extracted an Android-native workload-minimal set
into `VX_DENO_ROOT` and pushed `package.json`, `dist/0.4.0`,
`examples/tiny_receipt_vqa`, and both `build/tiny-receipt-kv-{f32,int8}`
packages into `VX_DENO_PROJECT`. It used a new `adb shell` process for every
sample. The measured 12-run order was
`WASM/FP32, WASM/INT8, WebGPU/FP32, WebGPU/INT8, WebGPU/FP32, WASM/INT8,
WASM/FP32, WebGPU/INT8, WASM/INT8, WebGPU/FP32, WebGPU/INT8, WASM/FP32`.

The August 14, 2026 Deno measurement command templates used the removed JS
runtime harnesses and are intentionally not restated as runnable commands in
the current checkout.
