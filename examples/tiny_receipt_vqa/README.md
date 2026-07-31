# TinyReceiptVQA examples

TinyReceiptVQA is application code built on VolvoxAI's model-neutral
lifecycle. It owns receipt image preprocessing, the packaged tokenizer,
question formatting, router selection, autoregressive generation, and answer
postprocessing.

The example is not exported from the VolvoxAI packages.

`tools/exporter/` contains only general-purpose, model-neutral import, IR,
optimization, calibration/PTQ, verification, and publication primitives. This
example owns every TinyReceipt-specific concern: its encoder/decoder package
manifest, family vocabulary and routing, specialization rules, calibration
orchestration, pass composition, task qualification, and release commands.

## Split-ONNX source import

The supported path starts from the producer's local, self-contained
`tiny_receipt_vqa_split_onnx_v1` directory. It does not accept a PyTorch
checkpoint. FP32 is the default variant:

~~~bash
python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source artifacts/tiny_receipt_vqa_split_onnx \
  --out-dir build/tiny-receipt-f32-imported \
  --target portable
~~~

Select the producer's static U8S8 W8A8 QDQ encoder and decoder explicitly:

~~~bash
python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source artifacts/tiny_receipt_vqa_split_onnx \
  --out-dir build/tiny-receipt-int8-imported \
  --variant int8-w8a8 \
  --target portable
~~~

The split importer accepts both the legacy 760-entry `CharVocab` and the
release `byte_fallback_bpe` v1 tokenizer. The latter is the NFC-normalized,
1,536-entry contract recorded by `config.vocab_size`, the source manifest, and
`vocab.json`; its ordered `itos`, ranked `merges`, atomic tokens, byte-fallback
tokens, and tokenizer hash are validated together. Graph signatures and
decoder output shapes are checked against that manifest-derived vocabulary
size rather than a hardcoded character-vocabulary width.

### Static INT8 handoff and portable optimization

Keep the producer directory outside the repository and pass it explicitly. A
portable, source-preserving import followed by the exact graph cleanup is:

~~~bash
MODEL_SOURCE=/path/to/tiny-receipt-split-onnx
INT8_IMPORTED_PACKAGE=build/tiny-receipt-int8-imported
DIRECT_EXACT_PACKAGE=build/tiny-receipt-int8-from-int8-exact

python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source "$MODEL_SOURCE" \
  --out-dir "$INT8_IMPORTED_PACKAGE" \
  --variant int8-w8a8 \
  --target portable

python3 -m examples.tiny_receipt_vqa.tools.optimize_direct_int8 \
  --source "$INT8_IMPORTED_PACKAGE" \
  --out "$DIRECT_EXACT_PACKAGE"
~~~

The first command validates hashes, opset 18, graph signatures, the static
U8S8 QDQ contract, and fixed-padding ONNX Runtime parity before exporting two
`volvox-graph/v1` graphs and safetensors stores. It specializes execution to
batch 1, question/decoder length 192, and memory length 402, and converts the
I64/BOOL runtime boundaries to I32. The producer INT8 pair remains a truthful
hybrid package: quantized Conv/MatMul/Gemm regions become `QConv2D`,
`QBatchMatMul`, `QGemm`, and `QLinear`, while normalization, softmax, and shape
work remains F32.

The second command runs the typed, semantics-preserving runtime optimizer and
then refreshes graph, tensor, and manifest hashes. Its default
`RuntimePackedQLinearSplitPass` recognizes
`QLinear -> DequantizeLinear -> optional F32 Add -> movement -> Slice`. It
slices the packed OUT_IN byte weight, I32 bias, and per-axis weight scale and
zero point; reuses the packed output affine; preserves the F32 bias and later
Mul/QuantizeLinear boundaries; and removes the now-unreferenced packed
tensors.

The default also runs exact static-QDQ layout cleanup. It interns only
byte-identical scalar affine references, converts a Transpose to a storage-only
Reshape only when it moves singleton axes without reordering non-singleton
axes, cancels proven inverse Transposes around scalar-affine DQ, hoists the
canonical pointwise Sigmoid/Mul/Q island across inverse Transposes, and moves a
closed `DQ -> layout -> Q` chain into the byte domain only when both ends use
the same producer affine. These passes are fail-closed. They do not change a
numeric scale or zero point, decode or requantize a weight, or alter serialized
weight payload bytes.

Without `--fuse-attention` or `--fuse-static-qdq-compute`, the exact typed
route deliberately retains the producer's `QBatchMatMul`/Softmax attention and
intermediate Q/DQ/Add/Mul/QuantizeLinear boundaries. It removes the packed full-width
movement without claiming that deleting or fusing those numerically visible
boundaries would be equivalent. The input package's serialized byte weights and
current affines are the source of truth: no pass reconstructs FP32 weights or
quantizes them a second time.

The comparable Direct INT8 deployment candidate opts into both available
numerical migrations—attention lifting and closed static-QDQ compute fusion—and
then applies the TinyReceipt-owned canonical deployment ABI. QSDPA versus
unfused `QBatchMatMul`, and fused Q* compute versus separate DQ/F32/Q
evaluation, can have different numeric and performance results on different
backends:

~~~bash
DIRECT_MIGRATED_PACKAGE=build/tiny-receipt-int8-from-int8

python3 -m examples.tiny_receipt_vqa.tools.optimize_direct_int8 \
  --source "$INT8_IMPORTED_PACKAGE" \
  --out "$DIRECT_MIGRATED_PACKAGE" \
  --fuse-attention \
  --fuse-static-qdq-compute \
  --canonical-deployment
~~~

This command selects the general registry's exact packed/layout groups plus its
quantized-attention and static-compute migration groups in one verified
transaction. The attention pass recognizes the static-QDQ chain, proves the
head layout and mask conversion, and emits `QSDPA`. The lift reuses the original
byte operands and affines and never
decodes or requantizes weights, but it removes the source score and probability
requantization boundaries. The compute flag replaces only closed canonical
`DQ -> Add/LayerNorm/GELU/SiLU/GroupNorm -> Q` islands with their Q* operators,
again using the input and output affines already present in the producer graph.
It performs no calibration. Both transforms are opt-in **numerical
migrations**, not bit-exact rewrites, because fused kernels may use a different
F32 evaluation or reduction order. `--canonical-deployment` then hoists the
decoder's canonical I32 `v4_keep` input and specializes F32 `logits` to
byte-domain first-index I32 `token_ids`. It changes the application ABI, not the
producer's affine values or initializer payloads, and is accepted only when
both numerical-migration flags are present. If a recognized attention block
cannot be fully legalized, the conversion transaction fails closed; successful
legalization is still only a candidate.
Run independent differential execution, task accuracy, family-routing,
strict-backend kernel, and paired latency qualification on this exact package
identity. It cannot inherit evidence from the default exact package.

The no-flag command and `DIRECT_EXACT_PACKAGE` retain the producer-facing ABI:
the decoder consumes no hoisted `v4_keep` input and returns F32 `logits` for a
host first-index argmax. Supplying the two fusion flags without
`--canonical-deployment` performs the same numerical migrations but still
retains that source ABI. Only the three-flag command above creates the
session-shaped Direct INT8 candidate used for a like-for-like comparison with
FP32-to-PTQ.

The unfused native `QBatchMatMul` does have an incremental one-row kernel, but
only for its strict batch-1 sequence layout when the complete right operand is
non-overlapping and clean in the current dependency closure. Other broadcast,
aliasing, or dirty-right-operand forms remain whole-tensor operations. That is
an operator capability, not proof that the entire direct-import graph satisfies
`--require-row`; the exact Direct INT8 logits ABI still uses the explicit host
loop below and needs its own target qualification.

Direct static INT8 import has one mandatory affine policy: the importer always
preserves the producer's activation scales, zero points, weight scales, and
serialized INT8/U8 weight bytes. It never collects another activation profile,
recalibrates the already-quantized model, or rewrites those numeric values. The
importer exposes no direct-INT8 recalibration mode or profile option. The
direct optimizer can remove or move compatible Q/DQ boundaries and can fuse
operators only by reusing the existing producer affines; it does not reconstruct
or requantize weights.

Do not carry node counts or one-off timings forward from an older package
revision. The optimizer output and each packaged `export_report.json` describe
the graph revision that was actually produced; the content-addressed native
heldout report described below is the source of truth for performance,
accuracy, routing, and cross-artifact agreement. Until that report is generated
for the four package identities in the profiling command, this document makes
no current numeric performance claim.

Build the inference-profile WASM binary and JavaScript bundle before executing
either graph:

~~~bash
make build_wasm
npm run build:all
~~~

At this boundary the encoder and decoder are independently runnable with a
strict WASM backend and forbidden operator fallback. Application inference must
then perform these steps:

1. Load each graph and its safetensors with `GraphLoader`, create a runtime with
   `backends: ['wasm']`, and compile with required WASM plus
   `operatorFallback: 'forbid'`.
2. Convert a receipt to grayscale F32 NCHW `[1,1,320,672]`, normalize it to
   `[-1,1]`, encode the question with the tokenizer declared by the package to
   padded I32 `[1,192]`, and pass I32 `family_ids=[-1]` for AUTO routing.
3. Execute the encoder once and retain `memory`, `memory_padding_mask`, and
   `selected_family_ids`.
4. Repeatedly execute the decoder with `[BOS, generated prefix, PAD...]`, the
   retained memory/mask, and the selected family. Read row
   `prefix_length_minus_one` from F32 `[1,192,V]` logits, where `V` is the
   package's `config.vocab_size` (1,536 for the release BPE model), use
   first-index argmax, and stop at EOS or 191 new tokens.
5. Compare router logits, selected family, token IDs, and decoded text against
   the ONNX Runtime oracle, then measure latency on the shipping WASM target.

The exact Direct INT8 package deliberately differs from the session ABI; the
canonical migration makes that application boundary explicit:

| Contract | Direct INT8 exact | Direct INT8 canonical migration | FP32-to-PTQ session package |
| --- | --- | --- | --- |
| Encoder inputs | image, question IDs, family IDs | image, question IDs, family IDs | image, question IDs, family IDs |
| Decoder inputs | prefix, memory, mask, family IDs | prefix, memory, mask, family IDs, `v4_keep` | prefix, memory, mask, family IDs, `v4_keep` |
| Decoder output | F32 `logits`; host first-index argmax | I32 `token_ids` from in-graph `QArgMax` | I32 `token_ids` from in-graph `QArgMax` |
| Numeric contract | Source-preserving exact rewrites only | Producer-affine numerical migration; separately qualified | Calibrated PTQ; separately qualified |

Passing `DIRECT_EXACT_PACKAGE` to `TinyReceiptSplitSession` or
`run_split_e2e.mjs` therefore fails manifest validation; use its explicit host
loop instead. `DIRECT_MIGRATED_PACKAGE` has the session-shaped ABI, but it is
deployable only after its own differential, task, routing, strict-backend, and
latency qualification. An explicitly route-specialized session package is also
supported; it removes `family_ids` from both graphs and records the one bound
family in manifest routing.

The importer validates the source manifest, hashes, ONNX signatures, opset,
absence of external tensor data, tokenizer and vocabulary contract, family
ordering, and fixed-padding parity with ONNX Runtime. It then calls the generic
`tools/export_safetensors.py` frontend for the encoder and decoder separately.
For `int8-w8a8`, it additionally verifies static per-tensor U8 activation
quantization, per-output-channel S8 weights, I32 accumulation metadata, and
the QDQ inputs of Conv, MatMul, and Gemm. Unused producer domain imports are
allowed, but a custom-domain node in a graph, subgraph, or local function is
rejected.

The generic exporter runs in source-preserving quantization mode. The package
manifest records the selected producer variant and the generic graph
classification for each graph. `complete_w8a8_fusion` is true only when both
compiled graphs are classified as the canonical `w8a8-v1` graph contract;
otherwise the package is honestly recorded as hybrid.

Import is a lossless/source-parity boundary, not the final application ABI.
Both imported variants expose the producer's decoder `logits`. The sole current
`TinyReceiptSplitSession` ABI instead exposes only I32 `token_ids` from an
in-graph first-index `QArgMax`; it uses the exact optimized graph input set
recorded by the package manifest. Runtime-routed packages retain `family_ids`
on both graphs, while explicitly specialized packages remove it from both.
Therefore, do not load either raw imported directory directly into
`TinyReceiptSplitSession`; use a separately authored canonical deployment
package.

The deployment path is:

~~~text
source-preserving split import (decoder logits)
  → fuse and optimize FP32 semantics
  → retain runtime family routing by default, or explicitly specialize an input
  → calibrate that exact graph from immutable representative records
  → optionally sweep all public routes and retain the coverage as qualification evidence
  → calibrated PTQ of canonical runtime operators without float-op policy exclusions
  → post-PTQ typed cleanup + decoder logits→token_ids specialization
  → differential/task/latency qualification
  → publish the refreshed manifest and asset hashes
~~~

A concrete runtime-routed, session-ready template starts from the producer FP32
graphs, not from its already-quantized QDQ graphs. Runtime routing is the
default; the command does not silently select `phone` or any other family. Only
historical family 0 (`phone`) and family 1 (`address`) specialized packages
currently have the documented qualification evidence. That evidence does not
qualify the runtime-routed artifact or the other six families. Calibration
records must be representative and disjoint from heldout evaluation. They do
not need to invent or relabel records for every manifest-public family: a
model-owned route sweep may execute the same immutable records through each
public route while retaining their original semantic family labels.

~~~bash
: "${MODEL_SOURCE:?set MODEL_SOURCE to the split ONNX producer directory}"
: "${PRODUCER_ROOT:?set PRODUCER_ROOT to the training producer source root}"

FP32_IMPORTED_PACKAGE=build/tiny-receipt-f32-imported
FP32_PACKAGE=build/tiny-receipt-f32-from-f32
CALIBRATION_RECORDS=build/tiny-receipt-calibration-records.json
CALIBRATION=build/tiny-receipt-runtime-calibration.json
INT8_FROM_FP32_PACKAGE=build/tiny-receipt-int8-from-fp32

python3 -m examples.tiny_receipt_vqa.tools.import_hf_split_onnx \
  --source "$MODEL_SOURCE" \
  --out-dir "$FP32_IMPORTED_PACKAGE" \
  --variant fp32 \
  --target portable

python3 -m examples.tiny_receipt_vqa.tools.optimize_split_fp32 \
  --source "$FP32_IMPORTED_PACKAGE" \
  --out "$FP32_PACKAGE" \
  --hoist-input v4_keep:int32

python3 -m examples.tiny_receipt_vqa.tools.make_calibration_records \
  --producer-root "$PRODUCER_ROOT" \
  --synth-root "$PRODUCER_ROOT/synth/data" \
  --receipt-limit 1024 \
  --samples 256 \
  --tasks-per-receipt 6 \
  --seed 71 \
  --out "$CALIBRATION_RECORDS"

VOLVOX_CALIBRATION_BACKEND=cpu \
node --import tsx \
  examples/tiny_receipt_vqa/tools/calibrate_split_activations.mjs \
  --package "$FP32_PACKAGE" \
  --calibration-data "$CALIBRATION_RECORDS" \
  --samples 256 \
  --prefixes-per-record 3 \
  --route-sweep balanced-public \
  --out "$CALIBRATION"

python3 -m examples.tiny_receipt_vqa.tools.quantize_split_package \
  --source "$FP32_PACKAGE" \
  --calibration "$CALIBRATION" \
  --out "$INT8_FROM_FP32_PACKAGE" \
  --activation-dtype int8 \
  --activation-scheme symmetric
~~~

PTQ defaults to `--activation-dtype int8 --activation-scheme symmetric`, which
uses signed bytes with zero point 0. Use `--activation-scheme asymmetric` with
`--activation-dtype int8` to derive a non-zero signed zero point from each
observed activation range. `--activation-dtype uint8` is also supported and,
when the scheme is omitted, retains its conventional asymmetric default; an
explicit `--activation-scheme symmetric` selects U8 with zero point 128. The
resolved dtype and scheme are recorded under
`variant.precision_policy` in the published package manifest.

The manifest command lazy-loads the producer's
`tiny_receipt_vqa/train.py` only in this offline example tool. With the release
settings above it deterministically reproduces 256 synthetic training records:
`phone/address/store/item_row=43` each and `item_math/item_lookup=42` each.
It content-hashes the loaded generator, eligible annotation inventory, and
every selected annotation/image. It never reads the heldout tree. There are no
producer records labelled `math` or `other`, so the manifest does not fabricate
or relabel them.

`--route-sweep balanced-public` is deliberately a separate execution
dimension. It keeps all 256 immutable records and assigns them deterministically
round-robin across the eight canonical public route IDs. The common model path
therefore sees all 256 records once, while every route-specific path sees 32.
Decoder calibration observes each execution at three teacher-forced prefixes.
Provenance retains `selected_records` and `family_counts.record` for the six
semantic labels, while `route_executions.family_counts` records the eight
numeric execution routes. No record is relabelled and the package keeps the
original eight-family `family_ids` ABI throughout. `all-public` remains the
explicit Cartesian alternative (2,048 encoder executions, 256 per route) when
the extra calibration cost is justified.

`optimize_split_fp32.py` runs the proven grouped-projection split by default,
before calibration and PTQ. It splits packed self- and cross-attention Q/K/V
projections while their FP32 semantics are still explicit; no
separate rank-changing compatibility pass is used. The optimizer emits
runnable `CrossSDPA`, and the later quantizer can produce the compact
incremental decoder ABI. With no `--freeze-input`,
`family_ids` remains a public input of both graphs and the refreshed manifest
records runtime routing. `--freeze-input family_ids=N` is an explicit optional
specialization. The optimizer binds the same non-negative ID in both graphs,
removes that input from the public ABI, and requires calibration bound to that
resulting specialized graph revision. The flag is not required for PTQ and has
no implicit phone default.

For a runtime-routed package, tensor calibration and route qualification are
separate contracts. The generic PTQ planner consumes ranges bound to the exact
graph fingerprint and preserves `family_ids` exactly as an opaque I32 public
input. TinyReceipt may additionally execute every representative record through
all public route IDs so route-specific numeric paths contribute observations.
The publisher records selected-record identity separately from actual route
execution count and publishes a zero-filled per-family coverage report. Missing
or zero-count families remain visible but do not reject quantization or silently
specialize the package. Coverage alone is not task-accuracy or latency
qualification for any family.

The optimizer publishes runnable `CrossSDPA` before calibration. The quantizer
then imports that exact package through typed RuntimeIR and uses the
model-neutral PTQ planner as its only quantization source of truth. That planner
forms maximal proven byte islands around canonical operators, inserts only the
required Q/DQ boundaries, and refuses incomplete calibration or unrepresentable
layouts. The publisher then adds decoder
`QArgMax` and refreshes the package ABI and hashes. The result is only
publishable after differential, task-accuracy, strict-backend, and latency
qualification.

The command above intentionally has no float-op exclusions: native
incremental execution needs the decoder's changing closure to remain entirely
in the row-capable quantized operator set. `--require-row` selects and validates
the incremental row scheduler; it does not turn arbitrary full-tensor
operators into row kernels. A mixed-precision browser experiment may
add exclusions such as `--float-ops Embedding` or
`--decoder-float-ops Add,LayerNorm`, but it must use a different output
directory and must not be passed to native `--require-row`.

No output, node-count, or latency result is embedded here for the current
package revisions. Generate native CPU accuracy and timing evidence from the
four content-identified artifacts with `benchmark_native_heldout.py` below.
WASM needs a separate target-specific report; a native row-kernel result must
not be reused as a WASM performance claim.

`examples/tiny_receipt_vqa/tools/make_calibration_records.py`,
`examples/tiny_receipt_vqa/tools/optimize_direct_int8.py`,
`examples/tiny_receipt_vqa/tools/optimize_split_fp32.py`,
`examples/tiny_receipt_vqa/tools/calibrate_split_activations.mjs`,
`examples/tiny_receipt_vqa/tools/quantize_split_package.py`, and
`examples/tiny_receipt_vqa/tools/package_manifest.py` are application-owned
TinyReceipt orchestration for these stages. They consume model-neutral
`tools/exporter` primitives; their split graph coupling, family rules, and
package policy must not move into the generic exporter. `--calibration-data`
takes an explicit
`volvox-calibration-records/v1` JSON file, not a directory. Its calibration
report stores no CLI-resolved local paths: the record manifest, calibrator,
source graphs and weights, and any fixture files are bound by fixed logical
relative names, byte counts, and SHA-256 identities. The quantizer is the stage
that refreshes the split manifest to the `token_ids` ABI. General typed
optimizer and PTQ contracts are documented in
[graph optimizer design](../../docs/graph-optimizer-design.md) and
[typed PTQ](../../docs/typed-ptq.md).

Publication is atomic and has this layout:

~~~text
package_manifest.json
config.json
vocab.json
encoder/
├── graph.json
├── model.safetensors
└── export_report.json
decoder/
├── graph.json
├── model.safetensors
└── export_report.json
~~~

`TinyReceiptSplitSession.js` consumes only the optimized, PTQ-materialized
manifest contract:

~~~javascript
import { VolvoxAI } from '../../ts/index.ts';
import { TinyReceiptSplitSession } from './TinyReceiptSplitSession.js';

const runtime = await VolvoxAI.createRuntime({
  backends: ['webgpu', 'wasm', 'cpu'],
});
const session = await TinyReceiptSplitSession.load({
  runtime,
  packageUrl: 'models/tiny-receipt-runtime/package_manifest.json',
  graphLoader: ({ graphUrl, weightsUrl, fetch, safetensorsCache }) =>
    loadApplicationGraph({ graphUrl, weightsUrl, fetch, safetensorsCache }),
});
const answer = await session.generate({
  image,
  prompt: 'phone number?',
  family: 'auto',
});
await session.close();
await runtime.close();
~~~

The split session selects its tokenizer from the package manifest. Legacy
`char-vocab` packages keep their character lookup behavior. A
`byte_fallback_bpe` v1 package verifies the tokenizer hash, normalizes JavaScript
input to NFC, preserves structural tags and digits as atomic tokens, applies
the published merge ranks, and falls back to the packaged UTF-8 byte tokens.

The encoder runs once. With the default `decodePolicy: 'auto'`, a WASM decoder
executes one full fixed-length seed and then updates only row
`prefixLength - 1`, retaining its dependency/KV state. Other backends use
ordinary fixed-length forwards by default. Set `decodePolicy` to `'required'`
to require the retained-row path on the selected backend, or to `'ordinary'`
to disable it; required mode fails instead of silently falling back.

The decoder input remains `[1,192]` `[BOS, prefix, PAD...]` and includes the
hoisted I32 `v4_keep` input when declared by the manifest. A source-preserving
package returns F32 `[1,192,V]` logits and applies first-index argmax on the
host; a canonical deployment returns I32 `[1,192]` `token_ids` from in-graph
`QArgMax`. Both read row `prefixLength - 1` and permit at most 191 new tokens.
The encoder's `memory_padding_mask` is passed through unchanged: nonzero means
blocked.

## Deterministic split end-to-end qualification

All runtime surfaces use the same versioned
`references/split_int8_e2e_ort_cpu.json` oracle and the
`synthetic-exact-f32-v1` workload. The image is generated directly as exact
F32 values, the prompt is `phone number last one`, family selection is
AUTO/-1, and generation performs four ordinary fixed-length decoder forwards.
The expected token IDs and family are exact. Each of the eight router logits
uses the reference-owned acceptance rule
`abs(actual - expected) <= atol + rtol * abs(expected)`, currently with
`atol=1e-4` and `rtol=1e-5`. Aggregate router summaries use bounds derived
from those eight per-value tolerances.

The JavaScript session and runner accept both manifest-declared decoder ABIs:
source-preserving F32 logits with host first-index argmax and canonical I32
token IDs. References remain content-bound to one exact package and tokenizer.
Generate a fresh ONNX Runtime oracle for a new imported package and pass it with
`--reference`; the checked-in oracle only applies when those identities match.
`--no-reference` deliberately skips the oracle comparison and is useful for a
strict-backend runnability smoke test, but is not qualification evidence.

Generate the package-bound reference from the validated producer INT8 ONNX
directory before running a newly imported package:

~~~bash
mkdir -p build/tiny-receipt-e2e
python3 -m examples.tiny_receipt_vqa.tools.generate_split_onnx_e2e_reference \
  --source /path/to/tiny-receipt-split-onnx-release \
  --package build/tiny-receipt-int8-from-int8 \
  --out build/tiny-receipt-e2e/ort-reference.json
~~~

Intended JavaScript CPU or WASM lifecycle:

~~~bash
mkdir -p build/tiny-receipt-e2e
node examples/tiny_receipt_vqa/tools/run_split_e2e.mjs \
  --backend=cpu \
  --package=build/tiny-receipt-int8-from-int8 \
  --reference=build/tiny-receipt-e2e/ort-reference.json \
  --out=build/tiny-receipt-e2e/cpu.json

node --experimental-wasm-relaxed-simd \
  examples/tiny_receipt_vqa/tools/run_split_e2e.mjs \
  --backend=wasm \
  --package=build/tiny-receipt-int8-from-int8 \
  --reference=build/tiny-receipt-e2e/ort-reference.json \
  --out=build/tiny-receipt-e2e/wasm.json
~~~

For a runnability-only smoke test, replace `--reference=...` with
`--no-reference`. Supplying both flags is rejected.

Emit the shared raw fixture without executing a backend:

~~~bash
node examples/tiny_receipt_vqa/tools/run_split_e2e.mjs \
  --package=build/tiny-receipt-int8-from-int8 \
  --fixtures-dir=build/tiny-receipt-e2e/fixtures \
  --fixtures-only
~~~

Run a physical WebGPU adapter through Deno. `VK_ICD_FILENAMES` must already
identify the host's NVIDIA Vulkan ICD rather than a software ICD:

~~~bash
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
DENO_WEBGPU_BACKEND=vulkan \
VK_ICD_FILENAMES="${VK_ICD_FILENAMES:?set the NVIDIA Vulkan ICD JSON}" \
DENO_DIR="$scratch/deno-cache" \
XDG_CACHE_HOME="$scratch/cache" \
deno run --allow-read --allow-write --unstable-webgpu \
  examples/tiny_receipt_vqa/tools/run_split_e2e_deno.js \
  --package=build/tiny-receipt-int8-from-int8 \
  --reference=build/tiny-receipt-e2e/ort-reference.json \
  --adapter=high-performance \
  --require-adapter='NVIDIA GeForce RTX 3090' \
  --out=build/tiny-receipt-e2e/webgpu.json
~~~

The Python evidence orchestrator `run_native_split_e2e.py` also accepts both
decoder ABIs. It uses repeated public `native/volvoxai run` commands, reads the
manifest's semantic-to-runtime tensor names, runs the encoder once, feeds its
ordinary raw outputs into four autoregressive decoder forwards, and validates
every CLI-produced `volvoxai.runtime-evidence` document:

~~~bash
mkdir -p build/tiny-receipt-e2e
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
python3 -m examples.tiny_receipt_vqa.tools.run_native_split_e2e \
  --package build/tiny-receipt-int8-from-int8 \
  --binary native/volvoxai \
  --cpu \
  --reference build/tiny-receipt-e2e/ort-reference.json \
  --artifact-dir "$scratch" \
  --json-out build/tiny-receipt-e2e/native-cpu.json
~~~

Use `--cuda --require-cuda-device` instead of `--cpu` with a CUDA-enabled
already-built binary. The additional flag requires one or more fully parsed
CUDA device banners in every CLI process; repeated banners and all five
invocations must identify the same physical name and compute capability. A
backend flag means strict provider requirement and forbidden operator
fallback; the orchestrator rejects missing, contradictory, fallback, or
unstable-result evidence. Saved reports contain only relative labels, public
device identity, content hashes, tensor summaries, and results. They never
contain resolved package, binary, artifact, or reference paths.

`tools/assemble_onnx_suite.py` remains a historical nine-package experiment. It is
not the split-ONNX release path.

## Paired heldout qualification

Run the two currently qualified specialized families against the same first
100 sorted heldout cases and the same local legacy oracle:

~~~bash
: "${RECEIPT_VQA_DATA_ROOT:?set RECEIPT_VQA_DATA_ROOT to the dataset root}"
node --experimental-wasm-relaxed-simd --import tsx \
  examples/tiny_receipt_vqa/tools/benchmark_heldout.mjs \
  --eval "$RECEIPT_VQA_DATA_ROOT/eval/heldout" \
  --count 100 \
  --split build/fam \
  --legacy build/legacy-w8a8-current \
  --families phone,address \
  --report build/tiny-receipt-heldout-phone-address.json
~~~

Accuracy mode generates the full 191-token autoregressive budget by default. A
short `--tokens` run is a latency probe only and is rejected unless it also
passes `--latency-only`:

~~~bash
node --experimental-wasm-relaxed-simd --import tsx \
  examples/tiny_receipt_vqa/tools/benchmark_heldout.mjs \
  --eval "$RECEIPT_VQA_DATA_ROOT/eval/heldout" \
  --count 10 \
  --split build/fam \
  --legacy build/legacy-w8a8-current \
  --families phone,address \
  --tokens 64 \
  --latency-only
~~~

`--families phone,address` is part of the qualification scope, not a display
filter. Families `store`, `item_row`, `item_math`, `item_lookup`, `math`, and
`other` remain unqualified until each has disjoint training-only calibration
and the same task, routing, and paired-latency gates. Heldout records are
evaluation inputs only and must never be supplied to the calibration command.

## Whole-model JavaScript W8A8 session

TinyReceiptW8A8Session.js loads a materialized W8A8 package through an
application-supplied Runtime:

~~~javascript
import { VolvoxAI } from '../../ts/index.ts';
import { TinyReceiptW8A8Session } from './TinyReceiptW8A8Session.js';

const runtime = await VolvoxAI.createRuntime({
  backends: ['webgpu', 'wasm', 'cpu'],
});

const session = await TinyReceiptW8A8Session.load({
  runtime,
  packageUrl: 'models/tiny-receipt/package_manifest.json',
});

const answer = await session.generate({
  image,
  prompt: 'phone number?',
  incremental: true,
});

await session.close();
await runtime.close();
~~~

The wrapper validates its manifest, creates a Model for the router and selected
family Graph, compiles each Model with an explicit provider policy, and gives
each compilation its own ExecutionContext.

Incremental generation uses context.decode.reset(), seed(), and step().
Every call returns a stable ExecutionResult; the wrapper reads its declared
token_ids output and closes the result. No output is a borrowed view of a
mutable device or context buffer.

Preload selected families without compiling over shared mutable backend state:

~~~javascript
await session.preload({ families: ['phone'] });
// or:
await session.preload({ families: 'all' });
~~~

The session caches immutable safetensors bytes while retaining independent
Model, CompiledModel, and ExecutionContext owners.

## Browser benchmarks

The measured BPE1536 FP32/INT8 comparison across ONNX Runtime, native C,
single-threaded WASM, and WebGPU is recorded in the
[one-thread qualification report](../../docs/tiny-receipt-vqa-bpe1536-benchmark.md).

Measure a cold request followed by a hot request with the same image and
question:

~~~bash
: "${RECEIPT_VQA_DATA_ROOT:?set RECEIPT_VQA_DATA_ROOT to the dataset root}"
npx tsx --experimental-wasm-relaxed-simd \
  examples/tiny_receipt_vqa/tools/benchmark_wasm_cold_hot.mjs \
  build/tiny-receipt-w8a8 \
  "$RECEIPT_VQA_DATA_ROOT/eval/heldout/images/00002.jpg" \
  "phone number last one" \
  dist/0.3.0/volvoxai.wasm
~~~

Run the Chrome/WebGPU benchmark:

~~~bash
: "${RECEIPT_VQA_DATA_ROOT:?set RECEIPT_VQA_DATA_ROOT to the dataset root}"
npm run build
node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs \
  --timeout-ms=600000 \
  --url='examples/tiny_receipt_vqa/tools/webgpu_w8a8_benchmark_tinyreceipt.html?maxNewTokens=100' \
  --model-dir=build/tiny-receipt-w8a8 \
  --image="$RECEIPT_VQA_DATA_ROOT/eval/heldout/images/00002.jpg"
~~~

The reports distinguish package loading, model compilation, context seed
latency, steady decode latency, selected provider, and provider-reported device
identity when available.

## Full-profile PTQ authoring

The PTQ source is an F32 package:

~~~text
full_model/
├── graph.json
└── model.safetensors
~~~

graph.json contains the exact root discriminator:

~~~json
{
  "format": "volvox-graph/v1"
}
~~~

tools/export_trained_ptq_source.mjs validates the package, maps the
TinyReceipt-specific parameter names and layouts, runs representative inputs
through an ordinary Runtime/Model/CompiledModel/ExecutionContext lifecycle,
and observes declared F32 ExecutionResult outputs with PTQObserver.

The mapping is explicit:

- convolution HWIO to OIHW;
- Linear and LoRA IN_OUT to OUT_IN;
- packed self-attention QKV transpose;
- cross-attention Q/K/V transpose and row concatenation;
- expert tensors to per-family rows;
- positional rank expansion;
- tied token/head copy.

F32 norms and selected biases remain F32. `materializePTQWeights()` produces new
packed safetensors bytes plus explicit scale and zero-point tensors. The
application-specific materializer writes strict `volvox-graph/v1` JSON with a
central reference table and explicit QLinear, QConv2D, quantize, requantize,
and dequantize boundaries; no numeric affine value is stored in JSON.

A structural smoke checks the complete artifact path:

~~~bash
: "${RECEIPT_VQA_DATA_ROOT:?set RECEIPT_VQA_DATA_ROOT to the dataset root}"
make -C examples run_cpp_receipt_train_smoke \
  RECEIPT_VQA_DATA_ROOT="$RECEIPT_VQA_DATA_ROOT"

npx tsx examples/tiny_receipt_vqa/tools/export_trained_ptq_source.mjs \
  --source build/volvoxai-receipt-smoke/full_model \
  --out-dir build/volvoxai-receipt-smoke/ptq-source \
  --structural-smoke

python3 -m examples.tiny_receipt_vqa.tools.materialize_tiny_receipt_vqa_w8a8 \
  --manifest build/volvoxai-receipt-smoke/ptq-source/manifest.json \
  --activation-calibration \
    build/volvoxai-receipt-smoke/ptq-source/activation_profile.json \
  --out-dir build/volvoxai-receipt-smoke/w8a8
~~~

Structural smoke inputs prove package wiring only. They are not representative
calibration and make no accuracy claim. Release calibration must use named
samples from a deployment-representative dataset that is disjoint from the
heldout score set, with complete dtype- and shape-correct input arrays. Routed
models must execute every family/profile during calibration. Immutable
Embedding output ranges cover the entire table; observing only BOS/PAD IDs is
not representative of autoregressive decode.

For one isolated mapping diagnostic,
tools/materialize_trained_linear_ptq.mjs emits a standalone QLinear island. Its
manifest marks that artifact as not runnable as the full TinyReceiptVQA model.

The sole tensor-backed affine storage contract is documented in
[affine quantization in safetensors](../../docs/w8a8-safetensors.md).

## Native W8A8 applications

The legacy whole-model and qualified encoder/decoder applications are opt-in,
inference-only examples. Neither is linked into `native/volvoxai` or
`native/volvoxai-full`, and neither changes the fixed release filenames.

Both native applications expect `--prompt` to be valid UTF-8 that is already
normalized to NFC. The JavaScript and Python hosts normalize question text,
but these small C examples deliberately do not bundle a Unicode normalization
library. This matters for decomposed Korean text: normalize it before invoking
the binary or its token IDs will not match the legacy character tokenizer or
the split package's BPE tokenizer.

### Qualified encoder/decoder split

Build the split application and its self-contained native contract test:

~~~bash
make -C examples native_receipt_split_inference_example
make -C examples test_native_receipt_split_inference
~~~

Run a session-ready package produced by the row-qualified FP32 optimization,
calibration, and PTQ flow above. The package may retain runtime family routing
or may have been explicitly specialized. The manifest may still classify
unavoidable static or uncalibrated boundaries as hybrid. Packages produced
with decoder float-op exclusions are not row-qualified. Incremental decoding
is already the default; `--incremental` selects it explicitly. `--cpu` makes
CPU a required provider and forbids operator fallback. `--require-row` selects
and validates the decoder's incremental row executor, failing if it cannot be
used. It does not rewrite unsupported full-tensor operators into row kernels,
which is why the pre-PTQ grouped-projection cleanup is part of the package
build:

~~~bash
examples/target/bin/tiny_receipt_split_w8a8 \
  build/tiny-receipt-int8-from-fp32 \
  --image receipt.png \
  --prompt "What is the phone number?" \
  --family phone \
  --max-new 96 \
  --incremental \
  --cpu \
  --threads 1 \
  --require-row
~~~

The package must use the
`volvoxai-tiny-receipt-vqa-split-onnx-package-v1` manifest. The host accepts
both the I32 `token_ids` decoder ABI and the F32 `logits` ABI with host
first-index argmax, each with either a graph-derived or hoisted `v4_keep`
input. It reads the vocabulary size and tokenizer from the package: legacy
`char-vocab` remains supported, while the release BPE path requires the
qualified NFC `byte_fallback_bpe` v1 contract with 1,536 entries. For a
runtime-routed package, `--family phone`
passes that family ID into the encoder and verifies the selected route; `auto`
passes `-1` and accepts the router's valid selection. For a specialized package,
an explicit `--family` is a consistency check and the caller must choose the
package specialized for that family. These execution modes do not expand the
documented `f0`/`f1` qualification scope.

The split host uses two independent public native lifecycles:

~~~text
VxRuntime
├── encoder VxModel → VxCompiledModel → VxExecutionContext
└── decoder VxModel → VxCompiledModel → VxExecutionContext
~~~

It preprocesses the image and question, executes the encoder once, retains the
declared memory and mask outputs, resets and seeds the decoder once, then calls
`vx_execution_context_decode_step()` with only the changed IDs/keep inputs and
the current position. Every inference execution returns an owned `VxResult`,
which the application releases after copying the declared outputs.

#### Native heldout performance and accuracy

Use `benchmark_native_heldout.py` to compare any number of split artifacts on
identical sorted heldout annotations and images. The harness always invokes the
dedicated C application with `--incremental --cpu --require-row --timing`; there
is no benchmark option that can silently weaken that execution contract.
`--threads` controls the native CPU worker count and defaults to 1; pass it
explicitly in published benchmark commands so the report's `cpu_threads`
provenance is unambiguous. It
records encoder, first-token, steady-token, generation, and whole-process wall
times, extracts the generated `<answer>`, and reports ground-truth exact match
and cross-artifact answer/full-text agreement. Package assets, the executable,
annotations, and images are identified by content hashes without saving local
filesystem paths. `--timing` emits only application summaries; unlike `--debug`,
it does not enable per-node tracing that would perturb the measured execution.

~~~bash
: "${RECEIPT_VQA_DATA_ROOT:?set RECEIPT_VQA_DATA_ROOT to the dataset root}"

python3 -m examples.tiny_receipt_vqa.tools.benchmark_native_heldout \
  --eval "$RECEIPT_VQA_DATA_ROOT/eval/heldout" \
  --binary examples/target/bin/tiny_receipt_split_w8a8 \
  --package f32=build/tiny-receipt-f32-from-f32 \
  --package fp32-ptq=build/tiny-receipt-int8-from-fp32 \
  --package direct-exact=build/tiny-receipt-int8-from-int8-exact \
  --package direct-migrated=build/tiny-receipt-int8-from-int8 \
  --count 100 \
  --repeat 3 \
  --threads 1 \
  --report build/tiny-receipt-native-heldout.json
~~~

These paths are the outputs of the commands in this document, not aliases for
an older local build:

| Profile label | Authored artifact |
| --- | --- |
| `f32` | `build/tiny-receipt-f32-from-f32`: imported FP32 followed by FP32 graph optimization |
| `fp32-ptq` | `build/tiny-receipt-int8-from-fp32`: the same optimized FP32 graph followed by calibrated PTQ |
| `direct-exact` | `build/tiny-receipt-int8-from-int8-exact`: imported producer INT8 followed only by exact rewrites; F32 logits ABI |
| `direct-migrated` | `build/tiny-receipt-int8-from-int8`: producer-affine migrations plus `--canonical-deployment`; I32 token-ID ABI |

Do not fill a documentation table from terminal snippets or a previous package
revision. Treat `build/tiny-receipt-native-heldout.json` as the generated result:
its provenance binds the executable, package graphs, weights, annotations, and
images by content hash. Read exact match from
`summary.packages.<label>.exact_match`, latency distributions from
`summary.packages.<label>.timing`, and answer/full-text agreement from
`summary.cross_artifact_agreement`. Node inventories belong to the corresponding
packaged graphs and export reports; they are not inferred from an older timing
run.

Accuracy mode defaults to the complete 191-new-token decoder budget. A shorter
budget is rejected because it can truncate `<answer>`. Use an explicit latency
probe when a full answer is intentionally not required; its JSON accuracy
fields are `null`:

~~~bash
python3 -m examples.tiny_receipt_vqa.tools.benchmark_native_heldout \
  --eval "$RECEIPT_VQA_DATA_ROOT/eval/heldout" \
  --package candidate=build/tiny-receipt-int8-from-fp32 \
  --ids 00001,00002 \
  --max-new 96 \
  --latency-only \
  --threads 1 \
  --report build/tiny-receipt-native-latency.json
~~~

`--repeat` adds timing samples for the same case and package; repeated generated
text and selected routing must remain deterministic or the harness fails.

To measure autoregressive execution without a retained K/V cache, use the
growing-prefix mode:

~~~bash
examples/target/bin/tiny_receipt_split_w8a8 \
  build/tiny-receipt-int8-from-fp32 \
  --image receipt.png \
  --prompt "What is the phone number?" \
  --family phone \
  --max-new 96 \
  --cpu \
  --no-kv \
  --timing
~~~

`--no-kv` calls `vx_execution_context_execute_prefix()` with the current
prefix length on every token. It recomputes the growing decoder prefix and
retains no dependency or self-attention K/V state, matching the execution
shape of autoregressive hosts that rerun a fixed ONNX decoder prefix. It is
mutually exclusive with `--incremental`, `--ordinary`, and `--require-row`.
The default incremental mode remains the deployment path and is expected to
be much faster after its one-time seed.

On x86, model preparation persistently packs immutable symmetric-I8 dense and
eligible 3x3 convolution weights. AVX2 hosts use an exact K4/N16
`VPMADDUBSW`/`VPMADDWD` microkernel for multi-row prefix work, while measured
M=1 projections stay on the raw GEMV dispatcher when that is faster. The
pack records whether any weight is -128. Packs without that endpoint use the
exact `abs(input) * sign(weight,input)` identity, whose adjacent pair is
bounded by 32512; packs containing -128 retain the two-part activation split.
Both preserve canonical graph-visible bytes instead of inheriting the usual
U8S8 pair-saturation behavior. CPU feature checks select the optimized path at
runtime; the release binary remains a baseline-ISA binary and retains portable
fallbacks.

The encoder's dense 3x3 im2col path partitions output rows and copies
contiguous in-bounds 3*C strips. Large QSiLU tensors and normalization rows or
groups also use the shared pool. AVX2 QLayerNorm vectorizes only the
affine/requantization pass after scalar-order statistics, and AVX2 QGroupNorm
uses four independent groups as SIMD lanes. Decoder-sized QSiLU/LayerNorm work
stays serial to avoid per-layer worker wake-up; all dispatch is runtime-gated.

To diagnose graph semantics independently of incremental state, force a full
ordinary decoder forward for every generated token:

~~~bash
examples/target/bin/tiny_receipt_split_w8a8 \
  build/tiny-receipt-int8-from-fp32 \
  --image receipt.png \
  --prompt "What is the phone number?" \
  --family phone \
  --max-new 96 \
  --cpu \
  --ordinary \
  --debug
~~~

`--ordinary` is a diagnostic path and is mutually exclusive with both
`--incremental` and `--require-row`. It recomputes the complete decoder graph
each step, so its latency is not representative of the qualified incremental
deployment path.

### Debug against the legacy model

Regenerate the legacy package with the current materializer before using it
for native comparison. The package keeps the complete `model.safetensors` for
existing JavaScript consumers, and also records graph-scoped weight files for
the router and each explicit family. The native application prefers those
scoped files, so it does not register thousands of tensors belonging only to
the other seven families. Older v1 manifests without `weight_files` remain
supported through the original top-level `weights.file` fallback, but a large
all-family file can still exceed the native tensor-table capacity.

Do not use a package produced with the single `--activation-scale` fallback
for token or answer validation. Such a package is only a structural diagnostic;
its manifest explicitly records `qualified_per_edge_calibration: false`, and
the native application warns before inference. Materialize the legacy package
with the matching per-edge calibration profile before comparing ordinary and
incremental output. Decode caching can preserve a package's output, but it
cannot repair unqualified activation scales.

Build both applications, then run equivalent legacy and split packages with
the same image, prompt, family, token limit, and `--debug`. Capture stdout and
stderr together so the selected route, per-step token IDs, stop condition, and
seed/steady timing remain next to each answer:

~~~bash
set -o pipefail

make -C examples native_receipt_inference_example
make -C examples native_receipt_split_inference_example

mkdir -p build/tiny-receipt-native-debug

examples/target/bin/tiny_receipt_w8a8 \
  build/tiny-receipt-w8a8 \
  --image receipt.png \
  --prompt "What is the phone number?" \
  --family phone \
  --max-new 96 \
  --incremental \
  --debug 2>&1 | tee build/tiny-receipt-native-debug/legacy.log

examples/target/bin/tiny_receipt_split_w8a8 \
  build/tiny-receipt-int8-from-fp32 \
  --image receipt.png \
  --prompt "What is the phone number?" \
  --family phone \
  --max-new 96 \
  --incremental \
  --cpu \
  --require-row \
  --debug 2>&1 | tee build/tiny-receipt-native-debug/split.log
~~~

Use packages derived from the same checkpoint, vocabulary, frozen family, and
comparable calibration profile.
Compare semantic debug fields and generated tokens rather than byte-diffing the
logs: timing values and graph-specific labels are expected to differ. If token
parity fails, rerun only the split command with `--ordinary --debug`. Matching
ordinary output localizes the discrepancy to incremental decode state; a
mismatch in both modes localizes it to package conversion, input binding, or
graph execution.

The legacy application validates its router-plus-family manifest, runs the
router, and selects one whole-model family graph. Ordinary per-token forwards
are its default. With `--incremental`, it seeds the fixed-shape graph once and
then uses decode steps that retain the image/encoder and cross-attention
branches and select row/KV execution when supported. The split application
instead defaults to incremental decoding; passing `--incremental` makes that
selection explicit. Both applications apply backend policy through
`vx_model_compile()`; an explicitly required unavailable provider fails during
compilation.

The CTest examples prove build wiring, manifest rejection, lifecycle behavior,
and small deterministic fixtures. They are not native qualification evidence.
Qualification requires comparison with the versioned oracle on representative
heldout inputs (router tolerances plus exact selected family, token IDs, and
decoded text), strict-provider evidence with forbidden fallback, and
reproducible cold, encoder, seed, steady-token, total-latency, and percentile
measurements on the shipping native target.

## Training ownership

JavaScript training uses volvoxai/full:

~~~javascript
import { VolvoxAI } from 'volvoxai/full';

const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
const model = runtime.createModel(trainingGraph);
const trainer = await VolvoxAI.createTrainer(model, { backend: 'cpu' });

const step = await trainer.trainStep({
  inputs,
  logitsTensor: 'logits',
  targets,
  trainableTensors,
  updateMode: 'adamw',
  optimizer: {
    learningRate: 1e-3,
    maxGradNorm: 1,
  },
});
await trainer.commit();

console.log(step.loss, step.updatedTensorNames);
~~~

Trainer privately clones the Model revision. `trainStep()` mutates only that
private working revision; `commit()` publishes a successor Model revision
atomically. There is no implicit publication. `rollback()` discards uncommitted
work and restores the last committed baseline. Step results expose copied
gradients and stable tensor-name metadata, not mutable graph tensor handles.

The application owns dataset discovery, image preprocessing, vocabulary,
batching, learning-rate scheduling, metrics, checkpoint retention, and PTQ
release policy.

## Tests

~~~bash
python3 -m unittest discover \
  -s examples/tiny_receipt_vqa/tests \
  -p 'test_*.py'

make -C examples test_native_receipt_inference
make -C examples test_native_receipt_split_inference
~~~
