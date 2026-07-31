# VolvoxAI graph optimizer design

Status: current implementation and qualification record. `volvox-graph/v1` is
the only supported graph contract. This design has no compatibility reader,
migration mode, or obligation to preserve older graph spellings.

Owner track: model-neutral deployment compiler
(import → optimize → quantize → qualify → ship).

## 0. Decision and current pipeline

VolvoxAI owns one offline compiler pipeline for ONNX, TensorFlow Lite, and
already-current VolvoxAI packages:

```text
ONNX / TensorFlow Lite bytes / volvox-graph/v1
                  │
                  ▼
lossless SourceIR + source-fidelity audit
                  │ explicit, legality-checked lowering
                  ▼
verified RuntimeIR
        ├── FP32 structural optimization and high-level fusion
        │       │ optional explicit input/value specialization
        │       ▼
        │ exact optimized FP32 revision
        │       │ representative, route-covering calibration
        │       ▼
        │     typed PTQ
        │
        └── existing static-QDQ INT8 graph
                │ preserve producer activation and weight affines
                │ exact typed packed-projection split
                │ exact affine-reference and layout cleanup
                │ optional static-QDQ compute + attention numerical migration
                ▼
post-PTQ Q/DQ cleanup and explicit output specialization
                                         │
                                         ▼
strict verification + independent differential execution
                                         │
                                         ▼
task score + routing + kernel + paired latency + build gates
                                         │
                                         ▼
atomic graph.json + safetensors publication
                                         │
                                         ▼
runtime CompiledModel backend preparation for the exact package revision
```

`tools/exporter/` is a general-purpose, model-neutral library and command layer.
It owns SourceIR/RuntimeIR, import and lowering, reusable graph passes,
calibration/PTQ primitives, verification, differential execution, and atomic
publication mechanisms. The protobuf optimizer registry selects the ordered
typed pipeline; application code supplies explicit feature requests and pass
inputs, not another pass list. The exporter does not own a model's package
manifest, graph pairing, family vocabulary or routing, task preprocessing, or
release policy.

TinyReceipt-specific encoder/decoder composition, package manifests, family
routing and specialization bindings, representative calibration records, and
qualification live under `examples/tiny_receipt_vqa/`. Those application tools
request registered model-neutral recipes and provide model data; they are not
another public IR, pass framework, or general exporter contract.

The release rules are:

1. Import is lossless before lowering. Unknown attributes, buffers, subgraphs,
   quantization records, and operator versions are preserved or rejected with
   an explicit diagnostic.
2. Lowering declares legal source and target operations. Every graph is
   verified before and after a rewrite.
3. Rewrites are transactional. A failed pass leaves graph and tensor bytes
   unchanged.
4. Exact optimization is the default. A numerical migration requires explicit
   opt-in and independent differential, task, and backend-profile
   qualification. Quantization authoring requires an immutable calibration
   profile for the exact graph it changes; it is not an exact rewrite.
5. Structural and semantic optimization precede calibration. Specialization
   also precedes calibration when the caller explicitly requests it; runtime
   routing remains live by default. A calibration profile is accepted only for
   the exact graph fingerprint it observed and the route set it declares.
6. Graph JSON and safetensors publish atomically. Numeric affine parameters
   live in safetensors, never inline in graph JSON.
7. Qualification forbids backend operator fallback and compares meaningful
   intermediate values as well as public outputs.
8. Node count is diagnostic. Publication depends on task accuracy, routing,
   paired latency distributions, kernel evidence, and inference/full builds.

## 1. The de-facto optimizer model

There is no single cross-framework “standard optimizer.” Mature runtimes use a
staged compiler: portable semantic rewrites first, then backend preparation
and measurement on the actual target. In VolvoxAI, backend selection and
physical compilation belong to `CompiledModel`; they are not a second offline
deployment artifact.

| System | Official mechanism | Lesson for VolvoxAI |
| --- | --- | --- |
| ONNX Runtime | [Graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), [execution-provider partitioning](https://onnxruntime.ai/docs/execution-providers/), [ORT model format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html), and [quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) | Separate target-independent rewrites from provider/layout rewrites; preprocess and optimize before quantization; measure Q/DQ overhead instead of assuming INT8 is faster. |
| TensorFlow | [Grappler](https://www.tensorflow.org/guide/graph_optimization) | Run configurable graph passes around a stable graph representation rather than embedding rewrite policy in kernels. |
| TensorFlow Lite / LiteRT | [TensorFlow conversion and compatibility](https://ai.google.dev/edge/litert/conversion/tensorflow/overview) and [post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) | Make conversion legality explicit; full-integer activation quantization needs representative data and deployment-target validation. |
| TensorRT | [How TensorRT works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html), [explicit quantized types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html), and [performance optimization](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html) | Keep portable network semantics separate from explicit Q/DQ precision, target engine building, fusion, measured tactic selection, timing caches, and serialized binaries. |
| MLIR | [Canonicalization](https://mlir.llvm.org/docs/Canonicalization/), [dialect conversion](https://mlir.llvm.org/docs/DialectConversion/), and [bufferization](https://mlir.llvm.org/docs/Bufferization/) | Use bounded canonicalization, declared legality during lowering, and delay physical buffer/layout decisions until tensor semantics stabilize. |

The resulting VolvoxAI order is deliberate:

1. Canonicalize and fuse representable FP32 semantics.
2. Specialize only values and ABI choices explicitly bound by the caller.
3. Calibrate that exact graph with representative, disjoint training records:
   every public family for runtime routing, or the one explicitly specialized
   family.
4. Apply a measured mixed-precision policy transactionally.
5. Remove redundant Q/DQ boundaries and specialize terminal outputs only when
   the caller explicitly requests the ABI change.
6. Verify and differentially execute the result.
7. Publish the final graph and safetensors as one package. A runtime
   `CompiledModel` then derives its provider plan, physical layouts, memory,
   and target tactics for that exact immutable package revision.

"Fuse before quantize" and "optimize after quantize" are both valid, but for
different rewrite classes. High-level semantic fusion normally happens while
scales, masks, and intent are recoverable. Exact post-PTQ cleanup is limited to
rewrites whose byte-domain equivalence is proven. The separately requested
quantized-attention lift is a **numerical migration**, not part of that exact
class: it preserves INT8 weights and the input package's affines but removes
score and probability requantization boundaries when emitting QSDPA. Its structural
proofs establish layout, affine, and mask legality; they do not prove bit-exact
outputs. Backend fusion of an explicit boundary with its consumer remains a
physical preparation decision.

## 2. One current package contract

Every persisted graph has the exact discriminator:

```json
{
  "format": "volvox-graph/v1"
}
```

The current contract is strict:

- each input has explicit `dtype`, and `outputs_dtype` exactly describes every
  node output port; there is no implicit F32 default;
- nodes use `opType`; the old `op` alias is rejected;
- `params` is an object;
- operator spellings must resolve through the generated `OperatorKind`
  vocabulary and the explicit registry mapping;
- affine metadata in graph JSON is a central
  `volvox-affine-safetensors/v1` reference table only;
- every numeric scale and zero point is a rank-one safetensors tensor;
- graph and safetensors changes commit together and are content-hashed.

The graph JSON describes topology, dtypes, semantics, and tensor references.
Safetensors is the numeric authority for weights and quantization data. This
keeps scales deduplicated, typed, immutable, and covered by the same artifact
identity as the weights. See [model-format.md](model-format.md),
[quantization.md](quantization.md), and
[w8a8-safetensors.md](w8a8-safetensors.md).

`proto/volvoxai.proto` is the authored source of truth for the shared
`DataType` and logical `OperatorKind` vocabulary. The kernel and optimizer
registries import those enum values, and code generation keeps Python,
TypeScript, and native projections aligned; no JSON capability list or
handwritten enum copy is authoritative. The proto is not an optimizer RPC and
does not contain pass order or backend kernel inventory. Graphs keep the current
CamelCase `opType` spelling, with the generated string-to-enum projection used
by the strict RuntimeIR loader.

`proto/kernel_registry.proto` is the separate internal source of truth for
backend route inventories and strict exporter target qualification. The
exporter consumes its generated Python projection; it does not maintain a
second handwritten op-by-target table. Static membership is only the first
gate: descriptor legality, device limits, and no-fallback evidence remain
fail-closed predicates identified by the registry and evaluated by code. The
generated backend matrix is published at
[`docs/generated/kernel-registry.md`](generated/kernel-registry.md).

`proto/optimizer_registry.proto` is the corresponding authored source of truth
for typed passes, pass groups, recipes, semantic effects, ordering, and target
requirements. Its generated Python catalog drives recipe resolution;
`typed_pipeline.py` does not maintain a second handwritten pass order. Stable
implementation IDs resolve to reviewable transactional `IRPass` code, while
target and recipe applicability are evaluated directly from typed generated
fields. Generation rejects unknown operators, passes, backend names, kernel
variants, ordering cycles, and inconsistent fixed-point groups. The generated
inventory is published at
[`docs/generated/optimizer-registry.md`](generated/optimizer-registry.md).
Neither registry is parsed by inference execution. A rewrite implementation
that is not registered cannot enter a publishing pipeline; a model-specific
tool cannot replace the registered order with a private dictionary pipeline.

## 3. Import, IR, and lowering

SourceIR and RuntimeIR serve different purposes:

- **SourceIR is lossless.** It retains interchange facts needed for audit,
  diagnostics, and later legal lowering. A frontend must not silently erase an
  unsupported attribute or quantization record to make a model appear usable.
- **RuntimeIR is executable.** It contains only current VolvoxAI operations,
  explicit dtypes and shapes, tensor references, and verified dataflow.
- **Lowering is explicit.** Each rule identifies accepted source forms, emitted
  operations, shape/dtype conditions, and failure diagnostics.

The ONNX importer is connected to this typed pipeline. TensorFlow Lite has the
same lossless SourceIR boundary, but complete TensorFlow Lite-to-RuntimeIR
lowering remains unfinished. Until that lowering is complete, a successful
TFLite parse/classification is not a promise that every model can be published.

An imported INT8 graph does not bypass optimization. It enters at verified
RuntimeIR, retains its source quantization facts, and is eligible for proven
Q/DQ cleanup. An imported FP32 graph follows structural fusion and, only when
the caller requests it, input/value specialization before representative
calibration and PTQ. Target-specific preparation happens only when a runtime
compiles the final package.

## 4. Portable optimization contract

Optimization carries three independent target identities:

| Identity | Controls | Must not control |
| --- | --- | --- |
| backend profile | legality of the persisted portable graph across every profile member | which machine happened to supply timing data |
| compile backend | derived kernel/layout/memory preparation owned by one `CompiledModel` | persisted graph vocabulary or quantization authority |
| tune backend/device | origin of measured cost evidence and timing-cache keys | package portability or another backend's compiled plan |

Those axes have separate content identities. The legality fingerprint contains
only the backend profile; the compile fingerprint adds the compile backend,
features, device/toolchain identity, and fallback policy; the measurement
fingerprint adds the tune backend, features, and device. Changing a browser or
benchmark host therefore invalidates timing evidence without invalidating the
portable-graph proof or an unrelated native/WASM compilation.

Registry pass semantics are independent of those target identities:

| Rewrite semantics | Contract | Required authority |
| --- | --- | --- |
| exact | Proves the represented value or storage transformation unchanged. Exact integer and quantized byte-domain results remain byte-identical. | Enabled by default after fail-closed legality and verification. |
| numerical migration | Intentionally changes a graph-visible rounding boundary, floating-point evaluation order, or quantized arithmetic placement. Algebraic equality alone is not an exact proof. | Explicit caller opt-in plus differential, task, and every-profile-member qualification. |
| quantization authoring | Creates a new precision contract, affine parameters, and quantized payloads from an immutable profile observed on the exact input graph. | Explicit PTQ request and matching calibration; never inferred for an already-quantized import. |

A public ABI change is an orthogonal effect: a pass can be exact for retained
values while adding, removing, or renaming an input or output. It therefore
requires its own explicit authorization even when its rewrite semantics are
`exact`.

For example, backend profile `portable`, compile backend `wasm`, and tune
backend `wasm` still publish a graph legal on `cpu-js`, `wasm`, `webgpu`, and
`native-cpu`. A later native `CompiledModel` compiles the same package and
selects native kernels. Selecting WASM for compilation or measurement never
turns the package into a WASM-only graph. A portable rewrite that emits an
operator is legal only when every backend-profile member qualifies that
operator and every declared required kernel condition. Target-specific fusion,
packing, scheduling, and code generation instead produce revision-bound
`CompiledModel` state and leave RuntimeIR unchanged.

All profile members execute the same logical operator contract: dtype and shape
rules, affine rounding and saturation, ArgMax ties, and attention mask meaning
do not vary by backend. Exact integer and quantized conformance is byte-for-byte.
Floating-point kernels are compared with declared operator tolerances because
parallel reduction and device arithmetic do not guarantee cross-device FP32 bit
identity; routing and public task results must still agree. A backend-only
physical fusion may remove buffers or dispatches only while preserving every
graph-visible Q/DQ and rounding boundary. If it cannot, the change belongs in
the portable candidate as an explicit numerical migration and is qualified on
all members of the selected profile.

The typed portable pipeline currently provides vocabulary checking,
shape-chain simplification, canonicalization, exact packed-QLinear splitting,
redundant Q/DQ removal, dead-code elimination, and explicit terminal output
ArgMax specialization. Typed dense PTQ is documented in
[typed-ptq.md](typed-ptq.md). Pipeline-eligible attention, layout, precision,
and cleanup passes follow the same registry and typed RuntimeIR contract. An
application may request registered features and supply bindings or calibration,
but no TinyReceipt package, family policy, or private pass order belongs in the
general optimizer.

The reusable typed structural pass library also exposes four model-neutral
building blocks. `RuntimeSiluFusionPass` recognizes only the canonical F32
`Sigmoid(x) -> Mul(x, sigmoid)` form. The formula is algebraically equal, but a
fused SiLU kernel can evaluate or round F32 differently, so the registry
correctly classifies it as a numerical migration with no ABI change.
`RuntimeConstantFoldingPass` evaluates only immutable storage/value-selection
operations and canonicalizes a proven constant-right MatMul to `Linear`; it
deliberately does not evaluate general F32 arithmetic and keeps the public ABI
unchanged. `RuntimeInputSpecializationPass` is exact for bytes explicitly bound
by the caller but removes that public input, while
`RuntimeInputHoistingPass` is exact only under the caller obligation to supply
the identical derived value and adds a public input. Hoisting is ordered after
value-producing rewrites. A caller may select either an existing tensor or a
versioned `semantic_id` plus source-value identity recorded by the producing
rewrite; it never predicts a generated tensor name. Identical proved attention
keep-mask conversions reuse one derived value before that value is hoisted.
Those two passes are therefore explicit ABI changes, never inferred
optimizations. They record the source/exported contracts (including a
specialized payload hash), attach typed provenance, and commit verified
RuntimeIR plus mutable tensor data transactionally.

`RuntimePackedQLinearSplitPass` is the post-import rule for static INT8 QDQ
graphs. It recognizes
`QLinear -> DequantizeLinear -> optional F32 Add -> movement -> Slice`, then
slices the OUT_IN byte weight, I32 bias, and per-axis weight scale/zero point
for each projection. It reuses the packed output's per-tensor affine, preserves
the F32 bias and later Mul/QuantizeLinear boundaries, and dead-code eliminates
the unreferenced packed tensors. Those constraints make the rewrite exact in
the represented byte domain. The serialized weight bytes and the input graph's
current affine parameters remain authoritative: the pass neither reconstructs
FP32 weights nor quantizes a weight a second time. The default does not infer
an attention fusion or remove a numerically visible Q/DQ boundary.

Each registered pass and its executable implementation define:

- accepted dialect and operator forms;
- executable checks for required shape, dtype, constant, alias, and
  quantization facts;
- whether it may alter the public ABI;
- a bounded rewrite budget or fixed-point limit;
- verification and differential-test expectations.

Optimization is a transaction over graph plus tensor store. A candidate is
verified before commit; failure restores the exact prior bytes. Output pruning
must remove unreachable tensor payloads, while target-private packed weights
remain derived caches and never become the logical weight or quantization
authority.

The reusable quantized-region analysis discovers maximal F32 compute regions
bounded by explicit Q/DQ edges. It records producer affine domains, static and
broadcast operands, auxiliary inputs, public/escaping values, and missing
intermediate affines. Discovery itself never rewrites a graph. A closed region
can yield two different candidate records:

- a portable logical rewrite, which requires a registered typed pass and the
  declared numerical-migration policy; or
- a backend-only `CompiledModel` fusion, which must preserve the graph's Q/DQ
  rounding boundaries and F32 evaluation order while eliminating physical
  dispatch or buffer overhead.

This distinction prevents a fast WASM schedule from silently becoming a new
portable operator contract.

Run the current typed optimizer with:

```bash
python3 -m tools.exporter.optimizer path/to/graph.json \
  --weights path/to/model.safetensors \
  --out optimized.graph.json \
  --out-weights optimized.safetensors \
  --backend-profile portable \
  --compile-backend wasm \
  --tune-backend wasm \
  --compile-feature wasm.simd128 \
  --tune-feature wasm.simd128 \
  --compile-device-fingerprint wasi-sdk-29 \
  --tune-device-fingerprint chrome-140-linux-x86_64 \
  --report optimizer-report.json \
  --compiled-plan wasm.compiled-plan.json

# Or transactionally replace the input package after staging and validation.
python3 -m tools.exporter.optimizer path/to/graph.json \
  --weights path/to/model.safetensors \
  --in-place \
  --report optimizer-report.json
```

Strict verification is unconditional. Omitting `--out` performs a dry run;
publication requires either both `--out` and `--out-weights`, or `--in-place`,
so graph and safetensors commit as one rollback-safe transaction. The optional
pass report is diagnostic output, not a deployment input. `--backend-profile`
defaults to `portable`; it alone controls persisted-graph legality. Compile and
tune backends, their repeatable `--compile-feature`/`--tune-feature` facts, and
their independent device fingerprints are recorded explicitly and do not
change pass selection for that profile.

General model-neutral overlays are explicit CLI choices:
`--allow-static-qdq-compute-migration`,
`--allow-float-attention-migration`,
`--allow-quantized-attention-migration`,
`--defer-static-qdq-layout-optimization`, and
`--prepare-fp32-for-ptq`. They resolve through the protobuf recipe and appear
in the report's exact feature/group/pass identity. Numerical-migration flags do
not recalibrate or rewrite imported affines, and `--prepare-fp32-for-ptq` ends
before calibration; PTQ authoring remains a separate fingerprint-bound stage.

`--compiled-plan` requires an explicit `--compile-backend`. It atomically writes
a derived `volvox-compiled-model-plan/v1` JSON document from the optimized typed
graph. The plan records backend routes and eligible physical variants, but the
planner cannot write backend-specific nodes, layouts, or packed payloads into
`graph.json`. Report, plan, graph, and safetensors paths must be distinct. The
same published graph can therefore produce another backend plan without being
re-optimized or republished:

```bash
python3 -m tools.exporter.optimizer optimized.graph.json \
  --weights optimized.safetensors \
  --backend-profile portable \
  --compile-backend native-cpu \
  --tune-backend native-cpu \
  --compile-feature x86.avx2 \
  --compiled-plan native-cpu.compiled-plan.json
```

The JSON plan is currently a strict, content-addressed inspection/search
artifact, not an alternate RuntimeIR package and not yet a native or TypeScript
runtime input. `CompiledModelPlan.from_dict()` rejects unknown fields, a stale
plan ID, or kernel-selection evidence bound to another graph, weight revision,
node, backend, device feature set, or kernel-registry revision. A physical
variant can be marked selected only by an executable predicate returning this
typed evidence; copying a predicate ID string into a plan is not proof.

`--output-argmax logits=token_ids` is an explicit output-ABI specialization;
the optimizer never guesses that change.

TinyReceipt's application-owned command composes the typed optimizer over both
graphs and refreshes its application package identities:

```bash
python3 -m examples.tiny_receipt_vqa.tools.optimize_direct_int8 \
  --source build/tiny-receipt-int8-imported \
  --out build/tiny-receipt-int8-from-int8-exact
```

The packed-QLinear rule is part of this default command. It preserves the input
package's affines and intentionally keeps `QBatchMatMul`/Softmax plus the
explicit intermediate precision boundaries. QSDPA can have a different
performance result on each backend, and Q* compute can round differently from
separate DQ/F32/Q evaluation, so the comparable Direct INT8 deployment opts
into both numerical migrations and the application-owned canonical ABI
explicitly:

```bash
python3 -m examples.tiny_receipt_vqa.tools.optimize_direct_int8 \
  --source build/tiny-receipt-int8-imported \
  --out build/tiny-receipt-int8-from-int8 \
  --fuse-attention \
  --fuse-static-qdq-compute \
  --canonical-deployment
```

The command resolves one protobuf-authored transaction containing the same
exact packed/layout groups and the requested migration groups. It recognizes the static-QDQ
`QBatchMatMul -> DQ -> Add/Softmax -> Q -> QBatchMatMul -> DQ`
structure. It proves the head layout and mask conversion, reuses the original
byte operands and affines, and emits canonical `QSDPA` without decoding or
requantizing weights. It also removes the source graph's score and probability
requantization operations. Consequently this lift is intentionally **not
bit-exact**. The compute flag separately replaces only closed canonical
`DQ -> Add/LayerNorm/GELU/SiLU/GroupNorm -> Q` islands with their Q* operators,
reusing the existing input/output affine references. It performs no calibration
and changes no initializer payload, but the fused kernel's F32 evaluation or
reduction order can differ, so this is also a numerical migration. Neither flag
inherits the default package's correctness or latency evidence.
`--canonical-deployment` is a separate TinyReceipt ABI specialization performed
after those migrations: it hoists canonical I32 `v4_keep` and replaces F32
`logits` with byte-domain first-index I32 `token_ids`. It preserves producer
affines and initializer payloads and requires both migration flags. Without
that flag, even a graph using both numerical migrations retains the producer
logits ABI. Failure to legalize every recognized attention block aborts the
transaction; successful legalization only creates a candidate. Independent
differential execution, task accuracy, routing, strict-backend kernel evidence,
and paired latency qualification are required for that exact package identity
before release.

The unfused form is not inherently incompatible with native incremental
decode. Native `QBatchMatMul` has a one-row path for the strict batch-1
sequence layout when the right operand is complete, non-overlapping, and clean
in the current dependency closure. Otherwise it remains a whole-tensor
operation, and `--require-row` refuses a graph whose full changing closure is
not row-capable. This per-operator capability is not, by itself, qualification
of a direct imported package.

Direct static INT8 import has one affine contract: preserve the producer's
activation scales, zero points, weight scales, and serialized INT8/U8 weight
bytes. No second activation-profile collection, recalibration, or affine rewrite
exists for an already-quantized ONNX model, and the importer exposes no such
mode or profile option. The generic exporter normally enables exact layout
cleanup; an importer that must run a structural fusion first can pass
`--defer-static-qdq-layout-optimization` and invoke the same cleanup in its
later explicit optimizer stage. Deferral changes only pass ordering, not the
affine or payload policy. Exact post-import passes may intern
only byte-identical scalar affine references, replace singleton-only Transposes
with storage-only Reshapes, cancel proven inverse Transposes, hoist a pointwise
island across inverse Transposes, and commute a closed same-affine
`DQ -> layout -> Q` chain into the byte domain. They keep numeric affine values
and weight payloads unchanged. Opt-in compute and attention migrations also
reuse those existing producer affines; they never reconstruct or requantize
weights.

There is no second deployable graph schema. The deployable semantic artifact is
exactly the optimized `volvox-graph/v1` document and its safetensors. The
optional `volvox-compiled-model-plan/v1` document described above is a derived,
content-addressed inspection/search record; current runtimes do not consume it
as a model package. Provider selection, partitioning, kernel preparation,
target memory, and executable caches remain owned by runtime `CompiledModel`
preparation and are recreated for the exact model revision when needed.

## 5. PTQ, specialization, and mixed precision

PTQ is the registry's `quantization-authoring` stage, not another exact cleanup
pass. First produce and freeze the intended FP32 RuntimeIR revision, then
calibrate that exact fingerprint, then invoke the PTQ recipe with the immutable
profile; only exact cleanup may follow in that transaction. An imported static
INT8 graph does not enter this authoring stage merely because it is being
optimized, and its producer affines are never recalibrated.

PTQ consumes a verified optimized graph and a calibration profile bound to that
graph's identity. A runtime-routed graph remains runtime-routed: quantization
preserves the exact ordered public input/output descriptors and does not know
whether an I32 input represents a family, route, language, tenant, or ordinary
model data. Its legality requires complete ranges for the tensors demanded by
the PTQ plan, not application-category coverage. Representative-record identity
and actual numeric graph execution count are recorded separately, because one
immutable record may be executed through several public routes. The latter is
the sample count used to bind aggregate ranges; the calibration artifact digest
still identifies the exact observations.

Application-owned route sweeps remain valuable qualification evidence. For
TinyReceipt, the publisher records every catalog family with an explicit count,
including zero for an unobserved route, without rejecting PTQ or changing the
family input ABI. A route-specialized graph may only be produced by the separate
explicit input-specialization recipe before calibration. Observer policy,
backend, source hashes, and training-record provenance remain auditable, and
calibration records remain disjoint from heldout evaluation.

The quantizer must prove supported accumulator and multiplier ranges, publish
saturation/reconstruction diagnostics, and materialize graph plus safetensors
atomically. Runtime JSON never contains numeric activation or weight scales.

Mixed precision is a first-class policy, not a failure to “finish INT8.” Keep an
operator class in float when the measured accuracy/latency frontier requires it,
and place explicit Q/DQ boundaries at the policy edges. The portable graph makes
those boundaries visible; a backend compiler may later fuse their physical
implementation without changing graph semantics.

Offline search uses an immutable, content-addressed optimization workspace. A
workspace lineage binds its source graph and weights to one optimizer and kernel
registry revision. Every root candidate must name those exact source artifacts,
and every derived candidate must reach such a root through its content-addressed
parent chain. A candidate identifies its exact graph and weight hashes, backend
profile, precision contract, ordered transforms, and parent revision.
Each transform binds its registry pass ID and semantic class to a hash of the
currently active implementation module. Deserialization recomputes that hash
and rejects stale records, so timing evidence cannot silently survive a pass
implementation change.
Timing-cache keys additionally bind an immutable compiled-model plan plus the
measuring backend, compiler, device, and workload. Profiles record
graph/region/node p50 and p95 latency, compile and packing time, scratch and
artifact sizes, and quality loss. Pareto selection compares different
candidates, precision contracts, and compiled plans only within one workspace
lineage, backend profile, measured backend, and workload. It rejects mixed
lineage instead of merging timings from unrelated source/config revisions.
Selection applies hard resource/quality constraints and then computes a
deterministic frontier; timing data is evidence for choosing a candidate, not
authority to mutate graph semantics.

The precision contract is intentionally open to future W4A8/W4A16 work: it
records logical activation, weight, accumulator, and output dtypes, scheme,
group size, and a content-bound quantization plan. INT4 support therefore does
not mean decoding an existing INT8 tensor to FP32 and quantizing it again. A
future INT8-to-INT4 authoring pass must explicitly declare a lossy precision
change and derive its new packed weights from the current authoritative source
revision under a qualified plan. Backend-specific nibble packing remains a
rebuildable `CompiledModel` cache.

Value specialization is legal only when the caller binds the value. The
optimizer then records the ABI change, evaluates affected pure subgraphs,
folds newly constant weights, and prunes unreachable banks before calibration.
The generic typed input specialization and constant-folding passes provide
those model-neutral primitives; dead-code elimination performs the final bank
pruning. A model-specific tool may bind a route and qualify the resulting
artifact, but route names, family values, and policy do not enter the passes.

## 6. TinyReceipt qualified result

The current qualification scope is deliberately narrow: family `f0` (`phone`)
and family `f1` (`address`) only. Families `f2` (`store`), `f3` (`item_row`),
`f4` (`item_math`), `f5` (`item_lookup`), `f6` (`math`), and `f7` (`other`) are
explicitly unqualified. No score or performance claim is made for them.

### Exact specialized calibration

Each qualified family was specialized before calibration, then calibrated on
eight disjoint training records with two decoder prefixes per record on WASM.
Both runs covered encoder and decoder and reported zero routing mismatches.

| Family | Records | Prefixes per record | Backend | Route mismatches | Records SHA-256 | Calibration profile SHA-256 | Published provenance artifact SHA-256 |
| --- | ---: | ---: | --- | ---: | --- | --- | --- |
| `f0` phone | 8 | 2 | WASM | 0 | `fe885da4b7d8a4771b7beffb22f489ae9005a4f45e56c6bc9fc8ec6dbae0387e` | `338a401deb0945f870d03a54cad983c3670edb1ce5b08e9f2556d1c6fb7136a3` | `2a96e5d65d0af30cdfc47edce388ed9f3098cda824fe00c4f7e415dee3356c95` |
| `f1` address | 8 | 2 | WASM | 0 | `15a42fe8647786392f38cbafa2007460cc57da9f4bc9a966a9d17fd0b4305604` | `ed8eaf5fc2c1315a6f35fd74dc0c6bfc51224b31fcde09b492d8192cb38abcf0` | `574ac085f1dcf399cb6948b315e34c37505b57912868d3a0c51451d5454f9c29` |

The published provenance artifact differs from the pre-publication profile
because publication finalizes and sanitizes provenance. Both identities are
recorded so the result can be audited without treating a path string as model
authority.

Shared source identities for both qualified packages are:

| Source | SHA-256 |
| --- | --- |
| producer source manifest | `1ecbfacad9c56173b299a26ffc19fcb37f997952c85e31765cc0134b7016e92f` |
| encoder ONNX | `1889e1f5ede379489d38192597d3c70ad0dfaf966aa82ef609df99704d816451` |
| decoder ONNX | `fd98d9f0a30c1fac869cca668ae2dcaa4ea260efe1ad3d2740e4f6677b48d494` |
| calibration implementation | `06817b82d8ad3c8d6432b8e0f39b0feab52f6cdcf2a362ba3dc23139ba77e431` |
| specialized pre-PTQ encoder graph | `e1734e3e7a5bdb5225b155e31897e418c1c19dd6f0ce2aa3fddb1cb63b79bd73` |
| specialized pre-PTQ decoder graph | `f1d35eaed7d3d1c934071a00896e41e9389e74bdf20213ec434dad287f201133` |

### Generated topology and runtime evidence

The calibration identities above document historical specialized inputs; they
do not supply current graph topology, accuracy, or performance evidence for a
new package revision. Node inventories must come from that package's encoder
and decoder graphs and export reports. Kernel microbenchmarks remain useful for
physical-kernel selection, but they do not substitute for end-to-end evidence.

## 7. Heldout evidence and interpretation

Generate heldout evidence for the exact package identities under qualification.
For the current TinyReceipt native comparison, the application-owned profiler
runs FP32-from-FP32, INT8-from-FP32, exact INT8-from-INT8, and canonical migrated
INT8-from-INT8 on identical sorted cases with strict native CPU, incremental
decode, required row execution, and application-only timing. Its versioned JSON
report binds all package assets, the executable, annotations, and images by
content hash and records timing distributions, exact match, route selection,
and cross-artifact answer/full-text agreement.

The report-generation command and stable package paths live in
`examples/tiny_receipt_vqa/README.md`. Numeric results belong in a report or a
release record that cites its digest; this design document intentionally has no
floating “current” node-count, latency, or accuracy table. Heldout evaluation
must remain disjoint from calibration and must not select scales, weights,
operator exceptions, or routing policy. A legacy artifact can be an additional
compatibility oracle only when its calibration overlap is disclosed; it is not
automatically a clean accuracy baseline.

## 8. Why the old path lost, and what remains

The original “1:1 ONNX import” was slower than the specialized legacy package
because graph equivalence is not execution equivalence. It retained provider-
oriented Q/DQ boundaries, layout conversions, generic route/control flow, and
intermediate materialization. ONNX Runtime's own quantization guidance warns
that Q/DQ overhead can erase INT8 gains. The legacy package had already baked
the family choice and adapter weights into a compact graph and emitted the
terminal token ID, so it was a pre-specialized deployment artifact rather than
merely a quantized interchange graph.

There are now two supported packed-projection cleanup paths, with deliberately
different contracts:

| Input path | Default exact rewrite | Resulting execution contract |
| --- | --- | --- |
| FP32 structural optimization before PTQ | `RuntimeGroupedProjectionSplitPass` splits packed self- and cross-attention Q/K/V before calibration; runtime routing stays live unless the caller explicitly specializes it | Session package after calibrated PTQ, attention legalization, and `token_ids` output specialization; qualification still applies per route and target |
| Imported static INT8 QDQ | Preserve producer activation/weight affines, use `RuntimePackedQLinearSplitPass`, then run exact affine-reference and layout cleanup without changing numeric affine or weight payload bytes; `--fuse-attention --fuse-static-qdq-compute` explicitly enables producer-affine numerical migrations | The no-flag exact package retains F32 logits and an explicit host argmax; adding `--canonical-deployment` after both migrations adds `v4_keep` and I32 `token_ids`, matching the session boundary but not inheriting PTQ correctness or performance evidence |

The exact layout passes remove only locally proven storage work; they do not
claim that the remaining producer hybrid graph is equivalent to the compact
session ABI. The canonical Direct candidate additionally enables both explicit
numerical migrations and the caller-ABI specialization above.

Current node counts, performance, and accuracy must be generated from the exact
published artifacts rather than copied from an earlier local probe. The
TinyReceipt native heldout profiler compares FP32-from-FP32,
INT8-from-FP32, exact INT8-from-INT8, and canonical migrated INT8-from-INT8 in a
single strict incremental native-CPU run. Its JSON report content-binds package
graphs and weights, the executable, annotations, and images, then publishes
per-package timing distributions and exact match plus cross-artifact answer and
structured-text agreement. The commands and stable artifact names are in
`examples/tiny_receipt_vqa/README.md`; this design document intentionally embeds
no current numeric result before that generated report exists.

If such a report shows a Direct-versus-PTQ gap, inspect the persisted graph and
split the timing into encoder, first decoder step, and steady decoder work.
Producer Q/DQ boundaries, residual F32 islands, layouts, and cold first-use
materialization can affect the first two even when the incremental decoder cache
is operating correctly; steady-token timing is the relevant cache-path signal.

`--require-row` selects and validates the native incremental row scheduler. It
does not convert arbitrary full-tensor operators into row kernels. The FP32
pre-PTQ cleanup makes the later session graph eligible for that scheduler; the
exact direct-INT8 cleanup alone makes no whole-graph promise. Its unfused
`QBatchMatMul` nodes can nevertheless use the native one-row kernel when their
left row changes and their complete right operand remains clean; any dirty,
broadcast-general, aliased, or otherwise unsupported form fails the row gate.

Incorrect results came from treating graph conversion and a small structural
check as sufficient qualification. The replacement process binds calibration
to the exact optimized graph after any explicitly requested specialization,
uses disjoint representative training records, records route coverage without
making it quantization legality, covers decoder prefixes and immutable embedding
ranges, compares intermediates independently, and gates on task answers.
Historical specialized calibration is complete for `f0` and `f1`; the
runtime-routed artifact and the other six families are not granted that
qualification by the structural changes alone.

The highest-value remaining work is compiler/backend work:

1. **Fuse explicit float islands internally.** Teach backend `CompiledModel`
   preparation to recognize `DQ → float Add/Embedding/LayerNorm → Q` regions
   and generate fused physical kernels or boundary-aware schedules. Preserve
   the explicit v1 graph and selected numerical policy while avoiding unnecessary buffers,
   conversions, and dispatches.
2. **Complete TensorFlow Lite lowering.** Move every supported TFLite operator
   from lossless SourceIR through explicit RuntimeIR legality; fail with precise
   diagnostics for unsupported forms. Do not create a second TFLite-specific
   graph contract.
3. **Improve backend compilation.** Derive capability-qualified partitions,
   physical layouts, memory schedules, and measured tactic caches while
   preparing `CompiledModel` for an exact package revision. These are runtime
   compiled state or rebuildable caches, never additional required package
   files; portable graph semantics and safetensors remain the sole deployment
   inputs.

Backward compatibility is not part of this roadmap. If the current contract
changes, producers and packages are regenerated to the new current contract
and non-current inputs are rejected.

## 9. Best-practice release checklist

- Import once into lossless SourceIR; never optimize by mutating importer-only
  structures.
- Make every dtype, output port, affine reference, public ABI change, and
  specialization binding explicit.
- Fuse high-level semantics before PTQ; restrict post-PTQ cleanup to proven
  byte-domain equivalences.
- Calibrate the exact optimized graph after any requested specialization with
  representative training data; for runtime routing, cover every public
  family. Record graph, data, implementation, route, sample, and observer
  identities.
- Keep scales and zero points in safetensors and publish graph/tensors
  atomically.
- Prefer an explicit mixed-precision island over an inaccurate full-INT8 claim.
- Compare intermediate tensors to an independent executor and test public task
  outputs, including routing and tie/mask semantics.
- Benchmark paired cases at the real sequence length; report distributions,
  package identities, backend, fallback policy, and qualification scope.
- Rebuild provider partitions and memory plans for every changed package
  revision. Treat backend binaries and packed weights as target-derived caches.
- Publish only qualified model families and targets. Absence of evidence is not
  compatibility or performance support.

## 10. Non-goals

- Runtime-load graph rewriting; optimization remains offline.
- Training ownership; PyTorch remains the experimentation/training system.
- Multiple graph schemas or affine encodings.
- Model-specific pass frameworks or private pass-order dictionaries.
- Recalibrating or silently replacing producer affines during static INT8
  import optimization.
- Silent fallback for unknown operators, attributes, dtypes, or stale packages.
- Judging success by node count or model size without task and latency evidence.
