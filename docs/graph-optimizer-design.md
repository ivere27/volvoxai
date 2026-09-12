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
fresh-path payload publication + graph availability sentinel
                                         │
                                         ▼
runtime CompiledModel backend preparation for the exact package revision
```

`tools/exporter/` is a general-purpose, model-neutral library and command layer.
It owns SourceIR/RuntimeIR, import and lowering, reusable graph passes,
calibration/PTQ primitives, verification, differential execution, and
graph-sentinel publication mechanisms. The protobuf optimizer registry selects
the ordered typed pipeline; application code supplies explicit feature requests
and pass inputs, not another pass list. The exporter does not own a model's package
manifest, graph pairing, family vocabulary or routing, task preprocessing, or
release policy.

TinyReceipt-specific encoder/decoder composition, package manifests, routing,
and qualification live under `examples/tiny_receipt_vqa/`. Its current
explicit-KV package-v2 path imports the producer-authored FP32 or static INT8
ONNX variant directly.

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
6. Graph JSON and Safetensors publish only to fresh paths: complete payloads
   first and the graph availability sentinel last. Graph-first readers either
   see no package or one complete immutable package. Existing paths and
   in-place replacement fail closed; this is not simultaneous multi-file
   visibility. Numeric affine parameters live in Safetensors, never inline in
   graph JSON.
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
  "format": "volvox-graph/v1",
  "dimensions": {}
}
```

The current contract is strict:

- each input has exactly `shape` and `dtype`; every symbolic shape axis names a
  finite constraint in `dimensions`;
- each node output port is one `{tensor, shape, dtype}` assertion checked
  against canonical whole-domain shape inference; split output maps and an
  implicit F32 default do not exist;
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

`proto/volvoxai.proto` owns the public `DataType` enum.
`proto/operator_vocabulary.proto` separately owns the internal logical
`OperatorKind` vocabulary. The kernel, parameter, and optimizer registries
import that internal enum, and code generation keeps Python, TypeScript, and
native projections aligned; no JSON capability list or handwritten enum copy
is authoritative. Neither schema contains pass order or backend kernel
inventory. Graphs keep the current CamelCase `opType` spelling, with the
generated string-to-enum projection used by the strict RuntimeIR loader.

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
backend `wasm` still publish a graph legal on `wasm`, `webgpu`, and
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
- verification and differential-comparison expectations.

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
```

Strict verification is unconditional. Omitting `--out` performs a dry run;
publication requires both `--out` and `--out-weights`, and both paths must be
unused. Complete Safetensors bytes publish before the graph sentinel. Official
loaders open the graph first, so they see either no package or the immutable
fresh package; arbitrary weights-first readers are outside this protocol.
`--in-place` fails closed because replacing two existing files cannot provide
that guarantee. Publish to a new path or directory and switch a higher-level
reference after success. The optional pass report is diagnostic output, not a
deployment input. `--backend-profile`
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

### 5.1 Island shape decides what static-QDQ fusion can reach

`static-qdq-compute-fusion` matches one shape only:

~~~text
byte -> DequantizeLinear -> <exactly one float op> -> QuantizeLinear -> byte
~~~

Both ends already carry measured affines, so the rewrite invents nothing. The
consequence is easy to misread as a coverage gap in the op list when it is
actually a **shape** constraint. In the TinyReceipt encoder the pass fused 7
GELU islands and refused everything else, because the remaining float region is

~~~text
byte -> DQ -> GroupNorm -> v25 -+-> Sigmoid -+
                                 \-----------+-> Mul -> Q -> byte
~~~

Two float computations sit between the DQ and the Q. `GroupNorm` is not closed
(its output feeds two float consumers and no `Q`); the `Sigmoid`/`Mul` pair is
not closed (its input arrives as float, not through a `DQ`). Neither is
individually a legal island, so both are correctly refused. That the pair is
algebraically `SiLU` does not help: folding it first still leaves two ops.

Splitting such a region requires a **new** quantization boundary on the
intermediate, which requires an affine, which requires calibration. See
[typed-ptq.md](typed-ptq.md) for why that affine cannot be synthesized from
weights. This is the general rule: *fusion coverage is limited by where
boundaries exist, and creating boundaries is authoring, not optimization.*

### 5.2 A fusion is only worth taking if the byte kernel is competitive

Byte-domain replacement is usually assumed to win because it quarters memory
traffic. That assumption must be measured, because the float kernel it replaces
may be the better-optimized one. Measured on the encoder's largest island
(1x160x336x48, 16 groups, 2.58M elements):

| path | ms | |
| --- | ---: | --- |
| `dequantize` | 2.33 | |
| `groupnorm_f32` | 8.21 | streaming + vectorized |
| `silu_f32` | 5.07 | |
| `quantize` | 1.11 | |
| **float island total** | **16.72** | |
| `qgroupnorm_i8u8` (original) | **19.13** | scalar, no SIMD |
| `qsilu_i8u8` | **1.43** | 256-entry LUT |
| **byte island total (original)** | **20.56** | **slower than float** |

`QSiLU` behaved exactly as predicted — a LUT beats `expf` per element by 3.5x.
`QGroupNorm` did not: it read a quarter of the bytes and still lost, because it
was a purely scalar validating kernel while `groupnorm_f32` had already been
vectorized. Fusing on those numbers would have *replaced an optimized float
kernel with an unoptimized quantized one*, and end-to-end measurement confirmed
no gain.

The fix keeps the contract intact. `QGroupNorm`'s statistics are order-dependent
float reductions and were left byte-for-byte alone; only the final transform —
which is elementwise, so traversal order cannot change a result — was moved from
`groups` strided passes to one contiguous sweep, with each group's `(mean,
1/sigma)` published per channel.

| | ms | |
| --- | ---: | --- |
| `qgroupnorm_i8u8` before | 19.13 | |
| `qgroupnorm_i8u8` after | **11.56** | **1.65x**, bit-identical on 54 differential cases |
| byte island total after | **12.99** | now **1.29x** faster than the float island |

End-to-end, three interleaved paired rounds favoured the fused package every
time (-25.5, -15.0, -57.0 ms cold generation), consistent with the 18.7 ms the
kernel arithmetic predicts across five sites.

**The rule this establishes:** before enabling a byte-domain fusion, time the
replacement against the float kernels it displaces on the real shape. A fusion
that is structurally valid and accuracy-neutral can still be a regression, and
the deciding factor is how much optimization work each kernel has already
received — not the dtype.

## 6. Best-practice release checklist

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
- Compare intermediate tensors to an independent executor and validate public task
  outputs, including routing and tie/mask semantics.
- Benchmark paired cases at the real sequence length; report distributions,
  package identities, backend, fallback policy, and qualification scope.
- Rebuild provider partitions and memory plans for every changed package
  revision. Treat backend binaries and packed weights as target-derived caches.
- Publish only qualified model families and targets. Absence of evidence is not
  compatibility or performance support.

## 7. Non-goals

- Runtime-load graph rewriting; optimization remains offline.
- Training ownership; PyTorch remains the experimentation/training system.
- Multiple graph schemas or affine encodings.
- Model-specific pass frameworks or private pass-order dictionaries.
- Recalibrating or silently replacing producer affines during static INT8
  import optimization.
- Silent fallback for unknown operators, attributes, dtypes, or stale packages.
- Judging success by node count or model size without task and latency evidence.
