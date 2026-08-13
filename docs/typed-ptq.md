# Typed RuntimeIR PTQ

The PTQ publication path operates on verified `volvox-graph/v1` RuntimeIR.

## Required stage order

1. Import ONNX, TensorFlow Lite, or an existing v1 package and lower it to a
   verified FP32 RuntimeIR.
2. Run semantics-preserving canonicalization and fuse-before-quantize passes,
   then apply every explicit input, route, shape, or semantic-output pruning
   specialization that changes the graph being calibrated. A terminal
   byte-domain ArgMax specialization runs after PTQ, when its monotonic affine
   precondition can be proved.
3. Calibrate that exact optimized and specialized graph. `CalibrationTable`
   records the graph fingerprint, finite F32
   `{min,max,samples,elements}` observations, the aggregate sample count, and a
   deterministic sample digest. A profile from any other graph revision is
   rejected.
4. Call `plan_runtime_ptq`. Review its explicit activation parameters, packed
   weight checksum, saturation count, reconstruction error, bias values, and
   byte-boundary plan.
5. Call `materialize_runtime_ptq`. It verifies that the graph and source
   constant bytes still match the plan, builds a working copy, verifies and
   serializes that copy, then commits GraphIR and the tensor store together.
6. Re-run the safe typed optimizer for dead-code/boundary cleanup, qualify the
   quantized graph on held-out inputs, and publish the final graph plus
   safetensors. Backend preparation must compile that exact published revision;
   no separate deployment-plan artifact is required.

For an existing INT8 v1 package, import it directly into verified RuntimeIR and
run only semantics-proven portable cleanup before qualification and
publication. The optimizer does not invent replacement affines, recover
high-level semantics that Q/DQ lowering discarded, or reinterpret
source-framework quantization metadata.

In abbreviated Python:

```python
graph = import_runtime_package(document, tensors)
default_runtime_pipeline().run(graph)

calibration = CalibrationTable(graph)
for captures in exact_optimized_fp32_captures:
    calibration.observe(captures)

plan = plan_runtime_ptq(graph, tensors, calibration.profile())
materialize_runtime_ptq(graph, tensors, plan)
default_runtime_pipeline().run(graph)

document, tensors = export_runtime_package(graph, tensors)
bounded_domain_proof = prove_dynamic_quantized_runtime_domain(document, tensors)
import_runtime_package(
    document,
    tensors,
    bounded_domain_proof=bounded_domain_proof,
)
# Publish document + tensors transactionally; runtime CompiledModel preparation
# owns provider selection and target-specific compiled state.
```

The default `PTQConfig()` authors symmetric I8 activations (`zero_point=0`).
Activation storage and affine scheme can also be selected independently:

```python
config = PTQConfig(
    activation_dtype="int8",
    activation_scheme="asymmetric",
)
plan = plan_runtime_ptq(
    graph, tensors, calibration.profile(), config=config,
)
```

| Configuration | Scale/range policy | Zero point |
| --- | --- | --- |
| `int8` + `symmetric` | `max(abs(min), abs(max)) / 127` | `0` |
| `int8` + `asymmetric` | observed zero-inclusive span over 255 levels | `[-128,127]` |
| `uint8` + `asymmetric` | observed zero-inclusive span over 255 levels | `[0,255]` |
| `uint8` + `symmetric` | `max(abs(min), abs(max)) / 127` | `128` |

When `activation_scheme` is omitted, I8 defaults to symmetric and U8 defaults
to asymmetric. This preserves the existing U8 authoring behavior while keeping
the overall PTQ default symmetric I8. Weight storage remains symmetric
per-output-axis I8 for every activation policy.

`exact_optimized_fp32_captures` must come from the graph being planned. The
small NumPy reference executor can provide them for its correctness-core
operators; broader models should capture the same named tensors from a
qualified FP32 runtime.

By default, `CalibrationTable(graph)` requests exactly the F32 activation
inputs and outputs demanded by the selected typed PTQ plan. It does not request
every F32 runtime tensor, so additive attention masks and other edges with no
PTQ demand are not mistaken for numeric activation ranges. Passing an explicit
tensor-name list remains available for diagnostics and retains strict
finite-value validation.

Each `CalibrationTable.observe()` call must provide the complete requested set
with one consistent binding for every repeated shape symbol. Concrete extents
must respect each symbol's bounds and divisibility constraint. The table
rejects missing, empty, wrong-dtype, wrong-shape, and non-finite captures, then
accumulates both capture count (`samples`) and scalar-element count (`elements`)
per tensor.

Externally aggregated ranges use the same evidence contract. Every entry must
contain finite ordered `min`/`max`. For a bounded symbolic descriptor it must
also contain an exact positive safe-integer `samples` equal to the profile
sample count and an exact positive safe-integer `elements` inside the total
element envelope implied by the declared bounds. Concrete descriptors derive
those counts exactly; supplied counts must agree. Missing demanded tensors,
stale fingerprints, invalid digests, or unverifiable counts fail closed.

## Current exact support

The author accepts concrete or bounded symbolic, fixed-rank activation
descriptors when it can prove the operator's complete declared shape domain.
Feature/channel axes that determine immutable storage or kernel geometry must
still have fixed extents.

- `Linear` and `MatMul` accept bounded outer activation dimensions with fixed
  input/output features. `Linear` requires its canonical ports, an explicit
  `din_dout` or `dout_din` layout, an immutable concrete rank-two F32 weight,
  and an optional immutable F32 `[d_out]` bias. `MatMul` is the exact static-RHS
  form with empty parameters and immutable F32 `[d_in,d_out]` RHS storage.
- `Conv2D` accepts bounded rank-four NHWC activation/output descriptors with
  fixed channels, exact grouped geometry, an optional vector bias, and an
  immutable concrete rank-four F32 OHWI or HWIO weight. Materialization emits
  canonical OHWI `QConv2D` storage.
- `Embedding` accepts bounded fixed-rank I32 IDs, a bounded output, and an
  immutable concrete rank-two F32 table. IDs must be public inputs or come from
  a canonical I32 `Clip` whose bounds fit the vocabulary, so preflight can
  reject invalid IDs before any output write.
- `Add` accepts two activations or one activation plus one immutable F32
  constant. Symbolic operands must already have the exact output shape;
  concrete portable broadcasts are materialized through byte `Expand`.
  Immutable constants require a concrete output shape so one byte payload can
  be authored.
- Shape-preserving `GELU` (`approximate='none'`) and `SiLU` lower to `QGELU`
  and `QSiLU`.
- `LayerNorm` accepts matching bounded rank-one through rank-eight
  `[...,D]` activation/output descriptors with fixed `D`; `GroupNorm` accepts
  matching bounded rank-four NHWC descriptors with fixed `C` and a group count
  dividing `C`. Both require immutable vector F32 weight and bias.
- `BatchMatMul` accepts bounded rank-two through rank-eight activation
  operands when contraction, descriptor broadcasting, and output geometry are
  exact over the declared domain.
- `CrossSDPA` accepts explicit bounded rank-two or rank-three Q/K/V tensors,
  fixed `D`/head geometry, explicit heads and causal mode, and U32-safe maximum
  tensor sizes. Its optional mask is an I32 keep mask shaped `[K]`, `[B,K]`,
  `[Q,K]`, or `[B,Q,K]`; F32 additive masks must be legalized first.
- Per-tensor I8 or U8 activations, per-output-axis narrow-range I8 weights, and
  I32 accumulator bias are materialized with ahead-of-time multiplier and
  overflow checks. Compatible adjacent selected nodes reuse byte edges and
  affines; quantize/dequantize boundaries remain at true F32 consumers and
  public F32 interfaces.

All activation and weight scales and zero points are rank-one arrays in
`model.safetensors`. Graph JSON contains only the central tensor reference
table. Weight packing and bias conversion use staged F32 arithmetic,
round-to-nearest ties-to-even, narrow-range symmetric I8 weights, positive-F32
multiplier checks, and an ahead-of-time I32 accumulator bound.

`PTQConfig.float_ops` retains every instance of a named supported op type in
F32. `PTQConfig.float_nodes` retains only the exact named graph instances and
rejects unknown names or conflicts with an explicit `selected_nodes` request.
This is an explicit mixed-precision policy, not permission to omit arbitrary
calibration ranges. For example, the TinyReceiptVQA publisher maps only
certified non-finite mask tensors in semantic domain
`additive-attention-mask-{0,-Infinity}/v1` to the supported source nodes that
touch them; every other demanded affine remains fail-closed.

A package with both a non-empty symbolic `dimensions` table and a non-empty
quantization tensor table is accepted back into typed RuntimeIR only with the
immutable canonical proof returned by
`prove_dynamic_quantized_runtime_domain` for the final graph and safetensors
payload. In the registered authoring pipeline, `shape_profile=None` preserves
that bounded symbolic graph; a supplied profile is explicit constant-binding
specialization. Import requires the proof's backend members to equal the
portable domain (`cpu-js`, `wasm`, `webgpu`, and `native-cpu`), recomputes it,
and rejects any disagreement in graph structure, bounds, quantization
references, or tensor bytes. Warm-shape runs and maximum-corner qualification
are useful tests, but they are not substitutes for this all-declared-domain
proof.

## Deliberate gaps

- Dynamic/runtime-RHS `MatMul`, `Gemm` attributes, non-vector bias,
  mutable or symbolic weights/affine parameters, non-NHWC convolution, and
  arbitrary operators outside the families above are not materialized.
- Packed `SDPA(qkv)` lacks independently calibrated Q/K/V affines. It must be
  canonicalized to explicit `CrossSDPA` before calibration or retained in F32.
  Additive F32 attention masks likewise require typed keep-mask legalization,
  or the attention node must be explicitly retained in F32.
- Symbolic `Add` broadcasting beyond exact-shape operands, and an immutable F32
  `Add` constant with a symbolic output, have no portable byte materialization
  contract and are retained in F32 by automatic planning. Explicitly selecting
  such an instance fails with its typed diagnostic.
- Percentile, histogram, KL, MSE-search, and outlier observers are not
  implemented. Calibration consumes exact finite min/max evidence.
- Unsupported explicitly selected topology, missing/non-finite observations,
  stale plans or proofs, invalid multipliers, bias overflow, and unprovable
  accumulator bounds fail before commit. Operators outside the supported
  source set remain FP32.
- Task score, backend-route attestation, and latency thresholds are
  qualification gates outside the rewrite itself; successful graph conversion
  is not evidence that a model preserves FP32 quality or improves a deployment
  baseline.
- Mixed precision is selected by leaving unsupported or accuracy-sensitive
  operators in F32 and quantizing only proven regions. The typed PTQ stage does
  not yet perform sensitivity search or choose the policy automatically.

## Two import paths, and why they do not converge

A package can reach the byte domain two ways, and they are **not**
interchangeable:

1. **FP32 source -> VolvoxAI calibration -> typed PTQ.** VolvoxAI chooses where
   the quantization boundaries go and measures each one.
2. **Pre-quantized source (e.g. an ORT QDQ model) -> optimizer lift.** VolvoxAI
   inherits whatever boundaries the upstream quantizer chose.

Path 2 carries strictly less information, and no optimizer pass can recover the
difference: **where the upstream tool left an activation in float, there is no
affine to lift, and none can be derived.** This is a property of the input, not
a gap in the pass.

Measured on the TinyReceiptVQA BPE1536 release, whose INT8 ONNX is produced by
ORT `quantize_static`:

| op | FP32 ONNX | INT8 ONNX | |
| --- | ---: | ---: | --- |
| `Conv` | 13 | 13 | wrapped in Q/DQ -> lifts to `QConv2D` |
| `MatMul` / `Gemm` | 32 / 8 | 32 / 8 | wrapped -> `QLinear` / `QGemm` |
| `InstanceNormalization` | 13 | **13** | left float |
| `Sigmoid` | 13 | **13** | left float |
| `Mul` / `Add` | 57 / 69 | **57 / 69** | left float |
| `QuantizeLinear` / `DequantizeLinear` | 0 | 120 / 167 | inserted boundaries |

That INT8 ONNX is therefore a *QDQ-annotated FP32 graph*: its int8-ness lives
only around `Conv`/`MatMul`. Everything else still computes in float, so the
encoder retains 13 float GroupNorm and 13 float SiLU sites that the typed PTQ
stage would have quantized natively had it seen the FP32 source (it supports
`QGroupNorm` and `QSiLU` today).

Structural convergence is reachable, but **the lever is upstream** — widen the
producing quantizer's op coverage so it emits the boundaries VolvoxAI needs.
Numerical convergence is not reachable at all: the two calibrators use different
range policies, so identical structure still yields different scales. "Same
graph" is attainable if defined by operator inventory, never if defined by bytes.

## Why a missing affine cannot be synthesized

It is tempting to derive a missing activation range analytically instead of
calibrating. For normalization outputs this looks especially safe, since
`GroupNorm` emits `gamma * g_hat + beta` where `g_hat` has mean 0 and variance 1
*by definition of normalisation* — the statistics are pinned by algebra, not by
data, and `gamma`/`beta` are initializers already in the graph.

This was tried and **it fails**. Probing the FP32 source over 100 real inputs:

| tensor | observed \|max\| | rms | analytic 6-sigma | ratio |
| --- | ---: | ---: | ---: | ---: |
| `group_norm` | 7.94 | 0.359 | 3.27 | 2.4x |
| `group_norm_1` | 13.13 | 0.405 | 2.98 | **4.4x** |
| `group_norm_11` | 4.70 | 0.428 | 2.99 | 1.6x |

The variance prediction was right — observed rms ~0.40 matches `gamma` ~0.4.
The **tail** was wrong: real maxima sit at 20-30 sigma, not 6. Quantization
range is set by the tail, and the tail is not a function of the weights. Any
data-free bound is therefore a guess wearing a proof's clothing.

The practical consequence is not "give up" but "calibrate narrowly": only the
boundaries the upstream tool omitted need observation, and every inherited
affine can stay untouched. That keeps the re-validation surface small.
