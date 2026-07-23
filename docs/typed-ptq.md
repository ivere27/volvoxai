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
3. Calibrate that exact optimized and specialized graph. `CalibrationTable` records the graph
   fingerprint, finite F32 observations, sample counts, and a deterministic
   sample digest. A profile from any other graph revision is rejected.
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

By default, `CalibrationTable(graph)` requests exactly the F32 inputs and
outputs consumed by the dense PTQ plan. It does not request every F32 runtime
tensor, so additive attention masks and other edges with no PTQ demand are not
mistaken for numeric activation ranges. Passing an explicit tensor-name list
remains available for diagnostics and retains strict finite-value validation.

## Current exact support

- `Linear` with exact `input`/`weight` ports, optional immutable F32 `[d_out]`
  bias, and explicit `weight_layout` equal to `IN_OUT` or `OUT_IN`.
- `MatMul` with exact `a`/`b` ports, empty parameters, and immutable rank-two
  F32 RHS storage `[d_in,d_out]`.
- Concrete F32 `[...,d_in] -> [...,d_out]` activation geometry.
- Per-tensor I8 or U8 activations, per-output-axis I8 weights normalized to
  `[d_out,d_in]`, and mandatory I32 bias (zero is synthesized when absent).
- Adjacent selected dense nodes share their byte edge. Quantize/dequantize
  boundaries remain only at true FP32 consumers or public FP32 outputs.

All activation and weight scales and zero points are rank-one arrays in
`model.safetensors`. Graph JSON contains only the central tensor reference
table. Weight packing and bias conversion use staged F32 arithmetic,
round-to-nearest ties-to-even, narrow-range symmetric I8 weights, positive-F32
multiplier checks, and an ahead-of-time I32 accumulator bound.

## Deliberate gaps

- `Conv2D`, dynamic-RHS/batched MatMul, Gemm attributes, non-vector bias,
  dynamic shapes, mixed-precision source constants, and percentile/histogram
  observers are not yet materialized by this stage.
- Unsupported selected dense topology, missing/non-finite observations, stale
  plans, invalid multipliers, bias overflow, and unprovable accumulator bounds
  fail before commit. Unsupported non-selected operators remain FP32.
- Task score, backend-route attestation, and latency thresholds are
  qualification gates outside the rewrite itself; successful graph conversion
  is not evidence that a model preserves FP32 quality or improves a deployment
  baseline.
- Mixed precision is selected by leaving unsupported or accuracy-sensitive
  operators in F32 and quantizing only proven regions. The typed PTQ stage does
  not yet perform sensitivity search or choose the policy automatically.
