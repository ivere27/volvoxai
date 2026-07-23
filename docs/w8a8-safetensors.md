# Affine quantization in safetensors

VolvoxAI has one graph format, `volvox-graph/v1`, and one packaged affine
quantization representation, `volvox-affine-safetensors/v1`. Graph JSON owns
topology and immutable tensor references. Every numeric scale and zero point is
a tensor in safetensors.

The words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, and **MAY** describe
requirements for conforming producers and loaders.

## Graph contract

A quantized graph has one central reference table:

```json
{
  "format": "volvox-graph/v1",
  "quantization": {
    "format": "volvox-affine-safetensors/v1",
    "tensors": {
      "hidden_q": {
        "scheme": "per_tensor",
        "scale_tensor": "__quant__.hidden.scale",
        "zero_point_tensor": "__quant__.hidden.zero_point"
      },
      "projection.weight": {
        "scheme": "per_axis",
        "axis": 0,
        "scale_tensor": "__quant__.projection.weight.scale",
        "zero_point_tensor": "__quant__.projection.weight.zero_point"
      }
    }
  }
}
```

The `quantization` object MUST contain exactly `format` and `tensors`. Its
`tensors` value MUST be a non-empty object. Each key is the exact name of a
declared I8 or U8 graph tensor. A descriptor has exactly one of these forms:

```json
{"scheme":"per_tensor","scale_tensor":"S","zero_point_tensor":"Z"}
{"scheme":"per_axis","axis":0,"scale_tensor":"S","zero_point_tensor":"Z"}
```

`axis` MAY name any valid axis, including a negative axis spelling, but it is
normalized against the target rank during validation. Operator contracts can
be stricter; for example, canonical `QLinear` weights are I8 `[d_out,d_in]`
with per-axis quantization on axis 0.

No numeric quantization value is legal in graph JSON. Parameter tensor names
are references, not a second descriptor namespace.

This prohibition is specific to affine quantization. A numeric operator
parameter with different semantics—for example the attention score multiplier
in `QSDPA.params.scale`—remains part of graph topology and is not an affine
activation or weight scale.

## Safetensors contract

For a target tensor `Q`:

- `Q` MUST have execution dtype I8 or U8.
- `scale_tensor` MUST resolve to an immutable F32 safetensors tensor of rank 1.
- `zero_point_tensor` MUST resolve to an immutable rank-1 safetensors tensor
  whose dtype exactly matches `Q`.
- The scale and zero-point names MUST be distinct.
- For `per_tensor`, both parameter tensors MUST have shape `[1]`.
- For `per_axis`, both parameter tensors MUST have shape `[Q.shape[axis]]`.
- Every scale MUST be finite and strictly greater than zero.
- A quantization parameter MUST NOT itself have an affine descriptor.

Zero points are explicit even for symmetric I8. A symmetric descriptor stores
an I8 zero vector; loaders do not infer an omitted zero point.

Safetensors `__metadata__` MAY contain unrelated application strings, but it is
not quantization authority. The retired `weights_quantization` and
`weights_quantization_storage` metadata keys are invalid.

For sharded packages, references resolve within the weight set assigned to one
graph. Tensor names MUST be unique in that set. A base tensor and its scale and
zero-point tensors MAY occupy different shards when the package manifest binds
all shards atomically; streaming implementations SHOULD keep them together.

## Numeric meaning

Let `q` be a stored element, `s` its selected F32 scale, and `z` its selected
typed zero point. Dequantization is:

```text
real = s * (integer(q) - integer(z))
```

For `per_tensor`, index `s[0]` and `z[0]`. For `per_axis`, select both values by
the target coordinate on `axis`.

This storage contract does not choose a calibration method. VolvoxAI PTQ uses
explicit F32 arithmetic, ties-to-even rounding, saturation to the destination
byte range, per-output-row narrow-range I8 weights where required, and an I32
accumulator-bound proof before publishing a fused integer operator. Those are
optimizer/operator requirements, not alternate storage encodings.

## Q/DQ identity

`QuantizeLinear` and `DequantizeLinear` make boundaries explicit. Their operand
names MUST be identical to the central descriptor references:

```json
{
  "opType": "QuantizeLinear",
  "inputs": {
    "input": "hidden",
    "scale": "__quant__.hidden.scale",
    "zero_point": "__quant__.hidden.zero_point"
  },
  "outputs": {"out": "hidden_q"},
  "outputs_shape": {"out": [1, 320]},
  "outputs_dtype": {"out": "int8"},
  "params": {}
}
```

A `QuantizeLinear` output descriptor must reference its `scale` and
`zero_point` operands. A `DequantizeLinear` input descriptor must do the same.
Loaders MUST reject a mismatch rather than silently choosing either source.

Fused quantized operators consume the same central descriptors through their
named byte tensors. Backend-private packed weights or precomputed multipliers
are derived caches and MUST NOT change the logical graph or become an alternate
source of affine values.

## Required validation

Before execution or publication, a loader/exporter MUST validate:

1. The graph format is exactly `volvox-graph/v1`.
2. The central table and every descriptor have exactly the supported fields.
3. Every target, scale, and zero-point reference resolves uniquely.
4. Target and parameter dtypes, ranks, axis, and counts match.
5. All scales are finite and positive.
6. Quantization parameters are immutable and are not themselves quantized.
7. Q/DQ operands exactly match central references.
8. Every fused operator satisfies its stricter dtype, shape, layout,
   quantization, multiplier, and accumulator-range contract.

Any failure aborts loading. A conforming loader does not coerce dtypes, invent
parameters, guess a scheme, select an arbitrary duplicate, or repair a stale
reference.

## Rejected non-v1 representations

There is no compatibility mode. `volvox-graph/v1` readers MUST reject:

- graph-level `weights_quantization` or `weights_quantization_storage`;
- input-level inline `quantization` values;
- node-level `outputs_quantization` values;
- safetensors quantization descriptor metadata;
- implicit `<weight>_scale` naming as authority;
- numeric scale or zero-point arrays embedded in JSON.

An artifact containing any of these fields is not a `volvox-graph/v1` package
and must be regenerated from its source model. Runtime readers and the general
optimizer do not migrate or reinterpret it.

## Runtime repacking

A backend MAY repack immutable data for VNNI, ARM dot-product, WASM SIMD,
WebGPU, or another target. Repacking MUST preserve the association between each
logical slice and its central scale/zero-point references. Backend caches are
invalidated when any referenced tensor identity changes.

Safetensors itself defines tensor storage, not a universal quantization schema.
`volvox-affine-safetensors/v1` is VolvoxAI's strict association contract.
