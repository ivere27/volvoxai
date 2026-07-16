# W8A8 Safetensors Companion Scales

This document defines the VolvoxAI F32 companion-scale interoperability
profile for symmetric I8 model weights stored in safetensors. It is an
application-level VolvoxAI contract, not an official safetensors quantization
schema.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document describe requirements for conforming producers and
loaders.

## Format identifier

The exact v1 identifier is:

```text
volvoxai-f32-companion-scales-v1
```

The package `config.json` MUST select this profile with an object containing
exactly the `format` field:

```json
{
  "weights_quantization_storage": {
    "format": "volvoxai-f32-companion-scales-v1"
  }
}
```

Under this marker, `config.json` MUST NOT contain a `weights_quantization`
descriptor table. Safetensors metadata is the sole authority for weight
descriptors. A loader MUST reject an unknown marker, malformed marker object,
config-level descriptor table, or malformed v1 metadata instead of falling
back to inline scales or an older representation.

Any incompatible extension requires a new format identifier.
Duplicate object keys at any nesting depth of the raw config JSON are invalid;
a loader MUST NOT resolve an ambiguous config by choosing the first or last
value.

An embedding API MAY accept a trusted, already-decoded config object instead
of raw JSON. In that case the object provider, not the loader, is responsible
for enforcing the raw duplicate-key rule before decoding. This exception does
not apply when the loader receives config-file bytes or text.

## Safetensors metadata

Each shard that stores at least one companion-quantized weight MUST contain
both of these string-valued entries in its safetensors `__metadata__` map:

```json
{
  "__metadata__": {
    "weights_quantization_storage": "volvoxai-f32-companion-scales-v1",
    "weights_quantization": "{\"decoder.proj.weight\":{\"scheme\":\"per_axis\",\"axis\":0},\"decoder.position\":{\"scheme\":\"per_tensor\"}}"
  }
}
```

The metadata marker is the direct format string, not the config object form.
Safetensors metadata values are strings, so `weights_quantization` is a JSON
string whose decoded value is an object mapping base tensor names to
descriptors.

Each shard's mapping MUST describe only base tensors physically stored in that
shard. An unquantized-only shard MAY omit both companion metadata entries. It
MAY instead carry the marker and an empty mapping (`"{}"`). Supplying only one
of the two entries is invalid. Unrelated string-valued safetensors metadata MAY
coexist with these entries. JSON and tensor names are case-sensitive; a
different spelling or case is an unrelated name, not an alias for a profile
field or tensor.

Producers MUST emit the descriptor map as compact JSON and SHOULD sort base
tensor names lexicographically for reproducible output. Object order and JSON
whitespace have no semantic meaning; loaders MUST parse the JSON rather than
compare its serialized bytes. Duplicate object keys at any nesting depth of
the encoded descriptor JSON are invalid, including escaped spellings that
decode to the same key.

## Tensor naming and storage

For every descriptor key `W`, the same shard MUST contain exactly one base
tensor and one inferred companion:

```text
W          I8
W_scale    F32
```

The companion name is formed by appending the literal ASCII suffix `_scale` to
the complete base tensor name. For example:

```text
decoder.proj.weight        I8
decoder.proj.weight_scale  F32
```

V1 has no companion-name override. A producer MUST reject a collision between
an inferred companion name and any other tensor. A companion MUST NOT itself
be declared as a quantized base tensor.

Only an authoritative descriptor establishes this relationship. A loader MUST
NOT infer quantization merely because an otherwise unclaimed tensor name ends
in `_scale`.

## Descriptor forms

### Per-axis

The only valid per-axis descriptor is:

```json
{"scheme":"per_axis","axis":0}
```

It MUST contain exactly those two fields. `axis` MUST be the integer `0`;
negative aliases and other axes are not valid in v1.

For a base tensor with shape `[O, D1, D2, ...]`, its rank MUST be at least one,
`O` MUST be positive, and its companion MUST have dtype `F32` and shape `[O]`.
Axis 0 is normally the output-row or output-channel dimension.

### Per-tensor

The only valid per-tensor descriptor is:

```json
{"scheme":"per_tensor"}
```

It MUST contain exactly that field. Its companion MUST have dtype `F32`, rank
one, and shape `[1]`.

Descriptors MUST NOT contain inline scales, zero points, offsets, counts,
companion references, or unknown fields. In particular, the following fields
are not part of v1:

```text
scale, scales, zero_point, zero_points, companion,
scale_tensor, scales_offset, scales_count
```

## Numeric interpretation

Let `Q` be the stored I8 tensor and `S` its F32 companion. For per-tensor
quantization:

```text
Wreal[i0, i1, ...] = float(Q[i0, i1, ...]) * S[0]
```

For per-axis quantization:

```text
Wreal[o, i1, ...] = float(Q[o, i1, ...]) * S[o]
```

Equivalently, the mapping is `Wreal = S * (Q - 0)`. The weight zero point is
always exactly zero and is not stored.

Every scale MUST be a finite F32 value strictly greater than zero. NaN,
positive or negative infinity, positive or negative zero, and negative values
are invalid. All I8 values from -128 through 127 are valid.

This profile defines decoding, not how a producer chooses scales or performs
rounding, clipping, calibration, or accuracy evaluation. A runtime MAY
dequantize to floating point or preserve the same mapping in a fused integer
kernel.

## Validation requirements

Before execution, a conforming loader MUST validate all of the following:

1. The raw config JSON has no duplicate object keys, the config marker object
   contains exactly the supported `format` value, and config has no
   `weights_quantization` table.
2. Each shard supplies either both companion metadata entries or neither, and
   every supplied shard marker exactly matches the config marker.
3. The raw safetensors header contains no duplicate object key at any nesting
   depth, including tensor names, `__metadata__` fields, and tensor-record
   fields; when present, `__metadata__` is an object whose values are all
   strings as required by safetensors.
4. Each metadata descriptor map is valid JSON with a JSON object at its root
   and contains no duplicate object key at any nesting depth.
5. Tensor names are globally unique across the loaded shard set.
6. Descriptor base names are globally unique across all shard-local maps.
7. Each descriptor has exactly one of the two supported forms.
8. Every declared base occurs exactly once, is local to its declaring shard,
   and has safetensors dtype `I8`.
9. Every inferred companion occurs exactly once in the same shard and has
   safetensors dtype `F32`, rank one, and the required shape.
10. Every companion value is finite and strictly positive.
11. No companion is also a declared base tensor.
12. Claimed companions remain internal storage records and are not exposed as
    graph inputs, outputs, mutable patch targets, or ordinary graph weights.
13. The union of shard-local descriptor maps contains at least one weight.

Any failure MUST abort loading. A loader MUST NOT choose the first or last of
duplicate tensors, coerce another scale dtype to F32, normalize another axis to
axis 0, infer missing metadata, or try a legacy representation.

Only tensors named in the union of the authoritative shard-local maps are
governed by this profile. Other I8 and F32 tensors retain their ordinary model
meaning unless a separate graph contract describes them.

## Sharded checkpoints

A sharded checkpoint is the disjoint union of its shard-local declarations.
The base tensor, companion tensor, and descriptor MUST move together when a
checkpoint is repartitioned. Across all loaded shards, every base and companion
name MUST remain unique.

A shard index SHOULD list companion tensors like any other safetensors tensor.
The syntax of a package manifest or shard index is outside this profile.

Keeping each declaration local allows streaming and model-parallel loaders to
validate a shard without consulting another shard's descriptor table.

## Loader outline

```text
MARKER = "volvoxai-f32-companion-scales-v1"

require config.weights_quantization_storage == { format: MARKER }
require config has no weights_quantization

descriptors = empty map
companions = empty set
seenTensorNames = empty set

for each shard:
    header = read_and_validate_safetensors_header_rejecting_duplicate_keys(shard)
    metadata = header.__metadata__ or {}

    for each tensor name in header:
        require name not in seenTensorNames
        add name to seenTensorNames

    hasMarker = metadata contains weights_quantization_storage
    hasMap = metadata contains weights_quantization
    require hasMarker == hasMap
    if not hasMarker:
        continue

    require metadata.weights_quantization_storage == MARKER
    local = parse_json_rejecting_duplicate_keys(
        metadata.weights_quantization
    )
    require local is an object

    for each (baseName, descriptor) in local:
        require baseName not in descriptors
        require header contains baseName
        require header[baseName].dtype == I8

        scaleName = baseName + "_scale"
        require header contains scaleName
        require header[scaleName].dtype == F32
        require header[scaleName].rank == 1

        if descriptor is exactly { scheme: "per_axis", axis: 0 }:
            require header[baseName].rank >= 1
            require header[baseName].shape[0] > 0
            require header[scaleName].shape == [header[baseName].shape[0]]
        else if descriptor is exactly { scheme: "per_tensor" }:
            require header[scaleName].shape == [1]
        else:
            fail

        descriptors[baseName] = descriptor
        add scaleName to companions

for each claimed companion:
    require every decoded F32 value is finite and greater than zero

require descriptors is not empty
require no claimed companion is also a descriptor base
require graph does not expose any claimed companion
```

Implementations SHOULD bound descriptor counts by the number of tensor records
and validate shape and offset arithmetic for overflow before allocation.

## Minimal independent reader

No VolvoxAI API is required to decode these weights. This illustrative Python
reader uses only the standard `safetensors` package and NumPy; a production
loader must still perform every validation above:

```python
import json
import numpy as np
from safetensors import safe_open

MARKER = "volvoxai-f32-companion-scales-v1"

with safe_open("model.safetensors", framework="numpy") as file:
    metadata = file.metadata() or {}
    if metadata.get("weights_quantization_storage") != MARKER:
        raise ValueError("unsupported weight quantization storage")
    descriptors = json.loads(metadata["weights_quantization"])

    name = "decoder.proj.weight"
    quantized = file.get_tensor(name)
    scales = file.get_tensor(f"{name}_scale")
    descriptor = descriptors[name]

    if descriptor == {"scheme": "per_tensor"}:
        dequantized = quantized.astype(np.float32) * scales[0]
    elif descriptor == {"scheme": "per_axis", "axis": 0}:
        broadcast = (scales.size,) + (1,) * (quantized.ndim - 1)
        dequantized = quantized.astype(np.float32) * scales.reshape(broadcast)
    else:
        raise ValueError("unsupported companion-scale descriptor")
```

An engine may use the I8 tensor and F32 scales directly in a dot-product
kernel instead of materializing `dequantized`. The separate model config is
still required to define graph topology and operator semantics.

## Runtime repacking

A loader MAY repack weights for native VNNI, ARM SDOT, WASM SIMD, Relaxed SIMD,
or WebGPU kernels. Repacking MUST preserve the association between every
logical axis-0 slice and its scale. If output rows are permuted, their scales
MUST undergo the same permutation.

A layout whose quantization dimension is not logical axis 0 is not directly
representable by v1 and requires conversion or a future format identifier.

## Scope

This profile defines only symmetric I8 weight storage and binary F32 companion
scales. It does not define:

- model topology or operator semantics;
- activation quantization or dynamic activation scales;
- bias storage;
- calibration, rounding, clipping, or accuracy policy;
- asymmetric weights or nonzero zero points;
- per-group quantization or axes other than zero;
- packed INT4/INT2 or kernel-specific packed layouts;
- package manifests or shard-index syntax.

The `W_scale` spelling is a VolvoxAI convention. Safetensors itself does not
define a universal quantization naming or scale-association schema.
