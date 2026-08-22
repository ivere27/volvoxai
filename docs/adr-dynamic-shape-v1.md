# ADR: Dynamic-first `volvox-graph/v1`

- Status: accepted for implementation
- Date: 2026-08-01
- Scope: graph schema, inference lifecycle, provider contracts, and portable release

## Context

The original `volvox-graph/v1` stores one concrete shape and byte count for
every tensor. Those values are embedded in snapshots, provider compilation,
allocation, input validation, and result validation. Retrofitting symbols into
those mutable concrete objects would make model identity request-dependent and
would allow contexts to race over shapes and buffers.

The dynamic-shape redesign is intentionally breaking. Existing packages and
integrations will be rebuilt instead of maintaining two graph readers or an
adapter that can silently confuse old and new v1 documents.

## Considered alternatives

Three shape systems were available. Each one satisfies half of what
whole-domain compile-time proof requires, and fails the other half.

| System | Named symbols | Finite bounds | Why it was rejected |
| --- | :---: | :---: | --- |
| Fixed concrete shapes | no | no | One package per shape. A variable sequence length or batch size forces either republication or padding to a worst case that is paid on every request. |
| Unbounded named symbols | yes | no | A symbol carrying no maximum admits an infinite domain. No provider can attest tensor, scratch, address-space, buffer-binding, or dispatch maxima over it, so specialization is necessarily deferred to dispatch. That is the runtime discovery this ADR exists to remove. |
| Per-tensor bounded profiles | no | yes | Bounds are declared per tensor, so one axis shared by two tensors becomes two independent ranges. The contract cannot state that the two extents are the same value; a mismatch is detected by whichever operator first contracts them, during execution, instead of being rejected at bind. |

`volvox-graph/v1` takes both halves. A dimension symbol is **named**, so
`["B", "S"]` and `["B", "S", 768]` denote one extent by construction and their
equality is a schema fact rather than a runtime coincidence. A dimension symbol
is also **bounded**, so `{ min, max, multiple_of }` closes the domain and makes
conservative maxima computable.

This pairing is not additive convenience; it is what makes the proof decidable
at all. Bounds without symbols cannot express the equality and divisibility
relations that a proof must discharge. Symbols without bounds leave every
maximum unbounded, so there is nothing to attest. The bounded-domain
qualification described below assumes both properties hold simultaneously.

## Decision

### One bounded, fixed-rank shape system

The sole graph format remains `volvox-graph/v1`. Bounded dynamic shape is part
of that format rather than a separately versioned feature:

```json
{
  "format": "volvox-graph/v1",
  "dimensions": {
    "B": { "min": 1, "max": 8 },
    "S": { "min": 1, "max": 2048, "multiple_of": 1 }
  },
  "inputs": {
    "ids": { "dtype": "int32", "shape": ["B", "S"] }
  },
  "nodes": [
    {
      "id": "embedding",
      "opType": "Embedding",
      "inputs": { "input": "ids", "weight": "token.weight" },
      "outputs": {
        "out": {
          "tensor": "hidden",
          "dtype": "float32",
          "shape": ["B", "S", 768]
        }
      },
      "params": {}
    }
  ],
  "outputs": ["hidden"]
}
```

The normative shape rules are:

- `dimensions` is required and may be empty.
- A symbol is case-sensitive and matches `^[A-Za-z][A-Za-z0-9_]{0,63}$`.
- `min`, `max`, and optional `multiple_of` are positive safe integers.
  `multiple_of` defaults to one. The constrained interval must contain at least
  one legal value.
- A tensor's rank is fixed. Each axis is a positive safe integer constant or a
  string naming a declared symbol.
- Repeating a symbol means exact equality, including across public inputs.
- Node IDs are non-empty unique strings.
- The executable JSON schema is closed: graph, input, node, and output
  descriptor objects reject unknown fields. Every node carries an explicit
  `params` object, including `{}` when the operator has no parameters.
- Each node output port has one required `{tensor, dtype, shape}` descriptor.
  The legacy `outputs_shape` and `outputs_dtype` maps do not exist.
- A node output shape is an assertion over canonical operator inference. An
  output-only symbol may first bind to an inferred axis; subsequent uses must
  match it. An unbound symbol cannot constrain an ordinary node input.
- Derived formulas are operator semantics, not a user-authored JSON expression
  language.
- Weights and trainable parameter shapes remain constant.

The closed root schema rejects a `shape_system` field. There is no static
compatibility mode selected by omitting or changing a secondary discriminator.

Object insertion order has no meaning. Dimension names, public input names,
and descriptor-map keys are ordered by unsigned UTF-8 bytes for hashing and
canonical processing. Node and public-output arrays retain declared order. The
canonical concrete signature is an unambiguous UTF-8 sequence beginning with
`v1|`; for each public input in canonical name order it appends the input
name's UTF-8 byte length and bytes, rank, and comma-separated decimal axes:

```text
v1|3:ids|2:2,128
```

Implementations may hash this sequence for caches and reports, but hash
collisions must never establish shape equality; the full canonical signature
is the correctness key.

All logical arithmetic is checked before conversion to backend metadata.
Declared and finalized dimensions, element counts, and byte counts are exact
integers in `[1, 2^53 - 1]`. Shape-formula intermediates use checked signed
safe-integer arithmetic. Compilation then proves target-specific `size_t`,
i32/u32, WASM address/page, and physical device buffer/binding/dispatch limits
over the entire declared domain.

### Logical models and concrete contexts

`Model` owns immutable logical topology, shape constraints,
fixed weights, and revisions. It never acquires a current request shape.

`CompiledModel` owns the selected provider, bounded-domain proof, immutable
topology preparation, invariant packed weights, tactic candidates, and
shape-independent generated code.

Each `ExecutionContext` exclusively owns its current `ShapeBinding`, concrete
`BoundExecutionGraph`, `ResolvedShapePlan` cache, activation and scratch
capacities, resource generation, request data, outputs, adapter selection, and
decode/KV state. Two contexts may therefore execute different shapes
concurrently. `ExecutionResult` owns exact concrete output descriptors and
stable logical output storage independently of context reuse or closure.

Ordinary execution is atomic through binding and preflight:

1. Normalize the complete set of shaped inputs without writing backend state.
2. Validate names, dtype, rank, dimensions, bounds, equality, and exact bytes.
3. Bind symbols and infer every intermediate and output shape.
4. Preflight operator, quantization, arithmetic, memory, and device constraints.
5. Prepare candidate plan metadata and capacity growth.
6. Commit the complete binding and resource generation atomically.
7. Only then copy/upload inputs, dispatch, and snapshot logical outputs.

A failure before commit leaves the prior successful binding, capacities, and
decode state usable. No input write, GPU submission, or visible plan-cache
mutation is allowed before graph-wide validation succeeds.

### Mandatory shaped public inputs

Every ordinary JavaScript input, including a constant-only model input, uses:

```ts
interface RuntimeTensorView {
  readonly data: RuntimeTypedArray | DeviceTensorReference;
  readonly shape: readonly number[];
}

type ExecutionInputs = Readonly<Record<string, RuntimeTensorView>>;
```

The raw typed-array shorthand is removed. Shape is never inferred from byte
length. Native and protobuf operations likewise carry one atomic batch of
named tensors with dtype, rank, concrete shape, data location, and byte count;
the shape-less per-input native setter is removed. Decode operations may define
explicit partial-update semantics, but ordinary execution requires every
public input.

Amendment: `DeviceTensorReference` is an opaque, live VolvoxAI-issued device
result, not a caller-provided raw buffer. The built-in WebGPU provider accepts
it only for ordinary execution on the same `GPUDevice` after exact dtype,
shape, and logical-byte validation. Core retains the source result until the
consumer's submitted work reaches a queue fence. Other providers and decode
seed/step reject it explicitly.

`VOLVOXAI_BACKEND_PROVIDER_VERSION`, `VX_BACKEND_ABI_VERSION`, and
`VX_NATIVE_API_VERSION` remain 1. The redesigned unreleased contracts replace
their earlier drafts in place; no compatibility shim is provided.

The provider shape-domain capability declares the proof and resource protocols
directly. `volvox-graph/v1` defines the dynamic-shape semantics.

### Provider proof and error policy

Warm profiles are optimization hints and do not restrict legality. During
compilation a provider must conservatively prove a deterministic route for
every legal binding, including a generic in-provider tactic wherever a faster
tactic has narrower divisibility constraints. Provider policy may try another
allowed provider while compilation is selecting one. After a `CompiledModel`
is returned, execution never retries on another provider.

Failures are classified consistently:

| Condition | Status |
| --- | --- |
| Legacy/malformed schema, impossible constraints, operator-shape contradiction, or model-domain arithmetic overflow | `INVALID_GRAPH` at model load |
| Missing input or caller dtype/rank/shape/bounds/equality/byte mismatch | `INVALID_ARGUMENT` before backend mutation |
| Malformed or wrong-sized public/provider descriptor | `INVALID_ARGUMENT` before callbacks execute |
| Unsupported provider ABI tag or invalid provider attestation descriptor | `ABI_UNSUPPORTED` |
| No selected/allowed provider can prove the complete bounded domain | `BACKEND_UNSUPPORTED` at compile |
| A valid concrete specialization cannot allocate within its admitted resource budget | `OUT_OF_MEMORY`, with candidate state rolled back |
| Provider output disagrees with the committed concrete descriptor | `EXECUTION_FAILED` |
| Device loss during specialization or execution | `DEVICE_LOST` and the documented terminal-context behavior |

An implementation bug or late kernel rejection cannot be converted into
cross-provider execution fallback.

### Cache, capacity, and constant-only policy

Correctness uses the exact `ShapeSignature`. Physical tactics may use a
narrower `TacticSignature`; allocation reuse may use a larger
`CapacityClass`. Those identities are distinct.

Each context has a deterministic LRU bounded by both plan-entry count and
metadata bytes. Plans are immutable metadata, not complete allocation arenas.
One reusable context-local capacity pool grows transactionally and
geometrically within model, context, and device bounds. In-flight resources
are retired only after completion. Results never expose unused capacity.

A constant-only graph follows the same schema and mandatory shaped-input API.
Compilation resolves the single legal binding, and context creation
materializes one private plan and capacities. Repeated execution validates the
shaped views but skips symbol binding, graph-wide shape inference,
specialization, and LRU lookup. Proven constant storage-view operations may
alias kernel-facing views, but never public result storage or state shared by
another context.

### Deferred semantics

The first complete release deliberately excludes:

- dynamic rank and zero-length tensors;
- unbounded dimensions and dynamic weights or optimizer-state shapes;
- runtime shape tensors and shape-dependent control flow;
- arbitrary user-authored shape expressions;
- ragged tensors; and
- compact value-dependent outputs such as `NonZero` or compact NMS.

NonMaxSuppression retains bounded capacity plus padding/valid-count semantics.
Decode uses an independently tracked active length inside bounded KV capacity;
it must not specialize once per generated token. General data-dependent result
cardinality requires a later ADR.

### Portable publication gate

A package may claim the `portable` profile only after CPU JS, WASM, WebGPU, and
native CPU each prove and execute its complete declared operator/shape domain.
Warm profiles or one successful concrete shape are insufficient. Publication
also requires cross-language shape vectors, backend result parity, bounded
cache/memory evidence, re-export of all shipped packages, constant-only
regression results, dynamic cold/warm measurements, inference/full profile and
symbol checks, and unchanged release filenames.

Until that gate passes, dynamic packages may be exercised only under an
explicit experimental or atomic backend qualification and must not be labeled
portable.

## Consequences

- Every existing graph package must be re-exported; there is no runtime
  compatibility path.
- Application, provider, native, protobuf, Rust, exporter, and checkpoint
  integrations must migrate as their implementation phases land.
- Backends retain concrete kernel interfaces, gaining late-bound plans and
  reusable capacities rather than symbolic kernel dimensions.
- Dynamic execution can avoid padded work, but cold specialization and cache
  costs remain visible in reports and performance acceptance criteria.
- Training and PTQ adopt the shape contract only in the full profile; inference
  continues to compile and export no training implementation or symbol.
