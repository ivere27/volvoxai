# VolvoxAI architecture

VolvoxAI has two capability profiles:

- inference: forward execution only.
- full: inference plus training and PTQ authoring.

The browser release also has a backend-restricted full composition:
volvoxai.wasm.js contains strict WASM inference and training but omits CPU,
WebNN, WebGPU, WGSL, and Node filesystem implementations. It is a deployment
composition, not another numerical backend.

Release artifacts are flat. Source code is grouped by ownership.

## Public ownership model

JavaScript inference uses:

~~~text
Runtime
  BackendProvider[]

Model
  immutable logical definition and bounded shape constraints
  immutable logical topology, tensor shape specifications, fixed weights,
  and exact revision

CompiledModel
  retained Runtime
  exact retained model and weight revision
  immutable provider-prepared execution blueprint, invariant resources,
  bounded-domain proof, and report

ExecutionContext
  retained CompiledModel
  private shape binding, resolved shape-plan cache, capacities, request,
  scratch, output, and decode state
  FIFO operation queue

ExecutionResult
  stable named concrete host/device output snapshots

Trainer                         full profile only
  retained Model and exact base revision
  private gradients, optimizer slots, accumulation, RNG, and working revision

PTQPlan                         full profile only
  retained Model and exact pinned revision
  immutable template/spec and private CPU engine
  private mutable observations and sample registry
~~~

Native inference uses matching opaque VxRuntime, VxModel, VxCompiledModel,
VxExecutionContext, and VxResult handles. External backends create explicit
provider-runtime, compiled, and context instances.

Only physical devices, queues, allocators, submission synchronization, and
model-independent module/pipeline caches may be shared. Graph topology,
weights, request data, outputs, decode state, gradients,
optimizer state and accumulation always have a named owner.

## Logical shapes and bound execution ownership

The normative cutover decisions are recorded in
[the dynamic-shape v1 ADR](docs/adr-dynamic-shape-v1.md). Runtime-selected
parameter groups (router-selected LoRA families, MoE experts) are designed in
[the weight-bank design](docs/weight-bank-design.md). Implemented bank-aware
routes own residency, global-slot mapping, and transactional staging in their
execution context. A backend/operator pair without a proved slot-translation
path fails compilation; inference or training support is never inferred from
another route. The retired `AdapterManager` is deliberately not their basis.

`volvox-graph/v1` is dynamic-first. Each immutable `Model` owns
a logical graph: fixed-rank `TensorShapeSpec` values, positive constants or
references to bounded named dimensions, operator topology, fixed weight
descriptors, and an exact weight revision. A logical tensor has no request
buffer, allocation capacity, or request-specific byte count. Binding a request
never mutates the snapshot, definition ID, topology revision, or weight
revision.

The executable graph document is a closed schema. Unknown root, input, node,
and output-descriptor fields are rejected, and every node carries an explicit
`params` object (including `{}`), so exporter and runtime fingerprints cannot
silently disagree about preserved extensions or implicit defaults.

Kernels continue to consume only concrete positive integer shapes. Before
dispatch, an `ExecutionContext` resolves one complete `ShapeBinding` into a
`ResolvedShapePlan` and a private `BoundExecutionGraph`. The plan contains the
concrete descriptor and checked logical byte count for every live tensor,
concrete output descriptors, node geometry, tactic/dispatch metadata, and
memory requirements. Symbolic values never enter kernel-facing `Tensor.shape`
arrays.

Every dynamic `ResolvedShapePlan`, plan-cache entry, `BoundExecutionGraph`,
activation/scratch capacity, pointer table, uniform/bind group, and mutable
specialization generation is owned by one `ExecutionContext`. Different
contexts may bind different shapes concurrently. `CompiledModel` may share
only immutable topology analysis, invariant packed weights, tactic candidates,
and shape-independent generated code. It does not own a mutable current shape
or a mutable activation arena.

Plan lookup is keyed by an exact logical `ShapeSignature`. Tactic selection may
use a narrower `TacticSignature`, and allocation reuse may use a larger
`CapacityClass`; neither changes the logical signature or observable tensor
shape. A context keeps a deterministically evicted, entry- and byte-bounded LRU
of immutable plan metadata and one reusable capacity pool or arena per
dtype/device class. It never caches one complete mutable arena per shape.
Resources referenced by submitted work are not destroyed or reused until that
work completes. Results own exact logical output snapshots and never expose a
capacity tail.

## Shape binding lifecycle

Ordinary execution accepts a complete set of shaped tensor views. Shape is
mandatory even for constant-only inputs and is never inferred from byte
length. Within the context FIFO, every backend performs this sequence:

1. Normalize the complete input set without copying or uploading input data.
2. Validate names, storage dtypes, fixed ranks, positive dimensions, symbol
   bounds and equality, and exact element and byte counts.
3. Bind symbols and run canonical operator shape inference over the entire
   graph, producing every concrete intermediate and output descriptor.
4. Preflight quantization, kernel predicates, checked arithmetic, memory,
   metadata, dispatch, and device limits.
5. Look up or prepare a candidate context-local plan and candidate capacity
   growth without changing the committed binding.
6. Enter the provider's explicit mutation boundary. Publish the complete
   binding, plan/cache metadata, and resource generation only after the
   provider's fallible specialization writes have succeeded.
7. Copy or upload inputs, dispatch, and capture exact logical outputs into
   result-owned snapshots.

No tensor write, input upload, GPU submission, decode-state mutation, or
visible cache commit occurs before graph-wide validation and preflight
succeed. A failure before the mutation boundary discards candidate state and
leaves the previous successful binding and capacities usable. A failure after
that boundary clears retained decode state; if a partial specialization write
could have touched reused storage, the provider also invalidates the physical
signature so the next call must rewrite the complete binding. The context
remains usable unless it enters a documented terminal state such as device
loss.

## Bounded-domain compilation

Warm shape profiles are optimization hints and never narrow a model's legal
domain. During compilation, each candidate provider must prove a deterministic
route for every binding admitted by the model's declared bounds. The proof
covers rank and dtype support, equality and broadcast relationships,
contracted dimensions, divisibility or an in-provider generic fallback,
quantization-axis legality, and conservative maxima for tensors, scratch,
metadata, address spaces, buffer bindings, and dispatch dimensions. Proof uses
symbolic equality, intervals, and divisibility facts; one sample shape or an
enumeration of warm profiles is not proof of the bounded domain.

Provider policy may consider another allowed provider while compilation is
selecting a provider. Once a `CompiledModel` exists, selection is final:
execution revalidates each concrete binding against the proved contract but
never retries or falls back to another provider. An inconclusive domain proof
is `BACKEND_UNSUPPORTED`, not permission to defer discovery to dispatch.

The JavaScript composition SPI is `BackendProvider` v1.
Compilation receives one frozen logical compile view containing the exact `Model` and
its accepted canonical symbolic-domain proof. A provider capability explicitly
declares either full bounded-domain support or unsupported; a compiled provider
must attest the same proof identity plus internally consistent maximum tensor,
resident-resource, and resource-limit evidence. Execution receives one frozen
resolved request whose input, tensor, and output descriptors are the exact
members of the committed `ResolvedShapePlan`. Raw graph factories and raw
typed-array execution adapters are not part of this contract.

DS3's built-in CPU provider derives immutable tensor lifetimes from the logical
topology. Ordinary dynamic contexts pack exact typed views into one
dtype-separated best-fit arena. The layout is accepted by the bound-graph layer
only when its immutable proof is bound to the exact graph fingerprint and shape
signature, reproduces every input/producer/consumer/public-output lifetime, and
never overlaps simultaneously live regions. The independently packed
maximum-domain layout is the allocation bound and fallback offset layout; this
avoids assuming that best-fit fragmentation is monotone when tensor sizes
shrink. Geometric growth is clamped to that proved maximum.

The CPU resident proof includes the immutable snapshot and the context's
separate invariant-weight materialization. Its ordinary peak is the maximum of
`2*weights + initial arena`, `2*weights + old arena + candidate arena`, and
`2*weights + arena + exact result snapshots + audited typed scratch`. A context
created with explicit decode options instead uses persistent per-tensor slots:
incremental node selection relies on prior intermediate values across seed and
step calls, so topology-lifetime aliasing is forbidden there. Its separate peak
uses the same three phases with the maximum persistent activation sum. Decode
on an ordinary liveness context fails before mutation, and a retained decode
context is rejected at creation when its proved peak exceeds either CPU
ceiling. These ceilings are intentionally distinct: at most 512 MiB of
committed activation capacity (the ordinary arena or retained decode slots)
and at most 1 GiB of total resident/transaction memory. Compilation evidence
attests both values; the generic `resourceLimitBytes` field names the latter.
Constant-only contexts retain their persistent fast path. Execution consumes
the already resolved plan through the context-local CPU LRU/capacity
implementation without repeating shape inference. Other built-in providers
apply the same proof-identity rule through backend-specific limits. WASM
ordinary contexts pack exact activation views into dtype-separated
topology-liveness arenas; contexts created with explicit decode options instead
retain persistent per-tensor storage so incremental node selection can reuse
prior intermediates. WASM charges the actual linker-defined static/stack prefix
reported by the sidecar's `__heap_base` exactly once, then proves its wasm32
arena and metadata ceiling; this automatically covers module-local kernel
panels without duplicating their sizes in TypeScript. WebGPU proves
buffer/binding/dispatch limits before late binding, and WebNN proves its
supported operator subset plus descriptor/device limits before building
exact-shape graph variants. Runtime selection rejects a provider before context
mutation whenever it cannot attest the snapshot's complete declared domain.

A constant-only graph has the same schema and mandatory shaped-input contract.
Compilation resolves its only legal binding once. Each context materializes
one private plan and capacities lazily on its first execution; later executions
perform the cheap input dtype/shape/byte checks and dispatch without symbol
binding, graph shape inference, specialization, or plan-cache lookup. This is
the constant-only fast path; it does not preserve the legacy graph schema or
raw typed-array API. A statically proved storage-view operator such as Identity
may alias its kernel-facing input/output view inside that private bound graph;
capacity ownership remains context-local and every public result still receives
fresh exact logical storage.

Host provider snapshots explicitly declare transfer or borrowed ownership. A
transferred exact host snapshot is adopted; a borrowed snapshot is cloned once.
Device snapshots declare their exact logical byte count and may expose only the
smallest physical buffer padding required by the device API (four-byte WebGPU
alignment), never a reusable capacity tail. Rejected device snapshots are
released exactly once.

## Source map

TypeScript:

~~~text
ts/
  core/       graph/data objects, loading, snapshots, runtime handles, results
  ops/        reusable operators, validation, and normalization
  backends/   provider SPI plus CPU, WASM, WebGPU, and WebNN resources
  training/   Trainer, builders, autograd, optimizers, checkpoints, and PTQ
  index.ts    inference entry
  full.ts     inference plus full training entry
  wasm.ts     strict WASM inference and training entry
~~~

Model-specific sessions belong under examples/. They may consume public
runtime handles, but package entries never import or export them.

Native:

~~~text
native/
  include/        opaque inference API/provider SPI plus full-only Trainer/PTQ APIs
  src/runtime/    runtime/model/compiled/context/result ownership and execution
  src/kernels/    portable and optimized CPU/WASM kernels
  src/backends/   Vulkan, OpenGL, CUDA, Metal, and NNAPI integrations
  src/training/   full-profile backward, optimizer, and PTQ implementation
  src/shader_store.*  embedded shader lookup and development override
  src/tokenization/  opt-in application tokenizer
  cli/            fixed model-agnostic command application
  tests/          native unit and integration tests
  third_party/    vendored dependencies and provenance
~~~

`proto/volvoxai.proto` is the sole public contract for the optional in-process
Synurang FFI plugin under `runtime/` and the authoritative shared vocabulary for
logical `DataType` and `OperatorKind` values. It declares every
application-facing inference, Trainer, and PTQ operation plus typed statuses,
stages, policies, and tensor metadata. Internal kernel and optimizer registries
import this vocabulary instead of copying it into JSON or handwritten tables.
Generated C, TypeScript, Rust, and exporter projections are canonical. The
native C lifecycle is the implementation layer behind those handlers, not a
second FFI contract.

The JavaScript and native provider callback contracts are implementation
composition SPIs, not Synurang application operations. A Synurang caller
selects an already-composed provider by name through protobuf `BackendPolicy`
and controls its use through the generated lifecycle and `OperationReport`.
Application policy such as streaming generation or teacher forcing stays
downstream and reaches execution through opaque handles.

## Authoritative generated inputs

`proto/kernel_registry.proto` is the internal source of truth for forward
backend inventories, logical routes, exporter qualification, and migrated
physical kernel variants. Its generated Python, TypeScript, native C, and
Markdown projections are build outputs and are never edited by hand. Runtime
shape, dtype, device, and fallback predicates remain executable code referenced
by stable predicate IDs; the registry does not replace `CompiledModel` kernel
selection or parse protobuf at inference time.

`proto/optimizer_registry.proto` is the internal source of truth for typed
optimizer pass contracts, deterministic groups and recipes, semantic effects,
and target requirements. Its generated Python and Markdown projections are
build outputs and are never edited by hand. Each implementation ID resolves to
the concrete transactional `IRPass`; target and recipe applicability are
evaluated directly from typed generated fields. Portable graph legality is
evaluated against every member of the selected backend profile. Compile-backend
and tune-backend identities
select or measure derived `CompiledModel` preparation only; they never narrow
the persisted graph or its logical operator contract. Exact rewrites,
numerical migrations, and calibrated quantization authoring are distinct
registry semantics and require distinct caller policy and qualification.
Calibrated FP32-to-INT8 authoring is interface-preserving: it verifies the
ordered public input/output descriptors and existing ABI history before and
after its atomic commit. Calibration is bound to the exact graph fingerprint
and numeric execution provenance, but the generic exporter never interprets
application route, family, or task coverage. Applications may record those as
separate qualification evidence; they cannot use them to silently specialize
the RuntimeIR interface.
Portable legality, physical compilation, and measurement have distinct hashes,
so changing the tune device never invalidates the shared graph contract or
silently narrows it to WASM/native. Serialized physical plans remain derived,
strictly content-addressed inspection/search artifacts until a runtime consumer
is explicitly implemented; they are never substitutes for RuntimeIR.

WGSL under shaders/ is authoritative source shared by browser WebGPU and the
native shader compiler. Generated native shaders, packs, and embedded byte
arrays are build outputs and are never edited by hand.

A WGSL module whose first line is // @volvoxai-browser-only remains
authoritative browser source and is excluded from native packs by the
generator.

CUDA source is authoritative in native/src/backends/cuda_kernels.cu and its
full-profile training counterpart. CUDA builds generate deterministic PTX
arrays; PTX is not stored in the WGSL-derived pack.

VOLVOXAI_SHADER_DIR is a development override. Release execution uses embedded
assets. The runtime logs once only when an external override is actually used.

## Portable result contract

`backend_profile` controls persisted-graph legality. The `portable` profile is
the exact set `cpu-js`, `wasm`, `webgpu`, and `native-cpu`; a portable rewrite
must be legal on all four. An atomic profile such as `wasm` narrows that set only
when the package author explicitly selects it. `compile_backend` derives one
backend's immutable `CompiledModel`, and `tune_backend` identifies measurement
evidence. Neither setting changes the package profile or removes another
backend from it.

Every qualified backend route implements the same logical `OperatorKind`,
dtype, shape, quantization, saturation, tie, and mask contract for the domain it
advertises; an absent route is unsupported rather than an implicit fallback.
Integer and exact quantized conformance compares graph-visible results byte for
byte. Floating-point conformance uses the operator's declared tolerances because
parallel reduction and device arithmetic do not promise universal FP32 bit
identity; public task and routing results must still agree. A physical fusion
may eliminate dispatch, packing, or intermediate storage, but it must preserve
graph-visible Q/DQ and rounding boundaries. A rewrite that intentionally changes
those boundaries or the floating-point evaluation order is a persisted
numerical-migration candidate, not a backend-only optimization, and must be
explicitly selected and qualified across every member of its backend profile.

## Vocabulary

- Operator: semantic graph operation, such as MatMul.
- Kernel: one implementation of an operation for a dtype and target.
- Backend provider: device integration that compiles a Model and
  creates isolated contexts.
- Runtime: root owner for initialized providers.
- Model: immutable logical definition, topology, bounded tensor
  shape specifications, fixed weights, and exact model/weight revision;
  it has no request-specific binding. `Model.load` fetches and captures a
  package in one step; `capture`/`derive` remain for in-memory sources and
  successor weight revisions.
- CompiledModel: exact revision and immutable provider-prepared execution
  blueprint. It is VolvoxAI's PreparedGraph/compiled-graph owner, not another
  persisted model format. Backend-specific fusion, packing, scheduling, memory
  planning, and target code live here and can be rebuilt from the same portable
  model revision.
- ResolvedShapePlan: one context-owned immutable concrete tensor/geometry plan
  for an exact logical shape signature.
- BoundExecutionGraph: one context-private concrete graph view consumed by
  kernels; it never contains symbolic axes.
- ExecutionContext: one mutable shape-binding, capacity, request, and decode
  owner.
- ExecutionResult: stable named concrete output snapshots.
- Trainer: full-profile private gradients, optimizer, accumulation, working
  revision, explicit commit, and rollback.

## Dependency rules

1. Inference entries never compile, import, or export training code.
2. TypeScript core owns graph/tensor data, loading, immutable snapshots,
   results, lifecycle, tokenization, adapters, and runtime orchestration.
3. Reusable computation and model-independent graph contracts belong in
   ts/ops/. ModelLoader delegates portable quantized validation and layout
   normalization there.
4. Backends own device resources. GPUBuffer values are backend/result state,
   never properties of the portable Tensor data model.
5. A backend never depends on another backend.
6. Built-in and external providers implement the same provider/compiled/context
   contract. A context-isolation claim requires independent mutable state.
7. Training may depend on core, ops, kernels, and provider interfaces. Core and
   inference entries never depend on training.
8. Trainer steps mutate only private state. Explicit commit returns one
   successor Model after a successful applied update; rollback
   restores the last committed baseline. No step publishes implicitly.
9. Model-family constructors, preprocessing, vocabulary policy, generation,
   and task postprocessing belong under examples/ or downstream applications.
10. Graph-document node inputs resolve only to declared graph inputs, loaded
    weights, or earlier node outputs.
11. Kernels never depend on CLI, builders, checkpoints, or sessions.
12. CLI and FFI handlers use public opaque handles; they do not resolve runtime
    state through internal tables.
13. Optional native features are composed by source/profile boundaries.
    Inference builds contain no training object or public training symbol.
14. Calibration and quantized package authoring are full-profile capabilities.
    Inference may execute quantized graphs but does not contain PTQ authoring.

## Lifecycle rules

- Each context serializes execute, decode seed/step/reset, adapter selection,
  and close through one FIFO.
- Different contexts may progress concurrently.
- A failed operation does not poison later work unless the context reaches a
  terminal state such as device loss.
- Starting close rejects new work and drains accepted work.
- JavaScript close/dispose is idempotent.
- Native retain/release is explicit; release(NULL) is a no-op.
- A child retains every parent needed for its operation.
- Context closure does not invalidate a stable result.
- Provider selection ends during compilation. Execution never retries on
  another provider.

## Build composition

~~~text
volvoxai.js           readable multi-backend inference
volvoxai.min.js       minified multi-backend inference
volvoxai.full.js      readable inference plus training/PTQ
volvoxai.full.min.js  minified inference plus training/PTQ
volvoxai.wasm.js      readable strict WASM inference/training
volvoxai.wasm.min.js  minified strict WASM inference/training
volvoxai.wasm         forward C/WASM kernels
volvoxai.full.wasm    forward plus training/PTQ C/WASM kernels
volvoxai              native inference runtime
volvoxai-full         native inference plus training/PTQ
~~~

Browser files are emitted under dist/<package-version>/. The inference
JavaScript profile resolves volvoxai.wasm; full and WASM-only profiles resolve
volvoxai.full.wasm.

Strict WASM training preflights the supported graph before forward execution or
optimizer mutation. JavaScript owns graph state, revisions, results,
checkpoints, safetensors, and browser file/storage policy.

Optional WASM instruction-set code is embedded as a child module. The baseline
parent validates independently, instantiates a supported child against its
memory, and otherwise uses baseline kernels without another fetch.

Native shader packs use one XZ block per enabled backend format and scope.
Inference packs contain forward blocks; full packs add training blocks.
CPU-only builds carry an empty pack. The first lookup decodes and caches only
the requested block.

Internal static targets may be modular. Distributed JavaScript bundles and
native executables remain single files apart from their WASM sidecar.

## Model package identity

Inference packages use:

~~~text
graph.json
*.safetensors
~~~

Every graph root, including named subgraphs such as `router.graph.json`, carries
the exact case-sensitive format discriminator:

~~~json
{
  "format": "volvox-graph/v1"
}
~~~

Bounded dynamic shape is intrinsic to `volvox-graph/v1`, not a separately
versioned capability. Loaders do not guess another schema from
`outputs_shape` or any other field, and there is no legacy reader or in-place
compatibility conversion. The closed root schema rejects extra discriminators.

The shape-bearing portion of the graph schema is:

~~~json
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
~~~

`dimensions` is required and may be empty for a constant-only graph. Symbol
names are case-sensitive ASCII strings matching
`^[A-Za-z][A-Za-z0-9_]{0,63}$`. Each declaration contains required positive
integer `min` and `max` and an optional positive integer `multiple_of`, which
defaults to one. The legal values satisfy `min <= value <= max` and
`value % multiple_of == 0`; declarations with an empty legal set are invalid.
Every shape has fixed rank, and each axis is either a positive integer constant
or a reference to a declared symbol. Dynamic rank, zero extents, unbounded
symbols, and arbitrary JSON shape expressions are not part of this contract.

Each node has a non-empty unique string `id`. Its `outputs` map replaces the
legacy `outputs`, `outputs_shape`, and `outputs_dtype` combination: every port
maps to one descriptor containing exact `tensor`, `dtype`, and `shape` fields.
The shape is an assertion checked against canonical operator inference. An
already-bound symbol must match; an output-only symbol may be bound for the
first time by that inference and must then satisfy its declared constraint.
An unbound symbol cannot constrain an ordinary node input. Derived formulas,
including convolution and pooling output dimensions, live in canonical
operator shape code rather than graph JSON.

JSON object member order is never semantic. Canonical model hashing sorts
dimension names, graph input names, and descriptor-map keys by unsigned UTF-8
byte order; node-array and public-output-array order remain semantic. A
`ShapeSignature` uses graph input names in that same sorted order and records
each name, rank, and concrete axis list. Repeated symbols are recorded through
their public axes rather than through object insertion order.

All graph dimensions are positive integers no larger than
`Number.MAX_SAFE_INTEGER` (`2^53 - 1`). Shape-formula intermediates use checked
safe-integer arithmetic and may be signed; finalized dimensions, element
products, and byte counts must be positive safe integers. Addition,
multiplication, and rounded division fail instead of wrapping or losing
precision. Compilation additionally proves the entire declared domain against
the selected target's `size_t`, signed and unsigned kernel metadata, WASM
address/page limits, and physical device buffer, binding, and dispatch limits.
The `portable` profile must meet the strictest applicable limit of all four
portable members.

Loaders accept only `graph.json` or named `*.graph.json` documents and never
search alternate filenames. Public source fields use `graph_path` and
`graphUrl`. A persisted graph declares a non-empty array of unique output
tensor names; only programmatic authoring may infer temporary leaf outputs.

The optimized `volvox-graph/v1` document and its safetensors are the portable
deployment source. Provider compilation may derive an operator schedule,
kernel choices, physical layouts, packed constants, memory plans, pipelines,
or target code for one exact revision. Invariant prepared state belongs to
CompiledModel and may be rebuilt; it is not published as `wasm.graph.json` or
as another required graph schema. ExecutionContext materializes and owns every
shape-specific plan, mutable per-execution arena, pointer table, scratch
buffer, request, output, adapter route, and decode state. Multiple contexts may
share prepared state only when it is immutable and shape-independent.

### WASM dynamic-shape memory ownership

The WASM compiled model owns one frozen, pointer-free operator schedule derived
from the logical topology. Core shape binding supplies each provider context a
metadata-only canonical plan at the smallest legal public shape, rounded for
every `multiple_of` constraint and normalized with that context's bank
residency. WASM instantiates its schedule against this plan during context
creation, before request timing, without synthesizing inputs or dispatching an
operator. The first real request therefore reuses the prepared shape or performs
the ordinary transactional rebind; shape changes rebuild descriptors and
offsets, not topology, invariant packs, or kernel routing. A failed eager
specialization disposes only its new fork and never publishes a context.

Within a context's linear memory, raw invariant weights and any F32/Q8 packed
panels form a persistent prefix. `heap_mark` records the start of a single
activation/metadata/scratch suffix, and `heap_rewind` may move the allocator only
to an aligned address in its already allocated prefix. A rebind first dry-runs
the complete activation layout, metadata, and maximum scratch allocation,
checks signed kernel arguments and wasm32 limits, and pre-grows memory. It then
rewinds and commits the measured layout. If measurement or commit fails, the
previous graph, capacities, pointers, descriptors, and shape signature are
restored; input bytes have not yet entered WASM memory.

Activation owners grow geometrically up to the per-tensor maxima attested for
the full bounded domain. Exact shape plans and bound graph metadata live in a
deterministic eight-entry, one-MiB context LRU, while all signatures share the
one physical arena. Cached typed-array views are recreated after every memory
growth. Public results always clone exact logical output bytes into
result-owned host storage, so arena reuse or context closure cannot mutate an
earlier result. Telemetry separates raw persistent weights, packed weights,
logical activation bytes, current/high-water activation capacity, variant
metadata/scratch, memory growth, plan-cache activity, and invariant pack counts.
The resource attestation sums individually aligned persistent descriptors for
every operator at its domain maximum, adds only the largest shared QConv scratch
request, and bounds linear memory as the greater of the module's initial extent
or the proved allocator end plus geometric-growth headroom. Host LRU metadata is
charged separately. Unknown descriptor-owning operators fail compilation until
their allocation rule is audited.

WASM's resident proof also covers provider- and result-owned host payloads. Let
`L0` be the initial linear-memory extent, `L` its proved maximum, `P` the host
plan-cache budget, `W` the complete raw weight revision, `B` the raw bytes of
all banked weights, and `R` the sum of maximum-domain public-output bytes. The
immutable `Model` and the context's invariant materialization contribute
`2*W`; linear-memory weights are already in `L`. Explicitly selecting all slots
of every bank is legal and retains one additional host `B`, while each result is
cloned once from its linear-memory view and transferred without another copy.
Bank staging visits canonical weight-name order. For bank `i` of full size
`F_i`, with `Q_i` bytes in preceding bank slices, its double-copy transaction is
`Q_i + 2*F_i`; `T` is the maximum of those transactions and `B`. The ordinary
peak is therefore `P + 2*W + max(L + B + R, L0 + T)`.

Decode has a distinct resident maximum because core retains public inputs
between seed and step. A reseed or fully changed step keeps the old snapshot
while cloning the replacement after the new result exists. With `I` equal to
the summed maximum-domain input bytes, its peak is
`P + 2*W + max(L + B + R + 2*I, L0 + T)`. Compilation evidence exposes both
maxima and the component terms. Ordinary compilation may remain valid when the
decode maximum exceeds the resident ceiling, but an explicit decode context is
then rejected before its WASM engine is forked.

## Change rules

- Separate file moves from behavioral changes when practical.
- Every operator starts with a portable implementation and correctness test.
- Accelerated kernels are compared with the portable CPU reference.
- Hot-path changes include reproducible benchmark evidence.
- Public API and model-format changes update current documentation and tests.
- CI typechecks/tests JavaScript profiles, builds both WASM sidecars, validates
  all eight browser artifacts, rebuilds both native profiles, and checks
  inference/full symbol boundaries.
- Generated files carry a DO NOT EDIT marker and deterministic input hash.
