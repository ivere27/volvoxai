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
  Model[]

Model
  immutable definition
  atomically published weight revision

CompiledModel
  retained Runtime
  exact retained model/weight/adapter revision
  immutable provider-prepared execution blueprint, resources, and report

ExecutionContext
  retained CompiledModel
  private request, scratch, output, adapter, and decode state
  FIFO operation queue

ExecutionResult
  stable named host/device output snapshots

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
weights, request data, outputs, adapter routes, decode state, gradients,
optimizer state and accumulation always have a named owner.

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

Every backend implements the same logical `OperatorKind`, dtype, shape,
quantization, saturation, tie, and mask contract. Integer and exact quantized
conformance compares graph-visible results byte for byte. Floating-point
conformance uses the operator's declared tolerances because parallel reduction
and device arithmetic do not promise universal FP32 bit identity; public task
and routing results must still agree. A physical fusion may eliminate dispatch,
packing, or intermediate storage, but it must preserve graph-visible Q/DQ and
rounding boundaries. A rewrite that intentionally changes those boundaries or
the floating-point evaluation order is a persisted numerical-migration
candidate, not a backend-only optimization, and must be explicitly selected and
qualified across every member of its backend profile.

## Vocabulary

- Operator: semantic graph operation, such as MatMul.
- Kernel: one implementation of an operation for a dtype and target.
- Backend provider: device integration that compiles a ModelSnapshot and
  creates isolated contexts.
- Runtime: root owner for initialized providers and models.
- Model: immutable definition plus atomically published weight revision.
- CompiledModel: exact revision and immutable provider-prepared execution
  blueprint. It is VolvoxAI's PreparedGraph/compiled-graph owner, not another
  persisted model format. Backend-specific fusion, packing, scheduling, memory
  planning, and target code live here and can be rebuilt from the same portable
  model revision.
- ExecutionContext: one mutable request/decode owner.
- ExecutionResult: stable named output snapshots.
- Trainer: full-profile private gradients, optimizer, accumulation, working
  revision, explicit commit, and rollback.

## Dependency rules

1. Inference entries never compile, import, or export training code.
2. TypeScript core owns graph/tensor data, loading, immutable snapshots,
   results, lifecycle, tokenization, adapters, and runtime orchestration.
3. Reusable computation and model-independent graph contracts belong in
   ts/ops/. GraphLoader delegates portable quantized validation and layout
   normalization there.
4. Backends own device resources. GPUBuffer values are backend/result state,
   never properties of the portable Tensor data model.
5. A backend never depends on another backend.
6. Built-in and external providers implement the same provider/compiled/context
   contract. A context-isolation claim requires independent mutable state.
7. Training may depend on core, ops, kernels, and provider interfaces. Core and
   inference entries never depend on training.
8. Trainer steps mutate only private state. Explicit commit publishes one
   successor Model revision after a successful applied update; rollback restores
   the last committed baseline. No step publishes implicitly.
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

Every graph root, including named subgraphs such as router.graph.json, carries
the exact case-sensitive discriminator:

~~~json
{
  "format": "volvox-graph/v1"
}
~~~

Loaders accept only `graph.json` or named `*.graph.json` documents and never
search alternate filenames. Public source fields use `graph_path` and
`graphUrl`. A persisted graph declares a non-empty array of unique output
tensor names; only programmatic authoring may infer temporary leaf outputs.

The optimized `volvox-graph/v1` document and its safetensors are the portable
deployment source. Provider compilation may derive an operator schedule,
kernel choices, physical layouts, packed constants, memory plans, pipelines,
or target code for one exact revision. That prepared state belongs to
CompiledModel and may be rebuilt; it is not published as `wasm.graph.json` or
as another required graph schema. ExecutionContext materializes and owns every
mutable per-execution arena, pointer table, scratch buffer, request, output,
adapter route, and decode state. Multiple contexts may share prepared state
only when it is immutable.

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
