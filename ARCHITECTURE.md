# VolvoxAI architecture

VolvoxAI turns a graph and its weights into a reusable computation on a CPU or
GPU. The same model package can serve a browser page, a native application, or
an edge service. This document explains where the work and memory live, why
compilation is separated from execution, and how to change the engine while
preserving those boundaries. For a first application, start with the
[quickstart](docs/quickstart.md); for concepts, use the [textbook](docs/textbook/README.md).

## Execution model

A model package describes operations, weights, inputs, and legal shapes.
Loading validates that description. Compilation chooses a backend and prepares
an executable for the admitted domain. Each request binds concrete inputs;
execution produces named output snapshots that the application can retain.

```text
Application: input preparation, task policy, output interpretation
  Model: graph and weight revision
    Compilation: validate shapes and select a backend
      ExecutionContext: current inputs, scratch, and decode state
        CPU kernels or GPU dispatches
          Result: independently retained named outputs
```

Both web profiles have one persistent C owner per host. Inference, scheduler,
planning, and the full profile's training/PTQ services share that owner's handle
registry and linear memory. Module compilation can be cached across hosts;
instances, handles and mutable memory are never shared. There is no JavaScript
numerical kernel, graph executor, backend provider API or compatibility facade.

Product TypeScript is limited to generated API projections, transport and the
WebGPU device bridge. Transport serializes protobuf calls, initializes the
companion WASM, transfers path-named input files and closes its owner. The bridge
owns browser GPU objects and asynchronous device operations. Former runtime
features, including model assembly and tokenization, run in C behind the
proto contract. Development-only registry projections live under
`tools/generated/` and are not imported by product entries.

## Profiles and release composition

| Entry | Host | C services | Backend availability |
| --- | --- | --- | --- |
| `volvoxai.js` | `EngineHost` | Platform, Text, Planning, Inference, Scheduler | WASM CPU inference |
| `volvoxai.full.js` | `FullEngineHost` | All services | WASM CPU and WebGPU inference/training; C PTQ |
| `native/volvoxai` | Generated C dispatch | Inference profile | CPU and compiled native GPU backends |
| `native/volvoxai-full` | Generated C dispatch | Full profile | Inference plus native training/PTQ |

Every JS entry also has its fixed `.min.js` sibling. Inference uses
`dist/<version>/volvoxai.wasm`; full uses `volvoxai.full.wasm`. Those four JS, two
WASM and two native files are the complete release inventory. Do not add another
control sidecar or embed a WASM copy into JS. The WASM parents contain their C
services directly, with no PTQ or relaxed-SIMD child module. SIMD128 remains in
portable numerical kernels.

Browser inference excludes the GPU bridge and all shaders; its WASM companion
compiles without WebGPU and imports monotonic time, entropy and the Synurang wakeup notification. All inference
profiles exclude training code, codecs, public training symbols and training
shaders physically. Full-only C objects are selected by the central recipe;
changing an exported surface or source list requires updating and validating
that recipe. The inference entry must never import full to reuse a helper.
Graph parameter metadata is common: inference can load a graph containing
attention Dropout parameters and seeds. Only the full Trainer executes their
training behavior; reading those graph attributes adds no training dependency.

`tools/release_profiles.mjs` owns browser/WASM recipes. CMake owns native object
composition and optimization classes. Both use explicit kernel/control/service
source roles. Numerical C kernels and libm compile with `-O3`; portable control
and service code use the size profile. Native ISA variants remain separate
objects, with ISA instructions excluded from baseline/control objects.

Generated kernel registry headers declare fixed-size tables; their companion
C files define each table once per linked runtime. Common metadata and full-only
variants remain separate. WASM includes those definitions in its C owner.
Native GPU admission recognizes only compiled backends, allowing browser and
CPU-only builds to discard native device proofs while full WebGPU retains its
shared validation helpers.

Release WASM links strip debug names and tool metadata. The Unicode license and
release provenance remain embedded; imports, exports and the proto API are
unchanged by stripping.

Release provenance binds source dependencies, compiler/linker identity and
runtime libraries, flags, ordered objects, profile and artifact hashes. WASM
verification rebuilds from the central recipe and compares payload and object
evidence. JS verification checks the profile closure and a reproducible bundle.
A successful local compile alone does not establish release compatibility.

The build Dockerfile pins the Ubuntu image, archive snapshot and LLVM package
version that supply the WASM toolchain. Update those inputs and the runtime
library hashes in `tools/release_profiles.mjs` together, then rebuild and verify
both profiles. The image also installs protoc and the Python dependencies from
`tools/requirements-codegen.txt` for the generated API checks in JS builds.

Build JS before WASM. JS builds remove stale WASM companions, so these builds
must not run concurrently. `check:release` verifies all eight files and native
inference/full symbol boundaries before packaging.

## Shapes, compilation, and reuse

A dynamic dimension has a name and finite bounds. If inputs share a symbol,
they must bind it to the same extent in a request. This lets the compiler reason
about the whole legal domain while the execution context prepares only the
concrete binding it needs. Equal byte counts do not imply equal shapes, and
checking only the minimum and maximum examples does not prove every shape.

For example, a transformer can accept `tokens: [1, S]` with `1 <= S <= 256`.
The logical graph keeps `S`; a six-token request binds it to 6. Fixed weight
storage belongs to the compiled revision. Mutable activations, layout plans,
and decode state belong to contexts, so one session's shape change cannot
rewrite another's inputs or retained outputs.

Compilation and request binding have different jobs:

| Phase | Responsibility |
| --- | --- |
| Load | Validate graph syntax, tensor references, weight storage, and bounds |
| Compile | Prove supported operator/shape behavior, choose a backend, prepare immutable resources |
| Bind | Validate the complete concrete input batch and reserve its required storage |
| Execute | Run the prepared work with this context's inputs and state |
| Publish result | Retain the exact logical output bytes independently of future requests |

The [dynamic-shape ADR](docs/adr-dynamic-shape-v1.md) explains the design choices.
The [scheduling guide](docs/scheduling-and-dynamic-batching-design.md) distinguishes
caller-authored bulk tensors, coalesced independent requests, and stateful decode
lanes. They all use a batch dimension, but their ownership and admission rules
are different.

## C ownership and lifecycle

```text
Runtime
  Model: immutable graph and weight revision
    CompiledModel: one admitted backend and immutable prepared resources
      ExecutionContext: mutable shape/decode state and scratch
    Trainer / PTQPlan: retained exact model revision (full only)
  Request: admitted scheduler input and result reservation
  ExecutionResult: independently retained output snapshot and budget lease
```

Public IDs are opaque owner-scoped capabilities. Release idempotently retires an
ID; accepted operations and descendants retain internal references. Releasing a
Runtime or Model ID does not destroy live descendants. Host close cancels calls, retires the entire module owner and drains cleanup.
Native modules have separate registries, just like separate WASM instances.
The module retains Runtime roots while descendants or asynchronous engine work
exist; idle roots are reaped after a polling turn. Work that finishes while the
host is idle is reaped at the next call or host close. Native shutdown returns
PENDING until its cleanup worker has joined every engine worker. Entropy-scoped IDs prevent a restarted
owner from accepting handles retained by an earlier instance.

Inputs are validated as an atomic named batch before mutation. Every tensor
carries explicit dtype, concrete shape and exact byte payload. Dynamic dimensions
have fixed rank and finite positive bounds. C validates symbol equality, byte
counts, operator rules, checked arithmetic, quantization and target limits.
Results contain exact logical bytes, never arena capacity tails.

Input validation reports retain the first violation as typed evidence: input
name/index, expected TensorSpec, actual TensorInfo, axis, required symbol/decode
extent and byte count when known. Reports own bounded copies of this evidence;
they never retain caller tensor pointers or echo payloads. A truncation flag
directs consumers to GetModelInfo for full names. That query exposes immutable
input/output contracts before compilation, including symbolic dimensions.

Model owns a parsed canonical JSON graph and semantic plan. Proof/planning borrow
that immutable view. Compilation clones it only for private lowering, then the
private execution engine parses the lowered representation. This removes the
extra source-graph parse; it does not claim allocation-free compilation or a
single representation for every internal phase. Instrumentation is described
in `docs/c-runtime-validation.md`.

Context operations serialize mutable state. Pre-commit refusal preserves the
prior binding and decode state. A failure after submission does not promise
rollback of device side effects; continuation, new prefill, and owner recreation
depend on the failure contract and its regression evidence. Device loss makes
resources on that device unusable.
Trainer steps mutate private parameters; commit publishes a successor model
revision, while rollback restores the committed baseline. PTQ plans retain the
exact model being calibrated and commit observation ranges atomically.

## Transport and storage

Generated C handlers are the authority for operation validation and reports.
The TypeScript clients are Synurang-generated `Vx*ServiceClient` classes:
`await client.method(request, { signal, timeoutMs })`. Native and WASM share
`open / send / half_close / receive / cancel / release` and bounded polling.
The host waits for wakeups when no work is ready; it has no periodic idle timer.
Requests are encoded before yielding to preserve a call-time byte snapshot.

`EngineHost` translates domain `OperationReport` failures into `VolvoxAIError`,
retaining the original response, report and method path. Synurang `RpcError`
represents call status, cancellation and deadlines. Transport checks the
terminal call status before reporting success. A successful PENDING response
still means engine work is live and requires the corresponding proto query.
Call completion/release does not release Model, Context, Result or Request IDs.

`WaitRequest(RequestRef)` is asynchronous: internal request watchers post wakeups
and the module samples completion in bounded polling turns. A cooperative WASM
turn may start one scheduler dispatch; it cannot preempt an individual CPU
kernel. Cancelling or timing out this RPC removes only its watcher. The separate
`CancelRequest` operation changes engine work. Shared memory and new queue
transports are deferred in `TODO.md`.

`NativeStatus`, `OperationCode` and `OperationStage` are the only public outcome,
detail-code and stage vocabulary. C reports carry these generated enums;
messages never determine an error's classification. The TypeScript error
adapter preserves `status`, `code` and `stage` directly, including full-profile
reports, without a separate phase vocabulary. RPC paths and service transport
names are generated from the service declarations in the same proto.

Native paths resolve through the operating system. Web `LoadModel` and
`PublishAdapter` paths are fetched and mounted into a private C VFS. A C
preflight runs generated decoding and path validation before network I/O.
File staging is bounded and privately named per call. Await a call before
releasing handles it needs; asynchronous preparation is not a transaction with
other calls. The default and hard upper
bound is 64 MiB per package transfer, streamed in at most 1 MiB blocks. An
application may tighten the limit. VFS entries grow dynamically and retain
stable storage while model maps/readers hold references; unmounting a transport
name does not revoke an accepted model's bytes.

LoadModel also accepts ModelPackage graph/SafeTensors bytes directly. Exactly
one source form is accepted; bytes and paths cannot be mixed. The common C
loader validates source selection, shard counts and the 64 MiB package limit
before snapshotting into the same retained storage used by path loading
(private files on native, VFS in WASM). Web transport applies a tighter configured
maxPackageBytes to inline packages as well. It performs no fetch or path
resolution for this source form.

DescribeApi exposes the active build's methods and, on request, their dependent
message/enum contracts. Service and method filters limit the response. A local
generator reads protobuf descriptors and validated @api annotations in the
schema comments to produce typed metadata tables and JSON/Markdown references.
The same profile filter used for Synurang excludes full-only contracts from
inference tables. This metadata documents required fields, presence, defaults,
oneofs and explicit semantic rules; C validators remain authoritative for
graph-dependent constraints and backend admission.

Full authoring/export calls use protobuf bytes in WASM. `ExportTrainerWeights`
returns SafeTensors shards. `ExportTrainerCheckpoint` returns a generated proto
message containing the exact logical graph, private weight shards, optimizer
SafeTensors and configuration, optimizer step, RNG seed and opaque application metadata.
`CreateTrainer.checkpoint` validates that state against the requested Model
before publishing a Trainer ID. The private checkpoint becomes the rollback
baseline until commit. `GetTrainerState` and `ResetTrainerAccumulation` inspect
or discard an unfinished gradient window in C; neither submits GPU work. State
includes the current tensor binding, activation capacity and plan cache usage.
Optimizer fields omitted from a step keep the last successful configuration;
rollback and checkpoint restoration preserve that configuration with the weights.
Quantized LoRA composes these same C operations: `DequantizeWeight` initializes
F32 masters, `TrainStep` updates their private Trainer, and
`ExportQuantizedTrainerWeights` produces an immutable I8 successor package for
`LoadModel`. Export preserves master values and template storage. Publishing a
new inference revision never changes an older model or retained result.
`CreateTrainer.shape_options` controls LRU entry/metadata limits and activation
capacity/growth. Defaults match the former implementation: 8 entries, 1 MiB,
512 MiB and factor 2. A plan exceeding the metadata budget executes uncached.
State lists cached shape signatures from least to most recently used and reports
metadata/storage bytes, oversize skips and uncached admissions. These are C
layout plans; compiled C operators replace the former JavaScript execution
closures. The counters do not claim a cache of GPU backward command lists.
Trainer engines retain an unfused graph and all forward activations in a reusable
C arena; step boundaries preserve capacity. Binding validates and reserves the
new layout before replacing the current one. State reports actual C metadata,
activation high water including initialization, and logical F32 gradient shapes.
`AuthorPtqTemplate` accepts graph and weight bytes;
`CreatePtqPlan` snapshots template bytes; `WritePtqPackage` returns graph and
weight bytes. Native callers may use the declared path forms. WASM rejects
unsupported output paths with `TRANSPORT_UNSUPPORTED`, without pretending to
write a browser download. Application code chooses how to save returned bytes.

Freestanding decimal parsing/formatting and scalar math execute in the project's
`native/src/runtime/wasm_decimal.c` and `wasm_math.c`. Decimal conversion uses
exact integers; math uses range reduction and convergent series with constants
derived by `tools/generate_wasm_math.py`. Their bounds and validation contract
are documented in [wasm_numbers.md](native/src/runtime/wasm_numbers.md).
Native releases use platform libc/libm. The non-GPU runtime imports are
monotonic time, entropy and Synurang wakeup; there are no JavaScript math or JSON-number
conversion imports.

`DecodeStep.dependency_update` explicitly recomputes the full dependency closure
of supplied inputs in a prefilled single-lane AUTO context. Empty inputs execute
no nodes, and the cursor is preserved. An omitted cursor continues ordinary
row advancement. Dependency updates reject REQUIRED row contexts and paged KV
before mutation; these stateful row routes have their own addressing contract.

## Text processing

`VxTextService` creates immutable C-owned tokenizers in both profiles. Vocabulary
and merge bytes are copied before the handle is published. JSON dictionaries,
little-endian binary dictionaries and explicit proto token entries share greedy
and ranked BPE execution; no tokenizer policy or state lives in TypeScript.
Encode supports the default pretokenized mode, explicit greedy matching and a
single BPE word. Decode converts each token independently with UTF-8 replacement,
including the former space-marker behavior. Unknown bytes/IDs retain the previous
skip/empty behavior, and an omitted output limit remains 256 tokens.

Pretokenization uses pinned Unicode 17.0.0 letter/number intervals generated from
upstream data, plus the explicit ECMAScript whitespace set. The Unicode generator
downloads the pinned source into memory and verifies its SHA-256 on every
generation or check; those commands require network access. The generated C
table remains committed and is embedded in both profiles. JSON token keys are
length-aware, including NUL; malformed UTF-8, unpaired JSON surrogates, non-integer
or duplicate IDs and truncated dictionaries are rejected before publication.
The Unicode notice is retained in native artifacts and the WASM `license.unicode`
custom section. Tests keep the unchanged `99dfd8f` TypeScript tokenizer as a
reference outside every product entry.

## WebGPU planning and asynchronous completion

`native/src/backends/webgpu_backend.c` creates physical dispatches. C chooses
operators, shader variants, parameters, spans and workgroups. TypeScript only
translates generated private commands into device objects, encodes/submits them,
handles validation scopes, maps staging buffers and reports device loss.

Only `FullEngineHost` instantiates WASM with stable deferred GPU imports. Before dispatch,
the private C transport preflight decodes the proto request and checks its live
model and backend policy. A valid CompileModel candidate list containing WebGPU, or a live-model
CreateTrainer request for WebGPU, requests asynchronous device preparation; metadata, CPU-only and invalid requests
do not acquire a device. TypeScript forwards that decision without selecting a
backend or replaying the operation. The generated public dispatch still runs once.

Private ABI signature/layout and shader IDs are generated. C/WASM and JS must
agree on both the GPU ABI hash and shader catalogue hash before enabling the
bridge. The full catalogue contains inference and training shaders. The private
C header selects the same generated closure as the full browser entry; browser
inference includes neither catalogue. Device limits include buffer/binding sizes, workgroup axes and WGSL
shared memory. Compile constructs and validates C plans without submitting GPU
work. Variants exceeding device limits are excluded before allocation.

Compile admits WebGPU only when it can execute the entire numerical graph.
`REQUIRE` returns a typed refusal otherwise. `PREFER` may select the whole model
on WASM. A graph cannot hide a GPU-to-CPU numerical handoff in the synchronous C
loop, and an execution failure is never retried on another backend.

Execute/Run/decode submit once and return a stable execution ID and result ID.
`PENDING` means accepted work. `GetResult` is a nonblocking completion query;
`ReadOutput` returns `BUSY` while pending and does not write its destination.
Terminal success publishes every output together. Failure publishes its report.

Each submission copies outputs into independent GPU staging snapshots. Later
input mutations, other contexts and future executions cannot change old results.
Callbacks only update bridge-owned completion data; they never write to a WASM
pointer. C polls a live ticket before copying into its retained snapshot. Release
and cancellation can retire pending results safely. A failed map does not destroy
buffers while already submitted queue work can still access them.

GPU spans and metadata arenas belong to contexts. Stable scratch blocks and
span tables grow with the admitted graph and row geometry; growing storage never
moves an address already used by an encoded command. Device limits and allocation
failures bound growth, separately from Runtime result budgets. Closing the owner drains pending
scopes/tickets, releases spans and destroys an automatically acquired device.
Closed host and WASM owners drop their bridge references; deferred imports stop
delegating immediately and a device arriving during close is drained and destroyed.
No graph state or numerical policy belongs in the device bridge.

Additional admission proofs cover the canonical typed graph domain:

- Dynamic shape: C proves symbolic relationships across the entire domain,
  then checks maximum descriptors against the device. Typed copies, quantized
  operators, convolution and attention use the same planners as fixed graphs.
  Dense/norm feature widths and affine banks retain their shader invariants;
  Softmax and reductions can vary their reduction width. Coincident endpoint
  values do not establish equality of independent symbols. Maximum descriptor
  checks use private scratch/span metadata and never alter the live context.
- Independent batching: C proves that operators preserve the leading request
  axis, including dense weight layouts and canonical normalization defaults.
  One aggregate GPU graph invocation feeds retained lane snapshots. Cancellation
  of a lane cannot revoke another; both lane and aggregate copies are budgeted.
- Required rows: scalar and explicitly declared lane batches share C row geometry.
  Pointwise/broadcast operations, typed dense/embedding, affine transforms, views,
  feature normalization and self/memory attention gather their row operands,
  invoke the existing numerical kernel and scatter only active lanes. Invariant
  branches stay cached. C projects causal lengths and K/BK/QK/BQK masks into
  each lane's attention row. CPU and WebGPU use the same geometry; neither has
  a fixed 32-lane limit. `DecodeLaneAction` declares advance, idle or parked
  lanes, and `GetDecodeState` returns per-lane lengths and cache generation.
  Required-row requests are proved before input mutation. Omitted step inputs
  reuse values and refresh the context's declared `decode_inputs` roots.

Runtime registration and exporter qualification serve different purposes.
Compilation requires a registered route, the canonical shape contract and the
backend's physical proof. An exporter's tested conversion subset does not limit
which proved runtime graphs the public API can compile.

Index and MoE route admission runs across the graph before input publication or
GPU commands. Public values are checked at binding; immutable values and known
device producers are proved at compile time. Device-produced values are never
validated by reading stale host buffers. Partially resident Gather/MoE banks
also require slot membership. Their index domain is the complete declared bank
extent, including absent trailing slots, so negative Gather indices retain the
same meaning across resident subsets.

Executable nodes store only `VxOperatorKind`, generated from
`proto/operator_vocabulary.proto`. Graph/API parsing converts names to that enum;
CPU, GPU, training, parameter validation and fusion compare the enum. Graph
serialization and diagnostics convert it back to its canonical name. Unknown
names are rejected before a node is published, including graph edits.

Fixed input/output roles use `PortKind` from the same vocabulary. Each executable
reference resolves its role when created; CPU, GPU and training lookups compare
that enum. The reference retains its document key for serialization and arbitrary
variadic ports. Concat's numeric-key parsing remains a document-boundary operation.
Built-in backend selection uses `BackendKind` from `proto/kernel_registry.proto`;
its provider-name projection is shared by admission, execution and diagnostics.
Custom provider names remain strings and cannot shadow any built-in name.
Layout symbols and affine quantization kinds are generated by the parameter
registry and parsed at graph boundaries before typed comparisons.

Node input/output references and typed parameter arrays are owned, variable-size
C allocations. Dependency indices live on the resolved references, not in a
fixed table of ports per node. Graph snapshots clone that ownership; graph edits,
rollback and metadata growth release or transfer it explicitly. Concat's
canonical input order is computed once when the node is loaded or edited.

`DecodeGenerate` retains the row session and runs its token feedback loop in C.
For GPU execution, C encodes the token/keep-mask copies and row dispatches; no
intermediate token readback or TypeScript generation loop occurs. Only the final
owned outputs are snapshotted. Admission validates the entire requested range
and cache capacity before mutation.

`ConfigureDecodeCache` binds internal causal-attention K/V activations to a
context-owned page table. Page allocation, prefix identity, copy-on-write,
lane retirement and generation tags are C state. `PublishDecodePrefix` retains
independent root/output prefix snapshots, while K/V pages share references.
`ReuseDecodePrefix` can populate an empty lane without a full-batch prefill.
`ReleaseDecodeLane`, prefix eviction and reset retire that ownership. Page
reservations roll back without GPU commands when admission fails. CPU and GPU
use the same logical row mapping. Cache state distinguishes resident-page bytes
from actual allocated pool and prefix-snapshot storage; paging does not claim
that an already allocated dense device pool has shrunk.

## Worker batching

`VxSchedulerService` also owns `BatchQueue`, the C replacement for the former
engine-independent BatchScheduler/ContinuousBatchScheduler policy. Workers submit
stateless contributions or prompts through `SubmitBatchWork`. C groups compatible
contributions, chunks prefill by token budget, controls padding and fill-first
waiting, and owns lane/page reservations and prefix reuse. Group components are
compared separately and retained without the former C fixed string lengths.

`NextBatchDispatch` exposes one stable dispatch until `CompleteBatchDispatch`
settles it. Polling does not repeat admission. The generated protocol replaces
an asynchronous JavaScript callback; the application worker executes the batch
and returns its values and named outputs. C copies all accepted bytes, preserves
per-request value history and last outputs, and bounds terminal retention.
`GetBatchWork` copies a retained result; `TakeBatchWork` copies and retires a
terminal result atomically. A cancelled request can be taken while its dispatch
is still in flight; internal work storage remains alive through settlement.
Invalid completion input leaves the dispatch pending. Cancellation cannot reuse
a lane beneath pending work. Close with drain finishes admitted work; close
without drain revokes pending dispatch IDs and cancels work.

A BatchQueue is a policy owner, separate from Runtime's compiled-model submission
coordinator. It holds no device resources, and its page plans describe storage
owned by the worker; they do not implicitly rebind an ExecutionContext cache.
`worker_busy_micros` measures dispatch exposure-to-completion time. It is not a
measurement of GPU hardware occupancy. Product TypeScript contains no scheduler
state machine or worker execution policy.

Full WebGPU training reuses C's backward planner. C owns the tape, optimizer
state, gradient accumulation and completion transaction. GPU shaders perform
forward, cross entropy, backward, accumulation, global norm/clipping and SGD or
AdamW. WGSL compute entry points and binding layouts are generated from source;
the bridge creates explicit device layouts without interpreting an operator.

TrainStep returns a microbatch ID with PENDING on WebGPU and READY on CPU.
GetTrainStep polls the latest accepted ID without submitting it again. A trainer
accepts one pending step; step/commit/rollback/export return BUSY until it ends.
Completion publishes parameters and optimizer state together after successful
readback and graph restoration. Failure restores the committed baseline, or
poisons a trainer if restoration fails. ReleaseTrainer can retire pending work.
The WebGPU training planner uses training shaders for Dropout and attention
forward execution, with the same C RNG parameters as backward. Incomplete
backward/device plans are refused before submission. There is no numerical
fallback during a step.

Other domains return typed refusals. The fixture matrix in
`docs/c-runtime-validation.md` records tested shapes/dtypes and policies;
an operator name alone does not guarantee support.

## Scheduler, planning and memory evidence

The C scheduler owns admission, priorities, deadlines, input snapshots and
result budgets. Direct mode bypasses it. Scheduled mode uses the same monotonic
clock as `VxPlatformService`; a host without that clock cannot create a scheduled
runtime. `PollRequest` observes without starting threadless execution.
`WaitRequest` can start one queued dispatch per polling turn; its call options
bound how long the caller waits. Hard deadlines reclaim queued or pending work; soft
misses are reported without discarding otherwise valid results. LATEST stream
replacement follows the atomic admission contract in the proto.

`VxPlanningService` accepts an exact Model revision or standalone graph bytes and
weight metadata. A typed `GraphDefinition` is an alternative to graph bytes;
C assigns missing node IDs and applies the same semantic validation.
`EditGraphPlan` applies an ordered transaction to a private draft, validates once,
and publishes a new standalone plan. Existing plans never change. `ExportGraphPlan`
returns the logical document, original storage descriptors (including F16) and
resolved quantization values needed to recreate that source.
Invalid definitions return no handle. A well-formed but unproved
domain returns a readable plan with typed refusal evidence. Public plans describe
semantic tensors, operators, parameters, dependencies, independent batching and
weight-bank residency. They never expose private scratch offsets or executable
GPU records. `ResolveGraphPlan` binds explicit exact/minimum dimensions.
`InspectSafetensors` performs metadata-only inspection; inline header prefixes
work in all transports, while path/mapped-view forms are transport-scoped.
`ReadSafetensors` and `WriteSafetensors` are immutable byte operations in both
profiles. Their storage vocabulary includes every SafeTensors dtype, arbitrary
rank, empty tensors and length-aware UTF-8 names/metadata. The existing C header
authority checks both input files and completed writes. Runtime weight limits
apply when storage is loaded as a model, not when it is edited as a file.
F16 widening, zero-filled allocation and tensor replacement execute in C.

The full profile adds `VxTrainingService.InitializeTensor`: zeros, ones, normal,
Xavier uniform and Xavier normal. C implements the former Mulberry32 stream,
UTF-16 string-seed hash and Box-Muller transform. The WASM profile uses the
private C scalar math implementation for double-precision trigonometry.
Inference excludes the initializer handler and its generated message closure.

Memory capture is opt-in through proto options and returns typed ownership and
byte accounting. Process RSS is an optional external envelope, not a substitute
for exact retained resource charges. Runtime reservations cover queued,
in-flight and caller-retained snapshots. Releasing the final owner returns its
charge; a pending batch aggregate remains charged while any lane retains it.

## Source authorities and generation

`proto/volvoxai.proto` is the single application contract for C, TypeScript,
Python, and agent callers. Operations reach generated Synurang dispatch:
native hosts load `Synurang_GetApi`, while web hosts transport calls into their
C/WASM owner. Adding an operation requires a schema change. Internal lifecycle
functions and hand-written public enums do not belong in `native/include/`.
[API discovery](docs/api-discovery.md) and generated references provide exact
field-level contracts; the guides explain how to combine them into workflows.

| Authority | Responsibility |
| --- | --- |
| `proto/volvoxai.proto` | All application operations, messages, enums and reports |
| `proto/operator_vocabulary.proto`, `operator_param_registry.proto` | Operator and parameter vocabulary |
| `proto/kernel_registry.proto`, `optimizer_registry.proto` | Kernel support and optimizer inventory |
| `native/src/api/` | Generated-dispatch handler adapters |
| `native/src/runtime/` | C ownership, semantics, planning and scheduling |
| `native/src/kernels/`, `backends/`, `training/` | Numerical execution and full-only services |
| `tools/wasm_internal_abi_manifest.mjs` | Private WASM transport ABI |
| `shaders/` | Authoritative WGSL; generated catalogues/packs are projections |
| `ts/host/`, `ts/core/` | WASM and protobuf transport |
| `ts/backends/WebGPUHostBridge.ts` | GPU device transport |

Do not hand-edit generated bindings, shader outputs or embedded byte arrays.
Preserve `VOLVOXAI_SHADER_DIR`; log once when an external override is actually
used. Normal builds check generated files and do not regenerate them implicitly.

The Synurang C call runtime is vendored under `native/third_party/synurang/`.
Bindings are committed under `runtime/generated/`. `tools/generate_proto.py`
downloads the official Synurang v0.8.0 release generator and runtime sources
from the matching GitHub tag, commit
`53180b484cf7ca07a1e7d6f24e58b8a19a2dcfa8`. Both archives have pinned SHA-256
digests and are cached under `build/` for offline regeneration. There is no
local source snapshot or Cargo build step. C uses `mode=module`; TypeScript and
Python use `mode=client`. Both TS profiles share one runtime implementation;
full adds only its own services/codecs. Release and schema provenance is
recorded with every projection. Normal builds use committed bindings and
vendored runtime files without network access or an adjacent checkout.

`tools/package_release.py` creates deterministic inference or full runtime ZIPs.
Each package contains that profile's JS, minified JS, WASM and native binary,
with hashes binding every artifact and the generated API/bridge contracts.
Model-specific runtime/shader pruning is deferred and has no product or build
implementation. Planning is available through the common generated service.

## Change and verification rules

Read this document before changing layout, build composition, operators,
backends, training or shader generation. Run the smallest relevant checks while
developing, and inference/full native and WASM builds before handoff. Schema
changes require regenerated C/TypeScript/Python projections and
`make proto_codegen_check`. `make api_conformance` runs inside native builds.

Final checks include typecheck, all JS profiles, proto/bridge and relevant full
service regressions, native invariant/profile checks and `npm run check:release`.
Real WebGPU qualification runs sequentially with Deno on an idle physical GPU.
Record the GPU model, driver and runtime versions, exact artifact hash, scripts
and results without personal account names, hostnames or home-directory paths.
Browser/MV3 and package checks use actual distribution files rather than
source-only mocks.
