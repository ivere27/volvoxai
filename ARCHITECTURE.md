# VolvoxAI architecture

VolvoxAI is an on-device edge engine for web browsers and robots. It supports
both inference and training while keeping their build and ownership boundaries
explicit.
One-shot latency matters, as do bounded multi-client browser serving and
multi-sensor robot workloads under constrained CPU, WASM, GPU, memory, and
energy budgets.

VolvoxAI has two capability profiles:

- inference: forward execution only.
- full: inference plus training and PTQ authoring.

The browser release also has a backend-restricted full composition:
volvoxai.wasm.js contains strict WASM inference and training but omits CPU,
WebGPU, WGSL, and Node filesystem implementations. It is a deployment
composition, not another numerical backend.

Release artifacts are flat. Source code is grouped by ownership.

## Public ownership model

JavaScript inference uses:

~~~text
Runtime
  BackendProvider[] and exact compiled-artifact registry
  one logical execution coordinator (allocated lazily)
  device/resource-domain dispatchers and global bounded admission
  tiny Runtime-wide result count/logical-byte ledger (always present)

Model
  immutable logical definition and bounded shape constraints
  immutable logical topology, tensor shape specifications, fixed weights,
  and exact revision

CompiledModel
  retained Runtime
  exact retained model and weight revision
  immutable provider-prepared execution blueprint, invariant resources,
  bounded-domain proof, batch contract, and report

TargetExecutor                   internal
  retained CompiledModel, one mutable context by default
  bounded context pool only when measured domain parallelism justifies it

BatchRoute                       internal
  opaque provider compatibility identity and immutable prepared shape plan
  bounded frame/workspace accounting; optional sequence store and PagedKV lanes

RuntimeRequestHandle             SCHEDULED
  cancellation and one result promise; no scheduler-owned result archive

ExecutionContext                 low-level/internal migration surface
  retained CompiledModel
  private shape binding, resolved shape-plan cache, capacities, request,
  scratch, output, and decode state
  FIFO operation queue

ExecutionResult
  stable named concrete host/device output snapshots
  closeable Runtime result-budget lease, independent of scheduler lifetime

Trainer                         full profile only
  retained Model and exact base revision
  private gradients, optimizer slots, accumulation, RNG, and working revision

PTQPlan                         full profile only
  retained Model and exact pinned revision
  immutable template/spec and private CPU engine
  private mutable observations and sample registry
~~~

Native inference uses matching opaque `VxRuntime`, `VxModel`,
`VxCompiledModel`, `VxRequest`, `VxExecutionContext`, and `VxResult` handles.
`vx_runtime_run` is the synchronous DIRECT boundary; ordinary concurrent work
uses `vx_runtime_submit` plus poll/wait/cancel/result on the Runtime's lazy
coordinator. External backends create explicit provider-runtime, compiled, and
context instances and may attest one true dense `context_execute_batch`
callback. Built-in Vulkan, OpenGL, and CUDA routes use the same core proof and
enter one authored symbolic-B engine forward per selected stateless group.
Explicit contexts remain the low-level stateful boundary. Native
per-resource-domain asynchronous completion/fence dispatch is still tracked in
the [scheduling and dynamic batching design](docs/scheduling-and-dynamic-batching-design.md);
the first request implementation uses one bounded worker.

Close has one ownership meaning with language-idiomatic completion. Starting
close rejects new root work and drains accepted coordinator/DIRECT work;
published results remain valid until explicitly released. The JavaScript
`Runtime.close()` promise additionally denotes physical provider teardown and
waits for retained compiled/context children. Native `vx_runtime_close()` is
the logical boundary; opaque children pin physical state until the final
`vx_runtime_release()`.

Physical devices, queues, allocators and submission synchronization are shared
by a resource domain. Exact compiled artifacts additionally share immutable
topology, packed weights, plans and pipelines. Request data, outputs, mutable
workspaces, decode/KV state, gradients, optimizer state and accumulation always
have a named owner and never cross a compatibility route implicitly.

Full-profile native training may retain private optimizer-backed tensor
capacity without widening the portable graph. Persisted LayerNorm/RMSNorm
affine descriptors remain exactly the normalized feature width; a private
Trainer's larger rank-one storage exposes only that active prefix to the
normalization kernel and leaves its capacity tail untouched. CUDA AdamW moments
remain device-authoritative until an explicit public boundary. Switching that
same private graph to a CPU update or saving optimizer state first materializes
the current moment mirrors and retires them before host mutation, regardless of
the currently selected backend flag; a state with no CUDA context takes a
side-effect-free no-mirror path.

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

The CPU resident proof includes the immutable snapshot and the compiled
model's separate invariant-weight materialization. Both revisions are retained
once per compiled artifact rather than once per context. Its one-context
ordinary peak is the maximum of
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
buffer/binding/dispatch limits before late binding. Linear-index workloads in
the qualified elementwise, copy, expand, transpose, quantize/dequantize and
scalar QConv routes use a bounded two-dimensional workgroup grid; the shader
reconstructs the same linear index and the physical-domain proof mirrors the
compiler's grid and fallback choice. This removes the single-axis workgroup
ceiling without weakening buffer or binding limits.

A native Vulkan graph context allocates its compute arena only from a
`DEVICE_LOCAL` memory type. Context creation fails closed if that placement or
allocation is unavailable; it never silently substitutes a host/system-memory
compute arena. Host ingress and result publication use a separate bounded
32-MiB `HOST_VISIBLE` staging ring. Coherent memory is preferred; otherwise
flush/invalidate ranges are aligned to `nonCoherentAtomSize`, and a staging
range is not reused until its queue fence completes. The bounded-domain
resident proof charges the actual compute-arena and staging allocations, while
live execution evidence reports their placement and successful per-forward
upload/download operations. A bounded dynamic CUDA context may retain at most
four exact-ShapeSignature replay plans in a deterministic context-local LRU;
every plan is additionally keyed by model generation, graph-slot epoch,
capacity generation, and domain mode, and any global-key mismatch destroys all
retained executables before direct fallback. Its resident proof charges fixed
host plan metadata plus four retained maximum-size owned signatures and one
transactional candidate signature. Opaque CUDA-driver storage for the
at-most-four requested GraphExec objects is documented as requested/unknown
rather than included in the numeric resident claim; compile evidence reports
the exact cache capacity, not an invented byte count. A Driver
destroy failure quarantines the owning cache entry; the engine performs direct
launches and retries destruction before reusing that entry or requesting
another GraphExec, so the four-object bound is never exceeded.

WebGPU resource creation is fail-closed. Tensor-generation, specialization,
result-snapshot and readback-staging allocations are enclosed by a nested
validation/out-of-memory scope pair whose pushes and pops occur in one
synchronous turn; candidate state is published only after the popped promises
settle. This avoids interleaving the device-global scope stack across an
`await`. Already-preflighted steady input uploads, command encoding and queue
submission are not given per-execution scope round trips. Runtime selection
rejects a provider before context mutation whenever it cannot attest the
snapshot's complete declared domain.

The allocation scope is a runtime boundary in addition to static API-limit
proof. On Deno 2.9.3/wgpu/Vulkan with NVIDIA driver 535, an isolated 104-MiB
buffer succeeds while the exact 105-MiB `Tensor_v1` allocation fails three out
of three times, despite advertised 128-MiB storage-binding and 256-MiB buffer
limits. The public execution reports `OUT_OF_MEMORY` and publishes no invalid or
corrupted result. This observed advertised-versus-effective allocator boundary
must not be relabeled as exhaustion of the RTX 3090's 24-GiB VRAM.

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

## Serving many requests: batching, paged KV, scheduling

The bounded domain above says what one execution may bind. This section says
how many executions share a device, and it is where the throughput of a
deployment is decided.

### Two kinds of B>1, and they are not interchangeable

The batch axis is spelled the same way in both, which is exactly why they have
to be named apart.

**Bulk execution** binds an explicitly declared batch symbol and runs
`execute({x: [B,…]})` as one provider invocation. Here B is caller-authored
model semantics: an operation may intentionally reduce or normalize across B,
and the Runtime returns the graph's outputs without splitting them into
independent requests. The complete bounded-domain proof and backend
qualification must cover that B range, but independent-row evidence is not
required.

**Scheduler coalescing** is the stricter bulk subset that stacks N independent
B=1 requests, invokes one `[B,…]` graph, and splits its outputs back into N
results. It additionally requires the exact typed independent-batch proof and
provider single-invocation attestation. A graph that mixes lanes may still
support caller-authored bulk execution, but remains B=1 for scheduler
coalescing. For fixed-B=1 packages, coalescing requires a typed-IR compiler
transform before shape resolution; blindly prepending an axis to a resolved
graph is forbidden.

**Row decode** advances one token for each of several resident sequences. Rows
are addressed rather than reshaped, so it is not the bulk path with a different
extent: the operators that participate are an explicit admit list
(`vx_runtime_node_decode_batch_supported`), because an untransformed row path
computes one contiguous span and would hand every lane lane 0's row.

A consequence worth stating plainly: **lane count is declared, not inferred.**
`[B,S,D]` and `[S,1,D]` are indistinguishable as shapes, so the number of lanes
a step touches is answered in one place and consulted, never re-derived from a
sample. A provider that disagrees with the declaration fails at context
creation rather than at the step that needs it.

### Paged KV

Retained attention state is paged: a lane's logical token positions map to
physical pages through a per-lane page table, so a sequence grows without
reserving its maximum length contiguously and two sequences can share the pages
of a common prefix. Ownership is deliberately route/session-local — no page
moves between incompatible caches, models or device epochs.

Two rules make the sharing safe rather than merely cheap:

- A published prefix ends on a page boundary. A partial tail page is still
  being appended to, and publishing it would hand another request bytes that
  are about to change.
- A prefix identity must cover everything that changes the K/V it stands for —
  model and weight revision, adapter revisions, tokenizer semantics,
  quantization, prompt tokens, mask and position semantics. The Runtime/provider
  constructs this identity structurally; an application string is metadata,
  never the cache authority.

Retirement bumps a lane generation, so work submitted against a previous
occupant can never be applied to its replacement.

### Admission and dispatch

Each Runtime owns exactly one logical coordinator. It globally arbitrates all
models in that Runtime, while resource-domain dispatchers may progress
independently. A compiled target reuses one mutable context by default across
its prepared shape routes; it grows a bounded pool only where measured domain
parallelism pays for the extra activation/heap state. Sequence/KV state remains
route/session-local. Different models share the global budget and device queue,
never a cache or physical batch.

The v1 surface has two execution modes. `ExecutionMode` is defined in
`proto/volvoxai.proto`; generated TypeScript and native bindings are the only
code-level enum authorities:

- `DIRECT` acquires a lightweight route lease and invokes one logical call
  directly, including a caller-authored bulk B=N tensor. It creates no Runtime
  coordinator, queue, timer, worker, request table or scheduler telemetry ring.
  It is an explicit global-arbitration opt-out; different compiled routes may
  progress concurrently, so concurrent multi-model applications use
  `SCHEDULED`. DIRECT does not promise zero allocation, synchronous completion
  in JavaScript, B=1, or bypass of a provider/device queue. A WebGPU result still
  owns an immutable snapshot, so its buffer allocation pays one validation/OOM
  scope pair before publication; that provider safety boundary is not a
  scheduler allocation.
- `SCHEDULED` creates the coordinator lazily and always uses bounded admission,
  owned or leased queued inputs, priority/EDF/aging fairness, deadlines, and
  stateless freshness. `maxBatchDelayMs=0` dispatches work-conservingly without
  a speculative coalescing wait; a positive value permits a bounded window that
  deadlines may shorten. Consume/ack channels, physical device/workspace
  accounting, stateful sessions and recovery are promotion work for browser
  extensions and robots, not another execution mode.

Scheduling is by **dispatch shape, not request lifetime**. The compatibility
identity is a structural tuple produced inside Runtime/provider code:

~~~text
(resource_domain, provider structural/opaque executable token, compiled+adapter revision,
 canonical shape/layout/dtype/tactic without B, phase/query layout, device epoch)
~~~

Tuple components are not delimiter-concatenated caller strings. Prefill and
decode remain separate phases until a flat-packed query contract proves their
mixed layout. The throughput bound is rows/tokens and bytes, not merely request
count; padding may never exceed batch, row, token or memory limits.

Static-legal and operating batch sizes are different facts. For one
model/provider/device revision, the static set is the intersection of the
producer batch domain, typed independence proof, kernel domain, and advertised
physical buffer/binding/dispatch limits. Observed allocator behavior can shrink
the executable set further. The target selector chooses an operating B only
from that set using a measured bounded cost curve `T(B)`, deadline slack,
energy, memory pressure and contention. The profitable B may be lower than the
static legal maximum and can change by runtime, driver, thermal state or
workload; a GPU warp or subgroup size is not a universal batch ceiling.

The current slice uses bounded batch/window rules, priority/EDF/aging (native
and TypeScript) plus bounded route-bypass fairness in TypeScript; it does not
yet autotune `T(B)` or reserve a whole-device resident budget across compiled
weights, activation/workspace maxima, aligned buffers, in-flight results and KV
pages. CPU and WASM can execute an attested explicit-axis B>1 graph in one
provider invocation, but a legal batch is not necessarily faster. Production
promotion requires the measured selector to keep a route at B=1 when B>1 shows
no throughput/energy benefit; until that selector exists, operators must cap
such a route explicitly. No architecture rule statically forces either outcome.

Current device/model evidence illustrates the distinction. With the advertised
RTX 3090 WebGPU limits, the receipt-reader graph is statically legal through
B=19, while B=20 is rejected when an F32 `[B,160,336,32]` intermediate exceeds
the 128-MiB storage-buffer binding limit. In the measured Deno/wgpu process,
FP32 and INT8 B15 both execute and improve useful throughput by 6.39x and 5.67x
over their same-route DIRECT B1 baselines; B16 reaches the effective allocator
boundary above and fails closed. Thus B19 is the static ceiling and B15 the
observed operating boundary, not a universal RTX maximum. Tiny Receipt VQA is
producer-capped at B=8, and its four FP32/INT8 encoder/decoder component routes
execute there with 1.74x–4.05x gains over independent B1 groups. These measured
points inform a future route-specific `T(B)` table; the current scheduler did
not autotune them.

On native CUDA, an explicitly authored receipt-reader B=4 graph is
byte-identical to four same-CUDA B1 lanes and yields about 1.30x useful
throughput on that RTX 3090. The native Vulkan, OpenGL, and CUDA built-ins now
promote only authored symbolic leading-B graphs that pass the graph-bound typed
proof and the complete backend domain proof. The coordinator stacks compatible
B1 inputs, performs one engine forward at B=N, and splits one immutable output
snapshot into owned lane results. CPU and Metal built-ins remain scheduler B=1;
fixed-B1 lifting and completion-driven per-domain GPU dispatch remain open.

Promotion reporting follows the same discipline: `steps` (work asked for) and
`dispatches` (backend entries) are reported *together*, because their ratio is
the batching win and a scheduler reporting only one could claim it without
having produced it. `device_busy` and `wall` come from one real clock — feeding
a virtual time into one side of that ratio produces a number that looks like a
measurement and is not one. Current per-result dispatch identity and focused
test counters prove the first physical batch path; public aggregate
device-busy/wall telemetry remains open in `TODO.md`.

### Invariant weights belong to the compiled model

Weights are immutable after compilation, so the target ownership rule is one
shared invariant representation for every context over one revision:
`weights + N × activations`, never `N × (weights + activations)`. This is a
correctness-shaped memory rule rather than an optimization — VRAM is the ceiling
on how many requests can be resident, so a weight copy per context lowers the
batch size that concurrency was opened to raise. The exact implemented scopes
and remaining native derived representations are stated below.

Sharing is safe because nothing writes to these buffers during execution: a
single context already reuses one buffer across every execution and replan. On
a device the lifetime is explicit, so the sharing is reference-counted — a
context that closes releases its reference rather than destroying storage its
siblings are still bound to.

The rule is mandatory in the JavaScript provider SPI rather than a provider
convention. Every compiled provider object exposes one exact invariant-resource
owner. Core captures its frozen owner identity and device epoch once, opens one
counted lease for each context, and passes only that lease through
`BackendProviderContextOptions`. The frozen lease can borrow a resource by name
through a provider-specific lookup but cannot define, replace, or dispose one.
Reaching zero borrowers retains the
materialized resources, so closing every context and reopening one does not
copy or upload weights again. Compiled close disposes the owner after all
leases drain. An epoch invalidation is terminal for that owner: it disposes its
resources and rejects both stale leases and new opens.

The built-in JavaScript providers implement that boundary at these exact
physical scopes:

- CPU JS lazily clones each immutable host weight into the compiled owner once.
  All contexts borrow the same typed arrays; activation arenas, decode state,
  and shape bindings remain private. The immutable `Model` payload and this
  compiled materialization explain the `2*weights` term in the one-context
  resource proof, but additional contexts do not add another weight pair.
- WebGPU compilation eagerly owns one host copy of every weight and one aligned
  `GPUBuffer` for each non-banked fixed weight. Allocation and upload are
  fail-closed under the compilation error scopes. Contexts borrow those buffers
  while selected bank slices remain context-private. Closing all contexts retains the buffers;
  compiled close destroys them. `device.lost` invalidates the compiled epoch
  and destroys the owner buffers once. Automatic device recreation and request
  recovery are not implemented.
- The production WASM provider has one `WebAssembly.Memory` root. Each compiled
  model owns one host-weight aggregate and one immutable linear-memory prefix
  containing its raw weights and F32/Q8 packed panels. Contexts borrow that
  prefix and reserve disjoint bounded mutable regions; a partial bank selection
  alone is copied into its context's mutable region. Closing all contexts keeps
  the prefix and its copy/pack counts unchanged until compiled close.

Native built-in compilation now reads and parses each accepted safetensors
shard once into a reference-counted `VxCompiledWeightStore` (separate from the
earlier Model-load validation reads). Compile validation and every
context borrow its immutable blob and metadata; each context clones only the
tensor descriptor table. Selected bank rows use a context-owned copy-on-write
overlay, so one route cannot mutate the compiled blob or a sibling route.
Closing every context retains the store and reopening performs no file read.
Training authoring continues to load a private writable revision.

That native change closes the file-read/blob duplication bug, not all prepared
resource ownership. Native CPU F16 widening and prepacked weight caches are
still context-owned, as are selected-bank overlays. Built-in native GPU graph
and device resources are also still prepared per context. Moving the immutable
parts of those resources to `VxCompiledModel`, bounding the legitimate mutable
overlays, and measuring aggregate native RSS/physical-device high water remain
active in [`TODO.md`](TODO.md).

## Source map

TypeScript:

~~~text
ts/
  core/       graph/data objects, loading, snapshots, runtime handles, results
              RuntimeScheduler, BatchScheduler, ContinuousBatchScheduler,
              PagedKVCache; only Runtime owns public serving coordination
  ops/        reusable operators, validation, and normalization
  backends/   provider SPI plus CPU, WASM, and WebGPU resources
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
                  paged_kv, batch_scheduler, continuous_batch_scheduler,
                  decode_row_set
  src/kernels/    portable and optimized CPU/WASM kernels
  src/backends/   Vulkan, OpenGL, CUDA, and Metal integrations
  src/training/   full-profile backward, optimizer, and PTQ implementation
  src/shader_store.*  embedded shader lookup and development override
  src/tokenization/  opt-in application tokenizer
  cli/            fixed model-agnostic command application
  tests/          native unit and integration tests
  third_party/    vendored dependencies and provenance
~~~

`proto/volvoxai.proto` is the sole public contract for the optional in-process
Synurang FFI plugin under `runtime/` and the authoritative shared vocabulary for
logical `DataType`, `OperatorKind`, and `ExecutionMode` values. It declares every
application-facing inference, Trainer, and PTQ operation plus typed statuses,
stages, policies, and tensor metadata. Internal kernel and optimizer registries
import this vocabulary instead of copying it into JSON or handwritten tables.
Generated C, TypeScript, Rust, and exporter projections are canonical. The
native C lifecycle is the implementation layer behind those handlers, not a
second FFI contract.

Semantic validation around those generated messages is handwritten and kept
outside generated directories. `runtime/src/memory_evidence.rs` validates
typed memory evidence before a successful `OperationReport` crosses the
Rust/Synurang boundary. TypeScript consumers construct the generated
`RuntimeServiceFfi` with the handwritten
`runtime/typescript/MemoryEvidenceValidatingPluginHost.ts` decorator. For every
report-bearing unary method, that decorator caps the raw response before
decoding, recursively validates direct and nested `OperationReport` messages,
and then forwards a stable byte-for-byte snapshot so the validation decode and
generated client consume the same response while the decorator preserves
unknown protobuf wire fields. The generated client is not itself this semantic
validation boundary.

`MemoryCaptureOptions` is the explicit runtime-scoped opt-in for producing this
evidence through the Synurang lifecycle. Absence leaves every report unchanged.
The first native adapter accepts boundary-only best-effort capture: it samples
current process RSS and process-lifetime peak RSS through the separate
`VxProcessMemorySampleV1` ABI, emits one `AFTER` snapshot, and represents every
requested but unsupported envelope as `UNAVAILABLE`. Periodic options are
validated but rejected explicitly until an operation-window sampler exists;
they are never silently treated as boundary samples. Native resource inventory
and domain-attestation collectors are not implemented yet, so their requested
result remains an empty `PARTIAL` inventory and an absent attestation. The
legacy `VxReport.allocated_bytes` estimate is never promoted into typed memory
evidence.

The JavaScript package runtime implements the same opt-in contract directly at
`RuntimeOptions.memoryCapture`. Its public evidence DTO follows the protobuf
field model but represents uint64 values as safe-integer numbers so frozen
reports remain JSON-serializable; an explicit protobuf boundary performs any
number-to-bigint promotion. Successful compilation projects the already-checked
bounded-domain provider attestation into coarse, non-observed bounds. Execution
and decode publish one boundary snapshot after the `ExecutionResult` has taken
or cloned output ownership, and failures publish one pre-cleanup `FAILURE`
snapshot. Node process/runtime counters provide RSS, managed-heap, external,
and ArrayBuffer envelopes when available; every unsupported request remains a
typed `UNAVAILABLE` value. The browser `performance.memory` managed-heap
fallback is quantized rather than live, so it carries its own sampler identity
and an `ESTIMATED` relation instead of borrowing the exact Node counter's.

Provider-owned resources cross a separate optional
`volvoxai-backend-memory-snapshot/v1` SPI. The WASM provider exposes exactly one
independent `WASM_LINEAR` capacity root for its provider memory generation.
Compiled raw/packed prefixes and context-mutable arenas are non-overlapping
suballocations of that root rather than additive roots. The inventory remains
`PARTIAL` because the compiled JavaScript host-weight aggregate and result
snapshots sit outside linear memory. WebGPU buffer sizes and native GPU
allocation requests must remain `API_REQUEST`/`REQUESTED`; no browser-safe
physical VRAM sampler exists, so those quantities are never inferred from API
sizes or process RSS.

The capture policy follows the native lifecycle rather than the public Runtime
handle. Each Model, CompiledModel, Context, Result, Trainer, and PTQPlan wrapper,
plus each in-flight retained call, holds a capture lease. Releasing the Runtime
handle therefore cannot disable evidence for a surviving child or race a child
report. Native numeric owner identities retain canonical decimal spelling;
Trainer and PTQPlan use their opaque Synurang handle identities because the v1
native report has no numeric lineage fields for them.

The native process sampler is a read-only inference-safe API that is deliberately
separate from the exact-size `VxReport`, `VxRuntimeOptions`, and backend-provider
v1 structs. Package entries under `ts/` still do not import the full FFI codec,
the decorator, or its validator. The direct TypeScript collector uses
inference-only generated enums and handwritten JSON-safe DTOs, which preserves
the inference/training composition cut while later CPU, WebGPU, and native GPU
resource collectors use the same protobuf model.

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
- Runtime: root owner for initialized providers, one lazy execution coordinator,
  compiled routes and resource-domain admission.
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
- TargetExecutor: the exact compiled artifact's mutable execution lease; one
  context by default, or a measured and explicitly bounded pool.
- BatchRoute: one exact compatibility identity and immutable prepared shape
  plan, with bounded frame accounting and optional session/KV state.
- ExecutionContext: low-level mutable shape-binding/capacity/decode owner used
  inside a route during the serving-API cutover.
- ExecutionResult: stable named concrete output snapshots.
- Trainer: full-profile private gradients, optimizer, accumulation, working
  revision, explicit commit, and rollback.
- Bulk execution: one dispatch over a declared batch axis. Every kernel
  supports it; the axis is ordinary shape.
- Row decode: one token advanced for several resident sequences, addressed by
  row rather than reshaped. Restricted to an explicit admit list.
- Lane: one resident sequence's slot in a row-decode step. The count is
  declared, never inferred from a shape.
- Paged KV cache: retained attention state mapped from logical token positions
  to physical pages per lane, so sequences grow without contiguous reservation
  and a shared prefix exists once.
- Shared prefix: resident pages bound into a lane instead of being recomputed,
  keyed by a Runtime/provider-built structural identity covering everything that
  changes the K/V; caller strings are metadata only.
- Contribution: one request's share of one dispatch — its rows, its kind, and
  its group key.
- Compatibility key: a structural, Runtime/provider-owned tuple containing
  resource domain, provider structural/opaque executable identity, exact revisions, canonical
  geometry/phase and epoch. Contributions that cannot stack into one `[B,…]`
  never land in one dispatch.
- Token budget: the per-dispatch throughput limit, in tokens rather than
  requests, so prefill chunks and decode tokens compare in one unit.

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

- Runtime close stops the coordinator first: reject admission, settle/cancel
  queued requests by policy, await submitted fences, close routes, then close
  providers.
- DIRECT holds only a route lease. A busy route returns `BUSY`; there is no
  hybrid fallback into SCHEDULED admission. A caller that wants admission makes
  a separate SCHEDULED call.
- Scheduled inputs are owned copies, transfers, or counted leases. A borrowed
  caller buffer never outlives the call that supplied it.
- Cancellation after submission suppresses publication but never permits lane,
  page or buffer reuse before completion. Device/route epochs reject stale
  completions after device loss.
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

The WASM compiled model owns one frozen, pointer-free operator schedule, one
host-weight aggregate, and one immutable raw/packed linear-memory prefix derived
from the logical topology and exact weight revision. Core shape binding supplies
each provider context a metadata-only canonical plan at the smallest legal
public shape, rounded for every `multiple_of` constraint and normalized with
that context's bank residency. WASM instantiates its schedule against this plan
during context creation, before request timing, without synthesizing inputs or
dispatching an operator. The first real request therefore reuses the prepared
shape or performs the ordinary transactional rebind; shape changes rebuild
descriptors and offsets, not topology, invariant packs, or kernel routing. A
failed eager specialization releases only its candidate mutable region and
never publishes a context.

Production contexts use one provider-owned `WebAssembly.Memory` and synchronous
pointer-only kernel instance. A shared JavaScript allocator gives every
compiled model a non-overlapping invariant prefix and every context a disjoint,
bounded mutable range. The module-global C allocator is sealed after this pool
is activated; direct low-level/full-profile callers that require mutable module
state continue to use independent `fork()` memories. Raw weights and eligible
F32/Q8 panels are copied or packed once into the compiled prefix. Full bank
residency borrows it, while a partial bank selection stages only the selected
slice and its derived pack in the context range. A released range is zeroed
before free-list reuse.

Within a context range, `heap_mark` records the start of one
activation/metadata/scratch suffix, and `heap_rewind` may move that context's
cursor only to an aligned address in the same range. A rebind first dry-runs the
complete activation layout, metadata, and maximum scratch allocation, checks
signed kernel arguments and wasm32 limits, and pre-grows the shared memory. It
then rewinds and commits the measured layout. All sibling engines refresh their
typed-array views after a memory growth. If measurement or commit fails, the
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
their allocation rule is audited. Telemetry names the one shared root, each
compiled prefix suballocation, and each context-mutable suballocation; it never
reports the same root capacity once per arena.

WASM's one-compiled/one-context resident proof also covers provider- and
result-owned host payloads. Let `L0` be the initial shared linear-memory extent,
`L` its proved extent for that compiled prefix plus one maximum mutable region,
`P` the context host plan-cache budget, `W` the complete raw weight revision,
`B` the raw bytes of all banked weights, and `R` the sum of maximum-domain
public-output bytes. The immutable `Model` and the compiled host-weight
aggregate contribute `2*W`; raw/packed linear weights are already in `L`.
Explicitly selecting all slots of every bank is legal and retains one additional
host `B`, while each result is cloned once from its linear-memory view and
transferred without another copy.
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
then rejected before its WASM engine is forked. Aggregate admission for several
compiled prefixes and context ranges sharing the root is the whole-device
budget tracked in `TODO.md`; this per-context proof does not claim to implement
that coordinator.

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
