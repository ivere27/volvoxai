# Scheduling and dynamic batching

A camera may produce frames faster than a model can consume them; a chat
application may need to advance several conversations without mixing their
state. Scheduling decides which request runs next, batching combines compatible
work, and decode contexts retain each sequence's state between steps.

Use direct execution for an isolated request. Use the Runtime scheduler when
several producers share a compiled model, and use a BatchQueue when your own
worker needs to control execution. This guide explains the tradeoffs, budgets,
and ownership behind those choices.

[ARCHITECTURE.md](../ARCHITECTURE.md) explains the execution model;
[volvoxai.proto](../proto/volvoxai.proto) gives exact request fields and defaults.
[TODO.md](../TODO.md) tracks further qualification and optional extensions.

Both inference and full expose the generated Inference and Scheduler services.
TypeScript serializes calls and transports device commands. C owns admission,
grouping, row/cache state, execution, and result lifetime. The full Trainer owns
its own optimizer, gradients, RNG, and mutable weights; Runtime scheduling does
not combine independent training steps.

## Three execution owners

| Owner | Public operations | Responsibility |
| --- | --- | --- |
| Runtime request coordinator | Submit, PollRequest, WaitRequest, CancelRequest, TakeRequestResult | Reserve contexts, admit typed execution requests, and batch compatible stateless work |
| ExecutionContext | Execute, Generate, context cache operations | Own mutable bindings, sequence rows, KV storage, and execution state |
| BatchQueue | SubmitBatchWork, NextBatchDispatch, CompleteBatchDispatch, GetBatchWork, TakeBatchWork | Group application work and return dispatches for a worker to execute |

A BatchQueue is an independent C policy owner. It holds copied application
inputs, work state, and page/slot reservations, but no compiled model or GPU
tensor. Its page plans describe worker-owned storage; they do not automatically
bind an ExecutionContext cache. An application worker can execute a returned
dispatch through generated Inference calls and report its outcome.

This separation lets applications use batching policy without giving the queue
ownership of arbitrary execution state. Integrating more execution resources
requires a measured use case and an explicit C ownership contract.

## Compiled resources, contexts, and results

Models and compiled targets retain exact graph, weight, and adapter revisions.
Contexts reference their CompiledModel and own mutable inputs, arenas, decode
state, overlays, and commands. Raw weight storage and CPU prepared weights have
a shared compiled owner. Built-in native GPU resources still require ownership
and residency measurements across multiple contexts; current WebGPU spans are
context-owned. Physical device allocation sharing must be demonstrated for the
particular backend.

Direct and scheduled calls use explicit contexts. One context admits one call
at a time; separate contexts retain independent bindings and state while sharing
a compiled model. Every compile pins exact graph, weight, and optional adapter
revision. Changing an adapter requires a new compilation and context.

Completed packets carry immutable tensors. Inline bytes belong to the caller;
owned BufferViews retain engine storage until `ReleaseBuffers`. Pending device
work and retained buffers keep their budget charges. Closing a web host drains
submitted work before reclaiming referenced GPU storage.

## Direct and scheduled requests

Omitting CreateRuntimeRequest.execution_mode selects DIRECT. SCHEDULED enables
Submit; both runtime modes allow direct Execute/Generate.

| Path | Behavior |
| --- | --- |
| Execute | Runs a named Graph entrypoint and declared scope on its context. |
| Generate | Continues the context's admitted autoregressive ID loop for one chunk. |
| Submit | Carries either request, reserves its context, copies input values and options, and returns a Request. |

Execute/Generate resolve with completed tensors. A scheduled Request owns its
packet until `TakeRequestResult` transfers it once. A second take fails with
NOT_FOUND. Releasing the Request frees an untaken packet; transferred owned
buffers remain valid. Output retention is independent of the context, which can
accept its next call once the execution settles.

A direct call can use a legal B=N tensor. This is one logical execution; it does
not automatically authorize scheduler batching of independent requests.

RuntimeBudget sets scheduled request/input limits, result count/byte limits,
and max_batch_delay_ns. Defaults are declared in the proto. A zero
delay requests immediate dispatch; a positive value bounds coalescing wait and
can be shortened by a deadline.

SubmitOptions provides priority, an absolute monotonic deadline, freshness,
and the stream key used by LATEST. C clamps priority, applies aging, then orders
equal effective priorities by deadline and request ID.

- ALL keeps admitted work eligible until completion or cancellation.
- LATEST replaces work with the same compiled route and nonzero stream key.
  Queued replacement is transactional; submitted work retains its resources
  while its logical publication is suppressed.
- DROP_IF_LATE treats the deadline as a hard queued/completion cutoff.
  ALL and LATEST can still publish successful work after recording a missed
  deadline.

ALL supports both Graph and generation requests. LATEST and DROP_IF_LATE require
stateless EVALUATE/full calls with all-fresh input values; reuse, feedback,
initialization, transitions, and Generate are rejected before replacing any work.
A same-context resubmission is BUSY, so independent LATEST frames use separate
contexts sharing the compiled route. Completion-time disposal invalidates that
execution's retained bindings. Later request retention expiry does not affect
subsequent context calls.

The scheduler deadline controls ordering and freshness. An ALL deadline miss is
reported without discarding successful output. A positive execution budget
deadline is a separate absolute monotonic nanosecond limit: before commit it
preserves READY and bindings; after commit it causes terminal FAILED without
output, after submitted work drains.

The native Runtime currently has one worker. Threadless WASM uses the same C
coordinator cooperatively: PollRequest and finite WaitRequest calls do not
execute queued work; an absent or UINT64_MAX timeout pumps it. The TypeScript
transport does not supply a separate scheduler or timer-driven executor.

## Physical batching and compatibility

Caller-authored bulk B=N can intentionally mix information between lanes.
Scheduler coalescing combines independent B=1 requests and therefore requires
C's typed independent-batch proof plus a backend route that executes the
aggregate graph once. The resulting outputs are split into stable lane packets and installed in each
context without another numerical invocation. The first context batch profile
requires complete stateless EVALUATE/full requests, all declared outputs, and
inline storage. Partial/empty selections and owned-buffer requests run separately.

Runtime compatibility includes the exact compiled target and compatible
non-batch shapes. The native provider SPI also binds its batch attestation to
the graph fingerprint, proof identity, resource domain, and executable token.
A provider can narrow a proved batch domain; it cannot replace a missing C
proof with its own assertion.

The built-in coalescing routes cover admitted WebGPU, Vulkan, OpenGL, and CUDA
graphs. CPU and Metal currently remain scheduler B=1. This describes route
availability; physical qualification still names the tested graph, dtype,
shape, device, and driver. See [runtime validation](c-runtime-validation.md).

BatchQueue uses the separate, application-supplied BatchGroup fields: model,
optional adapter_revision, and shape_signature. C compares each field
separately. A matching group is a scheduling decision; the worker remains
responsible for choosing an executable model/context with a valid tensor
contract.

## Legal B, operating B, and padding

The legal Runtime batch domain is bounded by the authored graph, C's
independence/shape proofs, backend support, and resource limits. A larger legal
batch can be slower than B=1. Current selection follows legal queue and budget
limits; it does not implement a measured T(B) autotuner or thermal epochs.

The Runtime coordinator selects legal multiples and leaves a remainder queued;
it does not synthesize padding. BatchQueue has its own multiple_of policy for
stateless work and can duplicate a valid item to fill a dispatch. Padding has
work_id zero, consumes the dispatch budget, and never publishes a user outcome.
Decode slots refer to real reserved sequence state.

Changes to batch/page sizes or queue structures should follow current workload
measurements of throughput, latency, and memory. Whole-device concurrency,
ready heaps, slabs, and mixed prefill/decode are possible extensions, not
prerequisites for the existing queue API.

## Admission and memory accounting

Runtime scheduled admission validates tensor metadata and shapes, reserves
request/input and result capacity, copies payloads, and publishes the request.
Known budget shortages return OVERLOADED before execution. A failed queued
LATEST replacement leaves its predecessor intact.

Result capacity is charged before numerical execution and shrinks to the owned
snapshot size after a validated success. Coalesced lane results keep a shared
aggregate charge while any lane still needs it. Logical cancellation does not
permit early reuse of resources referenced by submitted GPU work.

Current accounting includes Runtime request/result reservations and compiled
resource estimates. Depending on the backend, these estimates include weights,
alignment, fixed allocations, workspace, and snapshots. ContextCacheState
separately reports page use, reserved bytes, allocated pool bytes, and prefix
snapshot bytes. A free page remains part of its allocated pool.

These mechanisms do not establish one admission ledger for every allocation
sharing a physical GPU. Such a ledger must count shared resources once, retain
charges through completion, and distinguish engine-owned allocation bytes from
unobservable driver overhead. The remaining work is specified in TODO.md.

## BatchQueue and worker dispatch

CreateBatchQueue configures queue depth, token budget, maximum slots,
stateless divisibility, fill policy, retained-result capacity, and optional
decode-cache metadata. The queue has work-conserving and fill-first policies;
max_wait_ns bounds the latter.

SubmitBatchWork copies its payload and inputs. Decode work also declares prompt
and maximum generated-token counts. C reserves slots/pages, chunks prompt work,
groups ready contributions, and returns one stable BatchDispatch.

The worker follows this sequence:

1. Call NextBatchDispatch. A zero dispatch_id means no work is ready.
2. Execute a nonzero dispatch once using the selected worker.
3. Call CompleteBatchDispatch with that ID and its outcomes.
4. Retrieve terminal work through GetBatchWork or TakeBatchWork.

Repeated NextBatchDispatch returns the same pending dispatch. Malformed
completion leaves it pending. An execution error fails that dispatch while
other groups can continue. C owns the copied result history and terminal
retention; TakeBatchWork copies and retires one terminal result atomically.

CancelBatchWork can retire the logical request while a worker still holds a
dispatch. Its slot cannot be reused underneath that work. CloseBatchQueue with
drain finishes admitted work; without drain it cancels work and invalidates
pending dispatch IDs. Worker/device resource cleanup remains with the worker.

## Graph state, generation, and paged KV

The model declares initialization and transition entrypoints. The Graph caller
passes explicit active-length/position bindings and chooses a permitted full,
prefix, slot-row, or dependency scope. A slot row carries a position, recompute,
or empty action. Input omission never implicitly selects reuse: `InputBindings`
states each fresh value, retained input, or output feedback source.

Cache options belong to context creation. `GetExecutionContextInfo` reports
active lengths, empty slots, cache generation, and optional page telemetry.
`PublishContextPrefix` retains a page-aligned prefix with copy-on-write;
`ReuseContextPrefix` fills an empty slot. `ReleaseContextSlot` and
`EvictContextPrefixes` retire ownership. Failed admission rolls back reservations.

A supported autoregressive context uses the model's ArgMax/QArgMax feedback
contract. Its first Generate receives initial inputs and returns N newly selected
IDs using prefill plus N−1 feedback steps. Later chunks consume the pending ID
and perform N steps. Intermediate IDs stay on the device when the backend admits
that route; only the selected final chunk is delivered. The first profile is
batch one, GREEDY, fixed-count generation. Caller-controlled sampling and EOS use
Graph Execute. A terminal context is replaced for the next job.

BatchQueue's prompt chunks already exist as worker dispatches. Efficient
per-slot query/key kernel inputs and mixed or flat-packed execution are
additional numerical contracts, not missing queue policy.

## Failure, completion, and device lifetime

Direct Execute/Generate return after device completion, including an empty output
selection. WebGPU completion is asynchronous and does not block the JS event
loop. Readback or device-loss failures reject with structured report evidence.

Pre-commit validation and execution-budget refusal preserve bindings and READY.
Post-commit failure ends the context as FAILED and publishes no output. Cancelling
a queued Request releases its reservation and returns its context to READY;
`CancelExecutionContext` cancels the whole job even while queued. Running
cancellation settles at a safe boundary and drains submitted device work.
Transport cancellation stops delivery, not the engine job. A later cancellation
or release of an old Request cannot affect a newer context invocation.

The bridge detects device loss and fails access to that device. It does not
automatically rebuild a host, increment a public recovery epoch, or replay
requests. Close/create-host regression checks with the pinned Deno runner
are documented in [runtime validation](c-runtime-validation.md). Automatic
service recovery is a separate optional capability.

Output buffer release and host teardown preserve submitted-resource lifetimes.
A failed map or logical cancellation cannot release a buffer still referenced
by device work. Trainer rollback has its own full-profile contract and must
not be inferred from inference-result failure.

## Verification and measurements

[BatchQueue tests](../tests/parity/external/batch_queue.mjs) check atomic
admission refusals, owned and bounded results, fill-first deadlines, draining
and failure containment through the public API; `native/tests/test_batch_queue.c`
checks dispatch planning, padding, token budgets and page reservations.
[Decode tests](../tests/parity/external/webgpu_decode.mjs) and
[paged-cache tests](../tests/parity/external/decode_cache.mjs) check numerical
outputs, inactive rows, generations, prefix reuse, and retained snapshots.
Their finite fixture coverage is described in [runtime validation](c-runtime-validation.md).

Performance records should bind source and release artifacts, model/inputs,
backend/device/driver, and the exact trace. Count logical requests, physical
graph invocations, useful/padded work, queue delay, TTFT, token latency, and
memory separately. BatchQueue worker_busy_ns measures dispatch exposure to
completion; it is not GPU hardware occupancy. Use a real monotonic clock for
timing and retain reproducible seeds for policy tests.

Further fault injection, whole-model qualification, independent optimizer
references, and measured scheduler changes belong in [TODO.md](../TODO.md).
