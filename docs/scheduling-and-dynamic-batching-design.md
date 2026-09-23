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
| Runtime request coordinator | Submit, PollRequest, WaitRequest, CancelRequest, TakeRequestResult | Admit stateless model requests, select compatible work, and execute compiled routes |
| ExecutionContext | Execute, DecodePrefill, DecodeStep, DecodeGenerate, cache operations | Own mutable bindings, sequence rows, KV storage, and execution state |
| BatchQueue | SubmitBatchWork, NextBatchDispatch, CompleteBatchDispatch, GetBatchWork, TakeBatchWork | Group application work and return dispatches for a worker to execute |

A BatchQueue is an independent C policy owner. It holds copied application
inputs, work state, and page/lane reservations, but no compiled model or GPU
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

Stateless Run/Submit reuse a compiled target's internal execution context under
a route lock. Explicit decode contexts retain separate sequence state.
BatchQueue compatibility groups do not merge these contexts or their KV.

ExecutionResults own immutable output snapshots. Pending and caller-retained
results keep their budget charges until settlement or final release. Releasing
a parent public handle leaves retained descendants valid. Closing a web host
retires its C owner and drains GPU cleanup.

## Direct and scheduled requests

Omitting CreateRuntimeRequest.execution_mode selects DIRECT. SCHEDULED enables
Submit; a DIRECT runtime rejects Submit. Run is always the direct submission
operation and takes a compiled-model ID.

| Path | Behavior |
| --- | --- |
| Run | Reserves a result and leases the compiled route without creating a request queue or handle. A busy route returns BUSY. |
| Submit | Validates and reserves Runtime budgets, copies input bytes, and returns a request handle. |
| Execute / decode | Operates on the caller's explicit ExecutionContext and its retained state. |

A direct request may contain a caller-authored B=N tensor. It still represents
one logical execution and one result. It does not imply a synchronous GPU
completion: an ExecutionResult may be PENDING.

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

These are stateless request policies. Sequence-dependent steps use their
context or BatchQueue lifecycle.

The native Runtime currently has one worker. Threadless WASM uses the same C
coordinator cooperatively: PollRequest and finite WaitRequest calls do not
execute queued work; an absent or UINT64_MAX timeout pumps it. The TypeScript
transport does not supply a separate scheduler or timer-driven executor.

## Physical batching and compatibility

Caller-authored bulk B=N can intentionally mix information between lanes.
Scheduler coalescing combines independent B=1 requests and therefore requires
C's typed independent-batch proof plus a backend route that executes the
aggregate graph once. The resulting outputs are split into owned lane results.

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
Decode lanes refer to real reserved sequence state.

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
alignment, fixed allocations, workspace, and snapshots. DecodeCacheState
separately reports page use, reserved bytes, allocated pool bytes, and prefix
snapshot bytes. A free page remains part of its allocated pool.

These mechanisms do not establish one admission ledger for every allocation
sharing a physical GPU. Such a ledger must count shared resources once, retain
charges through completion, and distinguish engine-owned allocation bytes from
unobservable driver overhead. The remaining work is specified in TODO.md.

## BatchQueue and worker dispatch

CreateBatchQueue configures queue depth, token budget, maximum lanes,
stateless divisibility, fill policy, retained-result capacity, and optional
decode-cache metadata. The queue has work-conserving and fill-first policies;
max_wait_ns bounds the latter.

SubmitBatchWork copies its payload and inputs. Decode work also declares prompt
and maximum generated-token counts. C reserves lanes/pages, chunks prompt work,
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
dispatch. Its lane cannot be reused underneath that work. CloseBatchQueue with
drain finishes admitted work; without drain it cancels work and invalidates
pending dispatch IDs. Worker/device resource cleanup remains with the worker.

## Decode contexts and paged KV

DecodePrefill accepts a final prompt position or one position per declared
lane. DecodeStep advances a lane, recomputes its last row with idle, or skips it
with parked. GetDecodeState exposes active lengths, parked flags, and cache
generation. Lengths are execution state, not ragged tensor dimensions.

Omitted step inputs reuse the complete prefill binding. decode_inputs identifies
which roots need refresh. Explicit dependency_update has its own single-lane
AUTO contract and does not advance the cursor.

ConfigureDecodeCache binds internal causal-attention K/V activations to C-owned
page bookkeeping. PublishDecodePrefix retains a page-aligned prefix; writers
use copy-on-write. ReuseDecodePrefix fills an empty lane while preserving other
lanes and retained results. ReleaseDecodeLane, eviction, and reset retire cache
ownership. Page reservations roll back on admission failure.

The current C DecodeGenerate appends a fixed token count to a single-lane
required-row context using ArgMax/QArgMax feedback. WebGPU keeps intermediate
token copies on the device and snapshots only final outputs. Sampling,
EOS/stop sequences, and incremental generation results are optional API work.

BatchQueue's prompt chunks already exist as worker dispatches. Efficient
per-lane query/key kernel inputs and mixed or flat-packed execution are
additional numerical contracts, not missing queue policy.

## Failure, completion, and device lifetime

Execution acceptance and device completion are separate. Poll a PENDING
ExecutionResult through GetResult; ReadOutput returns BUSY until completion.
READY exposes all outputs together. FAILED contains the execution or readback
failure. Repeating execution to wait for completion would submit new work.

Pre-commit input/domain refusal preserves the prior binding and decode state.
A post-submission failure does not certify that all mutable device state was
restored. Continuation, new prefill, and owner recreation must follow the
specific failure contract and its regression evidence. C owns those decisions;
a TypeScript callback only reports device completion/loss.

The bridge detects device loss and fails access to that device. It does not
automatically rebuild a host, increment a public recovery epoch, or replay
requests. Close/create-host regression checks with the pinned Deno runner
are documented in [runtime validation](c-runtime-validation.md). Automatic
service recovery is a separate optional capability.

ReleaseResult retires the public result while submitted resources drain safely.
A failed map or logical cancellation cannot release a buffer still referenced
by device work. Trainer rollback has its own full-profile contract and must
not be inferred from inference-result failure.

## Verification and measurements

[BatchQueue tests](../tests/parity/external/batch_queue.mjs) compare dispatch,
page, and result traces with the pinned former implementation.
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
