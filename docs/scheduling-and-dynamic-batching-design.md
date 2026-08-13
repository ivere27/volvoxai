# Scheduling and dynamic batching design

VolvoxAI is an on-device edge engine for web browsers and robots. It supports
both inference and training. This document specifies Runtime scheduling for
forward execution: stateless requests, prefill, and decode. Trainer-owned
gradients, optimizer state, accumulation, RNG, and mutable revisions remain
under the full-profile `Trainer`; independent training steps are never silently
coalesced by the Runtime scheduler.

This document is the detailed scheduling contract. [`ARCHITECTURE.md`](../ARCHITECTURE.md)
owns system-wide invariants, and [`TODO.md`](../TODO.md) is the only authority
for unfinished work. Benchmark reports may retain retired planning/checklist
source names such as `PLAN.md`, `docs/roadmap.md`, and
`tests/parity/TODO.md` in immutable provenance fields because those were the
source names at measurement time; the active design is consolidated here. The
unreleased v1 contract is corrected in place. There is no v2 scheduler API or
compatibility shim.

## Invariants and ownership

One `Runtime` owns one logical forward-execution coordinator and one result
budget across all models compiled by that Runtime. The coordinator may dispatch
independent resource domains concurrently, but it must not merge model weights,
mutable route state, or session state.

```text
Runtime
  +-- result budget ledger
  +-- lazy request coordinator
  |     +-- resource-domain dispatcher(s)
  |     +-- route queues and admission accounting
  |     `-- request handles and policy state
  +-- CompiledModel A -- shared immutable graph and provider resources
  |     `-- route/context -- bounded mutable execution state
  `-- CompiledModel B -- distinct immutable artifact and routes
```

The ownership rules are:

- The target ownership rule is that an exact `CompiledModel` owns immutable
  graph state, weights, compiled plans, and pipelines once. A route or context
  then borrows those resources and owns only bounded mutable arenas, inputs,
  outputs, workspace, and optional KV state. The JavaScript CPU, WASM, and
  WebGPU providers implement that invariant-resource owner/lease boundary.
  Native shares the compiled raw safetensors store, but its CPU derived packs
  and built-in GPU graph/device resources remain context-owned work in
  [`TODO.md`](../TODO.md); do not read this target rule as a completed native
  device-residency claim.
- The default is one mutable execution route per exact compiled target,
  time-multiplexed by its resource-domain dispatcher. A bounded context pool is
  justified only by real overlapping execution and measured benefit.
- Sequence state, KV pages, lane generations, and prefix leases are
  route/session-local. Different models may share admission and device budgets,
  never a physical batch or mutable context state.
- The `ResultBudgetLedger` exists even for `DIRECT`, but using `DIRECT` does not
  allocate the request scheduler, queue, timer, worker, or telemetry ring.
- Published results own their snapshots and result-budget leases until
  `ExecutionResult.close()` or final native result release. Runtime shutdown
  does not forcibly close caller-held results.
- The inference entry remains free of every training dependency. The full
  profile may use the same Runtime for forward execution, but `Trainer` remains
  the separate owner of mutation. Cross-training/inference device arbitration
  is not implied by this scheduler.

Measure invariant ownership and batching separately. Where the provider has
implemented the compiled-owner rule, `weights once + N x bounded mutable route
capacity` is an ownership result; fewer provider graph invocations for N
requests is a batching result. Native measurements must itemize its remaining
context-owned derived/device resources.

## Execution modes

`ExecutionMode` is defined in `proto/volvoxai.proto`. Generated TypeScript and
native bindings are the code-level source of truth; schedulers must not maintain
handwritten string or integer copies of the enum. TypeScript obtains its public
values through `executionModes[ExecutionMode.Direct]` and
`executionModes[ExecutionMode.Scheduled]`; native uses
`VX_EXECUTION_MODE_DIRECT` and `VX_EXECUTION_MODE_SCHEDULED`.

| Mode | Contract |
| --- | --- |
| `DIRECT` | Executes one logical call through a route lease while bypassing the Runtime coordinator. It creates no scheduler queue, timer, worker, request table, or scheduler telemetry ring. A busy route returns `BUSY`; there is no hybrid fallback into admission. Calls on different routes opt out of global fairness and arbitration. |
| `SCHEDULED` | Lazily allocates the coordinator and always uses bounded queue admission, priority, EDF, aging, deadlines, and freshness. TypeScript `maxBatchDelayMs=0` or native `max_batch_delay_milliseconds=0` dispatches work-conservingly; a positive value permits a bounded coalescing delay that a deadline may shorten. |

DIRECT is about Runtime scheduling ownership, not the absence of all allocation,
synchronous completion in JavaScript, B=1, or bypass of a provider/device queue.
It may execute a caller-authored B=N bulk input, and its result snapshot and
provider resources still follow their normal ownership contracts.

The Runtime configuration is a capability bound: a DIRECT Runtime cannot enter
scheduled admission, while a SCHEDULED Runtime may run a narrower DIRECT call.
JavaScript exposes `CompiledModel.run()` for either mode and `submit()` for
SCHEDULED work only; native exposes the always-DIRECT `vx_runtime_run()` and
always-SCHEDULED `vx_runtime_submit()`. There is no separate public
dynamic-batch queue or callback-driven `step()` serving API. Browser Worker and
robot service policies are SCHEDULED configurations rather than distinct modes.
A caller that receives `BUSY` from DIRECT may issue a separate SCHEDULED call;
the DIRECT call itself never changes modes.

## Work kinds and batching semantics

Three forms of work must remain distinct.

### Caller-authored bulk execution

The caller supplies an input whose concrete public batch is `B=N`. The graph may
intentionally reduce, normalize, or communicate across B. The provider performs
one graph invocation and returns one bulk result; Runtime does not reinterpret
or split it into N independent requests. Training microbatches are in this
category.

### Scheduler coalescing

Runtime receives N logically independent B=1 requests, stacks compatible inputs,
performs one provider graph invocation over B=N, and splits outputs back into N
stable results. This requires both core's typed independence proof and the
provider's one-invocation batch attestation. A host loop containing N B=1 graph
calls is not physical batching.

### Stateful row execution

Decode addresses rows belonging to resident sequences and their KV state. It
uses an explicit admitted-lane list and declared query layout; it is not a bulk
tensor reshape and never invents padded KV lanes.

The common scheduling concept is a contribution:

```text
(request or session,
 structural route,
 phase,
 rows or query layout,
 state reference,
 deadline and policy)
```

Stateless work, bounded prefill, and decode may share admission and dispatch
policy without being forced through the same provider primitive. A multi-model
request is an explicit state machine across compiled targets, not automatic DAG
partitioning by the scheduler.

## Compatibility routes and typed independence proof

Compatibility is a provider- and core-produced structural identity, not a
caller string or delimiter-concatenated key:

```text
(resource domain,
 provider-owned compiled executable token,
 compiled/model/weight/adapter revisions,
 canonical non-B shape, layout, dtype and tactic,
 phase and query layout,
 device epoch)
```

Prefix reuse additionally includes tokenizer semantics, prompt tokens or bytes,
masks, positions, quantization state, and every revision that can affect KV.
For scheduler-coalescible work, admission resolves each request's immutable
B=1 concrete plan once. After group selection, the scheduler resolves one
separate B=N stacked plan and carries it through provider execution. A
caller-authored bulk call instead resolves its one B=N logical-request plan and
is not restacked or split. A phase must not repeatedly resolve a possibly
different version of any plan.

For TypeScript and native scheduler coalescing, core computes evidence bound to the exact
graph fingerprint and follows the request axis through every execution tensor.
Each typed operator configuration must preserve lane independence across:

- axis movement, broadcasting, slicing, indexing, concatenation, and reshape;
- reduction, normalization, attention, and other communication axes;
- fixed versus activation operands and quantization axes;
- aliases, liveness, output layout, and provider resource/tactic selection.

The provider separately attests that the selected executable accepts the full
batch domain in one provider graph invocation. Unknown operators, incomplete
evidence, fingerprint mismatch, or lane mixing fail closed to scheduler B=1.
Each core computes frozen evidence before provider compilation. An external
provider must echo the exact graph fingerprint, proof identity, proved axis and
a range contained by the proof; provider self-attestation cannot promote an
unproved graph. Native Vulkan, OpenGL, and CUDA built-ins consume that proof
directly after their complete bounded-domain qualification. Their current
stateless path stacks host inputs, enters the built-in engine once at B=N, and
splits the single output snapshot. Native CPU and Metal remain scheduler B=1.

Prefer a producer-authored symbolic B domain. Lifting a fixed-B=1 graph requires
a typed-IR `vmap` transform before shape resolution, followed by complete
shape/domain/alias/liveness/memory/tactic/backend re-proof. Prepending a batch
axis to an already resolved plan is forbidden. Rank- and default-derived
parameters must be materialized and remapped before the lift; regression cases
must include omitted-axis `ArgMax`, rank-dependent `SSMScan`, and omitted reverse
permutation `Transpose`. The typed operator registry and its adversarial corpus,
not a prose allowlist, are authoritative.

## Legal B, operating B, and padding

The legal batch set is the intersection of:

- producer-declared symbolic bounds;
- typed lane-independence proof for scheduler coalescing;
- provider/kernel physical-domain proof;
- public API buffer, binding, alignment, and dispatch limits; and
- any reproducible effective allocator boundary that is stricter than the
  advertised device limit.

Legal is not profitable. The operating B is selected only inside that set from
measured route/device `T(B)`, deadline slack, live memory pressure, energy, and
contention. A cold or invalidated route starts at B=1. Measurements are bounded
and invalidated by changes to compiled artifact, provider, device/driver,
resource epoch, tactic, thermal regime, or material memory-pressure regime.
Core count, warp/subgroup width, and advertised VRAM are not universal max-B
rules.

Current production schedulers do not synthesize padding: TypeScript's admitted
independent-batch contract requires `multiple_of=1`, while native selects the
largest legal multiple and leaves a remainder queued. If dense coalescing later
uses padding, it may duplicate only a valid input lane, must discard that output,
and must count it against rows, tokens, staging, outputs, and physical memory.
Stateful row decode never pads with nonexistent sequence or KV state. The
internal `BatchScheduler` exercises padding isolation as a policy-core test; it
is not evidence that production Runtime currently pads.

## Admission and bounded resource accounting

SCHEDULED admission is transactional:

1. validate names, dtypes, metadata, and concrete shapes without copying the
   full payload;
2. prepare one immutable concrete request plan — B=1 for a scheduler-
   coalescing candidate, while caller-authored bulk retains its B=N plan;
3. calculate exact or conservative resource requirements;
4. atomically reserve every admission-controlled budget;
5. acquire counted leases or snapshot/transfer input payloads;
6. publish the request to its route queue.

`DIRECT` instead preflights the caller's binding and metadata, acquires the
direct route lease, and reserves result capacity before provider mutation. It
does not publish a queued request or take the scheduled owned-input snapshot.

Any failure rolls back all claims and leaves an existing transactional
`LATEST` predecessor intact. A known capacity shortfall is `OVERLOADED` at
admission with a retryable policy signal, not a predictable allocation failure
after provider mutation.

Current production accounting covers active requests, owned/admitting/staging
host-input bytes, exact logical result count/output bytes, and configured route
batch/window limits. It does not yet claim a whole-device high-water bound over
compiled weights, aligned activations/workspace, result/readback buffers,
KV/prefix pages, or provider-private allocations. Those remaining budgets and
bounded request/frame/telemetry slabs are tracked in `TODO.md`.

Result capacity is reserved before provider execution. A successful reservation
transfers to the published result; failure or discarded output releases it.
Lanes backed by one physical allocation retain the shared byte lease until the
last lane closes. The result ledger is not an unbounded completion archive.

## Dispatch policy, fairness, and operating batch selection

Each free resource domain selects one eligible structural route. Independent
domains may progress concurrently. Within a route, FIFO is the base order and
priority, earliest deadline, aging, and bounded bypass prevent starvation.
The scalable target is per-route FIFO plus a per-domain ready heap, with no
per-dispatch linear scan of all requests.

With `maxBatchDelayMs=0`, SCHEDULED dispatches immediately or at the next
scheduler turn. A positive value may wait only inside a bounded window capped
by the earliest compatible deadline. A wait must be justified by measured
`T(B)` and current slack; arrival-rate EMA alone is not a service-time model.
Selection is bounded by batch count,
`multiple_of`, useful/padded rows, tokens, input/output/staging bytes, and the
physical resource plan.

Freshness semantics are:

- `ALL`: preserve the accepted request, or reject it at admission.
- `LATEST`: transactionally supersede older stateless work from the same stream.
  Queued work may transfer compatible claims; submitted work keeps physical
  leases through its fence while result publication is suppressed.
- `DROP_IF_LATE`: requires a deadline and uses hard queued/completion cutoffs.
  Accepted late `ALL`/`LATEST` work may publish while recording a missed
  deadline.

Do not apply stateless `LATEST` semantics to audio or decode sessions whose
state transition depends on every prior request.

## Stateful sessions, prefill, decode, and Paged KV

A production session and each of its physical contributions have separate
lifecycles. The target session phases are:

```text
admitted -> prefill-ready -> prefilling -> decode-ready
         -> decoding (zero or more steps)
         -> completed | cancelled | failed | reset-required -> retired
```

Every prefill chunk or decode step independently moves through
`queued -> reserved -> submitted -> settled`; a session may repeat that
contribution lifecycle many times. The exact stateful SCHEDULED transition table
remains an unfinished contract in `TODO.md`.

Session FIFO, lane generation, the active/query-length representation and bounds,
KV ownership, and device epoch are part of compatibility and settlement.
Concrete per-lane lengths are dispatch values, not route identity. Seed, step,
reset, and close are batch transactions. KV/page reservations are provisional
until the whole step succeeds; cancellation or failure restores every lane and
page. Pages retire only after the provider fence and every open result lease
that can reference them. A stale generation or device epoch can never mutate a
reused lane.

Prefix reuse is structural and route-local. Publish only fully committed,
page-aligned prefix boundaries. Published shared pages are immutable; a writer
must acquire a private copy before mutation. Eviction requires explicit
recomputation or reload; execution must not silently continue with incomplete
KV.

Prefill should evolve in measured stages:

1. bounded prompt-length buckets with padding/result isolation;
2. a typed chunked-prefill primitive carrying `(context, row_start, row_count,
   lanes, query-axis keep mask)`;
3. dense direct `q_len`/`kv_len` values that limit compute without participating
   in shape inference;
4. `[N_total, D] + cu_seqlens` VarLen packing when measurement proves it useful;
5. mixed prefill/decode only after the simpler modes are correct and profitable.

## Provider submission and completion

The concrete provider SPI is defined in [`backend-sdk.md`](backend-sdk.md). Its
scheduler-facing meaning is:

- a frozen v1 batch contract and provider-owned route/domain/epoch identities;
- exact compile-time physical-domain qualification;
- one provider graph invocation for a coalesced B>1 request;
- owned input copies, transfers, or counted leases through the completion fence;
- immutable per-lane result snapshots or closeable leases;
- no runtime fallback to a different backend after compilation; and
- no partial result publication after provider failure.

The TypeScript scheduler can await provider promises per resource domain. The
native production target is completion-driven submit/poll/fence dispatch rather
than blocking every domain behind one worker. WebGPU host stacking/readback,
the native external dense-batch callback, and the built-in native GPU one-forward
path prove physical batching, but do not by themselves prove zero-copy
device-resident I/O.

## Results, cancellation, failure, device loss, and shutdown

Invalid work is isolated before it joins a batch. Once one physical batch is
submitted, provider failure fails the selected batch atomically. Stateful
mutation commits as one step or rolls back as one step; EOS and other declared
per-lane outcomes may still differ after a successful invocation.

Cancellation and supersession suppress logical publication immediately where
the request contract permits it, but submitted buffers, routes, KV, and result
reservations remain owned until the physical completion fence. They are never
reused on logical cancellation alone.

Device loss increments the resource epoch and invalidates every old route,
pool, and lease. Only queued replayable stateless work may be recompiled and
retried when its freshness/deadline policy permits it. In-flight work fails
`DEVICE_LOST`; stateful work fails `SESSION_RESET_REQUIRED` unless a provider
has a separately proven recovery transaction.

Shutdown order is:

```text
stop admission
  -> settle/cancel queued work
  -> await submitted fences
  -> close routes/contexts
  -> close providers and device owners
```

Published result lifetime is independent of this logical Runtime shutdown.

## Telemetry and performance gates

Target production observability must record steps and provider graph invocations
together, with a shared dispatch identity for lanes from one invocation.
Required bounded metrics include:

- useful and padded rows/tokens, batch size, and group count;
- queue depth max and P50/P99, queue delay, request latency P50/P95/P99, and
  TTFT;
- device-busy and wall time;
- request/input/staging/result/device/workspace/KV high-water marks;
- deadline misses, fairness/aging/bypass decisions;
- page utilization and prefix reuse;
- host-device bytes, energy where available, and route-specific `T(B)`.

The injectable policy clock exists for deterministic tests. Elapsed/device
measurement uses a real monotonic clock. Never derive performance telemetry
from the virtual policy clock, and retain only bounded windows.

Promotion gates include B=1 DIRECT regression budgets, numerical equivalence
to independent B=1 execution, proof of one provider graph invocation, bounded
ownership evidence, and browser/robot traces covering tail latency and every
high-water counter. A legal B is not promoted merely because it fits.

## Browser, robot, and full-profile integration

A browser deployment should use one Worker or offscreen host to own Runtime,
device, and budgets for all clients. A robot process coordinates vision, audio,
and language streams through deadlines, priority, freshness, and session FIFO.
Both deployments require device-loss recovery and long-running bounded-memory
soak tests.

The full profile can reuse the forward Runtime, but Trainer mutation remains a
separate lifecycle. Scheduler code must not pull training code into the
inference entry or native inference profile.

## Implementation status and TODO boundary

| Capability | TypeScript | Native |
| --- | --- | --- |
| DIRECT and shared result budget | Implemented | Implemented |
| Lazy stateless SCHEDULED coordinator | Implemented vertical slice | First bounded worker slice |
| Authored symbolic-B coalescing | Core proof + provider echo on CPU JS, WASM, WebGPU | Core proof + exact external echo; built-in Vulkan/OpenGL/CUDA one-forward path |
| Scheduled device I/O | WebGPU host stack/readback | HOST snapshots only |
| Stateful Runtime sessions | Not implemented | Not implemented |
| Paged-KV/row-decode policy core | Low-level core and tests | Low-level core and tests |
| Fixed-B typed lift | Not implemented | Not implemented |
| Measured `T(B)` autotuning | Not implemented | Not implemented |
| Whole-device resident budget | Not implemented | Not implemented |
| Async per-domain fence dispatch | Promise/domain boundary present | Not implemented |
| Device-loss recovery | Detection only | Not implemented |

The detailed unfinished work, dependencies, and release gates live only in
[`TODO.md`](../TODO.md). This document must not accumulate a second roadmap or
completed benchmark diary.

## Internal batch scheduler core

`ts/core/BatchScheduler.ts`, `native/src/runtime/batch_scheduler.{c,h}`, and
their `ContinuousBatchScheduler` adapters form an internal, engine-independent
batch scheduling policy core. They exercise grouping, fill, padding,
stateful-lane, cancellation, backpressure, and telemetry rules with an injected
manual runner. They are active testable code, not a public serving owner and not
the production Runtime coordinator.

`BatchScheduler` does not own compiled artifacts, providers, resource domains,
or public request lifecycle. Production ownership is the exact `CompiledModel`
route plus Runtime's resource-domain coordinator. The name deliberately includes
`Batch` to distinguish this policy core from `RuntimeScheduler` and the native
`VxRuntimeCoordinator`; the stateful adapters use the more specific
`ContinuousBatchScheduler` name.

The twin suites use semantic invariants rather than document section numbers:

- **transparency**: independent reference and grouped results agree;
- **determinism**: the same arrivals and policy clock yield the same schedule;
- **route separation**: incompatible structural routes never merge;
- **padding isolation**: padding does not publish or mutate user state;
- **failure, cancellation, and drain**: ownership lasts through physical work;
- **bounded backpressure**: queue, result, frame, and telemetry storage remain
  bounded; and
- **telemetry**: useful work and physical invocations are not conflated.

These fixtures do not preserve an older public scheduler API. If a fixture
assumption conflicts with the production ownership contract, change or remove
the fixture in v1 rather than adding a compatibility layer.
