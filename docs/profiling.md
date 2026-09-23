# Profiling and memory

Use a trace to inspect host work and supported GPU execution intervals. Traces contain actual
observations, including individual executions and decoder steps. Read the typed
records to build tables, or download Chrome Trace JSON and open it in
[Perfetto](https://ui.perfetto.dev).

## Open the analysis tables

In the receipt digit reader or tiny receipt VQA example, enable profiling, run
the model, then click **Open in Perfetto**. The viewer opens the original timeline
and prepared analysis tables. No SQL entry is needed. To inspect a saved JSON
from C, Python, WASM or WebGPU, serve the
repository over HTTP, open [Trace analysis](../examples/profiling.html), choose
the file, and click **Open in Perfetto**.

| Table | Use it to answer |
| --- | --- |
| **Slow nodes** | Which executable nodes cost the most? See names, schedule indices, fusion, calls, total/average/maximum time and share of recorded node time. Models, compilations, contexts, phases and host/GPU domains stay distinct. |
| **Run summary** | How variable are repeated engine calls? See sample count, first observed call, later-call average, overall average, median, nearest-rank p95 and maximum. |
| **Runs** | Which individual call was slow? See chronological run number within each context/operation, execution ID, start time and host duration. |
| **Host operators / GPU operators** | Which operator types dominate? Aggregate nodes by backend, phase and operator, ordered by total time. |
| **Copies and waits** | Are transfers or synchronization expensive? See calls, copied bytes and elapsed times, separated by backend, device, queue, direction and activity. |
| **Memory** | What was the observed allocator peak and growth? See inventoried existing/live/peak capacities, allocated/freed MiB, observation start, first peak time when known, accounting coverage and dropped events. |
| **Capture** | Are these measurements usable? Check recording options, lost/truncated events, GPU coverage and interpretation notes. |

**Slow nodes** opens first for node traces; BASIC traces open on **Run summary**.
When the engine reports lost events, **Capture** opens first to show that loss.
Times are displayed in ms, allocator capacities in MiB, and copy sizes in bytes
so small transfers remain visible. Every table covers the entire
capture, independently of the selected timeline area. Node percentages use the
sum of observed node times within the same model, compilation, context, backend,
phase and timing domain. They do not measure application latency or utilization.
Host times describe CPU/WASM execution or GPU host preparation/submission/waits.
GPU times use `deviceDurationNs`, including CUDA/WebGPU instants. Passes, nested
programs and truncated node metadata are excluded from node/operator totals.
Missing events still make summaries incomplete; an empty table means no usable
observations, not zero cost.
A zero model, compilation or context ID means that public identity was not
reported; records can only be separated by the identities available. Schedule
index zero, however, identifies the first executable node.

Run statistics use recorded **host operation calls**, including compilation,
execution and training as separate groups. An execution call can finish before
asynchronous GPU work or output readback. The first observed call need not be a
cold start, and subsequent calls are not automatically classified as warmed up.
Compare runs with the same inputs and inspect the sample count before using p95.
No application-level request boundaries or decoder token labels are invented.

Copy, submission, blocking-wait and asynchronous-completion durations can overlap;
their rows must not be added together. GPU copy rows require actual device copy
timestamps. Host copy duration is not device bandwidth, and asynchronous
completion time is not CPU busy time.

Memory uses the engine's `otherData.allocators` summaries, which remain useful
when individual history events were dropped. Peak time comes from the first
matching recorded transition only when accounting and allocator history are
complete; otherwise it is absent. Inventories are partial even with complete
accounting. Capacities describe requested storage, not physical RAM/VRAM or
individual tensor lifetimes. Shared and runtime-scoped inventories are separate.
`existing_mib` includes preexisting storage as owners first appear, possibly later
in the capture. `growth_mib` is live minus that inventoried existing storage; it
does not measure the change between the first and last plotted counter samples.

**Export Perfetto Trace** still saves unmodified engine JSON. Chrome Trace JSON
does not carry executable SQL views or dashboard settings: dropping that file
directly into Perfetto opens its timeline. The example's viewer helper supplies
[startup commands](https://perfetto.dev/docs/visualization/deep-linking-to-perfetto-ui#startup-commands)
to create the analysis tables. The helper reads the exported summary metadata
and supplies it as quoted SQL values because Perfetto does not import `otherData`.
It passes original bytes to the new tab with `postMessage`
without uploading the trace or enabling sharing. The original JSON remains
available through the example's export button.
The browser must allow the new tab, and the serving page must preserve its
opener relationship (do not use `Cross-Origin-Opener-Policy: same-origin`). This
viewer setup and its query module load only when opening a completed trace and
add no collection work or runtime bundle dependency. Parsing the saved JSON and
running these analyses takes viewer-side time and memory after collection.

Perfetto's [Data Explorer](https://perfetto.dev/docs/visualization/data-explorer)
graph describes analysis steps such as filtering and aggregation. It does not
display a model's operator/tensor graph. That would require a separate viewer or
a [custom Perfetto UI plugin](https://perfetto.dev/docs/contributing/ui-plugins),
plus model topology data; the trace contains executable-node attribution rather
than every input/output edge.

## Capture a trace

Collection is disabled by default. Start after warmup to measure steady state,
or before loading a model to include loading and compilation. A runtime accepts
one collecting or draining trace; completed traces remain readable independently.

```js
import { VxProfilingServiceClient, pb } from './volvoxai.js';

const profiling = new VxProfilingServiceClient(host);
const trace = await profiling.startTrace(new pb.StartTraceRequest({
  runtimeId,
  detail: pb.TraceDetail.TRACE_DETAIL_NODES,
  capacityBytes: 4n * 1024n * 1024n,
  deviceTiming: true,
  memory: true,
}));

try {
  // Run your existing inference, training or decode calls here.
  await runWorkload();

  const ref = new pb.TraceRef(trace);
  let stopped = await profiling.stopTrace(ref);
  while (stopped.state !== pb.TraceState.TRACE_STATE_READY) {
    await new Promise(resolve => setTimeout(resolve, 0));
    stopped = await profiling.getTrace(ref);
  }

  const chunks = [];
  let offset = 0n;
  for (;;) {
    const chunk = await profiling.exportChromeTrace(new pb.ExportChromeTraceRequest({
      traceId: trace.traceId, offset,
    }));
    chunks.push(chunk.data);
    if (chunk.eof) break;
    offset = chunk.nextOffset;
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  const file = new Blob(chunks, {type: 'application/json'});
  // Save `file` using your application's ordinary download UI.
} finally {
  await profiling.releaseTrace(new pb.TraceRef(trace));
}
```

For tables, page through `ReadTrace` using `nextOffset` until `eof`. A page read
is repeatable and does not consume events. Stop before reading; `BUSY` means an
accepted host work or an asynchronous device query has not finished. Releasing a trace closes admission
without cancelling inference. Models, contexts and result tensors are not
retained by completed records.

The byte budget is allocated at start. Exhaustion drops additional events and
increments `droppedEvents`; inference continues. Always display this count when
presenting aggregates. Export serializes one event page at a time. Its `offset`
and `nextOffset` count source events, just like `ReadTrace`; `limit` defaults to
128 events and accepts 1–1,024. Concatenate the returned UTF-8 fragments from
offset zero until `eof`. The engine keeps no full JSON copy or export cursor:
repeating a request returns the same bytes, and scratch memory depends on the
page size. A nonzero offset at the event count returns an empty EOF chunk. Fixed trace metadata appears in the first/last page. Event names and
output metadata are bounded; `metadataTruncated` identifies a shortened value.
For an interactive UI, yield between pages with `setTimeout(resolve, 0)` or run
the host in a worker. The final Blob belongs to the application and still costs
memory proportional to the complete document.

For numerical qualification and profiling-overhead comparisons, see
[testing](testing.md#profiling-checks).

## Interpret the measurements

The public API uses unsigned integer nanoseconds for every time value: trace
observations, compilation/binding/execution metrics, the monotonic clock,
scheduler deadlines and batch delays/statistics. JavaScript clients expose
these as `bigint`; C and Python retain the full unsigned 64-bit range. Convert
only when displaying a value, for example `Number(durationNs) / 1e6` for ms.
Do arithmetic on timestamps as integers before converting a short difference.
Nanosecond storage does not imply nanosecond precision: the host clock is
sampled at microsecond resolution, and browser precision can be coarser.
`hostClockResolutionNs` reports effective native resolution; zero means unknown.
Scheduling delays round up to the existing scheduler tick, as specified in the
schema. They do not add a higher-frequency clock to ordinary execution.

Chrome JSON's standard `ts` and `dur` fields remain microseconds, as required
by that format, with `displayTimeUnit: "ns"`. This is an export-boundary
conversion. Custom duration fields remain ns. IDs and exact integer byte/ns
metadata are decimal strings; viewer counter values are numeric and may be
rounded above JavaScript's exact integer range.

Choose one detail level for the capture and whether to collect GPU time.
The engine selects the timestamp mechanism for each backend:

| Option | Measurements |
| --- | --- |
| `detail: BASIC` (default) | Host operations such as Execute, CompileModel and TrainStep; existing device passes when GPU timing is enabled |
| `detail: NODES` | Also executable host nodes, training phases and programs; supported GPU node/program intervals when GPU timing is enabled |
| `deviceTiming: false` (default) | No GPU timing resources or device queries, including during node tracing |
| `deviceTiming: true` | Collect GPU elapsed time at the selected detail; see actual coverage in the result |
| `memory: true` | Bounded allocation/free history, allocator peaks and two process envelope samples |

`startTrace({runtimeId})` therefore records basic host work. Request `NODES`
to investigate executable nodes, and enable `deviceTiming` to include GPU time.
There is no separate device granularity option. `memory` defaults to
false; `capacityBytes` controls storage, with a 4 MiB default. Detailed copy
and completion records consume that same budget. Increase it for long captures
and check `droppedEvents` before treating a trace as complete.

Each typed event contains exactly one observation payload. `event.host` has `startNs`
and `durationNs` on the capture-relative monotonic host clock. `event.device`
has `hostObservedNs` and `elapsedNs`: the former marks host encoding/submission,
and the latter is an elapsed device measurement. Optional `correlation` places
the device start within a capture-relative host time range; its method explains
whether that range comes from clock calibration or causal observation bounds. `event.phase` identifies forward, loss, backward, gradient
processing or optimizer work; `UNSPECIFIED` means no phase attribution.
`event.program` identifies an engine program by name and entry point. When
`event.node` is also present, it identifies the program's owning executable node.
Without a program, a node event measures that node's call or device interval.
The node's `scheduleIndex`, `outputName` and `fused` fields describe the executable
schedule, including index zero. `tensorName` identifies a loss target or updated
parameter when known. Missing origins stay absent. For example:

```js
for (const event of page.events) {
  if (event.host) console.log('Host duration:', event.host.durationNs);
  if (event.device) console.log('Device elapsed:', event.device.elapsedNs);
  if (event.memory) console.log('Allocation observation:', event.memory);
}
```

A host node includes CPU work on CPU/WASM. On asynchronous GPU backends it
measures host preparation and submission; synchronous backend waits can also
be included. It does not identify a kernel's GPU execution time.

| Backend | Device observations |
| --- | --- |
| CUDA | Forward/training passes, forward nodes and registered engine kernel launches; forward timing preserves Graph replay in a separate instrumented plan |
| OpenGL desktop | Forward nodes, forward/training programs and passes, using timer queries |
| Vulkan | Forward nodes, forward/training programs and passes, using queue timestamps |
| WebGPU | Existing passes at BASIC; forward nodes and forward/loss/backward/gradient/optimizer program dispatches at NODES |
| CPU/WASM, Metal, GLES, external providers | No device intervals; CPU/WASM host nodes measure numerical work |

`TraceInfo.devices` reports support separately from the recorded pass/node/program
counts. A row appears when a built-in device adapter or a host GPU activity is
observed. Host-only rows retain `UNOBSERVED` timestamp support; a copy or wait
does not prove device timing support. An empty list means neither was observed. `AVAILABLE` means timestamps are
supported even when the event budget is full or no interval has completed.
`UNAVAILABLE` records missing timing support, and `MIXED` means the same backend
encountered different support states. `nodeTimingAvailable` means node timestamp
support was observed on at least one encountered device path, even if basic
detail was requested. It does not promise node coverage for every operation;
forward-node support does not imply backward or optimizer-node support.
`programTimingAvailable` reports observed program timing support.
`passIntervals`, `nodeIntervals`, `programIntervals` and `copyIntervals` count
separate device observation scopes. A program with an owning node increments
only the program count. `calibratedIntervals` and `boundedIntervals` describe
how many of those intervals include each kind of clock correlation.
`hostCopyCalls`, `hostSubmitCalls`, `hostWaitCalls` and `hostAwaits` count host
activities separately.
`unavailablePasses` counts passes without timestamp support;
`failedIntervals` counts measurement failures. `droppedEvents` includes capacity
loss and measurement failures, but excludes unsupported passes.

Chrome JSON format `volvoxai-trace/v6` keeps `detail`, `deviceTiming`, `memory`
and backend coverage in `otherData`. Calibrated GPU intervals appear as slices
on named queue tracks, positioned at the midpoint of their start-time range.
Their arguments retain both range endpoints and `clockMethod: "calibrated"`;
the plotted midpoint is an estimate. Bounded intervals remain instant
annotations with `deviceDurationNs`, `clockAligned: false` and
`clockMethod: "bounded"`. They do not claim an exact GPU position. Per-event JSON
nanosecond values and identifiers are decimal strings; Chrome's `ts` and `dur`
fields alone use the format's microseconds. External backend providers expose
host operations only.

### Following a submission, copy and completion

`NODES` detail also records engine-owned GPU copy calls, submissions and
completion waits. It requires no extra switch. `event.activity` distinguishes
`WORK`, `COPY`, `SUBMIT`, `WAIT` and `AWAIT`. A `WAIT` is a blocking host call;
`AWAIT` is asynchronous completion latency and does not mean the CPU was busy
or blocked. Chrome exports the latter on a separate process track. A copy adds
source/destination `MemorySpace` and the actual transferred byte count, including
alignment padding when applicable.

Use `event.queue.deviceId` and `queueId` to follow the observed backend queue.
Combine them with a nonzero `submissionId` to relate commands and GPU intervals
belonging to one engine submission or batch. A batch can contain several CUDA
or OpenGL driver calls; the ID is not a per-kernel invocation ID. Zero means
that no submission association was observed, as for a standalone transfer.
These opaque IDs are meaningful within the capture, are never native pointers,
and do not identify one physical GPU across different APIs.

| Backend | Host observations at NODES | Device clock placement and copy timing with `deviceTiming` |
| --- | --- | --- |
| CUDA | Pinned upload/download calls, device-to-device copy submission, kernel/Graph submission, stream/event/context completion waits | Event-relative elapsed time plus causal host bounds. Pinned upload/download transfers use events around their existing wait. Device-to-device copies currently have host observations only. |
| Desktop OpenGL | Buffer upload/download/copy, compute dispatch and `glFinish` | `GL_TIMESTAMP` host brackets before/after the measured pass calibrate its query clock. Copies inside a timed pass have device intervals; standalone copies retain host observations. |
| Vulkan | Buffer-copy command encoding, `vkQueueSubmit` and existing fence waits | `VK_EXT_calibrated_timestamps` with `CLOCK_MONOTONIC` when available; causal bounds otherwise. Copies inside a timed pass have device intervals. Mapped staging buffers are `HOST_VISIBLE_DEVICE`. |
| WebGPU | `writeBuffer`, snapshot-copy encoding, queue submission and asynchronous queue completion | Pass timestamps plus causal submission/completion bounds. Standard WebGPU has no host-clock calibration or standalone copy timestamps in this adapter; copy observations are host calls. |

A clock range is `device.correlation.earliestStartNs` through `latestStartNs`
on the capture-relative host clock. `CALIBRATED` uses backend clock samples;
`BOUNDED` uses submission-before/completion-after observations plus the device's
relative markers. The latter is a possible placement window, not an estimated
queue delay. Missing correlation means elapsed time was measured without a
usable placement. Host quantization, backend timestamp precision and browser
privacy reductions still apply. Calibration includes reported/bracketed
uncertainty and the spread between the two anchors. Unambiguous counter width,
causal consistency and a maximum one-second calibration window are checked;
unusable calibration falls back to bounds or leaves placement absent.
All intervals inside one native pass share its placement window plus their
measured device offsets. This preserves GPU ordering and nesting when the host
clock uncertainty is larger than an individual kernel or copy.
See the [Vulkan calibration contract](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_calibrated_timestamps.html)
and [OpenGL timer-query specification](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_timer_query.txt).

Filter `activity` before making summaries. A host copy call can contain a
blocking wait; both can overlap a GPU copy interval. Keep these observations
separate instead of adding them. The queue IDs support joining records, but
the export does not invent flow arrows or per-kernel associations from a
batch ID. Engine observation does not include driver-internal transfers,
external-library queues or a breakdown of device-side dependency stalls.

A GPU interval is elapsed time between device markers, not necessarily time
spent actively computing. It can include transfers, dependency waits and gaps
between submissions. A host call that waits for completion can overlap that
interval; adding both durations double-counts work. Device utilization and
driver-internal transfer/wait breakdown require additional measurements.
A native node may dispatch multiple programs. Program names and entry points
identify only engine-owned invocations, not a complete inventory of driver or
external-library kernels. Vulkan inserts GPU ordering dependencies around node
and program measurements; enabled profiling can reduce overlap.

With GPU timing enabled, WebGPU BASIC detail preserves existing compute-pass
boundaries. NODES detail uses
standard [compute-pass timestamp writes](https://gpuweb.github.io/types/interfaces/GPUComputePassTimestampWrites.html)
to split instrumented execution at program boundaries. A forward node spans
its first and last query markers, including all measured programs; its duration
is not calculated by summing program durations. Exhausting query capacity inside
a node invalidates that partial node measurement while preserving numerical
execution and completed program intervals. It shares a query set and two buffers
across the batch, resolves in the same
command encoder, and preserves submission count. This mode reports
`splitsPasses: true`; its measurements include the effects of the changed pass
boundaries. It does not manufacture an additional whole-forward interval from
a sum of node durations. The optional timestamp feature is requested when a
device is acquired. Devices without that feature continue with host observations.
Timestamp precision may be reduced for privacy, producing valid zero-length
intervals; reversed results are failures, not zero durations.

### Reading training and program detail

Keep `detail: NODES` for both inference and training. No extra switches or
profiling operations are needed. CPU/WASM records actual backward node calls,
loss, gradient processing and optimizer host work. Native OpenGL/Vulkan backward
uses GPU programs, while loss and optimizer execute on CPU; those host events
therefore use the CPU backend. CUDA and WebGPU programs carry all five phases.
Backward GPU commands carry their originating node; loss and optimizer programs
carry a target tensor when available. Planning and validation emit no numerical
node/program observations.

To rank complete nodes, select `event.node && !event.program`. To inspect
programs, select `event.program` and group by backend, phase, name and entry point.
Use lineage plus node metadata to relate a program to its owner. Host command
spans and device program intervals are independent observations; there is no
one-to-one invocation ID across the two domains. A training command can lower
to multiple native kernel launches. Do not add a parent node/phase duration to
its programs, or combine host and GPU elapsed time into a total.

Chrome JSON uses `host.node`, `host.work`, `host.program`, `device.interval` and
`device.program` categories, with the same phase/program/tensor metadata.
Activity categories add `host.copy`, `device.copy`, `host.submit`, `host.wait`
and `host.await`. Asynchronous waits use matched Chrome `b`/`e` events with
independent IDs, so overlapping completion observations remain valid in
Perfetto and do not imply occupied CPU threads.
One typed wait becomes a JSON start/end pair; JSON metadata and memory counters
also add entries. Use `TraceInfo.eventCount` to count collected observations.
Collection retains bounded copies of names, so a trace can outlive its model,
training plan and device context. Truncation and capacity loss remain explicit.
Training origin sidecars are allocated only for NODES collection, bounded to
1,024 entries per plan. Query resources are allocated only when GPU timing is
requested; CUDA can retain them in its instrumented replay cache until retirement.

Native query batches contain at most 1,024 intervals and are additionally
limited by available record capacity. On the first device-timing request,
WebGPU prepares and asynchronously validates a pool of four query batches,
each with 1,024 intervals and two 16 KiB buffers. This preparation never
acquires a GPU for a CPU-only runtime. If the GPU is prepared later, it also
prepares the requested timestamp pool. Invalid timing resources are discarded
before any numerical command is encoded. Pool exhaustion or allocation failure
loses timing observations while execution continues normally. Batches return
to the pool after mapping and the last ticket release; idle storage lasts until
host close. Query slots also consume bounded trace record storage.
`StopTrace` closes admission and may return `DRAINING` until polling resolves
these tickets and asynchronous completion observations. Poll with `GetTrace`.
WebGPU retains at most 128 completion-observation tickets per bridge; exhausted
capacity increments loss without adding a queue wait. Completed tickets also
drain at the next captured operation boundary, so ordinary long captures do
not require applications to poll during collection. `activeOperations` counts
active host scopes and pending asynchronous host completions;
`pendingDeviceIntervals` counts device observations separately. Release
safely retires their GPU storage even if inference results have already been
released. No device promise captures a WASM destination pointer.

CUDA keeps one optional instrumented graph plan separate from its four ordinary
replay cache entries. Capture uses external event nodes, so replay updates the
measurements. Events live until that plan is replaced or its context is released;
they are never recorded or queried by ordinary execution. The first measured
execution can include observation/capture setup; warm the measured route when
comparing steady-state intervals.

For example, a host could spend 0.2 ms encoding commands and return while the GPU
takes another 3 ms. The host node reports 0.2 ms. If a backend waits before
returning, the host duration includes that wait. Neither observation alone
identifies the kernel's device execution time. The caller can measure complete
latency by waiting for `GetResult` to become `READY` and reading the output;
that also includes polling, readback and transport costs.

Events refer to the executable schedule and named output. Eliminated nodes have
no independent duration. Fused work is not divided into invented source-node
timings. Do not add parent operation durations to their contained nodes. A
zero-duration observation can mean the work was shorter than the clock's
resolution; it is different from an absent measurement.

`CompileModel.compileTimeNs` measures host compilation. Execution responses and
`GetResult` expose `metrics.hostTimeNs` and `metrics.outputBytes`. The former
ends at output submission and excludes asynchronous completion. The latter is
logical output payload, not RSS or total allocated memory. Measure the complete
application workflow separately when preprocessing, readback and rendering
matter.

The two receipt examples collect only when their profiling checkbox is selected
for the next run, including the memory history described below. Their node tables aggregate typed host node records;
device elapsed values are separate observations and are never added to host time.
Unobserved nodes show `—`. VQA decoder iterations remain separate in the
timeline even though the table sums their durations.

## Inspect memory

Memory inspection does not require a trace:

```js
const observed = await profiling.getMemorySnapshot(new pb.GetMemorySnapshotRequest({
  contextId,
  includeProcess: true,
}));
console.log(observed.snapshot);
```

Choose one scope: `process: true`, `runtimeId`, or `contextId`. Process snapshots
report RSS and process-lifetime peak RSS where the OS provides them. Unsupported
values have `UNAVAILABLE` and no byte value. In particular, browser WASM cannot
measure process RSS.

Runtime observations include the existing unconsumed-result budget counter.
Context observations include host arena capacity, `retained_result_capacity`
and `idle_result_capacity`. Idle capacity is a subset of retained capacity;
these counters must not be added together. These
are scoped accounting measurements, not an inventory of physical allocations.
Resource inventories explicitly remain `PARTIAL`; an empty partial inventory
must never be displayed as zero total memory. Process envelopes overlap these
counters and must not be summed with them. Sampling reports an observation
interval rather than claiming an atomic view across threads and the OS. Direct
queries use host monotonic nanoseconds; snapshots in trace pages use the same
capture-relative origin as the events.

Enable `memory: true` on `StartTrace` for allocation history. `event.memory`
contains an action (`EXISTING`, `ALLOCATE`, or `FREE`), timestamp, opaque
capture-local allocation ID, allocator name, memory space, requested bytes and
tracked live capacity after that event. No pointer or GPU address is exposed.
Address reuse receives a new identity. Aliases and reuse of a pooled buffer do
not create an allocation; returning a buffer to a pool does not free it.

`GetTrace`/`StopTrace` return `allocators` with `existingBytes`, `allocatedBytes`,
`freedBytes`, `liveBytes` and `peakBytes`. For a complete accounting stream,
`live = existing + allocated - freed`. Peak is the greatest observed live
capacity, including transient overlap while a buffer grows. This is requested
allocator/API storage, not OS RSS, physical VRAM or live tensor payload.
Chrome export includes allocation instants and capacity counter tracks, with
the same summaries in `otherData.allocators`.

| Allocator | Observed storage |
| --- | --- |
| `host.arena` | Static activation buffers and dynamic arena reservation/growth; native heap or WASM linear memory |
| `host.result` | Owned output copies and retained host result pool capacities |
| `cuda.graph` | Owned graph slots allocated through the CUDA Driver API |
| `opengl.graph` | Owned graph/training buffers and normalization scratch created through the graph allocator |
| `vulkan.graph` | Graph arena and staging backing `VkDeviceMemory` allocations; slot aliases are not counted again |
| `cuda.result`, `opengl.result`, `vulkan.result` | Retained output and feedback snapshot pools, including idle cached storage |
| `webgpu.buffer` | Bridge-owned `GPUBuffer`s, including graph, uniform, readback and optional timestamp buffers |

Inventories explicitly remain `PARTIAL`. An owner is inventoried when it first
participates in captured work, before execution shape binding. `EXISTING` does
not claim the allocation occurred during the capture. `observationStartNs`
names the first observed transition for that allocator. Collection ends at
`StopTrace`; later accepted host/device intervals may still drain. If a deferred
free is relevant, wait for its ordinary completion before stopping the trace.
WebGPU records release when `GPUBuffer.destroy()` runs, not when retirement is
requested. API release still does not prove when a driver reclaimed physical
memory. Backend release failures weaken accounting instead of inventing a free.

`scope: RUNTIME` aggregates observed owners of the captured runtime.
`scope: BACKEND_SHARED` is used for WebGPU because one host bridge can serve
several runtimes. It includes all buffers in that bridge, including concurrent
work. Do not add shared summaries from simultaneous runtime captures.
Unvisited owners, earlier ordinary output copies, tensor/weight heaps outside
the listed allocators, metadata, driver-private storage and some training
workspaces are not covered. A partial allocator peak cannot prove a whole-engine
memory peak or fully diagnose every out-of-memory failure.

The collector reserves a bounded live-identity table within `capacityBytes`.
If only event storage fills, counters continue and `droppedEvents` reports the
missing records. If identities cannot fit or release cannot be observed,
`accountingComplete` becomes false: the counters then describe the tracked
subset and cannot establish the complete peak. WebGPU's bridge identity map
is separately bounded to at most 16,384 entries per capture and 64 captures;
its JavaScript bookkeeping is outside the C collector byte budget. Completed
records retain neither allocations nor owners; a small weak observer can remain
with a retained allocation until that allocation or owner is released.

The same memory option also records two process observations, at start and
after accepted host work/device observations drain. It installs no periodic
sampler. The start sample includes the trace's own storage. The process RSS
peak covers the process lifetime, independently of allocator capture peaks.

Compilation bounds are separate: `CompileModel.memoryBounds` identifies the
logical graph, admitted shape domain and independent proved maxima. Bounds are
available without enabling profiling and do not describe current memory usage.

## Performance and contracts

The ordinary numerical loop contains no new per-node collector calls or clocks.
The execution boundary selects the instrumented loop only for an active trace.
Disabled collection allocates no trace journal, samples no clocks for memory,
and enumerates no allocations. Optional hooks run at allocation/free boundaries,
with no additional device waits or per-node memory work. The GPU adapters create, write and read no timing queries
or calibration samples on this path. Native results are read after existing completion waits; WebGPU
resolves in the same submission and maps asynchronously. Enabled collection
changes timing through its own work; compare workloads using an unprofiled run
too. Vulkan discovers and enables optional clock-calibration support once when
creating its shared device, even if collection is disabled; actual calibration
samples are taken only during a timed capture.

An absence of additional instrumentation does not guarantee identical wall
latency across builds. Small regressions remain possible, especially for short
WASM workloads affected by compilation tiering and warmup. Compare repeated,
alternating unprofiled runs against a matching baseline, include an
identical-build control, and measure startup separately from steady state.
Enabling node or device timing can add substantial overhead to small workloads.
The [testing guide](testing.md#profiling-checks) describes the maintained checks
and performance tools.

All operations are declared in [the public schema](../proto/volvoxai.proto).
The generated [inference contract](generated/api-contract.inference.md) and
[full contract](generated/api-contract.full.md) specify limits and handle
ownership. The same services are available through generated C, TypeScript and
Python clients.
