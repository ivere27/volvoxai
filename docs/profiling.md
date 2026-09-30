# Profiling and memory

For intermediate tensor values and step-by-step execution, use
[node debugging](debugging.md). It shares execution-plan identities with
profiling and has a separate session because capture changes execution timing.

VolvoxAI profiles one unified edge AI engine across native C/CPU, native GPU,
WASM and WebGPU. Use a trace to connect execution time, memory ownership and
CPU/GPU resource observations. Traces contain actual
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
plus a viewer for the canonical execution plans (`GetTracePlan`); Chrome timelines do not render
that model topology themselves.

## Capture a trace

Collection is disabled by default. Start after warmup to measure steady state,
or before loading a model to include loading and compilation. A runtime accepts
one collecting or draining trace; completed traces remain readable independently.

```js
import { VxProfilingServiceClient, pb } from './volvoxai.js';

const profiling = new VxProfilingServiceClient(host);
const trace = await profiling.startTrace(new pb.StartTraceRequest({
  runtimeId,
  options: new pb.TraceOptions({
    detail: pb.TraceDetail.TRACE_DETAIL_NODES,
    capacityBytes: 4n * 1024n * 1024n,
    deviceTiming: true,
    memory: true,
    utilization: true,
    executionPlans: true,
  }),
}));
const ref = new pb.TraceRef({traceId: trace.traceId});

try {
  // Run your existing inference, training or decode calls here.
  await runWorkload();

  // StopTrace replies when the trace is READY: no drain polling is needed.
  await profiling.stopTrace(ref, {timeoutMs: 30000});

  const chunks = [];
  let pageToken = '';
  do {
    const page = await profiling.exportChromeTrace(new pb.ExportChromeTraceRequest({
      traceId: trace.traceId, pageToken,
    }));
    chunks.push(page.data);
    pageToken = page.nextPageToken;
  } while (pageToken);
  const file = new Blob(chunks, {type: 'application/json'});
  // Save `file` using your application's ordinary download UI.
} finally {
  await profiling.releaseTrace(ref);
}
```

In Python, `InferenceSession.trace()` is a scope like `torch.profiler.profile`:

```python
with vx.InferenceSession("model/") as model:
    with model.trace(detail="nodes", memory=True) as trace:
        with trace.annotate("preprocess"):
            frame = prepare(image)
        model.run(frame)
    trace.export_chrome_trace("trace.json")
    slow = sorted((e for e in trace.events if e.HasField("node")),
                  key=lambda e: e.host.duration_ns, reverse=True)
```

`StopTrace` stops admission and replies after accepted host operations and
pending device timestamps drain. Cancelling or timing out the call stops only
the wait; `GetTrace` is the non-blocking status. `ReleaseTrace` retires any
pending timestamp queries and completes a waiting `StopTrace`.

Read the typed records with `ListTraceEvents` and
`ListTraceResourceSnapshots`, which page with `page_size`, `page_token` and
`next_page_token` (Google AIP-158): start with an empty token and stop when
`next_page_token` is empty. Tokens are opaque; pages are repeatable and do not
consume events. `GetTracePlan` returns one execution plan by the `plan_id`
that events carry. Listing a trace that is not READY returns `BUSY`. Releasing
a trace closes admission without cancelling inference. Models, contexts and
result tensors are not retained by completed records.

`AnnotateTrace` records an application range, like
`torch.profiler.record_function` or `jax.profiler.TraceAnnotation`, with
start and end times from `VxPlatformService.GetMonotonicTime`. Annotations
appear as `TRACE_ACTIVITY_ANNOTATION` events and as `user.annotation` slices
on an "Application annotations" Chrome track.

The byte budget is allocated at start. Exhaustion drops additional events and
increments `events.dropped`; inference continues. Always display this count when
presenting aggregates. `ExportChromeTrace` serializes one page of source events
per call; `page_size` defaults to 128 events and accepts 1–1,024. Concatenate
the returned UTF-8 fragments from an empty `page_token` until `next_page_token`
is empty. The engine keeps no full JSON copy or export cursor: repeating a
request returns the same bytes, and scratch memory depends on the page size.
Fixed trace metadata appears in the first/last page. Event names and
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
| `memory: true` | Bounded allocation/free history, allocator peaks and resource memory samples |
| `utilization: true` | Process CPU time and supported hardware GPU utilization at operation boundaries |
| `executionPlans: true` | Immutable concrete tensor/schedule metadata and known allocation placement |

`startTrace({runtimeId})` therefore records basic host work. Request `NODES`
to investigate executable nodes, and enable `deviceTiming` to include GPU time.
There is no separate device granularity option. `memory` defaults to
false; `capacityBytes` controls storage, with a 4 MiB default. Detailed copy
and completion records consume that same budget. Increase it for long captures
and check `events.dropped`, `resourceSnapshots.dropped` and `plans.dropped` before
treating a capture as complete.

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
observed. Host-only rows retain `UNSPECIFIED` timestamp support; a copy or wait
does not prove device timing support. An empty list means neither was observed. `AVAILABLE` means timestamps are
supported even when the event budget is full or no interval has completed.
`UNSUPPORTED` records missing timing support, and `MIXED` means the same backend
encountered different support states. `nodeTimingAvailable` means node timestamp
support was observed on at least one encountered device path, even if basic
detail was requested. It does not promise node coverage for every operation;
forward-node support does not imply backward or optimizer-node support.
`programTimingAvailable` reports observed program timing support.
`deviceIntervals.passes`, `.nodes`, `.programs` and `.copies` count
separate device observation scopes. A program with an owning node increments
only the program count. `deviceIntervals.calibrated` and `.bounded` describe
how many of those intervals include each kind of clock correlation.
`hostActivities.copies`, `.submits`, `.synchronizations` and `.completions` count host
activities separately. `deviceIntervals.unsupportedPasses` counts passes
without timestamp support; `deviceIntervals.failed` counts measurement
failures. `events.dropped` includes capacity loss and measurement failures,
but excludes unsupported passes.

Chrome JSON format `volvoxai-trace/v8` keeps `detail`, `deviceTiming`, `memory`
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
`COMPUTE`, `COPY`, `SUBMIT`, `SYNCHRONIZE` and `COMPLETION` (the CUPTI/Kineto
kernel, memcpy, launch and synchronization vocabulary). A `SYNCHRONIZE` is a
blocking host call; `COMPLETION` is asynchronous completion latency (a fence or
`onSubmittedWorkDone`) and does not mean the CPU was busy
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
interval; adding both durations double-counts work. The separate resource samples report supported hardware utilization.
Driver-internal transfer/wait breakdown remains outside these observations.
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
Activity categories add `host.copy`, `device.copy`, `host.submit`, `host.synchronize`
and `host.completion`. Asynchronous waits use matched Chrome `b`/`e` events with
independent IDs, so overlapping completion observations remain valid in
Perfetto and do not imply occupied CPU threads.
One typed wait becomes a JSON start/end pair; JSON metadata and memory counters
also add entries. Use `TraceInfo.events.count` to count collected observations.
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

`CompiledModelHandle.metrics.hostTimeNs` measures host compilation. Execution responses and
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

## Inspect memory and CPU/GPU load

VolvoxAI is a unified edge AI engine: native C/CPU and native GPU backends,
WASM CPU and WebGPU share this profiling contract. Browser restrictions change
what can be observed, not what a field means.

A resource snapshot does not require an active trace:

```js
const {snapshot} = await profiling.getResourceSnapshot(
  new pb.GetResourceSnapshotRequest({contextId, includeDevice: true}),
);
console.log(snapshot.memory.allocations); // Known owned storage and purpose.
console.log(snapshot.wasm);               // This module's linear heap.
console.log(snapshot.process);            // OS process RSS and lifetime peak.
console.log(snapshot.cpu);                // Cumulative process CPU time.
console.log(snapshot.gpu);                // Hardware-wide driver observations.
```

Select exactly one inventory scope: `module` (an `Empty` message), `runtimeId`, or
`contextId`. Module means every known owner in this engine module. A context includes the model and
compiled resources it retains. Shared backing allocations appear once in that
inventory. Runtime/host inventories also inspect the private context reused by
`Run` and the scheduler, attributing its storage to the compiled model that
owns it. A busy private route is skipped and marks the inventory truncated.
`maxAllocations` bounds the list; `truncated` reports a limit or an
owner that could not be inspected. Inventories are `PARTIAL`: omitted allocator,
VFS, driver, provider and training storage must not be interpreted as zero.
Each listed allocation has its lifetime owner, purpose, allocator, memory space
and capacity. Snapshots expose no addresses and do not retain tensors.

The observation interval covers independent readings, not an atomic view of
the process and every device. External values retain their scope even when the
inventory selects one context. Sampling starts before allocating the temporary
inventory response, so its buffers are not attributed to model storage.

| Measurement | Scope and meaning | Availability |
| --- | --- | --- |
| Allocation capacity | Known requested/API storage: weights, packed weights, arena, scratch, metadata and retained result buffers | Partial native CPU/WASM, native GPU and WebGPU inventories; GPU coverage depends on backend |
| WASM heap partition | One module's linear memory, allocator live/free blocks, headers, module prefix and page slack | WASM, in both browser profiles |
| Process RSS / peak RSS | Entire OS process; peak covers its lifetime | Supported native OS samplers; unavailable in browser WASM |
| Process CPU time | Cumulative user + kernel CPU time across all process threads | Native POSIX/Windows; unavailable in browser WASM |
| GPU activity / memory | Driver-reported device-wide activity and allocated VRAM, including other applications | Linux NVIDIA through optional NVML; other platforms/devices explicitly unsupported by this sampler |
| Device elapsed time | Engine passes, nodes and programs between timestamp markers | CUDA, desktop OpenGL, Vulkan and WebGPU according to trace coverage |

`AVAILABLE` contains a value, including a real zero. `UNSUPPORTED` means no
supported measurement route; `NOT_COLLECTED` means the option was off; `FAILED`
means an attempted observation failed. None means zero. Check status before
reading a value. For optional scalar presence in JavaScript, use the generated
message's `toJson()` result; a scalar property alone can have its default zero
even when the field was absent. Device sampling is
opt-in and loads no compute backend or CUDA context. A driver sample can cover
work before the request; an absent `samplingWindowNs` means its window length
was not reported. Opaque queue IDs in timing events are not driver GPU UUIDs.

### Is all reserved memory in use?

The WASM snapshot gives an exact allocator partition at its observation time:

```text
linearBytes = allocatedBytes + freeBytes + allocatorMetadataBytes
            + modulePrefixBytes + pageSlackBytes + untrackedBytes
```

`freeBytes` is reusable by malloc; `largestFreeBlockBytes` helps identify
fragmentation. Free capacity may be split across many blocks. `untrackedBytes`
accounts for raw heap reservations outside malloc. Free ranges inside an
allocated activation arena are included in `allocatedBytes`, not `freeBytes`.
Linear memory retains its high-water size and cannot be made smaller by freeing
an individual buffer. Neither linear capacity nor malloc's allocated bytes is
process RSS.

For arena use, capture `executionPlans: true`. The plan lists concrete tensor
shapes, dtypes, logical bytes, schedule lifetimes and known allocation offsets.
An alias refers to the same allocation and adds no second backing buffer.
The offline report unions overlapping live ranges at each step, so reused or
aliased tensors are not counted twice. Its peak is **derived from the plan**,
not sampled physical RAM. Missing placement or lifetime remains unknown. The
difference between arena capacity and this peak is not automatically removable:
alignment, conservative lifetimes, cached shapes and unobserved scratch can
still require storage.

`unconsumedResultBytes` is the runtime's result budget accounting.
`retainedResultCapacityBytes` includes both live retained results and idle pool
storage; `idleResultCapacityBytes` is its reusable subset. These values overlap
allocations. Do not add them to allocation capacities, WASM memory, RSS or VRAM.

### CPU and GPU utilization

Take two supported CPU observations from the same process:

```text
average occupied CPU cores = delta(processTimeNs) / delta(monotonic host time)
```

A value of 1 means one fully occupied core; 2.5 means an average of 2.5 cores.
This includes all process threads and profiler work. It is not per-model CPU
load. Use the observation interval to judge short-window uncertainty. Host
operation duration alone cannot provide CPU utilization because it can include
waits and descheduling.

GPU `gpuUtilization` and `memoryUtilization` (nvidia-smi's `utilization.gpu`
and `utilization.memory`) are driver activity fractions
in [0, 1]. Memory activity is memory-interface busy time, not used VRAM divided
by capacity. Compute activity is not SM occupancy or a percentage of peak FLOPS.
Timestamp-query durations, including WebGPU timestamps, do not measure global
GPU utilization. Keep hardware telemetry separate from engine work intervals.
Chrome export presents them as separate resource counter tracks.

With `memory` or `utilization` enabled, traces sample at start, after draining and
at encountered operation boundaries. `sampleIntervalNs` sets minimum spacing
(default 100 ms), not a guaranteed periodic rate. There is no sampler thread,
per-node memory sampling or additional GPU synchronization. Short operations
may have only the two boundary samples; long individual kernels have no samples
inside their duration. `utilization` additionally enables CPU/GPU telemetry.
The collector reserves its last sample slot for completion and reports dropped
samples when the bounded storage fills. `collectorBytes` describes its own
requested storage; sampling and export also have costs.

## Summarize nodes, operators and calls

`GetTraceSummary` aggregates a READY trace in the engine, so C, TypeScript,
Python and agents read the same table without reimplementing the rules:

```js
const {rows} = await profiling.getTraceSummary(new pb.GetTraceSummaryRequest({
  traceId, groupBy: pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_NODE, maxRows: 20,
}));
for (const row of rows)
  console.log(row.scheduleIndex, row.name, row.count, Number(row.totalNs) / 1e6, row.share);
```

```python
with model.trace(detail="nodes", execution_plans=True) as trace:
    model.run(frame)
print(trace.table("operator"))   # like torch.profiler's key_averages().table()
slow = trace.summary("node").rows[0]
```

| `groupBy` | One row per | Use it to answer |
| --- | --- | --- |
| `NODE` | Executable node of one plan, host or device | Which nodes cost the most? |
| `OPERATOR` | Backend, phase and operator type | Which operator types dominate? |
| `OPERATION` | Public call such as `Execute` or `TrainStep`, per context | How variable are repeated calls? |
| `ACTIVITY` | Copy, submission, blocking wait or completion | Are transfers or synchronization expensive? |

Each row has the call count, total, minimum, median, nearest-rank p95 and
maximum duration, and `share`: its fraction of the total among rows with the
same model, context, plan, backend, phase and time domain. Rows are ordered by
total time. The same rules as the analysis tables apply: host and device time
are separate rows, program intervals are never added to their node, and events
with truncated metadata are excluded and counted in `excludedEvents`. Check
`events.dropped` before trusting totals.

With `executionPlans: true`, every plan step carries a `StepCost`: the work
one call implies at its concrete shapes. `multiplyAccumulates` counts matrix
products, convolutions and attention, whatever the weight layout;
`elementwiseOperations` counts one operation per output element of other
arithmetic, and `transcendentalOperations` counts exp/tanh/erf-class functions
separately, as XLA's cost analysis does. `inputBytes` and `outputBytes` are
the logical sizes the step reads (including weights) and writes. `status` is
`EXACT` for closed-form counts, `ESTIMATED` for attention (masked keys make the
real work smaller), packed weights, normalization and windowed reductions,
and `UNKNOWN` when there is no model for the operator. These are counts, not
measurements, and a fused step includes its peer's work.

Node rows join the cost with time: `achievedFlopsPerSecond` is two operations
per multiply-accumulate divided by the row's time, and
`achievedBytesPerSecond` uses the step's bytes. They are present only when the
time measures that work: device intervals, or host nodes on CPU and WASM. A
host node on a GPU backend only prepares and submits commands, and a decode
step runs only part of its plan's rows, so neither reports a rate. They are
achieved rates, not utilization: the engine does not know a device's peak.
Compare them with your hardware's specification to place a node on a
roofline.

## Use Nsight, RenderDoc and other GPU tools

Vendor tools explain why a kernel is slow; the trace tells you which node it
belongs to. `TraceOptions.external` connects them:

```python
with model.trace(detail="nodes", external_annotations=True) as trace:
    model.run(frame)
```

With `annotations`, each traced call becomes an NVTX range in the `VolvoxAI`
domain, and with `NODES` detail each node and training phase becomes a nested
range named like the trace's events (`12 Linear -> h1`, with the schedule
index as payload and the phase as category). Backward nodes are ranges too
(`backward 12 Linear -> h1`), including on GPU Trainers, so Nsight Systems
attributes each backward kernel to its node (`nsys stats --report nvtx_kern_sum`). On Vulkan each node's commands
are wrapped in a `VK_EXT_debug_utils` label, on OpenGL in a `KHR_debug`
group, and on WebGPU each dispatch in a debug group inside a pass labeled
with the node, so RenderDoc, vendor GPU tools and WebGPU validation messages
show the same names. Pipelines and programs are always named after their
shaders when those extensions exist, and WebGPU buffers carry fixed labels
(`vx.tensor`, `vx.weight`, `vx.params`, `vx.readback`, `vx.timestamp.*`).
Ordinary execution never runs this code; only operations admitted while an
annotating trace is collecting are labeled.

| Mechanism | Where | Tools |
| --- | --- | --- |
| NVTX ranges | Native builds with a GPU backend, every backend | Nsight Systems (`nsys profile -t nvtx,cuda,vulkan,opengl`), Nsight Compute (`--nvtx --nvtx-include`) |
| `VK_EXT_debug_utils` labels | Vulkan, when the loader offers the extension | RenderDoc, Nsight Graphics, vendor tools |
| `KHR_debug` groups | OpenGL 4.3+ or GLES 3.2 | RenderDoc, Nsight Graphics, vendor tools |
| WebGPU debug groups and pass labels | WebGPU in the full WASM build | Browser WebGPU error messages; native captures of Dawn or wgpu (Deno), which pass labels to Vulkan/Metal/D3D12 |
| RenderDoc capture | Vulkan/OpenGL builds, RenderDoc already injected | RenderDoc |

The whole trace is also one NVTX range registered as `VolvoxAI trace`, from
StartTrace to StopTrace, which Nsight Systems can use as a capture range.
Headless compute has no present call to delimit a frame, so `capture: true`
asks a RenderDoc instance that is already injected into the process to
capture from the first traced call until StopTrace. The engine never loads
RenderDoc itself. Launch the program under RenderDoc instead:

```sh
renderdoccmd vulkanlayer --register --user   # once, for Vulkan captures
renderdoccmd capture -w -c /tmp/volvoxai python app.py
```

The capture holds the traced calls only, with each node's commands inside its
label and pipelines named after their shaders.

`TraceInfo.external` has one row per attempted mechanism with the number of
ranges emitted. `AVAILABLE` means the engine emitted them; it does not prove
that a tool was attached and recorded them. `UNSUPPORTED` means the mechanism
does not exist in this build or driver, for example NVTX in WASM.

Browser builds can show a trace in Chrome DevTools instead.
`TraceInfo.captureOriginNs` is the platform monotonic time of the trace's zero,
the same clock as `performance.now()` in the WASM host, so a finished trace can
be replayed into the page's Performance timeline without affecting what was
measured. `measureEngineTrace` in
[examples/common/EngineTrace.js](../examples/common/EngineTrace.js) does this
with User Timing measures on a `VolvoxAI` track; record a Performance profile
while the page calls it.

## Analyze a capture with people or AI

Use `ListTraceEvents`, `ListTraceResourceSnapshots` and `GetTracePlan` to read
the same immutable capture. `TraceInfo` reports the count and dropped count for
each kind (`events`, `resourceSnapshots`, `plans`). Events refer to a plan by
`planId`; plans retain model/context lineage and a concrete shape signature.
Metadata is copied once per captured layout/storage generation and survives
model release.

Plan steps identify every source model node they execute, including a fused
peer; `sourceMappingComplete` is false only when a peer cannot be identified.
Fusion does not invent per-source timings. `sourceMappingComplete`,
`placementComplete` and `metadataTruncated` make incomplete metadata explicit.
CPU/WASM expose host tensor placement. Native GPU and WebGPU retain available
host/logical metadata but do not claim complete device tensor placement.
The plan route currently covers inference contexts; training and external
providers still rely on their existing event attribution. A plan allocation's
optional `traceAllocationId` joins recorded allocator history; snapshot-local
allocation IDs alone never join different snapshots.

The repository includes an offline collector and deterministic JSON analyzer.
After stopping and draining a trace, a Node development script can save the
complete canonical artifact:

```js
import {writeFile} from 'node:fs/promises';
import {collectProfile} from './tools/profile_report.mjs';

const artifact = await collectProfile(profiling, pb, trace.traceId);
await writeFile('capture.json', JSON.stringify(artifact, null, 2));
```

```bash
node tools/profile_report.mjs analyze capture.json report.json
node tools/profile_report.mjs compare before.json after.json comparison.json
```

The report contains observed heap peaks, process CPU use, driver GPU readings,
per-plan arena range analysis, large tensors, timing statistics and coverage.
Timing rows keep model/context, concrete plan, phase and host/device domains
separate while aggregating repeated calls within each row.
Each analysis points to its source records with JSON paths. Comparison checks
graph, concrete shape and backend metadata; callers must also verify equal
weights, inputs, outputs, warmup and environment. Samples cannot prove a peak
between observations, and partial inventories cannot prove the absence of a leak.

The typed records are the canonical representation. Generated message
`toJson()`/Python `to_dict()` produces ProtoJSON without another runtime JSON
contract. Chrome JSON is the visual timeline projection; it includes resource
counter tracks and coverage, while typed records retain the full plan and
availability metadata for analysis.

## Follow allocation lifetimes

Enable `memory: true` on `StartTrace` for allocation history. `event.memory`
contains an action (`EXISTING`, `ALLOCATE`, or `FREE`), timestamp, opaque
capture-local allocation ID, purpose, allocator name, memory space, requested
bytes and tracked live capacity after that event. No pointer or GPU address is
exposed. Address reuse receives a new identity. Aliases and reuse of a pooled
buffer do not create an allocation; returning a buffer to a pool does not free it.

`GetTrace`/`StopTrace` return allocator summaries with `existingBytes`,
`allocatedBytes`, `freedBytes`, `liveBytes` and `peakBytes`. With complete
accounting, `live = existing + allocated - freed`. Peak includes transient
observed overlap while a buffer grows. These are requested/API capacities,
not OS RSS, physical VRAM or live tensor payload. Chrome export includes
allocation instants and capacity counters with the same summaries.

| Allocator history | Observed storage |
| --- | --- |
| `host.arena` | Static activation buffers and dynamic arena reservation/growth |
| `host.result` | Owned output copies and retained host result pools |
| `cuda.graph` | Graph slots allocated through the CUDA Driver API |
| `opengl.graph` | Graph/training buffers and normalization scratch |
| `vulkan.graph` | Graph arena and staging backing allocations, excluding slot aliases |
| `cuda.result`, `opengl.result`, `vulkan.result`, `metal.result` | Retained output/feedback snapshot pools |
| `webgpu.buffer` | Bridge-owned graph, uniform, readback and optional timestamp buffers |

An owner is inventoried when it first participates in captured work, before
execution shape binding. `EXISTING` does not claim the allocation occurred
during the capture. `observationStartNs` marks the first observed transition.
Allocation admission ends at `StopTrace`; accepted timing observations may
still drain. Wait for ordinary deferred release completion before stopping if
that release matters. WebGPU records `GPUBuffer.destroy()`, which does not prove
when a driver reclaimed physical memory. Release failures weaken accounting.

`RUNTIME` aggregates observed owners of that runtime. `BACKEND_SHARED` is used
for the WebGPU bridge and includes its other runtimes. Never add shared
summaries from simultaneous captures. Event history and on-demand inventory
have different coverage: the inventory additionally lists known weights,
packed weights and metadata but does not retroactively create their lifetimes.
Unvisited owners, VFS storage, providers, driver-private storage and some
training workspaces remain outside the inventory.

If event storage fills, counters continue and `events.dropped` reports the loss.
If allocation identities cannot fit, `accountingComplete` becomes false and
counters describe only the tracked subset. WebGPU's bridge identity map is
separately bounded to 16,384 entries per capture and 64 captures; its JavaScript
bookkeeping is outside the C collector budget. Small weak observers can outlive
the trace, but they retain neither event buffers nor tensor data.

Compilation bounds remain independent: `CompileModel.memoryBounds` identifies
the graph, admitted shape domain and proved maxima. They do not describe current
memory usage and require no active profiling session.

## Performance and contracts

The ordinary numerical loop contains no new per-node collector calls or clocks.
The execution boundary selects the instrumented loop only for an active trace.
Disabled collection allocates no trace journal, samples no clocks for memory,
collects no CPU/GPU usage or process RSS samples, and enumerates no allocations.
Resource snapshots run only when explicitly requested. Optional hooks run at allocation/free boundaries,
with no additional device waits or per-node memory work. The GPU adapters create, write and read no timing queries
or calibration samples on this path. Native results are read after existing completion waits; WebGPU
resolves in the same submission and maps asynchronously. Enabled collection
changes timing through its own work; compare workloads using an unprofiled run
too. Vulkan discovers and enables optional clock-calibration support once when
creating its shared device, even if collection is disabled; actual calibration
samples are taken only during a timed capture.

Disabled does not mean that the profiling feature occupies zero bytes or executes
zero instructions: fixed bookkeeping fields and admission/allocation guards
remain. There is no background sampler. `StopTrace` stops new collection but
preserves the captured data for reading; call `ReleaseTrace` when finished.
After release, CUDA may retain its instrumented replay cache until the context
retires, WebGPU retains its bounded timestamp pool until the host closes, and
the optional NVML library remains loaded until the engine module unloads.
These caches are inactive during ordinary inference. Small weak memory-observer
owners survive until their allocation owners retire. WASM linear memory also
keeps its high-water size after freed blocks become reusable. An application
that never starts a trace or requests a resource snapshot creates none of these
profiling caches or journals.

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
