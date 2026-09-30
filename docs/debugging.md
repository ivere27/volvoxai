# Inspect a graph one node at a time

VolvoxAI is a unified edge AI engine for native C/CPU, native GPU, WASM CPU and
WebGPU. Use node debugging to find where a model starts producing unexpected
values: inspect a node's inputs, execute it, then inspect its outputs. The
service belongs to the **full** profile (`volvoxai` / `FullEngineHost`,
`native/volvoxai`, Python's default `libvolvoxai`); inference builds carry
profiling but not debugging, to keep them small.

A debug session has one of three targets:

- **forward** runs one forward pass of a compiled model in a private context.
  It shares immutable model weights with its compiled model, but owns its
  inputs, intermediate buffers and position. An ordinary inference can run
  while this session is paused. Releasing the model or compiled-model handle
  does not invalidate an existing session.
- **decode_prefill / decode_step** attaches to an existing decode context and
  runs its next `DecodePrefill` or `DecodeStep` one node at a time, on the real
  KV cache. At every stop you can read each KV cache in token order. See
  [Debug a decode step and its KV cache](#debug-a-decode-step-and-its-kv-cache).
- **train_step** attaches to a Trainer and runs its next `TrainStep` one stage
  at a time: forward nodes, the loss, backward nodes, gradient accumulation
  and optimizer updates, with gradients as observable tensors. See
  [Debug a training step](#debug-a-training-step).

[Profiling](profiling.md) measures time, allocations and CPU/GPU resource use.
Debugging copies tensor values and can synchronize the GPU after each node, so
its timings are not representative inference performance. Both use the same
`ExecutionPlan` representation for schedule entries, source node IDs, tensor
IDs and storage placement. A plan ID is local to its trace or debug session;
join separate observations through graph identity, revision and source IDs.

## Choose the graph you want to inspect

An ordinary compiled model exposes its optimized schedule. A fused operation
may cover several source nodes; an eliminated tensor has no observable value.
Skipped entries are marked in the plan and their selected snapshots report
`OPTIMIZED_OUT`. A fused entry lists every source node it executes. The engine never invents values for those entries.

For every source boundary, compile a separate model with
`preserve_node_boundaries: true`. This disables node fusion and alias
elimination while retaining arena reuse. It does not change an existing
compiled model or retain every activation. Backend admission still applies:
requiring an unsupported GPU graph fails compilation instead of choosing CPU.
This executable can have different rounding and performance from the optimized
one; compare both when investigating an optimization.

The step unit is one schedule entry. It can dispatch multiple GPU kernels or,
for a skipped entry, no kernel. CPU/WASM and the built-in device backends use
the ordinary numerical dispatcher. External providers without node inspection
support return `BACKEND_UNSUPPORTED` when a session is requested.

## Create, step, inspect, release

The service follows the [Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/)
vocabulary: a session is `PAUSED` with a `stop_reason` (`ENTRY`, `STEP`,
`BREAKPOINT` or `NONFINITE`), `Step` runs one schedule entry and `Continue`
runs to a breakpoint or completion. This example uses the generated TypeScript
client with an existing host, compiled model and complete named input batch.

```ts
import { FullEngineHost, VxDebugServiceClient, pb } from 'volvoxai';

const debug = new VxDebugServiceClient(host);
let session = await debug.createDebugSession(new pb.CreateDebugSessionRequest({
  forward: new pb.DebugForward({ compiledModelId, inputs }),
  capture: new pb.DebugCapture({ values: true }),
}));
const ref = new pb.DebugSessionRef({ debugSessionId: session.debugSessionId });
try {
  // Run until the first node whose output contains NaN or infinity.
  session = await debug.continueDebugSession(new pb.ContinueDebugSessionRequest({
    debugSessionId: session.debugSessionId,
    expectedRevision: session.revision,
    breakOnNonfinite: true,
  }));
  if (session.state === pb.DebugState.DEBUG_STATE_FAILED) console.log(session.failure);
  const { plan } = await debug.getDebugPlan(ref);
  let pageToken = '';
  do {
    const page = await debug.listDebugEvents(new pb.ListDebugEventsRequest({
      debugSessionId: session.debugSessionId, pageToken,
    }));
    for (const event of page.events) {
      const step = plan.steps[event.step];
      for (const snapshot of event.snapshots)
        console.log(step.sourceNodeIds, event.point, plan.tensors[snapshot.tensorId].name,
          snapshot.statistics);
    }
    pageToken = page.nextPageToken;
  } while (pageToken);
} finally {
  await debug.releaseDebugSession(ref);
}
```

Python users can start the same session from an `InferenceSession`:

```python
import numpy as np
import volvoxai as vx

frame = np.load("input.npy", allow_pickle=False)
with vx.InferenceSession("model/") as model, model.debug(frame) as debug:
    debug.continue_(break_on_nonfinite=True)
    for event in debug.events():
        for snapshot in event.snapshots:
            if snapshot.status == vx.pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE:
                values = debug.tensor(snapshot)  # NumPy array, raw storage dtype
                with debug.tensor_buffer(snapshot) as retained:
                    print(retained.shape, retained.dtype)  # vx.Tensor, DLPack-capable
```

Python's `debug()` preserves source node boundaries by default using a
separate compiled model; ordinary inference retains its optimized schedule.
Use `model.debug(frame, preserve_node_boundaries=False)` to inspect the
optimized schedule. The wheel's full library supports debugging. An explicit
inference-only library override must be changed before creating the model
session. Collect captures and read snapshots while the debug scope is open.

`ExportDebugTensor` copies an AVAILABLE snapshot into an ordinary, independent
host buffer. It works with `VxBufferService` (`CopyTensors`, `ExportDLPack`)
and outlives the session; release it with `ReleaseBuffers`.

Creation validates and copies all inputs and returns `PAUSED` with reason
`ENTRY`. `StepDebugSession` runs one entry. `ContinueDebugSession` runs until
completion, a step executing one of `break_before_nodes`, or, with
`break_on_nonfinite`, the first step whose captured outputs contain NaN or
infinity. Breakpoints name source graph node IDs, the same IDs listed in
`ExecutionPlanStep.source_node_ids`; the entry the session is paused at runs
once, so continuing makes progress. The unary command replies only after its
observations have been published. During a pending command `GetDebugSession`
reports `RUNNING` and never advances execution.

Every command carries the last observed `revision` as `expected_revision`,
like an HTTP `If-Match` ETag. The revision changes on every state change; it is
a version, not a position (the position is `next_step`). Failures are typed so
that a client knows what to do without parsing messages:

| Status / code | Meaning | Action |
| --- | --- | --- |
| `REVISION_CONFLICT` | The revision is stale | Refresh with `GetDebugSession`, then retry |
| `BUSY` | A command is still running | Wait for its reply |
| `INVALID_ARGUMENT` + `DEBUG_SESSION_FINISHED` | `COMPLETED`, `CANCELLED` or `FAILED` | Create a new session |
| `INVALID_ARGUMENT` | Missing revision or unknown node ID | Fix the request |

A model that fails while being debugged is session state, not a failed RPC:
the reply succeeds with state `FAILED`, and `failure` carries the model's
report, including the offending node. Observations published before the
failure remain readable.

`ListDebugEvents` pages with `page_size`/`page_token`/`next_page_token`
(AIP-158). Events are append-only; an empty `next_page_token` means the end of
the events published so far. Each event contains the snapshots taken at that
point: `BEFORE` holds selected inputs and `AFTER` holds selected outputs. A
snapshot ID names an immutable storage version, whereas a tensor ID names a
tensor in the plan. Copies survive later arena reuse, aliasing, other
inferences and cancellation.

`ReadDebugTensor` returns raw little-endian storage bytes with
`read_offset`/`read_limit`, like `google.bytestream.Read`; reading is complete
when `read_offset` plus the returned length equals `size_bytes`. There is no
implicit dequantization. Quantized snapshots include their exact affine
parameters; `statistics` describe the integer storage values and
`real_statistics` describe the dequantized values. Statistics count finite
values, NaNs and both signs of infinity; min/max/mean/population variance
cover finite values only. `OPTIMIZED_OUT` (gdb's `<optimized out>`) marks a
tensor eliminated by optimization; the engine never invents its value.

## Debug a decode step and its KV cache

A decoder generates one token per `DecodeStep`. The step does not recompute the
whole sequence: it runs only the nodes whose inputs changed, writes one new row
per slot into each key/value cache, and attends over the rows already there.
Bugs in that path — a row written at the wrong position, a slot reading
another slot's rows, a stale page — are invisible in a full forward pass. A
decode target debugs the step itself.

Pass the request you would otherwise send as the session's target. The
operation is validated and staged exactly as the RPC would, including its page
reservation, and then waits at `ENTRY`:

```ts
let session = await debug.createDebugSession(new pb.CreateDebugSessionRequest({
  decodeStep: new pb.DecodeStepRequest({ contextId, slotActions, inputs }),
  capture: new pb.DebugCapture({ values: true }),
}));
const ref = new pb.DebugSessionRef({ debugSessionId: session.debugSessionId });
const { slots, caches } = await debug.getDebugDecodeState(ref);
// slots[i]: lengthBefore, lengthAfter and the rows [writeBegin, writeEnd) this step writes.
// caches[i]: one attention step's KEY or VALUE rows, the step that writes them and `written`.
```

Step as usual. Only the nodes the step actually runs are visited, in schedule
order; `next_step` names the next one. At every stop, `ReadDebugKVCache` reads
one slot of one cache live from the context, in logical token order —
gathered through the slot's page table when the cache is paged — as raw
storage rows `[tokens, ...token_shape]`:

```ts
const key = caches.find(cache => cache.role === pb.DebugKVRole.DEBUG_KV_ROLE_KEY);
const rows = await debug.readDebugKVCache(new pb.ReadDebugKVCacheRequest({
  debugSessionId: session.debugSessionId, cacheId: key.cacheId, slot: 0,
}));
// rows.data holds rows [0, rows.tokenCount); page with tokenOffset/tokenLimit.
```

Until a cache is `written` (its `writer_step` has run), the rows in
`[write_begin, write_end)` still hold their previous values; afterwards they
hold this step's. Every other row must be unchanged by the step, which is the
first thing to check when a decoder drifts. A fused QKV operand appears as two
caches, its key and value columns, over one tensor.

On completion the context has advanced exactly as the ordinary RPC would, and
`result_id` holds the result that RPC would have returned; read it with
`GetResult`/`ReadOutput` and release it with `ReleaseResult`. The caches stay
readable until you release the session. While a session is attached, every
other operation on the context — including another `DecodeStep` — fails with
`BUSY`; `ReleaseDebugSession` detaches it. Cancelling, or a failure partway
through, resets the context's decode state like a failed `DecodeStep`: the
context must be prefilled again, so no half-written cache survives.

Python has no decode convenience API; use the generated
`VxInferenceServiceClient` and `VxDebugServiceClient` with the same requests.

## Debug a training step

When a loss diverges or a gradient becomes NaN, `TrainStep` fails and restores
the Trainer; its report names the loss or parameter. To see where the value
first went wrong, pass the same request as a `train_step` target:

```ts
let session = await debug.createDebugSession(new pb.CreateDebugSessionRequest({
  trainStep: new pb.TrainStepRequest({trainerId, inputs, losses, trainableNames, optimizer}),
  capture: new pb.DebugCapture({values: true}),
}));
session = await debug.continueDebugSession(new pb.ContinueDebugSessionRequest({
  debugSessionId: session.debugSessionId, expectedRevision: session.revision, breakOnNonfinite: true,
}));
```

The plan's steps carry a `phase`: every forward node, one `LOSS` step, every
node's backward in reverse schedule order, `GRADIENT` steps for validation and
accumulation and, when the step applies the optimizer, clipping, and one
`OPTIMIZER` step per trainable parameter. Forward and backward steps list the
node's source ID, so `break_before_nodes` stops at both. Gradient tensors are
named `<tensor>.grad` and name their tensor in `gradient_of`; a backward step's
BEFORE snapshot holds the gradient arriving at its output and its AFTER
snapshot the gradients it produced. A gradient no loss reaches has no value
and is reported `OPTIMIZED_OUT`. `capture.gradients: false` leaves gradients
out.

`break_on_nonfinite` stops after the first step whose captured outputs contain
NaN or infinity, forward or backward. Select only gradients with
`capture.tensor_names` to find the first backward node that produced a
non-finite gradient, like PyTorch's `detect_anomaly`.

While the session is attached, every other operation on the Trainer returns
`BUSY`. Completing it leaves the Trainer exactly as the `TrainStep` would have,
byte for byte, and `train_step_result` holds what `TrainStep` would have
returned. Cancelling, a failure, or releasing an unfinished session restores
the Trainer like a failed `TrainStep`. CPU and WASM Trainers are supported; GPU
Trainers execute prebuilt backward plans and return `BACKEND_UNSUPPORTED`. A
request with an output selection is refused; capture forward values instead.

## Keep captures small

The default stores statistics for all node inputs and outputs. Full bytes
require `capture.values: true`. Select `capture.node_ids`, `capture.tensor_names`,
or input/output sides to narrow the investigation. Weights and other
initializers appear as inputs of every node that reads them; set
`capture.constants: false` to leave them out when you only compare activations. Empty selectors mean all;
unselected tensors are not copied and have no snapshot record.

`max_bytes` covers session metadata, the copied execution plan, quantization
metadata, retained values and capture staging. It excludes the private
execution context, shared weights, driver allocations and transport response
copies. It is a capture budget, not a bound on process RSS or total GPU memory.
Creation fails if the fixed metadata cannot fit. `max_events` and
`max_snapshots` bound record counts. Once a record/value cannot fit, inference
continues, earlier observations remain readable, and explicit statuses/loss
counters make the incomplete capture visible. Check `capture_complete`,
`events.dropped`, `snapshots.dropped`, `retained_bytes` and `peak_bytes` before
drawing conclusions.

Native GPU capture synchronizes host mirrors. WebGPU capture retains an
asynchronous staging ticket and yields to the host until it can copy bytes;
no JavaScript promise retains a WASM memory pointer. Cancellation is
cooperative between nodes and abandons any pending readback safely. It does
not preempt a submitted device kernel. Previously published snapshots remain
available; a partial final event is explicitly incomplete. Cancelling a Step
or Continue transport call also cancels its execution. Release cancels and
retires the handle; it is safe to repeat.

With no debug session there is no snapshot storage, statistics pass, debug
clock, readback or per-node debugger check. The ordinary forward loop remains
separate from the resumable debug path and shares its numerical dispatcher.

## Reproduce a difference

Use the same model revision, input bytes and graph mode on both backends.
Read the plan, before/after events, snapshot metadata and relevant raw chunks.
Match source node IDs, tensor names and phases rather than capture-local IDs.
Check dtype, shape and quantization before comparing values. Report the first
observable mismatch, its maximum absolute/relative error and nonfinite counts.
The first mismatch is evidence of where a difference became observable, not
proof that the node caused the underlying defect. A missing or dropped
snapshot makes that part of the comparison inconclusive.

`tools/debug_report.mjs` compares two collected sessions, like TFLite's
QuantizationDebugger or Polygraphy's layer-wise comparison. Collect each
session with `values: true`; compile both models with
`preserve_node_boundaries: true` when you want a value for every source node:

```js
import {collectDebugSession} from './tools/debug_report.mjs';
const artifact = await collectDebugSession(debug, pb, session.debugSessionId);
```

```bash
node tools/debug_report.mjs compare float.json int8.json report.json
```

Observations are aligned by the value they represent, the point (BEFORE or
AFTER) and the source node ID. PTQ authoring keeps node IDs but gives each
quantized activation a derived name (`__ptq__.<digest>.activation`); the report
maps it back to the float tensor it replaces, so `fc1`'s quantized output is
compared with the float `h1`. Quantized values are dequantized with their
recorded parameters before comparison.

The report chooses one of two modes, or takes `--mode`:

| Mode | Use it for | A row is a mismatch when |
| --- | --- | --- |
| `exact` | The same precision on two backends, or a fused and a preserved schedule | Any element exceeds `--atol`/`--rtol`, or nonfinite counts differ |
| `quantization` | A float model and its quantized package (chosen automatically when dtypes or quantization differ) | SQNR falls below `--min-sqnr` (default 20 dB), or nonfinite counts differ |

Every quantized layer differs from float, so in `quantization` mode read
`worst`: layer outputs ordered from the lowest signal-to-quantization-noise
ratio. Each row also reports cosine similarity, relative L2 error, maximum and
mean absolute error and, for quantized values, `rmseOverScale` (RMS error in
quantization steps; about 0.29 for uniformly distributed rounding error alone).
Errors accumulate: a layer can inherit low SQNR from the layers before it.

Check `comparable` and `activationCoverage` before trusting a clean result.
`comparable` is false when no values could be compared, for example two
unrelated models or sessions captured without values; the command then exits
with status 2. `unmatchedObservations` lists what had no counterpart. Float
weights replaced by packed storage in a quantized package are expected there,
with `kind: "constant"`.

Quantized packages written by PTQ record which float tensor each renamed
activation represents; plans report it as `ExecutionPlanTensor.source_tensor_name`,
and the comparison uses it (older packages fall back to the name derivation).

### Compare Python captures

Use `DebugSession.collect()` to save the same `volvoxai-debug/v1` format from
Python. Given a float model in `model/`, its quantized package in `quantized/`
and an input array matching both models:

```python
import json
from pathlib import Path
import numpy as np
import volvoxai as vx

inputs = np.load("input.npy", allow_pickle=False)

def capture(model, *, backend="cpu", preserve_node_boundaries=True):
    with vx.InferenceSession(model, backend=backend) as session:
        with session.debug(inputs, constants=False,
                           preserve_node_boundaries=preserve_node_boundaries) as debug:
            debug.continue_()
            if debug.failure is not None:
                raise RuntimeError(debug.failure.message)
            return debug.collect()

for model, filename in [("model", "float.json"), ("quantized", "int8.json")]:
    Path(filename).write_text(json.dumps(capture(model), indent=2), encoding="utf-8")

report = vx.debug_compare("float.json", "int8.json", mode="quantization", min_sqnr_db=20)
Path("report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report["comparable"], report["activationCoverage"], report["worst"])
```

For multiple inputs, load a mapping of input names to NumPy arrays instead.
The capture function finishes execution before collection; collecting a
paused session reads only its published observations and does not advance it.
Capture limits can still leave missing values, so inspect coverage and loss
counts before interpreting a clean comparison.

`vx.debug_compare` accepts capture dictionaries or JSON file paths and runs
offline without loading a model or a device. Python and JavaScript captures
can be mixed. Its keywords are `mode="auto"`, `atol=1e-5`, `rtol=1e-3` and
`min_sqnr_db=20`; report fields retain their JavaScript names such as
`firstMismatch`, `activationCoverage` and `sqnrDb`. `auto` selects the same
exact/quantization modes described above. Unsupported packed storage or missing
bytes provide no numerical evidence; `comparable` is false if nothing could
be compared.

The same helper compares CPU with a requested native GPU backend, or an
optimized schedule with a preserved one:

```python
backend_report = vx.debug_compare(capture("model"), capture("model", backend="cuda"),
                                  mode="exact", atol=1e-5, rtol=1e-3)
schedule_report = vx.debug_compare(capture("model"),
                                   capture("model", preserve_node_boundaries=False),
                                   mode="exact")
```

Choose a supported GPU backend on a machine with its device and driver.
Fused schedules can remove intermediate observations; a missing counterpart
is reported separately from the numerical comparison.

### Isolate a quantization error

To measure one layer's own error rather than what it inherits, feed it the
float reference. Pause before the layer (`break_before_nodes`), then replace
its activation input with `SetDebugTensor`, like DAP's `setVariable`: for a
quantized input, quantize the float value with that tensor's recorded affine.
Replacement is allowed only on CPU/WASM forward sessions compiled with
`preserve_node_boundaries`, only for a non-constant input of the next step,
and marks the session `modified`, since its observations no longer reproduce
an ordinary run.

Accumulated error hides which layer causes a loss: a layer can inherit low
SQNR from the layers before it. `tools/ptq_sensitivity.mjs` answers "which
nodes should stay in float?" by authoring one package per quantizable node,
like Polygraphy's `debug precision`. For each node it measures the graph
outputs' SQNR against the float model with only that node quantized
(`isolated`) and with every node except that one quantized
(`leaveOneFloat`, and `recoveryDb` over the fully quantized baseline):

```js
import {ptqSensitivity} from './tools/ptq_sensitivity.mjs';
const report = await ptqSensitivity({inference, quantization, p: pb, runtimeId,
  source: {graph, weights: [weights]}, calibration, evaluation, outputs: ['logits'],
  config: {activationDtype: pb.DataType.DATA_TYPE_I8}});
console.log(report.baseline.sqnrDb, report.bestToKeepFloat.slice(0, 3));
```

Pass the chosen IDs as `PtqAuthoringConfig.float_nodes`. The sweep only
composes the public PTQ calls, so it costs one authoring, calibration and
evaluation per node and mode; use a few representative batches. A selection
the package writer cannot express is reported with a note instead of a value.

Python offers the same CPU workflow through `vx.ptq_sensitivity` and
`volvoxai ptq sensitivity`. The
[Python sensitivity guide](../python/README.md#measure-ptq-sensitivity) shows
calibration/evaluation inputs, report interpretation and how to publish a
package with the chosen float nodes.

The [browser debugger example](../examples/debugging.html) shows a small graph
in the shared workbench: input/results, step/continue debugging, tensor values
and explicit performance/memory capture. The
[receipt workbench](../examples/receipt_digit_reader.html) adds image overlays
and digit attention through a model-specific tensor view. See the
[workbench guide](../examples/common/workbench/README.md) to reuse these
components with another model. The
native qualification script `tools/qualify_debugging.py` and WASM/WebGPU script
`tools/qualify_debugging.mjs` compare paused execution with ordinary inference,
including retained snapshots across arena reuse and concurrent ordinary runs.

For exhaustive field contracts, see [the proto](../proto/volvoxai.proto) and
[the generated API reference](generated/api-contract.full.md).
