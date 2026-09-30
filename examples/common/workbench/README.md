# Model workbench

The workbench connects three engineering workflows: run an input, explain its
execution, and measure its cost. The same shell, graph debugger, tensor browser
and profile views power the [receipt model](../../receipt_digit_reader.html)
and the [four-element vector model](../../debugging.html). Serve the repository
root over HTTP to open either example; the vector model needs no downloaded
weights. Use the receipt [setup guide](../../receipt_digit_reader/README.md)
for its packages and sample images.

VolvoxAI is a unified edge AI engine for native C, native GPU, WASM and WebGPU.
These are browser UI components over the generated services. The included
adapters execute browser backends; a native connection would need its own
transport/session adapter. There is no new engine API or model logic here.

## An engineering workflow

1. **Input & results:** choose an input and run the model. Keep model identity,
   backend and output together before investigating a numerical difference.
2. **Debug:** start a session, search the execution navigator, select a graph
   node and set a breakpoint before it. Step captures its inputs and outputs.
   Inspect Summary first, then open Tensor values or a model-specific view.
   Recapture one missing tensor explicitly instead of retaining every buffer.
3. **Performance & memory:** record an ordinary inference. Sort the measured
   operators, select a hotspot and follow its output into the debugger.
   Inspect memory ownership and CPU/GPU observation coverage before optimizing.
   Pin a baseline and export the structured evidence for people or tools.

Debug selection and execution position are separate. Moving between tabs starts
no profiler or session. Tensor bytes are read only when a requested values view
needs them; statistics and graph navigation use captured metadata. Ordinary
inference uses the adapter's normal execution path. UI modules and DOM still
occupy browser memory; inactivity does not mean the entire UI has zero cost.

The header keeps model controls and workflow tabs in one 52-pixel row on wide
screens, wrapping only when space is limited. Wide screens show the execution navigator, analysis
and inspector together; narrow screens switch these panes explicitly. Pane
dividers support pointer and arrow-key resizing. F10 steps, F8 continues,
F9 toggles the selected breakpoint, and `/` focuses search. Tabs use manual
activation: arrows move focus, Enter/Space activates. Components follow the
system color scheme and reduced-motion preference.

## Embed another model

`ModelWorkbench` owns presentation. Its adapter owns loading, input binding,
ordinary inference, preprocessing, output decoding and backend selection.
For a complete small adapter, read
[`VectorWorkbench.js`](../../debugging/VectorWorkbench.js).

```js
import {ModelWorkbench} from './examples/common/workbench/ModelWorkbench.js';

const workbench = new ModelWorkbench({
  root: document.getElementById('workbench'), // give this element a height
  pb: api.pb,
  title: 'My model',
  subtitle: 'Model revision and task',
  debug: {
    getSource: () => ({label: inputLabel, backend, variant, graph}),
    createSession: options => createDebugSessionForCurrentInput(options),
  },
  onRecord: async () => {
    // Adapter serializes engine use, reports errors, and releases the trace.
    const {trace, wallMs} = await recordCurrentInput();
    workbench.profiler.setTrace({trace, wallMs,
      source: {label: inputLabel, backend, variant}});
  },
  onBusyChange: busy => disableModelControls(busy),
});
// Mount application-specific DOM in workbench.inputRoot / controlsRoot.
workbench.debugger.setAvailable(true);
workbench.profiler.setAvailable(true);
workbench.setStatus('Ready');
```

The root mounts an open Shadow DOM. Internal selectors, IDs and styles are
local to each instance. `inputRoot` and `controlsRoot` live inside that scope;
query them directly instead of using document-wide IDs. Styles use common
tokens such as `--panel`, `--ink`, `--muted` and `--accent`. Attach model-specific
styles within the component's scope, not as page-wide selectors.

`createSession({preserveNodeBoundaries, capture, limits})` returns an
[`EngineDebugSession`](../EngineDebugSession.js) with its plan loaded. It must
bind a stable copy of the current input and honor the capture/graph options.
The debugger owns that returned session and releases it on restart/reset.
The adapter still owns its runtime, model and ordinary compiled handle.
`getSource()` provides provenance and optional extension context; `graph`
supplies source-node mapping when recapturing an optimized-away tensor.

[`startEngineTrace`](../EngineTrace.js) returns `finish()` and `release()`.
Finish reads events, resource samples, plans and the Chrome trace, then releases
the engine collector even on failure. Use `release()` in the adapter's `finally`
block if execution fails before finish. Pass `{trace, source, wallMs}` to
`profiler.setTrace()`; the source should contain serializable input/model identity,
not engine handles or functions. A complete trace contains `events`, `resources`
and `plans` arrays, including empty arrays for sections with no records. Missing
sections fail validation before replacing a displayed report. If a browser has
cached an older capture module during development, reload with the browser cache
bypassed and record again. Exports preserve protobuf JSON field semantics.

While an adapter operation is active, call `debugger.setAvailable(false)` and
`profiler.setBusy(true)` and lock model/input controls. Restore them in `finally`.
Before changing input or model, await `debugger.reset()` and call
`profiler.reset()`; clear capture filters with
`debugger.reset({clearCaptureSelection: true})` when the graph changes.
`show('input' | 'debug' | 'profile')` selects a workflow. On removal, await
`workbench.dispose()` before closing adapter-owned runtime resources. Do not
dispose during an in-flight adapter or debug operation.

## Add a tensor view

Pass `debug.tensorViews` entries with `{id, label, create}`. The unique lowercase
ID becomes an analysis tab; `graph` and `values` are reserved. The factory receives
`{root, pb, run, recapture, capture, select}`. Run asynchronous actions through
`run(action)` to share the debugger's operation lock and error reporting.

The returned view implements this small lifecycle:

| Method | Responsibility |
| --- | --- |
| `setContext({...source, session})` | Accept the current session and adapter context. |
| `show(entry, step)` | Display the selected tensor; either argument may be null. |
| `clearSelection()` | Drop selection-specific bytes and stale content. |
| `reset({keepPin})` | Release cached state; retain bounded comparison data only when requested. |
| `setBusy(boolean)` | Disable actions during an in-flight operation. |
| `toJson()` | Return serializable view metadata for the observation export. |

An entry contains `{tensor, tensorId, phase, snapshot}`; the snapshot can be
absent. Treat missing, statistics-only, dropped and optimized-away values as
distinct states. `recapture()` reruns only the selected tensor and phase.
`capture({tensorNames, targetName})` explicitly starts an original-graph,
output-only capture, runs to completion and selects the target (last name by
default). `select(step, tensorId)` selects an existing output without execution.
The supplied callbacks run within `run(action)`; do not acquire that lock twice.

The receipt adapter adds [`ReceiptSpatialView`](../../receipt_digit_reader/ReceiptSpatialView.js)
using verified image-grid mapping and slot labels. These policies do not belong
in the common graph or debugger. Its numerical slicing implementation,
[`SpatialTensor.js`](../SpatialTensor.js), is reusable with another model's
explicit coordinate descriptor. An overlay is an activation/attention view,
not a claim of causal attribution.

## Read measurements correctly

Host nodes, nested engine programs and GPU intervals overlap. The summary
excludes nested host programs and uses GPU program intervals when present,
otherwise node intervals, never both. It labels the interval basis and counts;
only recorded intervals contribute. Fused work is never divided among its
source nodes. Host tracks retain their measured timestamps. GPU clocks and
correlation uncertainty remain in the exported trace and Perfetto.

WASM memory is an exact module partition at observation time. Allocator
capacities overlap that heap or device storage, and logical tensors may alias
an arena. They are not additive to each other or to process RSS. Sampled peaks
are not continuous monitoring. The capture collector itself consumes resources;
its size and dropped records are visible. Release debug snapshots before
profiling inference alone.

Process CPU load requires comparable CPU-time and observation-time deltas;
one occupied core is 1.0 and multiple threads may exceed it. Hardware GPU
activity uses actual driver samples, includes other processes and is neither
SM occupancy nor percentage of peak compute. Browser APIs do not provide these
process/driver readings. Unsupported, failed, absent and exact-zero values stay
distinct. An operator's duration never substitutes for utilization.

Baseline comparison retains one wall-time/heap summary. Each capture is one
instrumented run, not a distribution or speedup claim. Analysis JSON retains
events, resource samples, plan identity, placement and capture loss so people
and AI tools can inspect the same evidence. Exhaustive contracts live in the
[proto](../../../proto/volvoxai.proto), with workflows in the
[debugging](../../../docs/debugging.md) and [profiling](../../../docs/profiling.md)
guides.

## Design references

The layout takes linked graph/tensor inspection from
[TensorBoard Debugger V2](https://www.tensorflow.org/tensorboard/debugger_v2)
and graph navigation from [Netron](https://github.com/lutzroeder/netron).
Explicit recording, selection-linked details and export follow
[Chrome DevTools Performance](https://developer.chrome.com/docs/devtools/performance/reference).
Frequently used actions stay visible while capture settings and secondary
examples use [progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/).
Keyboard tabs follow the [WAI-ARIA manual activation pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/).
