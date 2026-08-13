# Browser and Node runtime

VolvoxAI exposes one inference ownership model from every JavaScript profile:

~~~text
Runtime
  Model
    CompiledModel
      ExecutionContext
        ExecutionResult
~~~

Runtime owns initialized backend providers. Model owns an
immutable logical graph, the complete bounded shape domain, and one exact
fixed-weight revision. CompiledModel pins that snapshot and proves one provider
for its whole domain. Each ExecutionContext owns concrete shape bindings,
specializations, request/scratch capacity, and decode state. ExecutionResult
owns stable named output snapshots.

## Profiles

The readable and minified variants have the same API:

| Artifact | Capabilities | WASM sidecar |
| --- | --- | --- |
| volvoxai.js | Multi-backend inference | volvoxai.wasm |
| volvoxai.full.js | Multi-backend inference and training | volvoxai.full.wasm |
| volvoxai.wasm.js | Strict WASM inference and training | volvoxai.full.wasm |

The standard inference entry imports and exports no training code. The
WASM-only entry contains no CPU JS, WebGPU, WGSL, or Node filesystem
implementation.

The inference entry exports `Runtime`, `ModelLoader`,
`Model`, `CompiledModel`, `ExecutionContext`, and
`ExecutionResult`, their reports, the bounded-shape contracts,
`VolvoxAIError`, and the context-aware provider contract. Programmatic package
authoring uses `ModelBuilder`; package loading never constructs a
kernel-facing `Graph` or `Tensor`. Backend engines, mutable training state, and
the retired concrete package loader are not inference exports.

## Create a runtime

~~~javascript
import { VolvoxAI } from 'volvoxai';

const runtime = await VolvoxAI.createRuntime({
  backends: ['webgpu', 'wasm', 'cpu-js'],
  wasmUrl: new URL('./volvoxai.wasm', import.meta.url),
  onDiagnostic(event) {
    console.debug(event);
  },
});
~~~

The backends option selects which providers to initialize and records
unavailable providers for later compilation reports. Omitting it uses the
explicit built-in order WebGPU, WASM, then CPU JS (`cpu-js`).

The value is always a non-empty ordered array. It does not add CPU JS unless
`cpu-js` is present; there are no string or `auto` shorthands.

Runtime.listBackends() returns the selected provider names, including providers
whose initialization outcome is retained for diagnostics.

## Load a model

~~~javascript
import { Model } from 'volvoxai';

const snapshot = await Model.load(
  './models/detector/model.safetensors',
);
~~~

The first safetensors URL resolves a sibling graph.json by default. Multiple
weight shards are accepted:

~~~javascript
const snapshot = await Model.load(
  [
    './models/large/model-00001-of-00002.safetensors',
    './models/large/model-00002-of-00002.safetensors',
  ],
  { graphUrl: './models/large/graph.json' },
);
~~~

Every graph document must contain the exact root discriminator:

~~~json
{
  "format": "volvox-graph/v1",
  "dimensions": {}
}
~~~

Dynamic shape is intrinsic to this format, and the closed root schema rejects
secondary shape-system discriminators. The URL basename must be `graph.json` or
a named `*.graph.json` document, and the loader does not search alternate
filenames. Every dynamic dimension is declared with finite bounds. Every node output is a
unified `{ tensor, dtype, shape }` descriptor and every node input resolves to a
declared graph input, a named weight, or an earlier output.

Applications can instead author a logical graph with `ModelBuilder` and
capture its graph plus owned fixed weights in a `Model`.

## Compile with explicit policy

Preferred selection tries candidates in order until one compiles:

~~~javascript
const compiled = await runtime.compile(snapshot, {
  backend: {
    mode: 'prefer',
    order: ['webgpu', 'wasm', 'cpu-js'],
    operatorFallback: 'allow',
  },
});
~~~

Required selection accepts exactly one provider:

~~~javascript
const compiled = await runtime.compile(snapshot, {
  backend: {
    mode: 'require',
    backend: 'webgpu',
    operatorFallback: 'forbid',
  },
});
~~~

The two controls are independent:

- mode controls provider-tier selection.
- operatorFallback controls whether the selected provider may route individual
  operators elsewhere.

The forbid value requires the provider to attest that every node stays on the
selected engine. A shader or kernel tactic change on the same device is not
operator fallback.

compiled.report contains the requested policy, every candidate outcome, the
selected backend and provider-reported device identity when available, pinned
model and adapter revisions, compile time, available allocation totals,
tier-fallback evidence, and operator-route evidence. Backend selection ends
before execution. A failed execution never retries on another provider.

## Execute in independent contexts

~~~javascript
const first = await compiled.createContext();
const second = await compiled.createContext();

const [firstResult, secondResult] = await Promise.all([
  first.execute({ input: { data: firstValues, shape: [2, 128] } }),
  second.execute({ input: { data: secondValues, shape: [5, 64] } }),
]);
~~~

Every input carries explicit concrete shape. The runtime validates the exact
input set, storage dtype, rank, symbol equality, bounds, `multiple_of`, and byte
length before it resolves any kernel or allocation plan. Raw typed-array
shorthand is intentionally not supported, including for constant-only models.

One context serializes execute, decode, adapter selection, and close through a
FIFO queue. Different contexts may progress concurrently because each has
private mutable execution state.

Every context retains its compiled model, and each compiled model retains the
runtime resources it needs. Starting close rejects new work and drains work
already accepted. close() and dispose() are idempotent.

## Stable named results

~~~javascript
const result = await first.execute({
  input: { data: values, shape: [2, 128] },
});

for (const [name, tensor] of result.outputs) {
  console.log(name, tensor.shape, tensor.dtype, tensor.location);
}

const logits = await result.output('logits').read();
~~~

Every declared graph output appears by exact name. A host read returns a fresh
caller-owned typed array. A device tensor also exposes deviceBuffer, but the
buffer remains result-owned; the caller must not destroy it. read() performs a
typed readback and still returns a fresh array.

A result remains stable across later executions and context closure. Its
storage stays valid until result.close():

~~~javascript
const resultA = await first.execute({
  input: { data: valuesA, shape: [2, 128] },
});
const resultB = await first.execute({
  input: { data: valuesB, shape: [5, 64] },
});
await first.close();

const a = await resultA.output('logits').read();
const b = await resultB.output('logits').read();

await resultA.close();
await resultB.close();
~~~

## Decode contexts

Decode state belongs to its context:

~~~javascript
const decode = await compiled.createContext({
  decode: {
    changedInputs: ['tokens', 'attention_mask'],
    rowMode: 'auto',
  },
});

let result = await decode.decode.seed(seedInputs);
await result.close();

result = await decode.decode.step(nextInputs, { position: 1 });
const token = await result.output('token_ids').read();
await result.close();

await decode.decode.reset();
await decode.close();
~~~

seed performs the complete first pass. step uses the backend's supported
dependency or row path. reset clears retained decode state without closing the
context. A provider that does not support a requested decode operation rejects
with BACKEND_UNSUPPORTED.

Fixed-shape B=1 W8A8 decode has dependency, row, and K/V-cache implementations
on CPU JS, WASM, and WebGPU.

## Weight revisions and adapters

A logical snapshot owns one immutable fixed-weight revision. Create and compile
a successor snapshot to publish new weights; existing compiled models and
contexts remain pinned to the previous revision. Adapter mutation is not part
of the inference logical-snapshot contract. Full-profile training/PTQ must
publish a new logical snapshot before inference compilation.

## Diagnostics and errors

onDiagnostic receives compilation, execution, and execution-error events.
Compilation and execution reports are frozen and serializable. Execution
reports include execution/context identities, backend and device, pinned model
and adapter revisions, elapsed time, tier/operator route evidence, and
decode/cache state. Stable VolvoxAIError codes include:

Memory evidence is an explicit runtime-scoped opt-in. The public DTO mirrors
the protobuf contract while keeping uint64 fields as safe-integer numbers so
`JSON.stringify(report)` remains valid:

~~~javascript
import {
  MEMORY_CAPTURE_PROTOCOL,
  MemoryEnvelopeKind,
  VolvoxAI,
} from 'volvoxai';

const runtime = await VolvoxAI.createRuntime({
  memoryCapture: {
    protocol: MEMORY_CAPTURE_PROTOCOL,
    includeResourceInventory: true,
    includeDomainAttestation: true,
    requestedEnvelopes: [
      MemoryEnvelopeKind.ProcessRss,
      MemoryEnvelopeKind.ProcessManagedHeapUsed,
      MemoryEnvelopeKind.DeviceProcessUsed,
    ],
  },
});
~~~

Absence leaves report shapes unchanged and performs no sampling. A successful
compile/execute/decode report carries one `AFTER` snapshot; an execution error
report carries `FAILURE`. Every requested but unsupported envelope is retained
with `UNAVAILABLE` and no byte value. Envelopes overlap one another and resource
records, so consumers compare them as separate series and never add them.

`PROCESS_MANAGED_HEAP_USED` comes from the Node heap counter as an `EXACT`
`typescript-runtime-memory/v1` value. Where that counter is absent, browsers
supply `performance.memory` instead, which is quantized rather than live: the
fallback is reported as `ESTIMATED` under its own
`browser-performance-memory/v1` sampler, so a consumer can tell the two
collectors apart before comparing their series. Capture never changes the
report fields it sits beside; in particular `executionTimeMs` continues to
measure the execution and excludes collector cost.

The WASM provider exposes the complete linear-memory allocation only once as an
exact `WASM_LINEAR` capacity root. Its arena counters are subregions of that
root and are not additional physical bytes. The inventory is still `PARTIAL`
because JavaScript-side weights and result snapshots are outside the root.
WebGPU does not expose portable process VRAM or residency; API-requested buffer
sizes must not be relabeled as physical VRAM. Periodic capture is rejected until
an operation-window sampler is implemented.

The package entry uses only the inference-safe enum projection and handwritten
DTOs. It does not import the generated Synurang codec or its full-profile
Trainer/PTQ surface.

~~~text
BACKEND_UNAVAILABLE
BACKEND_UNSUPPORTED
BACKEND_REQUIRED
OPERATOR_FALLBACK_FORBIDDEN
EXECUTION_FAILED
RESULT_DISPOSED
HANDLE_DISPOSED
DEVICE_LOST
ABI_UNSUPPORTED
INVALID_ARGUMENT
~~~

Use the code and report fields for control flow; human-readable messages may
gain detail.

## External backend provider SPI

External devices use the same context-aware provider SPI as built-in
backends:

~~~javascript
const runtime = await VolvoxAI.createRuntime({
  backends: ['my-npu'],
  providers: {
    'my-npu': async factoryContext => createMyProvider(factoryContext),
  },
});
~~~

Providers compile an immutable Model, create independently owned
contexts, and return stable host or device output snapshots. See
[Backend SDK](backend-sdk.md) for the complete contract.

## WASM-only Manifest V3 extensions

Package one WASM-only JavaScript artifact, volvoxai.full.wasm, graph.json, and
the model weights. All executable JavaScript and WASM must be extension-owned.

~~~json
{
  "manifest_version": 3,
  "name": "Local model",
  "version": "1.0.0",
  "background": {
    "service_worker": "service-worker.js",
    "type": "module"
  },
  "content_security_policy": {
    "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self';"
  }
}
~~~

~~~javascript
import {
  Model,
  VolvoxAI,
} from './vendor/volvoxai.wasm.min.js';

const wasmUrl = chrome.runtime.getURL('vendor/volvoxai.full.wasm');
const runtime = await VolvoxAI.createRuntime({ wasmUrl });
const snapshot = await Model.load(
  chrome.runtime.getURL('model/model.safetensors'),
  { graphUrl: chrome.runtime.getURL('model/graph.json') },
);
const compiled = await runtime.compile(snapshot, {
  backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
});
~~~

The WASM-only build contains no dynamic import, which keeps it suitable for an
ES-module extension service worker. Extension pages require wasm-unsafe-eval.
Declare web-accessible resources only when a normal web page or another
extension must fetch them.

The full profile may train and author a successor weight revision, but it must
publish that revision as a new logical snapshot before inference compilation:

~~~javascript
const trainer = await VolvoxAI.createTrainer(sourceSnapshot, {
  backend: 'wasm',
  wasmUrl,
});
const step = await trainer.trainStep(options);
const successorSnapshot = await trainer.commit();
await trainer.close();

const compiledSuccessor = await runtime.compile(successorSnapshot, {
  backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
});
~~~

Trainer privately clones the immutable source snapshot and owns the mutable
working state. `trainStep()` never mutates the source. `commit()` returns a new
immutable weight revision; compile that successor, or call `rollback()` to
restore the last committed baseline. Strict WASM training preflights the
supported portable subset before forward execution or optimizer mutation.

## Backend characteristics

### WebGPU

WebGPU owns device buffers outside the portable Tensor model, compiles an
immutable graph plan, and shares only device-level module and pipeline caches.
Each context owns its input, activation, scratch, output, adapter, and decode
state.

Compilation proves the complete declared bounded-shape domain before it
publishes a context. The proof covers output extents, physical allocation
maxima, dispatch grids, uniforms, bind groups, auxiliary buffers, and device
limits for every qualified route. A concrete shape change specializes those
resources transactionally, and even equal-byte shapes receive their own
semantic signature. A rejection before the mutation boundary leaves the prior
signature, cache, decode state, and resource counters unchanged. If a queue
write fails after that boundary, retained decode state is cleared and the
physical signature is invalidated; the next request rewrites the complete
binding instead of dispatching through possibly partial uniforms.

Index-bearing public and invariant values are checked before any GPU or cache
mutation. Banked Gather translates global rows through the selected data-bank
table. A residency change stages a fresh weight-buffer generation and publishes
its payload and slot table together only after specialization succeeds; the
previous generation remains valid until submitted work retires. Unsupported
bank mappings or an unprovable device-produced value domain fail closed at
compilation.

Every declared output receives a stable result snapshot, including outputs
that alias graph inputs, weights, or transitive identity/Dropout values.
Unsupported operators reject compilation with a typed error; execution never
leaves an unwritten output.

Canonical W8A8 dense and convolution nodes may use packed integer dot products
when the adapter advertises packed_4x8_integer_dot_product. Eligible QLinear,
QMatMul, QGemm, and groups=1 QConv2D nodes have cooperative tiled paths;
otherwise portable packed-byte paths preserve arbitrary I8/U8 zero points and
reduction tails.

### WASM SIMD

The inference profile loads volvoxai.wasm. Full and WASM-only profiles load
volvoxai.full.wasm, whose strict training and PTQ routines are separated from
the live inference instance.

Both fixed parents remain baseline SIMD modules. An optional Relaxed-SIMD W8A8
child is embedded and instantiated only when the engine compiles it; rejection
falls back within the same WASM provider without another fetch.

QBatchMatMul uses a standard-SIMD128 parent export for groups of four adjacent
output columns. The engine selects it explicitly when present; custom/older
parents retain the scalar export, and N tails remain scalar inside the same
validated call. Neither route dequantizes through F32 MatMul or retries in the
JavaScript CPU provider.

### CPU JS (`cpu-js`)

The JavaScript CPU provider is the portable reference. It is dependency-free
and is used for correctness comparisons and CPU-only deployment.

Consult [the operation matrix](operation_list.md) for exact dtype, layout,
inference, and training coverage.

## Node behavior

The standard and full entries can use WASM or CPU JS in Node. The WASM-only
package subpaths are browser-only because they omit the filesystem loader.

Inference WGSL is bundled only into the browser release and is loaded only when
the WebGPU provider initializes. Importing the standard entry in Node does not
initialize WebGPU unless it was selected.
