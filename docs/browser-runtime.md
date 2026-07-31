# Browser and Node runtime

VolvoxAI exposes one inference ownership model from every JavaScript profile:

~~~text
Runtime
  Model
    CompiledModel
      ExecutionContext
        ExecutionResult
~~~

Runtime owns initialized backend providers. Model owns an immutable definition
and the currently published weight revision. CompiledModel pins one exact
definition and revision. Each ExecutionContext owns request, scratch, adapter,
and decode state. ExecutionResult owns stable named output snapshots.

## Profiles

The readable and minified variants have the same API:

| Artifact | Capabilities | WASM sidecar |
| --- | --- | --- |
| volvoxai.js | Multi-backend inference | volvoxai.wasm |
| volvoxai.full.js | Multi-backend inference and training | volvoxai.full.wasm |
| volvoxai.wasm.js | Strict WASM inference and training | volvoxai.full.wasm |

The standard inference entry imports and exports no training code. The
WASM-only entry contains no CPU, WebNN, WebGPU, WGSL, or Node filesystem
implementation.

The runtime exports are the five retained inference handles (`Runtime`,
`Model`, `CompiledModel`, `ExecutionContext`, and `ExecutionResult`), their
result tensors and reports, `VolvoxAIError`, and the context-aware provider
contract. Programmatic package authoring remains explicit through `Graph`,
`Tensor`, `ModelBuilder`, `GraphLoader`, `SafetensorsFile`,
`ReadOnlySafetensorsCache`, and `Tokenizer`. Backend engines, mutable adapter
managers, model-snapshot constructors, JSON parser helpers, and error-wrapping
helpers are internal and are not package exports.

## Create a runtime

~~~javascript
import { VolvoxAI } from 'volvoxai';

const runtime = await VolvoxAI.createRuntime({
  backends: ['webnn', 'webgpu', 'wasm', 'cpu'],
  wasmUrl: new URL('./volvoxai.wasm', import.meta.url),
  onDiagnostic(event) {
    console.debug(event);
  },
});
~~~

The backends option selects which providers to initialize and records
unavailable providers for later compilation reports. Omitting it uses the
explicit built-in order WebNN, WebGPU, WASM, then CPU.

The value is always a non-empty ordered array. It does not add CPU unless cpu
is present; there are no string or `auto` shorthands.

Runtime.listBackends() returns the selected provider names, including providers
whose initialization outcome is retained for diagnostics.

## Load a model

~~~javascript
const model = await runtime.loadModel(
  './models/detector/model.safetensors',
);
~~~

The first safetensors URL resolves a sibling graph.json by default. Multiple
weight shards are accepted:

~~~javascript
const model = await runtime.loadModel(
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
  "format": "volvox-graph/v1"
}
~~~

The loader rejects any other value before allocating weights or backend
resources. The URL basename must be `graph.json` or a named `*.graph.json`
document, and the loader does not search alternate filenames. Every node input
must resolve to a declared graph input, a named weight, or an earlier node
output.

Applications can also construct a Graph and call runtime.createModel(graph).
The model captures the graph; later execution never mutates caller-owned Graph
or Tensor storage.

## Compile with explicit policy

Preferred selection tries candidates in order until one compiles:

~~~javascript
const compiled = await model.compile({
  backend: {
    mode: 'prefer',
    order: ['webgpu', 'wasm', 'cpu'],
    operatorFallback: 'allow',
  },
});
~~~

Required selection accepts exactly one provider:

~~~javascript
const compiled = await model.compile({
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
  first.execute({ input: firstValues }),
  second.execute({ input: secondValues }),
]);
~~~

One context serializes execute, decode, adapter selection, and close through a
FIFO queue. Different contexts may progress concurrently because each has
private mutable execution state.

Every context retains its compiled model, and each compiled model retains the
runtime resources it needs. Starting close rejects new work and drains work
already accepted. close() and dispose() are idempotent.

## Stable named results

~~~javascript
const result = await first.execute({ input: values });

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
const resultA = await first.execute({ input: valuesA });
const resultB = await first.execute({ input: valuesB });
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
on CPU, WASM, and WebGPU. WebNN provides ordinary forward execution.

## Adapter selection

An adapter route can be pinned when the context is created or changed through
the context FIFO:

~~~javascript
const context = await compiled.createContext({ adapter: { name: 'tenant-a' } });
await context.selectAdapter({ name: 'tenant-b', version: 2 });
const result = await context.execute(inputs);
~~~

Compiled models pin weight and adapter revisions. Publishing a model successor
does not silently change an existing context.

## Diagnostics and errors

onDiagnostic receives compilation, execution, and execution-error events.
Compilation and execution reports are frozen and serializable. Execution
reports include execution/context identities, backend and device, pinned model
and adapter revisions, elapsed time, tier/operator route evidence, and
decode/cache state. Stable VolvoxAIError codes include:

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

Providers compile an immutable ModelSnapshot, create independently owned
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
  Graph,
  GraphLoader,
  VolvoxAI,
} from './vendor/volvoxai.wasm.min.js';

const wasmUrl = chrome.runtime.getURL('vendor/volvoxai.full.wasm');
const runtime = await VolvoxAI.createRuntime({ wasmUrl });
const graph = new Graph();
await GraphLoader.load(
  graph,
  chrome.runtime.getURL('model/model.safetensors'),
  { graphUrl: chrome.runtime.getURL('model/graph.json') },
);
const model = runtime.createModel(graph);
~~~

The WASM-only build contains no dynamic import, which keeps it suitable for an
ES-module extension service worker. Extension pages require wasm-unsafe-eval.
Declare web-accessible resources only when a normal web page or another
extension must fetch them.

Training uses the same retained Trainer API:

~~~javascript
const trainer = await VolvoxAI.createTrainer(model, { wasmUrl });
const step = await trainer.trainStep(options);
await trainer.commit();
await trainer.close();
~~~

Trainer privately clones the Model's current revision and owns that mutable
working state. `trainStep()` never publishes implicitly: call `commit()` before
compiling inference against the update, or `rollback()` to restore the last
committed baseline. Strict WASM training preflights the supported portable
subset before forward execution or optimizer mutation.

## Backend characteristics

### WebNN

WebNN builds through navigator.ml. The browser may route work to NPU, GPU, or
CPU depending on platform support.

Current mapped operations include MatMul, Linear, Gemm, Add, Mul, ReLU, GELU,
SiLU, Sigmoid, Softmax, Reshape/Flatten, LayerNorm, Conv2D, Embedding, and
SDPA decomposed into lower-level WebNN operations.

WebNN requires a secure context. Requesting deviceType: npu does not prove
physical NPU execution; use provider evidence when device identity matters.

### WebGPU

WebGPU owns device buffers outside the portable Tensor model, compiles an
immutable graph plan, and shares only device-level module and pipeline caches.
Each context owns its input, activation, scratch, output, adapter, and decode
state.

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

### CPU

The JavaScript CPU provider is the portable reference. It is dependency-free
and is used for correctness comparisons and CPU-only deployment.

Consult [the operation matrix](operation_list.md) for exact dtype, layout,
inference, and training coverage.

## Node behavior

The standard and full entries can use WASM or CPU in Node. The WASM-only
package subpaths are browser-only because they omit the filesystem loader.

Inference WGSL is bundled only into the browser release and is loaded only when
the WebGPU provider initializes. Importing the standard entry in Node does not
initialize WebGPU unless it was selected.
