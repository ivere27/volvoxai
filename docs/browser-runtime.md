# Browser and Node runtime

Use VolvoxAI to run a model locally in a web page, extension, or Node service.
A host loads the runtime, a model holds the graph and weights, and a compiled
model prepares them for the chosen backend. You can reuse that compilation for
many requests or create independent contexts for stateful sessions.

Start with the [quickstart](quickstart.md) for a runnable graph. This guide
explains how to connect your own package, inputs, and application lifecycle.

## Choose a profile

| Import | Host | Companion | Capabilities |
| --- | --- | --- | --- |
| `volvoxai/lite` | `EngineHost` | `volvoxai.lite.wasm` | WASM CPU inference, text, graph construction, scheduling |
| `volvoxai` | `FullEngineHost` | `volvoxai.wasm` | The above plus WebGPU inference, training, and PTQ |

The readable and minified variants have the same API. Each host needs its
matching WASM companion. WebGPU is available only in the full profile and only
when the device and graph are supported.

Each host owns a separate C/WASM instance. Share one host when several clients
should use the same models and Runtime scheduler. Create another host when you
need independent memory and lifetimes; IDs cannot cross between hosts.

## Create a runtime

```javascript
import { EngineHost, VxInferenceServiceClient, pb } from 'volvoxai/lite';

const host = new EngineHost();
const inference = new VxInferenceServiceClient(host);
const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
```

The default companion URL is relative to the JavaScript module. Set `wasmUrl`
when a bundler or CDN moves it:

```javascript
const host = new EngineHost({
  wasmUrl: new URL('/assets/volvoxai.lite.wasm', location.href),
});
```

Host options describe deployment and transport. Choose the numerical backend
when compiling a model. A Runtime defaults to DIRECT execution; scheduled work
is an explicit choice described below.

## Load a model

In a browser, provide the graph URL and ordered weight URLs:

```javascript
const model = await inference.loadModel(new pb.LoadModelRequest({
  runtimeId: runtime.runtimeId,
  graphPath: './models/my-model/graph.json',
  weightPaths: ['./models/my-model/model.safetensors'],
}));
const info = await inference.getModelInfo(new pb.ModelRef(model));
console.log(info.inputs, info.outputs, info.execution);
```

The graph path is explicit. Its basename is `graph.json` or a named
`*.graph.json`; multiple SafeTensors shards go in `weightPaths` in package order.
The [model format](model-format.md) describes tensor names, layouts, and bounded
dimensions. Inspect the model's I/O before choosing a batch shape or allocating
input arrays.

In Node, read local files and pass their bytes. This also works for packages
you construct in memory, including exported training or PTQ packages:

```javascript
import { readFile } from 'node:fs/promises';

const model = await inference.loadModel(new pb.LoadModelRequest({
  runtimeId: runtime.runtimeId,
  package: new pb.ModelPackage({
    graphDocument: new Uint8Array(await readFile('models/my-model/graph.json')),
    weightShards: [new Uint8Array(await readFile('models/my-model/model.safetensors'))],
  }),
}));
```

Use either the byte package or paths in one request. Path-based Node loading
can use HTTP URLs or a supplied `fetch` adapter; ordinary relative filesystem
paths are not valid Node fetch URLs. The WASM companion itself can be loaded
from a local file by the Node host.

Browser `fetch` and `resolveModelSource(graphPath, weightPaths)` options let you
adapt model paths to your storage. Ordinary path loads, inline packages and
adapter packages have a cumulative 64 MiB ceiling. Publish complete graph and
weight revisions together.

## Load a larger streamed package

For larger path sources, publish a manifest containing each complete file's
byte length and SHA-256 digest. Choose explicit source budgets when loading:

```javascript
const model = await inference.loadModel(new pb.LoadModelRequest({
  runtimeId: runtime.runtimeId,
  graphPath: manifest.graph.url,
  weightPaths: manifest.weights.map(source => source.url),
  sourceOptions: new pb.ModelSourceOptions({
    maxTotalBytes: BigInt(manifest.totalBytes),
    maxShardBytes: BigInt(manifest.largestFileBytes),
    graph: new pb.ModelSourceIntegrity({
      byteLength: BigInt(manifest.graph.bytes), sha256: manifest.graph.sha256,
    }),
    weightShards: manifest.weights.map(source => new pb.ModelSourceIntegrity({
      byteLength: BigInt(source.bytes), sha256: source.sha256,
    })),
  }),
}));
```

Here each `sha256` is a 32-byte `Uint8Array` prepared from your publisher's
digest. The descriptors follow the weight path order. C validates the entire
manifest before fetch and checks length and digest against accepted snapshots.
Source mismatch or transfer failure publishes no model and releases staged
storage. A generated `CallOptions.signal` cancels an unfinished transfer.

The host's `maxPackageBytes` can impose a tighter ceiling and admits at most
1 GiB. The default host ceiling is 1 GiB, while a request without
`sourceOptions` remains limited to 64 MiB. Large files require streamed
response bodies; transport chunks go directly through a bounded mailbox into
the private WASM VFS. Uniquely staged storage transfers to the model snapshot,
whose lifetime continues through compiled models and contexts. Closing the
host aborts a partial upload before releasing the module.

This profile keeps the existing WASM32 allocation/address ABI. Source bytes
share its approximately 2 GiB signed allocation space with prepared weights,
activations, contexts and scratch. A source within budget may still fail
compilation or execution due to memory or device limits. WebGPU checks actual
buffer/binding limits and refuses an oversized unsplit tensor. Low-precision
storage, tensor partitioning and measured large-model browser qualification
are separate work.

## Choose a backend

For CPU inference, the ordinary host's default compilation is sufficient:

```javascript
const compiled = await inference.compileModel(new pb.CompileModelRequest({
  modelId: model.modelId,
}));
```

To use WebGPU, create the runtime and load the model through a `FullEngineHost`
and clients imported from `volvoxai`. Then compile with an explicit policy:

```javascript
const compiled = await inference.compileModel(new pb.CompileModelRequest({
  modelId: model.modelId,
  policy: new pb.BackendPolicy({
    mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_PREFER,
    backends: ['webgpu', 'wasm'],
    operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID,
  }),
}));
console.log(compiled.report.backend);
```

PREFER tries the candidates in order. If the graph cannot run wholly on WebGPU,
it may select WASM for the whole model. For a strict GPU requirement, use
`BACKEND_POLICY_MODE_REQUIRE` and `backends: ['webgpu']`. An inference-only host
will reject that requirement because its build has no WebGPU backend.

Selection finishes during compilation. A later execution failure does not retry
on another backend. The [operator guide](operation_list.md) and
[validation matrix](c-runtime-validation.md) describe supported domains.

## Run and read named outputs

The following fragments assume your model has F32 input `x` with shape `[1,4]`
and an F32 output named `logits`. Substitute the contracts reported by
`GetModelInfo` for your model.

```javascript
const values = Float32Array.of(1, 2, 3, 4);
const inputs = [new pb.Tensor({
  name: 'x', dtype: pb.DataType.DATA_TYPE_F32, shape: [1n, 4n],
  inline: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
})];
const context = await inference.createExecutionContext(new pb.CreateExecutionContextRequest({
  compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution(),
}));
const result = await inference.execute(new pb.ExecuteRequest({
  contextId: context.contextId,
  inputs: new pb.InputBindings({values: inputs}),
  scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()}),
}));
const output = result.outputs.find(tensor => tensor.name === 'logits');
console.log(new Float32Array(output.inline.slice().buffer));
```

Names, dtype, shape, and exact byte length are required. Use the array's
`byteOffset` and `byteLength` when making a byte view: a subarray may occupy
only part of its underlying buffer. Shape dimensions and IDs use `bigint` in
the JavaScript projection. The engine validates all inputs before committing
a new binding; matching element counts alone do not establish matching shapes.

Both CPU and GPU calls resolve after computation and device completion. WebGPU
waiting yields to the event loop. Output tensors are immutable snapshots; later
calls cannot overwrite previously delivered bytes. No result polling is needed.

The default output storage is inline. To retain engine-owned output storage,
WASM CPU supports `outputs.storage = OUTPUT_STORAGE_OWNED_BUFFER`; the returned tensors have
`BufferView` values. Read or copy them through `VxBufferServiceClient` and release
their buffer IDs with `ReleaseBuffers` after use. Those buffers remain valid after
context release. WebGPU currently returns portable INLINE snapshots; an
OWNED_BUFFER request is refused before input binding. An encoder's INLINE result
can condition a decoder after the encoder context is released. This explicitly
reads the encoder result back to host bytes and uploads the decoder input;
it does not retain device storage across those contexts.
An absent output selection selects the entrypoint's outputs;
a present selection with no names selects none, but still waits for completion.

## Independent contexts and generation

Every execution uses an explicit context. Each context retains its compiled
model and owns bindings, scratch, and mutable state. A concurrent call on the
same context returns BUSY. Independent contexts can share one compiled model.

Read `info.execution` for the model's entrypoint names, effects, allowed ranges,
invariant inputs, and state bindings. For an ordinary stateless graph, use its
EVALUATE entrypoint and `full`. Stateful Graph execution first uses INITIALIZE,
then TRANSITION. `InputBindings.values` supplies fresh tensors, `reuse` names
retained inputs, and `feedback` connects previous outputs to current inputs.
Every required input must be bound exactly once. A failed validation leaves the
previous bindings intact. Invariant inputs use `reuse` after initialization.

Use Graph `Execute` for caller-controlled sampling and EOS. Incremental models
may admit `prefixRows`, typed per-slot `slotRows`, or explicit dependency roots;
only use ranges declared for the selected entrypoint. Slot position zero is an
ordinary position, and recompute/empty are separate alternatives.

For a model with a supported generation contract, create an autoregressive
context with explicit sampling and a positive total `maxSteps`. The first
`Generate` supplies `initialInputs` and a positive chunk `budget.maxSteps`.
Subsequent calls omit `initialInputs`. Each response contains only newly selected
IDs; prompt IDs are excluded. A first chunk of N IDs performs prefill and N−1
feedback steps, retaining the last selected ID for the next call. The current
engine profile supports one categorical stream. WASM CPU also admits seeded
RANDOM, temperature/top-k/top-p, EOS IDs and generated-ID stop sequences; see
[generation policies](compact-decoder.md). WebGPU currently admits fixed-count
GREEDY and refuses these additional policies during context admission.

Generation also requires a row execution route for the complete graph. The
TinyStories fused SDPA graph supports this on CPU/WASM. On WebGPU its Graph
context uses the existing dependency route, so autoregressive context creation
returns BACKEND_UNSUPPORTED before execution. Graph caller sampling remains
available there; supported CrossSDPA device feedback graphs can use `Generate`.

Context limits are admitted with the backend's resource proof. A total
`maxMemoryBytes` cap currently supports built-in CPU/WASM contexts without paged
cache, using a conservative bound across the model's complete shape domain.
GPU/provider contexts and the combination of paged cache with a total cap return
BACKEND_UNSUPPORTED. Paged contexts separately bound pages through their cache
options. Immutable weights and caller-retained outputs have separate owners and
budgets.

Create a new context for a new job after COMPLETED, CANCELLED, or FAILED.
Configure paged cache limits at context creation and inspect cache state with
`GetExecutionContextInfo`. See [scheduling and cache usage](scheduling-and-dynamic-batching-design.md).

## Schedule concurrent requests

Create a runtime with `executionMode: pb.ExecutionMode.EXECUTION_MODE_SCHEDULED`
when requests need bounded queue admission, priority, or deadlines. Load and
compile its models as usual. `VxSchedulerServiceClient` then provides `Submit`,
`WaitRequest`, `TakeRequestResult`, and cancellation/release operations.

`Submit` carries an `execute` or `generate` request and reserves its context.
`WaitRequest` waits asynchronously for the request to complete.
`TakeRequestResult` transfers a completed tensor packet once. Inline output needs
no release; owned buffers belong to the caller after transfer. Release the Request
when finished. Cancelling a wait stops only that wait; use
`CancelRequest` to cancel the engine request.

Compatible requests may share a physical batch only when the graph preserves
lane independence. Queues share Runtime admission and dispatch; different
models do not share weights or form a physical batch together. See
[scheduling](scheduling-and-dynamic-batching-design.md) for budgets and policies.

## Errors and cleanup

Calls accept `{ signal, timeoutMs }`. Catch `VolvoxAIError` for engine failures;
its `report`, `response`, and `operation` retain the structured failure evidence.
An input rejection may include `report.inputIssue` with the offending input,
expected/actual shape, and byte counts. Compare enum codes rather than message
text. [API discovery](api-discovery.md) explains the diagnostic fields.

`RpcError` represents transport status, cancellation, or deadlines. A timeout
does not undo already submitted device work. After device loss, resources on
that device cannot be reused; create a new host and reload the model as needed.

For a reusable host, release owned output buffers, contexts, compiled models, and models as
their application lifetimes end. Release calls are idempotent, and descendants
retain the parent state they need. At the end of the whole session:

```javascript
await host.close();
```

Always put host closure in a `finally` block in a complete application, as the
[quickstart example](../examples/call_inference.mjs) does. Closing the host
cancels calls, retires its remaining IDs, and drains device resources.

## Training and quantization

Full uses the same loaded Model for inference, training, and PTQ. A Trainer
keeps private working weights; commit publishes a new revision for subsequent
compilations. Existing compilations and results keep their earlier revision.
See [model construction and training](model_builder_training.md) for a complete
example, accumulation, and checkpoint persistence.

[PTQ](quantization.md) can author a template, collect calibration observations,
and return a quantized graph and weights entirely through full WASM. Exported
bytes can be saved as downloads or in application storage. Native filesystem
output paths are unsupported in the browser.

## Packaging and browser extensions

Serve WASM as `application/wasm`. Deploy each JS profile with its matching WASM,
plus your graph and weights. Current package exports do not include TypeScript
declarations; declaration packaging is tracked in [TODO](../TODO.md).

For a repository-built runtime ZIP:

```sh
python3 tools/package_release.py --profile inference --out build/inference.zip
python3 tools/package_release.py --profile full --out build/full.zip
```

These packages contain complete profiles. Model-specific runtime/shader
pruning is deferred.

An MV3 extension packages its JS, WASM, and model assets locally. Use
`chrome.runtime.getURL(...)` when configuring asset URLs. A minimal extension
manifest includes:

```json
{
  "manifest_version": 3,
  "name": "Local model",
  "version": "1.0.0",
  "background": { "service_worker": "service-worker.js", "type": "module" },
  "content_security_policy": {
    "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'"
  }
}
```

On service-worker restart, create a new host and reload the model. IDs belong
to the previous owner and are not persistent identifiers.
`tools/test_mv3_packaged_wasm.mjs` checks both profiles, the packaged assets,
CSP, and worker restart. See [testing](testing.md) for how to run it.
