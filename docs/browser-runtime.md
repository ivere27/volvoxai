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
import { EngineHost, VxInferenceServiceClient, pb } from 'volvoxai';

const host = new EngineHost();
const inference = new VxInferenceServiceClient(host);
const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
```

The default companion URL is relative to the JavaScript module. Set `wasmUrl`
when a bundler or CDN moves it:

```javascript
const host = new EngineHost({
  wasmUrl: new URL('/assets/volvoxai.wasm', location.href),
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
console.log(info.inputs, info.outputs);
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
adapt model paths to your storage. Transfers, including inline packages, have
a cumulative 64 MiB ceiling; `maxPackageBytes` can tighten it. Streaming fetches
stop at the actual byte budget. Publish complete graph/weight revisions together.

## Choose a backend

For CPU inference, the ordinary host's default compilation is sufficient:

```javascript
const compiled = await inference.compileModel(new pb.CompileModelRequest({
  modelId: model.modelId,
}));
```

To use WebGPU, create the runtime and load the model through a `FullEngineHost`
and clients imported from `volvoxai/full`. Then compile with an explicit policy:

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
const result = await inference.run(new pb.RunRequest({
  compiledModelId: compiled.compiledModelId, inputs,
}));
```

Names, dtype, shape, and exact byte length are required. Use the array's
`byteOffset` and `byteLength` when making a byte view: a subarray may occupy
only part of its underlying buffer. Shape dimensions and IDs use `bigint` in
the JavaScript projection. The engine validates all inputs before committing
a new binding; matching element counts alone do not establish matching shapes.

CPU execution normally returns READY immediately. GPU execution can return
PENDING. Submit once, then query the same result until it completes:

```javascript
try {
  let state = result.state;
  while (state === pb.ResultState.RESULT_STATE_PENDING) {
    await new Promise(resolve => setTimeout(resolve, 1));
    state = (await inference.getResult(new pb.ResultRef(result))).state;
  }
  const output = await inference.readOutput(new pb.ReadOutputRequest({
    resultId: result.resultId, name: 'logits',
  }));
  const logits = new Float32Array(output.tensor.inline.slice().buffer);
  console.log(logits);
} finally {
  await inference.releaseResult(new pb.ResultRef(result));
}
```

`ReadOutput` returns BUSY while pending. Each result owns a stable snapshot of
all declared outputs, so later executions cannot overwrite an earlier result.
Host reads return bytes you can keep after releasing the result. Holding a
result consumes its Runtime's result-memory budget; release it when finished.

## Independent execution and decode contexts

Use `Run` for stateless requests. Create explicit contexts when you want to
retain shape/decode state or manage several sessions independently:

```javascript
const first = await inference.createExecutionContext(
  new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId }));
const second = await inference.createExecutionContext(
  new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId }));
const result = await inference.execute(new pb.ExecuteRequest({
  contextId: first.contextId, inputs,
}));
// Read/release result as above; second has its own mutable state.
```

Each context retains its compiled model and owns bindings, scratch, and decode
state. Serialize operations that mutate one context. Separate contexts can
hold different legal shapes without changing the shared model definition.

Text generation has a prefill phase and a decode phase. `DecodePrefill` consumes
the prompt; `DecodeStep` advances the retained session. Omit its cursor to
advance normally. Lane actions can advance, idle, or park individual lanes.
`ResetDecode` clears the session for another prompt. Inputs and outputs remain
model-defined; the caller supplies tokenization and stopping policy.

An explicit `dependencyUpdate: new pb.Empty()` refreshes whole tensors in the
changed-input dependency closure without advancing the cursor. It requires a
prefilled, single-lane AUTO context without paged KV; an empty input list does
no work. Required-row execution and paged-cache operations have separate rules.
See [decode and paged KV](scheduling-and-dynamic-batching-design.md#decode-contexts-and-paged-kv).

## Schedule concurrent requests

Create a runtime with `executionMode: pb.ExecutionMode.EXECUTION_MODE_SCHEDULED`
when requests need bounded queue admission, priority, or deadlines. Load and
compile its models as usual. `VxSchedulerServiceClient` then provides `Submit`,
`WaitRequest`, `TakeRequestResult`, and cancellation/release operations.

`WaitRequest` waits asynchronously. Once the request completes,
`TakeRequestResult` transfers its output to an ordinary result handle for
reading and release. Releasing the RPC call is different from releasing the
request or result object. Cancelling a wait stops only that wait; use
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

For a reusable host, release results, contexts, compiled models, and models as
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
