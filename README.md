# VolvoxAI

**A zero-dependency deep-learning runtime for browsers, Node.js, and native
Windows, Linux, macOS, and Android targets.**

VolvoxAI runs compact graph packages without embedding a general-purpose ML
framework. It supports WebGPU, WASM SIMD, JavaScript CPU, native CPU, Vulkan,
OpenGL, optional CUDA, and Metal integrations.

It is an **on-device edge engine for web browsers and robots**, supporting both
inference and training:
browser pages/extensions may share one Runtime across concurrent clients, while
robot deployments coordinate simultaneous vision, audio, and language streams
under explicit latency and memory bounds.

The repository is also a from-scratch textbook:

- [Idea, Build, and Deep tracks](docs/textbook/README.md)
- [Architecture](ARCHITECTURE.md)
- [Dynamic-shape ADR](docs/adr-dynamic-shape-v1.md)

## Highlights

- One explicit inference lifecycle: Runtime → Model → CompiledModel →
  ExecutionContext → ExecutionResult.
- Bounded fixed-rank dynamic shapes: a dimension symbol is **named**, so axes
  shared across tensors are one extent by construction, and **bounded**, so the
  legal domain is finite. Both are required before a backend can prove the whole
  domain at compile time instead of specializing at dispatch. Execution takes
  explicit concrete input views and uses context-local specialization caches.
  See the [dynamic-shape ADR](docs/adr-dynamic-shape-v1.md) for the alternatives
  this replaces.
- Stable named outputs on every backend. Host reads return caller-owned arrays;
  WebGPU results may also expose result-owned device buffers.
- Independent execution and decode contexts with immutable compiled model and
  weight revisions.
- Required or preferred backend policy with independent operator-fallback
  control and machine-readable reports.
- A single context-aware provider contract for built-in and external devices.
- A full profile with a retained Trainer for CPU JS, WebGPU, or strict WASM
  training.
- Inspectable model packages using graph.json and safetensors.
- Strict inference/training composition boundaries in JavaScript, WASM, and
  native builds.

## Install

~~~bash
npm install volvoxai
~~~

For repository development:

~~~bash
npm install
npm run typecheck
npm run build:all
~~~

## Release artifacts

The fixed browser release files for package version 0.4.0 are:

~~~text
dist/0.4.0/volvoxai.js
dist/0.4.0/volvoxai.min.js
dist/0.4.0/volvoxai.full.js
dist/0.4.0/volvoxai.full.min.js
dist/0.4.0/volvoxai.wasm.js
dist/0.4.0/volvoxai.wasm.min.js
dist/0.4.0/volvoxai.wasm
dist/0.4.0/volvoxai.full.wasm
~~~

The standard JavaScript entry is inference-only and resolves the forward-only
WASM sidecar. The full entry adds training and resolves volvoxai.full.wasm.
The WASM-only JavaScript entry contains strict WASM inference and training but
no CPU JS, WebGPU, WGSL, or Node filesystem implementation.

Build all browser artifacts reproducibly with:

~~~bash
make build_web
~~~

## Model packages

An inference package contains:

~~~text
graph.json
model.safetensors
~~~

Every graph root, including named subgraphs, must carry the exact
case-sensitive format discriminator:

~~~json
{
  "format": "volvox-graph/v1",
  "dimensions": {}
}
~~~

Dynamic shape is intrinsic to `volvox-graph/v1`. Every symbol has finite
bounds. Every node output uses one `{ tensor, dtype, shape }` descriptor, and
every input resolves to a declared graph input, a named fixed weight, or an
earlier output.

## Inference

~~~javascript
import {
  ExecutionMode,
  Model,
  VolvoxAI,
  executionModes,
} from 'volvoxai';

const directMode = executionModes[ExecutionMode.Direct];
const scheduledMode = executionModes[ExecutionMode.Scheduled];

const runtime = await VolvoxAI.createRuntime({
  backends: ['webgpu', 'wasm', 'cpu-js'],
  execution: {
    mode: scheduledMode,
    results: {
      maxRetainedResults: 64,
      maxRetainedOutputBytes: 64 * 1024 * 1024,
    },
  },
  onDiagnostic(event) {
    console.debug(event.kind, event.report ?? event);
  },
});

const snapshot = await Model.load(
  './models/my-model/model.safetensors',
);
const compiled = await runtime.compile(snapshot, {
  backend: {
    mode: 'prefer',
    order: ['webgpu', 'wasm', 'cpu-js'],
    operatorFallback: 'allow',
  },
});

const result = await compiled.run({
  images: {
    data: new Float32Array(224 * 224 * 3),
    shape: [1, 224, 224, 3],
  },
}, { mode: directMode });
const scores = await result.output('scores').read();

await result.close();
await compiled.close();
await runtime.close();
~~~

ModelLoader resolves graph.json beside the first safetensors URL. Pass
graphUrl in its options when the graph is stored elsewhere; its basename must
be `graph.json` or a named `*.graph.json` document.

Compilation pins an immutable logical topology and weight revision and admits
only a provider that attests the complete bounded shape domain. Every execution
input carries explicit data and concrete shape; raw typed-array shorthand is
not accepted. Stateless serving goes through `CompiledModel.run()` or
`submit()`, so one Runtime can arbitrate scheduled work across every compiled
model. The current admission gate bounds active request count, owned host-input
snapshots, the exact extra dense-stack bytes reserved for a physical batch,
and exact logical bytes/results reserved before provider output production.
Explicit contexts and decode share the same Runtime result ledger; activation,
physical device-memory and KV budgets remain separate promotion work in
[`TODO.md`](TODO.md). Explicit contexts remain the low-level owner for independent
decode/session state; they are not a second public scheduling system.

### Execution modes and dynamic batching

One Runtime owns one logical execution coordinator across all of its models.
Different models share admission and dispatch arbitration, but never weights,
mutable context state or a physical batch. Only requests with an exact
provider-produced compiled/shape route identity can be coalesced. Full
physical-device, workspace and KV accounting is an explicit promotion gate
rather than a current API guarantee.

`ExecutionMode` comes from `proto/volvoxai.proto`; `ExecutionModeValue` and
`executionModes` are generated from that enum rather than maintained as a
second handwritten list.

- `direct`: one logical call, including caller-authored bulk B=N. It bypasses
  the Runtime coordinator and creates no scheduler queue, timer, worker, request
  table or scheduler telemetry ring. Use
  `compiled.run(inputs, { mode: directMode })` for a one-shot call or
  latency/memory baseline. A busy route returns `BUSY`; there is no hybrid
  fallback. A caller that wants queue admission makes a separate SCHEDULED
  `run()` or `submit()` call. DIRECT is a global-arbitration opt-out: concurrent
  calls on different compiled routes are neither coalesced nor fairly ordered.
  It does not promise zero allocation, synchronous JavaScript completion, B=1,
  or bypass of a provider/device queue.
- `scheduled`: the default. It allocates the coordinator on first use and always
  uses bounded queue admission, priority, earliest-deadline-first arbitration,
  aging, deadlines and stateless freshness. `scheduler.maxBatchDelayMs: 0`
  dispatches work-conservingly; a positive value permits a bounded coalescing
  delay that deadlines may shorten. Browser Worker and robot service policies
  are configurations of this mode, not separate execution modes.

`submit()` is always scheduled and therefore has no per-request `mode` field.
`run()` may select DIRECT or SCHEDULED within the Runtime's configured
capability.

`submit()` returns a `RuntimeRequestHandle` with `result`, `state`,
`deadlineMissed`, `wait()` and `cancel()`. `deadlineMissed` is `null` until the
request settles; a late accepted `all`/`latest` request still publishes its
result and records `true`, while `drop-if-late` rejects:

~~~javascript
const first = compiled.submit(firstInputs);
const second = compiled.submit(secondInputs);
const [firstResult, secondResult] = await Promise.all([
  first.result,
  second.result,
]);
~~~

The result ledger defaults to 64 retained results and 64 MiB of exact logical
output storage. A reservation spans queued/in-flight work and the published
result, and is returned by `ExecutionResult.close()`. Keep result closure in
the application lifecycle: an unclosed result intentionally backpressures new
work. A queued `latest` replacement transfers its result-count slot, but the
TypeScript v1 path conservatively requires real headroom for both old and new
input/output bytes until the replacement commits; on `OVERLOADED`, the old
frame remains queued unchanged.

When the provider compiler attests an independent public batch axis and core's
typed operator proof agrees, compatible B=1 requests enter one backend call
over `[B, ...]`. A shared leading symbol alone is insufficient: an operation
such as Softmax may communicate across it. The proof follows the request axis
through every execution tensor and admits only typed configurations whose
axis, broadcast, reshape, reduction, indexing, quantization, and fixed-weight
contracts preserve lane independence. CPU JS, WASM, WebGPU, and native use the
same fail-closed protocol bound to the exact graph fingerprint. Native
Vulkan/OpenGL/CUDA built-ins then run one authored symbolic-B engine forward;
external native providers must echo the exact core proof. A provider or graph
without the required proof remains a correct B=1 route; Runtime never hides B
independent provider calls behind a batching claim. This proves explicitly
authored symbolic-B graphs only. Lifting a fixed-B=1 graph still requires the
compiler transform and full re-proof described in the
[scheduling and dynamic batching design](docs/scheduling-and-dynamic-batching-design.md#compatibility-routes-and-typed-independence-proof).
WebGPU currently
host-stacks scheduled inputs and reads each physical B=N output back once;
lanes share immutable views over that host backing, so it is not yet the
zero-copy GPU qualification. DIRECT state and the
direct/scheduled comparison are inspectable with `npm run baseline:scheduler`;
`npm run baseline:batch -- --backend=cpu-js` (or `wasm`) measures a real MatMul
route without imposing a machine-specific speed threshold.

The independence proof gates scheduler stacking and result splitting, not a
caller's explicit bulk tensor. A graph may intentionally normalize or reduce
across its authored B axis and still execute a direct B=N call; Runtime simply
does not reinterpret that call as N independent requests.

ExecutionResult owns a stable snapshot of every declared graph output. A result
remains usable after later executions and after its context or Runtime closes;
Runtime shutdown neither closes nor waits for published results. Each read()
returns a fresh typed array. A device result may expose deviceBuffer; that
buffer remains owned by the result and must not be destroyed by the caller.
Ordinary execution may pass a live device `TensorResult` back as the `data` of
another shaped input. The built-in WebGPU provider accepts only results issued
by VolvoxAI on the same physical `GPUDevice`, with an exact matching dtype,
shape, and logical byte count. The source result must stay open until the
consumer execution has been accepted; VolvoxAI then retains it through the GPU
queue fence. CPU JS, WASM, cross-device, forged, closed, and decode
seed/step device inputs fail explicitly rather than copying or falling back.

Use a strict policy when execution must stay on one provider:

~~~javascript
const compiled = await runtime.compile(snapshot, {
  backend: {
    mode: 'require',
    backend: 'webgpu',
    operatorFallback: 'forbid',
  },
});
~~~

Backend selection finishes during compilation. Execution failure is reported
and is never retried on another provider.

## Training

Training is available only from the full and WASM-only profiles. Trainer owns
gradients, optimizer slots, accumulation, and a private working
revision. `trainStep()` mutates only that private revision. `commit()` atomically
returns an immutable successor snapshot; the source snapshot and already
compiled contexts remain pinned to their original weights.

~~~javascript
import {
  ModelBuilder,
  Model,
  Trainer,
  VolvoxAI,
} from 'volvoxai/full';

const builder = new ModelBuilder({
  dimensions: { B: { min: 1, max: 8 } },
  inputs: { x: { dtype: 'float32', shape: ['B', 4] } },
  weights: [{ name: 'projection', dtype: 'float32', shape: [4, 8] }],
  nodes: [{
    id: 'projection',
    opType: 'MatMul',
    inputs: { input: 'x', weight: 'projection' },
    outputs: {
      out: { tensor: 'logits', dtype: 'float32', shape: ['B', 8] },
    },
    params: {},
  }],
  outputs: ['logits'],
});
const source = Model.capture({
  graph: builder.snapshot(),
  weights: {
    projection: {
      name: 'projection', dtype: 'float32', shape: [4, 8],
      data: Float32Array.from({ length: 32 }, (_, i) => (i - 16) / 64),
    },
  },
});
const trainer = await Trainer.create(source, { backend: 'cpu-js' });

const step = await trainer.trainStep({
  inputs: {
    x: { data: new Float32Array([1, 2, 3, 4]), shape: [1, 4] },
  },
  logitsTensor: 'logits',
  targets: new Int32Array([3]),
  trainableTensors: ['projection'],
  updateMode: 'adamw',
  optimizer: { learningRate: 1e-3, maxGradNorm: 1 },
});
const successor = await trainer.commit();

await trainer.close();

const runtime = await VolvoxAI.createRuntime({ backends: ['cpu-js'] });
const compiled = await runtime.compile(successor);
await compiled.close();
await runtime.close();
~~~

The same Trainer contract accepts backend: 'webgpu' or backend: 'wasm'. WASM
training is strict and rejects an unsupported graph before mutating weights.
There is no implicit publication: call `commit()` before compiling inference
against the update, or `rollback()` to restore the last committed baseline.
The full profile also exports logical authoring, checkpoints, gradient
accumulation controls, and PTQ authoring tools. See
[model construction and training](docs/model_builder_training.md) and the
[operation matrix](docs/operation_list.md).

## WASM-only browser extensions

For a Manifest V3 extension, package one WASM-only JavaScript variant, the full
sidecar, and the model:

~~~text
vendor/volvoxai.wasm.min.js
vendor/volvoxai.full.wasm
model/graph.json
model/model.safetensors
~~~

~~~javascript
import {
  Model,
  VolvoxAI,
} from './vendor/volvoxai.wasm.min.js';

const runtime = await VolvoxAI.createRuntime({
  wasmUrl: chrome.runtime.getURL('vendor/volvoxai.full.wasm'),
});
const snapshot = await Model.load(
  chrome.runtime.getURL('model/model.safetensors'),
  { graphUrl: chrome.runtime.getURL('model/graph.json') },
);
const compiled = await runtime.compile(snapshot, {
  backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
});
~~~

Extension pages need wasm-unsafe-eval in their content security policy. The
WASM-only release has no dynamic import and contains no alternate backend.
See [Browser and Node runtime](docs/browser-runtime.md).

## Native use

~~~bash
make build_native

./native/volvoxai --help
./native/volvoxai-full --help
~~~

The inference executable provides model-agnostic tensor execution. The full
executable additionally provides training:

~~~bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32
~~~

Raw files use a storage suffix matching their declared dtype: .f32, .i32,
.i8, or .u8. Outputs contain the complete declared tensor; applications
select task-specific rows or slices. Model-specific tokenization, image
decoding, generation, and postprocessing live under examples/.

Native releases use embedded shaders. For shader development,
VOLVOXAI_SHADER_DIR may point to generated spv/, glsl/, gles/, and metal/
directories; VolvoxAI logs once when that external override is actually used.

## Example models

Weights are not committed. Recreate the example packages from public sources:

~~~bash
make models_deps
make models_efficientdet
make models_tinystories
make validate_model_packages
~~~

See [Models and exporters](docs/models.md).

## Repository layout

~~~text
ts/core/             graph/data objects and runtime ownership
ts/ops/              operators, validation, and normalization
ts/backends/         backend providers and device resources
ts/training/         Trainer, autograd, optimizers, checkpoints, and PTQ
examples/            model-specific applications and integrations
shaders/             authoritative WGSL source
native/include/      public opaque inference/provider and full Trainer/PTQ C APIs
native/src/runtime/  runtime/model/context/result implementation
native/src/kernels/  portable and optimized CPU/WASM kernels
native/src/backends/ native device integrations
native/src/training/ full-profile training implementation
runtime/             optional in-process Synurang FFI plugin
~~~

## Documentation

- [Quickstart](docs/quickstart.md)
- [Browser and Node runtime](docs/browser-runtime.md)
- [Native runtime](docs/native-runtime.md)
- [Backend SDK](docs/backend-sdk.md)
- [Scheduling and dynamic batching design](docs/scheduling-and-dynamic-batching-design.md)
- [Model format](docs/model-format.md)
- [Graph exporter and optimizer design](docs/graph-optimizer-design.md)
- [Typed PTQ](docs/typed-ptq.md)
- [Model construction and training](docs/model_builder_training.md)
- [Operation support matrix](docs/operation_list.md)
- [Testing and validation](docs/testing.md)
- [Models and exporters](docs/models.md)
- [Textbook](docs/textbook/README.md)

## License

MIT. See [LICENSE](LICENSE).
