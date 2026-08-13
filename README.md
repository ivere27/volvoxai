# VolvoxAI

**A zero-dependency deep-learning runtime for browsers, Node.js, and native
Windows, Linux, macOS, and Android targets.**

VolvoxAI runs compact graph packages without embedding a general-purpose ML
framework. It supports WebNN, WebGPU, WASM SIMD, JavaScript CPU, native CPU,
Vulkan, OpenGL, optional CUDA, Metal, and NNAPI integrations.

The repository is also a from-scratch textbook:

- [Idea, Build, and Deep tracks](docs/textbook/README.md)
- [Architecture](ARCHITECTURE.md)
- [Runtime design](REFACTORING.md)

## Highlights

- One explicit inference lifecycle: Runtime → Model → CompiledModel →
  ExecutionContext → ExecutionResult.
- Bounded fixed-rank dynamic shapes with explicit concrete input views, whole-domain
  backend qualification, and context-local specialization caches.
- Stable named outputs on every backend. Host reads return caller-owned arrays;
  WebGPU results may also expose result-owned device buffers.
- Independent execution and decode contexts with immutable compiled model and
  weight revisions.
- Required or preferred backend policy with independent operator-fallback
  control and machine-readable reports.
- A single context-aware provider contract for built-in and external devices.
- A full profile with a retained Trainer for CPU, WebGPU, or strict WASM
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
no CPU, WebNN, WebGPU, WGSL, or Node filesystem implementation.

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
  Model,
  VolvoxAI,
} from 'volvoxai';

const runtime = await VolvoxAI.createRuntime({
  backends: ['webnn', 'webgpu', 'wasm', 'cpu'],
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
    order: ['webgpu', 'wasm', 'cpu'],
    operatorFallback: 'allow',
  },
});
const context = await compiled.createContext();

const result = await context.execute({
  images: {
    data: new Float32Array(2 * 224 * 224 * 3),
    shape: [2, 224, 224, 3],
  },
});
const scores = await result.output('scores').read();

await result.close();
await context.close();
await compiled.close();
await runtime.close();
~~~

ModelLoader resolves graph.json beside the first safetensors URL. Pass
graphUrl in its options when the graph is stored elsewhere; its basename must
be `graph.json` or a named `*.graph.json` document.

Compilation pins an immutable logical topology and weight revision and admits
only a provider that attests the complete bounded shape domain. Every execution
input carries explicit data and concrete shape; raw typed-array shorthand is
not accepted. Create multiple contexts from one compiled model for independent
shape specialization, request, or decode state. Each context serializes its own
accepted operations, while different contexts may progress concurrently.

ExecutionResult owns a stable snapshot of every declared graph output. A result
remains usable after later executions and after its context closes. Each read()
returns a fresh typed array. A device result may expose deviceBuffer; that
buffer remains owned by the result and must not be destroyed by the caller.
Ordinary execution may pass a live device `TensorResult` back as the `data` of
another shaped input. The built-in WebGPU provider accepts only results issued
by VolvoxAI on the same physical `GPUDevice`, with an exact matching dtype,
shape, and logical byte count. The source result must stay open until the
consumer execution has been accepted; VolvoxAI then retains it through the GPU
queue fence. CPU, WASM, WebNN, cross-device, forged, closed, and decode
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
const trainer = await Trainer.create(source, { backend: 'cpu' });

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

const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
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
