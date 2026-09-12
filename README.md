# VolvoxAI

**A deep-learning runtime for browsers, Node.js, and native desktop, phone,
and robot applications.**

VolvoxAI runs compact model packages without embedding a general-purpose ML
framework. The JavaScript package has no runtime npm dependencies. Use it to recognize objects, process text, or adapt a small model on
the device where the data is produced. The same graph and weights can run in a
browser through WebAssembly or in a native C application.

The engine supports both inference and training. A browser application can
coordinate concurrent requests in one runtime; an edge service can schedule
vision and language workloads with explicit request and result-memory budgets.
Model-specific preprocessing, generation policy, and evaluation stay in the
application.

This repository is also a from-scratch textbook. Start with
[How AI Actually Works](docs/textbook/README.md) ([한국어](docs/textbook/ko/README.md))
to learn tensors, attention, training, quantization, and the engine itself.
For a first run, follow the [quickstart](docs/quickstart.md).

## Highlights

- **One model package across platforms.** Inspectable `graph.json` and
  SafeTensors weights describe the computation and its data.
- **CPU and GPU execution.** Browser/Node WASM SIMD, browser WebGPU, and native
  CPU with optional Vulkan, OpenGL/OpenGL ES, CUDA, and Metal backends.
- **Bounded dynamic shapes.** Name a dimension such as sequence length and give
  it finite bounds. The compiler checks the supported domain; each request
  supplies its actual shape without rebuilding the model.
- **Independent sessions and stable results.** Reuse a compiled model across
  execution contexts. Each context owns its shape and decode state, and each
  result retains its own named output snapshot until released.
- **Scheduling and batching.** Run a single request directly, or use bounded
  queues, priorities, deadlines, and compatible-request batching. Decode
  contexts support row execution and paged KV caches on qualified routes.
- **On-device training.** The full profile provides SGD/AdamW, gradient
  accumulation, checkpoints, and LoRA authoring. Training updates private
  weights; an explicit commit publishes them for new inference compilations.
- **Post-training quantization.** Calibrate a float model and produce an explicit
  W8A8 package, then measure its accuracy and latency against the original.
- **Separate inference and full builds.** Inference artifacts exclude compiled
  training code, optimizer state, and training shaders.

Support depends on the operator, dtype, shape, and selected backend. See the
[operator guide](docs/operation_list.md) and [validation coverage](docs/c-runtime-validation.md)
for the tested domains.

## Install and choose a profile

```sh
npm install volvoxai
```

| Your application needs | JavaScript entry | WASM companion |
| --- | --- | --- |
| CPU inference, text processing, graph construction, scheduling | `volvoxai` | `volvoxai.wasm` |
| WebGPU inference, training, or PTQ | `volvoxai/full` | `volvoxai.full.wasm` |

Both entries run WASM CPU inference. WebGPU belongs to the **full** entry.
Serve the matching WASM file beside the JavaScript bundle, or supply `wasmUrl`
when your bundler or CDN puts it elsewhere. See [browser and Node deployment](docs/browser-runtime.md).

For repository development, build both JavaScript entries and their companions:

```sh
make build_web
```

This uses the repository's Docker toolchain. [Quickstart](docs/quickstart.md)
explains prerequisites, local builds, and the first runnable example.

## Model packages

A typical package contains:

```text
graph.json
model.safetensors
```

The graph describes named inputs, operations, output tensors, and dimension
bounds. SafeTensors stores the weights; larger packages may use several shards.
Every graph has the exact `"format": "volvox-graph/v1"` discriminator.
A static graph uses an empty `dimensions` object; a dynamic graph declares each
symbol's finite range. See the [model format](docs/model-format.md).

Recreate the example models from their public sources:

```sh
make models_deps
make models_efficientdet
make models_tinystories
make validate_model_packages
```

Weights are not committed. [Models and exporters](docs/models.md) explains the
sources and export options. The browser demos are
[EfficientDet](examples/efficientdet_lite0.html) and [TinyStories](examples/tinystories.html).

## Web inference

The lifecycle is **create a runtime → load a model → compile → run → read outputs**.
This small ReLU graph needs no downloaded weights. Save the example as an `.mjs`
file in an installed project or this repository and run it with Node:

```javascript
import { EngineHost, VxInferenceServiceClient, pb } from 'volvoxai';

const host = new EngineHost();
const inference = new VxInferenceServiceClient(host);
try {
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  const graphDocument = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1', dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [2] } },
    nodes: [{
      id: 'relu', opType: 'ReLU', inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2] } },
      params: {},
    }],
    outputs: ['y'],
  }));
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    package: new pb.ModelPackage({ graphDocument }),
  }));
  const compiled = await inference.compileModel(new pb.CompileModelRequest({
    modelId: model.modelId,
  }));
  const values = Float32Array.of(-2, 3);
  const result = await inference.run(new pb.RunRequest({
    compiledModelId: compiled.compiledModelId,
    inputs: [new pb.Tensor({
      name: 'x', dtype: pb.DataType.DATA_TYPE_F32, shape: [2n],
      inline: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
    })],
  }));
  const output = await inference.readOutput(new pb.ReadOutputRequest({
    resultId: result.resultId, name: 'y',
  }));
  console.log(Array.from(new Float32Array(output.tensor.inline.slice().buffer)));
  // [0, 3]
} finally {
  await host.close();
}
```

`pb` contains the request, response, and enum types. Tensor shapes use `bigint`
values such as `2n`; tensor data is an exact byte view. Inputs always state their
name, dtype, and concrete shape so the engine can validate the whole batch.

For a downloaded model, supply `graphPath` and `weightPaths`, or load their bytes
into `ModelPackage`. `GetModelInfo` reports the required inputs and outputs.
The [browser runtime guide](docs/browser-runtime.md) shows both forms, WebGPU
selection, result polling, and session cleanup. The repository also includes
[a minimal inference script](examples/call_inference.mjs).

## Scheduling and generation

DIRECT is the default for one-shot inference. Choose SCHEDULED explicitly when
several producers need queue admission and arbitration in the same Runtime.
Only requests for a compatible compiled route can share a physical batch;
batching is useful when the graph preserves each request's independence.

For text generation, prefill a context with a prompt and then advance it one
step at a time. Its KV cache retains previous attention state. Paged caches can
share prefixes and retire individual lanes. `DecodeGenerate` handles a supported
fixed-count greedy feedback loop; sampling, stop conditions, and task policy
remain application decisions. See [scheduling and dynamic batching](docs/scheduling-and-dynamic-batching-design.md).

## Full-profile training

Training follows **construct/load → train → evaluate → commit or roll back → save**.
The Trainer owns private parameters, gradients, and optimizer state. Existing
compiled inference models keep their earlier weights after a commit; compile
again when you want to serve the new revision.

Use `FullEngineHost` and `VxTrainingServiceClient` from `volvoxai/full`.
The [training guide](docs/model_builder_training.md) walks through building a
small classifier, running AdamW, saving a checkpoint, and resuming it.
It also covers multiple losses, accumulation, shape-cache limits, and LoRA.

The [quantization guide](docs/quantization.md) continues from a float package to
calibration and W8A8 export. Full WASM supports the complete PTQ workflow using
bytes; your browser application chooses how to save them.

## Native use

```sh
make build_native_profiles
./native/volvoxai --help
./native/volvoxai-full --help
```

Both executables run named raw tensors. Full additionally provides `train`.
For example, after the [quickstart](docs/quickstart.md#4-run-tinystories-from-native-c)
prepares a TinyStories package and six I32 tokens and positions:

```sh
./native/volvoxai run models/tinystories_1m \
  --input 'tokens[1,6]=build/quickstart/tokens.i32' \
  --input 'positions[1,6]=build/quickstart/positions.i32' \
  --output logits=build/quickstart/logits.f32
```

Use the [native guide](docs/native-runtime.md) for input preparation, CPU threads,
backend selection, embedding, and macOS/Android builds. The
[task CLI](examples/native_task_cli/README.md) adds image decoding and detection.
Native releases embed shaders; `VOLVOXAI_SHADER_DIR` provides a development
override and logs once when used.

## Documentation and API reference

The [documentation index](docs/README.md) separates learning material from
integration and engine-development guides. Useful starting points:

- [Quickstart](docs/quickstart.md) and [textbook](docs/textbook/README.md)
- [Model format](docs/model-format.md) and [operator support](docs/operation_list.md)
- [Architecture](ARCHITECTURE.md) and [backend development](docs/backend-sdk.md)
- [Testing](docs/testing.md) and [profiling](docs/profiling.md)

The schema in [volvoxai.proto](proto/volvoxai.proto) defines the shared API for
applications and AI agents. For field-level lookup and programmatic discovery,
use [API discovery](docs/api-discovery.md) and the generated
[inference](docs/generated/api-contract.inference.md) / [full](docs/generated/api-contract.full.md)
references. [Runtime integration](runtime/README.md) covers C/Python bindings
and regeneration. These references complement the task-oriented guides above.

## Build and verify

```sh
npm ci
npm run build:all             # build JS first; this removes stale WASM companions
make build_wasm              # both WASM profiles
make build_native_profiles   # both native profiles
npm run test:proto-api
npm run test:wasm-ptq
npm run test:wasm-training-smoke
make test_native
npm run check:release
```

The fixed release inventory is four JS files (`volvoxai.js`, `volvoxai.min.js`,
`volvoxai.full.js`, `volvoxai.full.min.js`) and two WASM files (`volvoxai.wasm`,
`volvoxai.full.wasm`) under `dist/<package-version>/`, plus `native/volvoxai`
and `native/volvoxai-full`. See [testing](docs/testing.md) for release gates and
[deployment](docs/browser-runtime.md#packaging-and-browser-extensions) for runtime ZIPs and extensions.

## License

MIT. See [LICENSE](LICENSE).
