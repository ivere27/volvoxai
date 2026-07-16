# VolvoxAI

**A zero-dependency, bare-metal deep learning engine for the
browser, Node.js, and native Windows/Linux/macOS/Android targets.**

> 🧭 **New here — want to *understand* how AI actually works, not just use a library?**
> This repo doubles as a from-scratch **textbook** built on its own real code. Pick your path:
>
> - 🌱 **Just curious what AI really is?** → [Start the **Idea track**](docs/textbook/README.md) — plain words, analogies, **no code or math required**. A motivated 11-year-old can follow it.
> - 🔧 **Can code a little and want to see it run?** → [The **Build track**](docs/textbook/README.md) — the same ideas in graphs, JavaScript, and operators.
> - 🔬 **A developer who wants the engine?** → [The **Deep track**](docs/textbook/README.md) + [ARCHITECTURE.md](ARCHITECTURE.md) — quantization, native, optimization, and training internals.
>
> One book, three depths. Everything below this line is the **product / release reference** for people who just want to install and ship.

---

VolvoxAI runs and trains neural-network graphs without shipping a full ML
runtime. Load a small Volvox blueprint plus safetensors weights, or use the full
entry to create an empty model, initialize its parameters, and build it entirely
through the API.
The resulting graph can run through WebNN, WebGPU, WASM SIMD, pure JS, or a
freestanding native C binary.

The project is built for small, inspectable model packages, constrained web
apps, extensions, local tools, and edge devices where heavyweight runtimes such
as ONNX Runtime Web or TensorFlow.js are too large or too opaque. Training is an
explicit path: inference does not allocate gradients, optimizer state, or
backward pipelines. The inference bundle has no training dependency; the full
bundle adds training as a separate public entry.

## Highlights

- Runs in browsers, Node.js, and native applications through WebNN, WebGPU,
  WASM, JavaScript CPU, native CPU, Vulkan, OpenGL, and Metal.
- Compact current Linux x86-64 artifacts: 360 KiB WASM-only JS, 683 KiB
  multi-backend inference JS, 174 KiB inference WASM, and 936 KiB native
  inference; full training builds remain about 1–1.4 MiB.
- No external ML runtime: models use inspectable `config.json` graphs and
  safetensors weights.
- Clean inference/training separation: inference builds contain no autograd,
  optimizer state, backward shaders, or public training symbols.
- Built-in model construction, training, LoRA, checkpointing, PTQ, and portable
  W8A8 execution.
- Extensible browser and native backend APIs with embedded native shaders and
  portable CPU fallback.
- Includes EfficientDet, TinyStories, multimodal examples, and a three-level
  textbook.

## Install

```bash
npm install volvoxai
```

For local development from this repository:

```bash
npm install
npm run typecheck
npm run build:all
```

The TypeScript sources are checked before esbuild emits the six fixed-name
JavaScript bundles; the build preserves an existing WASM sidecar. For version
0.2.0, the complete browser release consists of:

```text
dist/0.2.0/volvoxai.js            # readable inference
dist/0.2.0/volvoxai.min.js        # minified inference
dist/0.2.0/volvoxai.full.js       # readable inference + training
dist/0.2.0/volvoxai.full.min.js   # minified inference + training
dist/0.2.0/volvoxai.wasm.js       # readable WASM-only inference + training/PTQ
dist/0.2.0/volvoxai.wasm.min.js   # minified WASM-only inference + training/PTQ
dist/0.2.0/volvoxai.wasm          # forward kernels used by the WASM backend
dist/0.2.0/volvoxai.full.wasm     # forward kernels plus C training/PTQ ABIs
```

`volvoxai.wasm` deliberately exports no training symbol. `volvoxai.full.wasm`
keeps every forward export and adds C loss, backward, gradient utility, SGD, and
AdamW operators plus generic PTQ observation, affine quantization, weight
packing, and bias packing. Quantized LoRA synchronization is one higher-level
use of those reusable kernels. From a clean checkout, the reproducible Docker
build creates all eight files:

```bash
make build_web
```

The npm `prepack` check rejects a missing sidecar or stale extra artifact.
`npm run build:all` bundles JavaScript but cannot compile C/WASM from a clean
checkout; use `make build_web` before `npm pack` or `npm publish`.

## Browser Usage

```javascript
import { VolvoxAI } from './dist/0.2.0/volvoxai.js';

const engine = await VolvoxAI.init(); // auto: WebNN, WebGPU, WASM, CPU
const graph = await engine.loadGraph('./models/my-model/model.safetensors');
const executor = await engine.compile(graph);

const inputs = {
  images: new Float32Array(1 * 224 * 224 * 3),
};

const output = await executor.execute(inputs);
```

Every input referenced by a blueprint node must be declared in `config.inputs`,
loaded as a named weight, or produced by an earlier node. The loader does not
invent a default image input or shape for an undeclared name.

On WebGPU, `execute()` currently returns a `GPUBuffer` for the final node's first
output. The executor owns GPU buffers; core `Tensor` objects contain portable
descriptors and optional CPU storage, not device handles. WASM and CPU return a
map keyed by `graph.outputNames`. All four browser engines share the versioned
backend lifecycle, named registration hook, and decode-session facade described
in [Browser and Node runtime](docs/browser-runtime.md#javascript-backend-contract).

### WASM-only Chrome extensions, training, and PTQ

For a Manifest V3 extension that needs no CPU, WebNN, WebGPU, or shader code,
ship exactly one JavaScript variant, the full WASM sidecar, and the model:

```text
vendor/volvoxai.wasm.min.js
vendor/volvoxai.full.wasm
model/config.json
model/model.safetensors
```

`volvoxai.wasm.min.js` names the only selectable backend, not a forward-only
capability set. It uses `volvoxai.full.wasm` because updating LoRA A/B still
requires backward propagation through the surrounding graph. The ordinary
`volvoxai.wasm` sidecar remains forward-only for the standard inference entry.
The `./wasm` and `./wasm/min` package subpaths are browser-only and deliberately
omit Node's filesystem loader; Node applications should use `.` or `./full`
and select the WASM backend.

```javascript
import { VolvoxAI } from './vendor/volvoxai.wasm.min.js';

const runtime = await VolvoxAI.init(
  'wasm',
  chrome.runtime.getURL('vendor/volvoxai.full.wasm'),
);
const graph = await runtime.loadGraph(
  chrome.runtime.getURL('model/model.safetensors'),
);
const executor = await runtime.compile(graph);

const step = await runtime.trainLoRAStep(graph, {
  inputs: teacherForcedInputs,
  logitsTensor: 'logits',
  targets: correctedTokenIds,
  trainableTensors: ['decoder.lora_a', 'decoder.lora_b'],
  updateMode: 'adamw',
  optimizer: { learningRate: 1e-4 },
});

// Applied updates refresh packed WASM weights, so this executor observes A/B.
const corrected = await executor.execute(nextInputs);
```

The same runtime exposes the existing stateless C PTQ implementation as a
generic typed toolkit, independent of LoRA:

```javascript
const ptq = await runtime.createPTQ();
try {
  const observer = ptq.createObserver();
  observer.observe(calibrationValues); // Float32Array; repeat for more samples

  const parameters = observer.parameters({
    dtype: 'int8',
    scheme: 'symmetric',
  });
  const activation = ptq.quantize(values, parameters);
  const weight = ptq.packWeight(weightValues, [outputSize, inputSize], { axis: 0 });
  const bias = ptq.packBias(biasValues, parameters.scale, weight.scales);
} finally {
  ptq.dispose();
}
```

These calls run in a private scratch WASM instance and return caller-owned
typed arrays. They do not rewrite a graph or choose how an application stores
or deploys the result. Reuse one toolkit across related operations and call
`dispose()` when finished so its isolated WASM memory can be garbage-collected.
Developers may use it for calibration, conversion, custom model builders, or
their own update workflow. Browser safetensors and graph/package authoring
remain JavaScript orchestration rather than C file I/O.

The model must represent LoRA A/B as explicit initialized F32 graph weights and
wire them through its low-rank MatMul/Add branch. Listing only those names in
`trainableTensors` freezes the base model. A corrected answer string is
application policy: tokenize it and construct teacher-forced model inputs,
target token IDs, and any loss mask before calling `trainLoRAStep()`.
Immutable staged adapter snapshots are deployment/routing objects, not
autograd parameters; checkpoint or export the updated explicit graph factors
after training.

To retain an existing W8A8 inference topology, use a separate supported F32
training graph as the persistent master and bind its A/B factors to the I8
factor weights already present in the inference graph:

```javascript
const trainer = await runtime.createQuantizedLoRATrainer(
  f32TrainingGraph,
  w8InferenceGraph,
  {
    bindings: [
      { master: 'decoder.lora_a', target: 'decoder.lora_a.i8', transpose: true },
      { master: 'decoder.lora_b', target: 'decoder.lora_b.i8', transpose: true },
    ],
  },
);

try {
  // Use this once only when starting from a W8 snapshot without an F32
  // checkpoint. Do not dequantize again after training begins.
  await trainer.initializeMastersFromQuantized();

  await trainer.trainStep({
    inputs: teacherForcedInputs,
    logitsTensor: 'logits',
    targets: correctedTokenIds,
    trainableTensors: ['decoder.lora_a', 'decoder.lora_b'],
    updateMode: 'adamw',
    optimizer: { learningRate: 1e-4 },
  });

  const corrected = await trainer.engine.execute(nextQuantizedInputs);
} finally {
  await trainer.dispose();
}
```

The C conversion helper transposes the builder's IN_OUT factors into canonical
OUT_IN I8 weights and recomputes symmetric axis-0 scales. The JavaScript trainer
stages every converted factor before atomically updating the graph, then
refreshes the WASM raw bytes, scale metadata, and packed Q8 caches without
changing nodes or activation descriptors. Version 0.2.0 requires I8 targets
with zero points of zero and all-zero I32 LoRA biases. Persist the F32
checkpoint and optimizer state as the resumable authority; the W8 graph is an
inference snapshot. After restoring an F32 checkpoint, call `trainer.sync()`
instead of `initializeMastersFromQuantized()`.

This is F32-master LoRA requantization, not QAT or backward support for a deep
W8A8 graph. `QLinear`/`QGemm` are still rejected by strict WASM training, so
the separate training graph must provide the supported F32 backward path. The
inference graph must already contain its quantized LoRA branch; this API does
not rewrite graph topology.

The extension must package all executable code locally and enable WebAssembly
for extension pages. Use an ES-module service worker and this CSP:

```json
{
  "manifest_version": 3,
  "background": { "service_worker": "service-worker.js", "type": "module" },
  "content_security_policy": {
    "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self';"
  }
}
```

The WASM-only release bundle contains no dynamic `import()`, which Chrome
extension service workers do not support. `web_accessible_resources` is not
needed when only extension-owned pages/workers fetch the packaged model and
sidecar; declare the narrow resources explicitly if a normal web page must
fetch them. See [Browser and Node runtime](docs/browser-runtime.md#wasm-only-manifest-v3-extensions)
for complete packaging notes.

## Training from APIs

Use the full entry when calling training, checkpoint, or gradient-accumulation
APIs:

```javascript
import { VolvoxAI } from './dist/0.2.0/volvoxai.full.js';
```

In this module, the familiar `VolvoxAI`, `Graph`, and `ModelBuilder` exports are
the training-capable variants; the explicit `TrainingVolvoxAI`,
`TrainingGraph`, and `TrainingModelBuilder` names are also available.
Initializers, optimizer state, and training-only builder helpers are deliberately
absent from the inference entry.

Models can start from an empty graph and initialized weights; no PyTorch export
or seed safetensors file is required. The JavaScript builder provides generic
GroupNorm, MoE and routed bottleneck adapters, deterministic Dropout, and
explicit trainable-tensor selection. The repository's
`examples/seq2seq_training/Seq2SeqBuilder.js` composes those primitives into an
encoder-decoder with multimodal source features and teacher forcing; that
model-family policy is not exported by either package entry.

`trainStep()` accepts either the legacy single cross-entropy target or a
`losses` list with independent logits, targets, weights, masks, and normalizers.
Repeated logits tensors are allowed and their gradients add. Accumulation has
reset/flush controls, and `maxGradNorm` clips one global norm over all trainable
tensors. JavaScript CPU and WebGPU regenerate the same SDPA/CrossSDPA
attention-dropout mask in forward and backward; inference never applies it.

For an explicit C-backed browser training path, initialize the full entry with
the full sidecar and set `backend: "wasm"`. It is strict: its current portable
contracts are listed in [the operation status reference](docs/operation_list.md).
Unsupported or non-canonical layouts are rejected before any model state
changes, including ambiguous square linear layouts.

Native CPU, Vulkan, OpenGL compute, and Metal support deterministic standalone
Dropout training while keeping inference as an identity. Their SDPA/CrossSDPA
training paths also implement after-softmax attention-probability dropout and
regenerate the mask during backward; unsupported GPU layouts fall back to the
matching complete native CPU path. See
[model construction, routing, and training](docs/model_builder_training.md) and
the [operation matrix](docs/operation_list.md) for exact backend limits.

## Native Usage

```bash
make build_native

# inference-only executable
./native/volvoxai --help

# inference + training executable
./native/volvoxai-full --help
```

The fixed release executables stay model-agnostic: both expose `run`, and only
`volvoxai-full` additionally exposes `train`. They do not choose vocabulary
files, decode images, or implement generation and task postprocessing.

Generic tensor execution:

```bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 \
  --row 4
```

Raw input and output filenames must end in the declared storage dtype suffix:
`.f32`, `.f16`, `.i32`, `.i8`, or `.u8`. Row output is currently F32-only.

Generic cross-entropy training is available only in the full executable:

```bash
./native/volvoxai-full train models/my_model \
  --input input=batch.f32 \
  --targets targets.i32 \
  --logits logits \
  --trainable classifier.weight \
  --trainable classifier.bias \
  --steps 10 \
  --learning-rate 0.001 \
  --output-weights trained.safetensors \
  --output-optimizer optimizer.safetensors
```

Use `--input-optimizer` to resume saved optimizer state. Run
`./native/volvoxai-full train --help` for all optimizer and backend options.

Model-facing native task wrappers are an opt-in example:

```bash
make -C examples native_task_cli

examples/target/bin/volvoxai-tasks generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" \
  --max-new 50
```

That example owns image decoding, vocabulary-file selection, generation loops,
and the `generate`, `classify`, `detect`, `ctc`, `seq2seq`, and `chat` commands.

Native executables do not need a shader directory. For shader development,
point `VOLVOXAI_SHADER_DIR` at a generated directory containing `spv/`,
`glsl/`, `gles/`, and `metal/`; VolvoxAI logs once when that override is used.

## Example Models

Model weights are not committed. Regenerate the example packages from public
sources:

```bash
make models_deps
make models_efficientdet
make models_tinystories
```

See [docs/models.md](docs/models.md) for export details.

## Repository Layout

```text
ts/core/             TypeScript inference graph/data objects and model-agnostic orchestration
ts/ops/              TypeScript operators plus graph validation and normalization
ts/backends/         TypeScript CPU, WASM, WebGPU, and WebNN execution/device resources
ts/training/         TypeScript training graphs/builders, autograd, optimizers, checkpoints
examples/             Model-specific applications and reference integrations
shaders/inference/   Forward WGSL sources
shaders/training/    Backward and training WGSL sources
native/include/      Public C APIs
native/src/shader_store.*  Lazy embedded-shader asset loader
native/src/runtime/  Model state, graph, memory, and execution
native/src/kernels/  Portable and optimized CPU/WASM kernels
native/src/backends/ Vulkan, OpenGL, Metal, and NNAPI integrations
native/src/training/ Training-specific orchestration
native/cli/          Fixed model-agnostic command-line applications
native/tests/        Native tests
runtime/             Rust service wrapper around the C engine
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for dependency and build-composition
rules.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Browser and Node runtime](docs/browser-runtime.md)
- [Native runtime](docs/native-runtime.md)
- [Custom backend SDK](docs/backend-sdk.md)
- [Model format](docs/model-format.md)
- [W8A8 safetensors companion scales](docs/w8a8-safetensors.md)
- [Post-training quantization](docs/quantization.md)
- [Model construction, routing, and training](docs/model_builder_training.md)
- [Models and exporters](docs/models.md)
- [Operation support matrix](docs/operation_list.md)
- [Testing and validation](docs/testing.md)
- [Roadmap](docs/roadmap.md)
- [Textbook walkthrough](docs/textbook/README.md) ([한국어](docs/textbook/ko/README.md))
- [EfficientDet benchmark notes](docs/efficientdet_tflite_vs_volvoxai.md)

## Current Status

VolvoxAI can run real browser and native inference paths, but it is still early.
Known gaps include WebGPU multi-output readback, broader WebNN coverage, additional
native/WebGPU shaders for a few fallback ops, browser-side generation helpers,
and formal CI wiring for the existing smoke tests.

See [docs/roadmap.md](docs/roadmap.md) for the detailed list.

## License

MIT. See [LICENSE](LICENSE).
