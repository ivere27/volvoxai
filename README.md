# VolvoxAI

**A zero-dependency, bare-metal deep learning inference engine for the browser,
Node.js, and native Windows/Linux/macOS/Android targets.**

VolvoxAI runs exported neural-network graphs without shipping a full ML runtime.
Train in PyTorch or another framework, export to a small Volvox blueprint plus
safetensors weights, and run inference through WebNN, WebGPU, WASM SIMD, pure JS,
or a freestanding native C binary.

The project is inference-only. It is built for small, inspectable model packages,
constrained web apps, extensions, local tools, and edge devices where heavyweight
runtimes such as ONNX Runtime Web or TensorFlow.js are too large or too opaque.

## Highlights

- Browser tiers: WebNN, WebGPU, WASM SIMD, and pure-JS CPU fallback.
- Zero runtime dependency path: vanilla JS plus WGSL with optional freestanding WASM.
- Small runtime artifacts: about 167 KiB minified JS, 55 KiB WASM, and a
  roughly 695 KiB native Linux executable in current builds.
- Native runtime: freestanding C with dynamic Vulkan, OpenGL, Metal, and Android
  NNAPI loading.
- Optimization-focused internals: operator fusion, weight packing, quantized CPU
  islands, vectorized WGSL/C kernels, and backend-specific dispatch.
- Model format: `config.json` graph blueprint plus `model.safetensors` weights.
- Shader source: WGSL for browser WebGPU and native shader generation.
- Examples: EfficientDet Lite0 object detection and TinyStories-1M text generation.
- License: MIT.

## Install

```bash
npm install volvoxai
```

For local development from this repository:

```bash
npm install
npm run build:all
```

The browser bundle is emitted to:

```text
dist/volvoxai.js
dist/volvoxai.min.js
```

The WASM tier also needs `volvoxai.wasm`. The reproducible Docker build copies it
into `dist/`:

```bash
make build_web
```

## Browser Usage

```javascript
import { VolvoxAI } from './volvoxai.js';

const engine = await VolvoxAI.init(); // auto: WebNN, WebGPU, WASM, CPU
const graph = await engine.loadGraph('./models/my-model/model.safetensors');
const executor = await engine.compile(graph);

const inputs = {
  images: new Float32Array(1 * 224 * 224 * 3),
};

const output = await executor.execute(inputs);
```

On WebGPU, `execute()` currently returns a `GPUBuffer` for the final node's first
output. WASM and CPU return a map keyed by `graph.outputNames`.

## Native Usage

```bash
make build_native

./native/volvoxai generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" \
  --max-new 50
```

Generic tensor execution:

```bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 \
  --last-token 4
```

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
js/          Browser and Node engine implementation
shaders/     WGSL compute shaders
native/      Freestanding C engine and native backends
runtime/     Rust service wrapper around the C engine
examples/    Browser examples
tools/       Export, shader, and test utilities
docs/        Detailed documentation
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Browser and Node runtime](docs/browser-runtime.md)
- [Native runtime](docs/native-runtime.md)
- [Model format](docs/model-format.md)
- [Models and exporters](docs/models.md)
- [Operation support matrix](docs/operation_list.md)
- [Testing and validation](docs/testing.md)
- [Roadmap](docs/roadmap.md)
- [Textbook walkthrough](docs/textbook/README.md) ([한국어](docs/textbook/ko/README.md))
- [EfficientDet benchmark notes](docs/efficientdet_tflite_vs_volvoxai.md)

## Current Status

VolvoxAI can run real browser and native inference paths, but it is still early.
Known gaps include WebGPU multi-output readback, broader WebNN coverage, additional
native/WebGPU shaders for a few fallback ops, browser-side generation helpers, and
formal CI wiring for the existing smoke tests.

See [docs/roadmap.md](docs/roadmap.md) for the detailed list.

## License

MIT. See [LICENSE](LICENSE).
