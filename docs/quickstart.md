# Quickstart

VolvoxAI has three common entry points:

- Browser runtime: import `dist/volvoxai.js`, optionally serve `dist/volvoxai.wasm`,
  and load a model package containing `config.json` plus `model.safetensors`.
- Node CLI: smoke-test model loading on the WASM or CPU tier.
- Native CLI: build `native/volvoxai` and run tensor, image, generation, or task
  wrappers directly.

## Build the Browser Bundle

The reproducible path uses the Docker image from this repository:

```bash
make build_web
```

That builds:

```text
dist/volvoxai.js
dist/volvoxai.min.js
dist/volvoxai.wasm
dist/v<version>/volvoxai.js
dist/v<version>/volvoxai.min.js
dist/v<version>/volvoxai.wasm
```

For local development without Docker:

```bash
npm install
npm run build:all
```

The local build bundles WGSL shaders into the JS file using esbuild's text loader.
It does not compile the WASM module; use `make build_wasm` or copy an existing
`volvoxai.wasm` into `dist/` when testing the WASM tier in examples.

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

On WebGPU, `execute()` currently returns a single `GPUBuffer` for the final node's
first output. Use `executor.readBuffer(buffer, byteLength)` to map it back. WASM
and CPU return an object keyed by `graph.outputNames`.

## Node CLI Smoke Test

`bin/volvox.js` is a small development entry point. It verifies initialization,
model loading, and compilation on WASM or CPU; it is not the production inference
path.

```bash
node bin/volvox.js info
node bin/volvox.js run --model ./models/tinystories_1m/model.safetensors --backend wasm
```

## Native Build

```bash
make build_native
./native/volvoxai --help
```

Example graph run:

```bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 \
  --last-token 4
```

Example text generation:

```bash
./native/volvoxai generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" \
  --max-new 50
```

See [native-runtime.md](native-runtime.md) for native task wrappers and accelerator
flags.

## Get Example Models

The `models/` directory is ignored because weights are large. Regenerate it from
public sources:

```bash
make models_deps
make models_efficientdet
make models_tinystories
```

Use `ONLY=int8 make models_efficientdet` to fetch and export only the int8
EfficientDet package.

See [models.md](models.md) for model export details.

