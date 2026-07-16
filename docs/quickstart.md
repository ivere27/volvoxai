# Quickstart

VolvoxAI has three common entry points:

- Browser inference: import `dist/<version>/volvoxai.js`, optionally serve its
  adjacent `volvoxai.wasm`, and load a model package containing `config.json`
  plus `model.safetensors`. Import `volvoxai.full.js` instead for inference plus
  training and serve its adjacent `volvoxai.full.wasm`; `.min.js` variants are
  emitted alongside both readable bundles. For a WASM-only extension, import
  `volvoxai.wasm.js` (or `.min.js`) and package `volvoxai.full.wasm`.
- Node CLI: smoke-test model loading on the WASM or CPU tier.
- Native CLI: build `native/volvoxai` (inference) and `native/volvoxai-full`
  (inference plus training) for model-agnostic tensor execution. Build the
  separate native task example for image, generation, and postprocessing
  wrappers.

## Build the Browser Bundle

The reproducible path uses the Docker image from this repository:

```bash
make build_web
```

That builds:

```text
dist/0.2.0/volvoxai.js
dist/0.2.0/volvoxai.min.js
dist/0.2.0/volvoxai.full.js
dist/0.2.0/volvoxai.full.min.js
dist/0.2.0/volvoxai.wasm.js
dist/0.2.0/volvoxai.wasm.min.js
dist/0.2.0/volvoxai.wasm
dist/0.2.0/volvoxai.full.wasm
```

For local development without Docker:

```bash
npm install
npm run build:all
```

The local build bundles WGSL shaders into the JS file using esbuild's text loader.
It does not compile the WASM modules; use `make build_wasm` or copy the needed
inference or full sidecar into `dist/<version>/` when testing the WASM tier.

`volvoxai.js` has no training dependency. For training, import the full bundle:

```javascript
import { VolvoxAI } from './dist/0.2.0/volvoxai.full.js';
```

The inference bundle uses forward-only `volvoxai.wasm`. The full bundle uses
`volvoxai.full.wasm`, which adds strict C-backed training for the documented
portable subset; JavaScript CPU and WebGPU remain available for broader graphs.
The WASM-only JavaScript bundle also uses `volvoxai.full.wasm`, but exposes no
CPU/WebGPU/WebNN backend or shader/PTQ implementation.
It is a browser-only distribution: the `./wasm` and `./wasm/min` package
subpaths omit Node's filesystem loader to keep extension service workers free
of dynamic imports. Node callers should use the standard or full entry and
select the WASM backend.

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
./native/volvoxai-full --help
```

`make build_native` builds both profiles (`native/volvoxai` and
`native/volvoxai-full`) via CMake.

Both fixed executables provide `run`; the full profile additionally provides
`train`. Apart from `--help` and `--version`, those are their complete command
sets. Run `volvoxai-full train --help` for the generic cross-entropy,
trainable-tensor, optimizer, and checkpoint options.

Example graph run:

```bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 \
  --row 4
```

The fixed runner requires raw file suffixes to match tensor storage dtype
(`.f32`, `.f16`, `.i32`, `.i8`, or `.u8`); `--row` output is F32-only.

Example text generation:

```bash
make -C examples native_task_cli

examples/target/bin/volvoxai-tasks generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" \
  --max-new 50
```

Image decoding, vocabulary selection, generation loops, and task postprocessing
belong to that opt-in example rather than the fixed release executables. See
[native-runtime.md](native-runtime.md) for its commands and accelerator flags.

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

EfficientDet acquisition, labels, and its direct-export smoke test are owned by
[`../examples/efficientdet_lite0/`](../examples/efficientdet_lite0/).

See [models.md](models.md) for model export details.
