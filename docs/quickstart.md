# Quickstart

This walkthrough gets a graph running before you download a model. Then it
shows how to prepare TinyStories or EfficientDet and where to continue for
training, WebGPU, or native deployment. Run shell commands from the repository
root.

## 1. Build the runtime

The repository uses Docker for its reproducible C/WASM and JavaScript toolchain.
With Docker available, run:

```sh
make build_web
```

This builds both JavaScript profiles and their matching WASM files in
`dist/0.6.0/`. For local JavaScript development with Node/npm installed:

```sh
npm ci
npm run build:all
make build_wasm
```

Build JS before WASM: the JS build removes stale companions. An installed npm
consumer can instead use `npm install volvoxai` and the packaged artifacts.

Choose `volvoxai` for WASM CPU inference. Choose `volvoxai/full` for WebGPU
inference, training, or PTQ. Both profiles include text processing, graph
construction, and scheduling. Each JS entry also has a `.min.js` variant.

## 2. Execute a tiny graph

```sh
node examples/call_inference.mjs
```

Expected output:

```text
[ 3, -7 ]
```

The script creates a two-element, weightless graph whose output is its input.
It loads the graph, compiles for WASM, supplies an F32 tensor, reads the result,
and closes the host. No model download or GPU is needed. The
[README's ReLU example](../README.md#web-inference) adds an actual operation and
explains the same lifecycle.

The returned IDs name objects retained by the host. Inputs carry names,
dtypes, concrete shapes, and bytes. `ReadOutput` retrieves a named output;
`host.close()` cleans up the session. In a long-running application, release
results as you consume them so they do not accumulate against the result budget.

## 3. Prepare a model

A normal package contains `graph.json` plus SafeTensors weights. The graph
states the operations and legal input shapes; the weights supply learned values.
The example exporters download public model sources and create that package:

```sh
make models_deps
make models_tinystories
make models_efficientdet
make validate_model_packages
```

You can run either model target independently. The downloads and converted
weights are not committed. See [models](models.md) for source selection and
[model format](model-format.md) for custom packages.

To open the browser demos, serve the repository over HTTP:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000/examples/tinystories.html` or
`http://localhost:8000/examples/efficientdet_lite0.html`. Select a prepared
package and an available backend. GPU inference uses the full browser profile;
WASM provides a CPU route. Stop the server when you finish.

## 4. Run TinyStories from native C

Build the two native executables:

```sh
make build_native_profiles
./native/volvoxai run --help
```

Create six token IDs and their positions. These values are an example input;
using a new prompt requires the matching tokenizer vocabulary and merge rules.

```sh
python3 - <<'PYINPUT'
from pathlib import Path
import struct
Path('build/quickstart').mkdir(parents=True, exist_ok=True)
Path('build/quickstart/tokens.i32').write_bytes(
    struct.pack('<6i', 7454, 2402, 257, 640, 11, 20037))
Path('build/quickstart/positions.i32').write_bytes(struct.pack('<6i', *range(6)))
PYINPUT
./native/volvoxai run models/tinystories_1m \
  --input 'tokens[1,6]=build/quickstart/tokens.i32' \
  --input 'positions[1,6]=build/quickstart/positions.i32' \
  --output logits=build/quickstart/logits.f32
```

The `[1,6]` shapes bind the package's dynamic sequence dimension. The output is
the complete F32 logits tensor, with one vocabulary-score row per input token.
To generate text, a caller selects a token and advances a decode context; raw
inference alone does not choose a sampling or stopping policy.

The [native guide](native-runtime.md) covers CPU threads, GPU selection, and
raw output handling. For image decoding and detection, use the
[task CLI example](../examples/native_task_cli/README.md).

## 5. Continue with your application

| Goal | Next guide |
| --- | --- |
| Load a package in Node or a browser | [Loading and inputs](browser-runtime.md#load-a-model) |
| Run on a browser GPU | [Backend selection](browser-runtime.md#choose-a-backend) |
| Serve concurrent requests or generate tokens | [Scheduling and decode](scheduling-and-dynamic-batching-design.md) |
| Build and train a small model | [Model construction and training](model_builder_training.md) |
| Make a quantized package | [Post-training quantization](quantization.md) |
| Learn what the model is doing | [Textbook](textbook/README.md) · [한국어](textbook/ko/README.md) |
| Validate a code or backend change | [Testing](testing.md) |

For individual request fields, use [API discovery](api-discovery.md). The
[architecture](../ARCHITECTURE.md) explains how the same model lifecycle runs
inside native and browser hosts.
