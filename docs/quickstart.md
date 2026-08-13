# Quickstart

VolvoxAI ships three JavaScript profiles:

- volvoxai.js: multi-backend inference.
- volvoxai.full.js: multi-backend inference plus Trainer and authoring tools.
- volvoxai.wasm.js: strict browser-only WASM inference and training.

All profiles use the same Runtime → Model → CompiledModel →
ExecutionContext → ExecutionResult inference lifecycle.

## Build

The reproducible browser build is:

~~~bash
make build_web
~~~

It creates exactly:

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

For local TypeScript and JavaScript work:

~~~bash
npm install
npm run typecheck
npm run build:all
~~~

The local JavaScript build does not compile C to WASM. Run make build_wasm when
the sidecars are required.

## Prepare a model

A package contains graph.json and one or more safetensors files:

~~~text
models/my-model/
  graph.json
  model.safetensors
~~~

graph.json must declare:

~~~json
{
  "format": "volvox-graph/v1",
  "dimensions": {}
}
~~~

The format discriminator is exact and case-sensitive. Dynamic shape is the
default `volvox-graph/v1` contract; symbols in input or node-output shape specs
require finite entries in `dimensions`.

## Run inference

~~~javascript
import {
  Model,
  VolvoxAI,
} from 'volvoxai';

const runtime = await VolvoxAI.createRuntime({
  backends: ['webgpu', 'wasm', 'cpu-js'],
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

Every public input is an explicit `{ data, shape }` view, including inputs to
constant-only packages. One compiled snapshot can accept different legal
concrete shapes in independent contexts without mutating its logical graph.

Use mode: require and operatorFallback: forbid when the request must compile
entirely for one provider:

~~~javascript
const compiled = await runtime.compile(snapshot, {
  backend: {
    mode: 'require',
    backend: 'webgpu',
    operatorFallback: 'forbid',
  },
});
~~~

Every result contains all declared graph outputs by name. read() returns a
fresh typed array on every call. Results remain valid across later executions
and context closure until result.close().

## Train from the full profile

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
  weights: [{ name: 'weight', dtype: 'float32', shape: [4, 2] }],
  nodes: [{
    id: 'projection',
    opType: 'MatMul',
    inputs: { input: 'x', weight: 'weight' },
    outputs: {
      out: { tensor: 'logits', dtype: 'float32', shape: ['B', 2] },
    },
    params: {},
  }],
  outputs: ['logits'],
});
const source = Model.capture({
  graph: builder.snapshot(),
  weights: {
    weight: {
      name: 'weight', dtype: 'float32', shape: [4, 2],
      data: Float32Array.from({ length: 8 }, (_, i) => (i - 4) / 16),
    },
  },
});
const trainer = await Trainer.create(source, { backend: 'cpu-js' });

const step = await trainer.trainStep({
  inputs: {
    x: { data: new Float32Array([1, 2, 3, 4]), shape: [1, 4] },
  },
  logitsTensor: 'logits',
  targets: new Int32Array([1]),
  trainableTensors: ['weight'],
  updateMode: 'adamw',
  optimizer: { learningRate: 1e-3 },
});
const successor = await trainer.commit();

await trainer.close();

const runtime = await VolvoxAI.createRuntime({ backends: ['cpu-js'] });
const compiled = await runtime.compile(successor, {
  backend: {
    mode: 'require',
    backend: 'cpu-js',
    operatorFallback: 'forbid',
  },
});
~~~

`trainStep()` changes only the Trainer's private working revision. `commit()`
returns the immutable successor snapshot; compile that returned snapshot for
inference. The source and previously compiled snapshots keep their original
weights. Every input carries its concrete shape, and one Trainer may accept any
shape inside the declared bounded domain. Call `rollback()` to discard private
updates and restore the last committed baseline.

Trainer also supports webgpu and wasm. WASM training is strict: unsupported
operators or layouts are rejected before weights change.

## Native build

~~~bash
make build_native
./native/volvoxai --help
./native/volvoxai-full --help
~~~

The fixed executables are model-agnostic. Both provide run; the full profile
also provides train.

~~~bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32
~~~

Raw input and output filenames use their declared storage suffix: .f32, .i32,
.i8, or .u8. The output file contains the complete declared tensor;
applications select task-specific rows or slices.

Model-specific image decoding, tokenization, generation, and postprocessing are
kept in examples:

~~~bash
make -C examples native_task_cli

examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=photo.png \
  --boxes boxes --scores scores --max-det 20
~~~

## Example models

~~~bash
make models_deps
make models_efficientdet
make models_tinystories
make validate_model_packages
~~~

Use ONLY=int8, ONLY=float16, or ONLY=float32 with
make models_efficientdet to export one precision.

Continue with:

- [Browser and Node runtime](browser-runtime.md)
- [Model construction and training](model_builder_training.md)
- [Native runtime](native-runtime.md)
- [Models and exporters](models.md)
