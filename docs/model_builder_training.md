# Logical model construction and training

Dynamic training v1 accepts one public model type: an immutable,
fixed-rank, bounded-shape `Model`. Concrete kernel graphs and
training builders are implementation details and are not package exports.
There is no compatibility path for a pre-v1 mutable graph checkpoint.

Training has this ownership model:

~~~text
Model (immutable topology, bounds, and fixed weights)
  └─ Trainer (private weights, optimizer slots, accumulation, shape cache)
       └─ commit() → new immutable Model
~~~

`trainStep()` never mutates its source snapshot. `commit()` captures the
Trainer's current weights as a successor and updates `trainer.snapshot` to that
successor. Previously compiled snapshots remain unchanged.

## Author a bounded logical snapshot

`ModelBuilder` edits validated logical graph documents. Weight bytes are
supplied only when the logical graph is captured as a snapshot.

~~~javascript
import {
  ModelBuilder,
  Model,
  Trainer,
  VolvoxAI,
} from 'volvoxai/full';

const builder = new ModelBuilder({
  dimensions: {
    B: { min: 1, max: 16, multiple_of: 1 },
  },
  inputs: {
    x: { dtype: 'float32', shape: ['B', 4] },
  },
  weights: [
    { name: 'projection.weight', dtype: 'float32', shape: [4, 8] },
  ],
  nodes: [{
    id: 'projection',
    opType: 'MatMul',
    inputs: { input: 'x', weight: 'projection.weight' },
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
    'projection.weight': {
      name: 'projection.weight',
      dtype: 'float32',
      shape: [4, 8],
      data: Float32Array.from(
        { length: 32 },
        (_, index) => (index - 16) / 64,
      ),
    },
  },
});
~~~

The shape domain is part of the graph fingerprint. Every symbol has finite
`min`, `max`, and `multiple_of` constraints, and ranks never change. All graph
states published by the builder are validated and immutable. Group related
topology edits with `builder.topologyTransaction(edit => { ... })`; a rejected
transaction leaves the prior snapshot intact.

For exported models, load `volvox-graph/v1` and safetensors bytes with
`ModelLoader`, then pass the loaded package to
`Model.capture()`.

## Create and use a Trainer

~~~javascript
const trainer = await Trainer.create(source, { backend: 'cpu' });

const first = await trainer.trainStep({
  inputs: {
    x: {
      data: new Float32Array([1, 2, 3, 4]),
      shape: [1, 4],
    },
  },
  logitsTensor: 'logits',
  targets: Int32Array.of(3),
  trainableTensors: ['projection.weight'],
  updateMode: 'adamw',
  optimizer: {
    learningRate: 1e-3,
    weightDecay: 1e-2,
    maxGradNorm: 1,
  },
});

// The same Trainer binds another concrete point in the declared domain.
const second = await trainer.trainStep({
  inputs: {
    x: {
      data: new Float32Array([
        1, 2, 3, 4,
        4, 3, 2, 1,
        0, 1, 0, 1,
      ]),
      shape: [3, 4],
    },
  },
  logitsTensor: 'logits',
  targets: Int32Array.of(3, 2, 1),
  trainableTensors: ['projection.weight'],
  updateMode: 'adamw',
  optimizer: { learningRate: 1e-3, maxGradNorm: 1 },
});

const successor = await trainer.commit();
await trainer.close();
~~~

Every input is a `{ data, shape }` view, even for a fully static graph. Storage
length never supplies an implicit shape. A step fails before mutation if an
input is missing, has the wrong dtype/rank/byte length, violates a symbol
constraint, or binds one symbol inconsistently across inputs.

The result contains copied gradients, stable updated parameter names, the
canonical shape and tactic signatures, and concrete activation/gradient shape
maps. It never exposes mutable internal tensors.

`VolvoxAI.createTrainer(source, options)` is an equivalent namespace form.
Calls on one Trainer are FIFO-serialized. `close()` rejects new work, drains
accepted work, and releases backend resources; `close()` and `dispose()` are
idempotent.

## Shape specialization and capacity

One Trainer retains a bounded LRU of immutable plan/tactic metadata and one
growable activation-capacity pool. Returning to a prior shape can reuse cached
plans and backend pipelines. WASM keeps one module instance and scratch arena;
WebGPU keeps one device/executor cache. Shape changes do not recreate a Trainer.

~~~javascript
const state = trainer.inspectShapeState();
// state.planCacheEntries, state.planCacheMetadataBytes,
// state.activationCapacityBytes, state.activationCapacityHighWaterBytes, ...
~~~

Capacity and metadata limits are explicit Trainer options:

~~~javascript
const boundedTrainer = await Trainer.create(source, {
  backend: 'webgpu',
  planCacheEntries: 8,
  planCacheMetadataBytes: 256 * 1024,
  maxCapacityBytes: 64 * 1024 * 1024,
  capacityGrowthFactor: 2,
});
~~~

A pending gradient-accumulation window is shape-stable. Flush or reset it
before changing the concrete shape.

## Backend selection

The full profile supports explicit `cpu`, `webgpu`, and `wasm` training:

~~~javascript
const gpuTrainer = await Trainer.create(source, {
  backend: 'webgpu',
  device, // omit to request a device owned by the Trainer
});

const wasmTrainer = await Trainer.create(source, {
  backend: 'wasm',
  wasmUrl: new URL('./volvoxai.full.wasm', import.meta.url),
});
~~~

Strict WASM training is also available from the WASM-only JavaScript profile.
It contains no CPU, WebGPU, or training fallback implementation. WASM and
WebGPU preflight their supported graph/layout subset before optimizer mutation.
A step never switches backend after execution starts.

## Publish and compile a successor

`commit()` requires at least one completed optimizer update and no incomplete
accumulation window. It returns a new snapshot; it does not publish into a
mutable Model object.

~~~javascript
const successor = await trainer.commit();
const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
const compiled = await runtime.compile(successor, {
  backend: {
    mode: 'require',
    backend: 'cpu',
    operatorFallback: 'forbid',
  },
});

const context = await compiled.createContext();
const result = await context.execute({
  x: { data: new Float32Array([1, 2, 3, 4]), shape: [1, 4] },
});
~~~

The successor retains definition identity when topology and shape bounds are
unchanged, while its weight revision changes. `rollback()` discards private
work and returns the Trainer to its last committed baseline.

## Losses

The single-loss shorthand accepts `logitsTensor`, `targets`, `ignoreIndex`,
`lossMask`, and `lastToken`. Use `losses` for several weighted cross-entropy
objectives:

~~~javascript
const step = await trainer.trainStep({
  inputs,
  losses: [
    {
      name: 'tokens',
      logitsTensor: 'token_logits',
      targets: tokenTargets,
      lossMask: tokenMask,
      weight: 1,
      normalizer: activeTokensAcrossWindow,
    },
    {
      name: 'router',
      logitsTensor: 'router_logits',
      targets: routeTargets,
      weight: 0.05,
      normalizer: routeRowsAcrossWindow,
    },
  ],
  trainableTensors,
  gradientAccumulationSteps: 4,
  updateMode: 'adamw',
  optimizer: { learningRate: 1e-4, maxGradNorm: 1 },
});
~~~

Without `normalizer`, each loss is divided by its active example count and then
multiplied by `weight`. With accumulation and multiple losses, every loss must
provide its full-window normalizer. Repeated logits tensors are allowed; their
seeded gradients add before backward. `maxGradNorm` clips one Euclidean norm
over the complete trainable set; zero disables clipping.

## Gradient accumulation

`gradientAccumulationSteps` defaults to one. Before a window completes, the
result reports `accumulating: true` and no updated parameter names. Apply a
short final window with `flushGradientAccumulation: true`, or discard pending
gradients with:

~~~javascript
const state = trainer.getGradientAccumulationState();
await trainer.resetGradientAccumulation();
~~~

Changing concrete shape, topology, optimizer/loss signature, or trainable set
while gradients are pending is rejected. Checkpoint export and commit also
require the window to be completed or reset.

## Checkpoints and exact resume

Dynamic checkpoints preserve the bounded logical document and fingerprint,
fixed parameter descriptors/bytes, optimizer descriptor and moments,
per-parameter steps, training step, quantization metadata, and optional
application/tokenizer metadata.

~~~javascript
import { importModelCheckpoint } from 'volvoxai/full';

const checkpoint = await trainer.exportCheckpoint({
  tokenizerMetadata,
  metadata: { epoch: 3 },
});

const restored = importModelCheckpoint(checkpoint);
const resumed = await Trainer.create(restored.snapshot, {
  backend: 'cpu',
  checkpoint,
});
~~~

`importModelCheckpoint()` returns `{ snapshot, trainingStep,
optimizerDescriptor, ... }`; it never returns a mutable graph. Passing the
checkpoint to `Trainer.create()` restores the exact private optimizer state.
The checkpoint and target snapshot must have the same logical fingerprint,
including all symbolic bounds. Legacy concrete checkpoint formats are rejected.

`exportModelCheckpoint(snapshot)` creates a checkpoint for an immutable source
without Trainer optimizer state. `Trainer.exportCheckpoint()` captures the
private working revision. Checkpoint payloads are structured-cloneable and use
safetensors `ArrayBuffer`s for parameters and optimizer slots.

## Dropout and trainable low-rank adapters

Author train-only Dropout as an ordinary logical node with a bounded symbolic
output and `{ p, seed }` parameters. Its mask is deterministic from the node
seed and training counter; inference treats it as identity. SDPA and CrossSDPA
may similarly declare attention-probability dropout for training.

LoRA is represented explicitly in the logical graph: fixed base projection,
F32 A and B weights, scale, low-rank MatMul nodes, and Add. Put only the A/B
weight names in `trainableTensors`. There is no public concrete builder helper
that hides those nodes. Immutable staged inference adapters are routing
snapshots, not differentiable training parameters.

## Training limits

- Trainable parameters must be F32 and have fixed storage shape.
- Public input ranks are fixed; only declared bounded extents are dynamic.
- Every operation between a loss and a trainable parameter needs a backward
  implementation; unsupported paths fail explicitly.
- SDPA and CrossSDPA support rank-2 `[sequence,width]` and rank-3
  `[batch,sequence,width]` inputs on documented training paths.
- Portable WebGPU and native GPU attention shaders limit head width to 64.
- Deep quantized operator backward, fake-quantization QAT, and FP16/AMP
  training are not supported.
- The inference entry contains no Trainer, optimizer, autograd, or compiled
  training code.

See [the operation matrix](operation_list.md) and
[training/PTQ runtime matrix](training-ptq-runtime-matrix.md) for backend
coverage.
