# Model construction and training

Training adjusts a model's weights so its outputs better match examples. In
VolvoxAI you load or construct a graph, choose which weights may change, and
send batches to a Trainer. The Trainer owns its private parameters, gradients,
optimizer state, and shape cache. You decide when to evaluate, save, or publish.

```text
Model: graph, bounded shapes, and a weight revision
  Trainer: private weights, gradients, optimizer, RNG
    train step -> private update
    commit -> successor revision for new inference compilations
    rollback -> last committed baseline
```

Existing compiled models and results keep their earlier revision. This lets
an application continue serving a model while preparing its next version.
Training requires the full profile; graph construction and SafeTensors storage
are available in both profiles.

## Build and train a small classifier

This complete example creates a linear classifier with four input features and
three classes. Its batch dimension `B` can range from 1 to 16. A step computes
logits, cross-entropy loss, gradients, and an AdamW update, then publishes the
new weights. Save it as an `.mjs` file in the repository or an installed project.

```javascript
import {
  FullEngineHost, VxInferenceServiceClient, VxPlanningServiceClient,
  VxTrainingServiceClient, pb,
} from 'volvoxai/full';

const host = new FullEngineHost();
const inference = new VxInferenceServiceClient(host);
const planning = new VxPlanningServiceClient(host);
const training = new VxTrainingServiceClient(host);
const f32 = (name, values, shape) => new pb.Tensor({
  name, dtype: pb.DataType.DATA_TYPE_F32, shape,
  inline: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
});
try {
  const graphDocument = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 16 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 4] } },
    nodes: [{
      id: 'projection', opType: 'MatMul',
      inputs: { input: 'x', weight: 'projection.weight' },
      outputs: { out: { tensor: 'logits', dtype: 'float32', shape: ['B', 3] } },
      params: { weight_layout: 'din_dout' },
    }],
    outputs: ['logits'],
  }));
  const initial = Float32Array.from({ length: 12 }, (_, i) => (i - 6) / 32);
  const weights = await planning.writeSafetensors(new pb.WriteSafetensorsRequest({
    edits: [new pb.SafetensorsEdit({
      setTensor: f32('projection.weight', initial, [4n, 3n]),
    })],
  }));
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    package: new pb.ModelPackage({ graphDocument, weightShards: [weights.data] }),
  }));
  const trainer = await training.createTrainer(new pb.CreateTrainerRequest({
    modelId: model.modelId, backend: 'wasm', rngSeed: 123n,
  }));
  const inputs = [f32('x', Float32Array.of(1, 2, 3, 4), [1n, 4n])];
  const step = await training.trainStep(new pb.TrainStepRequest({
    trainerId: trainer.trainerId, inputs,
    losses: [new pb.CrossEntropyLoss({
      name: 'classification', logitsName: 'logits', targets: [2],
    })],
    trainableNames: ['projection.weight'],
    optimizer: new pb.TrainerOptimizerOptions({
      kind: pb.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW,
      learningRate: 0.001, weightDecay: 0.01, maxGradientNorm: 1,
    }),
  }));
  console.log({ loss: step.loss, updateApplied: step.updateApplied });
  const revision = await training.commitTrainer(new pb.TrainerRef(trainer));
  console.log({ weightRevision: revision.weightRevision });

  const compiled = await inference.compileModel(new pb.CompileModelRequest({
    modelId: model.modelId,
    policy: new pb.BackendPolicy({ backends: ['wasm'] }),
  }));
  const result = await inference.run(new pb.RunRequest({
    compiledModelId: compiled.compiledModelId, inputs,
  }));
  const output = await inference.readOutput(new pb.ReadOutputRequest({
    resultId: result.resultId, name: 'logits',
  }));
  console.log(Array.from(new Float32Array(output.tensor.inline.slice().buffer)));
} finally {
  await host.close();
}
```

For another legal batch size, supply all named inputs with the new concrete
shape and matching byte length, plus one target per selected logits row.
Trainable parameters retain fixed storage shapes. A model's dimension bounds
are part of its definition; the runtime does not infer shapes from array size.

## Construct and edit a graph

The example uses the inspectable JSON [model format](model-format.md). For
programmatic construction, `GraphDefinition` describes the same inputs, nodes,
outputs, and bounded dimensions with typed messages. Pass it in
`GraphPlanningSource.definition` to `CreateGraphPlan`; external weight
specifications go in `GraphPlanningSource.weights` as `PlanningWeight` records.

A graph plan is immutable. `EditGraphPlan` applies an ordered group of edits to
a private draft and validates it once. On success it returns a new plan; a
rejected edit leaves the original intact. `ExportGraphPlan` returns the logical
graph bytes, which can be loaded together with weight shards as `ModelPackage`.
`SerializeGraph` accepts a definition when only graph bytes are needed.

`WriteSafetensors` creates or edits weight storage, as above; `ReadSafetensors`
returns typed tensors. Full's `InitializeTensor` supplies seeded zeros, ones,
normal, Xavier uniform, and Xavier normal initializers. A graph describes the
computation; these separate tensor operations supply its initial weights.

## Choose the training backend

Use `backend: 'wasm'` for CPU training in a browser or Node. Use
`backend: 'webgpu'` on a `FullEngineHost` for a supported browser GPU graph.
Native full supports its admitted native training backends. A requested backend
is exact: a step never silently switches device halfway through training.

WebGPU steps may be asynchronous. Keep the returned microbatch ID and poll
without submitting another step:

```javascript
let step = await training.trainStep(request);
while (step.state === pb.ResultState.RESULT_STATE_PENDING) {
  await new Promise(resolve => setTimeout(resolve, 1));
  step = await training.getTrainStep(new pb.TrainStepRef({
    trainerId: trainer.trainerId, microbatchId: step.microbatchId,
  }));
}
console.log(step.loss, step.optimizerStep);
```

This fragment assumes the clients, Trainer, and `request` were prepared as in
the complete example. Await completion before commit, rollback, or export;
those operations return BUSY during a pending step. Failed device work can
require restoring the committed baseline or recreating the Trainer. See the
[training matrix](training-ptq-runtime-matrix.md) for the qualified behavior.

## Losses and optimizer settings

Each `CrossEntropyLoss` names a logits tensor and target class IDs. Use
`ignoreIndex` for padding labels, `weight` to balance objectives, and several
loss records for tasks such as token prediction plus routing. `rowIndex` can
select a row where the graph supports it. Otherwise the loss consumes the full
logits tensor.

Without an explicit normalizer, a one-microbatch update uses its active-label
count. Accumulation windows larger than one require an explicit positive
normalizer for every loss. Choose the denominator for the complete window so
that changing the microbatch split does not change the intended objective.

SGD applies a gradient step; AdamW also retains first and second moments.
`maxGradientNorm` clips one global Euclidean norm across the trainable set.
Zero disables clipping. Omitted optimizer fields keep the Trainer's last
successful settings. A new Trainer starts with AdamW, learning rate 0.001,
betas 0.9/0.999, epsilon 1e-8, no weight decay, and no clipping. Supply settings
explicitly when comparing runs.

## Gradient accumulation and shapes

Set `TrainStepRequest.accumulationSteps` to accumulate several microbatches
before updating weights. An unfinished window has `updateApplied: false` and
an increasing `accumulatedMicrobatches` count. Use `flushAccumulation` on a final
short window, or discard its pending gradients with `ResetTrainerAccumulation`.
`GetTrainerState` reports the current window and optimizer state.

Keep the concrete shape, loss configuration, optimizer configuration, and
trainable set stable during an accumulation window. Complete or reset the
window before changing them. Commit and checkpoint export require no pending
GPU step or unfinished accumulation.

A Trainer retains an LRU cache of shape plans and a reusable activation arena.
`CreateTrainerRequest.shapeOptions` controls:

| Option | Default | Purpose |
| --- | --- | --- |
| `planCacheEntries` | 8 | Number of retained layout plans |
| `planCacheMetadataBytes` | 1 MiB | Metadata budget; oversized plans execute uncached |
| `maxActivationCapacityBytes` | 512 MiB | Limit on the C activation arena |
| `capacityGrowthFactor` | 2 | Growth factor when a larger binding is admitted |

Inspect `GetTrainerState.shape` for actual plan-cache usage and activation
capacity/high-water. Parameter, gradient, optimizer, and backend storage are
separate charges; arena capacity alone is not total training memory.

## Publish, evaluate, or roll back

`TrainStep` only changes private Trainer state. `CommitTrainer` requires a
completed optimizer update and publishes a successor revision on the retained
Model. Compile that model again for validation or serving with the new weights.
Earlier compiled models retain their original revisions.

`RollbackTrainer` restores the last committed baseline. Applications own the
dataset loop, learning-rate schedule, validation metric, and choice of best
checkpoint. The [textbook's training chapter](textbook/05-optimizer-and-training-loop.md)
explains these choices and their effect on learning.

## Save a checkpoint and resume

Weight export is enough for inference. To resume training, save the optimizer
and RNG state as well. This fragment uses the clients and Trainer from a live
session like the complete example:

```javascript
const saved = await training.exportTrainerCheckpoint(
  new pb.ExportTrainerCheckpointRequest({
    trainerId: trainer.trainerId,
    metadata: new TextEncoder().encode(JSON.stringify({ epoch: 3 })),
  }));
const checkpointBytes = saved.checkpoint.toBinary();
// Save checkpointBytes in application storage. Later, decode the saved bytes:
const checkpoint = pb.TrainerCheckpoint.fromBinary(checkpointBytes);
const restoredModel = await inference.loadModel(new pb.LoadModelRequest({
  runtimeId: runtime.runtimeId,
  package: new pb.ModelPackage({
    graphDocument: checkpoint.graph, weightShards: checkpoint.weightShards,
  }),
}));
const resumed = await training.createTrainer(new pb.CreateTrainerRequest({
  modelId: restoredModel.modelId, backend: 'wasm', checkpoint,
}));
```

A checkpoint includes the exact logical graph, private weight shards, optimizer
SafeTensors and settings, optimizer step, RNG seed, and opaque application
metadata. Restoration validates compatibility before publishing a Trainer ID.
The restored private state becomes the rollback baseline until the next commit.
A new process or browser session first creates its own host and Runtime; old
IDs are never part of the saved state.

`ExportTrainerWeights` with empty output paths returns SafeTensors shard bytes
for inference deployment. Native callers may request filesystem destinations.
Browser callers save the returned bytes themselves. Always close the host, or
release each Trainer/Model when its session ends.

## Dropout, LoRA, and quantized adapters

Dropout is active during training and acts as identity during inference. Its
seed and training RNG state make checkpoint continuation meaningful. Attention
operators can also declare training-time probability dropout.

LoRA adds small trainable A/B matrices beside a frozen projection. Include only
those weight names in `trainableNames`. `BuildLoraLinear` and
`BuildRoutedAdapter` can author the explicit graph branches and initialized
weights; the same training step machinery updates them.

For quantized LoRA, `DequantizeWeight` initializes F32 master parameters.
Train the masters and call `ExportQuantizedTrainerWeights` to create an I8
inference package. Export preserves the masters and earlier inference revisions.
For whole-model post-training quantization, continue with [PTQ](quantization.md).

## Limits and validation

- Trainable storage is fixed-shape F32; input extents may vary within declared
  bounds, but rank remains fixed.
- Every operation on a trainable loss path needs a complete backward
  implementation for its selected backend.
- Quantized activation backward, fake-QAT, and mixed-precision optimizer storage
  are not supported by these training paths.
- A passing step is not evidence that a model has learned its task. Evaluate on
  held-out data and compare the metric you intend to improve.

See [training/PTQ coverage](training-ptq-runtime-matrix.md),
[testing](testing.md), and the [full API reference](generated/api-contract.full.md)
for further details.
