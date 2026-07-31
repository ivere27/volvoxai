# Model construction and training

The full JavaScript profile adds training-aware Graph and ModelBuilder exports
plus a retained Trainer. Models can be built from initialized tensors without
an exporter or pre-existing safetensors file.

Training follows this ownership chain:

~~~text
Runtime
  Model
    published immutable weight revision

Trainer
  retained Model
  private mutable working revision
  gradients, optimizer slots, and accumulation state
~~~

A successful optimizer update remains in the Trainer's private working
revision. `commit()` publishes that revision as a new Model weight revision
atomically. Existing CompiledModel and ExecutionContext objects stay pinned to
the revision they compiled. No training step publishes implicitly.

## Build a graph

~~~javascript
import {
  ModelBuilder,
  VolvoxAI,
} from 'volvoxai/full';

const builder = new ModelBuilder();
const x = builder.input('x', [1, 4]);
const weight = builder.weight('projection', [4, 8], 'float32', {
  initializer: { type: 'xavierUniform', seed: 17 },
});

const hidden = builder.addOp(
  'MatMul',
  { input: x, weight },
  { out: { name: 'hidden', shape: [1, 8] } },
  {},
  { id: 'projection', wLayout: 'din' },
).out;
const logits = builder.addOp(
  'GELU',
  { input: hidden },
  { out: { name: 'logits', shape: [1, 8] } },
  {},
  { id: 'activation' },
).out;

builder.outputs(logits);
const graph = builder.build();

const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
const model = runtime.createModel(graph);
~~~

The full profile exports training-aware `ModelBuilder` and `Graph` classes. The
inference profile contains only the model-agnostic graph and builder; it does not include
initializers, optimizer state, Dropout authoring, or training helpers.

Built-in F32 initializers are deterministic for a given seed:

- zeros
- ones
- normal
- xavierUniform
- xavierNormal

normal defaults to mean 0 and standard deviation 0.02. Xavier initializers
derive fan-in and fan-out from the shape and accept an optional gain.

Every topology operation is transactional. An invalid add, insert, patch,
replace, remove, or rename leaves the graph unchanged. Group several edits with
builder.topologyTransaction():

~~~javascript
builder.topologyTransaction(current => {
  current.patchNode('projection', {
    params: { weight_layout: 'IN_OUT' },
  });
  current.replaceNode('activation', replacementNode);
});
~~~

Removing a value with consumers requires an explicit rewire map or cascade.
Published adapter versions refer to a topology and prevent structural edits
until those versions are removed.

## Create a Trainer

~~~javascript
const trainer = await VolvoxAI.createTrainer(model, {
  backend: 'cpu',
});

const step = await trainer.trainStep({
  inputs: {
    x: new Float32Array([1, 2, 3, 4]),
  },
  logitsTensor: 'logits',
  targets: new Int32Array([3]),
  trainableTensors: ['projection'],
  updateMode: 'adamw',
  optimizer: {
    learningRate: 1e-3,
    weightDecay: 1e-2,
    maxGradNorm: 1,
  },
});
await trainer.commit();
~~~

`trainStep()` mutates only the Trainer's private state. Call `commit()` before
compiling inference against the update. Call `rollback()` instead to discard
uncommitted work and restore the last committed baseline.

Trainer backends are cpu, webgpu, and wasm:

~~~javascript
const gpuTrainer = await VolvoxAI.createTrainer(model, {
  backend: 'webgpu',
  device,
});

const wasmTrainer = await VolvoxAI.createTrainer(model, {
  backend: 'wasm',
  wasmUrl: new URL('./volvoxai.full.wasm', import.meta.url),
});
~~~

If a WebGPU device is omitted, Trainer requests one. If supplied, the caller
retains ownership of it. WASM training preflights its strict portable subset
and never falls back to CPU.

Calls on one Trainer are FIFO-serialized. Starting close rejects new work,
drains accepted steps, releases optimizer/backend resources, and releases its
Model retention. close() and dispose() are idempotent.

~~~javascript
await trainer.close();
await model.close();
await runtime.close();
~~~

Model close waits for its Trainer. Close the Trainer explicitly so lifecycle
errors are reported at the point the application expects.

## Publish and compile revisions

When `step.updatedTensorNames` is non-empty, the Trainer has updated its private
working weights. The step result contains stable names and copied gradients,
never mutable internal tensor handles. Publish explicitly, then compile against
the new Model revision:

~~~javascript
const revision = await trainer.commit();
const compiled = await model.compile({
  backend: {
    mode: 'require',
    backend: 'cpu',
    operatorFallback: 'forbid',
  },
});
const context = await compiled.createContext();
const result = await context.execute(inputs);
~~~

The model definition identity remains the same, while weightRevision and
weightRevisionId change. `commit()` rejects an incomplete accumulation window.
`rollback()` restores the private working state to the last successfully
committed baseline and publishes nothing.

## Losses

The single-loss shorthand accepts logitsTensor, targets, ignoreIndex, lossMask,
and lastToken:

~~~javascript
await trainer.trainStep({
  inputs,
  logitsTensor: 'logits',
  targets: targetIds,
  ignoreIndex: -1,
  lossMask,
  trainableTensors,
  updateMode: 'adamw',
  optimizer: { learningRate: 1e-4 },
});
~~~

Use losses for multiple weighted cross-entropy objectives. Each item has a
unique name and its own logits, targets, mask, weight, and denominator:

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

Without normalizer, each loss is divided by its own active example count and
then multiplied by weight. With accumulation and more than one loss, every
loss must provide the full-window normalizer. Repeated logits tensors are
allowed; their seeded gradients add before backward.

The result reports total loss, per-loss metrics, `updatedTensorNames`, accumulation
state, and gradient clipping evidence. maxGradNorm clips one Euclidean norm
over the complete trainable set. Zero disables clipping.

## Gradient accumulation

gradientAccumulationSteps defaults to one. Before a window is complete,
trainStep returns `accumulating: true` and an empty `updatedTensorNames` array. The optimizer
state, graph training step, and private working revision advance only when the
window is applied. The Model revision does not advance until `commit()`.

~~~javascript
const state = trainer.getGradientAccumulationState();
// { pending, microbatches, accumulationSteps, examples }

await trainer.resetGradientAccumulation();
~~~

flushGradientAccumulation applies a partial window.
resetGradientAccumulation discards the pending window before processing the
current microbatch.

Changing backend, topology, optimizer/loss signature, or trainable tensor set
while gradients are pending is rejected. Reset the window first.

## Checkpoints

Trainer checkpoints preserve graph topology, weights, optimizer moments,
per-parameter steps, the training step, and optional application/tokenizer
metadata from the private working revision:

~~~javascript
import {
  importModelCheckpoint,
} from 'volvoxai/full';

const checkpoint = await trainer.exportCheckpoint({
  tokenizerMetadata,
  metadata: { epoch: 3 },
});

const restored = importModelCheckpoint(checkpoint);
const restoredModel = runtime.createModel(restored.graph);
const restoredTrainer = await VolvoxAI.createTrainer(
  restoredModel,
  { backend: 'cpu', checkpoint },
);
~~~

The checkpoint is structured-cloneable. Weight and optimizer payloads are
safetensors ArrayBuffers. A checkpoint cannot be exported while unapplied
gradient accumulation is pending; flush or reset it first. For low-level
authoring, `exportModelCheckpoint()` and `importModelCheckpoint()`
operate on explicit inputs. Post-step export uses `Trainer.exportCheckpoint()`
so it captures the Trainer's private revision.

## Dropout

builder.dropout() creates inverted train-only Dropout:

~~~javascript
const regularized = builder.dropout(hidden, {
  probability: 0.1,
  seed: 42,
  name: 'encoder.dropout',
});
~~~

Its mask is deterministic from the node seed and training counter. Backward
reconstructs the same mask. Inference treats the node as an exact identity and
does not allocate a mask or load a backward shader.

SDPA and CrossSDPA attention-probability dropout is applied after softmax
during training and omitted during inference. JavaScript CPU, WebGPU, native
CPU, Vulkan, OpenGL compute, and Metal regenerate the same deterministic rule
for backward.

## GroupNorm

builder.groupNorm() constructs NHWC GroupNorm with caller-supplied F32 affine
tensors shaped [C]. The channel count must be divisible by numGroups:

~~~javascript
const scale = builder.weight('norm.scale', [64], 'float32', {
  initializer: { type: 'ones' },
});
const bias = builder.weight('norm.bias', [64], 'float32', {
  initializer: { type: 'zeros' },
});
const normalized = builder.groupNorm(features, scale, bias, {
  numGroups: 8,
  epsilon: 1e-5,
  name: 'vision.norm',
});
~~~

GroupNorm and affine gradients are supported by JavaScript CPU, WebGPU, native
CPU, Vulkan, OpenGL compute, and Metal training paths.

## Mixture of Experts

MoERouter returns top-k expert indices and weights. MoELinear executes the
selected expert matrices and combines their results:

~~~javascript
const routerWeight = builder.weight(
  'router', [8, 4], 'float32', new Float32Array(32));
const experts = builder.weight(
  'experts', [4, 8, 16], 'float32', new Float32Array(512));
const routes = builder.moeRouter(hidden, routerWeight, {
  topK: 2,
  normalize: true,
  temperature: 1,
});
const routed = builder.moeLinear(hidden, experts, routes).out;
~~~

With normalize: true, selected weights are renormalized among the top-k. With
false, they remain probabilities from the softmax over all experts, so router
gradients include selected and unselected logits.

maskedMean(input, normalizedWeights) pools [B,T,D] to [B,D]. The caller supplies
already normalized F32 weights shaped [B,T].

Portable GPU MoE routing limits topK to 8. Native CPU does not inherit that
shader limit.

## Trainable LoRA

`ModelBuilder.loraLinear()` adds explicit F32 factor weights and returns
their exact names:

~~~javascript
const lora = builder.loraLinear(hidden, baseWeight, {
  rank: 8,
  alpha: 16,
  bias: baseBias,
  name: 'decoder.projection',
});
builder.outputs(lora.out);

await trainer.trainStep({
  inputs,
  logitsTensor: 'logits',
  targets,
  trainableTensors: lora.trainableTensors,
  updateMode: 'adamw',
  optimizer: { learningRate: 1e-4 },
});
await trainer.commit();
~~~

The helper creates A=[d_in,rank] and B=[rank,d_out], a frozen scalar scale, and
explicit base, low-rank, scale, and Add nodes. A uses Xavier-uniform
initialization and B starts at zero unless overridden. Only A and B appear in
trainableTensors; the base weight, bias, and scale remain frozen. Use
layout: 'dout' for a base matrix stored [d_out,d_in].

Immutable staged adapters are inference routing snapshots, not differentiable
parameters. Train explicit A/B graph tensors, checkpoint them with optimizer
state, and create a separate staged snapshot for deployment if needed.

## Adapter routing

An execution context owns its adapter selector:

~~~javascript
const route = builder.adapterRouting(
  builder.adapterRoute('tenant-a', 3, 0.75),
  builder.adapterRoute('tenant-b', 1, 1),
);

const compiled = await model.compile();
const context = await compiled.createContext();
const result = await context.execute(inputs, route);
~~~

A routed bottleneck adapter uses ordinary graph weights and can therefore be
trained:

~~~javascript
const down = builder.weight(
  'adapter.down', [experts, dModel, bottleneck], 'float32',
  { initializer: { type: 'xavierUniform', seed: 20 } },
);
const up = builder.weight(
  'adapter.up', [experts, bottleneck, dModel], 'float32',
  { initializer: { type: 'zeros' } },
);
const adapted = builder.routedBottleneckAdapter(
  hidden,
  down,
  up,
  routes,
  { dropout: 0.1, seed: 21, name: 'decoder.adapter' },
);
~~~

The helper applies routed down projection, GELU, routed up projection, optional
Dropout, and a residual Add.

## Encoder-decoder construction

Model-family builders stay under examples/. The seq2seq example composes the
generic builder into token and position embeddings, full encoder
self-attention, causal decoder self-attention, cross-attention, pre-norm
feed-forward blocks, and a vocabulary projection.

~~~javascript
import {
  buildEncoderDecoderTransformer,
} from '../examples/seq2seq_training/Seq2SeqBuilder.js';

const seq2seq = buildEncoderDecoderTransformer(builder, {
  batchSize: 2,
  sourceLength: 32,
  targetLength: 24,
  vocabSize: 8000,
  dModel: 256,
  numHeads: 8,
  dFF: 1024,
  encoderLayers: 4,
  decoderLayers: 4,
  padTokenId: 0,
  bosTokenId: 1,
  seed: 42,
});

const batch = seq2seq.teacherForcing(sourceTokenIds, targetTokenIds);
await trainer.trainStep({
  ...batch,
  updateMode: 'adamw',
  optimizer: { learningRate: 3e-4 },
});
await trainer.commit();
~~~

batchSize may be any positive model shape. Token, position, and attention-mask
values are I32. Teacher forcing shifts each target row right, inserts bosTokenId,
constructs position IDs and keep masks, and excludes padding from the loss.

An F32 sourceFeatures tensor shaped [B,F,D] may be prepended to source token
embeddings. Encoder memory and its mask then have length F+S. Feature-producing
CNN/projection weights can train end-to-end when listed in the trainable set.

SDPA and CrossSDPA masks may have shape [K], [B,K], [Q,K], or [B,Q,K];
nonzero means visible. Fully masked rows produce zeros rather than NaNs.

## Training limits

- Trainable tensors must be F32.
- Every operation between a loss and a trainable tensor needs a backward
  implementation; unsupported paths fail the step explicitly.
- SDPA and CrossSDPA support rank-2 [sequence,width] and rank-3
  [batch,sequence,width] inputs.
- Portable WebGPU and native GPU attention shaders limit head width to 64.
- Native and WebGPU BatchNorm training uses stored running statistics as an
  affine operation and does not update those statistics.
- Native GPU training preflights a complete backward plan. An unsupported GPU
  path selects the complete native CPU path before backward begins.
- The full inference entry remains free of gradient allocation until a Trainer
  is created; the inference-only entry contains no training dependency.

See [the operation matrix](operation_list.md) for exact backend coverage.
