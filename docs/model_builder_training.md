# Model construction, routing, and training APIs

VolvoxAI models no longer need to begin as files. They can be created as an
empty graph, populated through the JavaScript builder or the `CreateModel` RPC,
edited, compiled, trained, checkpointed, and resumed using VolvoxAI APIs alone.
No PyTorch, TensorFlow, ONNX exporter, or pre-existing safetensors file is
required for this workflow. A topology edit invalidates an already compiled
backend; compile the graph again before its next execution.

## JavaScript model builder

```js
import { VolvoxAI } from "volvoxai/full";

const volvox = await VolvoxAI.init("cpu");
const model = volvox.createModel();
const x = model.input("x", [1, 4]);
const w = model.weight("w", [4, 8], "float32", new Float32Array(32));

const { out: hidden } = model.addOp(
  "MatMul",
  { input: x, weight: w },
  { out: { name: "hidden", shape: [1, 8] } },
  {},
  { id: "projection", wLayout: "din" },
);
const { out: prediction } = model.addOp(
  "GELU",
  { input: hidden },
  { out: { name: "prediction", shape: [1, 8] } },
  {},
  { id: "activation" },
);
model.outputs(prediction);

const graph = model.build();
const engine = await volvox.compile(graph);
```

The full module aliases `Graph` and `ModelBuilder` to `TrainingGraph` and
`TrainingModelBuilder`; their explicit names are exported as well, and
`VolvoxAI.createGraph()`/`createModel()` return those training-capable variants.
The inference entry retains the model-agnostic core graph and builder. It does
not interpret initializer descriptors or carry optimizer/training state, and it
does not expose training-only helpers such as `dropout()` or
`routedBottleneckAdapter()`.

Weights can be initialized without a pre-existing checkpoint. Built-in
initializers are deterministic for a given seed and include `zeros`, `ones`,
`normal`, `xavierUniform`, and `xavierNormal`:

```js
const projection = model.weight("projection", [64, 256], "float32", {
  initializer: { type: "xavierUniform", seed: 17 },
});
```

`normal` defaults to mean 0 and standard deviation 0.02. Xavier initializers
derive their fan-in and fan-out from the tensor shape and accept an optional
`gain`. The native `TensorInitializer` message exposes the same algorithms and
seed stream, so a client can send a shape plus initializer instead of weight
bytes to `CreateModel` or `AddModelTensor`. Initializers currently create F32
weights.

Model-family construction is intentionally outside the package entry. The
repository's encoder-decoder example composes the generic full-profile builder
into learned token and position embeddings, full encoder self-attention,
causal decoder self-attention, cross-attention, pre-norm feed-forward blocks,
and a vocabulary projection:

```js
import {
  buildEncoderDecoderTransformer,
  createTeacherForcingBatchForGraph,
} from "../examples/seq2seq_training/Seq2SeqBuilder.js";

const seq2seq = buildEncoderDecoderTransformer(model, {
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
const graph = model.build();

// Row-major Int32 values: [batchSize, sourceLength/targetLength].
const batch = seq2seq.teacherForcing(sourceTokenIds, targetTokenIds);
const step = await volvox.trainStep(graph, {
  ...batch,
  updateMode: "adamw",
  optimizer: { learningRate: 3e-4 },
});
```

`batchSize` is a model shape, not a hard-coded runtime limit: it may be any
positive value, and teacher forcing validates the complete rectangular batch.
Token IDs, position IDs, and attention masks are I32 throughout the JavaScript,
WASM-forward, and native interfaces. Teacher forcing shifts each target row
right, inserts `bosTokenId`, creates position IDs and source/decoder keep masks,
and returns the unshifted labels plus a loss mask. Padding does not contribute
to the loss, and cross-entropy is normalized over active labels only.

An encoder can also receive an existing F32 `[B,F,D]` tensor through
`sourceFeatures`. Those feature rows are prepended to the source-token
embeddings, so encoder memory and the cross-attention mask have length `F+S`:

```js
const imageFeatures = model.input("image_features", [2, 16, 256]);
const seq2seq = buildEncoderDecoderTransformer(model, {
  batchSize: 2,
  sourceLength: 32,
  targetLength: 24,
  vocabSize: 8000,
  dModel: 256,
  numHeads: 8,
  sourceFeatures: imageFeatures,
});
```

`teacherForcing()` creates a `[B,F+S]` source mask, marks all `F` feature rows
visible, and then applies token padding to the following `S` rows. Feature
producers can be arbitrary differentiable builder operations, such as a CNN
and projection. Their weights are updated with the Transformer when they are
listed in `additionalTrainableTensors`; this supports end-to-end multimodal
training without a separate framework.

Encoder self-attention sets `causal: false`, so every visible source position
can attend to every other visible source position. Decoder self-attention sets
`causal: true`, so query position `q` cannot see a key position greater than
`q`. `SDPA` and `CrossSDPA` accept an optional binary `mask` input shaped `[K]`,
`[B,K]`, `[Q,K]`, or `[B,Q,K]`; nonzero means visible. A row with no visible keys
produces zeros instead of NaNs. These full/causal and padded-mask semantics are
shared by the JavaScript CPU, WebGPU, native CPU, and supported native GPU
training paths. `trainStep` also accepts `ignoreIndex` and a binary `lossMask`.

Full JavaScript checkpoints preserve the blueprint, every weight, AdamW moments
and per-tensor steps, the graph training step, and optional tokenizer/application
metadata:

```js
const checkpoint = volvox.exportCheckpoint(graph, {
  tokenizerMetadata: tokenizerConfig,
  metadata: { epoch: 3 },
});
const { graph: restored } = volvox.importCheckpoint(checkpoint);
const resumedBatch = createTeacherForcingBatchForGraph(
  restored, sourceTokenIds, targetTokenIds,
);
```

The checkpoint is a self-contained structured-cloneable package object whose
weight and optimizer payloads are safetensors `ArrayBuffer`s. Encoder-decoder
application metadata is included unchanged; the example's
`createTeacherForcingBatchForGraph()` validates it before reconstructing teacher
forcing after a standalone import. The checkpoint also stores the canonical optimizer descriptor
(update mode and hyperparameters), AdamW first/second moments, per-parameter
steps, and the authoritative graph training step. Calling `trainStep()` on the
restored graph therefore resumes with the persisted optimizer configuration;
per-call options can still override it deliberately.

A checkpoint cannot be exported while gradient accumulation has unapplied
microbatches. Flush the pending window with a training step, or call
`resetGradientAccumulation(graph)`, before exporting so a checkpoint never
silently loses partial gradients.

Nodes can be inserted, patched, replaced, or removed by id. Each mutation is
transactional: an invalid edit leaves the graph unchanged.

```js
model.insertNode("activation", nodeSpec, { position: "before" });
model.patchNode("projection", { params: { transpose_weight: false } });
model.replaceNode("activation", replacementSpec);
model.removeNode("projection", { rewire: { hidden: "x" } });
```

Application-level builders that perform several structural edits can wrap them
in `model.topologyTransaction((builder) => { ... })`. If the callback throws or
returns a rejected promise, VolvoxAI restores the prior nodes, tensors, selected
outputs, object identities, and topology revision. Tensor-value updates and
adapter lifecycle changes are deliberately outside this topology transaction.

Removing a value that still has consumers requires an explicit `rewire` map or
`{ cascade: true }`. Adapter versions lock structural edits because their
targets refer to a particular topology; remove those versions before changing
the graph.

## Training-oriented builder primitives

`groupNorm()` constructs NHWC GroupNorm with caller-supplied F32 affine
parameters. The channel count must be divisible by `numGroups`, and both
affine tensors are shaped `[C]`:

```js
const scale = model.weight("vision.norm.scale", [64], "float32", {
  initializer: { type: "ones" },
});
const bias = model.weight("vision.norm.bias", [64], "float32", {
  initializer: { type: "zeros" },
});
const normalized = model.groupNorm(features, scale, bias, {
  numGroups: 8,
  epsilon: 1e-5,
  name: "vision.norm",
});
```

`dropout()` constructs inverted, training-only Dropout. Its mask is derived
deterministically from the node seed and training counter, and backward
recomputes that same mask. Normal inference treats the node as an exact
identity and does not allocate a mask or load a backward shader:

```js
const regularized = model.dropout(hidden, {
  probability: 0.1,
  seed: 42,
  name: "encoder.dropout",
});
```

The high-level encoder-decoder builder accepts `dropout` and `dropoutSeed`.
`dropout` places standalone Dropout after feed-forward activations and attention
or feed-forward output projections, and is also the default attention-probability
dropout for every SDPA/CrossSDPA node. Set `attentionDropout` independently when
the two probabilities differ. JavaScript CPU and WebGPU training regenerate the
same deterministic attention mask in forward and backward; inference applies
neither form of dropout.

## Mixture-of-Experts

`MoERouter` calculates expert logits and returns top-k indices and weights.
`MoELinear` executes the selected expert matrices and combines their results.
Both operations have forward and backward implementations.

```js
const router = model.weight("router", [8, 4], "float32", new Float32Array(32));
const experts = model.weight("experts", [4, 8, 16], "float32", new Float32Array(512));
const routes = model.moeRouter(hidden, router, {
  topK: 2,
  normalize: true,
  temperature: 1,
});
const { out } = model.moeLinear(hidden, experts, routes);
model.outputs(out);
```

With `normalize: true`, the selected experts are renormalized among the top-k.
With `normalize: false`, their weights are probabilities from the softmax over
all experts. The latter therefore propagates router gradients through selected
and unselected logits.

`maskedMean(input, normalizedWeights)` pools `[B,T,D]` states to `[B,D]` with
caller-supplied F32 weights shaped `[B,T]`. The helper deliberately does not
compute the denominator: supply already normalized weights, such as
`mask / max(validTokenCount, 1)`. This is useful for producing a masked summary
for a router or classification head while preserving gradients through the
input states.

## Adapter routing

Adapters are immutable, versioned snapshots. An execution can select one
adapter for the whole request or one route per batch row; the route scale is
multiplied by the adapter target's own scale.

```js
const options = model.adapterRouting(
  model.adapterRoute("tenant-a", 3, 0.75),
  model.adapterRoute("tenant-b", 1, 1.0),
);
const outputs = await engine.execute(inputs, options);
```

No adapter dispatch, buffer, or shader is created when execution has no active
adapter route.

For trainable per-token adapter routing, use explicit expert tensors in the
graph. `routedBottleneckAdapter()` applies routed down projection, GELU, routed
up projection, optional train-only Dropout, and a residual add:

```js
const down = model.weight("adapter.down", [experts, dModel, bottleneck], "float32", {
  initializer: { type: "xavierUniform", seed: 20 },
});
const up = model.weight("adapter.up", [experts, bottleneck, dModel], "float32", {
  initializer: { type: "zeros" },
});
const adapted = model.routedBottleneckAdapter(hidden, down, up, routes, {
  dropout: 0.1,
  seed: 21,
  name: "decoder.adapter",
});
```

Unlike immutable staged adapter snapshots, `down`, `up`, optional biases, and
router weights are ordinary graph weights and can be named in
`trainableTensors`.

## Runtime model creation

`CreateModel` accepts declared inputs, initial tensors, a `GraphInfo`, declared
outputs, and execution options. The runtime creates private config and
safetensors backing and returns the same model handle shape as `LoadModel`.
The graph and output list may be empty, so a client can create an empty model
and append its first node through `PatchGraph`. This is the native/service form
of starting from an empty model; the client does not have to manufacture an
empty safetensors file.

Created models allow tensor updates and graph patches. `AddModelTensor` can add
raw or initializer-backed parameters, while `RemoveModelTensor` removes an
unreferenced parameter. `PatchGraph` inserts, removes, replaces, or rewires
operator nodes. Tensor removal fails while a node still references it, and graph
edits are serialized with execution. If `Run.output_names` is empty, the outputs
declared by `CreateModel.output_names` are returned.

`Seq2SeqTrainStep` is the native teacher-forcing convenience RPC. It accepts
row-major rectangular I32 source and target IDs plus batch/sequence dimensions,
shifts every target row with `bos_id`, creates optional masks and position
inputs, maps `pad_id` labels to the explicit `ignore_id`, and delegates the
actual loss, backward pass, and update to `TrainStep`.

`SaveTrainingCheckpoint` writes a VolvoxAI checkpoint directory containing the
current graph, all weight shards, optimizer moments and training step, a
manifest, metadata, and optional tokenizer bytes. `LoadTrainingCheckpoint`
validates the package and returns a new model handle whose next implicit step
continues from that state. This is separate from `SaveModelWeights`, which saves
weights but is not a complete training-resume package.

## Base-model and adapter training

`TrainStep` is the general supervised API. It attaches softmax cross-entropy to
`logits_tensor`, backpropagates, and updates every F32 tensor named by
`trainable_tensors`; those names may identify base weights, convolution and
normalization weights, router/expert weights, or LoRA A/B tensors represented
as ordinary nodes in the graph. Backward implementations cover embeddings,
normalization, self/cross-attention, activations, convolutions, dense and
elementwise operations, and MoE routing/expert operations subject to the
backend allowlist. SGD and AdamW are supported.

Staged adapter snapshots are immutable routing resources, not graph tensors.
`TrainStep` pins the base route, leaving any active staged adapter active and
unchanged. To train LoRA, construct its A/B MatMul branch as graph tensors,
train those names, then stage/export a new adapter snapshot. `UpdateAdapter` is
the copy-on-write API for externally computed adapter updates.

`TrainingModelBuilder.loraLinear()` creates that explicit branch and returns
the exact factor names to pass to the optimizer:

```js
const lora = model.loraLinear(hidden, baseWeight, {
  rank: 8,
  alpha: 16,                 // effective scale = alpha / rank
  bias: baseBias,
  name: "decoder.projection",
});
model.outputs(lora.out);

const step = await volvox.trainLoRAStep(graph, {
  backend: "wasm",
  inputs: teacherForcedInputs,
  logitsTensor: "logits",
  targets: correctedTokenIds,
  trainableTensors: lora.trainableTensors,
  updateMode: "adamw",
  optimizer: { learningRate: 1e-4 },
});
```

The helper creates canonical F32 `A=[d_in,rank]` and
`B=[rank,d_out]` graph weights, plus a frozen scalar scale and explicit
base/low-rank/scale/Add nodes. A uses Xavier-uniform initialization and B starts
at zero unless overridden. Only A/B appear in `trainableTensors`; the base
weight, bias, and scalar therefore remain frozen. Use `layout: "dout"` for a
base matrix stored as `[d_out,d_in]`.

The helper does not stage an `AdapterManager` snapshot. A full checkpoint can
resume the explicit factors and optimizer state. For adapter-only deployment,
copy the trained A/B buffers into a new inactive snapshot and export it against
a base-only graph; activating it on the explicit-LoRA graph would apply the
same delta twice.

The JavaScript API exposes the same distinction through the trainable tensor
list:

```js
const result = await volvox.trainStep(graph, {
  backend: "webgpu",           // omit for the JavaScript CPU trainer
  inputs,
  logitsTensor: "logits",
  targets: targetIds,
  trainableTensors: ["router", "experts"],
  updateMode: "adamw",
  optimizer: { learningRate: 1e-4 },
});
```

### Weighted losses, clipping, and accumulation

Use `losses` for multiple weighted cross-entropy objectives. Each entry has a
unique name and its own logits, targets, ignore index, mask, weight, and
optional denominator. Multiple entries may reference the same logits tensor;
their seeded gradients are added before backward:

```js
const step = await volvox.trainStep(graph, {
  backend: "webgpu",
  inputs,
  losses: [
    {
      name: "tokens",
      logitsTensor: seq2seq.logits.name,
      targets: tokenTargets,
      lossMask: tokenMask,
      weight: 1,
      normalizer: activeTokensAcrossWindow,
    },
    {
      name: "router",
      logitsTensor: routerLogits.name,
      targets: routeTargets,
      weight: 0.05,
      normalizer: routeRowsAcrossWindow,
    },
  ],
  trainableTensors,
  gradientAccumulationSteps: 4,
  updateMode: "adamw",
  optimizer: { learningRate: 1e-4, maxGradNorm: 1 },
});
```

Without `normalizer`, each loss is divided by its own active example count and
then multiplied by `weight`. An explicit `normalizer` replaces that denominator.
For multiple losses accumulated over more than one microbatch, every loss must
provide the full-window normalizer; this makes the sum of microbatch gradients
match the intended combined objective. The `losses` form and the legacy
top-level `targets` fields are mutually exclusive. Results include the total
`loss` and a per-loss metric record.

`optimizer.maxGradNorm` clips using one Euclidean norm over all named trainable
gradients after accumulation and before the optimizer update. It is global, not
per tensor. A value of zero disables clipping; an applied result reports
`globalGradNorm` and `gradientScale`.

`gradientAccumulationSteps` defaults to one. Before a window is complete,
`trainStep()` returns `accumulating: true`, an `accumulationStep`, and no updated
tensors; the graph training step and optimizer state advance only when the
window is applied. `flushGradientAccumulation: true` applies the current partial
window, while `resetGradientAccumulation: true` discards an existing partial
window before processing the current microbatch. State can also be managed
between calls:

```js
const state = await volvox.getGradientAccumulationState(graph);
// { pending, microbatches, accumulationSteps, examples }
await volvox.resetGradientAccumulation(graph);
```

Changing the backend, topology, optimizer/loss signature, or trainable tensor
set while gradients are pending fails explicitly; reset the window first.

Training state is lazy. Normal inference does not allocate gradient buffers,
optimizer state, or backward pipelines. Adapter factor buffers and adapter
shader pipelines are likewise created only when an adapter route is used.
Consequently, adding training support does not impose a backward-pass cost on
inference-only execution. Native training temporarily uses its unfused graph
and restores the optimized inference graph after the step. Browser builds keep
training in `volvoxai.full.js`; the inference-only `volvoxai.js` entry has no
training dependency. WebGPU loads and compiles backward shaders only when
training is requested.

The current training execution choices are:

| Environment | Training path |
| --- | --- |
| Browser/Node | Pure JavaScript CPU autograd, or browser WebGPU/WGSL autograd. |
| Native | Native CPU autograd, Vulkan, desktop OpenGL 4.3+ compute, OpenGL ES 3.1+ compute, or Metal on Apple platforms. |

Native GPU training keeps supported F32 forward activations and backward
gradients on the GPU and synchronizes trainable gradients for the CPU optimizer.
A graph with an unsupported GPU backward operator or layout is preflighted to
the complete native CPU backward path before GPU backward starts; unsupported
operations never silently stop a gradient. Metal shader generation and wiring
can be validated on Linux, but the Metal runtime itself must be exercised on
macOS with an Apple GPU.

Trainable tensors must be F32, and every operation between the loss and those
tensors must have a backward implementation. Unsupported nodes fail the step
explicitly. SDPA and CrossSDPA accept either rank-2 `[sequence, width]` tensors
or rank-3 `[batch, sequence, width]` tensors; batch size is not fixed to one.
The portable WebGPU/native-GPU attention shaders limit head dimensions to 64,
and portable GPU MoE routing limits `topK` to 8. Native CPU fallback does not
inherit those shader limits. Native and WebGPU BatchNorm training use the stored
running statistics as an affine operation; they do not update those statistics.
NHWC GroupNorm and its affine gradients are supported by the JavaScript CPU,
WebGPU, native CPU, Vulkan, OpenGL compute, and Metal training paths. Standalone
Dropout is deterministic for a node seed and optimizer step during training,
uses inverted-dropout scaling, and remains a zero-overhead alias during
inference. Attention-probability dropout is distinct: JavaScript CPU and WebGPU
training support the exact deterministic probability masks for `SDPA` and
`CrossSDPA`. Native CPU, Vulkan, OpenGL compute, and Metal implement the same
after-softmax training rule and regenerate the mask during backward. A native
GPU dispatch that cannot run the dropout-aware shader/layout falls back to the
matching CPU path rather than treating output Dropout as equivalent. Native Add
and Mul use right-aligned
shape-aware broadcasting in forward and backward, including batched mask
shapes such as `[B,D,T] * [B,1,T]`. `TrainStep` gradient clipping uses one
global norm across the complete trainable tensor set.

## Migrating `tiny_receipt_vqa` from PyTorch

The model, loss, backward pass, optimizer update, and checkpoint state in
`tiny_receipt_vqa/train.py` have a complete F32 operator mapping to VolvoxAI
APIs. This is a source port, not a way to execute the existing PyTorch classes
unchanged, and the repository does not yet include a ready-to-run VolvoxAI
receipt-training CLI for JavaScript. Build the graph with `createModel()` (or
native `CreateModel`) and write a JavaScript driver, or use the complete native
C++ example in `examples/tiny_receipt_vqa/tiny_receipt_vqa_train.cc`. Initializers can
create every weight, so the first run does not need an empty or pre-existing
safetensors file.

| PyTorch component | VolvoxAI mapping | Porting detail |
| --- | --- | --- |
| `ConvBlock` / `ResBlock` vision stem | `Conv2D`, `groupNorm()`, `SiLU`, broadcast `Add` | Volvox image tensors are NHWC. Reshape the final `[B,10,21,D]` feature map to `[B,210,D]`, add learned image position/type weights, and pass it as `sourceFeatures`; no NHWC transpose is needed. |
| Character question and decoder embeddings | `Embedding`, learned position weights, example `buildEncoderDecoderTransformer()` | Question IDs are the source-token sequence. Use `tieSourceTargetEmbeddings`, `tieTargetEmbeddings`, and `lmHeadBias` to reproduce shared vocabulary/head weights and output bias. For the same training parameterization, keep `type_q` separate with lower-level `Add` nodes; folding it into position weights is only an architecture-compatible simplification because it changes gradients and AdamW state. |
| Pre-norm Transformer encoder/decoder | Example `buildEncoderDecoderTransformer(model, { sourceFeatures, dropout, attentionDropout, ... })` | Full encoder attention, causal decoder attention, cross-attention, padding masks, and teacher forcing are composed outside the package entry. Set `finalEncoderNorm: false` to match the PyTorch encoder's lack of a final norm; keep the decoder final norm. `GELU` defaults to PyTorch-compatible erf semantics; set `approximate: "tanh"` only for the optional approximation. JS CPU, WebGPU, native CPU, Vulkan, OpenGL compute, and Metal implement training attention dropout. |
| Masked question pooling and `TaskRouter` | `maskedMean()` plus Linear/GELU/Linear | Supply normalized `[B,T]` question weights and use the resulting raw family logits for router CE. `MoERouter` is an alternative top-k architecture, not an exact replacement for this two-layer auxiliary classifier. For exact pre-encoder routing, construct the question embedding/router subgraph explicitly; routing from encoded memory is a deliberate architecture change. |
| Memory/decoder `TaskAdapter` modules | `routedBottleneckAdapter()` through `transformMemory` and `transformDecoderOutput` | Expert weights `[E,D,H]` and `[E,H,D]` support hard one-hot or learned top-k routes, optional biases, Dropout, and residual addition. |
| `LoRALinear` and base freezing | Explicit A/B MatMul branches and `trainableTensors` | List only A/B, router, adapter, embedding, or base weights that should update; stage/export an immutable adapter snapshot after training if needed. |
| Token CE plus `0.1 * router CE` | `losses: [{name: "tokens", ...}, {name: "router", weight: 0.1, ...}]` | Each loss is normalized independently. With accumulation, give every loss its full-window `normalizer`; repeated logits entries are also valid and their gradients add. |
| AdamW and `clip_grad_norm_(..., 1.0)` | `updateMode: "adamw"`, optimizer options, `maxGradNorm: 1` | Clipping uses one norm across the complete trainable set. `gradientAccumulationSteps` plus reset/flush controls replaces larger logical batches. |
| `best.pt` / `last.pt` resume state | `exportCheckpoint()` / `importCheckpoint()` or native training checkpoint RPCs | Full checkpoints preserve graph weights, optimizer moments, step, training metadata, and optional tokenizer/application metadata. |

VolvoxAI owns the numeric training graph, but it is intentionally not a data or
experiment framework. The replacement application remains responsible for:

- record discovery, synthetic-example generation, train/validation splitting,
  shuffling, batching, worker concurrency, and feeding rectangular typed arrays;
- fixed-shape graph policy: pad question/decoder batches to configured maxima,
  or compile/cache the per-batch sequence shapes used by the PyTorch collator;
- image decoding, resize/normalization, and NHWC layout conversion. Exact
  PyTorch pixel-stream reproduction additionally requires its PIL
  contrast/brightness/blur/rotation augmentation; the bundled C++ trainer uses
  deterministic `stb_image` preprocessing and omits that redundant second
  augmentation because the Navercap synthetic images already contain capture
  degradations;
- building and persisting `CharVocab`, encoding/decoding, BOS/EOS/PAD policy,
  target formatting, and tokenizer metadata;
- changed-vocabulary checkpoint row remapping and separate LoRA/task-adapter
  component import/export formats used by the original script;
- PyTorch-compatible from-scratch initialization when trajectory comparison
  matters. Supply raw F32 buffers for Kaiming/uniform parameters and clone the
  prototype encoder/decoder layer values instead of using independent builder
  seeds;
- autoregressive validation, `extract_answer`, exact-match/grouped metrics,
  held-out evaluation, logging, and best-checkpoint selection;
- cosine learning-rate scheduling: compute the learning rate in the driver and
  pass it to each `trainStep()`;
- mixed-precision policy. VolvoxAI training currently updates F32 trainables and
  does not provide PyTorch `autocast`/`GradScaler` parity, so the initial port
  should train in F32 unless mixed-precision engine support is added separately.

Once those application responsibilities are ported, the F32 training loop no
longer needs PyTorch for model construction, forward/backward, optimization, or
checkpoint resume. It will not be bit-for-bit identical to PyTorch because the
random-number streams and backend reduction order differ; the CUDA script's
FP16 autocast and GradScaler trajectory is also intentionally outside the F32
training contract.
