# Native full-profile training

This directory is compiled only into the full native profile. The inference
profile does not compile these sources, expose generated training dispatch,
publish a `train` command, or export any Trainer, optimizer, backward, or PTQ
symbol.

## Trainer ownership

The generated `VxTrainingService` exposes Trainer IDs. Behind its handlers the
internal ownership is:

~~~text
VxModel
  VxTrainer
    exact retained base weight revision
    private VxEngineState
    private graph inputs and activations
    private gradients and accumulation window
    private SGD/AdamW slots and step
    private deterministic RNG stream
    private working weights
~~~

The internal trainer creation path loads the exact base revision into a new engine
capsule. CPU is the default. A requested Vulkan, OpenGL, Metal, or CUDA backend
is an exact requirement: device initialization and the differentiable graph
plan must succeed, and training never retries another backend after work
starts. External inference providers are not differentiable through this API.

Generated `TrainStepRequest` inputs are copied into the internal Trainer. The
request names one or more cross-entropy losses, the F32 model
weights to train, SGD or AdamW parameters, and accumulation/reset/flush policy.
`TrainStepResult` returns stable numeric metrics, accumulation state, whether
an optimizer update was applied, the private optimizer step, and the exact
backend route.

One applied step is:

~~~text
strict preflight → forward → loss → backward → finite check
                 → global gradient clip → private optimizer update
~~~

An unfinished accumulation window changes no weight. A failed step publishes
nothing and restores or discards private work back to the Trainer's committed
baseline. The Model is never mutated by input binding, forward/backward,
accumulation, export, or rollback.

`CommitTrainer` is the only weight-publication operation. It rejects an
unfinished accumulation window and a Trainer with no applied update. It
serializes and validates a private successor, then compare-and-publishes it
against the exact retained base revision. Concurrent Trainers therefore have
independent workspaces, and at most one can publish from the same base;
another receives `VX_STATUS_REVISION_CONFLICT`.

Successful commit also records the Trainer's optimizer baseline. A later
rollback restores the last committed weights, optimizer slots, step, RNG
position, and empty accumulation state. Existing CompiledModels remain pinned
to their prior immutable revision.

## Fixed command

`native/volvoxai train` is a model-agnostic fixed command. Its in-tree
runner calls the same generated Training service available to C embedders. The
command binds raw typed inputs and targets, runs private microbatches, commits
once, and exports the requested SafeTensors shards. Task preprocessing,
tokenization, sampling, and postprocessing remain outside the fixed binary.

## Internal composition

The full implementation contains CPU reference backward kernels, strict GPU
backward-plan construction, deterministic training Dropout, cross-entropy loss
seeding, accumulation, SGD/AdamW, global clipping, optimizer checkpointing,
and PTQ package authoring. All mutable graph/training state resolves through
the current Trainer-owned `VxEngineState` scope. Only synchronized physical
device state and model-independent module/pipeline caches may be shared.

PTQ is also full-profile-only. It observes declared F32 values, derives
explicit quantization parameters, packs weights and biases, and writes a new
`graph.json` plus safetensors package carrying `volvox-graph/v1`. Inference can
execute an authored quantized graph but contains no authoring implementation.

Generated training shaders, shader packs, embedded arrays, and PTX arrays are
build products and are never edited by hand.
