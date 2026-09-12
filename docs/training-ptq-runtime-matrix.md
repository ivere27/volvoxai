# Training and PTQ profile matrix

All full operations reach generated C dispatch. A particular graph, backend,
dtype or resource demand can still produce a typed refusal.

| Composition | Training | PTQ |
| --- | --- | --- |
| Ordinary JS/WASM and native inference | Physically absent | Physically absent |
| Full JS/WASM | C trainer, WASM CPU or WebGPU | C authoring, calibration, inspection and package bytes |
| Native full | C trainer on supported native backends | C authoring/calibration, bytes or native paths |

The full host has one persistent C owner for inference, scheduler, training and
PTQ. TypeScript has no trainer, optimizer, graph execution or provider-selection
implementation. WebGPU training runs only when C can construct the entire
forward/backward/optimizer device plan; there is no fallback during a step.

## Trainer

CreateTrainer pins a Model revision. TrainStep accepts a complete typed batch,
losses, trainable names and optimizer options. Supported SGD/AdamW steps mutate
private C state. CommitTrainer publishes a successor revision; RollbackTrainer
restores the committed baseline. Releasing parent IDs does not invalidate a
retained Trainer.

WebGPU TrainStep returns PENDING. GetTrainStep queries its trainer/microbatch ID
until READY or FAILED and never replays work. Only the latest accepted step is
retained. Step, commit, rollback and export return BUSY while a step is pending;
ReleaseTrainer discards pending work safely. CPU steps return READY immediately.
Readback publishes weights and optimizer state atomically. Numerical/device
failure restores the committed baseline, with a poisoned state if restoration fails.

The WebGPU matrix covers 70 training graphs: dense layers and scalar/tensor
broadcasting, activations, normalization, pooling, Conv1D/Conv2D/transposed
convolution, spatial/view/index operators, embeddings, attention, Dropout and
MoE. Every fixture compares CPU/GPU loss and updates, and samples independent
finite-difference derivatives. Wide Concat/Split graphs exercise more than 32
ports. See the [training fixtures](../tests/parity/external/webgpu_training_operators.mjs)
and [runtime validation](c-runtime-validation.md) for the tested domains and
results; this does not assert every possible graph.

`ExportTrainerCheckpoint` / `CreateTrainer.checkpoint` preserve private weights,
SGD/AdamW state, optimizer settings, RNG and the rollback baseline.
`GetTrainerState`, `ResetTrainerAccumulation` and `shape_options` expose C-owned
accumulation, cache keys/LRU/bypass accounting and activation growth policy.
`InitializeTensor`, `BuildLoraLinear` and `BuildRoutedAdapter` are full-only C
authoring calls. Quantized LoRA keeps F32 masters during training and exports
immutable I8 inference snapshots through `ExportQuantizedTrainerWeights`.
The reference TypeScript implementation's F32 optimizer storage requirement is
retained; the migration does not introduce mixed-precision training.

ExportTrainerWeights with empty output paths returns `TrainerWeights.shards`
containing SafeTensors bytes. Native callers may request output paths; WASM
returns TRANSPORT_UNSUPPORTED for those paths. Browser applications save or
transfer the returned bytes themselves.

## Quantization

| Operation | Full WASM | Native full |
| --- | --- | --- |
| AuthorPtqTemplate from graph/shard bytes | Yes | Yes |
| AuthorPtqTemplate from filesystem paths | Typed refusal | Yes |
| CreatePtqPlan from template bytes | Yes | Yes |
| CreatePtqPlan from a native template path | Typed refusal | Yes |
| CalibratePtqPlan / InspectPtqPlan | C CPU | C CPU |
| WritePtqPackage with empty output paths | Graph and weights bytes | Graph and weights bytes |
| WritePtqPackage with output paths | Typed refusal | Native files |

The plan snapshots the template and retains the exact graph/weight/adapter
revision being calibrated. Calibration validates the whole batch and commits
observer ranges atomically. Observation-only plans may omit quantized layers;
materializing a quantized package requires at least one layer and complete
coverage. The writer currently accepts one source SafeTensors shard.

Returned package bytes remain usable after releasing the plan. Tests repeat the
write, release the plan, reload the exported model and compare its execution to
the F32 source. The same C authoring code produces matching canonical bytes in
native and WASM. There is no embedded PTQ child or JavaScript decimal formatter.

See [typed PTQ](typed-ptq.md), [training usage](model_builder_training.md), and
[validation evidence](c-runtime-validation.md).
