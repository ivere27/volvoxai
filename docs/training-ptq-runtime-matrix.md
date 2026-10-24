# Training and PTQ profile matrix

All full operations reach generated C dispatch. A particular graph, backend,
dtype or resource demand can still produce a typed refusal.

| Composition | Training | PTQ |
| --- | --- | --- |
| Ordinary JS/WASM and native inference | Physically absent | Physically absent |
| Full JS/WASM | C trainer, WASM CPU or WebGPU | C static PTQ, group weight packing and package bytes |
| Native full | C trainer on supported native backends | C static PTQ and group weight packing, bytes or native paths |

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

Weight-only W2/W3/W4 QAT is opt-in through `CreateTrainer.qat` on CPU/WASM and native CUDA.
It uses FP32 masters, temporary reconstructed weights and clipped STE gradients;
scales are refitted with no scale gradient. Other device backends and activation
QAT are unsupported. Checkpoints and trainer state retain the fixed policy.
See [the QAT workflow](model_builder_training.md#optional-weight-only-qat).

## Quantization

| Operation | Full WASM | Native full |
| --- | --- | --- |
| CreateQuantizationPlan from graph/shard bytes | Yes | Yes |
| CreateQuantizationPlan from filesystem paths | Typed refusal | Yes |
| CalibrateQuantizationPlan for static activation affines | C CPU | CPU or CUDA observations |
| CalibrateQuantizationPlan for IMPORTANCE weights | C CPU | CPU or CUDA observations |
| CalibrateQuantizationPlan for grouped RTN/MSE_CLIP weights | Typed refusal | Typed refusal |
| InspectQuantizationPlan / ReleaseQuantizationPlan | Yes | Yes |
| ExportQuantizedModel with empty output paths | Graph and weights bytes | Graph and weights bytes |
| ExportQuantizedModel with output paths | Typed refusal | Native files |

One plan owns the source and an explicit `QuantizationConfig`. Weight bits and
group size are separate from optional static activation affines and compiled
packed arithmetic. The
caller does not create a source Runtime or Model. Creation validates the policy
and snapshots one SafeTensors shard with its graph.

`CalibrationOptions.backend` is empty/`cpu` by default or explicitly `cuda`
on native. There is no backend fallback. `QuantizationPlanInfo.calibration_backend`
reports the actual observer executor, or an empty string for plans needing no
calibration. CUDA reduces activation ranges and input-energy statistics on the
device, then transfers compact statistics for atomic publication. Weight fitting
and package writing remain host work, including static W8: selecting CUDA
observations does not select a different export quantizer. An unrelated suffix need not execute once
all required observations have been produced; this is calibration support, not
a claim that every operator in the exported model supports CUDA inference.

STATIC plans privately retain the exact graph and weights they calibrate.
Calibration validates the whole batch and commits observation ranges atomically.
Every named profile needs a successful sample before export. Inspection reports
readiness, input contracts and calibration coverage. Supported static policies
use per-output-channel I8 RTN weights, or Conv-only I2/I3/I4 RTN/MSE/IMPORTANCE weights, and
calibrated I8/U8 activation affines. Conv4 stores two offset-signed nibbles per
U8 byte with one F32 scale per output channel; native CPU and WASM execute it.

Grouped RTN and MSE clipping plans prepare I2/I3/I4/I8 `PackedLinear` weights with
group size 32, 64, 128 or 256 without creating an inference engine. They are
ready immediately, expose no calibration state, and add no calibration-coverage
metadata. Importance fitting requires calibration. Compilation independently
selects F32, F16, DYNAMIC_I8 or F32_FAST arithmetic; dynamic I8 row scales and
F16 operand rounding happen at execution. See the
[storage and fitting policy table](quantization.md#choose-storage-and-fitting).

Returned package bytes remain usable after plan release and owner close.
Tests repeat export, retire the authoring owner, reload the result in full and
lite, and check the numerical contract. Native device backends belong to full;
lite contains CPU inference only. Apple Silicon execution remains a hardware
qualification task; NEON compilation and Metal shader translation are separate
checks. There is no embedded PTQ child or JavaScript quantization kernel.

See [typed static PTQ](typed-ptq.md), [training usage](model_builder_training.md),
and [validation evidence](c-runtime-validation.md).
