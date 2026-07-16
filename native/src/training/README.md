# Native training boundary

`training_forward_dropout.inc` owns the private training-mode state and the CPU
Dropout/attention-dropout forward helpers. `training_runtime.inc` is a thin
umbrella that composes the native training runtime from three cohesive
fragments, in dependency order:

- `training_backward_cpu.inc` — CPU reference backward kernels (`grad_*`
  helpers and the per-operator `backward_*` passes the accelerated kernels are
  validated against).
- `training_gpu_plan.inc` — GPU backward-plan construction, dispatch, and
  execution/synchronization, falling back to the CPU kernels when unsupported.
- `training_step.inc` — the backward driver that walks the graph, training
  graph preparation/restoration, gradient accumulation, and the train-step
  public API implementations.

`optimizer_runtime.inc` owns optimizer storage, tensor-update implementations,
and optimizer checkpoint APIs.

`quantization.c` owns stateless full-profile PTQ range/parameter and packing
math. The same allocation-free translation unit is linked into
`volvoxai.full.wasm`, which exposes its generic observer, affine quantization,
weight-packing, and bias-packing primitives through a typed scratch-backed
JavaScript API. Transpose-aware conversion helpers additionally support the
quantized-LoRA convenience workflow; `volvoxai.wasm` remains free of all PTQ
exports. `quantization_runtime.c` is the narrow internal bridge that
observes a named loaded F32 tensor and materializes new I8/F32 weight entries
while holding the model and metadata locks. `quantization_package.c` owns the
opaque explicit PTQ plan, generation checks, transactional named-sample
observation, validation against the loaded private graph, and
config+safetensors package emission.
These files compile as ordinary full-profile translation units and reach only
the small private runtime surface declared in `engine_internal.h`; the
inference target compiles none of them.

The three backward/train fragments listed first are included only from
`training_runtime.inc`, never on their own.

They are intentionally C includes rather than separately compiled translation
units. These implementations operate on private `static` runtime helpers and
state; compiling them separately would either expose that state as a broad
internal API or duplicate it. `engine_runtime.c` reaches the forward/backward
fragments once, and `engine.c` reaches the optimizer fragment once, all behind
`VOLVOXAI_ENABLE_TRAINING`. The full build preserves the existing encapsulation
while the inference build does not compile the training implementation or
export its public symbols.

Small forward helpers used by both profiles remain in `engine_runtime.c`, such
as shape/broadcast validation and GroupNorm forward execution. The inference-
side lifecycle hooks for accumulation and optimizer cleanup are private
compile-time no-ops.

Backend-local backward schedulers remain in each backend source because they
own that device's descriptor pools, buffers, pipelines, and synchronization.
Their tables, APIs, and training-only forward shaders use the same profile
guard, so inference objects contain neither scheduler code nor training shader
paths.

`VOLVOXAI_ENABLE_TRAINING` defaults to `0`. Full builds must opt in explicitly
with `VOLVOXAI_ENABLE_TRAINING=1`.
