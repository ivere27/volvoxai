# Native runtime

native/ is a freestanding C runtime for the same package used by JavaScript:

~~~text
graph.json
model.safetensors
~~~

Every graph root must contain the exact discriminator
format: volvox-graph/v1. Public model-source fields use graph terminology.

The public C API uses only opaque handles:

~~~text
VxRuntime
  VxModel
    VxCompiledModel
      VxExecutionContext
        VxResult
~~~

Every operation receives its scope explicitly. A child retains the parent state
needed for its work, and results own immutable output snapshots independently
from their execution context.

## Build

~~~bash
make build_native

./native/volvoxai --version
./native/volvoxai --help
./native/volvoxai-full --help
~~~

The two fixed release artifacts are:

~~~text
native/volvoxai
native/volvoxai-full
~~~

volvoxai contains inference only. volvoxai-full adds the training command and
training implementation. The inference profile compiles and exports no
training implementation or public training symbol.

The fixed command surface is deliberately model-agnostic. Both executables
provide run, --help, and --version. Only volvoxai-full provides train.
Tokenization policy, image decoding, generation loops, and task postprocessing
stay in opt-in applications under examples/.

## Opaque C API

Include the public header:

~~~c
#include "volvoxai.h"
~~~

Create the runtime and load a package:

~~~c
VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
VxModelSource source = VX_MODEL_SOURCE_INIT;
VxReport report = VX_REPORT_INIT;

const char* weight_paths[] = {
    "model/model.safetensors",
};

source.graph_path = "model/graph.json";
source.weight_paths = weight_paths;
source.weight_path_count = 1;

VxRuntime* runtime = NULL;
VxModel* model = NULL;

VxStatus status = vx_runtime_create(
    &runtime_options, &runtime, &report);
if (status != VX_STATUS_OK) {
    fprintf(stderr, "%s\n", report.message);
    return 1;
}

status = vx_runtime_load_model(runtime, &source, &model, &report);
if (status != VX_STATUS_OK) {
    fprintf(stderr, "%s\n", report.message);
    vx_runtime_release(runtime);
    return 1;
}
~~~

`source.graph_path` must name `graph.json` or a named `*.graph.json` document;
other basenames are rejected before the file is opened.

The runtime validates graph.json and its format discriminator before backend
allocation.

Compile with explicit policy:

~~~c
VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
policy.mode = VX_BACKEND_REQUIRE;
policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
const char* required_backends[] = { "cpu" };
policy.backends = required_backends;
policy.backend_count = 1;

VxCompiledModel* compiled = NULL;
status = vx_model_compile(model, &policy, &compiled, &report);
if (status != VX_STATUS_OK) {
    fprintf(stderr, "%s\n", report.message);
    vx_model_release(model);
    vx_runtime_release(runtime);
    return 1;
}
~~~

VX_BACKEND_REQUIRE requires exactly one entry in backends. VX_BACKEND_PREFER
tries the listed backends in order; a null list with count zero uses the default
CPU-only preference. Operator-fallback policy is independent from provider
selection. Selection finishes before execution, and an execution failure is
never retried on another provider.

Query immutable model revisions and publish an adapter revision explicitly:

~~~c
VxRevisionInfo current = VX_REVISION_INFO_INIT;
vx_model_revision_info(model, &current, &report);

VxAdapterSource adapter = VX_ADAPTER_SOURCE_INIT;
VxAdapterRevision published = VX_ADAPTER_REVISION_INIT;
adapter.adapter_name = "tenant-a";
adapter.package_path = "adapters/tenant-a.safetensors";
vx_model_publish_adapter(model, &adapter, &published, &report);
~~~

Compiled models remain pinned to the revision captured at compile time. Use
vx_compiled_model_report() to retrieve their stored policy and route evidence.

Create a context, set typed inputs, and execute:

~~~c
VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
VxExecutionContext* context = NULL;
VxResult* result = NULL;

status = vx_compiled_model_create_context(
    compiled, &context_options, &context, &report);
if (status != VX_STATUS_OK) {
    /* release compiled/model/runtime */
    return 1;
}

status = vx_execution_context_set_input(
    context,
    "input",
    VX_DTYPE_F32,
    input_values,
    input_bytes,
    &report);
if (status == VX_STATUS_OK) {
    status = vx_execution_context_execute(context, &result, &report);
}
~~~

For a fixed-capacity sequence graph that supports row-aware operators,
`vx_execution_context_execute_prefix(context, row_count, ...)` recomputes only
the leading rows. It is stateless ordinary execution: no self-attention K/V or
dependency cache is retained between calls. This is useful for growing-prefix
parity and no-KV benchmarks. Built-in backends validate the prefix contract;
external providers currently return `VX_STATUS_BACKEND_UNSUPPORTED`.

Adapter selection is an explicit FIFO context operation:

~~~c
vx_execution_context_select_adapter(context, &published, &report);
/* Explicitly adopt the Model's currently published adapter later. */
vx_execution_context_rebind_adapter(context, &report);
~~~

No context adopts a graph, weight, or adapter revision implicitly.

Input dtype and byte size must match the graph declaration. Use
vx_execution_context_input_count() and vx_execution_context_input_info() to
inspect declared inputs.

Every result contains each declared graph output by exact name. Query and copy
an output into caller-owned storage:

~~~c
size_t required = 0;
status = vx_result_read(
    result, "logits", NULL, 0, &required, &report);
if (status != VX_STATUS_OK) return 1;

void* output = malloc(required);
status = vx_result_read(
    result, "logits", output, required, NULL, &report);
~~~

vx_result_output_count() and vx_result_output_info() expose output metadata.
The result remains readable after its context and parents are released.

Release handles when ownership ends:

~~~c
vx_execution_context_release(context);
vx_compiled_model_release(compiled);
vx_model_release(model);
vx_runtime_release(runtime);

/* result owns its snapshot independently */
vx_result_release(result);
~~~

retain/release is thread-safe at the handle boundary. release(NULL) is a no-op.
Context logical close rejects new work, drains accepted work, and is
idempotent:

~~~c
vx_execution_context_close(context, &report);
vx_execution_context_release(context);
~~~

vx_runtime_close() similarly rejects new root work while retained children keep
the resources they require. An already-created full-profile Trainer may finish
private work, rollback its private engine, and commit a successor after logical
Runtime close; creating a new Trainer after close is rejected.

## CPU worker threads

The CPU kernel pool defaults to one worker per **physical core** available to
the process. The count is derived from `sched_getaffinity` plus the sysfs CPU
topology, so a container CPU quota or an explicit affinity mask is respected
rather than the machine's total processor count, and SMT siblings are not
counted twice: byte-domain GEMM and normalization kernels are load/store bound,
so a second thread on the same core adds contention without throughput.

Override the count with either:

- `--threads <n>` on `volvoxai run`, or the equivalent pool API; or
- the `VOLVOXAI_THREADS` environment variable, which applies when no explicit
  count was requested.

An explicit request always wins over the environment variable, which in turn
wins over the derived default. Values are clamped to at least one worker and to
the pool's compiled maximum. On ARM the default remains two workers, because
heterogeneous big.LITTLE clusters were tuned separately.

## Status and reports

Every fallible function returns VxStatus and may fill VxReport. Stable status
values distinguish invalid arguments, closed handles, I/O, graph validation,
provider availability/support, compilation, execution, lookup, buffer size,
allocation, and internal failure.

VxReport records:

- status and lifecycle stage.
- execution identity.
- selected backend and any device identity reported by that provider.
- machine-readable reason.
- human-readable message.

Use status and structured report fields for control flow. Messages are
diagnostic text.

## Native provider SPI

External devices implement VxBackendProvider from volvoxai_backend.h. The
descriptor creates explicit provider-runtime, compiled-model, and
execution-context instances. Context execution writes every declared output
through a copying VxBackendOutputSink. The sink accepts only the graph's exact
name, F32/I32/I8/U8 dtype, rank, dimensions, and byte size for each output;
one mismatch, duplicate, or omission rejects the complete result.

The native host registers a provider on an open Runtime before loading or
compiling a model that selects it:

~~~c
VxBackendProvider provider = {
    .struct_size = sizeof(VxBackendProvider),
    .abi_version = VX_BACKEND_ABI_VERSION,
    .name = "my-npu",
    .user_data = &driver,
    .runtime_create = provider_runtime_create,
    .runtime_destroy = provider_runtime_destroy,
    .compile = provider_compile,
    .compiled_destroy = provider_compiled_destroy,
    .context_create = provider_context_create,
    .context_set_input = provider_context_set_input,
    .context_execute = provider_context_execute,
    .context_close = provider_context_close,
    .context_destroy = provider_context_destroy,
};

VxReport report = VX_REPORT_INIT;
VxStatus status = vx_runtime_register_provider(runtime, &provider, &report);
~~~

This registration composes the native implementation; it is not a Synurang
application operation. Synurang callers select already-composed providers by
name through protobuf `BackendPolicy` and receive their route evidence through
generated reports.

Callbacks never resolve graph or tensor state through a process-global table.
Each context owns its mutable request/device state. See
[Backend SDK](backend-sdk.md) for the full contract.

## Raw tensor runner

~~~bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 \
  --debug
~~~

Input and output suffixes must match declared storage: .f32, .i32, .i8, or
.u8. Byte sizes must match exactly, and each output contains its complete
declared tensor. Applications select task-specific rows or slices. Weightless
graphs may omit weight files, but graph preflight still rejects an unresolved
weight reference.

The runner does not infer image shape, normalization, vocabulary, or output
postprocessing. Frontends decode media and provide named tensors.

## Full-profile training command

~~~bash
./native/volvoxai-full train models/my_model \
  --input input=batch.f32 \
  --targets targets.i32 \
  --logits logits \
  --trainable classifier.weight \
  --trainable classifier.bias \
  --microbatches 10 \
  --accumulation-steps 2 \
  --optimizer adamw \
  --learning-rate 0.001 \
  --output-weights trained.safetensors
~~~

Targets are raw I32 class IDs. `--trainable`, input weight `--weights`, and
`--output-weights` are repeatable. The command runs every microbatch in a
private Trainer, flushes the final accumulation window, atomically commits one
successor revision, and then exports its weight shards. `--vulkan`,
`--opengl`, `--metal`, and `--cuda` are exact training requirements; the
command does not retry CPU. The inference executable omits and rejects this
command.

## Full-profile Trainer API

Include both headers when embedding training:

~~~c
#include "volvoxai.h"
#include "volvoxai_full.h"

VxTrainerOptions trainer_options = VX_TRAINER_OPTIONS_INIT;
VxCrossEntropyLoss loss = VX_CROSS_ENTROPY_LOSS_INIT;
VxTrainStepOptions step = VX_TRAIN_STEP_OPTIONS_INIT;
VxTrainStepResult step_result = VX_TRAIN_STEP_RESULT_INIT;
VxRevisionInfo published = VX_REVISION_INFO_INIT;
VxTrainer* trainer = NULL;

trainer_options.backend = "cpu"; /* exact; NULL also selects CPU */
trainer_options.rng_seed = 42;
vx_model_create_trainer(model, &trainer_options, &trainer, &report);

vx_trainer_set_input(
    trainer, "input", VX_DTYPE_F32, input_values, input_bytes, &report);

const char* trainables[] = { "classifier.weight", "classifier.bias" };
loss.logits_name = "logits";
loss.targets = targets;
loss.target_count = target_count;

step.losses = &loss;
step.loss_count = 1;
step.trainable_names = trainables;
step.trainable_count = 2;
step.optimizer.kind = VX_OPTIMIZER_ADAMW;
step.optimizer.learning_rate = 1.0e-3f;

vx_trainer_train_step(trainer, &step, &step_result, &report);
if (step_result.update_applied)
    vx_trainer_commit(trainer, &published, &report);

vx_trainer_release(trainer);
~~~

The Trainer owns an exact retained base revision and a private execution graph,
inputs, activations, gradients, optimizer slots, accumulation window, RNG, and
working weights. A step or unfinished accumulation window never changes the
Model. Commit rejects pending accumulation and compare-and-publishes one
validated immutable successor. Concurrent Trainers created from the same base
receive independent state; after one commits, another commit returns
`VX_STATUS_REVISION_CONFLICT`.

`vx_trainer_rollback()` restores the last committed Trainer baseline and
discards private weight, optimizer, gradient, accumulation, and RNG progress.
`vx_trainer_export_weights()` writes the current private shards without
publishing. Existing CompiledModels and ExecutionContexts stay pinned to the
revision they retained.

## Task applications

Build the separate model-specific command application with:

~~~bash
make -C examples native_task_cli
~~~

It owns image decoding, label files, and task postprocessing. Its generic
decode command forwards explicit seed, step, and reset operations without
owning tokenization or sampling policy:

~~~bash
examples/target/bin/volvoxai-tasks classify models/classifier \
  --image image=photo.jpg \
  --logits logits \
  --labels labels.txt \
  --top-k 5

examples/target/bin/volvoxai-tasks detect models/detector \
  --image image=receipt.jpg \
  --classes classes \
  --max-det 20

examples/target/bin/volvoxai-tasks decode models/decoder --help
~~~

Build the legacy whole-model and qualified encoder/decoder TinyReceipt
applications separately:

~~~bash
make -C examples native_receipt_inference_example
make -C examples native_receipt_split_inference_example
~~~

Both command lines accept an explicit incremental mode:

~~~bash
examples/target/bin/tiny_receipt_w8a8 build/tiny-receipt-w8a8 \
  --image receipt.png --prompt "What is the phone number?" \
  --family phone --max-new 96 --incremental

examples/target/bin/tiny_receipt_split_w8a8 build/tiny-receipt-runtime-int8 \
  --image receipt.png --prompt "What is the phone number?" \
  --family phone --max-new 96 --incremental --cpu --require-row
~~~

The legacy `tiny_receipt_w8a8` command uses ordinary per-token forwards by
default. `--incremental` instead seeds once and then uses decode steps with
dependency and native CPU row/KV reuse where supported. The split
`tiny_receipt_split_w8a8` command uses incremental decoding by default and
also accepts `--incremental` to make that selection explicit. `--no-kv`
recomputes only the growing prefix with
`vx_execution_context_execute_prefix()` and retains no decoder cache;
`--ordinary` recomputes the complete fixed-capacity decoder tensor every
token. These modes are mutually exclusive, and `--require-row` applies only to
incremental execution.

The legacy command warns when a package manifest explicitly marks its
activation profile as `qualified_per_edge_calibration: false`. A single-scale
fallback package is suitable for graph and loader checks, not answer-quality
validation; ordinary and incremental execution are expected to reproduce the
same potentially inaccurate tokens from that package.

Image packages may declare per-input image_normalization metadata. Explicit
frontend options select zero-one, minus-one-one, or raw-255 behavior when
application policy requires it.

The TinyReceipt applications likewise validate their package manifests,
perform grayscale and resize preprocessing, run or consume their router, own
the autoregressive loop, and read the declared token_ids result. They are not
linked into either fixed release executable.

## Backend composition

Device source composition is controlled at CMake time:

~~~text
VOLVOXAI_ENABLE_VULKAN
VOLVOXAI_ENABLE_OPENGL
VOLVOXAI_ENABLE_CUDA
VOLVOXAI_ENABLE_METAL
VOLVOXAI_ENABLE_NNAPI
~~~

Each accepts ON or OFF. Disabled backend source, registration, and shader
blocks are omitted. Linux defaults to Vulkan and OpenGL; macOS also defaults to
Metal; Android enables NNAPI. CUDA is opt-in.

Driver libraries are resolved at runtime:

- Vulkan: libvulkan.
- OpenGL: libGL, opengl32, or the macOS OpenGL framework.
- CUDA: the NVIDIA Driver API; neither cudart nor libcuda is linked.
- Metal: the default MTLDevice through the Objective-C runtime.
- NNAPI: Android Neural Networks device integration.

An explicitly required provider that was not compiled or cannot initialize
fails with a backend status. It does not silently switch to CPU.

Android 15 deprecates NNAPI. Current Android device integrations should expose
QNN, LiteRT delegates, or another vendor driver through VxBackendProvider when
that is the selected deployment interface.

## CPU kernels

Normal releases use a baseline CPU target. Runtime checks select x86 AVX2,
AVX-VNNI, and AVX-512 VNNI or ARM NEON/SDOT W8A8 kernels only when the CPU and
OS support the required state. Portable scalar kernels remain the fallback.

The AVX2 QLinear route uses an exact U8-by-I8 `VPMADDUBSW` decomposition rather
than relying on its saturating I16 pair result directly; arbitrary I8/U8
zero-points therefore remain byte-identical to the portable kernel. Immutable
weights are packed while the native model is prepared. In addition to the
portable K-by-8 pack, x86 keeps a derived K4-by-8 companion pack. Symmetric I8
multi-row calls consume two panels at a time in an MR4/N16 microkernel. The
pack records whether any weight is -128. When none is present, signed input
bytes use `abs(input) * sign(weight,input)`; each pair is bounded by
`2*128*127` and cannot saturate. Packs containing -128 retain the two-part
unsigned decomposition. Both routes apply exact affine zero-point correction.
This extra pack is runtime state, not serialized model data. Benchmark policy
leaves ordinary M=1 decode on the faster raw GEMV route.

Dense groups=1 3x3 QConv2D builds the same flattened pack once for symmetric
I8 weights and reuses it through a bounded zero-point-padded im2col buffer.
For dilation-one convolutions, the im2col producer partitions output rows and
copies each in-bounds 3*C input strip contiguously; border strips are still
filled with the declared input zero point.
Unsupported geometry, asymmetric weights, allocation failure, and non-AVX2
hosts retain the direct or portable paths.
QBatchMatMul interleaves two K rows across sixteen output columns and uses the
same non-saturating decomposition, including exact affine zero-point
compensation. Its ARM route widens eight consecutive output columns with NEON.
AVX2 QSDPA vectorizes centered QK dots and value accumulation without changing
the online-softmax key order.

Use NATIVE_CPU_TARGET=baseline for a portable binary or
NATIVE_CPU_TARGET=avx2 for a deployment that guarantees AVX2/FMA.
NATIVE_CPU_FLAGS is the low-level build override.

Native Linux/Android AArch64 builds compile dot-product QLinear and QConv2D as
separately gated armv8.2-a+dotprod objects. Runtime HWCAP selects SDOT when it
is available; ordinary Armv8 cores keep the NEON baseline path in the same
binary.

Large QLinear and QBatchMatMul work partitions output rows; QConv2D partitions
independent NHWC output positions or dense-im2col output rows; whole-tensor
QSDPA partitions independent batch/query rows. QGroupNorm partitions groups,
QLayerNorm partitions final-axis rows, and large QSiLU tensors partition
contiguous byte ranges. Single-row decode and small work stay on the caller to
avoid worker wake-up. In particular, the QSiLU and QLayerNorm thresholds keep
the bounded TinyReceipt decoder prefix on the caller while still pooling its
larger encoder tensors.

On AVX2, QLayerNorm retains the canonical scalar order for mean and variance
and vectorizes only independent affine/requantization lanes. QGroupNorm can
evaluate four independent groups in SIMD lanes while preserving the scalar
spatial/channel reduction order within each group. Both are runtime-gated;
the baseline dispatcher and non-AVX2 fallback contain no AVX instructions.

QSiLU and QGELU may cache all 256 physical byte results for an immutable
input/output quantization descriptor. The table uses the same activation and
ties-to-even requantization math as the scalar route.

## GPU behavior

Vulkan, OpenGL, CUDA, and Metal have strict device implementations for their
documented operator subsets. Backend compilation validates dtype, shape,
layout, quantization, and operator-fallback policy before creating a context.

CUDA is an opt-in manual-PTX backend. Forward PTX is embedded in both profiles;
training/PTQ PTX is present only in the full profile. Host integration resolves
the Driver API dynamically.

~~~bash
cmake -S . -B build/cuda -DCMAKE_C_COMPILER=clang \
  -DVOLVOXAI_ENABLE_CUDA=ON -DVOLVOXAI_CUDA_ARCH=75
cmake --build build/cuda --target volvoxai volvoxai-full
~~~

See [CUDA](cuda.md) and [the operation matrix](operation_list.md) for exact
coverage and validation.

## Embedded shaders

Vulkan contributes SPIR-V, OpenGL contributes desktop GLSL and GLES, and Metal
contributes MSL. An all-GPU-off build carries an empty pack. CUDA PTX is
generated from its separate authoritative CUDA sources and is not part of the
WGSL pack.

Inference profiles embed forward shaders only. Full profiles add training
blocks. Each XZ-compressed block is decoded and cached only when first used.

For shader development:

~~~bash
make compile_shaders
VOLVOXAI_SHADER_DIR=native/shaders ./native/volvoxai --help
~~~

VOLVOXAI_SHADER_DIR is a development override. VolvoxAI logs once only when an
external shader is actually loaded. Missing external files fall back to the
embedded store.

Generated shader files and embedded byte arrays are never edited by hand.

## macOS

~~~bash
cmake -S . -B build/mac -DCMAKE_C_COMPILER=clang \
  -DVOLVOXAI_ENABLE_METAL=ON
cmake --build build/mac
~~~

Metal runtime validation requires macOS and an Apple GPU. Set VULKAN_SDK if the
same build also compiles the runtime-loaded Vulkan provider.

## Android cross-compile

~~~bash
export ANDROID_NDK="$HOME/Android/Sdk/ndk/<version>"
cmake -S . -B build/android -DCMAKE_C_COMPILER=clang \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29
cmake --build build/android
~~~

Push the executable and model directory; shaders are embedded:

~~~bash
adb push volvoxai_android /data/local/tmp/volvoxai
adb shell "cd /data/local/tmp && ./volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 --vulkan --debug"
~~~

## Full-profile PTQ plan API

`volvoxai_full.h` exposes PTQ authoring through one opaque `VxPTQPlan`. Plan
creation retains the Model, pins its exact graph/weight/adapter revision, copies
the caller-authored graph template, and creates a private CPU engine. Calibration
never uses a selected inference backend and never mutates or publishes the
Model. If the Model publishes a different pinned revision, later inspect,
calibrate, or write operations fail with `VX_STATUS_REVISION_CONFLICT`.

The current writer is deliberately exact: it supports W8A8 `QLinear` and
`QConv2D`, requires exactly one source safetensors shard, and writes a package
whose graph basename is exactly `graph.json` and whose single sibling weight
file ends in `.safetensors`. Output files must not already exist.

```c
VxReport report = VX_REPORT_INIT;
VxPTQObserverSpec observers[2] = {
    VX_PTQ_OBSERVER_SPEC_INIT,
    VX_PTQ_OBSERVER_SPEC_INIT,
};
VxPTQLayerSpec layer = VX_PTQ_LAYER_SPEC_INIT;
VxPTQPlanOptions options = VX_PTQ_PLAN_OPTIONS_INIT;
VxPTQInput input = VX_PTQ_INPUT_INIT;
VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
VxPTQPackageOptions package = VX_PTQ_PACKAGE_OPTIONS_INIT;
VxPTQPlan* plan = NULL;
uint64_t sample_count = 0;

observers[0].tensor_name = "input";
observers[0].dtype = VX_DTYPE_I8;
observers[1].tensor_name = "projected";
observers[1].dtype = VX_DTYPE_I8;

layer.kind = VX_PTQ_LAYER_QLINEAR;
layer.node_index = 0;
layer.input_tensor_name = "input";
layer.output_tensor_name = "projected";
layer.source_weight_name = "projection.weight";
layer.packed_weight_name = "projection.weight.i8";
layer.source_bias_name = "projection.bias";
layer.packed_bias_name = "projection.bias.i32";

options.template_graph_path = "authoring/graph.json";
options.observers = observers;
options.observer_count = 2;
options.layers = &layer;
options.layer_count = 1;
vx_model_create_ptq_plan(model, &options, &plan, &report);

input.name = "input";
input.dtype = VX_DTYPE_F32;
input.data = calibration_values;
input.byte_size = calibration_value_count * sizeof(float);
vx_ptq_plan_calibrate(
    plan, "sample-0", &input, 1, &sample_count, &report);

vx_ptq_plan_info(plan, &info, &report);
for (size_t index = 0; index < info.tensor_count; index++) {
    VxPTQTensorParameters parameters = VX_PTQ_TENSOR_PARAMETERS_INIT;
    vx_ptq_plan_tensor_parameters(plan, index, &parameters, &report);
    /* parameters contains scale, zero point, range, and observation count. */
}

package.output_graph_path = "dist/quantized/graph.json";
package.output_weights_path = "dist/quantized/model.safetensors";
vx_ptq_plan_write_package(plan, &package, &report);
vx_ptq_plan_close(plan, &report);
vx_ptq_plan_release(plan);
```

`vx_ptq_plan_input_count` and `vx_ptq_plan_input_info` expose the exact named
calibration inputs. Every sample must bind each descriptor once with matching
host dtype and byte size. Sample names are unique within the plan. Inspection
returns the pinned revision and accumulated tensor parameters without exposing
the plan's private engine.

## Validation and benchmarks

~~~bash
make test_native
make test_native_all
make test_native_gpu
make benchmark_native
~~~

Correctness tests compare accelerated kernels with the portable CPU
implementation. A required physical device that is unavailable is reported as
unavailable rather than counted as a pass. Benchmark results are device,
driver, model, and build specific.

## In-process Synurang FFI

The Rust crate under `runtime/` is an optional full-profile Synurang plugin over
the opaque C handles. `proto/volvoxai.proto` is its sole public contract; the
generated dispatcher calls native inference, Trainer, and PTQ operations in
the same process. Task policy remains outside the core runtime. See
[runtime/README.md](../runtime/README.md).

### Dual export surface

`libvolvoxai.{so,dylib}` exports two tiers from one library:

- `Synurang_*` — the generated Synurang protobuf FFI. Portable and
  language-neutral; any Synurang-supported caller uses it. Tensor payloads
  cross this boundary as protobuf `bytes`, so they are copied on encode and
  decode (control-plane and convenience data path).
- `vx_*` — the hand-written native C API of `native/include/volvoxai.h` and
  `volvoxai_full.h`. It takes raw pointers (`vx_execution_context_set_input(...,
  const void* data, ...)`) with no serialization copy, so the performance data
  path (zero-copy) links this directly. Callers that need zero-copy include the
  headers and link the same library; the portable Synurang entry points remain
  available for everyone else.

Both surfaces are the same in-process engine: the plugin adapts each RPC onto
the `vx_*` handles (`runtime/src/abi.rs`), and both are compiled into the one
cdylib.

The C engine is built with `-fvisibility=hidden`, so only `VX_API`
(`visibility("default")`) symbols are export-eligible; internal `vx_`-prefixed
helpers stay hidden. rustc otherwise localizes every symbol pulled from the
static engine archive, leaving only `Synurang_*` in the dynamic table. To
re-export the C API alongside it, `runtime/build.rs` emits a linker export list
into `OUT_DIR` and passes it to the final cdylib link:

- ELF (Linux, Android): a version script `{ global: Synurang_*; vx_*; local: *; };`
  via `-Wl,--version-script`.
- Mach-O (macOS): an exported-symbols list `_vx_*` / `_Synurang_*` via
  `-Wl,-exported_symbols_list`.
- Windows is skipped; `VX_API` carries no `__declspec(dllexport)`, so a `.def`
  file would be required to expose `vx_*` there.

Verify the export set on the built library:

~~~bash
nm -D --defined-only target/release/libvolvoxai.so | grep -E 'vx_|Synurang_'
~~~

Both `Synurang_*` and the public `vx_*` symbols must appear; internal helpers
must not.
