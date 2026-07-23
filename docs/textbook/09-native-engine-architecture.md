# Chapter 9 — The Native Runtime

*Read the badges that match you: 🌱 **Idea** (anyone, no code) · 🔧 **Build** (a little code) · 🔬
**Deep** (runtime developers). New here? Follow just the 🌱 sections.*

*Goal: understand how the same VolvoxAI package runs in a freestanding C program. We will follow
the public opaque-handle lifecycle, see how compilation fixes a provider route, and learn why
contexts and immutable results make concurrent native execution safe.*

> 🌱 **The big idea.** A native application does five things: create a runtime, load a model,
> compile it for a device, create a private execution context, and read a result. Loading and
> compilation are shared work. Mutable request state belongs to a context. Each completed result
> keeps its own output snapshot.

## 9.1 The same package, without a browser

The native runtime consumes the canonical package:

~~~text
model/
├── graph.json
└── model.safetensors
~~~

The Graph root has the exact discriminator:

~~~json
{
  "format": "volvox-graph/v1"
}
~~~

Graph topology, named inputs and outputs, dtypes, shapes, operator attributes, and weight
descriptors live in graph.json. Safetensors files hold tensor bytes. Native and JavaScript
frontends therefore compile the same model contract.

The fixed release programs are:

~~~text
native/volvoxai
native/volvoxai-full
~~~

The first contains inference only. The second adds the training command and its compiled training
implementation. The inference program has no training implementation or public training symbol.

## 9.2 The ownership tree

The public header native/include/volvoxai.h exposes five opaque handle types:

~~~text
VxRuntime
  VxModel
    VxCompiledModel
      VxExecutionContext
        VxResult
~~~

- **VxRuntime** owns provider runtimes and root policy.
- **VxModel** owns one validated package snapshot.
- **VxCompiledModel** owns a selected, compiled provider route.
- **VxExecutionContext** owns mutable inputs and request/device state.
- **VxResult** owns immutable declared-output snapshots from one execution.

Every handle has retain/release operations. A child retains the parent state it needs, so releasing
a Runtime variable does not invalidate an already retained context or result. Logical close rejects
new work and is idempotent.

This shape is the native form of the JavaScript lifecycle:

~~~text
Runtime → Model → CompiledModel → ExecutionContext → ExecutionResult
~~~

## 9.3 Create, load, and compile

🔧 Include volvoxai.h and initialize every options/report struct with its matching macro:

~~~c
#include "volvoxai.h"
#include <stdio.h>

VxReport report = VX_REPORT_INIT;
VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
VxRuntime* runtime = NULL;

VxStatus status = vx_runtime_create(
    &runtime_options, &runtime, &report);
if (status != VX_STATUS_OK) {
    fprintf(stderr, "%s\n", report.message);
    return 1;
}

const char* weights[] = {
    "model/model.safetensors",
};
VxModelSource source = VX_MODEL_SOURCE_INIT;
source.graph_path = "model/graph.json";
source.weight_paths = weights;
source.weight_path_count = 1;

VxModel* model = NULL;
status = vx_runtime_load_model(runtime, &source, &model, &report);
if (status != VX_STATUS_OK) {
    fprintf(stderr, "%s\n", report.message);
    vx_runtime_release(runtime);
    return 1;
}
~~~

Loading validates the Graph and weight descriptors before device allocation. Compilation applies an
explicit provider policy:

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

VX_BACKEND_REQUIRE requires exactly one entry in backends. VX_BACKEND_PREFER tries the listed
backends in order; a null list with count zero uses the default CPU-only preference. Operator
fallback is a separate choice. Selection ends at compile time: an execution failure is reported
and is never retried on another provider.

🔬 VxReport records the lifecycle stage, status, selected backend, any device identity reported by
that provider, reason, message, and execution identity. Programs should branch on VxStatus and
structured report fields; the message is diagnostic text.

## 9.4 Context execution

One compiled model can create multiple contexts. Their inputs, scratch storage, decode state, and
in-flight work never alias.

~~~c
VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
VxExecutionContext* context = NULL;

status = vx_compiled_model_create_context(
    compiled, &context_options, &context, &report);

if (status == VX_STATUS_OK) {
    status = vx_execution_context_set_input(
        context,
        "input",
        VX_DTYPE_F32,
        input_values,
        input_bytes,
        &report);
}

VxResult* result = NULL;
if (status == VX_STATUS_OK) {
    status = vx_execution_context_execute(context, &result, &report);
}
~~~

Use vx_execution_context_input_count and vx_execution_context_input_info to inspect declared inputs.
set_input rejects a name, dtype, or byte count that disagrees with the Graph.

Execution publishes every declared output exactly once. It does not expose mutable intermediate
tensors or a borrowed workspace pointer.

## 9.5 Stable results and readback

VxResult remains readable after later executions and after its context is closed. Query the required
size, allocate caller-owned storage, and copy by exact output name:

~~~c
#include <stdlib.h>

size_t required = 0;
status = vx_result_read(
    result, "logits", NULL, 0, &required, &report);

void* output = NULL;
if (status == VX_STATUS_OK) {
    output = malloc(required);
    if (output == NULL) {
        status = VX_STATUS_OUT_OF_MEMORY;
    }
}
if (status == VX_STATUS_OK) {
    status = vx_result_read(
        result, "logits", output, required, NULL, &report);
}
~~~

vx_result_output_count and vx_result_output_info expose names, shapes, dtypes, byte sizes, and memory
locations. vx_result_execution_id identifies the producing execution.

Release ownership explicitly:

~~~c
vx_execution_context_close(context, &report);
vx_execution_context_release(context);
vx_compiled_model_release(compiled);
vx_model_release(model);
vx_runtime_release(runtime);

/* The snapshot is independent from those handles. */
vx_result_release(result);
free(output);
~~~

## 9.6 Providers are instances, not globals

External device integrations implement VxBackendProvider from volvoxai_backend.h. The descriptor
creates an explicit provider-runtime instance, compiled instance, and context instance:

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

VxStatus registration =
    vx_runtime_register_provider(runtime, &provider, &report);
~~~

Context execution writes all declared outputs through VxBackendOutputSink. The sink copies before
write returns, so providers cannot lend mutable device or scratch memory to a result. Provider names
and descriptors are copied at registration; callback code and user_data must remain valid for every
handle created from that provider.

## 9.7 What happens inside compilation

🌱 Compilation is preparation: inspect the whole Graph, choose the device route, reserve memory, and
turn expensive repeated setup into one-time work.

🔬 A provider may:

- validate operator, dtype, shape, layout, and quantization coverage;
- fuse safe adjacent patterns;
- prepack constant weights;
- compile GPU pipelines or device graphs;
- plan scratch-buffer lifetimes;
- record route and device evidence in its compile report.

Unsupported required work fails compilation. If operator fallback is allowed, the compiled route
records those boundaries before execution. No node is silently skipped.

CPU execution uses portable kernels plus runtime-selected SIMD microkernels when the CPU and OS
support them. GPU integrations resolve driver libraries at runtime, which keeps the executable free
of mandatory vendor SDK linkage.

## 9.8 Memory planning and shaders

Temporary tensors usually have short, non-overlapping lives. A compilation-time lifetime plan can
reuse one arena region after its previous value is dead:

~~~text
time ───────────────────────────────────────────────▶
input       [==============]
hidden A           [========]
hidden B                    [==========]
output                                [==========]

arena slot 0 [ input ][ reuse for hidden B ]
arena slot 1         [ hidden A ][ reuse for output ]
~~~

This reduces allocation overhead and peak memory without changing Graph semantics. Contexts own
their arenas, so concurrent requests stay isolated.

GPU shader sources are generated from canonical templates. Generated outputs and embedded byte
arrays are build products, not editing surfaces. During development, VOLVOXAI_SHADER_DIR may point
to external shader files; the runtime logs once only when that override is actually used.

## 9.9 Decode and task applications

The native runtime deliberately assigns no meaning to tensor names. Tokenization, image decoding,
prompt formatting, sampling, detection postprocessing, and robot I/O belong to applications.

The public native execution API has no separate prefix/row entry points. An application binds named
inputs, executes a context, and reads declared results. A provider may keep device state inside its
context when its compiled route requires it. In JavaScript, the corresponding model-neutral decode
surface is ExecutionContext.decode.seed(), step(), and reset().

The opt-in task application demonstrates image preprocessing, classification,
detection, and model-neutral decode operations:

~~~bash
make -C examples native_task_cli

examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=photo.png --boxes boxes --scores scores --max-det 20
~~~

## 9.10 Build profiles

~~~bash
make build_native

./native/volvoxai --help
./native/volvoxai-full --help
~~~

Both programs provide the model-neutral run command. Only the full program provides train:

~~~bash
./native/volvoxai-full train models/my_model \
  --input input=batch.f32 \
  --targets targets.i32 \
  --logits logits \
  --trainable classifier.weight \
  --microbatches 10 \
  --accumulation-steps 2 \
  --optimizer adamw \
  --output-weights trained.safetensors
~~~

The command uses the full-only opaque VxTrainer lifecycle. Each Trainer owns
private inputs, gradients, optimizer slots, accumulation, RNG, and working
weights. Only an explicit conflict-checked commit publishes a Model revision.

Backend source composition is controlled at build time by the Vulkan, OpenGL, CUDA, Metal, and
NNAPI options. Requiring a provider that was not compiled or cannot initialize returns a backend
error; it does not switch to CPU.

## 9.11 What you just learned

> 🌱 **Idea recap.** Native inference is not one process-wide machine. It is a tree of owned objects:
> shared runtime/model/compilation state, private execution contexts, and immutable results.

- The native API is the opaque vx_* lifecycle.
- graph.json uses the exact volvox-graph/v1 discriminator.
- Provider and operator policy is resolved during compilation.
- Contexts isolate mutable request and device state.
- Results own stable named-output snapshots.
- External devices implement VxBackendProvider instances.
- Full-profile training uses a private opaque VxTrainer and atomic revision publication.
- The inference and full profiles remain physically separate.

**Next:** [Chapter 9C — The CUDA Backend →](09c-cuda-backend.md)
