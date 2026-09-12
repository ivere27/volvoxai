# Chapter 9 — The Native Runtime

*Read the badges that match you: 🌱 **Idea** (anyone, no code) · 🔧 **Build** (a little code) · 🔬
**Deep** (runtime developers). New here? Follow just the 🌱 sections.*

*Goal: understand how the same VolvoxAI package runs in a freestanding C program. We will follow
the generated proto API, see how compilation fixes a provider route, and learn why internal
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
  "format": "volvox-graph/v1",
  "dimensions": {}
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

## 9.2 Generated IDs outside, ownership tree inside

The public C surface is generated from `proto/volvoxai.proto`: `volvoxai_ffi.h` contains service
entry points and `volvoxai_lite.h` contains message codecs. Applications receive integer IDs in
generated messages; they do not receive engine pointers or a handwritten lifecycle API.

Behind the generated handlers, the native engine keeps this **internal** ownership tree:

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

Each internal child retains the parent state it needs. A public `Release*` operation idempotently
retires only that public ID. Accepted operations and descendants keep their internal references;
physical object close and drain begin only after the last reference disappears. Releasing a public
parent ID therefore does not invalidate a retained child or result.

Generated callers see the same ownership through IDs:

~~~text
runtime_id → model_id → compiled_model_id → context_id → result_id
~~~

## 9.3 Create, load, and compile

🔧 Include the generated FFI and lite codec. A request with optional, repeated, or oneof fields is
encoded and sent through its `_pb` entry point:

~~~c
#include "volvoxai_ffi.h"
#include "volvoxai_lite.h"

VolvoxaiV1CreateRuntimeRequest request;
VolvoxaiV1RuntimeHandle handle;
uint8_t *encoded = NULL, *response = NULL;
size_t encoded_len = 0;
int32_t response_len = 0;
int64_t runtime_id = 0;

volvoxai_v1_create_runtime_request_init(&request);
/* execution_mode absent means DIRECT. */
volvoxai_v1_create_runtime_request_encode(&request, &encoded, &encoded_len);
response = vx_inference_create_runtime_pb(
    encoded, (int32_t)encoded_len, &response_len);
synurang_lite_default_allocator()->deallocate(
    synurang_lite_default_allocator()->context, encoded);
volvoxai_v1_create_runtime_request_free(&request);

volvoxai_v1_runtime_handle_init(&handle);
if (response && volvoxai_v1_runtime_handle_decode(
        &handle, response, (size_t)response_len) == SYNURANG_LITE_OK &&
    handle.field_report &&
    handle.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK) {
    runtime_id = handle.field_runtime_id;
}
vx_inference_free(response);
volvoxai_v1_runtime_handle_free(&handle);
~~~

`LoadModelRequest` carries `runtime_id`, `graph_path`, and repeated `weight_paths`; its generated
`ModelHandle` returns `model_id`. `CompileModelRequest` carries that ID and a typed `BackendPolicy`;
its `CompiledModelHandle` returns `compiled_model_id`. An empty policy prefers CPU. `REQUIRE`
accepts exactly one backend, while `PREFER` tries its ordered candidates. Operator fallback is a
separate generated enum. Selection ends at compile time: execution never retries on another
provider.

🔬 Every generated response embeds an `OperationReport`. Programs branch on its typed status and
structured route evidence; its message is diagnostic text. Transport/codec failure is the separate
case where an FFI call returns `NULL`. The complete load/compile/release sequence is in
`examples/c_api_client_raw.c`.

## 9.4 Context execution

`RunRequest` is the simple stateless path: it carries only `compiled_model_id` and a complete list
of generated `Tensor` messages. The engine derives the retained Runtime; the caller cannot supply a
competing runtime ID.

For decode or reusable mutable state, call generated `CreateExecutionContext`. Its
`ExecutionContextHandle` carries both `context_id` and the declared input specs, so there is no
second input-list operation. One compiled model may create many contexts; their inputs, scratch,
decode state, and in-flight work never alias. `Execute`, `ExecutePrefix`, `DecodePrefill`, and
`DecodeStep` accept that context ID plus explicit tensors. `ResetDecode` clears decode state.
`SelectAdapter` pins the supplied exact published revision, or returns to the base model when the
revision is absent. `RebindAdapter` adopts the model's single most recently published revision.
Adapter operations mutate only the named context.

Execution rejects a tensor name, dtype, shape, or byte count that disagrees with the Graph.

Execution publishes every declared output exactly once. It does not expose mutable intermediate
tensors or a borrowed workspace pointer.

## 9.5 Stable results and readback

`Run` and context execution return an `ExecutionResultHandle` containing `result_id` and
`execution_id`. The result remains readable after later calls and after its context is released.
`GetResult` lists the stable declared outputs. `ReadOutput(result_id, name)` returns exact inline
bytes; its optional `BufferView into` instead requests a copy into caller-owned memory and reports
the required size when capacity is too small.

Release IDs explicitly with generated `ReleaseResult`, `ReleaseExecutionContext`,
`ReleaseCompiledModel`, `ReleaseModel`, and `ReleaseRuntime`. There are no separate Close RPCs.
Each release idempotently retires that public ID; accepted work and descendants keep references,
and physical close/drain waits for the last reference. A result's snapshot stays independent until
`ReleaseResult`.

## 9.6 Providers are instances, not globals

Engine embedders may implement the `VxBackendProvider` host-composition SPI from
`volvoxai_backend.h`. It is not an application lifecycle API and it does not add proto operations.
The descriptor creates an explicit provider-runtime instance, compiled instance, and context
instance behind the generated handlers:

~~~c
VxBackendProvider provider = {
    .struct_size = sizeof(VxBackendProvider),
    .abi_version = VX_BACKEND_ABI_VERSION,
    .name = "my-npu",
    .user_data = &driver,
    .shape_domain = VX_BACKEND_SHAPE_DOMAIN_CAPABILITY_INIT,
    .runtime_create = provider_runtime_create,
    .runtime_destroy = provider_runtime_destroy,
    .compile = provider_compile,
    .compiled_destroy = provider_compiled_destroy,
    .context_create = provider_context_create,
    .context_execute = provider_context_execute,
    .context_close = provider_context_close,
    .context_destroy = provider_context_destroy,
    .exact_contract_marker = VX_BACKEND_PROVIDER_EXACT_CONTRACT_MARKER,
    .exact_contract_extent = sizeof(VxBackendProvider),
};

provider.shape_domain.support = VX_BACKEND_SHAPE_DOMAIN_FULL;
~~~

Context execution writes all declared outputs through VxBackendOutputSink. The sink copies before
write returns, so providers cannot lend mutable device or scratch memory to a result. The native
composition root installs descriptors while building the host; ordinary applications only use the
generated service surface. Callback code and `user_data` must remain valid for every internal owner
created from that provider.

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

The lifetime plan is compile-time because it follows the topology; the region *sizes* are not,
because a symbolic dimension has no size until a request binds one. A context therefore sizes its
arena from the current shape binding and grows it geometrically when a larger legal binding arrives.
Growth is transactional: a binding that cannot fit the admitted budget returns generated
`NATIVE_STATUS_OUT_OF_MEMORY` with candidate state rolled back, leaving the previous binding usable. The
engine tracks dynamic_arena_capacity_bytes, dynamic_arena_high_water_bytes, and
dynamic_arena_grow_count per context (native/src/runtime/runtime_state.h); the high-water figure is
what a deployment should size its budget from, rather than a worst-case guess. This is the native
form of the browser behavior in Chapter 8 §8.6.

GPU shader sources are generated from canonical templates. Generated outputs and embedded byte
arrays are build products, not editing surfaces. During development, VOLVOXAI_SHADER_DIR may point
to external shader files; the runtime logs once only when that override is actually used.

## 9.9 Decode and task applications

The native runtime deliberately assigns no meaning to tensor names. Tokenization, image decoding,
prompt formatting, sampling, detection postprocessing, and robot I/O belong to applications.

The generated native API exposes the same model-neutral operations as every projection:
`ExecutePrefix`, `DecodePrefill`, `DecodeStep`, and `ResetDecode`. An application sends named
shape-bearing tensors and reads declared results. A provider may keep device state inside its
internal context when its compiled route requires it. JavaScript applications call those same
operations through `VxInferenceServiceClient` and `pb` messages.

The opt-in task application demonstrates image preprocessing, classification,
detection, and model-neutral decode operations. It and the shipped
`native/volvoxai` CLI are generated FFI + lite consumers; neither has a second
engine lifecycle. `examples/c_api_client_raw.c` shows the smaller embedding
pattern without task policy.

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

The fixed command is itself a generated public-API client. It exercises the
same full-only Training contract: each internal Trainer owner keeps private
gradients, optimizer slots, accumulation, RNG, and working weights, and commit
publishes a conflict-checked Model revision.

Backend source composition is controlled at build time by the Vulkan, OpenGL, CUDA, and Metal
options. Requiring a provider that was not compiled or cannot initialize returns a backend
error; it does not switch to CPU.

## 9.11 What you just learned

> 🌱 **Idea recap.** Native inference is not one process-wide machine. It is a tree of owned objects:
> shared runtime/model/compilation state, private execution contexts, and immutable results.

- The native application API is generated FFI + lite messages from `proto/volvoxai.proto`.
- graph.json uses the exact volvox-graph/v1 discriminator.
- Provider and operator policy is resolved during compilation.
- Contexts isolate mutable request and device state.
- Results own stable named-output snapshots.
- External devices implement VxBackendProvider instances.
- Full-profile training uses generated Trainer IDs; the engine keeps private internal state and
  publishes revisions atomically through `CommitTrainer`.
- The inference and full profiles remain physically separate.

**Next:** [Chapter 9C — The CUDA Backend →](09c-cuda-backend.md)
