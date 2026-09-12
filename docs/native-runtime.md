# Native runtime

Use the native runtime when your application needs to run on a desktop, phone,
or robot without a JavaScript host. It loads the same `graph.json` and
SafeTensors packages as the browser runtime. The command-line runner accepts
named raw tensors; C and Python applications can embed the engine and reuse
models and execution contexts.

Two profiles are available: `native/volvoxai` for inference and
`native/volvoxai-full` for inference, training, and PTQ. CPU is the default;
optional GPU backends are selected at build time and requested when compiling
a model. [Model format](model-format.md) describes the package and its bounded
input shapes.

## Build

Native compilation uses the tracked Synurang call runtime under
`native/third_party/synurang/`. It does not require the generator, Cargo or an
adjacent checkout. Regeneration uses the verified GitHub v0.8.0 release described in
[runtime/README.md](../runtime/README.md).

```sh
make build_native_profiles
make build_native_libraries
```

The fixed executables are `native/volvoxai` and `native/volvoxai-full`. Shared
and static libraries, generated C headers, and the optional Python module-host
shim are development build products; an installable SDK is a separate TODO.
Inference includes Platform, Text, Planning, Inference and Scheduler. Full adds
Training and Quantization; their implementation and codecs are absent from
inference.

## Run a model from the command line

```sh
./native/volvoxai run --help
./native/volvoxai-full train --help
```

Both executables provide `run`, help, and version information. Full also
provides `train`. For a TinyStories package and six token IDs and positions:

```sh
./native/volvoxai run models/tinystories_1m \
  --input 'tokens[1,6]=build/quickstart/tokens.i32' \
  --input 'positions[1,6]=build/quickstart/positions.i32' \
  --output logits=build/quickstart/logits.f32 \
  --threads 2
```

The [quickstart](quickstart.md#4-run-tinystories-from-native-c) creates those
input files. Explicit shapes are required for dynamic axes. For static inputs,
`--input name=file` uses the declared shape. Storage suffixes are `.f32`, `.i32`,
`.i8`, and `.u8`, and file byte counts must match the tensor exactly.

`--weights` adds a weight shard and is repeatable. `--output name=file` writes
that named output; ordinary output contains its complete logical tensor.
`--row <index>` selects one row of an F32 output when that is what the consumer
needs. `--report-json report.json` records structured success/failure evidence;
`--debug` enables diagnostics.

The raw runner expects already prepared tensors. Image resizing, normalization,
tokenization policy, and task postprocessing belong to the caller. The
[task CLI](../examples/native_task_cli/README.md) demonstrates image decoding,
classification, detection, and decode sessions. [Models](models.md) describes
how to obtain matching packages.

## CPU threads and native GPUs

Use `--threads <n>` on the CLI, or `CreateRuntimeRequest.cpuThreads` in a
JavaScript projection / `cpu_threads` in the schema, to set the CPU worker count.
Without an explicit count, `VOLVOXAI_THREADS` can override the native pool's
default. Counts are bounded by the compiled pool limit. Linux x86 detection uses
the process's allowed CPUs and physical-core topology; benchmark the workload
under its actual affinity and deployment limits.

An explicit native GPU flag requires that backend:

```sh
./native/volvoxai run models/tinystories_1m \
  --input 'tokens[1,6]=build/quickstart/tokens.i32' \
  --input 'positions[1,6]=build/quickstart/positions.i32' \
  --output logits=build/quickstart/logits.f32 --vulkan
```

Available flags are `--cpu`, `--vulkan`, `--opengl`, `--metal`, and `--cuda`.
A requested but unavailable or unsupported backend reports an error. The
runtime does not replay a failed execution on CPU.

| Backend | Build option | Notes |
| --- | --- | --- |
| CPU | Included | Portable kernels and runtime ISA selection |
| Vulkan | `VOLVOXAI_ENABLE_VULKAN` | Requires a usable Vulkan device/driver |
| OpenGL / OpenGL ES | `VOLVOXAI_ENABLE_OPENGL` | Platform graphics context and compute support |
| CUDA | `VOLVOXAI_ENABLE_CUDA` | Opt-in NVIDIA backend; see [CUDA](cuda.md) |
| Metal | `VOLVOXAI_ENABLE_METAL` | macOS/Apple GPU integration |

Compilation checks the graph's operators, dtypes, shapes, and resource needs
against the selected backend. An entry in a kernel inventory alone does not
prove that a complete model is admitted; consult [operator support](operation_list.md).

## Train selected weights

For a package with F32 input `input`, logits `logits`, and classifier weights:

```sh
./native/volvoxai-full train models/my_model \
  --input input=batch.f32 \
  --targets targets.i32 --logits logits \
  --trainable classifier.weight --trainable classifier.bias \
  --microbatches 10 --accumulation-steps 2 \
  --optimizer adamw --learning-rate 0.001 \
  --output-weights trained.safetensors
```

Substitute names and shapes from your model. Targets are raw I32 class IDs.
`--trainable`, `--weights`, and `--output-weights` are repeatable; output shard
count must match the loaded weight shards. The command trains privately,
flushes the final accumulation window, commits once, and exports the weights.
The CLI repeats the supplied batch; a dataset training loop belongs in an
application using the [training guide](model_builder_training.md).

## Build on macOS or for Android

Direct CMake builds use your platform toolchain and the dependencies required
by the selected backends. On macOS, enable Metal with:

```sh
cmake -S . -B build/mac -DCMAKE_C_COMPILER=clang \
  -DVOLVOXAI_ENABLE_METAL=ON
cmake --build build/mac --target volvoxai volvoxai-full
```

For Android, set `ANDROID_NDK` to your installed NDK directory, then configure
with its toolchain:

```sh
cmake -S . -B build/android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29
cmake --build build/android --target volvoxai volvoxai-full
adb push native/volvoxai /data/local/tmp/volvoxai
```

Builds publish the fixed executable names under `native/`; use separate
checkouts if you need to retain binaries for several platforms. Copy the model
and raw inputs to the device and run the same CLI there. Shaders are embedded.
GPU support still needs qualification on the target hardware; a successful
cross-compile establishes only the build.

## Embed the engine

Keep a Runtime and compiled model alive across requests to amortize model
loading and compilation. Use one ExecutionContext per independent mutable
session, and release each result after consuming its outputs. Native libraries
and the command-line runner use the same operations.

Application headers are generated under `runtime/generated/c/inference/` for
inference and `runtime/generated/c/` for full, using the include names
`volvoxai_ffi.h` and `volvoxai_lite.h`. The
[raw C client](../examples/c_api_client_raw.c) demonstrates a complete
load/compile/execute/read lifecycle. [Runtime integration](../runtime/README.md)
explains the C and Python host layouts. Backend implementers use the separate
[composition SPI](backend-sdk.md).

## Module calls

Load `Synurang_GetApi`, validate its ABI version/size, and create an instance
with MANUAL execution and a wakeup callback. Each generated RPC is addressed by
its full path from `proto/volvoxai.proto`:

1. Encode its generated request message.
2. Open the call, send its bytes, then half-close.
3. Poll bounded turns when notified and receive the message plus terminal status.
4. Free returned buffers with `api->free_buffer` and release the call.
5. Decode the response and inspect its typed `OperationReport`.

ABI entries for an instance are serialized. A wakeup callback only posts later
work; it never reenters the API. `destroy` can return `PENDING`, in which case
keep the module loaded, poll on notifications, and retry until `OK`.

[examples/c_api_client_raw.c](../examples/c_api_client_raw.c) is a complete
load/compile/execute consumer. The CLI's small blocking transport helper uses a
condition variable and the same module ABI; it contains no VolvoxAI operations
or handwritten per-method wrappers.

## Handle lifecycle

The internal ownership represented by generated IDs is:

```text
Runtime
  Model
    CompiledModel
      ExecutionContext
        Result
  scheduled Request -> one transferable Result
```

Full builds add Trainer and PTQPlan handles. Every child retains the state it
needs. A generated Release idempotently retires its public ID; accepted work
and descendants retain the object until their references leave, and only then
can physical close complete. Release children before parents for prompt
cleanup. A Result snapshot remains valid until `ReleaseResult` and consumes
its Runtime result-budget lease.

Create responses already carry useful discovery data: the execution-context,
trainer, and PTQ-plan handles include their declared inputs. Compile responses
carry the selected report. Separate close, report-fetch, and input-list RPCs are
intentionally absent.

## Direct and scheduled execution

`CreateRuntimeRequest.execution_mode` is optional. Absence selects DIRECT.
DIRECT `Run` performs one logical request without allocating a scheduler queue,
timer, worker, or request handle. It identifies only `compiled_model_id`; the
engine derives the retained Runtime that owns the compiled model.

Set `execution_mode` explicitly to `EXECUTION_MODE_SCHEDULED` to use
`VxSchedulerService`. `Submit` also identifies only `compiled_model_id`. It
returns a Request handle for poll, wait, cancellation, one-shot result transfer,
and release. Scheduler budgets, priorities, deadlines, freshness, and stream
keys are typed request fields. An absent optional field takes its documented
default; its zero value is not automatically presence.

The generated `REQUEST_STATE_UNSPECIFIED` is the protobuf zero value. Real
request observations report a concrete queued, running, completed, cancelled,
or failed state.

The native scheduler and browser scheduler share the portable C ordering,
deadline, freshness, and admission policy. Coalescing still requires the exact
typed independent-batch proof and provider attestation; an authored B axis by
itself is insufficient.

## Models and backends

`LoadModelRequest` names `graph.json` (or a named `*.graph.json`) and ordered
SafeTensors shards. It may select immutable weight-bank residency. The loader
validates the closed graph schema, SafeTensors descriptors, bounded shape
domain, and canonical portable-C graph contracts before publication.

An empty `BackendPolicy` selects the native default portable C CPU provider.
PREFER tries an ordered backend list during compilation. REQUIRE accepts one
backend and should normally pair with `OPERATOR_FALLBACK_FORBID`. Provider
selection ends at compile time; execution never retries another backend.
Populated lists contain at most 16 unique names. Each name is 1..63 ASCII
characters: a lowercase letter followed by lowercase letters, digits, `.`,
`_`, or `-`.

Optional built-in native providers include Vulkan, OpenGL/OpenGL ES, CUDA, and
Metal according to platform/build support. They extend the same compiled-model
and context contract and must prove their complete admitted shape/resource
domain. Unsupported or unavailable routes return typed reports.

External providers implement `native/include/volvoxai_backend.h` and are
registered by deployment composition. The SPI owns provider, compiled, and
context instances plus immutable resource leases. Applications still select an
already-composed provider only by its schema backend name.

## Full profile

`VxTrainingService` owns private Trainer weights, gradients, optimizer slots,
accumulation, and RNG. `TrainStep` does not mutate the source Model.
`CommitTrainer` publishes a successor revision on the retained Model handle;
`RollbackTrainer` restores the last committed baseline. Native full supports
filesystem weight export.

`VxQuantizationService` authors a supported F32 template, creates a plan pinned
to one Model revision, calibrates complete named samples, exposes typed
coverage/parameters, and writes a new graph/SafeTensors package. PTQ never
mutates the source package or an already compiled revision.

## Shaders and device code

Release executables use embedded shader packs. `VOLVOXAI_SHADER_DIR` is a
development override for generated SPIR-V, GLSL/GLES, and Metal assets. The
runtime logs once only when that external override is actually used. WGSL and
CUDA source are authoritative; generated packs, PTX arrays, and embedded bytes
must not be edited by hand.

## Synurang and exports

The libraries export `Synurang_GetApi`, generated message codecs and the provider
registration SPI. Engine lifecycle functions, old flattened calls and legacy
stream entry points remain hidden. Both profiles use the same module ABI;
physical source composition determines which RPCs and codecs are present.

The pinned Synurang runtime includes generic stream lifecycle support. The
current VolvoxAI schema declares unary methods only; a streaming application
operation requires a proto change, regeneration, and profile-correct handlers.

## Verification

```bash
make api_conformance
make build_native_profiles
make build_native_libraries
make test_native
python3 -m pip install -r python/requirements-test.txt
make test_python
```

The full release gate also builds both web profiles and sidecars:

```bash
make verify_release
```

See [backend SDK](backend-sdk.md), [model format](model-format.md),
[scheduling](scheduling-and-dynamic-batching-design.md), and
[profiling](profiling.md) for the focused contracts.
