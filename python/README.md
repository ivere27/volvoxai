# VolvoxAI for Python

Use VolvoxAI from Python to run models on CPUs and native GPUs, tokenize text,
plan model execution, schedule work, train or adapt models, and perform
post-training quantization. ONNX conversion and command-line PTQ are included.
Python calls the same C engine and generated protobuf API as native applications
and the JavaScript/WASM package.

`InferenceSession` and `AsyncInferenceSession` load a model, reuse its compiled
execution context and accept and return NumPy arrays. `quantize` consumes
calibration batches directly; `TrainingSession` supports training, saving and
checkpoint resumption. Native libraries, CPU threads and result lifetime are
managed automatically. Failures raise Python exceptions at the call site.

The package includes the native engine library, its shared-library loader, all
eight generated synchronous and asynchronous service clients, the protobuf
messages, and the pinned Synurang runtime. It also bundles the native inference
runner used by model tooling. No source checkout, compiler, CUDA toolkit or
Docker installation is needed to use an installed wheel.

## Installation and supported environments

The initial binary distribution targets **Linux x86_64 with glibc 2.28 or later**.
The wheel uses the C ABI through ctypes and a CPython stable-ABI DLPack capsule
adapter. One `cp310-abi3` wheel serves CPython 3.10–3.14.
CPU, CUDA, Vulkan and OpenGL backends are compiled in. GPU execution requires
the corresponding host device and driver; CPU execution works without them.
CUDA kernels use a compute-capability 7.5 PTX baseline. Windows, macOS and Linux
ARM64 wheels are not part of this release.

After this version is published to PyPI:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install volvoxai==0.5.0
```

To install the local build before publishing:

```sh
python -m pip install dist/python/0.5.0/*.whl
volvoxai --version
```

NumPy, ONNX and SafeTensors are installed as dependencies. The optional
`volvoxai[tflite]` extra installs the TensorFlow Lite importer dependencies.
Hugging Face downloads can use the separately installed `huggingface_hub` tool.
PyTorch is an optional DLPack consumer/producer, not a dependency of
the wheel. No separate DLPack library is installed.

## Run a model

Convert an ONNX model into a VolvoxAI package. Bind unresolved dimensions with
`--input-shape` or `--dimension-bound`; `volvoxai export --help` describes all
supported conversion options.

```sh
volvoxai export --model model.onnx --out model/model.safetensors
```

The resulting `graph.json` and `model.safetensors` are also usable by the npm
and native distributions. Conversion does not change application preprocessing.

For a model with one float32 input of shape `[1, 3]`:

```python
import numpy as np
import volvoxai as vx

values = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
with vx.InferenceSession("model") as session:
    outputs = session.run(values)
    for name, array in outputs.items():
        print(name, array.shape, array.dtype, array)
```

`"model"` is a directory path. FP32, INT8 and mixed precision are read from the
graph and weights; there is no precision argument. A quantized model can still
require float32 application inputs. Preserve the input dtype declared by the
model instead of casting inputs to INT8 because its weights are quantized.

The session finds `graph.json` (or a named `*.graph.json`) and its SafeTensors
weights. You can pass the directory, graph file or corresponding weight file.
Omit the model path to discover a single package in the current directory or
its immediate child directories:

```python
with vx.InferenceSession() as session:
    outputs = session.run(values)
```

If multiple models exist, such as separate encoder/decoder packages or FP32 and
INT8 exports, pass the intended model path. The SDK raises an ambiguity error
instead of choosing one. Named graphs prefer their matching weight filename.
If a graph directory contains multiple weight files, select its files or shards
explicitly with `weights=["path/to/shard1.safetensors", "path/to/shard2.safetensors"]`.
These paths are relative to the working directory. ONNX conversion remains the
explicit export step above.

For multiple inputs, use names from `session.inputs`, a tuple of immutable
Python `TensorSpec` objects. `session.outputs` exposes the output contracts. The
converter may assign canonical input names such as `input0`:

```python
with vx.InferenceSession("model") as session:
    print(session.inputs, session.outputs)
    outputs = session.run({session.inputs[0].name: values})
    # Repeated calls reuse the model and execution context.
    selected = session.run(values, output_names=[session.outputs[0].name])
```

Inspect names, shapes and dtype strings without interpreting protobuf enums:

```python
with vx.InferenceSession("model") as session:
    for spec in session.inputs:
        print(spec.name, spec.shape, spec.dtype)  # e.g. input0 (1, 3) float32
        for symbol, bounds in spec.constraints.items():
            print(symbol, bounds.min, bounds.max, bounds.multiple_of)
```

A dynamic contract can have `shape=(1, "S", 320)` with
`constraints["S"] == vx.Dimension(min=1, max=256, multiple_of=1)`.
These describe the legal domain; each returned array has its actual integer
shape. Metadata, its shape tuple and dimension constraints are read-only.
Storage formats without a NumPy representation still have dtype names such
as `bfloat16` or `float8_e5m2` in metadata.

`run()` always returns a dictionary of output names to owned, writable NumPy
arrays. Arrays remain valid after later runs and after session close. Dtypes
and concrete shapes are preserved; C validates each complete input batch.
Strided and non-native-byte-order input arrays are supported. Packed output
formats with no supported NumPy dtype raise `TypeError`; their bytes remain
available through the generated `ReadOutput` API.

Contiguous, aligned little-endian inputs are borrowed until `run()` returns;
do not modify them from another thread during the call. Other layouts are
normalized without changing precision or scalar rank. Native input buffers
and direct writes into output arrays avoid protobuf tensor-payload copies.
C still owns execution storage and output snapshots, so this is not a promise
of zero copies throughout the engine. Static output shapes need no extra
metadata query; dynamic outputs query the completed result's concrete shapes.

The default backend is CPU. CPU thread count is selected by the C engine;
on Linux x86_64 this accounts for CPU affinity and physical cores.
`VOLVOXAI_THREADS` or an explicit positive `cpu_threads` can override it.
`None` and `0` select automatic threading. GPU execution requires only a backend
name, such as `"cuda"`, `"vulkan"` or `"opengl"`:

```python
with vx.InferenceSession("model", backend="cuda") as session:
    outputs = session.run(values)
    print(session.backend)  # The actual compiled backend.
```

A requested GPU is required and cannot silently fall back to CPU.
`session.report` retains compilation and route evidence. The session waits for
pending execution, reads only requested outputs and releases each native result.
Calls on one session are serialized. Close it with a `with` block or `close()`;
this retires its host, model and execution context. Use generated services for
scheduling or incremental decoding.

## Keep tensors on the GPU and share them through DLPack

Use `run()` when you want NumPy outputs. Use `run_tensors()` when outputs should
remain in native memory, particularly between GPU inference calls. It accepts
NumPy arrays, VolvoxAI tensors, and compatible DLPack producers. The default
backend is CPU. Choose `cuda`, `vulkan`, `opengl` or `metal` to retain GPU outputs
on that backend, subject to the platform/build support below.

For a model package with one float32 input of shape `[1, 3]` and output `logits`:

```python
import torch
import volvoxai as vx

x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device="cuda")

with vx.InferenceSession("model", backend="cuda") as session:
    with session.run_tensors(x) as outputs:
        logits = outputs["logits"]
        print(logits.shape, logits.dtype, logits.device)
        # For example: (1, 10), float32, cuda:0

        scores = torch.from_dlpack(logits)  # Shares the CUDA output buffer.
        predicted_class = scores.argmax(dim=-1)
        print(predicted_class.cpu().tolist())  # Only the selected IDs go to CPU.

# The external view still owns its buffer after Tensor and session close.
print(scores.shape)
del scores
```

Install a PyTorch distribution suitable for your CUDA driver separately when
using this integration. Native C owns VolvoxAI tensor storage and device
lifetime. DLPack does not automatically translate device allocations between
different GPU APIs. See the
[DLPack Python protocol](https://dmlc.github.io/dlpack/latest/python_spec.html).

| Backend | Pass a VolvoxAI tensor to another session | External sharing |
| --- | --- | --- |
| CPU | Yes | NumPy and CPU DLPack |
| CUDA | Yes, on the same device | CUDA DLPack, including PyTorch |
| Vulkan | Yes, on the same VkDevice | Exported native VkBuffer; imports currently require a module-owned buffer |
| OpenGL | Yes, in the same EGL context | Exported native buffer name; imports currently require a module-owned buffer |
| Metal | Adapter implemented; macOS execution not yet qualified | Same-device MTLBuffer and compatible kDLMetal DLPack; four-byte aligned offsets and byte extents |

Vulkan/OpenGL use the native buffer descriptor, so `Tensor.__dlpack__()` raises
`BufferError` for those backends. Pass retained tensors directly between
VolvoxAI models. Graphics-API producers must finish writes before a call.
Metal DLPack objects must carry an actual `id<MTLBuffer>` as required by the
[DLPack C structure](https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h);
a framework's GPU support alone does not establish compatibility.
The initial Metal adapter uses buffer blits, whose macOS
[alignment requirements](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceoffset:to:destinationoffset:size:))
require four-byte offsets and byte lengths.

You can also connect two compatible models without installing a GPU array
library. For example, assume an encoder returns `memory` and a decoder accepts
that tensor plus an `int32` `tokens` input:

```python
import numpy as np
import volvoxai as vx

with vx.InferenceSession("encoder", backend="cuda") as encoder, \
     vx.InferenceSession("decoder", backend="cuda") as decoder:
    image = np.load("encoder_input.npy")  # Exact encoder input dtype/shape.
    with encoder.run_tensors(image, output_names=["memory"])["memory"] as memory:
        result = decoder.run_tensors({
            "memory": memory,
            "tokens": np.array([[1]], dtype=np.int32),
        }, output_names=["logits"])
        with result["logits"] as logits:
            host_logits = logits.numpy()  # Explicit, independent host copy.
            print(host_logits.shape, host_logits.dtype)
```

Use the same backend for both sessions; the example also works with `vulkan`
or `opengl` and their drivers. A CPU input uploads once, the intermediate
`memory` stays on the GPU, and `numpy()` performs the final explicit download.

Choose the actual input/output names and shapes from each model's `inputs` and
`outputs` metadata. Each call returns `TensorOutputs`, a dictionary of `Tensor`
objects. Use `with session.run_tensors(...) as outputs:` or `outputs.close()`
to release a group in one native call. Individual tensor context managers and
`tensor.close()` also work. Existing DLPack views retain their own leases when
the group closes. CPU tensors can also be
shared with `np.from_dlpack(tensor)`. `tensor.numpy()` always makes an independent
host copy. `Tensor` has no implicit NumPy conversion that downloads a GPU buffer.

For repeated execution, keep unchanged inputs and feedback state inside the
session. Supply a complete batch for the first call; later calls can combine
new inputs, `reuse_inputs`, and `feedback`. For example, a decoder whose cache
input is `past` and output is `present` can use:

```python
with decoder.run_tensors(initial_inputs, output_names=["logits"]) as outputs:
    token = int(outputs["logits"].numpy().argmax())

with decoder.run_tensors(
    {"token": np.array([[token]], dtype=np.int32)},
    reuse_inputs=["memory"],
    feedback={"past": "present"},
    output_names=["logits"],
) as outputs:
    token = int(outputs["logits"].numpy().argmax())
```

Use your model's actual names and supply any additional control inputs.
Together, the three input sources must bind each input exactly once. Feedback
uses the previous internal graph output, including outputs that were not
returned. Editing an exported tensor cannot change that internal state.
`output_names=[]` executes without exporting snapshots. Output selection does
not skip graph computation. A complete input batch starts a new sequence;
`run()` invalidates the previous `run_tensors()` reference state.

CUDA, Vulkan and OpenGL reuse unchanged input buffers in their reserved
domains. Feedback copies directly from the previous internal output to its
next input, eliminating the intermediate exported snapshot. Aliased sources
and movable layouts use private snapshots to preserve simultaneous binding
semantics. CPU and Metal use this safe snapshot path. Device-produced indices
or routing values that require host validation must still be supplied as host
arrays. Validation failures preserve prior state; an execution failure after
input commit requires a new complete input batch.

The memory and execution contract is explicit:

- Export through DLPack shares output storage. Inputs are copied into the
  engine's execution arena and outputs are copied into independent snapshots;
  GPU inputs and snapshots use device-to-device copies. The complete engine
  execution is therefore not a zero-copy operation.
- Released allocations are reused by the session, bounded to 64 idle buffers
  and 64 MiB. Session close frees idle storage; live outputs are not reused.
  CUDA/Vulkan/OpenGL group input and output copies before waiting for completion.
- Calls wait for GPU completion. Input producers establish ordering on the
  CUDA legacy default stream, including when PyTorch uses other streams.
  Vulkan, OpenGL and Metal producers complete writes before the call; raw
  native handles do not encode cross-queue fences or context switching.
  Output snapshots are ready before export. This first implementation favors
  a predictable synchronous contract; it does not overlap inference calls.
- Outputs survive subsequent calls and session close. External NumPy/PyTorch
  views retain their storage after the VolvoxAI wrapper closes. Those
  DLPack consumers hold exclusive access; release that external lease before
  reusing the original VolvoxAI tensor. Passing the modified PyTorch object itself
  allows its producer protocol to establish the required stream ordering.
- DLPack inputs need a supported exact dtype, dense contiguous layout and, for
  CUDA inputs, the session's CUDA device and primary context. Strided views,
  foreign devices, managed allocations and unsupported DLPack devices are rejected
  instead of being implicitly moved or converted. Make an explicit contiguous
  copy in the producing library when needed.
- Index and route inputs that require host-side value validation must remain
  NumPy arrays. This includes some embedding/gather indices. Large encoder
  activations can remain on the GPU while those small control inputs stay on CPU.
- DLPack shares tensor data, not autograd history. Detach PyTorch tensors with
  gradients before inference. Packed low-bit types are outside this initial
  dense tensor interface. AsyncInferenceSession continues to provide the
  asynchronous NumPy workflow; `run_tensors()` currently belongs to the
  synchronous InferenceSession.

### Shared owners and generated buffer operations

Use `vx.Runtime` to pass buffer IDs between sessions in one native owner:

```python
with vx.Runtime() as runtime:
    with runtime.inference_session("encoder") as encoder:
        with runtime.inference_session("head") as head:
            with encoder.run_tensors({"x": x}) as encoded:
                with head.run_tensors({"hidden": encoded["hidden"]}) as output:
                    scores = output["scores"].numpy()
```

Replace paths, names and shapes with the model contract. Standalone sessions
have separate owners; sharing between them requires an explicit DLPack import
or copy. `runtime.tensor(value)` produces an independent CPU snapshot.
`runtime.tensor(value, copy=False)` imports a dense CPU/CUDA DLPack owner once.
`tensor.numpy(copy=False)` acquires a direct read-only CPU mapping;
`writable=True` requests an exclusive write lease.

The generated API returns `Tensor.buffer` (`BufferView`), containing an ID,
offset and logical byte length. Reuse this descriptor in another model input
on the same owner. Use `VxBufferServiceClient.copy_tensors` for independent CPU
snapshots, inline readback or copies into `BorrowedBuffer` host destinations.
`retain_buffers` creates fresh IDs; `release_buffers` retires them in a batch.
Only `BeginBufferAccess` exposes a pointer/resource under an explicit lease.
`BorrowedBuffer` inputs can be carried in the execution call without a separate
import RPC. Names and shapes remain model contracts validated in C.

DLPack exports acquire an exclusive external lease, because consumers may
write. Other native reads/inputs/exports of that buffer return `BUSY` until the
consumer releases it. A NumPy `numpy(copy=False)` read view permits other
readers, and its lease survives Tensor/session close. Shared trainer parameters
use read-only leases; step/commit/rollback/reset return `BUSY` while they live.
Independent parameter snapshots do not block training. No memory-sharing
operation connects VolvoxAI autograd to PyTorch.

For bounded external CPU/CUDA access, `with tensor.torch_access() as view:`
ends permission at scope exit while remaining aliases retain only allocation
lifetime. CUDA records PyTorch's current stream on the tensor's device; supply
`streams=[first, second]` when multiple streams access the view. Order their
conflicting work yourself and stop using every alias after the scope. Native
input/readback/reuse then waits only for that allocation's recorded consumers.
Unrelated buffers remain usable. Ordinary `torch.from_dlpack(tensor)` continues
to use automatic lifetime and a conservative CUDA drain on final release.

See [buffers and tensors](../docs/buffers-and-tensors.md) for complete inference,
training, generated C and transport workflows, including current GPU and
WebGPU restrictions.

Native C uses these same messages without the Python adapter; see the complete
[C client](../native/tests/test_native_tensor_client.c) and
[before/after benchmark](benchmarks/README.md). GPU copies inside the engine
are included in those measurements.

See [the proto schema](../proto/volvoxai.proto) for the exhaustive operation
contracts and [native integration](../docs/native-runtime.md) for C dispatch.

## Async inference

```python
import asyncio
import numpy as np
import volvoxai as vx

async def main():
    values = np.array([[1, 2, 3]], dtype=np.float32)
    async with vx.AsyncInferenceSession("model") as session:
        outputs = await session.run(values, timeout=30)
        for name, array in outputs.items():
            print(name, array.shape, array.dtype, array)

asyncio.run(main())
```

Model initialization happens on entering the async context. Outside a context,
use `await session.open()` and `await session.close()`. Metadata and output
selection match the synchronous API. A session belongs to one event loop and
serializes its runs. It uses the generated async services; native work can run
while the event loop handles other tasks.

Each run snapshots its inputs when it acquires the session. Keep inputs stable
until that run returns, including while it waits behind another run.
`timeout` covers that wait and execution and raises `asyncio.TimeoutError`.
Cancellation raises `asyncio.CancelledError`. If native work has already begun,
the adapter waits for it to finish and releases the result before returning
the exception. This safe drain can exceed the timeout; it keeps arrays alive
until C has stopped using them. Cancellation while queued submits no work.
A completed cancellation leaves the session reusable.

## Handle errors

Catch an exception where your application can handle a failure. Uncaught
exceptions include a traceback pointing to the failed call:

```python
try:
    with vx.InferenceSession("model") as session:
        outputs = session.run(values)
except vx.VolvoxAIError as error:
    print(error.operation, error.status, error.code, error.stage)
    print(error.report)  # Original typed diagnostics and backend evidence.
    raise
```

The same rule applies to all synchronous and asyncio service clients exported
by `volvoxai`, including release methods that return an `OperationReport`
directly. Every non-OK operation status raises `VolvoxAIError`. The exception
retains `report`, `response`, `operation`, `status`, `code`, `stage`, `backend`
and the offending node index when present. Classify failures with proto enums;
message text is descriptive and does not determine error identity. A required
report missing from a response raises an internal error.

Successful responses keep their reports for backend and execution diagnostics.
A result with `state=PENDING` and an OK operation report is accepted work:
generated-service callers query `GetResult`; `InferenceSession` handles the wait.
Nested plan/revision evidence and
status descriptions are data; they are not recursively treated as failed calls.
For `BUSY`, overload or deadlines, inspect `error.status` and choose recovery
according to that operation's contract. The SDK performs no automatic retry.

`vx.FfiError` carries Synurang call failures and its `grpc_code`, including RPC
deadlines and unavailable methods. `vx.PluginClosedError` is its subclass for
a closed host. These remain distinct from an engine operation's report.
Async task cancellation propagates as `asyncio.CancelledError`. Awaiting an
async method raises at the `await` expression.

The package applies this policy to the generated methods after their single
response decode. Import service clients from `volvoxai`; the flat
`volvoxai_client` module is the raw generated transport binding and returns
operation reports without the package's exception policy. The former public
`check()` helper has been removed.

## Public API across languages

All three languages use the operations and types in `proto/volvoxai.proto`.
Their call and error conventions follow the host language:

| Language | Public call surface | Failure handling |
| --- | --- | --- |
| C | Generated request/response structs and codecs, dispatched through `Synurang_GetApi` | Check the terminal call status, then the operation report's status |
| Python | `InferenceSession`, `AsyncInferenceSession`, `quantize`, `TrainingSession`; `Vx*ServiceClient` / `AsyncClient` and `vx.pb` for the complete API | A direct call or `await` raises `VolvoxAIError`; call failures raise `FfiError` |
| TypeScript | `Vx*ServiceClient`, `pb` objects and `EngineHost` / `FullEngineHost` | `await` rejects with `VolvoxAIError`; call failures raise `RpcError` |

`InferenceSession` adapts paths and arrays and sequences existing generated
CreateRuntime, LoadModel, GetModelInfo, CompileModel, CreateExecutionContext,
Execute, GetResult, ReadOutput and ReleaseResult calls. It introduces no engine
operation or numerical implementation. Advanced service callers retain direct
access to every proto request, response and owner-scoped handle.

## Train, save and resume

Training uses the full library automatically. For a classifier whose exported
graph accepts one float32 input of shape `[1, 3]` and produces two or more
class logits named `logits`:

```python
import numpy as np
import volvoxai as vx

# Replace these illustrative batches with your preprocessed training dataset.
batches = [
    (np.array([[1, 2, 3]], dtype=np.float32), np.array([1], dtype=np.int64)),
    (np.array([[3, 1, 2]], dtype=np.float32), np.array([0], dtype=np.int64)),
]
with vx.TrainingSession(
    "classifier",
    loss=vx.CrossEntropyLoss(output="logits"),
    optimizer=vx.AdamW(lr=1e-3),
) as trainer:
    print(trainer.inputs, trainer.trainable_names)
    for inputs, targets in batches:
        result = trainer.step(inputs, targets)
        print(result.loss, result.optimizer_step, result.update_applied)
        for metric in result.metrics:
            print(metric.name, metric.correct, metric.examples)
    trained = trainer.save("trained-classifier")
    trainer.save_checkpoint("checkpoints/step-2.pb")

with vx.InferenceSession(trained) as session:
    print(session.run(batches[0][0]))
```

Loss, backward propagation and weight updates execute in C. The convenience
layer currently exposes cross entropy, AdamW and SGD, the choices supported by
the trainer contract. Labels must be integers that fit int32; int64 NumPy
labels are accepted after bounds checking. For multiple model inputs, pass a
name-to-array mapping. For multiple losses, pass a sequence of uniquely named
`CrossEntropyLoss` objects and a target mapping keyed by those loss names.

All float32 weights are selected by default. Use
`trainable_names=["classifier.weight", "classifier.bias"]` to train a subset
using names from your model's SafeTensors files. Unsupported graphs or parameter
choices raise the C engine's typed error. `optimizer=None` keeps the current C
configuration: a new trainer defaults to AdamW with learning rate 0.001,
beta1 0.9, beta2 0.999, epsilon 1e-8, zero weight decay and no clipping.
`vx.SGD(lr=0.01)` selects SGD. Both optimizer adapters expose `weight_decay` and
`max_gradient_norm`; AdamW also exposes `beta1`, `beta2` and `epsilon`.

`step()` waits for completion and changes only the trainer's private weights.
`save()` exports those weights with the exact loaded graph for inference.
Its `ModelPackage` result carries `path`, `graph_path` and ordered `weight_paths`
and can be passed directly to an inference session, including sharded models.
`commit()` publishes a model revision inside the session and makes the current
weights, optimizer and RNG the new rollback baseline. `rollback()` returns to
the last commit, the imported checkpoint, or the initial model state.
An explicitly supplied optimizer is applied again on each subsequent step.

Resume with the same graph, loss configuration and trainable selection:

```python
with vx.TrainingSession(
    "classifier", loss=vx.CrossEntropyLoss(output="logits"),
    checkpoint="checkpoints/step-2.pb",
) as trainer:
    result = trainer.step(batches[0][0], batches[0][1])
    print(result.optimizer_step)  # 3
```

Omitting `optimizer` preserves the checkpoint's exact optimizer configuration
and state. Checkpoints use the generated protobuf format and include weights,
optimizer slots, step count, RNG seed and optional application metadata.
Pass bytes with `save_checkpoint(path, metadata=b"...")`; omitting metadata
preserves imported metadata. C validates a checkpoint against the loaded model.
Loss selection, dataset position and trainable selection remain application
settings; store the latter two in application metadata when needed.

For gradient accumulation, set a positive `normalizer` on each loss for the
whole window and pass `accumulation_steps=N` to each `step`. `flush=True`
applies a partial window; `reset=True` discards unfinished accumulation before
the supplied batch. Export and checkpoint operations reject unfinished windows.
Saved model files and checkpoint paths are never overwritten; use a fresh
destination for each export. Close the session to release all native resources.

## Library selection and advanced services

Python ships one engine library, and discovery is automatic. An explicit file
override (`VOLVOXAI_LIBRARY`) is authoritative. Otherwise discovery checks
`VOLVOXAI_LIBRARY_DIR`, the bundled library, then the source build. A missing
explicit file or a selected library that cannot load raises an error.

`open_library()` reaches every service: CPU/GPU inference, tokenization, graph
planning, scheduling, buffers, training, optimizer/checkpoint operations,
adapters and PTQ. The repository also builds a smaller inference-only
`libvolvoxai-lite` for the native and browser releases; it is not installed by
the wheel, and the Training and Quantization clients do not work against it.

For advanced training control, load a model through `VxInferenceServiceClient`, then use
`VxTrainingServiceClient` on the same host to create a trainer,
submit training steps, inspect state and export or commit the result.
`RollbackTrainer` restores the pinned baseline. The
[training integration example](https://github.com/ivere27/volvoxai/blob/main/python/tests/test_training_end_to_end.py)
demonstrates an SGD update and rollback checked against an independent oracle.

Text processing uses `VxTextServiceClient`: create a tokenizer, encode text,
decode tokens, then release it. `VxPlanningServiceClient` inspects and edits
graph plans. `VxSchedulerServiceClient` manages execution requests and batch
queues. These clients share the selected host and its handle ownership.

Every service also has an `AsyncClient` for asyncio:

```python
import asyncio
import volvoxai as vx

async def main():
    async with vx.AsyncModuleHost(vx.open_library("full")) as host:
        platform = vx.VxPlatformServiceAsyncClient(host)
        info = await platform.get_platform_info(vx.pb.Empty())
        print(info.library_version, info.compiled_backends)

asyncio.run(main())
```

The [proto schema](https://github.com/ivere27/volvoxai/blob/main/proto/volvoxai.proto)
is the authoritative API contract and is also bundled under `volvoxai/schema/`.
For live discovery, call `VxPlatformServiceClient.describe_api` with
`pb.DescribeApiRequest(include_types=True)`. It returns the methods and message
contracts actually present in the loaded library.

## Post-training quantization

First export the original FP32 ONNX model as shown above. `vx.quantize` accepts
an iterable or generator of representative, preprocessed NumPy batches. A batch
is an array for a single-input model or a mapping for multiple inputs:

```python
from pathlib import Path
import numpy as np
import volvoxai as vx

def calibration_batches():
    for path in sorted(Path("calibration").glob("*.npz")):
        with np.load(path, allow_pickle=False) as batch:
            yield {name: batch[name] for name in batch.files}

result = vx.quantize("model", calibration_data=calibration_batches(), output="quantized")
print(result.graph_path, result.weight_paths)
print(result.quantized_nodes, result.retained_float_nodes)
print(result.calibration_batches, result.calibration_samples)
with vx.InferenceSession(result) as session:
    outputs = session.run(values)
```

The generator is consumed once, one batch at a time; no calibration file format
is required by `quantize`. Empty data or a failed batch publishes no model.
For a single complete batch, use `calibration_data=[array]` or
`calibration_data=[{"input0": array}]`.

The CLI uses the same Python workflow with a directory of `.npz` files. Each file is one
complete inference batch; keys, dtypes and shapes must match the exported model.
For example, an exported input named `input0` with shape `[1, 3]`:

```python
from pathlib import Path
import numpy as np

Path("calibration").mkdir(exist_ok=True)
# Replace this illustration with representative data from your application.
np.savez("calibration/000.npz", input0=np.array([[1, 2, 3]], dtype=np.float32))
```

```sh
volvoxai ptq --graph model/graph.json --weights model/model.safetensors \
  --calibration calibration --out quantized
```

This workflow calls native template authoring, plan creation, calibration and
package writing through `VxQuantizationServiceClient`. It produces a VolvoxAI
`quantized/graph.json` and `quantized/model.safetensors`, not a quantized ONNX file. It uses
CPU calibration and does not require a GPU. Defaults are symmetric INT8
activations and per-channel symmetric INT8 weights. Use `volvoxai ptq --help`
for explicit precision choices and float-retention options.
The corresponding Python keywords are `activation_dtype`, `activation_scheme`,
`float_operators`, `float_nodes`, `selected_nodes` and `reduce_range`.
`samples_per_batch` records the number of logical examples in each batch;
`cpu_threads` defaults to the C engine's automatic choice.

Run the result with the same inference API and the same preprocessed input:

```python
with vx.InferenceSession("quantized") as session:
    outputs = session.run(values)
```

`--samples-per-batch` records how many logical examples each `.npz` represents;
the default is one. Existing output package files are never overwritten. Model
downloads do not supply a representative calibration or evaluation corpus.
Keep calibration data separate from the accuracy evaluation set.

The repository has complete workflows for the
[receipt reader](https://github.com/ivere27/volvoxai/tree/main/examples/receipt_digit_reader)
and [receipt VQA](https://github.com/ivere27/volvoxai/tree/main/examples/tiny_receipt_vqa).
Model-specific preprocessing stays with those examples.

## Build and publish

From the repository root:

```sh
make build_wheel
```

This builds `volvoxai-wheel-build:0.5.0` from `python/Dockerfile.wheel`. Its
pinned [manylinux](https://github.com/pypa/manylinux) base supplies the Linux
compatibility baseline; native CPU/CUDA/Vulkan/OpenGL inference and full
libraries are compiled from the current checkout. The source checkout is
mounted read-only. Temporary native outputs do not replace the existing
`native/` or npm release artifacts. `WHEEL_BUILD_JOBS` defaults to two.
Native build intermediates are cached under `build/python-wheel/`, invalidated
when the builder image changes. Every qualification run creates fresh Python
environments. Failed candidates remain in that ignored build directory for
diagnosis and are not published into `dist/python/`.

The build checks native profile boundaries and CUDA PTX, repairs the wheel with
auditwheel, checks PyPI metadata with Twine, and installs it into clean CPython
3.10–3.14 environments. It exercises generated API coverage, sync/async calls,
CPU inference, tokenization, planning, scheduling, training/rollback, ONNX
conversion/PTQ and automatic exceptions across every sync/async service method.
Session qualification includes model/library discovery, repeated inference,
dynamic multi-input models, output selection, dtype preservation and cleanup
on errors. Both FP32 exports and native PTQ packages run through the NumPy API.
Workflow qualification also covers immutable metadata, host-buffer transport,
async cancellation/deadline cleanup, streaming PTQ, an independent SGD oracle,
AdamW checkpoint continuation, commit/rollback and fresh-file publication on
every supported Python version. Tensor tests cover DLPack ownership,
legacy/versioned capsules, CPU sharing,
dynamic shapes and late external releases. For source-tree DLPack tests, first
run `python python/build_dlpack_bridge.py` with setuptools and Python development
headers installed.

`wheel-validation.json` records the CPU scope
and test files for that exact wheel; a rebuild does not inherit GPU results
from an older artifact.
Physical GPU execution is qualified separately on a GPU host
using `VOLVOXAI_WHEEL_TEST_BACKEND=cuda`, `vulkan` or `opengl` with
`python/tests/test_installed_wheel.py`.

For CUDA tensor interoperability, install a compatible PyTorch distribution
on the GPU host. To require all three Linux GPU backends, run:

```sh
VOLVOXAI_TEST_CUDA=1 VOLVOXAI_TEST_NATIVE_BUFFERS=cuda,vulkan,opengl \
  python -I python/tests/test_tensor_interop.py
```

This additionally checks nondefault streams, exact shared pointers, results
that survive session close, cross-session reuse, INT32 values, a native INT8
PTQ package, interpreter shutdown and CUDA profiler transfer events.
`VOLVOXAI_TENSOR_TEST_EVIDENCE=path.json` saves the profiler's copy
event names and framework versions. The copy test uses a small warmed float32
model; it is not a receipt-model throughput benchmark.

The profiling tests import PyTorch before creating CUDA contexts. With the
qualified PyTorch 2.5.1/CUDA 12.1 and driver 535.309.01 combination, calling
`cuInit` before importing PyTorch can make CUPTI crash at process exit. A
driver-only reproducer confirms this without loading VolvoxAI. The import
order is a profiling-environment requirement; ordinary engine execution does
not depend on PyTorch or CUPTI.

Successful builds publish these local files:

```text
dist/python/0.5.0/volvoxai-0.5.0-cp310-abi3-manylinux_2_28_x86_64.whl
dist/python/0.5.0/SHA256SUMS
dist/python/0.5.0/wheel-validation.json
```

The engine uses its generated C ABI. The DLPack capsule adapter uses CPython's
stable ABI with a Python 3.10 minimum; it has no CUDA/framework link dependency.
The package is not a pure-Python `any` wheel. The version comes from `package.json`.
Build provenance records the source digest and whether the checkout was dirty.

To upload the already validated wheel yourself:

```sh
python -m pip install twine
python -m twine check --strict dist/python/0.5.0/*.whl
python -m twine upload dist/python/0.5.0/*.whl
```

Use Twine's interactive credentials, keyring or your release environment for
authentication. The build target only creates and validates local artifacts;
it does not upload packages. Attach the same wheel and checksum to GitHub
Releases. JS/WASM retain their existing `dist/0.5.0/` names and npm packaging.
