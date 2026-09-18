# Share tensors and control their lifetime

VolvoxAI tensors can connect models, feed a trainer, or share data with NumPy
and a DLPack consumer. A `Runtime` gives sessions one native owner. Within that
owner, passing a tensor passes a buffer capability and shape, without encoding
its values as protobuf bytes. C manages storage, access leases, snapshots and
recycling in both build profiles.

```python
import numpy as np
import volvoxai as vx

with vx.Runtime() as runtime:
    with runtime.inference_session("encoder") as encoder:
        with runtime.inference_session("head") as head:
            with encoder.run_tensors({"x": np.ones((2, 4), np.float32)}) as encoded:
                with head.run_tensors({"hidden": encoded["hidden"]}) as prediction:
                    scores = prediction["scores"].numpy()
                shared = encoded["hidden"].numpy(copy=False)

# A direct CPU view retains its native read lease after all sessions close.
print(shared.shape)
del shared
```

Replace model paths, names and shapes with the model's declared contract.
Standalone `InferenceSession` and `TrainingSession` constructors still create
independent owners. An ID from one owner cannot be replayed in another. Use a
shared `Runtime`, an explicit CPU snapshot with `runtime.tensor(value)`, or a
supported DLPack import with `runtime.tensor(value, copy=False)`. The snapshot
path accepts host arrays and VolvoxAI tensors. For a foreign CUDA tensor, first
import it with `copy=False` or explicitly download it through its producer;
`CopyTensors` does not read arbitrary borrowed device pointers.

## Values, storage and access

A tensor contains dtype, concrete shape and one payload. Its optional `name`
is a model binding label; it is not storage identity. This keeps the binding
API compact without introducing a separate named-tensor wrapper.

| Payload | Meaning | Lifetime |
| --- | --- | --- |
| `inline` | Transported value bytes | Owned by the request or response |
| `buffer` (`BufferView`) | Buffer ID, byte offset and logical length | C reference until the capability is released |
| `borrowed` (`BorrowedBuffer`) | Local native resource and byte range | All accesses finish before dispatch returns |

`BufferView.buffer_id` is never an address. `RetainBuffers` issues fresh IDs
for independent ownership, and `ReleaseBuffers` retires IDs idempotently.
Descriptor copies do not retain anything. Accepted operations acquire their
own references. A slice keeps the allocation alive; retaining one output does
not retain unrelated output allocations or their byte budgets. The execution
batch's budget slot remains reserved until its last output is released.

`NativeResource.kind` describes how to interpret a handle: CPU/CUDA address,
`VkBuffer`, OpenGL buffer name, or `MTLBuffer`. It does not describe physical
placement. `BufferInfo.cpu_accessible` reports direct mapping capability.
`BeginBufferAccess(host_mapping=True)` returns a CPU pointer only when a direct
mapping exists. An `MTLBuffer` object and its shared `contents` pointer remain
distinct representations, including on Apple Silicon. Private Metal storage
requires explicit readback. Managed Metal and arbitrary external Vulkan/OpenGL
mapping are not implemented by this interface.

`BeginBufferAccess` acquires a read or exclusive write lease. Concurrent read
leases are allowed; writes conflict with both reads and writes and return
`BUSY`. End external access after its operations finish, or supply explicit
CUDA consumer streams as described below. Unknown CUDA consumer work is
conservatively drained when an external lease ends. Current execution publishes
completed snapshots; beginning external access also waits for earlier declared
consumers of that allocation. Mappings never silently stage a copy.

```python
with runtime.tensor(np.arange(8, dtype=np.float32)) as tensor:
    snapshot = tensor.numpy()  # Independent CPU copy.
    view = tensor.numpy(copy=False)  # Read lease; NumPy cannot enable writes.
    del view
    writable = tensor.numpy(copy=False, writable=True)  # Exclusive lease.
    writable[:] *= 2
    del writable  # All aliases must finish before native reuse/readback.
```

Copies and mutation are different. A snapshot is independent of later engine
or trainer changes, but may be changed through its own authorized write lease.
`CopyTensors` produces CPU snapshots, portable inline bytes, or explicit copies
into local host destinations. A batch validates all sources/destinations before
writing destinations and preserves simultaneous-copy semantics for overlapping
ranges. A one-tensor CPU copy needs no intermediate staging; CUDA readback uses
the bounded pinned staging described below. Arbitrary device
allocation/upload and general strided views are outside the implemented domain.
`AllocateBuffers` currently creates zero-initialized CPU storage.

## Training uses the same tensors

Training needs the full engine library, which is what Python installs.
Importing the inference wrapper or allocating buffers does not load training
workflows, and training code is not compiled into the inference-only library.

```python
with vx.Runtime() as runtime:
    with runtime.training_session(
        "classifier", loss=vx.CrossEntropyLoss(output="logits"),
        optimizer=vx.AdamW(lr=1e-3), trainable_names=["head.weight"],
    ) as trainer:
        x = np.ones((2, 4), np.float32)
        with runtime.tensor(np.array([0, 1], np.int32)) as targets:
            with trainer.parameters(["head.weight"]) as before:
                with trainer.step(x, targets, output_names=["logits"]) as step:
                    print(step.loss)
                    logits = step["logits"].numpy()
                # before remains unchanged after backward and the optimizer.
                baseline = before["head.weight"].numpy()

        with trainer.parameters(["head.weight"], mode="shared_read") as parameters:
            inspect = parameters["head.weight"].numpy(copy=False)
            # step/commit/rollback/reset cannot mutate while inspect is live.
            del inspect
```

Targets are dense I32 tensors. Python integer sequences are checked and
converted to I32; native tensor targets are checked in C. Retained GPU targets
use explicit C readback for the current host loss-validation path. Transient
non-host targets require an explicit import/readback first. Device inputs use
the same binding/copy path as inference and retain backend admission checks for
index and route values. No host validation proof is bypassed.

Selected forward values are captured before backward and optimizer mutation.
They do not expose the autograd tape. Loss/accuracy stay scalar metrics;
gradients, optimizer slots and accumulation internals remain private. A native
`TrainStep` publishes selected snapshots once. `GetTrainStep` returns state and
metrics and never issues duplicate output handles. WebGPU currently rejects
nonempty forward-output selection before submitting work.

Parameter exports default to snapshots. CPU `shared_read` exports pin a
specific parameter allocation and block trainer mutation until every buffer
handle and external read lease is released. Closing a Python wrapper alone
cannot invalidate an ndarray that still references the parameter. GPU shared
parameter views are currently rejected; native GPU snapshots are supported.
KV caches retain their existing execution-context/page ownership and copy-on-
write rules; they are not automatically exposed as dense public tensors.

## DLPack shares storage, not autograd

```python
import torch

with session.run_tensors({"x": torch_input.detach()}) as outputs:
    shared = torch.from_dlpack(outputs["hidden"])
# shared retains the C allocation and the loaded module.
consume_in_torch(shared)
del shared
```

C consumes and exports standard `DLManagedTensor` or versioned DLPack owners.
Its deleter retains storage independently of public IDs. The Python extension
only adapts capsules, Python references and library lifetime; no PyTorch or
NumPy dependency is linked into the native storage service. Native applications
must keep the library loaded until all exported deleters have returned.

A normal DLPack consumer can write, so an export takes an exclusive external
lease. Native readback, a second export and native input use of that same
buffer return `BUSY` until the consumer releases it. Pass the external PyTorch
object itself when its producer protocol must establish a stream handoff.
Read-only trainer views cannot be exported through this writable path; export
an independent parameter snapshot instead. Persistent imports retain the
producer but cannot police mutation through aliases still held by that producer.
CUDA exports wait for their own producer if necessary; the final external
release conservatively waits for unknown consumer work. They reject `stream=-1`,
which would promise that the producer performs no synchronization.

CUDA import currently requires a CUDA session to have initialized the native
backend for the same device. The import then retains that backend independently:
closing the session or `Runtime` cannot remove the device state needed to read
or export the imported tensor. Import does not lazily initialize another device,
and CUDA managed/mapped host allocations are outside the admitted domain. The
[CUDA interop tests](../python/tests/test_tensor_interop.py) cover these lifetime
paths and producer ordering in both native profiles. They were qualified on a
physical RTX 3090.

Calling `requires_grad_()` on an imported PyTorch tensor tracks subsequent
PyTorch operations. It does not attach VolvoxAI inference or training history.
A cross-engine backward bridge would require a separate contract.

### Scope PyTorch access to avoid unrelated CUDA waits

Use `torch_access()` when all external work has an explicit end. The context
manager gives PyTorch an exclusive shared view. On exit, C records completion
on the declared consumer streams. A later native operation waits only when it
uses that allocation. Ending the scope does not enqueue a wait on the shared
engine stream, so another buffer can still proceed.

```python
import torch

with vx.Runtime() as runtime:
    with runtime.inference_session("encoder", backend="cuda") as encoder:
        with encoder.run_tensors({"x": np.ones((2, 4), np.float32)}) as outputs:
            tensor = outputs["hidden"]
            stream = torch.cuda.Stream(device=tensor.__dlpack_device__()[1])
            with tensor.torch_access(streams=[stream]) as view:
                with torch.cuda.stream(stream):
                    view.mul_(0.5)
            # Native reads/input reuse now depend on the recorded work.
            values = tensor.numpy()
            del view
```

Without `streams`, CUDA uses PyTorch's current stream on the tensor's device;
CPU needs no stream. This method does not change PyTorch's current stream or
order work between multiple consumer streams. If several streams touch the
view, pass every one in `streams=[...]` and order conflicting operations with
your framework's events. No view or derived alias may be used after scope exit.
An escaped alias keeps the allocation alive, but has no access permission.
Ordinary `torch.from_dlpack(tensor)` keeps its automatic permission lifetime and
conservative final drain. Neither form connects autograd graphs. Training inputs
and exported forward/parameter snapshots use the same scope in the full profile.

For C and other FFI clients, call `BeginBufferAccess(WRITE)`, optionally bind a
standard DLPack export with `ExportDLPack(access_id=...)`, enqueue all external
work, then call `EndBufferAccess(cuda={device_id, streams})`. The existing access
ID covers a byte range; an export must fit that range and reference the same
allocation. The DLPack deleter still releases its lifetime reference. No extra
completion handle or RPC is needed. The complete [C client](../native/tests/test_native_tensor_client.c)
exercises both CPU and CUDA scopes through generated dispatch.

Streams must be live handles from the allocation's device and primary context,
kept alive through the end call. C accepts explicit legacy default stream `1`,
and rejects `0`, per-thread stream `2`, or capturing streams. Python translates
its default stream to `1`. Arbitrary integers are not safe stream handles.
Admission errors leave the permission active. A driver error may already have
recorded some dependencies; retry or finish conservatively. At most 64 pending
distinct consumer streams are stored per allocation; exceeding this bound
returns `BUSY`. Event objects are cached and reused until allocation teardown.
Registration has no host wait. Last-owner destruction, pool eviction or shutdown
may wait for recorded work; final CUDA allocation frees can still synchronize.
For predictable warm-path behavior, retain storage in a live owner/pool.

## Copy and transport boundaries

| Environment | Implemented contract |
| --- | --- |
| Native CPU | All storage operations, direct mappings, transient borrows, CPU DLPack |
| Native CUDA | Device snapshots and handoff, CUDA DLPack, host readback; explicit scoped consumer completion or conservative external drain |
| Native Vulkan/OpenGL | Retained native snapshots and same-owner handoff; explicit host readback; external arbitrary imports rejected |
| Native Metal | Retained snapshots, validated borrowed buffers, direct shared-storage CPU mapping; physical Mac qualification remains outstanding |
| WASM CPU | Inline values and C-owned buffer capabilities, copies/retains/releases, retained execution outputs; foreign pointers/mappings/DLPack rejected |
| WebGPU | Existing full-profile asynchronous execution/training and staging; retained `GPUBuffer` interop is not exposed |
| Remote transport | Inline values and IDs belonging to that server owner; a native network adapter must reject local pointers, DLPack and CUDA stream descriptors before dispatch |

Inference still copies inputs into its execution arena and snapshots outputs.
Same-device handoff avoids a host download/upload, not those engine copies.
Output selection, unchanged-input reuse and previous-output feedback stay in C.
Feedback refers to engine state, so modifying an exported snapshot cannot change
it. Each context retains the existing pool limits of 64 idle allocations and
64 MiB. Live public/internal references, external leases and GPU work prevent
reuse. Native CUDA/Vulkan/OpenGL input and output copies remain batched.

CUDA tracks per-allocation completion events and reuses storage with known
dependencies on a nonblocking engine stream. Owned tensor handoff, unchanged
inputs, pooled snapshots and readback avoid context-wide drains on the qualified
warm paths. A borrowed DLPack input still depends on its producer's stream;
generic external lease release retains its conservative context wait. Scoped
external access records only the declared streams' dependencies. Public
execution still waits for its own output completion, and independent requests
still share an engine stream. Stream-aware reuse does not itself add request
parallelism. The [CUDA reuse benchmark](../python/benchmarks/cuda_stream_reuse.py)
measures the effect of unrelated GPU work and verifies exact outputs.

Native training installs device inputs after its cache
preparation so a forward pass cannot replace device values with stale host
shadows. OpenGL destroys owned GPU resources but keeps shared EGL display
initialization for the process lifetime, preserving other modules' contexts.

NumPy inputs use a transient descriptor in the execution call; they need no
`Import -> Execute -> Release` round trip. `TensorOutputs.close()` releases a
batch of IDs in one call while C external leases independently retain storage.
Explicit sharing has control-call costs: a direct NumPy view begins/ends an
access lease, DLPack export calls `ExportDLPack`, and the first Python device
query calls `GetBufferInfo` (then caches it). These do not add import calls to
the ordinary `run_tensors` path; native tensor inputs carry their existing IDs.
General multi-source host copies may stage to preserve overlap semantics.
Device readback and device-to-CPU snapshots require transfer/synchronization.
Pageable CUDA host transfers stage through pinned memory capped at 8 MiB per
loaded library. Each chunk adds a CPU copy and waits only for the engine stream
before the slab can be reused. This isolates known transfers from unrelated
default-stream work; idle latency can increase. Same-device tensor handoff
still performs only device-to-device copies and uses no host staging.
A direct mapping request that cannot share storage fails instead of copying.

## Generated C and other FFI clients

Every application operation uses the generated dispatch. The complete
[native C execution client](../native/tests/test_native_tensor_client.c) creates
a runtime/model/context, executes retained tensors, feeds them back, reads them
through `VxBufferService.CopyTensors`, and releases their buffer IDs.
The following fragments use its `CALL` macro and initialized response messages:

```c
VolvoxaiV1Tensor input = first.field_outputs.data[0];
input.field_name = next_input_name;  /* Only the binding label changes. */
execute.field_inputs.data = &input;
execute.field_inputs.len = 1;
CALL(VX_RPC_VX_INFERENCE_SERVICE_EXECUTE_TENSORS,
     volvoxai_v1_execute_tensors_request, &execute,
     volvoxai_v1_tensor_batch, &second);

VolvoxaiV1CopyTensorsRequest copy;
volvoxai_v1_copy_tensors_request_init(&copy);
copy.field_sources.data = second.field_outputs.data;
copy.field_sources.len = second.field_outputs.len;
copy.field_inline_result = 1;
CALL(VX_RPC_VX_BUFFER_SERVICE_COPY_TENSORS,
     volvoxai_v1_copy_tensors_request, &copy,
     volvoxai_v1_tensor_batch, &host_values);
```

Full-profile clients supply targets and optional forward selection in the same
training request. The labels below borrow stack-backed message fields only
until synchronous encoding; do not run generated free functions on those
borrowed stack messages:

```c
int32_t labels[] = {0, 1};
int64_t label_shape[] = {2};
VolvoxaiV1Tensor targets;
volvoxai_v1_tensor_init(&targets);
targets.field_dtype = VOLVOXAI_V1_DATA_TYPE_I32;
targets.field_shape.data = label_shape;
targets.field_shape.len = 1;
targets.which_payload = 5;
targets.field_inline.data = (uint8_t*)labels;
targets.field_inline.len = sizeof(labels);
loss.field_targets = &targets;
step.field_losses.data = &loss;
step.field_losses.len = 1;
step.field_inputs.data = training_inputs;
step.field_inputs.len = input_count;
CALL(VX_RPC_VX_TRAINING_SERVICE_TRAIN_STEP,
     volvoxai_v1_train_step_request, &step,
     volvoxai_v1_train_step_result, &trained);
```

Use the [schema](../proto/volvoxai.proto) and generated C/Python/TypeScript
references for all request fields and operation reports. This storage API adds
no separate hand-written public lifecycle entry point.
