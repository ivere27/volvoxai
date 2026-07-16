# Backend SDK

VolvoxAI has one backend lifecycle on native and browser targets. A device
integration registers a stable name, advertises the nodes it can execute, and
declines the rest. The portable CPU implementation remains the final
correctness fallback, so a vendor backend can start with one or two high-value
operators without copying the runtime or editing a core backend enum.

## Native C ABI

Include the public engine and backend headers, fill a `VxBackendV1`, and
register it before loading a model:

```c
#include "volvoxai.h"
#include "volvoxai_backend.h"
#include <string.h>

static int device_init(void* context) { /* open the driver */ return VX_INIT_READY; }
static int device_supports(void* context, const VxNode* node) {
    return vendor_can_execute_exactly(context, node) ? VX_HANDLED : VX_DECLINED;
}
static int device_run(void* context, const VxNode* node) {
    const VxTensor* input = vx_node_input_by_key(node, "input");
    const VxTensor* weight = vx_node_input_by_key(node, "weight");
    VxTensor* output = vx_node_output_by_key(node, "out");
    return vendor_linear(context, input, weight, output) == 0
        ? VX_HANDLED : VX_ERROR;
}
static void device_teardown(void* context) { /* close the driver */ }

VxBackendV1 backend = {
    .struct_size = sizeof(VxBackendV1),
    .abi_version = VX_BACKEND_ABI_V1,
    .name = "my-npu",
    .user_data = &driver,
    .init = device_init,
    .supports = device_supports,
    .run = device_run,
    .teardown = device_teardown,
};

volvoxai_register_backend(&backend);
volvoxai_engine_configure_backend("my-npu");
volvoxai_engine_init("config.json", "model.safetensors");
```

The compilable [host backend example](../examples/backend_sdk/host_backend.c)
is a public-header-only starting point for an out-of-tree integration. The
[Android NNAPI example](../examples/backend_sdk/android_nnapi_backend.c) uses
the same ABI for one conservative accelerator operation.

`VxNode` and `VxTensor` are borrowed opaque views. Use the accessors for
operator names, semantic operand keys, shapes, dtypes, storage, attributes, and
per-tensor/per-axis quantization; never include private `Node` or `T` headers.
The descriptor and name are copied, while `user_data` and the state it reaches
remain caller-owned while the backend can be selected.

All backend callbacks run under the engine model lock. They must not call
`volvoxai_engine_*` APIs. `supports()` and `run()` inspect their borrowed graph
views only with the public `vx_*` accessors; lifecycle callbacks may call the
vendor driver directly. The one intentional nested callback is
`run()` → `vx_tensor_sync_host()` → the selected backend's `sync_host()`, so
`sync_host()` must not wait on a vendor lock already held by `run()`. Handles
and metadata pointers are callback-scoped. A device backend may retain only a
tensor data pointer's identity as an opaque storage key until `reset()` or
`teardown()`, and may dereference it only when host storage is current.
`teardown()` is called after every non-ready `init()` attempt, so partially
initialized driver state should be released there.

`supports()` is a semantic promise, not just an operator-name filter. Check
every dtype, shape/layout relation, optional operand, attribute, quantization
descriptor, and fused behavior that changes the result. Return `VX_DECLINED`
for an otherwise valid node outside that exact subset so CPU can execute it;
reserve `VX_ERROR` for driver or execution failures after accepting the node.

ABI v1 describes complete ordinary-inference nodes only. The runtime therefore
bypasses public SDK callbacks during native training, adapter-modified Linear,
legacy QTensor execution, and active prefix/row/execution-row windows; the
remaining built-in registry routes, ending in CPU, retain those semantics. A
future ABI must expose an execution window explicitly before an out-of-tree
backend can opt into partial-node work. V1 also has no contract for publishing
the runtime-private attention K/V rows, so selecting any public backend makes
`volvoxai_engine_incremental_row_supported()` return false and rejects a
decode session that requires row mode. Complete-node dependency incremental
execution remains available.

Registration is discovery, not activation. `volvoxai_engine_configure_backend`
selects one registered backend by name, and an unavailable explicitly selected
device fails instead of silently changing policy. Built-in enum values remain
for source compatibility, but a new accelerator does not add another enum
member or another per-node `#if` branch.

A host-output backend writes through `vx_tensor_data()`. A backend that retains
device-resident outputs sets `VX_BACKEND_FLAG_DEVICE_RESIDENT_OUTPUTS` and
implements the paired forward/coherence hooks. The runtime keys coherence by
the borrowed host allocation, allowing the vendor to keep a private device
buffer table without putting device handles on the public tensor object.

## Browser contract

`BackendEngine` defines the matching JavaScript lifecycle:

- `allocateGraph(graph)` binds and prepares one graph;
- `execute(inputs, options)` runs it;
- `capabilities` declares incremental execution, row execution, and host/device
  output location;
- `createDecodeSession()` provides backend-neutral autoregressive caching.

`CPUEngine`, `WasmEngine`, `WebGPUEngine`, and `WebNNEngine` are peers under
that contract. `WebGPUEngine` owns the lower-level `GraphExecutor`, so WebGPU no
longer needs a special branch in `VolvoxAI.compile()`.

Out-of-tree browser devices register a factory by name:

```js
VolvoxAI.registerBackend('my-npu', async ({ name, runtime }) => {
  return new MyNpuEngine({ name, runtime });
});

const runtime = await VolvoxAI.init('my-npu');
```

The engine may subclass `BackendEngine` or implement the complete structural
API v1 equivalent: matching `backendApiVersion`/`backendName`, frozen
capabilities, `allocateGraph()`, `execute()`, and `createDecodeSession()`.
An engine advertising incremental execution must additionally implement
`resetDecodeCache()` and expose a non-negative, monotonically increasing safe
integer `decodeCacheGeneration`; extending `BackendEngine` provides both.
Built-in names cannot be replaced.

## Android and NNAPI

The existing NNAPI selection remains a compatibility backend for older
deployments, but NNAPI was deprecated in Android 15. New Android integrations
should use the named backend SDK for a vendor runtime or delegate instead of
adding another Android API branch. Google documents both the
[NNAPI migration path](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)
and [vendor NPU delegates for LiteRT](https://ai.google.dev/edge/litert/android/npu).
The VolvoxAI contract is deliberately model-runtime-neutral: a QNN, LiteRT,
or custom driver adapter implements the same opaque node/tensor ABI.

The SDK example registers as `android-nnapi-add`; it does not replace the
legacy `VOLVOXAI_BACKEND_NNAPI` enum or `--nnapi` option. Its eligibility
contract is deliberately narrow: exact-shape rank 1–4 F32 `Add`, with no
broadcasting or fused activation. The `test_backend_sdk` CTest case (run by
`make test_native`) compiles the example using only the public VolvoxAI headers;
an Android build cross-compiles it with the NDK CMake toolchain. See the
[example README](../examples/backend_sdk/README.md)
for integration and performance limitations.
