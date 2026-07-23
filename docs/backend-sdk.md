# Backend SDK

VolvoxAI has one provider SPI ownership contract across JavaScript and native
targets. The SPI composes backend implementations into a runtime; it is not an
application-facing Synurang operation:

~~~text
provider runtime instance
  compiled-model instance
    execution-context instance
      stable output snapshot
~~~

A provider receives explicit scope at every callback. It must not use
process-global graph, tensor, request, decode, or output state. Different
contexts created from one compiled model must own independent mutable execution
state.

Provider selection and operator routing are separate policies. Compilation
finishes provider selection before execution begins, and execution failure is
never retried on another provider.

## JavaScript provider SPI

Supply a provider factory under a canonical lowercase name when creating its
owning Runtime:

~~~javascript
import {
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  VolvoxAI,
  createBackendDeviceIdentity,
  createBackendProviderCapabilities,
} from 'volvoxai';

const myNpuProvider = async ({ name }) => {
    const device = await openVendorDevice();
    const deviceIdentity = createBackendDeviceIdentity({
      vendor: device.vendor,
      model: device.model,
    });

    return {
      providerVersion: VOLVOXAI_BACKEND_PROVIDER_VERSION,
      backendName: name,
      deviceIdentity,
      capabilities: createBackendProviderCapabilities({
        operatorFallback: 'none',
        outputLocation: 'host',
      }),

      async compile(snapshot, options) {
        const plan = await compileVendorPlan(snapshot, options);

        return {
          backendName: name,
          compilationEvidence: {
            device: deviceIdentity,
            allocationBytes: plan.allocationBytes ?? null,
          },

          async createContext(contextOptions = {}) {
            const state = await plan.createExecutionState(contextOptions);

            return {
              backendName: name,

              async execute(inputs, executionOptions = {}) {
                const outputs = await state.execute(
                  inputs,
                  executionOptions,
                );
                return {
                  outputs,
                  backendReport: {
                    route: {
                      operatorFallbackUsed: false,
                    },
                  },
                };
              },

              async close() {
                await state.close();
              },
            };
          },

          async close() {
            await plan.close();
          },
        };
      },

      async close() {
        await device.close();
      },
    };
};

const runtime = await VolvoxAI.createRuntime({
  backends: ['my-npu'],
  providers: { 'my-npu': myNpuProvider },
});
~~~

The factory receives name, runtime, and wasmUrl. It may return null when its
device is unavailable. The descriptor must use the exported provider version,
return frozen capabilities from createBackendProviderCapabilities(), and use a
backendName matching the configured name. createBackendDeviceIdentity() copies
non-empty string fields into frozen serializable metadata. Built-in names are
reserved.

A provider factory or already-created provider instance belongs only to the
Runtime whose `providers` option receives it. Runtime closure closes the
initialized provider; there is no process-global provider registry.

### Compilation

compile() receives an immutable ModelSnapshot and:

~~~ts
interface BackendProviderCompileOptions {
  readonly operatorFallback: 'allow' | 'forbid';
}
~~~

The snapshot identifies its definition, topology revision, weight revision,
declared outputs, tensor count, and node count. A provider may create a private
execution Graph through snapshot.createExecutionGraph(). It must not retain or
mutate the caller's original Graph or Tensor storage.

Compilation must reject unsupported operators, dtypes, shapes, layouts,
attributes, quantization descriptors, or fallback policy. A provider that
advertises operatorFallback: 'none' promises the complete selected graph stays
on that provider.

The optional compilationEvidence object records a provider-reported device
identity and a non-negative allocation-byte total, or null when the provider
cannot report them. VolvoxAI copies this evidence into the immutable
compilation report together with compile time and route evidence; it does not
invent a physical-device string.

The object returned by `compile()` is VolvoxAI's backend-prepared graph owner.
There is no additional public `PreparedGraph` lifecycle and no backend-specific
graph document. A provider may resolve operator routing, select kernels, build
an immutable schedule, prepack constant weights, choose physical layouts, plan
memory, or create pipelines and target code here. All such state is derived
from the exact snapshot revision and must be discarded or rebuilt when that
revision changes. The portable deployment source remains the optimized
`volvox-graph/v1` document plus safetensors.

Prepared state shared by contexts must be immutable. Pointer tables into a
context-owned heap, request-dependent shapes, scratch arenas, inputs, outputs,
adapter/decode state, and mutable command state are materialized by
`createContext()` and never shared between contexts. Rebuildable binary or
packed-weight caches are optional implementation details, not package inputs.

### Contexts

Each createContext() call returns a distinct mutable execution owner:

~~~ts
interface BackendProviderExecutionContext {
  readonly backendName: string;
  execute(inputs, options?): Promise<BackendExecutionSnapshot>;
  decodeSeed?(inputs, options?): Promise<BackendExecutionSnapshot>;
  decodeStep?(inputs, options?): Promise<BackendExecutionSnapshot>;
  decodeReset?(): Promise<void>;
  close(): Promise<void> | void;
}
~~~

Inputs are typed arrays keyed by declared graph input name. A context may own
device allocations, scratch, command encoders, adapter routing, and decode/KV
state. It must not share those mutable resources with another context.

close() is required. VolvoxAI serializes accepted context work and waits for it
before calling close.

### Output snapshots

A successful execution returns every declared graph output exactly once and by
exact name. Host output entries have this shape:

~~~javascript
{
  name: 'logits',
  shape: [1, 32, 8000],
  dtype: 'float32',
  location: 'host',
  data: new Float32Array(values),
}
~~~

Returning the snapshot transfers exclusive ownership of host data to
`ExecutionResult`. A provider must allocate isolated storage and must not retain
or mutate it after returning. `ExecutionResult` validates and adopts that
storage; each public `read()` still returns a fresh caller-owned copy. Supported
dtypes are float32, int32, int8, and uint8.

A device entry provides a result-owned GPUBuffer plus read and release
callbacks:

~~~javascript
{
  name: 'logits',
  shape: [1, 32, 8000],
  dtype: 'float32',
  location: 'device',
  deviceBuffer,
  async read() {
    return copyDeviceBufferToFloat32(deviceBuffer);
  },
  release() {
    deviceBuffer.destroy();
  },
}
~~~

The provider transfers ownership of that snapshot to ExecutionResult. Device
buffers must remain valid across later context executions and context closure,
until release is called during result close. read() must return storage
compatible with the declared dtype and shape.

backendReport may contain only copyable JSON data. A provider with
operatorFallback: 'reported' must set route.operatorFallbackUsed on every
execution and may name route.offendingNode. Decode implementations should
report decode mode, cache generation, and position. VolvoxAI validates and
freezes that evidence into the execution report together with elapsed time,
context identity, device identity, and pinned revisions.

### Provider lifetime

- Runtime owns the provider.
- CompiledModel owns the provider's compiled object.
- ExecutionContext owns the provider context.
- ExecutionResult owns host copies or retained device snapshots.
- Starting close rejects new work and drains already accepted work.
- Provider close occurs only after retained compiled children close.
- Every close implementation must tolerate exactly one call from the runtime;
  resource cleanup within the provider should still be idempotent.

## Native C provider SPI

Include the public headers:

~~~c
#include "volvoxai.h"
#include "volvoxai_backend.h"
~~~

VxBackendProvider supplies explicit runtime, compiled, and context instances:

~~~c
static VxStatus provider_runtime_create(
    void* user_data,
    const VxRuntimeOptions* options,
    void** out_runtime,
    VxReport* report);

static VxStatus provider_compile(
    void* runtime_instance,
    const VxModelSource* source,
    const VxBackendPolicy* policy,
    void** out_compiled,
    VxReport* report);

static VxStatus provider_context_create(
    void* compiled_instance,
    const VxContextOptions* options,
    void** out_context,
    VxReport* report);

static VxStatus provider_context_set_input(
    void* context_instance,
    const char* name,
    VxDataType dtype,
    const void* data,
    size_t byte_size,
    VxReport* report);

static VxStatus provider_context_execute(
    void* context_instance,
    const VxBackendOutputSink* sink,
    VxReport* report);

static VxStatus provider_context_select_adapter(
    void* context_instance,
    uint64_t adapter_id,
    uint64_t adapter_revision,
    const char* package_path,
    const char* version_name,
    VxReport* report);
~~~

Define the complete descriptor, then register it on each runtime that may
select it:

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
    .context_select_adapter = provider_context_select_adapter,
    .context_close = provider_context_close,
    .context_destroy = provider_context_destroy,
};
~~~

Names are canonical lowercase identifiers; cpu is reserved. The runtime copies
the descriptor and name. Callback code, user_data, and everything reachable
from them must remain valid for all handles created from the provider.

The provider owns each returned instance:

- runtime_destroy runs once for each successful runtime_create.
- compiled_destroy runs once for each successful compile.
- context_destroy runs once for each successful context_create.
- context_close is optional logical close and may run before context_destroy.

Partial construction must clean up before returning an error because no
successful instance was transferred.

### Native outputs

context_execute writes every declared output exactly once through the supplied
sink:

~~~c
float output[2] = { first, second };
int64_t shape[1] = { 2 };

return sink->write(
    sink->user_data,
    "logits",
    VX_DTYPE_F32,
    shape,
    1,
    output,
    sizeof(output));
~~~

The public execution dtypes are F32, I32, I8, and U8. Each sink write must
match the descriptor retained from `graph.json` exactly: name, dtype, rank,
every dimension, and byte size. An unknown, duplicate, missing, or mismatched
output rejects the complete result; no partial output snapshot is published.

The sink copies each output before write returns. Names must be unique and
non-empty. The dtype, rank, dimensions, and byte size must agree.

### Composing and selecting the provider

A native host composes the provider with a Runtime, then uses the same
opaque-handle lifecycle as a built-in provider:

~~~c
VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
VxModelSource source = VX_MODEL_SOURCE_INIT;
VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
VxReport report = VX_REPORT_INIT;

VxRuntime* runtime = NULL;
VxModel* model = NULL;
VxCompiledModel* compiled = NULL;
VxExecutionContext* context = NULL;
VxResult* result = NULL;

source.graph_path = "model/graph.json";
source.weight_paths = weight_paths;
source.weight_path_count = weight_count;

vx_runtime_create(&runtime_options, &runtime, &report);
vx_runtime_register_provider(runtime, &provider, &report);
vx_runtime_load_model(runtime, &source, &model, &report);

policy.mode = VX_BACKEND_REQUIRE;
policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
const char* required_backends[] = { "my-npu" };
policy.backends = required_backends;
policy.backend_count = 1;
vx_model_compile(model, &policy, &compiled, &report);
vx_compiled_model_create_context(
    compiled, &context_options, &context, &report);

vx_execution_context_set_input(
    context, "input", VX_DTYPE_F32, input, input_bytes, &report);
vx_execution_context_execute(context, &result, &report);
~~~

`vx_runtime_register_provider` is a provider-host SPI operation, not a
Synurang application call. A Synurang deployment composes providers in its
implementation before accepting calls. Its application controls provider
selection through protobuf `BackendPolicy` and observes provider use through
generated compilation and execution reports.

VX_BACKEND_REQUIRE requires one backend entry. VX_BACKEND_PREFER tries
policy.backends in order; a null list with count zero selects the default
CPU-only preference. Operator-fallback policy is independent from that order.

Check every returned VxStatus in production code. VxReport records stage,
backend, device, reason, message, and execution identity.

Results are independent handles. They remain readable after context release:

~~~c
vx_execution_context_release(context);
vx_compiled_model_release(compiled);
vx_model_release(model);
vx_runtime_release(runtime);

size_t required = 0;
vx_result_read(result, "logits", NULL, 0, &required, &report);
vx_result_read(result, "logits", output, required, NULL, &report);
vx_result_release(result);
~~~

release(NULL) is a no-op. Child handles retain every parent needed for their
operation.

## Android providers

Android integrations should wrap the selected vendor runtime or delegate in
the same provider SPI. NNAPI is deprecated in Android 15; current vendor APIs
can still implement the VolvoxAI runtime/compiled/context ownership boundary
without adding an Android-specific branch to the public runtime.

Use the provider's compile callback to validate the complete semantic contract:
operator, dtype, shape and layout relations, optional operands, attributes,
quantization, output names, and strict fallback policy. Decline compilation
before creating a context when any part is unsupported.
