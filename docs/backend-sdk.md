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

`VOLVOXAI_BACKEND_PROVIDER_VERSION` is the opaque discriminator for the current
exact source contract (its current numeric value is 1), not a compatibility
generation. A matching number does not make a provider built from an older
layout compatible. Build providers against the same VolvoxAI sources and
headers as their host; historical layouts, dual contracts, and compatibility
shims are not supported. Every compiled provider artifact must implement the
current invariant-resource, prepared-route, and dense public-batch contract.

Supply a provider factory under a canonical lowercase name when creating its
owning Runtime:

~~~javascript
import {
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  VolvoxAI,
  createBackendDeviceIdentity,
  createBackendProviderBatchContract,
  createBackendProviderPreparedBatchRoute,
  createBackendProviderCapabilities,
  InvariantResourceStore,
  requireHostExecutionInputs,
} from 'volvoxai';

const myNpuProvider = async ({ name }) => {
    const device = await openVendorDevice();
    const resourceDomain = Object.freeze({ device });
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
        dynamicShapeDomain: 'full',
      }),

      async compile(input, options) {
        const plan = await compileVendorPlan(input, options);
        const invariantResources = new InvariantResourceStore(
          (resource) => resource.allocationBytes,
          (resource) => resource.destroy(),
        );
        invariantResources.define('vendor-plan', plan);
        const independentBatch = input.batchSemantics.supported
          ? 'compiler-proved/v1'
          : 'unsupported';

        return {
          backendName: name,
          invariantResources,
          batchContract: createBackendProviderBatchContract('single-invocation', {
            independentBatch,
            deviceResident: true,
            hostFallback: 'forbidden',
          }),
          compilationEvidence: Object.freeze({
            device: deviceIdentity,
            allocationBytes: invariantResources.ownedBytes,
            batchSemantics: input.batchSemantics,
            shapeDomain: Object.freeze({
              proofProtocol: 'canonical-symbolic-domain-proof/v1',
              resourceProtocol: 'bounded-resource-maxima/v1',
              support: 'full',
              graphFingerprint: input.graphFingerprint,
              proof: input.shapeDomainProof,
              maximumTensorBytes: plan.maximumTensorBytes,
              maximumResidentBytes: plan.maximumResidentBytes,
              resourceLimitBytes: plan.resourceLimitBytes ?? null,
            }),
          }),

          prepareBatchRoute(shapePlan) {
            // The provider/compiler must derive this key from every non-B
            // layout, tactic and executable distinction. A canonical string is
            // stable for queued work and needs no object interner.
            const token = `vendor-plan/v1:${shapePlan.signature}`;
            return createBackendProviderPreparedBatchRoute(
              resourceDomain,
              token,
              invariantResources.deviceEpoch,
            );
          },

          async createContext(contextOptions) {
            const borrowedPlan = contextOptions.invariantResources.borrow('vendor-plan');
            const state = await borrowedPlan.createExecutionState(contextOptions);

            return {
              backendName: name,

              async execute(request) {
                const inputs = requireHostExecutionInputs(request, name);
                const outputs = await state.execute(
                  inputs,
                  request.options,
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
            await plan.drain();
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
device is unavailable. The descriptor uses
`VOLVOXAI_BACKEND_PROVIDER_VERSION`, returns frozen capabilities from
createBackendProviderCapabilities(), and uses a backendName matching the
configured name. createBackendDeviceIdentity() copies non-empty string fields
into frozen serializable metadata. Built-in names are reserved.

A provider factory or already-created provider instance belongs only to the
Runtime whose `providers` option receives it. Runtime closure closes the
initialized provider; there is no process-global provider registry.

### Compilation

compile() receives an immutable logical compile input and:

~~~ts
interface BackendProviderCompileOptions {
  readonly operatorFallback: 'allow' | 'forbid';
}
~~~

The input carries the immutable Model snapshot and Graph together with its
definition, topology revision, weight revision, declared inputs and outputs,
tensor count, node count, graph fingerprint, accepted bounded-shape proof, and
the core-owned typed independent-batch evidence for that exact fingerprint.
A provider may create a private execution Graph through
`input.snapshot.createExecutionGraph()`. It must not retain or mutate the
caller's original Graph or Tensor storage.

Compilation must reject unsupported operators, dtypes, shapes, layouts,
attributes, quantization descriptors, or fallback policy. A provider that
advertises operatorFallback: 'none' promises the complete selected graph stays
on that provider.

The required compilationEvidence object records a provider-reported device
identity, a non-negative allocation-byte total or null, and a frozen
`shapeDomain` attestation tied to the exact compile input proof and graph
fingerprint. A provider claiming independent batching also echoes the exact
compile-input `batchSemantics` object as described below. VolvoxAI validates and
copies this evidence into the immutable compilation report together with
compile time and route evidence; it does not invent a physical-device string.

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

### Invariant resource ownership

Every JavaScript compiled provider object must expose an
`invariantResources: BackendProviderInvariantResourceOwner`. Core captures that
exact frozen owner identity and device epoch during compiled-model validation;
swapping the owner or epoch later fails closed. Core opens one lease before
`createContext()` and supplies it as the required
`BackendProviderContextOptions.invariantResources` member.

The corresponding `BackendProviderInvariantResourceLease` is frozen and
borrow-only. A provider-specific subtype may add typed lookup methods such as
`borrow(key)`, but it must not expose resource creation, replacement, or
destruction. `InvariantResourceStore` is the built-in implementation: the
compiled path uses `define()` or `defineLazy()`, while context code receives
only the lease. A failed producer is never half-published or retried through a
second context.

Materialized resources remain owned when the borrower count reaches zero.
Closing every context and reopening one must therefore reuse the exact
compiled resource rather than copy, upload, or prepack it again. Compiled-model
close waits for context leases, invokes the provider compiled object's close,
and closes the captured resource owner even when provider close reports an
error. Owner close disposes each materialized resource exactly once.

A device-backed owner uses its opaque `deviceEpoch` in every prepared batch
route. Device loss invalidates that owner: the epoch changes, materialized
resources are disposed, and both old leases and new opens fail. Recovery
requires a new provider/compiled generation; invalidating an owner does not
authorize replay of already submitted work.

### Dense batch attestation

Every object returned by `compile()` carries a frozen `batchContract` made by
`createBackendProviderBatchContract()`:

~~~ts
interface BackendProviderBatchContract {
  readonly protocol: 'dense-public-batch/v1';
  readonly densePublicBatch: 'single-invocation' | 'unsupported';
  readonly independentBatch: 'compiler-proved/v1' | 'unsupported';
  readonly deviceResident: boolean;
  readonly hostFallback: 'forbidden' | 'possible' | 'not-applicable';
}
~~~

`single-invocation` means one provider graph invocation over the resolved
`[B, ...]` tensors, rather than a host loop of B graph executions. That graph
invocation may legitimately launch many operator kernels/commands.
`independentBatch: 'compiler-proved/v1'` separately attests that no
operator communicates, reduces or normalizes across request lanes. Core owns
that typed proof: `BackendLogicalCompileInput.batchSemantics` is frozen and
bound to the exact graph fingerprint. A provider claiming
`compiler-proved/v1` must echo that same object as
`compilationEvidence.batchSemantics` and separately guarantee that its backend
really executes B in one invocation. It may claim independence only when the
core evidence says `supported: true`; a leading symbolic dimension by itself is
never evidence. Runtime recomputes the semantic gate as defense in depth. A
provider must not loop over B independent executions and report the loop as a
dynamic batch. `unsupported` keeps the provider usable for direct execution but
prevents Runtime from coalescing independent requests on that route.

A device-resident provider must also say whether host fallback is forbidden or
possible. The built-in WebGPU provider currently attests `deviceResident: true`
and `hostFallback: 'possible'`: its graph runs on the GPU, but scheduled dense
inputs are host-stacked and outputs are read back before lane splitting. It is
therefore not yet a zero-copy, device-resident batching qualification. CPU and
WASM attest a genuine single host invocation with
`hostFallback: 'not-applicable'`. Runtime validates the exact frozen object
before accepting the compiled artifact.

Every compiled provider also implements `prepareBatchRoute(shapePlan)`. It
returns a frozen `BackendProviderPreparedBatchRoute` made with
`createBackendProviderPreparedBatchRoute(resourceDomain, compatibilityToken,
deviceEpoch)`. This is a pure, synchronous, metadata-only attestation boundary:
it must not execute, mutate a context or device, retain request data, or grow an
unbounded per-request cache. Runtime calls it before reserving/copying payload
storage, then reserves exact logical output capacity before provider execution.
Compatibility tokens are provider-owned canonical strings or frozen opaque
objects, never caller grouping labels. Equal tokens must mean the requests have
the same compiled executable, non-B shape/layout/tactic and adapter semantics.
A canonical string avoids an unbounded object interner. If a provider uses
object tokens, it must keep every token stable while an admitted request can
refer to it; FIFO eviction of a live route is invalid. A device reset publishes
a new epoch object.

### Contexts

Each `createContext(options)` call receives the exact compiled invariant lease
described above plus an optional metadata-only initial shape plan and decode
configuration. It returns a distinct mutable execution owner:

~~~ts
interface BackendProviderExecutionContext {
  readonly backendName: string;
  execute(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot>;
  decodeSeed?(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot>;
  decodeStep?(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot>;
  decodeReset?(): Promise<void>;
  close(): Promise<void> | void;
}
~~~

The resolved request contains the exact shaped public inputs, descriptors,
shape plan, adapters, options, and an opaque `deviceInputs` lease map. Host
providers call `requireHostExecutionInputs()` and receive typed arrays keyed by
declared graph input name. A provider must not inspect or retain a public
`TensorResult` or raw caller `GPUBuffer`; only the built-in WebGPU provider can
resolve a lease, and only against its exact physical `GPUDevice`. Device inputs
are currently admitted for ordinary `execute` only. A context may own device
allocations, scratch, command encoders, adapter routing, and decode/KV state.
It must not share those mutable resources with another context.

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
  ownership: 'transfer',
  data: new Float32Array(values),
}
~~~

With `ownership: 'transfer'`, returning the snapshot transfers exclusive host
storage to `ExecutionResult`; the provider must not retain or mutate it. With
`ownership: 'borrowed'`, `ExecutionResult` clones the exact logical storage once
and the provider may retain its original buffer. Each public `read()` still
returns a fresh caller-owned copy. Supported dtypes are float32, int32, int8,
and uint8.

A device entry provides a result-owned GPUBuffer plus read and release
callbacks:

~~~javascript
{
  name: 'logits',
  shape: [1, 32, 8000],
  dtype: 'float32',
  location: 'device',
  logicalSizeBytes: 1 * 32 * 8000 * Float32Array.BYTES_PER_ELEMENT,
  deviceBuffer,
  // Built-in WebGPU snapshots may opt into same-device handoff. These fields
  // are not a public raw-buffer input API and external providers omit them.
  deviceType: 'webgpu',
  device,
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
compatible with the declared dtype and shape. A handoff-capable built-in
WebGPU snapshot is also retained by each accepted consumer until submitted GPU
work reaches its queue fence. Closing the producer waits for accepted reads and
consumer leases; queue failure retires ownership without unsafe early buffer
destruction. Publicly constructed result-shaped objects cannot mint a lease.

backendReport may contain only copyable JSON data. A provider with
operatorFallback: 'reported' must set route.operatorFallbackUsed on every
execution and may name route.offendingNode. Decode implementations should
report decode mode, cache generation, and position. VolvoxAI validates and
freezes that evidence into the execution report together with elapsed time,
context identity, device identity, and pinned revisions.

### Provider lifetime

- Runtime owns the provider.
- CompiledModel owns the provider's compiled object.
- The compiled object exposes one invariant-resource owner; each context owns
  one counted borrow-only lease from it.
- ExecutionContext owns the provider context.
- ExecutionResult owns host copies or retained device snapshots.
- Starting close rejects new work and drains already accepted work.
- Provider close occurs only after retained compiled children close.
- Compiled resources survive zero context leases and are disposed at compiled
  close or terminal epoch invalidation, not at the last context close.
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
    const VxBackendCompileInput* input,
    const VxBackendPolicy* policy,
    void** out_compiled,
    VxBackendShapeDomainAttestation* attestation,
    VxReport* report);

static VxStatus provider_context_create(
    void* compiled_instance,
    const VxContextOptions* options,
    void** out_context,
    VxReport* report);

static VxStatus provider_context_execute(
    void* context_instance,
    const VxTensorBinding* inputs,
    size_t input_count,
    const VxBackendOutputSink* sink,
    VxReport* report);

static VxStatus provider_compiled_batch_contract(
    void* compiled_instance,
    VxBackendBatchContract* contract,
    VxReport* report);

static VxStatus provider_context_execute_batch(
    void* context_instance,
    const VxBackendBatchInvocation* invocation,
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
    .shape_domain = VX_BACKEND_SHAPE_DOMAIN_CAPABILITY_INIT,
    .runtime_create = provider_runtime_create,
    .runtime_destroy = provider_runtime_destroy,
    .compile = provider_compile,
    .compiled_destroy = provider_compiled_destroy,
    .compiled_batch_contract = provider_compiled_batch_contract,
    .context_execute_batch = provider_context_execute_batch,
    .context_create = provider_context_create,
    .context_execute = provider_context_execute,
    .context_select_adapter = provider_context_select_adapter,
    .context_close = provider_context_close,
    .context_destroy = provider_context_destroy,
    .exact_contract_marker = VX_BACKEND_PROVIDER_EXACT_CONTRACT_MARKER,
    .exact_contract_extent = sizeof(VxBackendProvider),
};

provider.shape_domain.support = VX_BACKEND_SHAPE_DOMAIN_FULL;
~~~

Provider, capability, compile, attestation, output-sink, and callback-side
tensor descriptors require exact `struct_size` values. The provider descriptor
also requires the current `exact_contract_marker` and
`exact_contract_extent`; registration validates them before invoking a callback.
The capability
declares the proof and resource protocols; graph shape semantics come from
`volvox-graph/v1` and have no second discriminator.

`VX_BACKEND_ABI_VERSION` follows the same latest-only rule as the JavaScript
discriminator. Its current numeric value is 1, but that number alone never
authorizes an older descriptor: every structure size and required callback
must match the installed header exactly. Recompile the provider whenever the
host contract changes.

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

### Native dense batch attestation

`compiled_batch_contract` and `context_execute_batch` are optional only as a
pair. Omitting both is the explicit B=1 contract. A provider that implements
them returns an exact `VxBackendBatchContract` with protocol
`provider-batch-contract/v1`, stable non-null resource-domain and compiled-route
tokens, a device epoch, one public symbolic batch axis, and legal
min/max/multiple values. Native core supplies a graph-fingerprint-bound
`typed-independent-batch-proof/v1` identity in `VxBackendCompileInput` only
when every typed operator preserves the public request axis. For
`max_batch > 1`, the provider must echo the exact graph fingerprint and proof
identity, keep the same proved axis, and return a range contained by that proof
and the complete shape domain. A provider may narrow core evidence; it cannot
promote an unproved graph by self-attestation.

The Runtime calls `context_execute_batch` once with one mutable context, one
dense stacked binding set and one output sink per logical lane. Every stacked
input has `shape[batch_axis] == batch_size`; request ids are tracing identities,
not grouping keys. The callback must validate the whole invocation before
dispatch and write every output for every lane. Any callback, sink, route or
fallback failure rejects the complete batch and publishes no partial result.
Calling `context_execute` B times is never an implementation of this callback.

The compatibility and resource-domain pointers remain provider-owned and
stable until `compiled_destroy`. They are opaque identities: applications do
not construct or serialize them, and the Runtime compares them structurally
alongside the exact compiled object, non-B shape and device epoch.

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

VxTensorBinding binding = VX_TENSOR_BINDING_INIT;
binding.name = "input";
binding.dtype = VX_DTYPE_F32;
binding.rank = 2;
binding.shape[0] = batch;
binding.shape[1] = sequence;
binding.data = input;
binding.byte_size = input_bytes;
binding.location = VX_MEMORY_HOST;
vx_execution_context_execute(context, &binding, 1, &result, &report);
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
the same provider SPI. Vendor APIs can implement the VolvoxAI
runtime/compiled/context ownership boundary without adding an Android-specific
branch to the public runtime.

Use the provider's compile callback to validate the complete semantic contract:
operator, dtype, shape and layout relations, optional operands, attributes,
quantization, output names, and strict fallback policy. Decline compilation
before creating a context when any part is unsupported.
