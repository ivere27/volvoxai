# Backend development

A backend prepares a model for a particular device and executes its operators.
Use this guide when adding a native accelerator or working on the browser GPU
path. Applications select available backends during model compilation; backend
implementers supply the resource ownership and execution behind that choice.

The central split is between reusable compiled state (weights and prepared
resources) and per-session state (inputs, scratch, and decode caches). Keeping
those lifetimes separate lets several contexts use one compiled model without
overwriting each other's state. The [architecture](../ARCHITECTURE.md) explains
the complete execution lifecycle.

## Start with an existing provider

Read [host_backend.c](../examples/backend_sdk/host_backend.c) alongside
[volvoxai_backend.h](../native/include/volvoxai_backend.h). Follow one model
through these callbacks before adding device-specific code:

| Stage | What your provider owns or does |
| --- | --- |
| Runtime creation | Initialize the device/driver and record capabilities |
| Compilation | Validate the entire graph/domain and retain immutable resources |
| Context creation | Allocate independent mutable bindings and scratch |
| Execution | Consume one complete input batch and publish every declared output |
| Closure | Finish accepted work and release each ownership scope exactly once |

Start with a small graph and an independent numerical reference. Add a second
context and retain an old result while executing again; this exposes accidental
sharing of mutable storage early. Expand operator and shape coverage only after
those ownership tests pass. The callback contracts below explain the limits.

Native deployment code can compose a provider through the C host SPI.
Web execution uses C/WASM backends and a thin TypeScript GPU device bridge.
Neither JavaScript entry exports a backend provider namespace or a model
compiler/executor callback interface.

## C-owned web execution

Both EngineHost and FullEngineHost own one persistent C/WASM instance.
EngineHost supports WASM CPU only; FullEngineHost also includes WebGPU.
C selects backend candidates, validates the graph/domain, plans operators,
and owns model/context/result handles. Full adds C training and PTQ services.

The WebGPU bridge creates device objects, encodes C-provided commands, submits
them, maps snapshots, and reports completion or device loss. Its private ABI
and shader catalog are generated. Custom device transport supplied through
FullEngineHost's gpuBridge option must implement that same contract; it does not
receive authority to compile graphs or choose numerical execution.

New web backend support belongs in C, its build recipe, and the required
device transport. Application-visible operations and reports remain in the
proto. There is no providers factory option on FullEngineHost.

## Native composition

The source-level host SPI is
[volvoxai_backend.h](../native/include/volvoxai_backend.h). Deployment code
implements VxBackendProvider and calls vx_backend_register_provider before
creating runtimes through generated dispatch. This composes the implementation
behind application operations; applications continue to use generated handles.

Registration copies the descriptor and its name. Callback code and user_data
remain valid for the process lifetime. Each later Runtime creates its own
provider instance. An unavailable optional provider is omitted from ListBackends;
other initialization failures fail runtime creation. Duplicate/reserved names
and invalid descriptors are rejected.

Providers must match the current header's ABI marker, extent, structure sizes,
and callback inventory. These checks identify one exact host contract.
[Native provider fixtures](../examples/backend_sdk/README.md) exercise the
interface. Installable generated-header/library packaging is tracked in
[TODO.md](../TODO.md).

## Ownership and compilation

A provider has three scopes:

```text
provider runtime instance
  compiled instance: immutable executable and invariant resources
    context instance: bindings, scratch, decode state, device commands
```

The compile callback receives graph/domain identities, tensor specifications,
and C's independent-batch evidence. A successful compile attests the exact
supported domain and resource maxima; it cannot promote an unproved graph.
Unsupported dtypes, layouts, shapes, or execution routes must be rejected
before publishing a compiled instance.

Provider implementations own invariant resources at compiled scope and mutable
resources at context scope. The built-in C runtime already shares raw weight
storage and CPU prepared weights; remaining native GPU sharing needs backend
measurements. A capability declaration alone does not demonstrate physical
residency or memory reuse.

The optional compiled_batch_contract/context_execute_batch pair attests one
backend invocation for the aggregate input. It echoes C's graph and proof
identity and may narrow the batch range. Without that pair the provider uses
the single-request route. See [scheduling](scheduling-and-dynamic-batching-design.md).

## Execution and results

Native provider execution callbacks are synchronous and accept HOST bindings
borrowed for the callback duration. VxBackendOutputSink copies every output
before write returns. The provider must produce exact names, dtypes, shapes,
and logical byte lengths; the complete execution is rejected if any output
violates the contract.

The native sink cannot return an unfinished GPU ticket or lend mutable scratch
as a device-resident application tensor. Public WebGPU PENDING results use the
separate C/device-bridge snapshot path. A general result-to-input device lease
would need its own proto contract and implementation.

Optional decode and adapter callbacks expose capabilities through the existing
generated operations. Missing callbacks return typed unsupported reports.
A successful adapter revision change invalidates that context's decode/KV
state; a failed change preserves the prior revision and state.

Pre-commit validation failures preserve bindings and decode state. Failures
after dispatch are not a promise to undo arbitrary device side effects.
Report the correct stage and status, preserve independently retained results,
and validate the supported reset/close behavior.

Device loss fails work on the affected device. The current SPI's batch identity
can describe a device epoch, but this does not implement automatic recovery or
replay. Model/context recreation policy requires explicit qualification.

## Verification

A provider change needs exact-layout validation, graph/domain and batch-proof
tests, numerical references, invalid-input tests, context isolation, retained
result checks, and closure tests. Check inference/full composition and use
physical hardware for device qualification.

Record requested/owned allocation bytes separately from driver residency.
Correctness and memory claims must name the actual graph, dtype, shape, device,
and release artifacts used for the check. Broader failure coverage and resource
accounting remain in [TODO.md](../TODO.md).
