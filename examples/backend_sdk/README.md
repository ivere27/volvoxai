# Native backend-provider examples

These examples include only the public headers volvoxai.h and
volvoxai_backend.h. They demonstrate the instance-oriented
VxBackendProvider contract.

`VX_BACKEND_ABI_VERSION` is 1. Provider descriptors require exact
`struct_size` values.

A provider descriptor creates three explicit ownership levels:

~~~text
provider runtime instance
  compiled-model instance
    execution-context instance
~~~

The runtime calls compile once with a VxBackendCompileInput and
VxBackendPolicy. The compile input carries the snapshotted source, exact graph
and proof identities, and complete logical input/output specifications. A
successful compile fills VxBackendShapeDomainAttestation with the matching
identities and conservative resource maxima. context_create then makes
isolated mutable state for each request stream. context_execute receives one
complete, atomic host binding batch and writes every declared output exactly
once through VxBackendOutputSink.

Create a Runtime, then register the descriptor before loading or compiling a
model that may select it:

~~~c
VxBackendProvider provider = {
    .struct_size = sizeof(VxBackendProvider),
    .abi_version = VX_BACKEND_ABI_VERSION,
    .name = "example-host",
    .user_data = &driver,
    .shape_domain = {
        .struct_size = sizeof(VxBackendShapeDomainCapability),
        .proof_protocol = VX_BACKEND_SHAPE_PROOF_PROTOCOL,
        .resource_protocol = VX_BACKEND_RESOURCE_PROTOCOL,
        .support = VX_BACKEND_SHAPE_DOMAIN_FULL,
    },
    .runtime_create = example_runtime_create,
    .runtime_destroy = example_runtime_destroy,
    .compile = example_compile,
    .compiled_destroy = example_compiled_destroy,
    .context_create = example_context_create,
    .context_execute = example_context_execute,
    .context_select_adapter = NULL,
    .context_close = example_context_close,
    .context_destroy = example_context_destroy,
};

VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
VxReport report = VX_REPORT_INIT;
VxRuntime* runtime = NULL;
VxStatus status = vx_runtime_create(&runtime_options, &runtime, &report);
if (status == VX_STATUS_OK) {
    status = vx_runtime_register_provider(runtime, &provider, &report);
}
~~~

Select it through ordinary model compilation:

~~~c
VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
policy.mode = VX_BACKEND_REQUIRE;
policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
const char* required_backends[] = { "example-host" };
policy.backends = required_backends;
policy.backend_count = 1;

VxCompiledModel* compiled = NULL;
status = vx_model_compile(model, &policy, &compiled, &report);
~~~

VX_BACKEND_REQUIRE uses exactly one backend entry. VX_BACKEND_PREFER tries the
listed entries in order.

The descriptor and name are copied. Callback code and user_data must remain
valid until every handle created from the provider has been released.

Binding descriptors, names, shapes, and data are borrowed only until the
synchronous execution callback returns. Only HOST bindings are accepted.
The provider must validate the complete domain/resource transition before it
mutates context state; a pre-commit failure preserves the prior bindings and
decode state. Dispatch or device-loss failures after commit do not imply a
portable rollback guarantee.

The output sink copies each output before write returns. A provider therefore
cannot lend a mutable scratch buffer or device pointer to VxResult. If a
provider needs asynchronous device work, it completes or snapshots that work
before publishing the output.

Providers that support adapter revisions implement context_select_adapter.
The callback receives the exact adapter identity/revision pinned by the
context; leaving it NULL makes non-base adapter selection fail explicitly.

The Android example shows where an NNAPI-backed implementation fits. Android
15 deprecates NNAPI, so new deployments should expose QNN, LiteRT delegates,
or another vendor runtime through the same provider seam.

A production provider should:

- validate the whole graph and complete logical tensor domain during compile;
- attest the exact graph/proof identities and bounded resource maxima;
- fail unsupported required routes before context creation;
- keep device/runtime state in the matching provider instance;
- keep mutable bindings and queues in each context instance;
- write all declared outputs with exact names, dtypes, shapes, and byte sizes;
- make close idempotent and destroy each instance exactly once;
- report a device identity through VxReport when the provider can identify it,
  together with the failure stage.

The complete contract is in
[Backend SDK](../../docs/backend-sdk.md).
