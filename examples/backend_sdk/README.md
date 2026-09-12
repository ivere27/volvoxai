# Native backend-provider fixtures

These files exercise the host-composition SPI in
`native/include/volvoxai_backend.h`. They are for provider implementers and
repository conformance tests; applications use the generated service API and
select an already-composed provider by name in `BackendPolicy`.

A descriptor creates three explicit ownership levels:

```text
provider runtime instance
  compiled-model instance
    execution-context instance
```

The compile callback receives the exact graph fingerprint, bounded-domain
proof identity, logical input/output specifications, and independent-batch
evidence. A successful compile must echo those identities, provide conservative
resource maxima, and publish immutable compiled state. Each context owns its
mutable bindings, scratch, decode state, and queue.

Provider descriptors must match the current header's exact ABI marker, extent,
structure sizes, and callback inventory. The numeric ABI version alone is not a
compatibility guarantee.

The host copies the descriptor and name. Callback code and `user_data` remain
valid for the process lifetime because registration has no unregister operation.
Binding names, shapes, and payloads are borrowed only for a synchronous callback.
The output sink copies every declared output before returning, so a provider
cannot lend mutable scratch or an uncompleted device result.

Optional adapter and decode callbacks expose capabilities. Leaving one absent
causes the corresponding generated application operation to return a typed
unsupported report. A batch contract is valid only with a true single backend
invocation for the complete stacked input; a host loop is not batching.

A production provider must:

- validate the complete graph, shape domain, resources, and fallback policy;
- keep immutable resources at compiled scope and mutable state at context
  scope;
- validate a complete execution before mutation or dispatch;
- publish exact output names, dtypes, shapes, and byte lengths once each;
- isolate simultaneous contexts;
- make closure idempotent and destroy every instance exactly once;
- report device identity and typed failure stages.

See the [backend composition guide](../../docs/backend-sdk.md).
