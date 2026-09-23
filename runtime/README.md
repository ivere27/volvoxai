# Generated API projections

`proto/volvoxai.proto` is VolvoxAI's single public API. This directory holds
the generated projections of that schema plus the consumer-side validation
that is not generated.

```text
generated/c/            C dispatch, server contract, and lite message codec
generated/c/inference/  inference-only C dispatch and message subset
generated/typescript/   TypeScript clients and lite message codec
generated/typescript/inference/  inference-only TypeScript projection
generated/python/       Python clients and lite message codec
typescript/             memory observation validation
```

## Module and call boundary

C `mode=module` generates `volvoxai_ffi.h/.c` typed service callbacks and
`volvoxai_lite.h/.c` message codecs. The engine exports `Synurang_GetApi`, which
creates independent owners and provides open, send, half-close, receive, cancel,
release, poll and destroy. There are no constructors or flattened RPC exports.
Native embeddings serialize ABI entries and post wakeups to their event loop.
`native/cli/call_client.h` demonstrates a blocking command-line consumer; it
contains no application operations. Both shared and static libraries are built
by `make build_native_libraries`. The [Python wheel](../python/README.md) bundles
both shared profiles, the loader and all generated clients. Standalone C SDK
packaging remains separate; the fixed runtime release contains eight files.

`volvoxai_ffi.ts` exposes one Promise client per service over Synurang `Transport`.
`EngineHost` and `FullEngineHost` use the same C module through a private WASM
owner, including path transfers and GPU notifications. `OperationReport` domain
failures reject with `VolvoxAIError`; call failures use Synurang `RpcError`.
Call options support `signal` and `timeoutMs`. Always await host close.

`volvoxai_client.py` supplies sync and asyncio clients over the vendored Python
`ModuleHost` / `AsyncModuleHost`. Applications import `Vx*ServiceClient` and
`Vx*ServiceAsyncClient` from `volvoxai`: these preserve the generated signatures
and raise `VolvoxAIError` for non-OK operation reports after decoding once.
The exception retains the response and typed report; transport failures remain
Synurang `FfiError`. The flat generated module is the raw binding without this
exception policy. The optional native loader is built alongside the engine
libraries. Every language reaches generated C dispatch and uses the same schema;
no host implements engine policy.

Python's `InferenceSession` provides the ordinary NumPy workflow by composing
these generated calls. It selects an available library, discovers a model
package, reuses a compiled execution context and releases results after reading
requested arrays. Model precision and CPU/thread defaults come from C. See the
[Python guide](../python/README.md#run-a-model) for the complete example.

`AsyncInferenceSession` provides the same NumPy workflow through generated
async clients, with safe cancellation and deadline cleanup. Python tensor
metadata is immutable and includes symbolic shape bounds. Native host views
avoid serializing tensor payloads. `quantize` streams calibration batches and
`TrainingSession` handles training, save and checkpoint workflows through the
full profile. These adapters add no engine operations; the proto remains the
public contract.

Both profiles keep their physical codec closure. Full's TS runtime reexports the
common implementation so classes and errors retain one identity. The
`typescript/MemoryEvidenceValidation.ts` and
`typescript/MemoryEvidenceValidatingTransport.ts` files provide development-side
snapshot and compilation-bound validation, outside the product entries.

## Regenerating

```sh
make proto_codegen
make proto_codegen_check SYNURANG_OFFLINE=1
```

`tools/generate_proto.py` downloads the official
[Synurang v0.8.0 release](https://github.com/ivere27/synurang/releases/tag/v0.8.0)
generator and the same tag's source archive. Pinned SHA-256 digests verify both.
`make proto_codegen_fetch` caches both archives under `build/` so subsequent
`SYNURANG_OFFLINE=1` runs need no network access. The release supplies Linux x64,
Linux ARM64 and Windows x64 generators; `PROTOC_GEN_SYNURANG_FFI` can select an
explicit generator on other systems. No Cargo build or adjacent checkout is
used. See [the release integration](../tools/synurang/README.md) for exact pins.

The five generation profiles are C full/inference, TypeScript full/inference,
and Python. C module generation also emits its message codec. Generated outputs
are never edited by hand. Normal CMake/npm builds consume committed projections;
API conformance checks services, handlers and profile boundaries. Offline
regeneration checks generated and vendored files against the same release.

`tools/generate_api_contract.py` also generates profile-specific metadata and
JSON/Markdown references from descriptors and validated `@api` schema comments.
`VxPlatformService.DescribeApi` reads these tables through normal generated
dispatch. Native builds verify their source/output manifest without requiring
protoc; `make proto_codegen_check` regenerates and compares the complete output.
