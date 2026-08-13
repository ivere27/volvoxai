# VolvoxAI in-process Synurang FFI plugin

`runtime/` builds the optional Synurang plugin that exposes VolvoxAI's complete
full-profile lifecycle through direct calls inside the caller's process.
Synurang's generated `RuntimeService` names identify the in-process dispatch
surface.

`proto/volvoxai.proto` is the sole public application FFI contract. Every
callable FFI operation, request, result, status, stage, policy, tensor dtype,
and full-profile operation is declared there. Generated C, TypeScript, and
Rust bindings are canonical; handwritten code only adapts those messages to
the native implementation ABI.

## Ownership

The FFI follows the public opaque native ownership tree:

```text
VxRuntime
  VxModel
    VxCompiledModel
      VxExecutionContext
        VxResult
    VxTrainer                 full profile
    VxPTQPlan                 full profile
```

An FFI ID names one retained native handle. Children retain every parent they
need, so callers may release a parent ID after creating a child. Closing a
resource is idempotent and rejects new work; releasing removes its ID and drops
the retained handle. A result owns stable output snapshots and remains readable
after its context closes.

Trainer steps mutate only private working weights, gradients, optimizer slots,
accumulation, and RNG state. `CommitTrainer` is the only operation that
publishes a successor Model weight revision. `RollbackTrainer` restores the
last committed baseline, and `ExportTrainerWeights` copies private weights
without publication. Concurrent Trainers use conflict-checked commits.

Each execution context owns its bindings, mutable backend state, decode state,
adapter selection, and operation ordering. Separate contexts may progress
concurrently. Backend selection ends at compilation; execution never retries
on another backend.

## Model and tensor contract

A model source names exactly one `graph.json` plus ordered safetensors shards.
The graph uses the closed bounded-shape v1 root contract; a minimal valid
pass-through document is:

```json
{
  "format": "volvox-graph/v1",
  "dimensions": {},
  "inputs": {"x": {"shape": [1], "dtype": "float32"}},
  "nodes": [],
  "outputs": ["x"]
}
```

The path must name `graph.json` or a named `*.graph.json` document. No alternate
graph filename or discriminator is searched. Legacy split node outputs such as
`outputs_shape` and `outputs_dtype` are rejected; every non-empty node uses an
`id`, unified output descriptors, and an explicit `params` object.

Tensor payloads are little-endian and row-major. Input payloads must use
`MEMORY_LOCATION_HOST` because protobuf bytes are caller-owned host memory.
Set-input handlers exact-match the named descriptor's shape, dtype, and byte
size before the native runtime copies the payload. Output payloads are copied
from an immutable `VxResult` through `vx_result_read`.

`DataType` is the canonical tensor element/storage vocabulary shared by the
protobuf and native contracts. Its full safetensors-compatible enum is not a
claim that every value is executable. Runtime tensors currently accept F32,
I32, I8, and U8; operation handlers reject `UNSPECIFIED` and every unsupported
dtype explicitly.

## Generated contract

Run the pinned Synurang generator and the slim shared-enum generator together:

```bash
make proto_codegen
make proto_codegen_check
```

The generated projections are:

- `runtime/generated/c/`: Synurang C lite/native bindings;
- `runtime/generated/typescript/`: Synurang TypeScript messages and client;
- `runtime/src/gen/`: Synurang Rust dispatcher and plugin trait;
- `native/include/volvoxai_enums.h`: inference-safe native enum projection;
- `native/include/volvoxai_full_enums.h`: full-profile native additions;
- `ts/generated/volvoxaiEnums.ts`: inference-safe TypeScript projection;
- `ts/generated/volvoxaiFullEnums.ts`: full-profile TypeScript additions.

Generated files are never edited by hand. CMake, Cargo, npm type checking, and
CI fail when tracked projections do not match the protobuf schema.

## Implementation boundary

Handwritten Rust has two roles:

- `src/lib.rs` implements the generated plugin trait and owns FFI ID registries;
- `src/abi.rs` declares only the native full-profile ABI used by that adapter.

The native C lifecycle is an implementation layer, not a competing public FFI.
Provider callback contracts are composition SPIs: deployment code supplies
provider implementations before serving application calls. A Synurang caller
does not install executable callbacks. It selects already-composed providers
through protobuf `BackendPolicy`, then controls their use and reads their
reports through generated `RuntimeService` operations.

## Build

```bash
make build_ffi
```

Linux produces `runtime/target/release/libvolvoxai.so`; macOS produces
`runtime/target/release/libvolvoxai.dylib`. On a host without a published
Synurang generator binary, set `PROTOC_GEN_SYNURANG_FFI` to a compatible
executable.

The plugin deliberately composes the native full profile, including Trainer,
PTQ authoring, and full shader assets. That does not alter the fixed native
profile boundary: `native/volvoxai` and `volvoxai.h` remain inference-only,
while `native/volvoxai-full` and `volvoxai_full.h` contain full capabilities.

CUDA remains opt-in:

```bash
make compile_shaders
cd runtime
VOLVOXAI_CUDA_ARCH=86 VOLVOXAI_CUDA_FAST_FP32=0 \
  cargo build --release --features cuda
```

CUDA builds embed deterministic forward and training PTX and load the NVIDIA
Driver API dynamically. They do not link cudart, cuBLAS, cuDNN, or libcuda.

## Shader assets

Release libraries embed generated shader packs. `VOLVOXAI_SHADER_DIR` remains
a development override; the native runtime logs once only when it actually
uses an external override. WGSL templates are authoritative. Generated shader
outputs, packed blobs, PTX arrays, and embedded byte arrays must be regenerated,
never edited.

## Checks

```bash
make proto_codegen_check
npm run typecheck
cargo test --manifest-path runtime/Cargo.toml
make test_native_all
```

Lifecycle tests cover multiple models and contexts, stable results, decode and
adapter isolation, exact input descriptors, idempotent close, required-backend
failure, private Trainer updates, commit conflicts, rollback/export, PTQ plan
ownership, and generated-contract drift.
