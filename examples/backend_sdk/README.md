# Native backend SDK example

`host_backend.c` is a minimal backend that includes only the public
`volvoxai_backend.h` contract. It handles exact-shape F32 `Add` and declines
broadcasting, fused activation, and every other node, leaving CPU as the
correctness fallback. `Add` is deliberately non-elided by the default graph
optimizer, so the example callback is exercised in a normal configuration.

Compile it into an embedding application together with VolvoxAI, then select
it before loading a model:

```c
int volvoxai_example_host_backend_register(void);

if (volvoxai_example_host_backend_register() != 0 ||
    volvoxai_engine_configure_backend("example-host") != 0) {
    /* registration error or device unavailable */
}
```

A vendor backend normally keeps the opaque node/tensor discovery code and
replaces the host `memcpy` with driver compilation/submission. If outputs stay
on the device, set `VX_BACKEND_FLAG_DEVICE_RESIDENT_OUTPUTS` and implement the
paired `mark_host`/`sync_host` callbacks. The full ABI and lifecycle contract
are documented in [Backend SDK](../../docs/backend-sdk.md).

## Android NNAPI compatibility example

`android_nnapi_backend.c` is a public-header-only Android example. It registers
as `android-nnapi-add`, leaving the built-in `VOLVOXAI_BACKEND_NNAPI` enum and
the existing `--nnapi` command-line option unchanged:

```c
#include "android_nnapi_backend.h"

if (volvoxai_example_nnapi_backend_register() != 0 ||
    volvoxai_engine_configure_backend("android-nnapi-add") != 0) {
    /* registration error or no NNAPI device */
}
```

The `test_backend_sdk` CTest case compiles this example
(`android_nnapi_backend.c`) into the native backend-SDK test on any host, so a
plain `make test_native` exercises it. On a non-Android host, the source still
compiles and its backend reports `VX_INIT_UNAVAILABLE`; the native SDK test uses
that path to verify explicit-selection policy. For an Android build, cross-compile
with the NDK CMake toolchain (see `native/CMakeLists.txt`).

The example intentionally handles only rank 1–4, exact-shape, contiguous F32
`Add` with no fused activation. Broadcasting, scalar tensors, quantized types,
fused ReLU, and every other operator are declined to the CPU fallback. It
creates and compiles a small NNAPI model for every handled node invocation and
writes the result to host storage, making it readable rather than a performance
backend. A production delegate should cache validated compilations and add
device-resident coherence hooks.

Backend ABI v1 is also whole-node only. VolvoxAI does not offer this example to
the SDK registry during training, adapter-modified Linear, legacy QTensor, or
prefix/row-windowed execution; built-in routes and CPU keep those semantics.

NNAPI itself is deprecated in Android 15. This file demonstrates how a legacy
Android accelerator can live outside core dispatch; new devices should use the
same named SDK seam with their vendor runtime or delegate.
