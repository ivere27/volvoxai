# NVTX v3 (core C API only)

Header-only NVIDIA Tools Extension, vendored unmodified from
<https://github.com/NVIDIA/NVTX> tag `v3.2.1`
(archive SHA-256 `737c3035f0e43a2252e7cd94c3f26e11e169f624236efe31794f044ce44a70af`),
`c/include/nvtx3/`. Only `nvToolsExt.h` and the `nvtxDetail/` files it includes
are kept; the CUDA, OpenCL, payload and memory extensions are omitted.
Licensed under Apache-2.0 WITH LLVM-exception (`LICENSE.txt`).

The engine emits NVTX ranges only for operations admitted while a trace with
`TraceOptions.external.annotations` is collecting. Without an attached tool
(`NVTX_INJECTION64_PATH`), each call returns after one pointer check.
