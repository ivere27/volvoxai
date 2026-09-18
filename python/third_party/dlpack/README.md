This directory vendors `include/dlpack/dlpack.h` and `LICENSE` from
[DLPack v1.2](https://github.com/dmlc/dlpack/tree/v1.2).

The header defines the standard tensor capsule ABI. It is used only to build
the Python capsule bridge; no DLPack shared library, CUDA SDK, or framework
dependency is added to VolvoxAI. Engine operations remain generated from
`proto/volvoxai.proto`.

The alignment comment omits the upstream list of example frameworks; ABI
declarations and the license are unchanged.
