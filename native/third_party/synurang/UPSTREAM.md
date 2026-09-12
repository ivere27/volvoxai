# Synurang C module and call runtime

These files are unmodified upstream sources from
<https://github.com/ivere27/synurang> at release `v0.8.0`, commit
`53180b484cf7ca07a1e7d6f24e58b8a19a2dcfa8` (module ABI version 1).
License: MIT; see `LICENSE`.

The verified GitHub source archive and regeneration procedure are documented in
[`tools/synurang/README.md`](../../../tools/synurang/README.md).
`tools/generate_proto.py` copies these files from that release archive; its runtime
manifest binds every copied file to a SHA-256 digest.

```text
include/synurang/c_runtime.h
include/synurang/call.h
include/synurang/module_host.h
src/c_runtime.c
src/call.c
src/wasm.c
src/module_host.c
```

The module links `c_runtime.c` and `call.c`; WASM adds `wasm.c`.
`module_host.c` is an optional native host loader, separate from the engine
module. Do not edit these files locally. Upgrade the release pins and
regenerate every projection together, then run `make proto_codegen_check` and
the inference/full release checks.
