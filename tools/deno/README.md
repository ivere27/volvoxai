# Deno WebGPU runner

This is an isolated test-runtime build, outside VolvoxAI's eight release files.
It applies two patches to Deno 2.9.6's pinned wgpu dependencies. Product
TypeScript, C, WASM and the public proto API do not depend on these build tools.

These patches address device recreation failures caused by retained empty
allocator blocks and memory-budget checks on heaps not selected for allocation.
The lifecycle regressions below exercise those paths on a physical GPU.

## Build

On Linux x86-64 with Rust 1.97.0, Cargo, a C/C++ toolchain, CMake, Python and
`patch`, run:

```sh
python3 tools/deno/build.py --jobs 3
build/deno/target/webgpu-fix/deno --version
```

The source archives are SHA-256 pinned. Preparation checks every source file
against fresh extraction plus the checked-in patches, and refuses to overwrite
a modified tree. `--prepare-only` verifies sources without building. Use a new
`--build-dir` when changing a patch. `RUSTUP_TOOLCHAIN` selects the installed
Rust toolchain; the script checks its exact version. It does not install a
toolchain or replace a system executable. Allow substantial build disk space;
source, Cargo downloads and build outputs stay under `build/deno/`.

The custom Cargo profile uses optimization level 1, no LTO and no debug symbols
to keep the CLI build practical. It is a correctness runner, not a Deno speed
or binary-size benchmark. `receipt.json` records source/patch/lock/recipe,
compiler and executable hashes. `--offline` requires cached Cargo dependencies
and the rusty_v8 prebuilt archive as well as the source archives.

## Patches

`wgpu-memory-budget.patch` backports
[wgpu #9643](https://github.com/gfx-rs/wgpu/pull/9643) and its
[#9838 follow-up](https://github.com/gfx-rs/wgpu/pull/9838) to wgpu-hal 29.0.3.
It selects the memory type according to allocation location, required flags and
compatible type bits. The prediction also applies this version's
`valid_ash_memory_types` mask, matching the actual allocator call.

`wgpu-device-cache.patch` adds explicit reclamation of **empty** allocator
blocks. Device destruction starts a nonblocking poll. Once submitted GPU work
has finished, wgpu unmaps buffers (including mapped-at-creation staging),
discards never-submitted writes, destroys their temporary
resources, processes deferred destruction and returns empty cached blocks.
Map callbacks use wgpu's deferred completion list and fire after the poll caller
releases registry locks, preserving callback reentrancy.
Objects with live allocations remain valid for their later destruction. Active
devices retain their normal cache behavior. GPU completion remains the condition
for releasing submitted resources.

This does not force native `Device::Drop` while JS objects are still reachable.
Internal device buffers and the VkDevice may remain until ordinary GC. The fix
targets the unused staging pools that caused the reproduced small-heap failure;
it does not promise zero native memory for retained destroyed device objects.
Allocation prediction still estimates resource size; it is not a reservation of
the next native pool block. The 97% allocation threshold and the separate 99%
device-loss check remain enabled and unchanged.

## GPU regression

Run sequentially on an idle physical NVIDIA RTX 3090 using the built executable.
The shutdown regression checks that GPU model explicitly.

```sh
deno run --no-config --unstable-webgpu --allow-read --allow-env --allow-ffi \
  tests/parity/external/webgpu_device_recreation.mjs
deno run --no-config --unstable-webgpu --allow-read --allow-env --allow-ffi \
  tests/parity/external/webgpu_device_shutdown.mjs 10
deno run --no-config --unstable-webgpu --allow-read --allow-env --allow-ffi \
  tests/parity/external/webgpu_composed_runtime.mjs \
  --bundle dist/0.5.0/volvoxai.min.js --wasm dist/0.5.0/volvoxai.wasm \
  --recreate-gpu-hosts
```

Here `deno` must refer to the patched executable, not an arbitrary system Deno.
These tests use ordinary memory budgets and no forced GC or Vulkan loader shim.
The shutdown regression retains destroyed devices/buffers and checks fresh
numerical copies after idle, unsubmitted, submitted, completed and initially
mapped work. The
composed test exercises the actual minified product through generated proto
dispatch. Record the GPU model, driver/runtime versions, artifact hashes,
commands and results without personal account names, hostnames or home-directory
paths. A successful build alone does not establish a passing GPU run.

Source licenses remain in the verified archives: Deno is MIT; wgpu and
gpu-allocator are MIT OR Apache-2.0. Their MIT notices accompany these patches.
The upstream authorship and links above
identify the backported changes. This patch set can be removed when an upstream
Deno release passes the same lifecycle regressions on the target adapter.
