# Testing and validation

Choose checks that exercise what you changed. Compilation detects layout,
source-composition, and API-boundary problems; numerical tests establish that a
model or kernel still computes the intended result. A device backend also needs
execution on real hardware before you claim coverage there.

Run commands from the repository root. The [runtime validation map](c-runtime-validation.md)
links each domain to its fixtures; this page explains the development workflow.

## Build the artifacts that tests consume

Many runtime tests use the actual files in `dist/0.4.0/`. Build both profiles
before testing changes to runtime code, generated bindings, or shaders:

```sh
npm ci
npm run build:all
make build_wasm
make build_native_profiles
```

JS must finish before WASM: JS builds remove stale companions. Native/WASM
Make targets use the repository's Docker image. For an all-Docker web build,
use `make build_web` instead of the first three commands.

## Pick a focused check

| Change | Starting checks |
| --- | --- |
| Documentation or examples | Check local links, verify commands against current help, execute changed examples with the matching built profiles |
| Public requests, handles, diagnostics, or transport | `npm run test:proto-api`, `make api_conformance` |
| Graph authoring or SafeTensors | `node --import tsx --test tests/authoring_api.test.mjs` |
| WASM training | `npm run test:wasm-training-smoke` |
| PTQ authoring, calibration, or export | `npm run test:wasm-ptq` |
| GPU device bridge | `npm run test:webgpu-contracts`, followed by relevant physical-device fixtures |
| Native lifecycle and composition | `make test_native` |
| Native ISA dispatch | `make verify_native_isa` |
| Python binding or native library consumption | Install `python/requirements-test.txt`, then `make test_python` |

For operator or backend changes, add or update a correctness case with an
independent expected result. Cover the affected dtype/shape domain and a
meaningful invalid-input case. A test that merely repeats the implementation
or compiles a shader does not establish numerical correctness.

## Schema and generated files

After a schema change, regenerate its C, TypeScript, and Python projections:

```sh
make proto_codegen
make proto_codegen_check SYNURANG_OFFLINE=1
make api_conformance
```

`make proto_codegen_fetch` prepares the pinned generator/runtime caches before
an offline run. `npm run typecheck` checks generated inventories and TypeScript;
the Unicode-table check requires access to its pinned upstream source.
Normal builds consume committed bindings. Do not edit generated API references,
shader outputs, or embedded byte arrays by hand.

## Numerical references and whole models

[External parity](../tests/parity/external/README.md) describes the independent
ONNX Runtime oracle and its source-model cases. Canonical quantized vectors
check project-specific integer semantics. The
[reference fixtures](../tests/parity/reference/README.md) retain selected old
TypeScript implementations for migration comparisons outside product entries.

Use the correct reference for the claim. CPU/GPU agreement can miss a shared
mistake. Optimizer-state self-consistency does not replace an independent AdamW
oracle. Whole-model validation should compare the declared outputs and the
application metric on unchanged inputs, with recorded per-tensor tolerances.

A passing fixture establishes its tested graph, dtype, shape, and backend.
Neither a registered operator nor successful compilation proves every possible
combination. [TODO](../TODO.md) tracks outstanding qualification work.

## Physical GPU checks

Run device checks sequentially on an idle GPU. Record the GPU, driver/runtime,
source revision, artifact hashes, fixture selection, commands, and results.
Use the pinned [Deno runner](../tools/deno/README.md) for the device-lifecycle
checks in [C runtime validation](c-runtime-validation.md#running-checks).

`npm run test:webgpu-contracts` includes simulated bridge failures. These are
useful for error handling and resource lifetimes, but do not measure a physical
GPU's numerical behavior or memory exhaustion. Native CUDA, Vulkan, OpenGL,
and Metal need their own target-device qualification; a WebGPU result does not
cover them.

## Browser deployment and packaging

After building the release profiles:

```sh
node tools/test_mv3_packaged_wasm.mjs
python3 -m unittest tools.tests.test_package_release
```

The MV3 test needs Chrome. It runs actual packaged JS/WASM under the extension
CSP, restarts the service worker, and verifies that previous-owner IDs are
rejected. Package tests check deterministic ZIP members, hashes, timestamps,
permissions, and reserved output paths.

## Inference and full release checks

Before handoff, build and check both profiles. The complete release gate is:

```sh
make verify_release
```

It builds web/WASM/native artifacts, runs the native checks and release-specific
planning/size checks, and verifies inventory, provenance, ABI, and symbol
boundaries. Run the relevant functional tests above as well; the release gate
is not a substitute for every numerical or physical-device test.

For already built artifacts, `npm run check:release` verifies the fixed four
JS files, two WASM companions, and two native executables. Inference must exclude
compiled training code and public training symbols. The browser inference
profile also excludes the GPU bridge and shaders.

Report what passed, what failed, and what was not exercised. Preserve failure
reports and distinguish an unavailable prerequisite from an executed test
failure. [Profiling](profiling.md) explains how correctness checks relate to
latency and memory measurements.
