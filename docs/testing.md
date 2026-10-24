# Testing and validation

Choose checks that exercise what you changed. Compilation detects layout,
source-composition, and API-boundary problems; numerical tests establish that a
model or kernel still computes the intended result. A device backend also needs
execution on real hardware before you claim coverage there.

Run commands from the repository root. The [runtime validation map](c-runtime-validation.md)
links each domain to its fixtures; this page explains the development workflow.

## Build the artifacts that tests consume

Many runtime tests use the actual files in `dist/<version>/`, where `<version>`
is `package.json.version`. Build both profiles before testing changes to runtime
code, generated bindings, or shaders:

```sh
npm ci
npm run build:all
make build_wasm
make build_native_profiles
```

JS must finish before WASM: JS builds remove stale companions. Native/WASM
Make targets use the repository's Docker image. For an all-Docker web build,
use `make build_web` instead of the first three commands.

## Updating the release version

The release version comes from `package.json.version`. To change it and update
npm's entry paths and lockfile together, run:

```sh
npm run version:set -- <new-version>
npm run version:check
```

Replace `<new-version>` with the desired release version before running the
command. The standard `npm version <new-version> --no-git-tag-version` command
also runs the synchronization hook. If you edit `package.json.version` directly, run
`npm run version:set` to synchronize the derived metadata. Typechecking checks
that those files agree before building.

Native CMake and the WASM recipe read the version and pass
`-DVOLVOXAI_VERSION="<version>"` to the compiler. Tests, repository tools and
example loaders derive their runtime paths from the same value, so a version
change needs no edits to C sources or those paths. Rebuild JS, WASM and both
native profiles using the commands above, then run `npm run check:release`.

JavaScript shell commands select current release artifacts through the shared
helper, run from the repository root:

```sh
VOLVOXAI_VERSION="$(node tools/release_version.mjs)"
ls "dist/$VOLVOXAI_VERSION/"
```

Python workflows read the same file using the standard library, without Node:

```sh
VOLVOXAI_VERSION="$(python3 -c 'import json; print(json.load(open("package.json", encoding="utf-8"))["version"])')"
ls "dist/python/$VOLVOXAI_VERSION/"
```

Use `dist/<version>/` for release inventories in documentation. Historical
reports and size baselines retain the versions of their measured artifacts;
refresh size baselines deliberately when features change the accepted sizes or
WASM contracts. Changing the version does not reset that gate.

CI checks the shared version reader, tool defaults and fixed release paths with
`node --test tests/release_version.test.mjs tests/release_version_literals.test.mjs tests/webgpu_tool_release_paths.test.mjs`.

## Pick a focused check

| Change | Starting checks |
| --- | --- |
| Documentation or examples | Check local links, verify commands against current help, execute changed examples with the matching built profiles |
| Perfetto viewer | `node --test tests/perfetto_viewer.test.mjs`, then open `examples/profiling.html` in Chrome and compare the node, run, copy/wait and memory tables with an exported trace |
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

For the native CUDA compact decoder, build the CUDA-enabled full shared library
and run on a compatible physical NVIDIA device:

```sh
VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda VOLVOXAI_LIBRARY="$PWD/native/libvolvoxai.so" PYTHONPATH=python python3 -m unittest discover -s python/tests -p test_cuda_compact_gqa.py -v
VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda VOLVOXAI_LIBRARY="$PWD/native/libvolvoxai.so" PYTHONPATH=python python3 -m unittest discover -s python/tests -p test_cuda_compact_decode.py -v
```

The first fixture covers compact attention, head RoPE and head normalization
against independent NumPy references. The second covers contiguous/paged KV,
required rows, inactive slots, residual fusions, retained outputs and chunked
greedy Generate. Both require CUDA with operator fallback forbidden. CPU-only
or inference-profile skips do not count as CUDA qualification. Use the
[compact model comparison](compact-decoder.md#reproduce-the-cached-numerical-check)
for frozen HF model logits and live K/V; it records artifact and route identities
separately from performance and whole-memory evidence.

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

## Execution contexts and generation

Model fixtures must declare `execution` in their graph document. Use an explicit
stateless evaluation contract for a stateless graph; stateful fixtures must name
their initialization, transition, invariant inputs and feedback rules. A missing
contract is an invalid package, including packages produced by test authoring code.

Exercise the public workflow through generated clients: create a graph or
autoregressive context, call `Execute` or `Generate`, consume its completed tensor
packet, and release any owned output buffers. For a debug session, transfer that
packet once with `TakeDebugResult`. Scheduler tests must cover both successful
delivery and completion discarded by cancellation or freshness policy.

```sh
node --import tsx --test tests/debugging.test.mjs \
  tests/decode_debugging.test.mjs tests/gpu_bridge_execution.test.mjs
python3 -m unittest discover -s python/tests -p test_execution_contract.py
ctest --test-dir build/cmake -R test_execution_contract --output-on-failure
```

The C contract cases are also part of `make test_native_invariants`. They cover
atomic input admission, retained output independence, feedback, context memory
limits, cancellation and chunked greedy generation against ordinary graph
execution. Physical device checks remain necessary for device feedback and
asynchronous completion.

The [native sensor replay](native-sensor-replay.md) checks prepared sensor
snapshots, caller clock/freshness policy, ALL feedback and LATEST admission through
the generated native SDK. Run its focused check with either native library:

```sh
PYTHONPATH=python python3 -m unittest discover -s python/tests -p test_native_sensor.py -v
```

Its capture records are numerical evidence. They do not qualify a physical
robot, measured input-rate throughput, latency percentiles or peak memory.
The guide also describes an optional paced CPU companion and its explicitly
selected artifact manifest. It records observed latency, backlog and deadline
misses for prepared inputs; performance comparisons, whole peak memory and
physical control-loop guarantees require separate validation.

## API migration comparisons

For an API migration, preserve each baseline's binaries, matching generated
clients, models, tokenizer and input corpus together. Compare complete vocabulary
logits and logical cache state as well as selected tokens. The TinyStories capture
comparison accepts explicit baseline and candidate directories:

```sh
python3 tools/api_migration_compare_tinystories.py \
  build/api-migration/baseline/tinystories-wasm-full \
  build/api-migration/candidate/tinystories-wasm-full \
  --out build/api-migration/candidate/tinystories-wasm-full-comparison.json
```

This numerical comparison does not qualify performance. Keep full corpus audits,
capture/profiling passes and idle-host timing runs separate. Before running a
comparison, declare its model, dtype, profile, backend and device matrix,
independent numerical tolerances, and performance regression limits in the
versioned validation/performance plan. Keep unsupported and unmeasured
combinations explicit. [TODO](../TODO.md#current-release-qualification) tracks
outstanding qualification work.

The full VQA comparison requires all 2,000 predictions for each dtype and profile,
with exact token, family and EOS agreement. Missing or partial rows remain
incomplete even when the available predictions agree:

```sh
python3 tools/api_migration_compare_audits.py --root build/api-migration \
  --candidate-stage candidate-final \
  --out build/api-migration/candidate-final/audit-comparison.json
```

Keep each candidate stage tied to its artifact manifest. A completed audit of an
earlier build does not verify the final build; preserve it separately and run the
final artifacts against the same frozen corpus.

Before timing, define the matching commands, frozen artifacts, workload metrics
and quiet-host proof in a performance plan. The runner fingerprints these inputs
at registration, runs identical-build A/A controls before ABBA comparisons and
keeps every raw report. Its module header documents the plan fields and report
adapters. Registration refuses to overwrite earlier evidence:

```sh
node tools/api_migration_performance.mjs register /path/to/performance-plan.json
node tools/api_migration_performance.mjs run /path/to/output/preregistration.json
```

TinyStories caller policy, VQA workflows and fixed-count Generate have separate
workers. Generate compares equal new-ID counts including the initial prefill;
its correctness pass measures copy/snapshot counts separately from timing.
Context creation and close are reported apart from the hot request and also
included in a complete context-lifetime metric. Failed A/A, missing idle evidence
and inconclusive confidence bounds cannot qualify performance.

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

## Profiling checks

The [profiling guide](profiling.md) describes the current timing, queue and
memory semantics. Run the focused lifecycle, export and device-bridge tests
against the built artifacts:

```sh
node --import tsx --test tests/profiling.test.mjs tests/webgpu_profiling.test.mjs
python3 -m unittest discover -s python/tests -p test_profiling.py
```

`make test_native` also runs the C collector and CUDA interception tests.
The release-artifact fixtures check numerical outputs together with timing
coverage, clock ranges, queue identities, memory accounting and trace export:

```sh
python3 tools/qualify_profiling.py --backend cpu --output build/profiling-checks
node tools/qualify_profiling.mjs wasm lite build/profiling-checks
node tools/qualify_profiling.mjs wasm full build/profiling-checks
```

For native CUDA, OpenGL or Vulkan, select `--backend cuda`, `opengl` or `vulkan`.
Run each fixture again with `--training` and `--queues` separately to exercise
training and concurrent execution contexts. On Linux, the optional driver
probe checks that disabled collection adds no timing queries, GPU calibration,
CPU/GPU usage or process RSS samples, submissions or waits. It covers the first
execution before any trace and execution after stopping/releasing traces. An
explicit resource snapshot verifies that the sampler probes actually observe
enabled sampling:

```sh
cc -shared -fPIC -O2 native/tests/gpu_profiling_probe.c -ldl -o build/gpu_profiling_probe.so
LD_PRELOAD=./build/gpu_profiling_probe.so python3 tools/qualify_profiling.py \
  --backend cuda --gpu-probe --output build/profiling-checks
```

Use the same probe command with `opengl` or `vulkan`. For physical WebGPU, use
the pinned Deno runner from the physical GPU workflow above:

```sh
build/deno/target/webgpu-fix/deno run --no-config --unstable-webgpu \
  --allow-read --allow-write --allow-env --allow-ffi \
  tools/qualify_profiling.mjs webgpu full build/profiling-checks
```

Deno checks do not establish browser-specific behavior. Qualify deployment in
the target browser separately. Generated reports and traces belong under
`build/`; they are disposable outputs rather than maintained documentation.

To compare disabled execution with another build, retain that build's release
files, `package.json`, `python/volvoxai` and `runtime/generated/python` together.
Pass its root explicitly; mixing clients and binaries from different schemas
does not establish performance compatibility:

```sh
python3 tools/profiling_performance_paired.py --baseline /path/to/baseline \
  --backend native --backend wasm --output build/profiling-performance.json
python3 tools/profiling_training_performance.py --baseline /path/to/baseline \
  --backend native --backend wasm --output build/profiling-training-performance.json
```

These tools also accept `cuda`, `opengl`, `vulkan` and `webgpu`. Run each GPU
backend sequentially. Repeat with `--baseline .` as an identical-build control;
use multiple rounds and account for warmup, runtime compilation and host noise.
For enabled collection, pass `--on` to the `profiling_performance.py` worker or
`on` after the bundle path to `profiling_performance.mjs`. Measure it separately
from disabled-path comparisons.

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

For npm packaging, `npm pack` and `npm publish` run the same gate with
`--web-only` through `prepack`. This checks the six packaged JS/WASM artifacts,
including provenance, ABI and size budgets, without requiring native artifacts
or linker maps. `make build_web` finishes by running the npm publication dry run
inside the build image. `make build_wheel` finishes by running the PyPI dry run,
which checks the wheel's `SHA256SUMS` and Twine metadata. A failed packaging
check fails the corresponding build command.

To repeat these checks against existing artifacts without rebuilding or uploading:

```sh
make publish_npm DRY_RUN=1
make publish_pypi DRY_RUN=1
```

Dry runs require no registry credentials and run the actual local validation
commands, unlike `make -n`, which only prints commands. Registry permissions,
network failures and rejection of an already published version can still fail
the real upload. Without `DRY_RUN=1`, the publication targets upload normally;
`publish_pypi` first builds and qualifies the wheel as usual.

`python3 -m unittest tools.tests.test_publish_targets -v` exercises the Make
publication and build-end checks with isolated tool stubs and no uploads.

Report what passed, what failed, and what was not exercised. Preserve failure
reports and distinguish an unavailable prerequisite from an executed test
failure. [Profiling](profiling.md) explains how correctness checks relate to
latency and memory measurements.
