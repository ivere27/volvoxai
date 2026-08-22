# Testing and validation

VolvoxAI validation follows the public ownership boundary:

- Runtime, Model, CompiledModel, ExecutionContext, and
  ExecutionResult lifetime.
- same-model and two-model context isolation.
- interleaved execution and decode state.
- stable host and device result snapshots.
- immutable revision pinning and atomic training publication.
- required/preferred provider policy and operator-fallback evidence.
- inference/full build and symbol separation.
- portable-versus-accelerated operator correctness.

## JavaScript checks

Run the TypeScript-backed suite from the repository root:

~~~bash
npm run typecheck
npm test
~~~

The test runner uses tsx with Node's test harness, so .mjs tests can import the
TypeScript sources while those sources keep JavaScript-compatible ESM
specifiers.

Focused runtime tests cover:

- caller Graph and Tensor immutability.
- context-local execution and decode state.
- FIFO operation and close behavior.
- CompiledModel and Runtime retention by contexts.
- stable named outputs after later executions and parent closure.
- host read copy ownership and WebGPU result-buffer lifetime.
- compilation and execution reports.
- required backend and forbidden operator-fallback errors.
- provider contract validation and cleanup after invalid output.
- Trainer publication, accumulation-only steps, rollback, and close.

Record reproducible CPU lifecycle evidence with:

~~~bash
npm run baseline:runtime
~~~

The JSON report includes median one-context latency, one/two-context mutable
storage, stable snapshot bytes, post-close storage, and compilation/execution
identity and route evidence. Its default five isolated processes, each with 10
warmups and 51 samples, avoid treating startup or one noisy scheduling interval
as the latency result. The gate uses the median of those five run medians. The
default workload is checked against `tests/baselines/runtime_cpu_identity.json`;
the command fails
when its median exceeds the committed reference by more than 5%, when a memory
budget is exceeded, or when lifecycle evidence fails. A reference records its
Node major version, platform, architecture, and exact workload, and a mismatch
is an error rather than an unverified pass. Use
`--reference=/path/to/reference.json` only to run the same gate against another
reviewed reference environment.

Graph and operator tests cover construction/editing, strict graph.json parsing,
portable quantization, CPU autograd, GroupNorm, GELU modes, deterministic
Dropout, weighted/repeated losses, global gradient clipping, gradient
accumulation, checkpoints, WebGPU dispatch, and example-owned model policies.

## Build and package gates

~~~bash
npm run build:all
make build_wasm
make build_native
npm run check:release
npm pack --dry-run
~~~

The fixed release set must contain exactly:

~~~text
dist/<package-version>/volvoxai.js
dist/<package-version>/volvoxai.min.js
dist/<package-version>/volvoxai.full.js
dist/<package-version>/volvoxai.full.min.js
dist/<package-version>/volvoxai.wasm.js
dist/<package-version>/volvoxai.wasm.min.js
dist/<package-version>/volvoxai.wasm
dist/<package-version>/volvoxai.full.wasm
native/volvoxai
native/volvoxai-full
~~~

The standard JavaScript bundles and volvoxai.wasm must contain no training
implementation or training exports. The full artifacts contain the Trainer and
training/PTQ implementations. The WASM-only JavaScript bundle contains no CPU,
WebGPU, WGSL, or Node filesystem implementation.

Validate model packages separately:

~~~bash
make validate_model_packages
~~~

This rejects alternate graph filenames and requires the exact
volvox-graph/v1 discriminator in every graph root.

## WebGPU correctness

tools/webgpu_op_tests.html builds small graphs, runs them on a physical or
software WebGPU adapter, and compares readback with the portable CPU result.

Local software WebGPU:

~~~bash
node --experimental-websocket tools/run_webgpu_tests.mjs
~~~

Local surfaceless Vulkan WebGPU on a physical adapter:

~~~bash
node --experimental-websocket tools/run_webgpu_tests.mjs --adapter=hardware
~~~

Android Chrome over adb:

~~~bash
adb reverse tcp:8091 tcp:8091
adb shell am start -a android.intent.action.VIEW \
  -d 'http://localhost:8091/tools/webgpu_op_tests.html' com.android.chrome
adb forward tcp:9222 localabstract:chrome_devtools_remote
node --experimental-websocket tools/run_webgpu_tests.mjs \
  --cdp=localhost:9222
~~~

The suite includes asymmetric-tail W8A8 QLinear, padded W8A8 QConv2D,
one/two/three-byte reduction tails, nonzero input/weight/output zero points,
ReLU, Conv2D, Slice, Pad, ConvTranspose2D, Where, Gather, reductions,
AveragePool2D, DequantizeLinear, Cast, and Expand.

Run the matched portable-versus-packed-dot benchmark on hardware with:

~~~bash
node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs \
  --require-dot --warmup=5 --iterations=20
~~~

The benchmark checks byte-exact CPU and portable-shader parity before reporting
decoder-row QLinear, cooperative multi-row QLinear, and QConv2D timings. Use
--adapter=swiftshader for software correctness, not a hardware performance
claim.

Whole-model WebGPU parity must read named ExecutionResult outputs. Tests cover
all declared output kinds, including graph-input, weight, and transitive
Dropout/identity aliases, and verify that context close does not invalidate a
live result.

## Native checks

The CMake build compiles WGSL, generates inference/full embedded shader packs,
and builds both native profiles. Generated shader outputs and embedded arrays
are build products.

~~~bash
make build_native
make test_native
make test_native_all
make verify_native_isa
~~~

The inference profile must compile no training implementation and export no
training symbol. Native public API tests cover opaque Runtime, Model,
CompiledModel, ExecutionContext, and Result handles; retain/release behavior;
same-model and two-model execution; stable result reads; provider instances;
and report/status handling. The provider suite rejects unknown, duplicate,
missing, wrong-dtype, wrong-rank, wrong-shape, wrong-byte-size, and F16 output
writes before a Result is published. Built-in coverage includes graph-input and
standalone safetensors outputs, including F16 weight storage widened to F32 at
the execution boundary.

List or filter CTest cases with:

~~~bash
ctest --test-dir build/cmake -N
ctest --test-dir build/cmake -L native -R context
ctest --test-dir build/cmake -L 'native|example'
~~~

Generate a deterministic package for a manual native smoke:

~~~bash
npm run build:all
node tools/gen_test_model.mjs /tmp/volvox_model_check

./native/volvoxai run /tmp/volvox_model_check \
  --input x=/tmp/volvox_model_check/input.f32 \
  --output /tmp/volvox_model_check/out.f32 \
  --debug
~~~

The generated expected.json records the reference result.

## Native GPU checks

Run configured Linux GPU backends with:

~~~bash
make test_native_gpu
~~~

This runs the gpu-labelled CTest set. Vulkan, OpenGL, CUDA, and Metal cases must
compare accelerated values with the portable CPU implementation. A required
physical device that is unavailable is reported as unavailable, not counted as
a correctness pass.

CUDA is opt-in:

~~~bash
cmake -S . -B build/cuda -DCMAKE_C_COMPILER=clang \
  -DVOLVOXAI_ENABLE_CUDA=ON -DVOLVOXAI_CUDA_ARCH=75
cmake --build build/cuda --target \
  test_cuda_kernels test_cuda_runtime test_cuda_public_dynamic \
  test_cuda_state_ownership test_cuda_training test_training_backward \
  test_dynamic_autograd test_cuda_ptq
ctest --test-dir build/cuda --output-on-failure \
  -R '^(test_cuda_(kernels|runtime|public_dynamic|state_ownership|training|ptq)|test_training_backward|test_dynamic_autograd|cuda_source_composition)$'
~~~

Strict CUDA builds disable FMA contraction. Performance experiments may set
VOLVOXAI_CUDA_FAST_FP32=ON, which changes FP32 rounding and therefore requires
separate evidence.

CUDA tests cover F32 and W8A8 forward kernels, coherence, row views, strict
route rejection, staged attention backward, deterministic dropout, gradient
accumulation, clipping, SGD, AdamW, optimizer resume, private backward planning, and
PTQ authoring. A detected device with module, JIT, or execution failure is a
test failure.

On macOS:

~~~bash
cmake -S . -B build/mac -DCMAKE_C_COMPILER=clang \
  -DVOLVOXAI_ENABLE_METAL=ON
cmake --build build/mac
ctest --test-dir build/mac -L gpu
~~~

Metal runtime validation requires macOS and an Apple GPU.

## WASM checks

~~~bash
make build_wasm
make test_wasm_relaxed_simd
~~~

The test verifies both baseline parents without optional instructions, then
checks the embedded Relaxed-SIMD child when enabled. It covers import/export
surface, shared memory, I8/U8 combinations, odd tails, asymmetric zero points,
portable-kernel parity, and fail-closed fallback. Separate tests verify strict
full-sidecar training and that the inference sidecar exposes no training
symbol.

## In-process Synurang FFI checks

~~~bash
make compile_shaders
cargo test --manifest-path runtime/Cargo.toml
~~~

The plugin must use the generated `proto/volvoxai.proto` contract, public
opaque C handles, and canonical `graph_path` field. Set VOLVOXAI_REQUIRE_VULKAN=1 or
VOLVOXAI_REQUIRE_OPENGL=1 when hardware execution is required.

## Parity gates

Before release handoff, run:

~~~bash
make parity
make parity_backward
make parity_decode
make parity_kvcache
~~~

Parity manifests record provider and provider-reported device identity when
available, graph and weight revisions, context identity, tier selection,
operator route evidence, and stable result evidence.

## Interpreting limits

- Software WebGPU validates semantics but not physical GPU performance.
- OpenGL and Metal runtime checks require suitable target hardware.
- Native GPU performance must be measured on the deployment device.
- A one-step or resume test validates state transitions, not convergence.
- Numeric goldens change only for an intentional reviewed numeric or shape
  change.
