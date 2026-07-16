# Testing and Validation

VolvoxAI uses Node unit tests, targeted smoke tests, and per-op parity tests.
Run the TypeScript-backed browser/Node suite from the repository root:

```bash
npm test
```

This executes the suite through `tsx --test`, allowing the `.mjs` harnesses to
load the repository's TypeScript sources while those sources retain
JavaScript-compatible ESM specifiers. The generic suite covers graph
construction and editing, CPU autograd, GroupNorm, erf-default and
tanh-approximate GELU, deterministic Dropout, weighted/repeated losses, global
gradient clipping, gradient accumulation and checkpoint guards, plus WebGPU
host/shader contracts. Example-owned tests cover the from-scratch seq2seq,
multimodal training, and TinyReceipt session policies. Real-device backend
checks remain separate as described below.

## JavaScript Build and Package Checks

```bash
npm test
npm run typecheck
npm run build:all
make build_wasm
npm run check:release
npm audit
npm pack --dry-run
node ./bin/volvox.js info
npx tsx -e "import('./ts/index.ts').then(m => console.log(Object.keys(m).sort()))"
npx tsx -e "import('./ts/full.ts').then(m => console.log(Object.keys(m).sort()))"
```

The npm build produces six readable/`.min.js` bundles under
`dist/<package-version>/`: multi-backend inference/full and browser-only WASM
inference-plus-training profiles. Both multi-backend inference variants must
remain free of training exports. `make build_wasm` adds the forward-only and
full training sidecars. After that, `npm run check:release` and
`npm pack --dry-run` should accept exactly eight canonical release artifacts
for the current version, without flat legacy artifacts, local models,
node_modules, native outputs, or shader backups.

## WebGPU Per-Op Tests

`tools/webgpu_op_tests.html` builds small graphs, runs them through the real WebGPU
`GraphExecutor`, and compares readback against `CPUEngine`.

Local SwiftShader/software WebGPU:

```bash
node --experimental-websocket tools/run_webgpu_tests.mjs
```

Android Chrome over adb:

```bash
adb reverse tcp:8091 tcp:8091
adb shell am start -a android.intent.action.VIEW -d 'http://localhost:8091/tools/webgpu_op_tests.html' com.android.chrome
adb forward tcp:9222 localabstract:chrome_devtools_remote
node --experimental-websocket tools/run_webgpu_tests.mjs --cdp=localhost:9222
```

The current suite includes asymmetric-tail W8A8 `QLinear`, padded W8A8
`QConv2D`, and 1/2/3-byte convolution reduction tails with a spatial tile tail
and nonzero input/weight/output zero points, in addition to ReLU, Conv2D, Slice,
Pad, ConvTranspose2D, Where, Gather, ReduceSum, ReduceMean, AveragePool,
DequantizeLinear, Cast, and Expand.
It prints whether `packed_4x8_integer_dot_product` is advertised, so the same
cases validate the portable fallback on older adapters and the DP4a/tiled
kernels on supporting adapters.

Run the matched portable-versus-DP4a W8A8 benchmark on the hardware adapter:

```bash
node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs \
  --require-dot --warmup=5 --iterations=20
```

Use `--adapter=swiftshader` for a software correctness run. The benchmark
checks byte-exact CPU and portable-shader parity before reporting decoder-row
QLinear, cooperative multi-row QLinear, and cooperative QConv2D timings.

## Native Build Smoke

The native C engine is built with CMake and tested with CTest, inside the
pinned Docker toolchain. The Makefile targets are thin wrappers:

```bash
make build_native        # cmake --build: engine binaries, tests, benchmarks
make test_native         # ctest -L native  (engine/kernel/runtime regression)
make verify_native_isa   # W8A8 VPDPBUSD / baseline-ISA codegen checks (x86)
make benchmark_native    # microbenchmarks (prints timings; not a gate)
```

The CMake build compiles WGSL, generates the inference/full embedded packs, and
links the XZ decoder and shader store; profile composition and source lists live
in `native/CMakeLists.txt`. The `verify` checks inspect baseline-built objects
for YMM/ZMM `VPDPBUSD` and assert the public dispatchers stay baseline-ISA.

Tests are grouped by CTest label — `native`, `example`, `verify`, `benchmark`,
`gpu`. List them with `ctest --test-dir build/cmake -N`, or run a subset
directly, e.g. inside the image:

```bash
ctest --test-dir build/cmake -L native -R gemm   # one regression test
ctest --test-dir build/cmake -L 'native|example' # the test_native_all set
```

Generate a small deterministic model:

```bash
npm run build:all
node tools/gen_test_model.mjs /tmp/volvox_model_check
```

Run native and compare with the JS reference:

```bash
./native/volvoxai run /tmp/volvox_model_check \
  --input x=/tmp/volvox_model_check/input.f32 \
  --output /tmp/volvox_model_check/out.f32 \
  --debug
```

The generated files include `expected.json`; compare it with `out.f32` using a small
script or CI helper.

## Native GPU Training Backends

Generate the native shaders, build the Linux backend tests in Docker, and run them
from the repository root with:

```bash
make test_native_gpu
```

That runs `ctest -L gpu`. The Vulkan and OpenGL tests can also be run
individually, e.g. `ctest --test-dir build/cmake -R test_vulkan_training`.
Native callers may allow CPU fallback, while an explicitly GPU-backed gRPC model
uses strict training dispatch and rejects an unsupported GPU backward plan.
Their training suites cover both GELU modes in forward and backward, including
the generated shader parameter layout.

On macOS, configure with the Metal backend (host build, no Docker) and run the
`gpu`-labelled tests — which include the Objective-C `test_metal_training`:

```bash
cmake -S . -B build/mac -DCMAKE_C_COMPILER=clang -DVOLVOXAI_ENABLE_METAL=ON
cmake --build build/mac
ctest --test-dir build/mac -L gpu
```

The macOS CLI build needs Vulkan headers at compile time because the runtime-loaded
Vulkan backend remains in the binary. Set `VULKAN_SDK` when the headers are supplied
by the Vulkan SDK. Metal itself links only the system Foundation and Metal frameworks.

## WASM Build Smoke

```bash
make build_wasm
make test_wasm_relaxed_simd
```

The second target first checks both fixed parents without enabling Relaxed
SIMD, then enables the Node proposal flag and verifies the embedded child's
import/export surface, final dot opcode, shared memory, every I8/U8 operand and
output combination, odd tails, asymmetric zero points, byte parity with the
portable QLinear kernel, and fail-closed rejection. The flagged pass also
reports a decoder-sized M=1 kernel comparison against packed W8A32. Before
those checks, an instrumented baseline-SIMD module proves that the W8A32
`f32x4` path is selected only for the audited symmetric signed-I8 descriptor
envelope. It covers odd K/N, scalar and per-channel metadata, bias/no-bias,
all scalar fallbacks, malformed metadata with no writes, non-dyadic tolerance,
and a maximum-K compensated-cancellation comparison against the portable
double reference. These are kernel tests, not an end-to-end model benchmark.
The release-artifact pass also uses a non-dyadic result sentinel that differs
from the scalar/double value, so a build that silently drops the production
SIMD dispatch cannot pass by timing the fallback.

## Rust Service Tests

```bash
make compile_shaders
cargo test --manifest-path runtime/Cargo.toml
```

This validates the Rust service wrapper and generated FFI boundary tests.
GPU-capable CI should set `VOLVOXAI_REQUIRE_VULKAN=1` and
`VOLVOXAI_REQUIRE_OPENGL=1`; otherwise unavailable hardware is reported and the
corresponding hardware exercise is skipped.

## Known Validation Limits

- WebGPU whole-model multi-output readback is a known gap.
- Exact attention-probability dropout has JavaScript CPU/WebGPU unit coverage
  and native Vulkan/OpenGL wrapper parity vectors. Executing the OpenGL test
  needs a suitable compute driver, and Metal runtime validation requires macOS
  and an Apple GPU. Standalone output Dropout is supported but is not
  mathematically equivalent.
- WebNN behavior depends heavily on browser version, flags, OS, and drivers.
- Native GPU performance claims should be rechecked on the actual target device.
