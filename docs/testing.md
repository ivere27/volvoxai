# Testing and Validation

VolvoxAI currently uses targeted smoke tests and per-op parity tests. The root
`npm test` script is still a placeholder, so publish/CI validation should call the
commands below directly.

## JavaScript Build and Package Checks

```bash
npm run build:all
npm audit
npm pack --dry-run
node ./bin/volvox.js info
node -e "import('./js/index.js').then(m => console.log(Object.keys(m).sort()))"
```

`npm pack --dry-run` should include the browser package files and not local generated
models, node_modules, native build outputs, or shader backup folders.

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

The current suite covers ReLU, Conv2D, Slice, Pad, ConvTranspose2D, Where, Gather,
ReduceSum, ReduceMean, AveragePool, DequantizeLinear, Cast, and Expand.

## Native Build Smoke

Compile native C:

```bash
clang -O3 -mavx2 -mfma -pthread -I native \
  native/cJSON.c native/safetensors.c native/kernels.c native/quant_cpu_opt.c \
  native/conv_f32_opt.c native/tensor_f32_opt.c native/engine_runtime.c \
  native/engine.c native/image_io.c native/kie_runtime.c native/vulkan_engine.c \
  native/opengl_engine.c native/tokenizer.c native/nnapi_engine.c native/main.c \
  -o /tmp/volvoxai_native_check -lm -ldl
```

Generate a small deterministic model:

```bash
npm run build:all
node tools/gen_test_model.mjs /tmp/volvox_model_check
```

Run native and compare with the JS reference:

```bash
/tmp/volvoxai_native_check run /tmp/volvox_model_check \
  --input x=/tmp/volvox_model_check/input.f32 \
  --output /tmp/volvox_model_check/out.f32 \
  --debug
```

The generated files include `expected.json`; compare it with `out.f32` using a small
script or CI helper.

## WASM Build Smoke

```bash
clang --target=wasm32 -O3 -msimd128 -nostdlib \
  -Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
  -o /tmp/volvoxai_check.wasm native/kernels.c
```

## Rust Service Tests

```bash
cargo test --manifest-path runtime/Cargo.toml
```

This validates the Rust service wrapper and generated FFI boundary tests.

## Known Validation Limits

- Root `npm test` is currently a placeholder.
- WebGPU whole-model multi-output readback is a known gap.
- WebNN behavior depends heavily on browser version, flags, OS, and drivers.
- Native GPU performance claims should be rechecked on the actual target device.

