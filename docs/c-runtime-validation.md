# C runtime validation

The public contract is [proto/volvoxai.proto](../proto/volvoxai.proto);
implementation boundaries are in [ARCHITECTURE.md](../ARCHITECTURE.md).
This guide maps runtime domains to repeatable checks. Exact graphs, expected
values, dtypes, layouts, shape bounds and tolerances belong to the executable
fixtures. An operator name in the registry alone does not establish coverage.

## Validation scope

Check inference and full release profiles independently. Both include the C
inference runtime. Browser WebGPU, training and PTQ checks require full; browser
inference checks also prove the absence of GPU imports and device acquisition.
A fixture result applies to its tested domain and backend. CPU/WebGPU parity
does not establish native GPU coverage or replace an independent numerical
oracle. Remaining qualification work is tracked in [TODO.md](../TODO.md).

| Area | Fixtures and assertions |
| --- | --- |
| Fixed and bounded inference | [Device bridge](../tests/parity/external/webgpu_device_bridge.mjs), [extended inference](../tests/parity/external/webgpu_extended_inference.mjs) and [bounded operators](../tests/parity/external/webgpu_bounded_operators.mjs) compare C CPU and WebGPU execution; selected cases also have independent expected values |
| Typed tensors and layouts | [Storage](../tests/parity/external/webgpu_typed_storage.mjs), [layouts](../tests/parity/external/webgpu_layout_variants.mjs) and [value preflight](../tests/parity/external/webgpu_value_preflight.mjs) check typed execution and rejection before submission |
| Graph domains | [Domain tests](../tests/parity/external/webgpu_graph_domains.mjs) and [CPU route regressions](../tests/c_cpu_route_regression.test.mjs) check shape rules and independent expected outputs |
| Scheduling and batching | [Control paths](../tests/parity/external/webgpu_control_paths.mjs) exercise dynamic bindings, independent batches and required rows; [BatchQueue](../tests/parity/external/batch_queue.mjs) compares dispatch, page and result traces |
| Decode state | [GPU decode](../tests/parity/external/webgpu_decode.mjs), [dependency updates](../tests/parity/external/webgpu_dependency_decode.mjs) and [paged cache](../tests/parity/external/decode_cache.mjs) check numerical results, lane state and retained snapshots |
| Generation | [DecodeGenerate](../tests/parity/external/decode_generate.mjs) checks fixed-step greedy feedback and paged execution |
| Training | [Operator tests](../tests/parity/external/webgpu_training_operators.mjs) compare CPU/GPU losses and updates with sampled finite-difference derivatives; [training smoke](../tests/webgpu_training_smoke.mjs) covers optimizer and lifecycle behavior, including an independent SGD update |
| Trainer persistence and adapters | [Checkpoint continuation](../tests/parity/external/trainer_checkpoint.mjs), [LoRA](../tests/parity/external/lora_api.mjs) and [quantized LoRA](../tests/parity/external/quantized_lora.mjs) check restored state, gradients and exported revisions |
| Text and authoring | [Tokenizer](../tests/parity/external/tokenizer_api.mjs) and [authoring](../tests/parity/external/authoring_api.mjs) exercise generated service calls; comparisons use the pinned [TypeScript references](../tests/parity/reference/README.md) |
| PTQ | [WASM authoring](../tests/ptq_wasm_authoring.test.mjs) and [full host](../tests/ptq_full_engine_host.test.mjs) check calibration, native/WASM package bytes, retained exports and quantized reload execution |

The [ONNX Runtime oracle](../tests/parity/external/README.md) supplies independent
full-output comparisons for its declared source-ONNX cases. Project-specific
quantized semantics use portable-C canonical vectors. Neither suite claims all
operators or all possible tensor domains.

Required unsupported backends return typed refusals. Tests must check route
evidence and numerical results; successful model loading or compilation alone
does not establish execution coverage.

## Lifecycle and device checks

[Host lifecycle tests](../tests/full_host_gpu_lifecycle.test.mjs) verify deferred
device preparation, concurrent preparation and close. Metadata, CPU-only and
invalid requests must not acquire a GPU. Closed host and WASM owners release
their bridge references, including when a device arrives during close.

[Bridge execution tests](../tests/gpu_bridge_execution.test.mjs) cover retained
results, asynchronous shader validation, device loss, staging and mapping
failures, encoder errors and close while work is pending. OOM injection tests
exercise API error handling; they do not measure physical GPU exhaustion.

Use the pinned [Deno runner](../tools/deno/README.md) for device recreation checks.
Its standalone copy and shutdown regressions isolate device lifecycle behavior;
[the composed runtime test](../tests/parity/external/webgpu_composed_runtime.mjs)
also checks CPU/GPU transitions and GPU-to-GPU host recreation through the
minified product. Run with ordinary memory-budget checks and without forced GC
or a Vulkan diagnostic shim. Automatic recovery of a running service is a
separate capability.

## Parse and allocation diagnostics

[measure_graph_parse.py](../tools/measure_graph_parse.py) preprocesses an isolated
diagnostic copy using the recorded inference build recipe, instruments C
parse/node-allocation/malloc calls, and links the ordered numerical objects.
Production sources and artifacts remain uninstrumented.
[measure_graph_parse.mjs](../tools/measure_graph_parse.mjs) uses generated service
calls with a 100-node ReLU graph after LoadModel.

The comparison substitutes a parse of retained source bytes for the lowering
clone. It excludes filesystem reads. Requested bytes count cumulative allocation
requests, not peak live memory; instrumented timings are not release performance
thresholds. Cloning still allocates JSON nodes, and the private engine still
parses the lowered graph once.

## Browser and package checks

[The MV3 check](../tools/test_mv3_packaged_wasm.mjs) loads an isolated Chrome
extension with the actual inference JS and minified full JS plus their WASM
files. CSP is `script-src 'self' 'wasm-unsafe-eval'; object-src 'self'`.
Both profiles execute a model before and after a service-worker restart.
Handles from the first owner must be rejected by the second.

[The package tool](../tools/package_release.py) creates deterministic inference
or full runtime ZIPs. Package checks verify artifact hashes, sorted members,
fixed timestamps and permissions, reproducibility, and reserved output paths.
Model-specific runtime or shader pruning is deferred.

## Running checks

The complete release gate builds the web, WASM and native profiles before
checking their API, provenance, ABI, symbols and sizes:

```sh
make verify_release
```

Additional browser, package and runtime boundary checks:

```sh
node tools/test_mv3_packaged_wasm.mjs
python3 -m unittest tools.tests.test_package_release \
  tools.tests.test_shader_override_contract tools.tests.test_threadless_context_reentry
```

For focused development checks, use the relevant commands from
`package.json`. Build JS before WASM because JS builds remove stale WASM
companions. After building, `npm run check:release` verifies all eight fixed
release files and the inference/full boundaries.

Run physical GPU checks sequentially on an idle GPU. Follow the
[Deno build and regression instructions](../tools/deno/README.md), then exercise
both minified profiles from the repository root:

```sh
for runtime_profile in volvoxai.lite volvoxai; do
  build/deno/target/webgpu-fix/deno run --no-config --unstable-webgpu \
    --allow-read --allow-env --allow-ffi \
    tests/parity/external/webgpu_composed_runtime.mjs \
    --bundle "dist/0.6.0/$runtime_profile.min.js" \
    --wasm "dist/0.6.0/$runtime_profile.wasm" --recreate-gpu-hosts
done
build/deno/target/webgpu-fix/deno run --no-config --unstable-webgpu \
  --allow-read --allow-env --allow-ffi tests/webgpu_training_smoke.mjs \
  --bundle dist/0.6.0/volvoxai.min.js \
  --wasm dist/0.6.0/volvoxai.wasm --require-physical
```

For each run, record the GPU model, driver/runtime versions, source revision,
artifact hashes, fixture selection, commands and results. Use repository-relative
paths in shared reports and omit personal account names, hostnames and home
directories. A passing build or simulated bridge test does not qualify a
physical GPU run.
