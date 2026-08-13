# Dynamic-shape package migration inventory

This is the DS0 inventory for the breaking dynamic-shape redesign in
[`TODO.md`](../TODO.md). It records where the legacy static
`volvox-graph/v1` contract is authored, generated, validated, tested, and
documented. The intended cutover has no compatibility reader: authored sources
are changed first, generated packages are rebuilt from those sources, and old
documents remain only as explicit rejection fixtures.

Inventory snapshot: commit `e9696b57c4f7`, branch `shape`, 2026-08-01. Counts
describe this worktree and are evidence for migration planning, not permanent
repository invariants.

## Post-cutover qualification update

The inventory below intentionally preserves the DS0 starting state. As of
2026-08-02, the four ignored repository packages have been regenerated through
their owning exporters and pass the canonical bounded-domain package validator.
The EfficientDet variants remain useful constant-only packages in the redesigned
v1 schema. TinyStories is now a bounded dynamic package: dimension `S` has
`min=1`, `max=256`, both public inputs are `[1,"S"]`, all 85 node outputs
preserve `S`, and logits are `[1,"S",50257]`. Its regenerated graph SHA-256 is
`05f6ee21ff9adee2e4e8eb7d0328ae606c2fb1e658be51170f620b8d33c9e76b`.
The reviewed TinyStories parity golden is regenerated with
`node tests/parity/run.mjs golden tinystories_1m`; generated package JSON was
not edited by hand.

## Reproduce the inventory

Run these commands from the repository root. The tracked-file queries avoid
counting ignored model downloads and generated parity output.

```bash
# Tracked persisted graph documents (zero at this snapshot).
git ls-files | rg '(^|/)(graph\.json|[^/]+\.graph\.json)$' | wc -l

# Tracked sources/docs that spell the legacy discriminator (117).
git ls-files -z \
  | xargs -0 rg -l 'volvox-graph/v1' \
  | wc -l

# Tracked code that embeds the legacy split node-output maps (80).
git ls-files -z -- '*.ts' '*.js' '*.mjs' '*.py' '*.c' '*.rs' \
  | xargs -0 rg -l 'outputs_shape|outputs_dtype' \
  | wc -l

# Tracked code that names graph.json or a named subgraph (89).
git ls-files -z -- '*.ts' '*.js' '*.mjs' '*.cjs' '*.py' '*.c' '*.h' '*.rs' \
  | xargs -0 rg -l 'graph\.json|\.graph\.json' \
  | wc -l

# Ignored packages and generated parity documents present in this worktree.
find models -type f \( -name graph.json -o -name '*.graph.json' \) | sort
find tests/parity/out/graphcases -type f -name graph.json | wc -l
find tests/parity/out/opcases -type f -name graph.json | wc -l

# Review every tracked discriminator site, grouped naturally by path.
git ls-files -z \
  | xargs -0 rg -l 'volvox-graph/v1' \
  | sort

# Find shape-schema sites even when the discriminator is imported indirectly.
rg -n 'outputs_shape|outputs_dtype|VOLVOX_GRAPH_FORMAT|GRAPH_FORMAT|GraphDocument' \
  ts tools tests native runtime examples proto \
  --glob '*.{ts,js,mjs,py,c,h,rs,proto}'
```

The second count intentionally excludes `TODO.md`, which names the
discriminator only when describing schema work rather than authoring a
package. The discriminator
search is a sentinel rather than a complete dependency graph: for example,
`tools/export_safetensors.py` imports `GRAPH_FORMAT` instead of spelling its
value, and `tests/parity/ops/cases.mjs` delegates document construction to a
shared authoring helper.

## Persisted packages present now

No `graph.json` or named `*.graph.json` document is tracked by Git. Model
packages under `models/` are intentionally ignored and must never be treated as
authored source. Four ignored packages are present locally:

| Generated package | Inputs | Nodes | Outputs | Shape/quantization characteristics | Authoritative producer |
| --- | ---: | ---: | ---: | --- | --- |
| `models/efficientdet_lite0_fp16/graph.json` | 1 | 263 | 2 | static image and detection extents | `make models_efficientdet` -> `examples/efficientdet_lite0/tools/fetch_model.sh` -> `tools/export_safetensors.py` |
| `models/efficientdet_lite0_fp32/graph.json` | 1 | 263 | 2 | static image and detection extents | same producer |
| `models/efficientdet_lite0_int8/graph.json` | 1 | 268 | 2 | static image/detection extents and affine tables | same producer |
| `models/tinystories_1m/graph.json` | 2 | 85 | 1 | fixed `[1,256]` token/position inputs | `make models_tinystories` -> `examples/tinystories/tools/fetch_model.sh` -> `examples/tinystories/tools/export_gptneo_safetensors.py` |

The three EfficientDet variants cover Conv2D, pooling, reshape/resize, concat,
elementwise, and quantized forms. TinyStories covers Embedding, Add, LayerNorm,
MatMul, GELU, and SDPA. Together they are useful constant-only parity packages,
but they do not exercise symbolic bindings until their family exporters are
given bounded shape profiles.

The current ignored parity output contains 41 graph cases and 15 operator cases
under `tests/parity/out/`. Those files are disposable products of
`tests/parity/graphs/cases.mjs` and `tests/parity/ops/cases.mjs`; migrate the
case definitions and `tests/parity/lib/authorpkg.mjs`, then regenerate output.

## Ownership: authored versus generated

| Artifact class | Ownership | Migration rule |
| --- | --- | --- |
| `models/**/graph.json`, model weights, tokenizer products | Generated and ignored | Change the family/generic exporter, remove the old output directory with the documented model target, and regenerate. Do not patch JSON by hand. |
| `tests/parity/out/**/graph.json` and parity result JSON | Generated and ignored | Change authored parity case definitions/helpers; recreate through the parity fixture commands. |
| Temporary graph packages made by JS/Python/Rust/native tests | Generated during a test from authored fixture code | Rewrite the fixture constructor. Retain an old-layout document only where the assertion verifies its explicit rejection. |
| `dist/`, WASM sidecars, native executables, generated proto/registry projections, shader packs, PTX/embedded arrays | Generated build output | Regenerate only through their owning tools. They are not package-schema sources. Never hand-edit shader or embedded output. |
| `ts/**`, `tools/**`, `native/src/**`, `runtime/src/**`, `proto/*.proto`, example tools, tests, and documentation | Authored | Edit directly in the phase that owns the semantic change. |
| Safetensors payload descriptors | Generated from fixed weights | Weight shapes remain concrete. Update graph/schema association and validators, but do not make weight extents symbolic. |

## Authoritative schema readers and validators

These paths must agree before any package is regenerated:

- `ts/types.ts` defines `GraphDocument`, concrete tensor shapes, and the legacy
  separate `outputs_shape`/`outputs_dtype` maps.
- `ts/core/GraphLoader.ts` owns the JavaScript discriminator, positive-integer
  shape parsing, topology construction, and load diagnostics.
- `ts/core/ModelBuilder.ts` serializes programmatically authored graphs;
  `ts/training/ModelCheckpoint.ts` and `ts/training/Quantization.ts` preserve or
  publish graph documents in the full profile.
- `tools/exporter/runtime_ir.py` is the typed Python importer/serializer.
  `tools/exporter/capabilities.py`, `tools/exporter/ir.py`, and
  `tools/exporter/runtime_tensors.py` enforce concrete static shapes during
  verification and qualification.
- `tools/validate_model_packages.mjs` owns package discovery, canonical
  filenames, strict JSON and safetensors checks, topology, unified output
  descriptors, and affine-table shape checks. It delegates semantic
  verification to `tools/exporter/validate_runtime_package.py`, which proves
  the exact graph, weights, symbols, and operator formulas over the whole
  bounded domain for `cpu-js`, `wasm`, `webgpu`, and `native-cpu` before
  importing RuntimeIR. Warm profiles are never qualification evidence.
- `native/src/runtime/engine_runtime_model.inc` parses and materializes the C
  engine graph. `native/src/runtime/public_api.c` independently validates the
  package root and public model/result descriptors.
- `proto/volvoxai.proto` and `runtime/src/lib.rs` define the FFI tensor
  vocabulary and currently compare concrete request shapes against static model
  descriptors. Generated protobuf projections follow these authored sources.
- `native/src/training/quantization_package.c` is an additional full-profile C
  graph serializer. It must emit the same redesigned schema without creating an
  inference dependency on training.

`ts/index.ts` and `ts/wasm.ts` re-export the JavaScript format constant. Their
profile boundaries must remain unchanged while the value's semantics are
redefined.

## Package emitters and transformation pipelines

### General-purpose emitters

- `tools/export_safetensors.py` emits TFLite packages and is the producer used
  for the three EfficientDet packages. It imports the discriminator from
  `tools/exporter/quantization_storage.py`.
- `tools/exporter/frontend_onnx.py` lowers ONNX directly and writes
  `graph.json`; it currently describes the operation as compiling a static DAG.
- `tools/exporter/runtime_ir.py` serializes optimizer/PTQ RuntimeIR back to the
  legacy input and node-output layout.
- `tools/exporter/optimizer/typed_pipeline.py`, optimizer publication helpers,
  and `tools/exporter/quantization_storage.py` read, transform, or republish
  graph documents and must preserve symbolic descriptors transactionally.
- `tools/exporter/cli.py` and `tools/exporter/optimizer/__main__.py` expose these
  paths to package authors.
- `tools/gen_test_model.mjs` emits the small package used by native CLI tests.
- `ts/core/ModelBuilder.ts`, `ts/training/Quantization.ts`, and
  `native/src/training/quantization_package.c` are non-Python authoring routes
  and therefore require independent cutover coverage.

### Model-family emitters

- EfficientDet ownership is
  `examples/efficientdet_lite0/tools/fetch_model.sh` plus the generic TFLite
  exporter. Its tests are in
  `examples/efficientdet_lite0/tests/test_export_safetensors_tflite.py`.
- TinyStories ownership is
  `examples/tinystories/tools/export_gptneo_safetensors.py`; its static
  256-token policy and tests are under `examples/tinystories/`.
- TinyReceipt split ONNX packages are staged by
  `examples/tiny_receipt_vqa/tools/import_hf_split_onnx.py`, which invokes the
  generic ONNX exporter and validates the closed bounded-active encoder and
  decoder interfaces.
- TinyReceipt package publication is owned by
  `examples/tiny_receipt_vqa/tools/import_hf_split_onnx.py`, for both the FP32
  and producer-authored static INT8 explicit-KV variants. TinyReceipt package
  verification is explicit-KV aware in `verify_split_package.mjs`.

TinyReceipt artifacts normally live under ignored `build/` paths and none is
present as an authored graph document in this snapshot. Re-export them from the
documented cache-enabled source/import pipeline.

## Authored fixture surface

The repository encodes most model fixtures inline rather than checking in JSON.
The following groups require a deliberate rewrite.

### JavaScript core and package tests

- `tests/context_runtime.test.mjs`
- `tests/runtime_graph_loader_cache.test.mjs`
- `tests/runtime_graph_loader_quantization_refs.test.mjs`
- `tests/model_builder.test.mjs`
- `tests/ptq.test.mjs`
- `tests/qmaskedmean.test.mjs`
- `tests/w8a8_graph_e2e.test.mjs`
- `tests/model_package_validation.test.mjs`
- `tests/parity/graphs/cases.mjs`
- `tests/parity/ops/cases.mjs`
- `tests/parity/lib/authorpkg.mjs`
- `tests/parity_artifact.test.mjs`

`tests/model_package_validation.test.mjs` should become the primary package
schema acceptance/rejection corpus. Keep one minimal old-layout fixture there
to prove its exact rejection diagnostic; convert all other positive fixtures
to the redesigned v1.

### Python exporter tests

Seventeen files under `tools/exporter/tests/` spell the old discriminator, and
additional frontend/publication tests inherit it through helpers. The principal
shared constructors occur in:

- `test_runtime_ir.py`, `test_runtime_contracts.py`, and `test_cli.py`;
- `test_capabilities.py` and `test_capabilities_registry_source.py`;
- `test_optimizer_publication.py`, `test_optimizer_registry_resolver.py`, and
  `test_typed_optimizer.py`;
- `test_portable_quantized_validation.py`, `test_quantization_storage.py`, and
  `test_quantized_regions.py`;
- typed layout/PTQ tests (`test_static_qdq_fusion.py`,
  `test_typed_affine_layout_optimization.py`, `test_typed_qdq_layout.py`,
  `test_typed_singleton_transpose.py`, and
  `test_typed_ptq_authoring_pipeline.py`).

Update shared constructors before individual assertions so tests cannot appear
green by continuing to generate the old layout.

### Native, Rust, provider, and example fixtures

- Native inline JSON fixtures are concentrated in `native/tests/test_*runtime.c`,
  `native/tests/test_public_api.c`, `native/tests/test_backend_provider_examples.c`,
  and PTQ/trainer/package tests. Thirteen native test files spell the legacy
  discriminator.
- `runtime/tests/runtime_lifecycle.rs` embeds request/package JSON for the Rust
  FFI lifecycle.
- `examples/backend_sdk/host_backend.c` and
  `examples/backend_sdk/android_nnapi_backend.c` embed provider-side package
  assumptions.
- `examples/native_task_cli/tests/test_cli.py` and
  `tools/tests/test_native_cli.py` create temporary primary and named graph
  documents.
- TinyReceipt has both JS/Python fixtures and native C embedded packages under
  `examples/tiny_receipt_vqa/tests/` and `examples/tiny_receipt_vqa/native/`.
- EfficientDet and TinyStories exporter tests generate temporary packages and
  are the family-level re-export gates.

The broad static-layout sentinel finds 80 tracked code files containing
`outputs_shape` or `outputs_dtype`. Some are validators or consumers rather
than package constructors, but every occurrence must either move to the unified
output descriptor or be documented as an internal concrete bound-plan field
with a different name.

## Documentation surface

Eighteen files under `docs/`, plus root `README.md` and `ARCHITECTURE.md`, spell
the old discriminator. The highest-priority contract documents are:

- `ARCHITECTURE.md`
- `docs/model-format.md`
- `docs/models.md`
- `docs/quickstart.md`
- `docs/browser-runtime.md`
- `docs/native-runtime.md`
- `docs/backend-sdk.md`
- `docs/graph-optimizer-design.md`
- `docs/quantization.md`, `docs/typed-ptq.md`, and
  `docs/training-ptq-runtime-matrix.md`
- `docs/testing.md`

The English and Korean model-production/runtime textbook chapters also contain
literal schema examples. Update both languages in the same documentation
cutover. `docs/operation_list.md`, `docs/w8a8-safetensors.md`, and the
TinyReceipt README include fixed-shape validation or benchmark language even
where the schema example is small.

## Migration order

1. **Lock DS0 semantics.** Land the ADR and architecture ownership first. Freeze
   symbol grammar, deterministic dimension ordering, checked limits, unified
   node-output spelling, and the dynamic-first graph contract.
2. **Change canonical in-memory and persisted contracts.** Update TypeScript
   logical types/loader/builder, Python RuntimeIR, native parser, protobuf/Rust
   tensor specs, and package validators together. Add the single intentional
   legacy-rejection fixture at this point.
3. **Change the generic producers.** Cut over RuntimeIR serialization, the ONNX
   and TFLite frontends, optimizer publication, JS builder/PTQ, and native PTQ
   publication. No producer may emit the redesigned discriminator with legacy
   `outputs_shape`/`outputs_dtype` contents.
4. **Migrate shared fixture constructors.** Change package-validation helpers,
   exporter test constructors, parity authoring helpers, native fixture helpers,
   and Rust lifecycle fixtures before editing one-off expectations.
5. **Migrate example-owned producers and consumers.** Do EfficientDet and
   TinyStories first as bounded-size smoke packages, then TinyReceipt split,
   routed-family, calibration, optimization, native, and browser workflows.
6. **Regenerate, never translate, ignored artifacts.** Recreate all four
   `models/` packages and all parity fixtures from the changed authored sources.
   Regenerate TinyReceipt packages from their original ONNX/checkpoint and
   calibration inputs.
7. **Update documentation atomically.** Replace legacy JSON examples, static
   shape claims, public input examples, provider examples, and both textbook
   languages in the cutover that publishes the new contract.
8. **Validate the complete inventory.** Re-run the commands at the top. The
   decisive checks are the exact closed `volvox-graph/v1` schema, absence of
   legacy output maps, explicit rejection coverage, and regenerated package
   validation.
9. **Record post-cutover baselines.** Compare constant-only and dynamic runs
   against the pre-redesign commands below before enabling the portable release.

Because the format name is intentionally still `volvox-graph/v1`, a global
string replacement cannot prove migration. Closed-root validation and unified
output descriptors are the reliable cutover markers.

## Existing baseline and benchmark commands

These are already versioned in the repository and should be run unchanged before
adding dynamic-shape variants.

### CPU JavaScript

```bash
npm run typecheck
npm run baseline:runtime
npx tsx tools/benchmark_w8a8.mjs --op all --warmup 5 --iterations 50 --json
```

`baseline:runtime` is the strongest existing DS0 harness: it records isolated
run samples and medians, compile and execution reports, one/two-context array
buffer ownership, result snapshot bytes, and lifecycle evidence for a static
Identity graph. Its committed latency reference is
`tests/baselines/runtime_cpu_identity.json`; the current run is summarized in
`docs/dynamic-shape-baseline.md`. The W8A8 command is a parity-checked direct
operator benchmark and reports mean/min/max and buffer footprint, not model
compile or context memory.

### WASM

```bash
make test_wasm_relaxed_simd
make benchmark_wasm_w8a8_seed
make benchmark_wasm_qbatch_matmul
```

For the current TinyReceipt explicit-KV comparison, after importing the FP32
and INT8 packages and building the WASM sidecar:

```bash
python3 -m examples.tiny_receipt_vqa.tools.benchmark_explicit_kv \
  --source "$KV_MODEL_SOURCE" \
  --fp32-package build/tiny-receipt-kv-f32 \
  --int8-package build/tiny-receipt-kv-int8 \
  --warmup 1 --repeat 3 --threads 1 \
  --report examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix.json
```

The Make targets require exact parity before timing representative kernels.
They report average kernel time and exercise memory growth, but do not currently
record generic model compile p50/p95, plan-cache state, or process peak memory.

### WebGPU

Run the portable-versus-packed kernel benchmark on physical hardware:

```bash
node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs \
  --require-dot --warmup=5 --iterations=20
```

The TinyReceipt split-session page can additionally separate package loading,
model compilation, seed, and steady decode:

```bash
npm run build
node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs \
  --timeout-ms=600000 \
  --url='examples/tiny_receipt_vqa/tools/webgpu_w8a8_benchmark_tinyreceipt.html?maxNewTokens=100' \
  --model-dir=build/tiny-receipt-int8-from-fp32 \
  --image="$RECEIPT_VQA_DATA_ROOT/eval/heldout/images/00002.jpg"
```

Use a real hardware adapter for performance evidence; `--adapter=swiftshader`
is correctness-only. The generic harness reports mean dispatch time and parity,
not p50/p95 or GPU allocation high-water.

### Native CPU

```bash
make benchmark_native
```

The target builds the native profiles, runs CTest's `benchmark` label, verifies
portable/dispatched parity, and times the registered kernels. The same
TinyReceipt matrix above compares native C with ONNX Runtime and WASM using an
identical explicit-cache request. To run only that matrix:

```bash
python3 -m examples.tiny_receipt_vqa.tools.benchmark_explicit_kv \
  --source "$KV_MODEL_SOURCE" \
  --fp32-package build/tiny-receipt-kv-f32 \
  --int8-package build/tiny-receipt-kv-int8 \
  --warmup 1 --repeat 3 --threads 1 \
  --report examples/tiny_receipt_vqa/reports/explicit_kv_v1_runtime_matrix.json
```

The microbenchmarks report per-iteration timing; the explicit-KV harness
separates encoder, seed, steady-token, and total decoder execution. Neither is
currently a generic native compile/allocation/peak-memory p50/p95 harness.

## Baseline coverage gaps to close in DS0/DS3

Existing commands are valuable pre-redesign evidence, but only the CPU Identity
harness records compile report plus lifecycle memory ownership. Before DS8, add
versioned, shape-aware harnesses that use equivalent constant and dynamic graphs
and report, per backend:

- compile time, first-specialization time, warm-hit time, p50, and p95;
- logical bytes, capacity bytes, high-water/peak memory, and allocation/grow count;
- specialization hit/miss/eviction and invariant weight-pack/pipeline counts;
- exact or tolerance-based CPU result parity;
- padded maximum versus active batch, sequence, and spatial workloads;
- repeated shape, small -> large -> small, alternating common shapes, and an
  adversarial bounded signature sequence.

Do not overwrite generated packages or benchmark artifacts to fill these gaps.
Add authored harnesses and store reviewed references/reports outside `dist/`,
shader output, and other generated release artifacts.
