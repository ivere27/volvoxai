# VolvoxAI architecture

VolvoxAI is one engine with two capability profiles:

- `inference`: forward execution only.
- `full`: inference plus training.

The browser release also provides a backend-restricted composition of the
full profile: `volvoxai.wasm.js` exposes only WASM execution and omits the CPU,
WebNN, WebGPU, and WGSL implementations. It retains strict WASM training and a
typed wrapper over the stateless C PTQ primitives, but omits the filesystem and
package-authoring parts of the native PTQ runtime. It is a deployment boundary
for extensions, not a third numerical backend.

Release artifacts are intentionally flat. Source code is grouped by ownership,
not by release packaging.

## Source map

TypeScript source (bundled to JavaScript release artifacts):

```text
ts/
  core/       inference graph/data objects and model-agnostic orchestration
  ops/        reusable operators plus graph validation and normalization
  backends/   public backend/decode contracts plus CPU, WASM, WebGPU, and WebNN resources
  training/   training graphs/builders, autograd, optimizers, checkpoints, and PTQ authoring
  index.ts    inference entry
  full.ts     inference plus training entry
  wasm.ts     WASM-only inference, strict training, and stateless PTQ entry
```

Model-specific sessions belong under `examples/`. JavaScript wrappers may use
the public inference API, but `ts/index.ts`, `ts/full.ts`, and files reachable
from those entries must not import or export them. Native wrappers live under
`examples/<model>/native/` and are not linked into either fixed native binary.

Native:

```text
native/
  include/      flat public C APIs for inference, backend integration, training, and tokenization
  src/shader_store.* compressed embedded-asset lookup shared by GPU backends
  src/runtime/  model state, graph, memory, backend registry/SDK bridge, and execution
  src/tokenization/ opt-in application tokenizer implementation
  src/kernels/  portable and optimized CPU/WASM kernels
  src/backends/ Vulkan, OpenGL, Metal, and NNAPI integrations
  src/training/ backward planning, losses, accumulation, train steps, and full-profile PTQ/package authoring
  cli/          fixed model-agnostic command-line application
  tests/        native unit and integration tests
  third_party/  vendored dependencies with provenance and local-config notes
```

The Rust service under `runtime/`, together with `proto/volvoxai.proto`, is an
optional downstream embedding application. It is not linked into the browser
package or either fixed native executable. Its task-convenience RPCs may own
application policy such as streaming generation or teacher-forcing transforms;
they must reach the engine through the public C API and must not make that
policy part of the core inference profiles.

WGSL under `shaders/` is authoritative source shared by browser WebGPU and the
native shader compiler. Generated native shaders and embedded byte arrays are
build outputs and are never edited by hand.
WGSL modules whose first line is `// @volvoxai-browser-only` remain
authoritative browser source but are intentionally excluded from native shader
compilation and packs; the compiler and packer consume that same source marker.

The small public C surface stays directly under `native/include/`. A nested
`include/volvoxai/` namespace becomes useful for an installed SDK with many
headers, but currently adds ceremony without resolving ambiguity.
Likewise, keep `ts/ops/` and `ts/backends/` shallow until a real second portable
kernel family requires another level.

## Vocabulary

- **Operator**: the semantic graph operation, such as `MatMul`.
- **Kernel**: one implementation of an operator for a dtype or target.
- **Backend**: device integration and resource ownership for CPU, WASM, WebGPU,
  WebNN, Vulkan, OpenGL, Metal, or NNAPI.
- **Executor**: schedules graph nodes and dispatches their implementations.
- **Runtime**: owns the loaded model, tensors, memory, and forward execution.
- **Trainer**: owns gradients, optimizer state, and accumulation around a runtime.

## Dependency rules

1. Inference entry points never compile, import, or export training code.
2. TypeScript core owns inference graph/tensor data, model loading and generic
   authoring, tokenization, adapters, and runtime orchestration. Core objects do
   not own optimizer/training state, initializers, backend resource handles, or
   packaged-model sessions.
3. Reusable computation and model-independent graph contracts belong in
   `ts/ops/`. `GraphLoader` owns package-loading orchestration and delegates
   portable quantized validation and operator layout normalization there.
4. Backends own device resources. In particular, GPU buffers are executor or
   backend state keyed by tensor identity; they are not properties of the core
   `Tensor` data model. Out-of-tree native backends consume only the public
   opaque node/tensor ABI; internal `Node`, `T`, and JSON storage never cross
   that boundary.
5. Training may depend on core runtime, operators, kernels, and backend
   interfaces. Training graph state, optimizer state, initializers, and
   training-only builder helpers and PTQ artifact authoring stay under
   `ts/training/`. They are reachable only from full-capability entries:
   `ts/full.ts` exposes every trainer and PTQ tool, while `ts/wasm.ts` exposes
   the strict WASM trainer, stateless C-backed PTQ primitives, and
   backend-neutral training data APIs. Graph calibration orchestration and
   safetensors/package materialization remain full-entry authoring tools.
6. Model-family graph constructors, preprocessing, vocabulary policy,
   generation loops, and session wrappers belong under `examples/` (or in
   downstream applications), never in either JavaScript package entry or either
   fixed native CLI artifact.
7. Blueprint node inputs must resolve to an explicitly declared graph input, a
   loaded weight, or an earlier node output. Loaders never invent an input or a
   model-specific default shape.
8. A backend never depends on another backend.
9. Kernels never depend on CLI, model-building, checkpoint, or session code.
10. CLI code uses public APIs; it does not reach into internal globals.
11. TypeScript inference entry points contain no training imports, including
   dynamic imports.
12. Optional features are composed at entry/source boundaries. Native training
   uses guarded private implementation fragments composed only in the full
   profile because separate translation units would expose broad runtime-static
   state. The inference profile compiles no training implementation or public
   training symbol, and feature checks do not spread through individual kernels.
13. External shader files are a development override. Release execution falls
   back to the embedded store, and logs once when `VOLVOXAI_SHADER_DIR` is used.
14. Calibration and quantized-artifact authoring are training/full-profile
    capabilities. Inference profiles may execute quantized graphs but do not
    import, compile, or export PTQ observers or materializers.

## Build composition

```text
volvoxai.js           = readable JavaScript core + forward backends
volvoxai.min.js       = minified inference bundle
volvoxai.full.js      = readable inference + training/PTQ bundle
volvoxai.full.min.js  = minified inference + training/PTQ bundle
volvoxai.wasm.js      = readable WASM-only inference + strict training/PTQ bundle
volvoxai.wasm.min.js  = minified WASM-only inference + strict training/PTQ bundle
volvoxai.wasm         = baseline forward C kernels + embedded optional accelerator children
volvoxai.full.wasm    = forward kernels + freestanding C training/PTQ ABIs + the same children
volvoxai              = native runtime + forward kernels/backends/shaders
volvoxai-full         = native inference components + training/PTQ components
```

Web files are emitted under `dist/<package-version>/`, retaining older version
directories as local release snapshots. The inference profile resolves the
forward-only `volvoxai.wasm`; the full profile resolves `volvoxai.full.wasm`, a
forward-compatible superset with versioned, stateless C training and PTQ ABIs.
Explicit WASM training is strict and preflights the supported portable subset
before forward execution or optimizer mutation. Generic PTQ calls run in a
separate scratch instance and expose observation, affine parameter derivation,
I8/U8 quantization, arbitrary-axis I8 weight packing, and I32 bias packing;
JavaScript continues to own browser graph state, safetensors, and file I/O.
The forward-only sidecar remains unchanged for ordinary inference.

Quantized LoRA synchronization is one policy built on those reusable kernels,
not the definition of the full sidecar. For an unchanged W8A8 inference
topology, a separate supported F32 graph owns the persistent LoRA masters and
optimizer state. C transforms caller-owned buffers; `WasmQuantizedLoRATrainer`
stages the resulting A/B weights, commits both graph updates atomically, and
rebuilds the WASM engine's copied scale metadata and packed Q8 cache. This
convenience workflow does not add QLinear/QGemm backward support or automatic
graph conversion.

Optional WASM instruction-set extensions must not make either fixed parent
artifact fail validation on an older engine.  Such code is compiled as a small
child module, embedded in a versioned parent custom section, and instantiated
against the parent's memory only after the child itself compiles successfully.
This preserves the two fixed WASM filenames and the one-fetch loader contract;
the baseline parent remains the fallback and owns allocation and weight packing.

Native shader packs use one XZ block per enabled backend format and scope. A
default Linux inference pack contains SPIR-V, desktop GLSL, and GLES forward
blocks; the full pack adds the corresponding training blocks. Metal blocks are
included only when Metal is compiled. The first shader lookup for a block
decodes and caches that block in memory. CPU-only builds carry an empty pack.

Internal static targets may be modular; the distributed JS modules and native
executables remain single files, apart from the WASM sidecar.

## Change rules

- Separate file moves from behavioral changes when practical.
- Every operator starts with a portable implementation and correctness test.
- Accelerated kernels are compared against the portable CPU reference.
- Hot-path changes include benchmark evidence.
- CI typechecks/tests the JavaScript profiles, builds both WASM sidecars, and
  validates the eight browser release artifacts. Release validation also
  rebuilds both native profiles and records JS, WASM, native, and embedded
  shader sizes.
- Public API and model-format changes include compatibility documentation and
  tests.
- Generated files include a `DO NOT EDIT` marker and deterministic input hash.
