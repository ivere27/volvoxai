# VolvoxAI architecture

VolvoxAI is a unified edge AI engine spanning native C/CPU, native GPU,
WASM CPU and WebGPU. It turns a graph and its weights into a reusable
computation on a CPU or GPU. The same model package can serve a browser page, a native application, or
an edge service. This document explains where the work and memory live, why
compilation is separated from execution, and how to change the engine while
preserving those boundaries. For a first application, start with the
[quickstart](docs/quickstart.md); for concepts, use the [textbook](docs/textbook/README.md).

## Execution model

A model package describes operations, weights, inputs, legal shapes and a typed
execution contract for entrypoints, state updates and supported generation.
Loading validates that description. Compilation chooses a backend and prepares
an executable for the admitted domain. Each request binds concrete inputs;
execution produces named output snapshots that the application can retain.

```text
Application: input preparation, task policy, output interpretation
  Model: graph and weight revision
    Compilation: validate shapes and select a backend
      ExecutionContext: current inputs, scratch, state and execution policy
        Execute (graph) or Generate (discrete ID loop)
          CPU kernels or GPU dispatches
            Completed packet: independently retained named tensors
```

The context fixes Graph or autoregressive mode at creation. Execute supplies an
entrypoint, an explicit computation range and complete input bindings. Generate
supplies initial conditioning once and a positive chunk budget; the engine owns
ID selection and feedback. They share admission, cancellation, completion and
output ownership. Generate supports batch-one, single-stream IDs. CPU/WASM CPU
and CUDA support greedy selection and seeded F32 random selection, with
context-owned RNG and temperature/top-k/top-p. Full-prefix greedy generation
also runs on Vulkan, OpenGL, Metal and WebGPU, including generated-ID, EOS and
sequence stops across call boundaries. Those backends read only the selected
I32 scalar for feedback and stop policy; WebGPU resumes its loop asynchronously
in C. Random selection on these four backends and unsupported state/backend
combinations are refused before mutation. See the
[model contract](docs/model-format.md) and [generated API](docs/api-discovery.md).

Dense Llama/Qwen adapters can declare compact KV and incremental row execution.
The MoE, recurrent Qwen, Llama4 and DeepSeek V2/V3/V4 adapters also accept
`--cached-decoder`: initialization evaluates the active prompt, while each row
transition preserves earlier activations and computes the appended token.
Private context activations retain projected K/V, causal convolution history,
delta recurrence state and completed compressed K/V blocks. Recurrence is
initialized only through the active prompt length, excluding capacity padding.
`GenerationContract.incremental_controls` names the position, sequence and
chunk-mask controls, zero-appended image routing and multimodal positions that
C updates during feedback; conditioning remains immutable. The initial tensor
geometry fixes this single-stream cache's capacity. DeepSeek V4 compression
requires contiguous logical positions so completed blocks cannot change past
queries. The explicit `--generate` full-prefix route remains the reference
mode. See the [HF model workflow](docs/hf-decoder.md) for inputs and limits.

Encoder-decoder models use the same tensor contract. The Whisper adapter emits
a log-mel encoder and a separate decoder conditioned on its immutable output;
generation reevaluates the decoder prefix without rerunning the encoder. The
application prepares PCM features, language/task prompt IDs and selection
rules. Image workflows likewise keep JPEG processing and question templates
with the caller. Whole-prefix generation accepts host-readable initial inputs;
a caller explicitly copies a device encoder result to host conditioning using
the public tensor-copy contract before generation uploads it to the GPU.
See [multimodal workflows](docs/multimodal-workflows.md) for
the qualified model revisions, processor branches and ownership examples.

Model adapters lower architectural differences to registered operators and
typed tensors. CPU and WASM share the F32 numerical implementations. CUDA,
Vulkan, OpenGL, Metal and WebGPU provide modern F32 device routes admitted by
physical buffer, index and launch proofs. These routes cover gated recurrence,
sparse/compressed attention, hyper-connections, partial rotary embedding,
lexical hashing and grouped or hashed gated experts. Integer n-gram IDs and
sharded embedding addresses remain I32 throughout. Native device backends
belong to the full profile; browser device execution requires `FullEngineHost`.
Vulkan, OpenGL and WebGPU have physical RTX 3090 numerical checks for small
fixtures. Metal has native implementation and shader translation checks;
execution on Apple hardware remains unqualified. The
[HF model workflow](docs/hf-decoder.md) records the tested configurations.
Portable memory admission includes reserved workspaces and the largest
temporary kernel allocation that can overlap them. The local checkpoint reader
can widen declared DeepSeek block-scaled FP8 and packed FP4 weights to F32.
Runtime activation/cache quantization and native low-precision arithmetic remain
separate capabilities in [TODO](TODO.md); storage conversion does not reproduce
the released model's quantized execution.

HF importers share graph construction, local checkpoint validation, F32 storage
conversion and package writing in `tools/exporter/hf_graph.py`. Each architecture owns
its configuration and graph lowering; backend kernels consume the resulting
typed operators. Adding a model family does not require a new backend execution
path. Native loading accepts original BF16 weights and shares their immutable
F32 conversion across a compiled model's contexts. `PackedLinear` keeps signed
INT2, INT3, INT4 or INT8 weights packed in I32 words, with explicit groups of 32, 64, 128 or
256 input features and F32 group scales. This storage grouping is independent
of execution tiles. Compilation selects its `PackedArithmetic`: F32 by default, IEEE F16
operand rounding, dynamic per-row I8, or explicitly permitted compensated
TF32 high/low arithmetic (`F32_FAST`). The artifact stores no activation mode;
all choices keep F32 outputs and accumulation. Strict F32 forbids reduced
operand precision. F32_FAST can use ordinary F32 kernels but never silently
selects F16 or I8. Its approximately 21-bit compensated mantissa can lose
precision or operand range at extreme values; callers needing full F32
operands select F32. The graph boundary
remains F32; precision changes occur inside this operator. Embeddings and other
operators retain their declared precision. `DYNAMIC_I4` reserves a future
activation selection; every backend currently refuses it during compilation,
without allocating an executable or falling back to another arithmetic.
Hardware INT4 support alone does not make this runtime option executable.
CPU, WASM and device backends share
this contract, with SSE2/SIMD128/NEON integer dots and authored CUDA half and integer
Tensor Core prefill paths. CPU x86 and WASM dynamic-I8 matrix projections share
decoded weight fragments across four input rows and four output channels. Their
bounded tile reuses the existing K-float scratch allocation as four I8 rows;
each output keeps exact integer group dots and ascending F32 group sums.
Single-row decode and partial row ranges retain the direct path. CUDA matrix
tiles reuse rounded activations in bounded
shared memory for qualified G128 shapes; other groups use a common direct
packed-weight path. Decode keeps direct packed-weight reads. No full dequantized
weight matrix or vendor math library is required. A8 column tiles cache a rounded
activation row when it fits within 16KiB of shared memory, otherwise streaming
one group at a time. Matrix projections with at least 128 output columns and A8
column projections with at least 8192 columns prepare each input row once in the
context's shared workspace. Smaller column projections and A16
decode keep their single launch. A8 stores packed INT8
values and a row scale; A16 stores rounded FP16 values and a row validity flag.
Long inputs use bounded row chunks, with each preparation immediately followed
by its matrix or column computation on the same stream. Output tiles share this
prepared input without repeating its conversion. Chunked preparation requires
at least 1024 output columns; A16 also requires an input width of at most 2048.
Other large shapes retain the ordinary route because chunk overhead outweighed
conversion savings in validation. Each chunk selects its own tile width while
preserving the matrix arithmetic. Prepared CUDA projections also select a row
tile from the full operation shape, before dividing it into workspace chunks.
Validated A8 shapes use 16 or 64 rows per tile on builds targeting SM80 or
newer. The long 64-row path overlaps asynchronous activation loads with the
current group's integer Tensor Core computation using two shared buffers.
Selected long A16 projections use 64 rows on SM75 or newer; SM80 builds use
eight warps for qualified shapes, retaining the four-warp tile where expanding
it did not improve measurements. These long
projections can use up to 32 MiB for prepared input, reducing row chunks and
preserving more parallel work per launch. On SM80 or newer builds, narrow
projections with input width 2048 and output width 256 also use this workspace
within bounded prefix lengths: W4A8 uses the 64-row tile for 1024–2048 rows,
while W4A16/W8A16 prepare 512–2048 rows together and retain the ordinary
16-row compute tile. These complete prefixes avoid the narrow chunked route.
Other shapes retain their 2 MiB
preparation limit and ordinary tile. These private entries reuse each weight tile across
more input rows without changing the group accumulation order. The build target
controls which instructions are available: the default SM75 target retains its
compatible A8 route, while `VOLVOXAI_CUDA_ARCH=86` enables the Ampere entries for
RTX 3090. The build sets the minimum PTX ISA for both forward and training
modules from that target. Missing scratch, incompatible
partial rows, aliases or device limits select the ordinary route before any
launch. Matrix launch dimensions use the same selected column tile as the
kernel. Eight lanes cooperate on each A8 output column, with exact
integer group dots and an ascending F32 scale accumulation. The scalar decode
entry overlaps four full INT4 group dots while preserving this F32 sum order;
remaining groups and partial input groups retain their ordinary path. The CUDA
forward registry also owns packed variant IDs and launch metadata; checked
plans keep function selection, tiles, grids, shared memory and preparation
workspace consistent. CUDA-private execution arguments carry route and column
tile instead of reusing the shared operator's `heads` and `window`. Existing
fused scalar decode entries retain their G128-only contract. See the
[group and tile design](docs/packed-linear-design.md) for extension rules.
Import of external
AWQ/GPTQ/GGUF packing is a separate
capability; the current exporter quantizes immutable Linear weights to this
format. One full-profile `QuantizationPlan` owns an immutable source, a
weight storage/fitting policy, ordered original-node rules and optional static
activation affines. `WeightFormat` contains only bits/group size; RTN,
MSE_CLIP and IMPORTANCE are authoring algorithms. Static per-channel I8
regions, Conv-only per-channel I2/I3/I4 regions and importance fitting require streamed
calibration; ordinary grouped
I2/I3/I4/I8 fitting creates no inference engine. Shared `CalibrationOptions` fixes
CPU workers, named coverage profiles and an exact optional native CUDA
calibration backend. CPU remains the default, and unsupported CUDA execution
never falls back to CPU. Plan inspection reports the actual observation backend
or an empty value when no executor is needed. The engine reports resolved node
policies and calibration necessity so SDKs do not duplicate matching logic.
`CreateQuantizationPlan`, `CalibrateQuantizationPlan`, `InspectQuantizationPlan`,
`ExportQuantizedModel` and `ReleaseQuantizationPlan` expose one lifecycle.
Calibrate commits sufficient statistics and returns compact progress; Inspect
returns the full data-only snapshot. Export fits against current observations
without replaying input batches, returns one file/byte artifact and leaves
previous exports unchanged. Every operation carries one top-level report.
Static and importance plans own private CPU or CUDA runtime/model state, so callers
never load a source Model for quantization. Both paths share C authoring
qualification and weight materialization.
CUDA observations reduce ranges and logical channel/im2col second moments on
the device, transferring compact statistics before atomic publication. A
calibration pass may stop after its last required observation; the immutable
source and exported graph remain complete. Host code still fits packed weights
and writes packages. Calibration backend support is independent of the exported
model's inference backend.
Native file sources are read once and the calibration model snapshots those
accepted bytes through the ordinary immutable model owner. PTQ byte sources on
native and WASM use the same checked snapshot route. Later calibration and
export never reopen the caller's files or read the caller's mutable buffers.
Aggregate size arithmetic, transport/address-space representability and actual
allocations are checked before accepting a source.

W2/W3 are optional CPU/WASM formats; native device and WebGPU routes refuse
them. W3 uses dense least-significant-first bitstreams across I32 word
boundaries, with zero unused high bits; W2/W4/W8 retain zero-code tail padding.
Input-aware fitting weights reconstruction error by observed input second
moments. Low-bit Conv collects logical grouped im2col moments with padding,
stride and dilation; no calibration inputs are retained. This diagonal proxy
is not a full output-reconstruction optimizer. Explicit original-node rules
choose mixed precisions; the engine performs no model-specific sensitivity
thresholding or automatic promotion.

`QConv2D` defaults to eight-bit weights. Its explicit `weight_bits=4` form
keeps static I8/U8 activation affines and I32 bias, with per-output-channel
weight scales. Immutable weights use U8 storage shaped
`[O, ceil(KH*KW*I_per_group/2)]`; `kernel_size` supplies the logical spatial
shape. Each byte stores two signed codes offset by eight, low nibble first;
an unused high nibble must encode zero as eight. Weight zero points are eight
on axis zero. Static low-bit Conv PTQ supports per-channel RTN, MSE clipping or
input-aware importance fitting with group size zero. W2 stores four offset
codes per byte; W3 uses a dense bitstream, with the same logical convolution
dimensions. CPU and WASM decode bounded panels of up to 64 output channels
and 4,096 terms, then reuse the existing exact I8 GEMM dispatcher across 64
input rows. Larger kernels use centered-I16 tiles. The context memory proof
reserves the maximum panel workspace using the shared pack sizing function
(1,053,232 bytes on x86 and 1,315,376 bytes on WASM); no whole weight bank is expanded.
Native GPU and WebGPU refuse this form before dispatch.
Independent small HF oracles live
in `tools/llm_baseline/generate_modern_incremental_stress.py`, shared by numerical
tests and corpus tools, with explicit capacity and seed inputs.
Development validators in `tools/llm_baseline/` share replay and policy modes
per runtime. A single checkpoint catalog resolves each model's reference
settings; one native measurement policy selects explicit cached-execution or
full-prefix-generation workloads. Numerical references, device-route evidence
and separate timing/memory/trace observations retain their own contracts. See
the [tool guide](tools/llm_baseline/README.md) before adding another runner.

Both web profiles have one persistent C owner per host. Inference, scheduler,
diagnostics, and the full profile's planning/training/PTQ services share that owner's handle
registry and linear memory. Module compilation can be cached across hosts;
instances, handles and mutable memory are never shared. There is no JavaScript
numerical kernel, graph executor, backend provider API or compatibility facade.

Product TypeScript is limited to generated API projections, transport and the
WebGPU device bridge. Transport serializes protobuf calls, initializes the
companion WASM, transfers path-named input files and closes its owner. The bridge
owns browser GPU objects and asynchronous device operations. Former runtime
features, including model assembly and tokenization, run in C behind the
proto contract. Development-only registry projections live under
`tools/generated/` and are not imported by product entries.

Node debugging attaches to a Graph execution context and runs its `Execute`
request one node at a time. The session owns its position (`next_step`);
commands carry its `revision` for optimistic concurrency. A separate context
lets ordinary inference continue while debugging is paused. The debugger uses
the ordinary numerical node dispatcher, with immutable before/after snapshots
copied before arena reuse. Graph initialize/transition execution uses the same
phases as ordinary decoding: the private run is
open, prepare, dispatch, complete and close (`VxContextRun`), and the engine's
incremental pass is begin, next/node, end (`VxIncrementalRun`). The debugger
replaces dispatch with a stepwise pass that holds no model lock, adapter route
or device pass between nodes, so admission, row addressing, paged KV and the
failure policy are the same code as the ordinary step. The context stays
attached, refusing other operations with `BUSY`, until the session is released. A failing model is session state
(`FAILED` plus `failure`), not a failed RPC. Debugging is a full-profile
development service; inference builds carry profiling but no debug session,
and the inference boundary check rejects `vx_debug_` symbols. The normal inference loop never enters
this path. Profiling and debugging share `ExecutionPlan` metadata, while their
session lifecycles remain separate. See [node debugging](docs/debugging.md).

## Profiles and release composition

| Entry | Host | C services | Backend availability |
| --- | --- | --- | --- |
| `volvoxai.lite.js` | `EngineHost` | Platform, Profiling, Text, Inference, Scheduler, Buffer | WASM CPU inference |
| `volvoxai.js` | `FullEngineHost` | All services | WASM CPU and WebGPU inference/training; C PTQ |
| `native/volvoxai-lite` | Generated C dispatch | Inference profile | CPU inference; fixed lite composition excludes native device backends |
| `native/volvoxai` | Generated C dispatch | Full profile | CPU and compiled native GPU inference, training and PTQ |

Every JS entry also has its fixed `.min.js` sibling. Inference uses
`dist/<version>/volvoxai.lite.wasm`; full uses `volvoxai.wasm`. Those four JS, two
WASM and two native files are the fixed runtime release inventory. Do not add another
control sidecar or embed a WASM copy into JS. The WASM parents contain their C
services directly, with no PTQ or relaxed-SIMD child module. SIMD128 remains in
portable numerical kernels.

`package.json.version` is the single release-version authority. CMake reads it
for native compile definitions; the central WASM recipe passes the same version
as `VOLVOXAI_VERSION`, with that definition included in build evidence and recipe
hashes. C sources carry no fallback release version. Repository tools and tests
derive release paths from this value; the example loader resolves matching JS
and WASM files. Node and Deno tools use `readPackageVersion(root)` from
`tools/release_version.mjs`; JavaScript shell examples obtain the same value
with `node tools/release_version.mjs`. Python workflows read `package.json`
with the standard-library JSON reader, so version lookup needs no Node runtime.
`npm run version:set -- <version>` synchronizes npm entry paths
and root lockfile versions. The standard `npm version` command uses the same
script through its version hook. See [the release workflow](docs/testing.md#updating-the-release-version).

Browser inference excludes the GPU bridge and all shaders; its WASM companion
compiles without WebGPU and imports monotonic time, entropy and the Synurang wakeup notification. All inference
profiles exclude training code, codecs, public training symbols and training
shaders physically. Full-only C objects are selected by the central recipe;
changing an exported surface or source list requires updating and validating
that recipe. The inference entry must never import full to reuse a helper.
Graph parameter metadata is common: inference can load a graph containing
attention Dropout parameters and seeds. Only the full Trainer executes their
training behavior; reading those graph attributes adds no training dependency.

`tools/release_profiles.mjs` owns browser/WASM recipes. CMake owns native object
composition and optimization classes. Both use explicit kernel/control/service
source roles. Numerical C kernels and libm compile with `-O3`; portable control
and service code use the size profile. Native ISA variants remain separate
objects, with ISA instructions excluded from baseline/control objects.
Linux Clang release executables use LLVM lld's safe identical-code folding to
share duplicate function bodies whose addresses are not observable. Their
matching CPU performance fixtures use the same linker policy. Shared and static
embedding libraries use the same numerical/control optimization classes in
separate position-independent objects. The full library preserves its ordered
source inventory across those classes. Libraries retain their independent link
policy; kernel optimization, unwind information and the public API remain intact.
Release evidence records
the executable's explicitly selected linker, including its version and hash.

Generated kernel registry headers declare fixed-size tables; their companion
C files define each table once per linked runtime. Common metadata and full-only
variants remain separate. WASM includes those definitions in its C owner.
Native GPU admission recognizes only compiled backends, allowing browser and
CPU-only builds to discard native device proofs while full WebGPU retains its
shared validation helpers.

Release WASM links strip debug names and tool metadata. The Unicode license and
release provenance remain embedded; imports, exports and the proto API are
unchanged by stripping.

Release WASM exposes only allocation, generated service transport, file mounting
and the full profile's device bridge support. Forward, training, quantization
and scheduler kernels stay internal to C dispatch. Direct kernel numerical
probes and microbenchmarks use a separate development fixture under `build/`;
public API tests exercise the actual inference and full release modules.

Release provenance binds source dependencies, compiler/linker/optimizer identity
and runtime libraries, flags, ordered objects, profile and artifact hashes. WASM
verification rebuilds from the central recipe and compares payload and object
evidence. JS verification checks the profile closure and a reproducible bundle.
A successful local compile alone does not establish release compatibility.

The build Dockerfile pins the Ubuntu image, archive snapshot and LLVM package
version that supply the WASM toolchain. Update those inputs and the runtime
library hashes in `tools/release_profiles.mjs` together, then rebuild and verify
both profiles. It also installs a checksum-pinned static Binaryen executable.
Both WASM profiles require its recorded optimization pass to succeed before
publishing an artifact; a missing or changed optimizer fails the build.
The image also installs protoc and the Python dependencies from
`tools/requirements-codegen.txt` for the generated API checks in JS builds.

Build JS before WASM. JS builds remove stale WASM companions, so these builds
must not run concurrently. `check:release` verifies all eight files and native
inference/full symbol boundaries before packaging.

Python distribution is packaged separately under `dist/python/<version>/` by
`make build_wheel`. A pinned manylinux Docker image builds and checks the native
profiles with CPU, CUDA, Vulkan and OpenGL support. The wheel packages the full
`libvolvoxai`, its module-host loader, the complete generated Python API and
model tooling; it introduces no additional engine contract. PTQ CLI
orchestration uses generated quantization calls. Python selects the full library
by default. A source-built inference library can be selected explicitly;
full-profile services require a separate full-library owner.
Wheel staging reuses exporter sources
under `volvoxai.exporter` without moving example-owned model policy into the SDK.
The [Python guide](python/README.md) covers installation, usage and qualification.

## Shapes, compilation, and reuse

A dynamic dimension has a name and finite bounds. If inputs share a symbol,
they must bind it to the same extent in a request. This lets the compiler reason
about the whole legal domain while the execution context prepares only the
concrete binding it needs. Equal byte counts do not imply equal shapes, and
checking only the minimum and maximum examples does not prove every shape.

For example, a transformer can accept `tokens: [1, S]` with `1 <= S <= 256`.
The logical graph keeps `S`; a six-token request binds it to 6. Fixed weight
storage belongs to the compiled revision. Mutable activations, layout plans,
and decode state belong to contexts, so one session's shape change cannot
rewrite another's inputs or retained outputs.

Compilation and request binding have different jobs:

| Phase | Responsibility |
| --- | --- |
| Load | Validate graph syntax, tensor references, weight storage, and bounds |
| Compile | Prove supported operator/shape behavior, choose a backend, prepare immutable resources |
| Bind | Validate the complete concrete input batch and reserve its required storage |
| Execute | Run the prepared work with this context's inputs and state |
| Publish result | Retain the exact logical output bytes independently of future requests |

CPU/WASM compilation validates activation descriptors without allocating their
buffers. The first input binding plans lifetimes and allocates the required
arena once; later bindings reuse it and grow geometrically only when needed.
This avoids both temporary storage for every intermediate tensor and unused
power-of-two capacity for a fixed shape. Immutable weights and prepared kernel
resources remain available during compilation. Training keeps its separate
retained-activation policy.

The portable activation plan keeps offsets within each dtype. CPU/WASM places
those typed arenas in shared host storage when their occupied regions have
disjoint lifetimes. Concurrent inputs, outputs and retained values keep distinct
storage; alignment and the original per-dtype offsets remain intact.

The model owns the validated logical tensor declarations and their name index.
Contexts borrow this immutable metadata through their retained compiled model;
each context resolves its own concrete shapes and owns its bindings and results.
The memory inventory attributes the shared metadata to the model once. Execution
memory admission retains its conservative allowance for this metadata.

Starting another task always creates a new context with fresh input and state
bindings. A compiled CUDA, Vulkan or OpenGL model may retain one idle prepared engine and
transfer it exclusively to that new context, avoiding repeated backend setup.
This applies only to dynamic base-adapter Graph models whose entrypoints all
require full execution. CPU/WASM, partial execution, debug, scheduler, traced,
cancelled and failed contexts keep the ordinary creation and teardown path.
The idle capacity belongs to the exact compiled revision and appears in its
memory inventory; releasing its final owner frees it. Output snapshots and
external buffer leases retain independent storage across context retirement.

The [dynamic-shape ADR](docs/adr-dynamic-shape-v1.md) explains the design choices.
The [scheduling guide](docs/scheduling-and-dynamic-batching-design.md) distinguishes
caller-authored bulk tensors, coalesced independent requests, and stateful decode
lanes. They all use a batch dimension, but their ownership and admission rules
are different.

## C ownership and lifecycle

```text
Runtime
  Model: immutable graph and weight revision
    CompiledModel: one admitted backend and immutable prepared resources
      ExecutionContext: mutable shape/decode state and scratch
    Trainer: retained exact model revision (full only)
  Request: admitted scheduler input and result reservation
  ExecutionPacket: completed tensor values, progress and report (no public result ID)
  Buffer: retained allocation, independent range views and external access leases
  Trace: bounded copied observations; completed data has an independent lifetime
QuantizationPlan: immutable source and policy, private static calibration (full only)
```

Profiling and debugging follow widely used conventions so that people and
agents can rely on prior knowledge: `List*` methods page with AIP-158
`page_size`/`page_token`/`next_page_token`, byte reads use ByteStream-style
`read_offset`/`read_limit`, response state enums reserve zero for
`UNSPECIFIED`, debug commands use Debug Adapter Protocol vocabulary (stop
reasons, breakpoints by source node) and Chrome Trace JSON is the timeline
export. `StopTrace` replies when the trace is READY; there is no client drain
loop. Documented bounds and numeric defaults are `@api` annotations that also
generate `native/src/api/generated/api_limits.h`.

Profiling is opt-in at the runtime boundary. The ordinary node loop has no
collector work. A trace records host operations and optionally host node calls,
training phases and engine program invocations;
supported backend adapters append device intervals with explicit clock evidence.
OpenGL/Vulkan can calibrate host placement; CUDA/WebGPU retain causal start
bounds. Opaque queue and batch IDs associate observations without exposing
native handles. Copy, submission, blocking wait and asynchronous completion
activities keep host call time separate from device elapsed time. Typed pages and Chrome Trace JSON project the same finalized data. Chrome export
serializes bounded event pages without scanning or caching the complete trace.
Native query results follow existing pass completion waits. CUDA keeps its
instrumented graph and event ownership separate from the ordinary replay cache.
One capture detail level selects basic operations or node/program attribution;
GPU timing is a separate boolean and defaults to off. With GPU timing enabled,
WebGPU measures existing passes at basic detail and splits compute passes at
program boundaries for node detail. Nodes and programs share bounded query batches. Explicit device-timing admission
prepares a small reusable WebGPU timestamp pool under separate error scopes;
only validated resources enter numerical command buffers. Exhaustion drops timing
observations. Bridge-owned tickets resolve asynchronously and C polls copied observations
at trace reads and the next captured operation. No GPU wait is added.
CUDA selects launch observation through its exclusively owned submission slot;
the shared driver function table remains immutable across concurrent contexts.
Host spans and device intervals are separate proto payloads; calibrated GPU
slices and bounded annotations use distinct Chrome Trace representations.
Coverage reports actual pass/node/program/copy counts, host activity counts,
clock correlation, observed timestamp support and
schedule changes. Releasing a trace retires
its tickets without cancelling inference or exposing WASM pointers to promises.
An optional memory capture records bounded allocation identities and observed
allocator peaks; pooled reuse and aliases do not create storage. Observers keep
a small weak collector owner so late frees cannot retain completed trace data.
WebGPU buffer scope is explicitly shared by its host bridge. Public time values
are uint64 nanoseconds; display and Chrome JSON conversions stay at the boundary.
Resource snapshots and compilation bounds remain available independently.
Typed resource samples distinguish module heap storage, scoped engine inventory,
process RSS/CPU and optional device-wide driver telemetry. CPU load derives from
process CPU-time deltas; GPU activity never derives from kernel duration. Browser
unsupported measurements remain explicit. There is no periodic sampler thread.
Bounded plan metadata joins concrete tensor shapes, schedule lifetimes and known
allocation offsets to executions; it retains no tensor storage. The collector
budget covers records, resource samples and immutable plans, with separate loss
counts. Protobuf pages are canonical; Chrome counters and offline JSON analysis
are projections. See [profiling and memory](docs/profiling.md).

The C training planner attaches optional bounded origin sidecars only for detailed
collection. CPU backward uses a separate instrumented loop; ordinary numerical
loops do not inspect the collector. GPU adapters select instrumented dispatch
routes at pass entry; each ordinary pass uses the ordinary route. Native adapters
restore their driver dispatch slots at completion. Program events
retain their owning node when known, while phase and target tensor describe loss,
gradient processing and optimizer work without inventing nodes. Node, program and
phase observations can overlap and are not additive. Metadata is copied before
the execution owner or training plan retires.

Public IDs are opaque owner-scoped capabilities. Release idempotently retires an
ID; accepted operations and descendants retain internal references. Releasing a
Runtime or Model ID does not destroy live descendants. Host close cancels calls, retires the entire module owner and drains cleanup.
Native modules have separate registries, just like separate WASM instances.
The module retains Runtime roots while descendants or asynchronous engine work
exist; idle roots are reaped after a polling turn. Work that finishes while the
host is idle is reaped at the next call or host close. Native shutdown returns
PENDING until its cleanup worker has joined every engine worker. Entropy-scoped IDs prevent a restarted
owner from accepting handles retained by an earlier instance.

Inputs are validated as an atomic named batch before mutation. Every tensor
carries explicit dtype, concrete shape and exact byte payload. Dynamic dimensions
have fixed rank and finite positive bounds. C validates symbol equality, byte
counts, operator rules, checked arithmetic, quantization and target limits.
Results contain exact logical bytes, never arena capacity tails.

Input validation reports retain the first violation as typed evidence: input
name/index, expected TensorSpec, actual TensorInfo, axis, required symbol/decode
extent and byte count when known. Reports own bounded copies of this evidence;
they never retain caller tensor pointers or echo payloads. A truncation flag
directs consumers to GetModelInfo for full names. That query exposes immutable
input/output contracts before compilation, including symbolic dimensions.

Model owns a parsed canonical JSON graph and semantic plan. Proof/planning borrow
that immutable view. Compilation clones it only for private lowering, then the
private execution engine parses the lowered representation. This removes the
extra source-graph parse; it does not claim allocation-free compilation or a
single representation for every internal phase. Instrumentation is described
in `docs/c-runtime-validation.md`.

Context operations serialize mutable state. Pre-commit refusal preserves the
prior binding and decode state. A failure after submission does not promise
rollback of device side effects. Committed failures make the context terminal;
a new task uses a new context. Device loss makes
resources on that device unusable.
Trainer steps mutate private parameters; commit publishes a successor model
revision, while rollback restores the committed baseline. Quantization plans
snapshot their source and privately own any static calibration model; observation
ranges commit atomically.

## Transport and storage

Generated C handlers are the authority for operation validation and reports.
The TypeScript clients are Synurang-generated `Vx*ServiceClient` classes:
`await client.method(request, { signal, timeoutMs })`. Native and WASM share
`open / send / half_close / receive / cancel / release` and bounded polling.
The host waits for wakeups when no work is ready; it has no periodic idle timer.
Requests are encoded before yielding to preserve a call-time byte snapshot.

`EngineHost` translates domain `OperationReport` failures into `VolvoxAIError`,
retaining the original response, report and method path. Synurang `RpcError`
represents call status, cancellation and deadlines. Transport checks the
terminal call status before reporting success. Execute and Generate reply only
after completion and expose no result polling/read/release lifecycle. Scheduler
and debugger retrieval transfer the same completed packet once. Call
completion/release does not release Model, Context, Buffer or Request IDs.

Python's public service clients apply the same operation-report rule after the
generated method decodes its response. Schema field metadata selects the outer
report, including report-only methods; nested diagnostic records are data.
Both synchronous and asyncio clients raise `VolvoxAIError` with the original
response/report and Python method identity. Synurang `FfiError` and asyncio
cancellation propagate unchanged. This adapter adds no RPC or decode, preserves
generated signatures, and never modifies generated classes or the vendored host.
C consumers inspect the same report explicitly after checking call completion.

Python's `InferenceSession` is a path/NumPy adapter over those generated clients.
It discovers an unambiguous model package, creates an owner and runtime, loads
and compiles once, and reuses an ExecutionContext. The C defaults select CPU
and automatic threads; explicit GPU selection requires that backend. Precision,
input validation and numerical behavior stay in C. The adapter awaits Execute,
consumes the completed packet and copies selected outputs into owned arrays or
returns retained tensors. Session close retires its context/model handles. Retained tensor
leases defer module close until external consumers finish. The adapter contains
no graph executor or numerical kernel.

`VxBufferService` owns common allocation, retention, copy, mapping and standard
C DLPack operations. Execute and Generate return inline tensors or owner-scoped
`BufferView` IDs and byte ranges according to their output options. Transient `BorrowedBuffer` descriptors identify
local resource representations and are consumed before dispatch returns.
Tensor dtype/shape and binding names are separate from storage identity;
physical placement is never inferred from a HOST/DEVICE tensor flag.

Each inference output retains its own storage and byte-budget lease. The batch
retains only metadata and its result slot until all outputs are released. C
access leases permit concurrent readers and exclude conflicting writes.
Retiring a public buffer ID does not invalidate an existing external lease.
The native storage layer and backend adapters have no Python framework or
training dependency. WASM CPU supports retained buffers and execution outputs;
foreign pointers and DLPack remain local-native operations. See
[the buffer guide](docs/buffers-and-tensors.md) for complete workflows and
transport/backend capabilities.

Each execution context caches at most 64 released native allocations and
64 MiB of idle capacity. Live outputs and DLPack leases are never recycled.
Context close drains the idle cache; remaining outputs retain independent
storage until their final consumer releases them. Capacity rounding does not
extend the public logical byte range, and an idle allocation cannot be
imported as a live tensor. CUDA, Vulkan and OpenGL submit input and output
copies in batches with one completion wait per batch. CUDA tracks snapshot
completion per allocation and orders reuse on its nonblocking engine stream.
Unknown external CUDA consumers still require a context drain when their final
access lease ends. An explicit `EndBufferAccess.cuda` instead records the declared
consumer streams, retaining up to 64 event dependencies per allocation. The
engine orders those dependencies on actual use/reuse, not when the scope ends;
unrelated buffers do not inherit that wait. Events are cached until teardown.
`ExportDLPack.access_id` binds a capsule to an existing writable access scope.
Ending it retires permission even while aliases retain the allocation; those
aliases must no longer be used. Last-owner destruction may wait for recorded
work. These common services compile without training dependencies. Metal currently keeps its
synchronous per-copy adapter. `ReleaseBuffers` retires several capabilities in
one dispatch; external C leases continue to retain the corresponding storage.

`InputBindings` binds each required input exactly once through new values,
unchanged-input reuse, or declared previous-output feedback. The execution context owns the
referenced values; no exported handle or Python object owns that state. Output
selection limits snapshots and result-budget reservations, while all graph
outputs remain available for the next call's feedback. Successful execution
publishes the next binding state. A committed failure or discarded completion
invalidates reuse/feedback; pre-commit validation failures preserve them.

CUDA, Vulkan and OpenGL capture private views of the reserved physical domain
before shape rebinding. Unchanged inputs regain their device-valid marks
without a copy; feedback copies directly into the next graph input. Outputs
whose storage overlaps any input, and backends without stable physical views,
use temporary snapshots before committing any input. This preserves simultaneous
swap/cycle semantics. No live arena address is exposed to applications, and
old exported snapshots remain independent of feedback and later execution.

CUDA imports validate the pointer's primary context, device, allocation range
and dense byte extent before binding. DLPack producers order writes on CUDA's
legacy default stream; borrowed inputs capture that dependency with an event
and enqueue a wait on the engine stream. Persistent imports complete that
producer handoff at admission. Owned buffer IDs and internal feedback already
have known readiness and do not wait for unrelated default-stream work. GPU
input data is copied only device-to-device into the arena;
GPU outputs are copied into independent snapshots. Snapshot exports share that
storage without another copy. Vulkan and OpenGL imports currently accept only
module-owned retained allocations on the same device/context. Metal accepts
same-device MTLBuffer objects with four-byte aligned offsets and byte extents.
Its adapter requires qualification on macOS. Producers using graphics APIs
finish their writes before calling the synchronous API. Inputs
used in host-side index/route admission proofs remain host inputs; device
payloads cannot be dereferenced by those validators. Native training uses the
same binding path and preserves those admission proofs. Later execution never
overwrites a retained snapshot. C excludes native access while an external
write lease is live; applications complete all work before returning that lease
or declare every consumer stream through the scoped CUDA completion contract.

CUDA execution remains synchronous at publication and shares one engine
submission stream per loaded library. It does not schedule independent requests
on separate streams. Pageable host uploads/readbacks use explicit transfers on
that stream through a pinned slab capped at 8 MiB, with a completion wait per
chunk and an extra CPU staging copy. This avoids implicit legacy-stream
dependencies but does not promise faster idle host transfers. A failed transfer
poisons slab reuse until backend teardown. `CopyTensors` reads a CUDA tensor
slice directly from its checked byte range after ordering the owner's producer
and consumer dependencies. It transfers only that range through the bounded
slab; reading one logits row does not stage the full retained output. Backends
without a range readback adapter retain whole-owner staging. Cold allocation, eviction, shutdown
and untracked external consumers can still impose broader waits. The
[CUDA reuse benchmark](python/benchmarks/cuda_stream_reuse.py) checks isolation
from unrelated streams and records latency, driver calls and exact output
comparisons.

Embedded CUDA modules store deterministically generated XZ data with a CRC32
check. The build preserves each compiler PTX as `.ptx.raw`, then removes line
comments before embedding. Quoted strings, character constants, escapes, block
comments and line breaks remain unchanged. A generated provenance file records
both hashes; kernel arguments and the runtime module use the same prepared PTX.
This reduces packaged text without changing instructions or the decoder bound.
Backend initialization decodes the bounded module, validates its complete
NUL-terminated image, then loads it through the Driver. The temporary decoded
buffer is released after loading. Compiler-derived argument layouts remain
bound to the uncompressed PTX. The private forward function registry owns each
kernel's optional argument-storage policy; the embedder derives its supported
set from that registry and checks it against the compiler's parameter layout.
Inference loads its inference module; the full
profile owns the separate training module. The development shader override
continues to use external files and logs when it is actually selected.

CUDA computation uses the NVIDIA Driver API and VolvoxAI's own kernels.
The backend does not link or load cuBLAS, cuBLASLt, cuDNN, cuTENSOR, NCCL,
or the CUDA Runtime. Keep this constraint when optimizing kernels; vendor
library measurements can provide performance targets. Single-row F32 dense
projections use a GEMV kernel for each supported weight layout. Multi-row
projections use authored GEMM kernels on the same owning submission stream
and follow the same replay, numerical and memory-ownership contracts.

Authored multi-row projections and aliased output ranges use tiled GEMM.
Matrices with more than 32 rows and at least 1024 output columns use a
32 by 64 register tile with 6336 bytes of block-local storage. Each thread
owns four rows by four columns, reusing eight shared operands for sixteen
multiply/add pairs at each input position. This route retains ascending input
accumulation without fused multiply-add. Short or
narrow matrices use a 16 by 16 compact tile. Its four warps compute fixed
contiguous partitions of the input axis in 8704 bytes of block-local storage,
with warp synchronization for each warp's private staging. Each warp prefetches
the next input and weight stage into registers before consuming the current
shared stage. Empty rows, columns and short input partitions load zeros.
A block barrier precedes the fixed-order combination of their partial sums;
bias is added once. Finite F32
results can round differently; the original F64 and normalized-RMS correctness
budgets remain unchanged. A new compact result that is zero or nonfinite is
recomputed with the original bias-initialized ascending accumulation. These
two tiled routes allocate no device workspace. A separate split route handles
17 through 32 rows when both widths are at least 1024 and the selected partial
output matrices fit in 2 MiB. It supports both weight layouts. One launch
writes 2, 4, 8 or 16 input partitions using 32 by 32 tiles; a second launch
combines them in fixed order, adds bias once, and repairs zero or nonfinite
results with ascending accumulation. The private parameter is named
`split_k_slices`; it partitions the input axis and does not change its width.

At device initialization, the Driver supplies the SM count and the split
kernel's resident block count. A small analytical rule targets two waves of
resident blocks, bounded by workspace and input work per partition. Unknown
capacity selects tiled GEMM. On a shape's first ordinary inference execution,
an optional calibration compares tiled GEMM with the admitted split counts
on the owning stream. It reads current input, weights and bias, writes a
private output, and measures each complete split-and-reduce pair with Driver
events. Three rounds of eight repetitions use their median; results within
3 percent of the best prefer tiled GEMM, then fewer partitions. Calibration
has a process-wide 50 ms soft wall-time budget, checked between bounded
measurement batches; an unfinished comparison retains the analytical choice.

The process retains up to 64 choices keyed by device ordinal, matrix shape
and weight layout. Each context freezes up to 32 choices, so replay never
changes the split count. A busy or full process cache uses the analytical
choice without waiting; a full context cache uses tiled GEMM for new shapes.
Existing process-cache hits remain usable when the cache is full.
Capture and validation freeze an unseen shape to tiled GEMM without measuring.
Standalone operators, training and Trace do not run calibration. Choices
survive module teardown within the same process, contain no tensor pointers,
and provide no numerical or memory-admission proof. There is no disk cache.

An initialized CUDA context optionally reserves a fixed 32 MiB shared workspace
and a 1 MiB calibration output before capture, including contexts whose graphs
do not select the split route. Two timing events belong to that context.
Contexts that never initialize CUDA reserve none. The pointer remains stable
until queued work and replay references drain; it never grows during execution.
F32 split/reduce and attention remain bounded to its first 2 MiB; qualified
long PackedLinear projections can use the full reservation. Memory admission
and profiling account for the entire physical reservation. Allocation failure
keeps ordinary computation available. Ordered F32 split/reduce pairs,
PackedLinear input preparation and admitted GQA attention reuse this same workspace; their temporary values are
consumed before the next operation
can overwrite it. PackedLinear chunks retain the original accumulation order,
and replay records both preparation and computation with the stable scratch
address and the current input/output views.
Single-row PackedLinear commands with immutable weights, scales and optional
bias also use the prepared CUDA row path. Admission shares the ordinary
PackedLinear shape, precision, group-size and alias checks. A private scalar
kernel entry records the input and output row pointers separately from its
fixed operands, so later rows can reuse the captured graph without rebuilding
the operator descriptor. Wide A8 projections record preparation and computation
as two ordered calls to that entry; their workspace and phase remain fixed.
Changed tensor bindings invalidate prepared commands before reuse, and a
submitted failure stops the row without scattering an incomplete output.
Single-row GQA with one batch, 32 query heads, four KV heads, 64 channels and
512 through 2048 keys can compute attention in one launch. Each 1024-thread
block handles one head and 32 output channels. Aligned 16-byte Q/K loads retain
the original scalar channel arithmetic; 48-key denominator and value partials
merge in the same ascending order as ordinary cached attention. A device-side
mask reduction returns zero before reading Q/K/V when every key is masked.
This path requires 16-byte Q/K alignment and 12432 bytes of shared memory.
Replay validates both the key-count range and each new query-row address before
changing argument records. Alignment or device-limit misses retain the four
ordered stages in 614400 workspace bytes: key-partitioned scores, normalization,
48-key value partials and their ascending merge. Missing or overlapping workspace
selects ordinary attention before submission. Other single-row shapes and causal
single-row calls retain ordinary attention. These paths accept no mask or a shared
key mask; a validated one-batch `[1,K]` mask is normalized to the identical `[K]`
addressing only when entering an optimized path. Once a launch is submitted,
failure stops the operation without retrying it or publishing partial output.

Grouped-query prefill with one batch, 32 query heads, four KV heads, 64 channels,
128 through 2048 queries and 49 through 2048 keys can transpose K once into the
first 2 MiB of the same workspace. Consecutive keys then have contiguous
addresses during each channel step. The compute kernel processes four query
rows per block when each row can attend to at most 512 keys, or two rows
otherwise. Causal inputs with at most 512 queries therefore use the four-row
kernel even when the physical KV capacity is 2048; transpose storage and K/V
strides continue to use that full physical capacity. Noncausal inputs and
causal inputs with more queries retain the selection based on key count.
The block caches each query's 64 channels in shared memory before the
channel dots. Invalid tail rows initialize zero and all threads reach the
cache barrier. The kernels reserve 13,360 bytes of static shared storage for
four rows or 18,976 bytes for two. The kernel reuses each loaded value across
those rows and evaluates independent 48-key value groups in parallel. The scalar channel dot, 48-key denominator and value partials,
and ascending partial merges preserve the ordinary cached kernel's FP32
arithmetic order. The first 48 causal queries still use the original short-row
dot and reduction, and noncausal inputs with at most 48 keys retain ordinary
attention. This route accepts no mask, a shared key mask or the equivalent one-batch
`[1,K]` mask after its declared extent is checked. Query-dependent mask layouts
and geometries outside the admitted range retain the general kernel. Numerical
tests compare the complete transpose/short-row/tiled chain bitwise with that
kernel and against an independent FP64 reference, including active keys above
1536, partial tiles, masked NaNs and fully masked rows.

Prefill validates every launch dimension, shared-memory requirement and
workspace alias before submitting its first stage. Missing workspace or a
device-limit miss selects the general kernel. Once submitted, a stage failure
ends the operation without retrying it. Replay stores the current views and
the stable workspace pointer in typed device argument records. Prefill uses
ordinary graph replay and does not extend the single-row replay recipe; the
stream finishes its temporary values before PackedLinear, dense matmul or
another attention operation reuses the workspace.
Memory admission conservatively charges the full bound even if allocation
fails. Partial-workspace allocation failure or a geometry, capacity or device-limit
miss selects tiled GEMM before either split launch. Calibration allocation
failure keeps a cached or analytical choice. A submitted launch or timing failure
ends the operation instead of retrying another kernel; unresolved work retains
its owners for cleanup. GEMV continues to use
fixed reductions and block-local storage without additional device workspace;
floating-point accumulation order can differ from GEMM.
Incremental decode uses interior resident views for consecutive physical rows
only when the numerical route supports those views. It prepares the entire
owning output before a partial write and keeps staging for overlapping output
ranges, empty rows, noncontiguous page selections and slot-only routes.
Consecutive staged rows use copies on the same CUDA stream without uploading
indices. Reserved staging capacity and synchronous publication stay unchanged.

Two adjacent OUT_IN projections with the same input can share one GEMV launch
while using separate weights and retaining both dense outputs. A supported
one-row, no-bias projection followed by plain residual addition can also write
both its original dense result and the declared sum in one launch. These
patterns use current typed bindings and disjoint resident owners. Their
ordinary routes retain unsupported layouts, bias, aliasing, adapters and
selection boundaries.

Retained CUDA contexts index disjoint owning allocations and prepare tensor
references and producer relationships against their current generation and
descriptors. Overlapping allocations keep the containing-owner scan. KV page
clearing and copy-on-write touch their exact device ranges and preserve the
rest of the owner. A new KV page clears up to 64 tensor ranges in one private
kernel using a bounded by-value descriptor image; each range is checked before
submission and no device workspace is allocated. Ordinary uploads snapshot caller bytes into separate pinned
spans and drain the stream before those spans can be reused; result publication
still waits for completion. Trace, capture and interop copies keep their own
completion boundaries. Adjacent SiLU and multiplication can share a private
CUDA launch while preserving both semantic outputs; node tracing and debugger
steps retain separate dispatches. Active-prefix initialization also keeps
separate SiLU and multiplication, avoiding full owner uploads merely to
preserve inactive rows. SiLU writes its active view; the ordinary multiplication
route still computes capacity padding.

For a supported single decode row, adjacent 128-channel head RMS normalization,
reshape and RoPE share one launch and preserve all three logical outputs. An
elided contiguous reshape may share the normalization span only when its
current host address, full extent and resolved device address match exactly.
The runtime re-describes that skipped view; it never borrows an absent numerical
command. Both stores write the same value at the same work-item index. The
rotated output remains disjoint from every other span; the two shared outputs
remain disjoint from inputs, controls and weights.
Current descriptors, producers, ports, positions and resident owners are proved
before coherence can enqueue work; a later failure fails the pass. In the
canonical decoder this supports the Q head chains. Paged K tensors, debugging,
node traces, training and incompatible row selections keep ordinary dispatch.

Causal single-slot CUDA Generate can restrict supported token-local prefill
to the active conditioning prefix. Its terminal LM head and ArgMax can compute
only the final selected row when logits have no other consumer. Ordinary
Execute, node tracing and unproved graph shapes retain the full output domain.
This also covers rank-three, single-batch `PackedLinear` inputs whose sequence
dimension matches the proved capacity. Packed descriptors keep their physical
buffer spans while narrowing logical work; subsequent row execution clears that
prefill selection before binding its own row.
Before a narrowed prefill starts, every live output establishes its complete
logical span inside the reserved CUDA domain. Later attention, fusion or ID
feedback may therefore resolve the full tensor without treating its computed
device prefix as a host-authored shape growth. This preparation neither copies
data nor changes the growth rules for mutable inputs.
These are private lowering decisions, not changes to the public ExecutionPlan.
The existing full-forward CUDA replay cache stores exact argument images using
ABI widths generated from its embedded compiler PTX. An argument or command
sequence change resumes ordinary dispatch from a verified prefix; unknown ABI
forms and bounded metadata exhaustion disable reuse. Incremental whole-graph
execution can instead prepare a private command sequence for the context's
proved row domain. Numerical commands retain typed operator parameters and
tensor indices against exact model and descriptor guards; current row, page
and control values are bound before each pass. These commands do not replace
the public execution plan or change tensor ownership.

Required-row admission can borrow these descriptors when the model generation
and every tensor's storage, dtype, shape and input role still match. It checks
the current changed closure, masks, positions and pages before committing any
input. A temporary probe clears control and span proofs on entry and exit and
restores its caller's row geometry. Cold bindings and debug, trace, training or
adapter routes use ordinary projection checks. Backend ownership is resolved
before CPU-only row-support probes, preserving the admitted device node set.
Required device-row sessions use the incremental pass's authoritative closure
check directly. AUTO still probes first to negotiate dependency execution;
CPU and handoff routes prepare their retained host prefix. Public precommit
admission remains separate, and no proof survives a mutable binding change.

Public GPU input value conditions are compiled once from graph control
lineage. Binding checks the current public leaves against their live shape
and resident-bank bounds, including both branches of conditional controls.
It does not repeat graph parsing or producer searches per token. The bounded
host predicate allocation, replacement peak and lineage scratch enter memory
admission; failed preparation preserves the old plan and commits no input.
An admitted CUDA row pass proves consecutive physical spans once and reuses
unchanged attention controls across layers. Empty slots and noncontiguous
pages keep the checked index path, and each new pass discards these proofs.

CUDA row attention reads the current projected mask before selecting its
existing masked or unmasked kernel mode. A valid, disjoint mask with every key
allowed needs no device mask reads; holes and empty slots retain the masked
mode. This decision adds no cached state or workspace. Changing the mask can
change the ordered upload sequence, so replay resumes its verified command
prefix through the existing ordinary-dispatch fallback.
Cross-attention rows with at most 48 active keys and at most 128 channels per
head compute each dot with eight lanes and a fixed reduction tree. Four warps
process sixteen keys at a time; softmax uses the existing warp reduction.
For decode with one query, at most 48 physical keys and at most 128 channels per
head, each block handles 32 contiguous output channels. Four warps accumulate
separate key partitions, then merge their four partials in a fixed order.
Other query lengths, longer or wider rows and a retained grid that no longer
covers the output tiles use the original full-width probability-times-value
path. Up to 2,048 active keys, cross-attention caches scores in bounded shared
storage to reuse each scalar dot product across max, denominator and value
passes. The first warp computes the maximum and independent 48-key denominator
partials, merging the denominator in ascending group order. Each output channel
retains the original ascending value-group sum. This preserves rounding before
later activation quantization; changing the reduction order can amplify small
differences across decoder layers. Longer rows stream and recompute scalar dots.
Cross-attention entries reserve
8,720 bytes of aligned static shared storage; packed attention retains 208
bytes. Launch admission accounts for these reservations. The short reductions
may round differently while retaining the same numerical budgets and replay ABI.

Paged KV membership and legal operator read ports compile once against the
prepared model references and current binding. A selected closure checks
duplicate writers through tensor indices, without comparing every earlier
node or searching tensor names again. Model generation, reference tables or
binding changes invalidate these rules; unprepared references retain the
ordinary name-based checks. Current page positions and addresses still use
their checked mapping path. Membership, read rules and selection scratch live
in the existing charged graph metadata slab, including its replacement peak.
Initial cache configuration checks consumers in one graph traversal using a
temporary tensor membership map. References use their index when its live name
matches, otherwise the checked name lookup preserves ordinary semantics. The
map is released before page and scratch preparation, and a failed check commits
no cache binding.

For the first CUDA required-row Generate prefill, this prepared cache also owns
the concrete shape plan already resolved during cache configuration. Binding
consumes it once when the compiled model, generation, input names, ranks,
dtypes and every input extent still match. A mismatch releases it before the
ordinary resolver runs. Input value checks, reference snapshots, copy-on-write
and transactional commit still run against the current request. Activation,
cancellation and cache teardown release an unused plan; its temporary owner
is included in the existing metadata peak admission.

Greedy CUDA selection uses four independent candidates per lane to shorten its
serial scan. Large vocabulary axes can use 1024 threads with the same fixed
256-byte reduction storage; devices with smaller block limits retain the
256-thread route. Ties keep the first index, including the existing NaN and
nonfinite rules.

Incremental CUDA replay records kernels, immutable host upload snapshots and
device copies in their admitted order. Supported scalar-argument kernels keep
stable addresses into one bounded device record buffer. Each reuse fills a
pinned argument image; the graph uploads it once before executing the original
numerical bodies. Ordinary launches and prefix fallback use direct arguments.
The private pointer helper preserves the global address-space fact for proven
device allocations and interior views when loading a record or entering an
attention device helper. Shared and local workspaces retain their own address
space. The pinned Clang PTX build checks the actual global memory instructions;
this does not change numerical arithmetic, kernel ABI or workspace capacity.
Other kernels retain checked argument updates. Changed copy addresses are
refreshed before one graph submission on the engine stream.
Supported consecutive row passes can compile a private recipe from an observed
command sequence: device operands advance by proved row strides, while named
attention key counts and RoPE positions bind current values. Exact model,
descriptor, binding, storage-epoch, closure, mask and physical-page guards
precede reuse. A guard miss retains ordinary node lowering. The recipe patches
the same bounded argument records and keeps the existing upload lifetimes and
graph completion path; it does not add a second executor or public plan.
Command identity, geometry, compiler-derived argument ABI and storage epochs
remain guarded. A topology or capacity miss submits the current verified
prefix and continues ordinary dispatch. Pinned upload snapshots have bounded
plan ownership and remain alive until the stream completes; failed completion
quarantines them together with their device records. Four retained plans each
admit at most 256 KiB of argument records and 256 KiB of control snapshots;
the fixed host/device allowances and actual allocations enter memory admission
and inventory. Debugging, node traces and unsupported commands retain the
ordinary numerical path. The public API continues to publish completed,
independently retained outputs.

The first eligible CUDA row executes ordinary commands while the engine
observes their checked argument images and owns their upload snapshots. At pass
end, it builds and instantiates the dynamic graph before draining the stream,
allowing CPU graph preparation to overlap the queued GPU work. That observed
row executes once; subsequent guarded rows can use the completed graph.
Unsupported commands or bounded metadata exhaustion retain ordinary execution
and disable reuse. Stream completion and owned-context restoration are still
required for a successful pass; a failure does not retry submitted work.

Python messages use the official protobuf runtime. The generator builds the
schema classes from protoc's descriptor and projects Synurang clients to
`SerializeToString` and `FromString`. Clients keep the same ModuleHost,
async, deadline and cancellation behavior. Message presence uses `HasField`;
message and repeated fields use the standard `CopyFrom` and container methods.
The Python package declares its qualified protobuf runtime dependency.

Python `Tensor` implements DLPack through a small CPython stable-ABI capsule
adapter built with the vendored DLPack header. The adapter only consumes and
creates capsules and manages their lifetime; it introduces no engine entry
points or mandatory framework dependency. C DLPack deleters own storage/access;
Python capsules retain module code after wrapper/session close. The wheel is `cp310-abi3` and is tested
on CPython 3.10–3.14. The asynchronous NumPy path remains separate; this initial
retained-tensor workflow is synchronous and guarantees device completion.

Python metadata exposes immutable names, shape tuples, dtype strings and
symbolic bounds projected from TensorSpec. NumPy input buffers use the existing
host BorrowedBuffer contract. Completed packets carry concrete output shapes;
CopyTensors transfers retained output ranges into owned NumPy arrays. AsyncInferenceSession
sequences the generated async services, snapshots inputs and serializes its
context. Cancellation/deadlines drain an in-flight operation and retire its
packet's buffers before releasing pointer owners. They do not assert that canceling a
transport call stops a C kernel.

Python's lazily imported full-profile workflows also compose existing RPCs.
`quantize` and the PTQ CLI use the same plan lifecycle for every supported
weight/activation policy. Static activations consume streaming calibration;
dynamic and floating activations export the ready packed snapshot directly.
`TrainingSession` maps array/native tensor batches, tensor targets and
SGD/AdamW options to the C trainer. Selected forward snapshots are captured
before backward/update. CPU shared parameter views retain a read lease that
blocks trainer mutation; ordinary parameter exports are independent snapshots. Model exports retain the loaded graph;
checkpoints use the generated protobuf codec and C validation. Neither adds
an engine operation, Python autograd, a numerical kernel or a dependency to the
native inference profile.

`WaitRequest(RequestRef)` is asynchronous: internal request watchers post wakeups
and the module samples completion in bounded polling turns. A cooperative WASM
turn may start one scheduler dispatch; it cannot preempt an individual CPU
kernel. Cancelling or timing out this RPC removes only its watcher. The separate
`CancelRequest` operation changes engine work. Shared memory and new queue
transports are deferred in `TODO.md`.

`NativeStatus`, `OperationCode` and `OperationStage` are the only public outcome,
detail-code and stage vocabulary. C reports carry these generated enums;
messages never determine an error's classification. The TypeScript error
adapter preserves `status`, `code` and `stage` directly, including full-profile
reports, without a separate phase vocabulary. RPC paths and service transport
names are generated from the service declarations in the same proto.

Native paths resolve through the operating system. Web `LoadModel` and
`PublishAdapter` paths are fetched and mounted into a private C VFS. A C
preflight runs generated decoding and path validation before network I/O.
File staging is bounded and privately named per call. Await a call before
releasing handles it needs; asynchronous preparation is not a transaction with
other calls. Ordinary web path, inline and adapter packages retain a 64 MiB bound.
Path `LoadModel.source_options` can admit up to 1 GiB of WASM32 source bytes,
with explicit total/per-file budgets and ordered length/SHA-256 descriptors.
The host cap can be tighter. Manifest bodies stream through a 1 MiB mailbox;
C checks accepted bytes, and private staging storage transfers to immutable
snapshots without a second source allocation. Closing an owner aborts an
unfinished upload before freeing the module. These are source budgets, not
reserved prepared/execution/device memory. VFS entries grow dynamically and retain
stable storage while model maps/readers hold references; unmounting a transport
name does not revoke an accepted model's bytes.

LoadModel also accepts ModelPackage graph/SafeTensors bytes directly. Exactly
one source form is accepted; bytes and paths cannot be mixed. The common C
loader validates source selection, shard counts and the 64 MiB package limit
before snapshotting into the same retained storage used by path loading
(private files on native, VFS in WASM). Web transport applies a tighter configured
maxPackageBytes to inline packages as well. It performs no fetch or path
resolution for this source form.

DescribeApi exposes the active build's methods and, on request, their dependent
message/enum contracts. Service and method filters limit the response. A local
generator reads protobuf descriptors and validated @api annotations in the
schema comments to produce typed metadata tables and JSON/Markdown references.
The same profile filter used for Synurang excludes full-only contracts from
inference tables. This metadata documents required fields, presence, defaults,
oneofs and explicit semantic rules; C validators remain authoritative for
graph-dependent constraints and backend admission.

Full authoring/export calls use protobuf bytes in WASM. `ExportTrainerWeights`
returns SafeTensors shards. `ExportTrainerCheckpoint` returns a generated proto
message containing the exact logical graph, private weight shards, optimizer
SafeTensors and configuration, optimizer step, RNG seed and opaque application metadata.
`CreateTrainer.checkpoint` validates that state against the requested Model
before publishing a Trainer ID. The private checkpoint becomes the rollback
baseline until commit. `GetTrainerState` and `ResetTrainerAccumulation` inspect
or discard an unfinished gradient window in C; neither submits GPU work. State
includes the current tensor binding, activation capacity and plan cache usage.
Optimizer fields omitted from a step keep the last successful configuration;
rollback and checkpoint restoration preserve that configuration with the weights.
Quantized LoRA composes these same C operations: `DequantizeWeight` initializes
F32 masters, `TrainStep` updates their private Trainer, and
`ExportQuantizedTrainerWeights` produces an immutable I8 successor package for
`LoadModel`. Export preserves master values and template storage. Publishing a
new inference revision never changes an older model or retained result.
`CreateTrainer.shape_options` controls LRU entry/metadata limits and activation
capacity/growth. Defaults match the former implementation: 8 entries, 1 MiB,
512 MiB and factor 2. A plan exceeding the metadata budget executes uncached.
State lists cached shape signatures from least to most recently used and reports
metadata/storage bytes, oversize skips and uncached admissions. These are C
layout plans; compiled C operators replace the former JavaScript execution
closures. The counters do not claim a cache of GPU backward command lists.
`CreateTrainer.qat` optionally enables weight-only W2/W3/W4 fake quantization
for explicitly named original Linear/MatMul/Conv2D nodes on CPU/WASM or native CUDA. The
trainer owns the copied policy, temporary reconstructed weights and clipped
straight-through masks. Forward/input gradients see fitted weights; the
optimizer and exported masters stay F32. Scales are refitted by the same PTQ
fitter, without scale gradients. Checkpoints and state preserve this fixed
policy; restoration, rollback and failure cleanup retain the ordinary master
transaction. CUDA keeps fitting, reconstructed weights and straight-through
masks on the device while preserving FP32 optimizer master identities. Other
device QAT, activation QAT and learned quantizer scales are
unsupported. Inference profiles contain no QAT implementation or policy state.

Trainer engines retain an unfused graph and all forward activations in a reusable
C arena; step boundaries preserve capacity. Binding validates and reserves the
new layout before replacing the current one. State reports actual C metadata,
activation high water including initialization, and logical F32 gradient shapes.
`CreateQuantizationPlan` accepts graph and weight bytes and snapshots them with
an explicit policy. `CalibrateQuantizationPlan` supplies complete input batches
when the plan requires static activation ranges or importance observations.
`InspectQuantizationPlan` reports readiness and `ExportQuantizedModel` returns independent graph and
weight bytes. Native callers may use the declared path forms. WASM rejects
unsupported output paths with `TRANSPORT_UNSUPPORTED`; application code chooses
how to save returned bytes. `ReleaseQuantizationPlan` retires the plan's private
source, packed snapshot and any calibration execution state.

Freestanding decimal parsing/formatting and scalar math execute in the project's
`native/src/runtime/wasm_decimal.c` and `wasm_math.c`. Decimal conversion uses
exact integers; math uses range reduction and convergent series with constants
derived by `tools/generate_wasm_math.py`. Their bounds and validation contract
are documented in [wasm_numbers.md](native/src/runtime/wasm_numbers.md).
Native releases use platform libc/libm. The non-GPU runtime imports are
monotonic time, entropy and Synurang wakeup; there are no JavaScript math or JSON-number
conversion imports.

An Execute scope with `dependencies` explicitly recomputes the dependency closure
of named input roots in an initialized single-slot AUTO context. Empty roots
execute no nodes and permit only reused bindings; the cursor is preserved.
Slot advancement instead uses explicit `slot_rows` actions. Dependency updates reject REQUIRED row contexts and paged KV
before mutation; these stateful row routes have their own addressing contract.

## Text processing

`VxTextService` creates immutable C-owned tokenizers in both profiles. Vocabulary
and merge bytes are copied before the handle is published. JSON dictionaries,
little-endian binary dictionaries and explicit proto token entries share greedy
and ranked BPE execution; no tokenizer policy or state lives in TypeScript.
Encode supports the default pretokenized mode, explicit greedy matching and a
single BPE word. Legacy decoding strips a leading BOM from each complete
vocabulary entry, joins its bytes, then applies UTF-8 replacement once and maps
the former space marker. This intentionally repairs multi-byte sequences split
across tokens; unknown IDs still contribute empty placeholders. Omitted encode
limits mean 256 tokens; explicit zero returns empty.

The separate `BYTE_LEVEL_DECODE_ONLY` profile accepts encoded HF ByteLevel
strings plus added/special flags. C builds an immutable lossless byte inverse.
`DecodeTokenBytes` returns joined raw bytes and ordered token-byte metadata;
`DecodeTokens` applies UTF-8 replacement once, preserving BOM without text
cleanup. Special tokens are kept by default; ordinary added tokens always
remain literal UTF-8. Unknown IDs fail atomically. Handle metadata distinguishes
all registered IDs being convertible from all 256 standalone bytes existing.
This profile rejects `EncodeText` and nonempty merges.

Complete `hugging_face_tokenizer_json` definitions select `HF_BYTE_LEVEL_BPE`
or `HF_SENTENCEPIECE_BPE`. The C core admits exact supported normalization,
pretokenization, BPE/byte-fallback, added-token and single-sequence postprocessor
configurations and refuses unsupported definitions atomically. ByteLevel
supports lossless token bytes; SentencePiece-style decoding needs the whole
ID sequence and does not publish independently concatenable token bytes.
The caller owns chat templates. Raw `.model`, Unigram and unsupported regex or
normalizers remain outside the admitted profiles. Definitions are bounded to
32 MiB and need no external tokenizer dependency in the deployed runtime.

Pretokenization uses pinned Unicode 17.0.0 letter/number and word intervals;
HF NFC also uses canonical decomposition, combining classes and composition
tables with algorithmic Hangul. Legacy category intervals remain unchanged.
The Unicode generator downloads the pinned data and composition exclusions
into memory and verifies their SHA-256 on every
generation or check; those commands require network access. The generated C
table remains committed and is embedded in both profiles. JSON token keys are
length-aware, including NUL; malformed UTF-8, unpaired JSON surrogates, non-integer
or duplicate IDs and truncated dictionaries are rejected before publication.
The Unicode notice is retained in native artifacts and the WASM `license.unicode`
custom section. Tests keep the unchanged `99dfd8f` TypeScript tokenizer as a
reference outside every product entry.

## WebGPU planning and asynchronous completion

`native/src/backends/webgpu_backend.c` creates physical dispatches. C chooses
operators, shader variants, parameters, spans and workgroups. TypeScript only
translates generated private commands into device objects, encodes/submits them,
handles validation scopes, maps staging buffers and reports device loss.

Only `FullEngineHost` instantiates WASM with stable deferred GPU imports. Before dispatch,
the private C transport preflight decodes the proto request and checks its live
model and backend policy. A valid CompileModel candidate list containing WebGPU, or a live-model
CreateTrainer request for WebGPU, requests asynchronous device preparation; metadata, CPU-only and invalid requests
do not acquire a device. TypeScript forwards that decision without selecting a
backend or replaying the operation. The generated public dispatch still runs once.

Private ABI signature/layout and shader IDs are generated. C/WASM and JS must
agree on both the GPU ABI hash and shader catalogue hash before enabling the
bridge. The full catalogue contains inference and training shaders. The private
C header selects the same generated closure as the full browser entry; browser
inference includes neither catalogue. Device limits include buffer/binding sizes, workgroup axes and WGSL
shared memory. Compile constructs and validates C plans without submitting GPU
work. Variants exceeding device limits are excluded before allocation.
Automatic device acquisition requests the adapter's supported buffer and
storage-binding limits; C still checks the granted device limits. Explicit
caller-provided devices retain their own capability limits.

Compile admits WebGPU only when it can execute the entire numerical graph.
`REQUIRE` returns a typed refusal otherwise. `PREFER` may select the whole model
on WASM. A graph cannot hide a GPU-to-CPU numerical handoff in the synchronous C
loop, and an execution failure is never retried on another backend.

Each graph execution submits its prepared work and keeps its generated call
pending until device completion. Full-prefix Generate can submit several graph
executions within that one public call: a C continuation awaits each selected
ID, updates feedback and stops, then dispatches the next prefix. Success
publishes the selected tensors, a stable execution ID and progress together.
Failure publishes its report and terminal progress; applications never receive
a pending result capability.

Each submission copies selected outputs into independent GPU staging snapshots. Later
input mutations, other contexts and future executions cannot change old results.
Callbacks only update bridge-owned completion data; they never write to a WASM
pointer. C polls a live ticket before copying into its retained snapshot. Release
and cancellation can retire pending results safely. A failed map does not destroy
buffers while already submitted queue work can still access them.

GPU spans and metadata arenas belong to contexts. Stable scratch blocks and
span tables grow with the admitted graph and row geometry; growing storage never
moves an address already used by an encoded command. Device limits and allocation
failures bound growth, separately from Runtime result budgets. Closing the owner drains pending
scopes/tickets, releases spans and destroys an automatically acquired device.
Closed host and WASM owners drop their bridge references; deferred imports stop
delegating immediately and a device arriving during close is drained and destroyed.
No graph state or numerical policy belongs in the device bridge.

Modern inference uses authoritative WGSL kernels translated by the normal Naga
generation path to Vulkan SPIR-V, OpenGL GLSL/GLES and Metal MSL. A shared C plan
specifies each backend's buffers, checked parameters and launches, with at most
eight storage bindings per modern dispatch. Weight banks are bound directly.
Sparse attention uses online softmax for compressed banks and short ordinary
attention. Ordinary attention with at least 512 keys fixes the maximum first,
then reduces probabilities and values in 32-key tiles with compensated scalar
sums. Neither path allocates a dense scores matrix. Gated experts reuse one
hidden workspace across selected experts.

Before encoding, WebGPU reserves the largest serial modern workspace across
the graph's admitted maximum shapes. Its GPU buffer retains that capacity while
each dispatch binds only its checked slice, so different layers and longer
prefixes reuse the same storage. Compile's resident estimate includes this
workspace's device capacity, host scratch blocks and optional dummy bindings.
This increment does not establish a complete WebGPU context memory budget;
`max_memory_bytes` continues to require the built-in CPU/WASM resource proof.

Additional admission proofs cover the canonical typed graph domain:

- Dynamic shape: C proves symbolic relationships across the entire domain,
  then checks maximum descriptors against the device. Typed copies, quantized
  operators, convolution and attention use the same planners as fixed graphs.
  Dense/norm feature widths and affine banks retain their shader invariants;
  Softmax and reductions can vary their reduction width. Coincident endpoint
  values do not establish equality of independent symbols. Maximum descriptor
  checks use private scratch/span metadata and never alter the live context.
- Independent batching: C proves that operators preserve the leading request
  axis, including dense weight layouts and canonical normalization defaults.
  One aggregate GPU graph invocation feeds retained lane snapshots. Cancellation
  of a lane cannot revoke another; both lane and aggregate copies are budgeted.
- Required rows: scalar and explicitly declared slot batches share C row geometry.
  Pointwise/broadcast operations, typed dense/embedding, affine transforms, views,
  feature normalization and self/memory attention gather their row operands,
  invoke the existing numerical kernel and scatter only active slots. Invariant
  branches stay cached. C projects causal lengths and K/BK/QK/BQK masks into
  each slot's attention row. CPU and WebGPU use the same geometry; neither has
  a fixed 32-slot limit. `DecodeSlotAction` declares advance, recompute or empty
  slots, and `GetExecutionContextInfo` returns per-slot lengths and cache generation.
  Required-row requests are proved before input mutation. Bindings explicitly
  identify reused values; the model contract declares transition update roots.

Runtime registration and exporter qualification serve different purposes.
Compilation requires a registered route, the canonical shape contract and the
backend's physical proof. An exporter's tested conversion subset does not limit
which proved runtime graphs the public API can compile.

Index and MoE route admission runs across the graph before input publication or
GPU commands. Public values are checked at binding; immutable values and known
device producers are proved at compile time. Device-produced values are never
validated by reading stale host buffers. Partially resident Gather/MoE banks
also require slot membership. Their index domain is the complete declared bank
extent, including absent trailing slots, so negative Gather indices retain the
same meaning across resident subsets.

Executable nodes store only `VxOperatorKind`, generated from
`proto/operator_vocabulary.proto`. Graph/API parsing converts names to that enum;
CPU, GPU, training, parameter validation and fusion compare the enum. Graph
serialization and diagnostics convert it back to its canonical name. Unknown
names are rejected before a node is published, including graph edits.

Fixed input/output roles use `PortKind` from the same vocabulary. Each executable
reference resolves its role when created; CPU, GPU and training lookups compare
that enum. The reference retains its document key for serialization and arbitrary
variadic ports. Concat's numeric-key parsing remains a document-boundary operation.
Built-in backend selection uses `BackendKind` from `proto/kernel_registry.proto`;
its provider-name projection is shared by admission, execution and diagnostics.
Custom provider names remain strings and cannot shadow any built-in name.
Layout symbols and affine quantization kinds are generated by the parameter
registry and parsed at graph boundaries before typed comparisons.

Node input/output references and typed parameter arrays are owned, variable-size
C allocations. Dependency indices live on the resolved references, not in a
fixed table of ports per node. Graph snapshots clone that ownership; graph edits,
rollback and metadata growth release or transfer it explicitly. Concat's
canonical input order is computed once when the node is loaded or edited.

Incremental Generate retains the row session and runs its token feedback loop in C.
For GPU execution, C encodes the token/keep-mask copies and row dispatches; no
intermediate token readback or TypeScript generation loop occurs. Only the final
selected ID chunk is snapshotted. The first call returns N new IDs using prefill
and N−1 feedback steps; continuation materializes the pending last ID before
selecting the next one. Admission validates topology, budgets, range and cache
capacity before mutation. TinyStories' fused WebGPU SDPA currently lacks this
row route and refuses Generate at context creation; its Graph dependency route
remains available.

A positive, single-slot continuation keeps caller reservation and validates its
current feedback once inside the owned row admission. The operation ticket and
cursor resolution precede that validation. Input staging, public budget
transfer, cache copy-on-write, binding commit and numerical work follow it.
First calls, zero-token stop calls and other slot counts retain the separate
outer validation. The inner check still proves current tensor data, dtypes,
shapes, capacity, cache generation, control values and feedback topology; it
does not cache a mutable embedding or graph proof. Validation failure follows
the existing admission cleanup without committing inputs.

Modern `full_prefix` Generate instead extends and reevaluates the entire active
prefix. Vulkan, OpenGL and Metal return one selected I32 value to C per step.
WebGPU snapshots four bytes using the tensor's exact base and a checked row
offset, retaining the context until its asynchronous C continuation finishes.
It applies greedy EOS, stop-ID and stop-sequence policy, preserving sequence
matches across Generate calls. The bridge maps the ticket; it owns neither the
feedback loop nor sampling. WebGPU publishes retained INLINE ID bytes and
refuses OWNED output transport before claiming the context. Its existing
incremental row profile remains greedy with a step limit; additional row stop
policies and random sampling are refused.

Context creation's cache options bind internal causal-attention K/V activations to a
context-owned page table. Page allocation, prefix identity, copy-on-write,
slot retirement and generation tags are C state. `PublishContextPrefix` retains
independent root/output prefix snapshots, while K/V pages share references.
`ReuseContextPrefix` can populate an empty slot without a full-batch prefill.
`ReleaseContextSlot`, prefix eviction and context release retire that ownership. Page
reservations roll back without GPU commands when admission fails. CPU and GPU
use the same logical row mapping. Cache state distinguishes resident-page bytes
from actual allocated pool and prefix-snapshot storage; paging does not claim
that an already allocated dense device pool has shrunk.

## Worker batching

`VxSchedulerService` also owns `BatchQueue`, the C replacement for the former
engine-independent BatchScheduler/ContinuousBatchScheduler policy. Workers submit
stateless contributions or prompts through `SubmitBatchWork`. C groups compatible
contributions, chunks prefill by token budget, controls padding and fill-first
waiting, and owns slot/page reservations and prefix reuse. Group components are
compared separately and retained without the former C fixed string lengths.

`NextBatchDispatch` exposes one stable dispatch until `CompleteBatchDispatch`
settles it. Polling does not repeat admission. The generated protocol replaces
an asynchronous JavaScript callback; the application worker executes the batch
and returns its values and named outputs. C copies all accepted bytes, preserves
per-request value history and last outputs, and bounds terminal retention.
`GetBatchWork` copies a retained result; `TakeBatchWork` copies and retires a
terminal result atomically. A cancelled request can be taken while its dispatch
is still in flight; internal work storage remains alive through settlement.
Invalid completion input leaves the dispatch pending. Cancellation cannot reuse
a slot beneath pending work. Close with drain finishes admitted work; close
without drain revokes pending dispatch IDs and cancels work.

A BatchQueue is a policy owner, separate from Runtime's compiled-model submission
coordinator. It holds no device resources, and its page plans describe storage
owned by the worker; they do not implicitly rebind an ExecutionContext cache.
`worker_busy_ns` measures dispatch exposure-to-completion time. It is not a
measurement of GPU hardware occupancy. Product TypeScript contains no scheduler
state machine or worker execution policy.

Full WebGPU training reuses C's backward planner. C owns the tape, optimizer
state, gradient accumulation and completion transaction. GPU shaders perform
forward, cross entropy, backward, accumulation, global norm/clipping and SGD or
AdamW. WGSL compute entry points and binding layouts are generated from source;
the bridge creates explicit device layouts without interpreting an operator.

TrainStep returns a microbatch ID with PENDING on WebGPU and READY on CPU.
GetTrainStep polls the latest accepted ID without submitting it again. A trainer
accepts one pending step; step/commit/rollback/export return BUSY until it ends.
Completion publishes parameters and optimizer state together after successful
readback and graph restoration. Failure restores the committed baseline, or
poisons a trainer if restoration fails. ReleaseTrainer can retire pending work.
The WebGPU training planner uses training shaders for Dropout and attention
forward execution, with the same C RNG parameters as backward. Incomplete
backward/device plans are refused before submission. There is no numerical
fallback during a step.

Other domains return typed refusals. The fixture matrix in
`docs/c-runtime-validation.md` records tested shapes/dtypes and policies;
an operator name alone does not guarantee support.

## Scheduler, planning and memory evidence

The C scheduler owns admission, priorities, deadlines, input snapshots and
result budgets. Direct mode bypasses it. Scheduled mode uses the same monotonic
clock as `VxPlatformService`; a host without that clock cannot create a scheduled
runtime. `PollRequest` observes without starting threadless execution.
`WaitRequest` can start one queued dispatch per polling turn; its call options
bound how long the caller waits. Hard deadlines reclaim queued or pending work; soft
misses are reported without discarding otherwise valid results. LATEST stream
replacement follows the atomic admission contract in the proto.

`VxPlanningService` accepts an exact Model revision or standalone graph bytes and
weight metadata. A typed `GraphDefinition` is an alternative to graph bytes;
C assigns missing node IDs and applies the same semantic validation.
`EditGraphPlan` applies an ordered transaction to a private draft, validates once,
and publishes a new standalone plan. Existing plans never change. `ExportGraphPlan`
returns the logical document, original storage descriptors (including F16) and
resolved quantization values needed to recreate that source.
Invalid definitions return no handle. A well-formed but unproved
domain returns a readable plan with typed refusal evidence. Public plans describe
semantic tensors, operators, parameters, dependencies, independent batching and
weight-bank residency. They never expose private scratch offsets or executable
GPU records. `ResolveGraphPlan` binds explicit exact/minimum dimensions.
`InspectSafetensors` performs metadata-only inspection; inline header prefixes
work in all transports, while path/mapped-view forms are transport-scoped.
`ReadSafetensors` and `WriteSafetensors` are immutable byte operations in both
profiles. Their storage vocabulary includes every SafeTensors dtype, arbitrary
rank, empty tensors and length-aware UTF-8 names/metadata. The existing C header
authority checks both input files and completed writes. Runtime weight limits
apply when storage is loaded as a model, not when it is edited as a file.
F16 widening, zero-filled allocation and tensor replacement execute in C.

The full profile adds `VxTrainingService.InitializeTensor`: zeros, ones, normal,
Xavier uniform and Xavier normal. C implements the former Mulberry32 stream,
UTF-16 string-seed hash and Box-Muller transform. The WASM profile uses the
private C scalar math implementation for double-precision trigonometry.
Inference excludes the initializer handler and its generated message closure.

Memory observations are opt-in through `VxProfilingService` and report their
partial resource coverage explicitly. Process RSS is an optional external
envelope, not a substitute for exact retained resource charges. Runtime reservations cover queued,
in-flight and caller-retained snapshots. Releasing the final owner returns its
charge; a pending batch aggregate remains charged while any lane retains it.

## Source authorities and generation

`proto/volvoxai.proto` is the single application contract for C, TypeScript,
Python, and agent callers. Operations reach generated Synurang dispatch:
native hosts load `Synurang_GetApi`, while web hosts transport calls into their
C/WASM owner. Adding an operation requires a schema change. Internal lifecycle
functions and hand-written public enums do not belong in `native/include/`.
[API discovery](docs/api-discovery.md) and generated references provide exact
field-level contracts; the guides explain how to combine them into workflows.

| Authority | Responsibility |
| --- | --- |
| `proto/volvoxai.proto` | All application operations, messages, enums and reports |
| `proto/operator_vocabulary.proto`, `operator_param_registry.proto` | Operator and parameter vocabulary |
| `proto/kernel_registry.proto`, `optimizer_registry.proto` | Kernel support and optimizer inventory |
| `native/src/api/` | Generated-dispatch handler adapters |
| `native/src/runtime/` | C ownership, semantics, planning and scheduling |
| `native/src/kernels/`, `backends/`, `training/` | Numerical execution and full-only services |
| `tools/wasm_internal_abi_manifest.mjs` | Private WASM transport ABI |
| `shaders/` | Authoritative WGSL; generated catalogues/packs are projections |
| `ts/host/`, `ts/core/` | WASM and protobuf transport |
| `ts/backends/WebGPUHostBridge.ts` | GPU device transport |

Do not hand-edit generated bindings, shader outputs or embedded byte arrays.
Preserve `VOLVOXAI_SHADER_DIR`; log once when an external override is actually
used. Normal builds check generated files and do not regenerate them implicitly.

The Synurang C call runtime is vendored under `native/third_party/synurang/`.
Bindings are committed under `runtime/generated/`. `tools/generate_proto.py`
downloads the official Synurang v0.8.0 release generator and runtime sources
from the matching GitHub tag, commit
`53180b484cf7ca07a1e7d6f24e58b8a19a2dcfa8`. Both archives have pinned SHA-256
digests and are cached under `build/` for offline regeneration. There is no
local source snapshot or Cargo build step. C uses `mode=module`; TypeScript and
Python use `mode=client`. Both TS profiles share one runtime implementation;
full adds only its own services/codecs. Release and schema provenance is
recorded with every projection. Normal builds use committed bindings and
vendored runtime files without network access or an adjacent checkout.

`tools/package_release.py` creates deterministic inference or full runtime ZIPs.
Each package contains that profile's JS, minified JS, WASM and native binary,
with hashes binding every artifact and the generated API/bridge contracts.
Model-specific runtime/shader pruning is deferred and has no product or build
implementation. Planning is available through the full profile's generated service.

## Change and verification rules

Read this document before changing layout, build composition, operators,
backends, training or shader generation. Run the smallest relevant checks while
developing, and inference/full native and WASM builds before handoff. Schema
changes require regenerated C/TypeScript/Python projections and
`make proto_codegen_check`. `make api_conformance` runs inside native builds.

Final checks include typecheck, all JS profiles, proto/bridge and relevant full
service regressions, native invariant/profile checks and `npm run check:release`.
Real WebGPU qualification runs sequentially with Deno on an idle physical GPU.
Record the GPU model, driver and runtime versions, exact artifact hash, scripts
and results without personal account names, hostnames or home-directory paths.
Browser/MV3 and package checks use actual distribution files rather than
source-only mocks.
