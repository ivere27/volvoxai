# Native CUDA Backend

This page is the canonical reference for the CUDA implementation in the
current VolvoxAI source tree. It documents supported behavior, architecture,
build composition, numerical contracts, operator coverage, and validation.

## Status

| Capability | Status |
| --- | --- |
| Native forward inference | Implemented as an opt-in CUDA Driver API/PTX backend |
| Explicit CUDA routing | Strict: initialization, validation, or execution failure is returned to the caller |
| Native F32 training | Implemented in the full native profile for the current differentiable graph contract |
| Native Trainer execution | Whole-plan preflight and execution with operator fallback forbidden |
| Public native training API | Generated `VxTrainingService`; the inference projection and binary expose none of it |
| Loss and optimization | CUDA cross-entropy, finite checking, accumulation, global-norm clipping, SGD, and AdamW |
| Training residency | Optimizer-updated weights and Adam moments remain device-authoritative until materialized |
| F32-to-W8 authoring | Implemented in the full profile, including optional I32 bias packing |
| W8A8 inference | Implemented with manual I8/U8 kernels |
| Public bounded dynamic shapes | Implemented for the qualified CUDA inference subset; complete domain, route, launch, slot, and resident-resource proof is required |
| CUDA Graph replay | Implemented for a conservative static-inference allowlist and a proof-qualified bounded-dynamic allowlist with an exact-shape four-entry cache |
| Native FP16/BF16 arithmetic | Not implemented; F16 package weights are widened and computed as F32 |
| TF32 and tensor cores | Not used |
| Quantized backward and QAT | Not implemented |
| cuBLAS, cuBLASLt, cuDNN, cuTENSOR, NCCL | Not used |
| CUDA Runtime API | Not linked or loaded |
| NVIDIA Driver API | Loaded dynamically at runtime |
| Multi-GPU training | Not implemented |

The backend covers VolvoxAI's current native F32 inference, the private
Trainer backward planner and optimizers, W8 authoring, and W8A8
inference. This does not imply that every forward-only graph operator has a
derivative or that the backend provides mixed precision, QAT, or multi-GPU
execution.

## Runtime dependency boundary

VolvoxAI owns the graph executor, tensor residency, forward and backward
kernels, loss construction, gradient accumulation, optimizers, PTQ kernels,
fusions, and tactics. A CUDA-enabled executable dynamically resolves a narrow
private declaration of the stable Driver API from the installed NVIDIA
driver.

The executable does not link or load:

- cudart;
- cuBLAS or cuBLASLt;
- cuDNN;
- cuTENSOR;
- NCCL; or
- another ML runtime.

An NVIDIA driver and compatible device are required at runtime. Building PTX
requires nvcc or a Clang installation with NVPTX support. The precise
description of this backend is “direct CUDA Driver API with
VolvoxAI-owned PTX kernels.”

CUDA PTX and Vulkan SPIR-V occupy similar layers in their respective driver
stacks, but they are different formats. The NVIDIA driver JIT-compiles PTX for
the selected device. The Vulkan driver consumes SPIR-V shader modules.

## Source architecture

The stable backend entry files are:

- [cuda_engine.c](../native/src/backends/cuda_engine.c): host composition
  translation unit;
- [cuda_engine.h](../native/src/backends/cuda_engine.h): private backend
  interface;
- [cuda_kernels.cu](../native/src/backends/cuda_kernels.cu): forward PTX
  composition root; and
- [cuda_training_kernels.cu](../native/src/backends/cuda_training_kernels.cu):
  training and PTQ PTX composition root.

Private implementation fragments are grouped by ownership:

- [cuda/host](../native/src/backends/cuda/host/): Driver API loading, module
  lifecycle, graph memory, profiling, operator launch orchestration, training,
  and PTQ host logic; and
- [cuda/kernels](../native/src/backends/cuda/kernels/): forward, backward,
  optimizer, and PTQ device kernels.

The host fragments are composed into one C translation unit so private backend
types and static helper boundaries remain internal. Device
fragments are composed into two PTX modules:

~~~text
cuda_kernels.cu
    -> forward PTX
       -> native/volvoxai-lite
       -> native/volvoxai

cuda_training_kernels.cu
    -> training/PTQ PTX
       -> native/volvoxai only
~~~

The forward function registry is the single source for declarations, ordered
module resolution, profiler registration, and cleanup of 73 forward PTX
entries. The training registry provides the same ownership for 79 training
and PTQ entries.

CUDA-specific runtime integration is privately composed through:

- [engine_runtime_cuda_f32.inc](../native/src/runtime/engine_runtime_cuda_f32.inc)
  for F32 runtime routing; and
- [training_cuda_trainstep.inc](../native/src/training/training_cuda_trainstep.inc)
  for the resident CUDA Trainer path.

The shared backward planner and optimizer integration remain under
[native/src/training](../native/src/training/).

### Profile boundary

The inference profile:

- compiles the CUDA forward backend and forward PTX only;
- executes F32 and already-authored W8A8 packages;
- contains no compiled training implementation;
- exposes no public training symbols; and
- does not embed the training/PTQ PTX module.

The full profile adds generated Training and Quantization service dispatch,
private Trainer execution and optimizer state, profiling, PTQ authoring, and
the training/PTQ PTX module.

CUDA PTX is separate from the native XZ shader pack.
VOLVOXAI_SHADER_DIR remains a development override for generated Vulkan,
OpenGL, and Metal shader formats; it does not replace CUDA PTX.

### Build dependency tracking

[native/CMakeLists.txt](../native/CMakeLists.txt) discovers device fragments
with CONFIGURE_DEPENDS and attaches every fragment reachable from each CUDA
composition root to the corresponding PTX build command.

The source composition keeps these invariants:

- complete and acyclic local include graphs;
- no device-to-host include path;
- no forward-to-training device dependency;
- complete CMake PTX dependencies for both roots.

Generated PTX embeddings are deterministic build outputs and must not be
edited by hand.

## Build and selection

CUDA is disabled by default. A strict F32 build for compute capability 8.6 is:

~~~bash
cmake -S . -B build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DVOLVOXAI_ENABLE_CUDA=ON \
  -DVOLVOXAI_CUDA_ARCH=86 \
  -DVOLVOXAI_CUDA_FAST_FP32=OFF
cmake --build build/cuda --target volvoxai-lite volvoxai
~~~

VOLVOXAI_CUDA_ARCH is the numeric compute capability, defaults to 75, and must
be at least 61 because the W8A8 QBatchMatMul route uses signed DP4A.
CMake prefers nvcc and otherwise searches for clang++ with NVPTX support. The
Clang path uses -nocudainc and -nocudalib, so CUDA headers and SDK link
libraries are not required.

Select a device with the zero-based `VOLVOXAI_CUDA_DEVICE` environment
variable. Device 0 is the default. Applications select CUDA only through the
generated `CompileModelRequest.policy`: put `"cuda"` in its ordered backend
list.

An explicitly required CUDA provider does not silently change to another
provider when CUDA was not compiled, initialization fails, a required route is
unsupported, or execution fails. For strict native compilation, set the
generated policy to `BACKEND_POLICY_MODE_REQUIRE`, use the sole backend
`"cuda"`, and set `OPERATOR_FALLBACK_FORBID`.

## FP32 numerical contract

CUDA has two build-time F32 policies:

| Policy | CMake option | Compiler behavior |
| --- | --- | --- |
| Strict F32, default | VOLVOXAI_CUDA_FAST_FP32=OFF | nvcc --fmad=false or Clang -ffp-contract=off |
| Fast F32 | VOLVOXAI_CUDA_FAST_FP32=ON | nvcc --fmad=true or Clang -ffp-contract=fast |

Strict F32 keeps eligible multiply and add operations separately rounded.
Fast F32 permits contraction into FMA. FMA performs multiplication and
addition before one final rounding, so its low bits can differ from separate
operations.

Strict F32 does not promise bit-identical CPU and GPU output for every graph.
Parallel reduction order and device math can still differ. It defines the
no-FMA PTX contract and the tolerances checked by the CUDA correctness tests.

Neither policy enables TF32. The manual kernels do not emit tensor-core TF32
instructions. Fast F32 enables ordinary FP32 FMA only; it does not enable
fast-math globally, FP16, BF16, or tensor cores.

## Execution and memory

The backend uses one ordinary CUDA stream. Its graph memory system maps
runtime-owned tensor storage to retained device-capacity slots. Static and
private direct execution can lazily grow a slot and recycle detached transient
allocations through an anonymous capacity pool. Public bounded-dynamic
execution instead uses a fixed reservation described below. Both paths retain:

- up to 8,192 tensor slots;
- a 16,384-entry exact-pointer hash table;
- a fail-safe exact linear lookup;
- containing-slot lookup with byte offsets for interior views;
- stable slot indices with epoch changes on resize; and
- host-dirty and device-dirty state for coherence.

The stream belongs to an explicit, mutex-protected `CudaDeviceState` together
with the Driver API loader, selected physical device, primary context, PTX
modules, and immutable function cache. Device initialization is reference
counted. Every `VxEngineState` instead owns a private CUDA capsule containing
its tensor slots, hash and epoch, replay plan/executable, quantized-activation
LUT, request and training workspaces, optimizer mirrors, profiling records,
and diagnostic counters. Work on the shared stream is serialized, but two
execution contexts never alias graph, request, replay, optimizer, or profile
state.

Intermediate tensors remain on device until a host consumer needs them.
Optimizer-updated weights and Adam moments remain device-authoritative across
CUDA steps. A tensor copy, checkpoint/package save, CPU or other-backend
handoff, graph mutation, or another host consumer materializes affected
weights.

The public task path uses host-owned graph inputs and outputs. Transfers use
synchronous pageable cuMemcpyHtoD_v2 and cuMemcpyDtoH_v2 calls. There is no
public device-buffer binding API, pinned-host allocator, or asynchronous
transfer pipeline.

Training teardown removes transient activation and gradient identities while
retaining stable model-weight slots. Activations and gradients do not yet use
one persistent device arena across Trainer steps.

### Bounded dynamic shapes

CUDA accepts a public bounded-dynamic graph only when compilation proves the
complete declared domain. The proof covers canonical tensor relations, the
qualified operator subset and its shape-dependent predicates, kernel launch
limits, tensor-slot count, and simultaneous host/device resident resources.
An unproved shape, route, launch, or resource bound rejects compilation; it
does not fall back to CPU.

Logical F16 activations are not admitted to a dynamic CUDA domain: the graph
ABI does not provide two-byte activation slots for its F32 compute routes.
Immutable F16 package weights remain supported and are widened, preloaded, and
retained as F32 before the maximum-domain reservation is published.

Context creation bootstraps the minimum-domain graph to preload invariant
weights and derived metadata, then releases every transient bootstrap slot.
It builds one fixed maximum-domain host liveness arena and atomically reserves
the corresponding physical CUDA spans. Concrete shape plans are projected
onto those fixed offsets and must fit the proved capacities. After context
publication, shape rebinding neither grows the host arena nor calls
`cuMemAlloc`; a missing or undersized reserved span fails closed.

A changed shape synchronizes submitted stream work, resets the reserved spans'
logical/coherence views, and commits the new semantic signature only after the
reserved binding succeeds. The previous binding remains usable when candidate
planning fails. Repeating the current signature resets only its per-forward
logical/coherence state; the binding fast path issues no additional stream
synchronization and does not advance shape/capacity generations. Because the
physical pointers and capacities remain fixed, a successful bind can select a
previously captured CUDA Graph for that exact signature instead of invalidating
graphs for other signatures.

### CUDA Graph replay

Qualified static and bounded-dynamic inference can capture and replay a
conservative graph. Each exact shape follows the same state machine:

1. the first eligible forward records routes, launch signatures, and
   read-before-write slots;
2. the next compatible forward stages inputs, captures, validates, and
   instantiates the graph; and
3. later compatible forwards replay the graph executable.

The static allowlist is:

- Linear, Gemm, MatMul, and QBatchMatMul;
- PReLU and Sigmoid;
- Conv2D and Add;
- Concat, MaxPool2D, ResizeNearest2D, Reshape, and Transpose; and
- QConv2D, QAdd, QuantizeLinear, DequantizeLinear, and RequantizeLinear.

The bounded-dynamic allowlist also admits proof-qualified ArgMax,
BatchMatMul, Cast, Clip, Div, Embedding, Equal, Expand, GELU, Gather,
GreaterOrEqual, GroupNorm, LayerNorm, Mul, Not, QGemm, QLinear, ReduceSum,
SiLU, Slice, Softmax, Squeeze, Sub, Unsqueeze, and Where nodes. Their scalar
launch arguments must derive only from immutable node parameters and the exact
shape signature; request values remain in the fixed device buffers staged
before replay. Data-dependent MoE and partial weight-bank routing are excluded.

A bounded context retains at most four replay plans in a deterministic
context-local LRU. A plan owns its exact shape signature and is additionally
keyed by model generation, tensor-slot epoch, capacity generation, and domain
mode. Thus an alternating B1/BN workload can keep both plans, while a fifth
signature destroys the least-recently-used executable. Model, slot, capacity,
or domain changes invalidate all retained plans before a direct OBSERVE pass;
a launch-list mismatch invalidates the selected plan. Debug routing, prefix or
row execution, SDK backends, adapter effects, training, event profiling, and
partial weight banks exclude replay. An unavailable Graph API, ineligible
forward, cache-allocation failure, or failure to begin capture uses ordinary
CUDA launches. A failure after capture has begun, during graph launch, or at
stream synchronization destroys the affected plan and returns a strict
execution failure rather than publishing uncertain output.
If the CUDA Driver cannot destroy an evicted GraphExec, that cache entry keeps
ownership and is quarantined. The current forward uses ordinary launches and
later forwards retry destruction before reusing the entry or requesting
another GraphExec, so the four-object bound remains strict.

The bounded resident proof charges the fixed host-side state for four plans,
four retained maximum-size owned signatures, and the transactional fifth
candidate signature that can coexist until LRU eviction. The CUDA Driver's
internal allocation for up to four requested GraphExec objects is opaque:
compile evidence reports the cache capacity, while the opaque bytes are documented as
requested/unknown and are not silently included in the numeric resident-byte
claim. Runtime route evidence reports `cuda_graph_replay=1` only after a cached
VALIDATE launch and its stream synchronization both succeed for that physical
forward; observe, capture, and fallback passes report `0`.

## F32 inference operators

The F32 dispatcher covers:

- dense and adapters: Linear, MatMul, Gemm, and routed LoRA;
- convolution: Conv1D, Conv2D, and ConvTranspose2D;
- elementwise: Add, Mul, Sub, Div, Where, and Mask;
- activations: ReLU, Sigmoid, GELU, SiLU, Tanh, HardSwish,
  HardSigmoid, LeakyReLU, PReLU, and Clip;
- normalization and reduction: LayerNorm, RMSNorm, GroupNorm, BatchNorm2D,
  final-axis Softmax/LogSoftmax, ReduceSum, ReduceMean, and ArgMax with I32
  output;
- pooling and resize: GlobalAveragePool, AveragePool2D, MaxPool2D, nearest and
  bilinear Resize, and linear 1D interpolation;
- movement and shape: Embedding, Transpose, Expand/Broadcast, axis-0 Gather,
  Pad, Slice, Split, Concat, same-F32 Cast/copy, and copy-like shape
  operations;
- attention: packed-QKV SDPA, query-range SDPA, CrossSDPA, and fused
  CrossAttention;
- mixture of experts: MoERouter and MoELinear; and
- vision/postprocessing: SpatialSoftargmaxY, ProfileX, ProfileY, MeanHeight,
  and NonMaxSuppression.

The runtime also supports fused Add with ReLU/ReLU6, Add3 with
ReLU/ReLU6, and eligible Conv2D-plus-residual-Add paths.

## W8A8 inference

I8/U8 execution covers:

- QLinear, QMatMul, QGemm, QEmbedding, QConv2D, and QAdd;
- QBatchMatMul with rank-2–8 right-aligned ONNX batch broadcasting;
- QSiLU and QGELU with cached 256-byte lookup tables;
- QGroupNorm, QLayerNorm, QSDPA, and query-range QSDPA;
- QArgMax and QMaskedMean;
- typed QuantizeLinear, DequantizeLinear, and RequantizeLinear;
- compatible byte copy and shape aliases; and
- same-domain nearest Resize, MaxPool2D, and Concat.

Byte copy, pooling, resize, and concat require compatible dtype, scale, and
zero point. Byte resize uses nearest/asymmetric/floor behavior.
QBatchMatMul uses independent per-tensor I8/U8 descriptors for both operands
and the output. It accumulates centered products in I32 and rejects a
descriptor whose worst-case dot product could overflow I32 before launch.
Complete groups of four contracted values use signed DP4A. U8 operands and
zero points are shifted into the signed-byte domain, so all I8/U8 pairings and
asymmetric zero points preserve the same exact centered-product contract;
`K % 4` uses a scalar tail. Dynamic route evidence reports cumulative packed
groups and tail values for the context. IMMA is not selected here: arbitrary
rank/broadcast geometry, arbitrary K/N, and decode M=1 would require padded
shape-specific tensor-core layouts plus a second generic tactic whose setup
cost dominates these small dynamic attention products.

## Full-command CUDA training

The native full command's training path:

1. builds and preflights the complete backward command plan;
2. runs the F32 forward graph;
3. seeds weighted cross-entropy loss gradients on device;
4. executes all backward commands with additive gradient destinations;
5. checks trainable gradients for non-finite values;
6. transactionally accumulates each finite microbatch;
7. computes one global gradient norm and clip scale; and
8. applies SGD or AdamW kernels.

Loss, correctness, example count, and status are reduced on device. The full
command reads only small metric/status values for active microbatches; it does
not download full logits or gradients.

Accumulation buffers persist across microbatches. A rejected microbatch does
not partially alter its accumulation window. Once a CUDA accumulation window
exists, its backend is fixed for that window. A failed optimizer apply
discards the window because arbitrary device failures do not provide
transactional rollback.

Required-CUDA training never executes a supported CUDA prefix followed by CPU
fallback. Optional CUDA selection can use CPU only when complete CUDA
preflight declines before CUDA state or an accumulation window is mutated.

### Backward operator coverage

The F32 planner and CUDA dispatcher cover:

- Linear, MatMul, and Gemm input, weight, and optional-bias gradients;
- broadcast Add and Mul gradients, including fused activation gates;
- ReLU, GELU, SiLU, Sigmoid, Tanh, LeakyReLU, HardSigmoid,
  HardSwish, Clip, and PReLU gradients;
- final-axis ReduceSum, ReduceMean, Softmax, and LogSoftmax;
- Embedding, Dropout, and copy-like shape operations;
- LayerNorm, RMSNorm, GroupNorm, and BatchNorm2D;
- Conv2D input, weight, and optional-bias gradients for ungrouped and
  depthwise supported layouts;
- GlobalAveragePool, MaxPool2D, nearest Resize/Upsample, Transpose, Concat,
  and Split;
- SDPA and CrossSDPA; and
- MoELinear and MoERouter.

Runtime-routed LoRA is a forward overlay. Trainable LoRA uses ordinary F32
MatMul/Add graph nodes, so its gradients and AdamW updates follow the same
CUDA contract. W8A8 graphs do not support routed LoRA overlays,
quantized backward, or QAT.

### Training attention

F32 SDPA and CrossSDPA use a CUDA-private workspace for softmax probabilities
and score gradients:

~~~text
P      [B, H, Q, K]
dScore [B, H, Q, K]
~~~

The workspace requires two F32 matrices, grows transactionally, remains
device-resident, and is released during CUDA cleanup. One 64-lane block owns
each logical attention row. Fixed shared-memory reduction trees preserve
repeatable strict-build behavior. Causal rules, supported masks,
deterministic dropout mapping, and all-masked-row behavior are retained.

CrossSDPA currently stages its workspace independently for separately planned
Q, K, and V backward commands.

## Full-profile training ownership

CUDA backward planning, saved values, gradient buffers, loss reduction, and
optimizer state are private to the owning engine/trainer CUDA capsule. The
generated native inference C header exposes no training symbol.
Unsupported required CUDA training work fails complete-plan preflight before
state mutation.

## CUDA W8 authoring

F32-to-W8 authoring is a full-profile capability implemented with Driver API
orchestration and manual CUDA kernels. The canonical destination is row-major
signed I8 with per-output-row metadata.

Supported behavior includes:

- symmetric narrow I8 quantization with zero point 0;
- asymmetric full I8 quantization;
- generated or caller-supplied scale/zero-point tensors and central graph
  references;
- OUT_IN input or device transpose from IN_OUT;
- optional F32-bias to I32 packing using input and weight scales;
- saturation counting; and
- transactional validation of dimensions, metadata, bias, and finite values.

Full-profile materialization and package writing route selected W8 tensors
through this path. Authored tensors feed the W8A8 kernels without a
CPU repack.

Authoring stages the F32 source to CUDA, uses temporary device buffers, and
copies completed package tensors to host. Activation calibration, package
file I/O, quantized backward, and QAT are not CUDA operations.

## Event profiler

The full profile has an opt-in CUDA-event profiler around the central launch
path. Set VOLVOXAI_CUDA_PROFILE_PATH before the first CUDA initialization:

~~~bash
VOLVOXAI_CUDA_PROFILE_PATH=/path/to/trainstep-kernels.csv \
  native/volvoxai train models/my_model --cuda ...
~~~

The variable is read once per CUDA initialization. Profiling a loaded model
covers forward, loss seeding, backward, accumulation, finite checking,
clipping, and optimizer launches. It disables CUDA Graph capture/replay for
the profiled forward so individual launches remain visible.

The CSV columns are:

~~~text
record,scope,complete,entry,grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_bytes,count,total_ms,mean_ms,max_ms
~~~

Rows aggregate by scope, PTX entry, and exact launch signature. Only a scope
row with complete=1 is an authoritative complete native Trainer
profile. CUDA-event time measures device kernel intervals; it excludes
transfers, host planning, allocation, API overhead, file I/O, and
synchronization wait. Transfer-inclusive wall time must be measured
separately.

When profiling is disabled, launch events, aggregate records, CSV output, and
profiling synchronization are not used. Profiler code is absent from the
inference profile.

## Tactics and fusions

Inference includes:

- ungrouped 1x1 Conv2D tiled tactics;
- specialized depthwise 3x3 and 5x5 tactics;
- Conv2D and QConv2D ReLU6 folding;
- Add3 and Conv2D-plus-residual-Add fusion;
- compatible shape and dequantize-after-concat aliases;
- tensor-slot hashing; and
- conservative CUDA Graph replay.

Training includes:

- shared-memory 16x16 dense forward, dInput, and dWeight tiles;
- deterministic 64-lane broadcast and normalization reductions;
- staged 64-lane self/cross-attention rows; and
- ungrouped HWIO 3x3 Conv2D dInput/dWeight tiles with channel-width
  selection.

Generic kernels remain available for supported shapes that do not match a
specialized tactic.

## Current limits

- F32 Conv2D training supports groups equal to 1 or depthwise groups equal to
  input channels. General grouped training convolution is not implemented.
- F32 convolution epilogues are none, ReLU, or ReLU6. The fused
  residual-Add path is restricted to eligible ungrouped 1x1 convolution.
- F32 SDPA and CrossSDPA require head dimension at most 64. Fused
  CrossAttention also requires model width at most 64.
- QSDPA requires head dimension divisible by 4 and at most 64.
- QBatchMatMul requires contiguous rank-2–8 byte operands and output, exact
  `[...,M,K] @ [...,K,N]` matrix axes, and right-aligned ONNX-compatible batch
  dimensions. Vector promotion, output/input aliasing, element counts beyond
  32-bit launch parameters, and an unsafe I32 accumulator bound are rejected.
- Softmax, LogSoftmax, ReduceSum, and ReduceMean training routes are
  final-axis.
- F32 Gather is axis 0. F32 Pad and Slice support rank at most 4; Slice uses
  positive steps.
- Launch dimensions and element counts are bounded by 32-bit kernel
  parameters.
- Public bounded-dynamic CUDA currently requires the base adapter and
  non-decode execution. Dynamic decode and non-base adapter selection fail
  closed until their persistent device layouts have separate complete-domain
  proofs and reservations.
- Examples without built-in F32 CUDA inference include RoPE,
  SSMScan/SelectiveScan, Sin, and Cos.
- There is no public caller-owned device-buffer API.
- Activations and gradients are not persistent across Trainer
  steps.
- Training schedule replay and general autotuning are not implemented.
- Native FP16/BF16, TF32/tensor cores, QAT, quantized backward, and
  multi-GPU collectives are not implemented.

## Validation

Build both CUDA release profiles:

~~~bash
cmake -S . -B build/cuda -G Ninja \
  -DCMAKE_C_COMPILER=clang \
  -DVOLVOXAI_ENABLE_CUDA=ON \
  -DVOLVOXAI_CUDA_ARCH=86 \
  -DVOLVOXAI_CUDA_FAST_FP32=OFF

cmake --build build/cuda --target \
  volvoxai-lite \
  volvoxai
~~~

This compiles and embeds the forward-only and full training/PTQ PTX modules in
their respective artifacts. Physical CUDA execution still requires a usable
NVIDIA Driver API device.
