# Chapter 9C — Inside the CUDA Backend (talking straight to an NVIDIA GPU)

*Read the badges that match you: 🌱 **Idea** (anyone, no code) · 🔧 **Build** (a little code) · 🔬 **Deep**
(engine developers). New here? Follow just the 🌱 sections — they tell the whole CUDA story on their own.*

*Goal: understand VolvoxAI's **CUDA backend** — the opt-in path that runs the same graph on an
**NVIDIA graphics card** — without bundling any of NVIDIA's big software kits. This is a companion to
[Chapter 9](09-native-engine-architecture.md): that chapter built the native engine and its many
backends; this one zooms all the way into the CUDA one, front to back.*

> 🌱 **The big idea.** Chapter 9 showed one small program that finds whatever chip a device has and
> uses it. One of those chips is an **NVIDIA GPU** — the kind of card gamers and AI labs use, with
> thousands of tiny workers that do arithmetic side by side. Talking to that card the usual way means
> installing gigabytes of NVIDIA software. VolvoxAI does something unusual instead: it brings **its
> own** math programs for the card and only borrows the card's built-in **driver** — the small piece
> of NVIDIA software that's already on the machine. So the same one program can light up a big GPU
> when there is one, and still run on a plain laptop when there isn't. This chapter is the whole tour
> of how that works: how it talks to the card, how it keeps the numbers *on* the card, how it can even
> **learn** on the card — and how it stays careful and honest the whole time.

🔧 Everything here lives under `native/src/backends/cuda/` plus the entry files
`cuda_engine.c`, `cuda_engine.h`, `cuda_kernels.cu`, and `cuda_training_kernels.cu`. CUDA is **off by
default**; you turn it on at build time and select it at run time. The canonical, terse spec is
[`docs/cuda.md`](../cuda.md) — this chapter is the friendly, complete walkthrough of the same thing.

🔬 The one-line description to keep in your head: **"direct CUDA Driver API with VolvoxAI-owned PTX
kernels."** No cuBLAS, no cuDNN, no cuTENSOR, no NCCL, no CUDA Runtime (`cudart`). The engine owns the
graph executor, the memory residency, every kernel, the loss, the optimizers, and the quantizer; it
rents exactly one thing from NVIDIA — the stable **Driver API**, resolved at runtime.

---

## 9C.1 What a GPU actually is (and what "CUDA" means)

> 🌱 **Idea.** A normal processor (the **CPU**) is like a few very clever chefs: each can do
> complicated work, but there are only a handful of them. A **GPU** is like a stadium full of line
> cooks: each one only does simple steps, but there are *thousands* of them working at once. Neural
> networks are mostly the same simple step — *multiply two numbers and add* — repeated a giant number
> of times, so a stadium of cooks finishes far faster than a few chefs. **CUDA** is just the name of
> NVIDIA's system for handing that stadium a to-do list. And a **kernel** is one job written for the
> cooks: "every cook, take your own number and double it." You launch a kernel and thousands of cooks
> run it at the same time, each on a different piece of the data.

🔧 Concretely: a CUDA **kernel** is a small function that runs once per **thread**; threads are grouped
into **blocks**; blocks form a **grid**. When VolvoxAI launches a kernel it picks a grid and block
size, and the GPU runs `grid × block` copies of the function, each with its own index. A typical
"one element per thread" launch looks like this (from the engine's launch helper):

```c
// cuda/host/cuda_graph_memory_host.inc
static int cuda_launch_1d(CUfunction function, uint32_t elements, void** args) {
    const unsigned int block = 256;                       // 256 cooks per block
    unsigned int grid = elements / block + (elements % block != 0u);
    return cuda_launch(function, grid, 1, 1, block, 1, 1, 0, args);
}
```

🔬 The device kernels are **freestanding**: they include no CUDA headers and call no NVIDIA library
routine. They even spell out a few math primitives in inline PTX assembly to control rounding
exactly — for example `ex2.approx.f32` for exponentials and `mul.rn.f32` for a separately-rounded
multiply (`cuda_kernels.cu`). That freestanding discipline is what lets the *same* `.cu` source be
compiled by either NVIDIA's `nvcc` **or** Clang's NVPTX backend, with no SDK present.

---

## 9C.2 The freestanding trick: borrow the driver, bring your own kernels

> 🌱 **Idea.** To use an NVIDIA card, most programs install a giant toolbox from NVIDIA and glue it in
> at build time. Then that program *needs* the toolbox forever, and only runs where the toolbox is.
> VolvoxAI refuses that deal. On the machine, NVIDIA has already installed one small, always-present
> piece: the **driver** — the translator between programs and the card. VolvoxAI walks up to that
> driver *at the moment it runs*, taps it on the shoulder, and asks for just the few basic services it
> needs: "make me some memory on the card," "copy these numbers over," "run this job." It brings all
> the actual math itself. Result: **one program file** that turns into a GPU powerhouse where a card
> exists and a quiet CPU program where it doesn't — no reinstalling, no separate build.

🔧 "Ask the driver at runtime" is literally `dlopen` + `dlsym` — the same runtime-loading move Chapter
9 showed for Vulkan and OpenGL (`cuda/host/cuda_driver_host.inc`):

```c
const char* names[] = {"libcuda.so.1", "libcuda.so"};   // the NVIDIA driver, not a toolkit
for (i = 0; i < 2; i++) { cuda_library = dlopen(names[i], RTLD_NOW | RTLD_LOCAL); if (cuda_library) break; }
// then look up each entry point by name:
p_cuInit            = dlsym(cuda_library, "cuInit");
p_cuMemAlloc        = dlsym(cuda_library, "cuMemAlloc_v2");
p_cuLaunchKernel    = dlsym(cuda_library, "cuLaunchKernel");
p_cuMemcpyHtoD      = dlsym(cuda_library, "cuMemcpyHtoD_v2");
// ... a short, fixed list of Driver API functions
```

🔬 The list of functions it resolves is deliberately tiny and stable: init/device query, a **primary
context**, one **stream**, module load/unload, get-function, alloc/free, two synchronous copies, one
launch, and error-name lookup. A few extras are **optional** — the CUDA **Graph** capture functions
(§9C.7) and, in the full profile, the **event timing** functions (§9C.14) — and the backend runs fine
if the driver is too old to provide them. What it never touches: `cudart`, cuBLAS/cuBLASLt, cuDNN,
cuTENSOR, NCCL, or any other ML runtime. The build proves it — there is no `-lcuda*` link flag; the
only related flag is `-ldl`.

---

## 9C.3 PTX: write the math once, let the driver bake it fresh

> 🌱 **Idea.** The card can't read the engine's C code directly. So VolvoxAI writes each math job once,
> and a build step turns it into **PTX** — think of it as a recipe written in the GPU's own
> handwriting. That recipe is tucked *inside* the program file. When the program starts and finds a
> real card, it hands the recipe to the driver, and the **driver bakes it** into the exact instructions
> that particular card understands. One recipe, and every NVIDIA card bakes its own perfect version —
> so VolvoxAI doesn't need a different program for each card.

🔧 The pipeline mirrors the "one source, many targets" discipline from Chapter 9, but for NVIDIA:

```
cuda_kernels.cu ───nvcc or clang(NVPTX)──▶ forward PTX ──┐
cuda_training_kernels.cu ─────────────────▶ training PTX ─┤  embed as bytes (tools/embed_cuda_ptx.py)
                                                          ▼
                                        inside native/volvoxai and native/volvoxai-full
                                                          │  at runtime:
                                          cuModuleLoadDataEx(...)  ← driver JIT-compiles PTX for THIS card
```

The generated PTX is **never checked in or hand-edited**; it's a deterministic build output embedded
by `tools/embed_cuda_ptx.py`. Each `.cu` file is a **composition root** that `#include`s many small
kernel fragments from `cuda/kernels/*.inc`, so the kernels stay organized by topic without becoming
many separate modules.

🔬 There are **two** PTX modules on purpose (see §9C.4): a forward module and a training/PTQ module.
The build offers two float policies — **strict** (`--fmad=false` / `-ffp-contract=off`) and **fast**
(`--fmad=true` / `-ffp-contract=fast`) — chosen with `-DVOLVOXAI_CUDA_FAST_FP32` (§9C.8). You also
pass the target compute capability with `-DVOLVOXAI_CUDA_ARCH` (default 75), though the driver's JIT
step means one PTX still runs across a range of cards. A source-composition test
(`tools/tests/test_cuda_source_composition.py`) checks that the include graph is complete and acyclic,
that no device fragment reaches into host code, and that the forward module never depends on training
kernels.

---

## 9C.4 Two backpacks: the inference profile and the full profile

> 🌱 **Idea.** VolvoxAI ships the GPU program in two sizes. The **small backpack** ("inference") can
> only *run* already-finished models — perfect for a phone, a camera, a shipped app. The **big
> backpack** ("full") can also *train* models and *shrink* them, so it carries extra tools. The small
> one literally does not contain the training tools — they aren't hidden or switched off, they're just
> not packed. That keeps the shipping program lean and means the training machinery can't accidentally
> run where it shouldn't.

🔧 The two backpacks are two build targets:

```bash
cmake --build build/cuda --target volvoxai volvoxai-full
```

- `volvoxai` (inference): the CUDA forward backend + the forward PTX module only.
- `volvoxai-full` (full): adds the full-only opaque `VxTrainer` API, its private training state,
  optimizers, the profiler, W8 authoring, and the training/PTX module.

🔬 The split is a **translation-unit boundary**, not a scatter of `#ifdef`s. In the inference profile
there is no compiled training implementation and no public training symbol, and the
training PTX is simply not embedded. `#if VOLVOXAI_ENABLE_TRAINING` guards the extra host fragments,
and a symbol-boundary test enforces that inference builds stay clean. This is why "does it even
contain a trainer?" has a hard, testable answer instead of a runtime flag.

---

## 9C.5 Strict routing: do it on the GPU, or say so out loud

> 🌱 **Idea.** When you ask for the GPU, VolvoxAI makes a promise: it will run your model on the GPU,
> **or tell you it couldn't** — it will never quietly finish the job on the slow CPU and let you
> believe the GPU did it. Why be so strict? Because a silent switch to CPU hides problems: your model
> looks like it "works on GPU" while secretly crawling. The compile policy states whether CUDA is a
> preference or a requirement and whether operator fallback is allowed. Compilation either produces
> an attested route that satisfies that policy or reports why it could not.

🔧 CUDA is selected when `vx_model_compile` receives a `VxBackendPolicy` whose `backend` is
`"cuda"`. Set `mode = VX_BACKEND_REQUIRE` and
`operator_fallback = VX_OPERATOR_FALLBACK_FORBID` for a strict CUDA route. Unsupported work then
fails compilation; `vx_execution_context_execute` does not retry the graph on CPU.

🔬 The CUDA provider validates the complete route and records provider, device, and route evidence in
the compiled-model report. Full-profile training uses the same all-or-nothing rule internally: the
complete backward plan is preflighted before any CUDA state is mutated, and a latched CUDA failure
prevents the remaining commands from running.

---

## 9C.6 Keep the numbers on the GPU's desk (residency)

> 🌱 **Idea.** The card has its own memory — its own **desk**. The slow part of using a GPU is often
> not the math but *carrying paper back and forth* between the computer's desk and the card's desk.
> So VolvoxAI tries hard to **leave the numbers on the card's desk** between steps. The model's weights
> get carried over once and stay there. The in-between results of a calculation stay there too, only
> coming back when something on the computer side actually needs to read them. To pull this off, the
> engine keeps a little notebook: for each pile of numbers on the card, it writes down which pile on
> the computer it matches, and whether the freshest copy is on the card or on the computer.

🔧 Callers never see device memory. `vx_execution_context_set_input` copies input bytes into
runtime-owned storage, and `vx_result_read` copies an immutable result snapshot back to the caller.
Inside the CUDA provider, a private **slot table** maps runtime-owned tensor storage to device
allocations. Its typed entry points are implementation details:

```c
int cuda_graph_matmul_f32(const float* input, const float* weight, const float* bias,
                          float* output, int rows, int d_in, int d_out);
void cuda_graph_mark_host(const void* host, size_t bytes, int is_weight); // "the CPU changed this"
int  cuda_graph_sync_host(const void* host, size_t bytes, int is_weight); // "CPU needs to read this"
```

🔬 The residency layer (`cuda/host/cuda_graph_memory_host.inc`) is the engine's most safety-critical
code:

- **Runtime-owned storage identifies a slot internally.** Up to **8,192** append-only slots, found
  through a **16,384-entry** open-addressed hash, with an exact linear scan as a fail-safe and a
  *containing-slot* lookup so an interior view (a row, a slice) resolves to its parent allocation plus
  a byte offset.
- **Two dirty bits per slot** (`host_dirty`, `device_dirty`) track where the fresh copy lives, so
  uploads and downloads happen only when truly needed. Weights upload once and stay resident;
  optimizer-updated weights and Adam moments stay device-authoritative across steps until something
  explicitly reads or saves them.
- **Address reuse is the danger, and it's handled.** When the C side frees a temporary tensor, its
  address can be recycled for a *different* tensor later. Before any driver call that might fail, the
  engine **quarantines** dead temporary identities (strips the storage pointer, keeps the device buffer
  as an anonymous orphan to free later) and bumps a **slot epoch** counter, so a recycled address can
  never accidentally read a dead tensor's leftover device data. True model weights are kept resident
  through all of this (`cuda_graph_release_transients`).

Transfers use plain synchronous `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2`; there is deliberately no
pinned-memory allocator or async pipeline yet, which keeps the model simple (§9C.16).

---

## 9C.7 Record the dance once, then replay it (CUDA Graph replay)

> 🌱 **Idea.** Handing the card one job at a time has a little overhead each time — like calling out
> dance moves one by one. If the *same routine* is about to repeat many times (as in inference, where
> every input runs the identical list of steps), VolvoxAI can **record the whole dance once** and then
> tell the card "do the recorded routine again" — much less back-and-forth. It's careful about it: it
> only records simple, well-behaved routines, it watches one full run first to make sure nothing
> surprising happens, and if *anything* about the model or its memory changes, it throws the recording
> away and re-learns it. The recording is only ever a speed-up; it can never change the answer.

🔧 This is NVIDIA's **CUDA Graph** capture-and-replay, wrapped in a careful state machine
(`cuda/host/cuda_graph_lifecycle_host.inc`). Each forward runs in one of a few passes:

```
OBSERVE   → run normally, but record the exact launch list and which inputs are read-before-written
CAPTURE   → replay the same list into a CUDA graph, instantiate it, launch it
VALIDATE  → the fast path: just launch the already-built graph again
(INELIGIBLE → plain launches, always correct)
```

🔬 The allowlist is conservative on purpose — Linear/Gemm/MatMul, PReLU/Sigmoid, Conv2D/Add,
Concat/MaxPool2D/ResizeNearest2D/Reshape, and the physical quantized ops. Anything that would make
replay unsafe **invalidates** the plan: a model-generation change, a slot-epoch change (memory moved),
a debug/prefix/row execution, an SDK-backend or adapter effect, training, event profiling, or any
capture/launch/sync failure. Crucially, if the driver can't capture a stream, the engine **falls back
before a single kernel was withheld**, so that forward still runs to completion. Replay never trades
correctness for speed.

---

## 9C.8 Careful arithmetic: the FP32 promise

> 🌱 **Idea.** Computers can round numbers slightly differently depending on *how* they do the math,
> and those tiny differences can pile up. VolvoxAI wants the GPU's answers to be trustworthy and
> repeatable, so by default it tells the card to do multiplication and addition as **separate,
> carefully-rounded steps** — the same way the CPU does — instead of a faster combined step that rounds
> only once. You can flip on the faster mode if you want raw speed and can accept slightly different
> low bits. Either way, it never turns on the "sloppy fast math" shortcuts, and it never uses the
> special low-precision tensor hardware; it's honest 32-bit arithmetic.

🔧 One build switch picks the policy:

| Policy | CMake option | Compiler behavior |
|---|---|---|
| Strict F32 (default) | `VOLVOXAI_CUDA_FAST_FP32=OFF` | `nvcc --fmad=false` or Clang `-ffp-contract=off` |
| Fast F32 | `VOLVOXAI_CUDA_FAST_FP32=ON` | `nvcc --fmad=true` or Clang `-ffp-contract=fast` |

🔬 Strict F32 keeps eligible multiply/add pairs separately rounded (no fused multiply-add), which is
what defines the no-FMA PTX contract that the correctness tests measure tolerances against. It does
**not** promise bit-identical CPU/GPU output for every graph — parallel reduction order still differs —
it promises the *rounding rule*. Neither policy enables **TF32** or tensor cores; the kernels never
emit tensor-core instructions. Some spots pin rounding by hand regardless of the flag (inline
`mul.rn`/`add.rn`), and softmax deliberately makes a masked `-infinity` logit an exact zero rather
than `exp(-80)`, matching the CPU reference.

---

## 9C.9 What it can run forward (F32 inference)

> 🌱 **Idea.** "Forward" just means *running* a finished model to get an answer (Chapter 1). The GPU
> backend knows how to do this for a big menu of the building blocks you met earlier — the layers that
> multiply and combine numbers, the "activation" shaping steps, the normalizers, the pooling and
> resizing steps, the attention that lets a model look back at earlier words, and the vision
> post-processing that turns a heat-map into boxes. If a model is built only from blocks on the menu,
> the whole thing runs on the card.

🔧 The F32 dispatcher covers (from `docs/cuda.md`, and the `cuda_graph_*_f32` entries in
`cuda_engine.h`):

- **Dense & adapters:** Linear, MatMul, Gemm, routed LoRA overlay.
- **Convolution:** Conv1D, Conv2D, ConvTranspose2D.
- **Elementwise & activations:** Add/Mul/Sub/Div/Where/Mask; ReLU, Sigmoid, GELU, SiLU, Tanh,
  HardSwish, HardSigmoid, LeakyReLU, PReLU, Clip.
- **Norm & reduce:** LayerNorm, RMSNorm, GroupNorm, BatchNorm2D, final-axis Softmax/LogSoftmax,
  ReduceSum/ReduceMean.
- **Pool & resize, movement & shape:** pooling, nearest/bilinear resize, Embedding, Transpose,
  Expand, axis-0 Gather, Pad, Slice, Split, Concat, copy-like shapes.
- **Attention & MoE & vision:** packed-QKV SDPA, query-range SDPA, CrossSDPA, fused CrossAttention;
  MoERouter, MoELinear; SpatialSoftargmaxY, ProfileX/Y, MeanHeight, NonMaxSuppression.

🔬 The runtime also fuses a few common tails — Add with ReLU/ReLU6, Add3 with ReLU/ReLU6, and eligible
Conv2D-plus-residual-Add — so the fused pattern is one launch. A handful of ops are intentionally
absent from CUDA F32 (ArgMax, RoPE, SSM/SelectiveScan, Sin, Cos); a model needing
those either routes those nodes elsewhere or is out of scope for a pure-CUDA run.

---

## 9C.10 Running the tiny-integer model (W8A8 inference)

> 🌱 **Idea.** Chapter 6 shrank a model by storing its numbers as small 8-bit integers instead of full
> 32-bit decimals — 4× smaller, and integer math is cheap. The GPU backend can run those shrunk models
> **as integers**, not by secretly blowing them back up to decimals. It carries the little "scale"
> tags each layer needs and does the integer arithmetic directly on the card.

🔧 Physical I8/U8 execution covers QLinear/QMatMul/QGemm/QEmbedding/QConv2D/QAdd, QSiLU/QGELU (with
cached 256-byte lookup tables), QGroupNorm/QLayerNorm, QSDPA and query-range QSDPA, QArgMax/QMaskedMean,
typed Quantize/Dequantize/Requantize, byte copies and shape aliases, and same-domain nearest
Resize/MaxPool2D/Concat. The entry points are the `cuda_graph_q*_i8u8(...)` functions in
`cuda_engine.h`.

🔬 Byte copy, pooling, resize, and concat require **compatible dtype, scale, and zero point** — the
backend won't silently reinterpret one quantization domain as another. This is real integer inference,
separate from the F32 path, and it reuses the same residency and (for the allowlisted ops) replay
machinery above.

---

## 9C.11 Learning on the GPU (native training)

> 🌱 **Idea.** The full backpack can do more than *run* a model — it can **teach** one, right on the
> card. Teaching (Chapter 4–5) means: run the model forward, measure how wrong it was, push that error
> **backward** to find how each weight should change, then nudge the weights. Doing all of that on the
> card keeps the numbers on the card's desk (fast) and keeps your training data on your machine
> (private). And it's transactional — a careful all-or-nothing: if a batch produces a broken number,
> that batch is thrown out cleanly instead of half-updating the model.

🔧 A `native/volvoxai-full` training step runs entirely on device:

1. build and **preflight** the whole backward command plan (reject early if any kernel is missing);
2. run the F32 forward graph;
3. seed weighted cross-entropy loss gradients on device;
4. execute every backward command, adding into gradient destinations;
5. check trainable gradients for non-finite values;
6. **transactionally** accumulate each finite microbatch;
7. compute one global gradient norm and clip scale;
8. apply **SGD** or **AdamW**.

The opaque `CudaTrainingWindow` (declared in `cuda_engine.h`) owns the accumulated gradients and the
tiny scalar reductions across microbatches.

🔬 A few design points worth seeing:

- **The backward menu mirrors the forward menu** — Linear/MatMul/Gemm gradients, broadcast Add/Mul
  (including fused activation gates), all the activation gradients, final-axis reductions/softmax,
  Embedding/Dropout/shape ops, LayerNorm/RMSNorm/GroupNorm/BatchNorm2D, ungrouped and depthwise
  Conv2D, pooling/resize/transpose/concat/split, SDPA and CrossSDPA, and MoE.
- **Backward is dispatched by name.** The shared planner speaks in WGSL-style commands
  (`matMulBackward`/`input_main`, `conv2DBackward`/`weight_main`, …); a translation layer
  (`cuda_training_command()` in `cuda/host/cuda_training_scheduler_host.inc`) maps each to a CUDA
  kernel. If a name doesn't map, preflight declines the whole plan — it never runs a partial one.
- **A rejected microbatch changes nothing.** Accumulation buffers persist across microbatches; a
  non-finite batch is dropped without touching the accumulation window. A failed optimizer apply
  discards the window (arbitrary device failures give no transactional rollback), which is the honest
  choice.
- **SDPA/CrossSDPA training** uses a private device workspace for softmax probabilities and score
  gradients, grown transactionally and released on cleanup, with fixed 64-lane reductions so a strict
  build is repeatable.

---

## 9C.12 Shrinking a model on the GPU (W8 authoring / PTQ)

> 🌱 **Idea.** Chapter 7 turned a trained model into its small integer version. The full backpack can
> do that shrinking **on the card** too. The clever, careful part: it never touches the real output
> until every check has passed. It works on scratch paper first — figure out the scale for each row,
> check for bad numbers, check the bias fits — and only when *all* of that is clean does it write the
> finished integer weights. So a bad input can never leave you with a half-shrunk, corrupted model.

🔧 The private full-profile implementation authors canonical row-major W8 with this internal entry:

```c
int cuda_training_quantize_w8_f32(const float* source, int8_t* output, float* scales,
        int32_t* zero_points, uint32_t rows, uint32_t columns,
        uint32_t transpose_source, uint32_t scheme, uint32_t scale_policy,
        const float* bias, float input_scale, int32_t* packed_bias,
        uint64_t* saturation_count, uint32_t* out_status);
```

Scheme 0 is symmetric narrow-I8 (zero point 0, `scale = max(|row|)/127`); scheme 1 is asymmetric
full-I8. `transpose_source` reads an `IN_OUT` master without a host transpose; optional F32 bias packs
to I32 using the input and weight scales.

🔬 The kernels (`cuda/kernels/cuda_quantize_w8_kernels.inc`) implement a **transactional pipeline**:
`reset → reduce → derive scratch params → validate bias → commit params → quantize → sum saturation →
pack bias`. Destination arrays (scales, zero points, bytes, I32 bias, saturation total) are written
only by a later stage *after* a shared status word is still zero, and the status word carries bit-flags for
each failure class (invalid argument / non-finite source / bad params / bad bias). Rounding uses f64
`cvt.rni` to match the CPU quantizer bit-for-bit. Authored tensors feed the W8A8 kernels of §9C.10
with no CPU repack.

---

## 9C.13 Full-profile ownership

> 🌱 **Idea.** Training needs temporary activations, gradients, optimizer state, and a backward plan.
> The full command owns all of them for the duration of a training step; inference applications never
> receive mutable handles to that state.

🔧 CUDA backward planning, saved values, gradient buffers, and optimizer state are private to
`native/volvoxai-full`. The base `volvoxai.h` header remains inference-only and consists of the opaque
`VxRuntime → VxModel → VxCompiledModel → VxExecutionContext → VxResult` lifecycle. The full-only
`volvoxai_full.h` header adds an opaque `VxTrainer`; it does not expose mutable graph or optimizer
storage.

🔬 The Driver loader, physical device, primary context, PTX modules, function cache, and one stream
live in a mutex-protected, reference-counted `CudaDeviceState`. Each `VxEngineState` has a separate
CUDA capsule for tensor residency, replay, activation LUTs, workspaces, optimizer mirrors, profiler
records, and counters. The shared stream is serialized; mutable graph or training state is never
shared between contexts.

🔬 JavaScript training uses the retained `Trainer` object. Its step result exposes copied gradients
and stable `updatedTensorNames`; checkpoint export is a separate operation after the step. Native
training uses the corresponding private-step, explicit-commit/rollback `VxTrainer` lifecycle.

---

## 9C.14 A stopwatch for the kernels (the event profiler)

> 🌱 **Idea.** If you want to know *which* steps are slow on the card, the full backpack has a built-in
> stopwatch. You switch it on with one setting, run your model, and it writes a little spreadsheet of
> how long each kind of job took. It measures only the card's own working time — not the time spent
> carrying numbers back and forth — so you have to keep that in mind when reading it.

🔧 Set one environment variable before the first CUDA init:

```bash
VOLVOXAI_CUDA_PROFILE_PATH=/path/trainstep-kernels.csv \
  native/volvoxai-full train models/my_model --cuda ...
```

The CSV aggregates by scope, PTX entry, and exact launch signature:
`record,scope,complete,entry,grid_*,block_*,shared_bytes,count,total_ms,mean_ms,max_ms`.

🔬 CUDA-event timing measures device kernel intervals only — it excludes transfers, host planning,
allocation, API overhead, and sync waits, so transfer-inclusive wall time must be measured separately.
Turning the profiler on **disables CUDA Graph capture/replay** for the profiled forward, so each launch
stays individually visible; a `complete=1` scope row is the authoritative sign of a complete native
full-command training profile. When profiling is off, none of this code runs, and it's absent entirely from the inference
profile.

---

## 9C.15 Going faster: tactics and fusions

> 🌱 **Idea.** For every job there's a plain, always-correct way to do it, and — for the shapes that
> show up a lot — a specially-tuned faster way. VolvoxAI keeps both: the plain version guarantees a
> right answer for any shape, and a hand-picked "tactic" speeds up the common cases. It also glues
> neighboring steps together (a "fusion") so the card does one combined job instead of two.

🔧 Inference tactics: ungrouped 1×1 Conv2D tiles, specialized depthwise 3×3/5×5, Conv2D/QConv2D ReLU6
folding, Add3 and Conv2D+residual-Add fusion, shape/dequantize-after-concat aliases, tensor-slot
hashing, and the conservative Graph replay of §9C.7. Training tactics: shared-memory 16×16 dense
forward/dInput/dWeight tiles, deterministic 64-lane reductions, staged 64-lane attention rows, and
ungrouped HWIO 3×3 Conv2D tiles with channel-width selection.

🔬 A trade-off to know honestly: the **generic** (non-tactic) kernels favor a simple, parity-matching
shape over speed — several recompute row statistics per element, so they can be quadratic in the
feature width. They're the correctness floor, with the tactics as the fast path for common shapes.
There is no autotuning; tactic selection is host-side heuristics, and any shape that misses every
tactic falls to the generic kernel.

---

## 9C.16 What it doesn't do (the honest limits)

> 🌱 **Idea.** A good manual says what a tool *can't* do, so you're never surprised. The CUDA backend
> is focused, not universal.

🔬 Current limits (from `docs/cuda.md`):

- **No native FP16/BF16 math** — 16-bit package weights are widened and computed as F32.
- **No TF32 or tensor cores; no cuBLAS/cuDNN/NCCL; no CUDA Runtime.**
- **No quantized backward and no QAT on CUDA; no multi-GPU.**
- **Conv training** supports groups = 1 or depthwise (= input channels); convolution epilogues are
  none/ReLU/ReLU6; the fused residual-Add path is restricted to eligible 1×1 conv.
- **Attention** needs head dim ≤ 64 (fused CrossAttention also width ≤ 64; QSDPA head dim divisible by
  4 and ≤ 64). Softmax/reductions are final-axis; F32 Gather is axis 0; Pad/Slice are rank ≤ 4.
- Launch sizes and element counts are bounded by **32-bit** kernel parameters.
- **No public caller-owned device-buffer API**, no pinned-host staging, and no async transfer pipeline;
  full-command training owns its transient activations and gradients privately.

None of these are bugs — they're the drawn edges of a deliberately small, auditable backend.

---

## 9C.17 How we know it's right (validation)

> 🌱 **Idea.** None of this is trusted on faith. There's a stack of tests that build the GPU backend
> and check that every piece gives the right answer — the running, the shrinking, the learning, even
> that the shipped small backpack really contains no training tools. If a real NVIDIA card is present,
> a test that fails is treated as a genuine defect, not shrugged off.

🔧 The focused suite builds and runs the CUDA kernel, runtime-routing, training, PTQ, and
profile-boundary tests, plus pure-Python checks for source composition, the strict/fast build contract,
and deterministic PTX embedding:

```bash
python3 -B -m unittest \
  tools.tests.test_cuda_source_composition \
  tools.tests.test_cuda_fp32_contract \
  tools.tests.test_embed_cuda_ptx
```

🔬 CUDA tests use CTest skip code 77 **only** when the Driver API or a usable device is unavailable;
once a device is found, any PTX load, JIT, module-resolution, validation, or execution error is a test
failure — never a skip. Correctness against the CPU/JS reference is the job of the parity harness under
`tests/parity/`, which is its own topic (see [`docs/testing.md`](../testing.md)).

---

## 9C.18 The CUDA backend in one picture

> 🌱 **Idea recap.** The same model you ran in a browser, and then as a small program on a device, can
> also run on an NVIDIA graphics card — by borrowing only the card's built-in driver and bringing its
> own math. It keeps the numbers on the card to go fast, records repeated routines to go faster,
> stays careful with its arithmetic, refuses to fake GPU work on the CPU, and — in the big version —
> can even train and shrink models on the card, always in a careful all-or-nothing way.

🔧

```
 graph.json + weights
          │
          ▼
 VxRuntime → VxModel → vx_model_compile("cuda") → VxCompiledModel
                                                       │
                                                       ▼
                                      VxExecutionContext → VxResult
                                                       │
                ┌──────────────────── CUDA provider ───┴────────────────────┐
                │ dlopen libcuda · embedded PTX JIT · private slot table    │
                │ F32/W8A8 kernels · optional CUDA Graph replay             │
                │ full only: backward plan · optimizer · W8 authoring       │
                └────────────────────────────────────────────────────────────┘
                  strict FP32 by default · no cuBLAS/cuDNN/cudart · no TF32/tensor cores
```

🔬 **The takeaways:**

- **"Driver API + own PTX" is the whole identity.** One rented dependency (the stable Driver API),
  everything else owned and auditable — no cuBLAS/cuDNN/cudart/NCCL, resolved at runtime so one binary
  spans "big GPU" and "no GPU."
- **Residency is the core abstraction.** Runtime-owned storage maps privately to device slots; weights
  stay on the card; address-reuse hazards are handled by quarantine and epochs.
- **Honesty is designed in.** Compile-time route attestation, latched training failure, transactional
  accumulation and W8 authoring, and replay that never withholds a kernel all express the same rule:
  satisfy the selected CUDA policy or report failure.
- **It follows the public native lifecycle.** The backend route is chosen at model compilation, then
  reused by execution contexts; result snapshots remain readable independently of context reuse.

**Next:** [Chapter 11 — Glossary & Next Steps →](11-glossary-and-next-steps.md)

*(Prefer the terse engineering spec? It's [`docs/cuda.md`](../cuda.md). Coming back from the native
tour? That's [Chapter 9 — The Native Engine](09-native-engine-architecture.md).)*
