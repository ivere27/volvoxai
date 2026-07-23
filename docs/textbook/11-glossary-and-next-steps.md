# Chapter 11 — Glossary & Next Steps

*Read the badges that match you: 🌱 **Idea** (anyone, no code) · 🔧 **Build** (a little code) · 🔬 **Deep**
(engine developers).*

*Goal: one place for every term, a concrete path for **each track**, and exercises that turn
reading into skill.*

> 🌱 **You made it — here's what you now understand.** An AI is not magic and not a brain. It's a
> **graph of tiny math steps** (mostly multiply-and-add) run over **grids of numbers** (tensors),
> using a pantry of learned numbers (weights). *Running* it once is a forward pass. It **learns** by
> guessing, measuring how wrong it was (the loss), and passing the blame backward to nudge every
> knob a little better — millions of times. Once it's smart, you **shrink** it by rounding its
> numbers (quantization) so it fits on a phone or a robot, and clever engineering makes it **run
> fast** without changing the answer — in a browser, or as a small program right on the device. If
> you followed the 🌱 thread, you genuinely understand how modern AI works. The glossary and paths
> below help you go further, at whatever depth you want.

---

## 11.1 Glossary

> 🌱 The glossary is for *everyone* — skim it, and come back whenever a word feels fuzzy. The
> plain-language idea comes first in each entry; the 🔬 details follow.

**Activation** — a tensor of intermediate values flowing between ops (as opposed to a *weight*).
Kept in fp32 by VolvoxAI (int8 on the native quantized path).

**AdamW** — the default training optimizer (Chapter 5): each weight gets an adaptive step size from
two running gradient *moments*, plus decoupled weight decay (the "W").

**Anchor** — a fixed reference box (a prior) that a detector adjusts, instead of predicting a box
from scratch. EfficientDet-Lite0 uses 9 per grid cell → 19,206 total.

**Arithmetic intensity (roofline)** — FLOPs done per byte moved from memory. High intensity (a 1×1
conv, a matmul) is *compute-bound*, where SIMD helps; low intensity (an elementwise `Add`) is
*memory-bound*, where fusion and buffer reuse help instead (Chapter 8).

**Attention (SDPA)** — the transformer mechanism where each token compares its **Query** to every
token's **Key** and blends their **Values** by similarity. "Which earlier words matter to me?"

**Autoregressive** — generating a sequence one token at a time, feeding each output back as input.

**Backbone** — the feature-extractor stage of a vision model (here, EfficientNet-Lite0).

**Backend** — a concrete executor for the graph's ops. Browser backends are the four *tiers*;
native backends are CPU, Vulkan, OpenGL/GLES, Metal, and NNAPI. VolvoxAI picks one per node.

**Backpropagation** — computing every weight's gradient by applying the chain rule one op at a time,
walking the graph in reverse; each forward op has a *backward twin* (Chapter 4).

**BiFPN** — Bi-directional Feature Pyramid Network; fuses features across resolutions with
resize/pool/add so every scale has both detail and meaning.

**BPE (Byte-Pair Encoding)** — the tokenizer algorithm: start from bytes, repeatedly merge the
most frequent adjacent pair, per a learned merge list.

**Broadcast** — stretching a smaller tensor to match a larger one in an element-wise op (e.g.
adding a per-channel bias `[C]` to `[N,H,W,C]`).

**Calibration** — running a trained model on representative sample inputs to *observe* activation
ranges, so post-training quantization can choose their scales (Chapter 7).

**Causal mask** — restricting attention so position *q* only sees positions `≤ q`; makes a
left-to-right generator.

**Channel** — one "feature plane" of a tensor (the `C` in NHWC). Input images have 3 (RGB);
hidden layers have many.

**Convolution (Conv2D)** — slide a small learned filter over an image, dot-product at each spot,
to detect a pattern everywhere. **Depthwise** = per-channel spatial filter; **Pointwise** = 1×1
channel mixer; the two together (**depthwise-separable**) are cheap and power the backbone.

**Cross-entropy** — the language-model training loss: how wrong the predicted next-token
distribution is versus the true token. Its gradient seeds the backward pass (Chapter 4).

**CUDA** — NVIDIA's system for running your own programs (**kernels**) on their GPUs. VolvoxAI's CUDA
backend uses only the NVIDIA **Driver API** plus its own **PTX** kernels — no cuBLAS/cuDNN/cudart
(Chapter 9C).

**CUDA Graph (capture / replay)** — record a fixed sequence of GPU launches once, then replay the whole
recording with far less per-launch overhead. VolvoxAI uses it as a conservative inference speed-up that
never changes the answer (Chapter 9C §9C.7).

**Dequantize** — convert int8 back to float: `r = (q − zero_point) × scale`.

**dlopen / dlsym** — load a shared library and look up its functions *at runtime* (not link
time). How VolvoxAI's native binary uses a GPU driver (`libvulkan`, `libGL`) without linking any
GPU SDK — the "no static GPU dependency" design.

**Driver API** — the small, stable set of low-level NVIDIA functions (make device memory, copy, launch a
kernel) that live in the always-present GPU driver. VolvoxAI resolves it at runtime via `dlopen`, so no
CUDA toolkit is linked (Chapter 9C §9C.2).

**Dropout** — randomly zeroing a fraction of activations *during training* (off at inference) so the
model can't over-rely on any single unit (Chapter 5).

**Embedding** — a learned vector that represents a discrete token; the `Embedding` op is a table
lookup.

**Epoch** — one full pass of the training loop over the entire training dataset (Chapter 5).

**Forward pass / Inference** — running the graph once, input → output. VolvoxAI runs this *and* its
reverse, the backward pass (Part II).

**Fusion** — merging adjacent ops (e.g. Conv+ReLU) so intermediate data is written once.

**GELU / ReLU / ReLU6 / Sigmoid** — nonlinearities; the "decision curves" between linear layers.
Without them, stacked MatMuls collapse into one.

**GEMM** — GEneral Matrix Multiply; the heavily-optimized routine most fast conv/matmul paths
reduce to.

**Gradient** — for a given weight, the slope `dLoss/dweight`: which way, and how fast, the loss
changes if that weight moves. The backward pass computes one per weight (Chapter 4).

**Gradient accumulation** — summing gradients over several small microbatches before one optimizer
step, to emulate a large batch that wouldn't fit in memory (Chapter 5).

**Gradient checkpointing** — saving only some forward activations and *recomputing* the rest during
the backward pass, trading extra compute for much lower memory (Chapter 4).

**Graph** — the model's op list: nodes (ops) connected by named tensors. Stored as `graph.json`.

**Head** — the final task-specific layer(s): the LM head (→ vocabulary logits) or the detector's
class/box heads.

**im2col** — "image to columns"; unfold conv input patches into a matrix so a conv becomes a GEMM.

**int8 / fp16 / fp32** — 8-bit integer / 16-bit float / 32-bit float number formats (1 / 2 / 4
bytes). See Chapter 6.

**KV-cache** — caching past tokens' Keys and Values so each generation step only computes the new
token's attention. In JavaScript it belongs to an ExecutionContext and is controlled through
context.decode.seed(), step(), and reset().

**LayerNorm / RMSNorm** — normalize a vector (mean 0, variance 1, then learned scale/shift) to
keep deep-network numbers stable.

**Learning rate** — the size of each optimizer step; the single most important training knob
(Chapter 5).

**Logits** — raw, un-normalized scores (pre-softmax/sigmoid). The LM head and class head emit
these.

**LoRA (Low-Rank Adaptation)** — fine-tuning by freezing the base weights and training a small
low-rank adapter beside each Linear: a fraction of the cost, and a tiny shareable delta (Chapter 5).

**Loss** — a single number measuring how wrong an output is; training minimizes it (Chapter 4).

**MatMul** — matrix multiply; the core feature-mixing op of transformers.

**MBConv** — Mobile inverted BOTTLENECK conv block: expand → depthwise → project, with a residual.
The backbone's repeating unit.

**NHWC / NCHW** — tensor dimension order (batch, height, width, channels) vs (batch, channels,
height, width). VolvoxAI vision models use NHWC.

**naga** — the Rust tool VolvoxAI uses to cross-compile one WGSL shader into SPIR-V (Vulkan),
GLSL (OpenGL), GLSL ES, and MSL (Metal). Generated shader output is not the same as runtime
support; the native dispatcher must wire a backend wrapper for an op to run there.

**NMS (Non-Max Suppression)** — postprocess that removes overlapping duplicate detections,
keeping the highest-scoring box per object.

**NNAPI** — Android's Neural Networks API; VolvoxAI's native engine can dispatch to it on Android
(`native/src/backends/nnapi_engine.c`; built with the Android NDK CMake toolchain).

**Node** — one entry in the graph: an op plus its input/output tensor names and parameters.

**Op / Operation / Kernel** — a single math routine (Add, Conv2D, SDPA…). "Op" is the graph-level
name; "kernel" is a specific implementation of it.

**Optimizer** — the rule that turns each weight's gradient into an actual update. VolvoxAI ships
**SGD** and **AdamW** (Chapter 5).

**Quantization** — representing weights/activations with fewer bits via `scale` + `zero_point`.

**PTQ (Post-Training Quantization)** — quantizing an already-trained model by calibrating ranges and
packing weights, with no retraining (Chapter 7).

**PTX** — NVIDIA's portable GPU instruction format — a "recipe in the GPU's own handwriting." VolvoxAI
compiles its `.cu` kernels to PTX, embeds it in the binary, and the driver JIT-compiles it for the exact
card at runtime (Chapter 9C §9C.3).

**QAT (Quantization-Aware Training)** — training *with* simulated int8 rounding in the forward pass
so the weights learn to tolerate it; gradients flow via the straight-through estimator (Chapter 7).

**Residency (device-resident)** — keeping a tensor's data on the GPU between steps instead of copying it
back to the CPU each time. VolvoxAI's CUDA backend maps each host pointer to a device slot and keeps
weights resident, so it copies only when the CPU truly needs the data (Chapter 9C §9C.6).

**Residual (skip connection)** — adding a block's input to its output (`out = x + f(x)`) so
information and gradients survive deep stacks. In both models.

**Safetensors** — the standard binary file format for the weights.

**Scale / Zero-point** — the two numbers of a quantization recipe: tick size, and which integer
means real 0.

**SGD (Stochastic Gradient Descent)** — the simplest optimizer: step each weight opposite its
gradient, `w -= lr·grad` (Chapter 5).

**Softmax** — turns a vector of scores into a probability distribution (positive, sums to 1).

**SPIR-V** — the binary shader format Vulkan consumes; `naga` compiles VolvoxAI's WGSL to it.

**Prefill / Decode** — the two phases of text generation: *prefill* runs the prompt once to fill the
KV-cache; *decode* runs one new token at a time using the cache. JavaScript callers use
ExecutionContext.decode.seed() and step(), and read each stable ExecutionResult by output name.
Native applications use their VxExecutionContext and declared VxResult outputs; the public C API
does not expose separate prefix/row functions.

**Straight-through estimator** — the QAT trick of treating the non-differentiable round-to-int step
as the identity in the backward pass, so gradients keep flowing (Chapter 7).

**Tensor** — a multi-dimensional array of numbers with a shape; the only data type in the engine.

**Tier** — one of VolvoxAI's browser providers (WebNN / WebGPU / WASM / CPU), selected and fixed by
Model.compile() policy.

**Token** — a chunk of text (word/sub-word/byte) mapped to an integer id.

**Weight** — a number learned during training, read-only at inference.

---

## 11.2 Three paths onward — pick your track

Everyone finishes this book in a different place. Here's where to go next depending on which rung you
were reading.

### 🌱 Idea track — "I want to understand more, still without code"

You don't need to touch the repo to keep learning:

- **Re-read the 🌱 thread as one story.** Skim just the "big idea" boxes, Chapter 1 → 9, back to back.
  It's a complete, plain-language account of how AI works — it lands harder the second time.
- **Watch a model actually run.** Ask someone with the repo to run the `detect` command
  (§11.3) on your photo, and match what happens to the stages in Chapter 3.
- **Explain it to someone else.** Try to describe "what attention does" or "why quantization shrinks a
  model" in your own words. Teaching it is the real test of understanding.
- **Do the 🌱 exercises** in §11.4 — they need only pen, paper, and the ideas you already have.

### 🔧 Build track — "I can code a little; show me it working"

Read the actual (readable) kernels and make small changes:

1. **Four naive kernels** — `ts/ops/add.ts`, `embedding.ts`, `layerNorm.ts`, `matMul.ts`. Each is a
   few dozen lines and maps straight to Chapters 1–2.
2. **The two hearts** — `ts/ops/sDPA.ts` (attention) and `ts/ops/conv2D.ts` (convolution).
3. **The executor** — `ts/backends/CPUEngine.ts`: the `for (node of graph.nodes)` loop + `switch`.
   *This is the whole runtime.*
4. **Run the models** (§11.3), then do the 🔧 exercises in §11.4 — including *adding your own op*.

### 🔬 Deep track — "I want to work on the engine"

Read in this order to go from "I get the concepts" to "I can modify the engine":

1. **The data model** — `ts/core/Tensor.ts`, `ts/core/Graph.ts`. Tiny; read fully.
2. **The executor** — `ts/backends/CPUEngine.ts` (the loop + dispatch).
3. **Four naive kernels** — `ts/ops/add.ts`, `embedding.ts`, `layerNorm.ts`, `matMul.ts`.
4. **The two models' graph documents** — skim `models/tinystories_1m/graph.json` and
   `models/efficientdet_lite0_fp32/graph.json`. Match nodes to Chapters 2–3.
5. **The attention + conv kernels** — `ts/ops/sDPA.ts`, `ts/ops/conv2D.ts`.
6. **Quantization** — `ts/ops/dequantizeLinear.ts`, then `native/src/kernels/quant_cpu_opt.c`.
7. **Optimization** — diff `ts/ops/conv2D.ts` against `native/src/kernels/conv_f32_opt.c` while reading
   `docs/microkernel_optimization_guide.md` and `docs/xnnpack_optimization_guide.md`.
8. **The GPU tier** — `shaders/{inference,training}/*.wgsl` and `ts/backends/GraphExecutor.ts`.
9. **The native engine** (Chapter 9) — `native/include/volvoxai.h` + `native/src/runtime/engine.c`,
   `native/src/runtime/engine_runtime.c` (`run_node`), then `native/src/backends/vulkan_engine.c` (see the
   `dlopen` at the top). `native/cli/main.c` is the fixed runner;
   `examples/native_task_cli/main.c` the opt-in task wrappers.

`docs/operation_list.md` is the per-op × per-backend support matrix — your reference map, and
[ARCHITECTURE.md](../../ARCHITECTURE.md) is the source map and dependency rules.

---

## 11.3 Run the models yourself (🔧🔬)

```bash
# Build the opt-in image/vocabulary/task frontend.
make -C examples native_task_cli

# Raw graph runner — dump the logits tensor for a fixed set of tokens.
make build_native
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32

# Object detector — decode an image into ranked boxes.
examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=photo.png --image-normalize raw-255 \
  --boxes boxes --scores scores --max-det 20

# In Node (WASM / pure-JS tiers), smoke-test any graph package:
node bin/volvox.js run --model models/tinystories_1m/model.safetensors --backend wasm
```

Add `--debug` to a task example command to see per-node timing. This is a
direct way to observe where execution time goes and how Chapter 8's
optimizations change it.

---

## 11.4 Exercises (reading → skill)

> 🌱 **No-code exercises** (pen, paper, and the ideas you already have):
>
> 1. **Draw the pipeline.** From memory, sketch the boxes-and-arrows for turning "Once upon a time"
>    into the next word (Chapter 2). Then check it against §2.2.
> 2. **Round a number by hand.** A ruler has ticks every `0.02`. Which tick is nearest to `0.31`? Turn
>    that tick back into a real number. How far off are you? Now imagine ticks every `0.2` — worse or
>    better? (That's quantization, Chapter 6.)
> 3. **Explain "attention" to a friend** in three sentences, using the "everyone holds up a sign"
>    picture from §2.5b.

🔧🔬 **Code exercises:**

1. **Trace by hand.** Take the sequence `[5, 5]` (two identical tokens) and a made-up 2-dim
   embedding. Walk `Embedding → Add(position) → LayerNorm` with pen and paper. Confirm the shapes
   match `graph.json`.
2. **Break causality.** In `ts/ops/sDPA.ts`, change `k <= q` to `k < seq_len`. Predict what
   happens to generated text and why. (Then revert.)
3. **Quantize a weight.** Pick `scale = 0.02`, `zero_point = -5`. Quantize `r = 0.31`, then
   dequantize it back. Report the round-trip error. Now try `scale = 0.002`. What did precision
   cost you in range?
4. **Count the FLOPs.** For the first `Conv2D` (stem: 320×320×3 → 160×160×32, 3×3 filter),
   estimate the multiply-adds. Compare to a 1×1 pointwise conv of the same output size. Why is
   depthwise-separable cheaper?
5. **Add an op.** Implement an element-wise `Abs` kernel in `ts/ops/`, wire it into
   `CPUEngine.ts`'s `switch`, and confirm it dispatches. (Follow `ts/ops/reLU.ts` as a template.)
6. **Find a fusion.** In `models/efficientdet_lite0_fp32/graph.json`, find a `Conv2D` whose
   `relu` param is set — that's a Conv+ReLU fusion already baked in. Explain what two ops it
   represents.

---

## 11.5 What this codebase does — and doesn't — cover

> 🌱 This book is an honest slice, not the whole of AI. It shows you how models **run, learn, and
> shrink**, using two real models. It does *not* cover the messy craft of gathering data, the math
> proofs underneath, the very largest "frontier" training recipes, or other model families
> (image-generators, and so on). The table below is the honest map of what's left.

🔬 VolvoxAI now spans the full loop this book follows: it runs forward passes (Part I), implements the
backward pass and optimizers to **train** weights (Part II), and **quantizes** trained models to int8
(Part III), across browser and native backends (Part IV). What it is *not* is a complete ML-research
platform. The honest remaining gaps:

| Missing area | What would need to be added | Why it matters |
|---|---|---|
| **Data & pipelines** | Dataset manifests, streaming/input pipelines, augmentation, cleaning, tokenizer training, and train/validation/test splits with leakage checks. | Model quality is usually bounded by data quality and experimental hygiene. |
| **Evaluation & experimentation** | Standard task metrics, baselines, ablations, hyperparameter sweeps, and bias-variance analysis, beyond the per-task exact-match the examples report. | This is how you know a model is actually better, not just different. |
| **Math foundations** | Linear-algebra derivations, calculus for the chain rule and gradients, probability, and entropy / KL / likelihood. | The tools for explaining *why* training and evaluation behave the way they do. |
| **Frontier LLM stack** | Pretraining at scale, RLHF/DPO preference training, distributed data/model parallelism, FlashAttention, and *prebuilt* grouped-query-attention / SwiGLU blocks. | The engine already has RoPE, RMSNorm, MoE, LoRA, and int8 — but not the largest-scale recipes or fused-attention kernels. (SwiGLU is composable today from `SiLU`+`Mul`+`Linear`; only the one-call block is missing.) |
| **Sub-8-bit quantization** | int4 / group-quantized weight formats and their unpack-in-register kernels. | The extra memory-bandwidth win for large LLM weights; documented as a future microkernel direction, not a shipping format. |
| **Architecture breadth** | Diffusion, graph neural networks, RNN/LSTM, reinforcement learning, VAE/GAN, retrieval/embedding, and state-space models. | The walkthrough covers a transformer LM, a CNN detector, and (Chapter 10) a multimodal VQA capstone; other domains use different inductive biases. |
| **Research practice** | Paper reproduction, controlled experiments, and scaling-law / error analysis. | The difference between running or training a model and producing reliable new knowledge. |

Listing these keeps the scope honest: a strong inference **and** training/optimization foundation,
not a complete training-and-research curriculum.

> 🔬 **Engine gaps vs. this list.** The table above is *capability-level*. For the concrete,
> near-term **engine** gaps that are already on the to-do list — missing GPU/WebNN op coverage,
> INT4 weights, a browser streaming helper, parity/benchmark harnesses — see the live
> [`docs/roadmap.md`](../roadmap.md).

---

## 11.6 Where to go from here (🔬)

The natural next steps split into two tracks:

- **Deepen this repo's inference path.** Compare `native/src/kernels/conv_f32_opt.c` to Google's **XNNPACK**;
  `docs/xnnpack_optimization_guide.md` in this repo is a guided tour. Then inspect
  `shaders/{inference,training}/*.wgsl` and the native GPU backends.
- **Scale the transformer.** GPT-2/3, LLaMA, Mistral, Qwen are Chapter 2's graph, wider/deeper,
  with tweaks: **RMSNorm** instead of LayerNorm, **RoPE** rotary positions instead of learned
  `wpe`, **grouped-query attention**, **SwiGLU** MLPs (compose them from `SiLU` + `Mul` + `Linear` —
  those ops already ship). Each is a small variation on ops you know.
- **Broaden architectures.** Classification, segmentation, pose, diffusion, retrieval,
  multimodal, MoE, and SSM systems all reuse the tensor/graph mental model, but add different
  blocks and training objectives.
- **Extend the training path.** The autograd, cross-entropy, AdamW, LoRA, and PTQ/QAT are already
  here (Parts II–III). The next concrete additions are a real dataset/input pipeline, a
  validation-and-metrics harness, and int4 weight quantization.
- **Read the source papers** once the mechanics are concrete: *Attention Is All You Need*
  (transformers), *EfficientDet* (this detector), *EfficientNet* (the backbone), and a
  quantization primer (e.g. the "gemmlowp"/TFLite integer-quantization write-ups).
- **Explore peer runtimes** from the landscape: **wonnx** (WebGPU/ONNX), **ncnn** (no-deps
  native), **ggml/llama.cpp** (portable C LLM inference).

The mental model you built here — *a model is a graph of small tensor ops with learned weights;
inference walks the graph forward, training walks it backward; performance is memory layout;
precision is a size/accuracy dial* — is the whole spine of this book, and it transfers directly to
the modules above.

---

*End of the VolvoxAI textbook. Back to the [index](README.md).*
