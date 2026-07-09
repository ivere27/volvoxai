# Chapter 7 — Glossary & Next Steps

*Goal: one place for every term, a concrete path through this repo, and exercises that turn
reading into skill.*

---

## 7.1 Glossary

**Activation** — a tensor of intermediate values flowing between ops (as opposed to a *weight*).
Kept in fp32 by VolvoxAI (int8 on the native quantized path).

**Anchor** — a fixed reference box (a prior) that a detector adjusts, instead of predicting a box
from scratch. EfficientDet-Lite0 uses 9 per grid cell → 19,206 total.

**Attention (SDPA)** — the transformer mechanism where each token compares its **Query** to every
token's **Key** and blends their **Values** by similarity. "Which earlier words matter to me?"

**Autoregressive** — generating a sequence one token at a time, feeding each output back as input.

**Backbone** — the feature-extractor stage of a vision model (here, EfficientNet-Lite0).

**Backend** — a concrete executor for the graph's ops. Browser backends are the four *tiers*;
native backends are CPU, Vulkan, OpenGL/GLES, Metal, and NNAPI. VolvoxAI picks one per node.

**BiFPN** — Bi-directional Feature Pyramid Network; fuses features across resolutions with
resize/pool/add so every scale has both detail and meaning.

**BPE (Byte-Pair Encoding)** — the tokenizer algorithm: start from bytes, repeatedly merge the
most frequent adjacent pair, per a learned merge list.

**Broadcast** — stretching a smaller tensor to match a larger one in an element-wise op (e.g.
adding a per-channel bias `[C]` to `[N,H,W,C]`).

**Causal mask** — restricting attention so position *q* only sees positions `≤ q`; makes a
left-to-right generator.

**Channel** — one "feature plane" of a tensor (the `C` in NHWC). Input images have 3 (RGB);
hidden layers have many.

**Convolution (Conv2D)** — slide a small learned filter over an image, dot-product at each spot,
to detect a pattern everywhere. **Depthwise** = per-channel spatial filter; **Pointwise** = 1×1
channel mixer; the two together (**depthwise-separable**) are cheap and power the backbone.

**Dequantize** — convert int8 back to float: `r = (q − zero_point) × scale`.

**dlopen / dlsym** — load a shared library and look up its functions *at runtime* (not link
time). How VolvoxAI's native binary uses a GPU driver (`libvulkan`, `libGL`) without linking any
GPU SDK — the "no static GPU dependency" design.

**Embedding** — a learned vector that represents a discrete token; the `Embedding` op is a table
lookup.

**Forward pass / Inference** — running the graph once, input → output. VolvoxAI does only this
(no training).

**Fusion** — merging adjacent ops (e.g. Conv+ReLU) so intermediate data is written once.

**GELU / ReLU / ReLU6 / Sigmoid** — nonlinearities; the "decision curves" between linear layers.
Without them, stacked MatMuls collapse into one.

**GEMM** — GEneral Matrix Multiply; the heavily-optimized routine most fast conv/matmul paths
reduce to.

**Graph** — the model's op list: nodes (ops) connected by named tensors. Stored as `config.json`.

**Head** — the final task-specific layer(s): the LM head (→ vocabulary logits) or the detector's
class/box heads.

**im2col** — "image to columns"; unfold conv input patches into a matrix so a conv becomes a GEMM.

**int8 / fp16 / fp32** — 8-bit integer / 16-bit float / 32-bit float number formats (1 / 2 / 4
bytes). See Chapter 4.

**KV-cache** — caching past tokens' Keys and Values so each generation step only computes the new
token's attention. In this repo: `engine_prefill` + `engine_decode`.

**LayerNorm / RMSNorm** — normalize a vector (mean 0, variance 1, then learned scale/shift) to
keep deep-network numbers stable.

**Logits** — raw, un-normalized scores (pre-softmax/sigmoid). The LM head and class head emit
these.

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
(`native/nnapi_engine.c`, `make build_android`).

**Node** — one entry in the graph: an op plus its input/output tensor names and parameters.

**Op / Operation / Kernel** — a single math routine (Add, Conv2D, SDPA…). "Op" is the graph-level
name; "kernel" is a specific implementation of it.

**Quantization** — representing weights/activations with fewer bits via `scale` + `zero_point`.

**Residual (skip connection)** — adding a block's input to its output (`out = x + f(x)`) so
information and gradients survive deep stacks. In both models.

**Safetensors** — the standard binary file format for the weights.

**Scale / Zero-point** — the two numbers of a quantization recipe: tick size, and which integer
means real 0.

**Softmax** — turns a vector of scores into a probability distribution (positive, sums to 1).

**SPIR-V** — the binary shader format Vulkan consumes; `naga` compiles VolvoxAI's WGSL to it.

**Prefill / Decode** — the two phases of native text generation: *prefill* runs the prompt once
to fill the KV-cache; *decode* runs one new token at a time using the cache. See `engine_prefill`
/ `engine_decode`.

**Tensor** — a multi-dimensional array of numbers with a shape; the only data type in the engine.

**Tier** — one of VolvoxAI's four browser backends (WebNN / WebGPU / WASM / Pure-JS), picked by
capability.

**Token** — a chunk of text (word/sub-word/byte) mapped to an integer id.

**Weight** — a number learned during training, read-only at inference.

---

## 7.2 A path through this repository

Read in this order to go from "I get the concepts" to "I can modify the engine":

1. **The data model** — `js/Tensor.js` (25 lines), `js/Graph.js` (48 lines). Tiny; read fully.
2. **The executor** — `js/CPUEngine.js`. See the `for (node of graph.nodes)` loop and the
   `switch` dispatch. This is the whole runtime.
3. **Four naive kernels** — `js/ops/add.js`, `embedding.js`, `layerNorm.js`, `matMul.js`. Each is
   a few dozen readable lines.
4. **The two models' blueprints** — skim `models/tinystories_1m/config.json` and
   `models/efficientdet_lite0_fp32/config.json`. Match nodes to Chapters 2–3.
5. **The attention + conv kernels** — `js/ops/sDPA.js`, `js/ops/conv2D.js`. The two "hearts."
6. **Quantization** — `js/ops/dequantizeLinear.js`, then `native/quant_cpu_opt.c` for the real
   int8 conv.
7. **Optimization** — diff `js/ops/conv2D.js` against `native/conv_f32_opt.c` while reading
   `docs/microkernel_optimization_guide.md` and `docs/xnnpack_optimization_guide.md`.
8. **The GPU tier** — `shaders/*.wgsl` (e.g. `matmul`), and `js/GraphExecutor.js`.
9. **The native engine** (Chapter 6) — `native/engine.h` + `native/engine.c` (lifecycle),
   `native/engine_runtime.c` (`run_node` backend selection), then a device backend such as
   `native/vulkan_engine.c` (see the `dlopen` at the top). `native/main.c` holds the task CLIs.

`docs/operation_list.md` is the per-op × per-backend support matrix — your reference map.

---

## 7.3 Run the models yourself

```bash
# Language model — generate text (greedy). Builds native binary first: `make build_native`.
./native/volvoxai generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" --max-new 50 [--debug]

# Raw graph runner — dump the logits tensor for a fixed set of tokens.
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 --last-token 4

# Object detector — decode an image into ranked boxes.
./native/volvoxai detect models/efficientdet_lite0_int8 \
  --image input0=photo.png --image-normalize raw-255 \
  --boxes boxes --scores scores --max-det 20

# In Node (WASM / pure-JS tiers), smoke-test any blueprint:
node bin/volvox.js run --model models/tinystories_1m/model.safetensors --backend wasm
```

Add `--debug` to `generate` to see per-node timing and tokens/sec — a great way to *feel* where
time goes (and to watch the optimizations from Chapter 5 pay off).

---

## 7.4 Exercises (reading → skill)

1. **Trace by hand.** Take the sequence `[5, 5]` (two identical tokens) and a made-up 2-dim
   embedding. Walk `Embedding → Add(position) → LayerNorm` with pen and paper. Confirm the shapes
   match `config.json`.
2. **Break causality.** In `js/ops/sDPA.js`, change `k <= q` to `k < seq_len`. Predict what
   happens to generated text and why. (Then revert.)
3. **Quantize a weight.** Pick `scale = 0.02`, `zero_point = -5`. Quantize `r = 0.31`, then
   dequantize it back. Report the round-trip error. Now try `scale = 0.002`. What did precision
   cost you in range?
4. **Count the FLOPs.** For the first `Conv2D` (stem: 320×320×3 → 160×160×32, 3×3 filter),
   estimate the multiply-adds. Compare to a 1×1 pointwise conv of the same output size. Why is
   depthwise-separable cheaper?
5. **Add an op.** Implement an element-wise `Abs` kernel in `js/ops/`, wire it into
   `CPUEngine.js`'s `switch`, and confirm it dispatches. (Follow `js/ops/reLU.js` as a template.)
6. **Find a fusion.** In `models/efficientdet_lite0_fp32/config.json`, find a `Conv2D` whose
   `relu` param is set — that's a Conv+ReLU fusion already baked in. Explain what two ops it
   represents.

---

## 7.5 Current gaps in this codebase

VolvoxAI is an inference engine, and this textbook follows that boundary. The codebase can load
trained weights and run forward passes; it does not yet include the systems needed to create,
tune, or scientifically validate models.

| Missing area | What would need to be added | Why it matters |
|---|---|---|
| **Training** | Reverse-mode autodiff, backward kernels, loss functions, optimizers such as Adam/SGD, learning-rate schedules, initialization, checkpointing, and regularization. | This is how weights are discovered instead of only consumed. |
| **Math foundations** | Linear algebra derivations, calculus for chain rule and gradients, probability, entropy/cross-entropy, KL divergence, and likelihood. | These are the tools for explaining why training and evaluation behave the way they do. |
| **Data** | Dataset manifests, streaming/input pipelines, augmentation, cleaning, tokenizer training, train/validation/test splits, and leakage checks. | Model quality is usually bounded by data quality and experimental hygiene. |
| **Evaluation & experimentation** | Task metrics, validation loops, baselines, ablations, hyperparameter sweeps, overfitting checks, and bias-variance analysis. | This is how you know a model is actually better, not just different. |
| **Architecture breadth** | Diffusion models, graph neural networks, RNN/LSTM, reinforcement learning, VAE/GAN, retrieval and embedding systems, multimodal models, mixture-of-experts, and state-space models. | The current walkthrough covers one transformer LM and one CNN detector; many domains use different inductive biases. |
| **Modern LLM training stack** | Pretraining loops, supervised fine-tuning, LoRA/adapters, RLHF/DPO preference training, distributed data/model parallelism, and FlashAttention internals. | Most frontier LLM work happens around training recipes, memory-efficient attention, and large-scale systems. |
| **Research practice** | Paper reproduction, result derivation, controlled experiments, scaling-law analysis, error analysis, and documentation of assumptions. | This is the difference between running a model and producing reliable new knowledge. |

These are future textbook/code modules, not prerequisites for the inference chapters. Listing
them makes the scope explicit: this repo is a strong inference/runtime foundation, not a complete
training-and-research curriculum.

---

## 7.6 Where to go from here

The natural next steps split into two tracks:

- **Deepen this repo's inference path.** Compare `native/conv_f32_opt.c` to Google's **XNNPACK**;
  `docs/xnnpack_optimization_guide.md` in this repo is a guided tour. Then inspect
  `shaders/*.wgsl` and the native GPU backends.
- **Scale the transformer.** GPT-2/3, LLaMA, Mistral, Qwen are Chapter 2's graph, wider/deeper,
  with tweaks: **RMSNorm** instead of LayerNorm, **RoPE** rotary positions instead of learned
  `wpe`, **grouped-query attention**, **SwiGLU** MLPs. Each is a small variation on ops you know.
- **Broaden architectures.** Classification, segmentation, pose, diffusion, retrieval,
  multimodal, MoE, and SSM systems all reuse the tensor/graph mental model, but add different
  blocks and training objectives.
- **Close the training gap deliberately.** A small autograd engine, a cross-entropy loss, Adam,
  a tiny dataset loader, and a validation loop would be the first concrete additions.
- **Read the source papers** once the mechanics are concrete: *Attention Is All You Need*
  (transformers), *EfficientDet* (this detector), *EfficientNet* (the backbone), and a
  quantization primer (e.g. the "gemmlowp"/TFLite integer-quantization write-ups).
- **Explore peer runtimes** from the landscape: **wonnx** (WebGPU/ONNX), **ncnn** (no-deps
  native), **ggml/llama.cpp** (portable C LLM inference).

The mental model you built here — *a model is a graph of small tensor ops with learned weights;
inference walks the graph; performance is memory layout; precision is a size/accuracy dial* —
transfers to those future modules, but it is one part of the full stack.

---

*End of the VolvoxAI textbook. Back to the [index](README.md).*
