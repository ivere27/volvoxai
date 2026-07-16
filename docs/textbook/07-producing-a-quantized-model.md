# Chapter 7 — Producing a Quantized Model

*Read the badges that match you: 🌱 **Idea** (anyone, no code) · 🔧 **Build** (a little code) · 🔬 **Deep**
(engine developers). New here? Follow just the 🌱 sections.*

*Goal: Chapter 6 explained the int8 **math** — the `scale` + `zero_point` recipe and how `QConv2D`
runs in integers. This chapter answers the question it left open: **where do those numbers come
from?** We follow VolvoxAI's quantization subsystem (`volvoxai_ptq_*` in `volvoxai_training.h`,
`native/src/training/quantization*.c`) as it turns a trained FP32 model into a shippable int8 one —
first with post-training quantization (PTQ), then with quantization-aware training (QAT).*

> 🌱 **The big idea.** Rounding a model onto the coarse ruler from Chapter 6 sounds automatic, but
> there's a catch: **how wide should the ruler be?** For the model's fixed knobs (weights) that's
> easy — just look at their biggest and smallest value. But the numbers that *flow through* the model
> while it runs depend on the input (a dark photo vs a bright one), so you can't know their range
> until you actually **watch the model run on some real examples.** That watching step is called
> **calibration**, and it's the heart of making a good small model. If the ruler is set well, the
> shrunk model is nearly as accurate as the big one; if it's set badly, the model turns to mush.

🔧 An int8 model is only as good as its scales. Pick them well and accuracy stays within ~1% of FP32;
pick them badly and the model is garbage. Producing those scales is a real, data-driven *process* —
not a format conversion. There are two families:

```
   trained FP32 model
        │
        ├── PTQ  (post-training quantization) ── observe ranges on a little data, pick scales, pack
        │                                          → fast, no retraining, ~1% accuracy cost
        └── QAT  (quantization-aware training) ── train WITH fake int8 rounding in the loop
                                                   → slower, recovers accuracy when PTQ isn't enough
```

## 7.1 Two kinds of number: weights you know, activations you don't

> 🌱 **Idea.** Two kinds of number live in a running model. The **weights** are the fixed knobs — they
> never change once training is done, so their range is easy: just read off the biggest and smallest.
> The **activations** are the numbers that flow between steps *while it runs*, and those depend
> entirely on the input. You simply can't know their range without running the model on some real
> examples first. That one difference is why shrinking a model takes a *process*, not a button.

🔧 Recall the int8 recipe needs a range to stretch the `−128…127` ruler across. For the two kinds of
tensor, that range is found very differently:

- **Weights** are *fixed* after training. Their range is just their min and max — you can read it
  straight off the tensor, no data required.
- **Activations** (the values flowing *between* ops) depend on the **input**. A receipt photo of a
  dark ceiling and one of a bright receipt produce wildly different activation magnitudes. You cannot
  know an activation's range without actually running the model on representative data.

That single asymmetry is the whole reason quantization needs a *process*: weights quantize
immediately; activations must be **observed**.

## 7.2 Weights: per-channel packing (no data needed)

> 🌱 **Idea.** The knobs are easy: each group of them gets its *own* ruler sized to its own biggest
> value, so no group loses precision to a neighbor's outlier. No data needed — it's pure arithmetic.

🔬 Weight quantization is deterministic. VolvoxAI packs a weight tensor to int8 with **one symmetric
scale per output channel** (`volvoxai_ptq_pack_weight_i8`):

```c
// pack row-major weights, one I8 scale per `axis` element (axis 0 = output channel)
int volvoxai_ptq_pack_weight_i8(const float *values, const int32_t *shape,
                                int32_t ndim, int32_t axis, int8_t *output,
                                float *scales, int32_t scale_count,
                                uint64_t *saturation_count);
```

Why **per-channel** and not one scale for the whole tensor? Because one unusually large channel would
stretch the ruler and crush everyone else's precision (Chapter 6 §6.2). Giving each output channel
its own `scale = max(|channel|)/127` keeps every channel's precision independent — the trick that
holds int8 accuracy near FP32. Biases get their own companion step (`volvoxai_ptq_pack_bias_i32`),
quantized into int32 using the product of the input and weight scales. `saturation_count` reports how
many values hit the ±127 rail, so you can catch a badly-scaled tensor.

## 7.3 Activations: calibration by observation

> 🌱 **Idea.** For the flowing numbers, you play detective: run the full-size model on a few hundred
> **typical** examples and just *watch* how big the numbers get at each spot. Those observed
> highs and lows set each ruler's width. Garbage in, garbage out — if your examples aren't typical,
> your rulers will be wrong.

🔧 For activations, VolvoxAI runs the FP32 model on a small **calibration set** (a few hundred
representative inputs) and *watches* each tensor go by. An **observer** just tracks the running range
(`volvoxai_training.h`):

```c
typedef struct volvoxai_ptq_observer {
    float minimum;
    float maximum;
    uint64_t sample_count;
} volvoxai_ptq_observer_t;

// after each forward pass, fold a named F32 tensor's values into its observer:
int volvoxai_engine_ptq_observe_tensor(const char *tensor_name,
                                       volvoxai_ptq_observer_t *observer);
```

🔬 After enough samples, the observed `[minimum, maximum]` becomes a `scale` + `zero_point`
(`volvoxai_ptq_calculate_params`), choosing between two schemes:

- **Symmetric** (`VOLVOXAI_PTQ_SYMMETRIC`) — range centered on zero, `zero_point = 0`. Best for
  weights and for activations that swing both ways.
- **Asymmetric** (`VOLVOXAI_PTQ_ASYMMETRIC`) — an arbitrary `[min,max]` mapped onto the ruler with a
  nonzero `zero_point`. Best for one-sided activations (e.g. post-ReLU, always ≥ 0), where it doesn't
  waste half the ticks on negatives that never occur.

> **Calibration data matters.** 🌱 The rulers are only as good as the examples you show — feed it
> blank inputs and you get meaningless rulers. 🔬 The `tiny_receipt` tooling makes this explicit: a
> `--structural-smoke` run with zero-filled inputs is allowed *only* to prove the plumbing, and the
> package records that it is **not** a real accuracy claim. Meaningful calibration streams actual
> receipts through the graph.

## 7.4 Post-training quantization, end to end

> 🌱 **Idea.** Put it together: measure the knobs, watch the flowing numbers on real examples, pick
> all the rulers, and write out a new — much smaller — model in the *same two-file format* from
> Chapter 1. No re-training needed. The engine even lets you *measure* how much accuracy you lost, so
> you decide honestly whether the small version is good enough.

🔧 VolvoxAI ties weight packing and activation calibration together in an explicit **PTQ plan**. The
plan observes a loaded FP32 graph but never rewrites it — *you* author which nodes become quantized
(the engine stays explicit; it never guesses how to cross a precision boundary):

```c
VolvoxAIPTQPlan *plan = volvoxai_ptq_plan_create();      // bound to the loaded FP32 model
volvoxai_ptq_plan_add_tensor(plan, &tensor_spec);        // an activation to calibrate (dtype, scheme)
volvoxai_ptq_plan_add_layer(plan, &layer_spec);          // a node → QLinear / QConv2D, its weights
// ... stream calibration inputs:
volvoxai_engine_ptq_plan_calibrate_sample(plan, "sample-0", bindings, n);
// ... inspect what it found, then emit the package:
volvoxai_ptq_plan_tensor_params(plan, "vqa.enc.0.out", &params);   // scale, zero_point, saturation
volvoxai_ptq_plan_write_package(plan, &options);         // two files: config + safetensors
```

The output is the same **two-file blueprint** you met in Chapter 1 — a `config.json` whose quantized
nodes are now `QLinear`/`QConv2D` with `weight_scale` descriptors, plus a `model.safetensors`
carrying the packed int8 weights and their scales. It loads and runs on the ordinary inference engine
(Chapters 8–9) with no quantization code involved at run time. This is the "train → PTQ → W8A8" path
the `tiny_receipt` example ships.

🔬 Two safety properties worth calling out, because they're baked into the API:

- **A plan is pinned to one model generation.** Reloading, editing the graph, training a step, or
  activating an adapter *invalidates* the plan — its scales would no longer describe the live weights.
  Stale operations fail rather than emit a corrupt package.
- **Error vs FP32 is measurable.** Because the FP32 graph is right there, you can compare the
  quantized output against it and get a concrete accuracy delta per tensor — the honest way to decide
  whether int8 is acceptable for a given layer.

## 7.5 When PTQ isn't enough: quantization-aware training

> 🌱 **Idea.** Sometimes plain rounding hurts too much — the small model gets noticeably dumber. The
> fix: let the model **practice while wearing the rounding.** During training you deliberately round
> the numbers the way the shrunk model will, so the model *feels* the roughness and learns to be
> robust to it. It's like training with ankle weights so the real thing feels easy. Slower (it's a
> full training run), but it recovers the lost accuracy.

🔧 PTQ takes a finished model and hopes it tolerates int8. Usually it does. When it doesn't — when the
~1% drop becomes a painful 5% — you **train the model to expect quantization**.

The trick is **fake quantization**: during the *training* forward pass, insert a
`QuantizeLinear → DequantizeLinear` pair so each value is rounded to its int8 grid and back to float.
The numbers the network sees are now the *actual* int8-rounded values, so the loss reflects the real
deployed error, and Chapters 4–5 do the rest — the weights learn to be robust to rounding.

🔬 The catch is the backward pass: the hard round-to-integer step has zero gradient almost everywhere,
which would kill training. VolvoxAI resolves this exactly where Chapter 4 §4.6 hinted:

```c
// the integer cast STOPS the gradient (round() has no useful slope) ...
volvoxai_training_cast_backward_f32(...);              // → dx stopped for integer casts
// ... but DequantizeLinear passes gradient straight THROUGH the fake-quant:
volvoxai_training_dequantize_linear_backward_f32(
    input, input_type, scale, zero_point, zero_point_type,
    dy, dx, dscale, elements);                          // dx (and even dscale) flow
```

That is the **straight-through estimator**: pretend the rounding was the identity for gradient
purposes, so learning continues while the forward pass still feels the real int8 grid. Because
`dscale` can also flow, the quantization scales themselves can be *learned* rather than only observed.
QAT costs a full training run, so it's the tool you reach for only after PTQ's measured error
(§7.4) says you need it.

## 7.6 How far down: int8, per-channel, and stopping there

> 🌱 **Idea.** How much should you shrink? VolvoxAI's answer: **int8 is the sweet spot** — big size
> win, tiny accuracy cost, runs on anything. Some delicate parts of the model are left un-shrunk on
> purpose (you don't round the pieces that can't take it). And the project deliberately *stops* at
> 8 bits rather than chasing ever-smaller formats that hurt cheap hardware.

🔬 Chapter 6's trade-off curve still governs *which* precision to use where. In practice:

- **Weights → int8, per-channel symmetric** is the default; it's where almost all the size win is
  (4× smaller) at almost no accuracy cost.
- **Activations → int8** (making it "W8A8") unlocks the real integer-math *speedup* of Chapter 6, but
  it's the part that needs calibration and is most likely to need QAT.
- **Sensitive layers stay wider.** Not every node should be int8 — that's why the plan is
  *node-by-node* and caller-authored. First/last layers and numerically delicate ops (norms, some
  attention paths) are commonly left in FP32; the `tiny_receipt` path keeps FP32 norms and biases.

And VolvoxAI deliberately **stops at 8 bits**. Sub-8-bit formats (int4 and friends) buy more memory
bandwidth but need fiddly bit-unpacking and hurt low-end hardware; the repo treats int4 as a
documented *future* microkernel direction, not a shipping format. So "produce a quantized model"
here means **int8**, done well, end to end.

## 7.7 What you just learned

> 🌱 **Idea recap.** Shrinking a model well isn't a button — it's a process. Measure the fixed knobs,
> **watch** the flowing numbers on real examples to size their rulers (**calibration**), and write out
> a small two-file model. If plain rounding hurts too much, let the model **practice with the rounding
> on** (QAT) until it toughens up. Done right, the tiny model matches the big one — and now it fits on
> a phone or robot.

🔧

- Quantization is a **process**, not a cast: an int8 model is only as good as its `scale`/`zero_point`
  numbers, and producing them is data-driven.
- **Weights** quantize immediately — deterministic **per-channel symmetric** packing
  (`pack_weight_i8`), one scale per output channel to protect precision.
- **Activations** must be **observed**: run the FP32 model on a representative **calibration set**,
  track each tensor's range with an **observer**, and turn it into a scale (symmetric or asymmetric).
- **PTQ** ties these together in an explicit, model-pinned **plan** that packs weights, calibrates
  activations, measures error vs FP32, and writes the ordinary two-file blueprint — no retraining.
- **QAT** goes further when PTQ's accuracy drop is too big: **fake-quantize** in the training forward
  pass and let the **straight-through estimator** (`dequantize_linear_backward`) keep gradients
  flowing, so the weights *learn* to tolerate int8.
- VolvoxAI's quantization ceiling is a well-executed **int8**; int4 is future work, not a format.

That completes Part III: we can now *find* weights (Part II) and *shrink* them (Part III). Part IV
returns to the question the whole book started with — how does the engine actually **run** any of
these graphs, fast, on real hardware?

**Next:** [Chapter 8 — Inside the Browser Engine →](08-inside-the-engine.md)
