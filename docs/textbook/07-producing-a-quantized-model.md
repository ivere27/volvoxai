# Chapter 7 — Producing a Quantized Model

*Read the badges that match you: 🌱 **Idea** (anyone, no code) · 🔧 **Build** (a little code) · 🔬 **Deep**
(engine developers). New here? Follow just the 🌱 sections.*

*Goal: Chapter 6 explained the int8 **math** — the `scale` + `zero_point` recipe and how `QConv2D`
runs in integers. This chapter answers the question it left open: **where do those numbers come
from?** We use the full profile's PTQ authoring tools to turn a trained FP32 Graph into a shippable
int8 package, then explain when quantization-aware training (QAT) is useful.*

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
>
> *Per-channel, by hand.* Say channel A's largest-magnitude weight is `0.8` and channel B's is `0.05`.
> One shared ruler sized to `0.8` would round every one of B's tiny weights onto almost the same few
> ticks — B loses all its detail. Give B its *own* ruler (`scale = 0.05/127`) and its small values
> spread back across the full 256 ticks. One ruler per channel, and nobody is crushed by a loud
> neighbor.

🔬 Weight quantization is deterministic. `packPTQWeight()` packs a weight tensor to int8 with
**one symmetric scale per output channel**:

```javascript
const packed = packPTQWeight(values, [outputChannels, inputChannels], {
  axis: 0,
  name: 'projection.weight.i8',
});
```

Why **per-channel** and not one scale for the whole tensor? Because one unusually large channel would
stretch the ruler and crush everyone else's precision (Chapter 6 §6.2). Giving each output channel
its own `scale = max(|channel|)/127` keeps every channel's precision independent — the trick that
holds int8 accuracy near FP32. `packPTQBias()` quantizes biases into int32 using the product of the
input and weight scales. `saturationCount` reports how
many values hit the ±127 rail, so you can catch a badly-scaled tensor.

> 🔬 **Under the hood: the narrow range and the int32 bias.** Symmetric weight packing uses
> `scale = max(|channel|) / 127` — dividing by **127, not 128** — so the range stays balanced
> `[−127, 127]` and no legal weight ever maps to the lone extra code `−128`. Biases are packed to
> **int32** as `round(bias / (input_scale · weight_scale))`, i.e. into the *same units* as the int32
> accumulator from Chapter 6 §6.4, so a bias can be added straight into the accumulate with no separate
> rescale. `saturation_count` counts values that slammed into the ±127 rail; a large count is your
> signal that the ruler is mis-sized and precision is leaking.

## 7.3 Activations: calibration by observation

> 🌱 **Idea.** For the flowing numbers, you play detective: run the full-size model on a few hundred
> **typical** examples and just *watch* how big the numbers get at each spot. Those observed
> highs and lows set each ruler's width. Garbage in, garbage out — if your examples aren't typical,
> your rulers will be wrong.

🔧 For activations, run the FP32 model on a small **calibration set** and declare every calibration
tensor as a Graph output. `ExecutionResult` exposes those stable output snapshots; `PTQObserver`
tracks their copied F32 ranges:

```javascript
const observer = new PTQObserver();
for (const inputs of calibrationSamples) {
  const result = await context.execute(inputs);
  try {
    observer.observe(await result.output('encoder.out').read());
  } finally {
    await result.close();
  }
}
```

🔬 After enough samples, `derivePTQParameters(observer, options)` turns the observed
`[minimum, maximum]` into a `scale` + `zero_point`, choosing between two schemes:

- **Symmetric** — range centered on zero, `zero_point = 0`. Best for
  weights and for activations that swing both ways.
- **Asymmetric** — an arbitrary `[min,max]` mapped onto the ruler with a
  nonzero `zero_point`. Best for one-sided activations (e.g. post-ReLU, always ≥ 0), where it doesn't
  waste half the ticks on negatives that never occur.

> 🔬 **Under the hood: min/max is the simplest observer, not the only one.** Tracking the running
> `[min, max]` is easy but fragile — one freak outlier stretches the ruler and coarsens everything else.
> Production calibrators often use **percentiles**, or a **histogram + KL-divergence** ("entropy"
> calibration) to clip rare outliers and keep the bulk of the distribution sharp, or an EMA across
> batches. VolvoxAI's `PTQObserver` deliberately uses transparent min/max state and a sample count,
> so you can see exactly what set each scale.

> **Calibration data matters.** 🌱 The rulers are only as good as the examples you show — feed it
> blank inputs and you get meaningless rulers. 🔬 The `tiny_receipt` tooling makes this explicit: a
> `--structural-smoke` run with zero-filled inputs is allowed *only* to prove the plumbing, and the
> package records that it is **not** a real accuracy claim. Meaningful calibration streams actual
> receipts through the graph.

## 7.4 Post-training quantization, end to end

> 🌱 **Idea.** Put it together: measure the knobs, watch the flowing numbers on real examples, pick
> all the rulers, and write out a new — much smaller — model in the package format from
> Chapter 1. No re-training needed. You can *measure* how much accuracy you lost, so
> you decide honestly whether the small version is good enough.

🔧 The graph author chooses the boundaries; the tools never guess how an unsupported operation
should cross a precision boundary. `materializePTQWeights()` packs selected F32 weights and biases
and returns new safetensors bytes plus `artifact.quantization`, a reference-only table for the new
Graph:

```javascript
const artifact = materializePTQWeights(trainingGraph, [{
  name: 'decoder.proj.weight',
  outputName: 'decoder.proj.weight.i8',
  scaleName: 'decoder.proj.weight.scale',
  zeroPointName: 'decoder.proj.weight.zero_point',
  bias: 'decoder.proj.bias',
  biasOutputName: 'decoder.proj.bias.i32',
  inputScale: activationParameters['decoder.proj.input'].scale,
  axis: 0,
}]);
```

The output is the same **package** you met in Chapter 1 — a `graph.json` with the exact root
discriminator `"format": "volvox-graph/v1"`, whose quantized nodes are now
`QLinear`/`QConv2D` and whose sole central table refers to scale and zero-point tensors, plus a
`model.safetensors` carrying the packed int8 weights and every numeric affine parameter. No numeric
scale or zero point is stored in JSON or safetensors metadata. It loads and runs through the ordinary
`Runtime → Model → CompiledModel → ExecutionContext → ExecutionResult` lifecycle
(Chapters 8–9) with no quantization code involved at run time. This is the "train → PTQ → W8A8" path
the `tiny_receipt` example ships.

🔬 Two safety properties worth calling out, because they're baked into the API:

- **Calibration records identify one Graph revision.** Editing the graph, training another step, or
  selecting another adapter means collecting new ranges; old scales no longer describe those weights.
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

> 🔬 **Under the hood: the "straight-through" gradient, precisely.** Forward, fake-quant rounds to the
> int8 grid; backward, the straight-through estimator treats that round as the **identity** — gradient
> `1` for values *inside* the representable range and **`0`** for values that saturated past the rails
> (a "clipped" STE, so a pinned value stops shoving further out). Crucially the fake-quant lives **only
> in the training forward pass**; the shipped model is plain int8 with no dequant in the middle. And
> because the scale's gradient `dscale` also flows, the ruler width itself becomes *learnable*, not just
> observed.

🔬 The catch is the backward pass: the hard round-to-integer step has zero gradient almost everywhere,
which would kill training. The QAT Graph therefore stops gradients at a real integer cast but uses a
fake-quant boundary whose backward rule is a **straight-through estimator**: pretend the rounding was
the identity for gradient
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
- **Weights** quantize immediately — deterministic **per-channel symmetric** packing with
  `packPTQWeight()`, one scale per output channel to protect precision.
- **Activations** must be **observed**: run the FP32 model on a representative **calibration set**,
  track each tensor's range with an **observer**, and turn it into a scale (symmetric or asymmetric).
- **PTQ** ties these together in an explicit authoring flow that packs weights, calibrates
  activations, measures error vs FP32, and writes a canonical package — no retraining.
- **QAT** goes further when PTQ's accuracy drop is too big: **fake-quantize** in the training forward
  pass and let the **straight-through estimator** keep gradients
  flowing, so the weights *learn* to tolerate int8.
- VolvoxAI's quantization ceiling is a well-executed **int8**; int4 is future work, not a format.

That completes Part III: we can now *find* weights (Part II) and *shrink* them (Part III). Part IV
returns to the question the whole book started with — how does the runtime actually **run** any of
these graphs, fast, on real hardware?

**Next:** [Chapter 8 — Inside the Browser Engine →](08-inside-the-engine.md)
