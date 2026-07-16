# Chapter 5 — The Optimizer & the Training Loop

*Read the badges that match you: 🌱 **Idea** (anyone, no code) · 🔧 **Build** (a little code) · 🔬 **Deep**
(engine developers). New here? Follow just the 🌱 sections.*

*Goal: turn the gradients from Chapter 4 into a model that actually gets better. We meet the two
optimizers VolvoxAI ships (SGD and AdamW), assemble one full training step, then wrap it in the loop
— batches, gradient accumulation, train-vs-eval mode, checkpoints, and **LoRA** fine-tuning — using
the real `tiny_receipt_vqa_train` example as our worked case.*

> 🌱 **The big idea.** Last chapter every knob got a note: *"turn me this way to be less wrong."* This
> chapter actually **turns the knobs** — a little, in the right direction — and then does it again,
> and again, over tons of examples. That "take a small step downhill" tool is the **optimizer**.
> Doing it over and over is the **training loop**. We'll also meet a clever shortcut called **LoRA**:
> instead of re-writing the whole machine, you stick a few tiny "sticky notes" on it and only train
> those — so you can teach a giant model a new trick cheaply.

🔧 Chapter 4 left every weight holding a **gradient**: the direction that would increase the loss. To
*decrease* the loss we step each weight a little the other way. Doing that once is the **optimizer**;
doing it over and over across data is the **training loop**. Both live behind
`volvoxai_engine_train_step(...)` and the kernels in `native/src/kernels/training_kernels.c`.

## 5.1 The simplest optimizer: SGD

> 🌱 **Idea.** The simplest rule: the note says "downhill is that way," so take one small step that
> way. Small steps = slow but safe. Big steps = fast but you might overshoot the valley and tumble
> out the other side. How big a step you take is called the **learning rate**, and it's the single
> most important dial in all of training.

🔧 If `grad` points uphill, walk downhill — a small step opposite the gradient. That is **Stochastic
Gradient Descent**, and its update is one line (`volvoxai_training_sgd_update_f32`):

```c
// for every weight element i:
weights[i] -= learning_rate * (gradient[i] + weight_decay * weights[i]);
```

- **`learning_rate`** — how big a step. Too small: training crawls. Too big: it overshoots and the
  loss diverges (this is the single most important knob).
- **`weight_decay`** — a gentle pull toward zero that discourages huge weights (regularization).

```
   loss                    each step: w ← w − lr · slope
    \                         big lr  ──►  ● ─────► ● overshoots ✗
     \___                     small lr ─►  ●─►●─►●  steady ✓
         \__●__●__●___
                       w*     (the minimum we're rolling toward)
```

SGD works, but it uses the same blunt step size for every weight and every direction. Most real
training uses something smarter.

## 5.2 The workhorse: AdamW

> 🌱 **Idea.** Plain SGD takes the same size step for every knob. **AdamW** is smarter: it gives each
> knob its *own* step size and a bit of **momentum** — like a ball rolling downhill that keeps its
> heading and doesn't get thrown off by every little bump. Knobs whose notes keep flip-flopping take
> small careful steps; knobs with a steady direction roll faster. It's the default choice for real
> training. The catch: it has to remember two extra numbers per knob, so training needs more memory
> than just running the model.

🔧 **AdamW** gives each weight its *own* adaptive step size by remembering two running averages
(called **moments**) of that weight's gradient history:

```c
uint32_t volvoxai_training_adamw_update_f32(
    float *weights, const float *gradient,
    float *first_moment, float *second_moment, uint32_t n,   // ← persistent per-weight state
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, uint32_t step);
```

- **`first_moment`** (decayed by `beta1`, ~0.9) — a smoothed *average* gradient: momentum. It keeps
  moving in a consistent direction and rides out noisy single-batch gradients.
- **`second_moment`** (decayed by `beta2`, ~0.999) — a smoothed average of the gradient *squared*:
  the typical *magnitude*. Dividing the step by its square root means weights with big, jumpy
  gradients take small careful steps, while weights with tiny gradients take relatively larger ones.
- **`step`** — the update count, used to bias-correct the two moments while they warm up.
- **`weight_decay`** — the "W" in AdamW: decoupled decay applied directly to the weight, not folded
  into the gradient (the detail that makes it behave better than plain Adam).

🔬 The cost is memory: Adam keeps **two extra full-size buffers per trainable weight**. That is why
an optimizer state file is ~2× the model, and why memory planning (Chapter 9) matters for training.
The `first_moment`/`second_moment` buffers are exactly what a **checkpoint** must save to resume
cleanly (§5.7).

🔬 VolvoxAI selects the rule per update through a mode enum (`volvoxai_training.h`):

```c
VOLVOXAI_TENSOR_UPDATE_SGD    // the §5.1 step
VOLVOXAI_TENSOR_UPDATE_ADAMW  // the §5.2 step   (the default for real training)
VOLVOXAI_TENSOR_UPDATE_ADD / _ASSIGN   // raw tensor edits, not gradient steps
```

## 5.3 One training step, end to end

> 🌱 **Idea.** One "step" strings together everything so far: run the machine, measure the wrongness,
> pass the blame backward, and nudge the knobs. One button does all four. It also lets you say
> *which* knobs are even allowed to move — handy for the sticky-note trick later.

🔧 A single step chains everything from Chapters 4–5. The engine exposes it as one call
(`volvoxai_engine_train_step`), which internally does forward → loss → backward → update:

```c
volvoxai_engine_train_step(
    "logits",              // where the forward pass wrote its scores
    targets, target_count, // the correct next-token ids for this batch
    ignore_index,          // a target value to skip (e.g. padding)
    trainable_names, trainable_count,   // WHICH tensors are allowed to change
    VOLVOXAI_TENSOR_UPDATE_ADAMW,
    learning_rate, beta1, beta2, epsilon, weight_decay,
    max_grad_norm,         // gradient clipping (see §5.4)
    step,
    &out_loss, &out_correct, &out_examples);   // metrics to log
```

Two design choices worth noting:

- **`trainable_names` is an explicit allow-list.** Only the tensors you name receive gradients and
  updates; everything else is frozen. This is the exact mechanism LoRA and adapter fine-tuning use
  (§5.8) — freeze the base model, list only the adapters.
- **The loss target lives in the call, not the graph.** The *same* forward graph you run for
  inference is reused for training; the cross-entropy loss is attached to its `logits` output at
  train time. There is no separate "training model."

## 5.4 Keeping it stable: gradient clipping

> 🌱 **Idea.** Once in a while a weird example produces a giant "turn the knobs HARD" note that would
> wreck everything. **Gradient clipping** is a seatbelt: if a step is about to be too big, shrink it
> back to a safe size. Cheap insurance against blowing up a long training run.

🔧 A single bad batch can produce a huge gradient that blows the weights apart. **Gradient clipping**
caps the overall gradient size before the step: if the global gradient norm exceeds `max_grad_norm`,
every gradient is scaled down to fit. Combined with the `volvoxai_training_all_finite_f32` NaN/Inf
check from Chapter 4, this is what keeps long training runs from diverging. The `tiny_receipt`
trainer clips every step; it is cheap insurance.

## 5.5 Batches and gradient accumulation

> 🌱 **Idea.** Judging from a *single* example is jumpy and unreliable, so we average the notes over a
> **batch** of examples before stepping — a steadier direction. But big batches need lots of memory.
> If the batch you want won't fit, **gradient accumulation** fakes it: process a few small groups,
> add up their notes, *then* take one step. Same effect, less memory.

🔧 We don't update on one example at a time — the gradient would be too noisy. We average the gradient
over a **batch** of examples and step once. Bigger batches give a smoother, more reliable direction,
but they also hold every example's activations in memory at once.

When the batch you *want* doesn't fit in memory, **gradient accumulation** fakes it: run several
small (even size-1) batches, **add** their gradients into the same buffer, and only *then* take one
optimizer step. VolvoxAI does this natively through `volvoxai_engine_train_step_multi`:

```
   accumulation_steps = 24, batch_size = 1   → effective batch of 24
   ┌────────── one optimizer update ──────────┐
   micro 1  micro 2  …  micro 24   flush → AdamW step
   (grads += )        (grads += )   (moments update, grads zeroed)
```

🔬 This is exactly the `tiny_receipt` default: `--batch-size 1 --accumulation-steps 24` reproduces an
effective batch of 24 "without retaining 24 copies of every activation." The backward pass's `+=`
accumulation from Chapter 4 §4.4 is the same machinery — here reused *across* microbatches.

## 5.6 Two modes: training vs evaluation

> 🌱 **Idea.** A model behaves a little differently while *studying* versus while being *tested*.
> During study it deliberately handicaps itself a bit (**dropout**) so it can't lean on any one trick
> and is forced to really learn. At test time the handicap is off. And you always test on questions
> it *didn't* study — otherwise it could just memorize the answers. Mixing these up is a classic
> beginner mistake.

🔧 Some ops behave differently while learning:

- **Dropout** randomly zeros a fraction of activations *during training* (so the model can't lean on
  any single unit), then is **off** at inference. VolvoxAI's `volvoxai_training_dropout_f32` is a
  deterministic *inverted* dropout — it scales the survivors up during training so the eval-time
  pass needs no correction.
- **Normalization** likewise has train/eval distinctions to respect.

So a training run alternates **two modes**: *train* mode (dropout on, weights updating) over the
training data, and *eval* mode (dropout off, no updates) over held-out validation data to measure
real progress. Mixing them up — evaluating with dropout on, or updating on the validation set — is a
classic bug.

## 5.7 The loop

> 🌱 **Idea.** Now wrap it all in a loop: go through the study material many times (each full pass is
> an **epoch**), taking small steps; every so often test on held-out questions to see if it's *really*
> improving; and **save your progress** so a crash doesn't cost you days of practice. Slow down the
> step size as you near the end, so the model settles gently into a good answer instead of bouncing.

🔧 A real run (again, `tiny_receipt_vqa_train`) looks like:

```
for each epoch:
    for each batch of the TRAINING data:        # train mode
        loss = train_step(..., step)            # forward→loss→backward→AdamW
        learning_rate = cosine_schedule(step)   # anneal LR down over the run
    for each batch of the VALIDATION data:      # eval mode, no updates
        measure exact-match accuracy
    if validation improved:  save "best.checkpoint"
    always:                  save "last.checkpoint"
```

🔬 Details that make it a *real* loop rather than a toy:

- **Learning-rate schedule.** The LR isn't constant — a **cosine schedule** warms up then decays it
  smoothly toward zero, which trains faster and lands softer. In VolvoxAI this is *caller policy*
  (the example computes it and passes the new `learning_rate` each step); the engine just applies it.
- **Evaluation metric.** Loss guides the optimizer, but humans want a task metric — `tiny_receipt`
  reports **exact-match** accuracy per task/field, and even a learned `router_accuracy`.
- **Checkpoints.** `volvoxai_engine_save_optimizer_state` / `load_optimizer_state` persist not just
  the weights but the **AdamW moments, the step counter, and the cosine horizon** — everything needed
  so `--resume last.checkpoint` continues *bit-for-bit* rather than restarting the optimizer cold.
  The example keeps both a rolling `last` and the best-so-far `best`.

## 5.8 LoRA: fine-tuning without touching the base model

> 🌱 **Idea.** Fully re-training a giant model is slow and makes a whole new giant copy. **LoRA** is
> the sticky-note trick: leave the original machine *frozen* exactly as it is, and stick a couple of
> tiny adjustable notes next to each big part. Only the notes learn. The result is a tiny file you
> can share, and one base machine can wear different "sticky-note packs" to do different jobs. This is
> how one model can be taught eight new tasks without eight full copies.

🔧 Full training updates every weight — expensive, and it produces a whole new multi-gigabyte model.
**LoRA** (Low-Rank Adaptation) is the modern shortcut: **freeze the pretrained weights** and inject a
tiny pair of low-rank matrices (rank 8 in `tiny_receipt`) beside each big Linear. Only those adapters
train.

```
   frozen base:  y = x · W          (W stays exactly as pretrained)
   + LoRA:       y = x · W  +  (x · A) · B      A:[in,8]  B:[8,out]   ← only A,B learn
```

🔬 Because rank 8 is minuscule next to a full weight matrix, the trainable set shrinks by orders of
magnitude — which is precisely what the `trainable_names` allow-list (§5.3) expresses. VolvoxAI gives
LoRA its own step entry, `volvoxai_engine_lora_train_step`, and a matching artifact layout:

- **`lora.safetensors`** — just the trained deltas, tiny and shareable.
- **`lora_base/`** — the untouched base package (zero inline LoRA tensors).
- Load the base + activate the adapter (`volvox.adapter.v1`) and you get the fine-tuned model back;
  the example even *verifies* that the merged-inline and base+adapter paths produce identical logits
  before it trusts the pair.

This is why the capstone (Chapter 10) can adapt one shared multimodal model to eight receipt tasks
without eight full copies — eight small adapters over one frozen backbone.

## 5.9 What you just learned

> 🌱 **Idea recap.** Learning = turn the knobs a little downhill (the **optimizer**), over and over,
> across many examples (the **loop**), with seatbelts (clipping), steadier averages (batches),
> honest testing (train vs eval), and saved progress (checkpoints). And when you just want to teach an
> existing model a new trick cheaply, use **LoRA** sticky notes instead of rebuilding it.

🔧

- The **optimizer** turns each weight's gradient into an actual step. **SGD** is `w -= lr·grad`;
  **AdamW** gives every weight an adaptive step from two running gradient **moments** (at the cost of
  two extra buffers per weight — the bulk of a training checkpoint).
- One **training step** is forward → loss → backward → update, exposed as `train_step`, with an
  explicit **`trainable_names`** allow-list deciding what may change.
- The **loop** adds the parts that make it work in practice: **batches** and **gradient
  accumulation** (effective batch 24 from physical batch 1), **gradient clipping** for stability,
  **train-vs-eval mode** (dropout on/off), a **cosine LR schedule**, a **task metric**, and
  **checkpoints** that save the optimizer state so a run can resume exactly.
- **LoRA** fine-tunes by freezing the base and training tiny rank-8 adapters — the same allow-list
  mechanism, a fraction of the cost, and a shareable `lora.safetensors`.

You can now *find* good weights. Part III asks the next question: once a model is trained, how do we
make it small and fast enough to ship? That starts with how numbers are stored.

**Next:** [Chapter 6 — Precision & Quantization →](06-precision-and-quantization.md)
