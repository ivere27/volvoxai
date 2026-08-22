# Weight banks: runtime-selected parameters for adapters and MoE

Status: stages 0-4 implemented across CPU, WASM, WebGPU, native C, and CUDA, for inference and training
Scope: `volvox-graph/v1` schema, `Model`, `CompiledModel`, `ExecutionContext`, backend SPI

## Problem

Two requirements that the current graph model cannot express:

1. **Selective residency.** A model carries N parameter groups but only k are
   needed for a given execution (k=1 for router-selected LoRA families, k=top_k
   for MoE). Today the whole bank is one dense weight and is fully resident.
2. **Post-export growth.** A new adapter family must be addable without
   re-exporting and recompiling the model.

### Evidence from the current code

MoE is already implemented and already on the logical path:

- `MoERouter`: `input`, `weight [feature, experts]` → `indices`, `weights`,
  each `[...prefix, top_k]`. Params `num_experts`, `top_k`, `temperature`,
  `normalize`.
- `MoELinear`: `input`, `expert_weight [experts, in, out]`, `route_indices`,
  `route_weights` → `[...prefix, out]`.
- Both have `inferConcrete` **and** `proveDomain`, so they are first-class in
  the bounded-shape system, not legacy-only.

Both walls are explicit in the source:

```ts
// ts/ops/operatorShapeContracts.ts — proveMoERouter
const experts = fixedDimensionValue(inputs.weight.shape[1], request.environment);
if (experts === undefined) {
  fail('UNPROVABLE_DYNAMIC_FEATURE', "operator input 'weight'.shape[1]",
       'the expert count must be fixed over the complete domain.');
}
```

```ts
// ts/ops/moeLinear.ts — the bank is one contiguous buffer
const expertBase = expert * dIn * dOut;
expertValue += input.buffer[row * dIn + d] * expertWeight.buffer[expertBase + d * dOut + col];
```

```ts
// ts/core/Graph.ts (logical) — weights cannot carry a symbolic extent
export interface WeightDescriptor extends TensorDescriptorBase {
  readonly kind: 'weight';
  /** Weight shapes are fixed, so this refinement contains numbers only. */
  readonly shape: readonly number[];
}
```

And there is no weight-only refresh path: `CompiledModel` captures
`weightRevision` at construction and exposes no `refreshWeights`/`updateWeights`.
Changing any weight today means recompiling.

## The reference workload

`tiny_receipt_vqa_..._lora_router_direct_novalue_e100_onnx`:

| | |
|---|---|
| total parameters | 22,069,783 |
| adapter + router | 708,008 (3.2%) |
| per family | 81,920 (`[64,320]` down + `[320,64]` up, encoder and decoder) |
| families | 8 |
| router | 52,648 (`[160,320]` → `[8,160]`) |

Routing is **per sequence** here: the encoder emits `selected_family_ids`, and
`family_ids` is a graph input on both encoder and decoder. Only one family is
ever used per request, so 573,440 parameters (~2.2 MB fp32) are resident for
nothing.

MoE routes **per token per layer**, so the selection cannot be hoisted to the
host. Any design that only serves the per-sequence case is a dead end.

## Design

Introduce a **bank**: a weight whose leading axis is a slot index, where slot
residency is a context-owned property rather than a compile-time constant.

### Graph document (implemented)

Weights are supplied by the safetensors payload, not declared in the document,
so a bank is declared as a separate table mapping a weight name to the bounded
dimension that governs its slot axis:

```json
{
  "dimensions": { "F": { "min": 1, "max": 32 } },
  "banks": { "decoder_adapters.down": "F", "decoder_adapters.up": "F" }
}
```

`parseBanks` in `ts/core/Graph.ts` checks that the dimension is declared, that
the named tensor is a supplied fixed weight, that it has a slot axis plus at
least one payload axis, and that the supplied slot count lies inside the bound
(honouring `multiple_of`). `tools/exporter/runtime_ir.py` accepts and validates
the same field so native and browser read one format.

The payload shape stays concrete — `WeightDescriptor.shape` is still numbers —
and `WeightDescriptor.bank` carries the bounds. The important consequence is in
the fingerprint: a bank weight contributes its **dimension name**, not its
current slot count, so filling a slot keeps the definition identity while
widening the bounds does not.

### Ownership

This follows the split the runtime already enforces for shapes.

| Owner | Holds |
|---|---|
| `Model` | logical bank layout, slot-count bounds, portable full-bank payload, weight revision |
| `CompiledModel` | immutable compiled host backing, slot layout, kernel selection, and every selection-independent raw/packed representation |
| `ExecutionContext` | **which slots are resident**, selected-row staging/COW, selection-dependent device or packed storage, residency generation |

`CompiledModel` must not own a resident slot set for the same reason it must
not own a mutable activation arena: two contexts may need different slots
concurrently.

### Selection

No new operator. `MoERouter`/`MoELinear` keep their contracts; the change is
that `expert_weight` may be a bank. For the per-sequence adapter case the
existing `Gather` over the bank axis is enough.

### What each requirement gets

| Requirement | Mechanism |
|---|---|
| 1 resident of 8 | context uploads one slot |
| top-k resident of N | context uploads the k slots the router selected |
| add a family, ≤ max | fill an unused slot — topology and fingerprint unchanged, **no recompile** |
| add a family, > max | `Model.derive()` with widened bounds → new definition identity → recompile |

`derive()` already has exactly this contract: *"Capture a successor weight
revision. Definition identity is retained only when the canonical logical
fingerprint, including bounds, is unchanged."*

## Required changes

1. **Schema** — `kind: "bank"` and `slot_dimension` on weight descriptors;
   permit a bounded dimension in axis 0 of a bank weight.
2. **Shape contracts** — `proveMoERouter`/`proveMoELinear` accept a bounded
   expert extent when the weight is a bank; require `top_k <= min`.
3. **Snapshot** — per-slot weight storage and a slot-granular weight revision.
4. **CompiledModel** — own the immutable full-bank backing and per-slot layout;
   share any raw or packed representation that does not depend on the context's
   selected slot set.
5. **ExecutionContext** — resident slot set, upload/evict, and a residency
   generation that participates in plan-cache keying.
6. **Backend SPI** — pass the exact resolved residency plan into context
   preparation. Backends that cannot stage a partial bank retain full-bank
   residency or reject the selection explicitly.
7. **Kernels** — `MoELinear` indexes a slot table instead of `expert * dIn * dOut`
   into one buffer.

## Staging

### Stage 0 — export-time collapse (done)

`Concat` is now foldable (`tools/exporter/optimizer/typed_constant_folding.py`).
A router-selected export emits `Unsqueeze(weight_k) -> Concat(all k) -> Gather`,
and folding turns the first two into **one stacked initializer** instead of a
tensor rebuilt on every execution.

Two consequences, both covered by tests in
`tools/exporter/tests/test_typed_structural_passes.py`:

- Router-selected package: `Gather` survives (the selector is a runtime input),
  but the bank is materialized once. Still N-resident.
- Family-pinned package: with the selector bound by
  `RuntimeInputSpecializationPass` the `Gather` folds as well, so unselected
  families never enter the payload. **This is 1-resident with no runtime
  mechanism at all.**

The fold only applies to internal values — a public output is never folded
away. In the reference model the gathered slice feeds `Transpose -> MatMul`, so
it qualifies.

Stage 0 means the runtime bank is needed only when **one** package must serve
any family while paying memory for just the resident ones, and for MoE.

Usefully, the folded stack is already exactly the bank layout: one contiguous
`[slots, ...]` initializer.

### Stage 1 — declare the bank (done)

The `banks` table, `WeightDescriptor.bank`, `Model.weightDescriptors[].bank`,
slot-count-independent fingerprinting, and exporter validation are implemented.
Covered by `tests/graph.test.mjs` and
`tools/exporter/tests/test_dynamic_runtime_ir.py`.

No execution behavior changes yet: a bank is still uploaded and resident as one
tensor. What it buys is that the runtime now knows axis 0 is sliceable and how
far it may grow.

### Stage 2 — context-owned residency (done)

`ExecutionContextOptions.bankResidency` names, per bank, the ascending global
slot ids to keep. A bank left out stays fully resident.

```js
const context = await compiled.createContext({
  bankResidency: { experts: [3] },
});
```

The resolver validates the request against the declared bank (unknown bank,
empty set, out-of-range id, duplicate or descending ids all fail with
`INVALID_BANK_RESIDENCY`), shrinks the bank's resolved tensor to
`[residentCount, ...payload]`, and `stageBankSlots` copies only those slots
into fresh contiguous storage.

Residency is part of plan identity: the plan signature gains a
`|banks:name=slots` suffix, so two contexts of one `CompiledModel` holding
different slots never share a plan-cache entry. `BoundExecutionGraph` and
`CPUShapeExecutionContext` recompute the same suffix when they revalidate.

### Stage 3 — bounded slot count (subsumed by stage 1)

The original plan was to relax `fixedDimensionValue` in
`proveMoERouter`/`proveMoELinear`. That turned out to be unnecessary: because a
bank keeps a **concrete** payload shape and expresses growth through the
fingerprint instead, the domain proof still sees a fixed extent and needs no
change. Post-export growth is already unblocked by stage 1's
slot-count-independent fingerprint.

What remains genuinely open is router growth. `MoERouter`'s weight is
`[feature, experts]`, so adding an expert changes the router too — that is a
retraining concern, not a runtime one, and no schema can hide it.

### Stage 4 — kernel slot indirection (done, all targets)

Routes stay expressed in **global** slot ids regardless of what is resident.
`buildGraph` attaches the resident slot ids to any node reading a bank, and each
kernel maps global id to staged row. The mapping is the same everywhere: a
`slot_rows` table of length `slot_domain` holding the staged row per global id,
or `VX_MOE_SLOT_ABSENT` (`0xffffffff`) when the context did not materialize it.

| Target | Entry point |
|---|---|
| CPU (JS) | `_cpuMoELinear` reads `node.residentSlots`; `backwardMoELinear` maps the same way |
| native C **and** WASM | `vx_moe_linear_banked_f32` in `portable_inference_kernels.c` — one implementation, `WASM_EXPORT`ed |
| native and WASM training | `volvoxai_training_moe_linear_banked_f32` and `..._backward_banked_f32`; `WasmTrainingKernels` selects the banked entries whenever the node carries a slot table, and uploads the table through the same arena as the gradients |
| WebGPU inference | `shaders/inference/moeLinear.wgsl` binding 7 plus `params.slot_domain` |
| WebGPU training | `shaders/training/moeLinearBackward.wgsl` uses binding 10 for global-slot→staged-row, binding 11 for staged-row→global-slot, and binding 12 for the 32-byte params uniform. `input_main` and `route_main` use the first map; `weight_main` and `bias_main` use the inverse |
| native Vulkan / OpenGL / Metal | `vk_graph_moe_linear_f32` / `opengl_graph_moe_linear_f32` / `metal_graph_moe_linear_f32` dispatch the forward `moeLinear` shader through `try_gpu_graph_moe_linear`; training consumes the same 13-binding backward ABI as WebGPU (maps 10/11, params 12) |
| native CUDA | `vx_cuda_moe_linear_f32` takes `slot_rows`/`slot_domain` for forward; its manual backward kernels and host launcher use the same 13-binding training ABI (maps 10/11, params 12) |

The native graph schema accepts the same optional `banks` table
(`vx_graph_schema_valid`), so a banked package loads on native as well as in
the browser. Gradients accumulate into the staged rows, so a partially resident
context trains exactly the families it materialized.

The 13-binding training layout is one current cross-backend contract, not a
legacy 11-binding alternative. Vulkan, OpenGL, Metal, and CUDA validate the
same two mapping buffers and params position before dispatch.

`moe_linear_f32` keeps its original signature and delegates with a NULL table,
so the fully resident path is unchanged.

Routing to a non-resident expert fails on every target rather than reading a
neighbouring slot: the CPU and WASM kernels raise, the C kernel returns 0, and
the shader skips the term.

One ownership consequence surfaced here. The CPU JS, WebGPU, and WASM compiled
owners retain the full immutable host bank once. CPU contexts borrow that
storage for full residency and stage a private selected slice for partial
residency. WebGPU additionally shares compiled device buffers only for
non-banked fixed weights; every bank device buffer is context-private because
its contents depend on residency. WASM owns one full raw/packed bank in its
compiled linear-memory prefix, borrows it for full residency, and stages only a
partial selection and its derived pack in the context's mutable region. This is
the same rule as the residency set itself: shared where invariant,
context-private where selected.

## Native residency policy and physical ownership

Native still declares the selected slot set on `VxModelSource`, so every context
created from that Model/compiled revision receives the same residency policy:

```c
typedef struct VxBankResidency {
    size_t struct_size;
    const char* bank;         /* named by the graph document's "banks" table */
    const uint32_t* slots;    /* ascending, unique, global slot ids */
    size_t slot_count;
} VxBankResidency;
```

`vx_runtime_load_model` scans the supplied weight metadata, requires every
declared bank to occur exactly once with a rank and slot extent allowed by its
dimension, and validates requested slots against that actual extent. The Model
retains the immutable request. Each built-in `VxCompiledModel` then reads and
parses each accepted safetensors shard once into a reference-counted immutable
weight store. Compile validation and every context borrow those blobs and
metadata; context creation does not reopen or parse the files.

`g_t` and `g_weight_files` are accessors for the currently scoped private
`VxEngineState`, not one process-global table shared by every context. A
compiled context clones the safetensors tensor descriptor table while retaining
the compiled blob, which gives residency metadata and selected payloads a
context-local owner even though the selection API remains on `VxModelSource`.

`engine_bank_residency.inc` then does the work between loading the weight files
and building the graph:

1. **Copy on write for compiled contexts.** The context copies the selected
   rows, in canonical slot order, into its own overlay and points its private
   safetensors and execution descriptors at that overlay. The compiled full-bank
   bytes and descriptor table never change. A standalone mutable authoring
   engine may still compact its private loader-owned bytes in place; that path
   does not borrow a compiled store.
2. **Follow the context descriptor.** The published `T` and the context's
   safetensors descriptor receive the staged `shape[0]`/`numel` and overlay
   pointer. Sibling contexts keep distinct descriptor tables and overlays while
   continuing to retain the same compiled source blob.
3. **Bind the slot table.** Every node reading a sliced bank gets
   `resident_slot_rows`/`resident_slot_domain`, which reaches the CPU-fallback
   and CUDA kernels unchanged.

Because the slice happens before `build_graph`, every downstream shape check
sees the staged extent rather than the bank's full one. Closing one context
frees only its descriptor table and selected-row overlay. The compiled store
survives when the live context count reaches zero, so reopening uses the same
raw blobs without another file read; final compiled release frees the store.

This fixes raw safetensors duplication but is not a claim that all native
prepared weights are compiled-owned. CPU F16 widening and CPU prepacked caches
remain context-private, as do the legitimate selected-bank overlays. Built-in
Vulkan/OpenGL/Metal/CUDA graph and device caches are also still prepared per
context. Moving their immutable portions to `VxCompiledModel` and measuring
their aggregate physical high water remain in `TODO.md`.

## Device MoELinear routes

Registry class `D` means "this route may decline and permit runtime fallback" —
not "this backend has no kernel". native-cpu and CUDA were always `D` **and**
had real kernels. Vulkan, OpenGL and Metal were the genuine gap: their node
routes had no `MoELinear` case, so every such node declined to the portable CPU
kernel — a device round trip in the middle of a graph, which matters more for
the native robot target than a missing feature would.

The forward shaders already existed
(`native/shaders/{spv,glsl,metal}/moe{Linear,Router}.*`, compiled from the same
WGSL the browser uses), so only the engine wiring was missing: `moe_linear` and
`moe_router` slots on the `VxF32Ops` vtable, one implementation per backend, the
shared `try_gpu_graph_moe_linear` / `try_gpu_graph_moe_router` validators, and a
case for each op in every node route. Because `moeLinear.wgsl` was already
bank-aware, partial residency came along for free.

`MoERouter`'s weight is `[d_model, experts]`, so its expert axis is 1 and it is
never a slot-indexed bank — the router route carries no slot table.

Verified on an RTX 3090 (`native/tests/test_moe_gpu_route.c`, a router feeding a
linear so one graph exercises both routes): Vulkan, OpenGL and CUDA all match
the portable kernel. CUDA needs `-DVOLVOXAI_ENABLE_CUDA=ON`, which is off by
default, so a stock build reports it as unavailable and skips it. Metal is
implemented but has no device in this environment.

## The exporter declares banks

`_detect_weight_banks` in `tools/exporter/runtime_ir.py` marks a fixed weight as
a bank when its axis 0 is selected at run time. Two shapes qualify:

- `MoELinear`'s `expert_weight`, which is an expert bank by definition.
- A `Gather` with `axis` 0 over an initializer whose **indices are not
  themselves constant** — the LoRA-family form that stage 0 folding leaves
  behind. A constant selection folds away entirely and is not a bank.

Each bank gets its own synthesized `bank_<tensor>` dimension with bounds
`[1, current slot count]`, so the exported count becomes the growth ceiling.
Declaring a bank is additive: it only tells a runtime that axis 0 is sliceable.
Nothing is required to use it.

Three closed-schema validators had to learn the field —
`vx_graph_schema_valid` (native), `_require_closed_dynamic_v1`
(`tools/export_safetensors.py`), and the TinyReceipt importer, which also had to
stop comparing `dimensions` verbatim now that bank dimensions are appended.

On the reference model this yields, with no hand editing:

```
encoder  banks: {"w125": "bank_w125", "w127": "bank_w127"}
decoder  banks: {"w86":  "bank_w86",  "w88":  "bank_w88"}
         bank_w125: {min: 1, max: 8}
```

## Quantized banks

A per-axis affine domain **on the slot axis** has no legal consumer, so residency
never meets one:

- `Gather` refuses it outright — "Gather indices would reorder the per-axis
  affine metadata" — because selecting slots would permute the scales.
- `MoELinear` is float32-only.

Per-tensor quantization is unaffected: one scale covers the whole tensor and
survives slicing untouched. `resolveGraphShapes` still slices per-axis metadata
on axis 0 alongside the payload, so the two stay consistent if a future consumer
ever allows that combination, but nothing reaches it today. Both behaviours are
pinned in `tests/weight_bank.test.mjs`.

## Rejected alternatives

- **Adapter matrices as graph inputs.** Simplest for the per-sequence case:
  the host uploads the selected family, no recompile ever, unbounded families.
  Rejected because MoE routes per token inside the graph, so the selection
  cannot be hoisted to the host. Adopting it would mean building the mechanism
  twice.
- **`AdapterManager`-style runtime weight injection.** Mutates graph topology
  and weights in place, which contradicts the immutable snapshot, the exact
  weight revision, and the bounded-domain proof. It also cannot express top-k
  routing. This is being deleted, not revived.
- **Bounded dimension on ordinary weights.** Insufficient on its own: it fixes
  post-export growth but not residency, because the weight is still one dense
  tensor.
- **Per-family specialized packages instead of a bank.** After stage 0 this is
  free and gives exact 1-resident behavior, so it is the right answer whenever
  the caller can choose the package. It is not a replacement for the bank: it
  cannot serve router-selected `auto` mode from one package, and it cannot
  express MoE, where routing is per token inside the graph.
