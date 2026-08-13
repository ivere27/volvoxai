# TODO

This is the authoritative list of unfinished work. Stable contracts and current
behavior belong in [`ARCHITECTURE.md`](ARCHITECTURE.md); completed implementation
history and benchmark evidence are intentionally not repeated here. Detailed
Runtime policy belongs in the
[scheduling and dynamic batching design](docs/scheduling-and-dynamic-batching-design.md).
An item is removed only after production integration and its
correctness/performance gate both pass. A standalone type, test seam, or callback
fixture is not completion.

Items are grouped by kind and tagged `KIND · PRIORITY`.

| Kind | Meaning |
| --- | --- |
| `BUG` | shipped code is wrong, or violates a contract the repository states |
| `CONTRACT` | a decision that has to be written down and versioned before code depends on it |
| `FEATURE` | new capability |
| `TEST` | a safety net or a proof that does not exist yet |
| `PERF` | measurement, budget, or optimization |
| `INFRA` | build, CI, distribution |

| Priority | Meaning |
| --- | --- |
| `P0` | wrong behaviour, or blocks a whole chain. Do these first. |
| `P1` | next. Something already planned depends on it. |
| `P2` | planned, nothing blocks on it today. |
| `P3` | research or undecided. |

### Order

Priority is authoritative across categories: close P0 correctness and contracts,
then P1 ownership/scheduling work and its release gates, then P2/P3 capability,
performance, and research items. Dependencies stated inside an item take
precedence over file order.

---

## 1. Bugs

- [ ] `BUG · P0` **Finish native derived/device invariant-resource ownership**
  - Move immutable native CPU F16 widening and prepacked weight-cache payloads
    from each `VxExecutionContext` to `VxCompiledModel`. Keep selected-bank
    copy-on-write overlays context-local and charge their exact bounded capacity.
  - Move immutable Vulkan/OpenGL/Metal/CUDA graph and device weight resources to
    the compiled owner. Contexts retain only mutable activation/workspace,
    binding, command, result, decode/KV, adapter, and bank-overlay state.
  - Vulkan mutable compute/staging allocations and CUDA replay-plan host
    metadata are now bounded and charged. This does not close the compiled-owner
    migration or the physical high-water proof for opaque CUDA GraphExec storage.
  - Measure one compiled model with N live contexts and close-all/reopen using
    native RSS/allocation counters and physical-device high water. Separate the
    compiled raw safetensors store from remaining CPU derived caches and native
    GPU allocations; requested API bytes alone are not physical VRAM evidence.

- [ ] `BUG · P1` **WebGPU `device.lost` has detection but no recovery**
  - Recreate the device, context, buffers, pipelines, and resource-domain epoch.
    Invalidate every old lease and prepared route. Retry only work that has not
    entered provider execution and whose policy permits it; submitted stateful
    work must fail with an explicit reset requirement rather than being replayed.

---

## 2. Contracts

Decisions that are still missing from the cross-language contract.

- [ ] `CONTRACT · P0` **One authoritative representation of per-sequence active length**
  - Specify one dense value representation for query/key active lengths in
    JavaScript, C, protobuf, and Rust; it must not introduce a public ragged
    tensor or participate in output-shape inference.

- [ ] `CONTRACT · P0` **Write down the per-lane semantics the code implements**
  - Per-sequence position, finished/inactive lane, causal-mask, adapter selection,
    output validity.
  - Batched-step atomicity: every selected lane commits or the whole step rolls
    back.
  - Page, slot, prefix and request ownership, and the legal transitions between
    free, reserved, resident, shared, evictable and retired.
  - Cancellation and context-close behaviour for queued and submitted work.
  - Which API layer owns the scheduler, and that it never mutates a logical
    snapshot or silently merges unrelated context state.

- [ ] `CONTRACT · P1` **`g_kcache` disposition — a decision, not code**
  - `SDPA` (packed QKV) keeps its own `g_kcache[node]` / `g_vcache[node]` heap
    arrays (`engine_runtime_dispatch.inc`), a third KV representation beside the
    retained-activation prefix TS and the native W8A8 row path share.
    `CrossSDPA(causal=true)` is how F32 causal decode is actually spelled, and
    `SDPA` is not in the TS attested row set at all — a native-only path with no
    TypeScript counterpart.
  - Decide first: (a) normalize `SDPA` into the `CrossSDPA` path and retire
    `g_kcache`, or (b) promote `SDPA` to an operator owning an explicit KV port.
    Either stops paged KV being implemented a second time inside `g_kcache`.

- [ ] `CONTRACT · P1` **Finish the stateful SCHEDULED contract**
  - Request states: admission, reservation, prefill, decode-ready, submitted,
    completed, cancelled, failed, retired.
  - Fairness, priority, deadlines, maximum queue depth, retry hints, consume/ack
    backpressure, and bounded result retention.
  - Deterministic output ordering and per-request result ownership when execution
    order differs from admission order.
  - Cancellation before admission, while queued, while submitted, after result
    publication.
  - Shutdown and device-loss behaviour without silently abandoning accepted work.
  - Session FIFO, state/KV ownership, recovery/reset, and freshness-policy
    interaction across multiple models and clients.

- [ ] `CONTRACT · P1` **Freeze the redefined unreleased v1 ABI/SPI/protobuf surfaces**
  - There is deliberately no v2, compatibility shim or stale-caller
    negotiation for this redesign. Before release, freeze the exact v1 struct
    layouts and provider contracts, regenerate every binding, and keep
    cross-language layout/provenance tests as the authority.

- [ ] `CONTRACT · P2` **Qualification hardware bar**
  - At least one real Vulkan adapter and one real OpenGL adapter; software
    renderers do not count. Keep the portability/consensus promotion gate
    (independent physical adapter identities) separate from first-route
    qualification.

---

## 3. Features

- [ ] `FEATURE · P1` **B>1 decode — transactional remainder**
  - Make seed, step, reset, error and close transactional across the whole batch.
  - Preserve exact result ownership after subsequent steps, resets and context
    closure.
  - Keep feature, head, vocabulary, quantization and weight geometry invariant
    across a seeded generation.

- [ ] `FEATURE · P1` **Metal device rows — compile and verify**
  - `MetalBinding.offset`, `MetalGraphWindow` and the eight converted decoder ops
    were written blind: no machine in this project can compile `.m`.
    `vx_runtime_backend_has_device_rows()` already includes `g_use_metal`, so the
    device-row path runs the moment it builds. A first macOS build starts there.

- [ ] `FEATURE · P1` **Native Vulkan/OpenGL public numerical qualification**
  - **Qualification authority**: define the exact Vulkan and OpenGL
    operator/tensor/layout/dtype sets in the kernel registry; complete
    bounded-domain proofs for every qualified operator and package; mark exporter
    routes qualified only after implementation and numerical tests pass
    (documentation or a capability probe is not authority); make package
    validation reject a claimed native-GPU route outside the proved domain;
    preserve the explicit capability reasons for unqualified packages.
  - **Public runtime execution**: bind exact shaped inputs, concrete outputs,
    capacities and dispatch metadata through the public native API/provider ABI;
    execute through `native/volvoxai` with no CPU fallback, operator fallback,
    private engine bypass or test-only graph construction; make
    allocation/rebinding transactional and retire resources only after device
    completion; keep stable copied result ownership and exact concrete shapes;
    report selected backend, physical adapter, model/source identity, shape
    signature, tactic, allocation high-water and runtime lineage; return stable
    structured failures for unsupported domains, device limits, compilation errors
    and device loss.
  - The numerical ladder and hardware evidence are under **Tests**.

- [ ] `FEATURE · P1` **Runtime coordinator — remaining production work**
  - Add worst-case route workspace, physical/aligned device allocations and KV
    pages to admission; expose retry hints and bounded telemetry/high-water
    accounting. Add exact-buffer reuse/transfer pools so equal-layout `LATEST`
    replacement does not require GC-dependent old-plus-new input headroom.
  - Add stable physical resource-domain identities for providers that expose
    multiple devices, measured `T(B)` selection, device-resident scheduled input
    leases and zero-copy/per-lane device result leases.
  - Replace native's single synchronous worker with completion-driven
    per-resource-domain submit/poll/fence dispatch. Remove avoidable double
    staging and make accepted work own its resources until the provider
    completion boundary. The synchronous authored-symbolic-B executor is
    already qualified for Vulkan/OpenGL/CUDA; Metal and asynchronous completion
    remain outside that completed slice.
  - Report device busy/wall, queue delay, useful/padded rows, high-water bytes,
    deadline misses, active group count, and fairness by phase/resource domain.
    Keep the monotonic policy clock distinct from elapsed measurement time and
    retain only bounded windows, including queue-depth max/P50/P99.

- [ ] `FEATURE · P1` **Fixed-B=1 typed-IR batch lifting**
  - Transform typed IR before shape resolution; never prepend an axis to a
    resolved plan. Introduce and track the request axis through public and
    intermediate tensors, remap axis/default-derived parameters, broadcasting,
    indexing, reshape, quantization, aliases and liveness, then rerun complete
    shape/domain/resource/memory/tactic/backend proof.
  - Fail closed to scheduler B=1 when any operator can mix lanes. Pin adversarial
    regressions for `ArgMax`'s default axis, `SSMScan` rank-dependent behavior,
    and `Transpose`'s omitted reverse permutation.
  - Reuse the completed canonical graph-fingerprint-bound lane-independence
    proof and exact provider-echo gate after the new lift. This item does not
    include reintroducing provider self-attestation: absent or mismatched core
    evidence must continue to fail closed to B=1.

- [ ] `FEATURE · P1` **Stateful Runtime session batching**
  - Move prefill/decode contributions onto a shared compiled target and Runtime
    coordinator instead of private per-session contexts. Preserve per-session
    FIFO, KV generation/ownership, cancellation, reset, and atomic batch commit.
  - Construct prefix identity inside Runtime/provider code from every semantic
    model, tokenizer, adapter, prompt, mask, position, quantization and device
    epoch input; never treat an application string as authority. Recompute or
    explicitly reload an evicted prefix before execution.
  - Move `TinyReceiptSplitSession` off its private serialized B=1 explicit-KV
    context without weakening its session FIFO or rollback guarantees.

- [ ] `FEATURE · P1` **Scheduler execution design remainder**
  - Bounded dense slot capacity with explicit active-slot metadata.
  - Replace current queue/route scans and per-batch temporary allocations with
    per-route FIFOs, a per-domain ready heap and bounded request/frame slabs.
    Measure scheduler CPU time and bytes separately from provider execution.
  - Enforce per-dispatch row, token, input/staging byte, output byte, and padding
    budgets in addition to request and lane counts.
  - Support mixed prefill/decode only after separate reference paths are correct;
    do not force one unsafe universal kernel.
  - Integrate membership without recompilation, lane parking, and per-lane
    failure isolation into Runtime serving.
  - Add a bounded device/route-specific `T(B)` table. Start a cold route at B=1,
    select only measured-profitable legal batches, and invalidate measurements
    on compiled/provider/device/driver/tactic epoch changes or material thermal,
    power, and memory-pressure shifts.

- [ ] `FEATURE · P1` **Whole-device, cross-model resident admission budget**
  - Extend Runtime admission to the sum of exact compiled weights,
    activation/workspace maxima, aligned physical device allocations, in-flight
    result/readback buffers and KV pages for every route in the resource domain.
  - A legal per-model B is not admissible when concurrent routes would exceed
    the shared device high-water. Reserve before mutation, release through the
    completion/result fence, and feed pressure back into the route's operating-B
    cap. Until this ships, successful static proof is not a claim that the legal
    maximum batch will fit alongside other browser clients or robot sensors.

- [ ] `FEATURE · P1` **Direct per-lane `q_len`/`kv_len` dispatch**
  - Connect runtime `kv_len: I32[B]` into the attention kernels to clamp loop
    boundaries per lane and skip the padding compute. This fits the existing dense
    layout: every lane still contributes the same number of query rows, only the
    key extent it may read differs.
  - These ports must never participate in output shape inference — that is what
    keeps a mixed-length batch from becoming a ragged tensor. The shape contract
    should prove it, the way `prove*` proves the rest.
  - `q_len` goes in with `kv_len`, not after. Mixed prefill/decode needs it, and
    adding it later breaks the ABI a second time.
  - Edit unreleased v1 in place with no compatibility layer. Update canonical
    WGSL, regenerate SPIR-V/GLSL/GLES/Metal projections, rebuild CUDA PTX, and
    rerun cross-backend layout/parity tests. Mixed query-row counts remain the
    separate flat-packing item below rather than a ragged dense tensor.

- [ ] `FEATURE · P1` **Bucketed and chunked prefill execution**
  - Add a typed `decode_prefill(row_start, row_count, lanes, ...)`-equivalent
    provider operation with exact query staging, causal/query keep-mask semantics,
    provisional KV ownership, and atomic commit/rollback.
  - First batch prompts in bounded length buckets with explicit padding/result
    isolation; then admit measured chunk sizes through the same route contract.
    Mixed prefill/decode remains separate until flat packing removes the common
    `rows_per_lane` requirement.

- [ ] `FEATURE · P1` **Browser/robot host integration**
  - Browser extensions: one multi-client Worker/offscreen host owns the Runtime,
    bounded result consume/ack, device-loss recovery, and client cancellation.
  - Robots: integrate long-lived video, audio, stateless inference, and stateful
    decode streams under one resource-domain budget and explicit freshness policy.
  - Ship one runnable native edge/robot application that connects the opaque
    lifecycle to real sensor input and actuator output.

- [ ] `FEATURE · P1` **Application-owned generation streaming**
  - Add a reusable helper over `ExecutionContext.decode` with greedy,
    temperature, top-k and top-p sampling; deterministic seeds; EOS, maximum-token
    and stop-sequence termination; and cancellation. Keep sampling, tokenization
    and text policy out of the core inference entry.
  - Define stable events and errors for cancellation, partial output, cache reset,
    and EOS or a stop observed while device work is already submitted, then expose
    the helper through `ReadableStream` / `AsyncIterator`.
  - Apply the same application policy around the optional in-process Synurang FFI
    plugin while inference stays on opaque `VxExecutionContext` / `VxResult`
    handles.

- [ ] `FEATURE · P2` **Shader-side paged addressing**
  - Zero-copy device addressing of the page table, replacing the staging gather.
    Same five-backend cost as any other attention shader change; justified by
    measurement, not by principle.

- [ ] `FEATURE · P2` **Paged KV — remaining lifetime and policy**
  - Retire device pages only after submitted work and open results no longer
    reference them; prove the device-completion side of the lifetime.
  - Make reference acquisition/release atomic with admission, cancellation and
    close.
  - Admission backpressure for requests that cannot reserve their maximum allowed
    working set.

- [ ] `FEATURE · P2` **int4 kernels**
  - Define the canonical group-quantized sub-8-bit Graph/storage contract,
    portable reference and parity suite before promoting any package or optimized
    unpack-in-register kernel. Int8 remains the package floor until then.
  - Extend the `physical_byte_tensor` gate
    (`tensor->elem_size != 1`), `physical_byte_dtype`, and the ~23 elementwise
    operators that multiply an element offset by `elem_size`. The comment on
    `physical_byte_tensor` in `engine_runtime_w8a8.inc` is the canonical list.
  - Do not change `dtype_size(VX_DTYPE_F4)` from 0: `numel * 1` is twice the real
    size, and that value is the bounds argument for gather/scatter.

- [ ] `FEATURE · P2` **Low-precision and decoder-attention execution**
  - Add real FP16/BF16 arithmetic where hardware supports it, including measured
    CUDA tensor-core tactics.
  - Add canonical grouped-query attention and F32 RoPE on WebGPU, Vulkan, OpenGL,
    Metal and CUDA; ship a reference RoPE/RMSNorm decoder exporter and package.
  - Revise the QSDPA Graph contract, portable reference, tests and native
    implementation before widening quantized head dimensions. Add fused-attention
    tactics and lift F32 GPU head limits only inside the resulting proved domain.

- [ ] `FEATURE · P2` **Provider operator coverage remainder**
  - Complete canonical Gather/GatherElements, typed ArgMax/NonMaxSuppression, and
    non-last-axis Softmax/LogSoftmax routes across the providers that claim them.
    Compact data-dependent results remain outside the public shape contract.

- [ ] `FEATURE · P2` **Speculative decoding**
  - Decode is memory-bandwidth-bound on edge devices: producing one token reads
    the whole weight set from RAM, so the arithmetic units idle waiting on the
    bus. Drafting k tokens cheaply and **verifying all k in one forward** amortizes
    that read over k tokens. It is the largest single tokens/sec lever on a phone
    or a robot, and it is entirely absent here.
  - Verify k drafted tokens through the chunked-prefill primitive above; reject
    the unaccepted tail by rolling its provisional active length and paged-KV
    reservation back atomically.
  - Draft source is a separate decision: a small draft model (a second compiled
    model — the scheduler's group key already carries `compiled_model`), a
    prompt-lookup/n-gram drafter (no model at all, very cheap on device), or a
    multi-token-prediction head where a checkpoint ships one. Start with
    prompt-lookup: it needs no second model and no training.
  - Acceptance rate is the whole economics. Report it, along with tokens/sec and
    the bandwidth saved, or the feature cannot be evaluated.

- [ ] `FEATURE · P3` **Beam search and constrained decoding**
  - Add them only on top of the context-owned cache, transactional lane state and
    application-owned generation policy; neither may bypass result ownership,
    cancellation or bounded admission.

- [ ] `FEATURE · P2` **StreamingLLM — bounded-memory infinite context**
  - An always-on device (robot, speaker, kiosk) runs for days. Today a long
    session grows KV until it hits the cache budget; dropping the oldest tokens
    outright degrades the model badly.
  - The published result is that keeping the **first few tokens** plus a recent
    window preserves quality, because those first positions act as an attention
    sink that the softmax denominator relies on. Policy: pin the sink pages, keep
    a sliding recent window, evict the middle.
  - Add (a) *pinning* for pages eviction may never take, (b) the sink+recent-window
    policy, and (c) positions for a window whose absolute positions no longer
    start at zero.
  - Separate from **learned attention sinks**, which are a model property (a
    per-head bias entering only the softmax denominator, e.g. gpt-oss) and are
    recorded under **gpt-oss — 공식 사양** in
    `docs/frontier-llm-support-plan.md`. StreamingLLM works without them; a
    checkpoint that has them needs both.
  - This is the repository's stated identity — bounded everything, no unbounded
    growth under any request sequence — applied to session length. It is also a
    precondition for the **Worst-case execution time envelope** item under
    **Undecided**: an engine whose memory grows with conversation length has no
    worst-case envelope.

- [ ] `FEATURE · P2` **1D flattened packing (`cu_seqlens` / VarLen attention)**
  - Unpadded `[N_total, D]` layout with cumulative sequence length offsets. This
    is what makes per-lane query length expressible without a ragged tensor, and
    therefore what unlocks mixed prefill/decode in one dispatch; see
    `ARCHITECTURE.md` under **Admission and dispatch**.
  - New binding plus an attention kernel that walks per-sequence extents from
    `cu_seqlens`, across all five backends. Do it when the scheduler's
    `1 - rows_useful/rows_dispatched` says partial batches are the dominant waste,
    not before.

- [ ] `FEATURE · P2` **Multi-threaded CPU/WASM execution**
  - Add bounded intra-op tiling/SIMD/threading and inter-lane work distribution,
    preserving the DIRECT B=1 regression budget.
  - WASM starts at B=1 and raises its cap only from measured `T(B)`, available
    workers, memory, and deadline slack; worker count alone is not a batch policy.
  - Design delivery within the fixed release filenames. A separate threaded WASM
    artifact requires explicit approval to change the release inventory.

- [ ] `FEATURE · P2` **GPU-resident token sampling**
  - GPU-side Argmax / Top-P / Temperature so a step reads back a 4-byte token ID
    instead of the full vocabulary logits tensor
    (`ts/backends/WebGPUResults.ts` still maps the whole output).

- [ ] `FEATURE · P2` **WebGPU `CrossSDPA` decode row acceleration**
  - Resolve the missing dynamic visible-prefix uniform
    (`ts/backends/WebGPUDecodeState.ts:249`) to enable incremental decode for
    encoder-decoder / VQA models.

- [ ] `FEATURE · P2` **Tokenizer: streaming decode and SentencePiece**
  - Add incremental decode that holds a partial UTF-8 sequence across steps
    instead of decoding whole buffers, and a SentencePiece/unigram vocabulary path
    beside the BPE one.

---

## 4. Tests

- [ ] `TEST · P0` **CI walks one ISA**
  - `.github/workflows/ci.yml` invokes `make verify_native_isa` once, so the
    runner exercises only its own native tier. Repeat it per clamp value
    (`baseline`, `avx2`, `avxvnni`, `avx512vnni`, downward from what the runner
    has) so one machine covers every compiled variant and automatically covers
    new ISA kernels.

- [ ] `TEST · P0` **Shared golden corpus for the decode/row domain proof**
  - `incrementalRowDomainSupported` (`ts/backends/quantizedRowExecution.ts`) and
    `hybrid_row_plan_build_locked` (`native/src/runtime/incremental_runtime.c`)
    prove the same property in two independently written implementations, with no
    corpus tying them together. Shape inference has seven vector files; this
    surface has none, and the symptom is the one shape inference already paid
    for — a decode graph accepted in the browser and refused on the robot, or the
    reverse.
  - Proposed `tests/decode_row_domain_vectors.json`: graph fragment +
    `changedInputs` + expected `{attested, refusal_node, refusal_reason}`. Both
    sides already carry refusal reasons as strings, so the corpus can compare
    *why* a node was refused, exactly as the shape vectors compare
    `expected_error.code/path`.
  - Distinct from `tests/decode_row_set_vectors.json`, which covers the row-index
    arithmetic (`decodeRowSet.ts` / `decode_row_set.c`), not the domain proof.

- [ ] `TEST · P1` **Shared golden corpus for independent-batch semantics**
  - TypeScript and native now implement the same
    `typed-independent-batch-proof/v1` trust boundary independently. Move the
    operator positive/negative vectors into one graph-fragment corpus consumed
    by both implementations so their accepted sets cannot drift silently.
  - Cover reduction/Softmax on B, row-major reshape mixing, transpose and
    squeeze axis transport, broadcast/concat/fixed-table gather, safe
    Conv/Linear/LayerNorm, BatchMatMul fixed operands, and per-axis
    quantization on B. Keep the current native end-to-end negative asserting
    scheduler B=1 and two physical forwards for an unproved cross-lane graph.

- [ ] `TEST · P1` **Shape inference: three implementations, one corpus**
  - 70 `infer*` and 70 `prove*` coexist in TypeScript (8,049 lines across
    `ts/ops/operatorShapeContracts*.ts`); native has 56 `vx_shape_infer_*`
    (5,752 lines across `shape_contract.c` + five `.inc`) and **zero** `prove*` —
    its symbolic proof lives in `public_api.c` under a different structure. The
    file split into eight modules changed the line counts and nothing else.
  - The risk is a graph the browser refuses and the robot accepts, or the reverse.
    Do the steps in order and stop at the end of each.
  - **Step 1 — widen the corpus to property-based generation** (highest leverage;
    it checks all three implementations at once). The shared corpus currently has
    6 files and 228 cases; the separate 7-case concat-affine corpus is TypeScript
    only. For each contract, sample legal bindings and assert
    `proveDomain → symbolic shape → substitute` equals `inferConcrete(concrete)`.
    Samples must include the first and last legal value after
    `min`/`max`/`multiple_of` alignment, degenerate `min == max` symbols, and rank
    bounds 1 and 8. **Emit the cases as JSON** in the existing
    `volvox-operator-shape-vectors/v1` schema and register the file in
    `native/CMakeLists.txt` — a TypeScript-only test leaves the native 56 blind.
    Generate failure cases too; `expected_error.code`/`path` are pinned today, and
    successes alone leave the refusal paths uncovered. **Run it against the
    current tree first** — a failing pair is an existing bug and outranks the
    refactor.
  - **Step 2 — TS ↔ native proof equivalence.** Does `proveGraphShapeDomain`
    (`ts/core/ResolvedShapePlan.ts`) accept and reject the same graphs as
    `vx_builtin_bounded_domain_proof` (`native/src/runtime/public_api.c`)?
    Rejection cases matter most. **Report divergences as a list; do not fix them
    here** — which side is right is a separate judgement.
  - **Step 3 — replace `inferConcrete` with substitution**, only if 1 and 2 are
    green. Hypothesis: v1 forces every axis to be constant-or-symbol, so a node
    `proveDomain` accepted has an output shape that is by definition a function of
    constants and bound symbols, and concrete inference folds to
    `shape.map(d => typeof d === 'number' ? d : symbols[d])`. Known exception:
    Concat's output-only symbols, needing
    `AcceptedGraphShapeDomainProof.affineSymbolRelations`. Survey all 70 pairs and
    classify: (a) identical but for wrapping, (b) `prove` adds checks and `infer`
    still folds, (c) genuinely different computation. **Several (c) means the
    hypothesis is wrong — drop step 3 and keep 1 and 2**, which already remove
    most of the divergence risk. Watch the params-dependent operators: `Reshape`,
    `Slice`, `Pad`, `Split`, `Concat`, `Expand`, `Gather`, `Resize`,
    `Conv2D`/`ConvTranspose2D`, anything using `declaredOutputs` or
    `variadicOutputs`. No big bang — pilot one family and stop. Do not weaken
    validation: check individually whether each dtype/quantization/params check
    `inferConcrete` performed is really shape-independent and already settled at
    proof time. Error codes and `path` strings are load-bearing diagnostics and
    stay.

- [ ] `TEST · P1` **B>1 negative and lifecycle tests**
  - Independent query/key lengths, causal masks, cross-attention memory, adapter
    selection and quantized KV.
  - One invalid lane rolls back the complete batched step.
  - No inactive lane or capacity suffix becomes observable.
  - Close/reset while compilation, allocation or device work is pending.

- [ ] `TEST · P1` **Actual-package numerical fidelity**
  - Localize the Tiny Receipt VQA INT8 Runtime-B1 differences from original ONNX
    Runtime in full encoder memory/KV and decoder logits/KV. Set
    tensor-specific justified tolerances, and re-run held-out token/field
    accuracy before calling INT8 end-to-end numerically qualified. Matching
    greedy tokens on a small fixture is not a substitute for this gate.
  - Keep several deterministic random and representative real-input fixtures per
    model and capture the exact first diverging node on whole-model failure.
  - Register a real F16 package and campaign in parity policy, or remove the
    otherwise-unused F16 tolerance policy.

- [ ] `TEST · P1` **Pinned tokenizer reference parity**
  - Compare the browser and native tokenizers against the reference implementation
    using pinned GPT-2/Neo release vocabulary and merge assets, including Unicode,
    partial UTF-8 and byte-level boundary cases. TinyReceipt's separate
    byte-fallback BPE vectors do not close this item.

- [ ] `TEST · P1` **Retained Trainer state and revision parity remainder**
  - Add a multi-step AdamW oracle for first/second moments and per-parameter step
    counts; prove `updatedTensorNames` and copied gradients from an earlier result
    remain stable after later steps.
  - Compile before and after a commit and prove each `CompiledModel` stays pinned
    to its own immutable revision. Compare checkpoint/resume with an uninterrupted
    run, including optimizer state.
  - Prove completed, accumulation-only and failed steps remain private across
    CPU JS, strict WASM and physical WebGPU.
  - Exercise `native/volvoxai-full train` numerically end to end while preserving
    the inference binary's zero-training-symbol boundary. Run the required Trainer
    subset on physical WebGPU; all three providers must match the independent
    reference within documented tolerances.

- [ ] `TEST · P1` **Decode numerical parity remainder**
  - Compare every declared output from prefill and each decode step on CPU JS,
    strict WASM and physical WebGPU, not only the chosen token or final cache.
  - Run physical numerical coverage for the device-feedback route through the
    same context-owned decode and result-lifetime contract.

- [ ] `TEST · P1` **Paged KV stress**
  - Allocation failure at *every* reservation point with exact rollback proved.
  - Shared-prefix reference counts under cancellation, copy-on-write, eviction and
    context/device close.
  - Cross-request data leak after page reuse.
  - Adversarial fragmentation and churn under a hard resident-page budget.
  - Page boundaries, partial tail pages, geometric growth, maximum capacity, mixed
    heads/layers.

- [ ] `TEST · P1` **Scheduler resilience**
  - Admission/removal every step, mixed lengths and completion times,
    cancellation, timeout, backpressure.
  - Determinism under repeated identical schedules.
  - Fairness and bounded starvation under adversarial arrivals.
  - Failure injection at every state transition (allocation, compilation,
    provider submission, device loss, completion publication).
  - Slot/page reuse cannot expose another request's inputs, KV, adapters or
    outputs.
  - Close contexts and the scheduler with queued and submitted work; verify the
    documented drain/cancel behaviour.
  - Exercise browser multi-client Worker/offscreen hosting and a long robot soak
    mixing video, audio, stateless frames, and stateful decode. Assert bounded
    request/result/device/KV high water, consume/ack backpressure, freshness,
    energy, and host/device traffic.

- [ ] `TEST · P1` **Native GPU numerical ladder and hardware evidence**
  - Focused kernel tests against an independent CPU reference; Level-1 isolated
    operator matrices on physical Vulkan and OpenGL; Level-2 mixed-graph matrices
    with exact run/fingerprint lineage; whole-model TinyStories and EfficientDet
    packages through the public runtime; quantized byte-exact checks where
    arithmetic is exact and documented tolerances/top-1 where floating-point order
    differs; dynamic small→large→small, allocation-failure rollback, context
    isolation and result-lifetime tests on physical devices.
  - Once a route is promoted to required execution, a capability skip is a failure
    for every package inside its qualified domain.
  - Bind evidence to adapter/vendor/device/driver, source fingerprint, model
    package, public runtime route and run ID. Keep first-route qualification
    distinct from cross-GPU consensus; require at least two independent physical
    adapter identities before claiming portability. TinyReceiptVQA is an optional
    physical performance workload once its required operators are publicly
    qualified — not a retroactive v1 blocker.

- [ ] `TEST · P2` **Feature promotion matrix**
  - Run this matrix when B>1 decode, paged KV, continuous batching, or native GPU
    qualification is nominated for promotion. Deferring an unrelated project does
    not block the base dynamic-shape v1 release.
  - Cross-language schema/protobuf/Rust vectors for the new length, page, slot and
    scheduler contracts.
  - CPU reference parity for every new state transition.
  - CPU/WASM/WebGPU/native-CPU B>1 and paged-KV parity.
  - Physical Vulkan/OpenGL L1, L2, whole-model, lifecycle and dynamic-shape parity
    for qualified routes.
  - Concurrent contexts/schedulers with different batches, lengths, adapters, page
    pressure and cancellation patterns.
  - Failure injection before and after every visible commit boundary.
  - Bounded cache/page/slot stress under adversarial request traces.
  - Inference/full composition, training-symbol boundary, generated-source,
    package and release-inventory checks.
  - Every hardware campaign seals provider/device/route evidence, trusted source
    revision, package fingerprint, declared output names and a finalized manifest.
    Byte-compare true-integer WebGPU outputs whenever exact arithmetic is required.
  - Keep `operation_list.md`, generated coverage, provider reports, build-profile
    documentation and package documentation synchronized with the promoted
    implementation.

- [ ] `TEST · P2` **Cross-provider isolated operator parity remainder**
  - Add parity-harness cases for CrossAttention, GatherElements, Interpolate1D,
    MoELinear, MoERouter, NonMaxSuppression, RoPE and SSMScan; promote QAdd and
    RequantizeLinear from whole-model-only coverage to isolated cases. Do not add
    undeclared operators merely to satisfy an old checklist.
  - Each case uses real integer indices where applicable, checks every declared
    output, and has an independent reference with explicit tolerances. Keep
    `operation_list.md` and the generated coverage report aligned; every supported
    provider passes or records an authored unsupported result.
  - Include representative multi-output provider cases that read only declared
    `ExecutionResult` outputs.

### Completion definitions

Each project completes independently; failing or deferring one does not
invalidate the dynamic-shape v1 release.

- **B>1 decode**: independent per-sequence lengths are authoritative and
  cross-language consistent; CPU, WASM, WebGPU and native CPU match full
  recomputation at every step; B=1 stays inside its regression budget; failure,
  reset, close and concurrency semantics are transactional and tested.
- **Paged KV**: private pages match contiguous KV exactly; sharing is immutable
  and reference-safe with correct copy-on-write; eviction never targets live work
  and cannot change results; page memory is bounded and improves representative
  long-context workloads.
- **Continuous batching**: admission, scheduling, cancellation, completion and
  shutdown are deterministic and failure-safe; results match independent request
  execution; churn causes no per-request recompilation or unbounded state;
  throughput or memory improves without violating latency/fairness budgets.
- **Native GPU qualification**: public runtime execution replaces capability skips
  for the declared qualified domains; physical L1, L2, whole-model, lifecycle and
  dynamic-shape evidence passes with exact source/model/device lineage; no
  qualified result uses fallback, bypass or a software renderer; unsupported
  domains still fail closed with authoritative reasons.

---

## 5. Performance and measurement

- [ ] `PERF · P1` **Baselines and regression budgets**
  - B=1 decode latency, throughput, memory high-water, specialization and
    KV-capacity.
  - Padded B>1 full-recompute baselines for short, mixed and maximum lengths.
  - Native Vulkan/OpenGL capability-probe time, allocation high-water and current
    unsupported reasons on the qualification machines.
  - Explicit regression budgets for the existing B=1 and constant-shape fast
    paths.

- [ ] `PERF · P1` **Continuous batching performance protocol**
  - Throughput, per-token latency, TTFT, P50/P95/P99, queue delay, fairness,
    padding waste, page utilization, high-water memory. Compare single-request,
    fixed padded batch, B>1 mixed-length and continuous batching under identical
    traces.
  - **Separate invariant-resource ownership from the result.** A route borrows
    one exact compiled artifact while a naïve baseline may create N contexts, so
    part of a memory win can be ownership rather than batching itself.
    Report the two separately, or first finish **native derived/device
    invariant-resource ownership** on the measured route. Not a blocker for the
    feature — only for the honesty of the number.

- [ ] `PERF · P2` **Select paged page size from measured workloads**
  - Rather than one hard-coded value per backend. CPU/WASM favour large pages
    (contiguous access), WebGPU/Vulkan favour small ones (fragmentation cost) —
    neither is claimed until measured.

- [ ] `PERF · P2` **Performance gates**
  - B>1: B=1 stays inside its regression budget; mixed-length B>1 performs less
    work or uses less memory than padded maximum recomputation; token steps do not
    compile once per lane or per generated token; plan/cache/capacity growth stays
    bounded under alternating B and length distributions. Apply this gate after
    direct per-lane query/key-length dispatch is integrated.
  - Continuous batching: demonstrate a throughput or memory benefit without
    exceeding the agreed tail-latency and B=1 regression budgets; prove
    specialization, pipeline, page and slot counts stay bounded during sustained
    churn.
  - No performance regression from the ISA work: verify with
    `benchmark_kernel_unit`, and build scratch benchmarks with clang — gcc numbers
    differ from the shipped build.

- [ ] `PERF · P2` **Close the EfficientDet CUDA/TensorRT gap**
  - Profile equivalent end-to-end workloads, add only measured tactics, and retain
    task-level numerical validation alongside latency and throughput evidence.

---

## 6. Infrastructure and distribution

- [ ] `INFRA · P2` **Re-grade native GPU backend tiers from evidence**
  - Establish Metal build/runtime evidence, then grade Vulkan, OpenGL, and Metal
    from current public-runtime qualification rather than historical assumptions.

- [ ] `INFRA · P2` **Distribution bundle slimming**
  - Exclude pure TS CPU operator kernels (`ts/ops/*.ts`) from the default
    production bundles in `dist/`.
  - Keep them reachable through a separate debug/reference entrypoint (e.g.
    `volvoxai/reference`) so DevTools step-through debugging and CI parity
    verification survive.

---

## 7. Undecided

Not scheduled. Recorded so they are not rediscovered.

- `P3` **Worst-case execution time envelope.** The largest unclaimed
  differentiator for a robotics engine, and entirely unexplored. Bounded shapes
  and a refused fallback path are the preconditions for a WCET claim and this
  repository has both — but there is no measurement, no envelope, and no statement
  of what would count as evidence.
- `P3` **Preemption between submitted device batches and explicit contexts.**
  Stateless SCHEDULED admission has priority/EDF/aging arbitration in TypeScript,
  but a submitted GPU batch and separately owned low-level context are not
  preemptible. Native and stateful session dispatch still need the same
  cross-route policy and device completion boundary.
- `P3` **One-call SwiGLU fusion.** Keep the composable `SiLU + Mul + Linear` form
  until a model-level benchmark proves a fused block is materially better.
- `P3` **Deferred platform breadth.** General dataset/augmentation/experiment and
  metrics platforms; frontier-scale pretraining, RLHF/DPO and higher-order
  autograd; multi-GPU collectives and tensor/pipeline parallelism; general Graph
  `If`/`Loop`; and model-family ecosystems without a concrete package, parity
  fixture and application owner remain unscheduled.
