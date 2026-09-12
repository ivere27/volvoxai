# TODO

This file tracks additional validation, implementation gaps, and optional
capabilities. The public contract is [volvoxai.proto](proto/volvoxai.proto);
current ownership and execution behavior are in [ARCHITECTURE.md](ARCHITECTURE.md).
Remove completed items. Historical measurements identify the artifacts they
tested; reproduce an old failure through the current API before treating it as
a current defect.

Priorities describe why work is needed:

- `P0`: a reproduced correctness or public-contract failure in the current
  release, with a failing case and expected behavior.
- `P1`: the next validation, resource-management, or delivery work.
- `P2`: an optional capability or optimization selected for a concrete workload.
- `P3`: research requiring a model, reference, and measurable success criterion.

Missing convenience features and unmeasured performance concerns alone do not
establish a `P0`. Product TypeScript remains generated API projections, thin
WASM/file transport, and the GPU device bridge. New application operations
require a proto change and generated dispatch.

## Current release qualification

- [ ] `P1` Reproduce TinyReceipt FP32/INT8 whole-model comparisons through the
  current generated API and current packages. Compare every declared encoder,
  prefill, and step output, including logits and explicit KV, on native CPU,
  WASM CPU, and physical WebGPU against an independent ONNX Runtime reference.
  Bind reports to exact inputs, packages, artifacts, and per-tensor tolerances;
  investigate any remaining mismatch before adjusting a tolerance.
- [ ] `P1` Add an independent multi-step AdamW oracle for native full, WASM CPU,
  and WebGPU. Check weights, moments, bias correction, weight decay, clipping,
  and accumulation. Extend checkpoint/resume and commit/revision-pinning
  coverage to native full; CPU/GPU agreement and checkpoint self-consistency
  alone cannot establish optimizer arithmetic.
- [ ] `P1` Qualify the C tokenizer with pinned GPT-2/Neo vocabulary and merge
  assets and independent expected token IDs/text. Include Unicode, partial
  UTF-8, byte boundaries, and token limits. The existing old-TypeScript
  comparisons establish migration parity, not this separate model-asset gate.
- [ ] `P1` Qualify native Metal row execution on macOS and native Vulkan/OpenGL
  on physical adapters through generated calls. Record isolated operators,
  mixed graphs, whole models, dynamic bindings, context isolation, retained
  outputs, and failure handling for each claimed backend/domain.
- [ ] `P1` Extend serialized native/WASM contract corpora with reproducible
  property-generated boundary and refusal cases for shape-domain proof,
  independent batching, and row arithmetic. Record seeds and expected
  outcomes; both transports must consume the same cases.

## Failure contracts and memory

- [ ] `P1` Define and test the decode failure matrix for B=1 and B>1, including
  one invalid lane, shader/submission failure, readback failure, and device
  loss. Pre-commit rejection preserves state; post-submission cases must
  explicitly establish whether state remains usable, needs a new prefill, or
  needs a recreated owner. Check retained results, reset, and close without
  promising universal rollback or automatic replay.
- [ ] `P1` Extend failure injection for both Runtime requests and BatchQueue:
  admission, LATEST replacement, completion, cancellation, publication, and
  drain. Use bounded adversarial traces to check ordering, fairness, released
  reservations, and isolation from other requests.
- [ ] `P1` Stress paged KV with allocation failure, prefix churn, copy-on-write,
  cancellation, eviction, fragmentation, and page reuse. Check lane/page
  generations and payloads alongside counters; distinguish reserved/live pages
  from the allocated pool and prefix-snapshot bytes.
- [ ] `P1` Measure and remove duplicate immutable native GPU allocations between
  contexts of one CompiledModel. Cover N contexts and closing/reopening contexts
  while the compiled owner remains alive. CPU prepared weights already have a
  shared owner; record which GPU resources still need that ownership boundary
  and verify memory high-water and final release.
- [ ] `P1` Add admission accounting across routes sharing a GPU. Build on the
  existing request/result budgets and compiled resource estimates; count
  shared weights once and reserve owned activation, workspace, staging,
  readback, and KV allocations through completion. Report unobservable driver
  overhead separately and test rejection before mutation and final uncharging.
- [ ] `P1` Audit SDPA's context-local `g_kcache`/`g_vcache` against explicit
  decode/KV bookkeeping. Measure duplicate storage and reconcile allocation,
  reset, and retirement rules where they overlap. Preserve any storage required
  by the fused operator and prove fixed/row numerical parity; internal cache
  cleanup alone does not require a new public KV port.

## Performance decisions

- [ ] `P1` Establish current B=1 and continuous-batching baselines on identical
  workloads: latency, throughput, TTFT, token latency, queue delay, P50/P95/P99,
  fairness, padding, page utilization, and physical memory high-water. Set
  regression budgets before promoting an optimization.
- [ ] `P2` Optimize scheduling only for a measured bottleneck. Candidates include
  native completion-driven dispatch, additional sharing of immutable execution
  resources across decode contexts, fewer queue scans/temporary allocations,
  and measured batch/page-size selection. Identify the affected C owner and
  compare latency, memory, and fairness before selecting queues, heaps, slabs,
  or an autotuning policy.
- [ ] `P2` Profile the current EfficientDet CUDA/TensorRT gap with matched inputs,
  precision, and timing boundaries. Promote only numerically qualified tactics
  that improve the measured bottleneck.

## Packaging and maintenance

- [ ] `P1` Ship TypeScript declarations for the npm JavaScript entries from the
  generated schema and thin host types. Define the declaration layout and
  package export mappings under the release filename policy, then typecheck a
  consumer against the packed inference/full package.
- [ ] `P1` Package the generated C headers and existing shared/static libraries
  as an installable SDK. Define its layout under the release filename policy
  and verify a generated-dispatch client from a clean installation for both
  profiles. This is delivery work; the library build targets already exist.
- [ ] `P2` Add cleanup conveniences only if generated-client usage demonstrates
  a need beyond explicit release and host close. Verify partial construction
  and repeated cleanup while preserving C-owned handle lifetimes.
- [ ] `P2` Replace the pinned Deno patch with an upstream version after the same
  raw/product device-lifecycle regressions pass. Record the replacement version
  and provenance before removing the patch.

## Optional capabilities

Select these for a concrete application or model. Each requires its own
bounded contract and qualification; it is not evidence of an unfinished
TypeScript-to-C migration.

- [ ] `P2` Evaluate Synurang shared memory and queue examples for a measured
  native camera/sensor workload. Start with existing BufferView transport and
  measure copying before adding registered buffer pools. Define slot ownership,
  generation IDs, exactly-once processed/dropped returns, cancellation, and
  draining a pool independently of the model host in the proto. Reuse the
  existing Scheduler/BatchQueue policies. Browser shared memory and GPU zero-copy
  require separate support and measurements. This is deferred work, outside the
  call runtime migration; no current workload makes it urgent.

- [ ] `P2` Add automatic device-loss recovery for an application that needs
  service continuity. Specify which model/context resources are recreated and
  which never-submitted requests may be retried. Keep recovery policy in C and
  expose its state through the proto; submitted work must not be silently
  replayed. This extends the already-tested close/create-host lifecycle.
- [ ] `P2` Add device-resident result-to-input transfer for a demonstrated
  split-model workload. Specify owner/device identity, exact tensor bounds,
  lifetime, completion, and failure semantics in the proto. Existing in-context
  paged KV and DecodeGenerate do not require this general transfer API.
- [ ] `P2` Add mixed adapter revisions within one execution batch if a workload
  needs them. Define per-lane selection, cache identity, and transition rules;
  current SelectAdapter is context-wide and BatchGroup already separates
  adapter revisions. Existing LoRA/routed-adapter authoring remains distinct.
- [ ] `P2` Extend row kernels for efficient bucketed/chunked prefill when traces
  show redundant work. BatchQueue already schedules prompt chunks; specify the
  additional execution primitive and per-lane query/key lengths in C.
  Qualify mixed prefill/decode or flat packing separately.
- [ ] `P2` Add direct shader page-table addressing only when it improves on the
  current gather path for a named workload, with equivalent prefix/COW,
  eviction, and failure behavior.
- [ ] `P2` Extend C generation beyond the current fixed-count ArgMax feedback:
  seeded sampling, EOS/stop sequences, cancellation, and incremental results.
  Declare semantics in the proto and check resumptions and retained outputs;
  application async iteration should only wrap generated calls.
- [ ] `P2` Add true FP16/BF16 arithmetic for specified operators/backends.
  Separate storage conversion, accumulation precision, and optimizer storage,
  and qualify each numerical contract against portable references.
- [ ] `P2` Qualify concrete GQA/RoPE decoder models and add missing backend
  routes they require. RoPE already has a C kernel and shape contract; identify
  the missing head layouts, row behavior, or physical routes before adding code.
- [ ] `P2` Add packed low-bit execution for a specified model format. Separate
  INT4/W4A16 numerical semantics from floating-point low-bit storage; extend the
  existing bit/span helpers for packing, alignment, odd widths, and checked
  bounds. Keep the whole-byte `dtype_size` result zero for sub-byte dtypes and
  qualify portable references before device kernels.
- [ ] `P2` Add model-specific runtime/shader pruning through a future `--model`
  packaging feature when requested. Measure size and qualify the resulting
  operator closure; current packages contain complete inference or full profiles.

## Research

- [ ] `P3` Evaluate speculative decoding, starting with prompt lookup, against
  an exact target-model reference. Measure acceptance and total latency or
  bandwidth saved, including verification and state rollback.
- [ ] `P3` Evaluate bounded-context policies such as attention sinks/StreamingLLM
  with explicit cache eviction rules and a model-specific quality benchmark.
- [ ] `P3` Select a concrete use case before planning beam search, constrained
  decoding, preemption/WCET, structured sparsity, or Hessian-aware quantization.
  Define a reference and a promotion metric for each selected experiment.
