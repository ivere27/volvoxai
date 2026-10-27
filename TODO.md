# TODO

This file tracks additional validation, implementation gaps, and optional
capabilities. The public contract is [volvoxai.proto](proto/volvoxai.proto);
current ownership and execution behavior are in [ARCHITECTURE.md](ARCHITECTURE.md).
Remove completed items. Historical measurements identify the artifacts they
tested; reproduce an old failure through the current API before treating it as
a current defect.

Release work targets broad open-source adoption through easy installation,
reproducible workflows, and clearly documented support.

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

## Release readiness and adoption

- [ ] `P1` Publish a qualified model/platform matrix for representative
  workloads. Name source models, exported packages, profiles, backends,
  devices/browsers, dtypes, bounded shapes, and known limitations. Bind each
  supported combination to current artifact hashes and reproducible evidence;
  distinguish registered, build-tested, physically tested, and supported routes.
  Operator vocabulary and shader availability alone do not establish support.
- [ ] `P1` Qualify conversion of a representative model corpus from ONNX
  and TFLite. Record conversion success and typed refusal reasons with source
  nodes; verify input/output names, dtypes, layouts, preprocessing, and complete
  outputs against the original runtime. Document unsupported domains, control
  flow, operator attributes, and specialization options rather than promising
  arbitrary ONNX/TFLite compatibility.
- [ ] `P1` Define release compatibility and support policy for application
  clients, model packages, and trainer checkpoints. Publish versioned change
  notes, migration guidance, supported environments, and known failure/recovery
  behavior. Verify clean installation and upgrade workflows against the actual
  distributed artifacts before extending a support claim.

## Current release qualification

- [ ] `P0` Fix EOS stopping in cached WebGPU generation. The reproduced case
  selects the configured EOS as its first generated ID but returns additional
  IDs when the requested chunk budget is larger than one; it must return that
  single ID and report the stop. Trace device-selected feedback and the C stop
  observer, and qualify asynchronous selected-ID readback rather than relying
  on stale host output. Cover EOS and sequence stops across Generate calls,
  retained results, and cancellation. Reject unsupported combinations before
  execution until their stop policy is implemented; keep the passing
  full-prefix path as a separate regression case.
- [ ] `P1` Reproduce TinyReceipt FP32/INT8 whole-model comparisons through the
  current generated API and current packages. Compare every declared encoder,
  prefill, and step output, including logits and explicit KV, on native CPU,
  WASM CPU, and physical WebGPU against an independent ONNX Runtime reference.
  Bind reports to exact inputs, packages, artifacts, and per-tensor tolerances;
  investigate any remaining mismatch before adjusting a tolerance. Report task
  accuracy, extracted-answer agreement, and whole-sequence agreement separately;
  high task accuracy does not establish logits/KV or token-sequence equivalence.
- [ ] `P1` Extend executable independent numerical oracles for the operators,
  dtypes, layouts, and shape boundaries used by the representative model corpus.
  The oracle strategy catalogue is broader than the executable independent ORT
  cases; report coverage separately from CPU/GPU parity and canonical-vector
  tests. Keep unsupported cases explicit and gate the selected whole models.
- [ ] `P1` Qualify WebGPU through the packaged runtime in the target browsers
  and physical devices, including simultaneous jobs and device lifecycle.
  Record browser/driver versions, artifact hashes, dynamic inputs, retained
  outputs, and failure behavior. Patched Deno measurements establish their own
  host's behavior and do not qualify execution in Chrome, Edge, Firefox, or
  Safari.
- [ ] `P1` Add an independent multi-step AdamW oracle for native full, WASM CPU,
  and WebGPU. Check weights, moments, bias correction, weight decay, clipping,
  and accumulation. Extend checkpoint/resume and commit/revision-pinning
  coverage to native full; CPU/GPU agreement and checkpoint self-consistency
  alone cannot establish optimizer arithmetic.
- [ ] `P1` Qualify the C tokenizer with pinned GPT-2/Neo vocabulary and merge
  assets and independent expected token IDs/text. Include Unicode, partial
  UTF-8, byte boundaries, and token limits. The existing old-TypeScript
  comparisons establish migration parity, not this separate model-asset gate.
- [ ] `P1` Qualify native Metal row execution on a physical Mac and extend
  Vulkan/OpenGL qualification to additional physical adapters through generated
  calls. Record isolated operators, mixed graphs, whole models, dynamic bindings,
  context isolation, retained outputs, and failure handling for each claimed
  backend/domain. Preserve the existing selected RTX 3090 model evidence;
  shader translation or a build alone does not qualify Metal execution.
- [ ] `P1` Extend serialized native/WASM contract corpora with reproducible
  property-generated boundary and refusal cases for shape-domain proof,
  independent batching, and row arithmetic. Record seeds and expected
  outcomes; both transports must consume the same cases.
- [ ] `P1` Qualify a selected physical ARM64/robot deployment with real
  camera/microphone/IMU inputs and sustained offline model execution. Keep
  source-clock conversion, synchronization, freshness, backpressure/drop policy,
  and actuator control in the caller. Use a fixed accepted input batch per
  generation task and separate task state when new observations arrive. Verify
  Buffer/Tensor handoff, write completion, access leases, cancellation/deadline
  races, drain, bounded backlog, and host/device memory over sustained operation.
  Existing x86 CPU synthetic ALL/LATEST sensor replay at 50 Hz for 60 seconds
  is a baseline, not physical sensor or ARM64 qualification. Add engine features
  only when the selected workload cannot use the existing tensor/state contracts.

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
- [ ] `P1` Run long-lived application lifecycle and memory qualification on
  supported targets. Cover repeated load/compile/context creation, concurrent
  sessions, retained results and DLPack leases, cancellation, allocation failure,
  close/reopen, and eventual resource release. Measure host/device high-water
  and post-cleanup storage; keep process/driver retention separate from live
  engine allocations and establish regression limits for the selected workloads.
- [ ] `P1` Establish complete simultaneous physical-memory peaks for selected
  LLM workloads: source/private snapshots, prepared/shared weights, activations,
  KV pools and resident pages, scratch/staging, logits, and independent outputs.
  Include BF16-to-F32 exporter transients and load/compile/first-call phases.
  Set workload-specific host/device/WASM limits before a matched before/after
  comparison. Sampled RSS/NVML, visited allocators, partial owner inventories,
  or known-owner cleanup do not establish an exact whole-process/device peak;
  retain unobservable components and timing gaps explicitly.
- [ ] `P1` Document and qualify package loading limits and peak memory for the
  selected models. The default browser/inline ModelPackage source budget is
  64 MiB; explicit ModelSourceOptions already provide ordered length/SHA-256
  checks and bounded streaming up to 1 GiB within the existing WASM32 allocation
  ABI. Native file-path loading is a separate route. Extend qualification to
  selected larger models and target browsers/devices: cumulative source bounds,
  prepared host/device storage, oversized refusal, cancellation, retry/close
  cleanup, and load/compile/execution high-water. Source-transfer capacity is
  not an executable model-size or whole-memory guarantee. Report the combined
  memory cost of split-model packages.
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

- [ ] `P2` Measure W3/W4 WASM SIMD128 internal costs independently: PackedLinear
  weight unpacking, activation quantization and group-scale accumulation, plus
  packed Conv3/Conv4 scratch traffic and tile reuse. Extend numerical and matched
  latency qualification to selected additional workloads with fixed weights,
  arithmetic, inputs, threads and output reads. Record browser/Node versions
  and preserve existing binary-size budgets.
- [ ] `P1` Compare PackedLinear G32/G64/G128/G256 quality, memory, and performance
  on selected real models quantized from the same floating-point source.
  Hold weight bits, fitting algorithm, calibration data, execution arithmetic,
  devices, and workloads fixed when isolating the effect of G. Measure held-out
  perplexity/task accuracy, packed weights and scales, package size, host/device
  peak memory, prefill, and sustained decode. Separate load/compile, input
  preparation, transfers, kernels, and whole-request timing; record actual
  dispatch variants and preserve a matched G128 baseline. Do not relabel an
  existing artifact or infer model quality from weight reconstruction error.
- [ ] `P2` Implement a small set of measured PackedLinear optimization kernels,
  starting from a concrete target such as Vulkan G64/W8A16 tiled execution.
  Reuse the existing kernel registry, shader catalogue, and build pipeline;
  add private variant selection whose kernel, BM/BN/BK, grid, device requirements,
  shared memory, and workspace also drive compile-time admission. Vulkan still uses the
  generic scalar shader plan and needs this extension before variants can be
  added with limited registration changes. Share unpack/rounding/scale helpers,
  retain the generic fallback, and preserve F16 operand-rounding order and
  complete DYNAMIC_I8 group dots independently of BK. Test forced selection,
  K/M/N tails, cached row ranges, unsupported conditions, and physical device
  limits; promote variants only after correctness and matched performance
  checks, including existing G128 regressions. Bound generated combinations;
  add autotuning only for an established workload need.
- [ ] `P1` Close the remaining F32 LLM request-latency gap with llama.cpp,
  prioritizing TTFT, single-row decode, and cold load/compile/first-call costs.
  Start from the [qualified adaptive split-K result](build/llm/cuda-parity-optimization-001/analysis-audit/AJ-final-report001.md)
  and retain all measured regressions. Keep precision, threads, active input
  rows, KV capacity, outputs, and timing boundaries matched; padded Execute
  prefill and active-row llama prefill are different workloads. Production CUDA
  uses the Driver API and authored kernels; cuBLAS is an external benchmark
  reference only. About 10% higher latency than cuBLAS on selected GEMM scopes
  remains a target requiring fresh matched operator measurements. Preserve strict
  F32/no-FMA and HF/logit/KV/ID/special-value gates, ownership/COW/replay behavior,
  simple bounded tactics, and actual native/browser binary-size budgets.
  Promote changes only after numerical validation and matched whole-request
  comparisons, with initialization costs reported separately from warm results.
- [ ] `P1` Qualify adaptive split-K selection on additional smaller/larger GPUs
  and workloads. SM count and actual kernel occupancy provide an initial guess;
  the bounded benchmark compares tiled GEMM and admitted `split_k_slices` values.
  RTX 3090 is the only physically measured GPU for this policy. Check cache-key
  isolation, fixed replay choices, workspace limits, concurrency, and fallback
  behavior across shapes and contexts. Measure startup/compile/first-call costs
  and the process-wide 50 ms soft budget, including an already-started batch
  overrunning that budget. Reduce startup cost without inferring unprofiled
  model choices from a separate profile or adding a large GPU/tactic table.
- [ ] `P1` Reproduce end-to-end competitor comparisons with the current release
  and selected models on the same devices and browsers. Compare relevant ORT,
  LiteRT, or mobile inference routes with matched inputs, precision, threads,
  output reads, and generation lengths. Record download size including WASM
  and weights, load/compile/first-result time, warm P50/P95/P99, task quality,
  memory, and mobile power/thermal behavior when claimed. Separate kernel,
  transfer, and complete-request timing; INT8 storage does not guarantee speed.
  Historical tables and different measurement hosts are not current superiority
  evidence.
- [ ] `P1` Document and qualify browser Worker deployment through generated
  calls. Measure UI/event-loop delay, input transfer costs, and cancellation
  latency for the selected WASM CPU workloads. Promise-based transport and
  cooperative polling do not preempt an individual C kernel; do not imply that
  RPC timeout rolls back accepted numerical work.
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

- [ ] `P1` Correct all guide and textbook imports of `volvoxai/full`: the current
  full package entry is `volvoxai`, and the undeclared subpath reproduces
  `ERR_PACKAGE_PATH_NOT_EXPORTED`. Run documented inference, WebGPU, training,
  and PTQ workflows against an installed npm tarball in a clean Node/browser
  consumer, without repository-only imports or aliases. Preserve the fixed
  release filenames and make examples agree with actual package exports.
- [ ] `P1` Correct the architecture's library-wide single-CUDA-stream
  description: current CUDA engines own their submission streams. Distinguish
  per-context streams from measured execution overlap and synchronous result
  publication, and verify the implementation when updating these explanations.
- [ ] `P1` Ship TypeScript declarations for the npm JavaScript entries from the
  generated schema and thin host types. Define the declaration layout and
  package export mappings under the release filename policy, then typecheck a
  consumer against the packed inference/full package.
- [ ] `P1` Package the generated C headers and existing shared/static libraries
  as an installable SDK. Define its layout under the release filename policy
  and verify a generated-dispatch client from a clean installation for both
  profiles. This is delivery work; the library build targets already exist.
- [ ] `P1` Extend exact distributed Python wheel qualification beyond the tested
  CPU and RTX 3090 CUDA PTQ/QAT workloads. Cover the remaining documented GPU
  backend/device combinations and record wheel/library hashes and profiles.
  Compiled-in backend availability and older GPU test reports do not qualify a
  new wheel. Keep the current CPU qualification and Linux x86_64/glibc platform
  boundary explicit until the corresponding installation and execution gates
  pass.
- [ ] `P1` Include the root LICENSE and applicable third-party notices in runtime
  ZIP distributions; the current archive inventory contains four runtime files
  and a manifest without standalone notices. Verify notice retention for each
  delivery format and document model/data distribution terms separately from
  the engine license, without changing the fixed runtime filenames.
- [ ] `P2` Add cleanup conveniences only if generated-client usage demonstrates
  a need beyond explicit release and host close. Verify partial construction
  and repeated cleanup while preserving C-owned handle lifetimes.
- [ ] `P2` Replace the pinned Deno patch with an upstream version after the same
  raw/runtime device-lifecycle regressions pass. Record the replacement version
  and provenance before removing the patch.

## Optional capabilities

Select these for a concrete application or model. Each requires its own
bounded contract and qualification; it is not evidence of an unfinished
TypeScript-to-C migration.

- [ ] `P2` Evaluate a training-free GPU inference composition for a concrete
  application deployment. In the current standard release inventory, lite
  excludes GPUs and GPU inference requires full; measure download, dependency,
  and memory costs before choosing a composition change. Preserve the fixed release
  inventory and both profile boundaries, or obtain an explicit release-policy
  change before adding names.
- [ ] `P2` Extend public training losses for a demonstrated user task.
  TrainStep currently accepts CrossEntropyLoss; regression, detection,
  contrastive, or custom-gradient workflows need explicit new contracts rather
  than a generic-training claim. Declare selected losses in the proto and
  qualify forward loss, independent derivatives, optimizer steps, checkpoint
  resumption, and backend admission for their supported domains.
- [ ] `P2` Add device-specific NPU/DSP integration only for a selected deployment
  where it improves measured latency, energy, or sustained performance. Current
  GPU backends do not provide QNN, NNAPI, Core ML/Apple Neural Engine, or WebNN
  execution. Define the operator/shape and buffer-lifetime contracts and qualify
  real devices; Metal GPU support is not Apple NPU support.
- [ ] `P2` Expand binary SDK/wheel targets for user-requested platforms beyond the
  current Linux x86_64 wheel. Package and clean-install-test each chosen Windows,
  macOS, Linux ARM64, or mobile target; source/build availability alone does not
  establish supported binary distribution or physical execution.
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
- [ ] `P2` Add CUDA public prefix sharing/copy-on-write and growing cached shapes
  for a selected workload. Current bounded incremental replay and CPU prefix
  support do not establish those CUDA capabilities. Define page references,
  retirement, shape/admission transitions, and replay invalidation; verify
  prefix isolation, retained outputs, allocation/submission failures, and
  numerical equivalence before extending the supported contract.
- [ ] `P2` Qualify a selected pretrained nontext autoregressive model with its
  independent processor/codebook decoder and reference outputs. Add a
  multi-codebook profile only if that model requires it, defining independent
  or sequential selection, delayed patterns, per-stream termination, and
  alignment. Existing VQA/Whisper workflows and synthetic single-codebook
  fixtures do not qualify pretrained audio/image/action generation. Keep media
  interpretation in the caller and reuse the common tensor/state API.
- [ ] `P2` Add true FP16/BF16 arithmetic for specified operators/backends.
  Separate storage conversion, accumulation precision, and optimizer storage,
  and qualify each numerical contract against portable references. Current F16
  weight storage is widened for F32 execution and does not establish FP16 speed
  or training-memory savings. `PackedLinear` separately defines W4A16/W8A16 operand
  rounding with F32 accumulation; it does not enable general BF16 arithmetic.
- [ ] `P2` Extend DeepSeek V4.1 beyond the F32 text/full-prefix path. Propagate
  compressed-row updates through index-key projection and rotary operators
  before enabling retained incremental caches; prove previous completed rows
  and shared source ownership. Add image encoder/routing and DSpark draft
  execution as separate qualified workflows. Validate released activation/KV
  quantization independently from checkpoint-to-F32 conversion.
- [ ] `P2` Add FP8/FP4 arithmetic and activation/KV quantization after the current
  Llama/Qwen/DeepSeek architecture implementation. DeepSeek E4M3 and packed E2M1
  checkpoint conversion with declared block scales (including UE8M0) now widens
  weights to F32; other storage formats, including E5M2, remain unsupported.
  Keep these separate capabilities: CPU conversion to F32 needs no recent GPU;
  native Tensor Core execution requires a supported architecture, format and
  toolchain. RTX 3090 has no native FP8/FP4 Tensor Core path. Qualify storage
  decoding independently from accumulation, numerical accuracy, memory and
  performance. Widened checkpoints execute as F32 and do not reproduce the
  released low-precision activation/cache path or establish its memory savings.
  See [NVIDIA's hardware/format matrix](https://nvidia.github.io/TransformerEngine/support_matrix.html).
- [ ] `P2` Evaluate Tensor Core or other CUDA tactics for a measured workload
  bottleneck. Packed A16 and A8 now have shared-tile half and integer Tensor Core
  prefill paths. Packed decode still repeats activation quantization across output
  tiles or scalar weight unpacking; measure bounded activation reuse and decode
  tiling against Q4/Q8 baselines without conflating their different arithmetic.
  Reuse the existing packed launch-plan records in early model admission,
  replacing its fixed 64-thread/two-output envelope. Preserve final selected
  block/shared-memory checks and qualify the complete bounded shape domain.
  CUDA attention up to 2,048 active keys now combines bounded score reuse with
  parallel denominator groups, preserving QK and value-sum order. Qualify any
  reassociation against whole-model packed logits: small attention differences
  can cross activation quantization boundaries. Longer rows retain streaming
  reductions; measure decode occupancy across models and GPU sizes.
  Ordinary F32 operators retain strict F32 arithmetic. Keep production execution on the
  Driver API and authored kernels, without cuBLAS/cuDNN. State
  arithmetic precision explicitly, qualify numerical differences, and demonstrate
  whole-request gains on matched models before promoting a tactic.
- [ ] `P2` Expand the qualified GQA/RoPE and modern-decoder corpus to selected
  additional Llama1/2 checkpoints, models, devices, and backend/profile routes.
  Existing SmolLM2/Qwen3/TinyLlama text workflows, selected Smol browser runs,
  VQA/Whisper browser workflows, and Qwen3-4B CUDA numerical/generation evidence
  establish their declared scopes. Test actual long prompts and sustained
  decode, larger models, and concurrent throughput where claimed; a capacity
  reservation/owner sweep using short inputs is not a long-context or concurrent
  throughput test. Pin tokenizer/chat-template/processor assets, precision,
  independent references, WebGPU head/buffer limits, task quality, whole memory,
  and physical devices. Add backend code only for demonstrated missing routes.
- [ ] `P2` Add CUDA PackedLinear inference for existing W2/W3 packages, starting
  with W2A8 and W3A8. Reuse the current group scales and dynamic-I8 activation
  contract; unpack weight fragments into I8 registers/shared tiles without
  materializing a full expanded weight matrix. Preserve dense W3 codes across
  I32 word boundaries, row tails, groups 32/64/128/256, and partial row ranges.
  Qualify independent integer dots, CPU/WASM parity, actual device dispatch,
  memory ownership and typed refusal before enabling these optional routes.
- [ ] `P2` Optimize CUDA W2/W3 inference after functional qualification. Measure
  weight unpacking, activation preparation and scale application separately;
  extend measured activation reuse and Tensor Core tiles to G32/G64 where useful.
  Compare W2/W3/W4/W8 from the same source with Conv precision held fixed. Record
  cold/warm latency, VQA encoder/prefill/token/generation times, package size,
  device memory and held-out quality. Smaller weight storage alone does not
  establish faster computation; keep model-specific choices optional.
- [ ] `P2` Add CUDA packed Conv2/3/4 inference with static I8/U8 activations.
  Reuse the existing low-bit Conv package and affine contracts; implement
  device unpacking, grouped geometry, bias/zero-point correction and checked
  accumulation bounds. Qualify mixed Conv/Linear models against independent
  numerical references and measure conversion/workspace costs before promotion.
- [ ] `P2` Implement optional CUDA `DYNAMIC_I4` execution, starting with Linear
  W4A4 on RTX 3090. Define activation scale granularity, rounding, clipping,
  zero/non-finite handling and group accumulation before enabling the reserved
  selection. Separate a numerical reference from an accelerated INT4 Tensor
  Core implementation; an I8 reference must not be reported as native I4 speed.
  Add packed activation preparation, bounded tiles and architecture admission
  through the existing Driver API/kernel registry. Compare actual throughput,
  memory and held-out accuracy with A8, including small/decode shapes. Evaluate
  activation-aware calibration or activation QAT separately if quality requires
  it; existing weight-only QAT does not establish activation-QAT support.
- [ ] `P2` Extend static W4A8 PTQ and inference to Linear on native CPU/WASM.
  Extend the unified `quantize` / `QuantizationConfig` workflow for Linear
  weights with calibrated static INT8 activations.
  Define supported weight granularity, scales, zero points, accumulation and
  requantization; implement WASM CPU scalar/SIMD execution. Keep calibration
  in the full profile and inference free of training dependencies. Verify
  packing boundaries, independent numerical accuracy and native/WASM parity;
  measure model quality, package size, peak memory and latency against static
  W8A8. Keep static W4A8 Linear explicitly unsupported until these paths are ready.
- [ ] `P2` Qualify external AWQ, GPTQ, and GGUF checkpoint adapters with explicit
  storage conversion and independent numerical references. Native PackedLinear
  support alone does not establish compatibility with their formats. Compare
  additional fitting methods with the existing RTN/MSE_CLIP/IMPORTANCE model
  authoring before selecting a weight-correction algorithm; measure held-out
  teacher-forced NLL/KL, long prompts, and sustained decode. Reproduce the
  historical 31-token TinyLlama W4A8 CPU/CUDA strict-reference failure through
  the current API and artifacts, distinguishing upstream floating-point changes
  that cross A8 rounding boundaries from implementation errors. Passing small
  HF fixtures or identical-input operator checks does not close that whole-model
  case. Extend qualification to long context and CUDA cached decode; quality
  scores cannot clear an unresolved implementation correctness failure.
  Keep INT4 separate from floating-point FP4 and retain checked packing bounds.
- [ ] `P2` Qualify packed linear execution on physical Apple Silicon CPU/Metal
  devices. NEON compilation and Metal shader translation do not establish
  M-series numerical behavior, memory use or speed. Benchmark the same pinned
  checkpoint and activation precision against its independent reference.
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
