# Roadmap

This file contains open implementation work only. It is ordered by dependency
and correctness leverage, not by release date.

Every item must preserve the public lifecycle:

~~~text
Runtime → Model → CompiledModel → ExecutionContext → ExecutionResult
~~~

Training remains a separate retained Trainer owner. Native integrations use
the matching opaque Vx* handles.

## P0 — Correctness and reproducibility

- [ ] Add WebNN hardware parity to the whole-model and operation matrices.
- [ ] Add tokenizer parity against the pinned GPT-2/Neo vocabulary and merge
      rules.
- [ ] Remove the parallel WGSL generation race with one shared generated-asset
      target or stamp.
- [ ] Require provider/device/route evidence, package fingerprints, stable
      named outputs, and finalized manifests in every hardware campaign.
- [ ] Keep the physical WebGPU, Vulkan, and OpenGL L1/L2/L3 campaign
      fail-closed on trusted revisions.

## P1 — Application-owned generation

- [ ] Add a reusable example/downstream streaming helper over
      ExecutionContext.decode with greedy decoding, temperature, top-k, top-p,
      deterministic seeds, EOS/max-token termination, stop sequences, and
      cancellation.
- [ ] Define one streaming event/error contract for cancellation, partial
      output, cache reset, and EOS encountered during submitted device work.
- [ ] Apply the same application policy around the optional in-process
      Synurang FFI plugin while
      keeping its inference operations on opaque VxExecutionContext/VxResult
      handles.

Sampling, tokenization, and text policy do not belong in the core inference
entry.

## P2 — Shape specialization and decode scheduling

- [ ] Define bounded dynamic-shape support or a shape-specialization cache.
      Preserve an explicit fixed-shape path for small deterministic
      deployments.
- [ ] Generalize fixed-shape B=1 W8A8 row/KV execution to independently owned
      active sequence lengths and batches.
- [ ] Design paged/block KV storage, prefix sharing, eviction, and continuous
      batching inside each ExecutionContext.
- [ ] Add speculative draft/verify, beam search, and constrained decoding only
      on top of that context-owned cache contract.
- [ ] Measure memory, latency, and throughput against the dense fixed-context
      implementation before selecting a storage format.

## P3 — Device performance and model footprint

- [ ] Add group-quantized sub-8-bit weight formats and
      unpack-in-register kernels. Int8 remains the package floor until the
      Graph contract, portable reference, and parity suite are complete.
- [ ] Add real FP16/BF16 arithmetic paths where hardware supports them,
      including CUDA tensor-core tactics.
- [ ] Add fused attention tactics and lift F32 GPU head-dimension limits where
      hardware permits.
- [ ] Revise the QSDPA Graph contract, portable reference, tests, and native
      implementation before expanding quantized head dimensions.
- [ ] Add a canonical grouped-query-attention composition.
- [ ] Add a one-call SwiGLU block or fusion only when model-level benchmark
      evidence beats the composable SiLU + Mul + Linear form.
- [ ] Add device-side sampling primitives after P1 semantics are fixed.
- [ ] Close the EfficientDet CUDA/TensorRT gap through equivalent-workload
      profiling and end-to-end validated tactics.
- [ ] Add public device-buffer input only for a measured caller, with explicit
      ownership, synchronization, and lifetime rules.

## P4 — Provider and model coverage

- [ ] Add WebNN mappings for RMSNorm, CrossSDPA, CrossAttention, LogSoftmax,
      DequantizeLinear, and Conv1D.
- [ ] Add RoPE to WebGPU, WebNN, Vulkan, OpenGL, Metal, and CUDA F32.
- [ ] Complete canonical Gather and GatherElements coverage across native and
      GPU providers.
- [ ] Complete typed ArgMax and NonMaxSuppression coverage across providers.
- [ ] Extend Softmax and LogSoftmax beyond the last axis when a supported model
      requires it.
- [ ] Ship one reference RoPE/RMSNorm decoder-family exporter and canonical
      graph.json package.
- [ ] Add representative multi-output provider tests using declared
      ExecutionResult outputs only.

Every provider addition needs correctness tests for supported dtypes, shapes,
layouts, failure policy, and provider-reported device identity when available.

## P5 — Examples and documentation

- [ ] Complete the TinyReceiptVQA capstone with a representative calibration
      dataset and task-level accuracy report.
- [ ] Add a runnable edge/robot application that connects the native opaque
      lifecycle to sensors and actuators.
- [ ] Keep operation_list.md, provider reports, build profiles, and package
      documentation synchronized with implementation.

## Deferred

- General dataset, augmentation, experiment, and metrics platforms.
- Frontier-scale pretraining, RLHF/DPO, and higher-order autograd.
- Multi-GPU collectives, tensor parallelism, and pipeline parallelism.
- General Graph control flow such as If and Loop.
- Model-family ecosystems without a concrete package, test fixture, and
  application owner.

## Native validation

Software Vulkan loaders are useful only for loader smoke tests. Hardware
claims require a named physical adapter, strict compile policy, route evidence,
and numerical comparison on that device.

CUDA is opt-in. Its exact forward/training coverage, numeric modes, and
hardware validation commands are documented in [cuda.md](cuda.md).
