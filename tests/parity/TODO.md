# Cross-provider parity checklist

This file tracks only open parity work. Completed implementation notes belong
in tests and current reports, not in this checklist.

All runtime producers must use the public lifecycle:

~~~text
Runtime → Model → CompiledModel → ExecutionContext → ExecutionResult
~~~

Native producers use the matching VxRuntime → VxModel → VxCompiledModel →
VxExecutionContext → VxResult handles. Training producers use one retained
Trainer created from a Model.

## Operator and graph coverage

- [ ] Add isolated GatherElements and TopK cases. Fixtures must include real
      integer indices and every declared output.
- [ ] Add NonMaxSuppression, RoPE, and CrossAttention.
- [ ] Add Conv1D, ConvTranspose2D, Resize, and Interpolate1D.
- [ ] Add MoELinear, MoERouter, and SSMScan.
- [ ] Add isolated QConv2D, QAdd, and RequantizeLinear cases.
- [ ] Give every added case an independent reference implementation and
      explicit tolerances.

Acceptance: operation_list.md and the generated coverage report agree, and
every supported provider either passes each case or records an authored
unsupported result.

## Training parity

Training cases must create an immutable bounded snapshot, then call:

~~~javascript
const trainer = await VolvoxAI.createTrainer(sourceSnapshot, {
  backend: 'cpu', // or 'wasm' / 'webgpu'
});
~~~

- [ ] Add a multi-step AdamW case that verifies first/second moments and
      per-parameter step counts against an independent oracle.
- [ ] Verify stable step metadata: updatedTensorNames and copied gradients must
      remain unchanged after another step.
- [ ] Verify every step remains private, including completed, accumulation-only,
      and failed steps.
- [ ] Verify `commit()` returns exactly one immutable successor revision and
      `rollback()` restores the last committed baseline without publication.
- [ ] Compile before and after a commit and prove each CompiledModel remains
      pinned to its own revision.
- [ ] Add native full-command numerical coverage without introducing a public
      native training symbol.
- [ ] Add checkpoint/resume equivalence once the retained Trainer checkpoint
      surface is covered end to end.

Acceptance: CPU, strict WASM, and physical WebGPU match the reference loss,
gradients, updated weights, and optimizer state within documented tolerances.

## Decode parity

Use one ExecutionContext per decode stream:

~~~javascript
await context.decode.reset();
let result = await context.decode.seed(promptInputs);
result = await context.decode.step(nextInputs, { position });
~~~

- [ ] Compare stable declared outputs for greedy prefill and every decode step
      on CPU, WASM, and physical WebGPU.
- [ ] Compare retained K/V state with a full recomputation using only declared
      diagnostic outputs.
- [ ] Add numerical coverage for any device-feedback decode route through the
      same context-owned decode contract.
- [ ] Run two contexts concurrently and prove their K/V state and outputs do
      not alias.
- [ ] Close a context while work is queued and verify accepted work drains,
      new work is rejected, and earlier results remain readable.

Acceptance: token sequences and retained cache values match the reference, and
context isolation survives interleaved execution.

## Model-level coverage

- [ ] Add a TinyReceiptVQA F32 and W8A8 package with representative named
      inputs and an independent reference.
- [ ] Add tokenizer parity against the pinned vocabulary/merge rules.
- [ ] Add several deterministic random and real-input fixtures per model.
- [ ] Add first-diverging-node diagnostic capture for a failed whole-model
      comparison.
- [ ] Add WebNN numerical parity on a runner that reports a real physical
      device.
- [ ] Add an F16 package or remove unused F16 tolerance policy.

Acceptance: model-level reports identify exact package bytes, preprocessing,
provider/device, revision, route, and per-output comparison.

## Physical GPU campaign

- [x] Run the final-source physical WebGPU whole-model, L1/L2, portable-closure,
      and KV-cache campaign and seal every required execution result.
- [ ] Byte-compare true-integer WebGPU outputs where exact arithmetic is
      required.
- [ ] Run the physical WebGPU training subset.
- [x] Seal native Vulkan/OpenGL whole-model capability probes separately from
      executed parity, with exact policy, registry, adapter, package, source,
      campaign, and compile-rejection evidence.

The persistent GPU runner must execute trusted repository revisions only. A
required execution job never accepts a software adapter or CPU route. A native
capability skip is accepted only when the generated registry still declares
the route unqualified and strict compilation returns the exact declared
`BACKEND_UNSUPPORTED` reason; it never counts as numerical parity.

## Native GPU execution qualification

- [ ] Complete public bounded-domain proof for native Vulkan and OpenGL.
- [ ] Qualify their exporter routes in the authoritative kernel registry and
      regenerate its projections.
- [ ] Replace capability skips with required whole-model execution when the
      registry and runtime both attest support.
- [ ] Run native Vulkan/OpenGL L1/L2 matrices and harden EfficientDet GPU
      consensus to seal strict no-fallback lifecycle evidence.

`parity_native_gpu_matrix` remains a future/manual qualification command.
`parity_gpu_consensus` is a future/manual diagnostic until these criteria and
its lifecycle-evidence hardening are satisfied; both are intentionally outside
the dynamic-v1 required execution campaign.

## Commands

~~~bash
make parity
make parity_ops
make parity_graphs
make parity_backward
make parity_decode
make parity_kvcache
make parity_webgpu_matrix
make parity_portable_webgpu
make parity_native_gpu
make parity_native_gpu_matrix
make parity_gpu_consensus
make parity_gpu_required
~~~
