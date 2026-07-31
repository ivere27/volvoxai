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

- [ ] Add isolated Gather, GatherElements, Where, Split, TopK, and Cast cases.
      Fixtures must include real integer indices, both condition branches, and
      every declared output.
- [ ] Add NonMaxSuppression, RoPE, CrossSDPA, and CrossAttention.
- [ ] Add Conv1D, ConvTranspose2D, Resize, and Interpolate1D.
- [ ] Add MoELinear, MoERouter, and SSMScan.
- [ ] Add one isolated case for each shipping Q operation:
      QLinear, QGemm, QConv2D, QSDPA, QLayerNorm, QGroupNorm, QGELU, QSiLU,
      QEmbedding, QArgMax, QMaskedMean, QAdd, and RequantizeLinear.
- [ ] Give every added case an independent reference implementation and
      explicit tolerances.

Acceptance: operation_list.md and the generated coverage report agree, and
every supported provider either passes each case or records an authored
unsupported result.

## Training parity

Training cases must create a Model, then call:

~~~javascript
const trainer = await VolvoxAI.createTrainer(model, {
  backend: 'cpu', // or 'wasm' / 'webgpu'
});
~~~

- [ ] Add a multi-step AdamW case that verifies first/second moments and
      per-parameter step counts against an independent oracle.
- [ ] Verify stable step metadata: updatedTensorNames and copied gradients must
      remain unchanged after another step.
- [ ] Verify every step remains private, including completed, accumulation-only,
      and failed steps.
- [ ] Verify `commit()` publishes exactly one successor revision and
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

- [ ] Run WebGPU, native Vulkan, and native OpenGL L1/L2 matrices.
- [ ] Run TinyStories and EfficientDet F32/W8A8 model cases.
- [ ] Byte-compare true-integer GPU outputs where exact arithmetic is required.
- [ ] Run decode/KV-cache and WebGPU training subsets.
- [ ] Seal all required job results into one campaign summary.

The persistent GPU runner must execute trusted repository revisions only. A
required hardware job never accepts a software adapter or CPU route.

## Commands

~~~bash
make parity
make parity_ops
make parity_graphs
make parity_backward
make parity_decode
make parity_kvcache
make parity_webgpu_matrix
make parity_native_gpu_matrix
make parity_gpu_required
~~~
