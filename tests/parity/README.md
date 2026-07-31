# Cross-provider parity harness

The parity suite checks that one canonical package produces equivalent named
outputs across providers and against independent references.

JavaScript producers use:

~~~text
Runtime → Model → CompiledModel → ExecutionContext → ExecutionResult
~~~

Native producers use:

~~~text
VxRuntime → VxModel → VxCompiledModel → VxExecutionContext → VxResult
~~~

Every tier-specific producer requires its provider during compilation. It
cannot publish a result under another provider's label.

## Comparison model

The suite has two references:

- The portable CPU implementation is the internal consistency reference.
- ONNX Runtime or pinned PyTorch execution is the independent correctness
  reference where a case provides one.

Accelerated tiers are compared with both references when available. Pairwise
GPU consensus is an additional exact-integer gate, not a substitute for an
independent oracle.

Each artifact records:

- package, fixture, policy, build, and relevant-source fingerprints;
- campaign and job identity;
- selected provider and provider-reported device identity when available;
- compile route evidence and model revision;
- exact output name, dtype, logical shape, and element count;
- finite statistics, top-k structure, and deterministic sample indices;
- byte size and SHA-256.

A missing required job, stale fingerprint, unexpected skip, provider mismatch,
software adapter, malformed tensor, or modified artifact fails closed.
Every native execution tier also seals the CLI's machine-readable lifecycle
evidence: strict policy, tier and operator route attestation, pinned revisions,
context/execution identity, and duplicate result reads across context closure.

## Layout

~~~text
policy.json          model cases, outputs, tolerances, and required tiers
lib/tensorio.mjs     typed raw tensor I/O and deterministic inputs
lib/extract.mjs      output signatures and strict comparison
lib/artifact.mjs     fingerprints, atomic artifacts, and manifests
lib/backend.mjs      physical-provider identity checks
lib/runmodel.mjs     Runtime/Model/CompiledModel/Context execution
run.mjs              L1, L2, and L3 orchestration
produce_native.sh    opaque native lifecycle through the command runner
goldens/             reviewed portable-reference signatures
out/manifests/       transient finalized manifests
out/                 transient fixtures, artifacts, and reports
~~~

## Portable commands

~~~bash
npm run build:all
make build_wasm
make build_native

make parity
make parity_ops
make parity_graphs
make parity_backward
make parity_decode
make parity_kvcache
make parity_coverage
make parity_image
~~~

make parity_goldens intentionally regenerates committed portable-reference
signatures. Run it only for an intended numerical or shape change and review
both signatures and their manifest.

The L3 comparator writes tests/parity/out/report.md and exits nonzero for a
missing, mislabeled, stale, structurally invalid, or numerically different
required result.

## Physical GPU commands

Run these only on a host with the named hardware:

~~~bash
make parity_gpu_required
make parity_webgpu
make parity_webgpu_matrix
make parity_native_gpu
make parity_native_gpu_matrix
make parity_gpu_consensus
make parity_kvcache_webgpu
~~~

parity_gpu_required authors portable references, runs required WebGPU,
Vulkan, and OpenGL jobs, verifies exact detector consensus and decode cache
state, then seals one summary:

~~~text
tests/parity/out/gpu_required_summary.json
~~~

Require a device-name substring with:

~~~bash
VOLVOXAI_PARITY_GPU_ADAPTER='RTX 3090' make parity_gpu_required
~~~

Known software adapters such as SwiftShader, llvmpipe, and lavapipe are
rejected for physical-hardware claims.

On a dual-GPU Linux host, choose the discrete adapter for Deno WebGPU as
needed:

~~~bash
DRI_PRIME=1 DENO_WEBGPU_BACKEND=vulkan make parity_kvcache_webgpu
~~~

## Training parity

Training cases create a retained Trainer from a Model:

~~~javascript
const trainer = await VolvoxAI.createTrainer(model, {
  backend: 'cpu',
});

const step = await trainer.trainStep(options);
console.log(step.updatedTensorNames);
await trainer.commit();
await trainer.close();
~~~

The suite compares losses, copied gradients, private updated weights, optimizer
effects, and the explicitly committed Model revision. Step metadata must remain
stable after later updates. `trainStep()` never publishes implicitly;
`rollback()` restores the last committed baseline.

Run physical WebGPU backward after authoring its CPU/WASM campaign:

~~~bash
node tests/parity/backward/run_backward.mjs

DRI_PRIME=1 DENO_WEBGPU_BACKEND=vulkan deno run --unstable-webgpu \
  --allow-read --allow-write --allow-env --allow-ffi \
  tests/parity/backward/run_backward.mjs webgpu

python3 tests/parity/backward/backward_torch_oracle.py
node tests/parity/backward/run_backward.mjs compare
~~~

## Decode parity

Decode cases use context-owned state:

~~~javascript
await context.decode.reset();
let result = await context.decode.seed(promptInputs);
const seedOutput = await result.output('token_ids').read();
await result.close();

result = await context.decode.step(nextInputs, { position });
const stepOutput = await result.output('token_ids').read();
await result.close();
~~~

The tests compare greedy token sequences and, where the Graph declares them,
diagnostic K/V outputs with a complete recomputation. Earlier results must
remain readable after later context work.

Run WebGPU decode:

~~~bash
DRI_PRIME=1 DENO_WEBGPU_BACKEND=vulkan deno run --unstable-webgpu \
  --allow-read --allow-write --allow-env --allow-ffi \
  tests/parity/decode/decode_parity.mjs webgpu

python3 tests/parity/decode/decode_torch_oracle.py
node tests/parity/decode/decode_parity.mjs compare
~~~

## Tolerances

Structure is exact: output name, dtype, shape, element count, sampling axes and
indices, and top-k layout must match.

Numerical samples use per-dtype rtol/atol from policy.json. F32 also gates
finite global statistics. True-integer GPU consensus cases byte-compare the
complete outputs at zero tolerance.

Task gates check exact decisions such as next-token ID or top detection in
addition to tensor closeness.

## CI

| Tier | Trigger | Role |
| --- | --- | --- |
| CPU, WASM, required native CPU | every CI revision | hard portable gate |
| WebGPU, Vulkan, OpenGL hardware | trusted protected revision/nightly | hard hardware campaign |
| individual GPU targets | manual | development |
| WebNN hardware | not configured | open work |

Persistent hardware runners execute trusted repository revisions only.
Repository administrators must also configure branch/deployment protection;
workflow files cannot enforce those settings.

Open coverage work is tracked in [TODO.md](TODO.md).
