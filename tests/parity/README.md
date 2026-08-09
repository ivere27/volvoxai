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
GPU consensus is a future exact-integer qualification gate after the native
routes are qualified and their strict lifecycle evidence is sealed; it is not
a substitute for an independent oracle.

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
Every successfully executed native tier also seals the CLI's machine-readable
lifecycle evidence: strict policy, tier and operator route attestation, pinned
revisions, context/execution identity, and duplicate result reads across context
closure. Native capability probes instead seal the exact compile rejection and
its registry/policy authority.

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
make parity_portable
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

The dynamic-v1 release campaign requires physical WebGPU execution. It also
audits native Vulkan/OpenGL on the same hardware host, but those public routes
currently declare capability skips rather than numerical execution:

~~~bash
make parity_gpu_required
make parity_webgpu
make parity_webgpu_matrix
make parity_portable_webgpu
make parity_native_gpu
make parity_kvcache_webgpu
~~~

`parity_gpu_required` authors portable references and requires physical WebGPU
whole-model, L1/L2 operator/graph, portable-closure, and KV-cache execution. It
also runs policy-aware native Vulkan/OpenGL capability probes and seals their
results separately from executed parity in:

~~~text
tests/parity/out/gpu_required_summary.json
~~~

An accepted native capability skip is not parity and is not a missing-device
waiver. Native backend initialization must identify a non-software physical
adapter, and strict compilation must fail with the exact model-specific
`BACKEND_UNSUPPORTED` reason declared by policy. The sealed evidence is bound to
the package, source, campaign, generated kernel-registry authority, and
`exporterQualified=false`. A missing device, infrastructure failure, wrong
reason, missing evidence, or unexpected execution fails the campaign.

Native Vulkan/OpenGL L1/L2 execution remains a future/manual qualification
command. WebGPU/native consensus is currently diagnostic and becomes
qualification evidence only after it also seals strict native lifecycle
reports. Run them only after the public native routes have complete
bounded-domain and exporter qualification:

~~~bash
make parity_native_gpu_matrix
make parity_gpu_consensus
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

The focused portable closure campaign re-authors 26 cases for the 24 operators
not already exercised by a required-tier L1/L2 case or a shipped L3 model. The
combined case inventory covers all 66 operators in the generated portable
inventory, but six operators are L3-only: Embedding, MaxPool2D, QAdd,
RequantizeLinear, Reshape, and ResizeNearest2D. Therefore `parity_portable`
verifies only the focused CPU/WASM/native-CPU closure; it is not a 66-operator
four-provider result by itself.

A full portable-inventory claim requires current whole-model CPU/WASM/native-CPU
manifests, current physical whole-model WebGPU artifacts, and the physical
closure artifacts. Run the campaigns in that order:

~~~bash
make parity
DRI_PRIME=1 DENO_WEBGPU_BACKEND=vulkan make parity_webgpu
DRI_PRIME=1 DENO_WEBGPU_BACKEND=vulkan make parity_portable_webgpu
~~~

The final closure comparison rejects stale L3 evidence, provider fallback, and
software WebGPU adapters when the physical closure campaign is present.

## Training parity

Training cases create a retained Trainer from an immutable logical snapshot:

~~~javascript
const trainer = await VolvoxAI.createTrainer(sourceSnapshot, {
  backend: 'cpu',
});

const step = await trainer.trainStep(options);
console.log(step.updatedTensorNames);
const successorSnapshot = await trainer.commit();
await trainer.close();
~~~

The suite compares losses, copied gradients, private updated weights, optimizer
effects, and the returned immutable successor revision. Step metadata must
remain stable after later updates. `trainStep()` never mutates its source;
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
finite global statistics. Once native routes and lifecycle evidence are
qualified, true-integer GPU consensus cases byte-compare the complete outputs
at zero tolerance.

Task gates check exact decisions such as next-token ID or top detection in
addition to tensor closeness.

## CI

| Tier | Trigger | Role |
| --- | --- | --- |
| CPU, WASM, required native CPU | every CI revision | hard portable gate |
| physical WebGPU execution | trusted protected revision/nightly | hard dynamic-v1 hardware gate |
| native Vulkan/OpenGL capability probes | trusted protected revision/nightly | exact registry/policy-backed audit; skips are not parity |
| qualified native Vulkan/OpenGL execution | manual until qualification | future L1/L2 gate; consensus remains diagnostic until lifecycle evidence is sealed |
| individual GPU targets | manual | development |
| WebNN hardware | not configured | open work |

Persistent hardware runners execute trusted repository revisions only.
Repository administrators must also configure branch/deployment protection;
workflow files cannot enforce those settings.

Open coverage work is tracked in [TODO.md](TODO.md).
