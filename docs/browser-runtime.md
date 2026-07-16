# Browser and Node Runtime

VolvoxAI exposes the same forward API from readable and minified artifacts in
`dist/<package-version>/`:

- `volvoxai.js` and `volvoxai.min.js` contain inference only and resolve
  adjacent `volvoxai.wasm`.
- `volvoxai.full.js` and `volvoxai.full.min.js` contain inference plus training
  and resolve adjacent `volvoxai.full.wasm`.
- `volvoxai.wasm.js` and `volvoxai.wasm.min.js` contain inference plus strict
  WASM training and stateless PTQ, expose no other backend, and resolve adjacent
  `volvoxai.full.wasm`.

The inference sidecar is forward-only. The full sidecar retains its forward
exports and adds ABI version 1 C operators for cross-entropy, backward kernels,
gradient utilities, SGD, AdamW, PTQ observation, affine I8/U8 quantization,
per-axis I8 weight packing, and I32 bias packing. F32-master LoRA
synchronization is one convenience workflow over those reusable operators.
`backend: "wasm"` training is strict and never silently falls back to CPU.

## Backend Selection

`VolvoxAI.init(preferredBackend = "auto", wasmUrl)` accepts a backend string or
a strict ordered backend array. The inference facade resolves `volvoxai.wasm`
beside itself; the full facade resolves `volvoxai.full.wasm`. Callers may
override either sidecar with a URL or file path.

The WASM-only facade accepts only `"wasm"` (or `"auto"` as an alias for that
single choice), never registers another backend, and never falls back. Its
default sidecar is the full module because a LoRA-only optimizer update still
needs backward kernels for the graph between the loss and A/B tensors.
The published `./wasm` and `./wasm/min` package subpaths are browser-only: their
release bundles deliberately omit Node's filesystem loader so Manifest V3
workers contain no dynamic `import()`. In Node, use the standard `.` or `./full`
entry and select the WASM backend instead.

| Call | Behavior |
| --- | --- |
| `VolvoxAI.init()` or `VolvoxAI.init("auto")` | Try WebNN, then WebGPU, then WASM, then CPU. |
| `VolvoxAI.init("webnn")` | Prefer WebNN; fall back to WASM then CPU. |
| `VolvoxAI.init("webgpu")` | Prefer WebGPU; fall back to WASM then CPU. |
| `VolvoxAI.init("wasm")` | Skip WebNN/WebGPU and use WASM with CPU fallback. |
| `VolvoxAI.init("cpu")` | CPU only. |
| `VolvoxAI.init(["wasm", "webgpu"])` | Strict allow-list. No CPU fallback unless `"cpu"` is included. |

Array mode is strict. `init()` throws if none of the listed backends initializes,
and `compile()` throws if the initialized backends cannot compile the graph.

## WASM-only Manifest V3 extensions

The minimal training-capable package is one of the two WASM-only JavaScript
files, `volvoxai.full.wasm`, and the model. Do not also ship
`volvoxai.full.js`, the forward-only `volvoxai.wasm`, WGSL, or other backend
files unless the application uses them.

Chrome extension service workers accept static ES-module imports when the
background worker declares `"type": "module"`, but do not accept dynamic
`import()`. The release build removes the Node filesystem loader and every
dynamic import from both WASM-only JavaScript variants. All executable JS and
WASM must be packaged with the extension.

```json
{
  "manifest_version": 3,
  "name": "Local correction demo",
  "version": "1.0.0",
  "background": {
    "service_worker": "service-worker.js",
    "type": "module"
  },
  "content_security_policy": {
    "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self';"
  }
}
```

The default extension CSP does not enable WebAssembly, so
`'wasm-unsafe-eval'` is required. Initialize with explicit extension URLs; this
also makes the sidecar/model location independent of the importing worker:

```javascript
import { VolvoxAI } from './vendor/volvoxai.wasm.min.js';

const runtime = await VolvoxAI.init(
  'wasm',
  chrome.runtime.getURL('vendor/volvoxai.full.wasm'),
);
const graph = await runtime.loadGraph(
  chrome.runtime.getURL('model/model.safetensors'),
);
```

Extension-owned pages and workers may fetch their own packaged resources
without declaring them web-accessible. Add a narrowly scoped
`web_accessible_resources` entry only when a normal web page or another
extension must fetch the sidecar/model. Chrome's official references cover the
[extension CSP](https://developer.chrome.com/docs/extensions/reference/manifest/content-security-policy),
[module service workers](https://developer.chrome.com/docs/extensions/develop/concepts/service-workers/basics),
and [web-accessible resources](https://developer.chrome.com/docs/extensions/reference/manifest/web-accessible-resources).

### Generic C-backed PTQ

The WASM-only facade exposes the stateless PTQ math already shared with the
native full profile. The typed API hides C pointers, struct layouts, and i64
arguments:

```javascript
const ptq = await runtime.createPTQ();
try {
  const observer = ptq.createObserver();
  observer.observe(firstCalibrationBatch);
  observer.observe(secondCalibrationBatch);

  const parameters = ptq.deriveParameters(observer, {
    dtype: 'uint8',
    scheme: 'asymmetric',
  });
  const quantized = ptq.quantize(values, parameters);
  const packedWeight = ptq.packWeight(weights, [outputSize, inputSize], {
    axis: 0,
    name: 'projection.weight.i8',
  });
  const packedBias = ptq.packBias(bias, parameters.scale, packedWeight.scales);
} finally {
  ptq.dispose();
}
```

The toolkit supports symmetric or asymmetric per-tensor I8/U8 values,
symmetric per-axis I8 weights for ranks 1 through 8, and I32 accumulator
biases. It creates a private scratch instance of `volvoxai.full.wasm`, so its
arena resets cannot invalidate an allocated inference graph. Reuse one toolkit
and call its idempotent `dispose()` when finished; this drops the references
that keep the scratch instance and linear memory alive. It returns typed arrays
and descriptors but deliberately does not mutate graph nodes, read or write
files, or decide package layout; those are caller or JavaScript authoring
responsibilities.

### Correction-driven LoRA updates

The strict WASM trainer consumes generic cross-entropy targets; it does not own
prompt formatting or response tokenization. The application converts a user's
corrected answer into teacher-forced inputs, token targets, and an optional
loss mask, then calls `trainLoRAStep()` with only explicit LoRA factor names:

```javascript
const result = await runtime.trainLoRAStep(graph, {
  inputs,
  logitsTensor: 'logits',
  targets: correctedTokenIds,
  lossMask,
  trainableTensors: ['decoder.lora_a', 'decoder.lora_b'],
  updateMode: 'adamw',
  optimizer: { learningRate: 1e-4, maxGradNorm: 1 },
});
```

LoRA A/B must be initialized F32 graph weights connected through explicit
MatMul/scale/Add nodes. Omit every base weight from `trainableTensors` to keep
it frozen. When an update is applied, `WasmAutograd` recompiles the bound
inference engine so retained packed Linear/MatMul weights cannot hide the new
factor values from the next inference. A pending gradient-accumulation step
does not recompile because it has not changed a weight.

This is distinct from `AdapterManager`: staged adapters are immutable,
versioned inference/export snapshots rather than differentiable graph tensors.
Train explicit A/B, use a full checkpoint to resume optimizer state, and create
an adapter-only deployment snapshot separately. Do not activate that snapshot
on a graph that still contains the same explicit LoRA branch, or the delta is
applied twice.

#### F32-master LoRA with unchanged W8A8 inference

An existing W8A8 inference graph can stay physically quantized and keep the
same nodes, tensor identities, activation descriptors, and output shapes. It
must already contain its I8 LoRA A/B branch. Train a separate supported F32
graph, then bind each persistent F32 master to the corresponding inference
weight:

```javascript
const trainer = await runtime.createQuantizedLoRATrainer(
  f32TrainingGraph,
  w8InferenceGraph,
  {
    bindings: [
      { master: 'decoder.lora_a', target: 'decoder.lora_a.i8', transpose: true },
      { master: 'decoder.lora_b', target: 'decoder.lora_b.i8', transpose: true },
    ],
  },
);

try {
  // Only for a W8-only starting snapshot. Skip this after restoring an F32
  // training checkpoint, then call sync() once instead.
  await trainer.initializeMastersFromQuantized();

  const result = await trainer.trainStep({
    inputs: teacherForcedInputs,
    logitsTensor: 'logits',
    targets: correctedTokenIds,
    lossMask,
    trainableTensors: ['decoder.lora_a', 'decoder.lora_b'],
    updateMode: 'adamw',
    optimizer: { learningRate: 1e-4, maxGradNorm: 1 },
  });

  // trainStep() has already requantized applied updates and rebuilt the W8
  // heap bytes, per-axis scale metadata, and packed Q8 weights exactly once.
  const corrected = await trainer.engine.execute(nextQuantizedInputs);
} finally {
  await trainer.dispose();
}
```

`TrainingModelBuilder.loraLinear()` creates IN_OUT masters A=`[d_in,rank]`
and B=`[rank,d_out]`; canonical QLinear weights are OUT_IN, so both normal
bindings use `transpose: true`. Targets must be rank-2 I8 weights with frozen
symmetric axis-0 scales, zero points of zero, and initialized all-zero I32
biases. The default `scalePolicy: "recompute"` replaces weight scales while
preserving every activation scale. `"preserve"` retains existing weight scales
and reports how many values saturated.

Keep the F32 graph checkpoint, including AdamW state, as the authoritative
resume state in IndexedDB. Treat the W8 values as a derived inference snapshot.
Never dequantize that snapshot back into the master after training has begun,
or repeated quantization error will accumulate and optimizer state will no
longer match the weights. Calls on one trainer are serialized, and one live
quantized trainer reserves its WASM engine until `dispose()`.

The full sidecar also exposes generic stateless PTQ; what is specific here is
the trainer's graph-binding, atomic staging, cache refresh, and rollback policy.
This workflow is not QAT or a QLinear backward implementation. Strict WASM
training still rejects a deep QLinear/QGemm W8A8 training graph. The separate
graph must provide an F32 path supported by the training allowlist; automatic
W8A8-to-F32 graph conversion is not part of version 0.2.0.

### JavaScript backend contract

The inference entry exports `BackendEngine` and backend API version 1. The four
built-in peers expose the same lifecycle:

| Member | Contract |
| --- | --- |
| `backendApiVersion` | `VOLVOXAI_BACKEND_API_VERSION`, currently `1`. |
| `backendName` | Stable registered name such as `cpu`, `wasm`, `webgpu`, or `webnn`. |
| `capabilities` | Frozen `incrementalExecution`, `incrementalRows`, and `outputLocation` flags. |
| `allocateGraph(graph)` | Bind and allocate/compile one graph; resolves or returns the engine. |
| `execute(inputs, options)` | Run the bound graph. Existing backend-specific result values are preserved. |
| `createDecodeSession(options)` | Create the common seed/step/reset decode facade described below. |
| `resetDecodeCache()` | Invalidate retained decode state; required when `incrementalExecution` is true. |
| `decodeCacheGeneration` | Non-negative, monotonically increasing safe integer changed by every cache reset or direct cache-owner replacement; required when `incrementalExecution` is true. |

`WebGPUEngine` is the public WebGPU peer. It owns a lower-level `GraphExecutor`,
and forwards `gpuBuffers`, `readBuffer()`, `readBufferRange()`, the WebGPU-only
`executeDeviceFeedbackDecode()` fast path, and the existing device-result
behavior. `readBufferRange(buffer, byteOffset, sizeBytes, dtype)` copies only
the requested logical output range. `fork()` creates an unallocated WebGPU peer
on the same `GPUDevice` with independent graph and cache ownership, which lets
an application keep several compiled graphs resident. `GraphExecutor` remains
exported for low-level integrations and also implements decode sessions for
compatibility.

An out-of-tree browser device can register by name instead of modifying a core
enum:

```javascript
import { BackendEngine, VolvoxAI } from 'volvoxai';

class MyNPUBackend extends BackendEngine {
  constructor(device) {
    super('my-npu', { outputLocation: 'host' });
    this.device = device;
  }
  async allocateGraph(graph) { this.resetDecodeCache(); this.graph = graph; return this; }
  async execute(inputs, options = {}) { /* vendor dispatch */ }
}

const unregister = VolvoxAI.registerBackend(
  'my-npu',
  async () => new MyNPUBackend(await openMyDevice()),
);
const runtime = await VolvoxAI.init(['my-npu', 'cpu']);
```

Factories receive `{ name, runtime, wasmUrl }`, may return `null` when their
device is unavailable, and must implement backend API v1 (normally by extending
`BackendEngine`) with a `backendName` matching the registered name.
Structural implementations that advertise incremental execution must also
provide `resetDecodeCache()` and a monotonic numeric `decodeCacheGeneration`;
subclassing `BackendEngine` supplies this cache-ownership protocol. An
incremental subclass must call `this._beginDecodeExecution(options)` at the
start of every `execute()` before it reads or mutates retained intermediates.
That distinguishes a session-owned call from a direct legacy call that replaces
the active session's cache ownership; a structural implementation must provide
the equivalent generation advance itself.
Built-in names cannot be replaced. `registerBackend()` returns an unregister
function for tests or dynamically unloaded integrations.

### Decode sessions

`DecodeSession` presents the retained-intermediate and W8A8 row/KV-cache paths
identically across browser backends while leaving model-specific token loops in
examples or applications. Session operations are serialized per engine because
one compiled engine owns one retained cache; a newer seed makes older sessions
stale rather than allowing asynchronous device submissions to overlap:

```javascript
const decode = executor.createDecodeSession({
  changedInputs: ['tokens', 'attention_mask'],
  rowMode: 'auto', // 'required' and 'disabled' are also available
});

try {
  let output = await decode.seed(allInputs);
  output = await decode.step(nextInputs, { position: 1 });
} finally {
  await decode.close();
}
```

`seed()` always performs a complete first pass. `step()` reruns the dependency
closure of `changedInputs`; when the backend supports fixed-shape W8A8 rows and
`position` is supplied, it updates that decoder row and retains its causal K/V
prefix. `reset()` invalidates retained state but permits another seed, while
`close()` permanently closes the session. Calls are serialized, and a session
detects when another session replaces its cache. `mode` reports the strongest
negotiated path (`incremental-row`, `incremental-dependency`, or
`ordinary-forward`), and `lastExecutionMode` reports what the last call used.

| Backend | Session behavior | Output location |
| --- | --- | --- |
| CPU | Dependency cache plus B=1 fixed-shape W8A8 row/KV cache. | Host typed arrays. |
| WASM | Dependency cache plus B=1 fixed-shape W8A8 row/KV cache. | Host typed arrays copied from WASM memory. |
| WebGPU | Dependency cache plus B=1 fixed-shape W8A8 row/KV cache. Row pipelines are compiled lazily for the selected decoder closure. | Device `GPUBuffer`; use `readBuffer()` or `readBufferRange()`. |
| WebNN | Ordinary-forward fallback; `requireIncremental: true` rejects it. | Host typed arrays. |

The legacy `execute(inputs, { incremental, incrementalReset, changedInputs,
incrementalRowPosition })` options remain supported. `DecodeSession` maps to
that behavior so existing integrations can migrate without changing results.

## Tier 1: WebNN

`WebNNEngine` builds an `MLGraphBuilder` graph and dispatches through
`navigator.ml`. The browser may route work to NPU, GPU, or CPU depending on the
platform, browser flags, drivers, and supported ops.

Current WebNN coverage:

`MatMul`/`Linear`/`Gemm`, `Add`, `Mul`, `ReLU`, `GELU`, `SiLU`/`Swish`,
`Sigmoid`, `Softmax`, `Reshape`/`Flatten`, `LayerNorm`, `Conv2D`,
`Embedding`, and `SDPA` decomposed into lower-level WebNN ops.

Missing low-cost mappings include `Sub`, `Div`, `Tanh`, `Clip`, `LeakyReLU`,
`PReLU`, `HardSwish`, `HardSigmoid`, `Transpose`, `Concat`, `Split`, `Slice`,
`Pad`, `Where`, `Gather`, `Cast`, `Expand`, reductions, pooling, batch norm,
resampling, and `ConvTranspose2D`.

Missing composite formulas include `RMSNorm`, `CrossSDPA`, `CrossAttention`,
`LogSoftmax`, `DequantizeLinear`, and `Conv1D` as a reshape plus `conv2d`.

WebNN requires a secure context such as HTTPS or localhost. Chromium WebNN support
is evolving and often needs `--enable-features=WebMachineLearningNeuralNetwork`.
`deviceType: "npu"` is not proof that an NPU executed the work; unsupported
accelerators can fall back to CPU.

Useful references:

- W3C WebNN: <https://www.w3.org/TR/webnn/>
- WebNN compatibility: <https://webnn.io/en/api-reference/browser-compatibility/api>
- Chromium flags: <https://webnn.io/en/api-reference/browser-compatibility/chrome-flags>

## Tier 2: WebGPU

`WebGPUEngine` owns a `GraphExecutor`, which allocates one GPU buffer per tensor,
uploads weights once, and builds one compute pipeline per node. The executor
owns those buffers in backend-local state keyed by tensor name; the portable
core `Tensor` object does not retain a `GPUBuffer`. `execute()` replays the
prebuilt pipelines in one command encoder, flushing every 20 dispatches.

For fixed-shape B=1 W8A8 decoding, the seed pass populates the complete device
tensor set. A later row step uploads only changed input rows, dispatches the
dependency-selected decoder closure against one-row scratch buffers, and copies
each result row back into its canonical device tensor. Causal `QSDPA` reads K/V
directly from those canonical buffers with `seqKV = position + 1`, so previous
rows stay device-resident; cross-attention keeps the complete encoder K/V. All
row copies and compute passes for one token share one GPU command submission.
Only row pipelines used by the selected closure are compiled, on the first row
step. Applications that need the generated token on the host should read its
four-byte I32 slot with `readBufferRange()` rather than copying the entire
fixed-length output tensor.

`executeDeviceFeedbackDecode()` avoids even that per-token synchronization for
strict fixed-range generation. It runs the complete seed, copies terminal
`QArgMax[p - 1]` directly into the next I32 token input, writes I32 one into the
corresponding keep-mask row, and enqueues row execution for `p`. Causal QSDPA
prefix lengths are written immediately before each row's ordered submission,
so each dispatch sees its own value without mapping an intermediate result:

```javascript
const outputBuffer = await engine.executeDeviceFeedbackDecode(seedInputs, {
  tokenInput: 'decoder_ids',
  keepInput: 'decoder_keep',
  output: 'token_ids',
  endPosition: 32,
});
let tokenIds = await engine.readBufferRange(
  outputBuffer, 0, 32 * Int32Array.BYTES_PER_ELEMENT, 'int32',
);
if (!tokenIds.includes(eosTokenId)) {
  await engine.executeDeviceFeedbackDecode(null, {
    tokenInput: 'decoder_ids',
    keepInput: 'decoder_keep',
    output: 'token_ids',
    startPosition: 32,
    endPosition: 64,
  });
  tokenIds = await engine.readBufferRange(
    outputBuffer, 32 * Int32Array.BYTES_PER_ELEMENT,
    32 * Int32Array.BYTES_PER_ELEMENT, 'int32',
  );
}
```

The fast path requires distinct I32 `[1,S]` token, keep, and declared output
tensors; a selected QEmbedding; a selected causal QSDPA masked by the keep
input; a terminal QArgMax with the same vocabulary as the embedding; and a row
plan for every dependency-selected node. It rejects the graph before the seed
if any condition is absent. Rows are submitted in queue order without awaiting
or mapping between them; `rowsPerSubmission` is currently fixed at one so each
causal QSDPA dispatch observes its own portable prefix uniform. A call
starting at zero owns the seed; a later call can resume only at the exact next
position while that cache remains valid. Reading 16- or 32-token chunks lets an
application stop near EOS with one map per chunk. A single fixed-range call
cannot stop dispatching at EOS, so truncate its final result at the first EOS
token on the host.

On the repository's AMD GCN-5 browser host, the real 23 MB TinyReceipt package
(`00002.jpg`, `phone number last one`, 80 generated tokens) measured 2,444.5 ms
hot with one four-byte map per token and 1,920.2 ms with 16-token device-feedback
chunks, a 21.5% reduction. Cold generation changed from 2,629.2 to 2,359.9 ms.
Both modes returned the same WebGPU answer. This is an end-to-end single-sample
timing, not an accuracy score; its answer differs from the WASM/native answer
for this sample, so WebGPU still needs the planned heldout accuracy pass before
deployment qualification.

The WebGPU path keeps the model resident on the GPU and returns a `GPUBuffer`.
Current limitation: it returns the final node's first output rather than a map of
all `graph.outputNames`. Multi-output models should use WASM/CPU until WebGPU
multi-output readback is implemented.

Canonical W8A8 dense and convolution nodes have an optional packed integer-dot
path. `GraphExecutor` selects it only when
`navigator.gpu.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')` is
true; only those specialized WGSL modules declare
`requires packed_4x8_integer_dot_product`. `QLinear`/`QMatMul`/`QGemm` use
`dot4I8Packed` for scalar and cooperative multi-row tiles. Eligible groups=1
`QConv2D` nodes use a cooperative four-spatial-position by 32-output-channel
tile that caches 32 reduction bytes in workgroup memory. Arbitrary I8/U8 zero
points are handled exactly, and reduction tails are filled with each tensor's
zero point. When packed dot is unavailable, eligible groups=1 convolutions
with at least 32 output channels use the same cooperative shape with portable
lane multiplies. Ineligible grouped or small convolutions, a tiled dispatch
beyond the device's per-dimension workgroup limit, and specialized pipeline
compilation failure retain the scalar packed-byte shader. Compute pipelines are cached per executor by
shader source, entry point, and override constants; each node still owns its
parameter buffers and bind group. Native shader generation restricts the DP4a
module to Vulkan SPIR-V; the portable tile is also available to the native
OpenGL backend.

Unsupported WebGPU shader nodes currently warn and skip, leaving the output buffer
unwritten. Use [operation_list.md](operation_list.md) to check whether a model's
ops are safe on WebGPU.

## Tier 3: WASM SIMD

`WasmEngine` loads the freestanding `volvoxai.wasm` module built from
`native/src/kernels/kernels.c`. The full sidecar additionally links
`native/src/kernels/training_kernels.c` and
`native/src/training/quantization.c`. Training and PTQ calls use separate
scratch instances so resetting a temporary arena cannot invalidate the live
inference graph. Canonical graph weights and checkpoint state remain ordinary
JS typed arrays. Some pointwise, broadcast, and gather inference operations
are handled in JS over typed-array views into the same heap.

Both fixed WASM files remain baseline SIMD modules.  They embed the optional
Relaxed-SIMD M=1 W8A8 QLinear child in the versioned
`volvoxai.relaxed_simd.v1` custom section.  `WasmEngine` compiles that actual
child as the feature check, imports only the parent's memory, and uses it for
immutable packed incremental decoder rows.  Missing or unsupported sections,
descriptor rejection, and child traps all fall back to the baseline packed and
then raw kernels without another fetch. Full-range I8/U8 activations are split
into low-seven-bit and high-bit dot products, so the instruction's unsigned
i7 operand never carries a set high bit and this kernel remains exact across
the proposal's permitted implementations. This checks engine support for the
instruction; WebAssembly does not expose whether the host physically has VNNI
or Arm dot-product hardware.

Ordinary packed W8A32 decode rows do not require Relaxed SIMD. Baseline
`simd128` uses two compensated `f32x4` accumulators for each eight-output
panel when the descriptor is inside the audited Tiny VQA envelope: `M=1`,
signed-I8 weights, effective zero point zero, `K<=1280`, weight scales no
larger than `0.025`, and finite F32 activations in `[-35,35]`. Compensated
summation removes the observed long-cancellation failure from naive sequential
F32 accumulation. The path remains tolerance-tested rather than bit-exact to
the portable double reference. Asymmetric, U8, larger, or out-of-envelope
inputs keep the existing scalar/double path.

## Tier 4: CPU

`CPUEngine` is the pure-JS reference backend. It is slower, but dependency-free and
useful for debugging, fallback behavior, and cross-tier correctness checks.

## Node Import Behavior

`ShaderLibrary.ts` statically imports every `shaders/inference/*.wgsl` file as text, which
only works in the bundled browser build. It is not re-exported from `ts/index.ts`.
`GraphExecutor` lazy-loads it only when the WebGPU path compiles, so plain Node
imports can use WASM/CPU without a bundler.
