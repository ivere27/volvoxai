# Training and PTQ runtime matrix

VolvoxAI has two capability profiles:

- inference: forward execution only.
- full: inference plus Trainer, backward kernels, optimizer state, checkpoints,
  and PTQ authoring.

The JavaScript inference lifecycle is always Runtime → Model → CompiledModel →
ExecutionContext → ExecutionResult. Training state has a separate retained
Trainer owner. The native public C surface uses matching opaque inference
handles; native/volvoxai-full additionally contains the model-agnostic train
command.

Legend: ✅ supported, ⚠️ supported with stated restrictions, ❌ unsupported.

## Release composition

| Profile | JavaScript | Native | WASM |
| --- | --- | --- | --- |
| Inference | volvoxai.js and minified variant | native/volvoxai | volvoxai.wasm |
| Full | volvoxai.full.js and minified variant | native/volvoxai-full | volvoxai.full.wasm |
| WASM-only browser | volvoxai.wasm.js and minified variant | N/A | volvoxai.full.wasm |
| Training implementation absent from inference | ✅ | ✅ | ✅ |
| PTQ authoring absent from inference | ✅ | ✅ | ✅ |

The WASM-only browser module contains strict WASM inference and Trainer but no
CPU, WebNN, WebGPU, WGSL, or Node filesystem implementation.

## Training ownership

| State | Owner |
| --- | --- |
| Immutable topology and published weights | Model |
| Exact compiled revision | CompiledModel |
| Request, scratch, adapter, and decode state | ExecutionContext |
| Stable output snapshots | ExecutionResult |
| Gradients and optimizer slots | Trainer |
| Accumulation state | Trainer |
| Private mutable working revision | Trainer |
| Published successor revision | Model, committed atomically by Trainer |

`trainStep()` never publishes a Model revision. A completed optimizer update
remains private until `commit()` publishes it atomically. `rollback()` restores
the last committed baseline. Compiled models created before a commit remain
pinned to their original weights.

## Trainer support

| Capability | JavaScript CPU | Browser WebGPU | Strict WASM | Native full command |
| --- | --- | --- | --- | --- |
| Create graph with full ModelBuilder | ✅ | ✅ | ✅ | Package supplied by caller |
| F32 forward/backward | ✅ | ✅ documented subset | ✅ documented subset | ✅ CPU/device subset |
| Cross-entropy | ✅ | ✅ | ✅ | ✅ |
| Multiple weighted losses | ✅ | ✅ | ✅ | ✅ |
| Loss mask and ignore index | ✅ | ✅ | ✅ | ✅ |
| Explicit trainable tensor set | ✅ | ✅ | ✅ | ✅ |
| SGD | ✅ | ✅ | ✅ | ✅ |
| AdamW | ✅ | ✅ | ✅ | ✅ |
| Weight decay | ✅ | ✅ | ✅ | ✅ |
| One global gradient norm clip | ✅ | ✅ | ✅ | ✅ |
| Non-finite rejection | ✅ | ✅ | ✅ | ✅ |
| Gradient accumulation | ✅ | ✅ | ✅ | ✅ |
| Flush/reset accumulation | ✅ | ✅ | ✅ | ✅ |
| Deterministic Dropout | ✅ | ✅ | ✅ subset | ✅ supported paths |
| Explicit F32 LoRA A/B training | ✅ | ✅ | ✅ | ✅ |
| Checkpoint weights and moments | ✅ | ✅ | ✅ through JS | ✅ files |
| Browser persistence | Application-owned | Application-owned | Application-owned | N/A |
| Filesystem persistence | Node/application | Node/application | Application-owned | ✅ |
| Automatic fallback during a step | ❌ | ❌ | ❌ | ❌ after plan selection |
| Deep QLinear/QGemm backward | ❌ | ❌ | ❌ | ❌ |
| Fake-quantization QAT | ❌ | ❌ | ❌ | ❌ |
| FP16/AMP training | ❌ | ❌ | ❌ | ❌ |

Use:

~~~javascript
const trainer = await VolvoxAI.createTrainer(model, {
  backend: 'cpu', // or 'webgpu' / 'wasm'
});

const step = await trainer.trainStep(options);
await trainer.commit();
await trainer.close();
~~~

Commit only after a completed optimizer update; an incomplete accumulation
window cannot be committed. There is no implicit publication.

WASM preflights the complete supported graph before forward execution or
optimizer mutation. WebGPU owns its device resources inside Trainer. Supplying
a GPUDevice keeps device ownership with the caller; an internally requested
device is released by Trainer.

CUDA full-profile training preflights its complete backward plan. Selected
CUDA work is strict and does not cross to CPU after execution starts. Exact
operator, residency, and numerical limits are in [cuda.md](cuda.md).

## PTQ authoring support

| Capability | JavaScript full profile | WASM numerical sidecar | Native profiles |
| --- | --- | --- | --- |
| Min/max observation | ✅ PTQObserver | Full sidecar kernels | Full build internals |
| Multiple calibration samples | ✅ Runtime result loop | JS orchestration | Full command/tooling |
| Symmetric parameters | ✅ | ✅ | ✅ full only |
| Asymmetric parameters | ✅ | ✅ | ✅ full only |
| Per-tensor I8 [-128,127] | ✅ | ✅ | ✅ full only |
| Per-tensor U8 [0,255] | ✅ | ✅ | ✅ full only |
| Ties-to-even rounding | ✅ | ✅ | ✅ |
| Saturation count | ✅ | ✅ | ✅ |
| Per-axis weight I8 [-127,127] | ✅ | ✅ | ✅ full only |
| Weight ranks one through eight | ✅ | ✅ | ✅ |
| Arbitrary valid axis | ✅ | ✅ | ✅ |
| I32 accumulator bias | ✅ | ✅ | ✅ full only |
| In-memory safetensors | ✅ | JS owns bytes | Tooling-owned |
| Write graph.json | Application-owned | Application-owned | Tooling-owned |
| Insert quantized graph nodes automatically | ❌ | ❌ | ❌ |
| Infer quantization boundaries | ❌ | ❌ | ❌ |
| Histogram/percentile/KL/MSE search | ❌ | ❌ | ❌ |
| SmoothQuant/GPTQ/AWQ | ❌ | ❌ | ❌ |
| QAT | ❌ | ❌ | ❌ |

Calibration uses an ordinary ExecutionContext and declares each observed
boundary as a graph output. TensorResult.read() supplies caller-owned F32 values
to PTQObserver. Device backends therefore use the same stable result contract;
no backend intermediate or stale host mirror is exposed.

Current full-profile authoring functions are:

| Purpose | JavaScript |
| --- | --- |
| Observe values | PTQObserver.observe() |
| Derive parameters | derivePTQParameters() |
| Quantize values | quantizePTQ() |
| Pack weights | packPTQWeight() |
| Pack bias | packPTQBias() |
| Build safetensors and central reference table | materializePTQWeights() |

The author writes an explicit graph.json with format: volvox-graph/v1 and the
selected Q operators, tensor descriptors, central scale/zero-point tensor
references, and quantize/dequantize boundaries. Every referenced numeric scale
and zero point lives in safetensors; none is embedded in graph JSON or
safetensors metadata. The returned property is `artifact.quantization`, not a
weight-quantization compatibility field.

The `Insert quantized graph nodes automatically` and `Infer quantization
boundaries` rows describe these low-level JavaScript/native authoring APIs. The
separate offline typed toolchain can plan and transactionally materialize its
exact supported dense subset; see [typed-ptq.md](typed-ptq.md). It fails closed
outside that subset and does not claim general automatic Conv, normalization,
attention, or mixed-precision policy selection.

## WASM boundary

~~~text
JavaScript
  Runtime/Model/Context ownership
  Trainer policy and checkpoints
  PTQ observations and package bytes
                   |
                   v
volvoxai.full.wasm
  forward/backward kernels
  optimizer math
  PTQ numerical primitives
~~~

WASM is a numerical backend, not a second browser model owner. JavaScript owns
the graph, published revisions, results, checkpoints, browser storage, and
package layout.

See:

- [Browser and Node runtime](browser-runtime.md)
- [Post-training quantization](quantization.md)
- [Model construction and training](model_builder_training.md)
- [Operation matrix](operation_list.md)
