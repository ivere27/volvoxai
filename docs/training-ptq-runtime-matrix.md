# Training and PTQ runtime matrix

This document compares the ownership and current support of training and
post-training quantization (PTQ) across VolvoxAI's JavaScript/TypeScript,
native C, and browser WebAssembly profiles.

The three columns are not completely independent implementations:

- **JS/TS full runtime** means `volvoxai.full.js` and its TypeScript sources,
  including the pure-JS CPU path and browser-backend orchestration.
- **Native C full runtime** means `native/volvoxai-full` and the public native
  training API.
- **WASM full runtime** means the WASM-only JavaScript facade controlling the C
  code compiled into `volvoxai.full.wasm`. JavaScript owns graph state and
  policy; WASM owns the portable numerical kernels.

Legend: ✅ supported, ⚠️ supported with restrictions or orchestration,
❌ not supported.

## Release profiles

| Profile | JavaScript | Native C | WebAssembly |
| --- | --- | --- | --- |
| Inference | `volvoxai.js`, `volvoxai.min.js` | `native/volvoxai` | `volvoxai.wasm` |
| Training/PTQ | `volvoxai.full.js`, `volvoxai.full.min.js` | `native/volvoxai-full` | `volvoxai.wasm.js` plus `volvoxai.full.wasm` |
| Training code excluded from inference | ✅ | ✅ | ✅ |
| PTQ code excluded from inference | ✅ | ✅ | ✅ |

## Training support

| Capability | JS/TS full runtime | Native C full runtime | WASM full runtime |
| --- | --- | --- | --- |
| Create graphs programmatically | ✅ `TrainingModelBuilder` | ⚠️ Primarily trains loaded or caller-authored blueprints | ✅ JS builder |
| Load graph and safetensors | ✅ | ✅ | ✅ JS loader |
| F32 forward | ✅ CPU/WebGPU and selectable backends | ✅ CPU/Vulkan/OpenGL/Metal | ✅ C/WASM kernels |
| Autograd scheduling | ✅ JS | ✅ Native runtime | ✅ JS `WasmAutograd` |
| Backward numerical kernels | ✅ JS CPU or WGSL | ✅ C/device kernels | ✅ C compiled to WASM |
| Cross-entropy loss | ✅ | ✅ | ✅ C/WASM |
| Multiple weighted losses | ✅ | ✅ `volvoxai_engine_train_step_multi()` | ✅ |
| Loss masking and ignore index | ✅ | ✅ | ✅ |
| Select explicit trainable tensors | ✅ | ✅ | ✅ |
| Freeze base weights | ✅ | ✅ | ✅ |
| SGD | ✅ | ✅ | ✅ C/WASM |
| AdamW | ✅ | ✅ | ✅ C/WASM |
| Weight decay | ✅ | ✅ | ✅ |
| Gradient clipping | ✅ | ✅ | ✅ |
| Non-finite gradient rejection | ✅ | ✅ | ✅ |
| Gradient accumulation | ✅ | ✅ | ✅ |
| Flush/reset accumulation | ✅ | ✅ | ✅ |
| Deterministic Dropout training | ✅ | ✅ | ✅ |
| Explicit F32 LoRA graph training | ✅ | ✅ | ✅ |
| User-correction/teacher-forcing training | ✅ Application builds targets | ✅ Caller builds targets | ✅ Application builds targets |
| Train only LoRA A/B | ✅ | ✅ | ✅ |
| F32-master to I8 LoRA conversion | ✅ JS policy | ✅ C conversion primitive | ✅ C/WASM conversion |
| Atomic update of several W8 graph weights | ✅ JS staging/rollback | ⚠️ Caller/native-runtime policy | ✅ JS `WasmQuantizedLoRATrainer` |
| Preserve an existing W8A8 topology | ✅ With separate graphs | ⚠️ Manual workflow | ✅ Convenience trainer |
| Train directly through a deep QLinear/QGemm W8A8 graph | ❌ | ❌ Current training contract | ❌ |
| QAT with fake quantization | ❌ | ❌ | ❌ |
| F32 checkpoints | ✅ Structured-clone/safetensors | ✅ Weights plus optimizer state | ✅ JS checkpoint |
| AdamW moment restoration | ✅ | ✅ | ✅ |
| Browser IndexedDB persistence | ✅ Application-controlled | N/A | ✅ Application-controlled |
| Filesystem checkpoint writing | ✅ Node only | ✅ | ❌ C/WASM; use JS browser storage |
| Automatic CPU fallback | Depends on selected backend | Configurable native backend | ❌ Strict WASM-only facade |
| Raw C training-kernel API exposed to applications | N/A | ✅ Public native training API | ❌ Raw pointers remain internal |
| Safe generic graph-level training API | ✅ | ✅ | ✅ `createWasmTrainer()` |

### Training ownership in the WASM profile

| Work | Owner |
| --- | --- |
| Graph traversal and backward planning | JavaScript |
| Tensor selection and freezing | JavaScript |
| Loss configuration | JavaScript |
| Numerical backward operations | C/WASM |
| SGD/AdamW calculations | C/WASM |
| Gradient-accumulation state | JavaScript with C/WASM computation |
| Checkpoint serialization | JavaScript |
| Inference-graph mutation and rollback | JavaScript |
| Weight-quantization calculation | C/WASM |

## PTQ support

| PTQ capability | JS/TS full runtime | Native C full runtime | WASM full runtime |
| --- | --- | --- | --- |
| Reset min/max observer | ✅ | ✅ | ✅ C/WASM |
| Accumulate observations | ✅ | ✅ | ✅ C/WASM |
| Multiple calibration batches | ✅ | ✅ | ✅ |
| Symmetric parameter calculation | ✅ | ✅ | ✅ C/WASM |
| Asymmetric parameter calculation | ✅ | ✅ | ✅ C/WASM |
| Per-tensor I8 quantization, full range `[-128,127]` | ✅ | ✅ | ✅ C/WASM |
| Per-tensor U8 quantization | ✅ | ✅ | ✅ C/WASM |
| Ties-to-even rounding | ✅ | ✅ | ✅ |
| Saturation counting | ✅ | ✅ | ✅ |
| Per-axis symmetric I8 weights, narrow range `[-127,127]` | ✅ | ✅ | ✅ C/WASM |
| Arbitrary weight axis | ✅ | ✅ | ✅ |
| Weight ranks 1–8 | ✅ | ✅ | ✅ |
| I32 accumulator bias packing | ✅ | ✅ | ✅ C/WASM |
| Transpose while packing LoRA weights | ✅ | ✅ Helper | ✅ Helper |
| Preserve existing weight scales | ✅ | ✅ Helper | ✅ Helper |
| I8 weight dequantization to F32 | ✅ | ✅ Helper | ✅ Helper |
| Run calibration inference | ✅ | ✅ | ✅ WASM engine |
| Observe named intermediate graph tensors | ✅ `calibratePTQ()` | ✅ Engine observer | ⚠️ JS reads the tensor and passes it to `WasmPTQ` |
| Dataset calibration loop | ✅ High-level calibrator | ✅ PTQ plan | ✅ JS loop plus C/WASM observer |
| Materialize packed tensors | ✅ | ✅ | ✅ Returns typed arrays |
| Automatically insert QLinear/QConv nodes | ❌ | ❌ | ❌ |
| Automatically rewrite graph topology | ❌ | ❌ | ❌ |
| Generate quantization descriptors | ✅ | ✅ Package plan | ✅ Weight descriptors returned; caller integrates them |
| Produce in-memory safetensors | ✅ | ⚠️ Native writer is file-based | ✅ Through JS authoring |
| Write config/safetensors files | ✅ Node/browser application | ✅ | ❌ Inside C/WASM; use JS |
| Validate a caller-authored quantized template | ✅ Materializer validation | ✅ PTQ plan | ⚠️ Application/JS layer |
| Transactional filesystem package publication | ❌ Browser; application policy | ✅ | ❌ |
| Histogram calibration | ❌ | ❌ | ❌ |
| Percentile calibration | ❌ | ❌ | ❌ |
| KL/entropy calibration | ❌ | ❌ | ❌ |
| MSE scale search | ❌ | ❌ | ❌ |
| SmoothQuant | ❌ | ❌ | ❌ |
| GPTQ | ❌ | ❌ | ❌ |
| AWQ | ❌ | ❌ | ❌ |
| Automatic mixed precision | ❌ | ❌ | ❌ |
| QAT | ❌ | ❌ | ❌ |

## Main public APIs

| Purpose | JavaScript | Native C | WASM-only |
| --- | --- | --- | --- |
| Generic training | `trainStep()` | `volvoxai_engine_train_step_multi()` | `trainStep()` / `createWasmTrainer()` |
| LoRA training | `trainStep()` with selected A/B | `volvoxai_engine_lora_train_step()` | `trainLoRAStep()` |
| Quantized LoRA | `WasmQuantizedLoRATrainer` when using WASM | C primitives/manual integration | `createQuantizedLoRATrainer()` |
| PTQ observer | `PTQObserver` | `volvoxai_ptq_observer_*` | `WasmPTQ.createObserver()` |
| PTQ parameters | `derivePTQParameters()` | `volvoxai_ptq_calculate_params()` | `deriveParameters()` |
| Quantize values | `quantizePTQ()` | `volvoxai_ptq_quantize_f32()` | `quantize()` |
| Pack weights | `packPTQWeight()` | `volvoxai_ptq_pack_weight_i8()` | `packWeight()` |
| Pack bias | `packPTQBias()` | `volvoxai_ptq_pack_bias_i32()` | `packBias()` |
| Full PTQ package | `materializePTQWeights()` | `VolvoxAIPTQPlan` | JS orchestration using `WasmPTQ` results |

## WASM execution boundary

```text
JavaScript
  graph + training policy + checkpoints + files
                         |
                         v
volvoxai.full.wasm
  forward + backward + optimizers + generic PTQ math
```

The WASM sidecar exposes reusable numerical computation, not a second
C-owned browser model runtime. The native PTQ plan and package writer depend on
native graph globals, locks, cJSON, safetensors file paths, and filesystem
publication. In browsers, JavaScript maps the typed results returned by
`WasmPTQ` into a graph, package, IndexedDB record, or application-specific
workflow.

`WasmPTQ` owns a private scratch WebAssembly instance. Reuse it across related
operations and call its idempotent `dispose()` when finished so the instance
and its linear memory are no longer retained by the toolkit.

See also:

- [Browser and Node runtime](browser-runtime.md)
- [Post-training quantization](quantization.md)
- [Model-builder training](model_builder_training.md)
- [Architecture](../ARCHITECTURE.md)
