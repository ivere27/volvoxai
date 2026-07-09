# VolvoxAI Documentation

This directory holds the detailed reference material for VolvoxAI. Start with
the root [README](../README.md) for the short overview, then use these pages for
the implementation details.

| Doc | What it covers |
| --- | --- |
| [quickstart.md](quickstart.md) | Build commands, browser bundle, CLI smoke tests, and model download commands. |
| [browser-runtime.md](browser-runtime.md) | Browser and Node runtime tiers: WebNN, WebGPU, WASM SIMD, and CPU. |
| [native-runtime.md](native-runtime.md) | Native C engine, native CLI tasks, GPU/NPU backends, Android cross-build notes. |
| [model-format.md](model-format.md) | Volvox blueprint format, safetensors loading, tensor layout, precision policy. |
| [models.md](models.md) | Regenerating EfficientDet, TinyStories, and TinyReceiptKIE model packages. |
| [operation_list.md](operation_list.md) | Per-op backend support matrix for browser and native runtimes. |
| [testing.md](testing.md) | WebGPU op tests, native smoke tests, parity checks, and known validation limits. |
| [roadmap.md](roadmap.md) | Current gaps and planned work. |
| [textbook/](textbook/README.md) | A from-scratch walkthrough of inference, tensors, models, and engine internals. |
| [efficientdet_tflite_vs_volvoxai.md](efficientdet_tflite_vs_volvoxai.md) | EfficientDet Lite0 CPU/GPU benchmark methodology and results. |
| [microkernel_optimization_guide.md](microkernel_optimization_guide.md) | CPU Conv/GEMM microkernel notes. |
| [operator_fusion_patterns.md](operator_fusion_patterns.md) | Graph fusion patterns applied by the native optimizer. |
| [xnnpack_optimization_guide.md](xnnpack_optimization_guide.md) | XNNPACK-style packing and indirection reference notes. |

