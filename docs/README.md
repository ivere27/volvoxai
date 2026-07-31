# VolvoxAI Documentation

There are **two kinds of docs** here. Pick the one that matches what you want:

- 📚 **Learn** — *understand how AI works, from scratch.* Start here if you're new, a student, or
  just curious. No prior ML knowledge assumed.
- 🛠️ **Reference** — *build with / on the engine.* Precise implementation details for developers
  integrating VolvoxAI, adding backends, or authoring models.

Most people want **Learn**.

---

## 📚 Learn — the textbook

| Doc | What it covers |
| --- | --- |
| **[textbook/](textbook/README.md)** | **How AI Actually Works** — a from-scratch walkthrough of tensors, models, training, quantization, and the engine, written in three depths (🌱 Idea for anyone · 🔧 Build for coders · 🔬 Deep for engine developers). **This is the front door for learning.** |
| [quickstart.md](quickstart.md) | The fastest hands-on path: build commands, browser bundle, CLI smoke tests, and model download commands. |

---

## 🛠️ Reference — building with the engine

Start with the root [README](../README.md) for the product overview and
[ARCHITECTURE.md](../ARCHITECTURE.md) for the source map, then use these pages for the details.

**Runtimes & platforms**

| Doc | What it covers |
| --- | --- |
| [browser-runtime.md](browser-runtime.md) | Browser and Node runtime tiers: WebNN, WebGPU, WASM SIMD, and CPU. |
| [native-runtime.md](native-runtime.md) | Native C engine, fixed raw-tensor CLI, opt-in task example, GPU/NPU backends, Android cross-build notes. |
| [cuda.md](cuda.md) | Canonical native CUDA architecture, operator coverage, numeric modes, model validation, RTX 3090 benchmarks, and remaining work. |
| [backend-sdk.md](backend-sdk.md) | Versioned native and browser contracts for custom GPUs, NPUs, and other devices. |

**Model format & data**

| Doc | What it covers |
| --- | --- |
| [model-format.md](model-format.md) | `volvox-graph/v1`, safetensors loading, tensor layout, and precision policy. |
| [w8a8-safetensors.md](w8a8-safetensors.md) | Normative central-reference affine quantization and safetensors contract. |
| [models.md](models.md) | Regenerating EfficientDet and TinyStories model packages. |
| [operation_list.md](operation_list.md) | Per-op backend support matrix for browser and native runtimes. |
| [generated/kernel-registry.md](generated/kernel-registry.md) | Generated backend inventories, exporter qualification, routes, and physical kernel variants sourced from `proto/kernel_registry.proto`. |

**Training & quantization**

| Doc | What it covers |
| --- | --- |
| [quantization.md](quantization.md) | Full-profile calibration, affine parameters, I8 packing, and explicit package authoring. |
| [model_builder_training.md](model_builder_training.md) | Building and training models through the API. |
| [training-ptq-runtime-matrix.md](training-ptq-runtime-matrix.md) | Training and PTQ ownership/support across JavaScript, native C, and browser WASM. |

**Exporter & graph optimizer**

| Doc | What it covers |
| --- | --- |
| [graph-optimizer-design.md](graph-optimizer-design.md) | The v1-only ONNX/TensorFlow Lite import, verified RuntimeIR optimizer, specialization, mixed-precision PTQ, differential qualification, and publication architecture. |
| [typed-ptq.md](typed-ptq.md) | Exact typed PTQ contracts, transactional materialization, supported dense topology, and deliberate gaps. |

**Performance & optimization**

| Doc | What it covers |
| --- | --- |
| [microkernel_optimization_guide.md](microkernel_optimization_guide.md) | CPU Conv/GEMM microkernel notes. |
| [operator_fusion_patterns.md](operator_fusion_patterns.md) | A design catalogue of valuable fusion candidates; it is not an implementation or coverage list. |
| [xnnpack_optimization_guide.md](xnnpack_optimization_guide.md) | XNNPACK-style packing and indirection reference notes. |
| [efficientdet_tflite_vs_volvoxai.md](efficientdet_tflite_vs_volvoxai.md) | EfficientDet Lite0 CPU/GPU benchmark methodology and results. |

**Project**

| Doc | What it covers |
| --- | --- |
| [testing.md](testing.md) | WebGPU op tests, native smoke tests, parity checks, and known validation limits. |
| [roadmap.md](roadmap.md) | Current gaps and planned work. |
