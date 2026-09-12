# VolvoxAI documentation

Choose a starting point based on what you want to do:

- **Learn how AI works.** The textbook explains tensors, models, training, and
  quantization from scratch, with Idea, Build, and Deep tracks.
- **Run a model.** Start with the quickstart, then the guide for your platform.
- **Build on the engine.** Read the architecture and the focused development
  guides. Exact API fields have a separate reference at the end of this page.

## Learn and try it

| Guide | What you will learn or do |
| --- | --- |
| [Textbook](textbook/README.md) · [한국어](textbook/ko/README.md) | How AI Actually Works: ideas for beginners, code for builders, internals for engine developers |
| [Quickstart](quickstart.md) | Build the runtime, execute a tiny graph, prepare an example model, and choose your next step |
| [Models and exporters](models.md) | Recreate EfficientDet and TinyStories packages from their original sources |

## Integrate a model

| Guide | What it covers |
| --- | --- |
| [Browser and Node runtime](browser-runtime.md) | Loading, backend selection, sessions, stable results, deployment, and browser extensions |
| [Native runtime](native-runtime.md) | Raw-tensor CLI, CPU threads, native GPUs, C embedding, and macOS/Android builds |
| [Model format](model-format.md) | Graphs, weights, tensor layouts, and bounded dynamic dimensions |
| [Model construction and training](model_builder_training.md) | Build a classifier, train it, accumulate gradients, and save/resume a checkpoint |
| [Quantization](quantization.md) | Calibrate a float model, export W8A8, and validate the resulting package |
| [Training and PTQ matrix](training-ptq-runtime-matrix.md) | Which profiles and backends support each training or PTQ workflow |
| [Scheduling and dynamic batching](scheduling-and-dynamic-batching-design.md) | Concurrent requests, budgets, batching, decode sessions, and paged KV |

## Understand and extend the engine

Start with [ARCHITECTURE.md](../ARCHITECTURE.md) for the execution model and
source map. Use these pages for a particular subsystem:

| Guide | What it covers |
| --- | --- |
| [Operators](operation_list.md) | Operator meaning, dtype/shape limits, and implementation routes |
| [Backend development](backend-sdk.md) | Provider ownership, graph compilation, input/output callbacks, and qualification |
| [Dynamic-shape ADR](adr-dynamic-shape-v1.md) | Why dimensions are named and bounded, and how requests bind them |
| [Graph optimizer](graph-optimizer-design.md) | Import, lowering, graph passes, validation, and package publication |
| [Typed PTQ](typed-ptq.md) · [W8A8 storage](w8a8-safetensors.md) | Quantized graph semantics and numeric/storage requirements |
| [Weight banks](weight-bank-design.md) | Resident subsets, expert routing, and compiled/context ownership |
| [CUDA](cuda.md) | Native CUDA architecture, operators, numeric modes, and build options |
| [Microkernels](microkernel_optimization_guide.md) · [XNNPACK notes](xnnpack_optimization_guide.md) | CPU packing, convolution, and GEMM techniques |
| [Fusion patterns](operator_fusion_patterns.md) | Candidate optimizations and their tradeoffs |
| [Testing](testing.md) · [Runtime validation](c-runtime-validation.md) | How to run checks and interpret their coverage |
| [Profiling](profiling.md) | Latency and memory measurements with explicit timing boundaries |

## Measurements

These reports describe particular models, devices, artifacts, and dates.
Historical CPU-JS measurements remain useful comparisons; that backend is no
longer shipped. Follow each report's provenance when interpreting its numbers.

- [Dynamic-shape baseline](dynamic-shape-baseline.md) and [package inventory](dynamic-shape-package-inventory.md)
- [EfficientDet comparison](efficientdet_tflite_vs_volvoxai.md)
- [Receipt digit reader](receipt-digit-reader-benchmark.md)
- [Tiny Receipt VQA](tiny-receipt-vqa-bpe1536-benchmark.md)

## API lookup and generated references

Applications and agents share the operations in [volvoxai.proto](../proto/volvoxai.proto).
The guides above explain workflows; these references describe individual calls,
fields, defaults, and structured errors:

- [API discovery and input diagnostics](api-discovery.md)
- [Inference API](generated/api-contract.inference.md) · [Full API](generated/api-contract.full.md)
- [Inference JSON](generated/api-contract.inference.json) · [Full JSON](generated/api-contract.full.json)
- [C, TypeScript, and Python integration](../runtime/README.md)
- [Kernel registry](generated/kernel-registry.md) · [Optimizer registry](generated/optimizer-registry.md)

Generated references are rebuilt from their source and are not edited by hand.
Open implementation work is tracked in [TODO.md](../TODO.md).
