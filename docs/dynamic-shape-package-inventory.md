# Dynamic-shape package migration inventory

This document records the DS0 package inventory for the breaking dynamic-shape
redesign described by [`TODO.md`](../TODO.md). The original inventory was taken
at commit `e9696b57c4f7`, branch `shape`, on 2026-08-01. Its counts were planning
evidence for that worktree, not permanent repository invariants.

> **Historical scope.** The original inventory included CPU-JS kernels,
> generated comparison packages, validation fixtures, and their commands. Those surfaces
> have since been removed. This revision preserves package and schema ownership
> facts without presenting retired paths or commands as current interfaces.

The cutover has no compatibility reader. Authored producers change first,
ignored packages are regenerated from those producers, and generated JSON,
shader projections, embedded arrays, and release artifacts are never edited by
hand.

## Post-cutover package state

The four ignored repository packages were regenerated through their owning
exporters and validated against the bounded-domain package contract. The
EfficientDet variants remain useful constant-only packages. TinyStories is a
bounded dynamic package: dimension `S` has `min=1`, `max=256`, both public
inputs are `[1,"S"]`, all 85 node outputs preserve `S`, and logits are
`[1,"S",50257]`. Its reviewed graph SHA-256 is
`05f6ee21ff9adee2e4e8eb7d0328ae606c2fb1e658be51170f620b8d33c9e76b`.

No `graph.json` or named `*.graph.json` document is authored as tracked source.
Model packages under `models/` are ignored build products:

| Generated package | Inputs | Nodes | Outputs | Shape/quantization characteristics | Authoritative producer |
| --- | ---: | ---: | ---: | --- | --- |
| `models/efficientdet_lite0_fp16/graph.json` | 1 | 263 | 2 | static image and detection extents | `make models_efficientdet` → EfficientDet fetch/import pipeline |
| `models/efficientdet_lite0_fp32/graph.json` | 1 | 263 | 2 | static image and detection extents | same producer |
| `models/efficientdet_lite0_int8/graph.json` | 1 | 268 | 2 | static extents and affine tables | same producer |
| `models/tinystories_1m/graph.json` | 2 | 85 | 1 | bounded symbolic sequence extent `S=1..256` | `make models_tinystories` → TinyStories exporter |

The EfficientDet family covers Conv2D, pooling, reshape/resize, concat,
elementwise, and quantized forms. TinyStories covers Embedding, Add, LayerNorm,
MatMul, GELU, and SDPA.

## Ownership rules

| Artifact class | Ownership | Migration rule |
| --- | --- | --- |
| `models/**/graph.json`, model weights, tokenizer products | Generated and ignored | Change the family or generic exporter, then regenerate. Do not patch package JSON by hand. |
| Temporary or benchmark packages | Generated from an authored producer | Change the producer and recreate the package. |
| `dist/`, WASM sidecars, native executables, generated proto/registry projections, shader packs, PTX and embedded arrays | Generated build output | Regenerate only through the owning tool. These are not package-schema sources. |
| `ts/**`, `tools/**`, `native/src/**`, `runtime/typescript/**`, `proto/*.proto`, example tools, and documentation | Authored | Edit directly in the phase that owns the semantic change. |
| Safetensors payload descriptors | Generated from fixed weights | Weight shapes remain concrete. Update graph/schema associations and validators, but do not make weight extents symbolic. |

## Authoritative readers and validators

These surfaces must agree before any package is regenerated:

- `ts/types.ts` defines `GraphDocument` and public tensor descriptors.
- `ts/core/GraphLoader.ts` owns the JavaScript discriminator, shape parsing,
  topology construction, and load diagnostics.
- `ts/core/ModelBuilder.ts` serializes programmatically authored graphs;
  full-profile training publication remains isolated from inference.
- `tools/exporter/runtime_ir.py` is the typed Python importer/serializer.
  Exporter capability, IR, and runtime-tensor modules enforce shapes during
  verification and qualification.
- `tools/validate_model_packages.mjs` owns package discovery, canonical
  filenames, strict JSON and Safetensors checks, topology, output descriptors,
  and affine-table shape checks. Semantic validation proves the declared
  operator/shape domain for WASM, WebGPU, and native CPU before RuntimeIR import.
- `native/src/runtime/engine_runtime_model.inc` parses and materializes the C
  engine graph. `native/src/runtime/public_api.c` independently validates the
  package root and public model/result descriptors.
- `proto/volvoxai.proto` defines the public FFI tensor vocabulary. Generated
  language projections follow that single authored source.
- `native/src/training/quantization_package.c` is a full-profile serializer and
  must not introduce a training dependency into the inference profile.

`ts/index.ts` and `ts/wasm.ts` retain their profile boundary while sharing the
same graph contract.

## Package emitters

- `tools/export_safetensors.py` emits TFLite-derived packages and owns the
  EfficientDet path together with its example import pipeline.
- `tools/exporter/frontend_onnx.py` lowers ONNX into `graph.json`.
- `tools/exporter/runtime_ir.py`, the typed optimizer pipeline, publication
  helpers, and quantization storage preserve symbolic descriptors
  transactionally.
- `tools/exporter/cli.py` and the optimizer command entry expose these paths to
  package authors.
- `ts/core/ModelBuilder.ts`, full-profile quantization publication, and native
  full-profile quantization publication are independent authoring routes and
  must emit the same schema.
- TinyStories is owned by its GPT-Neo Safetensors exporter. TinyReceipt split
  packages are owned by their ONNX import/export pipeline and are regenerated
  from the original source plus calibration inputs.

TinyReceipt artifacts normally live below ignored build paths; they are not
authored graph documents.

## Migration order retained from DS0

1. Freeze symbol grammar, deterministic dimension order, checked limits,
   unified output spelling, and the dynamic-first contract.
2. Change TypeScript logical types/loader/builder, Python RuntimeIR, the native
   parser, the protobuf contract and generated projections, and package
   validators together.
3. Change generic producers: RuntimeIR serialization, ONNX/TFLite frontends,
   optimizer publication, JS builder/PTQ, and native full-profile publication.
4. Migrate example-owned producers and consumers.
5. Regenerate ignored artifacts from their original inputs; never translate or
   hand-edit generated packages.
6. Update contract documentation and both textbook languages atomically.
7. Validate the closed `volvox-graph/v1` schema, unified output descriptors,
   bounded-domain proof, and regenerated packages.
8. Record constant-only and dynamic measurements without changing fixed release
   filenames.

Because the format name remains `volvox-graph/v1`, a global string replacement
cannot prove migration. Closed-root validation and unified output descriptors
are the reliable cutover markers.

## Current build and measurement commands

Use only retained production and baseline entry points:

```bash
npm run typecheck
make build_wasm
make build_native
npm run baseline:batch -- --backend=wasm
npm run baseline:dynamic -- --backend=wasm
npm run baseline:native-dynamic -- --native-build-dir=build/cmake
```

Model-family packages remain reproducible through their production targets:

```bash
make models_efficientdet
make models_tinystories
```

For every measurement, record compile and first-specialization time, warm p50
and p95, logical/capacity/high-water bytes, allocation growth, specialization
cache state, artifact and source fingerprints, and the exact device/provider
identity. Compare active dynamic work with an independently compiled constant
or padded-maximum package where that comparison is meaningful.
