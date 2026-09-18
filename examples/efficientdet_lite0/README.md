# EfficientDet Lite0 example

[VolvoxAI and TFLite benchmark comparison](BENCHMARK.md)

This directory owns the EfficientDet-Lite0 acquisition policy and COCO labels.
The reusable ONNX/TFLite lowering code
remains in `tools/export_safetensors.py` and does not infer detection semantics.

The browser demo remains at [`../efficientdet_lite0.html`](../efficientdet_lite0.html).

Synthetic 320x320 dog and cat sample images live in
[`assets/`](assets/README.md).

## Export packages

Install the generic exporter dependencies and fetch all three MediaPipe model
variants:

```bash
make models_deps
make models_efficientdet
```

To export only one variant, set `ONLY` to `int8`, `float16`, or `float32`:

```bash
ONLY=int8 make models_efficientdet
```

The example fetcher passes explicit `scores` and `boxes` output names to the
generic exporter. Output meaning belongs here; the generic exporter otherwise
uses positional names such as `output0` and `output1`.
Image decoding and normalization are application policy, not graph metadata.
Use `raw-255` for int8 and `zero-one` for fp16/fp32 in the calling application.
