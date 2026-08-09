# EfficientDet Lite0 example

This directory owns the EfficientDet-Lite0 acquisition policy, COCO labels,
and model-specific exporter smoke test. The reusable ONNX/TFLite lowering code
remains in `tools/export_safetensors.py` and does not infer detection semantics.

The browser demo remains at [`../efficientdet_lite0.html`](../efficientdet_lite0.html).

Synthetic 320x320 dog and cat test images and a native test command live in
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

## Test

Fetch the source models and build the native task CLI before running all of the
example tests:

```bash
make models_efficientdet
make build_native_task_cli
```

Then run the exporter contract and native JPEG inference regressions:

```bash
python3 -m unittest discover \
  -s examples/efficientdet_lite0/tests \
  -p 'test_*.py' -v
```

The end-to-end regressions export each TFLite precision into a temporary
directory; they do not commit or overwrite files under `models/`. They expect
the dog and cat fixtures to rank as their matching COCO classes. Tests that need
exporter dependencies, an ignored source model, or the opt-in native task CLI
are skipped when those prerequisites are absent. To run only the int8 tests,
`ONLY=int8 make models_efficientdet` is sufficient; the missing float-source
tests are skipped.
