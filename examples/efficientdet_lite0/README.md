# EfficientDet Lite0 example

[VolvoxAI and TFLite benchmark comparison](BENCHMARK.md)

This directory owns the EfficientDet-Lite0 acquisition policy and COCO labels.
The reusable ONNX/TFLite lowering code
remains in `tools/export_safetensors.py` and does not infer detection semantics.

The [browser workbench](../efficientdet_lite0.html) runs object detection,
inspects the graph and tensor values, and measures execution time and memory.

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

## Run, debug and profile an image

Build the browser release and serve the repository so the workbench can load
its model packages and matching full-profile runtime:

```bash
make build_web
python3 -m http.server 8000
```

Open <http://localhost:8000/examples/efficientdet_lite0.html>. The INT8 model
and cat sample run automatically. Choose INT8, FP16 or FP32 and WASM CPU or
WebGPU, then choose a sample or upload a PNG/JPEG. The page shows the selected
backend and any whole-model fallback to WASM when WebGPU cannot compile.
Score, class-aware NMS, maximum detections and class filtering update the
retained output immediately. If no score meets the threshold, the original
demo's highest-scoring candidate fallback is labeled explicitly. Box coordinates
use normalized source-image `xmin, ymin, xmax, ymax` values.

Select **Debug**, choose source graph or optimized graph, and start a session.
The debug run uses a copied input from the original image, without drawn boxes,
and its own execution context. Step, continue, set breakpoints, inspect tensor
statistics, or enable bounded value capture. Source graph mode compiles a
separate executable with preserved node boundaries; the ordinary detector
keeps its optimized executable. Releasing the capture frees the debug context
and retained snapshots. Image or model changes also release the old capture.

Select **Performance & memory** and record an inference to collect real engine
node timings, plan metadata and memory observations for an ordinary forward
pass. Export the analysis JSON or Chrome trace, pin a baseline to compare
variants, or inspect a measured node in Debug. Timing includes instrumentation;
the input view labels each last forward as ordinary or instrumented.
The browser workbench is shared with the other model examples; see the
[debugging guide](../../docs/debugging.md) for capture and comparison details.

**Camera** runs detection on successive camera frames. **Stop & keep frame**
keeps the last original frame for an independent debug or profile run. Camera
access requires localhost or HTTPS and the browser's camera permission.

URL options remain available: `backend=wasm|webgpu`,
`precision=int8|fp16|fp32`, `thresh=0.2`, `image=<image-url>`, and
`autorun=zero` for a black 320 × 320 input. For example:

<http://localhost:8000/examples/efficientdet_lite0.html?backend=wasm&precision=fp32>
