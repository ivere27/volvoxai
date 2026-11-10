# Models

The `models/` directory is ignored because generated model packages and source
weights are large. Regenerate examples from public sources when needed.
For the selected SmolLM2/Qwen3 checkpoints, the
[HF decoder workflow](hf-decoder.md) covers local conversion and native CPU
full-prefix inference.
The [compact workflow](compact-decoder.md) adds private KV reuse and typed greedy
generation through the same native runtime.

## Dependencies

Model export is development-only and runs on the host, not in the Docker build image.
Install exporter dependencies once:

```bash
make models_deps
```

This installs the generic ONNX/TFLite dependencies from
`tools/requirements-export.txt` and the family-specific TinyStories exporter
dependencies from `examples/tinystories/requirements-export.txt`.

## Commands

| Command | What it does |
| --- | --- |
| `make models` | Download and export EfficientDet Lite0 fp32/fp16/int8 plus TinyStories-1M. |
| `make models_efficientdet` | Export EfficientDet Lite0 packages from MediaPipe TFLite models. |
| `make models_tinystories` | Export `roneneldan/TinyStories-1M` plus tokenizer assets. |
| `make validate_model_packages` | Reject stale graph filenames and require exact `volvox-graph/v1` roots. |
| `make models_clean` | Remove regenerated example model directories. |

Each model family owns its acquisition policy under `examples/`.
`tools/fetch_models.sh` dispatches `efficientdet`, `tinystories`, or `all`. For
EfficientDet, set `ONLY=int8`, `ONLY=float16`, or `ONLY=float32` to fetch one
precision.

## Piper and Kokoro speech

The [Piper example](../examples/piper/README.md) converts text with eSpeak and
uses a pinned Lessac-low voice. The [Kokoro example](../examples/kokoro/README.md)
accepts explicit phonemes and uses the pinned `af_heart` voice. Both execute
all learned and numerical model stages in VolvoxAI and produce real WAV audio.
ONNX Runtime supplies an independent reference after synthesis.

Build the native shared library before preparing these examples:

```sh
python3 -m pip install -r examples/piper/requirements.txt
make build_native_libraries
python3 examples/piper/prepare.py --text 'Hello world.'
python3 examples/piper/synthesize.py --backend cpu --verify
python3 examples/kokoro/synth.py --backend cpu --validate \
  --output build/tts-examples/kokoro-complete
```

The full pipeline manifests are
`build/tts-examples/piper-full/manifest.json` and
`build/tts-examples/kokoro-complete/manifest.json`. A first Volvox stage predicts
durations; the host reads only the resulting frame count to select the second
stage's shape. Actual encoder outputs and durations remain runtime tensors.
Piper also takes explicit Gaussian noise inputs with the voice's default scales.

Prepared browser and benchmark packages fix the token count and acoustic frame
count. Every request still executes both stages and checks the predicted count.
Prepare a new output directory for a different sentence or duration. The native
preparation/synthesis workflows perform that shape specialization for fresh
requests.

Piper checks individual stages with `rtol=1e-4, atol=1e-5` during preparation and
declares a full-waveform limit of one PCM16 LSB peak error and 0.1 LSB RMS error.
Kokoro executes completely but currently fails the strict waveform comparison;
`--validate` preserves the generated audio and diagnostics and returns nonzero.
Its phase-sensitive spectral path also changes under ONNX Runtime's own
optimizer. These measurements do not establish numerical parity or perceptual
equivalence. See the model READMEs for the exact criteria and diagnostics.

Generated weights and inputs stay under ignored `build/tts-examples/`. The
[TTS benchmark workflow](../examples/tts_common/README.md) covers native CPU/GPU,
WASM and WebGPU, complete-pipeline timing, and explicit reporting of numerical
failures. Model and voice license information is linked from each example.

## EfficientDet Lite0

EfficientDet packages contain:

```text
graph.json
model.safetensors
labels.txt
```

The source TFLite models come from Google's MediaPipe model store, for example:

```text
https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite
```

The exported package has an NHWC image input and named raw outputs:

```text
input0: [1, 320, 320, 3] raw RGB 0..255
scores: [1, 19206, 90]
boxes:  [1, 19206, 4]
```

The COCO label file keeps the 90-slot class-id alignment. `???` entries are unused
placeholders and should not be deleted.

The fetcher, labels, and exporter smoke test live under
[`../examples/efficientdet_lite0/`](../examples/efficientdet_lite0/). The
generic ONNX/TFLite exporter assigns positional output names by default; this
example explicitly names the MediaPipe outputs `scores` and `boxes`.

## TinyStories

GPT-Neo graph construction, tokenizer export, and checkpoint acquisition live
under `examples/tinystories/`; the root exporter accepts ONNX and TFLite
sources and supported local HF decoder directories. GPT-Neo remains handled
by its example-specific exporter. See [`../examples/tinystories/README.md`](../examples/tinystories/README.md)
for direct exporter commands and offline tests.

TinyStories exports:

```text
graph.json
model.safetensors
vocab.bin
merges.txt
tokens.i32
positions.i32
```

The fixed native CLI can execute the package through named raw tensors:

```bash
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32
```

The output is the complete declared logits tensor. The runner remains
vocabulary-agnostic; row selection, tokenization, and sampling belong to the
calling application. The CLI itself is a generated-service client;
`examples/c_api_client_raw.c` is the smaller embedding example.
