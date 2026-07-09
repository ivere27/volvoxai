# Models

The `models/` directory is ignored because generated model packages and source
weights are large. Regenerate examples from public sources when needed.

## Dependencies

Model export is development-only and runs on the host, not in the Docker build image.
Install exporter dependencies once:

```bash
make models_deps
```

This installs `tools/requirements-export.txt`, including numpy, flatbuffers,
safetensors, torch, and transformers.

## Commands

| Command | What it does |
| --- | --- |
| `make models` | Download and export EfficientDet Lite0 fp32/fp16/int8 plus TinyStories-1M. |
| `make models_efficientdet` | Export EfficientDet Lite0 packages from MediaPipe TFLite models. |
| `make models_tinystories` | Export `roneneldan/TinyStories-1M` plus tokenizer assets. |
| `make models_clean` | Remove regenerated example model directories. |

`tools/fetch_models.sh` accepts `efficientdet`, `tinystories`, or `all`. For
EfficientDet, set `ONLY=int8`, `ONLY=float16`, or `ONLY=float32` to fetch one
precision.

## EfficientDet Lite0

EfficientDet packages contain:

```text
config.json
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

## TinyStories

TinyStories exports:

```text
config.json
model.safetensors
vocab.bin
merges.txt
tokens.i32
positions.i32
```

The native runtime can use the package directly:

```bash
./native/volvoxai generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" \
  --max-new 50
```

## TinyReceiptKIE

Custom TinyReceiptKIE checkpoints can be exported into the same package layout:

```bash
python3 tools/export_kie_safetensors.py \
  --checkpoint checkpoint.pt \
  --out models/tiny_receipt_kie
```

Then run it through the native chat alias:

```bash
./native/volvoxai chat models/tiny_receipt_kie \
  --image image=receipt.jpg \
  --prompt "What is the first number of the store's phone number?" \
  --family auto \
  --max-new 128
```

The package declares the `tiny_receipt_kie` chat interface in `config.json` and keeps
weights in `model.safetensors`. LoRA is folded into dense linear weights during
export. If the checkpoint is a base model plus components, add `--lora lora.pt` and
repeat `--adapter family=adapter_family.pt` for each adapter family.

