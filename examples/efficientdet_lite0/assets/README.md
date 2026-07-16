# EfficientDet Lite0 assets

`dog.jpg` and `cat.jpg` are synthetic, photorealistic
fixtures for manually testing the EfficientDet Lite0 models. Each image is an
exactly 320x320 RGB JPEG, matching the model input without an additional resize.
The generated packages declare `raw-255` normalization for int8 and `zero-one`
for fp16/fp32, and the task CLI applies that metadata automatically.

The detector uses generic COCO classes rather than animal breeds:

| File | Target label | Zero-based class ID |
| --- | --- | ---: |
| `dog.jpg` | `dog` | 17 |
| `cat.jpg` | `cat` | 16 |

The native end-to-end regression uses these as golden prediction fixtures. A
fresh export of the int8 model should rank `dog` at approximately 91.8% and
`cat` at approximately 80.9% confidence.

Generation prompts, summarized:

- `dog.jpg`: one photorealistic adult cream-white dog, full body,
  centered against a simple natural outdoor background in soft daylight.
- `cat.jpg`: one photorealistic adult brown mackerel-tabby
  domestic shorthair with white chest and paws, full body, centered
  against a simple courtyard background in soft daylight.

Run the native detector from the repository root:

```bash
examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=examples/efficientdet_lite0/assets/dog.jpg \
  --max-det 5
```

Replace `dog.jpg` with `cat.jpg` for the cat fixture.
The model directory's `labels.txt` is discovered automatically, so the table
includes both the numeric `class` and its `label`. It also shows the raw
`score` and a human-readable `score_pct`, such as `91.80%`.

For either floating-point package, only change the model directory, for example:

```bash
examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_fp16 \
  --image input0=examples/efficientdet_lite0/assets/cat.jpg \
  --max-det 5
```
