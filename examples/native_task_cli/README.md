# Native Task CLI Example

This opt-in application demonstrates how a native client can build task policy
on top of VolvoxAI's public C APIs. It owns the application-facing behavior
that does not belong in the model-agnostic engine or fixed release commands:

- image binding and normalization policy, using `../native_support/image_io.c`;
- vocabulary and merges-file discovery;
- autoregressive and encoder-decoder loops;
- classification, detection, and CTC postprocessing.

The fixed `native/volvoxai` and `native/volvoxai-full` executables do not link
this source. They expose raw tensor `run`, with generic `train` available only
in the full profile, plus help and version output. `volvoxai-tasks` is an
example binary and is not one of the fixed release artifacts.

## Build

From the repository root:

```bash
make -C examples native_task_cli
```

The executable is written to:

```text
examples/target/bin/volvoxai-tasks
```

## Commands

```bash
examples/target/bin/volvoxai-tasks generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" \
  --max-new 50

examples/target/bin/volvoxai-tasks classify models/classifier \
  --image image=photo.jpg \
  --logits logits \
  --labels labels.txt \
  --top-k 5

examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=photo.jpg \
  --max-det 20
```

The remaining task commands are `ctc`, `seq2seq`, and `chat`. Run the root or
command help for the full option list:

```bash
examples/target/bin/volvoxai-tasks --help
examples/target/bin/volvoxai-tasks seq2seq --help
```

When a model directory is passed, text commands may discover `vocab.bin` and
`merges.txt` there; `--vocab` and `--merges` override those paths. Detection
also discovers `labels.txt` from the model directory and prints a `label`
column; `--labels` overrides that path. Detection tables retain the raw
`score` and add `score_pct` (`score` multiplied by 100) as a human-readable
percentage. Detection reads `boxes` and `scores` by default. Models with other
output names must select them explicitly with `--boxes` and `--scores`; the
application does not guess detection semantics from tensor shapes. Image
commands accept named `--image tensor=file` bindings. When an input declares
`image_normalization` in `config.json`, the application automatically selects
its `zero-one`, `minus-one-one`, or `raw-255` mode. An explicit
`--image-normalize` overrides package metadata for every image binding. If
metadata is absent and no override is provided, the application fails instead
of guessing from dtype. The accepted modes are `zero-one`, `minus-one-one`, and
`raw-255`. This remains example policy, not engine behavior. Every mode
supports F32, U8, and I8 graph inputs. For byte inputs, `raw-255` preserves
physical pixels (U8 stores the pixel directly and I8 stores `pixel - 128`);
normalized modes use the input's declared per-tensor quantization scale and
zero point. The EfficientDet export workflow records `raw-255` for int8 and
`zero-one` for fp16/fp32.

CPU is the default backend. Pass at most one of `--vulkan`, `--opengl`,
`--metal`, or `--nnapi`; an explicitly requested unavailable backend is an
error. The application passes this policy through `VolvoxAIEngineOptions` and
uses only public engine and tokenizer APIs.
