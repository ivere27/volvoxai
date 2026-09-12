# Native Task CLI Example

This opt-in repository application demonstrates where native task policy
belongs. It uses only the generated C service API and lite protobuf messages
from `proto/volvoxai.proto`; it never reaches the private native lifecycle.
[`examples/c_api_client_raw.c`](../c_api_client_raw.c) is the smaller raw
embedding example.

The example owns behavior that does not belong in the model-agnostic engine or
fixed release commands:

- image binding and normalization policy, using `../native_support/image_io.c`;
- raw named-tensor file binding;
- classification and detection postprocessing; and
- generic decode operations over one retained execution context.

The fixed `native/volvoxai` and `native/volvoxai-full` executables do not link
this source. They expose the model-neutral raw-tensor runtime; `volvoxai-tasks`
is an example binary and is not one of the fixed release artifacts.

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
examples/target/bin/volvoxai-tasks run models/my_model \
  --input input=input.f32 \
  --output logits=logits.f32

examples/target/bin/volvoxai-tasks classify models/classifier \
  --image image=photo.jpg \
  --image-normalize zero-one \
  --logits logits \
  --labels labels.txt \
  --top-k 5

examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=photo.jpg \
  --image-normalize raw-255 \
  --max-det 20
```

Detection benchmarking accepts `--warmup_runs` and `--num_runs`. Add
`--include_transfers` to call the generated `GetResult` and `ReadOutput`
operations for every declared output inside each warmup and timed iteration.
That measures the complete public in-process path through host materialization
without including output file writes.

The fourth command is `decode`, which exposes model-neutral prefill and step
operations without assigning token semantics. `--prefill-position` names the
final active prompt position (zero for a one-token prompt). Each subsequent
step omits an explicit position so the retained context advances from that
point according to the public protobuf contract. Run the root or command help
for the exact options:

```bash
examples/target/bin/volvoxai-tasks --help
examples/target/bin/volvoxai-tasks decode --help
```

When a model directory is passed, detection discovers `labels.txt` and prints
a `label` column; `--labels` overrides that path. Detection tables retain the raw
`score` and add `score_pct` (`score` multiplied by 100) as a human-readable
percentage. Detection reads `boxes` and `scores` by default. Models with other
output names must select them explicitly with `--boxes` and `--scores`; the
application does not guess detection semantics from tensor shapes. Image
commands accept named `--image tensor=file` bindings. Image preprocessing is
application policy and is not legal graph metadata in the closed dynamic-v1
schema. Every image command therefore requires an explicit
`--image-normalize zero-one`, `minus-one-one`, or `raw-255`; the application
fails instead of guessing from dtype. Every mode supports F32, U8, and I8 graph
inputs. For byte inputs, `raw-255` preserves physical pixels (U8 stores the
pixel directly and I8 stores `pixel - 128`); normalized modes use the input's
declared per-tensor quantization scale and zero point. Use `raw-255` for the
int8 EfficientDet package and `zero-one` for its fp16/fp32 variants.

Fixed inputs may use `--input name=file` or `--image name=file`. A dynamic
input requires its concrete shape, for example
`--input 'tokens[3,128]=tokens.i32'`. Shape is validated against the logical
rank, bounds, symbol equality, and exact file byte length before execution.

CPU is the default backend. Pass at most one of `--vulkan`, `--opengl`,
`--metal`, or `--cuda`; an explicitly requested unavailable backend is an
error. The application translates this choice into the generated
`BackendPolicy` message used by every client. See the
[native CUDA status](../../docs/cuda.md) for CUDA build composition, strict
routing, operator limits, and RTX 3090 benchmark scope.
