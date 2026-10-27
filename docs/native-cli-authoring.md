# Quantization and QAT from the native CLI

The full `native/volvoxai` executable can fit a quantized model, train FP32
weights with an explicit QAT policy, and resume a training checkpoint. These
commands use the same generated public services as Python. The inference-only
`native/volvoxai-lite` executable supports `run`, including packed arithmetic
selection, but contains no PTQ or training implementation.

PTQ changes stored weights. `run --packed-arithmetic` chooses how a
`PackedLinear` executes. To run a packed W2/W3/W4 model with per-row INT8
activations, explicitly select `dynamic-i8`; the default remains F32 operands.
QAT simulates selected low-bit **weights**, keeps FP32 master weights, and does
not simulate INT8 activations. See [quantization](quantization.md) and
[training](model_builder_training.md) for the numerical contracts.

## Fit a packed model

```sh
native/volvoxai ptq source-model \
  --weight-bits 2 --group-size 32 --algorithm importance \
  --calibration calibration.json --threads 1 \
  --out packed-w2 --report-json ptq-report.json
```

A model directory contains `graph.json` and `model.safetensors`. Alternatively,
pass a graph path and `--weights FILE`. The current public PTQ service accepts
one SafeTensors shard. The engine snapshots the source when it creates the
plan. Calibration streams each batch once; export uses C-owned statistics and
does not replay the calibration files.

Weight bits are 2, 3, 4 or 8. Group sizes are 32, 64, 128 or 256 for packed
Linear weights. `--algorithm` accepts `rtn`, `mse-clip` or `importance`.
Importance requires calibration. Static byte-domain quantization also requires
calibration and uses group size zero; select it with
`--static-activations i8` or `u8` (asymmetric affines). Static low-bit weights
are supported for Conv2D. The engine rejects incompatible combinations.

`ptq --cpu` (the default) or `ptq --cuda` chooses the private calibration
executor. A CUDA request never falls back to CPU. The report's
`plan.calibrationBackend` records the actual observation backend; it is empty
for grouped RTN/MSE plans, which need no inference executor. Weight fitting and
package serialization remain host work. Calibration executes all required
observations and may omit an unrelated graph suffix. This does not establish
that the exported quantized graph supports CUDA inference.

Without explicit options the packed default is W4/group32/MSE_CLIP. With
`--static-activations`, the omitted group size becomes zero and omitted
algorithm becomes RTN. The bit width remains four unless supplied.

A calibration file describes raw tensors, not images or compressed arrays:

```json
{
  "batches": [
    {"inputs": ["input0[1,1,320,672]=calibration/000.f32"], "sampleName": "000", "sampleCount": 1},
    {"inputs": ["input0[1,1,320,672]=calibration/001.f32"], "sampleName": "001", "sampleCount": 1}
  ]
}
```

Paths inside a manifest are relative to that manifest. Each batch must bind
every graph input exactly once. Use `.f32`, `.i32`, `.i8` or `.u8` raw files
matching the declared dtype. Dynamic inputs require concrete dimensions in
`name[d0,d1,...]=file`; fixed inputs may use `name=file`. The existing CLI
binding limits are 32 inputs and rank eight. There is no fixed JSON byte-size
cap; file/address-space representability, allocation and parser validity are
checked.

`sampleName` defaults to a unique `batch-N` name and `sampleCount` to one.
Optional `profileName` declares named coverage; the engine must observe every
listed profile. Sample names must be unique. Supplied counts describe logical
examples, not the number of tensor values.

## Choose original nodes explicitly

`--rules rules.json` accepts the same array of public `QuantizationRule`
objects as the Python CLI. First matching rule wins; selection is resolved in
the engine against original node IDs before optimization. The CLI implements
no additional node matcher. For example:

```json
[
  {"nodeId": "node_125", "weights": {"format": {"bits": 4, "groupSize": 32}, "algorithm": "WEIGHT_ALGORITHM_MSE_CLIP"}},
  {"operatorName": "Conv2D", "keepFloat": {}}
]
```

A selector is exactly one of `nodeId`, `operatorName` or `nodeGlob`. The action
is exactly one of `weights` or the empty object `keepFloat`. Protobuf snake_case
names also work. Unknown keys, duplicate keys, duplicate snake/camel aliases,
malformed policies and selectors matching no source node are refused. JSON
algorithm values use the official enum names (or their defined integer values);
omitting `algorithm` selects the protobuf default RTN. Unknown names and values
are rejected. The short algorithm names above are command-line options. An optional
`{"rules": [...]}` wrapper is accepted, though the array is shared with Python.

Inspect a policy without calibration or export:

```sh
native/volvoxai ptq source-model --weight-bits 2 --group-size 32 \
  --algorithm importance --rules rules.json --dry-run \
  --report-json resolved-policy.json
```

The report contains the engine's resolved eligible-node policies, whether
calibration is required, readiness, and operation reports. A no-op policy with
no eligible quantized node is refused by the public service.

## Keep Conv8 and make only Linear weights W2

For the receipt digit graph, low-bit Linear fitting must precede static Conv
calibration. Use `linear-rules.json` containing
`[{"operatorName":"Conv2D","keepFloat":{}}]`, and
`conv-rules.json` containing
`[{"operatorName":"PackedLinear","keepFloat":{}}]`:

```sh
native/volvoxai ptq fp32-prepared \
  --weight-bits 2 --group-size 32 --algorithm importance \
  --rules linear-rules.json --calibration calibration.json --threads 1 \
  --out linear2 --report-json linear2-report.json

native/volvoxai ptq linear2 \
  --weight-bits 8 --group-size 0 --algorithm rtn \
  --static-activations i8 --rules conv-rules.json \
  --calibration calibration.json --threads 1 \
  --out conv8-linear2 --report-json mixed-report.json

native/volvoxai run conv8-linear2 --cpu --threads 1 \
  --packed-arithmetic dynamic-i8 \
  --input 'input0[1,1,320,672]=example.f32' \
  --output slot_logits=logits.f32 --report-json run-report.json
```

This is Conv8 plus Linear2 with dynamic-I8 Linear activations. It does not
quantize attention matrix products, Softmax, normalization or embeddings.
Changing the first stage to `--weight-bits 3` makes true dense packed I3 weights.
To retain selected Linear nodes at four bits, put their explicit weight rules
before the operator rule. Both stages return their own resolved plan and report.

`run --packed-arithmetic` accepts `f32`, `f16`, `dynamic-i8`, `f32-fast` and
`dynamic-i4`. The report records both the requested and resolved arithmetic.
`dynamic-i4` is reserved and is refused during compilation; it never silently
selects another arithmetic. CPU/WASM support packed W2/W3; GPU routes do not.
A storage package does not itself select its activation arithmetic.

## Train with QAT and resume

Start from an original FP32 graph and weights. A QAT policy is a public
`QatOptions` object with explicit original node IDs:

```json
{
  "nodes": [
    {"nodeId": "projection", "weights": {"format": {"bits": 2, "groupSize": 32}, "algorithm": "WEIGHT_ALGORITHM_MSE_CLIP"}},
    {"nodeId": "head", "weights": {"format": {"bits": 4, "groupSize": 32}, "algorithm": "WEIGHT_ALGORITHM_MSE_CLIP"}}
  ]
}
```

Use the node IDs and weight tensor names from your source graph. Supported
QAT widths are 2, 3 and 4. Conv2D uses group size zero; Linear uses supported
packed groups. RTN and MSE_CLIP are supported. IMPORTANCE is a PTQ fitting
algorithm, not a QAT option. QAT supports CPU/WASM and native CUDA. Other GPU
backends are refused. Use `train --cuda` to keep fake weights, gradients and
optimizer work on CUDA; unsupported forward/backward graphs fail explicitly.

Conv and Linear policies are independently optional:

| Operator | PTQ weight bits | QAT weight bits | Scale grouping |
| --- | --- | --- | --- |
| Conv2D | 2, 3, 4, 8 | 2, 3, 4 | Per output channel, `groupSize: 0` |
| Linear/MatMul | 2, 3, 4, 8 | 2, 3, 4 | Groups of 32, 64, 128 or 256 input features |

For example, this QAT policy chooses Conv2 plus Linear3 and leaves every other
node without fake quantization:

```json
{"nodes": [
  {"nodeId": "conv_body", "weights": {"format": {"bits": 2, "groupSize": 0}, "algorithm": "WEIGHT_ALGORITHM_MSE_CLIP"}},
  {"nodeId": "linear_head", "weights": {"format": {"bits": 3, "groupSize": 32}, "algorithm": "WEIGHT_ALGORITHM_MSE_CLIP"}}
]}
```

Use `--trainable` to select which FP32 masters receive updates; this is separate
from the QAT node policy. Repeat it for every selected weight; there is no fixed
trainable-count limit. Duplicate and empty names are rejected. To try Conv4,
Linear2 or a sensitive Linear4 head,
change the corresponding node's bits. For mixed PTQ, fit grouped Linear weights
first, then calibrate static Conv weights as in the two-stage example above.
Static Conv2/3/4 accepts RTN, MSE_CLIP or IMPORTANCE; static Conv8 accepts RTN.
QAT uses RTN or MSE_CLIP. Activation QAT and dynamic-I4 execution remain
unsupported. CUDA authoring does not add packed W2/W3 or low-bit Conv GPU
inference; exported CPU/WASM packages retain their existing execution contract.

A training manifest supplies varying input and cross-entropy target files:

```json
{
  "batches": [
    {"inputs": ["x[2,33]=training/batch0.f32"], "targets": "training/batch0.i32"},
    {"inputs": ["x[1,33]=training/batch1.f32"], "targets": "training/batch1.i32"}
  ]
}
```

Targets are raw I32 class indices, including an optional ignored index
(default `-100`). Shape/range validation and loss calculation are in C.
Batch files are traversed in listed order for every epoch; the CLI performs no
shuffle, augmentation, automatic split or layer selection.

```sh
native/volvoxai train source-model --cpu --threads 1 \
  --qat qat.json --batches training.json --epochs 2 \
  --logits logits --trainable projection.weight --trainable head.weight \
  --optimizer adamw --learning-rate 0.001 \
  --output-weights trained.safetensors --checkpoint-out trained.pb \
  --report-json train-report.json

native/volvoxai train source-model --cpu --threads 1 \
  --checkpoint-in trained.pb --batches training.json --epochs 1 \
  --logits logits --trainable projection.weight --trainable head.weight \
  --output-weights resumed.safetensors --checkpoint-out resumed.pb \
  --report-json resume-report.json
```

A checkpoint is the generated binary `TrainerCheckpoint` protobuf: exact graph,
FP32 master weights, optimizer state/configuration, optimizer step, RNG and QAT
policy. Resume still requires the compatible original model descriptor. Omitted
QAT and optimizer options inherit the checkpoint; an explicitly supplied QAT
policy must match. A fresh invocation retains the native CLI's SGD default.
An explicit optimizer flag overrides only that supplied option on resume.

For a batch that exceeds the default 512 MiB C activation arena limit, pass
`--max-activation-capacity-bytes 4294967296` to allow up to 4 GiB. The value is
a positive integer number of bytes. Parameters, gradients, optimizer state and
backend workspace consume separate memory. Supply the option again on resume;
the activation policy is not part of a checkpoint. Python accepts the same
policy through `TrainingSession(shape_options=pb.TrainerShapeOptions(...))`.

The older fixed `--input` plus `--targets` form remains available with
`--microbatches N`. It cannot be mixed with `--batches`. Fixed-input gradient
accumulation normalizes by the actual non-ignored target count and the actual
length of the current window, including a final short window. Batch manifests
currently require `--accumulation-steps 1`; the CLI refuses larger values before
training rather than using the wrong denominator for varying batches. A proven
all-ignored final window is explicitly reset through the public service. If
there was no private optimizer update, the CLI exports unchanged state without
claiming a commit.

Training exports FP32 masters. Fit those weights through `ptq` afterward using
the same graph and `--weights trained.safetensors`, then explicitly select
`run --packed-arithmetic dynamic-i8` for deployment. `train` rejects the run-only
`--packed-arithmetic` option because weight-only QAT does not simulate A8.

## Files, reports and failures

PTQ packages, training weight exports, checkpoints and authoring reports must
use new file paths. Existing files and symlinks are refused; there is no implicit
overwrite option. The CLI never rewrites the source model. An export failure
can leave newly created partial outputs; inspect the failure report and use a
fresh destination for retry. This does not change the existing `run` output-file
behavior.

Authoring reports include operation reports, resolved policy or trainer state,
calibration counters, and per-step losses/optimizer steps. Reports distinguish
weight-only QAT from packed execution. They are observations, not automatic
accuracy or speed qualification. Raw tensors, preprocessing and evaluation
remain the caller's responsibility.

The small regression suite exercises actual exported I2/I3/I4/I8 inference,
independent dynamic-I8 integer-dot references, Conv2/Conv3 export, mixed QAT,
varying batches, exact AdamW checkpoint continuation, default SGD, partial
accumulation, and refusal/file-preservation paths:

```sh
VOLVOXAI_NATIVE_CLI="$PWD/native/volvoxai" \
  python -m unittest discover -s tools/tests -p test_native_authoring_cli.py -v
```
