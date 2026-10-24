# Model quantization

Quantization chooses weight storage and, optionally, calibrated activation
affines. Compilation independently chooses the arithmetic of packed linear
operators. One grouped weight package can therefore run with F32, F16 or
dynamic I8 arithmetic without being exported again.

## Choose storage and fitting

`WeightFormat` describes stored bits and scale grouping. `WeightQuantization`
adds the algorithm used to fit that format; decoding does not need the algorithm.

| Storage | Configuration | Calibration |
| --- | --- | --- |
| Grouped I2/I3/I4/I8 | `bits=2`, `3`, `4` or `8`, `group_size=32`, `64`, `128` or `256` | Required only for importance fitting |
| Static byte regions | `bits=8`, `group_size=0`, `StaticActivations` | Required |
| Static Conv2/Conv3/Conv4 | `bits=2`, `3` or `4`, `group_size=0`, `StaticActivations`; Conv2D only | Required |

Grouped weights retain F32 graph boundaries and use I32 packed words with F32
scales. RTN uses absolute maximum scales; MSE clipping searches scales to reduce
weight reconstruction error; importance fitting weights that error by observed
input-channel second moments. The diagonal importance objective is a proxy for
output error, so evaluate task quality separately. Static regions use
per-channel I8 RTN weights and I8/U8 activation affines. Conv2D also accepts
per-output-channel I2/I3/I4 RTN, MSE or importance fitting. Conv importance
uses each logical im2col input column's second moment, including convolution
groups, stride, dilation and zero padding. It retains sufficient statistics,
not the images. This is a diagonal weighted-error objective, not full output
reconstruction or GPTQ. Static low-bit Linear/MatMul and grouped static
regions are rejected.
`StaticActivations.reduce_weight_range` restricts per-channel I8 codes to
[-64,64], providing headroom for pairwise U8 × I8 accumulation
(`255 × 64 × 2 = 32640`, within signed I16).
It is rejected for Conv2/Conv3/Conv4, which keep their full signed ranges.

Conv4 stores logical OHWI weights as U8 `[O,ceil(KH*KW*I_per_group/2)]`.
Each byte contains two offset-signed codes (`q+8`), low nibble first; an odd
row's final high nibble is 8. `QConv2D` declares `weight_bits=4` and
`kernel_size=[KH,KW]`. A channel has one F32 scale, a U8 zero point of 8, and
an optional I32 bias. MSE fitting searches 51 clipping ranges from 100% to 50%
of the channel maximum absolute weight, in 1% steps, retaining the wider range
on ties. Activations use source-graph min/max calibration and biases are rounded
against the input scale times the fitted weight scale. Native CPU and WASM
read the packed weights directly; device routes are refused. This halves the
weight code payload, apart from odd row padding, while scales, bias, activations
and other operators keep their own storage costs. Measure accuracy and latency
for the resulting package before choosing it.
Accumulator admission reserves `kernel_elements * max_centered_activation * 8`
plus absolute bias, so exceptionally large biases or kernels can be refused
even when their particular weight values would cancel.

Conv2 uses four `q+2` codes per byte and zero-code tail padding. Conv3 uses a
dense `q+4` bitstream: a code can cross a byte boundary, and unused high bits
of the last byte are zero. Its row size is `ceil(3*KH*KW*I_per_group/8)` bytes.
The weight scale and zero point are per output channel. W2 fitting searches
100% through 1% of its initial range; W3 searches 100% through 50%, both in
1% steps. W4/W8 fitting is unchanged. Low-bit Conv uses static I8/U8 activations;
W2/W3 do not imply A2/A3 arithmetic. Native CPU and WASM support these formats;
native device backends and WebGPU refuse them.

Python defaults to grouped I8 RTN storage. Select a policy explicitly when
publishing a model:

```python
import volvoxai as vx

packed = vx.quantize("model", output="model-i4",
                     weight_bits=4, group_size=128, algorithm="mse-clip")

important = vx.quantize("model", output="model-i4-important",
                        weight_bits=4, algorithm="importance",
                        calibration_data=representative_batches)

static = vx.quantize("model", output="model-static", group_size=0,
    static_activations=vx.pb.StaticActivations(
        dtype=vx.pb.DataType.DATA_TYPE_I8,
        scheme=vx.pb.AffineScheme.AFFINE_SCHEME_SYMMETRIC),
    calibration_data=representative_batches)

# This source contains Conv2D weight operators. Retain other weight families
# explicitly when using this policy on a graph that also contains dense layers.
conv4 = vx.quantize("conv-model", output="model-conv4",
    weight_bits=4, group_size=0, algorithm="mse-clip",
    static_activations=vx.pb.StaticActivations(
        dtype=vx.pb.DataType.DATA_TYPE_I8,
        scheme=vx.pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC),
    calibration_data=representative_batches)
```

Calibration consumes an iterable of complete NumPy batches once. Its necessity
comes from the engine's resolved plan, including rule overrides. Supplying data
when no selected node needs calibration is an error.

On native, add `calibration_backend="cuda"` to run required observations on
CUDA, or leave the default `"cpu"`. The choice is exact: an unsupported CUDA
graph fails without CPU fallback. `result.calibration_backend` reports the
actual observation executor and is empty when no calibration is required.
CUDA reduces activation ranges and input-energy moments on the device; final
weight fitting and package writing remain on the host. This option does not
select inference arithmetic or add GPU inference support to W2/W3 or low-bit
Conv packages. The Python CLI exposes `--calibration-backend cuda`; the native
CLI uses `ptq --cuda`. Sensitivity sweeps can use CUDA calibration while
evaluating all exported candidates consistently on CPU.

The group size partitions each output channel's input features. It changes the
fitted weights and scale count, independently of the backend's execution tile.
The default remains 128. Smaller groups use more scales and can reduce weight
reconstruction error; evaluate held-out task quality as well as latency before
choosing one. Changing an existing artifact's `group_size` does not convert it:
re-export from the floating source. `group_size=0` means the separate per-channel
I8 or Conv-only I2/I3/I4 format and is invalid for `PackedLinear`.

For a logical `[N,K]` matrix, packed words have shape
`[N,ceil(K*bits/32)]`, while F32 scales have shape `[N,ceil(K/G)]`.
W3 codes cross word boundaries; unused high bits in the last word are zero.
Other widths encode zero in unused final-word code slots.
Only the final word is padded; groups do not add packed padding. For complete
groups, weight-plus-scale storage costs `bits + 32/G` bits per weight:

| G | I2 bits/weight | I3 bits/weight | I4 bits/weight | I8 bits/weight |
| --- | ---: | ---: | ---: | ---: |
| 32 | 3 | 4 | 5 | 9 |
| 64 | 2.5 | 3.5 | 4.5 | 8.5 |
| 128 | 2.25 | 3.25 | 4.25 | 8.25 |
| 256 | 2.125 | 3.125 | 4.125 | 8.125 |

Tail words, partial groups, bias, metadata and unchanged tensors add separate
costs. W2/W3 use CPU/WASM implementations with the same F32/F16/dynamic-I8
contracts; their GPU routes are refused. W4/W8 retain their portable GPU support.
CUDA preserves measured G128 specializations and uses a general path for the
other groups; support alone does not imply equal throughput. The
[group and tile design](packed-linear-design.md) describes the storage,
arithmetic, launch invariants and extension workflow.

## Select packed execution arithmetic

```python
with vx.InferenceSession(packed, backend="cpu", packed_arithmetic="f32") as session:
    outputs = session.run(inputs)
```

Both synchronous and asynchronous sessions accept these canonical choices:

| Choice | Contract |
| --- | --- |
| `f32` | F32 operands and accumulation; default on every backend |
| `f16` | Inputs and reconstructed weights round to IEEE F16, then accumulate in F32 |
| `dynamic-i8` | Per-row dynamic I8 activations, exact integer group dots and F32 scaling |
| `f32-fast` | Explicitly permits compensated TF32 high/low activation decomposition with integer weight operands; may use ordinary F32 kernels |

`dynamic-i4` is a reserved selection for future kernels. Every backend currently
refuses it at compilation with `BACKEND_UNSUPPORTED`, before creating an
executable. It never falls back to A8, F16 or F32. This option does not imply
that CUDA hardware support has been implemented by this runtime, and it does
not change stored weights, graph activations, KV caches or static Conv affines.

Weight storage and activation arithmetic are independent. W4 remains packed
under `f32`, `f16` and `dynamic-i8`; requesting A8 permits integer dot kernels
but also incurs activation preparation and group scaling. Choose arithmetic
using numerical and latency measurements for the target backend. A lower
activation bit count alone does not guarantee higher throughput.

The generated API sets `CompileModelRequest.packed_arithmetic` and reports the
resolved value in `CompiledModelHandle.packed_arithmetic`. Unspecified resolves
to F32. Unsupported requested arithmetic fails compilation. Backend fallback
preserves the requested numerical contract. F32-fast never selects F16 or I8
activation rounding. Its compensated mantissa has approximately 21 bits, and
extreme or subnormal operands can overflow or underflow differently from F32.
Use F32 when full operand range and precision are required. F32 permits reassociation and FMA, so different backends
need not produce bit-identical outputs. Static affine regions keep their stored
byte-domain arithmetic.

## Select nodes and mixed weight precision

Rules have explicit selectors: `node_id`, canonical `operator_name`, or
`node_glob`. A rule chooses `weights` or `keep_float`. Rules are evaluated in
order against original source nodes; the first match wins. Unmatched eligible
nodes use the default weights. Globs support `*` and `?`; every selector must
match a source node. Inspect reports resolved actions and format counts for source weight-bearing
Linear/MatMul/Conv2D candidates. Weightless static conversions are excluded from
these counts; `float_nodes` counts original nodes retained in floating point.
A `weights` action requires an eligible operator with immutable weights. Create
rejects plans with no converted weight layer, including all-`keep_float` plans
and selections containing only weightless static operators.

```python
pb = vx.pb
rules = [
    pb.QuantizationRule(node_glob="*.v_proj", weights=pb.WeightQuantization(
        format=pb.WeightFormat(bits=8, group_size=128))),
    pb.QuantizationRule(node_id="output", keep_float=pb.Empty()),
]
package = vx.quantize("model", output="model-mixed", weight_bits=4,
                      algorithm="mse-clip", rules=rules)
```

Use IDs present in your graph. Generic recipes are configuration examples,
not aliases for llama.cpp formats or model-independent quality guarantees.
Retained consumers of a shared weight preserve its original storage; selected
nodes receive the representation their resolved format requires.

## Tune low-bit models explicitly

Start from the original floating model. Use representative inputs for
`algorithm="importance"`, then evaluate task quality on separate data. W2/W3
are opt-in formats; the engine does not promote or exclude sensitive layers.

```python
report = vx.ptq_sensitivity("model", weight_bits=2, group_size=32,
    algorithm="importance", calibration_data=calibration_batches,
    evaluation_data=validation_batches, packed_arithmetic="dynamic-i8",
    modes=["leave-one-higher"], higher_weight_bits=4)

# Application choice after inspecting the report and task metrics.
sensitive_nodes = ["projection"]  # replace with IDs from your model
rules = [pb.QuantizationRule(node_id=node, weights=pb.WeightQuantization(
    format=pb.WeightFormat(bits=4, group_size=32),
    algorithm=pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE))
    for node in sensitive_nodes]
mixed = vx.quantize("model", output="model-w2-w4", weight_bits=2,
    group_size=32, algorithm="importance", rules=rules,
    calibration_data=fresh_calibration_batches)
```

The optional sweep raises one node at a time and reports `higherRecoveryDb`
and `bestToKeepHigher`. It compares graph output SQNR, not application accuracy;
combined layer choices require their own validation. `higher_weight_bits=3`
also permits a W2/W3 comparison. No candidate becomes the default automatically.
The sweep preserves each node's fitting algorithm. Static I8 supports only
RTN, so raising a static MSE/IMPORTANCE node to eight bits is refused; choose
an explicit RTN rule separately when that different comparison is intended.
The CLI exposes `--mode leave-one-higher --higher-weight-bits 4`; the JavaScript
development helper uses `modes` and `higherWeightBits` with the same meaning.

For Conv, select `group_size=0` and `StaticActivations` with the same low-bit
formats and importance policy. A plan has one activation contract. Compose a
static Conv plan and a grouped Linear plan when a model needs both contracts,
as in the receipt examples. Mixed rules operate within each plan's contract.

If PTQ and mixed precision do not meet the model's quality target, optional
[weight-only QAT](model_builder_training.md#optional-weight-only-qat) trains
selected original nodes while simulating W2/W3/W4 weights. QAT policy,
training schedule, validation metric and deployment arithmetic remain explicit
application choices. The engine does not run a tuning loop or retrain a model
when quantization is requested.

## Use the plan lifecycle

Full native and WASM profiles expose **create → calibrate if required → inspect
→ export → release**. A plan owns an immutable snapshot; no caller-owned Model
or Runtime is needed. Source is exactly one `ModelFiles` or `ModelBytes` value.
The current writer accepts one SafeTensors shard with F32/BF16 source weights.
Accepted file and byte sources are copied into the plan's immutable
snapshot; later file or caller-buffer changes do not affect calibration or
export. Length/overflow validation, transport address ranges and actual memory
availability are checked when creating the plan.

```javascript
import { FullEngineHost, VxQuantizationServiceClient, pb } from 'volvoxai';

const host = new FullEngineHost();
const client = new VxQuantizationServiceClient(host);
let plan;
try {
  plan = await client.createQuantizationPlan(new pb.CreateQuantizationPlanRequest({
    bytes: new pb.ModelBytes({graph: graphBytes, weightShards: [weightBytes]}),
    config: new pb.QuantizationConfig({
      weights: new pb.WeightQuantization({
        format: new pb.WeightFormat({bits: 4, groupSize: 128}),
        algorithm: pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP,
      }),
    }),
  }));
  if (plan.info.requiresCalibration) {
    let index = 0;
    for await (const inputs of calibrationBatches) {
      await client.calibrateQuantizationPlan(new pb.CalibrateQuantizationPlanRequest({
        quantizationPlanId: plan.quantizationPlanId,
        sampleName: `batch-${index++}`, sampleCount: 1n, inputs,
      }));
    }
  }
  const {info} = await client.inspectQuantizationPlan(new pb.QuantizationPlanRef(plan));
  if (!info.ready) throw new Error('Calibration coverage is incomplete');
  const result = await client.exportQuantizedModel(new pb.ExportQuantizedModelRequest(plan));
  // result.bytes.graph and result.bytes.weightShards are independent snapshots.
} finally {
  try {
    if (plan) await client.releaseQuantizationPlan(new pb.QuantizationPlanRef(plan));
  } finally { await host.close(); }
}
```

Create returns a handle, data-only info and one top-level operation report.
Calibrate returns only counters, readiness and a report. Inspect returns the
full input contracts, resolved node rules, per-format counts, affines and
coverage. Nested data contains no operation reports.

`CalibrationOptions` supplies named profiles and CPU worker count for either
static or importance calibration. Every profile needs a successful complete
batch; omitted profiles select `default`. Each `sample_name` is unique within
a plan. Rejected batches leave observations unchanged; a failed name can be
retried. Successful batches accumulate sufficient statistics. Export fits from
current statistics without replaying input batches, and further calibration
changes only subsequent exports.

Without output paths, export returns `bytes`; native `output_files` instead
writes `graph.json` and a sibling SafeTensors file. Existing files are never
overwritten. WASM returns bytes for the application to save. Bytes remain valid
after plan release and host close.

## Use the CLI

The following commands use the Python package's CLI. The standalone C
executable also supports PTQ and QAT with raw tensor batch manifests; see
[native CLI authoring](native-cli-authoring.md).

```sh
volvoxai ptq --graph model/graph.json --weights model/model.safetensors \
  --weight-bits 4 --group-size 128 --algorithm mse-clip --out model-i4/

volvoxai ptq --graph model/graph.json --weights model/model.safetensors \
  --weight-bits 8 --group-size 0 --static-activations i8 \
  --affine-scheme asymmetric --calibration calibration/ --out model-static/
```

`--rules rules.json` accepts an array of protobuf JSON `QuantizationRule`
objects, for example `[{"nodeId":"output","keepFloat":{}}]`. Calibration files
are `.npz` batches whose names match model inputs. The sensitivity command also
accepts `--packed-arithmetic` to measure the intended execution contract.

## Convert individual matrices and validate quality

`QuantizeWeight` uses the same format and fitting vocabulary for per-channel I8
RTN or grouped I2/I3/I4/I8 RTN/MSE clipping. Importance fitting uses a model plan.
Its result contains codes, scales, format and the original logical shape, so
partial packed words round-trip through `DequantizeWeight`. Physical Tensor
shapes always describe their actual stored elements. Trainer snapshots are
exported through `VxTrainingService.ExportQuantizedTrainerWeights`.

Compare float and quantized packages on held-out task inputs. Measure prompt
processing and token generation separately on deployment hardware, recording
storage format, fitting algorithm and compiled arithmetic independently.
Integer storage alone does not establish faster execution.

See [static affine authoring](typed-ptq.md), [Python workflows](../python/README.md#post-training-quantization)
and the [complete protobuf contract](../proto/volvoxai.proto).
