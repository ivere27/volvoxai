# Post-training quantization

Post-training quantization (PTQ) turns a trained float model into a smaller
model with integer weights. W8A8 calibrates activation ranges from representative
inputs. Packed modes keep F32 graph boundaries and round activations inside each
Linear operation, using FP16 or dynamic per-row INT8 arithmetic.

The workflow is **prepare a float model → calibrate → inspect coverage → write
an integer package → compare accuracy and latency**. A successful conversion
does not establish that the smaller model is accurate enough for your task.

PTQ authoring requires full. Both native full and browser/Node full WASM support
template creation, calibration, inspection, and package export. Inference
profiles can run supported quantized graphs without including calibration code.

## Choose data and precision boundaries

Use inputs representative of deployment, with the same preprocessing and shapes.
Keep calibration and held-out evaluation sets separate. Empty images or an
unrepresentative prompt set can produce narrow ranges that saturate in use.
For several declared calibration profiles, collect successful samples for each.

Weights normally use per-output-channel symmetric I8 scales. Activations default
to symmetric I8; choose asymmetric U8 when that is the intended numerical
contract. Sensitive operations can remain F32. Template authoring exposes
`floatOperators`, `floatNodes`, and `selectedNodes` so this policy is explicit.
It does not guess which precision split will preserve your model's accuracy.

Choose a precision explicitly with Python's `vx.quantize(..., precision=...)`,
`volvoxai ptq --precision ...`, or the generated API's
`PtqAuthoringConfig.mode`:

| Precision | Weight representation | Activation arithmetic | Authored region |
| --- | --- | --- | --- |
| `w8a8` (default) | Per-output-channel INT8 | Calibrated I8/U8 affines | Supported byte-domain operators, including QLinear/QConv2D |
| `w4a8` | Group-128 INT4 | Dynamic symmetric INT8 per row | Immutable Linear/MatMul weights |
| `w4a16` | Group-128 INT4 | IEEE FP16 operands, F32 accumulation | Immutable Linear/MatMul weights |
| `w8a16` | Group-128 INT8 | IEEE FP16 operands, F32 accumulation | Immutable Linear/MatMul weights |

The three packed modes use the same `PackedLinear` operator. Bias, residuals,
embeddings and other retained operations remain F32. `float_nodes` and
`selected_nodes` allow a deliberate precision boundary. Packed modes require
the default symmetric settings and do not accept `reduce_range`. Their
calibration still validates finite representative inputs and records profile
coverage; weight scales use per-group maximum magnitude, and A8 activation
scales are computed during execution. Calibration does not turn this
round-to-nearest quantizer into AWQ or GPTQ.

For example, with an existing package and `.npz` input batches:

```sh
volvoxai ptq --graph model/graph.json --weights model/model.safetensors \
  --calibration calibration/ --out model-w4a16/ --precision w4a16
```

The writer preserves an original weight if an embedding or another retained
node still uses it. A tied embedding therefore limits the total storage saving.
Small upstream floating-point differences can cross a quantization rounding
boundary and change later logits or generated tokens. Evaluate the exported
model on held-out tasks and on its intended backend; operator correctness alone
does not establish whole-model quality or agreement across devices.

For performance comparisons, match the source checkpoint, quantized layers,
prompt tokens, generated length, cache precision and context capacity. A weight
format name alone does not specify activation arithmetic: llama.cpp's CUDA
`Q4_0` and `Q8_0` routes can quantize activations to INT8, while their weight
groups and scales also differ from these packed modes. Report those differences
and measure prompt processing and token generation separately. Short prompts
and long held-out text exercise different kernel and attention costs.

For the math behind scaling, rounding, and saturation, read the
[textbook's quantization chapters](textbook/06-precision-and-quantization.md).

## Calibrate and export a small model

This complete example builds a two-input linear layer, observes two samples,
and returns a W8A8 package. A real application supplies its trained graph and
weight bytes in place of this toy model and a representative calibration set.
Save the example as an `.mjs` file in the repository or an installed project.

```javascript
import {
  FullEngineHost, VxInferenceServiceClient, VxPlanningServiceClient,
  VxQuantizationServiceClient, pb,
} from 'volvoxai';

const host = new FullEngineHost();
const inference = new VxInferenceServiceClient(host);
const planning = new VxPlanningServiceClient(host);
const quantization = new VxQuantizationServiceClient(host);
const f32 = (name, values, shape) => new pb.Tensor({
  name, dtype: pb.DataType.DATA_TYPE_F32, shape,
  inline: new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
});
try {
  const graphBytes = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1', dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1, 2] } },
    nodes: [{
      id: 'dense', opType: 'Linear',
      inputs: { input: 'x', weight: 'dense.weight', bias: 'dense.bias' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 2] } },
      params: { weight_layout: 'dout_din' },
    }],
    outputs: ['y'],
    execution: {stateless: {}, entrypoints: [{name: 'evaluate', effect: 1, allowed_ranges: [1], inputs: ['x'], outputs: ['y']}]},
  }));
  const weights = await planning.writeSafetensors(new pb.WriteSafetensorsRequest({
    edits: [new pb.SafetensorsEdit({
      setTensor: f32('dense.weight', Float32Array.of(1, 2, 3, 4), [2n, 2n]),
    }), new pb.SafetensorsEdit({
      setTensor: f32('dense.bias', Float32Array.of(0.25, -0.5), [2n]),
    })],
  }));
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    package: new pb.ModelPackage({ graphDocument: graphBytes, weightShards: [weights.data] }),
  }));
  const template = await quantization.authorPtqTemplate(
    new pb.AuthorPtqTemplateRequest({
      sourceGraph: graphBytes, weightShards: [weights.data],
      config: new pb.PtqAuthoringConfig({
        activationDtype: pb.DataType.DATA_TYPE_U8,
        activationScheme: pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC,
        weightDtype: pb.DataType.DATA_TYPE_I8,
      }),
    }));
  const plan = await quantization.createPtqPlan(new pb.CreatePtqPlanRequest({
    modelId: model.modelId,
    templateGraph: template.templateGraph,
    observers: template.observers, layers: template.layers,
    profileNames: ['default'],
  }));
  const samples = [Float32Array.of(1, -0.5), Float32Array.of(-1, 0.75)];
  for (const [index, values] of samples.entries()) {
    await quantization.calibratePtqPlan(new pb.CalibratePtqPlanRequest({
      ptqPlanId: plan.ptqPlanId, profileName: 'default',
      sampleName: `sample-${index}`, sampleCount: 1n,
      inputs: [f32('x', values, [1n, 2n])],
    }));
  }
  const observed = await quantization.inspectPtqPlan(new pb.PtqPlanRef(plan));
  console.log({
    batches: observed.calibrationBatches,
    samples: observed.calibrationSamples,
    coverageComplete: observed.coverage.complete,
  });
  const packed = await quantization.writePtqPackage(
    new pb.WritePtqPackageRequest({ ptqPlanId: plan.ptqPlanId }));
  // Save packed.graph as graph.json and packed.weights as model.safetensors.
  const quantized = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    package: new pb.ModelPackage({
      graphDocument: packed.graph, weightShards: [packed.weights],
    }),
  }));
  console.log('Quantized model:', quantized.modelId);
} finally {
  await host.close();
}
```

`AuthorPtqTemplate` identifies supported float regions and returns observer and
layer records. `CreatePtqPlan` binds that template to the exact loaded Model
revision. Each `CalibratePtqPlan` request supplies every model input with its
explicit dtype, shape, and bytes. Ranges commit only after the entire sample
and all observations succeed.

## Inspect coverage before publishing

`InspectPtqPlan` reports ranges, derived parameters, and coverage. Distinguish
logical batches, represented samples, concrete shape signatures, symbol extrema,
and observed activation elements; they measure different aspects of calibration.
Every declared profile needs a successful sample before package writing.

The current observer uses min/max ranges. It is straightforward to inspect,
but outliers can widen a range and reduce precision for common inputs. More
samples help only when they improve representation of the real workload.
Changing a precision boundary or selecting a better calibration set should be
an explicit experiment measured against held-out quality.

The plan retains the model revision it observes. Training a successor or
releasing the original public Model ID does not silently retarget the plan.
Create a new plan when calibrating a different graph or weight revision.

## Save and reload the package

Empty output paths return `packed.graph` and `packed.weights` as owned byte
arrays. They remain usable after releasing the plan. In Node, save them in a
new output directory with your filesystem code; in a browser, use a download or
application storage. Load those bytes directly as `ModelPackage`, as above.

Native callers can request graph and weight output paths. Browser output-path
requests return `TRANSPORT_UNSUPPORTED`; this restriction does not prevent
browser calibration or byte-based export. When supplying template-authoring
source bytes, those bytes take precedence over source paths. Keep source forms
consistent so the intended package is unambiguous.

In a long-running session, call `ReleasePtqPlan` when calibration is finished.
A plan does not mutate the source package or any already compiled revision.
The current package writer accepts one source SafeTensors shard with F32 or
BF16 weights. BF16 is widened exactly for calibration; no intermediate F32
checkpoint is required. F16 source authoring is not supported by this writer.
Broader SafeTensors storage operations and model loading have their own shard
and dtype rules.

## Numeric and storage rules

For an affine activation, the real value is approximately
`scale * (integer - zero_point)`. Scale must be positive, and the integer is
rounded and saturated to its declared dtype's range. A per-channel weight
uses a separate scale for each output channel so one large channel does not
reduce precision for every other channel.

- Weights use symmetric I8 with an explicit output-channel axis. `reduceRange`
  requests seven-bit magnitude when that is the chosen contract.
- Bias packing uses I32 accumulator units derived from input and weight scales.
- Scale and zero-point values live in SafeTensors tensors. Graph JSON references
  those tensors and declares the Q/DQ boundaries.
- Rounding, saturation, axes, dtypes, and bias interpretation must agree across
  every backend on which you qualify the package.

[W8A8 SafeTensors](w8a8-safetensors.md) specifies the numerical and storage
contract. [Typed PTQ](typed-ptq.md) describes the supported authoring subset and
rejection rules; [operator support](operation_list.md) describes execution.

## Validate the result

Run the float and quantized models on the same held-out inputs. Compare every
required output, then evaluate the actual task metric: classification accuracy,
detection quality, exact-match answers, or generated-token behavior. Check
saturation and sensitive layers when quality drops before changing tolerances.

Measure latency and memory on the deployment device using identical timing
boundaries and warm-up. I8 storage alone is not proof of faster execution; the
selected backend must have the appropriate integer routes. Keep the source
model, calibration inputs, configuration, output package, and measurements
identified together so the experiment can be repeated.

See [profiling](profiling.md), [testing](testing.md), and the
[training/PTQ matrix](training-ptq-runtime-matrix.md) for measurement and coverage.
