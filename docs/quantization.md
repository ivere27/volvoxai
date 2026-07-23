# Post-training quantization

Runtime-side post-training quantization is a full-profile authoring capability.
The Python exporter/optimizer/PTQ pipeline is an offline development tool, not
an inference dependency. Inference profiles execute already-quantized graphs
but contain no calibration observer, packing, or package-authoring
implementation.

The general offline authoring flow is:

1. losslessly import ONNX or TensorFlow Lite and lower supported semantics to a
   verified F32 RuntimeIR;
2. run portable canonicalization, high-level fusion, and explicit route/input/
   shape specialization;
3. calibrate that exact graph revision with representative, heldout-disjoint
   inputs and complete route/profile coverage;
4. choose a target-measured mixed-precision policy rather than assuming every
   eligible operation should be INT8;
5. derive I8/U8 activation parameters, pack selected weights as per-axis I8,
   and pack optional biases in the I32 accumulator domain;
6. materialize graph and safetensors transactionally, consolidate byte islands,
   apply any explicit terminal output specialization, and verify the result
   differentially before publication; and
7. publish the final graph and safetensors, then qualify runtime
   `CompiledModel` preparation for that exact package revision.

Fuse-before-quantize preserves information that is often lost after Q/DQ
lowering: attention causality, keep-mask meaning, bias placement, and the
high-level region boundary. Post-PTQ optimization is still useful, but only for
integer-domain work such as redundant Q/DQ removal, byte-island propagation,
dead-code elimination, and proven output specialization.

The typed planner can select and materialize only the exact dense topology in
[typed-ptq.md](typed-ptq.md). The lower-level JavaScript authoring API described
below packs values but leaves graph boundaries to the author. Neither surface
mechanically replaces arbitrary F32 operators or guesses how unsupported
normalization, attention, residual, or dynamic-shape paths cross a precision
boundary. Unsupported or unproved cases remain F32 or fail before commit.

## Calibration with the runtime lifecycle

Use a calibration graph that declares every tensor to observe as a graph
output. ExecutionResult publishes only declared outputs; it never exposes
mutable backend intermediates.

~~~javascript
import {
  PTQObserver,
  VolvoxAI,
} from 'volvoxai/full';

const runtime = await VolvoxAI.createRuntime({
  backends: ['webgpu', 'wasm', 'cpu'],
});
const model = runtime.createModel(calibrationGraph);
const compiled = await model.compile({
  backend: {
    mode: 'require',
    backend: 'cpu',
    operatorFallback: 'forbid',
  },
});
const context = await compiled.createContext();

const observers = new Map([
  ['encoder.out', new PTQObserver()],
  ['decoder.hidden', new PTQObserver()],
]);

for (const inputs of calibrationSamples) {
  const result = await context.execute(inputs);
  try {
    for (const [name, observer] of observers) {
      const values = await result.output(name).read();
      if (!(values instanceof Float32Array)) {
        throw new Error(name + ' must be an F32 calibration output');
      }
      observer.observe(values);
    }
  } finally {
    await result.close();
  }
}

await context.close();
await compiled.close();
await model.close();
await runtime.close();
~~~

This path works for host and device outputs because TensorResult.read() returns
a fresh caller-owned typed array. It cannot sample a stale host mirror.

Calibration data should represent the deployment distribution. Run semantic
specialization first: a profile collected from an unspecialized graph is not a
profile for one specialized route. Record sample identity, preprocessing,
exact graph fingerprint, observer policy, selected backend/device, sample
digest/count, and route/profile coverage so package creation is reproducible.
Never reuse the heldout task-score set for calibration.

## Derive activation parameters

~~~javascript
import { derivePTQParameters } from 'volvoxai/full';

const activationParameters = Object.fromEntries(
  [...observers].map(([name, observer]) => [
    name,
    derivePTQParameters(observer, {
      dtype: 'int8',
      scheme: 'symmetric',
    }),
  ]),
);
~~~

Symmetric parameters use:

~~~text
scale = max(abs(minimum), abs(maximum)) / 127
zero_point = 0 for I8
~~~

Asymmetric parameters map an ordered range that includes zero into the complete
I8 or U8 domain with ties-to-even rounding.

PTQObserver rejects non-finite values, counts every observed element, and can
be reset and reused. parameters() is a shorthand for
derivePTQParameters(observer, options).

## Quantize activations

~~~javascript
import { quantizePTQ } from 'volvoxai/full';

const quantized = quantizePTQ(
  sourceValues,
  activationParameters['decoder.hidden'],
);

console.log(quantized.data, quantized.saturationCount);
~~~

Generic affine I8 quantization uses the physical range [-128,127]. Parameters
derived from an observed symmetric range normally map that range into
[-127,127], but an out-of-range value may saturate to -128.

U8 quantization uses [0,255]. saturationCount records values clamped at either
end of the selected domain.

## Pack weights and biases

~~~javascript
import {
  packPTQBias,
  packPTQWeight,
} from 'volvoxai/full';

const packedWeight = packPTQWeight(
  floatWeight,
  [outputChannels, inputChannels],
  {
    axis: 0,
    name: 'projection.weight.i8',
  },
);

const packedBias = packPTQBias(
  floatBias,
  activationParameters['projection.input'].scale,
  packedWeight.scales,
);
~~~

Weights use canonical symmetric per-axis narrow-range I8 storage [-127,127],
balanced around zero point 0. Ranks one through eight and arbitrary valid axes
are supported. For canonical QLinear and QConv2D packages, the output-channel
axis is normally zero.

Bias values use one I32 accumulator scale per output channel:

~~~text
bias_scale[channel] =
  input_activation_scale * weight_scale[channel]
~~~

Packing rejects a non-finite value, invalid shape/axis, mismatched bias channel
count, or a value outside the I32 accumulator domain.

## Materialize safetensors

materializePTQWeights() packs selected F32 graph weights and returns a
SafetensorsFile plus a reference-only quantization table:

~~~javascript
import {
  materializePTQWeights,
} from 'volvoxai/full';

const artifact = materializePTQWeights(trainingGraph, [
  {
    name: 'decoder.proj.weight',
    outputName: 'decoder.proj.weight.i8',
    scaleName: 'decoder.proj.weight.scale',
    zeroPointName: 'decoder.proj.weight.zero_point',
    bias: 'decoder.proj.bias',
    biasOutputName: 'decoder.proj.bias.i32',
    inputScale: activationParameters['decoder.proj.input'].scale,
    axis: 0,
  },
]);

const safetensorsBytes = artifact.weights.toArrayBuffer();
const quantization = artifact.quantization;
~~~

Unselected weights are copied by default. Set includeUnselected: false only
when the destination graph needs none of them. Output tensor names must not
collide with graph inputs, node outputs, or unselected weights.

`artifact.quantization` is the reference-only table for tensors materialized by
this call. Install it directly when it describes every byte tensor in the
destination, or combine its `tensors` entries with separately materialized
activation entries under one root table. The referenced F32 scale tensors and
matching I8 zero-point tensors are already in `artifact.weights`. The author
writes the exact root discriminator:

~~~json
{
  "format": "volvox-graph/v1",
  "quantization": {
    "format": "volvox-affine-safetensors/v1",
    "tensors": {
      "decoder.proj.weight.i8": {
        "scheme": "per_axis",
        "axis": 0,
        "scale_tensor": "decoder.proj.weight.scale",
        "zero_point_tensor": "decoder.proj.weight.zero_point"
      }
    }
  }
}
~~~

No numeric affine scale or zero point is legal in graph JSON or safetensors
metadata. JSON contains tensor names only; the numeric payloads are rank-one
safetensors arrays.

The destination graph explicitly owns:

- QLinear, QGemm, QConv2D, or other selected Q operators;
- input/output tensor names and dtypes;
- activation scale and zero-point tensor references;
- packed-weight per-axis tensor references;
- any QuantizeLinear, RequantizeLinear, or DequantizeLinear boundary;
- the declared graph outputs.

The materializer does not mutate the source Model or publish a training
revision. Package creation produces new bytes; load and compile that package
through a new Runtime/Model lifecycle to validate it.

## Validate the package

~~~bash
make validate_model_packages
~~~

Then compare the quantized model with the F32 reference through named
ExecutionResult outputs and, where the reference executor supports the graph,
compare stable intermediate tensor names to locate the first divergence.
Record the selected provider, provider-reported device identity when available,
graph/weight revision, route evidence, saturation/reconstruction diagnostics,
and per-output error metrics. A successful conversion or a lower node count is
not an accuracy or latency result; qualify task score and paired latency on
untouched heldout inputs with operator fallback forbidden.

Operator-specific dtype, layout, and backend coverage is listed in
[operation_list.md](operation_list.md). The central-reference affine
safetensors contract is documented in
[w8a8-safetensors.md](w8a8-safetensors.md).

## Ownership and profile boundaries

- Calibration contexts own execution and device state.
- Each ExecutionResult owns stable output snapshots.
- Observers own only copied F32 ranges.
- Materializers own newly allocated typed arrays and safetensors bytes.
- Inference entries import and export no PTQ observers or materializers.
- Native inference builds contain no PTQ authoring implementation or public
  PTQ symbol.
- Generated packages use graph.json plus the exact volvox-graph/v1 format.
