# Post-training quantization

Post-training quantization (PTQ) belongs to the full profile. The inference
entries contain the operators and kernels needed to run an already-quantized
graph, but no calibration observers, authoring helpers, or weight
materialization symbols.

The built-in flow has no Python or NumPy dependency:

1. run representative inputs through the F32 graph;
2. collect finite min/max ranges for selected activation boundaries;
3. derive symmetric or asymmetric I8/U8 scale and zero-point descriptors;
4. pack selected weights as symmetric per-axis I8 and optional biases in the
   I32 accumulator domain;
5. emit safetensors weights plus immutable quantization metadata against a
   caller-authored quantized blueprint.

## JavaScript full profile

Import PTQ from `volvoxai.full.js` (or `ts/full.ts` in a source checkout):

```js
import {
  calibratePTQ,
  materializePTQWeights,
} from './dist/0.2.0/volvoxai.full.js';

const graph = await runtime.loadGraph('model.safetensors');
const executor = await runtime.compile(graph);
const calibrator = await calibratePTQ(executor, calibrationSamples, {
  graph,
  tensorNames: ['encoder.out', 'decoder.hidden'],
});
const activationParameters = calibrator.parameters({
  dtype: 'int8',
  scheme: 'symmetric',
});

const artifact = materializePTQWeights(graph, [{
  name: 'decoder.proj.weight',
  bias: 'decoder.proj.bias',
  inputScale: activationParameters['decoder.hidden'].scale,
}]);

const safetensorsBytes = artifact.weights.toArrayBuffer();
const weightsQuantization = artifact.weightsQuantization;
```

`calibratePTQ` accepts an iterable or async iterable of named input objects.
CPU and WASM tensors can be observed through CPU-visible graph storage. A
device-output backend requires an explicit `readback` function or per-tensor
readback map; the calibrator never treats a stale host mirror as an
observation.

`PTQObserver`, `derivePTQParameters`, `quantizePTQ`, `packPTQWeight`, and
`packPTQBias` expose the individual textbook-sized steps. The materializer
returns a `SafetensorsFile`, a `weights_quantization` fragment, and records of
the tensors it replaced. These exports are absent from `volvoxai.js`.

## WASM-only full profile

`volvoxai.wasm.js` and its minified variant expose the same stateless numerical
steps through the existing C implementation in `volvoxai.full.wasm`:

```js
import { VolvoxAI } from 'volvoxai/wasm';

const runtime = await VolvoxAI.init('wasm');
const ptq = await runtime.createPTQ();
try {
  const observer = ptq.createObserver();
  observer.observe(calibrationValues);

  const parameters = observer.parameters({
    dtype: 'int8',
    scheme: 'symmetric',
  });
  const values = ptq.quantize(sourceValues, parameters);
  const weight = ptq.packWeight(sourceWeight, [outputs, inputs], { axis: 0 });
  const bias = ptq.packBias(sourceBias, parameters.scale, weight.scales);
} finally {
  ptq.dispose();
}
```

The PTQ ABI is versioned and capability-checked. `WasmPTQ` creates a private
scratch instance, copies inputs into its arena, and returns caller-owned typed
arrays, so a PTQ call cannot reset memory belonging to a compiled inference
graph. Reuse one toolkit and call its idempotent `dispose()` when finished;
subsequent compute calls reject, while previously returned arrays remain valid.
It supports reusable min/max observers, symmetric/asymmetric I8 or U8
per-tensor quantization, rank-1 through rank-8 symmetric per-axis I8 weight
packing, and I32 accumulator bias packing.

The two I8 conversion surfaces intentionally use different integer domains:

| Surface | I8 domain | Purpose |
| --- | --- | --- |
| `quantizePTQ()`, `WasmPTQ.quantize()`, `volvoxai_ptq_quantize_f32()` | Full range `[-128, 127]` | Generic per-tensor affine quantization uses every physical I8 value. |
| `packPTQWeight()`, `WasmPTQ.packWeight()`, `volvoxai_ptq_pack_weight_i8()` | Narrow range `[-127, 127]` | Symmetric per-axis weights keep equal positive and negative magnitudes around zero point 0. |

Symmetric parameters derived from an observed range use
`scale = max(abs(min), abs(max)) / 127`, so values inside that range normally
quantize into `[-127, 127]` even through the generic full-range surface. The
generic surface can produce `-128` for a value outside the calibrated range or
with caller-supplied parameters; packed symmetric weights never produce
`-128`. This distinction is part of the API contract, not a backend mismatch.

This surface intentionally stops at stateless compute. The native plan,
loaded-model mutation, and package writer depend on native graph globals,
locks, cJSON, safetensors file paths, and filesystem publication. In a browser,
the application or JavaScript authoring layer maps the returned arrays and
descriptors into its graph or package. `WasmQuantizedLoRATrainer` is one such
policy; it is not the only supported use of the PTQ kernels.

## Native full profile

Include `volvoxai_training.h` and reuse one observer across calibration
forwards:

```c
volvoxai_ptq_observer_t hidden;
volvoxai_ptq_params_t parameters;

volvoxai_ptq_observer_reset(&hidden);
for (int sample = 0; sample < calibration_count; sample++) {
    set_calibration_inputs(sample);
    if (volvoxai_engine_forward() != 0 ||
        volvoxai_engine_ptq_observe_tensor("decoder.hidden", &hidden) != 0)
        return -1;
}
if (volvoxai_ptq_calculate_params(
        &hidden, VOLVOXAI_DTYPE_I8, VOLVOXAI_PTQ_SYMMETRIC, &parameters) != 0)
    return -1;
```

The array APIs provide per-tensor quantization, per-axis I8 weight packing,
and I32 bias packing. For a loaded model,
`volvoxai_engine_ptq_materialize_weight_i8()` adds named I8 weight and F32
scale tensors to weight file zero. Persist them with
`volvoxai_engine_save_weight_file(0, path)`.

For an end-to-end package, use an explicit `VolvoxAIPTQPlan`:

```c
VolvoxAIPTQPlan* plan = volvoxai_ptq_plan_create();
volvoxai_ptq_tensor_spec_t tensor = VOLVOXAI_PTQ_TENSOR_SPEC_INIT;
tensor.tensor_name = "encoder.input";
volvoxai_ptq_plan_add_tensor(plan, &tensor);
tensor.tensor_name = "encoder.proj";
volvoxai_ptq_plan_add_tensor(plan, &tensor);

volvoxai_ptq_layer_spec_t layer = VOLVOXAI_PTQ_LAYER_SPEC_INIT;
layer.node_index = 0; /* matching FP32 source and quantized-template node */
layer.input_tensor_name = "encoder.input";
layer.output_tensor_name = "encoder.proj";
layer.source_weight_name = "proj.weight";
layer.packed_weight_name = "proj.weight.i8";
layer.source_bias_name = "proj.bias";
layer.packed_bias_name = "proj.bias.i32";
volvoxai_ptq_plan_add_layer(plan, &layer);

for (int sample = 0; sample < calibration_count; sample++) {
    volvoxai_ptq_input_binding_t input = VOLVOXAI_PTQ_INPUT_BINDING_INIT;
    input.tensor_name = "encoder.input";
    input.data = calibration[sample];
    input.nbytes = calibration_bytes;
    char sample_name[32];
    snprintf(sample_name, sizeof(sample_name), "receipt-%d", sample);
    if (volvoxai_engine_ptq_plan_calibrate_sample(
            plan, sample_name, &input, 1) != 0) return -1;
}

volvoxai_ptq_package_options_t package = VOLVOXAI_PTQ_PACKAGE_OPTIONS_INIT;
package.template_config_path = "quantized-template.json";
package.source_weights_path = "model.safetensors";
package.output_config_path = "model-int8.json";
package.output_weights_path = "model-int8.safetensors";
if (volvoxai_ptq_plan_write_package(plan, &package) != 0) return -1;
```

Each calibration call must bind every graph input exactly once. It preflights
names, dtypes, and byte sizes, runs one ordinary forward, and commits all named
F32 observer ranges together. A failed sample cannot leave half-updated
observers. Explicit sample names must be unique and are recorded in the
package's `ptq_authoring.samples` list; the shorter calibration function assigns
stable `sample-N` names. `volvoxai_ptq_plan_tensor_params()` exposes the
resulting activation mapping when an application needs to quantize package
inputs. Calibration always runs every row, then restores the caller's execution
row. A plan is tied to the loaded model generation: reload, shutdown, graph,
weight, or training mutation, and adapter activation make it permanently stale.
Create a new plan after any such change. Plan creation returns `NULL` without a
loaded base model, calibration/parameter-query/write calls return `-1` for a stale plan,
and its sample-count query returns zero. Creation, calibration, and package
writing reject active and merged adapters because package weights are packed
from the base model.

The first package format supports canonical physical-byte `QLinear` and
`QConv2D` nodes with axis-zero symmetric I8 weights. `QLinear` requires an I32
bias; `QConv2D` accepts an optional I32 bias. The writer verifies that the
loaded source node at each planned index is the matching `Linear`/`Conv2D`,
uses exactly the declared source weight and bias, and declares canonical
`OUT_IN`/`OHWI` storage. It verifies that those loaded FP32 source tensors
exactly match the supplied source safetensors file,
then copies pass-through tensors and adds the packed tensors. A selected FP32
source is dropped when the target template no longer references it, while a
shared source used by an unconverted node is retained. The writer fills
graph-input, node-output, and `weights_quantization` descriptors plus auditable
observed ranges under `ptq_authoring`.

The quantized template owns the graph structure, operator choice, shapes,
dtypes, tensor names, and Q-node mappings. Planned internal activation inputs
must be outputs of an earlier planned layer; insert any quantize/dequantize or
residual-boundary nodes explicitly in the template. Leave quantization
descriptors for planned inputs, outputs, and packed weights absent so the writer
can add calibrated values without replacing hand-authored metadata. Output
paths must not already exist. Each output is published with an atomic
no-replace operation even when multiple processes race. The weights file is
published first and the config is the commit marker. The pair cannot be made
transactionally crash-atomic with portable filesystem operations: a process
crash between those publications can leave an orphan weights file without a
config. Ordinary errors remove that writer's weights publication; callers may
remove an orphan weights file when no matching config exists after a crash.

The writer preflights the complete template namespace and dependency graph.
Packed weight names cannot collide with graph inputs or outputs, and every
unselected external tensor must already exist in the single source safetensors
file. If an unselected dependency is available only from another loaded shard,
author a consolidated source file first; otherwise the advertised two-file
package would not be self-contained.

The PTQ functions are compiled and exported only by `native/volvoxai-full`.
The native inference profile contains neither their implementation nor public
PTQ symbols.

## Graph boundary remains explicit

Quantization cannot safely rename every F32 operator mechanically. The author
must choose calibration boundaries and supply a template with supported nodes
in their canonical `Q*` forms. JavaScript exposes materialized tensors and
metadata for that explicit lowering; native C can additionally validate the
template and write the two-file package. Neither path guesses how to cross an
unsupported normalization, attention, or residual boundary.

The TinyReceiptVQA Python tools remain a model-specific release validator and
package assembler. The generic calibration math, range collection, packing,
and safetensors emission no longer require that toolchain; the same primitives
can be driven from JavaScript or native C for other models.
