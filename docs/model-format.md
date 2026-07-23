# Model Format

VolvoxAI loads inference packages made of:

```text
graph.json
model.safetensors
optional tokenizer/assets, such as vocab.bin, merges.txt, labels.txt
```

`graph.json` is the graph descriptor (inputs, nodes, outputs) and `model.safetensors`
holds the weights. Its root `"format"` must equal `"volvox-graph/v1"` exactly,
including case. Missing, differently typed, or differently versioned values are
rejected before weights are fetched or the destination graph is mutated. No other
filename is searched as a graph-package fallback.

Large `.safetensors` files and generated model directories should stay out of git.

## Graph Model

A model graph contains:

- `Tensor`: name, shape, dtype, and `isWeight`.
- `Node`: `opType`, named `inputs`, named `outputs`, and `params`.
- `graph.outputNames`: exact output tensor names returned by ExecutionResult.

`GraphLoader.ts` builds this graph from `graph.json` and `model.safetensors`.
Applications can also construct graphs programmatically with `addInput`,
`addWeight`, and `addOp`.

When the graph URL cannot be derived from the safetensors URL, pass it explicitly
as `graphUrl`. Its basename must be `graph.json` or a named `*.graph.json`
document.

## Volvox Graph Document

The only persisted graph format is a precomputed `volvox-graph/v1` document:

```json
{
  "format": "volvox-graph/v1",
  "inputs": {
    "image": { "shape": [1, 320, 320, 3], "dtype": "float32" }
  },
  "nodes": [
    {
      "opType": "Conv2D",
      "inputs": { "input": "image", "weight": "stem.weight", "bias": "stem.bias" },
      "outputs": { "out": "stem.out" },
      "outputs_shape": { "out": [1, 160, 160, 32] },
      "outputs_dtype": { "out": "float32" },
      "params": { "stride": [2, 2], "padding": [1, 1], "weight_layout": "OHWI" }
    }
  ],
  "outputs": ["stem.out"]
}
```

Exporters map source-model ops to Volvox `opType`s, precompute tensor shapes,
and write immutable tensor payloads into safetensors. Import and lowering are
different stages: ONNX and TensorFlow Lite source adapters first preserve the
source graph in SourceIR without graph rewrites, then a legality-checked lowerer
creates verified RuntimeIR. Unsupported semantics fail with a source diagnostic;
the importer does not silently approximate them. The browser and native
runtimes only validate, map memory, select a prequalified route, and execute.

The generic `tools/export_safetensors.py` converter accepts ONNX and TFLite
graphs. It names outputs `output0`, `output1`, and so on by default; callers may
repeat `--output-name NAME` in source-output order when an application owns a
semantic output contract. It does not infer task meaning, preprocessing,
generation policy, ArgMax, or decode state from tensor shapes. ONNX lowering
uses the shared typed compiler. TensorFlow Lite already uses the lossless
SourceIR audit boundary, but its full lowering is still being moved from the
direct exporter into that compiler; this is an implementation boundary, not a
second package format.

A raw lossless import is not automatically a deployment specialization. Apply
portable optimization, any explicit input/output specialization, PTQ or
existing-INT8 cleanup, differential verification, and target qualification
before publishing the runtime package. See
[graph optimizer design](graph-optimizer-design.md).

The model-agnostic exporter command writes `graph.json` beside the requested
safetensors file and stages publication atomically:

```bash
python3 tools/export_safetensors.py \
  --model model.onnx \
  --out build/model/model.safetensors \
  --target portable \
  --report build/model/export-report.json \
  --report-format json

python3 tools/export_safetensors.py \
  --model model.tflite \
  --out build/model-tflite/model.safetensors \
  --target portable
```

`--quant-mode preserve` is the default and preserves source-described integer
islands. `--quant-mode require-w8a8` adds a fail-closed requirement that the
published graph classify as canonical W8A8; it does not quantize an F32 model.
Use repeatable `--input-shape`, `--input-dtype`, `--output-dtype`, and
`--specialize-input` bindings for supported ONNX lowering. Those bindings are
not yet enabled on the transitional TensorFlow Lite lowerer.

Image-capable frontends may use an optional normalization contract on each
graph input:

```json
{
  "inputs": {
    "image": {
      "shape": [1, 320, 320, 3],
      "dtype": "float32",
      "image_normalization": "zero-one"
    }
  }
}
```

The supported values are `zero-one`, `minus-one-one`, and `raw-255`. This is
application preprocessing metadata rather than a graph operation: runtimes
still receive an already prepared tensor. A frontend must not infer the value
from the input dtype, because F32 and byte inputs can each use different source
pixel ranges. `tools/export_safetensors.py` writes this metadata only when the
export workflow supplies `--image-normalization INPUT=MODE`; repeat the option
for multiple image inputs. The native task example requires this metadata for
`--image` unless the caller supplies an explicit `--image-normalize` override;
it does not silently choose a pixel range.

Every node input name must resolve to a tensor declared in `inputs`, a named
tensor loaded from `model.safetensors`, or an output produced by an earlier
node. There is no implicit image tensor or fallback shape: a package with an
undeclared reference is rejected during graph construction.

Every graph execution tensor has one of four dtypes: `float32`, `int32`,
`int8`, or `uint8`. These names are lowercase and case-sensitive. Safetensors
may store internal weights as F16, but F16 is storage rather than an execution
dtype. A standalone F16 weight selected as a graph output is exposed as F32.
Current v1 requires `outputs_dtype` for every node output port, including
`float32`; loaders reject a missing or incomplete map rather than infer an
implicit dtype. Explicit output dtypes keep typed round trips and target
capability checks unambiguous.

Graph documents must select public outputs with a non-empty, unique
tensor-name array, for example `"outputs": ["scores", "boxes"]`. Each name
resolves directly to a declared graph tensor. Package loading does not infer
leaf outputs. Every backend returns all declared outputs by exact name through
ExecutionResult.

`GraphLoader` owns package parsing and graph-assembly orchestration. Reusable,
model-independent operator layout normalization and portable quantized-graph
validation live under `ts/ops/`, alongside the computation contracts they
protect.

## Safetensors Loading

VolvoxAI reads standard safetensors weights. Tensor metadata provides dtype, shape,
and byte offsets. Graph topology is authoritative in `graph.json`.

### Safetensors-backed affine quantization

Materialized byte graphs keep every scale and zero point as a safetensors
tensor. The graph contains a single reference-only descriptor table:

```json
{
  "quantization": {
    "format": "volvox-affine-safetensors/v1",
    "tensors": {
      "hidden_q": {
        "scheme": "per_tensor",
        "scale_tensor": "__quant__.hidden.scale",
        "zero_point_tensor": "__quant__.hidden.zero_point"
      }
    }
  }
}
```

The referenced scale is rank-1 F32. The zero point is rank-1 I8/U8 matching the
target. Per-tensor parameters have one element; per-axis parameters match the
selected target dimension. Symmetric zero points are stored explicitly. JSON
never contains numeric affine values, and safetensors metadata is not a second
descriptor authority.

The exact validation, Q/DQ identity, sharding, and rejected-representation
contract is specified in
[affine quantization in safetensors](w8a8-safetensors.md).

## Layout Policy

Runtime image tensors use NHWC:

```text
[batch, height, width, channels]
```

Browser Conv2D kernels use HWIO/HWCM internally. Native TFLite exports may keep
TFLite-native layouts such as `OHWI` for regular conv and `1HWO` for depthwise conv;
the native engine prepares HWIO/HWCM compute caches at load time.

## Precision Policy

VolvoxAI supports both ordinary FP32 graphs and canonical W8A8 graph islands.

- FP32 remains the general portable activation format.
- Canonical W8A8 edges use physical I8/U8 activation storage with immutable
  per-tensor scale/zero-point tensor references; QLinear/QConv/QEmbedding
  weights use per-output-channel references and I32 accumulation/bias.
- INT8 reduces weight bandwidth and activation-buffer footprint without INT4
  unpacking. Some nonlinear, normalization, and attention kernels use private
  F32 scratch, but do not materialize an F32 graph activation edge.
- FP16 weights can be kept on disk for exported vision models; runtime paths widen or
  handle them according to backend support.

INT8 handling:

- Ordinary `MatMul` keeps its W8A32 packed-weight path (FP32 activations/output).
- `QConv2D`, `QLinear`, `QEmbedding`, typed normalization/attention, and typed
  shape/concat edges form the explicit W8A8 path across CPU(JS), WASM, WebGPU,
  native CPU, and native ordinary-forward GPU dispatch.
- Native Vulkan/OpenGL/Metal use packed-byte graph paths where supported.
  Compilation policy determines whether CPU operator routing is allowed, and
  reports the selected route. Opt-in CUDA has a separate broad packed-byte
  allowlist and strict no-CPU-fallback routing; see
  [cuda.md](cuda.md#status).
- The fixed-B=1 TinyReceipt [browser/Node wrapper](../examples/tiny_receipt_vqa/TinyReceiptW8A8Session.js)
  and [native wrapper](../examples/tiny_receipt_vqa/native/tiny_receipt_w8a8.c)
  have an opt-in incremental mode. Both are examples rather than fixed runtime
  entry points or release artifacts. Browser execution uses a private
  ExecutionContext for each decode stream; seed, step, and reset are FIFO
  operations on that context.

See [the operation matrix](operation_list.md#quantized-execution) for the exact
operator and backend contract, including the TinyReceiptVQA materialized W8A8
package workflow.

## LLM Scope

VolvoxAI includes the core ops used by small GPT-style models: `Embedding`,
`LayerNorm`, `MatMul`, `SDPA`, and `GELU`. Decode state belongs to an
ExecutionContext. TinyStories vocabulary selection and autoregressive
generation live in downstream applications.

It is not a frontier LLM runtime today:

- No INT4 weight format.
- Browser text generation currently needs application-side autoregressive loops.
- Native and browser clients build generation loops from context-local decode
  operations.
