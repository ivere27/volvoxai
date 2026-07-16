# Model Format

VolvoxAI loads inference packages made of:

```text
config.json
model.safetensors
optional tokenizer/assets, such as vocab.bin, merges.txt, labels.txt
```

Large `.safetensors` files and generated model directories should stay out of git.

## Graph Model

A model graph contains:

- `Tensor`: name, shape, dtype, and `isWeight`.
- `Node`: `opType`, named `inputs`, named `outputs`, and `params`.
- `graph.outputNames`: output tensor names returned by WASM/CPU.

`GraphLoader.ts` builds this graph from `config.json` and `model.safetensors`.
Applications can also construct graphs programmatically with `addInput`,
`addWeight`, and `addOp`.

## Volvox Blueprint

The primary format is a precomputed Volvox blueprint:

```json
{
  "inputs": {
    "image": { "shape": [1, 320, 320, 3], "dtype": "float32" }
  },
  "nodes": [
    {
      "opType": "Conv2D",
      "inputs": { "input": "image", "weight": "stem.weight", "bias": "stem.bias" },
      "outputs": { "out": "stem.out" },
      "outputs_shape": { "out": [1, 160, 160, 32] },
      "params": { "stride": [2, 2], "padding": [1, 1], "weight_layout": "OHWI" }
    }
  ]
}
```

Exporters map source-model ops to Volvox `opType`s, precompute tensor shapes, and
write the weights into safetensors. The browser then only maps memory and executes.

The generic `tools/export_safetensors.py` converter accepts ONNX and TFLite
graphs. It names outputs `output0`, `output1`, and so on by default; callers may
repeat `--output-name NAME` in source-output order when an application owns a
semantic output contract. The generic converter does not infer task meaning
from tensor shapes.

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

New blueprints select public graph outputs with a unique tensor-name array,
for example `"outputs": ["scores", "boxes"]`. The loader also accepts the
legacy source-alias object form (`{"source_name": "scores"}`) used by existing
exported packages; aliases do not become core graph tensor names.

`GraphLoader` owns package parsing and graph-assembly orchestration. Reusable,
model-independent operator layout normalization and portable quantized-graph
validation live under `ts/ops/`, alongside the computation contracts they
protect.

## Hugging Face Config Path

If `config.json` has `model_type` but no topological `nodes`, `GraphLoader` checks
`GraphLoader.ModelBuilders`. A model family can register a builder:

```javascript
GraphLoader.ModelBuilders.llama = (graph, config, tensors) => {
  // call graph.addOp(...) to assemble the model
};
```

## Safetensors Loading

VolvoxAI reads standard safetensors weights. Tensor metadata provides dtype, shape,
and byte offsets. Graph topology can also be embedded in safetensors metadata, but
the normal package layout keeps topology in `config.json`.

### Binary W8A8 companion scales

Materialized W8A8 packages keep weight scales as binary F32 tensors instead of
expanding and repeating them as JSON numbers. The config selects the storage
profile but carries no descriptor table:

```json
{
  "weights_quantization_storage": {
    "format": "volvoxai-f32-companion-scales-v1"
  }
}
```

Each safetensors shard carries its authoritative local descriptor map in
string-valued metadata. Every described I8 tensor `W` has a same-shard F32
companion named `W_scale`, with shape `[W.shape[0]]` for axis-0 quantization or
`[1]` for per-tensor quantization. The symmetric zero point is implicitly zero,
and companions remain internal storage records rather than graph-visible
weights. Unquantized-only shards may omit the companion metadata.

The exact metadata, validation, sharding, and numeric contract is specified in
[W8A8 safetensors companion scales](w8a8-safetensors.md).

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
  per-tensor scale/zero-point metadata; QLinear/QConv/QEmbedding weights use
  per-output-channel metadata and I32 accumulation/bias.
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
- Native Vulkan/OpenGL/Metal use packed-byte graph paths where supported and
  otherwise fall back to native CPU. Their packed-byte kernels remain
  ordinary-full-tensor paths.
- The fixed-B=1 TinyReceipt [browser/Node wrapper](../examples/tiny_receipt_vqa/TinyReceiptW8A8Session.js)
  and [native wrapper](../examples/tiny_receipt_vqa/native/tiny_receipt_w8a8.c)
  have an opt-in incremental mode. Both are examples rather than fixed runtime
  entry points or release artifacts. Their first step
  seeds the full graph; later native CPU steps retain image/encoder
  intermediates and prior physical self-attention K/V rows while refreshing
  only the current decoder row. Browser/Node CPU and WASM execution use the
  same row/KV contract. WebGPU shares the dependency cache but retains
  full-tensor decoder kernels.

See [the operation matrix](operation_list.md#quantized-execution) for the exact
operator and backend contract, including the TinyReceiptVQA materialized W8A8
package workflow.

## LLM Scope

VolvoxAI includes the core ops used by small GPT-style models: `Embedding`,
`LayerNorm`, `MatMul`, `SDPA`, and `GELU`. The native runtime exposes generic
prefix/row and output-row APIs for KV-cache orchestration. TinyStories
vocabulary selection and autoregressive generation live in the opt-in
`examples/native_task_cli/` application.

It is not a frontier LLM runtime today:

- No INT4 weight format.
- Browser text generation currently needs application-side autoregressive loops.
- The native task example demonstrates generation ahead of browser helpers.
