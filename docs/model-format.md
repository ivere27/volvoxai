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

`GraphLoader.js` builds this graph from `config.json` and `model.safetensors`.
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

## Layout Policy

Runtime image tensors use NHWC:

```text
[batch, height, width, channels]
```

Browser Conv2D kernels use HWIO/HWCM internally. Native TFLite exports may keep
TFLite-native layouts such as `OHWI` for regular conv and `1HWO` for depthwise conv;
the native engine prepares HWIO/HWCM compute caches at load time.

## Precision Policy

VolvoxAI focuses on FP32 activations/intermediates and INT8-packed or quantized
weights.

- FP32 is the portable activation format across WebGPU, WASM, CPU, and native paths.
- INT8 reduces weight bandwidth and storage while avoiding the bit-unpacking cost of
  INT4 on low-end GPUs.
- FP16 weights can be kept on disk for exported vision models; runtime paths widen or
  handle them according to backend support.

INT8 handling:

- MatMul keeps an INT8 plus per-channel-scale fast path in browser shaders and native
  kernels.
- Browser Conv2D/Conv1D dequantizes quantized weights to FP32 at load time.
- Native CPU can keep quantized Conv/Add islands in int8 via `native/quant_cpu_opt.c`.
- Native Vulkan/OpenGL graph paths are FP32-only, so `QConv2D` stays on CPU.

## LLM Scope

VolvoxAI includes the core ops used by small GPT-style models: `Embedding`,
`LayerNorm`, `MatMul`, `SDPA`, and `GELU`. The native CLI includes TinyStories
generation with prefill/decode KV-cache orchestration.

It is not a frontier LLM runtime today:

- No INT4 weight format.
- Browser text generation currently needs application-side autoregressive loops.
- Native generation is ahead of browser generation helpers.

