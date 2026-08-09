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

- bounded dimension constraints and fixed-rank logical tensor descriptors;
- nodes with an `id`, `opType`, named inputs, unified output assertions, and
  explicit `params`;
- exact public output tensor names returned by `ExecutionResult`.

`ModelLoader.ts` parses the closed graph document and safetensors into a
`Model`. Compilation proves its complete bounded shape domain;
an `ExecutionContext` then binds one concrete public-input shape set without
mutating the snapshot.

When the graph URL cannot be derived from the safetensors URL, pass it explicitly
as `graphUrl`. Its basename must be `graph.json` or a named `*.graph.json`
document.

## Volvox Graph Document

The only persisted graph format is the closed, bounded-shape
`volvox-graph/v1` document:

```json
{
  "format": "volvox-graph/v1",
  "dimensions": {},
  "inputs": {
    "image": { "shape": [1, 320, 320, 3], "dtype": "float32" }
  },
  "nodes": [
    {
      "id": "stem",
      "opType": "Conv2D",
      "inputs": { "input": "image", "weight": "stem.weight", "bias": "stem.bias" },
      "outputs": {
        "out": { "tensor": "stem.out", "shape": [1, 160, 160, 32], "dtype": "float32" }
      },
      "params": {
        "kernel": [3, 3], "stride": [2, 2], "dilation": [1, 1],
        "pads": [1, 1, 1, 1], "groups": 1, "data_layout": "NHWC",
        "weight_layout": "HWIO"
      }
    }
  ],
  "outputs": ["stem.out"]
}
```

Each symbolic axis names an entry in `dimensions`; every entry has finite
caller-supplied `min`/`max` bounds and may add `multiple_of`. Anonymous source
dimensions become symbols only when the caller supplies such bounds. Exporters
map source-model ops to Volvox `opType`s, preserve symbolic shape formulas, and
write immutable tensor payloads into safetensors. Import and lowering are
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

Image normalization is application preprocessing, not executable graph
semantics. The closed input descriptor therefore contains exactly `shape` and
`dtype`; task manifests or calling applications own color conversion,
normalization, resize policy, and semantic aliases. Exporters reject attempts
to place that metadata in `graph.json`.

Every node input name must resolve to a tensor declared in `inputs`, a named
tensor loaded from `model.safetensors`, or an output produced by an earlier
node. There is no implicit image tensor or fallback shape: a package with an
undeclared reference is rejected during graph construction.

Every graph execution tensor has one of four dtypes: `float32`, `int32`,
`int8`, or `uint8`. These names are lowercase and case-sensitive. Safetensors
may store internal weights as F16, but F16 is storage rather than an execution
dtype. A standalone F16 weight selected as a graph output is exposed as F32.
Every node output port contains one exact `{tensor, shape, dtype}` assertion.
The loader rejects split output-name/shape/dtype maps and verifies each unified
assertion against canonical operator shape inference over the complete bounded
domain; it never infers an implicit F32 dtype.

Graph documents must select public outputs with a non-empty, unique
tensor-name array, for example `"outputs": ["scores", "boxes"]`. Each name
resolves directly to a declared graph tensor. Package loading does not infer
leaf outputs. Every backend returns all declared outputs by exact name through
ExecutionResult.

`ModelLoader` owns package parsing and logical snapshot assembly.
Reusable operator shape proofs and portable quantized-graph validation live
under `ts/ops/`, alongside the computation contracts they protect.

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

Ordinary FP32 `Conv2D` weights ship in the layout the microkernels index:
`HWIO` for regular and grouped convolution, `HWCM` for depthwise. The exporter
performs the permutation once, so no runtime transposes at load or holds a second
copy of the weight.

Canonical W8A8 `QConv2D` weights stay `OHWI`, which is the layout its own kernels
index and the axis its per-output-channel affine metadata is bound to.

`Conv1D` and `ConvTranspose2D` follow the same rule, with output channels
innermost so the accumulation loop stays contiguous:

```text
Conv1D            NLC  [batch, length, channels]   WIO  [k, in_per_group, out_c]
ConvTranspose2D   NHWC [batch, height, width, c]   HWIO [kh, kw, in_c, out_c]
```

Keeping `Conv1D` channels-last also means it needs no layout transpose at its
boundaries when it sits between `LayerNorm`/`Linear`/attention in a sequence
model, which is where the channels-first form cost the most.

There is no compatibility interpretation for retired ordinary-convolution
layouts; packages must be re-exported into the canonical form.

FP32 `Linear`/`MatMul`/`Gemm` weights carry an explicit `weight_layout` of
`din_dout` (`[d_in, d_out]`) or `dout_din` (`[d_out, d_in]`). Both are
first-class: for an immutable model weight the runtime may pack either into the
same physical B panel. W8A8 `QLinear`/`QMatMul`/`QGemm` instead have one
operator-defined layout, `dout_din`; their `params` object is empty and axis 0
is where per-output-channel scales are bound. Weight-only W8A32 also uses
`dout_din`.

Validation rejects a canonical FP32 dense node that omits `weight_layout`, and
rejects a quantized dense node that tries to override its fixed layout. It does
not guess from a non-square weight, accept `transB` as a package alias, or choose
a default for square FP32 weights.

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
- The bounded-active TinyReceipt
  [browser/Node split session](../examples/tiny_receipt_vqa/TinyReceiptSplitSession.js)
  and [native split application](../examples/tiny_receipt_vqa/native/tiny_receipt_split_w8a8.c)
  use separate encoder and retained-decoder contexts. Both are examples rather
  than fixed runtime entry points or release artifacts. Seed, step, and reset
  are FIFO operations on each private decoder context.

See [the operation matrix](operation_list.md#quantized-execution) for the exact
operator and backend contract, including the TinyReceiptVQA split W8A8 package
workflow.

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
