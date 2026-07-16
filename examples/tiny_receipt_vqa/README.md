# TinyReceiptVQA examples

## JavaScript W8A8 inference session

[`TinyReceiptW8A8Session.js`](TinyReceiptW8A8Session.js) is a browser/Node
reference wrapper for a materialized TinyReceiptVQA W8A8 package. It validates
the package manifest, applies the model-specific character vocabulary and
image preprocessing, runs the router, and autoregressively executes the chosen
family graph through an ordinary `VolvoxAI` inference runtime.

The wrapper deliberately lives in `examples/` and is not exported from
`ts/index.ts` or included in the npm inference bundle. Repository callers can
import it directly:

```js
import { VolvoxAI } from '../../ts/index.ts';
import { TinyReceiptW8A8Session } from './TinyReceiptW8A8Session.js';

const runtime = await VolvoxAI.init('auto');
const session = await TinyReceiptW8A8Session.load({
  runtime,
  packageUrl: '/models/tiny-receipt/package_manifest.json',
});
const result = await session.generate({ image, prompt: 'phone number?', incremental: true });
```

On WebGPU, `incremental: true` automatically selects the on-device feedback
path when the compiled decoder satisfies its strict row contract. Terminal
`QArgMax` IDs are copied into the next decoder input without a JavaScript
round-trip, and the wrapper maps one 16-token chunk at a time to check EOS.
`deviceFeedbackChunkSize` can tune that tradeoff: a larger value maps less
often but may compute more rows past the first EOS. The returned `execution`
field is `device-feedback-row-kv-cache` when this path is active.

When the runtime supports detached compilation, the session gives the router
and each selected family an independent backend owner. The small router can no
longer evict the much larger family graph and force it to compile again on the
next request.

The session owns a read-only SafeTensors cache, so its router and family graphs
fetch and parse the shared model file once while retaining independent graph,
tensor, and SafeTensors wrapper objects. Applications can move graph loading
off the first request without compiling over a shared mutable backend engine:

```js
await session.preload({ families: ['phone'] }); // or families: 'all'
```

Measure a genuinely cold graph request followed by a hot request on the same
session with the exact same preprocessed image and question:

```bash
export NAVERCAP_ROOT=/path/to/navercap
npx tsx --experimental-wasm-relaxed-simd \
  examples/tiny_receipt_vqa/tools/benchmark_wasm_cold_hot.mjs \
  /tmp/volvoxai-tinyreceipt-safetensors-v1 \
  "$NAVERCAP_ROOT/eval/heldout/images/00002.jpg" \
  "phone number last one" \
  dist/0.2.0/volvoxai.wasm
```

The benchmark uses Pillow only as a development-time JPEG-to-RGB decoder;
model preprocessing and both inference calls stay in the JavaScript/WASM path.
Its JSON reports model fetch counts, graph-load/compile costs, seed latency,
steady token rate, and the exact cold-minus-hot difference.

Run the real package through Chrome/WebGPU, including detached router/family
compilation, row K/V reuse, device feedback, chunked EOS checks, and image
preprocessing, with:

```bash
export NAVERCAP_ROOT=/path/to/navercap
npm run build
node --experimental-websocket tools/run_webgpu_w8a8_benchmark.mjs \
  --timeout-ms=600000 \
  --url='/examples/tiny_receipt_vqa/tools/webgpu_w8a8_benchmark_tinyreceipt.html?chunk=16&maxNewTokens=100' \
  --model-dir=/tmp/volvoxai-tinyreceipt-safetensors-v1 \
  --image="$NAVERCAP_ROOT/eval/heldout/images/00002.jpg"
```

Set `feedback=0` in the page URL for the per-token-map A/B baseline. The page
reports cold/hot wall time, graph load/compile time, model fetch counts, every
chunk completion, execution mode, and exact answer text.

## Train -> VolvoxAI PTQ -> W8A8 hero path

The native C++ trainer's `full_model/` package is an F32 `volvox.api.v1`
graph, not the published PyTorch INT8 release consumed by the older Python
tools. [`tools/export_trained_ptq_source.mjs`](tools/export_trained_ptq_source.mjs)
validates that exact source pair, maps the trained parameter names and layouts,
and invokes the full-profile VolvoxAI PTQ APIs for every quantized weight.
The output has its own truthful source identity,
`tiny_receipt_vqa_volvox_trained_int8_safetensors_v1`.

The mapping is explicit and checked: convolution `HWIO -> OIHW`, Linear/LoRA
`IN_OUT -> OUT_IN`, self-attention packed QKV transpose, cross-attention Q/K/V
transpose and row concatenation, eight-expert tensors to per-family rows,
`q_pos` rank expansion, and an exact tied token/head copy. F32 norms and biases
remain F32. Both production and the trainer's smaller `--smoke` stem channel
sets are represented in the normalized manifest.

This command chain performs one complete structural smoke:

```bash
make -C examples run_cpp_receipt_train_smoke NAVERCAP_ROOT=/path/to/navercap

npx tsx examples/tiny_receipt_vqa/tools/export_trained_ptq_source.mjs \
  --source /tmp/volvoxai-receipt-smoke/full_model \
  --out-dir /tmp/volvoxai-receipt-smoke/ptq-source \
  --structural-smoke

python3 examples/tiny_receipt_vqa/tools/materialize_tiny_receipt_vqa_w8a8.py \
  --manifest /tmp/volvoxai-receipt-smoke/ptq-source/manifest.json \
  --activation-calibration \
    /tmp/volvoxai-receipt-smoke/ptq-source/activation_profile.json \
  --out-dir /tmp/volvoxai-receipt-smoke/w8a8

make -C examples native_receipt_inference_example
examples/target/bin/tiny_receipt_w8a8 \
  /tmp/volvoxai-receipt-smoke/w8a8 \
  --image /path/to/receipt.png \
  --prompt "What is the phone number?" \
  --max-new 30 \
  --incremental
```

`--structural-smoke` runs the actual trained F32 graph once with zero-filled
inputs. It proves the artifact mapping, JS PTQ, Python graph assembly, package
loading, and W8A8 execution chain; it is explicitly recorded as
non-representative calibration and is not an accuracy claim.

For meaningful calibration, replace `--structural-smoke` with
`--samples /path/to/named-samples.json`. The document must supply every trained
graph input for each sample, with flat arrays matching the config's dtype and
shape exactly:

```json
{
  "format": "volvoxai-tiny-receipt-vqa-named-calibration-samples-v1",
  "samples": [
    {
      "vqa.image": [0.0],
      "vqa.question_ids": [0],
      "vqa.question_positions": [0]
    }
  ]
}
```

The abbreviated arrays above show the spelling only; a real record must include
all inputs and their complete flattened contents. The exporter runs those
records through the F32 graph with `calibratePTQ`, observes every F32 graph
value, and writes a single global fallback profile. The package manifest keeps
that qualification visible: it is not per-edge calibration. The existing
Python materializer remains the model-specific graph assembler and strict
release validator; generic F32 weight quantization no longer depends on
NumPy/PyTorch in this path.

For a smaller mapping diagnostic,
[`tools/materialize_trained_linear_ptq.mjs`](tools/materialize_trained_linear_ptq.mjs)
can extract one selected trained `Linear` boundary as a standalone runnable
`QLinear` island. Its manifest says
`runnable_as_full_tiny_receipt_vqa: false`; it cannot be confused with the
router/eight-family package.

## Python W8A8 package tooling

The model-specific release normalizer, calibration workflow, and package
materializer live under [`tools/`](tools/). They are example tooling and are
not part of VolvoxAI's generic root `tools/` surface.

This older path starts from the separately published
`tiny_receipt_vqa_int8_safetensors_v1` PyTorch-layout release. It remains
supported and keeps its independent source/runtime validation; that source
format is never assigned to a native-trained artifact.

Create a development calibration record and materialize a runnable W8A8
package with:

```bash
python3 examples/tiny_receipt_vqa/tools/calibrate_tiny_receipt_vqa_w8a8.py \
  --release-dir /path/to/tiny-receipt-release \
  --annotations-dir /path/to/heldout/annotations \
  --images-dir /path/to/heldout/images \
  --out /tmp/tinyreceipt-calibration.json \
  --include-explicit-families
python3 examples/tiny_receipt_vqa/tools/materialize_tiny_receipt_vqa_w8a8.py \
  --manifest /path/to/tiny-receipt-release/int8/manifest.json \
  --out-dir /tmp/tinyreceipt-w8a8 \
  --development-calibration /tmp/tinyreceipt-calibration.json
```

The materializer writes the materialized package format. Every quantized tensor
`W` has one same-shard F32 scale companion named `W_scale`: shape `[W.shape[0]]`
for axis-0 weights and `[1]` for per-tensor constants. Router and family configs
retain only the companion-storage format marker; authoritative local
scheme/axis descriptors live in safetensors metadata. The v1 descriptors are
symmetric and contain no inline scale or zero-point values. See the normative
[W8A8 safetensors companion-scale contract](../../docs/w8a8-safetensors.md).

[`tools/import_tiny_receipt_vqa_int8.py`](tools/import_tiny_receipt_vqa_int8.py)
is the narrower weight-only normalizer for the published named SafeTensors
layout. It does not construct a graph or activation calibration.

Legacy local PyTorch KIE checkpoints use the example-owned exporter:

```bash
python3 examples/tiny_receipt_vqa/tools/export_kie_safetensors.py \
  --checkpoint /path/to/checkpoint.pt \
  --out /tmp/tinyreceipt-kie/model.safetensors
```

Run the example-owned Python tests from the repository root:

```bash
python3 -m unittest discover \
  -s examples/tiny_receipt_vqa/tests \
  -p 'test_*.py'
```

[`tools/make_model_graph_figure.py`](tools/make_model_graph_figure.py) generates
the TinyReceiptVQA W8A8 architecture figure. Its default output is the
example-local `examples/tiny_receipt_vqa/model_graph.svg`.

## Native W8A8 inference session

[`native/tiny_receipt_w8a8.c`](native/tiny_receipt_w8a8.c) is the standalone C
equivalent. The session, its Pillow-compatible grayscale preprocessing, and its
tests live entirely in this example; they are not linked into `native/volvoxai`
or `native/volvoxai-full`.

Build and test it with:

```bash
make -C examples native_receipt_inference_example
make -C examples test_native_receipt_inference
```

Then run a materialized package directly:

```bash
examples/target/bin/tiny_receipt_w8a8 /path/to/package \
  --image receipt.png \
  --prompt "What is the phone number?" \
  --max-new 96 \
  --incremental
```

The example validates the package manifest, runs the hard router, selects one
of the eight explicit family graphs, and reads the terminal I32 `QArgMax` token
IDs. Ordinary full-graph execution is the default; `--incremental` enables the
generic dependency and native CPU row/KV caches.

Backend flags are passed through the public `VolvoxAIEngineOptions` and
`volvoxai_engine_configure()` API. Pass at most one accelerator flag; an
explicit backend that is unavailable fails instead of silently selecting CPU.
The example does not import or manage private backend devices.

For `--incremental`, the native CPU backend is still the simplest latency
baseline. With built-in `--vulkan`, `--opengl`, or Metal selected, a compatible
session runs the seed on that GPU and automatically synchronizes its retained
state once before using the native CPU row/KV loop for later tokens. This avoids
rerunning and synchronizing roughly one hundred small GPU nodes per token. Set
`VOLVOXAI_DISABLE_GPU_CPU_ROW=1` only to measure the old device dependency path.
The measured Renoir comparison and profiler interpretation are in
[Native Runtime](../../docs/native-runtime.md#native-gpu-and-npu-backends).

## C++ training example

`tiny_receipt_vqa_train.cc` is the native C++ source port of Navercap's
`tiny_receipt_vqa/train.py`. It does not load or invoke PyTorch. Typed messages
generated from `proto/volvoxai.proto` are sent through the VolvoxAI
gRPC/Synurang FFI service boundary.

The caller owns record discovery, synthetic/real record construction,
deterministic image preprocessing, UTF-8 character vocabulary, batching,
cosine learning-rate scheduling, and exact-match evaluation. VolvoxAI owns the
graph, forward and backward passes, AdamW, gradient clipping, model
safetensors, LoRA persistence, and full optimizer checkpoints.

## Build

The example requires a C++17 compiler, `pkg-config`, and protobuf development
files. On Debian/Ubuntu these are provided by `pkg-config`, `libprotobuf-dev`,
and `protobuf-compiler`. Image decoding uses the repository's checked-in
`stb_image.h`.

```bash
make build_server
make -C examples cpp_receipt_train_example
```

The protobuf C++ files are generated under `runtime/target/generated/cpp`; they
are build products and are not committed.

## Train on Navercap synth and real data

The defaults reproduce the production-size graph: `d_model=320`, eight heads,
six encoder layers, four decoder layers, 210 image tokens, eight task-adapter
families, an auxiliary router, and rank-8 LoRA on every encoder/decoder FFN
linear. The safe physical batch default is one with 24 gradient-accumulation
steps, preserving an effective batch of 24 without retaining 24 copies of every
activation. Both values remain configurable. Dataset paths are always supplied
by the caller; the executable contains no machine-specific checkout path.

```bash
examples/target/bin/tiny_receipt_vqa_train \
  --navercap-root /path/to/navercap \
  --backend cpu \
  --batch-size 1 \
  --accumulation-steps 24 \
  --task-profile structured_qa \
  --target-mode rationale \
  --out runs/volvoxai_tiny_receipt_vqa
```

For raw-data mode, pass either `--navercap-root`, or pass all three explicit
paths: `--synth-root`, `--real-ann`, and `--real-img`. JSONL manifest mode needs
only `--train-records` and `--val-records`; image paths may be absolute or
relative to each manifest.

Use `libvolvoxai.dylib` in the `--lib` path on macOS. The executable chooses
the platform-correct filename automatically when `--lib` is omitted.

CPU is the program default. `--backend vulkan`, `--backend opengl`, and
`--backend metal` are explicit requests: the complete backward plan must execute
on that backend or `TrainStep` fails. Forward execution can still use documented
native CPU fallbacks, currently including the MoE router/linear path. Metal
additionally requires an Apple runtime build; a non-Apple executable rejects it
before loading the service. On macOS the Cargo service compiles
`native/src/backends/metal_engine.m` and links the Foundation and Metal frameworks.

Before loading the runtime, the example sums the concrete `CreateModel` tensor
shapes and prints a `resource_preflight` event. Vulkan uses a single mapped
arena: 512 MiB by default, controlled by `VOLVOX_VULKAN_MB`, with a current
maximum of 4096 MiB. OpenGL and Metal use the same conservative working-set
estimate against `--gpu-memory-mb`, whose default is 4096 MiB; raise that value
only when the selected device really has the corresponding free memory. The
preflight includes a conservative F32 gradient reserve.
A physical batch of 24 has more than 7 GiB of forward-resident tensors and needs
over 14 GiB with the gradient reserve, so it is rejected deterministically.
Passing this memory check does not claim that every backward shader or driver
path works. Inspect a reduced request without initializing a GPU with:

```bash
VOLVOX_VULKAN_MB=1024 examples/target/bin/tiny_receipt_vqa_train \
  --dry-run --backend vulkan --batch-size 1 \
  --out /tmp/volvoxai-receipt-vulkan-plan
```

Until the complete graph has passed strict backward validation on the target
device, keep real training on CPU. If testing a GPU backend, use a headless or
non-display device: a Linux display-driver reset can terminate the desktop
session even when the kernel successfully resets the device.

The built-in C++ record generator reads both raw layouts directly:

- `synth/data/{train,val}/{ann,img}/STEM.{json,jpg}`
- `data/{annotations,images}/STEM.{json,jpg}`

Images are decoded through the shared `../native_support/image_io.c` helper and
`../../native/third_party/stb_image.h`, converted to
grayscale, bilinearly resized from the Navercap `670x320` files to the model's
fixed `672x320` input, and normalized to `[-1,1]`. The trainer deliberately
does not add online contrast, brightness, blur, or rotation: Navercap's
synthetic generator already applies perspective, rotation, paper curl,
lighting, blur, sensor noise, and JPEG compression, while `data/images`
provides real examples.

It covers the released `v1` phone/address tasks plus the meaningful
`structured_qa` task/operator set: store and address copying/character fields,
phone spans and arithmetic, positional item rows, name-based and absent-item
lookups, item products, sums, extrema, and reverse metric lookups. Its question
wording is intentionally compact. To use an exact externally generated wording
distribution, pass full record manifests:

```bash
examples/target/bin/tiny_receipt_vqa_train \
  --train-records /path/to/train_records.jsonl \
  --val-records /path/to/val_records.jsonl \
  --eval-batches 0 \
  --backend cpu \
  --out runs/volvoxai_tiny_receipt_vqa
```

Each JSONL row uses Navercap's existing `image`, `question`, `target`, `answer`,
`task`, `field`, `op`, `source`, `receipt_id`, and `family` fields.
Relative image paths are resolved against the manifest directory. The loader
consumes every JSONL row; `--eval-batches 0` evaluates the entire validation
manifest. Evaluation reports exact matches grouped by task, operation, source,
adapter family, and task/operation pair. Training uses supervised family routes;
evaluation defaults to learned automatic routing and reports `router_accuracy`.
Use `--eval-routing explicit` only when an oracle route is intentional.
`--max-steps`, when nonzero, limits complete optimizer updates per epoch; each
update still consumes `--accumulation-steps` physical batches.

## Smoke test

`--smoke` keeps the same multimodal/encoder-decoder/router/adapter/LoRA topology
but scales it to one layer and one batch item, then performs one real training
step and autoregressive evaluation:

```bash
make -C examples run_cpp_receipt_train_smoke NAVERCAP_ROOT=/path/to/navercap
```

Use `--dry-run` to validate data and emit the serialized `CreateModel` request
without loading the runtime. `make -C examples check_cpp_receipt_train_graph
NAVERCAP_ROOT=/path/to/navercap` performs that CPU-only check and cannot
initialize a GPU backend.

## Artifacts and resume

The output directory contains:

- `full_model/{config.json,model.safetensors,vocab.json}`: a directly loadable
  model package containing the trained inline LoRA and task-adapter tensors;
- `last.checkpoint/` and `best.checkpoint/`: graph topology, weights, AdamW
  moments, optimizer step, fixed cosine-schedule horizon, best metrics, and an
  embedded copy of the caller config/vocabulary contract;
- `lora_base/{config.json,model.safetensors,vocab.json}` plus
  `lora.safetensors`: a composable base-and-adapter pair. The base package has
  zero inline LoRA tensors, so loading and activating the official
  `volvox.adapter.v1` artifact applies the trained delta exactly once;
- `task_adapters.safetensors`: named encoder/decoder bottleneck expert tensors
  for transfer into the matching graph. They are already present and active in
  `full_model` and `lora_base`;
- `vocab.json`: caller-owned UTF-8 character vocabulary;
- `config.json`: the canonical caller/model/training contract.

At export time the example unloads the training model, reloads `lora_base`,
loads `lora.safetensors` through `LoadAdapter`, activates it, and compares its
logits with the inline-LoRA full model. A mismatch fails the run instead of
leaving an unverified adapter pair.

Resume the numeric graph and optimizer with:

```bash
examples/target/bin/tiny_receipt_vqa_train \
  --resume runs/volvoxai_tiny_receipt_vqa/last.checkpoint \
  --backend cpu \
  --out runs/volvoxai_tiny_receipt_vqa
```

Resume validates the newly loaded records' complete vocabulary, canonical
record fields, and referenced image contents, plus the caller/model/training
settings embedded in the checkpoint, before overwriting `vocab.json` or
`config.json`. The backend may be changed, but graph dimensions, data
cardinalities, optimizer settings, epoch horizon, and accumulation policy must
match. Best metrics and the original cosine horizon are restored rather than
restarted. Task adapters are ordinary trainable graph tensors because the
runtime adapter RPC currently represents LoRA only; restore them through
`full_model` or a full checkpoint.

## Architecture differences from a bit-identical PyTorch trajectory

The topology, operator-boundary shapes, losses, routing policy, optimizer,
clipping, and cosine schedule match the source workflow. Storage layouts and a
few parameter shapes differ where VolvoxAI uses equivalent native layouts.
Initial weights use VolvoxAI's
deterministic normal/Xavier initializers, and native training is F32. PyTorch's
prototype-layer cloning, online image augmentation, RNG streams, CUDA reduction
order, FP16 autocast, and GradScaler are therefore intentionally not bit-for-bit
reproduced.
