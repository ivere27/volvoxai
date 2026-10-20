# Run a Hugging Face decoder

Convert a local Hugging Face checkpoint into a VolvoxAI package, prepare its
token tensors in your application and execute the decoder on native CPU. The
first workflow uses F32 full-prefix execution: each new token recomputes the
prefix. An external tokenizer and the caller's greedy loop provide text input
and output; the engine runs the graph and owns its tensor snapshots.

The [compact decoder workflow](compact-decoder.md) exports direct GQA and reuses
private K/V through typed row transitions and engine greedy `Generate`.
That guide also explains the full-library CUDA route, additional TinyLlama and
public Llama 3.2 fine-tune checks, bounded Llama3 scaled RoPE exports and the
[original Qwen3-4B CUDA check](compact-decoder.md#original-qwen3-4b-on-cuda).

Python can also load a local dense Llama, Qwen2 or Qwen3 checkpoint directly,
using its original F32/F16/BF16 SafeTensors shards:

```python
import numpy as np
from volvoxai import InferenceSession

# Use IDs produced by this checkpoint's tokenizer.
token_ids = np.array([[1, 42, 7]], dtype=np.int32)
with InferenceSession.from_pretrained("/path/to/local/checkpoint",
        backend="cuda", max_context=128) as model:
    logits = model.run({"token_ids": token_ids,
        "position_ids": np.arange(token_ids.shape[1], dtype=np.int32),
        "attention_mask": np.ones_like(token_ids)})["logits"]
```

This local-only convenience builds the graph from configuration and shard
metadata. C loads and snapshots the original files, widening F16/BF16 for F32
execution. No F32 weight export or download occurs. Scaled RoPE creates a small
temporary shard containing shared position tables; loading retains its snapshot
and immediately removes the temporary files. `cached=True` selects required
incremental rows; supply `active_lengths` at `initialize`, then use explicit
transition scopes as in the compact workflow. The default evaluates each prefix.
The same convenience is available from the installed Python wheel.

Whisper uses a separate log-mel encoder and conditioned text decoder. The
[multimodal workflow](multimodal-workflows.md) starts from raw PCM or JPEG,
explains caller processor/task policy, and checks encoder reuse, immutable
conditioning and ID-to-tensor output with the public lifecycle.

Modern MoE and hybrid architectures also have explicit adapters. They preserve
the checkpoint's grouped attention, recurrent state equations and expert
routing. `--generate` declares full-prefix feedback; adapters with retained-state
support use `--cached-decoder` to retain
K/V, convolution history, recurrence and completed compressed blocks between
token rows. The cache workflow below uses a fixed input capacity and an explicit
active prompt length.
Small independent HF models exercise these adapters in both native CPU
profiles. Public WASM CPU and full-profile CUDA, Vulkan, OpenGL and WebGPU
fixtures cover their operators and generation. Metal has native routes and
shader translation checks; execution on Apple hardware remains unqualified.
This evidence establishes the tested small configurations; full published
checkpoints, device capacity and latency need separate qualification.

| Checkpoint architecture | Accepted configuration | Decoder features |
| --- | --- | --- |
| Qwen3 MoE | `qwen3_moe` | Q/K normalization, grouped attention, routed gated experts |
| Llama 4 | `llama4`, `llama4_text` | Interleaved RoPE, NoPE temperature, chunk masks, input-scaled routed and shared experts |
| Qwen3-Next | `qwen3_next` | Causal convolution, gated delta recurrence, packed attention gate, shared and routed experts |
| Qwen3.5 / compatible Qwen3.8-27B | `qwen3_5`, `qwen3_5_text`, corresponding `_moe` types | Hybrid recurrence/attention; optional image features and three-axis rotary positions |
| Qwen3.8-Flash-Next | `qwen4_exp`, `qwen4_exp_text` | Hyper-connections, hashed lexical embeddings and QSA block selection |
| DeepSeek V2 / V3 | `deepseek_v2`, `deepseek_v3` | Distinct MLA key/value widths, partial RoPE and grouped routed experts |
| DeepSeek V4 | `deepseek_v4` | mHC streams, shared/sliding KV, HCA/CSA compression and indexer selection |
| DeepSeek V4.1 Flash | `deepseek_v41`, `deepseek_v41_text` | F32 text backbone, CSA2 shared KV/index selection, hierarchical candidates, delayed mHC and Engram; full-prefix execution/Generate |
| Whisper | `whisper` | Separate log-mel encoder, learned positions and full-prefix conditioned decoder; explicit caller generation policy |

The importer follows `config.json`'s actual architecture. A marketing version
number does not select an alias: Qwen3.8-27B describes a Qwen3.5 text decoder,
whereas Flash-Next describes Qwen4Exp with different residual and attention
math. Unsupported structural flags and undeclared storage formats fail import.

The original dense qualification selects SmolLM2-135M (`LlamaForCausalLM`) and Qwen3-0.6B.
Qualification fixes their exact revisions, batch one, a single sequence and a
context bound of 128. This early full-prefix scope does not establish every
Llama/Qwen configuration, cached decoding, `Generate`, GPU execution or a
physical robot. Separate cached/device qualifications are recorded in the
[compact decoder guide](compact-decoder.md#reproduce-the-cached-numerical-check).

The dense importer also accepts `model_type=qwen2` with
`Qwen2ForCausalLM`, including full-attention Qwen2.5 and
DeepSeek-R1-Distill-Qwen checkpoints. It retains their Q/K/V projection biases,
bias-free output projection and unnormalized Q/K heads. Inactive
`sliding_window` metadata is normalized when `use_sliding_window=false`;
actual sliding layers remain an explicit refusal. These adapters are checked
against small independent HF models; they do not establish numerical or device
qualification for a downloaded Qwen2.5 or R1 checkpoint.

For smaller Linear weight storage, select `--weight-precision w4a16`,
`--weight-precision w4a8` or `--weight-precision w8a16` during HF export:

```sh
python3 tools/export_safetensors.py --model /path/to/local/llama --out build/llm/packages/llama-w4a16/model.safetensors --target native-cpu --max-context 128 --compact-kv --cached-decoder --weight-precision w4a16
```

The shared exporter packs each immutable Linear weight into group-128 INT4 or
INT8 storage. W4A16 and W8A16 round activations and decoded weights to IEEE FP16 before F32
accumulation. W4A8 uses dynamic symmetric INT8 activations with one scale per
row and exact integer dot products within each weight group. Graph inputs,
outputs and residuals remain F32. Embeddings, normalization and other operators
keep their original storage and arithmetic; a tied embedding retains its F32
matrix alongside the packed output projection. MoE expert operators are not
converted by this Linear pass. This explicit package format does not imply AWQ,
GPTQ or GGUF import compatibility. Check the selected backend and whole-model
quality before using a lossy precision mode in an application. For an existing
F32 package, the [Python PTQ workflow](../python/README.md#post-training-quantization)
provides the same packed choices with calibration coverage, per-node float
retention and a sensitivity report.

DeepSeek V2/V3 and V4 have separate causal-prefix adapters. V2/V3 expand MLA
latents into distinct key/value widths and preserve their rotary channel layout
and grouped expert routing. V4 uses mHC streams, shared KV, sliding attention,
HCA/CSA compression and indexer selection, inverse output RoPE and grouped
output projection. Export with an explicit `--max-context`; these adapters
support both full-prefix `--generate` and incremental `--cached-decoder`.
The V4 compressed cache requires `position_ids[i] == i` for active tokens;
positions ahead of the logical token order could expose subsequently completed
blocks to earlier queries and are rejected before execution.

The local reader accepts F32/F16/BF16 weights and preserves checked integer hash
metadata. It also converts declared DeepSeek E4M3FN blocks and packed E2M1 expert
weights with UE8M0 scales to F32. Conversion validates scale geometry and finite
values before publishing a package; the full decoded checkpoint must fit host
memory. It does not emulate activation or KV quantization. Auxiliary MTP layers
are excluded from the ordinary causal decoder. Small independent reference comparisons
exercise both native CPU profiles; actual large DeepSeek checkpoints and their
device capacity remain separate work. Equal router or indexer scores use the
portable lowest-index tie rule. HF `topk` does not promise the same tie choice,
so tied cases can produce different logits even when non-tied math agrees.
The corpus generator defaults to `--reference-policy raw-hf` and retains those
disagreements as failures. Its explicit `--reference-policy canonical-lowest-index`
mode keeps HF's scorer and numerical layers, but orders equal V4 indexer scores
by ascending compressed-block index. The corpus and validation receipts name
that policy: passing it proves the portable deterministic model definition,
not equality with an arbitrary `torch.topk` tie choice. Exact ties are never
resolved by perturbing scores, changing seeds or increasing numerical tolerances.
See the [reference workflow](../tools/llm_baseline/README.md#deepseek-v4-reference-policies).

DeepSeek V4 imports accept both Transformers names and the official release's
`embed`/`layers.*.attn`/`layers.*.ffn` names. Mixing the two namespaces is refused.
Declared auxiliary draft stages are excluded from the ordinary causal decoder.

DeepSeek V4.1 Flash has a separate text adapter. It follows the official
[pinned architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/5254448103b6254fc646dacbca1cbbf20e9216c1),
including per-layer Full/Reindex/Reuse attention, the decoder's candidate pool,
Engram's tokenizer normalization and the delayed mHC input mixture. Export a
local checkpoint with an explicit context bound:

```sh
python3 tools/export_safetensors.py --model /models/deepseek-v41 --out build/llm/deepseek-v41-f32/model.safetensors \
  --max-context 128 --generate --target backend:cuda
```

The local directory must contain `config.json`, `tokenizer.json` when Engram is
enabled, and its SafeTensors weights/index. F32 execution supports text prefixes
and engine `Generate`; use contiguous zero-based position IDs and an all-one
attention mask. `--cached-decoder` is refused because compressed-row updates
need a distinct cache proof. Image encoding/routing and DSpark speculative
decoding are outside this text adapter. Small pinned-source reference models
qualify the F32 architecture; they do not qualify the published 552B checkpoint,
its FP8/FP4 activation/cache path, maximum context, memory fit or speed.
The lowest-index tie rule also applies to V4.1 index keys and candidate blocks.
Its reference fixtures record raw official selection separately from the
explicit canonical policy; a canonical pass does not establish raw `torch.topk`
identity on tied scores.

## Prepare the checkpoint and runtime

Follow the [native build guide](native-runtime.md#build) to build inference/full
libraries and the loader. Install the repository's
[Python dependencies](../python/requirements-test.txt). Download and conversion
are development tools; the engine itself does not depend on Transformers or
PyTorch. This example additionally uses Transformers for caller tokenization.
The [reference guide](../tools/llm_baseline/README.md) records the fixed reference
environment and revision/config/tokenizer/weight hashes.

Run from the repository root. The download command verifies the actual pinned
snapshot bytes and keeps its cache under the explicit snapshot directory:

```sh
python3 tools/llm_reference.py download --model smollm2-135m --snapshot build/llm/snapshots/smollm2-135m
python3 tools/llm_reference.py download --model qwen3-0.6b --snapshot build/llm/snapshots/qwen3-0.6b
```

| Checkpoint | Fixed revision | Imported F32 weight file |
| --- | --- | --- |
| HuggingFaceTB/SmolLM2-135M | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 538,092,856 bytes |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | 2,384,243,312 bytes |

The source snapshots use BF16 storage. Conversion expands values to F32 and
shares tied embedding/output-projection weights. These are F32 execution
packages; they do not demonstrate BF16 arithmetic support.

## Convert to a package

Choose fresh output directories. The exporter refuses to replace an existing
package. `--target native-cpu` validates the graph against the native CPU
contract; runtime compilation still validates the actual artifact and device.
Modern adapters can also select `--target wasm`, `--target webgpu`,
`--target backend:cuda`, `--target backend:vulkan`, `--target backend:opengl` or
`--target backend:metal`. These targets validate individual exporter subsets.
`portable` requires the intersection of WASM, WebGPU and native CPU; it does
not include CUDA, Vulkan, OpenGL or Metal. Repeating `--target` requires every
selected subset. Exporter acceptance does not replace runtime compilation's
proof of the complete graph, its shape/index domain and the actual device's
limits. Native GPU execution requires the full native profile with that backend
compiled; WebGPU requires `FullEngineHost` and its full WASM companion. Native
CPU supports the kernels through portable scalar C; AVX2 is optional. The
native CPU comparisons here ran on x86; ARM execution has not been qualified.

```sh
python3 tools/export_safetensors.py --model build/llm/snapshots/smollm2-135m --out build/llm/packages/smollm2-135m/model.safetensors --target native-cpu --max-context 128 --capture-kv --report build/llm/packages/smollm2-135m/export-report.json --report-format json
python3 tools/export_safetensors.py --model build/llm/snapshots/qwen3-0.6b --out build/llm/packages/qwen3-0.6b/model.safetensors --target native-cpu --max-context 128 --capture-kv --report build/llm/packages/qwen3-0.6b/export-report.json --report-format json
```

Each package contains `graph.json` and `model.safetensors`. The graph declares
these named I32 inputs, with `1 <= S <= 128`:

| Input | Shape | Caller supplies |
| --- | --- | --- |
| `token_ids` | `[1,S]` | Tokenizer IDs for the complete prefix |
| `position_ids` | `[S]` | Absolute positions for that prefix |
| `attention_mask` | `[1,S]` | One for a valid position, zero for masked padding |

The principal output is F32 `logits: [1,S,V]`. `--capture-kv` additionally
exposes each layer's `layer.N.k_rope` and `layer.N.v` in `[1,S,Hkv*D]` layout for
numerical comparison. Key outputs follow RoPE and any configured Q/K normalization. Omit that
flag for a package with only logits, or select only logits at execution as below.

This full-prefix attention path materializes repeated K/V heads with Gather before
ordinary attention. Compact layer outputs do not establish compact-cache
attention or KV reuse. Unsupported architecture, bias, RoPE scaling or sliding
settings are refused explicitly in this repeated-head workflow; the importer
does not silently ignore them. For `rope_type=llama3`, use the
[compact workflow](compact-decoder.md#additional-llama-checkpoints-and-scaled-positions)
with an explicit context bound and compact KV.

Qwen2 linear and YaRN scaled RoPE likewise require `--compact-kv` and an explicit
`--max-context`. They use fixed F32 cosine/sine lookup tables shared by all query
and key heads, with `RotaryEmbedding` applying each rotation. Absolute positions
remain within that declared range. For a local
Qwen2.5 checkpoint configured with YaRN, for example:

```sh
python3 tools/export_safetensors.py --model /path/to/local/qwen2.5 --out build/llm/packages/qwen2.5/model.safetensors --target native-cpu --max-context 128 --compact-kv --cached --report build/llm/packages/qwen2.5/export-report.json --report-format json
```

The lowering follows the [official Qwen2 attention definition](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2/modeling_qwen2.py)
and [linear/YaRN frequency definitions](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/modeling_rope_utils.py).
The shared tables have a 1 GiB export bound and occupy
`8 × max_context × head_dim` bytes regardless of head or layer count. No dense
rotation matrix is stored or evaluated. Sequence-dependent dynamic
NTK, custom YaRN `mscale` variants and actual sliding attention are refused.
Tokenization, chat templates and stopping policy are separate from graph import.

## Generate four tokens with a caller loop

Save this as a Python script in the repository and run it with
`PYTHONPATH=python python3 path/to/script.py`. Tokenizer loading is local-only;
after preparation, this workflow needs no network.

```python
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import volvoxai as vx

alias = "smollm2-135m"  # Or "qwen3-0.6b".
snapshot = Path("build/llm/snapshots") / alias
package = Path("build/llm/packages") / alias
tokenizer = AutoTokenizer.from_pretrained(
    str(snapshot), local_files_only=True, trust_remote_code=False)
ids = np.array([tokenizer.encode(
    "The future of on-device AI is", add_special_tokens=False)], dtype=np.int32)
steps = 4
if ids.shape[1] < 1 or ids.shape[1] + steps > 128:
    raise ValueError("Choose a nonempty prefix that fits the package's context bound.")

generated = []
with vx.Runtime(cpu_threads=2, budget=vx.pb.RuntimeBudget(
        max_unconsumed_result_bytes=128 * 1024 * 1024)) as runtime:
    with runtime.inference_session(package, backend="cpu") as session:
        for _ in range(steps):
            inputs = {
                "token_ids": ids,
                "position_ids": np.arange(ids.shape[1], dtype=np.int32),
                "attention_mask": np.ones(ids.shape, dtype=np.int32),
            }
            logits = session.run(inputs, output_names=["logits"])["logits"]
            next_id = int(logits[0, -1].argmax())
            generated.append(next_id)
            ids = np.concatenate((ids, np.array([[next_id]], dtype=np.int32)), axis=1)
print(tokenizer.decode(generated))
```

This is fixed-count greedy generation with the first-index tie rule. It does
not apply a chat template or stop at EOS. Those decisions remain caller policy.
The graph receives a complete prefix on every call; this loop does not invoke
the engine's `Generate` operation or carry a decoder cache between calls.

The runtime's default unconsumed-result budget is 64 MiB. Admission reserves
selected outputs over the graph's declared maximum shape, even if the first
request contains only one token. Qwen3's maximum logits require 77,791,232 bytes
(`128 * 151936 * 4`), so this example explicitly allows 128 MiB and selects only
logits. All Qwen3 layer K/V captures add 29,360,128 bytes, for 107,151,360 bytes
in total; SmolLM2's all-output maximum is 31,064,064 bytes.

`session.run` copies selected results into independently owned NumPy arrays and
releases their engine buffers before returning. Runtime/session context managers
release execution, model and compiled resources. Existing arrays stay valid
after later calls and close. See [buffers and tensors](buffers-and-tensors.md)
for retained native tensors and explicit resource ownership.

## Generate with a modern decoder

Export a local floating checkpoint into a fresh directory:

```sh
python3 tools/export_safetensors.py --model /path/to/local/checkpoint --out build/llm/modern/model.safetensors --target native-cpu --max-context 128 --generate --report build/llm/modern/export-report.json --report-format json
```

`--generate` selects the existing compact cached route for dense Llama/Qwen
decoders. For the modern hybrid/MoE and DeepSeek adapters it declares full-prefix
feedback: each step resets recurrent state and evaluates all preceding tokens.
`--cached-decoder` selects incremental generation for these adapters except
DeepSeek V4.1, which currently supports full-prefix execution and generation.
The export report records the selected execution mode.

For modern full-prefix generation, supply the exact active prompt length,
without padding, and an all-one `attention_mask`. The engine grows the token,
keep and rotary-position tensors. Llama4 additionally takes `sequence_ids`
(`arange(S)`) and, for chunked layers, `chunk_attention_mask` of shape `[1,S,S]`:
one means the two sequence positions belong to the same chunk. These sequence
controls are distinct from the caller's absolute rotary positions. Generate
updates all declared controls and retains only fixed conditioning inputs.

Use the public service-client lifecycle in the
[compact decoder example](compact-decoder.md#generate-ids-while-reusing-the-cache),
with this package's exact prompt tensors in `GenerationInputs`; omit its
`active_lengths` and cache options for the modern full-prefix package. The
generated IDs are named `generated_ids`. Budget each call with `GenerationBudget`
and leave `initial_inputs` absent after the first call. See the
[public schema](../proto/volvoxai.proto) for cancellation, sampling and output
ownership contracts.

CPU/WASM CPU and CUDA support GREEDY and seeded RANDOM generation. Modern
full-prefix packages also support GREEDY on Vulkan, OpenGL, Metal and WebGPU,
with EOS IDs, stop IDs and stop sequences. Matching includes newly generated
IDs, keeps the matching IDs in the terminal output and can span calls with
different step budgets. RANDOM and its temperature/top-k/top-p settings remain
unsupported on these four backends. Use INLINE outputs for WebGPU: the call
awaits its C continuation and returns retained ID bytes; OWNED transport is
refused before context mutation. This full-prefix support does not add stop
policies to the existing WebGPU incremental row profile.

Select the actual backend with a required compilation policy and forbid
operator fallback when checking device execution. The graph stays on that
device; C reads one four-byte selected ID per prefix step for feedback and
stopping. The graph's maximum context bound still limits generation. Scratch
storage is reused across layers and prefixes, while the model weights and
full-prefix activations must fit the chosen device.

To reuse the modern decoder's cache, export with `--cached-decoder`:

```sh
python3 tools/export_safetensors.py --model /path/to/local/checkpoint --out build/llm/modern-cache/model.safetensors --target native-cpu --max-context 128 --cached-decoder
```

Prepare batch-one token and mask arrays at the chosen fixed capacity `S`, and
set `active_lengths = [prompt_length]`. The active prompt occupies the beginning
of the arrays; its mask values are one, and unused positions have mask zero.
Supply `position_ids` as `arange(S)`. Include the same Llama4 sequence and chunk
controls as above, sized for this capacity. The context keeps that geometry
through all continuation calls; its capacity cannot grow after initialization.

Use the [cached Generate lifecycle](compact-decoder.md#generate-ids-while-reusing-the-cache)
with this package's input tensors. C advances the new token's controls and
preserves the earlier rows. For direct Graph execution, call `initialize` with
FULL scope, then `transition` with one SLOT_ROWS position and REQUIRED row mode.
Reuse every declared invariant input, including `active_lengths` and image
features, rather than resubmitting it in a transition. Edits to already cached
tokens or controls require a new context and prompt initialization.

These cache transitions require SLOT_ROWS; dependency-only execution is refused
because it could recompute retained recurrent state. Supply I32 token and control
inputs as explicit host values at initialization. Output references cannot bind
those inputs or replace transition inputs; immutable F32 conditioning may still
come from an output reference. DeepSeek V4 compressed cache additionally requires
`position_ids[i] == i` throughout the active prefix.

For these modern hybrid/MoE and DeepSeek cache contracts, CPU/WASM CPU and CUDA
retain GREEDY and seeded RANDOM selection. Vulkan, OpenGL and Metal cached
generation supports GREEDY. WebGPU follows its existing row profile: GREEDY
with INLINE outputs, without additional stop policies or RANDOM.
Metal has implementation and shader translation checks; Apple device execution
remains unqualified. Full-prefix generation keeps the separate policies above.
WebGPU uses `FullEngineHost` and its full WASM companion; the lite web profile
retains its CPU backend. Native device backends also require the full profile.
Dense Llama/Qwen compact-cache `Generate` has a separate admission path and
remains unsupported on native Vulkan, OpenGL and Metal; its direct Graph row
execution is described in the [compact guide](compact-decoder.md).

Image-conditioned packages keep their prepared features and prompt image
routing. Generated text rows clear image routing; Qwen's three rotary planes
advance together from the largest active multimodal position. The image encoder
remains a separate reusable package. Cache reuse reduces repeated token work;
the retained graph activations and declared maximum workspaces still need to
fit the chosen backend.

## Run an image encoder and image-conditioned decoder

Llama4 and Qwen3.5-compatible multimodal checkpoints export two ordinary model
packages. The first produces image features; the second consumes those features
at explicitly routed prompt positions. Both execute through the same native
graph API as text models.

For Qwen, choose the fixed temporal/height/width grid returned by the official
checkpoint processor. This example uses one `1 × 32 × 32` grid:

```sh
python3 tools/export_safetensors.py --model /path/to/local/qwen-checkpoint --out build/llm/qwen-image/model.safetensors --target native-cpu --hf-vision-grid '[[1,32,32]]'
python3 tools/export_safetensors.py --model /path/to/local/qwen-checkpoint --out build/llm/qwen-text/model.safetensors --target native-cpu --max-context 2048 --hf-image-features 256 --generate
```

The encoder input `pixel_patches` is F32 `[N,C*T*P*P]`, in the official
processor's flattened patch order. The fixed grid determines frame attention,
bilinear learned positions, axial RoPE and spatial merging. With merge size
two, the example's 1,024 patches produce 256 feature rows. Grids must be
positive and spatially divisible by the merge size; the exporter checks its
bounded attention-mask and tensor allocation limits.

For Llama4, select a fixed number of normalized image tiles:

```sh
python3 tools/export_safetensors.py --model /path/to/local/llama4-checkpoint --out build/llm/llama4-image/model.safetensors --target native-cpu --hf-vision-tiles 1
python3 tools/export_safetensors.py --model /path/to/local/llama4-checkpoint --out build/llm/llama4-text/model.safetensors --target native-cpu --max-context 2048 --hf-image-features 256 --generate
```

For the standard 448-pixel image, 14-pixel patches and half-size pixel shuffle,
each tile produces 256 rows. The encoder takes F32 `pixel_values` in normalized
NCHW `[tiles,C,image_size,image_size]` layout, then runs class-token attention,
pixel shuffle, adapter and multimodal projector. Other admitted geometries
determine their own feature count; inspect the encoder output descriptor.

Run the encoder once with `session.run` and pass its `image_features` F32
`[M,hidden_size]` output to the text package. Alongside normal token, keep and
position inputs, supply:

| Decoder input | Shape / type | Meaning |
| --- | --- | --- |
| `image_feature_ids` | `[S]` I32 | Feature-row index; use zero at ordinary text positions |
| `image_token_mask` | `[1,S]` I32 | One replaces the token embedding with that feature row; zero keeps the text embedding |
| `mrope_position_ids` | `[3,S]` I32, Qwen only | The processor's temporal, height and width rotary positions |

Use the processor's image placeholder order to build the two routing arrays.
Qwen's image rotary positions can differ across axes; its separate
`position_ids` remains the text sequence control. Full-prefix Generate retains
the feature matrix, appends zero routing for new text tokens and advances every
multimodal axis from one plus the maximum existing axis position.

The application owns decoding images, resizing, normalization, tile selection,
the tokenizer and chat template. The packages start from the normalized tensor
boundary. Deep-stack feature connections, unsupported rotary variants and raw
FP8/FP4 storage in these Qwen/Llama vision adapters are explicit refusals. Full published model loading also remains
subject to graph, tensor, address-space and device-memory limits; architecture
support alone does not establish that a large checkpoint fits a laptop or GPU.

## Reproduce the numerical comparison

The fixed comparison uses two prompts, prefix lengths 1/16/31 and four
continuation steps per prefix. HF captures include prefill logits, cached-step
logits and every layer's compact K/V. VolvoxAI executes each corresponding full
prefix and compares its selected rows and layer state against those references.

Generate references with the commands in the
[reference guide](../tools/llm_baseline/README.md), then compare a capture-enabled
package using a new report filename:

```sh
PYTHONPATH=python python3 tools/llm_compare.py --package build/llm/packages/smollm2-135m --reference build/llm/reference-v2/smollm2-135m --profile full --backend cpu --out build/llm/smollm2-cpu-full-comparison.json
PYTHONPATH=python python3 tools/llm_compare.py --package build/llm/packages/qwen3-0.6b --reference build/llm/reference-v2/qwen3-0.6b --profile inference --backend cpu --out build/llm/qwen3-cpu-inference-comparison.json
```

Run each model with both `--profile inference` and `--profile full` to cover both
native libraries. The comparison reserves the exact maximum output bytes,
records artifact/input/reference identities and applies the fixed budgets:
logits `atol=2e-3, rtol=1e-3`, RMS at most `5e-4`; layer K/V
`atol=1e-3, rtol=5e-4`; greedy IDs exact.

These runs capture numerical and resource evidence. They do not establish
latency, peak whole-process/device memory limits, cached decoding, CUDA/browser
model execution or robot support. Those require separate workloads and device
qualification. The [sensor replay guide](native-sensor-replay.md) covers the
small prepared-sensor fixture separately.

The modern importer regressions construct small deterministic models directly
from the official HF classes and compare all active logits, greedy IDs and image
features with both native profiles. They cover mixed dense/routed layers,
absolute rotary positions, padding, chunking, recurrent convolution, lexical
EOS boundaries and separate image-conditioned execution:

```sh
python3 -m unittest tools.tests.test_hf_extended_importer tools.tests.test_hf_multimodal_importer -v
PYTHONPATH=python python3 -m unittest python.tests.test_hf_incremental_generation -v
```

The [final pinned native receipt](../build/llm/architecture-expansion/hf-pinned-final-receipt.json)
records 60 passing tests on Clang14/LLD14 inference/full artifacts, including
58 logits and 14 image-feature outputs, DeepSeek and dense regressions, and
workspace admission. Nine small decoder families also produce
54 exact HF greedy IDs through 36 public native `Generate` calls, split into
budgets of one and two. The [WASM receipt](../build/llm/architecture-expansion/wasm-modern.json)
records 35 numerical Execute cases and nine HF generation cases per profile,
88 passing results in total. Those WASM cases exercise the operators and
small text generators; they do not qualify a full image processor or a large
browser checkpoint. The [final CUDA receipt](../build/llm/modern-primitives-public/actual-cuda-pinned-final-006/transfer-receipt.json)
records 19 passing primitive/MoE/Where/Gather tests and all nine decoder
families, including Qwen4 n-gram composition. INLINE and OWNED generation
produce 54 exact HF IDs across 36 calls, with a required CUDA backend, no
fallback and retained outputs checked after context, compiled-model, model and
runtime retirement.

These CPU/WASM/CUDA receipts preserve the original qualification. The subsequent
[native Vulkan/OpenGL receipt](../build/llm/portable-gpu-expansion/actual-native-002/manifest.json)
records physical RTX 3090 execution on each backend: 36 numerical Execute
calls and 36 Generate calls across nine small decoder families, producing
54 exact HF IDs. Each backend also checks 18 OWNED outputs after owner
retirement. The [WebGPU receipt](../build/llm/portable-gpu-expansion/webgpu-final-005.json)
records 35 numerical Execute calls, nine decoder families and 45 Generate calls
with 54 exact IDs, including EOS and stop sequences across call boundaries.
Its 124 route attestations require WebGPU with fallback forbidden; 44 OWNED
requests and nine RANDOM configurations are explicitly refused, and 80 INLINE
snapshots remain valid after owners retire. Floating primitive expectations
use independent NumPy F64 values with unchanged `atol=rtol=1e-5`; integer
results and generation IDs match exactly.

The [Generate lifecycle evidence](../build/llm/portable-gpu-expansion/webgpu-generation-lifecycle-summary.json)
checks expired admission, cancellation, committed deadlines, pending owner
release, transport abort and host shutdown on the actual device. Deadlines
preserve proved execution evidence. Transport abort and shutdown drain the
finite admitted chunk and retire its GPU storage. The complete multi-device
Deno invocation remains failed: a standalone 48-byte copy reproduces the
second-device failure without this library. The final two lifecycle cases
passed in fresh processes; this does not qualify browser device recreation.

The [raw shader receipt](../build/llm/portable-shader-check/numerical-004.json)
adds 41 cases and 123 dispatches on a software Vulkan adapter, including
large integer hashes, a nonindexed compressed bank larger than 512 rows and
gated-expert controls. It is numerical unit evidence. The
[translation receipt](../build/llm/portable-shader-check/translation-final-002.json)
records 17 WGSL modules translated and validated to SPIR-V, GLSL, GLES and MSL
(68 checks). Metal's native implementation and MSL translation have no Apple
device qualification yet.

Existing dense WebGPU head RoPE, GQA and Q/K norm with head widths up to 128
retain their separate admission. Its incremental row graphs support greedy
generation and a step limit; RANDOM and additional row EOS/ID stops are
refused. The new expert, recurrent and sparse routes serve admitted modern F32
graphs and full-prefix generation. The earlier profile, provenance, ABI and
size checks cover the [eight final release artifacts](../build/llm/portable-gpu-expansion/release-final.log).
These small fixtures do not qualify full published checkpoints, a complete
multimodal processor, device capacity or performance. Runtime activation/cache
quantization and native low-precision arithmetic remain deferred in
[TODO](../TODO.md).

The Qwen4Exp reference additionally pins official Transformers revision
`cb33194ad6152bd9fad6305378d92db385dd7b32`. Place its
[`configuration_qwen4_exp.py`](https://github.com/huggingface/transformers/blob/cb33194ad6152bd9fad6305378d92db385dd7b32/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py),
[`modeling_qwen4_exp.py`](https://github.com/huggingface/transformers/blob/cb33194ad6152bd9fad6305378d92db385dd7b32/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
[`vision_utils.py`](https://github.com/huggingface/transformers/blob/cb33194ad6152bd9fad6305378d92db385dd7b32/src/transformers/vision_utils.py)
and [`utils/generic.py`](https://github.com/huggingface/transformers/blob/cb33194ad6152bd9fad6305378d92db385dd7b32/src/transformers/utils/generic.py)
in a reference directory; rename the last two to `qwen4_vision_utils.py` and
`qwen4_generic.py`. The fixture verifies each source SHA before importing its
unmodified arithmetic and supplies only the missing version-specific class and
helper associations for Transformers 5.13:

```sh
VOLVOXAI_QWEN4_REFERENCE_DIR=/path/to/pinned/reference python3 -m unittest tools.tests.test_hf_extended_importer -v
```
