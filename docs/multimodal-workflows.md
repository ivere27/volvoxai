# Run image and audio models

VolvoxAI runs tensor graphs. Applications decode media, apply the model's
processor and choose its prompt and output policy. A separate encoder computes
conditioning once; the decoder then reuses that conditioning to produce IDs.
For text, the application decodes those IDs with the matching tokenizer. A
nontext decoder can instead map categorical IDs into ordinary tensors.

The workflows here qualify two actual small checkpoints on native CPU, with
both inference and full libraries, and native CUDA, Vulkan and OpenGL with the
full library on an NVIDIA RTX 3090. Actual Chrome checks also cover both WASM
CPU profiles and the full WebGPU profile on that GPU, using raw PCM and JPEG
inputs. Native callers also check a fixed synthetic single-codebook example.
This scope does not establish arbitrary multimodal
architectures, pretrained nontext models, multiple codebooks or media quality.
The existing Llama4 and Qwen3.5 image adapters have separate
[prepared image graph workflows](hf-decoder.md#run-an-image-encoder-and-image-conditioned-decoder).

## Prepare the runtime and pinned assets

Build the two profiles using the [native runtime guide](native-runtime.md#build).
The validation caller additionally needs NumPy, Pillow, SafeTensors, PyTorch,
Transformers and ONNX Runtime. Only the development caller and independent
references use those packages; the native engine acquires no dependency on them.

The [multimodal baseline](../tools/llm_baseline/multimodal_sensor.json) pins every
selected model, tokenizer, processor and source asset by revision and SHA-256.

| Workflow | Checkpoint | Revision |
| --- | --- | --- |
| Receipt image/question | `ivere27/tiny-receipt-vqa-structured-qa-21m` | `f536c6ed181ca88929214f0bbc08f126f148c7ae` |
| Audio/text | `openai/whisper-tiny` | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |

Whisper's pinned SafeTensors file is 151,061,672 bytes. The caller uses an
explicit snapshot directory, verifies the downloaded bytes, and loads the HF
reference offline. It does not write another Hub weight cache. Temporary split
packages are removed after each profile. With the selected source cached, the
full validation is:

```sh
python3 tools/llm_baseline/verify_multimodal_native.py \
  --download-whisper --whisper-snapshot /tmp/volvoxai-h-stage-whisper-169d4a \
  --out build/llm/h-stage/actual-multimodal.json
```

The image workflow also needs the pinned source and a converted F32 package.
Follow the TinyReceiptVQA [download and import instructions](../examples/tiny_receipt_vqa/README.md).
Pass the resulting package using `--vqa-package build/tiny-receipt-kv-f32`.
The current cached qualification package is
`build/receipt-vqa-ptq-head-29ffb7e/fp32`. Use `--workflows audio`,
`--workflows image` or `--workflows codebook` to select one workflow.

## Raw JPEG to a receipt answer

The caller starts with the pinned English and Korean JPEG files and ordinary
question strings. [prepare_receipt_image](../tools/receipt_image.py) decodes
grayscale `L`, resizes with Pillow bilinear to 672 by 320, and computes
`(pixel/255 - 0.5)/0.5`. The encoder receives independent F32 NCHW pixels with
shape `[1,1,320,672]`. The caller applies NFC/whitespace normalization and the
release's byte-fallback BPE1536, including its EOS and 192-ID question limit.

Pixel tensors and question IDs must match the pinned producer's processor
exactly. The original ONNX encoder and decoder provide independent model
results. Both native profiles compare every encoder output and every decoder
logit row at `atol=rtol=1e-4`; all selected IDs must match exactly through EOS.
The English example produces 16 new IDs and the Korean example 13, including
EOS. These are reference parity observations, not general receipt accuracy.

The image encoder runs once per question. Its outputs include the selected
router family, memory mask and four layers of cross K/V. The decoder accepts
one current token plus explicit past K/V, and appends one cache row per call.
This workflow uses caller greedy decoding through `Execute`. Its one-token
external-KV graph is not a whole-prefix `Generate` graph.

The caller declares `initialize` and `transition` using the graph's tensor
feedback metadata. It preserves the producer's positive empty-cache sentinel:
one zero row with a blocked past mask. After initialization, cross K/V, memory
mask and family are invariant inputs supplied by `reuse`. The test overwrites
the caller's image and conditioning arrays between decoder calls and checks
that the accepted state and independent OWNED encoder snapshots remain intact.

## Raw PCM to conditioned Whisper generation

[prepare_whisper_features](../tools/whisper_audio.py) accepts a finite,
nonempty mono F32 PCM array at 16 kHz. The selected processor right-pads or
truncates to 480,000 samples, uses a 400-sample periodic Hann window with
160-sample hops, reflection padding, and the pinned 80-bin Slaney mel filter
matrix. Its normalized log-mel output is `[1,80,3000]`. Implicit resampling and
multichannel input are refused; applications perform those conversions first.

The primary feature oracle is the official HF **NumPy CPU** branch, with the
fixed `1e-5` absolute/relative tolerance. Transformers' default torch FFT branch
is observed separately: the selected quiet two-tone fixture differs by about
`3.19e-5`. That observation does not widen the primary gate or establish torch
FFT parity at `1e-5`.

The selected raw PCM fixture is a known 1.5-second 220/730 Hz two-tone waveform.
It establishes processor, model and generation parity with the fixed reference;
it does not rate speech recognition quality.

The [Whisper importer](../tools/exporter/hf_whisper.py) emits ordinary F32
Conv2D, GELU, learned embeddings, LayerNorm, Linear and CrossSDPA graphs.
The encoder produces `encoder_memory` with shape `[1,1500,384]`; the decoder
accepts that immutable conditioning and explicit token IDs, positions and a
keep mask. `Generate` recomputes only the supplied decoder prefix, without
rerunning the encoder or claiming incremental K/V execution.

Export the two packages into fresh directories:

```sh
python3 tools/export_safetensors.py \
  --model /tmp/volvoxai-h-stage-whisper-169d4a \
  --out build/whisper/encoder/model.safetensors --target native-cpu \
  --hf-whisper-encoder
python3 tools/export_safetensors.py \
  --model /tmp/volvoxai-h-stage-whisper-169d4a \
  --out build/whisper/decoder/model.safetensors --target native-cpu \
  --max-context 16 --generate
```

These CLI packages expose raw forward logits and raw greedy selection. They do
not automatically apply Whisper task or timestamp policy. The validation caller
builds its declared selection policy explicitly with ordinary `Where` masks:
English/transcribe/no-timestamps prefix `[50258,50259,50359,50363]`, checkpoint
`suppress_tokens` plus timestamp IDs `50364..51864`, no begin-only suppression,
greedy selection, and EOS `50257`. It compares against HF forward evaluation
and the official `SuppressTokensLogitsProcessor` with the same fixed policy.
It does not claim default HF `Whisper.generate`, beam search, VAD, automatic
language detection, temperature fallback or transcription quality.

The fixed encoder gate is `atol=rtol=2e-4`, with RMS error at most `2e-5`.
Raw decoder logits use `atol=2e-3`, `rtol=1e-3`, with RMS at most `5e-4`.
Generated IDs are exact. These budgets were fixed before candidate execution.
The initial encoder failure exposed long-key F32 attention accumulation error;
fixed 48-key partial sums reduce it without FP64 or a score matrix. Independent
F64 tests include key permutations, masks, wider grouped heads and exact
row-versus-full consistency. CUDA multi-head and compact grouped CrossSDPA use
bounded 48-key scalar partial reductions too; their independent F64 regression
includes masks, all-masked rows, key reversals and 47/48/49-key boundaries.
A mismatch does not widen a numerical budget.

The CPU caller exercises both OWNED encoder conditioning and caller-owned
borrowed conditioning. It splits generation across requests, reuses source arrays after
acceptance, checks the conditioning snapshot byte-for-byte and verifies that
the encoder executed only once. INLINE outputs must remain readable after
every model, compiled model, context, buffer and runtime owner is released.

## Run the selected workflows in Chrome

The development browser caller accepts the same raw F32 PCM and the two pinned
JPEGs. Its Whisper STFT/log-mel preparation runs in JavaScript outside the
product runtime and keeps the pinned F64 Slaney filter taps and complex64 FFT
rounding boundary. Independent HF NumPy checks retain the original feature
budget; all encoder and decoder budgets and expected IDs stay unchanged.

The JPEG branch uses Chrome `createImageBitmap` with color conversion disabled,
then the pinned Pillow integer RGB-to-gray coefficients and F32 normalization.
Both selected JPEGs already have the exact 672 by 320 geometry, and their pixel
hashes match the pinned Pillow processor exactly. This qualification covers
those exact-size images; the caller refuses other dimensions rather than
silently substituting a browser resize algorithm. The question strings pass
through the pinned byte-fallback BPE and produce the same English/Korean IDs.

WASM CPU retains OWNED encoder snapshots and uses the public buffer-copy API
for validation. Browser WebGPU does not currently support OWNED_BUFFER output.
It returns independent INLINE snapshots after asynchronous device completion;
these bytes already include the encoder readback. The caller releases the
encoder context, supplies that host conditioning to the decoder, and uploads it
to the selected backend. The audio handoff is 2,304,000 bytes. This is an
explicit host handoff, not a device-resident invariant or zero-copy path.

The browser caller checks conditioning after source mutation, encoder context
retirement and later decoder requests. VQA appends exact K/V prefixes and
reuses accepted invariant inputs. Retained INLINE results must also survive
release of the entire host. A WebGPU OWNED_BUFFER admission test requires a
typed refusal with unchanged context state before retrying valid INLINE output.

Prepare the independent native replay bundle once, then project its unchanged
58 HF/ORT arrays into the browser fixture container:

```sh
python3 tools/llm_baseline/prepare_multimodal_replay.py \
  --whisper-snapshot /tmp/pinned-whisper-tiny \
  --out build/browser-media/reference
python3 tools/llm_baseline/prepare_browser_multimodal.py \
  --bundle build/browser-media/reference \
  --out build/browser-media/projection.json
node tools/llm_baseline/verify_browser_multimodal.mjs \
  --bundle build/browser-media/reference --backend wasm --profile inference \
  --mode correctness --observe true --rounds 1 \
  --out build/browser-media/inference-wasm.json
```

For the full CPU profile, use `--profile full`. Hardware browser WebGPU uses
`--backend webgpu --profile full` and refuses software/fallback adapters.
Supply the Chrome executable explicitly with `--chrome` when needed; the driver
pins its actual Node executable automatically. The browser needs the matching
fixed release bundle and companion WASM.

Correctness, timing and memory receipts are separate. Timing uses
`--mode timing --observe false --rounds 8`: one fresh-host first workflow, two
warmups, then five measured workflows. Three fresh browser processes supply
15 warm observations per cell. No numerical oracle is fetched, and RSS,
profiling snapshots and GPU-buffer hooks are disabled. Host-wide swap counters
must remain unchanged. Load/compile, caller preparation and named public
execution, copy and generation calls
and the wall time of each workflow remain separately labelled; the latter
includes input integrity and lifetime checks and is not engine-only latency.
Memory uses a separate fresh process with `--mode timing --observe true`.
Allocation inventories contain at most 128 rows with explicit truncation;
decoder snapshots cover the first and final token and owner boundaries.
The final retired-owner inventory must be untruncated. Browser RSS includes
the caller and these retained observer records. WASM capacity, known engine
allocation inventory, GPU API buffer capacity,
sampled browser RSS/PSS and sampled NVIDIA driver residency remain separate
views, with no exact physical-peak claim.

On the pinned RTX 3090 host, three fresh workers with five measured rounds each
give full WASM CPU/WebGPU median complete-workflow times of 16.237/1.606 seconds
for raw PCM audio, 1.243/1.313 seconds for English JPEG VQA and 1.196/1.107 seconds
for Korean JPEG VQA. These include caller processing, portable conditioning
transfers and ID/cache/lifetime checks; loading and compilation are separate.
The small VQA graph uses explicit K/V readback on each decoder step, so these
numbers describe this caller's complete workflow. They are not a before/after
refactoring or general GPU-speed guarantee. All selected workers keep the same
source/artifact pins and zero swap activity. Rejected environmental attempts are
preserved; selection uses the earliest three qualified workers under a five
attempt cap, independent of measured latency.

The retained browser results, original failures and fixed qualification scope
are in [browser multimodal evidence](../build/llm/browser-multimodal-001/README.md).

## IDs without a tokenizer

`--workflows codebook` checks a fixed synthetic single-codebook contract.
Public `Generate` selects categorical IDs `[0,2,1]` in a domain of three. A
second ordinary graph uses `Embedding` as a codebook lookup and returns F32
features with shape `[1,3,4]`, exactly matching an independent NumPy table selection.
The test uses no tokenizer or text interpretation. It qualifies this small
ID-to-tensor path, while actual pretrained audio/video codebooks and multiple
codebook coordination remain unqualified.

## Evidence and resource observations

The caller records immutable source/package/library identities, numerical
errors, exact IDs, route attestations and ownership checks. It checks the
library's reported build profile before creating a runtime. Every selected node
must execute on the required backend, with operator fallback forbidden.
Native GPU execution requires the full profile and a compiled device backend;
acceptance of `--backend cuda` or another backend is not a hardware claim.

To keep independent references outside a candidate GPU process, prepare a fresh
bundle first. This verifies the upstream image processor, tokenizer, JPEG and
ONNX assets, as well as the converted graph and weight identities. Replay loads
the stored HF/ONNX reference tensors without importing either reference engine:

```sh
python3 tools/llm_baseline/prepare_multimodal_replay.py \
  --out /tmp/volvox-multimodal-reference
python3 tools/llm_baseline/verify_multimodal_replay.py \
  --bundle /tmp/volvox-multimodal-reference \
  --backend cuda --profiles full \
  --out build/llm/h-stage/replay-cuda.json
```

The replay accepts raw PCM and raw JPEG, checks the exact prepared pixels and
fixed question IDs, and keeps the original numerical budgets. Full-prefix
generation requires host-readable initial bindings. CUDA audio replay therefore
uses public `CopyTensors` to read the 2,304,000-byte OWNED encoder output into a
caller HOST snapshot before `Generate`. That byte count describes the initial
conditioning handoff; additional validation readbacks are excluded. Generation
retains its own accepted
snapshot and uploads conditioning to the required GPU. The test mutates the
caller HOST array between chunks and checks the original device OWNED snapshot
again. Direct device initial bindings are refused by this contract; this
workflow makes no device-resident invariant or zero-copy claim. GPU listing and
graph admission alone do not qualify a model.

The selected CPU receipts and guarded final-profile replays are listed in the
[evidence index](../build/llm/h-stage/evidence-index-002.json). These checks
include the reported library profiles and the fixed synthetic codebook path.
[`replay-cuda-final-003.json`](../build/llm/h-stage/replay-cuda-final-003.json)
records the actual CUDA checkpoint parity, route and ownership checks. Whisper
returns the fixed reference IDs `[291,50257]`; the image examples return their
16 and 13 reference IDs through EOS. The expanded native Vulkan/OpenGL and
WASM CPU replays are recorded in the [additional validation evidence](../build/llm/remaining-validation-001/README.md). Metal and browser WebGPU have separate
hardware and numerical gates.

Numerical capture and operational timing are separate. Accuracy receipts label
their elapsed time as the entire accuracy workflow, including model loading,
compilation and output capture. Their high-water RSS includes HF references,
NumPy and native owners; it is not engine-only memory or latency. OWNED buffer
inventories report the actual returned allocation and payload bytes within
that explicitly retained scope. Resource observations do not establish idle
performance, optimized latency, large checkpoint capacity or general device
support. FP8/FP4 source decoding remains deferred.

For a separate observation without loading HF or ONNX reference engines, use a
passing correctness receipt from the selected checkpoint and profile:

```sh
python3 tools/llm_baseline/observe_multimodal_native.py \
  --workflow audio --profile inference \
  --correctness build/llm/h-stage/actual-multimodal.json \
  --out build/llm/h-stage/observe-audio.json
```

This reports caller raw-input-to-first-result wall time, process-lifetime
VmHWM/RSS, and the actual retained OWNED output allocation inventory. The caller
includes preprocessing, package preparation, loading and compilation in that
wall interval. It releases every owner before its final memory observation.

For exhaustive tensor ownership and execution contracts, see
[proto/volvoxai.proto](../proto/volvoxai.proto) and the generated API reference.
