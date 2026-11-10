# Piper speech with VolvoxAI

This example runs the complete published `en_US-lessac-low` Piper voice in
VolvoxAI: phoneme embeddings, text encoder, stochastic duration predictor,
alignment, reverse flow and neural vocoder. ONNX Runtime supplies the independent
reference waveform. Phonemization, random input generation, shape selection and
PCM file writing belong to the application.

Piper determines audio length from its duration predictions. The example uses
two VolvoxAI stages. The first computes those predictions and encoder tensors;
the host reads the resulting frame count to select the second stage's shape.
The second computes alignment and the entire audio waveform. Encoder tensors,
durations and Gaussian noise remain request inputs, and neither activations nor
recorded audio are embedded in model weights.

## Prepare and synthesize

Build the native library as described in the [Python guide](../../python/README.md).
From the repository root, install the example dependencies into an environment
or the ignored local dependency directory:

```sh
python -m pip install -r examples/piper/requirements.txt \
  --target build/tts-examples/python
PYTHONPATH=build/tts-examples/python python examples/piper/prepare.py \
  --text 'Hello world.' --seed 42 --out build/tts-examples/piper-full
python examples/piper/synthesize.py \
  --manifest build/tts-examples/piper-full/manifest.json \
  --backend cpu --verify --out build/tts-examples/piper-full/speech.wav
```

Preparation downloads the official 63.2 MB voice once, verifies pinned SHA-256
hashes and reuses matching files in `build/tts-examples/assets/piper-lessac-low`.
It preserves Piper's eSpeak phonemes and BOS/padding/EOS token mapping. It then
exports and executes both stages in VolvoxAI CPU, derives the second shape from
the first stage's output, and compares final audio with the whole ONNX source.
A preparation failure remains an error; it does not select another inference
engine. Use `--library` to select a specific native library.

The prepared request retains the voice's ordinary scales `[0.667, 1, 0.8]`.
Both independent standard Gaussian draws are explicit inputs generated from
`--seed`, so the comparison uses identical noise while retaining stochastic
speech synthesis. Change the seed when preparing a different rendition.

Outputs include `manifest.json`, an explicit-noise `reference.onnx`, prepared
request/reference tensors, `duration/` and `acoustic/` stage packages, exporter
reports and `preparation-report.json`. The synthesis command runs both stages
again and records their execution, input transfer and total request time beside
the WAV file. `--verify` enables the separate ONNX reference comparison.

Each prepared package admits its phoneme count and predicted frame count.
Prepare a new output directory for a different sentence or noise seed. During
execution, the duration stage still runs and its frame count is checked before
the acoustic stage; a different count is rejected. The default preparation
limits are 256 phoneme IDs and 4,096 acoustic frames, adjustable with explicit
CLI options. These limits control resource use; they do not truncate speech.

`--backend cuda`, `vulkan`, `opengl` or `metal` requires a suitable full native
library and device. Each stage requires the selected backend and forbids
operator fallback. Availability and numerical validation must be established by
the generated run reports for the particular build and device.

## Compare complete inference

```sh
python examples/tts_common/benchmark_native.py \
  --manifest build/tts-examples/piper-full/manifest.json --engine both \
  --backend cpu --threads 1 --warmups 3 --repeats 10 \
  --out build/tts-examples/piper-full/cpu-comparison.json
```

Compare the complete two-stage VolvoxAI request with the complete ONNX reference.
Stage-only reports help identify cost but do not establish an end-to-end speedup.
Inference timing includes completed host outputs and stage handoff; setup,
phonemization and file I/O are reported separately or excluded. For GPU runs,
select the desired native backend and `--ort-provider CUDAExecutionProvider`
where available. Add `--ort-allow-cpu-nodes` to measure ONNX's usual mixed CUDA
and CPU placement; the report records both providers' executed operations.
The [shared benchmark workflow](../tts_common/README.md)
describes browser WASM/WebGPU execution and provider availability reporting.
The [browser workbench](../piper.html) consumes the same full pipeline manifest.

## Numerical comparison protocol

Each individual stage must meet `rtol=1e-4, atol=1e-5` against ONNX Runtime on
identical inputs. Duration counts and masks must agree exactly. For the complete
waveform, small encoder differences can grow through the reverse flow. Its
explicit manifest criterion allows at most one PCM16 least-significant bit of
peak error (`1/32767`) and one tenth of a bit in RMS error (`0.1/32767`). Shape and
finiteness checks still apply. This criterion is specific to this Piper waveform;
it does not alter operator tests or other models' tolerances.

Reports retain the stricter elementwise comparison separately, including its
pass/fail result and mismatch count. The initial CPU run had two strict
mismatches among 14,592 samples, peak error `1.504e-5`, RMS error `1.212e-6` and
99.67 dB signal-to-error ratio. Those errors are 0.493 and 0.040 PCM16 bits,
respectively. Both individual stages passed the strict check. Benchmark reports
record the actual results for each subsequent backend/build; a backend exceeding
the waveform bounds fails validation.

## W8A8 Quantization and contractive activations

Piper uses a VITS-based neural vocoder with LeakyReLU activations throughout its
convolutional residual flow. Because LeakyReLU has a contractive derivative ($|f'(x)| \le 1$),
activation discretization noise naturally attenuates across layers without explosive
harmonic resonance. Under standard post-training quantization (PTQ) or QAT, Piper W8A8
achieves an SNR of 18.6 dB and preserves clean speech without the high-frequency hissing
observed in models with oscillatory activations (such as Snake in Kokoro).

## Model lowering and source

The pinned source has 2,755 nodes. Its three duration splines implement
independent elementwise calculations with dynamic `NonZero`, `GatherND` and
`ScatterND`. The adapter replaces each compact/scatter implementation with the
same spline calculations over dense lanes and an identity-tail selection.
Checkpoint parameters remain unchanged. Tests compare each rewritten region
with the original ONNX region, including the exact tail boundaries.

Shape/index computations are folded from explicit shape metadata. Scalar scale
lookups become equivalent one-element broadcast tensors. The two random operators
become explicit Gaussian inputs. These changes do not precompute any learned
activation from the prepared text. Export reports describe the default safe
optimizer; backend compilation may additionally fuse supported patterns. This
workflow does not generate an arbitrary fused kernel for the whole model.

The voice is pinned to the official
[`rhasspy/piper-voices` revision `375a0fe…`](https://huggingface.co/rhasspy/piper-voices/tree/375a0fe641dea077c2a47b4e9a056d6da521eed3/en/en_US/lessac/low).
Its ONNX SHA-256 is
`f7d01dde371555732c4c314111ac79672b1a5ce2fc19266ab42178fd8df7f375`.
The [upstream model card](https://huggingface.co/rhasspy/piper-voices/blob/375a0fe641dea077c2a47b4e9a056d6da521eed3/en/en_US/lessac/low/MODEL_CARD)
identifies the Lessac Blizzard 2013 dataset and links its
[dataset license](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html).
That agreement permits research purposes and excludes commercial use; the model
card does not supply a separate permissive weights license. Treat this voice as
a research integration and benchmark fixture. The pretrained assets and
phonemizer retain their upstream terms; no such binaries are committed here.
