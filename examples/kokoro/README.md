# Complete Kokoro speech synthesis

This example runs the pinned FP32 Kokoro-82M-v1.0 model with the `af_heart`
voice and produces a 24 kHz waveform. VolvoxAI executes the complete model:
token embeddings, transformer, six bidirectional LSTMs, duration prediction,
alignment, acoustic networks, oscillator, spectral transforms and waveform
synthesis. ONNX Runtime is an optional independent oracle after synthesis.

## Synthesize a request

Run from the repository root after building the native shared library:

```sh
python3 -m pip install -r examples/tts_common/requirements.txt
python3 examples/kokoro/synth.py --backend cpu \
  --output build/tts-examples/kokoro-complete
```

The default phonemes are `həlˈoʊ wˈɜːld!` (“Hello world!”). Listen to
`build/tts-examples/kokoro-complete/volvox.wav`. The output directory also
contains a reusable full-model pipeline manifest, packages, original request
inputs and backend route evidence. All numerical stages must be selected by
the requested backend; CPU fallback is forbidden.

For a different request, use a fresh output directory:

```sh
python3 examples/kokoro/synth.py --phonemes 'həlˈoʊ!' \
  --text 'Hello!' --speed 1.0 --backend cpu \
  --output build/tts-examples/kokoro-complete-short --validate
```

`--text` is a label. Supply phonemes from the upstream
[Misaki phonemizer](https://github.com/hexgrad/misaki) when converting arbitrary
text. Unknown vocabulary characters are rejected. Tokens have leading and
trailing zero; the style vector is the `af_heart` row indexed by the unpadded
phoneme count. Speed must be a positive F32 scalar vector.

`--validate` runs the original, unmodified ONNX model only after Volvox has
produced the waveform. It writes `reference.wav` and records maximum absolute
error, RMSE, SNR and the strict elementwise comparison in the manifest. A failed
comparison returns a nonzero exit status while preserving the waveform and
report. Successful execution and numerical parity are separate results; do
not describe a waveform as matching the oracle when its recorded comparison
failed. The model's oscillator and near-zero spectral phase are sensitive to
F32 rounding differences, including between mathematically equivalent FFTs.

For a reproducible stability diagnostic, compare the original ONNX graph with
and without optimizer transformations and perturb only the oracle's encoder
features by one F32 representable value:

```sh
python3 examples/kokoro/validate.py --phase-sensitivity
```

`oracle-stability.json` retains the strict comparison and independent waveform
spectral magnitude errors (1024-sample periodic Hann window, 256-sample hop).
On the default CPU request with ONNX Runtime 1.23.2, Volvox had RMSE `0.00385`
and spectral magnitude relative L2 error `2.77%`; enabling versus disabling
the original oracle's optimizer alone produced RMSE `0.00220` and `1.72%`.
Both failed the stated elementwise tolerance. A one-ULP encoder perturbation
also produced a comparable waveform change. These are sensitivity measurements,
not relaxed parity thresholds or a claim of perceptual equivalence. The
[upstream generator](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py)
feeds raw spectral phase alongside magnitude into learned convolutions, so
phase branch changes can affect the output even at very small magnitudes.
The [source-phase follow-up](../tts_common/BENCHMARK.md#thread-dispatch-optimization-and-source-phase-conditioning-follow-up)
also reproduces large waveform changes with a one-ULP perturbation of a constant
excitation segment entirely inside ORT, without a phase branch crossing.
Replacing the unstable phase with a canonical value changes the original
model's waveform substantially; it is a diagnostic, not an accepted correction.

## Known issue: strict waveform parity

`KOKORO-STRICT-001` is an accepted known limitation for the current TTS work.
The original strict comparison remains **FAIL**; investigation is deferred so
independent ARM optimizations with verified baseline output preservation can
proceed. This classification does not establish full-waveform parity or
perceptual equivalence.

The original ORT model also changes its waveform under optimizer changes or
one-ULP input perturbations. This demonstrates numerical sensitivity, not an
ORT kernel bug or nondeterminism with identical inputs and settings. It does
not account for every remaining native difference. The retained CPU short
baseline has RMSE `0.00599463611` and `14,047/33,600` strict mismatches; the
separate A56 same-device ORT comparison has RMSE `0.0043437764` and
`13,866/33,600` mismatches. See the [measured investigation](../tts_common/BENCHMARK.md#candidate-revalidation-arm-policy-and-kokoro-strict-waveform-status).

Keep the original model, reference, tolerances, failure results and exit codes.
For the same fixture, a changed native baseline output or increased error needs
a new regression review; new local operator failures are not covered by this
acceptance. Model, provider, device or input-range changes require fresh
qualification. Existing diagnostic timing collection remains available through
[`--report-waveform-differences`](../tts_common/README.md#interpret-numerical-results).

## Duration and execution

The first Volvox graph computes text features, predictor features and
cumulative token durations. The application reads only the final duration to
choose the second graph's extent. The second Volvox graph consumes the actual
first-stage outputs, computes alignment and synthesizes audio. No learned
activation or predicted alignment becomes a constant. The host does not run a
numerical model stage.

The default `--max-frames 2048` admits up to 51.2 seconds at 600 waveform samples
per duration frame. The bounded I64-to-I32 adapter clips each positive duration
to `max_frames + 1` before casting, proves the entire cumulative sum fits I32,
and rejects totals above the limit. Every admitted request retains its source
durations; clipping cannot make an oversized request admissible.

The adapter specializes only tensor geometry and constant shape programs:
reflection becomes Gather; resize becomes Gather and the source interpolation
arithmetic; depthwise transposed convolution becomes zero insertion and grouped
convolution; the 20-point STFT becomes a runtime Bluestein FFT graph. Constant
window normalization replaces the inverse-transform index/scatter program.
The original centered variance, square root and learned style transforms stay
as their original operations, preserving the source's epsilon placement.
`provenance.lowering` records these transformations.

STFT geometry uses scalar C99 `sinf`/`cosf` with F32 arguments and results,
matching the source transform's precision. Python double math and NumPy's
vector transcendental evaluator can produce different last bits. Preparation
requires those C99 functions in the host math library and reports an explicit
error if they are unavailable. Host libraries can still differ across platforms;
this does not promise bit identity with every ORT build or device.

Regenerate a prepared decoder in a fresh output directory after updating this
geometry code. Upgrading the runtime alone cannot change constants already
stored in an existing model package. The corrected lowering matches the original
ORT waveform in the measured same-input CPU case, while the complete native
pipeline still fails strict waveform parity. The [detailed recheck](../tts_common/BENCHMARK.md#kokoro-waveform-threading-and-size-recheck)
separates this conversion fix from runtime sensitivity and the remaining failure.

## Quantization and QAT (W8A8 vs Grouped W8/F32)

Kokoro's decoder contains 242.7 MB of weights across 71 Linear layers (22.8 MB, 9.4%),
88 Conv2D layers (205.1 MB, 84.5%) and embeddings/normalization (14.8 MB, 6.1%).
In FP32 WASM, Conv2D accounts for ~79.5% of the total runtime (~6.45 seconds).

### Periodic activation sensitivity (Snake vs LeakyReLU)

Kokoro's generative vocoder uses periodic Snake activations:
$$\text{Snake}(x) = x + \frac{1}{\alpha}\sin^2(\alpha x) \quad (\alpha \approx 10 \sim 50)$$
The first derivative $f'(x) = 1 + \sin(2\alpha x)$ oscillates rapidly. Under naive
post-training quantization (PTQ), activation discretization steps ($\Delta x \sim 1/128$)
are amplified across 25 residual convolutional blocks into severe high-frequency
hissing noise:
- 0–1 kHz (speech baseband): 0.936x energy (speech retained)
- 4–8 kHz (high-frequency band): 3.320x energy surge
- 8–12 kHz (ultra-high-frequency band): 3.826x energy surge
- Silence floor: average amplitude rises 8x (from 0.00049 to 0.00379)

Leaving Conv2D in FP32 ("selective quantization") avoids the noise but keeps 84.5%
of weights and 79.5% of execution in FP32, achieving only a 7% weight reduction.
Full W8A8 speed (1.2s WASM) and compact footprint (72 MB) require quantizing
Conv2D while controlling activation noise.

### VolvoxAI Audio QAT pipeline

To eliminate periodic activation noise while quantizing all Conv2D layers:

1. **A8 FakeQuantize with STE (`native/src/training/training_qat.inc`)**:
   During training forward pass, dynamic symmetric INT8 discretization is simulated:
   $$\hat{x} = \text{clamp}\left(\left\lfloor \frac{x}{S} \right\rceil, -128, 127\right) \cdot S$$
   During backward pass, the Straight-Through Estimator (STE) passes gradients directly
   to Conv2D weights ($\partial \hat{x} / \partial x = 1$). Weight gradients adjust
   convolution kernels to anticipate and cancel activation discretization harmonics.
   Supported on both CPU/WASM and native CUDA backends (`test_cuda_qat_training.py`).
2. **Multi-Resolution STFT Loss (`tools/audio_spectral_loss.py`)**:
   Combines spectral convergence $\mathcal{L}_{\text{sc}}$ and log-magnitude loss
   $\mathcal{L}_{\text{mag}}$ across 3+ FFT window sizes (1024, 2048, 512). This loss
   directly penalizes the high-frequency hissing induced by Snake discretization.
3. **PyTorch QDQ Import Bridge (`tools/exporter/`)**:
   Translates fine-tuned PyTorch `torch.ao.quantization` QuantizeLinear / DequantizeLinear
   graphs into native `QConv2D` (`OHWI` / `NHWC`) packages with 0–1 LSB parity.
4. **Physical CUDA QAT Verification**:
   Verified on remote NVIDIA RTX 3090: vocoder QAT fine-tuning
   reduced high-frequency noise ratio from $1.000$ to $0.997$x within 50 steps (1.17s,
   60.5 MB peak VRAM) while converging at SNR 28.95 dB.
5. **Grouped W8/F32 Production Baseline**:
   For deployments prior to QAT fine-tuning, the Grouped W8/F32 profile quantizes all
   71 Linear layers to packed INT8 while retaining Conv2D layers in FP32. This yields
   a completely noise-free production package (correlation 0.9997, 1.036x HF ratio,
   silence floor 0.00052) with reduced memory requirements.

## Reuse and compare

The top-level manifest contains only the original token, style and speed
inputs. Its two stage manifests describe the encoder and decoder packages.
The shared runner executes both stages for every measured request and verifies
that the returned duration agrees with the prepared decoder shape:

```sh
python3 examples/tts_common/benchmark_native.py \
  --manifest build/tts-examples/kokoro-complete/manifest.json \
  --out build/tts-examples/kokoro-complete/native-cpu.json \
  --backend cpu --threads 4 --warmups 2 --repeats 5
```

Use `--validate` during preparation to provide an independent ONNX reference
for subsequent benchmark comparisons. Without validation, the saved waveform
is a Volvox output and the manifest explicitly makes no oracle parity claim.
Each case declares `validation.reference_engine`; a Volvox reference must not
be labelled or scored as an ONNX Runtime parity result.
The [browser example](../kokoro.html) accepts the same full pipeline manifest.
A prepared browser request must produce its recorded duration; prepare a new
package when the duration changes. Native `synth.py` handles that specialization
for fresh requests. See the [shared workflow](../tts_common/README.md) for build,
backend and timing details.

The default request needs more than a 2 GiB Vulkan graph arena. Its decoder
describes about 3263 MiB of distinct output buffers and 244 MiB of weights,
plus the backend's 128 MiB base and 1 MiB scratch reservation. Use
`VOLVOX_VULKAN_MB=4096` for this prepared request and record the setting with
the report. This is a resource requirement, not an unsupported-operator result;
larger durations require a fresh memory assessment.

`prepare.py`, `synth.py` and `complete.py` enter the same complete Volvox
workflow. The source helper module and `prepare_stage2.py` retain isolated
component diagnostics for operator development; their captured component
inputs and timings are not complete-model synthesis results.

## Source and license

Assets come from
[onnx-community/Kokoro-82M-v1.0-ONNX at revision 1939ad2](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/1939ad2a8e416c0acfeecc08a694d14ef25f2231),
using `onnx/model.onnx`, `voices/af_heart.bin` and its tokenizer/config files.
The model card declares Apache-2.0; the original checkpoint is
[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
`fetch.py` checks the pinned SHA256 digests. Downloads and generated packages
stay in ignored `build/tts-examples/`; no model or voice data is committed.
