# Complete speech examples and ONNX comparisons

[Piper](../piper/README.md) and [Kokoro](../kokoro/README.md) run their complete
checkpoint computations in VolvoxAI. Each request executes an encoder/duration
stage and an acoustic/waveform stage. The host reads the predicted frame count
to select a shape and passes the actual intermediate tensors to the next stage.
ONNX Runtime is used only as an independent comparison oracle.

Execution and waveform accuracy have separate results. Piper declares explicit
PCM16 error bounds. Kokoro currently produces a waveform but fails the strict
ONNX comparison; its diagnostic mode retains that failure alongside timings.
The [benchmark record](BENCHMARK.md) identifies measured configurations and
remaining qualification work. Each report applies to its recorded artifact,
request and device.

The [runtime optimization follow-up](BENCHMARK.md#runtime-optimization-after-the-ort-comparison)
compares the preserved and revised kernels on the same full requests, including
FP32, static W8A8 and grouped W8/F32 variants. It separates faster execution
from waveform qualification: quantization does not automatically improve either
latency or accuracy, and the default prepared packages remain FP32.

## Build, prepare and listen

From the repository root, install the shared dependencies and Piper's phonemizer,
then build the native and browser artifacts:

```sh
python3 -m pip install -r examples/piper/requirements.txt
make build_native_libraries
make build_web
make build_wasm
```

Prepare and execute a real request:

```sh
python3 examples/piper/prepare.py --text 'Hello world.'
python3 examples/piper/synthesize.py --backend cpu --verify
python3 examples/kokoro/synth.py --backend cpu --validate \
  --output build/tts-examples/kokoro-complete
```

Piper writes `build/tts-examples/piper-full/speech.wav`. Kokoro writes
`build/tts-examples/kokoro-complete/volvox.wav`; its current strict validation
returns nonzero while retaining the waveform, ONNX reference and comparison.
Use `--validate` for Kokoro benchmark preparation: the shared runner requires
an ONNX reference and rejects a saved Volvox output as an oracle parity result.
The model READMEs describe text/phoneme inputs, voice selection and licenses.

Native, web and quantized comparisons require explicit
`validation.reference_engine: "onnxruntime"` on the reference request.
Quantized comparisons also require `validation.deterministic: true` on both
the FP32 and candidate requests. Current Piper and validated Kokoro preparation
record the reference origin after running ORT, and quantization preserves it.
Legacy manifests without the reference field are rejected.
Prepare a new output directory with the current script
and recreate the quantized variant from that FP32 manifest, reusing the verified
assets. Do not relabel an old reference without independently verifying its
source graph, exact inputs and saved output against ORT.

The full manifests are `build/tts-examples/piper-full/manifest.json` and
`build/tts-examples/kokoro-complete/manifest.json`. Downloads and generated files
remain under ignored `build/tts-examples/`. Verified downloads are reused, and
preparation refuses to overwrite an existing request directory.

Raw benchmark JSON, profiling output, logs and qualification receipts also stay
under ignored `build/tts-examples/`. The retained benchmark reports are in
`build/tts-examples/reports/`; new remote runs have their own directories under
`build/tts-examples/remote/`. Keep example code, workflows and summarized
benchmark tables in Git, with run artifacts outside the example source tree.

Prepared packages specialize token count N and acoustic frame count T. Every
measured or browser request still executes both stages and checks the returned
duration against T. Prepare a new output directory when text, noise or speed
changes the required shape. The native preparation/synthesis commands perform
this specialization from a fresh Volvox duration result. No learned activation
or predicted alignment is stored as a replacement for model computation.

## Measure complete inference

The native runner retains its library, runtime, compiled models and execution
contexts. The ONNX runner retains its full-model session with `ORT_ENABLE_ALL`.
Both receive the original request tensors and finish with host audio. Pipeline
timing includes both Volvox stages, intermediate transfer/binding, duration
admission, synchronization and output transfer. Loading, phonemization, file I/O,
correctness checks and WAV encoding are outside the measured interval. Reports
retain setup, first-call and warmup observations separately.

```sh
python3 examples/tts_common/benchmark_native.py \
  --manifest build/tts-examples/piper-full/manifest.json --engine both \
  --backend cpu --threads 1 --warmups 3 --repeats 10 \
  --out build/tts-examples/piper-full/native-cpu.json
node examples/tts_common/benchmark_web.mjs \
  --manifest build/tts-examples/piper-full/manifest.json \
  --backend wasm --profile inference --warmup 3 --repeats 10 \
  --out build/tts-examples/piper-full/wasm.json
deno run --no-config --unstable-webgpu -A examples/tts_common/benchmark_web.mjs \
  --manifest build/tts-examples/piper-full/manifest.json \
  --backend webgpu --profile full --warmup 3 --repeats 10 \
  --out build/tts-examples/piper-full/webgpu.json
```

Use Node.js 20 or newer for the Node runner, or Deno for either browser backend.
Select native `cuda`, `vulkan`, `opengl` or `metal` with the full library to require
that backend. Compilation and execution of every stage must attest zero
fallback. Native CPU and WASM are available in both inference and full profiles;
WebGPU requires the full profile. The callers use the generated inference API
from [the public schema](../../proto/volvoxai.proto).

`--ort-provider CUDAExecutionProvider` independently requests ONNX CUDA and
requires every profiled node to use CUDA by default. Add
`--ort-allow-cpu-nodes` to measure ONNX Runtime's CUDA plus CPU placement, which
can retain shape operations on CPU. The report lists each provider's node and
operation counts; CUDA must still execute at least one node. Runtime provider
fallback stays disabled. An unavailable provider is recorded explicitly.
Reports include provider placement,
threads, versions, model/runtime hashes, median, p90 and real-time factor
(latency divided by waveform duration). Compare results with matching workload,
thread settings and hardware; full pipeline timing includes the stage handoff.
Component manifests remain useful for diagnosis, with their narrower scope
recorded explicitly.

### Profile a complete request

Capture profiling separately from the latency benchmark. Both profilers warm
the retained contexts, execute both stages with live inputs, save the waveform,
and collect the existing public profiling service's plans, memory observations
and events:

```sh
python3 examples/tts_common/profile_native.py \
  --manifest build/tts-examples/kokoro-complete/manifest.json \
  --backend cuda --threads 4 --out build/tts-examples/profiles/kokoro-cuda.json
deno run --no-config --unstable-webgpu -A examples/tts_common/profile_web.mjs \
  --manifest build/tts-examples/piper-full/manifest.json \
  --backend webgpu --out build/tts-examples/profiles/piper-webgpu.json
```

The web profiler also accepts `--backend wasm`. Its ordinary request counts
buffer/pipeline/bind-group creation, uploads, submissions and mapped readbacks.
It then records separate detailed and basic traces. Detailed WebGPU timing
splits passes and has a bounded timestamp pool; check coverage and dropped
observations before ranking kernels. Basic timing measures complete GPU passes.
CUDA tracing can disable replay, and a host upload span may include a wait for
earlier kernels. Use device program intervals to attribute GPU computation;
host, device, pass and node intervals overlap and must not be added together.

Compare the saved **untraced** waveforms across builds to distinguish a new
regression from Kokoro's existing ONNX difference. Profiling measurements are
diagnostics; use the separate two-warmup/five-request benchmark for speedups.

### Compare ONNX Runtime profiles and quantized models

`profile_ort.py` executes the original complete ONNX model. It uses separate
sessions for ordinary latency and profiling, retaining the raw trace and the
optimized graph's structure. Both sessions use the same provider configuration,
inputs, graph optimization and thread count. CUDA permits and reports CPU node
placement, disables TF32 and disables runtime provider fallback.

```sh
python3 examples/tts_common/profile_ort.py \
  --manifest build/tts-examples/kokoro-complete/manifest.json \
  --provider cpu --threads 4 --warmups 2 --repeats 5 \
  --out build/tts-examples/profiles/kokoro-ort-cpu.json
```

Use `--provider cuda` with an installed CUDA-capable ORT environment. The report
separates first execution, warmups and measured requests. ORT's
`Node/*_kernel_time` events are host operator intervals, including for CUDA;
they must not be compared as device durations against Volvox GPU timestamps.
`--external-capture` marks measured CUDA requests for a separate Nsight Systems
capture. `summarize_nsys.py --sqlite TRACE.sqlite --out SUMMARY.json` summarizes
those request ranges, excluding warmups and keeping kernel, transfer and host
API intervals separate. `compare_profiles.py --help` describes offline graph
and source-region comparisons.

Quantization changes numerical behavior and may change predicted duration.
Keep the original FP32 package and reference. For a published quantized ONNX
model, pass its path using `profile_ort.py --onnx MODEL.onnx --variant NAME` and
use `--compare-outputs` with the corresponding FP32 profiler's `.outputs.npz`.
The model override must preserve semantic input/output names. Changed output
lengths are recorded without truncating or aligning waveforms.

For an experimental Volvox INT8 package, the public quantization service can
prepare supported Conv/Linear operators while retaining other operations:

```sh
python3 examples/tts_common/quantize_tts.py \
  --manifest build/tts-examples/piper-full/manifest.json \
  --engine volvox --scheme static-int8 --stages acoustic \
  --out-dir build/tts-examples/piper-acoustic-int8
python3 examples/tts_common/benchmark_quantized.py \
  --manifest build/tts-examples/piper-acoustic-int8/manifest.json \
  --fp32-manifest build/tts-examples/piper-full/manifest.json \
  --engine volvox --backend cpu --threads 4 --warmups 2 --repeats 5 \
  --out build/tts-examples/profiles/piper-acoustic-int8.json
```

Both stages still execute on every request; this example quantizes the acoustic
stage while keeping duration computation FP32. The authoring receipt lists
quantized operators, retained float operators, storage bytes and calibration
coverage. Grouped weight-only INT8 (`--scheme packed-int8`) and ORT dynamic or
static INT8 are separate schemes, with different coverage and arithmetic.
The helper's static calibration uses the prepared evaluation requests only;
these experiments do not establish speech quality on unseen text or voices.
Piper's PCM16 criterion remains unchanged, and a quantized candidate can fail
it. A changed Volvox duration outside its prepared extent is a refusal, not a
successful timing result. See the [benchmark record](BENCHMARK.md) for measured
coverage, latency and numerical outcomes.

## Interpret numerical results

The default elementwise tolerance is `rtol=1e-4, atol=1e-5`. Shape, dtype,
finiteness and backend route checks are mandatory. Piper preparation also
checks each stage independently at that tolerance and requires exact durations
and masks. Its full waveform declares a peak error limit of `1/32767` and an
RMS limit of `0.1/32767`, corresponding to one and one tenth of a PCM16 LSB.
Reports retain the stricter elementwise pass/fail and mismatch count alongside
this declared criterion. The default voice noise scales remain enabled; explicit
Gaussian request inputs make the comparison reproducible.

Kokoro retains the strict criterion. Its oscillator and spectral phase are
sensitive to F32 rounding. In the recorded default CPU diagnostic with ONNX
Runtime 1.23.2, Volvox versus the original model had waveform RMSE `0.00385`;
enabling versus disabling the original ONNX optimizer alone gave RMSE `0.00220`.
Both failed the strict comparison. The [Kokoro diagnostic](../kokoro/README.md)
records spectral magnitude errors and one-ULP perturbation experiments. These
observations establish sensitivity, not numerical parity or perceptual quality.

The existing discrepancy is tracked as accepted known issue
[`KOKORO-STRICT-001`](../kokoro/README.md#known-issue-strict-waveform-parity).
It is separated from independent ARM optimization work that preserves baseline
outputs. This work-priority decision leaves strict failures and exit codes
unchanged; it does not waive new output differences or regressions.

To collect execution timings while preserving a waveform failure:

```sh
python3 examples/tts_common/benchmark_native.py \
  --manifest build/tts-examples/kokoro-complete/manifest.json --engine both \
  --backend cpu --threads 1 --warmups 3 --repeats 10 \
  --report-waveform-differences \
  --out build/tts-examples/kokoro-complete/native-cpu-diagnostic.json
node examples/tts_common/benchmark_web.mjs \
  --manifest build/tts-examples/kokoro-complete/manifest.json \
  --backend wasm --profile inference --report-waveform-differences \
  --out build/tts-examples/kokoro-complete/wasm-diagnostic.json
```

`--report-waveform-differences` keeps the default criterion, records numerical
failure fields and returns exit code 3 when the waveform differs. It still
requires valid shapes, dtypes, finite outputs and complete backend execution.
These timings describe a numerically failing run. Without that option, a failed
comparison stops the run. Exit codes are 0 for passed requested engines, 1 for
failure, 2 for unsupported/unavailable execution and 3 for retained numerical
differences.

## Run in a browser

Serve the repository root after preparing the full manifests and building:

```sh
python3 -m http.server 8000
```

Open [Piper](../piper.html) or [Kokoro](../kokoro.html). Their default manifests
select `piper-full` and `kokoro-complete`. WASM CPU or WebGPU executes both
Volvox stages from original request inputs. The pages compare against saved
ONNX reference waveforms and expose the numerical result; they do not run
ONNX Runtime Web. A changed N or T requires preparing a new package. The saved
reference can be played separately from the generated waveform.

## Remote GPU measurements

On an authorized SSH target with an existing Python environment containing
NumPy, protobuf and ONNX Runtime, and Deno with physical WebGPU, run:

```sh
python3 examples/tts_common/run_remote.py --target USER@HOST --mode full \
  --python /path/to/venv/bin/python --deno /path/to/deno
```

Prepare both full manifests first. Full mode transfers the required reference
model, runtime packages and request fixtures into a unique task-owned directory.
It verifies transfer hashes and runs each backend sequentially. Kokoro uses the
explicit waveform-difference reporting mode; a numerical failure remains a
failed qualification in the matrix. No packages are installed remotely and
shared model caches are unchanged.

CUDA profiling is optional. On the remote host, put `nsys` on `PATH` or set
`NSYS` in the runner's environment to its executable path. The runner also
checks `/opt/nvidia/nsight-systems/bin/nsys` before running without profiling.

The controller checks disk space before transfer, retrieves verified reports,
cleans up its task-owned files and reports recovered space. Reports stay locally
under `build/tts-examples/remote/`. If retrieval fails, it removes large task
artifacts and retains compact remote reports for recovery. Retry with
`--target USER@HOST --recover build/tts-examples/remote/RUN_DIRECTORY`; verified
local copies are required before the remaining remote reports are removed.

Metal implementations require separate numerical qualification on Apple
hardware. See [the operator guide](../../docs/tts-operators.md) for the supported
export subset and primitive validation workflows.
