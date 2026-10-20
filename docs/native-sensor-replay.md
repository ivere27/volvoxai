# Native sensor replay

Replay timestamped sensor features through the native engine to check input
snapshots, scheduling, state feedback and stable output ownership. The supplied
small F32 fixture has an independent NumPy reference. Camera/IMU preparation,
clock conversion, freshness selection and actuator policy belong to the caller.

The replay qualifies prepared tensors on the chosen native artifact. It does
not execute a physical robot or the selected audio model. The fixed workload and
image/audio references are in
[multimodal_sensor.json](../tools/llm_baseline/multimodal_sensor.json).

## Run the prepared-tensor replay

Build the matching native libraries following the
[native runtime guide](native-runtime.md#build), and use the repository's
[Python test dependencies](../python/requirements-test.txt). Run from the repository root.

The workload uses native prepared-tensor replay. It uses synthetic source
timestamps at 50 Hz, 12 samples and fixed seed 1701. Inputs are F32 image
features `[1,4]`, IMU `[1,6]`, proprioception `[1,3]`, relative observation time
`[1,1]`, and I32 sample index `[1]`. This is a tiny deterministic numerical
fixture, not a learned robot controller. Linear projections and additions have
an independent NumPy reference. Source nanoseconds are converted by the caller
to relative F32 seconds; the engine does not interpret their meaning.

```sh
PYTHONPATH=python python3 tools/native_sensor_replay.py --output build/llm/sensor-replay/cpu-full
VOLVOXAI_LIBRARY="$PWD/native/libvolvoxai-lite.so" PYTHONPATH=python python3 tools/native_sensor_replay.py --output build/llm/sensor-replay/cpu-inference
PYTHONPATH=python python3 -m unittest discover -s python/tests -p test_native_sensor.py -v
```

## State and tensor ownership

The runner loads/compiles two F32 packages, then validates these workflows:

- Stateful ALL executes explicit initialization followed by eleven transitions
  on one context. Output feedback updates a three-value accumulator. Mutating
  copied results cannot change internal feedback. The first native output buffer
  remains valid after later transitions and after context release.
- Stateless LATEST uses independent contexts sharing one compiled model. Four
  bursts choose samples 2/5/8/11. Caller clock/age policy rejects stale/future
  observations and resolves equal timestamps by sample index. Each accepted
  input survives immediate producer-buffer reuse after Submit returns.
- Stateful LATEST and stateless LATEST with input reuse are refused before
  mutation. The same contexts remain available for correctly admitted calls.
- Every result, request, context, compiled model and runtime is released. Output
  selection and exact CopyTensors destination receipts are checked.

LATEST supersedes eligible work that overlaps. A tiny CPU request can complete
before the next submission, so it remains a valid successful result. The runner
validates either actual supersession without publication or a completed result
against its own snapshot. It records both counts. Current native CPU replay
does not demonstrate queued supersession or engine batching; adding a delay
does not establish those capabilities.

Configured backlog is four requests, scheduled input budget 256 KiB and result
budget 64 KiB. Numerical tolerance is `atol=rtol=1e-5`; timestamp/index echo and
ownership checks require exact equality. Each `result.json` records actual
native library/profile/hash, OS, CPU/ISA flags, ABI, RAM, fixture hashes and
observed scheduling results. The replay is offline and runs as quickly as
possible; 50-Hz timestamps do not prove 50-Hz wall-clock throughput.

## Qualification scope

The fast replay produces correctness captures. Use the paced companion below
for CPU latency, deadline and caller-backlog observations. Whole engine/device
peak memory and physical sensor measurements remain unqualified. CUDA requires
the selected CUDA library and physical device; use
`--backend cuda` on the designated native GPU host. CPU success does not imply
CUDA, ARM64, DMA/camera interop, physical robot or hard realtime qualification.

## Optional paced CPU replay

Use the separate `tools/native_sensor_paced.py` companion to observe a sustained
prepared-input workload. The original `native_sensor_replay.py` command remains
the fast correctness capture. The paced command requires an accepted artifact
manifest that names the exact CPU library path and SHA; a different library is
refused. Run from the source checkout containing the companion and its policy,
with the matching Python SDK on `PYTHONPATH`.

```sh
PYTHONPATH=python python3 tools/native_sensor_paced.py \
  --library "$PWD/native/libvolvoxai.so.0.7.0" \
  --artifacts "$PWD/build/llm/d-stage/official-artifacts.json" \
  --output "$PWD/build/llm/sensor-paced/cpu-full"
PYTHONPATH=python python3 tools/native_sensor_paced.py \
  --library "$PWD/native/libvolvoxai-lite.so.0.7.0" \
  --artifacts "$PWD/build/llm/d-stage/official-artifacts.json" \
  --output "$PWD/build/llm/sensor-paced/cpu-inference"
```

The output directory must be new. Before loading or observing results, the
runner saves the policy, source and artifact hashes in `prospective.json`.
The paced lifecycle tests use the same explicit artifact selection. Run them
for each library; ordinary test discovery skips these two native observation
tests when `VOLVOXAI_SENSOR_ARTIFACTS` is absent.

```sh
PYTHONPATH=python VOLVOXAI_LIBRARY="$PWD/native/libvolvoxai.so.0.7.0" \
  VOLVOXAI_SENSOR_ARTIFACTS="$PWD/build/llm/d-stage/official-artifacts.json" \
  python3 -m unittest discover -s python/tests -p test_native_sensor.py -v
```

The separate [paced policy](../tools/llm_baseline/native_sensor_paced.json)
fixes 50 Hz, three seconds, 150 planned observations, three separate warmup
steps and two CPU threads. The default workload uses one fresh context for
ALL initialization and subsequent tensor feedback. Dropped observations never
enter the engine; accepted positions follow their own contiguous step order.
The stateless LATEST option uses separate contexts with complete fresh inputs
and coalesces eligible observations in the caller queue; it is a separate mode.

A producer releases fixed prepared observations at absolute monotonic times.
The caller bounds queued plus processing observations to four and drops the
newest arrival on overflow. It rejects future source times and observations
older than 60 ms before Submit. A 20-ms target is measured from each planned
source release; Submit receives the mapped native monotonic target. This is
an ALL/LATEST soft scheduling target. It does not impose an Execute hard
deadline or guarantee a control-loop deadline.

Each result records planned source time, actual caller enqueue, Submit,
acceptance, first observed terminal status and output-consumption time.
Enqueue-to-completion p50/p95/p99 include caller queuing and polling transport;
they exclude CopyTensors and deferred NumPy assertions. A per-request
observation bracket and poll count expose completion precision. Source-to-
completion latency, producer timer overshoot and engine/caller deadline misses
are separate observations. Backlog samples and accepted/completed/cancelled/
caller-drop accounting must balance. External cancellation stops the producer,
cancels accepted pending work, drops queued observations and releases all
owners; a cancelled or failed stateful context is never resumed.

Every output requires a valid OWNED BufferView. Copies and NumPy assertions
run outside the measured completion interval. The first snapshot stays valid
after context release. Prepared input and completed output bytes are saved
separately, allowing numerical checks without replaying timing. Request,
context and output-handle creation/release counts must balance, and the final
runtime inventory must show zero unconsumed result bytes.

Output copying, ownership receipts and periodic inventory queries still run
between requests. They affect this caller's consumption time, later queuing,
backlog and whole-process CPU/RSS observations. The measured workload includes
that integration cost; its enqueue-to-terminal percentiles are not a pure
engine benchmark. Actual elapsed time, observed completions per second and
drain time beyond the planned window are reported separately from the planned
150 observations over three seconds.

RSS samples come from `/proc/self/status`, with actual sample intervals and
setup/pacing/teardown phases recorded. VmHWM is the OS high-water mark over
the entire process lifetime, including imports, loading and compilation.
Known engine inventory capacities are reported separately and never added to
RSS. Sampling does not establish an exact interval peak or a memory
nonregression result. These runs describe the selected device and workload;
they are not a comparative performance pass, hard realtime, physical robot,
ARM64 or CUDA qualification.

## Image and audio reference selection

The image reference reuses
[TinyReceiptVQA](https://huggingface.co/ivere27/tiny-receipt-vqa-structured-qa-21m/tree/f536c6ed181ca88929214f0bbc08f126f148c7ae)
revision `f536c6ed181ca88929214f0bbc08f126f148c7ae`. Existing local config,
manifest, tokenizer, inference/preprocessing source, receipt samples and both
F32 ONNX model hashes match that pinned upstream revision. No model weights
were downloaded again. The producer's processor converts an image to grayscale,
resizes to 672×320 with Pillow bilinear interpolation and maps pixels to [-1,1]
in `[1,1,320,672]` F32 layout. BPE1536 applies NFC normalization. Existing
qualification is linked from the policy; it is historical evidence, not a new
H-stage execution or a physical camera test.

The audio reference selects
[Whisper-tiny](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af)
revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`. Config, generation,
feature-extractor and tokenizer assets are hashed in the policy and retained
under `build/llm/multimodal-metadata/whisper-tiny/` (about 4.1 MiB). Model weights
were not downloaded or executed. Transformers 5.13.0 is the fixed processor
reference: mono 16-kHz F32 PCM becomes `[1,80,3000]` input features. Caller task,
language, prompt/suppression settings and tokenizer remain outside the engine.
Processor numerical parity, conversion, encoder/decoder execution, audio data
selection and output/memory qualification are future H work.
