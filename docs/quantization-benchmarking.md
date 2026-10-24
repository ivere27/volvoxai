# Quantization benchmarks

Use these tools to compare the same packed weights under different execution
arithmetic, measuring both completed generation calls and task quality. The
[RTX 3090 report](quantization-benchmark-20261023.md) is a historical baseline
for TinyLlama 1.1B, not a qualification of other models or serving concurrency.

Run commands from the repository root. Keep downloaded models, native binaries,
generated policies and raw results under ignored `build/` directories. Commit
reusable tools and configurations, plus small reviewed result summaries.

## Prepare the model and runtime

Use the pinned `tinyllama-1.1b-chat` entry in the
[checkpoint catalog](../tools/llm_baseline/models.json). Its revision is
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`.

```sh
python3 tools/llm_reference.py download --model tinyllama-1.1b-chat \
  --snapshot build/llm/snapshots/tinyllama
```

Prepare three packages using the [HF decoder workflow](hf-decoder.md):

| Directory under `build/quantization-benchmark/models/` | Package |
| --- | --- |
| `f32/` | Dense reference, computed in F32 |
| `i4-rtn/` | Grouped I4 RTN weights, group size 128 |
| `i8-rtn/` | Grouped I8 RTN weights, group size 128 |

Each directory must contain `graph.json` and `model.safetensors`. Use a
batch-one cached decoder with compact KV and maximum context 2048. For example,
export a dense package, then author each packed package once:

```sh
python3 tools/export_safetensors.py \
  --model build/llm/snapshots/tinyllama \
  --out build/quantization-benchmark/models/f32/model.safetensors \
  --target native-cpu --max-context 2048 --compact-kv --cached-decoder
```

```python
from pathlib import Path
import volvoxai as vx

models = Path("build/quantization-benchmark/models")
for bits in (4, 8):
    vx.quantize(models / "f32", output=models / f"i{bits}-rtn",
                weight_bits=bits, group_size=128, algorithm="rtn")
```

Use the repository Python SDK and generated bindings by setting
`PYTHONPATH=python:runtime/generated/python`. Model authoring needs a native
full library. Performance measurement needs a CUDA-enabled full library,
NVIDIA driver and NVML; the runner rejects CPU fallback. Set `VOLVOXAI_LIBRARY`
to the absolute path of that library. The historical build used SM86,
clang 14, nvcc 11.5 and `VOLVOXAI_CUDA_FAST_FP32=OFF`.

The historical dense package retained original BF16 storage and widened it for
F32 computation. A current export can produce different package bytes. Reuse
the original validated packages for an artifact-identical comparison; otherwise
record the new package hashes as a new run. The
[provenance record](benchmarks/quantization-20261023/provenance.json) identifies
the historical graphs, weights, runtime and tools. Later tool edits do not
change those identities.

The runtime tools use NumPy and the generated Synurang Python runtime. The
quality tests also use SafeTensors; development versions are in
[`python/requirements-test.txt`](../python/requirements-test.txt). Dataset
preparation additionally needs Transformers and PyArrow. Keep one Python
environment with those dependencies for the following commands.

## Measure generation

Copy the [performance configuration](../tools/llm_baseline/quantization_performance.json)
next to the prepared `models/` directory:

```sh
cp tools/llm_baseline/quantization_performance.json \
  build/quantization-benchmark/performance-config.json
python3 tools/remote_benchmark_runner.py \
  --config build/quantization-benchmark/performance-config.json \
  --out build/quantization-benchmark/performance.json
```

Despite its filename, this runner executes on the current machine. Paths in
the configuration resolve relative to the configuration file. It contains the
exact historical prompt IDs, nine variants and ten workloads. Prompt lengths
31, 128, 512 and 1536 include BOS. Every context has capacity 2048. Generation
budgets are 32 or 128 IDs; two additional cases request individual tokens.

Each condition runs one first-context cycle, two warmups, seven measured cycles
and a separate memory replay sampled at 20 ms. TTFT measures the first
`Generate(max_steps=1)` call; context creation is recorded separately. Decode
throughput covers the remaining committed IDs. Grouped calls provide throughput
and amortized latency; only individual-token calls provide inter-token
percentiles. No stop IDs shorten the requested work. The runner verifies
completed token counts and CUDA route evidence.

Keep other GPU workloads idle and record clocks, driver, build configuration and
variant order. Seven-cycle p95 is a descriptive statistic, not a production
tail estimate. NVML samples can miss short allocation peaks.

## Evaluate quality

The preparation tool fixes WikiText-2 raw test and ARC-Easy test revisions and
uses the local TinyLlama tokenizer. It does not choose samples from model scores.

```sh
python3 tools/prepare_quantization_quality.py \
  --tokenizer build/llm/snapshots/tinyllama \
  --output-dir build/quantization-benchmark/quality-data \
  --wiki-tokens 16384 --block-tokens 1024 --arc-questions 100
python3 tools/evaluate_quantization_quality.py \
  --policy build/quantization-benchmark/quality-data/quality-policy.json \
  --model f32=build/quantization-benchmark/models/f32 \
  --model i4-rtn=build/quantization-benchmark/models/i4-rtn \
  --model i8-rtn=build/quantization-benchmark/models/i8-rtn \
  --reference f32 --backend cuda \
  --packed-arithmetic dynamic-i8 f16 f32 f32-fast \
  --out build/quantization-benchmark/quality.json
```

The policy, every model and output path are required explicitly. Missing models
fail the run. The reference uses F32 once; each candidate uses every requested
arithmetic mode. Quality evaluation uses full causal prefill logits, with
prefill/incremental and fresh/reused-context checks before scoring.

The language metric uses 16 disjoint blocks of 1024 target tokens, each reset
with BOS. It is subset perplexity, not standard full-corpus WikiText perplexity.
ARC scores the first 100 questions by answer likelihood, reporting both raw and
Unicode-character-normalized accuracy. No evaluation data is used for RTN
fitting. These observations do not establish general quality equivalence.

Preparation records dataset and tokenizer file hashes. Its JSON also includes
local dataset paths, so the complete policy hash can differ between machines.
To compare evaluation inputs to the historical record, hash the `cases` array
with `hashlib.sha256(json.dumps(policy["cases"], sort_keys=True,
separators=(",", ":")).encode()).hexdigest()` and compare `quality.cases_sha256`
in the provenance record. Also compare tokenizer and dataset hashes.

## Run on an SSH host

The [remote spec example](../tools/llm_baseline/quantization_remote.example.json)
describes a prepared payload, weight sources and ordered commands. Copy it to
`build/quantization-benchmark/remote.json`. Local `payload_dir` and weight paths
resolve relative to this spec file. Edit weight hashes for your actual packages;
the example contains the historical hashes. Each step's `argv` executes from
the remote payload directory. Choose a remote Python executable with the
required dependencies.

Prepare `build/quantization-benchmark/payload/` with this layout:

```text
native/libvolvoxai.so
native/libsynurang_module_host.so
python/volvoxai/                  # repository SDK
runtime/generated/python/       # generated API and vendored Synurang runtime
tools/remote_benchmark_runner.py
tools/evaluate_quantization_quality.py
models/{f32,i4-rtn,i8-rtn}/graph.json
performance-config.json
quality-policy.json
```

Copy the config and generated policy into the payload root. Include the complete
SDK and generated Python directories, and compatible full CUDA and module-host
libraries. Keep weight files outside the payload archive. The controller adds
each weight at its declared `destination` after transfer or shared-file linking.
Use `local` for a local weight file, or replace it with `shared` containing an
absolute remote path to an existing read-only weight. Reuse shared weights when
available instead of transferring duplicates. Check remote free space before
large transfers and set `minimum_free_bytes` in the spec for the payload size.

```sh
python3 tools/run_remote_benchmark.py \
  --host user@gpu-host \
  --spec build/quantization-benchmark/remote.json \
  --out-dir build/quantization-benchmark/remote-run
```

The default scratch root is the remote user's home. Use `--remote-root` for an
existing absolute directory on another volume. The controller creates a unique
owned directory, verifies weight hashes, recovers `results/` and removes its
temporary files. It preserves shared weights. Inspect `remote-receipt.json` and
the supervisor result even when a command fails; a failed run is not a usable
benchmark baseline.

## Retain and validate results

Keep raw JSON results, exact policies, source manifests, binaries and full audit
receipts as local or external artifacts. Commit compact CSV summaries and
provenance beside a reviewed English report. The historical
[performance CSV](benchmarks/quantization-20261023/performance.csv) and
[quality CSV](benchmarks/quantization-20261023/quality.csv) preserve all reported
conditions; they do not replace the raw timing and per-case evidence.

Run the measurement-contract, numerical-scoring and remote-controller checks:

```sh
python3 -m unittest tools.tests.test_remote_benchmark_runner \
  tools.tests.test_quantization_quality tools.tests.test_run_remote_benchmark
```

The quality tests need a current native full library. Remote-controller tests
use local transport substitutes and do not submit jobs to an SSH host.
