# Digit reader benchmarks

The [benchmark record](../BENCHMARK.md) contains the latest accuracy and latency
for this example. Published evidence stays in [reports](../reports).
The conversion, PTQ and 2,000-case validation commands are in the
[example guide](../README.md).

Run from the repository root with the frozen receipt benchmark specification:

```sh
SPEC=/path/inside/checkout/to/spec.json
NODE20=/path/to/node-v20.11.1-linux-x64/bin/node
OUT=examples/receipt_digit_reader/benchmarks/results

python3 examples/receipt_digit_reader/benchmarks/run.py \
  --spec "$SPEC" --out "$OUT/local" \
  --deno build/deno/target/webgpu-fix/deno --node "$NODE20" \
  --backends ort cpu wasm --host-label cpu-wasm-host --cpu 0 \
  --warmup 3 --repeat 15 --allow-busy-host
```

On the idle GPU host, use the same checkout-relative model/runtime inputs:

```sh
python3 examples/receipt_digit_reader/benchmarks/run.py \
  --spec "$SPEC" --out "$OUT/gpu" \
  --deno build/deno/target/webgpu-fix/deno \
  --backends cuda vulkan opengl webgpu --host-label gpu-host --cpu 0 \
  --warmup 3 --repeat 15
```

This entry point selects only the digit reader. The two receipt examples share
one implementation of session timing, host contention checks and publication
under [VQA benchmarks](../../tiny_receipt_vqa/benchmarks/README.md). That guide
contains dependency installation, the shared corpus/specification preparation
and publication commands. Keeping those checks together prevents the examples
from silently using different timing or validation rules.

Keep sources, tests, this guide, `reports/benchmark.json` and
`reports/validation.json` in Git. Publication replaces these two files with
the latest qualified results instead of appending dated reports. Use
`benchmarks/work/` for private corpora and converted packages, `results/` for
unpublished runs, and `.venv/` for a local environment. The example's
[.gitignore](../.gitignore) excludes those directories, logs and temporary
files; it does not exclude published report JSON.
