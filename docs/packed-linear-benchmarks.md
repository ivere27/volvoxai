# Automatic PackedLinear kernel benchmarks

The native and WASM benchmark tools derive their workload from model graph
metadata. Supply each encoder/decoder graph with `--graph`; the tools extract
all PackedLinear input/output shapes, packed weight bits and group sizes,
combine identical conditions, and retain their graph-node occurrence counts.
No hand-maintained model shape list is needed.

The benchmark measures dynamic-I8 PackedLinear using deterministic synthetic
inputs and weights. It does not load receipt data or trained weight payloads.
The model's graph supplies the workload dimensions and storage format; the
benchmark explicitly selects dynamic-I8 arithmetic because that choice belongs
to compilation rather than the serialized weights. Whole-model latency and
accuracy remain separate qualification steps.

## Run from model graphs

Build standalone fixtures using the pinned VolvoxAI build toolchain:

```sh
node tools/build_packed_linear_tile_fixture.mjs \
  --rows 4 --tile2 0 --out build/packed-before
node tools/build_packed_linear_tile_fixture.mjs \
  --rows 32 --tile2 1 --out build/packed-after
```

`--rows` selects a kernel's maximum row tile, independently of the model's
input row count. The builder defaults to production W4-only panel admission;
`--all-bits 1` explicitly enables the broader experimental policy.

Run Node or Chrome with graph extraction before the measured region:

```sh
node tools/benchmark_packed_linear_tiles.mjs \
  --variant before=build/packed-before --variant after=build/packed-after \
  --graph /path/to/encoder/graph.json --graph /path/to/decoder/graph.json \
  --browser chrome --out build/packed-chrome.json
```

The JavaScript tool calls the shared stdlib Python generator once, before
loading or measuring the kernels. It uses `python3` by default; `--python` or
`PYTHON` selects another interpreter. Chrome uses Puppeteer from the sibling
`unaveraged` checkout by default, or from `--puppeteer-root DIR`.

The native tool accepts the same workload arguments. Build each library from
the same fixture and numerical tests, with the intended tile definitions:

```sh
clang -shared -fPIC -O3 -ffp-contract=off \
  -Inative/src/kernels -Inative/src \
  -DVX_PACKED_LINEAR_TILE_ROWS=64 -DVX_PACKED_LINEAR_TILE2=1 \
  -DVX_PACKED_LINEAR_TILE_ALL_BITS=0 \
  examples/tiny_receipt_vqa/benchmarks/packed_row_tiles.c \
  native/tests/test_packed_linear_i8.c -o build/packed-after.so

python3 tools/benchmark_packed_linear_tiles.py \
  --variant before=build/packed-before.so --variant after=build/packed-after.so \
  --graph /path/to/encoder/graph.json --graph /path/to/decoder/graph.json \
  --out build/packed-native.json
```

Build the native `before` library with the same command using tile rows 4,
tile2 0 and its own output path. The first supplied variant is the comparison
reference. Every variant must produce byte-identical outputs and preserve
scratch guards. Independent C numerical oracles run before timing.

## Generated workload

For `X[M,K] @ W[N,K].T`, extraction obtains M by multiplying the input's leading
axes, K from its last axis and N from the output's last axis. Input and output
leading shapes must agree. A rank-one projection has M=1. Identical model cases
are merged with a list of source graph hashes and node IDs. Occurrences count
static graph nodes per invocation, not observed decoder token frequency.
Duplicate graph content is rejected so repeated arguments cannot inflate counts.

Boundary conditions are generated from the admitted tile and group sizes:

| Family | Coverage |
|---|---|
| `model` | Every resolved PackedLinear node in the supplied graphs |
| `row_boundaries` | 1/2 rows and neighbors of 4/8/16/32/64-row tiles, including two-row remainders |
| `tails` | K just below/at/above 32 and N=3/4/5 to cover both panel eligibility and fallback |
| `bit_controls` | W2/W3/W4/W8 at small row counts |
| `group_controls` | K just below/at/above G32/G64/G128/G256, with complete four-column panels |

There are currently 76 generated boundary/control conditions. With no `--graph`,
the tools run those conditions alone. `--no-boundaries` selects model-derived
conditions only; `--family NAME` filters families and can repeat or contain
comma-separated names. Conditions outside a fixture's capacity fail explicitly;
none are silently discarded.

Symbolic dimensions use declared finite bounds and explicit bindings, for
example `--dim sequence=128`. Equal min/max bounds can resolve a symbol directly.
The generator supports bounded integer expressions using `+`, `-`, `*`, `//`
and exact `/`; it evaluates no Python code or function calls. Zero-valued
symbols are allowed inside positive expressions, such as `past=0` in `past+1`.
Each resolved M/K/N must be a positive signed-I32 value. Unresolved, unsupported
or inconsistent PackedLinear shapes stop generation. This tool does not yet
import resolved execution plans; a timing-only profile lacks the tensor shapes
needed to replace explicit dynamic bindings.

The fixture allocates per-case tensors and scratch during setup, then reuses
them throughout timing. A 128 MiB tensor/scratch payload bound replaces the old
VQA-specific M/K/N caps. It includes scratch guards but excludes module,
allocator and stack overhead. Failed initialization releases all buffers.
Legacy v1 fixtures remain usable within their recorded fixed shape limits.

## Reproduce an exact workload

Every v2 result embeds `case_suite`: all selected cases, occurrences, source
hashes, symbolic bindings, generation policy, generator identity and canonical
case/suite hashes. It also identifies the current runner and case loader.
Original generation identity is preserved when replaying with a later loader.
A previous result can therefore reproduce its workload without the original
graph files:

```sh
node tools/benchmark_packed_linear_tiles.mjs \
  --variant before=build/packed-before --variant after=build/packed-after \
  --cases build/packed-native.json --browser node --out build/packed-replay.json
```

`--cases` accepts either a complete v2 benchmark report or a generated v2 suite.
It validates all hashes and node counts and cannot be combined with graph,
family, dimension or boundary overrides. A suite can also be inspected or
saved separately without running a benchmark:

```sh
python3 tools/packed_linear_cases.py \
  --graph /path/to/encoder/graph.json --out build/generated-cases.json
```

Benchmark outputs require new filenames. Default measurement uses two warmup
calls and five cycles with alternating variant order. `--work-per-sample`
(default 100,000,000) chooses the initial repeat count from M*K*N. Both runners
calibrate a common repeat count outside measured cycles toward 1 ms, capped at
10,000 calls, so small generated cases have observable timing. Each result
records the calibration and actual repeats. Zero or nonfinite measured times
fail instead of producing invalid speedup values. Input generation, allocation,
model dispatch and I/O are excluded; run on an otherwise idle host for usable
performance comparisons.

The removed `tools/packed_linear_tile_cases.json` was the historical 41-condition
manual workload. Its dated reports and source patches retain their original
identities. Reproduce those measurements with their frozen source patch;
newly generated workloads are separately identified and do not rewrite old
performance evidence. These tools generate benchmark conditions; they do not
change production runtime tile defaults or add startup autotuning.

## Check the tooling

Run the graph/replay and allocation-lifetime checks, then the JavaScript bridge
tests from the repository root:

```sh
python3 -m unittest discover -s tests -p 'test_packed_linear_*.py'
node --test tests/packed_linear_benchmark_cases.test.mjs
```

The allocation tests compile a temporary fixture with Clang's address and
undefined-behavior sanitizers and inject allocation failures. Kernel numerical
oracles and cross-variant output checks also run during every benchmark.
