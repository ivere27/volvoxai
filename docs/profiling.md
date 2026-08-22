# Profiling

This page is about **instrumentation**: which surfaces VolvoxAI exposes for
measuring time and memory, what each number means, and what it costs to collect.
It is not an optimization guide. For technique, see
[microkernel_optimization_guide.md](microkernel_optimization_guide.md),
[operator_fusion_patterns.md](operator_fusion_patterns.md), and
[xnnpack_optimization_guide.md](xnnpack_optimization_guide.md).

Every surface here is opt-in. None of them changes numerical results, and none
of them is enabled by default.

## Choosing a surface

| Question | Surface |
| --- | --- |
| Where does WASM compile time go? | [`__VOLVOX_WASM_COMPILE_PROFILE`](#wasm-compile-phase-breakdown) |
| How much memory does a lifecycle object hold? | [Memory evidence](#memory-evidence) |
| Is this kernel actually faster than ONNX Runtime's? | [`benchmark_kernel_unit`](#kernel-throughput) |
| Did a change regress end-to-end latency? | [Baseline harnesses](#end-to-end-baselines) |
| Which CUDA kernels dominate a training step? | [`VOLVOXAI_CUDA_PROFILE_PATH`](cuda.md) |

## Memory evidence

The runtime-scoped memory capture opt-in is the only surface that reports
memory rather than time. Its complete contract — evidence families, value
relations, and why envelopes are never summed — is in
[browser-runtime.md](browser-runtime.md); `ARCHITECTURE.md` covers the
ownership model and `runtime/README.md` the native adapter.

For profiling purposes, the three things worth knowing:

**It is per-runtime and inherited.** Pass `memoryCapture` to
`createRuntime()` and every Model, CompiledModel, ExecutionContext, and
ExecutionResult beneath it reports evidence. Absence performs no sampling at
all.

**Envelopes overlap; resources do not.** `PROCESS_RSS` contains the JS heap,
external bytes, and WASM linear memory all at once. Adding envelopes together,
or adding an envelope to a resource, produces a meaningless number. Compare
each envelope against itself over time.

**Node's `arrayBuffers` does not contain WASM linear memory.** A 64 MiB
`WebAssembly.Memory` moves `process.memoryUsage()` by:

~~~text
external     +64.0 MiB
rss          +64.1 MiB
arrayBuffers  +0.0 MiB
heapUsed      +0.0 MiB
~~~

So `PROCESS_ARRAY_BUFFER_BYTES` will not account for a WASM backend's arena,
and the `WASM_LINEAR` resource root is not double-counted by it. Count the
`WebAssembly.Memory` instance directly, through the provider resource record or
`buffer.byteLength`. WebGPU buffers have the same property: they are device
allocations, and no portable JavaScript counter observes their residency.

### What a relation claims

`valueRelation` is the confidence label, and it is load-bearing:

| Relation | Meaning |
| --- | --- |
| `EXACT` | The collector observed this quantity directly. |
| `ESTIMATED` | A stand-in for the exact quantity. Browser `performance.memory` is quantized, so the managed-heap fallback is `ESTIMATED` under the separate `browser-performance-memory/v1` sampler. |
| `REQUESTED` | An API-requested size. It makes no claim about physical residency, which is why WebGPU buffer sizes and native GPU allocation requests stay here. |
| `UNAVAILABLE` | Requested but not measurable on this platform; carries no byte value. |

Compare `sampler` before comparing two series: two collectors for the same
`kind` are different measurements, not interchangeable readings.

## Model load: no phase breakdown

There is currently no load-phase instrumentation on the shipping path.
`Model.load()` delegates to `ModelLoader`, which carries no timers. Load time is
observable only as wall time, through the
[baseline harnesses](#end-to-end-baselines) or your own clock.

`ts/core/RuntimeGraphLoader.ts` does define a `GraphLoadProfile` behind
`globalThis.__VOLVOX_GRAPH_LOAD_PROFILE`, and it is easy to mistake for the
loader you want. It is not:

- it is absent from all three shipped bundles and from every package entry,
  which `tests/build_profiles.test.mjs` asserts as a legacy-input boundary;
- `Model.load()` never reaches it; and
- `RuntimeGraphLoader.load()` has one caller in the repository, a single test.

Setting that global while calling `Model.load()` produces an empty result array,
not a breakdown. Treat it as instrumentation on a retired path until the loader
itself is either revived or removed.

## WASM compile phase breakdown

Set `globalThis.__VOLVOX_WASM_COMPILE_PROFILE` before compiling. When it is off,
each instrumentation site costs one field read.

This instruments `WasmEngine.compile()`, the engine-level graph allocation —
**not** `Runtime.compile()`. The results array stays empty until a context is
created, because that is when the engine actually lays the graph out in linear
memory:

~~~javascript
globalThis.__VOLVOX_WASM_COMPILE_PROFILE = true;
globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULTS = [];

const compiled = await runtime.compile(model, {
  backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
});
// still empty here
const context = await compiled.createContext();

console.table(globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULTS);
~~~

`__VOLVOX_WASM_COMPILE_PROFILE_RESULT` holds the most recent compile;
`__VOLVOX_WASM_COMPILE_PROFILE_RESULTS` appends every compile, which is what a
multi-graph package (encoder plus decoder) needs.

| Field | Covers |
| --- | --- |
| `preflightMs` | Pre-allocation checks. |
| `tensorAllocMs` / `metadataAllocMs` | Linear-memory allocation for tensors and node metadata. |
| `descriptorMs` | Node descriptor construction. |
| `packWeightMs` | Weight packing into kernel-preferred layouts. |
| `viewRefreshMs` | Rebinding typed-array views after the buffer is replaced. |
| `totalMs` / `unattributedMs` | Whole compile, and the part the phases did not claim. |
| `growMs` / `growCount` / `growPages` | `memory.grow()` cost, nested inside the allocator phases above. |
| `allocBytesCount` / `viewRefreshCount` / `descriptorCount` | Operation counts. |
| `tensorCopyBytes` / `packBytes` / `finalHeapBytes` | Bytes copied, packed, and finally resident. |

### Reading the phases

`growMs` is **nested inside** the allocator phases (`tensorAllocMs`,
`packWeightMs`, `metadataAllocMs`), not additional to them. Summing every
millisecond field double-counts it.

`tensorAllocMs` and `packWeightMs` also overlap: weight packing that happens
during the tensor-allocation window is counted by both. `descriptorMs`
subtracts the phases it nests, but `tensorAllocMs` does not, so the phases do
not partition `compile()` and `unattributedMs` — computed as the residual — can
come out negative. A real encoder compile:

~~~text
preflightMs        0.33     totalMs          142.44
tensorAllocMs    114.10     unattributedMs   -63.58
descriptorMs      23.12
packWeightMs      67.08     growMs             0.92  (nested)
viewRefreshMs      0.45     growCount             7
metadataAllocMs    0.93     growPages          1651
~~~

Treat each phase as an inclusive span, not a slice of a pie, and read a
negative `unattributedMs` as the amount of overlap rather than as missing work.
The source comment on `WasmCompileProfile` still claims the fields partition
`compile()`; the measurement above shows they do not.

`tools/runtime_baseline.mjs` wires this up for the `wasm` backend and is the
reference for the collect-and-reset pattern.

## Kernel throughput

`benchmark_kernel_unit` is the smallest-unit per-kernel measurement, and it
proves byte-exactness against the portable reference before reporting a
throughput. **Performance claims about a kernel should cite it**, not a
whole-model wall time that mixes in load, bind, and dispatch.

~~~bash
cmake -S . -B build/profiling -DCMAKE_C_COMPILER=clang
cmake --build build/profiling --target benchmark_kernel_unit -j"$(nproc)"

build/profiling/native/benchmark_kernel_unit --help
build/profiling/native/benchmark_kernel_unit --op qlinear --threads 1
~~~

Flags are `--op <name>`, `--threads <n>`, and `--no-check` (skip the exactness
check). Output is CSV:

~~~text
kernel,onnx_op,shape,unit,work,ms,throughput,exact
~~~

`VOLVOXAI_CPU_ISA` clamps the runtime dispatcher so one binary can report
several tiers:

~~~text
baseline | neon | neondotprod | neoni8mm | sve2 | avx2 | avxvnni | avx512vnni
~~~

An explicit value must name a tier available on the current architecture and
host. Unknown, cross-architecture, and unavailable requests fail engine
configuration instead of silently falling back or running an unclamped tier.

The target deliberately compiles without `-mavx2`-style flags: every kernel it
measures selects its ISA at runtime, so a target-wide flag would make the clamp
compare a tier against itself.

To put a kernel next to ONNX Runtime's equivalent:

~~~bash
python3 tools/compare_kernel_onnx.py \
  --binary build/profiling/native/benchmark_kernel_unit
python3 tools/compare_kernel_onnx.py --threads 1 --op qlinear
~~~

The same shape table drives both sides, and the script checks ORT's integer
result against an exact NumPy reference. Read the `onnx_exact` column before
trusting a ratio: on AVX2 without VNNI, ORT's u8 × s8 kernel uses a saturating
`VPMADDUBSW`, so a throughput number there may not correspond to a correct
result.

## WASM microbenchmarks

`tools/benchmark_wasm_*.mjs` measure a single WASM kernel — currently `silu`,
`groupnorm`, `qbatch_matmul`, and `dequantize_linear`. Each builds its own
module with clang and runs a size sweep.

Two rules these harnesses already follow, and any new one must:

- **Allocate through the module's `alloc_bytes` export.** Never hand a kernel a
  hardcoded linear-memory address. A raw address that happens to work will
  silently overlap the allocator's own bookkeeping as sizes change, and the
  resulting numbers are not comparable to anything.
- **Do not call libm per element.** In a WASM build `expf`, `logf`, `powf`, and
  friends are host imports, so calling one per element crosses the JS boundary
  per element and measures the boundary rather than the kernel. Use
  `native/src/kernels/fast_exp.h`, which the shipping kernels use.

When a microbenchmark disagrees with the model-level measurement by a
suspicious margin, suspect the harness before the kernel.

## End-to-end baselines

The `npm run baseline:*` scripts record whole-lifecycle latency across backends.
Their methodology, flags, and the recorded pre-redesign numbers are in
[dynamic-shape-baseline.md](dynamic-shape-baseline.md).

~~~bash
npm run baseline:runtime -- --backend=cpu-js
npm run baseline:runtime -- --backend=wasm --wasm=dist/0.4.0/volvoxai.wasm
npm run baseline:cpu-shape
npm run baseline:cpu-public
npm run baseline:dynamic
npm run baseline:native-dynamic -- --native-build-dir=build/cmake
npm run baseline:padded-static
npm run baseline:webgpu
~~~

Model-level benchmark results live in
[tiny-receipt-vqa-bpe1536-benchmark.md](tiny-receipt-vqa-bpe1536-benchmark.md)
and [efficientdet_tflite_vs_volvoxai.md](efficientdet_tflite_vs_volvoxai.md).

## Collection cost

Order-of-magnitude guidance, measured on Linux x64 / Node v20.11.1 with clang.
These are not a baseline record; re-measure on the machine that matters.

| Surface | Off | On |
| --- | --- | --- |
| `__VOLVOX_WASM_COMPILE_PROFILE` | One field read per site | Negligible against compile |
| Memory evidence, JS runtime | No sampling | ~9 µs per snapshot (`process.memoryUsage()`) |
| Memory evidence, native | No sampling | ~16 µs per snapshot, independent of resident size |
| `benchmark_kernel_unit` | — | Separate binary; never linked into a release build |

The native sampler reads `/proc/self/status` on Linux for both the instant and
the peak resident size. That choice is deliberate: `smaps`/`smaps_rollup` expose
the same total but only after walking every mapping's page tables, which costs
roughly 26 µs per resident mebibyte — about 26 ms per gibibyte, on every sampled
operation. `getrusage`'s `ru_maxrss` is cheaper still but reports a lazily
refreshed high-water mark that can read *below* the live resident size, so it
cannot back an `EXACT` peak.

Memory capture does not change the report fields beside it. In particular
`executionTimeMs` continues to time the execution and excludes collector cost.

## Keeping numbers honest

- **Build with clang.** The release artifacts are clang builds; gcc numbers do
  not transfer. This applies to throwaway scratch benchmarks too, where the
  temptation to use whatever `cc` points at is strongest.
- **Let the optimizer see the result.** A scratch benchmark that writes into a
  buffer it never reads can have the entire loop, and the allocation, removed.
  Escape the pointer (`__asm__ volatile("" :: "r"(p) : "memory")`) or consume
  the value.
- **Warm up, then measure.** Every harness here separates warmup from the
  measured window; a first call includes compilation, allocation, and page
  faults.
- **State what the number covers.** `providerTimeMs`, `executionTimeMs`, and
  `shapeBindTimeMs` in an `ExecutionReport` are nested spans, not addends.
- **Prove exactness before reporting throughput.** `benchmark_kernel_unit`
  checks against the portable reference by default; `--no-check` is for
  iterating, not for producing a number anyone will quote.
