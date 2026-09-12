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
| How long does loading or compilation take? | [API wall time](#model-load-and-compile-time) |
| How much memory does a lifecycle object hold? | [Memory evidence](#memory-evidence) |
| Is this native kernel actually faster? | [Kernel throughput](#kernel-throughput) |
| Did a change regress end-to-end latency? | [Baseline harnesses](#end-to-end-baselines) |
| Which CUDA kernels dominate a training step? | [`VOLVOXAI_CUDA_PROFILE_PATH`](cuda.md) |

## Memory evidence

Memory capture helps distinguish a growing process from a growing model or
context. Opt in when creating the Runtime, then inspect `report.memoryEvidence`
on its lifecycle responses:

```javascript
const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest({
  memoryCapture: new pb.MemoryCaptureOptions({
    protocol: 'volvoxai-memory-capture/v1',
    includeResourceInventory: true,
    includeDomainAttestation: true,
    requestedEnvelopes: [pb.MemoryEnvelopeKind.MEMORY_ENVELOPE_KIND_PROCESS_RSS],
  }),
}));
console.log(runtime.report.memoryEvidence);
```

Omitting `memoryCapture` disables collection. Capture is best effort: a requested
measurement may be `UNAVAILABLE`, a resource inventory may be `PARTIAL`, and a
domain attestation may be absent. The current C adapter samples RSS/peak RSS
where its platform sampler supports them; it does not provide a complete
allocator or device inventory. In particular, an empty partial inventory does
not mean that the engine allocated zero bytes.

[Architecture](../ARCHITECTURE.md) explains ownership, and the
[generated API reference](generated/api-contract.inference.md) defines the
capture options and evidence types. When interpreting the results:

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

This is a host-counter example, not a promise that the runtime reports every
Node counter. An ArrayBuffer count cannot stand in for WASM linear-memory
capacity. If instrumenting `WebAssembly.Memory` directly, record its
`buffer.byteLength` as capacity, not live tensor bytes. WebGPU buffer request
sizes likewise do not measure physical device residency.

### What a relation claims

`valueRelation` is the confidence label, and it is load-bearing:

| Relation | Meaning |
| --- | --- |
| `EXACT` | The collector observed this quantity directly. |
| `UPPER_BOUND` / `LOWER_BOUND` | A bound on the quantity, rather than an observed total. |
| `ESTIMATED` | An approximation whose collector and method must be identified. |
| `REQUESTED` | An API-requested size. It makes no claim about physical residency, which is why WebGPU buffer sizes and native GPU allocation requests stay here. |
| `UNAVAILABLE` | Requested but not measurable on this platform; carries no byte value. |

Compare `sampler` before comparing two series: two collectors for the same
`kind` are different measurements, not interchangeable readings. Also compare
`temporalCoverage`: an instantaneous RSS sample, a sampled maximum, and a
process-lifetime peak cover different intervals.

## Model load and compile time

Measure the generated API calls with a monotonic clock. Include the first-call
WASM initialization when measuring cold startup; initialize and warm up the
host first when measuring steady-state operations.

```javascript
const started = performance.now();
const compiled = await inference.compileModel(new pb.CompileModelRequest({
  modelId: model.modelId,
  policy: new pb.BackendPolicy({
    mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
    backends: ['wasm'],
  }),
}));
console.log({
  callMs: performance.now() - started,
  engineCompileMs: compiled.report.timings?.compileTimeMs,
});
```

Measure `LoadModel`, `CompileModel` and `CreateExecutionContext` separately when
comparing lifecycle phases. The clock around a call includes transport and host
work; `report.timings` contains the engine's compile/execution measurements.
These fields do not provide a breakdown of every compiler pass. Record the
artifact fingerprint, model, backend policy and warm-up procedure with results.

## Kernel throughput

The former standalone native kernel benchmark target has been retired. Use a
purpose-built scratch driver linked against the production inference library,
and keep its source revision, compiler, shape table, thread count, warm-up, and
numerical oracle with the report. Do not infer kernel throughput from a
whole-model wall time that mixes load, bind, dispatch, and result ownership.

`VOLVOXAI_CPU_ISA` clamps the runtime dispatcher so one binary can report
several tiers:

~~~text
baseline | neon | neondotprod | neoni8mm | sve2 | avx2 | avxvnni | avx512vnni
~~~

An explicit value must name a tier available on the current architecture and
host. Unknown, cross-architecture, and unavailable requests fail engine
configuration instead of silently falling back or running an unclamped tier.

A scratch driver should compile without target-wide `-mavx2`-style flags:
production kernels select their ISA at runtime, so such a flag can make an ISA
clamp compare a tier against itself. When comparing with another runtime, drive
both sides from the same shapes and verify integer output against an independent
oracle before trusting a ratio.

## WASM microbenchmarks

`tools/benchmark_wasm_*.mjs` measure a single WASM kernel — currently `silu`,
`groupnorm`, `qbatch_matmul`, and `dequantize_linear`. Each builds its own
module with clang and runs a size sweep.

Two rules these harnesses already follow, and any new one must:

- **Allocate through the module's `alloc_bytes` export.** Never hand a kernel a
  hardcoded linear-memory address. A raw address that happens to work will
  silently overlap the allocator's own bookkeeping as sizes change, and the
  resulting numbers are not comparable to anything.
- **Use the shipping math path.** The release uses C/WASM math implementations
  and the appropriate kernel approximations. A scratch module that imports a
  JavaScript math function for every element adds a host boundary that the
  shipping kernel does not have. Inspect the built module's imports and use the
  same kernel sources when comparing throughput.

When a microbenchmark disagrees with the model-level measurement by a
suspicious margin, suspect the harness before the kernel.

## End-to-end baselines

The retained `npm run baseline:*` scripts record whole-lifecycle latency across backends.
Their methodology, flags, and the recorded pre-redesign numbers are in
[dynamic-shape-baseline.md](dynamic-shape-baseline.md).

~~~bash
npm run baseline:batch -- --backend=wasm
npm run baseline:dynamic -- --backend=wasm
npm run baseline:native-dynamic -- --native-build-dir=build/cmake
~~~

Model-level benchmark results live in
[tiny-receipt-vqa-bpe1536-benchmark.md](tiny-receipt-vqa-bpe1536-benchmark.md)
and [efficientdet_tflite_vs_volvoxai.md](efficientdet_tflite_vs_volvoxai.md).

## Collection cost

Historical order-of-magnitude measurements on Linux x64 / Node v20.11.1 with
clang follow. The JavaScript sampler predates the shared C host and is not a
current API feature. Re-measure collection overhead on the deployed artifacts.

| Surface | Off | On |
| --- | --- | --- |
| Retired JavaScript sampler | No sampling | ~9 µs per snapshot (`process.memoryUsage()`) |
| Memory evidence, native | No sampling | ~16 µs per snapshot, independent of resident size |

The native sampler reads `/proc/self/status` on Linux for both the instant and
the peak resident size. That choice is deliberate: `smaps`/`smaps_rollup` expose
the same total but only after walking every mapping's page tables, which costs
roughly 26 µs per resident mebibyte — about 26 ms per gibibyte, on every sampled
operation. `getrusage`'s `ru_maxrss` is cheaper still but reports a lazily
refreshed high-water mark that can read *below* the live resident size, so it
cannot back an `EXACT` peak.

Memory capture does not change the report fields beside it. In particular
`report.timings.executionTimeMs` measures engine execution; the wall time around
the call also includes serialization and evidence collection.

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
- **State what the number covers.** Engine time in `OperationReport.timings`,
  host call wall time, and device-profiler spans cover different work. Do not add
  them as if they were disjoint phases.
- **Prove exactness before reporting throughput.** A scratch measurement must
  check against an independent oracle; an unchecked run is for iteration, not
  for producing a number anyone will quote.
