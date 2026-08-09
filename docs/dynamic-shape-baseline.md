# Dynamic-shape redesign baseline

This document records the pre-redesign measurements used by DS0 of
[`PLAN.md`](../PLAN.md). It is evidence for regression comparisons, not a claim
about every deployment machine.

## Source state

- Date: 2026-08-01
- Git revision: `e9696b5` (`wasm+`)
- Branch: `shape`
- Runtime: Node `v20.11.1`, Linux x64
- Processor: AMD Ryzen 5 5600U with Radeon Graphics
- Backends measured: CPU JavaScript, strict WASM, browser WebGPU, and native CPU
- Existing user-owned untracked files were not part of the measurement.

## Reproduction

```bash
npm run typecheck
npm run baseline:runtime -- --backend=cpu

# Regenerate the forward WASM sidecar before recording a release candidate.
make build_wasm
npm run baseline:runtime -- \
  --backend=wasm \
  --wasm=dist/0.4.0/volvoxai.wasm

# Regenerate the CPU-only native object set before recording a release candidate.
make build_native
npm run baseline:runtime -- \
  --backend=native-cpu \
  --native-build-dir=build/cmake

# Compare active dynamic native execution with an independently compiled
# padded-maximum graph and exercise the bounded plan cache.
npm run baseline:native-dynamic -- --native-build-dir=build/cmake

# Uses the physical adapter selected by headless Chrome.
npm run baseline:webgpu

# Record the padded-static CPU comparison ceiling.
npm run baseline:padded-static
```

`npm run typecheck` passed, including all generated protobuf-enum and kernel
registry consistency checks.

The runtime baseline executes a single F32 Identity node with input shape
`[1, 65536]`, ten warm-up executions, 51 measured executions per run, and five
measurement runs. The JavaScript modes also verify context/result lifecycle and
retained-memory ownership. The native mode compiles
`tools/native_runtime_baseline.c` against the inference-only
`volvoxai_cpu_only` CMake object set in a temporary directory, exercises only
the public opaque-handle API, and deletes its temporary graph and executable.

The WASM artifact measured here was 213,172 bytes with SHA-256
`2e24c818fd0fdb835f5ee91bfdd7b3553f47a62cc80334dfdb25ed499ea63411`.
The native CPU-only executable that supplied the object set was 742,912 bytes
with SHA-256
`e83373b3032ff8dcd7ad7e894af2646269a64acb6c9cb2aa9e5db2d5cad9e718`.
These hashes identify this local evidence; they are not release checksums.

## Results

| Measurement | Pre-redesign result |
| --- | ---: |
| Median execution latency | 0.263477 ms |
| Minimum sample | 0.232809 ms |
| Maximum sample | 0.908676 ms |
| Reference median | 0.253058 ms |
| Allowed median | 0.265711 ms |
| Regression against reference | 4.1172% |
| Gate | pass |
| Compile time | 1.617786 ms |
| Compiled allocation | 524,288 bytes |
| One idle context array-buffer delta | 524,288 bytes |
| Two idle contexts array-buffer delta | 1,048,576 bytes |
| Stable output size | 262,144 bytes |

Run medians were:

```text
0.310527 ms
0.253429 ms
0.338309 ms
0.263477 ms
0.244532 ms
```

Lifecycle evidence passed:

- caller reads returned fresh owned arrays with identical bytes;
- a result remained readable after its context closed;
- already accepted work drained during close;
- context and result close operations were idempotent;
- unretained owners released their array-buffer storage.

## Generic cross-tier Identity report

The expanded harness was rerun on the same source state on 2026-08-01. P50 and
P95 below use the nearest-rank statistic across all 255 samples. The existing
CPU regression gate remains the median of the five run medians, so adding the
new report fields does not change that gate.

| Measurement | CPU JavaScript | Strict WASM | Native CPU |
| --- | ---: | ---: | ---: |
| Execution P50 | 0.277945 ms | 0.208984 ms | 0.014648 ms |
| Execution P95 | 0.486528 ms | 0.326927 ms | 0.041016 ms |
| Median of run medians | 0.262816 ms | 0.207011 ms | 0.016113 ms |
| Minimum / maximum sample | 0.173397 / 0.758251 ms | 0.159832 / 0.495215 ms | 0.011475 / 0.055664 ms |
| Population variance | 0.008305 ms² | 0.002744 ms² | 0.000138 ms² |
| Compile time | 1.624747 ms | 5.306757 ms | 3.707520 ms P50 / 3.760010 ms P95 |
| Logical input / output bytes | 262,144 / 262,144 | 262,144 / 262,144 | 262,144 / 262,144 |
| Logical live-tensor bytes | 524,288 | 524,288 | 524,288 |
| Output snapshot bytes | 262,144 | 262,144 | 262,144 |
| Byte-exact Identity digest | `1480849845` | `1480849845` | `1480849845` |

The per-run medians were:

```text
CPU JS:    0.311548, 0.352005, 0.185710, 0.262816, 0.245784 ms
WASM:      0.195729, 0.193476, 0.207011, 0.235936, 0.233652 ms
Native:    0.034912, 0.032471, 0.013672, 0.012451, 0.016113 ms
```

Timing scopes are explicit and should be used for per-tier regression, not as
a claim that the three host APIs have identical overhead. CPU JavaScript and
WASM measure the wall time of `ExecutionContext.execute()` through result
snapshot creation and close. Native uses `VxReport.execution_time_ms`, which
covers the native execution and owned result snapshot but not JavaScript
Promise/handle overhead. All three exclude caller output readback from the
latency sample and verify it separately.

### WASM allocation and high-water evidence

The public compilation report accounted for 524,288 logical tensor bytes. The
opt-in `WasmEngine` allocation profile reported the following physical linear
memory behavior for every independently owned context:

- final context heap capacity: 1,179,648 bytes;
- two simultaneously idle context heaps: 2,359,296 bytes;
- one grow by nine 64-KiB pages per context from the module's initial memory;
- five total observed grows across the compile seed, two idle contexts, latency
  context, and stable-result context;
- no weights or packed weights for this Identity graph.

The maximum forced-GC process snapshot was 122,687,488 RSS bytes,
11,068,552 heap-used bytes, 6,174,710 external bytes, and 1,597,995 ordinary
array-buffer bytes. Those Node process fields and the exact WASM linear-memory
capacity are deliberately separate. The process snapshots are sampled
high-water observations, not allocation-event tracing.

WASM result parity, fresh caller-owned reads, result readability after context
close, accepted-work drain, and idempotent context/result close all passed. A
WASM latency budget is not retroactively invented from this single host; this
report is the pre-redesign reference from which a budget can be approved.

### Native allocation and high-water evidence

The native public reports distinguish allocation lineage from logical tensor
bytes:

| Native field | Result |
| --- | ---: |
| Compile retained-lineage bytes | 4,401 |
| Context retained-lineage bytes | 6,027,873 |
| Execution lineage including result | 6,291,521 |
| Result snapshot bytes | 262,144 |
| Maximum sampled current RSS | 10,043,392 bytes |
| Maximum sampled current RSS delta from the retained baseline | 7,270,400 bytes |
| OS process high-water | 11,436,032 bytes |
| OS high-water delta during the measured phases | 0 bytes |

The OS high-water delta is zero because ELF loading had already established a
higher `ru_maxrss` before the caller-buffer baseline was sampled. It must not be
misread as zero runtime allocation. The phase-sampled current RSS and the public
allocation lineage expose the observed growth; allocator-exact native tensor
high-water is not available in the pre-redesign API. Freed allocations also
remain in the process allocator, so current RSS after handle close is not a
liveness oracle. Native output was byte-exact with the CPU/WASM input and used
one CPU worker thread.

## Generic browser WebGPU Identity report

Node `v20.11.1` on this host has no `navigator.gpu`. A strict Node runtime probe
therefore failed closed with `BACKEND_REQUIRED`; its candidate evidence said
`Backend provider 'webgpu' initialization failed: WebGPU is unavailable.` Deno,
the repository's normal surfaceless WebGPU parity runtime, is not installed.

Chrome `148.0.7778.178` and `/dev/dri/renderD128` are present. The new browser
baseline selected the physical `amd / gcn-5` adapter with packed-dot support.
It ran the same F32 Identity shape `[1,65536]`, with ten warm-ups and 51 measured
executions. Each latency sample waits for `GPUQueue.onSubmittedWorkDone()` and
closes its owned result; caller readback is measured separately.

| WebGPU field | Result |
| --- | ---: |
| Execution P50 / P95 | 2.400000 / 2.600000 ms |
| Minimum / maximum sample | 2.300000 / 2.800000 ms |
| Compile wall time / public report | 17.400000 / 16.900000 ms |
| Public logical allocation | 524,288 bytes |
| Requested GPUBuffer bytes after compile | 524,304 bytes |
| Requested GPUBuffer high-water during execution/readback | 1,048,592 bytes |
| Total requested buffer bytes across the run | 17,301,520 bytes |
| Pipelines / shader modules / bind groups | 1 / 1 / 1 |
| Buffers created / explicitly destroyed | 67 / 66 |
| Explicitly undestroyed bytes after close | 16 bytes |
| Caller readback for two owned copies | 5.000000 ms |
| Identity digest | `1480849845` (exact) |

The resource counters instrument `GPUDevice.createBuffer`, `GPUBuffer.destroy`,
pipeline, shader-module, bind-group, and command-encoder API calls. Requested
buffer byte totals are a reproducible API-level high-water observation, not an
allocator-exact physical VRAM measurement. The remaining 16-byte buffer is
reported rather than silently treated as freed. Chrome exposed 4,294,967,292-byte
buffer and storage-binding limits. Its post-close JavaScript heap snapshot was
4,665,991 bytes used of 7,601,519 bytes committed; browser heap and device
buffer measurements remain deliberately separate.

The earlier broad correctness probe remains useful but is not substituted for
the timing report:

```bash
node --experimental-websocket tools/run_webgpu_tests.mjs --adapter=hardware
```

It passed 45 of 48 correctness cases and exited nonzero for the existing grouped
Conv1D CPU-reference fixture and two Dequantize scale-gradient expectations.
Those operator fixture failures are tracked separately from this byte-exact
Identity baseline.

## Padded-static batch, sequence, and spatial comparisons

`tools/padded_static_baseline.mjs` compares independently compiled constant
graphs at an active shape and its padded maximum. It alternates the timing order,
uses five warm-ups and 31 samples per graph, and checks that the active output
region is byte-identical. These are pre-redesign CPU measurements and therefore
represent the available compute/memory-saving ceiling, not a claim that dynamic
binding has already achieved it.

| Workload | Active → padded dimensions | Input-byte ratio | Active p50 / p95 | Padded p50 / p95 | Padded / active p50 | Active-region parity |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Batch Linear, F32 width 128 | `B=2 → 16` | 8x | 0.140926 / 0.363506 ms | 0.742692 / 0.940295 ms | 5.270x | exact |
| Sequence Linear, F32 width 128 | `S=64 → 512` | 8x | 2.761513 / 2.961020 ms | 21.870157 / 24.794818 ms | 7.920x | exact |
| NHWC Conv2D, F32 C=8 | `24x24 → 72x72` | 9x | 2.152654 / 2.272630 ms | 19.306226 / 21.028237 ms | 8.969x | exact |

The batch graph uses `[B,128]`, the sequence graph `[1,S,128]`, and the spatial
graph a same-padded 3x3 HWIO convolution. Padded inputs are zero outside the
active region. Logical input storage grows from 1,024 to 8,192 bytes, 32,768 to
262,144 bytes, and 18,432 to 165,888 bytes respectively. The post-redesign gate
must rerun these exact dimensions through one polymorphic model/context and
report shape-bind/specialization overhead separately.

## Remaining host-specific budget decisions

- Promote host-specific WASM and native budgets only after the baseline numbers
  and timing scopes are reviewed. The WebGPU baseline likewise records evidence
  without inventing a cross-device latency budget. CPU JS is the only tier with
  an already committed five-percent median gate.

## Test-suite baseline sanitation

The first full `npm test` run reported 870 passing and two failing test results.
Those two results were one stale nested assertion and its parent aggregate in
`tests/js_wasm_direct_fallback_ops.test.mjs`, not two runtime failures. Commit
`a196aa6` had intentionally widened the tracked F32 GEMM policy from MR 4 to MR
8, while the older test still expected the MR-4-derived tile geometry.

The test expectations were synchronized to the already-authoritative kernel
policy (MR 8, KC 368, 23,808-byte working set). The isolated file then passed
all 17 tests. A new full-suite result is recorded after the DS1 additions settle
so its expanded test count is not confused with this pre-redesign baseline.

## Required post-redesign comparisons

Repeat this exact workload after the constant-only fast path lands. Record the
new shape-binding, specialization, allocation, and execution fields separately.
The constant-only path must remain within the regression budget fixed by DS0.

### DS3 constant-only CPU result

The isolated dynamic-v1 CPU context now runs the same F32 Identity workload via
`npm run baseline:cpu-shape`. Five independent Node processes each perform one
cold execution, ten warm executions, and 51 measured executions. As in the
committed runtime-baseline protocol, each worker collects unrelated
model-construction garbage before creating the latency context. The timing scope
starts at mandatory shaped-view validation and ends after creation of the exact,
result-owned output snapshot.

The 2026-08-01 result was:

| Measurement | Dynamic-v1 constant path |
| --- | ---: |
| Median of run medians | 0.228251 ms |
| P50 / P95 | 0.228371 / 0.335524 ms |
| Reference / allowed median | 0.253058 / 0.265711 ms |
| Regression against reference | -9.8029% |
| Gate | pass |
| Snapshot capture median | 5.985121 ms |
| Context creation median | 0.160453 ms |
| Cold execution median | 6.234642 ms |
| Logical input / output bytes | 262,144 / 262,144 |
| Context capacity / high-water | 524,288 / 524,288 bytes |
| Capacity grow events | 1 |
| Byte-exact Identity digest | `1480849845` |

Run medians were `0.218051`, `0.234382`, `0.223232`, `0.228251`, and
`0.235255` ms. The prebound constant Identity uses its already-validated input
capacity as a storage view, removing a redundant kernel copy; execution results
still receive fresh exact storage and remain readable after context closure.

The public lifecycle is gated separately by
`npm run baseline:cpu-public`. It uses `createRuntime({ backends: ['cpu'] })`,
provider compilation, public `ExecutionContext.execute`, and exact
`ExecutionResult` ownership over the same five-independent-process, 10-warmup,
51-sample workload. Its timing ends once the public result owns the exact output;
result disposal remains outside the sample.

The 2026-08-02 public result was:

| Measurement | Public Runtime / CPU provider |
| --- | ---: |
| Median of run medians | 0.207172 ms |
| Reference / allowed median | 0.253058 / 0.265711 ms |
| Regression against reference | -18.1326% |
| Gate | pass |
| Context creation median | 0.607498 ms |
| Cold execution median | 7.049858 ms |
| Logical input / output bytes | 262,144 / 262,144 |
| Byte-exact Identity digest | `1480849845` |

Public run medians were `0.207172`, `0.181923`, `0.175872`, `0.210789`, and
`0.237959` ms. Two preceding independent five-worker campaigns also passed at
`0.251084` and `0.239272` ms. Every worker reported the same concrete shape
signature and kept its retained result readable after context closure.

Add separate dynamic workloads instead of replacing this reference:

1. Repeated hot execution of one bounded shape.
2. Small → large → small on one context.
3. Alternating common shape signatures.
4. An adversarial sequence that forces plan eviction and capacity growth.
5. Padded maximum-shape versus active dynamic batch/sequence/spatial work.

CPU, WASM, WebGPU, and native CPU need their own reproducible hardware reports
before the DS8 portable gate. Cold compile, first specialization, warm hit,
logical bytes, capacity bytes, and high-water memory must not be combined into
one latency number.

### Post-redesign CPU variable-shape result

`npm run baseline:dynamic -- --backend=cpu` runs one polymorphic context against
independently compiled active-static and padded-maximum references. The
2026-08-02 campaign used 3 warmups and 15 measured executions per route:

| Workload | Active dynamic p50 | Active static p50 | Padded maximum p50 | Padded / dynamic |
| --- | ---: | ---: | ---: | ---: |
| Batch Linear, B 2 / 16 | 0.1990 ms | 0.0571 ms | 0.2006 ms | 1.01x |
| Sequence Linear, S 32 / 256 | 0.4335 ms | 0.3414 ms | 2.5166 ms | 5.81x |
| Spatial Conv2D, 16x16 / 48x48 | 0.8195 ms | 0.6449 ms | 5.3573 ms | 6.54x |
| Exact Add, 2x8 / 8x32 | 0.1939 ms | 0.0636 ms | 0.0890 ms | 0.46x |

All active, maximum, and extracted padded regions had zero numerical error.
The adversarial twelve-signature runs kept eight plan entries, performed
deterministic eviction, and stopped capacity growth at 8,192, 131,072,
131,072, and 98,304 bytes respectively.

The result is deliberately workload-dependent. Avoiding 8x sequence padding or
9x spatial padding materially wins; for tiny arithmetic whose padded kernel is
already cheaper than shape binding, dynamic execution does not improve latency.
Exact-signature hot hits skip plan reserialization and capacity scans, while
shape validation, result ownership, and bounded telemetry remain enabled.

The equivalent `--backend=wasm` campaign also had exact parity and one bounded
memory grow to a 1,179,648-byte heap high-water. Its padded/dynamic p50 ratios
were 0.41x (Batch Linear), 1.43x (Sequence Linear), 2.41x (spatial Conv2D), and
0.48x (exact Add). WASM's much faster small kernels make public binding overhead
more visible, but variable sequence and spatial work still benefits once the
avoided padded work is large enough.

### Post-redesign WebGPU result

`npm run baseline:webgpu` now uses the public dynamic-v1
Runtime→Model→CompiledModel→ExecutionContext lifecycle. The
legacy `Graph`/raw-array harness is no longer executable or accepted. The
2026-08-02 run used headless Chrome 148 on the physical `amd / gcn-5` adapter,
five warm-ups, and 31 measured executions. Each sample includes public
execution, `GPUQueue.onSubmittedWorkDone()`, exact result ownership, and result
close; the four parity reads are outside the latency samples.

The constant `[1,65536]` F32 Identity retained its pre-redesign 2.400000 ms P50:

| Constant WebGPU field | Dynamic-v1 result |
| --- | ---: |
| Compile wall / reported | 4.700000 / 4.100000 ms |
| Cold execution wall / reported | 25.100000 / 20.100000 ms |
| Warm P50 / P95 | 2.400000 / 2.700000 ms |
| Logical / capacity / high-water bytes | 524,288 / 524,288 / 524,288 |
| Plan hits / misses / evictions | 36 / 1 / 0 |
| Identity digest | `1091172597` (byte-exact) |

The digest differs from the DS0 value because this harness uses a signed
deterministic input sequence; output remains byte-identical to that input and
two caller reads own distinct buffers. The pre-redesign P50/P95 was
2.400000/2.600000 ms. No cross-device latency budget is inferred from the
0.100000 ms P95 difference.

The variable workload was one bounded ReLU with active shape `[1,8192]` and
padded maximum `[1,65536]`:

| Dynamic WebGPU field | Active dynamic | Padded maximum |
| --- | ---: | ---: |
| Input bytes | 32,768 | 262,144 |
| Cold wall time | 7.300000 ms | 6.300000 ms |
| Shape bind / provider / reported execution | 0.900000 / 1.100000 / 2.000000 ms | 0.200000 / 1.000000 / 1.200000 ms |
| Logical activation bytes | 65,536 | 524,288 |
| Capacity bytes after binding | 65,536 | 524,288 |
| Warm P50 / P95 | 2.500000 / 2.800000 ms | 2.400000 / 2.800000 ms |

For this launch-overhead-bound unary kernel, avoiding eight-fold work did not
reduce wall latency; it did reduce the active logical work and initial capacity
by exactly eight times. Active, padded, and extracted active-region results all
had zero error. Twelve adversarial lengths left eight metadata plans, six
deterministic evictions, one capacity grow, a 524,288-byte context capacity
high-water, and no extra ReLU pipeline. All 213 instrumented buffers were
explicitly destroyed by final runtime close; requested live-buffer high-water
across the complete constant and dynamic campaign was 1,638,448 bytes.

The physical run also exposed a browser-only conformance bug hidden by the old
mock: deferred uniform writes supplied byte counts to the typed-array
`GPUQueue.writeBuffer` overload, whose offset and size units are elements.
Specialization writes now normalize every source to a byte view before range
validation and commit. The stricter WebGPU dynamic/forward mock campaign passes
82/82 and the physical command reports `SUMMARY ALL_PASS`.

### Post-redesign native CPU variable-shape result

`npm run baseline:native-dynamic` compiles
`tools/native_dynamic_shape_performance.c` against the inference-only native
CPU object set and uses only the public native handles and shaped tensor
bindings. One bounded Add graph (`N=2..65536`, multiple of two) is compared
with an independently compiled constant `[65536]` graph. The 2026-08-02 run
used one CPU thread, five warm-ups, and 31 alternating measurements at active
`N=8192`:

| Native CPU field | Active dynamic | Padded maximum |
| --- | ---: | ---: |
| Compile time | 3.549561 ms | 3.449951 ms |
| Cold execution / shape bind | 0.117188 / 0.048000 ms | 0.265381 ms / static |
| Warm P50 / P95 | 0.013916 / 0.014404 ms | 0.124023 / 0.129395 ms |
| Logical activation bytes | 98,304 | 786,432 |
| Initial dynamic arena capacity | 131,072 | 1,048,576 after maximum bind |

The active path performed eight times less logical tensor work and its warm P50
was 8.91 times lower on this host. Active and maximum dynamic results, plus the
independent padded result, were element-exact. The initial active bind was cold,
the maximum bind was cold, and returning to the active signature was a cache
hit without shrinking the grown arena. A seven-shape adversarial sequence over
the four-entry plan cache deterministically evicted and rebuilt the first
signature; capacity stayed bounded at its 1,048,576-byte high-water with two
total grows. Process RSS and OS high-water are reported separately from the
runtime's exact logical/arena telemetry.
