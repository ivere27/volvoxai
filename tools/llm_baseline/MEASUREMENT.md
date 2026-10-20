# Native model and WASM control baselines

This procedure measures the current engine with the preserved SmolLM2-135M and
Qwen3-0.6B F32 packages. It establishes observations before optimization. It does
not invent a regression allowance or compare incompatible historical captures.
[native_measurement.json](native_measurement.json) fixes the workload and how to
interpret each observation before a model measurement starts.

Use CPU full, CPU inference and CUDA full, with two CPU threads, batch/stream
one and context capacity 128. Each fresh timing process measures load, compile
and the first prefill/continuation. Then each of the fixed English/Korean
prefixes 1/16/31 runs two warmup cycles and five measured cycles. A cycle owns a
new cached context and executes one full prefill followed by four required ROW
steps. Decode inputs follow the frozen HF sequence. The first four selected IDs
are checked exactly; the fifth ID is recorded because the fourth continuation
also evaluates its resulting state. Export only `generated` I32, never logits
or layer KV. This output policy differs from numerical capture workflows.

Run three timing processes per cell serially. These are process-cold model
owners, with an uncontrolled OS page cache: verifying file hashes can warm the
cache. Do not drop system caches, stop services, change power policies or call
these an idle-host qualification. Completed public call durations include
serialization, engine execution and the requested I32 output. They are not
kernel times. Background work and swapping must remain visible in evidence.

Reuse large assets read-only. Create fresh pin receipts, results and private
temporary directories in the task-owned work area. The native loader may create
a private weight snapshot roughly as large as the selected weight file; reserve
disk for one model at a time. Do not duplicate the two F32 weight files.
The preparation receipt records filesystem blocks, unique F32 tensor payload
ranges and repeated graph references. Shared names and aliases are counted once
for unique payload; these overlapping views are never added to RSS or VRAM.

```sh
python3 tools/llm_baseline/measure_native_baseline.py --prepare \
  --source-root "$MEASUREMENT_SOURCE_ROOT" --model smollm2-135m \
  --profile full --backend cpu --package "$SMOL_EXECUTION_PACKAGE" \
  --reference "$SMOL_REFERENCE" --library "$MEASUREMENT_FULL_LIBRARY" \
  --bridge "$MEASUREMENT_MODULE_HOST" --out "$MEASUREMENT_RESULTS/smol-full-cpu-pins.json"

# The controller creates a new empty TMPDIR for every worker and runs cells serially.
python3 tools/llm_baseline/measure_native_baseline.py --model smollm2-135m \
  --profile full --backend cpu --pins "$MEASUREMENT_RESULTS/smol-full-cpu-pins.json" \
  --mode timing --tmpdir "$MEASUREMENT_PRIVATE_TMP" \
  --out "$MEASUREMENT_RESULTS/smol-full-cpu-timing-001.json"
```

Repeat with new names for timing 002/003, then `--mode memory` and `--mode trace`.
The aggregate retains raw nanosecond distributions and reports throughput as
the total selected IDs divided by the sum of completed-call durations. Cached
decode counts one selected ID per call; Generate counts all returned IDs. It
also reports the distribution of individual call rates and context creation
and retirement times. These call rates include output serialization and exclude
tokenization, context setup, loading and compilation; prefill rates count the
selected output ID, rather than the number of prompt tokens processed.
Use the inference library for CPU inference and the full library for CUDA. The
preparation receipt verifies the original graph, F32 weights, reference, SDK,
generated source, CUDA sources, WGSL authorities, build generators, library and
bridge. Workers verify these identities before
and after every run. REQUIRE and FORBID operator fallback apply to every cell.

The memory process samples `/proc/self/status` and private snapshot files every
10ms, and reads public resource snapshots at operation boundaries. Linux VmHWM
is a process-lifetime high-water report, including imports/setup. Samples within
a load/compile/execution interval are lower bounds on that interval's peak.
Inventory capacities, RSS, snapshot-file logical sizes/blocks and hardware-wide
GPU memory overlap; retain them separately. Truncated or partial inventories
remain partial, and a sampled value is never promoted to an exact whole peak.
CUDA memory workers also sample NVML compute-process residency and device-wide
memory every 10ms, with device free-memory observations before/after library
ownership. Process residency includes CUDA context and driver costs; it is not
the graph's requested allocation capacity. System CPU/swap counters at worker
boundaries preserve evidence of background work without adding a timing sampler.

The trace process uses sequential public NODES collectors with device timing
off and 64 MiB each, within the existing API limit. The eight segments cover
load/compile and the first cycle, all six cases, and owner retirement. It
preserves per-segment allocator peaks, copy/submission/synchronization events,
start/end inventories, and the engine's seeded accounting scope. Full raw
event and snapshot streams are retained in gzip sidecars beside each receipt;
their identities are verified by the controller and aggregate. These
instrumented durations are not timing samples. Reject lost records in every
segment. Never add segment or allocator peaks into a simultaneous model peak.

Execution visits attach and seed allocator observers. A retirement-only
collector does not reattach earlier execution owners, so missing allocator
rows there are unvisited and unknown, rather than measured zero. Public
inventory snapshots and the separate memory worker provide known cleanup
evidence. Inventory restored after retirement does not establish complete
process or driver leak freedom.
Memory workers compare the untruncated known module allocation inventory before
runtime creation and after all runtime/model/context owners retire. Runtime
unconsumed-result accounting must be known zero before retirement. The final
module inventory retains its declared coverage; an absent module result count
remains unknown rather than becoming a zero.
Reject lost trace events/snapshots or incomplete tracked allocator accounting.
Even complete accounting for `cuda.graph` or a host allocator retains its
declared inventory scope; it does not establish coverage of untracked owners.

For a small WASM control baseline, run each profile/mode in a fresh process:

```sh
node tools/llm_baseline/measure_wasm_baseline.mjs inference memory \
  "$HF_GENERATION_FIXTURES" "$MEASUREMENT_RESULTS/wasm-inference-memory.json"
```

Replace `inference` with `full`, and use `timing` in separate runs. The retained
tiny Qwen3.5 fixture checks exact IDs while reading the public WASM memory
partition at boundaries in memory mode. This establishes a control/fixture
baseline, not browser performance or full model capacity. Qwen3's preserved F32
source is 2,384,243,312 bytes, exceeding the current 1GiB WASM streaming source
profile and signed allocation ABI; do not upload it to claim a model run. Smol's
538,092,856-byte source fits that source limit, but whole-model memory and
execution still require their own admission and actual run.

The `full_prefix_generate` workload in the same [measurement policy](native_measurement.json) uses the trained
checkpoint's full-prefix E graphs and its frozen Korean/emoji HF input IDs.
It selects GREEDY, default RANDOM (default seed zero), and RANDOM with seed7,
temperature0.7, top-k40 and top-p0.85. Each new context admits exactly four
steps; one Generate call requests four IDs with a 16-byte output budget and
INLINE storage. No EOS shortening or numeric tensor output is requested.
Run CPU full and CUDA full: three fresh timing workers, two warmups and five
measured cycles for each of the six prompt/selection combinations; memory and
trace workers remain separate. Input tokenization is already checked by the E
workflow and is outside these Generate call durations. These results describe
full-prefix generation, not incremental KV performance or a tokenizer timing.

Use the same tool with `--workload full_prefix_generate`
and prepare pins with `--graph` pointing to the canonical E generation graph.
`--package` supplies the preserved F32 weight file and `--reference` the E HF
generation manifest directory. Exact four-ID output checks use the matching
default/filtered RANDOM witnesses; this policy adds no new oracle capture.

Aggregate completed workers with `summarize_native_baseline.py RESULTS --policy
POLICY --source-storage SOURCE_STORAGE_RECEIPT --out NEW_SUMMARY_JSON`.
Select the same `--workload` used by the workers. The aggregate requires the
planned process counts, exact case/repetition scope, stable
identities and retired private snapshots. Incomplete matrices remain `partial`.
It preserves process VmHWM, sampled stage/device bounds, allocator scope and
current artifact sizes as separate observations, without a regression verdict.

Prepare and run each cell explicitly with the worker commands above, using the
same `--workload` during preparation, execution and aggregation. The policy
stores shared observation rules once, with separate graphs, reference hashes,
outputs and backend/profile cells for each workload. Pins bind both the complete
policy bytes and the selected workload, so two workloads in one file cannot
exchange preparation receipts. New models are selected from the workload policy
rather than a hard-coded CLI alias list.

The fixed-image historical matrix and transport-continuation controllers have
been retired. Their source and original receipts remain in Git history. For a
new interrupted run, preserve failure evidence and prepare fresh pins and a new
matrix; do not reuse an old recovery plan or rewrite a receipt's identity.

For long remote runs, detach the controller, close its stdin, and redirect both
stdout and stderr to a durable task-owned log. Record its PID, exact launch
command, frozen input identities and launch UTC. Check the current
`progress.json`, the actual supervisor PID and current UTC together; an old SSH
output stream can remain stale after the controller has exited. Keep a fixed
worker deadline and preserve failure evidence before considering continuation.

The consolidated browser controller accepts `--selection all` with `--stage`
and `--cells`, or `--selection qualified` with `--reference`, `--profile` and
`--backend`. Qualified selection retains the first successful timing workers
under a prospective attempt cap; only idle/swap environmental failures allow a
retry. Correctness, identity, cleanup and worker-scope failures stop the series.
Timing values never decide which worker is retained.

`measure_portable_attention.py --series --plan PLAN --runtime-inventory PINS
--out NEW_DIRECTORY` runs paired before/after fresh timing workers and separate
memory workers. `--prepare` and ordinary single-worker calls remain available.
The shared edge telemetry is in `measure_native_baseline.py`; capacity and
attention controllers do not import a browser CLI to observe the host.

Capacity experiments retain a fail-closed source admission contract in
`measure_native_capacity.py`. Three of its five reviewed source hashes differ
from the current tree, so it must refuse preparation until that arithmetic has
been reviewed against the new implementation. This refactor does not requalify
those historical admission claims. The paced sensor runner also retains its
prospective base-policy hash; it currently differs from `multimodal_sensor.json`
and requires a reviewed new workload policy before execution.
