#!/usr/bin/env python3
"""Bounded cached-context geometry and overlapping-owner observations.

Changing only the declared S upper bound reuses shared F32 weights. The fixed
HF prompt/IDs qualify short active computation in a larger capacity; they do not
qualify an active prompt as long as that capacity. Concurrent owners are live
together, but execute serially. This does not measure parallel throughput.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import importlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.llm_baseline import measure_native_baseline as baseline


ADMISSION_PROTOCOL = "coarse-provider-with-source-backed-private-packs/v1"
# These source identities justify the admission arithmetic, not a physical
# peak claim. Fail closed if the provider or CPU packing contract changes.
ADMISSION_SOURCES = {
    "native/src/runtime/public_api_bounded_domain_proof.inc": "dfadf4596a4dbb9b3522db458a435f51a2b9bc194bdcbf6fb641b91a34bbacad",
    "native/src/runtime/public_api_native_gpu_domain.inc": "d6a53895192ab1cee0df4e6cd10bcadca5c380dae28c874e72485ed67e3e9f54",
    "native/src/runtime/engine_runtime_f32_cpu.inc": "6e0d33615742b6aa14efe627c29b5a0d467ae7c5db644177ef22362a51e680f2",
    "native/src/runtime/engine_runtime_dispatch.inc": "b147b4b9f01cfa290ed4a053f6b9e4f0c610db66a8192ce995d2700b379d37e8",
    "native/src/kernels/gemm_f32.c": "b0a17b72a84088889a8393496864ecd40daaf97099056144cf9d9349f54a5843",
}


def cpu_pack_upper_bound(graph, weights):
    """Count every dense F32 node's private pack, including shared weights.

    The pinned implementation allocates k*ceil(n/nr)*nr floats per node,
    with nr in {8,16,32}. Rounding at 32 bounds each supported panel width.
    Counting disabled/fused consumers too is conservative. Other packing
    paths are outside this small decoder experiment and must fail closed.
    """
    allowed = {"Add", "ArgMax", "CrossSDPA", "Embedding", "Linear", "Mul",
               "RMSNorm", "Reshape", "RoPE", "SiLU"}
    baseline.require(all(node["opType"] in allowed for node in graph["nodes"]),
                     "source-backed pack bound requires the fixed dense decoder operators")
    baseline.require(all(value["dtype"] in ("F32", "I32") for name, value in weights.items()
                         if name != "__metadata__"), "source-backed pack bound requires F32 weights and I32 auxiliaries")
    tensors = dict(graph["inputs"])
    tensors.update({value["tensor"]: value for node in graph["nodes"]
                    for value in node["outputs"].values()})
    rows = []
    for index, node in enumerate(graph["nodes"]):
        if node["opType"] != "Linear":
            continue
        source = tensors.get(node["inputs"]["input"])
        output = node["outputs"].get("out")
        weight = weights.get(node["inputs"]["weight"])
        baseline.require(source and output and weight and source["dtype"] == "float32"
                         and output["dtype"] == "float32" and source["shape"] and output["shape"],
                         "dense pack input/output/weight contract differs")
        k, n = source["shape"][-1], output["shape"][-1]
        baseline.require(type(k) is int and type(n) is int and 0 < k <= (1 << 31) - 1
                         and 0 < n <= (1 << 31) - 1,
                         "dense pack features must be fixed positive I32 extents")
        baseline.require(weight["dtype"] == "F32"
                         and node.get("params", {}).get("weight_layout") == "dout_din"
                         and weight["shape"] == [n, k], "dense pack weight layout differs")
        size = 4 * k * ((n + 31) // 32) * 32
        baseline.require(size <= (1 << 64) - 1, "dense pack byte bound overflows")
        rows.append({"node_index": index, "node_id": node["id"], "k": k, "n": n,
                     "upper_bound_bytes": size})
    total = sum(row["upper_bound_bytes"] for row in rows)
    baseline.require(rows and total <= (1 << 64) - 1, "dense pack sum is empty or overflows")
    return dict(cpu_packed_upper_bound_bytes=total, maximum_panel_width=32,
                per_node=rows, scope="Private per-engine CPU packs; no sharing across nodes or contexts assumed.")


def bound(graph, capacity, weight_bytes, contexts, backend):
    """Conservative scheduling estimate, separate from engine physical proof.

    Every named output is counted even when a runtime could alias or recycle it.
    A second complete output domain allows staging; two largest output spans and
    twice KV allow temporary operations/page capacity. Fixed overhead is 1 GiB
    for process/driver and 256 MiB per owner. Immutable file views are shared on
    host. CPU GEMM packed caches and native device weight residency belong to
    each materialized engine state: charge one packed copy per live CPU context
    or one device copy per live device context. This estimate is not an exact
    allocator ceiling.
    """
    tensors = {}
    for node in graph["nodes"]:
        for value in node["outputs"].values():
            baseline.require(value["dtype"] in ("float32", "int32"), "capacity estimate requires F32/I32")
            size = math.prod(capacity if dim == "S" else dim for dim in value["shape"]) * 4
            tensors[value["tensor"]] = size
    output = sum(tensors.values()); largest = max(tensors.values())
    kv = sum(size for name, size in tensors.items() if name.endswith((".k_rope", ".v")))
    per_owner = 2 * output + 2 * largest + 2 * kv + (256 << 20)
    return {"output_domain_bytes": output, "largest_output_bytes": largest, "kv_domain_bytes": kv,
            "activation_staging_scheduling_bytes": per_owner,
            "per_context_scheduling_bytes": per_owner + weight_bytes,
            "host_scheduling_bytes": (1 + contexts if backend == "cpu" else 2) * weight_bytes + contexts * per_owner + (1 << 30),
            "device_scheduling_bytes": contexts * (weight_bytes + per_owner) + (1 << 30) if backend != "cpu" else 0,
            "scope": "Conservative external scheduling estimate; engine native backend admission is independent. No physical peak guarantee or proof of all hidden driver allocations."}


def canonical_provider_maximum(value):
    """Validate public bound arithmetic without interpreting coarse spaces."""
    baseline.require(value.get("proofProtocol") == "canonical-symbolic-domain-proof/v1" and
                     value.get("resourceProtocol") == "bounded-resource-maxima/v1",
                     "unknown memory bound protocol")
    baseline.require(bool(value.get("graphFingerprint")) and bool(value.get("shapeDomainProofIdentity")),
                     "memory bound identity is absent")
    def size(parent, field):
        baseline.require(field in parent and isinstance(parent[field], dict), "bound byte message is absent")
        raw = parent[field].get("bytes", 0)
        baseline.require(not isinstance(raw, bool) and isinstance(raw, (int, str)), "bound byte type differs")
        parsed = int(raw)
        baseline.require(0 <= parsed <= (1 << 64) - 1, "bound byte arithmetic overflow")
        return parsed
    unique = set(); resident = None; maximum_tensor = None
    for item in value.get("bounds", []):
        kind = item.get("kind")
        key = (item.get("budgetDomainId"), kind)
        baseline.require(bool(key[0]) and key not in unique, "duplicate or absent bound identity")
        unique.add(key)
        maximum = size(item, "maximumBytes")
        if "limitBytes" in item:
            baseline.require(maximum <= size(item, "limitBytes"), "bound exceeds advertised limit")
        case_ids = set(); totals = []
        for case in item.get("cases", []):
            baseline.require(bool(case.get("caseId")) and case["caseId"] not in case_ids, "duplicate or absent peak case")
            case_ids.add(case["caseId"])
            terms = case.get("terms", [])
            baseline.require(bool(terms), "empty peak case")
            term_ids = set(); total = 0
            for term in terms:
                baseline.require(bool(term.get("termId")) and term["termId"] not in term_ids, "duplicate or absent bound term")
                term_ids.add(term["termId"])
                total += size(term, "upperBoundBytes")
                baseline.require(total <= (1 << 64) - 1, "bound checked sum overflow")
            baseline.require(total == size(case, "totalBytes"), "peak case checked sum differs")
            totals.append(total)
        if totals:
            baseline.require(max(totals) == maximum, "peak case maximum differs")
        if key[0] == "provider-resident" and kind in (3, "MEMORY_BOUND_KIND_ORDINARY_RESIDENT"):
            resident = maximum
        if key[0] == "maximum-tensor" and kind in (1, "MEMORY_BOUND_KIND_MAXIMUM_TENSOR"):
            maximum_tensor = maximum
    baseline.require(resident is not None and maximum_tensor is not None and maximum_tensor <= resident,
                     "canonical provider/tensor maximum is absent or inconsistent")
    return resident


def proof_scheduling_estimate(plan, proof):
    maximum = canonical_provider_maximum(proof)
    contexts = plan["live_contexts"]
    baseline.require(type(contexts) is int and 0 < contexts <= 8, "live owner bound differs")
    contract = plan.get("source_backed_admission", {})
    baseline.require(contract.get("protocol") == ADMISSION_PROTOCOL and
                     contract.get("source_sha256") == ADMISSION_SOURCES and
                     all(plan["pins"]["source"].get(path) == sha for path, sha in ADMISSION_SOURCES.items()),
                     "source-backed admission contract changed or is absent")
    baseline.require(maximum >= plan["pins"]["storage_inventory"]["source_file_logical_bytes"],
                     "provider bound does not cover the immutable file mapping")
    cpu = plan["pins"]["backend"] == "cpu"
    pack_upper = contract["private_weights"]["cpu_packed_upper_bound_bytes"]
    baseline.require(type(pack_upper) is int and
                     0 < pack_upper <= plan["pins"]["storage_inventory"]["unique_payload_bytes"],
                     "private CPU packs exceed the unchanged validation preflight weight charge")
    private_pack = pack_upper if cpu else 0
    per_owner = maximum + private_pack
    baseline.require(contexts * per_owner + (1 << 30) <= (1 << 64) - 1,
                     "source-backed admission sum overflows")
    # R already covers mapped immutable weights plus native GPU per-consumer
    # converted/packed/aligned representations. Keep R whole in each space;
    # add only CPU packs, whose node cache allocations are outside that proof.
    return dict(provider_resident_maximum_bytes=maximum,
                host_scheduling_bytes=contexts * per_owner + (1 << 30),
                device_scheduling_bytes=contexts * maximum + (1 << 30) if not cpu else 0,
                per_context_scheduling_bytes=per_owner,
                private_cpu_pack_bytes_per_owner=private_pack, admission_protocol=ADMISSION_PROTOCOL,
                whole_physical_peak_qualified=False,
                scope="Full mixed-space coarse provider maximum per owner/space, plus source-backed private CPU packs and 1GiB caller/process/driver allowance. No partition or subtraction from the provider bound; overlapping spaces are not summed. No exact whole physical peak claim.")


def collect_repeatable_cycles(receipt, cycle, warmup_cycles, measured_cycles):
    """Retain excluded warmup IDs and reject drift outside public-call timing."""
    cold = cycle("cold")
    receipt["cold_first"] = cold
    expected = cold["selected_ids_by_owner"][0]
    def check(value):
        baseline.require(len(expected) == 5 and all(ids == expected for ids in value["selected_ids_by_owner"]),
                         "all five returned IDs differ across cold/warm/measured cycles or live owners")
    check(cold)
    for ordinal in range(-warmup_cycles, measured_cycles):
        value = cycle(ordinal)
        if ordinal < 0:
            receipt.setdefault("excluded_warmup_cycles", []).append({
                "ordinal": ordinal, "selected_ids_by_owner": value["selected_ids_by_owner"]})
        else:
            receipt["cycles"].append(value)
        check(value)


def prepare(args):
    source = json.loads(args.graph.read_text())
    graph = copy.deepcopy(source)
    graph["dimensions"]["S"]["max"] = args.capacity
    metadata = json.loads((args.reference / "manifest.json").read_text())
    baseline.require(args.capacity <= metadata["config"]["max_position_embeddings"], "capacity exceeds original model context")
    baseline.require(args.capacity >= 36 and args.capacity % 2 == 0, "capacity must fit fixed short HF cases and two-token pages")
    directory = args.out.parent
    directory.mkdir(parents=True, exist_ok=True)
    graph_path = directory / (args.out.stem + ".graph.json")
    with graph_path.open("x") as stream:
        json.dump(graph, stream, separators=(",", ":")); stream.write("\n")
    policy = baseline.read_policy(ROOT / "tools/llm_baseline/native_measurement.json")
    policy["max_context"] = args.capacity
    policy["models"][args.model]["execution_graph_sha256"] = baseline.identity(graph_path)["sha256"]
    policy["cells"] = [{"profile": "full", "backend": backend} for backend in ("cpu", "cuda", "vulkan", "opengl")] + [{"profile": "inference", "backend": "cpu"}]
    policy_path = directory / (args.out.stem + ".policy.json")
    with policy_path.open("x") as stream:
        catalog = {"format": policy["format"], "shared": {},
                   "workloads": {policy["workload"]: {key: value for key, value in policy.items()
                                                     if key not in ("format", "workload")}}}
        json.dump(catalog, stream, indent=2); stream.write("\n")
    fake = argparse.Namespace(source_root=args.source_root, package=args.package, graph=graph_path,
        reference=args.reference, library=args.library, bridge=args.bridge, policy=policy_path,
        model=args.model, profile=args.profile, backend=args.backend)
    pins = baseline.prepare(fake, policy)
    baseline.require(all(pins["source"].get(path) == sha for path, sha in ADMISSION_SOURCES.items()),
                     "source-backed admission implementation differs")
    with args.package.joinpath("model.safetensors").open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        baseline.require(header_length <= 16 << 20, "unexpected weight header extent")
        private_weights = cpu_pack_upper_bound(graph, json.loads(stream.read(header_length)))
    baseline.require(private_weights["cpu_packed_upper_bound_bytes"] <= pins["storage_inventory"]["unique_payload_bytes"],
                     "private CPU packs exceed the unchanged validation preflight weight charge")
    case = next(case for case in pins["cases"] if case["id"] == "prompt0-prefix031")
    estimate = bound(graph, args.capacity, pins["storage_inventory"]["unique_payload_bytes"], args.contexts, args.backend)
    probe = bound(graph, graph["dimensions"]["S"]["min"],
                  pins["storage_inventory"]["unique_payload_bytes"], 1, args.backend)
    receipt = {"format": "volvox-native-capacity-plan/v1", "status": "PREPARED", "helper": baseline.identity(__file__),
        "pins": pins, "capacity": args.capacity, "live_contexts": args.contexts, "case": case,
        "source_backed_admission": {"protocol": ADMISSION_PROTOCOL, "source_sha256": ADMISSION_SOURCES,
                                   "private_weights": private_weights},
        "scheduling_estimate": estimate, "cpu_threads": 2, "minimum_free_host_bytes_after_estimate": 4 << 30,
        "compile_probe_scheduling_estimate": probe,
        "compile_probe_scope": "Native CompileModel builds a validation engine and may execute a minimum-domain GPU forward. This is not an execution-free metadata query; one validation consumer and private weight packing/residency are preflighted at S.min before compiler admission.",
        "minimum_free_device_bytes_after_estimate": 4 << 30,
        "minimum_free_disk_bytes": 2 * pins["storage_inventory"]["source_file_logical_bytes"] + (1 << 30),
        "warmup_cycles": 2, "measured_cycles": 5, "fresh_timing_processes": 3,
        "memory_sample_interval_ns": 10_000_000,
        "memory_scope": "Separate sampled VmRSS/PSS/NVML and known module/runtime inventory at first/last-token and owner boundaries; VmHWM includes process setup and retained observer records. Listed owner capacity is an observed boundary high-water, not an exact allocation-time peak. No addition of overlapping views or exact whole peak claim.",
        "active_sequence_scope": "HF prefix31 plus four appended independent greedy IDs; final call returns one additional ID without HF reference. Capacity padding is not active long-context qualification.",
        "concurrency_scope": "Private state for live overlapping contexts, serial interleaved calls; no parallel request throughput claim.",
        "context_limit_scope": "ExecutionLimits.max_memory_bytes refuses native GPU/paged caches; conservative admission before model load reserves all owners. Host headroom is checked before each owner; memory workers also query device headroom. Timing workers close preflight-only NVML before model loading.",
        "swap_policy": "Any system pswpin/out delta invalidates performance observations; no service changes or automatic retry.",
        "service_actions": 0}
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2); stream.write("\n")


def available_host_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host available memory is unknown")


def admission(plan, nvml, estimate=None):
    host = available_host_bytes()
    estimate = estimate or plan["scheduling_estimate"]
    device = nvml.sample() if nvml else []
    result = {"host_available_bytes": host, "device": device, "estimate": estimate,
              "host_admitted": host >= estimate["host_scheduling_bytes"] + plan["minimum_free_host_bytes_after_estimate"]}
    result["device_admitted"] = not nvml or bool(device and device[0]["free_bytes"] >= estimate["device_scheduling_bytes"] + plan["minimum_free_device_bytes_after_estimate"])
    result["admitted"] = result["host_admitted"] and result["device_admitted"]
    return result


def finish_observers(sampler, nvml, receipt):
    """Retain failure evidence and close every observer before reporting errors."""
    errors = []
    if sampler:
        try:
            sampler.stop(); receipt["memory_samples"] = sampler.samples
            receipt["observed_memory"] = {name: max((row.get(name, 0) for row in sampler.samples), default=0) for name in ("VmRSS", "VmHWM", "Pss", "VmSwap", "SwapPss")}
            receipt["maximum_sample_gap_ns"] = max((b["monotonic_ns"] - a["monotonic_ns"] for a, b in zip(sampler.samples, sampler.samples[1:])), default=None)
            receipt["nvml_sampled_max_worker_bytes"] = max((device["worker_process_used_bytes"] for row in sampler.samples for device in row.get("nvml", []) if device["worker_process_used_bytes"] is not None), default=None)
            if sampler.error is not None:
                errors.append("memory sampler failed: " + str(sampler.error))
        except Exception as error:
            errors.append("memory observer cleanup failed: " + repr(error))
    if nvml:
        try:
            receipt["nvml_after_close"] = nvml.sample()
        except Exception as error:
            errors.append("final NVML observation failed: " + repr(error))
        finally:
            try:
                nvml.close()
            except Exception as error:
                errors.append("NVML close failed: " + repr(error))
    try:
        receipt["process_lifetime_final"] = baseline.proc_status()
    except Exception as error:
        errors.append("final process observation failed: " + repr(error))
    receipt["observer_cleanup_errors"] = errors
    baseline.require(not errors, "; ".join(errors))


def execute(args, plan, receipt):
    pins = plan["pins"]
    sys.path[:0] = [str(Path(pins["source_root"]) / "python"), str(Path(pins["source_root"]) / "runtime/generated/python")]
    vx = importlib.import_module("volvoxai")
    from volvoxai._diagnostics import _proto_json
    pb = vx.pb
    nvml = baseline.NvmlSampler() if pins["backend"] != "cpu" else None
    sampler = None
    try:
        disk = os.statvfs(args.tmpdir)
        baseline.require(disk.f_bavail * disk.f_frsize >= plan["minimum_free_disk_bytes"], "insufficient disk for immutable private snapshots and report headroom")
        estimate = plan["compile_probe_scheduling_estimate"] if args.mode == "proof" else plan["scheduling_estimate"]
        if args.admission_proof:
            baseline.require(args.mode != "proof", "a proof probe cannot consume another probe")
            proof_receipt = json.loads(args.admission_proof.read_text())
            baseline.require(proof_receipt.get("status") == "PROOF_CAPTURED" and proof_receipt["plan"] == baseline.identity(args.plan),
                             "admission proof is not successful or belongs to another plan")
            refined = proof_scheduling_estimate(plan, proof_receipt["compiled_memory_bounds"])
            receipt["admission_proof"] = {"receipt": baseline.identity(args.admission_proof), "refined_scheduling_estimate": refined,
                                           "original_scheduling_estimate": plan["scheduling_estimate"]}
            # Compiler attestation supplies a resource-domain maximum. Keep
            # the earlier heuristic for comparison, and use the complete
            # attested maximum with explicit extra charges for admission.
            estimate = refined
        receipt["effective_scheduling_estimate"] = estimate
        receipt["admission"] = admission(plan, nvml, estimate)
        if not receipt["admission"]["admitted"]:
            receipt["status"] = "SKIPPED_CONSERVATIVE_RESOURCE_BOUND"
            return
        if args.mode == "timing" and nvml:
            # Preflight safety telemetry finishes before model loading. There
            # is no device observer/query between the timed workload calls.
            nvml.close(); nvml = None
        if args.mode in ("memory", "proof"):
            sampler = baseline.RssSampler(plan["memory_sample_interval_ns"], args.tmpdir, nvml)
            sampler.thread.start()
        def mark(phase):
            if sampler:
                sampler.phase = phase
                receipt.setdefault("phase_markers", []).append({"phase": phase, "monotonic_ns": time.perf_counter_ns()})
        with vx.open_library(pins["inputs"]["library"]["path"], loader=pins["inputs"]["bridge"]["path"]) as host:
            client = vx.VxInferenceServiceClient(host)
            profiling = vx.VxProfilingServiceClient(host)
            platform_info = vx.VxPlatformServiceClient(host).get_platform_info(pb.Empty())
            expected_profile = pb.BuildProfile.BUILD_PROFILE_FULL if pins["profile"] == "full" else pb.BuildProfile.BUILD_PROFILE_INFERENCE
            baseline.require(platform_info.build_profile == expected_profile, "library profile differs")
            receipt["platform"] = _proto_json(platform_info)
            def snapshot(label):
                if sampler:
                    observed = profiling.get_resource_snapshot(pb.GetResourceSnapshotRequest(module=pb.Empty(), include_device=pins["backend"] != "cpu", max_allocations=65536))
                    receipt.setdefault("resource_snapshots", []).append({"phase": label, "snapshot": _proto_json(observed.snapshot)})
            snapshot("before_runtime")
            with ExitStack() as owners:
                mark("runtime_create")
                runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=plan["cpu_threads"]))
                owners.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
                mark("load"); start = time.perf_counter_ns()
                model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                    graph_path=pins["inputs"]["graph"]["path"], weight_paths=[pins["inputs"]["weights"]["path"]]))
                receipt["load_ns"] = time.perf_counter_ns() - start if args.mode == "timing" else None
                owners.callback(client.release_model, pb.ModelRef(model_id=model.model_id)); snapshot("loaded")
                mark("compile"); start = time.perf_counter_ns()
                compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                        backends=[pins["backend"]], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                owners.callback(client.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                receipt["compile_ns"] = time.perf_counter_ns() - start if args.mode == "timing" else None
                receipt["compile_route"] = baseline.route(compiled.report, pb, pins["backend"])
                receipt["compiled_memory_bounds"] = _proto_json(compiled.memory_bounds)
                canonical_provider_maximum(receipt["compiled_memory_bounds"])
                if args.admission_proof:
                    baseline.require(receipt["compiled_memory_bounds"] == proof_receipt["compiled_memory_bounds"],
                                     "compiler memory attestation differs from the prospective probe")
                snapshot("compiled")
                capacity = plan["capacity"]; case = plan["case"]
                caches = [f"layer.{i}.{suffix}" for i in range(pins["config"]["num_hidden_layers"]) for suffix in ("k_rope", "v")]
                def cycle(ordinal):
                    contexts = []; chosen = [[] for _ in range(plan["live_contexts"])]; observations = []
                    mark(f"cycle{ordinal}:contexts_create"); start = time.perf_counter_ns()
                    try:
                        for peer in range(plan["live_contexts"]):
                            # Initial estimate reserves all owners; additionally preserve fixed free headroom while growing.
                            baseline.require(available_host_bytes() >= estimate["per_context_scheduling_bytes"] + plan["minimum_free_host_bytes_after_estimate"], "host free headroom exhausted before owner creation")
                            if nvml:
                                baseline.require(nvml.sample()[0]["free_bytes"] >= plan["minimum_free_device_bytes_after_estimate"], "device free headroom exhausted before owner creation")
                            context = client.create_execution_context(pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id,
                                graph=pb.GraphExecution(slots=1, require_incremental=True, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED),
                                limits=pb.ExecutionLimits(cache=pb.ContextCacheOptions(tensors=caches, page_tokens=2, max_pages=capacity // 2, clear_on_recycle=True))))
                            contexts.append(context.context_id)
                            snapshot(f"cycle{ordinal}:owner{peer}:created")
                        create_ns = time.perf_counter_ns() - start
                        for step in range(5):
                            length = case["prefix_length"] + step
                            tokens = case["initial_ids"] + case["greedy_ids"][:step] + [0] * (capacity - length)
                            values = [pb.Tensor(name="token_ids", dtype=pb.DataType.DATA_TYPE_I32, shape=[1, capacity], inline=struct.pack("<" + "i" * capacity, *tokens)),
                                pb.Tensor(name="position_ids", dtype=pb.DataType.DATA_TYPE_I32, shape=[capacity], inline=struct.pack("<" + "i" * capacity, *range(capacity))),
                                pb.Tensor(name="attention_mask", dtype=pb.DataType.DATA_TYPE_I32, shape=[1, capacity], inline=struct.pack("<" + "i" * capacity, *([1] * length + [0] * (capacity - length))))]
                            if step == 0:
                                values.append(pb.Tensor(name="active_lengths", dtype=pb.DataType.DATA_TYPE_I32, shape=[1], inline=struct.pack("<i", length)))
                            for peer, context in enumerate(contexts):
                                if step == 0:
                                    baseline.require(available_host_bytes() >= estimate["per_context_scheduling_bytes"] + plan["minimum_free_host_bytes_after_estimate"], "host free headroom exhausted before initializing owner")
                                    if nvml:
                                        baseline.require(nvml.sample()[0]["free_bytes"] >= estimate["per_context_scheduling_bytes"] + plan["minimum_free_device_bytes_after_estimate"], "device free headroom exhausted before initializing owner")
                                mark(f"cycle{ordinal}:owner{peer}:step{step}")
                                request = pb.ExecuteRequest(context_id=context, inputs=pb.InputBindings(values=values, reuse=[] if step == 0 else ["active_lengths"]),
                                    scope=pb.ExecutionScope(entrypoint="initialize", full=pb.Empty()) if step == 0 else pb.ExecutionScope(entrypoint="transition", slot_rows=pb.SlotRows(slots=[pb.SlotRow(position=length - 1)])),
                                    outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE, selection=pb.TensorOutputSelection(names=["generated"])))
                                start = time.perf_counter_ns(); packet = client.execute(request); elapsed = time.perf_counter_ns() - start
                                route = baseline.route(packet.report, pb, pins["backend"])
                                baseline.require(len(packet.outputs) == 1 and len(packet.outputs[0].inline) == capacity * 4, "output policy or geometry differs")
                                selected = struct.unpack_from("<i", packet.outputs[0].inline, (length - 1) * 4)[0]
                                if step < len(case["greedy_ids"]):
                                    baseline.require(selected == case["greedy_ids"][step], "independent short-prefix HF ID differs")
                                chosen[peer].append(selected)
                                observations.append({"owner": peer, "step": step, "phase": "prefill" if step == 0 else "decode", "call_ns": elapsed if args.mode == "timing" else None, "route": route})
                                if step in (0, 4): snapshot(f"cycle{ordinal}:owner{peer}:step{step}:completed")
                        baseline.require(all(ids == chosen[0] for ids in chosen), "overlapping context state differs")
                        snapshot(f"cycle{ordinal}:all_owners_live_after_work")
                    finally:
                        mark(f"cycle{ordinal}:contexts_release"); start = time.perf_counter_ns()
                        for context in reversed(contexts):
                            client.release_execution_context(pb.ExecutionContextRef(context_id=context))
                        release_ns = time.perf_counter_ns() - start
                        snapshot(f"cycle{ordinal}:all_contexts_released")
                    return {"ordinal": ordinal, "selected_ids_by_owner": chosen, "steps": observations,
                            "create_contexts_ns": create_ns if args.mode == "timing" else None,
                            "release_contexts_ns": release_ns if args.mode == "timing" else None}
                if args.mode != "proof":
                    collect_repeatable_cycles(receipt, cycle,
                        plan["warmup_cycles"] if args.mode == "timing" else 0,
                        plan["measured_cycles"] if args.mode == "timing" else 1)
                snapshot("before_model_retirement")
                mark("owner_retirement")
            snapshot("all_owners_released")
            if sampler:
                initial = receipt["resource_snapshots"][0]["snapshot"]["memory"]
                final = receipt["resource_snapshots"][-1]["snapshot"]["memory"]
                baseline.require(not initial.get("truncated", False) and not final.get("truncated", False), "known inventory truncated")
                baseline.require(initial.get("allocations", []) == final.get("allocations", []), "known allocations did not return to baseline")
                receipt["known_ownership_cleanup"] = {"allocations_returned_to_baseline": True, "whole_driver_leak_qualified": False}
        receipt["status"] = "PROOF_CAPTURED" if args.mode == "proof" else "PASS"
    finally:
        finish_observers(sampler, nvml, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    for name in ("source-root", "package", "graph", "reference", "library", "bridge", "plan", "tmpdir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--model", choices=("smollm2-135m", "qwen3-0.6b"))
    parser.add_argument("--profile", choices=("full", "inference"), default="full")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl"), default="cpu")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--contexts", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--mode", choices=("timing", "memory", "proof"), default="memory")
    parser.add_argument("--admission-proof", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); baseline.require(not args.out.exists(), "refusing to overwrite evidence")
    if args.prepare:
        baseline.require(all((args.source_root, args.package, args.graph, args.reference, args.library, args.bridge, args.model)), "prepare needs model/artifact/reference paths")
        prepare(args); return 0
    baseline.require(args.plan and args.tmpdir and args.tmpdir.is_dir() and not any(args.tmpdir.iterdir()), "worker needs plan and empty task-owned TMPDIR")
    os.environ["TMPDIR"] = str(args.tmpdir.resolve())
    plan = json.loads(args.plan.read_text()); baseline.require(plan["helper"] == baseline.identity(__file__), "capacity helper differs")
    before = baseline.verify_pins(plan["pins"])
    system_before = baseline.system_observation()
    receipt = {"format": "volvox-native-capacity-observation/v1", "status": "FAIL", "plan": baseline.identity(args.plan),
               "host": platform.node(), "pid": os.getpid(), "mode": args.mode, "cycles": [], "system_before": system_before,
               "identities_before": before, "whole_peak_qualified": False, "service_actions": 0}
    try:
        execute(args, plan, receipt)
    except BaseException as error:
        receipt["status"] = "FAIL"; receipt["error"] = {"message": str(error), "traceback": traceback.format_exc()}
    finally:
        try:
            receipt["identities_after"] = baseline.verify_pins(plan["pins"])
        except BaseException as error:
            receipt["status"] = "FAIL"; receipt["identity_error"] = str(error)
        receipt["system_after"] = baseline.system_observation()
        receipt["system_swap_delta"] = {name: receipt["system_after"][name] - system_before[name] for name in ("swap_pages_in", "swap_pages_out")}
        if any(receipt["system_swap_delta"].values()):
            receipt["status"] = "FAIL"; receipt["performance_invalidated_by_swap"] = True
        receipt["private_tmp_remaining"] = [{"path": str(p), "bytes": p.stat().st_size} for p in args.tmpdir.iterdir() if p.is_file()]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": receipt["status"], "out": str(args.out)}))
    return 0 if receipt["status"] in ("PASS", "PROOF_CAPTURED", "SKIPPED_CONSERVATIVE_RESOURCE_BOUND") else 1


if __name__ == "__main__":
    raise SystemExit(main())
