#!/usr/bin/env python3
"""Measure pinned native decoder calls; keep timing, memory and trace runs separate.

--prepare writes identities before any model measurement. A worker accepts that
receipt via --pins and runs one model/profile/backend in a fresh process. The
caller launches the policy's three timing workers serially, plus memory/trace
workers; existing services are never modified. No logits or KV are exported.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import gzip
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import threading
import time
import traceback

POLICY = Path(__file__).with_name("native_measurement.json")
WORKLOADS = ("cached_execute", "full_prefix_generate")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_policy(path=POLICY, workload="cached_execute"):
    """Select an explicit workload; shared observations do not imply shared inputs."""
    catalog = json.loads(Path(path).read_text())
    require(catalog.get("format") == "volvox-native-llm-baseline-policy/v2", "unsupported policy")
    require(workload in WORKLOADS and workload in catalog["workloads"], "unplanned workload")
    shared, selected = catalog["shared"], catalog["workloads"][workload]
    require(not shared.keys() & selected.keys(), "shared and workload policy fields overlap")
    require(not {"format", "workload"} & (shared.keys() | selected.keys()), "reserved policy fields")
    return {"format": catalog["format"], **shared, **selected, "workload": workload}


def identity(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def cpu_ticks():
    return list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))


def edge_samples():
    rows = []
    for _ in range(3):
        before = cpu_ticks(); start = time.monotonic_ns(); time.sleep(1); after = cpu_ticks()
        delta = [b - a for a, b in zip(before, after)]; total = sum(delta)
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
        devices = []
        for line in gpu.splitlines():
            uuid, name, utilization, memory = [v.strip() for v in line.split(",")]
            devices.append({"uuid": uuid, "name": name, "utilization_percent": int(utilization), "device_used_mib": int(memory)})
        rows.append({"start_monotonic_ns": start, "end_monotonic_ns": time.monotonic_ns(),
            "cpu_busy_percent": 100 * (total - delta[3] - delta[4]) / total, "cpu_ticks_before": before,
            "cpu_ticks_after": after, "gpu_devices": devices})
    return rows


def storage_inventory(graph_path, weight_path):
    """Count file bytes, tensor payload and graph aliases without reading payload."""
    stat = weight_path.stat()
    with weight_path.open("rb") as stream:
        header_bytes = struct.unpack("<Q", stream.read(8))[0]
        require(header_bytes <= 16 << 20, "unexpected safetensors header extent")
        header = json.loads(stream.read(header_bytes))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    ranges = sorted({tuple(value["data_offsets"]) for value in tensors.values()})
    unique = end = 0
    for begin, finish in ranges:
        require(0 <= begin <= finish <= stat.st_size - header_bytes - 8, "weight payload range differs")
        unique += max(0, finish - max(begin, end))
        end = max(end, finish)
    graph = json.loads(graph_path.read_text())
    references = {}
    for node in graph["nodes"]:
        for value in node.get("inputs", {}).values():
            if isinstance(value, str) and value in tensors:
                references[value] = references.get(value, 0) + 1
    return {"source_file_logical_bytes": stat.st_size, "source_file_blocks_bytes": stat.st_blocks * 512,
            "source_file_device": stat.st_dev, "source_file_inode": stat.st_ino,
            "safetensors_header_bytes": header_bytes + 8, "named_tensor_count": len(tensors),
            "tensor_payload_bytes_by_dtype": {dtype: sum(v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values() if v["dtype"] == dtype)
                                              for dtype in sorted({v["dtype"] for v in tensors.values()})},
            "named_tensor_payload_bytes": sum(v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values()),
            "unique_payload_bytes": unique, "unique_payload_ranges": len(ranges),
            "aliased_named_tensor_ranges": len(tensors) - len(ranges),
            "graph_weight_reference_count": sum(references.values()),
            "graph_unique_referenced_weights": len(references),
            "graph_unreferenced_tensor_count": len(tensors.keys() - references.keys()),
            "graph_unreferenced_tensor_payload_bytes": sum(v["data_offsets"][1] - v["data_offsets"][0]
                                                          for name, v in tensors.items() if name not in references),
            "graph_unreferenced_tensor_names": sorted(tensors.keys() - references.keys()),
            "graph_repeated_weight_references": sum(max(0, count - 1) for count in references.values()),
            "graph_shared_weight_references": {name: count for name, count in references.items() if count > 1},
            "interpretation": "One preserved F32 file, payload ranges and graph references overlap; do not sum them or private snapshots with engine memory."}


def source_identity(root):
    result = {}
    suffixes = {".c", ".h", ".inc", ".m", ".cu", ".cuh", ".cpp", ".hpp", ".cc", ".S", ".s", ".wgsl", ".proto", ".py", ".rs", ".mjs", ".ts"}
    for relative in ("native/src", "native/include", "native/cli", "native/third_party", "proto", "shaders", "python/volvoxai", "runtime/generated/python", "runtime/generated/c",
                     "tools/metal_shader_compiler/src", "tools/generated"):
        directory = root / relative
        require(directory.is_dir(), f"missing source directory: {relative}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                result[path.relative_to(root).as_posix()] = identity(path)["sha256"]
    for path in sorted((root / "native").glob("*")):
        if path.is_file() and path.suffix in suffixes:
            result[path.relative_to(root).as_posix()] = identity(path)["sha256"]
    build_tools = sorted({p for pattern in ("generate_*.py", "generate_*.mjs", "build_wasm_*.mjs", "*wasm*manifest*.mjs")
                          for p in (root / "tools").glob(pattern)})
    for path in build_tools:
        result[path.relative_to(root).as_posix()] = identity(path)["sha256"]
    for relative in ("native/CMakeLists.txt", "Makefile", "tools/release_profiles.mjs", "tools/compile_shaders.sh",
                     "tools/pack_native_shaders.py", "tools/wasm_host_imports.mjs", "tools/metal_shader_compiler/Cargo.toml",
                     "tools/metal_shader_compiler/Cargo.lock", "package.json", "package-lock.json"):
        result[relative] = identity(root / relative)["sha256"]
    return result


def prepared_inputs(manifest, reference, policy):
    if policy.get("workload") == "full_prefix_generate":
        variants = {v["name"]: v for v in policy["selection_variants"]}
        cases = [c for c in manifest["generation_cases"] if c["selection"]["name"] in variants]
        require(manifest["steps"] == policy["generation_steps"] and len(cases) == 6, "generation reference scope differs")
        result = []
        for case in cases:
            require(case["selection"] == variants[case["selection"]["name"]], "generation selection policy differs")
            payload = identity(reference / Path(case["payload"]["path"]).name)
            require(payload["sha256"] == case["payload"]["sha256"], "HF generation payload changed")
            data = struct.pack("<" + "i" * len(case["input_ids"]), *case["input_ids"])
            result.append({"id": case["name"], "prefix_length": len(case["input_ids"]), "initial_ids": case["input_ids"],
                           "selection": case["selection"], "expected_ids": case["ids"], "input_ids_sha256": hashlib.sha256(data).hexdigest(),
                           "reference_payload": payload})
        return result
    cases = manifest["cases"]
    expected = {f"prompt{i}-prefix{n:03d}" for i in policy["prompt_indices"] for n in policy["prefix_lengths"]}
    require(len(cases) == len(expected) and {c["id"] for c in cases} == expected, "reference case set differs")
    result = []
    for case in cases:
        initial = case["input_token_ids"]
        require(case["prefix_length"] in policy["prefix_lengths"] and len(initial) == case["prefix_length"], "prefix differs")
        require(len(case["greedy_ids"]) == policy["cached_decode_steps"], "greedy sequence extent differs")
        data = struct.pack("<" + "i" * len(initial), *initial)
        pin = identity(reference / case["engine_input_ids"]["path"])
        require(pin["sha256"] == case["engine_input_ids"]["sha256"] == hashlib.sha256(data).hexdigest(), "reference I32 input differs")
        result.append({"id": case["id"], "prefix_length": case["prefix_length"],
                       "initial_ids": initial, "greedy_ids": case["greedy_ids"], "input_file": pin})
    return result


def prepare(args, policy):
    root = args.source_root.resolve()
    graph_path = args.graph or args.package / "graph.json"
    weight_path = args.package / "model.safetensors"
    reference_path = args.reference / "manifest.json"
    fixed = policy["models"][args.model]
    inputs = {"harness": identity(__file__), "policy": identity(args.policy),
              "models_policy": identity(root / policy["models_policy"]), "library": identity(args.library),
              "bridge": identity(args.bridge), "graph": identity(graph_path), "weights": identity(weight_path),
              "reference": identity(reference_path)}
    require(inputs["weights"]["bytes"] == fixed["weights_bytes"] and inputs["weights"]["sha256"] == fixed["weights_sha256"], "preserved F32 weights differ")
    require(inputs["graph"]["sha256"] == fixed["execution_graph_sha256"], "preserved execution graph differs")
    require(inputs["reference"]["sha256"] == fixed["reference_manifest_sha256"], "reference manifest differs")
    graph = json.loads(graph_path.read_text())
    require(graph["dimensions"]["S"] == {"min": 1, "max": policy["max_context"]}, "graph context bound differs")
    if policy.get("workload") == "full_prefix_generate":
        require(graph["execution"]["generation"]["full_prefix"] is not None, "graph is not the fixed full-prefix generator")
    else:
        require(graph["outputs"] == ["logits", "generated"] and "incremental" in graph["execution"], "graph is not the fixed cached decoder")
    manifest = json.loads(reference_path.read_text())
    model_policy = json.loads((root / policy["models_policy"]).read_text())["models"][args.model]
    require(manifest["model"] == args.model and manifest["revision"] == model_policy["revision"], "reference model/revision differs")
    return {"format": "volvox-native-llm-baseline-pins/v1", "source_root": str(root),
            "model": args.model, "profile": args.profile, "backend": args.backend,
            "workload": policy["workload"],
            "inputs": inputs, "source": source_identity(root),
            "storage_inventory": storage_inventory(graph_path, weight_path),
            "original_weights_reference": {"dtype": model_policy["source_dtype"], "files": model_policy["weights"],
                "verification_scope": "Original BF16 manifest identity; original file is not duplicated on the measurement worker."},
            "cases": prepared_inputs(manifest, args.reference, policy), "config": manifest["config"]}


def verify_pins(pins):
    current = {name: identity(value["path"]) for name, value in pins["inputs"].items()}
    require(current == pins["inputs"], "pinned artifact/input changed")
    require(source_identity(Path(pins["source_root"])) == pins["source"], "pinned source closure changed")
    for case in pins["cases"]:
        if "input_file" in case:
            require(identity(case["input_file"]["path"]) == case["input_file"], "pinned input bytes changed")
        if "reference_payload" in case:
            require(identity(case["reference_payload"]["path"]) == case["reference_payload"], "pinned HF generation payload changed")
    return current


def proc_status():
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmRSS", "VmHWM", "VmSwap"):
            values[name] = int(value.split()[0]) * 1024
    return values


def system_observation():
    counters = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    memory = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.split(":", 1)[0] in ("MemAvailable", "SwapTotal", "SwapFree")}
    return {"monotonic_ns": time.perf_counter_ns(), "loadavg": Path("/proc/loadavg").read_text().strip(),
            "aggregate_cpu_jiffies": [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]],
            "swap_pages_in": int(counters["pswpin"]), "swap_pages_out": int(counters["pswpout"]), **memory}


class NvmlSampler:
    """Driver process/device samples; separate from public graph allocation counts."""
    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    class Process(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                    ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        require(self.lib.nvmlInit_v2() == 0, "NVML initialization failed")
        self.lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Memory)]
        self.processes = self.lib.nvmlDeviceGetComputeRunningProcesses_v2
        self.processes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(self.Process)]
        self.graphics_processes = self.lib.nvmlDeviceGetGraphicsRunningProcesses_v2
        self.graphics_processes.argtypes = self.processes.argtypes
        count = ctypes.c_uint()
        require(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0, "NVML device count failed")
        self.devices = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            require(self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) == 0, "NVML device handle failed")
            self.devices.append(handle)

    def sample(self):
        rows = []
        for index, handle in enumerate(self.devices):
            memory = self.Memory()
            require(self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0, "NVML device memory query failed")
            memberships = {}
            for kind, query in (("compute", self.processes), ("graphics", self.graphics_processes)):
                count = ctypes.c_uint()
                status = query(handle, ctypes.byref(count), None)
                require(status in (0, 7), "NVML process count query failed")
                capacity = count.value + 16
                values = (self.Process * capacity)()
                count.value = capacity
                require(query(handle, ctypes.byref(count), values) == 0, "NVML process memory query failed")
                memberships[kind] = [{"pid": p.pid, "used_bytes": None if p.usedGpuMemory == (1 << 64) - 1 else p.usedGpuMemory}
                                     for p in values[:count.value]]
            # A PID can appear in both lists; these are overlapping views.
            selected = [p["used_bytes"] for rows_ in memberships.values() for p in rows_
                        if p["pid"] == os.getpid() and p["used_bytes"] is not None]
            rows.append({"device_index": index, "total_bytes": memory.total, "free_bytes": memory.free,
                         "used_bytes": memory.used, "compute_processes": memberships["compute"],
                         "graphics_processes": memberships["graphics"],
                         "worker_process_used_bytes": max(selected, default=None)})
        return rows

    def close(self):
        self.lib.nvmlShutdown()


class RssSampler:
    def __init__(self, interval_ns, directory, nvml=None):
        self.interval = interval_ns / 1e9
        self.directory = directory
        self.nvml = nvml
        self.phase = "setup"
        self.samples = []
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        try:
            while not self.done.is_set():
                row = {"monotonic_ns": time.perf_counter_ns(), "phase": self.phase, **proc_status()}
                for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
                    name, _, value = line.partition(":")
                    if name in ("Pss", "Private_Clean", "Private_Dirty", "SwapPss"):
                        row[name] = int(value.split()[0]) * 1024
                if self.nvml:
                    row["nvml"] = self.nvml.sample()
                logical = blocks = 0
                for path in self.directory.glob("volvoxai-*"):
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        continue  # Normal retirement between directory listing and stat.
                    if path.is_file():
                        logical += stat.st_size
                        blocks += stat.st_blocks * 512
                row["private_snapshot_logical_bytes"] = logical
                row["private_snapshot_blocks_bytes"] = blocks
                self.samples.append(row)
                self.done.wait(self.interval)
        except BaseException as error:
            self.error = error

    def stop(self):
        self.done.set()
        self.thread.join()


def route(report, pb, backend):
    require(report.status == pb.NativeStatus.NATIVE_STATUS_OK and report.backend == backend, "required backend operation failed")
    value = report.route
    require(value and value.builtin and value.attested and value.active_nodes > 0 and
            value.selected_nodes == value.active_nodes and value.fallback_nodes == 0 and value.missing_nodes == 0,
            "route is not fully attested")
    require(not report.fallback or not report.fallback.operator_fallback_used, "operator fallback used")
    return {"backend": report.backend, "provider": value.provider, "active": value.active_nodes,
            "selected": value.selected_nodes, "digest": value.digest}


def distribution(values):
    ordered = sorted(values)
    return {"count": len(values), "unit": "nanoseconds", "min": min(values), "max": max(values),
            **{f"p{q}": ordered[math.ceil(q / 100 * len(ordered)) - 1] for q in (50, 95, 99)}}


def worker(args, policy, pins, receipt):
    root = Path(pins["source_root"])
    sys.path[:0] = [str(root / "python"), str(root / "runtime/generated/python")]
    vx = importlib.import_module("volvoxai")
    from volvoxai._diagnostics import Trace, _proto_json
    pb = vx.pb
    phase = "setup"
    nvml = NvmlSampler() if args.mode == "memory" and args.backend != "cpu" else None
    if nvml:
        receipt["nvml_before_library"] = nvml.sample()
    sampler = RssSampler(policy["memory_rss_sample_interval_ns"], args.tmpdir, nvml) if args.mode == "memory" else None
    if sampler:
        sampler.thread.start()
    receipt["process_before_library"] = proc_status()
    receipt["system_before_library"] = system_observation()
    cleanup = []
    trace = None
    trace_baseline = None
    snapshot_client = None
    runtime_id = 0

    def mark(name):
        nonlocal phase
        phase = name
        if sampler:
            sampler.phase = name
            receipt.setdefault("phase_markers", []).append({"phase": name, "monotonic_ns": time.perf_counter_ns()})

    def snapshot(name, runtime_scope=False):
        if args.mode == "memory" and snapshot_client is not None:
            mark("snapshot:" + name)
            observed = snapshot_client.get_resource_snapshot(pb.GetResourceSnapshotRequest(
                runtime_id=runtime_id if runtime_scope else None,
                module=None if runtime_scope else pb.Empty(), include_device=args.backend != "cpu", max_allocations=65536))
            receipt["resource_snapshots"].append({"phase": name, "snapshot": _proto_json(observed.snapshot)})

    def timed(name, action):
        mark(name)
        start = time.perf_counter_ns()
        value = action()
        duration = time.perf_counter_ns() - start
        # Instrumented durations are deliberately excluded from timing output.
        if args.mode == "timing":
            receipt["cold"][name] = duration
        snapshot(name)
        return value

    def start_trace(host, name):
        nonlocal trace, trace_baseline
        require(trace is None, "trace segments must not overlap")
        trace = Trace(host, runtime_id, pb.TraceOptions(detail=pb.TraceDetail.TRACE_DETAIL_NODES,
            memory=True, device_timing=False, utilization=False, execution_plans=False,
            capacity_bytes=policy["trace_capacity_bytes"], sample_interval_ns=policy["trace_sample_interval_ns"]), cleanup)
        trace.__enter__()
        observed = snapshot_client.get_resource_snapshot(pb.GetResourceSnapshotRequest(
            runtime_id=runtime_id, include_device=args.backend != "cpu", max_allocations=65536))
        trace_baseline = {"name": name, "start_resource_snapshot": _proto_json(observed.snapshot)}

    def finish_trace(require_cuda_graph=True):
        nonlocal trace, trace_baseline
        current, baseline = trace, trace_baseline
        require(current is not None, "no active trace segment")
        try:
            current.stop()
            observed = snapshot_client.get_resource_snapshot(pb.GetResourceSnapshotRequest(
                runtime_id=runtime_id if require_cuda_graph else None,
                module=None if require_cuda_graph else pb.Empty(),
                include_device=args.backend != "cpu", max_allocations=65536))
            segment = {**baseline, "info": _proto_json(current.info),
                "end_resource_snapshot": _proto_json(observed.snapshot),
                "copy_wait_summary": _proto_json(current.summary("activity")),
                "copy_wait_events": sum(e.activity in (pb.TraceActivity.TRACE_ACTIVITY_COPY,
                    pb.TraceActivity.TRACE_ACTIVITY_SUBMIT, pb.TraceActivity.TRACE_ACTIVITY_SYNCHRONIZE,
                    pb.TraceActivity.TRACE_ACTIVITY_COMPLETION) for e in current.events),
                "accounting_scope": "Visited execution owners seed existing requested capacity; peak bytes belong to this segment and its declared allocator inventory. Segment or allocator peaks must not be added into whole-model or physical memory.",
                "retirement_only_allocator_coverage": "Unvisited/unknown: a new retirement-only trace does not reattach earlier allocator observers. Empty allocator rows do not measure zero; independent inventory snapshots describe known reachable owners."
                    if not require_cuda_graph else None}
            segments = receipt.setdefault("trace", {"segments": [], "scope": policy["trace_collection_scope"]})["segments"]
            raw_path = args.out.with_name(f"{args.out.stem}.trace-{len(segments) + 1:03d}.json.gz")
            with raw_path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    stream.write('{"format":"volvox-native-trace-segment/v1","name":')
                    json.dump(baseline["name"], stream)
                    stream.write(',"events":[')
                    for index, event in enumerate(current.events):
                        if index:
                            stream.write(',')
                        json.dump(_proto_json(event), stream, allow_nan=False, separators=(',', ':'))
                    stream.write('],"resource_snapshots":[')
                    for index, observed in enumerate(current.resource_snapshots):
                        if index:
                            stream.write(',')
                        json.dump(_proto_json(observed), stream, allow_nan=False, separators=(',', ':'))
                    stream.write(']}\n')
            segment["raw_events_and_snapshots_gzip"] = identity(raw_path)
            segments.append(segment)
            require(current.info.events.dropped == 0 and current.info.resource_snapshots.dropped == 0,
                    "trace segment collector lost events or resource snapshots")
            require(all(a.accounting_complete and a.dropped_events == 0 for a in current.info.allocators),
                    "trace segment tracked allocator accounting or event coverage is incomplete")
            if args.backend == "cuda" and require_cuda_graph:
                selected = [a for a in current.info.allocators if a.allocator == "cuda.graph" and a.backend == "cuda"]
                require(len(selected) == 1 and selected[0].peak_bytes > 0,
                        "CUDA graph allocation peak was not observed in this work segment")
                segment["selected_cuda_graph_allocator"] = _proto_json(selected[0])
        finally:
            current.release()
            trace = trace_baseline = None

    try:
        with vx.open_library(pins["inputs"]["library"]["path"], loader=pins["inputs"]["bridge"]["path"]) as host:
            engine = vx.VxInferenceServiceClient(host)
            snapshot_client = vx.VxProfilingServiceClient(host)
            platform_info = vx.VxPlatformServiceClient(host).get_platform_info(pb.Empty())
            expected = pb.BuildProfile.BUILD_PROFILE_FULL if args.profile == "full" else pb.BuildProfile.BUILD_PROFILE_INFERENCE
            require(platform_info.build_profile == expected, "library profile differs")
            receipt["platform"] = _proto_json(platform_info)
            snapshot("baseline_before_runtime_create")
            with ExitStack() as owners:
                runtime = timed("runtime_create", lambda: engine.create_runtime(pb.CreateRuntimeRequest(cpu_threads=policy["cpu_threads"])))
                runtime_id = runtime.runtime_id
                owners.callback(engine.release_runtime, pb.RuntimeRef(runtime_id=runtime_id))
                snapshot("baseline")
                if args.mode == "trace":
                    start_trace(host, "load_compile_first_cycle")
                model = timed("load", lambda: engine.load_model(pb.LoadModelRequest(runtime_id=runtime_id,
                    graph_path=pins["inputs"]["graph"]["path"], weight_paths=[pins["inputs"]["weights"]["path"]])))
                owners.callback(engine.release_model, pb.ModelRef(model_id=model.model_id))
                compiled = timed("compile", lambda: engine.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=[args.backend],
                        operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID))))
                owners.callback(engine.release_compiled_model, pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                receipt["compile_route"] = route(compiled.report, pb, args.backend)
                require(compiled.report.compilation.policy_mode == pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE and
                        compiled.report.compilation.operator_fallback == pb.OperatorFallback.OPERATOR_FALLBACK_FORBID, "compile policy differs")
                capacity = policy["max_context"]
                layers = pins["config"]["num_hidden_layers"]
                caches = [f"layer.{i}.{suffix}" for i in range(layers) for suffix in ("k_rope", "v")]

                def cached_cycle(case, ordinal):
                    label = f"{case['id']}:cycle{ordinal}"
                    mark(label + ":context_create")
                    create_start = time.perf_counter_ns()
                    context = engine.create_execution_context(pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id,
                        graph=pb.GraphExecution(slots=1, require_incremental=True, row_mode=pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED),
                        limits=pb.ExecutionLimits(cache=pb.ContextCacheOptions(tensors=caches, page_tokens=2,
                            max_pages=capacity // 2, clear_on_recycle=True))))
                    create_ns = time.perf_counter_ns() - create_start
                    steps = []
                    tokens = list(case["initial_ids"]) + [0] * (capacity - case["prefix_length"])
                    chosen = []
                    try:
                        snapshot(label + ":context_created")
                        for step in range(policy["cached_decode_steps"] + 1):
                            length = case["prefix_length"] + step
                            if step:
                                tokens[length - 1] = case["greedy_ids"][step - 1]
                            mark(label + (":prefill" if step == 0 else f":decode{step}"))
                            values = [pb.Tensor(name="token_ids", dtype=pb.DataType.DATA_TYPE_I32, shape=[1, capacity],
                                                inline=struct.pack("<" + "i" * capacity, *tokens)),
                                      pb.Tensor(name="position_ids", dtype=pb.DataType.DATA_TYPE_I32, shape=[capacity],
                                                inline=struct.pack("<" + "i" * capacity, *range(capacity))),
                                      pb.Tensor(name="attention_mask", dtype=pb.DataType.DATA_TYPE_I32, shape=[1, capacity],
                                                inline=struct.pack("<" + "i" * capacity, *([1] * length + [0] * (capacity - length))))]
                            if step == 0:
                                values.append(pb.Tensor(name="active_lengths", dtype=pb.DataType.DATA_TYPE_I32, shape=[1], inline=struct.pack("<i", length)))
                            request = pb.ExecuteRequest(context_id=context.context_id,
                                inputs=pb.InputBindings(values=values, reuse=[] if step == 0 else ["active_lengths"]),
                                scope=pb.ExecutionScope(entrypoint="initialize", full=pb.Empty()) if step == 0 else
                                    pb.ExecutionScope(entrypoint="transition", slot_rows=pb.SlotRows(slots=[pb.SlotRow(position=length - 1)])),
                                outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE,
                                    selection=pb.TensorOutputSelection(names=policy["output"]["names"])))
                            start = time.perf_counter_ns()
                            packet = engine.execute(request)
                            duration = time.perf_counter_ns() - start
                            evidence = route(packet.report, pb, args.backend)
                            require(len(packet.outputs) == 1, "unexpected output selection")
                            output = packet.outputs[0]
                            require(output.name == "generated" and output.dtype == pb.DataType.DATA_TYPE_I32 and output.buffer is None,
                                    "timing output contains a numerical tensor or wrong storage")
                            require(len(output.inline) == capacity * 4, "generated I32 output extent differs")
                            selected = struct.unpack_from("<i", output.inline, (length - 1) * 4)[0]
                            if step < len(case["greedy_ids"]):
                                require(selected == case["greedy_ids"][step], "fixed greedy control differs")
                            chosen.append(selected)
                            steps.append({"phase": "prefill" if step == 0 else "decode", "step": step,
                                          "call_ns": duration if args.mode == "timing" else None, "route": evidence})
                            snapshot(label + f":completed{step}")
                        state = engine.get_execution_context_info(pb.ExecutionContextRef(context_id=context.context_id))
                        require(state.mode == pb.DecodeMode.DECODE_MODE_ROW, "cached request did not use ROW mode")
                    finally:
                        mark(label + ":context_release")
                        start = time.perf_counter_ns()
                        engine.release_execution_context(pb.ExecutionContextRef(context_id=context.context_id))
                        close_ns = time.perf_counter_ns() - start
                    snapshot(label + ":context_released")
                    return {"case": case["id"], "ordinal": ordinal, "context_create_ns": create_ns if args.mode == "timing" else None,
                            "context_release_ns": close_ns if args.mode == "timing" else None, "selected_ids": chosen,
                            "exact_reference_ids_checked": len(case["greedy_ids"]), "steps": steps}

                def generate_cycle(case, ordinal):
                    label = f"{case['id']}:cycle{ordinal}"
                    selection = case["selection"]
                    options = {k: v for k, v in selection.items() if k not in ("name", "sampling")}
                    mark(label + ":context_create")
                    begin = time.perf_counter_ns()
                    context = engine.create_execution_context(pb.CreateExecutionContextRequest(
                        compiled_model_id=compiled.compiled_model_id, autoregressive=pb.AutoregressiveExecution(
                            max_steps=policy["generation_steps"], sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY
                            if selection["sampling"] == "GREEDY" else pb.SamplingMode.SAMPLING_MODE_RANDOM, **options)))
                    create_ns = time.perf_counter_ns() - begin
                    try:
                        snapshot(label + ":context_created")
                        ids = case["initial_ids"]
                        count = len(ids)
                        values = [pb.Tensor(name="token_ids", dtype=pb.DataType.DATA_TYPE_I32, shape=[1, count],
                                    inline=struct.pack("<" + "i" * count, *ids)),
                                  pb.Tensor(name="position_ids", dtype=pb.DataType.DATA_TYPE_I32, shape=[count],
                                    inline=struct.pack("<" + "i" * count, *range(count))),
                                  pb.Tensor(name="attention_mask", dtype=pb.DataType.DATA_TYPE_I32, shape=[1, count],
                                    inline=struct.pack("<" + "i" * count, *([1] * count)))]
                        request = pb.GenerateRequest(context_id=context.context_id, initial_inputs=pb.GenerationInputs(values=values),
                            budget=pb.GenerationBudget(max_steps=policy["generation_steps"], max_output_bytes=policy["generation_output_byte_budget"]),
                            outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE))
                        mark(label + ":generate")
                        begin = time.perf_counter_ns()
                        packet = engine.generate(request)
                        duration = time.perf_counter_ns() - begin
                        evidence = route(packet.report, pb, args.backend)
                        require(len(packet.outputs) == 1 and packet.outputs[0].name == "generated_ids" and packet.outputs[0].dtype == pb.DataType.DATA_TYPE_I32,
                                "Generate output differs from I32-only selection")
                        output = packet.outputs[0]
                        require(output.buffer is None and len(output.inline) == policy["generation_steps"] * 4,
                                "Generate did not return all fixed INLINE IDs")
                        chosen = list(struct.unpack("<" + "i" * policy["generation_steps"], output.inline))
                        require(chosen == case["expected_ids"], "Generate fixed HF control IDs differ")
                        snapshot(label + ":completed")
                    finally:
                        mark(label + ":context_release")
                        begin = time.perf_counter_ns()
                        engine.release_execution_context(pb.ExecutionContextRef(context_id=context.context_id))
                        close_ns = time.perf_counter_ns() - begin
                    snapshot(label + ":context_released")
                    return {"case": case["id"], "ordinal": ordinal, "selection": selection, "selected_ids": chosen,
                            "context_create_ns": create_ns if args.mode == "timing" else None,
                            "context_release_ns": close_ns if args.mode == "timing" else None,
                            "exact_reference_ids_checked": len(chosen), "steps": [{"phase": "generate", "step": 0,
                                "generated_ids": len(chosen), "call_ns": duration if args.mode == "timing" else None, "route": evidence}]}

                cycle = generate_cycle if policy["workload"] == "full_prefix_generate" else cached_cycle

                first = cycle(pins["cases"][0], "first")
                receipt["first"] = first
                if args.mode == "trace":
                    finish_trace()
                if args.mode == "timing":
                    receipt["cold"]["first_" + first["steps"][0]["phase"] + "_call"] = first["steps"][0]["call_ns"]
                    if first["steps"][0]["phase"] == "prefill":
                        receipt["cold"]["first_decode_calls"] = [step["call_ns"] for step in first["steps"][1:]]
                if args.mode == "timing":
                    for case in pins["cases"]:
                        for ordinal in range(-policy["warmup_cycles_per_case"], policy["measured_cycles_per_case"]):
                            value = cycle(case, ordinal)
                            if ordinal >= 0:
                                receipt["cycles"].append(value)
                else:
                    for case in pins["cases"]:
                        if args.mode == "trace":
                            start_trace(host, "case:" + case["id"])
                        receipt["cycles"].append(cycle(case, 0))
                        if args.mode == "trace":
                            finish_trace()
                if args.mode == "trace":
                    start_trace(host, "owner_retirement")
                mark("model_owners_release")
                snapshot("runtime_before_owner_retirement", runtime_scope=True)
            snapshot("all_runtime_model_context_owners_released")
            if args.mode == "memory":
                snapshots = {value["phase"]: value["snapshot"]["memory"] for value in receipt["resource_snapshots"]}
                baseline = snapshots["baseline_before_runtime_create"]
                final = snapshots["all_runtime_model_context_owners_released"]
                require(not baseline.get("truncated", False) and not final.get("truncated", False), "cleanup inventory is truncated")
                require(baseline.get("inventory") == final.get("inventory"), "cleanup inventory coverage changed")
                rows = lambda value: sorted(value.get("allocations", []), key=lambda row: json.dumps(row, sort_keys=True))
                require(rows(baseline) == rows(final), "known module allocation inventory did not return to baseline")
                runtime_memory = snapshots["runtime_before_owner_retirement"]
                require(int(runtime_memory.get("unconsumedResultBytes", -1)) == 0,
                        "runtime result accounting is unknown or unconsumed results remained before retirement")
                receipt["known_ownership_cleanup"] = {"module_allocations_returned_to_baseline": True,
                    "module_inventory": final.get("inventory"), "module_inventory_truncated": False,
                    "runtime_unconsumed_result_bytes_before_retirement": 0,
                    "module_unconsumed_result_bytes_after_retirement": final.get("unconsumedResultBytes"),
                    "whole_process_or_driver_leak_qualified": False,
                    "scope": "Known reachable module allocations and runtime unconsumed-result accounting; partial inventories retain unknown owners."}
            if args.mode == "trace":
                finish_trace(require_cuda_graph=False)
                require(len(receipt["trace"]["segments"]) == len(pins["cases"]) + 2, "trace segment coverage differs")
        receipt["status"] = "observed"
    finally:
        if sampler:
            sampler.stop()
            receipt["rss_samples"] = sampler.samples
            receipt["rss_sampled_max_bytes"] = max((s.get("VmRSS", 0) for s in sampler.samples), default=0)
            receipt["pss_sampled_max_bytes"] = max((s.get("Pss", 0) for s in sampler.samples), default=0)
            receipt["maximum_memory_sample_gap_ns"] = max((b["monotonic_ns"] - a["monotonic_ns"]
                for a, b in zip(sampler.samples, sampler.samples[1:])), default=None)
            receipt["rss_sampled_max_by_phase"] = {phase: max(s.get("VmRSS", 0) for s in sampler.samples if s["phase"] == phase)
                                                  for phase in sorted({s["phase"] for s in sampler.samples})}
            if sampler.error:
                receipt["sampler_error"] = str(sampler.error)
                receipt["status"] = "failed"
        if nvml:
            receipt["nvml_after_library_closed"] = nvml.sample()
            receipt["nvml_interval_sample_scope"] = "Compute/graphics process driver residency (max per PID, never sum overlapping lists) and device-wide memory; samples are lower bounds, include context/driver costs, and are not graph allocation counters."
            selected = [d["worker_process_used_bytes"] for row in sampler.samples for d in row["nvml"]
                        if d["worker_process_used_bytes"] is not None]
            receipt["nvml_worker_sampled_max_bytes"] = max(selected, default=None)
            nvml.close()
            require(selected, "NVML did not observe the GPU worker PID; use the host PID namespace for a container")
        receipt["process_lifetime_final"] = proc_status()
        receipt["system_final"] = system_observation()
        receipt["whole_peak_qualified"] = False
        receipt["idle_host_qualified"] = False
        receipt["performance_nonregression_qualified"] = False
        receipt["memory_nonregression_qualified"] = False
        if args.mode == "timing" and receipt["cycles"]:
            phases = sorted({step["phase"] for c in receipt["cycles"] for step in c["steps"]})
            receipt["timing_summary"] = {phase: distribution([step["call_ns"] for c in receipt["cycles"] for step in c["steps"] if step["phase"] == phase])
                                         for phase in phases}
            receipt["timing_by_case"] = {case: {
                phase: distribution([step["call_ns"] for c in receipt["cycles"] if c["case"] == case
                                     for step in c["steps"] if step["phase"] == phase])
                for phase in phases}
                for case in sorted({c["case"] for c in receipt["cycles"]})}
        if sampler and sampler.error:
            raise RuntimeError("RSS sampler failed") from sampler.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--workload", choices=WORKLOADS, default="cached_execute")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", required=True, help="model alias in the selected workload policy")
    parser.add_argument("--profile", choices=("full", "inference"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl"), required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--graph", type=Path, help="explicit graph override for full-prefix generation")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--mode", choices=("timing", "memory", "trace"), default="timing")
    parser.add_argument("--tmpdir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "refusing to overwrite evidence")
    policy = read_policy(args.policy, args.workload)
    require(args.model in policy["models"], "unplanned model")
    require({"profile": args.profile, "backend": args.backend} in policy["cells"], "unplanned backend/profile")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        require(all((args.source_root, args.package, args.reference, args.library, args.bridge)), "prepare needs source/package/reference/library/bridge")
        receipt = prepare(args, policy)
        with args.out.open("x") as stream:
            json.dump(receipt, stream, indent=2); stream.write("\n")
        print(json.dumps({"status": "prepared", "pins_sha256": identity(args.out)["sha256"]}))
        return
    require(args.pins and args.tmpdir and args.tmpdir.is_dir(), "worker needs pins and a private TMPDIR")
    require(not any(args.tmpdir.iterdir()), "private TMPDIR must start empty")
    os.environ["TMPDIR"] = str(args.tmpdir.resolve())
    pins = json.loads(args.pins.read_text())
    require(pins["format"] == "volvox-native-llm-baseline-pins/v1", "unsupported pins")
    require((pins["model"], pins["profile"], pins["backend"]) == (args.model, args.profile, args.backend), "worker cell differs from pins")
    require(pins.get("workload") == args.workload, "worker workload differs from pins")
    require(identity(args.policy)["sha256"] == pins["inputs"]["policy"]["sha256"], "worker policy differs")
    receipt = {"format": "volvox-native-llm-baseline-observation/v1", "status": "failed", "mode": args.mode,
        "model": args.model, "profile": args.profile, "backend": args.backend, "cpu_threads": policy["cpu_threads"],
        "workload": policy.get("workload", "cached_execute"),
        "warm_scope": policy["warm_scope"],
        "pins": identity(args.pins), "host": platform.node(), "pid": os.getpid(), "cold": {}, "cycles": [],
        "resource_snapshots": [], "service_actions": 0, "capture_logits_kv": False, "claims": policy["interpretation"],
        "prepared_storage_inventory": pins["storage_inventory"]}
    before = verify_pins(pins)
    receipt["identities_before"] = before
    receipt["source_closure_sha256"] = hashlib.sha256(json.dumps(pins["source"], sort_keys=True).encode()).hexdigest()
    failed = None
    try:
        worker(args, policy, pins, receipt)
    except BaseException as error:
        failed = error
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        try:
            receipt["identities_after"] = verify_pins(pins)
        except BaseException as error:
            receipt["status"] = "failed";receipt["identity_error"] = str(error);failed = failed or error
        receipt["private_tmp_remaining"] = [{"name": p.name, "bytes": p.stat().st_size} for p in args.tmpdir.iterdir() if p.is_file()]
        with args.out.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
        print(json.dumps({"status": receipt["status"], "mode": args.mode, "model": args.model,
                          "receipt_sha256": identity(args.out)["sha256"]}))
    if failed:
        raise failed


if __name__ == "__main__":
    main()
