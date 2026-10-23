#!/usr/bin/env python3
"""Measure completed CUDA Generate calls from an explicit benchmark manifest.

A model entry fixes graph/weight paths and PackedArithmetic. Cases fix exact
prompt IDs, generation budget and context capacity. Timing uses fresh execution
contexts, with cold/warmup/measured cycles kept in the report. GPU memory is
sampled in a separate replay so the observer does not perturb timed cycles.

The first Generate selects one token (TTFT); subsequent calls select either the
whole remainder or ``decode_chunk_size`` IDs. A grouped call provides throughput
and amortized token latency, not a distribution of individual token latencies.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "python" / "volvoxai").is_dir() else HERE
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "runtime/generated/python"))

# Load bindings only when executing a workload; manifest/statistics tests need
# neither a native library nor an installed Synurang Python module.
vx = pb = None


def load_bindings():
    global vx, pb
    import volvoxai
    vx, pb = volvoxai, volvoxai.pb


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return None
    if not np.isfinite(data).all():
        raise ValueError("benchmark measurements must be finite")
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def summarize_cycles(cycles, prompt_tokens):
    measured = [cycle for cycle in cycles if cycle["phase"] == "measured"]
    if not measured:
        raise ValueError("at least one measured cycle is required")
    def metric(key):
        return distribution([cycle[key] for cycle in measured])
    decode = [c for c in measured if c["decode_tokens"]]
    calls = [call for cycle in measured for call in cycle["decode_calls"]]
    all_single = bool(calls) and all(call["tokens"] == 1 for call in calls)
    return {
        "measured_cycles": len(measured),
        "ttft_ms": metric("ttft_ms"),
        "prefill_tokens_per_sec": distribution([
            prompt_tokens * 1000.0 / c["ttft_ms"] for c in measured]),
        "decode_ms": metric("decode_ms"),
        "decode_tokens_per_sec": distribution([
            c["decode_tokens"] * 1000.0 / c["decode_ms"] for c in decode]),
        "amortized_decode_ms_per_token": distribution([
            c["decode_ms"] / c["decode_tokens"] for c in decode]),
        "inter_token_ms": distribution([call["delivery_interval_ms"] for call in calls]) if all_single else None,
        "decode_call_ms": distribution([call["call_ms"] for call in calls]),
        "total_ms": metric("total_ms"),
        "end_to_end_tokens_per_sec": distribution([
            c["generated_tokens"] * 1000.0 / c["total_ms"] for c in measured]),
        "context_create_ms": metric("context_create_ms"),
        "greedy_tokens_stable": all(c["ids"] == measured[0]["ids"] for c in measured),
    }


def validate_manifest(config, base, selected=None):
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("manifest.models must be a nonempty array")
    names = [m["name"] for m in config["models"]]
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    if selected and set(selected) - set(names):
        raise ValueError(f"unknown requested models: {sorted(set(selected) - set(names))}")
    if not isinstance(config.get("cases"), dict) or not config["cases"]:
        raise ValueError("manifest.cases must be a nonempty object")
    for case_id, case in config["cases"].items():
        ids = case.get("initial_ids")
        if not isinstance(ids, list) or not ids or any(type(v) is not int or v < 0 or v >= 2**31 for v in ids):
            raise ValueError(f"{case_id}: initial_ids must be nonempty nonnegative I32 IDs")
        for key in ("budget", "capacity"):
            if type(case.get(key)) is not int or case[key] < 1:
                raise ValueError(f"{case_id}: {key} must be a positive integer")
        if "decode_chunk_size" in case and (type(case["decode_chunk_size"]) is not int or case["decode_chunk_size"] < 0):
            raise ValueError(f"{case_id}: decode_chunk_size must be an integer >= 0")
        if len(ids) + case["budget"] > case["capacity"]:
            raise ValueError(f"{case_id}: prompt plus generated tokens exceed capacity")
    protocol = {"cold_cycles": 1, "warmup_cycles": 2, "measured_cycles": 5,
                "decode_chunk_size": 0, "memory_interval_ms": 20, "cpu_threads": 2,
                **config.get("protocol", {})}
    for key in ("cold_cycles", "warmup_cycles", "measured_cycles", "decode_chunk_size", "cpu_threads", "memory_interval_ms"):
        minimum = 1 if key in ("measured_cycles", "cpu_threads", "memory_interval_ms") else 0
        if type(protocol[key]) is not int or protocol[key] < minimum:
            raise ValueError(f"protocol.{key} must be an integer >= {minimum}")
    models = []
    for entry in config["models"]:
        if selected and entry["name"] not in selected:
            continue
        model = dict(entry)
        if model.get("packed_arithmetic", "f32") not in ("f32", "f32-fast", "f16", "dynamic-i8"):
            raise ValueError(f"{model['name']}: invalid packed_arithmetic")
        model["packed_arithmetic"] = model.get("packed_arithmetic", "f32")
        if not isinstance(model.get("weight_paths"), list) or not model["weight_paths"]:
            raise ValueError(f"{model['name']}: weight_paths must be nonempty")
        def resolve(path):
            path = Path(path).expanduser()
            path = path if path.is_absolute() else base / path
            if not path.is_file():
                raise FileNotFoundError(path)
            return str(path.resolve())
        model["graph_path"] = resolve(model["graph_path"])
        model["weight_paths"] = [resolve(path) for path in model["weight_paths"]]
        models.append(model)
    return models, protocol


class _MemoryInfo(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class GpuMemorySampler:
    """Separate-replay sampled memory; never claim an exact allocation peak."""
    def __init__(self, interval_ms=20):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self.errors = []
        self.event = threading.Event()
        self.library = None
        self.source = "nvidia-smi"
        self.device = os.environ.get("VOLVOXAI_BENCH_GPU_INDEX", "0")
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                raise RuntimeError("nvmlInit_v2 failed")
            self.library = lib
            self.handle = ctypes.c_void_p()
            selector = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
            if selector.startswith("GPU-"):
                status = lib.nvmlDeviceGetHandleByUUID(selector.encode(), ctypes.byref(self.handle))
            else:
                index = int(selector) if selector else int(self.device)
                self.device = str(index)
                status = lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle))
            if status:
                raise RuntimeError(f"NVML GPU selection failed: {status}")
            self.source = "nvml"
        except (OSError, AttributeError, RuntimeError, ValueError) as exc:
            self.errors.append(str(exc))
            if self.library:
                self.library.nvmlShutdown()
            self.library = None
            self.interval = max(self.interval, .25)

    def sample(self):
        try:
            if self.library:
                info = _MemoryInfo()
                status = self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info))
                if status:
                    raise RuntimeError(f"nvmlDeviceGetMemoryInfo: {status}")
                device_mib = info.used / 2**20
                process_mib = None
                process_ids = None
                if hasattr(self.library, "nvmlDeviceGetComputeRunningProcesses_v2"):
                    count = ctypes.c_uint(128)
                    processes = (_ProcessInfo * count.value)()
                    status = self.library.nvmlDeviceGetComputeRunningProcesses_v2(self.handle, ctypes.byref(count), processes)
                    if status == 0:
                        process_ids = [int(processes[i].pid) for i in range(count.value)]
                        current = [processes[i].usedGpuMemory for i in range(count.value)
                                   if processes[i].pid == os.getpid()]
                        if current and all(value < 2**64 - 1 for value in current):
                            process_mib = sum(current) / 2**20
            else:
                output = subprocess.check_output([
                    "nvidia-smi", "-i", self.device, "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                device_mib = float(output.strip().splitlines()[0])
                process_mib = None
                process_ids = None
            self.samples.append({"monotonic_ns": time.perf_counter_ns(), "device_mib": device_mib,
                                 "process_mib": process_mib, "compute_process_ids": process_ids})
        except Exception as exc:
            message = str(exc)
            if message not in self.errors:
                self.errors.append(message)

    def _run(self):
        while not self.event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.event.set()
        self.thread.join(timeout=6)
        self.sample()
        if self.library:
            self.library.nvmlShutdown()

    def report(self):
        device = [s["device_mib"] for s in self.samples]
        process = [s["process_mib"] for s in self.samples if s["process_mib"] is not None]
        return {"method": "separate_replay_sampled_peak", "source": self.source,
                "requested_interval_ms": self.interval * 1000, "sample_count": len(self.samples),
                "device_peak_mib": max(device) if device else None,
                "process_peak_mib": max(process) if process else None,
                "device_before_mib": device[0] if device else None,
                "other_compute_process_ids": sorted({pid for s in self.samples
                    for pid in s["compute_process_ids"] or [] if pid != os.getpid()}),
                "samples": self.samples, "errors": self.errors}


def input_tensors(case):
    ids, capacity = case["initial_ids"], case["capacity"]
    names = {"token_ids": "token_ids", "position_ids": "position_ids",
             "attention_mask": "attention_mask", "active_lengths": "active_lengths",
             **case.get("input_names", {})}
    values = {"token_ids": np.zeros((1, capacity), np.int32),
              "position_ids": np.arange(capacity, dtype=np.int32),
              "attention_mask": np.zeros((1, capacity), np.int32),
              "active_lengths": np.array([len(ids)], np.int32)}
    values["token_ids"][0, :len(ids)] = ids
    values["attention_mask"][0, :len(ids)] = 1
    return [pb.Tensor(name=names[key], dtype=pb.DataType.DATA_TYPE_I32,
                      shape=value.shape, inline=value.tobytes()) for key, value in values.items()]


def route_evidence(report):
    route = report.route
    if (report.status != pb.NativeStatus.NATIVE_STATUS_OK or report.backend != "cuda"
            or not route or not route.builtin or not route.attested or route.provider != "cuda"
            or route.active_nodes <= 0 or route.selected_nodes != route.active_nodes
            or route.fallback_nodes or route.missing_nodes
            or (report.fallback and report.fallback.operator_fallback_used)):
        raise RuntimeError("Generate did not attest a complete CUDA route without fallback")
    return {"backend": report.backend, "device": report.device, "provider": route.provider,
            "attested": route.attested, "active_nodes": route.active_nodes,
            "selected_nodes": route.selected_nodes, "fallback_nodes": route.fallback_nodes,
            "missing_nodes": route.missing_nodes, "digest": route.digest}


def response_ids(response):
    outputs = [tensor for tensor in response.outputs if tensor.dtype == pb.DataType.DATA_TYPE_I32]
    if len(outputs) != 1:
        raise RuntimeError("Generate must return one I32 selected-ID tensor")
    ids = np.frombuffer(outputs[0].inline, dtype=np.int32).tolist()
    if len(ids) != response.progress.steps:
        raise RuntimeError("Generate output count differs from committed progress.steps")
    return ids


def run_cycle(client, compiled_id, case, tensors, chunk_size, phase):
    start = time.perf_counter_ns()
    context = client.create_execution_context(pb.CreateExecutionContextRequest(
        compiled_model_id=compiled_id, autoregressive=pb.AutoregressiveExecution(
            sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY, max_steps=case["budget"])))
    context_create_ms = (time.perf_counter_ns() - start) / 1e6
    try:
        first = pb.GenerateRequest(context_id=context.context_id,
            initial_inputs=pb.GenerationInputs(values=tensors), budget=pb.GenerationBudget(max_steps=1),
            outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE))
        start = time.perf_counter_ns()
        response = client.generate(first)
        completed = time.perf_counter_ns()
        generation_start, prefill_completed = start, completed
        ttft_ms = (completed - start) / 1e6
        prefill_route = route_evidence(response.report)
        ids = response_ids(response)
        if len(ids) != 1:
            raise RuntimeError("prefill did not generate exactly one token")
        calls = []
        while len(ids) < case["budget"]:
            if not response.progress.can_continue:
                raise RuntimeError(f"generation stopped early at {len(ids)}/{case['budget']} tokens")
            remaining = case["budget"] - len(ids)
            steps = min(remaining, chunk_size) if chunk_size else remaining
            request = pb.GenerateRequest(context_id=context.context_id,
                budget=pb.GenerationBudget(max_steps=steps),
                outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE))
            start = time.perf_counter_ns()
            response = client.generate(request)
            previous_completed, completed = completed, time.perf_counter_ns()
            call_ms = (completed - start) / 1e6
            delivery_interval_ms = (completed - previous_completed) / 1e6
            decode_route = route_evidence(response.report)
            selected = response_ids(response)
            if len(selected) != steps:
                raise RuntimeError(f"decode selected {len(selected)} IDs for requested {steps}")
            calls.append({"tokens": len(selected), "call_ms": call_ms,
                          "delivery_interval_ms": delivery_interval_ms, "route": decode_route})
            ids.extend(selected)
        decode_ms = (completed - prefill_completed) / 1e6
        return {"phase": phase, "context_create_ms": context_create_ms, "ttft_ms": ttft_ms,
                "decode_ms": decode_ms, "total_ms": (completed - generation_start) / 1e6,
                "generate_call_ms": ttft_ms + sum(call["call_ms"] for call in calls),
                "generated_tokens": len(ids), "decode_tokens": len(ids) - 1,
                "decode_calls": calls, "ids": ids, "prefill_route": prefill_route,
                "stop_reason": int(response.progress.stop_reason)}
    finally:
        client.release_execution_context(pb.ExecutionContextRef(context_id=context.context_id))


def benchmark_model(model, cases, protocol, on_progress=None):
    from volvoxai._library import open_library
    print(f"[{model['name']}] {model['packed_arithmetic']}", flush=True)
    arithmetic = {"f32": pb.PackedArithmetic.PACKED_ARITHMETIC_F32,
                  "f32-fast": pb.PackedArithmetic.PACKED_ARITHMETIC_F32_FAST,
                  "f16": pb.PackedArithmetic.PACKED_ARITHMETIC_F16,
                  "dynamic-i8": pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8}[model["packed_arithmetic"]]
    result = {"model": model, "cases": {}}
    with contextlib.ExitStack() as owners:
        host = open_library()
        owners.callback(host.close)
        client = vx.VxInferenceServiceClient(host)
        runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=protocol["cpu_threads"]))
        owners.callback(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
        start = time.perf_counter_ns()
        loaded = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
            graph_path=model["graph_path"], weight_paths=model["weight_paths"]))
        result["load_ms"] = (time.perf_counter_ns() - start) / 1e6
        owners.callback(client.release_model, pb.ModelRef(model_id=loaded.model_id))
        start = time.perf_counter_ns()
        compiled = client.compile_model(pb.CompileModelRequest(model_id=loaded.model_id,
            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=["cuda"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID),
            packed_arithmetic=arithmetic))
        result["compile_ms"] = (time.perf_counter_ns() - start) / 1e6
        owners.callback(client.release_compiled_model,
                        pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        result["resolved_packed_arithmetic"] = int(compiled.packed_arithmetic)
        result["compilation"] = {"backend": compiled.report.backend,
            "device": compiled.report.device,
            "tier_fallback_used": bool(compiled.report.compilation and compiled.report.compilation.tier_fallback_used)}
        if compiled.report.backend != "cuda" or result["compilation"]["tier_fallback_used"]:
            raise RuntimeError("compilation did not select CUDA without tier fallback")
        if compiled.packed_arithmetic != arithmetic:
            raise RuntimeError("resolved packed arithmetic does not match requested contract")
        if on_progress:
            on_progress(result)
        for case_id, case in cases.items():
            tensors = input_tensors(case)
            chunk_size = case.get("decode_chunk_size", protocol["decode_chunk_size"])
            phases = (["cold"] * protocol["cold_cycles"] + ["warmup"] * protocol["warmup_cycles"]
                      + ["measured"] * protocol["measured_cycles"])
            cycles = []
            for index, phase in enumerate(phases):
                cycle = run_cycle(client, compiled.compiled_model_id, case, tensors,
                                  chunk_size, phase)
                cycles.append(cycle)
                print(f"  {case_id} {phase} {index + 1}/{len(phases)}: TTFT {cycle['ttft_ms']:.1f} ms, total {cycle['total_ms']:.1f} ms", flush=True)
            # Observe memory after timed work, while the same model/executable
            # remains loaded, with a fresh context and identical request geometry.
            with GpuMemorySampler(protocol["memory_interval_ms"]) as sampler:
                memory_cycle = run_cycle(client, compiled.compiled_model_id, case, tensors,
                                         chunk_size, "memory")
            result["cases"][case_id] = {"prompt_tokens": len(case["initial_ids"]),
                "generation_budget": case["budget"], "context_capacity": case["capacity"], "decode_chunk_size": chunk_size,
                "summary": summarize_cycles(cycles, len(case["initial_ids"])),
                "cycles": cycles, "memory": sampler.report(), "memory_cycle": memory_cycle}
            if on_progress:
                on_progress(result)
    return result


def device_metadata():
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,pstate,power.limit,clocks.sm,clocks.mem",
               "--format=csv,noheader"]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        output = str(exc)
    return {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
            "nvidia_smi": output, "pid": os.getpid(),
            "environment": {name: os.environ.get(name) for name in
                ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "VOLVOXAI_LIBRARY", "OMP_NUM_THREADS")}}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", default="benchmark_summary.json", type=Path)
    parser.add_argument("--models", help="Comma-separated manifest model names; unknown/missing entries fail")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selected = {name.strip() for name in args.models.split(",") if name.strip()} if args.models else None
    models, protocol = validate_manifest(config, args.config.resolve().parent, selected)
    report = {"schema": "volvoxai-native-generate-benchmark/v2", "status": "RUNNING",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "config_sha256": sha256_file(args.config), "runner_sha256": sha256_file(__file__),
              "protocol": protocol, "cases": config["cases"], "device": device_metadata(),
              "measurement_contract": {
                  "ttft": "Completed first Generate(max_steps=1), including initial input transfer, prefill and ID selection; excludes tokenization, model load, compile and context creation",
                  "total": "Wall time from first Generate invocation through final response, including application gaps between continuation calls; context creation/release excluded",
                  "decode": "Committed generated tokens after the first token divided by wall time from first response to final response, including continuation request preparation",
                  "inter_token": "Response-to-response wall intervals, available only when every continuation Generate selects one token; includes API and caller overhead",
                  "memory": "Separate replay sampled GPU memory; sampling can miss short-lived allocations; device totals include other processes",
                  "cold": "First fresh-context cycle per case; process, OS file caches and CUDA code cache are not reset between cases/models",
                  "load": "Measured after artifact hashing, so OS cache may already be warm",
                  "percentiles": "NumPy linear percentiles across retained measured cycles/calls; descriptive, no confidence interval",
              }, "artifacts": {}, "results": {}, "failures": []}
    write_report(args.out, report)
    for model in models:
        try:
            if vx is None:
                load_bindings()
                from volvoxai._library import find_library
                library = find_library().resolve()
                report["engine_library"] = {"path": str(library), "bytes": library.stat().st_size,
                                            "sha256": sha256_file(library)}
                report["generated_api_sha256"] = sha256_file(pb.__file__)
            files = [model["graph_path"], *model["weight_paths"]]
            for file in files:
                if file not in report["artifacts"]:
                    report["artifacts"][file] = {"bytes": Path(file).stat().st_size, "sha256": sha256_file(file)}
            def persist_partial(partial):
                report["results"][model["name"]] = partial
                write_report(args.out, report)
            result = benchmark_model(model, config["cases"], protocol, on_progress=persist_partial)
            result["package_size_mib"] = sum(report["artifacts"][file]["bytes"] for file in files) / 2**20
            report["results"][model["name"]] = result
            write_report(args.out, report)
        except BaseException as exc:
            report["status"] = "FAILED"
            report["failures"].append({"model": model["name"], "type": type(exc).__name__, "message": str(exc)})
            write_report(args.out, report)
            raise
    report["status"] = "COMPLETED"
    report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_report(args.out, report)
    print(f"Benchmark complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
