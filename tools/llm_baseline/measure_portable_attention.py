#!/usr/bin/env python3
"""Measure unchanged WGSL inputs before/after a precision correction.

Preparation performs no device operation. Run each shader in separate fresh
processes, then compare receipts. Completed queue calls exclude allocation,
compilation, oracle computation and output transfer. Timing and memory runs are
separate. API buffer capacity is not driver residency or a physical peak.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.llm_baseline.measure_native_baseline import edge_samples


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def system():
    counters = dict(row.split() for row in Path("/proc/vmstat").read_text().splitlines())
    return {"monotonic_ns": time.monotonic_ns(),
            "pswpin": int(counters["pswpin"]), "pswpout": int(counters["pswpout"]),
            "loadavg": Path("/proc/loadavg").read_text().strip()}


def distribution(values):
    values = sorted(values)
    return {"count": len(values), "unit": "nanoseconds", "min": values[0], "max": values[-1],
            **{f"p{percent}": values[math.ceil(percent * len(values) / 100) - 1]
               for percent in (50, 95, 99)}}


def arrays(case):
    import numpy as np
    rng = np.random.default_rng(case["seed"])
    q = rng.normal(size=(case["queries"], case["heads"], case["dim"])).astype(np.float32) * np.float32(.3)
    k, v = [rng.normal(size=(case["keys"], case["heads"], case["dim"])).astype(np.float32) * np.float32(.3)
            for _ in range(2)]
    if case["distribution"] == "flat-positive":
        q.fill(0)
        v += np.float32(16)
    return q, k, v


def input_digest(values):
    h = hashlib.sha256()
    for value in values:
        h.update(str(value.shape).encode()); h.update(value.dtype.str.encode()); h.update(value.tobytes())
    return h.hexdigest()


def prepare(args):
    import numpy as np
    cases = []
    for keys in (31, 511, 512, 1500):
        for queries in (1, keys):
            for kind in ("varying", "flat-positive"):
                case = dict(id=f"keys{keys}-queries{queries}-{kind}", keys=keys, queries=queries,
                            heads=6, dim=64, seed=40195 + keys + queries, distribution=kind,
                            causal=False, output_policy="resident F32; no readback during timing")
                values = arrays(case)
                case["inputs_sha256"] = input_digest(values)
                case["storage_buffer_capacity_bytes"] = sum(a.nbytes for a in values) + values[0].nbytes + 4
                case["uniform_buffer_capacity_bytes"] = 48 if args.operator == "cross" else 128
                case["sampled_oracle_rows"] = sorted(set((0, queries // 2, queries - 1)))
                # Freeze the independent mathematical reference before device
                # measurement. Workers do not compute an oracle inside their
                # timing or memory process.
                q, k, v = values
                references = []
                for row in case["sampled_oracle_rows"]:
                    for head in range(case["heads"]):
                        scores = k[:, head].astype(np.float64) @ q[row, head].astype(np.float64) / math.sqrt(case["dim"])
                        probabilities = np.exp(scores - scores.max()); probabilities /= probabilities.sum()
                        references.append({"query": row, "head": head,
                                           "f64_values": (probabilities @ v[:, head].astype(np.float64)).tolist()})
                case["frozen_sampled_f64_reference"] = references
                cases.append(case)
    plan = {"format": "volvox-portable-attention-measurement/v1", "status": "PREPARED",
            "operator": args.operator,
            "helper": identity(__file__), "before": identity(args.before), "after": identity(args.after),
            "telemetry_helper": identity(ROOT / "tools/llm_baseline/measure_native_baseline.py"),
            "cases": cases, "warmup_calls_per_case": 2, "measured_calls_per_case": 5,
            "fresh_timing_processes_per_variant": 3, "memory_processes_per_variant": 1,
            "power_preference": "high-performance", "required_vendor": 4318,
            "minimum_free_disk_bytes": 1 << 30, "maximum_owned_buffer_capacity_bytes": 16 << 20,
            "atol": 1e-5, "rtol": 1e-7,
            "timing_scope": "Host encode, submit and on_submitted_work_done_sync; allocation/compile/readback/oracle excluded. This is raw wgpu-native WGSL execution, not browser/public-model latency.",
            "comparison_scope": "Matching F32 operands and geometry; historical before shader may fail the numerical gate. Its timing remains diagnostic and is never accepted as a correct alternative.",
            "physical_peak_qualified": False, "performance_regression_allowance": None,
            "swap_policy": "Any system pswpin/pswpout delta invalidates the timing receipt; stop later workers.",
            "service_actions": 0}
    if args.intermediate:
        plan["c7b"] = identity(args.intermediate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(plan, stream, indent=2); stream.write("\n")


def run(args):
    import numpy as np
    import wgpu
    from tools.llm_baseline import measure_native_baseline as telemetry
    plan = json.loads(args.plan.read_text())
    require(plan["helper"] == identity(__file__), "measurement helper differs")
    require(plan["telemetry_helper"] == identity(ROOT / "tools/llm_baseline/measure_native_baseline.py"), "telemetry helper differs")
    for variant in ("before", "c7b", "after"):
        if variant not in plan:
            continue
        require(identity(plan[variant]["path"]) == plan[variant], "pinned WGSL source differs")
    require(args.variant in plan, "selected shader variant was not prepared")
    stat = os.statvfs(args.out.parent)
    require(stat.f_bavail * stat.f_frsize >= plan["minimum_free_disk_bytes"], "insufficient free disk")
    before_system = system()
    adapter = device = None
    sampler = nvml = None
    live = []
    receipt = {"format": "volvox-portable-attention-observation/v1", "status": "FAIL",
               "plan": identity(args.plan), "helper": identity(__file__), "variant": args.variant,
               "mode": args.mode, "shader": plan[args.variant], "host": platform.node(),
               "pid": os.getpid(), "system_before": before_system, "cases": [],
               "service_actions": 0, "physical_peak_qualified": False}
    try:
        if args.mode == "memory":
            require(args.tmpdir and args.tmpdir.is_dir(), "memory worker needs a task-owned directory")
            nvml = telemetry.NvmlSampler()
            sampler = telemetry.RssSampler(10_000_000, args.tmpdir, nvml)
            sampler.thread.start()
        def mark(phase):
            if sampler:
                sampler.phase = phase
                receipt.setdefault("phase_markers", []).append({"phase": phase, "monotonic_ns": time.perf_counter_ns()})
        mark("adapter_device_pipeline")
        adapter = wgpu.gpu.request_adapter_sync(power_preference=plan["power_preference"])
        require(adapter is not None, "no device adapter")
        info = dict(adapter.info)
        require(info.get("vendor_id") == plan["required_vendor"] and info.get("adapter_type") != "CPU",
                "required physical NVIDIA adapter missing")
        receipt["adapter"] = info; receipt["wgpu_version"] = wgpu.__version__
        device = adapter.request_device_sync(required_limits={"max-storage-buffers-per-shader-stage": 8})
        started = time.perf_counter_ns()
        module = device.create_shader_module(code=Path(plan[args.variant]["path"]).read_text())
        pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
        receipt["pipeline_create_ns"] = time.perf_counter_ns() - started
        for case in plan["cases"]:
            mark(case["id"] + ":operand_buffers")
            q, k, v = arrays(case)
            require(input_digest((q, k, v)) == case["inputs_sha256"], "operand identity differs")
            capacity = case["storage_buffer_capacity_bytes"] + case["uniform_buffer_capacity_bytes"]
            require(capacity <= plan["maximum_owned_buffer_capacity_bytes"], "owned memory budget exceeded")
            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
            def own(value):
                live.append(value)
                return value
            buffers = [own(device.create_buffer_with_data(data=a, usage=usage)) for a in (q, k, v)]
            zero = own(device.create_buffer_with_data(data=np.zeros(1, np.int32), usage=usage))
            output = own(device.create_buffer(size=q.nbytes, usage=usage))
            cross = plan["operator"] == "cross"
            words = bytearray(48 if cross else 128)
            fields = ({0: case["queries"], 1: case["keys"], 2: case["heads"] * case["dim"],
                       3: case["heads"], 4: case["dim"], 5: 1, 9: case["heads"]} if cross else
                      {0: 1, 1: case["queries"], 2: case["keys"], 3: case["heads"],
                       4: case["heads"], 5: case["dim"], 6: case["dim"],
                       22: case["queries"] * case["heads"]})
            for index, value in fields.items():
                struct.pack_into("<I", words, index * 4, value)
            struct.pack_into("<f", words, (6 if cross else 26) * 4, 1 / math.sqrt(case["dim"]))
            uniform = own(device.create_buffer_with_data(data=words, usage=wgpu.BufferUsage.UNIFORM))
            bind_buffers = buffers + ([zero, output, uniform] if cross else [zero, zero, zero, zero, output, uniform])
            group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
                entries=[{"binding": i, "resource": {"buffer": buffer}} for i, buffer in enumerate(bind_buffers)])
            def dispatch():
                encoder = device.create_command_encoder()
                compute = encoder.begin_compute_pass(); compute.set_pipeline(pipeline)
                compute.set_bind_group(0, group)
                if cross:
                    compute.dispatch_workgroups((case["queries"] + 63) // 64, case["heads"], 1)
                else:
                    compute.dispatch_workgroups((case["queries"] * case["heads"] + 63) // 64)
                compute.end(); device.queue.submit([encoder.finish()])
                device.queue.on_submitted_work_done_sync()
            # Warm calls explicitly finish before the measured region.
            mark(case["id"] + ":warmup")
            for _ in range(plan["warmup_calls_per_case"]):
                dispatch()
            times = []
            mark(case["id"] + ":completed_queue_calls")
            if args.mode == "timing":
                for _ in range(plan["measured_calls_per_case"]):
                    started = time.perf_counter_ns(); dispatch()
                    times.append(time.perf_counter_ns() - started)
            else:
                dispatch()
            if nvml:
                receipt.setdefault("resident_device_samples", []).append({"case": case["id"],
                    "phase": "after_completed_calls_before_readback", "monotonic_ns": time.perf_counter_ns(),
                    "nvml": nvml.sample(), "known_api_buffer_capacity_bytes": capacity})
            mark(case["id"] + ":output_check")
            actual = np.frombuffer(device.queue.read_buffer(output), dtype=np.float32).copy().reshape(q.shape)
            # Only compare the previously frozen reference after timing.
            sampled = []
            for reference in case["frozen_sampled_f64_reference"]:
                row, head = reference["query"], reference["head"]
                expected = np.asarray(reference["f64_values"], np.float64)
                error = np.abs(actual[row, head].astype(np.float64) - expected)
                sampled.append({"query": row, "head": head, "maximum_absolute_error": float(error.max()),
                                "pass": bool(np.all(error <= plan["atol"] + plan["rtol"] * np.abs(expected)))})
            entry = {"id": case["id"], "inputs_sha256": case["inputs_sha256"],
                     "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                     "sampled_f64_reference": sampled, "sampled_numerical_pass": all(r["pass"] for r in sampled),
                     "known_live_api_buffer_capacity_bytes": capacity,
                     "raw_completed_queue_call_ns": times, "distribution": distribution(times) if times else None}
            for buffer in live:
                buffer.destroy()
            live.clear(); device.queue.on_submitted_work_done_sync()
            entry["known_live_api_buffer_capacity_after_case"] = 0
            receipt["cases"].append(entry)
        # The current implementation must satisfy the fixed numerical gate.
        if args.variant == "after":
            require(all(case["sampled_numerical_pass"] for case in receipt["cases"]), "candidate numerical gate failed")
        receipt["measurement_valid"] = True
        receipt["status"] = "PASS"
    except BaseException as error:
        receipt["error"] = {"message": str(error), "traceback": traceback.format_exc()}
    finally:
        for buffer in live:
            buffer.destroy()
        if device is not None:
            device.destroy()
        if sampler:
            sampler.stop(); receipt["memory_samples"] = sampler.samples
            receipt["maximum_sample_gap_ns"] = max((b["monotonic_ns"] - a["monotonic_ns"]
                for a, b in zip(sampler.samples, sampler.samples[1:])), default=None)
            receipt["rss_sampled_max_bytes"] = max((row["VmRSS"] for row in sampler.samples), default=None)
            receipt["pss_sampled_max_bytes"] = max((row.get("Pss", 0) for row in sampler.samples), default=None)
            if sampler.error:
                receipt["status"] = "FAIL"; receipt["sampler_error"] = str(sampler.error)
        if nvml:
            receipt["nvml_after_device_close"] = nvml.sample(); nvml.close()
        receipt["system_after"] = system()
        receipt["system_swap_delta"] = {name: receipt["system_after"][name] - before_system[name]
                                        for name in ("pswpin", "pswpout")}
        if any(receipt["system_swap_delta"].values()):
            receipt["status"] = "FAIL"; receipt["measurement_valid"] = False
            receipt["swap_invalidated"] = True
        receipt["known_live_api_buffer_capacity_after_device_close"] = 0
        receipt["process_lifetime_vmhwm_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        receipt["vmhwm_scope"] = "Whole process including setup/reference; no engine-only or exact request memory claim."
        with args.out.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": receipt["status"], "cases": len(receipt["cases"]), "out": str(args.out)}))
    return 0 if receipt["status"] == "PASS" else 1


def swap():
    rows = dict(row.split() for row in Path("/proc/vmstat").read_text().splitlines())
    return {name: int(rows[name]) for name in ("pswpin", "pswpout")}


def run_series(args):
    out = args.out.resolve()
    require(not out.exists(), "fresh task-owned output directory required")
    out.mkdir(parents=True)
    helper = Path(__file__)
    plan = json.loads(args.plan.read_text())
    runtime = json.loads(args.runtime_inventory.read_text())
    def verify_runtime():
        for entry in runtime:
            file = args.runtime_inventory.parent / entry["path"]
            actual = identity(file)
            require(actual["bytes"] == entry["bytes"] and actual["sha256"] == entry["sha256"], "runtime dependency differs")
    verify_runtime()
    order = [(v, "timing", i) for i in range(3) for v in (("before", "after") if i % 2 == 0 else ("after", "before"))]
    order += [(v, "memory", 0) for v in ("before", "after")]
    policy = dict(format="volvox-portable-attention-series-policy/v1", plan=identity(args.plan),
                  helper=identity(helper), series_helper=identity(Path(__file__)),
                  prepost_idle_samples=3, maximum_cpu_busy_percent=5, maximum_gpu_utilization_percent=5,
                  system_swap_delta_required=0, elapsed_limit_seconds=args.deadline_seconds,
                  runtime_inventory=identity(args.runtime_inventory),
                  worker_order=order, timing_observers=False, existing_service_actions=0,
                  idle_scope="Three host/device samples at each worker edge; background activity during a worker is not independently observed.",
                  performance_scope=plan["timing_scope"], physical_peak_qualified=False)
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    report = dict(format="volvox-portable-attention-series/v1", status="RUNNING",
                  policy=identity(out / "policy.json"), workers=[], existing_service_actions=0)
    report["system_swap_before"] = swap()
    child = None; interrupted = []
    def stop(signum, _frame):
        interrupted.append(signum)
        if child and child.poll() is None: child.terminate()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, stop)
    deadline = time.monotonic() + args.deadline_seconds
    try:
        for variant, mode, index in order:
            require(not interrupted, "series interrupted")
            require(time.monotonic() < deadline, "series deadline reached")
            file = out / f"{variant}-{mode}-{index}.json"
            scratch = out / f"temporary-{variant}-{mode}-{index}"
            scratch.mkdir()
            command = ["python3", str(helper), "--plan", str(args.plan), "--variant", variant,
                       "--mode", mode, "--tmpdir", str(scratch), "--out", str(file)]
            worker = dict(variant=variant, mode=mode, index=index, command=command, status="RUNNING")
            report["workers"].append(worker)
            if mode == "timing":
                worker["before_idle_samples"] = edge_samples()
                require(all(r["cpu_busy_percent"] <= 5 and all(g["utilization_percent"] <= 5 for g in r["gpu_devices"])
                            for r in worker["before_idle_samples"]), "before-worker idle edge failed")
            try:
                with file.with_suffix(".log").open("w") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    try: code = child.wait(timeout=max(1, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        raise RuntimeError("worker/series timeout")
                worker["exit_code"] = code
                require(not interrupted, "series interrupted")
                require(code == 0 and file.exists(), "worker failed")
                receipt = json.loads(file.read_text())
                worker["receipt"] = identity(file)
                require(receipt["status"] == "PASS" and receipt.get("measurement_valid"), "invalid worker receipt")
                require(all(delta == 0 for delta in receipt["system_swap_delta"].values()), "swap invalidated worker")
                if mode == "timing":
                    worker["after_idle_samples"] = edge_samples()
                    require(all(r["cpu_busy_percent"] <= 5 and all(g["utilization_percent"] <= 5 for g in r["gpu_devices"])
                                for r in worker["after_idle_samples"]), "after-worker idle edge failed")
                worker["status"] = "PASS"
            finally:
                shutil.rmtree(scratch)
            (out / "series.json").write_text(json.dumps(report, indent=2) + "\n")
        require(identity(helper) == policy["helper"], "worker helper changed")
        require(identity(args.plan) == policy["plan"], "plan changed")
        verify_runtime()
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"; report["error"] = repr(error)
    finally:
        if child and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for scratch in out.glob("temporary-*"):
            if scratch.is_dir(): shutil.rmtree(scratch)
        report["system_swap_after"] = swap()
        report["system_swap_delta"] = {key: report["system_swap_after"][key] - value
                                       for key, value in report["system_swap_before"].items()}
        if any(report["system_swap_delta"].values()):
            report["status"] = "FAIL"; report["swap_invalidated"] = True
        (out / "series.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], workers=len(report["workers"]), out=str(out))))
    return 0 if report["status"] == "PASS" else 1



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--series", action="store_true", help="run paired fresh timing and memory workers")
    parser.add_argument("--operator", choices=("sparse", "cross"), default="sparse")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--intermediate", type=Path, help="optional preserved c7b precision shader")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--tmpdir", type=Path)
    parser.add_argument("--variant", choices=("before", "c7b", "after"))
    parser.add_argument("--mode", choices=("timing", "memory"), default="timing")
    parser.add_argument("--runtime-inventory", type=Path, help="required runtime file pins for --series")
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.series:
        if not args.plan or not args.runtime_inventory:
            parser.error("--series requires --plan and --runtime-inventory")
        if args.variant or args.tmpdir or args.before or args.after or args.intermediate:
            parser.error("--series cannot select a single worker or prepare shader inputs")
        if args.deadline_seconds <= 0:
            parser.error("--deadline-seconds must be positive")
        return run_series(args)
    if args.runtime_inventory:
        parser.error("--runtime-inventory requires --series")
    require(not args.out.exists(), "refusing to overwrite a receipt")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        require(args.before and args.after, "prepare needs both source files")
        prepare(args); return 0
    require(args.plan and args.variant, "run needs plan and variant")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
