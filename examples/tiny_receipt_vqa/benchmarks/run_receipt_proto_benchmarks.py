#!/usr/bin/env python3
"""Run receipt latency measurements sequentially with host-contention gates.

Run on the benchmark machine. The leaf drivers time the public API; this
controller records the environment and rejects foreign CPU/GPU work by default.
An explicit shared-host run retains contention evidence without classifying it
as an idle-host benchmark or GPU qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def cpu_snapshot():
    values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    return sum(values), values[3] + values[4]


def process_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[11]) + int(fields[12])
    except FileNotFoundError:
        return None


def busy_cores(before, after):
    total = after[0] - before[0]
    return (1 - (after[1] - before[1]) / total) * os.cpu_count() if total > 0 else 0


def quiet_cpu():
    before, samples = cpu_snapshot(), []
    for _ in range(3):
        time.sleep(.4)
        after = cpu_snapshot()
        samples.append(busy_cores(before, after))
        before = after
    return {"busy_core_samples": samples, "median_busy_cores": statistics.median(samples)}


def gpu_state(own_pid=None):
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,pstate,clocks.current.sm",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return {"available": False, "reason": "nvidia-smi-not-found"}
    if query.returncode:
        return {"available": False}
    names = ["model", "driver", "utilization_percent", "memory_used_mib", "memory_total_mib", "pstate", "sm_clock_mhz"]
    devices = [dict(zip(names, [v.strip() for v in line.split(",")])) for line in query.stdout.splitlines()]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=10)
    foreign = 0
    for text in apps.stdout.splitlines():
        if not text.strip().isdigit() or int(text) == own_pid:
            continue
        try:
            state = Path(f"/proc/{int(text)}/stat").read_text().rsplit(") ", 1)[1].split()[0]
            if state not in ["T", "t", "Z"]:
                foreign += 1
        except FileNotFoundError:
            pass
    return {"available": True, "devices": devices, "foreign_active_compute_processes": foreign}


def main(model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--deno", type=Path, required=True)
    parser.add_argument("--node", type=Path, help="use this Node executable for WASM; WebGPU still uses Deno")
    parser.add_argument("--models", nargs="+", choices=[model] if model else ["digit", "vqa"],
                        default=[model] if model else ["digit", "vqa"])
    parser.add_argument("--profiles", nargs="+", choices=["inference", "full"], default=["inference"])
    parser.add_argument("--variants", nargs="+", choices=["fp32", "int8", "ptq", "ptq-wasm"],
                        default=["fp32", "int8", "ptq", "ptq-wasm"])
    parser.add_argument("--backends", nargs="+", choices=["ort", "cpu", "cuda", "vulkan", "opengl", "wasm", "webgpu"],
                        required=True)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--max-foreign-cpu", type=float, default=1.0)
    parser.add_argument("--allow-busy-host", action="store_true",
                        help="allow CPU/WASM desktop background work, retaining failed idle-host gates")
    parser.add_argument("--host-label", choices=["cpu-wasm-host", "gpu-host"], required=True,
                        help="execution role; hardware and account identities are not collected")
    parser.add_argument("--ort-reference", type=Path, help="independent current-host VQA INT8 reference")
    args = parser.parse_args()
    assert args.cpu in os.sched_getaffinity(0)
    assert args.repeat >= 1 and args.warmup >= 1 and 0 < args.max_foreign_cpu <= 1
    assert not args.allow_busy_host or set(args.backends) <= {"ort", "cpu", "wasm"}, \
        "GPU measurements require the idle-host gates"
    assert set(args.backends) <= ({"ort", "cpu", "wasm"} if args.host_label == "cpu-wasm-host"
                                  else {"cuda", "vulkan", "opengl", "webgpu"})
    spec = read_json(args.spec)
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, expected in spec["artifacts"].items():
        assert sha(ROOT / "dist/0.5.0" / filename) == expected, filename
    for profile, expected in spec["libraries"].items():
        assert sha(ROOT / "native" / ("libvolvoxai" + ("-full" if profile == "full" else "") + ".so")) == expected
    metadata = {"host_label": args.host_label, "pinned_cpu": args.cpu,
                "deno_sha256": sha(args.deno), "max_foreign_cpu_cores": args.max_foreign_cpu,
                "node_sha256": sha(args.node) if args.node else None,
                "allow_busy_host": args.allow_busy_host,
                "ort_reference_sha256": sha(args.ort_reference) if args.ort_reference else None,
                "controller_sha256": sha(Path(__file__)), "spec_sha256": sha(args.spec)}
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("VOLVOX")}
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        environment[name] = "1"
    environment["DENO_WEBGPU_BACKEND"] = "vulkan"
    environment["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    jobs = []
    for model in args.models:
        for backend in args.backends:
            profiles = ["full"] if backend == "webgpu" else ["inference"] if backend == "ort" else args.profiles
            for profile in profiles:
                variants = [v for v in args.variants if backend != "ort" or v in ["fp32", "int8"]]
                if profile == "full":
                    variants.reverse()
                for variant in variants:
                    jobs.append((model, backend, profile, variant))
    state = {"schema": "volvoxai.receipt-benchmark-suite/v1", "started_utc": stamp(), "host": metadata, "jobs": {}}
    for model, backend, profile, variant in jobs:
        key = "-".join([model, backend, profile, variant])
        output = args.out / (key + ".json")
        if output.exists():
            previous = read_json(output)
            reusable = previous.get("host_quiet_verified") or (args.allow_busy_host and previous.get("measurement_completed"))
            if (reusable and previous.get("spec_sha256") == sha(args.spec)
                    and previous.get("host") == metadata
                    and previous.get("repeat_per_case") == args.repeat and previous.get("warmup_per_case") == args.warmup
                    and previous.get("harness_sha256") == sha(BENCHMARKS / ("receipt_proto_benchmark.mjs" if backend in ["wasm", "webgpu"] else "receipt_proto_benchmark.py"))):
                state["jobs"][key] = {"state": "complete", "reused_report": True,
                                      "host_quiet_verified": previous.get("host_quiet_verified", False)}
                continue
            # A failed replacement must not leave an older report publishable.
            output.replace(output.with_suffix(".json.previous"))
        pre_cpu, pre_gpu = quiet_cpu(), gpu_state()
        gpu_backend = backend in ["cuda", "vulkan", "opengl", "webgpu"]
        admitted = pre_cpu["median_busy_cores"] <= args.max_foreign_cpu
        if gpu_backend:
            admitted &= pre_gpu.get("available", False) and pre_gpu["foreign_active_compute_processes"] == 0
            admitted &= all(float(d["utilization_percent"]) == 0 for d in pre_gpu.get("devices", []))
        if (not admitted and not args.allow_busy_host) or (gpu_backend and not pre_gpu.get("available")):
            state["jobs"][key] = {"state": "host_busy", "pre_cpu": pre_cpu, "pre_gpu": pre_gpu}
            save(args.out / "suite-state.json", state)
            print(json.dumps({"state": "host_busy", "job": key, "cpu": pre_cpu, "gpu": pre_gpu}), flush=True)
            return 2
        leaf = "receipt_proto_benchmark.mjs" if backend in ["wasm", "webgpu"] else "receipt_proto_benchmark.py"
        command = ([str(args.node)] if backend == "wasm" and args.node else
                   [str(args.deno), "run", "--no-config", "--allow-all", "--unstable-webgpu"]
                   if backend in ["wasm", "webgpu"] else [args.python])
        command += [str(BENCHMARKS / leaf), "--spec", str(args.spec.resolve()), "--model", model,
                    "--backend", backend, "--profile", profile, "--variant", variant,
                    "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--out", str(output.resolve())]
        if args.ort_reference and (model, backend, variant) == ("vqa", "ort", "int8"):
            command += ["--ort-reference", str(args.ort_reference.resolve())]
        command = ["taskset", "-c", str(args.cpu), *command]
        with output.with_suffix(".log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        state["jobs"][key] = {"state": "running", "started_utc": stamp(), "pid": child.pid}
        save(args.out / "suite-state.json", state)
        observations = []
        before, own_before, clock_before = cpu_snapshot(), process_ticks(child.pid), time.monotonic()
        while child.poll() is None:
            time.sleep(.5)
            after, own_after, clock_after = cpu_snapshot(), process_ticks(child.pid), time.monotonic()
            if own_before is not None and own_after is not None:
                own = (own_after - own_before) / os.sysconf("SC_CLK_TCK") / (clock_after - clock_before)
                observations.append({"foreign_cpu_cores": max(0, busy_cores(before, after) - own)})
                if gpu_backend and len(observations) % 4 == 0:
                    gpu = gpu_state(child.pid)
                    observations[-1]["foreign_gpu_processes"] = gpu.get("foreign_active_compute_processes", -1)
                    observations[-1]["gpu_devices"] = gpu.get("devices", [])
            before, own_before, clock_before = after, own_after, clock_after
        post_cpu, post_gpu = quiet_cpu(), gpu_state()
        foreign = [o["foreign_cpu_cores"] for o in observations]
        foreign_mean = statistics.mean(foreign) if foreign else 0
        qualified = (admitted and child.returncode == 0 and foreign_mean <= args.max_foreign_cpu
                     and post_cpu["median_busy_cores"] <= args.max_foreign_cpu)
        if gpu_backend:
            qualified &= post_gpu.get("available", False) and post_gpu["foreign_active_compute_processes"] == 0
            qualified &= all(o.get("foreign_gpu_processes", 0) == 0 for o in observations)
        conditions = {"pre_cpu": pre_cpu, "pre_gpu": pre_gpu, "post_cpu": post_cpu, "post_gpu": post_gpu,
                      "during_foreign_cpu_mean": foreign_mean, "during_foreign_cpu_max": max(foreign, default=0),
                      "during_samples": observations}
        save(output.with_name(output.stem + "-conditions.json"), conditions)
        ended = stamp()
        if child.returncode == 0:
            report = read_json(output)
            assert report.get("output_matches_benchmark_reference", report["output_matches_2000_reference"])
            assert report["repeated_outputs_identical"]
            report.update(host_quiet_verified=qualified, host=metadata, conditions=conditions,
                          measurement_completed=True,
                          measurement_scope="idle-host" if qualified else "shared-host-observation",
                          started_utc=state["jobs"][key]["started_utc"], ended_utc=ended)
            save(output, report)
        complete = child.returncode == 0 and (qualified or args.allow_busy_host)
        state["jobs"][key].update(state="complete" if complete else "failed_or_contended",
                                  host_quiet_verified=qualified,
                                  exit_code=child.returncode, ended_utc=ended)
        save(args.out / "suite-state.json", state)
        print(json.dumps({"job": key, "qualified": qualified, "foreign_cpu_mean": foreign_mean}), flush=True)
        if not complete:
            return 1
    state["finished_utc"] = stamp()
    save(args.out / "suite-state.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
