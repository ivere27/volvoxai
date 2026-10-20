#!/usr/bin/env python3
"""Observe an unchanged browser qualifier with independent NVML samples.

Driver process bytes are sampled physical observations, separate from browser
PSS, WASM heap, and GPU API buffer capacities. Compute/graphics aliases of one
PID occur once. Task identity comes from its unique executable path and procfs
start time; existing GPU services and assets are never modified.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from measure_native_baseline import NvmlSampler, identity, require


class BrowserNvml(NvmlSampler):
    def __init__(self):
        super().__init__()
        self.graphics = self.lib.nvmlDeviceGetGraphicsRunningProcesses_v2
        self.graphics.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
                                  ctypes.POINTER(self.Process)]
        self.names = []
        for handle in self.devices:
            entry = {}
            for key, function in (("name", "nvmlDeviceGetName"), ("uuid", "nvmlDeviceGetUUID")):
                call = getattr(self.lib, function)
                call.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
                data = ctypes.create_string_buffer(128)
                require(call(handle, data, len(data)) == 0, f"NVML {key} query failed")
                entry[key] = data.value.decode()
            self.names.append(entry)

    def sample(self):
        rows = super().sample()
        for index, (handle, row) in enumerate(zip(self.devices, rows)):
            row.update(self.names[index])
            count = ctypes.c_uint()
            require(self.graphics(handle, ctypes.byref(count), None) in (0, 7),
                    "NVML graphics process count failed")
            capacity = count.value + 16
            values = (self.Process * capacity)()
            count.value = capacity
            require(self.graphics(handle, ctypes.byref(count), values) == 0,
                    "NVML graphics process memory failed")
            row["graphics_processes"] = [
                {"pid": value.pid,
                 "used_bytes": None if value.usedGpuMemory == (1 << 64) - 1 else value.usedGpuMemory}
                for value in values[:count.value]]
        return rows


def chrome_processes(executable):
    rows = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            # Chromium rewrites a subprocess's argv[0] to "chrome". Linux's
            # executable link retains the unique task-owned binary identity.
            actual_executable = os.readlink(item / "exe")
            if actual_executable not in (
                    str(executable), str(executable.with_name("chrome_crashpad_handler"))):
                continue
            stat = (item / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            rows[int(item.name)] = {"pid": int(item.name), "starttime": fields[19],
                                   "process_group": int(fields[2]), "state": fields[0]}
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--browser-out", type=Path, required=True)
    parser.add_argument("--max-task-vram-bytes", type=int, default=4 * 1024 ** 3)
    parser.add_argument("--driver", choices=("checkpoint", "multimodal"), default="checkpoint")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--mode", choices=("correctness", "timing"), default="correctness")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    browser_out = args.browser_out.resolve()
    require(root.name.startswith("volvoxai-remaining-validation-"), "unique task-owned root required")
    require(out.is_relative_to(root) and browser_out.is_relative_to(root), "reports must be task-owned")
    require(not out.exists() and not browser_out.exists(), "fresh report paths required")
    chrome = root / ".testchrome/chrome"
    node = root / ".testnode/node"
    helper = root / "tools/llm_baseline" / ("verify_browser_checkpoint.mjs" if args.driver == "checkpoint"
                                         else "verify_browser_multimodal.mjs")
    pins = {key: identity(file) for key, file in (("chrome", chrome), ("node", node),
            ("driver", helper), ("observer", Path(__file__)),
            ("nvml_helper", Path(__file__).with_name("measure_native_baseline.py")))}
    command = [str(node), str(helper), "--backend", "webgpu", "--profile", "full",
               "--chrome", str(chrome), "--out", str(browser_out)]
    if args.driver == "multimodal":
        require(args.bundle is not None and args.bundle.resolve().is_relative_to(root), "task-owned prepared bundle required")
        command += ["--bundle", str(args.bundle.resolve()), "--mode", args.mode,
                    "--rounds", str(args.rounds), "--observe", "true"]
    candidate = {"format": "volvox-browser-nvml-candidate/v1", "pins": pins,
                 "command": command, "sample_interval_ms": 100,
                 "max_task_vram_bytes": args.max_task_vram_bytes,
                 "scope": "Sampled NVIDIA driver residency of task-owned Chrome PIDs; compute and graphics aliases deduplicated."}
    out.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = out.with_suffix(".candidate.json")
    candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
    report = {"format": "volvox-browser-nvml/v1", "status": "running",
              "candidate": identity(candidate_path), "samples": [], "observed_processes": {},
              "scope": "Independent sampled driver residency, not an exact physical peak. Device-wide memory includes unrelated work and is not added to process bytes."}
    sampler = BrowserNvml()
    child = None
    previous = None
    maximum_gap = 0
    peak_task = 0
    nonzero = 0
    interrupted = []

    def stop(signum, _frame):
        interrupted.append(signum)
        if child and child.poll() is None:
            child.terminate()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        log_path = out.with_suffix(".driver.log")
        with log_path.open("w") as log:
            child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 22 * 60
            while child.poll() is None:
                now = time.monotonic_ns()
                if previous is not None:
                    maximum_gap = max(maximum_gap, now - previous)
                previous = now
                current = chrome_processes(chrome)
                report["observed_processes"].update({str(pid): row for pid, row in current.items()})
                devices = sampler.sample()
                task_total = 0
                for device in devices:
                    selected = {}
                    for kind in ("compute_processes", "graphics_processes"):
                        for process in device[kind]:
                            if process["pid"] in current:
                                require(process["used_bytes"] is not None, "task GPU process residency unavailable")
                                selected[process["pid"]] = max(selected.get(process["pid"], 0), process["used_bytes"])
                    device["task_chrome_processes"] = [{"pid": pid, "used_bytes": size,
                        "starttime": current[pid]["starttime"]} for pid, size in selected.items()]
                    device["task_chrome_used_bytes"] = sum(selected.values())
                    task_total += device["task_chrome_used_bytes"]
                report["samples"].append({"monotonic_ns": now, "devices": devices, "task_used_bytes": task_total})
                peak_task = max(peak_task, task_total)
                nonzero += task_total > 0
                require(task_total <= args.max_task_vram_bytes, "prospective task GPU residency budget exceeded")
                require(time.monotonic() < deadline, "22 minute observer deadline exceeded")
                time.sleep(0.1)
            require(child.returncode == 0 and not interrupted, "browser driver failed or was interrupted")
        browser = json.loads(browser_out.read_text())
        require(browser["status"] == "PASS" and browser["cleanup"]["process_tree_drained"],
                "complete successful browser qualification required")
        require(nonzero > 0, "no task-owned Chrome NVIDIA residency observed")
        require(not chrome_processes(chrome), "task-owned Chrome processes still live")
        for key, pin in pins.items():
            require(identity(pin["path"]) == pin, f"{key} changed during observation")
        report["browser_receipt"] = identity(browser_out)
        report["driver_log"] = identity(log_path)
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        fallback_killed = []
        for pid, expected in chrome_processes(chrome).items():
            try:
                current = chrome_processes(chrome).get(pid)
                if current and current["starttime"] == expected["starttime"]:
                    os.kill(pid, signal.SIGKILL)
                    fallback_killed.append(pid)
            except (ProcessLookupError, FileNotFoundError):
                pass
        for _ in range(100):
            if not chrome_processes(chrome):
                break
            time.sleep(0.05)
        sampler.close()
        report["peak_sampled_task_vram_bytes"] = peak_task
        report["nonzero_task_samples"] = nonzero
        report["maximum_sample_gap_ms"] = maximum_gap / 1e6
        report["sample_count"] = len(report["samples"])
        report["task_chrome_processes_after"] = chrome_processes(chrome)
        report["fallback_killed_task_pids"] = fallback_killed
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in ("status", "error", "peak_sampled_task_vram_bytes", "sample_count")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
