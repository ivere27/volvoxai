#!/usr/bin/env python3
"""Run isolated browser correctness, timing and memory workers serially.

The default selection runs every planned worker once and stops on failure.
Qualified selection retains the earliest successful timing workers under a
fixed attempt cap; only environmental idle/swap failures are retried. Latency
values never take part in selection. Every worker owns its browser cleanup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.llm_baseline.measure_native_baseline import edge_samples, identity, require


def idle(rows):
    return all(v["cpu_busy_percent"] <= 5 and all(g["utilization_percent"] <= 5 for g in v["gpu_devices"]) for v in rows)


def run_all(args):
    root, out, bundle = args.root.resolve(), args.out.resolve(), args.bundle.resolve()
    require(root.name.startswith("volvoxai-remaining-validation-"), "task-owned root required")
    require(out.is_relative_to(root) and bundle.is_relative_to(root), "task-owned outputs and prepared bundle required")
    require(not out.exists(), "use a fresh output directory")
    out.mkdir(parents=True)
    node, chrome = root / ".testnode/node", root / ".testchrome/chrome"
    helper = root / "tools/llm_baseline/verify_browser_multimodal.mjs"
    observer = root / "tools/llm_baseline/observe_browser_checkpoint.py"
    source = root / "tools/llm_baseline/browser_multimodal_case.mjs"
    processor = root / "tools/llm_baseline/browser_media_processors.mjs"
    corpus = json.loads((bundle / "corpus.json").read_text())
    version = json.loads((root / "package.json").read_text())["version"]
    pinned = [node, chrome, helper, observer, source, processor, Path(__file__), root / "package.json",
        root / "tools/release_version.mjs", bundle / "corpus.json",
        bundle / "audio.pcm.f32", bundle / "browser-references.safetensors", root / "tools/llm_baseline/multimodal_sensor.json",
        root / "tools/llm_baseline/measure_native_baseline.py", root / "build/receipt-vqa-ptq-head-29ffb7e/fp32/vocab.json"]
    for role in corpus["packages"]: pinned += [bundle / role / "graph.json", bundle / role / "model.safetensors"]
    for cell in args.cells:
        suffix = ".lite" if cell.startswith("inference") else ""
        pinned += [root / "dist" / version / ("volvoxai" + suffix + ext) for ext in (".js", ".wasm")]
    pins = {str(path.relative_to(root)): identity(path) for path in sorted(set(pinned))}
    policy = {"format": "volvox-browser-multimodal-series-policy/v1", "pins": pins, "stage": args.stage, "cells": args.cells,
        "timing": {"fresh_workers": 3, "rounds": 8, "cold_rounds": [0], "excluded_warmups": [1, 2], "measured_rounds": [3, 4, 5, 6, 7],
            "oracle_loaded": False, "memory_observers": False, "swap_counter_delta": 0},
        "memory": {"fresh_workers": 1, "rounds": 3, "oracle_loaded": False, "rss_pss_wasm_api_capacities_separate": True,
            "allocation_rows_per_snapshot": 128, "allocation_inventory_truncation_explicit": True,
            "decoder_snapshot_tokens": "first and final; phase boundaries also observed",
            "browser_tree_rss_limit_bytes": 8 << 30, "nvml_task_residency_limit_bytes": 8 << 30,
            "wasm_linear_limit_bytes": 2147418112, "gpu_api_buffer_limit_bytes": 2 << 30},
        "idle_edges": {"samples_per_edge": 3, "seconds_per_sample": 1, "cpu_busy_percent_max": 5, "gpu_utilization_percent_max": 5,
            "scope": "Three host/device observations before and after each timing worker; unrelated activity during the worker is not independently observed."},
        "existing_service_actions": 0, "execution_policy": "serial workers, stop on failure, owned browser cleanup on failure/signals/timeout"}
    policy_path = out / "policy.json"; policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    report = {"format": "volvox-browser-multimodal-series/v1", "status": "RUNNING", "policy": identity(policy_path), "workers": [], "existing_service_actions": 0}
    child = None; interrupted = []

    def stop(signum, _frame):
        interrupted.append(signum)
        if child and child.poll() is None: child.terminate()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, stop)
    try:
        stages = ("correctness", "timing", "memory") if args.stage == "all" else (args.stage,)
        for stage in stages:
            for cell in args.cells:
                profile, backend = cell.split("-")
                for index in range(3 if stage == "timing" else 1):
                    require(not interrupted, "series interrupted")
                    file = out / (cell + "-" + stage + "-" + str(index) + ".json")
                    command = [str(node), str(helper), "--backend", backend, "--profile", profile, "--chrome", str(chrome), "--bundle", str(bundle),
                        "--mode", "correctness" if stage == "correctness" else "timing", "--observe", "false" if stage == "timing" else "true",
                        "--rounds", "8" if stage == "timing" else "3" if stage == "memory" else "1", "--out", str(file)]
                    nvml_file = None
                    if stage == "memory" and backend == "webgpu":
                        nvml_file = file.with_name(file.stem + "-nvml.json")
                        command = ["python3", str(observer), "--root", str(root), "--out", str(nvml_file), "--browser-out", str(file),
                            "--max-task-vram-bytes", str(8 << 30), "--driver", "multimodal", "--bundle", str(bundle), "--mode", "timing", "--rounds", "3"]
                    worker = {"cell": cell, "stage": stage, "index": index, "command": command, "status": "RUNNING"}
                    report["workers"].append(worker)
                    if stage == "timing":
                        worker["before_idle_samples"] = edge_samples()
                        require(all(v["cpu_busy_percent"] <= 5 and all(g["utilization_percent"] <= 5 for g in v["gpu_devices"]) for v in worker["before_idle_samples"]), "before worker idle edge failed")
                    with file.with_suffix(".log").open("w") as log:
                        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                        try: code = child.wait(timeout=23 * 60)
                        except subprocess.TimeoutExpired:
                            child.terminate()
                            try: child.wait(timeout=30)
                            except subprocess.TimeoutExpired: child.kill(); child.wait()
                            raise RuntimeError("browser worker deadline exceeded")
                    child = None
                    require(code == 0 and not interrupted, "browser worker failed or interrupted")
                    value = json.loads(file.read_text()); result = value["browser_result"]
                    require(value["status"] == "PASS" and result["status"] == "PASS" and value["cleanup"]["process_tree_drained"], "complete browser PASS required")
                    require(result["oracle_loaded"] == (stage == "correctness"), "oracle scope changed")
                    require(value["sampled_memory_enabled"] == (stage != "timing"), "memory sampling scope changed")
                    if stage == "timing":
                        require(not result["snapshots"] and not result["device_buffers"] and not value["samples"], "timing instrumentation not absent")
                        worker["after_idle_samples"] = edge_samples()
                        require(all(v["cpu_busy_percent"] <= 5 and all(g["utilization_percent"] <= 5 for g in v["gpu_devices"]) for v in worker["after_idle_samples"]), "after worker idle edge failed")
                        worker["system_swap_gate"] = value["system_swap_gate"]
                    worker["receipt"] = identity(file)
                    if nvml_file: require(json.loads(nvml_file.read_text())["status"] == "PASS", "independent NVIDIA observation failed"); worker["nvml_receipt"] = identity(nvml_file)
                    worker["status"] = "PASS"
                    for pin in pins.values(): require(identity(Path(pin["path"])) == pin, "input/source/artifact changed during series")
                    (out / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({"cell": cell, "stage": stage, "index": index, "status": "PASS"}), flush=True)
        report["status"] = "PASS"
    except BaseException as error:
        report["status"] = "FAIL"; report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if child and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        report["interrupted_signals"] = interrupted
        report["child_drained"] = child is None or child.poll() is not None
        (out / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1



def run_qualified(args):
    root, out = args.root.resolve(), args.out.resolve()
    require(root.name.startswith("volvoxai-remaining-validation-"), "task-owned root required")
    require(out.is_relative_to(root) and args.bundle.resolve().is_relative_to(root) and not out.exists(), "fresh task-owned output required")
    require(1 <= args.qualified_workers <= args.attempt_cap <= 5, "prospective attempt cap must be at most five")
    require(args.backend != "webgpu" or args.profile == "full", "WebGPU requires full profile")
    reference = json.loads(args.reference.read_text())
    expected = reference["pins"]
    source_pins = {str(p): identity(p) for p in (Path(__file__), root / "tools/llm_baseline/measure_native_baseline.py")}

    def immutable():
        for pin in list(expected.values()) + list(source_pins.values()):
            require(identity(Path(pin["path"])) == pin, "input, observation source or artifact changed")

    def worker_pass(value):
        result = value["browser_result"]
        return (value["status"] == result["status"] == "PASS" and value["cleanup"]["process_tree_drained"]
            and value["system_swap_gate"]["before"] == value["system_swap_gate"]["after"])

    def inspect_receipt(file):
        value = json.loads(file.read_text()); result = value["browser_result"]
        require(value["pins"] == expected, "worker input/artifact identities differ")
        require(result["profile"] == args.profile and result["backend"] == args.backend
            and result["mode"] == "timing" and not result["diagnostic_only"] and result["qualification_claim"], "worker scope differs")
        require(not result["oracle_loaded"] and not result["snapshots"] and not result["device_buffers"] and not value["samples"], "timing observer/oracle present")
        require(len(result["cases"]) == 24 and len(result["workflow_timings"]) == 24, "eight complete workflow rounds required")
        return value

    immutable(); out.mkdir(parents=True)
    policy = {"format": "volvox-browser-multimodal-bounded-policy/v1", "input_artifact_pins": expected,
        "observation_source_pins": source_pins, "reference": identity(args.reference), "existing": [identity(p) for p in args.existing],
        "profile": args.profile, "backend": args.backend, "attempt_cap_including_existing": args.attempt_cap,
        "qualified_fresh_workers_required": args.qualified_workers, "selection": "Chronological first qualified workers; no selection by latency values.",
        "rounds": 8, "cold_round": 0, "excluded_warmups": [1, 2], "measured_rounds": [3, 4, 5, 6, 7],
        "idle_edges": {"samples_per_edge": 3, "seconds_per_sample": 1, "cpu_busy_percent_max": 5, "gpu_utilization_percent_max": 5},
        "swap_counter_delta": 0, "retry_only": "Idle edge or host-swap environmental rejection, with successful workflow and drained owners.",
        "remote_queries_during_worker": "None; original SSH stream only.", "existing_service_actions": 0}
    policy_path = out / "policy.json"; policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    report = {"format": "volvox-browser-multimodal-bounded-series/v1", "status": "RUNNING", "policy": identity(policy_path),
        "workers": [], "qualified": [], "existing_service_actions": 0}
    child = None; interrupted = []

    def stop(signum, _frame):
        interrupted.append(signum)
        if child and child.poll() is None: child.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, stop)
    try:
        for file in args.existing:
            value = inspect_receipt(file)
            completion = json.loads((file.parent / "completion.json").read_text())
            previous = next(w for w in completion["workers"] if str(file) in w["command"])
            passed = worker_pass(value) and previous["status"] == "PASS" and idle(previous["before_idle_samples"]) and idle(previous["after_idle_samples"])
            row = {"existing": True, "receipt": identity(file), "status": "PASS" if passed else "FAIL"}
            report["workers"].append(row)
            if passed: report["qualified"].append(row["receipt"])
        for index in range(len(args.existing), args.attempt_cap):
            if len(report["qualified"]) == args.qualified_workers: break
            require(not interrupted, "interrupted")
            immutable()
            file = out / f"{args.profile}-{args.backend}-timing-{index}.json"
            command = [str(root / ".testnode/node"), str(root / "tools/llm_baseline/verify_browser_multimodal.mjs"),
                "--profile", args.profile, "--backend", args.backend, "--chrome", str(root / ".testchrome/chrome"),
                "--bundle", str(args.bundle), "--mode", "timing", "--observe", "false", "--rounds", "8", "--out", str(file)]
            row = {"existing": False, "index": index, "command": command, "status": "RUNNING", "before_idle_samples": edge_samples()}
            report["workers"].append(row)
            if not idle(row["before_idle_samples"]):
                row.update(status="FAIL", rejection="before idle edge")
            else:
                with file.with_suffix(".log").open("w") as log:
                    child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                    try: code = child.wait(timeout=23 * 60)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try: child.wait(timeout=30)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        raise RuntimeError("browser worker deadline exceeded")
                child = None; require(not interrupted, "interrupted")
                value = inspect_receipt(file); row["receipt"] = identity(file)
                row["system_swap_gate"] = value["system_swap_gate"]
                if code != 0 or not worker_pass(value):
                    require(value["browser_result"]["status"] == "PASS" and value["cleanup"]["process_tree_drained"]
                        and "system swap activity invalidates timing measurement" in value.get("error", ""), "non-environmental worker failure")
                    row.update(status="FAIL", rejection="host swap activity")
                else:
                    row["after_idle_samples"] = edge_samples()
                    if not idle(row["after_idle_samples"]): row.update(status="FAIL", rejection="after idle edge")
                    else: row["status"] = "PASS"
            immutable()
            if row["status"] == "PASS": report["qualified"].append(row["receipt"])
            (out / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"profile": args.profile, "backend": args.backend, "attempt": index,
                "status": row["status"], "rejection": row.get("rejection"), "qualified_workers": len(report["qualified"])}), flush=True)
        require(len(report["qualified"]) == args.qualified_workers, "attempt cap reached before three qualified workers")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error={"type": type(error).__name__, "message": str(error)})
        if report["workers"] and report["workers"][-1]["status"] == "RUNNING": report["workers"][-1]["status"] = "FAIL"
    finally:
        if child and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        report["interrupted_signals"] = interrupted
        report["child_drained"] = child is None or child.poll() is not None
        (out / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", choices=("all", "qualified"), default="all",
                        help="all planned workers, or earliest qualified timing workers")
    parser.add_argument("--stage", choices=("correctness", "timing", "memory", "all"))
    parser.add_argument("--cells", nargs="+", choices=("inference-wasm", "full-wasm", "full-webgpu"))
    qualified = parser.add_argument_group("qualified timing selection")
    qualified.add_argument("--reference", type=Path)
    qualified.add_argument("--profile", choices=("inference", "full"))
    qualified.add_argument("--backend", choices=("wasm", "webgpu"))
    qualified.add_argument("--existing", type=Path, nargs="*", default=[])
    qualified.add_argument("--attempt-cap", type=int)
    qualified.add_argument("--qualified-workers", type=int)
    args = parser.parse_args()
    if args.selection == "qualified":
        if not all((args.reference, args.profile, args.backend)):
            parser.error("qualified selection requires --reference, --profile and --backend")
        if args.cells or args.stage not in (None, "timing"):
            parser.error("qualified selection supports one timing cell only")
        args.attempt_cap = 5 if args.attempt_cap is None else args.attempt_cap
        args.qualified_workers = 3 if args.qualified_workers is None else args.qualified_workers
        return run_qualified(args)
    if any((args.reference, args.profile, args.backend, args.existing,
            args.attempt_cap is not None, args.qualified_workers is not None)):
        parser.error("qualified timing options require --selection qualified")
    args.stage = args.stage or "all"
    args.cells = args.cells or ("inference-wasm", "full-wasm", "full-webgpu")
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
