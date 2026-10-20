#!/usr/bin/env python3
"""Prepare/run exactly one native capacity worker, then retire all owners.

Small-model measurement can yield the shared device between every invocation.
Only task-owned temporary files/cache directories are removed. There is no
service action, model export/download, or loop launching the next worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from measure_native_baseline import edge_samples


def require(value, message):
    if not value: raise RuntimeError(message)


def identity(path):
    path = path.resolve(); digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""): digest.update(data)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def swap():
    rows = dict(row.split() for row in Path("/proc/vmstat").read_text().splitlines())
    return {key: int(rows[key]) for key in ("pswpin", "pswpout")}


def reserve_vulkan_allocations(plan, arena_mib):
    """Preflight configured fixed allocations before the validation device opens.

    The 128 MiB legacy base and 1 MiB scratch are inside the arena. Staging
    is a separate 32 MiB allocation. Buffers belong to each VxEngineState;
    only the device/pipelines are shared. Charge the whole fixed pair in each
    relevant space for external safety, without claiming a space partition.
    The returned compiler proof later includes queried allocation sizes.
    """
    arena = arena_mib << 20; staging = 32 << 20
    reserve = arena + staging
    for field, consumers in (("compile_probe_scheduling_estimate", 1), ("scheduling_estimate", plan["live_contexts"])):
        estimate = plan[field]
        estimate["host_scheduling_bytes"] += consumers * reserve
        estimate["device_scheduling_bytes"] += consumers * reserve
        estimate["per_context_scheduling_bytes"] += reserve
        estimate["vulkan_fixed_pair_extra_reserve_bytes"] = consumers * reserve
    plan["native_backend_environment"] = {"VOLVOX_VULKAN_MB": str(arena_mib)}
    plan["vulkan_fixed_allocation_reserve"] = dict(arena_requested_bytes=arena,
        graph_base_included_bytes=128 << 20, graph_scratch_included_bytes=1 << 20,
        staging_requested_bytes=staging, per_engine_state=True,
        usable_graph_region_before_alignment_bytes=arena - (129 << 20),
        scope="External heuristic reserves each configured fixed pair in both host/device spaces before validation. The 1GiB process/driver allowance remains extra; native compilation checks real allocation alignment, maximum tensor/storage caps and adds queried fixed allocations to its coarse provider bound. Admission-proof workers consume that complete bound without adding this heuristic again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("smollm2-135m", "qwen3-0.6b"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl"), required=True)
    parser.add_argument("--profile", choices=("full", "inference"), default="full")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--contexts", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--stage", choices=("proof", "memory", "timing"), required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--admission-proof", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--vulkan-arena-mib", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args(); root = args.root.resolve(); out = args.out.resolve()
    require(root.name.startswith("volvoxai-remaining-validation-"), "unique task-owned source root required")
    require(out.is_relative_to(root / "build/llm"), "output must belong to the task-owned root")
    require(not out.exists(), "fresh worker directory required")
    require(args.profile == "full" or args.backend == "cpu", "inference profile admits only CPU")
    require(args.timeout_seconds > 0, "positive timeout required")
    require(args.vulkan_arena_mib is None or args.backend == "vulkan" and args.vulkan_arena_mib > 129,
            "an explicit Vulkan arena must fit the included legacy base and scratch")
    out.mkdir(parents=True)
    helper = root / "tools/llm_baseline/measure_native_capacity.py"
    library = args.library or root / "native" / ("libvolvoxai-lite.so" if args.profile == "inference" else "libvolvoxai.so")
    bridge = args.bridge or root / "native/libsynurang_module_host.so"
    package = args.package or root / "build/llm/packages" / args.model
    reference = args.reference or root / "build/llm/reference-v2" / args.model
    graph = args.graph
    if not graph and not args.plan:
        candidates = [root / "build/llm/c-stage/packages" / args.model / part / "graph.json" for part in ("execution", "cached")]
        graph = next((p for p in candidates if p.is_file()), None)
        require(graph is not None, "cached graph is absent; provide --graph")
    pins = {name: identity(file) for name, file in (("wrapper", Path(__file__)), ("helper", helper), ("library", library), ("bridge", bridge))}
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(root / ".testdeps"), str(root / "python"), str(root / "runtime/generated/python")))
    environment.pop("VOLVOX_VULKAN_MB", None)
    backend_environment = {"VOLVOX_VULKAN_MB": str(args.vulkan_arena_mib)} if args.vulkan_arena_mib is not None else {}
    environment.update(backend_environment)
    policy = dict(format="volvox-native-capacity-worker-policy/v1", pins=pins,
                  model=args.model, profile=args.profile, backend=args.backend, capacity=args.capacity,
                  live_contexts=args.contexts, stage=args.stage, owner_lifecycle="One fresh worker, then all owners drain before returning the slot.",
                  backend_environment=backend_environment,
                  timing_observers=False, timing_idle_edges=dict(samples=3, maximum_cpu_busy_percent=5, maximum_gpu_utilization_percent=5),
                  system_swap_delta_required=0, driver_cache_scope="Fresh task-owned CUDA/OpenGL/Mesa/XDG cache directories for this worker.",
                  timeout_seconds=args.timeout_seconds, existing_service_actions=0)
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    report = dict(format="volvox-native-capacity-worker/v1", status="RUNNING", policy=identity(out / "policy.json"),
                  system_swap_before=swap(), existing_service_actions=0)
    child = None; interrupted = []
    def stop(signum, _frame):
        interrupted.append(signum)
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, stop)
    try:
        plan = args.plan or out / "plan.json"
        if not args.plan:
            prepare = [sys.executable, str(helper), "--prepare", "--source-root", str(root), "--model", args.model,
                       "--profile", args.profile, "--backend", args.backend, "--capacity", str(args.capacity),
                       "--contexts", str(args.contexts), "--graph", str(graph), "--package", str(package),
                       "--reference", str(reference), "--library", str(library), "--bridge", str(bridge), "--out", str(plan)]
            with (out / "prepare.log").open("w") as log:
                child = subprocess.Popen(prepare, cwd=root, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    require(child.wait(timeout=120) == 0 and not interrupted, "plan preparation failed or interrupted")
                except subprocess.TimeoutExpired:
                    raise RuntimeError("plan preparation timed out")
        prepared = json.loads(plan.read_text())
        if not args.plan and args.vulkan_arena_mib is not None:
            reserve_vulkan_allocations(prepared, args.vulkan_arena_mib)
            plan.write_text(json.dumps(prepared, indent=2) + "\n")
        require(prepared.get("native_backend_environment", {}) == backend_environment,
                "native backend environment differs from the prepared plan")
        require(prepared["pins"]["model"] == args.model and prepared["pins"]["backend"] == args.backend and
                prepared["pins"]["profile"] == args.profile and prepared["capacity"] == args.capacity and
                prepared["live_contexts"] == args.contexts, "selected plan cell differs")
        report["plan"] = identity(plan)
        if args.stage == "timing":
            report["before_idle_samples"] = edge_samples()
            require(all(row["cpu_busy_percent"] <= 5 and all(d["utilization_percent"] <= 5 for d in row["gpu_devices"])
                        for row in report["before_idle_samples"]), "before-worker idle edge failed")
        with tempfile.TemporaryDirectory(prefix="temporary-owner-", dir=out) as scratch:
            for name in ("cuda-cache", "gl-cache", "mesa-cache", "xdg-cache"):
                Path(scratch, name).mkdir()
            worker_environment = dict(environment, CUDA_CACHE_PATH=str(Path(scratch, "cuda-cache")),
                                      __GL_SHADER_DISK_CACHE_PATH=str(Path(scratch, "gl-cache")),
                                      MESA_SHADER_CACHE_DIR=str(Path(scratch, "mesa-cache")),
                                      XDG_CACHE_HOME=str(Path(scratch, "xdg-cache")))
            owner_tmp = Path(scratch, "snapshots"); owner_tmp.mkdir()
            worker_environment["TMPDIR"] = str(owner_tmp)
            command = [sys.executable, str(helper), "--plan", str(plan), "--mode", args.stage,
                       "--tmpdir", str(owner_tmp), "--out", str(out / "worker.json")]
            if args.admission_proof: command += ["--admission-proof", str(args.admission_proof)]
            report["command"] = command; start = time.monotonic_ns()
            with (out / "worker.log").open("w") as log:
                child = subprocess.Popen(command, cwd=root, env=worker_environment, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try: code = child.wait(timeout=args.timeout_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    raise RuntimeError("owned worker timed out")
            report["elapsed_ns"] = time.monotonic_ns() - start; report["exit_code"] = code
            require(not interrupted, "worker interrupted")
            require(code == 0 and (out / "worker.json").is_file(), "owned worker failed")
            receipt = json.loads((out / "worker.json").read_text()); report["receipt"] = identity(out / "worker.json")
            expected_status = "PROOF_CAPTURED" if args.stage == "proof" else "PASS"
            require(receipt["status"] in (expected_status, "SKIPPED_CONSERVATIVE_RESOURCE_BOUND"), "worker receipt does not qualify the requested stage")
            require(not any(receipt["system_swap_delta"].values()), "worker swap invalidated")
            report["worker_status"] = receipt["status"]
            report["private_snapshot_remaining_after_owner_close"] = receipt["private_tmp_remaining"]
            require(not receipt["private_tmp_remaining"], "private immutable snapshot remained after owner close")
        if args.stage == "timing":
            report["after_idle_samples"] = edge_samples()
            require(all(row["cpu_busy_percent"] <= 5 and all(d["utilization_percent"] <= 5 for d in row["gpu_devices"])
                        for row in report["after_idle_samples"]), "after-worker idle edge failed")
        require(pins == {name: identity(file) for name, file in (("wrapper", Path(__file__)), ("helper", helper), ("library", library), ("bridge", bridge))}, "wrapper/helper/artifact changed")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"; report["error"] = repr(error)
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        # The TemporaryDirectory scope handles owned snapshots and caches;
        # this also covers interruption while exiting a native call.
        for owned in out.glob("temporary-owner-*"):
            if owned.is_dir(): shutil.rmtree(owned)
        report["system_swap_after"] = swap()
        report["system_swap_delta"] = {key: report["system_swap_after"][key] - value
                                       for key, value in report["system_swap_before"].items()}
        if any(report["system_swap_delta"].values()): report["status"] = "FAIL"; report["swap_invalidated"] = True
        report["temporary_owned_paths_remaining"] = [str(p) for p in out.glob("temporary-owner-*")]
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], worker_status=report.get("worker_status"), out=str(out))))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
