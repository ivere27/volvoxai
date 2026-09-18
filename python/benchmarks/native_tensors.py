#!/usr/bin/env python3
"""Reproducible native tensor handoff benchmark, including pre-refactor trees."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


def gpu_processes():
    result = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True, capture_output=True, check=True)
    return {int(value.strip()) for value in result.stdout.splitlines() if value.strip().isdigit()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl", "metal"), required=True)
    parser.add_argument("--label", choices=("before", "after"), required=True)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--profile-copies", action="store_true", help="record CUDA transfer counts separately from timing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 10 or args.warmup < 1:
        parser.error("use at least 10 measured samples and one warmup")
    root = args.source_tree.resolve()
    sys.path.insert(0, str(root / "python"))
    import numpy as np
    import volvoxai as vx
    import volvoxai._library as library

    selected_library = root / "native" / library.library_filename()
    if not selected_library.is_file():
        raise FileNotFoundError(selected_library)
    os.environ["VOLVOXAI_LIBRARY"] = str(selected_library)
    if args.profile_copies and args.backend != "cuda":
        parser.error("--profile-copies requires CUDA")
    if args.profile_copies:
        import torch
        torch.cuda.init()
    stop = threading.Event()
    contamination = []
    gpu = args.backend in ("cuda", "vulkan", "opengl")
    device_name = None
    if gpu:
        if gpu_processes() - {os.getpid()}:
            raise RuntimeError("GPU has another compute process; refusing contaminated measurement")
        device_name = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"
        ], text=True).strip()

    def monitor():
        while not stop.wait(.2):
            try:
                if gpu_processes() - {os.getpid()}:
                    contamination.append("foreign GPU compute process")
            except Exception:
                contamination.append("GPU monitoring failed")

    observer = threading.Thread(target=monitor, daemon=True) if gpu else None
    if observer:
        observer.start()
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="volvoxai-native-benchmark-") as temporary:
            directory = Path(temporary)
            for shape in ((2, 4), (256, 1024)):
                graph = {"format": "volvox-graph/v1", "dimensions": {},
                    "inputs": {"x": {"shape": shape, "dtype": "float32"}},
                    "nodes": [{"id": "add", "opType": "Add", "inputs": {"a": "x", "b": "x"},
                        "outputs": {"out": {"tensor": "y", "shape": shape, "dtype": "float32"}}, "params": {}}],
                    "outputs": ["y"]}
                (directory / "graph.json").write_text(json.dumps(graph))
                x = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 97) / 8
                options = dict(backend=args.backend, cpu_threads=args.cpu_threads)
                with ExitStack() as stack:
                    # Before/after trees have different ownership contracts.
                    # Current buffer IDs must share one native owner.
                    if hasattr(vx, "Runtime"):
                        runtime = stack.enter_context(vx.Runtime(cpu_threads=args.cpu_threads))
                        create_session = runtime.inference_session
                    else:
                        create_session = vx.InferenceSession
                    producer = stack.enter_context(create_session(directory, **options))
                    consumer = stack.enter_context(create_session(directory, **options))
                    def single_host():
                        return producer.run(x)["y"]

                    def single_resident():
                        with producer.run_tensors(x)["y"] as result:
                            return result.numpy()

                    def host_pipeline():
                        return consumer.run(producer.run(x)["y"])["y"]

                    def resident_pipeline():
                        with producer.run_tensors(x)["y"] as first:
                            with consumer.run_tensors(first)["y"] as second:
                                return second.numpy()

                    cases = (("single_host", single_host, 1), ("single_resident", single_resident, 1),
                             ("host_pipeline", host_pipeline, 2), ("resident_pipeline", resident_pipeline, 2))
                    for mode, execute, executions in cases:
                        expected = x * (2 ** executions)
                        row = {"shape": list(shape), "dtype": "float32", "mode": mode,
                               "input_bytes": x.nbytes, "model_executions_per_sample": executions}
                        for _ in range(args.warmup):
                            np.testing.assert_array_equal(execute(), expected)
                        samples = []
                        for _ in range(args.samples):
                            started = time.perf_counter_ns()
                            actual = execute()
                            samples.append((time.perf_counter_ns() - started) / 1e6)
                            np.testing.assert_array_equal(actual, expected)
                        row.update(supported=True, samples_ms=samples,
                            median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                            mean_ms=float(np.mean(samples)), outputs_exact=True)
                        if args.profile_copies:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                                execute()
                            trace = directory / "trace.json"
                            profiler.export_chrome_trace(str(trace))
                            copies, calls = {}, {}
                            for event in json.loads(trace.read_text())["traceEvents"]:
                                if event.get("cat") in ("cuda_driver", "cuda_runtime"):
                                    name = event["name"]
                                    calls[name] = calls.get(name, 0) + 1
                                if event.get("cat") != "gpu_memcpy":
                                    continue
                                direction = next((kind for kind in ("HtoD", "DtoH", "DtoD")
                                                  if kind in event["name"]), event["name"])
                                totals = copies.setdefault(direction, {"count": 0, "bytes": 0})
                                totals["count"] += 1
                                totals["bytes"] += int(event.get("args", {}).get("bytes", 0))
                            if not copies:
                                raise RuntimeError("CUDA profiler did not observe transfers")
                            row["gpu_transfers"] = copies
                            row["gpu_api_calls"] = calls
                        rows.append(row)
    finally:
        stop.set()
        if observer:
            observer.join(timeout=5)
    if gpu and (contamination or gpu_processes() - {os.getpid()}):
        raise RuntimeError("GPU measurement was contaminated; no report published")
    report = {"format": "volvoxai-native-tensor-benchmark/v1", "label": args.label,
        "backend": args.backend, "cpu_threads": args.cpu_threads,
        "warmup": args.warmup, "samples": args.samples, "python": sys.version.split()[0],
        "gpu": device_name, "foreign_gpu_processes_detected": False if gpu else None,
        "library_sha256": hashlib.sha256(selected_library.read_bytes()).hexdigest(),
        "caller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"label": args.label, "backend": args.backend,
        "rows": [{k: v for k, v in row.items() if k != "samples_ms"} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
