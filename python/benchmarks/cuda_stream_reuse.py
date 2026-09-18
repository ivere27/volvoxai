#!/usr/bin/env python3
"""Qualify native CUDA buffer reuse against unrelated stream work.

Run this identical caller against baseline and candidate source trees. The
candidate must pass --require-isolation; the baseline records its waits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


@contextmanager
def driver_call_counts():
    """Observe Driver API calls directly; Kineto may omit this API domain.

    This instrumentation is outside every latency measurement. CUPTI's callback
    ABI is documented at https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html.
    """
    try:
        cupti = ctypes.CDLL("libcupti.so")
    except OSError:
        spec = importlib.util.find_spec("nvidia.cuda_cupti")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("CUPTI is required to verify native driver calls") from None
        directory = Path(next(iter(spec.submodule_search_locations))) / "lib"
        cupti = ctypes.CDLL(str(next(directory.glob("libcupti.so.*"))))

    class CallbackPrefix(ctypes.Structure):
        _fields_ = [("site", ctypes.c_int), ("name", ctypes.c_char_p)]

    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_uint32, ctypes.c_void_p)
    calls = {}

    @callback_type
    def on_call(_userdata, domain, _callback_id, data):
        if domain == 1:  # CUPTI_CB_DOMAIN_DRIVER_API
            entry = ctypes.cast(data, ctypes.POINTER(CallbackPrefix)).contents
            if entry.site == 0:  # CUPTI_API_ENTER
                name = entry.name.decode("ascii")
                calls[name] = calls.get(name, 0) + 1

    cupti.cuptiSubscribe.argtypes = [ctypes.POINTER(ctypes.c_void_p), callback_type, ctypes.c_void_p]
    cupti.cuptiEnableDomain.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
    cupti.cuptiUnsubscribe.argtypes = [ctypes.c_void_p]

    def checked(status):
        if status:
            raise RuntimeError(f"CUPTI callback collection failed with status {status}")

    subscriber = ctypes.c_void_p()
    checked(cupti.cuptiSubscribe(ctypes.byref(subscriber), on_call, None))
    try:
        checked(cupti.cuptiEnableDomain(1, subscriber, 1))
        yield calls
    finally:
        checked(cupti.cuptiUnsubscribe(subscriber))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--sleep-cycles", type=int, default=300_000_000)
    parser.add_argument("--require-isolation", action="store_true")
    parser.add_argument("--external-completion", action="store_true",
                        help="also compare ordinary DLPack release with scoped consumer completion")
    args = parser.parse_args()
    if args.samples < 3 or args.sleep_cycles < 1:
        parser.error("samples must be at least 3 and sleep-cycles must be positive")
    root = args.source_tree.resolve()
    sys.path.insert(0, str(root / "python"))
    # Initialize the optional CUDA integration before the native driver.
    import torch
    import numpy as np
    import volvoxai as vx
    import volvoxai._library as library

    if not torch.cuda.is_available():
        raise RuntimeError("A working physical CUDA device is required")
    import os
    def check_idle():
        result = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], text=True)
        others = {int(line.strip()) for line in result.splitlines() if line.strip().isdigit()} - {os.getpid()}
        if others:
            raise RuntimeError("Another GPU compute process is active; refusing a contaminated measurement")
    check_idle()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
        "--format=csv,noheader"], text=True).strip()
    report = {"format": "volvoxai-cuda-stream-reuse/v1", "gpu": gpu,
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "samples": args.samples, "sleep_cycles": args.sleep_cycles,
        "caller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with tempfile.TemporaryDirectory(prefix="volvox-stream-") as temporary:
        directory = Path(temporary)
        selected = root / "native" / library.library_filename()
        os.environ["VOLVOXAI_LIBRARY"] = str(selected)
        profile_result = report["library"] = {
            "library_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(), "rows": []}
        for shape in ((2, 4), (256, 1024)):
            graph = {"format": "volvox-graph/v1", "dimensions": {},
                "inputs": {"x": {"shape": shape, "dtype": "float32"}},
                "nodes": [{"id": "add", "opType": "Add", "inputs": {"a": "x", "b": "x"},
                    "outputs": {"out": {"tensor": "y", "shape": shape, "dtype": "float32"}}, "params": {}}],
                "outputs": ["y"]}
            (directory / "graph.json").write_text(json.dumps(graph))
            x = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 97) / 8
            with vx.Runtime() as runtime, \
                    runtime.inference_session(directory, backend="cuda") as producer, \
                    runtime.inference_session(directory, backend="cuda") as consumer:
                retained = producer.run_tensors(x)["y"]
                try:
                    cases = {
                        "pooled_reuse": (lambda: producer.run_tensors({}, reuse_inputs=["x"]), x * 2),
                        "host_input": (lambda: producer.run_tensors(x), x * 2),
                        "native_input": (lambda: consumer.run_tensors(retained), x * 4),
                        "host_readback": (lambda: retained.numpy(), x * 2),
                    }
                    def verify(result, expected):
                        if isinstance(result, np.ndarray):
                            np.testing.assert_array_equal(result, expected)
                        else:
                            try:
                                np.testing.assert_array_equal(result["y"].numpy(), expected)
                            finally:
                                result.close()
                    for execute, expected in cases.values():
                        for _ in range(8):
                            verify(execute(), expected)
                    for scenario in ("idle", "independent_stream", "legacy_default_stream"):
                        background = torch.cuda.default_stream() if scenario == "legacy_default_stream" else torch.cuda.Stream()
                        for name, (execute, expected) in cases.items():
                            samples, unfinished = [], 0
                            check_idle()
                            for _ in range(args.samples):
                                torch.cuda.synchronize()
                                completed = torch.cuda.Event()
                                if scenario != "idle":
                                    with torch.cuda.stream(background):
                                        torch.cuda._sleep(args.sleep_cycles)
                                        completed.record()
                                started = time.perf_counter_ns()
                                result = execute()
                                samples.append((time.perf_counter_ns() - started) / 1e6)
                                if scenario != "idle":
                                    unfinished += not completed.query()
                                    completed.synchronize()
                                verify(result, expected)
                            row = {"shape": list(shape), "case": name, "scenario": scenario,
                                "p50_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
                                "samples_ms": samples, "unrelated_work_still_pending": unfinished,
                                "outputs_exact": True}
                            profile_result["rows"].append(row)
                            if args.require_isolation and scenario != "idle" and not unfinished:
                                raise AssertionError(f"{shape}/{name} waited for all unrelated {scenario} work")

                    with driver_call_counts() as calls:
                        result = consumer.run_tensors(retained)
                        result.close()
                    if not any("MemcpyDtoD" in name for name in calls):
                        raise AssertionError("The profiler did not observe native device copies")
                    profile_result.setdefault("driver_calls", []).append({"shape": list(shape), "calls": calls})
                    if args.require_isolation:
                        assert not any("CtxSynchronize" in name or "DeviceSynchronize" in name for name in calls), calls
                        assert not any("MemAlloc" in name or "MemFree" in name for name in calls), calls
                    if args.external_completion:
                        consumer_stream = torch.cuda.Stream()
                        def begin_access(scoped):
                            scope = retained.torch_access(streams=[consumer_stream]) if scoped else None
                            view = scope.__enter__() if scope else torch.from_dlpack(retained)
                            with torch.cuda.stream(consumer_stream):
                                view.add_(0)
                            return scope, view
                        for scoped in (False, True):
                            # Warm events and framework kernels outside timing.
                            for _ in range(4):
                                scope, view = begin_access(scoped)
                                if scope:
                                    scope.__exit__(None, None, None)
                                del view
                            for scenario in ("idle", "independent_stream", "legacy_default_stream"):
                                background = torch.cuda.default_stream() if scenario == "legacy_default_stream" else torch.cuda.Stream()
                                samples, unfinished = [], 0
                                check_idle()
                                for _ in range(args.samples):
                                    torch.cuda.synchronize()
                                    scope, view = begin_access(scoped)
                                    completed = torch.cuda.Event()
                                    if scenario != "idle":
                                        with torch.cuda.stream(background):
                                            torch.cuda._sleep(args.sleep_cycles)
                                            completed.record()
                                    started = time.perf_counter_ns()
                                    if scope:
                                        scope.__exit__(None, None, None)
                                    del view
                                    samples.append((time.perf_counter_ns() - started) / 1e6)
                                    if scenario != "idle":
                                        unfinished += not completed.query()
                                        completed.synchronize()
                                    np.testing.assert_array_equal(retained.numpy(), x * 2)
                                row = {"shape": list(shape), "case": "scoped_completion" if scoped else "ordinary_deleter",
                                    "scenario": scenario, "p50_ms": float(np.median(samples)),
                                    "p95_ms": float(np.percentile(samples, 95)), "samples_ms": samples,
                                    "unrelated_work_still_pending": unfinished, "outputs_exact": True}
                                profile_result.setdefault("external_completion", []).append(row)
                                if args.require_isolation and scenario != "idle":
                                    assert (unfinished > 0) if scoped else (unfinished == 0), row
                            scope, view = begin_access(scoped)
                            with driver_call_counts() as calls:
                                if scope:
                                    scope.__exit__(None, None, None)
                                del view
                            profile_result.setdefault("external_driver_calls", []).append({
                                "shape": list(shape), "scoped": scoped, "calls": calls})
                            drains = sum(count for name, count in calls.items() if "CtxSynchronize" in name)
                            assert drains == (0 if scoped else 1), calls
                            if scoped:
                                assert any("EventRecord" in name for name in calls), calls
                                assert not any("EventCreate" in name or "EventDestroy" in name or
                                               "MemAlloc" in name or "MemFree" in name for name in calls), calls
                finally:
                    retained.close()
            check_idle()
    report["status"] = "pass"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "pass", "gpu": gpu}))


if __name__ == "__main__":
    main()
