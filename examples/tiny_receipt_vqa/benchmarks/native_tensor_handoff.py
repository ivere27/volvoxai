#!/usr/bin/env python3
"""Compare NumPy and retained native tensors using frozen receipt inputs.

Run each source tree in a separate process. Timing includes complete requests,
explicit host reads and tensor release, but excludes model loading and checks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--label", choices=("before", "after"), required=True)
    parser.add_argument("--model", choices=("digit", "vqa"), required=True)
    parser.add_argument("--backend", choices=("cuda", "vulkan", "opengl"), required=True)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--variants", nargs="+", choices=("fp32", "int8"), default=["fp32", "int8"])
    parser.add_argument("--modes", nargs="+", choices=("numpy", "resident", "stateful"), default=["numpy", "resident"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-api", action="store_true", help="count public calls separately from latency sampling")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 1:
        parser.error("repeat and warmup must be positive")
    source = args.source_tree.resolve()
    sys.path.insert(0, str(source / "python"))
    import numpy as np
    import volvoxai as vx
    import receipt_proto_benchmark as receipt
    # The inference-only native release, measured directly. Python installs the
    # full library, so name this build's file instead of asking for a profile.
    library = source / "native" / "libvolvoxai-lite.so"
    os.environ["VOLVOXAI_LIBRARY"] = str(library)
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    data = args.data_root.resolve()
    stop = threading.Event()
    contaminated = []

    def foreign_processes():
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], text=True)
        return {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()} - {os.getpid()}

    if foreign_processes():
        raise RuntimeError("GPU has another compute owner")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
        "--format=csv,noheader"], text=True).strip()

    def observe():
        while not stop.wait(.2):
            try:
                if foreign_processes():
                    contaminated.append(True)
            except Exception:
                contaminated.append(True)

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    rows = []
    cases = []
    model_hashes = {}
    for item in spec["cases"]:
        raw = (data / item["pixels"]).read_bytes()
        assert receipt.digest(raw) == item["sha256"]
        cases.append(dict(item, pixels=np.frombuffer(raw, dtype=np.float32).reshape(1, 1, 320, 672)))

    class Runner:
        native = True

        def __init__(self, variant, mode):
            self.contracts = spec["models"][args.model]["variants"][variant]
            self.resident, self.sessions, self.live = mode != "numpy", {}, {}
            self.stateful, self.decoder_started = mode == "stateful", False
            try:
                for role, contract in self.contracts.items():
                    for key in ("graph", "weights"):
                        path = contract[key]
                        actual = receipt.digest((data / path).read_bytes())
                        assert actual == spec["file_hashes"][path]
                        model_hashes[f"{variant}/{role}/{key}"] = actual
                    self.sessions[role] = vx.InferenceSession(data / contract["graph"],
                        backend=args.backend, profile="inference", cpu_threads=1)
            except BaseException:
                self.close()
                raise

        def run(self, role, feed, *, need_cache=True):
            contract = self.contracts[role]
            inputs = {contract["inputs"][name]: value for name, value in feed.items()}
            names = {name: contract["outputs"][name] for name in receipt.NEEDED_OUTPUTS[role]}
            reuse, feedback = [], {}
            if self.stateful and role == "decoder":
                if self.decoder_started:
                    for name in feed:
                        actual = contract["inputs"][name]
                        if name.startswith("past_"):
                            feedback[actual] = contract["outputs"][name.replace("past_", "present_", 1)]
                            del inputs[actual]
                        elif name.startswith("cross_") or name == "memory_padding_mask":
                            reuse.append(actual)
                            del inputs[actual]
                selected = [names["logits"]]
            else:
                selected = list(names.values())
            started = time.perf_counter_ns()
            if not self.resident:
                result = self.sessions[role].run(inputs, output_names=list(names.values()))
                output = receipt.read_needed_outputs(role, lambda name: result[names[name]], need_cache=need_cache)
            else:
                result = self.sessions[role].run_tensors(inputs, output_names=selected,
                    reuse_inputs=reuse, feedback=feedback)
                if role == "decoder":
                    self.decoder_started = True
                previous = self.live.get(role)
                self.live[role] = result

                def read(name):
                    if self.stateful and role == "decoder" and name.startswith("present_"):
                        return names[name]  # The next call names the context-owned output.
                    tensor = result[names[name]]
                    # Activations and K/V stay on the GPU. Control values and
                    # greedy token selection require explicit small host reads.
                    if name.startswith(("cross_k_", "cross_v_", "present_k_", "present_v_")):
                        return tensor
                    return tensor.numpy()

                try:
                    output = receipt.read_needed_outputs(role, read, need_cache=need_cache)
                finally:
                    self.release_outputs(previous)
            return output, (time.perf_counter_ns() - started) / 1e6

        @staticmethod
        def release_outputs(tensors):
            if tensors is not None:
                tensors.close()

        def release_request(self):
            for tensors in self.live.values():
                self.release_outputs(tensors)
            self.live.clear()
            self.decoder_started = False

        def close(self):
            self.release_request()
            for session in self.sessions.values():
                session.close()

    try:
        for variant in args.variants:
            expected = spec["expected"][f"{args.model}/{args.backend}/inference/{variant}"]
            for mode in args.modes:
                row = {"variant": variant, "mode": mode}
                runner = Runner(variant, mode)
                samples = []
                try:
                    for case in cases:
                        for iteration in range(-args.warmup, args.repeat):
                            started = time.perf_counter_ns()
                            try:
                                output, metrics = receipt.request(runner, args.model, case)
                            finally:
                                runner.release_request()
                            metrics["request_ms"] = (time.perf_counter_ns() - started) / 1e6
                            assert receipt.signature(output) == receipt.signature(expected[str(case["index"])]), "frozen output signature differs"
                            if iteration >= 0:
                                samples.append(dict(metrics, case_index=case["index"]))
                    row.update(supported=True, samples=samples,
                        latency=receipt.summarize([sample["request_ms"] for sample in samples]),
                        by_case={str(case["index"]): receipt.summarize([
                            sample["request_ms"] for sample in samples if sample["case_index"] == case["index"]])
                            for case in cases},
                        all_outputs_match_reference=True)
                    if args.profile_api:
                        class ProfiledEngine:
                            def __init__(self, engine):
                                self.engine, self.calls = engine, {}

                            def __getattr__(self, name):
                                method = getattr(self.engine, name)

                                def call(*args, **kwargs):
                                    start = time.perf_counter_ns()
                                    try:
                                        return method(*args, **kwargs)
                                    finally:
                                        totals = self.calls.setdefault(name, {"count": 0, "ms": 0.0})
                                        totals["count"] += 1
                                        totals["ms"] += (time.perf_counter_ns() - start) / 1e6
                                return call

                        profiled = {role: ProfiledEngine(session._engine) for role, session in runner.sessions.items()}
                        for role, session in runner.sessions.items():
                            session._engine = profiled[role]
                        row["api_profile"] = []
                        for case in cases:
                            for engine in profiled.values():
                                engine.calls.clear()
                            output, metrics = receipt.request(runner, args.model, case)
                            runner.release_request()
                            assert receipt.signature(output) == receipt.signature(expected[str(case["index"])])
                            row["api_profile"].append({"case_index": case["index"], "metrics": metrics,
                                "calls": {role: dict(engine.calls) for role, engine in profiled.items()}})
                        for role, session in runner.sessions.items():
                            session._engine = profiled[role].engine
                    rows.append(row)
                finally:
                    runner.close()
    finally:
        stop.set()
        observer.join(timeout=5)
    if contaminated or foreign_processes():
        raise RuntimeError("GPU measurement contaminated; no report published")
    report = {"format": "volvoxai-receipt-native-tensors/v1", "label": args.label,
        "model": args.model, "backend": args.backend, "profile": "inference", "cpu_threads": 1,
        "gpu": gpu, "foreign_gpu_processes_detected": False,
        "case_indices": [case["index"] for case in cases], "warmup": args.warmup, "repeat": args.repeat,
        "spec_sha256": receipt.digest(spec_bytes), "library_sha256": receipt.digest(library.read_bytes()),
        "caller_sha256": receipt.digest(Path(__file__).read_bytes()), "model_hashes": model_hashes,
        "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"label": args.label, "model": args.model, "backend": args.backend,
        "rows": [{key: value for key, value in row.items() if key != "samples"} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
