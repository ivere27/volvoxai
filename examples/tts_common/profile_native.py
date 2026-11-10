#!/usr/bin/env python3
"""Capture one complete live TTS request using the public profiling service.

Capture overhead is deliberately separate from benchmark_native.py latency.
Stage collectors use their existing runtime owners; both stages receive live
inputs on every request, including the captured request.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np
from google.protobuf.json_format import MessageToDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
from examples.tts_common.benchmark_native import NativePipeline, compare_outputs
from examples.tts_common.artifacts import sha256
from volvoxai._diagnostics import Trace, trace_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads <= 0 or args.warmups < 0:
        parser.error("threads must be positive and warmups nonnegative")
    if args.out.suffix != ".json":
        parser.error("out must end in .json to keep waveform artifacts separate")
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    case = manifest["cases"][0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runner = NativePipeline(root, manifest["volvox"], args.backend,
                            args.threads, args.library, 300)
    try:
        with np.load(root / case["inputs"]) as arrays:
            runner.prepare(dict(arrays))
        for _ in range(1 + args.warmups):
            untraced_outputs, _ = runner.run()
        untraced_path = args.out.with_suffix(".untraced.outputs.npz")
        np.savez(untraced_path, **untraced_outputs)
        traces = {}
        with contextlib.ExitStack() as scopes:
            for name, engine in runner.stages:
                runtime = next(ref.runtime_id for _, ref in engine.resources
                               if isinstance(ref, engine.pb.RuntimeRef))
                traces[name] = scopes.enter_context(Trace(engine.host, runtime,
                    trace_options(detail="nodes", device_timing=args.backend != "cpu",
                                  memory=True, execution_plans=True,
                                  capacity_bytes=64 * 1024 * 1024), engine.resources))
            outputs, timings = runner.run()
        with np.load(root / case["reference"]) as arrays:
            validation = compare_outputs(outputs, dict(arrays), deterministic=True,
                audio_output="audio", rtol=1e-4, atol=1e-5,
                waveform=case.get("validation", {}).get("waveform"),
                report_waveform_differences=True)
        report = {"format": "volvoxai-tts-native-profile/v1", "backend": args.backend,
                  "threads": args.threads, "manifest_sha256": sha256(args.manifest),
                  "caller_sha256": sha256(Path(__file__)), "runtime": runner.metadata,
                  "untraced_outputs_sha256": sha256(untraced_path),
                  "timings_ms": timings, "validation": validation,
                  "route": runner.route, "stages": {}}
        for name, trace in traces.items():
            report["stages"][name] = {
                "info": MessageToDict(trace.info),
                "events": [MessageToDict(event) for event in trace.events],
                "plans": [MessageToDict(plan) for plan in trace.plans],
                "resources": [MessageToDict(sample) for sample in trace.resource_snapshots],
                "operator_summary": MessageToDict(trace.summary("operator")),
                "activity_summary": MessageToDict(trace.summary("activity")),
            }
            print(name, trace.table("operator"), sep="\n")
        np.savez(args.out.with_suffix(".outputs.npz"), **outputs)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
