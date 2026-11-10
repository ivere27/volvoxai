#!/usr/bin/env python3
"""Compare complete quantized TTS requests with their own engine's FP32 result.

Quantization may change natural waveform length. Length differences are reported
without alignment, truncation or a pointwise accuracy claim. Fixed Volvox extent
refusals remain failures, with no fabricated complete-request latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tts_common.artifacts import sha256
from examples.tts_common.benchmark_native import (NativePipeline, Ort, audio_seconds,
                                                 load_npz, summarize)


def peak_rss_bytes():
    if sys.platform.startswith("linux"):
        # VmHWM belongs to this executable's address space. Unlike ru_maxrss,
        # it cannot retain a launcher's pre-exec high-water allocation.
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def output_sha256(audio):
    return hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()


def maximum_errors(measurements, comparison):
    return {name: max(row[comparison][name] for row in measurements)
            for name in ("max_absolute_error", "rmse")
            if all(name in row[comparison] for row in measurements)}


def summarize_comparison(measurements, comparison):
    values = [row[comparison] for row in measurements]
    result = {**values[-1], **maximum_errors(measurements, comparison),
              "numerical_parity": all(value["numerical_parity"] for value in values)}
    if "strict_elementwise_pass" in result:
        result["strict_elementwise_pass"] = all(value["strict_elementwise_pass"] for value in values)
    snr = [value["snr_db"] for value in values if value.get("snr_db") is not None]
    if snr:
        result["snr_db"] = min(snr)
    return result


def waveform_metrics(actual, reference, criterion=None):
    actual, reference = np.asarray(actual), np.asarray(reference)
    for name, waveform in (("candidate", actual), ("reference", reference)):
        if waveform.dtype != np.float32 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError(f"a finite nonempty FP32 {name} waveform is required")
    row = {"actual_shape": list(actual.shape), "reference_shape": list(reference.shape),
           "same_shape": actual.shape == reference.shape, "finite": True}
    if actual.shape != reference.shape:
        return {**row, "numerical_parity": False,
                "reason": "different lengths; no alignment, truncation or pointwise comparison"}
    error = np.asarray(actual, np.float64) - np.asarray(reference, np.float64)
    maximum = float(np.max(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    power = float(np.mean(np.asarray(reference, np.float64) ** 2))
    strict = bool(np.all(np.abs(error) <= 1e-5 + 1e-4 * np.abs(reference)))
    accepted = strict if criterion is None else (
        maximum <= criterion["max_absolute_error"] and rmse <= criterion["max_rmse"])
    return {**row, "max_absolute_error": maximum, "rmse": rmse,
            "snr_db": 10 * math.log10(power / (rmse * rmse)) if power > 0 and rmse > 0 else None,
            "strict_elementwise_pass": strict, "numerical_parity": accepted,
            "criterion": criterion or {"rtol": 1e-4, "atol": 1e-5}}


def make_runner(manifest, root, args, temporary):
    if args.engine == "ort":
        runner_tmp = Path(temporary) / f"ort-{time.time_ns()}"
        runner_tmp.mkdir(parents=True, exist_ok=True)
        return Ort(root, manifest["onnx"], args.ort_provider, args.threads,
                   json.loads(args.ort_provider_options), runner_tmp,
                   allow_cpu_nodes=args.ort_allow_cpu_nodes)
    return NativePipeline(root, manifest["volvox"], args.backend, args.threads,
                          args.library, 300)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fp32-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=("ort", "volvox"), required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--ort-provider", default="CPUExecutionProvider")
    parser.add_argument("--ort-provider-options", default="{}")
    parser.add_argument("--ort-allow-cpu-nodes", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.warmups < 0 or args.repeats <= 0:
        parser.error("positive threads/repeats and nonnegative warmups are required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifests = {"candidate": json.loads(args.manifest.read_text()),
                 "fp32": json.loads(args.fp32_manifest.read_text())}
    roots = {"candidate": args.manifest.resolve().parent, "fp32": args.fp32_manifest.resolve().parent}
    report = {"format": "volvoxai-tts-quantized-benchmark/v1", "status": "failed",
              "engine": args.engine, "backend": args.backend if args.engine == "volvox" else args.ort_provider,
              "manifest_sha256": sha256(args.manifest), "fp32_manifest_sha256": sha256(args.fp32_manifest),
              "policy": manifests["candidate"].get("quantization"),
              "settings": {"threads": args.threads, "warmups": args.warmups, "repeats": args.repeats},
              "timing_contract": "prepared host input through completed host output; every stage executes each request",
              "quality_limit": "Only prepared requests evaluated; authored static variants calibrated on these inputs; no speech-quality qualification",
              "memory": {"scope": "Process high-water RSS includes simultaneously loaded FP32 and candidate sessions, setup, host outputs and diagnostics; not per-model memory or GPU device memory",
                         "source": "Linux VmHWM with platform getrusage fallback",
                         "peak_rss_before_sessions_bytes": peak_rss_bytes()},
              "runtime": {}, "routes": {}, "cases": []}
    fp32_cases = {case["id"]: case for case in manifests["fp32"]["cases"]}
    runners = {}
    try:
        if not manifests["candidate"]["cases"]:
            raise ValueError("at least one prepared request is required")
        for case in manifests["candidate"]["cases"]:
            original = fp32_cases[case["id"]]
            validation = original.get("validation", {})
            if validation.get("reference_engine") != "onnxruntime":
                raise ValueError("ONNX comparison requires an explicit ONNX Runtime reference; prepare with validation enabled")
            if validation.get("deterministic") is not True or case.get("validation", {}).get("deterministic") is not True:
                raise ValueError("quantized waveform comparison requires deterministic prepared requests")
        with tempfile.TemporaryDirectory(prefix="volvox-tts-quantized-") as temporary:
            try:
                for name in ("fp32", "candidate"):
                    runners[name] = make_runner(manifests[name], roots[name], args, temporary)
                    report["runtime"][name] = runners[name].metadata
                report["memory"]["peak_rss_after_sessions_bytes"] = peak_rss_bytes()
                for case in manifests["candidate"]["cases"]:
                    original = fp32_cases[case["id"]]
                    inputs = load_npz(roots["candidate"] / case["inputs"])
                    original_inputs = load_npz(roots["fp32"] / original["inputs"])
                    if inputs.keys() != original_inputs.keys() or any(
                            inputs[key].dtype != original_inputs[key].dtype or
                            not np.array_equal(inputs[key], original_inputs[key]) for key in inputs):
                        raise ValueError("candidate and FP32 must receive identical original request inputs")
                    for runner in runners.values():
                        runner.prepare(inputs)
                    fp32_outputs, _ = runners["fp32"].run()
                    reference = load_npz(roots["fp32"] / original["reference"])["audio"]
                    criterion = original.get("validation", {}).get("waveform")
                    def validate(outputs, timing):
                        return {"request_ms": timing["request_ms"],
                            "output_sha256": output_sha256(outputs["audio"]),
                            "against_own_fp32": waveform_metrics(outputs["audio"], fp32_outputs["audio"], criterion),
                            "against_original_ort": waveform_metrics(outputs["audio"], reference, criterion)}
                    outputs, first = runners["candidate"].run()
                    expected_shape = outputs["audio"].shape
                    first_validation = validate(outputs, first)
                    warmup_requests = []
                    for _ in range(args.warmups):
                        outputs, timing = runners["candidate"].run()
                        if outputs["audio"].shape != expected_shape:
                            raise ValueError("candidate waveform shape changed during warmup")
                        warmup_requests.append(validate(outputs, timing))
                    samples = []
                    measured_requests = []
                    for index in range(args.repeats):
                        outputs, timing = runners["candidate"].run()
                        if outputs["audio"].shape != expected_shape or not np.isfinite(outputs["audio"]).all():
                            raise ValueError("candidate waveform shape changed or became nonfinite across identical requests")
                        samples.append(timing["request_ms"])
                        measured_requests.append({"index": index, **validate(outputs, timing)})
                    seconds = audio_seconds(outputs, case, manifests["candidate"])
                    validated_requests = [first_validation, *warmup_requests, *measured_requests]
                    own_difference = summarize_comparison(validated_requests, "against_own_fp32")
                    oracle_difference = summarize_comparison(validated_requests, "against_original_ort")
                    row = {"id": case["id"], "first_request": first,
                           "first_output_sha256": first_validation["output_sha256"],
                           "first_validation": first_validation,
                           "warmup_requests": warmup_requests,
                           "own_fp32_output_sha256": output_sha256(fp32_outputs["audio"]),
                           "measured_requests": measured_requests,
                           "identical_measured_output_hashes": len({item["output_sha256"] for item in measured_requests}) == 1,
                           "maximum_measured_errors": {key: maximum_errors(measured_requests, key)
                               for key in ("against_own_fp32", "against_original_ort")},
                           "maximum_request_errors": {key: maximum_errors(validated_requests, key)
                               for key in ("against_own_fp32", "against_original_ort")},
                           "measurements": {"request_ms": summarize(samples)},
                           "audio_seconds": seconds, "audio_samples": outputs["audio"].shape[-1],
                           "median_real_time_factor": summarize(samples)["median"] / (seconds * 1000),
                           "against_own_fp32": own_difference, "against_original_ort": oracle_difference,
                           "own_fp32_against_original_ort": waveform_metrics(fp32_outputs["audio"], reference, criterion),
                           "status": "pass" if all(item["against_own_fp32"]["numerical_parity"] and
                                                       item["against_original_ort"]["numerical_parity"]
                                                       for item in validated_requests)
                                      else "numerical_difference"}
                    report["cases"].append(row)
                    np.savez(args.out.with_name(args.out.stem + f".{case['id']}.outputs.npz"),
                             candidate=outputs["audio"], own_fp32=fp32_outputs["audio"])
                report["status"] = "pass" if all(row["status"] == "pass" for row in report["cases"]) else "numerical_difference"
            finally:
                errors = []
                for name, runner in runners.items():
                    report["routes"][name] = runner.route
                    try:
                        runner.close()
                    except Exception as error:
                        errors.append(str(error))
                if errors:
                    raise RuntimeError("runner cleanup failed: " + "; ".join(errors))
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["memory"]["peak_rss_bytes"] = peak_rss_bytes()
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "pass" else 3 if report["status"] == "numerical_difference" else 1


if __name__ == "__main__":
    raise SystemExit(main())
