#!/usr/bin/env python3
"""Device verification runner for PackedLinear, PackedLSTM, and QConv2D.

Runs on the remote GPU host with Python (numpy).
Executes ./volvoxai run with --cuda and --vulkan for each case.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def verify_case(case_name: str, case_dir: Path, volvox_bin: Path, backend: str) -> dict:
    out_f32 = case_dir / f"y_{backend}.f32"
    rep_json = case_dir / f"report_{backend}.json"
    ref_f32 = case_dir / "y_ref.f32"

    if out_f32.exists():
        out_f32.unlink()
    if rep_json.exists():
        rep_json.unlink()

    cmd = [
        str(volvox_bin), "run", str(case_dir),
        f"--{backend}",
        "--input", f"x={case_dir}/x.f32",
        "--output", f"y={out_f32}",
        "--report-json", f"{rep_json}",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "case": case_name,
            "backend": backend,
            "status": "FAILED",
            "error": proc.stderr or proc.stdout,
            "exit_code": proc.returncode,
        }

    if not rep_json.exists():
        return {
            "case": case_name,
            "backend": backend,
            "status": "FAILED",
            "error": "report JSON not generated",
        }

    rep = json.loads(rep_json.read_text())
    comp = rep.get("compile", {})
    route = comp.get("route", {})
    exec_rep = rep.get("execute", {})

    # Validate route attestation
    compile_ok = comp.get("status") == "NATIVE_STATUS_OK" and comp.get("backend") == backend
    route_attested = route.get("attested") is True and route.get("provider") == backend
    no_fallback = route.get("fallbackNodes") == 0 and route.get("missingNodes") == 0
    execute_ok = exec_rep.get("status") == "NATIVE_STATUS_OK" and exec_rep.get("backend") == backend

    if not (compile_ok and route_attested and no_fallback and execute_ok):
        return {
            "case": case_name,
            "backend": backend,
            "status": "ROUTE_FAILURE",
            "compile_ok": compile_ok,
            "route_attested": route_attested,
            "no_fallback": no_fallback,
            "execute_ok": execute_ok,
            "report": rep,
        }

    # Verify numerical correctness against reference
    ref_bytes = ref_f32.read_bytes()
    out_bytes = out_f32.read_bytes()
    if len(ref_bytes) != len(out_bytes):
        return {
            "case": case_name,
            "backend": backend,
            "status": "LENGTH_MISMATCH",
            "ref_len": len(ref_bytes),
            "out_len": len(out_bytes),
        }

    ref = np.frombuffer(ref_bytes, dtype=np.float32)
    actual = np.frombuffer(out_bytes, dtype=np.float32)

    diff = np.abs(actual - ref)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))

    # SNR calculation
    noise_power = np.mean((actual - ref) ** 2)
    sig_power = np.mean(ref ** 2)
    snr = float(10 * np.log10(sig_power / noise_power)) if noise_power > 0 else float("inf")

    # Cosine similarity / correlation
    norm_a = np.linalg.norm(actual)
    norm_r = np.linalg.norm(ref)
    cosine = float(np.dot(actual, ref) / (norm_a * norm_r)) if norm_a > 0 and norm_r > 0 else 0.0

    passed = max_err < 1e-4 or snr > 50.0

    return {
        "case": case_name,
        "backend": backend,
        "status": "PASSED" if passed else "ACCURACY_FAIL",
        "max_err": max_err,
        "mean_err": mean_err,
        "snr_db": snr,
        "cosine": cosine,
        "active_nodes": route.get("activeNodes"),
        "selected_nodes": route.get("selectedNodes"),
        "device": comp.get("device"),
    }


def main():
    root = Path(__file__).resolve().parent
    volvox_bin = root / "volvoxai"
    cases_dir = root / "cases"

    cases = ["packed_linear", "packed_lstm", "qconv2d"]
    backends = ["cuda", "vulkan"]

    results = []
    for case in cases:
        c_dir = cases_dir / case
        if not c_dir.is_dir():
            continue
        for backend in backends:
            res = verify_case(case, c_dir, volvox_bin, backend)
            results.append(res)
            print(f"[{res['status']}] Case: {case:16s} | Backend: {backend:6s} | MaxErr: {res.get('max_err', 'N/A')} | SNR: {res.get('snr_db', 'N/A')} dB")

    summary = {
        "format": "volvox-packed-gpu-verification-v1",
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASSED"),
        "results": results,
    }

    out_file = root / "verification_results.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nFinal Summary: {summary['passed']}/{summary['total']} tests PASSED.")
    if summary["passed"] != summary["total"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
