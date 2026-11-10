#!/usr/bin/env python3
"""Run the prepared TTS matrix sequentially inside a task-owned checkout.

The calling transfer script owns the checkout and cleans it after collecting
reports. This runner downloads no assets and writes only under --out.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deno", default="deno")
    parser.add_argument("--mode", choices=("full", "all", "wasm", "components"), default="full")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    version = json.loads(Path("package.json").read_text())["version"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    summary = {"format": "volvoxai-tts-matrix/v1", "gpu_before": gpu, "cpu": cpu,
               "os": platform.platform(), "threads": args.threads, "warmups": args.warmups,
               "repeats": args.repeats, "trials": []}

    nsys_path = shutil.which(os.environ.get("NSYS", "nsys"))
    if not nsys_path:
        nsys_path = shutil.which("/opt/nvidia/nsight-systems/bin/nsys")

    def run(name, command):
        output = args.out / (name + ".json")
        environment = dict(os.environ)
        if name in {"kokoro-generator-stage2-vulkan", "kokoro-full-vulkan"}:
            environment["VOLVOX_VULKAN_MB"] = "4096" if name == "kokoro-full-vulkan" else "2048"
        with (args.out / (name + ".log")).open("w") as log:
            result = subprocess.run([*command, "--out", str(output)], stdout=log, stderr=subprocess.STDOUT, timeout=600, env=environment)
        for extra in args.out.glob("*.nsys-rep"):
            extra.unlink(missing_ok=True)
        for extra in args.out.glob("*.sqlite"):
            extra.unlink(missing_ok=True)
        for extra in Path.cwd().glob("*.nsys-rep"):
            extra.unlink(missing_ok=True)
        for extra in Path.cwd().glob("*.sqlite"):
            extra.unlink(missing_ok=True)
        summary["trials"].append({"name": name, "exit_code": result.returncode, "report": output.name})
        (args.out / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(name, "exit", result.returncode, flush=True)

    if args.mode == "full":
        for family, directory in (("piper", "piper-full"), ("kokoro", "kokoro-complete")):
            manifest = str(Path("build/tts-examples") / directory / "manifest.json")
            native = [sys.executable, "examples/tts_common/benchmark_native.py", "--manifest", manifest,
                "--threads", str(args.threads), "--warmups", str(args.warmups), "--repeats", str(args.repeats),
                *(["--report-waveform-differences"] if family == "kokoro" else [])]
            backends = [args.backend] if args.backend else ["cpu", "cuda", "vulkan", "opengl"]
            for backend in backends:
                if backend in ("cpu", "cuda", "vulkan", "opengl"):
                    cmd = [*native, "--library", f"native/libvolvoxai.so.{version}", "--backend", backend]
                    if backend == "cuda" and nsys_path:
                        cmd = [str(nsys_path), "profile", "--trace=cuda", "--stats=true", "--force-overwrite=true",
                               "-o", str(args.out / f"{family}-{backend}-profile"), *cmd]
                    run(f"{family}-full-{backend}", cmd)
            if not args.backend or args.backend == "cpu-lite":
                run(f"{family}-full-cpu-lite", [*native, "--library", f"native/libvolvoxai-lite.so.{version}"])
            if not args.backend or args.backend in ("wasm", "webgpu"):
                for backend, profile in (("wasm", "inference"), ("wasm", "full"), ("webgpu", "full")):
                    if args.backend and backend != args.backend:
                        continue
                    executable = [args.deno, "run", "--no-config", *(["--unstable-webgpu"] if backend == "webgpu" else []), "-A"]
                    run(f"{family}-full-{backend}-{profile}", [*executable, "examples/tts_common/benchmark_web.mjs",
                        "--manifest", manifest, "--backend", backend, "--profile", profile,
                        "--warmup", str(args.warmups), "--repeats", str(args.repeats),
                        *(["--report-waveform-differences"] if family == "kokoro" else [])])
        codes = [row["exit_code"] for row in summary["trials"]]
        code = 1 if any(value not in (0, 3) for value in codes) else 3 if 3 in codes else 0
        summary["status"] = "failed" if code == 1 else "numerical_difference" if code == 3 else "passed"
        (args.out / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
        return code

    for family, component in (("piper", "vocoder"), ("kokoro", "spectral-head"), ("kokoro", "generator-stage2")):
        root = Path("build/tts-examples") / family
        native = [sys.executable, "examples/tts_common/benchmark_native.py", "--threads", str(args.threads),
                  "--warmups", str(args.warmups), "--repeats", str(args.repeats)]
        library = ["--library", f"native/libvolvoxai.so.{version}"]
        if args.mode != "wasm":
            if args.mode == "all" and component != "generator-stage2":
                run(f"{family}-whole-cpu", [*native, "--manifest", str(root / "manifest.json"), *library])
            for backend in ("cpu", "cuda", "vulkan", "opengl"):
                run(f"{family}-{component}-{backend}", [*native, "--manifest", str(root / component / "manifest.json"),
                    *library, "--backend", backend])
            run(f"{family}-{component}-cpu-lite", [*native, "--manifest", str(root / component / "manifest.json"),
                "--library", f"native/libvolvoxai-lite.so.{version}"])
        for backend, profile in (("wasm", "inference"), ("wasm", "full"), ("webgpu", "full")):
            if args.mode == "wasm" and backend != "wasm":
                continue
            executable = [args.deno, "run", "--no-config", *( ["--unstable-webgpu"] if backend == "webgpu" else []), "-A"]
            run(f"{family}-{component}-{backend}-{profile}", [*executable, "examples/tts_common/benchmark_web.mjs",
                "--manifest", str(root / component / "manifest.json"), "--backend", backend, "--profile", profile,
                "--warmup", str(args.warmups), "--repeats", str(args.repeats)])
    return int(any(row["exit_code"] != 0 and not (row["exit_code"] == 2 and "-whole-" in row["name"])
                   for row in summary["trials"]))


if __name__ == "__main__":
    raise SystemExit(main())
