#!/usr/bin/env python3
"""Measure receipt-digit-reader inference latency on every available route.

Timing contract
---------------
A measured sample covers shape binding, dispatch, synchronization, and the
owned output snapshot. Model load, graph compilation, context or session
creation, image decoding, preprocessing, slot decoding, and process startup are
all excluded, so these numbers are inference latency rather than process wall
time. Each route holds one session open across its samples.

Every route must reproduce the same record. A route whose record changes
between runs, or disagrees with the FP32 reference, is reported as a failure
rather than as a fast result: timing a wrong answer measures nothing.

Reported statistic is the **median**, because a cold page fault or a scheduler
migration skews a mean far more than it moves a median, and neither is a
property of the kernel being compared.

    python3 -m examples.receipt_digit_reader.tools.benchmark_backends \\
        --package build/receipt-digit-reader-fp32 \\
        --image receipt.jpg --onnx model.onnx \\
        --native-binary build/cmake/native/receipt_digit_reader \\
        --api dist/0.6.0/volvoxai.js --repeat 30
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_artifact_hashes(package: Path) -> dict[str, str]:
    """Bind a report to every fixed receipt package payload without paths."""

    return {
        filename: _sha256(package / filename)
        for filename in ("manifest.json", "graph.json", "model.safetensors")
    }


def _requested_route_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return every requested route that did not publish a measurement row."""

    return [
        f"{row.get('route', 'unknown route')}: {row['error']}"
        for row in rows
        if isinstance(row.get("error"), str) and row["error"]
    ]


def _public_route_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove decoded receipt values and route stderr from publication output."""

    reference = next(
        ((row["phone"], row["street"]) for row in rows
         if isinstance(row.get("phone"), str)
         and isinstance(row.get("street"), str)),
        None,
    )
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        public = {
            key: value
            for key, value in row.items()
            if key not in {"phone", "street", "error"}
        }
        record = (
            (row["phone"], row["street"])
            if isinstance(row.get("phone"), str)
            and isinstance(row.get("street"), str)
            else None
        )
        if record is None:
            public["status"] = "error"
        else:
            public["status"] = "measured"
            public["decoded_record_matches_reference"] = record == reference
        public_rows.append(public)
    return public_rows


def _pinned(command: Sequence[str], cpu: int | None) -> list[str]:
    """Pin to one logical CPU when the platform offers it.

    Without pinning, a scheduler migration mid-run shows up as a latency
    outlier that has nothing to do with the kernel under test.
    """

    if cpu is None or not shutil.which("taskset"):
        return list(command)
    return ["taskset", "-c", str(cpu), *command]


@contextmanager
def _pinned_process(cpu: int | None):
    """Temporarily pin the benchmark parent so in-process ORT cannot migrate."""

    if cpu is None:
        yield
        return
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("--pin-cpu requires process-affinity support on this host")
    original = set(os.sched_getaffinity(0))
    if cpu not in original:
        raise RuntimeError(
            f"--pin-cpu {cpu} is outside the process affinity {sorted(original)}"
        )
    os.sched_setaffinity(0, {cpu})
    try:
        yield
    finally:
        os.sched_setaffinity(0, original)


def check_host_is_quiet(limit: float, window: float = 0.4) -> tuple[bool, float]:
    """Refuse to publish a latency number measured on a contended host.

    Pinning with `taskset` does not help here: an unrelated job spread across
    every core contends for the pinned one too. This has already produced a
    1.8x swing in the ONNX Runtime reference between runs, which is larger than
    any difference this document exists to measure, so the check is a hard gate
    rather than a note.

    The measurement is a short live sample of non-idle CPU rather than the load
    average. A load average is a decaying one-minute mean that this harness's
    own run inflates, so gating on it both lags real contention and blocks the
    second variant of a sequential sweep against the first one's own load. The
    harness is idle during this window, so what it sees is foreign work.
    """

    def snapshot() -> tuple[int, int] | None:
        try:
            with open("/proc/stat", encoding="ascii") as handle:
                fields = [int(value) for value in handle.readline().split()[1:]]
        except (OSError, ValueError, IndexError):
            return None
        # user nice system idle iowait irq softirq steal ...
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return sum(fields), idle

    # Median of several short windows: one window is easily skewed by a
    # transient (an editor redraw, the agent driving this run), and rejecting a
    # quiet host on a spike is as unhelpful as accepting a busy one.
    samples = []
    previous = snapshot()
    for _ in range(3):
        if previous is None:
            return True, float("nan")
        time.sleep(window)
        current = snapshot()
        if current is None or current[0] <= previous[0]:
            return True, float("nan")
        busy_fraction = 1.0 - (current[1] - previous[1]) / (current[0] - previous[0])
        samples.append(busy_fraction * (os.cpu_count() or 1))
        previous = current
    busy_cores = statistics.median(samples)
    return busy_cores <= limit, busy_cores


def _cpu_governor() -> str:
    """Record the scaling governor: `powersave` lets clock drift between runs."""

    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                  encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return "unknown"


def _single_thread_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # Nested math libraries otherwise pick their own thread counts, which makes
    # a one-core comparison silently multi-core for some routes only.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def preprocess(image_path: Path, width: int, height: int) -> np.ndarray:
    from PIL import Image

    image = Image.open(image_path).convert("L").resize(
        (width, height), Image.Resampling.BILINEAR,
    )
    plane = np.asarray(image, dtype=np.float32) / 255.0
    return ((plane - 0.5) / 0.5)[None, None, :, :].astype(np.float32)


def decode(logits: np.ndarray, layout: Mapping[str, int]) -> tuple[str, str]:
    blank = int(layout["blank_class"])
    phone_slots = int(layout["phone_slots"])
    argmax = logits.reshape(int(layout["slots"]), int(layout["num_classes"])).argmax(-1)

    def read(start: int, stop: int) -> str:
        digits = []
        for index in range(start, stop):
            if int(argmax[index]) == blank:
                break
            digits.append(str(int(argmax[index])))
        return "".join(digits)

    return read(0, phone_slots), read(phone_slots, int(layout["slots"]))


def measure_onnxruntime(model: Path, batch: np.ndarray, layout: Mapping[str, int],
                        repeat: int, warmup: int, threads: int) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {name: batch})
    samples = []
    record = None
    for _ in range(repeat):
        started = time.perf_counter_ns()
        outputs = session.run(None, {name: batch})
        samples.append((time.perf_counter_ns() - started) / 1e6)
        current = decode(outputs[0], layout)
        if record is not None and current != record:
            raise SystemExit("ONNX Runtime record changed between runs")
        record = current
    return {
        "route": "onnxruntime",
        "backend": "CPUExecutionProvider",
        "phone": record[0],
        "street": record[1],
        "runs": repeat,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def measure_native(binary: Path, package: Path, image: Path, backend: str,
                   repeat: int, warmup: int, cpu: int | None) -> dict:
    command = _pinned([
        str(binary), "--package", str(package), "--image", str(image),
        "--backend", backend, "--repeat", str(repeat), "--warmup", str(warmup),
    ], cpu)
    completed = subprocess.run(command, capture_output=True, text=True,
                               env=_single_thread_environment(), check=False)
    if completed.returncode != 0:
        return {"route": f"native/{backend}", "backend": backend,
                "error": completed.stderr.strip() or "native benchmark failed"}
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload["route"] = f"native/{backend}"
    payload.pop("samples", None)
    return payload


def measure_node(package: Path, raw: Path, backend: str, repeat: int, warmup: int,
                 api: Path, wasm_url: Path | None, cpu: int | None) -> dict:
    script = REPOSITORY_ROOT / "examples/receipt_digit_reader/tools/run_backends.mjs"
    command = [
        "npx", "tsx", str(script), "--package", str(package), "--raw", str(raw),
        "--backend", backend, "--repeat", str(repeat), "--warmup", str(warmup),
        "--api", str(api), "--json", "--include-private-records",
    ]
    if wasm_url is not None:
        command += ["--wasm-url", str(wasm_url)]
    completed = subprocess.run(_pinned(command, cpu), capture_output=True, text=True,
                               cwd=REPOSITORY_ROOT, env=_single_thread_environment(),
                               check=False)
    if completed.returncode != 0:
        return {"route": f"js/{backend}", "backend": backend,
                "error": completed.stderr.strip()[-400:] or "node benchmark failed"}
    payload = json.loads(completed.stdout.strip().splitlines()[-1])["records"][0]
    payload["route"] = f"js/{backend}"
    payload.pop("samples", None)
    payload.pop("ms", None)
    return payload


def _measure_node_routes(package: Path, batch: np.ndarray,
                         backends: Sequence[str], repeat: int, warmup: int,
                         api: Path, wasm_url: Path | None,
                         cpu: int | None) -> list[dict]:
    """Stage the receipt tensor in a private, exception-safe temporary path."""

    # The normalized tensor can fingerprint the source receipt. Keep it out of
    # the package tree and let TemporaryDirectory remove it on success or on
    # any route exception before the failure is propagated.
    with tempfile.TemporaryDirectory(
        prefix="volvoxai-receipt-benchmark-",
    ) as temporary:
        raw_path = Path(temporary) / "input.f32"
        raw_path.write_bytes(batch.tobytes())
        return [
            measure_node(package, raw_path, backend, repeat, warmup,
                         api, wasm_url, cpu)
            for backend in backends
        ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--onnx", type=Path, help="producer ONNX for the reference route")
    parser.add_argument(
        "--native-binary", required=True, type=Path,
        help="receipt_digit_reader application built by the native CMake target",
    )
    parser.add_argument(
        "--api", required=True, type=Path,
        help="exact JavaScript API artifact used by the JS/WASM routes",
    )
    parser.add_argument("--native-backend", action="append", default=None,
                        help="repeat per native backend identity (default: cpu)")
    parser.add_argument("--js-backend", action="append", default=None,
                        help="repeat per JS backend (default: wasm)")
    parser.add_argument("--wasm-url", type=Path,
                        default=REPOSITORY_ROOT / "dist/0.6.0/volvoxai.wasm")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pin-cpu", type=int, default=0,
                        help="logical CPU to pin every route to; -1 disables")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-load", type=float, default=1.0,
                        help="refuse to measure when foreign work occupies more "
                             "than this many cores, sampled live. The default "
                             "tolerates the shell or agent driving the run "
                             "while still catching a competing job; use "
                             "--allow-busy-host to override")
    parser.add_argument("--allow-busy-host", action="store_true",
                        help="measure anyway and mark the report as contended")
    args = parser.parse_args(argv)

    if not args.native_binary.is_file() or not os.access(args.native_binary, os.X_OK):
        parser.error(
            "--native-binary must be an executable receipt_digit_reader application"
        )
    if not args.api.is_file():
        parser.error("--api must name an existing JavaScript API artifact")

    quiet, load = check_host_is_quiet(args.max_load)
    if not quiet and not args.allow_busy_host:
        print(
            f"foreign work is using {load:.2f} cores, above --max-load "
            f"{args.max_load:.2f}. Another process is competing for the pinned "
            "core, and a latency number measured now is not reproducible. "
            "Wait for the host to settle, or pass --allow-busy-host to record "
            "a contended measurement.",
            file=sys.stderr,
        )
        return 2

    manifest = _read_json(args.package / "manifest.json")
    layout = manifest["decode"]
    width = int(manifest["preprocess"]["width"])
    height = int(manifest["preprocess"]["height"])
    cpu = None if args.pin_cpu < 0 else args.pin_cpu

    batch = preprocess(args.image, width, height)

    rows: list[dict] = []
    # Native and JavaScript children are explicitly taskset-pinned below. Pin
    # this parent for the same interval so the in-process ORT route observes the
    # identical one-logical-CPU contract, then restore the caller's affinity.
    with _pinned_process(cpu):
        if args.onnx is not None:
            rows.append(measure_onnxruntime(args.onnx, batch, layout, args.repeat,
                                            args.warmup, args.threads))
        for backend in (args.native_backend or ["cpu"]):
            rows.append(measure_native(args.native_binary, args.package, args.image,
                                       backend, args.repeat, args.warmup, cpu))
        rows.extend(_measure_node_routes(
            args.package,
            batch,
            args.js_backend or ["wasm"],
            args.repeat,
            args.warmup,
            args.api,
            args.wasm_url,
            cpu,
        ))

    reference = next((row for row in rows if row.get("route") == "onnxruntime"), None)
    records = {(row["phone"], row["street"]) for row in rows if "phone" in row}
    route_errors = _requested_route_errors(rows)
    for row in rows:
        # Reported as "how many times longer than ONNX Runtime", because a
        # speedup ratio below 1.00 reads as a small number for a large
        # regression and is easy to misreport as a percentage.
        if reference and "median_ms" in row and reference.get("median_ms"):
            row["times_onnxruntime_latency"] = row["median_ms"] / reference["median_ms"]

    report = {
        "package_manifest_sha256": hashlib.sha256(
            (args.package / "manifest.json").read_bytes()
        ).hexdigest(),
        "package_artifacts": _package_artifact_hashes(args.package),
        "artifacts": {
            "native_binary_sha256": _sha256(args.native_binary),
            "api_sha256": _sha256(args.api),
            **(
                {"wasm_sha256": _sha256(args.wasm_url)}
                if args.wasm_url is not None and args.wasm_url.is_file()
                else {}
            ),
            **(
                {"onnx_sha256": _sha256(args.onnx)}
                if args.onnx is not None and args.onnx.is_file()
                else {}
            ),
        },
        "variant": manifest.get("variant"),
        "package_class": manifest.get("package_class"),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "threads": args.threads,
        "pinned_cpu": cpu,
        "host": {
            "platform": platform.platform(),
            "cpu_governor": _cpu_governor(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "records_agree": len(records) <= 1,
        "requested_routes_succeeded": not route_errors,
        "host_foreign_busy_cores": load,
        "host_contended": not quiet,
        "privacy": {
            "input_paths_included": False,
            "decoded_records_included": False,
            "route_stderr_included": False,
        },
        "rows": _public_route_rows(rows),
    }
    text = json.dumps(report, indent=1)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["records_agree"]:
        print("routes disagree on the decoded record", file=sys.stderr)
        return 1
    if route_errors:
        print("requested benchmark route failed:", file=sys.stderr)
        for error in route_errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
