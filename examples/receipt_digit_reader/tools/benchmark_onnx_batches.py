#!/usr/bin/env python3
"""Compare producer ONNX B=N execution with N independent B=1 calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .benchmark_backends import check_host_is_quiet


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _decode(logits: np.ndarray, layout: dict) -> str:
    classes = logits.argmax(-1)
    blank = int(layout["blank_class"])
    phone_slots = int(layout["phone_slots"])

    def read(values) -> str:
        digits = []
        for value in values:
            if int(value) == blank:
                break
            digits.append(str(int(value)))
        return "".join(digits)

    return f"{read(classes[:phone_slots])}/{read(classes[phone_slots:])}"


def _summary(samples: list[float], logical_requests: int) -> dict:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "logical_requests_per_second": (
            logical_requests * len(samples) * 1000 / sum(samples)
        ),
    }


def benchmark(
    model: Path,
    manifest: dict,
    raws: Sequence[Path],
    batch_sizes: Sequence[int],
    *,
    repeat: int,
    warmup: int,
    threads: int,
) -> list[dict]:
    import onnxruntime as ort

    input_shape = list(manifest["abi"]["input"]["shape"])
    elements = int(np.prod(input_shape))
    lanes = []
    for path in raws:
        value = np.fromfile(path, dtype=np.float32)
        if value.size != elements:
            raise ValueError(
                f"{path} has {value.size} F32 values, expected {elements}"
            )
        lanes.append(value.reshape(input_shape))

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    layout = manifest["decode"]
    rows = []
    for size in batch_sizes:
        selected = lanes[:size]
        batch = np.concatenate(selected, axis=0)

        def execute_batched() -> np.ndarray:
            return session.run(None, {input_name: batch})[0]

        def execute_independent() -> np.ndarray:
            return np.concatenate([
                session.run(None, {input_name: lane})[0] for lane in selected
            ], axis=0)

        for route, execute, physical_per_group in (
            ("independent-b1", execute_independent, size),
            (f"physical-b{size}", execute_batched, 1),
        ):
            for _ in range(warmup):
                execute()
            samples = []
            hashes = set()
            representative = None
            for _ in range(repeat):
                started = time.perf_counter_ns()
                output = execute()
                samples.append((time.perf_counter_ns() - started) / 1e6)
                hashes.add(_digest(output))
                representative = output
            if len(hashes) != 1 or representative is None:
                raise RuntimeError(f"{route} B={size} output changed between runs")
            rows.append({
                "route": route,
                "logical_batch": size,
                "physical_invocations_per_group": physical_per_group,
                "output_shape": list(representative.shape),
                "records": [_decode(row, layout) for row in representative],
                "lane_output_sha256": [_digest(row) for row in representative],
                **_summary(samples, size),
                "_output": representative,
            })

        independent, physical = rows[-2:]
        physical["max_abs_error_vs_independent_b1"] = float(np.max(np.abs(
            physical["_output"] - independent["_output"]
        )))
        for row in (independent, physical):
            row.pop("_output")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw", action="append", required=True, type=Path)
    parser.add_argument("--batch-size", action="append", type=int, default=[])
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-load", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    batch_sizes = args.batch_size or [1, 2, 4]
    if (
        args.repeat < 1 or args.warmup < 0 or args.threads < 1
        or any(size < 1 or size > len(args.raw) for size in batch_sizes)
    ):
        parser.error("repeat/threads must be positive and batch sizes must fit --raw")
    quiet, busy = check_host_is_quiet(args.max_load)
    if not quiet:
        parser.error(
            f"foreign work occupies {busy:.2f} cores, above --max-load {args.max_load:.2f}"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = benchmark(
        args.onnx, manifest, args.raw, batch_sizes,
        repeat=args.repeat, warmup=args.warmup, threads=args.threads,
    )
    report = {
        "schema": "volvoxai.receipt-digit-onnx-batches/v1",
        "model": str(args.onnx),
        "raws": [str(path) for path in args.raw],
        "batch_sizes": list(batch_sizes),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "threads": args.threads,
        "host_foreign_busy_cores": busy,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "rows": rows,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
