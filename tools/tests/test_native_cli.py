#!/usr/bin/env python3
"""Focused integration checks for the fixed model-agnostic native CLI."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def run(binary: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(binary), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require(condition: bool, message: str, result: subprocess.CompletedProcess[bytes] | None = None) -> None:
    if condition:
        return
    if result is not None:
        message += (
            f"\nexit={result.returncode}"
            f"\nstdout={result.stdout.decode(errors='replace')}"
            f"\nstderr={result.stderr.decode(errors='replace')}"
        )
    raise AssertionError(message)


def write_safetensors(path: Path, values: tuple[float, ...]) -> None:
    data = struct.pack(f"={len(values)}f", *values)
    header = json.dumps(
        {
            "w": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, len(data)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: {sys.argv[0]} <native-volvoxai-binary> [native-volvoxai-full-binary]",
            file=sys.stderr,
        )
        return 2
    binary = Path(sys.argv[1]).resolve()
    require(binary.is_file(), f"Native CLI not found: {binary}")
    full_binary = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else None
    if full_binary is not None:
        require(full_binary.is_file(), f"Native full CLI not found: {full_binary}")

    with tempfile.TemporaryDirectory(prefix="volvoxai-cli-") as temporary:
        root = Path(temporary)
        graph = root / "graph.json"
        x_path = root / "x.f32"
        ids_path = root / "ids.i32"
        ids_wrong_suffix = root / "ids.f32"
        ids_short = root / "ids-short.i32"
        output = root / "y.f32"
        ids_output = root / "ids-out.i32"
        lifecycle_report = root / "runtime-evidence.json"
        unresolved = root / "unresolved.json"

        graph.write_text(
            json.dumps(
                {
                    "format": "volvox-graph/v1",
                    "inputs": {
                        "x": {"shape": [2], "dtype": "float32"},
                        "ids": {"shape": [2], "dtype": "int32"},
                    },
                    "nodes": [
                        {
                            "opType": "Sin",
                            "inputs": {"input": "x"},
                            "outputs": {"out": "y"},
                            "outputs_shape": {"out": [2]},
                        }
                    ],
                    "outputs": ["y", "ids"],
                }
            ),
            encoding="utf-8",
        )
        x_path.write_bytes(struct.pack("=2f", 0.0, 1.0))
        ids_path.write_bytes(struct.pack("=2i", 7, 11))
        ids_wrong_suffix.write_bytes(ids_path.read_bytes())
        ids_short.write_bytes(struct.pack("=i", 7))
        unresolved.write_text(
            json.dumps(
                {
                    "format": "volvox-graph/v1",
                    "inputs": {"x": {"shape": [1, 2], "dtype": "float32"}},
                    "nodes": [
                        {
                            "opType": "MatMul",
                            "inputs": {"input": "x", "weight": "missing.weight"},
                            "outputs": {"out": "logits"},
                            "outputs_shape": {"out": [1, 1]},
                        }
                    ],
                    "outputs": ["logits"],
                }
            ),
            encoding="utf-8",
        )

        result = run(
            binary,
            "run",
            str(graph),
            "--input",
            f"x={x_path}",
            "--input",
            f"ids={ids_path}",
            "--output",
            f"y={output}",
            "--output",
            f"ids={ids_output}",
            "--cpu",
            "--report-json",
            str(lifecycle_report),
        )
        require(result.returncode == 0, "Weightless F32/I32 run failed", result)
        values = struct.unpack("=2f", output.read_bytes())
        require(abs(values[0]) < 1.0e-6 and abs(values[1] - math.sin(1.0)) < 1.0e-6,
                f"Unexpected output: {values}")
        require(struct.unpack("=2i", ids_output.read_bytes()) == (7, 11),
                "I32 output was not written as exact raw data")
        evidence = json.loads(lifecycle_report.read_text(encoding="utf-8"))
        compilation = evidence.get("compilation", {})
        execution = evidence.get("execution", {})
        stable = evidence.get("stableResult", {})
        require(evidence.get("schema") == "volvoxai.runtime-evidence" and
                evidence.get("version") == 1,
                "Native runtime evidence has the wrong schema")
        require(compilation.get("policy") == {
                    "mode": "require", "backend": "cpu", "operatorFallback": "forbid"
                } and compilation.get("selectedBackend") == "cpu",
                "Native runtime evidence did not preserve strict CPU policy")
        for report_name, report in (("compilation", compilation), ("execution", execution)):
            route = report.get("route", {})
            require(route.get("tierFallback") is False and
                    route.get("operator") == {
                        "attestation": "reported", "used": False, "offendingNode": None
                    }, f"Native {report_name} route evidence is not exact")
            revisions = report.get("revisions", {})
            require(str(revisions.get("topologyRevision", "")).isdigit() and
                    str(revisions.get("weightRevision", "")).isdigit() and
                    str(revisions.get("weightRevisionId", "")).startswith("native-weight-") and
                    len(revisions.get("adapterRevisionIds", [])) == 1,
                    f"Native {report_name} revision evidence is incomplete")
        require(str(execution.get("executionId", "")).startswith("native-execution-") and
                str(execution.get("contextId", "")).startswith("native-context-") and
                execution.get("backend") == "cpu" and execution.get("outcome") == "success",
                "Native execution identity evidence is incomplete")
        require({item.get("name") for item in stable.get("outputs", [])} == {"y", "ids"} and
                stable.get("freshCallerOwnedReads") is True and
                stable.get("readableAfterContextClose") is True and
                stable.get("contextClosedBeforeResult") is True and
                stable.get("resultClosedAfterVerification") is True,
                "Native stable-result evidence is incomplete")

        result = run(binary, "run", str(unresolved))
        require(result.returncode != 0 and b"Native init failed" in result.stderr,
                "Weightless initialization accepted an unresolved weight", result)

        result = run(binary, "run", str(graph), "--input", f"x={x_path}",
                     "--input", f"ids={ids_wrong_suffix}")
        require(result.returncode != 0 and b"requires a .i32 file" in result.stderr,
                "Input dtype/suffix mismatch was not rejected", result)

        result = run(binary, "run", str(graph), "--input", f"x={x_path}",
                     "--input", f"ids={ids_short}")
        require(result.returncode != 0 and b"expects 8 raw bytes" in result.stderr,
                "Short raw input was not rejected", result)

        wrong_output = root / "ids-out.f32"
        result = run(binary, "run", str(graph), "--input", f"x={x_path}",
                     "--input", f"ids={ids_path}", "--output", f"ids={wrong_output}")
        require(result.returncode != 0 and b"requires a .i32 file" in result.stderr,
                "Output dtype/suffix mismatch was not rejected", result)

        for invalid_row in ("not-a-row", "2147483648", "-2"):
            result = run(binary, "run", str(graph), "--row", invalid_row)
            require(result.returncode == 2 and b"--row must be" in result.stderr,
                    f"Invalid row was not rejected: {invalid_row}", result)

        dev_full = Path("/dev/full")
        if dev_full.exists():
            full_output = root / "full.f32"
            full_output.symlink_to(dev_full)
            result = run(binary, "run", str(graph), "--input", f"x={x_path}",
                         "--input", f"ids={ids_path}", "--output", f"y={full_output}")
            require(result.returncode != 0 and b"Cannot write output file" in result.stderr,
                    "Output write failure was not reported", result)

        if full_binary is not None:
            train_root = root / "train-model"
            train_root.mkdir()
            train_graph = train_root / "graph.json"
            train_weights = train_root / "model.safetensors"
            train_input = train_root / "x.f32"
            train_targets = train_root / "targets.i32"
            trained_weights = train_root / "trained.safetensors"
            train_graph.write_text(
                json.dumps(
                    {
                        "format": "volvox-graph/v1",
                        "inputs": {"x": {"shape": [1, 2], "dtype": "float32"}},
                        "nodes": [
                            {
                                "opType": "MatMul",
                                "inputs": {"input": "x", "weight": "w"},
                                "outputs": {"output": "logits"},
                                "outputs_shape": {"output": [1, 2]},
                            }
                        ],
                        "outputs": ["logits"],
                    }
                ),
                encoding="utf-8",
            )
            write_safetensors(train_weights, (0.25, -0.4, 0.15, 0.3))
            train_input.write_bytes(struct.pack("=2f", 1.0, -0.5))
            train_targets.write_bytes(struct.pack("=i", 0))

            inference_help = run(binary, "--help")
            require(b"\n  train " not in inference_help.stdout,
                    "Inference help exposes the full train command", inference_help)
            result = run(binary, "train", str(train_root))
            require(result.returncode == 2,
                    "Inference binary accepted the full train command", result)
            full_help = run(full_binary, "--help")
            require(b"\n  train " in full_help.stdout,
                    "Full help omits the train command", full_help)
            result = run(
                full_binary,
                "train",
                str(train_root),
                "--input",
                f"x={train_input}",
                "--targets",
                str(train_targets),
                "--logits",
                "logits",
                "--trainable",
                "w",
                "--optimizer",
                "adamw",
                "--output-weights",
                str(trained_weights),
            )
            require(result.returncode == 0,
                    "Full opaque-API train command failed", result)
            require(trained_weights.is_file() and
                    trained_weights.read_bytes() != train_weights.read_bytes(),
                    "Full train command did not export changed weights", result)

    print("Native CLI inference and full Trainer boundary tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
